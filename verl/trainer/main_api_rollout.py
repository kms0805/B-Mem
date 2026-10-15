# Copyright 2026 B-Mem authors
"""Policy-by-API rollout without verl workers (2026-09-24).

Runs exactly the validation path of RayPPOTrainer._validate (same config, envs, memory service, TrajectoryCollector,
mem_adaptor, EpisodeRewardManager, and trajectory dump) but replaces the FSDP/vLLM ActorRollout WorkerGroup with an
in-process OpenAIApiRollout. No policy model is loaded; only the tokenizer (Qwen2.5 chat template) is needed so the
student-facing prompts are rendered identically to training. GPU use: none (the BGE-M3 embedding server runs separately).

Usage: same hydra overrides as main_ppo with actor_rollout_ref.rollout.name=openai_api and trainer.val_only=True:
    python3 -m verl.trainer.main_api_rollout <overrides...>
"""
import os
import json

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class ApiRolloutWG:
    """Duck-typed stand-in for the ActorRollout RayWorkerGroup: only generate_sequences is used by validation."""
    world_size = 1

    def __init__(self, rollout_cfg, tokenizer):
        from verl.workers.rollout.openai_api_rollout import OpenAIApiRollout
        self.tokenizer = tokenizer
        self.rollout = OpenAIApiRollout(rollout_cfg, tokenizer)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # same meta the FSDP worker adds before calling rollout.generate_sequences
        prompts.meta_info.update({"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id})
        return self.rollout.generate_sequences(prompts=prompts)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    OmegaConf.resolve(config)
    assert config.actor_rollout_ref.rollout.name == "openai_api", "main_api_rollout needs rollout.name=openai_api"
    assert bool(OmegaConf.select(config, "mem_adaptor.use_actor_rollout_wg", default=True)), "adaptor must share the API policy"

    if not ray.is_initialized():   # env workers are CPU Ray actors
        ray_init = OmegaConf.to_container(config.get("ray_init", {}), resolve=True) or {}
        rt = OmegaConf.to_container(OmegaConf.merge(get_ppo_ray_runtime_env(), OmegaConf.create(ray_init.get("runtime_env", {}))), resolve=True)
        ray_init["runtime_env"] = rt
        ray.init(**{k: v for k, v in ray_init.items() if k in ("num_cpus", "runtime_env", "object_store_memory")})

    memory_server_handle = None
    if config.env.memory.enabled:
        from agent_system.memory.local_service import start_local_memory_server
        memory_server_handle = start_local_memory_server(memory_config=config.env.memory, task_name=config.env.env_name)
        if memory_server_handle is not None:
            with open_dict(config.env.memory):
                config.env.memory.backend = "http"
                config.env.memory.vdb_base_url = memory_server_handle.base_url
    try:
        from agent_system.environments import make_envs
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from agent_system.reward_manager import EpisodeRewardManager
        from agent_system.reward_manager.episode import DEFAULT_NUM_EXAMINE_VAL
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.utils import hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn

        _, val_envs = make_envs(config)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=config.data.get("trust_remote_code", False))
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, None)
        val_loader = StatefulDataLoader(dataset=val_dataset, batch_size=config.data.val_batch_size or len(val_dataset),
                                        num_workers=0, shuffle=False, drop_last=False, collate_fn=collate_fn)

        t = RayPPOTrainer.__new__(RayPPOTrainer)   # borrow _validate and its dump helpers; no workers are created
        t.config = config
        t.tokenizer = tokenizer
        t.processor = None
        t.val_dataset = val_dataset
        t.val_dataloader = val_loader
        t.val_envs = val_envs
        t.val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=DEFAULT_NUM_EXAMINE_VAL, normalize_by_length=False, config=config)
        t.traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=None, adaptor_tokenizer=None)
        t.actor_rollout_wg = ApiRolloutWG(config.actor_rollout_ref.rollout, tokenizer)
        t.adaptor_rollout_wg = None
        t.global_steps = 0
        metrics = t._validate()
        print("API_ROLLOUT_METRICS " + json.dumps({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) or hasattr(v, "item")}))
    finally:
        if memory_server_handle is not None and hasattr(memory_server_handle, "stop"):
            try: memory_server_handle.stop()
            except Exception: pass


if __name__ == "__main__":
    main()
