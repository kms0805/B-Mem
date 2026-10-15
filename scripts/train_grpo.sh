#!/bin/bash
# GRPO on ALFWorld with B-Mem, starting from the SFT model. The skill bank stays fixed; the same policy writes the
# belief, the query, the skill adaptation, and the action.
#   MODEL_PATH     SFT initialization (bf16 HF dir)
#   METHOD         bmem (default) | wo_adapt | wo_belief_cond
#   QUERY_PENALTY  0 (default) | 1: reward -1 when an episode's query count is outside [2, 5]; the first decision
#                  is prompted to query, and skill-adaptation responses are trained with the episode advantage
#   TOTAL_STEPS (80), N_GPUS (2), RUN, OUT_ROOT (./outputs)
# 16 tasks x group 8 per step, lr 1e-6, KL 0.01 (low-var), clip 0.2, rollout T=1.0, validation T=0.4 on the 140 ID games every 10 steps.
# The BGE-M3 server shares the last GPU. Convert a checkpoint for evaluation with
#   bash scripts/export_checkpoint.sh $CKPT/global_step_<N> outputs/models/<name>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
METHOD=${METHOD:-bmem}; set_method "$METHOD" || exit 1
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the SFT initialization}
TOTAL_STEPS=${TOTAL_STEPS:-80}; N_GPUS=${N_GPUS:-2}
if [ "${QUERY_PENALTY:-0}" = 1 ]; then
  export BMEM_QUERY_RANGE=2,5 BMEM_QUERY_PENALTY=1.0 BMEM_FIRST_QUERY_PROMPT=1 BMEM_TRAIN_SHARED_REWRITER=1
else
  unset BMEM_QUERY_RANGE BMEM_QUERY_PENALTY; export BMEM_FIRST_QUERY_PROMPT=0 BMEM_TRAIN_SHARED_REWRITER=0
fi
RUN=${RUN:-grpo_${METHOD}_qp${QUERY_PENALTY:-0}_$(date +%m%d-%H%M)}
EXP=$OUT_ROOT/train/$RUN; CKPT=$OUT_ROOT/checkpoints/$RUN; mkdir -p "$EXP" "$CKPT"
DATA=$OUT_ROOT/data/alfworld_id_sizes; prepare_data "$DATA" --infer_alfworld_sizes --alfworld_eval_split eval_in_distribution || exit 1

IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
EMBED_PID=""; trap '[ -n "$EMBED_PID" ] && kill $EMBED_PID 2>/dev/null' EXIT
start_embed_server "${GPUS[${#GPUS[@]}-1]}" 0.08 "$EXP/embed_server.log" || exit 1

echo "[bmem] GRPO method=$METHOD query_penalty=${QUERY_PENALTY:-0} model=$MODEL_PATH run=$RUN steps=$TOTAL_STEPS"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/text/train.parquet" data.val_files="$DATA/text/test.parquet" \
  data.train_batch_size=16 data.val_batch_size=140 \
  data.max_prompt_length=4096 data.max_response_length=512 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.trainable=True actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS:-16} \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=$((12288+512)) \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.55} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  '+actor_rollout_ref.rollout.stop=["</query>"]' '+actor_rollout_ref.rollout.include_stop_str_in_output=True' '+actor_rollout_ref.rollout.detokenize=True' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  mem_adaptor.enable=$RECON mem_adaptor.retrieval_top_k=$ADAPTOR_TOPK mem_adaptor.use_actor_rollout_wg=True mem_adaptor.train_memory_adaptor=false mem_adaptor.model.path="$MODEL_PATH" \
  env.env_name=alfworld/AlfredTWEnv env.alfworld.eval_dataset=eval_in_distribution env.alfworld.validate_on_train_split=False \
  env.seed=0 env.max_steps=50 env.memory.count_query_as_step=False \
  env.memory.enabled=True env.memory.store_dir="$EXP/memory_vdb" env.memory.write_back=False env.memory.mode=recreate \
  env.memory.experience_summarizer.mode=none env.memory.retrieval_mode=agentic env.memory.retrieve_key=$RETRIEVE_KEY \
  env.memory.experience_utility.enable=False \
  env.memory.memory_text_retrieval_dedupe_similarity_threshold=$DEDUP_THR env.memory.memory_text_insert_dedupe_similarity_threshold=null \
  "env.memory.rebuild_source_path=$BANK_JSONL" \
  "env.memory.embedding_api_url=$EMBEDDING_API_URL" env.memory.embedding_api_key=EMPTY env.memory.embedding_model=bge_m3 env.memory.embedding_dim=1024 \
  reward_model.format_reward.enable=True reward_model.format_reward.weight_outcome=1.0 reward_model.format_reward.weight_format=0.1 \
  reward_model.format_reward.require_memory_retrieve=False reward_model.format_reward.max_memory_retrieve_segments=0 reward_model.format_reward.format_warmup_global_steps=0 \
  reward_model.format_reward.warmup_weight_format_multiplier=0.0 reward_model.format_reward.warmup_require_memory_retrieve=False reward_model.format_reward.warmup_penalize_chinese_chars=False \
  env.rollout.n=8 env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 trainer.logger=['console'] ray_init.num_cpus=${RAY_CPUS:-40} \
  trainer.project_name=bmem_alfworld_grpo trainer.experiment_name="$RUN" \
  trainer.default_local_dir="$CKPT" trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.save_freq=10 trainer.max_actor_ckpt_to_keep=3 trainer.save_best_val_ckpt=True trainer.save_best_val_max_to_keep=2 trainer.save_best_val_mode=max trainer.save_best_val_metric=val/success_rate \
  trainer.test_freq=10 trainer.total_epochs=150 trainer.total_training_steps=$TOTAL_STEPS \
  trainer.validation_data_dir="$EXP/val_traj" trainer.val_before_train=${VAL_BEFORE:-True} trainer.val_only=False \
  hydra.run.dir="$EXP/hydra" \
  "$@" 2>&1 | tee -a "$EXP/train.log"
RC=${PIPESTATUS[0]}
echo "BMEM_GRPO_DONE rc=$RC run=$RUN ckpt=$CKPT"
exit $RC
