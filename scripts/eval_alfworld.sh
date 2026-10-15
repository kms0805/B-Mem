#!/bin/bash
# ALFWorld inference with B-Mem, a baseline, or an ablation. One validation pass over a pinned game list: worker i plays game i.
#
#   METHOD   bmem | wo_adapt | wo_belief_cond | exprag | remem | react            (default bmem)
#   SPLIT    id (140 unique in-distribution games) | ood (134 out-of-distribution games) | train (200 training games, teacher data)
#   POLICY   api   : prompted model behind an OpenAI-compatible endpoint (policy and skill adaptation are the same model)
#                    API_BASE (default https://api.openai.com/v1), API_MODEL, API_KEY_ENV (name of the env var holding the key,
#                    default OPENAI_API_KEY), REASONING (low | none), TOKENIZER_PATH (any local Qwen2.5 checkpoint dir; only its
#                    tokenizer is used for prompt-length checks)
#            local : finetuned student served with vLLM on the first visible GPU; MODEL_PATH
#   TOPK (3), SEED (0; 7 for SPLIT=train), TAG, OUT_ROOT (./outputs)
# Output: $OUT_ROOT/eval/<TAG>_<date>/val_traj/0.jsonl (every policy call: full prompt and output, reward, game file) and
#         adaptor_io.jsonl (every skill-adaptation call). Success rate is printed at the end (val/success_rate).
#
# Examples:
#   METHOD=bmem POLICY=api API_MODEL=gpt-6-luna TOKENIZER_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/eval_alfworld.sh
#   METHOD=bmem POLICY=api API_BASE=https://openrouter.ai/api/v1 API_MODEL=qwen/qwen3-235b-a22b-2507 API_KEY_ENV=OPENROUTER_API_KEY \
#     REASONING=none TOKENIZER_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/eval_alfworld.sh
#   METHOD=bmem POLICY=local MODEL_PATH=outputs/models/sft_bmem_bf16 bash scripts/eval_alfworld.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
METHOD=${METHOD:-bmem}; SPLIT=${SPLIT:-id}; POLICY=${POLICY:-api}
set_method "$METHOD" || exit 1
case $SPLIT in
  id)    ENV_SPLIT=eval_in_distribution;     GAMES=$ASSETS/games/alfworld_id140.txt;    ON_TRAIN=False; SEED=${SEED:-0} ;;
  ood)   ENV_SPLIT=eval_out_of_distribution; GAMES=$ASSETS/games/alfworld_ood134.txt;   ON_TRAIN=False; SEED=${SEED:-0} ;;
  train) ENV_SPLIT=eval_in_distribution;     GAMES=$ASSETS/games/alfworld_train200.txt; ON_TRAIN=True;  SEED=${SEED:-7} ;;
  *) echo "SPLIT must be id | ood | train"; exit 1 ;;
esac
export BMEM_GAME_LIST=${GAME_LIST:-$GAMES}; N=$(grep -c . "$BMEM_GAME_LIST")
TAG=${TAG:-${POLICY}_${METHOD}_${SPLIT}}
EXP=$OUT_ROOT/eval/${TAG}_$(date +%m%d-%H%M%S); mkdir -p "$EXP"
export BMEM_ADAPTOR_DUMP=$EXP/adaptor_io.jsonl
DATA=$OUT_ROOT/data/alfworld_n$N; prepare_data "$DATA" --train_data_size 16 --val_data_size "$N" || exit 1

IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
EMBED_PID=""; POLICY_PID=""
trap '[ -n "$EMBED_PID" ] && kill $EMBED_PID 2>/dev/null; [ -n "$POLICY_PID" ] && kill $POLICY_PID 2>/dev/null' EXIT
if [ "$MEMORY_ENABLED" = True ]; then
  if [ "$POLICY" = local ] && [ ${#GPUS[@]} -eq 1 ]; then start_embed_server "${GPUS[0]}" 0.06 "$EXP/embed_server.log" || exit 1
  else start_embed_server "${GPUS[${#GPUS[@]}-1]}" "${EMBED_UTIL:-0.3}" "$EXP/embed_server.log" || exit 1; fi
fi

API=(); FIRST_QUERY=0
if [ "$POLICY" = api ]; then
  KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}; [ -n "${!KEY_ENV:-}" ] || { echo "$KEY_ENV is not set"; exit 1; }
  BASE_URL=${API_BASE:-https://api.openai.com/v1}; MODEL_ARG=${TOKENIZER_PATH:?set TOKENIZER_PATH to a local Qwen2.5 checkpoint dir}
  export BMEM_SYSTEM_PROMPT_FILE=$SYSTEM_PROMPT_FILE
  [ "$MEMORY_ENABLED" = True ] && FIRST_QUERY=1   # prompted models open every episode with a memory query
  API=(actor_rollout_ref.rollout.openai_api.base_url=$BASE_URL actor_rollout_ref.rollout.openai_api.model=${API_MODEL:?set API_MODEL}
       actor_rollout_ref.rollout.openai_api.api_key=ENV:$KEY_ENV actor_rollout_ref.rollout.openai_api.max_concurrent=${CONC:-32}
       actor_rollout_ref.rollout.openai_api.timeout_sec=1800 actor_rollout_ref.rollout.openai_api.max_retries=8 actor_rollout_ref.rollout.openai_api.retry_backoff_sec=5)
  [ "${REASONING:-low}" != none ] && API+=("+actor_rollout_ref.rollout.openai_api.extra_body.reasoning_effort=${REASONING:-low}")
  [[ "$BASE_URL" == *api.openai.com* ]] && API+=("+actor_rollout_ref.rollout.openai_api.extra_body.service_tier=${SERVICE_TIER:-flex}")
elif [ "$POLICY" = local ]; then
  MODEL_ARG=${MODEL_PATH:?set MODEL_PATH}; PORT=$(free_port)
  CUDA_VISIBLE_DEVICES=${GPUS[0]} nohup vllm serve "$MODEL_PATH" --served-model-name policy --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization ${POLICY_UTIL:-0.8} --max-model-len $((16384+512)) --enable-prefix-caching --disable-log-requests > "$EXP/policy_server.log" 2>&1 &
  POLICY_PID=$!; wait_health "$PORT" "$POLICY_PID" "$EXP/policy_server.log" || exit 1
  export BMEM_API_PREFILL=1   # the student continues its own assistant message after the memory block (tool-call style)
  unset BMEM_SYSTEM_PROMPT_FILE   # students use the default Qwen system message; the protocol is in the observation template
  API=(actor_rollout_ref.rollout.openai_api.base_url=http://127.0.0.1:$PORT/v1 actor_rollout_ref.rollout.openai_api.model=policy
       actor_rollout_ref.rollout.openai_api.api_key=EMPTY actor_rollout_ref.rollout.openai_api.max_concurrent=64
       actor_rollout_ref.rollout.openai_api.timeout_sec=600 actor_rollout_ref.rollout.openai_api.max_retries=3
       "actor_rollout_ref.rollout.openai_api.system_prompt='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'")
else echo "POLICY must be api | local"; exit 1; fi
export BMEM_FORCE_FIRST_QUERY=${FORCE_FIRST_QUERY:-$FIRST_QUERY}

echo "[bmem] method=$METHOD split=$SPLIT games=$N policy=$POLICY exp=$EXP"
python3 -m verl.trainer.main_api_rollout \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/text/train.parquet" data.val_files="$DATA/text/test.parquet" \
  data.train_batch_size=2 data.val_batch_size="$N" \
  data.max_prompt_length=16384 data.max_response_length=512 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_ARG" actor_rollout_ref.actor.trainable=False \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 actor_rollout_ref.rollout.free_cache_engine=False actor_rollout_ref.rollout.max_num_seqs=64 actor_rollout_ref.rollout.enforce_eager=True mem_adaptor.ref_param_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=openai_api actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  "${API[@]}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  mem_adaptor.enable=$RECON mem_adaptor.retrieval_top_k=$ADAPTOR_TOPK mem_adaptor.use_actor_rollout_wg=True mem_adaptor.train_memory_adaptor=false \
  mem_adaptor.max_prompt_chars=40000 mem_adaptor.max_new_tokens=2048 \
  env.env_name=alfworld/AlfredTWEnv env.alfworld.eval_dataset=$ENV_SPLIT env.alfworld.validate_on_train_split=$ON_TRAIN \
  env.seed=$SEED env.max_steps=50 env.memory.count_query_as_step=False \
  env.memory.enabled=$MEMORY_ENABLED env.memory.store_dir="$EXP/memory_vdb" env.memory.write_back=False env.memory.server_startup_timeout=600 \
  env.memory.experience_summarizer.mode=none env.memory.retrieval_mode=agentic env.memory.retrieve_key=$RETRIEVE_KEY env.memory.top_k=$ADAPTOR_TOPK \
  env.memory.experience_utility.enable=False \
  env.memory.memory_text_retrieval_dedupe_similarity_threshold=$DEDUP_THR env.memory.memory_text_insert_dedupe_similarity_threshold=null \
  "env.memory.rebuild_source_path=$BANK_JSONL" \
  "env.memory.embedding_api_url=${EMBEDDING_API_URL:-http://127.0.0.1:1/v1}" env.memory.embedding_api_key=EMPTY env.memory.embedding_model=bge_m3 env.memory.embedding_dim=1024 \
  reward_model.format_reward.enable=True reward_model.format_reward.require_memory_retrieve=$MEMORY_ENABLED \
  env.rollout.n=1 env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 trainer.logger=['console'] ray_init.num_cpus=${RAY_CPUS:-32} +ray_init.object_store_memory=4000000000 \
  trainer.project_name=bmem_alfworld_eval trainer.experiment_name="$TAG" \
  trainer.default_local_dir="$EXP/ckpt" trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq=-1 trainer.save_best_val_ckpt=False trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.validation_data_dir="$EXP/val_traj" trainer.val_before_train=True trainer.val_only=True \
  hydra.run.dir="$EXP/hydra" \
  "$@" 2>&1 | tee "$EXP/eval.log"
RC=${PIPESTATUS[0]}
grep -E "val/success_rate" "$EXP/eval.log" | tail -1 | grep -oE "val/success_rate[^,]*"
echo "BMEM_EVAL_DONE rc=$RC exp=$EXP"
exit $RC
