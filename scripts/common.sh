#!/bin/bash
# Shared setup for the B-Mem ALFWorld scripts: paths, method settings, BGE-M3 embedding server, verl-agent data files.
# Sourced by eval_alfworld.sh and train_grpo.sh.

BMEM_ROOT=${BMEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$BMEM_ROOT"
ASSETS=$BMEM_ROOT/bmem
OUT_ROOT=${OUT_ROOT:-$BMEM_ROOT/outputs}
EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}   # served as "bge_m3" through vLLM's OpenAI-compatible embedding endpoint
BANK_JSONL=${BANK_JSONL:-$ASSETS/skill_bank/alfworld_skill_bank.jsonl}

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export PYTHONPATH="$BMEM_ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN} WANDB_MODE=${WANDB_MODE:-disabled}
export RAY_memory_monitor_refresh_ms=0
export TMPDIR=${TMPDIR:-/tmp/bmem_$USER} RAY_TMPDIR=${RAY_TMPDIR:-/tmp/bmem_ray_$USER}
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$OUT_ROOT"
unset RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true

# ---- method settings (B-Mem, baselines, ablations). Shared harness: inline tool-call continuation, belief/query/guidance kept in the
# turn history, fixed skill bank retrieved by description (state_text), top-3, retrieval dedupe 0.92, one joint adaptation call.
set_method() {
  local m=$1
  export BMEM_CHAT_CONT=inline BMEM_TURN_HISTORY=1 RETRIEVE_KEY=state_text DEDUP_THR=0.92 ADAPTOR_TOPK=${TOPK:-3} BMEM_REWRITER_JOINT=1
  export BMEM_REWRITER_SYSTEM_FILE=$ASSETS/prompts/adapt_system.txt BMEM_REWRITER_USER_FILE=$ASSETS/prompts/adapt_user.txt
  MEMORY_ENABLED=True; SYSTEM_PROMPT_FILE=$ASSETS/prompts/policy_system_bmem.txt
  case $m in
    bmem)           export BMEM_PROMPT_VARIANT=plan;     RECON=True ;;                                     # B-Mem (full)
    wo_adapt)       export BMEM_PROMPT_VARIANT=plan;     RECON=False ;;                                    # w/o adaptation: raw top-k skills
    wo_belief_cond) export BMEM_PROMPT_VARIANT=plan;     RECON=True; export BMEM_ABL_REWRITER_NO_BELIEF=1 ;; # w/o belief conditioning
    exprag)         export BMEM_PROMPT_VARIANT=nobelief; RECON=False; SYSTEM_PROMPT_FILE=$ASSETS/prompts/policy_system_nobelief.txt ;;
    remem)          export BMEM_PROMPT_VARIANT=nobelief; RECON=True;  SYSTEM_PROMPT_FILE=$ASSETS/prompts/policy_system_nobelief.txt
                    export BMEM_REWRITER_SYSTEM_FILE=$ASSETS/prompts/adapt_system_nobelief.txt BMEM_REWRITER_USER_FILE=$ASSETS/prompts/adapt_user_nobelief.txt ;;
    react)          export BMEM_PROMPT_VARIANT=plain;    RECON=False; MEMORY_ENABLED=False; SYSTEM_PROMPT_FILE=$ASSETS/prompts/policy_system_react.txt ;;
    *) echo "unknown METHOD=$m (bmem | wo_adapt | wo_belief_cond | exprag | remem | react)"; return 1 ;;
  esac
  [ "$m" = wo_belief_cond ] || unset BMEM_ABL_REWRITER_NO_BELIEF
}

free_port() { python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }

wait_health() {   # wait_health <port> <pid> <log>
  for _ in $(seq 1 360); do
    curl -sf "localhost:$1/health" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "server died:"; tail -30 "$3"; return 1; }
    sleep 5
  done
  echo "server not healthy after 30 min"; return 1
}

start_embed_server() {   # start_embed_server <gpu> <gpu_mem_util> <log>  -> EMBEDDING_API_URL, EMBED_PID
  local port; port=$(free_port)
  CUDA_VISIBLE_DEVICES=$1 nohup vllm serve "$EMBED_MODEL" --served-model-name bge_m3 --task embed --host 127.0.0.1 --port "$port" \
      --gpu-memory-utilization "$2" --max-model-len 8192 --disable-log-requests > "$3" 2>&1 &
  EMBED_PID=$!
  wait_health "$port" "$EMBED_PID" "$3" || return 1
  export EMBEDDING_API_URL="http://127.0.0.1:$port/v1"
  echo "[bmem] embedding server pid=$EMBED_PID gpu=$1 url=$EMBEDDING_API_URL"
}

prepare_data() {   # prepare_data <dir> [extra args]: verl-agent placeholder parquet files (env workers supply the actual games)
  local d=$1; shift
  python3 -m examples.data_preprocess.prepare --mode text --local_dir "$d" "$@" >/dev/null || return 1
}
