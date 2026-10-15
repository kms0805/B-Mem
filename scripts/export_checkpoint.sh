#!/bin/bash
# Merge a GRPO actor checkpoint (FSDP shards) into a Hugging Face model for vLLM evaluation.
#   bash scripts/export_checkpoint.sh outputs/checkpoints/<run>/global_step_<N> outputs/models/<name>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP_DIR=${1:?checkpoint dir global_step_<N>}; OUT=${2:?output dir}; A=$STEP_DIR/actor
mkdir -p "$A/huggingface"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json; do
  [ -f "$A/$f" ] && cp "$A/$f" "$A/huggingface/"
done
PYTHONPATH="$PWD:${PYTHONPATH:-}" python3 bmem/tools/model_merger.py --local_dir "$A"
mkdir -p "$OUT" && cp "$A"/huggingface/* "$OUT"/ && echo "exported -> $OUT"
