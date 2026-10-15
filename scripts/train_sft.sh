#!/bin/bash
# Supervised finetuning of Qwen2.5-7B-Instruct on teacher trajectories (verl fsdp_sft_trainer, multi-turn messages).
#   DATA_DIR   output of build_sft_data.sh;  MODEL_ID (default Qwen/Qwen2.5-7B-Instruct);  SAVE_DIR
#   STEPS      training steps (default 100); EPOCHS=2 instead trains two full epochs (the RL initialization)
#   NPROC      GPUs (default 4)
# Global batch 16, lr 1e-5 cosine with 10% warmup, max length 4096. With STEPS, the cosine schedule spans ceil(STEPS / steps per
# epoch) epochs and training stops at STEPS. Convert the last checkpoint for vLLM with
#   python bmem/tools/convert_to_bf16.py $SAVE_DIR/global_step_<N> outputs/models/<name>_bf16
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1 WANDB_MODE=${WANDB_MODE:-disabled} PYTHONUNBUFFERED=1
DATA_DIR=${DATA_DIR:?set DATA_DIR}; MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}; NPROC=${NPROC:-4}
if [ -n "${EPOCHS:-}" ]; then LEN=(); NAME=ep$EPOCHS
else
  STEPS=${STEPS:-100}; NAME=s$STEPS
  ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/train.parquet')))") || exit 1
  SPE=$(( (ROWS + 15) / 16 )); EPOCHS=$(( (STEPS + SPE - 1) / SPE )); LEN=(trainer.total_training_steps=$STEPS)
  echo "[bmem] $ROWS rows, $SPE steps/epoch -> $STEPS steps ($EPOCHS epochs in the schedule)"
fi
SAVE_DIR=${SAVE_DIR:-outputs/models/sft_$(basename "$DATA_DIR")_$NAME}; mkdir -p "$SAVE_DIR"
MICRO=${MICRO:-1}   # micro batch per GPU; gradient accumulation fills the global batch of 16
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" data.val_files="$DATA_DIR/val.parquet" \
  data.multiturn.enable=true data.multiturn.messages_key=messages \
  data.train_batch_size=16 data.micro_batch_size_per_gpu=$MICRO data.max_length=4096 data.truncation=right \
  model.partial_pretrain="$MODEL_ID" model.enable_gradient_checkpointing=true \
  optim.lr=1e-5 optim.warmup_steps_ratio=0.1 optim.lr_scheduler=cosine \
  trainer.default_local_dir="$SAVE_DIR" trainer.project_name=bmem_sft trainer.experiment_name=$(basename "$SAVE_DIR") \
  trainer.logger=['console'] trainer.total_epochs=$EPOCHS ${LEN[@]+"${LEN[@]}"} trainer.default_hdfs_dir=null \
  ulysses_sequence_parallel_size=1 use_remove_padding=true "$@"
