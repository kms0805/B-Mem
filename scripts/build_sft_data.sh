#!/bin/bash
# Teacher rollouts on the 200 training games -> SFT parquet (one row per assistant turn, plus the skill-adaptation calls).
#   EXPS     glob of teacher run dirs from eval_alfworld.sh SPLIT=train (e.g. 'outputs/eval/teacher_bmem_*')
#   VARIANT  full (methods with a belief) | nobelief (exprag, remem, react)
#   OUT      output dir (default outputs/data/sft/<name>)
# Keeps successful, protocol-clean episodes (one per game); the memory block inside a query+continuation turn is masked from the loss.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXPS=${EXPS:?set EXPS}; VARIANT=${VARIANT:-full}; OUT=${OUT:-outputs/data/sft/$(basename "${EXPS%%\**}")}
python3 bmem/tools/traj_to_sft.py --exps "$EXPS" --variant "$VARIANT" --max-queries 99 --out-dir "${OUT}_episode"
python3 bmem/tools/to_step_rows.py --inline --in-dir "${OUT}_episode" --out-dir "$OUT"
echo "SFT data -> $OUT/{train,val}.parquet"
