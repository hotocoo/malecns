#!/usr/bin/env bash
# Continuous headless training supervisor: restarts the ES loop if it ever
# dies, and resumes from the checkpoint so no generations are lost.
#
#   ./run_training.sh                      # Monaco, curriculum, popsize 128
#   POPSIZE=32 ./run_training.sh --seed 3  # any train.py flag passes through
#
# Training never renders or paces: watch it from another process with
# ./run_viewer.sh (reads checkpoints/es.pt) or record a run with
# python3 src/evaluate.py --record logs/run.npz and replay it.
set -u
cd "$(dirname "$0")"

CHECKPOINT=${CHECKPOINT:-checkpoints/es.pt}
BEST=${BEST:-checkpoints/best.pt}
LOG=${LOG:-logs/train.jsonl}
POPSIZE=${POPSIZE:-128}

while true; do
  python3 -W ignore src/train.py \
    --checkpoint "$CHECKPOINT" \
    --best "$BEST" \
    --log "$LOG" \
    --popsize "$POPSIZE" \
    "$@"
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "[supervisor] training exited cleanly"
    break
  fi
  echo "[supervisor] exit $status, restarting from checkpoint in 5s"
  sleep 5
done
