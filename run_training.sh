#!/usr/bin/env bash
# Continuous headless training supervisor: restarts the ES loop if it ever
# dies, and resumes from the checkpoint so no generations are lost.
#
#   ./run_training.sh                      # Monaco, popsize 128 x 4 islands x 6 starts, 12k-step cap
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
ISLANDS=${ISLANDS:-4}
# Every member is scored on all start points each generation: one start per
# generation made the objective rotate (a straight start scored +20, the
# hairpin start -18) and the curve looked like noise.
STARTS_PER_GEN=${STARTS_PER_GEN:-6}
# Episode cap in control steps (16 ms each). 12,000 = 192 s of driving, enough
# for a full Monaco lap at the teacher's pace (~150-180 s) plus margin, so the
# lap bonus stays reachable. Without a cap one slow survivor kept a whole
# 3,072-body generation running for hours (zero generations in an hour).
# Finished bodies are dropped from the GPU as they end (see train.py rollout).
EPISODE_STEPS=${EPISODE_STEPS:-12000}

while true; do
  python3 -W ignore src/train.py \
    --checkpoint "$CHECKPOINT" \
    --best "$BEST" \
    --log "$LOG" \
    --popsize "$POPSIZE" \
    --islands "$ISLANDS" \
    --starts-per-gen "$STARTS_PER_GEN" \
    --episode-steps "$EPISODE_STEPS" \
    "$@"
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "[supervisor] training exited cleanly"
    break
  fi
  echo "[supervisor] exit $status, restarting from checkpoint in 5s"
  sleep 5
done
