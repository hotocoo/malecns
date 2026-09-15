#!/usr/bin/env bash
# Continuous headless training supervisor: restarts the ES loop if it ever
# dies, and resumes from the checkpoint so no generations are lost.
#
#   ./run_training.sh                      # Monaco, popsize 64 x 4 islands x 6 starts, 6k-step cap
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
# 64 members x 4 islands x 6 starts = 1,536 bodies per kernel launch, ~90 % of
# the Metal kernel's peak body-steps/s (3,072 bodies: 96 %, but every
# generation takes twice as long). The GPU is the only compute here: more
# islands or members do not add throughput, they only stretch a generation.
# ES over ~150 interface parameters needs many generations more than it needs
# a bigger population, so the default favours generations per hour.
POPSIZE=${POPSIZE:-64}
ISLANDS=${ISLANDS:-4}
# Every member is scored on all start points each generation: one start per
# generation made the objective rotate (a straight start scored +20, the
# hairpin start -18) and the curve looked like noise.
STARTS_PER_GEN=${STARTS_PER_GEN:-6}
# Episode cap in control steps (16 ms each). 6,000 = 96 s of driving: from six
# starts the population covers the lap twice over each generation, and a
# generation fits in minutes instead of the 40 min (uncapped) that gave two
# generations in three hours. EPISODE_STEPS=12000 keeps the lap bonus reachable
# at the cost of 2x per generation. Finished bodies are dropped from the GPU as
# they end (train.py rollout), so crashes cost nothing after they happen.
EPISODE_STEPS=${EPISODE_STEPS:-6000}
# ES step control for a mean that already drives (calibrated readout): small
# search radius, a trust region on each generation's move, and a deterministic
# evaluation of the mean every generation so a worse mean is reverted at once
# (--eval-tolerance) instead of after 10 generations.
SIGMA=${SIGMA:-0.02}
SIGMA_MAX=${SIGMA_MAX:-0.06}
LR=${LR:-0.02}
MAX_STEP_FRAC=${MAX_STEP_FRAC:-0.03}
EVAL_EVERY=${EVAL_EVERY:-1}

while true; do
  python3 -W ignore src/train.py \
    --checkpoint "$CHECKPOINT" \
    --best "$BEST" \
    --log "$LOG" \
    --popsize "$POPSIZE" \
    --islands "$ISLANDS" \
    --starts-per-gen "$STARTS_PER_GEN" \
    --episode-steps "$EPISODE_STEPS" \
    --sigma "$SIGMA" --sigma-max "$SIGMA_MAX" --lr "$LR" --max-step-frac "$MAX_STEP_FRAC" --eval-every "$EVAL_EVERY" \
    "$@"
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "[supervisor] training exited cleanly"
    break
  fi
  echo "[supervisor] exit $status, restarting from checkpoint in 5s"
  sleep 5
done
