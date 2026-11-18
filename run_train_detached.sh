#!/bin/bash
# Run training detached from terminal
cd /Users/acotech/workspace/malecns
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)  # project venv carries torch

# Ignore all terminal signals
trap '' INT TERM HUP

# Run training, redirect all output
"$PY" -W ignore src/train.py \
  --checkpoint checkpoints/es.pt \
  --best checkpoints/best.pt \
  --log logs/train.jsonl \
  --popsize 128 \
  --starts-per-gen 6 \
  >> logs/train_detached.log 2>&1