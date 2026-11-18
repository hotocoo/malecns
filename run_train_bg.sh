#!/bin/bash
# Run training in background, logging to file
cd /Users/acotech/workspace/malecns
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)  # project venv carries torch
exec "$PY" -W ignore src/train.py \
  --checkpoint checkpoints/es.pt \
  --best checkpoints/best.pt \
  --log logs/train.jsonl \
  --popsize 128 \
  --starts-per-gen 6 \
  >> logs/train_bg.log 2>&1