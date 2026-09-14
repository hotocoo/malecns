#!/bin/bash
# Run training in background, logging to file
cd /Users/acotech/workspace/malecns
exec python3 -W ignore src/train.py \
  --checkpoint checkpoints/es.pt \
  --best checkpoints/best.pt \
  --log logs/train.jsonl \
  --popsize 128 \
  --starts-per-gen 6 \
  >> logs/train_bg.log 2>&1