#!/bin/bash
# Run training detached from terminal
cd /Users/acotech/workspace/malecns

# Ignore all terminal signals
trap '' INT TERM HUP

# Run training, redirect all output
python3 -W ignore src/train.py \
  --checkpoint checkpoints/es.pt \
  --best checkpoints/best.pt \
  --log logs/train.jsonl \
  --popsize 128 \
  --starts-per-gen 6 \
  >> logs/train_detached.log 2>&1