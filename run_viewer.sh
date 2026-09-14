#!/usr/bin/env bash
# Live browser viewer for the connectome driver. Safe to run alongside
# run_training.sh: it reads the checkpoint, never writes it.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f data/graph_w5/positions.npy ]; then
  echo "[viewer] building anatomical positions (one-off)"
  PYTHONPATH=src python3 -W ignore src/build_positions.py
fi

PORT="${PORT:-8765}"
echo "[viewer] http://127.0.0.1:${PORT}"
PYTHONPATH=src exec python3 -W ignore src/viewer.py --port "$PORT" "$@"
