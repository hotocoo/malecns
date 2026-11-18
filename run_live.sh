#!/usr/bin/env bash
# KL street driver seen through its own camera, with the YOLO detector's boxes.
# (run_viewer.sh is the Monaco lidar viewer; this is the real-street one.)
set -euo pipefail
cd "$(dirname "$0")"
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)  # project venv carries torch
PORT="${PORT:-8808}"
echo "[live] http://127.0.0.1:${PORT}"
PYTHONPATH=src exec "$PY" -W ignore src/live.py --port "$PORT" "$@"
