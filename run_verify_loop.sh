#!/usr/bin/env bash
# Periodic end-to-end lap proof while training runs: whenever best.pt changes
# (a promoted checkpoint) it is driven from every start for 16,000 steps and
# the result appended to logs/laps.jsonl; es.pt (the live ES mean) is checked
# every ES_EVERY seconds. The record answers "is every lap complete and is it
# getting faster" independently of the trainer's own numbers.
set -u
cd "$(dirname "$0")"
BEST=${BEST:-checkpoints/best.pt}
ES=${ES:-checkpoints/es.pt}
POLL=${POLL:-120}
ES_EVERY=${ES_EVERY:-7200}
last_best=""
last_es=$(date +%s)
while true; do
  now=$(date +%s)
  if [ -f "$BEST" ]; then
    stamp=$(stat -f %m "$BEST")
    if [ "$stamp" != "$last_best" ]; then
      last_best=$stamp
      echo "[verify] $(date '+%H:%M:%S') best.pt changed; verifying"
      python3 -W ignore src/verify_laps.py --checkpoint "$BEST" --steps 16000
    fi
  fi
  if [ $((now - last_es)) -ge "$ES_EVERY" ]; then
    last_es=$now
    echo "[verify] $(date '+%H:%M:%S') es.pt periodic check"
    python3 -W ignore src/verify_laps.py --checkpoint "$ES" --steps 16000
  fi
  sleep "$POLL"
done
