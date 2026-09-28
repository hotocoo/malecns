#!/bin/bash
# Run MaleCNS trading training in a loop, restarting if it fails

cd /Users/acotech/workspace/malecns

while true; do
    echo "$(date): Starting training..."
    python3 src/train_trading_parallel.py \
        --graph data/graph_w5 \
        --out checkpoints/trading_es_v2.pt \
        --workers 4 \
        --population 48 \
        --generations 100000 \
        --episodes 5 \
        --episode-length 252 \
        --mutation 0.05 \
        --log-interval 5 \
        --save-interval 25 \
        --stagnation-window 50 \
        --resume checkpoints/trading_es_v2.pt \
        >> logs/training_output.log 2>&1
    
    EXIT_CODE=$?
    echo "$(date): Training exited with code $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date): Training completed successfully. Exiting loop."
        break
    else
        echo "$(date): Training failed. Restarting in 30 seconds..."
        sleep 30
    fi
done
