#!/bin/bash
# MaleCNS 24/7 Monitor - keeps training and trading running

cd /Users/acotech/workspace/malecns

while true; do
    # Check training process
    if ! pgrep -f "train_trading_real.py" > /dev/null; then
        echo "$(date): Training not running. Restarting..." >> logs/monitor.log
        python3 src/train_trading_real.py --graph data/graph_w5 --out checkpoints/trading_real.pt \
            --population 32 --generations 100000 --episodes 5 \
            --mutation 0.08 --mutation-max 0.3 --log-interval 5 --save-interval 10 --stagnation-window 20 \
            >> logs/training_output.log 2>&1 &
        echo "$(date): Training restarted with PID $!" >> logs/monitor.log
    fi

    # Check trading process
    if ! pgrep -f "demo_trade.py" > /dev/null; then
        echo "$(date): Trading not running. Restarting..." >> logs/monitor.log
        python3 src/demo_trade.py >> logs/demo_trading.log 2>&1 &
        echo "$(date): Trading restarted with PID $!" >> logs/monitor.log
    fi

    # Check for errors in recent logs
    if grep -q "Error\|Traceback\|Exception" logs/demo_trading.log 2>/dev/null; then
        echo "$(date): Errors detected in trading log" >> logs/monitor.log
    fi

    sleep 60
done
