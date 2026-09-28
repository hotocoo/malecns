#!/bin/bash
# Quick check on MaleCNS trading training status

cd /Users/acotech/workspace/malecns

echo "=== MaleCNS Trading Training Status ==="
echo "Time: $(date)"
echo ""

# Check if training is running
if pgrep -f "train_trading_v2.py" > /dev/null; then
    echo "Training: RUNNING"
    PID=$(pgrep -f "train_trading_v2.py")
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o %mem= | tr -d ' ')
    echo "  PID: $PID"
    echo "  CPU: ${CPU}%"
    echo "  MEM: ${MEM}%"
else
    echo "Training: NOT RUNNING"
fi

# Check if supervisor is running
if pgrep -f "training_supervisor.py" > /dev/null; then
    echo "Supervisor: RUNNING"
else
    echo "Supervisor: NOT RUNNING"
fi

echo ""

# Show latest training metrics
if [ -f logs/train_trading_v2.jsonl ]; then
    python3 src/training_report.py 2>&1 | head -30
else
    echo "No training log found."
fi
