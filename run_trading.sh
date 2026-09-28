#!/bin/bash
# MaleCNS Trading System - Training and Viewing

set -e

GRAPH="data/graph_w5"
CHECKPOINT="checkpoints/trading_es.pt"
DEVICE="${DEVICE:-auto}"

command="$1"
shift

case "$command" in
    train)
        echo "Training MaleCNS trading agent..."
        python3 src/train_trading.py \
            --graph "$GRAPH" \
            --out "$CHECKPOINT" \
            --device "$DEVICE" \
            "$@"
        ;;
    view)
        echo "Starting trading viewer at http://127.0.0.1:8766/trading.html"
        python3 src/trading_viewer.py \
            --graph "$GRAPH" \
            --checkpoint "$CHECKPOINT" \
            --device "$DEVICE" \
            "$@"
        ;;
    test)
        echo "Running trading agent test..."
        python3 -c "
import sys
sys.path.insert(0, 'src')
from trading_agent import TradingAgent
agent = TradingAgent('$GRAPH', checkpoint='$CHECKPOINT', device='$DEVICE')
agent.reset()
for i in range(10):
    action = agent.act(agent.obs)
    obs, reward, done, info = agent.env.step(action)
    agent.obs = obs
    if done: break
metrics = agent.get_performance()
print(f'Return: {metrics[\"total_return\"]*100:.2f}%')
print(f'Sharpe: {metrics[\"sharpe_ratio\"]:.2f}')
print(f'Max DD: {metrics[\"max_drawdown\"]*100:.2f}%')
"
        ;;
    *)
        echo "Usage: $0 {train|view|test} [options]"
        echo ""
        echo "Commands:"
        echo "  train    Train the trading agent (ES)"
        echo "  view     Start the live trading viewer"
        echo "  test     Run a quick test episode"
        echo ""
        echo "Environment:"
        echo "  DEVICE   Device to use (cpu, mps, cuda, auto). Default: auto"
        ;;
esac
