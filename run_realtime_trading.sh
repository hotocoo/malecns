#!/bin/bash
# MaleCNS Real-Time Trading

set -e

GRAPH="data/graph_w5"
CHECKPOINT="checkpoints/trading_es.pt"
DEVICE="${DEVICE:-auto}"

command="$1"
shift

case "$command" in
    demo)
        echo "Starting demo real-time trading (simulated market)..."
        python3 src/realtime_trading.py --broker demo --graph "$GRAPH" --checkpoint "$CHECKPOINT" --device "$DEVICE" "$@"
        ;;
    oanda)
        echo "Starting real-time trading with OANDA..."
        python3 src/realtime_trading.py --broker oanda --graph "$GRAPH" --checkpoint "$CHECKPOINT" --device "$DEVICE" "$@"
        ;;
    etoro)
        echo "Starting real-time trading with eToro..."
        python3 src/realtime_trading.py --broker etoro --graph "$GRAPH" --checkpoint "$CHECKPOINT" --device "$DEVICE" "$@"
        ;;
    test)
        echo "Testing broker connection..."
        python3 -c "
import os, sys
sys.path.insert(0, 'src')
broker = os.environ.get('BROKER', 'demo')
if broker == 'demo':
    from demo_broker import DemoBroker
    b = DemoBroker()
    if b.connect():
        print(f'EUR_USD: {b.get_price(\"EUR_USD\"):.5f}')
        print(f'Balance: \${b.get_account_balance():,.2f}')
        b.disconnect()
elif broker == 'oanda':
    from realtime_trading import OandaBroker
    token = os.environ.get('OANDA_ACCESS_TOKEN', '')
    account = os.environ.get('OANDA_ACCOUNT_ID', '')
    if not token or not account:
        print('Set OANDA_ACCESS_TOKEN and OANDA_ACCOUNT_ID')
        sys.exit(1)
    b = OandaBroker(token, account, practice=True)
    if b.connect():
        print(f'EUR_USD: {b.get_price(\"EUR_USD\"):.5f}')
        print(f'Balance: \${b.get_account_balance():,.2f}')
        b.disconnect()
else:
    from realtime_trading import EtoroBroker
    token = os.environ.get('ETORO_ACCESS_TOKEN', '')
    if not token:
        print('Set ETORO_ACCESS_TOKEN')
        sys.exit(1)
    b = EtoroBroker(token)
    if b.connect():
        print(f'Balance: \${b.get_account_balance():,.2f}')
        b.disconnect()
"
        ;;
    *)
        echo "Usage: $0 {demo|oanda|etoro|test} [options]"
        echo ""
        echo "Commands:"
        echo "  demo     Demo trading with simulated market (no API needed)"
        echo "  oanda    Trade Forex via OANDA API"
        echo "  etoro    Trade via eToro API"
        echo "  test     Test broker connection"
        echo ""
        echo "Options:"
        echo "  --instrument EUR_USD    Trading instrument"
        echo "  --instruments EUR_USD,GBP_USD  Multiple instruments"
        echo "  --paper                 Use paper/practice account (default)"
        echo "  --live                  Use live account"
        echo "  --risk 0.01             Risk per trade (1% of balance)"
        echo "  --max-positions 3       Max simultaneous positions"
        echo "  --stop-loss 0.02        Stop loss (2%)"
        echo "  --take-profit 0.04      Take profit (4%)"
        echo ""
        echo "Getting Started:"
        echo "  1. Demo: ./run_realtime_trading.sh demo --instrument EUR_USD"
        echo "  2. OANDA: Create account at https://practice-oanda.com"
        echo "  3. Set OANDA_ACCESS_TOKEN and OANDA_ACCOUNT_ID env vars"
        echo "  4. ./run_realtime_trading.sh oanda --instrument EUR_USD --paper"
        ;;
esac
