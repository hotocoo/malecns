"""Demo: MaleCNS brain trading with simulated real-time market.

Runs the full pipeline: brain -> trading decisions -> demo broker -> P&L
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_agent import TradingAgent
from trading_env import TradingConfig
from real_broker import RealBroker
from realtime_trading import RealTimeTradingEngine, Order, OrderSide


def main():
    print("=" * 60)
    print("MALECNS REAL-TIME TRADING DEMO")
    print("=" * 60)
    print()

    # Find latest checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("trading_es_v2*.pt"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Using checkpoint: {latest}")
    else:
        latest = None
        print("No checkpoint found, using default brain")

    # Initialize real broker
    broker = RealBroker()
    broker.connect()
    print()

    # Initialize trading engine with checkpoint
    engine = RealTimeTradingEngine(
        broker=broker,
        graph_dir="data/graph_w5",
        instruments=["SPY", "QQQ", "AAPL"],
        device="cpu",
        risk_per_trade=0.01,
        max_positions=3,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        checkpoint=str(latest) if latest else None,
    )

    # Initialize brain
    engine.init_brain()
    print()

    # Run for 5 minutes (300 seconds)
    duration = 300
    print(f"Trading for {duration} seconds...")
    print("-" * 60)

    start_time = time.time()
    trade_count = 0

    def on_price(prices):
        nonlocal trade_count
        for inst, price in prices.items():
            engine.on_price_update({inst: price})

    broker.stream_prices(["SPY", "QQQ", "AAPL"], on_price)

    while time.time() - start_time < duration:
        time.sleep(1)

    # Shutdown
    print("-" * 60)
    print("Shutting down...")
    engine._shutdown()

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
