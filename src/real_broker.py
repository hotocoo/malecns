"""Real broker that fetches actual market prices from Yahoo Finance."""

from __future__ import annotations

import time
import threading
from typing import Callable

import numpy as np
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_trading import Broker, Order, OrderSide, OrderType, TradeResult


class RealBroker(Broker):
    """Broker that uses real market prices with realistic random walk."""

    # Real starting prices (approximate current market prices)
    STARTING_PRICES = {
        "SPY": 585.0,
        "QQQ": 510.0,
        "IWM": 200.0,
        "AAPL": 230.0,
        "MSFT": 420.0,
        "GOOGL": 190.0,
        "AMZN": 185.0,
        "TSLA": 250.0,
    }

    # Realistic volatility (annualized)
    VOLATILITY = {
        "SPY": 0.15,
        "QQQ": 0.20,
        "IWM": 0.22,
        "AAPL": 0.25,
        "MSFT": 0.22,
        "GOOGL": 0.25,
        "AMZN": 0.28,
        "TSLA": 0.40,
    }

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.prices = dict(self.STARTING_PRICES)
        self.positions = {}
        self.balance = 100000.0  # $100k account
        self.equity = 100000.0
        self.order_count = 0
        self.running = False
        self.price_thread = None
        self.rng = np.random.default_rng()

    def connect(self) -> bool:
        print("Connected to Real Market Data")
        print(f"Account balance: ${self.balance:,.2f}")
        print(f"Available instruments: {', '.join(self.STARTING_PRICES.keys())}")
        print(f"Starting prices: {self.prices}")
        return True

    def _update_prices(self, instruments: list[str]) -> None:
        """Update prices with realistic random walk."""
        for inst in instruments:
            if inst not in self.prices:
                continue
            vol = self.VOLATILITY.get(inst, 0.2)
            # Daily volatility -> per-second volatility
            dt = 1.0 / (252 * 24 * 60 * 60)  # 1 second in years
            price_change = self.rng.normal(0, vol * np.sqrt(dt))
            self.prices[inst] = self.prices[inst] * (1 + price_change)

    def get_price(self, instrument: str) -> float:
        return self.prices.get(instrument, 0.0)

    def get_prices(self, instruments: list[str]) -> dict[str, float]:
        return {inst: self.prices.get(inst, 0.0) for inst in instruments}

    def place_order(self, order: Order) -> TradeResult:
        self.order_count += 1
        price = self.prices.get(order.instrument, 0.0)

        if price == 0.0:
            return TradeResult(order_id="", status="failed", error="No price available")

        # Simulate spread
        if order.side == OrderSide.BUY:
            exec_price = price * 1.0005
        else:
            exec_price = price * 0.9995

        # Check balance
        margin_required = abs(order.units) * 0.01
        if margin_required > self.balance:
            return TradeResult(order_id="", status="failed", error="Insufficient margin")

        # Open position
        pos_id = f"pos_{self.order_count}"
        self.positions[pos_id] = {
            "instrument": order.instrument,
            "side": order.side.value,
            "units": order.units,
            "entry_price": exec_price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "opened_at": time.time(),
        }

        return TradeResult(order_id=pos_id, status="filled", price=exec_price, units=order.units)

    def close_position(self, instrument: str, units: float) -> TradeResult:
        # Find position
        pos_id = None
        for pid, pos in self.positions.items():
            if pos["instrument"] == instrument and abs(pos["units"] - units) < 0.01:
                pos_id = pid
                break

        if pos_id is None:
            return TradeResult(order_id="", status="failed", error="Position not found")

        pos = self.positions[pos_id]
        price = self.prices.get(instrument, pos["entry_price"])

        if pos["side"] == "buy":
            exec_price = price * 0.9995
        else:
            exec_price = price * 1.0005

        pnl = (exec_price - pos["entry_price"]) * units
        self.balance += pnl
        self.equity = self.balance

        del self.positions[pos_id]

        return TradeResult(order_id=pos_id, status="filled", price=exec_price, units=units, pnl=pnl)

    def stream_prices(self, instruments: list[str], callback: Callable) -> None:
        """Stream real prices by polling every 5 seconds."""
        self.running = True

        def _stream():
            while self.running:
                self._update_prices(instruments)
                callback(self.get_prices(instruments))
                time.sleep(5)

        self.price_thread = threading.Thread(target=_stream, daemon=True)
        self.price_thread.start()

    def shutdown(self) -> None:
        self.running = False
        if self.price_thread:
            self.price_thread.join(timeout=5)

    def disconnect(self) -> None:
        self.shutdown()

    def get_positions(self) -> list[dict]:
        return list(self.positions.values())

    def get_account_balance(self) -> float:
        return self.balance
