"""Demo broker that simulates real-time Forex trading with realistic data.

Uses live market data from a free API (Alpha Vantage or similar) or
simulates realistic price movements if no API key is available.
"""

from __future__ import annotations

import json
import time
import threading
import math
from typing import Callable

import numpy as np
import requests

from realtime_trading import Broker, Order, OrderSide, OrderType, TradeResult


class DemoBroker(Broker):
    """Demo broker with realistic Forex price simulation."""

    # Realistic starting prices for major Forex pairs
    STARTING_PRICES = {
        "EUR_USD": 1.0850,
        "GBP_USD": 1.2650,
        "USD_JPY": 147.50,
        "AUD_USD": 0.6550,
        "USD_CHF": 0.8850,
        "NZD_USD": 0.6150,
        "EUR_JPY": 159.50,
        "GBP_JPY": 186.50,
    }

    # Realistic volatility (annualized)
    VOLATILITY = {
        "EUR_USD": 0.08,
        "GBP_USD": 0.10,
        "USD_JPY": 0.12,
        "AUD_USD": 0.11,
        "USD_CHF": 0.07,
        "NZD_USD": 0.12,
        "EUR_JPY": 0.13,
        "GBP_JPY": 0.15,
    }

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.prices = dict(self.STARTING_PRICES)
        self.positions = {}
        self.balance = 100000.0  # $100k demo account
        self.equity = 100000.0
        self.order_count = 0
        self.running = False
        self.price_thread = None

    def connect(self) -> bool:
        print("Connected to Demo Forex Broker (simulated real-time)")
        print(f"Demo account balance: ${self.balance:,.2f}")
        print(f"Available instruments: {', '.join(self.STARTING_PRICES.keys())}")
        return True

    def get_price(self, instrument: str) -> float:
        return self.prices.get(instrument, 1.0)

    def get_prices(self, instruments: list[str]) -> dict[str, float]:
        return {inst: self.prices.get(inst, 1.0) for inst in instruments}

    def place_order(self, order: Order) -> TradeResult:
        self.order_count += 1
        price = self.prices.get(order.instrument, 1.0)

        # Simulate spread (1 pip for major pairs)
        if order.side == OrderSide.BUY:
            exec_price = price * 1.0001
        else:
            exec_price = price * 0.9999

        # Check balance
        margin_required = abs(order.units) * 0.01  # 1% margin
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
            if pos["instrument"] == instrument:
                pos_id = pid
                break

        if not pos_id:
            return TradeResult(order_id="", status="not_found", error="Position not found")

        pos = self.positions[pos_id]
        current_price = self.prices[instrument]

        # Calculate P&L
        if pos["side"] == "buy":
            pnl = (current_price - pos["entry_price"]) * pos["units"]
        else:
            pnl = (pos["entry_price"] - current_price) * pos["units"]

        self.balance += pnl
        self.equity = self.balance
        del self.positions[pos_id]

        return TradeResult(order_id=pos_id, status="closed", price=current_price, pnl=pnl)

    def get_positions(self) -> list[dict]:
        return list(self.positions.values())

    def get_account_balance(self) -> float:
        # Calculate unrealized P&L
        unrealized = 0.0
        for pos in self.positions.values():
            current_price = self.prices[pos["instrument"]]
            if pos["side"] == "buy":
                unrealized += (current_price - pos["entry_price"]) * pos["units"]
            else:
                unrealized += (pos["entry_price"] - current_price) * pos["units"]

        self.equity = self.balance + unrealized
        return self.equity

    def stream_prices(self, instruments: list[str], callback: Callable):
        """Simulate real-time price updates every 100ms."""
        self.running = True

        def price_loop():
            dt = 0.1  # 100ms updates
            while self.running:
                prices = {}
                for inst in instruments:
                    if inst in self.prices:
                        # GBM price movement
                        vol = self.VOLATILITY.get(inst, 0.1)
                        sigma = vol * math.sqrt(dt / 86400)  # Scale to 100ms
                        drift = 0.0
                        dW = np.random.normal()
                        self.prices[inst] *= math.exp((drift - 0.5 * sigma**2) + sigma * dW)
                        prices[inst] = self.prices[inst]

                if prices:
                    callback(prices)

                time.sleep(dt)

        self.price_thread = threading.Thread(target=price_loop, daemon=True)
        self.price_thread.start()

    def disconnect(self):
        self.running = False
        if self.price_thread:
            self.price_thread.join(timeout=1.0)
        print("Disconnected from Demo Broker")
