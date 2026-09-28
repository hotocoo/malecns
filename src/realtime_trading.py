"""Real-time trading system for MaleCNS.

Connects to real brokers (OANDA, eToro) for live Forex/CFD trading.
Uses the fly brain as the trading decision engine with real-time market data.

Features:
- Real-time market data via WebSocket streaming
- Live order execution through broker APIs
- Multiple broker support (OANDA v20, eToro)
- Risk management (stop loss, take profit, position sizing)
- Paper trading mode for testing
- Real-time P&L tracking

Broker APIs:
- OANDA v20: https://developer.oanda.com/rest-live-v20/
- eToro: https://api-portal.etoro.com/ (via etoropy SDK)

Usage:
    python3 src/realtime_trading.py --broker oanda --instrument EUR_USD --paper
    python3 src/realtime_trading.py --broker etoro --instrument BTCUSD --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_agent import TradingAgent
from trading_env import TradingConfig


# --- Order Types ---
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    instrument: str
    side: OrderSide
    units: float
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    label: str = "malecns"


@dataclass
class TradeResult:
    order_id: str
    status: str
    price: float | None = None
    units: float | None = None
    pnl: float | None = None
    error: str | None = None


# --- Broker Base Class ---
class Broker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def get_price(self, instrument: str) -> float:
        pass

    @abstractmethod
    def get_prices(self, instruments: list[str]) -> dict[str, float]:
        pass

    @abstractmethod
    def place_order(self, order: Order) -> TradeResult:
        pass

    @abstractmethod
    def close_position(self, instrument: str, units: float) -> TradeResult:
        pass

    @abstractmethod
    def get_positions(self) -> list[dict]:
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        pass

    @abstractmethod
    def stream_prices(self, instruments: list[str], callback: Callable):
        pass

    @abstractmethod
    def disconnect(self):
        pass


# --- OANDA v20 Broker ---
class OandaBroker(Broker):
    """OANDA v20 REST API broker for Forex trading."""

    BASE_URL = "https://api-fxtrade.oanda.com/v3"
    PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"

    def __init__(self, access_token: str, account_id: str, practice: bool = True):
        self.access_token = access_token
        self.account_id = account_id
        self.practice = practice
        self.base_url = self.PRACTICE_URL if practice else self.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self.ws = None
        self.ws_thread = None

    def connect(self) -> bool:
        """Test connection by fetching account info."""
        try:
            resp = self.session.get(f"{self.base_url}/accounts/{self.account_id}")
            if resp.status_code == 200:
                print(f"Connected to OANDA {'practice' if self.practice else 'live'} account: {self.account_id}")
                return True
            else:
                print(f"OANDA connection failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            print(f"OANDA connection error: {e}")
            return False

    def get_price(self, instrument: str) -> float:
        """Get current price for instrument."""
        resp = self.session.get(f"{self.base_url}/accounts/{self.account_id}/instruments/{instrument}/candles", params={
            "granularity": "S5",
            "count": 1,
            "price": "BAM",
        })
        if resp.status_code == 200:
            data = resp.json()
            if data.get("candles"):
                return float(data["candles"][-1]["m"]["c"])
        raise Exception(f"Failed to get price for {instrument}: {resp.text}")

    def get_prices(self, instruments: list[str]) -> dict[str, float]:
        """Get current prices for multiple instruments."""
        prices = {}
        for inst in instruments:
            try:
                prices[inst] = self.get_price(inst)
            except Exception as e:
                print(f"Error getting price for {inst}: {e}")
        return prices

    def place_order(self, order: Order) -> TradeResult:
        """Place a market order."""
        payload = {
            "order": {
                "type": "MARKET",
                "instrument": order.instrument,
                "units": str(int(order.units)) if order.side == OrderSide.BUY else str(-int(order.units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "label": order.label,
            }
        }
        if order.stop_loss:
            payload["order"]["stopLossOnFill"] = {"price": str(order.stop_loss)}
        if order.take_profit:
            payload["order"]["takeProfitOnFill"] = {"price": str(order.take_profit)}

        resp = self.session.post(
            f"{self.base_url}/accounts/{self.account_id}/orders",
            json=payload,
        )

        if resp.status_code == 202:
            data = resp.json()
            order_id = data["orderCreateTransaction"]["orderID"]
            return TradeResult(order_id=str(order_id), status="filled")
        else:
            return TradeResult(order_id="", status="failed", error=resp.text)

    def close_position(self, instrument: str, units: float) -> TradeResult:
        """Close a position."""
        positions = self.get_positions()
        for pos in positions:
            if pos["instrument"] == instrument:
                pos_id = pos["id"]
                payload = {"units": "ALL", "timeInForce": "FOK"}
                resp = self.session.post(
                    f"{self.base_url}/accounts/{self.account_id}/positions/{pos_id}/close",
                    json=payload,
                )
                if resp.status_code == 202:
                    return TradeResult(order_id=pos_id, status="closed")
                else:
                    return TradeResult(order_id=pos_id, status="failed", error=resp.text)
        return TradeResult(order_id="", status="not_found", error="Position not found")

    def get_positions(self) -> list[dict]:
        """Get open positions."""
        resp = self.session.get(f"{self.base_url}/accounts/{self.account_id}/positions")
        if resp.status_code == 200:
            return resp.json().get("positions", [])
        return []

    def get_account_balance(self) -> float:
        """Get account balance."""
        resp = self.session.get(f"{self.base_url}/accounts/{self.account_id}")
        if resp.status_code == 200:
            return float(resp.json()["account"]["balance"])
        return 0.0

    def stream_prices(self, instruments: list[str], callback: Callable):
        """Stream real-time prices via WebSocket."""
        url = "wss://stream-fxpractice.oanda.com/v3/prices" if self.practice else "wss://stream-fxtrade.oanda.com/v3/prices"
        instruments_param = ",".join(instruments)

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("type") == "PRICE":
                    prices = {data["instrument"]: float(data["prices"][0]["b"])}
                    callback(prices)
            except Exception as e:
                print(f"WebSocket error: {e}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_open(ws):
            subscribe = {
                "type": "PRICE",
                "instrumentNames": instruments,
            }
            ws.send(json.dumps(subscribe))

        self.ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_open=on_open,
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def disconnect(self):
        if self.ws:
            self.ws.close()


# --- eToro Broker (via etoropy) ---
class EtoroBroker(Broker):
    """eToro broker using etoropy SDK."""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.client = None

    def connect(self) -> bool:
        try:
            import etoropy
            self.client = etoropy.EtoroClient(access_token=self.access_token)
            # Test connection
            self.client.get_account()
            print("Connected to eToro")
            return True
        except Exception as e:
            print(f"eToro connection error: {e}")
            return False

    def get_price(self, instrument: str) -> float:
        # eToro uses different instrument codes
        data = self.client.get_instrument_price(instrument)
        return float(data["price"])

    def get_prices(self, instruments: list[str]) -> dict[str, float]:
        prices = {}
        for inst in instruments:
            try:
                prices[inst] = self.get_price(inst)
            except Exception as e:
                print(f"Error getting price for {inst}: {e}")
        return prices

    def place_order(self, order: Order) -> TradeResult:
        try:
            if order.side == OrderSide.BUY:
                result = self.client.open_long(instrument=order.instrument, amount=order.units)
            else:
                result = self.client.open_short(instrument=order.instrument, amount=order.units)
            return TradeResult(order_id=str(result.get("tradeId", "")), status="filled")
        except Exception as e:
            return TradeResult(order_id="", status="failed", error=str(e))

    def close_position(self, instrument: str, units: float) -> TradeResult:
        try:
            trades = self.client.get_trades()
            for trade in trades:
                if trade["instrument"] == instrument:
                    self.client.close_trade(trade["tradeId"])
                    return TradeResult(order_id=str(trade["tradeId"]), status="closed")
            return TradeResult(order_id="", status="not_found", error="Position not found")
        except Exception as e:
            return TradeResult(order_id="", status="failed", error=str(e))

    def get_positions(self) -> list[dict]:
        try:
            return self.client.get_trades()
        except Exception:
            return []

    def get_account_balance(self) -> float:
        try:
            account = self.client.get_account()
            return float(account.get("netLiquidation", 0))
        except Exception:
            return 0.0

    def stream_prices(self, instruments: list[str], callback: Callable):
        # eToro streaming via etoropy
        def on_price(data):
            callback({data["instrument"]: float(data["price"])})

        self.client.stream_prices(instruments, on_price)

    def disconnect(self):
        if self.client:
            self.client.close()


# --- Real-Time Trading Engine ---
class RealTimeTradingEngine:
    """Real-time trading engine using MaleCNS brain."""

    def __init__(
        self,
        broker: Broker,
        graph_dir: str,
        instruments: list[str],
        checkpoint: str | None = None,
        device: str = "auto",
        risk_per_trade: float = 0.01,  # 1% of balance per trade
        max_positions: int = 3,
        stop_loss_pct: float = 0.02,   # 2% stop loss
        take_profit_pct: float = 0.04, # 4% take profit
    ):
        self.broker = broker
        self.graph_dir = graph_dir
        self.instruments = instruments
        self.checkpoint = checkpoint
        self.device = device
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Trading state
        self.positions: dict[str, dict] = {}
        self.balance = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.running = False

        # Brain agent (shared across instruments)
        self.agent = None

    def init_brain(self):
        """Initialize the MaleCNS brain."""
        print("Initializing MaleCNS brain...")
        self.agent = TradingAgent(
            self.graph_dir,
            checkpoint=self.checkpoint,
            device=self.device,
        )
        print("Brain initialized")

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk per trade."""
        if price <= 0:
            return 0.0
        balance = self.broker.get_account_balance()
        risk_amount = balance * self.risk_per_trade
        # For Forex, 1 standard lot = 100,000 units
        # Risk 1% with 2% stop loss = 50,000 units (0.5 lots)
        position_size = risk_amount / (price * self.stop_loss_pct)
        return min(position_size, 100000)  # Max 1 standard lot

    def on_price_update(self, prices: dict[str, float]):
        """Handle real-time price updates."""
        for instrument, price in prices.items():
            self._process_instrument(instrument, price)

    def _process_instrument(self, instrument: str, price: float):
        """Process trading logic for one instrument."""
        # Check existing position for stop loss / take profit
        if instrument in self.positions:
            pos = self.positions[instrument]
            entry_price = pos["entry_price"]
            side = pos["side"]

            if side == "buy":
                pnl_pct = (price - entry_price) / entry_price
                if pnl_pct <= -self.stop_loss_pct:
                    print(f"STOP LOSS: {instrument} at {price:.5f}")
                    self._close_position(instrument, pos["units"])
                    return
                elif pnl_pct >= self.take_profit_pct:
                    print(f"TAKE PROFIT: {instrument} at {price:.5f}")
                    self._close_position(instrument, pos["units"])
                    return
            else:  # sell
                pnl_pct = (entry_price - price) / entry_price
                if pnl_pct <= -self.stop_loss_pct:
                    print(f"STOP LOSS: {instrument} at {price:.5f}")
                    self._close_position(instrument, pos["units"])
                    return
                elif pnl_pct >= self.take_profit_pct:
                    print(f"TAKE PROFIT: {instrument} at {price:.5f}")
                    self._close_position(instrument, pos["units"])
                    return

        # Check if we can open new position
        if len(self.positions) >= self.max_positions:
            return

        # Get brain decision
        if self.agent is None:
            return

        # Create observation from price data
        obs = self._create_observation(instrument, price)
        action = self.agent.act(obs)

        # Map brain action to trade
        from trading_env import TradeAction
        if action == TradeAction.BUY_SMALL or action == TradeAction.BUY_MEDIUM or action == TradeAction.BUY_LARGE:
            units = self.calculate_position_size(price)
            if action == TradeAction.BUY_MEDIUM:
                units *= 1.5
            elif action == TradeAction.BUY_LARGE:
                units *= 2.0

            sl = price * (1 - self.stop_loss_pct)
            tp = price * (1 + self.take_profit_pct)

            order = Order(
                instrument=instrument,
                side=OrderSide.BUY,
                units=units,
                stop_loss=sl,
                take_profit=tp,
            )
            result = self.broker.place_order(order)
            if result.status == "filled":
                print(f"BUY {instrument}: {units:.0f} units at {price:.5f} (SL: {sl:.5f}, TP: {tp:.5f})")
                self.positions[instrument] = {
                    "entry_price": price,
                    "units": units,
                    "side": "buy",
                    "order_id": result.order_id,
                }
                self.trade_count += 1

        elif action == TradeAction.SELL_SMALL or action == TradeAction.SELL_MEDIUM or action == TradeAction.SELL_LARGE:
            units = self.calculate_position_size(price)
            if action == TradeAction.SELL_MEDIUM:
                units *= 1.5
            elif action == TradeAction.SELL_LARGE:
                units *= 2.0

            sl = price * (1 + self.stop_loss_pct)
            tp = price * (1 - self.take_profit_pct)

            order = Order(
                instrument=instrument,
                side=OrderSide.SELL,
                units=units,
                stop_loss=sl,
                take_profit=tp,
            )
            result = self.broker.place_order(order)
            if result.status == "filled":
                print(f"SELL {instrument}: {units:.0f} units at {price:.5f} (SL: {sl:.5f}, TP: {tp:.5f})")
                self.positions[instrument] = {
                    "entry_price": price,
                    "units": units,
                    "side": "sell",
                    "order_id": result.order_id,
                }
                self.trade_count += 1

    def _create_observation(self, instrument: str, price: float) -> dict:
        """Create observation dict for brain from real price data."""
        # In a full implementation, this would use historical data
        # For now, use simplified observation
        return {
            "price_change": 0.0,
            "price_change_5": 0.0,
            "price_change_20": 0.0,
            "price_normalized": price / 1.0,
            "position_pct": 0.0,
            "equity_normalized": 1.0,
            "unrealized_pnl": 0.0,
            "rsi": 0.0,
            "macd": 0.0,
            "macd_line": 0.0,
            "bb_position": 0.0,
            "bb_width": 0.02,
            "atr": 0.01,
            "momentum_5": 0.0,
            "realized_vol": 0.1,
            "spread": 0.0001,
            "drawdown": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.02,
            "regime_trending_up": 0.0,
            "regime_trending_down": 0.0,
            "regime_mean_reverting": 1.0,
            "regime_high_volatility": 0.0,
            "step": 0.0,
        }

    def _close_position(self, instrument: str, units: float):
        """Close a position and record P&L."""
        result = self.broker.close_position(instrument, units)
        if result.status == "closed":
            pos = self.positions.pop(instrument)
            current_price = self.broker.get_price(instrument)
            if pos["side"] == "buy":
                pnl = (current_price - pos["entry_price"]) * units
            else:
                pnl = (pos["entry_price"] - current_price) * units
            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1
            print(f"Closed {instrument}: P&L = {pnl:.2f} USD")

    def run(self):
        """Start the real-time trading engine."""
        print("Starting real-time trading engine...")
        self.broker.connect()
        self.init_brain()

        self.balance = self.broker.get_account_balance()
        print(f"Account balance: ${self.balance:.2f}")
        print(f"Instruments: {self.instruments}")
        print(f"Risk per trade: {self.risk_per_trade*100}%")
        print(f"Max positions: {self.max_positions}")
        print("Waiting for price updates...")

        self.running = True
        self.broker.stream_prices(self.instruments, self.on_price_update)

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self._shutdown()

    def _shutdown(self):
        """Close all positions and disconnect."""
        self.running = False
        print("Closing all positions...")
        for instrument in list(self.positions.keys()):
            pos = self.positions[instrument]
            self._close_position(instrument, pos["units"])
        self.broker.disconnect()
        print(f"Total P&L: ${self.total_pnl:.2f}")
        print(f"Trades: {self.trade_count}, Wins: {self.win_count}")
        if self.trade_count > 0:
            print(f"Win rate: {self.win_count/self.trade_count*100:.1f}%")


def get_demo_broker():
    from demo_broker import DemoBroker
    return DemoBroker()

def main():
    parser = argparse.ArgumentParser(description="MaleCNS Real-Time Trading")
    parser.add_argument("--broker", choices=["oanda", "etoro"], default="oanda")
    parser.add_argument("--instrument", default="EUR_USD", help="Trading instrument (e.g., EUR_USD, GBP_USD, BTCUSD)")
    parser.add_argument("--instruments", default=None, help="Comma-separated list of instruments")
    parser.add_argument("--paper", action="store_true", help="Use paper/practice account")
    parser.add_argument("--live", action="store_true", help="Use live account")
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade (fraction of balance)")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--stop-loss", type=float, default=0.02, help="Stop loss percentage")
    parser.add_argument("--take-profit", type=float, default=0.04, help="Take profit percentage")
    args = parser.parse_args()

    # Determine instruments
    instruments = [args.instruments] if args.instruments else [args.instrument]

    # Create broker
    if args.broker == "oanda":
        access_token = os.environ.get("OANDA_ACCESS_TOKEN", "")
        account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        if not access_token or not account_id:
            print("Set OANDA_ACCESS_TOKEN and OANDA_ACCOUNT_ID environment variables")
            print("Get a free practice account at: https://practice-oanda.com")
            return
        practice = args.paper or not args.live
        broker = OandaBroker(access_token, account_id, practice=practice)
    elif args.broker == "etoro":
        access_token = os.environ.get("ETORO_ACCESS_TOKEN", "")
        if not access_token:
            print("Set ETORO_ACCESS_TOKEN environment variable")
            print("Get an API token at: https://etoro.com/api")
            return
        broker = EtoroBroker(access_token)

    # Create trading engine
    engine = RealTimeTradingEngine(
        broker=broker,
        graph_dir=args.graph,
        instruments=instruments,
        checkpoint=args.checkpoint,
        device=args.device,
        risk_per_trade=args.risk,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )

    engine.run()


if __name__ == "__main__":
    main()
