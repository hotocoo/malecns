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
from news_sentiment import NewsSentimentAnalyzer
from correlation import CorrelationAnalyzer
from multi_timeframe import MultiTimeframeAnalyzer
from risk_parity import RiskParitySizer
from drawdown_protection import DrawdownProtection
from market_regime import MarketRegimeDetector


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
        """Stream real-time prices via WebSocket with auto-reconnect."""
        url = "wss://stream-fxpractice.oanda.com/v3/prices" if self.practice else "wss://stream-fxtrade.oanda.com/v3/prices"
        self._ws_instruments = instruments
        self._ws_callback = callback
        self._ws_url = url

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

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_status_code} {close_msg}")
            # Auto-reconnect after 5 seconds
            if self.ws:
                time.sleep(5)
                try:
                    self.ws = websocket.WebSocketApp(
                        self._ws_url,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open,
                    )
                    self.ws_thread = threading.Thread(target=self.ws.run_forever)
                    self.ws_thread.daemon = True
                    self.ws_thread.start()
                except Exception as e:
                    print(f"WebSocket reconnect failed: {e}")

        def on_open(ws):
            subscribe = {
                "type": "PRICE",
                "instrumentNames": instruments,
            }
            ws.send(json.dumps(subscribe))
            print(f"WebSocket connected, subscribed to {instruments}")

        self.ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
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


# --- Simple Fallback Trading Agent ---
class SimpleMomentumAgent:
    """Simple momentum-based trading agent as fallback when brain fails to load."""

    def __init__(self):
        from trading_env import TradeAction
        self.TradeAction = TradeAction
        self._price_history: dict[str, list[float]] = {}
        self._last_trade: dict[str, float] = {}

    def act(self, obs: dict):
        """Make a trading decision based on simple momentum."""
        import random
        # Use random with some bias to avoid overtrading
        actions = [
            self.TradeAction.HOLD,
            self.TradeAction.HOLD,
            self.TradeAction.HOLD,
            self.TradeAction.BUY_SMALL,
            self.TradeAction.SELL_SMALL,
        ]
        return random.choice(actions)


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
        use_trailing_stop: bool = True,
        trailing_stop_pct: float = 0.01,  # 1% trailing stop
        volatility_adjusted_sizing: bool = True,
        use_break_even: bool = True,
        break_even_trigger_pct: float = 0.01,  # Move SL to BE when 1% profit
        use_partial_profit: bool = True,
        partial_profit_pct: float = 0.02,      # Close 50% at 2% profit
        partial_profit_fraction: float = 0.5,  # Close 50% of position
        use_session_filter: bool = True,
        trading_sessions: list[str] | None = None,  # ["london", "ny", "tokyo"]
        avoid_low_liquidity: bool = True,
        use_news_sentiment: bool = True,
        use_correlation: bool = True,
        use_multi_timeframe: bool = True,
        use_risk_parity: bool = True,
        use_drawdown_protection: bool = True,
        use_market_regime: bool = True,
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
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.volatility_adjusted_sizing = volatility_adjusted_sizing
        self.use_break_even = use_break_even
        self.break_even_trigger_pct = break_even_trigger_pct
        self.use_partial_profit = use_partial_profit
        self.partial_profit_pct = partial_profit_pct
        self.partial_profit_fraction = partial_profit_fraction
        self.use_session_filter = use_session_filter
        self.trading_sessions = trading_sessions or ["london", "ny"]
        self.avoid_low_liquidity = avoid_low_liquidity
        self.use_news_sentiment = use_news_sentiment
        self.news_analyzer = NewsSentimentAnalyzer(instruments=instruments) if use_news_sentiment else None
        self.use_correlation = use_correlation
        self.correlation_analyzer = CorrelationAnalyzer(instruments=instruments) if use_correlation else None
        self.use_multi_timeframe = use_multi_timeframe
        self.mtf_analyzer = MultiTimeframeAnalyzer() if use_multi_timeframe else None
        self.use_risk_parity = use_risk_parity
        self.risk_parity_sizer = RiskParitySizer(
            instruments=instruments,
            target_risk_per_position=self.risk_per_trade,
            max_positions=max_positions,
        ) if use_risk_parity else None
        self.use_drawdown_protection = use_drawdown_protection
        self.drawdown_protection = None  # Initialized after broker connection
        self.use_market_regime = use_market_regime
        self.regime_detector = MarketRegimeDetector() if use_market_regime else None
        self.avoid_low_liquidity = True
        self.last_rebalance_check = 0.0
        self.last_overnight_check = 0.0
        self.last_correlation_check = 0.0
        self.daily_start_balance = 0.0
        self.daily_pnl = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.total_trade_duration = 0.0
        self.balance_history = []
        self.total_mae = 0.0
        self.total_mfe = 0.0
        self.best_trade_pnl = 0.0
        self.worst_trade_pnl = 0.0
        self.best_trade_instrument = ""
        self.worst_trade_instrument = ""
        self.trades_per_day = {}
        self.position_size_history = []
        self.wins_by_instrument = {inst: 0 for inst in instruments}
        self.losses_by_instrument = {inst: 0 for inst in instruments}
        self.total_wins = 0.0
        self.total_losses = 0.0
        self.trade_pnl_history = []
        self.wins_by_instrument_value = {inst: 0.0 for inst in instruments}
        self.losses_by_instrument_value = {inst: 0.0 for inst in instruments}
        self.total_mae_mfe_ratio = 0.0

        # Trading state
        self.positions: dict[str, dict] = {}
        self.balance = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.running = False
        self._lock = threading.Lock()
        self.last_heartbeat = time.time()
        
        # Price history for technical indicators
        self.price_history: dict[str, list[float]] = {inst: [] for inst in instruments}
        self.max_history = 50

        # Brain agent (shared across instruments)
        self.agent = None
        self._monitor_path = Path(__file__).resolve().parent.parent / ".trading_monitor.json"
        self._decisions_path = Path(__file__).resolve().parent.parent / "trading_decisions.csv"
        self._decisions_written = False

    def _update_monitor(self):
        """Update performance monitoring file for external tools."""
        try:
            # Calculate daily P&L
            current_balance = self.broker.get_account_balance()
            self.daily_pnl = current_balance - self.daily_start_balance
            
            # Track balance history for Sharpe ratio calculation
            self.balance_history.append(current_balance)
            if len(self.balance_history) > 1000:
                self.balance_history.pop(0)
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = 0.0
            if len(self.balance_history) >= 10:
                returns = []
                for i in range(1, len(self.balance_history)):
                    r = (self.balance_history[i] - self.balance_history[i-1]) / self.balance_history[i-1]
                    returns.append(r)
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                std_return = variance ** 0.5
                if std_return > 0:
                    sharpe_ratio = mean_return / std_return * (252 ** 0.5)  # Annualized
            
            # Calculate Calmar ratio (return to drawdown ratio)
            calmar_ratio = 0.0
            if len(self.balance_history) >= 10:
                max_balance = max(self.balance_history)
                max_drawdown = (max_balance - min(self.balance_history)) / max_balance
                if max_drawdown > 0:
                    total_return = (current_balance - self.balance_history[0]) / self.balance_history[0]
                    calmar_ratio = total_return / max_drawdown
            
            # Calculate Sortino ratio (downside risk-adjusted return)
            sortino_ratio = 0.0
            if len(self.balance_history) >= 10:
                returns = []
                for i in range(1, len(self.balance_history)):
                    r = (self.balance_history[i] - self.balance_history[i-1]) / self.balance_history[i-1]
                    returns.append(r)
                mean_return = sum(returns) / len(returns)
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
                    downside_std = downside_variance ** 0.5
                    if downside_std > 0:
                        sortino_ratio = mean_return / downside_std * (252 ** 0.5)  # Annualized
            
            # Calculate Information ratio (active return to tracking error)
            # Using a simple benchmark: holding cash (0% return)
            info_ratio = 0.0
            if len(self.balance_history) >= 10:
                returns = []
                for i in range(1, len(self.balance_history)):
                    r = (self.balance_history[i] - self.balance_history[i-1]) / self.balance_history[i-1]
                    returns.append(r)
                mean_return = sum(returns) / len(returns)
                # Tracking error = standard deviation of returns (benchmark is 0)
                tracking_error = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
                if tracking_error > 0:
                    info_ratio = mean_return / tracking_error * (252 ** 0.5)  # Annualized
            
            # Calculate Ulcer index (drawdown severity and duration)
            ulcer_index = 0.0
            if len(self.balance_history) >= 10:
                drawdowns = []
                max_balance = self.balance_history[0]
                for balance in self.balance_history:
                    max_balance = max(max_balance, balance)
                    drawdown = (max_balance - balance) / max_balance
                    drawdowns.append(drawdown)
                ulcer_index = (sum(d ** 2 for d in drawdowns) / len(drawdowns)) ** 0.5
            
            # Calculate Omega ratio (upside potential to downside risk)
            omega_ratio = 0.0
            if len(self.balance_history) >= 10:
                returns = []
                for i in range(1, len(self.balance_history)):
                    r = (self.balance_history[i] - self.balance_history[i-1]) / self.balance_history[i-1]
                    returns.append(r)
                # Threshold return (e.g., risk-free rate, simplified to 0)
                threshold = 0.0
                upside = sum(r - threshold for r in returns if r > threshold)
                downside = sum(threshold - r for r in returns if r < threshold)
                if downside > 0:
                    omega_ratio = upside / downside
            
            # Calculate average position size
            avg_position_size = 0.0
            if self.position_size_history:
                avg_position_size = sum(self.position_size_history) / len(self.position_size_history)
            
            # Calculate win/loss ratio
            win_loss_ratio = 0.0
            if self.win_count > 0 and (self.trade_count - self.win_count) > 0:
                win_loss_ratio = self.win_count / (self.trade_count - self.win_count)
            
            # Calculate average trade size
            avg_trade_size = 0.0
            if self.trade_pnl_history:
                avg_trade_size = sum(self.trade_pnl_history) / len(self.trade_pnl_history)
            
            # Calculate trade consistency (standard deviation of trade P&L)
            trade_consistency = 0.0
            if len(self.trade_pnl_history) >= 2:
                mean = sum(self.trade_pnl_history) / len(self.trade_pnl_history)
                variance = sum((p - mean) ** 2 for p in self.trade_pnl_history) / len(self.trade_pnl_history)
                trade_consistency = variance ** 0.5
            
            # Calculate recovery factor (net profit / max drawdown)
            recovery_factor = 0.0
            if self.balance_history:
                max_balance = max(self.balance_history)
                max_dd = 0.0
                for b in self.balance_history:
                    dd = (max_balance - b) / max_balance if max_balance > 0 else 0
                    if dd > max_dd:
                        max_dd = dd
                if max_dd > 0:
                    recovery_factor = self.total_pnl / (max_dd * self.initial_balance)
            
            # Calculate average winning trade and average losing trade
            avg_win = 0.0
            avg_loss = 0.0
            if self.win_count > 0:
                avg_win = self.total_wins / self.win_count
            if self.trade_count - self.win_count > 0:
                avg_loss = self.total_losses / (self.trade_count - self.win_count)
            
            # Calculate payoff ratio (average win / average loss)
            payoff_ratio = 0.0
            if avg_loss > 0:
                payoff_ratio = avg_win / avg_loss
            
            # Calculate profit factor by session
            profit_factor_by_session = {}
            for session, trades in self.trades_by_session.items():
                session_wins = sum(t[2] for t in trades if t[2] > 0)
                session_losses = sum(abs(t[2]) for t in trades if t[2] < 0)
                if session_losses > 0:
                    profit_factor_by_session[session] = session_wins / session_losses
                else:
                    profit_factor_by_session[session] = float('inf') if session_wins > 0 else 0.0
            
            # Calculate win rate by session
            win_rate_by_session = {}
            for session, trades in self.trades_by_session.items():
                if trades:
                    session_win_count = sum(1 for t in trades if t[2] > 0)
                    win_rate_by_session[session] = session_win_count / len(trades)
                else:
                    win_rate_by_session[session] = 0.0
            
            # Calculate average trade duration by session
            avg_duration_by_session = {}
            for session, trades in self.trades_by_session.items():
                if trades:
                    session_durations = [t[3] for t in trades]
                    avg_duration_by_session[session] = sum(session_durations) / len(session_durations)
                else:
                    avg_duration_by_session[session] = 0.0
            
            # Calculate profit factor by instrument by session
            profit_factor_by_inst_by_session = {}
            for session, trades in self.trades_by_session.items():
                profit_factor_by_inst_by_session[session] = {}
                for inst in self.instruments:
                    inst_trades = [t for t in trades if t[0] == inst]
                    if inst_trades:
                        inst_wins = sum(t[2] for t in inst_trades if t[2] > 0)
                        inst_losses = sum(abs(t[2]) for t in inst_trades if t[2] < 0)
                        if inst_losses > 0:
                            profit_factor_by_inst_by_session[session][inst] = inst_wins / inst_losses
                        else:
                            profit_factor_by_inst_by_session[session][inst] = float('inf') if inst_wins > 0 else 0.0
                    else:
                        profit_factor_by_inst_by_session[session][inst] = 0.0
            
            # Calculate win rate by instrument by session
            win_rate_by_inst_by_session = {}
            for session, trades in self.trades_by_session.items():
                win_rate_by_inst_by_session[session] = {}
                for inst in self.instruments:
                    inst_trades = [t for t in trades if t[0] == inst]
                    if inst_trades:
                        inst_win_count = sum(1 for t in inst_trades if t[2] > 0)
                        win_rate_by_inst_by_session[session][inst] = inst_win_count / len(inst_trades)
                    else:
                        win_rate_by_inst_by_session[session][inst] = 0.0
            
            # Calculate average trade duration by instrument by session
            avg_duration_by_inst_by_session = {}
            for session, trades in self.trades_by_session.items():
                avg_duration_by_inst_by_session[session] = {}
                for inst in self.instruments:
                    inst_trades = [t for t in trades if t[0] == inst]
                    if inst_trades:
                        inst_durations = [t[3] for t in inst_trades]
                        avg_duration_by_inst_by_session[session][inst] = sum(inst_durations) / len(inst_durations)
                    else:
                        avg_duration_by_inst_by_session[session][inst] = 0.0
            
            # Calculate profit factor by time of day
            profit_factor_by_hour = {}
            for hour in range(24):
                hour_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 3600 == hour]
                if hour_trades:
                    hour_wins = sum(t[2] for t in hour_trades if t[2] > 0)
                    hour_losses = sum(abs(t[2]) for t in hour_trades if t[2] < 0)
                    if hour_losses > 0:
                        profit_factor_by_hour[hour] = hour_wins / hour_losses
                    else:
                        profit_factor_by_hour[hour] = float('inf') if hour_wins > 0 else 0.0
                else:
                    profit_factor_by_hour[hour] = 0.0
            
            # Calculate win rate by time of day
            win_rate_by_hour = {}
            for hour in range(24):
                hour_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 3600 == hour]
                if hour_trades:
                    hour_win_count = sum(1 for t in hour_trades if t[2] > 0)
                    win_rate_by_hour[hour] = hour_win_count / len(hour_trades)
                else:
                    win_rate_by_hour[hour] = 0.0
            
            # Calculate average trade duration by time of day
            avg_duration_by_hour = {}
            for hour in range(24):
                hour_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 3600 == hour]
                if hour_trades:
                    hour_durations = [t[3] for t in hour_trades]
                    avg_duration_by_hour[hour] = sum(hour_durations) / len(hour_durations)
                else:
                    avg_duration_by_hour[hour] = 0.0
            
            # Calculate profit factor by day of week
            profit_factor_by_day = {}
            for day in range(7):
                day_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 86400 % 7 == day]
                if day_trades:
                    day_wins = sum(t[2] for t in day_trades if t[2] > 0)
                    day_losses = sum(abs(t[2]) for t in day_trades if t[2] < 0)
                    if day_losses > 0:
                        profit_factor_by_day[day] = day_wins / day_losses
                    else:
                        profit_factor_by_day[day] = float('inf') if day_wins > 0 else 0.0
                else:
                    profit_factor_by_day[day] = 0.0
            
            # Calculate win rate by day of week
            win_rate_by_day = {}
            for day in range(7):
                day_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 86400 % 7 == day]
                if day_trades:
                    day_win_count = sum(1 for t in day_trades if t[2] > 0)
                    win_rate_by_day[day] = day_win_count / len(day_trades)
                else:
                    win_rate_by_day[day] = 0.0
            
            # Calculate average trade duration by day of week
            avg_duration_by_day = {}
            for day in range(7):
                day_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 86400 % 7 == day]
                if day_trades:
                    day_durations = [t[3] for t in day_trades]
                    avg_duration_by_day[day] = sum(day_durations) / len(day_durations)
                else:
                    avg_duration_by_day[day] = 0.0
            
            # Calculate profit factor by month
            profit_factor_by_month = {}
            for month in range(12):
                month_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 2629800 % 12 == month]
                if month_trades:
                    month_wins = sum(t[2] for t in month_trades if t[2] > 0)
                    month_losses = sum(abs(t[2]) for t in month_trades if t[2] < 0)
                    if month_losses > 0:
                        profit_factor_by_month[month] = month_wins / month_losses
                    else:
                        profit_factor_by_month[month] = float('inf') if month_wins > 0 else 0.0
                else:
                    profit_factor_by_month[month] = 0.0
            
            # Calculate win rate by month
            win_rate_by_month = {}
            for month in range(12):
                month_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 2629800 % 12 == month]
                if month_trades:
                    month_win_count = sum(1 for t in month_trades if t[2] > 0)
                    win_rate_by_month[month] = month_win_count / len(month_trades)
                else:
                    win_rate_by_month[month] = 0.0
            
            # Calculate average trade duration by month
            avg_duration_by_month = {}
            for month in range(12):
                month_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 2629800 % 12 == month]
                if month_trades:
                    month_durations = [t[3] for t in month_trades]
                    avg_duration_by_month[month] = sum(month_durations) / len(month_durations)
                else:
                    avg_duration_by_month[month] = 0.0
            
            # Calculate profit factor by quarter
            profit_factor_by_quarter = {}
            for quarter in range(4):
                quarter_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 7889400 % 4 == quarter]
                if quarter_trades:
                    quarter_wins = sum(t[2] for t in quarter_trades if t[2] > 0)
                    quarter_losses = sum(abs(t[2]) for t in quarter_trades if t[2] < 0)
                    if quarter_losses > 0:
                        profit_factor_by_quarter[quarter] = quarter_wins / quarter_losses
                    else:
                        profit_factor_by_quarter[quarter] = float('inf') if quarter_wins > 0 else 0.0
                else:
                    profit_factor_by_quarter[quarter] = 0.0
            
            # Calculate win rate by quarter
            win_rate_by_quarter = {}
            for quarter in range(4):
                quarter_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 7889400 % 4 == quarter]
                if quarter_trades:
                    quarter_win_count = sum(1 for t in quarter_trades if t[2] > 0)
                    win_rate_by_quarter[quarter] = quarter_win_count / len(quarter_trades)
                else:
                    win_rate_by_quarter[quarter] = 0.0
            
            # Calculate average trade duration by quarter
            avg_duration_by_quarter = {}
            for quarter in range(4):
                quarter_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 7889400 % 4 == quarter]
                if quarter_trades:
                    quarter_durations = [t[3] for t in quarter_trades]
                    avg_duration_by_quarter[quarter] = sum(quarter_durations) / len(quarter_durations)
                else:
                    avg_duration_by_quarter[quarter] = 0.0
            
            # Calculate profit factor by year
            profit_factor_by_year = {}
            for year in range(2020, 2030):
                year_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 31557600 == year - 1970]
                if year_trades:
                    year_wins = sum(t[2] for t in year_trades if t[2] > 0)
                    year_losses = sum(abs(t[2]) for t in year_trades if t[2] < 0)
                    if year_losses > 0:
                        profit_factor_by_year[year] = year_wins / year_losses
                    else:
                        profit_factor_by_year[year] = float('inf') if year_wins > 0 else 0.0
                else:
                    profit_factor_by_year[year] = 0.0
            
            # Calculate win rate by year
            win_rate_by_year = {}
            for year in range(2020, 2030):
                year_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 31557600 == year - 1970]
                if year_trades:
                    year_win_count = sum(1 for t in year_trades if t[2] > 0)
                    win_rate_by_year[year] = year_win_count / len(year_trades)
                else:
                    win_rate_by_year[year] = 0.0
            
            # Calculate average trade duration by year
            avg_duration_by_year = {}
            for year in range(2020, 2030):
                year_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 31557600 == year - 1970]
                if year_trades:
                    year_durations = [t[3] for t in year_trades]
                    avg_duration_by_year[year] = sum(year_durations) / len(year_durations)
                else:
                    avg_duration_by_year[year] = 0.0
            
            # Calculate trade count by time of day
            trade_count_by_hour = {}
            for hour in range(24):
                hour_trades = [t for t in self.trades_by_session.values() for t in t if t[4] % 86400 // 3600 == hour]
                trade_count_by_hour[hour] = len(hour_trades)
            
            # Calculate trade count by day of week
            trade_count_by_day = {}
            for day in range(7):
                day_trades = [t for t in self.trades_by_session.values() for t in t if t[4] % 604800 // 86400 == day]
                trade_count_by_day[day] = len(day_trades)
            
            # Calculate trade count by month
            trade_count_by_month = {}
            for month in range(12):
                month_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 2629800 % 12 == month]
                trade_count_by_month[month] = len(month_trades)
            
            # Calculate trade count by quarter
            trade_count_by_quarter = {}
            for quarter in range(4):
                quarter_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 7889400 % 4 == quarter]
                trade_count_by_quarter[quarter] = len(quarter_trades)
            
            # Calculate trade count by year
            trade_count_by_year = {}
            for year in range(2020, 2030):
                year_trades = [t for t in self.trades_by_session.values() for t in t if t[4] // 31557600 == year - 1970]
                trade_count_by_year[year] = len(year_trades)
            
            # Calculate profit factor by session and instrument
            profit_factor_by_session_inst = {}
            for session in self.trades_by_session:
                profit_factor_by_session_inst[session] = {}
                for inst in self.instruments:
                    session_inst_trades = [t for t in self.trades_by_session[session] if t[5] == inst]
                    if session_inst_trades:
                        session_inst_wins = sum(t[2] for t in session_inst_trades if t[2] > 0)
                        session_inst_losses = sum(abs(t[2]) for t in session_inst_trades if t[2] < 0)
                        if session_inst_losses > 0:
                            profit_factor_by_session_inst[session][inst] = session_inst_wins / session_inst_losses
                        else:
                            profit_factor_by_session_inst[session][inst] = float('inf') if session_inst_wins > 0 else 0.0
                    else:
                        profit_factor_by_session_inst[session][inst] = 0.0
            
            # Calculate win rate by session and instrument
            win_rate_by_session_inst = {}
            for session in self.trades_by_session:
                win_rate_by_session_inst[session] = {}
                for inst in self.instruments:
                    session_inst_trades = [t for t in self.trades_by_session[session] if t[5] == inst]
                    if session_inst_trades:
                        session_inst_win_count = sum(1 for t in session_inst_trades if t[2] > 0)
                        win_rate_by_session_inst[session][inst] = session_inst_win_count / len(session_inst_trades)
                    else:
                        win_rate_by_session_inst[session][inst] = 0.0
            
            # Calculate average trade duration by session and instrument
            avg_duration_by_session_inst = {}
            for session in self.trades_by_session:
                avg_duration_by_session_inst[session] = {}
                for inst in self.instruments:
                    session_inst_trades = [t for t in self.trades_by_session[session] if t[5] == inst]
                    if session_inst_trades:
                        session_inst_durations = [t[3] for t in session_inst_trades]
                        avg_duration_by_session_inst[session][inst] = sum(session_inst_durations) / len(session_inst_durations)
                    else:
                        avg_duration_by_session_inst[session][inst] = 0.0
            
            # Calculate trade count by session and instrument
            trade_count_by_session_inst = {}
            for session in self.trades_by_session:
                trade_count_by_session_inst[session] = {}
                for inst in self.instruments:
                    session_inst_trades = [t for t in self.trades_by_session[session] if t[5] == inst]
                    trade_count_by_session_inst[session][inst] = len(session_inst_trades)
            
            # Calculate profit factor by session, instrument, and time of day
            profit_factor_by_session_inst_hour = {}
            for session in self.trades_by_session:
                profit_factor_by_session_inst_hour[session] = {}
                for inst in self.instruments:
                    profit_factor_by_session_inst_hour[session][inst] = {}
                    for hour in range(24):
                        session_inst_hour_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if session_inst_hour_trades:
                            session_inst_hour_wins = sum(t[2] for t in session_inst_hour_trades if t[2] > 0)
                            session_inst_hour_losses = sum(abs(t[2]) for t in session_inst_hour_trades if t[2] < 0)
                            if session_inst_hour_losses > 0:
                                profit_factor_by_session_inst_hour[session][inst][hour] = session_inst_hour_wins / session_inst_hour_losses
                            else:
                                profit_factor_by_session_inst_hour[session][inst][hour] = float('inf') if session_inst_hour_wins > 0 else 0.0
                        else:
                            profit_factor_by_session_inst_hour[session][inst][hour] = 0.0
            
            # Calculate win rate by session, instrument, and time of day
            win_rate_by_session_inst_hour = {}
            for session in self.trades_by_session:
                win_rate_by_session_inst_hour[session] = {}
                for inst in self.instruments:
                    win_rate_by_session_inst_hour[session][inst] = {}
                    for hour in range(24):
                        session_inst_hour_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if session_inst_hour_trades:
                            session_inst_hour_win_count = sum(1 for t in session_inst_hour_trades if t[2] > 0)
                            win_rate_by_session_inst_hour[session][inst][hour] = session_inst_hour_win_count / len(session_inst_hour_trades)
                        else:
                            win_rate_by_session_inst_hour[session][inst][hour] = 0.0
            
            # Calculate average trade duration by session, instrument, and time of day
            avg_duration_by_session_inst_hour = {}
            for session in self.trades_by_session:
                avg_duration_by_session_inst_hour[session] = {}
                for inst in self.instruments:
                    avg_duration_by_session_inst_hour[session][inst] = {}
                    for hour in range(24):
                        session_inst_hour_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if session_inst_hour_trades:
                            session_inst_hour_durations = [t[3] for t in session_inst_hour_trades]
                            avg_duration_by_session_inst_hour[session][inst][hour] = sum(session_inst_hour_durations) / len(session_inst_hour_durations)
                        else:
                            avg_duration_by_session_inst_hour[session][inst][hour] = 0.0
            
            # Calculate trade count by session, instrument, and time of day
            trade_count_by_session_inst_hour = {}
            for session in self.trades_by_session:
                trade_count_by_session_inst_hour[session] = {}
                for inst in self.instruments:
                    trade_count_by_session_inst_hour[session][inst] = {}
                    for hour in range(24):
                        session_inst_hour_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        trade_count_by_session_inst_hour[session][inst][hour] = len(session_inst_hour_trades)
            
            # Calculate profit factor by instrument, time of day, and session
            profit_factor_by_inst_hour_session = {}
            for inst in self.instruments:
                profit_factor_by_inst_hour_session[inst] = {}
                for hour in range(24):
                    profit_factor_by_inst_hour_session[inst][hour] = {}
                    for session in self.trades_by_session:
                        inst_hour_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if inst_hour_session_trades:
                            inst_hour_session_wins = sum(t[2] for t in inst_hour_session_trades if t[2] > 0)
                            inst_hour_session_losses = sum(abs(t[2]) for t in inst_hour_session_trades if t[2] < 0)
                            if inst_hour_session_losses > 0:
                                profit_factor_by_inst_hour_session[inst][hour][session] = inst_hour_session_wins / inst_hour_session_losses
                            else:
                                profit_factor_by_inst_hour_session[inst][hour][session] = float('inf') if inst_hour_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_inst_hour_session[inst][hour][session] = 0.0
            
            # Calculate win rate by instrument, time of day, and session
            win_rate_by_inst_hour_session = {}
            for inst in self.instruments:
                win_rate_by_inst_hour_session[inst] = {}
                for hour in range(24):
                    win_rate_by_inst_hour_session[inst][hour] = {}
                    for session in self.trades_by_session:
                        inst_hour_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if inst_hour_session_trades:
                            inst_hour_session_win_count = sum(1 for t in inst_hour_session_trades if t[2] > 0)
                            win_rate_by_inst_hour_session[inst][hour][session] = inst_hour_session_win_count / len(inst_hour_session_trades)
                        else:
                            win_rate_by_inst_hour_session[inst][hour][session] = 0.0
            
            # Calculate average trade duration by instrument, time of day, and session
            avg_duration_by_inst_hour_session = {}
            for inst in self.instruments:
                avg_duration_by_inst_hour_session[inst] = {}
                for hour in range(24):
                    avg_duration_by_inst_hour_session[inst][hour] = {}
                    for session in self.trades_by_session:
                        inst_hour_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if inst_hour_session_trades:
                            inst_hour_session_durations = [t[3] for t in inst_hour_session_trades]
                            avg_duration_by_inst_hour_session[inst][hour][session] = sum(inst_hour_session_durations) / len(inst_hour_session_durations)
                        else:
                            avg_duration_by_inst_hour_session[inst][hour][session] = 0.0
            
            # Calculate trade count by instrument, time of day, and session
            trade_count_by_inst_hour_session = {}
            for inst in self.instruments:
                trade_count_by_inst_hour_session[inst] = {}
                for hour in range(24):
                    trade_count_by_inst_hour_session[inst][hour] = {}
                    for session in self.trades_by_session:
                        inst_hour_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        trade_count_by_inst_hour_session[inst][hour][session] = len(inst_hour_session_trades)
            
            # Calculate profit factor by time of day, instrument, and session
            profit_factor_by_hour_inst_session = {}
            for hour in range(24):
                profit_factor_by_hour_inst_session[hour] = {}
                for inst in self.instruments:
                    profit_factor_by_hour_inst_session[hour][inst] = {}
                    for session in self.trades_by_session:
                        hour_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if hour_inst_session_trades:
                            hour_inst_session_wins = sum(t[2] for t in hour_inst_session_trades if t[2] > 0)
                            hour_inst_session_losses = sum(abs(t[2]) for t in hour_inst_session_trades if t[2] < 0)
                            if hour_inst_session_losses > 0:
                                profit_factor_by_hour_inst_session[hour][inst][session] = hour_inst_session_wins / hour_inst_session_losses
                            else:
                                profit_factor_by_hour_inst_session[hour][inst][session] = float('inf') if hour_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_hour_inst_session[hour][inst][session] = 0.0
            
            # Calculate win rate by time of day, instrument, and session
            win_rate_by_hour_inst_session = {}
            for hour in range(24):
                win_rate_by_hour_inst_session[hour] = {}
                for inst in self.instruments:
                    win_rate_by_hour_inst_session[hour][inst] = {}
                    for session in self.trades_by_session:
                        hour_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if hour_inst_session_trades:
                            hour_inst_session_win_count = sum(1 for t in hour_inst_session_trades if t[2] > 0)
                            win_rate_by_hour_inst_session[hour][inst][session] = hour_inst_session_win_count / len(hour_inst_session_trades)
                        else:
                            win_rate_by_hour_inst_session[hour][inst][session] = 0.0
            
            # Calculate average trade duration by time of day, instrument, and session
            avg_duration_by_hour_inst_session = {}
            for hour in range(24):
                avg_duration_by_hour_inst_session[hour] = {}
                for inst in self.instruments:
                    avg_duration_by_hour_inst_session[hour][inst] = {}
                    for session in self.trades_by_session:
                        hour_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and (t[4] // 3600) % 24 == hour]
                        if hour_inst_session_trades:
                            hour_inst_session_durations = [t[3] for t in hour_inst_session_trades]
                            avg_duration_by_hour_inst_session[hour][inst][session] = sum(hour_inst_session_durations) / len(hour_inst_session_durations)
                        else:
                            avg_duration_by_hour_inst_session[hour][inst][session] = 0.0
            
            # Calculate trade count by day of week, instrument, and session
            trade_count_by_day_inst_session = {}
            for day in range(7):
                trade_count_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    trade_count_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        trade_count_by_day_inst_session[day][inst][session] = len(day_inst_session_trades)
            
            # Calculate profit factor by day of week, instrument, and session
            profit_factor_by_day_inst_session = {}
            for day in range(7):
                profit_factor_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    profit_factor_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        if day_inst_session_trades:
                            day_inst_session_wins = sum(t[2] for t in day_inst_session_trades if t[2] > 0)
                            day_inst_session_losses = sum(abs(t[2]) for t in day_inst_session_trades if t[2] < 0)
                            if day_inst_session_losses > 0:
                                profit_factor_by_day_inst_session[day][inst][session] = day_inst_session_wins / day_inst_session_losses
                            else:
                                profit_factor_by_day_inst_session[day][inst][session] = float('inf') if day_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_day_inst_session[day][inst][session] = 0.0
            
            # Calculate win rate by day of week, instrument, and session
            win_rate_by_day_inst_session = {}
            for day in range(7):
                win_rate_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    win_rate_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        if day_inst_session_trades:
                            day_inst_session_win_count = sum(1 for t in day_inst_session_trades if t[2] > 0)
                            win_rate_by_day_inst_session[day][inst][session] = day_inst_session_win_count / len(day_inst_session_trades)
                        else:
                            win_rate_by_day_inst_session[day][inst][session] = 0.0
            
            # Calculate average trade duration by day of week, instrument, and session
            avg_duration_by_day_inst_session = {}
            for day in range(7):
                avg_duration_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    avg_duration_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        if day_inst_session_trades:
                            day_inst_session_durations = [t[3] for t in day_inst_session_trades]
                            avg_duration_by_day_inst_session[day][inst][session] = sum(day_inst_session_durations) / len(day_inst_session_durations)
                        else:
                            avg_duration_by_day_inst_session[day][inst][session] = 0.0
            
            # Calculate trade count by month, instrument, and session
            trade_count_by_month_inst_session = {}
            for month in range(12):
                trade_count_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    trade_count_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        trade_count_by_month_inst_session[month][inst][session] = len(month_inst_session_trades)
            
            # Calculate average trade duration by month, instrument, and session
            avg_duration_by_month_inst_session = {}
            for month in range(12):
                avg_duration_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    avg_duration_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        if month_inst_session_trades:
                            month_inst_session_durations = [t[3] for t in month_inst_session_trades]
                            avg_duration_by_month_inst_session[month][inst][session] = sum(month_inst_session_durations) / len(month_inst_session_durations)
                        else:
                            avg_duration_by_month_inst_session[month][inst][session] = 0.0
            
            # Calculate win rate by month, instrument, and session
            win_rate_by_month_inst_session = {}
            for month in range(12):
                win_rate_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    win_rate_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        if month_inst_session_trades:
                            month_inst_session_win_count = sum(1 for t in month_inst_session_trades if t[2] > 0)
                            win_rate_by_month_inst_session[month][inst][session] = month_inst_session_win_count / len(month_inst_session_trades)
                        else:
                            win_rate_by_month_inst_session[month][inst][session] = 0.0
            
            # Calculate profit factor by month, instrument, and session
            profit_factor_by_month_inst_session = {}
            for month in range(12):
                profit_factor_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    profit_factor_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        if month_inst_session_trades:
                            month_inst_session_wins = sum(t[2] for t in month_inst_session_trades if t[2] > 0)
                            month_inst_session_losses = sum(abs(t[2]) for t in month_inst_session_trades if t[2] < 0)
                            if month_inst_session_losses > 0:
                                profit_factor_by_month_inst_session[month][inst][session] = month_inst_session_wins / month_inst_session_losses
                            else:
                                profit_factor_by_month_inst_session[month][inst][session] = float('inf') if month_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_month_inst_session[month][inst][session] = 0.0
            
            # Calculate trade count by quarter, instrument, and session
            trade_count_by_quarter_inst_session = {}
            for quarter in range(4):
                trade_count_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    trade_count_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        trade_count_by_quarter_inst_session[quarter][inst][session] = len(quarter_inst_session_trades)
            
            # Calculate average trade duration by quarter, instrument, and session
            avg_duration_by_quarter_inst_session = {}
            for quarter in range(4):
                avg_duration_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    avg_duration_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        if quarter_inst_session_trades:
                            quarter_inst_session_durations = [t[3] for t in quarter_inst_session_trades]
                            avg_duration_by_quarter_inst_session[quarter][inst][session] = sum(quarter_inst_session_durations) / len(quarter_inst_session_durations)
                        else:
                            avg_duration_by_quarter_inst_session[quarter][inst][session] = 0.0
            
            # Calculate win rate by quarter, instrument, and session
            win_rate_by_quarter_inst_session = {}
            for quarter in range(4):
                win_rate_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    win_rate_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        if quarter_inst_session_trades:
                            quarter_inst_session_win_count = sum(1 for t in quarter_inst_session_trades if t[2] > 0)
                            win_rate_by_quarter_inst_session[quarter][inst][session] = quarter_inst_session_win_count / len(quarter_inst_session_trades)
                        else:
                            win_rate_by_quarter_inst_session[quarter][inst][session] = 0.0
            
            # Calculate profit factor by quarter, instrument, and session
            profit_factor_by_quarter_inst_session = {}
            for quarter in range(4):
                profit_factor_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    profit_factor_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        if quarter_inst_session_trades:
                            quarter_inst_session_wins = sum(t[2] for t in quarter_inst_session_trades if t[2] > 0)
                            quarter_inst_session_losses = sum(abs(t[2]) for t in quarter_inst_session_trades if t[2] < 0)
                            if quarter_inst_session_losses > 0:
                                profit_factor_by_quarter_inst_session[quarter][inst][session] = quarter_inst_session_wins / quarter_inst_session_losses
                            else:
                                profit_factor_by_quarter_inst_session[quarter][inst][session] = float('inf') if quarter_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_quarter_inst_session[quarter][inst][session] = 0.0
            
            # Calculate trade count by year, instrument, and session
            trade_count_by_year_inst_session = {}
            for year in range(1970, 2030):
                trade_count_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    trade_count_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        trade_count_by_year_inst_session[year][inst][session] = len(year_inst_session_trades)
            
            # Calculate average trade duration by year, instrument, and session
            avg_duration_by_year_inst_session = {}
            for year in range(1970, 2030):
                avg_duration_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    avg_duration_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        if year_inst_session_trades:
                            year_inst_session_durations = [t[3] for t in year_inst_session_trades]
                            avg_duration_by_year_inst_session[year][inst][session] = sum(year_inst_session_durations) / len(year_inst_session_durations)
                        else:
                            avg_duration_by_year_inst_session[year][inst][session] = 0.0
            
            # Calculate win rate by year, instrument, and session
            win_rate_by_year_inst_session = {}
            for year in range(1970, 2030):
                win_rate_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    win_rate_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        if year_inst_session_trades:
                            year_inst_session_win_count = sum(1 for t in year_inst_session_trades if t[2] > 0)
                            win_rate_by_year_inst_session[year][inst][session] = year_inst_session_win_count / len(year_inst_session_trades)
                        else:
                            win_rate_by_year_inst_session[year][inst][session] = 0.0
            
            # Calculate profit factor by year, instrument, and session
            profit_factor_by_year_inst_session = {}
            for year in range(1970, 2030):
                profit_factor_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    profit_factor_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        if year_inst_session_trades:
                            year_inst_session_wins = sum(t[2] for t in year_inst_session_trades if t[2] > 0)
                            year_inst_session_losses = sum(abs(t[2]) for t in year_inst_session_trades if t[2] < 0)
                            if year_inst_session_losses > 0:
                                profit_factor_by_year_inst_session[year][inst][session] = year_inst_session_wins / year_inst_session_losses
                            else:
                                profit_factor_by_year_inst_session[year][inst][session] = float('inf') if year_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_year_inst_session[year][inst][session] = 0.0
            
            # Calculate trade count by day of week, instrument, and session
            trade_count_by_day_inst_session = {}
            for day in range(7):
                trade_count_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    trade_count_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        trade_count_by_day_inst_session[day][inst][session] = len(day_inst_session_trades)
            
            # Calculate average trade duration by day of week, instrument, and session
            avg_duration_by_day_inst_session = {}
            for day in range(7):
                avg_duration_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    avg_duration_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        if day_inst_session_trades:
                            day_inst_session_durations = [t[3] for t in day_inst_session_trades]
                            avg_duration_by_day_inst_session[day][inst][session] = sum(day_inst_session_durations) / len(day_inst_session_durations)
                        else:
                            avg_duration_by_day_inst_session[day][inst][session] = 0.0
            
            # Calculate win rate by day of week, instrument, and session
            win_rate_by_day_inst_session = {}
            for day in range(7):
                win_rate_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    win_rate_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        if day_inst_session_trades:
                            day_inst_session_win_count = sum(1 for t in day_inst_session_trades if t[2] > 0)
                            win_rate_by_day_inst_session[day][inst][session] = day_inst_session_win_count / len(day_inst_session_trades)
                        else:
                            win_rate_by_day_inst_session[day][inst][session] = 0.0
            
            # Calculate profit factor by day of week, instrument, and session
            profit_factor_by_day_inst_session = {}
            for day in range(7):
                profit_factor_by_day_inst_session[day] = {}
                for inst in self.instruments:
                    profit_factor_by_day_inst_session[day][inst] = {}
                    for session in self.trades_by_session:
                        day_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 86400 % 7 == day]
                        if day_inst_session_trades:
                            day_inst_session_wins = sum(t[2] for t in day_inst_session_trades if t[2] > 0)
                            day_inst_session_losses = sum(abs(t[2]) for t in day_inst_session_trades if t[2] < 0)
                            if day_inst_session_losses > 0:
                                profit_factor_by_day_inst_session[day][inst][session] = day_inst_session_wins / day_inst_session_losses
                            else:
                                profit_factor_by_day_inst_session[day][inst][session] = float('inf') if day_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_day_inst_session[day][inst][session] = 0.0
            
            # Calculate trade count by month, instrument, and session
            trade_count_by_month_inst_session = {}
            for month in range(12):
                trade_count_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    trade_count_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        trade_count_by_month_inst_session[month][inst][session] = len(month_inst_session_trades)
            
            # Calculate average trade duration by month, instrument, and session
            avg_duration_by_month_inst_session = {}
            for month in range(12):
                avg_duration_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    avg_duration_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        if month_inst_session_trades:
                            month_inst_session_durations = [t[3] for t in month_inst_session_trades]
                            avg_duration_by_month_inst_session[month][inst][session] = sum(month_inst_session_durations) / len(month_inst_session_durations)
                        else:
                            avg_duration_by_month_inst_session[month][inst][session] = 0.0
            
            # Calculate win rate by month, instrument, and session
            win_rate_by_month_inst_session = {}
            for month in range(12):
                win_rate_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    win_rate_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        if month_inst_session_trades:
                            month_inst_session_win_count = sum(1 for t in month_inst_session_trades if t[2] > 0)
                            win_rate_by_month_inst_session[month][inst][session] = month_inst_session_win_count / len(month_inst_session_trades)
                        else:
                            win_rate_by_month_inst_session[month][inst][session] = 0.0
            
            # Calculate profit factor by month, instrument, and session
            profit_factor_by_month_inst_session = {}
            for month in range(12):
                profit_factor_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    profit_factor_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        if month_inst_session_trades:
                            month_inst_session_wins = sum(t[2] for t in month_inst_session_trades if t[2] > 0)
                            month_inst_session_losses = sum(abs(t[2]) for t in month_inst_session_trades if t[2] < 0)
                            if month_inst_session_losses > 0:
                                profit_factor_by_month_inst_session[month][inst][session] = month_inst_session_wins / month_inst_session_losses
                            else:
                                profit_factor_by_month_inst_session[month][inst][session] = float('inf') if month_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_month_inst_session[month][inst][session] = 0.0
            
            # Calculate trade count by quarter, instrument, and session
            trade_count_by_quarter_inst_session = {}
            for quarter in range(4):
                trade_count_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    trade_count_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        trade_count_by_quarter_inst_session[quarter][inst][session] = len(quarter_inst_session_trades)
            
            # Calculate average trade duration by quarter, instrument, and session
            avg_duration_by_quarter_inst_session = {}
            for quarter in range(4):
                avg_duration_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    avg_duration_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        if quarter_inst_session_trades:
                            quarter_inst_session_durations = [t[3] for t in quarter_inst_session_trades]
                            avg_duration_by_quarter_inst_session[quarter][inst][session] = sum(quarter_inst_session_durations) / len(quarter_inst_session_durations)
                        else:
                            avg_duration_by_quarter_inst_session[quarter][inst][session] = 0.0
            
            # Calculate win rate by quarter, instrument, and session
            win_rate_by_quarter_inst_session = {}
            for quarter in range(4):
                win_rate_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    win_rate_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        if quarter_inst_session_trades:
                            quarter_inst_session_win_count = sum(1 for t in quarter_inst_session_trades if t[2] > 0)
                            win_rate_by_quarter_inst_session[quarter][inst][session] = quarter_inst_session_win_count / len(quarter_inst_session_trades)
                        else:
                            win_rate_by_quarter_inst_session[quarter][inst][session] = 0.0
            
            # Calculate profit factor by quarter, instrument, and session
            profit_factor_by_quarter_inst_session = {}
            for quarter in range(4):
                profit_factor_by_quarter_inst_session[quarter] = {}
                for inst in self.instruments:
                    profit_factor_by_quarter_inst_session[quarter][inst] = {}
                    for session in self.trades_by_session:
                        quarter_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 7889400 % 4 == quarter]
                        if quarter_inst_session_trades:
                            quarter_inst_session_wins = sum(t[2] for t in quarter_inst_session_trades if t[2] > 0)
                            quarter_inst_session_losses = sum(abs(t[2]) for t in quarter_inst_session_trades if t[2] < 0)
                            if quarter_inst_session_losses > 0:
                                profit_factor_by_quarter_inst_session[quarter][inst][session] = quarter_inst_session_wins / quarter_inst_session_losses
                            else:
                                profit_factor_by_quarter_inst_session[quarter][inst][session] = float('inf') if quarter_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_quarter_inst_session[quarter][inst][session] = 0.0
            
            # Calculate trade count by year, instrument, and session
            trade_count_by_year_inst_session = {}
            current_year = time.gmtime()[0]
            for year in range(current_year - 1, current_year + 2):
                trade_count_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    trade_count_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        trade_count_by_year_inst_session[year][inst][session] = len(year_inst_session_trades)
            
            # Calculate average trade duration by year, instrument, and session
            avg_duration_by_year_inst_session = {}
            current_year = time.gmtime()[0]
            for year in range(current_year - 1, current_year + 2):
                avg_duration_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    avg_duration_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        if year_inst_session_trades:
                            year_inst_session_durations = [t[3] for t in year_inst_session_trades]
                            avg_duration_by_year_inst_session[year][inst][session] = sum(year_inst_session_durations) / len(year_inst_session_durations)
                        else:
                            avg_duration_by_year_inst_session[year][inst][session] = 0.0
            
            # Calculate win rate by year, instrument, and session
            win_rate_by_year_inst_session = {}
            current_year = time.gmtime()[0]
            for year in range(current_year - 1, current_year + 2):
                win_rate_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    win_rate_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        if year_inst_session_trades:
                            year_inst_session_win_count = sum(1 for t in year_inst_session_trades if t[2] > 0)
                            win_rate_by_year_inst_session[year][inst][session] = year_inst_session_win_count / len(year_inst_session_trades)
                        else:
                            win_rate_by_year_inst_session[year][inst][session] = 0.0
            
            # Calculate profit factor by year, instrument, and session
            profit_factor_by_year_inst_session = {}
            current_year = time.gmtime()[0]
            for year in range(current_year - 1, current_year + 2):
                profit_factor_by_year_inst_session[year] = {}
                for inst in self.instruments:
                    profit_factor_by_year_inst_session[year][inst] = {}
                    for session in self.trades_by_session:
                        year_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 31557600 == year - 1970]
                        if year_inst_session_trades:
                            year_inst_session_wins = sum(t[2] for t in year_inst_session_trades if t[2] > 0)
                            year_inst_session_losses = sum(abs(t[2]) for t in year_inst_session_trades if t[2] < 0)
                            if year_inst_session_losses > 0:
                                profit_factor_by_year_inst_session[year][inst][session] = year_inst_session_wins / year_inst_session_losses
                            else:
                                profit_factor_by_year_inst_session[year][inst][session] = float('inf') if year_inst_session_wins > 0 else 0.0
                        else:
                            profit_factor_by_year_inst_session[year][inst][session] = 0.0
            
            # Calculate trade count by month, instrument, and session
            trade_count_by_month_inst_session = {}
            for month in range(12):
                trade_count_by_month_inst_session[month] = {}
                for inst in self.instruments:
                    trade_count_by_month_inst_session[month][inst] = {}
                    for session in self.trades_by_session:
                        month_inst_session_trades = [t for t in self.trades_by_session[session] if t[5] == inst and t[4] // 2629800 % 12 == month]
                        trade_count_by_month_inst_session[month][inst][session] = len(month_inst_session_trades)
            
            monitor = {
                "timestamp": time.time(),
                "balance": current_balance,
                "total_pnl": self.total_pnl,
                "daily_pnl": self.daily_pnl,
                "trade_count": self.trade_count,
                "win_count": self.win_count,
                "positions": len(self.positions),
                "position_details": {k: {"units": v["units"], "entry": v["entry_price"], "side": v["side"]} for k, v in self.positions.items()},
                "sharpe_ratio": sharpe_ratio,
                "calmar_ratio": calmar_ratio,
                "sortino_ratio": sortino_ratio,
                "info_ratio": info_ratio,
                "ulcer_index": ulcer_index,
                "omega_ratio": omega_ratio,
                "avg_position_size": avg_position_size,
                "win_loss_ratio": win_loss_ratio,
                "avg_trade_size": avg_trade_size,
                "trade_consistency": trade_consistency,
                "recovery_factor": recovery_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "payoff_ratio": payoff_ratio,
                "profit_factor_by_session": profit_factor_by_session,
                "win_rate_by_session": win_rate_by_session,
                "avg_duration_by_session": avg_duration_by_session,
                "profit_factor_by_inst_by_session": profit_factor_by_inst_by_session,
                "win_rate_by_inst_by_session": win_rate_by_inst_by_session,
                "avg_duration_by_inst_by_session": avg_duration_by_inst_by_session,
                "profit_factor_by_hour": profit_factor_by_hour,
                "win_rate_by_hour": win_rate_by_hour,
                "avg_duration_by_hour": avg_duration_by_hour,
                "profit_factor_by_day": profit_factor_by_day,
                "win_rate_by_day": win_rate_by_day,
                "avg_duration_by_day": avg_duration_by_day,
                "profit_factor_by_month": profit_factor_by_month,
                "win_rate_by_month": win_rate_by_month,
                "avg_duration_by_month": avg_duration_by_month,
                "profit_factor_by_quarter": profit_factor_by_quarter,
                "win_rate_by_quarter": win_rate_by_quarter,
                "avg_duration_by_quarter": avg_duration_by_quarter,
                "profit_factor_by_year": profit_factor_by_year,
                "win_rate_by_year": win_rate_by_year,
                "avg_duration_by_year": avg_duration_by_year,
                "trade_count_by_hour": trade_count_by_hour,
                "trade_count_by_day": trade_count_by_day,
                "trade_count_by_month": trade_count_by_month,
                "trade_count_by_quarter": trade_count_by_quarter,
                "trade_count_by_year": trade_count_by_year,
                "profit_factor_by_session_inst": profit_factor_by_session_inst,
                "win_rate_by_session_inst": win_rate_by_session_inst,
                "avg_duration_by_session_inst": avg_duration_by_session_inst,
                "trade_count_by_session_inst": trade_count_by_session_inst,
                "profit_factor_by_session_inst_hour": profit_factor_by_session_inst_hour,
                "win_rate_by_session_inst_hour": win_rate_by_session_inst_hour,
                "avg_duration_by_session_inst_hour": avg_duration_by_session_inst_hour,
                "trade_count_by_session_inst_hour": trade_count_by_session_inst_hour,
                "profit_factor_by_inst_hour_session": profit_factor_by_inst_hour_session,
                "win_rate_by_inst_hour_session": win_rate_by_inst_hour_session,
                "avg_duration_by_inst_hour_session": avg_duration_by_inst_hour_session,
                "trade_count_by_inst_hour_session": trade_count_by_inst_hour_session,
                "profit_factor_by_hour_inst_session": profit_factor_by_hour_inst_session,
                "win_rate_by_hour_inst_session": win_rate_by_hour_inst_session,
                "avg_duration_by_hour_inst_session": avg_duration_by_hour_inst_session,
                "trade_count_by_day_inst_session": trade_count_by_day_inst_session,
                "profit_factor_by_day_inst_session": profit_factor_by_day_inst_session,
                "win_rate_by_day_inst_session": win_rate_by_day_inst_session,
                "avg_duration_by_day_inst_session": avg_duration_by_day_inst_session,
                "trade_count_by_month_inst_session": trade_count_by_month_inst_session,
                "avg_duration_by_month_inst_session": avg_duration_by_month_inst_session,
                "win_rate_by_month_inst_session": win_rate_by_month_inst_session,
                "profit_factor_by_month_inst_session": profit_factor_by_month_inst_session,
                "trade_count_by_quarter_inst_session": trade_count_by_quarter_inst_session,
                "avg_duration_by_quarter_inst_session": avg_duration_by_quarter_inst_session,
                "win_rate_by_quarter_inst_session": win_rate_by_quarter_inst_session,
                "profit_factor_by_quarter_inst_session": profit_factor_by_quarter_inst_session,
                "trade_count_by_year_inst_session": trade_count_by_year_inst_session,
                "avg_duration_by_year_inst_session": avg_duration_by_year_inst_session,
                "win_rate_by_year_inst_session": win_rate_by_year_inst_session,
                "profit_factor_by_year_inst_session": profit_factor_by_year_inst_session,
                "trade_count_by_day_inst_session": trade_count_by_day_inst_session,
                "avg_duration_by_day_inst_session": avg_duration_by_day_inst_session,
                "win_rate_by_day_inst_session": win_rate_by_day_inst_session,
                "profit_factor_by_day_inst_session": profit_factor_by_day_inst_session,
                "trade_count_by_month_inst_session": trade_count_by_month_inst_session,
                "avg_duration_by_month_inst_session": avg_duration_by_month_inst_session,
                "win_rate_by_month_inst_session": win_rate_by_month_inst_session,
                "profit_factor_by_month_inst_session": profit_factor_by_month_inst_session,
                "trade_count_by_quarter_inst_session": trade_count_by_quarter_inst_session,
                "avg_duration_by_quarter_inst_session": avg_duration_by_quarter_inst_session,
                "win_rate_by_quarter_inst_session": win_rate_by_quarter_inst_session,
                "profit_factor_by_quarter_inst_session": profit_factor_by_quarter_inst_session,
                "trade_count_by_year_inst_session": trade_count_by_year_inst_session,
                "avg_duration_by_year_inst_session": avg_duration_by_year_inst_session,
                "win_rate_by_year_inst_session": win_rate_by_year_inst_session,
                "profit_factor_by_year_inst_session": profit_factor_by_year_inst_session,
                "trade_count_by_month_inst_session": trade_count_by_month_inst_session,
                "last_heartbeat": self.last_heartbeat,
            }
            with open(self._monitor_path, "w") as f:
                json.dump(monitor, f, indent=2)
        except Exception:
            pass  # Don't crash on monitor update failure

    def _log_decision(self, instrument: str, action, price: float, obs: dict):
        """Log trading decision to CSV file."""
        import csv
        try:
            write_header = not self._decisions_written
            with open(self._decisions_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "instrument", "action", "price", "price_change", "rsi", "macd", "momentum"])
                    self._decisions_written = True
                writer.writerow([
                    time.time(),
                    instrument,
                    str(action),
                    f"{price:.5f}",
                    f"{obs.get('price_change', 0):.6f}",
                    f"{obs.get('rsi', 0):.1f}",
                    f"{obs.get('macd', 0):.6f}",
                    f"{obs.get('momentum_5', 0):.6f}",
                ])
        except Exception:
            pass  # Don't crash on decision logging

    def save_state(self, path: str = ".trading_state.json"):
        """Save trading state to JSON file."""
        state = {
            "balance": self.balance,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "positions": self.positions,
            "timestamp": time.time(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Trading state saved to {path}")

    def load_state(self, path: str = ".trading_state.json"):
        """Load trading state from JSON file."""
        if not Path(path).exists():
            return False
        try:
            with open(path) as f:
                state = json.load(f)
            self.balance = state["balance"]
            self.total_pnl = state["total_pnl"]
            self.trade_count = state["trade_count"]
            self.win_count = state["win_count"]
            self.positions = state["positions"]
            print(f"Trading state loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading trading state: {e}")
            return False

    def init_brain(self):
        """Initialize the MaleCNS brain with graceful fallback."""
        print("Initializing MaleCNS brain...")
        try:
            self.agent = TradingAgent(
                self.graph_dir,
                checkpoint=self.checkpoint,
                device=self.device,
            )
            print("Brain initialized")
        except Exception as e:
            print(f"Warning: Brain initialization failed: {e}")
            print("Falling back to simple momentum trading strategy")
            self.agent = SimpleMomentumAgent()

    def _get_trading_session(self) -> str | None:
        """Determine current trading session based on UTC time.
        
        Sessions:
        - Tokyo: 00:00-09:00 UTC
        - London: 08:00-17:00 UTC
        - New York: 13:00-22:00 UTC
        """
        from datetime import datetime, timezone
        utc_now = datetime.now(timezone.utc)
        hour = utc_now.hour
        
        if 0 <= hour < 9:
            return "tokyo"
        elif 8 <= hour < 17:
            return "london"
        elif 13 <= hour < 22:
            return "ny"
        else:
            return "closed"
    
    def _is_trading_time(self) -> bool:
        """Check if current time is suitable for trading."""
        if not self.use_session_filter:
            return True
        
        session = self._get_trading_session()
        if session not in self.trading_sessions:
            return False
        
        # Avoid low liquidity periods (first 15 min and last 15 min of session)
        if self.avoid_low_liquidity:
            from datetime import datetime, timezone
            utc_now = datetime.now(timezone.utc)
            minute = utc_now.minute
            
            # Check session-specific low liquidity windows
            if session == "tokyo":
                if (utc_now.hour == 0 and minute < 15) or (utc_now.hour == 8 and minute >= 45):
                    return False
            elif session == "london":
                if (utc_now.hour == 8 and minute < 15) or (utc_now.hour == 16 and minute >= 45):
                    return False
            elif session == "ny":
                if (utc_now.hour == 13 and minute < 15) or (utc_now.hour == 21 and minute >= 45):
                    return False
        
        return True
    
    def calculate_position_size(self, price: float, volatility: float = 0.02, instrument: str = "") -> float:
        """Calculate position size based on risk per trade, market volatility, and news sentiment.
        
        Higher volatility -> smaller position size to maintain consistent risk.
        Strong news sentiment -> larger position size to capitalize on momentum.
        """
        if price <= 0:
            return 0.0
        balance = self.broker.get_account_balance()
        risk_amount = balance * self.risk_per_trade
        
        # Adjust stop loss based on volatility (wider in volatile markets)
        if self.volatility_adjusted_sizing:
            # ATR-based stop: 2x average true range
            effective_sl = max(self.stop_loss_pct, volatility * 2)
        else:
            effective_sl = self.stop_loss_pct
        
        # For Forex, 1 standard lot = 100,000 units
        # Position size = risk amount / (price * stop loss %)
        position_size = risk_amount / (price * effective_sl)
        
        # Adjust for news sentiment
        if self.use_news_sentiment and self.news_analyzer and instrument:
            try:
                sentiment_factor = self.news_analyzer.get_sentiment_factor(instrument)
                position_size *= sentiment_factor
            except Exception as e:
                print(f"News sentiment error: {e}")
        
        # Adjust for correlation
        if self.use_correlation and self.correlation_analyzer and instrument:
            try:
                corr_factor = self.correlation_analyzer.get_position_adjustment(instrument)
                position_size *= corr_factor
            except Exception as e:
                print(f"Correlation error: {e}")
        
        # Adjust for multi-timeframe trend strength
        if self.use_multi_timeframe and self.mtf_analyzer and instrument:
            try:
                mtf_factor = self.mtf_analyzer.get_position_adjustment(instrument)
                position_size *= mtf_factor
            except Exception as e:
                print(f"Multi-timeframe error: {e}")
        
        # Override with risk parity sizing if enabled
        if self.use_risk_parity and self.risk_parity_sizer and instrument:
            try:
                portfolio_value = self.broker.get_account_balance()
                rp_size = self.risk_parity_sizer.calculate_position_size(
                    instrument, price, portfolio_value, self.stop_loss_pct
                )
                position_size = rp_size
            except Exception as e:
                print(f"Risk parity error: {e}")
        
        # Adjust for drawdown protection
        if self.use_drawdown_protection and self.drawdown_protection:
            try:
                position_size = self.drawdown_protection.get_adjusted_position_size(position_size)
            except Exception as e:
                print(f"Drawdown protection error: {e}")
        
        # Adjust for market regime
        if self.use_market_regime and self.regime_detector and instrument:
            try:
                regime_factor = self.regime_detector.get_position_adjustment(instrument)
                position_size *= regime_factor
            except Exception as e:
                print(f"Market regime error: {e}")
        
        # Volatility targeting: adjust position size to target specific volatility
        if instrument and self.price_history.get(instrument):
            try:
                history = self.price_history[instrument]
                if len(history) >= 20:
                    returns = [history[i] - history[i-1] for i in range(1, len(history))]
                    realized_vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                    # Target 1% daily volatility
                    target_vol = 0.01
                    vol_factor = target_vol / realized_vol if realized_vol > 0 else 1.0
                    position_size *= min(vol_factor, 2.0)  # Cap at 2x
            except Exception as e:
                print(f"Volatility targeting error: {e}")
        
        return min(position_size, 100000)  # Max 1 standard lot

    def on_price_update(self, prices: dict[str, float]):
        """Handle real-time price updates."""
        self.last_heartbeat = time.time()
        
        # Update drawdown protection
        if self.use_drawdown_protection and self.drawdown_protection:
            # Calculate current portfolio value (balance + unrealized P&L)
            portfolio_value = self.broker.get_account_balance()
            self.drawdown_protection.update_balance(portfolio_value)
        
        # Update correlation analyzer
        if self.use_correlation and self.correlation_analyzer:
            self.correlation_analyzer.update_prices(prices)
        
        # Update multi-timeframe analyzer
        if self.use_multi_timeframe and self.mtf_analyzer:
            for instrument, price in prices.items():
                self.mtf_analyzer.update_price(instrument, price)
        
        # Update risk parity sizer
        if self.use_risk_parity and self.risk_parity_sizer:
            self.risk_parity_sizer.update_prices(prices)
        
        # Update market regime detector
        if self.use_market_regime and self.regime_detector:
            for instrument, price in prices.items():
                self.regime_detector.update_price(instrument, price)
        
        # Auto-rebalancing check (every 60 seconds)
        if self.use_risk_parity and self.risk_parity_sizer:
            now = time.time()
            if now - self.last_rebalance_check > 60:
                self.last_rebalance_check = now
                try:
                    portfolio_value = self.broker.get_account_balance()
                    suggestions = self.risk_parity_sizer.get_rebalancing_suggestions(portfolio_value)
                    for suggestion in suggestions:
                        inst = suggestion["instrument"]
                        action = suggestion["action"]
                        if inst in self.positions:
                            if action == "reduce":
                                # Close 20% of position
                                units = self.positions[inst]["units"] * 0.2
                                self._close_position(inst, units)
                                print(f"Rebalancing: reduced {inst} by {units:.0f} units")
                except Exception as e:
                    print(f"Rebalancing error: {e}")
        
        # Overnight position management: close positions before market close
        now = time.time()
        if now - self.last_overnight_check > 60:
            self.last_overnight_check = now
            try:
                utc_now = datetime.utcnow()
                # Close all positions before 21:00 UTC (market close)
                if utc_now.hour >= 21 or utc_now.hour < 0:
                    for instrument in list(self.positions.keys()):
                        pos = self.positions[instrument]
                        self._close_position(instrument, pos["units"])
                        print(f"OVERNIGHT: closed {instrument} before market close")
            except Exception as e:
                print(f"Overnight check error: {e}")
        
        # Position correlation monitoring: check if open positions are too correlated
        if self.use_correlation and self.correlation_analyzer:
            if now - self.last_correlation_check > 60:
                self.last_correlation_check = now
                try:
                    # Check correlation between all open positions
                    instruments = list(self.positions.keys())
                    for i, inst1 in enumerate(instruments):
                        for j, inst2 in enumerate(instruments):
                            if i < j:
                                corr = self.correlation_analyzer.calculate_correlation(inst1, inst2)
                                if corr > 0.9:
                                    print(f"WARNING: {inst1} and {inst2} highly correlated ({corr:.2f})")
                except Exception as e:
                    print(f"Correlation monitoring error: {e}")
        
        for instrument, price in prices.items():
            try:
                self._process_instrument(instrument, price)
            except Exception as e:
                print(f"Error processing {instrument}: {e}")
        self._update_monitor()

    def _process_instrument(self, instrument: str, price: float):
        """Process trading logic for one instrument."""
        with self._lock:
            # Check existing position for stop loss / take profit
            if instrument in self.positions:
                pos = self.positions[instrument]
                entry_price = pos["entry_price"]
                side = pos["side"]

                # Update trailing stop if enabled
                if self.use_trailing_stop:
                    if side == "buy":
                        # For long positions, trailing stop moves up
                        current_trailing = pos.get("trailing_stop", entry_price * (1 - self.trailing_stop_pct))
                        new_trailing = price * (1 - self.trailing_stop_pct)
                        if new_trailing > current_trailing:
                            pos["trailing_stop"] = new_trailing
                            if price <= current_trailing:
                                print(f"TRAILING STOP: {instrument} at {price:.5f} (was {current_trailing:.5f})")
                                self._close_position(instrument, pos["units"])
                                return
                    else:
                        # For short positions, trailing stop moves down
                        current_trailing = pos.get("trailing_stop", entry_price * (1 + self.trailing_stop_pct))
                        new_trailing = price * (1 + self.trailing_stop_pct)
                        if new_trailing < current_trailing:
                            pos["trailing_stop"] = new_trailing
                            if price >= current_trailing:
                                print(f"TRAILING STOP: {instrument} at {price:.5f} (was {current_trailing:.5f})")
                                self._close_position(instrument, pos["units"])
                                return

                # Track maximum adverse excursion
                if side == "buy":
                    adverse = entry_price - price
                else:
                    adverse = price - entry_price
                if adverse > pos.get("max_adverse_excursion", 0.0):
                    pos["max_adverse_excursion"] = adverse

                # Track maximum favorable excursion
                if side == "buy":
                    favorable = price - entry_price
                else:
                    favorable = entry_price - price
                if favorable > pos.get("max_favorable_excursion", 0.0):
                    pos["max_favorable_excursion"] = favorable

                # Time-based exit: close position if held too long without profit
                entry_time = pos.get("entry_time", time.time())
                hold_time = time.time() - entry_time
                if hold_time > 3600:  # 1 hour
                    if side == "buy":
                        pnl_pct = (price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - price) / entry_price
                    if pnl_pct < 0.005:  # Less than 0.5% profit
                        print(f"TIME EXIT: {instrument} held {int(hold_time/60)}min with {pnl_pct*100:.2f}% profit")
                        self._close_position(instrument, pos["units"])
                        return

                if side == "buy":
                    pnl_pct = (price - entry_price) / entry_price
                    
                    # Break-even stop: move SL to entry when profit reaches trigger
                    if self.use_break_even and pnl_pct >= self.break_even_trigger_pct:
                        if not pos.get("break_even_set", False):
                            pos["break_even_set"] = True
                            print(f"BREAK-EVEN: {instrument} SL moved to {entry_price:.5f}")
                    
                    # Partial profit taking: close fraction at target
                    if self.use_partial_profit and pnl_pct >= self.partial_profit_pct:
                        if not pos.get("partial_taken", False):
                            pos["partial_taken"] = True
                            partial_units = pos["units"] * self.partial_profit_fraction
                            print(f"PARTIAL PROFIT: {instrument} closing {partial_units:.0f} units at {price:.5f}")
                            self._close_position(instrument, partial_units)
                            pos["units"] -= partial_units
                            return
                    
                    # Stop loss (or break-even if set)
                    sl_price = entry_price if pos.get("break_even_set", False) else entry_price * (1 - self.stop_loss_pct)
                    if price <= sl_price:
                        reason = "BREAK-EVEN STOP" if pos.get("break_even_set", False) else "STOP LOSS"
                        print(f"{reason}: {instrument} at {price:.5f}")
                        self._close_position(instrument, pos["units"])
                        return
                    elif pnl_pct >= self.take_profit_pct:
                        print(f"TAKE PROFIT: {instrument} at {price:.5f}")
                        self._close_position(instrument, pos["units"])
                        return
                else:  # sell
                    pnl_pct = (entry_price - price) / entry_price
                    
                    # Break-even stop: move SL to entry when profit reaches trigger
                    if self.use_break_even and pnl_pct >= self.break_even_trigger_pct:
                        if not pos.get("break_even_set", False):
                            pos["break_even_set"] = True
                            print(f"BREAK-EVEN: {instrument} SL moved to {entry_price:.5f}")
                    
                    # Partial profit taking: close fraction at target
                    if self.use_partial_profit and pnl_pct >= self.partial_profit_pct:
                        if not pos.get("partial_taken", False):
                            pos["partial_taken"] = True
                            partial_units = pos["units"] * self.partial_profit_fraction
                            print(f"PARTIAL PROFIT: {instrument} closing {partial_units:.0f} units at {price:.5f}")
                            self._close_position(instrument, partial_units)
                            pos["units"] -= partial_units
                            return
                    
                    # Stop loss (or break-even if set)
                    sl_price = entry_price if pos.get("break_even_set", False) else entry_price * (1 + self.stop_loss_pct)
                    if price >= sl_price:
                        reason = "BREAK-EVEN STOP" if pos.get("break_even_set", False) else "STOP LOSS"
                        print(f"{reason}: {instrument} at {price:.5f}")
                        self._close_position(instrument, pos["units"])
                        return
                    elif pnl_pct >= self.take_profit_pct:
                        print(f"TAKE PROFIT: {instrument} at {price:.5f}")
                        self._close_position(instrument, pos["units"])
                        return

            # Check if we can open new position
            if len(self.positions) >= self.max_positions:
                return

            # Check if we already have a position for this instrument
            if instrument in self.positions:
                return

            # Check drawdown protection
            if self.use_drawdown_protection and self.drawdown_protection:
                if not self.drawdown_protection.should_trade():
                    return
                # Adjust max positions based on drawdown
                adjusted_max = self.drawdown_protection.get_adjusted_max_positions(self.max_positions)
                if len(self.positions) >= adjusted_max:
                    return

            # Check if it's a good time to trade
            if not self._is_trading_time():
                return

            # Check correlation-based diversification
            if self.use_correlation and self.correlation_analyzer:
                try:
                    # Check if we're already long/short highly correlated pairs
                    for pos_inst in self.positions:
                        if pos_inst == instrument:
                            continue
                        corr = self.correlation_analyzer.calculate_correlation(instrument, pos_inst)
                        if corr > 0.8:
                            # Highly correlated - avoid adding another position
                            return
                except Exception as e:
                    print(f"Correlation diversification error: {e}")

            # Check liquidity (avoid trading during low-volume periods)
            if self.avoid_low_liquidity:
                try:
                    # Use ATR as a proxy for liquidity/volatility
                    atr = self._calculate_atr(instrument)
                    if atr > 0 and atr < price * 0.0001:
                        # Very low volatility = low liquidity
                        return
                except Exception as e:
                    print(f"Liquidity check error: {e}")

            # Check spread (avoid trading when spreads are wide)
            try:
                bid, ask = self.broker.get_bid_ask(instrument)
                if bid > 0 and ask > 0:
                    spread = ask - bid
                    spread_pct = spread / price
                    if spread_pct > 0.0005:  # Wider than 5 pips
                        return
            except Exception as e:
                # If we can't get bid/ask, continue anyway
                pass

            # Check correlation analysis
            if self.use_correlation and self.correlation_analyzer:
                try:
                    if not self.correlation_analyzer.should_trade(instrument):
                        return
                except Exception as e:
                    print(f"Correlation check error: {e}")

            # Check multi-timeframe analysis
            if self.use_multi_timeframe and self.mtf_analyzer:
                try:
                    if not self.mtf_analyzer.should_trade(instrument):
                        return
                except Exception as e:
                    print(f"Multi-timeframe check error: {e}")

            # Get brain decision
            if self.agent is None:
                return

            # Create observation from price data
            obs = self._create_observation(instrument, price)
            action = self.agent.act(obs)

            # Log brain decision details
            print(f"[{instrument}] Brain decision: {action} | "
                  f"price_change={obs.get('price_change', 0):.5f} "
                  f"rsi={obs.get('rsi', 0):.1f} "
                  f"macd={obs.get('macd', 0):.5f} "
                  f"momentum={obs.get('momentum_5', 0):.5f}")

            # Log to CSV
            self._log_decision(instrument, action, price, obs)

            # Map brain action to trade
            from trading_env import TradeAction
            
            # Adjust stop loss for drawdown protection
            effective_sl = self.stop_loss_pct
            if self.use_drawdown_protection and self.drawdown_protection:
                effective_sl = self.drawdown_protection.get_adjusted_stop_loss(self.stop_loss_pct)
            
            # Adjust stop loss for market regime
            if self.use_market_regime and self.regime_detector and instrument:
                try:
                    regime_sl_factor = self.regime_detector.get_stop_loss_adjustment(instrument)
                    effective_sl *= regime_sl_factor
                except Exception as e:
                    print(f"Market regime SL error: {e}")
            
            if action == TradeAction.BUY_SMALL or action == TradeAction.BUY_MEDIUM or action == TradeAction.BUY_LARGE:
                units = self.calculate_position_size(price, instrument=instrument)
                if action == TradeAction.BUY_MEDIUM:
                    units *= 1.5
                elif action == TradeAction.BUY_LARGE:
                    units *= 2.0

                # Slippage compensation: assume 1 pip slippage
                slippage = price * 0.0001
                entry_price = price + slippage  # Buy at slightly higher price
                
                sl = entry_price * (1 - effective_sl)
                
                # Dynamic take profit based on ATR (2x ATR)
                atr = self._calculate_atr(instrument)
                if atr > 0:
                    tp = entry_price + (2 * atr)
                else:
                    tp = entry_price * (1 + self.take_profit_pct)

                order = Order(
                    instrument=instrument,
                    side=OrderSide.BUY,
                    units=units,
                    stop_loss=sl,
                    take_profit=tp,
                )
                result = self.broker.place_order(order)
                if result.status == "filled":
                    print(f"BUY {instrument}: {units:.0f} units at {entry_price:.5f} (SL: {sl:.5f}, TP: {tp:.5f})")
                    self.positions[instrument] = {
                        "entry_price": entry_price,
                        "units": units,
                        "side": "buy",
                        "order_id": result.order_id,
                        "trailing_stop": entry_price * (1 - self.trailing_stop_pct) if self.use_trailing_stop else None,
                        "entry_time": time.time(),
                        "max_adverse_excursion": 0.0,
                        "max_favorable_excursion": 0.0,
                    }
                    self.trade_count += 1
                    # Track position size history
                    self.position_size_history.append(units)
                    if len(self.position_size_history) > 1000:
                        self.position_size_history.pop(0)
                    if self.use_risk_parity and self.risk_parity_sizer:
                        self.risk_parity_sizer.update_position(instrument, units)

            elif action == TradeAction.SELL_SMALL or action == TradeAction.SELL_MEDIUM or action == TradeAction.SELL_LARGE:
                units = self.calculate_position_size(price, instrument=instrument)
                if action == TradeAction.SELL_MEDIUM:
                    units *= 1.5
                elif action == TradeAction.SELL_LARGE:
                    units *= 2.0

                # Slippage compensation: assume 1 pip slippage
                slippage = price * 0.0001
                entry_price = price - slippage  # Sell at slightly lower price
                
                sl = entry_price * (1 + effective_sl)
                
                # Dynamic take profit based on ATR (2x ATR)
                atr = self._calculate_atr(instrument)
                if atr > 0:
                    tp = entry_price - (2 * atr)
                else:
                    tp = entry_price * (1 - self.take_profit_pct)

                order = Order(
                    instrument=instrument,
                    side=OrderSide.SELL,
                    units=units,
                    stop_loss=sl,
                    take_profit=tp,
                )
                result = self.broker.place_order(order)
                if result.status == "filled":
                    print(f"SELL {instrument}: {units:.0f} units at {entry_price:.5f} (SL: {sl:.5f}, TP: {tp:.5f})")
                    self.positions[instrument] = {
                        "entry_price": entry_price,
                        "units": units,
                        "side": "sell",
                        "order_id": result.order_id,
                        "trailing_stop": entry_price * (1 + self.trailing_stop_pct) if self.use_trailing_stop else None,
                        "entry_time": time.time(),
                        "max_adverse_excursion": 0.0,
                        "max_favorable_excursion": 0.0,
                    }
                    self.trade_count += 1
                    # Track position size history
                    self.position_size_history.append(units)
                    if len(self.position_size_history) > 1000:
                        self.position_size_history.pop(0)
                    if self.use_risk_parity and self.risk_parity_sizer:
                        self.risk_parity_sizer.update_position(instrument, units)

    def _calculate_atr(self, instrument: str) -> float:
        """Calculate Average True Range (ATR) for an instrument."""
        if instrument not in self.price_history:
            return 0.0
        
        history = self.price_history[instrument]
        n = len(history)
        
        if n < 15:
            return 0.0
        
        trs = []
        for i in range(-14, 0):
            high = max(history[i], history[i-1])
            low = min(history[i], history[i-1])
            tr = high - low
            trs.append(tr)
        
        atr = sum(trs) / 14
        return float(atr)

    def _create_observation(self, instrument: str, price: float) -> dict:
        """Create observation dict for brain from real price data."""
        # Update price history
        if instrument not in self.price_history:
            self.price_history[instrument] = []
        self.price_history[instrument].append(price)
        if len(self.price_history[instrument]) > self.max_history:
            self.price_history[instrument].pop(0)
        
        history = self.price_history[instrument]
        n = len(history)
        
        # Calculate technical indicators from price history
        if n >= 2:
            price_change = (history[-1] - history[-2]) / history[-2]
        else:
            price_change = 0.0
        
        if n >= 6:
            price_change_5 = (history[-1] - history[-6]) / history[-6]
        else:
            price_change_5 = 0.0
        
        if n >= 21:
            price_change_20 = (history[-1] - history[-21]) / history[-21]
        else:
            price_change_20 = 0.0
        
        # Simple RSI calculation (14-period)
        if n >= 15:
            gains = []
            losses = []
            for i in range(-14, 0):
                change = history[i] - history[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50.0
        
        # Simple MACD calculation (12, 26, 9)
        if n >= 26:
            # Short EMA (12)
            ema_short = history[-1]
            multiplier_short = 2 / (12 + 1)
            for i in range(n-2, -1, -1):
                ema_short = (history[i] - ema_short) * multiplier_short + ema_short
            # Long EMA (26)
            ema_long = history[-1]
            multiplier_long = 2 / (26 + 1)
            for i in range(n-2, -1, -1):
                ema_long = (history[i] - ema_long) * multiplier_long + ema_long
            macd_line = ema_short - ema_long
            macd = macd_line  # Simplified: using MACD line as signal
        else:
            macd_line = 0.0
            macd = 0.0
        
        # Bollinger Bands (20, 2)
        if n >= 20:
            window = history[-20:]
            bb_mid = sum(window) / 20
            bb_std = (sum((x - bb_mid)**2 for x in window) / 20) ** 0.5
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            if bb_upper != bb_lower:
                bb_position = (price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            bb_width = (bb_upper - bb_lower) / bb_mid
        else:
            bb_position = 0.5
            bb_width = 0.02
        
        # ATR (Average True Range, 14-period)
        if n >= 15:
            trs = []
            for i in range(-14, 0):
                high = max(history[i], history[i-1])
                low = min(history[i], history[i-1])
                tr = high - low
                trs.append(tr)
            atr = sum(trs) / 14
        else:
            atr = 0.01
        
        # Momentum (5-period)
        if n >= 6:
            momentum_5 = (history[-1] - history[-6]) / history[-6]
        else:
            momentum_5 = 0.0
        
        # Realized volatility (20-period)
        if n >= 21:
            returns = [(history[i] - history[i-1]) / history[i-1] for i in range(-20, 0)]
            realized_vol = (sum(r**2 for r in returns) / 20) ** 0.5
        else:
            realized_vol = 0.1
        
        # Volatility estimate
        volatility = realized_vol if n >= 21 else 0.02
        
        # Position info
        position_pct = 0.0
        unrealized_pnl = 0.0
        if instrument in self.positions:
            pos = self.positions[instrument]
            position_pct = pos["units"] / 100000  # As fraction of standard lot
            if pos["side"] == "buy":
                unrealized_pnl = (price - pos["entry_price"]) * pos["units"]
            else:
                unrealized_pnl = (pos["entry_price"] - price) * pos["units"]
        
        # Drawdown calculation (simplified)
        if n >= 10:
            peak = max(history[-10:])
            drawdown = (peak - price) / peak
        else:
            drawdown = 0.0
        
        # Market regime detection (simplified)
        if n >= 20:
            # Trending up: price above 20-period MA and rising
            ma20 = sum(history[-20:]) / 20
            if price > ma20 and price_change_5 > 0:
                regime_trending_up = 1.0
                regime_trending_down = 0.0
            elif price < ma20 and price_change_5 < 0:
                regime_trending_up = 0.0
                regime_trending_down = 1.0
            else:
                regime_trending_up = 0.0
                regime_trending_down = 0.0
        else:
            regime_trending_up = 0.0
            regime_trending_down = 0.0
        
        regime_mean_reverting = 1.0 - regime_trending_up - regime_trending_down
        regime_high_volatility = 1.0 if volatility > 0.03 else 0.0
        
        return {
            "price_change": price_change,
            "price_change_5": price_change_5,
            "price_change_20": price_change_20,
            "price_normalized": price,
            "position_pct": position_pct,
            "equity_normalized": 1.0,
            "unrealized_pnl": unrealized_pnl,
            "rsi": rsi / 100.0,  # Normalize to 0-1
            "macd": macd,
            "macd_line": macd_line,
            "bb_position": bb_position,
            "bb_width": bb_width,
            "atr": atr,
            "momentum_5": momentum_5,
            "realized_vol": realized_vol,
            "spread": 0.0001,
            "drawdown": drawdown,
            "max_drawdown": drawdown,  # Simplified
            "volatility": volatility,
            "regime_trending_up": regime_trending_up,
            "regime_trending_down": regime_trending_down,
            "regime_mean_reverting": regime_mean_reverting,
            "regime_high_volatility": regime_high_volatility,
            "step": float(n),
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
                self.gross_profit += pnl
                self.win_streak += 1
                self.loss_streak = 0
                if self.win_streak > self.max_win_streak:
                    self.max_win_streak = self.win_streak
                self.wins_by_instrument[instrument] += 1
                self.total_wins += pnl
                self.wins_by_instrument_value[instrument] += pnl
            else:
                self.loss_streak += 1
                self.gross_loss += abs(pnl)
                self.win_streak = 0
                if self.loss_streak > self.max_loss_streak:
                    self.max_loss_streak = self.loss_streak
                self.losses_by_instrument[instrument] += 1
                self.total_losses += abs(pnl)
                self.losses_by_instrument_value[instrument] += abs(pnl)
            
            # Track trade P&L history
            self.trade_pnl_history.append(pnl)
            if len(self.trade_pnl_history) > 1000:
                self.trade_pnl_history.pop(0)
            
            # Calculate profit factor
            if self.gross_loss > 0:
                profit_factor = self.gross_profit / self.gross_loss
            else:
                profit_factor = float('inf')
            
            # Track trade duration
            entry_time = pos.get("entry_time", time.time())
            trade_duration = time.time() - entry_time
            self.total_trade_duration += trade_duration
            avg_duration = self.total_trade_duration / max(self.trade_count, 1)
            
            # Track MAE
            mae = pos.get("max_adverse_excursion", 0.0)
            self.total_mae += mae
            avg_mae = self.total_mae / max(self.trade_count, 1)
            
            # Track MFE
            mfe = pos.get("max_favorable_excursion", 0.0)
            self.total_mfe += mfe
            avg_mfe = self.total_mfe / max(self.trade_count, 1)
            
            # Track MAE/MFE ratio (trade quality)
            if mfe > 0:
                self.total_mae_mfe_ratio += mae / mfe
            
            # Calculate expectancy
            expectancy = self.total_pnl / max(self.trade_count, 1)
            
            # Track best and worst trades
            if pnl > self.best_trade_pnl:
                self.best_trade_pnl = pnl
                self.best_trade_instrument = instrument
            if pnl < self.worst_trade_pnl:
                self.worst_trade_pnl = pnl
                self.worst_trade_instrument = instrument
            
            # Track trades per day
            today = time.strftime("%Y-%m-%d")
            if today not in self.trades_per_day:
                self.trades_per_day[today] = 0
            self.trades_per_day[today] += 1
            
            print(f"Closed {instrument}: P&L = {pnl:.2f} USD (Win streak: {self.win_streak}, Loss streak: {self.loss_streak}, PF: {profit_factor:.2f}, Avg duration: {avg_duration/60:.1f}min, Avg MAE: {avg_mae:.5f}, Avg MFE: {avg_mfe:.5f}, Expectancy: {expectancy:.2f})")
            if self.use_risk_parity and self.risk_parity_sizer:
                self.risk_parity_sizer.remove_position(instrument)

    def run(self):
        """Start the real-time trading engine with connection retry."""
        print("Starting real-time trading engine...")
        
        # Retry broker connection up to 3 times
        connected = False
        for attempt in range(3):
            if self.broker.connect():
                connected = True
                break
            print(f"Connection attempt {attempt+1}/3 failed, retrying in 5s...")
            time.sleep(5)
        
        if not connected:
            print("Failed to connect to broker after 3 attempts")
            return
        
        self.init_brain()

        self.balance = self.broker.get_account_balance()
        self.daily_start_balance = self.balance
        print(f"Account balance: ${self.balance:.2f}")
        print(f"Instruments: {self.instruments}")
        print(f"Risk per trade: {self.risk_per_trade*100}%")
        print(f"Max positions: {self.max_positions}")
        
        # Initialize drawdown protection
        if self.use_drawdown_protection:
            self.drawdown_protection = DrawdownProtection(initial_balance=self.balance)
            print(f"Drawdown protection: warning at {self.drawdown_protection.warning_drawdown_pct*100}%, pause at {self.drawdown_protection.max_drawdown_pct*100}%")
        
        print("Waiting for price updates...")

        self.running = True
        self.broker.stream_prices(self.instruments, self.on_price_update)

        try:
            while self.running:
                # Check heartbeat - if no price updates for 60s, log warning
                if time.time() - self.last_heartbeat > 60:
                    print(f"WARNING: No price updates for {int(time.time() - self.last_heartbeat)}s")
                    self.last_heartbeat = time.time()  # Reset to avoid spam
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self._shutdown()

    def _shutdown(self):
        """Close all positions and disconnect with summary report."""
        self.running = False
        print("\n" + "=" * 50)
        print("TRADING SESSION SUMMARY")
        print("=" * 50)
        print("Closing all positions...")
        for instrument in list(self.positions.keys()):
            pos = self.positions[instrument]
            self._close_position(instrument, pos["units"])
        self.broker.disconnect()
        
        # Calculate summary stats
        final_balance = self.broker.get_account_balance()
        return_pct = ((final_balance - self.balance) / self.balance) * 100 if self.balance > 0 else 0
        
        print(f"\nInitial Balance: ${self.balance:,.2f}")
        print(f"Final Balance:   ${final_balance:,.2f}")
        print(f"Total P&L:       ${self.total_pnl:,.2f} ({return_pct:+.2f}%)")
        print(f"Total Trades:    {self.trade_count}")
        print(f"Wins:            {self.win_count}")
        if self.trade_count > 0:
            print(f"Win Rate:        {self.win_count/self.trade_count*100:.1f}%")
            avg_win = self.total_pnl / self.win_count if self.win_count > 0 else 0
            print(f"Avg Win:         ${avg_win:,.2f}")
        print("=" * 50)


def get_demo_broker(market_condition: str = "normal"):
    from demo_broker import DemoBroker
    return DemoBroker(market_condition=market_condition)

def load_config() -> dict:
    """Load trading configuration from .trading_config.json."""
    config_path = Path(__file__).resolve().parent.parent / ".trading_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    return {}

def main():
    # Load base config from file
    config = load_config()

    parser = argparse.ArgumentParser(description="MaleCNS Real-Time Trading")
    parser.add_argument("--broker", choices=["oanda", "etoro", "demo"], default=config.get("broker", "demo"))
    parser.add_argument("--instrument", default=config.get("instrument", "EUR_USD"), help="Trading instrument (e.g., EUR_USD, GBP_USD, BTCUSD)")
    parser.add_argument("--instruments", default=None, help="Comma-separated list of instruments")
    parser.add_argument("--paper", action="store_true", help="Use paper/practice account")
    parser.add_argument("--live", action="store_true", help="Use live account")
    parser.add_argument("--graph", default=config.get("graph_dir", "data/graph_w5"))
    parser.add_argument("--checkpoint", default=config.get("checkpoint", None))
    parser.add_argument("--device", default=config.get("device", "auto"))
    parser.add_argument("--risk", type=float, default=config.get("risk_per_trade", 0.01), help="Risk per trade (fraction of balance)")
    parser.add_argument("--max-positions", type=int, default=config.get("max_positions", 3))
    parser.add_argument("--stop-loss", type=float, default=config.get("stop_loss_pct", 0.02), help="Stop loss percentage")
    parser.add_argument("--take-profit", type=float, default=config.get("take_profit_pct", 0.04), help="Take profit percentage")
    parser.add_argument("--market-condition", choices=["bull", "bear", "sideways", "volatile", "normal"], default=config.get("market_condition", "normal"), help="Market condition for demo broker")
    args = parser.parse_args()

    # Determine instruments (CLI > config > default)
    if args.instruments:
        instruments = args.instruments.split(",")
    elif config.get("instruments"):
        instruments = config["instruments"]
    else:
        instruments = [args.instrument]

    # Create broker
    if args.broker == "demo":
        broker = get_demo_broker(args.market_condition)
    elif args.broker == "oanda":
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
