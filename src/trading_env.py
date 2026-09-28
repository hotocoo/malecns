"""Realistic trading environment for the malecns connectome.

Simulates a financial market with:
- Geometric Brownian Motion (GBM) price dynamics with stochastic volatility
- Order book with bid/ask spread dynamics
- Multiple market regimes (trending, mean-reverting, high-volatility)
- Realistic transaction costs (spread, slippage, commissions)
- Technical indicators: RSI, MACD, Bollinger Bands, ATR
- Performance metrics: Sharpe, Sortino, Calmar, max drawdown
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class Regime(IntEnum):
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    MEAN_REVERTING = 2
    HIGH_VOLATILITY = 3


class TradeAction(IntEnum):
    HOLD = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    BUY_LARGE = 3
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6


def _default_regime_params():
    return {
        Regime.TRENDING_UP: {"drift": 0.001, "volatility": 0.015},
        Regime.TRENDING_DOWN: {"drift": -0.0008, "volatility": 0.018},
        Regime.MEAN_REVERTING: {"drift": 0.0, "volatility": 0.012},
        Regime.HIGH_VOLATILITY: {"drift": 0.0001, "volatility": 0.04},
    }


@dataclass(frozen=True)
class TradingConfig:
    initial_price: float = 100.0
    drift: float = 0.0002
    volatility: float = 0.02
    vol_of_vol: float = 0.3
    mean_reversion_speed: float = 0.05
    dt: float = 1.0 / 252.0
    base_spread: float = 0.001
    spread_volatility: float = 0.5
    order_size: float = 100.0
    max_position: float = 1000.0
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0002
    regime_switch_prob: float = 0.01
    regime_params: dict = field(default_factory=_default_regime_params)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    initial_capital: float = 100000.0
    risk_free_rate: float = 0.02
    steps_per_episode: int = 252
    max_drawdown_limit: float = 0.3
    sharpe_window: int = 21
    drawdown_penalty: float = 2.0
    volatility_penalty: float = 1.0
    turnover_penalty: float = 0.1


class TradingEnvironment:
    """A realistic trading environment with market microstructure."""

    def __init__(self, config=None, seed=None):
        self.config = config or TradingConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        cfg = self.config
        self.price = cfg.initial_price
        self.volatility = cfg.volatility
        self.prev_price = self.price
        self.high_price = self.price
        self.low_price = self.price
        self.regime = Regime.MEAN_REVERTING
        self.regime_params = cfg.regime_params[self.regime]
        self.position = 0.0
        self.cash = cfg.initial_capital
        self.peak_equity = cfg.initial_capital
        self.max_drawdown = 0.0
        self.prices = [self.price]
        self.highs = [self.price]
        self.lows = [self.price]
        self.closes = [self.price]
        self.returns = []
        self.spread = cfg.base_spread
        self.update_spread()
        self.step_count = 0
        self.total_trades = 0
        self.total_commissions = 0.0
        self.total_slippage = 0.0
        self.done = False
        self.info = {}
        return self._get_observation()

    def update_spread(self):
        cfg = self.config
        spread_target = cfg.base_spread * (1 + 0.5 * (self.volatility / cfg.volatility - 1))
        self.spread += 0.1 * (spread_target - self.spread) + cfg.spread_volatility * self.spread * self.rng.normal()
        self.spread = max(cfg.base_spread * 0.5, self.spread)

    def _simulate_price_step(self):
        cfg = self.config
        params = self.regime_params
        vol_mean = params["volatility"]
        vol_drift = cfg.mean_reversion_speed * (vol_mean - self.volatility)
        vol_diff = cfg.vol_of_vol * self.volatility * self.rng.normal()
        self.volatility = max(0.001, self.volatility + vol_drift * cfg.dt + vol_diff * math.sqrt(cfg.dt))
        drift = params["drift"]
        dW = self.rng.normal() * math.sqrt(cfg.dt)
        log_return = (drift - 0.5 * self.volatility ** 2) * cfg.dt + self.volatility * dW
        if self.regime == Regime.MEAN_REVERTING:
            mean_reversion = 0.02 * (cfg.initial_price - self.price) / cfg.initial_price
            log_return += mean_reversion * cfg.dt
        self.prev_price = self.price
        self.price = self.price * math.exp(log_return)
        self.high_price = max(self.prev_price, self.price)
        self.low_price = min(self.prev_price, self.price)
        self.prices.append(self.price)
        self.highs.append(self.high_price)
        self.lows.append(self.low_price)
        self.closes.append(self.price)
        if self.rng.random() < cfg.regime_switch_prob:
            self.regime = Regime(self.rng.integers(0, 4))
            self.regime_params = cfg.regime_params[self.regime]
        self.update_spread()

    def _compute_indicators(self):
        cfg = self.config
        lookback = max(cfg.rsi_period, cfg.macd_slow, cfg.bb_period, cfg.atr_period) * 2
        closes = np.array(self.closes[-lookback:])
        highs = np.array(self.highs[-len(closes):])
        lows = np.array(self.lows[-len(closes):])
        indicators = {}
        if len(closes) > cfg.rsi_period:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-cfg.rsi_period:])
            avg_loss = np.mean(losses[-cfg.rsi_period:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                indicators["rsi"] = 100 - 100 / (1 + rs)
            else:
                indicators["rsi"] = 100.0
        else:
            indicators["rsi"] = 50.0
        if len(closes) > cfg.macd_slow:
            ema_fast = self._ema(closes, cfg.macd_fast)
            ema_slow = self._ema(closes, cfg.macd_slow)
            macd_line = ema_fast - ema_slow
            indicators["macd"] = macd_line
            indicators["macd_line"] = macd_line
        else:
            indicators["macd"] = 0.0
            indicators["macd_line"] = 0.0
        if len(closes) > cfg.bb_period:
            bb_mid = np.mean(closes[-cfg.bb_period:])
            bb_std = np.std(closes[-cfg.bb_period:])
            bb_upper = bb_mid + cfg.bb_std * bb_std
            bb_lower = bb_mid - cfg.bb_std * bb_std
            if bb_upper != bb_lower:
                indicators["bb_position"] = (self.price - bb_lower) / (bb_upper - bb_lower)
            else:
                indicators["bb_position"] = 0.5
            indicators["bb_width"] = (bb_upper - bb_lower) / bb_mid
        else:
            indicators["bb_position"] = 0.5
            indicators["bb_width"] = 0.02
        if len(closes) > cfg.atr_period:
            true_ranges = []
            for i in range(1, len(closes)):
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                true_ranges.append(tr)
            indicators["atr"] = np.mean(true_ranges[-cfg.atr_period:]) / self.price
        else:
            indicators["atr"] = 0.01
        if len(closes) > 5:
            indicators["momentum_5"] = (closes[-1] / closes[-6] - 1)
        else:
            indicators["momentum_5"] = 0.0
        if len(self.returns) > 1:
            indicators["realized_vol"] = np.std(self.returns[-21:]) * math.sqrt(252)
        else:
            indicators["realized_vol"] = self.volatility * math.sqrt(252)
        return indicators

    def _ema(self, data, period):
        if len(data) < period:
            return float(data[-1])
        multiplier = 2 / (period + 1)
        ema = float(data[0])
        for value in data[1:]:
            ema = (float(value) - ema) * multiplier + ema
        return ema

    def _get_observation(self):
        indicators = self._compute_indicators()
        equity = self.cash + self.position * self.price
        return {
            "price": self.price,
            "price_change": self.price / self.prev_price - 1 if self.prev_price > 0 else 0.0,
            "price_change_5": self.price / self.prices[-6] - 1 if len(self.prices) > 5 else 0.0,
            "price_change_20": self.price / self.prices[-21] - 1 if len(self.prices) > 20 else 0.0,
            "price_normalized": self.price / self.config.initial_price,
            "position": self.position,
            "position_pct": self.position / self.config.max_position,
            "equity": equity,
            "equity_normalized": equity / self.config.initial_capital,
            "unrealized_pnl": self.position * (self.price - self.prev_price),
            "rsi": (indicators["rsi"] - 50) / 50,
            "macd": indicators["macd"] / self.price,
            "macd_line": indicators["macd_line"] / self.price,
            "bb_position": indicators["bb_position"] - 0.5,
            "bb_width": indicators["bb_width"],
            "atr": indicators["atr"],
            "momentum_5": indicators["momentum_5"],
            "realized_vol": indicators["realized_vol"],
            "spread": self.spread,
            "bid": self.price * (1 - self.spread / 2),
            "ask": self.price * (1 + self.spread / 2),
            "drawdown": (self.peak_equity - equity) / self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "regime_trending_up": 1.0 if self.regime == Regime.TRENDING_UP else 0.0,
            "regime_trending_down": 1.0 if self.regime == Regime.TRENDING_DOWN else 0.0,
            "regime_mean_reverting": 1.0 if self.regime == Regime.MEAN_REVERTING else 0.0,
            "regime_high_volatility": 1.0 if self.regime == Regime.HIGH_VOLATILITY else 0.0,
            "step": self.step_count / self.config.steps_per_episode,
            "day_of_year": self.step_count % 365,
        }

    def step(self, action):
        cfg = self.config
        self.step_count += 1
        trade_value = 0.0
        if action != TradeAction.HOLD:
            if action in (TradeAction.BUY_SMALL, TradeAction.SELL_SMALL):
                size_mult = 1.0
            elif action in (TradeAction.BUY_MEDIUM, TradeAction.SELL_MEDIUM):
                size_mult = 2.0
            else:
                size_mult = 3.0
            order_size = cfg.order_size * size_mult
            if action in (TradeAction.BUY_SMALL, TradeAction.BUY_MEDIUM, TradeAction.BUY_LARGE):
                trade_price = self.price * (1 + self.spread / 2)
                cost = order_size * trade_price
                commission = cost * cfg.commission_rate
                slippage = cost * cfg.slippage_rate
                if self.cash >= cost + commission + slippage:
                    self.cash -= cost + commission + slippage
                    self.position += order_size
                    trade_value = cost
                    self.total_commissions += commission
                    self.total_slippage += slippage
                    self.total_trades += 1
            elif action in (TradeAction.SELL_SMALL, TradeAction.SELL_MEDIUM, TradeAction.SELL_LARGE):
                trade_price = self.price * (1 - self.spread / 2)
                proceeds = order_size * trade_price
                commission = proceeds * cfg.commission_rate
                slippage = proceeds * cfg.slippage_rate
                if self.position >= order_size:
                    self.cash += proceeds - commission - slippage
                    self.position -= order_size
                    trade_value = proceeds
                    self.total_commissions += commission
                    self.total_slippage += slippage
                    self.total_trades += 1
        self._simulate_price_step()
        equity = self.cash + self.position * self.price
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)
        if self.step_count > 1:
            prev_equity = self.cash + self.position * self.prev_price
            daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            self.returns.append(daily_return)
        reward = self._compute_reward(equity, drawdown, trade_value)
        done = False
        if self.step_count >= cfg.steps_per_episode:
            done = True
        if drawdown >= cfg.max_drawdown_limit:
            done = True
            self.info["termination_reason"] = "max_drawdown"
        if equity <= 0:
            done = True
            self.info["termination_reason"] = "bankrupt"
        self.info["equity"] = equity
        self.info["position"] = self.position
        self.info["cash"] = self.cash
        self.info["trade_value"] = trade_value
        self.info["total_trades"] = self.total_trades
        self.info["total_commissions"] = self.total_commissions
        self.info["total_slippage"] = self.total_slippage
        self.info["drawdown"] = drawdown
        self.info["max_drawdown"] = self.max_drawdown
        self.info["regime"] = self.regime.name
        if done:
            self.info["final_equity"] = equity
            self.info["total_return"] = (equity - cfg.initial_capital) / cfg.initial_capital
            if len(self.returns) > 1:
                self.info["sharpe_ratio"] = self._sharpe_ratio()
                self.info["sortino_ratio"] = self._sortino_ratio()
                self.info["calmar_ratio"] = self._calmar_ratio()
        return self._get_observation(), reward, done, self.info

    def _compute_reward(self, equity, drawdown, trade_value):
        cfg = self.config
        prev_equity = self.cash + self.position * self.prev_price
        pnl = equity - prev_equity
        pnl_reward = pnl / cfg.initial_capital
        drawdown_penalty = -cfg.drawdown_penalty * drawdown
        vol_penalty = -cfg.volatility_penalty * (self.volatility ** 2)
        turnover_penalty = -cfg.turnover_penalty * (trade_value / cfg.initial_capital)
        sharpe_bonus = 0.0
        if len(self.returns) >= cfg.sharpe_window:
            window_returns = self.returns[-cfg.sharpe_window:]
            std = np.std(window_returns)
            if std > 0:
                sharpe = np.mean(window_returns) / std * math.sqrt(252)
                sharpe_bonus = 0.01 * sharpe
        return pnl_reward + drawdown_penalty + vol_penalty + turnover_penalty + sharpe_bonus

    def _sharpe_ratio(self):
        if len(self.returns) < 2:
            return 0.0
        excess = np.array(self.returns) - self.config.risk_free_rate / 252
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(252))

    def _sortino_ratio(self):
        if len(self.returns) < 2:
            return 0.0
        excess = np.array(self.returns) - self.config.risk_free_rate / 252
        downside = excess[excess < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return float(np.mean(excess) / np.std(downside) * math.sqrt(252))

    def _calmar_ratio(self):
        if self.max_drawdown == 0:
            return 0.0
        equity = self.cash + self.position * self.price
        total_return = (equity - self.config.initial_capital) / self.config.initial_capital
        return float(total_return / self.max_drawdown)

    def get_performance_metrics(self):
        equity = self.cash + self.position * self.price
        total_return = (equity - self.config.initial_capital) / self.config.initial_capital
        return {
            "final_equity": equity,
            "total_return": total_return,
            "annualized_return": total_return * (252 / max(self.step_count, 1)),
            "sharpe_ratio": self._sharpe_ratio(),
            "sortino_ratio": self._sortino_ratio(),
            "calmar_ratio": self._calmar_ratio(),
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "total_commissions": self.total_commissions,
            "total_slippage": self.total_slippage,
            "win_rate": self._win_rate(),
            "profit_factor": self._profit_factor(),
            "avg_trade_pnl": self._avg_trade_pnl(),
            "best_trade": self._best_trade(),
            "worst_trade": self._worst_trade(),
        }

    def _win_rate(self):
        if len(self.returns) == 0:
            return 0.0
        wins = sum(1 for r in self.returns if r > 0)
        return wins / len(self.returns)

    def _profit_factor(self):
        if len(self.returns) == 0:
            return 0.0
        gains = sum(r for r in self.returns if r > 0)
        losses = abs(sum(r for r in self.returns if r < 0))
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    def _avg_trade_pnl(self):
        if len(self.returns) == 0:
            return 0.0
        return float(np.mean(self.returns))

    def _best_trade(self):
        if len(self.returns) == 0:
            return 0.0
        return float(np.max(self.returns))

    def _worst_trade(self):
        if len(self.returns) == 0:
            return 0.0
        return float(np.min(self.returns))
