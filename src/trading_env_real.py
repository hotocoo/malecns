"""Real-world trading environment using actual historical market data."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RealTradingConfig:
    data_file: str = "data/real_market_data.csv"
    assets: tuple = ("SPY", "QQQ", "IWM")
    initial_capital: float = 100000.0
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0002
    max_position_pct: float = 0.35
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_drawdown_limit: float = 0.25
    risk_free_rate: float = 0.02
    steps_per_episode: int = 252
    train_start: int = 0
    train_end: int = 1200
    test_start: int = 1200
    test_end: int = 1687


class RealTradingEnvironment:
    """Trading environment using real historical market data."""

    def __init__(self, config=None, seed=None):
        self.config = config or RealTradingConfig()
        self.rng = np.random.default_rng(seed)
        self._load_data()
        self.reset()

    def _load_data(self):
        """Load real historical market data."""
        data_file = self.config.data_file
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Market data not found: {data_file}")

        self.dates = []
        self.prices = {asset: [] for asset in self.config.assets}
        self.volumes = {asset: [] for asset in self.config.assets}

        with open(data_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Skip 3 header rows
        for row in rows[3:]:
            if not row or not row[0]:
                continue
            date = row[0]
            self.dates.append(date)

            # Parse prices and volumes from the row
            # Row format: [Date, QQQ_AdjClose, IWM_Close, QQQ_Close, SPY_Close, IWM_High, QQQ_High, SPY_High, IWM_Low, QQQ_Low, SPY_Low, IWM_Open, QQQ_Open, SPY_Open, IWM_Volume, QQQ_Volume, SPY_Volume]
            # But some columns may be empty
            values = {}
            for i, val in enumerate(row[1:], start=1):
                if val:
                    values[i] = float(val)

            # Map column indices to assets
            # Based on the header: Close columns are at indices 3(IWM), 4(QQQ), 5(SPY) in the data row
            # Actually let me parse the header properly
            pass

        # Let me re-parse with proper header mapping
        self.dates = []
        self.prices = {asset: [] for asset in self.config.assets}
        self.volumes = {asset: [] for asset in self.config.assets}

        # Parse header to find column indices
        header_row1 = rows[0]  # Price, Adj Close, Close, Close, Close, High, High, High, Low, Low, Low, Open, Open, Open, Volume, Volume, Volume
        header_row2 = rows[1]  # Ticker, QQQ, IWM, QQQ, SPY, IWM, QQQ, SPY, IWM, QQQ, SPY, IWM, QQQ, SPY, IWM, QQQ, SPY

        close_cols = {}
        volume_cols = {}
        for i in range(len(header_row1)):
            if header_row1[i] == "Close":
                close_cols[header_row2[i]] = i
            elif header_row1[i] == "Volume":
                volume_cols[header_row2[i]] = i

        for row in rows[3:]:
            if not row or not row[0]:
                continue
            date = row[0]
            self.dates.append(date)

            for asset in self.config.assets:
                if asset in close_cols:
                    col = close_cols[asset]
                    if col < len(row) and row[col]:
                        self.prices[asset].append(float(row[col]))
                    else:
                        self.prices[asset].append(0.0)
                else:
                    self.prices[asset].append(0.0)

                if asset in volume_cols:
                    col = volume_cols[asset]
                    if col < len(row) and row[col]:
                        self.volumes[asset].append(float(row[col]))
                    else:
                        self.volumes[asset].append(0.0)
                else:
                    self.volumes[asset].append(0.0)

        self.n_days = len(self.dates)
        print(f"Loaded {self.n_days} days of market data for {self.config.assets}")

    def reset(self):
        """Reset environment to start of episode."""
        cfg = self.config
        self.step_idx = cfg.train_start
        self.end_idx = min(cfg.train_start + cfg.steps_per_episode, cfg.train_end)

        self.cash = cfg.initial_capital
        self.positions = {asset: 0.0 for asset in cfg.assets}
        self.avg_prices = {asset: 0.0 for asset in cfg.assets}

        self.peak_equity = cfg.initial_capital
        self.equity_history = [cfg.initial_capital]
        self.trade_history = []
        self.total_commissions = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0

        self.prices_now = {}
        for asset in cfg.assets:
            self.prices_now[asset] = self.prices[asset][self.step_idx]

        self.prices_prev = {}
        for asset in cfg.assets:
            prev_idx = max(0, self.step_idx - 1)
            self.prices_prev[asset] = self.prices[asset][prev_idx]

        return self._get_observation()

    def _get_observation(self):
        """Get current market observation with technical indicators."""
        obs = {}
        for asset in self.config.assets:
            price = self.prices_now[asset]
            prev_price = self.prices_prev[asset]

            obs[f"{asset}_price_change"] = (price - prev_price) / prev_price if prev_price > 0 else 0
            obs[f"{asset}_position"] = self.positions[asset]
            obs[f"{asset}_avg_price"] = self.avg_prices[asset] if self.positions[asset] != 0 else price

            window = min(20, self.step_idx + 1)
            recent_prices = self.prices[asset][self.step_idx - window + 1:self.step_idx + 1]
            obs[f"{asset}_sma_20"] = np.mean(recent_prices) / price - 1.0 if price > 0 else 0

            gains = []
            losses = []
            for i in range(1, len(recent_prices)):
                change = recent_prices[i] - recent_prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.0001
            rs = avg_gain / avg_loss
            obs[f"{asset}_rsi"] = 100 - (100 / (1 + rs))

        equity = self.cash + sum(
            self.positions[a] * self.prices_now[a] for a in self.config.assets
        )
        obs["cash_pct"] = self.cash / equity if equity > 0 else 1.0
        obs["equity"] = equity / self.config.initial_capital

        self.peak_equity = max(self.peak_equity, equity)
        obs["drawdown"] = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        return obs

    def step(self, action):
        """Execute a trading action."""
        cfg = self.config

        asset_idx = action // 3
        trade_size = action % 3
        asset = cfg.assets[asset_idx]

        if trade_size == 1:
            self._buy(asset, 0.1)
        elif trade_size == 2:
            self._buy(asset, 0.2)
        elif trade_size == 3:
            self._buy(asset, 0.3)
        elif trade_size == 4:
            self._sell(asset, 0.1)
        elif trade_size == 5:
            self._sell(asset, 0.2)
        elif trade_size == 6:
            self._sell(asset, 0.3)

        self.step_idx += 1
        if self.step_idx >= self.end_idx:
            return self._get_observation(), 0.0, True, {}

        for a in cfg.assets:
            self.prices_prev[a] = self.prices_now[a]
            self.prices_now[a] = self.prices[a][self.step_idx]

        for a in cfg.assets:
            if self.positions[a] > 0:
                pnl = (self.prices_now[a] - self.avg_prices[a]) / self.avg_prices[a] if self.avg_prices[a] > 0 else 0
                if pnl < -cfg.stop_loss_pct:
                    self._sell(a, 1.0)
                elif pnl > cfg.take_profit_pct:
                    self._sell(a, 0.5)

        equity = self.cash + sum(
            self.positions[a] * self.prices_now[a] for a in cfg.assets
        )
        prev_equity = self.equity_history[-1]
        reward = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0

        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown > cfg.max_drawdown_limit:
            reward -= 0.05

        self.equity_history.append(equity)

        return self._get_observation(), reward, False, {"equity": equity}

    def _buy(self, asset, fraction):
        """Buy a fraction of equity in an asset."""
        cfg = self.config
        equity = self.cash + sum(
            self.positions[a] * self.prices_now[a] for a in cfg.assets
        )
        max_position = equity * cfg.max_position_pct
        current_value = self.positions[asset] * self.prices_now[asset]
        available = min(equity * fraction, max_position - current_value)

        if available > 0 and self.cash > 0:
            shares = available / self.prices_now[asset]
            cost = shares * self.prices_now[asset]
            commission = cost * cfg.commission_rate
            slippage = cost * cfg.slippage_rate

            self.cash -= (cost + commission + slippage)
            self.positions[asset] += shares
            self.total_commissions += commission
            self.total_slippage += slippage
            self.total_trades += 1

            total_cost = self.positions[asset] * self.prices_now[asset]
            self.avg_prices[asset] = total_cost / self.positions[asset]

    def _sell(self, asset, fraction):
        """Sell a fraction of position in an asset."""
        cfg = self.config
        if self.positions[asset] <= 0:
            return

        shares = self.positions[asset] * fraction
        revenue = shares * self.prices_now[asset]
        commission = revenue * cfg.commission_rate
        slippage = revenue * cfg.slippage_rate

        self.cash += (revenue - commission - slippage)
        self.positions[asset] -= shares
        self.total_commissions += commission
        self.total_slippage += slippage
        self.total_trades += 1

    def get_performance_metrics(self):
        """Compute performance metrics."""
        equity = self.cash + sum(
            self.positions[a] * self.prices_now[a] for a in self.config.assets
        )
        total_return = (equity - self.config.initial_capital) / self.config.initial_capital

        returns = []
        for i in range(1, len(self.equity_history)):
            returns.append(
                (self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
            )
        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (mean_return * 252 - self.config.risk_free_rate) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0

        max_dd = 0
        peak = self.config.initial_capital
        for eq in self.equity_history:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": self.total_trades,
            "total_commissions": self.total_commissions,
            "total_slippage": self.total_slippage,
            "final_equity": equity,
        }
