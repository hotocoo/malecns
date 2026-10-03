#!/usr/bin/env python3
"""Backtest the MaleCNS trading strategy on historical data.

Usage:
    python3 src/backtest_trading.py --instrument EUR_USD --period 1mo
    python3 src/backtest_trading.py --instruments EUR_USD,GBP_USD --period 3mo
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading_agent import TradingAgent
from trading_env import TradeAction
from realtime_trading import load_config


def load_historical_data(instrument: str, period: str = "1mo", interval: str = "1d"):
    """Load historical price data using yfinance."""
    import yfinance as yf
    
    # yfinance uses different symbols for Forex: EURUSD=X, GBPUSD=X, USDJPY=X
    if "_" in instrument:
        yf_symbol = instrument.replace("_", "") + "=X"
    else:
        yf_symbol = instrument
    
    print(f"Downloading {instrument} data ({period}, {interval})...")
    try:
        df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {yf_symbol}")
        print(f"  Got {len(df)} data points")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_backtest(agent, df, instrument: str, risk_per_trade: float = 0.01,
                 stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04,
                 slippage_pct: float = 0.0001, commission_per_trade: float = 0.0):
    """Run backtest on historical data with slippage and commission modeling.
    
    Args:
        agent: Trading agent
        df: Historical price data
        instrument: Currency pair symbol
        risk_per_trade: Risk per trade as fraction of balance
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        slippage_pct: Slippage as fraction of price (default 0.01% = 1 pip for EUR/USD)
        commission_per_trade: Commission per trade in USD
    """
    # Handle both regular and multi-index DataFrames
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].values.flatten()
    else:
        prices = df["Close"].values
    n = len(prices)
    
    balance = 100000.0
    position = 0.0  # units
    entry_price = 0.0
    trades = []
    equity_curve = []
    total_commission = 0.0
    total_slippage_cost = 0.0
    
    for i in range(1, n):
        price = float(prices[i])
        prev_price = float(prices[i-1])
        
        # Apply slippage to current price
        # Buy: pay slightly more; Sell: receive slightly less
        buy_price = price * (1 + slippage_pct)
        sell_price = price * (1 - slippage_pct)
        
        # Create observation (all values must be scalars)
        obs = {
            "price_change": float((price - prev_price) / prev_price),
            "price_change_5": float((price - prices[max(0, i-5)]) / prices[max(0, i-5)]) if i >= 5 else 0.0,
            "price_change_20": float((price - prices[max(0, i-20)]) / prices[max(0, i-20)]) if i >= 20 else 0.0,
            "price_normalized": float(price / prices[0]),
            "position_pct": float(position * price / balance) if balance > 0 else 0.0,
            "equity_normalized": float(balance / 100000.0),
            "unrealized_pnl": float((price - entry_price) * position) if position != 0 else 0.0,
            "rsi": 50.0,
            "macd": 0.0,
            "macd_line": 0.0,
            "bb_position": 0.5,
            "bb_width": 0.02,
            "atr": float(price * 0.01),
            "momentum_5": float((price - prices[max(0, i-5)]) / prices[max(0, i-5)]) if i >= 5 else 0.0,
            "realized_vol": 0.01,
            "spread": 0.0001,
            "drawdown": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.01,
        }
        
        # Get action
        action = agent.act(obs)
        
        # Execute action
        if action in [TradeAction.BUY_SMALL, TradeAction.BUY_MEDIUM, TradeAction.BUY_LARGE]:
            if position <= 0:
                size_mult = 1.0 if action == TradeAction.BUY_SMALL else (1.5 if action == TradeAction.BUY_MEDIUM else 2.0)
                units = (balance * risk_per_trade * size_mult) / (price * stop_loss_pct)
                position = units
                entry_price = buy_price  # Entry at buy price (with slippage)
                slippage_cost = (buy_price - price) * units
                total_slippage_cost += slippage_cost
                if commission_per_trade > 0:
                    balance -= commission_per_trade
                    total_commission += commission_per_trade
                trades.append({"type": "BUY", "price": buy_price, "units": units, "date": i})
        
        elif action in [TradeAction.SELL_SMALL, TradeAction.SELL_MEDIUM, TradeAction.SELL_LARGE]:
            if position > 0:
                size_mult = 1.0 if action == TradeAction.SELL_SMALL else (1.5 if action == TradeAction.SELL_MEDIUM else 2.0)
                units = min(position, (balance * risk_per_trade * size_mult) / (price * stop_loss_pct))
                pnl = (sell_price - entry_price) * units  # Exit at sell price (with slippage)
                slippage_cost = (price - sell_price) * units
                total_slippage_cost += slippage_cost
                if commission_per_trade > 0:
                    balance -= commission_per_trade
                    total_commission += commission_per_trade
                balance += pnl
                position -= units
                trades.append({"type": "SELL", "price": sell_price, "units": units, "pnl": pnl, "date": i})
        
        # Check stop loss / take profit
        if position > 0:
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                exit_price = sell_price  # Exit at sell price (with slippage)
                pnl = (exit_price - entry_price) * position
                slippage_cost = (price - exit_price) * position
                total_slippage_cost += slippage_cost
                if commission_per_trade > 0:
                    balance -= commission_per_trade
                    total_commission += commission_per_trade
                balance += pnl
                trades.append({"type": "SL/TP", "price": exit_price, "units": position, "pnl": pnl, "date": i})
                position = 0.0
        
        # Track equity
        equity = balance + position * price
        equity_curve.append(equity)
    
    # Calculate metrics
    total_return = (equity_curve[-1] - 100000.0) / 100000.0
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len([t for t in trades if "pnl" in t]) if any("pnl" in t for t in trades) else 0.0
    
    # Max drawdown
    max_dd = 0.0
    peak = equity_curve[0]
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)
    
    return {
        "total_return": total_return,
        "final_equity": equity_curve[-1],
        "num_trades": len(trades),
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "total_commission": total_commission,
        "total_slippage_cost": total_slippage_cost,
        "equity_curve": equity_curve,
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest MaleCNS trading strategy")
    parser.add_argument("--instrument", default="EUR_USD", help="Trading instrument")
    parser.add_argument("--instruments", default=None, help="Comma-separated instruments")
    parser.add_argument("--period", default="1mo", help="Data period (1mo, 3mo, 6mo, 1y)")
    parser.add_argument("--interval", default="1d", help="Data interval (1d, 1h, 15m)")
    parser.add_argument("--graph", default="data/graph_w5", help="Connectome graph directory")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint")
    parser.add_argument("--device", default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade")
    parser.add_argument("--stop-loss", type=float, default=0.02, help="Stop loss pct")
    parser.add_argument("--take-profit", type=float, default=0.04, help="Take profit pct")
    parser.add_argument("--slippage", type=float, default=0.0001, help="Slippage as fraction (default 0.01% = 1 pip)")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade in USD")
    parser.add_argument("--sweep", action="store_true", help="Sweep risk parameters (0.5%, 1%, 2%, 3%)")
    parser.add_argument("--sltp-sweep", action="store_true", help="Sweep SL/TP ratios (1:1, 1:2, 1:3, 1:4)")
    parser.add_argument("--output", default=None, help="Export results to CSV file")
    parser.add_argument("--periods", default=None, help="Comma-separated periods to test (e.g., 1mo,3mo,6mo)")
    args = parser.parse_args()
    
    # Determine instruments
    if args.instruments:
        instruments = args.instruments.split(",")
    else:
        instruments = [args.instrument]
    
    # Load brain
    print("Loading MaleCNS brain...")
    agent = TradingAgent(args.graph, checkpoint=args.checkpoint, device=args.device)
    
    # Determine risk levels to test
    if args.sweep:
        risk_levels = [0.005, 0.01, 0.02, 0.03]
    else:
        risk_levels = [args.risk]
    
    # Determine SL/TP ratios to test
    if args.sltp_sweep:
        sltp_ratios = [(0.01, 0.01), (0.01, 0.02), (0.01, 0.03), (0.01, 0.04)]
    else:
        sltp_ratios = [(args.stop_loss, args.take_profit)]
    
    # Determine periods to test
    if args.periods:
        periods = args.periods.split(",")
    else:
        periods = [args.period]
    
    # Collect results for CSV export
    all_results = []
    
    # Run backtest for each instrument, period, risk, and SL/TP
    for instrument in instruments:
        for period in periods:
            print(f"\n{'=' * 50}")
            print(f"Backtesting {instrument} ({period})")
            print(f"{'=' * 50}")
            
            df = load_historical_data(instrument, period, args.interval)
            if df is None:
                continue
            
            for risk in risk_levels:
                for sl, tp in sltp_ratios:
                    results = run_backtest(
                        agent, df, instrument,
                        risk_per_trade=risk,
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                        slippage_pct=args.slippage,
                        commission_per_trade=args.commission,
                    )
                    
                    risk_label = f" (risk={risk*100:.1f}%)" if args.sweep else ""
                    sltp_label = f" (SL={sl*100:.0f}%/TP={tp*100:.0f}%)" if args.sltp_sweep else ""
                    print(f"\nResults for {instrument} ({period}){risk_label}{sltp_label}:")
                    print(f"  Total Return:  {results['total_return']*100:.2f}%")
                    print(f"  Final Equity:  ${results['final_equity']:,.2f}")
                    print(f"  Number of Trades: {results['num_trades']}")
                    print(f"  Win Rate:      {results['win_rate']*100:.1f}%")
                    print(f"  Max Drawdown:  {results['max_drawdown']*100:.2f}%")
                    
                    all_results.append({
                        "instrument": instrument,
                        "period": period,
                        "risk": risk,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "total_return": results["total_return"],
                        "final_equity": results["final_equity"],
                        "num_trades": results["num_trades"],
                        "win_rate": results["win_rate"],
                        "max_drawdown": results["max_drawdown"],
                    })
    
    # Export to CSV if requested
    if args.output and all_results:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults exported to {args.output}")


if __name__ == "__main__":
    main()
