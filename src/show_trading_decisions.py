"""Show the virtual expert panel's trading decisions in real-time.

This script runs the trained trading agent and displays:
1. Market conditions (price, indicators)
2. Expert panel analysis (technical, fundamental, risk, psychology)
3. Brain's decision (BUY/SELL/HOLD and size)
4. Brain's internal state (neuron activation patterns)
"""

from __future__ import annotations

import argparse
import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentConfig
from brain import LIFConfig, load_connectome, pick_device
from trading_agent import TradingAgent, REAL_OBS_FEATURES
from trading_env_real import RealTradingEnvironment, RealTradingConfig


class ExpertPanel:
    """Virtual expert panel that analyzes market conditions."""

    def __init__(self):
        self.technicals = {
            "rsi": 50.0,
            "macd": 0.0,
            "bb_upper": 0.0,
            "bb_lower": 0.0,
            "atr": 0.0,
        }
        self.fundamentals = {
            "price_trend": "neutral",
            "volume_trend": "neutral",
        }
        self.risk = {
            "position_size": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
        }
        self.psychology = {
            "sentiment": "neutral",
            "fear_greed": 50.0,
        }

    def analyze(self, obs):
        """Analyze market conditions and produce expert opinions."""
        # Get SPY data (primary asset)
        spy_price_change = obs.get("SPY_price_change", 0.0)
        spy_rsi = obs.get("SPY_rsi", 50.0)
        spy_sma_20 = obs.get("SPY_sma_20", 0.0)
        spy_avg_price = obs.get("SPY_avg_price", 0.0)
        spy_position = obs.get("SPY_position", 0.0)

        # Get QQQ data
        qqq_price_change = obs.get("QQQ_price_change", 0.0)
        qqq_rsi = obs.get("QQQ_rsi", 50.0)

        # Get IWM data
        iwm_price_change = obs.get("IWM_price_change", 0.0)
        iwm_rsi = obs.get("IWM_rsi", 50.0)

        # Portfolio data
        cash_pct = obs.get("cash_pct", 1.0)
        equity = obs.get("equity", 1.0)
        drawdown = obs.get("drawdown", 0.0)

        # RSI analysis (average across assets)
        avg_rsi = (spy_rsi + qqq_rsi + iwm_rsi) / 3.0
        if avg_rsi > 70:
            rsi_signal = "OVERBOUGHT - Consider selling"
        elif avg_rsi < 30:
            rsi_signal = "OVERSOLD - Consider buying"
        else:
            rsi_signal = "NEUTRAL"

        # Price trend (SPY as proxy)
        if spy_price_change > 0.001:
            trend = "UP"
        elif spy_price_change < -0.001:
            trend = "DOWN"
        else:
            trend = "FLAT"

        # Market breadth (how many assets are up)
        assets_up = sum([spy_price_change > 0, qqq_price_change > 0, iwm_price_change > 0])
        if assets_up == 3:
            breadth = "BROAD BULLISH (3/3 up)"
        elif assets_up == 2:
            breadth = "MODERATE BULLISH (2/3 up)"
        elif assets_up == 1:
            breadth = "MODERATE BEARISH (1/3 up)"
        else:
            breadth = "BROAD BEARISH (0/3 up)"

        # Sentiment
        if avg_rsi > 70:
            sentiment = "VERY BULLISH"
        elif avg_rsi > 50:
            sentiment = "BULLISH"
        elif avg_rsi < 30:
            sentiment = "VERY BEARISH"
        elif avg_rsi < 50:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        # Fear/Greed index (0 = extreme fear, 100 = extreme greed)
        fear_greed = 50.0 + (avg_rsi - 50.0) * 0.5

        return {
            "technical": {
                "spy_rsi": spy_rsi,
                "qqq_rsi": qqq_rsi,
                "iwm_rsi": iwm_rsi,
                "avg_rsi": avg_rsi,
                "rsi_signal": rsi_signal,
                "breadth": breadth,
            },
            "fundamental": {
                "price_trend": trend,
                "spy_change": spy_price_change,
                "qqq_change": qqq_price_change,
                "iwm_change": iwm_price_change,
                "spy_position": spy_position,
            },
            "portfolio": {
                "cash_pct": cash_pct,
                "equity": equity,
                "drawdown": drawdown,
            },
            "psychology": {
                "sentiment": sentiment,
                "fear_greed": fear_greed,
            },
        }


def action_to_text(action):
    """Convert action index to text."""
    actions = ["HOLD", "BUY SMALL", "BUY MEDIUM", "BUY LARGE",
               "SELL SMALL", "SELL MEDIUM", "SELL LARGE"]
    return actions[action]


def main():
    parser = argparse.ArgumentParser(description="Show trading decisions with expert panel")
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--checkpoint", default="checkpoints/trading_comprehensive_best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=50, help="Number of trading steps to show")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load connectome and agent
    connectome = load_connectome(args.graph)
    agent = TradingAgent(
        args.graph,
        checkpoint=args.checkpoint,
        device=device,
        agent_config=AgentConfig(substeps=2),
        obs_features=REAL_OBS_FEATURES,
        connectome=connectome,
    )

    # Create environment
    env = RealTradingEnvironment()
    agent.env = env

    # Expert panel
    panel = ExpertPanel()

    # Reset
    obs = env.reset()
    agent.obs = obs
    agent.total_reward = 0.0
    agent.agent.reset(batch=1)

    print("\n" + "=" * 80)
    print("MALECNS TRADING AGENT - VIRTUAL EXPERT PANEL ANALYSIS")
    print("=" * 80)

    for step in range(args.steps):
        # Get expert panel analysis
        analysis = panel.analyze(obs)

        # Get brain's decision
        action = agent.act(agent.obs)

        # Get brain's internal state
        brain = agent.agent.brain
        membrane = brain.v.detach().cpu().numpy()

        # Print analysis
        print(f"\n--- Step {step} ---")
        print(f"SPY: {obs.get('SPY_avg_price', 0):.2f} ({obs.get('SPY_price_change', 0):+.4%})")
        print(f"QQQ: ({obs.get('QQQ_price_change', 0):+.4%})  IWM: ({obs.get('IWM_price_change', 0):+.4%})")

        # Technical analysis
        tech = analysis["technical"]
        print(f"\n📊 TECHNICAL ANALYST:")
        print(f"   RSI: SPY {tech['spy_rsi']:.1f} | QQQ {tech['qqq_rsi']:.1f} | IWM {tech['iwm_rsi']:.1f}")
        print(f"   Avg RSI: {tech['avg_rsi']:.1f} → {tech['rsi_signal']}")
        print(f"   Market Breadth: {tech['breadth']}")

        # Fundamental analysis
        fund = analysis["fundamental"]
        print(f"\n📈 FUNDAMENTAL ANALYST:")
        print(f"   Price Trend: {fund['price_trend']} (SPY {fund['spy_change']:+.4%})")
        print(f"   SPY Position: {fund['spy_position']:.4f}")

        # Portfolio status
        port = analysis["portfolio"]
        print(f"\n💼 PORTFOLIO MANAGER:")
        print(f"   Cash: {port['cash_pct']:.1%} | Equity: {port['equity']:.4f} | Drawdown: {port['drawdown']:.2%}")

        # Market psychology
        psych = analysis["psychology"]
        print(f"\n🧠 MARKET PSYCHOLOGIST:")
        print(f"   Sentiment: {psych['sentiment']}")
        print(f"   Fear/Greed: {psych['fear_greed']:.1f}/100")

        # Brain's decision
        print(f"\n🦟 FLY BRAIN DECISION:")
        print(f"   Action: {action_to_text(action)}")
        print(f"   Neurons Active: {int(np.sum(membrane > 0.1))}/{membrane.shape[0]}")
        print(f"   Mean Membrane Potential: {np.mean(membrane):.4f}")

        # Step environment
        obs, reward, done, info = env.step(action)
        agent.obs = obs

        if done:
            print("\nEpisode complete!")
            metrics = env.get_performance_metrics()
            print(f"Total Return: {metrics['total_return']*100:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"Total Trades: {metrics['total_trades']}")
            break


if __name__ == "__main__":
    main()
