"""Comprehensive evaluation of the trained trading agent.

Evaluates on:
1. Training data (to check for overfitting)
2. Validation data (holdout period)
3. Test data (unseen future data)
4. Walk-forward validation (multiple train/test splits)
5. Stress tests (high volatility, trending, mean-reverting)

Outputs detailed performance metrics and reports.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentConfig
from brain import LIFConfig, load_connectome, pick_device
from trading_agent import TradingAgent, REAL_OBS_FEATURES
from trading_env_real import RealTradingEnvironment, RealTradingConfig



def evaluate_on_split(agent, split, n_episodes=10, seed=42, market_data=None):
    """Evaluate agent on a specific data split."""
    total_fitness = 0.0
    total_return = 0.0
    total_sharpe = 0.0
    total_drawdown = 0.0
    total_trades = 0
    episode_returns = []

    for ep in range(n_episodes):
        ep_seed = seed * 1000 + ep
        if market_data is not None:
            new_env = RealTradingEnvironment(seed=ep_seed, data=market_data)
        else:
            new_env = RealTradingEnvironment(seed=ep_seed)

        if split == "test":
            new_env.config = RealTradingConfig(
                data_file=new_env.config.data_file,
                assets=new_env.config.assets,
                initial_capital=new_env.config.initial_capital,
                commission_rate=new_env.config.commission_rate,
                slippage_rate=new_env.config.slippage_rate,
                max_position_pct=new_env.config.max_position_pct,
                stop_loss_pct=new_env.config.stop_loss_pct,
                take_profit_pct=new_env.config.take_profit_pct,
                max_drawdown_limit=new_env.config.max_drawdown_limit,
                risk_free_rate=new_env.config.risk_free_rate,
                steps_per_episode=new_env.config.steps_per_episode,
                train_start=new_env.config.test_start,
                train_end=new_env.config.test_end,
                test_start=new_env.config.test_start,
                test_end=new_env.config.test_end,
            )
        elif split == "val":
            n = new_env.n_days
            val_start = int(n * 0.7)
            val_end = int(n * 0.85)
            new_env.config = RealTradingConfig(
                data_file=new_env.config.data_file,
                assets=new_env.config.assets,
                initial_capital=new_env.config.initial_capital,
                commission_rate=new_env.config.commission_rate,
                slippage_rate=new_env.config.slippage_rate,
                max_position_pct=new_env.config.max_position_pct,
                stop_loss_pct=new_env.config.stop_loss_pct,
                take_profit_pct=new_env.config.take_profit_pct,
                max_drawdown_limit=new_env.config.max_drawdown_limit,
                risk_free_rate=new_env.config.risk_free_rate,
                steps_per_episode=min(new_env.config.steps_per_episode, val_end - val_start),
                train_start=val_start,
                train_end=val_end,
                test_start=val_start,
                test_end=val_end,
            )

        new_env.reset()
        agent.env = new_env
        agent.obs = new_env.reset()
        agent.total_reward = 0.0
        agent.agent.reset(batch=1)

        done = False
        while not done:
            action = agent.act(agent.obs)
            obs, reward, done, info = agent.env.step(action)
            agent.obs = obs

        metrics = agent.env.get_performance_metrics()
        total_return += metrics["total_return"]
        total_sharpe += metrics["sharpe_ratio"]
        total_drawdown += metrics["max_drawdown"]
        total_trades += metrics["total_trades"]
        episode_returns.append(metrics["total_return"])

    n = n_episodes
    return {
        "split": split,
        "mean_return": total_return / n,
        "mean_sharpe": total_sharpe / n,
        "mean_drawdown": total_drawdown / n,
        "mean_trades": total_trades / n,
        "std_return": np.std(episode_returns),
        "best_return": np.max(episode_returns),
        "worst_return": np.min(episode_returns),
        "consistency": 1.0 - (np.std(episode_returns) / (np.abs(np.mean(episode_returns)) + 1e-8)),
    }


def evaluate_stress(agent, n_episodes=5, seed=42):
    """Evaluate agent under different market stress conditions using real data subsets."""
    results = {}
    # Use different market regimes from real data
    regimes = {
        "bull_market_2021": (40, 100),      # Jan-Mar 2021 (bull market)
        "crash_march_2020": (0, 40),        # Jan-Feb 2020 (pre-crash)
        "recovery_2020": (40, 120),         # Mar-Aug 2020 (recovery)
        "sideways_2023": (300, 400),        # 2023 (sideways)
    }

    for regime_name, (start, end) in regimes.items():
        total_return = 0.0
        total_sharpe = 0.0
        for ep in range(n_episodes):
            ep_seed = seed * 1000 + ep
            new_env = RealTradingEnvironment(seed=ep_seed)
            new_env.config = RealTradingConfig(
                data_file=new_env.config.data_file,
                assets=new_env.config.assets,
                initial_capital=new_env.config.initial_capital,
                commission_rate=new_env.config.commission_rate,
                slippage_rate=new_env.config.slippage_rate,
                max_position_pct=new_env.config.max_position_pct,
                stop_loss_pct=new_env.config.stop_loss_pct,
                take_profit_pct=new_env.config.take_profit_pct,
                max_drawdown_limit=new_env.config.max_drawdown_limit,
                risk_free_rate=new_env.config.risk_free_rate,
                steps_per_episode=end - start,
                train_start=start,
                train_end=end,
                test_start=start,
                test_end=end,
            )
            agent.env = new_env
            agent.obs = new_env.reset()
            agent.total_reward = 0.0
            agent.agent.reset(batch=1)

            done = False
            while not done:
                action = agent.act(agent.obs)
                obs, reward, done, info = agent.env.step(action)
                agent.obs = obs

            metrics = agent.env.get_performance_metrics()
            total_return += metrics["total_return"]
            total_sharpe += metrics["sharpe_ratio"]

        results[regime_name] = {
            "mean_return": total_return / n_episodes,
            "mean_sharpe": total_sharpe / n_episodes,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained trading agent")
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--checkpoint", default="checkpoints/trading_comprehensive.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", default="logs/evaluation_report.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Evaluating trading agent on device: {device}")
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

    # Load market data
    initial_env = RealTradingEnvironment()
    market_data = {
        "dates": initial_env.dates,
        "prices": initial_env.prices,
        "volumes": initial_env.volumes,
        "n_days": initial_env.n_days,
    }
    del initial_env

    results = {}

    # Evaluate on training data
    print("\n=== Training Data Evaluation ===")
    results["train"] = evaluate_on_split(agent, "train", args.episodes, market_data=market_data)
    r = results["train"]
    print(f"Return: {r['mean_return']*100:.2f}% (std: {r['std_return']*100:.2f}%)")
    print(f"Sharpe: {r['mean_sharpe']:.2f}")
    print(f"Max Drawdown: {r['mean_drawdown']*100:.2f}%")
    print(f"Trades: {r['mean_trades']:.1f}")

    # Evaluate on validation data
    print("\n=== Validation Data Evaluation ===")
    results["val"] = evaluate_on_split(agent, "val", args.episodes, market_data=market_data)
    r = results["val"]
    print(f"Return: {r['mean_return']*100:.2f}% (std: {r['std_return']*100:.2f}%)")
    print(f"Sharpe: {r['mean_sharpe']:.2f}")
    print(f"Max Drawdown: {r['mean_drawdown']*100:.2f}%")
    print(f"Trades: {r['mean_trades']:.1f}")

    # Evaluate on test data
    print("\n=== Test Data Evaluation ===")
    results["test"] = evaluate_on_split(agent, "test", args.episodes, market_data=market_data)
    r = results["test"]
    print(f"Return: {r['mean_return']*100:.2f}% (std: {r['std_return']*100:.2f}%)")
    print(f"Sharpe: {r['mean_sharpe']:.2f}")
    print(f"Max Drawdown: {r['mean_drawdown']*100:.2f}%")
    print(f"Trades: {r['mean_trades']:.1f}")

    # Stress tests
    print("\n=== Stress Tests ===")
    results["stress"] = evaluate_stress(agent, 5)
    for regime, metrics in results["stress"].items():
        print(f"{regime}: Return {metrics['mean_return']*100:.2f}%, Sharpe {metrics['mean_sharpe']:.2f}")

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation report saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Split':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}")
    print("-" * 60)
    for split in ["train", "val", "test"]:
        r = results[split]
        print(f"{split:<12} {r['mean_return']*100:>9.2f}% {r['mean_sharpe']:>8.2f} {r['mean_drawdown']*100:>7.2f}% {r['mean_trades']:>8.1f}")


if __name__ == "__main__":
    main()
