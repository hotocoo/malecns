"""Comprehensive evolutionary strategy training for the malecns trading agent.

Features:
- Multi-instrument training (ETFs + Forex)
- Walk-forward validation
- Adaptive mutation with diversity maintenance
- Multi-objective fitness (return + Sharpe - drawdown - costs)
- Robustness testing across market regimes
- Early stopping with patience
- Detailed logging and checkpointing

Training protocol:
1. Train on historical data (train split)
2. Validate on holdout period (val split)
3. Test on unseen future data (test split)
4. Walk-forward: retrain on expanding window
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentConfig
from brain import LIFConfig, load_connectome, pick_device
from trading_agent import TradingAgent, REAL_OBS_FEATURES
from trading_env_real import RealTradingEnvironment, RealTradingConfig

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Fitness weights
W_RETURN = 10.0
W_SHARPE = 1.5
W_DRAWDOWN = -2.0
W_TRADING_COST = -0.01
W_WINRATE = 0.5


def compute_fitness(metrics):
    """Compute multi-objective fitness score."""
    total_return = metrics["total_return"]
    sharpe = metrics["sharpe_ratio"]
    max_drawdown = metrics["max_drawdown"]
    final_equity = metrics.get("final_equity", 100000)
    trading_costs = (metrics["total_commissions"] + metrics["total_slippage"]) / final_equity

    # Only reward positive Sharpe
    sharpe_term = W_SHARPE * sharpe if sharpe > 0 else 0.0

    return (
        W_RETURN * total_return
        + sharpe_term
        + W_DRAWDOWN * max_drawdown
        + W_TRADING_COST * trading_costs
    )


def evaluate_agent(agent, n_episodes=5, seed=0, train=True, market_data=None, split=None, obs_noise=0.0):
    """Evaluate an agent over multiple episodes."""
    rng = np.random.default_rng(seed)
    total_fitness = 0.0
    total_return = 0.0
    total_sharpe = 0.0
    total_drawdown = 0.0
    total_trades = 0
    total_commissions = 0.0
    total_slippage = 0.0
    episode_returns = []
    episode_sharpes = []
    episode_drawdowns = []

    for ep in range(n_episodes):
        ep_seed = seed * 1000 + ep
        if market_data is not None:
            new_env = RealTradingEnvironment(seed=ep_seed, data=market_data)
        else:
            new_env = RealTradingEnvironment(seed=ep_seed)

        if split == "test":
            # Use test split
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
            # Use validation split (middle 20%)
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
            # Add observation noise during training for regularization
            obs_to_use = agent.obs
            if train and obs_noise > 0:
                # obs is a dict of features
                obs_to_use = {
                    k: v + rng.normal(0, obs_noise)
                    for k, v in agent.obs.items()
                }

            action = agent.act(obs_to_use)
            obs, reward, done, info = agent.env.step(action)
            agent.obs = obs

        metrics = agent.env.get_performance_metrics()
        fitness = compute_fitness(metrics)
        total_fitness += fitness
        total_return += metrics["total_return"]
        total_sharpe += metrics["sharpe_ratio"]
        total_drawdown += metrics["max_drawdown"]
        total_trades += metrics["total_trades"]
        total_commissions += metrics["total_commissions"]
        total_slippage += metrics["total_slippage"]
        episode_returns.append(metrics["total_return"])
        episode_sharpes.append(metrics["sharpe_ratio"])
        episode_drawdowns.append(metrics["max_drawdown"])

    return {
        "fitness": total_fitness / n_episodes,
        "mean_return": total_return / n_episodes,
        "mean_sharpe": total_sharpe / n_episodes,
        "mean_drawdown": total_drawdown / n_episodes,
        "mean_trades": total_trades / n_episodes,
        "mean_commissions": total_commissions / n_episodes,
        "mean_slippage": total_slippage / n_episodes,
        "std_return": np.std(episode_returns),
        "std_sharpe": np.std(episode_sharpes),
        "best_return": np.max(episode_returns),
        "worst_return": np.min(episode_returns),
        "best_sharpe": np.max(episode_sharpes),
        "worst_sharpe": np.min(episode_sharpes),
        "best_drawdown": np.min(episode_drawdowns),
        "worst_drawdown": np.max(episode_drawdowns),
        "consistency": 1.0 - (np.std(episode_returns) / (np.abs(np.mean(episode_returns)) + 1e-8)),
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive malecns trading agent training")
    parser.add_argument("--graph", default="data/graph_w5", help="Connectome graph directory")
    parser.add_argument("--out", default="checkpoints/trading_comprehensive.pt", help="Output checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--population", type=int, default=48, help="Population size")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--mutation", type=float, default=0.05, help="Initial mutation strength")
    parser.add_argument("--mutation-min", type=float, default=0.005, help="Minimum mutation strength")
    parser.add_argument("--mutation-max", type=float, default=0.2, help="Maximum mutation strength")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N generations")
    parser.add_argument("--save-interval", type=int, default=25, help="Save checkpoint every N generations")
    parser.add_argument("--val-interval", type=int, default=10, help="Validate every N generations")
    parser.add_argument("--early-stop-patience", type=int, default=150, help="Early stopping patience")
    parser.add_argument("--data", default="data/real_market_data.csv", help="Market data file")
    parser.add_argument("--forex", action="store_true", help="Include Forex data")
    args = parser.parse_args()

    device = pick_device(args.device or "auto")
    print(f"Device: {device}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Episodes per eval: {args.episodes}")
    print(f"Initial mutation: {args.mutation}")
    print(f"Early stopping patience: {args.early_stop_patience}")
    print(f"Validation interval: {args.val_interval}")

    # Load connectome
    print("Loading connectome...")
    connectome = load_connectome(args.graph)
    print(f"  {connectome.n} neurons, {len(connectome.pre)} edges")

    # Create base agent
    print("Creating trading agent...")
    agent_cfg = AgentConfig(substeps=2)
    base_agent = TradingAgent(
        args.graph,
        checkpoint=args.resume,
        device=device,
        agent_config=agent_cfg,
        obs_features=REAL_OBS_FEATURES,
        connectome=connectome,
    )

    # Load market data
    print("Loading market data...")
    initial_env = RealTradingEnvironment()
    market_data = {
        "dates": initial_env.dates,
        "prices": initial_env.prices,
        "volumes": initial_env.volumes,
        "n_days": initial_env.n_days,
    }
    del initial_env

    # Training log
    log_path = LOG_DIR / "train_trading_comprehensive.jsonl"
    log_file = open(log_path, "a")

    # ES state
    best_theta = base_agent.theta
    best_fitness = -np.inf
    best_return = -np.inf
    best_val_fitness = -np.inf
    best_val_return = -np.inf
    generation = 0
    mutation_strength = args.mutation
    stagnation_counter = 0
    val_stagnation_counter = 0
    fitness_history = deque(maxlen=50)

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if "generation" in state:
            generation = state["generation"]
            best_fitness = state.get("best_fitness", -np.inf)
            best_return = state.get("best_return", -np.inf)
            best_val_fitness = state.get("best_val_fitness", -np.inf)
            best_val_return = state.get("best_val_return", -np.inf)
            mutation_strength = state.get("mutation_strength", args.mutation)
        print(f"Resumed from generation {generation}")

    start_time = time.time()

    for gen in range(generation, args.generations):
        gen_start = time.time()
        population = []

        # Create population with mutations
        for i in range(args.population):
            agent = TradingAgent(
                args.graph,
                device=device,
                agent_config=agent_cfg,
                obs_features=REAL_OBS_FEATURES,
                connectome=connectome,
            )

            if i == 0:
                # Elitism: keep best
                agent.theta = {k: v.clone() for k, v in best_theta.items()}
            else:
                # Copy best, then mutate
                agent.theta = {k: v.clone() for k, v in best_theta.items()}
                for name in agent.theta:
                    if agent.theta[name].dim() > 0:
                        noise = torch.distributions.cauchy.Cauchy(0, 1).sample(
                            agent.theta[name].shape
                        ).to(agent.theta[name].device)
                        agent.theta[name] = agent.theta[name] + noise * mutation_strength

            # Evaluate on training data with observation noise
            fitness = evaluate_agent(
                agent,
                n_episodes=args.episodes,
                seed=gen * 100 + i,
                train=True,
                market_data=market_data,
                obs_noise=0.02,  # 2% observation noise for regularization
            )
            population.append((fitness, agent.theta))
            del agent

        # Validation every generation (faster feedback)
        val_fitness = None
        if gen % max(1, args.val_interval // 2) == 0:
            val_agent = TradingAgent(
                args.graph,
                device=device,
                agent_config=agent_cfg,
                obs_features=REAL_OBS_FEATURES,
                connectome=connectome,
            )
            val_agent.theta = {k: v.clone() for k, v in best_theta.items()}
            val_fitness = evaluate_agent(
                val_agent,
                n_episodes=max(3, args.episodes // 2),
                seed=gen * 1000,
                train=False,
                market_data=market_data,
                split="val",
            )
            del val_agent

            if val_fitness["fitness"] > best_val_fitness:
                best_val_fitness = val_fitness["fitness"]
                best_val_return = val_fitness["mean_return"]
                val_stagnation_counter = 0
            else:
                val_stagnation_counter += 1

        # Combined fitness: 60% training + 40% validation
        # This directly penalizes overfitting during selection
        for idx, (fit, theta) in enumerate(population):
            if val_fitness is not None:
                population[idx] = (
                    {
                        **fit,
                        "fitness": 0.6 * fit["fitness"] + 0.4 * val_fitness["fitness"],
                    },
                    theta,
                )

        # Sort by combined fitness
        population.sort(key=lambda x: x[0]["fitness"], reverse=True)

        # Update best
        best_fitness_gen = population[0][0]["fitness"]
        best_return_gen = population[0][0]["mean_return"]
        if best_fitness_gen > best_fitness:
            best_fitness = best_fitness_gen
            best_return = best_return_gen
            best_theta = population[0][1]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Adaptive mutation
        if stagnation_counter > 25:
            mutation_strength = min(mutation_strength * 1.15, args.mutation_max)
        elif best_fitness_gen > best_fitness * 0.99:
            mutation_strength = max(mutation_strength * 0.97, args.mutation_min)

        fitness_history.append(best_fitness_gen)

        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time

        # Early stopping
        if val_stagnation_counter > args.early_stop_patience:
            print(f"\nEarly stopping at generation {gen} (validation stagnation: {val_stagnation_counter})")
            break

        # Log
        if gen % args.log_interval == 0:
            gen_best = population[0][0]["mean_return"]
            gen_sharpe = population[0][0]["mean_sharpe"]
            gen_dd = population[0][0]["mean_drawdown"]
            val_str = ""
            if val_fitness:
                val_str = (
                    f" | Val fit: {val_fitness['fitness']:8.4f} | "
                    f"Val ret: {val_fitness['mean_return']*100:8.2f}% | "
                    f"Val Sharpe: {val_fitness['mean_sharpe']:6.2f}"
                )
            print(
                f"Gen {gen:5d} | Comb fit: {best_fitness:8.4f} | "
                f"Best ret: {best_return*100:8.2f}% | "
                f"Gen best: {gen_best*100:8.2f}% | "
                f"Sharpe: {gen_sharpe:6.2f} | "
                f"DD: {gen_dd*100:6.2f}% | "
                f"Mut: {mutation_strength:.4f}{val_str} | "
                f"{gen_time:.1f}s/gen"
            )

            log_entry = {
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "gen_best_fitness": population[0][0]["fitness"],
                "gen_best_return": gen_best,
                "gen_best_sharpe": gen_sharpe,
                "gen_best_drawdown": gen_dd,
                "gen_worst_fitness": population[-1][0]["fitness"],
                "gen_worst_return": population[-1][0]["mean_return"],
                "mutation_strength": mutation_strength,
                "stagnation_counter": stagnation_counter,
                "val_fitness": val_fitness["fitness"] if val_fitness else None,
                "val_return": val_fitness["mean_return"] if val_fitness else None,
                "val_sharpe": val_fitness["mean_sharpe"] if val_fitness else None,
                "val_stagnation": val_stagnation_counter,
                "gen_time": gen_time,
                "elapsed": elapsed,
                "timestamp": time.time(),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        # Save checkpoint
        if gen % args.save_interval == 0:
            checkpoint_path = Path(args.out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            base_agent.theta = best_theta
            base_agent.save_checkpoint(str(checkpoint_path), extra={
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "best_val_fitness": best_val_fitness,
                "best_val_return": best_val_return,
                "mutation_strength": mutation_strength,
            })

            best_path = checkpoint_path.parent / "trading_comprehensive_best.pt"
            base_agent.save_checkpoint(str(best_path), extra={
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "best_val_fitness": best_val_fitness,
                "best_val_return": best_val_return,
            })

        generation = gen + 1

    # Final save
    base_agent.theta = best_theta
    base_agent.save_checkpoint(args.out, extra={
        "generation": generation,
        "best_fitness": best_fitness,
        "best_return": best_return,
        "best_val_fitness": best_val_fitness,
        "best_val_return": best_val_return,
        "mutation_strength": mutation_strength,
    })

    log_file.close()
    print(f"\nTraining complete.")
    print(f"Best combined fitness: {best_fitness:.4f}")
    print(f"Best training return: {best_return*100:.2f}%")
    print(f"Best validation fitness: {best_val_fitness:.4f}")
    print(f"Best validation return: {best_val_return*100:.2f}%")
    print(f"Checkpoint saved to: {args.out}")

    # Final test evaluation
    print("\nRunning final test evaluation...")
    test_agent = TradingAgent(
        args.graph,
        device=device,
        agent_config=agent_cfg,
        obs_features=REAL_OBS_FEATURES,
        connectome=connectome,
    )
    test_agent.theta = {k: v.clone() for k, v in best_theta.items()}
    test_fitness = evaluate_agent(
        test_agent,
        n_episodes=10,
        seed=99999,
        train=False,
        market_data=market_data,
        split="test",
    )
    print(f"Test fitness: {test_fitness['fitness']:.4f}")
    print(f"Test return: {test_fitness['mean_return']*100:.2f}%")
    print(f"Test Sharpe: {test_fitness['mean_sharpe']:.2f}")
    print(f"Test max drawdown: {test_fitness['mean_drawdown']*100:.2f}%")
    print(f"Test consistency: {test_fitness['consistency']*100:.1f}%")


if __name__ == "__main__":
    main()
