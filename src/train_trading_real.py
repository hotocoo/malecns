"""Evolutionary strategy training using real-world market data.

Features:
- Train/test split to prevent overfitting
- Early stopping based on validation performance
- Adaptive mutation strength
- Multi-objective fitness (return + Sharpe - drawdown - costs)
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
W_SHARPE = 1.0
W_DRAWDOWN = -1.0
W_TRADING_COST = -0.01


def compute_fitness(metrics):
    """Compute multi-objective fitness score."""
    total_return = metrics["total_return"]
    sharpe = metrics["sharpe_ratio"]
    max_drawdown = metrics["max_drawdown"]
    trading_costs = (metrics["total_commissions"] + metrics["total_slippage"]) / metrics.get("final_equity", 100000)
    sharpe_term = W_SHARPE * sharpe if sharpe > 0 else 0.0
    return (
        W_RETURN * total_return
        + sharpe_term
        + W_DRAWDOWN * max_drawdown
        + W_TRADING_COST * trading_costs
    )


def evaluate_agent(agent, env, n_episodes=5, seed=0, train=True):
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

    for ep in range(n_episodes):
        ep_seed = seed * 1000 + ep
        new_env = RealTradingEnvironment(seed=ep_seed)
        if not train:
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
        fitness = compute_fitness(metrics)
        total_fitness += fitness
        total_return += metrics["total_return"]
        total_sharpe += metrics["sharpe_ratio"]
        total_drawdown += metrics["max_drawdown"]
        total_trades += metrics["total_trades"]
        total_commissions += metrics["total_commissions"]
        total_slippage += metrics["total_slippage"]
        episode_returns.append(metrics["total_return"])

    return {
        "fitness": total_fitness / n_episodes,
        "mean_return": total_return / n_episodes,
        "mean_sharpe": total_sharpe / n_episodes,
        "mean_drawdown": total_drawdown / n_episodes,
        "mean_trades": total_trades / n_episodes,
        "mean_commissions": total_commissions / n_episodes,
        "mean_slippage": total_slippage / n_episodes,
        "std_return": np.std(episode_returns),
        "best_return": np.max(episode_returns),
        "worst_return": np.min(episode_returns),
    }


def main():
    parser = argparse.ArgumentParser(description="Train malecns trading agent on real market data")
    parser.add_argument("--graph", default="data/graph_w5", help="Connectome graph directory")
    parser.add_argument("--out", default="checkpoints/trading_real.pt", help="Output checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--population", type=int, default=32, help="Population size")
    parser.add_argument("--generations", type=int, default=100000, help="Number of generations")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--mutation", type=float, default=0.08, help="Initial mutation strength")
    parser.add_argument("--mutation-min", type=float, default=0.01, help="Minimum mutation strength")
    parser.add_argument("--mutation-max", type=float, default=0.3, help="Maximum mutation strength")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N generations")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N generations")
    parser.add_argument("--stagnation-window", type=int, default=50, help="Generations to detect stagnation")
    parser.add_argument("--early-stop-patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--val-interval", type=int, default=20, help="Validate every N generations")
    args = parser.parse_args()

    device = pick_device(args.device or "auto")
    print(f"Device: {device}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Episodes per eval: {args.episodes}")
    print(f"Initial mutation: {args.mutation}")
    print(f"Early stopping patience: {args.early_stop_patience}")
    print(f"Validation interval: {args.val_interval}")

    print("Loading connectome...")
    connectome = load_connectome(args.graph)
    print(f"  {connectome.n} neurons, {len(connectome.pre)} edges")

    print("Creating trading agent...")
    agent_cfg = AgentConfig(substeps=2)
    base_agent = TradingAgent(
        args.graph,
        checkpoint=args.resume,
        device=device,
        agent_config=agent_cfg,
        obs_features=REAL_OBS_FEATURES,
    )

    # Create validation environment
    val_env = RealTradingEnvironment()

    log_path = LOG_DIR / "train_trading_real.jsonl"
    log_file = open(log_path, "a")

    best_theta = base_agent.theta
    best_fitness = -np.inf
    best_return = -np.inf
    best_val_fitness = -np.inf
    generation = 0
    mutation_strength = args.mutation
    stagnation_counter = 0
    val_stagnation_counter = 0
    fitness_history = deque(maxlen=args.stagnation_window)

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if "generation" in state:
            generation = state["generation"]
            best_fitness = state.get("best_fitness", -np.inf)
            best_return = state.get("best_return", -np.inf)
            best_val_fitness = state.get("best_val_fitness", -np.inf)
            mutation_strength = state.get("mutation_strength", args.mutation)
        print(f"Resumed from generation {generation}")

    start_time = time.time()

    for gen in range(generation, args.generations):
        gen_start = time.time()
        population = []

        for i in range(args.population):
            agent = TradingAgent(
                args.graph,
                device=device,
                agent_config=agent_cfg,
                obs_features=REAL_OBS_FEATURES,
            )

            if i == 0:
                agent.theta = {k: v.clone() for k, v in best_theta.items()}
            else:
                agent.theta = {k: v.clone() for k, v in best_theta.items()}
                for name in agent.theta:
                    if agent.theta[name].dim() > 0:
                        noise = torch.distributions.cauchy.Cauchy(0, 1).sample(agent.theta[name].shape).to(agent.theta[name].device)
                        agent.theta[name] = agent.theta[name] + noise * mutation_strength

            fitness = evaluate_agent(agent, val_env, n_episodes=args.episodes, seed=gen * 100 + i, train=True)
            population.append((fitness, agent.theta))
            del agent

        population.sort(key=lambda x: x[0]["fitness"], reverse=True)

        best_fitness_gen = population[0][0]["fitness"]
        best_return_gen = population[0][0]["mean_return"]
        if best_fitness_gen > best_fitness:
            best_fitness = best_fitness_gen
            best_return = best_return_gen
            best_theta = population[0][1]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter > args.stagnation_window // 2:
            mutation_strength = min(mutation_strength * 1.2, args.mutation_max)
        elif best_fitness_gen > best_fitness * 0.99:
            mutation_strength = max(mutation_strength * 0.95, args.mutation_min)

        fitness_history.append(best_fitness_gen)

        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time

        # Validation (on test set) to detect overfitting
        val_fitness = None
        if gen % args.val_interval == 0:
            val_agent = TradingAgent(
                args.graph,
                device=device,
                agent_config=agent_cfg,
                obs_features=REAL_OBS_FEATURES,
            )
            val_agent.theta = {k: v.clone() for k, v in best_theta.items()}
            val_fitness = evaluate_agent(val_agent, val_env, n_episodes=args.episodes, seed=gen * 1000, train=False)
            del val_agent

            if val_fitness["fitness"] > best_val_fitness:
                best_val_fitness = val_fitness["fitness"]
                val_stagnation_counter = 0
            else:
                val_stagnation_counter += 1

        # Early stopping based on validation
        if val_stagnation_counter > args.early_stop_patience:
            print(f"\nEarly stopping at generation {gen} (validation stagnation: {val_stagnation_counter})")
            break

        if gen % args.log_interval == 0:
            gen_best = population[0][0]["mean_return"]
            gen_sharpe = population[0][0]["mean_sharpe"]
            gen_dd = population[0][0]["mean_drawdown"]
            val_str = f" | Val fit: {val_fitness['fitness']:8.4f} | Val ret: {val_fitness['mean_return']*100:8.2f}%" if val_fitness else ""
            print(f"Gen {gen:5d} | Best fit: {best_fitness:8.4f} | Best ret: {best_return*100:8.2f}% | Gen best: {gen_best*100:8.2f}% | Sharpe: {gen_sharpe:6.2f} | DD: {gen_dd*100:6.2f}% | Mut: {mutation_strength:.4f}{val_str} | {gen_time:.1f}s/gen")

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
                "val_stagnation": val_stagnation_counter,
                "gen_time": gen_time,
                "elapsed": elapsed,
                "timestamp": time.time(),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        if gen % args.save_interval == 0:
            checkpoint_path = Path(args.out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            base_agent.theta = best_theta
            base_agent.save_checkpoint(str(checkpoint_path), extra={
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "best_val_fitness": best_val_fitness,
                "mutation_strength": mutation_strength,
            })

            best_path = checkpoint_path.parent / "trading_real_best.pt"
            base_agent.save_checkpoint(str(best_path), extra={
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "best_val_fitness": best_val_fitness,
            })

        generation = gen + 1

    # Final save
    base_agent.theta = best_theta
    base_agent.save_checkpoint(args.out, extra={
        "generation": generation,
        "best_fitness": best_fitness,
        "best_return": best_return,
        "best_val_fitness": best_val_fitness,
        "mutation_strength": mutation_strength,
    })

    log_file.close()
    print(f"\nTraining complete. Best fitness: {best_fitness:.4f}, Best return: {best_return*100:.2f}%")
    print(f"Best validation fitness: {best_val_fitness:.4f}")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
