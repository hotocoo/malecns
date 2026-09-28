"""Improved evolutionary strategy training for the malecns trading agent.

Key improvements over v1:
- Longer episodes (252 steps = 1 year of daily data)
- Adaptive mutation strength based on population diversity
- Multi-objective fitness (return + Sharpe - drawdown)
- Better checkpointing and logging
- Stagnation detection with automatic mutation increase
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
from trading_agent import TradingAgent
from trading_env import TradingEnvironment, TradingConfig

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
    
    # Only reward positive Sharpe
    sharpe_term = W_SHARPE * sharpe if sharpe > 0 else 0.0
    
    return (
        W_RETURN * total_return
        + sharpe_term
        + W_DRAWDOWN * max_drawdown
        + W_TRADING_COST * trading_costs
    )


def evaluate_agent(agent, n_episodes=5, seed=0):
    """Evaluate an agent over multiple episodes with different market conditions."""
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
        new_env = TradingEnvironment(agent.trading_config, seed=ep_seed)
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
    parser = argparse.ArgumentParser(description="Train malecns trading agent (v2)")
    parser.add_argument("--graph", default="data/graph_w5", help="Connectome graph directory")
    parser.add_argument("--out", default="checkpoints/trading_es_v2.pt", help="Output checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--population", type=int, default=48, help="Population size")
    parser.add_argument("--generations", type=int, default=2000, help="Number of generations")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--episode-length", type=int, default=252, help="Steps per episode")
    parser.add_argument("--substeps", type=int, default=2, help="Brain substeps per decision")
    parser.add_argument("--mutation", type=float, default=0.05, help="Initial mutation strength")
    parser.add_argument("--mutation-min", type=float, default=0.01, help="Minimum mutation strength")
    parser.add_argument("--mutation-max", type=float, default=0.2, help="Maximum mutation strength")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N generations")
    parser.add_argument("--save-interval", type=int, default=25, help="Save checkpoint every N generations")
    parser.add_argument("--stagnation-window", type=int, default=50, help="Generations to detect stagnation")
    parser.add_argument("--early-stop-patience", type=int, default=200, help="Early stopping patience (generations without improvement)")
    args = parser.parse_args()

    device = pick_device(args.device or "auto")
    print(f"Device: {device}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Episodes per eval: {args.episodes}")
    print(f"Episode length: {args.episode_length}")
    print(f"Initial mutation: {args.mutation}")

    # Load connectome
    print("Loading connectome...")
    connectome = load_connectome(args.graph)
    print(f"  {connectome.n} neurons, {len(connectome.pre)} edges")

    # Create base agent
    print("Creating trading agent...")
    agent_cfg = AgentConfig(substeps=args.substeps)
    trading_cfg = TradingConfig(steps_per_episode=args.episode_length)
    base_agent = TradingAgent(
        args.graph,
        checkpoint=args.resume,
        device=device,
        agent_config=agent_cfg,
        trading_config=trading_cfg,
    )

    # Training log
    log_path = LOG_DIR / "train_trading_v2.jsonl"
    log_file = open(log_path, "a")

    # ES state
    best_theta = base_agent.theta
    best_fitness = -np.inf
    best_return = -np.inf
    generation = 0
    mutation_strength = args.mutation
    stagnation_counter = 0
    fitness_history = deque(maxlen=args.stagnation_window)

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if "generation" in state:
            generation = state["generation"]
            best_fitness = state.get("best_fitness", -np.inf)
            best_return = state.get("best_return", -np.inf)
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
                trading_config=trading_cfg,
            )

            if i == 0:
                # Elitism: keep best
                agent.theta = {k: v.clone() for k, v in best_theta.items()}
            else:
                # Copy best, then mutate with adaptive strength
                agent.theta = {k: v.clone() for k, v in best_theta.items()}
                for name in agent.theta:
                    if agent.theta[name].dim() > 0:
                        # Use Cauchy distribution for heavier tails
                        noise = torch.distributions.cauchy.Cauchy(0, 1).sample(agent.theta[name].shape).to(agent.theta[name].device)
                        agent.theta[name] = agent.theta[name] + noise * mutation_strength

            # Evaluate
            fitness = evaluate_agent(agent, n_episodes=args.episodes, seed=gen * 100 + i)
            population.append((fitness, agent.theta))

            del agent  # Free memory

        # Sort by fitness
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

        # Adaptive mutation: increase if stagnating, decrease if improving
        if stagnation_counter > args.stagnation_window // 2:
            mutation_strength = min(mutation_strength * 1.2, args.mutation_max)
        elif best_fitness_gen > best_fitness * 0.99:
            mutation_strength = max(mutation_strength * 0.95, args.mutation_min)

        fitness_history.append(best_fitness_gen)

        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time

        # Log
        if gen % args.log_interval == 0:
            log_entry = {
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "gen_best_fitness": population[0][0]["fitness"],
                "gen_best_return": population[0][0]["mean_return"],
                "gen_best_sharpe": population[0][0]["mean_sharpe"],
                "gen_best_drawdown": population[0][0]["mean_drawdown"],
                "gen_worst_fitness": population[-1][0]["fitness"],
                "gen_worst_return": population[-1][0]["mean_return"],
                "mutation_strength": mutation_strength,
                "stagnation_counter": stagnation_counter,
                "gen_time": gen_time,
                "elapsed": elapsed,
                "timestamp": time.time(),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            print(f"Gen {gen:5d} | Best fit: {best_fitness:8.4f} | "
                  f"Best ret: {best_return*100:8.2f}% | "
                  f"Gen best: {population[0][0]['mean_return']*100:8.2f}% | "
                  f"Sharpe: {population[0][0]['mean_sharpe']:6.2f} | "
                  f"DD: {population[0][0]['mean_drawdown']*100:6.2f}% | "
                  f"Mut: {mutation_strength:.4f} | "
                  f"{gen_time:.1f}s/gen")

        # Save checkpoint periodically
        if gen % args.save_interval == 0:
            checkpoint_path = Path(args.out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            base_agent.theta = best_theta
            base_agent.save_checkpoint(str(checkpoint_path), extra={
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "mutation_strength": mutation_strength,
            })

            # Also save best checkpoint
            best_path = checkpoint_path.parent / "trading_best_v2.pt"
            base_agent.save_checkpoint(str(best_path), extra={
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
            })

        # Early stopping
        if stagnation_counter > args.early_stop_patience:
            print(f"\nEarly stopping at generation {gen} (no improvement for {stagnation_counter} generations)")
            break

        generation = gen + 1

    # Final save
    base_agent.theta = best_theta
    base_agent.save_checkpoint(args.out, extra={
        "generation": generation,
        "best_fitness": best_fitness,
        "best_return": best_return,
        "mutation_strength": mutation_strength,
    })

    log_file.close()
    print(f"\nTraining complete. Best fitness: {best_fitness:.4f}, Best return: {best_return*100:.2f}%")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
