"""Evolutionary strategy training for the malecns trading agent.

Usage:
    python3 src/train_trading.py --graph data/graph_w5 --out checkpoints/trading_es.pt
    python3 src/train_trading.py --graph data/graph_w5 --resume checkpoints/trading_es.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentConfig
from brain import LIFConfig, load_connectome, pick_device
from trading_agent import TradingAgent


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def evaluate_agent(agent: TradingAgent, n_episodes: int = 3, seed: int = 0) -> dict:
    """Evaluate an agent over multiple episodes with different market conditions."""
    from trading_env import TradingEnvironment, TradingConfig
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    total_return = 0.0
    total_sharpe = 0.0
    total_drawdown = 0.0
    total_trades = 0
    episode_returns = []

    for ep in range(n_episodes):
        # Use different seed for each episode to test robustness
        ep_seed = seed * 1000 + ep
        new_env = TradingEnvironment(agent.trading_config, seed=ep_seed)
        agent.env = new_env
        agent.obs = new_env.reset()
        agent.total_reward = 0.0
        
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.act(agent.obs)
            obs, reward, done, info = agent.env.step(action)
            agent.obs = obs
            episode_reward += reward

        total_reward += episode_reward
        metrics = agent.get_performance()
        total_return += metrics["total_return"]
        total_sharpe += metrics["sharpe_ratio"]
        total_drawdown += metrics["max_drawdown"]
        total_trades += metrics["total_trades"]
        episode_returns.append(metrics["total_return"])

    return {
        "mean_reward": total_reward / n_episodes,
        "mean_return": total_return / n_episodes,
        "mean_sharpe": total_sharpe / n_episodes,
        "mean_drawdown": total_drawdown / n_episodes,
        "mean_trades": total_trades / n_episodes,
        "std_return": np.std(episode_returns),
        "best_return": np.max(episode_returns),
        "worst_return": np.min(episode_returns),
    }


def main():
    parser = argparse.ArgumentParser(description="Train malecns trading agent")
    parser.add_argument("--graph", default="data/graph_w5", help="Connectome graph directory")
    parser.add_argument("--out", default="checkpoints/trading_es.pt", help="Output checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--population", type=int, default=32, help="Population size")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per evaluation")
    parser.add_argument("--episode-length", type=int, default=100, help="Steps per episode")
    parser.add_argument("--substeps", type=int, default=2, help="Brain substeps per decision")
    parser.add_argument("--mutation", type=float, default=0.1, help="Mutation strength")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N generations")
    args = parser.parse_args()

    device = pick_device(args.device or "auto")
    print(f"Device: {device}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Episodes per eval: {args.episodes}")

    # Load connectome
    print("Loading connectome...")
    connectome = load_connectome(args.graph)
    print(f"  {connectome.n} neurons, {len(connectome.pre)} edges")

    # Create base agent
    print("Creating trading agent...")
    from agent import AgentConfig
    from trading_env import TradingConfig
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
    log_path = LOG_DIR / "train_trading.jsonl"
    log_file = open(log_path, "a")

    # ES state
    best_theta = base_agent.theta
    best_fitness = -np.inf
    generation = 0

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if "generation" in state:
            generation = state["generation"]
            best_fitness = state.get("best_fitness", -np.inf)
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
                agent.theta = best_theta
            else:
                # Mutate best
                for name in agent.theta:
                    if agent.theta[name].dim() > 0:
                        noise = torch.randn_like(agent.theta[name]) * args.mutation
                        agent.theta[name] = agent.theta[name] + noise

            # Evaluate
            fitness = evaluate_agent(agent, n_episodes=args.episodes, seed=gen * 100 + i)
            population.append((fitness, agent.theta))

            del agent  # Free memory

        # Sort by fitness (mean return)
        population.sort(key=lambda x: x[0]["mean_return"], reverse=True)

        # Update best
        best_fitness_gen = population[0][0]["mean_return"]
        if best_fitness_gen > best_fitness:
            best_fitness = best_fitness_gen
            best_theta = population[0][1]

        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time

        # Log
        if gen % args.log_interval == 0:
            log_entry = {
                "generation": gen,
                "best_return": best_fitness,
                "gen_best_return": population[0][0]["mean_return"],
                "gen_best_sharpe": population[0][0]["mean_sharpe"],
                "gen_best_drawdown": population[0][0]["mean_drawdown"],
                "gen_worst_return": population[-1][0]["mean_return"],
                "gen_time": gen_time,
                "elapsed": elapsed,
                "timestamp": time.time(),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            print(f"Gen {gen:5d} | Best: {best_fitness*100:8.2f}% | "
                  f"Gen best: {population[0][0]['mean_return']*100:8.2f}% | "
                  f"Sharpe: {population[0][0]['mean_sharpe']:6.2f} | "
                  f"DD: {population[0][0]['mean_drawdown']*100:6.2f}% | "
                  f"{gen_time:.1f}s/gen")

        # Save checkpoint periodically
        if gen % 50 == 0:
            checkpoint_path = Path(args.out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            base_agent.theta = best_theta
            base_agent.save_checkpoint(str(checkpoint_path), extra={
                "generation": gen,
                "best_fitness": best_fitness,
            })

        generation = gen + 1

    # Final save
    base_agent.theta = best_theta
    base_agent.save_checkpoint(args.out, extra={
        "generation": generation,
        "best_fitness": best_fitness,
    })

    log_file.close()
    print(f"\nTraining complete. Best return: {best_fitness*100:.2f}%")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
