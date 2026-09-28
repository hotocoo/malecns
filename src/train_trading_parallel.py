"""Parallel evolutionary strategy training for the malecns trading agent.

Uses multiprocessing to evaluate multiple agents in parallel.
Tensors are moved to CPU for inter-process communication.
"""

from __future__ import annotations

import argparse
import json
import time
import multiprocessing as mp
from pathlib import Path
from collections import deque

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentConfig
from brain import LIFConfig, load_connectome, pick_device
from trading_env import TradingEnvironment, TradeAction, TradingConfig

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


def evaluate_agent_worker(args):
    """Worker function for parallel evaluation."""
    graph_dir, theta_cpu, n_episodes, seed, episode_length, substeps, device_str = args
    
    # Load connectome and create brain in this process
    connectome = load_connectome(graph_dir)
    device = pick_device(device_str)
    
    from brain import Brain
    brain = Brain(connectome, batch=1, config=LIFConfig(), device=device)
    
    from agent import ConnectomeAgent
    agent_cfg = AgentConfig(substeps=substeps)
    agent = ConnectomeAgent(brain, connectome.neurons, agent_cfg)
    
    # Move theta to device
    theta = {}
    for name, value in theta_cpu.items():
        theta[name] = value.to(device)
    agent.theta = theta
    
    # Evaluate
    trading_cfg = TradingConfig(steps_per_episode=episode_length)
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
        env = TradingEnvironment(trading_cfg, seed=ep_seed)
        obs = env.reset()
        agent.reset(batch=1)
        
        done = False
        while not done:
            # Convert obs to tensor
            obs_tensor = torch.tensor([obs["price_change"], obs["price_change_5"], obs["price_change_20"],
                                       obs["price_normalized"], obs["position_pct"], obs["equity_normalized"],
                                       obs["unrealized_pnl"], obs["rsi"], obs["macd"], obs["macd_line"],
                                       obs["bb_position"], obs["bb_width"], obs["atr"], obs["momentum_5"],
                                       obs["realized_vol"], obs["spread"], obs["drawdown"], obs["max_drawdown"],
                                       obs["volatility"], 0.5], dtype=torch.float32, device=device).unsqueeze(0)
            output = agent.act(obs_tensor, agent.theta)
            
            # Convert output to action
            steer = output[0, 0].item()
            throttle = output[0, 1].item()
            size = min(3, max(1, int(abs(throttle) * 3) + 1))
            if steer > 0.1:
                if size == 1: action = TradeAction.BUY_SMALL
                elif size == 2: action = TradeAction.BUY_MEDIUM
                else: action = TradeAction.BUY_LARGE
            elif steer < -0.1:
                if size == 1: action = TradeAction.SELL_SMALL
                elif size == 2: action = TradeAction.SELL_MEDIUM
                else: action = TradeAction.SELL_LARGE
            else:
                action = TradeAction.HOLD
            
            obs, reward, done, info = env.step(action)
        
        metrics = env.get_performance_metrics()
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
        "std_return": float(np.std(episode_returns)),
        "best_return": float(np.max(episode_returns)),
        "worst_return": float(np.min(episode_returns)),
    }


def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Train malecns trading agent (parallel)")
    parser.add_argument("--graph", default="data/graph_w5", help="Connectome graph directory")
    parser.add_argument("--out", default="checkpoints/trading_es_parallel.pt", help="Output checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--population", type=int, default=32, help="Population size")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per evaluation")
    parser.add_argument("--episode-length", type=int, default=100, help="Steps per episode")
    parser.add_argument("--substeps", type=int, default=2, help="Brain substeps per decision")
    parser.add_argument("--mutation", type=float, default=0.05, help="Initial mutation strength")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N generations")
    parser.add_argument("--save-interval", type=int, default=25, help="Save checkpoint every N generations")
    parser.add_argument("--stagnation-window", type=int, default=50, help="Generations to detect stagnation")
    args = parser.parse_args()

    device = pick_device(args.device or "auto")
    print(f"Device: {device}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Episodes per eval: {args.episodes}")
    print(f"Episode length: {args.episode_length}")
    print(f"Workers: {args.workers}")
    print(f"Initial mutation: {args.mutation}")

    # Load connectome
    print("Loading connectome...")
    connectome = load_connectome(args.graph)
    print(f"  {connectome.n} neurons, {len(connectome.pre)} edges")

    # Create base agent
    print("Creating trading agent...")
    from brain import Brain
    brain = Brain(connectome, batch=1, config=LIFConfig(), device=device)
    from agent import ConnectomeAgent
    agent_cfg = AgentConfig(substeps=args.substeps)
    base_agent = ConnectomeAgent(brain, connectome.neurons, agent_cfg)
    base_theta = base_agent.initial_params().unsqueeze(0)
    base_theta = base_agent.unpack(base_theta)
    base_theta = {k: v.to(device) for k, v in base_theta.items()}

    # Training log
    log_path = LOG_DIR / "train_trading_parallel.jsonl"
    log_file = open(log_path, "a")

    # ES state
    best_theta = base_theta
    best_fitness = -np.inf
    best_return = -np.inf
    generation = 0
    mutation_strength = args.mutation

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
        
        # Create population with mutations
        population_thetas = []
        for i in range(args.population):
            if i == 0:
                # Elitism: keep best
                theta = {k: v.clone() for k, v in best_theta.items()}
            else:
                # Mutate best
                theta = {}
                for name in best_theta:
                    if best_theta[name].dim() > 0:
                        noise = torch.distributions.cauchy.Cauchy(0, 1).sample(best_theta[name].shape).to(best_theta[name].device)
                        theta[name] = best_theta[name] + noise * mutation_strength
                    else:
                        theta[name] = best_theta[name].clone()
            population_thetas.append(theta)
        
        # Move thetas to CPU for inter-process communication
        population_thetas_cpu = []
        for theta in population_thetas:
            theta_cpu = {k: v.cpu() for k, v in theta.items()}
            population_thetas_cpu.append(theta_cpu)
        
        # Evaluate in parallel
        eval_args = []
        for i, theta_cpu in enumerate(population_thetas_cpu):
            eval_args.append((args.graph, theta_cpu, args.episodes, gen * 100 + i, args.episode_length, args.substeps, str(device)))
        
        with mp.Pool(args.workers) as pool:
            results = pool.map(evaluate_agent_worker, eval_args)
        
        # Sort by fitness
        population = list(zip(results, population_thetas))
        population.sort(key=lambda x: x[0]["fitness"], reverse=True)
        
        # Update best
        best_fitness_gen = population[0][0]["fitness"]
        best_return_gen = population[0][0]["mean_return"]
        if best_fitness_gen > best_fitness:
            best_fitness = best_fitness_gen
            best_return = best_return_gen
            best_theta = population[0][1]
        
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
                "mutation_strength": mutation_strength,
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
                  f"{gen_time:.1f}s/gen")

        # Save checkpoint periodically
        if gen % args.save_interval == 0:
            checkpoint_path = Path(args.out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            for name in best_theta:
                params.append(best_theta[name].cpu())
            torch.save({
                "mu": params,
                "generation": gen,
                "best_fitness": best_fitness,
                "best_return": best_return,
                "mutation_strength": mutation_strength,
            }, str(checkpoint_path))

        generation = gen + 1

    # Final save
    params = []
    for name in best_theta:
        params.append(best_theta[name].cpu())
    torch.save({
        "mu": params,
        "generation": generation,
        "best_fitness": best_fitness,
        "best_return": best_return,
        "mutation_strength": mutation_strength,
    }, args.out)

    log_file.close()
    print(f"\nTraining complete. Best fitness: {best_fitness:.4f}, Best return: {best_return*100:.2f}%")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
