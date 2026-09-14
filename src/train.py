"""Continuous ES training of the MaleCNS connectome on the driving task.

Evolution strategies is used instead of backprop because the connectome is a
fixed, non-differentiable spiking substrate: only the interface parameters in
`ConnectomeAgent` (input gains, DN excitability, motor readout) are searched.

Headless by construction: no rendering, no pacing, no per-step host syncs. The
loop runs until stopped, checkpointing after every generation so it can be
killed and resumed without losing progress. `viewer.py` watches the checkpoint
from a separate process; `evaluate.py --record` captures a run for replay.

Curriculum: the road starts wide and episodes short; both tighten as the
population's mean lap fraction improves (see `CURRICULUM`). Every
`--eval-every` generations the ES mean is evaluated deterministically on all
start points and the best such result is kept in `checkpoints/best.pt`.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device, synchronize
from car_env import CarConfig, CarEnv, Track, build_centerline, monaco_config
from exploits import ExploitMonitor

STOP = False

# (road half-width multiplier, episode steps, laps_mean over the last
# `--curriculum-window` generations needed to advance)
CURRICULUM: tuple[tuple[float, int, float], ...] = (
    (1.6, 1500, 0.12),
    (1.25, 2000, 0.20),
    (1.0, 3000, float("inf")),
)


def _handle_stop(signum, frame) -> None:  # noqa: ANN001
    global STOP
    STOP = True
    print("\n[stop] finishing generation, then checkpointing", flush=True)


def rank_normalise(fitness: torch.Tensor) -> torch.Tensor:
    """Ranks mapped to [-0.5, 0.5]; ties share their mean rank.

    With plain argsort, a population whose members all score the same gets an
    arbitrary permutation of ranks, which is a random gradient of full size:
    that is how a flat landscape walked the readout into saturation.
    """
    n = fitness.numel()
    order = fitness.argsort()
    sorted_f = fitness[order]
    ranks = torch.empty(n, device=fitness.device)
    ranks[order] = torch.arange(n, device=fitness.device, dtype=torch.float32)
    # average ranks inside runs of equal fitness
    same_as_prev = torch.cat([torch.tensor([False], device=fitness.device), sorted_f[1:] == sorted_f[:-1]])
    group_start = torch.cumsum((~same_as_prev).long(), 0) - 1
    counts = torch.bincount(group_start, minlength=int(group_start.max()) + 1).float()
    firsts = torch.cumsum(counts, 0) - counts
    mean_rank = firsts + (counts - 1) / 2
    tied = mean_rank[group_start]
    ranks[order] = tied
    if n == 1:
        return torch.zeros_like(fitness)
    return ranks / (n - 1) - 0.5


def rollout(
    agent: ConnectomeAgent,
    env: CarEnv,
    theta: dict[str, torch.Tensor],
    steps: int,
    seed: int,
    poll_every: int = 25,
    monitor: bool = True,
) -> dict[str, torch.Tensor | float | int]:
    """Run one episode for the whole population. Returns fitness and telemetry."""
    agent.seed(seed)
    obs = env.reset()
    agent.reset()
    watch = ExploitMonitor(env) if monitor else None
    alive = torch.ones(env.batch, device=env.device)
    fitness = torch.zeros(env.batch, device=env.device)
    steps_alive = torch.zeros(env.batch, device=env.device)
    speed_sum = torch.zeros(env.batch, device=env.device)
    vis_hz = torch.zeros((), device=env.device)
    ran = 0
    with torch.inference_mode():
        for step in range(steps):
            action = agent.act(obs, theta)
            obs, reward, done = env.step(action)
            if watch is not None:
                watch.observe(reward, alive.bool())
            fitness = fitness + reward * alive
            steps_alive = steps_alive + alive
            speed_sum = speed_sum + env.speed * alive
            vis_hz = vis_hz + agent.sensory_rates(obs, theta)[:, : agent.cfg.n_rays].mean()
            alive = alive * (~done).float()
            ran = step + 1
            # Reading alive.sum() is a GPU sync that stalls the command queue;
            # polling every 25 steps wastes at most 24 steps of dead-population
            # compute instead of stalling every step.
            if step % poll_every == poll_every - 1 and float(alive.sum()) == 0:
                break
    return {
        "fitness": fitness,
        "laps": env.laps.clone(),
        "steps_alive": steps_alive,
        "speed_mean": float((speed_sum / steps_alive.clamp(min=1)).mean()),
        "dn_hz": float(agent.dn_rate_hz.mean()),
        "vis_hz": float(vis_hz / max(1, ran)),
        "steps_run": ran,
        "exploits": watch.report() if watch is not None else None,
        **env.telemetry(),
    }


class TrackBank:
    """Rasterised tracks per curriculum stage, built on first use."""

    def __init__(self, base_cfg: CarConfig, layout: str, n_tracks: int, device: torch.device):
        self.base_cfg = base_cfg
        self.layout = layout
        self.n_tracks = n_tracks
        self.device = device
        self._cache: dict[tuple[int, int], tuple[CarConfig, Track]] = {}

    def get(self, stage: int, index: int = 0) -> tuple[CarConfig, Track]:
        width_mult = CURRICULUM[stage][0]
        key = (stage, index if self.layout == "loop" else 0)
        if key not in self._cache:
            cfg = replace(self.base_cfg, track_halfwidth=self.base_cfg.track_halfwidth * width_mult)
            seed = index if self.layout == "loop" else 0
            self._cache[key] = (cfg, Track(build_centerline(cfg, seed), cfg, self.device))
        return self._cache[key]


def make_env(bank: TrackBank, stage: int, generation: int, popsize: int, starts_per_gen: int) -> CarEnv:
    """Environment for one generation: `starts_per_gen` start points per member."""
    if bank.layout == "monaco":
        cfg, track = bank.get(stage)
        base = generation % bank.n_tracks
        fractions = [((base + k) % bank.n_tracks) / bank.n_tracks for k in range(starts_per_gen)]
    else:
        cfg, track = bank.get(stage, generation % bank.n_tracks)
        fractions = [k / starts_per_gen for k in range(starts_per_gen)]
    per_car = torch.tensor(fractions).repeat(popsize)
    return CarEnv(popsize * starts_per_gen, bank.device, cfg, track=track, start_fraction=per_car)


def make_eval_env(bank: TrackBank, stage: int, batch: int) -> CarEnv:
    """Every start point, cycled over the whole batch, on the stage's track."""
    cfg, track = bank.get(stage)
    fractions = torch.tensor([(b % bank.n_tracks) / bank.n_tracks for b in range(batch)])
    return CarEnv(batch, bank.device, cfg, track=track, start_fraction=fractions)


def evaluate_mean(
    agent: ConnectomeAgent, bank: TrackBank, mu: torch.Tensor, stage: int, steps: int, seed: int
) -> dict[str, float]:
    batch = agent.brain.batch
    env = make_eval_env(bank, stage, batch)
    theta = agent.unpack(mu.unsqueeze(0).repeat(batch, 1))
    result = rollout(agent, env, theta, steps, seed)
    starts = torch.arange(batch, device=env.device) % bank.n_tracks
    per_start_fit = torch.stack([result["fitness"][starts == s].mean() for s in range(bank.n_tracks)])
    per_start_laps = torch.stack([result["laps"][starts == s].mean() for s in range(bank.n_tracks)])
    return {
        "eval_fitness": float(per_start_fit.mean()),
        "eval_fitness_min": float(per_start_fit.min()),
        "eval_laps": float(per_start_laps.mean()),
        "eval_laps_min": float(per_start_laps.min()),
        "eval_crash": int(result["crash"]),
        "eval_stuck": int(result["stuck"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument(
        "--popsize",
        type=int,
        default=128,
        help="per-body cost of the Metal step falls to 128 bodies (26 us/body vs 32 at 64) and the ES gradient gets 2x the samples",
    )
    parser.add_argument(
        "--starts-per-gen",
        type=int,
        default=1,
        help="start points each member is scored on per generation (batch = popsize x this)",
    )
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="0.9 let the mean drift into saturation on flat fitness; 0.5 holds",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=0,
        help="0 follows the curriculum (1500 -> 3000); otherwise a fixed cap",
    )
    parser.add_argument("--substeps", type=int, default=defaults.SUBSTEPS)
    parser.add_argument("--dt-ms", type=float, default=defaults.DT_MS)
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=defaults.WEIGHT_SCALE,
        help="global synaptic scaling; 0.15 keeps the network out of runaway",
    )
    parser.add_argument("--adapt-mv", type=float, default=defaults.ADAPT_MV)
    parser.add_argument(
        "--layout",
        default="monaco",
        choices=("monaco", "loop"),
        help="monaco: the real circuit (or any --geojson) at full scale; loop: procedural circuits",
    )
    parser.add_argument("--geojson", default="data/tracks/monaco.geojson", help="circuit centerline for --layout monaco")
    parser.add_argument(
        "--tracks",
        type=int,
        default=defaults.MONACO_STARTS,
        help="loop: number of random circuits; monaco: start points around the lap",
    )
    parser.add_argument("--no-curriculum", action="store_true", help="train at the final stage only")
    parser.add_argument("--curriculum-window", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=10, help="deterministic evaluation of the mean; 0 disables")
    parser.add_argument("--seed", type=int, default=0, help="perturbations and sensory noise are seeded from this")
    parser.add_argument("--no-exploit-monitor", action="store_true", help="skip the per-step exploit detector")
    parser.add_argument(
        "--reset-blocks",
        default="",
        help="comma-separated parameter blocks to restart from init on resume, e.g. w_out,b_out for a pinned readout",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/es.pt", type=Path)
    parser.add_argument("--best", default="checkpoints/best.pt", type=Path)
    parser.add_argument("--log", default="logs/train.jsonl", type=Path)
    parser.add_argument(
        "--generations", type=int, default=0, help="0 runs until interrupted"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.popsize % 2:
        raise SystemExit("--popsize must be even (antithetic sampling)")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    device = pick_device(args.device)
    connectome = load_connectome(args.graph)
    neurons: pd.DataFrame = connectome.neurons
    print(
        f"connectome {connectome.n:,} neurons / {len(connectome.pre):,} edges "
        f"on {device}"
    )

    batch = args.popsize * args.starts_per_gen
    brain = Brain(
        connectome,
        batch=batch,
        config=LIFConfig(dt_ms=args.dt_ms, adapt_mv=args.adapt_mv),
        device=device,
        weight_scale=args.weight_scale,
    )
    agent_cfg = AgentConfig(substeps=args.substeps)
    agent = ConnectomeAgent(brain, neurons, agent_cfg)
    dt_s = defaults.control_dt_s(args.dt_ms, args.substeps)
    if args.layout == "monaco":
        base_cfg = replace(monaco_config(dt_s), geojson_path=args.geojson)
    else:
        base_cfg = CarConfig(dt_s=dt_s)
    bank = TrackBank(base_cfg, args.layout, args.tracks, device)
    _, track0 = bank.get(len(CURRICULUM) - 1)
    print(
        f"circuit {track0.length_m:,.0f} m, {2 * base_cfg.track_halfwidth:.0f} m wide, "
        f"tightest {track0.min_radius_m:.1f} m vs car {base_cfg.min_turn_radius:.1f} m; "
        f"params {agent.n_params:,}; bodies {batch}; control step {dt_s * 1000:.0f} ms"
    )

    mu = agent.initial_params().to(device)
    momentum = torch.zeros_like(mu)
    generation = 0
    stage = len(CURRICULUM) - 1 if args.no_curriculum else 0
    best_eval = -float("inf")
    recent_laps: list[float] = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device)
        generation = int(state["generation"])
        try:
            reset = tuple(b for b in args.reset_blocks.split(",") if b)
            mu_cpu, momentum_cpu, notes = agent.migrate_state(state, reset=reset)
            mu = agent.clamp_params(mu_cpu.to(device).unsqueeze(0))[0]
            momentum = momentum_cpu.to(device)
            print(f"resumed at generation {generation}" + (f"; migrated: {', '.join(notes)}" if notes else ""))
        except ValueError as exc:
            print(f"{exc}: parameters restart from init, generation counter continues at {generation}")
        stage = int(state.get("stage", stage)) if not args.no_curriculum else stage
        best_eval = float(state.get("best_eval", best_eval))
        recent_laps = list(state.get("recent_laps", []))

    def save(path: Path) -> None:
        torch.save(
            {
                "mu": mu.cpu(),
                "momentum": momentum.cpu(),
                "generation": generation,
                "stage": stage,
                "best_eval": best_eval,
                "recent_laps": recent_laps[-args.curriculum_window :],
                "n_params": agent.n_params,
                "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()},
                "agent_cfg": asdict(agent_cfg),
                "car_cfg": asdict(bank.get(stage)[0]),
                "layout": args.layout,
                "starts": args.tracks,
                # str() everything: Path objects make torch.load refuse the
                # file under the default weights_only=True.
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            path,
        )

    half = args.popsize // 2
    sample_gen = torch.Generator().manual_seed(args.seed)
    while not STOP and (args.generations == 0 or generation < args.generations):
        started = time.time()
        steps = args.episode_steps or CURRICULUM[stage][1]
        sample_gen.manual_seed(args.seed * 1_000_003 + generation)
        eps = torch.randn(half, mu.numel(), generator=sample_gen).to(device)
        perturb = torch.cat([eps, -eps], dim=0)
        params = agent.clamp_params(mu.unsqueeze(0) + args.sigma * perturb)
        theta = agent.unpack(params.repeat_interleave(args.starts_per_gen, dim=0))

        env = make_env(bank, stage, generation, args.popsize, args.starts_per_gen)
        result = rollout(agent, env, theta, steps, seed=args.seed * 7919 + generation, monitor=not args.no_exploit_monitor)
        fitness = result["fitness"].view(args.popsize, args.starts_per_gen).mean(1)
        laps = result["laps"].view(args.popsize, args.starts_per_gen).mean(1)

        advantage = rank_normalise(fitness)
        grad = (perturb * advantage.unsqueeze(1)).sum(0) / (args.popsize * args.sigma)
        momentum = args.momentum * momentum + grad
        mu = agent.clamp_params((mu + args.lr * momentum).unsqueeze(0))[0]

        generation += 1
        recent_laps.append(float(laps.mean()))
        recent_laps = recent_laps[-args.curriculum_window :]
        record = {
            "generation": generation,
            "stage": stage,
            "steps": steps,
            "starts_per_gen": args.starts_per_gen,
            "track": (generation - 1) % args.tracks,
            "fitness_mean": float(fitness.mean()),
            "fitness_best": float(fitness.max()),
            "fitness_std": float(fitness.std()),
            "laps_best": float(laps.max()),
            "laps_mean": float(laps.mean()),
            "steps_alive_mean": float(result["steps_alive"].mean()),
            "speed_mean": result["speed_mean"],
            "lat_g_max": result["lat_g_max"],
            "steer_abs_mean": result["steer_abs_mean"],
            "crash": result["crash"],
            "reverse": result["reverse"],
            "stuck": result["stuck"],
            "dn_hz": result["dn_hz"],
            "vis_hz": result["vis_hz"],
            "exploits": result["exploits"],
            "at_bounds": agent.fraction_at_bounds(mu),
            "momentum_norm": float(momentum.norm()),
            "sigma": args.sigma,
        }

        # Curriculum: advance when the rolling mean lap fraction clears the bar.
        if (
            not args.no_curriculum
            and stage < len(CURRICULUM) - 1
            and len(recent_laps) >= args.curriculum_window
            and float(np.mean(recent_laps)) >= CURRICULUM[stage][2]
        ):
            stage += 1
            recent_laps = []
            record["stage_advanced_to"] = stage
            print(f"[curriculum] stage {stage}: road x{CURRICULUM[stage][0]}, {CURRICULUM[stage][1]} steps")

        if args.eval_every and generation % args.eval_every == 0:
            eval_steps = args.episode_steps or CURRICULUM[stage][1]
            record.update(evaluate_mean(agent, bank, mu, stage, eval_steps, seed=args.seed))
            if record["eval_fitness"] > best_eval:
                best_eval = record["eval_fitness"]
                record["new_best"] = True
                save(args.best)

        synchronize(device)
        elapsed = time.time() - started
        record["seconds"] = round(elapsed, 2)
        record["body_steps_per_s"] = round(batch * result["steps_run"] / max(elapsed, 1e-6))
        with args.log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"gen {generation:5d} s{stage} | fit {record['fitness_mean']:8.2f} "
            f"best {record['fitness_best']:8.2f} | laps {record['laps_mean']:.3f}/{record['laps_best']:.3f} "
            f"| alive {record['steps_alive_mean']:6.0f}/{steps} | {record['speed_mean'] * 3.6:5.0f} km/h "
            f"| c{record['crash']} r{record['reverse']} s{record['stuck']} "
            f"| {record['seconds']:.1f}s {record['body_steps_per_s']:,} body-steps/s"
            + (f" | eval {record['eval_fitness']:.1f} laps {record['eval_laps']:.3f}" if "eval_fitness" in record else ""),
            flush=True,
        )
        if result["exploits"] and result["exploits"]["flagged_cars"]:
            print(f"      {ExploitMonitor.describe(result['exploits'])}", flush=True)
        save(args.checkpoint)

    print(f"[done] generation {generation}, checkpoint {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
