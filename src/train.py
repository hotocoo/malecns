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
from dataclasses import asdict, fields as dataclass_fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device, synchronize
from car_env import DONE_NAMES, CarConfig, CarEnv, Track, build_centerline, monaco_config
from exploits import ExploitMonitor

STOP = False

# Alive-body compaction (see `rollout`): once the surviving fraction of the
# batch drops to COMPACT_BELOW, the finished bodies are dropped from the brain,
# the environment and the exploit monitor; at least COMPACT_MIN_DROP bodies
# (one 32-body kernel word) must go for the re-pack to pay for itself.
COMPACT_BELOW = 0.75
COMPACT_MIN_DROP = 32

# (road half-width multiplier, optional episode step cap, laps_mean over the
# last `--curriculum-window` generations needed to advance)
# By default episodes are uncapped (`--episode-steps -1`): a car drives until
# it crashes, stalls, reverses or completes `CarConfig.max_laps`, so fitness
# is bounded by driving, not by a clock. The step caps here apply only with
# `--episode-steps 0`, the old fixed-horizon mode.
CURRICULUM: tuple[tuple[float, int, float], ...] = (
    (1.6, 2000, 0.10),
    (1.25, 3000, 0.20),
    (1.0, 6000, float("inf")),
)


def episode_cap(episode_steps: int, stage: int) -> int:
    """Step cap for one generation: 0 means uncapped (see `--episode-steps`)."""
    if episode_steps < 0:
        return 0
    return episode_steps or CURRICULUM[stage][1]


def _handle_stop(signum, frame) -> None:  # noqa: ANN001
    global STOP
    STOP = True
    print("\n[stop] finishing generation, then checkpointing", flush=True)


def rank_normalise(fitness: torch.Tensor, tie_tol: float = 0.0, flat_threshold: float = 0.0) -> torch.Tensor:
    """Ranks mapped to [-0.5, 0.5]; ties share their mean rank.

    Members whose fitness differs by no more than `tie_tol` from the next one
    in sorted order are tied. With every member crashing on the same step the
    remaining differences are progress noise below one step's time tax, and
    ranking them anyway turned that noise into a full-size update: the mean
    then random-walked on a flat landscape instead of holding still.

    With plain argsort, a population whose members all score the same gets an
    arbitrary permutation of ranks, which is a random gradient of full size:
    that is how a flat landscape walked the readout into saturation.

    `flat_threshold` is a separate, larger gate: if the entire population's
    fitness range is below this, the landscape is effectively flat and no
    selection gradient should be generated, even if individual differences
    exceed `tie_tol`. This catches the collapse mode where all cars crash
    at the same step and the remaining spread is just progress noise.
    """
    n = fitness.numel()
    order = fitness.argsort()
    sorted_f = fitness[order]
    # A completely flat landscape should not generate a selection gradient.
    # Adjacent ties alone are insufficient because a narrow crash landscape can
    # still accumulate tiny ordering differences across the whole population.
    # Treat the entire population as non-informative when its total fitness
    # range is below the tie threshold OR the flatness threshold.
    spread = (sorted_f[-1] - sorted_f[0]).abs()
    if n > 1 and (spread <= tie_tol or (flat_threshold > 0 and spread <= flat_threshold)):
        return torch.zeros_like(fitness)
    ranks = torch.empty(n, device=fitness.device)
    ranks[order] = torch.arange(n, device=fitness.device, dtype=torch.float32)
    # average ranks inside runs of (near-)equal fitness
    same_as_prev = torch.cat(
        [torch.tensor([False], device=fitness.device), (sorted_f[1:] - sorted_f[:-1]).abs() <= tie_tol]
    )
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
    compact_below: float = COMPACT_BELOW,
    compact_min_drop: int = COMPACT_MIN_DROP,
) -> dict[str, torch.Tensor | float | int]:
    """Run one episode for the whole population. Returns fitness and telemetry.

    The Metal kernel costs the same for a crashed car as for a driving one,
    and with long or uncapped episodes the last survivors used to keep the
    whole population on the GPU for most of a generation. Every `poll_every`
    steps the finished bodies are therefore dropped from the brain, the car
    environment, the exploit monitor and the unpacked parameters
    (`compact_below`, 0 disables). Results are scattered back into full-size
    tensors in the original body order, so callers see one entry per body.
    """
    agent.seed(seed)
    obs = env.reset()
    agent.reset(batch=env.batch)
    watch = ExploitMonitor(env) if monitor else None
    device = env.device
    full = env.batch
    owner = torch.arange(full, device=device)  # current body -> original body index
    start_index = env.start_index.clone()
    # Final per-body results in the original order, filled as bodies retire.
    laps_full = torch.zeros(full, device=device)
    done_full = torch.zeros(full, dtype=torch.long, device=device)
    # Per-body accumulators in the current (compacted) order.
    live: dict[str, torch.Tensor] = {
        "alive": torch.ones(full, device=device),
        "fitness": torch.zeros(full, device=device),
        "steps_alive": torch.zeros(full, device=device),
        "speed_sum": torch.zeros(full, device=device),
        "first_done_step": torch.full((full,), -1, dtype=torch.long, device=device),
        "first_crash_clearance": torch.full((full,), float("nan"), device=device),
        "first_crash_progress": torch.full((full,), float("nan"), device=device),
        "first_crash_speed": torch.full((full,), float("nan"), device=device),
        "first_lap_step": torch.full((full,), -1, dtype=torch.long, device=device),
    }
    final: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in live.items()}
    vis_hz = torch.zeros((), device=device)
    ran = 0
    body_steps = 0
    compactions = 0
    action_std_sum = torch.zeros((), device=device)
    motor_std_sum = torch.zeros((), device=device)
    sensory_std_sum = torch.zeros((), device=device)
    obs_std_same_start_sum = torch.zeros((), device=device)
    first_action = None
    first_motor = None
    first_sensory = None
    first_obs = None
    steer_sat_sum = torch.zeros((), device=device)
    pedal_sat_sum = torch.zeros((), device=device)

    def retire(mask: torch.Tensor) -> None:
        """Copy the results of the bodies in `mask` (current order) into the full-size tensors."""
        nonlocal laps_full, done_full
        dst = owner[mask]
        for key, value in live.items():
            final[key] = final[key].index_put((dst,), value[mask])
        laps_full = laps_full.index_put((dst,), env.laps[mask])
        done_full = done_full.index_put((dst,), env.done_reason[mask])

    with torch.inference_mode():
        for step in defaults.step_range(steps):
            if first_obs is None:
                # Each genome is evaluated on the same ordered start set.
                # Compare across genomes at each start to detect reset/input collapse.
                first_obs = obs.detach().clone()
            action = agent.act(obs, theta)
            if first_action is None:
                first_action = action.detach().clone()
                first_motor = agent.last_motor.detach().clone()
                first_sensory = agent.last_rates_hz.detach().clone()
            action_std_sum = action_std_sum + action.std(dim=0).mean()
            motor_std_sum = motor_std_sum + agent.last_motor.std(dim=0).mean()
            sensory_std_sum = sensory_std_sum + agent.last_rates_hz.std(dim=0).mean()
            obs, reward, done = env.step(action)
            alive = live["alive"]
            newly_done = done & (live["first_done_step"] < 0)
            live["first_done_step"] = torch.where(newly_done, torch.full_like(live["first_done_step"], step + 1), live["first_done_step"])
            clearance = env.last_terms["clearance"]
            crash = newly_done & (env.done_reason == 1)
            live["first_crash_clearance"] = torch.where(crash, clearance, live["first_crash_clearance"])
            live["first_crash_progress"] = torch.where(crash, env.laps, live["first_crash_progress"])
            live["first_crash_speed"] = torch.where(crash, env.speed, live["first_crash_speed"])
            crossed = (env.laps >= 1.0) & (live["first_lap_step"] < 0) & (alive > 0)
            live["first_lap_step"] = torch.where(crossed, torch.full_like(live["first_lap_step"], step + 1), live["first_lap_step"])
            steer_sat_sum = steer_sat_sum + (action[:, 0].abs() > 0.98).float().mean()
            pedal_sat_sum = pedal_sat_sum + (action[:, 1].abs() > 0.98).float().mean()
            if watch is not None:
                watch.observe(reward, alive.bool())
            live["fitness"] = live["fitness"] + reward * alive
            live["steps_alive"] = live["steps_alive"] + alive
            live["speed_sum"] = live["speed_sum"] + env.speed * alive
            # Telemetry from the rates `act` already computed for this step: calling
            # `agent.sensory_rates` again here overwrote `prev_proximity` with the
            # post-step view, which zeroed the looming channel on every control step.
            vis_hz = vis_hz + agent.last_rates_hz[:, : agent.cfg.n_rays].mean()
            live["alive"] = alive * (~done).float()
            ran = step + 1
            body_steps += env.batch
            # alive.sum() is a GPU sync that stalls the command queue, so poll
            # every `poll_every` steps; the same poll drives compaction.
            if step % poll_every == poll_every - 1:
                n_alive = int(live["alive"].sum())
                if n_alive == 0:
                    break
                if compact_below > 0 and n_alive <= compact_below * env.batch and env.batch - n_alive >= compact_min_drop:
                    keep = live["alive"] > 0
                    retire(~keep)
                    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
                    live = {k: v[idx] for k, v in live.items()}
                    owner = owner[idx]
                    theta = {k: v[idx] for k, v in theta.items()}
                    obs = obs[idx]
                    if watch is not None:
                        watch.compact(keep)  # reads the environment: before env.compact
                    env.compact(keep)
                    agent.compact(keep)
                    compactions += 1
        retire(torch.ones(env.batch, dtype=torch.bool, device=device))
    if first_obs is not None:
        # The rollout layout is [member0 starts..., member1 starts..., ...].
        # start_fraction is the per-member schedule; infer its period.
        sf = env.start_fraction if env.start_fraction.numel() == full else start_index.float()
        period = 1
        if sf.numel() > 1:
            first = sf[:1]
            matches = torch.nonzero((sf == first).squeeze())
            if matches.numel() > 1:
                period = int(matches[1].item())
        if period > 1 and full % period == 0:
            obs_groups = first_obs.view(full // period, period, -1)
            obs_std_same_start_sum = obs_groups.std(dim=0).mean()
        else:
            obs_std_same_start_sum = first_obs.std(dim=0).mean()
    counts = torch.bincount(done_full, minlength=5).tolist()
    return {
        "fitness": final["fitness"],
        "action_std_mean": float(action_std_sum / max(1, ran)),
        "motor_std_mean": float(motor_std_sum / max(1, ran)),
        "sensory_std_mean": float(sensory_std_sum / max(1, ran)),
        "obs_std_same_start": float(obs_std_same_start_sum),
        "action_std_first": float(first_action.std(dim=0).mean()) if first_action is not None else 0.0,
        "motor_std_first": float(first_motor.std(dim=0).mean()) if first_motor is not None else 0.0,
        "sensory_std_first": float(first_sensory.std(dim=0).mean()) if first_sensory is not None else 0.0,
        "laps": laps_full,
        "steps_alive": final["steps_alive"],
        "dn_hz": float(agent.dn_rate_hz.mean()),
        "vis_hz": float(vis_hz / max(1, ran)),
        "motor_abs_mean": float(agent.last_motor.abs().mean()),
        "steps_run": ran,
        "body_steps": body_steps,
        "compactions": compactions,
        "first_done_step_mean": float(final["first_done_step"][final["first_done_step"] >= 0].float().mean()) if (final["first_done_step"] >= 0).any() else 0.0,
        "first_done_step_min": int(final["first_done_step"][final["first_done_step"] >= 0].min()) if (final["first_done_step"] >= 0).any() else -1,
        "first_crash_clearance_mean": float(torch.nanmean(final["first_crash_clearance"])) if torch.isfinite(final["first_crash_clearance"]).any() else float("nan"),
        "first_crash_progress_mean": float(torch.nanmean(final["first_crash_progress"])) if torch.isfinite(final["first_crash_progress"]).any() else float("nan"),
        "first_crash_speed_mean": float(torch.nanmean(final["first_crash_speed"])) if torch.isfinite(final["first_crash_speed"]).any() else float("nan"),
        "first_lap_step": final["first_lap_step"],
        "laps_completed": int((final["first_lap_step"] > 0).sum()),
        "steer_saturation_fraction": float(steer_sat_sum / max(1, ran)),
        "pedal_saturation_fraction": float(pedal_sat_sum / max(1, ran)),
        "exploits": watch.report() if watch is not None else None,
        **env.telemetry(),
        # Whole-population figures (the environment only holds the survivors).
        "speed_mean": float((final["speed_sum"] / final["steps_alive"].clamp(min=1)).mean()),
        "laps_min": float(laps_full.min()),
        "crash": counts[1],
        "reverse": counts[2],
        "stuck": counts[3],
        "finished": counts[4],
        "alive": counts[0],
        "start_index": start_index,
        "done_reason": done_full,
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


def make_env(bank: TrackBank, stage: int, generation: int, popsize: int, starts_per_gen: int, steps: int = 0) -> CarEnv:
    """Environment for one generation: `starts_per_gen` start points per member."""
    if bank.layout == "monaco":
        cfg, track = bank.get(stage)
        base = generation % bank.n_tracks
        fractions = [((base + k) % bank.n_tracks) / bank.n_tracks for k in range(starts_per_gen)]
    else:
        cfg, track = bank.get(stage, generation % bank.n_tracks)
        fractions = [k / starts_per_gen for k in range(starts_per_gen)]
    per_car = torch.tensor(fractions).repeat(popsize)
    cfg = replace(cfg, episode_steps=max(0, steps))  # the environment charges unused budget at early endings
    return CarEnv(popsize * starts_per_gen, bank.device, cfg, track=track, start_fraction=per_car)


def make_eval_env(bank: TrackBank, stage: int, islands: int, starts_per_gen: int, steps: int = 0) -> CarEnv:
    """Evaluate every island on the same deterministic start set."""
    cfg, track = bank.get(stage)
    fractions = torch.tensor([(b % bank.n_tracks) / bank.n_tracks for b in range(starts_per_gen)]).repeat(islands)
    cfg = replace(cfg, episode_steps=max(0, steps))
    return CarEnv(islands * starts_per_gen, bank.device, cfg, track=track, start_fraction=fractions)


def evaluate_mean(
    agent: ConnectomeAgent, bank: TrackBank, mu: torch.Tensor, stage: int, steps: int, seed: int, starts_per_gen: int
) -> dict[str, float]:
    islands = mu.shape[0]
    # One body per (island, start): `rollout` runs the brain at this reduced
    # batch. Evaluating on the brain's full batch drove 64 identical copies of
    # every (island, start) pair (shared noise), so each evaluation cost as
    # much as a whole generation and halved the number of generations per hour.
    bodies_per_island = starts_per_gen
    n_tracks = bank.n_tracks
    env = make_eval_env(bank, stage, islands, starts_per_gen, steps)
    batch = env.batch
    mu_expanded = mu.repeat_interleave(bodies_per_island, dim=0)
    theta = agent.unpack(mu_expanded)
    result = rollout(agent, env, theta, steps, seed)
    # Group by start slot, NOT by centerline index: env.start_index is a sample
    # index into the (2048-point) centerline, so comparing it against
    # range(n_tracks) matched only the cars whose start is exactly sample 0 and
    # left every other group empty, making eval_fitness/eval_laps NaN for any
    # layout whose start fractions land on nonzero centerline samples.
    slots = torch.arange(batch, device=env.device) % starts_per_gen
    n_groups = max(starts_per_gen, n_tracks)
    fit_sum = torch.zeros(n_groups, device=env.device).index_add_(0, slots, result["fitness"])
    laps_sum = torch.zeros(n_groups, device=env.device).index_add_(0, slots, result["laps"])
    counts = torch.zeros(n_groups, device=env.device).index_add_(0, slots, torch.ones(batch, device=env.device))
    present = counts > 0
    per_start_fit = (fit_sum / counts.clamp(min=1))[present]
    per_start_laps = (laps_sum / counts.clamp(min=1))[present]
    island_fit = result["fitness"].view(islands, bodies_per_island).mean(1)
    island_laps = result["laps"].view(islands, bodies_per_island).mean(1)
    # Lap gate per island: the worst body of the island (every start) must
    # complete the lap; `eval_lap_steps_island` is that island's mean
    # first-lap step, the lap time the promotion is judged on.
    island_laps_min = result["laps"].view(islands, bodies_per_island).min(1).values
    first_lap = result["first_lap_step"].view(islands, bodies_per_island).float()
    lap_steps_island = torch.where(first_lap > 0, first_lap, torch.full_like(first_lap, float("nan"))).nanmean(1)
    best_island = int(island_fit.argmax())
    return {
        "eval_fitness": float(per_start_fit.mean()),
        "eval_fitness_best_island": float(island_fit.max()),
        "eval_fitness_min": float(per_start_fit.min()),
        "eval_laps": float(per_start_laps.mean()),
        "eval_laps_best_island": float(island_laps.max()),
        "eval_laps_min": float(per_start_laps.min()),
        "eval_best_island": best_island,
        "eval_laps_min_best_island": float(island_laps_min[best_island]),
        "eval_lap_steps_best_island": float(lap_steps_island[best_island]) if torch.isfinite(lap_steps_island[best_island]) else None,
        "eval_all_laps": bool((island_laps_min >= 1.0).any()),
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
        "--islands",
        type=int,
        default=4,
        help="independent ES populations evaluated concurrently in one batched brain; 4 x 128 x 6 starts = 3,072 bodies runs at 96%% of the Metal kernel's peak body-steps/s (14 islands: 100%%, but a generation takes 3.5x longer)",
    )
    parser.add_argument(
        "--starts-per-gen",
        type=int,
        default=1,
        help="start points each member is scored on per generation (batch = popsize x this)",
    )
    parser.add_argument("--sigma", type=float, default=0.05, help="perturbation scale; grows while the landscape is flat")
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=0.4,
        help="ceiling for the adaptive perturbation scale (0 disables adaptation)",
    )
    parser.add_argument(
        "--sigma-grow",
        type=float,
        default=1.25,
        help="factor applied to sigma after a generation with no fitness differences above the tie tolerance",
    )
    parser.add_argument(
        "--sigma-shrink",
        type=float,
        default=0.8,
        help="factor pulling sigma back towards --sigma after an informative generation",
    )
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=0.25,
        help="fitness differences at or below this are ties; use 0 to rank every distinct score",
    )
    parser.add_argument(
        "--flat-threshold",
        type=float,
        default=0.5,
        help="if the entire population's fitness range is below this, treat the generation as non-informative (no gradient); catches collapse mode where all cars crash at the same step",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--max-step-frac",
        type=float,
        default=0.03,
        help="trust region: cap one generation's move of an island mean at this fraction of the mean's norm (0 disables)",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=("fp16", "fp32"),
        help="brain state precision on the Metal path (default: MALECNS_PRECISION or fp16)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="0.9 let the mean drift into saturation on flat fitness; 0.5 holds",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=-1,
        help="-1 (default): no cap, episodes end by crash/stuck/reverse/finished max_laps; 0: the curriculum's step caps; >0: a fixed cap",
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
    parser.add_argument(
        "--car",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override any CarConfig field (reward terms, vehicle, sensing), e.g. --car speed_penalty=0.02 --car lap_bonus=200; repeatable",
    )
    parser.add_argument("--curriculum-window", type=int, default=25)
    parser.add_argument(
        "--progress-bins",
        type=int,
        default=24,
        help="bins around the lap for the where-did-they-end histogram in the log",
    )
    parser.add_argument("--eval-every", type=int, default=10, help="deterministic evaluation of the mean; 0 disables")
    parser.add_argument(
        "--eval-tolerance",
        type=float,
        default=2.0,
        help="the mean is kept only if its evaluation fitness is within this of the last accepted evaluation; otherwise it reverts to the accepted mean, sigma shrinks and momentum clears (a verified-improvement guard; negative disables)",
    )
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

    batch = args.islands * args.popsize * args.starts_per_gen
    brain = Brain(
        connectome,
        batch=batch,
        config=LIFConfig(dt_ms=args.dt_ms, adapt_mv=args.adapt_mv),
        device=device,
        weight_scale=args.weight_scale,
        precision=args.precision,
    )
    # A resumed checkpoint decides the interface configuration (eye encoding,
    # readout normalisation, ...): building the agent from defaults would flag
    # the checkpoint's readout as mismatched and reset it (that threw away a
    # lap-completing calibrated readout once).
    saved_agent_cfg = None
    if args.checkpoint.exists():
        try:
            saved_agent_cfg = torch.load(args.checkpoint, map_location="cpu").get("agent_cfg")
        except (RuntimeError, EOFError, KeyError):
            saved_agent_cfg = None
    agent_cfg = AgentConfig.from_saved(saved_agent_cfg, substeps=args.substeps)
    agent = ConnectomeAgent(brain, neurons, agent_cfg)
    dt_s = defaults.control_dt_s(args.dt_ms, args.substeps)
    if args.layout == "monaco":
        base_cfg = replace(monaco_config(dt_s), geojson_path=args.geojson)
    else:
        base_cfg = CarConfig(dt_s=dt_s)
    for item in args.car:
        # any CarConfig field from the command line: reward terms, vehicle, sensing
        key, _, raw = item.partition("=")
        known = {f.name for f in dataclass_fields(CarConfig)}
        if key not in known:
            raise SystemExit(f"--car {item}: unknown CarConfig field {key!r}; known: {', '.join(sorted(known))}")
        current = getattr(base_cfg, key)
        value = raw.lower() in ("1", "true", "yes") if isinstance(current, bool) else type(current)(raw)
        base_cfg = replace(base_cfg, **{key: value})
    bank = TrackBank(base_cfg, args.layout, args.tracks, device)
    _, track0 = bank.get(len(CURRICULUM) - 1)
    tie_tol = args.tie_tol if args.tie_tol >= 0 else base_cfg.time_tax
    print(
        f"circuit {track0.length_m:,.0f} m, {2 * base_cfg.track_halfwidth:.0f} m wide, "
        f"tightest {track0.min_radius_m:.1f} m vs car {base_cfg.min_turn_radius:.1f} m; "
        f"params {agent.n_params:,}; islands {args.islands}; bodies {batch}; control step {dt_s * 1000:.0f} ms; "
        f"brain {brain.precision} {'metal' if brain.uses_metal else 'torch'}; tie tolerance {tie_tol}; flat threshold {args.flat_threshold}"
    )

    # Independent ES means share one batched Brain across all islands.
    mu = agent.initial_params().to(device).repeat(args.islands, 1)
    momentum = torch.zeros_like(mu)
    generation = 0
    stage = len(CURRICULUM) - 1 if args.no_curriculum else 0
    best_eval = -float("inf")
    best_lap_steps: float | None = None
    accepted_mu: torch.Tensor | None = None
    accepted_eval: float | None = None
    sigma = args.sigma
    recent_laps: list[float] = []
    # best.pt is only ever overwritten by a better deterministic evaluation,
    # whatever the resumed checkpoint remembers: a resumed run that had lost
    # its best_eval once replaced a lap-completing best.pt with a crash.
    if args.best.exists():
        try:
            best_eval = float(torch.load(args.best, map_location="cpu").get("best_eval", best_eval))
        except (RuntimeError, EOFError, KeyError):
            pass
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device)
        generation = int(state["generation"])
        if agent.load_readout(state):
            print("calibrated readout loaded from checkpoint")
        try:
            reset = tuple(b for b in args.reset_blocks.split(",") if b)
            mu_cpu, momentum_cpu, notes = agent.migrate_state(state, reset=reset)
            loaded_mu = mu_cpu.to(device)
            loaded_momentum = momentum_cpu.to(device)
            if loaded_mu.ndim == 1:
                loaded_mu = loaded_mu.unsqueeze(0)
            if loaded_momentum.ndim == 1:
                loaded_momentum = loaded_momentum.unsqueeze(0)
            if loaded_mu.shape[0] != args.islands:
                old_islands = loaded_mu.shape[0]
                if old_islands > 0 and args.islands % old_islands == 0:
                    factor = args.islands // old_islands
                    loaded_mu = loaded_mu.repeat_interleave(factor, dim=0)
                    loaded_momentum = loaded_momentum.repeat_interleave(factor, dim=0)
                    notes.append(f"expanded checkpoint islands {old_islands} -> {args.islands} (x{factor})")
                else:
                    raise ValueError(f"checkpoint has {old_islands} islands, requested {args.islands}; requested count must be a multiple of the saved count")
            if state.get("readout") and any("(reset)" in n for n in notes) and not reset:
                # A calibrated readout must never be replaced by init values
                # behind the operator's back: that overwrote a lap-completing
                # checkpoint with a non-driving one once.
                raise SystemExit(f"checkpoint {args.checkpoint} carries a calibrated readout but its blocks would be reset ({', '.join(notes)}); refusing to train. Pass --reset-blocks explicitly if that is intended.")
            mu = agent.clamp_params(loaded_mu)
            momentum = loaded_momentum
            print(f"resumed at generation {generation}" + (f"; migrated: {', '.join(notes)}" if notes else ""))
        except ValueError as exc:
            print(f"{exc}: parameters restart from init, generation counter continues at {generation}")
        stage = int(state.get("stage", stage)) if not args.no_curriculum else stage
        best_eval = max(best_eval, float(state.get("best_eval", -float("inf"))))
        best_lap_steps = state.get("best_lap_steps", None)
        accepted_eval = state.get("accepted_eval", None)
        accepted_mu = state["accepted_mu"].to(device) if state.get("accepted_mu") is not None else None
        if accepted_mu is not None and accepted_mu.shape != mu.shape:
            accepted_mu, accepted_eval = None, None
        sigma = float(state.get("sigma", sigma))
        recent_laps = list(state.get("recent_laps", []))

    def save(path: Path) -> None:
        torch.save(
            {
                "mu": mu.cpu(),
                "momentum": momentum.cpu(),
                "islands": args.islands,
                "generation": generation,
                "stage": stage,
                "best_eval": best_eval,
                "best_lap_steps": best_lap_steps,
                "accepted_eval": accepted_eval,
                "accepted_mu": accepted_mu.cpu() if accepted_mu is not None else None,
                "sigma": sigma,
                "recent_laps": recent_laps[-args.curriculum_window :],
                "n_params": agent.n_params,
                "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()},
                "agent_cfg": asdict(agent_cfg),
                "readout": agent.readout_state(),
                "car_cfg": asdict(replace(bank.get(stage)[0], episode_steps=episode_cap(args.episode_steps, stage))),
                "layout": args.layout,
                "starts": args.tracks,
                "precision": brain.precision,
                "curriculum": [list(c) for c in CURRICULUM],
                "saved_at": time.time(),
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
        steps = episode_cap(args.episode_steps, stage)
        sample_gen.manual_seed(args.seed * 1_000_003 + generation)
        eps = torch.randn(args.islands, half, mu.shape[1], generator=sample_gen).to(device)
        perturb = torch.cat([eps, -eps], dim=1)
        params = agent.clamp_params(mu[:, None, :] + sigma * perturb).reshape(args.islands * args.popsize, -1)
        theta = agent.unpack(params.repeat_interleave(args.starts_per_gen, dim=0))

        total_members = args.islands * args.popsize
        env = make_env(bank, stage, generation, total_members, args.starts_per_gen, steps)
        result = rollout(agent, env, theta, steps, seed=args.seed * 7919 + generation, monitor=not args.no_exploit_monitor)
        fitness = result["fitness"].view(args.islands, args.popsize, args.starts_per_gen).mean(2)
        laps = result["laps"].view(args.islands, args.popsize, args.starts_per_gen).mean(2)
        # Where around the lap did each car end (its lap fraction at the last
        # step), so a corner that kills the whole population shows up by name.
        starts_frac = result["start_index"].float() / env.track.centerline.shape[0]
        end_frac = (starts_frac + result["laps"]) % 1.0
        end_hist = torch.histc(end_frac, bins=args.progress_bins, min=0.0, max=1.0).to(torch.int64).tolist()
        reasons = result["done_reason"]
        reason_by_start = [
            {name: int((reasons.view(total_members, args.starts_per_gen)[:, k] == code).sum()) for code, name in DONE_NAMES.items()}
            for k in range(args.starts_per_gen)
        ]

        advantage = torch.stack([rank_normalise(fitness[i], tie_tol, args.flat_threshold) for i in range(args.islands)])
        rank_informative = bool((advantage != 0).any())
        exploit_report = result["exploits"]
        # Only CRITICAL (reward accounting, phasing, bonus farming) vetoes the
        # update: HIGH flags such as idle_reward fire on honest slow crawling
        # after a spin and were discarding whole generations (and widening
        # sigma, which made the next one worse).
        exploit_block = bool(exploit_report and exploit_report.get("severity") == "CRITICAL")
        # A high-severity exploit means the fitness signal is demonstrably
        # being produced by invalid behavior (for example the generation-747
        # oscillation collapse). Do not turn that signal into an ES update.
        informative = rank_informative and not exploit_block
        grad = torch.zeros_like(mu)
        if informative:
            grad = (perturb * advantage[:, :, None]).sum(1) / (args.popsize * sigma)
            momentum = args.momentum * momentum + grad
            step = args.lr * momentum
            if args.max_step_frac > 0:
                # Trust region: one generation may move an island's mean by at
                # most this fraction of its norm. Generation 1 of the calibrated
                # run moved it by 26 % (grad norm 37, lr 0.05) and the driver
                # went from 1.3 laps to crawling at 8 km/h.
                limit = args.max_step_frac * mu.norm(dim=1, keepdim=True).clamp(min=1e-6)
                scale = (limit / step.norm(dim=1, keepdim=True).clamp(min=1e-12)).clamp(max=1.0)
                step = step * scale
            mu = agent.clamp_params(mu + step)
        else:
            # In particular, a flat/rejected generation must not continue to
            # move the means through stale momentum from earlier generations.
            momentum.zero_()

        # Adaptive search radius: widen when selection is unavailable (flat or
        # exploit-contaminated), and pull back towards the base value when the
        # generation provides a trustworthy selection signal.
        if args.sigma_max > 0:
            sigma = min(args.sigma_max, sigma * args.sigma_grow) if not informative else max(args.sigma, sigma * args.sigma_shrink)

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
            "fitness_min": float(fitness.min()),
            "fitness_std": float(fitness.std()),
            # Keep the independent ES islands observable in the log. The
            # aggregate metrics above are useful for the trainer as a whole,
            # but they hide whether one island is improving while another
            # collapses. These small arrays make the web viewer able to plot
            # the parallel trainers without storing per-member data.
            "island_fitness_mean": [round(float(v), 4) for v in fitness.mean(1)],
            "island_fitness_best": [round(float(v), 4) for v in fitness.max(1).values],
            "island_laps_mean": [round(float(v), 5) for v in laps.mean(1)],
            "island_laps_best": [round(float(v), 5) for v in laps.max(1).values],
            "informative": informative,
            "laps_best": float(laps.max()),
            "laps_mean": float(laps.mean()),
            "laps_min": result["laps_min"],
            "steps_alive_mean": float(result["steps_alive"].mean()),
            "steps_alive_max": float(result["steps_alive"].max()),
            "steps_run": result["steps_run"],
            "first_done_step_mean": result["first_done_step_mean"],
            "first_done_step_min": result["first_done_step_min"],
            "first_crash_clearance_mean": result["first_crash_clearance_mean"],
            "first_crash_progress_mean": result["first_crash_progress_mean"],
            "first_crash_speed_mean": result["first_crash_speed_mean"],
            "steer_saturation_fraction": result["steer_saturation_fraction"],
            "pedal_saturation_fraction": result["pedal_saturation_fraction"],
            "speed_mean": result["speed_mean"],
            "speed_max": result["speed_max"],
            "lat_g_max": result["lat_g_max"],
            "lat_g_mean": result["lat_g_mean"],
            "steer_abs_mean": result["steer_abs_mean"],
            "motor_abs_mean": result["motor_abs_mean"],
            "unique_genomes": int(torch.unique(params, dim=0).shape[0]),
            "genome_std_mean": float(params.std(dim=0).mean()),
            "action_std_first": result["action_std_first"],
            "action_std_mean": result["action_std_mean"],
            "motor_std_first": result["motor_std_first"],
            "motor_std_mean": result["motor_std_mean"],
            "sensory_std_first": result["sensory_std_first"],
            "sensory_std_mean": result["sensory_std_mean"],
            "obs_std_same_start": result["obs_std_same_start"],
            "crash": result["crash"],
            "reverse": result["reverse"],
            "stuck": result["stuck"],
            "alive": result["alive"],
            "dn_hz": result["dn_hz"],
            "vis_hz": result["vis_hz"],
            "exploits": result["exploits"],
            "at_bounds": agent.fraction_at_bounds(mu),
            "mu_norm": float(mu.norm()),
            "grad_norm": float(grad.norm()),
        "step_norm": float((args.lr * momentum).norm()) if informative else 0.0,
            "momentum_norm": float(momentum.norm()),
            "sigma": sigma,
            "lr": args.lr,
            "popsize": args.popsize,
            "precision": brain.precision,
            "road_halfwidth": bank.get(stage)[0].track_halfwidth,
            "fitness_per_start": [round(float(v), 3) for v in result["fitness"].view(total_members, args.starts_per_gen).mean(0)],
            "laps_per_start": [round(float(v), 4) for v in result["laps"].view(total_members, args.starts_per_gen).mean(0)],
            "start_fractions": [round(float(v), 4) for v in starts_frac.view(total_members, args.starts_per_gen)[0]],
            "end_progress_hist": end_hist,
            "end_reason_per_start": reason_by_start,
            "time": time.time(),
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
            eval_steps = episode_cap(args.episode_steps, stage)
            record.update(evaluate_mean(agent, bank, mu, stage, eval_steps, seed=args.seed, starts_per_gen=args.starts_per_gen))
            # Verified-improvement guard: an ES step that made the *mean* drive
            # worse (fitness of the deterministic evaluation dropped by more
            # than the tolerance) is undone, so the checkpoint the viewer and
            # the lap gate read never drifts away from a driving policy.
            current_eval = record["eval_fitness_best_island"]
            if args.eval_tolerance >= 0 and accepted_eval is not None and current_eval < accepted_eval - args.eval_tolerance:
                mu = accepted_mu.clone()
                momentum.zero_()
                sigma = max(args.sigma * 0.25, sigma * 0.7)
                record["reverted"] = True
                record["accepted_eval"] = accepted_eval
                print(f"[guard] mean evaluation {current_eval:.2f} < accepted {accepted_eval:.2f} - {args.eval_tolerance}: reverted to the accepted mean, sigma {sigma:.4f}")
            else:
                accepted_mu = mu.clone()
                accepted_eval = current_eval
                record["accepted_eval"] = accepted_eval
            lap_ok = record["eval_laps_min_best_island"] >= 1.0
            lap_steps = record["eval_lap_steps_best_island"]
            if lap_ok and lap_steps is not None:
                # promotion on lap time once laps are being completed: best_lap_steps
                # is the record to beat, best_eval keeps the fitness for the log
                if best_lap_steps is None or lap_steps < best_lap_steps:
                    best_lap_steps = lap_steps
                    best_eval = max(best_eval, record["eval_fitness_best_island"])
                    record["new_best"] = True
                    record["best_lap_s"] = round(lap_steps * dt_s, 1)
                    save(args.best)
                    print(f"[best] lap {lap_steps * dt_s:.1f} s from every start; saved {args.best}")
            elif best_lap_steps is None and record["eval_fitness_best_island"] > best_eval:
                best_eval = record["eval_fitness_best_island"]
                record["new_best"] = True
                save(args.best)

        synchronize(device)
        elapsed = time.time() - started
        record["seconds"] = round(elapsed, 2)
        record["body_steps_per_s"] = round(result["body_steps"] / max(elapsed, 1e-6))
        record["body_steps"] = int(result["body_steps"])
        record["compactions"] = int(result["compactions"])
        record["control_step_ms"] = round(1000.0 * elapsed / max(1, result["steps_run"]), 2)
        with args.log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"gen {generation:5d} s{stage} | fit {record['fitness_mean']:8.2f} "
            f"best {record['fitness_best']:8.2f} | laps {record['laps_mean']:.3f}/{record['laps_best']:.3f} "
            f"| alive {record['steps_alive_mean']:6.0f}/{steps} | {record['speed_mean'] * 3.6:5.0f} km/h "
            f"| c{record['crash']} r{record['reverse']} s{record['stuck']} "
            f"| s{sigma:.3f} | {record['seconds']:.1f}s {record['body_steps_per_s']:,} body-steps/s"
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
