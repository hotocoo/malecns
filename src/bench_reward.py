"""Reward and physics benchmarks on the CPU environment, no brain in the loop.

    python3 src/bench_reward.py

Prints, with numbers:
  A. progress payout density: fraction of control steps that pay nothing at
     15 and 60 m/s under nearest-sample progress (old) vs sub-sample (new)
  B. the "dying pays" ranking: a slow survivor vs an early crash, with and
     without the step-budget terminal charge (`CarConfig.episode_steps`)
  C. deterministic evaluation cost: body-steps per evaluation on the brain's
     full batch (old) vs one body per (island, start) (new)
  D. vehicle model vs public Mercedes-AMG F1 W11 figures: launch times, top
     speed, braking distance, lateral grip by speed
  E. Monaco reference speed profile (`speed_profile`): the model's own pole
     lap, the pace term at the trainer's current cruising speed
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from car_env import DONE_NAMES, CarConfig, CarEnv, Track, build_centerline, load_geojson_centerline, monaco_config, speed_profile  # noqa: E402

CPU = torch.device("cpu")
G = 9.81
MONACO = Path(__file__).resolve().parents[1] / "data" / "tracks" / "monaco.geojson"


def circle_track(cfg: CarConfig) -> Track:
    return Track(build_centerline(cfg, 1), cfg, CPU)


def pursuit_steer(env: CarEnv, lookahead_s: float = 0.8, min_m: float = 12.0) -> torch.Tensor:
    """Pure-pursuit steering towards a centerline point ahead; holds any line at any speed."""
    track = env.track
    n = track.centerline.shape[0]
    idx = (env.last_progress * n).long().clamp(0, n - 1)
    ahead = (torch.maximum(torch.full_like(env.speed, min_m), env.speed * lookahead_s) / track.spacing_m).clamp(min=1.0).long()
    target = track.centerline[(idx + ahead) % n]
    to = target - env.pos
    alpha = torch.atan2(to[:, 1], to[:, 0]) - env.heading
    alpha = torch.atan2(alpha.sin(), alpha.cos())
    dist = to.norm(dim=1).clamp(min=1.0)
    steer = torch.atan(2.0 * env.cfg.wheelbase * alpha.sin() / dist) / env.cfg.max_steer_rad
    return steer.clamp(-1.0, 1.0)


def drive(env: CarEnv, steps: int, target: torch.Tensor, brake_from: int | None = None, crash_first: bool = False) -> dict:
    alive = torch.ones(env.batch, dtype=torch.bool)
    total = torch.zeros(env.batch)
    end_step = torch.full((env.batch,), -1, dtype=torch.long)
    deltas, poses, speeds, terms = [], [], [], []
    for i in range(steps):
        steer = pursuit_steer(env)
        if crash_first:
            steer[0] = 1.0  # car 0 turns into the barrier
        pedal = torch.where(env.speed < target, torch.ones(env.batch), torch.full((env.batch,), -0.2))
        if brake_from is not None and i >= brake_from:
            pedal = torch.full((env.batch,), -1.0)
        _, reward, done = env.step(torch.stack([steer, pedal], dim=1))
        total = total + reward * alive
        end_step = torch.where(done & alive, torch.full_like(end_step, i + 1), end_step)
        alive = alive & ~done
        deltas.append(env.last_terms["delta"].clone())
        poses.append(env.pos.clone())
        speeds.append(env.speed.clone())
        terms.append({k: v.clone() for k, v in env.last_terms.items() if k in ("pace", "align", "wall")})
        if not bool(alive.any()):
            break
    return {"total": total, "end_step": end_step, "deltas": torch.stack(deltas), "poses": torch.stack(poses), "speeds": torch.stack(speeds), "reason": env.done_reason.clone(), "terms": terms}


def section_a(cfg: CarConfig, track: Track) -> None:
    print("\nA. progress payout density (perfect circle, radius 280 m, pure-pursuit driver)")
    print(f"{'speed':>8} {'steps':>6} {'old zero-pay':>13} {'new zero-pay':>13} {'old max/step':>13} {'new max/step':>13}")
    for target in (15.0, 60.0):
        env = CarEnv(1, CPU, cfg, track=track)
        env.reset()
        run = drive(env, 600, torch.tensor([target]))
        warm = 300
        new = run["deltas"][warm:, 0] * env.progress_scale
        cells = track._cell_index(run["poses"][:, 0])
        old_prog = track.progress[cells[0], cells[1]]
        old = (old_prog[warm:] - old_prog[warm - 1 : -1]).remainder(1.0) * env.progress_scale
        print(
            f"{target * 3.6:6.0f}km/h {new.numel():6d} {float((old == 0).float().mean()):12.1%} {float((new == 0).float().mean()):12.1%}"
            f" {float(old.max()):13.3f} {float(new.max()):13.3f}"
        )


def section_b(cfg: CarConfig, track: Track, budget: int) -> None:
    print(f"\nB. slow survivor (14 km/h for {budget} steps) vs early crash; fitness = sum of reward while alive")
    target = torch.tensor([30.0, 4.0])
    print(f"{'reward':<30} {'crasher':>10} {'ended':>14} {'survivor':>10} {'ended':>10} {'ranks first':>12}")
    for label, c in (("old: fixed -20, no budget", replace(cfg, pace_penalty=0.0, align_penalty=0.0)), ("new: budget + pace + align", replace(cfg, episode_steps=budget))):
        env = CarEnv(2, CPU, c, track=track)
        env.reset()
        run = drive(env, budget, target, crash_first=True)
        crash_end = f"{DONE_NAMES[int(run['reason'][0])]} @ {int(run['end_step'][0])}"
        surv_end = DONE_NAMES[int(run["reason"][1])]
        first = "crasher" if float(run["total"][0]) > float(run["total"][1]) else "survivor"
        print(f"{label:<30} {float(run['total'][0]):10.2f} {crash_end:>14} {float(run['total'][1]):10.2f} {surv_end:>10} {first:>12}")


def section_c(islands: int = 4, popsize: int = 64, starts: int = 6, steps: int = 6000) -> None:
    print("\nC. deterministic evaluation cost per generation (run_training.sh defaults)")
    old = islands * popsize * starts * steps
    new = islands * starts * steps
    print(f"  old: brain full batch {islands * popsize * starts:,} bodies x {steps:,} steps = {old:,} body-steps (= one whole generation)")
    print(f"  new: {islands * starts} bodies x {steps:,} steps = {new:,} body-steps ({old / new:.0f}x fewer)")


def section_d(cfg: CarConfig) -> None:
    print("\nD. vehicle model vs public W11 / F1 figures (pure-pursuit runs on a 700 m circle)")
    big = replace(cfg, loop_scale=10.0, grid_res=1024)
    track = circle_track(big)
    env = CarEnv(1, CPU, big, track=track)
    env.reset()
    run = drive(env, 4000, torch.tensor([200.0]))
    v = run["speeds"][:, 0]
    t = torch.arange(1, v.numel() + 1) * big.dt_s

    def first_time(kmh: float) -> str:
        hit = torch.nonzero(v * 3.6 >= kmh)
        return f"{float(t[hit[0, 0]]):.2f} s" if hit.numel() else "never"

    print(f"  {'metric':<34}{'model':>12}   reference")
    print(f"  {'0-100 km/h':<34}{first_time(100):>12}   ~2.6 s")
    print(f"  {'0-200 km/h':<34}{first_time(200):>12}   ~4.5 s")
    print(f"  {'0-300 km/h':<34}{first_time(300):>12}   ~8.5-9 s")
    print(f"  {'top speed (drag limited)':<34}{float(v.max()) * 3.6:>9.0f} km/h   ~290 Monaco speed trap; ~340 with this downforce")
    print(f"  {'ended':<34}{DONE_NAMES[int(run['reason'][0])]:>12}   (must be alive: the circle is 700 m, lateral load {float(v.max()) ** 2 / 700 / G:.2f} g at top speed)")
    top_step = int(v.argmax())
    env.reset()
    run = drive(env, 4000, torch.tensor([200.0]), brake_from=top_step)
    sp = run["speeds"][:, 0]
    braking = sp[top_step:]
    stop = int((braking < 1.0).nonzero()[0, 0]) if bool((braking < 1.0).any()) else braking.numel() - 1
    dist = float((braking[:stop] * big.dt_s).sum())
    decel = (braking[:-1] - braking[1:]) / big.dt_s / G
    print(f"  {'braking ' + f'{float(braking[0]) * 3.6:.0f}' + ' km/h to rest':<34}{dist:>10.0f} m   ~120-130 m from 300 km/h")
    print(f"  {'peak braking':<34}{float(decel.max()) if decel.numel() else 0.0:>10.2f} g   4-5 g")
    print("  lateral grip envelope (friction circle, downforce grows with speed^2):")
    for kmh in (50, 100, 200, 280):
        vv = torch.tensor([kmh / 3.6])
        print(f"    {kmh:>3} km/h: {float(env.grip_g(vv)):.2f} g      (F1: ~1.8-2 g slow hairpin, ~4-5 g fast corners; model cap {big.grip_max_g:.1f} g)")


def section_e() -> None:
    print("\nE. Monaco reference speed profile (`speed_profile`, centerline, no racing line)")
    if not MONACO.exists():
        print("  data/tracks/monaco.geojson missing")
        return
    cfg = monaco_config(0.016)
    centerline = load_geojson_centerline(MONACO, cfg.track_scale, cfg.n_points, cfg.smooth_m).numpy()
    v = speed_profile(centerline, cfg)
    seg = np.hypot(*(np.roll(centerline, -1, axis=0) - centerline).T)
    lap_s = float((seg / v).sum())
    print(f"  lap {seg.sum():,.0f} m; reference lap time {lap_s:.1f} s = {int(lap_s // 60)}:{lap_s % 60:06.3f}   (real pole: 1:10.166 W10 2019; 1:10.3 2021)")
    print(f"  reference speed: min {v.min() * 3.6:.0f} km/h (hairpin), mean {(seg.sum() / lap_s) * 3.6:.0f} km/h, max {v.max() * 3.6:.0f} km/h; samples below 100 km/h: {(v < 100 / 3.6).mean():.0%}")
    log = Path(__file__).resolve().parents[1] / "logs" / "train.jsonl"
    cruise = None
    if log.exists():
        lines = [line for line in log.read_text().splitlines() if line.strip()]
        if lines:
            cruise = float(json.loads(lines[-1]).get("speed_mean", 0.0)) or None
    if cruise is None:
        print("  (no logs/train.jsonl yet: skipping the pace term at the trainer's cruise speed)")
        return
    print(f"  trainer's latest population speed_mean (logs/train.jsonl, last generation): {cruise * 3.6:.0f} km/h")
    deficit = np.clip(1.0 - cruise / (cfg.pace_margin * v), 0.0, None)
    pace = -cfg.pace_penalty * deficit * deficit
    progress = cfg.progress_per_m * cruise * cfg.dt_s
    print(f"  at that cruise: pace term mean {pace.mean():+.4f}/step (worst {pace.min():+.4f} on the fastest stretch), progress +{progress:.4f}/step, time tax -{cfg.time_tax}/step")
    print(f"  at 90% of the reference pace everywhere the pace term is 0 and progress alone pays +{cfg.progress_per_m * (seg.sum() / lap_s) * cfg.dt_s:.4f}/step")


def main() -> int:
    started = time.time()
    cfg = CarConfig(grid_res=1024, loop_difficulty=0.0, dt_s=0.016)
    track = circle_track(cfg)
    print(f"Circuit: perfect circle, {track.length_m:,.0f} m, road {2 * cfg.track_halfwidth:.0f} m wide, W11 body {2 * cfg.car_halfwidth:.0f} m; reference speed here {float(track.speed_ref.min()) * 3.6:.0f} km/h")
    section_a(cfg, track)
    section_b(cfg, track, budget=3000)
    section_c()
    section_d(cfg)
    section_e()
    print(f"\n({time.time() - started:.1f} s on CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
