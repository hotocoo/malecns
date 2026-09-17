"""Reward shaping: continuous progress, no reward for dying under a step budget, apex margin.

All on the CPU environment with a scripted driver on a perfect circle, so the
tests pin the reward's semantics without a brain in the loop.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_env import DONE_ALIVE, DONE_CRASH, CarConfig, CarEnv, Track, build_centerline  # noqa: E402
from exploits import ExploitMonitor  # noqa: E402

CPU = torch.device("cpu")
# A perfect circle (no harmonics), radius 4 x 70 m = 280 m, on a coarse grid for speed.
CIRCLE = CarConfig(grid_res=1024, loop_difficulty=0.0, dt_s=0.016)


@pytest.fixture(scope="module")
def circle() -> Track:
    return Track(build_centerline(CIRCLE, 1), CIRCLE, CPU)


def circle_steer(cfg: CarConfig, track: Track) -> float:
    """Normalised steering that holds the circle's radius on the kinematic bicycle model."""
    radius = float(track.centerline.norm(dim=1).mean())
    return math.atan(cfg.wheelbase / radius) / cfg.max_steer_rad


def drive(env: CarEnv, steps: int, steer: torch.Tensor, target_speed: torch.Tensor, monitor: bool = False) -> dict:
    """Bang-bang speed governor at fixed steering; returns totals, end steps, per-step rewards and poses."""
    watch = ExploitMonitor(env) if monitor else None
    alive = torch.ones(env.batch, dtype=torch.bool)
    total = torch.zeros(env.batch)
    end_step = torch.full((env.batch,), -1, dtype=torch.long)
    rewards, deltas, poses, speeds = [], [], [], []
    for i in range(steps):
        pedal = torch.where(env.speed < target_speed, torch.ones(env.batch), torch.full((env.batch,), -0.2))
        _, reward, done = env.step(torch.stack([steer, pedal], dim=1))
        if watch is not None:
            watch.observe(reward, alive)
        total = total + reward * alive
        newly = done & alive
        end_step = torch.where(newly, torch.full_like(end_step, i + 1), end_step)
        alive = alive & ~done
        rewards.append(reward.clone())
        deltas.append(env.last_terms["delta"].clone())
        poses.append(env.pos.clone())
        speeds.append(env.speed.clone())
        if not bool(alive.any()):
            break
    return {
        "total": total,
        "end_step": end_step,
        "rewards": torch.stack(rewards),
        "deltas": torch.stack(deltas),
        "poses": torch.stack(poses),
        "speeds": torch.stack(speeds),
        "report": watch.report() if watch is not None else None,
        "reason": env.done_reason.clone(),
    }


def test_progress_pays_every_step_not_in_sample_sized_lumps(circle):
    env = CarEnv(1, CPU, CIRCLE, track=circle)
    env.reset()
    steer = torch.tensor([circle_steer(CIRCLE, circle)])
    run = drive(env, 400, steer, torch.tensor([15.0]))
    assert int(env.done_reason[0]) == DONE_ALIVE, "the scripted driver must hold the circle"
    warm = 100  # after the launch, at ~15 m/s
    deltas = run["deltas"][warm:, 0]
    assert bool((deltas > 0).all()), "moving forward must pay progress on every control step"
    # The old nearest-sample progress paid nothing on most steps and one sample's worth on the rest.
    cells = circle._cell_index(run["poses"][:, 0])
    old = circle.progress[cells[0], cells[1]]
    old_deltas = (old[warm:] - old[warm - 1 : -1]).remainder(1.0)
    assert float((old_deltas == 0).float().mean()) > 0.5
    # Same net distance either way: the interpolation redistributes, it does not invent.
    n = circle.centerline.shape[0]
    assert abs(float(deltas.sum()) - float(old_deltas.sum())) < 2.0 / n


def test_progress_is_continuous_across_sample_boundaries(circle):
    n = circle.centerline.shape[0]
    spacing = 2 * math.pi * float(circle.centerline.norm(dim=1).mean()) / n
    # Walk 3 samples along the circle in 60 tiny steps; the progress must grow smoothly.
    theta = torch.linspace(0.0, 3 * 2 * math.pi / n, 60)
    radius = float(circle.centerline.norm(dim=1).mean())
    pts = torch.stack([radius * theta.cos(), radius * theta.sin()], dim=1)
    prog = circle.progress_at(pts)
    step = (prog[1:] - prog[:-1]).remainder(1.0) * n * spacing  # metres per tiny step
    expected = radius * float(theta[1] - theta[0])
    assert float(step.min()) > 0.0
    assert float((step - expected).abs().max()) < 0.35 * expected  # grid-cell wobble only


def test_ending_early_never_scores_above_driving_on(circle):
    budget = 2000
    cfg = CIRCLE
    steer = torch.tensor([1.0, circle_steer(cfg, circle)])  # car 0 turns into the barrier, car 1 holds the circle
    target = torch.tensor([30.0, 4.0])  # car 1 crawls: 14 km/h, just above the 3 m/s stuck floor

    old_env = CarEnv(2, CPU, replace(cfg, episode_steps=0, unfinished_per_m=0.0), track=circle)
    old_env.reset()
    old = drive(old_env, budget, steer, target)
    new_env = CarEnv(2, CPU, replace(cfg, episode_steps=budget), track=circle)
    new_env.reset()
    new = drive(new_env, budget, steer, target, monitor=True)

    for run in (old, new):
        assert int(run["reason"][0]) == DONE_CRASH and int(run["end_step"][0]) > 0
        assert int(run["reason"][1]) == DONE_ALIVE and int(run["end_step"][1]) == -1
    # The flaw: with neither the budget charge nor the unfinished-lap forfeit, the crash on
    # step ~60 outscored 2,000 steps of driving.
    assert float(old["total"][0]) > float(old["total"][1])
    # Fixed: the survivor is ahead, and by the crash penalty plus its progress.
    assert float(new["total"][1]) > float(new["total"][0])
    k = int(new["end_step"][0])
    terminal = float(new["rewards"][k - 1, 0])
    # The ended car is charged as standing still for the rest of the budget: time tax plus
    # the full pace and alignment penalties per unused step, on top of the crash penalty.
    per_step = cfg.time_tax + cfg.pace_bonus + cfg.pace_penalty + cfg.align_penalty
    # ...plus the metres of the lap it never drove, at `unfinished_per_m`.
    laps_at_crash = float(new["deltas"][:k, 0].sum())
    forfeit = cfg.unfinished_per_m * (cfg.max_laps - laps_at_crash) * new_env.track.length_m
    assert terminal == pytest.approx(-(cfg.crash_penalty + forfeit + per_step * (budget - k)), abs=1e-2)
    # No survivor pays more per step than the standing-still rate, so the crasher's total
    # is below what any survivor with the same progress could score.
    survivor_driven = float(new["deltas"][:, 1].sum())
    survivor_forfeit = cfg.unfinished_per_m * (cfg.max_laps - survivor_driven) * new_env.track.length_m
    survivor_floor = survivor_driven * new_env.progress_scale - per_step * budget - survivor_forfeit
    assert float(new["total"][1]) >= survivor_floor - 1e-3
    # The crasher: progress minus the time tax while alive, the crash penalty, and the
    # standing-still rate for the unused budget (its pace/alignment terms while alive are <= 0).
    crasher_ceiling = float(new["deltas"][:, 0].sum()) * new_env.progress_scale - cfg.time_tax * k - cfg.crash_penalty - per_step * (budget - k)
    assert float(new["total"][0]) <= crasher_ceiling + 1e-3
    # The exploit monitor accounts for every term and the terminal charge: honest driving raises nothing.
    report = new["report"]
    assert report["flags"]["accounting"] == 0 and report["flags"]["over_bound"] == 0 and report["flags"]["idle_reward"] == 0, report


def test_budget_zero_charges_the_unfinished_lap(circle):
    env = CarEnv(1, CPU, CIRCLE, track=circle)
    env.reset()
    run = drive(env, 400, torch.tensor([1.0]), torch.tensor([30.0]))
    k = int(run["end_step"][0])
    laps_at_crash = float(run["deltas"][:k, 0].sum())
    forfeit = CIRCLE.unfinished_per_m * (CIRCLE.max_laps - laps_at_crash) * env.track.length_m
    assert k > 0 and float(run["rewards"][k - 1, 0]) == pytest.approx(-(CIRCLE.crash_penalty + forfeit), abs=1e-2)
    assert float(env.last_terms["crash_cost"][0]) == pytest.approx(CIRCLE.crash_penalty + forfeit, abs=1e-2)


def test_driving_one_more_metre_always_beats_ending_there(circle):
    """Uncapped episodes: the forfeit must make every extra metre worth more than its charges."""
    cfg = replace(CIRCLE, episode_steps=0)
    steer = torch.tensor([circle_steer(cfg, circle)] * 2)
    env = CarEnv(2, CPU, cfg, track=circle)
    env.reset()
    run = drive(env, 900, steer, torch.tensor([21.0, 21.0]))
    # Score of ending at step i = reward banked so far, minus the crash charge owed there.
    banked = run["rewards"][:, 0].cumsum(0)
    driven = run["deltas"][:, 0].cumsum(0) * env.track.length_m
    owed = cfg.crash_penalty + cfg.unfinished_per_m * (cfg.max_laps * env.track.length_m - driven)
    score_if_ending_here = banked - owed
    # The last recorded step is the crash itself (its reward is already the charge).
    k = int(run["end_step"][0])
    last = k - 1 if k > 0 else score_if_ending_here.shape[0]
    gains = score_if_ending_here[1:last] - score_if_ending_here[: last - 1]
    # Ignore the launch, where the car is below the pace the charges assume.
    assert float(gains[60:].min()) > 0.0, float(gains[60:].min())


def test_running_the_budget_out_owes_the_unfinished_lap(circle):
    budget = 300
    cfg = replace(CIRCLE, episode_steps=budget)
    env = CarEnv(1, CPU, cfg, track=circle)
    env.reset()
    run = drive(env, budget, torch.tensor([circle_steer(cfg, circle)]), torch.tensor([20.0]))
    assert int(run["end_step"][0]) == -1  # still driving when the budget ran out
    laps = float(env.laps[0])
    forfeit = cfg.unfinished_per_m * (cfg.max_laps - laps) * env.track.length_m
    last, before = float(run["rewards"][-1, 0]), float(run["rewards"][-2, 0])
    assert last == pytest.approx(before - forfeit, abs=0.2)


def test_the_same_distance_driven_sooner_scores_higher(circle):
    """Fastest-lap pressure: time is the only thing separating two cars that get equally far."""
    cfg = replace(CIRCLE, episode_steps=0)
    speeds = torch.tensor([15.0, 25.0, 35.0])
    steer = torch.full((3,), circle_steer(cfg, circle))
    env = CarEnv(3, CPU, cfg, track=circle)
    env.reset()
    run = drive(env, 4000, steer, speeds)
    driven = run["deltas"].sum(0) * env.track.length_m
    assert float(driven.std() / driven.mean()) < 0.05, driven  # same distance, different times
    totals = run["total"]
    assert float(totals[0]) < float(totals[1]) < float(totals[2]), totals


def test_apex_margin_leaves_most_of_the_road_untaxed():
    cfg = CarConfig()
    # A W11 body at the centre of an 11 m road has 4.5 m to the barrier; a line
    # brushing the barrier at 1 m must not be taxed more than the progress it earns.
    assert cfg.wall_margin <= 0.75
    clearance = torch.tensor([1.0, 0.75, 0.5, 0.0])
    near = (1.0 - clearance / cfg.wall_margin).clamp(0.0, 1.0)
    wall = cfg.wall_penalty * near * near
    progress_per_step_at_15_mps = cfg.progress_per_m * 15.0 * cfg.dt_s
    assert float(wall[0]) == 0.0 and float(wall[1]) == 0.0
    assert float(wall[2]) < 0.2 * progress_per_step_at_15_mps


def test_ended_cars_are_frozen(circle):
    env = CarEnv(2, CPU, CIRCLE, track=circle)
    env.reset()
    steer = torch.tensor([1.0, circle_steer(CIRCLE, circle)])
    run = drive(env, 120, steer, torch.tensor([30.0, 15.0]))
    k = int(run["end_step"][0])
    assert 0 < k < 120 and int(env.done_reason[0]) == DONE_CRASH
    assert int(env.done_reason[1]) == DONE_ALIVE, "the circle driver must still be running"
    frozen = (run["poses"][k - 1, 0], run["deltas"][k:, 0], run["rewards"][k:, 0], run["speeds"][k:, 0])
    assert torch.equal(run["poses"][k:, 0], frozen[0].expand_as(run["poses"][k:, 0]))
    assert bool((frozen[1] == 0).all()) and bool((frozen[2] == 0).all())
    assert bool((frozen[3] == frozen[3][0]).all())
    # A masked reset is the only way back.
    env.reset(torch.tensor([True, False]))
    assert int(env.done_reason[0]) == DONE_ALIVE and float(env.speed[0]) == 0.0


def test_pace_is_paid_and_rises_with_the_speed_the_road_allows(circle):
    slow = replace(CIRCLE, max_speed=20.0)  # the circle's reference speed caps at 20 m/s
    track = Track(build_centerline(slow, 1), slow, CPU)
    assert float(track.speed_ref.min()) == pytest.approx(20.0, abs=0.5)
    env = CarEnv(1, CPU, slow, track=track)
    env.reset()
    steer = torch.tensor([circle_steer(slow, track)])
    seen = []
    for _ in range(400):
        pedal = torch.where(env.speed < 19.5, torch.ones(1), torch.full((1,), -0.2))
        env.step(torch.stack([steer, pedal], dim=1))
        seen.append((float(env.speed[0]), float(env.last_terms["pace"][0])))
    assert int(env.done_reason[0]) == DONE_ALIVE, "the scripted driver must hold the circle at 19.5 m/s"
    early = [p for v, p in seen if v < 5.0]
    mid = [p for v, p in seen if 8.0 < v < 12.0]
    late = [p for v, p in seen if v >= 0.9 * 20.0]
    assert early and mid and late
    # Paid, not charged, and worth more the closer the car is to what the road allows.
    assert 0.0 <= max(early) < min(mid) < max(mid) < min(late)
    assert max(late) == pytest.approx(slow.pace_bonus, abs=5e-3)
    # Standing still earns nothing: the pace term must never pay for idling.
    env.reset()
    env.step(torch.tensor([[0.0, 0.0]]))
    assert float(env.last_terms["pace"][0]) == pytest.approx(0.0, abs=1e-6)


def test_beating_the_centerline_reference_pays_up_to_the_cap(circle):
    """A car that straightens a corner with the full width of the road may pass the reference."""
    cfg = replace(CIRCLE, max_speed=20.0)
    track = Track(build_centerline(cfg, 1), cfg, CPU)
    # The reference stays at the slow track's 20 m/s; the car itself may exceed it.
    env = CarEnv(2, CPU, replace(cfg, max_speed=60.0), track=track)
    env.reset()
    ref = float(track.speed_ref[0])
    env.speed = torch.tensor([ref, ref * (cfg.pace_cap + 0.4)])
    env.step(torch.zeros(2, 2))
    at_ref, above = (float(v) for v in env.last_terms["pace"])
    assert above > at_ref == pytest.approx(cfg.pace_bonus, abs=5e-3)
    assert above == pytest.approx(cfg.pace_bonus * cfg.pace_cap**2, abs=5e-3)



def test_alignment_term_is_graded_in_heading_error(circle):
    env = CarEnv(5, CPU, CIRCLE, track=circle)
    env.reset()
    offsets = torch.tensor([0.0, 0.3, 0.6, 1.2, math.pi])
    env.heading = env.heading + offsets
    env.step(torch.zeros(5, 2))
    align = env.last_terms["align"]
    assert float(align[0]) == pytest.approx(0.0, abs=0.002), "pointing along the road costs nothing"
    assert bool((align[1:] < align[:-1] + 1e-6).all()), "more heading error must cost more"
    assert float(align[4]) == pytest.approx(-2 * CIRCLE.align_penalty, abs=0.003), "facing backwards costs twice the penalty"
    assert float(align[2]) == pytest.approx(-CIRCLE.align_penalty * (1 - math.cos(0.6)), abs=0.003)


MONACO = ROOT / "data" / "tracks" / "monaco.geojson"


@pytest.mark.skipif(not MONACO.exists(), reason="Monaco GeoJSON not downloaded")
def test_monaco_reference_profile_is_a_plausible_pole_lap():
    from car_env import load_geojson_centerline, monaco_config, speed_profile

    cfg = monaco_config(0.016)
    centerline = load_geojson_centerline(MONACO, cfg.track_scale, cfg.n_points, cfg.smooth_m).numpy()
    v = speed_profile(centerline, cfg)
    seg = ((centerline - centerline[[*range(1, len(centerline)), 0]]) ** 2).sum(1) ** 0.5
    lap_s = float((seg / v).sum())
    assert 60.0 < lap_s < 110.0, f"reference lap {lap_s:.1f} s; the real W11 pole is ~70 s"
    assert 8.0 < float(v.min()) < 25.0, "the Fairmont hairpin is a 45-80 km/h corner"
    assert float(v.max()) <= cfg.max_speed and float(v.max()) > 60.0, "the tunnel run must be fast"
    assert bool((v > 0).all())


# --- eye geometry -------------------------------------------------------------
def test_rays_report_the_road_edge_not_the_coarse_sample_past_it():
    """The march locates the wall to a sample; the refinement puts it on the edge."""
    from dataclasses import replace as _replace

    from car_env import CarEnv

    cfg = CarConfig(grid_res=1024)
    coarse = _replace(cfg, ray_refine_steps=0)
    starts = torch.linspace(0.0, 0.9, 8)
    fine_env = CarEnv(8, CPU, cfg, seed=0, start_fraction=starts)
    obs = fine_env.reset()
    coarse_env = CarEnv(8, CPU, coarse, seed=0, start_fraction=starts, track=fine_env.track)
    obs_coarse = coarse_env.reset()

    # ground truth: march the same rays at 2.5 cm
    step = torch.arange(1, 4001).float() * 0.025
    angles = fine_env.heading.unsqueeze(1) + fine_env.ray_angles.unsqueeze(0)
    direction = torch.stack([angles.cos(), angles.sin()], dim=-1)
    points = fine_env.pos[:, None, None, :] + direction[:, :, None, :] * step[None, None, :, None]
    blocked = ~fine_env.track.is_drivable(points)
    hit = torch.where(blocked.any(-1), blocked.float().argmax(-1), torch.full_like(blocked[..., 0], 3999, dtype=torch.long))
    truth = step[hit]

    n = cfg.n_rays
    # only rays whose wall is inside the reference march's 100 m reach
    seen = truth < 90.0
    refined = (obs[:, :n] * cfg.max_range - truth).abs()[seen]
    coarse_dist = (obs_coarse[:, :n] * cfg.max_range - truth)[seen]
    assert seen.sum() > 50
    # A grazing ray can still skip a thin off-road wedge between two march
    # samples, so the tail is judged separately from the typical ray.
    assert float(refined.median()) < 0.1, float(refined.median())
    assert float(refined.median()) < float(coarse_dist.abs().median()) / 5
    assert float(refined.mean()) < float(coarse_dist.abs().mean())
    # the coarse march can only overshoot: it reports the first sample past the edge
    assert float(coarse_dist.min()) > -0.05


def test_the_eye_resolves_the_road_at_ten_degrees():
    """19 rays over 180 degrees: neighbours are 10 degrees apart, not 22.5."""
    cfg = CarConfig()
    assert cfg.n_rays == 19
    assert cfg.fov_deg / (cfg.n_rays - 1) == 10.0
