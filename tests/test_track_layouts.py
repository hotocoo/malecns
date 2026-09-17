"""Tests for real-circuit loading, shared tracks and the reshaped reward."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_env import (  # noqa: E402
    CarConfig,
    CarEnv,
    Track,
    build_centerline,
    load_geojson_centerline,
    monaco_config,
    resample_closed,
)

MONACO = ROOT / "data" / "tracks" / "monaco.geojson"
needs_monaco = pytest.mark.skipif(not MONACO.exists(), reason="monaco.geojson not downloaded")
CPU = torch.device("cpu")


def test_resample_closed_is_even_and_closed():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    out = resample_closed(square, 200)
    step = np.hypot(*np.diff(out, axis=0, append=out[:1]).T)
    assert out.shape == (200, 2)
    assert step.std() / step.mean() < 0.05, "arc-length sampling should be near uniform"
    assert step[-1] < 3 * step.mean(), "last point closes back onto the first"


def test_geojson_loader_projects_scales_and_orients_clockwise(tmp_path):
    # a counter-clockwise circle about 1 km across near Monaco's latitude;
    # 64 points, like a surveyed circuit, so the spline barely overshoots
    lon0, lat0 = 7.42, 43.73
    r_lon, r_lat = 0.0062, 0.0045
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = [[lon0 + r_lon * np.cos(a), lat0 + r_lat * np.sin(a)] for a in ang]
    ring.append(ring[0])
    path = tmp_path / "sq.geojson"
    path.write_text(json.dumps({"features": [{"geometry": {"type": "LineString", "coordinates": ring}}]}))

    pts = load_geojson_centerline(path, scale=0.5, n_points=400).numpy()

    assert pts.shape == (400, 2)
    assert np.allclose(pts.mean(axis=0), 0.0, atol=1.0)
    span = pts.max(axis=0) - pts.min(axis=0)
    assert 450 < span[0] < 550 and 450 < span[1] < 550, f"~1 km * 0.5 expected, got {span}"
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    assert area < 0, "must be re-oriented clockwise"


@needs_monaco
def test_monaco_track_has_real_length_and_fits_its_grid():
    cfg = monaco_config(dt_s=0.016)
    track = Track(build_centerline(cfg), cfg, CPU)
    assert 3250 < track.length_m < 3340, "3.337 km circuit, corners smoothed"
    assert track.min_radius_m > 6.0, "hairpin must stay drivable after smoothing"
    assert track.extent > cfg.grid_extent, "extent must grow to hold the circuit"
    assert track.cell < 0.7, "cells must stay finer than the 11 m road"
    assert bool(track.is_drivable(track.centerline[:50]).all())


@needs_monaco
def test_start_fractions_share_one_track_and_start_on_it():
    cfg = monaco_config(dt_s=0.016)
    track = Track(build_centerline(cfg), cfg, CPU)
    envs = [CarEnv(2, CPU, cfg, track=track, start_fraction=f) for f in (0.0, 0.5)]
    assert envs[0].track is envs[1].track
    assert int(envs[0].start_index[0]) != int(envs[1].start_index[0])
    for env in envs:
        obs = env.reset()
        assert bool(track.is_drivable(env.pos).all())
        assert obs.shape == (2, env.obs_dim)
        _, reward, done = env.step(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        assert not bool(done.any())


def test_reward_pays_progress_not_speed_and_ramps_near_wall():
    cfg = CarConfig()
    env = CarEnv(2, CPU, cfg, seed=1)
    env.reset()
    for _ in range(30):  # get rolling first
        env.step(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    # car 1 hugs the wall: shift it sideways so its body edge is 0.5 m from it
    start, heading = env._start_pose(env.start_index)
    normal = torch.stack([-heading.sin(), heading.cos()], dim=1)
    env.pos[1] = env.pos[0] + normal[0] * (cfg.track_halfwidth - cfg.car_halfwidth - 0.5)
    _, reward, done = env.step(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert not bool(done.any())
    assert float(reward[1]) < float(reward[0]), "wall proximity must cost something"
    assert float(reward[0]) > -(cfg.time_tax + cfg.pace_penalty), "moving forward beats standing still (time tax plus full pace deficit)"


def test_completing_a_lap_pays_the_bonus():
    # Off by default; the payment path is still tested with it turned on.
    cfg = CarConfig(lap_bonus=100.0)
    env = CarEnv(1, CPU, cfg, seed=2)
    env.reset()
    n = env.track.centerline.shape[0]
    env.laps = torch.tensor([0.999])
    env.last_progress = torch.tensor([(n - 1) / n])
    # just past the start line, one sample on, moving forward
    env.pos = env.track.centerline[1].unsqueeze(0).clone()
    env.heading = torch.atan2(*(env.track.centerline[10] - env.track.centerline[1]).flip(0).unsqueeze(1))
    _, reward, _ = env.step(torch.tensor([[0.0, 1.0]]))
    assert float(env.laps[0]) >= 1.0
    assert float(reward[0]) > cfg.lap_bonus * 0.9
    assert n > 0


@needs_monaco
def test_monaco_track_carries_the_surveyed_grade_and_gravity_acts_on_the_car():
    """With `monaco_dem.json` beside the circuit the track climbs and a coasting car rolls downhill."""
    from pathlib import Path as _P

    if not _P("data/tracks/monaco_dem.json").exists():
        pytest.skip("run src/fetch_terrain.py first")
    cfg = monaco_config(0.016)
    track = Track(build_centerline(cfg), cfg, CPU)
    assert track.height is not None and 35.0 <= track.climb_m <= 65.0
    grade = track.grade
    assert 0.05 < float(grade.abs().max()) < 0.2, "Monaco's steepest stretch is a real hill, not a cliff"
    # the same car coasting from the same speed: downhill it gains on the flat-world twin
    steep = int(grade.argmin())  # most downhill sample (grade < 0 means descending along the lap)
    flat_cfg = CarConfig(**{**cfg.__dict__, "road_grade": False})
    flat = Track(build_centerline(flat_cfg), flat_cfg, CPU)
    assert float(flat.grade.abs().max()) == 0.0
    speeds = []
    for tr, c in ((track, cfg), (flat, flat_cfg)):
        env = CarEnv(1, CPU, c, track=tr, start_fraction=[steep / tr.centerline.shape[0]])
        env.reset()
        env.speed[:] = 15.0
        for _ in range(25):  # 0.4 s of coasting
            env.step(torch.tensor([[0.0, 0.0]]))
        speeds.append(float(env.speed[0]))
    # g * grade * t at a ~15 % grade over 0.4 s is ~0.6 m/s
    assert speeds[0] > speeds[1] + 0.3, f"downhill {speeds[0]:.2f} m/s should beat flat {speeds[1]:.2f} m/s"
