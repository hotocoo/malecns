"""Road eye encoding, calibrated readout plumbing, linear teacher and ridge fit."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import calibrate  # noqa: E402
from agent import AgentConfig, ConnectomeAgent  # noqa: E402
from car_env import CarConfig  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402
from teacher import LinearTeacher  # noqa: E402

GRAPH = Path("data/graph_w5")
needs_graph = pytest.mark.skipif(not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first")
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def agent():
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=2, config=LIFConfig(dt_ms=2.0), device=CPU, use_metal=False)
    return ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=1, readout_norm="channel"))


@needs_graph
def test_road_eyes_read_each_ray_against_its_expected_distance(agent):
    # Centred on a straight 11 m road: side rays see 5.5 m, the 45 deg rays 7.8 m -> every group at 0.5.
    angles = torch.linspace(math.pi / 2, -math.pi / 2, agent.cfg.n_rays)
    cfg = agent.cfg
    centred = torch.minimum(cfg.eye_halfwidth_m / angles.sin().abs().clamp(min=1e-3), torch.tensor(cfg.eye_front_ref_m)) / cfg.eye_range_m
    prox = agent.proximity(centred.unsqueeze(0))[0]
    assert torch.allclose(prox, torch.full((agent.cfg.n_rays,), 0.5), atol=1e-4)
    # Twice as close saturates (0.4/octave -> 0.9 at one octave, 1.0 at 1.25), twice as far goes quiet.
    assert float(agent.proximity((centred / 2).unsqueeze(0))[0, 0]) == pytest.approx(0.9, abs=1e-3)
    assert float(agent.proximity((centred * 4).unsqueeze(0))[0, 0]) == pytest.approx(0.0, abs=1e-6)
    # A wall nearer on the left than the right shows up as a left-minus-right difference of the *same* size at 4 vs 7 m.
    left_near = centred.clone()
    left_near[0], left_near[-1] = 4.0 / 150.0, 7.0 / 150.0
    p = agent.proximity(left_near.unsqueeze(0))[0]
    assert float(p[0] - p[-1]) > 0.3


@needs_graph
def test_calibrated_readout_roundtrips_through_a_checkpoint(agent):
    assert not agent.readout_calibrated and agent.readout_state() is None
    projection = torch.randn(agent.n_readout, agent.cfg.readout_dim)
    mean, std = torch.randn(agent.cfg.readout_dim), torch.rand(agent.cfg.readout_dim) + 0.5
    agent.set_readout(projection, mean, std)
    state = {"agent_cfg": {**AgentConfig(readout_norm="channel").__dict__}, "readout": agent.readout_state()}
    signal = torch.randn(2, agent.n_readout)
    z = agent.channels(signal)
    assert torch.allclose(z, (signal @ projection - mean) / std, atol=1e-5)
    # a fresh agent must refuse to trust readout blocks until the projection is installed
    fresh_flag = agent.readout_calibrated
    agent.readout_calibrated = False
    assert not agent.readout_matches(state)
    agent.readout_calibrated = fresh_flag
    assert agent.readout_matches(state)
    assert agent.load_readout({}) is False
    assert agent.load_readout(state) is True
    with pytest.raises(ValueError):
        agent.set_readout(torch.zeros(3, 3), mean, std)


def test_linear_teacher_steers_away_from_the_nearer_wall_and_brakes_for_a_wall_ahead():
    teacher = LinearTeacher()
    n = CarConfig().n_rays
    centred = torch.full((1, n), 0.5)
    left_wall = centred.clone()
    left_wall[0, : n // 2] = 0.9
    right_wall = centred.clone()
    right_wall[0, n // 2 + 1 :] = 0.9
    slow = torch.tensor([0.1])
    assert float(teacher.act(left_wall, slow)[0, 0]) < -0.2  # steer right (negative)
    assert float(teacher.act(right_wall, slow)[0, 0]) > 0.2
    assert float(teacher.act(centred, slow)[0, 0]) == pytest.approx(0.0, abs=1e-6)
    ahead = centred.clone()
    ahead[0, 3:6] = 1.0
    assert float(teacher.act(ahead, slow)[0, 1]) < float(teacher.act(centred, slow)[0, 1])
    fast = torch.tensor([0.9])
    assert float(teacher.act(centred, fast)[0, 1]) < 0.0  # governor brakes above the target speed
    with pytest.raises(ValueError):
        LinearTeacher([1.0, 2.0])


def test_ridge_recovers_a_linear_map_and_picks_a_penalty_by_held_out_cars():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4000, 20, generator=g)
    w_true = torch.randn(20, 2, generator=g)
    b_true = torch.tensor([0.3, -0.2])
    y = x @ w_true + b_true + 0.01 * torch.randn(4000, 2, generator=g)
    w, b = calibrate.ridge(x, y, 1e-3)
    assert torch.allclose(w, w_true, atol=0.02) and torch.allclose(b, b_true, atol=0.02)
    car = torch.arange(4000) % 6
    w2, b2, lam, r2 = calibrate.fit_readout(x, y, car, (1e-3, 1.0, 1e4))
    assert lam < 1e4 and float(r2.min()) > 0.99
