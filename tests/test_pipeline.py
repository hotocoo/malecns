"""Behavioural tests for the connectome, brain, environment and interface.

Run with:  python3 -m pytest tests -q
The connectome tests need `data/graph_w5`; they skip if it has not been built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import AgentConfig, ConnectomeAgent  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402
from car_env import CarConfig, CarEnv, make_centerline  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
needs_graph = pytest.mark.skipif(
    not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first"
)
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def connectome():
    return load_connectome(GRAPH)


@needs_graph
def test_graph_is_signed_and_complete(connectome):
    assert connectome.n > 100_000
    assert len(connectome.pre) > 1_000_000
    assert (connectome.weight > 0).any() and (connectome.weight < 0).any()
    assert connectome.pre.max() < connectome.n
    assert connectome.post.max() < connectome.n


@needs_graph
def test_every_role_is_populated(connectome):
    for role in ("photoreceptor", "visual_projection", "descending", "motor"):
        assert len(connectome.roles[role]) > 100


@needs_graph
def test_network_is_silent_without_input(connectome):
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU)
    fired = sum(float(brain.step().sum()) for _ in range(25))
    assert fired == 0.0, "no basal firing is expected without drive"


@needs_graph
def test_firing_rate_increases_with_drive(connectome):
    brain = Brain(
        connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU, weight_scale=0.15
    )
    index = brain.role_index("visual_projection")

    def mean_rate(prob: float) -> float:
        brain.reset()
        torch.manual_seed(0)
        total = 0.0
        for step in range(60):
            external = torch.zeros(1, brain.n)
            external[:, index] = (torch.rand(1, index.numel()) < prob).float() * 8.0
            spikes = brain.step(external)
            if step >= 30:
                total += float(spikes.mean())
        return total / 30

    assert mean_rate(0.4) > mean_rate(0.02)


def test_track_geometry_and_progress():
    env = CarEnv(2, CPU, CarConfig(), seed=3)
    start = env.track.centerline[0].unsqueeze(0)
    assert bool(env.track.is_drivable(start)[0])
    far = torch.tensor([[0.0, 0.0]])  # loop interior is not drivable
    assert not bool(env.track.is_drivable(far)[0])
    assert 0.0 <= float(env.track.progress_at(start)[0]) <= 1.0


def test_driving_forward_earns_progress_reward():
    env = CarEnv(4, CPU, CarConfig(), seed=1)
    env.reset()
    action = torch.tensor([[0.0, 0.5]]).repeat(4, 1)
    rewards = [float(env.step(action)[1].mean()) for _ in range(120)]
    assert not bool(env.step(action)[2].any()), "gentle straight driving stays on the road"
    assert float(env.laps.min()) > 0.0, "forward driving accumulates lap fraction"
    standing_still = -(env.cfg.time_tax + env.cfg.pace_penalty) * 120  # time tax plus full pace deficit
    assert sum(rewards) > standing_still, "progress pays above standing still"


def test_idling_is_penalised():
    env = CarEnv(1, CPU, CarConfig(), seed=1)
    env.reset()
    action = torch.tensor([[0.0, 0.0]])
    total = sum(float(env.step(action)[1][0]) for _ in range(10))
    assert total < 0.0


def test_observation_shape_and_range():
    env = CarEnv(3, CPU, CarConfig(), seed=2)
    obs = env.reset()
    assert obs.shape == (3, env.obs_dim)
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0


def test_centerline_is_a_closed_loop():
    points = make_centerline(7)
    gap = float(torch.linalg.norm(points[0] - points[-1]))
    step = float(torch.linalg.norm(points[1] - points[0]))
    assert gap < 3 * step


@needs_graph
def test_agent_parameter_roundtrip(connectome):
    brain = Brain(connectome, batch=2, config=LIFConfig(dt_ms=2.0), device=CPU)
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=1))
    params = agent.initial_params().unsqueeze(0).repeat(2, 1)
    assert params.shape[1] == agent.n_params
    theta = agent.unpack(params)
    assert theta["w_out"].shape == (2, agent.cfg.readout_dim, 2)
    assert agent.projection.shape == (agent.n_readout, agent.cfg.readout_dim)
    assert theta["ray_gain"].shape == (2, agent.cfg.n_rays)


@needs_graph
def test_agent_produces_bounded_actions(connectome):
    brain = Brain(
        connectome, batch=2, config=LIFConfig(dt_ms=2.0), device=CPU, weight_scale=0.15
    )
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=2))
    theta = agent.unpack(agent.initial_params().unsqueeze(0).repeat(2, 1))
    env = CarEnv(2, CPU, CarConfig(dt_s=0.004), seed=0)
    obs = env.reset()
    agent.reset()
    for _ in range(3):
        action = agent.act(obs, theta)
        assert action.shape == (2, 2)
        assert float(action[:, 0].abs().max()) <= 1.0
        assert -1.0 <= float(action[:, 1].min()) and float(action[:, 1].max()) <= 1.0
        obs = env.step(action)[0]


@needs_graph
def test_visual_groups_partition_the_input_population(connectome):
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU)
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig())
    covered = np.concatenate([g.numpy() for g in agent.ray_groups])
    assert len(covered) == len(set(covered.tolist()))
    assert set(covered.tolist()) == set(brain.roles[agent.cfg.input_role])
