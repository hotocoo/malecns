"""Regression tests for the saturation failure seen at generation ~2,100.

The ES mean drifted to |w_out| ~ 70 and steer bias -30, pinning tanh at -1 so
the car steered hard left forever, U-turned and drove the loop backwards.
These tests pin the two guards added against that: parameter bounds and
episode termination on net reverse progress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import AgentConfig, ConnectomeAgent  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402
from car_env import CarConfig, CarEnv  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
needs_graph = pytest.mark.skipif(
    not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first"
)
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def agent():
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU)
    return ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=1))


@needs_graph
def test_initial_params_already_satisfy_bounds(agent):
    init = agent.initial_params()
    assert torch.equal(agent.clamp_params(init.unsqueeze(0))[0], init)


@needs_graph
def test_clamp_pins_the_saturated_checkpoint_back_into_range(agent):
    exploded = agent.initial_params() * 0 + 70.0
    exploded[-2:] = torch.tensor([-30.0, 36.0])  # b_out as found at gen 2151
    theta = agent.unpack(agent.clamp_params(exploded.unsqueeze(0)))
    eps = 1e-6  # bounds are python floats, params are float32
    for name, (lo, hi) in agent.PARAM_BOUNDS.items():
        if name in theta:
            assert float(theta[name].min()) >= lo - eps
            assert float(theta[name].max()) <= hi + eps
    # the bias alone can no longer saturate the steer squash
    assert abs(float(torch.tanh(theta["b_out"][0, 0]))) < 0.95


@needs_graph
def test_clamp_is_batched_and_leaves_in_range_values_alone(agent):
    base = agent.initial_params().unsqueeze(0).repeat(3, 1)
    base[1] += 1000.0
    out = agent.clamp_params(base)
    assert out.shape == base.shape
    assert torch.equal(out[0], base[0])
    assert torch.equal(out[2], base[2])
    assert float(out[1].max()) <= 3.0


def test_driving_backwards_ends_the_episode():
    env = CarEnv(1, CPU, CarConfig(), seed=1)
    env.reset()
    env.heading = env.heading + torch.pi  # face the wrong way down the track
    done = torch.tensor([False])
    for _ in range(400):
        _, reward, done = env.step(torch.tensor([[0.0, 1.0]]))
        if bool(done[0]):
            break
    assert bool(done[0]), "reverse driver should be terminated"
    assert float(env.laps[0]) <= env.cfg.reverse_limit_laps
    assert float(reward[0]) == -env.cfg.crash_penalty


def test_forward_driving_is_not_affected_by_reverse_guard():
    env = CarEnv(1, CPU, CarConfig(), seed=1)
    env.reset()
    for _ in range(60):
        _, _, done = env.step(torch.tensor([[0.0, 1.0]]))
        assert not bool(done[0])
    assert float(env.laps[0]) > 0.0
