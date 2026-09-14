"""Checkpoints from older parameter layouts load block by block, not from scratch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import asdict  # noqa: E402

from agent import AgentConfig, ConnectomeAgent  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
needs_graph = pytest.mark.skipif(not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first")
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def agent():
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU, use_metal=False)
    return ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=1))


@needs_graph
def test_legacy_141_param_checkpoint_keeps_trained_blocks(agent):
    assert agent.n_params == 142
    legacy = torch.arange(141, dtype=torch.float32) / 100.0
    state = {"mu": legacy, "momentum": torch.ones(141), "agent_cfg": asdict(agent.cfg)}
    mu, momentum, notes = agent.migrate_state(state)
    assert mu.numel() == 142 and momentum.numel() == 142
    assert notes == ["loom_gain from init"]
    theta = agent.unpack(mu.unsqueeze(0))
    assert torch.equal(theta["ray_gain"][0], legacy[:9])
    assert torch.equal(theta["bias_hz"][0], legacy[9:10])
    assert torch.equal(theta["speed_gain"][0], legacy[10:11])
    assert torch.equal(theta["w_out"][0].reshape(-1), legacy[11:139])
    assert torch.equal(theta["b_out"][0], legacy[139:141])
    assert float(theta["loom_gain"][0]) == 0.0  # init value
    # no momentum on the block that was never trained
    m = agent.unpack(momentum.unsqueeze(0))
    assert float(m["loom_gain"][0]) == 0.0 and float(m["ray_gain"][0].sum()) == 9.0


@needs_graph
def test_named_layout_migration_reorders_and_drops(agent):
    shapes = {"b_out": [2], "ray_gain": [9], "extra": [5]}
    saved = torch.cat([torch.tensor([7.0, 8.0]), torch.full((9,), 3.0), torch.zeros(5)])
    mu, notes = agent.migrate_params(saved, shapes)
    theta = agent.unpack(mu.unsqueeze(0))
    assert torch.equal(theta["b_out"][0], torch.tensor([7.0, 8.0]))
    assert torch.equal(theta["ray_gain"][0], torch.full((9,), 3.0))
    assert "extra dropped" in notes
    assert any(n.startswith("w_out from init") for n in notes)


@needs_graph
def test_same_layout_roundtrips_exactly(agent):
    mu = agent.initial_params() + 0.01
    state = {
        "mu": mu,
        "momentum": torch.zeros_like(mu),
        "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()},
        "agent_cfg": asdict(agent.cfg),
    }
    out, _, notes = agent.migrate_state(state)
    assert torch.equal(out, mu) and notes == []


@needs_graph
def test_readout_from_another_scale_is_reset(agent):
    """A readout evolved at another readout_scale pins tanh here; only the sensory gains carry over."""
    legacy = torch.full((141,), 2.0)
    for state in (
        {"mu": legacy, "momentum": torch.ones(141)},  # nothing recorded: pre-audit trainer
        {"mu": legacy, "momentum": torch.ones(141), "agent_cfg": {**asdict(agent.cfg), "readout_scale": 50.0}},
    ):
        mu, momentum, notes = agent.migrate_state(state)
        theta = agent.unpack(mu.unsqueeze(0))
        assert torch.equal(theta["ray_gain"][0], torch.full((9,), 2.0))
        # w_out is drawn fresh (small random), b_out is the fixed init: neither is the saved 2.0
        assert float(theta["w_out"][0].abs().mean()) < 0.3 and not (theta["w_out"][0] == 2.0).any()
        assert torch.equal(theta["b_out"][0], torch.tensor([0.0, 0.5]))
        assert any("w_out" in n and "reset" in n for n in notes)
        assert float(momentum.abs().sum()) == 0.0


@needs_graph
def test_explicit_reset_blocks(agent):
    mu0 = agent.initial_params() + 1.0
    state = {"mu": mu0, "momentum": torch.ones_like(mu0), "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()}, "agent_cfg": asdict(agent.cfg)}
    mu, _, notes = agent.migrate_state(state, reset=("ray_gain",))
    theta = agent.unpack(mu.unsqueeze(0))
    assert torch.equal(theta["ray_gain"][0], torch.full((9,), 0.5))  # init
    assert torch.equal(theta["bias_hz"][0], agent.unpack(mu0.unsqueeze(0))["bias_hz"][0])
    assert notes == ["ray_gain (reset)"]


@needs_graph
def test_unknown_size_without_layout_is_an_error(agent):
    with pytest.raises(ValueError):
        agent.migrate_params(torch.zeros(77), None)
