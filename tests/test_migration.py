"""Checkpoints from older parameter layouts load block by block, not from scratch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    mu, momentum, notes = agent.migrate_state({"mu": legacy, "momentum": torch.ones(141)})
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
    state = {"mu": mu, "momentum": torch.zeros_like(mu), "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()}}
    out, _, notes = agent.migrate_state(state)
    assert torch.equal(out, mu) and notes == []


@needs_graph
def test_unknown_size_without_layout_is_an_error(agent):
    with pytest.raises(ValueError):
        agent.migrate_params(torch.zeros(77), None)
