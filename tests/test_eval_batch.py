"""The brain runs at a reduced batch for deterministic evaluation (`Brain.reset(batch)`).

`train.evaluate_mean` drives one body per (island, start) instead of the brain's
full batch; the reduced batch must compute exactly what the first bodies of the
full batch compute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain import Brain, LIFConfig, load_connectome  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
needs_graph = pytest.mark.skipif(not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first")
needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal kernel needs MPS device")
MPS = torch.device("mps")


@pytest.fixture(scope="module")
def connectome():
    return load_connectome(GRAPH)


def _run(brain: Brain, idx: torch.Tensor, kicks: torch.Tensor, steps: int) -> torch.Tensor:
    spikes = torch.zeros(brain.batch, brain.n, device=brain.device)
    for _ in range(steps):
        spikes = spikes + brain.step(external_index=idx, external_values=kicks).float()
    return spikes


@needs_graph
@needs_mps
def test_reduced_batch_matches_leading_bodies_of_full_batch(connectome):
    full, small = 64, 24
    brain = Brain(connectome, batch=full, config=LIFConfig(dt_ms=2.0), device=MPS, weight_scale=0.15, use_metal=True)
    idx = torch.tensor(connectome.roles["visual_projection"], dtype=torch.long, device=MPS)
    gen = torch.Generator().manual_seed(0)
    kicks = (torch.rand(idx.numel(), full, generator=gen) < 0.3).float().mul(8.0).to(MPS)

    spikes_full = _run(brain, idx, kicks, 20)
    u_full = brain.u[:, :small].clone()

    brain.reset(batch=small)
    assert brain.batch == small and brain.u.shape == (brain.n, small)
    spikes_small = _run(brain, idx, kicks[:, :small], 20)
    assert brain.batch == small
    assert torch.equal(spikes_small, spikes_full[:small]), "reduced batch must spike exactly like the full batch's first bodies"
    assert float((brain.u - u_full).abs().max()) < 1e-3
    assert float(spikes_small.sum()) > 0

    brain.reset()
    assert brain.batch == full and brain.u.shape == (brain.n, full)


@needs_graph
@needs_mps
def test_reset_batch_bounds(connectome):
    brain = Brain(connectome, batch=32, config=LIFConfig(dt_ms=2.0), device=MPS, weight_scale=0.15, use_metal=True)
    with pytest.raises(ValueError):
        brain.reset(batch=0)
    with pytest.raises(ValueError):
        brain.reset(batch=33)
    brain.reset(batch=32)
    assert brain.batch == 32
