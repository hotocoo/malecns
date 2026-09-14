"""The fused Metal LIF kernel must reproduce the torch reference step.

Both paths share constants and update order; only the summation order over
synapses differs, so spikes are expected to match exactly for many steps and
the state to agree to float rounding. Skipped where there is no MPS device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metal_lif  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
needs_graph = pytest.mark.skipif(not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first")
needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal kernel needs an MPS device")
MPS = torch.device("mps")


@pytest.fixture(scope="module")
def connectome():
    return load_connectome(GRAPH)


def _pair(connectome, batch: int):
    ref = Brain(connectome, batch=batch, config=LIFConfig(dt_ms=2.0), device=MPS, weight_scale=0.15, use_metal=False)
    met = Brain(connectome, batch=batch, config=LIFConfig(dt_ms=2.0), device=MPS, weight_scale=0.15, use_metal=True)
    assert met.uses_metal and not ref.uses_metal
    return ref, met


@needs_graph
@needs_mps
@pytest.mark.parametrize("batch", [1, 40, 64])
def test_metal_step_matches_torch_reference(connectome, batch):
    ref, met = _pair(connectome, batch)
    idx = torch.tensor(connectome.roles["visual_projection"], dtype=torch.long, device=MPS)
    gen = torch.Generator().manual_seed(0)
    mismatched = 0
    total = 0
    for _ in range(30):
        kicks = (torch.rand(idx.numel(), batch, generator=gen) < 0.3).float().mul(8.0).to(MPS)
        a = ref.step(external_index=idx, external_values=kicks)
        b = met.step(external_index=idx, external_values=kicks)
        mismatched += int((a != b).sum())
        total += a.numel()
    # Summation order over synapses differs, so a membrane sitting exactly on
    # the threshold can round either way; one such spike then cascades through
    # the network, so only the rate of mismatch is meaningful. Bit-exact runs
    # (the common case) must also agree on the membrane state.
    assert mismatched / total < 1e-6, f"{mismatched} of {total} spikes differ"
    if mismatched == 0:
        assert float((ref.u - met.u).abs().max()) < 1e-3
    assert float(a.sum()) > 0, "the drive must make the network spike, or the test proves nothing"


@needs_graph
@needs_mps
def test_metal_packed_readouts_agree_with_dense(connectome):
    _, met = _pair(connectome, 8)
    idx = torch.tensor(connectome.roles["visual_projection"], dtype=torch.long, device=MPS)
    kicks = torch.full((idx.numel(), 8), 8.0, device=MPS)
    for _ in range(5):
        met.step(external_index=idx, external_values=kicks, dense=False)
    dense = met.dense_spikes()
    dn = torch.tensor(connectome.roles["descending"], dtype=torch.int32, device=MPS)
    assert torch.equal(met.spikes_of(dn), dense[:, dn.long()])
    assert dense.shape == (8, connectome.n)


@needs_graph
@needs_mps
def test_metal_dense_input_and_silence(connectome):
    ref, met = _pair(connectome, 4)
    a = ref.step()
    b = met.step()
    assert float(a.sum()) == 0 and float(b.sum()) == 0
    ext = (torch.rand(4, connectome.n, generator=torch.Generator().manual_seed(1)) < 0.002).float().mul(9.0).to(MPS)
    for _ in range(3):
        a = ref.step(external_mv=ext)
        b = met.step(external_mv=ext)
    assert torch.equal(a, b)


@needs_graph
@needs_mps
def test_gain_switches_to_torch_path_and_back_without_losing_spikes(connectome):
    _, met = _pair(connectome, 4)
    idx = torch.tensor(connectome.roles["visual_projection"], dtype=torch.long, device=MPS)
    kicks = torch.full((idx.numel(), 4), 8.0, device=MPS)
    met.step(external_index=idx, external_values=kicks, dense=False)
    newest = met.dense_spikes().clone()
    met.set_gain(torch.ones(4, connectome.n, device=MPS))
    assert not met.uses_metal
    assert torch.equal(met.dense_spikes(), newest)
    met.set_gain(None)
    assert met.uses_metal
    assert torch.equal(met.dense_spikes(), newest)


def test_split_rows_by_degree():
    rowptr = torch.tensor([0, 2, 2, 100, 103], dtype=torch.int32)
    short, long = metal_lif.split_rows(rowptr, threshold=64)
    assert short.tolist() == [0, 1, 3]
    assert long.tolist() == [2]
