"""Fused Metal kernel for the LIF step on Apple GPUs (torch.mps.compile_shader).

The torch path in `brain.py` spends its time in `torch.sparse.mm(W, spikes)`:
for every one of the 6.24M synapses it gathers a full row of presynaptic
spikes (batch x 4 bytes) and scatters a row of current, ~1.6 GB of traffic per
step at a population of 64, then makes ten more full-width passes over the
state for the membrane update. This kernel does one pass:

  * spikes live as bit masks, one bit per body, so a synapse reads 4 bytes
    per 32 bodies instead of 128;
  * the synaptic sum, current decay, membrane integration, threshold, reset,
    adaptation and refractory bookkeeping run in registers per neuron;
  * one thread owns one (neuron, 32-body word), so spike bits are assembled
    without atomics.

Semantics match `Brain.step` in brain.py: same update order, same constants.
Summation order over synapses differs, so results agree to float rounding.
"""

from __future__ import annotations

import torch

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// fp: syn_kick, decay_s, decay_v, decay_a, adapt_kick, u_thresh, u_reset, ref_steps
// ip: n, batch, words, read_slot, write_slot, n_short, n_long

inline uint membrane_update(
    device float* u, device float* j, device float* adapt, device float* refrac,
    device const float* ext, int slot, device const float* fp,
    uint i, uint batch, uint b0, uint k, float acc)
{
    const uint idx = i * batch + b0 + k;
    const float jj = (j[idx] + fp[0] * acc) * fp[1];
    j[idx] = jj;
    const float a = adapt[idx];
    const float r = refrac[idx];
    float inflow = jj - fp[4] * a;
    if (slot >= 0) inflow += ext[uint(slot) * batch + b0 + k];
    if (r > 0.0f) inflow = 0.0f;
    float uu = fp[2] * u[idx] + inflow;
    const bool fired = uu >= fp[5];
    if (fired) uu = fp[6];
    u[idx] = uu;
    adapt[idx] = fp[3] * a + (fired ? 1.0f : 0.0f);
    refrac[idx] = fired ? fp[7] : max(r - 1.0f, 0.0f);
    return fired ? (1u << k) : 0u;
}

// One thread per (short row, 32-body word): rows with few synapses.
kernel void lif_short(
    device float* u              [[buffer(0)]],   // (n, batch) membrane, relative to rest
    device float* j              [[buffer(1)]],   // (n, batch) synaptic current, pre-scaled
    device float* adapt          [[buffer(2)]],   // (n, batch) adaptation, spike units
    device float* refrac         [[buffer(3)]],   // (n, batch) refractory steps left
    device uint* bits            [[buffer(4)]],   // (slots, n, words) spike bit masks
    device const int* rowptr     [[buffer(5)]],   // (n + 1) CSR by postsynaptic neuron
    device const int* col        [[buffer(6)]],   // (nnz) presynaptic neuron
    device const float* val      [[buffer(7)]],   // (nnz) signed synapse count
    device const int* ext_slot   [[buffer(8)]],   // (n) row of `ext` driving this cell, -1 none
    device const float* ext      [[buffer(9)]],   // (k, batch) external input, mV per step
    device const float* fp       [[buffer(10)]],
    device const int* ip         [[buffer(11)]],
    device const int* rows       [[buffer(12)]],  // (n_short) row ids handled here
    uint tid [[thread_position_in_grid]])
{
    const uint n = uint(ip[0]);
    const uint batch = uint(ip[1]);
    const uint words = uint(ip[2]);
    const uint r = tid / words;
    const uint w = tid - r * words;
    if (r >= uint(ip[5])) return;
    const uint i = uint(rows[r]);

    float acc[32];
    for (uint k = 0; k < 32; ++k) acc[k] = 0.0f;
    device const uint* rb = bits + uint(ip[3]) * n * words + w;
    const int e1 = rowptr[i + 1];
    for (int e = rowptr[i]; e < e1; ++e) {
        const uint b = rb[uint(col[e]) * words];
        if (b == 0u) continue;
        const float v = val[e];
        for (uint k = 0; k < 32; ++k) acc[k] += (b & (1u << k)) ? v : 0.0f;
    }

    const uint b0 = w * 32u;
    const uint nb = min(32u, batch - b0);
    const int slot = ext_slot[i];
    uint out = 0u;
    for (uint k = 0; k < nb; ++k) {
        out |= membrane_update(u, j, adapt, refrac, ext, slot, fp, i, batch, b0, k, acc[k]);
    }
    bits[(uint(ip[4]) * n + i) * words + w] = out;
}

// One SIMD group (32 lanes) per (long row, word): lanes stride over the
// synapses, reduce, then lane k integrates body k. Hub neurons with
// thousands of inputs no longer serialise the step.
kernel void lif_long(
    device float* u              [[buffer(0)]],
    device float* j              [[buffer(1)]],
    device float* adapt          [[buffer(2)]],
    device float* refrac         [[buffer(3)]],
    device uint* bits            [[buffer(4)]],
    device const int* rowptr     [[buffer(5)]],
    device const int* col        [[buffer(6)]],
    device const float* val      [[buffer(7)]],
    device const int* ext_slot   [[buffer(8)]],
    device const float* ext      [[buffer(9)]],
    device const float* fp       [[buffer(10)]],
    device const int* ip         [[buffer(11)]],
    device const int* rows       [[buffer(12)]],  // (n_long)
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint n = uint(ip[0]);
    const uint batch = uint(ip[1]);
    const uint words = uint(ip[2]);
    const uint group = tid / 32u;
    const uint r = group / words;
    const uint w = group - r * words;
    if (r >= uint(ip[6])) return;
    const uint i = uint(rows[r]);

    float acc[32];
    for (uint k = 0; k < 32; ++k) acc[k] = 0.0f;
    device const uint* rb = bits + uint(ip[3]) * n * words + w;
    const int e1 = rowptr[i + 1];
    for (int e = rowptr[i] + int(lane); e < e1; e += 32) {
        const uint b = rb[uint(col[e]) * words];
        if (b == 0u) continue;
        const float v = val[e];
        for (uint k = 0; k < 32; ++k) acc[k] += (b & (1u << k)) ? v : 0.0f;
    }
    // total input for body k lands in lane k
    float mine = 0.0f;
    for (uint k = 0; k < 32; ++k) {
        const float total = simd_sum(acc[k]);
        if (k == lane) mine = total;
    }

    const uint b0 = w * 32u;
    const uint nb = min(32u, batch - b0);
    uint bit = 0u;
    if (lane < nb) {
        bit = membrane_update(u, j, adapt, refrac, ext, ext_slot[i], fp, i, batch, b0, lane, mine);
    }
    const uint out = simd_or(bit);
    if (lane == 0) bits[(uint(ip[4]) * n + i) * words + w] = out;
}

// bits (n, words) -> dense (batch, n) float for callers that want every spike
kernel void unpack_all(
    device float* out            [[buffer(0)]],   // (batch, n)
    device const uint* bits      [[buffer(1)]],   // (n, words)
    device const int* ip         [[buffer(2)]],   // n, batch, words
    uint tid [[thread_position_in_grid]])
{
    const uint n = uint(ip[0]);
    const uint batch = uint(ip[1]);
    const uint words = uint(ip[2]);
    const uint i = tid / batch;
    const uint b = tid - i * batch;
    if (i >= n) return;
    const uint word = bits[i * words + (b >> 5)];
    out[b * n + i] = (word >> (b & 31u)) & 1u ? 1.0f : 0.0f;
}

// bits (n, words) -> dense (batch, m) float for a subset of m neurons
kernel void unpack_rows(
    device float* out            [[buffer(0)]],   // (batch, m)
    device const uint* bits      [[buffer(1)]],   // (n, words)
    device const int* rows       [[buffer(2)]],   // (m)
    device const int* ip         [[buffer(3)]],   // m, batch, words
    uint tid [[thread_position_in_grid]])
{
    const uint m = uint(ip[0]);
    const uint batch = uint(ip[1]);
    const uint words = uint(ip[2]);
    const uint r = tid / batch;
    const uint b = tid - r * batch;
    if (r >= m) return;
    const uint word = bits[uint(rows[r]) * words + (b >> 5)];
    out[b * m + r] = (word >> (b & 31u)) & 1u ? 1.0f : 0.0f;
}
"""

_LIB = None


def available(device: torch.device) -> bool:
    return device.type == "mps" and hasattr(torch.mps, "compile_shader")


def library():
    """Compile once per process."""
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(SOURCE)
    return _LIB


def csr_by_post(post, pre, weight, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CSR (rowptr, col, val) with rows = postsynaptic neuron, duplicates summed."""
    idx = torch.stack([torch.as_tensor(post, dtype=torch.long), torch.as_tensor(pre, dtype=torch.long)])
    coo = torch.sparse_coo_tensor(idx, torch.as_tensor(weight, dtype=torch.float32), (n, n)).coalesce()
    csr = coo.to_sparse_csr()
    return (
        csr.crow_indices().to(torch.int32).to(device),
        csr.col_indices().to(torch.int32).to(device),
        csr.values().to(torch.float32).to(device),
    )


# Rows with more synapses than this get a whole SIMD group; the connectome's
# in-degree runs from 0 to 6,660 (mean 37), and above ~64 one thread per row
# is latency-bound on its sequential gathers.
LONG_ROW_EDGES = 64


def split_rows(rowptr: torch.Tensor, threshold: int = LONG_ROW_EDGES) -> tuple[torch.Tensor, torch.Tensor]:
    """(short_rows, long_rows) int32 on rowptr's device, by in-degree."""
    degree = rowptr[1:] - rowptr[:-1]
    order = torch.arange(degree.numel(), dtype=torch.int32, device=rowptr.device)
    return order[degree <= threshold].contiguous(), order[degree > threshold].contiguous()
