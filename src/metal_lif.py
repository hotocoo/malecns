"""Fused Metal kernel for the LIF step on Apple GPUs (torch.mps.compile_shader).

The torch path in `brain.py` spends its time in `torch.sparse.mm(W, spikes)`
and in ten full-width passes over the state. This kernel does one pass:

  * spikes live as bit masks, one bit per body, so a synapse reads 4 bytes
    per 32 bodies instead of 128, and a 21 KB per-neuron "any body spiked"
    bitmap lets the synapse loop skip silent presynaptic cells with one
    cache-resident bit test;
  * one thread owns one (neuron, 32-body word) and walks the row's synapses
    once for all 32 bodies; hub neurons with hundreds of inputs get a whole
    SIMD group striding over their synapses instead, and rows are sorted by
    in-degree so the threads of a group finish together;
  * the synaptic sum, current decay, membrane integration, threshold, reset
    and adaptation run in registers;
  * state is stored at the precision the caller picks (`float` or `half`);
    the membrane step is bandwidth-bound, so `half` roughly halves its cost;
  * with a one-step refractory period (the default at 2 ms) the previous
    step's own spike bit *is* the refractory flag, so no refractory array is
    read or written at all;
  * sensory input is optionally drawn inside the kernel: the caller passes a
    per-cell kick probability and a seed, and a counter hash decides which
    cells kick this step, so no random tensors cross the bus per substep;
  * descending-neuron spikes are counted into a small (batch, n_dn) buffer as
    they fire, so the readout costs no extra pass over the network;
  * a neuron-word whose bodies all sit at exactly zero state and receive no
    input this step is skipped without touching its state (exact: zero in,
    zero out, no spike possible).

Semantics match `Brain.step` in brain.py: same update order, same constants.
Summation order over synapses differs, so results agree to float rounding at
`float` precision; `half` state agrees statistically (see tests/test_metal.py).
"""

from __future__ import annotations

import torch

# Buffer layout shared by lif_short and lif_long. `fp` and `ip` are small
# device tensors so one Python call carries every constant.
FP_SYN_KICK, FP_DECAY_S, FP_DECAY_V, FP_DECAY_A, FP_ADAPT_KICK, FP_U_THRESH, FP_U_RESET, FP_REF_STEPS, FP_KICK_MV, FP_N_DN = range(10)
IP_N, IP_BATCH, IP_WORDS, IP_READ, IP_WRITE, IP_ROWS, IP_LAST, IP_EXT_MODE, IP_N_DN, IP_SHARED = range(10)
EXT_NONE, EXT_VALUES, EXT_POISSON = 0, 1, 2

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// fp: syn_kick, decay_s, decay_v, decay_a, adapt_kick, u_thresh, u_reset, ref_steps, kick_mv, n_dn
// ip: n, batch, words, read_slot, write_slot, n_rows, last_slot, ext_mode, n_dn, shared_noise

// Counter hash -> uniform [0, 1). Three 32-bit inputs (seed, cell, body).
inline float hash01(uint a, uint b, uint c) {
    uint h = a * 0x9E3779B1u;
    h ^= (b + 0x7F4A7C15u) * 0x85EBCA77u;
    h ^= (c + 0x165667B1u) * 0xC2B2AE3Du;
    h ^= h >> 15; h *= 0x2C1B3C6Du;
    h ^= h >> 12; h *= 0x297A2D39u;
    h ^= h >> 15;
    return float(h >> 8) * (1.0f / 16777216.0f);
}

// Per-thread constants, loaded once and shared by the 32 bodies of a word.
struct RowCtx {
    uint batch;
    uint words;
    int mode;       // ext_mode: 0 none, 1 values, 2 poisson
    bool shared;    // poisson draws shared across bodies
    int slot;       // row of `ext` driving this neuron, -1 none
    int dn;         // column of `dn_acc` for this neuron, -1 none
    uint last_bits; // previous step's spike word for this (row, word)
    bool one_step_ref;
    float ref_steps;
    uint seed;
};

inline RowCtx row_ctx(device const int* ip, device const float* fp, device const uint* bits,
                      device const int* ext_slot, device const int* dn_slot, uint seed, uint n, uint i, uint w)
{
    RowCtx c;
    c.batch = uint(ip[1]);
    c.words = uint(ip[2]);
    c.mode = ip[7];
    c.shared = ip[9] != 0;
    c.slot = ext_slot[i];
    c.dn = dn_slot[i];
    c.ref_steps = fp[7];
    c.one_step_ref = fp[7] <= 1.0f;
    c.last_bits = c.one_step_ref ? bits[(uint(ip[6]) * n + i) * c.words + w] : 0u;
    c.seed = seed;
    return c;
}

// One body of one neuron: current decay, membrane integration, threshold,
// reset, adaptation, refractory bookkeeping and the DN spike count.
inline uint membrane(
    device STATE_T* u, device STATE_T* j, device STATE_T* adapt, device float* refrac,
    device const float* ext, device float* dn_acc, device const float* fp,
    const thread RowCtx& c, uint i, uint lane, uint b, float acc, thread bool& zero)
{
    const uint idx = i * c.batch + b;
    const float jj = (float(j[idx]) + fp[0] * acc) * fp[1];
    j[idx] = STATE_T(jj);
    const float a = float(adapt[idx]);
    float inflow = jj - fp[4] * a;
    if (c.slot >= 0) {
        const float e = ext[uint(c.slot) * c.batch + b];
        if (c.mode == 1) inflow += e;
        else if (hash01(c.seed, uint(c.slot), c.shared ? 0u : b) < e) inflow += fp[8];
    }
    bool refractory;
    float r = 0.0f;
    if (c.one_step_ref) {
        refractory = ((c.last_bits >> lane) & 1u) != 0u;
    } else {
        r = refrac[idx];
        refractory = r > 0.0f;
    }
    if (refractory) inflow = 0.0f;
    float uu = fp[2] * float(u[idx]) + inflow;
    const bool fired = uu >= fp[5];
    if (fired) uu = fp[6];
    u[idx] = STATE_T(uu);
    const float an = fp[3] * a + (fired ? 1.0f : 0.0f);
    adapt[idx] = STATE_T(an);
    zero = zero && (STATE_T(uu) == STATE_T(0.0f)) && (STATE_T(jj) == STATE_T(0.0f)) && (STATE_T(an) == STATE_T(0.0f));
    if (!c.one_step_ref) refrac[idx] = fired ? c.ref_steps : max(r - 1.0f, 0.0f);
    if (c.dn >= 0 && fired) dn_acc[b * uint(fp[9]) + uint(c.dn)] += 1.0f;
    return fired ? (1u << lane) : 0u;
}

#define LIF_ARGS \
    device STATE_T* u            [[buffer(0)]],  \
    device STATE_T* j            [[buffer(1)]],  \
    device STATE_T* adapt        [[buffer(2)]],  \
    device float* refrac         [[buffer(3)]],  \
    device uint* bits            [[buffer(4)]],  \
    device const int* rowptr     [[buffer(5)]],  \
    device const int* col        [[buffer(6)]],  \
    device const float* val      [[buffer(7)]],  \
    device const int* ext_slot   [[buffer(8)]],  \
    device const float* ext      [[buffer(9)]],  \
    device const float* fp       [[buffer(10)]], \
    device const int* ip         [[buffer(11)]], \
    device const int* rows       [[buffer(12)]], \
    device const int* dn_slot    [[buffer(13)]], \
    device float* dn_acc         [[buffer(14)]], \
    constant int& seed_in        [[buffer(15)]], \
    device const uint* any_r     [[buffer(16)]], \
    device atomic_uint* any_w    [[buffer(17)]], \
    device uchar* quiet          [[buffer(18)]], \
    uint tid [[thread_position_in_grid]],        \
    uint lane [[thread_index_in_simdgroup]]

// One thread per (row, 32-body word): the thread walks the row's synapses
// once and integrates its 32 bodies. Rows are sorted by in-degree so the 32
// threads of a SIMD group finish their loops together. The accumulator loops
// are fully unrolled: a dynamically indexed register array would otherwise
// spill to memory and cost more than the synapses themselves.
kernel void lif_short(LIF_ARGS)
{
    const uint n = uint(ip[0]);
    const uint batch = uint(ip[1]);
    const uint words = uint(ip[2]);
    const uint r = tid / words;
    const uint w = tid - r * words;
    if (r >= uint(ip[5])) return;
    const uint i = uint(rows[r]);

    float acc[32];
#pragma clang loop unroll(full)
    for (uint k = 0; k < 32; ++k) acc[k] = 0.0f;
    device const uint* rb = bits + uint(ip[3]) * n * words + w;
    const int e1 = rowptr[i + 1];
    for (int e = rowptr[i]; e < e1; ++e) {
        const uint pre = uint(col[e]);
        // 21 KB bitmap, cache-resident: most presynaptic cells are silent in
        // every body this step, and they cost one bit test instead of a load.
        if (((any_r[pre >> 5] >> (pre & 31u)) & 1u) == 0u) continue;
        const uint b = rb[pre * words];
        if (b == 0u) continue;
        const float v = val[e];
#pragma clang loop unroll(full)
        for (uint k = 0; k < 32; ++k) acc[k] += (b & (1u << k)) ? v : 0.0f;
    }

    const uint b0 = w * 32u;
    const uint nb = min(32u, batch - b0);
    const RowCtx c = row_ctx(ip, fp, bits, ext_slot, dn_slot, uint(seed_in), n, i, w);
    // A word whose 32 bodies all sit at exactly zero state, with no synaptic
    // or sensory input this step, stays at zero and cannot fire: skip its
    // state traffic entirely (about a quarter of the network under drive).
    bool any_acc = false;
#pragma clang loop unroll(full)
    for (uint k = 0; k < 32; ++k) any_acc = any_acc || (acc[k] != 0.0f);
    const uint qidx = i * words + w;
    if (quiet[qidx] != 0 && !any_acc && c.slot < 0) {
        bits[(uint(ip[4]) * n + i) * words + w] = 0u;
        return;
    }
    uint out = 0u;
    bool zero = true;
#pragma clang loop unroll(full)
    for (uint k = 0; k < 32; ++k) {
        if (k < nb) out |= membrane(u, j, adapt, refrac, ext, dn_acc, fp, c, i, k, b0 + k, acc[k], zero);
    }
    bits[(uint(ip[4]) * n + i) * words + w] = out;
    quiet[qidx] = (zero && c.one_step_ref) ? 1 : 0;
    if (out != 0u) atomic_fetch_or_explicit(&any_w[i >> 5], 1u << (i & 31u), memory_order_relaxed);
}

// One SIMD group per (row, word) for hub neurons: lanes stride over the
// synapses and reduce, then lane k integrates body k. A single thread on a
// 6,000-synapse row would otherwise hold the whole step hostage.
kernel void lif_long(LIF_ARGS)
{
    const uint n = uint(ip[0]);
    const uint batch = uint(ip[1]);
    const uint words = uint(ip[2]);
    const uint group = tid / 32u;
    const uint r = group / words;
    const uint w = group - r * words;
    if (r >= uint(ip[5])) return;
    const uint i = uint(rows[r]);

    float acc[32];
#pragma clang loop unroll(full)
    for (uint k = 0; k < 32; ++k) acc[k] = 0.0f;
    device const uint* rb = bits + uint(ip[3]) * n * words + w;
    const int e1 = rowptr[i + 1];
    for (int e = rowptr[i] + int(lane); e < e1; e += 32) {
        const uint pre = uint(col[e]);
        if (((any_r[pre >> 5] >> (pre & 31u)) & 1u) == 0u) continue;
        const uint b = rb[pre * words];
        if (b == 0u) continue;
        const float v = val[e];
#pragma clang loop unroll(full)
        for (uint k = 0; k < 32; ++k) acc[k] += (b & (1u << k)) ? v : 0.0f;
    }
    float mine = 0.0f;
#pragma clang loop unroll(full)
    for (uint k = 0; k < 32; ++k) {
        const float total = simd_sum(acc[k]);
        if (k == lane) mine = total;
    }

    const uint b0 = w * 32u;
    const RowCtx c = row_ctx(ip, fp, bits, ext_slot, dn_slot, uint(seed_in), n, i, w);
    const uint qidx = i * words + w;
    const bool any_acc = simd_any(mine != 0.0f);
    if (quiet[qidx] != 0 && !any_acc && c.slot < 0) {
        if (lane == 0) bits[(uint(ip[4]) * n + i) * words + w] = 0u;
        return;
    }
    uint bit = 0u;
    bool zero = true;
    if (b0 + lane < batch) bit = membrane(u, j, adapt, refrac, ext, dn_acc, fp, c, i, lane, b0 + lane, mine, zero);
    const uint out = simd_or(bit);
    const bool all_zero = simd_all(zero);
    if (lane == 0) {
        bits[(uint(ip[4]) * n + i) * words + w] = out;
        quiet[qidx] = (all_zero && c.one_step_ref) ? 1 : 0;
        if (out != 0u) atomic_fetch_or_explicit(&any_w[i >> 5], 1u << (i & 31u), memory_order_relaxed);
    }
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

STATE_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
_METAL_TYPES = {"fp32": "float", "fp16": "half"}
_LIBS: dict[str, object] = {}


def available(device: torch.device) -> bool:
    return device.type == "mps" and hasattr(torch.mps, "compile_shader")


def library(precision: str = "fp32"):
    """Compile once per process and precision."""
    if precision not in _METAL_TYPES:
        raise ValueError(f"precision must be one of {sorted(_METAL_TYPES)}, got {precision!r}")
    lib = _LIBS.get(precision)
    if lib is None:
        lib = torch.mps.compile_shader(f"#define STATE_T {_METAL_TYPES[precision]}\n" + SOURCE)
        _LIBS[precision] = lib
    return lib


def hash01(seed: torch.Tensor | int, cell: torch.Tensor, body: torch.Tensor | int) -> torch.Tensor:
    """Torch twin of the kernel's counter hash, for the reference path and tests.

    Works in int64 with explicit 32-bit wrapping so the bit pattern is the
    kernel's on every backend.
    """
    mask = 0xFFFFFFFF
    cell = cell.to(torch.int64)
    seed_t = torch.as_tensor(seed, dtype=torch.int64, device=cell.device)
    body_t = torch.as_tensor(body, dtype=torch.int64, device=cell.device)
    h = (seed_t * 0x9E3779B1) & mask
    h = h ^ (((cell + 0x7F4A7C15) & mask) * 0x85EBCA77 & mask)
    h = h ^ (((body_t + 0x165667B1) & mask) * 0xC2B2AE3D & mask)
    h = h ^ (h >> 15)
    h = (h * 0x2C1B3C6D) & mask
    h = h ^ (h >> 12)
    h = (h * 0x297A2D39) & mask
    h = h ^ (h >> 15)
    return (h >> 8).to(torch.float32) * (1.0 / 16777216.0)


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


# Rows with more synapses than this get a SIMD group striding over their
# synapses; below it one thread per row is cheaper than 32 SIMD reductions.
# The connectome's in-degree runs from 0 to 6,660 (mean 37, 99th pct 279);
# measured at batch 128 the step is fastest with the split here.
LONG_ROW_EDGES = 256


def split_rows(rowptr: torch.Tensor, threshold: int = LONG_ROW_EDGES) -> tuple[torch.Tensor, torch.Tensor]:
    """(short_rows, long_rows) int32 on rowptr's device, each sorted by in-degree.

    Sorting puts rows of similar length in the same SIMD group, so no lane
    idles while a neighbour finishes a longer synapse list.
    """
    degree = rowptr[1:] - rowptr[:-1]
    order = torch.argsort(degree, descending=True, stable=True)
    sorted_degree = degree[order]
    order = order.to(torch.int32)
    return order[sorted_degree <= threshold].contiguous(), order[sorted_degree > threshold].contiguous()
