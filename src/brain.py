"""Batched leaky integrate-and-fire simulation of the MaleCNS connectome.

Torch implementation (CPU / MPS / CUDA) of the model described in Shiu et al.
2024, kept vectorised over a population dimension so many bodies can be driven
by many parameter variants in parallel:

    tau_m dV/dt = -(V - V_rest) + I_syn
    tau_s dI/dt = -I,  spike -> I_post += w_syn * |weight| * sign_pre * gain_pre

Defaults (Shiu et al. 2024, restated in Lin et al. 2025):
    V_rest = V_reset = -52 mV, V_thresh = -45 mV, t_ref = 2.2 ms,
    tau_m = 20 ms, tau_s = 5 ms, w_syn = 0.275 mV, delay = 1.8 ms.

Delay is applied as a one-step ring buffer rounded to the integration step.

Layout. State lives as (n_neurons, batch), the shape the sparse matmul
produces, so no transposes or strided copies happen inside the step. The
public `step()` still returns spikes as (batch, n_neurons): it is a transposed
view, free to make. At 166,700 neurons every full-width pass over the state is
a ~0.4 ms memory sweep on Apple GPUs, so the update is written to touch the
state as few times as possible (10 passes plus the matmul).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import metal_lif


@dataclass(frozen=True)
class LIFConfig:
    v_rest: float = -52.0
    v_reset: float = -52.0
    v_thresh: float = -45.0
    t_ref_ms: float = 2.2
    tau_m_ms: float = 20.0
    tau_s_ms: float = 5.0
    w_syn_mv: float = 0.275
    delay_ms: float = 1.8
    dt_ms: float = 1.0
    # Spike-frequency adaptation. Not part of Shiu et al. 2024, but without it a
    # fully recurrent connectome latches into self-sustained activity and stops
    # tracking its input; fly neurons do adapt, so a mild current is used.
    adapt_mv: float = 0.6
    tau_adapt_ms: float = 120.0


@dataclass(frozen=True)
class Connectome:
    neurons: pd.DataFrame
    roles: dict[str, list[int]]
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray

    @property
    def n(self) -> int:
        return len(self.neurons)


def load_connectome(graph_dir: str | Path) -> Connectome:
    graph_dir = Path(graph_dir)
    neurons = pd.read_parquet(graph_dir / "neurons.parquet")
    roles = json.loads((graph_dir / "roles.json").read_text())
    edges = np.load(graph_dir / "edges.npz")
    return Connectome(
        neurons=neurons,
        roles=roles,
        pre=edges["pre"],
        post=edges["post"],
        weight=edges["weight"],
    )


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued GPU work is done (for timing and clean shutdown)."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def default_precision() -> str:
    """State precision for the Metal path: `MALECNS_PRECISION` or fp16."""
    return os.environ.get("MALECNS_PRECISION", "fp16")


class Brain:
    """Population-batched LIF network over a fixed connectome.

    The synaptic matrix is a sparse tensor `W` with W[post, pre] = signed
    synapse count, so one spmm per step delivers all spikes.

    On Apple GPUs the step runs as one fused Metal kernel (`metal_lif.py`)
    with spikes packed one bit per body; the torch path below is the reference
    implementation and is used on CPU/CUDA, when a per-neuron `gain` is set,
    or when `MALECNS_NO_METAL` is in the environment.

    Sensory input comes either as explicit mV values per driven cell or as a
    kick probability per cell (`poisson_prob`), in which case the step draws
    the kicks itself from a counter hash seeded by the caller: identical seeds
    give identical episodes on both paths.
    """

    def __init__(
        self,
        connectome: Connectome,
        batch: int = 1,
        config: LIFConfig | None = None,
        device: torch.device | None = None,
        weight_scale: float = 1.0,
        use_metal: bool | None = None,
        precision: str | None = None,
    ) -> None:
        self.cfg = config or LIFConfig()
        self.device = device or pick_device()
        self.n = connectome.n
        self.batch = batch
        self.full_batch = batch  # `compact()` shrinks `batch`; `reset()` restores this
        self.roles = connectome.roles
        self.precision = precision or default_precision()
        if self.precision not in metal_lif.STATE_DTYPES:
            raise ValueError(f"precision must be one of {sorted(metal_lif.STATE_DTYPES)}, got {self.precision!r}")

        idx = torch.from_numpy(
            np.stack([connectome.post.astype(np.int64), connectome.pre.astype(np.int64)])
        )
        vals = torch.from_numpy(connectome.weight.astype(np.float32) * weight_scale)
        coo = torch.sparse_coo_tensor(idx, vals, (self.n, self.n)).coalesce()
        # MPS has no compressed-sparse support; its COO spmm is ~15x faster than
        # CPU CSR anyway, so only non-MPS devices get the CSR conversion.
        self.W = (
            coo.to(self.device)
            if self.device.type == "mps"
            else coo.to_sparse_csr().to(self.device)
        )

        dt, cfg = self.cfg.dt_ms, self.cfg
        self.decay_v = float(np.exp(-dt / cfg.tau_m_ms))
        self.decay_s = float(np.exp(-dt / cfg.tau_s_ms))
        self.decay_a = float(np.exp(-dt / cfg.tau_adapt_ms))
        # Membrane integration is dt-scaled, so results do not depend on dt.
        # A single synapse should peak at w_syn_mv of PSP: integrating a current
        # that decays with tau_s through a membrane with tau_m costs a factor
        # tau_s/tau_m, so the injected current is pre-compensated here.
        v_scale = dt / cfg.tau_m_ms
        psp_gain = cfg.tau_m_ms / cfg.tau_s_ms
        # Synaptic current is stored pre-multiplied by v_scale (mV of membrane
        # change per step) and adaptation in spike units, so the membrane update
        # is a single fused sub/add instead of separate scaling passes.
        self.syn_kick = v_scale * cfg.w_syn_mv * psp_gain
        self.adapt_kick = v_scale * cfg.adapt_mv * psp_gain
        self.u_thresh = cfg.v_thresh - cfg.v_rest
        self.u_reset = cfg.v_reset - cfg.v_rest
        self.ref_steps = max(1, int(round(cfg.t_ref_ms / dt)))
        self.delay_steps = max(1, int(round(cfg.delay_ms / dt)))

        self.gain: torch.Tensor | None = None
        self.kick_mv = 0.0
        self.dn_index: torch.Tensor | None = None
        self.dn_acc = torch.zeros(batch, 0, device=self.device)
        if use_metal is None:
            use_metal = metal_lif.available(self.device) and "MALECNS_NO_METAL" not in os.environ
        self.metal = None
        if use_metal:
            self.metal = metal_lif.library(self.precision)
            self.state_dtype = metal_lif.STATE_DTYPES[self.precision]
            self.words = (batch + 31) // 32
            coo_cpu = coo.coalesce()
            self.rowptr, self.col, self.val = metal_lif.csr_by_post(
                coo_cpu.indices()[0], coo_cpu.indices()[1], coo_cpu.values(), self.n, self.device
            )
            self.fparams = torch.tensor(
                [self.syn_kick, self.decay_s, self.decay_v, self.decay_a, self.adapt_kick,
                 self.u_thresh, self.u_reset, float(self.ref_steps), 0.0, 0.0],
                dtype=torch.float32, device=self.device,
            )
            self.short_rows, self.long_rows = metal_lif.split_rows(self.rowptr)
            self._ip_cache: dict[tuple, torch.Tensor] = {}
            self._rows_ip_cache: dict[tuple[int, int], torch.Tensor] = {}
            self._unpack_ip_cache: dict[tuple[int, int], torch.Tensor] = {}
            self.no_ext_slot = torch.full((self.n,), -1, dtype=torch.int32, device=self.device)
            self.no_ext = torch.zeros(1, batch, dtype=torch.float32, device=self.device)
            self.no_dn_slot = torch.full((self.n,), -1, dtype=torch.int32, device=self.device)
            self.dn_slot = self.no_dn_slot
            self._slot_cache: tuple[int, torch.Tensor] | None = None
        else:
            self.state_dtype = torch.float32
        self.reset()

    @property
    def uses_metal(self) -> bool:
        return self.metal is not None and self.gain is None

    def reset(self, batch: int | None = None) -> None:
        """Fresh state for `batch` bodies (default: the full batch this brain was built for)."""
        target = self.full_batch if batch is None else int(batch)
        if not 1 <= target <= self.full_batch:
            raise ValueError(f"reset batch {target} outside 1..{self.full_batch}")
        if self.batch != target:
            self._set_batch(target)
        shape = (self.n, self.batch)
        # u = V - V_rest, so rest is zero and the leak is a plain multiply.
        self.u = torch.zeros(shape, device=self.device, dtype=self.state_dtype)
        self.j_syn = torch.zeros(shape, device=self.device, dtype=self.state_dtype)
        self.adapt = torch.zeros(shape, device=self.device, dtype=self.state_dtype)
        # The refractory counter is only stored when the period spans more than
        # one step; at one step the previous spike itself is the flag.
        self.refrac = torch.zeros(shape if self.ref_steps > 1 else (1, 1), device=self.device)
        if self.uses_metal:
            # Spikes live as bit masks on the Metal path; the dense float
            # history is only built if `set_gain` switches to the torch path.
            # At 3,072 bodies these three arrays alone were 4.6 GB per reset.
            self.last_fired = None
            self.last_spikes = None
            self.spike_buffer = []
        else:
            self.last_fired = torch.zeros(shape, dtype=torch.bool, device=self.device)
            self.last_spikes = torch.zeros(shape, device=self.device)
            self.spike_buffer = [
                torch.zeros(shape, device=self.device) for _ in range(self.delay_steps)
            ]
        self.buf_pos = 0
        self.dn_acc = torch.zeros(self.batch, self.dn_index.numel() if self.dn_index is not None else 0, device=self.device)
        if self.metal is not None:
            self.bits = torch.zeros(
                (self.delay_steps + 1, self.n, self.words), dtype=torch.int32, device=self.device
            )
            # one bit per neuron per ring slot: did any body spike this step
            self.any_bits = torch.zeros((self.delay_steps + 1, (self.n + 31) // 32), dtype=torch.int32, device=self.device)
            # one byte per (neuron, word): every body at exactly zero state
            self.quiet = torch.ones(self.n * self.words, dtype=torch.uint8, device=self.device)
            self.written_slot = self.delay_steps % (self.delay_steps + 1)

    def _set_batch(self, batch: int) -> None:
        """Resize the per-body bookkeeping (word count, input scratch, kernel parameter caches)."""
        self.batch = int(batch)
        if self.metal is not None:
            self.words = (self.batch + 31) // 32
            self.no_ext = torch.zeros(1, self.batch, dtype=torch.float32, device=self.device)
            self._ip_cache.clear()
            self._rows_ip_cache.clear()
            self._unpack_ip_cache.clear()

    def compact(self, keep: torch.Tensor) -> None:
        """Keep only the bodies where `keep` is True, in their current order.

        The kernel's cost is proportional to the number of 32-body words, so
        once most of a population has finished its episode the trainer packs
        the survivors into the leading words and the rest of the episode runs
        on a fraction of the state. Membrane, current and adaptation are
        gathered per body; the spike ring is re-packed bit by bit; the
        "any body spiked" prefilter stays as a superset and the quiet-word
        flags are cleared (the kernel recomputes both on the next step).
        `reset()` restores the full batch. With `shared_noise` the Poisson
        draws do not depend on the body index, so a compacted body sees the
        same input stream as before.
        """
        keep = keep.to(self.device, torch.bool)
        if keep.numel() != self.batch:
            raise ValueError(f"keep has {keep.numel()} entries for {self.batch} bodies")
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        new_batch = int(idx.numel())
        if new_batch == 0:
            raise ValueError("compact() needs at least one surviving body")
        if new_batch == self.batch:
            return
        self.u = self.u[:, idx]
        self.j_syn = self.j_syn[:, idx]
        self.adapt = self.adapt[:, idx]
        if self.refrac.shape == (self.n, self.batch):
            self.refrac = self.refrac[:, idx]
        self.dn_acc = self.dn_acc[idx] if self.dn_acc.shape[1] else torch.zeros(new_batch, 0, device=self.device)
        if self.gain is not None:
            self.gain = self.gain[:, idx]
        if self.uses_metal:
            self.bits = self._gather_bits(self.bits, idx)
            self.quiet = torch.zeros(self.n * ((new_batch + 31) // 32), dtype=torch.uint8, device=self.device)
        else:
            self.last_fired = self.last_fired[:, idx]
            self.last_spikes = self.last_spikes[:, idx]
            self.spike_buffer = [b[:, idx] for b in self.spike_buffer]
        self._set_batch(new_batch)

    def _gather_bits(self, bits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(slots, n, words) bit masks -> the same for the bodies in `idx`, packed into consecutive bits."""
        # Advanced indexing, not index_select: torch 2.13 index_select on int32
        # MPS tensors returns wrong data (see tests/test_compaction.py).
        slots = bits.shape[0]
        new_words = (idx.numel() + 31) // 32
        out = torch.zeros((slots, self.n, new_words), dtype=torch.int32, device=self.device)
        for w in range(new_words):
            sel = idx[w * 32 : (w + 1) * 32]
            src = bits[:, :, torch.div(sel, 32, rounding_mode="floor")]
            bit = (src >> (sel % 32).to(torch.int32)) & 1
            lanes = torch.arange(sel.numel(), dtype=torch.int32, device=self.device)
            out[:, :, w] = (bit << lanes).sum(dim=-1, dtype=torch.int32)
        return out

    @property
    def v(self) -> torch.Tensor:
        """Membrane potential in mV, shape (batch, n)."""
        return (self.u.float() + self.cfg.v_rest).t()

    def set_dn_index(self, index: torch.Tensor | None) -> None:
        """Neurons whose spikes are counted into `dn_acc` (batch, len(index)) every step."""
        if index is None:
            self.dn_index = None
            self.dn_acc = torch.zeros(self.batch, 0, device=self.device)
            if self.metal is not None:
                self._ip_cache.clear()
                self._rows_ip_cache.clear()
                self.fparams[metal_lif.FP_N_DN] = 0.0
                self.dn_slot = self.no_dn_slot
            return
        self.dn_index = index.to(self.device).to(torch.long)
        self.dn_acc = torch.zeros(self.batch, self.dn_index.numel(), device=self.device)
        if self.metal is not None:
            self._ip_cache.clear()
            self._rows_ip_cache.clear()
            self.fparams[metal_lif.FP_N_DN] = float(self.dn_index.numel())
            slots = torch.full((self.n,), -1, dtype=torch.int32, device=self.device)
            slots[self.dn_index] = torch.arange(self.dn_index.numel(), dtype=torch.int32, device=self.device)
            self.dn_slot = slots

    def set_gain(self, gain: torch.Tensor | None) -> None:
        """Per-neuron presynaptic excitability multiplier, shape (batch, n).

        `None` removes the multiplier and skips its memory pass entirely. A
        gain forces the torch path; the Metal kernel reads spikes as bits and
        has no per-synapse multiplier. Switching paths mid-run carries the
        state across (spike history is rebuilt densely / re-packed).
        """
        was_metal = self.uses_metal
        self.gain = None if gain is None else gain.to(self.device).t().contiguous()
        if self.metal is None or was_metal == self.uses_metal:
            return
        d, ring = self.delay_steps, self.delay_steps + 1
        if was_metal:
            # Metal slots (written_slot - k) hold steps t-k. The torch path with
            # buf_pos 0 reads buffer[0] as the oldest (step t+1-d) and buffer[d-1]
            # as the newest (step t).
            self.spike_buffer = [None] * d
            for k in range(d):
                self.spike_buffer[d - 1 - k] = self._unpack_all((self.written_slot - k) % ring).t().contiguous()
            self.last_spikes = self.spike_buffer[d - 1]
            self.last_fired = self.last_spikes > 0
            self.buf_pos = 0
            self.u, self.j_syn, self.adapt = self.u.float(), self.j_syn.float(), self.adapt.float()
        else:
            # torch buffer[(buf_pos + k) % d] holds step t+1-d+k; Metal reads slot
            # buf_pos as step t+1-d, so slot k takes that same entry.
            p0 = self.buf_pos
            for k in range(d):
                self.bits[k] = self._pack(self.spike_buffer[(p0 + k) % d])
                self.any_bits[k] = self._pack_any(self.bits[k])
            self.buf_pos = 0
            self.written_slot = d - 1
            self.u, self.j_syn, self.adapt = (t.to(self.state_dtype) for t in (self.u, self.j_syn, self.adapt))
            self.quiet.zero_()

    # --- Metal helpers ------------------------------------------------------------------
    def _pack(self, dense: torch.Tensor) -> torch.Tensor:
        """(n, batch) float spikes -> (n, words) int32 bit masks."""
        padded = torch.zeros(self.n, self.words * 32, dtype=torch.int32, device=self.device)
        padded[:, : self.batch] = (dense > 0).to(torch.int32)
        shifts = torch.arange(32, dtype=torch.int32, device=self.device)
        return (padded.view(self.n, self.words, 32) << shifts).sum(dim=-1, dtype=torch.int32)

    def _pack_any(self, packed: torch.Tensor) -> torch.Tensor:
        """(n, words) bit masks -> (ceil(n/32),) int32: bit i set when neuron i spiked in any body."""
        active = (packed != 0).any(dim=1).to(torch.int32)
        padded = torch.zeros(self.any_bits.shape[1] * 32, dtype=torch.int32, device=self.device)
        padded[: self.n] = active
        shifts = torch.arange(32, dtype=torch.int32, device=self.device)
        return (padded.view(-1, 32) << shifts).sum(dim=-1, dtype=torch.int32)

    def _unpack_all(self, slot: int) -> torch.Tensor:
        out = torch.empty(self.batch, self.n, dtype=torch.float32, device=self.device)
        self.metal.unpack_all(out, self.bits[slot], self._ip(0, metal_lif.EXT_NONE, False), threads=self.n * self.batch)
        return out

    def _ext_slots(self, external_index: torch.Tensor | None) -> torch.Tensor:
        if external_index is None:
            return self.no_ext_slot
        key = (external_index.data_ptr(), external_index.numel())
        if self._slot_cache is None or self._slot_cache[0] != key:
            slots = torch.full((self.n,), -1, dtype=torch.int32, device=self.device)
            slots[external_index.to(self.device)] = torch.arange(
                external_index.numel(), dtype=torch.int32, device=self.device
            )
            self._slot_cache = (key, slots)
        return self._slot_cache[1]

    def _ip(self, pos: int, ext_mode: int, shared: bool) -> torch.Tensor:
        """Integer parameter block for ring position `pos` (built once per combination)."""
        key = (pos, ext_mode, shared)
        ip = self._ip_cache.get(key)
        if ip is None:
            ring = self.delay_steps + 1
            write = (pos + self.delay_steps) % ring
            ip = torch.tensor(
                [self.n, self.batch, self.words, pos, write, 0, (write + ring - 1) % ring, ext_mode,
                 self.dn_acc.shape[1], int(shared)],
                dtype=torch.int32, device=self.device,
            )
            self._ip_cache[key] = ip
        return ip

    def _rows_ip(self, ip: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        key = (ip.data_ptr(), rows.numel())
        out = self._rows_ip_cache.get(key)
        if out is None:
            out = ip.clone()
            out[metal_lif.IP_ROWS] = rows.numel()
            self._rows_ip_cache[key] = out
        return out

    def _step_metal(
        self,
        external_mv: torch.Tensor | None,
        external_index: torch.Tensor | None,
        external_values: torch.Tensor | None,
        poisson_prob: torch.Tensor | None,
        seed: int,
        shared: bool,
    ) -> None:
        mode = metal_lif.EXT_VALUES
        if external_mv is not None:
            # dense input: every cell gets a slot
            ext = external_mv.t().contiguous()
            slots = torch.arange(self.n, dtype=torch.int32, device=self.device)
            if external_values is not None:
                ext = ext.clone()
                ext.index_add_(0, external_index, external_values)
        elif poisson_prob is not None:
            ext = poisson_prob if poisson_prob.is_contiguous() else poisson_prob.contiguous()
            slots = self._ext_slots(external_index)
            mode = metal_lif.EXT_POISSON
        elif external_values is not None:
            ext = external_values if external_values.is_contiguous() else external_values.contiguous()
            slots = self._ext_slots(external_index)
        else:
            ext, slots, mode = self.no_ext, self.no_ext_slot, metal_lif.EXT_NONE
        ip = self._ip(self.buf_pos, mode, shared)
        write_slot = (self.buf_pos + self.delay_steps) % (self.delay_steps + 1)
        any_w = self.any_bits[write_slot]
        any_w.zero_()
        common = (self.u, self.j_syn, self.adapt, self.refrac, self.bits,
                  self.rowptr, self.col, self.val, slots, ext, self.fparams)
        tail = (self.dn_slot, self.dn_acc, int(seed) & 0x7FFFFFFF, self.any_bits[self.buf_pos], any_w, self.quiet)
        if self.short_rows.numel():
            self.metal.lif_short(*common, self._rows_ip(ip, self.short_rows), self.short_rows, *tail,
                                 threads=self.short_rows.numel() * self.words, group_size=256)
        if self.long_rows.numel():
            self.metal.lif_long(*common, self._rows_ip(ip, self.long_rows), self.long_rows, *tail,
                                threads=self.long_rows.numel() * self.words * 32, group_size=256)
        self.written_slot = (self.buf_pos + self.delay_steps) % (self.delay_steps + 1)
        self.buf_pos = (self.buf_pos + 1) % (self.delay_steps + 1)

    def spikes_of(self, index: torch.Tensor) -> torch.Tensor:
        """Spikes of the last step for the neurons in `index`, as (batch, len(index)) float."""
        if self.uses_metal:
            rows = index if index.dtype == torch.int32 else index.to(torch.int32)
            out = torch.empty(self.batch, rows.numel(), dtype=torch.float32, device=self.device)
            key = (rows.data_ptr(), rows.numel())
            ip = self._unpack_ip_cache.get(key)
            if ip is None:
                ip = torch.tensor([rows.numel(), self.batch, self.words], dtype=torch.int32, device=self.device)
                self._unpack_ip_cache[key] = ip
            self.metal.unpack_rows(out, self.bits[self.written_slot], rows, ip, threads=rows.numel() * self.batch)
            return out
        return self.last_spikes[index].t()

    def dense_spikes(self) -> torch.Tensor:
        """Spikes of the last step as a (batch, n) float tensor."""
        if self.uses_metal:
            return self._unpack_all(self.written_slot)
        return self.last_spikes.t()

    def set_kick_mv(self, kick_mv: float) -> None:
        """Size of one sensory kick for the `poisson_prob` input mode."""
        self.kick_mv = float(kick_mv)
        if self.metal is not None:
            self.fparams[metal_lif.FP_KICK_MV] = self.kick_mv

    def step(
        self,
        external_mv: torch.Tensor | None = None,
        external_index: torch.Tensor | None = None,
        external_values: torch.Tensor | None = None,
        dense: bool = True,
        poisson_prob: torch.Tensor | None = None,
        seed: int = 0,
        shared_noise: bool = True,
    ) -> torch.Tensor | None:
        """Advance one dt. Returns the binary spike tensor as a (batch, n) view.

        Input current in mV is either dense `external_mv` (batch, n) or, far
        cheaper when only a few thousand sensory cells are driven, the pair
        `external_index` (k,) and `external_values` (k, batch). With
        `poisson_prob` (k, batch) instead of values, each driven cell receives
        a `kick_mv` kick with that probability, decided by a counter hash of
        (`seed`, cell, body) — or (`seed`, cell) for every body when
        `shared_noise` — so the same seed reproduces the same kicks.

        With `dense=False` nothing is returned; read spikes with `spikes_of`
        or `dense_spikes`. On the Metal path that saves a full-width unpack.
        """
        if self.uses_metal:
            self._step_metal(external_mv, external_index, external_values, poisson_prob, seed, shared_noise)
            return self.dense_spikes() if dense else None

        if poisson_prob is not None:
            cells = torch.arange(external_index.numel(), device=self.device)
            body = 0 if shared_noise else torch.arange(self.batch, device=self.device).unsqueeze(0)
            draw = metal_lif.hash01(seed, cells.unsqueeze(1), body)
            external_values = (draw < poisson_prob).to(poisson_prob.dtype) * self.kick_mv

        delayed = self.spike_buffer[self.buf_pos % self.delay_steps]
        pre = delayed if self.gain is None else delayed * self.gain
        drive = torch.sparse.mm(self.W, pre)
        # j = decay_s * (j + kick * drive)
        self.j_syn.add_(drive, alpha=self.syn_kick).mul_(self.decay_s)

        inflow = torch.sub(self.j_syn, self.adapt, alpha=self.adapt_kick)
        if external_mv is not None:
            inflow.add_(external_mv.t())
        if external_index is not None:
            inflow.index_add_(0, external_index, external_values)
        if self.ref_steps == 1:
            refractory = self.last_fired
        else:
            refractory = self.refrac > 0
        inflow.masked_fill_(refractory, 0.0)

        # u = decay_v * u + inflow, in one pass
        self.u = torch.add(inflow, self.u, alpha=self.decay_v)

        fired = self.u >= self.u_thresh
        spikes = fired.to(self.u.dtype)
        self.u.masked_fill_(fired, self.u_reset)
        # adapt = decay_a * adapt + spikes, in one pass
        self.adapt = torch.add(spikes, self.adapt, alpha=self.decay_a)
        if self.ref_steps == 1:
            self.last_fired = fired
        else:
            # a cell that fires is silent for `ref_steps` steps, then free
            self.refrac = torch.where(fired, torch.full_like(self.refrac, float(self.ref_steps)), (self.refrac - 1.0).clamp_(min=0.0))

        self.spike_buffer[self.buf_pos % self.delay_steps] = spikes
        self.last_spikes = spikes
        self.buf_pos = (self.buf_pos + 1) % self.delay_steps
        if self.dn_index is not None and self.dn_acc.shape[1]:
            self.dn_acc.add_(spikes[self.dn_index].t())
        return spikes.t() if dense else None

    def role_index(self, role: str, limit: int | None = None) -> torch.Tensor:
        idx = self.roles[role]
        if limit is not None:
            idx = idx[:limit]
        return torch.tensor(idx, dtype=torch.long, device=self.device)
