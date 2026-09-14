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


class Brain:
    """Population-batched LIF network over a fixed connectome.

    The synaptic matrix is a sparse tensor `W` with W[post, pre] = signed
    synapse count, so one spmm per step delivers all spikes.

    On Apple GPUs the step runs as one fused Metal kernel (`metal_lif.py`)
    with spikes packed one bit per body; the torch path below is the reference
    implementation and is used on CPU/CUDA, when a per-neuron `gain` is set,
    or when `MALECNS_NO_METAL` is in the environment.
    """

    def __init__(
        self,
        connectome: Connectome,
        batch: int = 1,
        config: LIFConfig | None = None,
        device: torch.device | None = None,
        weight_scale: float = 1.0,
        use_metal: bool | None = None,
    ) -> None:
        self.cfg = config or LIFConfig()
        self.device = device or pick_device()
        self.n = connectome.n
        self.batch = batch
        self.roles = connectome.roles

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
        if use_metal is None:
            use_metal = metal_lif.available(self.device) and "MALECNS_NO_METAL" not in os.environ
        self.metal = None
        if use_metal:
            self.metal = metal_lif.library()
            self.words = (batch + 31) // 32
            coo_cpu = coo.coalesce()
            self.rowptr, self.col, self.val = metal_lif.csr_by_post(
                coo_cpu.indices()[0], coo_cpu.indices()[1], coo_cpu.values(), self.n, self.device
            )
            self.fparams = torch.tensor(
                [self.syn_kick, self.decay_s, self.decay_v, self.decay_a, self.adapt_kick,
                 self.u_thresh, self.u_reset, float(self.ref_steps)],
                dtype=torch.float32, device=self.device,
            )
            self.short_rows, self.long_rows = metal_lif.split_rows(self.rowptr)
            # one (n, batch, words, read, write, n_short, n_long) tuple per ring position, built once
            self.iparams = [
                torch.tensor(
                    [self.n, batch, self.words, pos, (pos + self.delay_steps) % (self.delay_steps + 1),
                     self.short_rows.numel(), self.long_rows.numel()],
                    dtype=torch.int32, device=self.device,
                )
                for pos in range(self.delay_steps + 1)
            ]
            self._rows_ip_cache: dict[tuple[int, int], torch.Tensor] = {}
            self.no_ext_slot = torch.full((self.n,), -1, dtype=torch.int32, device=self.device)
            self.no_ext = torch.zeros(1, batch, dtype=torch.float32, device=self.device)
            self._slot_cache: tuple[int, torch.Tensor] | None = None
        self.reset()

    @property
    def uses_metal(self) -> bool:
        return self.metal is not None and self.gain is None

    def reset(self) -> None:
        shape = (self.n, self.batch)
        # u = V - V_rest, so rest is zero and the leak is a plain multiply.
        self.u = torch.zeros(shape, device=self.device)
        self.j_syn = torch.zeros(shape, device=self.device)
        self.adapt = torch.zeros(shape, device=self.device)
        self.last_fired = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.refrac = torch.zeros(shape, device=self.device)
        self.last_spikes = torch.zeros(shape, device=self.device)
        self.spike_buffer = [
            torch.zeros(shape, device=self.device) for _ in range(self.delay_steps)
        ]
        self.buf_pos = 0
        if self.metal is not None:
            self.bits = torch.zeros(
                (self.delay_steps + 1, self.n, self.words), dtype=torch.int32, device=self.device
            )
            self.written_slot = self.delay_steps % (self.delay_steps + 1)

    @property
    def v(self) -> torch.Tensor:
        """Membrane potential in mV, shape (batch, n)."""
        return (self.u + self.cfg.v_rest).t()

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
            for k in range(d):
                self.spike_buffer[d - 1 - k] = self._unpack_all((self.written_slot - k) % ring).t().contiguous()
            self.last_spikes = self.spike_buffer[d - 1]
            self.last_fired = self.last_spikes > 0
            self.buf_pos = 0
        else:
            # torch buffer[(buf_pos + k) % d] holds step t+1-d+k; Metal reads slot
            # buf_pos as step t+1-d, so slot k takes that same entry.
            p0 = self.buf_pos
            for k in range(d):
                self.bits[k] = self._pack(self.spike_buffer[(p0 + k) % d])
            self.buf_pos = 0
            self.written_slot = d - 1

    # --- Metal helpers ------------------------------------------------------------------
    def _pack(self, dense: torch.Tensor) -> torch.Tensor:
        """(n, batch) float spikes -> (n, words) int32 bit masks."""
        padded = torch.zeros(self.n, self.words * 32, dtype=torch.int32, device=self.device)
        padded[:, : self.batch] = (dense > 0).to(torch.int32)
        shifts = torch.arange(32, dtype=torch.int32, device=self.device)
        return (padded.view(self.n, self.words, 32) << shifts).sum(dim=-1, dtype=torch.int32)

    def _unpack_all(self, slot: int) -> torch.Tensor:
        out = torch.empty(self.batch, self.n, dtype=torch.float32, device=self.device)
        self.metal.unpack_all(out, self.bits[slot], self.iparams[0], threads=self.n * self.batch)
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

    def _step_metal(
        self,
        external_mv: torch.Tensor | None,
        external_index: torch.Tensor | None,
        external_values: torch.Tensor | None,
    ) -> None:
        if external_mv is not None:
            # dense input: every cell gets a slot
            ext = external_mv.t().contiguous()
            slots = torch.arange(self.n, dtype=torch.int32, device=self.device)
            if external_values is not None:
                ext = ext.clone()
                ext.index_add_(0, external_index, external_values)
        elif external_values is not None:
            ext = external_values if external_values.is_contiguous() else external_values.contiguous()
            slots = self._ext_slots(external_index)
        else:
            ext, slots = self.no_ext, self.no_ext_slot
        params = self.iparams[self.buf_pos]
        common = (self.u, self.j_syn, self.adapt, self.refrac, self.bits,
                  self.rowptr, self.col, self.val, slots, ext, self.fparams, params)
        if self.short_rows.numel():
            self.metal.lif_short(*common, self.short_rows, threads=self.short_rows.numel() * self.words, group_size=256)
        if self.long_rows.numel():
            self.metal.lif_long(*common, self.long_rows, threads=self.long_rows.numel() * self.words * 32, group_size=256)
        self.written_slot = (self.buf_pos + self.delay_steps) % (self.delay_steps + 1)
        self.buf_pos = (self.buf_pos + 1) % (self.delay_steps + 1)

    def spikes_of(self, index: torch.Tensor) -> torch.Tensor:
        """Spikes of the last step for the neurons in `index`, as (batch, len(index)) float."""
        if self.uses_metal:
            rows = index if index.dtype == torch.int32 else index.to(torch.int32)
            out = torch.empty(self.batch, rows.numel(), dtype=torch.float32, device=self.device)
            key = (rows.data_ptr(), rows.numel())
            ip = self._rows_ip_cache.get(key)
            if ip is None:
                ip = torch.tensor([rows.numel(), self.batch, self.words], dtype=torch.int32, device=self.device)
                self._rows_ip_cache[key] = ip
            self.metal.unpack_rows(out, self.bits[self.written_slot], rows, ip, threads=rows.numel() * self.batch)
            return out
        return self.last_spikes[index].t()

    def dense_spikes(self) -> torch.Tensor:
        """Spikes of the last step as a (batch, n) float tensor."""
        if self.uses_metal:
            return self._unpack_all(self.written_slot)
        return self.last_spikes.t()

    def step(
        self,
        external_mv: torch.Tensor | None = None,
        external_index: torch.Tensor | None = None,
        external_values: torch.Tensor | None = None,
        dense: bool = True,
    ) -> torch.Tensor | None:
        """Advance one dt. Returns the binary spike tensor as a (batch, n) view.

        Input current in mV is either dense `external_mv` (batch, n) or, far
        cheaper when only a few thousand sensory cells are driven, the pair
        `external_index` (k,) and `external_values` (k, batch).

        With `dense=False` nothing is returned; read spikes with `spikes_of`
        or `dense_spikes`. On the Metal path that saves a full-width unpack.
        """
        if self.uses_metal:
            self._step_metal(external_mv, external_index, external_values)
            return self.dense_spikes() if dense else None

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
        return spikes.t() if dense else None

    def role_index(self, role: str, limit: int | None = None) -> torch.Tensor:
        idx = self.roles[role]
        if limit is not None:
            idx = idx[:limit]
        return torch.tensor(idx, dtype=torch.long, device=self.device)
