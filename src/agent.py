"""Body/brain interface: lidar -> visual neurons, descending neurons -> controls.

The connectome is never modified. What is learned is only the interface and the
excitability a real fly would get from development and neuromodulation:

    ray_gain    (n_rays,)   how strongly each visual direction drives its
                            visual-projection-neuron group
    loom_gain   (1,)        extra drive from an *expanding* edge (proximity
                            increasing), the stimulus LC/LPLC looming cells
                            respond to most strongly
    bias_hz     (1,)        tonic drive on the visual sheet
    speed_gain  (1,)        proprioceptive speed drive onto ascending neurons
    dn_gain     (n_dn,)     per-descending-neuron excitability multiplier
    w_out       (readout_dim, 2)  readout from projected descending-neuron rates
    b_out       (2,)        readout bias

Sensory input is delivered as Poisson spike kicks (as in Shiu et al. 2024)
rather than constant current: constant current makes the target cells switch
between silent and refractory-limited saturation with nothing in between.

Visual input targets visual projection neurons (LC/LPLC classes) instead of
photoreceptors because photoreceptor output is histaminergic (inhibitory) and
the lamina/medulla stages that invert it are not what this task needs; LC-type
projection neurons are the looming and feature detectors that actually drive
descending steering. Ray 0 is the leftmost ray and feeds the left-eye group.

Readout. Descending-neuron rates are low-pass filtered (`motor_tau`), the
population mean is subtracted (the common mode a random projection would
otherwise turn into a large random offset per channel), and the result goes
through a fixed random projection to `readout_dim` channels. There is no
temporal high-pass: one was tried and it removed the steady-state signal a car
needs to hold a constant-radius corner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brain import Brain


@dataclass(frozen=True)
class AgentConfig:
    n_rays: int = 9
    input_role: str = "visual_projection"
    speed_role: str = "ascending"
    kick_mv: float = 8.0
    max_input_hz: float = 300.0
    # Common random numbers: every population member sees the same Poisson draw
    # so fitness differences come from parameters, not from input noise. Without
    # this the noise between two identical-stimulus runs is as large as the
    # difference between a left-wall and a right-wall stimulus.
    shared_noise: bool = True
    # Low-pass on descending rates: 0.2 at a 16 ms step is an 80 ms time
    # constant, about the visual-motor latency of a fly.
    motor_tau: float = 0.2
    # Projected channels have unit-order magnitude at typical DN rates with
    # this scale (measured: DN rates ~12 Hz, |channel| ~ 3 at scale 50).
    readout_scale: float = 16.0
    common_mode: bool = True
    # Looming: positive change in proximity per control step, scaled so a wall
    # approached at speed gives values of order 0.1-1.
    loom: bool = True
    loom_scale: float = 10.0
    substeps: int = 8
    max_dn: int = 1314
    # The readout runs through a fixed random projection of the descending
    # population. A per-DN readout is 2,628 parameters, which evolution
    # strategies searches badly at a population of 64; 64 mixed channels keep
    # the same information in ~140 parameters and move far faster.
    readout_dim: int = 64
    projection_seed: int = 17
    learn_dn_gain: bool = False


def _azimuth_order(roles: dict[str, list[int]], neurons, role: str) -> np.ndarray:
    """Role indices ordered across the visual field, left eye first.

    Optic-lobe hex coordinates are used when present; the v1.0 annotations
    carry none for the visual projection neurons, so soma side is the reliable
    axis and the within-eye order falls back to body-id order (fixed, if
    arbitrary).
    """
    idx = np.asarray(roles[role], dtype=np.int64)
    side = neurons["somaSide"].to_numpy()[idx]
    hex1 = neurons["assignedOlHex1"].to_numpy()[idx].astype(float)
    if np.isfinite(hex1).any():
        fill = float(np.nanmedian(hex1[np.isfinite(hex1)]))
        hex1 = np.where(np.isfinite(hex1), hex1, fill)
        span = max(1.0, float(hex1.max() - hex1.min()))
        within = (hex1 - hex1.min()) / span
    else:
        within = np.linspace(0.0, 1.0, len(idx))
    is_right = np.isin(side, ["R", "RHS", "right"])
    azimuth = np.where(is_right, 0.5 + 0.5 * within, 0.5 - 0.5 * within)
    return idx[np.argsort(azimuth, kind="stable")]


def make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    """A device-local generator, or None where the backend has none."""
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except (RuntimeError, TypeError):
        try:
            return torch.Generator().manual_seed(seed)
        except RuntimeError:
            return None


class ConnectomeAgent:
    """Drives a batched `Brain` from observations and reads out motor commands."""

    def __init__(self, brain: Brain, neurons, cfg: AgentConfig | None = None) -> None:
        self.brain = brain
        self.cfg = cfg or AgentConfig()
        self.device = brain.device
        self.neurons = neurons

        ordered = _azimuth_order(brain.roles, neurons, self.cfg.input_role)
        self.ray_groups = [
            torch.tensor(g, dtype=torch.long, device=self.device)
            for g in np.array_split(ordered, self.cfg.n_rays)
        ]
        self.speed_index = torch.tensor(
            brain.roles[self.cfg.speed_role], dtype=torch.long, device=self.device
        )

        dn = np.asarray(brain.roles["descending"], dtype=np.int64)[: self.cfg.max_dn]
        self.dn_index = torch.tensor(dn, dtype=torch.long, device=self.device)
        self.dn_bodies = neurons["bodyId"].to_numpy()[dn]
        self.dn_types = neurons["type"].to_numpy()[dn]
        self.n_dn = len(dn)
        self.motor_state = torch.zeros(brain.batch, self.n_dn, device=self.device)
        self.prev_proximity: torch.Tensor | None = None

        # All driven cells in one index, each tagged with the rate channel that
        # feeds it (ray 0..n_rays-1, then speed). One gather + one Bernoulli
        # draw per substep replaces a Python loop of per-group kernels and a
        # full-width zeros(batch, n) buffer.
        self.input_index = torch.cat(self.ray_groups + [self.speed_index])
        self.input_channel = torch.cat(
            [torch.full((g.numel(),), k, dtype=torch.long) for k, g in enumerate(self.ray_groups)]
            + [torch.full((self.speed_index.numel(),), self.cfg.n_rays, dtype=torch.long)]
        ).to(self.device)

        generator = torch.Generator().manual_seed(self.cfg.projection_seed)
        self.projection = (
            torch.randn(self.n_dn, self.cfg.readout_dim, generator=generator)
            / np.sqrt(self.n_dn)
        ).to(self.device)
        self.generator: torch.Generator | None = None
        self.seed(0)

    def seed(self, seed: int) -> None:
        """Fix the sensory Poisson draws; identical seeds give identical episodes."""
        self.generator = make_generator(self.device, seed)

    # --- parameter vector plumbing -------------------------------------------------
    @property
    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {
            "ray_gain": (self.cfg.n_rays,),
            "bias_hz": (1,),
            "speed_gain": (1,),
        }
        if self.cfg.loom:
            shapes["loom_gain"] = (1,)
        if self.cfg.learn_dn_gain:
            shapes["dn_gain"] = (self.n_dn,)
        shapes["w_out"] = (self.cfg.readout_dim, 2)
        shapes["b_out"] = (2,)
        return shapes

    @property
    def n_params(self) -> int:
        return sum(int(np.prod(s)) for s in self.param_shapes.values())

    # Hard bounds per parameter, applied to the ES mean and to every sampled
    # perturbation. Without them a flat fitness landscape lets momentum walk the
    # readout to |w| ~ 70 and the biases to ±30, at which point tanh is pinned
    # and the controller degenerates into bang-bang steering.
    PARAM_BOUNDS: dict[str, tuple[float, float]] = {
        "ray_gain": (0.0, 2.0),
        "bias_hz": (0.0, 0.6),
        "speed_gain": (0.0, 2.0),
        "loom_gain": (0.0, 3.0),
        "dn_gain": (0.2, 3.0),
        "w_out": (-2.5, 2.5),
        "b_out": (-1.5, 2.5),
    }

    def _bounds(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.empty(self.n_params, device=device)
        hi = torch.empty(self.n_params, device=device)
        offset = 0
        for name, shape in self.param_shapes.items():
            size = int(np.prod(shape))
            low, high = self.PARAM_BOUNDS[name]
            lo[offset : offset + size] = low
            hi[offset : offset + size] = high
            offset += size
        return lo, hi

    def clamp_params(self, params: torch.Tensor) -> torch.Tensor:
        """Return `params` (…, n_params) with every block inside PARAM_BOUNDS."""
        lo, hi = self._bounds(params.device)
        return torch.maximum(torch.minimum(params, hi), lo)

    def fraction_at_bounds(self, params: torch.Tensor, tol: float = 1e-4) -> float:
        """Share of parameters pinned at a bound: a saturation warning light."""
        lo, hi = self._bounds(params.device)
        pinned = (params <= lo + tol) | (params >= hi - tol)
        return float(pinned.float().mean())

    def initial_params(self, generator: torch.Generator | None = None) -> torch.Tensor:
        init = {
            "ray_gain": lambda size: torch.full((size,), 0.5),
            "bias_hz": lambda size: torch.full((size,), 0.1),
            "speed_gain": lambda size: torch.full((size,), 0.2),
            "loom_gain": lambda size: torch.zeros(size),
            "dn_gain": lambda size: torch.ones(size),
            # Small: 64 unit-scale channels at 0.5 gave a motor std of ~12 and
            # a tanh pinned from generation 0 (flat landscape, then drift).
            "w_out": lambda size: torch.randn(size, generator=generator) * 0.1,
            # Start rolling: tanh(0.5) is 46% throttle, so the very first
            # generation already produces progress for ES to shape.
            "b_out": lambda size: torch.tensor([0.0, 0.5])[:size],
        }
        return torch.cat(
            [init[name](int(np.prod(shape))) for name, shape in self.param_shapes.items()]
        )

    def unpack(self, params: torch.Tensor) -> dict[str, torch.Tensor]:
        """params: (batch, n_params) -> dict of (batch, ...) tensors."""
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape in self.param_shapes.items():
            size = int(np.prod(shape))
            out[name] = params[:, offset : offset + size].reshape(
                params.shape[0], *shape
            )
            offset += size
        return out

    # --- closed loop ----------------------------------------------------------------
    def reset(self) -> None:
        self.brain.reset()
        self.motor_state = torch.zeros(self.brain.batch, self.n_dn, device=self.device)
        self.prev_proximity = None

    @property
    def dn_rate_hz(self) -> torch.Tensor:
        """Filtered descending-neuron rate per (body, neuron), in spikes/second."""
        return self.motor_state * (1000.0 / self.brain.cfg.dt_ms)

    def sensory_rates(self, obs: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """Poisson rates (batch, n_rays + 1) in Hz for the ray groups and speed."""
        cfg = self.cfg
        lidar = obs[:, : cfg.n_rays]
        speed = obs[:, cfg.n_rays : cfg.n_rays + 1]
        # Near wall -> high rate, mirroring an expanding edge on the retina.
        proximity = (1.0 - lidar).clamp(0.0, 1.0)
        drive = proximity * theta["ray_gain"] + theta["bias_hz"]
        if cfg.loom:
            prev = proximity if self.prev_proximity is None else self.prev_proximity
            loom = ((proximity - prev) * cfg.loom_scale).clamp(min=0.0)
            drive = drive + loom * theta["loom_gain"]
            self.prev_proximity = proximity
        vis_hz = drive.clamp(0.0, 1.0) * cfg.max_input_hz
        speed_hz = (speed * theta["speed_gain"]).clamp(0.0, 1.0) * cfg.max_input_hz
        return torch.cat([vis_hz, speed_hz], dim=1)

    def act(
        self,
        obs: torch.Tensor,
        theta: dict[str, torch.Tensor],
        spike_sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One control step: inject sensation, run `substeps` of LIF, read DNs.

        `spike_sink` (batch, n), if given, accumulates every substep's spikes
        for callers that want to watch the whole network, not just the DNs.
        """
        cfg = self.cfg
        batch = obs.shape[0]
        rates_hz = self.sensory_rates(obs, theta)

        if "dn_gain" in theta:
            gain = torch.ones(batch, self.brain.n, device=self.device)
            gain[:, self.dn_index] = theta["dn_gain"]
            self.brain.set_gain(gain)
        elif self.brain.gain is not None:
            self.brain.set_gain(None)

        # Bernoulli spike kicks (as in Shiu et al. 2024): per driven cell, the
        # probability of a kick this substep from its channel's rate. Laid out
        # (cells, bodies) to match the brain's native state layout.
        prob = (rates_hz[:, self.input_channel] * (self.brain.cfg.dt_ms / 1000.0)).clamp_(
            0.0, 1.0
        ).t()
        rows = 1 if cfg.shared_noise else batch
        draws = torch.rand(
            cfg.substeps, self.input_index.numel(), rows, device=self.device, generator=self.generator
        )

        spikes = torch.zeros(batch, self.n_dn, device=self.device)
        for s in range(cfg.substeps):
            kicks = (draws[s] < prob).to(prob.dtype).mul_(cfg.kick_mv)
            fired = self.brain.step(external_index=self.input_index, external_values=kicks)
            spikes = spikes + fired[:, self.dn_index]
            if spike_sink is not None:
                spike_sink.add_(fired)

        rate = spikes / cfg.substeps
        self.motor_state = (1 - cfg.motor_tau) * self.motor_state + cfg.motor_tau * rate
        signal = self.motor_state * cfg.readout_scale
        if cfg.common_mode:
            signal = signal - signal.mean(dim=1, keepdim=True)
        mixed = signal @ self.projection
        motor = torch.einsum("bd,bdk->bk", mixed, theta["w_out"]) + theta["b_out"]
        steer = torch.tanh(motor[:, 0])
        pedal = torch.tanh(motor[:, 1])
        return torch.stack([steer, pedal], dim=1)
