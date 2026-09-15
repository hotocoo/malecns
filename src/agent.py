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
    w_out       (readout_dim, 2)  readout *direction* over the projected
                            output-population channels (normalised in use)
    g_out       (2,)        readout gain: motor pre-activation scale
    b_out       (2,)        readout bias

Sensory input is delivered as Poisson spike kicks (as in Shiu et al. 2024)
rather than constant current: constant current makes the target cells switch
between silent and refractory-limited saturation with nothing in between.

Visual input targets visual projection neurons (LC/LPLC classes) instead of
photoreceptors because photoreceptor output is histaminergic (inhibitory) and
the lamina/medulla stages that invert it are not what this task needs; LC-type
projection neurons are the looming and feature detectors that actually drive
descending steering. Ray 0 is the leftmost ray and feeds the left-eye group.

Readout. The output population is `readout_roles`: by default the descending
neurons (the brain's only path to the body) *and* the VNC motor neurons they
drive, so the whole brain -> descending -> ventral cord -> motor path is read.
Their rates are low-pass filtered (`motor_tau`), the
population mean is subtracted (the common mode a random projection would
otherwise turn into a large random offset per channel), and the result goes
through a fixed random projection to `readout_dim` channels. The channel
vector is then normalised to unit length (`readout_norm="layer"`), and `w_out`
is used as a unit direction scaled by the bounded gain `g_out`: the motor
pre-activation is `g_out * cos(angle between pattern and readout)`
plus bias and lies within +-`g_out`, so neither the firing level nor a
drifting readout norm can pin `tanh` (which is what turned steering into
bang-bang and flattened the fitness landscape before).
There is no temporal high-pass: one was tried and it removed the steady-state
signal a car needs to hold a constant-radius corner.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

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
    # "layer": the projected channel vector is scaled to unit length (with a
    # floor of `readout_floor_hz` of population activity below which the
    # scaling relaxes to linear, so silence is not amplified into a full-scale
    # command); the motor pre-activation is then g_out * cos(pattern, w_out)
    # and lies in [-g_out, g_out]. "none": raw projected rates.
    # "channel": every projected channel is standardised with a calibrated
    # mean and std (`set_readout`, stored in the checkpoint), which is what a
    # readout fitted by `calibrate.py` needs to reproduce its fit exactly.
    readout_norm: str = "layer"
    readout_floor_hz: float = 5.0
    readout_eps: float = 1e-6
    common_mode: bool = True
    # Eye encoding. "road": every ray is read relative to the distance that
    # ray would see from the middle of a straight road of half-width
    # `eye_halfwidth_m` (capped at `eye_front_ref_m` for the forward rays), on
    # a log scale of `eye_octave_gain` per octave: a wall at half the expected
    # distance saturates the group, one at twice the expected distance
    # silences it. This is where the steering information lives. The old
    # "linear" encoding (1 - d / range) put a 4 m and a 12 m wall 0.05 apart
    # on a 0..1 scale and the descending neurons could not tell left from
    # right at all (population d' 0.6 for a 4 m / 7 m offset; 33 with "road").
    eye_encoding: str = "road"
    eye_range_m: float = 150.0
    eye_fov_deg: float = 180.0
    eye_halfwidth_m: float = 5.5
    eye_front_ref_m: float = 60.0
    eye_octave_gain: float = 0.4
    # Looming: positive change in proximity per control step, scaled so a wall
    # approached at speed gives values of order 0.1-1.
    loom: bool = True
    loom_scale: float = 10.0
    substeps: int = 8
    max_dn: int = 1314
    # Populations whose rates feed the motor readout, comma-separated roles.
    readout_roles: str = "descending,motor"
    # The readout runs through a fixed random projection of the descending
    # population. A per-DN readout is 2,628 parameters, which evolution
    # strategies searches badly at a population of 64; 64 mixed channels keep
    # the same information in ~140 parameters and move far faster.
    readout_dim: int = 64
    projection_seed: int = 17
    learn_dn_gain: bool = False

    @classmethod
    def from_saved(cls, saved: dict | None, **overrides) -> "AgentConfig":
        """Config from a checkpoint's `agent_cfg`, ignoring fields this version no longer has."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (saved or {}).items() if k in known}
        kwargs.update(overrides)
        return cls(**kwargs)


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
        fov = float(np.deg2rad(self.cfg.eye_fov_deg))
        ray_angles = torch.linspace(fov / 2, -fov / 2, self.cfg.n_rays)
        self.ray_ref_m = torch.minimum(
            self.cfg.eye_halfwidth_m / ray_angles.sin().abs().clamp(min=1e-3),
            torch.full_like(ray_angles, self.cfg.eye_front_ref_m),
        ).to(self.device)

        dn = np.asarray(brain.roles["descending"], dtype=np.int64)[: self.cfg.max_dn]
        self.dn_index = torch.tensor(dn, dtype=torch.long, device=self.device)
        self.dn_index32 = self.dn_index.to(torch.int32)
        self.dn_bodies = neurons["bodyId"].to_numpy()[dn]
        self.dn_types = neurons["type"].to_numpy()[dn]
        self.n_dn = len(dn)

        # Output population: every role in `readout_roles`, in order, deduplicated.
        readout: list[int] = []
        seen: set[int] = set()
        self.readout_role_of: list[str] = []
        for role in [r.strip() for r in self.cfg.readout_roles.split(",") if r.strip()]:
            cells = dn if role == "descending" else np.asarray(brain.roles.get(role, []), dtype=np.int64)
            for c in cells.tolist():
                if c not in seen:
                    seen.add(c)
                    readout.append(c)
                    self.readout_role_of.append(role)
        if not readout:
            raise ValueError(f"readout_roles {self.cfg.readout_roles!r} selects no neurons")
        self.readout_index = torch.tensor(readout, dtype=torch.long, device=self.device)
        self.n_readout = len(readout)
        self.readout_bodies = neurons["bodyId"].to_numpy()[readout]
        self.readout_types = neurons["type"].to_numpy()[readout]
        position = {c: k for k, c in enumerate(readout)}
        self.dn_in_readout = torch.tensor([position.get(int(c), -1) for c in dn], dtype=torch.long, device=self.device)
        self.motor_state = torch.zeros(brain.batch, self.n_readout, device=self.device)
        self.prev_proximity: torch.Tensor | None = None
        self.last_proximity = torch.zeros(brain.batch, self.cfg.n_rays, device=self.device)
        self.last_loom = torch.zeros(brain.batch, self.cfg.n_rays, device=self.device)
        self.last_motor = torch.zeros(brain.batch, 2, device=self.device)
        brain.set_dn_index(self.readout_index)
        brain.set_kick_mv(self.cfg.kick_mv)

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
            torch.randn(self.n_readout, self.cfg.readout_dim, generator=generator)
            / np.sqrt(self.n_readout)
        ).to(self.device)
        # Channel statistics for readout_norm="channel"; identity until calibrated.
        self.readout_mean = torch.zeros(self.cfg.readout_dim, device=self.device)
        self.readout_std = torch.ones(self.cfg.readout_dim, device=self.device)
        self.readout_calibrated = False
        self.noise_seed = 0
        self.step_counter = 0
        self.seed(0)

    def seed(self, seed: int) -> None:
        """Fix the sensory Poisson draws; identical seeds give identical episodes.

        Kicks are drawn inside the brain step from a counter hash of (seed,
        substep, cell[, body]), so the sequence is reproducible on every
        backend and costs no random tensors on the bus.
        """
        self.noise_seed = int(seed)
        self.step_counter = 0

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
        shapes["g_out"] = (2,)
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
        # |motor| <= g_out (reached only by a pattern aligned with w_out);
        # a random pattern gives ~g_out / sqrt(readout_dim).
        # Lower bound 0.1: the readout calibrated by imitation lands at ~0.35
        # (`calibrate.py`); a 0.5 floor made the brain steer 1.5x too hard.
        "g_out": (0.1, 8.0),
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
            "g_out": lambda size: torch.full((size,), 1.5),
            # Start with modest throttle: tanh(0.3) is 29%, enough to move
            # but leaves room to brake. Steering bias at 0.
            "b_out": lambda size: torch.tensor([0.0, 0.3])[:size],
        }
        return torch.cat(
            [init[name](int(np.prod(shape))) for name, shape in self.param_shapes.items()]
        )

    # Layout of checkpoints written before `param_shapes` was saved: the same
    # blocks in the same order, without the looming channel.
    LEGACY_SHAPES: dict[int, dict[str, tuple[int, ...]]] = {
        141: {"ray_gain": (9,), "bias_hz": (1,), "speed_gain": (1,), "w_out": (64, 2), "b_out": (2,)},
        142: {"ray_gain": (9,), "bias_hz": (1,), "speed_gain": (1,), "loom_gain": (1,), "w_out": (64, 2), "b_out": (2,)},
    }

    def migrate_params(
        self, params: torch.Tensor, shapes: dict[str, list[int] | tuple[int, ...]] | None
    ) -> tuple[torch.Tensor, list[str]]:
        """Fit a saved parameter vector to this agent's layout, block by block.

        Blocks present in both with the same shape are copied; blocks this
        agent has and the checkpoint lacks start from `initial_params`; blocks
        the checkpoint has and this agent lacks are dropped. Returns the new
        vector and a note per block that was not a straight copy. `shapes` is
        the checkpoint's `param_shapes`; when it is missing the legacy layout
        for that parameter count is assumed.
        """
        flat = params.detach().reshape(-1).cpu()
        if shapes is None:
            if flat.numel() == self.n_params:
                return flat.clone(), []
            shapes = self.LEGACY_SHAPES.get(flat.numel())
            if shapes is None:
                raise ValueError(f"checkpoint has {flat.numel()} params, agent {self.n_params}, and no layout to migrate from")
        shapes = {k: tuple(v) for k, v in shapes.items()}
        if sum(int(np.prod(s)) for s in shapes.values()) != flat.numel():
            raise ValueError(f"checkpoint layout {shapes} does not cover its {flat.numel()} params")
        saved: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape in shapes.items():
            size = int(np.prod(shape))
            saved[name] = flat[offset : offset + size].reshape(shape)
            offset += size
        init = self.unpack(self.initial_params().unsqueeze(0))
        notes: list[str] = []
        blocks: list[torch.Tensor] = []
        for name, shape in self.param_shapes.items():
            if name in saved and saved[name].shape == shape:
                blocks.append(saved[name].reshape(-1))
            else:
                blocks.append(init[name][0].reshape(-1).cpu())
                notes.append(f"{name} from init" + (f" (saved shape {tuple(saved[name].shape)})" if name in saved else ""))
        for name in saved:
            if name not in self.param_shapes:
                notes.append(f"{name} dropped")
        return torch.cat(blocks), notes

    # Blocks whose meaning depends on the readout configuration: a w_out trained
    # at another `readout_scale` / `readout_dim` / projection seed is a random
    # matrix here, and one from before those were recorded pinned tanh hard.
    READOUT_BLOCKS = ("w_out", "g_out", "b_out")

    def readout_matches(self, state: dict) -> bool:
        """False only when the checkpoint *records* a different readout configuration.

        A checkpoint without `agent_cfg` is trusted as-is: resetting a trained
        readout because its metadata is missing threw away a working policy
        once (generation 112 of the first Monaco run).
        """
        saved = state.get("agent_cfg")
        if not saved:
            return True
        if bool(state.get("readout")) != self.readout_calibrated:
            return False
        return all(
            saved.get(key) == getattr(self.cfg, key)
            for key in ("readout_norm", "readout_roles", "readout_dim", "projection_seed", "common_mode", "motor_tau")
        )

    def _offsets(self, state: dict, reset: tuple[str, ...]) -> list[tuple[int, int]]:
        """(offset, size) of every block *not* in `reset`, in the saved vector's own layout."""
        flat = state["mu"].reshape(-1)
        shapes = state.get("param_shapes")
        if shapes is None:
            shapes = self.LEGACY_SHAPES.get(flat.numel()) if flat.numel() != self.n_params else self.param_shapes
        out = []
        offset = 0
        for name, shape in shapes.items():
            size = int(np.prod(shape))
            if name not in reset:
                out.append((offset, size))
            offset += size
        return out

    def migrate_state(self, state: dict, reset: tuple[str, ...] = ()) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """(mu, momentum, notes) from a checkpoint dict for this agent's layout.

        `reset` names blocks to take from `initial_params` regardless of what
        the checkpoint holds. Readout blocks are reset automatically when the
        checkpoint's readout configuration differs from this agent's (or was
        never recorded): those weights only mean something at the scale they
        were evolved at.
        """
        # ES checkpoints store one parameter vector per island as (islands,
        # n_params).  The block layout, however, describes one vector.  The
        # old implementation flattened the whole population before validating
        # the layout, so a 4-island checkpoint with 144 parameters/island was
        # interpreted as a 576-parameter single genome and rejected.  Migrate
        # each island independently and retain the population dimension.
        saved_mu = state["mu"]
        if saved_mu.ndim > 1:
            saved_momentum = state.get("momentum")
            mus: list[torch.Tensor] = []
            momenta: list[torch.Tensor] = []
            notes: list[str] = []
            for island in range(saved_mu.shape[0]):
                island_state = dict(state)
                island_state["mu"] = saved_mu[island]
                if saved_momentum is not None and saved_momentum.ndim > 1:
                    island_state["momentum"] = saved_momentum[island]
                mu_i, momentum_i, notes_i = self.migrate_state(island_state, reset=reset)
                mus.append(mu_i)
                momenta.append(momentum_i)
                notes.extend(f"island {island}: {n}" for n in notes_i)
            return torch.stack(mus), torch.stack(momenta), notes

        shapes = state.get("param_shapes")
        reset = tuple(reset)
        if not self.readout_matches(state):
            reset = reset + tuple(b for b in self.READOUT_BLOCKS if b not in reset)
        if reset and shapes is None:
            flat = state["mu"].reshape(-1)
            shapes = self.LEGACY_SHAPES.get(flat.numel()) if flat.numel() != self.n_params else dict(self.param_shapes)
            if shapes is None:
                raise ValueError(f"checkpoint has {flat.numel()} params, agent {self.n_params}, and no layout to migrate from")
        if reset:
            shapes = {k: v for k, v in shapes.items() if k not in reset}
            keep = torch.cat(
                [state["mu"].reshape(-1).cpu()[o : o + n] for o, n in self._offsets(state, reset)]
            ) if shapes else torch.zeros(0)
            mu, notes = self.migrate_params(keep, shapes)
            notes = [
                next((f"{b} (reset)" for b in reset if n.startswith(b + " from init")), n) for n in notes
            ]
            momentum = torch.zeros_like(mu)
            return mu, momentum, notes
        mu, notes = self.migrate_params(state["mu"], shapes)
        if "momentum" in state and state["momentum"].numel() == state["mu"].numel():
            momentum, _ = self.migrate_params(state["momentum"], shapes)
            # blocks that came from init carry no momentum
            if notes:
                offset = 0
                for name, shape in self.param_shapes.items():
                    size = int(np.prod(shape))
                    if any(n.startswith(name + " from init") for n in notes):
                        momentum[offset : offset + size] = 0.0
                    offset += size
        else:
            momentum = torch.zeros_like(mu)
        return mu, momentum, notes

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
        self.motor_state = torch.zeros(self.brain.batch, self.n_readout, device=self.device)
        self.prev_proximity = None
        self.last_motor = torch.zeros(self.brain.batch, 2, device=self.device)
        self.step_counter = 0

    def compact(self, keep: torch.Tensor) -> None:
        """Drop finished bodies from the brain and from the per-body agent state (see `Brain.compact`).

        Call `env.compact(keep)` with the same mask; `unpack()`ed parameters
        must be index-selected by the caller (`theta[k][keep]`).
        """
        keep = keep.to(self.device, torch.bool)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        batch = self.brain.batch
        for name in ("motor_state", "prev_proximity", "last_proximity", "last_loom", "last_motor", "last_rates_hz"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == batch:
                setattr(self, name, value[idx])
        self.brain.compact(keep)

    def channels(self, signal: torch.Tensor) -> torch.Tensor:
        """Projected, normalised readout channels (batch, readout_dim) from common-mode-free rates."""
        cfg = self.cfg
        mixed = signal @ self.projection
        if cfg.readout_norm == "layer":
            floor = cfg.readout_floor_hz * self.brain.cfg.dt_ms / 1000.0  # spikes per substep
            return mixed * torch.rsqrt(mixed.square().sum(dim=1, keepdim=True) + cfg.readout_dim * floor * floor)
        if cfg.readout_norm == "channel":
            return (mixed - self.readout_mean) / self.readout_std
        return mixed

    def set_readout(self, projection: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install a calibrated projection and channel statistics (see `calibrate.py`)."""
        if projection.shape != (self.n_readout, self.cfg.readout_dim):
            raise ValueError(f"projection {tuple(projection.shape)} does not match ({self.n_readout}, {self.cfg.readout_dim})")
        self.projection = projection.to(self.device, torch.float32)
        self.readout_mean = mean.to(self.device, torch.float32).reshape(-1)
        self.readout_std = std.to(self.device, torch.float32).reshape(-1).clamp(min=1e-6)
        self.readout_calibrated = True

    def readout_state(self) -> dict | None:
        """Calibrated readout for a checkpoint, or None when the projection is the seeded random one."""
        if not self.readout_calibrated:
            return None
        return {
            "projection": self.projection.detach().cpu(),
            "mean": self.readout_mean.detach().cpu(),
            "std": self.readout_std.detach().cpu(),
        }

    def load_readout(self, state: dict | None) -> bool:
        """Install the checkpoint's calibrated readout if it carries one. Returns True when it did."""
        saved = (state or {}).get("readout")
        if not saved:
            return False
        self.set_readout(saved["projection"], saved["mean"], saved["std"])
        return True

    def steer_weight(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """Effective steering weight per output neuron: the fixed projection folded into `w_out`'s direction and gain."""
        w = theta["w_out"][0, :, 0]
        w_hat = w * torch.rsqrt(w.square().sum() + self.cfg.readout_eps)
        return self.projection @ w_hat * theta["g_out"][0, 0]

    def dn_steer_weight(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """`steer_weight` restricted to the descending neurons (zero for DNs outside the readout)."""
        w = self.steer_weight(theta)[self.dn_in_readout.clamp(min=0)]
        return torch.where(self.dn_in_readout >= 0, w, torch.zeros_like(w))

    @property
    def readout_rate_hz(self) -> torch.Tensor:
        """Filtered output-population rate per (body, neuron), in spikes/second."""
        return self.motor_state * (1000.0 / self.brain.cfg.dt_ms)

    @property
    def dn_rate_hz(self) -> torch.Tensor:
        """Filtered descending-neuron rate per (body, neuron), in spikes/second (zero for DNs outside the readout)."""
        rates = self.readout_rate_hz
        picked = rates[:, self.dn_in_readout.clamp(min=0)]
        return torch.where((self.dn_in_readout >= 0).unsqueeze(0), picked, torch.zeros_like(picked))

    def proximity(self, lidar: torch.Tensor) -> torch.Tensor:
        """Per-ray drive in [0, 1] from normalised lidar (batch, n_rays): near wall -> high."""
        cfg = self.cfg
        if cfg.eye_encoding == "linear":
            return (1.0 - lidar).clamp(0.0, 1.0)
        if cfg.eye_encoding != "road":
            raise ValueError(f"unknown eye_encoding {cfg.eye_encoding!r}; expected 'road' or 'linear'")
        metres = (lidar * cfg.eye_range_m).clamp(min=0.1)
        octaves = torch.log2(self.ray_ref_m / metres)
        return (cfg.eye_octave_gain * octaves + 0.5).clamp(0.0, 1.0)

    def sensory_rates(self, obs: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """Poisson rates (batch, n_rays + 1) in Hz for the ray groups and speed."""
        cfg = self.cfg
        lidar = obs[:, : cfg.n_rays]
        speed = obs[:, cfg.n_rays : cfg.n_rays + 1]
        proximity = self.proximity(lidar)
        drive = proximity * theta["ray_gain"] + theta["bias_hz"]
        loom = torch.zeros_like(proximity)
        if cfg.loom:
            prev = proximity if self.prev_proximity is None else self.prev_proximity
            loom = ((proximity - prev) * cfg.loom_scale).clamp(min=0.0)
            drive = drive + loom * theta["loom_gain"]
            self.prev_proximity = proximity
        # Kept for telemetry (the viewer's first-person view): what each eye
        # group is being told, before the rate nonlinearity.
        self.last_proximity = proximity
        self.last_loom = loom
        # Soft saturation: a hard clamp made every ray read the same once the
        # gains grew, erasing the left/right difference the steering needs.
        vis_hz = (1.0 - torch.exp(-drive.clamp(min=0.0))) * cfg.max_input_hz
        speed_hz = (1.0 - torch.exp(-(speed * theta["speed_gain"]).clamp(min=0.0))) * cfg.max_input_hz
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
        self.last_rates_hz = rates_hz

        if "dn_gain" in theta:
            gain = torch.ones(batch, self.brain.n, device=self.device)
            gain[:, self.dn_index] = theta["dn_gain"]
            self.brain.set_gain(gain)
        elif self.brain.gain is not None:
            self.brain.set_gain(None)

        # Bernoulli spike kicks (as in Shiu et al. 2024): per driven cell, the
        # probability of a kick this substep from its channel's rate, laid out
        # (cells, bodies) to match the brain's native state layout. The draws
        # themselves happen inside the brain step.
        prob = (rates_hz[:, self.input_channel] * (self.brain.cfg.dt_ms / 1000.0)).clamp_(
            0.0, 1.0
        ).t().contiguous()
        base = (self.noise_seed * 1_000_003 + self.step_counter * cfg.substeps) & 0x7FFFFFFF
        self.step_counter += 1

        self.brain.dn_acc.zero_()
        for s in range(cfg.substeps):
            self.brain.step(
                external_index=self.input_index,
                poisson_prob=prob,
                seed=(base + s) & 0x7FFFFFFF,
                shared_noise=cfg.shared_noise,
                dense=False,
            )
            if spike_sink is not None:
                spike_sink.add_(self.brain.dense_spikes())

        rate = self.brain.dn_acc / cfg.substeps
        self.motor_state = (1 - cfg.motor_tau) * self.motor_state + cfg.motor_tau * rate
        signal = self.motor_state
        if cfg.common_mode:
            signal = signal - signal.mean(dim=1, keepdim=True)
        mixed = self.channels(signal)
        w_hat = theta["w_out"] * torch.rsqrt(theta["w_out"].square().sum(dim=1, keepdim=True) + cfg.readout_eps)
        motor = theta["g_out"] * torch.einsum("bd,bdk->bk", mixed, w_hat) + theta["b_out"]
        self.last_motor = motor
        return torch.tanh(motor)
