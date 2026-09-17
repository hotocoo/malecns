"""Charge the driver for breaking the road law, using the survey's own numbers.

`roadlaw.py` says what the law is along a circuit - the posted limit at each
point, which side to keep, where the signals and stop lines stand. This module
turns that into a per-step charge the trainer can add to its reward.

The split matters. Nothing here decides *what* is illegal: the limit is
whatever OSM posts for the road under the car, the side is whatever the survey
records for the country, and a signal exists only where a surveyed node says
one does. What lives here is how hard each breach is charged, which is reward
tuning, not law - `LawWeights` holds those and nothing else.

One thing the survey cannot supply: OSM records that a junction is signalised,
not what the lights are doing right now. The phase is therefore simulated, with
its period declared in `SignalTiming` and its offset derived from the node's own
id so a junction behaves the same way every episode. That is a property of the
simulation, and it is named as one rather than dressed up as law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from roadlaw import UNKNOWN_LANES, ControlPoint, LegalProfile

# Signal phases, in the order a Malaysian junction runs them.
GREEN, AMBER, RED = 0, 1, 2


@dataclass(frozen=True)
class SignalTiming:
    """How long a simulated signal holds each phase.

    OpenStreetMap maps that a junction is signalised; it does not publish the
    controller's plan. These periods are a simulation parameter, adjustable per
    run, and are not presented as a legal quantity.
    """

    green_s: float = 25.0
    amber_s: float = 3.0
    red_s: float = 22.0

    @property
    def cycle_s(self) -> float:
        return self.green_s + self.amber_s + self.red_s


@dataclass(frozen=True)
class LawWeights:
    """How hard each breach is charged. Reward tuning, not law."""

    speeding: float = 0.6  # per (fraction over the posted limit) squared
    speeding_free: float = 0.0  # fraction of the limit tolerated before charging
    wrong_side: float = 0.8  # per metre on the wrong half of a two-way road
    red_signal: float = 12.0  # for entering the junction against a red
    stop_line: float = 8.0  # for crossing a stop line without stopping
    crossing_speed: float = 0.4  # for carrying speed across a pedestrian crossing

    # How close counts as "at" a control point, and how slow counts as stopped.
    control_reach_m: float = 8.0
    stop_speed_mps: float = 0.5
    crossing_speed_ref_mps: float = 8.3  # the speed a crossing is charged against


def signal_phase(node_ids: torch.Tensor, elapsed_s: torch.Tensor, timing: SignalTiming) -> torch.Tensor:
    """The phase of each signal at `elapsed_s`, as GREEN / AMBER / RED.

    The offset within the cycle comes from the node id, so two junctions are
    not in lockstep and each one is reproducible across episodes.
    """
    offset = (node_ids.to(torch.float64) % 997.0) / 997.0 * timing.cycle_s
    t = torch.remainder(elapsed_s.to(torch.float64) + offset, timing.cycle_s)
    phase = torch.full_like(t, RED, dtype=torch.long)
    phase = torch.where(t < timing.green_s + timing.amber_s, torch.full_like(phase, AMBER), phase)
    phase = torch.where(t < timing.green_s, torch.full_like(phase, GREEN), phase)
    return phase


def signal_phase_numpy(node_ids: np.ndarray, elapsed_s: np.ndarray, timing: SignalTiming) -> np.ndarray:
    """`signal_phase` on numpy arrays, for the renderer.

    The lights the camera sees must show the phase the reward charges against,
    so both read the same function rather than two copies of the same idea.
    """
    phase = signal_phase(
        torch.as_tensor(np.asarray(node_ids, dtype=np.int64)),
        torch.as_tensor(np.asarray(elapsed_s, dtype=np.float32)),
        timing,
    )
    return phase.cpu().numpy()


def _progress_index(progress: torch.Tensor, n: int) -> torch.Tensor:
    return (progress * n).long().clamp(0, n - 1) % n


def signed_offset(
    pos: torch.Tensor, centerline: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Metres left (+) or right (-) of the centerline at `index`.

    Left and right are taken in the direction of travel, so the sign means the
    same thing to the keep-left rule wherever the car is on the lap.
    """
    n = centerline.shape[0]
    here = centerline[index]
    ahead = centerline[(index + 1) % n]
    tangent = ahead - here
    tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp(min=1e-6)
    delta = pos - here
    # Cross product z-component: positive when the car is to the left.
    return tangent[:, 0] * delta[:, 1] - tangent[:, 1] * delta[:, 0]


class LawEnforcer:
    """The road law of one circuit, charged per step against a population of cars."""

    def __init__(
        self,
        profile: LegalProfile,
        control_points: Iterable[ControlPoint],
        centerline: torch.Tensor,
        device: torch.device,
        weights: LawWeights | None = None,
        timing: SignalTiming | None = None,
    ) -> None:
        self.weights = weights or LawWeights()
        self.timing = timing or SignalTiming()
        self.device = device
        self.centerline = centerline.to(device)
        self.n = centerline.shape[0]

        # An unposted limit is not a limit of zero: it is no charge at all.
        limit = np.where(np.isfinite(profile.limit_mps), profile.limit_mps, np.inf)
        self.limit_mps = torch.tensor(limit, dtype=torch.float32, device=device)
        self.two_way = torch.tensor(profile.oneway == 0, dtype=torch.bool, device=device)
        self.lanes = torch.tensor(profile.lanes, dtype=torch.long, device=device)
        self.driving_side = profile.driving_side
        # +1 when the law keeps the car left of the centerline, -1 right, 0 when
        # the survey does not say and the rule is therefore not enforced.
        self.side_sign = {"left": -1.0, "right": 1.0}.get(profile.driving_side or "", 0.0)

        points = list(control_points)
        self.signals = self._control_arrays(points, "red_signal")
        self.stops = self._control_arrays(points, "traffic_sign", tag_values={"highway": ("stop", "give_way")})
        self.crossings = self._control_arrays(points, "pedestrian_crossing")

    def _control_arrays(
        self,
        points: list[ControlPoint],
        rule_id: str,
        tag_values: dict[str, tuple[str, ...]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Lap positions and node ids of the control points that carry `rule_id`."""
        chosen = []
        for point in points:
            if rule_id not in point.rules or point.progress < 0.0:
                continue
            if tag_values and not any(point.tags.get(k) in v for k, v in tag_values.items()):
                continue
            chosen.append(point)
        if not chosen:
            empty = torch.zeros(0, device=self.device)
            return {"progress": empty, "node_id": empty.long()}
        return {
            "progress": torch.tensor([p.progress for p in chosen], dtype=torch.float32, device=self.device),
            "node_id": torch.tensor([p.node_id for p in chosen], dtype=torch.long, device=self.device),
        }

    @property
    def has_signals(self) -> bool:
        return self.signals["progress"].numel() > 0

    def limit_at(self, progress: torch.Tensor) -> torch.Tensor:
        """The posted limit under each car, in m/s; `inf` where nothing is posted."""
        return self.limit_mps[_progress_index(progress, self.n)]

    def _crossed(self, prev_progress: torch.Tensor, progress: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        """(batch, marks) True where a car passed that mark during this step."""
        if marks.numel() == 0:
            return torch.zeros(progress.shape[0], 0, dtype=torch.bool, device=self.device)
        lo = prev_progress.unsqueeze(1)
        hi = progress.unsqueeze(1)
        wrapped = hi < lo
        within = (marks.unsqueeze(0) > lo) & (marks.unsqueeze(0) <= hi)
        over_wrap = (marks.unsqueeze(0) > lo) | (marks.unsqueeze(0) <= hi)
        return torch.where(wrapped.expand_as(within), over_wrap, within)

    def charge(
        self,
        progress: torch.Tensor,
        prev_progress: torch.Tensor,
        speed: torch.Tensor,
        pos: torch.Tensor,
        elapsed_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per-body charges for this step, one entry per rule plus the total.

        Every charge is zero or negative. A rule with no data on this circuit
        contributes zeros rather than an assumption.
        """
        weights = self.weights
        index = _progress_index(progress, self.n)
        zero = torch.zeros_like(speed)

        # Section 40(1): driving above the limit posted for this road.
        limit = self.limit_mps[index]
        allowed = limit * (1.0 + weights.speeding_free)
        over = torch.where(torch.isfinite(allowed), torch.relu(speed - allowed) / allowed.clamp(min=1e-6), zero)
        speeding = -weights.speeding * over * over

        # Rule 3, Road Traffic Rules 1959: keep to the surveyed side. Only on a
        # two-way road, and only where the survey states a side.
        offset = signed_offset(pos, self.centerline, index)
        if self.side_sign == 0.0:
            wrong_side = zero
        else:
            # `offset` is positive to the left of the centerline. Where traffic
            # keeps left (side_sign -1) the wrong half is the right one, so the
            # product is positive exactly when the car is where it should not
            # be; where traffic keeps right (+1) the same expression flips.
            wrong = torch.relu(offset * self.side_sign)
            wrong_side = -weights.wrong_side * wrong * self.two_way[index].to(speed.dtype)

        # Second Schedule item (iii): entering against a red. Charged on the
        # step the car passes the junction, not while it waits at it.
        crossed_signal = self._crossed(prev_progress, progress, self.signals["progress"])
        if crossed_signal.numel() == 0:
            red_signal = zero
        else:
            phase = signal_phase(
                self.signals["node_id"].unsqueeze(0).expand(speed.shape[0], -1),
                elapsed_s.unsqueeze(1).expand(-1, self.signals["node_id"].shape[0]),
                self.timing,
            )
            against_red = crossed_signal & (phase == RED)
            red_signal = -weights.red_signal * against_red.any(dim=1).to(speed.dtype)

        # Section 79(2) at a stop or give-way sign: the stop line must be met at
        # rest. Charged in proportion to the speed carried across it.
        crossed_stop = self._crossed(prev_progress, progress, self.stops["progress"])
        if crossed_stop.numel() == 0:
            stop_line = zero
        else:
            rolling = torch.relu(speed - weights.stop_speed_mps) / weights.crossing_speed_ref_mps
            stop_line = -weights.stop_line * crossed_stop.any(dim=1).to(speed.dtype) * rolling.clamp(max=1.0)

        # Section 75: a pedestrian crossing is not taken at speed.
        crossed_crossing = self._crossed(prev_progress, progress, self.crossings["progress"])
        if crossed_crossing.numel() == 0:
            crossing = zero
        else:
            carried = (speed / weights.crossing_speed_ref_mps).clamp(min=0.0)
            crossing = -weights.crossing_speed * crossed_crossing.any(dim=1).to(speed.dtype) * carried * carried

        total = speeding + wrong_side + red_signal + stop_line + crossing
        return {
            "law_speeding": speeding,
            "law_wrong_side": wrong_side,
            "law_red_signal": red_signal,
            "law_stop_line": stop_line,
            "law_crossing": crossing,
            "law_total": total,
            "law_limit_mps": limit,
            "law_offset_m": offset,
        }

    def summary(self) -> str:
        posted = torch.isfinite(self.limit_mps).float().mean().item()
        return (
            f"law: {posted:.0%} of the lap posts a limit, side={self.driving_side or 'unknown'}, "
            f"{self.signals['progress'].numel()} signals, {self.stops['progress'].numel()} stop lines, "
            f"{self.crossings['progress'].numel()} crossings"
        )
