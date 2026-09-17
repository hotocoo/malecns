"""Scripted lidar teacher for readout calibration (`calibrate.py`).

A *linear* reactive driver over the agent's own eye features: steering is an
antisymmetric sum of left-minus-right ray proximities, the pedal a speed
governor braked by the forward rays. Linear in exactly the features the
brain transmits well, so a linear readout of the connectome can imitate it;
the earlier hand-written nonlinear teacher (steer towards the longest ray)
was only 60% linearly predictable from its own inputs and 35% from the brain.

The weights were found by a small evolution search on the CPU environment
(see `data/teacher_linear.json`); it laps full-scale Monaco from every start
without touching a wall.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

DEFAULT_PATH = Path("data/teacher_linear.json")
# Fallback weights: steer w[0..3] for ray pairs (0,8), (1,7), (2,6), (3,5);
# pedal a0 - a1*p4 - a2*(p3+p5) - a3*(p2+p6) - a4*speed.
FALLBACK = [1.47, 0.56, 0.50, 0.57, 1.65, 1.30, 0.13, 0.12, 3.93]


class LinearTeacher:
    """`act(prox, speed)` -> (batch, 2) steer/pedal from road-relative proximities (batch, n_rays >= 9) and speed/max_speed (batch,)."""

    def __init__(self, params: list[float] | torch.Tensor | None = None, path: Path | str = DEFAULT_PATH) -> None:
        if params is None:
            path = Path(path)
            params = json.loads(path.read_text())["params"] if path.exists() else FALLBACK
        self.params = torch.as_tensor(params, dtype=torch.float32).reshape(-1)
        if self.params.numel() != 9:
            raise ValueError(f"linear teacher takes 9 weights, got {self.params.numel()}")

    def act(self, prox: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        # The weights were found on a 9-ray eye. They describe angular
        # positions, not ray indices, so a finer eye is resampled onto the same
        # nine directions across the same field of view and the teacher drives
        # exactly as it did before.
        if prox.shape[-1] != 9:
            if prox.shape[-1] < 9:
                raise ValueError(f"linear teacher needs at least 9 rays, got {prox.shape[-1]}")
            prox = torch.nn.functional.interpolate(prox.unsqueeze(1), size=9, mode="linear", align_corners=True).squeeze(1)
        p = self.params.to(prox.device)
        asym = prox[:, :4] - prox[:, [8, 7, 6, 5]]  # left minus right
        steer = -(asym * p[:4]).sum(1)
        pedal = p[4] - p[5] * prox[:, 4] - p[6] * (prox[:, 3] + prox[:, 5]) - p[7] * (prox[:, 2] + prox[:, 6]) - p[8] * speed
        return torch.stack([steer.clamp(-1, 1), pedal.clamp(-1, 1)], dim=1)
