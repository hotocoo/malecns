"""Turn camera frames into the signal the connectome drives on.

The chain is: `camera.py` renders pixels, this module runs a real detector over
them, and the detections become the brain's sensory input. Nothing skips a
link. The driver is never told where a car is; it is told what the detector
found, with the detector's misses and mistakes intact, because those are what a
camera-driven vehicle actually has to cope with.

The encoding keeps the shape the connectome already expects from its eye: the
view is split into columns left to right, the same order the visual neuron
groups are fed in. Each column carries, per class, how near the nearest
instance of that class is - box height standing in for range, the way it does
for a camera with a fixed lens - scaled by the detector's own confidence.

A traffic light gets one extra treatment: its phase is read out of the pixels
inside its box, not from the simulation. If the detector finds a light and the
lit lens is red, the driver sees red; if the detector misses the light, the
driver sees nothing, and the law still charges for running it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scene import (
    CLASS_BUS,
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_PERSON,
    CLASS_STOP_SIGN,
    CLASS_TRAFFIC_LIGHT,
    CLASS_TRUCK,
)

# The classes the driver is given a channel for. Trucks and buses fold into the
# car channel: to a driver they are the same obstacle, only larger, and the box
# already carries the size.
CHANNELS = (CLASS_CAR, CLASS_MOTORCYCLE, CLASS_PERSON, CLASS_TRAFFIC_LIGHT, CLASS_STOP_SIGN)
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS)}
FOLD_INTO_CAR = (CLASS_TRUCK, CLASS_BUS)

# Signal phases read out of the pixels, as three extra channels.
PHASE_CHANNELS = ("signal_red", "signal_amber", "signal_green")

DEFAULT_WEIGHTS = "yolo26n.pt"


@dataclass(frozen=True)
class Detection:
    """One thing the detector found, in image coordinates."""

    kind: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centre_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.x2 - self.x1, 0.0) * max(self.y2 - self.y1, 0.0)


@dataclass(frozen=True)
class EyeConfig:
    """How detections become the brain's input."""

    columns: int = 12
    confidence: float = 0.25
    # A box this tall fills the near field; taller saturates. Chosen as a
    # fraction of frame height, so it holds at any resolution.
    near_height_fraction: float = 0.45
    include_phase: bool = True

    @property
    def width(self) -> int:
        return self.columns * (len(CHANNELS) + (len(PHASE_CHANNELS) if self.include_phase else 0))


def classify_lens(patch: np.ndarray) -> str | None:
    """Which lens is lit in a traffic-light crop, from the crop's own pixels.

    Reads the brightest saturated pixels and asks whether they sit in the red,
    amber or green part of the hue circle. Returns None when the crop carries
    no lit lens, which is the honest answer for a light seen side-on.
    """
    if patch.size == 0:
        return None
    rgb = patch.astype(np.float32) / 255.0
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    saturation = np.where(high > 0.0, (high - low) / np.maximum(high, 1e-6), 0.0)
    lit = (saturation > 0.35) & (high > 0.35)
    if lit.sum() < 3:
        return None
    r, g, b = (rgb[..., i][lit].mean() for i in range(3))
    if r > 0.45 and g < 0.45 and b < 0.45:
        return "signal_red"
    if r > 0.45 and g > 0.40 and b < 0.45:
        return "signal_amber"
    if g > 0.40 and g >= r:
        return "signal_green"
    return None


class Detector:
    """A real object detector over rendered frames.

    Weights are loaded once and reused; a batch of frames goes through in one
    call, because the driving loop renders one frame per body per step and the
    per-call overhead would otherwise dominate.
    """

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | None = None, imgsz: int = 640) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = dict(self.model.names)
        self.device = device
        self.imgsz = imgsz

    def detect(self, frames: list[np.ndarray] | np.ndarray, confidence: float = 0.25) -> list[list[Detection]]:
        """Detections per frame. Frames are RGB `uint8`; the detector wants BGR."""
        batch = [frames] if isinstance(frames, np.ndarray) and frames.ndim == 3 else list(frames)
        if not batch:
            return []
        bgr = [frame[:, :, ::-1] for frame in batch]
        results = self.model.predict(
            bgr, verbose=False, conf=confidence, imgsz=self.imgsz, device=self.device
        )
        out: list[list[Detection]] = []
        for result in results:
            found: list[Detection] = []
            for box in result.boxes:
                kind = self.names[int(box.cls)]
                if kind in FOLD_INTO_CAR:
                    kind = CLASS_CAR
                if kind not in CHANNEL_INDEX:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                found.append(Detection(kind, float(box.conf), x1, y1, x2, y2))
            out.append(found)
        return out


@dataclass
class DetectionEye:
    """Detections rendered into the vector the connectome's visual groups read."""

    config: EyeConfig = field(default_factory=EyeConfig)

    @property
    def width(self) -> int:
        return self.config.width

    def channel_names(self) -> tuple[str, ...]:
        names = list(CHANNELS)
        if self.config.include_phase:
            names += list(PHASE_CHANNELS)
        return tuple(f"{name}[{column}]" for name in names for column in range(self.config.columns))

    def encode(self, detections: list[Detection], frame: np.ndarray) -> np.ndarray:
        """One frame's detections as a flat vector in [0, 1].

        Column 0 is the left edge of the view, matching the ray order the eye
        used before, so the visual groups keep their meaning.
        """
        cfg = self.config
        height, width = frame.shape[:2]
        names = list(CHANNELS) + (list(PHASE_CHANNELS) if cfg.include_phase else [])
        grid = np.zeros((len(names), cfg.columns), dtype=np.float32)
        index = {name: i for i, name in enumerate(names)}

        near = cfg.near_height_fraction * height
        for found in detections:
            if found.confidence < cfg.confidence:
                continue
            column = int(found.centre_x / width * cfg.columns)
            column = min(max(column, 0), cfg.columns - 1)
            nearness = min(found.height / max(near, 1e-6), 1.0) * found.confidence
            row = index[found.kind]
            grid[row, column] = max(grid[row, column], nearness)

            if cfg.include_phase and found.kind == CLASS_TRAFFIC_LIGHT:
                crop = frame[
                    max(int(found.y1), 0) : max(int(found.y2), 1),
                    max(int(found.x1), 0) : max(int(found.x2), 1),
                ]
                phase = classify_lens(crop)
                if phase is not None:
                    prow = index[phase]
                    grid[prow, column] = max(grid[prow, column], nearness)

        return grid.reshape(-1)

    def encode_batch(self, detections: list[list[Detection]], frames: list[np.ndarray]) -> np.ndarray:
        """(batch, width) for a population of cars."""
        if not frames:
            return np.zeros((0, self.width), dtype=np.float32)
        return np.stack([self.encode(d, f) for d, f in zip(detections, frames)])


def describe(detections: list[Detection]) -> str:
    counts: dict[str, int] = {}
    for found in detections:
        counts[found.kind] = counts.get(found.kind, 0) + 1
    return ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())) or "nothing"
