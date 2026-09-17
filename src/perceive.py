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

# The drivable surface, measured from the frame itself. A detector reports
# objects; it says nothing about where the road goes, and a driver given only
# objects has no information at all on an empty street. These two channels
# carry what the pixels say about the road: how far it runs in each direction,
# and where its painted markings are.
ROAD_CHANNELS = ("road_extent", "lane_marking")

# Rows of the frame, as fractions of its height, where the road's left and
# right boundary are measured. Near rows carry lateral offset, far rows carry
# heading error; together they are what a driver reads off the kerb line.
EDGE_ROWS = (0.96, 0.88, 0.78, 0.68)
EDGE_CHANNELS = tuple(f"{side}_edge[{k}]" for k in range(len(EDGE_ROWS)) for side in ("left", "right"))

# Rows where the lane the car sits in is measured, near to far. The lane is
# read from the paint that brackets the frame centre, so what comes out is the
# lane the car is in rather than the carriageway it belongs to.
LANE_ROWS = (0.97, 0.90, 0.83, 0.76, 0.69, 0.62)
LANE_CHANNELS = tuple(f"lane_offset[{k}]" for k in range(len(LANE_ROWS))) + tuple(
    f"lane_width[{k}]" for k in range(len(LANE_ROWS))
) + ("lane_heading", "lane_curvature", "lane_seen")

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
    include_road: bool = True
    # A box this tall fills the near field; taller saturates. Chosen as a
    # fraction of frame height, so it holds at any resolution.
    near_height_fraction: float = 0.45
    include_phase: bool = True
    include_edges: bool = True
    edge_rows: tuple[float, ...] = EDGE_ROWS
    include_lane: bool = True
    lane_rows: tuple[float, ...] = LANE_ROWS

    @property
    def width(self) -> int:
        channels = len(CHANNELS)
        if self.include_phase:
            channels += len(PHASE_CHANNELS)
        if self.include_road:
            channels += len(ROAD_CHANNELS)
        extra = len(EDGE_CHANNELS) if self.include_edges else 0
        extra += len(LANE_CHANNELS) if self.include_lane else 0
        return self.columns * channels + extra


def surface_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which pixels look like road, and which of those look painted.

    Asphalt is dark and close to grey; paint is bright and close to grey.
    Everything else in the scene - render, glass, pavers, kerb paint, grass -
    is kept off neutral grey by the renderer so that it cannot be mistaken
    for road here.
    """
    rgb = frame.astype(np.float32) / 255.0
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    saturation = np.where(high > 0.0, (high - low) / np.maximum(high, 1e-6), 0.0)
    grey = saturation < 0.18
    asphalt = grey & (high > 0.06) & (high < 0.55)
    paint = grey & (high >= 0.55)
    return asphalt | paint, paint


def road_edges(frame: np.ndarray, rows: tuple[float, ...] = EDGE_ROWS) -> np.ndarray:
    """The road's left and right boundary at each row, in [-1, 1] of frame width.

    The scan starts at the road pixel nearest the frame centre and walks out
    either way while the row stays road-like, so what comes back is the
    carriageway the car is on rather than the widest run in the row. A row
    with no road under the centre reports both edges at the centre, which is
    what the driver has when the road has gone from view.
    """
    height, width = frame.shape[:2]
    surface, _paint = surface_mask(frame)
    centre = width // 2
    reach = max(width // 3, 1)
    out = np.zeros(2 * len(rows), dtype=np.float32)
    for k, fraction in enumerate(rows):
        y = int(np.clip(fraction * height, 0, height - 1))
        line = surface[y]
        start = -1
        if line[centre]:
            start = centre
        else:
            for step in range(1, reach):
                if centre - step >= 0 and line[centre - step]:
                    start = centre - step
                    break
                if centre + step < width and line[centre + step]:
                    start = centre + step
                    break
        if start < 0:
            continue
        left = start
        while left > 0 and line[left - 1]:
            left -= 1
        right = start
        while right < width - 1 and line[right + 1]:
            right += 1
        out[2 * k] = (left / max(width - 1, 1)) * 2.0 - 1.0
        out[2 * k + 1] = (right / max(width - 1, 1)) * 2.0 - 1.0
    return out


def lane_geometry(frame: np.ndarray, rows: tuple[float, ...] = LANE_ROWS) -> np.ndarray:
    """The lane the car is in, measured from the paint at each row.

    At every row the nearest painted pixel either side of the frame centre is
    found, inside the road surface. Their midpoint is where the lane centre
    projects, and their distance is how wide the lane looks, both in units of
    half the frame width. From the midpoints across rows come a heading error
    (how fast the lane centre slides across the frame) and a curvature (how
    that slide itself changes). A row with no paint either side reports zero,
    and `lane_seen` says what share of rows were measured, so the driver can
    tell "lane dead ahead" from "no paint found".
    """
    height, width = frame.shape[:2]
    surface, paint = surface_mask(frame)
    centre = width // 2
    half = max(width / 2.0, 1.0)
    offsets = np.zeros(len(rows), dtype=np.float32)
    widths = np.zeros(len(rows), dtype=np.float32)
    seen = np.zeros(len(rows), dtype=bool)
    for k, fraction in enumerate(rows):
        y = int(np.clip(fraction * height, 0, height - 1))
        line = paint[y] & surface[y]
        # Paint comes in runs several pixels wide, and a run can straddle the
        # frame centre, so the runs are found first and then the nearest one
        # either side is taken by its own centre. Taking the nearest painted
        # pixel instead reported one run as both edges of the lane.
        edges = np.flatnonzero(np.diff(np.concatenate(([False], line, [False])).astype(np.int8)))
        starts, ends = edges[0::2], edges[1::2]
        if starts.size == 0:
            continue
        middles = (starts + ends - 1) / 2.0
        to_left = middles[middles < centre]
        to_right = middles[middles > centre]
        if to_left.size == 0 or to_right.size == 0:
            continue
        left, right = float(to_left.max()), float(to_right.min())
        offsets[k] = ((left + right) / 2.0 - centre) / half
        widths[k] = (right - left) / half
        seen[k] = True

    # Heading is the slope of the lane centre across the measured rows, and
    # curvature its change: a straight lane slides linearly up the frame, a
    # bend does not.
    heading = 0.0
    curvature = 0.0
    index = np.flatnonzero(seen)
    if index.size >= 2:
        fit = np.polyfit(index.astype(np.float64), offsets[index].astype(np.float64), 1)
        heading = float(np.clip(fit[0], -1.0, 1.0))
    if index.size >= 3:
        fit = np.polyfit(index.astype(np.float64), offsets[index].astype(np.float64), 2)
        curvature = float(np.clip(fit[0] * 4.0, -1.0, 1.0))
    share = float(seen.mean())
    return np.concatenate([offsets, widths, np.array([heading, curvature, share], dtype=np.float32)])


def road_profile(frame: np.ndarray, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """How far the road runs in each column, and where its markings are.

    Read from the frame and nothing else. Asphalt is dark and close to grey;
    paint is bright and close to grey. For each column band the road extent is
    the fraction of the image height, measured up from the bottom, that stays
    road-like without a break, which is short where the carriageway curves away
    and long where it runs ahead. The marking channel is the share of those
    pixels that are painted, which rises as the car nears a line.

    This is a measurement of the picture, not a lookup of the track: it is
    wrong where the light is wrong, and it is the driver's only sense of the
    road, exactly as a camera-only vehicle's would be.
    """
    height, width = frame.shape[:2]
    surface, paint = surface_mask(frame)

    # The road can only lie below the horizon, which for a level camera at
    # eye height is the middle row. Counting above it let a grey building
    # wall continue the run to the top of the frame, and a column with a
    # tower in it read as road all the way. Measured against the horizon the
    # extent also uses the whole 0..1 range: 1.0 is road to the skyline.
    horizon = height // 2
    below = surface[horizon:]
    # One pass over the whole image instead of one per column: this runs once
    # per body per control step, and the per-column loop cost more than the
    # detector did.
    run = np.cumprod(below[::-1], axis=0).sum(axis=0)  # road pixels up from the bottom
    band = np.minimum((np.arange(width) * columns) // max(width, 1), columns - 1)
    extent = np.zeros(columns, dtype=np.float32)
    marking = np.zeros(columns, dtype=np.float32)
    rows_below = max(height - horizon, 1)
    for c in range(columns):
        take = band == c
        if not take.any():
            continue
        extent[c] = min(float(np.median(run[take])) / rows_below, 1.0)
        marking[c] = float(paint[horizon:, take].mean())
    return extent, marking


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
        grid = tuple(
            f"{name}[{column}]" for name in self._rows() for column in range(self.config.columns)
        )
        grid += EDGE_CHANNELS if self.config.include_edges else ()
        return grid + (LANE_CHANNELS if self.config.include_lane else ())

    def _rows(self) -> list[str]:
        names = list(CHANNELS)
        if self.config.include_phase:
            names += list(PHASE_CHANNELS)
        if self.config.include_road:
            names += list(ROAD_CHANNELS)
        return names

    def encode(self, detections: list[Detection], frame: np.ndarray) -> np.ndarray:
        """One frame's detections as a flat vector in [0, 1].

        Column 0 is the left edge of the view, matching the ray order the eye
        used before, so the visual groups keep their meaning.
        """
        cfg = self.config
        height, width = frame.shape[:2]
        names = self._rows()
        grid = np.zeros((len(names), cfg.columns), dtype=np.float32)
        index = {name: i for i, name in enumerate(names)}

        if cfg.include_road:
            extent, marking = road_profile(frame, cfg.columns)
            grid[index["road_extent"]] = extent
            grid[index["lane_marking"]] = marking

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

        measured = [grid.reshape(-1)]
        if cfg.include_edges:
            measured.append(road_edges(frame, cfg.edge_rows))
        if cfg.include_lane:
            measured.append(lane_geometry(frame, cfg.lane_rows))
        return np.concatenate(measured) if len(measured) > 1 else measured[0]

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
