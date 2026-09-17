"""The world the camera sees: road surface, markings, signals, signs and actors.

Everything with a place in this scene has that place because the survey put it
there. The road is the circuit's own centerline widened by the lane count OSM
records; the lane markings follow the direction of travel the survey states; a
traffic light stands where a node tagged `highway=traffic_signals` stands, and a
stop sign where one tagged `highway=stop` does.

Geometry only: this module builds vertex buffers and actor states in world
metres. `camera.py` turns them into pixels, `perceive.py` turns those into
detections, and `lawreward.py` charges the driver against the same survey.

Lane width is the one dimension OSM usually leaves out. Where a way tags
`width`, that width is used; otherwise the road is as wide as its lane count
times the run's `lane_width_m`, which is a simulation parameter, declared here
rather than buried in the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lawreward import GREEN, RED, SignalTiming, signal_phase_numpy
from roadlaw import UNKNOWN_LANES, ControlPoint, LegalProfile

# Object classes the detector is asked to find. These are COCO class names,
# because the detector is a COCO model; the scene builds objects that really
# are these things rather than props that merely resemble them.
CLASS_CAR = "car"
CLASS_MOTORCYCLE = "motorcycle"
CLASS_PERSON = "person"
CLASS_TRAFFIC_LIGHT = "traffic light"
CLASS_STOP_SIGN = "stop sign"
CLASS_TRUCK = "truck"
CLASS_BUS = "bus"

SCENE_CLASSES = (
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_PERSON,
    CLASS_TRAFFIC_LIGHT,
    CLASS_STOP_SIGN,
    CLASS_TRUCK,
    CLASS_BUS,
)


@dataclass(frozen=True)
class SceneConfig:
    """Dimensions the survey does not carry. Simulation parameters, not law."""

    lane_width_m: float = 3.5
    default_lanes: int = 2
    shoulder_m: float = 0.5
    marking_width_m: float = 0.12
    dash_len_m: float = 3.0
    dash_gap_m: float = 6.0

    signal_height_m: float = 4.2  # head height above the road
    signal_offset_m: float = 0.6  # clear of the kerb, on the near side
    signal_head_m: float = 0.9  # head is this tall
    sign_height_m: float = 2.2
    sign_size_m: float = 0.75

    car_length_m: float = 4.4
    car_width_m: float = 1.8
    car_height_m: float = 1.45
    motorcycle_length_m: float = 2.1
    motorcycle_width_m: float = 0.7
    motorcycle_height_m: float = 1.5
    person_height_m: float = 1.7
    person_width_m: float = 0.5


@dataclass(frozen=True)
class RoadMesh:
    """The drivable ribbon and its markings, as triangles in world metres."""

    surface: np.ndarray  # (v, 3) float32 road surface vertices
    surface_colour: np.ndarray  # (v, 3) float32
    markings: np.ndarray  # (v, 3) float32 painted lines, drawn just above
    markings_colour: np.ndarray  # (v, 3) float32
    halfwidth_m: np.ndarray  # (n,) per centerline sample


@dataclass
class Actor:
    """One thing on or beside the road that the detector should find."""

    kind: str  # a member of SCENE_CLASSES
    pos: np.ndarray  # (3,) world metres, at the object's base
    heading: float  # radians
    size: np.ndarray  # (3,) length, width, height
    colour: np.ndarray  # (3,) float32
    progress: float = -1.0  # lap fraction, for the ones that travel
    speed_mps: float = 0.0
    node_id: int = -1  # the surveyed node it belongs to, when it has one
    state: int = -1  # signal phase, for a traffic light


def lane_halfwidth(profile: LegalProfile, cfg: SceneConfig) -> np.ndarray:
    """Half the carriageway at each centerline sample, in metres.

    Lane count comes from the survey; where it is missing the road gets
    `cfg.default_lanes`, which is declared rather than inferred.
    """
    lanes = np.where(profile.lanes > UNKNOWN_LANES, profile.lanes, cfg.default_lanes)
    return lanes * cfg.lane_width_m / 2.0 + cfg.shoulder_m


def frame_at(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangent and left normal at every centerline sample."""
    ahead = np.roll(centerline, -1, axis=0)
    behind = np.roll(centerline, 1, axis=0)
    tangent = ahead - behind
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return tangent, normal


def _ribbon(
    centerline: np.ndarray,
    heights: np.ndarray,
    normal: np.ndarray,
    inner: np.ndarray,
    outer: np.ndarray,
    lift: float,
) -> np.ndarray:
    """Two triangles per sample between the `inner` and `outer` offsets.

    Offsets may be per-sample (a road whose width follows the lane count) or a
    single number (a painted line of constant width).
    """
    n = centerline.shape[0]
    inner = np.broadcast_to(np.asarray(inner, dtype=np.float64), (n,))
    outer = np.broadcast_to(np.asarray(outer, dtype=np.float64), (n,))
    left = centerline + normal * outer[:, None]
    right = centerline + normal * inner[:, None]
    z = heights + lift
    a = np.concatenate([left, z[:, None]], axis=1)
    b = np.concatenate([right, z[:, None]], axis=1)
    a2, b2 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    tris = np.stack([a, b, b2, a, b2, a2], axis=1)
    return tris.reshape(-1, 3).astype(np.float32)


def build_road(
    centerline: np.ndarray,
    heights: np.ndarray,
    profile: LegalProfile,
    cfg: SceneConfig | None = None,
) -> RoadMesh:
    """The road surface and its painted markings.

    A two-way road gets a dashed centre line; a one-way carriageway does not,
    because there is no opposing traffic to divide it from. Both get solid
    edge lines. Which is which comes from the survey's `oneway` tag.
    """
    cfg = cfg or SceneConfig()
    centerline = np.asarray(centerline, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    half = lane_halfwidth(profile, cfg)
    _, normal = frame_at(centerline)

    surface = _ribbon(centerline, heights, normal, -half, half, 0.0)
    surface_colour = np.tile(np.array([0.24, 0.24, 0.25], dtype=np.float32), (surface.shape[0], 1))

    mark = cfg.marking_width_m
    edge = half - cfg.shoulder_m
    pieces = [
        _ribbon(centerline, heights, normal, edge - mark, edge, 0.01),
        _ribbon(centerline, heights, normal, -edge, -edge + mark, 0.01),
    ]
    colours = [np.tile(np.array([0.92, 0.92, 0.88], dtype=np.float32), (pieces[0].shape[0], 1)) for _ in pieces]

    two_way = profile.oneway == 0
    if two_way.any():
        spacing = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())
        period = max(1, int(round((cfg.dash_len_m + cfg.dash_gap_m) / max(spacing, 1e-6))))
        painted = (np.arange(centerline.shape[0]) % period) < max(1, int(round(cfg.dash_len_m / max(spacing, 1e-6))))
        keep = two_way & painted
        if keep.any():
            centre = _ribbon(centerline, heights, normal, -mark / 2.0, mark / 2.0, 0.01)
            per_sample = centre.shape[0] // centerline.shape[0]
            mask = np.repeat(keep, per_sample)
            centre = centre[mask]
            pieces.append(centre)
            colours.append(np.tile(np.array([0.95, 0.90, 0.35], dtype=np.float32), (centre.shape[0], 1)))

    return RoadMesh(
        surface=surface,
        surface_colour=surface_colour,
        markings=np.concatenate(pieces).astype(np.float32),
        markings_colour=np.concatenate(colours).astype(np.float32),
        halfwidth_m=half,
    )


def place_signals(
    control_points: list[ControlPoint] | tuple[ControlPoint, ...],
    centerline: np.ndarray,
    heights: np.ndarray,
    profile: LegalProfile,
    driving_side: str | None,
    cfg: SceneConfig | None = None,
) -> list[Actor]:
    """A traffic light at every surveyed signal node, on the near side of the road."""
    cfg = cfg or SceneConfig()
    half = lane_halfwidth(profile, cfg)
    tangent, normal = frame_at(np.asarray(centerline, dtype=np.float64))
    side = -1.0 if (driving_side or "left") == "left" else 1.0
    n = centerline.shape[0]

    actors: list[Actor] = []
    for point in control_points:
        if "red_signal" not in point.rules or point.progress < 0.0:
            continue
        i = min(int(point.progress * n), n - 1)
        base = centerline[i] + normal[i] * side * (half[i] + cfg.signal_offset_m)
        actors.append(
            Actor(
                kind=CLASS_TRAFFIC_LIGHT,
                pos=np.array([base[0], base[1], heights[i] + cfg.signal_height_m], dtype=np.float64),
                heading=float(np.arctan2(tangent[i, 1], tangent[i, 0])) + np.pi,
                size=np.array([0.3, 0.35, cfg.signal_head_m]),
                colour=np.array([0.1, 0.1, 0.1], dtype=np.float32),
                progress=point.progress,
                node_id=point.node_id,
                state=RED,
            )
        )
    return actors


def place_signs(
    control_points: list[ControlPoint] | tuple[ControlPoint, ...],
    centerline: np.ndarray,
    heights: np.ndarray,
    profile: LegalProfile,
    driving_side: str | None,
    cfg: SceneConfig | None = None,
) -> list[Actor]:
    """A stop sign at every surveyed `highway=stop` node."""
    cfg = cfg or SceneConfig()
    half = lane_halfwidth(profile, cfg)
    tangent, normal = frame_at(np.asarray(centerline, dtype=np.float64))
    side = -1.0 if (driving_side or "left") == "left" else 1.0
    n = centerline.shape[0]

    actors: list[Actor] = []
    for point in control_points:
        if point.tags.get("highway") != "stop" or point.progress < 0.0:
            continue
        i = min(int(point.progress * n), n - 1)
        base = centerline[i] + normal[i] * side * (half[i] + cfg.signal_offset_m)
        actors.append(
            Actor(
                kind=CLASS_STOP_SIGN,
                pos=np.array([base[0], base[1], heights[i] + cfg.sign_height_m], dtype=np.float64),
                heading=float(np.arctan2(tangent[i, 1], tangent[i, 0])) + np.pi,
                size=np.array([0.05, cfg.sign_size_m, cfg.sign_size_m]),
                colour=np.array([0.78, 0.08, 0.10], dtype=np.float32),
                progress=point.progress,
                node_id=point.node_id,
            )
        )
    return actors


def update_signals(actors: list[Actor], elapsed_s: float, timing: SignalTiming | None = None) -> None:
    """Set each light to the phase the reward will charge against at this moment."""
    timing = timing or SignalTiming()
    lights = [a for a in actors if a.kind == CLASS_TRAFFIC_LIGHT]
    if not lights:
        return
    ids = np.array([a.node_id for a in lights], dtype=np.int64)
    phases = signal_phase_numpy(ids, np.full(ids.shape, float(elapsed_s)), timing)
    for actor, phase in zip(lights, phases):
        actor.state = int(phase)


@dataclass
class Traffic:
    """The moving population: other cars, motorcycles and people on foot.

    They travel the same surveyed lap the driver does, on the side the survey
    says traffic keeps to, at a share of the posted limit. Nothing about them
    is random per step: one seed gives one traffic pattern, every episode.
    """

    actors: list[Actor] = field(default_factory=list)
    cfg: SceneConfig = field(default_factory=SceneConfig)

    @classmethod
    def populate(
        cls,
        centerline: np.ndarray,
        heights: np.ndarray,
        profile: LegalProfile,
        control_points: tuple[ControlPoint, ...],
        vehicles: int,
        motorcycles: int,
        pedestrians: int,
        seed: int = 0,
        cfg: SceneConfig | None = None,
    ) -> "Traffic":
        cfg = cfg or SceneConfig()
        rng = np.random.default_rng(seed)
        half = lane_halfwidth(profile, cfg)
        n = centerline.shape[0]
        side = -1.0 if (profile.driving_side or "left") == "left" else 1.0
        limit = np.where(np.isfinite(profile.limit_mps), profile.limit_mps, cfg.lane_width_m * 4.0)
        tangent, normal = frame_at(np.asarray(centerline, dtype=np.float64))
        actors: list[Actor] = []

        def travelling(kind: str, size: np.ndarray, lateral: float, pace: float) -> Actor:
            progress = float(rng.uniform())
            i = min(int(progress * n), n - 1)
            base = centerline[i] + normal[i] * side * lateral
            return Actor(
                kind=kind,
                pos=np.array([base[0], base[1], heights[i]], dtype=np.float64),
                heading=float(np.arctan2(tangent[i, 1], tangent[i, 0])),
                size=size,
                colour=rng.uniform(0.15, 0.85, size=3).astype(np.float32),
                progress=progress,
                speed_mps=float(limit[i] * pace),
            )

        for _ in range(vehicles):
            lane = half.mean() * rng.uniform(0.25, 0.6)
            actors.append(
                travelling(
                    CLASS_CAR,
                    np.array([cfg.car_length_m, cfg.car_width_m, cfg.car_height_m]),
                    lane,
                    float(rng.uniform(0.6, 0.95)),
                )
            )
        for _ in range(motorcycles):
            lane = half.mean() * rng.uniform(0.4, 0.8)
            actors.append(
                travelling(
                    CLASS_MOTORCYCLE,
                    np.array([cfg.motorcycle_length_m, cfg.motorcycle_width_m, cfg.motorcycle_height_m]),
                    lane,
                    float(rng.uniform(0.7, 1.05)),
                )
            )

        crossings = [p for p in control_points if "pedestrian_crossing" in p.rules and p.progress >= 0.0]
        for index in range(pedestrians):
            if crossings:
                point = crossings[index % len(crossings)]
                progress = point.progress
            else:
                progress = float(rng.uniform())
            i = min(int(progress * n), n - 1)
            across = float(rng.uniform(-1.0, 1.0)) * half[i]
            base = centerline[i] + normal[i] * across
            actors.append(
                Actor(
                    kind=CLASS_PERSON,
                    pos=np.array([base[0], base[1], heights[i]], dtype=np.float64),
                    heading=float(np.arctan2(normal[i, 1], normal[i, 0])),
                    size=np.array([cfg.person_width_m, cfg.person_width_m, cfg.person_height_m]),
                    colour=rng.uniform(0.2, 0.8, size=3).astype(np.float32),
                    progress=progress,
                    speed_mps=float(rng.uniform(0.8, 1.6)),
                )
            )
        return cls(actors=actors, cfg=cfg)

    def step(
        self,
        dt_s: float,
        centerline: np.ndarray,
        heights: np.ndarray,
        profile: LegalProfile,
    ) -> None:
        """Advance the travelling actors along the lap."""
        n = centerline.shape[0]
        spacing = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())
        lap_m = spacing * n
        half = lane_halfwidth(profile, self.cfg)
        tangent, normal = frame_at(np.asarray(centerline, dtype=np.float64))
        side = -1.0 if (profile.driving_side or "left") == "left" else 1.0
        for actor in self.actors:
            if actor.speed_mps <= 0.0 or actor.progress < 0.0:
                continue
            if actor.kind == CLASS_PERSON:
                continue  # people stand at their crossing until actors get behaviour
            actor.progress = (actor.progress + actor.speed_mps * dt_s / lap_m) % 1.0
            i = min(int(actor.progress * n), n - 1)
            lateral = half[i] * 0.45
            base = centerline[i] + normal[i] * side * lateral
            actor.pos = np.array([base[0], base[1], heights[i]], dtype=np.float64)
            actor.heading = float(np.arctan2(tangent[i, 1], tangent[i, 0]))

    def visible_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for actor in self.actors:
            counts[actor.kind] = counts.get(actor.kind, 0) + 1
        return counts
