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

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lawreward import GREEN, RED, SignalTiming, signal_phase_numpy
import streets
from roadlaw import UNKNOWN_LANES, ControlPoint, LegalProfile, parse_lanes, parse_oneway

# Classes a camera should see as road. Wider than the set a route is built on:
# a service road beside the lap is still asphalt the driver looks at.
DRIVABLE_CLASSES = frozenset(
    {
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "unclassified", "residential",
        "living_street", "service", "road", "busway",
    }
)

# Lanes to assume for a class the survey did not tag. Malaysian federal and
# state roads are dual-lane as a rule, a residential street single.
LANES_BY_CLASS = {
    "motorway": 4, "motorway_link": 2,
    "trunk": 4, "trunk_link": 2,
    "primary": 4, "primary_link": 2,
    "secondary": 2, "secondary_link": 2,
    "tertiary": 2, "tertiary_link": 2,
    "unclassified": 2, "residential": 2, "living_street": 1,
    "service": 1, "road": 2, "busway": 2,
}

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

    # What a building is given when the survey measured neither its height nor
    # its storeys. Malaysian shoplots are typically two or three storeys.
    storey_m: float = 3.2
    unknown_storeys: float = 3.0

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
    # Per-vertex material of `surface` (see `streets.MAT_*`); None means the
    # whole surface is asphalt. Markings are always flat paint.
    surface_material: np.ndarray | None = None


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
    # Free motion in world metres per second, used by actors that leave the
    # lane: someone stepping off a kerb, a ball rolling, a car turning across.
    velocity: np.ndarray | None = None
    behaviour: str = "lane"  # "lane" follows the lap, "free" integrates velocity
    hazard: str = ""  # the event that owns this actor, for telemetry
    lane_offset_m: float = 0.0
    target_speed_mps: float = 0.0
    ttl_s: float = -1.0  # seconds left before the actor is removed; -1 never


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
    paint: bool = True,
) -> RoadMesh:
    """The road surface and its painted markings.

    A two-way road gets a dashed centre line; a one-way carriageway does not,
    because there is no opposing traffic to divide it from. Both get solid
    edge lines. Which is which comes from the survey's `oneway` tag.

    With `paint=False` only the surface is built: when the whole street
    network is meshed (`build_network`) its paint already stops at junctions,
    and a second set of lines along the lap would run straight through them.
    """
    cfg = cfg or SceneConfig()
    centerline = np.asarray(centerline, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    half = lane_halfwidth(profile, cfg)
    _, normal = frame_at(centerline)

    surface = _ribbon(centerline, heights, normal, -half, half, 0.0)
    surface_colour = np.tile(np.array([0.24, 0.24, 0.25], dtype=np.float32), (surface.shape[0], 1))
    if not paint:
        empty = np.zeros((0, 3), dtype=np.float32)
        return RoadMesh(surface, surface_colour, empty, empty, half)

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


def build_network(
    survey_path,
    proj,
    cfg: SceneConfig | None = None,
    ground_z: float = 0.0,
    classes: frozenset[str] | None = None,
    centre: np.ndarray | None = None,
    max_distance_m: float | None = 1200.0,
) -> RoadMesh:
    """Every surveyed road near the lap, built as a street rather than a ribbon.

    Carriageways are joined where they meet, edge lines stop at every side
    road, centre lines are cut out of junctions, and streets with footways get
    kerbs and pavements. See `streets.build_streets` for the construction. The
    result's `surface` carries a per-vertex material so the renderer can
    texture asphalt, paving and kerbstone; the paint is in `markings`.

    Ways are drawn flat at `ground_z`; the lap's own surface is drawn over them
    by `build_road`, so the road under the car keeps its surveyed height.
    Ways further than `max_distance_m` from the lap are left out.
    """
    from streets import MAT_FLAT, MAT_ASPHALT, StreetConfig, build_streets, read_ways

    cfg = cfg or SceneConfig()
    classes = classes or DRIVABLE_CLASSES
    ways = read_ways(
        survey_path,
        proj,
        cfg.lane_width_m,
        cfg.shoulder_m,
        cfg.default_lanes,
        LANES_BY_CLASS,
        classes,
        centre=centre,
        max_distance_m=max_distance_m,
    )
    mesh = build_streets(ways, ground_z, cfg.lane_width_m, cfg.shoulder_m, cfg.marking_width_m, StreetConfig())
    if not mesh.vertices.shape[0]:
        empty = np.zeros((0, 3), dtype=np.float32)
        return RoadMesh(empty, empty, empty, empty, np.zeros(0))

    is_paint = mesh.materials == MAT_FLAT
    return RoadMesh(
        surface=mesh.vertices[~is_paint],
        surface_colour=mesh.colours[~is_paint],
        markings=mesh.vertices[is_paint],
        markings_colour=mesh.colours[is_paint],
        halfwidth_m=np.zeros(0),
        surface_material=mesh.materials[~is_paint],
    )


def build_buildings(
    path,
    proj,
    cfg: SceneConfig | None = None,
    ground_z: float = 0.0,
    max_distance_m: float = 900.0,
    centre: np.ndarray | None = None,
) -> RoadMesh:
    """Every surveyed building near the circuit, extruded to its own height.

    A street with no buildings is a road in a field, and a camera driving it
    sees only sky. OSM carries the footprint of nearly every building in a
    Malaysian city, and a height or a storey count for many of them; a building
    the survey leaves unmeasured is given `cfg.unknown_storeys` storeys, which
    is declared here rather than inferred from nothing.

    Only what stands within `max_distance_m` of the lap is built: a city's worth
    of walls that the camera can never see costs memory and buys no pixels.
    """
    cfg = cfg or SceneConfig()
    path = Path(path)
    if not path.exists():
        empty = np.zeros((0, 3), dtype=np.float32)
        return RoadMesh(empty, empty, empty, empty, np.zeros(0))

    data = json.loads(path.read_text())
    walls: list[np.ndarray] = []
    shades: list[np.ndarray] = []
    materials: list[np.ndarray] = []
    for feature in data.get("features", []):
        rings = (feature.get("geometry") or {}).get("coordinates") or []
        if not rings or len(rings[0]) < 4:
            continue
        outline = proj.project(np.array(rings[0][:-1], dtype=np.float64))
        if centre is not None:
            near = np.linalg.norm(outline[:, None, :] - centre[None, :, :], axis=2).min()
            if near > max_distance_m:
                continue
        height = float((feature.get("properties") or {}).get("height") or 0.0)
        if height <= 0.0:
            height = cfg.unknown_storeys * cfg.storey_m
        wall, shade, material = _extrude(outline, ground_z, height, cfg)
        if wall.shape[0]:
            walls.append(wall)
            shades.append(shade)
            materials.append(material)

    if not walls:
        empty = np.zeros((0, 3), dtype=np.float32)
        return RoadMesh(empty, empty, empty, empty, np.zeros(0))
    surface = np.concatenate(walls)
    return RoadMesh(
        surface=surface,
        surface_colour=np.concatenate(shades),
        markings=np.zeros((0, 3), dtype=np.float32),
        markings_colour=np.zeros((0, 3), dtype=np.float32),
        halfwidth_m=np.zeros(0),
        surface_material=np.concatenate(materials),
    )


# Facade families. A Malaysian street is shoplots at the bottom of the scale,
# commercial blocks in the middle and glass towers above; drawing all three as
# one grey box is what made every street look the same.
FACADE_SHOPLOT = 0
FACADE_BLOCK = 1
FACADE_TOWER = 2

SHOPLOT_MAX_M = 16.0
TOWER_MIN_M = 45.0
SHOPFRONT_M = 4.2
PARAPET_M = 0.9

# Painted render, as shoplots are kept: repainted often, rarely grey.
# Every tone here is kept off neutral grey (saturation over about 0.2), because
# `perceive.road_profile` measures the road as the grey pixels below the
# horizon: a grey wall at eye height would read as more road.
SHOPLOT_PAINT = (
    (0.88, 0.83, 0.70),
    (0.64, 0.84, 0.70),
    (0.89, 0.76, 0.67),
    (0.62, 0.74, 0.86),
    (0.90, 0.85, 0.61),
    (0.83, 0.71, 0.63),
    (0.58, 0.74, 0.64),
)
# Tile and fair-faced concrete, the middle of the market.
BLOCK_PAINT = (
    (0.78, 0.72, 0.61),
    (0.70, 0.62, 0.52),
    (0.80, 0.74, 0.62),
    (0.62, 0.55, 0.45),
    (0.72, 0.60, 0.52),
)
# Curtain wall, read as the glass it is rather than as painted wall.
TOWER_GLASS = (
    (0.42, 0.50, 0.57),
    (0.35, 0.44, 0.52),
    (0.47, 0.53, 0.55),
    (0.31, 0.40, 0.47),
)
CONCRETE = (0.66, 0.60, 0.50)


def _facade_seed(outline: np.ndarray) -> float:
    """A number steady for one footprint, so a building keeps its own look."""
    x, y = float(outline[0, 0]), float(outline[0, 1])
    return abs(x * 3.7 + y * 11.3 + float(outline[1, 0]) * 5.1) * 0.618 % 1.0


def _facade_kind(height: float) -> int:
    if height <= SHOPLOT_MAX_M:
        return FACADE_SHOPLOT
    return FACADE_TOWER if height >= TOWER_MIN_M else FACADE_BLOCK


def _facade_paint(kind: int, seed: float) -> np.ndarray:
    family = (SHOPLOT_PAINT, BLOCK_PAINT, TOWER_GLASS)[kind]
    return np.array(family[int(seed * 977.0) % len(family)], dtype=np.float32)


def _band(outline: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """The side walls of a footprint between two heights."""
    lower = np.concatenate([outline, np.full((outline.shape[0], 1), z0)], axis=1)
    upper = lower.copy()
    upper[:, 2] = z1
    nxt = np.roll(np.arange(outline.shape[0]), -1)
    return np.stack([lower, lower[nxt], upper[nxt], lower, upper[nxt], upper], axis=1).reshape(-1, 3)


def _cap(outline: np.ndarray, z: float) -> np.ndarray:
    """A flat lid over a footprint, fanned from its first vertex."""
    top = np.concatenate([outline, np.full((outline.shape[0], 1), z)], axis=1)
    fan = np.stack([np.repeat(top[:1], outline.shape[0] - 2, axis=0), top[1:-1], top[2:]], axis=1)
    return fan.reshape(-1, 3)


def _roof_plant(outline: np.ndarray, roof_z: float, seed: float) -> np.ndarray:
    """Water tanks and air handling on the roof, which break the skyline."""
    span = outline.max(axis=0) - outline.min(axis=0)
    footprint = float(min(span[0], span[1]))
    if footprint < 8.0:
        return np.zeros((0, 3), dtype=np.float32)
    centre = outline.mean(axis=0)
    reach = footprint * 0.22
    pieces: list[np.ndarray] = []
    for k in range(1 + int(seed * 31.0) % 3):
        phase = seed * 7.0 + k * 1.7
        offset = np.array([np.cos(phase), np.sin(phase)]) * reach
        half = 1.2 + 1.4 * ((phase * 0.37) % 1.0)
        tall = 1.4 + 1.8 * ((phase * 0.71) % 1.0)
        corner = centre + offset
        square = np.array(
            [
                [corner[0] - half, corner[1] - half],
                [corner[0] + half, corner[1] - half],
                [corner[0] + half, corner[1] + half],
                [corner[0] - half, corner[1] + half],
            ]
        )
        pieces.append(_band(square, roof_z, roof_z + tall))
        pieces.append(_cap(square, roof_z + tall))
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros((0, 3), dtype=np.float32)


def _extrude(outline: np.ndarray, base_z: float, height: float, cfg: SceneConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One footprint as a shopfront, a facade, a parapet and a roof.

    The building is built in bands rather than one extrusion so the ground
    floor can be glazed like the shop it is, the roof can carry a parapet and
    plant, and each band can name its own material for the renderer to texture.
    """
    if outline.shape[0] < 3:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.float32)

    seed = _facade_seed(outline)
    kind = _facade_kind(height)
    roof_z = base_z + height
    plinth_top = base_z + min(SHOPFRONT_M, height * 0.5)
    paint = _facade_paint(kind, seed)
    concrete = np.array(CONCRETE, dtype=np.float32) * (0.9 + 0.2 * seed)

    # The ground floor is split in three bands rather than textured by height,
    # because the renderer only knows a vertex's absolute z: a building whose
    # base sits 30 m up the terrain would otherwise get its fascia halfway up
    # the glazing.
    tile_top = min(base_z + 0.45, plinth_top)
    fascia_base = max(plinth_top - 0.8, tile_top)
    bands = [
        (_band(outline, base_z, tile_top), paint * 0.66, streets.MAT_PLINTH),
        (_band(outline, tile_top, fascia_base), paint * 0.92, streets.MAT_SHOPFRONT),
        (_band(outline, fascia_base, plinth_top), paint * 0.78, streets.MAT_FASCIA),
        (
            _band(outline, plinth_top, roof_z),
            paint,
            streets.MAT_WALL_GLASS if kind == FACADE_TOWER else streets.MAT_WALL,
        ),
        (_band(outline, roof_z, roof_z + PARAPET_M), concrete, streets.MAT_ROOF),
        (_cap(outline, roof_z), concrete * 0.88, streets.MAT_ROOF),
        (_roof_plant(outline, roof_z, seed), concrete * 0.82, streets.MAT_ROOF),
    ]

    surface = np.concatenate([band for band, _, _ in bands if band.shape[0]]).astype(np.float32)
    colours = np.concatenate(
        [np.tile(tint, (band.shape[0], 1)) for band, tint, _ in bands if band.shape[0]]
    ).astype(np.float32)
    materials = np.concatenate(
        [np.full(band.shape[0], material, dtype=np.float32) for band, _, material in bands if band.shape[0]]
    )
    return surface, colours, materials


def _dashed(
    points: np.ndarray, heights: np.ndarray, inner: float, outer: float, lift: float, cfg: SceneConfig
) -> np.ndarray:
    """A broken line down a way: painted for `dash_len_m`, blank for `dash_gap_m`."""
    if points.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    run = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    period = max(cfg.dash_len_m + cfg.dash_gap_m, 1e-3)
    painted = (run % period) < cfg.dash_len_m
    full = _open_ribbon(points, heights, np.full(points.shape[0], inner), np.full(points.shape[0], outer), lift)
    if not full.shape[0]:
        return full
    per_segment = full.shape[0] // max(points.shape[0] - 1, 1)
    keep = np.repeat(painted[:-1], per_segment)
    return full[: keep.shape[0]][keep]


def _open_ribbon(
    points: np.ndarray, heights: np.ndarray, inner: np.ndarray, outer: np.ndarray, lift: float
) -> np.ndarray:
    """`_ribbon` for a way that does not close on itself.

    The lap wraps, so its last sample joins its first; an ordinary street ends,
    and joining its ends would draw a road across the city.
    """
    ahead = np.diff(points, axis=0, append=points[-1:])
    ahead[-1] = points[-1] - points[-2]
    norm = np.linalg.norm(ahead, axis=1, keepdims=True).clip(min=1e-6)
    tangent = ahead / norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    left = points + normal * outer[:, None]
    right = points + normal * inner[:, None]
    z = heights + lift
    a = np.concatenate([left, z[:, None]], axis=1)
    b = np.concatenate([right, z[:, None]], axis=1)
    tris = np.stack([a[:-1], b[:-1], b[1:], a[:-1], b[1:], a[1:]], axis=1)
    return tris.reshape(-1, 3).astype(np.float32)


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
            offset = side * lateral
            base = centerline[i] + normal[i] * offset
            speed = float(limit[i] * pace)
            return Actor(
                kind=kind,
                pos=np.array([base[0], base[1], heights[i]], dtype=np.float64),
                heading=float(np.arctan2(tangent[i, 1], tangent[i, 0])),
                size=size,
                colour=rng.uniform(0.15, 0.85, size=3).astype(np.float32),
                progress=progress,
                speed_mps=speed,
                # Without these the director would pull every vehicle onto the
                # centreline and stack the whole population in one lane.
                lane_offset_m=offset,
                target_speed_mps=speed,
            )

        for _ in range(vehicles):
            lane = half.mean() * rng.uniform(0.3, 0.75)
            actors.append(
                travelling(
                    CLASS_CAR,
                    np.array([cfg.car_length_m, cfg.car_width_m, cfg.car_height_m]),
                    lane,
                    float(rng.uniform(0.6, 0.95)),
                )
            )
        for _ in range(motorcycles):
            # Two-wheelers sit wide of the car line, which is where they ride.
            lane = half.mean() * rng.uniform(0.75, 1.15)
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
            across = float(rng.uniform(0.9, 1.5)) * half[i] * (1.0 if rng.uniform() < 0.5 else -1.0)
            base = centerline[i] + normal[i] * across
            actors.append(
                Actor(
                    kind=CLASS_PERSON,
                    pos=np.array([base[0], base[1], heights[i]], dtype=np.float64),
                    heading=float(np.arctan2(normal[i, 1], normal[i, 0])),
                    size=np.array([cfg.person_width_m, cfg.person_width_m, cfg.person_height_m]),
                    colour=rng.uniform(0.2, 0.8, size=3).astype(np.float32),
                    # People stand where they are put, beside the road or at a
                    # crossing; they do not travel down the lane like a vehicle.
                    progress=-1.0,
                    behaviour="free",
                    velocity=np.zeros(3),
                    speed_mps=0.0,
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
