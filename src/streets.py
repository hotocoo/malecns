"""Street geometry from the survey: carriageways that meet, kerbs, pavements, paint.

`scene.build_network` used to draw every surveyed way as its own flat ribbon
with its own pair of edge lines. Where two ways met the ribbons simply
overlapped and the edge lines ran straight across the junction, so the camera
saw a stack of grey tape rather than a street. This module builds the road as
a city engineer would draw it:

- every carriageway is buffered to the width its lane count implies and the
  buffers are joined, so a junction is one patch of asphalt with no seam;
- the edge lines are the inset boundary of that joined surface, so they stop
  where a side road opens and pick up again beyond it;
- centre and lane dividers are broken lines along each way, cut out of every
  junction zone (a node three or more carriageway arms share);
- streets that carry footways get a raised kerb and a pavement on both sides,
  cut wherever a driveway or service road crosses them;
- each surface carries a material so the renderer can texture asphalt,
  paving, kerbstone and ground differently instead of painting them flat.

Everything is world metres in the track frame; `camera.py` and the 3D page
draw the same triangles. Data (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

from roadlaw import parse_lanes, parse_oneway

# Surface materials, one float per vertex; `camera.py` textures by them.
MAT_FLAT = 0.0
MAT_ASPHALT = 1.0
MAT_PAVEMENT = 2.0
MAT_KERB = 3.0
MAT_GROUND = 4.0
MAT_WALL = 5.0
MAT_WALL_GLASS = 6.0
MAT_SHOPFRONT = 7.0
MAT_ROOF = 8.0
MAT_FASCIA = 9.0
MAT_PLINTH = 10.0

ASPHALT_COLOUR = (0.24, 0.24, 0.25)
PAINT_WHITE = (0.90, 0.90, 0.86)
PAVEMENT_COLOUR = (0.64, 0.53, 0.44)  # interlocking pavers, warm on purpose
KERB_COLOUR = (0.80, 0.71, 0.36)  # kerbs here are painted, not bare concrete

# Streets with a footway along them: a kerb and a pavement on both sides.
FOOTWAY_CLASSES = frozenset(
    {
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "residential", "unclassified",
        "living_street", "road",
    }
)
# Bare asphalt meeting the ground: car-park aisles, driveways, bus lanes.
NO_EDGE_LINE_CLASSES = frozenset({"service", "living_street"})


@dataclass(frozen=True)
class StreetConfig:
    """Dimensions of the built street that the survey does not carry."""

    kerb_width_m: float = 0.25
    kerb_height_m: float = 0.12
    pavement_width_m: float = 2.0
    junction_margin_m: float = 1.0  # how far past the widest arm the paint stops
    buffer_segments: int = 3  # arc resolution of buffers; 3 keeps vertex counts sane citywide
    lane_line_len_m: float = 3.0
    lane_line_gap_m: float = 6.0


@dataclass
class Way:
    points: np.ndarray  # (n, 2) track metres
    nodes: list[int]
    highway: str
    lanes: int
    oneway: bool
    half_m: float


@dataclass
class StreetMesh:
    """Triangles in world metres, one row per vertex."""

    vertices: np.ndarray  # (v, 3) float32
    colours: np.ndarray  # (v, 3) float32
    materials: np.ndarray  # (v,) float32

    @staticmethod
    def empty() -> "StreetMesh":
        return StreetMesh(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        )

    @staticmethod
    def concat(parts: list["StreetMesh"]) -> "StreetMesh":
        parts = [p for p in parts if p.vertices.shape[0]]
        if not parts:
            return StreetMesh.empty()
        return StreetMesh(
            np.concatenate([p.vertices for p in parts]),
            np.concatenate([p.colours for p in parts]),
            np.concatenate([p.materials for p in parts]),
        )


def _mesh(vertices: np.ndarray, colour: tuple[float, float, float], material: float) -> StreetMesh:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    return StreetMesh(
        vertices,
        np.tile(np.array(colour, dtype=np.float32), (vertices.shape[0], 1)),
        np.full(vertices.shape[0], material, dtype=np.float32),
    )


def read_ways(
    survey_path: Path | str,
    proj,
    lane_width_m: float,
    shoulder_m: float,
    default_lanes: int,
    lanes_by_class: dict[str, int],
    classes: frozenset[str],
    centre: np.ndarray | None = None,
    max_distance_m: float | None = None,
) -> list[Way]:
    """Every drivable way in the survey, projected, with the width its lanes imply.

    Ways further than `max_distance_m` from any point of `centre` (the lap)
    are dropped: a city's worth of streets the camera never reaches costs
    triangles and buys no pixels.
    """
    data = json.loads(Path(survey_path).read_text())
    coarse = None
    if centre is not None and max_distance_m is not None:
        coarse = np.asarray(centre, dtype=np.float64)[:: max(1, len(centre) // 256)]
    ways: list[Way] = []
    for way in data.get("elements", []):
        tags = way.get("tags") or {}
        highway = str(tags.get("highway"))
        if highway not in classes:
            continue
        geometry = way.get("geometry") or []
        if len(geometry) < 2:
            continue
        lonlat = np.array([[p["lon"], p["lat"]] for p in geometry], dtype=np.float64)
        points = np.asarray(proj.project(lonlat), dtype=np.float64)
        if points.shape[0] < 2:
            continue
        if coarse is not None:
            gap = np.linalg.norm(points[:, None, :] - coarse[None, :, :], axis=2).min()
            if gap > max_distance_m:
                continue
        nodes = list(way.get("nodes") or [])
        if len(nodes) != points.shape[0]:
            nodes = [-(hash((way.get("id", 0), k)) & 0x7FFFFFFF) for k in range(points.shape[0])]
        lanes = parse_lanes(tags.get("lanes"))
        if lanes is None:
            lanes = lanes_by_class.get(highway, default_lanes)
        ways.append(
            Way(
                points=points,
                nodes=nodes,
                highway=highway,
                lanes=int(lanes),
                oneway=parse_oneway(tags.get("oneway")) not in (None, 0),
                half_m=lanes * lane_width_m / 2.0 + shoulder_m,
            )
        )
    return ways


def junction_zones(ways: list[Way], cfg: StreetConfig):
    """Discs over every node three or more carriageway arms share.

    An interior node of a way has two arms of that way; an endpoint has one.
    Two ways that merely continue one another share an endpoint with degree
    two, which is not a junction and keeps its centre line.
    """
    degree: Counter = Counter()
    where: dict[int, np.ndarray] = {}
    radius: dict[int, float] = {}
    for way in ways:
        last = len(way.nodes) - 1
        for k, node in enumerate(way.nodes):
            degree[node] += 1 if k in (0, last) else 2
            where[node] = way.points[k]
            radius[node] = max(radius.get(node, 0.0), way.half_m)
    discs = [
        Point(where[node]).buffer(radius[node] + cfg.junction_margin_m, quad_segs=cfg.buffer_segments)
        for node, d in degree.items()
        if d >= 3
    ]
    return unary_union(discs) if discs else None


def _ccw(tris: np.ndarray) -> np.ndarray:
    """Every triangle wound counter-clockwise seen from above, so normals agree."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flipped = tris.copy()
    flipped[:, 1], flipped[:, 2] = tris[:, 2], tris[:, 1]
    return np.where((area < 0)[:, None, None], flipped, tris)


def triangulate(geom, z: float) -> np.ndarray:
    """A polygon (with holes) as (v, 3) float32 triangles at height `z`."""
    if geom is None or geom.is_empty:
        return np.zeros((0, 3), dtype=np.float32)
    tris = shapely.constrained_delaunay_triangles(geom)
    coords = shapely.get_coordinates(tris)
    if coords.shape[0] % 4:
        return np.zeros((0, 3), dtype=np.float32)
    flat = _ccw(coords.reshape(-1, 4, 2)[:, :3, :]).reshape(-1, 2)
    return np.concatenate([flat, np.full((flat.shape[0], 1), z)], axis=1).astype(np.float32)


def _rings(geom) -> list[np.ndarray]:
    """Every boundary ring of a (multi)polygon, as (n, 2) closed coordinate arrays."""
    if geom is None or geom.is_empty:
        return []
    rings: list[np.ndarray] = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.geom_type != "Polygon":
            continue
        rings.append(np.asarray(poly.exterior.coords, dtype=np.float64))
        rings += [np.asarray(ring.coords, dtype=np.float64) for ring in poly.interiors]
    return rings


def _lines(geom) -> list[np.ndarray]:
    """Every line of a (multi)linestring as an (n, 2) array."""
    if geom is None or geom.is_empty:
        return []
    out: list[np.ndarray] = []
    for part in getattr(geom, "geoms", [geom]):
        if part.geom_type == "LineString" and len(part.coords) >= 2:
            out.append(np.asarray(part.coords, dtype=np.float64))
        elif part.geom_type in ("MultiLineString", "GeometryCollection"):
            out += _lines(part)
    return out


def wall(ring: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """A vertical strip standing on `ring` between two heights."""
    if ring.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    a = np.concatenate([ring[:-1], np.full((ring.shape[0] - 1, 1), z0)], axis=1)
    b = np.concatenate([ring[1:], np.full((ring.shape[0] - 1, 1), z0)], axis=1)
    a1, b1 = a.copy(), b.copy()
    a1[:, 2], b1[:, 2] = z1, z1
    tris = np.stack([a, b, b1, a, b1, a1], axis=1)
    return tris.reshape(-1, 3).astype(np.float32)


def stroke(line: np.ndarray, width_m: float, z: float) -> np.ndarray:
    """A painted line of constant width along a polyline."""
    if line.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    ahead = np.diff(line, axis=0, append=line[-1:])
    ahead[-1] = line[-1] - line[-2]
    norm = np.linalg.norm(ahead, axis=1, keepdims=True).clip(min=1e-6)
    tangent = ahead / norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    left = line + normal * (width_m / 2.0)
    right = line - normal * (width_m / 2.0)
    a = np.concatenate([left, np.full((line.shape[0], 1), z)], axis=1)
    b = np.concatenate([right, np.full((line.shape[0], 1), z)], axis=1)
    tris = np.stack([a[:-1], b[:-1], b[1:], a[:-1], b[1:], a[1:]], axis=1)
    return tris.reshape(-1, 3).astype(np.float32)


def dashes(line: LineString, on_m: float, off_m: float, width_m: float, z: float) -> list[np.ndarray]:
    """Broken line along `line`: painted `on_m`, blank `off_m`."""
    out: list[np.ndarray] = []
    length = float(line.length)
    s = 0.0
    while s < length:
        piece = substring(line, s, min(s + on_m, length))
        if piece.geom_type == "LineString" and len(piece.coords) >= 2:
            out.append(stroke(np.asarray(piece.coords, dtype=np.float64), width_m, z))
        s += on_m + off_m
    return out


def _buffer(way: Way, extra_m: float, cfg: StreetConfig):
    return LineString(way.points).buffer(way.half_m + extra_m, cap_style="flat", quad_segs=cfg.buffer_segments)


def build_streets(
    ways: list[Way],
    ground_z: float,
    lane_width_m: float,
    shoulder_m: float,
    marking_width_m: float,
    cfg: StreetConfig | None = None,
) -> StreetMesh:
    """The whole street network as one textured mesh. See the module docstring."""
    cfg = cfg or StreetConfig()
    if not ways:
        return StreetMesh.empty()
    paint_z = ground_z + 0.01
    top_z = ground_z + cfg.kerb_height_m

    asphalt_polys = [_buffer(w, 0.0, cfg) for w in ways]
    carriageway = unary_union(asphalt_polys)
    parts: list[StreetMesh] = [_mesh(triangulate(carriageway, ground_z), ASPHALT_COLOUR, MAT_ASPHALT)]

    # Kerb and pavement follow the streets that have footways, and stop
    # wherever any carriageway (a driveway, a side road) crosses them.
    footway = [w for w in ways if w.highway in FOOTWAY_CLASSES]
    if footway:
        kerbed = unary_union([_buffer(w, 0.0, cfg) for w in footway])
        kerb_outer = kerbed.buffer(cfg.kerb_width_m, quad_segs=cfg.buffer_segments)
        pave_outer = kerbed.buffer(cfg.kerb_width_m + cfg.pavement_width_m, quad_segs=cfg.buffer_segments)
        kerb_top = kerb_outer.difference(carriageway)
        pavement = pave_outer.difference(kerb_outer).difference(carriageway)
        raised = pave_outer.difference(carriageway)
        parts.append(_mesh(triangulate(kerb_top, top_z), KERB_COLOUR, MAT_KERB))
        parts.append(_mesh(triangulate(pavement, top_z), PAVEMENT_COLOUR, MAT_PAVEMENT))
        faces = [wall(ring, ground_z, top_z) for ring in _rings(raised)]
        if faces:
            parts.append(_mesh(np.concatenate(faces), KERB_COLOUR, MAT_KERB))

    # Edge lines: the inset outline of the through roads, minus wherever a
    # service road or driveway opens onto them.
    lined = [w for w in ways if w.highway not in NO_EDGE_LINE_CLASSES]
    if lined:
        main = unary_union([_buffer(w, 0.0, cfg) for w in lined])
        inset = main.buffer(-shoulder_m, quad_segs=cfg.buffer_segments)
        edges = inset.boundary
        unlined = [w for w in ways if w.highway in NO_EDGE_LINE_CLASSES]
        if unlined:
            edges = edges.difference(unary_union([_buffer(w, 0.0, cfg) for w in unlined]))
        strokes = [stroke(line, marking_width_m, paint_z) for line in _lines(edges)]
        if strokes:
            parts.append(_mesh(np.concatenate(strokes), PAINT_WHITE, MAT_FLAT))

    # Centre and lane lines, broken, cut out of every junction.
    zones = junction_zones(ways, cfg)
    painted: list[np.ndarray] = []
    for way in ways:
        if way.highway in NO_EDGE_LINE_CLASSES or way.lanes < 2:
            continue
        axis = LineString(way.points)
        if way.oneway:
            offsets = [(k - way.lanes / 2.0) * lane_width_m for k in range(1, way.lanes)]
        else:
            per_side = way.lanes // 2
            offsets = [0.0] + [s * k * lane_width_m for s in (-1.0, 1.0) for k in range(1, per_side)]
        for offset in offsets:
            line = axis if abs(offset) < 1e-6 else axis.offset_curve(offset, quad_segs=cfg.buffer_segments)
            if zones is not None:
                line = line.difference(zones)
            for part in _lines(line):
                painted += dashes(LineString(part), cfg.lane_line_len_m, cfg.lane_line_gap_m, marking_width_m, paint_z + 0.002)
    if painted:
        parts.append(_mesh(np.concatenate(painted), PAINT_WHITE, MAT_FLAT))

    return StreetMesh.concat(parts)


def crossing_markings(
    centerline: np.ndarray,
    heights: np.ndarray,
    halfwidth: np.ndarray,
    control_points,
    driving_side: str | None,
    marking_width_m: float,
) -> StreetMesh:
    """Stop lines before every signal and zebra stripes at every crossing on the lap.

    Both are read from the same control points the law charges at, so the
    paint stands exactly where the rule bites.
    """
    centerline = np.asarray(centerline, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    n = centerline.shape[0]
    ahead = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangent = ahead / np.linalg.norm(ahead, axis=1, keepdims=True).clip(min=1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    side = -1.0 if (driving_side or "left") == "left" else 1.0
    spacing = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).mean())

    pieces: list[np.ndarray] = []
    for point in control_points:
        if point.progress < 0.0:
            continue
        i = min(int(point.progress * n), n - 1)
        half = float(halfwidth[i])
        z = float(heights[i]) + 0.012
        if "red_signal" in point.rules or point.tags.get("highway") == "stop":
            # A bar across the near-side half of the carriageway, a car
            # length short of the node.
            back = max(0, i - int(round(1.5 / max(spacing, 1e-6))))
            base = centerline[back]
            inner = base + normal[back] * side * marking_width_m
            outer = base + normal[back] * side * (half - marking_width_m)
            pieces.append(stroke(np.stack([inner, outer]), 0.4, z))
        if "pedestrian_crossing" in point.rules:
            # Zebra: half-metre stripes across the full width, three metres deep.
            stripes = int(2.0 * half / 1.0)
            for k in range(stripes):
                lateral = -half + 0.5 + k * 1.0
                start = centerline[i] + normal[i] * lateral - tangent[i] * 1.5
                end = centerline[i] + normal[i] * lateral + tangent[i] * 1.5
                pieces.append(stroke(np.stack([start, end]), 0.5, z))
    if not pieces:
        return StreetMesh.empty()
    return _mesh(np.concatenate(pieces), PAINT_WHITE, MAT_FLAT)
