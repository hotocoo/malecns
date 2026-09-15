"""Real terrain under the circuit: a surveyed elevation grid and what the viewer derives from it.

`fetch_terrain.py` stores the survey (Copernicus EU-DEM or Terrarium tiles) as
a regular lon/lat grid. This module samples it in the track frame and builds:

  * the road's height profile along the centerline (surface height, smoothed
    over the survey's cell size; inside a mapped tunnel the road runs on a
    straight grade between the two portals, since a survey sees the hill above,
    not the road below),
  * a heightfield for the ground around the circuit, carved flat under the
    open road and dropped to the seabed under water,
  * the ground height each building stands on.

Nothing is invented: every height is a survey sample or an interpolation
between two of them. The physics stays planar; heights are for the eye.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from car_env import GeoProjection

ROAD_BED_M = 0.05  # the carved ground sits this far under the road ribbon
G = 9.81


@dataclass(frozen=True)
class TerrainRule:
    """How a survey becomes a road profile; the viewer passes its SceneryLoadConfig values, the physics uses these defaults."""

    stride: int = 2  # tunnel spans are expressed on the page's strided centerline
    smooth_m: float = 30.0
    min_height_m: float = 1.2
    portal_probe_m: float = 40.0
    tunnel_reach_m: float = 14.0
    tunnel_aligned_fraction: float = 0.85
    tunnel_min_length_m: float = 100.0
    tunnel_gap_samples: int = 40
    tunnel_min_span_m: float = 60.0
    road_classes: tuple[str, ...] = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential")


def _length(pts: np.ndarray) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _aligned(pts: np.ndarray, centerline: np.ndarray, reach_m: float, min_fraction: float = 0.85) -> bool:
    d = np.sqrt(((pts[:, None, :] - centerline[None, ::2, :]) ** 2).sum(-1)).min(1)
    return bool((d < reach_m).mean() >= min_fraction)


def tunnel_spans(
    centerline: np.ndarray,
    tunnel_lines: list[np.ndarray],
    reach_m: float,
    stride: int,
    gap_samples: int = 40,
    min_span_m: float = 60.0,
) -> list[list[int]]:
    """Index ranges of the (strided) centerline that run inside a mapped road tunnel.

    A centerline sample counts as in-tunnel when a tunnel-way vertex lies
    within `reach_m`; short gaps are bridged so one tunnel is one span.
    """
    if not tunnel_lines:
        return []
    verts = np.concatenate(tunnel_lines)
    n = len(centerline)
    inside = np.zeros(n, dtype=bool)
    chunk = 2048
    for start in range(0, n, chunk):
        block = centerline[start : start + chunk]
        d = np.sqrt(((block[:, None, :] - verts[None, :, :]) ** 2).sum(-1)).min(1)
        inside[start : start + chunk] = d < reach_m
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return []
    spans: list[list[int]] = []
    s0 = prev = int(idx[0])
    for i in idx[1:]:
        if i - prev > gap_samples:
            spans.append([s0, prev])
            s0 = int(i)
        prev = int(i)
    spans.append([s0, prev])
    # only spans long enough to be the circuit's tunnel (Monaco's is ~ 300 m)
    spacing = float(np.hypot(*np.diff(centerline, axis=0).T).mean())
    spans = [s for s in spans if (s[1] - s[0]) * spacing > min_span_m]
    return [[a // stride, b // stride] for a, b in spans]


def circuit_tunnels(scenery_path: Path, proj: GeoProjection, centerline: np.ndarray, rule: TerrainRule) -> list[np.ndarray]:
    """Projected OSM tunnel ways that run along the circuit (not footways, ramps or roads passing under)."""
    if not scenery_path.exists():
        return []
    data = json.loads(scenery_path.read_text())
    out = []
    for f in data["features"]:
        if f["properties"].get("kind") != "tunnel":
            continue
        pts = proj.project(np.asarray(f["geometry"]["coordinates"])[:, :2])
        if (
            f["properties"].get("highway") in rule.road_classes
            and _aligned(pts, centerline, rule.tunnel_reach_m, rule.tunnel_aligned_fraction)
            and _length(pts) >= rule.tunnel_min_length_m
        ):
            out.append(pts)
    return out


def road_profile_for_circuit(geojson: str | Path, centerline: np.ndarray, proj: GeoProjection, rule: TerrainRule = TerrainRule()) -> np.ndarray | None:
    """Road heights for a circuit from the survey beside its GeoJSON (`<stem>_dem.json`), or None without one.

    The circuit's tunnel comes from `<stem>_scenery.geojson` when present. This
    is the one profile the physics (grade), the viewer and the page share.
    """
    geojson = Path(geojson)
    dem_path = geojson.with_name(geojson.stem + "_dem.json")
    if not dem_path.exists():
        return None
    terrain = Terrain.load(dem_path)
    tunnels = circuit_tunnels(geojson.with_name(geojson.stem + "_scenery.geojson"), proj, centerline, rule)
    spans = tunnel_spans(centerline, tunnels, rule.tunnel_reach_m, rule.stride, rule.tunnel_gap_samples, rule.tunnel_min_span_m)
    return road_profile(centerline, proj, terrain, spans, rule.stride, rule.smooth_m, rule.min_height_m, rule.portal_probe_m)


def grade_of(heights: np.ndarray, centerline: np.ndarray) -> np.ndarray:
    """Slope dz/ds along a closed centerline (central difference, wraps)."""
    spacing = float(np.hypot(*np.diff(centerline, axis=0).T).mean())
    return (np.roll(heights, -1) - np.roll(heights, 1)) / (2.0 * max(spacing, 1e-9))


@dataclass(frozen=True)
class Terrain:
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    heights: np.ndarray  # (rows, cols), metres above sea level, row 0 = south
    source: str = ""
    attribution: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Terrain":
        data = json.loads(Path(path).read_text())
        rows, cols = int(data["rows"]), int(data["cols"])
        heights = np.asarray(data["heights"], dtype=np.float64).reshape(rows, cols)
        if not np.isfinite(heights).all():
            raise ValueError(f"{path}: terrain grid has non-finite cells")
        return cls(
            lon0=float(data["lon0"]),
            lat0=float(data["lat0"]),
            dlon=float(data["dlon"]),
            dlat=float(data["dlat"]),
            heights=heights,
            source=str(data.get("source", "")),
            attribution=str(data.get("attribution", "")),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    def sample_lonlat(self, lonlat: np.ndarray) -> np.ndarray:
        """Bilinear height at (lon, lat) pairs; clamped to the grid's edge outside it."""
        ll = np.asarray(lonlat, dtype=np.float64).reshape(-1, 2)
        rows, cols = self.heights.shape
        fx = np.clip((ll[:, 0] - self.lon0) / self.dlon, 0.0, cols - 1.000001)
        fy = np.clip((ll[:, 1] - self.lat0) / self.dlat, 0.0, rows - 1.000001)
        return bilinear(self.heights, fx, fy)

    def sample_xy(self, xy: np.ndarray, proj: GeoProjection) -> np.ndarray:
        """Height at track-frame metres."""
        return self.sample_lonlat(proj.unproject(xy))


def bilinear(grid: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Bilinear lookup of `grid[row=fy, col=fx]` for fractional, in-range indices."""
    j0 = np.floor(fx).astype(int)
    i0 = np.floor(fy).astype(int)
    j1 = np.minimum(j0 + 1, grid.shape[1] - 1)
    i1 = np.minimum(i0 + 1, grid.shape[0] - 1)
    tx = fx - j0
    ty = fy - i0
    return (
        grid[i0, j0] * (1 - tx) * (1 - ty)
        + grid[i0, j1] * tx * (1 - ty)
        + grid[i1, j0] * (1 - tx) * ty
        + grid[i1, j1] * tx * ty
    )


def smooth_circular(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing of a closed (periodic) 1-D signal."""
    n = len(values)
    if sigma_samples <= 0 or n < 3:
        return values.copy()
    half = int(np.ceil(3 * sigma_samples))
    k = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (k / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([values[-half:], values, values[:half]])
    return np.convolve(padded, kernel, mode="valid")


def road_profile(
    centerline: np.ndarray,
    proj: GeoProjection,
    terrain: Terrain,
    tunnel_spans: list[list[int]],
    stride: int,
    smooth_m: float,
    min_height_m: float,
    portal_probe_m: float = 0.0,
) -> np.ndarray:
    """Road height along the (full-resolution) centerline.

    Open road follows the surveyed surface, smoothed over `smooth_m` so one
    coarse cell cannot put a step in the road. A survey sees the hill over a
    tunnel and, within a cell or so of each mouth, a mix of hill and road; so
    from `portal_probe_m` before the entrance to `portal_probe_m` after the
    exit the road is a straight grade between the two surveyed road heights
    at those probe points. Nothing is below `min_height_m` (the quay top: a
    survey cell straddling the waterline reads sea level, the road on it does not).
    """
    n = len(centerline)
    spacing = float(np.hypot(*np.diff(centerline, axis=0).T).mean())
    surface = terrain.sample_xy(centerline, proj)
    z = smooth_circular(surface, smooth_m / max(spacing, 1e-6))
    z = np.maximum(z, min_height_m)
    probe = int(round(portal_probe_m / max(spacing, 1e-6)))
    for a_s, b_s in tunnel_spans:
        a = max(0, a_s * stride - probe)
        b = min(n - 1, (b_s + 1) * stride - 1 + probe)
        if b <= a:
            continue
        za = z[a]
        zb = z[b]
        t = np.linspace(0.0, 1.0, b - a + 1)
        z[a : b + 1] = za + (zb - za) * t
    return z


def tunnel_mask(n: int, tunnel_spans: list[list[int]], stride: int) -> np.ndarray:
    inside = np.zeros(n, dtype=bool)
    for a, b in tunnel_spans:
        inside[max(0, a * stride) : min(n, (b + 1) * stride)] = True
    return inside


def nearest_on_centerline(points: np.ndarray, centerline: np.ndarray, sub: int = 2, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """(distance, index into the full centerline) of the nearest centerline sample per point."""
    ref = centerline[::sub]
    dist = np.empty(len(points))
    idx = np.empty(len(points), dtype=int)
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        d2 = ((block[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        k = d2.argmin(1)
        dist[start : start + chunk] = np.sqrt(d2[np.arange(len(block)), k])
        idx[start : start + chunk] = k * sub
    return dist, idx


def lowest_road_within(points: np.ndarray, centerline: np.ndarray, road_z: np.ndarray, reach: float, sub: int = 2, chunk: int = 4096) -> np.ndarray:
    """Lowest road height among centerline samples within `reach` of each point (NaN when none).

    Where two stretches of road run close (a hairpin's legs, a steep grade) the
    ground between them must sit under the lower one, or it pokes through it.
    """
    ref = centerline[::sub]
    zref = road_z[::sub]
    out = np.full(len(points), np.nan)
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        d2 = ((block[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        near = d2 < reach * reach
        z = np.where(near, zref[None, :], np.inf).min(1)
        out[start : start + chunk] = np.where(np.isfinite(z), z, np.nan)
    return out


def heightfield(
    terrain: Terrain,
    proj: GeoProjection,
    half: float,
    cell: float,
    centerline: np.ndarray,
    road_z: np.ndarray,
    halfwidth: float,
    in_tunnel: np.ndarray,
    wet_fn: Callable[[np.ndarray], np.ndarray] | None,
    land_min_m: float,
    sea_level_m: float,
    seabed_drop_m: float,
    carve_shoulder_m: float,
    carve_blend_m: float,
    tunnel_cover_min_m: float = 0.0,
) -> tuple[dict, np.ndarray]:
    """Ground heights on a square grid of `cell` metres covering [-half, half]^2 in the track frame.

    Returns the payload for the page and the land heights before the water
    drop (the ground a waterfront building actually stands on).

    Land follows the survey (never below `land_min_m`, the quay top). Under
    and beside the open road the ground is carved to the road's height and
    blends back to the survey over `carve_blend_m` beyond `carve_shoulder_m`;
    over a tunnel the survey stands (that is the hill the road goes under),
    but never less than `tunnel_cover_min_m` above the road: a tunnel has a
    roof, and whatever stands there stands on it.
    Water cells sit `seabed_drop_m` under the sea so the water plane covers them.
    """
    cols = rows = int(np.ceil(2 * half / cell)) + 1
    xs = -half + np.arange(cols) * cell
    ys = -half + np.arange(rows) * cell
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    z = terrain.sample_xy(pts, proj)
    z = np.maximum(z, land_min_m)
    dist, idx = nearest_on_centerline(pts, centerline)
    open_road = ~in_tunnel[idx]
    # the flat shoulder must cover every grid node whose cell touches the road,
    # otherwise a node just outside it lifts the ground through the asphalt
    reach = halfwidth + carve_shoulder_m + cell * 0.71
    blend = np.clip((dist - reach) / max(carve_blend_m, 1e-6), 0.0, 1.0)
    blend = blend * blend * (3 - 2 * blend)  # smoothstep
    # the road ribbon is its own mesh; the ground under it sits a hair lower,
    # and under the lowest of any road stretches within reach
    lowest = lowest_road_within(pts, centerline, road_z, reach)
    under = np.where(np.isnan(lowest), road_z[idx], np.minimum(lowest, road_z[idx]))
    carved = (under - ROAD_BED_M) * (1 - blend) + z * blend
    z = np.where(open_road, carved, z)
    # Over the tunnel the ground is the tunnel's roof at least, easing back to
    # the survey beyond the shoulder. Directly over the corridor there is no
    # ground mesh at all (the tunnel shell is the structure there); the
    # headwall at each mouth closes the hill face.
    roof = road_z[idx] + tunnel_cover_min_m
    raised = np.maximum(z, roof * (1 - blend) + z * blend)
    z = np.where(open_road, z, raised)
    hole = ~open_road & (dist < reach)
    land = z.copy()
    if wet_fn is not None:
        wet = wet_fn(pts) & ~(dist < reach)
        z = np.where(wet, sea_level_m - seabed_drop_m, z)
    field = {
        "x0": float(xs[0]),
        "y0": float(ys[0]),
        "cell": float(cell),
        "cols": int(cols),
        "rows": int(rows),
        "z": [None if h else round(float(v), 2) for v, h in zip(z, hole)],
        "sea_level": float(sea_level_m),
        "min": round(float(z[~hole].min()), 2),
        "max": round(float(z[~hole].max()), 2),
        "carve_reach_m": float(reach),
        "carve_blend_m": float(carve_blend_m),
        "tunnel_cover_min_m": float(tunnel_cover_min_m),
    }
    return field, land.reshape(rows, cols)


def sample_heightfield(field: dict, xy: np.ndarray) -> np.ndarray:
    """Bilinear ground height from a `heightfield()` payload at track-frame points (NaN over the tunnel corridor)."""
    grid = np.asarray([np.nan if v is None else v for v in field["z"]], dtype=np.float64).reshape(field["rows"], field["cols"])
    p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    fx = np.clip((p[:, 0] - field["x0"]) / field["cell"], 0.0, field["cols"] - 1.000001)
    fy = np.clip((p[:, 1] - field["y0"]) / field["cell"], 0.0, field["rows"] - 1.000001)
    return bilinear(grid, fx, fy)


def building_bases(
    buildings: list[dict],
    field: dict,
    land: np.ndarray,
    centerline: np.ndarray,
    road_z: np.ndarray,
    in_tunnel: np.ndarray,
    corridor_m: float,
    tunnel_cover_min_m: float,
) -> list[float]:
    """Ground height each footprint stands on.

    The lowest land sample around the outer ring (a waterfront building stands
    on the quay, not on the seabed). A footprint over the tunnel corridor
    stands on the tunnel's roof: never less than `tunnel_cover_min_m` above
    the road beneath it.
    """
    out = []
    for b in buildings:
        ring = np.asarray(b["rings"][0], dtype=np.float64)
        fx = np.clip((ring[:, 0] - field["x0"]) / field["cell"], 0.0, field["cols"] - 1.000001)
        fy = np.clip((ring[:, 1] - field["y0"]) / field["cell"], 0.0, field["rows"] - 1.000001)
        base = float(bilinear(land, fx, fy).min())
        dist, idx = nearest_on_centerline(ring, centerline, sub=1)
        over = (dist < corridor_m) & in_tunnel[idx]
        if over.any():
            base = max(base, float(road_z[idx[over]].max()) + tunnel_cover_min_m)
        out.append(round(base, 2))
    return out
