"""Live browser viewer: watch the connectome drive, neuron by neuron.

Two modes, both rendering the same page:

  live    (default)  runs one car under the current checkpoint in a background
                     thread and streams every control step to the browser over
                     Server-Sent Events. The checkpoint is re-read whenever it
                     changes on disk, so the car improves as training improves.
  replay  --replay run.npz   plays a run captured by `evaluate.py --record`
                     with the full spike mask per step: the brain is not
                     simulated, so this costs the trainer nothing.

  python3 src/viewer.py --checkpoint checkpoints/best.pt
  open http://127.0.0.1:8765

Only the standard library is used for serving; there is no build step and no
JavaScript dependency. Real scenery (buildings, the tunnel, coastline) comes
from `data/tracks/*_scenery.geojson` (OpenStreetMap, ODbL) projected with the
same transform as the circuit centerline, so it lands where it stands.
"""

from __future__ import annotations

import argparse
import base64
import json
from urllib.parse import parse_qs, urlsplit
import queue
import threading
import time
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device, synchronize
from car_env import DONE_NAMES, CarConfig, CarEnv, GeoProjection, Track, build_centerline, load_geojson_centerline, monaco_config
from viewer_config import RasterConfig, SceneryLoadConfig, ViewerConfig, load_viewer_config

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Default raster sample per role; the live value comes from ViewerConfig.raster.
RASTER_QUOTA = RasterConfig().quota

# Every scalar the live frame carries, with how to show it. The page renders
# its telemetry panels from this list, so a new field here appears on screen
# with no JavaScript change. group: fly (what it senses and commands), body
# (vehicle state), reward, brain (network activity), sim (solver/link).
FRAME_SCHEMA: list[dict] = [
    {"key": "sensory_hz", "label": "eye drive per ray", "unit": "Hz", "group": "fly", "kind": "rays"},
    {"key": "speed_hz", "label": "speed sense (ascending)", "unit": "Hz", "group": "fly", "digits": 1},
    {"key": "motor_steer", "label": "steer pre-activation", "unit": "", "group": "fly", "digits": 2, "bipolar": True},
    {"key": "motor_pedal", "label": "pedal pre-activation", "unit": "", "group": "fly", "digits": 2, "bipolar": True},
    {"key": "steer", "label": "steer command", "unit": "", "group": "fly", "digits": 2, "bipolar": True},
    {"key": "steer_actual", "label": "steering angle (actuator)", "unit": "of lock", "group": "fly", "digits": 2, "bipolar": True},
    {"key": "pedal", "label": "pedal command", "unit": "", "group": "fly", "digits": 2, "bipolar": True},
    {"key": "speed", "label": "speed", "unit": "km/h", "group": "body", "digits": 0, "scale": "kmh"},
    {"key": "lat_g", "label": "lateral load", "unit": "g", "group": "body", "digits": 2},
    {"key": "clearance", "label": "body clearance to barrier", "unit": "m", "group": "body", "digits": 2},
    {"key": "laps", "label": "lap progress", "unit": "laps", "group": "body", "digits": 3},
    {"key": "heading", "label": "heading", "unit": "rad", "group": "body", "digits": 2},
    {"key": "reward", "label": "reward (step)", "unit": "", "group": "reward", "digits": 3, "bipolar": True},
    {"key": "reward_progress", "label": "progress term", "unit": "", "group": "reward", "digits": 3, "bipolar": True},
    {"key": "reward_bonus", "label": "lap bonus", "unit": "", "group": "reward", "digits": 1},
    {"key": "reward_wall", "label": "wall term", "unit": "", "group": "reward", "digits": 3, "bipolar": True},
    {"key": "reward_time", "label": "time tax", "unit": "", "group": "reward", "digits": 3, "bipolar": True},
    {"key": "episode_return", "label": "episode return", "unit": "", "group": "reward", "digits": 2, "bipolar": True},
    {"key": "spiking", "label": "neurons spiking this step", "unit": "", "group": "brain", "digits": 0},
    {"key": "brain_ms", "label": "brain solve", "unit": "ms/step", "group": "sim", "digits": 1},
    {"key": "env_ms", "label": "vehicle + lidar", "unit": "ms/step", "group": "sim", "digits": 2},
    {"key": "headroom", "label": "headroom vs real time", "unit": "x", "group": "sim", "digits": 2},
    {"key": "fps", "label": "frames streamed", "unit": "/s", "group": "sim", "digits": 1},
    {"key": "step", "label": "step in episode", "unit": "", "group": "sim", "digits": 0},
    {"key": "episode", "label": "episode", "unit": "", "group": "sim", "digits": 0},
    {"key": "uptime", "label": "viewer uptime", "unit": "s", "group": "sim", "digits": 0},
]
CONTENT_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".glb": "model/gltf-binary",
    ".wasm": "application/wasm",
    ".hdr": "application/octet-stream",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}


def stratified_sample(
    roles: dict[str, list[int]], seed: int = 0, quota: dict[str, int] | None = None
) -> tuple[np.ndarray, list[dict]]:
    """Fixed sample of neuron indices, grouped by role for raster banding."""
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    bands: list[dict] = []
    offset = 0
    for role, quota in (quota or RASTER_QUOTA).items():
        pool = np.asarray(roles.get(role, []), dtype=np.int64)
        if pool.size == 0:
            continue
        take = min(quota, pool.size)
        chosen = np.sort(rng.choice(pool, size=take, replace=False))
        picked.append(chosen)
        bands.append({"role": role, "start": offset, "count": take, "total": int(pool.size)})
        offset += take
    return np.concatenate(picked), bands


def tunnel_spans(
    centerline: np.ndarray,
    tunnel_lines: list[np.ndarray],
    reach_m: float,
    stride: int,
    gap_samples: int = SceneryLoadConfig().tunnel_gap_samples,
    min_span_m: float = SceneryLoadConfig().tunnel_min_span_m,
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


def _length(pts: np.ndarray) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _aligned(pts: np.ndarray, centerline: np.ndarray, reach_m: float, min_fraction: float = 0.85) -> bool:
    d = np.sqrt(((pts[:, None, :] - centerline[None, ::2, :]) ** 2).sum(-1)).min(1)
    return bool((d < reach_m).mean() >= min_fraction)


def load_scenery(
    path: Path,
    proj: GeoProjection,
    centerline: np.ndarray,
    halfwidth: float,
    stride: int,
    load: SceneryLoadConfig | None = None,
) -> dict:
    """Project OSM features into the track frame; drop buildings standing on open road."""
    load = load or SceneryLoadConfig()
    data = json.loads(path.read_text())
    buildings, tunnels, lines = [], [], []
    for f in data["features"]:
        kind = f["properties"].get("kind")
        geom = f["geometry"]
        if kind == "building" and geom["type"] == "Polygon":
            rings = [proj.project(np.asarray(r)[:, :2]) for r in geom["coordinates"]]
            buildings.append(
                {
                    "rings": [r.round(2).tolist() for r in rings],
                    "height": float(f["properties"].get("height") or 0.0),
                    "name": f["properties"].get("name"),
                    "roof": f["properties"].get("roof"),
                    "colour": f["properties"].get("colour"),
                }
            )
        elif kind == "tunnel":
            pts = proj.project(np.asarray(geom["coordinates"])[:, :2])
            # Only road tunnels that run *along* the circuit count: footways,
            # car-park ramps and roads passing underneath are not the track.
            if (
                f["properties"].get("highway") in load.road_classes
                and _aligned(pts, centerline, load.tunnel_reach_m, load.tunnel_aligned_fraction)
                and _length(pts) >= load.tunnel_min_length_m
            ):
                tunnels.append(pts)
        elif geom["type"] == "LineString":
            lines.append({"kind": kind, "points": proj.project(np.asarray(geom["coordinates"])[:, :2]).round(2).tolist()})
    spans = tunnel_spans(
        centerline, tunnels, reach_m=load.tunnel_reach_m, stride=stride,
        gap_samples=load.tunnel_gap_samples, min_span_m=load.tunnel_min_span_m,
    )
    coast = [np.asarray(line["points"], dtype=np.float64) for line in lines if line["kind"] == "coastline"]
    ground_half = float(np.abs(centerline).max() + halfwidth + 2.0) * 1.02 * load.ground_margin_factor
    water = water_and_land(
        coast, centerline, halfwidth, ground_half,
        fine_m=load.water_fine_m, coarse_m=load.water_coarse_m, pad_m=load.water_pad_m,
        level_m=load.water_level_m, road_keep_m=load.water_road_keep_m,
    )
    # Buildings whose footprint touches the open (non-tunnel) road are survey
    # mismatches: skip them. Buildings above the tunnel are real and stay.
    in_tunnel = np.zeros(len(centerline), dtype=bool)
    for a, b in spans:
        in_tunnel[a * stride : (b + 1) * stride] = True
    open_road = centerline[~in_tunnel]
    kept = []
    for b in buildings:
        outer = np.asarray(b["rings"][0])
        d = np.sqrt(((outer[:, None, :] - open_road[None, ::4, :]) ** 2).sum(-1)).min()
        if d > halfwidth + load.building_road_clearance_m:
            kept.append(b)
    return {
        "attribution": data.get("attribution", "(c) OpenStreetMap contributors, ODbL"),
        "buildings": kept,
        "dropped_on_road": len(buildings) - len(kept),
        "tunnel_spans": spans,
        "tunnels": [t.round(2).tolist() for t in tunnels],
        "lines": lines,
        "water": water,
    }


def coast_segments(lines: list[np.ndarray]) -> np.ndarray:
    """(m, 2, 2) segments from coastline polylines (OSM: land on the left, water on the right)."""
    segs = [np.stack([pts[:-1], pts[1:]], axis=1) for pts in lines if len(pts) >= 2]
    return np.concatenate(segs) if segs else np.zeros((0, 2, 2))


def water_side(points: np.ndarray, segments: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """True where each point lies on the water side of the nearest coastline segment.

    Ties at shared vertices go to the segment the point is most clearly beside
    (largest |cross|), the usual angle-weighted rule for signed distance to a
    polyline.
    """
    if len(segments) == 0 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    a = segments[:, 0]
    d = segments[:, 1] - segments[:, 0]
    dd = np.maximum((d * d).sum(1), 1e-12)
    out = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        rel = p[:, None, :] - a[None, :, :]
        t = np.clip((rel * d[None]).sum(-1) / dd[None], 0.0, 1.0)
        foot = a[None] + t[..., None] * d[None]
        dist = np.hypot(*(p[:, None, :] - foot).transpose(2, 0, 1))
        cross = d[None, :, 0] * rel[..., 1] - d[None, :, 1] * rel[..., 0]
        tie = dist <= dist.min(1, keepdims=True) + 1e-6
        pick = np.where(tie, np.abs(cross), -1.0).argmax(1)
        out[start : start + chunk] = cross[np.arange(len(p)), pick] < 0.0
    return out


def merge_rects(mask: np.ndarray, x0: float, y0: float, cell: float) -> list[list[float]]:
    """Greedy rectangles [x0, y0, x1, y1] covering the True cells of a (rows, cols) mask."""
    rows, cols = mask.shape
    runs: list[list[list[float]]] = []
    for r in range(rows):
        row = mask[r]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
        runs.append([[float(edges[k]), float(edges[k + 1])] for k in range(0, len(edges), 2)])
    rects: list[list[float]] = []
    open_: dict[tuple[float, float], int] = {}  # (c0, c1) -> start row
    for r in range(rows + 1):
        current = {tuple(run): True for run in runs[r]} if r < rows else {}
        for key, start in list(open_.items()):
            if key not in current:
                rects.append([x0 + key[0] * cell, y0 + start * cell, x0 + key[1] * cell, y0 + r * cell])
                del open_[key]
        for key in current:
            if key not in open_:
                open_[key] = r
    return [[round(v, 1) for v in rect] for rect in rects]


def water_and_land(
    coast: list[np.ndarray],
    centerline: np.ndarray,
    halfwidth: float,
    ground_half: float,
    fine_m: float = 4.0,
    coarse_m: float = 40.0,
    pad_m: float = 120.0,
    level_m: float = -1.2,
    road_keep_m: float = 2.0,
) -> dict:
    """Sea and harbour water as rectangles, plus the land that is left.

    A fine grid covers the coastline's bounding box (padded); a coarse grid
    covers the rest of the ground square. Cells under or beside the road are
    always land, so survey offsets in the coastline can never flood the track.
    """
    segments = coast_segments(coast)
    if len(segments) == 0:
        return {"water": [], "land": [], "quays": [], "level": 0.0}
    verts = segments.reshape(-1, 2)
    lo = np.maximum(verts.min(0) - pad_m, -ground_half)
    hi = np.minimum(verts.max(0) + pad_m, ground_half)
    lo = np.floor(lo / fine_m) * fine_m
    hi = np.ceil(hi / fine_m) * fine_m
    road_keep = halfwidth + road_keep_m

    def classify(x0: float, y0: float, cols: int, rows: int, cell: float, skip_box: tuple | None) -> np.ndarray:
        cx = x0 + (np.arange(cols) + 0.5) * cell
        cy = y0 + (np.arange(rows) + 0.5) * cell
        gx, gy = np.meshgrid(cx, cy)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        wet = water_side(pts, segments)
        # any cell touching the road corridor is land
        reach = road_keep + cell * 0.71
        near_road = np.zeros(len(pts), dtype=bool)
        for start in range(0, len(pts), 4096):
            block = pts[start : start + 4096]
            dmin = np.sqrt(((block[:, None, :] - centerline[None, ::2, :]) ** 2).sum(-1)).min(1)
            near_road[start : start + 4096] = dmin < reach
        wet &= ~near_road
        if skip_box is not None:
            bx0, by0, bx1, by1 = skip_box
            inside = (pts[:, 0] > bx0) & (pts[:, 0] < bx1) & (pts[:, 1] > by0) & (pts[:, 1] < by1)
            return wet.reshape(rows, cols), inside.reshape(rows, cols)
        return wet.reshape(rows, cols), np.zeros((rows, cols), dtype=bool)

    fcols = int(round((hi[0] - lo[0]) / fine_m))
    frows = int(round((hi[1] - lo[1]) / fine_m))
    fine_wet, _ = classify(lo[0], lo[1], fcols, frows, fine_m, None)
    g0 = -ground_half
    ccols = crows = int(np.ceil(2 * ground_half / coarse_m))
    coarse_wet, coarse_inside = classify(g0, g0, ccols, crows, coarse_m, (lo[0], lo[1], hi[0], hi[1]))
    # coarse cells overlapping the fine box are handled by the fine grid
    cx = g0 + (np.arange(ccols) + 0.5) * coarse_m
    cy = g0 + (np.arange(crows) + 0.5) * coarse_m
    gx, gy = np.meshgrid(cx, cy)
    overlap = (gx + coarse_m / 2 > lo[0]) & (gx - coarse_m / 2 < hi[0]) & (gy + coarse_m / 2 > lo[1]) & (gy - coarse_m / 2 < hi[1])
    coarse_wet = coarse_wet & ~overlap
    coarse_land = ~coarse_wet & ~overlap
    water = merge_rects(fine_wet, lo[0], lo[1], fine_m) + merge_rects(coarse_wet, g0, g0, coarse_m)
    land = merge_rects(~fine_wet, lo[0], lo[1], fine_m) + merge_rects(coarse_land, g0, g0, coarse_m)
    return {
        "water": water,
        "land": land,
        "quays": [pts.round(2).tolist() for pts in coast if len(pts) >= 2],
        "level": level_m,
        "fine_m": fine_m,
        "coarse_m": coarse_m,
        "cells_wet": int(fine_wet.sum() + coarse_wet.sum()),
    }


class Source:
    """Shared plumbing for the live simulation and the replay: clients and payloads."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config: ViewerConfig = load_viewer_config(args.config)
        self.device = pick_device(args.device)
        self.connectome = load_connectome(args.graph)
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.episodes: list[dict] = []  # recent episode endings, newest last
        self.track_name = ""
        sample, self.bands = stratified_sample(self.connectome.roles, self.config.raster.seed, self.config.raster.quota)
        self.sample_np = sample
        self.role_names = list(self.connectome.roles.keys())
        self.role_flat_np = np.concatenate(
            [np.asarray(self.connectome.roles[r], dtype=np.int64) for r in self.role_names]
        )
        self.role_owner_np = np.concatenate(
            [np.full(len(self.connectome.roles[r]), i, dtype=np.int64) for i, r in enumerate(self.role_names)]
        )
        self.positions, self.position_known = self.load_positions()
        self.role_ids = self.build_role_ids()
        self.projection: GeoProjection | None = None
        self.scenery: dict | None = None

    # --- anatomy ---------------------------------------------------------------------
    def load_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        path = Path(self.args.graph) / "positions.npy"
        if not path.exists():
            print(f"[viewer] no {path}; run src/build_positions.py for the 3D brain")
            return None, None
        pos = np.load(path).astype(np.float32)
        known_path = Path(self.args.graph) / "positions_known.npy"
        known = np.load(known_path) if known_path.exists() else np.ones(len(pos), bool)
        print(f"[viewer] anatomy {len(pos):,} somata, {int(known.sum()):,} measured")
        return pos, known

    def build_role_ids(self) -> np.ndarray:
        """Role index per neuron, 255 for cells in no labelled role."""
        ids = np.full(self.connectome.n, 255, dtype=np.uint8)
        ids[self.role_flat_np] = self.role_owner_np.astype(np.uint8)
        return ids

    # --- track and scenery -------------------------------------------------------------
    def build_track(self, layout: str, geojson: str, start_fraction: float, dt_s: float, halfwidth: float | None = None) -> None:
        if layout == "monaco":
            self.car_cfg = replace(monaco_config(dt_s, halfwidth=halfwidth), geojson_path=geojson)
            centerline, self.projection = load_geojson_centerline(
                geojson, self.car_cfg.track_scale, self.car_cfg.n_points, self.car_cfg.smooth_m, self.car_cfg.mirror, return_projection=True
            )
            props = json.loads(Path(geojson).read_text())["features"][0].get("properties", {})
            self.track_name = str(props.get("Name") or props.get("name") or Path(geojson).stem)
            self.track_props = {k: v for k, v in props.items() if isinstance(v, (str, int, float))}
        else:
            self.car_cfg = CarConfig(dt_s=dt_s) if halfwidth is None else CarConfig(dt_s=dt_s, track_halfwidth=halfwidth)
            centerline = build_centerline(self.car_cfg, self.args.track)
            self.track_name = f"procedural loop {self.args.track}"
            self.track_props = {"seed": self.args.track}
        self.track = Track(centerline, self.car_cfg, self.device)
        self.env = CarEnv(1, self.device, self.car_cfg, track=self.track, start_fraction=start_fraction)
        self.layout = layout
        scenery_path = Path(self.args.scenery) if self.args.scenery else Path(geojson).with_name(Path(geojson).stem + "_scenery.geojson")
        if layout == "monaco" and self.projection is not None and scenery_path.exists():
            self.scenery = load_scenery(
                scenery_path, self.projection, centerline.numpy(), self.car_cfg.track_halfwidth,
                self.config.track_stride, self.config.scenery_load,
            )
            print(
                f"[viewer] scenery {len(self.scenery['buildings'])} buildings "
                f"({self.scenery['dropped_on_road']} on open road dropped), tunnel spans {self.scenery['tunnel_spans']}, "
                f"water {len(self.scenery['water']['water'])} rects from {len(self.scenery['water']['quays'])} coastline ways"
            )
        else:
            self.scenery = None

    def track_payload(self) -> dict:
        centerline = self.track.centerline.cpu().numpy()
        cfg = self.car_cfg
        stride = self.config.track_stride
        return {
            "name": self.track_name,
            "properties": self.track_props,
            "centerline": centerline[::stride].round(2).tolist(),
            "stride": stride,
            "halfwidth": cfg.track_halfwidth,
            "extent": self.track.extent,
            "layout": self.layout,
            "urban": self.layout == "monaco",
            "length_m": round(self.track.length_m, 1),
            "min_radius_m": round(self.track.min_radius_m, 1),
            "start_index": int(self.env.start_index[0]) // stride,
            "start_fraction": float(self.env.start_index[0]) / centerline.shape[0],
            "n_rays": cfg.n_rays,
            "fov_deg": cfg.fov_deg,
            "max_range": cfg.max_range,
            "max_speed": cfg.max_speed,
            "car": {
                "length": cfg.car_length,
                "width": 2 * cfg.car_halfwidth,
                "wheelbase": cfg.wheelbase,
                "tyre_radius": cfg.tyre_radius_m,
                "min_turn_radius": round(cfg.min_turn_radius, 2),
                "name": cfg.vehicle_name,
            },
            "car_cfg": asdict(cfg),
            "has_scenery": self.scenery is not None,
            "has_water": bool(self.scenery and self.scenery["water"]["water"]),
            "tunnel_spans": self.scenery["tunnel_spans"] if self.scenery else [],
            "style": asdict(self.config.drive_style),
        }

    def scenery_payload(self) -> dict:
        base = self.scenery or {"buildings": [], "tunnel_spans": [], "tunnels": [], "lines": [], "water": {"water": [], "land": [], "quays": [], "level": 0.0}}
        return {**base, "style": asdict(self.config.scenery_style)}

    def record_episode(self, frame: dict) -> None:
        """Keep the ending of every episode: where, why, how fast."""
        entry = {
            "episode": frame["episode"],
            "generation": frame["generation"],
            "steps": frame["step"],
            "sim_s": round(frame["step"] * frame.get("control_dt_s", 0.0), 2),
            "reason": DONE_NAMES.get(int(frame.get("done_reason", 0)), "ended"),
            "laps": frame["laps"],
            "lap_fraction": round((frame.get("start_fraction", 0.0) + frame["laps"]) % 1.0, 4),
            "speed_kmh": round(frame["speed"] * self.config.ui.kmh_per_mps, 1),
            "clearance": frame.get("clearance"),
            "steer": frame["steer"],
            "pedal": frame["pedal"],
            "return": frame.get("episode_return"),
            "time": time.time(),
        }
        self.episodes.append(entry)
        del self.episodes[: -self.config.episode_log_keep]
        try:
            with Path(self.args.episode_log).open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError as exc:
            print(f"[viewer] episode log: {exc}")

    # --- streaming ---------------------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        client: queue.Queue = queue.Queue(maxsize=8)
        with self.lock:
            self.clients.add(client)
        if self.latest is not None:
            client.put_nowait(self.latest)
        return client

    def unsubscribe(self, client: queue.Queue) -> None:
        with self.lock:
            self.clients.discard(client)

    def broadcast(self, frame: dict) -> None:
        self.latest = frame
        with self.lock:
            targets = list(self.clients)
        for client in targets:
            try:
                client.put_nowait(frame)
            except queue.Full:
                pass  # slow tab: drop the frame rather than stall the simulation

    def neural_from_mask(self, spiked: np.ndarray, window_s: float) -> dict:
        sums = np.bincount(self.role_owner_np, weights=spiked[self.role_flat_np], minlength=len(self.role_names))
        rates = {
            role: round(float(sums[i] / len(self.connectome.roles[role]) / window_s), 1)
            for i, role in enumerate(self.role_names)
        }
        return {
            "fired": np.flatnonzero(spiked[self.sample_np]).tolist(),
            "rates": rates,
            "spiking": int(spiked.sum()),
            "mask": base64.b64encode(np.packbits(spiked)).decode(),
        }

    def meta_common(self) -> dict:
        return {
            "neurons": self.connectome.n,
            "edges": int(len(self.connectome.pre)),
            "synapses": int(np.abs(self.connectome.weight).sum()),
            "roles": {role: len(idx) for role, idx in self.connectome.roles.items()},
            "bands": self.bands,
            "device": str(self.device),
            "anatomy": self.positions is not None,
            "measured": int(self.position_known.sum()) if self.positions is not None else 0,
            "role_names": self.role_names,
            "track": self.args.track,
            "track_name": self.track_name,
            "layout": self.layout,
            "done_names": {str(k): v for k, v in DONE_NAMES.items()},
            "config": self.config.as_payload(),
            "lif_cfg": asdict(self.brain.cfg) if hasattr(self, "brain") else None,
            "car_cfg": asdict(self.car_cfg),
            "frame_schema": FRAME_SCHEMA,
        }


class Simulation(Source):
    """Drives one car with the connectome and broadcasts frames to clients."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        state = self.peek_checkpoint()
        dt_ms, substeps = args.dt_ms, args.substeps
        if state is not None and "args" in state:
            dt_ms = float(state["args"].get("dt_ms", dt_ms))
            substeps = int(state["args"].get("substeps", substeps))
        self.dt_ms, self.substeps = dt_ms, substeps
        self.brain = Brain(
            self.connectome,
            batch=1,
            config=LIFConfig(dt_ms=dt_ms, adapt_mv=args.adapt_mv),
            device=self.device,
            weight_scale=args.weight_scale,
            precision=args.precision,
        )
        agent_cfg = AgentConfig.from_saved(state["agent_cfg"]) if state and "agent_cfg" in state else AgentConfig(substeps=substeps)
        self.agent = ConnectomeAgent(self.brain, self.connectome.neurons, agent_cfg)
        dt_s = defaults.control_dt_s(dt_ms, substeps)
        halfwidth = None
        if state and "car_cfg" in state and args.follow_curriculum:
            halfwidth = float(state["car_cfg"].get("track_halfwidth"))
        self.build_track(args.layout, args.geojson, args.track / args.starts, dt_s, halfwidth)
        self.sample = torch.tensor(self.sample_np, dtype=torch.long, device=self.device)

        self.mu = self.agent.initial_params().to(self.device)
        self.theta = self.agent.unpack(self.mu.unsqueeze(0))
        self.generation = 0
        self.stage = -1
        self.checkpoint_mtime = 0.0
        self.checkpoint_note = "untrained interface"
        self._meta: dict = {}
        self.load_checkpoint()
        self.refresh_meta()

    def peek_checkpoint(self) -> dict | None:
        path = Path(self.args.checkpoint)
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location=self.device)
        except (RuntimeError, EOFError):
            return None

    def load_checkpoint(self) -> bool:
        path = Path(self.args.checkpoint)
        if not path.exists():
            return False
        mtime = path.stat().st_mtime
        if mtime == self.checkpoint_mtime:
            return False
        try:
            state = torch.load(path, map_location=self.device)
        except (RuntimeError, EOFError):
            return False  # mid-write from the trainer; try again next episode
        self.checkpoint_mtime = mtime
        try:
            mu, _, notes = self.agent.migrate_state(state)
        except ValueError as exc:
            self.checkpoint_note = f"{exc}: ignored"
            print(f"[viewer] {self.checkpoint_note}")
            return False
        self.mu = mu.to(self.device)
        self.theta = self.agent.unpack(self.mu.unsqueeze(0))
        self.generation = int(state["generation"])
        self.stage = int(state.get("stage", -1))
        self.checkpoint_note = f"{path.name} generation {self.generation}" + (f" ({', '.join(notes)})" if notes else "")
        if notes:
            print(f"[viewer] {self.checkpoint_note}")
        if self.args.follow_curriculum and "car_cfg" in state:
            hw = float(state["car_cfg"].get("track_halfwidth"))
            if abs(hw - self.car_cfg.track_halfwidth) > 1e-6:
                dt_s = defaults.control_dt_s(self.dt_ms, self.substeps)
                self.build_track(self.args.layout, self.args.geojson, self.args.track / self.args.starts, dt_s, hw)
                print(f"[viewer] curriculum road half-width now {hw:.2f} m; reload the page for the new track mesh")
        return True

    def dn_influence(self) -> np.ndarray:
        """Effective steering weight per output neuron (descending + motor), readout folded back."""
        return self.agent.steer_weight(self.theta).detach().cpu().numpy()

    def top_dn_index(self) -> torch.Tensor:
        order = np.argsort(-np.abs(self.dn_influence()))[: self.config.top_readout]
        return torch.tensor(order.copy(), dtype=torch.long, device=self.device)

    def params_payload(self) -> dict:
        """Every learned parameter block: values (small blocks), summary and bound status."""
        theta = self.agent.unpack(self.mu.unsqueeze(0))
        out = {}
        for name, (lo, hi) in self.agent.PARAM_BOUNDS.items():
            if name not in theta:
                continue
            v = theta[name][0].reshape(-1).detach().cpu()
            out[name] = {
                "shape": list(self.agent.param_shapes[name]),
                "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4),
                "mean": round(float(v.mean()), 4),
                "at_bounds": int(((v <= lo + 1e-4) | (v >= hi - 1e-4)).sum()),
                "bounds": [lo, hi],
                "values": [round(float(x), 4) for x in v] if v.numel() <= 16 else None,
            }
        return out

    def meta_payload(self) -> dict:
        """Served to HTTP threads from a cache: every tensor read here happens on the simulation thread.

        torch MPS work from two threads at once (the solver's kernels and a
        handler computing readout weights) tripped a Metal command-buffer
        assertion and took the viewer down.
        """
        return self._meta

    def refresh_meta(self) -> None:
        self._meta = self._compute_meta()

    def _compute_meta(self) -> dict:
        steer_w = self.dn_influence()
        order = np.argsort(-np.abs(steer_w))[: self.config.top_readout]
        return {
            **self.meta_common(),
            "mode": "live",
            "stage": self.stage,
            "road_halfwidth": self.car_cfg.track_halfwidth,
            "params": self.agent.n_params,
            "param_blocks": self.params_payload(),
            "agent_cfg": asdict(self.agent.cfg),
            "readout": {
                "roles": self.agent.cfg.readout_roles,
                "neurons": self.agent.n_readout,
                "channels": self.agent.cfg.readout_dim,
                "descending": self.agent.n_dn,
            },
            "brain": {"precision": self.brain.precision, "metal": self.brain.uses_metal},
            "dt_ms": self.dt_ms,
            "substeps": self.substeps,
            "control_dt_s": defaults.control_dt_s(self.dt_ms, self.substeps),
            "checkpoint": self.checkpoint_note,
            "checkpoint_path": str(self.args.checkpoint),
            "speed": self.args.speed,
            "top_dn": [
                {
                    "type": str(self.agent.readout_types[i]),
                    "body": int(self.agent.readout_bodies[i]),
                    "role": self.agent.readout_role_of[i],
                    "weight": round(float(steer_w[i]), 4),
                }
                for i in order
            ],
        }

    def run(self) -> None:
        top_dn = self.top_dn_index()
        self.agent.seed(int(time.time()) % 100_000)
        obs = self.env.reset()
        self.agent.reset()
        step, episode, wall = 0, 0, time.time()
        last_emit = wall
        episode_return = 0.0
        sim_seconds = defaults.control_dt_s(self.dt_ms, self.substeps)
        while True:
            # Nobody watching: do not take GPU time from the trainer. Keep
            # following the checkpoint so the first frame after a tab opens is
            # the current policy.
            if not self.clients:
                if self.load_checkpoint():
                    top_dn = self.top_dn_index()
                    self.refresh_meta()
                    obs = self.env.reset()
                    self.agent.reset()
                    step = 0
                time.sleep(0.25)
                wall += 0.25
                continue
            fired_total = torch.zeros(1, self.brain.n, device=self.device)
            started = time.time()
            with torch.inference_mode():
                action = self.agent.act(obs, self.theta, spike_sink=fired_total)
                synchronize(self.device)
                brain_done = time.time()
                obs, reward, done = self.env.step(action)
            step += 1

            body = self.body_state(action, reward, obs)
            neural = self.readout(fired_total, top_dn, sim_seconds)
            elapsed = max(1e-6, time.time() - started)
            episode_return += body["reward"]
            frame = {
                "step": step,
                "episode": episode,
                "generation": self.generation,
                "stage": self.stage,
                "checkpoint": self.checkpoint_note,
                "max_speed": self.car_cfg.max_speed,
                "control_dt_s": sim_seconds,
                "start_fraction": float(self.env.start_index[0]) / self.track.centerline.shape[0],
                "episode_return": round(episode_return, 3),
                **body,
                **neural,
                "fps": round(1.0 / max(1e-6, time.time() - last_emit), 1),
                # How much faster than a real fly the solver could run if it
                # were not paced back for the browser.
                "headroom": round(sim_seconds / elapsed, 2),
                "brain_ms": round(1000.0 * (brain_done - started), 2),
                "env_ms": round(1000.0 * (time.time() - brain_done), 2),
                "uptime": round(time.time() - wall, 1),
            }
            self.broadcast(frame)
            last_emit = time.time()

            # The simulation runs several times faster than the fly does; pace
            # it back so the browser sees the car at a watchable speed instead
            # of hundreds of frames a second it cannot draw.
            if self.args.speed > 0:
                remaining = sim_seconds / self.args.speed - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)

            if bool(done[0]) or step >= self.args.episode_steps:
                self.record_episode(frame)
                episode += 1
                step = 0
                episode_return = 0.0
                if self.load_checkpoint():
                    top_dn = self.top_dn_index()
                    self.refresh_meta()
                obs = self.env.reset()
                self.agent.reset()

    def body_state(self, action: torch.Tensor, reward: torch.Tensor, obs: torch.Tensor) -> dict:
        """Car pose, controls, reward terms, sensory drive and lidar in one device->host transfer."""
        env = self.env
        terms = env.last_terms
        n_rays = self.car_cfg.n_rays
        packed = torch.cat(
            [
                env.pos[0],
                env.heading[:1],
                env.speed[:1],
                env.laps[:1],
                reward[:1],
                action[0],
                env.lat_g[:1],
                env.done_reason[:1].float(),
                env.steer[:1] / self.car_cfg.max_steer_rad,
                terms["progress"][:1],
                terms["bonus"][:1],
                terms["wall"][:1],
                terms["time_tax"][:1],
                terms["clearance"][:1],
                self.agent.last_motor[0],
                obs[0, : n_rays],
                self.agent.last_rates_hz[0],
            ]
        )
        v = packed.to("cpu").numpy().round(4).tolist()
        k = 18  # fixed scalars above, then lidar, then sensory rates (rays + speed channel)
        lidar = v[k : k + n_rays]
        rates = v[k + n_rays : k + 2 * n_rays + 1]
        return {
            "pos": v[0:2],
            "heading": v[2],
            "speed": v[3],
            "laps": v[4],
            "reward": v[5],
            "steer": v[6],
            "pedal": v[7],
            "throttle": max(0.0, v[7]),
            "brake": max(0.0, -v[7]),
            "lat_g": v[8],
            "done_reason": int(v[9]),
            "steer_actual": v[10],
            "reward_progress": v[11],
            "reward_bonus": v[12],
            "reward_wall": v[13],
            "reward_time": v[14],
            "clearance": v[15],
            "motor_steer": v[16],
            "motor_pedal": v[17],
            "lidar": lidar,
            "sensory_hz": rates[:n_rays],
            "speed_hz": rates[n_rays] if len(rates) > n_rays else 0.0,
        }

    def readout(self, fired_total: torch.Tensor, top_dn: torch.Tensor, window_s: float) -> dict:
        """Whole-brain spike mask, raster hits, role rates, and top-DN rates.

        The spike row is narrowed to bytes on the device before it crosses to
        the host: as float32 it is 667 kB per control step, which at 60 steps a
        second costs more than the network solve.
        """
        spiked = (fired_total[0] > 0).to(torch.uint8).to("cpu").numpy()
        dn_rates = self.agent.readout_rate_hz[0, top_dn].to("cpu").numpy()
        return {**self.neural_from_mask(spiked, window_s), "dn": [round(float(v), 2) for v in dn_rates]}


class Replay(Source):
    """Streams a recorded run (`evaluate.py --record`) with its spike masks."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        data = np.load(args.replay)
        self.meta = json.loads(str(data["meta"]))
        self.rec = {k: data[k] for k in data.files if k != "meta"}
        self.steps = len(self.rec["pos"])
        self.dt_ms = float(self.meta["dt_ms"])
        self.substeps = int(self.meta["substeps"])
        dt_s = defaults.control_dt_s(self.dt_ms, self.substeps)
        self.build_track(self.meta["layout"], self.meta.get("geojson", args.geojson), float(self.meta["start_fraction"]), dt_s, float(self.meta["track_halfwidth"]))
        self.generation = int(self.meta.get("generation", 0))
        dn = np.asarray(self.connectome.roles["descending"], dtype=np.int64)
        self.dn_types = self.connectome.neurons["type"].to_numpy()[dn]
        self.dn_bodies = self.connectome.neurons["bodyId"].to_numpy()[dn]
        # the most active DNs over the run stand in for "top by influence"
        self.top = np.argsort(-self.rec["dn_hz"].mean(0))[: self.config.top_readout]
        print(f"[viewer] replay {args.replay}: {self.steps} steps, generation {self.generation}")

    def meta_payload(self) -> dict:
        return {
            **self.meta_common(),
            "mode": "replay",
            "params": 0,
            "dt_ms": self.dt_ms,
            "substeps": self.substeps,
            "control_dt_s": defaults.control_dt_s(self.dt_ms, self.substeps),
            "checkpoint": f"replay {Path(self.args.replay).name}, generation {self.generation}",
            "top_dn": [
                {"type": str(self.dn_types[i]), "body": int(self.dn_bodies[i]), "weight": round(float(self.rec["dn_hz"][:, i].mean()), 2)}
                for i in self.top
            ],
        }

    def run(self) -> None:
        wall = time.time()
        window_s = defaults.control_dt_s(self.dt_ms, self.substeps)
        episode = 0
        while True:
            for step in range(self.steps):
                started = time.time()
                spiked = np.unpackbits(self.rec["mask"][step])[: self.connectome.n]
                frame = {
                    "step": step + 1,
                    "episode": episode,
                    "generation": self.generation,
                    "max_speed": self.car_cfg.max_speed,
                    "pos": self.rec["pos"][step].round(4).tolist(),
                    "heading": float(self.rec["heading"][step]),
                    "speed": float(self.rec["speed"][step]),
                    "laps": float(self.rec["laps"][step]),
                    "reward": float(self.rec["reward"][step]),
                    "steer": float(self.rec["steer"][step]),
                    "pedal": float(self.rec["pedal"][step]),
                    "throttle": max(0.0, float(self.rec["pedal"][step])),
                    "lat_g": float(self.rec["lat_g"][step]),
                    "done_reason": 0,
                    "lidar": self.rec["lidar"][step].round(4).tolist(),
                    **self.neural_from_mask(spiked, window_s),
                    "dn": [round(float(v), 2) for v in self.rec["dn_hz"][step, self.top]],
                    "fps": round(1.0 / max(1e-3, self.args.speed and window_s / self.args.speed or 1e-3), 1),
                    "headroom": 0.0,
                    "uptime": round(time.time() - wall, 1),
                }
                self.broadcast(frame)
                if self.args.speed > 0:
                    remaining = window_s / self.args.speed - (time.time() - started)
                    if remaining > 0:
                        time.sleep(remaining)
            episode += 1


def read_log(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def split_runs(records: list[dict]) -> list[list[dict]]:
    """Runs concatenated in one log: a generation counter that goes *down* starts a new run.

    A repeated generation (two trainers writing at once, or a resumed run
    re-logging its last generation) stays in the current run, deduplicated.
    """
    runs: list[list[dict]] = [[]]
    for r in records:
        g = r.get("generation", 0)
        if runs[-1]:
            last = runs[-1][-1].get("generation", 0)
            if g < last:
                runs.append([])
            elif g == last:
                runs[-1][-1] = r
                continue
        runs[-1].append(r)
    return [run for run in runs if run] or [[]]


def numeric_keys(records: list[dict]) -> list[str]:
    """Every field that is a plain number in at least one record, in first-seen order."""
    seen: dict[str, None] = {}
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                seen.setdefault(k, None)
    return list(seen)


def curve_payload(log: Path, points: int, keys: list[str] | None = None) -> dict:
    """Downsampled series of any numeric log fields, plus every key available.

    Records are the concatenation of every run written to the log; a run
    boundary is a generation counter that goes backwards. `run` selects the
    latest one by default (index -1).
    """
    records = read_log(log)
    if not records:
        return {"generations": 0, "fitness": [], "laps": [], "keys": [], "series": {}, "runs": 0}
    runs = split_runs(records)
    latest = runs[-1]
    stride = max(1, len(latest) // points)
    sampled = latest[::stride]
    keys = keys or ["fitness_mean", "fitness_best", "laps_best"]
    evals = [r for r in latest if "eval_fitness" in r]
    return {
        "generations": len(latest),
        "runs": len(runs),
        "run_sizes": [len(run) for run in runs],
        "hours": round(sum(r.get("seconds", 0.0) for r in latest) / 3600.0, 2),
        # legacy fields kept for the tests and old pages
        "fitness": [round(r["fitness_mean"], 2) for r in sampled if "fitness_mean" in r],
        "best": [round(r["fitness_best"], 2) for r in sampled if "fitness_best" in r],
        "laps": [round(r["laps_best"], 4) for r in sampled if "laps_best" in r],
        "stage": [int(r.get("stage", 0)) for r in sampled],
        "generation": [int(r.get("generation", 0)) for r in sampled],
        "series": {k: [r.get(k) if isinstance(r.get(k), (int, float)) else None for r in sampled] for k in keys},
        "eval": [round(r["eval_fitness"], 2) for r in evals][-points:],
        "eval_gen": [int(r["generation"]) for r in evals][-points:],
        "eval_laps": [round(r.get("eval_laps", 0.0), 4) for r in evals][-points:],
        "keys": numeric_keys(latest),
        "stride": stride,
    }


def training_payload(log: Path, episodes: list[dict], alive_factor: float, window: int = 20) -> dict:
    """Trainer status from its log: last record in full, rate, liveness; plus the viewer's episode endings."""
    records = read_log(log)
    if not records:
        return {"present": False, "episodes": episodes}
    runs = split_runs(records)
    latest_run = runs[-1]
    recent = latest_run[-window:]
    seconds = [r.get("seconds", 0.0) for r in recent]
    mean_gen_s = sum(seconds) / max(1, len(seconds))
    last = latest_run[-1]
    age = time.time() - (last.get("time") or log.stat().st_mtime)
    times = [r.get("time") for r in recent if r.get("time")]
    gens_per_hour = (len(times) - 1) / max(1e-6, (times[-1] - times[0])) * 3600.0 if len(times) > 1 else (3600.0 / mean_gen_s if mean_gen_s else 0.0)
    return {
        "present": True,
        "last": last,
        "generation": last.get("generation"),
        "run": len(runs),
        "generations_in_run": len(latest_run),
        "log_age_s": round(age, 1),
        "mean_generation_s": round(mean_gen_s, 2),
        "generations_per_hour": round(gens_per_hour, 1),
        "alive": bool(age < max(30.0, alive_factor * mean_gen_s)),
        "best_eval": max((r["eval_fitness"] for r in latest_run if "eval_fitness" in r), default=None),
        "best_laps": max((r.get("laps_best", 0.0) for r in latest_run), default=None),
        "stage": last.get("stage"),
        "keys": numeric_keys(latest_run),
        "recent": recent,
        "episodes": episodes,
    }


def make_handler(sim: Source, args: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # noqa: ANN002
            pass  # the frame stream would otherwise flood the console

        def send_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_binary(self, payload: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_file(self, name: str) -> None:
            path = (WEB_DIR / name).resolve()
            if not path.is_file() or WEB_DIR.resolve() not in path.parents:
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "text/plain"))
            self.send_header("Content-Length", str(len(body)))
            # vendored libraries and textures are immutable; page code is not
            cacheable = "/vendor/" in str(path) or "/assets/" in str(path)
            self.send_header("Cache-Control", "max-age=86400" if cacheable else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            client = sim.subscribe()
            try:
                while True:
                    frame = client.get()
                    self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                sim.unsubscribe(client)

        def do_HEAD(self) -> None:  # noqa: N802
            """Existence probe for optional assets (the drive view asks for w11.glb this way)."""
            route = self.path.split("?")[0]
            path = (WEB_DIR / route.lstrip("/")).resolve()
            ok = route.startswith(("/vendor/", "/assets/")) and path.is_file() and WEB_DIR.resolve() in path.parents
            self.send_response(200 if ok else 404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            route = self.path.split("?")[0]
            if route in ("/", "/index.html"):
                self.send_file("index.html")
            elif route.startswith(("/vendor/", "/assets/")) or route in (
                "/app.js",
                "/brain.js",
                "/drive.js",
                "/scenery.js",
                "/style.css",
            ):
                self.send_file(route.lstrip("/"))
            elif route == "/api/track":
                self.send_json(sim.track_payload())
            elif route == "/api/scenery":
                self.send_json(sim.scenery_payload())
            elif route == "/api/meta":
                self.send_json(sim.meta_payload())
            elif route == "/api/positions.bin":
                if sim.positions is None:
                    self.send_error(404)
                else:
                    self.send_binary(sim.positions.astype(np.float32).tobytes())
            elif route == "/api/roles.bin":
                if sim.position_known is None:
                    self.send_error(404)
                else:
                    # role id per neuron, then the measured/inferred flag
                    self.send_binary(sim.role_ids.tobytes() + sim.position_known.astype(np.uint8).tobytes())
            elif route == "/api/curve":
                params = parse_qs(urlsplit(self.path).query)
                keys = [k for k in ",".join(params.get("keys", [])).split(",") if k]
                self.send_json(curve_payload(Path(args.log), sim.config.curve_points, keys or None))
            elif route == "/api/config":
                self.send_json(sim.config.as_payload())
            elif route == "/api/training":
                self.send_json(training_payload(Path(args.log), sim.episodes, sim.config.trainer_alive_factor))
            elif route == "/api/stream":
                self.stream()
            elif route == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_error(404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--checkpoint", default="checkpoints/es.pt")
    parser.add_argument("--replay", default=None, help="play a run recorded with evaluate.py --record instead of simulating")
    parser.add_argument("--log", default="logs/train.jsonl")
    parser.add_argument("--layout", default="monaco", choices=("monaco", "loop"))
    parser.add_argument("--geojson", default="data/tracks/monaco.geojson")
    parser.add_argument("--scenery", default=None, help="OSM scenery GeoJSON; default <geojson>_scenery.geojson")
    parser.add_argument("--track", type=int, default=0, help="loop seed, or start point index on the circuit")
    parser.add_argument("--starts", type=int, default=defaults.MONACO_STARTS)
    parser.add_argument("--episode-steps", type=int, default=defaults.EVAL_STEPS)
    parser.add_argument("--dt-ms", type=float, default=defaults.DT_MS)
    parser.add_argument("--substeps", type=int, default=defaults.SUBSTEPS)
    parser.add_argument("--weight-scale", type=float, default=defaults.WEIGHT_SCALE)
    parser.add_argument("--adapt-mv", type=float, default=defaults.ADAPT_MV)
    parser.add_argument(
        "--follow-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drive on the road width of the checkpoint's curriculum stage (default); --no-follow-curriculum for the real 11 m road",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default=None, help="JSON overriding any ViewerConfig field (see viewer_config.py)")
    parser.add_argument("--episode-log", default="logs/viewer_episodes.jsonl", help="every episode ending the viewer sees is appended here")
    parser.add_argument("--precision", default=None, choices=("fp16", "fp32"), help="brain state precision (default: MALECNS_PRECISION or fp16)")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed; 1.0 is real time, 0 runs uncapped",
    )
    args = parser.parse_args(argv)

    sim: Source = Replay(args) if args.replay else Simulation(args)
    print(
        f"connectome {sim.connectome.n:,} neurons / {len(sim.connectome.pre):,} edges "
        f"on {sim.device}"
    )
    print(f"[viewer] {sim.meta_payload()['checkpoint']}")
    threading.Thread(target=sim.run, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(sim, args))
    server.daemon_threads = True
    print(f"[viewer] http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
