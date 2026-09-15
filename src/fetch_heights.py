"""Fill in building heights the OpenStreetMap survey left blank, from real sources only.

Priority per footprint:
  1. OSM `height` / `building:height`, then `building:levels` (already in the scenery file)
  2. Overture Maps buildings (OSM + Microsoft ML Buildings): `height`, then `num_floors`
  3. the median height of surveyed buildings within `--neighbour-m` (a block's
     neighbours, not a random number); flagged `height_source: "neighbours"`

  python3 src/fetch_heights.py --scenery data/tracks/monaco_scenery.geojson

Overture data is fetched with the `overturemaps` CLI (pip install overturemaps)
for the scenery file's bounding box unless `--overture` names a GeoJSON already
downloaded. Overture Maps Foundation data, ODbL / CDLA-Permissive-2.0.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

LEVEL_M = 3.2
EARTH_M_PER_DEG_LAT = 111_320.0


def ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    pts = np.asarray(ring, dtype=np.float64)[:, :2]
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][:2]
        x2, y2 = ring[(i + 1) % n][:2]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def overture_height(props: dict) -> tuple[float, str] | None:
    h = props.get("height")
    if isinstance(h, (int, float)) and h > 0:
        return float(h), "overture:height"
    floors = props.get("num_floors")
    if isinstance(floors, (int, float)) and floors > 0:
        return float(floors) * LEVEL_M, "overture:num_floors"
    return None


def load_overture(path: Path | None, bbox: tuple[float, float, float, float], keep: Path | None) -> list[dict]:
    if path is None:
        out = keep or Path("data/tracks/overture_buildings.geojson")
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["overturemaps", "download", "--bbox", ",".join(f"{v:.6f}" for v in bbox), "-f", "geojson", "--type", "building", "-o", str(out)]
        print("[overture] " + " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
        path = out
    return json.loads(path.read_text())["features"]


def outer_rings(feature: dict) -> list[list[list[float]]]:
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


def fill(scenery: dict, overture: list[dict], neighbour_m: float) -> dict:
    buildings = [f for f in scenery["features"] if f["properties"].get("kind") == "building"]
    for f in buildings:
        f["properties"].setdefault("height_source", "osm" if (f["properties"].get("height") or 0) > 0 else None)
    # Overture footprints with a height, indexed by centroid for a cheap match
    ov = []
    for feat in overture:
        hs = overture_height(feat["properties"])
        if hs is None:
            continue
        for ring in outer_rings(feat):
            cx, cy = ring_centroid(ring)
            ov.append((cx, cy, ring, hs))
    ov_xy = np.asarray([[o[0], o[1]] for o in ov]) if ov else np.zeros((0, 2))
    lat0 = float(np.mean([ring_centroid(f["geometry"]["coordinates"][0])[1] for f in buildings])) if buildings else 0.0
    m_per_deg = np.array([EARTH_M_PER_DEG_LAT * np.cos(np.radians(lat0)), EARTH_M_PER_DEG_LAT])
    counts = {"osm": 0, "overture:height": 0, "overture:num_floors": 0, "neighbours": 0, "none": 0}
    for f in buildings:
        if (f["properties"].get("height") or 0) > 0:
            counts["osm"] += 1
            continue
        ring = f["geometry"]["coordinates"][0]
        cx, cy = ring_centroid(ring)
        if len(ov):
            d = np.hypot(*((ov_xy - [cx, cy]) * m_per_deg).T)
            order = np.argsort(d)[:8]
            hit = None
            for k in order:
                if d[k] > 25.0:
                    break
                ocx, ocy, oring, hs = ov[k]
                if point_in_ring(cx, cy, oring) or point_in_ring(ocx, ocy, ring) or d[k] < 6.0:
                    hit = hs
                    break
            if hit is not None:
                f["properties"]["height"] = round(hit[0], 1)
                f["properties"]["height_source"] = hit[1]
                counts[hit[1]] += 1
    # neighbours: median of surveyed heights within reach, from what is now known
    known = [(ring_centroid(f["geometry"]["coordinates"][0]), f["properties"]["height"]) for f in buildings if (f["properties"].get("height") or 0) > 0]
    known_xy = np.asarray([k[0] for k in known]) if known else np.zeros((0, 2))
    known_h = np.asarray([k[1] for k in known]) if known else np.zeros(0)
    for f in buildings:
        if (f["properties"].get("height") or 0) > 0:
            continue
        cx, cy = ring_centroid(f["geometry"]["coordinates"][0])
        if len(known):
            d = np.hypot(*((known_xy - [cx, cy]) * m_per_deg).T)
            reach = neighbour_m
            near = known_h[d < reach]
            while len(near) < 3 and reach < 8 * neighbour_m:  # an isolated block: widen until three surveyed neighbours
                reach *= 2
                near = known_h[d < reach]
            if len(near) >= 3:
                f["properties"]["height"] = round(float(statistics.median(near.tolist())), 1)
                f["properties"]["height_source"] = "neighbours" if reach == neighbour_m else f"neighbours:{int(reach)}m"
                counts["neighbours"] += 1
                continue
        counts["none"] += 1
    scenery["height_sources"] = counts
    scenery.setdefault("attribution", "")
    if "Overture" not in scenery["attribution"]:
        scenery["attribution"] += "; building heights also from Overture Maps (OpenStreetMap, Microsoft ML Buildings), ODbL / CDLA-Permissive-2.0"
    return scenery


def bbox_of(scenery: dict) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for f in scenery["features"]:
        if f["geometry"]["type"] == "Polygon":
            for x, y in (p[:2] for p in f["geometry"]["coordinates"][0]):
                xs.append(x)
                ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenery", default=Path("data/tracks/monaco_scenery.geojson"), type=Path)
    parser.add_argument("--overture", default=None, type=Path, help="Overture buildings GeoJSON already downloaded")
    parser.add_argument("--keep-overture", default=None, type=Path, help="where to store the Overture download")
    parser.add_argument("--neighbour-m", type=float, default=150.0)
    args = parser.parse_args(argv)
    scenery = json.loads(args.scenery.read_text())
    overture = load_overture(args.overture, bbox_of(scenery), args.keep_overture)
    out = fill(scenery, overture, args.neighbour_m)
    args.scenery.write_text(json.dumps(out, separators=(",", ":")))
    print(f"[ok] {args.scenery} heights by source: {out['height_sources']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
