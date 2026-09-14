"""Fetch real scenery around a circuit from OpenStreetMap (Overpass API).

Buildings (footprint + height/levels), road tunnels and the coastline inside
the circuit's bounding box, saved as compact GeoJSON for the viewer:

  python3 src/fetch_scenery.py --track data/tracks/monaco.geojson \
      --out data/tracks/monaco_scenery.geojson

Data (c) OpenStreetMap contributors, ODbL 1.0. Nothing here is needed for
training; the physics only knows the track. The viewer extrudes the footprints
in the same projection as the centerline so they land where they stand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
LEVEL_M = 3.2  # metres per storey when only building:levels is tagged


def bbox_of(track: Path, pad_deg: float) -> tuple[float, float, float, float]:
    coords = json.loads(track.read_text())["features"][0]["geometry"]["coordinates"]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lats) - pad_deg, min(lons) - pad_deg, max(lats) + pad_deg, max(lons) + pad_deg


def queries(bbox: tuple[float, float, float, float]) -> list[str]:
    """One small query per feature kind: a single union times out on public mirrors."""
    s, w, n, e = bbox
    box = f"{s},{w},{n},{e}"
    parts = (
        f'way["building"]({box});',
        f'relation["building"]({box});',
        f'way["tunnel"]["highway"]({box});',
        f'(way["natural"="coastline"]({box}); way["man_made"="pier"]({box}); way["man_made"="breakwater"]({box}););',
    )
    return [f"[out:json][timeout:60];{part}out geom;" for part in parts]


def parse_height(tags: dict) -> float:
    for key in ("height", "building:height"):
        raw = tags.get(key)
        if raw:
            try:
                return float(str(raw).replace("m", "").strip())
            except ValueError:
                pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return float(levels) * LEVEL_M
        except ValueError:
            pass
    return 0.0  # unknown: the viewer picks a plausible height from the footprint


def ring_of(geometry: list[dict]) -> list[list[float]]:
    return [[p["lon"], p["lat"]] for p in geometry]


def convert(raw: dict) -> dict:
    features = []
    for el in raw.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            ring = ring_of(el["geometry"])
            if "building" in tags:
                if len(ring) < 4:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "building",
                            "height": parse_height(tags),
                            "levels": tags.get("building:levels"),
                            "name": tags.get("name"),
                            "building": tags.get("building"),
                            "roof": tags.get("roof:shape"),
                            "colour": tags.get("building:colour"),
                        },
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                )
            elif tags.get("tunnel"):
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "tunnel",
                            "name": tags.get("name"),
                            "highway": tags.get("highway"),
                            "layer": tags.get("layer"),
                        },
                        "geometry": {"type": "LineString", "coordinates": ring},
                    }
                )
            else:
                kind = "coastline" if tags.get("natural") == "coastline" else tags.get("man_made", "line")
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"kind": kind, "name": tags.get("name")},
                        "geometry": {"type": "LineString", "coordinates": ring},
                    }
                )
        elif el["type"] == "relation" and "building" in tags:
            outers = [ring_of(m["geometry"]) for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
            inners = [ring_of(m["geometry"]) for m in el.get("members", []) if m.get("role") == "inner" and "geometry" in m]
            for outer in outers:
                if len(outer) < 4:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "building",
                            "height": parse_height(tags),
                            "levels": tags.get("building:levels"),
                            "name": tags.get("name"),
                            "building": tags.get("building"),
                            "roof": tags.get("roof:shape"),
                            "colour": tags.get("building:colour"),
                        },
                        "geometry": {"type": "Polygon", "coordinates": [outer, *inners]},
                    }
                )
    return {
        "type": "FeatureCollection",
        "attribution": "(c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright",
        "features": features,
    }


def fetch(text: str) -> dict:
    """POST one Overpass query through curl (public mirrors 406/504 plain urllib)."""
    errors = []
    for url in MIRRORS:
        result = subprocess.run(
            [
                "curl", "-sS", "-m", "150", "-A", "malecns-scenery/1.0 (+https://github.com/hotocoo/malecns)",
                "-H", "Content-Type: application/x-www-form-urlencoded",
                "--data-urlencode", f"data={text}", url,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.lstrip().startswith("{"):
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                errors.append(f"{url}: {exc}")
                continue
        errors.append(f"{url}: curl {result.returncode} {result.stderr.strip()[:120]} {result.stdout[:120]!r}")
        print(f"[overpass] {url} failed, trying next mirror", file=sys.stderr)
        time.sleep(3)
    raise SystemExit("all Overpass mirrors failed:\n  " + "\n  ".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=Path("data/tracks/monaco.geojson"), type=Path)
    parser.add_argument("--out", default=Path("data/tracks/monaco_scenery.geojson"), type=Path)
    parser.add_argument("--pad-deg", type=float, default=0.0025, help="~250 m around the circuit")
    parser.add_argument("--raw", type=Path, default=None, help="also keep the raw Overpass reply")
    args = parser.parse_args(argv)

    bbox = bbox_of(args.track, args.pad_deg)
    print(f"[overpass] bbox {bbox}", file=sys.stderr)
    raw: dict = {"elements": []}
    for i, text in enumerate(queries(bbox)):
        if i:
            time.sleep(2)  # be polite to the public mirrors
        raw["elements"].extend(fetch(text)["elements"])
    if args.raw:
        args.raw.write_text(json.dumps(raw))
    out = convert(raw)
    kinds: dict[str, int] = {}
    for f in out["features"]:
        kinds[f["properties"]["kind"]] = kinds.get(f["properties"]["kind"], 0) + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"[ok] {args.out} {kinds}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
