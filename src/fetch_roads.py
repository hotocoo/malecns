"""Fetch the legal road survey around a circuit from OpenStreetMap (Overpass API).

Three things the reward needs and cannot invent:

  * every `highway` way with its legal tags (`maxspeed`, `lanes`, `oneway`,
    `junction`, `tunnel`), kept as raw Overpass geometry so `roadlaw.py` can
    project it into the track frame;
  * the control nodes that make the law enforceable at a point - traffic
    signals, stop and give-way lines, crossings, speed cameras, signed limits;
  * the driving side, read from the enclosing country boundary relation rather
    than assumed from the coordinates.

  python3 src/fetch_roads.py --track data/tracks/monaco.geojson \
      --out data/tracks/monaco_osm_highways.json

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fetch_scenery import bbox_of, fetch

ATTRIBUTION = "(c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright"

# Point features that carry a rule the driver must obey at a place. Each is a
# node tag OSM already uses; none is invented here.
CONTROL_NODE_TAGS = (
    'node["highway"="traffic_signals"]',
    'node["highway"="stop"]',
    'node["highway"="give_way"]',
    'node["highway"="crossing"]',
    'node["highway"="speed_camera"]',
    'node["highway"="mini_roundabout"]',
    'node["traffic_calming"]',
    'node["traffic_sign"]',
    'node["maxspeed"]',
    'node["barrier"="toll_booth"]',
    'node["railway"="level_crossing"]',
)


def fetch_retry(text: str, tries: int = 3, pause_s: float = 20.0) -> dict:
    """`fetch` with a second and third pass over the mirrors.

    Overpass answers a busy moment with `Dispatcher_Client ... timeout`, which
    the same query survives minutes later; giving up on the first sweep would
    make the survey depend on the mirror's mood.
    """
    for attempt in range(1, tries + 1):
        try:
            return fetch(text)
        except SystemExit:
            if attempt == tries:
                raise
            print(f"[overpass] all mirrors busy, retry {attempt + 1}/{tries} in {pause_s:.0f}s", file=sys.stderr)
            time.sleep(pause_s)
    raise SystemExit("unreachable")


def way_query(bbox: tuple[float, float, float, float]) -> str:
    """Every mapped highway in the box, with geometry and tags."""
    s, w, n, e = bbox
    return f'[out:json][timeout:120];way["highway"]({s},{w},{n},{e});out geom;'


def node_queries(bbox: tuple[float, float, float, float]) -> list[str]:
    """One query per control-node kind: a single union times out on public mirrors."""
    s, w, n, e = bbox
    box = f"{s},{w},{n},{e}"
    return [f"[out:json][timeout:120];{tag}({box});out body;" for tag in CONTROL_NODE_TAGS]


def driving_side_query(lat: float, lon: float) -> str:
    """The areas containing a point, tags only.

    The country area carries `driving_side` where the community has surveyed
    it; nothing is derived from the longitude.
    """
    return f"[out:json][timeout:60];is_in({lat},{lon});out tags;"


def driving_side_from(reply: dict) -> str | None:
    """The `driving_side` of the smallest admin level that states one, or None.

    Several nested relations may answer; the most local statement wins, and
    when no relation states a side the survey simply does not know.
    """
    best: tuple[int, str] | None = None
    for element in reply.get("elements", []):
        tags = element.get("tags") or {}
        side = str(tags.get("driving_side") or "").strip().lower()
        if side not in ("left", "right"):
            continue
        try:
            level = int(tags.get("admin_level", 0))
        except (TypeError, ValueError):
            level = 0
        if best is None or level > best[0]:
            best = (level, side)
    return None if best is None else best[1]


def centre_of(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    s, w, n, e = bbox
    return (s + n) / 2.0, (w + e) / 2.0


def summarise(ways: list[dict], nodes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"ways": len(ways), "control_nodes": len(nodes)}
    for key in ("maxspeed", "lanes", "oneway", "junction", "tunnel"):
        counts[f"ways_with_{key}"] = sum(1 for w in ways if key in (w.get("tags") or {}))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=Path("data/tracks/monaco.geojson"), type=Path)
    parser.add_argument("--out", default=None, type=Path, help="default: <track stem>_osm_highways.json")
    parser.add_argument("--pad-deg", type=float, default=0.0025, help="~250 m around the circuit")
    parser.add_argument("--no-nodes", action="store_true", help="skip the control-node query")
    args = parser.parse_args(argv)

    out = args.out or args.track.with_name(args.track.stem + "_osm_highways.json")
    bbox = bbox_of(args.track, args.pad_deg)
    print(f"[overpass] bbox {bbox}", file=sys.stderr)

    reply = fetch_retry(way_query(bbox))
    ways = [e for e in reply.get("elements", []) if e.get("type") == "way"]

    nodes: list[dict] = []
    seen: set[int] = set()
    if not args.no_nodes:
        for text in node_queries(bbox):
            time.sleep(2)  # be polite to the public mirrors
            for element in fetch_retry(text).get("elements", []):
                # A node can answer several queries (a signalised crossing is
                # both); keep one copy, tags merged as the survey has them.
                if element.get("type") == "node" and element.get("id") not in seen:
                    seen.add(element["id"])
                    nodes.append(element)

    time.sleep(2)
    lat, lon = centre_of(bbox)
    side = driving_side_from(fetch_retry(driving_side_query(lat, lon)))
    if side is None:
        print("[warn] no relation states driving_side; the layer stays unknown", file=sys.stderr)

    payload = {
        "version": reply.get("version"),
        "generator": reply.get("generator"),
        "attribution": ATTRIBUTION,
        "bbox": list(bbox),
        "driving_side": side,
        "elements": ways,
        "control_nodes": nodes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[ok] {out} {summarise(ways, nodes)} driving_side={side}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
