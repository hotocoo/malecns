"""Pull the whole Malaysian road network from OpenStreetMap and cut driving regions out of it.

Overpass serves a neighbourhood, not a country. For all of Malaysia the real
source is the Geofabrik extract, a complete daily rebuild of the OSM database
for the region, which this module downloads, verifies against the publisher's
own MD5, and filters down to what a driver needs:

  * every drivable `highway` way with its legal tags,
  * the control nodes that carry a rule at a place (signals, stop and give-way
    lines, crossings, cameras, level crossings, signed limits).

  python3 src/fetch_malaysia_osm.py --filter
  python3 src/fetch_malaysia_osm.py --export --bbox 3.10,101.65,3.18,101.72 \
      --out data/tracks/kl_osm_highways.json

The export writes exactly the schema `roadlaw.py` reads, so a Malaysian region
and the Monaco survey are interchangeable to everything downstream.

Data (c) OpenStreetMap contributors, ODbL 1.0. Extract by Geofabrik GmbH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import osmium

REGION = "asia/malaysia-singapore-brunei"
BASE = "https://download.geofabrik.de"
OSM_DIR = Path("data/osm")
COUNTRY_PBF = OSM_DIR / "malaysia-singapore-brunei-latest.osm.pbf"
ROADS_PBF = OSM_DIR / "malaysia_roads.osm.pbf"
ATTRIBUTION = "(c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright"

# Highway classes a car may lawfully drive on. Service roads and tracks are
# included because Malaysian kampung and industrial routes are tagged that way;
# footways, cycleways and steps are not roads for a car and are dropped.
DRIVABLE = frozenset(
    {
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "unclassified", "residential", "living_street",
        "service", "road", "track", "busway",
    }
)

# Node tags that carry a rule at a place. Same set the Overpass path fetches,
# so a region cut from the country file and one fetched live agree.
CONTROL_KEYS = ("traffic_calming", "traffic_sign", "maxspeed", "crossing")
CONTROL_VALUES = {
    "highway": {"traffic_signals", "stop", "give_way", "crossing", "speed_camera", "mini_roundabout"},
    "barrier": {"toll_booth"},
    "railway": {"level_crossing"},
}

# Way tags the law reads. Keeping only these shrinks a country of roads to
# something a training run can load, and drops nothing the reward consults.
LEGAL_KEYS = (
    "highway", "maxspeed", "maxspeed:type", "lanes", "lanes:forward", "lanes:backward",
    "lanes:psv", "oneway", "junction", "tunnel", "bridge", "layer", "surface", "lit",
    "width", "shoulder", "sidewalk", "cycleway", "overtaking", "access", "motor_vehicle",
    "turn:lanes", "traffic_calming", "bus", "psv", "name", "ref", "toll",
)


def is_control_node(tags) -> bool:
    for key, values in CONTROL_VALUES.items():
        if tags.get(key) in values:
            return True
    return any(key in tags for key in CONTROL_KEYS)


def legal_tags(tags) -> dict:
    return {k: tags[k] for k in LEGAL_KEYS if k in tags}


def download(force: bool = False) -> Path:
    """The country extract, verified against Geofabrik's published MD5."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{REGION}-latest.osm.pbf"
    md5_url = f"{url}.md5"
    expected = subprocess.run(
        ["curl", "-sSL", "-m", "60", md5_url], capture_output=True, text=True, check=False
    ).stdout.split()
    if not expected:
        raise SystemExit(f"could not read the publisher's checksum from {md5_url}")
    want = expected[0]
    if COUNTRY_PBF.exists() and not force and md5_of(COUNTRY_PBF) == want:
        print(f"[skip] {COUNTRY_PBF} ({COUNTRY_PBF.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)
        return COUNTRY_PBF
    print(f"[get ] {url}", file=sys.stderr)
    result = subprocess.run(["curl", "-SL", "--retry", "3", "-o", str(COUNTRY_PBF), url], check=False)
    if result.returncode != 0:
        raise SystemExit(f"download failed (curl {result.returncode})")
    got = md5_of(COUNTRY_PBF)
    if got != want:
        raise SystemExit(f"{COUNTRY_PBF}: md5 {got} != published {want}")
    print(f"[ok  ] {COUNTRY_PBF} ({COUNTRY_PBF.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)
    return COUNTRY_PBF


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching the publisher's own checksum, not a security use
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_roads(src: Path = COUNTRY_PBF, dst: Path = ROADS_PBF) -> Path:
    """Write a second PBF holding only drivable ways, their nodes and the control nodes."""
    keep_nodes: set[int] = set()
    ways: list[tuple[int, list[int], dict]] = []
    for obj in osmium.FileProcessor(str(src)).with_filter(osmium.filter.KeyFilter("highway")):
        if obj.type_str() != "w" or obj.tags.get("highway") not in DRIVABLE:
            continue
        refs = [n.ref for n in obj.nodes]
        if len(refs) < 2:
            continue
        ways.append((obj.id, refs, legal_tags(obj.tags)))
        keep_nodes.update(refs)
    print(f"[filter] {len(ways)} drivable ways, {len(keep_nodes)} nodes", file=sys.stderr)

    writer = osmium.SimpleWriter(str(dst), overwrite=True)
    controls = 0
    for obj in osmium.FileProcessor(str(src)):
        if obj.type_str() != "n":
            continue
        tagged = is_control_node(obj.tags)
        if obj.id not in keep_nodes and not tagged:
            continue
        controls += tagged
        writer.add_node(obj.replace(tags=dict(obj.tags) if tagged else {}))
    for way_id, refs, tags in ways:
        writer.add_way(osmium.osm.mutable.Way(id=way_id, nodes=refs, tags=tags))
    writer.close()
    print(f"[ok] {dst} ({dst.stat().st_size / 1e6:.0f} MB), {controls} control nodes", file=sys.stderr)
    return dst


def export_region(
    bbox: tuple[float, float, float, float],
    out: Path,
    src: Path = ROADS_PBF,
    driving_side: str | None = None,
) -> dict:
    """Cut a bounding box out of the filtered country file into `roadlaw`'s schema."""
    south, west, north, east = bbox
    inside = lambda lat, lon: south <= lat <= north and west <= lon <= east  # noqa: E731

    elements: list[dict] = []
    for obj in osmium.FileProcessor(str(src)).with_locations().with_filter(osmium.filter.KeyFilter("highway")):
        if obj.type_str() != "w":
            continue
        geometry = [
            {"lat": n.location.lat, "lon": n.location.lon}
            for n in obj.nodes
            if n.location.valid()
        ]
        if len(geometry) < 2 or not any(inside(p["lat"], p["lon"]) for p in geometry):
            continue
        elements.append(
            {
                "type": "way",
                "id": obj.id,
                "geometry": geometry,
                "nodes": [n.ref for n in obj.nodes],
                "tags": dict(obj.tags),
            }
        )

    control_nodes = [
        {"type": "node", "id": obj.id, "lat": obj.location.lat, "lon": obj.location.lon, "tags": dict(obj.tags)}
        for obj in osmium.FileProcessor(str(src))
        if obj.type_str() == "n" and obj.tags and obj.location.valid() and inside(obj.location.lat, obj.location.lon)
    ]

    payload = {
        "generator": f"fetch_malaysia_osm from {src.name}",
        "attribution": ATTRIBUTION,
        "bbox": list(bbox),
        "driving_side": driving_side,
        "elements": elements,
        "control_nodes": control_nodes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[ok] {out} {len(elements)} ways, {len(control_nodes)} control nodes", file=sys.stderr)
    return payload


def centre_of(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    south, west, north, east = bbox
    return (south + north) / 2.0, (west + east) / 2.0


def surveyed_driving_side(bbox: tuple[float, float, float, float]) -> str | None:
    """The driving side for this region, read from OSM's own boundary tags."""
    from fetch_roads import driving_side_from, driving_side_query, fetch_retry

    lat, lon = centre_of(bbox)
    return driving_side_from(fetch_retry(driving_side_query(lat, lon)))


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be south,west,north,east")
    south, west, north, east = parts
    if south >= north or west >= east:
        raise argparse.ArgumentTypeError("bbox must be south,west,north,east with south<north and west<east")
    return south, west, north, east


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch the country extract")
    parser.add_argument("--filter", action="store_true", help="reduce it to drivable roads and control nodes")
    parser.add_argument("--export", action="store_true", help="cut a region into roadlaw's schema")
    parser.add_argument("--bbox", type=parse_bbox, help="south,west,north,east")
    parser.add_argument("--out", type=Path, help="where the region goes")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.download or args.filter or not (args.export):
        download(args.force)
    if args.filter:
        filter_roads()
    if args.export:
        if args.bbox is None or args.out is None:
            parser.error("--export needs --bbox and --out")
        export_region(args.bbox, args.out, driving_side=surveyed_driving_side(args.bbox))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
