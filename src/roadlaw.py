"""The legal road layer: what OpenStreetMap says is allowed, sampled onto the track.

`fetch_roads.py` stores the highways around a circuit as raw Overpass JSON
(`<stem>_osm_highways.json`). This module turns those tags into per-centerline
arrays the reward can charge against:

  * `limit_mps` - the posted speed limit,
  * `lanes` / `oneway` - how much road the car is entitled to and in which
    direction it runs,
  * `driving_side` - which side the law keeps you on,
  * `way_id` / `match_m` - which surveyed way each sample came from and how far
    away it was, so a bad match is visible rather than silent.

Nothing here invents a number. A tag that carries no usable value stays
unknown (`nan` for speeds, `0` for lane counts, `-1` for oneway), and a way
with no `maxspeed` falls back only to the *median of its own highway class as
observed in this very file* - never to a literal written in this source. A
class nobody in the file has tagged has no default at all. That keeps the law a
property of the survey: change the country and the limits change with it.

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

KMH_TO_MPS = 1000.0 / 3600.0
MPH_TO_MPS = 1609.344 / 3600.0
KNOT_TO_MPS = 1852.0 / 3600.0

# Unit suffixes OSM writes after a maxspeed number. The bare number is km/h by
# the tag's definition, so it needs no entry.
SPEED_UNITS = {
    "km/h": KMH_TO_MPS,
    "kmh": KMH_TO_MPS,
    "kph": KMH_TO_MPS,
    "mph": MPH_TO_MPS,
    "knots": KNOT_TO_MPS,
    "knot": KNOT_TO_MPS,
}

ONEWAY_FORWARD = {"yes", "true", "1"}
ONEWAY_REVERSE = {"-1", "reverse"}
ONEWAY_NONE = {"no", "false", "0"}

UNKNOWN_SPEED = float("nan")
UNKNOWN_LANES = 0
UNKNOWN_ONEWAY = -1  # distinct from the reverse direction, which is stored as 2
ONEWAY_CODE = {1: 1, -1: 2, 0: 0}  # parse result -> stored code


def parse_maxspeed(raw: str | float | None) -> float | None:
    """Metres per second from an OSM `maxspeed` value, or None when it names no number.

    `none`, `signals`, `walk` and the country-code forms (`RO:urban`) all carry
    a real-world meaning that depends on data this module does not have, so
    they are reported as unknown rather than resolved to a guess.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    # A variable limit lists several values ("50; 70"); the first is the one
    # posted at the start of the way.
    text = text.split(";")[0].strip()
    parts = text.split()
    number = parts[0]
    unit = " ".join(parts[1:]).strip()
    try:
        value = float(number)
    except ValueError:
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    if not unit:
        return value * KMH_TO_MPS
    factor = SPEED_UNITS.get(unit)
    return None if factor is None else value * factor


def parse_lanes(raw: str | float | None) -> int | None:
    """Whole lanes from an OSM `lanes` value, or None when it names none.

    A fractional count (`2.5`, used where a lane starts mid-way) is floored:
    the car can only rely on the lanes that run the whole length.
    """
    if raw is None:
        return None
    text = str(raw).strip().split(";")[0].strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    lanes = int(np.floor(value))
    return lanes if lanes > 0 else None


def parse_oneway(raw: str | None) -> int | None:
    """1 forward, -1 against the way's direction, 0 two-way, None when untagged."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ONEWAY_FORWARD:
        return 1
    if text in ONEWAY_REVERSE:
        return -1
    if text in ONEWAY_NONE:
        return 0
    return None


def class_speed_defaults(elements: Iterable[dict]) -> dict[str, float]:
    """Median posted limit (m/s) per `highway` class, measured in this file.

    A way without `maxspeed` inherits the limit its neighbours of the same
    class actually carry here. A class with no tagged member gets no entry, and
    its ways stay unknown.
    """
    by_class: dict[str, list[float]] = {}
    for element in elements:
        tags = element.get("tags") or {}
        highway = tags.get("highway")
        if not highway:
            continue
        speed = parse_maxspeed(tags.get("maxspeed"))
        if speed is not None:
            by_class.setdefault(str(highway), []).append(speed)
    # `np.median` of an even-length list averages the middle pair and could
    # produce a speed nobody posted; the lower middle value is always one that
    # appears in the survey.
    return {name: float(sorted(values)[(len(values) - 1) // 2]) for name, values in by_class.items()}


@dataclass(frozen=True)
class Way:
    """One surveyed highway, projected into track metres."""

    way_id: int
    highway: str
    points: np.ndarray  # (k, 2) metres in the track frame
    limit_mps: float  # nan when neither the tag nor the class median gives one
    lanes: int  # 0 when untagged
    oneway: int  # 0 two-way, 1 forward, 2 against the way, -1 untagged
    tunnel: bool
    roundabout: bool


@dataclass(frozen=True)
class LegalProfile:
    """The law along one circuit, one entry per centerline sample."""

    limit_mps: np.ndarray  # (n,) float, nan where the survey posts nothing
    lanes: np.ndarray  # (n,) int, 0 where untagged
    oneway: np.ndarray  # (n,) int, -1 where untagged
    tunnel: np.ndarray  # (n,) bool
    roundabout: np.ndarray  # (n,) bool
    way_id: np.ndarray  # (n,) int, -1 where no way was near enough
    match_m: np.ndarray  # (n,) float, distance to that way
    driving_side: str | None  # "left", "right", or None when the survey omits it
    source: str

    @property
    def known_limit_fraction(self) -> float:
        return float(np.isfinite(self.limit_mps).mean())


@dataclass(frozen=True)
class ControlPoint:
    """A place on the road where a rule bites: a signal, a stop line, a crossing."""

    node_id: int
    rules: tuple[str, ...]  # rule ids from the law corpus this node triggers
    pos: np.ndarray  # (2,) metres in the track frame
    tags: dict
    progress: float = -1.0  # where along the lap it sits, 0..1; -1 when not placed
    offset_m: float = float("inf")  # how far it stands from the centerline


def parse_selector(selector: str) -> tuple[str, str | None] | None:
    """`osm:highway=traffic_signals` -> ("highway", "traffic_signals"); `osm:maxspeed` -> ("maxspeed", None).

    The law corpus states where each rule is measured; this reads that
    statement so the tag mapping lives with the law, not in this module.
    """
    if not selector.startswith("osm:"):
        return None
    body = selector[len("osm:") :]
    key, _, value = body.partition("=")
    key = key.strip()
    return (key, value.strip() or None) if key else None


def rules_triggered_by(tags: dict, corpus) -> tuple[str, ...]:
    """Which rules of `corpus` this node's tags put in play."""
    hits: list[str] = []
    for rule in corpus:
        for key, selector in rule.applies_to.items():
            if not key.endswith("_source") or not isinstance(selector, str):
                continue
            parsed = parse_selector(selector)
            if parsed is None:
                continue
            tag_key, tag_value = parsed
            if tag_key in tags and (tag_value is None or str(tags[tag_key]) == tag_value):
                hits.append(rule.rule_id)
                break
    return tuple(hits)


def control_points_for_circuit(
    geojson: str | Path,
    proj,
    corpus,
    centerline: np.ndarray | None = None,
    max_offset_m: float = 25.0,
) -> tuple[ControlPoint, ...]:
    """The rule-bearing nodes of the survey, projected into the track frame.

    A node the law corpus has no rule for is dropped: the driver can only be
    judged by rules the corpus carries. With a `centerline`, the region's nodes
    are cut down to those standing within `max_offset_m` of the lap and each
    one is placed at the fraction of the lap it governs, which is what lets the
    reward know a signal is ahead rather than merely somewhere in the city.
    """
    path = highways_path(geojson)
    if not path.exists() or proj is None:
        return ()
    data = json.loads(path.read_text())
    raw: list[tuple[int, tuple[str, ...], np.ndarray, dict]] = []
    for node in data.get("control_nodes", []):
        tags = node.get("tags") or {}
        rules = rules_triggered_by(tags, corpus)
        if not rules or node.get("lat") is None:
            continue
        pos = proj.project(np.array([[node["lon"], node["lat"]]], dtype=np.float64))[0]
        raw.append((int(node.get("id", -1)), rules, pos, tags))
    if not raw:
        return ()
    if centerline is None:
        return tuple(ControlPoint(node_id=i, rules=r, pos=p, tags=t) for i, r, p, t in raw)

    line = np.asarray(centerline, dtype=np.float64)
    positions = np.stack([p for _, _, p, _ in raw])
    nearest = np.empty(positions.shape[0], dtype=np.int64)
    offsets = np.empty(positions.shape[0])
    for lo in range(0, positions.shape[0], 512):
        chunk = positions[lo : lo + 512]
        dist = np.linalg.norm(chunk[:, None, :] - line[None, :, :], axis=-1)
        nearest[lo : lo + 512] = dist.argmin(axis=1)
        offsets[lo : lo + 512] = dist.min(axis=1)
    n = line.shape[0]
    return tuple(
        ControlPoint(
            node_id=node_id,
            rules=rules,
            pos=pos,
            tags=tags,
            progress=float(nearest[i]) / n,
            offset_m=float(offsets[i]),
        )
        for i, (node_id, rules, pos, tags) in enumerate(raw)
        if offsets[i] <= max_offset_m
    )


def highways_path(geojson: str | Path) -> Path:
    geojson = Path(geojson)
    return geojson.with_name(geojson.stem + "_osm_highways.json")


def load_highways(path: str | Path, proj, max_gap_m: float = 0.0) -> list[Way]:
    """Surveyed highways from raw Overpass JSON, projected with `proj`.

    `proj` is the circuit's `GeoProjection`, so way coordinates land in the
    same metric frame as the centerline.
    """
    data = json.loads(Path(path).read_text())
    elements = [e for e in data.get("elements", []) if e.get("geometry")]
    defaults = class_speed_defaults(elements)
    ways: list[Way] = []
    for element in elements:
        tags = element.get("tags") or {}
        highway = str(tags.get("highway") or "")
        if not highway:
            continue
        lonlat = np.array([[p["lon"], p["lat"]] for p in element["geometry"]], dtype=np.float64)
        if lonlat.shape[0] < 2:
            continue
        speed = parse_maxspeed(tags.get("maxspeed"))
        if speed is None:
            speed = defaults.get(highway, UNKNOWN_SPEED)
        lanes = parse_lanes(tags.get("lanes"))
        oneway = parse_oneway(tags.get("oneway"))
        ways.append(
            Way(
                way_id=int(element.get("id", -1)),
                highway=highway,
                points=proj.project(lonlat).astype(np.float64),
                limit_mps=float(speed),
                lanes=UNKNOWN_LANES if lanes is None else int(lanes),
                oneway=UNKNOWN_ONEWAY if oneway is None else ONEWAY_CODE[oneway],
                tunnel=bool(tags.get("tunnel")) and str(tags.get("tunnel")).lower() not in ONEWAY_NONE,
                roundabout=str(tags.get("junction") or "").lower() == "roundabout",
            )
        )
    return ways


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of `points` (n, 2) to each segment a->b (m, 2), as (n, m)."""
    ab = b - a  # (m, 2)
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 0.0, denom, 1.0)
    ap = points[:, None, :] - a[None, :, :]  # (n, m, 2)
    t = np.clip(np.einsum("nmj,mj->nm", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


def _nearest_way(centerline: np.ndarray, ways: Sequence[Way], block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For each centerline sample, the index of the closest way and that distance."""
    starts = np.concatenate([w.points[:-1] for w in ways])
    ends = np.concatenate([w.points[1:] for w in ways])
    owner = np.concatenate([np.full(w.points.shape[0] - 1, i, dtype=np.int64) for i, w in enumerate(ways)])
    best_i = np.full(centerline.shape[0], -1, dtype=np.int64)
    best_d = np.full(centerline.shape[0], np.inf)
    for lo in range(0, centerline.shape[0], block):
        chunk = centerline[lo : lo + block]
        dist = _segment_distance(chunk, starts, ends)
        near = dist.argmin(axis=1)
        best_i[lo : lo + block] = owner[near]
        best_d[lo : lo + block] = dist[np.arange(chunk.shape[0]), near]
    return best_i, best_d


def driving_side_of(data: dict) -> str | None:
    """The side the law keeps you on, as recorded by the fetch, or None.

    Stored by `fetch_roads.py` from the enclosing boundary relation's
    `driving_side` tag. Absent means unknown: no continent is assumed here.
    """
    side = str(data.get("driving_side") or "").strip().lower()
    return side if side in ("left", "right") else None


def legal_profile_for_circuit(
    geojson: str | Path,
    centerline: np.ndarray,
    proj,
    max_match_m: float = 40.0,
) -> LegalProfile | None:
    """The legal layer for a circuit from the highway survey beside its GeoJSON, or None without one.

    `max_match_m` bounds how far a centerline sample may be from a mapped way
    before it is treated as off-survey; those samples stay unknown instead of
    inheriting a distant road's limit.
    """
    path = highways_path(geojson)
    if not path.exists() or proj is None:
        return None
    data = json.loads(path.read_text())
    ways = load_highways(path, proj)
    if not ways:
        return None

    centerline = np.asarray(centerline, dtype=np.float64)
    index, distance = _nearest_way(centerline, ways)
    matched = distance <= max_match_m

    def gather(attr: str, dtype, unknown):
        values = np.array([getattr(w, attr) for w in ways], dtype=dtype)
        out = np.full(centerline.shape[0], unknown, dtype=dtype)
        out[matched] = values[index[matched]]
        return out

    return LegalProfile(
        limit_mps=gather("limit_mps", np.float64, UNKNOWN_SPEED),
        lanes=gather("lanes", np.int64, UNKNOWN_LANES),
        oneway=gather("oneway", np.int64, UNKNOWN_ONEWAY),
        tunnel=gather("tunnel", bool, False),
        roundabout=gather("roundabout", bool, False),
        way_id=np.where(matched, np.array([w.way_id for w in ways], dtype=np.int64)[index], -1),
        match_m=distance,
        driving_side=driving_side_of(data),
        source=str(path),
    )
