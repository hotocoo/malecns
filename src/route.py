"""Build a drivable circuit out of the real road network.

The circuits this project drove until now were either a hand-drawn loop or a
traced Grand Prix outline. A Malaysian training route is neither: it is a
closed run over roads that exist, through the junctions they actually meet at,
with the direction of travel the survey says is legal.

  python3 src/route.py --survey data/tracks/kl_osm_highways.json \
      --out data/tracks/kl.geojson --length-km 6 --seed 0

The route is found on the directed road graph: out to roughly half the target
length, then back by a different way, so the loop is a real circuit rather than
a there-and-back. One-way ways are followed in their legal direction only,
which is what makes the result drivable rather than merely connected.

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from roadlaw import ONEWAY_CODE, parse_oneway

EARTH_RADIUS_M = 6_371_000.0

# Classes a training route may use. Service roads and tracks connect car parks
# and plantations rather than carrying through traffic, so a route is not built
# along them; they stay in the survey for the law and the scenery.
ROUTE_CLASSES = frozenset(
    {
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "unclassified", "residential", "living_street",
    }
)


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Metres between two (lon, lat) points."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


@dataclass(frozen=True)
class Edge:
    """One legal hop between two surveyed nodes."""

    to: int
    way_id: int
    length_m: float


@dataclass(frozen=True)
class RoadGraph:
    """The drivable road network of one region, as a directed graph."""

    coords: dict[int, tuple[float, float]]
    out_edges: dict[int, tuple[Edge, ...]]
    way_tags: dict[int, dict]

    @property
    def node_count(self) -> int:
        return len(self.coords)

    @property
    def edge_count(self) -> int:
        return sum(len(e) for e in self.out_edges.values())

    def neighbours(self, node: int) -> tuple[Edge, ...]:
        return self.out_edges.get(node, ())


def build_graph(payload: dict, classes: frozenset[str] = ROUTE_CLASSES) -> RoadGraph:
    """The directed graph of the survey's drivable ways.

    A way tagged `oneway=yes` contributes forward edges only, `oneway=-1`
    backward only; anything else is traversable both ways. That is the whole
    reason the graph is directed: it is the survey's legality, not a modelling
    choice made here.
    """
    coords: dict[int, tuple[float, float]] = {}
    out_edges: dict[int, list[Edge]] = {}
    way_tags: dict[int, dict] = {}

    for way in payload.get("elements", []):
        tags = way.get("tags") or {}
        if tags.get("highway") not in classes:
            continue
        geometry = way.get("geometry") or []
        refs = way.get("nodes") or []
        if len(geometry) < 2 or len(refs) != len(geometry):
            continue
        way_id = int(way.get("id", -1))
        way_tags[way_id] = tags
        oneway = ONEWAY_CODE.get(parse_oneway(tags.get("oneway")) or 0, 0)
        for ref, point in zip(refs, geometry):
            coords[ref] = (float(point["lon"]), float(point["lat"]))
        for first, second in zip(refs, refs[1:]):
            length = haversine_m(coords[first], coords[second])
            if length <= 0.0:
                continue
            if oneway != 2:  # not reverse-only
                out_edges.setdefault(first, []).append(Edge(second, way_id, length))
            if oneway != 1:  # not forward-only
                out_edges.setdefault(second, []).append(Edge(first, way_id, length))

    return RoadGraph(coords=coords, out_edges={k: tuple(v) for k, v in out_edges.items()}, way_tags=way_tags)


def shortest_paths(graph: RoadGraph, source: int, blocked: set[tuple[int, int]] | None = None) -> tuple[dict[int, float], dict[int, int]]:
    """Dijkstra from `source`, optionally refusing a set of directed hops."""
    blocked = blocked or set()
    dist: dict[int, float] = {source: 0.0}
    prev: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        d, node = heapq.heappop(queue)
        if d > dist.get(node, math.inf):
            continue
        for edge in graph.neighbours(node):
            if (node, edge.to) in blocked:
                continue
            nd = d + edge.length_m
            if nd < dist.get(edge.to, math.inf):
                dist[edge.to] = nd
                prev[edge.to] = node
                heapq.heappush(queue, (nd, edge.to))
    return dist, prev


def path_between(prev: dict[int, int], source: int, target: int) -> list[int]:
    """The node sequence `source` -> `target`, or [] when `prev` has no route."""
    if source == target:
        return [source]
    path = [target]
    while path[-1] != source:
        step = prev.get(path[-1])
        if step is None:
            return []
        path.append(step)
    path.reverse()
    return path


def path_length_m(graph: RoadGraph, nodes: list[int]) -> float:
    return sum(haversine_m(graph.coords[a], graph.coords[b]) for a, b in zip(nodes, nodes[1:]))


def find_circuit(
    graph: RoadGraph,
    target_m: float,
    seed: int | None = None,
    rng: random.Random | None = None,
    tolerance: float = 0.35,
    attempts: int = 200,
) -> list[int]:
    """A closed drivable loop of roughly `target_m`, as a node sequence.

    Out to a node about half the target away, back by a route that may not
    reuse the outbound hops. Both legs obey the survey's directions, so the
    loop can be driven, not merely traced.
    """
    rng = rng or random.Random(0)
    candidates = [n for n, edges in graph.out_edges.items() if len(edges) >= 2]
    if not candidates:
        raise ValueError("the survey has no node with two ways out; nothing to drive")

    best: list[int] = []
    best_error = math.inf
    for _ in range(attempts):
        start = seed if seed is not None and not best else rng.choice(candidates)
        out_dist, out_prev = shortest_paths(graph, start)
        half = target_m / 2.0
        reachable = [n for n, d in out_dist.items() if abs(d - half) < half * tolerance and n != start]
        if not reachable:
            seed = None
            continue
        far = rng.choice(reachable)
        outbound = path_between(out_prev, start, far)
        if len(outbound) < 2:
            seed = None
            continue
        used = set(zip(outbound, outbound[1:])) | set(zip(outbound[1:], outbound))
        _, back_prev = shortest_paths(graph, far, blocked=used)
        inbound = path_between(back_prev, far, start)
        if len(inbound) < 2:
            seed = None
            continue
        loop = outbound + inbound[1:]
        error = abs(path_length_m(graph, loop) - target_m) / target_m
        if error < best_error:
            best, best_error = loop, error
        if best_error <= tolerance:
            break
        seed = None
    if not best:
        raise ValueError(f"no closed route near {target_m:.0f} m found in this survey")
    return best


def circuit_geojson(graph: RoadGraph, nodes: list[int], name: str) -> dict:
    """The loop as the LineString GeoJSON the rest of the pipeline reads."""
    coordinates = [list(graph.coords[n]) for n in nodes]
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    named = {
        graph.way_tags.get(edge.way_id, {}).get("name")
        for first, second in zip(nodes, nodes[1:])
        for edge in graph.neighbours(first)
        if edge.to == second
    }
    ways = sorted(name for name in named if name)
    return {
        "type": "FeatureCollection",
        "attribution": "(c) OpenStreetMap contributors, ODbL 1.0, https://www.openstreetmap.org/copyright",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "length_m": round(path_length_m(graph, nodes), 1),
                    "nodes": len(nodes),
                    "roads": ways,
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, required=True, help="a *_osm_highways.json region")
    parser.add_argument("--out", type=Path, required=True, help="where the circuit GeoJSON goes")
    parser.add_argument("--length-km", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0, help="random seed for the search")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.survey.read_text())
    graph = build_graph(payload)
    print(f"[graph] {graph.node_count} nodes, {graph.edge_count} directed edges")
    loop = find_circuit(graph, args.length_km * 1000.0, rng=random.Random(args.seed))
    name = args.name or args.out.stem
    feature = circuit_geojson(graph, loop, name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(feature, separators=(",", ":")))
    props = feature["features"][0]["properties"]
    print(f"[ok] {args.out} {props['length_m'] / 1000.0:.2f} km over {len(props['roads'])} named roads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
