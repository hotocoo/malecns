"""Tests for building a drivable circuit out of the surveyed road network.

A route is only useful if it can actually be driven: every hop legal in the
direction taken, the loop closed, and the geometry the survey's own, never
smoothed into something that is not a road.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from route import (  # noqa: E402
    Edge,
    RoadGraph,
    build_graph,
    circuit_geojson,
    find_circuit,
    haversine_m,
    path_between,
    path_length_m,
    shortest_paths,
)

SURVEY = ROOT / "data" / "tracks" / "kl_osm_highways.json"
needs_survey = pytest.mark.skipif(not SURVEY.exists(), reason="Malaysian survey not exported")


def grid_survey() -> dict:
    """A tiny hand-made survey: a square of four two-way ways plus a one-way spur."""
    corners = {1: (101.0, 3.0), 2: (101.001, 3.0), 3: (101.001, 3.001), 4: (101.0, 3.001)}

    def way(way_id, refs, tags):
        return {
            "type": "way",
            "id": way_id,
            "nodes": list(refs),
            "geometry": [{"lon": corners[r][0], "lat": corners[r][1]} for r in refs],
            "tags": tags,
        }

    return {
        "elements": [
            way(10, (1, 2), {"highway": "residential", "name": "South"}),
            way(11, (2, 3), {"highway": "residential", "name": "East"}),
            way(12, (3, 4), {"highway": "residential", "name": "North"}),
            way(13, (4, 1), {"highway": "residential", "oneway": "yes", "name": "West"}),
            way(14, (1, 2), {"highway": "footway", "name": "Not a road"}),
        ]
    }


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_m((101.0, 3.0), (101.0, 3.0)) == pytest.approx(0.0)

    def test_a_thousandth_of_a_degree_is_about_a_hundred_metres(self):
        d = haversine_m((101.0, 3.0), (101.001, 3.0))
        assert 100.0 < d < 120.0

    def test_symmetric(self):
        a, b = (101.0, 3.0), (101.002, 3.003)
        assert haversine_m(a, b) == pytest.approx(haversine_m(b, a))


class TestBuildGraph:
    def test_a_two_way_road_is_traversable_both_ways(self):
        graph = build_graph(grid_survey())
        assert any(e.to == 2 for e in graph.neighbours(1))
        assert any(e.to == 1 for e in graph.neighbours(2))

    def test_a_oneway_road_is_traversable_one_way_only(self):
        graph = build_graph(grid_survey())
        assert any(e.to == 1 for e in graph.neighbours(4))  # West runs 4 -> 1
        assert not any(e.to == 4 and e.way_id == 13 for e in graph.neighbours(1))

    def test_a_footway_is_not_a_road(self):
        graph = build_graph(grid_survey())
        assert 14 not in graph.way_tags

    def test_edges_carry_their_length(self):
        graph = build_graph(grid_survey())
        for edges in graph.out_edges.values():
            for edge in edges:
                assert edge.length_m > 0.0

    def test_way_tags_are_kept_for_naming(self):
        graph = build_graph(grid_survey())
        assert graph.way_tags[10]["name"] == "South"


class TestShortestPaths:
    def test_reaches_every_node_of_a_connected_square(self):
        graph = build_graph(grid_survey())
        dist, _ = shortest_paths(graph, 1)
        assert set(dist) == {1, 2, 3, 4}

    def test_blocked_hops_are_not_used(self):
        """With West one-way the wrong way, blocking the South hop strands node 2."""
        graph = build_graph(grid_survey())
        _, prev = shortest_paths(graph, 1, blocked={(1, 2), (2, 1)})
        assert path_between(prev, 1, 2) == []

    def test_a_blocked_hop_is_routed_around_when_a_legal_way_exists(self):
        survey = grid_survey()
        for way in survey["elements"]:
            way["tags"].pop("oneway", None)  # make West two-way
        graph = build_graph(survey)
        _, prev = shortest_paths(graph, 1, blocked={(1, 2), (2, 1)})
        path = path_between(prev, 1, 2)
        assert path == [1, 4, 3, 2]

    def test_path_between_returns_empty_without_a_route(self):
        assert path_between({}, 1, 9) == []

    def test_path_to_itself_is_one_node(self):
        assert path_between({}, 1, 1) == [1]


class TestFindCircuit:
    def test_the_loop_closes(self):
        graph = build_graph(grid_survey())
        loop = find_circuit(graph, 400.0, rng=random.Random(0), tolerance=0.9)
        assert loop[0] == loop[-1]

    def test_every_hop_of_the_loop_is_legal(self):
        graph = build_graph(grid_survey())
        loop = find_circuit(graph, 400.0, rng=random.Random(0), tolerance=0.9)
        for first, second in zip(loop, loop[1:]):
            assert any(e.to == second for e in graph.neighbours(first)), f"{first}->{second}"

    def test_a_graph_with_no_junction_is_refused(self):
        graph = RoadGraph(coords={1: (0.0, 0.0)}, out_edges={}, way_tags={})
        with pytest.raises(ValueError):
            find_circuit(graph, 1000.0)

    def test_an_impossible_length_is_refused_not_faked(self):
        graph = build_graph(grid_survey())
        with pytest.raises(ValueError):
            find_circuit(graph, 5_000_000.0, rng=random.Random(0), attempts=5)


class TestCircuitGeoJSON:
    def test_shape_matches_what_the_pipeline_loads(self):
        graph = build_graph(grid_survey())
        loop = find_circuit(graph, 400.0, rng=random.Random(0), tolerance=0.9)
        feature = circuit_geojson(graph, loop, "grid")["features"][0]
        assert feature["geometry"]["type"] == "LineString"
        assert len(feature["geometry"]["coordinates"]) >= 4

    def test_coordinates_are_the_surveys_own(self):
        graph = build_graph(grid_survey())
        loop = find_circuit(graph, 400.0, rng=random.Random(0), tolerance=0.9)
        coords = circuit_geojson(graph, loop, "grid")["features"][0]["geometry"]["coordinates"]
        for point in coords:
            assert tuple(point) in set(graph.coords.values())

    def test_it_records_the_roads_it_runs_on(self):
        graph = build_graph(grid_survey())
        loop = find_circuit(graph, 400.0, rng=random.Random(0), tolerance=0.9)
        props = circuit_geojson(graph, loop, "grid")["features"][0]["properties"]
        assert props["roads"]
        assert props["length_m"] > 0.0


@needs_survey
class TestRealSurvey:
    @pytest.fixture(scope="class")
    def graph(self):
        return build_graph(json.loads(SURVEY.read_text()))

    def test_the_region_has_a_road_network(self, graph):
        assert graph.node_count > 1000
        assert graph.edge_count > graph.node_count

    def test_a_circuit_of_the_asked_length_is_found(self, graph):
        loop = find_circuit(graph, 6000.0, rng=random.Random(0))
        length = path_length_m(graph, loop)
        assert 3000.0 < length < 9000.0

    def test_the_circuit_only_uses_legal_directions(self, graph):
        loop = find_circuit(graph, 6000.0, rng=random.Random(0))
        for first, second in zip(loop, loop[1:]):
            assert any(e.to == second for e in graph.neighbours(first))

    def test_the_circuit_loads_as_a_centerline(self, graph, tmp_path):
        from car_env import load_geojson_centerline

        loop = find_circuit(graph, 6000.0, rng=random.Random(0))
        path = tmp_path / "loop.geojson"
        path.write_text(json.dumps(circuit_geojson(graph, loop, "loop")))
        pts = load_geojson_centerline(path)
        assert pts.shape[1] == 2
        assert pts.shape[0] > 100
