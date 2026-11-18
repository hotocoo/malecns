"""The street is built as a street: joined carriageways, kerbs, paint that stops at junctions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadlaw import ControlPoint  # noqa: E402
from scene import RoadMesh, SceneConfig, build_network  # noqa: E402
from streets import (  # noqa: E402
    MAT_ASPHALT,
    MAT_FLAT,
    MAT_KERB,
    MAT_PAVEMENT,
    StreetConfig,
    Way,
    build_streets,
    crossing_markings,
    junction_zones,
)


class FlatProjection:
    """Degrees in, metres out: one degree is one metre, so tests can think in metres."""

    def project(self, lonlat: np.ndarray) -> np.ndarray:
        return np.asarray(lonlat, dtype=np.float64)


def way(points, highway="residential", lanes=2, oneway=False, nodes=None) -> Way:
    points = np.asarray(points, dtype=np.float64)
    cfg = SceneConfig()
    if nodes is None:
        nodes = [hash((highway, tuple(p))) & 0xFFFFFFF for p in map(tuple, points)]
    return Way(points, list(nodes), highway, lanes, oneway, lanes * cfg.lane_width_m / 2.0 + cfg.shoulder_m)


def crossroads() -> list[Way]:
    """An east-west street crossed at the origin by a north-south one; node 1 is shared."""
    ew = way([(-100, 0), (0, 0), (100, 0)], nodes=[10, 1, 11])
    ns = way([(0, -100), (0, 0), (0, 100)], nodes=[20, 1, 21])
    return [ew, ns]


def triangles_of(mesh, material) -> np.ndarray:
    pick = mesh.materials == material
    return mesh.vertices[pick].reshape(-1, 3, 3)


def covers(tris: np.ndarray, point: np.ndarray) -> bool:
    """Whether any triangle (seen from above) contains `point`."""
    a, b, c = tris[:, 0, :2], tris[:, 1, :2], tris[:, 2, :2]
    p = np.asarray(point, dtype=np.float64)

    def side(u, v):
        return (v[:, 0] - u[:, 0]) * (p[1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (p[0] - u[:, 0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    return bool(inside.any())


class TestCarriageway:
    def test_the_junction_is_one_surface(self):
        mesh = build_streets(crossroads(), 0.0, 3.5, 0.5, 0.12)
        asphalt = triangles_of(mesh, MAT_ASPHALT)
        assert covers(asphalt, (0.0, 0.0))
        assert covers(asphalt, (3.0, 3.0))  # the corner of the junction, inside both arms' width
        assert not covers(asphalt, (12.0, 12.0))

    def test_all_asphalt_lies_on_the_ground(self):
        mesh = build_streets(crossroads(), 0.0, 3.5, 0.5, 0.12)
        assert np.allclose(triangles_of(mesh, MAT_ASPHALT)[:, :, 2], 0.0)

    def test_edge_lines_stop_at_the_junction(self):
        mesh = build_streets(crossroads(), 0.0, 3.5, 0.5, 0.12)
        paint = mesh.vertices[mesh.materials == MAT_FLAT]
        assert paint.shape[0] > 0
        # Nothing painted crosses the mouth of the side road: no paint vertex
        # sits on the east-west edge line inside the north-south road's width.
        edge_y = 3.5 - 0.5  # the inset edge of a two-lane road
        on_edge = np.abs(np.abs(paint[:, 1]) - edge_y) < 0.2
        inside_mouth = np.abs(paint[:, 0]) < 3.5
        assert not (on_edge & inside_mouth).any()

    def test_centre_line_is_cut_out_of_the_junction(self):
        mesh = build_streets(crossroads(), 0.0, 3.5, 0.5, 0.12)
        paint = mesh.vertices[mesh.materials == MAT_FLAT]
        centre = np.abs(paint[:, 1]) < 0.1  # on the east-west centre line
        assert centre.any(), "a two-way street has a centre line"
        assert np.abs(paint[centre, 0]).min() > 4.0  # none within the junction disc

    def test_one_way_street_has_no_centre_line(self):
        mesh = build_streets([way([(-100, 0), (100, 0)], oneway=True, lanes=1)], 0.0, 3.5, 0.5, 0.12)
        paint = mesh.vertices[mesh.materials == MAT_FLAT]
        assert not (np.abs(paint[:, 1]) < 0.1).any()


class TestKerbsAndPavements:
    def test_a_residential_street_gets_a_kerb_and_pavement_both_sides(self):
        mesh = build_streets([way([(-100, 0), (100, 0)])], 0.0, 3.5, 0.5, 0.12, StreetConfig())
        pavement = triangles_of(mesh, MAT_PAVEMENT)
        kerb = triangles_of(mesh, MAT_KERB)
        assert covers(pavement, (0.0, 5.0)) and covers(pavement, (0.0, -5.0))
        assert covers(kerb, (0.0, 4.1)) and covers(kerb, (0.0, -4.1))

    def test_the_pavement_stands_a_kerb_above_the_road(self):
        cfg = StreetConfig()
        mesh = build_streets([way([(-100, 0), (100, 0)])], 0.0, 3.5, 0.5, 0.12, cfg)
        pavement = triangles_of(mesh, MAT_PAVEMENT)
        assert np.allclose(pavement[:, :, 2], cfg.kerb_height_m)

    def test_a_service_road_has_neither(self):
        mesh = build_streets([way([(-100, 0), (100, 0)], highway="service", lanes=1)], 0.0, 3.5, 0.5, 0.12)
        assert not (mesh.materials == MAT_PAVEMENT).any()
        assert not (mesh.materials == MAT_KERB).any()

    def test_a_driveway_breaks_the_pavement(self):
        street = way([(-100, 0), (100, 0)], nodes=[1, 2])
        drive = way([(0, 0), (0, 30)], highway="service", lanes=1, nodes=[3, 4])
        mesh = build_streets([street, drive], 0.0, 3.5, 0.5, 0.12)
        pavement = triangles_of(mesh, MAT_PAVEMENT)
        assert covers(pavement, (20.0, 5.0))
        assert not covers(pavement, (0.0, 5.0))  # where the driveway crosses it


class TestJunctions:
    def test_a_shared_interior_node_is_a_junction(self):
        zones = junction_zones(crossroads(), StreetConfig())
        assert zones is not None and zones.contains(__import__("shapely").geometry.Point(0.0, 0.0))

    def test_two_ways_that_merely_continue_do_not_make_one(self):
        a = way([(-100, 0), (0, 0)], nodes=[1, 2])
        b = way([(0, 0), (100, 0)], nodes=[2, 3])
        assert junction_zones([a, b], StreetConfig()) is None


class TestCrossings:
    def _lap(self):
        n = 256
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        centerline = np.stack([300.0 * np.cos(theta), 300.0 * np.sin(theta)], axis=1)
        return centerline, np.zeros(n), np.full(n, 4.0)

    def test_a_signal_gets_a_stop_line_and_a_crossing_gets_stripes(self):
        centerline, heights, half = self._lap()
        signal = ControlPoint(1, ("red_signal",), centerline[64], {"highway": "traffic_signals"}, progress=0.25)
        zebra = ControlPoint(2, ("pedestrian_crossing",), centerline[192], {"highway": "crossing"}, progress=0.75)
        mesh = crossing_markings(centerline, heights, half, (signal, zebra), "left", 0.12)
        assert mesh.vertices.shape[0] > 0
        assert (mesh.materials == MAT_FLAT).all()
        near_signal = np.linalg.norm(mesh.vertices[:, :2] - centerline[64], axis=1) < 8.0
        near_zebra = np.linalg.norm(mesh.vertices[:, :2] - centerline[192], axis=1) < 8.0
        assert near_signal.any() and near_zebra.any()
        # Stripes span the whole width; the stop line covers only the near half.
        assert near_zebra.sum() > near_signal.sum()

    def test_unplaced_points_paint_nothing(self):
        centerline, heights, half = self._lap()
        lost = ControlPoint(3, ("red_signal",), centerline[0], {}, progress=-1.0)
        assert crossing_markings(centerline, heights, half, (lost,), "left", 0.12).vertices.shape[0] == 0


class TestBuildNetwork:
    def test_reads_the_survey_and_reports_materials(self, tmp_path):
        survey = {
            "elements": [
                {
                    "type": "way",
                    "id": 1,
                    "nodes": [1, 2, 3],
                    "geometry": [{"lon": -100, "lat": 0}, {"lon": 0, "lat": 0}, {"lon": 100, "lat": 0}],
                    "tags": {"highway": "residential", "lanes": "2"},
                },
                {
                    "type": "way",
                    "id": 2,
                    "nodes": [4, 2, 5],
                    "geometry": [{"lon": 0, "lat": -100}, {"lon": 0, "lat": 0}, {"lon": 0, "lat": 100}],
                    "tags": {"highway": "service"},
                },
                {
                    "type": "way",
                    "id": 3,
                    "nodes": [6, 7],
                    "geometry": [{"lon": 5000, "lat": 5000}, {"lon": 5100, "lat": 5000}],
                    "tags": {"highway": "primary"},
                },
            ]
        }
        path = tmp_path / "survey.json"
        path.write_text(json.dumps(survey))
        centre = np.array([[x, 0.0] for x in np.linspace(-100, 100, 32)])
        mesh = build_network(path, FlatProjection(), SceneConfig(), centre=centre, max_distance_m=500.0)
        assert isinstance(mesh, RoadMesh)
        assert mesh.surface_material is not None and mesh.surface_material.shape[0] == mesh.surface.shape[0]
        assert set(np.unique(mesh.surface_material)) == {MAT_ASPHALT, MAT_PAVEMENT, MAT_KERB}
        assert mesh.markings.shape[0] > 0
        # The far-off primary road was dropped: nothing meshed near (5000, 5000).
        assert np.abs(mesh.surface[:, 0]).max() < 1000.0


class TestDirectEye:
    def test_direct_encoding_reads_nearness_as_given(self):
        from agent import AgentConfig, ConnectomeAgent
        import torch

        agent = object.__new__(ConnectomeAgent)
        agent.cfg = AgentConfig(eye_encoding="direct", n_rays=4)
        values = torch.tensor([[0.0, 0.25, 0.9, 1.4]])
        assert torch.allclose(agent.proximity(values), torch.tensor([[0.0, 0.25, 0.9, 1.0]]))

    def test_road_encoding_would_saturate_an_empty_channel(self):
        from agent import AgentConfig, ConnectomeAgent
        import torch

        agent = object.__new__(ConnectomeAgent)
        agent.cfg = AgentConfig(eye_encoding="road", n_rays=4)
        agent.ray_ref_m = torch.full((4,), 60.0)
        empty = torch.zeros(1, 4)
        assert torch.allclose(agent.proximity(empty), torch.ones(1, 4)), "this is the bug direct fixes"

    def test_camera_agent_config_uses_direct(self):
        from eye_camera import agent_config_for

        class Eye:
            width = 120

        class Sensor:
            eye = Eye()

        cfg = agent_config_for(Sensor())
        assert cfg.eye_encoding == "direct" and cfg.n_rays == 120


class TestStaticAssembly:
    def test_materials_follow_the_parts(self):
        pytest.importorskip("moderngl")
        from camera import assemble_static
        from streets import MAT_GROUND, MAT_WALL

        street = build_streets([way([(-50, 0), (50, 0)])], 0.0, 3.5, 0.5, 0.12)
        is_paint = street.materials == MAT_FLAT
        net = RoadMesh(
            street.vertices[~is_paint], street.colours[~is_paint], street.vertices[is_paint], street.colours[is_paint],
            np.zeros(0), street.materials[~is_paint],
        )
        walls = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 3]], dtype=np.float32)
        buildings = RoadMesh(walls, np.ones((3, 3), dtype=np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0), np.full(3, MAT_WALL, np.float32))
        flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        lap = RoadMesh(flat, np.ones((3, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0))
        vertices, colours, normals, materials = assemble_static(lap, net, buildings, (0.3, 0.33, 0.26))
        assert vertices.shape[0] == colours.shape[0] == normals.shape[0] == materials.shape[0]
        assert materials[:6].tolist() == [MAT_GROUND] * 6
        assert (materials == MAT_WALL).sum() == 3
        assert (materials == MAT_FLAT).sum() == is_paint.sum()
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
