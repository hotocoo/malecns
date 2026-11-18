from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_env import GeoProjection  # noqa: E402
from scene import SceneConfig, build_buildings  # noqa: E402


def test_a_footprint_drawn_over_the_road_is_cut_back_off_it(tmp_path):
    proj = GeoProjection(lon0=101.7, lat0=3.15, scale=1.0)
    # a 20 m square plot with a road along y = 0 running through its middle
    corners = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    lat0 = np.deg2rad(3.15)
    lon = 101.7 + np.rad2deg(corners[:, 0] / (6_371_000.0 * np.cos(lat0)))
    lat = 3.15 + np.rad2deg(corners[:, 1] / 6_371_000.0)
    ring = np.stack([lon, lat], axis=1).tolist()
    path = tmp_path / "b.geojson"
    path.write_text(json.dumps({"features": [{"geometry": {"coordinates": [ring + [ring[0]]]}, "properties": {"height": 9}}]}))
    road = LineString([(-50, 0), (50, 0)]).buffer(4.0)

    clipped = build_buildings(path, proj, SceneConfig(), clear_of=road)
    whole = build_buildings(path, proj, SceneConfig())

    assert whole.surface.shape[0] and clipped.surface.shape[0]
    assert float(whole.surface[:, 1].min()) < -9.0
    # nothing of the clipped building stands inside the road
    inside = [road.buffer(-0.05).contains(Point(x, y)) for x, y, _ in clipped.surface]
    assert not any(inside)
    # the road splits the plot: what stands is either side of it, nothing on it
    assert bool((np.abs(clipped.surface[:, 1]) >= 4.0 - 1e-3).all())


def test_every_footprint_faces_out_whatever_way_the_survey_wound_it(tmp_path):
    proj = GeoProjection(lon0=101.7, lat0=3.15, scale=1.0)
    lat0 = np.deg2rad(3.15)
    for corners in (
        np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]),  # counter-clockwise
        np.array([[-10.0, -10.0], [-10.0, 10.0], [10.0, 10.0], [10.0, -10.0]]),  # clockwise
    ):
        lon = 101.7 + np.rad2deg(corners[:, 0] / (6_371_000.0 * np.cos(lat0)))
        lat = 3.15 + np.rad2deg(corners[:, 1] / 6_371_000.0)
        ring = np.stack([lon, lat], axis=1).tolist()
        path = tmp_path / "w.geojson"
        path.write_text(json.dumps({"features": [{"geometry": {"coordinates": [ring + [ring[0]]]}, "properties": {"height": 9}}]}))
        for clear in (None, LineString([(-50, -12), (50, -12)]).buffer(6.0)):
            mesh = build_buildings(path, proj, SceneConfig(), clear_of=clear)
            a, b, c = (mesh.surface[i].astype(np.float64) for i in range(3))
            normal = np.cross(b - a, c - a)[:2]
            # a wall of a convex footprint faces away from the footprint's centre
            assert float(np.dot(normal, a[:2] - mesh.surface[:, :2].mean(axis=0))) > 0.0
