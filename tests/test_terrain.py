"""Surveyed terrain under the circuit: sampling, the road profile, the carved ground and building bases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_env import GeoProjection, load_geojson_centerline  # noqa: E402
from terrain import (  # noqa: E402
    Terrain,
    building_bases,
    heightfield,
    road_profile,
    sample_heightfield,
    smooth_circular,
    tunnel_mask,
)

MONACO = ROOT / "data" / "tracks" / "monaco.geojson"
DEM = ROOT / "data" / "tracks" / "monaco_dem.json"
needs_monaco = pytest.mark.skipif(not (MONACO.exists() and DEM.exists()), reason="monaco.geojson or monaco_dem.json missing")


def flat_projection() -> GeoProjection:
    return GeoProjection(lon0=7.42, lat0=43.73, scale=1.0)


def sloped_terrain(proj: GeoProjection, slope_per_m: float = 0.1, cell_m: float = 25.0, half_m: float = 600.0) -> Terrain:
    """A synthetic survey: height rises linearly with track-frame x."""
    lonlat_lo = proj.unproject(np.array([[-half_m, -half_m]]))[0]
    lonlat_hi = proj.unproject(np.array([[half_m, half_m]]))[0]
    cols = rows = int(2 * half_m / cell_m) + 1
    lons = np.linspace(lonlat_lo[0], lonlat_hi[0], cols)
    lats = np.linspace(lonlat_lo[1], lonlat_hi[1], rows)
    glon, glat = np.meshgrid(lons, lats)
    xy = proj.project(np.stack([glon.ravel(), glat.ravel()], 1))
    heights = (10.0 + slope_per_m * xy[:, 0]).reshape(rows, cols)
    return Terrain(lon0=float(lons[0]), lat0=float(lats[0]), dlon=float(lons[1] - lons[0]), dlat=float(lats[1] - lats[0]), heights=heights, source="synthetic")


def straight_centerline(n: int = 400, length_m: float = 800.0) -> np.ndarray:
    return np.stack([np.linspace(-length_m / 2, length_m / 2, n, endpoint=False), np.zeros(n)], axis=1)


def test_unproject_inverts_project():
    proj = GeoProjection(lon0=7.4272, lat0=43.7394, scale=1.0, dx=12.5, dy=-40.0)
    lonlat = np.array([[7.4213, 43.7331], [7.4300, 43.7410]])
    back = proj.unproject(proj.project(lonlat))
    assert np.allclose(back, lonlat, atol=1e-9)


def test_terrain_samples_bilinearly_and_clamps_outside():
    proj = flat_projection()
    terrain = sloped_terrain(proj, slope_per_m=0.1)
    xy = np.array([[0.0, 0.0], [100.0, 50.0], [-250.0, -10.0]])
    z = terrain.sample_xy(xy, proj)
    assert np.allclose(z, 10.0 + 0.1 * xy[:, 0], atol=0.05)
    far = terrain.sample_xy(np.array([[5000.0, 0.0]]), proj)
    assert far[0] == pytest.approx(terrain.heights.max(), abs=0.05), "outside the grid the edge value holds; nothing is extrapolated"


def test_smooth_circular_preserves_mean_and_wraps():
    values = np.zeros(200)
    values[0] = 100.0
    out = smooth_circular(values, 3.0)
    assert out.sum() == pytest.approx(values.sum(), rel=1e-6)
    assert out[-1] == pytest.approx(out[1], rel=1e-6), "the closed curve smooths across the seam"


def test_road_profile_follows_survey_with_quay_floor_and_tunnel_grade():
    proj = flat_projection()
    terrain = sloped_terrain(proj, slope_per_m=0.05)  # -10 m at x=-400 ... +30 m at x=+400
    centerline = straight_centerline()
    z_open = road_profile(centerline, proj, terrain, [], stride=1, smooth_m=0.0, min_height_m=1.2)
    assert np.allclose(z_open[centerline[:, 0] > 0], 10.0 + 0.05 * centerline[centerline[:, 0] > 0, 0], atol=0.1)
    assert z_open.min() == pytest.approx(1.2), "a survey cell at the waterline reads sea level; the road on it sits on the quay"
    # a tunnel from sample 200 to 259 with a 20 m probe outside each mouth: straight grade between the probes
    spacing = 2.0
    z_tun = road_profile(centerline, proj, terrain, [[200, 259]], stride=1, smooth_m=0.0, min_height_m=1.2, portal_probe_m=20.0)
    a, b = 200 - 10, 260 - 1 + 10
    grade = np.diff(z_tun[a : b + 1])
    assert np.allclose(grade, grade[0], atol=1e-6), "inside the tunnel the road is one straight grade"
    assert z_tun[a] == pytest.approx(z_open[a]) and z_tun[b] == pytest.approx(z_open[b])
    assert np.allclose(z_tun[: a], z_open[: a]) and np.allclose(z_tun[b + 1 :], z_open[b + 1 :])
    del spacing


def test_heightfield_is_flat_under_the_open_road_and_has_no_ground_over_the_tunnel():
    proj = flat_projection()
    terrain = sloped_terrain(proj, slope_per_m=0.1)
    centerline = straight_centerline()
    spans = [[250, 299]]
    in_tunnel = tunnel_mask(len(centerline), spans, 1)
    road_z = road_profile(centerline, proj, terrain, spans, stride=1, smooth_m=0.0, min_height_m=1.0, portal_probe_m=10.0)
    field, land = heightfield(
        terrain, proj, half=500.0, cell=8.0, centerline=centerline, road_z=road_z, halfwidth=5.5,
        in_tunnel=in_tunnel, wet_fn=None, land_min_m=1.0, sea_level_m=0.0, seabed_drop_m=2.0,
        carve_shoulder_m=2.5, carve_blend_m=12.0, tunnel_cover_min_m=7.4,
    )
    assert field["rows"] == field["cols"] == len(field["z"]) ** 0.5
    assert land.shape == (field["rows"], field["cols"])
    # under the open road (away from the portals) the ground sits just below the road across its whole width
    # away from the portals and from the open ends of this (not closed) test straight
    open_idx = np.flatnonzero(~in_tunnel)
    open_idx = open_idx[((open_idx < 230) | (open_idx > 320)) & (np.abs(centerline[open_idx, 0]) < 340)]
    for offset in (-5.5, 0.0, 5.5):
        pts = centerline[open_idx] + np.array([0.0, offset])
        diff = sample_heightfield(field, pts) - road_z[open_idx]
        assert np.nanmax(diff) <= 0.0 + 1e-6, f"ground pokes through the road at offset {offset}: {np.nanmax(diff):.2f} m"
        # on this 10 % grade the ground sits under the lowest road within reach: never more than a skirt's depth below
        assert np.nanmin(diff) >= -2.0, "the carved ground stays right under the asphalt"
    # far from the road the survey stands
    far = sample_heightfield(field, np.array([[0.0, 200.0], [100.0, -240.0]]))
    assert np.allclose(far, [10.0, 20.0], atol=0.6)
    # over the tunnel corridor there is no ground node; beside it the ground is at least the tunnel roof
    z = np.asarray([np.nan if v is None else v for v in field["z"]]).reshape(field["rows"], field["cols"])
    xs = field["x0"] + np.arange(field["cols"]) * field["cell"]
    ys = field["y0"] + np.arange(field["rows"]) * field["cell"]
    mid_x = float(centerline[275, 0])
    j = int(np.argmin(np.abs(xs - mid_x)))
    i0 = int(np.argmin(np.abs(ys - 0.0)))
    assert np.isnan(z[i0, j]), "no ground mesh directly over the tunnel"
    # the first ground node beside the corridor (20 m out, half-way through the 12 m blend) is
    # well above both the survey and the road: it eases down from the tunnel roof
    i_side = int(np.argmin(np.abs(ys - 20.0)))
    survey = 10.0 + 0.1 * xs[j]
    assert z[i_side, j] > survey + 1.0 and z[i_side, j] > road_z[275] + 3.0, "the ground beside the corridor rises onto the tunnel roof"


def test_building_bases_stand_on_land_and_on_the_tunnel_roof():
    proj = flat_projection()
    terrain = sloped_terrain(proj, slope_per_m=0.1)
    centerline = straight_centerline()
    spans = [[250, 299]]
    in_tunnel = tunnel_mask(len(centerline), spans, 1)
    road_z = road_profile(centerline, proj, terrain, spans, stride=1, smooth_m=0.0, min_height_m=1.0, portal_probe_m=10.0)
    wet = lambda pts: pts[:, 1] < -100.0  # noqa: E731 - the sea south of y=-100
    field, land = heightfield(
        terrain, proj, half=500.0, cell=8.0, centerline=centerline, road_z=road_z, halfwidth=5.5,
        in_tunnel=in_tunnel, wet_fn=wet, land_min_m=1.0, sea_level_m=0.0, seabed_drop_m=2.0,
        carve_shoulder_m=2.5, carve_blend_m=12.0, tunnel_cover_min_m=7.4,
    )
    x_mid = float(centerline[275, 0])
    buildings = [
        {"rings": [[[-300.0, 150.0], [-280.0, 150.0], [-280.0, 170.0], [-300.0, 170.0]]]},  # on the hill
        {"rings": [[[-300.0, -95.0], [-280.0, -95.0], [-280.0, -130.0], [-300.0, -130.0]]]},  # waterfront, partly over the sea
        {"rings": [[[x_mid - 10, -4.0], [x_mid + 10, -4.0], [x_mid + 10, 4.0], [x_mid - 10, 4.0]]]},  # over the tunnel
    ]
    bases = building_bases(buildings, field, land, centerline, road_z, in_tunnel, corridor_m=7.4, tunnel_cover_min_m=7.4)
    assert bases[0] == pytest.approx(10.0 + 0.1 * -300.0, abs=0.6) or bases[0] >= 1.0
    assert bases[1] >= 1.0, "a waterfront building stands on the quay, never on the seabed"
    under_footprint = road_z[np.abs(centerline[:, 0] - x_mid) <= 10.0]
    assert bases[2] >= under_footprint.max() + 7.4 - 1e-6, "a building over the tunnel stands on its roof"


@needs_monaco
def test_monaco_terrain_matches_the_circuit():
    """Copernicus EU-DEM under the real circuit: Casino Square is the high point, the harbour the low one."""
    centerline, proj = load_geojson_centerline(MONACO, return_projection=True)
    centerline = centerline.numpy()
    terrain = Terrain.load(DEM)
    z = road_profile(centerline, proj, terrain, [], stride=2, smooth_m=30.0, min_height_m=1.2)
    assert 35.0 <= z.max() - z.min() <= 65.0, f"Monaco climbs about 42 m officially; the survey says {z.max() - z.min():.1f}"
    lonlat = proj.unproject(centerline)
    casino = np.argmin(np.hypot(lonlat[:, 0] - 7.4279, lonlat[:, 1] - 43.7398))
    tabac = np.argmin(np.hypot(lonlat[:, 0] - 7.4262, lonlat[:, 1] - 43.7352))
    assert z[casino] > z[tabac] + 25.0
    spacing = float(np.hypot(*np.diff(centerline, axis=0).T).mean())
    assert np.abs(np.gradient(z) / spacing).max() < 0.2, "no grade steeper than 20 % on a smoothed 25 m survey"
    raw = json.loads(DEM.read_text())
    assert raw["source"] in ("eudem25m", "srtm30m", "terrarium") and raw["attribution"]
