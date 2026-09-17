"""Tests for the legal road layer: OSM tags sampled onto a circuit centerline.

Every number the reward will later charge against (speed limit, lane count,
driving side) has to come from the survey. These tests pin that: a tag that is
absent stays absent, a class default is the median of the classes actually
present in the file, and nothing falls back to a literal written in the source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadlaw import (  # noqa: E402
    KMH_TO_MPS,
    MPH_TO_MPS,
    LegalProfile,
    class_speed_defaults,
    legal_profile_for_circuit,
    load_highways,
    parse_lanes,
    parse_maxspeed,
    parse_oneway,
)

HIGHWAYS = ROOT / "data" / "tracks" / "monaco_osm_highways.json"
TRACK = ROOT / "data" / "tracks" / "monaco.geojson"

needs_survey = pytest.mark.skipif(
    not HIGHWAYS.exists() or not TRACK.exists(), reason="Monaco survey not downloaded"
)


class TestParseMaxspeed:
    def test_bare_number_is_kmh(self):
        assert parse_maxspeed("50") == pytest.approx(50.0 * KMH_TO_MPS)

    def test_explicit_kmh_unit(self):
        assert parse_maxspeed("50 km/h") == pytest.approx(50.0 * KMH_TO_MPS)

    def test_mph_is_converted(self):
        assert parse_maxspeed("30 mph") == pytest.approx(30.0 * MPH_TO_MPS)

    def test_knots_are_converted(self):
        assert parse_maxspeed("10 knots") == pytest.approx(10.0 * 1852.0 / 3600.0)

    @pytest.mark.parametrize("raw", ["none", "signals", "walk", "RO:urban", "", None, "fast"])
    def test_non_numeric_is_unknown(self, raw):
        """No numeric limit in the tag means no limit known, never a guess."""
        assert parse_maxspeed(raw) is None

    def test_variable_limit_takes_the_numeric_part(self):
        assert parse_maxspeed("50; 70") == pytest.approx(50.0 * KMH_TO_MPS)

    def test_negative_is_rejected(self):
        assert parse_maxspeed("-50") is None


class TestParseLanes:
    def test_integer(self):
        assert parse_lanes("2") == 2

    def test_float_string_rounds_down_to_whole_lanes(self):
        assert parse_lanes("2.5") == 2

    @pytest.mark.parametrize("raw", ["0", "-1", "", None, "two"])
    def test_unusable_is_unknown(self, raw):
        assert parse_lanes(raw) is None


class TestParseOneway:
    @pytest.mark.parametrize("raw", ["yes", "true", "1"])
    def test_forward(self, raw):
        assert parse_oneway(raw) == 1

    @pytest.mark.parametrize("raw", ["-1", "reverse"])
    def test_reverse(self, raw):
        assert parse_oneway(raw) == -1

    @pytest.mark.parametrize("raw", ["no", "false", "0"])
    def test_two_way(self, raw):
        assert parse_oneway(raw) == 0

    @pytest.mark.parametrize("raw", [None, "", "maybe"])
    def test_unknown(self, raw):
        assert parse_oneway(raw) is None


class TestClassDefaults:
    def test_default_is_the_median_of_that_class_in_this_file(self):
        ways = [
            {"tags": {"highway": "residential", "maxspeed": "30"}},
            {"tags": {"highway": "residential", "maxspeed": "50"}},
            {"tags": {"highway": "residential", "maxspeed": "30"}},
            {"tags": {"highway": "primary", "maxspeed": "70"}},
        ]
        defaults = class_speed_defaults(ways)
        assert defaults["residential"] == pytest.approx(30.0 * KMH_TO_MPS)
        assert defaults["primary"] == pytest.approx(70.0 * KMH_TO_MPS)

    def test_class_without_any_tagged_member_gets_no_default(self):
        """An untagged class stays unknown; the module invents no limit for it."""
        ways = [{"tags": {"highway": "service"}}, {"tags": {"highway": "service"}}]
        assert class_speed_defaults(ways) == {}

    def test_empty_input(self):
        assert class_speed_defaults([]) == {}


@needs_survey
class TestLegalProfile:
    @pytest.fixture(scope="class")
    def profile(self):
        from car_env import load_geojson_centerline

        pts, proj = load_geojson_centerline(TRACK, return_projection=True)
        return legal_profile_for_circuit(TRACK, pts.numpy(), proj)

    def test_profile_exists_for_a_surveyed_circuit(self, profile):
        assert isinstance(profile, LegalProfile)

    def test_one_entry_per_centerline_sample(self, profile):
        from car_env import load_geojson_centerline

        n = load_geojson_centerline(TRACK).shape[0]
        for name in ("limit_mps", "lanes", "oneway", "way_id", "match_m"):
            assert getattr(profile, name).shape == (n,), name

    def test_every_sample_matches_a_mapped_way(self, profile):
        """The circuit is public road; each sample should find a way close by."""
        assert (profile.way_id >= 0).all()
        assert profile.match_m.max() < 30.0

    def test_limits_are_plausible_road_speeds(self, profile):
        known = profile.limit_mps[np.isfinite(profile.limit_mps)]
        assert known.size > 0
        assert known.min() > 0.0
        assert known.max() < 100.0  # m/s; no road in the file is that fast

    def test_limits_come_from_the_file_not_from_code(self, profile):
        """Every limit in the profile is a value that appears in the survey."""
        raw = json.loads(HIGHWAYS.read_text())["elements"]
        tagged = {parse_maxspeed(w.get("tags", {}).get("maxspeed")) for w in raw}
        tagged.discard(None)
        defaults = set(class_speed_defaults(raw).values())
        allowed = tagged | defaults
        seen = set(np.unique(profile.limit_mps[np.isfinite(profile.limit_mps)]))
        for value in seen:
            assert any(abs(value - a) < 1e-6 for a in allowed), value

    def test_driving_side_is_read_not_assumed(self, profile):
        """Left/right comes from the survey; an unsurveyed file leaves it unknown."""
        assert profile.driving_side in ("left", "right", None)

    def test_lanes_are_whole_and_positive_where_known(self, profile):
        known = profile.lanes[profile.lanes > 0]
        assert known.size > 0
        assert (known == known.astype(int)).all()

    def test_missing_survey_returns_none(self, tmp_path):
        track = tmp_path / "nowhere.geojson"
        track.write_text(json.dumps({"features": [{"geometry": {"coordinates": [[0, 0], [0, 1]]}}]}))
        assert legal_profile_for_circuit(track, np.zeros((4, 2)), None) is None


@needs_survey
class TestLoadHighways:
    def test_ways_are_projected_into_track_metres(self):
        from car_env import load_geojson_centerline

        pts, proj = load_geojson_centerline(TRACK, return_projection=True)
        ways = load_highways(HIGHWAYS, proj)
        assert ways
        extent = np.abs(np.concatenate([w.points for w in ways])).max()
        assert extent < 20_000.0  # metres, not degrees

    def test_every_way_keeps_its_osm_identity(self):
        from car_env import load_geojson_centerline

        _, proj = load_geojson_centerline(TRACK, return_projection=True)
        ways = load_highways(HIGHWAYS, proj)
        assert all(w.way_id > 0 for w in ways)
        assert len({w.way_id for w in ways}) == len(ways)
