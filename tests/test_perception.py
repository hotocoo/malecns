"""Tests for the perception chain: scene geometry, the camera, and the detection eye.

The chain has one rule that matters more than any other: the driver is never
handed the answer. These tests hold the seam between the label pass, which
exists only to train a detector, and the eye, which only ever reads detections.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lawreward import AMBER, GREEN, RED, SignalTiming  # noqa: E402
from perceive import (  # noqa: E402
    CHANNELS,
    PHASE_CHANNELS,
    Detection,
    DetectionEye,
    EyeConfig,
    classify_lens,
    describe,
)
from roadlaw import ControlPoint, LegalProfile  # noqa: E402
from scene import (  # noqa: E402
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_PERSON,
    CLASS_STOP_SIGN,
    CLASS_TRAFFIC_LIGHT,
    Actor,
    SceneConfig,
    Traffic,
    build_road,
    frame_at,
    lane_halfwidth,
    place_signals,
    place_signs,
    update_signals,
)

N = 128


def ring(n: int = N, radius: float = 200.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def profile(lanes: int = 2, oneway: int = 0, side: str = "left", n: int = N) -> LegalProfile:
    return LegalProfile(
        limit_mps=np.full(n, 50.0 / 3.6),
        lanes=np.full(n, lanes, dtype=np.int64),
        oneway=np.full(n, oneway, dtype=np.int64),
        tunnel=np.zeros(n, dtype=bool),
        roundabout=np.zeros(n, dtype=bool),
        way_id=np.zeros(n, dtype=np.int64),
        match_m=np.zeros(n),
        driving_side=side,
        source="test",
    )


class TestRoadGeometry:
    def test_width_follows_the_surveyed_lane_count(self):
        cfg = SceneConfig()
        two = lane_halfwidth(profile(lanes=2), cfg)
        four = lane_halfwidth(profile(lanes=4), cfg)
        assert (four > two).all()
        assert two[0] == pytest.approx(2 * cfg.lane_width_m / 2.0 + cfg.shoulder_m)

    def test_an_unsurveyed_lane_count_falls_back_to_the_declared_default(self):
        cfg = SceneConfig(default_lanes=3)
        half = lane_halfwidth(profile(lanes=0), cfg)
        assert half[0] == pytest.approx(3 * cfg.lane_width_m / 2.0 + cfg.shoulder_m)

    def test_the_frame_is_orthonormal(self):
        tangent, normal = frame_at(ring())
        assert np.allclose(np.linalg.norm(tangent, axis=1), 1.0, atol=1e-6)
        assert np.allclose((tangent * normal).sum(axis=1), 0.0, atol=1e-6)

    def test_a_two_way_road_gets_a_centre_line(self):
        line = ring()
        both = build_road(line, np.zeros(N), profile(oneway=0))
        single = build_road(line, np.zeros(N), profile(oneway=1))
        assert both.markings.shape[0] > single.markings.shape[0]

    def test_the_surface_covers_every_sample(self):
        mesh = build_road(ring(), np.zeros(N), profile())
        assert mesh.surface.shape[0] == N * 6  # two triangles per sample


class TestFixtures:
    def signal_points(self):
        return [ControlPoint(node_id=5, rules=("red_signal",), pos=np.zeros(2), tags={}, progress=0.25, offset_m=2.0)]

    def test_a_signal_stands_where_the_survey_puts_one(self):
        actors = place_signals(self.signal_points(), ring(), np.zeros(N), profile(), "left")
        assert len(actors) == 1
        assert actors[0].kind == CLASS_TRAFFIC_LIGHT
        assert actors[0].node_id == 5

    def test_no_surveyed_signal_means_no_light(self):
        assert place_signals([], ring(), np.zeros(N), profile(), "left") == []

    def test_a_signal_stands_on_the_side_traffic_keeps_to(self):
        line = ring()
        left = place_signals(self.signal_points(), line, np.zeros(N), profile(), "left")[0]
        right = place_signals(self.signal_points(), line, np.zeros(N), profile(), "right")[0]
        assert not np.allclose(left.pos[:2], right.pos[:2])

    def test_only_a_stop_node_becomes_a_stop_sign(self):
        stop = ControlPoint(5, ("traffic_sign",), np.zeros(2), {"highway": "stop"}, 0.3, 2.0)
        camera = ControlPoint(6, ("traffic_sign",), np.zeros(2), {"highway": "speed_camera"}, 0.4, 2.0)
        signs = place_signs([stop, camera], ring(), np.zeros(N), profile(), "left")
        assert [s.kind for s in signs] == [CLASS_STOP_SIGN]

    def test_a_light_shows_the_phase_the_reward_charges(self):
        actors = place_signals(self.signal_points(), ring(), np.zeros(N), profile(), "left")
        timing = SignalTiming()
        seen = set()
        for t in np.arange(0.0, timing.cycle_s, 1.0):
            update_signals(actors, float(t), timing)
            seen.add(actors[0].state)
        assert seen == {GREEN, AMBER, RED}


class TestTraffic:
    def populate(self, **kwargs):
        return Traffic.populate(ring(), np.zeros(N), profile(), (), seed=0, **kwargs)

    def test_the_asked_population_appears(self):
        traffic = self.populate(vehicles=5, motorcycles=3, pedestrians=2)
        counts = traffic.visible_counts()
        assert counts[CLASS_CAR] == 5
        assert counts[CLASS_MOTORCYCLE] == 3
        assert counts[CLASS_PERSON] == 2

    def test_one_seed_gives_one_traffic_pattern(self):
        a = self.populate(vehicles=4, motorcycles=0, pedestrians=0)
        b = self.populate(vehicles=4, motorcycles=0, pedestrians=0)
        assert [x.progress for x in a.actors] == [x.progress for x in b.actors]

    def test_vehicles_travel_when_stepped(self):
        traffic = self.populate(vehicles=3, motorcycles=0, pedestrians=0)
        before = [a.progress for a in traffic.actors]
        traffic.step(1.0, ring(), np.zeros(N), profile())
        assert [a.progress for a in traffic.actors] != before

    def test_traffic_keeps_to_the_surveyed_side(self):
        traffic = self.populate(vehicles=6, motorcycles=0, pedestrians=0)
        traffic.step(0.5, ring(), np.zeros(N), profile())
        half = lane_halfwidth(profile(), SceneConfig()).mean()
        radii = np.array([np.linalg.norm(a.pos[:2]) for a in traffic.actors])
        # Every vehicle sits on one side of the ring, within the carriageway.
        assert ((radii - 200.0).min() > -half) and ((radii - 200.0).max() < half)
        assert np.sign(radii - 200.0).std() == 0.0  # all on the same side


class TestLensClassification:
    def patch(self, rgb: tuple[float, float, float]) -> np.ndarray:
        return np.tile(np.array(rgb, dtype=np.uint8), (8, 8, 1))

    def test_red_lens(self):
        assert classify_lens(self.patch((235, 30, 30))) == "signal_red"

    def test_green_lens(self):
        assert classify_lens(self.patch((25, 215, 65))) == "signal_green"

    def test_amber_lens(self):
        assert classify_lens(self.patch((250, 185, 15))) == "signal_amber"

    def test_an_unlit_crop_reports_nothing(self):
        assert classify_lens(self.patch((40, 40, 42))) is None

    def test_an_empty_crop_reports_nothing(self):
        assert classify_lens(np.zeros((0, 0, 3), dtype=np.uint8)) is None


class TestDetectionEye:
    def frame(self) -> np.ndarray:
        return np.zeros((384, 640, 3), dtype=np.uint8)

    def test_width_matches_the_declared_layout(self):
        eye = DetectionEye(EyeConfig(columns=8))
        assert eye.width == 8 * (len(CHANNELS) + len(PHASE_CHANNELS))

    def test_no_detections_is_an_empty_view(self):
        eye = DetectionEye(EyeConfig(columns=6))
        assert not eye.encode([], self.frame()).any()

    def test_a_car_lights_its_own_channel_and_column(self):
        eye = DetectionEye(EyeConfig(columns=4, include_phase=False))
        found = Detection(CLASS_CAR, 0.9, 480.0, 150.0, 560.0, 250.0)  # right of centre
        vector = eye.encode([found], self.frame()).reshape(len(CHANNELS), 4)
        assert vector[CHANNELS.index(CLASS_CAR), 3] > 0.0
        assert vector[CHANNELS.index(CLASS_CAR), 0] == 0.0
        assert vector[CHANNELS.index(CLASS_PERSON)].sum() == 0.0

    def test_a_nearer_object_reads_stronger(self):
        eye = DetectionEye(EyeConfig(columns=4, include_phase=False))
        far = Detection(CLASS_CAR, 1.0, 300.0, 180.0, 330.0, 200.0)
        near = Detection(CLASS_CAR, 1.0, 260.0, 100.0, 380.0, 300.0)
        f = eye.encode([far], self.frame()).max()
        n = eye.encode([near], self.frame()).max()
        assert n > f

    def test_a_low_confidence_detection_is_ignored(self):
        eye = DetectionEye(EyeConfig(columns=4, confidence=0.5, include_phase=False))
        weak = Detection(CLASS_CAR, 0.2, 300.0, 100.0, 380.0, 300.0)
        assert not eye.encode([weak], self.frame()).any()

    def test_columns_run_left_to_right(self):
        eye = DetectionEye(EyeConfig(columns=4, include_phase=False))
        left = Detection(CLASS_CAR, 1.0, 10.0, 100.0, 90.0, 300.0)
        vector = eye.encode([left], self.frame()).reshape(len(CHANNELS), 4)
        assert vector[CHANNELS.index(CLASS_CAR), 0] > 0.0

    def test_a_red_light_is_read_from_the_pixels(self):
        eye = DetectionEye(EyeConfig(columns=3))
        frame = self.frame()
        frame[150:200, 300:340] = (235, 30, 30)
        found = Detection(CLASS_TRAFFIC_LIGHT, 0.9, 300.0, 150.0, 340.0, 200.0)
        names = list(CHANNELS) + list(PHASE_CHANNELS)
        vector = eye.encode([found], frame).reshape(len(names), 3)
        assert vector[names.index("signal_red"), 1] > 0.0
        assert vector[names.index("signal_green"), 1] == 0.0

    def test_a_light_with_no_lit_lens_sets_no_phase(self):
        eye = DetectionEye(EyeConfig(columns=3))
        frame = self.frame()
        frame[150:200, 300:340] = (40, 40, 42)
        found = Detection(CLASS_TRAFFIC_LIGHT, 0.9, 300.0, 150.0, 340.0, 200.0)
        names = list(CHANNELS) + list(PHASE_CHANNELS)
        vector = eye.encode([found], frame).reshape(len(names), 3)
        assert vector[len(CHANNELS) :].sum() == 0.0

    def test_batch_encoding_is_per_body(self):
        eye = DetectionEye(EyeConfig(columns=4, include_phase=False))
        frames = [self.frame(), self.frame()]
        found = [[Detection(CLASS_CAR, 1.0, 300.0, 100.0, 380.0, 300.0)], []]
        out = eye.encode_batch(found, frames)
        assert out.shape == (2, eye.width)
        assert out[0].any() and not out[1].any()

    def test_channel_names_match_the_vector(self):
        eye = DetectionEye(EyeConfig(columns=5))
        assert len(eye.channel_names()) == eye.width

    def test_describe_counts_what_was_found(self):
        found = [Detection(CLASS_CAR, 0.9, 0, 0, 1, 1), Detection(CLASS_CAR, 0.8, 0, 0, 1, 1)]
        assert describe(found) == "carx2"
        assert describe([]) == "nothing"


class TestRenderer:
    """The renderer needs a GL context; skipped where none can be created."""

    @pytest.fixture(scope="class")
    def camera(self):
        pytest.importorskip("moderngl")
        from camera import CameraConfig, DriverCamera

        try:
            return DriverCamera(CameraConfig(width=160, height=96))
        except Exception as exc:  # pragma: no cover - depends on the host
            pytest.skip(f"no GL context: {exc}")

    def scene_actors(self):
        return [
            Actor(
                kind=CLASS_CAR,
                pos=np.array([20.0, 0.0, 0.0]),
                heading=np.pi,
                size=np.array([4.4, 1.8, 1.45]),
                colour=np.array([0.8, 0.1, 0.1], dtype=np.float32),
            )
        ]

    def test_a_frame_comes_back_as_an_image(self, camera):
        frame = camera.render(np.zeros(2), 0.0, 0.0, self.scene_actors())
        assert frame.shape == (96, 160, 3)
        assert frame.dtype == np.uint8

    def test_an_object_ahead_is_labelled(self, camera):
        _, boxes = camera.render_with_labels(np.zeros(2), 0.0, 0.0, self.scene_actors())
        assert [b[0] for b in boxes] == [CLASS_CAR]

    def test_an_object_behind_is_not(self, camera):
        actors = self.scene_actors()
        actors[0].pos = np.array([-30.0, 0.0, 0.0])
        _, boxes = camera.render_with_labels(np.zeros(2), 0.0, 0.0, actors)
        assert boxes == []

    def test_culling_keeps_what_the_lens_can_see(self, camera):
        ahead = self.scene_actors()[0]
        behind = self.scene_actors()[0]
        behind.pos = np.array([-50.0, 0.0, 0.0])
        kept = camera.visible(np.zeros(2), 0.0, [ahead, behind])
        assert kept == [ahead]

    def test_a_box_tightens_as_the_object_nears(self, camera):
        near = self.scene_actors()
        far = self.scene_actors()
        far[0].pos = np.array([30.0, 0.0, 0.0])
        _, near_boxes = camera.render_with_labels(np.zeros(2), 0.0, 0.0, near)
        _, far_boxes = camera.render_with_labels(np.zeros(2), 0.0, 0.0, far)
        near_area = (near_boxes[0][3] - near_boxes[0][1]) * (near_boxes[0][4] - near_boxes[0][2])
        far_area = (far_boxes[0][3] - far_boxes[0][1]) * (far_boxes[0][4] - far_boxes[0][2])
        assert near_area > far_area
