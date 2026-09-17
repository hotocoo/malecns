"""Tests for driving on the camera instead of the ray march.

The point of the swap is that the track's geometry stops reaching the brain.
These tests hold that: the observation width changes, the values come from the
detector, and the detector runs at its own rate rather than the physics rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceive import Detection, EyeConfig  # noqa: E402
from scene import CLASS_CAR, SceneConfig, Traffic  # noqa: E402

moderngl = pytest.importorskip("moderngl")

from camera import CameraConfig  # noqa: E402
from eye_camera import CameraSensor, SensorConfig  # noqa: E402

N = 96


class StubDetector:
    """Answers with a fixed detection, and counts how often it was asked."""

    def __init__(self, found: list[Detection] | None = None) -> None:
        self.calls = 0
        self.found = found if found is not None else [Detection(CLASS_CAR, 0.9, 300.0, 100.0, 380.0, 300.0)]

    def detect(self, frames, confidence=0.25):
        self.calls += 1
        return [list(self.found) for _ in frames]


class StubScene:
    """The smallest scene a sensor needs: a ring road with no fixtures."""

    def __init__(self) -> None:
        from roadlaw import LegalProfile
        from scene import build_road

        theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        self.centerline = np.stack([200.0 * np.cos(theta), 200.0 * np.sin(theta)], axis=1)
        self.heights = np.zeros(N)
        self.profile = LegalProfile(
            limit_mps=np.full(N, 50.0 / 3.6),
            lanes=np.full(N, 2, dtype=np.int64),
            oneway=np.zeros(N, dtype=np.int64),
            tunnel=np.zeros(N, dtype=bool),
            roundabout=np.zeros(N, dtype=bool),
            way_id=np.zeros(N, dtype=np.int64),
            match_m=np.zeros(N),
            driving_side="left",
            source="test",
        )
        self.control_points = ()
        self.road = build_road(self.centerline, self.heights, self.profile)
        self.fixtures = []


def sensor(stride: int = 1, detector: StubDetector | None = None) -> CameraSensor:
    scene = StubScene()
    traffic = Traffic.populate(
        scene.centerline, scene.heights, scene.profile, (), vehicles=4, motorcycles=2, pedestrians=1, seed=0
    )
    try:
        return CameraSensor(
            scene,
            traffic,
            CameraConfig(width=160, height=96),
            EyeConfig(columns=6),
            SensorConfig(stride=stride),
            SceneConfig(),
            detector=detector or StubDetector(),
        )
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no GL context: {exc}")


def pose(batch: int = 3):
    return (
        torch.zeros(batch, 2),
        torch.zeros(batch),
        torch.full((batch,), 12.0),
        torch.linspace(0.0, 0.5, batch),
    )


class TestObservation:
    def test_width_is_the_detection_vector_plus_own_speed(self):
        eye = sensor()
        assert eye.width == eye.eye.width + 1

    def test_one_row_per_body(self):
        eye = sensor()
        pos, heading, speed, progress = pose(4)
        out = eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        assert out.shape == (4, eye.width)

    def test_speed_is_the_last_channel(self):
        eye = sensor()
        pos, heading, speed, progress = pose(2)
        out = eye.observe(pos, heading, speed, progress, 0.0, 24.0)
        assert out[:, -1].tolist() == pytest.approx([0.5, 0.5])

    def test_what_the_detector_reports_reaches_the_brain(self):
        eye = sensor(detector=StubDetector([Detection(CLASS_CAR, 0.9, 80.0, 20.0, 130.0, 80.0)]))
        pos, heading, speed, progress = pose(1)
        out = eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        assert out[0, :-1].abs().sum() > 0.0

    def test_a_detector_that_finds_nothing_leaves_the_view_empty(self):
        eye = sensor(detector=StubDetector([]))
        pos, heading, speed, progress = pose(1)
        out = eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        assert out[0, :-1].abs().sum() == 0.0


class TestStride:
    def test_the_detector_runs_once_per_stride(self):
        stub = StubDetector()
        eye = sensor(stride=3, detector=stub)
        pos, heading, speed, progress = pose(2)
        for _ in range(6):
            eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        assert stub.calls == 2

    def test_held_detections_still_track_the_cars_own_speed(self):
        eye = sensor(stride=4)
        pos, heading, _, progress = pose(2)
        first = eye.observe(pos, heading, torch.full((2,), 10.0), progress, 0.0, 40.0)
        second = eye.observe(pos, heading, torch.full((2,), 30.0), progress, 0.1, 40.0)
        assert torch.allclose(first[:, :-1], second[:, :-1])
        assert second[0, -1] > first[0, -1]

    def test_a_changed_population_forces_a_fresh_look(self):
        stub = StubDetector()
        eye = sensor(stride=10, detector=stub)
        pos, heading, speed, progress = pose(2)
        eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        pos, heading, speed, progress = pose(5)
        eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        assert stub.calls == 2

    def test_reset_starts_the_cycle_again(self):
        stub = StubDetector()
        eye = sensor(stride=5, detector=stub)
        pos, heading, speed, progress = pose(2)
        eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        eye.reset()
        eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        assert stub.calls == 2


class TestEnvSwap:
    def test_attaching_a_sensor_changes_what_the_brain_reads(self):
        from car_env import CarEnv, CarConfig

        cfg = CarConfig(layout="loop", n_points=256, max_laps=1)
        env = CarEnv(batch=2, device=torch.device("cpu"), cfg=cfg)
        ray_dim = env.obs_dim
        eye = sensor()
        env.attach_sensor(eye)
        assert env.obs_dim == eye.width
        assert env.obs_dim != ray_dim

    def test_detaching_restores_the_ray_eye(self):
        from car_env import CarEnv, CarConfig

        cfg = CarConfig(layout="loop", n_points=256, max_laps=1)
        env = CarEnv(batch=2, device=torch.device("cpu"), cfg=cfg)
        ray_dim = env.obs_dim
        env.attach_sensor(sensor())
        env.attach_sensor(None)
        assert env.obs_dim == ray_dim

    def test_observations_come_from_the_sensor_once_attached(self):
        from car_env import CarEnv, CarConfig

        cfg = CarConfig(layout="loop", n_points=256, max_laps=1)
        env = CarEnv(batch=2, device=torch.device("cpu"), cfg=cfg)
        eye = sensor()
        env.attach_sensor(eye)
        obs = env.observe()
        assert obs.shape == (2, eye.width)


class TestAgentWiring:
    def test_the_agent_reads_one_group_per_eye_channel(self):
        from eye_camera import agent_config_for

        eye = sensor()
        cfg = agent_config_for(eye)
        assert cfg.n_rays == eye.eye.width

    def test_the_brains_input_width_matches_the_sensor(self):
        from eye_camera import agent_config_for

        eye = sensor()
        cfg = agent_config_for(eye)
        # The observation is the eye's channels plus the car's own speed, and
        # the agent reads exactly that: n_rays columns, then speed.
        assert cfg.n_rays + 1 == eye.width


class TestTrafficStepping:
    def test_traffic_moves_when_the_sensor_steps_it(self):
        eye = sensor()
        before = [a.progress for a in eye.traffic.actors]
        eye.step_traffic(1.0)
        assert [a.progress for a in eye.traffic.actors] != before

    def test_stats_report_the_work_done(self):
        eye = sensor()
        pos, heading, speed, progress = pose(3)
        eye.observe(pos, heading, speed, progress, 0.0, 80.0)
        stats = eye.stats()
        assert stats["frames_rendered"] == 3.0
        assert stats["detector_calls"] == 1.0
