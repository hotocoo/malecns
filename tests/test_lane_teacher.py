from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibrate_camera import LaneTeacher  # noqa: E402
from car_env import CarConfig, CarEnv  # noqa: E402


def test_teacher_targets_the_cars_minimum_lap_time_profile_and_comfort_only_lowers_it():
    env = CarEnv(2, torch.device("cpu"), CarConfig())
    fast = LaneTeacher(env, law=None, lane_offset_m=0.0, lookahead_m=10.0, pace_fraction=1.0, gain=1.0)
    assert torch.equal(fast.corner_speed, env.track.speed_ref)
    comfy = LaneTeacher(env, law=None, lane_offset_m=0.0, lookahead_m=10.0, pace_fraction=1.0, gain=1.0, corner_g=0.3)
    assert bool((comfy.corner_speed <= fast.corner_speed + 1e-6).all())
    assert bool((comfy.corner_speed < fast.corner_speed).any())
