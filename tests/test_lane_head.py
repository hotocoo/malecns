from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lane_head import LaneHead, LaneHeadConfig, LaneNet, labels_for  # noqa: E402

CFG = LaneHeadConfig()


def straight(n: int = 400, spacing: float = 1.0) -> np.ndarray:
    return np.stack([np.arange(n) * spacing, np.zeros(n)], axis=1)


def test_a_car_on_its_lane_path_reads_no_offset_and_dead_ahead():
    line = straight()
    on_lane = line[[100]] + [0.0, CFG.lane_offset_m]  # left normal of a +x road is +y
    y = labels_for(line, np.array([100]), on_lane, np.array([0.0]), CFG)
    assert np.allclose(y, 0.0, atol=1e-6)


def test_left_of_the_lane_reads_positive_offset_and_the_road_lies_to_the_right():
    line = straight()
    y = labels_for(line, np.array([100]), line[[100]] + [0.0, CFG.lane_offset_m + 1.5], np.array([0.0]), CFG)
    assert np.isclose(y[0, 0] * CFG.offset_scale_m, 1.5)
    assert (y[0, 1:] < 0.0).all()  # every bearing points back right, towards the lane
    assert abs(y[0, 1]) > abs(y[0, -1])  # nearer points need the sharper turn


def test_a_left_bend_ahead_reads_as_a_bearing_to_the_left():
    theta = np.linspace(0.0, np.pi, 400)
    line = np.stack([50.0 * np.sin(theta), 50.0 * (1.0 - np.cos(theta))], axis=1)  # turns left
    normal = np.array([-np.sin(theta[10]), np.cos(theta[10])])
    y = labels_for(line, np.array([10]), line[[10]] + normal * CFG.lane_offset_m, np.array([theta[10]]), CFG)
    assert (y[0, 1:] > 0.0).all()


def test_each_output_drives_one_side_of_its_mirrored_pair(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "lane.pt"
    torch.save({"state": LaneNet(CFG.outputs).state_dict(), "cfg": asdict(CFG)}, path)
    head = LaneHead(path, device="cpu")
    frames = np.random.default_rng(0).integers(0, 255, (3, 192, 320, 3), dtype=np.uint8)
    out = head.encode(frames, 15)
    assert out.shape == (3, 15)
    assert (out >= 0.0).all() and (out <= 1.0).all()
    pairs = out[:, : 2 * CFG.outputs].reshape(3, CFG.outputs, 2)
    assert (pairs.min(axis=2) == 0.0).all()  # never both sides at once
    assert (out[:, 2 * CFG.outputs :] == 0.0).all()
