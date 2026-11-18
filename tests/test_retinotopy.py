from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import AgentConfig, ConnectomeAgent  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402

GRAPH = Path("data/graph_w5")
pytestmark = pytest.mark.skipif(not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first")
N = 8


def groups(azimuth):
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=torch.device("cpu"), use_metal=False)
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(n_rays=N, substeps=1, channel_azimuth=azimuth))
    side = connectome.neurons["somaSide"].to_numpy()
    return [g.numpy() for g in agent.ray_groups], side


def test_a_channel_looking_left_feeds_the_left_eye_whatever_its_position_in_the_vector():
    default, side = groups(None)
    # channel order reversed in the vector, azimuth says channel 0 looks far right
    reversed_, _ = groups(tuple(np.linspace(1.0, 0.0, N)))
    for k in range(N):
        assert np.array_equal(reversed_[k], default[N - 1 - k])
    assert set(side[reversed_[N - 1]]) <= {"L", "LHS", "left"}  # looks far left -> left eye
    assert set(side[reversed_[0]]) <= {"R", "RHS", "right"}


def test_a_wrong_length_azimuth_is_refused():
    with pytest.raises(ValueError):
        groups((0.0, 1.0))
