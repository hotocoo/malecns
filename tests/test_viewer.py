"""Behavioural tests for the live viewer's data plumbing.

Run with: python3 -m pytest tests -q
Nothing here opens a socket or needs the connectome; the pieces that shape
what the browser receives are exercised directly.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_positions import infer_missing  # noqa: E402
from viewer import RASTER_QUOTA, curve_payload, stratified_sample  # noqa: E402


def test_stratified_sample_respects_quota_and_bands():
    roles = {role: list(range(i * 1000, i * 1000 + 500)) for i, role in enumerate(RASTER_QUOTA)}
    sample, bands = stratified_sample(roles, seed=1)

    assert len(sample) == sum(RASTER_QUOTA.values())
    assert len(set(sample.tolist())) == len(sample), "no neuron sampled twice"
    for band in bands:
        chunk = sample[band["start"] : band["start"] + band["count"]]
        assert all(i in roles[band["role"]] for i in chunk.tolist())
        assert band["total"] == 500


def test_stratified_sample_skips_empty_roles_and_caps_small_ones():
    roles = {"descending": list(range(7)), "motor": [], "ascending": list(range(100, 300))}
    sample, bands = stratified_sample(roles)
    by_role = {b["role"]: b for b in bands}
    assert "motor" not in by_role
    assert by_role["descending"]["count"] == 7
    assert by_role["ascending"]["count"] == RASTER_QUOTA["ascending"]
    assert len(sample) == 7 + RASTER_QUOTA["ascending"]


def test_stratified_sample_is_deterministic():
    roles = {role: list(range(2000)) for role in RASTER_QUOTA}
    a, _ = stratified_sample(roles, seed=0)
    b, _ = stratified_sample(roles, seed=0)
    assert np.array_equal(a, b)


def test_curve_payload_downsamples_and_totals(tmp_path):
    log = tmp_path / "train.jsonl"
    records = [
        {"generation": g, "fitness_mean": float(g), "fitness_best": g + 1.0, "laps_best": g / 1000, "seconds": 36.0}
        for g in range(1, 1001)
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_payload(log, points=100)

    assert curve["generations"] == 1000
    assert curve["hours"] == 10.0
    assert curve["stride"] == 10
    assert len(curve["fitness"]) == 100
    assert curve["fitness"][0] == 1.0 and curve["best"][-1] == 992.0
    assert max(curve["laps"]) == 0.991


def test_curve_payload_handles_missing_or_empty_log(tmp_path):
    assert curve_payload(tmp_path / "none.jsonl", 10) == {"generations": 0, "fitness": [], "laps": []}
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    assert curve_payload(empty, 10)["generations"] == 0


def test_spike_mask_roundtrip_matches_browser_decoding():
    """numpy.packbits is MSB-first; the client reads bit (7 - i%8) of byte i//8."""
    spiked = np.zeros(166_700, dtype=np.uint8)
    hits = np.array([0, 7, 8, 1000, 166_699])
    spiked[hits] = 1
    encoded = base64.b64encode(np.packbits(spiked)).decode()

    mask = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    decoded = [i for i in range(len(spiked)) if (mask[i >> 3] >> (7 - (i & 7))) & 1]
    assert decoded == hits.tolist()


def test_infer_missing_places_unlocated_cells_between_partners():
    pos = np.array([[0.0, 0, 0], [10.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]], dtype=np.float32)
    known = np.array([True, True, False, False])
    # cell 2 receives from 0 and 1; cell 3 is only connected to cell 2
    pre = np.array([0, 1, 2])
    post = np.array([2, 2, 3])

    out = infer_missing(pos, known, pre, post, rounds=3)

    assert np.allclose(out[2], [5.0, 0, 0]), "midpoint of its two located partners"
    assert np.allclose(out[3], out[2]), "second round inherits from cell 2"
    assert np.array_equal(out[:2], pos[:2]), "measured cells never move"
    assert np.array_equal(pos[2], [0, 0, 0]), "input is not mutated"


def test_infer_missing_falls_back_to_centroid_for_isolated_cells():
    pos = np.array([[0.0, 0, 0], [2.0, 2, 2], [9.0, 9, 9]], dtype=np.float32)
    known = np.array([True, True, False])
    out = infer_missing(pos, known, np.array([0]), np.array([1]), rounds=2)
    assert np.allclose(out[2], [1.0, 1, 1])


# --- harbour water -----------------------------------------------------------------

def test_water_side_uses_right_hand_side_of_coastline():
    from viewer import coast_segments, water_side

    # a coastline running east along y = 0: land (left) is y > 0, water (right) y < 0
    segments = coast_segments([np.array([[-100.0, 0.0], [0.0, 0.0], [100.0, 0.0]])])
    pts = np.array([[0.0, 10.0], [0.0, -10.0], [50.0, 3.0], [50.0, -3.0], [500.0, -1.0], [-500.0, 1.0]])
    assert water_side(pts, segments).tolist() == [False, True, False, True, True, False]


def test_water_side_is_consistent_around_a_corner():
    from viewer import coast_segments, water_side

    # a quay corner: east then north. Land stays on the left of travel, so the
    # inner (north-west) side is land and the outer wedge (south-east) is water.
    segments = coast_segments([np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])])
    assert water_side(np.array([[12.0, -2.0]]), segments).tolist() == [True]
    assert water_side(np.array([[8.0, 2.0]]), segments).tolist() == [False]


def test_merge_rects_covers_mask_exactly():
    from viewer import merge_rects

    mask = np.array([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 1]], dtype=bool)
    rects = merge_rects(mask, x0=0.0, y0=0.0, cell=2.0)
    covered = np.zeros(mask.shape, dtype=int)
    for x0, y0, x1, y1 in rects:
        covered[int(y0 / 2) : int(y1 / 2), int(x0 / 2) : int(x1 / 2)] += 1
    assert (covered == mask.astype(int)).all()  # every cell once, nothing outside
    assert len(rects) == 2  # two vertical merges: the 2x2 block and the 3x1 column


def test_water_never_covers_the_road():
    from viewer import water_and_land

    centerline = np.stack([np.linspace(-200, 200, 400), np.zeros(400)], axis=1)
    # coastline right along the road's edge: everything south should be water except the corridor
    coast = [np.array([[-300.0, -3.0], [300.0, -3.0]])]
    out = water_and_land(coast, centerline, halfwidth=5.5, ground_half=400.0, fine_m=4.0, coarse_m=80.0)
    assert out["water"], "the sea south of the coast must be rendered"
    for x0, y0, x1, y1 in out["water"]:
        assert y1 <= -5.5 or x0 >= 200 or x1 <= -200, f"water rect {x0, y0, x1, y1} overlaps the road corridor"
