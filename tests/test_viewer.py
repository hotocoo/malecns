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
from viewer import RASTER_QUOTA, curve_fleet_payload, curve_island_payload, curve_payload, json_safe, stratified_sample  # noqa: E402


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
    assert curve["fitness"][0] == 1.0 and curve["fitness"][-1] == 1000.0
    assert curve["best"][-1] == 1001.0
    assert max(curve["laps"]) == 1.0


def test_curve_payload_keeps_all_runs_instead_of_only_latest(tmp_path):
    log = tmp_path / "train.jsonl"
    records = [
        {"generation": 1, "fitness_best": 10.0, "seconds": 2.0},
        {"generation": 2, "fitness_best": 20.0, "seconds": 2.0},
        # Generation reset represents a trainer restart/resume.
        {"generation": 1, "fitness_best": 30.0, "seconds": 3.0},
        {"generation": 2, "fitness_best": 40.0, "seconds": 3.0},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_payload(log, points=20, keys=["fitness_best"])

    assert curve["generations"] == 4
    assert curve["runs"] == 2
    assert curve["series"]["fitness_best"] == [10.0, 20.0, 30.0, 40.0]
    assert curve["series_generation"]["fitness_best"] == [1, 2, 1, 2]
    assert curve["series_axis"]["fitness_best"] == [1, 2, 4, 5]
    assert curve["run_boundaries_axis"] == [4]


def test_curve_payload_handles_missing_or_empty_log(tmp_path):
    missing = curve_payload(tmp_path / "none.jsonl", 10)
    assert missing["generations"] == 0 and missing["fitness"] == [] and missing["laps"] == [] and missing["keys"] == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    assert curve_payload(empty, 10)["generations"] == 0


def test_curve_fleet_payload_combines_trainers(tmp_path):
    logs = []
    for name, offset in (("trainer_a.jsonl", 0.0), ("trainer_b.jsonl", 100.0)):
        log = tmp_path / name
        records = [
            {"generation": g, "fitness_mean": g + offset, "fitness_best": g + offset + 1.0, "seconds": 2.0}
            for g in range(1, 6)
        ]
        log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        logs.append(log)

    curve = curve_fleet_payload(logs, points=10, keys=["fitness_best"])

    assert curve["multi"] is True
    assert curve["trainers"] == ["trainer_a.jsonl", "trainer_b.jsonl"]
    assert curve["series"]["trainer_a.jsonl: fitness_best"][-1] == 6.0
    assert curve["series"]["trainer_b.jsonl: fitness_best"][-1] == 106.0
    assert curve["series_generation"]["trainer_a.jsonl: fitness_best"] == [1, 2, 3, 4, 5]
    assert curve["series_generation"]["trainer_b.jsonl: fitness_best"] == [1, 2, 3, 4, 5]


def test_curve_island_payload_exposes_parallel_trainers(tmp_path):
    log = tmp_path / "train.jsonl"
    records = [
        {
            "generation": g,
            "fitness_mean": float(g),
            "fitness_best": float(g + 10),
            "island_fitness_mean": [g, g + 100, g + 200],
            "island_fitness_best": [g + 1, g + 101, g + 201],
            "island_laps_best": [0.1 * g, 0.2 * g, 0.3 * g],
            "seconds": 3.0,
        }
        for g in range(1, 6)
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_island_payload(log, island=1, points=10, keys=["fitness_best"])

    assert curve["generations"] == 5
    assert curve["generation"] == [1, 2, 3, 4, 5]
    assert curve["series"]["fitness_best"] == [102, 103, 104, 105, 106]


def test_curve_island_payload_keeps_shared_scalar_telemetry(tmp_path):
    log = tmp_path / "train.jsonl"
    records = [
        {
            "generation": g,
            "fitness_best": float(g + 10),
            "island_fitness_mean": [g, g + 100],
            "island_fitness_best": [g + 1, g + 101],
            "sigma": 0.05 + g * 0.001,
            "lr": 0.01,
            "seconds": 2.0,
        }
        for g in range(1, 4)
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_island_payload(log, island=1, points=10, keys=["fitness_best", "sigma", "lr"])

    assert curve["series"]["fitness_best"] == [102, 103, 104]
    assert curve["series"]["sigma"] == [0.051, 0.052, 0.053]
    assert curve["series"]["lr"] == [0.01, 0.01, 0.01]
    assert curve["series_generation"]["sigma"] == [1, 2, 3]


def test_curve_island_payload_rejects_missing_island(tmp_path):
    log = tmp_path / "train.jsonl"
    log.write_text(json.dumps({"generation": 1, "island_fitness_mean": [1.0]}) + "\n")
    curve = curve_island_payload(log, island=2, points=10)
    assert curve["generations"] == 0


def test_curve_island_payload_ignores_legacy_records_before_island_telemetry(tmp_path):
    log = tmp_path / "train.jsonl"
    records = [
        {"generation": 1, "fitness_best": 1.0, "seconds": 2.0},
        {"generation": 2, "fitness_best": 2.0, "seconds": 2.0},
        {
            "generation": 3,
            "fitness_best": 3.0,
            "island_fitness_mean": [3.0, 13.0],
            "island_fitness_best": [4.0, 14.0],
            "seconds": 2.0,
        },
        {
            "generation": 4,
            "fitness_best": 4.0,
            "island_fitness_mean": [4.0, 14.0],
            "island_fitness_best": [5.0, 15.0],
            "seconds": 2.0,
        },
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_island_payload(log, island=1, points=10, keys=["fitness_best"])

    assert curve["generations"] == 2
    assert curve["generation"] == [3, 4]
    assert curve["series"]["fitness_best"] == [14.0, 15.0]


def test_curve_payload_preserves_nonfinite_values_as_json_null(tmp_path):
    log = tmp_path / "train.jsonl"
    records = [
        {"generation": 1, "fitness_mean": float("nan"), "fitness_best": float("inf")},
        {"generation": 2, "fitness_mean": 2.0, "fitness_best": 3.0},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_payload(log, points=10, keys=["fitness_mean", "fitness_best"])
    decoded = json.loads(json.dumps(json_safe(curve), allow_nan=False))
    assert decoded["series"]["fitness_mean"] == [None, 2.0]
    assert decoded["series"]["fitness_best"] == [None, 3.0]


def test_curve_payload_keeps_sparse_evaluations_on_their_generation_axis(tmp_path):
    log = tmp_path / "train.jsonl"
    records = []
    for generation in range(1, 101):
        record = {"generation": generation, "fitness_best": float(generation), "seconds": 1.0}
        if generation in (10, 50, 100):
            record["eval_fitness"] = generation + 0.5
        records.append(record)
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_payload(log, points=10, keys=["fitness_best", "eval_fitness"])

    assert curve["series"]["eval_fitness"] == [10.5, 50.5, 100.5]
    assert curve["series_generation"]["eval_fitness"] == [10, 50, 100]


def test_curve_payload_caps_sparse_evaluations_to_point_budget(tmp_path):
    log = tmp_path / "train.jsonl"
    records = [
        {"generation": generation, "fitness_best": float(generation), "eval_fitness": generation + 0.5, "seconds": 1.0}
        for generation in range(1, 21)
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    curve = curve_payload(log, points=5, keys=["eval_fitness"])

    assert curve["series"]["eval_fitness"] == [16.5, 17.5, 18.5, 19.5, 20.5]
    assert curve["series_generation"]["eval_fitness"] == [16, 17, 18, 19, 20]


def test_json_safe_recurses_and_nulls_nonfinite_numbers():
    payload = {"nan": float("nan"), "nested": [1.0, float("inf"), {"x": float("-inf")}]}
    assert json_safe(payload) == {"nan": None, "nested": [1.0, None, {"x": None}]}


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
