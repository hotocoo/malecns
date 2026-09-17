"""Tests for charging the driver under the road law.

Every charge must trace to something the survey states. Where the survey is
silent - no posted limit, no recorded driving side, no signal - the charge is
zero, never a guess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lawreward import (  # noqa: E402
    AMBER,
    GREEN,
    RED,
    LawEnforcer,
    LawWeights,
    SignalTiming,
    signal_phase,
    signed_offset,
)
from roadlaw import ControlPoint, LegalProfile  # noqa: E402

DEVICE = torch.device("cpu")
N = 64


def straight_centerline(n: int = N, spacing: float = 10.0) -> torch.Tensor:
    """A closed rectangle: the first half runs east, so left is +y there."""
    x = np.linspace(0.0, spacing * n / 2.0, n // 2)
    top = np.stack([x, np.zeros_like(x)], axis=1)
    bottom = np.stack([x[::-1], np.full_like(x, 50.0)], axis=1)
    return torch.tensor(np.concatenate([top, bottom]), dtype=torch.float32)


def profile(limit_kmh: float | None = 50.0, oneway: int = 0, side: str | None = "left", n: int = N) -> LegalProfile:
    limit = np.full(n, np.nan if limit_kmh is None else limit_kmh / 3.6)
    return LegalProfile(
        limit_mps=limit,
        lanes=np.full(n, 2, dtype=np.int64),
        oneway=np.full(n, oneway, dtype=np.int64),
        tunnel=np.zeros(n, dtype=bool),
        roundabout=np.zeros(n, dtype=bool),
        way_id=np.zeros(n, dtype=np.int64),
        match_m=np.zeros(n),
        driving_side=side,
        source="test",
    )


def point(rule: str, progress: float, node_id: int = 101, tags: dict | None = None) -> ControlPoint:
    return ControlPoint(
        node_id=node_id,
        rules=(rule,),
        pos=np.zeros(2),
        tags=tags or {},
        progress=progress,
        offset_m=1.0,
    )


def enforcer(points=(), weights=None, **kwargs) -> LawEnforcer:
    return LawEnforcer(
        profile(**kwargs),
        points,
        straight_centerline(),
        DEVICE,
        weights=weights or LawWeights(),
    )


def state(progress, prev, speed, pos=None, elapsed=0.0):
    batch = len(progress)
    return {
        "progress": torch.tensor(progress, dtype=torch.float32),
        "prev_progress": torch.tensor(prev, dtype=torch.float32),
        "speed": torch.tensor(speed, dtype=torch.float32),
        "pos": torch.zeros(batch, 2) if pos is None else torch.tensor(pos, dtype=torch.float32),
        "elapsed_s": torch.full((batch,), elapsed),
    }


class TestSignalPhase:
    def test_cycle_covers_all_three_phases(self):
        timing = SignalTiming()
        ids = torch.zeros(1, dtype=torch.long)
        seen = {
            int(signal_phase(ids, torch.tensor([t], dtype=torch.float32), timing)[0])
            for t in np.arange(0.0, timing.cycle_s, 0.5)
        }
        assert seen == {GREEN, AMBER, RED}

    def test_phase_repeats_every_cycle(self):
        timing = SignalTiming()
        ids = torch.tensor([7], dtype=torch.long)
        first = signal_phase(ids, torch.tensor([3.0]), timing)
        later = signal_phase(ids, torch.tensor([3.0 + timing.cycle_s]), timing)
        assert int(first[0]) == int(later[0])

    def test_two_junctions_are_not_in_lockstep(self):
        timing = SignalTiming()
        times = torch.zeros(2)
        phases = signal_phase(torch.tensor([1, 500]), times, timing)
        assert int(phases[0]) != int(phases[1])

    def test_same_node_gives_the_same_phase_every_run(self):
        timing = SignalTiming()
        a = signal_phase(torch.tensor([42]), torch.tensor([10.0]), timing)
        b = signal_phase(torch.tensor([42]), torch.tensor([10.0]), timing)
        assert int(a[0]) == int(b[0])


class TestSignedOffset:
    def test_left_of_an_eastbound_line_is_positive(self):
        line = straight_centerline()
        pos = torch.tensor([[10.0, 3.0]])
        index = torch.tensor([1])
        assert float(signed_offset(pos, line, index)[0]) > 0.0

    def test_right_is_negative(self):
        line = straight_centerline()
        pos = torch.tensor([[10.0, -3.0]])
        assert float(signed_offset(pos, line, torch.tensor([1]))[0]) < 0.0

    def test_on_the_line_is_zero(self):
        line = straight_centerline()
        pos = line[1:2].clone()
        assert float(signed_offset(pos, line, torch.tensor([1]))[0]) == pytest.approx(0.0, abs=1e-4)


class TestSpeeding:
    def test_under_the_limit_costs_nothing(self):
        law = enforcer(limit_kmh=50.0)
        out = law.charge(**state([0.1], [0.09], [10.0]))
        assert float(out["law_speeding"][0]) == 0.0

    def test_over_the_limit_is_charged(self):
        law = enforcer(limit_kmh=50.0)
        out = law.charge(**state([0.1], [0.09], [20.0]))
        assert float(out["law_speeding"][0]) < 0.0

    def test_the_charge_grows_with_the_excess(self):
        law = enforcer(limit_kmh=50.0)
        small = law.charge(**state([0.1], [0.09], [16.0]))["law_speeding"][0]
        large = law.charge(**state([0.1], [0.09], [25.0]))["law_speeding"][0]
        assert float(large) < float(small) < 0.0

    def test_an_unposted_road_charges_nothing(self):
        """No maxspeed in the survey means no offence, not a limit of zero."""
        law = enforcer(limit_kmh=None)
        out = law.charge(**state([0.1], [0.09], [80.0]))
        assert float(out["law_speeding"][0]) == 0.0
        assert not torch.isfinite(out["law_limit_mps"][0])

    def test_the_limit_reported_is_the_surveyed_one(self):
        law = enforcer(limit_kmh=70.0)
        out = law.charge(**state([0.1], [0.09], [5.0]))
        assert float(out["law_limit_mps"][0]) == pytest.approx(70.0 / 3.6, rel=1e-5)


class TestKeepLeft:
    """The test centerline's first half runs east, so +y is its left."""

    def test_the_right_half_is_charged_where_traffic_keeps_left(self):
        law = enforcer(side="left", oneway=0)
        out = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, -6.0]]))
        assert float(out["law_wrong_side"][0]) < 0.0

    def test_the_left_half_is_free(self):
        law = enforcer(side="left", oneway=0)
        out = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, 6.0]]))
        assert float(out["law_wrong_side"][0]) == 0.0

    def test_the_sides_swap_where_traffic_keeps_right(self):
        law = enforcer(side="right", oneway=0)
        left = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, 6.0]]))
        right = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, -6.0]]))
        assert float(left["law_wrong_side"][0]) < 0.0
        assert float(right["law_wrong_side"][0]) == 0.0

    def test_a_oneway_road_has_no_side_rule(self):
        law = enforcer(side="left", oneway=1)
        out = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, -6.0]]))
        assert float(out["law_wrong_side"][0]) == 0.0

    def test_an_unsurveyed_side_is_not_enforced(self):
        law = enforcer(side=None, oneway=0)
        out = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, -6.0]]))
        assert float(out["law_wrong_side"][0]) == 0.0

    def test_the_charge_grows_with_the_distance_across(self):
        law = enforcer(side="left", oneway=0)
        near = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, -1.0]]))["law_wrong_side"][0]
        far = law.charge(**state([0.02], [0.01], [10.0], pos=[[10.0, -8.0]]))["law_wrong_side"][0]
        assert float(far) < float(near) < 0.0


class TestRedSignal:
    def signals(self):
        return [point("red_signal", 0.5, node_id=1)]

    def test_passing_on_red_is_charged(self):
        law = enforcer(self.signals())
        timing = law.timing
        red_time = timing.green_s + timing.amber_s + 1.0
        # Node 1's offset is tiny, so the cycle starts almost at t=0.
        out = law.charge(**state([0.55], [0.45], [10.0], elapsed=red_time))
        assert float(out["law_red_signal"][0]) < 0.0

    def test_passing_on_green_is_free(self):
        law = enforcer(self.signals())
        out = law.charge(**state([0.55], [0.45], [10.0], elapsed=1.0))
        assert float(out["law_red_signal"][0]) == 0.0

    def test_waiting_at_a_red_is_free(self):
        """The charge lands on entering the junction, not on standing before it."""
        law = enforcer(self.signals())
        red_time = law.timing.green_s + law.timing.amber_s + 1.0
        out = law.charge(**state([0.45], [0.45], [0.0], elapsed=red_time))
        assert float(out["law_red_signal"][0]) == 0.0

    def test_a_circuit_without_signals_charges_nothing(self):
        law = enforcer([])
        assert not law.has_signals
        out = law.charge(**state([0.55], [0.45], [30.0], elapsed=40.0))
        assert float(out["law_red_signal"][0]) == 0.0

    def test_a_lap_wrap_still_sees_the_signal(self):
        law = enforcer([point("red_signal", 0.99, node_id=1)])
        red_time = law.timing.green_s + law.timing.amber_s + 1.0
        out = law.charge(**state([0.02], [0.97], [10.0], elapsed=red_time))
        assert float(out["law_red_signal"][0]) < 0.0


class TestStopLine:
    def stops(self):
        return [point("traffic_sign", 0.5, node_id=3, tags={"highway": "stop"})]

    def test_rolling_through_is_charged(self):
        law = enforcer(self.stops())
        out = law.charge(**state([0.55], [0.45], [12.0]))
        assert float(out["law_stop_line"][0]) < 0.0

    def test_stopping_is_free(self):
        law = enforcer(self.stops())
        out = law.charge(**state([0.55], [0.45], [0.2]))
        assert float(out["law_stop_line"][0]) == 0.0

    def test_a_sign_that_is_not_a_stop_line_is_ignored(self):
        law = enforcer([point("traffic_sign", 0.5, node_id=3, tags={"highway": "speed_camera"})])
        out = law.charge(**state([0.55], [0.45], [12.0]))
        assert float(out["law_stop_line"][0]) == 0.0

    def test_give_way_counts_as_a_stop_line(self):
        law = enforcer([point("traffic_sign", 0.5, node_id=3, tags={"highway": "give_way"})])
        out = law.charge(**state([0.55], [0.45], [12.0]))
        assert float(out["law_stop_line"][0]) < 0.0


class TestCrossing:
    def test_speed_across_a_crossing_is_charged(self):
        law = enforcer([point("pedestrian_crossing", 0.5, node_id=9)])
        fast = law.charge(**state([0.55], [0.45], [20.0]))["law_crossing"][0]
        slow = law.charge(**state([0.55], [0.45], [2.0]))["law_crossing"][0]
        assert float(fast) < float(slow) < 0.0

    def test_no_crossing_means_no_charge(self):
        law = enforcer([])
        assert float(law.charge(**state([0.55], [0.45], [20.0]))["law_crossing"][0]) == 0.0


class TestTotals:
    def test_every_charge_is_a_cost_never_a_reward(self):
        law = enforcer([point("red_signal", 0.5, node_id=1)])
        out = law.charge(**state([0.55, 0.2], [0.45, 0.1], [30.0, 5.0], pos=[[10.0, -6.0], [10.0, 6.0]], elapsed=30.0))
        for key, value in out.items():
            if key.startswith("law_") and key not in ("law_limit_mps", "law_offset_m"):
                assert (value <= 0.0).all(), key

    def test_the_total_is_the_sum_of_the_parts(self):
        law = enforcer([point("red_signal", 0.5, node_id=1)])
        out = law.charge(**state([0.55], [0.45], [30.0], pos=[[10.0, -6.0]], elapsed=30.0))
        parts = sum(
            out[k] for k in ("law_speeding", "law_wrong_side", "law_red_signal", "law_stop_line", "law_crossing")
        )
        assert float(out["law_total"][0]) == pytest.approx(float(parts[0]), rel=1e-5)

    def test_a_lawful_driver_pays_nothing(self):
        law = enforcer([])
        out = law.charge(**state([0.2], [0.1], [10.0], pos=[[10.0, 4.0]]))
        assert float(out["law_total"][0]) == 0.0

    def test_charges_are_per_body(self):
        law = enforcer()
        out = law.charge(**state([0.2, 0.2], [0.1, 0.1], [10.0, 30.0]))
        assert out["law_total"].shape == (2,)
        assert float(out["law_total"][0]) > float(out["law_total"][1])


class TestRealCircuit:
    CIRCUIT = ROOT / "data" / "tracks" / "kl.geojson"

    @pytest.mark.skipif(not CIRCUIT.exists(), reason="Malaysian circuit not built")
    def test_the_kl_lap_enforces_real_rules(self):
        from car_env import load_geojson_centerline
        from law import load_law
        from roadlaw import control_points_for_circuit, legal_profile_for_circuit

        pts, proj = load_geojson_centerline(self.CIRCUIT, return_projection=True)
        legal = legal_profile_for_circuit(self.CIRCUIT, pts.numpy(), proj)
        points = control_points_for_circuit(self.CIRCUIT, proj, load_law(), centerline=pts.numpy())
        law = LawEnforcer(legal, points, pts, DEVICE)

        assert law.driving_side == "left"
        assert law.has_signals
        out = law.charge(**state([0.2], [0.19], [40.0], pos=[[0.0, 0.0]]))
        assert float(out["law_speeding"][0]) < 0.0  # 144 km/h on a posted road
