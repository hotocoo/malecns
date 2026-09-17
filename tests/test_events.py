"""Tests for the events a street throws at a driver.

Two properties carry the weight: an event is staged from the survey's geometry
rather than dropped into the lane from nowhere, and the same seed gives the
same sequence so two drivers can be judged on the same road.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from events import (  # noqa: E402
    BREAKDOWN,
    CHILD_BALL,
    CUT_IN,
    DART_OUT,
    DOUBLE_PARKED,
    FILTERING,
    HAZARDS,
    RED_RUNNER,
    SUDDEN_BRAKE,
    TAILGATER,
    Director,
    HazardRates,
)
from roadlaw import LegalProfile  # noqa: E402
from scene import CLASS_MOTORCYCLE, CLASS_PERSON, SceneConfig, Traffic  # noqa: E402

N = 256
RADIUS = 300.0


def ring() -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    return np.stack([RADIUS * np.cos(theta), RADIUS * np.sin(theta)], axis=1)


def profile(side: str = "left") -> LegalProfile:
    return LegalProfile(
        limit_mps=np.full(N, 50.0 / 3.6),
        lanes=np.full(N, 2, dtype=np.int64),
        oneway=np.zeros(N, dtype=np.int64),
        tunnel=np.zeros(N, dtype=bool),
        roundabout=np.zeros(N, dtype=bool),
        way_id=np.zeros(N, dtype=np.int64),
        match_m=np.zeros(N),
        driving_side=side,
        source="test",
    )


def director(seed: int = 0, vehicles: int = 6, rates: HazardRates | None = None) -> Director:
    line, heights, legal = ring(), np.zeros(N), profile()
    traffic = Traffic.populate(line, heights, legal, (), vehicles=vehicles, motorcycles=4, pedestrians=2, seed=seed)
    return Director(line, heights, legal, traffic, seed=seed, rates=rates, cfg=SceneConfig())


def drive(d: Director, seconds: float = 120.0, speed: float = 14.0, dt: float = 0.05) -> None:
    progress = 0.0
    for _ in range(int(seconds / dt)):
        progress = (progress + speed * dt / d.lap_m) % 1.0
        d.step(dt, progress, d.point_at(progress, d.lane_centre(progress)), speed)


class TestGeometry:
    def test_a_point_on_the_lap_is_on_the_ring(self):
        d = director()
        point = d.point_at(0.25, 0.0)
        assert float(np.hypot(point[0], point[1])) == pytest.approx(RADIUS, rel=1e-3)

    def test_a_lane_centre_sits_inside_the_carriageway(self):
        d = director()
        assert abs(d.lane_centre(0.1)) < d.half.max()

    def test_looking_ahead_wraps_the_lap(self):
        d = director()
        assert 0.0 <= d.ahead_of(0.99, 500.0) < 1.0


class TestStaging:
    @pytest.mark.parametrize("kind", HAZARDS)
    def test_every_event_can_be_staged(self, kind):
        d = director()
        hazard = d.stage(kind, 0.2, 14.0)
        assert hazard is not None
        assert hazard.kind == kind
        assert hazard.actors

    def test_a_staged_actor_joins_the_traffic(self):
        d = director()
        before = len(d.traffic.actors)
        d.stage(DART_OUT, 0.2, 14.0)
        assert len(d.traffic.actors) > before

    def test_a_dart_out_starts_beside_the_road_not_in_it(self):
        d = director()
        person = d.stage(DART_OUT, 0.2, 14.0).actors[0]
        distance = float(np.hypot(person.pos[0], person.pos[1]))
        assert abs(distance - RADIUS) > d.half.mean() * 0.8

    def test_a_child_comes_with_a_ball(self):
        d = director()
        hazard = d.stage(CHILD_BALL, 0.2, 14.0)
        assert len(hazard.actors) == 2
        heights = sorted(float(a.size[2]) for a in hazard.actors)
        assert heights[0] < 0.4 < heights[1]  # a ball and a child, not two adults

    def test_a_tailgater_starts_behind(self):
        d = director()
        hazard = d.stage(TAILGATER, 0.5, 14.0)
        gap = ((hazard.actors[0].progress - 0.5 + 0.5) % 1.0 - 0.5) * d.lap_m
        assert gap < 0.0

    def test_a_filtering_motorcycle_is_a_motorcycle(self):
        d = director()
        assert d.stage(FILTERING, 0.3, 14.0).actors[0].kind == CLASS_MOTORCYCLE

    def test_a_parked_car_does_not_move(self):
        d = director()
        actor = d.stage(DOUBLE_PARKED, 0.3, 14.0).actors[0]
        assert actor.target_speed_mps == 0.0

    def test_a_red_runner_crosses_the_road(self):
        d = director()
        actor = d.stage(RED_RUNNER, 0.3, 14.0).actors[0]
        assert actor.behaviour == "free"
        assert float(np.linalg.norm(actor.velocity)) > 1.0

    def test_events_are_tallied(self):
        d = director()
        d.stage(CUT_IN, 0.3, 14.0)
        d.stage(CUT_IN, 0.4, 14.0)
        assert d.summary()[CUT_IN] == 2


class TestBehaviour:
    def test_a_person_stepping_out_moves_across_the_road(self):
        d = director()
        person = d.stage(DART_OUT, 0.2, 14.0).actors[0]
        before = float(np.hypot(person.pos[0], person.pos[1]))
        for _ in range(20):
            d.step(0.05, 0.2, d.point_at(0.2, 0.0), 14.0)
        after = float(np.hypot(person.pos[0], person.pos[1]))
        assert abs(after - RADIUS) < abs(before - RADIUS)  # heading for the road

    def test_a_lead_vehicle_braking_slows_down(self):
        d = director()
        actor = d.stage(SUDDEN_BRAKE, 0.2, 14.0).actors[0]
        actor.progress = 0.201  # right in front of the car
        opening = actor.speed_mps
        for _ in range(30):
            d.step(0.05, 0.2, d.point_at(0.2, 0.0), 14.0)
        assert actor.speed_mps < opening

    def test_a_cut_in_moves_towards_the_drivers_lane(self):
        d = director()
        actor = d.stage(CUT_IN, 0.2, 14.0).actors[0]
        actor.progress = 0.2005
        start = abs(actor.lane_offset_m - d.lane_centre(actor.progress))
        for _ in range(30):
            d.step(0.05, 0.2, d.point_at(0.2, 0.0), 14.0)
        assert abs(actor.lane_offset_m - d.lane_centre(actor.progress)) < start

    def test_ordinary_traffic_keeps_its_lane(self):
        d = director(vehicles=6)
        lanes = [a.lane_offset_m for a in d.traffic.actors if a.progress >= 0.0]
        drive(d, seconds=5.0)
        after = [a.lane_offset_m for a in d.traffic.actors if a.progress >= 0.0 and not a.hazard]
        assert after and all(abs(v) > 0.1 for v in after)  # not collapsed onto the centreline

    def test_pedestrians_do_not_drive_down_the_lane(self):
        d = director()
        people = [a for a in d.traffic.actors if a.kind == CLASS_PERSON and not a.hazard]
        assert people
        assert all(a.progress < 0.0 for a in people)


class TestOverTime:
    def test_driving_stages_events(self):
        d = director()
        drive(d, seconds=200.0)
        assert sum(d.summary().values()) > 0

    def test_one_seed_gives_one_sequence(self):
        a, b = director(seed=5), director(seed=5)
        drive(a, seconds=120.0)
        drive(b, seconds=120.0)
        assert a.summary() == b.summary()

    def test_different_seeds_differ(self):
        a, b = director(seed=1), director(seed=2)
        drive(a, seconds=200.0)
        drive(b, seconds=200.0)
        assert a.summary() != b.summary()

    def test_finished_events_are_cleaned_up(self):
        d = director()
        drive(d, seconds=300.0)
        staged = sum(d.summary().values())
        assert staged > 0
        assert len(d.active) < staged  # the old ones have gone

    def test_a_standing_car_gets_no_events(self):
        d = director()
        for _ in range(400):
            d.step(0.05, 0.1, d.point_at(0.1, 0.0), 0.0)
        assert sum(d.summary().values()) == 0

    def test_nearest_hazard_is_infinite_when_nothing_is_running(self):
        d = director()
        assert d.nearest_hazard_m(np.zeros(2)) == float("inf")

    def test_nearest_hazard_measures_a_running_one(self):
        d = director()
        hazard = d.stage(DART_OUT, 0.2, 14.0)
        here = d.point_at(0.2, 0.0)
        assert d.nearest_hazard_m(here) < 1e6

    def test_rates_control_how_often_events_appear(self):
        quiet = HazardRates(per_km={k: 0.0 for k in HAZARDS})
        d = director(rates=quiet)
        drive(d, seconds=300.0)
        assert sum(d.summary().values()) == 0
