"""The things that actually happen on a Malaysian road, staged near the driver.

Traffic that only flows is not traffic. A real street throws a person off a
kerb between parked cars, a child after a ball, a motorcycle up the gap between
two lanes, a lorry that brakes without warning, a car that cuts in with no
signal and one that makes a U-turn where it should not. This module stages
those, near enough to the car to matter and often enough to be learned from.

Every hazard is scripted from the survey's own geometry: a dart-out happens at
a kerb the survey drew, a red-runner crosses at a junction the survey mapped.
Nothing teleports into the lane out of nowhere.

Two properties matter for training:

  * reproducible - one seed gives one sequence of events on one lap, so two
    drivers can be compared on the same road with the same surprises;
  * not clairvoyant - the driver learns about a hazard the way it learns about
    everything else, by the camera seeing it and the detector reporting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scene import (
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_PERSON,
    CLASS_TRUCK,
    Actor,
    SceneConfig,
    frame_at,
    lane_halfwidth,
)

# What can happen. Each name is both the telemetry label and the key the rates
# below are looked up under.
DART_OUT = "pedestrian_dart"
CHILD_BALL = "child_after_ball"
TAILGATER = "tailgater"
SUDDEN_BRAKE = "lead_brakes_hard"
CUT_IN = "cut_in"
FILTERING = "motorcycle_filters"
ILLEGAL_UTURN = "illegal_uturn"
DOUBLE_PARKED = "double_parked"
JAYWALK = "jaywalk_group"
BUS_STOPS = "bus_halts_in_lane"
ANIMAL = "animal_crossing"
RED_RUNNER = "crossing_vehicle_runs_red"
BREAKDOWN = "broken_down_vehicle"

HAZARDS = (
    DART_OUT, CHILD_BALL, TAILGATER, SUDDEN_BRAKE, CUT_IN, FILTERING,
    ILLEGAL_UTURN, DOUBLE_PARKED, JAYWALK, BUS_STOPS, ANIMAL, RED_RUNNER, BREAKDOWN,
)


@dataclass(frozen=True)
class HazardRates:
    """How often each event is staged, in events per kilometre driven.

    These are exposure rates for training, not a claim about how often any of
    this happens on a real road. A driver that meets a dart-out once a year
    would never learn to expect one, so the rehearsal is denser than life.
    """

    per_km: dict[str, float] = field(
        default_factory=lambda: {
            DART_OUT: 1.6,
            CHILD_BALL: 0.5,
            TAILGATER: 1.2,
            SUDDEN_BRAKE: 1.0,
            CUT_IN: 1.4,
            FILTERING: 3.0,  # the most common of all on a Malaysian street
            ILLEGAL_UTURN: 0.4,
            DOUBLE_PARKED: 1.0,
            JAYWALK: 0.8,
            BUS_STOPS: 0.5,
            ANIMAL: 0.4,
            RED_RUNNER: 0.3,
            BREAKDOWN: 0.3,
        }
    )

    # Where ahead of the car an event is staged. Close enough to demand a
    # response, far enough that a response exists.
    stage_ahead_m: tuple[float, float] = (18.0, 70.0)
    behind_m: tuple[float, float] = (6.0, 20.0)  # for the ones that come from behind
    lifetime_s: float = 14.0
    min_gap_s: float = 2.0  # between two staged events, so they do not pile up


@dataclass
class Hazard:
    """One staged event and the actors it owns."""

    kind: str
    actors: list[Actor]
    age_s: float = 0.0
    lifetime_s: float = 14.0
    started_at_m: float = 0.0

    @property
    def expired(self) -> bool:
        return self.age_s >= self.lifetime_s


class Director:
    """Stages hazards near the driver, and gives every actor its behaviour.

    The director owns three jobs: keep ordinary traffic flowing in its lane and
    stopping at red, decide when to stage an event, and run the events it has
    staged until they finish.
    """

    def __init__(
        self,
        centerline: np.ndarray,
        heights: np.ndarray,
        profile,
        traffic,
        seed: int = 0,
        rates: HazardRates | None = None,
        cfg: SceneConfig | None = None,
    ) -> None:
        self.centerline = np.asarray(centerline, dtype=np.float64)
        self.heights = np.asarray(heights, dtype=np.float64)
        self.profile = profile
        self.traffic = traffic
        self.cfg = cfg or SceneConfig()
        self.rates = rates or HazardRates()
        self.rng = np.random.default_rng(seed)
        self.tangent, self.normal = frame_at(self.centerline)
        self.half = lane_halfwidth(profile, self.cfg)
        self.n = self.centerline.shape[0]
        spacing = float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).mean())
        self.lap_m = spacing * self.n
        self.side = -1.0 if (profile.driving_side or "left") == "left" else 1.0
        self.active: list[Hazard] = []
        self.history: dict[str, int] = {}
        self.distance_m = 0.0
        self._since_last_s = 0.0

    # ---- geometry helpers -------------------------------------------------

    def index_at(self, progress: float) -> int:
        return int(progress * self.n) % self.n

    def point_at(self, progress: float, lateral_m: float = 0.0) -> np.ndarray:
        i = self.index_at(progress)
        base = self.centerline[i] + self.normal[i] * lateral_m
        return np.array([base[0], base[1], self.heights[i]])

    def heading_at(self, progress: float) -> float:
        i = self.index_at(progress)
        return float(np.arctan2(self.tangent[i, 1], self.tangent[i, 0]))

    def ahead_of(self, progress: float, metres: float) -> float:
        return float((progress + metres / self.lap_m) % 1.0)

    def lane_centre(self, progress: float, lane: int = 0) -> float:
        """Lateral offset of a lane centre, positive to the left of the road."""
        i = self.index_at(progress)
        inner = self.half[i] * 0.45
        return self.side * (inner + lane * self.cfg.lane_width_m)

    # ---- staging ----------------------------------------------------------

    def _spawn(self, kind: str, actors: list[Actor], progress: float) -> Hazard:
        for actor in actors:
            actor.hazard = kind
        hazard = Hazard(kind=kind, actors=actors, lifetime_s=self.rates.lifetime_s, started_at_m=self.distance_m)
        self.active.append(hazard)
        self.traffic.actors.extend(actors)
        self.history[kind] = self.history.get(kind, 0) + 1
        return hazard

    def _person(self, pos: np.ndarray, heading: float, speed: float, height: float = 1.7) -> Actor:
        return Actor(
            kind=CLASS_PERSON,
            pos=pos,
            heading=heading,
            size=np.array([0.5, 0.5, height]),
            colour=self.rng.uniform(0.25, 0.85, size=3).astype(np.float32),
            behaviour="free",
            velocity=np.array([np.cos(heading) * speed, np.sin(heading) * speed, 0.0]),
            ttl_s=self.rates.lifetime_s,
        )

    def _vehicle(self, kind: str, progress: float, lateral: float, speed: float) -> Actor:
        heading = self.heading_at(progress)
        size = {
            CLASS_CAR: (self.cfg.car_length_m, self.cfg.car_width_m, self.cfg.car_height_m),
            CLASS_TRUCK: (7.2, 2.4, 3.1),
            CLASS_MOTORCYCLE: (
                self.cfg.motorcycle_length_m,
                self.cfg.motorcycle_width_m,
                self.cfg.motorcycle_height_m,
            ),
        }[kind]
        return Actor(
            kind=kind,
            pos=self.point_at(progress, lateral),
            heading=heading,
            size=np.array(size),
            colour=self.rng.uniform(0.2, 0.85, size=3).astype(np.float32),
            progress=progress,
            speed_mps=speed,
            lane_offset_m=lateral,
            target_speed_mps=speed,
            ttl_s=self.rates.lifetime_s,
        )

    def stage(self, kind: str, ego_progress: float, ego_speed: float) -> Hazard | None:
        """Put one event on the road ahead of (or behind) the driver."""
        ahead = float(self.rng.uniform(*self.rates.stage_ahead_m))
        at = self.ahead_of(ego_progress, ahead)
        cruise = max(ego_speed, 5.0)

        if kind in (DART_OUT, CHILD_BALL, JAYWALK, ANIMAL):
            i = self.index_at(at)
            kerb = self.side * (self.half[i] + 0.8)
            start = self.point_at(at, kerb)
            across = float(np.arctan2(-self.normal[i, 1] * self.side, -self.normal[i, 0] * self.side))
            if kind == DART_OUT:
                return self._spawn(kind, [self._person(start, across, float(self.rng.uniform(1.6, 3.2)))], at)
            if kind == ANIMAL:
                dog = self._person(start, across, float(self.rng.uniform(2.5, 4.5)), height=0.45)
                dog.size = np.array([0.7, 0.3, 0.45])
                return self._spawn(kind, [dog], at)
            if kind == CHILD_BALL:
                ball = self._person(start, across, float(self.rng.uniform(3.5, 5.0)), height=0.22)
                ball.size = np.array([0.22, 0.22, 0.22])
                child = self._person(start, across, float(self.rng.uniform(2.2, 3.4)), height=1.15)
                child.size = np.array([0.4, 0.4, 1.15])
                return self._spawn(kind, [ball, child], at)
            group = [
                self._person(
                    start + np.array([float(self.rng.normal(0, 1.2)), float(self.rng.normal(0, 1.2)), 0.0]),
                    across,
                    float(self.rng.uniform(1.0, 1.8)),
                )
                for _ in range(int(self.rng.integers(2, 5)))
            ]
            return self._spawn(kind, group, at)

        if kind == TAILGATER:
            behind = self.ahead_of(ego_progress, -float(self.rng.uniform(*self.rates.behind_m)))
            actor = self._vehicle(CLASS_CAR, behind, self.lane_centre(behind), cruise * 1.25)
            actor.behaviour = "tailgate"
            return self._spawn(kind, [actor], behind)

        if kind == SUDDEN_BRAKE:
            actor = self._vehicle(
                CLASS_TRUCK if self.rng.uniform() < 0.4 else CLASS_CAR, at, self.lane_centre(at), cruise
            )
            actor.behaviour = "brake_hard"
            return self._spawn(kind, [actor], at)

        if kind == CUT_IN:
            actor = self._vehicle(CLASS_CAR, at, self.lane_centre(at, 1), cruise * 1.05)
            actor.behaviour = "cut_in"
            return self._spawn(kind, [actor], at)

        if kind == FILTERING:
            behind = self.ahead_of(ego_progress, -float(self.rng.uniform(4.0, 14.0)))
            bike = self._vehicle(CLASS_MOTORCYCLE, behind, self.lane_centre(behind) * 0.15, cruise * 1.35)
            bike.behaviour = "filter"
            return self._spawn(kind, [bike], behind)

        if kind == ILLEGAL_UTURN:
            actor = self._vehicle(CLASS_CAR, at, -self.lane_centre(at), cruise * 0.35)
            actor.behaviour = "uturn"
            return self._spawn(kind, [actor], at)

        if kind in (DOUBLE_PARKED, BREAKDOWN):
            lateral = self.lane_centre(at) * (1.35 if kind == DOUBLE_PARKED else 1.0)
            actor = self._vehicle(CLASS_CAR, at, lateral, 0.0)
            actor.behaviour = "parked"
            actor.target_speed_mps = 0.0
            hazard = self._spawn(kind, [actor], at)
            hazard.lifetime_s = self.rates.lifetime_s * 2.0
            return hazard

        if kind == BUS_STOPS:
            actor = self._vehicle(CLASS_TRUCK, at, self.lane_centre(at), cruise * 0.5)
            actor.behaviour = "halt"
            return self._spawn(kind, [actor], at)

        if kind == RED_RUNNER:
            i = self.index_at(at)
            side_start = self.point_at(at, -self.side * (self.half[i] + 12.0))
            across = float(np.arctan2(self.normal[i, 1] * self.side, self.normal[i, 0] * self.side))
            actor = Actor(
                kind=CLASS_CAR,
                pos=side_start,
                heading=across,
                size=np.array([self.cfg.car_length_m, self.cfg.car_width_m, self.cfg.car_height_m]),
                colour=self.rng.uniform(0.2, 0.85, size=3).astype(np.float32),
                behaviour="free",
                velocity=np.array([np.cos(across), np.sin(across), 0.0]) * float(self.rng.uniform(7.0, 12.0)),
                ttl_s=self.rates.lifetime_s,
            )
            return self._spawn(kind, [actor], at)
        return None

    # ---- stepping ---------------------------------------------------------

    def step(self, dt_s: float, ego_progress: float, ego_pos: np.ndarray, ego_speed: float) -> list[str]:
        """Advance traffic and events by `dt_s`; returns the events now running."""
        self.distance_m += max(ego_speed, 0.0) * dt_s
        self._since_last_s += dt_s
        self._advance_actors(dt_s, ego_progress, ego_pos, ego_speed)
        self._retire()
        self._maybe_stage(dt_s, ego_progress, ego_speed)
        return [h.kind for h in self.active]

    def _maybe_stage(self, dt_s: float, ego_progress: float, ego_speed: float) -> None:
        if self._since_last_s < self.rates.min_gap_s or ego_speed <= 0.5:
            return
        travelled_km = ego_speed * dt_s / 1000.0
        for kind, per_km in self.rates.per_km.items():
            if self.rng.uniform() < per_km * travelled_km:
                if self.stage(kind, ego_progress, ego_speed) is not None:
                    self._since_last_s = 0.0
                    return  # one at a time, so events stay legible

    def _advance_actors(self, dt_s: float, ego_progress: float, ego_pos: np.ndarray, ego_speed: float) -> None:
        for actor in self.traffic.actors:
            if actor.ttl_s > 0.0:
                actor.ttl_s -= dt_s
            if actor.behaviour == "free":
                if actor.velocity is not None:
                    actor.pos = actor.pos + actor.velocity * dt_s
                continue
            self._advance_lane_actor(actor, dt_s, ego_progress, ego_speed)

    def _advance_lane_actor(self, actor: Actor, dt_s: float, ego_progress: float, ego_speed: float) -> None:
        """One lane-following actor, including whatever its event tells it to do."""
        if actor.progress < 0.0:
            return
        gap = ((actor.progress - ego_progress + 0.5) % 1.0 - 0.5) * self.lap_m  # metres ahead of the car

        if actor.behaviour == "brake_hard" and 0.0 < gap < 45.0:
            actor.target_speed_mps = 0.0
        elif actor.behaviour == "halt" and 0.0 < gap < 35.0:
            actor.target_speed_mps = 0.0
        elif actor.behaviour == "tailgate":
            # Sits about a second behind and matches speed, which is what makes
            # it a tailgater rather than an overtaker.
            actor.target_speed_mps = ego_speed * (1.35 if gap < -8.0 else 0.98)
        elif actor.behaviour == "filter":
            actor.target_speed_mps = max(ego_speed * 1.3, 6.0)
            # Slides across to the lane line as it comes alongside.
            actor.lane_offset_m += (0.0 - actor.lane_offset_m) * min(dt_s * 1.5, 1.0)
        elif actor.behaviour == "cut_in" and -25.0 < gap < 25.0:
            target = self.lane_centre(actor.progress)
            actor.lane_offset_m += (target - actor.lane_offset_m) * min(dt_s * 1.2, 1.0)
        elif actor.behaviour == "uturn":
            actor.target_speed_mps = 3.0
            target = self.lane_centre(actor.progress)
            actor.lane_offset_m += (target - actor.lane_offset_m) * min(dt_s * 0.6, 1.0)
        elif actor.behaviour == "parked":
            actor.target_speed_mps = 0.0

        rate = 6.0 if actor.target_speed_mps < actor.speed_mps else 2.5
        actor.speed_mps += np.clip(actor.target_speed_mps - actor.speed_mps, -rate * dt_s, rate * dt_s)
        actor.speed_mps = max(actor.speed_mps, 0.0)
        actor.progress = (actor.progress + actor.speed_mps * dt_s / self.lap_m) % 1.0
        actor.pos = self.point_at(actor.progress, actor.lane_offset_m)
        actor.heading = self.heading_at(actor.progress)
        if actor.behaviour == "uturn":
            actor.heading += np.pi

    def _retire(self) -> None:
        """Drop finished events and the actors they brought."""
        # Compared by identity: an `Actor` holds arrays, so `==` on a hazard
        # would compare them elementwise and raise.
        expired = {id(h) for h in self.active if all(a.ttl_s <= 0.0 for a in h.actors)}
        if not expired:
            return
        gone = {id(a) for h in self.active if id(h) in expired for a in h.actors}
        self.traffic.actors = [a for a in self.traffic.actors if id(a) not in gone]
        self.active = [h for h in self.active if id(h) not in expired]

    # ---- reporting --------------------------------------------------------

    def summary(self) -> dict[str, int]:
        return dict(sorted(self.history.items()))

    def nearest_hazard_m(self, ego_pos: np.ndarray) -> float:
        """Metres to the closest actor belonging to a running event, or inf."""
        best = float("inf")
        for hazard in self.active:
            for actor in hazard.actors:
                best = min(best, float(np.linalg.norm(actor.pos[:2] - np.asarray(ego_pos)[:2])))
        return best
