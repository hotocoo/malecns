"""Batched top-down car driving environment (pure torch, no external deps).

A closed loop track, either a smooth random centerline or a real circuit
loaded from GeoJSON, is rasterised once into an occupancy grid plus progress
and clearance fields, so lidar ray-marching, collision and reward lookups are
constant-time grid samples. Every quantity is batched, so one environment
object drives a whole ES population at once.

Vehicle. A single-track (bicycle) model with a friction-circle grip envelope,
parameterised for a 2020 Mercedes-AMG F1 W11 by default (`VEHICLE_W11`):
power-limited acceleration, aerodynamic drag, downforce-dependent grip,
steering actuator lag and a real body width. Lateral demand from steering is
served first; braking and traction get what is left of the grip circle, so the
car has to slow for corners and understeers when it does not.

Observation (per car): `n_rays` normalised lidar distances + normalised speed.
Action: steering in [-1, 1] (positive = left / counter-clockwise), pedal in
[-1, 1] (positive = throttle, negative = brake). There is no reverse gear.

Episode ends on: leaving the track (crash), net reverse progress, or stuck
(too little progress over `stuck_window_s`). The step cap lives in the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

EARTH_RADIUS_M = 6_371_000.0
G = 9.81

DONE_ALIVE, DONE_CRASH, DONE_REVERSE, DONE_STUCK, DONE_FINISH = 0, 1, 2, 3, 4
DONE_NAMES = {DONE_ALIVE: "alive", DONE_CRASH: "crash", DONE_REVERSE: "reverse", DONE_STUCK: "stuck", DONE_FINISH: "finished"}


@dataclass(frozen=True)
class CarConfig:
    # --- sensing -----------------------------------------------------------------
    # 19 rays over 180 degrees: one every 10 degrees. At 9 rays the spacing was
    # 22.5 degrees, which at 30 m ahead puts neighbouring rays 11 m apart, wider
    # than the road: a barrier could sit entirely between two rays and never be
    # seen, and the shape of a corner was not resolvable at all.
    n_rays: int = 19
    fov_deg: float = 180.0
    max_range: float = 150.0
    march_steps: int = 96
    # Ray samples are spaced as (i/N)^march_power * max_range: fine near the
    # car, coarse far away, so 96 samples cover 150 m with 0.3 m near the body.
    march_power: float = 1.5
    # The march only locates the sample where the ray first leaves the road, and
    # its spacing grows to 2.4 m at the far end, so the reported distance was
    # biased long by up to 4 m: the car was told it had room it did not have.
    # Bisecting between the last free sample and the first blocked one this many
    # times brings that under 0.2 m for four extra road lookups per ray.
    ray_refine_steps: int = 4
    dt_s: float = 0.016
    # --- vehicle (Mercedes-AMG F1 W11, 2020; public figures) ---------------------
    wheelbase: float = 3.70
    car_halfwidth: float = 1.00  # 2.0 m body
    car_length: float = 5.70  # nose to tail
    tyre_radius_m: float = 0.36
    vehicle_name: str = "Mercedes-AMG F1 W11"
    # Collision is tested on the oriented body rectangle: this many points
    # along each long side (corners included), so a nose or tail poking into
    # the barrier at an angle crashes even while the centre is clear.
    body_samples: int = 3
    mass_kg: float = 795.0  # 746 kg minimum plus a Monaco fuel load
    power_w: float = 750_000.0  # ~1,000 hp combined ICE + MGU-K
    cda_m2: float = 1.60  # high-downforce configuration
    roll_decel: float = 0.30  # rolling + driveline losses, m/s^2
    traction_g: float = 1.15  # launch traction limit (0-100 km/h in ~2.6 s)
    grip_mech_g: float = 1.80  # mechanical grip at rest
    grip_aero_g: float = 2.80  # extra grip from downforce at `grip_v_ref`
    grip_v_ref: float = 80.0
    grip_max_g: float = 4.50
    max_steer_rad: float = 0.42  # ~24 deg road-wheel lock (Monaco rack) -> 8.3 m minimum radius
    steer_tau_s: float = 0.08  # steering actuator lag; 0 is instantaneous
    max_speed: float = 95.0  # aero-limited, ~340 km/h
    # --- track -------------------------------------------------------------------
    track_halfwidth: float = 5.5  # 11 m road, Monaco average
    grid_res: int = 2048
    grid_extent: float = 130.0  # grows automatically to fit the circuit
    # "loop": procedural random circuit from `seed`. "geojson": the circuit in
    # `geojson_path`, scaled by `track_scale`, resampled to `n_points`.
    layout: str = "loop"
    geojson_path: str = "data/tracks/monaco.geojson"
    track_scale: float = 1.0
    n_points: int = 2048
    smooth_m: float = 8.0  # keeps the Fairmont hairpin centerline at ~8.5 m radius
    loop_scale: float = 4.0  # procedural loops: 70 m base radius times this
    loop_difficulty: float = 1.0  # harmonic amplitude multiplier
    mirror: bool = False  # drive the circuit the other way round
    # Real circuits climb: with a survey beside the GeoJSON (`<stem>_dem.json`,
    # see fetch_terrain.py) gravity along the road acts on the car.
    road_grade: bool = True
    # --- termination -------------------------------------------------------------
    # Net progress below this ends the episode as a crash. A car that turns
    # round and drives the loop backwards otherwise survives, and evolution
    # finds that before it finds cornering.
    reverse_limit_laps: float = -0.01
    # Stuck: less than `stuck_min_m` of net progress over `stuck_window_s`, a
    # pace floor of 3 m/s (11 km/h). With no step cap a crawler would otherwise
    # hold a whole generation open for hours.
    stuck_window_s: float = 4.0
    stuck_min_m: float = 12.0
    # Finished: `max_laps` completed (0 disables). Episodes have no step cap; a
    # car drives until it crashes, stalls, reverses or completes the lap, so
    # the only way to score more is to get round, and, via the time tax, faster.
    max_laps: float = 1.0
    # --- reward ------------------------------------------------------------------
    # Progress along the track is the only thing paid for, at `progress_per_m`
    # per metre, so different circuits pay the same for the same driving. No
    # speed bonus: it paid for speed in any direction. The wall term gives a
    # smooth ramp inside `wall_margin` metres of the body's edge so the search
    # sees a gradient before the cliff of a crash.
    progress_per_m: float = 0.10
    # Flat reward for each *newly* completed lap (crossing the line back and
    # forth pays once). Off by default, and it should stay off: at 100 it was
    # 30 per cent of a Monaco lap's progress pay handed over at the line
    # regardless of how the lap was driven, so an episode that accumulated -70
    # of wall, pace, alignment and time charges still scored +30 and evolution
    # read a scraped, 60 km/h, 197 s lap as a success. Crossing the line is
    # already paid: the metres of it earn `progress_per_m` like any other
    # metres, and finishing ends the episode, which under a step budget saves
    # the full `time_tax + pace_penalty + align_penalty` charge on every unused
    # step (about 0.09/step, hundreds of points) - an incentive that grows the
    # faster the lap is, which a flat bonus is not.
    lap_bonus: float = 0.0
    # Per control step, so 1.75 points per second at a 16 ms step: once two
    # cars both get round, this and the pace term are the whole difference
    # between them. At 0.02 a lap 10 s quicker was worth about 20 points
    # against a lap worth 657, inside the spread between start points, and the
    # search had no reason to prefer it. The unfinished-lap forfeit is what
    # keeps this from making driving on a losing move: an extra metre is worth
    # `progress_per_m + unfinished_per_m` = 0.30 against roughly 0.12 of
    # charges at racing pace.
    time_tax: float = 0.035
    # 0.75 m from the body's edge: an 11 m road leaves 4.5 m of clearance at
    # the centre, and a Monaco line clips barriers at arm's length. The old
    # 1.5 m margin taxed the outer 3 m of usable road and kept the apex out of
    # reach; 0.75 m still gives the search a gradient before the crash cliff.
    # 0.4 m from the body's edge, half the old margin, at half the old rate: an
    # F1 line puts the wheels on the kerb, and taxing the outer road taught the
    # car to drive down the middle. What is left is a gradient in the last half
    # metre before the barrier so the search feels the cliff coming.
    wall_margin: float = 0.4
    wall_penalty: float = 0.015
    crash_penalty: float = 20.0
    # Forfeit for the lap not completed, per metre of it, charged at any ending
    # that is not the finish (crash, reverse, stuck). Uncapped episodes
    # (`episode_steps = 0`, the default) removed the budget charge below and
    # with it the guarantee that driving on beats ending: at the trainer's
    # 77 km/h cruise one control step paid +0.034 of progress and cost 0.041 of
    # time tax, pace and alignment, so every extra metre driven *lowered* the
    # score while an early crash cost a flat 20. Evolution stalled at 0.58 laps
    # with a negative return. Charging the unfinished metres at
    # `unfinished_per_m` restores the invariant without a clock: ending at
    # fraction f of the lap forfeits `(1 - f) * length_m * unfinished_per_m`, so
    # driving one more metre is worth `progress_per_m + unfinished_per_m` minus
    # that metre's charges - positive by a wide margin at any sane pace. A
    # completed lap pays the whole distance and forfeits nothing, so the only
    # way left to score higher is to finish, and to finish sooner.
    # At 0.10 (equal to `progress_per_m`) the margin was still thin: an extra
    # metre was worth 0.2 against per-metre charges of about 0.17 at the pace
    # the trainer actually drives. 0.20 makes finishing the dominant term and
    # leaves the per-step charges as the tiebreaker between two laps that both
    # get round, which is what "fastest lap" means here.
    unfinished_per_m: float = 0.20
    # Step budget the caller runs the episode for (0 = uncapped). With a budget
    # a car that ends early (crash, reverse, stuck) is treated as standing still
    # for the steps it did not drive: it is charged the time tax plus the full
    # pace penalty for each of them (`crash_cost` in `last_terms`). No survivor
    # pays more per step than that, so dying never scores above driving on;
    # only finishing the lap early saves any of the budget. Before this a car
    # holding 36 km/h for 12,000 steps scored -75 while one crashing on step
    # 400 scored -28, and evolution was paid for the crash.
    episode_steps: int = 0
    # Speed penalty, (speed/max_speed)^2 per step. Off by default: the goal is
    # the fastest clean lap, and crashes already cost `crash_penalty` plus the
    # progress not made. At 0.05 it charged a 60 m/s straight as much as the
    # time tax and pulled evolution towards slow driving.
    speed_penalty: float = 0.0
    speed_free_fraction: float = 0.25
    # Discourage the degenerate low-speed limit cycle seen in the ES collapse:
    # the controller holds near-full steering while barely moving, then gets
    # terminated by the stuck detector. This only activates below `stall_speed`
    # and above half-lock, so normal cornering and low-speed hairpins retain
    # their steering authority while a stationary steering attractor becomes
    # visibly worse than centering and accelerating out.
    # Pace: below `pace_margin` of the reference speed the road allows at the
    # car's position (`Track.speed_ref`: the fastest this vehicle's grip circle,
    # power, drag and brakes can pass each centerline sample, see
    # `speed_profile`) a quadratic deficit is charged per step. Progress pays
    # for speed everywhere alike; this charges slowness only where there is
    # room to go faster (straights, corner exits), where a 56 km/h cruise on a
    # 250 km/h straight used to cost nothing beyond the flat time tax. A car
    # standing still pays `pace_penalty` per step, the most any survivor pays.
    # `pace_margin` is the fraction of that reference speed the car is charged
    # against: at 0.9 a car already doing 90 per cent of what the road allows
    # paid nothing more for going faster, so the only remaining pull towards a
    # quicker lap was the flat time tax (1.25 points per second). Charging to
    # 1.0 keeps a gradient all the way to the limit. With the finish bonus gone
    # the pace term is the main thing separating a fast lap from a slow one, so
    # it is worth more than a wall scrape per step: a Monaco lap at the limit
    # now scores about 43 points above the same car 20 seconds slower.
    # Cut from 0.06 to a third: at 0.06 the pace term alone charged 3.75 per
    # second against a progress pay of 2.1 per second at the trainer's cruise,
    # so the per-step return of driving was negative and the search had nothing
    # to climb. It is a shaping term for where the road allows more speed, not
    # the objective; the objective is finishing, then finishing sooner, and
    # `time_tax` carries that everywhere while this only speaks where the road
    # allows more. At 0.03 the two together still left a car cruising at a
    # quarter of the reference pace marginally better off crashing.
    # ...and at 0.02 it was too quiet to ask for speed at all: the population
    # cruised at 77 km/h and never once saturated the throttle (pedal at the
    # stop on 0.07 per cent of steps) on a circuit whose straights allow 290.
    # A penalty for being slow and a bonus for being fast have the same
    # gradient; only the bonus cannot make driving on a losing move, so the
    # pace term is now paid, not charged, and `pace_penalty` stays at 0.
    pace_penalty: float = 0.0
    # Paid per step for the speed the road allows at the car's position,
    # (speed / speed_ref)^2, so it is worth the most on the straights, where a
    # 290 km/h section used to pay exactly what a 90 km/h one did.
    pace_bonus: float = 0.06
    # The reference is the fastest pass over the *centerline*. A car that uses
    # the full width straightens the corner and can legitimately beat it, so
    # the ratio is clamped above 1: this is what pays for an out-in-out line
    # and an apex instead of tracking the middle of the road.
    pace_cap: float = 1.3
    pace_margin: float = 1.0
    # Alignment: heading error to a look-ahead point on the centerline,
    # max(`align_lookahead_m`, `align_lookahead_s` of travel) ahead. Graded, not
    # a fixed fee: 0 when pointed at it, one `align_penalty` at 90 degrees off,
    # twice that facing backwards, so steering 0.8 of what a corner needs is
    # paid between steering it fully and not at all. The look-ahead grows with
    # speed, so an apex a few metres off the centerline costs a few degrees.
    # Cut from 0.03 and given a dead zone: a racing line points across the road
    # on corner entry and exit, so charging every degree off the centerline
    # look-ahead is charging for the only line that is quick. What is left is
    # an anti-wandering term: free inside `align_free_rad`, then graded, so
    # facing sideways or backwards is still worse than pointing down the road.
    align_penalty: float = 0.012
    # 35 degrees: wider than any racing line's yaw against the centerline
    # heading at Monaco, narrower than a spin.
    align_free_rad: float = 0.61
    align_lookahead_m: float = 15.0
    align_lookahead_s: float = 1.0
    stall_speed_mps: float = 5.0
    stall_steer_start: float = 0.5
    stall_steer_penalty: float = 0.12

    @property
    def min_turn_radius(self) -> float:
        return self.wheelbase / float(np.tan(self.max_steer_rad))


def monaco_config(dt_s: float, halfwidth: float | None = None, mirror: bool = False) -> CarConfig:
    """Circuit de Monaco at full scale: 3.29 km smoothed lap (3.337 km official), 11 m road, ~0.5 m cells."""
    return CarConfig(
        dt_s=dt_s,
        layout="geojson",
        track_halfwidth=5.5 if halfwidth is None else halfwidth,
        grid_res=2048,
        track_scale=1.0,
        mirror=mirror,
    )


def make_centerline(
    seed: int,
    n_points: int = 1024,
    device: torch.device | None = None,
    scale: float = 1.0,
    difficulty: float = 1.0,
) -> torch.Tensor:
    """Smooth closed loop: radius modulated by a few low-frequency harmonics.

    `difficulty` scales the harmonic amplitudes: 1.0 is the training family,
    1.6 gives tighter corners for stress tests (still above the minimum turn
    radius of the default vehicle at `scale` 4).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.linspace(0, 2 * torch.pi, n_points + 1)[:-1]
    radius = torch.full_like(theta, 70.0)
    for harmonic in (2, 3, 4, 5):
        amp = torch.rand(1, generator=gen).item() * 14.0 / harmonic * difficulty
        phase = torch.rand(1, generator=gen).item() * 2 * torch.pi
        radius = radius + amp * torch.cos(harmonic * theta + phase)
    points = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1) * scale
    return points.to(device) if device is not None else points


def resample_closed(points: np.ndarray, n_points: int, subdivisions: int = 32) -> np.ndarray:
    """Periodic Catmull-Rom spline through `points`, sampled evenly by arc length."""
    p = np.asarray(points, dtype=np.float64)
    m = len(p)
    t = np.linspace(0.0, 1.0, subdivisions, endpoint=False)[:, None]
    dense = []
    for i in range(m):
        p0, p1, p2, p3 = p[(i - 1) % m], p[i], p[(i + 1) % m], p[(i + 2) % m]
        dense.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    dense = np.concatenate(dense)
    step = np.diff(dense, axis=0, append=dense[:1])
    seg = np.hypot(step[:, 0], step[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    total = float(seg.sum())
    target = np.linspace(0.0, total, n_points, endpoint=False)
    return np.stack(
        [np.interp(target, s, dense[:, 0]), np.interp(target, s, dense[:, 1])], axis=1
    )


def smooth_closed(points: np.ndarray, sigma_m: float) -> np.ndarray:
    """Circular Gaussian smoothing of an evenly sampled closed curve.

    Survey centerlines are a few dozen metres apart, so a spline through them
    kinks at the apex of tight corners into radii no car can follow; a few
    metres of smoothing removes the kinks without moving the road.
    """
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    spacing = float(np.hypot(*np.diff(p, axis=0, append=p[:1]).T).mean())
    sigma = sigma_m / spacing
    half = int(np.ceil(3 * sigma))
    if half == 0:
        return p
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.empty_like(p)
    for axis in range(2):
        padded = np.concatenate([p[-half:, axis], p[:, axis], p[:half, axis]])
        out[:, axis] = np.convolve(padded, k, mode="valid")[:n]
    return out


def curvature_radius(points: np.ndarray, span_m: float = 6.0) -> np.ndarray:
    """Radius of curvature (metres) at every sample of a closed polyline.

    Fitted as the circle through the points `span_m` before and after each
    sample, which is robust to the sub-metre wobble left by resampling.
    """
    p = np.asarray(points, dtype=np.float64)
    spacing = float(np.hypot(*np.diff(p, axis=0, append=p[:1]).T).mean())
    k = max(1, int(round(span_m / max(spacing, 1e-6))))
    a, b, c = np.roll(p, k, axis=0), p, np.roll(p, -k, axis=0)
    ab, bc, ac = b - a, c - b, c - a
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0])
    lengths = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    return lengths / np.maximum(2 * cross, 1e-9)


def speed_profile(centerline: np.ndarray, cfg: CarConfig, grade: np.ndarray | None = None) -> np.ndarray:
    """Quasi-steady reference speed (m/s) at every centerline sample.

    The fastest a car with `cfg`'s grip circle, power, drag and brakes can pass
    each point of the *centerline*: cornering speed from the speed-dependent
    grip envelope, a backward pass for braking zones and a forward pass for
    traction-limited acceleration, each run twice round the loop so the lap
    closes. The racing line is wider than the centerline, so this is a little
    conservative; `lap_time_ref_s` from it is the model's own pole-lap estimate.
    """
    p = np.asarray(centerline, dtype=np.float64)
    n = len(p)
    seg = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
    radius = np.clip(curvature_radius(p), 1.0, 1e5)
    slope = np.zeros(n) if grade is None else np.asarray(grade, dtype=np.float64)
    a_slope = -G * slope / np.sqrt(1.0 + slope * slope)  # along the road, downhill positive
    m, a, vr, gmax = cfg.grip_mech_g, cfg.grip_aero_g, cfg.grip_v_ref, cfg.grip_max_g
    # v^2 = grip(v) g R with grip = m + a (v / vr)^2 has the closed form below while k R < 1.
    k = a * G / vr**2
    with np.errstate(divide="ignore", invalid="ignore"):
        corner = np.where(k * radius < 1.0, np.sqrt(m * G * radius / np.maximum(1.0 - k * radius, 1e-9)), np.inf)
    v = np.minimum(np.minimum(corner, np.sqrt(gmax * G * radius)), cfg.max_speed)

    def grip(speed: float) -> float:
        return min(m + a * (speed / vr) ** 2, gmax) * G

    def drag(speed: float) -> float:
        return 0.5 * 1.225 * cfg.cda_m2 * speed * speed / cfg.mass_kg + cfg.roll_decel

    for _ in range(2):  # braking: v[i] must be able to slow to v[i+1] over seg[i]
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            vj = v[j]
            g_total = grip(vj)
            a_lat = vj * vj / radius[j]
            a_brake = np.sqrt(max(g_total**2 - a_lat**2, 0.0)) + drag(vj) - a_slope[j]
            v[i] = min(v[i], np.sqrt(max(vj * vj + 2.0 * max(a_brake, 0.5) * seg[i], 0.0)))
    for _ in range(2):  # traction: v[i+1] cannot exceed what v[i] can accelerate to over seg[i]
        for i in range(n):
            j = (i + 1) % n
            vi = max(v[i], 1.0)
            g_total = grip(vi)
            traction = cfg.traction_g * G * (g_total / (cfg.grip_mech_g * G))
            a_power = min(cfg.power_w / (cfg.mass_kg * vi), traction)
            a_long_cap = np.sqrt(max(g_total**2 - (vi * vi / radius[i]) ** 2, 0.0))
            a_acc = min(a_power, a_long_cap) - drag(vi) + a_slope[i]
            v[j] = min(v[j], np.sqrt(max(vi * vi + 2.0 * max(a_acc, 0.0) * seg[i], 1.0)))
    return np.clip(v, 1.0, cfg.max_speed)


@dataclass(frozen=True)
class GeoProjection:
    """Equirectangular projection shared by the circuit and its scenery.

    x = (lon - lon0) * R * cos(lat0) * scale - dx,  y = (lat - lat0) * R * scale - dy
    with angles in radians. `dx, dy` recentre the circuit on the origin.
    """

    lon0: float
    lat0: float
    scale: float
    dx: float = 0.0
    dy: float = 0.0

    def project(self, lonlat: np.ndarray) -> np.ndarray:
        ll = np.asarray(lonlat, dtype=np.float64).reshape(-1, 2)
        lat0 = np.deg2rad(self.lat0)
        x = np.deg2rad(ll[:, 0] - self.lon0) * EARTH_RADIUS_M * np.cos(lat0) * self.scale - self.dx
        y = np.deg2rad(ll[:, 1] - self.lat0) * EARTH_RADIUS_M * self.scale - self.dy
        return np.stack([x, y], axis=1)

    def unproject(self, xy: np.ndarray) -> np.ndarray:
        """Inverse of `project`: track-frame metres back to (lon, lat) degrees."""
        p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        lat0 = np.deg2rad(self.lat0)
        lon = self.lon0 + np.rad2deg((p[:, 0] + self.dx) / (EARTH_RADIUS_M * np.cos(lat0) * self.scale))
        lat = self.lat0 + np.rad2deg((p[:, 1] + self.dy) / (EARTH_RADIUS_M * self.scale))
        return np.stack([lon, lat], axis=1)

    def as_dict(self) -> dict:
        return {"lon0": self.lon0, "lat0": self.lat0, "scale": self.scale, "dx": self.dx, "dy": self.dy}


def load_geojson_centerline(
    path: str | Path,
    scale: float = 1.0,
    n_points: int = 2048,
    smooth_m: float = 8.0,
    mirror: bool = False,
    return_projection: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, GeoProjection]:
    """Circuit centerline from a GeoJSON LineString of lon/lat, in metres.

    Projected equirectangularly about the circuit's own centroid (a few km
    across, so distortion is negligible), centred on the origin, and oriented
    clockwise, which is the direction every real circuit here is raced;
    `mirror` flips that for stress tests.
    """
    data = json.loads(Path(path).read_text())
    coords = data["features"][0]["geometry"]["coordinates"]
    lonlat = np.asarray(coords, dtype=np.float64)[:, :2]
    if np.allclose(lonlat[0], lonlat[-1]):
        lonlat = lonlat[:-1]
    proj = GeoProjection(lon0=float(lonlat[:, 0].mean()), lat0=float(lonlat[:, 1].mean()), scale=scale)
    pts = resample_closed(proj.project(lonlat), n_points * 4)
    pts = resample_closed(smooth_closed(pts, smooth_m), n_points, subdivisions=2)
    signed_area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if (signed_area > 0) != mirror:  # counter-clockwise -> reverse to clockwise
        pts = pts[::-1].copy()
    centre = pts.mean(axis=0)
    pts -= centre
    proj = replace(proj, dx=float(centre[0]), dy=float(centre[1]))
    out = torch.tensor(pts, dtype=torch.float32)
    return (out, proj) if return_projection else out


def build_centerline(cfg: CarConfig, seed: int = 0) -> torch.Tensor:
    if cfg.layout == "geojson":
        return load_geojson_centerline(
            cfg.geojson_path, cfg.track_scale, cfg.n_points, cfg.smooth_m, cfg.mirror
        )
    if cfg.layout == "loop":
        pts = make_centerline(seed, scale=cfg.loop_scale, difficulty=cfg.loop_difficulty)
        return pts.flip(0) if cfg.mirror else pts
    raise ValueError(f"unknown layout {cfg.layout!r}; expected 'loop' or 'geojson'")


def surveyed_heights(cfg: CarConfig, centerline: np.ndarray) -> np.ndarray | None:
    """Road heights for `cfg.geojson_path` from the survey beside it, or None without one."""
    from terrain import road_profile_for_circuit  # lazy: terrain imports this module

    pts, proj = load_geojson_centerline(cfg.geojson_path, cfg.track_scale, cfg.n_points, cfg.smooth_m, cfg.mirror, return_projection=True)
    if pts.shape[0] != centerline.shape[0] or not np.allclose(pts.numpy(), centerline, atol=1e-3):
        return None  # not the circuit this config describes
    return road_profile_for_circuit(cfg.geojson_path, centerline, proj)


class Track:
    """Rasterised drivable mask, progress and clearance fields for one centerline."""

    def __init__(self, centerline: torch.Tensor, cfg: CarConfig, device: torch.device):
        # The grid must contain the whole circuit plus lidar reach.
        needed = float(centerline.abs().max()) + cfg.track_halfwidth + 2.0
        self.cfg = cfg if needed <= cfg.grid_extent else replace(cfg, grid_extent=needed * 1.02)
        cfg = self.cfg
        self.device = device
        self.centerline = centerline.to(device)
        res, extent = cfg.grid_res, cfg.grid_extent
        axis = torch.linspace(-extent, extent, res, device=device)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        cells = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

        nearest_d = torch.full((cells.shape[0],), 1e9, device=device)
        nearest_i = torch.zeros(cells.shape[0], dtype=torch.long, device=device)
        chunk = 16384
        for start in range(0, cells.shape[0], chunk):
            block = cells[start : start + chunk]
            dist = torch.cdist(block, self.centerline)
            d, i = dist.min(dim=1)
            nearest_d[start : start + chunk] = d
            nearest_i[start : start + chunk] = i

        self.clearance = (cfg.track_halfwidth - nearest_d).reshape(res, res)
        self.drivable = self.clearance > 0
        self.progress = (nearest_i.float() / self.centerline.shape[0]).reshape(res, res)
        # Sub-sample progress: nearest sample per cell plus the unit tangent per
        # sample, so `progress_at` projects a point onto the local centerline
        # direction. Paying by nearest sample alone paid in 1.6 m steps (one
        # sample spacing on Monaco): most control steps paid nothing, then one
        # paid 0.16, which also read as "reward while stationary" to the monitor.
        self.nearest = nearest_i.to(torch.int32).reshape(res, res)
        tangent = self.centerline.roll(-1, dims=0) - self.centerline.roll(1, dims=0)
        self.tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.spacing_m = float((self.centerline.roll(-1, dims=0) - self.centerline).norm(dim=1).mean())
        self.cell = 2 * extent / (res - 1)
        self.min_radius_m = float(curvature_radius(self.centerline.cpu().numpy()).min())
        # Surveyed height and slope per sample (None on a flat world): the
        # profile the physics drives on is the one the viewer draws.
        self.height: torch.Tensor | None = None
        self.grade = torch.zeros(self.centerline.shape[0], device=device)
        if cfg.layout == "geojson" and cfg.road_grade:
            heights = surveyed_heights(cfg, self.centerline.cpu().numpy())
            if heights is not None:
                from terrain import grade_of  # lazy: terrain imports this module

                self.height = torch.tensor(heights, dtype=torch.float32, device=device)
                self.grade = torch.tensor(grade_of(heights, self.centerline.cpu().numpy()), dtype=torch.float32, device=device)
        self.climb_m = float(self.height.max() - self.height.min()) if self.height is not None else 0.0
        # Reference speed per sample (`speed_profile`) and the lap time it implies:
        # what the pace term measures the car against, and the model's own pole lap.
        profile = speed_profile(self.centerline.cpu().numpy(), cfg, self.grade.cpu().numpy())
        self.speed_ref = torch.tensor(profile, dtype=torch.float32, device=device)
        self.lap_time_ref_s = float(np.sum(np.hypot(*(np.roll(self.centerline.cpu().numpy(), -1, axis=0) - self.centerline.cpu().numpy()).T) / profile))

    @property
    def extent(self) -> float:
        return self.cfg.grid_extent

    @property
    def length_m(self) -> float:
        step = self.centerline.roll(-1, dims=0) - self.centerline
        return float(step.norm(dim=1).sum())

    def _cell_index(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res, extent = self.cfg.grid_res, self.cfg.grid_extent
        col = ((xy[..., 0] + extent) / self.cell).round().long().clamp(0, res - 1)
        row = ((xy[..., 1] + extent) / self.cell).round().long().clamp(0, res - 1)
        return row, col

    def clearance_at(self, xy: torch.Tensor, bilinear: bool = False) -> torch.Tensor:
        """Metres from the point to the nearest track edge; negative once off it.

        Nearest-cell lookup by default (lidar samples, thousands per car);
        `bilinear` interpolates the four surrounding cells for sub-cell
        accuracy where it matters, the car body. Points outside the grid read
        as far off the track.
        """
        inside = (xy.abs() < self.cfg.grid_extent).all(dim=-1)
        far = torch.full_like(xy[..., 0], -1e3)
        if not bilinear:
            row, col = self._cell_index(xy)
            return torch.where(inside, self.clearance[row, col], far)
        res, extent = self.cfg.grid_res, self.cfg.grid_extent
        fx = (xy[..., 0] + extent) / self.cell
        fy = (xy[..., 1] + extent) / self.cell
        x0 = fx.floor().long().clamp(0, res - 2)
        y0 = fy.floor().long().clamp(0, res - 2)
        tx = (fx - x0).clamp(0.0, 1.0)
        ty = (fy - y0).clamp(0.0, 1.0)
        c = self.clearance
        top = (1 - tx) * c[y0, x0] + tx * c[y0, x0 + 1]
        bottom = (1 - tx) * c[y0 + 1, x0] + tx * c[y0 + 1, x0 + 1]
        return torch.where(inside, (1 - ty) * top + ty * bottom, far)

    def is_drivable(self, xy: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """True where a body of half-width `margin` fits on the road."""
        return self.clearance_at(xy) > margin

    def progress_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Lap fraction in [0, 1): nearest centerline sample plus the signed offset along its tangent."""
        row, col = self._cell_index(xy)
        index = self.nearest[row, col].long()
        along = ((xy - self.centerline[index]) * self.tangent[index]).sum(dim=-1) / self.spacing_m
        n = self.centerline.shape[0]
        # The nearest sample is looked up per grid cell, so a point can sit up to a
        # cell's half-diagonal past the sample's midline; allow the offset to run
        # past the neighbouring samples and only bound it for points far off road.
        return torch.remainder(index.float() + along.clamp(-2.0, 2.0), n) / n


class CarEnv:
    """`batch` cars driving the same track, reset independently on termination.

    Pass a prebuilt `track` to share one rasterisation between environments
    that differ only in start point. `start_fraction` is either one number for
    every car or a sequence of `batch` numbers, the point along the lap (0..1)
    where each car is placed at reset.
    """

    def __init__(
        self,
        batch: int,
        device: torch.device,
        cfg: CarConfig | None = None,
        seed: int = 0,
        track: Track | None = None,
        start_fraction: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
    ) -> None:
        self.cfg = cfg or CarConfig()
        self.device = device
        self.batch = batch
        self.track = track or Track(build_centerline(self.cfg, seed), self.cfg, device)
        n = self.track.centerline.shape[0]
        fractions = torch.as_tensor(start_fraction, dtype=torch.float32).reshape(-1)
        if fractions.numel() == 1:
            fractions = fractions.repeat(batch)
        if fractions.numel() != batch:
            raise ValueError(f"start_fraction has {fractions.numel()} entries for {batch} cars")
        self.start_index = ((fractions * n).long() % n).to(device)
        self.progress_scale = self.cfg.progress_per_m * self.track.length_m
        fov = float(np.deg2rad(self.cfg.fov_deg))
        # Ray 0 looks left (+fov/2), the last ray right: the same left-to-right
        # order as the visual neuron groups the agent feeds them into.
        self.ray_angles = torch.linspace(fov / 2, -fov / 2, self.cfg.n_rays, device=device)
        frac = torch.arange(1, self.cfg.march_steps + 1, device=device).float() / self.cfg.march_steps
        self.march = frac.pow(self.cfg.march_power) * self.cfg.max_range
        self.stuck_steps = max(1, int(round(self.cfg.stuck_window_s / self.cfg.dt_s)))
        self.max_step_progress = 3.0 * self.cfg.max_speed * self.cfg.dt_s / self.track.length_m
        # Body outline in the car frame (x forward, y left): both long sides
        # sampled nose to tail, so collision sees the whole rectangle.
        along = torch.linspace(-0.5, 0.5, max(2, self.cfg.body_samples), device=device) * self.cfg.car_length
        self.body_offsets = torch.cat(
            [torch.stack([along, torch.full_like(along, side * self.cfg.car_halfwidth)], dim=1) for side in (-1.0, 1.0)]
        )
        # Set by `attach_sensor` when the car drives on a camera instead of
        # rays. Until then the eye is the ray march below.
        self.sensor = None
        # Set by `attach_law` when the circuit has a road-law survey. Without
        # one no legal charge is made: the driver cannot break a rule the
        # survey does not record.
        self.law = None
        self.reset()

    def attach_law(self, law) -> None:
        """Charge this population under the road law of the circuit.

        `law` is a `lawreward.LawEnforcer` built from the survey beside the
        track. Its charges are added to the step reward and appear in
        `last_terms` under their own names, so the exploit monitor and the
        telemetry can see what the driver is paying for.
        """
        self.law = law

    def attach_sensor(self, sensor) -> None:
        """Drive on a camera and a detector instead of the ray march.

        With a sensor attached the observation is whatever the detector made of
        the rendered frame; the track's geometry stops reaching the brain
        entirely. The physics, the endings and the law are unchanged: only what
        the driver can see is different.
        """
        self.sensor = sensor
        if sensor is not None:
            sensor.reset()

    @property
    def elapsed_s(self) -> torch.Tensor:
        """Seconds of driving each body has done this episode."""
        return self.step_count.to(torch.float32) * self.cfg.dt_s

    @property
    def start_fraction(self) -> torch.Tensor:
        """Start fractions for each car, derived from start_index."""
        n = self.track.centerline.shape[0]
        return self.start_index.float() / n

    @property
    def obs_dim(self) -> int:
        if getattr(self, "sensor", None) is not None:
            return self.sensor.width
        return self.cfg.n_rays + 1

    def _start_pose(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cl = self.track.centerline
        n = cl.shape[0]
        start = cl[index]
        nxt = cl[(index + 8) % n]
        heading = torch.atan2(nxt[:, 1] - start[:, 1], nxt[:, 0] - start[:, 0])
        return start, heading

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(self.batch, dtype=torch.bool, device=self.device)
            zeros = torch.zeros(self.batch, device=self.device)
            self.pos = torch.zeros(self.batch, 2, device=self.device)
            self.heading = zeros.clone()
            self.speed = zeros.clone()
            self.steer = zeros.clone()
            self.lat_g = zeros.clone()
            self.laps = zeros.clone()
            self.best_laps = zeros.clone()
            self.last_progress = zeros.clone()
            self.anchor_laps = zeros.clone()
            self.anchor_step = torch.zeros(self.batch, dtype=torch.long, device=self.device)
            self.step_count = torch.zeros(self.batch, dtype=torch.long, device=self.device)
            self.done_reason = torch.zeros(self.batch, dtype=torch.long, device=self.device)
            self.prev_pos = self.pos.clone()
            self.prev_heading = self.heading.clone()
            self.last_terms: dict[str, torch.Tensor] = {}
        # Functional (out-of-place) masked reset. The viewer resets single
        # cars between control steps that ran under `torch.inference_mode()`,
        # and PyTorch refuses in-place writes into inference tensors outside
        # that mode ("Inplace update to inference tensor outside InferenceMode
        # is not allowed"), so every field is rebuilt with `torch.where`.
        start_all, heading_all = self._start_pose(self.start_index)
        m1 = mask
        m2 = mask.unsqueeze(1)
        zero = torch.zeros_like(self.speed)
        zero_l = torch.zeros_like(self.step_count)
        self.pos = torch.where(m2, start_all, self.pos)
        self.prev_pos = torch.where(m2, start_all, self.prev_pos)
        self.heading = torch.where(m1, heading_all, self.heading)
        self.prev_heading = torch.where(m1, heading_all, self.prev_heading)
        self.speed = torch.where(m1, zero, self.speed)
        self.steer = torch.where(m1, zero, self.steer)
        self.lat_g = torch.where(m1, zero, self.lat_g)
        self.laps = torch.where(m1, zero, self.laps)
        self.best_laps = torch.where(m1, zero, self.best_laps)
        self.last_progress = torch.where(m1, self.track.progress_at(start_all), self.last_progress)
        self.anchor_laps = torch.where(m1, zero, self.anchor_laps)
        self.anchor_step = torch.where(m1, zero_l, self.anchor_step)
        self.step_count = torch.where(m1, zero_l, self.step_count)
        self.done_reason = torch.where(m1, torch.full_like(self.done_reason, DONE_ALIVE), self.done_reason)
        return self.observe()

    def compact(self, keep: torch.Tensor) -> None:
        """Drop the bodies where `keep` is False; the survivors keep their relative order.

        Used by the trainer once most of a population has finished: a
        crashed car costs the brain kernel as much as a driving one, so the
        batch shrinks to the cars that are still on the road. Every per-body
        tensor on the environment (pose, progress, counters, start index and
        the last reward terms) is index-selected; the track is shared.
        """
        keep = keep.to(self.device, torch.bool)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            raise ValueError("compact() needs at least one surviving body")
        old = self.batch
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == old and name not in ("ray_angles", "march", "body_offsets"):
                setattr(self, name, value[idx])
        self.last_terms = {
            k: v[idx] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == old else v
            for k, v in self.last_terms.items()
        }
        self.batch = int(idx.numel())

    def body_points(self, pos: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """World coordinates of the body outline samples, (batch, points, 2)."""
        c, s = heading.cos(), heading.sin()
        ox, oy = self.body_offsets[:, 0], self.body_offsets[:, 1]
        x = pos[:, 0:1] + ox * c.unsqueeze(1) - oy * s.unsqueeze(1)
        y = pos[:, 1:2] + ox * s.unsqueeze(1) + oy * c.unsqueeze(1)
        return torch.stack([x, y], dim=-1)

    def body_clearance(self, pos: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """Metres between the body outline and the road edge; <= 0 means the body touches or crosses it."""
        return self.track.clearance_at(self.body_points(pos, heading), bilinear=True).min(dim=1).values

    def observe(self) -> torch.Tensor:
        sensor = getattr(self, "sensor", None)
        if sensor is not None:
            return sensor.observe(
                self.pos,
                self.heading,
                self.speed,
                self.last_progress,
                float(self.step_count.max()) * self.cfg.dt_s,
                self.cfg.max_speed,
            )
        angles = self.heading.unsqueeze(1) + self.ray_angles.unsqueeze(0)
        direction = torch.stack([angles.cos(), angles.sin()], dim=-1)
        points = self.pos[:, None, None, :] + direction[:, :, None, :] * self.march[
            None, None, :, None
        ]
        blocked = ~self.track.is_drivable(points)
        hit_any = blocked.any(dim=-1)
        first_hit = torch.where(
            hit_any,
            blocked.float().argmax(dim=-1),
            torch.full_like(blocked[..., 0], self.cfg.march_steps - 1, dtype=torch.long),
        )
        # Bisect between the last sample still on the road and the first one off
        # it, so the reported distance is the road edge, not the coarse sample
        # that happened to land past it.
        near = torch.where(first_hit > 0, self.march[(first_hit - 1).clamp(min=0)], torch.zeros_like(self.march[first_hit]))
        far = self.march[first_hit]
        for _ in range(self.cfg.ray_refine_steps):
            mid = 0.5 * (near + far)
            free = self.track.is_drivable(self.pos[:, None, :] + direction * mid.unsqueeze(-1))
            near = torch.where(free, mid, near)
            far = torch.where(free, far, mid)
        dist = torch.where(hit_any, far, torch.full_like(far, self.cfg.max_range)) / self.cfg.max_range
        speed = (self.speed / self.cfg.max_speed).unsqueeze(1)
        return torch.cat([dist, speed], dim=1)

    def grip_g(self, speed: torch.Tensor) -> torch.Tensor:
        """Total grip available in g at `speed`: mechanical plus downforce."""
        cfg = self.cfg
        aero = cfg.grip_aero_g * (speed / cfg.grip_v_ref) ** 2
        return (cfg.grip_mech_g + aero).clamp(max=cfg.grip_max_g)

    def _drive(self, action: torch.Tensor) -> None:
        """Advance the vehicle model one control step in place."""
        cfg = self.cfg
        dt = cfg.dt_s
        steer_cmd = action[:, 0].clamp(-1, 1) * cfg.max_steer_rad
        pedal = action[:, 1].clamp(-1, 1)
        if cfg.steer_tau_s > 0:
            blend = 1.0 - float(np.exp(-dt / cfg.steer_tau_s))
            self.steer = self.steer + blend * (steer_cmd - self.steer)
        else:
            self.steer = steer_cmd

        v = self.speed
        grip = self.grip_g(v) * G
        # Lateral demand from the bicycle model, served first from the grip
        # circle; beyond it the front washes out (understeer) rather than the
        # car spinning, which keeps the model stable at a 16 ms step.
        yaw_free = v / cfg.wheelbase * torch.tan(self.steer)
        a_lat = (v * yaw_free).abs()
        scale = torch.where(a_lat > grip, grip / a_lat.clamp(min=1e-6), torch.ones_like(a_lat))
        yaw = yaw_free * scale
        a_lat = a_lat * scale
        self.lat_g = a_lat / G
        a_long_cap = (grip**2 - a_lat**2).clamp(min=0.0).sqrt()

        # Longitudinal: power-limited above the traction limit, drag and rolling
        # losses always, brakes limited by what the grip circle has left.
        # Traction grows with downforce like the rest of the grip envelope.
        traction = cfg.traction_g * G * (grip / (cfg.grip_mech_g * G))
        a_power = torch.minimum(cfg.power_w / (cfg.mass_kg * v.clamp(min=1.0)), traction)
        a_drive = pedal.clamp(min=0.0) * torch.minimum(a_power, a_long_cap)
        a_brake = (-pedal).clamp(min=0.0) * a_long_cap
        a_drag = 0.5 * 1.225 * cfg.cda_m2 * v * v / cfg.mass_kg + cfg.roll_decel
        # Gravity along the road: uphill costs, downhill pays (slope from the survey; zero on a flat world).
        n = self.track.grade.shape[0]
        slope = self.track.grade[(self.last_progress * n).long().clamp(0, n - 1)]
        a_slope = -G * slope / torch.sqrt(1.0 + slope * slope)
        self.speed = (v + (a_drive - a_brake - a_drag + a_slope) * dt).clamp(0.0, cfg.max_speed)

        self.heading = self.heading + yaw * dt
        step_vec = torch.stack([self.heading.cos(), self.heading.sin()], dim=1) * (
            self.speed * dt
        ).unsqueeze(1)
        self.pos = self.pos + step_vec

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """action: (batch, 2) = steer in [-1,1], pedal in [-1,1]. Returns (obs, reward, done)."""
        cfg = self.cfg
        # Cars that already ended are frozen: pose, speed and lap count stay at
        # the ending step (a crashed car does not slide on; its lap count read
        # at any later poll equals the one read at death), reward is 0 and done
        # stays set. `reset(mask)` is the only way back.
        was_done = self.done_reason != DONE_ALIVE
        frozen = {
            name: getattr(self, name)
            for name in (
                "pos", "heading", "speed", "steer", "lat_g", "laps", "best_laps", "last_progress",
                "anchor_laps", "anchor_step", "step_count", "done_reason", "prev_pos", "prev_heading",
            )
        }
        self.prev_pos = self.pos
        self.prev_heading = self.heading
        self._drive(action)
        self.step_count = self.step_count + 1

        progress = self.track.progress_at(self.pos)
        delta = progress - self.last_progress
        delta = torch.where(delta < -0.5, delta + 1.0, delta)  # lap wrap
        delta = torch.where(delta > 0.5, delta - 1.0, delta)  # backwards wrap
        # A car cannot advance more than one step's travel along the lap. A
        # larger jump means its nearest centerline point snapped to another
        # part of the circuit (typically the step it leaves the road where two
        # sections run close together); count it as no progress rather than
        # pay or charge for a teleport.
        raw_delta = delta
        delta = torch.where(delta.abs() > self.max_step_progress, torch.zeros_like(delta), delta)
        self.last_progress = progress
        self.laps = self.laps + delta
        # Lap bonus only for laps never completed before by this car: crossing
        # the line forwards, reversing over it and crossing again pays once.
        new_laps = (torch.floor(self.laps) - torch.floor(self.best_laps)).clamp(min=0.0)
        self.best_laps = torch.maximum(self.best_laps, self.laps)

        clearance = self.body_clearance(self.pos, self.heading)
        crashed = clearance <= 0.0
        reversed_ = self.laps <= cfg.reverse_limit_laps
        window_over = (self.step_count - self.anchor_step) >= self.stuck_steps
        moved_m = (self.laps - self.anchor_laps) * self.track.length_m
        stuck = window_over & (moved_m < cfg.stuck_min_m)
        self.anchor_laps = torch.where(window_over, self.laps, self.anchor_laps)
        self.anchor_step = torch.where(window_over, self.step_count, self.anchor_step)

        finished = (self.laps >= cfg.max_laps) if cfg.max_laps > 0 else torch.zeros_like(stuck)

        reason = torch.zeros_like(self.done_reason)
        reason = torch.where(finished, torch.full_like(reason, DONE_FINISH), reason)
        reason = torch.where(stuck, torch.full_like(reason, DONE_STUCK), reason)
        reason = torch.where(reversed_, torch.full_like(reason, DONE_REVERSE), reason)
        reason = torch.where(crashed, torch.full_like(reason, DONE_CRASH), reason)
        self.done_reason = reason
        alive = reason == DONE_ALIVE

        near_wall = (1.0 - clearance / cfg.wall_margin).clamp(0.0, 1.0)
        progress_term = delta * self.progress_scale
        bonus_term = new_laps * cfg.lap_bonus
        wall_term = -cfg.wall_penalty * near_wall * near_wall
        speed_norm = self.speed / cfg.max_speed
        speed_excess = torch.relu(speed_norm - cfg.speed_free_fraction)
        speed_term = -cfg.speed_penalty * speed_excess * speed_excess
        low_speed = torch.relu(1.0 - self.speed / cfg.stall_speed_mps)
        steer_excess = torch.relu(self.steer.abs() / cfg.max_steer_rad - cfg.stall_steer_start)
        stall_term = -cfg.stall_steer_penalty * low_speed * steer_excess * steer_excess
        n_ref = self.track.speed_ref.shape[0]
        speed_ref = self.track.speed_ref[(self.last_progress * n_ref).long().clamp(0, n_ref - 1)]
        pace_ratio = (self.speed / (cfg.pace_margin * speed_ref).clamp(min=1.0)).clamp(0.0, cfg.pace_cap)
        pace_deficit = torch.relu(1.0 - self.speed / (cfg.pace_margin * speed_ref).clamp(min=1.0))
        pace_term = cfg.pace_bonus * pace_ratio * pace_ratio - cfg.pace_penalty * pace_deficit * pace_deficit
        ref_index = (self.last_progress * n_ref).long().clamp(0, n_ref - 1)
        lookahead_m = torch.maximum(torch.full_like(self.speed, cfg.align_lookahead_m), self.speed * cfg.align_lookahead_s)
        ahead = (lookahead_m / self.track.spacing_m).clamp(min=1.0).long()
        to_target = self.track.centerline[(ref_index + ahead) % n_ref] - self.pos
        align_err = torch.atan2(to_target[:, 1], to_target[:, 0]) - self.heading
        align_err = torch.atan2(align_err.sin(), align_err.cos())
        # Dead zone first: the yaw a racing line needs against the centerline
        # is free, everything past it is graded as before.
        align_excess = torch.relu(align_err.abs() - cfg.align_free_rad)
        align_term = -cfg.align_penalty * (1.0 - align_excess.cos())
        reward = progress_term + bonus_term + wall_term + speed_term + stall_term + pace_term + align_term - cfg.time_tax
        # The road law, charged from the survey's own numbers. A circuit with
        # no survey attached contributes nothing here rather than a guess.
        law_terms: dict[str, torch.Tensor] = {}
        if self.law is not None:
            law_terms = self.law.charge(
                progress=progress,
                # `last_progress` has already advanced to `progress` by here;
                # the step's start is one clamped delta behind it.
                prev_progress=torch.remainder(progress - delta, 1.0),
                speed=self.speed,
                pos=self.pos,
                elapsed_s=self.step_count.to(reward.dtype) * cfg.dt_s,
            )
            reward = reward + law_terms["law_total"]
        # Keep the terminal event a fixed cost. Making the crash penalty depend
        # on impact speed adds a large, orthogonal gradient whose easiest local
        # solution is to slow down rather than learn the steering response.
        # That is exactly the behavioral-collapse mode seen around generations
        # 538-549: speed fell from ~25 km/h to ~17 km/h while steering drifted
        # upward and progress collapsed. Speed is already shaped continuously
        # by `speed_penalty`; the terminal event should not double-count it.
        # Finishing is not a crash: the last step is paid normally.
        # Under a step budget an ended car is charged as if it stood still for
        # the unused steps (time tax plus full pace penalty each), so no early
        # ending scores above driving on; finishing the lap early is the only
        # way to save any of the budget.
        if cfg.episode_steps > 0:
            remaining = (cfg.episode_steps - self.step_count).clamp(min=0).to(reward.dtype)
        else:
            remaining = torch.zeros_like(reward)
        # Metres of the lap left undriven at an ending that is not the finish.
        # `laps` is the net lap fraction, so a car that ends at 0.58 of the lap
        # forfeits 0.42 of it. Clamped at 0 so a finisher forfeits nothing.
        unfinished_m = (cfg.max_laps - self.laps).clamp(min=0.0) * self.track.length_m if cfg.max_laps > 0 else torch.zeros_like(self.laps)
        crash_cost = (
            cfg.crash_penalty
            + cfg.unfinished_per_m * unfinished_m
            # A standing car earns no pace bonus, so the bonus counts as a
            # charge here: an ended body must never be cheaper per step than
            # the worst survivor.
            + (cfg.time_tax + cfg.pace_bonus + cfg.pace_penalty + cfg.align_penalty) * remaining
        )
        reward = torch.where(alive | finished, reward, -crash_cost)
        # Running the budget out without finishing owes the same unfinished
        # metres as ending early does, minus the crash penalty. Without this a
        # car that survives to the last step pays nothing for the lap it never
        # completed while one that crashes at 0.9 laps pays for the last tenth,
        # and the safest way to score is to potter around inside the budget.
        expiry_term = torch.zeros_like(reward)
        if cfg.episode_steps > 0:
            expired = alive & ~finished & (self.step_count >= cfg.episode_steps)
            expiry_term = torch.where(expired, -cfg.unfinished_per_m * unfinished_m, expiry_term)
        reward = reward + expiry_term
        # A crashed car stays at its last legal pose: the crash frame shows the
        # body against the barrier, not one step's travel through it.
        self.pos = torch.where(crashed.unsqueeze(1), self.prev_pos, self.pos)
        self.heading = torch.where(crashed, self.prev_heading, self.heading)
        # Per-term breakdown for the exploit monitor, telemetry and debugging.
        for name, old in frozen.items():
            new = getattr(self, name)
            setattr(self, name, torch.where(was_done.unsqueeze(1) if new.dim() == 2 else was_done, old, new))
        reward = torch.where(was_done, torch.zeros_like(reward), reward)
        delta = torch.where(was_done, torch.zeros_like(delta), delta)
        raw_delta = torch.where(was_done, torch.zeros_like(raw_delta), raw_delta)
        alive = alive & ~was_done
        self.last_terms = {
            "progress": progress_term,
            "bonus": bonus_term,
            "wall": wall_term,
            "stall": stall_term,
            "pace": pace_term,
            "speed_ref": speed_ref,
            "align": align_term,
            "align_err": align_err,
            "time_tax": torch.full_like(reward, -cfg.time_tax),
            "expiry": expiry_term,
            "crash_cost": crash_cost,
            "delta": delta,
            "raw_delta": raw_delta,
            "clearance": clearance,
            "alive": alive,
        }
        # Charges are zeroed for a body that had already ended, the way every
        # other term is. The two diagnostics (`law_limit_mps`, `law_offset_m`)
        # are measurements rather than charges and are reported as they were.
        for name, value in law_terms.items():
            diagnostic = name in ("law_limit_mps", "law_offset_m")
            self.last_terms[name] = value if diagnostic else torch.where(was_done, torch.zeros_like(value), value)
        return self.observe(), reward, ~alive

    def telemetry(self) -> dict[str, float]:
        """Population summary for logs: speed, steering, lateral load, endings."""
        counts = torch.bincount(self.done_reason, minlength=5).tolist()
        return {
            "speed_mean": float(self.speed.mean()),
            "speed_max": float(self.speed.max()),
            "steer_abs_mean": float(self.steer.abs().mean()) / self.cfg.max_steer_rad,
            "lat_g_max": float(self.lat_g.max()),
            "lat_g_mean": float(self.lat_g.mean()),
            "laps_min": float(self.laps.min()),
            "crash": counts[DONE_CRASH],
            "reverse": counts[DONE_REVERSE],
            "stuck": counts[DONE_STUCK],
            "finished": counts[DONE_FINISH],
            "alive": counts[DONE_ALIVE],
        }
