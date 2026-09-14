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

DONE_ALIVE, DONE_CRASH, DONE_REVERSE, DONE_STUCK = 0, 1, 2, 3
DONE_NAMES = {DONE_ALIVE: "alive", DONE_CRASH: "crash", DONE_REVERSE: "reverse", DONE_STUCK: "stuck"}


@dataclass(frozen=True)
class CarConfig:
    # --- sensing -----------------------------------------------------------------
    n_rays: int = 9
    fov_deg: float = 180.0
    max_range: float = 150.0
    march_steps: int = 96
    # Ray samples are spaced as (i/N)^march_power * max_range: fine near the
    # car, coarse far away, so 96 samples cover 150 m with 0.3 m near the body.
    march_power: float = 1.5
    dt_s: float = 0.016
    # --- vehicle (Mercedes-AMG F1 W11, 2020; public figures) ---------------------
    wheelbase: float = 3.70
    car_halfwidth: float = 1.00  # 2.0 m body
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
    # --- termination -------------------------------------------------------------
    # Net progress below this ends the episode as a crash. A car that turns
    # round and drives the loop backwards otherwise survives, and evolution
    # finds that before it finds cornering.
    reverse_limit_laps: float = -0.01
    # Stuck: less than `stuck_min_m` of net progress over `stuck_window_s`.
    stuck_window_s: float = 4.0
    stuck_min_m: float = 5.0
    # --- reward ------------------------------------------------------------------
    # Progress along the track is the only thing paid for, at `progress_per_m`
    # per metre, so different circuits pay the same for the same driving; each
    # *newly* completed lap adds `lap_bonus` (crossing the line back and forth
    # pays once). No speed bonus: it paid for speed in any direction. The wall
    # term gives a smooth ramp inside `wall_margin` metres of the body's edge
    # so the search sees a gradient before the cliff of a crash.
    progress_per_m: float = 0.10
    lap_bonus: float = 100.0
    time_tax: float = 0.02
    wall_margin: float = 1.5
    wall_penalty: float = 0.03
    crash_penalty: float = 20.0

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
        self.cell = 2 * extent / (res - 1)
        self.min_radius_m = float(curvature_radius(self.centerline.cpu().numpy()).min())

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

    def clearance_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Metres from the point to the nearest track edge; negative once off it.

        Points outside the grid read as far off the track.
        """
        row, col = self._cell_index(xy)
        inside = (xy.abs() < self.cfg.grid_extent).all(dim=-1)
        return torch.where(inside, self.clearance[row, col], torch.full_like(xy[..., 0], -1e3))

    def is_drivable(self, xy: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """True where a body of half-width `margin` fits on the road."""
        return self.clearance_at(xy) > margin

    def progress_at(self, xy: torch.Tensor) -> torch.Tensor:
        row, col = self._cell_index(xy)
        return self.progress[row, col]


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
        self.reset()

    @property
    def obs_dim(self) -> int:
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
        start, heading = self._start_pose(self.start_index[mask])
        self.pos[mask] = start
        self.prev_pos[mask] = start
        self.heading[mask] = heading
        self.prev_heading[mask] = heading
        self.speed[mask] = 0.0
        self.steer[mask] = 0.0
        self.lat_g[mask] = 0.0
        self.laps[mask] = 0.0
        self.best_laps[mask] = 0.0
        self.last_progress[mask] = self.track.progress_at(start)
        self.anchor_laps[mask] = 0.0
        self.anchor_step[mask] = 0
        self.step_count[mask] = 0
        self.done_reason[mask] = DONE_ALIVE
        return self.observe()

    def observe(self) -> torch.Tensor:
        angles = self.heading.unsqueeze(1) + self.ray_angles.unsqueeze(0)
        direction = torch.stack([angles.cos(), angles.sin()], dim=-1)
        points = self.pos[:, None, None, :] + direction[:, :, None, :] * self.march[
            None, None, :, None
        ]
        blocked = ~self.track.is_drivable(points)
        first_hit = torch.where(
            blocked.any(dim=-1),
            blocked.float().argmax(dim=-1),
            torch.full_like(blocked[..., 0], self.cfg.march_steps - 1, dtype=torch.long),
        )
        dist = self.march[first_hit] / self.cfg.max_range
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
        self.speed = (v + (a_drive - a_brake - a_drag) * dt).clamp(0.0, cfg.max_speed)

        self.heading = self.heading + yaw * dt
        step_vec = torch.stack([self.heading.cos(), self.heading.sin()], dim=1) * (
            self.speed * dt
        ).unsqueeze(1)
        self.pos = self.pos + step_vec

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """action: (batch, 2) = steer in [-1,1], pedal in [-1,1]. Returns (obs, reward, done)."""
        cfg = self.cfg
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

        clearance = self.track.clearance_at(self.pos) - cfg.car_halfwidth
        crashed = clearance <= 0.0
        reversed_ = self.laps <= cfg.reverse_limit_laps
        window_over = (self.step_count - self.anchor_step) >= self.stuck_steps
        moved_m = (self.laps - self.anchor_laps) * self.track.length_m
        stuck = window_over & (moved_m < cfg.stuck_min_m)
        self.anchor_laps = torch.where(window_over, self.laps, self.anchor_laps)
        self.anchor_step = torch.where(window_over, self.step_count, self.anchor_step)

        reason = torch.zeros_like(self.done_reason)
        reason = torch.where(stuck, torch.full_like(reason, DONE_STUCK), reason)
        reason = torch.where(reversed_, torch.full_like(reason, DONE_REVERSE), reason)
        reason = torch.where(crashed, torch.full_like(reason, DONE_CRASH), reason)
        self.done_reason = reason
        alive = reason == DONE_ALIVE

        near_wall = (1.0 - clearance / cfg.wall_margin).clamp(0.0, 1.0)
        progress_term = delta * self.progress_scale
        bonus_term = new_laps * cfg.lap_bonus
        wall_term = -cfg.wall_penalty * near_wall * near_wall
        reward = progress_term + bonus_term + wall_term - cfg.time_tax
        reward = torch.where(alive, reward, torch.full_like(reward, -cfg.crash_penalty))
        # Per-term breakdown for the exploit monitor and debugging.
        self.last_terms = {
            "progress": progress_term,
            "bonus": bonus_term,
            "wall": wall_term,
            "delta": delta,
            "raw_delta": raw_delta,
            "clearance": clearance,
            "alive": alive,
        }
        return self.observe(), reward, ~alive

    def telemetry(self) -> dict[str, float]:
        """Population summary for logs: speed, steering, lateral load, endings."""
        counts = torch.bincount(self.done_reason, minlength=4).tolist()
        return {
            "speed_mean": float(self.speed.mean()),
            "speed_max": float(self.speed.max()),
            "steer_abs_mean": float(self.steer.abs().mean()) / self.cfg.max_steer_rad,
            "lat_g_max": float(self.lat_g.max()),
            "crash": counts[DONE_CRASH],
            "reverse": counts[DONE_REVERSE],
            "stuck": counts[DONE_STUCK],
        }
