"""Every constant the live viewer and its web page use, in one place.

The page receives this whole block from `/api/config` (and inside `/api/meta`)
and derives all labels, colours, thresholds and scenery dimensions from it, so
nothing about the simulation, the vehicle or the track is written into the
JavaScript. Override any field with `viewer.py --config my.json` (a JSON object
with the same nesting; unknown keys are an error, so typos cannot pass
silently).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RasterConfig:
    # Neurons shown in the spike raster, per role. A stratified sample keeps
    # the payload small while still showing each stage of the sensorimotor
    # path as its own band.
    quota: dict[str, int] = field(
        default_factory=lambda: {
            "photoreceptor": 60,
            "visual_projection": 140,
            "mechanosensory": 60,
            "ascending": 80,
            "descending": 160,
            "motor": 100,
        }
    )
    seed: int = 0
    column_px: int = 2


@dataclass(frozen=True)
class SceneryLoadConfig:
    # OSM tunnel ways count as the circuit's tunnel when this close to the centerline...
    tunnel_reach_m: float = 14.0
    # ...at least this fraction of their vertices, and at least this long.
    tunnel_aligned_fraction: float = 0.85
    tunnel_min_length_m: float = 100.0
    # centerline samples inside one tunnel may be interrupted by up to this many samples
    tunnel_gap_samples: int = 40
    # spans shorter than this are car-park ramps, not the circuit's tunnel
    tunnel_min_span_m: float = 60.0
    road_classes: tuple[str, ...] = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential")
    # buildings whose footprint comes within this of the open road are survey mismatches
    building_road_clearance_m: float = 1.0
    ground_margin_factor: float = 1.6
    # sea/harbour rasterisation
    water_fine_m: float = 4.0
    water_coarse_m: float = 40.0
    water_pad_m: float = 120.0
    water_level_m: float = -1.2  # flat world only; with a terrain grid the sea is at 0 m (heights are above sea level)
    water_road_keep_m: float = 2.0
    # --- terrain (`<track>_dem.json` from fetch_terrain.py; absent = flat world) ---
    terrain_cell_m: float = 8.0  # ground heightfield spacing in the track frame
    road_smooth_m: float = 30.0  # smooth the surveyed surface along the road over about one survey cell
    land_min_m: float = 1.2  # quay top: a survey cell straddling the waterline reads 0, the land on it is not
    seabed_drop_m: float = 2.0  # water cells sit this far under the sea so the water plane covers them
    carve_shoulder_m: float = 2.5  # ground is flat at road height this far beyond the barrier...
    carve_blend_m: float = 12.0  # ...then blends back to the survey over this distance
    portal_probe_m: float = 40.0  # a tunnel portal's road height is read this far outside the mouth, where the survey sees road
    tunnel_cover_min_m: float = 7.4  # ground over a tunnel is at least the arch (6.4 m) plus its shell above the road


@dataclass(frozen=True)
class SceneryStyleConfig:
    # dimensions the page uses to build the 3D scenery
    storey_m: float = 3.2
    facade_bay_m: float = 3.1
    default_storeys_min: int = 2
    default_storeys_max: int = 14
    default_storeys_base: float = 3.0
    default_storeys_per_sqrt_m2: float = 0.22
    default_storeys_jitter: float = 4.0
    rail_heights_m: tuple[float, ...] = (0.32, 0.7, 1.08)  # three W-beams, Monaco style
    rail_offset_m: float = 0.03  # barrier face just outside the road edge (the physics wall)
    post_spacing_m: float = 2.0
    post_size_m: tuple[float, float, float] = (0.12, 1.45, 0.18)
    post_height_m: float = 0.72
    post_setback_m: float = 0.1
    facade_palette: tuple[tuple[str, str], ...] = (
        ("#e6d7bf", "#c9b799"),  # Monaco cream
        ("#e9c9a4", "#c7a37d"),  # ochre
        ("#f0e3d4", "#cbbca9"),  # pale stone
        ("#d9c5b0", "#b39a82"),  # sand
        ("#e3b8a4", "#c1907a"),  # terracotta pink
        ("#cfd5d9", "#a3adb5"),  # modern grey
        ("#f2e9dc", "#d3c6b2"),  # white-cream
        ("#dcc7a1", "#b8a07a"),  # yellow ochre
    )
    tunnel_wall_offset_m: float = 0.9  # barrier, narrow pavement, then the wall
    tunnel_wall_height_m: float = 4.4
    tunnel_apex_m: float = 6.4
    tunnel_arc_segments: int = 12
    tunnel_lamp_spacing_m: float = 8.0
    water_tile_m: float = 18.0
    land_tile_m: float = 6.0
    quay_drop_m: float = 0.8


@dataclass(frozen=True)
class DriveStyleConfig:
    road_texture_m: float = 6.0  # one asphalt tile covers this many metres
    kerb_width_m: float = 1.2
    line_width_m: float = 0.25
    line_inset_m: float = 0.5
    ground_extent_factor: float = 3.2
    ground_repeat_urban: tuple[float, float] = (420.0, 420.0)
    ground_repeat_rural: tuple[float, float] = (140.0, 140.0)
    camera_fov_deg: float = 55.0
    camera_back_m: float = 12.0
    camera_back_per_mps: float = 0.12
    camera_back_max_extra_m: float = 10.0
    camera_height_m: float = 4.2
    camera_height_per_mps: float = 0.03
    camera_height_max_extra_m: float = 3.0
    camera_ahead_m: float = 10.0
    camera_ahead_per_mps: float = 0.6
    camera_ahead_max_extra_m: float = 40.0
    camera_ease: float = 6.0
    lidar_height_m: float = 0.75
    # onboard camera: on the roll hoop behind the driver, looking down the road
    # over the fly at the wheel (the car stays in view)
    cockpit_height_m: float = 1.45
    cockpit_back_m: float = 2.1
    cockpit_ahead_m: float = 22.0
    cockpit_fov_deg: float = 72.0
    cockpit_roll_per_g: float = 0.05
    # the driver: a fly of this body length seated in the car at (seat_x back
    # from the centre, seat_y up), facing the nose
    fly_length_m: float = 1.1
    fly_seat_x_m: float = -0.35
    fly_seat_y_m: float = 0.72
    fly_body_colour: str = "#2b2a30"
    fly_eye_colour: str = "#8c1b1b"
    fly_wing_colour: str = "#d7e6ff"
    fog_density: float = 0.0014
    sun_intensity: float = 3.2
    sun_offset_m: tuple[float, float, float] = (60.0, 90.0, 30.0)
    exposure: float = 0.85
    car_model_url: str = "/assets/w11.glb"  # drop a W11 glTF here and it replaces the stand-in
    car_model_forward: str = "-z"
    stand_in_url: str = "/assets/ferrari.glb"
    stand_in_shadow_url: str = "/assets/ferrari_ao.png"
    stand_in_body_colour: str = "#1fb07a"


@dataclass(frozen=True)
class BrainViewConfig:
    activation_decay: float = 0.82
    yaw: float = 0.6
    pitch: float = 0.15
    distance: float = 1.75
    distance_min: float = 1.2
    distance_max: float = 8.0
    auto_spin_rad_per_frame: float = 0.0016
    drag_rad_per_px: float = 0.008
    zoom_per_wheel_unit: float = 0.002
    point_scale: float = 1.6
    point_scale_min: float = 1.1
    point_scale_ref_px: int = 620
    inferred_dim: float = 0.45


@dataclass(frozen=True)
class UiConfig:
    # colour per neuron role; roles missing here get `role_fallback_colour`
    role_colours: dict[str, str] = field(
        default_factory=lambda: {
            "photoreceptor": "#4aa3e8",
            "visual_projection": "#3ddc97",
            "mechanosensory": "#b58ce0",
            "ascending": "#e8c44a",
            "descending": "#e8894a",
            "motor": "#e05c6e",
        }
    )
    role_fallback_colour: str = "#8494a4"
    negative_weight_colour: str = "#5c9ee0"
    pedal_deadband: float = 0.05
    steer_deadband: float = 0.02
    trail_points: int = 1200
    end_banner_ms: int = 1400
    curve_refresh_ms: int = 15000
    training_refresh_ms: int = 5000
    rate_bar_floor_hz: float = 12.0
    kmh_per_mps: float = 3.6
    # default series on the training chart; the selector offers every numeric log field
    curve_default_keys: tuple[str, ...] = ("fitness_mean", "fitness_best", "eval_fitness")
    curve_secondary_keys: tuple[str, ...] = ("laps_mean", "laps_best", "eval_laps")


@dataclass(frozen=True)
class ViewerConfig:
    raster: RasterConfig = field(default_factory=RasterConfig)
    scenery_load: SceneryLoadConfig = field(default_factory=SceneryLoadConfig)
    scenery_style: SceneryStyleConfig = field(default_factory=SceneryStyleConfig)
    drive_style: DriveStyleConfig = field(default_factory=DriveStyleConfig)
    brain_view: BrainViewConfig = field(default_factory=BrainViewConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    track_stride: int = 2  # centerline decimation for the page
    top_readout: int = 16  # output neurons listed by steering influence
    curve_points: int = 400
    episode_log_keep: int = 200  # recent episode endings kept for /api/training
    # trainer counts as running while its log is younger than this many
    # times its recent generation time
    trainer_alive_factor: float = 3.0

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: Any, override: dict[str, Any], path: str = "") -> Any:
    """Return `base` (a frozen dataclass) with `override` applied, nested."""
    known = {f.name: f for f in fields(base)}
    kwargs: dict[str, Any] = {}
    for key, value in override.items():
        if key not in known:
            raise KeyError(f"unknown viewer config key {path + key!r}")
        current = getattr(base, key)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[key] = _merge(current, value, path + key + ".")
        elif isinstance(current, tuple) and isinstance(value, list):
            kwargs[key] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        else:
            kwargs[key] = value
    return base.__class__(**{**{k: getattr(base, k) for k in known}, **kwargs})


def load_viewer_config(path: str | Path | None) -> ViewerConfig:
    base = ViewerConfig()
    if path is None:
        return base
    override = json.loads(Path(path).read_text())
    if not isinstance(override, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return _merge(base, override)
