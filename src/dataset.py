"""Build a detection dataset from the driving sim, so the detector can learn this road.

A COCO-trained detector sees nothing in a frame from `camera.py`: the cars it
knows are photographs of cars, not shaded geometry. The fix is not to hand the
driver the answer - it is to train the detector on the road it will actually
work on, exactly as a real fleet trains one on its own footage.

  python3 src/dataset.py --track data/tracks/kl.geojson --frames 4000 \
      --out data/perception/kl

Labels come from the renderer's index pass, which paints each actor in a colour
encoding its identity and reads the frame buffer back. A box is therefore the
object's true extent in that frame, and an object hidden behind another gets no
box at all. These labels train the detector and nothing else: the driver never
sees them, only what the detector makes of the pixels.

Data (c) OpenStreetMap contributors, ODbL 1.0 for the road the scene is built on.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camera import CameraConfig, DriverCamera
from law import load_law
from lawreward import SignalTiming
from roadlaw import control_points_for_circuit, legal_profile_for_circuit
from scene import (
    SCENE_CLASSES,
    SceneConfig,
    Traffic,
    build_road,
    frame_at,
    lane_halfwidth,
    place_signals,
    place_signs,
    update_signals,
)

CLASS_INDEX = {name: i for i, name in enumerate(SCENE_CLASSES)}


@dataclass(frozen=True)
class CaptureConfig:
    """How the capture wanders, so the detector sees the road from everywhere.

    A detector trained on one racing line learns that line. These ranges spread
    the camera across the carriageway and the traffic across the lap.
    """

    lateral_fraction: float = 0.7  # of the half-width, either side of the centre
    heading_jitter_rad: float = 0.25
    # Roughly a busy urban lane: about 35 cars and 28 motorcycles per kilometre
    # of a 5 km lap. Malaysian traffic carries far more two-wheelers than a
    # European street, which is why the motorcycle count is nearly the car one.
    vehicles: int = 190
    motorcycles: int = 150
    pedestrians: int = 60
    traffic_step_s: float = 0.35
    val_fraction: float = 0.15
    # Most of the lap is empty road. A detector learns from what it can see, so
    # most frames are taken from behind something rather than at random.
    follow_fraction: float = 0.75
    follow_gap_m: tuple[float, float] = (9.0, 70.0)
    # A frame where one object fills the view means the camera spawned inside
    # it. That is not a view a driver ever gets, so it is discarded.
    max_box_coverage: float = 0.55


@dataclass
class SceneBundle:
    """Everything one circuit needs to be rendered."""

    centerline: np.ndarray
    heights: np.ndarray
    profile: object
    control_points: tuple
    road: object
    fixtures: list
    halfwidth: np.ndarray


def load_scene(track: Path, scene_cfg: SceneConfig | None = None) -> SceneBundle:
    """The circuit, its law and its roadside fixtures, ready to render."""
    from car_env import load_geojson_centerline

    scene_cfg = scene_cfg or SceneConfig()
    points, proj = load_geojson_centerline(track, return_projection=True)
    centerline = points.numpy().astype(np.float64)
    profile = legal_profile_for_circuit(track, centerline, proj)
    if profile is None:
        raise SystemExit(f"{track} has no road survey beside it; run fetch_roads.py or fetch_malaysia_osm.py")
    control_points = control_points_for_circuit(track, proj, load_law(), centerline=centerline)
    heights = surveyed_heights_or_flat(track, centerline)
    road = build_road(centerline, heights, profile, scene_cfg)
    fixtures = place_signals(control_points, centerline, heights, profile, profile.driving_side, scene_cfg)
    fixtures += place_signs(control_points, centerline, heights, profile, profile.driving_side, scene_cfg)
    return SceneBundle(
        centerline=centerline,
        heights=heights,
        profile=profile,
        control_points=control_points,
        road=road,
        fixtures=fixtures,
        halfwidth=lane_halfwidth(profile, scene_cfg),
    )


def surveyed_heights_or_flat(track: Path, centerline: np.ndarray) -> np.ndarray:
    """The road's surveyed height profile where one exists, otherwise a flat road."""
    try:
        from terrain import road_profile_for_circuit
        from car_env import load_geojson_centerline

        _, proj = load_geojson_centerline(track, return_projection=True)
        heights = road_profile_for_circuit(track, centerline, proj)
    except Exception:
        heights = None
    return np.zeros(centerline.shape[0]) if heights is None else np.asarray(heights, dtype=np.float64)


def _camera_sample(rng, bundle: SceneBundle, traffic: Traffic, cfg: CaptureConfig, n: int) -> int:
    """Where along the lap to stand the camera for this frame.

    Mostly a set distance behind something that is on the road, so the frame
    has an object in it; the rest of the time anywhere, so the detector also
    learns what empty road looks like.
    """
    placed = [a for a in traffic.actors if a.progress >= 0.0]
    if not placed or rng.uniform() > cfg.follow_fraction:
        return int(rng.integers(0, n))
    target = placed[int(rng.integers(0, len(placed)))]
    spacing = float(np.linalg.norm(np.diff(bundle.centerline, axis=0), axis=1).mean())
    gap_m = float(rng.uniform(*cfg.follow_gap_m))
    behind = int(round(gap_m / max(spacing, 1e-6)))
    return int((int(target.progress * n) - behind) % n)


def capture(
    track: Path,
    out: Path,
    frames: int,
    seed: int = 0,
    camera_cfg: CameraConfig | None = None,
    capture_cfg: CaptureConfig | None = None,
) -> dict:
    """Render `frames` labelled views of the circuit into a YOLO dataset at `out`."""
    camera_cfg = camera_cfg or CameraConfig()
    capture_cfg = capture_cfg or CaptureConfig()
    scene_cfg = SceneConfig()
    bundle = load_scene(track, scene_cfg)
    rng = np.random.default_rng(seed)

    traffic = Traffic.populate(
        bundle.centerline,
        bundle.heights,
        bundle.profile,
        bundle.control_points,
        vehicles=capture_cfg.vehicles,
        motorcycles=capture_cfg.motorcycles,
        pedestrians=capture_cfg.pedestrians,
        seed=seed,
        cfg=scene_cfg,
    )

    camera = DriverCamera(camera_cfg, scene_cfg)
    camera.set_static(bundle.road)
    tangent, normal = frame_at(bundle.centerline)
    n = bundle.centerline.shape[0]

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    from PIL import Image

    counts: dict[str, int] = {}
    written = 0
    empty = 0
    for index in range(frames):
        elapsed = index * capture_cfg.traffic_step_s
        traffic.step(capture_cfg.traffic_step_s, bundle.centerline, bundle.heights, bundle.profile)
        update_signals(bundle.fixtures, elapsed, SignalTiming())

        i = _camera_sample(rng, bundle, traffic, capture_cfg, n)
        lateral = float(rng.uniform(-1.0, 1.0)) * bundle.halfwidth[i] * capture_cfg.lateral_fraction
        pos = bundle.centerline[i] + normal[i] * lateral
        heading = float(np.arctan2(tangent[i, 1], tangent[i, 0]))
        heading += float(rng.normal(0.0, capture_cfg.heading_jitter_rad))

        actors = bundle.fixtures + traffic.actors
        frame, boxes = camera.render_with_labels(pos, heading, float(bundle.heights[i]), actors)
        area = float(camera_cfg.width * camera_cfg.height)
        if any((x2 - x1) * (y2 - y1) / area > capture_cfg.max_box_coverage for _, x1, y1, x2, y2 in boxes):
            continue  # the camera spawned inside an object
        if not boxes:
            empty += 1
            # A road with nothing on it is a real view and worth some of the
            # dataset, but a detector learns little from a page of them.
            if empty % 8:
                continue

        split = "val" if rng.uniform() < capture_cfg.val_fraction else "train"
        stem = f"{track.stem}_{index:06d}"
        Image.fromarray(frame).save(out / "images" / split / f"{stem}.jpg", quality=88)
        lines = []
        for kind, x1, y1, x2, y2 in boxes:
            counts[kind] = counts.get(kind, 0) + 1
            cx = (x1 + x2) / 2.0 / camera_cfg.width
            cy = (y1 + y2) / 2.0 / camera_cfg.height
            w = (x2 - x1) / camera_cfg.width
            h = (y2 - y1) / camera_cfg.height
            lines.append(f"{CLASS_INDEX[kind]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1

    camera.release()
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(SCENE_CLASSES)}",
                "names:",
                *[f"  {i}: {name}" for i, name in enumerate(SCENE_CLASSES)],
                "",
            ]
        )
    )
    manifest = {
        "track": str(track),
        "frames_written": written,
        "frames_requested": frames,
        "boxes": counts,
        "camera": {"width": camera_cfg.width, "height": camera_cfg.height, "fov_deg": camera_cfg.fov_deg},
        "classes": list(SCENE_CLASSES),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=Path("data/tracks/kl.geojson"))
    parser.add_argument("--out", type=Path, default=Path("data/perception/kl"))
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=CameraConfig.width)
    parser.add_argument("--height", type=int, default=CameraConfig.height)
    args = parser.parse_args(argv)

    manifest = capture(
        args.track,
        args.out,
        args.frames,
        seed=args.seed,
        camera_cfg=CameraConfig(width=args.width, height=args.height),
    )
    print(f"[ok] {args.out} {manifest['frames_written']} frames, boxes {manifest['boxes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
