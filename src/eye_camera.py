"""The camera eye: what the connectome sees instead of lidar rays.

`CarEnv` used to hand the brain a fan of ray distances measured against the
track geometry. This replaces that with the real chain: render the view from
each car, run a detector over the frames, and feed the detections in.

The swap is deliberate and total. A car driving on this sensor has no access to
the track's geometry, no ray, no clearance, no knowledge of where the road
edge is. It has pixels, a detector's opinion about them, and its own speed -
which is what a camera-driven vehicle actually has.

Rendering and detecting cost far more than a ray march, so the sensor batches:
one render per body per control step, then one detector call for the whole
population. `stride` lets perception run slower than the physics, holding the
last detections in between, the way a real camera at 20 Hz feeds a controller
running at 60 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from camera import CameraConfig, DriverCamera
from perceive import DEFAULT_WEIGHTS, Detector, DetectionEye, EyeConfig
from scene import SceneConfig, Traffic, update_signals
from lawreward import SignalTiming


@dataclass(frozen=True)
class SensorConfig:
    """How often the eye looks, and what it looks with."""

    stride: int = 3  # control steps between detector passes; 3 at 62.5 Hz is ~21 Hz
    weights: str = DEFAULT_WEIGHTS
    device: str | None = None
    detector_confidence: float = 0.25


class CameraSensor:
    """Renders, detects and encodes the view for a population of cars."""

    def __init__(
        self,
        scene,
        traffic: Traffic,
        camera_cfg: CameraConfig | None = None,
        eye_cfg: EyeConfig | None = None,
        sensor_cfg: SensorConfig | None = None,
        scene_cfg: SceneConfig | None = None,
        timing: SignalTiming | None = None,
        detector=None,
    ) -> None:
        self.scene = scene
        self.traffic = traffic
        self.cfg = sensor_cfg or SensorConfig()
        self.camera = DriverCamera(camera_cfg or CameraConfig(), scene_cfg or SceneConfig())
        self.camera.set_static(scene.road)
        self.eye = DetectionEye(eye_cfg or EyeConfig())
        # A caller may pass its own detector, which is how the tests run the
        # whole chain without loading weights.
        self.detector = detector if detector is not None else Detector(self.cfg.weights, device=self.cfg.device)
        self.timing = timing or SignalTiming()
        self._last: np.ndarray | None = None
        self._step = 0
        self.frames_rendered = 0
        self.detector_calls = 0

    @property
    def width(self) -> int:
        """Channels the brain receives, the detection vector plus own speed."""
        return self.eye.width + 1

    def reset(self) -> None:
        self._last = None
        self._step = 0

    def road_height_at(self, progress: np.ndarray) -> np.ndarray:
        n = self.scene.centerline.shape[0]
        index = np.clip((progress * n).astype(np.int64), 0, n - 1)
        return self.scene.heights[index]

    def observe(
        self,
        pos: torch.Tensor,
        heading: torch.Tensor,
        speed: torch.Tensor,
        progress: torch.Tensor,
        elapsed_s: float,
        max_speed: float,
    ) -> torch.Tensor:
        """(batch, width) for the population, detections first and speed last.

        Between detector passes the last detections are held, which is what a
        controller running faster than its camera actually sees.
        """
        device = speed.device
        batch = speed.shape[0]
        due = self._last is None or self._last.shape[0] != batch or self._step % max(self.cfg.stride, 1) == 0
        self._step += 1

        if due:
            update_signals(self.scene.fixtures, elapsed_s, self.timing)
            actors = self.scene.fixtures + self.traffic.actors
            positions = pos.detach().cpu().numpy()
            headings = heading.detach().cpu().numpy()
            road_z = self.road_height_at(progress.detach().cpu().numpy())
            frames = [
                self.camera.render(positions[i], float(headings[i]), float(road_z[i]), actors)
                for i in range(batch)
            ]
            self.frames_rendered += batch
            self.detector_calls += 1
            detections = self.detector.detect(frames, self.cfg.detector_confidence)
            self._last = self.eye.encode_batch(detections, frames)

        vector = torch.from_numpy(self._last).to(device=device, dtype=speed.dtype)
        own_speed = (speed / max_speed).unsqueeze(1)
        return torch.cat([vector, own_speed], dim=1)

    def step_traffic(self, dt_s: float) -> None:
        self.traffic.step(dt_s, self.scene.centerline, self.scene.heights, self.scene.profile)

    def stats(self) -> dict[str, float]:
        return {
            "frames_rendered": float(self.frames_rendered),
            "detector_calls": float(self.detector_calls),
            "eye_width": float(self.width),
        }

    def release(self) -> None:
        self.camera.release()


def agent_config_for(sensor: CameraSensor, base=None):
    """An `AgentConfig` whose visual groups match this sensor's channels.

    The agent splits its visual projection neurons evenly across `n_rays`
    input groups and reads the observation's first `n_rays` columns, with the
    car's own speed last. The camera eye keeps that contract - it simply has
    more columns than a ray fan, one per class per view column - so pointing
    `n_rays` at the eye's width is the whole of the swap on the brain's side.
    """
    from dataclasses import replace

    from agent import AgentConfig

    base = base or AgentConfig()
    return replace(base, n_rays=sensor.eye.width)


def build_sensor(
    track: Path,
    vehicles: int = 190,
    motorcycles: int = 150,
    pedestrians: int = 60,
    seed: int = 0,
    camera_cfg: CameraConfig | None = None,
    eye_cfg: EyeConfig | None = None,
    sensor_cfg: SensorConfig | None = None,
) -> CameraSensor:
    """A sensor over one circuit, with its surveyed fixtures and a traffic population."""
    from dataset import load_scene

    scene_cfg = SceneConfig()
    scene = load_scene(track, scene_cfg)
    traffic = Traffic.populate(
        scene.centerline,
        scene.heights,
        scene.profile,
        scene.control_points,
        vehicles=vehicles,
        motorcycles=motorcycles,
        pedestrians=pedestrians,
        seed=seed,
        cfg=scene_cfg,
    )
    return CameraSensor(scene, traffic, camera_cfg, eye_cfg, sensor_cfg, scene_cfg)
