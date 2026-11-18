"""Watch the car drive, through its own camera, live.

This serves the real thing: the frames the renderer produces for the driver,
with the boxes the detector actually returned drawn on them, and the charges
the law actually made this step. Nothing on the page is staged - if the
detector misses a motorcycle, the stream shows no box on that motorcycle, and
the driver did not see it either.

  python3 src/live.py --track data/tracks/kl.geojson \\
      --weights checkpoints/perception/kl/weights/best.pt

Then open http://127.0.0.1:8near/ . Two endpoints do the work:

  /stream.mjpg  the camera feed as multipart JPEG, one part per rendered frame
  /telemetry    what the law is charging, the posted limit, what was detected

The simulation runs in its own thread at the control rate and publishes the
newest frame; the stream serves whatever is newest when a client asks, so a
slow browser drops frames instead of holding the car back.
"""

from __future__ import annotations

import argparse
import json
import threading
from urllib.parse import unquote
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from camera import CameraConfig
import defaults
from defaults import CONTROL_DT_S
from law import load_law
from lawreward import GREEN, AMBER, RED, LawEnforcer, SignalTiming, signal_phase
from perceive import CHANNELS, DEFAULT_WEIGHTS, Detector, DetectionEye, EyeConfig, describe
from roadlaw import control_points_for_circuit, legal_profile_for_circuit

BOX_COLOUR = {
    "car": (60, 220, 90),
    "motorcycle": (250, 190, 40),
    "person": (250, 90, 90),
    "traffic light": (90, 170, 255),
    "stop sign": (255, 120, 200),
}
PHASE_NAME = {GREEN: "green", AMBER: "amber", RED: "red"}


@dataclass
class Shared:
    """The newest frame and telemetry, handed from the sim thread to the server."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    jpeg: bytes | None = None
    telemetry: dict = field(default_factory=dict)
    world: dict | None = None
    world_bin: bytes = b""
    moving: dict | None = None
    frame_index: int = 0
    running: bool = True

    def publish(self, jpeg: bytes, telemetry: dict) -> None:
        with self.lock:
            self.jpeg = jpeg
            self.telemetry = telemetry
            self.frame_index += 1

    def latest(self) -> tuple[bytes | None, dict, int]:
        with self.lock:
            return self.jpeg, dict(self.telemetry), self.frame_index


# What the driver's own car looks like on the 3D page.
EGO_MODEL = "kenney_car_kit/Models/GLB format/sedan-sports.glb"


def _asset_available(relative: str) -> bool:
    from assets import available

    return available(relative)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def draw_overlay(frame: np.ndarray, detections, telemetry: dict) -> np.ndarray:
    """The detector's boxes and the law's reading, painted on the frame."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    for found in detections:
        colour = BOX_COLOUR.get(found.kind, (220, 220, 220))
        draw.rectangle([found.x1, found.y1, found.x2, found.y2], outline=colour, width=2)
        draw.text((found.x1 + 2, max(found.y1 - 11, 0)), f"{found.kind} {found.confidence:.2f}", fill=colour)

    lines = [
        f"{telemetry['speed_kmh']:.0f} km/h",
        f"limit {telemetry['limit_kmh']}" if telemetry["limit_kmh"] else "limit unposted",
        f"keep {telemetry['driving_side']}",
        f"law {telemetry['law_total']:+.2f}",
    ]
    for event in telemetry.get("events", [])[:3]:
        lines.append(f"! {event}")
    for i, text in enumerate(lines):
        draw.text((8, 8 + 12 * i), text, fill=(245, 245, 245))
    return np.asarray(image)


class Simulation:
    """One car, driving the surveyed circuit, seen through its own camera."""

    def __init__(
        self,
        track: Path,
        weights: str,
        camera_cfg: CameraConfig,
        eye_cfg: EyeConfig,
        vehicles: int,
        motorcycles: int,
        pedestrians: int,
        seed: int,
        device: str | None,
        checkpoint: Path | None = None,
        detect_every: int = 4,
    ) -> None:
        from car_env import CarEnv, load_geojson_centerline, street_car_config
        from dataset import load_scene
        from eye_camera import CameraSensor, SensorConfig
        from scene import SceneConfig, Traffic

        self.track = track
        scene_cfg = SceneConfig()
        self.scene = load_scene(track, scene_cfg)
        self.traffic = Traffic.populate(
            self.scene.centerline,
            self.scene.heights,
            self.scene.profile,
            self.scene.control_points,
            vehicles=vehicles,
            motorcycles=motorcycles,
            pedestrians=pedestrians,
            seed=seed,
            cfg=scene_cfg,
        )
        self.sensor = CameraSensor(
            self.scene,
            self.traffic,
            camera_cfg,
            eye_cfg,
            SensorConfig(stride=detect_every, weights=weights, device=device),
            scene_cfg,
        )
        self.detector: Detector = self.sensor.detector
        self.eye: DetectionEye = self.sensor.eye
        # The events a real street throws at a driver, staged near this car.
        from events import Director

        self.director = Director(
            self.scene.centerline, self.scene.heights, self.scene.profile, self.traffic, seed=seed
        )

        cfg = street_car_config(
            layout="geojson",
            geojson_path=str(track),
            max_laps=0,
            max_speed=17.0,  # 61 km/h: a street car under a 35-60 km/h limit
            lane_offset_m=-1.9,  # start in the left lane, as Malaysia drives
        )
        self.device = torch.device("cpu")
        self.env = CarEnv(batch=1, device=self.device, cfg=cfg)
        points, proj = load_geojson_centerline(
            track, cfg.track_scale, cfg.n_points, cfg.smooth_m, cfg.mirror, return_projection=True
        )
        centerline = points.numpy()
        self.projection = proj
        profile = legal_profile_for_circuit(track, centerline, proj)
        control_points = control_points_for_circuit(track, proj, load_law(), centerline=centerline)
        self.law = LawEnforcer(profile, control_points, self.env.track.centerline.cpu(), self.device)
        self.env.attach_law(self.law)
        # The camera is the eye here too: without this the environment would
        # still hand out ray distances, and the connectome - evolved on a
        # 120-channel detection view - would be reading a 20-column ray fan.
        self.env.attach_sensor(self.sensor)
        self.timing = SignalTiming()
        self.agent = None
        self.brain = None
        self.frame_index = 0
        self.last_detections = None
        self.detect_every = detect_every
        self.last_observation = None
        self.last_action = None
        self.driver_note = ""
        self.driver = self._load_driver(checkpoint)
        print(f"[live] driver: {self.driver_note}")
        self.env.reset()

    def _load_driver(self, checkpoint: Path | None):
        """The evolved connectome, when a checkpoint for this eye exists.

        A checkpoint is only usable if it was evolved against an eye of the same
        width: the observation is one column per channel, and a brain built for
        96 channels reads a 120-channel view as nonsense. A mismatch is reported
        and the stream falls back to a scripted cruise, which still shows the
        perception and the law working - it is simply not the connectome
        driving, and the page says so.
        """
        if checkpoint is None or not Path(checkpoint).exists():
            self.driver_note = "no checkpoint: scripted cruise"
            return None
        try:
            saved = torch.load(checkpoint, map_location="cpu")
        except (RuntimeError, EOFError) as exc:
            self.driver_note = f"checkpoint unreadable ({exc.__class__.__name__}): scripted cruise"
            return None

        width = int(saved.get("eye_width", -1))
        if width != self.eye.width:
            self.driver_note = f"checkpoint eye {width} != {self.eye.width}: scripted cruise"
            return None

        from agent import AgentConfig, ConnectomeAgent
        from brain import Brain, LIFConfig, load_connectome, pick_device

        connectome = load_connectome(Path("data/graph"))
        # The brain is the slow link in this loop: on CPU one control step of
        # the full connectome costs hundreds of milliseconds and pins the page
        # at a couple of frames a second.
        device = pick_device("auto")
        brain = Brain(connectome, batch=1, config=LIFConfig(), device=device, weight_scale=defaults.WEIGHT_SCALE)
        # The interface the checkpoint was evolved against: its eye encoding,
        # readout normalisation and, when calibrated, the fitted readout
        # itself. Built to this build's defaults instead, a fitted readout
        # reads as noise and the car drives into the kerb.
        from eye_camera import agent_config_for

        saved_cfg = saved.get("agent_cfg") or {}
        base = AgentConfig.from_saved(saved_cfg, substeps=int(saved_cfg.get("substeps", defaults.SUBSTEPS)))
        cfg = agent_config_for(self.sensor, base)
        agent = ConnectomeAgent(brain, connectome.neurons, cfg)
        calibrated = agent.load_readout(saved)
        theta = agent.unpack(saved["mu"].reshape(-1).to(device).unsqueeze(0))
        agent.reset(batch=1)
        agent.seed(0)
        self.agent = agent
        self.brain = brain
        self.driver_note = (
            f"connectome, generation {saved.get('generation', 0)}, eyes {cfg.eye_encoding}"
            + (", calibrated readout" if calibrated else "")
        )

        def drive(observation: torch.Tensor) -> torch.Tensor:
            # The environment lives on the CPU; the brain may not.
            return agent.act(observation.to(device), theta).to(observation.device)

        return drive

    def control(self, observation: torch.Tensor) -> torch.Tensor:
        """Steer along the lap and hold a speed under whatever is posted."""
        if self.driver is not None:
            return self.driver(observation)
        env = self.env
        n = env.track.centerline.shape[0]
        index = (env.last_progress * n).long().clamp(0, n - 1)
        ahead = env.track.centerline[(index + 12) % n] - env.pos
        want = torch.atan2(ahead[:, 1], ahead[:, 0]) - env.heading
        want = torch.atan2(want.sin(), want.cos())
        steer = (want / env.cfg.max_steer_rad).clamp(-1.0, 1.0)
        limit = self.law.limit_at(env.last_progress)
        target = torch.where(torch.isfinite(limit), limit, torch.full_like(limit, env.cfg.max_speed * 0.4))
        pedal = ((target - env.speed) / 4.0).clamp(-1.0, 1.0)
        return torch.stack([steer, pedal], dim=1)

    def telemetry(self, detections) -> dict:
        env, terms = self.env, self.env.last_terms
        limit = float(terms.get("law_limit_mps", torch.tensor([float("inf")]))[0])
        signals = self.law.signals
        phases = {}
        if signals["progress"].numel():
            elapsed = float(env.step_count[0]) * env.cfg.dt_s
            phase = signal_phase(signals["node_id"], torch.full_like(signals["progress"], elapsed), self.timing)
            phases = {int(i): PHASE_NAME[int(p)] for i, p in zip(signals["node_id"].tolist(), phase.tolist())}
        return {
            "step": int(env.step_count[0]),
            "speed_kmh": float(env.speed[0]) * 3.6,
            "limit_kmh": round(limit * 3.6) if np.isfinite(limit) else None,
            "driving_side": self.law.driving_side or "unknown",
            "offset_m": float(terms.get("law_offset_m", torch.zeros(1))[0]),
            "lap": float(env.laps[0]),
            "law_total": float(terms.get("law_total", torch.zeros(1))[0]),
            "law": {
                key[4:]: float(terms[key][0])
                for key in ("law_speeding", "law_wrong_side", "law_red_signal", "law_stop_line", "law_crossing")
                if key in terms
            },
            "detections": describe(detections),
            "detection_count": len(detections),
            "events": sorted({h.kind for h in self.director.active}),
            "events_total": sum(self.director.summary().values()),
            "nearest_hazard_m": round(self.director.nearest_hazard_m(env.pos[0].numpy()), 1),
            "signals": phases,
            "track": str(self.track),
            "driver": self.driver_note,
            "senses": self.senses(),
            "brain": self.brain_state(),
            "controls": {
                "steer": float(self.last_action[0]) if self.last_action is not None else 0.0,
                "pedal": float(self.last_action[1]) if self.last_action is not None else 0.0,
            },
        }

    def world(self) -> tuple[dict, bytes]:
        """The static city for the browser to build once: a header and one binary mesh.

        The very triangles the camera renders (`camera.assemble_static`):
        joined carriageways, kerbs, pavements, paint, buildings, ground, each
        vertex with its material so the page textures it the same way. The
        header carries the counts and the lap; the bytes are float32 positions
        (x, y east-north, z up), uint8 colours and uint8 materials.
        """
        from camera import assemble_static

        scene = self.scene
        vertices, colours, _normals, materials = assemble_static(
            scene.road, scene.network, scene.buildings, self.sensor.camera.cfg.ground_colour
        )
        payload = (
            vertices.astype("<f4").tobytes()
            + np.clip(np.rint(colours * 255.0), 0, 255).astype(np.uint8).tobytes()
            + np.clip(np.rint(materials), 0, 255).astype(np.uint8).tobytes()
        )
        header = {
            "vertices": int(vertices.shape[0]),
            "lap": [[round(float(x), 2), round(float(y), 2)] for x, y in scene.centerline],
            "lap_width": round(float(scene.halfwidth.mean() * 2.0), 2),
            "buildings": int(scene.buildings.surface.shape[0] // 3) if scene.buildings is not None else 0,
            "driving_side": self.law.driving_side or "left",
            "sky": list(self.sensor.camera.cfg.sky_colour),
            "attribution": "(c) OpenStreetMap contributors, ODbL 1.0",
        }
        return header, payload

    def moving(self) -> dict:
        """Where everything is right now, for the 3D view to move its objects."""
        env = self.env
        actors = []
        for actor in self.scene.fixtures + self.traffic.actors:
            actors.append(
                {
                    "kind": actor.kind,
                    "x": round(float(actor.pos[0]), 2),
                    "y": round(float(actor.pos[1]), 2),
                    "z": round(float(actor.pos[2]), 2),
                    "heading": round(float(actor.heading), 3),
                    "size": [round(float(v), 2) for v in actor.size],
                    "state": int(actor.state),
                    "hazard": actor.hazard,
                    "model": self.model_for(actor),
                }
            )
        return {
            "car": {
                "x": round(float(env.pos[0][0]), 2),
                "y": round(float(env.pos[0][1]), 2),
                "heading": round(float(env.heading[0]), 3),
                "speed_kmh": round(float(env.speed[0]) * 3.6, 1),
                "size": [round(float(env.cfg.car_length), 2), round(float(env.cfg.car_halfwidth * 2.0), 2), 1.45],
                "model": EGO_MODEL if _asset_available(EGO_MODEL) else None,
            },
            "actors": actors,
            "frame": self.frame_index,
        }

    def model_for(self, actor) -> str | None:
        """The published model the camera draws this actor with, for the 3D page.

        The same choice the renderer makes (`DriverCamera._emit_model`), so
        the page shows the car the detector saw. None when the kind has no
        glTF model installed; the page then draws a box, as the camera does.
        """
        from assets import MissingAsset, pick
        from camera import MODELS_FOR

        family = MODELS_FOR.get(actor.kind)
        if not family:
            return None
        try:
            relative = pick(family, max(actor.node_id, 0) if actor.node_id >= 0 else self.sensor.camera._variant(actor))
        except MissingAsset:
            return None
        return relative if relative.lower().endswith((".glb", ".gltf")) else None

    def senses(self) -> dict:
        """What the eye handed the brain this step, column by column.

        This is the driver's whole world: nothing else reaches it. Each row is
        one channel of the detection view, each column one slice of the field
        of view, left to right.
        """
        if self.last_observation is None:
            return {}
        rows = self.eye._rows()
        columns = self.eye.config.columns
        grid = self.last_observation[: rows.__len__() * columns].reshape(len(rows), columns)
        return {
            "columns": columns,
            "rows": rows,
            "values": [[round(float(v), 3) for v in row] for row in grid],
            "own_speed": round(float(self.last_observation[-1]), 3),
        }

    def brain_state(self) -> dict:
        """How much of the connectome is firing, and how hard it is driving.

        `rate_hz` is what the eye injected into the visual neurons; `dn_hz` is
        what the descending neurons - the cells that actually carry a command
        out of the brain - fired back. A driver that sees nothing has a flat
        sensory row; one that has stopped reacting has a flat descending row.
        """
        agent = self.agent
        if agent is None:
            return {}
        state: dict = {}
        rates = getattr(agent, "last_rates_hz", None)
        if rates is not None and rates.numel():
            row = rates[0].detach().cpu().numpy()
            state["sensory_hz"] = [round(float(v), 2) for v in row[: min(row.size, 64)]]
            state["sensory_mean_hz"] = round(float(row.mean()), 2)
        brain = self.brain
        if brain is not None and getattr(brain, "dn_acc", None) is not None:
            dn = brain.dn_acc.detach().cpu().numpy()
            if dn.size:
                per_body = dn[0] if dn.ndim > 1 else dn
                state["dn_active"] = int((per_body > 0).sum())
                state["dn_total"] = int(per_body.size)
                state["dn_mean"] = round(float(per_body.mean()), 4)
        return state

    def run(self, shared: Shared, fps: float) -> None:
        """Drive, render, detect and publish until asked to stop."""
        from scene import update_signals

        period = 1.0 / max(fps, 1e-3)
        observation = self.env.observe()
        while shared.running:
            started = time.time()
            action = self.control(observation)
            self.last_observation = observation.detach().cpu().numpy()[0]
            self.last_action = action.detach().cpu().numpy()[0]
            observation, _, done = self.env.step(action)
            if bool(done[0]):
                self.env.reset()
                self.sensor.reset()

            elapsed = float(self.env.step_count[0]) * self.env.cfg.dt_s
            # The director moves every actor, including ordinary traffic, so
            # the lane-following and the events run off one clock.
            self.director.step(
                CONTROL_DT_S,
                float(self.env.last_progress[0]),
                self.env.pos[0].numpy(),
                float(self.env.speed[0]),
            )
            update_signals(self.scene.fixtures, elapsed, self.timing)
            actors = self.scene.fixtures + self.traffic.actors
            # The driver already rendered its view and ran the detector over it
            # inside `env.step`. Doing either again here would double the cost
            # of the loop for the same picture, so the page shows exactly what
            # the driver saw.
            self.frame_index += 1
            frame = self.sensor.last_frames[0] if self.sensor.last_frames else None
            if frame is None:
                road_z = float(self.sensor.road_height_at(self.env.last_progress.numpy())[0])
                frame = self.sensor.camera.render(
                    self.env.pos[0].numpy(), float(self.env.heading[0]), road_z, actors
                )
            detections = self.sensor.last_detections[0] if self.sensor.last_detections else []
            telemetry = self.telemetry(detections)
            shared.publish(encode_jpeg(draw_overlay(frame, detections, telemetry)), telemetry)
            with shared.lock:
                shared.moving = self.moving()

            rest = period - (time.time() - started)
            if rest > 0:
                time.sleep(rest)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>malecns - driver camera</title>
<style>
 body{margin:0;background:#101114;color:#e8e8ea;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:12px 16px;border-bottom:1px solid #26272c}
 main{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 img{width:min(880px,96vw);border:1px solid #26272c;background:#000}
 table{border-collapse:collapse;min-width:280px}
 .panel{margin-top:14px}
 .label{color:#8b8d96;margin-bottom:5px;font-size:12px}
 canvas{background:#0b0c0f;border:1px solid #26272c;width:min(880px,96vw)}
 td{padding:3px 10px 3px 0;vertical-align:top}
 td.k{color:#8b8d96}
 .warn{color:#ff8a7a}
</style></head>
<body>
<header>driver camera &mdash; real frames, real detector, real charges &nbsp;|&nbsp; <a href="/3d" style="color:#7fc6ff">3D world view</a></header>
<main>
  <div>
    <img id="feed" src="/stream.mjpg" alt="driver camera">
    <div class="panel">
      <div class="label">what the brain receives &mdash; rows are channels, columns the field of view</div>
      <canvas id="senses" width="880" height="230"></canvas>
    </div>
    <div class="panel">
      <div class="label">steering and pedal</div>
      <canvas id="controls" width="880" height="60"></canvas>
    </div>
  </div>
  <table id="telemetry"></table>
</main>
<script>
const rows = [
  ["step","step"],["speed","speed_kmh"],["posted limit","limit_kmh"],["keeps","driving_side"],
  ["offset from centre","offset_m"],["lap","lap"],["detected","detections"],["law this step","law_total"],
  ["events running","events"],["events so far","events_total"],["nearest hazard","nearest_hazard_m"],
  ["driven by","driver"],
];
async function poll(){
  try{
    const t = await (await fetch("/telemetry")).json();
    const body = rows.map(([label,key])=>{
      let v = t[key];
      if(key==="speed_kmh") v = v.toFixed(0)+" km/h";
      else if(key==="limit_kmh") v = (v===null? "unposted" : v+" km/h");
      else if(key==="offset_m") v = v.toFixed(2)+" m";
      else if(key==="lap") v = v.toFixed(3);
      else if(key==="law_total") v = v.toFixed(3);
      else if(key==="events") v = (v && v.length) ? v.join(", ") : "none";
      else if(key==="nearest_hazard_m") v = (v>1e6? "-" : v+" m");
      return `<tr><td class="k">${label}</td><td>${v}</td></tr>`;
    }).join("");
    const charges = Object.entries(t.law||{}).filter(([,v])=>v<0)
      .map(([k,v])=>`<tr><td class="k">${k}</td><td class="warn">${v.toFixed(3)}</td></tr>`).join("");
    const brain = t.brain || {};
    const extra = Object.entries(brain).filter(([k])=>!Array.isArray(brain[k]))
      .map(([k,v])=>`<tr><td class="k">${k}</td><td>${v}</td></tr>`).join("");
    document.getElementById("telemetry").innerHTML = body + charges + extra;
    drawSenses(t); drawControls(t);
  }catch(e){}
  setTimeout(poll, 250);
}
function drawSenses(t){
  const s = t.senses; if(!s || !s.values) return;
  const c = document.getElementById("senses"), g = c.getContext("2d");
  g.clearRect(0,0,c.width,c.height);
  const rows = s.values.length, cols = s.columns;
  const labelW = 120, cellW = (c.width-labelW)/cols, cellH = c.height/rows;
  for(let r=0;r<rows;r++){
    for(let k=0;k<cols;k++){
      const v = Math.max(0, Math.min(1, s.values[r][k]));
      // Dark where the channel is silent, bright where it is driving hard.
      g.fillStyle = `rgb(${Math.round(20+200*v)},${Math.round(24+150*v)},${Math.round(30+60*v)})`;
      g.fillRect(labelW+k*cellW, r*cellH, cellW-1, cellH-1);
    }
    g.fillStyle = "#8b8d96"; g.font = "10px ui-monospace,monospace";
    g.fillText(s.rows[r], 4, r*cellH + cellH*0.7);
  }
}
function drawControls(t){
  const c = document.getElementById("controls"), g = c.getContext("2d");
  g.clearRect(0,0,c.width,c.height);
  const mid = c.width/2;
  g.strokeStyle="#26272c"; g.beginPath(); g.moveTo(mid,0); g.lineTo(mid,c.height); g.stroke();
  const ctl = t.controls || {steer:0,pedal:0};
  g.fillStyle="#5ad48a"; g.fillRect(mid, 8, ctl.steer*mid, 18);
  g.fillStyle= ctl.pedal>=0 ? "#5aa0d4" : "#ff8a7a";
  g.fillRect(mid, 34, ctl.pedal*mid, 18);
  g.fillStyle="#8b8d96"; g.font="10px ui-monospace,monospace";
  g.fillText("steer "+ctl.steer.toFixed(2), 6, 21);
  g.fillText("pedal "+ctl.pedal.toFixed(2), 6, 47);
}
poll();
</script>
</body></html>
"""


WORLD_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>malecns - 3D world</title>
<style>
body{margin:0;background:#0b0c0f;color:#e8e8ea;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
#hud{position:fixed;top:0;left:0;padding:10px 14px;z-index:5;text-shadow:0 1px 3px #000}
#hud a{color:#7fc6ff}
#keys{position:fixed;bottom:0;left:0;padding:10px 14px;color:#8b8d96;z-index:5}
canvas{display:block}
</style></head>
<body>
<div id="hud">loading city&hellip; &nbsp;<a href="/">camera view</a></div>
<div id="keys">1 chase &middot; 2 overhead &middot; 3 driver &middot; drag orbit, wheel to zoom</div>
<script type="importmap">{"imports": {"three": "/vendor/three/three.module.js"}}</script>
<script type="module">
import * as THREE from "three";
import { GLTFLoader } from "/vendor/three/loaders/GLTFLoader.js";

// Track frame: x east, y north, z up. three.js: y up. Every point is swizzled
// (x, y, z) -> (x, z, -y) on the way in, so the mesh the camera renders and
// the mesh drawn here are the same numbers.
const SKY = new THREE.Color(0x9fb4cc);
const scene = new THREE.Scene();
scene.background = SKY;

const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.5, 4000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
addEventListener("resize", () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

scene.add(new THREE.HemisphereLight(0xdfe8f5, 0x3a3f33, 1.0));
const sun = new THREE.DirectionalLight(0xfff2dd, 1.15);
sun.position.set(-260, 420, 320);
scene.add(sun);

// The static city: one buffer, textured per vertex material by the same GLSL
// the driver's camera uses (spliced in by the server).
const SURFACE_GLSL = `/*SURFACE_GLSL*/`;
const cityMaterial = new THREE.ShaderMaterial({
  uniforms: {
    sunDir: {value: new THREE.Vector3(-0.35, 0.25, 0.90).normalize()},
    ambient: {value: 0.45},
    fogColour: {value: SKY},
    fogStart: {value: 250.0},
    fogEnd: {value: 1400.0},
  },
  vertexShader: `
    attribute float material;
    attribute vec3 colour;
    varying vec3 vWorld;
    varying vec3 vColour;
    varying float vMaterial;
    varying float vDepth;
    void main() {
      vec3 p = vec3(position.x, position.z, -position.y);
      vec4 view = modelViewMatrix * vec4(p, 1.0);
      gl_Position = projectionMatrix * view;
      vWorld = position;
      vColour = colour;
      vMaterial = material;
      vDepth = -view.z;
    }`,
  fragmentShader: `
    precision highp float;
    uniform vec3 sunDir;
    uniform float ambient;
    uniform vec3 fogColour;
    uniform float fogStart;
    uniform float fogEnd;
    varying vec3 vWorld;
    varying vec3 vColour;
    varying float vMaterial;
    varying float vDepth;
    ` + SURFACE_GLSL + `
    void main() {
      // Flat shading from the surface's own slope, in the track frame.
      vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
      if (n.z < 0.0) n = -n;
      float lit = ambient + (1.0 - ambient) * max(dot(n, sunDir), 0.0);
      vec3 shaded = surface_colour(vColour * lit, vMaterial, vWorld);
      float t = clamp((vDepth - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
      gl_FragColor = vec4(mix(shaded, fogColour, t), 1.0);
    }`,
  side: THREE.DoubleSide,
});

const COLOURS = {
  "car":0xc8452f, "motorcycle":0x2b2b30, "person":0xe0b357,
  "truck":0x4a6fa5, "bus":0x4a6fa5, "traffic light":0x1d1f22, "stop sign":0xb3202a
};
const SIGNAL = {0:0x22cc55, 1:0xffbb11, 2:0xee2222};

let actorPool = [], carMesh = null, mode = 1;
addEventListener("keydown", e => { if ("123".includes(e.key)) mode = +e.key; });

let orbit = {on:false, x:0, y:0, yaw:0.6, pitch:0.5, dist:90};
addEventListener("mousedown", e => { orbit.on=true; orbit.x=e.clientX; orbit.y=e.clientY; });
addEventListener("mouseup", () => orbit.on=false);
addEventListener("mousemove", e => {
  if(!orbit.on) return;
  orbit.yaw += (e.clientX-orbit.x)*0.005; orbit.pitch += (e.clientY-orbit.y)*0.005;
  orbit.pitch = Math.max(0.08, Math.min(1.4, orbit.pitch));
  orbit.x=e.clientX; orbit.y=e.clientY;
});
addEventListener("wheel", e => {
  orbit.dist = Math.max(12, Math.min(900, orbit.dist * (1 + e.deltaY*0.001)));
}, {passive:true});

async function buildWorld(){
  const w = await (await fetch("/world.json")).json();
  const raw = await (await fetch("/world.bin")).arrayBuffer();
  const n = w.vertices;
  const positions = new Float32Array(raw, 0, n*3);
  const colours8 = new Uint8Array(raw, n*12, n*3);
  const materials8 = new Uint8Array(raw, n*15, n);
  const colours = new Float32Array(n*3);
  for (let i=0;i<n*3;i++) colours[i] = colours8[i]/255;
  const materials = new Float32Array(n);
  for (let i=0;i<n;i++) materials[i] = materials8[i];
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  g.setAttribute("colour", new THREE.BufferAttribute(colours, 3));
  g.setAttribute("material", new THREE.BufferAttribute(materials, 1));
  // Bounding volume in the swizzled frame, so frustum culling does not drop the city.
  g.computeBoundingSphere(); g.boundingSphere.radius *= 2;
  const city = new THREE.Mesh(g, cityMaterial);
  city.frustumCulled = false;
  scene.add(city);
  document.getElementById("hud").innerHTML =
    `${(n/3).toLocaleString()} triangles &middot; drives on the ${w.driving_side} &middot; ` +
    `<a href="/">camera view</a> &nbsp; <span id="live"></span><br><small>${w.attribution}</small>`;
}

// Actors are drawn with the published models the camera drew them with
// (served from data/assets/raw); a kind with no glTF model is a box, as in
// the camera. glTF is +Y up, -Z forward; the track frame is +Z up, +X at
// heading 0, and (x, y, z) -> (x, z, -y) here, so a model at heading h is
// turned by h - pi/2 about Y and its length lies along its own Z.
const loader = new GLTFLoader();
const gltfCache = new Map();
function loadModel(path){
  if (!gltfCache.has(path)){
    gltfCache.set(path, new Promise(resolve => {
      loader.load("/assets/" + encodeURI(path), g => {
        const box = new THREE.Box3().setFromObject(g.scene);
        resolve({scene: g.scene, box});
      }, undefined, () => resolve(null));
    }));
  }
  return gltfCache.get(path);
}

function forwardOf(h){ return new THREE.Vector3(Math.cos(h), 0, -Math.sin(h)); }

function boxFor(a, colour){
  const [l, wdt, h] = a.size;
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(l,0.2), Math.max(h,0.2), Math.max(wdt,0.2)),
    new THREE.MeshLambertMaterial({color: colour}));
  m.userData.box = true;
  return m;
}

function signalColour(a){
  return (a.kind === "traffic light" && a.state >= 0) ? SIGNAL[a.state] : (COLOURS[a.kind] || 0x888888);
}

// A holder per actor: the model (or box) is attached inside it once loaded,
// scaled to the actor's real size, so moving the holder moves the actor.
function actorHolder(a){
  const holder = new THREE.Group();
  holder.userData = {model: null, kind: a.kind};
  const fallback = boxFor(a, signalColour(a));
  holder.add(fallback);
  holder.userData.fallback = fallback;
  if (a.model){
    loadModel(a.model).then(m => {
      if (!m) return;
      const inst = m.scene.clone(true);
      const size = new THREE.Vector3(); m.box.getSize(size);
      const [l, wdt, hgt] = a.size;
      let sx, sy, sz;
      if (a.kind === "traffic light"){
        // The whole light on its pole, as tall as the head stands.
        const tall = Math.max(a.z + 0.45, 0.5);
        sy = tall / Math.max(size.y, 1e-3); sx = sz = sy;
      } else {
        sx = Math.max(wdt,0.2) / Math.max(size.x, 1e-3);
        sy = Math.max(hgt,0.2) / Math.max(size.y, 1e-3);
        sz = Math.max(l,0.2) / Math.max(size.z, 1e-3);
      }
      inst.scale.set(sx, sy, sz);
      inst.position.set(-(m.box.min.x + size.x/2)*sx, -m.box.min.y*sy, -(m.box.min.z + size.z/2)*sz);
      inst.rotation.y = -Math.PI/2;  // model -Z forward -> holder +X forward
      const wrap = new THREE.Group(); wrap.add(inst);
      holder.userData.model = wrap;
      holder.remove(fallback);
      holder.add(wrap);
      if (a.kind === "traffic light"){
        const lens = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 12),
                                    new THREE.MeshBasicMaterial({color: SIGNAL[Math.max(a.state,0)] || 0xee2222}));
        lens.position.set(0.15, Math.max(a.z, 0.5), 0);
        holder.userData.lens = lens;
        holder.add(lens);
      }
    });
  }
  return holder;
}

function placeActor(holder, a){
  holder.visible = true;
  const base = holder.userData.model ? a.z : a.z + a.size[2]/2;
  holder.position.set(a.x, holder.userData.model ? 0 : base, -a.y);
  if (holder.userData.model && a.kind !== "traffic light") holder.position.y = a.z;
  holder.rotation.y = a.heading;
  if (holder.userData.fallback && !holder.userData.model){
    const f = holder.userData.fallback;
    f.material.color.setHex(signalColour(a));
    f.scale.set(Math.max(a.size[0],0.2)/f.geometry.parameters.width,
                Math.max(a.size[2],0.2)/f.geometry.parameters.height,
                Math.max(a.size[1],0.2)/f.geometry.parameters.depth);
  }
  if (holder.userData.lens && a.state >= 0) holder.userData.lens.material.color.setHex(SIGNAL[a.state]);
}

let carHeading = 0;
async function poll(){
  try {
    const s = await (await fetch("/state.json")).json();
    if (s.actors){
      while (actorPool.length < s.actors.length){
        const holder = actorHolder(s.actors[actorPool.length]);
        scene.add(holder); actorPool.push(holder);
      }
      s.actors.forEach((a, i) => {
        let holder = actorPool[i];
        // A slot reused for a different kind of thing gets rebuilt.
        if (holder.userData.kind !== a.kind){
          scene.remove(holder); holder = actorHolder(a); scene.add(holder); actorPool[i] = holder;
        }
        placeActor(holder, a);
      });
      for (let i=s.actors.length;i<actorPool.length;i++) actorPool[i].visible = false;
    }
    if (s.car){
      if (!carMesh){
        carMesh = actorHolder({kind: "car", model: s.car.model, size: s.car.size || [4.5, 1.8, 1.45], z: 0, state: -1, x: s.car.x, y: s.car.y, heading: s.car.heading});
        if (carMesh.userData.fallback) carMesh.userData.fallback.material.color.setHex(0x27d07a);
        scene.add(carMesh);
      }
      placeActor(carMesh, {kind: "car", size: s.car.size || [4.5, 1.8, 1.45], z: 0, state: -1, x: s.car.x, y: s.car.y, heading: s.car.heading});
      carHeading = s.car.heading;
      const live = document.getElementById("live");
      if (live) live.textContent = `${s.car.speed_kmh.toFixed(0)} km/h`;
    }
  } catch (e) {}
  setTimeout(poll, 60);
}

function frame(){
  if (carMesh){
    const fwd = forwardOf(carHeading);
    const p = carMesh.position;
    if (mode === 1){
      camera.position.set(p.x - fwd.x*16, 7.5, p.z - fwd.z*16);
      camera.lookAt(p.x + fwd.x*6, 1.0, p.z + fwd.z*6);
    } else if (mode === 2){
      camera.position.set(
        p.x + Math.sin(orbit.yaw)*orbit.dist*Math.cos(orbit.pitch),
        orbit.dist*Math.sin(orbit.pitch) + 6,
        p.z + Math.cos(orbit.yaw)*orbit.dist*Math.cos(orbit.pitch)
      );
      camera.lookAt(p);
    } else {
      camera.position.set(p.x + fwd.x*1.2, 1.25, p.z + fwd.z*1.2);
      camera.lookAt(p.x + fwd.x*40, 1.0, p.z + fwd.z*40);
    }
  }
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

buildWorld().then(() => { poll(); frame(); });
</script>
</body></html>
"""


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quieter than the default access log
            pass

        def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/telemetry"):
                _, telemetry, index = shared.latest()
                telemetry["frame"] = index
                body = json.dumps(telemetry).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/stream.mjpg"):
                self.stream()
                return
            if self.path.startswith("/world.json"):
                self.json_reply(shared.world or {})
                return
            if self.path.startswith("/world.bin"):
                body = shared.world_bin
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/state.json"):
                self.json_reply(shared.moving or {})
                return
            if self.path.startswith("/3d"):
                from camera import SURFACE_GLSL

                self.html_reply(WORLD_PAGE.replace("/*SURFACE_GLSL*/", SURFACE_GLSL))
                return
            if self.path.startswith("/vendor/"):
                self.serve_vendor(self.path[len("/vendor/") :])
                return
            if self.path.startswith("/assets/"):
                self.serve_asset(unquote(self.path[len("/assets/") :]))
                return
            self.send_error(404)

        def json_reply(self, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def html_reply(self, page: str) -> None:
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def serve_asset(self, relative: str) -> None:
            """A published model (data/assets/raw), so the 3D page draws the cars the camera saw."""
            from assets import RAW_DIR

            root = Path(RAW_DIR).resolve()
            target = (root / relative).resolve()
            if not str(target).startswith(str(root)) or not target.is_file():
                self.send_error(404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "model/gltf-binary" if target.suffix == ".glb" else "application/octet-stream")
            self.send_header("Cache-Control", "max-age=3600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def serve_vendor(self, relative: str) -> None:
            """The bundled three.js build, so the 3D view needs no network."""
            root = (Path("web") / "vendor").resolve()
            target = (root / relative).resolve()
            if not str(target).startswith(str(root)) or not target.is_file():
                self.send_error(404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream(self) -> None:
            """Multipart JPEG: one part per newly published frame."""
            boundary = "malecnsframe"
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while shared.running:
                    jpeg, _, index = shared.latest()
                    if jpeg is None or index == last:
                        time.sleep(0.005)
                        continue
                    last = index
                    head = (
                        f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode()
                    self.wfile.write(head)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the viewer closed the tab

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=Path("data/tracks/kl.geojson"))
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="detector weights; the fine-tuned ones by default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--detect-every",
        type=int,
        default=4,
        help="frames between detector passes; the stream runs at --fps regardless",
    )
    parser.add_argument("--width", type=int, default=CameraConfig.width)
    parser.add_argument("--height", type=int, default=CameraConfig.height)
    parser.add_argument("--columns", type=int, default=EyeConfig.columns)
    parser.add_argument("--vehicles", type=int, default=190)
    parser.add_argument("--motorcycles", type=int, default=150)
    parser.add_argument("--pedestrians", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/camera/driver.pt"),
        help="the evolved driver to watch; falls back to a scripted cruise when it does not fit this eye",
    )
    args = parser.parse_args(argv)

    shared = Shared()
    built: dict[str, Simulation] = {}

    def drive() -> None:
        # The GL context belongs to the thread that creates it: macOS segfaults
        # if a context made on the main thread is drawn from another. So the
        # whole simulation, camera included, is built here and never crosses
        # the thread boundary.
        try:
            simulation = Simulation(
                args.track,
                args.weights,
                CameraConfig(width=args.width, height=args.height),
                EyeConfig(columns=args.columns),
                args.vehicles,
                args.motorcycles,
                args.pedestrians,
                args.seed,
                args.device,
                args.checkpoint,
                args.detect_every,
            )
        except Exception as exc:  # the page would otherwise sit blank forever
            print(f"[live] simulation failed to start: {exc}")
            shared.running = False
            raise
        built["simulation"] = simulation
        print(f"[live] {simulation.law.summary()}")
        header, mesh = simulation.world()
        with shared.lock:
            shared.world = header
            shared.world_bin = mesh
        print(f"[live] 3D world: {header['vertices']:,} vertices, {header['buildings']:,} building triangles, {len(mesh) / 1e6:.1f} MB")
        simulation.run(shared, args.fps)

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    print(f"[live] http://{args.host}:{args.port}/  (camera {args.width}x{args.height}, detector {args.weights})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shared.running = False
        server.server_close()
        if "simulation" in built:
            built["simulation"].sensor.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
