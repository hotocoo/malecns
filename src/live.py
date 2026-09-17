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
    ) -> None:
        from car_env import CarConfig, CarEnv, load_geojson_centerline
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
            SensorConfig(stride=1, weights=weights, device=device),
            scene_cfg,
        )
        self.detector: Detector = self.sensor.detector
        self.eye: DetectionEye = self.sensor.eye
        # The events a real street throws at a driver, staged near this car.
        from events import Director

        self.director = Director(
            self.scene.centerline, self.scene.heights, self.scene.profile, self.traffic, seed=seed
        )

        cfg = CarConfig(
            layout="geojson",
            geojson_path=str(track),
            max_laps=0,
            max_speed=17.0,  # a street car; the F1 default cannot corner here
            lane_offset_m=-1.9,  # start in the left lane, as Malaysia drives
        )
        self.device = torch.device("cpu")
        self.env = CarEnv(batch=1, device=self.device, cfg=cfg)
        points, proj = load_geojson_centerline(
            track, cfg.track_scale, cfg.n_points, cfg.smooth_m, cfg.mirror, return_projection=True
        )
        centerline = points.numpy()
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
        device = pick_device("cpu")
        brain = Brain(connectome, batch=1, config=LIFConfig(), device=device, weight_scale=defaults.WEIGHT_SCALE)
        saved_cfg = saved.get("agent_cfg") or {}
        cfg = AgentConfig(
            n_rays=self.eye.width,
            substeps=int(saved_cfg.get("substeps", defaults.SUBSTEPS)),
        )
        agent = ConnectomeAgent(brain, connectome.neurons, cfg)
        theta = agent.unpack(saved["mu"].to(device).unsqueeze(0))
        agent.reset(batch=1)
        agent.seed(0)
        self.agent = agent
        self.brain = brain
        self.driver_note = f"connectome, generation {saved.get('generation', 0)}"

        def drive(observation: torch.Tensor) -> torch.Tensor:
            return agent.act(observation.to(device), theta)

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
            road_z = float(self.sensor.road_height_at(self.env.last_progress.numpy())[0])
            frame = self.sensor.camera.render(
                self.env.pos[0].numpy(), float(self.env.heading[0]), road_z, actors
            )
            detections = self.detector.detect([frame], self.sensor.cfg.detector_confidence)[0]
            telemetry = self.telemetry(detections)
            shared.publish(encode_jpeg(draw_overlay(frame, detections, telemetry)), telemetry)

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
<header>driver camera &mdash; real frames, real detector, real charges</header>
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
            self.send_error(404)

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
    parser.add_argument("--fps", type=float, default=20.0)
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
            )
        except Exception as exc:  # the page would otherwise sit blank forever
            print(f"[live] simulation failed to start: {exc}")
            shared.running = False
            raise
        built["simulation"] = simulation
        print(f"[live] {simulation.law.summary()}")
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
