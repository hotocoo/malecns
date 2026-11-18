"""A learned lane sense: camera frame in, where the road goes out.

The hand-written lane measurement (`perceive.lane_geometry`) reads grey and
white pixels. Measured against the simulator's ground truth it barely tracks
the car's offset or heading (|r| <= 0.26 on every channel), and the eye as a
whole could not predict the teacher's steering at all (held-out R^2 < 0).
Pedal fitted fine, so the brain was never told where the road goes.

This is the camera-only answer real lane-keeping systems use: a small CNN that
regresses road geometry from the image. Labels are free here because the
simulator knows the pose: the car's signed offset from the road centreline and
the bearing (in the car's own frame) to the centreline at a few distances
ahead. Those bearings are what pure pursuit steers by, so they carry the
curve-ahead information offset and heading alone lack.

Each output is signed, but the brain's sensory drive is rectified
(`agent.sensory_rates` clamps at zero), so a negative value would fire like
nothing at all. Each output is therefore split into a mirrored pair of
positive channels, left and right, the way the fly's own visual system codes
direction in opposed populations.

Train on the pool routes only; the held-out route (`kl_r6`) stays unseen so the
reported error says something about streets it has not memorised.

  python3 src/lane_head.py --routes data/tracks/kl.geojson data/tracks/kl_pool \
      --holdout data/tracks/kl_holdout/kl_r6.geojson

This is trained on rendered frames; nothing here shows it transfers to real
dashcam footage.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

LANE_HEAD_WEIGHTS = "checkpoints/perception/lane_head.pt"


def default_lane_head() -> str | None:
    """The trained lane model when one is on disk, else None (pixel-rule lane measurement)."""
    return LANE_HEAD_WEIGHTS if Path(LANE_HEAD_WEIGHTS).exists() else None


@dataclass(frozen=True)
class LaneHeadConfig:
    # Distances ahead (m) at which the bearing to the road centreline is read.
    bearings_m: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
    # Network input, downsampled from the camera frame.
    input_width: int = 160
    input_height: int = 96
    # Physical units mapped to ~1 so the targets share a scale.
    offset_scale_m: float = 1.5
    bearing_scale_rad: float = 0.25
    # The lane the car keeps, as a signed offset along the road's left normal
    # (calibrate_camera/train_camera --lane-offset). Labels are measured to
    # this lane path, not the centreline: against the centreline every label
    # carried a constant bias (-1.86 m) that hid the small on-lane variation.
    lane_offset_m: float = -1.9
    # Pose noise around the lane while capturing: what a driver in its lane
    # sees, plus enough drift to learn the recovery. Capturing over the whole
    # carriageway left the on-lane error (5 m bearing RMSE 0.21 rad) larger
    # than the on-lane spread (0.085 rad).
    capture_offset_m: float = 1.2
    capture_heading_rad: float = 0.12
    # Share of each route's traffic on the road while capturing.
    capture_traffic: float = 0.5

    @property
    def outputs(self) -> int:
        return 1 + len(self.bearings_m)


def channel_names(cfg: LaneHeadConfig, width: int) -> tuple[str, ...]:
    """Mirrored pair per output, padded with unused slots to `width`."""
    names = []
    for label in ["offset"] + [f"bearing_{int(m)}m" for m in cfg.bearings_m]:
        names += [f"{label}+", f"{label}-"]
    return tuple(names + [f"unused[{k}]" for k in range(width - len(names))])


def labels_for(centerline: np.ndarray, index: np.ndarray, pos: np.ndarray, heading: np.ndarray, cfg: LaneHeadConfig) -> np.ndarray:
    """(batch, outputs) scaled targets: offset from the lane path, bearings to it ahead."""
    n = centerline.shape[0]
    seg = np.hypot(*np.diff(centerline, axis=0, append=centerline[:1]).T)
    spacing = float(seg.mean())
    tangent = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    left = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    lane = centerline + left * cfg.lane_offset_m
    offset = ((pos - lane[index]) * left[index]).sum(1)
    out = [offset / cfg.offset_scale_m]
    for metres in cfg.bearings_m:
        ahead = lane[(index + int(round(metres / max(spacing, 1e-6)))) % n] - pos
        bearing = np.arctan2(ahead[:, 1], ahead[:, 0]) - heading
        out.append(np.arctan2(np.sin(bearing), np.cos(bearing)) / cfg.bearing_scale_rad)
    return np.stack(out, axis=1).astype(np.float32)


def downsample(frames: np.ndarray, cfg: LaneHeadConfig) -> torch.Tensor:
    """(batch, H, W, 3) uint8 -> (batch, 3, h, w) float in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    return nn.functional.interpolate(x, size=(cfg.input_height, cfg.input_width), mode="area")


class LaneNet(nn.Module):
    def __init__(self, outputs: int) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        # 96x160 input -> 3x5 after five stride-2 blocks (MPS pools only exact divisors)
        self.features = nn.Sequential(block(3, 24), block(24, 32), block(32, 48), block(48, 64), block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(), nn.Linear(96 * 15, 128), nn.ReLU(inplace=True), nn.Linear(128, outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class LaneHead:
    """Loaded lane model: frames -> mirrored positive channels for the eye."""

    def __init__(self, path: str | Path, device: str | torch.device | None = None) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = LaneHeadConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved["cfg"].items()})
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.net = LaneNet(self.cfg.outputs)
        self.net.load_state_dict(saved["state"])
        self.net.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
        x = downsample(np.stack(list(frames)), self.cfg).to(self.device)
        return self.net(x).float().cpu().numpy()

    def encode(self, frames: list[np.ndarray] | np.ndarray, width: int) -> np.ndarray:
        """(batch, width) in [0, 1]: (+, -) per output, then unused zeros."""
        pred = np.clip(self.predict(frames), -1.0, 1.0)
        pairs = np.stack([np.maximum(pred, 0.0), np.maximum(-pred, 0.0)], axis=2).reshape(pred.shape[0], -1)
        out = np.zeros((pred.shape[0], width), dtype=np.float32)
        out[:, : min(width, pairs.shape[1])] = pairs[:, :width]
        return out


def capture(track: Path, frames_per_route: int, cfg: LaneHeadConfig, seed: int, args) -> tuple[np.ndarray, np.ndarray]:
    """Rendered frames at noisy poses along one route, with their scaled labels."""
    from train_camera import Stage, build_world

    route_args = argparse.Namespace(**{**vars(args), "track": track})
    scene, traffic, sensor, _ = build_world(route_args, torch.device("cpu"), Stage("lane capture", cfg.capture_traffic, 0.0))
    rng = np.random.default_rng(seed)
    centerline = np.asarray(scene.centerline, dtype=np.float64)[:, :2]
    n = centerline.shape[0]
    tangent = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    left = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    road_heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    actors = scene.fixtures + traffic.actors

    index = rng.integers(0, n, frames_per_route)
    offset = cfg.lane_offset_m + rng.uniform(-cfg.capture_offset_m, cfg.capture_offset_m, frames_per_route)
    pos = centerline[index] + left[index] * offset[:, None]
    heading = road_heading[index] + rng.normal(0.0, cfg.capture_heading_rad, frames_per_route)
    images = np.zeros((frames_per_route, cfg.input_height, cfg.input_width, 3), dtype=np.uint8)
    for k in range(frames_per_route):
        frame = sensor.camera.render(pos[k], float(heading[k]), float(scene.heights[index[k]]), actors)
        small = downsample(frame[None], cfg)[0].permute(1, 2, 0).numpy()
        images[k] = (small * 255.0).astype(np.uint8)
    sensor.release()
    return images, labels_for(centerline, index, pos, heading, cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routes", type=Path, nargs="+", required=True, help="route files or directories of them (training)")
    parser.add_argument("--holdout", type=Path, required=True, help="route for the reported error only, never trained on")
    parser.add_argument("--frames-per-route", type=int, default=2500)
    parser.add_argument("--holdout-frames", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", type=Path, default=Path(LANE_HEAD_WEIGHTS))
    parser.add_argument("--seed", type=int, default=0)
    # world construction (as train_camera's)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--perception-stride", type=int, default=3)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--vehicles", type=int, default=120)
    parser.add_argument("--motorcycles", type=int, default=100)
    parser.add_argument("--pedestrians", type=int, default=40)
    args = parser.parse_args(argv)
    if args.weights is None:
        from perceive import TRAINED_WEIGHTS

        args.weights = TRAINED_WEIGHTS

    cfg = LaneHeadConfig()
    routes: list[Path] = []
    for r in args.routes:
        routes += sorted(p for p in r.glob("*.geojson") if not p.stem.endswith("_buildings")) if r.is_dir() else [r]
    if args.holdout.resolve() in {r.resolve() for r in routes}:
        raise SystemExit(f"{args.holdout} is also a training route")

    started = time.time()
    xs, ys = [], []
    for k, route in enumerate(routes):
        x, y = capture(route, args.frames_per_route, cfg, args.seed + k, args)
        xs.append(x)
        ys.append(y)
        print(f"[capture] {route.stem}: {len(x)} frames ({time.time() - started:.0f}s)", flush=True)
    hx, hy = capture(args.holdout, args.holdout_frames, cfg, args.seed + 1000, args)
    print(f"[capture] holdout {args.holdout.stem}: {len(hx)} frames", flush=True)
    x = torch.from_numpy(np.concatenate(xs)).permute(0, 3, 1, 2)
    y = torch.from_numpy(np.concatenate(ys))
    hx = torch.from_numpy(hx).permute(0, 3, 1, 2)
    hy = torch.from_numpy(hy)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    net = LaneNet(cfg.outputs).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * ((len(x) + args.batch - 1) // args.batch))

    def evaluate(xe: torch.Tensor, ye: torch.Tensor) -> np.ndarray:
        net.eval()
        with torch.inference_mode():
            pred = torch.cat([net(xe[i : i + 256].to(device).float() / 255.0).cpu() for i in range(0, len(xe), 256)])
        net.train()
        return (1.0 - ((pred - ye) ** 2).mean(0) / ye.var(0).clamp(min=1e-9)).numpy()

    for epoch in range(args.epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), args.batch):
            idx = perm[i : i + args.batch]
            xb = x[idx].to(device).float() / 255.0
            # brightness jitter: the sun and the paint are not the same everywhere
            xb = (xb * torch.empty(len(idx), 1, 1, 1, device=device).uniform_(0.75, 1.25)).clamp(0.0, 1.0)
            loss = nn.functional.smooth_l1_loss(net(xb), y[idx].to(device), beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        r2 = evaluate(hx, hy)
        print(f"[epoch {epoch:2d}] loss {total / len(x):.4f}  holdout R^2 " + " ".join(f"{v:.2f}" for v in r2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": {k: v.cpu() for k, v in net.state_dict().items()}, "cfg": asdict(cfg), "holdout_r2": evaluate(hx, hy).tolist(),
         "routes": [str(r) for r in routes], "holdout": str(args.holdout)},
        args.out,
    )
    print(f"[ok] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
