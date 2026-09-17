"""Evolve the connectome to drive a Malaysian street from its camera.

The ray trainer in `train.py` evolves a driver that is told where the road
edge is. This one does not tell it anything: every body renders its own view,
a real detector reads that view, and the detections are the whole of what the
brain receives. The road it is learning is a surveyed Malaysian one, with the
traffic and the events a street actually throws at a driver, and the reward
charges it under the road law.

  python3 src/train_camera.py --track data/tracks/kl.geojson \\
      --weights checkpoints/perception/kl/weights/best.pt --generations 200

Rendering and detecting cost far more than a ray march, so the population is
small and the camera modest. That is the honest trade: this is not a search
over thousands of bodies, it is a slower search over bodies that can only see
what a camera sees.

Fitness is the environment's own reward - progress, pace, staying on the road -
plus the law's charges and whatever the events cost a driver who does not
react. A body that drives into a pedestrian ends its episode, and the crash
charge is what evolution feels.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device
from camera import CameraConfig
from car_env import DONE_ALIVE, DONE_COLLIDE, DONE_NAMES, CarConfig, CarEnv, load_geojson_centerline
from dataset import load_scene
from events import Director, HazardRates
from eye_camera import CameraSensor, SensorConfig
from law import load_law
from lawreward import LawEnforcer
from perceive import EyeConfig
from roadlaw import control_points_for_circuit, legal_profile_for_circuit
from scene import SceneConfig, Traffic
from train import rank_normalise

CHECKPOINT_DIR = Path("checkpoints/camera")


def build_world(args, device: torch.device):
    """The circuit, its law, its traffic and the sensor that looks at it."""
    scene_cfg = SceneConfig()
    scene = load_scene(args.track, scene_cfg)
    traffic = Traffic.populate(
        scene.centerline,
        scene.heights,
        scene.profile,
        scene.control_points,
        vehicles=args.vehicles,
        motorcycles=args.motorcycles,
        pedestrians=args.pedestrians,
        seed=args.seed,
        cfg=scene_cfg,
    )
    sensor = CameraSensor(
        scene,
        traffic,
        CameraConfig(width=args.width, height=args.height),
        EyeConfig(columns=args.columns),
        SensorConfig(stride=args.perception_stride, weights=args.weights, device=args.detector_device),
        scene_cfg,
    )
    director = Director(
        scene.centerline, scene.heights, scene.profile, traffic, seed=args.seed, rates=HazardRates(), cfg=scene_cfg
    )
    return scene, traffic, sensor, director


def build_env(args, scene, device: torch.device, batch: int) -> tuple[CarEnv, LawEnforcer]:
    """A population on the surveyed circuit, charged under its road law."""
    cfg = CarConfig(
        layout="geojson",
        geojson_path=str(args.track),
        max_laps=0.0,  # a street has no finish line; the episode ends by budget
        episode_steps=args.steps,
        max_speed=args.max_speed,
    )
    starts = torch.linspace(0.0, 1.0, batch + 1)[:batch]
    env = CarEnv(batch, device, cfg, start_fraction=starts)

    points, proj = load_geojson_centerline(
        args.track, cfg.track_scale, cfg.n_points, cfg.smooth_m, cfg.mirror, return_projection=True
    )
    centerline = points.numpy()
    profile = legal_profile_for_circuit(args.track, centerline, proj)
    control_points = control_points_for_circuit(args.track, proj, load_law(), centerline=centerline)
    law = LawEnforcer(profile, control_points, env.track.centerline.cpu().to(device), device)
    env.attach_law(law)
    return env, law


def obstacle_provider(traffic, fixtures):
    """Positions and reach of everything solid, for the collision test."""

    def provide():
        actors = [a for a in traffic.actors if a.kind != "traffic light"] + []
        if not actors:
            return None, None
        positions = np.array([[a.pos[0], a.pos[1]] for a in actors], dtype=np.float32)
        # Half the object's width, plus a little, is how close counts as a hit.
        radii = np.array([max(float(a.size[1]) * 0.5, 0.3) + 0.4 for a in actors], dtype=np.float32)
        return positions, radii

    return provide


def run_episode(
    agent: ConnectomeAgent,
    env: CarEnv,
    sensor: CameraSensor,
    director: Director,
    theta: dict[str, torch.Tensor],
    steps: int,
    seed: int,
) -> dict:
    """One episode for the whole population; returns fitness and what happened."""
    agent.seed(seed)
    env.attach_sensor(sensor)
    obs = env.reset()
    agent.reset(batch=env.batch)

    fitness = torch.zeros(env.batch, device=env.device)
    alive = torch.ones(env.batch, dtype=torch.bool, device=env.device)
    law_charge = torch.zeros(env.batch, device=env.device)
    events_seen: dict[str, int] = {}
    dt = env.cfg.dt_s

    for step in range(steps):
        action = agent.act(obs, theta)
        obs, reward, done = env.step(action)
        fitness = fitness + reward * alive.to(reward.dtype)
        terms = env.last_terms
        if "law_total" in terms:
            law_charge = law_charge + terms["law_total"] * alive.to(reward.dtype)
        alive = alive & ~done
        if not bool(alive.any()):
            break
        # The world moves on for everyone: the events are staged around the
        # body that has driven furthest, so the population shares one street.
        lead = int(torch.argmax(env.laps).item())
        running = director.step(
            dt,
            float(env.last_progress[lead]),
            env.pos[lead].detach().cpu().numpy(),
            float(env.speed[lead]),
        )
        for kind in running:
            events_seen[kind] = events_seen.get(kind, 0) + 1

    reasons = env.done_reason.detach().cpu()
    counts = {DONE_NAMES[int(r)]: int((reasons == r).sum()) for r in sorted(set(reasons.tolist()))}
    return {
        "fitness": fitness.detach(),
        "laps": env.laps.detach().cpu(),
        "speed": env.speed.detach().cpu(),
        "law": law_charge.detach().cpu(),
        "endings": counts,
        "events": events_seen,
        "steps": step + 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=Path("data/tracks/kl.geojson"))
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--weights", default="checkpoints/perception/kl/weights/best.pt")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "driver.pt")
    parser.add_argument("--log", type=Path, default=CHECKPOINT_DIR / "log.csv")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=16, help="bodies per generation; each renders its own view")
    parser.add_argument("--steps", type=int, default=1200, help="control steps per episode")
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--perception-stride", type=int, default=3)
    parser.add_argument("--max-speed", type=float, default=33.0, help="m/s; a street car, not an F1 car")
    parser.add_argument("--vehicles", type=int, default=120)
    parser.add_argument("--motorcycles", type=int, default=100)
    parser.add_argument("--pedestrians", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--dt-ms", type=float, default=defaults.DT_MS)
    parser.add_argument("--substeps", type=int, default=defaults.SUBSTEPS)
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    connectome = load_connectome(args.graph)
    neurons: pd.DataFrame = connectome.neurons

    scene, traffic, sensor, director = build_world(args, device)
    print(f"[world] eye {sensor.eye.width} channels, camera {args.width}x{args.height}, detector {args.weights}")

    brain = Brain(
        connectome,
        batch=args.popsize,
        config=LIFConfig(dt_ms=args.dt_ms, adapt_mv=defaults.ADAPT_MV),
        device=device,
        weight_scale=defaults.WEIGHT_SCALE,
    )
    agent_cfg = AgentConfig(n_rays=sensor.eye.width, substeps=args.substeps)
    agent = ConnectomeAgent(brain, neurons, agent_cfg)

    env, law = build_env(args, scene, device, args.popsize)
    env.attach_traffic(obstacle_provider(traffic, scene.fixtures))
    print(f"[world] {law.summary()}")

    mu = agent.initial_params().to(device)
    if args.checkpoint.exists():
        saved = torch.load(args.checkpoint, map_location=device)
        if saved.get("eye_width") == sensor.eye.width:
            mu = saved["mu"].to(device)
            print(f"[resume] generation {saved.get('generation', 0)}")
        else:
            print(f"[resume] checkpoint eye {saved.get('eye_width')} != {sensor.eye.width}; starting fresh")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    sigma = args.sigma
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for generation in range(args.generations):
        started = time.time()
        half = max(args.popsize // 2, 1)
        noise = torch.randn(half, mu.numel(), generator=generator).to(device)
        perturb = torch.cat([noise, -noise], dim=0)[: args.popsize]  # mirrored pairs
        params = mu.unsqueeze(0) + sigma * perturb
        theta = agent.unpack(params)

        result = run_episode(agent, env, sensor, director, theta, args.steps, args.seed + generation)
        fitness = result["fitness"]
        ranked = rank_normalise(fitness)
        gradient = (ranked.unsqueeze(1) * perturb).mean(dim=0)
        mu = mu + args.lr * gradient

        elapsed = time.time() - started
        row = {
            "generation": generation,
            "fitness_mean": float(fitness.mean()),
            "fitness_best": float(fitness.max()),
            "laps_best": float(result["laps"].max()),
            "laps_mean": float(result["laps"].mean()),
            "law_mean": float(result["law"].mean()),
            "speed_mean": float(result["speed"].mean()),
            "steps": result["steps"],
            "seconds": round(elapsed, 1),
            "collisions": result["endings"].get("collision", 0),
            "crashes": result["endings"].get("crash", 0),
            "events": sum(result["events"].values()),
            "sigma": sigma,
        }
        rows.append(row)
        print(
            f"gen {generation:4d} fit {row['fitness_mean']:9.2f}/{row['fitness_best']:9.2f} "
            f"laps {row['laps_best']:.3f} law {row['law_mean']:7.2f} "
            f"coll {row['collisions']:2d} crash {row['crashes']:2d} "
            f"{row['seconds']:5.1f}s"
        )

        torch.save(
            {
                "mu": mu.cpu(),
                "generation": generation,
                "eye_width": sensor.eye.width,
                "agent_cfg": asdict(agent_cfg),
                "track": str(args.track),
                "weights": args.weights,
            },
            args.checkpoint,
        )
        pd.DataFrame(rows).to_csv(args.log, index=False)

    sensor.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
