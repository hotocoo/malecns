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
from dataclasses import asdict, dataclass, replace
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
from lawreward import LawEnforcer, LawWeights
from perceive import EyeConfig
from roadlaw import control_points_for_circuit, legal_profile_for_circuit
from scene import SceneConfig, Traffic
from train import rank_normalise

CHECKPOINT_DIR = Path("checkpoints/camera")


@dataclass(frozen=True)
class Stage:
    """One rung of the curriculum: how wide the road, how busy, how eventful.

    A connectome that cannot yet hold a heading has nothing to learn from a
    motorcycle filtering past it: every body ends the same way and the search
    cannot tell them apart. So the road starts wide and empty, and traffic,
    then events, arrive once the population can survive what is already there.
    Nothing about the law or the geometry changes between rungs - only how much
    is happening at once.
    """

    name: str
    width_mult: float
    traffic_fraction: float
    event_fraction: float


CURRICULUM = (
    Stage("empty road", 3.0, 0.0, 0.0),
    Stage("wide with traffic", 2.2, 0.35, 0.0),
    Stage("narrowing", 1.6, 0.7, 0.4),
    Stage("real street", 1.0, 1.0, 1.0),
)


def populate_for(args, scene, stage: Stage, scene_cfg: SceneConfig) -> Traffic:
    """The traffic this rung of the curriculum puts on the road."""
    share = stage.traffic_fraction
    return Traffic.populate(
        scene.centerline,
        scene.heights,
        scene.profile,
        scene.control_points,
        vehicles=int(args.vehicles * share),
        motorcycles=int(args.motorcycles * share),
        pedestrians=int(args.pedestrians * share),
        seed=args.seed,
        cfg=scene_cfg,
    )


def rates_for(stage: Stage) -> HazardRates:
    """The event rates this rung runs at; an empty road stages nothing."""
    base = HazardRates()
    return HazardRates(per_km={k: v * stage.event_fraction for k, v in base.per_km.items()})


def build_world(args, device: torch.device, stage: Stage | None = None):
    """The circuit, its law, its traffic and the sensor that looks at it."""
    stage = stage or CURRICULUM[0]
    scene_cfg = SceneConfig()
    scene = load_scene(args.track, scene_cfg)
    traffic = populate_for(args, scene, stage, scene_cfg)
    sensor = CameraSensor(
        scene,
        traffic,
        CameraConfig(width=args.width, height=args.height),
        EyeConfig(columns=args.columns),
        SensorConfig(stride=args.perception_stride, weights=args.weights, device=args.detector_device),
        scene_cfg,
    )
    director = Director(
        scene.centerline,
        scene.heights,
        scene.profile,
        traffic,
        seed=args.seed,
        rates=rates_for(stage),
        cfg=scene_cfg,
    )
    return scene, traffic, sensor, director


def build_env(
    args, scene, device: torch.device, batch: int, width_mult: float = 1.0
) -> tuple[CarEnv, LawEnforcer]:
    """A population on the surveyed circuit, charged under its road law.

    `width_mult` widens the drivable corridor for the early generations. An
    untrained connectome steers close to randomly and leaves a nine-metre
    street within a second, so every body ends the same way and the search has
    nothing to rank. Starting wide lets the first useful behaviour - hold a
    heading, follow a bend - survive long enough to be selected, and the road
    narrows back to its surveyed width as the population learns to stay on it.
    The law, the traffic and the events do not change: only the margin does.
    """
    cfg = CarConfig(
        layout="geojson",
        geojson_path=str(args.track),
        max_laps=0.0,  # a street has no finish line; the episode ends by budget
        episode_steps=args.steps,
        max_speed=args.max_speed,
        track_halfwidth=CarConfig.track_halfwidth * width_mult,
    )
    starts = torch.linspace(0.0, 1.0, batch + 1)[:batch]
    env = CarEnv(batch, device, cfg, start_fraction=starts)

    points, proj = load_geojson_centerline(
        args.track, cfg.track_scale, cfg.n_points, cfg.smooth_m, cfg.mirror, return_projection=True
    )
    centerline = points.numpy()
    profile = legal_profile_for_circuit(args.track, centerline, proj)
    control_points = control_points_for_circuit(args.track, proj, load_law(), centerline=centerline)
    # Keeping left means nothing on a corridor three times its real width: the
    # car can sit sixteen metres from the centre and still be on a road that
    # does not exist. The side rule fades in as the road narrows to the
    # surveyed one; the speed limit applies throughout, because it is about the
    # car rather than the corridor.
    weights = LawWeights()
    if width_mult > 1.0:
        weights = replace(weights, wrong_side=weights.wrong_side / (width_mult * width_mult))
    law = LawEnforcer(
        profile,
        control_points,
        env.track.centerline.cpu().to(device),
        device,
        weights=weights,
        dt_s=cfg.dt_s,
    )
    env.attach_law(law)

    # The pace term references a quasi-steady racing speed, which on a city
    # street is 119 km/h. Asking for that while the law charges for exceeding
    # 35 km/h leaves no speed that scores well, and the population sits at a
    # constant penalty whatever it does. On a street the reference is the
    # posted limit, a little under it so obeying the law is what pays best.
    reference = law.limit_mps.clone()
    posted = torch.isfinite(reference)
    if posted.any():
        street = torch.where(posted, reference * args.pace_fraction, env.track.speed_ref)
        env.track.speed_ref = torch.minimum(env.track.speed_ref, street)
    return env, law


def clear_starts(traffic, env, radius_m: float = 14.0) -> int:
    """Take traffic off the start line.

    Bodies are placed around the lap; a vehicle already standing there means a
    collision on the first step, which teaches nothing about driving. Actors
    within `radius_m` of any start are removed before the episode begins.
    """
    if not traffic.actors:
        return 0
    starts = env.track.centerline[env.start_index].detach().cpu().numpy()
    keep = []
    for actor in traffic.actors:
        gap = float(np.linalg.norm(starts - actor.pos[:2], axis=1).min())
        if gap > radius_m:
            keep.append(actor)
    removed = len(traffic.actors) - len(keep)
    traffic.actors = keep
    return removed


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
    parser.add_argument("--stage", type=int, default=0, help="curriculum rung to start on")
    parser.add_argument("--advance-at", type=float, default=0.6, help="survival fraction that promotes a rung")
    parser.add_argument("--advance-after", type=int, default=3, help="consecutive generations at that survival")
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--perception-stride", type=int, default=3)
    parser.add_argument(
        "--max-speed",
        type=float,
        default=17.0,
        help=(
            "m/s. A street car, not an F1 car: 17 m/s is 61 km/h, so a 35 km/h "
            "road can be exceeded but not by so much that the speeding charge "
            "swamps every other term, and the car can still corner."
        ),
    )
    parser.add_argument(
        "--pace-fraction",
        type=float,
        default=0.92,
        help="the share of the posted limit the pace term asks for",
    )
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

    scene, traffic, sensor, director = build_world(args, device, CURRICULUM[min(max(args.stage, 0), len(CURRICULUM) - 1)])
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

    stage_index = min(max(args.stage, 0), len(CURRICULUM) - 1)
    stage = CURRICULUM[stage_index]
    env, law = build_env(args, scene, device, args.popsize, stage.width_mult)
    removed = clear_starts(traffic, env)
    env.attach_traffic(obstacle_provider(traffic, scene.fixtures))
    print(f"[world] stage {stage_index} '{stage.name}': road x{stage.width_mult:.2f}, "
          f"{len(traffic.actors)} road users ({removed} moved off the start line)")
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
    steady = 0
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for generation in range(args.generations):
        started = time.time()
        half = max(args.popsize // 2, 1)
        noise = torch.randn(half, mu.numel(), generator=generator).to(device)
        perturb = torch.cat([noise, -noise], dim=0)[: args.popsize]  # mirrored pairs
        params = mu.unsqueeze(0) + sigma * perturb
        theta = agent.unpack(params)

        result = run_episode(agent, env, sensor, director, theta, args.steps, args.seed + generation)
        survived = result["endings"].get("alive", 0) / max(args.popsize, 1)
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
            "stage": stage_index,
            "width_mult": stage.width_mult,
            "survived": round(survived, 3),
        }
        rows.append(row)
        print(
            f"gen {generation:4d} fit {row['fitness_mean']:9.2f}/{row['fitness_best']:9.2f} "
            f"laps {row['laps_best']:.3f} law {row['law_mean']:7.2f} "
            f"coll {row['collisions']:2d} crash {row['crashes']:2d} "
            f"alive {survived:.2f} s{stage_index} {row['seconds']:5.1f}s"
        )

        # Promote once the population can survive this rung for a few
        # generations running, not on one lucky episode.
        steady = steady + 1 if survived >= args.advance_at else 0
        if steady >= args.advance_after and stage_index + 1 < len(CURRICULUM):
            stage_index += 1
            stage = CURRICULUM[stage_index]
            steady = 0
            traffic.actors = populate_for(args, scene, stage, SceneConfig()).actors
            director.traffic = traffic
            director.rates = rates_for(stage)
            director.active.clear()
            sensor.traffic = traffic
            env, law = build_env(args, scene, device, args.popsize, stage.width_mult)
            clear_starts(traffic, env)
            env.attach_traffic(obstacle_provider(traffic, scene.fixtures))
            print(f"         -> stage {stage_index} '{stage.name}': road x{stage.width_mult:.2f}, "
                  f"{len(traffic.actors)} road users")

        torch.save(
            {
                "mu": mu.cpu(),
                "generation": generation,
                "eye_width": sensor.eye.width,
                "stage": stage_index,
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
