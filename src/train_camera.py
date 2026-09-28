"""Evolve the connectome to drive a Malaysian street from a camera.

The ray trainer in `train.py` evolves a driver that is told where the road
edge is. This one does not tell it anything: every body renders its own view,
a real detector reads that view, and the detections are the whole of what the
brain receives. The road it is learning is the surveyed Malaysian one, the
traffic and events are what that street actually throws at a driver, and the
reward charges under its road law.

    python3 src/calibrate_camera.py --out checkpoints/camera/driver.pt   # first
    python3 src/train_camera.py --track data/tracks/kl.geojson \\
        --weights checkpoints/perception/kl/weights/best.pt --generations 200

Start from a calibrated checkpoint. Evolution strategies over a random
readout of a 166,700-neuron brain found nothing in 24 generations here, as it
found nothing in 795 on Monaco: two directions in a 2,129-neuron output
population are not found by chance. `calibrate_camera.py` fits them by
imitation on this very eye and writes the checkpoint this script refines.

The road is the surveyed one from the first generation. An earlier curriculum
tripled its width so a random driver could survive long enough to be ranked;
with a calibrated start that crutch is not needed, and a driver that learned a
sixteen-metre corridor had to unlearn it. Only what happens on the street is
staged: an empty street first, then traffic, then the events.

Rendering and detecting cost far more than a ray march, so the population is
small and the camera modest. That is the honest trade: not a search over
thousands of bodies, but a slower search over bodies that only see what a
camera sees.
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
from car_env import DONE_NAMES, CarConfig, CarEnv, load_geojson_centerline, street_car_config
from dataset import load_scene
from events import Director, HazardRates
from eye_camera import CameraSensor, SensorConfig, agent_config_for
from law import load_law
from lawreward import LawEnforcer, LawWeights
from perceive import EyeConfig
from roadlaw import control_points_for_circuit, legal_profile_for_circuit
from scene import SceneConfig, Traffic

CHECKPOINT_DIR = Path("checkpoints/camera")


@dataclass(frozen=True)
class Stage:
    """One rung of the curriculum: how busy the street is, how eventful.

    Nothing about the law or the geometry changes between rungs; the road is
    the surveyed one throughout. Only how much is happening on it at once.
    """

    name: str
    traffic_fraction: float
    event_fraction: float
    width_mult: float = 1.0


CURRICULUM = (
    Stage("empty street", 0.0, 0.0),
    Stage("light traffic", 0.35, 0.0),
    Stage("busy street", 0.7, 0.4),
    Stage("real street", 1.0, 1.0),
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
    """The event rates this rung runs at; an empty street stages nothing."""
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
        cfg=scene_cfg,
        rates=rates_for(stage),
    )
    return scene, traffic, sensor, director


def build_env(
    args, scene, device: torch.device, batch: int, width_mult: float = 1.0
) -> tuple[CarEnv, LawEnforcer]:
    """A population on the surveyed circuit, charged under its road law.

    `width_mult` is 1.0 by default: the corridor is the surveyed street. It is
    kept as an override for experiments, and the keep-left charge is scaled
    down with it because keeping left means little on a corridor wider than
    the road it stands for.
    """
    cfg = street_car_config(
        layout="geojson",
        geojson_path=str(args.track),
        max_laps=0.0,  # a street has no finish line; the episode ends by budget
        episode_steps=args.steps,
        max_speed=args.max_speed,
        track_halfwidth=CarConfig.track_halfwidth * width_mult,
        # Start in the lane the law keeps you in, not on the centre line.
        lane_offset_m=args.lane_offset,
    )
    starts = torch.linspace(0.0, 1.0, batch + 1)[:batch]
    env = CarEnv(batch, device, cfg, start_fraction=starts)

    points, proj = load_geojson_centerline(
        args.track, cfg.track_scale, cfg.n_points, cfg.smooth_m, cfg.mirror, return_projection=True
    )
    centerline = points.numpy()
    profile = legal_profile_for_circuit(args.track, centerline, proj)
    control_points = control_points_for_circuit(args.track, proj, load_law(), centerline=centerline)
    weights = LawWeights()
    if width_mult > 1.0:
        weights = replace(weights, wrong_side=weights.wrong_side / (width_mult * width_mult))
    law = LawEnforcer(
        profile,
        control_points,
        env.track.centerline.cpu(),
        device,
        weights=weights,
        dt_s=cfg.dt_s,
    )
    env.attach_law(law)

    # The pace term compares speed with the track's quasi-steady grip limit,
    # which on a street is 119 km/h. A car doing 35 km/h in a 50 zone leaves
    # no way to score well on pace: the population settles for the constant
    # penalty. On a street the reference is the posted limit, a little under
    # it, so obeying the law is what pays best.
    reference = law.limit_mps.clone()
    posted = torch.isfinite(reference)
    if posted.any():
        street = torch.where(posted, reference * args.pace_fraction, env.track.speed_ref)
        env.track.speed_ref = torch.minimum(env.track.speed_ref, street)
    return env, law


def clear_starts(traffic, env, radius_m: float = 14.0) -> int:
    """Take traffic off the start line.

    Bodies placed around a car already standing there means a collision on the
    first step, which teaches nothing about driving. Actors within `radius_m`
    of any start are removed before the episode begins.
    """
    if not traffic.actors:
        return 0
    starts = env.track.centerline[env.start_index].detach().cpu().numpy()
    keep = []
    for actor in traffic.actors:
        gap = float(np.linalg.norm(starts - actor.pos[:2], axis=1).min())
        if gap >= radius_m:
            keep.append(actor)
    removed = len(traffic.actors) - len(keep)
    traffic.actors = keep
    return removed


def obstacle_provider(traffic, fixtures):
    """Positions of everything solid on the road, for the collision test."""

    def provide():
        actors = [a for a in traffic.actors if a.kind != "traffic light"]
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
    step = 0
    with torch.inference_mode():
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
            # The world moves on for everyone: the events are staged around
            # the body furthest along, so the population shares one street.
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


def rank_normalise(fitness: torch.Tensor) -> torch.Tensor:
    """Centred ranks in [-0.5, 0.5]: one outlier moves the mean no more than any other body."""
    n = fitness.numel()
    if n < 2:
        return torch.zeros_like(fitness)
    order = torch.argsort(fitness)
    ranks = torch.empty_like(fitness)
    ranks[order] = torch.arange(n, device=fitness.device, dtype=fitness.dtype)
    return ranks / (n - 1) - 0.5


def agent_for(saved: dict | None, sensor: CameraSensor, substeps: int) -> AgentConfig:
    """The agent interface a checkpoint was evolved against, or the camera default.

    A readout fitted by `calibrate_camera.py` only means something under the
    `readout_norm` it was fitted with; the checkpoint records that, and the
    agent is built to match rather than to this build's defaults.
    """
    base = AgentConfig(substeps=substeps, readout_norm="channel")
    if saved and saved.get("agent_cfg"):
        base = AgentConfig.from_saved(saved["agent_cfg"], substeps=substeps)
    return agent_config_for(sensor, base)


def install_stage(args, scene, traffic, director, sensor, device, stage_index: int):
    """Put the curriculum rung's traffic and events on the street; return env and law."""
    stage = CURRICULUM[stage_index]
    traffic.actors = populate_for(args, scene, stage, SceneConfig()).actors
    director.traffic = traffic
    director.rates = rates_for(stage)
    director.active.clear()
    sensor.traffic = traffic
    env, law = build_env(args, scene, device, args.popsize, stage.width_mult * args.width_mult)
    removed = clear_starts(traffic, env)
    env.attach_traffic(obstacle_provider(traffic, scene.fixtures))
    print(
        f"[world] stage {stage_index} '{stage.name}': road x{stage.width_mult * args.width_mult:.2f}, "
        f"{len(traffic.actors)} road users ({removed} moved off the start line)"
    )
    return env, law, stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", type=Path, default=Path("data/tracks/kl.geojson"))
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--weights", default="checkpoints/perception/kl/weights/best.pt")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "driver.pt")
    parser.add_argument("--best", type=Path, default=CHECKPOINT_DIR / "driver_best.pt")
    parser.add_argument("--log", type=Path, default=CHECKPOINT_DIR / "log.csv")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=16, help="bodies per generation; the mean drives as body 0")
    parser.add_argument("--steps", type=int, default=1200, help="control steps per episode (16 ms each)")
    parser.add_argument("--sigma", type=float, default=0.04, help="search radius around the mean")
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument(
        "--lane-offset",
        type=float,
        default=-1.9,
        help="metres left (+) or right (-) of the centre line the car starts; Malaysia drives on the left",
    )
    parser.add_argument("--stage", type=int, default=0, help="curriculum rung to start on")
    parser.add_argument("--advance-at", type=float, default=0.6, help="share of bodies alive at the end that counts as steady")
    parser.add_argument("--advance-after", type=int, default=3, help="steady generations before the next rung")
    parser.add_argument("--width-mult", type=float, default=1.0, help="corridor width multiplier (1.0 = the surveyed street)")
    parser.add_argument(
        "--revert-tolerance",
        type=float,
        default=25.0,
        help="the mean is put back to its best if it scores this much worse for --revert-after generations",
    )
    parser.add_argument("--revert-after", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--perception-stride", type=int, default=3)
    parser.add_argument(
        "--max-speed",
        type=float,
        default=17.0,
        help="m/s. 17 is 61 km/h: a street car under a 35-60 km/h limit that can still corner",
    )
    parser.add_argument("--pace-fraction", type=float, default=0.92, help="the share of the posted limit the pace term asks for")
    parser.add_argument("--vehicles", type=int, default=120)
    parser.add_argument("--motorcycles", type=int, default=100)
    parser.add_argument("--pedestrians", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--dt-ms", type=float, default=defaults.DT_MS)
    parser.add_argument("--substeps", type=int, default=defaults.SUBSTEPS)
    args = parser.parse_args(argv)
    if args.popsize % 2 == 0:
        # Body 0 is the mean itself; the rest are mirrored pairs.
        args.popsize += 1
        print(f"[es] popsize raised to {args.popsize} so the mean drives as body 0 with mirrored pairs behind it")

    device = pick_device(args.device)
    connectome = load_connectome(args.graph)
    neurons: pd.DataFrame = connectome.neurons

    saved = None
    if args.checkpoint.exists():
        saved = torch.load(args.checkpoint, map_location="cpu")

    stage_index = min(max(args.stage, 0), len(CURRICULUM) - 1)
    if saved is not None and args.stage == 0:
        stage_index = min(int(saved.get("stage", 0)), len(CURRICULUM) - 1)
    scene, traffic, sensor, director = build_world(args, device, CURRICULUM[stage_index])
    print(f"[world] eye {sensor.eye.width} channels, camera {args.width}x{args.height}, detector {args.weights}")

    brain = Brain(
        connectome,
        batch=args.popsize,
        config=LIFConfig(dt_ms=args.dt_ms, adapt_mv=defaults.ADAPT_MV),
        device=device,
        weight_scale=defaults.WEIGHT_SCALE,
    )
    agent_cfg = agent_for(saved, sensor, args.substeps)
    agent = ConnectomeAgent(brain, neurons, agent_cfg)

    env, law, stage = install_stage(args, scene, traffic, director, sensor, device, stage_index)
    print(f"[world] {law.summary()}")

    mu = agent.initial_params().to(device)
    generation0 = 0
    if saved is not None:
        if saved.get("eye_width") != sensor.eye.width:
            raise SystemExit(
                f"checkpoint eye {saved.get('eye_width')} != {sensor.eye.width}: "
                "the driver was evolved against a different view; pass a matching --columns"
            )
        mu = saved["mu"].reshape(-1).to(device)
        if agent.load_readout(saved):
            print("[resume] calibrated readout installed from the checkpoint")
        generation0 = int(saved.get("generation", 0)) + (1 if saved.get("generation") is not None and saved.get("trained") else 0)
        print(f"[resume] generation {generation0}, stage {stage_index}, {mu.numel()} parameters")
    elif agent_cfg.readout_norm == "channel":
        print(
            "[warn] no checkpoint: starting from a random readout. Run calibrate_camera.py first; "
            "ES from here has not learned to steer in any run so far."
        )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.log.exists() and saved is not None:
        try:
            rows = pd.read_csv(args.log).to_dict("records")
        except Exception:
            rows = []
    sigma = float(saved.get("sigma", args.sigma)) if saved is not None else args.sigma
    steady = 0
    best_mu = mu.clone()
    best_mu_fitness = float(saved.get("best_mu_fitness", -float("inf"))) if saved is not None else -float("inf")
    worse_streak = 0
    generator = torch.Generator(device="cpu").manual_seed(args.seed + generation0)

    def save(path: Path, generation: int, mu_fitness: float) -> None:
        torch.save(
            {
                "mu": mu.detach().cpu(),
                "generation": generation,
                "trained": True,
                "eye_width": sensor.eye.width,
                "stage": stage_index,
                "sigma": sigma,
                "best_mu_fitness": best_mu_fitness,
                "mu_fitness": mu_fitness,
                "n_params": agent.n_params,
                "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()},
                "agent_cfg": asdict(agent_cfg),
                "readout": agent.readout_state(),
                "track": str(args.track),
                "weights": args.weights,
                "columns": args.columns,
                "camera": {"width": args.width, "height": args.height},
                "saved_at": time.time(),
            },
            path,
        )

    for generation in range(generation0, generation0 + args.generations):
        started = time.time()
        half = (args.popsize - 1) // 2
        noise = torch.randn(half, mu.numel(), generator=generator).to(device)
        perturb = torch.cat([torch.zeros(1, mu.numel(), device=device), noise, -noise], dim=0)
        params = agent.clamp_params(mu.unsqueeze(0) + sigma * perturb)
        theta = agent.unpack(params)

        result = run_episode(agent, env, sensor, director, theta, args.steps, args.seed + generation)
        fitness = result["fitness"]
        mu_fitness = float(fitness[0])
        survived = result["endings"].get("alive", 0) / max(args.popsize, 1)

        # The mean's own score is the thing being improved; a mean that has
        # drifted somewhere worse for several generations goes back to the
        # best one seen and searches more narrowly from there.
        if mu_fitness > best_mu_fitness:
            best_mu_fitness = mu_fitness
            best_mu = mu.clone()
            worse_streak = 0
            save(args.best, generation, mu_fitness)
        elif mu_fitness < best_mu_fitness - args.revert_tolerance:
            worse_streak += 1
        else:
            worse_streak = 0
        reverted = False
        if worse_streak >= args.revert_after:
            mu = best_mu.clone()
            sigma = max(sigma * 0.5, args.sigma_min)
            worse_streak = 0
            reverted = True
        else:
            pairs = fitness[1:]
            ranked = rank_normalise(pairs)
            gradient = (ranked.unsqueeze(1) * perturb[1:]).sum(dim=0) / max(half, 1)
            mu = agent.clamp_params((mu + args.lr * gradient / sigma).unsqueeze(0))[0]

        elapsed = time.time() - started
        row = {
            "generation": generation,
            "fitness_mean": float(fitness.mean()),
            "fitness_best": float(fitness.max()),
            "fitness_mu": mu_fitness,
            "best_mu_fitness": best_mu_fitness,
            "laps_best": float(result["laps"].max()),
            "laps_mean": float(result["laps"].mean()),
            "laps_mu": float(result["laps"][0]),
            "law_mean": float(result["law"].mean()),
            "speed_mean": float(result["speed"].mean()),
            "steps": result["steps"],
            "seconds": round(elapsed, 1),
            "collisions": result["endings"].get("collision", 0),
            "crashes": result["endings"].get("crash", 0),
            "events": sum(result["events"].values()),
            "sigma": sigma,
            "stage": stage_index,
            "width_mult": stage.width_mult * args.width_mult,
            "survived": round(survived, 3),
            "reverted": reverted,
        }
        rows.append(row)
        print(
            f"gen {generation:4d} mu {mu_fitness:8.2f} (best {best_mu_fitness:8.2f}) "
            f"pop {row['fitness_mean']:8.2f}/{row['fitness_best']:8.2f} "
            f"laps mu {row['laps_mu']:.3f} best {row['laps_best']:.3f} law {row['law_mean']:7.2f} "
            f"coll {row['collisions']:2d} crash {row['crashes']:2d} alive {survived:.2f} "
            f"s{stage_index} sig {sigma:.3f} {row['seconds']:5.1f}s" + (" REVERT" if reverted else "")
        )

        # Promote once the population can survive this rung for a few
        # generations running, not on one lucky episode.
        steady = steady + 1 if survived >= args.advance_at else 0
        if steady >= args.advance_after and stage_index + 1 < len(CURRICULUM):
            stage_index += 1
            steady = 0
            env, law, stage = install_stage(args, scene, traffic, director, sensor, device, stage_index)
            # A new rung is a new objective: the best score on the old one no
            # longer says anything about this one.
            best_mu_fitness = -float("inf")
            worse_streak = 0

        save(args.checkpoint, generation, mu_fitness)
        pd.DataFrame(rows).to_csv(args.log, index=False)

    sensor.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
