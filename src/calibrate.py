"""Calibrate the motor readout by imitation, then hand the result to ES.

Evolution strategies over a random readout of a 166,700-neuron spiking brain
went 795 generations without learning to steer, because (a) the eye encoding
hid the left/right difference (fixed in `agent.py`, `eye_encoding="road"`)
and (b) a 144-parameter random search cannot find the two directions in a
2,129-neuron output population that carry steering and braking. This script
finds those directions directly:

  1. a scripted lidar teacher (`teacher.py`) drives `--cars` cars round Monaco
     while the brain watches through its eyes; every control step records the
     output population (descending + motor neurons, common mode removed);
  2. ridge regression from that population to the teacher's steering and
     pedal pre-activations gives two readout vectors; they become channels 0
     and 1 of the agent's projection (the other channels stay random, so ES
     can still recombine), channel statistics are calibrated, and `w_out`,
     `g_out`, `b_out` are set to reproduce the fit exactly;
  3. DAgger: the brain then drives itself (mixing in the teacher at `--beta`
     for the middle rounds), the teacher labels what it would have done, the
     data are pooled and the fit repeated, so the readout is trained on the
     states the brain actually visits;
  4. the brain is evaluated closed-loop and the interface is written as an ES
     checkpoint (`--out`) for `train.py` to refine.

  python3 src/calibrate.py --out checkpoints/es.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device, synchronize
from car_env import DONE_NAMES, CarEnv, Track, build_centerline, monaco_config
from teacher import LinearTeacher
from train import CURRICULUM

# Sensory interface the readout is calibrated for; ES starts from these.
SENSORY = {"ray_gain": 1.0, "bias_hz": 0.05, "speed_gain": 0.5, "loom_gain": 0.5}
PRE_CLIP = 0.9  # teacher actions are clipped here before atanh


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Least squares with an L2 penalty on the weights (not the intercept). Returns (W (D, K), b (K,))."""
    mean_x, mean_y = x.mean(0), y.mean(0)
    xc, yc = x - mean_x, y - mean_y
    gram = xc.T @ xc
    gram.diagonal().add_(lam)
    w = torch.linalg.solve(gram, xc.T @ yc)
    return w, mean_y - mean_x @ w


def r_squared(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - ((pred - y) ** 2).mean(0) / y.var(0).clamp(min=1e-9)


def fit_readout(x: torch.Tensor, y: torch.Tensor, car: torch.Tensor, lams: tuple[float, ...]) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Ridge with the penalty picked by held-out cars (odd car indices)."""
    train, test = car % 2 == 0, car % 2 == 1
    best = None
    for lam in lams:
        w, b = ridge(x[train], y[train], lam)
        r2 = r_squared(x[test] @ w + b, y[test])
        if best is None or float(r2.mean()) > best[0]:
            best = (float(r2.mean()), lam, r2)
    _, lam, r2 = best
    w, b = ridge(x, y, lam)
    return w, b, lam, r2


class Collector:
    """Rolls the brain and the car together; keeps the output population and teacher labels."""

    def __init__(self, agent: ConnectomeAgent, env: CarEnv, teacher: LinearTeacher, theta: dict[str, torch.Tensor], target_smooth: float = 1.0) -> None:
        self.agent, self.env, self.teacher, self.theta = agent, env, teacher, theta
        self.target_smooth = target_smooth

    def teacher_action(self, obs: torch.Tensor) -> torch.Tensor:
        n = self.agent.cfg.n_rays
        return self.teacher.act(self.agent.proximity(obs[:, :n]), obs[:, n])

    def run(self, steps: int, beta: float, seed: int, record: bool = True) -> dict:
        agent, env = self.agent, self.env
        agent.seed(seed)
        obs = env.reset()
        agent.reset()
        batch = env.batch
        alive = torch.ones(batch, dtype=torch.bool, device=env.device)
        first_end = torch.full((batch,), -1, dtype=torch.long)
        reason = torch.zeros(batch, dtype=torch.long)
        speed_sum = torch.zeros(batch, device=env.device)
        xs, ys, cars = [], [], []
        car_index = torch.arange(batch)
        smooth = self.target_smooth
        label = None
        with torch.inference_mode():
            for step in range(steps):
                target = self.teacher_action(obs)
                label = target if label is None or smooth >= 1.0 else (1.0 - smooth) * label + smooth * target
                student = agent.act(obs, self.theta)
                action = beta * target + (1.0 - beta) * student
                if record:
                    signal = agent.motor_state - agent.motor_state.mean(dim=1, keepdim=True)
                    keep = alive.cpu()
                    xs.append(signal[alive].half().cpu())
                    ys.append(label[alive].cpu())
                    cars.append(car_index[keep])
                obs, _, done = env.step(action)
                speed_sum += env.speed * alive
                ended = done & alive
                if bool(ended.any()):
                    idx = ended.cpu()
                    first_end[idx] = step + 1
                    reason[idx] = env.done_reason.cpu()[idx]
                alive &= ~done
                if not bool(alive.any()):
                    break
        synchronize(env.device)
        steps_alive = torch.where(first_end > 0, first_end, torch.full_like(first_end, steps))
        return {
            "x": torch.cat(xs) if xs else None,
            "y": torch.cat(ys) if ys else None,
            "car": torch.cat(cars) if cars else None,
            "laps": env.laps.cpu(),
            "steps_alive": steps_alive,
            "reason": reason,
            "speed_kmh": (speed_sum.cpu() / steps_alive.float() * 3.6),
        }


def describe(name: str, result: dict) -> str:
    ended = [DONE_NAMES[int(r)] if r else "time" for r in result["reason"]]
    counts = {k: ended.count(k) for k in sorted(set(ended))}
    laps = result["laps"].numpy()
    return (
        f"{name}: laps min {laps.min():.3f} mean {laps.mean():.3f} max {laps.max():.3f} | "
        f"alive {result['steps_alive'].float().mean():.0f} steps | {result['speed_kmh'].mean():.0f} km/h | {counts}"
    )


def install_fit(
    agent: ConnectomeAgent, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, theta_row: torch.Tensor, boost: tuple[float, float] = (1.0, 1.0)
) -> tuple[torch.Tensor, dict]:
    """Put the ridge directions into channels 0/1, calibrate channel stats, set w_out/g_out/b_out to reproduce the fit."""
    cfg = agent.cfg
    generator = torch.Generator().manual_seed(cfg.projection_seed)
    projection = torch.randn(agent.n_readout, cfg.readout_dim, generator=generator) / np.sqrt(agent.n_readout)
    norms = w.norm(dim=0).clamp(min=1e-9)
    projection[:, :2] = w / norms
    mixed = x @ projection
    mean, std = mixed.mean(0), mixed.std(0).clamp(min=1e-6)
    agent.set_readout(projection, mean, std)
    # pre_k = ||w_k|| * (mixed_k) + b_k = ||w_k|| * std_k * z_k + (||w_k|| * mean_k + b_k)
    boost_t = torch.tensor(boost, dtype=torch.float32)
    g_out = norms * std[:2] * boost_t
    b_out = (norms * mean[:2] + b) * boost_t
    shapes = agent.param_shapes
    params = theta_row.clone()
    offset = 0
    notes = {}
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        if name == "w_out":
            block = torch.zeros(cfg.readout_dim, 2)
            block[0, 0] = 1.0
            block[1, 1] = 1.0
            params[offset : offset + size] = block.reshape(-1)
        elif name == "g_out":
            params[offset : offset + size] = g_out
            notes["g_out"] = g_out.tolist()
        elif name == "b_out":
            params[offset : offset + size] = b_out
            notes["b_out"] = b_out.tolist()
        offset += size
    clamped = agent.clamp_params(params.unsqueeze(0))[0]
    if not torch.allclose(clamped, params):
        notes["clamped"] = True
    return clamped, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--geojson", default="data/tracks/monaco.geojson")
    parser.add_argument("--cars", type=int, default=12, help="cars (brain bodies) driven per round, spread round the lap")
    parser.add_argument("--steps", type=int, default=4000, help="control steps per collection round")
    parser.add_argument("--rounds", type=int, default=3, help="DAgger rounds after the teacher-driven one")
    parser.add_argument("--beta", type=float, default=0.5, help="teacher share of the action in the middle DAgger rounds")
    parser.add_argument("--eval-steps", type=int, default=6000)
    parser.add_argument("--lams", default="3,10,30,100,300,1000")
    parser.add_argument("--motor-tau", type=float, default=None, help="override AgentConfig.motor_tau (readout low-pass)")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="scale the teacher's pedal intercept: <1 drives slower, leaving margin for imitation error")
    parser.add_argument("--steer-boost", type=float, default=1.0, help="multiply the fitted steering gain: >1 corrects faster than the teacher")
    parser.add_argument("--pedal-boost", type=float, default=1.0, help="multiply the fitted pedal gain")
    parser.add_argument("--target-smooth", type=float, default=1.0, help="EMA factor on the teacher's actions used as targets (1 = raw); <1 drops jitter the brain cannot follow")
    parser.add_argument("--teacher", type=Path, default=None, help="teacher weights JSON (default data/teacher_linear.json)")
    parser.add_argument("--stage", type=int, default=len(CURRICULUM) - 1, help="curriculum stage the ES checkpoint starts at")
    parser.add_argument("--precision", default=None, choices=("fp16", "fp32"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="ES checkpoint to write (refuses to overwrite without --force)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out is not None and args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite (archive it first)")

    device = pick_device(args.device)
    connectome = load_connectome(args.graph)
    brain = Brain(
        connectome,
        batch=args.cars,
        config=LIFConfig(dt_ms=defaults.DT_MS, adapt_mv=defaults.ADAPT_MV),
        device=device,
        weight_scale=defaults.WEIGHT_SCALE,
        precision=args.precision,
    )
    overrides = {"motor_tau": args.motor_tau} if args.motor_tau is not None else {}
    agent_cfg = AgentConfig(substeps=defaults.SUBSTEPS, readout_norm="channel", **overrides)
    agent = ConnectomeAgent(brain, connectome.neurons, agent_cfg)
    dt_s = defaults.control_dt_s(defaults.DT_MS, defaults.SUBSTEPS)
    car_cfg = monaco_config(dt_s)
    if car_cfg.max_range != agent_cfg.eye_range_m or car_cfg.fov_deg != agent_cfg.eye_fov_deg or car_cfg.n_rays != agent_cfg.n_rays:
        raise SystemExit("eye geometry in AgentConfig must match CarConfig (range, fov, rays)")
    car_cfg = type(car_cfg)(**{**asdict(car_cfg), "geojson_path": args.geojson})
    track = Track(build_centerline(car_cfg), car_cfg, device)
    env = CarEnv(args.cars, device, car_cfg, track=track, start_fraction=[i / args.cars for i in range(args.cars)])
    teacher = LinearTeacher(path=args.teacher) if args.teacher else LinearTeacher()
    if args.speed_scale != 1.0:
        scaled = teacher.params.clone()
        scaled[4] *= args.speed_scale
        teacher = LinearTeacher(scaled)

    params = agent.initial_params()
    theta_dict = agent.unpack(params.unsqueeze(0))
    offset = 0
    for name, shape in agent.param_shapes.items():
        size = int(np.prod(shape))
        if name in SENSORY:
            params[offset : offset + size] = SENSORY[name]
        offset += size
    theta = agent.unpack(params.unsqueeze(0).repeat(args.cars, 1).to(device))
    collector = Collector(agent, env, teacher, theta, target_smooth=args.target_smooth)
    lams = tuple(float(v) for v in args.lams.split(","))
    print(
        f"{connectome.n:,} neurons, readout over {agent.n_readout} cells -> {agent_cfg.readout_dim} channels; "
        f"{args.cars} cars x {args.steps} steps per round on {track.length_m:,.0f} m Monaco; brain {brain.precision} {'metal' if brain.uses_metal else 'torch'}"
    )

    pooled_x, pooled_y, pooled_car = [], [], []
    started = time.time()
    for round_index in range(args.rounds + 1):
        beta = 1.0 if round_index == 0 else (args.beta if round_index < args.rounds else 0.0)
        result = collector.run(args.steps, beta, seed=args.seed + round_index)
        print(f"[round {round_index}] beta {beta:.2f} " + describe("drive", result) + f" ({time.time() - started:.0f}s)")
        pooled_x.append(result["x"])
        pooled_y.append(result["y"])
        pooled_car.append(result["car"])
        x = torch.cat(pooled_x).float()
        y = torch.atanh(torch.cat(pooled_y).clamp(-PRE_CLIP, PRE_CLIP))
        car = torch.cat(pooled_car)
        w, b, lam, r2 = fit_readout(x, y, car, lams)
        params, notes = install_fit(agent, x, w, b, params, boost=(args.steer_boost, args.pedal_boost))
        theta = agent.unpack(params.unsqueeze(0).repeat(args.cars, 1).to(device))
        collector.theta = theta
        print(f"[round {round_index}] fit on {x.shape[0]:,} samples, lambda {lam:g}: held-out R2 steer {float(r2[0]):.3f} pedal {float(r2[1]):.3f}; {notes}")

    final = collector.run(args.eval_steps, 0.0, seed=args.seed + 100, record=False)
    print("[final] " + describe(f"brain alone, {args.eval_steps} steps", final))

    if args.out is None:
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage_cfg = type(car_cfg)(**{**asdict(car_cfg), "track_halfwidth": car_cfg.track_halfwidth * CURRICULUM[args.stage][0]})
    torch.save(
        {
            "mu": params.unsqueeze(0).cpu(),
            "momentum": torch.zeros(1, agent.n_params),
            "islands": 1,
            "generation": 0,
            "stage": args.stage,
            "best_eval": -float("inf"),
            "sigma": 0.05,
            "recent_laps": [],
            "n_params": agent.n_params,
            "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()},
            "agent_cfg": asdict(agent_cfg),
            "readout": agent.readout_state(),
            "car_cfg": asdict(stage_cfg),
            "layout": "monaco",
            "starts": defaults.MONACO_STARTS,
            "precision": brain.precision,
            "curriculum": [list(c) for c in CURRICULUM],
            "saved_at": time.time(),
            "calibration": {
                "teacher": teacher.params.tolist(),
                "speed_scale": args.speed_scale,
                "steer_boost": args.steer_boost,
                "pedal_boost": args.pedal_boost,
                "r2_steer": float(r2[0]),
                "r2_pedal": float(r2[1]),
                "final_laps": final["laps"].tolist(),
                "final_steps_alive": final["steps_alive"].tolist(),
                "rounds": args.rounds,
                "steps": args.steps,
            },
            "args": {k: str(v) for k, v in vars(args).items()},
        },
        args.out,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
