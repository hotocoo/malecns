"""Calibrate the camera driver's motor readout by imitation, then hand it to ES.

`calibrate.py` did this for the Monaco lidar driver and it is what made that
driver drive: evolution strategies over a random readout of the 166,700-neuron
brain went 795 generations without steering, and the camera driver went 24 the
same way. Two directions in a 2,129-neuron output population that carry
steering and braking are not found by chance; they are fitted.

Here the brain watches through the camera eye (`eye_camera.CameraSensor`):
its own render of the surveyed Malaysian street, a real detector over it, the
detections and the road profile as its only senses. Nothing about the road's
geometry reaches the brain. It does reach the teacher, which is the point of a
teacher: a scripted driver that follows the lane from the true state and holds
a speed under the posted limit.

1. The teacher drives `--cars` cars along the street while the brain watches;
   every control step records the output population and the teacher's action.
2. Ridge regression from the population to the teacher's steering and pedal
   pre-activations gives two readout vectors; they become channels 0 and 1 of
   the agent's projection, the channel statistics are calibrated, and
   `w_out`, `g_out`, `b_out` are set to reproduce the fit exactly.
3. DAgger: the brain drives itself (mixing in the teacher at `--beta` in the
   middle rounds), the teacher labels what it would have done, the data is
   pooled and the fit repeated, so the readout is trained on the views the
   brain actually reaches.
4. The brain is evaluated alone and the interface written to the ES checkpoint
   `train_camera.py` starts from.

    python3 src/calibrate_camera.py --out checkpoints/camera/driver.pt --force
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import defaults
from agent import ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device, synchronize
from calibrate import PRE_CLIP, fit_readout, install_fit
from car_env import DONE_NAMES, CarEnv
from train_camera import (
    CURRICULUM,
    Stage,
    agent_for,
    build_env,
    build_world,
    clear_starts,
    obstacle_provider,
)

# Sensory gains the fit is made under; ES refines them afterwards. Eye values
# are nearness in [0, 1] read directly (`eye_encoding="direct"`), so a gain of
# one puts a filled channel at ~190 Hz and an empty one at the bias.
SENSORY = {"ray_gain": 1.0, "bias_hz": 0.05, "speed_gain": 0.5, "loom_gain": 0.5}


class LaneTeacher:
    """Follows the lane from the true state; the label source, never the driver's sense.

    Pure pursuit on the lap centreline offset into the lane, and a speed
    governor that asks for a share of the posted limit (the track's grip
    reference where nothing is posted), braking for the corner ahead through
    the same reference. Smooth in the state, so a linear readout of a brain
    that sees the road can imitate it.
    """

    def __init__(
        self,
        env: CarEnv,
        law,
        lane_offset_m: float,
        lookahead_m: float,
        pace_fraction: float,
        gain: float,
        corner_g: float = 0.30,
        explore: float = 0.0,
        explore_tau_s: float = 0.6,
    ) -> None:
        self.env = env
        self.law = law
        self.lane_offset_m = lane_offset_m
        self.lookahead_m = lookahead_m
        self.pace_fraction = pace_fraction
        self.gain = gain
        cl = env.track.centerline
        self.n = cl.shape[0]
        ahead = torch.roll(cl, -1, dims=0) - torch.roll(cl, 1, dims=0)
        tangent = ahead / ahead.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.left = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=1)
        spacing = float((torch.roll(cl, -1, dims=0) - cl).norm(dim=1).mean())
        self.look = max(1, int(round(lookahead_m / max(spacing, 1e-6))))
        # Corner speed from the centreline's own curvature at a comfortable
        # lateral acceleration: a right-angle street corner of 6 m radius at
        # 0.3 g is 15 km/h, which is how a street car takes it. Read ahead
        # over a braking distance so the slowing starts before the corner.
        angle = torch.atan2(tangent[:, 1], tangent[:, 0])
        turn = torch.remainder(torch.roll(angle, -1, dims=0) - angle + torch.pi, 2 * torch.pi) - torch.pi
        radius = spacing / turn.abs().clamp(min=1e-4)
        corner = torch.sqrt(corner_g * 9.81 * radius).clamp(max=env.cfg.max_speed)
        window = max(1, int(round(35.0 / max(spacing, 1e-6))))
        stacked = torch.stack([torch.roll(corner, -k, dims=0) for k in range(window)], dim=1)
        self.corner_speed = stacked.min(dim=1).values
        # Exploration (DART): the executed steering wanders with slow noise
        # while the label stays the clean correction, so the recorded data
        # covers the offsets and heading errors the brain must learn to undo.
        self.explore = explore
        self.explore_alpha = float(np.exp(-env.cfg.dt_s / max(explore_tau_s, 1e-3)))
        self.noise = torch.zeros(env.batch, device=env.device)
        self.generator = torch.Generator(device="cpu").manual_seed(1234)

    def wander(self) -> torch.Tensor:
        """Slow steering noise added to what is executed, never to the label."""
        if self.explore <= 0.0:
            return torch.zeros_like(self.noise)
        fresh = torch.randn(self.noise.shape[0], generator=self.generator).to(self.noise.device)
        self.noise = self.explore_alpha * self.noise + np.sqrt(1.0 - self.explore_alpha**2) * fresh * self.explore
        return self.noise

    def act(self) -> torch.Tensor:
        env = self.env
        index = (env.last_progress * self.n).long().clamp(0, self.n - 1)
        target_i = (index + self.look) % self.n
        target = env.track.centerline[target_i] + self.left[target_i] * self.lane_offset_m
        delta = target - env.pos
        want = torch.atan2(delta[:, 1], delta[:, 0]) - env.heading
        want = torch.atan2(want.sin(), want.cos())
        steer = (want * self.gain / env.cfg.max_steer_rad).clamp(-1.0, 1.0)

        # Speed: the posted limit's share where posted, else the grip
        # reference; whichever is lower over the next few samples, so the car
        # slows before a corner rather than in it.
        corner = self.corner_speed[index]
        limit = self.law.limit_at(env.last_progress)
        posted = torch.where(torch.isfinite(limit), limit * self.pace_fraction, torch.full_like(limit, env.cfg.max_speed))
        target_speed = torch.minimum(torch.minimum(corner, posted), torch.full_like(corner, env.cfg.max_speed * 0.95))
        pedal = ((target_speed - env.speed) / 3.0).clamp(-1.0, 1.0)
        return torch.stack([steer, pedal], dim=1)


def collect(agent: ConnectomeAgent, env: CarEnv, sensor, teacher: LaneTeacher, theta, steps: int, beta: float, seed: int, record: bool) -> dict:
    """Roll brain and cars together; keep the output population and the teacher's labels."""
    agent.seed(seed)
    env.attach_sensor(sensor)
    obs = env.reset()
    agent.reset(batch=env.batch)
    batch = env.batch
    alive = torch.ones(batch, dtype=torch.bool, device=env.device)
    first_end = torch.full((batch,), -1, dtype=torch.long)
    reason = torch.zeros(batch, dtype=torch.long)
    speed_sum = torch.zeros(batch, device=env.device)
    xs, ys, cars = [], [], []
    car_index = torch.arange(batch)
    with torch.inference_mode():
        for step in range(steps):
            target = teacher.act()
            student = agent.act(obs, theta)
            executed = target.clone()
            executed[:, 0] = (executed[:, 0] + teacher.wander()).clamp(-1.0, 1.0)
            action = beta * executed + (1.0 - beta) * student
            if record:
                signal = agent.motor_state - agent.motor_state.mean(dim=1, keepdim=True)
                keep = alive.cpu()
                xs.append(signal[alive].half().cpu())
                ys.append(target[alive].cpu())
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
        "speed_kmh": speed_sum.cpu() / steps_alive.float() * 3.6,
    }


def describe(name: str, result: dict, steps: int) -> str:
    ended = [DONE_NAMES[int(r)] if int(a) < steps else "time" for r, a in zip(result["reason"], result["steps_alive"])]
    counts = {k: ended.count(k) for k in sorted(set(ended))}
    laps = result["laps"].numpy()
    return (
        f"{name}: laps min {laps.min():.3f} mean {laps.mean():.3f} max {laps.max():.3f}; "
        f"alive {result['steps_alive'].float().mean():.0f}/{steps} steps; {result['speed_kmh'].mean():.0f} km/h; {counts}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", type=Path, default=Path("data/tracks/kl.geojson"))
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--weights", default="checkpoints/perception/kl/weights/best.pt")
    parser.add_argument("--cars", type=int, default=12, help="cars (brain bodies) driven per round, spread along the street")
    parser.add_argument("--steps", type=int, default=1500, help="control steps per collection round")
    parser.add_argument("--rounds", type=int, default=2, help="DAgger rounds after the teacher-driven one")
    parser.add_argument("--beta", type=float, default=0.5, help="teacher share of the action in the middle rounds")
    parser.add_argument("--eval-steps", type=int, default=1500)
    parser.add_argument("--lams", default="3,10,30,100,300,1000")
    parser.add_argument("--lookahead", type=float, default=10.0, help="metres ahead the teacher aims at")
    parser.add_argument("--steer-gain", type=float, default=1.6)
    parser.add_argument("--corner-g", type=float, default=0.30, help="lateral g the teacher takes corners at")
    parser.add_argument("--explore", type=float, default=0.12, help="steering noise (DART) on the executed action while collecting")
    parser.add_argument("--steer-boost", type=float, default=1.0)
    parser.add_argument("--pedal-boost", type=float, default=1.0)
    parser.add_argument("--traffic", type=float, default=0.0, help="share of the street's traffic present while calibrating")
    parser.add_argument("--lane-offset", type=float, default=-1.9)
    parser.add_argument("--max-speed", type=float, default=17.0)
    parser.add_argument("--pace-fraction", type=float, default=0.92)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--perception-stride", type=int, default=3)
    parser.add_argument("--vehicles", type=int, default=120)
    parser.add_argument("--motorcycles", type=int, default=100)
    parser.add_argument("--pedestrians", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--substeps", type=int, default=defaults.SUBSTEPS)
    parser.add_argument("--dt-ms", type=float, default=defaults.DT_MS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/camera/driver.pt"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite the driver")
    args.popsize = args.cars
    args.width_mult = 1.0

    device = pick_device(args.device)
    connectome = load_connectome(args.graph)
    stage = Stage("calibration", args.traffic, 0.0)
    scene, traffic, sensor, director = build_world(args, device, stage)
    env, law = build_env(args, scene, device, args.cars, 1.0)
    removed = clear_starts(traffic, env)
    env.attach_traffic(obstacle_provider(traffic, scene.fixtures))
    print(f"[world] eye {sensor.eye.width} channels; {len(traffic.actors)} road users ({removed} cleared off the starts)")
    print(f"[world] {law.summary()}")

    brain = Brain(
        connectome,
        batch=args.cars,
        config=LIFConfig(dt_ms=args.dt_ms, adapt_mv=defaults.ADAPT_MV),
        device=device,
        weight_scale=defaults.WEIGHT_SCALE,
    )
    agent_cfg = agent_for(None, sensor, args.substeps)
    agent = ConnectomeAgent(brain, connectome.neurons, agent_cfg)
    teacher = LaneTeacher(
        env, law, args.lane_offset, args.lookahead, args.pace_fraction, args.steer_gain,
        corner_g=args.corner_g, explore=args.explore,
    )

    params = agent.initial_params()
    offset = 0
    for name, shape in agent.param_shapes.items():
        size = int(np.prod(shape))
        if name in SENSORY:
            params[offset : offset + size] = SENSORY[name]
        offset += size
    theta = agent.unpack(params.unsqueeze(0).repeat(args.cars, 1).to(device))
    lams = tuple(float(v) for v in args.lams.split(","))
    print(
        f"[brain] {connectome.n:,} neurons, {agent.n_readout} output cells, {agent_cfg.readout_dim} channels, "
        f"{args.cars} cars x {args.steps} steps, {brain.precision} on {device}"
    )

    pooled_x, pooled_y, pooled_car = [], [], []
    started = time.time()
    r2 = torch.zeros(2)
    for round_index in range(args.rounds + 1):
        beta = 1.0 if round_index == 0 else (args.beta if round_index < args.rounds else 0.0)
        result = collect(agent, env, sensor, teacher, theta, args.steps, beta, args.seed + round_index, record=True)
        print(f"[round {round_index}] beta {beta:.2f} " + describe("drive", result, args.steps) + f" ({time.time() - started:.0f}s)")
        pooled_x.append(result["x"])
        pooled_y.append(result["y"])
        pooled_car.append(result["car"])
        x = torch.cat(pooled_x).float()
        y = torch.atanh(torch.cat(pooled_y).clamp(-PRE_CLIP, PRE_CLIP))
        car = torch.cat(pooled_car)
        w, b, lam, r2 = fit_readout(x, y, car, lams)
        params, notes = install_fit(agent, x, w, b, params, boost=(args.steer_boost, args.pedal_boost))
        theta = agent.unpack(params.unsqueeze(0).repeat(args.cars, 1).to(device))
        print(
            f"[fit] {x.shape[0]:,} samples, lambda {lam:g}, held-out R^2 steer {float(r2[0]):.3f} pedal {float(r2[1]):.3f}; {notes}"
        )

    teacher.explore = 0.0
    final = collect(agent, env, sensor, teacher, theta, args.eval_steps, 0.0, args.seed + 100, record=False)
    print("[final] " + describe(f"brain alone, {args.eval_steps} steps", final, args.eval_steps))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mu": params.detach().cpu(),
            "generation": 0,
            "trained": False,
            "eye_width": sensor.eye.width,
            "stage": 0,
            "sigma": 0.04,
            "n_params": agent.n_params,
            "param_shapes": {k: list(v) for k, v in agent.param_shapes.items()},
            "agent_cfg": asdict(agent_cfg),
            "readout": agent.readout_state(),
            "track": str(args.track),
            "weights": args.weights,
            "columns": args.columns,
            "camera": {"width": args.width, "height": args.height},
            "curriculum": [asdict(s) for s in CURRICULUM],
            "saved_at": time.time(),
            "calibration": {
                "r2_steer": float(r2[0]),
                "r2_pedal": float(r2[1]),
                "final_laps": final["laps"].tolist(),
                "final_steps_alive": final["steps_alive"].tolist(),
                "rounds": args.rounds,
                "steps": args.steps,
                "cars": args.cars,
                "lookahead_m": args.lookahead,
                "steer_gain": args.steer_gain,
            },
        },
        args.out,
    )
    print(f"[ok] wrote {args.out}; refine with train_camera.py")
    sensor.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
