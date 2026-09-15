"""Probe ES parameter diversity through the controller into actions."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device
from car_env import CarEnv, Track, build_centerline, monaco_config


def digest(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]


def stat(name: str, x: torch.Tensor) -> None:
    y = x.detach().float().cpu()
    print(f"{name:18} shape={tuple(y.shape)!s:18} hash={digest(x)} min={y.min():+.6g} max={y.max():+.6g} std={y.std():.6g}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/es.pt"))
    p.add_argument("--graph", default="data/graph_w5")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = pick_device(args.device)
    connectome = load_connectome(args.graph)
    cfg = AgentConfig.from_saved(state.get("agent_cfg"))
    brain = Brain(connectome, batch=4, config=LIFConfig(dt_ms=defaults.DT_MS, adapt_mv=defaults.ADAPT_MV), device=device, weight_scale=defaults.WEIGHT_SCALE, precision="fp32")
    agent = ConnectomeAgent(brain, connectome.neurons, cfg)
    mu, _, notes = agent.migrate_state(state)
    mu = agent.clamp_params(mu.to(device))
    if mu.ndim == 2:
        print(f"checkpoint islands={mu.shape[0]}; probing island 0")
        mu = mu[0]
    print(f"generation={state.get('generation')} params={agent.n_params} migration={notes or 'none'}")
    stat("mu", mu)
    g = torch.Generator().manual_seed(123)
    eps = torch.randn(2, mu.numel(), generator=g, device=device)
    params = agent.clamp_params(mu.unsqueeze(0) + 0.05 * torch.cat([eps, -eps], dim=0))
    theta = agent.unpack(params)
    for k, v in theta.items():
        stat(f"theta.{k}", v)
    car_cfg = monaco_config(defaults.control_dt_s(defaults.DT_MS, cfg.substeps))
    track = Track(build_centerline(car_cfg), car_cfg, device)
    env = CarEnv(4, device, car_cfg, track=track, start_fraction=0.0)
    obs = env.reset()
    agent.reset()
    reset_obs = obs.clone()
    # Same genome/reset must reproduce exactly; this separates reset/RNG state
    # leakage from a controller/physics divergence.
    theta0 = agent.unpack(params[:1].repeat(4, 1))
    a0 = agent.act(obs, theta0)
    agent.reset()
    obs2 = env.reset()
    a0_repeat = agent.act(obs2, theta0)
    print(f"reset_obs_repeat_max={float((reset_obs - obs2).abs().max()):.3g}")
    print(f"same_genome_action_repeat_max={float((a0 - a0_repeat).abs().max()):.3g}")
    agent.reset()
    obs = env.reset()
    stat("initial obs", obs)
    for step in range(32):
        action = agent.act(obs, theta)
        print(f"step={step}")
        stat("action", action)
        if step == 0:
            for i in range(4):
                stat(f"action[{i}]", action[i])
        obs, reward, _ = env.step(action)
        stat("reward", reward)
        stat("pos", env.pos)
        if bool(env.done_reason.any()):
            print(f"done_reason={env.done_reason.tolist()} at step={step + 1}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
