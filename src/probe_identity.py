"""Identity-collapse probe: where do different genomes stop behaving differently?

Runs one instrumented ES-style roll-out from the current checkpoint and, at
every level of the pipeline, measures cross-body diversity AND determinism:

  params (genomes)     -> pairwise distances, unique rows
  theta blocks         -> per-block std across bodies
  obs at reset         -> identical per start point (expected) or corrupted
  sensory rates        -> does identical obs + different theta give different input
  motor_state (DN rates over readout) -> does the brain spread the difference
  motor pre-tanh       -> readout output spread
  actions              -> post-tanh spread
  trajectories         -> crash step, progress, pairwise distance over time

Determinism twin checks:
  * Two bodies share genome A exactly: if their trajectories diverge, the
    batched brain leaks state across bodies (a real identity bug).
  * Same rollout twice with the same seed must reproduce exactly.
"""

from __future__ import annotations

import argparse
import hashlib

import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device
from car_env import CarEnv, Track, build_centerline, monaco_config
from dataclasses import replace


def digest(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:10]


def spread(x: torch.Tensor) -> float:
    x = x.detach().float()
    return float(x.std(dim=0).mean())


def run_once(agent, env, theta, steps, seed, captures):
    agent.seed(seed)
    obs = env.reset()
    agent.reset()
    captures["obs_reset"] = obs.detach().cpu()
    pos_hist, act_hist, motor_hist, rates_hist, crash_step = [], [], [], [], []
    done_step = torch.full((env.batch,), -1, dtype=torch.long)
    with torch.inference_mode():
        for step in range(steps):
            action = agent.act(obs, theta)
            if step < 120:
                act_hist.append(action.detach().cpu())
                motor_hist.append(agent.last_motor.detach().cpu())
                rates_hist.append(agent.motor_state.detach().cpu())
            obs, reward, done = env.step(action)
            pos_hist.append(env.pos.detach().cpu())
            newly = done.detach().cpu() & (done_step < 0)
            done_step[newly] = step
            if bool(done.all()):
                break
    captures["actions"] = torch.stack(act_hist) if act_hist else None
    captures["motor"] = torch.stack(motor_hist) if motor_hist else None
    captures["rates"] = torch.stack(rates_hist) if rates_hist else None
    captures["pos"] = torch.stack(pos_hist)
    captures["crash_step"] = done_step
    captures["laps"] = env.laps.detach().cpu()
    captures["reasons"] = env.done_reason.detach().cpu()
    return captures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/es.pt")
    ap.add_argument("--graph", default="data/graph_w5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--precision", default=None, choices=("fp16", "fp32"))
    args = ap.parse_args()

    device = pick_device(args.device)
    connectome = load_connectome(args.graph)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent_cfg = AgentConfig.from_saved(state.get("agent_cfg"))
    batch = args.popsize * args.starts
    brain = Brain(
        connectome, batch=batch,
        config=LIFConfig(dt_ms=defaults.DT_MS, adapt_mv=defaults.ADAPT_MV),
        device=device, weight_scale=defaults.WEIGHT_SCALE, precision=args.precision,
    )
    agent = ConnectomeAgent(brain, connectome.neurons, agent_cfg)
    agent.load_readout(state)
    mu, _, notes = agent.migrate_state(state)
    if mu.ndim == 2:
        mu = mu[0]
    mu = agent.clamp_params(mu.to(device))
    print(f"device={device} precision={brain.precision} metal={brain.uses_metal} batch={batch}")
    print(f"checkpoint gen={state.get('generation')} notes={notes or 'none'}")

    # ES-style population around mu. Genomes 0 and 1 are FORCED IDENTICAL to
    # detect cross-body leakage in the batched brain: they must stay identical.
    half = args.popsize // 2
    g = torch.Generator().manual_seed(1234)
    eps = torch.randn(half, mu.numel(), generator=g).to(device)
    perturb = torch.cat([eps, -eps], dim=0)
    pop = agent.clamp_params(mu.unsqueeze(0) + args.sigma * perturb)
    pop[1] = pop[0]
    params = pop.repeat_interleave(args.starts, dim=0)

    print(f"\n== genome level ==  hash(mu)={digest(mu)}")
    print(f"unique genomes: {torch.unique(pop, dim=0).shape[0]}/{pop.shape[0]} "
          f"(10 expected: 0==1 forced)")
    d = (pop[:, None, :] - pop[None, :, :]).abs().amax(dim=-1)
    print(f"pairwise max|dparam|: min={d[d>0].min():.4f} max={d.max():.4f}; rel. to |mu|={float(mu.norm()):.2f}")
    theta = agent.unpack(params)
    for k, v in theta.items():
        print(f"theta.{k:9s} std over bodies={spread(v):.6f}")

    car_cfg = replace(monaco_config(defaults.control_dt_s(defaults.DT_MS, agent_cfg.substeps)),
                      track_halfwidth=5.5 * 1.6)
    track = Track(build_centerline(car_cfg), car_cfg, device)
    fractions = torch.tensor([0.0, 0.333] * args.popsize)[:batch]
    env = CarEnv(batch, device, car_cfg, track=track, start_fraction=fractions)

    caps = run_once(agent, env, theta, args.steps, seed=42, captures={})

    print("\n== reset ==")
    obs0 = caps["obs_reset"]
    view = obs0.view(args.popsize, args.starts, -1)
    print(f"obs std across genomes @same start: {view.std(dim=0).mean():.3e}")
    print(f"obs hash by body: {[digest(obs0[i]) for i in range(min(6, batch))]}")

    acts, motors, rates = caps["actions"], caps["motor"], caps["rates"]
    print("\n== pipeline spread over first control steps (across-genome std) ==")
    for step in (0, 1, 5, 20, 60, 119):
        if step < acts.shape[0]:
            print(f"step {step:3d}: sensory-rate-now?  motor_state_std={spread(rates[step]):.4f} "
                  f"motor_std={spread(motors[step]):.4f} action_std={spread(acts[step]):.4f}")

    print("\n== twin check (bodies 0..S-1 and S..2S-1 are genome A) ==")
    s = args.starts
    pa, pb = caps["pos"][:, :s], caps["pos"][:, s:2 * s]
    drift = (pa - pb).norm(dim=-1).max()
    aa = (acts[:, :s] - acts[:, s:2 * s]).abs().max()
    print(f"identical-genome twin: max action diff={float(aa):.3e}, max position drift={float(drift):.3e} "
          f"(must be exactly 0; >0 means cross-body state leakage)")

    print("\n== trajectories ==")
    cs = caps["crash_step"]
    print(f"crash step per genome (start 0 body): {cs[::s][:args.popsize].tolist()}  (-1 = survived)")
    print(f"laps per genome: {[round(float(x), 4) for x in caps['laps'][::s]]}")
    print(f"reasons: {caps['reasons'][::s].tolist()}  (1 crash 2 reverse 3 stuck)")
    pos = caps["pos"]
    pairwise = (pos[:, ::s][:, :, None, :] - pos[:, ::s][:, None, :, :]).norm(dim=-1)
    for t in (10, 50, 100, 200, len(pos) - 1):
        t = min(t, len(pos) - 1)
        m = pairwise[t]
        off = m[~torch.eye(args.popsize, dtype=torch.bool)][:]
        print(f"t={t:4d}: genome-pair pos distance mean={float(off.mean()):8.2f} m  max={float(off.max()):8.2f} m")

    print("\n== replay determinism ==")
    caps2 = run_once(agent, env, theta, args.steps, seed=42, captures={})
    pd = (caps2["pos"] - caps["pos"]).abs().max()
    print(f"same seed re-run: max pos diff={float(pd):.3e} (must be exactly 0)")
    # Same genomes, DIFFERENT sensory seed: how much of fitness is input noise?
    caps3 = run_once(agent, env, theta, args.steps, seed=43, captures={})
    pd3 = (caps3["pos"] - caps["pos"]).abs().max()
    print(f"different seed re-run: max pos diff={float(pd3):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
