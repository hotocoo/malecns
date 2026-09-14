"""Evaluate a checkpoint: long-horizon runs, every start point, stress tracks.

  python3 src/evaluate.py                          # best.pt (or es.pt), all 6 starts, 3000 steps
  python3 src/evaluate.py --suite                  # + mirrored Monaco, narrow road, hard loops
  python3 src/evaluate.py --start 2 --plot run.png # one start, trajectory plot
  python3 src/evaluate.py --record logs/run.npz    # capture a run for `viewer.py --replay`

Deterministic: sensory noise is seeded (`--seed`), so two runs of the same
checkpoint give identical numbers. Timestep settings are taken from the
checkpoint so the policy is evaluated at the control period it was trained at.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import defaults
from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device
from car_env import DONE_NAMES, CarConfig, CarEnv, Track, build_centerline, curvature_radius, monaco_config
from exploits import ExploitMonitor


def load_checkpoint(path: Path | None, device: torch.device) -> dict | None:
    if path is None:
        for candidate in (Path("checkpoints/best.pt"), Path("checkpoints/es.pt")):
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        return None
    state = torch.load(path, map_location=device)
    state["path"] = str(path)
    return state


def agent_config_from(state: dict | None, substeps: int) -> AgentConfig:
    if state and "agent_cfg" in state:
        return AgentConfig.from_saved(state["agent_cfg"])
    return AgentConfig(substeps=substeps)


def timestep_from(state: dict | None, dt_ms: float, substeps: int) -> tuple[float, int]:
    if state and "args" in state:
        return float(state["args"].get("dt_ms", dt_ms)), int(state["args"].get("substeps", substeps))
    return dt_ms, substeps


def plot_run(track: Track, traces: list[np.ndarray], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = track.extent
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        track.drivable.cpu().numpy(),
        origin="lower",
        extent=(-extent, extent, -extent, extent),
        cmap="Greys",
        alpha=0.3,
    )
    colours = plt.cm.viridis(np.linspace(0, 1, len(traces)))
    for trace, colour in zip(traces, colours):
        ax.plot(trace[:, 0], trace[:, 1], lw=1.4, color=colour)
        ax.scatter(trace[0, 0], trace[0, 1], s=30, color=colour, zorder=3)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot] {out}")


def run_episode(
    agent: ConnectomeAgent,
    env: CarEnv,
    theta: dict[str, torch.Tensor],
    steps: int,
    seed: int,
    record: dict | None = None,
    sample: torch.Tensor | None = None,
) -> dict:
    """One batched episode; per-car results plus optional per-step recording."""
    agent.seed(seed)
    obs = env.reset()
    agent.reset()
    batch = env.batch
    alive = torch.ones(batch, dtype=torch.bool, device=env.device)
    total = torch.zeros(batch, device=env.device)
    steps_alive = torch.zeros(batch, dtype=torch.long, device=env.device)
    first_lap_step = torch.full((batch,), -1, dtype=torch.long, device=env.device)
    speed_sum = torch.zeros(batch, device=env.device)
    lat_g_max = torch.zeros(batch, device=env.device)
    traces = [[env.pos[b].cpu().numpy().copy()] for b in range(batch)]
    reason = torch.zeros(batch, dtype=torch.long, device=env.device)
    frames = record is not None
    watch = ExploitMonitor(env)
    with torch.inference_mode():
        for step in range(steps):
            sink = torch.zeros(batch, agent.brain.n, device=env.device) if frames else None
            action = agent.act(obs, theta, spike_sink=sink)
            obs, reward, done = env.step(action)
            watch.observe(reward, alive)
            total = total + reward * alive
            steps_alive = steps_alive + alive.long()
            speed_sum = speed_sum + env.speed * alive
            lat_g_max = torch.maximum(lat_g_max, env.lat_g * alive)
            crossed = (env.laps >= 1.0) & (first_lap_step < 0) & alive
            first_lap_step = torch.where(crossed, torch.full_like(first_lap_step, step + 1), first_lap_step)
            newly_done = done & alive
            reason = torch.where(newly_done, env.done_reason, reason)
            pos = env.pos.cpu().numpy()
            for b in range(batch):
                if bool(alive[b]):
                    traces[b].append(pos[b].copy())
            if frames:
                record["pos"].append(pos[0].copy())
                record["heading"].append(float(env.heading[0]))
                record["speed"].append(float(env.speed[0]))
                record["steer"].append(float(action[0, 0]))
                record["pedal"].append(float(action[0, 1]))
                record["reward"].append(float(reward[0]))
                record["laps"].append(float(env.laps[0]))
                record["lat_g"].append(float(env.lat_g[0]))
                record["lidar"].append(obs[0, : env.cfg.n_rays].cpu().numpy().copy())
                record["dn_hz"].append(agent.dn_rate_hz[0].cpu().numpy().copy())
                spiked = (sink[0] > 0).to(torch.uint8).cpu().numpy()
                record["mask"].append(np.packbits(spiked))
                if sample is not None:
                    record["fired"].append(spiked[sample.cpu().numpy()].copy())
            alive = alive & ~done
            if not bool(alive.any()):
                break
    return {
        "reward": total.cpu().numpy(),
        "laps": env.laps.cpu().numpy(),
        "steps_alive": steps_alive.cpu().numpy(),
        "speed_mean": (speed_sum / steps_alive.clamp(min=1).float()).cpu().numpy(),
        "lat_g_max": lat_g_max.cpu().numpy(),
        "first_lap_step": first_lap_step.cpu().numpy(),
        "reason": reason.cpu().numpy(),
        "traces": [np.asarray(t) for t in traces],
        "exploits": watch.report(),
    }


def summarise(name: str, env: CarEnv, result: dict, steps: int) -> dict:
    dt = env.cfg.dt_s
    lap_steps = result["first_lap_step"]
    lap_times = [s * dt for s in lap_steps if s > 0]
    ended = [DONE_NAMES[int(r)] if r else "time" for r in result["reason"]]
    row = {
        "scenario": name,
        "cars": int(env.batch),
        "steps": steps,
        "reward_mean": float(result["reward"].mean()),
        "reward_min": float(result["reward"].min()),
        "laps_mean": float(result["laps"].mean()),
        "laps_min": float(result["laps"].min()),
        "distance_m_mean": float(result["laps"].mean() * env.track.length_m),
        "steps_alive_mean": float(result["steps_alive"].mean()),
        "speed_kmh_mean": float(result["speed_mean"].mean() * 3.6),
        "lat_g_max": float(result["lat_g_max"].max()),
        "laps_completed": int((lap_steps > 0).sum()),
        "best_lap_s": min(lap_times) if lap_times else None,
        "ended": {k: ended.count(k) for k in sorted(set(ended))},
        "exploits": result.get("exploits"),
    }
    return row


def print_table(rows: list[dict]) -> None:
    head = f"{'scenario':<22}{'cars':>5}{'reward':>10}{'laps':>8}{'min':>8}{'dist m':>9}{'alive':>8}{'km/h':>7}{'latG':>6}  best lap / endings"
    print(head)
    print("-" * len(head))
    for r in rows:
        lap = f"{r['best_lap_s']:.1f}s" if r["best_lap_s"] else "-"
        print(
            f"{r['scenario']:<22}{r['cars']:>5}{r['reward_mean']:>10.1f}{r['laps_mean']:>8.3f}{r['laps_min']:>8.3f}"
            f"{r['distance_m_mean']:>9.0f}{r['steps_alive_mean']:>8.0f}{r['speed_kmh_mean']:>7.0f}{r['lat_g_max']:>6.2f}  {lap} {r['ended']}"
        )


def scenarios(args: argparse.Namespace, dt_s: float, device: torch.device, n_starts: int) -> list[tuple[str, CarConfig, Track, torch.Tensor]]:
    """(name, cfg, track, start fractions) for the run. `--suite` adds stress tracks."""
    out = []
    if args.layout == "monaco":
        cfg = replace(monaco_config(dt_s), geojson_path=args.geojson)
        track = Track(build_centerline(cfg), cfg, device)
        starts = torch.tensor([args.start / n_starts]) if args.start is not None else torch.arange(n_starts) / n_starts
        out.append(("monaco", cfg, track, starts))
        # The tightest corner on its own: start 150 m before the Fairmont
        # hairpin so cornering is scored separately from the rest of the lap.
        cl = track.centerline.cpu().numpy()
        n = len(cl)
        tightest = int(np.argmin(curvature_radius(cl)))
        spacing = track.length_m / n
        entry = (tightest - int(150.0 / spacing)) % n
        out.append(("monaco hairpin entry", cfg, track, torch.tensor([entry / n])))
        if args.suite:
            narrow = replace(cfg, track_halfwidth=cfg.track_halfwidth * 0.8)
            out.append(("monaco narrow x0.8", narrow, Track(build_centerline(narrow), narrow, device), starts))
            mirror = replace(cfg, mirror=True)
            out.append(("monaco mirrored", mirror, Track(build_centerline(mirror), mirror, device), starts))
    else:
        cfg = CarConfig(dt_s=dt_s)
        seeds = [args.start] if args.start is not None else list(range(n_starts))
        for s in seeds:
            out.append((f"loop {s}", cfg, Track(build_centerline(cfg, s), cfg, device), torch.tensor([0.0])))
    if args.suite:
        hard = CarConfig(dt_s=dt_s, loop_difficulty=1.6)
        for s in (101, 102, 103):
            out.append((f"hard loop {s}", hard, Track(build_centerline(hard, s), hard, device), torch.tensor([0.0])))
        narrow_loop = CarConfig(dt_s=dt_s, track_halfwidth=4.0)
        out.append(("narrow loop 7", narrow_loop, Track(build_centerline(narrow_loop, 7), narrow_loop, device), torch.tensor([0.0])))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, type=Path, help="default: checkpoints/best.pt, else es.pt")
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--layout", default="monaco", choices=("monaco", "loop"))
    parser.add_argument("--geojson", default="data/tracks/monaco.geojson")
    parser.add_argument("--start", type=int, default=None, help="one start point (monaco) or loop seed; default all")
    parser.add_argument("--starts", type=int, default=defaults.MONACO_STARTS)
    parser.add_argument("--steps", type=int, default=defaults.EVAL_STEPS, help="long-horizon cap; a W11 lap is ~4,600 steps")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--suite", action="store_true", help="add stress tracks: narrow, mirrored, hard loops")
    parser.add_argument("--dt-ms", type=float, default=defaults.DT_MS)
    parser.add_argument("--substeps", type=int, default=defaults.SUBSTEPS)
    parser.add_argument("--weight-scale", type=float, default=defaults.WEIGHT_SCALE)
    parser.add_argument("--adapt-mv", type=float, default=defaults.ADAPT_MV)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="write the summary table as JSON")
    parser.add_argument("--record", type=Path, default=None, help="save the first scenario's first car for viewer replay")
    parser.add_argument("--top-dn", type=int, default=15)
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    state = load_checkpoint(args.checkpoint, device)
    dt_ms, substeps = timestep_from(state, args.dt_ms, args.substeps)
    dt_s = defaults.control_dt_s(dt_ms, substeps)
    connectome = load_connectome(args.graph)
    plan = scenarios(args, dt_s, device, args.starts)
    batch = max(int(s[3].numel()) for s in plan)
    brain = Brain(
        connectome,
        batch=batch,
        config=LIFConfig(dt_ms=dt_ms, adapt_mv=args.adapt_mv),
        device=device,
        weight_scale=args.weight_scale,
    )
    agent = ConnectomeAgent(brain, connectome.neurons, agent_config_from(state, substeps))
    mu = None
    if state is not None:
        try:
            mu_cpu, _, notes = agent.migrate_state(state)
            mu = mu_cpu.to(device)
            print(
                f"checkpoint {state['path']} generation {state['generation']}, control step {dt_s * 1000:.0f} ms"
                + (f"; migrated: {', '.join(notes)}" if notes else "")
            )
        except ValueError as exc:
            print(f"{exc}; using untrained interface")
    else:
        print("no checkpoint: evaluating untrained interface")
    if mu is None:
        mu = agent.initial_params().to(device)

    rows = []
    traces_for_plot: list[np.ndarray] = []
    plot_track: Track | None = None
    record: dict | None = None
    for i, (name, cfg, track, starts) in enumerate(plan):
        n = int(starts.numel())
        if n != batch:
            # cars beyond the scenario's starts duplicate the first start; they
            # are dropped from the summary
            starts = torch.cat([starts, starts[:1].repeat(batch - n)])
        env = CarEnv(batch, device, cfg, track=track, start_fraction=starts)
        rec = None
        sample = None
        if i == 0 and args.record is not None:
            rec = {k: [] for k in ("pos", "heading", "speed", "steer", "pedal", "reward", "laps", "lat_g", "lidar", "dn_hz", "mask", "fired")}
        theta = agent.unpack(mu.unsqueeze(0).repeat(batch, 1))
        result = run_episode(agent, env, theta, args.steps, args.seed, record=rec, sample=sample)
        exploits = result.pop("exploits")
        result = {k: (v[:n] if isinstance(v, np.ndarray) else v[:n]) for k, v in result.items()}
        result["exploits"] = exploits
        sub_env = CarEnv.__new__(CarEnv)
        sub_env.batch, sub_env.cfg, sub_env.track = n, cfg, track
        rows.append(summarise(name, sub_env, result, args.steps))
        if i == 0:
            traces_for_plot = result["traces"]
            plot_track = track
            if rec is not None:
                record = rec
                record["meta"] = {
                    "layout": args.layout,
                    "geojson": args.geojson,
                    "start_fraction": float(starts[0]),
                    "dt_ms": dt_ms,
                    "substeps": substeps,
                    "generation": int(state["generation"]) if state else 0,
                    "track_halfwidth": cfg.track_halfwidth,
                    "n_rays": cfg.n_rays,
                    "fov_deg": cfg.fov_deg,
                    "max_range": cfg.max_range,
                    "max_speed": cfg.max_speed,
                }

    print()
    print_table(rows)
    print("\nexploit detector (per scenario, counts are cars flagged):")
    for r in rows:
        print(f"  {r['scenario']:<22} {ExploitMonitor.describe(r['exploits'])}")

    # The learned readout lives in the projected space; fold the fixed random
    # projection back in to get an effective weight per descending neuron.
    theta1 = agent.unpack(mu.unsqueeze(0))
    steer_w = agent.dn_steer_weight(theta1).cpu().numpy()
    rate = agent.dn_rate_hz[0].cpu().numpy()
    influence = steer_w * rate
    order = np.argsort(-np.abs(influence))[: args.top_dn]
    print("\ntop descending neurons by steering influence (weight x rate at episode end):")
    for i in order:
        print(
            f"  {str(agent.dn_types[i]):<14} body {agent.dn_bodies[i]:>10}  "
            f"rate {rate[i]:6.2f} Hz  w {steer_w[i]:+.3f}  infl {influence[i]:+.3f}"
        )

    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"[json] {args.json}")
    if args.plot is not None and plot_track is not None:
        plot_run(plot_track, traces_for_plot, args.plot, f"{rows[0]['scenario']}: {rows[0]['laps_mean']:.3f} laps mean")
    if record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.record,
            meta=json.dumps(record["meta"]),
            pos=np.asarray(record["pos"], dtype=np.float32),
            heading=np.asarray(record["heading"], dtype=np.float32),
            speed=np.asarray(record["speed"], dtype=np.float32),
            steer=np.asarray(record["steer"], dtype=np.float32),
            pedal=np.asarray(record["pedal"], dtype=np.float32),
            reward=np.asarray(record["reward"], dtype=np.float32),
            laps=np.asarray(record["laps"], dtype=np.float32),
            lat_g=np.asarray(record["lat_g"], dtype=np.float32),
            lidar=np.asarray(record["lidar"], dtype=np.float32),
            dn_hz=np.asarray(record["dn_hz"], dtype=np.float32),
            mask=np.asarray(record["mask"], dtype=np.uint8),
        )
        print(f"[record] {args.record} ({len(record['pos'])} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
