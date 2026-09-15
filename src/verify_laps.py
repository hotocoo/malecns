"""End-to-end lap verification gate.

Drives the brain alone from every start point for `--steps` control steps on
full-scale Monaco and reports, per start, whether a lap was completed, the lap
time, the mean speed and how the episode ended. Exit status is 0 only when
every start completes at least `--min-laps` laps with no crash, so it can gate
a checkpoint promotion. Every run is appended to `--log` (JSON lines), which is
the record that lap times are falling from one accepted checkpoint to the next.

  python3 src/verify_laps.py --checkpoint checkpoints/es.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import defaults
from agent import ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device
from car_env import DONE_NAMES, CarEnv, Track, build_centerline, monaco_config
from evaluate import agent_config_from, load_checkpoint, run_episode, timestep_from


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/es.pt"))
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--geojson", default="data/tracks/monaco.geojson")
    parser.add_argument("--starts", type=int, default=defaults.MONACO_STARTS)
    parser.add_argument("--steps", type=int, default=16000, help="256 s at the 16 ms control step")
    parser.add_argument("--min-laps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--log", type=Path, default=Path("logs/laps.jsonl"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    state = load_checkpoint(args.checkpoint, device)
    if state is None:
        raise SystemExit(f"no checkpoint at {args.checkpoint}")
    dt_ms, substeps = timestep_from(state, defaults.DT_MS, defaults.SUBSTEPS)
    dt_s = defaults.control_dt_s(dt_ms, substeps)
    connectome = load_connectome(args.graph)
    brain = Brain(connectome, batch=args.starts, config=LIFConfig(dt_ms=dt_ms, adapt_mv=defaults.ADAPT_MV), device=device, weight_scale=defaults.WEIGHT_SCALE)
    agent = ConnectomeAgent(brain, connectome.neurons, agent_config_from(state, substeps))
    agent.load_readout(state)
    mu, _, notes = agent.migrate_state(state)
    if any("(reset)" in n for n in notes):
        raise SystemExit(f"checkpoint readout would be reset ({notes}); refusing to verify a different policy")
    mu = mu[0] if mu.ndim == 2 else mu
    cfg = monaco_config(dt_s)
    cfg = type(cfg)(**{**cfg.__dict__, "geojson_path": args.geojson})
    track = Track(build_centerline(cfg), cfg, device)
    starts = torch.arange(args.starts) / args.starts
    env = CarEnv(args.starts, device, cfg, track=track, start_fraction=starts)
    theta = agent.unpack(mu.to(device).unsqueeze(0).repeat(args.starts, 1))
    started = time.time()
    result = run_episode(agent, env, theta, args.steps, seed=args.seed)
    elapsed = time.time() - started

    rows = []
    for k in range(args.starts):
        lap_step = int(result["first_lap_step"][k])
        reason = int(result["reason"][k])
        rows.append(
            {
                "start": round(float(starts[k]), 4),
                "laps": round(float(result["laps"][k]), 3),
                "lap_time_s": round(lap_step * dt_s, 1) if lap_step > 0 else None,
                "speed_kmh": round(float(result["speed_mean"][k]) * 3.6, 1),
                "steps_alive": int(result["steps_alive"][k]),
                "ended": DONE_NAMES[reason] if reason else "time",
            }
        )
    completed = [r for r in rows if r["laps"] >= args.min_laps]
    crashed = [r for r in rows if r["ended"] == "crash"]
    lap_times = [r["lap_time_s"] for r in rows if r["lap_time_s"]]
    passed = len(completed) == len(rows) and not crashed
    record = {
        "time": time.time(),
        "checkpoint": str(args.checkpoint),
        "generation": int(state.get("generation", 0)),
        "steps": args.steps,
        "passed": passed,
        "completed": len(completed),
        "starts": len(rows),
        "crashes": len(crashed),
        "laps_min": min(r["laps"] for r in rows),
        "laps_mean": round(float(np.mean([r["laps"] for r in rows])), 3),
        "best_lap_s": min(lap_times) if lap_times else None,
        "mean_lap_s": round(float(np.mean(lap_times)), 1) if lap_times else None,
        "speed_kmh": round(float(np.mean([r["speed_kmh"] for r in rows])), 1),
        "rows": rows,
        "wall_s": round(elapsed, 1),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: {len(completed)}/{len(rows)} starts completed >= {args.min_laps} lap(s), {len(crashed)} crashes, "
          f"laps min {record['laps_min']:.3f} mean {record['laps_mean']:.3f}, best lap {record['best_lap_s']} s, mean lap {record['mean_lap_s']} s, {record['speed_kmh']} km/h ({elapsed:.0f}s wall)")
    for r in rows:
        print(f"  start {r['start']:.3f}: laps {r['laps']:.3f}  lap {r['lap_time_s'] or '-':>7} s  {r['speed_kmh']:5.1f} km/h  alive {r['steps_alive']:5d}  {r['ended']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
