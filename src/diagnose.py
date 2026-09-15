"""Neural diagnostics: does visual input produce avoidance commands?

Runs the connectome under controlled stimuli, all scenarios in parallel as
separate bodies of one batched brain that share the same Poisson draws, so
differences between scenarios are differences in stimulus, not noise:

  open track; wall on the left / right at several distances; wall ahead at
  several distances; braking and accelerating (speed channel ramps); a
  left-hand and a right-hand bend approaching.

Every scenario starts with an open-track baseline window. Recorded per control
step: spikes of all visual projection neurons and descending neurons, per-role
firing rates, and the motor outputs. From those:

  - baseline vs stimulus rates per role and per neuron
  - directional sensitivity  delta = rate(wall left) - rate(wall right)
  - silent / saturated / structurally disconnected neurons
  - the steering and pedal commands per scenario, with a PASS / WARN verdict
    on whether the car steers away from walls and brakes for one ahead
  - on a real Monaco run: correlation of each descending neuron's rate with
    steering and pedal, and the avoidance gain (steer vs lidar asymmetry)

  python3 src/diagnose.py --checkpoint checkpoints/best.pt --plot logs/diagnostics.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import defaults
from agent import ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device
from car_env import CarEnv, Track, build_centerline, monaco_config
from evaluate import agent_config_from, load_checkpoint, timestep_from

WALL_DISTANCES_M = (5.0, 15.0, 40.0, 100.0)
AHEAD_DISTANCES_M = (20.0, 60.0, 120.0)


def lidar_for(angles: np.ndarray, max_range: float, left: float | None = None, right: float | None = None, ahead: float | None = None) -> np.ndarray:
    """Normalised lidar for infinite walls parallel (left/right) or perpendicular (ahead) to the heading."""
    dist = np.full(angles.shape, np.inf)
    if left is not None:
        s = np.sin(angles)
        dist = np.where(s > 1e-3, np.minimum(dist, left / np.maximum(s, 1e-3)), dist)
    if right is not None:
        s = -np.sin(angles)
        dist = np.where(s > 1e-3, np.minimum(dist, right / np.maximum(s, 1e-3)), dist)
    if ahead is not None:
        c = np.cos(angles)
        dist = np.where(c > 1e-3, np.minimum(dist, ahead / np.maximum(c, 1e-3)), dist)
    return np.clip(dist / max_range, 0.0, 1.0)


def build_scenarios(angles: np.ndarray, max_range: float, steps: int, speed: float = 0.5) -> tuple[list[str], np.ndarray]:
    """Names and observations (steps, scenarios, n_rays + 1) for the stimulus window."""
    n_rays = len(angles)
    names: list[str] = []
    obs: list[np.ndarray] = []

    def constant(name: str, lidar: np.ndarray, spd: float = speed) -> None:
        names.append(name)
        frame = np.concatenate([lidar, [spd]]).astype(np.float32)
        obs.append(np.repeat(frame[None], steps, axis=0))

    constant("open", np.ones(n_rays))
    for d in WALL_DISTANCES_M:
        constant(f"wall_left_{d:g}m", lidar_for(angles, max_range, left=d))
    for d in WALL_DISTANCES_M:
        constant(f"wall_right_{d:g}m", lidar_for(angles, max_range, right=d))
    for d in AHEAD_DISTANCES_M:
        constant(f"wall_ahead_{d:g}m", lidar_for(angles, max_range, ahead=d))

    ramp = np.linspace(0.0, 1.0, steps)
    names.append("braking")
    obs.append(np.stack([np.concatenate([np.ones(n_rays), [0.8 - 0.7 * t]]) for t in ramp]).astype(np.float32))
    names.append("accelerating")
    obs.append(np.stack([np.concatenate([np.ones(n_rays), [0.1 + 0.7 * t]]) for t in ramp]).astype(np.float32))
    # A bend: the outside wall is ahead and to one side, closing in; the inside
    # wall stays at road half-width.
    for name, side in (("bend_left", "right"), ("bend_right", "left")):
        names.append(name)
        frames = []
        for t in ramp:
            ahead = 120.0 - 100.0 * t
            kwargs = {side: 5.5, "ahead": None}
            base = lidar_for(angles, max_range, **{side: 5.5})
            outside = lidar_for(angles, max_range, ahead=ahead)
            side_mask = np.sin(angles) < 0 if side == "right" else np.sin(angles) > 0
            # the wall ahead only fills the outside half of the field
            lidar = np.where(side_mask | (np.abs(np.sin(angles)) < 1e-3), np.minimum(base, outside), base)
            frames.append(np.concatenate([lidar, [speed]]))
        obs.append(np.asarray(frames, dtype=np.float32))
    return names, np.stack(obs, axis=1)


def reachable_from(brain: Brain, sources: torch.Tensor, hops: int) -> torch.Tensor:
    """Boolean (n,) mask of neurons reachable from `sources` within `hops` synapses."""
    reach = torch.zeros(brain.n, 1, device=brain.device)
    reach[sources] = 1.0
    frontier = reach.clone()
    w = brain.W
    for _ in range(hops):
        frontier = (torch.sparse.mm(w, frontier).abs() > 0).float()
        reach = torch.maximum(reach, frontier)
    return reach[:, 0] > 0


def pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlation of every column of x (T, k) with y (T,)."""
    xc = x - x.mean(0)
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum(0) * (yc**2).sum()) + 1e-9
    return (xc * yc[:, None]).sum(0) / denom


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, type=Path, help="default: checkpoints/best.pt, else es.pt")
    parser.add_argument("--graph", default="data/graph_w5")
    parser.add_argument("--baseline-steps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=60, help="stimulus window per scenario (control steps)")
    parser.add_argument("--track-steps", type=int, default=400, help="on-track run for DN/steering correlation; 0 skips")
    parser.add_argument("--geojson", default="data/tracks/monaco.geojson")
    parser.add_argument("--dt-ms", type=float, default=defaults.DT_MS)
    parser.add_argument("--substeps", type=int, default=defaults.SUBSTEPS)
    parser.add_argument("--weight-scale", type=float, default=defaults.WEIGHT_SCALE)
    parser.add_argument("--adapt-mv", type=float, default=defaults.ADAPT_MV)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("logs/diagnostics"))
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    state = load_checkpoint(args.checkpoint, device)
    dt_ms, substeps = timestep_from(state, args.dt_ms, args.substeps)
    dt_s = defaults.control_dt_s(dt_ms, substeps)
    connectome = load_connectome(args.graph)
    car_cfg = monaco_config(dt_s)
    angles = np.linspace(np.deg2rad(car_cfg.fov_deg) / 2, -np.deg2rad(car_cfg.fov_deg) / 2, car_cfg.n_rays)
    names, stimulus = build_scenarios(angles, car_cfg.max_range, args.steps)
    n_scn = len(names)

    brain = Brain(connectome, batch=n_scn, config=LIFConfig(dt_ms=dt_ms, adapt_mv=args.adapt_mv), device=device, weight_scale=args.weight_scale)
    agent = ConnectomeAgent(brain, connectome.neurons, agent_config_from(state, substeps))
    agent.load_readout(state)
    if state is not None and state["mu"].reshape(-1, agent.n_params).shape[1] == agent.n_params and state["mu"].numel() % agent.n_params == 0:
        # ES checkpoints hold one mean per island; diagnose island 0
        mu = state["mu"].reshape(-1, agent.n_params)[0].to(device)
        source = f"{state['path']} generation {state['generation']}"
    else:
        mu = agent.initial_params().to(device)
        source = "untrained interface"
    theta = agent.unpack(mu.unsqueeze(0).repeat(n_scn, 1))
    print(f"diagnostics on {source}; {n_scn} scenarios x ({args.baseline_steps} baseline + {args.steps} stimulus) steps; {device}")

    vpn = agent.input_index[: sum(g.numel() for g in agent.ray_groups)]
    dn = agent.dn_index
    role_names = list(connectome.roles.keys())
    role_index = [torch.tensor(connectome.roles[r], dtype=torch.long, device=device) for r in role_names]
    steps_total = args.baseline_steps + args.steps
    open_obs = stimulus[0, 0]  # scenario 0 is the open track
    vpn_spikes = np.zeros((steps_total, n_scn, vpn.numel()), dtype=np.uint8)
    dn_spikes = np.zeros((steps_total, n_scn, dn.numel()), dtype=np.uint8)
    motor = np.zeros((steps_total, n_scn, 2), dtype=np.float32)
    role_hz = np.zeros((steps_total, n_scn, len(role_names)), dtype=np.float32)
    window_s = dt_s

    agent.seed(args.seed)
    agent.reset()
    with torch.inference_mode():
        for t in range(steps_total):
            frame = np.repeat(open_obs[None], n_scn, axis=0) if t < args.baseline_steps else stimulus[t - args.baseline_steps]
            obs = torch.tensor(frame, device=device)
            sink = torch.zeros(n_scn, brain.n, device=device)
            action = agent.act(obs, theta, spike_sink=sink)
            vpn_spikes[t] = sink[:, vpn].clamp(max=255).to(torch.uint8).cpu().numpy()
            dn_spikes[t] = sink[:, dn].clamp(max=255).to(torch.uint8).cpu().numpy()
            motor[t] = action.cpu().numpy()
            role_hz[t] = np.stack([(sink[:, ix].sum(1) / ix.numel() / window_s).cpu().numpy() for ix in role_index], axis=1)

    b0, b1 = max(0, args.baseline_steps - 15), args.baseline_steps
    s0, s1 = args.baseline_steps + args.steps // 3, steps_total
    to_hz = 1.0 / window_s
    vpn_base = vpn_spikes[b0:b1].mean(0) * to_hz  # (scn, n_vpn)
    vpn_stim = vpn_spikes[s0:s1].mean(0) * to_hz
    dn_base = dn_spikes[b0:b1].mean(0) * to_hz
    dn_stim = dn_spikes[s0:s1].mean(0) * to_hz
    motor_stim = motor[s0:s1].mean(0)  # (scn, 2)
    motor_base = motor[b0:b1].mean(0)

    idx = {name: i for i, name in enumerate(names)}
    left_groups = np.concatenate([np.arange(sum(g.numel() for g in agent.ray_groups[:k]), sum(g.numel() for g in agent.ray_groups[: k + 1])) for k in range(car_cfg.n_rays // 2)])
    right_groups = np.arange(left_groups.max() + 1 + agent.ray_groups[car_cfg.n_rays // 2].numel(), vpn.numel())

    report: dict = {"source": source, "scenarios": names, "dt_ms": dt_ms, "substeps": substeps}

    # --- per-role rates ------------------------------------------------------------
    print("\nfiring rate by role (Hz), baseline open track vs stimulus window:")
    print(f"  {'scenario':<18}" + "".join(f"{r[:12]:>13}" for r in role_names) + f"{'steer':>8}{'pedal':>8}")
    rows = {}
    for i, name in enumerate(names):
        hz = role_hz[s0:s1, i].mean(0)
        rows[name] = {"role_hz": dict(zip(role_names, map(float, hz))), "steer": float(motor_stim[i, 0]), "pedal": float(motor_stim[i, 1])}
        print(f"  {name:<18}" + "".join(f"{v:>13.2f}" for v in hz) + f"{motor_stim[i, 0]:>8.3f}{motor_stim[i, 1]:>8.3f}")
    report["by_scenario"] = rows

    # --- lateralisation of the visual sheet -----------------------------------------
    print("\nvisual projection neurons, left eye vs right eye (Hz):")
    lateral = {}
    for name in [n for n in names if n.startswith("wall_")]:
        i = idx[name]
        l, r = float(vpn_stim[i, left_groups].mean()), float(vpn_stim[i, right_groups].mean())
        lateral[name] = {"left_hz": l, "right_hz": r}
        print(f"  {name:<18} left {l:7.2f}  right {r:7.2f}")
    report["vpn_lateralisation"] = lateral

    # --- directional sensitivity of descending neurons --------------------------------
    print("\ndirectional sensitivity of descending neurons: delta = rate(wall left) - rate(wall right)")
    deltas = []
    for d in WALL_DISTANCES_M:
        delta = dn_stim[idx[f"wall_left_{d:g}m"]] - dn_stim[idx[f"wall_right_{d:g}m"]]
        deltas.append(delta)
        strong = int((np.abs(delta) > 2.0).sum())
        print(f"  {d:>5g} m: |delta| > 2 Hz in {strong:4d} / {dn.numel()} DNs; mean |delta| {np.abs(delta).mean():.2f} Hz; population delta {delta.mean():+.2f} Hz")
    delta_mean = np.mean(deltas, axis=0)
    order = np.argsort(-np.abs(delta_mean))[: args.top]
    steer_w = agent.dn_steer_weight(agent.unpack(mu.unsqueeze(0))).cpu().numpy()
    print(f"  top {args.top} DNs by |delta| averaged over distances (with their effective steering weight):")
    top = []
    for j in order:
        top.append({"type": str(agent.dn_types[j]), "body": int(agent.dn_bodies[j]), "delta_hz": float(delta_mean[j]), "steer_w": float(steer_w[j])})
        print(f"    {str(agent.dn_types[j]):<14} body {agent.dn_bodies[j]:>10}  delta {delta_mean[j]:+7.2f} Hz  w_steer {steer_w[j]:+.3f}")
    report["dn_directional"] = {"per_distance_strong": [int((np.abs(d) > 2).sum()) for d in deltas], "top": top}

    # --- silent / saturated / disconnected --------------------------------------------
    max_hz = 1000.0 / dt_ms
    vpn_any = vpn_spikes.sum((0, 1))
    dn_any = dn_spikes.sum((0, 1))
    vpn_peak = np.maximum(vpn_stim, vpn_base).max(0)
    dn_peak = np.maximum(dn_stim, dn_base).max(0)
    pre = torch.tensor(connectome.pre, dtype=torch.long)
    post = torch.tensor(connectome.post, dtype=torch.long)
    out_deg = torch.bincount(pre, minlength=brain.n)
    in_deg = torch.bincount(post, minlength=brain.n)
    vpn_cpu, dn_cpu = vpn.cpu(), dn.cpu()
    reach = reachable_from(brain, vpn, hops=4).cpu()
    health = {
        "vpn_total": int(vpn.numel()),
        "vpn_silent": int((vpn_any == 0).sum()),
        "vpn_saturated": int((vpn_peak >= 0.4 * max_hz).sum()),
        "vpn_no_outputs": int((out_deg[vpn_cpu] == 0).sum()),
        "dn_total": int(dn.numel()),
        "dn_silent": int((dn_any == 0).sum()),
        "dn_saturated": int((dn_peak >= 0.4 * max_hz).sum()),
        "dn_no_inputs": int((in_deg[dn_cpu] == 0).sum()),
        "dn_unreachable_4_hops": int((~reach[dn_cpu]).sum()),
        "saturation_threshold_hz": 0.4 * max_hz,
    }
    report["health"] = health
    print("\nneuron health:")
    print(f"  visual projection: {health['vpn_silent']}/{health['vpn_total']} silent, {health['vpn_saturated']} saturated (>= {0.4 * max_hz:.0f} Hz, refractory-limited), {health['vpn_no_outputs']} with no outgoing synapses")
    print(f"  descending:        {health['dn_silent']}/{health['dn_total']} silent, {health['dn_saturated']} saturated, {health['dn_no_inputs']} with no inputs, {health['dn_unreachable_4_hops']} unreachable from vision within 4 hops")

    # --- verdicts on the commands ----------------------------------------------------
    verdicts = {}
    steer = {n: float(motor_stim[idx[n], 0]) for n in names}
    pedal = {n: float(motor_stim[idx[n], 1]) for n in names}
    left_minus_right = np.mean([steer[f"wall_left_{d:g}m"] - steer[f"wall_right_{d:g}m"] for d in WALL_DISTANCES_M[:3]])
    verdicts["steers_away_from_walls"] = bool(left_minus_right < -0.02)
    verdicts["steer_left_minus_right"] = float(left_minus_right)
    ahead_close = pedal["wall_ahead_20m"] - pedal["open"]
    verdicts["brakes_for_wall_ahead"] = bool(ahead_close < -0.02)
    verdicts["pedal_ahead_minus_open"] = float(ahead_close)
    bend = steer["bend_left"] - steer["bend_right"]
    verdicts["turns_into_bends"] = bool(bend > 0.02)
    verdicts["steer_bend_left_minus_right"] = float(bend)
    graded = sorted(WALL_DISTANCES_M)
    monotone = all(
        abs(steer[f"wall_left_{graded[k]:g}m"] - steer[f"wall_right_{graded[k]:g}m"]) >= abs(steer[f"wall_left_{graded[k + 1]:g}m"] - steer[f"wall_right_{graded[k + 1]:g}m"]) - 0.02
        for k in range(len(graded) - 1)
    )
    verdicts["response_grows_with_proximity"] = bool(monotone)
    report["verdicts"] = verdicts
    print("\ncommand verdicts (steer > 0 is left, pedal < 0 is brake):")
    for key in ("steers_away_from_walls", "brakes_for_wall_ahead", "turns_into_bends", "response_grows_with_proximity"):
        print(f"  {'PASS' if verdicts[key] else 'WARN'}  {key}")
    print(f"  steer(wall left) - steer(wall right) = {left_minus_right:+.3f}   pedal(wall 20 m ahead) - pedal(open) = {ahead_close:+.3f}   steer(bend left) - steer(bend right) = {bend:+.3f}")

    # --- on-track correlation ---------------------------------------------------------
    if args.track_steps > 0:
        cfg = monaco_config(dt_s)
        cfg = cfg.__class__(**{**cfg.__dict__, "geojson_path": args.geojson})
        track = Track(build_centerline(cfg), cfg, device)
        env = CarEnv(n_scn, device, cfg, track=track, start_fraction=torch.arange(n_scn) / n_scn)
        agent.seed(args.seed + 1)
        obs = env.reset()
        agent.reset()
        alive = torch.ones(n_scn, dtype=torch.bool, device=device)
        dn_hist, act_hist, asym_hist, alive_hist = [], [], [], []
        with torch.inference_mode():
            for _ in range(args.track_steps):
                action = agent.act(obs, theta)
                lidar = obs[:, : cfg.n_rays]
                prox = 1 - lidar
                asym = prox[:, : cfg.n_rays // 2].mean(1) - prox[:, cfg.n_rays // 2 + 1 :].mean(1)
                dn_hist.append(agent.dn_rate_hz.cpu().numpy().copy())
                act_hist.append(action.cpu().numpy().copy())
                asym_hist.append(asym.cpu().numpy().copy())
                alive_hist.append(alive.cpu().numpy().copy())
                obs, _, done = env.step(action)
                alive = alive & ~done
        dn_hist = np.asarray(dn_hist)  # (T, B, n_dn)
        act_hist = np.asarray(act_hist)
        asym_hist = np.asarray(asym_hist)
        keep = np.asarray(alive_hist).reshape(-1)
        x = dn_hist.reshape(-1, dn.numel())[keep]
        a = act_hist.reshape(-1, 2)[keep]
        asym = asym_hist.reshape(-1)[keep]
        corr_steer = pearson(x, a[:, 0])
        corr_pedal = pearson(x, a[:, 1])
        gain = float(np.polyfit(asym, a[:, 0], 1)[0]) if asym.std() > 1e-6 else 0.0
        r_asym = float(np.corrcoef(asym, a[:, 0])[0, 1]) if asym.std() > 1e-6 else 0.0
        corr = {
            "samples": int(keep.sum()),
            "avoidance_gain_steer_per_asymmetry": gain,
            "avoidance_corr": r_asym,
            "avoids": bool(gain < -0.05),
            "top_steer": [
                {"type": str(agent.dn_types[j]), "body": int(agent.dn_bodies[j]), "r": float(corr_steer[j])}
                for j in np.argsort(-np.abs(np.nan_to_num(corr_steer)))[: args.top]
            ],
            "top_pedal": [
                {"type": str(agent.dn_types[j]), "body": int(agent.dn_bodies[j]), "r": float(corr_pedal[j])}
                for j in np.argsort(-np.abs(np.nan_to_num(corr_pedal)))[: args.top]
            ],
        }
        report["on_track"] = corr
        print(f"\non-track ({args.track_steps} steps, {n_scn} starts, {corr['samples']} live samples):")
        print(f"  {'PASS' if corr['avoids'] else 'WARN'}  avoidance gain d(steer)/d(left-right proximity) = {gain:+.3f} (r = {r_asym:+.3f}); negative means it steers away from the nearer wall")
        print("  DNs most correlated with steering:")
        for e in corr["top_steer"]:
            print(f"    {e['type']:<14} body {e['body']:>10}  r {e['r']:+.3f}")
        print("  DNs most correlated with pedal:")
        for e in corr["top_pedal"]:
            print(f"    {e['type']:<14} body {e['body']:>10}  r {e['r']:+.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        args.out / "recording.npz",
        names=np.asarray(names),
        vpn_spikes=vpn_spikes,
        dn_spikes=dn_spikes,
        motor=motor,
        role_hz=role_hz,
        role_names=np.asarray(role_names),
        dn_bodies=agent.dn_bodies,
        dn_types=np.asarray(agent.dn_types).astype(str),
        baseline_steps=args.baseline_steps,
    )
    print(f"\n[out] {args.out}/summary.json, recording.npz")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        groups = [vpn_stim[:, sum(g.numel() for g in agent.ray_groups[:k]) : sum(g.numel() for g in agent.ray_groups[: k + 1])].mean(1) for k in range(car_cfg.n_rays)]
        im = ax.imshow(np.asarray(groups).T, aspect="auto", cmap="magma")
        ax.set_yticks(range(n_scn))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("visual group (left -> right)")
        ax.set_title("visual projection neuron rate per ray group (Hz)")
        fig.colorbar(im, ax=ax)
        ax = axes[0, 1]
        ax.hist(delta_mean, bins=60, color="#c0392b")
        ax.set_title("DN directional sensitivity: rate(wall left) - rate(wall right), Hz")
        ax.set_xlabel("delta Hz")
        ax = axes[1, 0]
        xs = np.arange(n_scn)
        ax.bar(xs - 0.2, motor_stim[:, 0], width=0.4, label="steer (+ left)")
        ax.bar(xs + 0.2, motor_stim[:, 1], width=0.4, label="pedal (- brake)")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=70, fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend()
        ax.set_title("motor commands per scenario")
        ax = axes[1, 1]
        if "on_track" in report:
            ax.scatter(asym, a[:, 0], s=3, alpha=0.3)
            ax.set_xlabel("left - right proximity")
            ax.set_ylabel("steer")
            ax.set_title(f"on-track avoidance, gain {gain:+.3f}")
        else:
            ax.axis("off")
        fig.suptitle(f"MaleCNS diagnostics: {source}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"[plot] {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
