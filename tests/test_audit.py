"""Audit tests: reward hacking, vehicle dynamics, termination, ES, determinism.

Pure-environment tests run without the connectome; the ones marked
`needs_graph` skip when `data/graph_w5` has not been built.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import AgentConfig, ConnectomeAgent  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402
from car_env import (  # noqa: E402
    DONE_CRASH,
    DONE_REVERSE,
    DONE_STUCK,
    CarConfig,
    CarEnv,
    Track,
    build_centerline,
    curvature_radius,
    make_centerline,
    monaco_config,
)
from train import CURRICULUM, TrackBank, make_env, rank_normalise, rollout  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
MONACO = ROOT / "data" / "tracks" / "monaco.geojson"
needs_graph = pytest.mark.skipif(not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first")
needs_monaco = pytest.mark.skipif(not MONACO.exists(), reason="monaco.geojson missing")
CPU = torch.device("cpu")

# A small, fast track for dynamics tests: coarse grid, default vehicle.
FAST = CarConfig(grid_res=1024)


@pytest.fixture(scope="module")
def loop_track() -> Track:
    return Track(build_centerline(FAST, 1), FAST, CPU)


def drive(env: CarEnv, steer: float, pedal: float, steps: int) -> torch.Tensor:
    done = torch.zeros(env.batch, dtype=torch.bool)
    for _ in range(steps):
        _, _, done = env.step(torch.tensor([[steer, pedal]]).repeat(env.batch, 1))
        if bool(done.all()):
            break
    return done


# --- 1, 17: reward -------------------------------------------------------------------
def test_finish_line_oscillation_pays_the_lap_bonus_once(loop_track):
    # max_laps=0: completing the lap must not end (and freeze) the car, or it cannot drive back.
    env = CarEnv(1, CPU, replace(FAST, max_laps=0.0), track=loop_track)
    env.reset()
    cl = loop_track.centerline
    n = cl.shape[0]
    # park just before the line, 0.999 laps in
    env.pos = cl[n - 1].unsqueeze(0).clone()
    env.last_progress = env.track.progress_at(env.pos)
    env.laps = torch.tensor([0.999])
    env.best_laps = torch.tensor([0.999])
    total_bonus = 0.0
    for _ in range(3):
        env.pos = cl[1].unsqueeze(0).clone()  # forward over the line (2 samples, within one step's travel)
        _, reward, _ = env.step(torch.tensor([[0.0, 0.0]]))
        total_bonus += max(0.0, float(reward[0]))
        env.pos = cl[n - 1].unsqueeze(0).clone()  # back over it
        env.step(torch.tensor([[0.0, 0.0]]))
    assert env.laps[0] < 1.0, "net progress is back below one lap"
    assert FAST.lap_bonus * 0.9 < total_bonus < FAST.lap_bonus * 1.2, total_bonus


def test_progress_reward_is_a_potential(loop_track):
    """Sum of progress rewards depends only on where you end up, not the path."""
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    cl = loop_track.centerline
    # teleport through a wiggly path: the sum of deltas equals end - start
    start = float(env.track.progress_at(env.pos)[0])
    deltas = 0.0
    for i in (5, 3, 9, 6, 14):
        env.pos = cl[i].unsqueeze(0).clone()
        p = env.track.progress_at(env.pos)
        d = p - env.last_progress
        env.last_progress = p
        deltas += float(d[0])
    end = float(env.track.progress_at(env.pos)[0])
    assert abs(deltas - (end - start)) < 1e-6


def test_progress_jump_across_the_circuit_is_not_paid(loop_track):
    """Snapping to a far part of the track (crash step) must not move laps."""
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    n = loop_track.centerline.shape[0]
    env.pos = loop_track.centerline[n // 3].unsqueeze(0).clone()  # a third of a lap away
    env.speed = torch.tensor([1.0])
    _, reward, _ = env.step(torch.tensor([[0.0, 0.0]]))
    assert abs(float(env.laps[0])) < env.max_step_progress
    assert float(reward[0]) < 1.0


def test_idling_is_terminated_as_stuck(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    done = drive(env, 0.0, 0.0, env.stuck_steps + 2)
    assert bool(done[0])
    assert int(env.done_reason[0]) == DONE_STUCK


def test_circling_on_the_spot_is_terminated(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    env.speed = torch.tensor([8.0])
    done = drive(env, 1.0, 0.2, env.stuck_steps * 3)
    assert bool(done[0])
    assert int(env.done_reason[0]) in (DONE_STUCK, DONE_CRASH, DONE_REVERSE)


def test_moving_forward_is_not_flagged_stuck(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    done = drive(env, 0.0, 0.3, env.stuck_steps + 10)
    assert not bool(done[0]), env.done_reason


# --- 3, 4, 5, 12, 13: dynamics -------------------------------------------------------
def test_positive_steer_turns_left(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    env.heading = torch.tensor([0.0])
    env.speed = torch.tensor([10.0])
    env._drive(torch.tensor([[1.0, 0.0]]))
    for _ in range(5):
        env._drive(torch.tensor([[1.0, 0.0]]))
    assert float(env.heading[0]) > 0.0, "steer > 0 must increase heading (counter-clockwise, left)"
    assert float(env.pos[0, 1]) > 0.0


def test_ray_zero_looks_left(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    assert float(env.ray_angles[0]) > 0 > float(env.ray_angles[-1])
    assert abs(float(env.ray_angles[env.cfg.n_rays // 2])) < 1e-6


def test_no_reverse_gear_and_brake_stops(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    env._drive(torch.tensor([[0.0, -1.0]]))
    assert float(env.speed[0]) == 0.0, "braking from rest must not go backwards"
    env.speed = torch.tensor([30.0])
    env._drive(torch.tensor([[0.0, -1.0]]))
    assert 0.0 < float(env.speed[0]) < 30.0
    env.speed = torch.tensor([30.0])
    before = float(env.speed[0])
    env._drive(torch.tensor([[0.0, 0.0]]))
    assert float(env.speed[0]) < before, "coasting loses speed to drag"


def test_steering_actuator_lags_the_command(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    env._drive(torch.tensor([[1.0, 0.0]]))
    first = float(env.steer[0])
    assert 0.0 < first < FAST.max_steer_rad * 0.5
    for _ in range(60):
        env._drive(torch.tensor([[1.0, 0.0]]))
    assert float(env.steer[0]) > FAST.max_steer_rad * 0.99


def test_grip_circle_limits_lateral_g_and_gives_understeer(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    env.speed = torch.tensor([60.0])
    env.heading = torch.tensor([0.0])
    for _ in range(40):
        env._drive(torch.tensor([[1.0, 0.0]]))
    yaw_rate_free = 60.0 / FAST.wheelbase * np.tan(FAST.max_steer_rad)
    yaw_rate = float(env.heading[0]) / (40 * FAST.dt_s)
    assert yaw_rate < yaw_rate_free * 0.5, "full lock at 216 km/h must understeer"
    assert float(env.lat_g[0]) <= FAST.grip_max_g + 1e-3


def test_w11_straight_line_performance(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    speeds = []
    for _ in range(1500):
        env._drive(torch.tensor([[0.0, 1.0]]))
        speeds.append(float(env.speed[0]))
    v = np.asarray(speeds)
    t100 = float(np.argmax(v >= 100 / 3.6)) * FAST.dt_s
    assert 2.2 < t100 < 3.2, f"0-100 km/h in {t100:.2f}s"
    assert 300 < v.max() * 3.6 < 345, f"top speed {v.max() * 3.6:.0f} km/h"


def test_collision_uses_car_width(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    start, heading = env._start_pose(env.start_index)
    normal = torch.stack([-heading.sin(), heading.cos()], dim=1)[0]
    # body edge just off the wall but centre still on the tarmac
    env.pos = start + normal * (FAST.track_halfwidth - FAST.car_halfwidth * 0.5)
    env.speed = torch.tensor([1.0])
    _, _, done = env.step(torch.tensor([[0.0, 0.0]]))
    assert bool(done[0]) and int(env.done_reason[0]) == DONE_CRASH


def test_no_tunnelling_at_top_speed(loop_track):
    assert FAST.max_speed * FAST.dt_s < 2 * FAST.track_halfwidth, "one step must not cross the road"


# --- 6, 7: track fields ------------------------------------------------------------
def test_lidar_reports_nearer_wall_on_the_correct_side(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    env.reset()
    start, heading = env._start_pose(env.start_index)
    normal = torch.stack([-heading.sin(), heading.cos()], dim=1)[0]  # left of heading
    env.pos = start + normal * (FAST.track_halfwidth - 2.0)  # 2 m from the left wall
    obs = env.observe()[0]
    assert float(obs[0]) < float(obs[FAST.n_rays - 1]), "left ray must see the nearer left wall"
    assert float(obs[FAST.n_rays]) == 0.0


def test_lidar_range_and_resolution(loop_track):
    env = CarEnv(1, CPU, FAST, track=loop_track)
    assert float(env.march[-1]) == pytest.approx(FAST.max_range)
    assert float(env.march[0]) < 0.5, "first sample within half a metre of the body"


# --- 18, 19, 20: evolution -----------------------------------------------------------
def test_rank_normalise_handles_ties():
    assert torch.allclose(rank_normalise(torch.tensor([1.0, 1.0, 1.0])), torch.zeros(3))
    r = rank_normalise(torch.tensor([3.0, 1.0, 2.0]))
    assert torch.allclose(r, torch.tensor([0.5, -0.5, 0.0]))
    r = rank_normalise(torch.tensor([2.0, 1.0, 1.0, 5.0]))
    assert float(r[1]) == float(r[2])
    assert abs(float(r.sum())) < 1e-6


def test_curriculum_stages_tighten():
    widths = [s[0] for s in CURRICULUM]
    steps = [s[1] for s in CURRICULUM]
    assert widths == sorted(widths, reverse=True) and widths[-1] == 1.0
    assert steps == sorted(steps) and 2000 <= steps[0] and steps[-1] <= 6000


def test_track_bank_builds_wider_roads_for_early_stages():
    bank = TrackBank(FAST, "loop", 2, CPU)
    cfg0, t0 = bank.get(0, 0)
    cfg2, t2 = bank.get(len(CURRICULUM) - 1, 0)
    assert cfg0.track_halfwidth > cfg2.track_halfwidth
    assert t0 is not t2 and bank.get(0, 0)[1] is t0


def test_make_env_gives_each_member_its_own_starts():
    bank = TrackBank(FAST, "loop", 2, CPU)
    env = make_env(bank, 0, generation=0, popsize=2, starts_per_gen=3)
    assert env.batch == 6
    idx = env.start_index.tolist()
    assert idx[:3] == idx[3:], "members see the same set of starts"
    assert len(set(idx[:3])) == 3


# --- 24: stress tracks -------------------------------------------------------------
@pytest.mark.parametrize("difficulty", [1.0, 1.6])
def test_hard_loops_stay_above_the_minimum_turn_radius(difficulty):
    for seed in (101, 102, 103):
        pts = make_centerline(seed, scale=FAST.loop_scale, difficulty=difficulty).numpy()
        assert curvature_radius(pts).min() > FAST.min_turn_radius * 1.5


@needs_monaco
def test_mirrored_monaco_is_reversed():
    cfg = monaco_config(0.016)
    a = build_centerline(cfg).numpy()
    b = build_centerline(cfg.__class__(**{**cfg.__dict__, "mirror": True})).numpy()
    area = lambda p: 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])  # noqa: E731
    assert area(a) < 0 < area(b)


# --- 14, 20, 22: brain and agent -----------------------------------------------------
@needs_graph
def test_rollout_is_deterministic_given_a_seed():
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=2, config=LIFConfig(dt_ms=2.0), device=CPU, weight_scale=0.15)
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=2))
    theta = agent.unpack(agent.initial_params().unsqueeze(0).repeat(2, 1))
    env = CarEnv(2, CPU, FAST, seed=1)
    a = rollout(agent, env, theta, steps=6, seed=3)
    b = rollout(agent, env, theta, steps=6, seed=3)
    assert torch.equal(a["fitness"], b["fitness"])
    assert torch.equal(a["laps"], b["laps"])
    for key in ("speed_mean", "dn_hz", "steps_alive", "crash", "stuck"):
        assert key in a


@needs_graph
def test_brain_layout_matches_reference_update():
    """The fused (n, batch) step must reproduce a plain-python LIF update."""
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU, weight_scale=0.15)
    index = brain.role_index("visual_projection")
    torch.manual_seed(0)
    kicks = (torch.rand(index.numel(), 1) < 0.3).float() * 8.0
    spikes1 = brain.step(external_index=index, external_values=kicks)
    assert spikes1.shape == (1, brain.n)
    assert float(spikes1.sum()) > 0
    # membrane of a kicked, non-spiking cell: decay_v * 0 + 8 mV = 8 mV above rest
    v = brain.v[0]
    kicked = index[kicks[:, 0] > 0]
    quiet = kicked[~spikes1[0, kicked].bool()]
    assert torch.allclose(v[quiet], torch.full_like(v[quiet], -52.0 + 8.0), atol=1e-4)


@needs_graph
def test_agent_loom_channel_responds_to_approach():
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU)
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=1))
    params = agent.initial_params()
    theta = agent.unpack(params.unsqueeze(0))
    theta["loom_gain"] = torch.ones_like(theta["loom_gain"])
    # road-relative eyes: 150 m reads as open road, 7.5 m as a wall closer than expected on every ray
    far = torch.cat([torch.ones(1, 9), torch.tensor([[0.5]])], dim=1)
    near = torch.cat([torch.full((1, 9), 0.05), torch.tensor([[0.5]])], dim=1)
    agent.reset()
    agent.sensory_rates(far, theta)
    approaching = agent.sensory_rates(near, theta)[0, :9]
    agent.reset()
    agent.sensory_rates(near, theta)
    steady = agent.sensory_rates(near, theta)[0, :9]
    assert bool((approaching > steady).all())


@needs_graph
def test_readout_keeps_a_steady_signal_and_is_not_saturated_at_init():
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU, weight_scale=0.15)
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=4))
    theta = agent.unpack(agent.initial_params().unsqueeze(0))
    obs = torch.cat([torch.ones(1, 9), torch.tensor([[0.3]])], dim=1)
    obs[0, :4] = 0.1  # wall on the left
    agent.reset()
    agent.seed(1)
    steers = [float(agent.act(obs, theta)[0, 0]) for _ in range(12)]
    assert max(abs(s) for s in steers) < 0.95, "initial readout must not pin tanh"
    assert agent.fraction_at_bounds(agent.initial_params()) < 0.5


# --- uncapped episodes -------------------------------------------------------------
def test_episode_ends_as_finished_after_max_laps(loop_track):
    from car_env import DONE_FINISH

    cfg = replace_cfg(loop_track.cfg, max_laps=0.02)
    env = CarEnv(1, CPU, cfg, track=loop_track)
    env.reset()
    done = drive(env, 0.0, 1.0, 400)
    assert bool(done.all())
    assert int(env.done_reason[0]) == DONE_FINISH
    assert float(env.last_terms["alive"][0]) == 0.0
    assert env.telemetry()["finished"] == 1


def test_finishing_is_not_penalised_like_a_crash(loop_track):
    cfg = replace_cfg(loop_track.cfg, max_laps=0.02)
    env = CarEnv(1, CPU, cfg, track=loop_track)
    env.reset()
    reward = None
    for _ in range(400):
        _, reward, done = env.step(torch.tensor([[0.0, 1.0]]))
        if bool(done.all()):
            break
    assert reward is not None and float(reward[0]) > -cfg.crash_penalty / 2


def test_step_range_is_uncapped_by_default():
    import defaults

    assert len(defaults.step_range(0)) == defaults.EPISODE_HARD_CAP
    assert len(defaults.step_range(-1)) == defaults.EPISODE_HARD_CAP
    assert len(defaults.step_range(120)) == 120
    from train import episode_cap

    assert episode_cap(-1, 0) == 0, "-1 is the uncapped default"
    assert episode_cap(0, 0) == CURRICULUM[0][1], "0 keeps the curriculum's caps"
    assert episode_cap(777, 2) == 777


def replace_cfg(cfg, **kw):
    from dataclasses import replace

    return replace(cfg, **kw)
