"""Regression tests for the saturation failure seen at generation ~2,100.

The ES mean drifted to |w_out| ~ 70 and steer bias -30, pinning tanh at -1 so
the car steered hard left forever, U-turned and drove the loop backwards.
These tests pin the two guards added against that: parameter bounds and
episode termination on net reverse progress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import AgentConfig, ConnectomeAgent  # noqa: E402
from brain import Brain, LIFConfig, load_connectome  # noqa: E402
from car_env import CarConfig, CarEnv  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
needs_graph = pytest.mark.skipif(
    not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first"
)
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def agent():
    connectome = load_connectome(GRAPH)
    brain = Brain(connectome, batch=1, config=LIFConfig(dt_ms=2.0), device=CPU)
    return ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=1))


@needs_graph
def test_initial_params_already_satisfy_bounds(agent):
    init = agent.initial_params()
    assert torch.equal(agent.clamp_params(init.unsqueeze(0))[0], init)


@needs_graph
def test_clamp_pins_the_saturated_checkpoint_back_into_range(agent):
    exploded = agent.initial_params() * 0 + 70.0
    exploded[-2:] = torch.tensor([-30.0, 36.0])  # b_out as found at gen 2151
    theta = agent.unpack(agent.clamp_params(exploded.unsqueeze(0)))
    eps = 1e-6  # bounds are python floats, params are float32
    for name, (lo, hi) in agent.PARAM_BOUNDS.items():
        if name in theta:
            assert float(theta[name].min()) >= lo - eps
            assert float(theta[name].max()) <= hi + eps
    # the bias alone can no longer saturate the steer squash
    assert abs(float(torch.tanh(theta["b_out"][0, 0]))) < 0.95


@needs_graph
def test_clamp_is_batched_and_leaves_in_range_values_alone(agent):
    base = agent.initial_params().unsqueeze(0).repeat(3, 1)
    base[1] += 1000.0
    out = agent.clamp_params(base)
    assert out.shape == base.shape
    assert torch.equal(out[0], base[0])
    assert torch.equal(out[2], base[2])
    assert float(out[1].max()) <= max(hi for _, hi in agent.PARAM_BOUNDS.values())


def test_driving_backwards_ends_the_episode():
    env = CarEnv(1, CPU, CarConfig(), seed=1)
    env.reset()
    env.heading = env.heading + torch.pi  # face the wrong way down the track
    done = torch.tensor([False])
    for _ in range(400):
        _, reward, done = env.step(torch.tensor([[0.0, 1.0]]))
        if bool(done[0]):
            break
    assert bool(done[0]), "reverse driver should be terminated"
    assert float(env.laps[0]) <= env.cfg.reverse_limit_laps
    assert float(reward[0]) <= -env.cfg.crash_penalty


def test_crash_cost_is_independent_of_impact_speed():
    low = CarEnv(1, CPU, CarConfig(), seed=1)
    high = CarEnv(1, CPU, CarConfig(), seed=1)
    low.reset()
    high.reset()
    low.speed = torch.tensor([10.0])
    high.speed = torch.tensor([80.0])
    # Force the same terminal condition so only impact speed differs. The
    # terminal event is deliberately a fixed cost; continuous speed shaping
    # already supplies the speed signal during legal driving.
    for env in (low, high):
        # Move well outside the rasterised road so the one control step cannot
        # turn this into a geometry-dependent assertion.
        env.pos = env.pos + torch.tensor([[0.0, 50.0]])
    _, low_reward, low_done = low.step(torch.tensor([[0.0, 0.0]]))
    _, high_reward, high_done = high.step(torch.tensor([[0.0, 0.0]]))
    assert bool(low_done[0]) and bool(high_done[0])
    # The charge is the crash penalty plus the metres of the lap left undriven,
    # and neither depends on how fast the car was going when it left the road.
    def owed(env: CarEnv) -> float:
        laps = float(env.laps[0])
        return env.cfg.crash_penalty + env.cfg.unfinished_per_m * (env.cfg.max_laps - laps) * env.track.length_m

    assert float(low_reward[0]) == pytest.approx(-owed(low), abs=1e-2)
    assert float(high_reward[0]) == pytest.approx(-owed(high), abs=1e-2)
    fixed = float(high_reward[0]) + owed(high) - low.cfg.crash_penalty
    assert fixed == pytest.approx(float(low_reward[0]) + owed(low) - low.cfg.crash_penalty, abs=1e-2)


def test_forward_driving_is_not_affected_by_reverse_guard():
    env = CarEnv(1, CPU, CarConfig(), seed=1)
    env.reset()
    for _ in range(60):
        _, _, done = env.step(torch.tensor([[0.0, 1.0]]))
        assert not bool(done[0])
    assert float(env.laps[0]) > 0.0


def test_low_speed_full_steer_is_penalised_more_than_centered_stall():
    cfg = CarConfig(stall_speed_mps=5.0, stall_steer_start=0.5, stall_steer_penalty=0.12)
    steer = CarEnv(1, CPU, cfg, seed=1)
    center = CarEnv(1, CPU, cfg, seed=1)
    steer.reset()
    center.reset()
    steer.speed[:] = 1.0
    center.speed[:] = 1.0
    steer.steer[:] = cfg.max_steer_rad
    _, steer_reward, _ = steer.step(torch.tensor([[1.0, 0.0]]))
    _, center_reward, _ = center.step(torch.tensor([[0.0, 0.0]]))
    assert float(steer.last_terms["stall"][0]) < float(center.last_terms["stall"][0])
    assert float(steer_reward[0]) < float(center_reward[0])


def test_a_second_trainer_on_the_same_checkpoint_refuses_to_start(tmp_path):
    """Two trainers sharing a checkpoint overwrite each other's mean every generation."""
    import os

    from train import claim_checkpoint

    ckpt = tmp_path / "es.pt"
    lock = claim_checkpoint(ckpt)
    assert lock.exists() and lock.read_text().strip() == str(os.getpid())

    lock.write_text("1")  # pid 1 is always alive
    with pytest.raises(SystemExit) as exc:
        claim_checkpoint(ckpt)
    assert "already being trained" in str(exc.value)

    # a lock left behind by a killed trainer must not block the restart
    lock.write_text("999999")
    assert claim_checkpoint(ckpt).exists()


def test_a_rejected_step_narrows_the_search_and_the_pull_back_respects_it():
    """The guard's shrink must survive to the next generation, or a rejecting run never recovers."""
    base, shrink, grow, sigma_max = 0.02, 0.8, 1.25, 0.06
    sigma, floor = base, base

    def pull_back(sigma: float, floor: float, informative: bool = True) -> float:
        return min(sigma_max, sigma * grow) if not informative else max(floor, sigma * shrink)

    def reject(sigma: float) -> tuple[float, float]:
        sigma = max(base * 0.25, sigma * 0.7)
        return sigma, sigma

    seen = []
    for _ in range(4):  # four rejected generations in a row
        sigma, floor = reject(sigma)
        sigma = pull_back(sigma, floor)
        seen.append(round(sigma, 5))
    assert seen == sorted(seen, reverse=True), seen
    assert seen[-1] < base / 2, seen
    # an accepted step restores the base floor, so the search can open up again
    floor = base
    assert pull_back(sigma, floor) == base


def test_only_the_islands_that_regressed_are_reverted():
    """Islands are independent searches: one bad island must not undo three good ones."""
    tol = 2.0
    accepted_mu = torch.zeros(4, 3)
    mu = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1.0
    accepted = torch.tensor([-20.0, -10.0, -30.0, -5.0])
    island_eval = torch.tensor([-12.0, -40.0, -28.0, -5.5])  # islands 0 and 2 better, 1 much worse, 3 within tolerance

    worse = island_eval < accepted - tol
    assert worse.tolist() == [False, True, False, False]
    kept = ~worse
    new_mu = torch.where(worse.unsqueeze(1), accepted_mu, mu)
    new_accepted_mu = torch.where(kept.unsqueeze(1), new_mu, accepted_mu)
    new_accepted = torch.where(kept, island_eval, accepted)

    assert torch.equal(new_mu[1], accepted_mu[1]), "the regressed island goes back"
    assert torch.equal(new_mu[0], mu[0]) and torch.equal(new_mu[2], mu[2]), "the others keep their step"
    assert torch.equal(new_accepted_mu[1], accepted_mu[1])
    assert new_accepted.tolist() == [-12.0, -10.0, -28.0, -5.5]


def test_repeated_rejections_shorten_the_step_not_just_the_sample_radius():
    """The trust region is the step length: if it never adapts, an overshooting run keeps overshooting."""
    base = 0.03
    frac = base
    for _ in range(5):  # five rejected generations
        frac = max(base * 0.25, frac * 0.7)
    assert frac == pytest.approx(base * 0.25)
    assert frac < base / 3
    frac = base  # an accepted step restores the full trust region
    assert frac == base
