"""Alive-body compaction and the functional masked reset.

The trainer drops finished bodies from the brain, the environment and the
exploit monitor mid-episode (`train.rollout`). Dropping a body must not change
what the survivors compute: the same spikes, membrane state, poses and rewards
as the uncompacted run, in the original body order once scattered back.
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
from exploits import ExploitMonitor  # noqa: E402
from train import rollout  # noqa: E402

GRAPH = ROOT / "data" / "graph_w5"
needs_graph = pytest.mark.skipif(not (GRAPH / "edges.npz").exists(), reason="run src/build_graph.py first")
needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal kernel needs an MPS device")
CPU = torch.device("cpu")
MPS = torch.device("mps")


# --- environment -----------------------------------------------------------------------


def test_masked_reset_after_inference_mode_step():
    """The viewer resets single cars between steps run under inference mode."""
    env = CarEnv(4, CPU, CarConfig(dt_s=0.016), seed=3)
    with torch.inference_mode():
        for _ in range(5):
            env.step(torch.tensor([[0.3, 1.0]] * 4))
    before = env.pos.clone()
    mask = torch.tensor([False, True, False, False])
    obs = env.reset(mask)  # used to raise "Inplace update to inference tensor outside InferenceMode"
    start, heading = env._start_pose(env.start_index[1:2])
    assert torch.allclose(env.pos[1], start[0])
    assert torch.allclose(env.heading[1:2], heading)
    assert float(env.speed[1]) == 0.0 and int(env.step_count[1]) == 0
    assert torch.equal(env.pos[[0, 2, 3]], before[[0, 2, 3]])
    assert obs.shape[0] == 4


def test_env_compact_keeps_survivors_in_order():
    env = CarEnv(6, CPU, CarConfig(dt_s=0.016), seed=1, start_fraction=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    torch.manual_seed(0)
    for _ in range(8):
        env.step(torch.rand(6, 2) * 2 - 1)
    keep = torch.tensor([True, False, True, True, False, True])
    pos, laps, start = env.pos.clone(), env.laps.clone(), env.start_index.clone()
    env.compact(keep)
    assert env.batch == 4
    assert torch.equal(env.pos, pos[keep])
    assert torch.equal(env.laps, laps[keep])
    assert torch.equal(env.start_index, start[keep])
    assert env.observe().shape[0] == 4
    obs, reward, done = env.step(torch.zeros(4, 2))
    assert obs.shape[0] == 4 and reward.shape == (4,) and done.shape == (4,)


def test_exploit_monitor_compact_banks_dropped_verdicts():
    env = CarEnv(5, CPU, CarConfig(dt_s=0.016), seed=2)
    watch = ExploitMonitor(env)
    alive = torch.ones(5, dtype=torch.bool)
    for _ in range(4):
        _, reward, _ = env.step(torch.tensor([[0.0, 1.0]] * 5))
        watch.observe(reward, alive)
    full = watch.report()
    keep = torch.tensor([True, True, False, True, False])
    watch.compact(keep)
    env.compact(keep)
    part = watch.report()
    assert part["cars"] == full["cars"] == 5
    assert part["flags"] == full["flags"]
    assert part["flagged_cars"] == full["flagged_cars"]
    assert watch.gross.shape == (3,)


# --- brain -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def connectome():
    return load_connectome(GRAPH)


def _drive(brain: Brain, cells: torch.Tensor, prob: torch.Tensor, seed: int) -> None:
    brain.step(external_index=cells, poisson_prob=prob, seed=seed, shared_noise=True, dense=False)


def _compare_after_compaction(connectome, device: torch.device, use_metal: bool, batch: int, keep: torch.Tensor) -> None:
    kw = dict(config=LIFConfig(dt_ms=2.0), device=device, weight_scale=0.15, use_metal=use_metal)
    if use_metal:
        kw["precision"] = "fp32"
    full = Brain(connectome, batch=batch, **kw)
    part = Brain(connectome, batch=batch, **kw)
    for b in (full, part):
        b.set_kick_mv(8.0)
        b.set_dn_index(b.role_index("descending", 64))
    cells = full.role_index("visual_projection", 1500).to(torch.long)
    gen = torch.Generator().manual_seed(11)
    prob = (torch.rand(cells.numel(), batch, generator=gen) * 0.6).to(device)
    for s in range(6):
        _drive(full, cells, prob, seed=100 + s)
        _drive(part, cells, prob, seed=100 + s)
    keep = keep.to(device)
    part.compact(keep)
    assert part.batch == int(keep.sum())
    assert part.u.shape == (part.n, part.batch)
    prob_part = prob[:, keep]
    for s in range(6):
        _drive(full, cells, prob, seed=200 + s)
        _drive(part, cells, prob_part, seed=200 + s)
        assert torch.equal(part.dense_spikes(), full.dense_spikes()[keep])
    assert torch.equal(part.u, full.u[:, keep])
    assert torch.equal(part.j_syn, full.j_syn[:, keep])
    assert torch.equal(part.dn_acc, full.dn_acc[keep])
    part.reset()
    assert part.batch == batch and part.u.shape == (part.n, batch)


@needs_graph
@needs_mps
def test_metal_compaction_matches_full_batch(connectome):
    keep = torch.zeros(70, dtype=torch.bool)
    keep[[0, 3, 7, 8, 20, 31, 32, 33, 45, 63, 64, 69]] = True
    _compare_after_compaction(connectome, MPS, True, 70, keep)


@needs_graph
def test_torch_compaction_matches_full_batch(connectome):
    keep = torch.tensor([True, False, False, True])
    _compare_after_compaction(connectome, CPU, False, 4, keep)


# --- rollout ---------------------------------------------------------------------------


@needs_graph
@needs_mps
def test_rollout_with_compaction_matches_uncompacted(connectome):
    batch = 12
    brain = Brain(connectome, batch=batch, config=LIFConfig(dt_ms=2.0), device=MPS, weight_scale=0.15, precision="fp32")
    agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=8))
    cfg = CarConfig(dt_s=0.016)

    def theta_for(n: int) -> dict[str, torch.Tensor]:
        params = agent.initial_params().to(MPS).repeat(n, 1)
        theta = agent.unpack(params)
        # Steer biases from hard left to hard right: cars leave the road at different times.
        theta["b_out"] = theta["b_out"].clone()
        theta["b_out"][:, 0] = torch.linspace(-4.0, 4.0, n, device=MPS)
        theta["b_out"][:, 1] = 3.0
        return theta

    fractions = [(k % 4) / 4 for k in range(batch)]
    env_a = CarEnv(batch, MPS, cfg, seed=5, start_fraction=fractions)
    ref = rollout(agent, env_a, theta_for(batch), steps=240, seed=9, poll_every=10, compact_below=0.0)
    env_b = CarEnv(batch, MPS, cfg, seed=5, start_fraction=fractions)
    out = rollout(agent, env_b, theta_for(batch), steps=240, seed=9, poll_every=10, compact_below=1.0, compact_min_drop=1)

    assert out["compactions"] > 0, "test needs cars that finish before the cap"
    assert out["body_steps"] < ref["body_steps"]
    assert torch.equal(out["done_reason"], ref["done_reason"])
    assert torch.equal(out["steps_alive"], ref["steps_alive"])
    assert torch.allclose(out["fitness"], ref["fitness"], atol=1e-4)
    assert torch.allclose(out["laps"], ref["laps"], atol=1e-6)
    assert torch.equal(out["start_index"], ref["start_index"])
    for key in ("crash", "reverse", "stuck", "finished", "alive"):
        assert out[key] == ref[key]
    assert out["exploits"]["cars"] == ref["exploits"]["cars"] == batch
    assert brain.batch == batch  # rollout's final reset restores the full population
