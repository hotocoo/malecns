"""`train.rollout` and `train.evaluate_mean` plumbing at population scale, without a brain.

A stub agent drives like the linear teacher with per-body noise, so bodies end
at different times and the rollout compacts several times. This pins two
things the connectome tests are too slow to cover at 1,536 bodies:

* every per-body tensor (environment, monitor, agent, theta) stays the same
  size through compaction, and a following rollout on a differently sized
  environment starts from a brain reset to *that* size. The trainer used to
  build its evaluation environment from `agent.brain.batch`, which after a
  compacted generation was a leftover like 859 while the expanded means had
  4 x 214 = 856 rows: every evaluation crashed, no checkpoint was ever saved
  past generation 0.
* the exploit monitor's reward accounting matches the environment exactly for
  honest driving under the step budget, pace and alignment terms.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train  # noqa: E402
from car_env import CarConfig, CarEnv, Track, build_centerline  # noqa: E402
from teacher import LinearTeacher  # noqa: E402

CPU = torch.device("cpu")
FAST = CarConfig(grid_res=1024)


class StubBrain:
    def __init__(self, batch: int) -> None:
        self.full_batch = batch
        self.batch = batch
        self.n = 10

    def reset(self, batch: int | None = None) -> None:
        self.batch = self.full_batch if batch is None else int(batch)

    def compact(self, keep: torch.Tensor) -> None:
        if keep.numel() != self.batch:
            raise ValueError(f"keep has {keep.numel()} entries for {self.batch} bodies")
        self.batch = int(keep.sum())


from car_env import CarConfig as _CarConfig  # noqa: E402


class StubCfg:
    n_rays = _CarConfig.n_rays


class StubAgent:
    """Linear-teacher driver with a per-body offset, exposing what `rollout` touches."""

    def __init__(self, batch: int) -> None:
        self.brain = StubBrain(batch)
        self.cfg = StubCfg()
        self.teacher = LinearTeacher()
        self.gen = torch.Generator().manual_seed(0)

    def seed(self, seed: int) -> None:
        self.gen = torch.Generator().manual_seed(seed)

    def reset(self, batch: int | None = None) -> None:
        self.brain.reset(batch)
        b = self.brain.batch
        self.last_motor = torch.zeros(b, 2)
        self.last_rates_hz = torch.zeros(b, self.cfg.n_rays + 1)
        self.dn_rate_hz = torch.zeros(b, 4)
        self.noise = (torch.rand(b, 2, generator=self.gen) - 0.5) * 0.6

    def compact(self, keep: torch.Tensor) -> None:
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        for name in ("last_motor", "last_rates_hz", "dn_rate_hz", "noise"):
            value = getattr(self, name)
            if value.shape[0] != self.brain.batch:
                raise AssertionError(f"{name} has {value.shape[0]} rows for {self.brain.batch} bodies")
            setattr(self, name, value[idx])
        self.brain.compact(keep)

    def sensory_rates(self, obs: torch.Tensor, theta: dict) -> torch.Tensor:
        n = self.cfg.n_rays
        prox = (1.0 - obs[:, :n]).clamp(0.0, 1.0)
        return torch.cat([prox * 100.0, obs[:, n : n + 1] * 100.0], dim=1)

    def act(self, obs: torch.Tensor, theta: dict) -> torch.Tensor:
        if obs.shape[0] != theta["ray_gain"].shape[0] or obs.shape[0] != self.brain.batch:
            raise AssertionError(f"obs {obs.shape[0]}, theta {theta['ray_gain'].shape[0]}, brain {self.brain.batch}")
        n = self.cfg.n_rays
        prox = (1.0 - obs[:, :n]).clamp(0.0, 1.0)
        action = (self.teacher.act(prox, obs[:, n]) + self.noise * theta["ray_gain"][:, :2]).clamp(-1.0, 1.0)
        self.last_motor = action
        self.last_rates_hz = self.sensory_rates(obs, theta)
        return action


def make_env(track: Track, cfg: CarConfig, batch: int, steps: int) -> CarEnv:
    fractions = torch.tensor([(b % 6) / 6 for b in range(6)]).repeat(batch // 6 + 1)[:batch]
    return CarEnv(batch, CPU, replace(cfg, episode_steps=steps), track=track, start_fraction=fractions)


def test_compacted_rollout_then_smaller_rollout_keeps_every_tensor_in_step():
    track = Track(build_centerline(FAST, 1), FAST, CPU)
    agent = StubAgent(384)
    steps = 900
    env = make_env(track, FAST, 384, steps)
    theta = {"ray_gain": torch.ones(384, agent.cfg.n_rays)}
    out = train.rollout(agent, env, theta, steps, seed=3, compact_min_drop=8)
    assert out["compactions"] > 0, "the noisy population must end at different times for this test to bite"
    assert out["fitness"].shape == (384,) and out["laps"].shape == (384,)
    assert agent.brain.batch < 384, "the brain is left compacted after a rollout, as in the trainer"
    # The next rollout is smaller (24 bodies, as `evaluate_mean` now runs): the agent must be
    # reset to *its* size, not to the leftover compacted batch and not to the full batch.
    small = make_env(track, FAST, 24, 300)
    out2 = train.rollout(agent, small, {"ray_gain": torch.ones(24, agent.cfg.n_rays)}, 300, seed=4)
    assert out2["fitness"].shape == (24,)
    assert agent.brain.batch <= 24


def test_monitor_accounting_is_exact_for_honest_driving_at_scale():
    track = Track(build_centerline(FAST, 2), FAST, CPU)
    agent = StubAgent(240)
    steps = 600
    env = make_env(track, FAST, 240, steps)
    out = train.rollout(agent, env, {"ray_gain": torch.ones(240, agent.cfg.n_rays)}, steps, seed=5, compact_min_drop=8)
    flags = out["exploits"]["flags"]
    assert flags["accounting"] == 0, flags
    assert flags["over_bound"] == 0 and flags["bonus_farming"] == 0 and flags["idle_reward"] == 0, flags
    assert out["exploits"]["max_reward_over_bound"] < 0.0


def test_eval_env_has_one_body_per_island_and_start():
    bank = train.TrackBank(FAST, "loop", 6, CPU)
    env = train.make_eval_env(bank, stage=2, islands=4, starts_per_gen=6, steps=1200)
    assert env.batch == 24
    assert env.cfg.episode_steps == 1200
    slots = torch.arange(24) % 6
    for s in range(6):
        starts = env.start_index[slots == s]
        assert bool((starts == starts[0]).all()), "each island sees the same start in each slot"
