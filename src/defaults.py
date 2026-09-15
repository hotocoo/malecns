"""Timestep and episode constants shared by the trainer, evaluator and viewer.

The control period is `DT_MS * SUBSTEPS` milliseconds of brain time. A policy
evolved at one control period behaves differently at another, so every entry
point reads these from one place, and `evaluate.py` / `viewer.py` take the
values stored in the checkpoint over these when a checkpoint is given.
"""

DT_MS = 2.0
SUBSTEPS = 8
CONTROL_DT_S = DT_MS * SUBSTEPS / 1000.0  # 16 ms, 62.5 Hz
EPISODE_STEPS = 0  # no cap: an episode ends when the car crashes, stalls, reverses or finishes `max_laps`
EVAL_STEPS = 0  # same for evaluation
# Safety ceiling for uncapped episodes (67 min of sim time at 16 ms); a policy
# that neither ends nor completes `max_laps` in this time is a bug, not a driver.
EPISODE_HARD_CAP = 250_000
WEIGHT_SCALE = 0.15
ADAPT_MV = 0.6
MONACO_STARTS = 6


def control_dt_s(dt_ms: float, substeps: int) -> float:
    return dt_ms * substeps / 1000.0


def step_range(steps: int) -> range:
    """Steps to run: `steps` when positive, otherwise the uncapped episode's safety ceiling."""
    return range(steps) if steps > 0 else range(EPISODE_HARD_CAP)
