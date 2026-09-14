"""Timestep and episode constants shared by the trainer, evaluator and viewer.

The control period is `DT_MS * SUBSTEPS` milliseconds of brain time. A policy
evolved at one control period behaves differently at another, so every entry
point reads these from one place, and `evaluate.py` / `viewer.py` take the
values stored in the checkpoint over these when a checkpoint is given.
"""

DT_MS = 2.0
SUBSTEPS = 8
CONTROL_DT_S = DT_MS * SUBSTEPS / 1000.0  # 16 ms, 62.5 Hz
EPISODE_STEPS = 1500  # 24 s of driving
EVAL_STEPS = 3000  # long-horizon evaluation, 48 s
WEIGHT_SCALE = 0.15
ADAPT_MV = 0.6
MONACO_STARTS = 6


def control_dt_s(dt_ms: float, substeps: int) -> float:
    return dt_ms * substeps / 1000.0
