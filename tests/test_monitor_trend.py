"""The training-curve verdict must call a climb a climb and noise noise."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitor import slope, spearman, verdict  # noqa: E402


def test_slope_is_the_change_per_generation():
    assert slope([0.0, 2.0, 4.0, 6.0]) == 2.0
    assert slope([5.0]) == 0.0


def test_a_rising_curve_is_improving_even_with_noise():
    rising = [0.0, 1.0, 0.5, 2.0, 1.8, 3.0, 2.9, 4.0]
    assert spearman(rising) > 0.8
    assert verdict(rising) == "IMPROVING"


def test_a_flat_curve_with_outliers_is_not_a_trend():
    flat = [1.0, 1.2, 0.8, 1.1, 0.9, 5.0, 0.95, 1.05]
    assert verdict(flat) == "FLAT"


def test_a_constant_curve_is_flat_not_improving():
    """Ties must share a rank, or a dead-flat series reads as a perfect climb."""
    assert spearman([2.0] * 8) == 0.0
    assert verdict([2.0] * 8) == "FLAT"


def test_a_falling_curve_is_regressing_and_short_ones_are_not_judged():
    assert verdict([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]) == "REGRESSING"
    assert verdict([1.0, 2.0]) == "TOO SHORT"
