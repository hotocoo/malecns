"""Summarise training progress from logs/train.jsonl.

  python3 src/monitor.py --plot logs/curve.png
  python3 src/monitor.py --log logs/verify/newreward.jsonl --since 1

`--trend` answers the only question that matters mid-run: is the driver
actually getting better, or is the curve noise? Each series gets a
least-squares slope per generation and a Spearman rank correlation, which
together separate a real climb from a flat line with outliers. Fitness is only
comparable within one reward, so use `--since` to cut the log at the
generation where the reward last changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no log at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def slope(values: list[float]) -> float:
    """Least-squares change per generation."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    var = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / var if var else 0.0


def spearman(values: list[float]) -> float:
    """Rank correlation with the generation number, in [-1, 1]."""
    n = len(values)
    if n < 3:
        return 0.0
    order = sorted(range(n), key=lambda i: values[i])
    rank = [0.0] * n
    i = 0
    while i < n:  # average the ranks of ties, or a flat series looks like a trend
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2
        for k in range(i, j + 1):
            rank[order[k]] = mean_rank
        i = j + 1
    mx = (n - 1) / 2
    mr = sum(rank) / n
    num = sum((x - mx) * (r - mr) for x, r in zip(range(n), rank))
    den = (sum((x - mx) ** 2 for x in range(n)) * sum((r - mr) ** 2 for r in rank)) ** 0.5
    return num / den if den else 0.0


def verdict(values: list[float]) -> str:
    """IMPROVING / FLAT / REGRESSING from the rank correlation."""
    rho = spearman(values)
    if len(values) < 5:
        return "TOO SHORT"
    if rho > 0.4:
        return "IMPROVING"
    if rho < -0.4:
        return "REGRESSING"
    return "FLAT"


def rolling(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=Path("logs/train.jsonl"), type=Path)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--since", type=int, default=0, help="drop the first N records (use after a reward change)")
    parser.add_argument("--no-trend", action="store_true")
    args = parser.parse_args(argv)

    records = read_log(args.log)[args.since :]
    if not records:
        raise SystemExit("no records left after --since")
    mean = [r["fitness_mean"] for r in records]
    best = [r["fitness_best"] for r in records]
    laps = [r["laps_best"] for r in records]
    seconds = sum(r["seconds"] for r in records)
    smooth = rolling(mean, args.window)

    print(f"generations       {len(records)}")
    print(f"compute           {seconds / 3600:.2f} h")
    print(f"fitness now       {smooth[-1]:.2f} (rolling {args.window})")
    print(f"fitness start     {smooth[min(args.window, len(smooth)) - 1]:.2f}")
    print(f"best ever         {max(best):.2f} at generation {best.index(max(best)) + 1}")
    print(f"best lap fraction {max(laps):.3f}")
    accepted = [r.get("accepted_eval") for r in records if r.get("accepted_eval") is not None]
    if accepted:
        steps = sum(1 for a, b in zip(accepted, accepted[1:]) if b > a)
        print(f"accepted mean     {accepted[0]:.2f} -> {accepted[-1]:.2f} in {steps} accepted step(s) of {len(records)} generations")

    if not args.no_trend:
        series = {
            # The trainer only ever keeps a mean whose deterministic evaluation
            # beat the accepted one (`--eval-tolerance`), so this ratchet, not
            # the per-generation evaluation, is the driver that actually ships.
            "accepted eval (ratchet)": [r.get("accepted_eval") for r in records],
            "eval fitness (best island)": [r.get("eval_fitness_best_island") for r in records],
            "eval fitness (mean)": [r.get("eval_fitness") for r in records],
            "eval laps": [r.get("eval_laps") for r in records],
            "eval lap time (s)": [
                r["eval_lap_steps_best_island"] * 0.016 if r.get("eval_lap_steps_best_island") else None for r in records
            ],
            "population fitness": [r.get("fitness_mean") for r in records],
            "population laps (best)": [r.get("laps_best") for r in records],
            "speed (km/h)": [r["speed_mean"] * 3.6 if r.get("speed_mean") is not None else None for r in records],
        }
        print()
        print(f"{'series':<24}{'n':>4}{'first':>10}{'last':>10}{'per gen':>10}{'rho':>7}  verdict")
        for name, raw in series.items():
            values = [float(v) for v in raw if v is not None]
            if len(values) < 2:
                print(f"{name:<24}{len(values):>4}{'-':>10}{'-':>10}{'-':>10}{'-':>7}  NO DATA")
                continue
            # Lower is better for lap time: report it as such in the verdict.
            call = verdict([-v for v in values]) if "lap time" in name else verdict(values)
            print(
                f"{name:<24}{len(values):>4}{values[0]:>10.3f}{values[-1]:>10.3f}"
                f"{slope(values):>10.4f}{spearman(values):>7.2f}  {call}"
            )

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(mean, lw=0.6, alpha=0.4, color="#7f8c8d", label="generation mean")
        ax.plot(smooth, lw=2.0, color="#c0392b", label=f"rolling {args.window}")
        ax.set_xlabel("generation")
        ax.set_ylabel("episode fitness")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"[plot] {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
