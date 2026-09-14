"""Summarise training progress from logs/train.jsonl.

  python3 src/monitor.py --plot logs/curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no log at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    args = parser.parse_args(argv)

    records = read_log(args.log)
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
