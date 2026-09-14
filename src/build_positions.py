"""Attach real anatomical soma coordinates to the compiled connectome graph.

The MaleCNS annotation table carries a `somaLocation` voxel coordinate for most
cells. Those coordinates are what makes a whole-brain view anatomical rather
than decorative, so they are pulled out once and cached alongside the graph:

  python3 src/build_positions.py --graph data/graph_w5

Writes `positions.npy` (n, 3) float32 in a centred, unit-scaled frame, and
`positions_known.npy` (n,) bool marking which of those are measured.

About 16% of cells have no soma location in the release (optic-lobe cells
annotated only by hex coordinate, plus fragments). Dropping them would punch
holes in exactly the visual pathway this task drives, so they are placed at the
mean position of their synaptic partners, iterated a few times. That is a
layout, not a measurement, which is why the mask is saved next to it.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Voxel size of the MaleCNS EM volume. Only the ratio matters for a view, but
# using the real value keeps the cloud in proportion.
VOXEL_NM = 8.0


def soma_table(raw_dir: Path) -> pd.DataFrame:
    annotations = pd.read_feather(
        raw_dir / "body-annotations.feather", columns=["bodyId", "somaLocation"]
    )
    annotations = annotations[annotations["somaLocation"].notna()]
    coords = np.stack(annotations["somaLocation"].to_numpy()).astype(np.float32)
    return pd.DataFrame(
        {
            "bodyId": annotations["bodyId"].to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
        }
    )


def infer_missing(
    pos: np.ndarray,
    known: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    rounds: int,
) -> np.ndarray:
    """Place unlocated cells at the mean position of their located partners."""
    pos = pos.copy()
    placed = known.copy()
    for _ in range(rounds):
        total = np.zeros_like(pos)
        count = np.zeros(len(pos), dtype=np.float32)
        for source, target in ((pre, post), (post, pre)):
            usable = placed[source] & ~placed[target]
            np.add.at(total, target[usable], pos[source[usable]])
            np.add.at(count, target[usable], 1.0)
        newly = count > 0
        if not newly.any():
            break
        pos[newly] = total[newly] / count[newly, None]
        placed = placed | newly
    if not placed.all():
        pos[~placed] = pos[placed].mean(axis=0)
    return pos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default=Path("data/graph_w5"), type=Path)
    parser.add_argument("--raw", default=Path("data/raw"), type=Path)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args(argv)

    neurons = pd.read_parquet(args.graph / "neurons.parquet", columns=["bodyId"])
    edges = np.load(args.graph / "edges.npz")
    soma = soma_table(args.raw)

    merged = neurons.merge(soma, on="bodyId", how="left")
    pos = np.array(merged[["x", "y", "z"]].to_numpy(dtype=np.float32), copy=True)
    known = np.isfinite(pos).all(axis=1)
    pos[~known] = 0.0
    print(f"{known.sum():,} of {len(known):,} neurons have a measured soma location")

    pos = infer_missing(
        pos, known, edges["pre"].astype(np.int64), edges["post"].astype(np.int64), args.rounds
    )

    pos *= VOXEL_NM
    centre = pos[known].mean(axis=0)
    pos -= centre
    scale = float(np.abs(pos[known]).max())
    pos /= scale

    np.save(args.graph / "positions.npy", pos.astype(np.float32))
    np.save(args.graph / "positions_known.npy", known)
    print(f"[write] {args.graph / 'positions.npy'}  scale {scale / 1000:.1f} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
