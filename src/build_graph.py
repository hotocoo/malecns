"""Build a signed, indexed connectivity graph from the MaleCNS flat connectome.

Output (data/graph/):
  neurons.parquet  one row per modelled neuron, in matrix-index order
  edges.npz        pre/post indices (int32) and signed weights (float32)
  roles.json       index lists for sensory, motor and other functional groups

Sign convention follows Shiu et al. 2024 (Nature 634:210-219): acetylcholine,
dopamine, serotonin and octopamine are excitatory; GABA and glutamate are
inhibitory. Neurons with no confident prediction default to excitatory
(cholinergic is by far the most common) and are flagged in `nt_known`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

# Histamine is inhibitory in the fly: photoreceptor histamine opens chloride
# channels on lamina monopolar cells (Hardie 1989).
EXCITATORY = {"acetylcholine", "dopamine", "serotonin", "octopamine"}
INHIBITORY = {"gaba", "glutamate", "histamine"}

# Functional groups used to wire the brain to a body. Keys are role names,
# values are `superclass` values in the MaleCNS annotations.
ROLE_SUPERCLASS = {
    "photoreceptor": ["ol_sensory"],
    "visual_projection": ["visual_projection"],
    "descending": ["descending_neuron"],
    "motor": ["vnc_motor", "cb_motor"],
    "ascending": ["ascending_neuron", "sensory_ascending"],
    "mechanosensory": ["vnc_sensory"],
}


def load_neurons(raw: Path) -> pd.DataFrame:
    ann = pd.read_feather(
        raw / "body-annotations.feather",
        columns=[
            "bodyId",
            "superclass",
            "class",
            "type",
            "instance",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    )
    ann = ann[ann["superclass"].notna()].copy()

    nt = pd.read_feather(
        raw / "body-neurotransmitters.feather",
        columns=["body", "consensus_nt", "predicted_nt_confidence"],
    ).rename(columns={"body": "bodyId"})

    neurons = ann.merge(nt, on="bodyId", how="left")
    consensus = neurons["consensus_nt"].fillna("unknown").str.lower()
    sign = np.where(
        consensus.isin(INHIBITORY), -1.0, 1.0
    )  # unknown -> excitatory prior
    neurons["nt"] = consensus
    neurons["sign"] = sign.astype(np.float32)
    neurons["nt_known"] = consensus.isin(EXCITATORY | INHIBITORY)
    return neurons.reset_index(drop=True)


def load_edges(
    raw: Path, body_to_idx: dict[int, int], min_weight: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = ipc.open_file(raw / "connectome-weights.feather")
    keys = np.fromiter(body_to_idx.keys(), dtype=np.int64, count=len(body_to_idx))
    vals = np.fromiter(body_to_idx.values(), dtype=np.int32, count=len(body_to_idx))
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]

    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    w_chunks: list[np.ndarray] = []

    def lookup(bodies: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(keys, bodies)
        pos = np.clip(pos, 0, len(keys) - 1)
        hit = keys[pos] == bodies
        out = np.where(hit, vals[pos], -1)
        return out.astype(np.int32)

    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        weight = batch.column("weight").to_numpy()
        keep = weight >= min_weight
        if not keep.any():
            continue
        pre = lookup(batch.column("body_pre").to_numpy()[keep])
        post = lookup(batch.column("body_post").to_numpy()[keep])
        valid = (pre >= 0) & (post >= 0)
        if not valid.any():
            continue
        pre_chunks.append(pre[valid])
        post_chunks.append(post[valid])
        w_chunks.append(weight[keep][valid].astype(np.float32))

    return (
        np.concatenate(pre_chunks),
        np.concatenate(post_chunks),
        np.concatenate(w_chunks),
    )


def build_roles(neurons: pd.DataFrame) -> dict[str, list[int]]:
    roles: dict[str, list[int]] = {}
    for role, superclasses in ROLE_SUPERCLASS.items():
        idx = np.flatnonzero(neurons["superclass"].isin(superclasses).to_numpy())
        roles[role] = idx.astype(int).tolist()
    return roles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=Path("data/raw"), type=Path)
    parser.add_argument("--out", default=Path("data/graph"), type=Path)
    parser.add_argument(
        "--min-weight",
        type=int,
        default=1,
        help="drop connections with fewer than this many synapses",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    neurons = load_neurons(args.raw)
    body_to_idx = {int(b): i for i, b in enumerate(neurons["bodyId"].to_numpy())}
    print(f"neurons: {len(neurons):,}")

    pre, post, weight = load_edges(args.raw, body_to_idx, args.min_weight)
    signed = weight * neurons["sign"].to_numpy(np.float32)[pre]
    print(f"edges:   {len(pre):,}  synapses: {weight.sum():,.0f}")

    neurons.to_parquet(args.out / "neurons.parquet", index=False)
    np.savez_compressed(
        args.out / "edges.npz", pre=pre, post=post, weight=signed.astype(np.float32)
    )
    roles = build_roles(neurons)
    (args.out / "roles.json").write_text(json.dumps(roles))
    print({k: len(v) for k, v in roles.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
