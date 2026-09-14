"""Download the MaleCNS v1.0 flat-connectome files (HHMI Janelia FlyEM, CC-BY 4.0).

Only the three files needed to build a signed connectivity graph are fetched.
The synapse-point files (12.7 GB / 6.8 GB) are not needed for a LIF model and
are skipped on purpose.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0"
    "/connectome-data/flat-connectome"
)

FILES = {
    "body-annotations.feather": (
        f"{BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ),
    "body-neurotransmitters.feather": (
        f"{BASE}/body-neurotransmitters-male-cns-v1.0.feather"
    ),
    "connectome-weights.feather": (
        f"{BASE}/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    ),
}

MIN_BYTES = {
    "body-annotations.feather": 10_000_000,
    "body-neurotransmitters.feather": 30_000_000,
    "connectome-weights.feather": 900_000_000,
}


def fetch(name: str, url: str, out_dir: Path, force: bool) -> Path:
    dest = out_dir / name
    if dest.exists() and not force and dest.stat().st_size >= MIN_BYTES[name]:
        print(f"[skip] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[get ] {name}")
    result = subprocess.run(
        ["curl", "-fL", "--retry", "3", "-o", str(dest), url], check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"download failed for {name} (curl {result.returncode})")
    size = dest.stat().st_size
    if size < MIN_BYTES[name]:
        raise RuntimeError(f"{name} truncated: {size} bytes")
    print(f"[ok  ] {name} ({size / 1e6:.1f} MB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        fetch(name, url, args.out, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
