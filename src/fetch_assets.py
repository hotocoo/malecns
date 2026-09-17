"""Fetch the real 3D models the scene is built from, and record where each came from.

The traffic, the pedestrians, the signals and the signs are published models,
not shapes assembled here. Each pack is downloaded from its publisher, checked
against a pinned SHA-256 and unpacked, and the manifest records the licence and
the credit line every one of them carries.

  python3 src/fetch_assets.py            # download, verify, unpack
  python3 src/fetch_assets.py --list     # what is installed, and under what licence

Licences are honoured, not assumed: a CC0 pack needs no credit, and a CC-BY one
is not used unless its credit is recorded here. Nothing ships without a source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ASSET_DIR = Path("data/assets")
RAW_DIR = ASSET_DIR / "raw"
MANIFEST = ASSET_DIR / "manifest.json"


@dataclass(frozen=True)
class AssetPack:
    """One published model pack, pinned by content hash."""

    name: str
    url: str
    sha256: str
    licence: str
    source_page: str
    credit: str
    archive: str  # the file as downloaded
    unpack_to: str | None  # a directory under data/assets, or None to keep the file as is

    @property
    def download(self) -> Path:
        return RAW_DIR / self.archive

    @property
    def target(self) -> Path | None:
        return None if self.unpack_to is None else RAW_DIR / self.unpack_to


PACKS = (
    AssetPack(
        name="kenney-car-kit",
        url="https://kenney.nl/media/pages/assets/car-kit/1a312ec241-1775131960/kenney_car-kit.zip",
        sha256="fac7dacac5c7874348cf19729af3ef205f3d366493edaf0a827d93f4fdf3d0c4",
        licence="CC0 1.0",
        source_page="https://kenney.nl/assets/car-kit",
        credit="Kenney (kenney.nl)",
        archive="kenney_car-kit.zip",
        unpack_to="kenney_car_kit",
    ),
    AssetPack(
        name="kenney-blocky-characters",
        url="https://kenney.nl/media/pages/assets/blocky-characters/8369c0cf30-1749547469/kenney_blocky-characters_20.zip",
        sha256="5e123859aa0c1598342b600c6db197024a1d63eb9ec531398b310725f589887e",
        licence="CC0 1.0",
        source_page="https://kenney.nl/assets/blocky-characters",
        credit="Kenney (kenney.nl)",
        archive="kenney_blocky-characters.zip",
        unpack_to="kenney_characters",
    ),
    AssetPack(
        name="kenney-city-kit-roads",
        url="https://kenney.nl/media/pages/assets/city-kit-roads/74288c9459-1787042796/kenney_city-kit-roads.zip",
        sha256="22058af3d68173a7cf9bda9f0e243a8cef6bd68168c302ebc76327063849674e",
        licence="CC0 1.0",
        source_page="https://kenney.nl/assets/city-kit-roads",
        credit="Kenney (kenney.nl)",
        archive="kenney_city-kit-roads.zip",
        unpack_to="kenney_roads",
    ),
    AssetPack(
        name="kenney-city-kit-commercial",
        url="https://kenney.nl/media/pages/assets/city-kit-commercial/a742d900eb-1753115042/kenney_city-kit-commercial_2.1.zip",
        sha256="f8b09b081c2bb88bcc126e2dec1cb40fd0dad7e7e591b6c26aaefe96fb35276b",
        licence="CC0 1.0",
        source_page="https://kenney.nl/assets/city-kit-commercial",
        credit="Kenney (kenney.nl)",
        archive="kenney_city-kit-commercial.zip",
        unpack_to="kenney_city",
    ),
    AssetPack(
        name="oga-fancy-motorcycle",
        url="https://opengameart.org/sites/default/files/bike.obj",
        sha256="2b17d916f146c581faa421b76d2b79471ed09972171d1c394ac877552f2891dc",
        licence="CC0 1.0",
        source_page="https://opengameart.org/content/fancy-motorcycle",
        credit="OpenGameArt, 'Fancy Motorcycle'",
        archive="bike.obj",
        unpack_to=None,
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(pack: AssetPack, force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if pack.download.exists() and not force and sha256_of(pack.download) == pack.sha256:
        print(f"[skip] {pack.name}", file=sys.stderr)
        return pack.download
    print(f"[get ] {pack.name}", file=sys.stderr)
    result = subprocess.run(
        ["curl", "-sSL", "-m", "300", "-A", "malecns-assets/1.0", "-o", str(pack.download), pack.url],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"download failed for {pack.name} (curl {result.returncode})")
    got = sha256_of(pack.download)
    if got != pack.sha256:
        raise SystemExit(
            f"{pack.name}: sha256 {got} does not match the pinned {pack.sha256}. "
            "The publisher reissued the pack; look at it before pinning the new hash."
        )
    return pack.download


def unpack(pack: AssetPack, force: bool = False) -> None:
    if pack.target is None:
        return
    if pack.target.exists() and not force and any(pack.target.rglob("*.glb")):
        return
    with zipfile.ZipFile(pack.download) as archive:
        archive.extractall(pack.target)


def installed(pack: AssetPack) -> int:
    """How many models this pack contributes."""
    if pack.target is None:
        return 1 if pack.download.exists() else 0
    if not pack.target.exists():
        return 0
    return sum(1 for _ in pack.target.rglob("*.glb"))


def write_manifest() -> dict:
    payload = {
        "note": (
            "Models are published work, downloaded from their publisher and pinned by hash. "
            "Credit lines are kept whether or not the licence demands one."
        ),
        "packs": [
            {
                "name": p.name,
                "licence": p.licence,
                "credit": p.credit,
                "source_page": p.source_page,
                "url": p.url,
                "sha256": p.sha256,
                "models": installed(p),
                "path": str(p.target or p.download),
            }
            for p in PACKS
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show what is installed")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for pack in PACKS:
            print(f"{pack.name:30s} {pack.licence:10s} {installed(pack):4d} models  {pack.credit}")
        return 0

    for pack in PACKS:
        download(pack, args.force)
        unpack(pack, args.force)
    payload = write_manifest()
    total = sum(entry["models"] for entry in payload["packs"])
    print(f"[ok] {MANIFEST} {len(PACKS)} packs, {total} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
