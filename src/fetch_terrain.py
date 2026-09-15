"""Fetch the real terrain around a circuit as a small elevation grid.

Primary source: Copernicus EU-DEM v1.1 (25 m) through the public
opentopodata.org API. Fallback: Mapzen/Tilezen Terrarium tiles on AWS Open
Data (SRTM 30 m outside Europe). Both are free, keyless, and real surveys;
nothing here is invented.

  python3 src/fetch_terrain.py --track data/tracks/monaco.geojson \
      --out data/tracks/monaco_dem.json

The grid is regular in lon/lat, row-major from the south-west corner, metres
above sea level. Cells the survey leaves empty (open sea, tile edges) are
filled from their nearest surveyed neighbour and flagged in `filled`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

EARTH_M_PER_DEG_LAT = 111_320.0
OPENTOPO = "https://api.opentopodata.org/v1/{dataset}?locations={locations}"
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
USER_AGENT = "malecns-terrain/1.0 (+https://github.com/hotocoo/malecns)"
ATTRIBUTION = {
    "eudem25m": "Copernicus EU-DEM v1.1 (c) European Union, 2016, via opentopodata.org",
    "srtm30m": "NASA SRTM GL1 30 m, via opentopodata.org",
    "terrarium": "Mapzen Terrain Tiles (Terrarium) on AWS Open Data; SRTM, EU-DEM and others; see https://github.com/tilezen/joerd/blob/master/docs/attribution.md",
}


def bbox_of(track: Path, pad_m: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) of the centerline padded by `pad_m` metres."""
    coords = json.loads(track.read_text())["features"][0]["geometry"]["coordinates"]
    lons = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords])
    lat_mid = float(lats.mean())
    dlat = pad_m / EARTH_M_PER_DEG_LAT
    dlon = pad_m / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat_mid)))
    return float(lons.min() - dlon), float(lats.min() - dlat), float(lons.max() + dlon), float(lats.max() + dlat)


def grid_axes(bbox: tuple[float, float, float, float], cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = bbox
    lat_mid = 0.5 * (south + north)
    dlat = cell_m / EARTH_M_PER_DEG_LAT
    dlon = cell_m / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat_mid)))
    lons = np.arange(west, east + dlon * 0.5, dlon)
    lats = np.arange(south, north + dlat * 0.5, dlat)
    return lons, lats


def fetch_opentopo(dataset: str, lons: np.ndarray, lats: np.ndarray, per_request: int = 100, pause_s: float = 1.05) -> np.ndarray:
    """Query every grid node; returns (rows, cols) heights with NaN where the survey has no value."""
    glon, glat = np.meshgrid(lons, lats)
    flat_lon = glon.ravel()
    flat_lat = glat.ravel()
    out = np.full(flat_lon.shape, np.nan)
    total = len(flat_lon)
    for start in range(0, total, per_request):
        locs = "|".join(f"{la:.6f},{lo:.6f}" for lo, la in zip(flat_lon[start : start + per_request], flat_lat[start : start + per_request]))
        url = OPENTOPO.format(dataset=dataset, locations=locs)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.load(resp)
                break
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                if attempt == 3:
                    raise
                print(f"[opentopodata] {exc}; retrying", file=sys.stderr)
                time.sleep(3.0 * (attempt + 1))
        if body.get("status") != "OK":
            raise SystemExit(f"opentopodata {dataset}: {body}")
        vals = [r["elevation"] for r in body["results"]]
        out[start : start + len(vals)] = [np.nan if v is None else float(v) for v in vals]
        print(f"[opentopodata] {min(start + per_request, total)}/{total}", file=sys.stderr)
        time.sleep(pause_s)
    return out.reshape(len(lats), len(lons))


def _tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch_terrarium(lons: np.ndarray, lats: np.ndarray, z: int = 15) -> np.ndarray:
    """Sample Terrarium PNG tiles (h = R*256 + G + B/256 - 32768) bilinearly at the grid nodes."""
    from PIL import Image  # only needed on this path

    x0, y1 = _tile_xy(float(lats.min()), float(lons.min()), z)
    x1, y0 = _tile_xy(float(lats.max()), float(lons.max()), z)
    txs = range(int(math.floor(x0)), int(math.floor(x1)) + 1)
    tys = range(int(math.floor(y0)), int(math.floor(y1)) + 1)
    rows = []
    for ty in tys:
        row = []
        for tx in txs:
            req = urllib.request.Request(TERRARIUM.format(z=z, x=tx, y=ty), headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                rgb = np.asarray(Image.open(io.BytesIO(resp.read())).convert("RGB")).astype(np.float64)
            row.append(rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0)
        rows.append(np.concatenate(row, axis=1))
    mosaic = np.concatenate(rows, axis=0)
    glon, glat = np.meshgrid(lons, lats)
    px = np.empty(glon.shape)
    py = np.empty(glon.shape)
    for i in range(glon.shape[0]):
        for j in range(glon.shape[1]):
            tx, ty = _tile_xy(float(glat[i, j]), float(glon[i, j]), z)
            px[i, j] = (tx - txs.start) * 256.0 - 0.5
            py[i, j] = (ty - tys.start) * 256.0 - 0.5
    px = np.clip(px, 0, mosaic.shape[1] - 1.001)
    py = np.clip(py, 0, mosaic.shape[0] - 1.001)
    j0 = np.floor(px).astype(int)
    i0 = np.floor(py).astype(int)
    fx = px - j0
    fy = py - i0
    return (
        mosaic[i0, j0] * (1 - fx) * (1 - fy)
        + mosaic[i0, j0 + 1] * fx * (1 - fy)
        + mosaic[i0 + 1, j0] * (1 - fx) * fy
        + mosaic[i0 + 1, j0 + 1] * fx * fy
    )


def fill_nodata(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour fill of NaN cells; returns (filled grid, mask of filled cells)."""
    filled = grid.copy()
    missing = np.isnan(filled)
    if not missing.any():
        return filled, missing
    rows, cols = np.indices(grid.shape)
    known = ~missing
    src = np.stack([rows[known], cols[known]], axis=1)
    vals = filled[known]
    for r, c in zip(rows[missing], cols[missing]):
        k = int(np.argmin((src[:, 0] - r) ** 2 + (src[:, 1] - c) ** 2))
        filled[r, c] = vals[k]
    return filled, missing


def build(track: Path, out: Path, pad_m: float, cell_m: float, dataset: str) -> dict:
    bbox = bbox_of(track, pad_m)
    lons, lats = grid_axes(bbox, cell_m)
    print(f"[terrain] bbox {bbox}, grid {len(lats)} x {len(lons)} at {cell_m} m", file=sys.stderr)
    source = dataset
    try:
        if dataset == "terrarium":
            raise ValueError("terrarium requested")
        heights = fetch_opentopo(dataset, lons, lats)
    except Exception as exc:  # noqa: BLE001 - fall back to the tile source
        print(f"[terrain] {dataset} unavailable ({exc}); falling back to Terrarium tiles", file=sys.stderr)
        heights = fetch_terrarium(lons, lats)
        source = "terrarium"
    filled, mask = fill_nodata(heights)
    payload = {
        "source": source,
        "attribution": ATTRIBUTION[source],
        "track": str(track),
        "cell_m": cell_m,
        "lon0": float(lons[0]),
        "lat0": float(lats[0]),
        "dlon": float(lons[1] - lons[0]),
        "dlat": float(lats[1] - lats[0]),
        "cols": int(len(lons)),
        "rows": int(len(lats)),
        "heights": [round(float(v), 2) for v in filled.ravel()],
        "filled": [int(i) for i in np.flatnonzero(mask.ravel())],
        "fetched": time.strftime("%Y-%m-%d"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"[ok] {out}: {source}, {payload['rows']}x{payload['cols']} cells, "
        f"{float(np.nanmin(heights)):.1f}..{float(np.nanmax(heights)):.1f} m, {int(mask.sum())} filled",
        file=sys.stderr,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default=Path("data/tracks/monaco.geojson"), type=Path)
    parser.add_argument("--out", default=None, type=Path, help="default: <track>_dem.json next to the track")
    parser.add_argument("--pad-m", type=float, default=450.0, help="metres of terrain beyond the circuit's bounding box")
    parser.add_argument("--cell-m", type=float, default=25.0, help="grid spacing; EU-DEM is 25 m, finer only interpolates")
    parser.add_argument("--dataset", default="eudem25m", choices=("eudem25m", "srtm30m", "terrarium"))
    args = parser.parse_args(argv)
    out = args.out or args.track.with_name(args.track.stem + "_dem.json")
    build(args.track, out, args.pad_m, args.cell_m, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
