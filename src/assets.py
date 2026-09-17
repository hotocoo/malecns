"""Load the published 3D models into the renderer's own frame.

`fetch_assets.py` puts real model files on disk. This turns one into what the
camera draws: triangles, per-vertex colours taken from the model's materials,
and normals for the light.

Two conversions happen here and nowhere else, so the rest of the code can treat
every object alike:

  * axes - glTF is Y-up with -Z forward, this simulator is Z-up with +X
    forward, so each model is rotated once on load;
  * scale - a model is authored at whatever size its artist chose, so it is
    scaled to the real-world length the scene asks for, and stood on z = 0.

Loading and converting a model costs far more than drawing it, so the result is
cached in memory per (file, length) and reused for every instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSET_DIR = Path("data/assets")
RAW_DIR = ASSET_DIR / "raw"
MANIFEST = ASSET_DIR / "manifest.json"

# Where each kind of object's models live, relative to `RAW_DIR`. Every entry
# is a file that `fetch_assets.py` installed; nothing is generated here.
CAR_MODELS = (
    "kenney_car_kit/Models/GLB format/sedan.glb",
    "kenney_car_kit/Models/GLB format/suv.glb",
    "kenney_car_kit/Models/GLB format/hatchback-sports.glb",
    "kenney_car_kit/Models/GLB format/sedan-sports.glb",
    "kenney_car_kit/Models/GLB format/taxi.glb",
    "kenney_car_kit/Models/GLB format/van.glb",
    "kenney_car_kit/Models/GLB format/suv-luxury.glb",
    "kenney_car_kit/Models/GLB format/police.glb",
)
TRUCK_MODELS = (
    "kenney_car_kit/Models/GLB format/truck.glb",
    "kenney_car_kit/Models/GLB format/delivery.glb",
    "kenney_car_kit/Models/GLB format/garbage-truck.glb",
    "kenney_car_kit/Models/GLB format/truck-flat.glb",
)
BUS_MODELS = ("kenney_car_kit/Models/GLB format/delivery-flat.glb",)
MOTORCYCLE_MODELS = ("bike.obj",)
PERSON_MODELS = tuple(f"kenney_characters/Models/GLB format/character-{c}.glb" for c in "abcdefghijklmnopqr")
TRAFFIC_LIGHT_MODELS = ("kenney_roads/Models/GLB format/traffic-light.glb",)
STOP_SIGN_MODELS = ("kenney_roads/Models/GLB format/road-sign-object-stop.glb",)

# glTF: +Y up, -Z forward. Here: +Z up, +X forward.
GLTF_TO_WORLD = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
# The motorcycle OBJ is a Blender export written Y-up with its length along
# +X, so only the two vertical axes need swapping.
OBJ_TO_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)

DEFAULT_COLOUR = np.array([0.62, 0.62, 0.64], dtype=np.float32)


@dataclass(frozen=True)
class Mesh:
    """A loaded model, ready to be placed."""

    name: str
    vertices: np.ndarray  # (v, 3) float32, +X forward, standing on z = 0
    colours: np.ndarray  # (v, 3) float32
    normals: np.ndarray  # (v, 3) float32
    length_m: float
    width_m: float
    height_m: float

    @property
    def triangles(self) -> int:
        return self.vertices.shape[0] // 3


class MissingAsset(FileNotFoundError):
    """A model the scene asked for is not installed."""


def asset_path(relative: str) -> Path:
    path = RAW_DIR / relative
    if not path.exists():
        raise MissingAsset(f"{path} not installed; run `python3 src/fetch_assets.py`")
    return path


def available(relative: str) -> bool:
    return (RAW_DIR / relative).exists()


def credits() -> list[dict]:
    """The licence and credit line of every installed pack."""
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text()).get("packs", [])


def _vertex_colours(mesh, count: int) -> np.ndarray:
    """Per-vertex colour from the model's own visual, or a neutral grey.

    Kenney's kits paint with a small texture atlas; sampling it per vertex
    gives the model's real colours without the renderer needing textures.
    """
    visual = getattr(mesh, "visual", None)
    try:
        if visual is not None and getattr(visual, "kind", None) == "vertex":
            colours = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            return colours.reshape(-1, 3)
        if visual is not None and getattr(visual, "kind", None) == "texture":
            converted = visual.to_color()
            colours = np.asarray(converted.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            return colours.reshape(-1, 3)
        material = getattr(visual, "material", None)
        base = getattr(material, "baseColorFactor", None) if material is not None else None
        if base is not None:
            return np.tile(np.asarray(base, dtype=np.float32)[:3], (count, 1))
    except Exception:
        pass  # an unusual material is not a reason to fail a frame
    return np.tile(DEFAULT_COLOUR, (count, 1))


@lru_cache(maxsize=128)
def load_mesh(
    relative: str,
    length_m: float | None = None,
    upright: bool = True,
    fit: tuple[float, float, float] | None = None,
) -> Mesh:
    """One model, rotated into world axes and sized to the scene's dimensions.

    `fit` gives the real length, width and height the object has in the world
    and scales each axis to it. Published models are stylised - a kit car is
    chunkier than a real one - so scaling by length alone would leave a car
    2.6 m wide. `length_m` keeps the model's own proportions and is used where
    only one dimension matters.
    """
    import trimesh

    path = asset_path(relative)
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):  # a kit file may hold several parts
        loaded = trimesh.util.concatenate([g for g in loaded.geometry.values()])

    faces = np.asarray(loaded.faces)
    points = np.asarray(loaded.vertices, dtype=np.float64)
    rotation = OBJ_TO_WORLD if path.suffix.lower() == ".obj" else GLTF_TO_WORLD
    points = points @ rotation.T

    per_vertex = _vertex_colours(loaded, points.shape[0])
    vertices = points[faces].reshape(-1, 3)
    colours = per_vertex[faces].reshape(-1, 3) if per_vertex.shape[0] == points.shape[0] else np.tile(
        DEFAULT_COLOUR, (vertices.shape[0], 1)
    )

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if fit is not None:
        wanted = np.asarray(fit, dtype=np.float64)
        scale = np.where(extent > 1e-6, wanted / np.maximum(extent, 1e-6), 1.0)
        vertices = vertices * scale
        extent = vertices.max(axis=0) - vertices.min(axis=0)
    elif length_m is not None and extent[0] > 1e-6:
        vertices = vertices * (length_m / extent[0])
        extent = vertices.max(axis=0) - vertices.min(axis=0)
    # Stand the model on the road and centre it over its own footprint, so a
    # placement puts the wheels where the road is.
    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices[:, 0] -= centre[0]
    vertices[:, 1] -= centre[1]
    if upright:
        vertices[:, 2] -= vertices[:, 2].min()

    triangles = vertices.reshape(-1, 3, 3)
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.where(lengths > 1e-9, face_normals / np.maximum(lengths, 1e-9), np.array([0.0, 0.0, 1.0]))
    normals = np.repeat(face_normals, 3, axis=0)

    return Mesh(
        name=Path(relative).stem,
        vertices=vertices.astype(np.float32),
        colours=colours.astype(np.float32),
        normals=normals.astype(np.float32),
        length_m=float(extent[0]),
        width_m=float(extent[1]),
        height_m=float(extent[2]),
    )


def place(mesh: Mesh, pos: np.ndarray, heading: float, tint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mesh standing at `pos`, yawed by `heading`, optionally recoloured.

    `tint` multiplies the model's own colours rather than replacing them, so a
    repainted car keeps its windows, lights and tyres.
    """
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    vertices = mesh.vertices @ rot.T + np.asarray(pos, dtype=np.float32)
    normals = mesh.normals @ rot.T
    colours = mesh.colours if tint is None else np.clip(mesh.colours * np.asarray(tint, dtype=np.float32), 0.0, 1.0)
    return vertices, colours, normals


def pick(models: tuple[str, ...], index: int) -> str:
    """One model from a family, chosen by a stable index rather than at random."""
    installed = [m for m in models if available(m)]
    if not installed:
        raise MissingAsset(f"none of {models} are installed; run `python3 src/fetch_assets.py`")
    return installed[index % len(installed)]
