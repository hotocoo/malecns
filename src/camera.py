"""The driver's camera: render what the car can see, offscreen, as real pixels.

This is the front of the perception chain. `scene.py` places the road, the
signals, the signs and the traffic; this module puts a camera at the car's eye
point and renders the frame the detector will actually look at. No shortcut
runs around it: nothing downstream is told where an object is, only what came
out of the frame buffer.

The renderer is deliberately plain - flat-shaded triangles, one draw per pass,
no post-processing - because every millisecond here is paid once per body per
control step. Frames come back as `uint8` arrays in the layout the detector
wants, so the only copy is the read from the frame buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np

from scene import (
    CLASS_BUS,
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_PERSON,
    CLASS_STOP_SIGN,
    CLASS_TRAFFIC_LIGHT,
    CLASS_TRUCK,
    Actor,
    RoadMesh,
    SceneConfig,
)
from assets import (
    BUS_MODELS,
    CAR_MODELS,
    MOTORCYCLE_MODELS,
    PERSON_MODELS,
    STOP_SIGN_MODELS,
    TRAFFIC_LIGHT_MODELS,
    TRUCK_MODELS,
    MissingAsset,
    available,
    load_mesh,
    pick,
    place,
)

# Which published models stand in for each class, and how long each is in the
# world. The lengths are real vehicle dimensions; the models are scaled to them
# so a car in frame is the size a car is.
MODELS_FOR = {
    CLASS_CAR: CAR_MODELS,
    CLASS_TRUCK: TRUCK_MODELS,
    CLASS_BUS: BUS_MODELS,
    CLASS_MOTORCYCLE: MOTORCYCLE_MODELS,
    CLASS_PERSON: PERSON_MODELS,
    CLASS_TRAFFIC_LIGHT: TRAFFIC_LIGHT_MODELS,
    CLASS_STOP_SIGN: STOP_SIGN_MODELS,
}


def assets_available() -> bool:
    """True when the published models are installed and can be drawn."""
    return any(available(m) for family in MODELS_FOR.values() for m in family)
from lawreward import AMBER, GREEN, RED

# Surface texture, by material, in world metres. One GLSL function shared by
# this renderer and the browser's 3D page (live.py splices it into its
# shader), so both draw the same asphalt grain, paving slabs and kerbstone.
# `streets.MAT_*` numbers the materials.
SURFACE_GLSL = """
float surf_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float surf_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = surf_hash(i);
    float b = surf_hash(i + vec2(1.0, 0.0));
    float c = surf_hash(i + vec2(0.0, 1.0));
    float d = surf_hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
vec3 surface_colour(vec3 base, float material, vec3 w) {
    if (material < 0.5) return base;
    if (material < 1.5) {
        // Asphalt: fine aggregate grain, patches of older and newer surface,
        // and two slightly polished wheel tracks per lane are not modelled;
        // the tone stays inside the band the road profile reads as asphalt.
        float fine = surf_noise(w.xy * 18.0);
        float mid = surf_noise(w.xy * 2.5);
        float coarse = surf_noise(w.xy * 0.35);
        float tone = 0.84 + 0.22 * fine + 0.10 * (coarse - 0.5) + 0.06 * (mid - 0.5);
        return base * tone;
    }
    if (material < 2.5) {
        // Pavement: 0.6 m slabs with dark joints, weathered unevenly.
        vec2 g = fract(w.xy / 0.6);
        float joint = (g.x < 0.05 || g.y < 0.05) ? 0.74 : 1.0;
        float n = surf_noise(w.xy * 6.0);
        return base * joint * (0.90 + 0.18 * n);
    }
    if (material < 3.5) {
        // Kerbstone: concrete, a joint every metre along either axis.
        vec2 g = fract(w.xy / 1.0);
        float joint = (g.x < 0.03 || g.y < 0.03) ? 0.82 : 1.0;
        float n = surf_noise(w.xy * 9.0);
        return base * joint * (0.92 + 0.14 * n);
    }
    if (material < 4.5) {
        // Ground: grass and bare earth in patches.
        float n1 = surf_noise(w.xy * 0.9);
        float n2 = surf_noise(w.xy * 7.0);
        vec3 earth = vec3(0.42, 0.36, 0.28);
        vec3 grass = base * (0.82 + 0.30 * n2);
        return mix(grass, earth, smoothstep(0.66, 0.92, n1));
    }
    // Facade bays run along whichever wall axis this face lies on. One
    // number per building block keeps two neighbours from sharing a grid.
    float block = surf_hash(floor(w.xy / 32.0));
    float bay_m = 1.8 + 0.9 * block;
    float bay = fract(max(abs(fract(w.x / bay_m) - 0.5), abs(fract(w.y / bay_m) - 0.5)) + 0.5);
    float storey = fract(w.z / 3.2);
    float floor_index = floor(w.z / 3.2);

    if (material < 5.5) {
        // Painted render: punched windows, a recessed frame around each, a
        // slab line between storeys and a blind panel here and there.
        bool pane = storey > 0.28 && storey < 0.74 && bay > 0.60 && bay < 0.95;
        bool frame = storey > 0.22 && storey < 0.80 && bay > 0.54 && bay < 0.99;
        float lit = surf_hash(vec2(floor(w.x / bay_m) + floor(w.y / bay_m), floor_index));
        vec3 glass = mix(vec3(0.20, 0.26, 0.34), vec3(0.46, 0.56, 0.40), step(0.86, lit));
        float slab = (storey < 0.07) ? 0.88 : 1.0;
        float weather = 0.95 + 0.07 * surf_noise(w.xy * 2.2 + w.z * 0.4);
        if (pane) return glass * (0.85 + 0.35 * surf_noise(w.xy * 0.6 + w.z));
        if (frame) return base * 0.82 * weather;
        return base * slab * weather;
    }
    if (material < 6.5) {
        // Curtain wall: a glazed band per storey between spandrel panels,
        // mullions every bay, and a sky gradient that brightens with height.
        float mullion = smoothstep(0.02, 0.06, min(bay, 1.0 - bay));
        float spandrel = step(0.74, storey);
        // Glass takes most of its colour from the sky it faces, so a tower
        // reads as glazing rather than as a dark box: brighter with height,
        // with one pane in a while catching the sun.
        float sky = clamp(0.95 + w.z * 0.006, 0.9, 1.35);
        float flash = surf_hash(vec2(floor(w.x / bay_m) + floor(w.y / bay_m), floor_index));
        vec3 pane = mix(base, vec3(0.62, 0.70, 0.80), 0.45) * sky * (0.92 + 0.45 * flash);
        vec3 panel = mix(base, vec3(0.30, 0.33, 0.36), 0.5);
        vec3 face = mix(pane, panel, spandrel);
        return face * mix(0.80, 1.0, mullion);
    }
    if (material < 7.5) {
        // Shop glazing: panes between dark mullions, dim but never black. It
        // reflects the street and shows the lit interior behind, which is what
        // a camera reads at this height. The tint stays off neutral grey on
        // purpose: `perceive.road_profile` measures road as grey pixels, and a
        // grey shopfront at eye height would read as more road.
        float panel = fract(max(abs(fract(w.x / 1.4) - 0.5), abs(fract(w.y / 1.4) - 0.5)) + 0.5);
        if (panel < 0.62) return base * 0.55;
        float lit = surf_hash(floor(w.xy / 4.2));
        vec3 inside = mix(vec3(0.22, 0.27, 0.35), vec3(0.42, 0.39, 0.31), step(0.72, lit));
        float sheen = 0.86 + 0.26 * fract(w.z * 0.6);
        return inside * sheen * (0.94 + 0.14 * surf_noise(w.xy * 1.6 + w.z));
    }
    if (material < 8.5) {
        // Roof: gravel and screed, with wet patches where it ponds.
        float grit = surf_noise(w.xy * 9.0);
        float pond = surf_noise(w.xy * 0.7);
        return base * (0.88 + 0.22 * grit) * mix(1.0, 0.82, smoothstep(0.6, 0.85, pond));
    }
    if (material < 9.5) {
        // Signboard over the shopfront: painted board, one colour per unit,
        // with the lettering left to the eye at this resolution.
        float unit = surf_hash(floor(w.xy / 6.0));
        vec3 board = mix(vec3(0.48, 0.22, 0.19), vec3(0.20, 0.30, 0.42), step(0.5, unit));
        board = mix(board, vec3(0.56, 0.47, 0.20), step(0.82, unit));
        return mix(base * 0.8, board, 0.55) * (0.94 + 0.10 * surf_noise(w.xy * 3.0));
    }
    // Plinth under the glazing: glazed tiles, a joint every 30 cm.
    vec2 tile = fract(w.xy / 0.3);
    float joint = (tile.x < 0.08 || tile.y < 0.08) ? 0.80 : 1.0;
    return base * joint * (0.94 + 0.10 * surf_noise(w.xy * 8.0));
}
"""

VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform vec3 sun;
uniform float ambient;
in vec3 in_position;
in vec3 in_colour;
in vec3 in_normal;
in float in_material;
out vec3 v_colour;
out vec3 v_world;
out float v_material;
out float v_depth;
void main() {
    vec4 clip = mvp * vec4(in_position, 1.0);
    gl_Position = clip;
    // Flat sun plus ambient. Without it every face of a car is one colour and
    // the shape disappears; with it the roof, bonnet and flanks separate.
    float lit = ambient + (1.0 - ambient) * max(dot(normalize(in_normal), sun), 0.0);
    v_colour = in_colour * lit;
    v_world = in_position;
    v_material = in_material;
    v_depth = clip.w;
}
"""

FRAGMENT_SHADER = (
    """
#version 330
uniform vec3 fog_colour;
uniform float fog_start;
uniform float fog_end;
in vec3 v_colour;
in vec3 v_world;
in float v_material;
in float v_depth;
out vec4 f_colour;
"""
    + SURFACE_GLSL
    + """
void main() {
    vec3 shaded = surface_colour(v_colour, v_material, v_world);
    float t = clamp((v_depth - fog_start) / max(fog_end - fog_start, 1.0), 0.0, 1.0);
    f_colour = vec4(mix(shaded, fog_colour, t), 1.0);
}
"""
)


def face_normals(vertices: np.ndarray) -> np.ndarray:
    """Unit normal of every triangle, repeated for its three vertices."""
    tris = np.asarray(vertices, dtype=np.float32).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(min=1e-9)
    return np.repeat(n, 3, axis=0).astype(np.float32)


def assemble_static(
    road: RoadMesh,
    network: RoadMesh | None,
    buildings: RoadMesh | None,
    ground_colour: tuple[float, float, float],
    ground_height_m: float = -0.05,
    extent_m: float = 12_000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Everything that never moves, as one set of (vertices, colours, normals, materials).

    Used by the driver's camera and served to the browser's 3D page, so the two
    draw the same city from the same triangles. Materials follow `streets.MAT_*`:
    the ground is grass, a surface without its own material is asphalt, paint
    is flat.
    """
    ground = ground_plane(extent_m, ground_height_m)
    parts: list[np.ndarray] = [ground]
    tints: list[np.ndarray] = [np.tile(np.array(ground_colour, dtype=np.float32), (ground.shape[0], 1))]
    materials: list[np.ndarray] = [np.full(ground.shape[0], 4.0, dtype=np.float32)]

    def add(mesh: RoadMesh | None, default_material: float) -> None:
        if mesh is None or not mesh.surface.shape[0]:
            return
        parts.append(mesh.surface)
        tints.append(mesh.surface_colour)
        if mesh.surface_material is not None and mesh.surface_material.shape[0] == mesh.surface.shape[0]:
            materials.append(np.asarray(mesh.surface_material, dtype=np.float32))
        else:
            materials.append(np.full(mesh.surface.shape[0], default_material, dtype=np.float32))
        if mesh.markings.shape[0]:
            parts.append(mesh.markings)
            tints.append(mesh.markings_colour)
            materials.append(np.zeros(mesh.markings.shape[0], dtype=np.float32))

    add(network, 1.0)
    add(buildings, 5.0)
    add(road, 1.0)
    vertices = np.concatenate(parts).astype(np.float32)
    colours = np.concatenate(tints).astype(np.float32)
    return vertices, colours, face_normals(vertices), np.concatenate(materials)


# The lens on the car. A road camera is wider than a photographer's normal
# lens and narrower than a fisheye; these are the run's optics, not the law's.
@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 384
    fov_deg: float = 72.0
    eye_height_m: float = 1.25
    eye_forward_m: float = 1.2  # ahead of the car's centre, at the windscreen
    near_m: float = 0.5
    far_m: float = 400.0
    sky_colour: tuple[float, float, float] = (0.62, 0.71, 0.82)
    ground_colour: tuple[float, float, float] = (0.27, 0.37, 0.23)
    fog_start_m: float = 120.0
    # A high sun a little behind the driver's left shoulder: near noon, which
    # is what a tropical street looks like for most of the day.
    sun_direction: tuple[float, float, float] = (-0.35, 0.25, 0.90)
    ambient: float = 0.45


SIGNAL_LENS = {
    GREEN: (0.10, 0.85, 0.25),
    AMBER: (0.98, 0.72, 0.05),
    RED: (0.92, 0.12, 0.12),
}


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward).clip(min=1e-6)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right).clip(min=1e-6)
    true_up = np.cross(right, forward)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = right, true_up, -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def box_triangles(pos: np.ndarray, heading: float, size: np.ndarray) -> np.ndarray:
    """A box standing on `pos`, `size` = (length, width, height), yawed by `heading`."""
    length, width, height = size
    hx, hy = length / 2.0, width / 2.0
    corners = np.array(
        [
            [-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0],
            [-hx, -hy, height], [hx, -hy, height], [hx, hy, height], [-hx, hy, height],
        ],
        dtype=np.float64,
    )
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = corners @ rot.T + pos
    faces = (
        (0, 1, 2), (0, 2, 3),  # base
        (4, 6, 5), (4, 7, 6),  # top
        (0, 5, 1), (0, 4, 5),  # sides
        (1, 6, 2), (1, 5, 6),
        (2, 7, 3), (2, 6, 7),
        (3, 4, 0), (3, 7, 4),
    )
    return np.concatenate([world[list(face)] for face in faces]).astype(np.float32)


def ground_plane(extent_m: float, height_m: float) -> np.ndarray:
    """A large quad under everything, so the horizon is ground and not sky."""
    e = extent_m
    quad = np.array(
        [[-e, -e, height_m], [e, -e, height_m], [e, e, height_m], [-e, -e, height_m], [e, e, height_m], [-e, e, height_m]],
        dtype=np.float32,
    )
    return quad


class DriverCamera:
    """An offscreen camera that renders the scene from a car's pose."""

    def __init__(self, config: CameraConfig | None = None, scene_config: SceneConfig | None = None) -> None:
        self.cfg = config or CameraConfig()
        self.scene_cfg = scene_config or SceneConfig()
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.fbo = self.ctx.simple_framebuffer((self.cfg.width, self.cfg.height), components=3)
        self._static: moderngl.VertexArray | None = None
        self._static_buffer: moderngl.Buffer | None = None
        self._static_count = 0
        self._dynamic: moderngl.VertexArray | None = None
        self._dynamic_buffer: moderngl.Buffer | None = None
        self.program["fog_colour"].value = self.cfg.sky_colour
        self.program["fog_start"].value = self.cfg.fog_start_m
        self.program["fog_end"].value = self.cfg.far_m
        sun = np.array(self.cfg.sun_direction, dtype=np.float32)
        self.program["sun"].value = tuple(sun / np.linalg.norm(sun))
        self.program["ambient"].value = self.cfg.ambient
        self._meshes_ready = assets_available()

    @property
    def renderer_name(self) -> str:
        return str(self.ctx.info["GL_RENDERER"])

    def _vertex_array(
        self,
        vertices: np.ndarray,
        colours: np.ndarray,
        normals: np.ndarray | None = None,
        materials: np.ndarray | None = None,
    ) -> tuple[moderngl.VertexArray, moderngl.Buffer]:
        """A vertex array and the buffer behind it.

        The buffer is handed back because releasing a `VertexArray` does not
        free it: the dynamic geometry is rebuilt every frame, so a buffer left
        behind each time is a leak measured in hundreds of megabytes a minute.
        """
        if normals is None or normals.shape != vertices.shape:
            normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (vertices.shape[0], 1))
        if materials is None or materials.shape[0] != vertices.shape[0]:
            materials = np.zeros(vertices.shape[0], dtype=np.float32)
        data = np.concatenate(
            [vertices, colours, normals, materials.reshape(-1, 1)], axis=1
        ).astype("f4").tobytes()
        buffer = self.ctx.buffer(data)
        array = self.ctx.vertex_array(
            self.program, [(buffer, "3f 3f 3f 1f", "in_position", "in_colour", "in_normal", "in_material")]
        )
        return array, buffer

    def _set_dynamic(self, vertices: np.ndarray, colours: np.ndarray, normals: np.ndarray) -> None:
        """Replace the per-frame geometry, freeing what the last frame used."""
        if self._dynamic is not None:
            self._dynamic.release()
            self._dynamic = None
        if self._dynamic_buffer is not None:
            self._dynamic_buffer.release()
            self._dynamic_buffer = None
        self._dynamic, self._dynamic_buffer = self._vertex_array(vertices, colours, normals)

    def set_static(
        self,
        road: RoadMesh,
        ground_height_m: float = -0.05,
        extent_m: float = 12_000.0,
        network: RoadMesh | None = None,
        buildings: RoadMesh | None = None,
    ) -> None:
        """Upload the geometry that does not change: ground, roads, markings.

        `network` is every other surveyed road in the region. It is drawn under
        the lap's own surface, so a junction the driver reaches has the streets
        that really lead off it.
        """
        vertices, colours, normals, materials = assemble_static(
            road, network, buildings, self.cfg.ground_colour, ground_height_m, extent_m
        )
        self._static, self._static_buffer = self._vertex_array(vertices, colours, normals, materials)
        self._static_count = vertices.shape[0]

    def visible(self, pos_xy: np.ndarray, heading: float, actors: list[Actor]) -> list[Actor]:
        """The actors that could land in this frame.

        Building triangles for a whole city's traffic costs more than drawing
        them, so anything behind the camera or beyond the far plane is dropped
        before its geometry is ever built. The margin is generous: an object
        whose centre sits just outside the cone can still have a visible edge.
        """
        if not actors:
            return []
        cfg = self.cfg
        centres = np.array([[a.pos[0], a.pos[1]] for a in actors])
        delta = centres - np.asarray(pos_xy, dtype=np.float64)[None, :]
        distance = np.linalg.norm(delta, axis=1)
        forward = np.array([np.cos(heading), np.sin(heading)])
        ahead = delta @ forward
        # Half the horizontal field, widened by the largest object radius so a
        # long vehicle crossing the edge is not clipped away whole.
        half_fov = np.radians(cfg.fov_deg) / 2.0 * (cfg.width / cfg.height) ** 0.0
        margin = np.tan(half_fov) * np.maximum(ahead, 1.0) + 6.0
        lateral = np.abs(delta @ np.array([-forward[1], forward[0]]))
        keep = (distance < cfg.far_m) & (ahead > -8.0) & (lateral < margin + 8.0)
        return [a for a, k in zip(actors, keep) if k]

    def _actor_geometry(
        self, actors: list[Actor], with_spans: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Triangles, colours and normals for `actors`, optionally with per-actor spans.

        Each actor is drawn as its published model, scaled to the real size the
        scene gave it and tinted by its own colour so a street is not eight
        identical cars. Where a model is not installed the actor falls back to a
        plain box, which keeps the simulator runnable before `fetch_assets.py`
        has been run.

        The label pass needs to know which vertices belong to which actor, and
        rebuilding the geometry per actor to find out costs more than the draw
        itself; `with_spans` hands that back from the one pass that built it.
        """
        chunks: list[np.ndarray] = []
        colours: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        spans: list[int] = []

        def emit(vertices: np.ndarray, colour: np.ndarray, normal: np.ndarray | None = None) -> None:
            chunks.append(vertices)
            colours.append(colour if colour.ndim == 2 else np.tile(colour, (vertices.shape[0], 1)))
            normals.append(
                normal
                if normal is not None
                else np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (vertices.shape[0], 1))
            )

        for actor in actors:
            before = sum(c.shape[0] for c in chunks)
            drawn = False
            if self._meshes_ready:
                drawn = self._emit_model(actor, emit)
            if not drawn:
                self._emit_box(actor, emit)
            spans.append(sum(c.shape[0] for c in chunks) - before)

        if not chunks:
            empty = np.zeros((0, 3), dtype=np.float32)
            return (empty, empty, empty, spans) if with_spans else (empty, empty, empty)
        vertices = np.concatenate(chunks)
        colour_out = np.concatenate(colours)
        normal_out = np.concatenate(normals)
        return (vertices, colour_out, normal_out, spans) if with_spans else (vertices, colour_out, normal_out)

    def _emit_model(self, actor: Actor, emit) -> bool:
        """Draw `actor` as its published model. False when none is installed."""
        family = MODELS_FOR.get(actor.kind)
        if not family:
            return False
        try:
            relative = pick(family, max(actor.node_id, 0) if actor.node_id >= 0 else self._variant(actor))
            # Rounded to the centimetre so two actors of the same nominal size
            # share one cached mesh instead of each filling a cache slot.
            mesh = load_mesh(relative, fit=tuple(round(float(v), 2) for v in actor.size))
        except (MissingAsset, Exception):
            return False

        if actor.kind == CLASS_TRAFFIC_LIGHT:
            # The model is a light head; it stands at the height the scene set,
            # on a pole reaching the road.
            base = np.array([actor.pos[0], actor.pos[1], 0.0])
            pole = box_triangles(base, actor.heading, np.array([0.12, 0.12, max(actor.pos[2], 0.1)]))
            emit(pole, np.array([0.33, 0.33, 0.35], dtype=np.float32))
            head_pos = np.array([actor.pos[0], actor.pos[1], actor.pos[2] - mesh.height_m])
            vertices, tint, normals = place(mesh, head_pos, actor.heading)
            emit(vertices, tint, normals)
            emit(*self._signal_lens(actor, mesh))
            return True

        if actor.kind == CLASS_STOP_SIGN:
            base = np.array([actor.pos[0], actor.pos[1], 0.0])
            pole = box_triangles(base, actor.heading, np.array([0.09, 0.09, max(actor.pos[2], 0.1)]))
            emit(pole, np.array([0.45, 0.45, 0.47], dtype=np.float32))
            plate = np.array([actor.pos[0], actor.pos[1], actor.pos[2] - mesh.height_m])
            vertices, tint, normals = place(mesh, plate, actor.heading)
            emit(vertices, tint, normals)
            return True

        tint = None if actor.kind == CLASS_PERSON else actor.colour * 1.3
        vertices, colour, normals = place(mesh, actor.pos, actor.heading, tint)
        emit(vertices, colour, normals)
        return True

    def _signal_lens(self, actor: Actor, mesh) -> tuple[np.ndarray, np.ndarray]:
        """The lit lens, drawn proud of the head so the phase reads from the road."""
        forward = np.array([np.cos(actor.heading), np.sin(actor.heading), 0.0])
        centre = np.array([actor.pos[0], actor.pos[1], actor.pos[2] - mesh.height_m * 0.5]) + forward * (
            mesh.length_m * 0.55
        )
        size = np.array([0.05, max(mesh.width_m * 0.55, 0.12), max(mesh.height_m * 0.22, 0.12)])
        lens = box_triangles(centre - np.array([0.0, 0.0, size[2] / 2.0]), actor.heading, size)
        colour = np.array(SIGNAL_LENS.get(actor.state, SIGNAL_LENS[RED]), dtype=np.float32)
        return lens, colour

    def _variant(self, actor: Actor) -> int:
        """A stable model choice for an actor with no surveyed identity."""
        return int(abs(actor.pos[0]) * 7.0 + abs(actor.pos[1]) * 13.0) % 97

    def _emit_box(self, actor: Actor, emit) -> None:
        """The fallback shape, used only where a published model is missing."""
        if actor.kind == CLASS_TRAFFIC_LIGHT:
            pole = box_triangles(
                np.array([actor.pos[0], actor.pos[1], 0.0]), actor.heading, np.array([0.12, 0.12, actor.pos[2]])
            )
            emit(pole, np.array([0.35, 0.35, 0.35], dtype=np.float32))
            housing = box_triangles(actor.pos - np.array([0.0, 0.0, actor.size[2]]), actor.heading, actor.size)
            emit(housing, actor.colour)
            lens_size = np.array([0.06, actor.size[1] * 0.7, actor.size[2] * 0.28])
            offset = np.array([np.cos(actor.heading), np.sin(actor.heading), 0.0]) * (actor.size[0] / 2.0)
            lens = box_triangles(
                actor.pos - np.array([0.0, 0.0, actor.size[2] * 0.45]) + offset, actor.heading, lens_size
            )
            emit(lens, np.array(SIGNAL_LENS.get(actor.state, SIGNAL_LENS[RED]), dtype=np.float32))
            return
        if actor.kind == CLASS_STOP_SIGN:
            pole = box_triangles(
                np.array([actor.pos[0], actor.pos[1], 0.0]), actor.heading, np.array([0.09, 0.09, actor.pos[2]])
            )
            emit(pole, np.array([0.45, 0.45, 0.45], dtype=np.float32))
            plate = box_triangles(actor.pos - np.array([0.0, 0.0, actor.size[2] / 2.0]), actor.heading, actor.size)
            emit(plate, actor.colour)
            return

        emit(box_triangles(actor.pos, actor.heading, actor.size), actor.colour)
        if actor.kind == CLASS_PERSON:
            head = box_triangles(
                actor.pos + np.array([0.0, 0.0, actor.size[2]]), actor.heading, np.array([0.22, 0.22, 0.24])
            )
            emit(head, np.array([0.72, 0.58, 0.48], dtype=np.float32))
        elif actor.kind == CLASS_MOTORCYCLE:
            rider = box_triangles(
                actor.pos + np.array([0.0, 0.0, actor.size[2]]), actor.heading, np.array([0.4, 0.45, 0.85])
            )
            emit(rider, np.array([0.18, 0.18, 0.22], dtype=np.float32))

    def render(self, pos_xy: np.ndarray, heading: float, road_z: float, actors: list[Actor]) -> np.ndarray:
        """One frame from the car's eye point, as (height, width, 3) uint8 RGB."""
        cfg = self.cfg
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        eye = np.array([pos_xy[0], pos_xy[1], road_z + cfg.eye_height_m]) + forward * cfg.eye_forward_m
        view = look_at(eye, eye + forward, np.array([0.0, 0.0, 1.0]))
        proj = perspective(cfg.fov_deg, cfg.width / cfg.height, cfg.near_m, cfg.far_m)
        self.program["mvp"].write((proj @ view).T.astype("f4").tobytes())

        self.fbo.use()
        self.ctx.clear(*cfg.sky_colour, depth=1.0)
        if self._static is not None:
            self._static.render(moderngl.TRIANGLES, vertices=self._static_count)

        vertices, colours, normals = self._actor_geometry(self.visible(pos_xy, heading, actors))
        if vertices.shape[0]:
            self._set_dynamic(vertices, colours, normals)
            self._dynamic.render(moderngl.TRIANGLES, vertices=vertices.shape[0])

        raw = self.fbo.read(components=3, dtype="f1")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(cfg.height, cfg.width, 3)
        return np.flipud(frame).copy()  # GL origin is bottom-left; images are top-left

    def render_with_labels(
        self,
        pos_xy: np.ndarray,
        heading: float,
        road_z: float,
        actors: list[Actor],
        min_pixels: int = 12,
    ) -> tuple[np.ndarray, list[tuple[str, int, int, int, int]]]:
        """The frame, plus a tight box around every actor actually visible in it.

        The boxes come from a second pass that paints each actor in a colour
        encoding its index and reads the result back, so a box is exact and an
        actor hidden behind another contributes nothing. These are labels for
        training a detector, never an input to the driver: the connectome sees
        the frame, and only ever what a detector made of it.
        """
        frame = self.render(pos_xy, heading, road_z, actors)
        actors = self.visible(pos_xy, heading, actors)
        if not actors:
            return frame, []

        cfg = self.cfg
        forward = np.array([np.cos(heading), np.sin(heading), 0.0])
        eye = np.array([pos_xy[0], pos_xy[1], road_z + cfg.eye_height_m]) + forward * cfg.eye_forward_m
        view = look_at(eye, eye + forward, np.array([0.0, 0.0, 1.0]))
        proj = perspective(cfg.fov_deg, cfg.width / cfg.height, cfg.near_m, cfg.far_m)
        self.program["mvp"].write((proj @ view).T.astype("f4").tobytes())
        self.program["fog_start"].value = cfg.far_m * 4.0
        self.program["fog_end"].value = cfg.far_m * 8.0
        # The index pass must write the identity colour untouched. Lighting
        # multiplies it, and an index of 1 is 1/255 in the blue channel, so a
        # lit index pass rounds two different actors to the same byte and the
        # labels silently collapse onto whichever one drew last.
        self.program["ambient"].value = 1.0

        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, depth=1.0)
        vertices, _, _, spans = self._actor_geometry(actors, with_spans=True)
        colours = np.zeros_like(vertices)
        offset = 0
        for index, span in enumerate(spans, start=1):
            tag = np.array(
                [(index >> 16 & 255) / 255.0, (index >> 8 & 255) / 255.0, (index & 255) / 255.0],
                dtype=np.float32,
            )
            colours[offset : offset + span] = tag
            offset += span
        boxes: list[tuple[str, int, int, int, int]] = []
        if vertices.shape[0]:
            self._set_dynamic(vertices, colours, np.zeros_like(vertices))
            self._dynamic.render(moderngl.TRIANGLES, vertices=vertices.shape[0])
            raw = self.fbo.read(components=3, dtype="f1")
            ids = np.frombuffer(raw, dtype=np.uint8).reshape(cfg.height, cfg.width, 3)
            ids = np.flipud(ids).astype(np.int32)
            index_map = (ids[:, :, 0] << 16) | (ids[:, :, 1] << 8) | ids[:, :, 2]
            # One sweep over the painted pixels rather than one per actor: a
            # frame holds a quarter of a million pixels and a busy street holds
            # dozens of actors, so the per-actor scan dominated the capture.
            flat = index_map.reshape(-1)
            painted = np.nonzero(flat)[0]
            if painted.size:
                ids = flat[painted]
                ys, xs = np.divmod(painted, cfg.width)
                size = len(actors) + 1
                counts = np.bincount(ids, minlength=size)
                x_min = np.full(size, cfg.width, dtype=np.int64)
                y_min = np.full(size, cfg.height, dtype=np.int64)
                x_max = np.full(size, -1, dtype=np.int64)
                y_max = np.full(size, -1, dtype=np.int64)
                np.minimum.at(x_min, ids, xs)
                np.minimum.at(y_min, ids, ys)
                np.maximum.at(x_max, ids, xs)
                np.maximum.at(y_max, ids, ys)
                for index, actor in enumerate(actors, start=1):
                    if index >= size or counts[index] < min_pixels:
                        continue
                    boxes.append(
                        (actor.kind, int(x_min[index]), int(y_min[index]), int(x_max[index]), int(y_max[index]))
                    )

        self.program["fog_start"].value = cfg.fog_start_m
        self.program["fog_end"].value = cfg.far_m
        self.program["ambient"].value = cfg.ambient
        return frame, boxes

    def release(self) -> None:
        for obj in (self._static, self._static_buffer, self._dynamic, self._dynamic_buffer, self.fbo):
            if obj is not None:
                obj.release()
        self.ctx.release()
