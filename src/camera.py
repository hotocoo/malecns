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
    CLASS_MOTORCYCLE,
    CLASS_PERSON,
    CLASS_STOP_SIGN,
    CLASS_TRAFFIC_LIGHT,
    Actor,
    RoadMesh,
    SceneConfig,
)
from lawreward import AMBER, GREEN, RED

VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec3 in_colour;
out vec3 v_colour;
out float v_depth;
void main() {
    vec4 clip = mvp * vec4(in_position, 1.0);
    gl_Position = clip;
    v_colour = in_colour;
    v_depth = clip.w;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec3 fog_colour;
uniform float fog_start;
uniform float fog_end;
in vec3 v_colour;
in float v_depth;
out vec4 f_colour;
void main() {
    float t = clamp((v_depth - fog_start) / max(fog_end - fog_start, 1.0), 0.0, 1.0);
    f_colour = vec4(mix(v_colour, fog_colour, t), 1.0);
}
"""

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
    ground_colour: tuple[float, float, float] = (0.30, 0.33, 0.26)
    fog_start_m: float = 120.0


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
        self._static_count = 0
        self._dynamic: moderngl.VertexArray | None = None
        self.program["fog_colour"].value = self.cfg.sky_colour
        self.program["fog_start"].value = self.cfg.fog_start_m
        self.program["fog_end"].value = self.cfg.far_m

    @property
    def renderer_name(self) -> str:
        return str(self.ctx.info["GL_RENDERER"])

    def _vertex_array(self, vertices: np.ndarray, colours: np.ndarray) -> moderngl.VertexArray:
        data = np.concatenate([vertices, colours], axis=1).astype("f4").tobytes()
        buffer = self.ctx.buffer(data)
        return self.ctx.vertex_array(self.program, [(buffer, "3f 3f", "in_position", "in_colour")])

    def set_static(self, road: RoadMesh, ground_height_m: float = -0.05, extent_m: float = 4000.0) -> None:
        """Upload the geometry that does not change: ground, road, markings."""
        ground = ground_plane(extent_m, ground_height_m)
        ground_colour = np.tile(np.array(self.cfg.ground_colour, dtype=np.float32), (ground.shape[0], 1))
        vertices = np.concatenate([ground, road.surface, road.markings])
        colours = np.concatenate([ground_colour, road.surface_colour, road.markings_colour])
        self._static = self._vertex_array(vertices, colours)
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
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[int]]:
        """Triangles and colours for `actors`, optionally with each actor's vertex count.

        The label pass needs to know which vertices belong to which actor, and
        rebuilding the geometry per actor to find out costs more than the draw
        itself; `with_spans` hands that back from the one pass that built it.
        """
        chunks: list[np.ndarray] = []
        colours: list[np.ndarray] = []
        spans: list[int] = []
        for actor in actors:
            before = sum(c.shape[0] for c in chunks)
            if actor.kind == CLASS_TRAFFIC_LIGHT:
                # Pole, housing, then the lit lens as a separate bright face so
                # the phase is visible from the road rather than implied.
                pole = box_triangles(
                    np.array([actor.pos[0], actor.pos[1], 0.0]),
                    actor.heading,
                    np.array([0.12, 0.12, actor.pos[2]]),
                )
                chunks.append(pole)
                colours.append(np.tile(np.array([0.35, 0.35, 0.35], dtype=np.float32), (pole.shape[0], 1)))
                housing = box_triangles(actor.pos - np.array([0.0, 0.0, actor.size[2]]), actor.heading, actor.size)
                chunks.append(housing)
                colours.append(np.tile(actor.colour, (housing.shape[0], 1)))
                lens_size = np.array([0.06, actor.size[1] * 0.7, actor.size[2] * 0.28])
                lens_pos = actor.pos - np.array([0.0, 0.0, actor.size[2] * 0.45])
                offset = np.array([np.cos(actor.heading), np.sin(actor.heading), 0.0]) * (actor.size[0] / 2.0)
                lens = box_triangles(lens_pos + offset, actor.heading, lens_size)
                chunks.append(lens)
                colours.append(
                    np.tile(np.array(SIGNAL_LENS.get(actor.state, SIGNAL_LENS[RED]), dtype=np.float32), (lens.shape[0], 1))
                )
                spans.append(sum(c.shape[0] for c in chunks) - before)
                continue
            if actor.kind == CLASS_STOP_SIGN:
                pole = box_triangles(
                    np.array([actor.pos[0], actor.pos[1], 0.0]),
                    actor.heading,
                    np.array([0.09, 0.09, actor.pos[2]]),
                )
                chunks.append(pole)
                colours.append(np.tile(np.array([0.45, 0.45, 0.45], dtype=np.float32), (pole.shape[0], 1)))
                plate = box_triangles(actor.pos - np.array([0.0, 0.0, actor.size[2] / 2.0]), actor.heading, actor.size)
                chunks.append(plate)
                colours.append(np.tile(actor.colour, (plate.shape[0], 1)))
                spans.append(sum(c.shape[0] for c in chunks) - before)
                continue

            body = box_triangles(actor.pos, actor.heading, actor.size)
            chunks.append(body)
            colours.append(np.tile(actor.colour, (body.shape[0], 1)))
            if actor.kind == CLASS_PERSON:
                head = box_triangles(
                    actor.pos + np.array([0.0, 0.0, actor.size[2]]),
                    actor.heading,
                    np.array([0.22, 0.22, 0.24]),
                )
                chunks.append(head)
                colours.append(np.tile(np.array([0.72, 0.58, 0.48], dtype=np.float32), (head.shape[0], 1)))
            elif actor.kind == CLASS_MOTORCYCLE:
                rider = box_triangles(
                    actor.pos + np.array([0.0, 0.0, actor.size[2]]),
                    actor.heading,
                    np.array([0.4, 0.45, 0.85]),
                )
                chunks.append(rider)
                colours.append(np.tile(np.array([0.18, 0.18, 0.22], dtype=np.float32), (rider.shape[0], 1)))
            spans.append(sum(c.shape[0] for c in chunks) - before)

        if not chunks:
            empty = np.zeros((0, 3), dtype=np.float32)
            return (empty, empty, spans) if with_spans else (empty, empty)
        vertices, colours_out = np.concatenate(chunks), np.concatenate(colours)
        return (vertices, colours_out, spans) if with_spans else (vertices, colours_out)

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

        vertices, colours = self._actor_geometry(self.visible(pos_xy, heading, actors))
        if vertices.shape[0]:
            if self._dynamic is not None:
                self._dynamic.release()
            self._dynamic = self._vertex_array(vertices, colours)
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

        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, depth=1.0)
        vertices, _, spans = self._actor_geometry(actors, with_spans=True)
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
            if self._dynamic is not None:
                self._dynamic.release()
            self._dynamic = self._vertex_array(vertices, colours)
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
        return frame, boxes

    def release(self) -> None:
        for obj in (self._static, self._dynamic, self.fbo):
            if obj is not None:
                obj.release()
        self.ctx.release()
