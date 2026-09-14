"use strict";

/* Whole-brain point cloud: one GPU point per neuron at its real soma position.
 *
 * Raw WebGL2, no library. Positions upload once as a static buffer; every
 * control step only pushes a one-byte-per-neuron activation buffer, which is
 * what makes 166,700 live neurons affordable at 60 frames a second.
 *
 * Activation decays on the GPU-side value we keep in JS: a spike sets it to
 * full, and each frame multiplies it down, so a cell that just fired glows and
 * fades instead of strobing for a single frame. */

const VERTEX_SHADER = `#version 300 es
in vec3 position;
in float activation;
in float roleId;
in float measured;

uniform mat4 mvp;
uniform float pointScale;
uniform vec3 roleColors[8];
uniform float inferredDim;

out vec4 vColor;

void main() {
  gl_Position = mvp * vec4(position, 1.0);
  float depth = clamp(1.0 - gl_Position.z / gl_Position.w * 0.5, 0.35, 1.6);
  int role = int(roleId);
  vec3 base = role > 6 ? vec3(0.36, 0.42, 0.49) : roleColors[role];

  // Resting cells stay as faint structure so the anatomy is always legible;
  // activation lifts a cell towards white so spikes read at a glance.
  vec3 lit = mix(base * 0.55, mix(base, vec3(1.0), 0.55), activation);
  float alpha = mix(0.16, 1.0, activation) * mix(inferredDim, 1.0, measured);

  gl_PointSize = pointScale * depth * (1.0 + activation * 2.2);
  vColor = vec4(lit, alpha);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 fragColor;

void main() {
  vec2 offset = gl_PointCoord - vec2(0.5);
  float r2 = dot(offset, offset);
  if (r2 > 0.25) discard;
  float falloff = 1.0 - smoothstep(0.05, 0.25, r2);
  fragColor = vec4(vColor.rgb, vColor.a * falloff);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

/* ----------------------------------------------------------------- matrices */

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      out[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ]);
}

function orbit(distance, yaw, pitch) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  // Rotate about Y then X, then push the camera back along -Z.
  return new Float32Array([
    cy, sy * sp, -sy * cp, 0,
    0, cp, sp, 0,
    sy, -cy * sp, cy * cp, 0,
    0, 0, -distance, 1,
  ]);
}

/* ------------------------------------------------------------------- viewer */

export class BrainView {
  constructor(canvas, roleColors) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
    if (!this.gl) throw new Error("WebGL2 unavailable");
    this.roleColors = roleColors;
    this.count = 0;
    this.activation = null;
    this.decay = 0.82;
    this.yaw = 0.6;
    this.pitch = 0.15;
    this.distance = 1.75;
    this.autoSpin = true;
    this.dirty = false;
    this._initProgram();
    this._initInput();
  }

  _initProgram() {
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    this.program = program;
    this.uniform = {
      mvp: gl.getUniformLocation(program, "mvp"),
      pointScale: gl.getUniformLocation(program, "pointScale"),
      roleColors: gl.getUniformLocation(program, "roleColors"),
      inferredDim: gl.getUniformLocation(program, "inferredDim"),
    };
    this.vao = gl.createVertexArray();
  }

  _initInput() {
    const canvas = this.canvas;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      this.autoSpin = false;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      this.yaw += (event.clientX - lastX) * 0.008;
      this.pitch = Math.max(
        -1.45,
        Math.min(1.45, this.pitch + (event.clientY - lastY) * 0.008),
      );
      lastX = event.clientX;
      lastY = event.clientY;
      this.dirty = true;
    });
    const stop = () => {
      dragging = false;
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.distance = Math.max(1.2, Math.min(8, this.distance + event.deltaY * 0.002));
        this.dirty = true;
      },
      { passive: false },
    );
  }

  /** positions: Float32Array(n*3); roleIds, measured: Uint8Array(n) */
  load(positions, roleIds, measured) {
    const gl = this.gl;
    this.count = roleIds.length;
    this.activation = new Float32Array(this.count);

    gl.bindVertexArray(this.vao);
    const attach = (data, location, size, type, normalized) => {
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const index = gl.getAttribLocation(this.program, location);
      gl.enableVertexAttribArray(index);
      gl.vertexAttribPointer(index, size, type, normalized, 0, 0);
      return buffer;
    };
    attach(positions, "position", 3, gl.FLOAT, false);
    attach(roleIds, "roleId", 1, gl.UNSIGNED_BYTE, false);
    attach(measured, "measured", 1, gl.UNSIGNED_BYTE, false);

    this.activationBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.activationBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.activation, gl.DYNAMIC_DRAW);
    const index = gl.getAttribLocation(this.program, "activation");
    gl.enableVertexAttribArray(index);
    gl.vertexAttribPointer(index, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.dirty = true;
  }

  /** mask: Uint8Array bitfield, one bit per neuron, LSB-last (numpy packbits) */
  applySpikes(mask) {
    const activation = this.activation;
    if (!activation) return;
    for (let i = 0; i < activation.length; i += 1) {
      const bit = (mask[i >> 3] >> (7 - (i & 7))) & 1;
      activation[i] = bit ? 1.0 : activation[i] * this.decay;
    }
    this.dirty = true;
  }

  resize() {
    const canvas = this.canvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      return true;
    }
    return false;
  }

  render() {
    const gl = this.gl;
    if (this.autoSpin) {
      this.yaw += 0.0016;
      this.dirty = true;
    }
    const resized = this.resize();
    if (!this.count || (!this.dirty && !resized)) return;
    this.dirty = false;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.activationBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.activation);

    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clearColor(0.039, 0.051, 0.067, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    gl.useProgram(this.program);
    const aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    const mvp = multiply(
      perspective(Math.PI / 4, aspect, 0.1, 30),
      orbit(this.distance, this.yaw, this.pitch),
    );
    gl.uniformMatrix4fv(this.uniform.mvp, false, mvp);
    gl.uniform1f(
      this.uniform.pointScale,
      Math.max(1.1, (gl.drawingBufferHeight / 620) * 1.6),
    );
    gl.uniform3fv(this.uniform.roleColors, this.roleColors);
    gl.uniform1f(this.uniform.inferredDim, 0.45);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.bindVertexArray(null);
  }
}
