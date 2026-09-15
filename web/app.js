"use strict";

/* Page controller. Nothing about the simulation is written here: labels,
 * colours, thresholds, units, vehicle and track figures all arrive from the
 * server (/api/meta carries the viewer config and the frame schema). Panels
 * are built from those descriptions, so a new telemetry field on the server
 * appears on the page without touching this file. */

import { BrainView } from "/brain.js";
import { DriveView } from "/drive.js";
import { drawFirstPerson } from "/firstperson.js";

const state = {
  track: null,
  meta: null,
  cfg: null, // meta.config
  ui: null, // meta.config.ui
  trail: [],
  bandOf: [],
  sampleSize: 0,
  brain: null,
  drive: null,
  frame: null,
  metaGeneration: null,
  ended: null,
  curveKeys: [],
  training: null,
  curve: null,
  trainers: [],
  trainerId: null,
  compareTrainerIds: [],
  chartView: { xScale: 1, xOffset: 0, yScale: 1, yOffset: 0 },
};

const el = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(d));
const roleColour = (role) => (state.ui.role_colours[role] || state.ui.role_fallback_colour);
const label = (s) => String(s).replace(/_/g, " ");

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Size a canvas' backing store to its CSS box (device pixels). */
function fitCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const h = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

/* ---------------------------------------------------------------- minimap */

function trackTransform(canvas, extent) {
  const size = Math.min(canvas.width, canvas.height);
  const scale = (size * 0.94) / (2 * extent);
  return { x: (wx) => canvas.width / 2 + wx * scale, y: (wy) => canvas.height / 2 - wy * scale, scale };
}

/* The road ribbon and its dashed centreline are 2,048-point paths; stroking
 * them (dashes especially) every frame was the single most expensive draw on
 * the page, so they are rendered once per canvas size and blitted. */
const minimap = { base: null, width: 0, height: 0 };

function trackBase(canvas) {
  if (minimap.base && minimap.width === canvas.width && minimap.height === canvas.height) return minimap.base;
  const base = document.createElement("canvas");
  base.width = canvas.width;
  base.height = canvas.height;
  const ctx = base.getContext("2d");
  const { centerline, halfwidth, extent } = state.track;
  const t = trackTransform(canvas, extent);
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  centerline.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(t.x(x), t.y(y)) : ctx.lineTo(t.x(x), t.y(y))));
  ctx.closePath();
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.lineWidth = 2 * halfwidth * t.scale;
  ctx.strokeStyle = cssVar("--road");
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.strokeStyle = cssVar("--line");
  ctx.setLineDash([6, 12]);
  ctx.stroke();
  ctx.setLineDash([]);
  minimap.base = base;
  minimap.width = canvas.width;
  minimap.height = canvas.height;
  return base;
}

function drawTrack(frame) {
  const canvas = el("track-canvas");
  fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const { extent, n_rays, fov_deg, max_range } = state.track;
  const t = trackTransform(canvas, extent);
  ctx.drawImage(trackBase(canvas), 0, 0);
  if (!frame) return;

  if (state.trail.length > 1) {
    ctx.beginPath();
    state.trail.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(t.x(x), t.y(y)) : ctx.lineTo(t.x(x), t.y(y))));
    ctx.strokeStyle = cssVar("--accent");
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  const [cx, cy] = frame.pos;
  const px = t.x(cx);
  const py = t.y(cy);
  const fov = (fov_deg * Math.PI) / 180;
  const warn = hexToRgb(cssVar("--warn")).map((v) => Math.round(v * 255));
  frame.lidar.forEach((norm, i) => {
    const offset = n_rays === 1 ? 0 : fov / 2 - (fov * i) / (n_rays - 1); // ray 0 looks left (car_env.py)
    const angle = frame.heading + offset;
    const reach = norm * max_range;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(t.x(cx + Math.cos(angle) * reach), t.y(cy + Math.sin(angle) * reach));
    ctx.strokeStyle = `rgba(${warn.join(",")}, ${0.2 + (1 - norm) * 0.65})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // every car in the fleet as a dot coloured by its ES island; the followed car keeps the footprint below
  if (frame.fleet) {
    frame.fleet.forEach((c, k) => {
      if (k === frame.follow) return;
      ctx.beginPath();
      ctx.arc(t.x(c.pos[0]), t.y(c.pos[1]), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = islandColour(c.island);
      ctx.globalAlpha = c.done_reason ? 0.35 : 0.95;
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  }
  // car footprint at true scale, oriented
  const car = state.track.car;
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-frame.heading);
  ctx.fillStyle = cssVar("--text");
  const L = Math.max(car.length * t.scale, 8);
  const W = Math.max(car.width * t.scale, 5);
  ctx.beginPath();
  ctx.moveTo(L / 2, 0);
  ctx.lineTo(-L / 2, W / 2);
  ctx.lineTo(-L / 2, -W / 2);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/* ----------------------------------------------------------------- raster */

const raster = { ctx: null, buffer: null };

function initRaster() {
  initFleet();
  const canvas = el("raster-canvas");
  fitCanvas(canvas);
  raster.ctx = canvas.getContext("2d");
  raster.ctx.fillStyle = cssVar("--canvas");
  raster.ctx.fillRect(0, 0, canvas.width, canvas.height);
  raster.buffer = document.createElement("canvas");
  raster.buffer.width = canvas.width;
  raster.buffer.height = canvas.height;

  const legend = el("raster-legend");
  legend.innerHTML = "";
  state.meta.bands.forEach((band) => {
    const li = document.createElement("li");
    const swatch = document.createElement("i");
    swatch.style.background = roleColour(band.role);
    li.append(swatch, `${label(band.role)} ${band.count}/${band.total.toLocaleString()}`);
    legend.append(li);
  });
  state.sampleSize = state.meta.bands.reduce((sum, b) => sum + b.count, 0);
  state.bandOf = new Array(state.sampleSize);
  state.meta.bands.forEach((band) => {
    for (let i = 0; i < band.count; i += 1) state.bandOf[band.start + i] = band.role;
  });
  el("raster-label").textContent = `spikes · ${state.sampleSize} sampled neurons, one column per ${state.meta.dt_ms * state.meta.substeps} ms step`;
}

function drawRaster(frame) {
  const ctx = raster.ctx;
  const canvas = ctx.canvas;
  if (fitCanvas(canvas)) {
    raster.buffer.width = canvas.width;
    raster.buffer.height = canvas.height;
  }
  const step = state.cfg.raster.column_px;
  raster.buffer.getContext("2d").drawImage(canvas, 0, 0);
  ctx.drawImage(raster.buffer, -step, 0);
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(canvas.width - step, 0, step, canvas.height);
  const rowHeight = canvas.height / state.sampleSize;
  frame.fired.forEach((i) => {
    ctx.fillStyle = roleColour(state.bandOf[i]);
    ctx.fillRect(canvas.width - step, i * rowHeight, step, Math.max(1, rowHeight));
  });
}

/* ------------------------------------------------------------------ fleet */

const ISLAND_COLOURS = ["#3ddc97", "#e8c44a", "#4aa3e8", "#e05c6e", "#b58ce0", "#e8894a", "#8494a4", "#f2f2f2"];
const islandColour = (i) => ISLAND_COLOURS[(i || 0) % ISLAND_COLOURS.length];
const fleet = { ctx: null, buffer: null, rows: 0 };

function initFleet() {
  const canvas = el("fleet-canvas");
  if (!canvas) return;
  fitCanvas(canvas);
  fleet.ctx = canvas.getContext("2d");
  fleet.buffer = document.createElement("canvas");
  fleet.buffer.width = canvas.width;
  fleet.buffer.height = canvas.height;
  fleet.ctx.fillStyle = cssVar("--canvas");
  fleet.ctx.fillRect(0, 0, canvas.width, canvas.height);
  const m = state.meta;
  el("fleet-label").textContent = `all ${m.cars || 1} cars · brains side by side · ${m.islands || 1} ES island${(m.islands || 1) > 1 ? "s" : ""} · click a row to follow`;
  canvas.onclick = (ev) => {
    if (!state.frame || !state.frame.fleet) return;
    const rowH = canvas.clientHeight / state.frame.fleet.length;
    const k = Math.min(state.frame.fleet.length - 1, Math.floor(ev.offsetY / rowH));
    fetch(`/api/follow?car=${k}`).catch(() => {});
  };
}

/** One row per car: label on the left, that car's sampled-neuron activity scrolling on the right. */
function drawFleet(frame) {
  const ctx = fleet.ctx;
  if (!ctx || !frame.fleet) return;
  const canvas = ctx.canvas;
  if (fitCanvas(canvas)) {
    fleet.buffer.width = canvas.width;
    fleet.buffer.height = canvas.height;
  }
  const W = canvas.width;
  const H = canvas.height;
  const n = frame.fleet.length;
  const rowH = H / n;
  const labelW = Math.min(W * 0.42, 260 * (window.devicePixelRatio || 1));
  const step = state.cfg.raster.column_px;
  // scroll the raster area left by one column
  fleet.buffer.getContext("2d").drawImage(canvas, 0, 0);
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(fleet.buffer, labelW, 0, W - labelW, H, labelW - step, 0, W - labelW, H);
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(W - step, 0, step, H);
  const font = `${Math.max(9, Math.min(13, rowH * 0.28))}px ${cssVar("--mono")}`;
  frame.fleet.forEach((c, k) => {
    const y = k * rowH;
    // newest column of this car's brain: 96 sampled neurons in 8 density bands
    const bytes = decodeMask(c.raster); // numpy packbits: big-endian bit order
    const nBits = bytes.length * 8;
    const bitAt = (i) => (bytes[i >> 3] >> (7 - (i & 7))) & 1;
    const bands = 8;
    const per = Math.max(1, Math.floor(nBits / bands));
    const colour = hexToRgb(islandColour(c.island)).map((v) => Math.round(v * 255));
    for (let b = 0; b < bands; b++) {
      let hits = 0;
      for (let i = b * per; i < Math.min(nBits, (b + 1) * per); i++) hits += bitAt(i);
      const level = hits / per;
      if (level > 0) {
        ctx.fillStyle = `rgba(${colour.join(",")}, ${0.25 + 0.75 * Math.min(1, level * 3)})`;
        ctx.fillRect(W - step, y + (b * rowH) / bands, step, Math.max(1, rowH / bands - 1));
      }
    }
    // label block
    ctx.fillStyle = k === frame.follow ? "rgba(255,255,255,0.08)" : cssVar("--canvas");
    ctx.fillRect(0, y, labelW, rowH);
    ctx.fillStyle = islandColour(c.island);
    ctx.fillRect(0, y + 2, 4, rowH - 4);
    ctx.fillStyle = k === frame.follow ? cssVar("--text") : cssVar("--muted");
    ctx.font = font;
    ctx.textAlign = "left";
    const ended = c.done_reason ? (state.meta.done_names[String(c.done_reason)] || "ended").toUpperCase() : "";
    ctx.fillText(`${k === frame.follow ? "▶" : " "} car ${k} · island ${c.island} · start ${fmt(c.start_fraction, 2)}`, 10, y + rowH * 0.42);
    ctx.fillText(`${fmt(c.laps, 3)} laps · ${fmt(c.speed * state.ui.kmh_per_mps, 0)} km/h · ${c.spiking.toLocaleString()} spk · DN ${fmt(c.dn_hz, 1)} Hz ${ended}`, 10, y + rowH * 0.8);
    ctx.strokeStyle = cssVar("--line");
    ctx.beginPath();
    ctx.moveTo(0, y + rowH);
    ctx.lineTo(W, y + rowH);
    ctx.stroke();
  });
}

/* --------------------------------------------------------------- bar lists */

function buildList(container, items) {
  container.innerHTML = "";
  return items.map((item) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = item.label;
    name.title = item.title || item.label;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.background = item.color;
    bar.append(fill);
    const out = document.createElement("output");
    out.textContent = "-";
    li.append(name, bar, out);
    container.append(li);
    return { fill, out };
  });
}

let rateRows = [];
let dnRows = [];

function initLists() {
  const m = state.meta;
  state.rateRoles = Object.keys(m.roles);
  el("rates-heading").textContent = `population rates · ${m.neurons.toLocaleString()} neurons by role`;
  rateRows = buildList(
    el("rate-list"),
    state.rateRoles.map((role) => ({
      label: label(role),
      title: `${role}: ${m.roles[role].toLocaleString()} neurons`,
      color: roleColour(role),
    })),
  );
  const r = m.readout || {};
  el("dn-heading").textContent = m.mode === "replay"
    ? `most active output neurons in this recording`
    : `output neurons by steering influence · readout over ${label(r.roles || "")} (${(r.neurons || 0).toLocaleString()} cells to ${r.channels || 0} channels)`;
  dnRows = buildList(
    el("dn-list"),
    m.top_dn.map((dn) => ({
      label: `${dn.type === "None" ? String(dn.body) : dn.type}${dn.role ? ` · ${label(dn.role)}` : ""}`,
      title: `body ${dn.body}; effective steering weight ${dn.weight}`,
      color: dn.weight >= 0 ? roleColour(dn.role || "descending") : state.ui.negative_weight_colour,
    })),
  );
}

function updateLists(frame) {
  const maxRate = Math.max(state.ui.rate_bar_floor_hz, ...Object.values(frame.rates));
  state.rateRoles.forEach((role, i) => {
    const hz = frame.rates[role] ?? 0;
    rateRows[i].fill.style.width = `${(hz / maxRate) * 100}%`;
    rateRows[i].out.textContent = `${fmt(hz, 1)} Hz`;
  });
  const maxDn = Math.max(1, ...frame.dn.map(Math.abs));
  frame.dn.forEach((hz, i) => {
    if (!dnRows[i]) return;
    dnRows[i].fill.style.width = `${(Math.abs(hz) / maxDn) * 100}%`;
    dnRows[i].out.textContent = `${fmt(hz, 1)} Hz`;
  });
}

/* ------------------------------------------------------- schema telemetry */

const telemetryRows = new Map(); // key -> {value, bar}
const gaugeRows = new Map();

function schemaByGroup() {
  const groups = new Map();
  state.meta.frame_schema.forEach((f) => {
    if (!groups.has(f.group)) groups.set(f.group, []);
    groups.get(f.group).push(f);
  });
  return groups;
}

function valueText(field, v) {
  if (v === undefined || v === null) return "-";
  if (Array.isArray(v)) return v.map((x) => fmt(x, field.digits ?? 0)).join(" ");
  let x = Number(v);
  if (field.scale === "kmh") x *= state.ui.kmh_per_mps;
  return `${fmt(x, field.digits ?? 2)}${field.unit ? ` ${field.unit}` : ""}`;
}

function initTelemetry() {
  el("telemetry-heading").textContent = "live telemetry · every field the simulation reports each control step";
  const container = el("telemetry");
  container.innerHTML = "";
  telemetryRows.clear();
  schemaByGroup().forEach((fields, group) => {
    const box = document.createElement("div");
    box.className = "tgroup";
    const h = document.createElement("h3");
    h.textContent = group;
    box.append(h);
    const dl = document.createElement("dl");
    dl.className = "kv";
    fields.forEach((f) => {
      if (f.kind === "rays") return; // drawn in the fly panel
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = f.label;
      const dd = document.createElement("dd");
      dd.textContent = "-";
      row.append(dt, dd);
      dl.append(row);
      telemetryRows.set(f.key, { value: dd, field: f });
    });
    box.append(dl);
    container.append(box);
  });

  // gauges for the commands, from the schema's bipolar fly fields
  const gauges = el("gauges");
  gauges.innerHTML = "";
  gaugeRows.clear();
  state.meta.frame_schema
    .filter((f) => f.group === "fly" && f.bipolar && (f.key === "steer" || f.key === "pedal"))
    .forEach((f) => {
      const g = document.createElement("div");
      g.className = "gauge";
      const name = document.createElement("span");
      name.className = "gauge-label";
      name.textContent = f.label;
      const left = document.createElement("span");
      left.className = "gauge-end";
      left.textContent = f.key === "steer" ? "R" : "brake";
      const bar = document.createElement("div");
      bar.className = "bar bipolar";
      const fill = document.createElement("i");
      bar.append(fill);
      const right = document.createElement("span");
      right.className = "gauge-end";
      right.textContent = f.key === "steer" ? "L" : "gas";
      const out = document.createElement("output");
      g.append(name, left, bar, right, out);
      gauges.append(g);
      gaugeRows.set(f.key, { fill, out, field: f });
    });

  const strip = el("strip-readout");
  strip.innerHTML = "";
  ["laps", "spiking", "fps", "headroom"].forEach((key) => {
    const f = state.meta.frame_schema.find((x) => x.key === key);
    if (!f) return;
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = f.label;
    const dd = document.createElement("dd");
    dd.id = `strip-${key}`;
    dd.textContent = "-";
    row.append(dt, dd);
    strip.append(row);
  });
  const link = document.createElement("div");
  link.innerHTML = `<dt>link</dt><dd id="stat-link" class="bad">connecting</dd>`;
  strip.append(link);
}

function updateTelemetry(frame) {
  telemetryRows.forEach(({ value, field }, key) => {
    value.textContent = valueText(field, frame[key]);
    if (field.bipolar && typeof frame[key] === "number") value.className = frame[key] < 0 ? "neg" : "pos";
  });
  gaugeRows.forEach(({ fill, out }, key) => {
    const v = frame[key] ?? 0;
    // steer > 0 is a LEFT turn (counter-clockwise, car_env.py): the bar grows leftwards for positive
    const grow = key === "steer" ? -v : v;
    fill.style.width = `${Math.abs(grow) * 50}%`;
    fill.style.left = grow >= 0 ? "50%" : `${50 - Math.abs(grow) * 50}%`;
    fill.style.background = key === "pedal" && v < 0 ? cssVar("--bad") : "";
    out.textContent = fmt(v);
  });
  ["laps", "spiking", "fps", "headroom"].forEach((key) => {
    const f = state.meta.frame_schema.find((x) => x.key === key);
    const dd = el(`strip-${key}`);
    if (f && dd) dd.textContent = valueText(f, frame[key]);
  });
}

/* ---------------------------------------------------------------- fly panel */

function initFly() {
  const m = state.meta;
  const rays = state.track.n_rays;
  el("fly-heading").textContent = `what the fly is doing · ${m.roles.visual_projection?.toLocaleString() || "?"} visual projection neurons in ${rays} groups (one per lidar ray) drive the ${m.neurons.toLocaleString()}-neuron connectome; ${(m.readout?.neurons || 0).toLocaleString()} ${label(m.readout?.roles || "output")} neurons are read out as steering and pedal`;
  el("eyes-caption").textContent = `eyes: Poisson drive per ray group (Hz); ray 0 is the leftmost of ${rays} across ${state.track.fov_deg}° of view, ${state.track.max_range} m reach`;
  el("fp-caption").textContent = `first person: one wall column per eye group (ray 0 left), height = true distance, brightness = proximity the group is driven with (road-relative), white = looming; horizon rolls with lateral load, pitches with the pedal, edges glow as the body nears the barrier; bars = what the fly feels and intends`;
  el("controls-caption").textContent = `commands: steering wheel = steer command (${fmt(state.track.car_cfg.max_steer_rad, 2)} rad lock), pedals = throttle / brake`;
}

function drawFly(frame) {
  const canvas = el("fp-canvas");
  if (!canvas) return;
  fitCanvas(canvas);
  drawFirstPerson(canvas, frame, { track: state.track, meta: state.meta, ui: state.ui, cssVar, roleColour, fmt });
}

function drawEyes(frame) {
  const canvas = el("eyes-canvas");
  fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(0, 0, W, H);
  const rays = frame.sensory_hz || [];
  const n = rays.length;
  if (!n) return;
  const maxHz = state.meta.agent_cfg?.max_input_hz || Math.max(1, ...rays);
  const cx = W / 2;
  const cy = H * 0.92;
  const R = Math.min(W * 0.46, H * 0.84);
  const fov = (state.track.fov_deg * Math.PI) / 180;
  const accent = hexToRgb(roleColour("visual_projection")).map((v) => Math.round(v * 255));
  // sectors: ray 0 on the left of the picture (car's left), fanning forward
  rays.forEach((hz, i) => {
    const a0 = Math.PI / 2 + fov / 2 - (fov * i) / n; // screen angle of sector start (0 rad = right, pi/2 = up)
    const a1 = a0 - fov / n;
    const level = Math.min(1, hz / maxHz);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, -a0, -a1, false);
    ctx.closePath();
    ctx.fillStyle = `rgba(${accent.join(",")}, ${0.12 + 0.85 * level})`;
    ctx.fill();
    ctx.strokeStyle = cssVar("--line");
    ctx.stroke();
    // lidar return length inside the sector
    const dist = frame.lidar[i];
    const mid = (a0 + a1) / 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(mid) * R * dist, cy - Math.sin(mid) * R * dist);
    ctx.strokeStyle = cssVar("--warn");
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = cssVar("--text");
    ctx.font = `${Math.max(10, H * 0.06)}px ${cssVar("--mono")}`;
    ctx.textAlign = "center";
    ctx.fillText(`${fmt(hz, 0)}`, cx + Math.cos(mid) * R * 0.78, cy - Math.sin(mid) * R * 0.78);
  });
  ctx.fillStyle = cssVar("--muted");
  ctx.textAlign = "left";
  ctx.fillText("L", 6, cy);
  ctx.textAlign = "right";
  ctx.fillText("R", W - 6, cy);
  ctx.textAlign = "center";
  ctx.fillText(`speed sense ${fmt(frame.speed_hz, 0)} Hz · looming gain ${state.meta.param_blocks?.loom_gain ? fmt(state.meta.param_blocks.loom_gain.values?.[0], 2) : "-"}`, cx, H * 0.995);
}

function drawControls(frame) {
  const canvas = el("controls-canvas");
  fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(0, 0, W, H);
  const font = `${Math.max(10, H * 0.06)}px ${cssVar("--mono")}`;
  ctx.font = font;

  // steering wheel: rim rotates by the command (left = counter-clockwise on screen)
  const wx = W * 0.32;
  const wy = H * 0.5;
  const r = Math.min(W * 0.22, H * 0.36);
  const lock = state.track.car_cfg.max_steer_rad;
  const wheelTurn = -frame.steer * Math.PI * 0.75; // visual rotation of the rim for full lock
  ctx.save();
  ctx.translate(wx, wy);
  ctx.rotate(wheelTurn);
  ctx.strokeStyle = cssVar("--text");
  ctx.lineWidth = Math.max(3, r * 0.12);
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = Math.max(2, r * 0.08);
  [Math.PI / 2, Math.PI * 1.15, Math.PI * 1.85].forEach((a) => {
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
    ctx.stroke();
  });
  ctx.fillStyle = cssVar("--accent");
  ctx.beginPath();
  ctx.arc(0, -r, Math.max(3, r * 0.1), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  ctx.fillStyle = cssVar("--muted");
  ctx.textAlign = "center";
  ctx.fillText(`steer ${fmt(frame.steer)} (${fmt(frame.steer_actual * lock, 2)} rad at wheels)`, wx, wy + r + H * 0.09);
  ctx.fillText(`pre-activation ${fmt(frame.motor_steer)}`, wx, wy + r + H * 0.17);

  // pedals: brake (left) and throttle (right), pressed depth = command
  const px = W * 0.66;
  const pw = W * 0.1;
  const ph = H * 0.5;
  const top = H * 0.2;
  const draw = (x, level, colour, name) => {
    ctx.strokeStyle = cssVar("--line");
    ctx.strokeRect(x, top, pw, ph);
    ctx.fillStyle = colour;
    ctx.fillRect(x, top + ph * (1 - level), pw, ph * level);
    ctx.fillStyle = cssVar("--muted");
    ctx.fillText(name, x + pw / 2, top + ph + H * 0.09);
    ctx.fillText(fmt(level), x + pw / 2, top + ph + H * 0.16);
  };
  draw(px, frame.brake ?? Math.max(0, -frame.pedal), cssVar("--bad"), "brake");
  draw(px + pw * 1.6, frame.throttle ?? Math.max(0, frame.pedal), cssVar("--accent"), "throttle");
  ctx.fillStyle = cssVar("--muted");
  ctx.fillText(`pre-activation ${fmt(frame.motor_pedal)}`, px + pw * 1.3, top - H * 0.035);
  const name = state.meta.done_names[String(frame.done_reason)] || "alive";
  ctx.fillStyle = frame.done_reason > 0 ? cssVar("--bad") : cssVar("--accent");
  ctx.fillText(`${name.toUpperCase()} · ${fmt(frame.speed * state.ui.kmh_per_mps, 0)} km/h · ${fmt(frame.lat_g, 1)} g · clearance ${fmt(frame.clearance, 2)} m`, W / 2, H * 0.06);
}

function updateFlyTelemetry(frame) {
  const box = el("fly-telemetry");
  const fields = state.meta.frame_schema.filter((f) => f.group === "fly" && f.kind !== "rays");
  if (!box.dataset.built) {
    box.innerHTML = "";
    const dl = document.createElement("dl");
    dl.className = "kv";
    fields.forEach((f) => {
      const row = document.createElement("div");
      row.innerHTML = `<dt>${f.label}</dt><dd data-key="${f.key}">-</dd>`;
      dl.append(row);
    });
    box.append(dl);
    box.dataset.built = "1";
  }
  fields.forEach((f) => {
    const dd = box.querySelector(`dd[data-key="${f.key}"]`);
    if (dd) dd.textContent = valueText(f, frame[f.key]);
  });
}

/* ------------------------------------------------------------------- curve */

function initCurveControls(keys) {
  const box = el("curve-controls");
  const defaults = new Set([...state.ui.curve_default_keys, ...state.ui.curve_secondary_keys]);
  // A fleet comparison is primarily a like-for-like comparison. Starting with
  // one metric prevents N trainers × M metrics from turning the graph into an
  // unreadable bundle of traces; additional metrics remain one checkbox away.
  const preferred = state.compareTrainerIds.length > 1 && keys.includes("fitness_best")
    ? ["fitness_best"]
    : state.ui.curve_default_keys;
  const existingKeys = [...box.querySelectorAll("input")].map((input) => input.value);
  const keySet = new Set(keys);
  const needsRebuild = existingKeys.length !== keys.length || existingKeys.some((key) => !keySet.has(key));
  if (!needsRebuild && box.dataset.built) return;
  const previous = new Set(state.curveKeys);
  state.curveKeys = (previous.size ? keys.filter((key) => previous.has(key)) : preferred.filter((k) => keys.includes(k)));
  if (!state.curveKeys.length) state.curveKeys = preferred.filter((k) => keys.includes(k));
  box.innerHTML = "";
  box.dataset.built = "1";
  keys.forEach((key) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = key;
    cb.checked = state.curveKeys.includes(key);
    cb.addEventListener("change", () => {
      state.curveKeys = [...box.querySelectorAll("input:checked")].map((c) => c.value);
      refreshCurve();
    });
    lab.append(cb, ` ${label(key)}`);
    if (!defaults.has(key)) lab.className = "dim";
    box.append(lab);
  });
}

function trainerColour(trainerId, trainers) {
  const index = Math.max(0, (trainers || []).indexOf(trainerId));
  const count = Math.max(1, (trainers || []).length);
  const hue = Math.round((index * 360) / count + 205) % 360;
  return `hsl(${hue} 72% 64%)`;
}

function trainerDash(trainerId, trainers) {
  const index = Math.max(0, (trainers || []).indexOf(trainerId));
  return [[], [7, 4], [2, 4], [10, 3, 2, 3], [4, 3, 1, 3]][index % 5];
}

function resetChartView() {
  state.chartView = { xScale: 1, xOffset: 0, yScale: 1, yOffset: 0 };
  if (state.curve) drawCurve(state.curve);
}

function fitChartView() {
  resetChartView();
}

function chartPointBudget() {
  const canvas = el("curve-canvas");
  if (!canvas) return state.cfg?.curve_points || 400;
  // Request enough samples to preserve visible detail at the current canvas
  // width without making huge historical logs expensive to transfer.
  const width = Math.max(320, Math.round(canvas.clientWidth || 0));
  return Math.min(10000, Math.max(state.cfg?.curve_points || 400, width * 4));
}

function updateChartDataNote(curve) {
  const note = el("chart-data-note");
  if (!note) return;
  const seriesCount = Object.keys(curve.series || {}).length;
  const trainerCount = (curve.trainers || []).length;
  const points = Math.max(0, ...Object.values(curve.series || {}).map((v) => Array.isArray(v) ? v.length : 0));
  note.textContent = `${points.toLocaleString()} samples · ${seriesCount.toLocaleString()} series${trainerCount ? ` · ${trainerCount.toLocaleString()} trainers` : ""}`;
}

function initChartInteractions() {
  const shell = el("chart-shell");
  const canvas = el("curve-canvas");
  const toolbar = document.querySelector(".chart-toolbar");
  if (!shell || !canvas || !toolbar || shell.dataset.bound) return;
  shell.dataset.bound = "1";
  toolbar.querySelectorAll("[data-chart-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.chartAction;
      if (action === "fit" || action === "reset") fitChartView();
      if (action === "zoom-in") {
        state.chartView.xScale = Math.min(12, state.chartView.xScale * 1.25);
        if (state.curve) drawCurve(state.curve);
      }
      if (action === "zoom-out") {
        state.chartView.xScale = Math.max(1, state.chartView.xScale / 1.25);
        if (state.curve) drawCurve(state.curve);
      }
      if (action === "fullscreen") {
        if (shell.requestFullscreen) shell.requestFullscreen();
        else shell.classList.toggle("expanded");
      }
    });
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.chartView.xScale = Math.max(1, Math.min(12, state.chartView.xScale * (event.deltaY < 0 ? 1.15 : 0.87)));
    if (state.curve) drawCurve(state.curve);
  }, { passive: false });
  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    drag = { x: event.clientX, offset: state.chartView.xOffset };
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const generationSets = state.curve?.multi
      ? Object.values(state.curve.generation || {})
      : [state.curve?.generation || []];
    const generations = generationSets.flat().map(Number).filter(Number.isFinite);
    const span = generations.length > 1
      ? Math.max(1, Math.max(...generations) - Math.min(...generations))
      : 1;
    state.chartView.xOffset = Math.max(-span, Math.min(span, drag.offset - (event.clientX - drag.x) * span / Math.max(1, canvas.clientWidth)));
    if (state.curve) drawCurve(state.curve);
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointercancel", () => { drag = null; });
}

function drawCurve(curve) {
  const canvas = el("curve-canvas");
  fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const pad = { left: 62, right: 18, top: 18, bottom: 34 };
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const series = Object.entries(curve.series || {})
    .map(([key, values]) => [key, values.map((value) => Number.isFinite(value) ? value : null)])
    .filter(([, v]) => v.some((x) => x !== null));
  const legend = el("chart-legend");
  legend.innerHTML = "";
  if (!series.length) {
    el("chart-overlay").textContent = "no numeric data";
    return;
  }
  ctx.font = `11px ${cssVar("--mono")}`;
  const width = Math.max(1, canvas.width - pad.left - pad.right);
  const height = Math.max(1, canvas.height - pad.top - pad.bottom);
  // Scale each metric from its own distribution. A corrupt scalar in one
  // trace (seen in long historical logs) must never flatten the useful traces
  // from the other metrics/trainers. Keep the actual extrema only when they
  // are plausible; otherwise clip the display at a robust 1st/99th percentile
  // and report how many points were clipped.
  const bounds = new Map();
  const metricGroups = new Map();
  let clipped = 0;
  series.forEach(([key, vals]) => {
    const metric = curve.multi ? key.split(": ").slice(1).join(": ") : key;
    if (!metricGroups.has(metric)) metricGroups.set(metric, []);
    metricGroups.get(metric).push(...vals.filter((x) => Number.isFinite(x)));
  });
  metricGroups.forEach((groupValues, metric) => {
    const values = groupValues.sort((a, b) => a - b);
    const rawLo = values[0];
    const rawHi = values[values.length - 1];
    const q = (fraction) => values[Math.min(values.length - 1, Math.floor((values.length - 1) * fraction))];
    const robustLo = q(0.01);
    const robustHi = q(0.99);
    const robustSpan = robustHi - robustLo;
    let lo = rawLo;
    let hi = rawHi;
    if (robustSpan > 0 && (rawHi - rawLo) > Math.max(100 * robustSpan, 1e6)) {
      lo = robustLo;
      hi = robustHi;
      series.forEach(([key, vals]) => {
        const keyMetric = curve.multi ? key.split(": ").slice(1).join(": ") : key;
        if (keyMetric !== metric) return;
        vals.forEach((v) => { if (v !== null && (v < lo || v > hi)) clipped += 1; });
      });
    }
    series.forEach(([key]) => {
      const keyMetric = curve.multi ? key.split(": ").slice(1).join(": ") : key;
      if (keyMetric === metric) bounds.set(key, [lo, hi]);
    });
  });
  const allBounds = [...metricGroups.keys()].map((metric) => {
    const firstKey = series.find(([key]) => (curve.multi ? key.split(": ").slice(1).join(": ") : key) === metric)?.[0];
    return bounds.get(firstKey);
  }).filter(Boolean);
  const singleMetric = metricGroups.size === 1;
  const globalLo = singleMetric ? Math.min(...allBounds.map(([lo]) => lo)) : 0;
  const globalHi = singleMetric ? Math.max(...allBounds.map(([, hi]) => hi)) : 1;
  const generationFor = (key, i) => {
    const trainer = curve.multi ? key.split(": ")[0] : null;
    const g = curve.series_axis?.[key]
      || curve.series_generation?.[key]
      || (curve.multi ? (curve.generation?.[trainer] || []) : (curve.generation || []));
    return Number.isFinite(Number(g[i])) ? Number(g[i]) : i;
  };
  const allGenerations = series.flatMap(([key, vals]) => vals.map((v, i) => v === null ? null : generationFor(key, i)).filter((v) => v !== null));
  const minGeneration = Math.min(...allGenerations);
  const maxGeneration = Math.max(...allGenerations);
  const generationSpan = maxGeneration - minGeneration || 1;
  const view = state.chartView;
  const visibleSpan = generationSpan / view.xScale;
  const center = (minGeneration + maxGeneration) / 2 + view.xOffset;
  const visibleMin = center - visibleSpan / 2;
  const visibleMax = center + visibleSpan / 2;
  const px = (key, i) => pad.left + ((generationFor(key, i) - visibleMin) / Math.max(1, visibleMax - visibleMin)) * width;
  const py = (key, v) => {
    const [rawLo, rawHi] = bounds.get(key);
    const span = rawHi - rawLo || 1;
    return Math.max(pad.top, Math.min(pad.top + height, pad.top + height - ((v - rawLo) / span) * height));
  };
  ctx.lineWidth = 1;
  [0, 0.5, 1].forEach((f) => {
    const y = pad.top + height * f;
    ctx.strokeStyle = cssVar("--line");
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(canvas.width - pad.right, y);
    ctx.stroke();
    if (singleMetric) {
      ctx.fillStyle = cssVar("--muted");
      ctx.fillText(fmt(globalHi - (globalHi - globalLo) * f, 1), 8, y + 4);
    }
  });
  const stageGenerations = Array.isArray(curve.stage)
    ? curve.stage.map((stage, i) => [stage, generationFor(series[0]?.[0] || "", i)])
    : [];
  stageGenerations.forEach(([stage, generation], i) => {
    if (i === 0 || stage === stageGenerations[i - 1][0]) return;
    ctx.strokeStyle = cssVar("--yellow");
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    const sx = pad.left + ((generation - minGeneration) / generationSpan) * width;
    ctx.moveTo(sx, pad.top);
    ctx.lineTo(sx, pad.top + height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--yellow");
    ctx.fillText(`stage ${stage}`, sx + 4, pad.top + 10);
  });
  if (Array.isArray(curve.run_boundaries_axis) && curve.run_boundaries_axis.length) {
    curve.run_boundaries_axis.forEach((generationValue) => {
      const generation = Number(generationValue);
      if (!Number.isFinite(generation)) return;
      const rx = pad.left + ((generation - visibleMin) / Math.max(1, visibleMax - visibleMin)) * width;
      if (rx < pad.left || rx > pad.left + width) return;
      ctx.strokeStyle = cssVar("--muted");
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(rx, pad.top);
      ctx.lineTo(rx, pad.top + height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = cssVar("--muted");
      ctx.fillText("run restart", rx + 4, pad.top + height - 8);
    });
  }
  series.forEach(([key, vals], k) => {
    // In fleet mode colour identifies the trainer; line width identifies the
    // first selected metric. This keeps six parallel trainers visually
    // separable instead of cycling colours across trainer/metric combinations.
    const trainerIndex = curve.multi
      ? Math.max(0, (curve.trainers || []).indexOf(key.split(": ")[0]))
      : 0;
    const colour = trainerColour(curve.multi ? key.split(": ")[0] : null, curve.trainers);
    const isEval = key.startsWith("eval_");
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.setLineDash(trainerDash(curve.multi ? key.split(": ")[0] : null, curve.trainers));
    if (isEval) {
      vals.forEach((v, i) => {
        if (v === null) return;
        ctx.beginPath();
        ctx.arc(px(key, i), py(key, v), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      ctx.lineWidth = curve.multi ? 2 : (k === 0 ? 2 : 1);
      ctx.beginPath();
      let started = false;
      vals.forEach((v, i) => {
        if (v === null) {
          started = false;
          return;
        }
        if (!started) ctx.moveTo(px(key, i), py(key, v));
        else ctx.lineTo(px(key, i), py(key, v));
        started = true;
      });
      ctx.stroke();
    }
    ctx.setLineDash([]);
  });
  const gens = Array.isArray(curve.axis_generation)
    ? curve.axis_generation
    : Array.isArray(curve.generation)
      ? curve.generation
      : Object.values(curve.generation || {}).flat();
  if (gens.length) {
    ctx.fillStyle = cssVar("--muted");
    ctx.textAlign = "right";
    ctx.fillText(`generation ${maxGeneration.toLocaleString()}`, canvas.width - pad.right, pad.top + 10);
    ctx.textAlign = "left";
  }

  // Keep the graph readable without requiring the user to guess what the
  // horizontal coordinate means. Labels are anchored inside the canvas, so
  // they can never be clipped by the page edge.
  ctx.fillStyle = cssVar("--muted");
  ctx.textAlign = "left";
  ctx.fillText(`generation ${minGeneration.toLocaleString()}`, pad.left, canvas.height - 8);
  ctx.textAlign = "right";
  ctx.fillText(`generation ${maxGeneration.toLocaleString()}`, canvas.width - pad.right, canvas.height - 8);
  ctx.textAlign = "left";

  series.forEach(([key]) => {
    const item = document.createElement("span");
    item.className = "chart-legend-item";
    const swatch = document.createElement("i");
    swatch.style.background = trainerColour(curve.multi ? key.split(": ")[0] : null, curve.trainers);
    const text = document.createElement("span");
    text.textContent = label(key);
    text.title = key;
    item.append(swatch, text);
    legend.append(item);
  });
  updateChartDataNote(curve);
  const scaleNote = singleMetric ? "" : "multiple metrics · each metric uses its own display scale";
  el("chart-overlay").textContent = clipped
    ? `${clipped.toLocaleString()} extreme historical point${clipped === 1 ? "" : "s"} clipped from display scale${scaleNote ? ` · ${scaleNote}` : ""}`
    : scaleNote;
}

function trainerStatus(trainer) {
  if (trainer.alive) return "LIVE";
  return trainer.age_s < 3600 ? "IDLE" : "STALE";
}

function trainerMetric(v, digits = 2) {
  return v === null || v === undefined ? "-" : fmt(v, digits);
}

function renderTrainerFleet(payload) {
  state.trainers = payload.trainers || [];
  if (!state.trainerId || !state.trainers.some((t) => t.id === state.trainerId)) {
    state.trainerId = payload.primary || state.trainers[0]?.id || null;
  }
  const available = new Set(state.trainers.map((t) => t.id));
  state.compareTrainerIds = state.compareTrainerIds.filter((id) => available.has(id));
  if (!state.compareTrainerIds.length && state.trainers.length) {
    // If the active log exposes independent ES islands, those are the actual
    // parallel trainers and are the most useful default comparison. Fall back
    // to the discovered log files when running an older log without island
    // telemetry.
    const islands = state.trainers.filter((t) => Number.isInteger(t.island));
    state.compareTrainerIds = islands.length > 1
      ? islands.map((t) => t.id)
      : state.trainers.map((t) => t.id);
  }
  const select = el("trainer-select");
  select.innerHTML = "";
  state.trainers.forEach((trainer) => {
    const option = document.createElement("option");
    option.value = trainer.id;
    option.textContent = `${trainer.name} · g${trainer.generation}`;
    option.selected = trainer.id === state.trainerId;
    select.append(option);
  });

  const quick = el("trainer-quick-actions");
  quick.querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      const mode = button.dataset.trainers;
      if (mode === "all") state.compareTrainerIds = state.trainers.map((t) => t.id);
      else if (mode === "best2") {
        state.compareTrainerIds = [...state.trainers]
          .sort((a, b) => Number(b.best ?? -Infinity) - Number(a.best ?? -Infinity))
          .slice(0, 2).map((t) => t.id);
      } else state.compareTrainerIds = [];
      renderTrainerFleet({ primary: state.trainerId, trainers: state.trainers });
      refreshCurve();
    };
  });
  el("trainer-fleet-note").textContent = `${state.trainers.length} logs · select any trainer to inspect its curve/status`;

  const fleet = el("trainer-fleet");
  fleet.innerHTML = "";
  state.trainers.forEach((trainer) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `trainer-card${trainer.id === state.trainerId ? " selected" : ""}`;
    button.addEventListener("click", () => selectTrainer(trainer.id));
    const top = document.createElement("div");
    top.className = "trainer-card-top";
    top.innerHTML = `<strong>${trainer.name}</strong><span class="trainer-state ${trainer.alive ? "live" : ""}">${trainerStatus(trainer)}</span>`;
    const stats = document.createElement("div");
    stats.className = "trainer-card-stats";
    stats.innerHTML = `<span>gen <b>${Number(trainer.generation).toLocaleString()}</b></span><span>best <b>${trainerMetric(trainer.best)}</b></span><span>lap <b>${trainerMetric(trainer.laps, 3)}</b></span><span>eval <b>${trainerMetric(trainer.eval)}</b></span>`;
    button.append(top, stats);
    fleet.append(button);
  });

  const compare = el("trainer-compare");
  compare.innerHTML = "";
  state.trainers.forEach((trainer) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.compareTrainerIds.includes(trainer.id);
    cb.addEventListener("change", () => {
      if (cb.checked) state.compareTrainerIds.push(trainer.id);
      else state.compareTrainerIds = state.compareTrainerIds.filter((id) => id !== trainer.id);
      refreshCurve();
    });
    lab.append(cb, trainer.name);
    compare.append(lab);
  });
}

async function refreshTrainers() {
  try {
    const payload = await (await fetch("/api/trainers", { cache: "no-store" })).json();
    const previous = state.trainerId;
    renderTrainerFleet(payload);
    return previous !== state.trainerId;
  } catch (err) {
    console.warn("trainer fleet refresh failed", err);
    return false;
  }
}

async function selectTrainer(id) {
  if (!id || id === state.trainerId) return;
  state.trainerId = id;
  // Selecting a trainer means inspect that trainer. Comparison is an explicit
  // separate action through the checkboxes/shortcuts; otherwise a previous
  // island selection can make the graph appear unrelated to the selected log.
  state.compareTrainerIds = [id];
  resetChartView();
  renderTrainerFleet({ primary: state.trainers[0]?.id, trainers: state.trainers });
  await Promise.all([refreshCurve(), refreshTraining()]);
}

async function refreshCurve() {
  try {
    const keys = state.curveKeys.length ? state.curveKeys : state.ui.curve_default_keys;
    if (!state.trainerId && !state.compareTrainerIds.length) return;
    el("curve-status").textContent = "loading…";
    const selected = state.compareTrainerIds.length ? state.compareTrainerIds : [state.trainerId];
    const query = selected.length > 1
      ? `trainers=${encodeURIComponent(selected.join(","))}`
      : `trainer=${encodeURIComponent(selected[0])}`;
    const points = chartPointBudget();
    const curve = await (await fetch(`/api/curve?${query}&points=${points}&keys=${encodeURIComponent(keys.join(","))}`, { cache: "no-store" })).json();
    state.curve = curve;
    initCurveControls(curve.keys || []);
    initChartInteractions();
    drawCurve(curve);
    const runs = !curve.multi && curve.runs > 1
      ? ` · ${curve.runs} runs in the log (${curve.run_sizes.join("+")} generations)`
      : "";
    const compared = curve.multi ? ` · ${curve.trainers.length} trainers compared` : "";
    el("curve-heading").textContent =
      `training · ${curve.generations.toLocaleString()} generations · ${curve.hours} h compute${compared}${runs}`;
    el("curve-status").textContent = curve.generations ? "" : "no numeric training data";
  } catch (err) {
    el("curve-status").textContent = "graph unavailable — retrying";
    console.warn("curve refresh failed", err);
  }
}

function redrawResponsiveCanvases() {
  if (state.curve) drawCurve(state.curve);
  if (state.training?.present) drawEndings(state.training.last);
  if (state.frame) {
    drawTrack(state.frame);
    drawFly(state.frame);
    drawEyes(state.frame);
    drawControls(state.frame);
  }
}

function watchCanvasResize() {
  const observer = new ResizeObserver(() => redrawResponsiveCanvases());
  ["curve-canvas", "ending-canvas", "eyes-canvas", "controls-canvas", "track-canvas", "raster-canvas"].forEach((id) => {
    const canvas = el(id);
    if (canvas) observer.observe(canvas);
  });
}

/* ---------------------------------------------------------------- training */

function kvRows(dl, entries) {
  dl.innerHTML = "";
  entries.forEach(([k, v]) => {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label(k);
    const dd = document.createElement("dd");
    dd.textContent = v;
    row.append(dt, dd);
    dl.append(row);
  });
}

function formatValue(v) {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : fmt(v, 3);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.map(formatValue).join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function drawEndings(record) {
  const canvas = el("ending-canvas");
  fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = cssVar("--canvas");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const hist = record && record.end_progress_hist;
  if (!hist || !hist.length) {
    ctx.fillStyle = cssVar("--muted");
    ctx.font = `11px ${cssVar("--mono")}`;
    ctx.fillText("no end-of-episode histogram in this log yet", 8, 16);
    return;
  }
  const max = Math.max(1, ...hist);
  const w = canvas.width / hist.length;
  hist.forEach((count, i) => {
    const h = (count / max) * (canvas.height - 18);
    ctx.fillStyle = cssVar("--bad");
    ctx.fillRect(i * w + 1, canvas.height - 14 - h, w - 2, h);
  });
  (record.start_fractions || []).forEach((f) => {
    ctx.fillStyle = cssVar("--accent");
    ctx.fillRect(f * canvas.width - 1, canvas.height - 12, 2, 10);
  });
  ctx.fillStyle = cssVar("--muted");
  ctx.font = `11px ${cssVar("--mono")}`;
  ctx.fillText("start of lap", 4, canvas.height - 2);
  ctx.textAlign = "right";
  ctx.fillText("end of lap", canvas.width - 4, canvas.height - 2);
  ctx.textAlign = "left";
}

function renderTraining(t) {
  state.training = t;
  el("training-status-heading").textContent = "trainer";
  el("ending-heading").textContent = "where the population ended (last generation, bars) and the start points (ticks)";
  el("episodes-heading").textContent = `viewer episodes · last ${(t.episodes || []).length} endings`;
  el("record-summary").textContent = t.present ? `full record of generation ${t.generation} (${Object.keys(t.last).length} fields)` : "no trainer log";
  if (!t.present) {
    kvRows(el("training-status"), [["status", "no log found"]]);
    return;
  }
  kvRows(el("training-status"), [
    ["status", t.alive ? "running" : "stopped"],
    ["generation", t.generation],
    ["run", `${t.run} (${t.generations_in_run} generations)`],
    ["log age", `${fmt(t.log_age_s, 0)} s`],
    ["generation time", `${fmt(t.mean_generation_s, 1)} s`],
    ["generations per hour", fmt(t.generations_per_hour, 1)],
    ["stage", t.stage],
    ["road half-width", `${fmt(t.last.road_halfwidth, 2)} m`],
    ["sigma", fmt(t.last.sigma, 3)],
    ["population", `${t.last.popsize} x ${t.last.starts_per_gen} starts`],
    ["throughput", `${formatValue(t.last.body_steps_per_s)} body-steps/s`],
    ["best eval fitness", fmt(t.best_eval, 2)],
    ["best lap fraction", fmt(t.best_laps, 3)],
    ["brain precision", t.last.precision],
  ]);
  drawEndings(t.last);
  const perStart = (t.last.fitness_per_start || []).map((f, i) => {
    const reasons = (t.last.end_reason_per_start || [])[i];
    const r = reasons ? Object.entries(reasons).filter(([, c]) => c > 0).map(([k, c]) => `${k} ${c}`).join(" ") : "";
    return [`start ${i} @ ${fmt((t.last.start_fractions || [])[i], 2)} lap`, `fitness ${fmt(f, 1)} · laps ${fmt((t.last.laps_per_start || [])[i], 3)} · ${r}`];
  });
  kvRows(el("per-start"), perStart);

  const table = el("episodes");
  const eps = [...(t.episodes || [])].reverse();
  const cols = eps.length ? Object.keys(eps[0]).filter((k) => k !== "time") : [];
  table.innerHTML = "";
  if (cols.length) {
    const head = document.createElement("tr");
    cols.forEach((c) => {
      const th = document.createElement("th");
      th.textContent = label(c);
      head.append(th);
    });
    table.append(head);
    eps.slice(0, 12).forEach((e) => {
      const tr = document.createElement("tr");
      cols.forEach((c) => {
        const td = document.createElement("td");
        td.textContent = formatValue(e[c]);
        if (c === "reason") td.className = e[c] === "alive" ? "pos" : "neg";
        tr.append(td);
      });
      table.append(tr);
    });
  }
  kvRows(el("last-record"), Object.entries(t.last).map(([k, v]) => [k, formatValue(v)]));
}

async function refreshTraining() {
  try {
    if (!state.trainerId) return;
    renderTraining(await (await fetch(`/api/training?trainer=${encodeURIComponent(state.trainerId)}`, { cache: "no-store" })).json());
  } catch (err) {
    console.warn("training refresh failed", err);
  }
}

/* ------------------------------------------------------------- parameters */

function renderParams() {
  const m = state.meta;
  el("params-heading").textContent = `learned interface · ${m.params} parameters over ${m.edges.toLocaleString()} fixed connections · ${m.checkpoint}`;
  const table = el("params");
  table.innerHTML = "";
  const blocks = m.param_blocks || {};
  const head = document.createElement("tr");
  ["block", "shape", "min", "mean", "max", "bounds", "at bounds", "values"].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    head.append(th);
  });
  table.append(head);
  Object.entries(blocks).forEach(([name, b]) => {
    const tr = document.createElement("tr");
    [name, b.shape.join("x"), fmt(b.min, 3), fmt(b.mean, 3), fmt(b.max, 3), b.bounds.join(" .. "), b.at_bounds, b.values ? b.values.map((v) => fmt(v, 2)).join(" ") : "-"].forEach((v, i) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (i === 6 && b.at_bounds > 0) td.className = "neg";
      tr.append(td);
    });
    table.append(tr);
  });
  el("config-summary").textContent = "agent, vehicle, LIF and viewer configuration (as served)";
  el("config-dump").textContent = JSON.stringify(
    { agent_cfg: m.agent_cfg, car_cfg: m.car_cfg, lif_cfg: m.lif_cfg, brain: m.brain, readout: m.readout, viewer: m.config },
    null,
    2,
  );
}

/* ------------------------------------------------------------------ 3D */

function initDrive() {
  const canvas = el("drive-canvas");
  let drive;
  try {
    drive = new DriveView(canvas, state.track.style);
  } catch (err) {
    console.warn("3D drive view unavailable", err);
    return;
  }
  drive.load(state.track);
  state.drive = drive;
  const toggle = el("camera-toggle");
  if (toggle) {
    toggle.onclick = () => {
      const next = drive.mode === "cockpit" ? "chase" : "cockpit";
      drive.setMode(next);
      toggle.textContent = next === "cockpit" ? "chase view" : "cockpit view";
    };
  }
  const loop = () => {
    drive.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

async function initBrain() {
  const canvas = el("brain-canvas");
  el("brain-label").textContent = `fly brain · ${state.meta.neurons.toLocaleString()} neurons at their soma positions`;
  if (!state.meta.anatomy) {
    el("brain-meta").textContent = "no soma positions on the server (run src/build_positions.py)";
    return;
  }
  const order = state.meta.role_names;
  const colours = new Float32Array(8 * 3);
  order.forEach((role, i) => {
    if (i < 8) colours.set(hexToRgb(roleColour(role)), i * 3);
  });
  let brain;
  try {
    brain = new BrainView(canvas, colours, state.cfg.brain_view);
  } catch (err) {
    el("brain-meta").textContent = `3D unavailable: ${err.message}`;
    return;
  }
  const [posBuf, roleBuf] = await Promise.all([
    fetch("/api/positions.bin").then((r) => r.arrayBuffer()),
    fetch("/api/roles.bin").then((r) => r.arrayBuffer()),
  ]);
  const n = state.meta.neurons;
  brain.load(new Float32Array(posBuf), new Uint8Array(roleBuf, 0, n), new Uint8Array(roleBuf, n, n));
  state.brain = brain;
  el("brain-meta").textContent =
    `${n.toLocaleString()} neurons · ${state.meta.measured.toLocaleString()} measured somata · drag to rotate, scroll to zoom`;
  const loop = () => {
    brain.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function decodeMask(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/* ------------------------------------------------------------------ stream */

function onFrame(frame) {
  state.frame = frame;
  if (frame.step <= 1) state.trail = [];
  state.trail.push(frame.pos);
  if (state.trail.length > state.ui.trail_points) state.trail.shift();

  const stageNote = frame.stage >= 0 ? ` · stage ${frame.stage}` : "";
  el("stage-generation").textContent = `generation ${frame.generation.toLocaleString()}${stageNote}`;
  if (frame.generation !== state.metaGeneration) refreshMeta(frame);

  const pedal = frame.pedal ?? frame.throttle;
  const ui = state.ui;
  el("pill-speed").textContent = `${fmt(frame.speed * ui.kmh_per_mps, 0)} km/h · ${fmt(frame.lat_g ?? 0, 1)} g`;
  const pill = el("pill-state");
  if (frame.done_reason > 0) {
    state.ended = { reason: state.meta.done_names[String(frame.done_reason)] || "ended", until: performance.now() + ui.end_banner_ms };
  }
  const showEnd = state.ended && performance.now() < state.ended.until;
  const reset = showEnd || frame.step <= 1;
  pill.textContent = showEnd
    ? state.ended.reason.toUpperCase()
    : reset ? "RESET" : pedal < -ui.pedal_deadband ? "BRAKING" : pedal > ui.pedal_deadband ? "THROTTLE" : "COASTING";
  pill.classList.toggle("crashed", reset);
  pill.classList.toggle("braking", !reset && pedal < -ui.pedal_deadband);
  el("pill-foot").textContent =
    `episode ${frame.episode} · step ${frame.step} · ${state.meta.mode} · return ${fmt(frame.episode_return, 1)} · ${fmt(frame.uptime, 1)} s up`;

  drawTrack(frame);
  drawRaster(frame);
  drawFleet(frame);
  updateLists(frame);
  updateTelemetry(frame);
  drawFly(frame);
  drawEyes(frame);
  drawControls(frame);
  updateFlyTelemetry(frame);
  if (state.drive) {
    state.drive.maxInputHz = state.meta.agent_cfg?.max_input_hz || 300;
    state.drive.update(frame);
  }
  if (state.brain && frame.mask) state.brain.applySpikes(decodeMask(frame.mask));

  const m = state.meta;
  const road = m.road_halfwidth ? ` · road ${fmt(m.road_halfwidth * 2, 1)} m` : "";
  el("footer-meta").textContent =
    `${m.neurons.toLocaleString()} neurons · ${m.edges.toLocaleString()} connections · ${m.device} ${m.brain ? `${m.brain.precision}${m.brain.metal ? " metal" : ""}` : ""} · ` +
    `${m.dt_ms} ms x ${m.substeps} substeps · ${frame.checkpoint || m.checkpoint}${road} · start ${m.track} · ${m.track_name}`;
}

/* The trainer rewrites the checkpoint every generation; the viewer reloads it
 * between episodes and re-ranks the output neurons, so the lists, parameter
 * table and checkpoint note are refreshed whenever the generation changes. */
let metaRefreshing = false;
async function refreshMeta(frame) {
  if (metaRefreshing) return;
  metaRefreshing = true;
  state.metaGeneration = frame.generation;
  try {
    state.meta = await (await fetch("/api/meta")).json();
    state.cfg = state.meta.config;
    state.ui = state.cfg.ui;
    initLists();
    renderParams();
  } catch (err) {
    console.warn("meta refresh failed", err);
  } finally {
    metaRefreshing = false;
  }
}

function connect() {
  const source = new EventSource("/api/stream");
  const link = () => el("stat-link");
  source.onopen = () => {
    link().textContent = "live";
    link().className = "good";
  };
  // The simulation streams every 16 ms control step. EventSource has no
  // backpressure: when a frame takes longer to draw than the next takes to
  // arrive, the backlog grows without bound and the page falls seconds behind
  // (the "laggy map"). Keep only the newest frame and draw it on the next
  // animation frame; each frame still carries the full state.
  let pending = null;
  let scheduled = false;
  source.onmessage = (event) => {
    pending = event.data;
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      const data = pending;
      pending = null;
      if (data) onFrame(JSON.parse(data));
    });
  };
  source.onerror = () => {
    link().textContent = "reconnecting";
    link().className = "bad";
  };
}

async function boot() {
  const [track, meta] = await Promise.all([
    fetch("/api/track").then((r) => r.json()),
    fetch("/api/meta").then((r) => r.json()),
  ]);
  state.track = track;
  state.meta = meta;
  state.cfg = meta.config;
  state.ui = state.cfg.ui;
  state.metaGeneration = null;
  document.title = `${track.name} · ${track.car.name} · connectome driving`;
  el("stage-title").textContent = `malecns · ${meta.neurons.toLocaleString()}-neuron connectome driving · ${track.name}, ${track.length_m.toLocaleString()} m · ${track.car.name} dynamics`;
  initRaster();
  initLists();
  initTelemetry();
  initFly();
  renderParams();
  drawTrack(null);
  await refreshTrainers();
  await refreshCurve();
  await refreshTraining();
  watchCanvasResize();
  setInterval(refreshCurve, state.ui.curve_refresh_ms);
  setInterval(refreshTraining, state.ui.training_refresh_ms);
  setInterval(refreshTrainers, state.ui.training_refresh_ms);
  el("trainer-select").addEventListener("change", (event) => selectTrainer(event.target.value));
  el("trainer-refresh").addEventListener("click", async () => {
    await refreshTrainers();
    await Promise.all([refreshCurve(), refreshTraining()]);
  });
  initDrive();
  initBrain();
  connect();
}

window.__malecns = state; // read-only diagnostics handle for the browser console
boot();
