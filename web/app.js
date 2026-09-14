"use strict";

import { BrainView } from "/brain.js";
import { DriveView } from "/drive.js";

const ROLE_COLOR = {
  photoreceptor: "#4aa3e8",
  visual_projection: "#3ddc97",
  mechanosensory: "#b58ce0",
  ascending: "#e8c44a",
  descending: "#e8894a",
  motor: "#e05c6e",
};

const state = {
  track: null,
  meta: null,
  trail: [],
  bandOf: [],
  sampleSize: 0,
  brain: null,
  frame: null,
  metaGeneration: null,
  ended: null,
};

const el = (id) => document.getElementById(id);
const fmt = (v, d = 2) => Number(v).toFixed(d);

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/* ---------------------------------------------------------------- track view */

function trackTransform(canvas, extent) {
  const size = Math.min(canvas.width, canvas.height);
  const scale = (size * 0.94) / (2 * extent);
  return {
    x: (wx) => canvas.width / 2 + wx * scale,
    y: (wy) => canvas.height / 2 - wy * scale,
    scale,
  };
}

function drawTrack(frame) {
  const canvas = el("track-canvas");
  const ctx = canvas.getContext("2d");
  const { centerline, halfwidth, extent, n_rays, fov_deg, max_range } = state.track;
  const t = trackTransform(canvas, extent);

  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.beginPath();
  centerline.forEach(([x, y], i) => {
    const px = t.x(x);
    const py = t.y(y);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = halfwidth * 2 * t.scale;
  ctx.strokeStyle = "#161d25";
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#2b3542";
  ctx.setLineDash([6, 12]);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!frame) return;

  if (state.trail.length > 1) {
    ctx.beginPath();
    state.trail.forEach(([x, y], i) => {
      const px = t.x(x);
      const py = t.y(y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "rgba(61, 220, 151, 0.6)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  const [cx, cy] = frame.pos;
  const px = t.x(cx);
  const py = t.y(cy);
  const fov = (fov_deg * Math.PI) / 180;

  frame.lidar.forEach((norm, i) => {
    const offset = n_rays === 1 ? 0 : fov / 2 - (fov * i) / (n_rays - 1); // ray 0 looks left
    const angle = frame.heading + offset;
    const reach = norm * max_range;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(t.x(cx + Math.cos(angle) * reach), t.y(cy + Math.sin(angle) * reach));
    ctx.strokeStyle = `rgba(232, 137, 74, ${0.2 + (1 - norm) * 0.65})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-frame.heading);
  ctx.fillStyle = "#e9eef4";
  ctx.beginPath();
  ctx.moveTo(11, 0);
  ctx.lineTo(-7, 6.5);
  ctx.lineTo(-7, -6.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/* --------------------------------------------------------------- spike raster */

const raster = { ctx: null, buffer: null };

function initRaster() {
  const canvas = el("raster-canvas");
  raster.ctx = canvas.getContext("2d");
  raster.ctx.fillStyle = "#05070a";
  raster.ctx.fillRect(0, 0, canvas.width, canvas.height);
  raster.buffer = document.createElement("canvas");
  raster.buffer.width = canvas.width;
  raster.buffer.height = canvas.height;

  const legend = el("raster-legend");
  legend.innerHTML = "";
  state.meta.bands.forEach((band) => {
    const li = document.createElement("li");
    const swatch = document.createElement("i");
    swatch.style.background = ROLE_COLOR[band.role] || "#8494a4";
    li.append(swatch, band.role.replace(/_/g, " "));
    legend.append(li);
  });

  state.sampleSize = state.meta.bands.reduce((sum, b) => sum + b.count, 0);
  state.bandOf = new Array(state.sampleSize);
  state.meta.bands.forEach((band) => {
    for (let i = 0; i < band.count; i += 1) state.bandOf[band.start + i] = band.role;
  });
  el("raster-meta").textContent = `${state.sampleSize} sampled`;
}

function drawRaster(frame) {
  const ctx = raster.ctx;
  const canvas = ctx.canvas;
  const step = 2;

  raster.buffer.getContext("2d").drawImage(canvas, 0, 0);
  ctx.drawImage(raster.buffer, -step, 0);
  ctx.fillStyle = "#05070a";
  ctx.fillRect(canvas.width - step, 0, step, canvas.height);

  const rowHeight = canvas.height / state.sampleSize;
  frame.fired.forEach((i) => {
    ctx.fillStyle = ROLE_COLOR[state.bandOf[i]] || "#8494a4";
    ctx.fillRect(canvas.width - step, i * rowHeight, step, Math.max(1, rowHeight));
  });
}

/* ------------------------------------------------------------------ bar lists */

function buildList(container, items) {
  container.innerHTML = "";
  return items.map((item) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = item.label;
    label.title = item.title || item.label;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.background = item.color;
    bar.append(fill);
    const out = document.createElement("output");
    out.textContent = "0";
    li.append(label, bar, out);
    container.append(li);
    return { fill, out };
  });
}

let rateRows = [];
let dnRows = [];

function initLists() {
  state.rateRoles = Object.keys(state.meta.roles);
  rateRows = buildList(
    el("rate-list"),
    state.rateRoles.map((role) => ({
      label: role.replace(/_/g, " "),
      title: `${role} — ${state.meta.roles[role].toLocaleString()} neurons`,
      color: ROLE_COLOR[role] || "#8494a4",
    })),
  );

  dnRows = buildList(
    el("dn-list"),
    state.meta.top_dn.map((dn) => ({
      label: dn.type === "None" ? String(dn.body) : dn.type,
      title: `body ${dn.body} — readout weight ${dn.weight}`,
      color: dn.weight >= 0 ? ROLE_COLOR.descending : "#5c9ee0",
    })),
  );
}

function updateLists(frame) {
  const maxRate = Math.max(12, ...Object.values(frame.rates));
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

/* ----------------------------------------------------------------- curve view */

function drawCurve(curve) {
  const canvas = el("curve-canvas");
  const ctx = canvas.getContext("2d");
  const pad = { left: 46, right: 12, top: 12, bottom: 20 };
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!curve.fitness.length) return;

  const width = canvas.width - pad.left - pad.right;
  const height = canvas.height - pad.top - pad.bottom;
  const lo = Math.min(...curve.fitness);
  const hi = Math.max(...curve.best);
  const span = hi - lo || 1;
  const px = (i) => pad.left + (i / (curve.fitness.length - 1 || 1)) * width;
  const py = (v) => pad.top + height - ((v - lo) / span) * height;

  ctx.lineWidth = 1;
  [0, 0.5, 1].forEach((f) => {
    const y = pad.top + height * f;
    ctx.strokeStyle = "#1b222b";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(canvas.width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "#55636f";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(fmt(hi - span * f, 1), 8, y + 4);
  });

  const line = (values, color, lw) => {
    ctx.beginPath();
    values.forEach((v, i) => (i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v))));
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.stroke();
  };
  // curriculum stage boundaries
  (curve.stage || []).forEach((stage, i) => {
    if (i === 0 || stage === curve.stage[i - 1]) return;
    ctx.strokeStyle = "rgba(232, 196, 74, 0.5)";
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(px(i), pad.top);
    ctx.lineTo(px(i), pad.top + height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#e8c44a";
    ctx.fillText(`stage ${stage}`, px(i) + 4, pad.top + 10);
  });
  line(curve.best, "rgba(232, 137, 74, 0.4)", 1);
  line(curve.fitness, "#3ddc97", 2);
  // deterministic evaluations of the mean, at their generation
  if (curve.eval && curve.eval.length && curve.eval_gen) {
    const gx = (gen) => pad.left + ((gen - 1) / Math.max(1, curve.generations - 1)) * width;
    ctx.fillStyle = "#e8894a";
    curve.eval.forEach((v, i) => {
      const y = Math.max(pad.top, Math.min(pad.top + height, py(v)));
      ctx.beginPath();
      ctx.arc(gx(curve.eval_gen[i]), y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  const evalNote = curve.eval && curve.eval.length ? ` · eval ${fmt(curve.eval[curve.eval.length - 1], 1)} (${curve.eval.length} runs)` : "";
  el("curve-meta").textContent =
    `${curve.generations.toLocaleString()} generations · ${curve.hours} h compute · ` +
    `best lap ${fmt(Math.max(...curve.laps), 3)}${evalNote}`;
}

async function refreshCurve() {
  try {
    drawCurve(await (await fetch("/api/curve")).json());
  } catch (err) {
    console.warn("curve refresh failed", err);
  }
}

/* ---------------------------------------------------------------- 3D drive */

function initDrive() {
  const canvas = el("drive-canvas");
  let drive;
  try {
    drive = new DriveView(canvas);
  } catch (err) {
    console.warn("3D drive view unavailable", err);
    return;
  }
  drive.load(state.track);
  state.drive = drive;
  const loop = () => {
    drive.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

/* ------------------------------------------------------------------ 3D brain */

async function initBrain() {
  const canvas = el("brain-canvas");
  if (!state.meta.anatomy) {
    el("brain-meta").textContent = "no positions.npy — run src/build_positions.py";
    return;
  }
  const order = state.meta.role_names;
  const colors = new Float32Array(8 * 3);
  order.forEach((role, i) => {
    if (i < 8) colors.set(hexToRgb(ROLE_COLOR[role] || "#8494a4"), i * 3);
  });

  let brain;
  try {
    brain = new BrainView(canvas, colors);
  } catch (err) {
    el("brain-meta").textContent = `3D unavailable: ${err.message}`;
    return;
  }

  const [posBuf, roleBuf] = await Promise.all([
    fetch("/api/positions.bin").then((r) => r.arrayBuffer()),
    fetch("/api/roles.bin").then((r) => r.arrayBuffer()),
  ]);
  const n = state.meta.neurons;
  brain.load(
    new Float32Array(posBuf),
    new Uint8Array(roleBuf, 0, n),
    new Uint8Array(roleBuf, n, n),
  );
  state.brain = brain;
  el("brain-meta").textContent =
    `${n.toLocaleString()} neurons · ${state.meta.measured.toLocaleString()} measured somata · ` +
    `drag to rotate, scroll to zoom`;

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

/* --------------------------------------------------------------------- stream */

function onFrame(frame) {
  state.frame = frame;
  if (frame.step <= 1) state.trail = [];
  state.trail.push(frame.pos);
  if (state.trail.length > 1200) state.trail.shift();

  const stageNote = frame.stage >= 0 ? ` · stage ${frame.stage}` : "";
  el("stage-generation").textContent = `generation ${frame.generation.toLocaleString()}${stageNote}`;
  if (frame.generation !== state.metaGeneration) refreshMeta(frame);
  const pedal = frame.pedal ?? frame.throttle;
  el("pill-speed").textContent = `${fmt(frame.speed * 3.6, 0)} km/h · ${fmt(frame.lat_g ?? 0, 1)} g`;
  const pill = el("pill-state");
  // the last frame of an episode carries why it ended; hold that on screen
  // through the reset so a crash is not just a car teleporting back
  if (frame.done_reason > 0) state.ended = { reason: END_REASON[frame.done_reason] || "ENDED", until: performance.now() + 1400 };
  const showEnd = state.ended && performance.now() < state.ended.until;
  const crashed = showEnd || frame.step <= 1;
  pill.textContent = showEnd ? state.ended.reason : crashed ? "RESET" : pedal < -0.05 ? "BRAKING" : pedal > 0.05 ? "THROTTLE" : "COASTING";
  pill.classList.toggle("crashed", crashed);
  pill.classList.toggle("braking", !crashed && pedal < -0.05);
  el("pill-foot").textContent =
    `episode ${frame.episode} · step ${frame.step} · ${state.meta.mode === "replay" ? "replay" : "live"} · ${fmt(frame.uptime, 1)} s elapsed`;

  el("stat-laps").textContent = fmt(frame.laps, 3);
  el("stat-spiking").textContent = frame.spiking.toLocaleString();
  el("stat-fps").textContent = `${fmt(frame.fps, 1)}/s`;
  el("stat-realtime").textContent = `${fmt(frame.headroom, 2)}x`;

  const pedalBar = el("bar-throttle");
  pedalBar.style.width = `${Math.abs(pedal) * 50}%`;
  pedalBar.style.left = pedal >= 0 ? "50%" : `${50 - Math.abs(pedal) * 50}%`;
  pedalBar.style.background = pedal >= 0 ? "" : "#e05c6e";
  // steer > 0 is a LEFT turn (counter-clockwise, as in car_env.py), so the
  // bar grows leftwards from centre for positive steer and rightwards for negative
  const steer = el("bar-steer");
  steer.style.width = `${Math.abs(frame.steer) * 50}%`;
  steer.style.left = frame.steer > 0 ? `${50 - frame.steer * 50}%` : "50%";
  el("val-steer").textContent = `${frame.steer > 0.02 ? "L " : frame.steer < -0.02 ? "R " : ""}${fmt(Math.abs(frame.steer))}`;
  el("val-throttle").textContent = fmt(pedal);

  drawTrack(frame);
  drawRaster(frame);
  updateLists(frame);
  if (state.drive) state.drive.update(frame);
  if (state.brain && frame.mask) state.brain.applySpikes(decodeMask(frame.mask));

  const road = state.meta.road_halfwidth ? ` · road ${fmt(state.meta.road_halfwidth * 2, 1)} m` : "";
  el("footer-meta").textContent =
    `${state.meta.neurons.toLocaleString()} neurons · ${state.meta.edges.toLocaleString()} synapses · ` +
    `${state.meta.device} · ${state.meta.dt_ms} ms x ${state.meta.substeps} substeps · ` +
    `${frame.checkpoint || state.meta.checkpoint}${road} · start ${state.meta.track} · reward ${fmt(frame.reward)}`;
}

const END_REASON = { 1: "CRASH", 2: "REVERSE", 3: "STUCK" };

/* The trainer rewrites the checkpoint every generation; the viewer reloads it
 * between episodes and re-ranks the descending neurons, so the DN list and
 * checkpoint note are refreshed whenever the generation changes. */
let metaRefreshing = false;
async function refreshMeta(frame) {
  if (metaRefreshing) return;
  metaRefreshing = true;
  state.metaGeneration = frame.generation;
  try {
    state.meta = await (await fetch("/api/meta")).json();
    initLists();
  } catch (err) {
    console.warn("meta refresh failed", err);
  } finally {
    metaRefreshing = false;
  }
}

function connect() {
  const source = new EventSource("/api/stream");
  const link = el("stat-link");
  source.onopen = () => {
    link.textContent = "live";
    link.className = "good";
  };
  source.onmessage = (event) => onFrame(JSON.parse(event.data));
  source.onerror = () => {
    link.textContent = "reconnecting";
    link.className = "bad";
  };
}

async function boot() {
  const [track, meta] = await Promise.all([
    fetch("/api/track").then((r) => r.json()),
    fetch("/api/meta").then((r) => r.json()),
  ]);
  state.track = track;
  state.meta = meta;
  state.metaGeneration = null;
  initRaster();
  initLists();
  drawTrack(null);
  await refreshCurve();
  setInterval(refreshCurve, 15000);
  initDrive();
  initBrain();
  connect();
}

boot();
