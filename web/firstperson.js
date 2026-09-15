"use strict";

/* First-person view: what the fly is told and what it feels, every frame.
 *
 * Walls: one column per lidar ray (ray 0 on the left), height from the true
 * distance, brightness from the proximity the eye group is actually driven
 * with (road-relative, agent.py) and a white flash where an edge is looming.
 * The horizon rolls with lateral load and pitches with the pedal; the picture
 * edges glow red as the body closes on a barrier. The strip underneath shows
 * the other senses and the motor intent as bars. Every number and label comes
 * from the frame and the server-side schema/config. */

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function rgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function bar(ctx, x, y, w, h, level, colour, label, valueText, font, textColour, bipolar = false) {
  ctx.fillStyle = "rgba(255,255,255,0.06)";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = colour;
  if (bipolar) {
    const half = w / 2;
    const len = clamp(level, -1, 1) * half;
    ctx.fillRect(len >= 0 ? x + half : x + half + len, y, Math.abs(len), h);
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillRect(x + half - 0.5, y - 2, 1, h + 4);
  } else {
    ctx.fillRect(x, y, clamp(level, 0, 1) * w, h);
  }
  ctx.fillStyle = textColour;
  ctx.font = font;
  ctx.textAlign = "left";
  ctx.fillText(label, x, y - 4);
  ctx.textAlign = "right";
  ctx.fillText(valueText, x + w, y - 4);
}

/**
 * @param canvas  target canvas (already sized)
 * @param frame   live frame from the server
 * @param ctx     { track, meta, ui, cssVar, roleColour, fmt } page state and helpers
 */
export function drawFirstPerson(canvas, frame, page) {
  const g = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const { track, meta, cssVar, roleColour, fmt } = page;
  const n = frame.lidar?.length || 0;
  if (!n) return;
  const maxRange = track.max_range;
  const car = track.car_cfg;
  const mono = cssVar("--mono");
  const small = `${Math.max(10, H * 0.032)}px ${mono}`;
  const big = `700 ${Math.max(16, H * 0.11)}px ${mono}`;
  const eyeColour = roleColour("visual_projection");
  const dnColour = roleColour("descending");
  const warn = cssVar("--warn");
  const bad = cssVar("--bad");
  const accent = cssVar("--accent");
  const text = cssVar("--text");
  const muted = cssVar("--muted");

  // --- attitude: roll from lateral load (sign from the steering angle), pitch from the pedal
  const viewH = H * 0.66;
  const horizon = viewH * 0.55;
  const roll = -Math.sign(frame.steer_actual || 0) * clamp(frame.lat_g / (car.grip_max_g || 4.5), 0, 1) * 0.22;
  const pitch = -clamp(frame.pedal, -1, 1) * viewH * 0.06;

  g.save();
  g.beginPath();
  g.rect(0, 0, W, viewH);
  g.clip();
  g.translate(W / 2, horizon + pitch);
  g.rotate(roll);
  // sky and ground, oversized so the roll never shows a corner
  const sky = g.createLinearGradient(0, -viewH, 0, 0);
  sky.addColorStop(0, "#0b1020");
  sky.addColorStop(1, "#2a3550");
  g.fillStyle = sky;
  g.fillRect(-W, -viewH * 1.5, 2 * W, viewH * 1.5);
  const ground = g.createLinearGradient(0, 0, 0, viewH);
  ground.addColorStop(0, "#3a3a3a");
  ground.addColorStop(1, "#151515");
  g.fillStyle = ground;
  g.fillRect(-W, 0, 2 * W, viewH * 1.5);
  // horizon line
  g.strokeStyle = "rgba(255,255,255,0.25)";
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(-W, 0);
  g.lineTo(W, 0);
  g.stroke();

  // --- walls: a column per ray, ray 0 (left) drawn on the left
  const colW = W / n;
  const wallScale = viewH * 0.9 * 4.0; // a wall 4 m away fills the view
  for (let i = 0; i < n; i++) {
    const metres = Math.max(0.5, frame.lidar[i] * maxRange);
    const h = Math.min(viewH * 0.95, wallScale / metres);
    const x = -W / 2 + i * colW;
    const prox = frame.proximity ? frame.proximity[i] : 1 - frame.lidar[i];
    const loom = frame.loom ? frame.loom[i] : 0;
    g.fillStyle = rgba(eyeColour, 0.15 + 0.8 * clamp(prox, 0, 1));
    g.fillRect(x + 1, -h / 2, colW - 2, h);
    if (loom > 0) {
      g.fillStyle = `rgba(255,255,255,${clamp(loom * 1.5, 0, 0.9)})`;
      g.fillRect(x + 1, -h / 2, colW - 2, h);
    }
    g.strokeStyle = "rgba(0,0,0,0.5)";
    g.strokeRect(x + 1, -h / 2, colW - 2, h);
    g.fillStyle = text;
    g.font = small;
    g.textAlign = "center";
    g.fillText(`${fmt(metres, metres < 10 ? 1 : 0)} m`, x + colW / 2, Math.min(viewH * 0.42, h / 2 + H * 0.045));
    g.fillStyle = muted;
    g.fillText(`${fmt(frame.sensory_hz?.[i] ?? 0, 0)} Hz`, x + colW / 2, -Math.min(viewH * 0.45, h / 2 + H * 0.02));
  }
  g.restore();

  // --- barrier proximity: red glow on the edges as body clearance shrinks
  const margin = track.car_cfg.wall_margin || 3.0;
  const danger = clamp(1 - (frame.clearance ?? margin) / margin, 0, 1);
  if (danger > 0) {
    const glow = g.createLinearGradient(0, 0, W, 0);
    glow.addColorStop(0, rgba(bad, 0.7 * danger));
    glow.addColorStop(0.25, "rgba(0,0,0,0)");
    glow.addColorStop(0.75, "rgba(0,0,0,0)");
    glow.addColorStop(1, rgba(bad, 0.7 * danger));
    g.fillStyle = glow;
    g.fillRect(0, 0, W, viewH);
  }
  // crash flash
  if (frame.done_reason === 1) {
    g.fillStyle = rgba(bad, 0.45);
    g.fillRect(0, 0, W, viewH);
  }

  // --- centre readouts: speed and lap
  g.fillStyle = text;
  g.font = big;
  g.textAlign = "center";
  g.fillText(`${fmt(frame.speed * (page.ui.kmh_per_mps || 3.6), 0)} km/h`, W / 2, viewH * 0.93);
  g.font = small;
  g.fillStyle = muted;
  g.textAlign = "left";
  g.fillText(`L`, 6, viewH * 0.5);
  g.textAlign = "right";
  g.fillText(`R`, W - 6, viewH * 0.5);

  // --- feelings strip
  const y0 = viewH + H * 0.075;
  const rowH = H * 0.055;
  const gap = H * 0.115;
  const colX = [W * 0.03, W * 0.36, W * 0.69];
  const colWid = W * 0.28;
  const maxHz = meta.agent_cfg?.max_input_hz || 300;
  const rows = [
    // column 0: senses
    [colX[0], y0, (frame.speed_hz || 0) / maxHz, eyeColour, "speed sense (ascending neurons)", `${fmt(frame.speed_hz, 0)} Hz`, false],
    [colX[0], y0 + gap, clamp(frame.lat_g / (car.grip_max_g || 4.5), 0, 1), warn, "lateral load", `${fmt(frame.lat_g, 2)} g`, false],
    [colX[0], y0 + 2 * gap, danger, bad, "barrier closeness (body clearance)", `${fmt(frame.clearance, 2)} m`, false],
    // column 1: brain -> intent
    [colX[1], y0, (frame.dn_hz || 0) / 20, dnColour, "output population (descending + motor)", `${fmt(frame.dn_hz, 2)} Hz`, false],
    [colX[1], y0 + gap, -Math.tanh(frame.motor_steer || 0), accent, "steer intent (pre-activation, R ... L)", fmt(frame.motor_steer, 2), true],
    [colX[1], y0 + 2 * gap, Math.tanh(frame.motor_pedal || 0), accent, "pedal intent (brake ... gas)", fmt(frame.motor_pedal, 2), true],
    // column 2: body and reward
    [colX[2], y0, -(frame.steer_actual || 0), text, "wheels (R ... L)", `${fmt(frame.steer_actual, 2)} of lock`, true],
    [colX[2], y0 + gap, frame.pedal || 0, frame.pedal < 0 ? bad : accent, "pedal (brake ... gas)", fmt(frame.pedal, 2), true],
    [colX[2], y0 + 2 * gap, clamp((frame.reward || 0) / 0.5, -1, 1), frame.reward < 0 ? bad : accent, "reward this step", fmt(frame.reward, 3), true],
  ];
  rows.forEach(([x, y, level, colour, labelText, valueText, bipolar]) => bar(g, x, y, colWid, rowH, level, colour, labelText, valueText, small, muted, bipolar));
}
