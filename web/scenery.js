"use strict";

/* Real Monaco scenery for the drive view: OpenStreetMap building footprints
 * extruded to their mapped heights with procedural facades, the Boulevard
 * Louis II tunnel as an arched enclosure swept along the circuit where OSM
 * maps the road underground, and triple-stacked Armco guardrail with posts
 * along both edges of the road. Everything is placed with the same projection
 * as the centerline, so buildings stand where they stand.
 *
 * Data (c) OpenStreetMap contributors, ODbL 1.0.
 */

import * as THREE from "three";
import { mergeGeometries } from "/vendor/three/utils/BufferGeometryUtils.js";

const STOREY_M = 3.2;
const BAY_M = 3.1;
const RAIL_HEIGHTS = [0.32, 0.7, 1.08]; // three W-beams, Monaco style
const POST_SPACING_M = 2.0;
const PALETTE = [
  ["#e6d7bf", "#c9b799"], // Monaco cream
  ["#e9c9a4", "#c7a37d"], // ochre
  ["#f0e3d4", "#cbbca9"], // pale stone
  ["#d9c5b0", "#b39a82"], // sand
  ["#e3b8a4", "#c1907a"], // terracotta pink
  ["#cfd5d9", "#a3adb5"], // modern grey
  ["#f2e9dc", "#d3c6b2"], // white-cream
  ["#dcc7a1", "#b8a07a"], // yellow ochre
];

/* Sim frame -> world frame: sim x is world x, sim y is world -z, up is +y. */
const toWorld = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

function hash(i) {
  let h = (i * 2654435761) >>> 0;
  h ^= h >>> 13;
  h = (h * 0x5bd1e995) >>> 0;
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/* --------------------------------------------------------------- textures */

function facadeTexture(base, trim, seed) {
  const w = 256;
  const h = 256; // one storey tall, one bay wide, repeated
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, w, h);
  // stone joints and weathering
  for (let i = 0; i < 260; i += 1) {
    const x = hash(seed * 7919 + i) * w;
    const y = hash(seed * 104729 + i) * h;
    ctx.fillStyle = `rgba(0,0,0,${0.02 + hash(i + seed) * 0.05})`;
    ctx.fillRect(x, y, 6 + hash(i * 3 + seed) * 20, 1 + hash(i * 5 + seed) * 2);
  }
  // floor slab line
  ctx.fillStyle = trim;
  ctx.fillRect(0, h - 10, w, 10);
  // window with frame, sill, shutters and a dark glass gradient
  const wx = w * 0.3;
  const wy = h * 0.22;
  const ww = w * 0.4;
  const wh = h * 0.5;
  ctx.fillStyle = trim;
  ctx.fillRect(wx - 10, wy - 10, ww + 20, wh + 20);
  const glass = ctx.createLinearGradient(0, wy, 0, wy + wh);
  glass.addColorStop(0, "#6f8598");
  glass.addColorStop(0.5, "#2c3944");
  glass.addColorStop(1, "#141b21");
  ctx.fillStyle = glass;
  ctx.fillRect(wx, wy, ww, wh);
  ctx.fillStyle = "rgba(255,255,255,0.18)";
  ctx.fillRect(wx + 4, wy + 4, ww * 0.35, wh * 0.4);
  ctx.fillStyle = "#e9eef4";
  ctx.fillRect(wx + ww / 2 - 2, wy, 4, wh); // mullion
  ctx.fillRect(wx, wy + wh / 2 - 2, ww, 4); // transom
  // shutters on some facades
  if (hash(seed + 11) > 0.45) {
    ctx.fillStyle = hash(seed + 13) > 0.5 ? "#4f6b52" : "#5d6d7e";
    ctx.fillRect(wx - 10 - ww * 0.22, wy - 4, ww * 0.22, wh + 8);
    ctx.fillRect(wx + ww + 10, wy - 4, ww * 0.22, wh + 8);
    for (let k = 0; k < 6; k += 1) {
      ctx.fillStyle = "rgba(0,0,0,0.25)";
      ctx.fillRect(wx - 10 - ww * 0.22, wy + (k / 6) * wh, ww * 0.22, 2);
      ctx.fillRect(wx + ww + 10, wy + (k / 6) * wh, ww * 0.22, 2);
    }
  }
  // balcony railing on the storey line for some facades
  if (hash(seed + 17) > 0.5) {
    ctx.fillStyle = "rgba(20,24,28,0.7)";
    ctx.fillRect(w * 0.12, h - 40, w * 0.76, 3);
    for (let k = 0; k < 16; k += 1) ctx.fillRect(w * 0.12 + (k / 15) * w * 0.76, h - 40, 2, 30);
  }
  // sill shadow
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fillRect(wx - 12, wy + wh + 10, ww + 24, 6);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.repeat.set(1 / BAY_M, 1 / STOREY_M);
  return tex;
}

function roofTexture(seed) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const terracotta = hash(seed + 3) > 0.55;
  ctx.fillStyle = terracotta ? "#a8583f" : "#8d8a83";
  ctx.fillRect(0, 0, 128, 128);
  for (let i = 0; i < 400; i += 1) {
    ctx.fillStyle = `rgba(${terracotta ? "40,15,5" : "20,20,20"},${0.05 + hash(i + seed) * 0.15})`;
    ctx.fillRect(hash(i * 3 + seed) * 128, hash(i * 7 + seed) * 128, 3 + hash(i) * 6, 2);
  }
  if (terracotta) {
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    for (let y = 0; y < 128; y += 12) ctx.fillRect(0, y, 128, 2);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.repeat.set(1 / 6, 1 / 6);
  return tex;
}

function concreteTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#8f9296";
  ctx.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 1800; i += 1) {
    ctx.fillStyle = `rgba(${hash(i) > 0.5 ? "255,255,255" : "0,0,0"},${0.03 + hash(i * 3) * 0.08})`;
    ctx.fillRect(hash(i * 5) * 256, hash(i * 11) * 256, 1 + hash(i * 13) * 3, 1 + hash(i * 17) * 3);
  }
  // panel joints and soot line
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.fillRect(0, 126, 256, 3);
  ctx.fillRect(126, 0, 3, 256);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.repeat.set(1 / 4, 1 / 4);
  return tex;
}

/* --------------------------------------------------------------- buildings */

function polygonArea(ring) {
  let a = 0;
  for (let i = 0; i < ring.length; i += 1) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[(i + 1) % ring.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

function buildingHeight(b, i) {
  if (b.height > 0) return b.height;
  // Monaco: dense 6-14 storey blocks; small footprints are lower annexes.
  const area = polygonArea(b.rings[0]);
  const storeys = Math.max(2, Math.min(14, Math.round(3 + Math.sqrt(area) * 0.22 + hash(i) * 4)));
  return storeys * STOREY_M;
}

/** Vertex range [start, start+count) of a non-indexed geometry as its own geometry. */
function sliceGeometry(geometry, start, count) {
  const out = new THREE.BufferGeometry();
  for (const name of ["position", "normal", "uv"]) {
    const attr = geometry.getAttribute(name);
    if (!attr) continue;
    const size = attr.itemSize;
    out.setAttribute(name, new THREE.Float32BufferAttribute(attr.array.slice(start * size, (start + count) * size), size));
  }
  return out;
}

export function buildBuildings(buildings) {
  const group = new THREE.Group();
  const facades = PALETTE.map(([base, trim], k) => ({
    wall: new THREE.MeshStandardMaterial({ map: facadeTexture(base, trim, k + 1), roughness: 0.9, metalness: 0.0 }),
    roof: new THREE.MeshStandardMaterial({ map: roofTexture(k + 1), roughness: 1.0 }),
    walls: [],
    roofs: [],
  }));
  buildings.forEach((b, i) => {
    const outer = b.rings[0];
    if (!outer || outer.length < 3) return;
    const shape = new THREE.Shape(outer.map(([x, y]) => new THREE.Vector2(x, y)));
    for (const hole of b.rings.slice(1)) {
      if (hole.length >= 3) shape.holes.push(new THREE.Path(hole.map(([x, y]) => new THREE.Vector2(x, y))));
    }
    const height = buildingHeight(b, i);
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: false, steps: 1 });
    geometry.rotateX(-Math.PI / 2); // extrusion +z -> up, shape y -> world -z
    // ExtrudeGeometry is non-indexed with two groups: [0] caps, [1] side walls.
    const [caps, sides] = geometry.groups;
    const f = facades[Math.floor(hash(i * 31 + 7) * facades.length)];
    f.roofs.push(sliceGeometry(geometry, caps.start, caps.count));
    f.walls.push(sliceGeometry(geometry, sides.start, sides.count));
    geometry.dispose();
  });
  facades.forEach((f) => {
    if (f.walls.length) {
      const merged = mergeGeometries(f.walls, false);
      const mesh = new THREE.Mesh(merged, f.wall);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
    }
    if (f.roofs.length) {
      const merged = mergeGeometries(f.roofs, false);
      const mesh = new THREE.Mesh(merged, f.roof);
      mesh.receiveShadow = true;
      group.add(mesh);
    }
  });
  return group;
}

/* ------------------------------------------------------------- sweep utils */

/** Sweep a 2D profile [(lateral, height), ...] along centerline frames i0..i1 (inclusive). */
function sweep(frames, i0, i1, profile, closedLoop = false, uScale = 1) {
  const n = frames.length;
  const count = closedLoop ? n + 1 : i1 - i0 + 1;
  const positions = [];
  const uvs = [];
  const indices = [];
  const m = profile.length;
  for (let k = 0; k < count; k += 1) {
    const fi = closedLoop ? k % n : i0 + k;
    const f = frames[fi];
    const s = closedLoop && k === n ? frames[n - 1].s + 2 : f.s;
    profile.forEach(([lat, hgt], j) => {
      positions.push(f.p[0] + f.normal[0] * lat, hgt, -(f.p[1] + f.normal[1] * lat));
      uvs.push(s / uScale, j / (m - 1));
    });
    if (k < count - 1) {
      for (let j = 0; j < m - 1; j += 1) {
        const a = k * m + j;
        const b = a + m;
        indices.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/* ---------------------------------------------------------------- barriers */

/** W-beam (Armco) profile in metres: lateral bulge x, height y; corrugated. */
function wBeamProfile(side, base) {
  const d = 0.055 * side; // corrugation depth, away from the road
  const pts = [
    [0, 0],
    [d, 0.035],
    [0, 0.075],
    [d, 0.115],
    [0, 0.155],
    [d, 0.195],
    [0, 0.235],
    [d, 0.275],
    [0, 0.31],
  ];
  return pts.map(([x, y]) => [x, base + y]);
}

export function buildBarriers(frames, offset) {
  const group = new THREE.Group();
  const steel = new THREE.MeshStandardMaterial({
    color: 0xb4b8bd,
    metalness: 0.85,
    roughness: 0.42,
    side: THREE.DoubleSide,
  });
  const postMaterial = new THREE.MeshStandardMaterial({ color: 0x6e7378, metalness: 0.7, roughness: 0.6 });
  const railGeos = [];
  [-1, 1].forEach((side) => {
    RAIL_HEIGHTS.forEach((base) => {
      const profile = wBeamProfile(side, base).map(([x, y]) => [side * offset + x, y]);
      railGeos.push(sweep(frames, 0, frames.length - 1, profile, true, 4));
    });
  });
  const rails = new THREE.Mesh(mergeGeometries(railGeos, false), steel);
  rails.castShadow = true;
  rails.receiveShadow = true;
  group.add(rails);

  // Posts: I-section uprights every 2 m, instanced. Two per position (both sides).
  const total = frames[frames.length - 1].s;
  const nPosts = Math.floor(total / POST_SPACING_M);
  const post = new THREE.BoxGeometry(0.12, 1.45, 0.18);
  const posts = new THREE.InstancedMesh(post, postMaterial, nPosts * 2);
  const dummy = new THREE.Object3D();
  let fi = 0;
  let placed = 0;
  for (let k = 0; k < nPosts; k += 1) {
    const s = k * POST_SPACING_M;
    while (fi < frames.length - 1 && frames[fi + 1].s < s) fi += 1;
    const f = frames[fi];
    const heading = Math.atan2(-f.normal[0], f.normal[1]); // tangent (tx, ty) = (n[1], -n[0])
    [-1, 1].forEach((side) => {
      const lat = side * (offset + 0.1); // just behind the rail, away from the road
      dummy.position.set(f.p[0] + f.normal[0] * lat, 0.72, -(f.p[1] + f.normal[1] * lat));
      dummy.rotation.set(0, heading, 0);
      dummy.updateMatrix();
      posts.setMatrixAt(placed, dummy.matrix);
      placed += 1;
    });
  }
  posts.count = placed;
  posts.castShadow = true;
  posts.receiveShadow = true;
  group.add(posts);
  return group;
}

/* ------------------------------------------------------------------ tunnel */

export function buildTunnel(frames, spans, halfwidth) {
  const group = new THREE.Group();
  if (!spans || !spans.length) return group;
  const concrete = new THREE.MeshStandardMaterial({ map: concreteTexture(), roughness: 0.95, side: THREE.DoubleSide });
  const shell = new THREE.MeshStandardMaterial({ color: 0x3b3f44, roughness: 1.0, side: THREE.DoubleSide });
  const lamp = new THREE.MeshStandardMaterial({ color: 0xffc98a, emissive: 0xffb45c, emissiveIntensity: 2.4, roughness: 0.4 });
  const wall = halfwidth + 0.9; // barrier, narrow pavement, then the wall
  const h0 = 4.4; // vertical wall height
  const apex = 6.4;
  // interior: vertical walls up to h0 then an arch to the apex
  const profile = [];
  profile.push([-wall - 0.5, -0.1], [-wall, 0.0], [-wall, h0]);
  const arcN = 12;
  for (let k = 1; k < arcN; k += 1) {
    const t = k / arcN;
    const x = -wall + t * 2 * wall;
    const y = h0 + (apex - h0) * Math.sin(Math.PI * t);
    profile.push([x, y]);
  }
  profile.push([wall, h0], [wall, 0.0], [wall + 0.5, -0.1]);
  // outer shell 0.6 m thicker so it reads as a structure where visible
  const outer = profile.map(([x, y]) => [x * 1.06, y + 0.6]);
  spans.forEach(([i0, i1]) => {
    const a = Math.max(0, Math.min(i0, frames.length - 1));
    const b = Math.max(0, Math.min(i1, frames.length - 1));
    if (b <= a) return;
    const inner = new THREE.Mesh(sweep(frames, a, b, profile, false, 4), concrete);
    inner.receiveShadow = true;
    group.add(inner);
    group.add(new THREE.Mesh(sweep(frames, a, b, outer, false, 4), shell));
    // portal frames at both ends
    [a, b].forEach((end) => {
      const e0 = Math.max(a, end - 1);
      const e1 = Math.min(b, end + 1);
      const ring = profile.map(([x, y]) => [x * 1.12, y + 0.9]);
      group.add(new THREE.Mesh(sweep(frames, e0, e1, ring, false, 4), shell));
    });
    // sodium lamps along the crown every ~8 m, plus a continuous light rail
    const lampGeos = [];
    let lastBay = -1;
    for (let i = a; i <= b; i += 1) {
      const f = frames[i];
      const bay = Math.floor(f.s / 8);
      if (bay === lastBay) continue;
      lastBay = bay;
      const g = new THREE.BoxGeometry(1.2, 0.18, 0.5);
      g.translate(f.p[0], apex - 0.35, -f.p[1]);
      lampGeos.push(g);
    }
    if (lampGeos.length) group.add(new THREE.Mesh(mergeGeometries(lampGeos, false), lamp));
    const rail = sweep(frames, a, b, [[-0.3, apex - 0.15], [0.3, apex - 0.15]], false, 4);
    group.add(new THREE.Mesh(rail, new THREE.MeshStandardMaterial({ color: 0xffe2b8, emissive: 0xffd08a, emissiveIntensity: 0.9 })));
    // a wide pavement strip inside on both sides
    [-1, 1].forEach((side) => {
      const kerb = sweep(frames, a, b, [[side * (halfwidth + 0.1), 0.0], [side * wall, 0.12], [side * wall, 0.0]], false, 4);
      group.add(new THREE.Mesh(kerb, concrete));
    });
  });
  return group;
}

/* ------------------------------------------------------------- quays/piers */

export function buildPiers(lines) {
  const group = new THREE.Group();
  const concrete = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.9 });
  lines
    .filter((l) => l.kind === "pier" || l.kind === "breakwater")
    .forEach((l) => {
      const pts = l.points;
      for (let i = 0; i + 1 < pts.length; i += 1) {
        const [x0, y0] = pts[i];
        const [x1, y1] = pts[i + 1];
        const len = Math.hypot(x1 - x0, y1 - y0);
        if (len < 0.5) continue;
        const geometry = new THREE.BoxGeometry(len, 1.2, l.kind === "breakwater" ? 8 : 4);
        const mesh = new THREE.Mesh(geometry, concrete);
        mesh.position.copy(toWorld((x0 + x1) / 2, (y0 + y1) / 2, 0.3));
        mesh.rotation.y = Math.atan2(-(y1 - y0), x1 - x0);
        mesh.receiveShadow = true;
        mesh.castShadow = true;
        group.add(mesh);
      }
    });
  return group;
}

export { toWorld };
