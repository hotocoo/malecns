"use strict";

/* Chase-camera 3D view of the driving task, rendered with three.js.
 *
 * The simulation is planar: a car on a flat track with a 2D lidar. Everything
 * here is a faithful lift of that state into 3D. The road surface is the true
 * drivable band extruded from the centerline, the rays are the real lidar
 * returns, the car sits exactly where the simulated body is. Lighting comes
 * from a real sky capture so the paint and asphalt respond like materials.
 *
 * Assets (all vendored under /assets and /vendor):
 *   Ferrari 458 model: three.js examples, by vicent091036, CC-BY (stand-in
 *   body; the simulated dynamics are a Mercedes-AMG F1 W11)
 *   asphalt_02, kloofendal_48d_partly_cloudy_puresky: Poly Haven, CC0
 *   buildings, tunnel, quays: OpenStreetMap contributors, ODbL (see scenery.js)
 */

import * as THREE from "three";
import { GLTFLoader } from "/vendor/three/loaders/GLTFLoader.js";
import { DRACOLoader } from "/vendor/three/loaders/DRACOLoader.js";
import { RGBELoader } from "/vendor/three/loaders/RGBELoader.js";
import { buildBarriers, buildBuildings, buildGround, buildPiers, buildTunnel, buildWater } from "/scenery.js";
import { FlyRig } from "/flyrig.js";

/* All dimensions, camera figures, asset URLs and colours come from the
 * server's DriveStyleConfig (track.style) and CarConfig (track.car): nothing
 * about the vehicle or the scene is written here. A car glTF at
 * `style.car_model_url` replaces the stand-in body; either model is fitted to
 * the physics footprint (track.car.length x track.car.width) from its own
 * bounding box, so what you see is exactly what can hit the barrier. */

/* Sim frame -> world frame: sim x is world x, sim y is world -z, up is +y.
 * The physics is planar; heights come from the surveyed terrain the server
 * attaches to the track (track.centerline_z, scenery.terrain) and are added
 * here so the car, the road and the buildings stand where the ground is. */
const toWorld = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

function loadTexture(loader, url, repeat, colorSpace) {
  const tex = loader.load(url);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  if (repeat) tex.repeat.set(repeat[0], repeat[1]);
  if (colorSpace) tex.colorSpace = colorSpace;
  return tex;
}

/* ------------------------------------------------------------- track meshes */

function centerlineFrames(centerline, heights) {
  const n = centerline.length;
  let length = 0;
  return centerline.map((p, i) => {
    const prev = centerline[(i - 1 + n) % n];
    const next = centerline[(i + 1) % n];
    const tx = next[0] - prev[0];
    const ty = next[1] - prev[1];
    const len = Math.hypot(tx, ty) || 1;
    if (i > 0) length += Math.hypot(p[0] - prev[0], p[1] - prev[1]);
    return { p, normal: [-ty / len, tx / len], s: length, h: heights ? heights[i] : 0 };
  });
}

/* Nearest centreline frame through a coarse spatial grid (GRID_M metre cells),
 * so the per-frame lookups for every car, ray end and camera point stay O(1)
 * instead of scanning all 2,048 frames each time. */
const GRID_M = 25;

function buildFrameGrid(frames) {
  const grid = new Map();
  frames.forEach((f, i) => {
    const key = `${Math.floor(f.p[0] / GRID_M)},${Math.floor(f.p[1] / GRID_M)}`;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  });
  return grid;
}

function nearestFrame(frames, x, y) {
  let best = -1;
  let bestD = Infinity;
  const grid = frames.grid;
  if (grid) {
    const cx = Math.floor(x / GRID_M);
    const cy = Math.floor(y / GRID_M);
    for (let r = 1; r <= 3 && best < 0; r += 1) {
      for (let dx = -r; dx <= r; dx += 1) {
        for (let dy = -r; dy <= r; dy += 1) {
          const cell = grid.get(`${cx + dx},${cy + dy}`);
          if (!cell) continue;
          for (const i of cell) {
            const p = frames[i].p;
            const d = (p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y);
            if (d < bestD) {
              bestD = d;
              best = i;
            }
          }
        }
      }
    }
    if (best >= 0) return best;
  }
  for (let i = 0; i < frames.length; i += 1) {
    const p = frames[i].p;
    const d = (p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** Road height at a sim point: surveyed heights interpolated along the
 * centreline segment the point projects onto. Snapping to the nearest frame
 * made the car and camera step up and down by the height difference between
 * neighbouring frames forty times a second (the "bouncing" car). */
function roadHeightAt(frames, x, y) {
  if (!frames || !frames.length) return 0;
  const n = frames.length;
  const i = nearestFrame(frames, x, y);
  const p = frames[i].p;
  let best = frames[i].h;
  let bestD = Infinity;
  for (const j of [(i + 1) % n, (i - 1 + n) % n]) {
    const q = frames[j].p;
    const ex = q[0] - p[0];
    const ey = q[1] - p[1];
    const len2 = ex * ex + ey * ey || 1;
    const t = Math.max(0, Math.min(1, ((x - p[0]) * ex + (y - p[1]) * ey) / len2));
    const dx = p[0] + ex * t - x;
    const dy = p[1] + ey * t - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = frames[i].h + (frames[j].h - frames[i].h) * t;
    }
  }
  return best;
}

/** Road pitch (rad, nose up positive) under a car of length `span` at heading. */
function roadPitchAt(frames, x, y, heading, span) {
  if (!frames || !frames.length || !span) return 0;
  const hx = (Math.cos(heading) * span) / 2;
  const hy = (Math.sin(heading) * span) / 2;
  return Math.atan2(roadHeightAt(frames, x + hx, y + hy) - roadHeightAt(frames, x - hx, y - hy), span);
}

/** Pose a car mesh on the road: position, heading and grade pitch (Euler YZX: yaw, then pitch about the lateral axis). */
function poseCar(mesh, frames, x, y, heading, length) {
  mesh.position.copy(toWorld(x, y, roadHeightAt(frames, x, y)));
  mesh.rotation.order = "YZX";
  mesh.rotation.y = heading;
  mesh.rotation.z = roadPitchAt(frames, x, y, heading, length);
}

/** Closed ribbon between offsets [a, b] from the centerline, UV v along length. */
function ribbon(frames, a, b, height, vScale) {
  const n = frames.length;
  const positions = [];
  const uvs = [];
  const indices = [];
  for (let i = 0; i <= n; i += 1) {
    const f = frames[i % n];
    const s = i === n ? frames[n - 1].s + Math.hypot(f.p[0] - frames[n - 1].p[0], f.p[1] - frames[n - 1].p[1]) : f.s;
    [a, b].forEach((offset, k) => {
      positions.push(f.p[0] + f.normal[0] * offset, f.h + height, -(f.p[1] + f.normal[1] * offset));
      uvs.push(k, s / vScale);
    });
    if (i < n) {
      const base = i * 2;
      indices.push(base, base + 2, base + 1, base + 1, base + 2, base + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** Closed vertical ribbon at a fixed offset from the centerline, y0..y1. */
function verticalRibbon(frames, offset, y0, y1) {
  const n = frames.length;
  const positions = [];
  const uvs = [];
  const indices = [];
  for (let i = 0; i <= n; i += 1) {
    const f = frames[i % n];
    const x = f.p[0] + f.normal[0] * offset;
    const z = -(f.p[1] + f.normal[1] * offset);
    positions.push(x, f.h + y0, z, x, f.h + y1, z);
    uvs.push(f.s / 4, 0, f.s / 4, 1);
    if (i < n) {
      const b = i * 2;
      indices.push(b, b + 2, b + 1, b + 1, b + 2, b + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function kerbTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 4;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#c8332f";
  ctx.fillRect(0, 0, 4, 32);
  ctx.fillStyle = "#e9e6df";
  ctx.fillRect(0, 32, 4, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  return tex;
}

/* ------------------------------------------------------------------ viewer */

export class DriveView {
  constructor(canvas, style) {
    this.canvas = canvas;
    this.style = style;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = style.exposure;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(style.camera_fov_deg, 1.6, 0.3, 2400);
    this.cameraGoal = new THREE.Vector3(0, 5, 12);
    this.lookGoal = new THREE.Vector3();
    this.camera.position.copy(this.cameraGoal);

    this.frame = null;
    this.wheels = [];
    this.wheelSpin = 0;
    // fleet: one mesh per simulated car (car k at fleet[k]); the camera follows frame.follow
    this.fleet = [];
    this.fleetSize = 1;
    // displayed pose per car: eased towards the latest simulated pose and
    // dead-reckoned between frames (see _smoothPose)
    this.smooth = [];
    this.frameTime = performance.now();
    this.lastStep = null;
    this.simRate = 1.0; // simulated seconds per wall second, measured from frame arrivals
    this.mode = "chase";
    this.roll = 0;
    this.rig = new FlyRig(style); // the driver, seated in the followed car (see _seatDriver)
    this.rigParent = null;
    this.lastTime = performance.now();
    this.textures = new THREE.TextureLoader();
    this._initLights();
    this._initSky();
  }

  _initLights() {
    const sun = new THREE.DirectionalLight(0xfff2dc, this.style.sun_intensity);
    sun.position.set(...this.style.sun_offset_m);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    const cam = sun.shadow.camera;
    cam.near = 20;
    cam.far = 260;
    cam.left = cam.bottom = -110;
    cam.right = cam.top = 110;
    this.sun = sun;
    this.scene.add(sun, sun.target);
    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x4a5a3a, 0.35));
    this.scene.fog = new THREE.FogExp2(0xb9c6d3, this.style.fog_density);
  }

  _initSky() {
    new RGBELoader().load("/assets/sky_1k.hdr", (hdr) => {
      hdr.mapping = THREE.EquirectangularReflectionMapping;
      this.scene.environment = hdr;
      this.scene.background = hdr;
      this.scene.backgroundBlurriness = 0.02;
      this.scene.environmentIntensity = 0.9;
    });
  }

  load(track) {
    // Everything the track owns lives in one group so a curriculum road-width
    // change can rebuild it: without this, a second load() stacked a second
    // road, kerbs and barriers on top of the first.
    if (this.trackGroup) {
      this.scene.remove(this.trackGroup);
      this.trackGroup.traverse((node) => {
        if (node.geometry) node.geometry.dispose();
      });
    }
    this.trackGroup = new THREE.Group();
    this.scene.add(this.trackGroup);
    this.track = track;
    const frames = centerlineFrames(track.centerline, track.centerline_z);
    frames.grid = buildFrameGrid(frames);
    const hw = track.halfwidth;

    const asphalt = new THREE.MeshStandardMaterial({
      map: loadTexture(this.textures, "/assets/asphalt_diff.jpg", [3, 1], THREE.SRGBColorSpace),
      normalMap: loadTexture(this.textures, "/assets/asphalt_nor.jpg", [3, 1]),
      roughnessMap: loadTexture(this.textures, "/assets/asphalt_rough_ao.jpg", [3, 1]),
      aoMap: loadTexture(this.textures, "/assets/asphalt_rough_ao.jpg", [3, 1]),
      roughness: 1.0,
      metalness: 0.0,
      color: 0x9a9a9a,
    });
    const st = this.style;
    const road = new THREE.Mesh(ribbon(frames, -hw, hw, 0.0, st.road_texture_m), asphalt);
    road.receiveShadow = true;
    this.trackGroup.add(road);

    const kerbMaterial = new THREE.MeshStandardMaterial({ map: kerbTexture(), roughness: 0.75 });
    kerbMaterial.map.repeat.set(1, 1);
    // The kerb is the outer 1.2 m of the drivable band and the barrier face
    // stands exactly at the road edge, which is where the physics puts the
    // wall: lidar rays end there and a body touching it has crashed.
    [-1, 1].forEach((side) => {
      const kerb = new THREE.Mesh(
        ribbon(frames, side * (hw - st.kerb_width_m), side * hw, 0.04, 2.5),
        kerbMaterial,
      );
      kerb.receiveShadow = true;
      this.trackGroup.add(kerb);
      const line = new THREE.Mesh(
        ribbon(frames, side * (hw - st.line_width_m - st.line_inset_m), side * (hw - st.line_inset_m), 0.01, 1),
        new THREE.MeshStandardMaterial({ color: 0xe8e4d8, roughness: 0.6 }),
      );
      this.trackGroup.add(line);
    });

    // Road-edge skirt: on a grade the carved ground beside the road sits under
    // the lowest nearby stretch (so it never pokes through the asphalt); this
    // face closes the gap between the road edge and that ground.
    if (track.has_terrain) {
      const skirtMaterial = new THREE.MeshStandardMaterial({ color: 0x5a5a5c, roughness: 1.0, side: THREE.DoubleSide });
      [-1, 1].forEach((side) => {
        const skirt = new THREE.Mesh(verticalRibbon(frames, side * hw, -3.0, 0.0), skirtMaterial);
        skirt.receiveShadow = true;
        this.trackGroup.add(skirt);
      });
    }

    // Armco with posts along both edges; the rail face stands where the
    // physics puts the wall (the road edge) plus the configured offset.
    this.sceneryStyle = null;
    this.frames = frames;
    this.barrierOffset = hw;

    const urban = track.urban;
    const groundRepeat = urban ? st.ground_repeat_urban : st.ground_repeat_rural;
    const groundMaterial = new THREE.MeshStandardMaterial({
      map: loadTexture(this.textures, urban ? "/assets/asphalt_diff.jpg" : "/assets/ground_diff.jpg", groundRepeat, THREE.SRGBColorSpace),
      normalMap: loadTexture(this.textures, urban ? "/assets/asphalt_nor.jpg" : "/assets/ground_nor.jpg", groundRepeat),
      roughnessMap: loadTexture(this.textures, urban ? "/assets/asphalt_rough_ao.jpg" : "/assets/ground_rough_ao.jpg", groundRepeat),
      roughness: 1.0,
      color: urban ? 0x7d7f82 : 0x8b9a6a,
    });
    this.groundMaterial = groundMaterial;
    if (!track.has_water && !track.has_terrain) {
      const extent = track.extent * st.ground_extent_factor;
      const ground = new THREE.Mesh(new THREE.PlaneGeometry(extent, extent), groundMaterial);
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = -0.03;
      ground.receiveShadow = true;
      this.trackGroup.add(ground);
    }

    this._loadScenery(frames, track);

    this._loadCar();
    this._initRays(track.n_rays);
  }

  /** Scenery (buildings, tunnel, quays, water) and the barriers: all dimensions from the server's SceneryStyleConfig. */
  async _loadScenery(frames, track) {
    try {
      const scenery = await (await fetch("/api/scenery")).json();
      const t0 = performance.now();
      const sty = scenery.style;
      this.trackGroup.add(buildBarriers(frames, track.halfwidth + sty.rail_offset_m, sty));
      if (scenery.terrain) {
        // surveyed ground: the hill the circuit climbs, carved flat under the road
        const ground = buildGround(scenery.terrain, this.groundMaterial);
        this.trackGroup.add(ground);
        this.terrain = scenery.terrain;
      }
      if (!track.has_scenery) return;
      this.trackGroup.add(buildBuildings(scenery.buildings, sty));
      this.trackGroup.add(buildTunnel(frames, scenery.tunnel_spans, track.halfwidth, sty, scenery.terrain || null));
      this.trackGroup.add(buildPiers(scenery.lines || [], scenery.water ? scenery.water.level : 0));
      const water = buildWater(scenery.water, this.groundMaterial, sty, scenery.terrain);
      this.trackGroup.add(water.group);
      this.waterMaterial = water.material;
      // the tunnel needs a longer shadow reach and a slightly darker fog inside
      console.info(
        `scenery: ${scenery.buildings.length} buildings, tunnel spans ${JSON.stringify(scenery.tunnel_spans)}, ` +
          `${scenery.water ? scenery.water.water.length : 0} water rects, terrain ${scenery.terrain ? `${scenery.terrain.rows}x${scenery.terrain.cols} (${scenery.terrain.source})` : "flat"} in ${Math.round(performance.now() - t0)} ms`,
      );
      this.tunnelSpans = scenery.tunnel_spans;
    } catch (err) {
      console.warn("scenery unavailable", err);
    }
  }

  _loadCar() {
    const draco = new DRACOLoader().setDecoderPath("/vendor/three/draco/");
    const loader = new GLTFLoader().setDRACOLoader(draco);
    this.car = new THREE.Group();
    this.scene.add(this.car);

    const url = this.style.car_model_url;
    fetch(url, { method: "HEAD" })
      .then((res) => {
        if (!res.ok) throw new Error(`${url} ${res.status}`);
        return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));
      })
      .then((gltf) => this._mountModel(gltf))
      .catch((err) => {
        console.info(`no vehicle model at ${url} (${err.message}); using the stand-in body fitted to the physics footprint`);
        this._loadStandIn(loader);
      });
  }

  /** Fit any car glTF to the physics footprint: length and width from its bounding box, wheels on the ground, nose down sim +x. */
  _mountModel(gltf) {
    const model = gltf.scene;
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const forwardAxis = this.style.car_model_forward.replace("-", "");
    const length = forwardAxis === "x" ? size.x : size.z;
    const width = forwardAxis === "x" ? size.z : size.x;
    const car = this.track.car;
    const kL = car.length / Math.max(length, 1e-3);
    const kW = car.width / Math.max(width, 1e-3);
    const k = kL;
    model.scale.set(forwardAxis === "x" ? kL : kW, kL, forwardAxis === "x" ? kW : kL);
    model.position.set(-box.getCenter(new THREE.Vector3()).x * model.scale.x, -box.min.y * k, -box.getCenter(new THREE.Vector3()).z * model.scale.z);
    model.traverse((node) => {
      if (node.isMesh) {
        node.castShadow = true;
        node.receiveShadow = true;
      }
    });
    this.wheels = [];
    model.traverse((node) => {
      if (/wheel|tyre|tire/i.test(node.name) && !/brake|disc|rim_?cap/i.test(node.name)) this.wheels.push(node);
    });
    const rig = new THREE.Group();
    rig.add(model);
    // rotate the model's nose onto sim +x (world +x)
    const yaw = { "-z": -Math.PI / 2, z: Math.PI / 2, x: 0, "-x": Math.PI }[this.style.car_model_forward] ?? -Math.PI / 2;
    rig.rotation.y = yaw;
    this.car.add(rig);
    this._spawnFleet();
    console.info(`vehicle model mounted: ${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)} native, fitted to ${car.length} x ${car.width} m, ${this.wheels.length} wheel nodes`);
  }

  _loadStandIn(loader) {
    loader.load(this.style.stand_in_url, (gltf) => {
      const model = gltf.scene.children[0];
      const body = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(this.style.stand_in_body_colour),
        metalness: 1.0,
        roughness: 0.42,
        clearcoat: 1.0,
        clearcoatRoughness: 0.06,
      });
      const details = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 1.0, roughness: 0.5 });
      const glass = new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        metalness: 0.25,
        roughness: 0,
        transmission: 1.0,
        transparent: true,
      });
      model.getObjectByName("body").material = body;
      ["rim_fl", "rim_fr", "rim_rr", "rim_rl", "trim"].forEach((name) => {
        const part = model.getObjectByName(name);
        if (part) part.material = details;
      });
      const glassPart = model.getObjectByName("glass");
      if (glassPart) glassPart.material = glass;
      this.wheels = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
        .map((name) => model.getObjectByName(name))
        .filter(Boolean);
      model.traverse((node) => {
        if (node.isMesh) {
          node.castShadow = true;
          node.receiveShadow = true;
        }
      });

      const shadow = new THREE.Mesh(
        new THREE.PlaneGeometry(0.655 * 4, 1.3 * 4),
        new THREE.MeshBasicMaterial({
          map: loadTexture(this.textures, this.style.stand_in_shadow_url, null, THREE.SRGBColorSpace),
          blending: THREE.MultiplyBlending,
          toneMapped: false,
          transparent: true,
        }),
      );
      shadow.rotation.x = -Math.PI / 2;
      shadow.position.y = 0.005;
      shadow.renderOrder = 2;
      // The model's nose points along -z; the sim's heading is +x. Rotate the
      // whole car once so heading 0 drives down world +x.
      const rig = new THREE.Group();
      rig.add(model, shadow);
      rig.rotation.y = -Math.PI / 2;
      // Stretch the stand-in body to the physics footprint (length along the
      // model's -z nose axis, width along x) so it covers exactly what can
      // touch the barrier; height follows the width so proportions stay sane.
      model.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const car = this.track.car;
      const kL = car.length / Math.max(size.z, 1e-3);
      const kW = car.width / Math.max(size.x, 1e-3);
      rig.scale.set(kW, kW, kL);
      this.car.add(rig);
      this._spawnFleet();
      console.info(`stand-in body ${size.x.toFixed(2)} x ${size.z.toFixed(2)} m native, fitted to ${car.length} x ${car.width} m`);
    });
  }

  /** Number of cars to draw; clones of the vehicle mesh are made once the model is mounted. */
  setFleetSize(n) {
    this.fleetSize = Math.max(1, n | 0);
    this._spawnFleet();
  }

  _spawnFleet() {
    if (!this.car || this.car.children.length === 0) return;
    if (this.fleet.length === 0) this.fleet.push(this.car);
    while (this.fleet.length < this.fleetSize) {
      const clone = this.car.clone(true);
      this.scene.add(clone);
      this.fleet.push(clone);
    }
    this.fleet.forEach((mesh, k) => {
      mesh.visible = k < this.fleetSize;
    });
  }

  /** "chase" (default) or "cockpit": the fly's eye point inside the followed car with the fly rig in view. */
  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    const st = this.style;
    this.camera.fov = mode === "cockpit" ? st.cockpit_fov_deg : st.camera_fov_deg;
    this.camera.updateProjectionMatrix();
  }

  /** Put the fly in the followed car's seat (moves with the follow selection). */
  _seatDriver(mesh) {
    if (!mesh || this.rigParent === mesh) return;
    if (this.rigParent) this.rigParent.remove(this.rig.group);
    mesh.add(this.rig.group);
    this.rigParent = mesh;
  }

  _initRays(nRays) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(nRays * 6), 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(nRays * 6), 3));
    this.rays = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }),
    );
    this.rays.frustumCulled = false;
    this.scene.add(this.rays);
  }

  update(frame) {
    this.frame = frame;
    if (frame.fleet && frame.fleet.length !== this.fleetSize) this.setFleetSize(frame.fleet.length);
    const now = performance.now();
    if (this.lastStep !== null && frame.step > this.lastStep && now > this.frameTime) {
      // With the GPU shared with the trainer the simulation runs well below
      // real time; dead reckoning at the simulated speed must use this rate
      // or the car runs ahead of every frame and is pulled back by the next.
      const rate = ((frame.step - this.lastStep) * (frame.control_dt_s || 0.016)) / ((now - this.frameTime) / 1000);
      this.simRate += (Math.min(1.5, rate) - this.simRate) * 0.2;
    }
    this.lastStep = frame.step;
    this.frameTime = now;
    if (frame.fleet) {
      frame.fleet.forEach((car, k) => {
        const mesh = this.fleet[k];
        if (!mesh) return;
        this._setTarget(k, car.pos[0], car.pos[1], car.heading, car.speed || 0, car.done_reason || 0);
        mesh.visible = true;
      });
      this._seatDriver(this.fleet[frame.follow]);
    } else {
      this._setTarget(0, frame.pos[0], frame.pos[1], frame.heading, frame.speed || 0, frame.done_reason || 0);
      this._seatDriver(this.car);
    }
    const { n_rays, fov_deg, max_range } = this.track;
    const fov = (fov_deg * Math.PI) / 180;
    const pos = this.rays.geometry.getAttribute("position");
    const color = this.rays.geometry.getAttribute("color");
    const roadH = roadHeightAt(this.frames, frame.pos[0], frame.pos[1]);
    const eye = toWorld(frame.pos[0], frame.pos[1], roadH + this.style.lidar_height_m);
    frame.lidar.forEach((norm, i) => {
      // ray 0 looks left (+fov/2), the last ray right, as in car_env.py
      const offset = n_rays === 1 ? 0 : fov / 2 - (fov * i) / (n_rays - 1);
      const angle = frame.heading + offset;
      const reach = norm * max_range;
      const hit = toWorld(frame.pos[0] + Math.cos(angle) * reach, frame.pos[1] + Math.sin(angle) * reach, roadH + this.style.lidar_height_m);
      pos.setXYZ(i * 2, eye.x, eye.y, eye.z);
      pos.setXYZ(i * 2 + 1, hit.x, hit.y, hit.z);
      const hot = 1 - norm;
      color.setXYZ(i * 2, 1.0, 0.55 + 0.3 * (1 - hot), 0.25);
      color.setXYZ(i * 2 + 1, 1.0, 0.35 + 0.4 * (1 - hot), 0.15);
    });
    pos.needsUpdate = true;
    color.needsUpdate = true;
  }

  _setTarget(k, x, y, heading, speed, done) {
    const prev = this.smooth[k];
    const snapM = this.style.pose_snap_m || 8;
    const jump = !prev || Math.hypot(prev.tx - x, prev.ty - y) > snapM;
    if (!prev || jump) {
      // first frame or a reset: snap, no glide across the map
      this.smooth[k] = { x, y, heading, tx: x, ty: y, th: heading, speed, done, t: this.frameTime };
      return;
    }
    prev.tx = x;
    prev.ty = y;
    prev.th = heading;
    prev.speed = done ? 0 : speed;
    prev.done = done;
    prev.t = this.frameTime;
  }

  /* Frames arrive at the simulation's pace (well below the display rate when the
   * GPU is shared with the trainer) and unevenly. Snapping the car to each new
   * pose while the camera eased made it look as if it shifted forward and back.
   * The displayed pose dead-reckons from the last simulated pose at the
   * simulated speed and heading, and eases towards it. */
  _smoothPose(k, now, dt) {
    const s = this.smooth[k];
    if (!s) return null;
    const ahead = Math.min(0.25, Math.max(0, ((now - s.t) / 1000) * this.simRate));
    const gx = s.tx + Math.cos(s.th) * s.speed * ahead;
    const gy = s.ty + Math.sin(s.th) * s.speed * ahead;
    const ease = 1 - Math.exp(-dt * 18);
    s.x += (gx - s.x) * ease;
    s.y += (gy - s.y) * ease;
    let dh = s.th - s.heading;
    dh = Math.atan2(Math.sin(dh), Math.cos(dh));
    s.heading += dh * ease;
    return s;
  }

  resize() {
    const canvas = this.canvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      this.renderer.setPixelRatio(ratio);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
  }

  render() {
    if (!this.track) return;
    this.resize();
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastTime) / 1000);
    this.lastTime = now;

    const f = this.frame;
    const followIndex = f && f.fleet ? f.follow : 0;
    this.fleet.forEach((mesh, k) => {
      const sp = this._smoothPose(k, now, dt);
      if (sp && mesh.visible) poseCar(mesh, this.frames, sp.x, sp.y, sp.heading, this.track.car.length);
    });
    if (this.fleet.length === 0) this._smoothPose(0, now, dt);
    const sp = this.smooth[followIndex];
    const heading = sp ? sp.heading : f ? f.heading : 0;
    const [sx, sy] = sp ? [sp.x, sp.y] : f ? f.pos : [this.track.centerline[0][0], this.track.centerline[0][1]];
    const roadH = roadHeightAt(this.frames, sx, sy);
    const carPos = toWorld(sx, sy, roadH);
    const followed = (f && this.fleet[f.follow]) || this.car;
    if (this.fleet.length === 0 && sp) poseCar(followed, this.frames, sx, sy, heading, this.track.car.length);

    if (f) {
      // wheel spin from the simulated speed and the configured tyre radius
      this.wheelSpin += (f.speed / this.track.car.tyre_radius) * dt;
      this.wheels.forEach((wheel) => {
        wheel.rotation.x = this.wheelSpin;
      });
    }

    const speed = f ? f.speed : 0;
    const st = this.style;
    this.rig.update(f, dt, this.maxInputHz || 300, this.track.car.grip_max_g);
    if (this.mode === "cockpit") {
      // onboard camera on the roll hoop: the fly at the wheel in the foreground, the nose and the road ahead
      const eye = toWorld(sx - Math.cos(heading) * st.cockpit_back_m, sy - Math.sin(heading) * st.cockpit_back_m, roadH + st.cockpit_height_m);
      const lookX = sx + Math.cos(heading) * st.cockpit_ahead_m;
      const lookY = sy + Math.sin(heading) * st.cockpit_ahead_m;
      const look = toWorld(lookX, lookY, roadHeightAt(this.frames, lookX, lookY) + st.cockpit_height_m * 0.5);
      this.camera.position.copy(eye);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(look);
      // roll with lateral load: the outside of the corner rises
      const rollTarget = f ? -Math.sign(f.steer_actual || 0) * Math.min(1, (f.lat_g || 0) / (this.track.car.grip_max_g || 4.5)) * st.cockpit_roll_per_g * 4.5 : 0;
      this.roll += (rollTarget - this.roll) * Math.min(1, dt * 6);
      this.camera.rotateZ(this.roll);
      this.lookGoal.copy(look);
      this.renderer.render(this.scene, this.camera);
      return;
    }
    const back = st.camera_back_m + Math.min(st.camera_back_max_extra_m, speed * st.camera_back_per_mps);
    const ahead = st.camera_ahead_m + Math.min(st.camera_ahead_max_extra_m, speed * st.camera_ahead_per_mps);
    const height = st.camera_height_m + Math.min(st.camera_height_max_extra_m, speed * st.camera_height_per_mps);
    const goal = toWorld(sx - Math.cos(heading) * back, sy - Math.sin(heading) * back, roadH + height);
    const aheadX = sx + Math.cos(heading) * ahead;
    const aheadY = sy + Math.sin(heading) * ahead;
    const look = toWorld(aheadX, aheadY, roadHeightAt(this.frames, aheadX, aheadY) + st.lidar_height_m);
    const ease = f && f.step <= 1 ? 1 : 1 - Math.exp(-dt * st.camera_ease);
    this.camera.position.lerp(goal, ease);
    this.lookGoal.lerp(look, ease);
    this.camera.lookAt(this.lookGoal);

    if (this.waterMaterial) {
      // slow drift of the ripple normal map: the harbour is sheltered water
      this.waterMaterial.normalMap.offset.x += dt * 0.012;
      this.waterMaterial.normalMap.offset.y += dt * 0.007;
    }

    // keep the shadow frustum centred on the car
    const [ox, oy, oz] = this.style.sun_offset_m;
    this.sun.position.set(carPos.x + ox, oy, carPos.z + oz);
    this.sun.target.position.copy(carPos);
    this.sun.target.updateMatrixWorld();

    this.renderer.render(this.scene, this.camera);
  }
}
