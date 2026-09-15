"use strict";

/* The fly at the wheel: the driver of the car.
 *
 * A stylised Drosophila built from primitives and seated in the followed
 * car's cockpit (a child of the car mesh), so it is visible from the chase
 * camera and fills the onboard view. Every motion is driven by the
 * simulation frame: the head yaws into the turn with the steering command,
 * the body rolls with lateral load and pitches with the pedal, the front legs
 * turn a small wheel, the hind legs press pedals, the wings beat harder with
 * throttle, the antennae bend back with speed, and the compound eyes glow
 * with the visual drive the eye groups receive. Dimensions and colours come
 * from the server's DriveStyleConfig. */

import * as THREE from "three";

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const lerp = (a, b, t) => a + (b - a) * t;

export class FlyRig {
  constructor(style) {
    this.style = style;
    this.group = new THREE.Group();
    const L = style.fly_length_m;
    this.L = L;
    this.phase = 0;
    this.headYaw = 0;
    this.roll = 0;
    this.pitch = 0;
    this.wheelTurn = 0;
    this.pedalPress = [0, 0];

    const body = new THREE.MeshStandardMaterial({ color: new THREE.Color(style.fly_body_colour), roughness: 0.55, metalness: 0.1 });
    const bristle = new THREE.MeshStandardMaterial({ color: 0x151418, roughness: 0.9 });
    this.eyeMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color(style.fly_eye_colour),
      emissive: new THREE.Color(style.fly_eye_colour),
      emissiveIntensity: 0.3,
      roughness: 0.35,
      flatShading: true,
    });
    const wingMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(style.fly_wing_colour),
      transparent: true,
      opacity: 0.35,
      roughness: 0.1,
      side: THREE.DoubleSide,
    });

    // body: root at the thorax; the fly faces -z (same as the camera)
    this.body = new THREE.Group();
    this.group.add(this.body);
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(L * 0.19, 24, 18), body);
    thorax.scale.set(1, 0.85, 1.15);
    this.body.add(thorax);
    this.abdomen = new THREE.Mesh(new THREE.SphereGeometry(L * 0.21, 24, 18), body);
    this.abdomen.scale.set(0.9, 0.8, 1.6);
    this.abdomen.position.set(0, -L * 0.04, L * 0.42);
    this.body.add(this.abdomen);
    // striped abdomen bands
    for (let i = 0; i < 4; i++) {
      const band = new THREE.Mesh(new THREE.TorusGeometry(L * 0.19 - i * L * 0.02, L * 0.012, 8, 32), bristle);
      band.position.set(0, -L * 0.04, L * 0.3 + i * L * 0.11);
      this.body.add(band);
    }

    // head with two compound eyes and antennae
    this.head = new THREE.Group();
    this.head.position.set(0, L * 0.06, -L * 0.27);
    this.body.add(this.head);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(L * 0.13, 20, 16), body);
    skull.scale.set(1.15, 1, 0.9);
    this.head.add(skull);
    [-1, 1].forEach((side) => {
      const eye = new THREE.Mesh(new THREE.IcosahedronGeometry(L * 0.085, 2), this.eyeMaterial);
      eye.position.set(side * L * 0.1, L * 0.01, -L * 0.03);
      this.head.add(eye);
    });
    this.antennae = [-1, 1].map((side) => {
      const pivot = new THREE.Group();
      pivot.position.set(side * L * 0.035, L * 0.05, -L * 0.12);
      const stalk = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.006, L * 0.004, L * 0.14, 6), bristle);
      stalk.position.set(0, L * 0.07, 0);
      pivot.add(stalk);
      const arista = new THREE.Mesh(new THREE.ConeGeometry(L * 0.01, L * 0.05, 6), bristle);
      arista.position.set(0, L * 0.16, 0);
      pivot.add(arista);
      pivot.rotation.x = -0.6;
      pivot.userData.side = side;
      this.head.add(pivot);
      return pivot;
    });

    // wings: hinged at the thorax, beating about the hinge
    this.wings = [-1, 1].map((side) => {
      const hinge = new THREE.Group();
      hinge.position.set(side * L * 0.1, L * 0.14, L * 0.02);
      const blade = new THREE.Mesh(new THREE.PlaneGeometry(L * 0.5, L * 0.18), wingMat);
      blade.position.set(side * L * 0.25, 0, L * 0.05);
      blade.rotation.x = -Math.PI / 2;
      blade.rotation.z = side * 0.1;
      hinge.add(blade);
      // vein
      const vein = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.004, L * 0.003, L * 0.48, 5), bristle);
      vein.rotation.z = Math.PI / 2;
      vein.position.set(side * L * 0.24, 0.001, L * 0.02);
      hinge.add(vein);
      hinge.userData.side = side;
      this.body.add(hinge);
      return hinge;
    });

    // steering wheel in front of the fly, front legs on it
    this.wheel = new THREE.Group();
    this.wheel.position.set(0, -L * 0.02, -L * 0.62);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(L * 0.2, L * 0.018, 10, 40), new THREE.MeshStandardMaterial({ color: 0x1c1c1f, roughness: 0.5 }));
    this.wheel.add(rim);
    [0, 2.1, 4.2].forEach((a) => {
      const spoke = new THREE.Mesh(new THREE.BoxGeometry(L * 0.03, L * 0.19, L * 0.02), new THREE.MeshStandardMaterial({ color: 0x9aa3ad, metalness: 0.8, roughness: 0.3 }));
      spoke.position.set(Math.sin(a) * L * 0.095, Math.cos(a) * L * 0.095, 0);
      spoke.rotation.z = -a;
      this.wheel.add(spoke);
    });
    const marker = new THREE.Mesh(new THREE.SphereGeometry(L * 0.02, 8, 8), new THREE.MeshStandardMaterial({ color: 0xffb020 }));
    marker.position.set(0, L * 0.2, 0);
    this.wheel.add(marker);
    this.group.add(this.wheel);

    // six legs: two front pairs on the wheel, hind pair on the pedals
    this.legs = [];
    const legMat = bristle;
    const mkLeg = (x, y, z, len) => {
      const pivot = new THREE.Group();
      pivot.position.set(x, y, z);
      const femur = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.012, L * 0.009, len, 6), legMat);
      femur.position.set(0, -len / 2, 0);
      pivot.add(femur);
      const knee = new THREE.Group();
      knee.position.set(0, -len, 0);
      const tibia = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.009, L * 0.006, len * 0.9, 6), legMat);
      tibia.position.set(0, -len * 0.45, 0);
      knee.add(tibia);
      pivot.add(knee);
      pivot.userData.knee = knee;
      this.body.add(pivot);
      return pivot;
    };
    this.frontLegs = [-1, 1].map((side) => {
      const leg = mkLeg(side * L * 0.13, -L * 0.02, -L * 0.14, L * 0.22);
      leg.rotation.x = -1.35; // reach forward to the wheel
      leg.rotation.z = side * 0.25;
      leg.userData.side = side;
      return leg;
    });
    this.midLegs = [-1, 1].map((side) => {
      const leg = mkLeg(side * L * 0.17, -L * 0.05, L * 0.02, L * 0.2);
      leg.rotation.z = side * 0.9;
      leg.userData.knee.rotation.z = -side * 1.3;
      return leg;
    });
    this.hindLegs = [-1, 1].map((side) => {
      const leg = mkLeg(side * L * 0.12, -L * 0.08, L * 0.2, L * 0.24);
      leg.rotation.x = -0.9;
      leg.rotation.z = side * 0.35;
      leg.userData.side = side;
      return leg;
    });
    // pedals under the hind legs: left brake, right throttle
    this.pedals = [-1, 1].map((side) => {
      const pedal = new THREE.Mesh(new THREE.BoxGeometry(L * 0.09, L * 0.02, L * 0.14), new THREE.MeshStandardMaterial({ color: side < 0 ? 0x8d2f36 : 0x3c8a52, roughness: 0.5 }));
      pedal.position.set(side * L * 0.14, -L * 0.36, L * 0.02);
      pedal.rotation.x = -0.5;
      this.group.add(pedal);
      return pedal;
    });

    // seated in the car: the car group's +x is the nose, the rig is built facing -z
    this.group.rotation.y = -Math.PI / 2;
    this.group.position.set(style.fly_seat_x_m, style.fly_seat_y_m, 0);
    this.group.traverse((node) => {
      if (node.isMesh) {
        node.castShadow = true;
        node.receiveShadow = true;
      }
    });
  }

  /** Animate from a live frame; `dt` in seconds. */
  update(frame, dt, maxHz, gripMaxG) {
    if (!frame) return;
    const st = this.style;
    const k = 1 - Math.exp(-dt * 10);
    // head looks into the turn; body rolls with lateral load, pitches with the pedal
    this.headYaw = lerp(this.headYaw, clamp(frame.steer, -1, 1) * 0.7, k);
    this.head.rotation.y = this.headYaw;
    const rollTarget = -Math.sign(frame.steer_actual || 0) * clamp(frame.lat_g / (gripMaxG || 4.5), 0, 1) * 0.35;
    this.roll = lerp(this.roll, rollTarget, k);
    const pitchTarget = frame.pedal < 0 ? 0.18 * -frame.pedal : -0.08 * frame.pedal;
    this.pitch = lerp(this.pitch, pitchTarget, k);
    this.body.rotation.z = this.roll;
    this.body.rotation.x = this.pitch;
    // the wheel turns with the command (left = counter-clockwise on screen); front legs follow
    this.wheelTurn = lerp(this.wheelTurn, clamp(frame.steer, -1, 1) * 1.3, k);
    this.wheel.rotation.z = this.wheelTurn;
    this.frontLegs.forEach((leg) => {
      leg.rotation.z = leg.userData.side * 0.25 + this.wheelTurn * 0.35;
      leg.userData.knee.rotation.x = 0.9 + this.wheelTurn * leg.userData.side * 0.3;
    });
    // hind legs press the pedals: left brake, right throttle
    const press = [clamp(-frame.pedal, 0, 1), clamp(frame.pedal, 0, 1)];
    this.pedalPress = this.pedalPress.map((v, i) => lerp(v, press[i], k));
    this.hindLegs.forEach((leg, i) => {
      leg.rotation.x = -0.9 - this.pedalPress[i] * 0.35;
      leg.userData.knee.rotation.x = 1.1 - this.pedalPress[i] * 0.5;
    });
    this.pedals.forEach((pedal, i) => {
      pedal.rotation.x = -0.5 - this.pedalPress[i] * 0.45;
    });
    // wings beat with throttle (a fly pushing), idle flutter otherwise
    const beat = 6 + 55 * this.pedalPress[1] + 8 * clamp(frame.speed / 40, 0, 1);
    this.phase += dt * beat * Math.PI * 2;
    const amp = 0.12 + 0.5 * this.pedalPress[1];
    this.wings.forEach((hinge) => {
      hinge.rotation.z = hinge.userData.side * (0.35 + Math.sin(this.phase) * amp);
    });
    // antennae swept back by airflow
    const wind = clamp(frame.speed / (frame.max_speed || 95), 0, 1);
    this.antennae.forEach((pivot) => {
      pivot.rotation.x = -0.6 + wind * 0.9;
      pivot.rotation.z = pivot.userData.side * (0.15 + wind * 0.2);
    });
    // compound eyes glow with the visual drive the eye groups receive
    const rays = frame.sensory_hz || [];
    const drive = rays.length ? rays.reduce((a, b) => a + b, 0) / rays.length / (maxHz || 300) : 0;
    this.eyeMaterial.emissiveIntensity = 0.2 + 1.6 * clamp(drive, 0, 1);
    // abdomen breathing
    this.abdomen.scale.y = 0.8 + 0.03 * Math.sin(performance.now() / 380);
    // crash: the whole fly jolts forward
    if (frame.done_reason === 1) this.body.position.z = -this.L * 0.15;
    else this.body.position.z = lerp(this.body.position.z, 0, k);
  }
}
