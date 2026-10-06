// The rendering layer: a three.js viewer, and nothing else.
//
// This file knows about cameras, lights, materials and buffers. It does
// NOT know what a Seifert surface is, what a slider is, or how any
// particular page is laid out -- it is handed plain triangle arrays and
// puts them on screen. Every module's viewer was a copy of this with
// its own colours; the copies are what this replaces.
//
// THE INTERCHANGE FORMAT between computation and rendering is a plain
// object -- { positions: Float32Array, indices: Uint32Array, normals? }
// -- with no three.js types in it. That is what lets the computation
// layer run in node, against the Python engine, with no browser.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

/** Wrap plain arrays as a three.js geometry. Vertex normals are
 *  computed only when the caller did not supply them -- a swept tube
 *  knows its own normals exactly, and recomputing them from the
 *  triangles would round off its creases. */
export function toGeometry({ positions, indices, normals }) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(
    positions instanceof Float32Array ? positions : Float32Array.from(positions), 3));
  if (indices && indices.length) {
    geo.setIndex(new THREE.BufferAttribute(
      indices instanceof Uint32Array ? indices : Uint32Array.from(indices), 1));
  }
  if (normals) {
    geo.setAttribute('normal', new THREE.BufferAttribute(
      normals instanceof Float32Array ? normals : Float32Array.from(normals), 3));
  } else {
    geo.computeVertexNormals();
  }
  return geo;
}

const DEFAULTS = {
  viewDir: [1.5, -2.2, 1.1],
  fov: 36,
  up: [0, 0, 1],
  distance: 4.6,
  minDistance: 1.2,
  maxDistance: 40,
  spinRate: 0.25,
};

/** A lit, orbitable stage holding one set of parts at a time. */
export class Stage {
  constructor(canvas, options = {}) {
    const o = { ...DEFAULTS, ...options };
    this.options = o;
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(o.fov, 1, 0.01, 200);
    this.camera.position.set(...o.viewDir).normalize().multiplyScalar(o.distance);
    this.camera.up.set(...o.up);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = o.minDistance;
    this.controls.maxDistance = o.maxDistance;
    // Aim before the first frame, not during it: raycast picking against
    // an unaimed camera silently misses, which cost an afternoon on the
    // Hopf page.
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld(true);
    this.controls.update();

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    for (const [intensity, at] of [[1.7, [1.8, -1.9, 1.6]],
                                   [0.7, [-2.4, -0.8, 0.5]],
                                   [0.8, [-1.2, 1.9, 1.1]],
                                   [0.5, [0.2, 0.5, -1.8]]]) {
      const light = new THREE.DirectionalLight(0xffffff, intensity);
      light.position.set(...at);
      this.scene.add(light);
    }

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.parts = [];
    this._spin = 0;
    this.framedAt = null;

    addEventListener('resize', () => this.resize());
    this.resize();
    let last = performance.now();
    const loop = (now) => {
      this._frame = requestAnimationFrame(loop);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (this._spin) this.root.rotation.z += this._spin * dt;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this._frame = requestAnimationFrame(loop);
  }

  get spin() { return this._spin; }
  set spin(v) { this._spin = v; }
  toggleSpin() {
    this._spin = this._spin ? 0 : this.options.spinRate;
    return this._spin !== 0;
  }

  resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Replace everything on the stage.
   *
   *  Each part is { positions, indices, normals?, color, side,
   *  wireframe, emissive, emissiveIntensity, roughness, metalness }. */
  show(parts) {
    this.clear();
    for (const part of parts) {
      if (!part || !part.positions || !part.positions.length) continue;
      const mat = new THREE.MeshStandardMaterial({
        color: part.color ?? 0x6fb3f2,
        roughness: part.roughness ?? 0.45,
        metalness: part.metalness ?? 0.0,
        side: part.side === 'front' ? THREE.FrontSide : THREE.DoubleSide,
        wireframe: !!part.wireframe,
        ...(part.emissive === undefined ? {} : {
          emissive: new THREE.Color(part.emissive),
          emissiveIntensity: part.emissiveIntensity ?? 0.35,
        }),
      });
      const mesh = new THREE.Mesh(toGeometry(part), mat);
      mesh.frustumCulled = false;
      this.root.add(mesh);
      this.parts.push(mesh);
    }
  }

  clear() {
    for (const m of this.parts) {
      this.root.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    this.parts = [];
  }

  /** Stand back far enough to see something of the given extent. */
  frame(extent, margin = 1.25) {
    const half = Math.tan(this.camera.fov * Math.PI / 360);
    const d = Math.max(this.controls.minDistance,
                       Math.min(this.controls.maxDistance,
                                extent * margin / half));
    this.camera.position.normalize().multiplyScalar(d);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.framedAt = extent;
  }

  resetView() {
    this.root.rotation.set(0, 0, 0);
    this.camera.position.set(...this.options.viewDir).normalize();
    this.controls.target.set(0, 0, 0);
    this.frame(this.framedAt || 2);
  }

  dispose() {
    cancelAnimationFrame(this._frame);
    this.clear();
    this.renderer.dispose();
  }
}
