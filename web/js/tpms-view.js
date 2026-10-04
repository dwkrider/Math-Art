// The TPMS scene.
//
// A nodal surface divides space into two interpenetrating labyrinths,
// and that is the thing worth seeing, so the two sides are coloured
// differently: the material is double-sided with a darker back face,
// which means the colour you see tells you which labyrinth you are
// looking into.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { block } from './tpms-math.js';

const VIEW_DIR = [1.5, -2.2, 1.1];
const FRONT = 0x6fb3f2;        // the site's accent, for the near side
const BACK = 0x1d3448;

export class TpmsView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.6);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 20;
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld(true);
    this.controls.update();

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const key = new THREE.DirectionalLight(0xffffff, 1.7);
    key.position.set(1.8, -1.9, 1.6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.7);
    fill.position.set(-2.4, -0.8, 0.5);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.8);
    rim.position.set(-1.2, 1.9, 1.1);
    this.scene.add(rim);
    const under = new THREE.DirectionalLight(0xffffff, 0.5);
    under.position.set(0.2, 0.5, -1.8);
    this.scene.add(under);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.mesh = null;
    this.spin = 0;

    addEventListener('resize', () => this.resize());
    this.resize();
    let t = performance.now();
    const loop = (now) => {
      requestAnimationFrame(loop);
      const dt = Math.min(0.1, (now - t) / 1000);
      t = now;
      if (this.spin) this.root.rotation.z += this.spin * dt;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Build a block of cells. */
  build({ kind, cells, res, offset = 0, wireframe = false, scale = 2 }) {
    const t0 = performance.now();
    const mesh = block(kind, cells, res, scale, offset);
    this._drop();

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
    geo.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: FRONT, roughness: 0.45, metalness: 0.0,
      side: THREE.DoubleSide, wireframe,
      // the far side of the sheet, which is the other labyrinth
      emissive: new THREE.Color(BACK),
      emissiveIntensity: 0.35,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.root.add(this.mesh);
    this.lastGeometry = { positions: mesh.positions, indices: mesh.indices };

    // How far the block reaches, so the camera can be stood back far
    // enough: one period is `scale` units wide, so a four-cell block is
    // four times the size of a one-cell block and would otherwise put
    // the camera inside it.
    let extent = 0;
    for (let i = 0; i < mesh.positions.length; i += 3) {
      const r = Math.hypot(mesh.positions[i], mesh.positions[i + 1],
                           mesh.positions[i + 2]);
      if (r > extent) extent = r;
    }

    return {
      extent,
      vertices: mesh.positions.length / 3,
      triangles: mesh.indices.length / 3,
      cells: mesh.cells,
      buildMs: performance.now() - t0,
    };
  }

  _drop() {
    if (!this.mesh) return;
    this.root.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh = null;
  }

  /** Stand back far enough to see the whole block. */
  frame(extent, margin = 1.25) {
    const half = Math.tan(this.camera.fov * Math.PI / 360);
    const d = Math.max(1.2, Math.min(40, extent * margin / half));
    this.camera.position.normalize().multiplyScalar(d);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.framedAt = extent;
  }

  resetView() {
    this.root.rotation.set(0, 0, 0);
    this.camera.position.set(...VIEW_DIR).normalize();
    this.controls.target.set(0, 0, 0);
    this.frame(this.framedAt || 2);
  }
}
