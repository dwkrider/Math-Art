// The Scherk-Collins sculpture's WebGL scene.
//
// The surface is a sheet with a boundary, so it is drawn double-sided
// and lit from both -- a one-sided material would make half of every
// saddle disappear as you turn it. Smooth normals are computed per
// vertex from the triangles: the mid-surface is smooth everywhere
// except at the flange edges, and those are boundaries rather than
// creases, so nothing needs splitting.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { surfaceMesh, wallThickness } from './scherk-math.js';
import { weld, orient, solidify } from './stl.js';

const VIEW_DIR = [1.4, -2.3, 1.0];

// Carved wood, roughly: the sculptures these come from are Brent
// Collins' wood carvings, and a warm matte reads better than the
// studio grey against a dark page.
const FRONT = 0xd9a066;
const BACK = 0x8a5a33;

export class ScherkView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.4);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 14;
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld(true);
    this.controls.update();

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
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
    this.last = null;

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

  /** Build the sculpture for a parameter set.
   *
   *  With a thickness the sheet is given a wall, by the same
   *  solidifier the STL export uses -- so what is on screen is what
   *  comes out of the file, rather than two different approximations
   *  of the same sculpture. Welding first matters: each saddle patch
   *  is built separately and meets its neighbours along shared edges,
   *  and offsetting without welding would open a crack down every
   *  one of them. */
  build(p, { wireframe = false } = {}) {
    const t0 = performance.now();
    const mesh = surfaceMesh(p);
    const wall = wallThickness(p) * mesh.factor;
    let positions = mesh.positions;
    let indices = mesh.indices;
    let solid = false;
    let orientable = true;
    let misWound = 0;
    if (wall > 1e-6) {
      const merged = weld(positions, indices);
      // The saddle patches are wound every which way until they are
      // welded together and oriented; without this the offset faces
      // one way here and the other way there, and the surface goes
      // dark in patches.
      const facing = orient(merged.positions, merged.indices);
      orientable = facing.orientable;
      misWound = facing.flipped;
      const built = solidify(merged.positions, facing.indices, wall);
      positions = built.positions;
      indices = built.indices;
      solid = true;
    }
    this._drop();

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(
      indices.BYTES_PER_ELEMENT === 4 ? indices : new Uint32Array(indices), 1));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: FRONT, roughness: 0.62, metalness: 0.0,
      // A closed solid only needs its outside drawn. A bare sheet has
      // to be lit from both, or half of every saddle vanishes -- and
      // so does a solid with no consistent outside, where culling
      // would punch holes wherever the winding disagrees.
      side: solid && orientable ? THREE.FrontSide : THREE.DoubleSide,
      wireframe,
      emissive: new THREE.Color(BACK).multiplyScalar(solid ? 0.04 : 0.12),
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.root.add(this.mesh);
    this.last = mesh;

    this.lastGeometry = { positions, indices };
    return {
      vertices: positions.length / 3,
      triangles: indices.length / 3,
      patches: mesh.patches,
      closes: mesh.closes,
      solid,
      wall,
      orientable,
      misWound,
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

  resetView() {
    this.root.rotation.set(0, 0, 0);
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.4);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }
}
