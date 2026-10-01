// The warm-up figure: three fibre bundles simple enough to see whole.
//
// Base space, fibre, total space -- the vocabulary the Hopf fibration
// needs, on examples where nothing is hidden. A disc with a line over
// every point is a cylinder; a circle with a circle over every point
// is a torus; and a circle with a segment over every point is either a
// band or, if the segments turn as you go round, a Mobius strip. The
// last one is the point: same base, same fibre, different total space,
// because of how the fibres are glued rather than what they are.
//
// The fibres are drawn one at a time, spaced out, because that is what
// makes them fibres rather than a surface. A real bundle has one over
// every point of the base, infinitely many, infinitely thin.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

export const BUNDLES = {
  CYLINDER: {
    label: 'Disc × line = cylinder',
    base: 'a disc', fibre: 'a line', total: 'a cylinder',
    note: 'Trivial: the total space is just the base times the fibre, '
        + 'and nothing is twisted.',
  },
  TORUS: {
    label: 'Circle × circle = torus',
    base: 'a circle', fibre: 'a circle', total: 'a torus',
    note: 'Also trivial, and also a surface you already know. The Hopf '
        + 'fibration has circles over a sphere instead — and is not '
        + 'trivial.',
  },
  MOBIUS: {
    label: 'Circle × segment, with a twist',
    base: 'a circle', fibre: 'a segment', total: 'a Möbius strip',
    note: 'Nontrivial: the fibres are glued with a half turn, so the '
        + 'total space is not the base times the fibre. This is the '
        + 'smallest example of the twisting the Hopf fibration has.',
  },
};

const FIB = 0x7fd4ff;      // fibres
const BASE = 0xffb454;     // the base space

export class BundleView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
    this.camera.position.set(3.4, -4.6, 2.9);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.minDistance = 2.5;
    this.controls.maxDistance = 12;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xffffff, 1.9);
    key.position.set(1.6, -2.0, 1.8);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-2.0, -0.6, 0.4);
    this.scene.add(fill);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    addEventListener('resize', () => this.resize());
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
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

  _clear() {
    for (const o of [...this.root.children]) {
      this.root.remove(o);
      o.geometry?.dispose();
      o.material?.dispose();
    }
  }

  /** `t` runs 0 to 1: how much of each fibre has grown. Growing them
   *  is a lie -- the fibres are all there at once -- but it is the
   *  clearest way to show that the total space is made of them. */
  show(kind, t = 1) {
    this._clear();
    const fibMat = new THREE.MeshStandardMaterial({ color: FIB, roughness: 0.4 });
    const baseMat = new THREE.MeshStandardMaterial({
      color: BASE, roughness: 0.6, side: THREE.DoubleSide });
    const grow = Math.max(0.001, t);

    if (kind === 'CYLINDER') {
      const disc = new THREE.Mesh(new THREE.CircleGeometry(1, 64), baseMat);
      disc.position.z = -1.0;
      this.root.add(disc);
      const rings = 5, per = 16;
      for (let r = 1; r <= rings; r++) {
        const rad = r / rings;
        for (let k = 0; k < per; k++) {
          const a = 2 * Math.PI * k / per + (r % 2) * Math.PI / per;
          const g = new THREE.CylinderGeometry(0.012, 0.012, 2 * grow, 6);
          const m = new THREE.Mesh(g, fibMat);
          m.rotation.x = Math.PI / 2;
          m.position.set(rad * Math.cos(a), rad * Math.sin(a), -1 + grow);
          this.root.add(m);
        }
      }
      return;
    }

    if (kind === 'TORUS') {
      const R = 1.0, r = 0.42;
      const ring = new THREE.Mesh(new THREE.TorusGeometry(R, 0.02, 8, 96), baseMat);
      this.root.add(ring);
      const per = 28;
      for (let k = 0; k < per; k++) {
        const a = 2 * Math.PI * k / per;
        const g = new THREE.TorusGeometry(r, 0.022, 7, 48, 2 * Math.PI * grow);
        const m = new THREE.Mesh(g, fibMat);
        m.position.set(R * Math.cos(a), R * Math.sin(a), 0);
        m.rotation.set(Math.PI / 2, 0, a + Math.PI / 2);
        this.root.add(m);
      }
      return;
    }

    // MOBIUS: the same circle base, a segment over each point, turned
    // by half a turn over the whole loop -- so the strip has one side.
    // The fibres are placed from the strip's own parametrisation, and
    // the strip itself is drawn faintly behind them: without it the
    // segments read as spokes and the twist is invisible, which is the
    // one thing this example exists to show.
    const R = 1.0, half = 0.42;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(R, 0.02, 8, 96), baseMat);
    this.root.add(ring);

    const at = (u, v) => new THREE.Vector3(
      (R + v * Math.cos(u / 2)) * Math.cos(u),
      (R + v * Math.cos(u / 2)) * Math.sin(u),
      v * Math.sin(u / 2));

    const NU = 160, NV = 8;
    const pos = [], idx = [];
    for (let i = 0; i <= NU; i++) {
      const u = 2 * Math.PI * i / NU;
      for (let j = 0; j <= NV; j++) {
        const v = -half + 2 * half * j / NV;
        const p = at(u, v * grow);
        pos.push(p.x, p.y, p.z);
      }
    }
    for (let i = 0; i < NU; i++) {
      for (let j = 0; j < NV; j++) {
        const a = i * (NV + 1) + j, b = (i + 1) * (NV + 1) + j;
        idx.push(a, b, b + 1, a, b + 1, a + 1);
      }
    }
    const strip = new THREE.BufferGeometry();
    strip.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    strip.setIndex(idx);
    strip.computeVertexNormals();
    this.root.add(new THREE.Mesh(strip, new THREE.MeshStandardMaterial({
      color: FIB, roughness: 0.5, side: THREE.DoubleSide,
      transparent: true, opacity: 0.22, depthWrite: false,
    })));

    const per = 36;
    for (let k = 0; k < per; k++) {
      const u = 2 * Math.PI * k / per;
      const a = at(u, -half * grow), b = at(u, half * grow);
      const mid = a.clone().add(b).multiplyScalar(0.5);
      const len = a.distanceTo(b) || 0.001;
      const g = new THREE.CylinderGeometry(0.013, 0.013, len, 6);
      const m = new THREE.Mesh(g, fibMat);
      m.position.copy(mid);
      // the cylinder runs along its own +y, so point that at the fibre
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0),
                                      b.clone().sub(a).normalize());
      this.root.add(m);
    }
  }
}
