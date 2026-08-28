// The WebGL viewer: one scene, reused for every solid.
//
// Everything is computed in the browser from the record's own geometry.
// Nothing here is pre-baked, which is why changing style or coloring is
// instant and why the view always agrees with the data.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { buildSurface, buildEdges, buildBallAndStick, boundingRadius }
  from './geometry.js';

export const STYLES = ['solid', 'wireframe', 'ball-and-stick'];

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.set(1.35, -2.2, 0.95).normalize().multiplyScalar(4.2);
    this.camera.up.set(0, 0, 1);          // Z-up, as the database stores it

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;

    // A three-point rig echoing the studio the thumbnails are shot in:
    // a strong key, a soft fill to keep shadow sides readable, and a rim
    // to separate the silhouette from the background.
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.position.set(1.8, -1.9, 1.6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.55);
    fill.position.set(-2.4, -0.8, 0.5);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.9);
    rim.position.set(-1.2, 1.9, 1.1);
    this.scene.add(rim);

    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.style = 'solid';
    this.coloring = 'auto';
    this.spinning = true;
    this.record = null;

    this._onResize = () => this.resize();
    addEventListener('resize', this._onResize);
    this.resize();

    this.controls.addEventListener('start', () => { this.spinning = false; });

    let last = performance.now();
    const loop = (now) => {
      this._raf = requestAnimationFrame(loop);
      const dt = (now - last) / 1000;
      last = now;
      if (this.spinning) this.group.rotation.z += dt * 0.28;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this._raf = requestAnimationFrame(loop);
  }

  resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  clear() {
    for (const child of [...this.group.children]) {
      this.group.remove(child);
      child.geometry?.dispose();
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else child.material?.dispose();
    }
  }

  show(rec) {
    this.record = rec;
    this.rebuild();
  }

  setStyle(style) { this.style = style; this.rebuild(); }
  setColoring(c) { this.coloring = c; this.rebuild(); }

  rebuild() {
    if (!this.record) return;
    this.clear();
    const rec = this.record;
    // Normalise every solid to the same apparent size. Records are stored
    // at edge length 1, so a 92-face Johnson solid and a tetrahedron
    // differ hugely in bulk; without this the camera would need moving
    // for each one.
    const s = 1 / boundingRadius(rec);

    if (this.style === 'ball-and-stick') {
      const g = buildBallAndStick(rec);
      const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
        color: 0xd6d6dc, roughness: 0.45, metalness: 0.0,
        side: THREE.DoubleSide,
      }));
      this.group.add(mesh);
    } else if (this.style === 'wireframe') {
      const g = buildEdges(rec);
      this.group.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({
        color: 0x9fd0ff,
      })));
    } else {
      const g = buildSurface(rec, { coloring: this.coloring });
      // DoubleSide is not optional: 240 of the 448 solids are non-convex
      // and the star forms show their inner faces, so back-face culling
      // would punch holes in them.
      const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.5, metalness: 0.0,
        side: THREE.DoubleSide, flatShading: true,
      }));
      this.group.add(mesh);
      const e = buildEdges(rec);
      this.group.add(new THREE.LineSegments(e, new THREE.LineBasicMaterial({
        color: 0x000000, transparent: true, opacity: 0.28,
      })));
    }
    this.group.scale.setScalar(s);
  }

  resetView() {
    this.camera.position.set(1.35, -2.2, 0.95).normalize().multiplyScalar(4.2);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.spinning = true;
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    removeEventListener('resize', this._onResize);
    this.clear();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
