// The Chair44 patch as a WebGL scene.
//
// One indexed mesh for the whole patch. Every chair is the SAME solid
// in one of 24 cubic rotations at an integer translation, so the tile
// is built once by chair44-math.js and only the pose is repeated --
// which is what makes a 512-chair patch of the featured tile (1.1
// million vertices) something a browser will still build in a second.
//
// Shading is flat, and deliberately: the panels and the little
// pyramids are flat, and smoothing them would round away the very
// features that decide which chairs may meet. `flatShading` takes the
// normal from screen-space derivatives in the fragment shader, so the
// geometry can stay indexed -- separating every triangle to carry a
// face normal would triple the memory for the same picture.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import {
  bareTileMesh, arrowTileMesh, tileMesh, triangulate, patch, contacts,
  inAtlas, frameIndex, matApply, CENTROID, ETA_SHOWN, ETA_TRUE,
  HEIGHT_TRUE,
} from './chair44-math.js';

const VIEW_DIR = [1.5, -2.3, 1.15];

// The add-on's own palette, so a patch here and the same patch built in
// Blender are coloured alike: a golden-angle hue wheel at s=0.70,
// v=0.86 (spacefill_generator._wheel).
function wheel(n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(hsv((i * 0.618033988749895) % 1.0, 0.70, 0.86));
  return out;
}

function hsv(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

// The three arrow colours, as the add-on sets them.
const ARROW_RGB = [[0.11, 0.21, 0.52], [0.52, 0.78, 0.24], [0.85, 0.15, 0.14]];
const PLAIN_RGB = [0.88, 0.88, 0.86];

// sRGB is what the add-on's material colours are given in; three.js
// wants linear working colour, so convert rather than let the numbers
// mean two different things in the two pictures.
function toLinear(c) {
  return c.map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
}

/** The tile for a feature mode, as {verts, tris, colors} where colors
 *  is null (take the chair's own colour) or one RGB per vertex. */
export function chairTile(featureMode, relief) {
  if (featureMode === 'NONE') {
    const { verts, faces } = bareTileMesh();
    return { verts, tris: triangulate(faces), arrowOf: null };
  }
  if (featureMode === 'ARROWS') {
    const { verts, faces, colors } = arrowTileMesh();
    // carry each face's arrow colour onto its vertices; body and arrow
    // never share a vertex, so this is unambiguous
    const arrowOf = new Int8Array(verts.length).fill(-1);
    faces.forEach((f, i) => {
      if (colors[i] === null) return;
      for (const v of f) arrowOf[v] = colors[i];
    });
    return { verts, tris: triangulate(faces), arrowOf };
  }
  const { verts, faces } = featureMode === 'TRUE'
    ? tileMesh(ETA_TRUE, HEIGHT_TRUE)
    : tileMesh(ETA_SHOWN, relief * HEIGHT_TRUE);
  return { verts, tris: triangulate(faces), arrowOf: null };
}

export class ChairView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.6);
    this.camera.up.set(0, 0, 1);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.4;
    this.controls.maxDistance = 12;

    // The same three-point rig as the other modules.
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

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.mesh = null;
    this.spin = 0;

    this._onResize = () => this.resize();
    addEventListener('resize', this._onResize);
    this.resize();

    let last = performance.now();
    const loop = (now) => {
      this._raf = requestAnimationFrame(loop);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (this.spin) this.root.rotation.z += this.spin * dt;
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

  /** Build the patch. Returns what the page reports to the reader. */
  build({ depth, features, relief, gap, colorBy }) {
    const t0 = performance.now();
    const tile = chairTile(features, relief);
    const poses = patch(depth);
    const n = poses.length;
    const nv = tile.verts.length;
    const nt = tile.tris.length;

    const pos = new Float32Array(n * nv * 3);
    const col = new Float32Array(n * nv * 3);
    const idx = (n * nv > 65535 ? new Uint32Array(n * nt * 3)
                                : new Uint16Array(n * nt * 3));

    const slots = colorBy === 'FRAME' ? 24 : colorBy === 'PARENT' ? 8 : 1;
    const palette = (colorBy === 'NONE' ? [PLAIN_RGB] : wheel(slots)).map(toLinear);
    const arrowRGB = ARROW_RGB.map(toLinear);

    // rotate the tile once per distinct frame, not once per chair
    const rotated = new Map();
    let lo = [Infinity, Infinity, Infinity];
    let hi = [-Infinity, -Infinity, -Infinity];

    poses.forEach((pose, ci) => {
      const kf = pose.G.flat().join(',');
      let R = rotated.get(kf);
      if (R === undefined) {
        R = tile.verts.map((p) => matApply(pose.G, [p[0] - CENTROID[0],
                                                    p[1] - CENTROID[1],
                                                    p[2] - CENTROID[2]]));
        rotated.set(kf, R);
      }
      const c = matApply(pose.G, CENTROID);
      const cx = c[0] + pose.t[0], cy = c[1] + pose.t[1], cz = c[2] + pose.t[2];
      const tag = colorBy === 'FRAME' ? frameIndex(pose.G)
        : colorBy === 'PARENT' ? pose.group : 0;
      const base = palette[tag % palette.length];
      const o = ci * nv;
      for (let i = 0; i < nv; i++) {
        // gap shrinks each chair about its OWN centroid, so the pack
        // reads as separate solids (the add-on's Gap Factor)
        const x = cx + R[i][0] * gap, y = cy + R[i][1] * gap, z = cz + R[i][2] * gap;
        const k = (o + i) * 3;
        pos[k] = x; pos[k + 1] = y; pos[k + 2] = z;
        const rgb = tile.arrowOf && tile.arrowOf[i] >= 0
          ? arrowRGB[tile.arrowOf[i]] : base;
        col[k] = rgb[0]; col[k + 1] = rgb[1]; col[k + 2] = rgb[2];
        if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
        if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
        if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
      }
      const io = ci * nt * 3;
      for (let f = 0; f < nt; f++) {
        idx[io + f * 3] = o + tile.tris[f][0];
        idx[io + f * 3 + 1] = o + tile.tris[f][1];
        idx[io + f * 3 + 2] = o + tile.tris[f][2];
      }
    });

    // one fit over the whole patch, centred and scaled into a 2 m cube
    const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    const k = 2 / span;
    for (let i = 0; i < pos.length; i += 3) {
      pos[i] = (pos[i] - mid[0]) * k;
      pos[i + 1] = (pos[i + 1] - mid[1]) * k;
      pos[i + 2] = (pos[i + 2] - mid[2]) * k;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, flatShading: true, roughness: 0.5, metalness: 0.0,
      side: THREE.DoubleSide,
    });

    this._drop();
    this.mesh = new THREE.Mesh(geo, mat);
    this.root.add(this.mesh);

    return {
      chairs: n,
      vertices: n * nv,
      triangles: n * nt,
      buildMs: performance.now() - t0,
    };
  }

  /** The contact report: how many face contacts the patch has, and
   *  whether every one is in the 44-contact atlas. */
  static report(depth) {
    const cs = contacts(patch(depth));
    let inside = 0;
    for (const c of cs) if (inAtlas(c.G, c.t)) inside++;
    return { total: cs.length, inside };
  }

  _drop() {
    if (!this.mesh) return;
    this.root.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh = null;
  }

  resetView() {
    this.root.rotation.z = 0;
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.6);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }
}
