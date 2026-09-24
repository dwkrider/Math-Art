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
  bareTileMesh, arrowTileMesh, tileMesh, triangulate, patch, walkPatch,
  contacts, inAtlas, frameIndex, matApply, CENTROID, ETA_SHOWN, ETA_TRUE,
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

// WHAT LIMITS A PATCH. Not memory any more: the tile is shared, so a
// chair costs one 4x4 matrix and a colour, 76 bytes, and even two
// million of them is a few hundred megabytes. What limits it is the
// triangles the GPU redraws every frame while you orbit. Twenty
// million is about the most that stays fluid on ordinary hardware, so
// that is the budget the fallback aims at -- and the deepest patch of
// all overruns it knowingly, with a warning, because at that size the
// Built slider is also the throttle.
export const MAX_TRIANGLES = 20e6;

const TILE_CACHE = new Map();

// Cheapest first: the order to fall back along when a patch is too big.
export const FEATURE_ORDER = ['NONE', 'ARROWS', 'TRUE', 'EXAGGERATED'];

/** Triangles in one tile, by feature mode -- built once and cached, so
 *  the cost of a patch is known without building it. */
export function tileTriangles(featureMode) {
  return chairTile(featureMode, 30).tris.length;
}

/** How to draw the rule at this depth: the most detailed mode that
 *  fits the triangle budget, starting from the one asked for, and
 *  whether even that overruns it. */
export function planFor(depth, wanted) {
  const chairs = 8 ** depth;
  let best = null;
  for (const mode of FEATURE_ORDER) {
    if (chairs * tileTriangles(mode) <= MAX_TRIANGLES) best = mode;
    if (mode === wanted) break;
  }
  if (best !== null) {
    return { mode: best, fellBack: best !== wanted, heavy: false,
             triangles: chairs * tileTriangles(best) };
  }
  // Nothing fits: draw the bare chair anyway and say what it costs.
  return { mode: 'NONE', fellBack: wanted !== 'NONE', heavy: true,
           triangles: chairs * tileTriangles('NONE') };
}

/** The tile for a feature mode, as {verts, tris, colors} where colors
 *  is null (take the chair's own colour) or one RGB per vertex. */
export function chairTile(featureMode, relief) {
  // The featured tile is the expensive one to build (2,138 vertices
  // welded out of a 9x9 grid on each of 24 panels), and the page
  // rebuilds on every slider nudge, so keep the last few.
  const key = `${featureMode}:${featureMode === 'EXAGGERATED' ? relief : 0}`;
  const hit = TILE_CACHE.get(key);
  if (hit) return hit;
  const built = buildChairTile(featureMode, relief);
  if (TILE_CACHE.size > 8) TILE_CACHE.clear();
  TILE_CACHE.set(key, built);
  return built;
}

function buildChairTile(featureMode, relief) {
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


/** One geometry from the faces the filter keeps, welded to just the
 *  vertices they use, centroid-relative. `colorOf` bakes the arrow
 *  colours in as vertex colours. */
function geometryFor(tile, keepVertex, colorOf) {
  const remap = new Int32Array(tile.verts.length).fill(-1);
  const pos = [];
  const col = [];
  const idx = [];
  for (const tri of tile.tris) {
    if (!tri.every((v) => keepVertex(v))) continue;
    for (const v of tri) {
      if (remap[v] < 0) {
        remap[v] = pos.length / 3;
        pos.push(tile.verts[v][0] - CENTROID[0],
                 tile.verts[v][1] - CENTROID[1],
                 tile.verts[v][2] - CENTROID[2]);
        if (colorOf) {
          const rgb = toLinear(ARROW_RGB[colorOf[v]]);
          col.push(rgb[0], rgb[1], rgb[2]);
        }
      }
      idx.push(remap[v]);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  if (colorOf) geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

/** The eight corners of the tile's bounding box, centroid-relative. */
function tileBounds(verts) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const p of verts) {
    for (let a = 0; a < 3; a++) {
      const v = p[a] - CENTROID[a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  }
  const out = [];
  for (const x of [lo[0], hi[0]]) {
    for (const y of [lo[1], hi[1]]) {
      for (const z of [lo[2], hi[2]]) out.push([x, y, z]);
    }
  }
  return out;
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
    // `fit` carries the centring and scaling of the whole patch; `root`
    // carries the spin, so one does not undo the other.
    this.fit = new THREE.Group();
    this.root.add(this.fit);
    this.meshes = [];
    this.chairCount = 0;
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
    const n = 8 ** depth;
    const nt = tile.tris.length;

    const slots = colorBy === 'FRAME' ? 24 : colorBy === 'PARENT' ? 8 : 1;
    const palette = (colorBy === 'NONE' ? [PLAIN_RGB] : wheel(slots)).map(toLinear);

    // The tile is shared: every chair is the same solid, so it goes to
    // the GPU once and each chair is a 4x4 matrix. Bodies and arrows
    // are two meshes over the SAME instance matrices, because a body
    // takes its colour per chair and an arrow takes it per vertex, and
    // three.js multiplies the two when a mesh carries both.
    const body = geometryFor(tile, (i) => tile.arrowOf === null || tile.arrowOf[i] < 0);
    const arrows = tile.arrowOf === null ? null
      : geometryFor(tile, (i) => tile.arrowOf[i] >= 0, tile.arrowOf);

    // The meshes come first so the walk can write straight into their
    // instance buffers: at two million chairs a staging copy of the
    // matrices would be another 134 MB for nothing.
    this._drop();
    const mat = new THREE.MeshStandardMaterial({
      flatShading: true, roughness: 0.5, metalness: 0.0, side: THREE.DoubleSide,
    });
    const bodyMesh = new THREE.InstancedMesh(body, mat, n);
    bodyMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    bodyMesh.frustumCulled = false;
    const mats = bodyMesh.instanceMatrix.array;
    const cols = bodyMesh.instanceColor.array;

    // The tile's own bounding box, centroid-relative: a signed
    // permutation maps a box to a box, so transforming these eight
    // corners gives each chair's exact extent without touching its
    // vertices.
    const bb = tileBounds(tile.verts);
    let lo = [Infinity, Infinity, Infinity];
    let hi = [-Infinity, -Infinity, -Infinity];

    walkPatch(depth, (G, t, group, ci) => {
      const c = matApply(G, CENTROID);
      const cx = c[0] + t[0], cy = c[1] + t[1], cz = c[2] + t[2];
      // column-major, as three.js stores Matrix4: the rotation scaled
      // by the gap, then the chair's centre. `gap` shrinks each chair
      // about its OWN centroid, which is the add-on's Gap Factor.
      const m = ci * 16;
      mats[m] = G[0][0] * gap; mats[m + 1] = G[1][0] * gap; mats[m + 2] = G[2][0] * gap;
      mats[m + 4] = G[0][1] * gap; mats[m + 5] = G[1][1] * gap; mats[m + 6] = G[2][1] * gap;
      mats[m + 8] = G[0][2] * gap; mats[m + 9] = G[1][2] * gap; mats[m + 10] = G[2][2] * gap;
      mats[m + 12] = cx; mats[m + 13] = cy; mats[m + 14] = cz;
      mats[m + 15] = 1;

      const tag = colorBy === 'FRAME' ? frameIndex(G) : colorBy === 'PARENT' ? group : 0;
      const rgb = palette[tag % palette.length];
      cols[ci * 3] = rgb[0]; cols[ci * 3 + 1] = rgb[1]; cols[ci * 3 + 2] = rgb[2];

      for (let k = 0; k < 8; k++) {
        const q = matApply(G, bb[k]);
        for (let a = 0; a < 3; a++) {
          const v = [cx, cy, cz][a] + q[a] * gap;
          if (v < lo[a]) lo[a] = v;
          if (v > hi[a]) hi[a] = v;
        }
      }
    });

    bodyMesh.instanceMatrix.needsUpdate = true;
    bodyMesh.instanceColor.needsUpdate = true;
    this.meshes = [bodyMesh];

    if (arrows) {
      const arrowMat = new THREE.MeshStandardMaterial({
        vertexColors: true, flatShading: true, roughness: 0.5, metalness: 0.0,
        side: THREE.DoubleSide,
      });
      const arrowMesh = new THREE.InstancedMesh(arrows, arrowMat, n);
      // the same poses, shared rather than copied
      arrowMesh.instanceMatrix = bodyMesh.instanceMatrix;
      arrowMesh.frustumCulled = false;
      this.meshes.push(arrowMesh);
    }

    // one fit over the whole patch, on the group rather than baked into
    // vertices -- with instancing there are no per-patch vertices left
    // to bake it into
    const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    const k = 2 / span;
    this.fit.scale.setScalar(k);
    this.fit.position.set(-mid[0] * k, -mid[1] * k, -mid[2] * k);
    for (const m of this.meshes) this.fit.add(m);

    this.chairCount = n;
    this.setShown(n);

    return {
      chairs: n,
      vertices: n * tile.verts.length,
      triangles: n * nt,
      buildMs: performance.now() - t0,
    };
  }

  /** Draw only the first `k` chairs of the patch. Instances are in the
   *  order the substitution made them, so this is the build order. */
  setShown(k) {
    const n = Math.max(0, Math.min(this.chairCount, Math.round(k)));
    for (const m of this.meshes) m.count = n;
    this.shown = n;
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
    for (const m of this.meshes) {
      this.fit.remove(m);
      m.geometry.dispose();
      m.material.dispose();
      m.dispose();
    }
    this.meshes = [];
  }

  resetView() {
    this.root.rotation.z = 0;
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.6);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }
}
