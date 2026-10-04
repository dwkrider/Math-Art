// The TPMS scene.
//
// A nodal surface divides space into two interpenetrating labyrinths,
// and that is the thing worth seeing, so the two sides are coloured
// differently: the material is double-sided with a darker back face,
// which means the colour you see tells you which labyrinth you are
// looking into.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { block, clipToSphere, facesOf, triangulate, boundaryLoops,
         RIM_SMOOTH_DEFAULT } from './tpms-math.js';
import { weld, orient, solidify } from './stl.js';

const VIEW_DIR = [1.5, -2.2, 1.1];
const RIM_SIDES = 8;

/** A tube swept along a polyline, with a rotation-minimising frame.
 *  The rim of a clipped TPMS curves in every direction, and a Frenet
 *  frame would spin the tube around it wherever the curve has an
 *  inflection. */
function tubeAlong(points, closed, radius, sides, out, outward = null) {
  const n = points.length;
  if (n < 2) return;
  // Lift the tube off the cut along the outward conormal, so it RESTS
  // against the edge instead of being threaded onto it. Centred on the
  // rim, half of a round tube is buried in the sheet and the sheet
  // pokes through it; lifted by its own radius, the tube touches the
  // edge and nothing else.
  if (outward) {
    points = points.map((p, i) => [p[0] + radius * outward[i][0],
                                   p[1] + radius * outward[i][1],
                                   p[2] + radius * outward[i][2]]);
  }
  const base = out.pos.length / 3;
  const T = [];
  for (let i = 0; i < n; i++) {
    const a = closed ? points[(i - 1 + n) % n] : points[Math.max(0, i - 1)];
    const b = closed ? points[(i + 1) % n] : points[Math.min(n - 1, i + 1)];
    const t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const m = Math.hypot(t[0], t[1], t[2]) || 1;
    T.push([t[0] / m, t[1] / m, t[2] / m]);
  }
  let nx, ny, nz;
  {
    const t = T[0];
    const up = Math.abs(t[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    nx = up[1] * t[2] - up[2] * t[1];
    ny = up[2] * t[0] - up[0] * t[2];
    nz = up[0] * t[1] - up[1] * t[0];
    const m = Math.hypot(nx, ny, nz) || 1;
    nx /= m; ny /= m; nz /= m;
  }
  for (let i = 0; i < n; i++) {
    const [tx, ty, tz] = T[i];
    const d = nx * tx + ny * ty + nz * tz;
    nx -= d * tx; ny -= d * ty; nz -= d * tz;
    const m = Math.hypot(nx, ny, nz) || 1;
    nx /= m; ny /= m; nz /= m;
    const bx = ty * nz - tz * ny;
    const by = tz * nx - tx * nz;
    const bz = tx * ny - ty * nx;
    for (let s = 0; s < sides; s++) {
      const a = 2 * Math.PI * s / sides;
      const ca = Math.cos(a), sa = Math.sin(a);
      const ux = nx * ca + bx * sa, uy = ny * ca + by * sa, uz = nz * ca + bz * sa;
      out.pos.push(points[i][0] + radius * ux,
                   points[i][1] + radius * uy,
                   points[i][2] + radius * uz);
      out.nor.push(ux, uy, uz);
    }
  }
  const rings = closed ? n : n - 1;
  for (let i = 0; i < rings; i++) {
    const i0 = base + i * sides;
    const i1 = base + ((i + 1) % n) * sides;
    for (let s = 0; s < sides; s++) {
      const s1 = (s + 1) % sides;
      out.idx.push(i0 + s, i1 + s, i1 + s1, i0 + s, i1 + s1, i0 + s1);
    }
  }
}
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
    this.rim = null;
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

  /** Build a block of cells.
   *
   *  Order matters and follows the generator's: march the field, clip
   *  the block to a ball if asked, take the rim the clip opened, and
   *  only then give the sheet a thickness. Thickening first would wall
   *  the rim shut and leave the tube with nothing to sit on. */
  build({ kind, cells, res, offset = 0, wireframe = false, scale = 2,
          clip = 0, thickness = 0, rim = 0, rimSmooth = RIM_SMOOTH_DEFAULT }) {
    const t0 = performance.now();
    const mesh = block(kind, cells, res, scale, offset);
    let positions = mesh.positions;
    let indices = mesh.indices;
    let faces = null;
    let loops = [];
    let clippedAway = false;

    // 1 is a real radius, not "off": the ball then has the block's own
    // half-width and still bites its corners away. Only 0 is off.
    if (clip > 0) {
      let lo = [Infinity, Infinity, Infinity];
      let hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < positions.length; i += 3) {
        for (let k = 0; k < 3; k++) {
          if (positions[i + k] < lo[k]) lo[k] = positions[i + k];
          if (positions[i + k] > hi[k]) hi[k] = positions[i + k];
        }
      }
      // a FRACTION of the block's own half-extent, so it keeps its
      // meaning when the cell count changes
      const half = 0.5 * Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
      const out = clipToSphere(positions, facesOf(indices), clip * half);
      if (out.faces.length) {
        positions = out.positions;
        faces = out.faces;
        indices = triangulate(out.faces);
      } else {
        clippedAway = true;              // the ball missed the surface
      }
    }

    if (rim > 0) {
      loops = boundaryLoops(positions, faces || facesOf(indices), rimSmooth);
    }

    let solid = false;
    if (thickness > 0) {
      const merged = weld(positions, indices);
      const facing = orient(merged.positions, merged.indices);
      const built = solidify(merged.positions, facing.indices, thickness);
      positions = built.positions;
      indices = built.indices;
      solid = true;
    }
    this._drop();

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(
      positions instanceof Float32Array ? positions : Float32Array.from(positions), 3));
    geo.setIndex(new THREE.BufferAttribute(
      indices instanceof Uint32Array ? indices : Uint32Array.from(indices), 1));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: FRONT, roughness: 0.45, metalness: 0.0,
      // a thickened block is closed, so only its outside is needed; a
      // bare sheet has to be lit from both sides
      side: solid ? THREE.FrontSide : THREE.DoubleSide, wireframe,
      // the far side of the sheet, which is the other labyrinth
      emissive: new THREE.Color(BACK),
      emissiveIntensity: 0.35,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.root.add(this.mesh);
    // the rim tube, swept along what the clip opened
    let rimGeom = null;
    if (rim > 0 && loops.length) {
      const out = { pos: [], nor: [], idx: [] };
      for (const l of loops) {
        tubeAlong(l.points, l.closed, rim, RIM_SIDES, out, l.outward);
      }
      const rgeo = new THREE.BufferGeometry();
      rgeo.setAttribute('position', new THREE.Float32BufferAttribute(out.pos, 3));
      rgeo.setAttribute('normal', new THREE.Float32BufferAttribute(out.nor, 3));
      rgeo.setIndex(out.idx);
      const rmat = new THREE.MeshStandardMaterial({
        color: 0xffb454, roughness: 0.4, metalness: 0.0,
      });
      this.rim = new THREE.Mesh(rgeo, rmat);
      this.rim.frustumCulled = false;
      this.root.add(this.rim);
      rimGeom = { pos: Float32Array.from(out.pos), idx: Uint32Array.from(out.idx) };
    }

    // What the export takes. The rim tube is part of the object, not
    // decoration on top of it: a printed TPMS with a wire round its
    // cut edge needs the wire in the file.
    if (rimGeom) {
      const nv = positions.length / 3;
      const pos = new Float32Array(positions.length + rimGeom.pos.length);
      pos.set(positions, 0);
      pos.set(rimGeom.pos, positions.length);
      const idx = new Uint32Array(indices.length + rimGeom.idx.length);
      idx.set(indices, 0);
      for (let i = 0; i < rimGeom.idx.length; i++) {
        idx[indices.length + i] = rimGeom.idx[i] + nv;
      }
      this.lastGeometry = { positions: pos, indices: idx };
    } else {
      this.lastGeometry = { positions, indices };
    }

    // How far the block reaches, so the camera can be stood back far
    // enough: one period is `scale` units wide, so a four-cell block is
    // four times the size of a one-cell block and would otherwise put
    // the camera inside it.
    let extent = 0;
    for (let i = 0; i < positions.length; i += 3) {
      const r = Math.hypot(positions[i], positions[i + 1], positions[i + 2]);
      if (r > extent) extent = r;
    }

    return {
      extent,
      vertices: positions.length / 3,
      triangles: indices.length / 3,
      cells: mesh.cells,
      solid,
      clippedAway,
      rimLoops: loops.length,
      rimPoints: loops.reduce((a, l) => a + l.points.length, 0),
      buildMs: performance.now() - t0,
    };
  }

  _drop() {
    for (const m of [this.mesh, this.rim]) {
      if (!m) continue;
      this.root.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    this.mesh = null;
    this.rim = null;
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
