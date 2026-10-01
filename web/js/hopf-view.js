// The Hopf fibration's WebGL scene: the fibres on the left, the base
// sphere you pick them on beside it.
//
// Every fibre is a closed polyline from hopf-math.js, drawn as a tube
// so it reads as a solid loop rather than a hairline -- WebGL line
// width is 1 pixel on most platforms, and at a hundred overlapping
// circles hairlines turn to fog. Tubes for a whole picture go into one
// merged buffer with a rotation-minimising frame per fibre; the cost
// is a few hundred thousand vertices for the biggest preset, which is
// rebuilt in a few milliseconds.
//
// The base sphere is its own small scene sharing the page's colour
// convention: a dot sits where each fibre comes from, in exactly the
// colour of that fibre, so the two views read as one picture.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { buildFibers, paletteRgb, normalize3, apply3, rotMatrix, TILT } from './hopf-math.js';

const VIEW_DIR = [1.4, -2.2, 1.0];

function toLinear(c) {
  return c.map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
}

/** A tube around a polyline, written into preallocated buffers.
 *
 *  Frames are carried along the curve by rotation minimisation rather
 *  than taken from the Frenet frame: a circle's Frenet normal spins
 *  with the curvature and would twist the tube, and a Hopf fibre is a
 *  circle.
 *
 *  The buffers are sized before any of this runs and written by index.
 *  Growing plain arrays instead cost 1.3 seconds for the largest
 *  preset and made the flow stutter at 16 frames a second; the work
 *  per vertex is the same, it was all allocation.
 */
function writeTube(P, closed, radius, sides, rgb, buf, cur) {
  const n = P.length / 3;
  if (n < 2) return;
  const base = cur.v;
  const T = cur.tangents.length >= n * 3 ? cur.tangents
    : (cur.tangents = new Float64Array(n * 3));
  for (let i = 0; i < n; i++) {
    const a = closed ? (i - 1 + n) % n : Math.max(0, i - 1);
    const b = closed ? (i + 1) % n : Math.min(n - 1, i + 1);
    const tx = P[b * 3] - P[a * 3];
    const ty = P[b * 3 + 1] - P[a * 3 + 1];
    const tz = P[b * 3 + 2] - P[a * 3 + 2];
    const m = Math.hypot(tx, ty, tz) || 1;
    T[i * 3] = tx / m; T[i * 3 + 1] = ty / m; T[i * 3 + 2] = tz / m;
  }
  let nx, ny, nz;
  {
    const tx = T[0], ty = T[1], tz = T[2];
    const ux = Math.abs(tz) < 0.9 ? 0 : 1;
    const uz = Math.abs(tz) < 0.9 ? 1 : 0;
    nx = -uz * ty;
    ny = uz * tx - ux * tz;
    nz = ux * ty;
    const m = Math.hypot(nx, ny, nz) || 1;
    nx /= m; ny /= m; nz /= m;
  }
  const cosA = cur.cos, sinA = cur.sin;      // the ring, computed once
  for (let i = 0; i < n; i++) {
    const tx = T[i * 3], ty = T[i * 3 + 1], tz = T[i * 3 + 2];
    const d = nx * tx + ny * ty + nz * tz;
    nx -= d * tx; ny -= d * ty; nz -= d * tz;
    const m = Math.hypot(nx, ny, nz) || 1;
    nx /= m; ny /= m; nz /= m;
    const bx = ty * nz - tz * ny;
    const by = tz * nx - tx * nz;
    const bz = tx * ny - ty * nx;
    const px = P[i * 3], py = P[i * 3 + 1], pz = P[i * 3 + 2];
    for (let sI = 0; sI < sides; sI++) {
      const ca = cosA[sI], sa = sinA[sI];
      const ux = nx * ca + bx * sa;
      const uy = ny * ca + by * sa;
      const uz = nz * ca + bz * sa;
      const o = cur.v * 3;
      buf.pos[o] = px + radius * ux;
      buf.pos[o + 1] = py + radius * uy;
      buf.pos[o + 2] = pz + radius * uz;
      buf.nor[o] = ux; buf.nor[o + 1] = uy; buf.nor[o + 2] = uz;
      buf.col[o] = rgb[0]; buf.col[o + 1] = rgb[1]; buf.col[o + 2] = rgb[2];
      cur.v++;
    }
  }
  const rings = closed ? n : n - 1;
  for (let i = 0; i < rings; i++) {
    const i0 = base + i * sides;
    const i1 = base + ((i + 1) % n) * sides;
    for (let sI = 0; sI < sides; sI++) {
      const s1 = (sI + 1) % sides;
      buf.idx[cur.i++] = i0 + sI;
      buf.idx[cur.i++] = i1 + sI;
      buf.idx[cur.i++] = i1 + s1;
      buf.idx[cur.i++] = i0 + sI;
      buf.idx[cur.i++] = i1 + s1;
      buf.idx[cur.i++] = i0 + s1;
    }
  }
}

class Stage {
  constructor(canvas, { distance = 4.2, fov = 38, pan = false } = {}) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(fov, 1, 0.01, 100);
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(distance);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = pan;
    // Aim it now, not on the first animation frame. Until the camera
    // looks at the origin its matrices describe a camera pointing down
    // -Z from wherever it was put, and anything that casts a ray --
    // picking a point on the base sphere -- misses everything. Nobody
    // would notice in a screenshot, because by then a frame has run.
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld(true);
    this.controls.update();
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(1.8, -1.9, 1.6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-2.4, -0.8, 0.5);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.8);
    rim.position.set(-1.2, 1.9, 1.1);
    this.scene.add(rim);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.onFrame = null;
    addEventListener('resize', () => this.resize());
    this.resize();
    let last = performance.now();
    const loop = (now) => {
      requestAnimationFrame(loop);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (this.onFrame) this.onFrame(dt);
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
}

/** The fibres. */
export class HopfView extends Stage {
  constructor(canvas) {
    super(canvas, { distance: 4.2 });
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 20;
    this.mesh = null;
    this.spin = 0;
  }

  /** Draw a set of fibres. `opts` goes straight to buildFibers, plus
   *  `palette`, `radius` and `sides` for how they are drawn. */
  build(opts) {
    const t0 = performance.now();
    const { palette = 'RAINBOW', radius = 0.02, sides = 8, ...rest } = opts;
    const built = buildFibers(rest);
    this.canvas.dataset.dbg = `s3Rot=${rest.s3Rot} keys=${Object.keys(rest).join(',')}`
      + ` p0=${built.fibers[0] ? built.fibers[0][0].toFixed(4) : 'none'}`;

    // size everything first, then fill: see writeTube
    let verts = 0, tris = 0;
    built.fibers.forEach((f, i) => {
      const n = f.length / 3;
      if (n < 2) return;
      verts += n * sides;
      tris += (built.closed[i] ? n : n - 1) * sides * 2;
    });
    this._drop();
    if (!verts) {
      this.built = built;
      return { fibers: 0, dropped: built.dropped, vertices: 0,
               buildMs: performance.now() - t0 };
    }
    const buf = {
      pos: new Float32Array(verts * 3),
      nor: new Float32Array(verts * 3),
      col: new Float32Array(verts * 3),
      idx: verts > 65535 ? new Uint32Array(tris * 3) : new Uint16Array(tris * 3),
    };
    const cos = new Float64Array(sides), sin = new Float64Array(sides);
    for (let s = 0; s < sides; s++) {
      cos[s] = Math.cos(2 * Math.PI * s / sides);
      sin[s] = Math.sin(2 * Math.PI * s / sides);
    }
    const cur = { v: 0, i: 0, cos, sin, tangents: new Float64Array(0) };
    built.fibers.forEach((f, i) => {
      writeTube(f, built.closed[i], radius, sides,
                toLinear(paletteRgb(built.bases[i], palette)), buf, cur);
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(buf.pos, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(buf.nor, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(buf.col, 3));
    geo.setIndex(new THREE.BufferAttribute(buf.idx, 1));
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.35, metalness: 0.0,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.root.add(this.mesh);

    this.built = built;
    return {
      fibers: built.fibers.length,
      dropped: built.dropped,
      vertices: verts,
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
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.2);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }
}

/** The base sphere: shows where the fibres come from, and lets the
 *  reader put them there. */
/** A little text label that always faces the reader. */
function label(text, colour, position, scale = 0.26) {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = colour;
  g.font = 'bold 44px system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, 32, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false }));
  sp.position.set(...position);
  sp.scale.setScalar(scale);
  return sp;
}

export class BaseSphere extends Stage {
  constructor(canvas, { onPick = null } = {}) {
    // Far enough back for the polar axis and the N/S labels, which
    // stand off the surface and are the whole point of the markings.
    super(canvas, { distance: 4.6, fov: 34 });
    this.controls.enableZoom = false;
    this.controls.minDistance = 3.0;
    this.controls.maxDistance = 7;
    this.homePosition = this.camera.position.clone();
    this.onPick = onPick;
    this.painting = false;
    this.paintMode = false;      // drag turns the sphere unless this is on

    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 32),
      new THREE.MeshStandardMaterial({
        color: 0x2b303a, roughness: 0.85, metalness: 0.0,
        transparent: true, opacity: 0.92,
      }));
    this.root.add(ball);
    this.ball = ball;

    // The graticule, and the three lines that say which way up it is:
    // the equator, the prime meridian the longitude colours start from,
    // and the axis through the poles. Without them a featureless ball
    // turns under the pointer and the reader loses track of where the
    // fibres are coming from.
    const grid = new THREE.Group();
    const faint = new THREE.LineBasicMaterial({
      color: 0x566072, transparent: true, opacity: 0.45 });
    const equatorMat = new THREE.LineBasicMaterial({
      color: 0x9fb4d8, transparent: true, opacity: 0.95 });
    const meridianMat = new THREE.LineBasicMaterial({
      color: 0xffb454, transparent: true, opacity: 0.9 });

    const ring = (beta, mat) => {
      const r = Math.sin(beta), z = Math.cos(beta);
      const pts = [];
      for (let i = 0; i <= 96; i++) {
        const a = 2 * Math.PI * i / 96;
        pts.push(new THREE.Vector3(r * Math.cos(a) * 1.003,
                                   r * Math.sin(a) * 1.003, z * 1.003));
      }
      grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
    };
    for (let k = 1; k < 6; k++) {
      if (k === 3) continue;                       // the equator, below
      ring(Math.PI * k / 6, faint);
    }
    ring(Math.PI / 2, equatorMat);

    for (let k = 0; k < 6; k++) {
      const lam = Math.PI * k / 6;
      const pts = [];
      for (let i = 0; i <= 64; i++) {
        const b = Math.PI * i / 64;
        pts.push(new THREE.Vector3(Math.sin(b) * Math.cos(lam) * 1.003,
                                   Math.sin(b) * Math.sin(lam) * 1.003,
                                   Math.cos(b) * 1.003));
      }
      grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
                              k === 0 ? meridianMat : faint));
    }
    this.root.add(grid);

    // the polar axis, poking out at both ends, with the poles named
    const axisMat = new THREE.MeshStandardMaterial({
      color: 0x8f9bb3, roughness: 0.5 });
    const axis = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 2.3, 8), axisMat);
    axis.rotation.x = Math.PI / 2;
    this.root.add(axis);
    const capN = new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.14, 12),
                                new THREE.MeshStandardMaterial({ color: 0xf2f4f8 }));
    capN.position.set(0, 0, 1.14);
    capN.rotation.x = Math.PI / 2;
    this.root.add(capN);
    const capS = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 10),
                                new THREE.MeshStandardMaterial({ color: 0x4a5162 }));
    capS.position.set(0, 0, -1.14);
    this.root.add(capS);
    this.root.add(label('N', '#f2f4f8', [0, 0, 1.33]));
    this.root.add(label('S', '#97a1b5', [0, 0, -1.33]));

    this.dots = null;
    this.raycaster = new THREE.Raycaster();
    this._bindPicking();
  }

  /** Put the sphere back the way it started. */
  home() {
    this.camera.position.copy(this.homePosition);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  _bindPicking() {
    const pointerTo = (ev) => {
      const r = this.canvas.getBoundingClientRect();
      return new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1,
                               -((ev.clientY - r.top) / r.height) * 2 + 1);
    };
    const hit = (ev) => {
      this.raycaster.setFromCamera(pointerTo(ev), this.camera);
      const h = this.raycaster.intersectObject(this.ball, false);
      if (!h.length) return null;
      const p = h[0].point;
      return normalize3([p.x, p.y, p.z]);
    };

    // A drag turns the sphere and a click drops a fibre, which is the
    // division of labour people expect from every other 3-D view on
    // the site. The two are told apart afterwards, by how far the
    // pointer moved -- so the orbit never has to be given up to make
    // picking possible. Paint mode is the exception: there a drag is
    // a stroke of fibres, and the orbit stands down for its duration.
    const MOVED = 5;             // pixels; below this a drag is a click
    let downAt = null;
    let moved = 0;

    this.canvas.addEventListener('pointerdown', (ev) => {
      if (ev.button !== 0) return;
      downAt = [ev.clientX, ev.clientY];
      moved = 0;
      if (!this.paintMode) return;
      const b = hit(ev);
      if (!b) return;
      ev.preventDefault();
      this.painting = true;
      this.canvas.setPointerCapture(ev.pointerId);
      this.controls.enabled = false;
      this.last = b;
      if (this.onPick) this.onPick(b, 'start');
    });

    this.canvas.addEventListener('pointermove', (ev) => {
      if (downAt) {
        moved = Math.max(moved, Math.hypot(ev.clientX - downAt[0],
                                           ev.clientY - downAt[1]));
      }
      if (!this.painting) return;
      const b = hit(ev);
      if (!b) return;
      // one new point every few degrees, so a stroke paints a curve of
      // fibres rather than a thousand on top of each other
      const d = Math.acos(Math.max(-1, Math.min(1,
        b[0] * this.last[0] + b[1] * this.last[1] + b[2] * this.last[2])));
      if (d < 0.06) return;
      this.last = b;
      if (this.onPick) this.onPick(b, 'paint');
    });

    const finish = (ev) => {
      const wasPainting = this.painting;
      if (this.painting) {
        this.painting = false;
        this.controls.enabled = true;
        try { this.canvas.releasePointerCapture(ev.pointerId); } catch { /* gone */ }
      }
      if (!wasPainting && downAt && moved < MOVED) {
        const b = hit(ev);
        if (b && this.onPick) this.onPick(b, 'click');
      }
      downAt = null;
    };
    this.canvas.addEventListener('pointerup', finish);
    this.canvas.addEventListener('pointercancel', () => { downAt = null; });
  }

  /** Show these base points, in their fibres' colours. `tilted` says
   *  the points already have the generator's tilt applied. */
  show(points, palette, tilted = true) {
    if (this.dots) {
      this.root.remove(this.dots);
      this.dots.geometry.dispose();
      this.dots.material.dispose();
      this.dots = null;
    }
    if (!points.length) return;
    const R = tilted ? null : rotMatrix(...TILT);
    const geo = new THREE.SphereGeometry(0.045, 10, 8);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.4 });
    const mesh = new THREE.InstancedMesh(geo, mat, points.length);
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    points.forEach((p0, i) => {
      const p = R ? apply3(R, p0) : p0;
      m.makeTranslation(p[0] * 1.02, p[1] * 1.02, p[2] * 1.02);
      mesh.setMatrixAt(i, m);
      const rgb = paletteRgb(p, palette);
      c.setRGB(...toLinear(rgb));
      mesh.setColorAt(i, c);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    this.dots = mesh;
    this.root.add(mesh);
  }
}
