// The belt trick's WebGL scene.
//
// WHAT RUNS WHERE. The belts' mathematics runs on the CPU in float64
// (web/js/belt-math.js), because float32 would change it: measured, the
// width pass takes the other branch of its threshold at a few samples,
// and that branch is carried along the belt. The GPU gets the part that
// only affects the picture. Each frame the CPU writes every belt's
// centre line and width direction -- 160 samples each -- into a float
// texture, and the vertex shader builds the cross-section and the normal
// from them. That is about eleven times less data per frame than
// uploading finished vertices, and no normal computation on the CPU at
// all.
//
// The belt mesh itself is static: its vertices carry only which belt,
// which sample along it and where across it they sit.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { colour, qaxis, dot, CAGE_TUBE } from './belt-math.js';

const ACROSS = 11;                     // samples across a belt, as the generator

// Where the camera starts, as a direction. The marked face is chosen to
// face it at 0 degrees, so the face the reader is told to watch is in
// view at exactly the moments -- 0, 360, 720 -- it is there to be watched.
const VIEW_DIR = [1.35, -2.2, 0.95];

// The solid's base colour, and the one face that is marked so the solid
// can be SEEN to come back at 360 degrees. A cube at 360 is otherwise
// indistinguishable from a cube at 0, and that return is half the lesson.
const SOLID_RGB = [0.84, 0.84, 0.86];
const MARK_RGB = [1.0, 0.52, 0.12];

export class BeltView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.set(...VIEW_DIR).normalize().multiplyScalar(4.4);
    this.camera.up.set(0, 0, 1);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.6;
    this.controls.maxDistance = 9;

    // The polyhedra viewer's three-point rig, so the modules look alike.
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
    this.belts = null;
    this.solid = null;
    this.cage = null;
    this.onFrame = null;

    this._onResize = () => this.resize();
    addEventListener('resize', this._onResize);
    this.resize();

    let last = performance.now();
    const loop = (now) => {
      this._raf = requestAnimationFrame(loop);
      const dt = Math.min(0.1, (now - last) / 1000);   // no leap after a hidden tab
      last = now;
      if (this.onFrame) this.onFrame(dt);
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

  _drop(obj) {
    if (!obj) return;
    this.root.remove(obj);
    obj.traverse((o) => {
      o.geometry?.dispose();
      if (o.material) {
        o.material.uniforms?.uLine?.value?.dispose();
        o.material.dispose();
      }
    });
  }

  /** Rebuild for a new set of decisions from prepare(). */
  setup(prep, ns = 160) {
    this._drop(this.belts);
    this._drop(this.solid);
    this.prep = prep;
    this.ns = ns;
    this.belts = this._buildBelts(prep, ns);
    this.solid = this._buildSolid(prep);
    this.root.add(this.belts, this.solid);
    if (!this.cage) {
      this.cage = this._buildCage(prep.fit);
      this.root.add(this.cage);
    }
  }

  _buildBelts(prep, ns) {
    const nb = prep.belts.length;
    const nv = nb * ns * ACROSS;
    const aSample = new Float32Array(nv);
    const aBelt = new Float32Array(nv);
    const aAcross = new Float32Array(nv);
    const col = new Float32Array(nv * 3);
    let v = 0;
    for (let b = 0; b < nb; b++) {
      const c = colour(b);
      for (let i = 0; i < ns; i++) {
        for (let j = 0; j < ACROSS; j++) {
          aSample[v] = i;
          aBelt[v] = b;
          aAcross[v] = -1.0 + 2.0 * j / (ACROSS - 1);
          col[v * 3] = c[0]; col[v * 3 + 1] = c[1]; col[v * 3 + 2] = c[2];
          v++;
        }
      }
    }
    const idx = new Uint32Array(nb * (ns - 1) * (ACROSS - 1) * 6);
    let k = 0;
    for (let b = 0; b < nb; b++) {
      for (let i = 0; i < ns - 1; i++) {
        for (let j = 0; j < ACROSS - 1; j++) {
          const a = (b * ns + i) * ACROSS + j;
          // the generator's quad [a, a+1, a+nlam+1, a+nlam], as two triangles
          idx[k++] = a; idx[k++] = a + 1; idx[k++] = a + ACROSS + 1;
          idx[k++] = a; idx[k++] = a + ACROSS + 1; idx[k++] = a + ACROSS;
        }
      }
    }
    const g = new THREE.BufferGeometry();
    // Position and normal are computed in the shader; these exist only
    // because three.js expects the attributes to be there.
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    g.setAttribute('aSample', new THREE.BufferAttribute(aSample, 1));
    g.setAttribute('aBelt', new THREE.BufferAttribute(aBelt, 1));
    g.setAttribute('aAcross', new THREE.BufferAttribute(aAcross, 1));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));

    // Row 2b holds belt b's centre line, row 2b+1 its width direction.
    this.lineData = new Float32Array(ns * 2 * nb * 4);
    const tex = new THREE.DataTexture(this.lineData, ns, 2 * nb,
                                      THREE.RGBAFormat, THREE.FloatType);
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    this.lineTex = tex;

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, side: THREE.DoubleSide, roughness: 0.55, metalness: 0.0,
    });
    const uniforms = {
      uLine: { value: tex },
      uHalfW: { value: 0.5 * prep.width },
      uFit: { value: prep.fit },
      uNs: { value: ns },
    };
    mat.uniforms = uniforms;           // kept so _drop can dispose the texture
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, uniforms);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
uniform highp sampler2D uLine;
uniform float uHalfW;
uniform float uFit;
uniform int uNs;
attribute float aSample;
attribute float aBelt;
attribute float aAcross;
vec3 beltC(int i, int b) { return texelFetch(uLine, ivec2(i, 2 * b), 0).xyz; }
vec3 beltD(int i, int b) { return texelFetch(uLine, ivec2(i, 2 * b + 1), 0).xyz; }`)
        // The normal is the tangent crossed with the width direction:
        // exactly the strip's normal, from the same samples the
        // position comes from, so shading and shape cannot disagree.
        .replace('#include <beginnormal_vertex>', `
int bS = int(aSample + 0.5);
int bB = int(aBelt + 0.5);
vec3 bC = beltC(bS, bB);
vec3 bD = beltD(bS, bB);
vec3 bT = beltC(min(bS + 1, uNs - 1), bB) - beltC(max(bS - 1, 0), bB);
vec3 objectNormal = normalize(cross(normalize(bT), bD));
#ifdef USE_TANGENT
vec3 objectTangent = normalize(bT);
#endif`)
        .replace('#include <begin_vertex>',
                 'vec3 transformed = (bC + aAcross * uHalfW * bD) * uFit;');
    };
    const mesh = new THREE.Mesh(g, mat);
    // Bounds would come from the placeholder positions, all zero.
    mesh.frustumCulled = false;
    return mesh;
  }

  /**
   * The face to mark. Two things decide it: it should lie ACROSS the
   * spin axis, so that it travels visibly round the axis rather than
   * turning in place; and it should face the camera at 0 degrees, so it
   * is in view when the reader is told to watch it come back. The first
   * version took only the first criterion and marked a face pointing
   * away from the camera at 0 and at 360 -- the two moments it existed
   * for.
   */
  _markedFace(prep) {
    const vl = Math.hypot(...VIEW_DIR);
    const view = VIEW_DIR.map((c) => c / vl);
    let best = -1, bestScore = -Infinity;
    prep.faces.forEach((f, i) => {
      const a = prep.verts[f[0]], b = prep.verts[f[1]], c = prep.verts[f[2]];
      const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const nf = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                  e1[0] * e2[1] - e1[1] * e2[0]];
      const l = Math.hypot(nf[0], nf[1], nf[2]);
      const across = Math.abs(dot(nf, prep.n)) / l;       // 0 = across the axis
      const facing = dot(nf, view) / l;                      // 1 = at the camera
      const score = facing - 2.0 * across;
      if (score > bestScore + 1e-9) { bestScore = score; best = i; }
    });
    return best;
  }

  _buildSolid(prep) {
    const mark = this._markedFace(prep);
    const pos = [], col = [];
    prep.faces.forEach((f, fi) => {
      const c = fi === mark ? MARK_RGB : SOLID_RGB;
      for (let t = 1; t + 1 < f.length; t++) {
        for (const vi of [f[0], f[t], f[t + 1]]) {
          const p = prep.verts[vi];
          pos.push(p[0] * prep.fit, p[1] * prep.fit, p[2] * prep.fit);
          col.push(c[0], c[1], c[2]);
        }
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(col), 3));
    g.computeVertexNormals();        // flat: the triangles share no vertices
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, flatShading: true, roughness: 0.5, metalness: 0.0,
    });
    return new THREE.Mesh(g, mat);
  }

  _buildCage(fit) {
    // The generator's globe: six meridian great circles and five
    // latitude rings. Decoration rather than mathematics, so three.js's
    // tori stand in for the generator's swept tubes.
    const R = fit, tube = CAGE_TUBE * fit;
    const mat = new THREE.MeshStandardMaterial({ color: 0x9aa3af, roughness: 0.6 });
    const cage = new THREE.Group();
    const rings = 6;
    for (let i = 0; i < rings; i++) {
      const m = new THREE.Mesh(new THREE.TorusGeometry(R, tube, 6, 96), mat);
      m.rotation.set(Math.PI / 2, 0, 0);            // into the xz plane
      const holder = new THREE.Group();
      holder.rotation.z = Math.PI * i / rings;       // then about the axis
      holder.add(m);
      cage.add(holder);
    }
    for (let i = 1; i < rings; i++) {
      const z = R * Math.cos(Math.PI * i / rings);
      const r = Math.sqrt(Math.max(0, R * R - z * z));
      const m = new THREE.Mesh(new THREE.TorusGeometry(r, tube, 6, 96), mat);
      m.position.z = z;
      cage.add(m);
    }
    return cage;
  }

  /** Draw the belts at a turn (degrees): write the lines, turn the solid. */
  update(lines, turnDeg) {
    const ns = this.ns, d = this.lineData;
    for (let b = 0; b < lines.length; b++) {
      const P = lines[b].P, D = lines[b].dir;
      const c0 = (2 * b) * ns * 4, d0 = (2 * b + 1) * ns * 4;
      for (let i = 0; i < ns; i++) {
        const p = P[i], q = D[i];
        d[c0 + i * 4] = p[0]; d[c0 + i * 4 + 1] = p[1]; d[c0 + i * 4 + 2] = p[2];
        d[d0 + i * 4] = q[0]; d[d0 + i * 4 + 1] = q[1]; d[d0 + i * 4 + 2] = q[2];
      }
    }
    this.lineTex.needsUpdate = true;
    // The solid turns rigidly with the belts' roots: the rotation about n
    // through the TURN (twice the loop parameter psi).
    const q = qaxis(this.prep.n, turnDeg * Math.PI / 180);
    this.solid.quaternion.set(q[1], q[2], q[3], q[0]);   // three.js is x,y,z,w
  }

  setVisible({ solid = true, cage = true } = {}) {
    if (this.solid) this.solid.visible = solid;
    if (this.cage) this.cage.visible = cage;
  }
}
