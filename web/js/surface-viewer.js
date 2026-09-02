// surface-viewer.js -- an interactive triangle-mesh viewer, no dependencies.
//
// Written for surfaces rather than solids, which drives every design
// decision that differs from the polyhedron viewer next door:
//
//   * Surfaces here are OPEN and one-sided in the topological sense, so
//     the two faces of a sheet are shaded differently -- that is the only
//     way to read which way a minimal surface folds through itself.
//   * Normals are not stored.  Flat shading is derived per-fragment from
//     screen-space derivatives, which halves the payload and means a
//     mesh needs nothing but positions and indices.
//   * Viewers can be LINKED, so several of them share one orientation.
//     Dragging any member turns all of them together, which is what makes
//     two reconstructions of the same surface comparable at a glance.
//
// It is deliberately free of build steps and of three.js: it runs from a
// plain <script type="module">, and it also inlines cleanly into a
// single-file page, which is how the surface validation report uses it.
//
// Usage:
//
//     import { SurfaceViewer, linkViewers } from './surface-viewer.js';
//     const v = new SurfaceViewer(canvas, { background: [0,0,0,0] });
//     v.setMesh({ positions: Float32Array, indices: Uint32Array });
//     linkViewers([v, other]);        // share one orientation
//
// `setMesh` also accepts plain arrays, an {verts, faces} object with
// polygonal faces (they are fan-triangulated), or a packed payload from
// `decodeMesh` for pages that embed geometry as base64.

const VERT_SRC = `#version 300 es
precision highp float;
in vec3 aPos;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vWorld;
void main() {
  vWorld = (uModel * vec4(aPos, 1.0)).xyz;
  gl_Position = uMVP * vec4(aPos, 1.0);
}`;

// Flat shading from derivatives: the facet normal is recovered from how
// the interpolated world position changes across one pixel, so the mesh
// needs no normal attribute and no per-face vertex duplication.
const FRAG_SRC = `#version 300 es
precision highp float;
in vec3 vWorld;
uniform vec3 uFront;
uniform vec3 uBack;
uniform float uAmbient;
uniform int uSmooth;
out vec4 outColor;

void main() {
  vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  if (uSmooth == 1) {
    // A cheap smooth-ish look: soften the facet normal toward the view
    // direction so large flat triangles stop reading as a polyhedron.
    n = normalize(mix(n, normalize(vWorld - vec3(0.0, 0.0, -3.0)), 0.0));
  }
  bool front = gl_FrontFacing;
  vec3 base = front ? uFront : uBack;
  if (!front) n = -n;

  vec3 key  = normalize(vec3( 0.45, 0.35, 0.82));
  vec3 fill = normalize(vec3(-0.62, 0.18, 0.42));
  vec3 rim  = normalize(vec3( 0.10, -0.85, 0.30));
  float l = uAmbient
          + 0.72 * max(dot(n, key), 0.0)
          + 0.26 * max(dot(n, fill), 0.0)
          + 0.16 * max(dot(n, rim), 0.0);
  // A touch of specular keeps curvature readable on a matte palette.
  vec3 h = normalize(key + vec3(0.0, 0.0, 1.0));
  float s = pow(max(dot(n, h), 0.0), 28.0) * 0.30;
  outColor = vec4(base * l + vec3(s), 1.0);
}`;

const EDGE_VERT = `#version 300 es
precision highp float;
in vec3 aPos;
uniform mat4 uMVP;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_Position.z -= 0.0006 * gl_Position.w;   // lift wires off the surface
}`;

const EDGE_FRAG = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 outColor;
void main() { outColor = uColor; }`;

// ---------------------------------------------------------------- maths
// Small, explicit, and column-major to match WebGL. No matrix library.

function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function mat4Multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1]
                   + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const o = new Float32Array(16);
  o[0] = f / aspect; o[5] = f; o[11] = -1;
  o[10] = (far + near) / (near - far);
  o[14] = (2 * far * near) / (near - far);
  return o;
}

function mat4FromQuat(q) {
  const [x, y, z, w] = q;
  const o = mat4Identity();
  o[0] = 1 - 2 * (y * y + z * z); o[1] = 2 * (x * y + z * w);
  o[2] = 2 * (x * z - y * w);
  o[4] = 2 * (x * y - z * w);     o[5] = 1 - 2 * (x * x + z * z);
  o[6] = 2 * (y * z + x * w);
  o[8] = 2 * (x * z + y * w);     o[9] = 2 * (y * z - x * w);
  o[10] = 1 - 2 * (x * x + y * y);
  return o;
}

function quatMultiply(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function quatFromAxisAngle(axis, angle) {
  const s = Math.sin(angle / 2);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

function quatNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

// ------------------------------------------------------------- geometry

/** Fan-triangulate polygonal faces and drop unreferenced vertices. */
export function triangulate(verts, faces) {
  const tris = [];
  for (const f of faces) {
    for (let i = 2; i < f.length; i++) tris.push(f[0], f[i - 1], f[i]);
  }
  const used = new Int32Array(verts.length).fill(-1);
  for (const i of tris) used[i] = 0;
  let n = 0;
  for (let i = 0; i < used.length; i++) if (used[i] === 0) used[i] = n++;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < used.length; i++) {
    if (used[i] < 0) continue;
    positions[used[i] * 3] = verts[i][0];
    positions[used[i] * 3 + 1] = verts[i][1];
    positions[used[i] * 3 + 2] = verts[i][2];
  }
  const Idx = n > 65535 ? Uint32Array : Uint16Array;
  const indices = new Idx(tris.length);
  for (let i = 0; i < tris.length; i++) indices[i] = used[tris[i]];
  return { positions, indices };
}

/** Unique undirected edges of a triangle soup, as a line index buffer. */
export function edgeIndices(indices, vertexCount) {
  const seen = new Set();
  const out = [];
  for (let i = 0; i < indices.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[i + k], b = indices[i + (k + 1) % 3];
      const key = a < b ? a * vertexCount + b : b * vertexCount + a;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(a, b);
    }
  }
  const Idx = vertexCount > 65535 ? Uint32Array : Uint16Array;
  return new Idx(out);
}

/**
 * Decode a mesh packed as base64 by `pack_mesh` on the Python side:
 * positions quantised to uint16 over the mesh's own bounding box, and
 * indices as uint16 or uint32.  Quantisation is invisible at 16 bits
 * per axis and roughly halves an embedded payload.
 */
export function decodeMesh(packed) {
  const bytes = (b64) => {
    const s = atob(b64);
    const a = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
    return a;
  };
  const q = new Uint16Array(bytes(packed.p).buffer);
  const n = q.length / 3;
  const [lo, hi] = [packed.lo, packed.hi];
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      positions[i * 3 + k] = lo[k] + (hi[k] - lo[k]) * (q[i * 3 + k] / 65535);
    }
  }
  const raw = bytes(packed.i).buffer;
  const indices = packed.w === 4 ? new Uint32Array(raw) : new Uint16Array(raw);
  return { positions, indices };
}

// --------------------------------------------------------------- viewer

const DEFAULTS = {
  front: [0.80, 0.64, 0.32],
  back: [0.30, 0.46, 0.52],
  background: [0, 0, 0, 0],
  ambient: 0.22,
  wireframe: false,
  autoRotate: false,
  distance: 2.7,
  fov: 32,
};

export class SurfaceViewer {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({}, DEFAULTS, options);
    this.rotation = quatNormalize(options.rotation
      || quatMultiply(quatFromAxisAngle([1, 0, 0], -0.42),
                      quatFromAxisAngle([0, 1, 0], 0.62)));
    this.zoom = 1;
    this.group = null;
    this.dragging = false;
    this.dirty = true;
    this.mesh = null;

    const gl = canvas.getContext('webgl2', {
      antialias: true, alpha: true, premultipliedAlpha: false,
    });
    if (!gl) { this.fail('WebGL 2 is not available in this browser'); return; }
    this.gl = gl;
    this.prog = this.link(VERT_SRC, FRAG_SRC);
    this.edgeProg = this.link(EDGE_VERT, EDGE_FRAG);
    if (!this.prog || !this.edgeProg) return;

    this.buf = gl.createBuffer();
    this.ibo = gl.createBuffer();
    this.ebo = gl.createBuffer();
    gl.enable(gl.DEPTH_TEST);
    // Two-sided by design: a minimal surface has no inside to cull to.
    gl.disable(gl.CULL_FACE);

    this.bindInput();
    this.observer = new ResizeObserver(() => { this.dirty = true; });
    this.observer.observe(canvas);
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  fail(message) {
    this.error = message;
    const p = document.createElement('p');
    p.className = 'surface-viewer-error';
    p.textContent = message;
    if (this.canvas.parentNode) this.canvas.parentNode.appendChild(p);
    this.canvas.style.display = 'none';
  }

  link(vs, fs) {
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        this.fail('shader: ' + gl.getShaderInfoLog(s));
        return null;
      }
      return s;
    };
    const v = compile(gl.VERTEX_SHADER, vs);
    const f = compile(gl.FRAGMENT_SHADER, fs);
    if (!v || !f) return null;
    const p = gl.createProgram();
    gl.attachShader(p, v);
    gl.attachShader(p, f);
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      this.fail('link: ' + gl.getProgramInfoLog(p));
      return null;
    }
    return p;
  }

  /**
   * Give the viewer a mesh.  Accepts {positions, indices} typed arrays,
   * {verts, faces} with polygonal faces, or a packed payload.
   * The mesh is centred and scaled to a unit ball so that meshes of very
   * different sizes -- one cell against a block of eight -- can be
   * compared without either being a speck.
   */
  setMesh(mesh) {
    if (!this.gl) return;
    let m = mesh;
    if (mesh && mesh.p && mesh.i) m = decodeMesh(mesh);
    else if (mesh && mesh.verts) m = triangulate(mesh.verts, mesh.faces);
    else if (mesh && !(m.positions instanceof Float32Array)) {
      m = { positions: new Float32Array(mesh.positions),
            indices: (mesh.positions.length / 3) > 65535
              ? new Uint32Array(mesh.indices)
              : new Uint16Array(mesh.indices) };
    }
    const P = m.positions;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        if (P[i + k] < lo[k]) lo[k] = P[i + k];
        if (P[i + k] > hi[k]) hi[k] = P[i + k];
      }
    }
    const c = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    const s = 2 / span;
    const N = new Float32Array(P.length);
    for (let i = 0; i < P.length; i += 3) {
      for (let k = 0; k < 3; k++) N[i + k] = (P[i + k] - c[k]) * s;
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, N, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.indices, gl.STATIC_DRAW);
    this.mesh = {
      count: m.indices.length,
      u32: m.indices instanceof Uint32Array,
      vertexCount: N.length / 3,
      indices: m.indices,
    };
    this.edges = null;
    this.dirty = true;
    return this;
  }

  buildEdges() {
    if (this.edges || !this.mesh) return;
    const e = edgeIndices(this.mesh.indices, this.mesh.vertexCount);
    const gl = this.gl;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, e, gl.STATIC_DRAW);
    this.edges = { count: e.length, u32: e instanceof Uint32Array };
  }

  set(option, value) {
    this.opts[option] = value;
    if (option === 'wireframe' && value) this.buildEdges();
    this.dirty = true;
    return this;
  }

  /** Face the surface down one of its axes; handy for a "top" button. */
  setView(name) {
    const views = {
      home: quatMultiply(quatFromAxisAngle([1, 0, 0], -0.42),
                         quatFromAxisAngle([0, 1, 0], 0.62)),
      front: [0, 0, 0, 1],
      top: quatFromAxisAngle([1, 0, 0], -Math.PI / 2),
      side: quatFromAxisAngle([0, 1, 0], Math.PI / 2),
    };
    if (views[name]) this.setRotation(views[name], true);
    return this;
  }

  setRotation(q, propagate) {
    this.rotation = quatNormalize(q);
    this.dirty = true;
    if (propagate && this.group) this.group.broadcast(this, this.rotation);
  }

  setZoom(z, propagate) {
    this.zoom = Math.min(4, Math.max(0.3, z));
    this.dirty = true;
    if (propagate && this.group) this.group.broadcastZoom(this, this.zoom);
  }

  bindInput() {
    const c = this.canvas;
    c.style.touchAction = 'none';
    let last = null;
    const down = (e) => {
      last = [e.clientX, e.clientY];
      this.dragging = true;
      c.setPointerCapture(e.pointerId);
    };
    const move = (e) => {
      if (!this.dragging || !last) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      const k = 0.0095;
      // Trackball about the SCREEN axes, applied on the left, so drag
      // direction always matches what the eye expects however far the
      // object has already been turned.
      const q = quatMultiply(quatFromAxisAngle([0, 1, 0], dx * k),
                             quatFromAxisAngle([1, 0, 0], dy * k));
      this.setRotation(quatMultiply(q, this.rotation), true);
    };
    const up = (e) => {
      this.dragging = false;
      last = null;
      if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
    };
    c.addEventListener('pointerdown', down);
    c.addEventListener('pointermove', move);
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.setZoom(this.zoom * Math.exp(-e.deltaY * 0.0012), true);
    }, { passive: false });
    c.addEventListener('dblclick', () => this.setView('home'));
  }

  resize() {
    const c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(c.clientWidth * dpr));
    const h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return [w, h];
  }

  loop() {
    if (this.gl && !this.error) {
      if (this.opts.autoRotate && !this.dragging) {
        this.setRotation(quatMultiply(
          quatFromAxisAngle([0, 1, 0], 0.004), this.rotation), false);
      }
      if (this.dirty) { this.draw(); this.dirty = false; }
    }
    requestAnimationFrame(this.loop);
  }

  draw() {
    const gl = this.gl;
    const [w, h] = this.resize();
    gl.viewport(0, 0, w, h);
    const bg = this.opts.background;
    gl.clearColor(bg[0], bg[1], bg[2], bg[3]);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.mesh) return;

    const model = mat4FromQuat(this.rotation);
    const d = this.opts.distance / this.zoom;
    const view = mat4Identity();
    view[14] = -d;
    const proj = mat4Perspective(this.opts.fov * Math.PI / 180,
                                 w / h, 0.05, 100);
    const mvp = mat4Multiply(proj, mat4Multiply(view, model));

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'uMVP'), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'uModel'),
                        false, model);
    gl.uniform3fv(gl.getUniformLocation(this.prog, 'uFront'),
                  new Float32Array(this.opts.front));
    gl.uniform3fv(gl.getUniformLocation(this.prog, 'uBack'),
                  new Float32Array(this.opts.back));
    gl.uniform1f(gl.getUniformLocation(this.prog, 'uAmbient'),
                 this.opts.ambient);
    gl.uniform1i(gl.getUniformLocation(this.prog, 'uSmooth'), 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.drawElements(gl.TRIANGLES, this.mesh.count,
                    this.mesh.u32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, 0);

    if (this.opts.wireframe) {
      this.buildEdges();
      gl.useProgram(this.edgeProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.edgeProg, 'uMVP'),
                          false, mvp);
      gl.uniform4f(gl.getUniformLocation(this.edgeProg, 'uColor'),
                   0, 0, 0, 0.28);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
      gl.drawElements(gl.LINES, this.edges.count,
                      this.edges.u32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, 0);
      gl.disable(gl.BLEND);
    }
  }

  destroy() {
    if (this.observer) this.observer.disconnect();
    if (this.group) this.group.remove(this);
    this.gl = null;
  }
}

// ---------------------------------------------------------------- groups

/**
 * Share one orientation between several viewers.
 *
 * The whole point of the comparison this was written for: two
 * reconstructions of the same surface, turned together, so a difference
 * in shape cannot hide behind a difference in viewpoint.
 */
export class ViewerGroup {
  constructor(members = []) {
    this.members = [];
    members.forEach((m) => this.add(m));
  }

  add(viewer) {
    if (!viewer || this.members.includes(viewer)) return this;
    this.members.push(viewer);
    viewer.group = this;
    const lead = this.members[0];
    if (viewer !== lead) {
      viewer.setRotation(lead.rotation, false);
      viewer.setZoom(lead.zoom, false);
    }
    return this;
  }

  remove(viewer) {
    this.members = this.members.filter((m) => m !== viewer);
    if (viewer) viewer.group = null;
    return this;
  }

  broadcast(from, rotation) {
    for (const m of this.members) if (m !== from) m.setRotation(rotation, false);
  }

  broadcastZoom(from, zoom) {
    for (const m of this.members) if (m !== from) m.setZoom(zoom, false);
  }
}

export function linkViewers(viewers) {
  return new ViewerGroup(viewers);
}

export default SurfaceViewer;
