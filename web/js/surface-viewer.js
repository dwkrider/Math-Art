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

// One draw call per mesh, instanced.
//
// A periodic surface's unit cell is one patch repeated under a symmetry
// group -- up to 192 times here -- so sending the assembled cell sends
// the same few hundred triangles nearly two hundred times over.  Sending
// the patch once and the group as instance transforms is two orders of
// magnitude smaller, and it means a page can afford the patch at FULL
// resolution instead of throwing triangles away to fit.
//
// `aFlip` carries the sign of each instance's determinant.  A reflected
// copy has its winding reversed, which instancing cannot fix by
// re-ordering indices, so the shader flips the facing test instead.
const VERT_SRC = `#version 300 es
precision highp float;
in vec3 aPos;
in mat4 aInst;
in float aFlip;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform int uInstanced;
out vec3 vWorld;
flat out float vFlip;
void main() {
  vec4 p = uInstanced == 1 ? aInst * vec4(aPos, 1.0) : vec4(aPos, 1.0);
  vFlip = uInstanced == 1 ? aFlip : 1.0;
  vWorld = (uModel * p).xyz;
  gl_Position = uMVP * p;
}`;

// Flat shading from derivatives: the facet normal is recovered from how
// the interpolated world position changes across one pixel, so the mesh
// needs no normal attribute and no per-face vertex duplication.
const FRAG_SRC = `#version 300 es
precision highp float;
in vec3 vWorld;
flat in float vFlip;
uniform vec3 uFront;
uniform vec3 uBack;
uniform float uAmbient;
out vec4 outColor;

void main() {
  vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  bool front = (vFlip < 0.0) ? !gl_FrontFacing : gl_FrontFacing;
  vec3 base = front ? uFront : uBack;
  if (!front) n = -n;

  // The documentation studio's lamps, in CAMERA space.
  //
  // docs/render_docs.py lights every figure in the repository with a fixed
  // four-light rig. Its lamps sit at fixed positions in the Blender world
  // and so does its camera, which means they are fixed relative to the
  // camera -- and this shader's normals are in the viewer's camera frame.
  // So each direction below is the lamp's Blender unit vector resolved
  // onto the studio camera basis (right, up, toward-camera) built in
  // STUDIO_VIEW above: key (1.8,-1.9,1.6), fill (-2.4,-0.8,0.5), rims
  // (-1.7,1.9,1.1) and (1.9,1.7,0.8), top (0,0.3,2.6).
  //
  // Weights are the studio's own energies: key 320, rims 750 each scaled
  // by subjects.STUDIO_RIM_SCALE = 0.35, top 150, fill 70.
  vec3 key  = normalize(vec3( 0.176,  0.201,  0.964));
  vec3 fill = normalize(vec3(-0.956,  0.259, -0.142));
  vec3 rimL = normalize(vec3(-0.164,  0.684, -0.711));
  vec3 rimR = normalize(vec3( 0.939,  0.340, -0.057));
  vec3 top  = normalize(vec3( 0.060,  0.966,  0.251));

  // WRAPPED lighting, and a hemisphere term, so that nothing ever falls
  // to black.  With plain lambert lights every face pointing away from
  // all of them dropped to ambient alone, and on a surface as folded as
  // these that is most of what you see from any given angle -- half the
  // object read as a black silhouette and the detail in it was simply
  // gone.  Cycles never has that problem because the studio's dome
  // bounces light back; wrapping is the cheap stand-in for that bounce.
  float w = 0.45;
  float lk = clamp((dot(n, key)  + w) / (1.0 + w), 0.0, 1.0);
  float lf = clamp((dot(n, fill) + w) / (1.0 + w), 0.0, 1.0);
  float ll = clamp((dot(n, rimL) + w) / (1.0 + w), 0.0, 1.0);
  float lr = clamp((dot(n, rimR) + w) / (1.0 + w), 0.0, 1.0);
  float lt = clamp((dot(n, top)  + w) / (1.0 + w), 0.0, 1.0);
  float sky = 0.5 + 0.5 * n.y;            // the dome, softly
  float l = uAmbient + 0.10 * sky
          + 0.44 * lk + 0.10 * lf + 0.13 * ll + 0.13 * lr + 0.20 * lt;

  // The studio's White Plastic is roughness 0.38, which is tighter and
  // brighter than the broad sheen this had before.
  vec3 h = normalize(key + vec3(0.0, 0.0, 1.0));
  float s = pow(max(dot(n, h), 0.0), 42.0) * 0.30;
  outColor = vec4(base * l + vec3(s), 1.0);
}`;

const EDGE_VERT = `#version 300 es
precision highp float;
in vec3 aPos;
in mat4 aInst;
uniform mat4 uMVP;
uniform int uInstanced;
void main() {
  vec4 p = uInstanced == 1 ? aInst * vec4(aPos, 1.0) : vec4(aPos, 1.0);
  gl_Position = uMVP * p;
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
  const out = { positions, indices };
  if (packed.m) out.instances = new Float32Array(bytes(packed.m).buffer);
  return out;
}

// --------------------------------------------------------------- viewer

/**
 * The documentation studio's material: White Plastic at base colour
 * (0.84, 0.84, 0.86) -- see docs/render_docs.py.
 */
export const STUDIO_PLASTIC = [0.84, 0.84, 0.86];

// ONE material on both sides, as the studio has it. Cycles renders these
// surfaces with a single material and simply flips the normal on a back
// face, and the sheets still read: the fold is carried by the shading and
// the silhouette, not by a colour change.
//
// An earlier version tinted the back face to tell the sides apart. That
// is a genuinely useful thing to do -- it is the quickest way to see
// which way a minimal surface passes through itself -- but it makes the
// viewer disagree with every thumbnail and documentation figure of the
// same surface, and matching those matters more here. It remains one
// option away: pass `back` to get it.
//
// Whatever you pass, keep it LIGHT. A dark second colour tells the sides
// apart by throwing away the shading that carries the shape, and a
// surface with one bright side and one near-black one reads as half a
// surface.
const DEFAULTS = {
  front: STUDIO_PLASTIC,
  back: STUDIO_PLASTIC,
  background: [0, 0, 0, 0],
  ambient: 0.30,
  wireframe: false,
  autoRotate: false,
  distance: 2.7,
  fov: 32,
};

/**
 * A canvas that can take a NEW WebGL context.
 *
 * A canvas whose context has been lost hands back that same dead context
 * from `getContext` forever after -- every shader then fails to compile
 * and `getShaderInfoLog` returns null, which is where "shader: null"
 * comes from.  The only cure is a fresh element, so a canvas that has
 * been through a viewer is replaced by a clone before being reused.
 */
/**
 * The orientation the documentation studio shoots from.
 *
 * docs/render_docs.py puts its camera on the direction (1.35, -2.2, 0.95)
 * in Blender's Z-up world, aimed at the origin with world +Z as up. Every
 * hero figure, every variant tile and every surface thumbnail in
 * web/thumbs/ is taken from there, so a viewer that opens on the same
 * orientation shows the reader the picture they just clicked.
 *
 * NO axis remap belongs here. The meshes in web/surfaces/ are exported
 * straight out of Blender, so their coordinates are already Z-up: the
 * rotation below has to convert that convention AND aim the camera, and
 * doing both at once is exactly what a frame built from the Blender
 * camera vector and Blender's world up does. Remapping the camera vector
 * first converts twice, which put the catenoid's axis down the line of
 * sight instead of standing it up as the thumbnail has it.
 */
export const STUDIO_VIEW = (() => {
  const norm = (v) => {
    const L = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / L, v[1] / L, v[2] / L];
  };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                           a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0]];
  const f = norm([1.35, -2.2, 0.95]);       // toward the studio camera
  const r = norm(cross([0, 0, 1], f));      // camera right (Blender up)
  const u = cross(f, r);                    // camera up, orthogonalised
  // Rows r, u, f: the matrix sending the camera frame to x, y, z.
  const m = [r, u, f];
  const tr = m[0][0] + m[1][1] + m[2][2];
  let q;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
         (m[1][0] - m[0][1]) / s, 0.25 * s];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
    q = [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s,
         (m[2][1] - m[1][2]) / s];
  } else if (m[1][1] > m[2][2]) {
    const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
    q = [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s,
         (m[0][2] - m[2][0]) / s];
  } else {
    const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
    q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s,
         (m[1][0] - m[0][1]) / s];
  }
  const L = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / L, q[1] / L, q[2] / L, q[3] / L];
})();

export function freshCanvas(canvas) {
  if (!canvas.dataset.svUsed || !canvas.parentNode) {
    canvas.dataset.svUsed = '1';
    return canvas;
  }
  const next = canvas.cloneNode(false);
  next.dataset.svUsed = '1';
  canvas.parentNode.replaceChild(next, canvas);
  return next;
}

export class SurfaceViewer {
  constructor(canvas, options = {}) {
    this.canvas = canvas = freshCanvas(canvas);
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
    if (this.error) return;              // one message, not one per shader
    this.error = message;
    const host = this.canvas.parentNode;
    if (host) {
      const old = host.querySelector(':scope > .surface-viewer-error');
      if (old) old.remove();
      const p = document.createElement('p');
      p.className = 'surface-viewer-error';
      p.textContent = message;
      host.appendChild(p);
    }
    this.canvas.style.display = 'none';
  }

  link(vs, fs) {
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        // A null log almost always means the context is already gone
        // rather than that the source is bad; say so, because "shader:
        // null" tells the reader nothing.
        const log = gl.getShaderInfoLog(s);
        this.fail(log ? 'shader: ' + log
                      : (gl.isContextLost() ? 'graphics context was lost'
                                            : 'shader failed to compile'));
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
    // Fixed slots: a mat4 attribute occupies four consecutive ones, so
    // aInst takes 1-4 and aFlip has to start at 5.
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.bindAttribLocation(p, 1, 'aInst');
    gl.bindAttribLocation(p, 5, 'aFlip');
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
    const inst = m.instances || null;
    const count = inst ? inst.length / 16 : 1;

    // Fit the WHOLE assembly, not the patch: with instancing the patch
    // is a fraction of what is on screen, and framing on it alone would
    // put the cell far outside the view.
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    const bump = (x, y, z) => {
      if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
      if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
      if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
    };
    for (let i = 0; i < P.length; i += 3) {
      if (!inst) { bump(P[i], P[i + 1], P[i + 2]); continue; }
      for (let j = 0; j < count; j++) {
        const M = inst.subarray(j * 16, j * 16 + 16);   // column-major
        bump(M[0] * P[i] + M[4] * P[i + 1] + M[8] * P[i + 2] + M[12],
             M[1] * P[i] + M[5] * P[i + 1] + M[9] * P[i + 2] + M[13],
             M[2] * P[i] + M[6] * P[i + 1] + M[10] * P[i + 2] + M[14]);
      }
    }
    const c = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    const s = 2 / span;

    const gl = this.gl;
    let positions = P;
    let instances = inst;
    if (inst) {
      // Fold the centring and scaling into the instance transforms so
      // the patch data stays untouched and shared.
      instances = new Float32Array(inst);
      for (let j = 0; j < count; j++) {
        const o = j * 16;
        for (let k = 0; k < 12; k++) instances[o + k] *= s;
        instances[o + 12] = (inst[o + 12] - c[0]) * s;
        instances[o + 13] = (inst[o + 13] - c[1]) * s;
        instances[o + 14] = (inst[o + 14] - c[2]) * s;
      }
    } else {
      positions = new Float32Array(P.length);
      for (let i = 0; i < P.length; i += 3) {
        for (let k = 0; k < 3; k++) positions[i + k] = (P[i + k] - c[k]) * s;
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.indices, gl.STATIC_DRAW);

    if (instances) {
      if (!this.ibuf) this.ibuf = gl.createBuffer();
      if (!this.fbuf) this.fbuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.ibuf);
      gl.bufferData(gl.ARRAY_BUFFER, instances, gl.STATIC_DRAW);
      const flip = new Float32Array(count);
      for (let j = 0; j < count; j++) {
        const M = instances.subarray(j * 16, j * 16 + 16);
        // det of the upper-left 3x3, column-major
        flip[j] = Math.sign(
          M[0] * (M[5] * M[10] - M[9] * M[6])
          - M[4] * (M[1] * M[10] - M[9] * M[2])
          + M[8] * (M[1] * M[6] - M[5] * M[2])) || 1;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.fbuf);
      gl.bufferData(gl.ARRAY_BUFFER, flip, gl.STATIC_DRAW);
    }

    this.mesh = {
      count: m.indices.length,
      u32: m.indices instanceof Uint32Array,
      vertexCount: positions.length / 3,
      indices: m.indices,
      instances: instances ? count : 0,
    };
    this.edges = null;
    this.dirty = true;
    return this;
  }

  /** Bind aPos, and the per-instance mat4 + flip when there is one. */
  bindAttribs(prog, withFlip) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    const aPos = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(aPos, 0);

    const aInst = gl.getAttribLocation(prog, 'aInst');
    if (aInst >= 0) {
      if (this.mesh.instances) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.ibuf);
        for (let k = 0; k < 4; k++) {
          gl.enableVertexAttribArray(aInst + k);
          gl.vertexAttribPointer(aInst + k, 4, gl.FLOAT, false, 64, k * 16);
          gl.vertexAttribDivisor(aInst + k, 1);
        }
      } else {
        for (let k = 0; k < 4; k++) gl.disableVertexAttribArray(aInst + k);
      }
    }
    const aFlip = withFlip ? gl.getAttribLocation(prog, 'aFlip') : -1;
    if (aFlip >= 0) {
      if (this.mesh.instances) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.fbuf);
        gl.enableVertexAttribArray(aFlip);
        gl.vertexAttribPointer(aFlip, 1, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(aFlip, 1);
      } else {
        gl.disableVertexAttribArray(aFlip);
        gl.vertexAttrib1f(aFlip, 1.0);
      }
    }
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

    const n = this.mesh.instances;
    const type = this.mesh.u32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;

    gl.useProgram(this.prog);
    this.bindAttribs(this.prog, true);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'uMVP'), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'uModel'),
                        false, model);
    gl.uniform3fv(gl.getUniformLocation(this.prog, 'uFront'),
                  new Float32Array(this.opts.front));
    gl.uniform3fv(gl.getUniformLocation(this.prog, 'uBack'),
                  new Float32Array(this.opts.back));
    gl.uniform1f(gl.getUniformLocation(this.prog, 'uAmbient'),
                 this.opts.ambient);
    gl.uniform1i(gl.getUniformLocation(this.prog, 'uInstanced'), n ? 1 : 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    if (n) {
      gl.drawElementsInstanced(gl.TRIANGLES, this.mesh.count, type, 0, n);
    } else {
      gl.drawElements(gl.TRIANGLES, this.mesh.count, type, 0);
    }

    if (this.opts.wireframe) {
      this.buildEdges();
      gl.useProgram(this.edgeProg);
      this.bindAttribs(this.edgeProg, false);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.edgeProg, 'uMVP'),
                          false, mvp);
      gl.uniform1i(gl.getUniformLocation(this.edgeProg, 'uInstanced'),
                   n ? 1 : 0);
      gl.uniform4f(gl.getUniformLocation(this.edgeProg, 'uColor'),
                   0, 0, 0, 0.28);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
      const etype = this.edges.u32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      if (n) {
        gl.drawElementsInstanced(gl.LINES, this.edges.count, etype, 0, n);
      } else {
        gl.drawElements(gl.LINES, this.edges.count, etype, 0);
      }
      gl.disable(gl.BLEND);
    }
  }

  /**
   * Release the viewer and its WebGL context.
   *
   * Explicitly losing the context matters on a page with many viewers:
   * browsers cap live WebGL contexts at around sixteen and silently drop
   * the oldest, so a long gallery has to recycle them rather than leak
   * them.  `ViewerPool` below does exactly that.
   */
  destroy() {
    if (this.observer) this.observer.disconnect();
    if (this.group) this.group.remove(this);
    const gl = this.gl;
    this.gl = null;
    this.mesh = null;
    if (gl) {
      const lose = gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
    }
  }
}

/**
 * Keep at most `limit` viewers alive, oldest recycled first.
 *
 * For a page that shows dozens of surfaces: build viewers as cards come
 * into view, and let the pool retire the ones that have scrolled away
 * before the browser starts dropping contexts on its own.
 */
export class ViewerPool {
  constructor(limit = 12) {
    this.limit = limit;
    this.live = new Map();          // key -> {viewers, teardown}
  }

  /** `make()` must return an array of viewers; it runs only if needed. */
  acquire(key, make) {
    if (this.live.has(key)) {
      const entry = this.live.get(key);
      this.live.delete(key);
      this.live.set(key, entry);    // refresh recency
      return entry.viewers;
    }
    const viewers = make() || [];
    this.live.set(key, { viewers });
    while (this.live.size > this.limit) {
      const oldest = this.live.keys().next().value;
      if (oldest === key) break;
      this.release(oldest);
    }
    return viewers;
  }

  release(key) {
    const entry = this.live.get(key);
    if (!entry) return;
    this.live.delete(key);
    entry.viewers.forEach((v) => v && v.destroy());
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
