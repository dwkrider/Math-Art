// Turn a database record into three.js geometry.
//
// Colours and component grouping deliberately match
// tools/render_polyhedra_thumbs.py, so the live model looks like the
// catalogue tile that led the reader to it.

import * as THREE from 'three';
import { tessellateFace } from './tessellate.js';

// The add-on's face-size palette (regular_solids_generator.py).
const FACE_PALETTE = {
  3: 0xe65c3b, 4: 0x4585c9, 5: 0x4cb06b, 6: 0xf2c44a, 7: 0x9e66bf,
  8: 0x40b8b8, 9: 0xe88fb5, 10: 0x8c994a, 12: 0x857366,
};
const PLAIN = 0xd6d6dc;

// compound_generator.py's component palette.
const COMPOUND_PALETTE = [
  0xe65c3b, 0x4585c9, 0x4cb06b, 0xf2c44a, 0x9e66bf,
  0x40b8b8, 0xe88fb5, 0x8c994a, 0xcc7a4d, 0x738ccc,
];

/**
 * Group faces into connected components by SHARED EDGE.
 *
 * Shared-vertex connectivity is the obvious choice and it is wrong: the
 * components of a compound routinely share vertices. The five cubes
 * inscribed in a dodecahedron sit on its twenty vertices, so vertex
 * connectivity reports one blob and the compound renders in one colour.
 * Components never share an EDGE, which is the relation used here.
 */
export function faceComponents(faces) {
  const parent = faces.map((_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const seen = new Map();
  faces.forEach((f, fi) => {
    for (let k = 0; k < f.length; k++) {
      const a = f[k], b = f[(k + 1) % f.length];
      const e = Math.min(a, b) + ':' + Math.max(a, b);
      if (seen.has(e)) {
        const ra = find(seen.get(e)), rb = find(fi);
        if (ra !== rb) parent[ra] = rb;
      } else seen.set(e, fi);
    }
  });
  const roots = new Map();
  const compOf = faces.map((_, fi) => {
    const r = find(fi);
    if (!roots.has(r)) roots.set(r, roots.size);
    return roots.get(r);
  });
  return { compOf, count: roots.size };
}

/**
 * Build the solid's surface.
 *
 * Returns a non-indexed BufferGeometry with flat normals and per-vertex
 * colour. Non-indexed because each face contributes its own triangles
 * with its own flat normal -- polyhedra are flat by definition, and
 * sharing vertices between faces would smooth away the very edges the
 * picture is about.
 */
export function buildSurface(rec, opts = {}) {
  const V = rec.geometry.vertices;
  const faces = rec.geometry.faces;
  const { compOf, count } = faceComponents(faces);
  // Colour a compound by component; anything else by face size. Colouring
  // a compound by face size would paint all five cubes the same blue and
  // lose the very thing the picture is of.
  const byComponent = count > 1;
  const mode = opts.coloring || 'auto';

  const pos = [];
  const col = [];
  const nrm = [];
  const tmp = new THREE.Color();

  faces.forEach((f, fi) => {
    const { points, tris, normal } = tessellateFace(f.map((i) => V[i]));
    let hex = PLAIN;
    if (mode !== 'none') {
      if (byComponent) hex = COMPOUND_PALETTE[compOf[fi] % COMPOUND_PALETTE.length];
      else hex = FACE_PALETTE[f.length] ?? PLAIN;
    }
    tmp.setHex(hex);
    for (const [a, b, c] of tris) {
      for (const i of [a, b, c]) {
        pos.push(points[i][0], points[i][1], points[i][2]);
        col.push(tmp.r, tmp.g, tmp.b);
        nrm.push(normal[0], normal[1], normal[2]);
      }
    }
  });

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  return g;
}

/** The polyhedron's edge set, as one LineSegments geometry. */
export function buildEdges(rec) {
  const V = rec.geometry.vertices;
  const seen = new Set();
  const pos = [];
  for (const f of rec.geometry.faces) {
    for (let k = 0; k < f.length; k++) {
      const a = f[k], b = f[(k + 1) % f.length];
      const key = Math.min(a, b) + ':' + Math.max(a, b);
      if (seen.has(key)) continue;
      seen.add(key);
      pos.push(V[a][0], V[a][1], V[a][2], V[b][0], V[b][1], V[b][2]);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  return g;
}

/** The solid's edge length: the natural unit for strut thickness. */
export function edgeLength(rec) {
  const stated = rec.metrics?.edge_length?.value;
  if (stated) return stated;
  // Not every record states one (a solid with unequal edges has no single
  // value), so fall back to measuring the first edge.
  const V = rec.geometry.vertices;
  const f = rec.geometry.faces[0];
  if (!f || f.length < 2) return 1;
  const a = V[f[0]], b = V[f[1]];
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) || 1;
}

/**
 * Ball-and-stick: a sphere at each vertex, a cylinder along each edge.
 *
 * Thickness is a fraction of EDGE LENGTH, not of the bounding radius.
 * Keying it off the bounding radius makes the struts scale with how
 * bulky the solid is, so a 60-vertex truncated icosahedron comes out
 * spidery while a tetrahedron comes out clubbed; edge length is the
 * scale the model is actually built at, and holds the look steady
 * across all 448.
 *
 * Built as merged geometry rather than many objects so a 200-vertex
 * compound stays one draw call.
 */
export function buildBallAndStick(rec, thickness) {
  const V = rec.geometry.vertices;
  const r = (thickness ?? 0.06) * edgeLength(rec);
  const parts = [];
  const sphere = new THREE.SphereGeometry(r * 1.8, 16, 12);
  for (const v of V) {
    const g = sphere.clone();
    g.translate(v[0], v[1], v[2]);
    parts.push(g);
  }
  const seen = new Set();
  const up = new THREE.Vector3(0, 1, 0);
  for (const f of rec.geometry.faces) {
    for (let k = 0; k < f.length; k++) {
      const ai = f[k], bi = f[(k + 1) % f.length];
      const key = Math.min(ai, bi) + ':' + Math.max(ai, bi);
      if (seen.has(key)) continue;
      seen.add(key);
      const a = new THREE.Vector3(...V[ai]);
      const b = new THREE.Vector3(...V[bi]);
      const d = new THREE.Vector3().subVectors(b, a);
      const len = d.length();
      if (len < 1e-9) continue;
      const g = new THREE.CylinderGeometry(r, r, len, 10, 1);
      g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(
        up, d.clone().normalize()));
      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      g.translate(mid.x, mid.y, mid.z);
      parts.push(g);
    }
  }
  return mergeGeometries(parts);
}

// A small local merge: three's BufferGeometryUtils lives in the examples
// tree, and vendoring one more file to concatenate float arrays is not
// worth it.
//
// Every part is expanded with toNonIndexed() first. Sphere and cylinder
// geometries are INDEXED -- their position array holds unique vertices
// and the triangles live in the index buffer -- so concatenating the
// positions alone silently discards the topology and draws a triangle
// soup between unrelated vertices. It does not look like an error; it
// looks like very thin struts.
function mergeGeometries(indexedParts) {
  const parts = indexedParts.map((g) => {
    const flat = g.index ? g.toNonIndexed() : g;
    if (flat !== g) g.dispose();
    return flat;
  });
  let n = 0;
  for (const g of parts) n += g.getAttribute('position').count;
  const pos = new Float32Array(n * 3);
  const nrm = new Float32Array(n * 3);
  let o = 0;
  for (const g of parts) {
    const p = g.getAttribute('position');
    const m = g.getAttribute('normal');
    pos.set(p.array.subarray(0, p.count * 3), o * 3);
    if (m) nrm.set(m.array.subarray(0, m.count * 3), o * 3);
    o += p.count;
    g.dispose();
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  return g;
}

/**
 * Leonardo style: every face becomes a solid panel with a polygonal
 * opening, joined to its neighbours along the original edges -- the
 * open-faced models Leonardo da Vinci drew for Luca Pacioli's
 * "De divina proportione" (1509).
 *
 * The frame width is ABSOLUTE, not a fraction of the face. That is the
 * whole design point, and math_art/leonardo_style.py records why: scaling
 * every face by one common factor makes the frame proportional to the
 * face, so on a solid with faces of different sizes -- a truncated
 * icosahedron, or anything Conway has operated on -- the big faces get
 * fat frames and the small ones thin, and the model looks unmade rather
 * than designed. For a frame of width w each face is inset by its own
 * factor (r - w) / r, with r that face's inradius.
 *
 * The add-on recovers r from area and corner count, exact for a regular
 * n-gon. Here r is the mean distance from the centroid to the edge lines,
 * which agrees with that for regular faces and, unlike the area formula,
 * stays meaningful on the star faces -- a pentagram's shoelace area is
 * not its filled area, so the area route would hand back a nonsense
 * inradius for 117 of these solids.
 */
export function buildLeonardo(rec, opts = {}) {
  const V = rec.geometry.vertices;
  const el = edgeLength(rec);
  const w = (opts.border ?? 0.16) * el;       // frame width, absolute
  const t = (opts.thickness ?? 0.05) * el;    // panel thickness

  const pos = [];
  const nrm = [];
  const col = [];
  const { compOf, count } = faceComponents(rec.geometry.faces);
  const byComponent = count > 1;
  const mode = opts.coloring || 'auto';
  const tmp = new THREE.Color();

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const crossV = (a, b) => [a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0]];

  function quad(a, b, c, d) {
    const n = crossV(sub(b, a), sub(d, a));
    const L = Math.hypot(n[0], n[1], n[2]) || 1;
    const u = [n[0] / L, n[1] / L, n[2] / L];
    for (const v of [a, b, c, a, c, d]) {
      pos.push(v[0], v[1], v[2]);
      nrm.push(u[0], u[1], u[2]);
      col.push(tmp.r, tmp.g, tmp.b);
    }
  }

  rec.geometry.faces.forEach((f, fi) => {
    const P = f.map((i) => V[i]);
    const n = P.length;
    const c = [0, 1, 2].map((k) => P.reduce((s, p) => s + p[k], 0) / n);
    const frame = planeBasisOf(P);
    if (!frame) return;

    // Inradius as the mean distance from the centroid to each edge line.
    let r = 0;
    for (let i = 0; i < n; i++) {
      const a = P[i], b = P[(i + 1) % n];
      const ab = sub(b, a);
      const L = Math.hypot(ab[0], ab[1], ab[2]) || 1;
      const ac = sub(c, a);
      const x = crossV(ab, ac);
      r += Math.hypot(x[0], x[1], x[2]) / L;
    }
    r /= n;
    // A frame wider than the face would invert the opening; clamp so the
    // panel degenerates to a nearly solid face instead of turning inside
    // out. Small faces on a mixed solid legitimately hit this.
    const s = Math.min(0.92, Math.max(0.08, (r - w) / (r || 1)));

    let hex = PLAIN;
    if (mode !== 'none') {
      hex = byComponent
        ? COMPOUND_PALETTE[compOf[fi] % COMPOUND_PALETTE.length]
        : (FACE_PALETTE[n] ?? PLAIN);
    }
    tmp.setHex(hex);

    const off = mul(frame.n, t / 2);
    const outerF = P.map((p) => add(p, off));
    const outerB = P.map((p) => sub(p, off));
    const inner = P.map((p) => add(c, mul(sub(p, c), s)));
    const innerF = inner.map((p) => add(p, off));
    const innerB = inner.map((p) => sub(p, off));

    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      quad(outerF[i], outerF[j], innerF[j], innerF[i]);   // front frame
      quad(innerB[i], innerB[j], outerB[j], outerB[i]);   // back frame
      quad(outerB[i], outerB[j], outerF[j], outerF[i]);   // outer rim
      quad(innerF[i], innerF[j], innerB[j], innerB[i]);   // opening wall
    }
  });

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return g;
}

// Newell normal + centroid for a face, reusing the tessellator's frame so
// the panel's "up" is the same direction the solid style shades against.
function planeBasisOf(pts) {
  const { normal } = tessellateFace(pts);
  return normal ? { n: normal } : null;
}

/** Radius of the smallest origin-centred sphere containing the solid. */
export function boundingRadius(rec) {
  let r = 0;
  for (const v of rec.geometry.vertices) {
    r = Math.max(r, Math.hypot(v[0], v[1], v[2]));
  }
  return r || 1;
}
