// The Dirac belt trick, computed in the browser.
//
// A port of math_art/belt_trick_generator.py, held to it: the site's
// gate runs the generator in CPython and this module in node over the
// same cases and requires agreement to 1e-9. So this file is a
// TRANSCRIPTION, not a reinterpretation -- read the generator's header
// for the mathematics, which is not repeated here -- and it keeps the
// generator's names, structure and ORDER OF OPERATIONS. Floating-point
// addition does not associate, so "the same formula, tidied" is a
// different number, and a different number is a failed parity check
// that then has to be told apart from a real bug.
//
// Three places where Python and JavaScript do not mean the same thing,
// and what is done about each:
//
//   round()        Python rounds half to EVEN and is correctly rounded.
//                  roundHalfEven() below. It decides which half-turn the
//                  width eases back to, and getting it wrong reverses a
//                  whole row of the strip.
//   sorted(set())  Python compares tuples numerically; JavaScript's
//                  default sort compares strings. The solids' normals
//                  are sorted, and their order is the belts' order, and
//                  the belts' order is their colours.
//   dict keys      The generator dedupes hull faces with round(x, 6) and
//                  geodesic vertices with round(c, 9). Those become
//                  distance tests here: the same result, without
//                  depending on how each language rounds a decimal.
//
// Nothing here depends on history: every call computes from scratch.
// That is what makes the cycle exactly periodic in the generator, and
// the port keeps it by construction.

export const TAU = 2.0 * Math.PI;
export const FULL_TURN = 2.0 * TAU;            // 720 degrees, in radians

export const KAPPA = 0.25;
export const SMOOTH = 0.5;
export const TAPER = 0.04;
export const RAMP = 0.5;
export const CLEAR = 1.15;
export const SMOOTHING = 0.05;
export const CAGE_TUBE = 0.004;

const Z = [0.0, 0.0, 1.0];
const M = [0.0, 1.0, 0.0];
const PHI = 0.5 * (1.0 + Math.sqrt(5.0));
const TRIBONACCI = 1.839286755214161;
const DEG = Math.PI / 180.0;                   // CPython's math.radians factor

export const SOLID_NAMES = ['TWO', 'TETRA', 'CUBE', 'OCTA', 'DODECA',
                            'ICOSA', 'SNUB', 'GEO'];

// Spin axis -> [n, m], verbatim from the generator.
export const AXES = {
  X: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
  Y: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
  Z: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
};

// ------------------------------------------------------------------
// Python semantics
// ------------------------------------------------------------------

/** Python's round(x): nearest integer, ties to even. */
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return (f % 2 === 0) ? f : f + 1;
}

/** Lexicographic numeric comparison, as Python compares tuples. */
function tupleCmp(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return a.length - b.length;
}

/** sorted(set(tuples)): dedupe by value, then numeric order. */
function sortedSet(list) {
  const seen = new Map();
  for (const t of list) {
    // Python's set treats 1 and 1.0 as one element and so does this:
    // JS numbers are all doubles.
    const key = t.join(',');
    if (!seen.has(key)) seen.set(key, t);
  }
  return [...seen.values()].sort(tupleCmp);
}

// ------------------------------------------------------------------
// Vectors and quaternions (w, x, y, z), as the generator's
// ------------------------------------------------------------------

export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]];
}
export function norm(a) { return Math.sqrt(dot(a, a)); }
export function unit(a) {
  const m = norm(a);
  return [a[0] / m, a[1] / m, a[2] / m];
}
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }

export function qmul(a, b) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz,
          aw * bx + ax * bw + ay * bz - az * by,
          aw * by + ay * bw + az * bx - ax * bz,
          aw * bz + az * bw + ax * by - ay * bx];
}

export function qrot(q, v) {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2.0 * (y * vz - z * vy);
  const ty = 2.0 * (z * vx - x * vz);
  const tz = 2.0 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty),
          vy + w * ty + (z * tx - x * tz),
          vz + w * tz + (x * ty - y * tx)];
}

export function qaxis(axis, angle) {
  const h = 0.5 * angle;
  const sh = Math.sin(h);
  return [Math.cos(h), sh * axis[0], sh * axis[1], sh * axis[2]];
}

// ------------------------------------------------------------------
// The solids
// ------------------------------------------------------------------

function signs(...coords) {
  let out = [[]];
  for (const c of coords) {
    const vals = c ? [c, -c] : [0.0];
    const next = [];
    for (const o of out) for (const v of vals) next.push([...o, v]);
    out = next;
  }
  return sortedSet(out);
}

function cyclic(pts) {
  const all = [];
  for (const p of pts) {
    for (let i = 0; i < 3; i++) {
      all.push([0, 1, 2].map((k) => p[(k + i) % 3]));
    }
  }
  return sortedSet(all);
}

const SOLID_DATA = {
  TETRA: [[[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
          [[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]],
  CUBE: [signs(1, 1, 1), cyclic([[1, 0, 0], [-1, 0, 0]])],
  OCTA: [cyclic([[1, 0, 0], [-1, 0, 0]]), signs(1, 1, 1)],
  DODECA: [signs(1, 1, 1).concat(cyclic(signs(0, 1.0 / PHI, PHI))),
           cyclic(signs(0, PHI, 1))],
  ICOSA: [cyclic(signs(0, PHI, 1)),
          signs(1, 1, 1).concat(cyclic(signs(0, 1.0 / PHI, PHI)))],
};
const SIDES = { TETRA: 3, CUBE: 4, OCTA: 3, DODECA: 5, ICOSA: 3 };

/** Sort a face's vertex indices anticlockwise about its normal. */
function orderFace(idx, verts, nf) {
  const k = idx.length;
  const c = [0, 1, 2].map((t) => {
    let s = 0;
    for (const i of idx) s += verts[i][t];
    return s / k;
  });
  const e1 = unit(sub(verts[idx[0]], c));
  const e2 = cross(nf, e1);
  const key = new Map(idx.map((i) => {
    const d = sub(verts[i], c);
    return [i, Math.atan2(dot(d, e2), dot(d, e1))];
  }));
  // Array.prototype.sort is stable, as Python's sort is.
  return idx.slice().sort((a, b) => key.get(a) - key.get(b));
}

function platonic(kind) {
  const name = kind === 'TWO' ? 'CUBE' : kind;
  const [verts0, normals0] = SOLID_DATA[name];
  const normals = normals0.map((n) => unit(n.map(Number)));
  const verts = verts0.map((v) => v.map(Number));
  const faces = [];
  for (const nf of normals) {
    let top = -Infinity;
    for (const v of verts) top = Math.max(top, dot(nf, v));
    const idx = [];
    verts.forEach((v, i) => { if (dot(nf, v) > top - 1e-9) idx.push(i); });
    if (idx.length !== SIDES[name]) {
      throw new Error('face normal does not match the vertex set');
    }
    faces.push(orderFace(idx, verts, nf));
  }
  return [verts, faces];
}

/**
 * Faces of a convex vertex set, as the supporting planes of vertex
 * triples. Faces are kept in the order they are first FOUND, over the
 * same i < j < k loop as the generator's, because that order becomes
 * the order of the belts and so their colours. Coplanar triples are
 * merged by comparing normals by distance rather than by rounded key.
 */
function hullFaces(verts) {
  const n = verts.length;
  const found = [];                     // {nf, idx:Set}
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        const a = verts[i], b = verts[j], c = verts[k];
        let nf = cross(sub(b, a), sub(c, a));
        const ln = norm(nf);
        if (ln < 1e-12) continue;
        nf = nf.map((x) => x / ln);
        let d = dot(nf, a);
        if (d < 0.0) { nf = nf.map((x) => -x); d = -d; }
        let outside = false;
        for (const v of verts) if (dot(nf, v) > d + 1e-9) { outside = true; break; }
        if (outside) continue;
        let face = null;
        for (const f of found) {
          if (Math.abs(f.nf[0] - nf[0]) < 1e-6 && Math.abs(f.nf[1] - nf[1]) < 1e-6
              && Math.abs(f.nf[2] - nf[2]) < 1e-6) { face = f; break; }
        }
        if (!face) { face = { nf, idx: new Set() }; found.push(face); }
        face.idx.add(i); face.idx.add(j); face.idx.add(k);
      }
    }
  }
  // Which vertex starts a face only rotates its cycle, and nothing
  // downstream depends on the rotation beyond rounding at 1e-16.
  return found.map((f) => orderFace([...f.idx].sort((p, q) => p - q),
                                    verts, unit(f.nf)));
}

function snubCube(mirror) {
  const t = TRIBONACCI;
  const base = [1.0, 1.0 / t, t];
  const perms = [[[0, 1, 2], 0], [[1, 2, 0], 0], [[2, 0, 1], 0],
                 [[0, 2, 1], 1], [[2, 1, 0], 1], [[1, 0, 2], 1]];
  const verts = [];
  for (const [perm, parity] of perms) {
    for (const sx of [1, -1]) {
      for (const sy of [1, -1]) {
        for (const sz of [1, -1]) {
          const minus = (sx < 0) + (sy < 0) + (sz < 0);
          if (minus % 2 !== parity) continue;
          let v = [sx * base[perm[0]], sy * base[perm[1]], sz * base[perm[2]]];
          if (mirror) v = [-v[0], v[1], v[2]];
          verts.push(v);
        }
      }
    }
  }
  return [verts, hullFaces(verts)];
}

function geodesic(freq) {
  const [iv, ifaces] = platonic('ICOSA');
  const verts = [];
  const vid = (pt) => {
    // Distinct vertices of a frequency-2 geodesic are ~0.3 apart on the
    // unit sphere, so 1e-7 merges only true duplicates.
    for (let i = 0; i < verts.length; i++) {
      const v = verts[i];
      if (Math.abs(v[0] - pt[0]) < 1e-7 && Math.abs(v[1] - pt[1]) < 1e-7
          && Math.abs(v[2] - pt[2]) < 1e-7) return i;
    }
    verts.push(pt);
    return verts.length - 1;
  };
  const faces = [];
  for (const fa of ifaces) {
    const A = iv[fa[0]], B = iv[fa[1]], C = iv[fa[2]];
    const grid = new Map();
    for (let i = 0; i <= freq; i++) {
      for (let j = 0; j <= freq - i; j++) {
        const k = freq - i - j;
        const pt = [0, 1, 2].map((t) => (i * A[t] + j * B[t] + k * C[t]) / freq);
        grid.set(i + ',' + j, vid(unit(pt)));
      }
    }
    const g = (i, j) => grid.get(i + ',' + j);
    for (let i = 0; i < freq; i++) {
      for (let j = 0; j < freq - i; j++) {
        faces.push([g(i, j), g(i + 1, j), g(i, j + 1)]);
        if (j < freq - i - 1) faces.push([g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)]);
      }
    }
  }
  return [verts, faces.map((f) => {
    const a = verts[f[0]], b = verts[f[1]], c = verts[f[2]];
    const nf = cross(sub(b, a), sub(c, a));
    return dot(nf, a) > 0.0 ? f : f.slice().reverse();
  })];
}

/**
 * The solid with its nearest face at distance `half`: [verts, faces,
 * belts], a belt being [u (face normal), w (width direction), rf (face
 * inradius), hf (face distance)].
 */
export function solid(kind, half = 0.15, n = Z, m = M, mirror = false, freq = 2) {
  let verts, faces;
  if (kind === 'SNUB') [verts, faces] = snubCube(mirror);
  else if (kind === 'GEO') [verts, faces] = geodesic(freq);
  else [verts, faces] = platonic(kind);
  let mind = Infinity;
  for (const f of faces) {
    const a = verts[f[0]], b = verts[f[1]], c = verts[f[2]];
    const nf = unit(cross(sub(b, a), sub(c, a)));
    mind = Math.min(mind, dot(nf, a));
  }
  const sc = half / mind;
  verts = verts.map((v) => v.map((c) => sc * c));
  const belts = [];
  for (const f of faces) {
    const a = verts[f[0]], b = verts[f[1]], c = verts[f[2]];
    const nf = unit(cross(sub(b, a), sub(c, a)));
    if (kind === 'TWO' && Math.abs(nf[2]) < 0.5) continue;
    const cen = [0, 1, 2].map((t) => {
      let s = 0;
      for (const i of f) s += verts[i][t];
      return s / f.length;
    });
    const hf = dot(nf, cen);
    const dm = dot(m, nf);
    let wm = [m[0] - dm * nf[0], m[1] - dm * nf[1], m[2] - dm * nf[2]];
    if (norm(wm) < 1e-6) {
      const dn = dot(n, nf);
      wm = [n[0] - dn * nf[0], n[1] - dn * nf[1], n[2] - dn * nf[2]];
    }
    const w = unit(wm);
    let rf = Infinity;
    for (let j = 0; j < f.length; j++) {
      const e = unit(sub(verts[f[(j + 1) % f.length]], verts[f[j]]));
      rf = Math.min(rf, norm(cross(sub(verts[f[j]], cen), e)));
    }
    belts.push([nf, w, rf, hf]);
  }
  return [verts, faces, belts];
}

export function circumradius(verts) {
  let r = 0;
  for (const v of verts) r = Math.max(r, norm(v));
  return r;
}

// ------------------------------------------------------------------
// Packing
// ------------------------------------------------------------------

export function arcGap(belts, lam, samples = 13) {
  const pts = belts.map(([u, w]) => {
    const arc = [];
    for (let j = 0; j < samples; j++) {
      const a = lam * (-1.0 + 2.0 * j / (samples - 1));
      const ca = Math.cos(a), sa = Math.sin(a);
      arc.push([ca * u[0] + sa * w[0], ca * u[1] + sa * w[1], ca * u[2] + sa * w[2]]);
    }
    return arc;
  });
  let best = Math.PI;
  for (let i = 0; i < belts.length; i++) {
    for (let j = i + 1; j < belts.length; j++) {
      if (dot(belts[i][0], belts[j][0]) < Math.cos(Math.min(Math.PI, 2.0 * lam + best))) continue;
      for (const p of pts[i]) {
        for (const q of pts[j]) {
          const d = Math.max(-1.0, Math.min(1.0, dot(p, q)));
          const a = Math.acos(d);
          if (a < best) best = a;
        }
      }
    }
  }
  return best;
}

export function attachRadius(half, width) {
  return Math.sqrt(half * half + 0.25 * width * width);
}

export function widthLimit(belts, half, thickness, margin = 0.02) {
  const need = Math.max(margin, 2.5 * thickness);
  let face = Infinity;
  for (const b of belts) face = Math.min(face, b[2]);
  face = 2.0 * face;
  let lo = 0.0, hi = face;
  for (let it = 0; it < 40; it++) {
    const mid = 0.5 * (lo + hi);
    const hs = attachRadius(half, mid);
    const lam = Math.asin(Math.min(1.0, mid / (2.0 * hs)));
    const gap = arcGap(belts, lam);
    if (2.0 * hs * Math.sin(0.5 * gap) >= need) lo = mid;
    else hi = mid;
  }
  const hs = attachRadius(half, lo);
  return [lo, arcGap(belts, Math.asin(Math.min(1.0, lo / (2.0 * hs))))];
}

// ------------------------------------------------------------------
// The frame field
// ------------------------------------------------------------------

export function profile(x, ramp = RAMP) {
  x = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
  let y;
  if (x < ramp) {
    const u = x / ramp;
    y = ramp * u * u * u * (1.0 - 0.5 * u);
  } else if (x <= 1.0 - ramp) {
    y = x - 0.5 * ramp;
  } else {
    const u = (1.0 - x) / ramp;
    y = 1.0 - ramp - ramp * u * u * u * (1.0 - 0.5 * u);
  }
  return y / (1.0 - ramp);
}

export function field(g, psi, n = Z, m = M) {
  const rho = 0.5 * Math.PI * g;
  const c = Math.cos(rho), s = Math.sin(rho);
  const cp = Math.cos(psi), sp = Math.sin(psi);
  const A = c * c + s * s * cp;
  const B = c * s * (1.0 - cp);
  const C = s * sp;
  const l = cross(n, m);
  return [A,
          B * (cp * m[0] + sp * l[0]) + C * n[0],
          B * (cp * m[1] + sp * l[1]) + C * n[1],
          B * (cp * m[2] + sp * l[2]) + C * n[2]];
}

// ------------------------------------------------------------------
// One belt
// ------------------------------------------------------------------

function resample(pts, vecs, count) {
  const n = pts.length;
  const arc = [0.0];
  for (let i = 1; i < n; i++) arc.push(arc[i - 1] + norm(sub(pts[i], pts[i - 1])));
  const total = arc[n - 1];
  const outP = [], outV = [];
  let j = 0;
  for (let c = 0; c < count; c++) {
    const s = total * c / (count - 1);
    while (j < n - 2 && arc[j + 1] < s) j++;
    const seg = arc[j + 1] - arc[j];
    const f = seg <= 0.0 ? 0.0 : Math.min(1.0, Math.max(0.0, (s - arc[j]) / seg));
    const a = pts[j], b = pts[j + 1];
    outP.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])]);
    const va = vecs[j], vb = vecs[j + 1];
    const v = [va[0] + f * (vb[0] - va[0]), va[1] + f * (vb[1] - va[1]), va[2] + f * (vb[2] - va[2])];
    outV.push(norm(v) > 1e-9 ? unit(v) : vecs[j]);
  }
  outP[count - 1] = pts[n - 1];
  outV[count - 1] = vecs[n - 1];
  return [outP, outV, total];
}

/**
 * Symmetric pentadiagonal solve, banded Gaussian elimination without
 * pivoting -- the generator's, including the order in which the back
 * substitution subtracts its two terms (its row dicts are iterated in
 * insertion order: the i+1 term, then the i+2).
 */
function solveBanded(diag, off1, off2, rhs) {
  const n = diag.length;
  // row i holds columns i-2 .. i+2 at offsets 0 .. 4
  const A = new Float64Array(n * 5);
  for (let i = 0; i < n; i++) {
    A[i * 5 + 2] = diag[i];
    if (i >= 1) A[i * 5 + 1] = off1[i - 1];
    if (i >= 2) A[i * 5 + 0] = off2[i - 2];
    if (i + 1 < n) A[i * 5 + 3] = off1[i];
    if (i + 2 < n) A[i * 5 + 4] = off2[i];
  }
  const at = (r, c) => r * 5 + (c - r + 2);
  const b = rhs.map((r) => Float64Array.from(r));
  for (let i = 0; i < n; i++) {
    const piv = A[at(i, i)];
    for (const r of [i + 1, i + 2]) {
      if (r >= n) continue;
      const fac = A[at(r, i)] / piv;
      if (fac === 0.0) continue;
      for (let c = i; c <= i + 2 && c < n; c++) {
        A[at(r, c)] = A[at(r, c)] - fac * A[at(i, c)];
      }
      for (let k = 0; k < b.length; k++) b[k][r] -= fac * b[k][i];
    }
  }
  const x = b.map(() => new Float64Array(n));
  for (let k = 0; k < b.length; k++) {
    for (let i = n - 1; i >= 0; i--) {
      let v = b[k][i];
      if (i + 1 < n) v -= A[at(i, i + 1)] * x[k][i + 1];
      if (i + 2 < n) v -= A[at(i, i + 2)] * x[k][i + 2];
      x[k][i] = v / A[at(i, i)];
    }
  }
  return x;
}

function smoothCurve(target, vecs, runOut, runIn, L) {
  const n = target.length;
  const [t, v, total] = resample(target, vecs, n);
  const h = total / (n - 1);
  const mu = (L / h) ** 4;
  const fixed = new Array(n).fill(false);
  for (let i = 0; i < n; i++) {
    const s = h * i;
    if (s <= runOut + 1e-12 || s >= total - runIn - 1e-12) fixed[i] = true;
  }
  fixed[0] = fixed[1] = fixed[n - 1] = fixed[n - 2] = true;
  const r0 = norm(t[0]), r1 = norm(t[n - 1]);
  t[1] = t[0].map((c) => c * (1.0 - h / r0));
  t[n - 2] = t[n - 1].map((c) => c * (1.0 + h / r1));
  const diag = new Array(n).fill(0.0);
  const off1 = new Array(n - 1).fill(0.0);
  const off2 = new Array(n - 2).fill(0.0);
  for (let i = 1; i < n - 1; i++) {
    diag[i - 1] += mu;
    diag[i] += 4.0 * mu;
    diag[i + 1] += mu;
    off1[i - 1] += -2.0 * mu;
    off1[i] += -2.0 * mu;
    off2[i - 1] += mu;
  }
  const rhs = [0, 1, 2].map(() => new Array(n).fill(0.0));
  for (let i = 0; i < n; i++) {
    if (!fixed[i]) {
      diag[i] += 1.0;
      for (let k = 0; k < 3; k++) rhs[k][i] += t[i][k];
    }
  }
  for (let i = 0; i < n; i++) {
    if (!fixed[i]) continue;
    const couplings = [
      [i - 2, i >= 2 ? off2[i - 2] : 0.0],
      [i - 1, i >= 1 ? off1[i - 1] : 0.0],
      [i + 1, i + 1 < n ? off1[i] : 0.0],
      [i + 2, i + 2 < n ? off2[i] : 0.0],
    ];
    for (const [j, coef] of couplings) {
      if (j >= 0 && j < n && !fixed[j]) {
        for (let k = 0; k < 3; k++) rhs[k][j] -= coef * t[i][k];
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if (!fixed[i]) continue;
    diag[i] = 1.0;
    for (let k = 0; k < 3; k++) rhs[k][i] = t[i][k];
    if (i >= 1) off1[i - 1] = 0.0;
    if (i + 1 < n) off1[i] = 0.0;
    if (i >= 2) off2[i - 2] = 0.0;
    if (i + 2 < n) off2[i] = 0.0;
  }
  const x = solveBanded(diag, off1, off2, rhs);
  const P = [];
  for (let i = 0; i < n; i++) P.push([x[0][i], x[1][i], x[2][i]]);
  return [P, v, t];
}

/**
 * One belt at half-turn psi: its drawn centre line P and, at each
 * sample, the unit width direction perpendicular to the line. The
 * cross-section is then P[i] + y * dir[i] for y across the width --
 * which is what the renderer does on the GPU, and what rowsOf() below
 * does for the parity check.
 */
export function beltLine(psi, u, w, rOut = 1.0, half = 0.15, width = 0.16,
                         reach = 0.5, ns = 160, n = Z, m = M,
                         rMin = null, smoothing = SMOOTHING,
                         kappa = KAPPA, smooth = SMOOTH, taper = TAPER) {
  if (rMin === null) rMin = half;
  const hs = half;
  const dr = hs - rOut;
  const sig1 = Math.min(1.0, (rOut - Math.max(rMin, hs)) / (rOut - hs));
  const sigA = sig1 * (1.0 - reach);
  const h = 1.0 / (ns - 1);
  const sigma = [], r = [], U = [], W = [];
  for (let i = 0; i < ns; i++) {
    const s = i * h;
    sigma.push(s);
    r.push(rOut + dr * s);
    const g = profile((s - sigA) / (sig1 - sigA));
    const q = field(g, psi, n, m);
    U.push(qrot(q, u));
    W.push(qrot(q, w));
  }
  const Ttan = [], tn = [];
  for (let i = 0; i < ns; i++) {
    if (i === 0 || i === ns - 1) { Ttan.push([0.0, 0.0, 0.0]); tn.push(0.0); continue; }
    const a = U[i - 1], b = U[i + 1];
    const d = [r[i] * (b[0] - a[0]) / (2.0 * h),
               r[i] * (b[1] - a[1]) / (2.0 * h),
               r[i] * (b[2] - a[2]) / (2.0 * h)];
    Ttan.push(d);
    tn.push(norm(d));
  }
  const Lw = new Array(ns);
  let i1 = -1;
  for (let i = 0; i < ns; i++) {
    if (tn[i] >= 1e-13) {
      const tt = Ttan[i], ti = tn[i];
      Lw[i] = cross(U[i], [tt[0] / ti, tt[1] / ti, tt[2] / ti]);
      i1 = i;
    } else {
      Lw[i] = W[i];
    }
  }
  if (i1 >= 0) {
    if (dot(Lw[i1], W[i1]) < 0.0) Lw[i1] = Lw[i1].map((c) => -c);
    for (let i = i1 - 1; i >= 0; i--) {
      if (tn[i] >= 1e-13 && dot(Lw[i], Lw[i + 1]) < 0.0) Lw[i] = Lw[i].map((c) => -c);
    }
  }
  const eps = kappa * Math.abs(dr);
  const Fv = U.map((ui, i) => cross(ui, W[i]));
  let alpha = [];
  let prev = 0.0;
  for (let i = 0; i < ns; i++) {
    const a = tn[i] * tn[i] / (tn[i] * tn[i] + eps * eps);
    const L = Lw[i], Wi = W[i];
    const v = [a * L[0] + (1.0 - a) * Wi[0],
               a * L[1] + (1.0 - a) * Wi[1],
               a * L[2] + (1.0 - a) * Wi[2]];
    let ang = Math.atan2(dot(v, Fv[i]), dot(v, W[i]));
    while (ang - prev > 0.5 * Math.PI) ang -= Math.PI;
    while (ang - prev < -0.5 * Math.PI) ang += Math.PI;
    alpha.push(ang);
    prev = ang;
  }
  if (smooth > 0.0) {
    const sg = smooth * width / Math.abs(dr);
    const halfk = Math.ceil(3.0 * sg / h);
    if (halfk >= 1) {
      let kern = [];
      for (let j = -halfk; j <= halfk; j++) {
        const t = j * h / sg;
        kern.push(Math.exp(-0.5 * (t * t)));
      }
      let ksum = 0;
      for (const kv of kern) ksum += kv;
      kern = kern.map((kv) => kv / ksum);
      const sm = [];
      for (let i = 0; i < ns; i++) {
        let acc = 0.0;
        for (let j = 0; j < kern.length; j++) {
          let idx = i + j - halfk;
          idx = idx < 0 ? 0 : (idx > ns - 1 ? ns - 1 : idx);
          acc += kern[j] * alpha[idx];
        }
        sm.push(acc);
      }
      alpha = sm;
    }
  }
  const kEnd = Math.PI * roundHalfEven(alpha[ns - 1] / Math.PI);
  for (let i = 0; i < ns; i++) {
    let tp = (sigma[i] - (sig1 - taper)) / taper;
    tp = tp < 0.0 ? 0.0 : (tp > 1.0 ? 1.0 : tp);
    tp = tp * tp * (3.0 - 2.0 * tp);
    alpha[i] = kEnd + (1.0 - tp) * (alpha[i] - kEnd);
    let t0 = sigma[i] / Math.max(sigA, 1e-9);
    t0 = t0 < 0.0 ? 0.0 : (t0 > 1.0 ? 1.0 : t0);
    alpha[i] *= t0 * t0 * (3.0 - 2.0 * t0);
  }
  const wv = new Array(ns), target = new Array(ns);
  for (let i = 0; i < ns; i++) {
    const ca = Math.cos(alpha[i]), sa = Math.sin(alpha[i]);
    const Wi = W[i], Fi = Fv[i], Ui = U[i], ri = r[i];
    wv[i] = [ca * Wi[0] + sa * Fi[0], ca * Wi[1] + sa * Fi[1], ca * Wi[2] + sa * Fi[2]];
    target[i] = [ri * Ui[0], ri * Ui[1], ri * Ui[2]];
  }
  const runOut = rOut - (rOut + dr * sigA);
  const runIn = (rOut + dr * sig1) - hs;
  let P, wd;
  if (smoothing > 0.0) [P, wd] = smoothCurve(target, wv, runOut, runIn, smoothing);
  else { P = target; wd = wv; }
  const dir = new Array(ns);
  for (let i = 0; i < ns; i++) {
    const a = P[Math.max(i - 1, 0)], b = P[Math.min(i + 1, ns - 1)];
    const t = unit(sub(b, a));
    const d = dot(wd[i], t);
    let wi = [wd[i][0] - d * t[0], wd[i][1] - d * t[1], wd[i][2] - d * t[2]];
    wi = norm(wi) > 1e-9 ? unit(wi) : wd[i];
    dir[i] = wi;
  }
  return { P, dir, width, tn };
}

/** The generator's cross-section rows, from a belt line. */
export function rowsOf(line, nlam = 11) {
  const rows = [];
  for (let i = 0; i < line.P.length; i++) {
    const P = line.P[i], d = line.dir[i];
    const row = [];
    for (let j = 0; j < nlam; j++) {
      const y = 0.5 * line.width * (-1.0 + 2.0 * j / (nlam - 1));
      row.push([P[0] + y * d[0], P[1] + y * d[1], P[2] + y * d[2]]);
    }
    rows.push(row);
  }
  return rows;
}

// ------------------------------------------------------------------
// The operator's decisions (the generator's prepare / diagnose)
// ------------------------------------------------------------------

export function minBendRadius(belts, half, width, reach, rMin, n = Z, m = M,
                              smoothing = SMOOTHING, ns = 120) {
  let best = 1e30;
  for (const [u, w, , hf] of belts) {
    for (const psi of [Math.PI, Math.PI / 3.0, 262.5 * DEG]) {
      const rows = rowsOf(beltLine(psi, u, w, 1.0, hf, width, reach, ns, n, m,
                                   rMin, smoothing), 3);
      const pts = rows.map((row) => row[1]);
      for (let i = 1; i < ns - 1; i++) {
        const ab = sub(pts[i], pts[i - 1]);
        const bc = sub(pts[i + 1], pts[i]);
        const ac = sub(pts[i + 1], pts[i - 1]);
        const cr = norm(cross(ab, bc));
        if (cr > 1e-30) best = Math.min(best, norm(ab) * norm(bc) * norm(ac) / (2.0 * cr));
      }
    }
  }
  return best;
}

export function prepare({ kind = 'TWO', size = 0.16, beltWidth = 0.0,
                          reach = 0.95, smoothing = SMOOTHING,
                          thickness = 0.012, spinAxis = 'Z', mirror = false,
                          freq = 2, scale = 1.0 } = {}) {
  let half = 0.5 * size;
  const [n, m] = AXES[spinAxis];
  let [verts, faces, belts] = solid(kind, half, n, m, mirror, freq);
  const cr = circumradius(verts);
  if (cr > 1e-9) {
    half *= half * Math.sqrt(3.0) / cr;
    [verts, faces, belts] = solid(kind, half, n, m, mirror, freq);
  }
  const rMin = CLEAR * circumradius(verts);
  const [limit] = widthLimit(belts, half, thickness);
  const width = beltWidth > 1e-6
    ? beltWidth
    : Math.min(limit, minBendRadius(belts, half, limit, reach, rMin, n, m, smoothing));
  const fit = scale / (1.0 + Math.max(0.5 * thickness, CAGE_TUBE));
  const solidThick = Math.min(thickness, 0.5 * width);
  return { kind, n, m, half, verts, faces, belts, rMin, limit, width, fit,
           solidThick, thickness, reach, smoothing };
}

/** Every belt at half-turn psi, as lines. The per-frame call. */
export function allBelts(prep, psi, ns = 160) {
  return prep.belts.map(([u, w, , hf]) =>
    beltLine(psi, u, w, 1.0, hf, prep.width, prep.reach, ns, prep.n, prep.m,
             prep.rMin, prep.smoothing));
}

export function cubeGap(half, width, belts) {
  const hs = attachRadius(half, width);
  const lam = Math.asin(Math.min(1.0, width / (2.0 * hs)));
  const gap = arcGap(belts, lam);
  return [gap, 2.0 * hs * Math.sin(0.5 * Math.max(0.0, gap))];
}

export function beltClearance(allRows, perBelt = 20) {
  let best = 1e30;
  const pts = allRows.map((rows) => {
    const step = Math.max(1, Math.floor(rows.length / perBelt));
    const out = [];
    for (let i = 0; i < rows.length; i += step) {
      const row = rows[i];
      out.push(row[0], row[Math.floor(row.length / 2)], row[row.length - 1]);
    }
    return out;
  });
  for (let a = 0; a < pts.length; a++) {
    for (let b = a + 1; b < pts.length; b++) {
      for (const p of pts[a]) {
        for (const q of pts[b]) {
          const dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
          const d = dx * dx + dy * dy + dz * dz;
          if (d < best) best = d;
        }
      }
    }
  }
  return Math.sqrt(best);
}

/** The operator's report: [level, measurements, warning or null]. */
export function diagnose(prep, allRows) {
  const { fit, width, belts, half } = prep;
  const [gap, dist] = cubeGap(half, width, belts);
  const bend = minBendRadius(belts, half, width, prep.reach, prep.rMin,
                             prep.n, prep.m, prep.smoothing) * fit;
  const clear = beltClearance(allRows) * fit;
  const meas = { belts: belts.length, width: width * fit, gap, dist: dist * fit,
                 bend, clear, limit: prep.limit * fit };
  if (width > prep.limit) {
    return ['WARNING', meas, `wider than ${(prep.limit * fit).toFixed(3)}, so the `
      + 'belts meet one another at the solid; narrow it, or use fewer belts'];
  }
  if (belts.length > 1 && clear < prep.thickness * fit) {
    return ['WARNING', meas, `belts come within ${clear.toFixed(3)} of one another `
      + 'at this turn; lower Smoothing or narrow the belt'];
  }
  if (bend < width * fit) {
    return ['WARNING', meas, 'the belt is wider than its tightest bend and will '
      + 'crease there; raise Smoothing or narrow it'];
  }
  if (prep.solidThick < prep.thickness) {
    return ['WARNING', meas, 'so many faces that the belts came out narrow; '
      + `thinned to ${(prep.solidThick * fit).toFixed(3)} to keep them twice as `
      + 'wide as they are thick.  For proper belts enlarge the solid, thin the '
      + 'belt, or use fewer faces'];
  }
  return ['INFO', meas, null];
}

/** Evenly spaced hues, as the generator's _colour. */
export function colour(i) {
  const h = (i * 0.618033988749895) % 1.0;
  const sat = 0.82;
  const v = i % 2 === 0 ? 0.98 : 0.86;
  const k = Math.floor(h * 6.0);
  const f = h * 6.0 - k;
  const p = v * (1.0 - sat);
  const q = v * (1.0 - sat * f);
  const t = v * (1.0 - sat * (1.0 - f));
  return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][k % 6];
}
