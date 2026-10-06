// Triply-periodic minimal surfaces, by their nodal approximations.
//
// A port of math_art/minsurf/tpms.py: the published level-set formulas,
// and the marching-tetrahedra extraction that turns one of them into a
// mesh. tests/web/test_tpms.mjs runs that engine and checks this
// against it, vertex by vertex.
//
// WHAT A NODAL SURFACE IS. A triply-periodic minimal surface repeats in
// all three directions like a crystal, and most of them have no formula
// in closed form. What they do have is a short trigonometric expression
// whose zero set -- the places where it equals nothing -- lies very
// close to the true surface. The gyroid's is
//
//     sin x cos y + sin y cos z + sin z cos x = 0,
//
// three terms, and the shape that comes out is the one Alan Schoen
// found in 1970. These are approximations, not the surfaces themselves,
// and the module says so wherever it matters.
//
// EVERY FORMULA HERE IS PUBLISHED. None is invented; the sources are
// named per surface in the Python module's header, chiefly Schwarz
// (1890), Schoen (1970), von Schnering and Nesper (1991), Wohlgemuth et
// al. (2001) and the Fisher et al. (2023) compilation.

const { sin, cos, sinh, cos: _c } = Math;
import { weld, orient, solidify } from './stl.js';

export const TAU = 2 * Math.PI;

const C = Math.cos, S = Math.sin;

// ---------------------------------------------------------------- fields

const f_p = (x, y, z) => C(x) + C(y) + C(z);

const f_d = (x, y, z) => (S(x) * S(y) * S(z) + S(x) * C(y) * C(z)
                        + C(x) * S(y) * C(z) + C(x) * C(y) * S(z));

const f_g = (x, y, z) => S(x) * C(y) + S(y) * C(z) + S(z) * C(x);

const f_neovius = (x, y, z) => 3 * (C(x) + C(y) + C(z)) + 4 * C(x) * C(y) * C(z);

const f_iwp = (x, y, z) => (2 * (C(x) * C(y) + C(y) * C(z) + C(z) * C(x))
                          - (C(2 * x) + C(2 * y) + C(2 * z)));

const f_frd = (x, y, z) => (4 * C(x) * C(y) * C(z)
  - (C(2 * x) * C(2 * y) + C(2 * y) * C(2 * z) + C(2 * z) * C(2 * x)));

const f_lidinoid = (x, y, z) => (0.5 * (S(2 * x) * C(y) * S(z)
  + S(2 * y) * C(z) * S(x) + S(2 * z) * C(x) * S(y)
  - C(2 * x) * C(2 * y) - C(2 * y) * C(2 * z) - C(2 * z) * C(2 * x)) + 0.15);

const f_splitp = (x, y, z) => (1.1 * (S(2 * x) * S(z) * C(y)
    + S(2 * y) * S(x) * C(z) + S(2 * z) * S(y) * C(x))
  - 0.2 * (C(2 * x) * C(2 * y) + C(2 * y) * C(2 * z) + C(2 * z) * C(2 * x))
  - 0.4 * (C(2 * x) + C(2 * y) + C(2 * z)));

const f_scherk_tower = (x, y, z) => S(z) - sinh(x) * sinh(y);

const f_octo = (x, y, z) => {
  const cx = C(x), cy = C(y), cz = C(z);
  return 0.6 * (cx * cy + cy * cz + cz * cx) - 0.4 * (cx + cy + cz) + 0.25;
};

const f_fk_s = (x, y, z) => (C(2 * x) * S(y) * C(z) + C(x) * C(2 * y) * S(z)
                           + S(x) * C(y) * C(2 * z));

const f_fk_cs = (x, y, z) => (C(2 * x) + C(2 * y) + C(2 * z)
  + 2 * (S(3 * x) * S(2 * y) * C(z) + C(x) * S(3 * y) * S(2 * z)
       + S(2 * x) * C(y) * S(3 * z))
  + 2 * (S(2 * x) * C(3 * y) * S(z) + S(x) * S(2 * y) * C(3 * z)
       + C(3 * x) * S(y) * S(2 * z)));

const f_fk_y = (x, y, z) => (C(x) * C(y) * C(z) + S(x) * S(y) * S(z)
  + S(2 * x) * S(y) + S(2 * y) * S(z) + S(x) * S(2 * z)
  + C(x) * S(2 * y) + C(y) * S(2 * z) + S(2 * x) * C(z));

const f_fk_pmy = (x, y, z) => (2 * C(x) * C(y) * C(z)
  + S(2 * x) * S(y) + S(2 * y) * S(z) + S(x) * S(2 * z));

const f_fk_cpmy = (x, y, z) => (-2 * C(x) * C(y) * C(z)
  + S(2 * x) * S(y) + S(2 * y) * S(z) + S(x) * S(2 * z));

const f_fk_cy = (x, y, z) => (-S(x) * S(y) * S(z)
  + S(2 * x) * S(y) + S(2 * y) * S(z) + S(x) * S(2 * z)
  - C(x) * C(y) * C(z)
  + S(2 * x) * C(z) + C(x) * S(2 * y) + C(y) * S(2 * z));

const f_cd = (x, y, z) => (C(3 * x + y) * C(z) - S(3 * x - y) * S(z)
  + C(x + 3 * y) * C(z) + S(x - 3 * y) * S(z)
  + C(x - y) * C(3 * z) - S(x + y) * S(3 * z));

const f_cg = (x, y, z) => (3 * (S(x) * C(y) + S(y) * C(z) + C(x) * S(z))
  + 2 * (S(3 * x) * C(y) + S(3 * y) * C(z) + C(x) * S(3 * z))
  - 2 * (S(x) * C(3 * y) + S(y) * C(3 * z) + C(3 * x) * S(z)));

const f_gprime = (x, y, z) => (S(2 * x) * C(y) * S(z) + S(x) * S(2 * y) * C(z)
  + C(x) * S(y) * S(2 * z) + 0.32);

const f_dprime = (x, y, z) => (0.5 * (C(x) * C(y) * C(z) + C(x) * S(y) * S(z)
    + S(x) * C(y) * S(z) + S(x) * S(y) * C(z))
  - 0.5 * (S(2 * x) * S(2 * y) + S(2 * y) * S(2 * z) + S(2 * z) * S(2 * x))
  - 0.2);

const f_k = (x, y, z) => {
  const cx = C(x), cy = C(y), cz = C(z);
  return (0.3 * (cx + cy + cz) + 0.3 * (cx * cy + cy * cz + cz * cx)
        - 0.4 * (C(2 * x) + C(2 * y) + C(2 * z)) + 0.2);
};

const f_ci2y = (x, y, z) => (2 * (S(2 * x) * C(y) * S(z)
    + S(x) * S(2 * y) * C(z) + C(x) * S(y) * S(2 * z))
  + C(2 * x) * C(2 * y) + C(2 * y) * C(2 * z) + C(2 * x) * C(2 * z));

const f_frd2 = (x, y, z) => {
  const c2x = C(2 * x), c2y = C(2 * y), c2z = C(2 * z);
  return (8 * C(x) * C(y) * C(z) + c2x * c2y * c2z
        - (c2x * c2y + c2y * c2z + c2z * c2x));
};

/** The surfaces, in the generator's order: [label, field, triplyPeriodic].
 *  The names are the engine's own, ASCII and all -- they are what the
 *  parity test holds us to. `prettyLabel` is for showing a reader. */
export const TPMS = {
  P: ['Schwarz P', f_p, true],
  D: ['Schwarz D', f_d, true],
  G: ['Gyroid', f_g, true],
  NEOVIUS: ['Neovius', f_neovius, true],
  IWP: ['Schoen I-WP', f_iwp, true],
  FRD: ['Schoen F-RD', f_frd, true],
  LIDINOID: ['Lidinoid', f_lidinoid, true],
  SPLITP: ['Split P', f_splitp, true],
  SCHERKT: ['Scherk Tower (singly periodic)', f_scherk_tower, false],
  OCTO: ['Schoen O,C-TO (nodal approximation)', f_octo, true],
  FK_S: ['Fischer-Koch S (nodal approximation)', f_fk_s, true],
  FK_CS: ['Fischer-Koch C(S) (nodal approximation)', f_fk_cs, true],
  FK_Y: ['Fischer-Koch Y (nodal approximation)', f_fk_y, true],
  FK_PMY: ['Fischer-Koch +-Y (nodal approximation)', f_fk_pmy, true],
  FK_CPMY: ['Fischer-Koch C(+-Y) (nodal approximation)', f_fk_cpmy, true],
  FK_CY: ['Fischer-Koch C(Y) (nodal approximation)', f_fk_cy, true],
  CD: ['Complementary D (nodal approximation)', f_cd, true],
  CG: ['Complementary Gyroid (nodal approximation)', f_cg, true],
  GPRIME: ["G' Alternating Gyroid (nodal approximation)", f_gprime, true],
  DPRIME: ["D' (nodal approximation)", f_dprime, true],
  KSURF: ['Karcher K (nodal approximation)', f_k, true],
  CI2Y: ['C(I2-Y**) Rod Packing (nodal approximation)', f_ci2y, true],
  FRD2: ['Schoen F-RD (Wohlgemuth variant, nodal)', f_frd2, true],
};

// ------------------------------------------------- marching tetrahedra

const CUBE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
              [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const TETS = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
              [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];
// one corner on its own side: [lone, the other three]
const ONE = { 1: [0, [1, 2, 3]], 2: [1, [0, 2, 3]], 4: [2, [0, 1, 3]],
              8: [3, [0, 1, 2]], 14: [0, [1, 2, 3]], 13: [1, [0, 2, 3]],
              11: [2, [0, 1, 3]], 7: [3, [0, 1, 2]] };
// two against two
const TWO = { 3: [[0, 1], [2, 3]], 5: [[0, 2], [1, 3]], 9: [[0, 3], [1, 2]],
              6: [[1, 2], [0, 3]], 10: [[1, 3], [0, 2]], 12: [[2, 3], [0, 1]] };

const BIG = 1e30;

/** Whether each (tet, case) emits triangles wound against the field's
 *  gradient. Worked out once on an exactly linear field, where the
 *  crossing polygon is square to the gradient, so the answer is
 *  combinatorial and no sliver triangle can upset it. */
const ORIENT = (() => {
  const flags = new Map();
  const solve3 = (M, b) => {
    // Cramer's rule; M is 3x3 as rows
    const det = (m) => (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                      - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                      + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));
    const d = det(M);
    const out = [];
    for (let c = 0; c < 3; c++) {
      const Mc = M.map((row, r) => row.map((v, k) => (k === c ? b[r] : v)));
      out.push(det(Mc) / d);
    }
    return out;
  };
  for (let ti = 0; ti < TETS.length; ti++) {
    const P = TETS[ti].map((i) => CUBE[i]);
    const M = [1, 2, 3].map((i) => [P[i][0] - P[0][0], P[i][1] - P[0][1],
                                    P[i][2] - P[0][2]]);
    for (const cd of [...Object.keys(ONE), ...Object.keys(TWO)].map(Number)) {
      const f = [0, 1, 2, 3].map((i) => ((cd >> i) & 1 ? -1 : 1));
      const g = solve3(M, [f[1] - f[0], f[2] - f[0], f[3] - f[0]]);
      const x = (ci, cj) => {
        const t = f[ci] / (f[ci] - f[cj]);
        return [0, 1, 2].map((k) => P[ci][k] + t * (P[cj][k] - P[ci][k]));
      };
      let p0, p1, p2;
      if (ONE[cd]) {
        const [lone, o] = ONE[cd];
        p0 = x(lone, o[0]); p1 = x(lone, o[1]); p2 = x(lone, o[2]);
      } else {
        const [[n0, n1], [q0, q1]] = TWO[cd];
        p0 = x(n0, q0); p1 = x(n0, q1); p2 = x(n1, q1);
      }
      const u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
      const v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                 u[0] * v[1] - u[1] * v[0]];
      flags.set(`${ti},${cd}`, n[0] * g[0] + n[1] * g[1] + n[2] * g[2] < 0);
    }
  }
  return flags;
})();

/**
 * The zero level set of `field` on a grid over the box.
 *
 * Crossings are identified by the LATTICE EDGE they lie on rather than
 * by position, which is what makes the result watertight: the same
 * crossing reached from two tetrahedra gives the same key, where
 * interpolating a->b in one and b->a in the other gives t and 1-t --
 * equal in exact arithmetic but not in floating point.
 *
 * Returns { positions, indices } with triangles wound along the
 * gradient.
 */
export function marchingTets(field, boxMin, boxMax, res, nudge = 1e-9) {
  const nx = res[0] + 1, ny = res[1] + 1, nz = res[2] + 1;
  const lin = (a, b, n) => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = n === 1 ? a : a + (b - a) * i / (n - 1);
    return out;
  };
  const xs = lin(boxMin[0], boxMax[0], nx);
  const ys = lin(boxMin[1], boxMax[1], ny);
  const zs = lin(boxMin[2], boxMax[2], nz);
  const npt = nx * ny * nz;

  // sample once; a value sitting exactly on the surface is displaced,
  // or the crossing it produces is degenerate
  const val = new Float64Array(npt);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const base = (i * ny + j) * nz;
      for (let k = 0; k < nz; k++) {
        let v = field(xs[i], ys[j], zs[k]);
        if (!Number.isFinite(v)) v = v < 0 ? -BIG : BIG;
        val[base + k] = Math.abs(v) < nudge ? nudge : v;
      }
    }
  }

  // triangles, as triples of edge keys
  const triKeys = [];
  const corner = new Int32Array(8);
  const fv = new Float64Array(8);
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let k = 0; k < nz - 1; k++) {
        const g0 = (i * ny + j) * nz + k;
        let anyNeg = false, allNeg = true;
        for (let c = 0; c < 8; c++) {
          const o = CUBE[c];
          const idx = g0 + (o[0] * ny + o[1]) * nz + o[2];
          corner[c] = idx;
          const v = val[idx];
          fv[c] = v;
          if (v < 0) anyNeg = true; else allNeg = false;
        }
        if (!anyNeg || allNeg) continue;          // no crossing in this cube
        for (let ti = 0; ti < 6; ti++) {
          const tet = TETS[ti];
          let code = 0;
          for (let c = 0; c < 4; c++) if (fv[tet[c]] < 0) code |= 1 << c;
          if (code === 0 || code === 15) continue;
          const key = (ci, cj) => {
            const a = corner[tet[ci]], b = corner[tet[cj]];
            return a < b ? a * npt + b : b * npt + a;
          };
          const flip = ORIENT.get(`${ti},${code}`);
          if (ONE[code]) {
            const [lone, o] = ONE[code];
            let p0 = key(lone, o[0]), p1 = key(lone, o[1]), p2 = key(lone, o[2]);
            if (flip) { const t = p1; p1 = p2; p2 = t; }
            triKeys.push(p0, p1, p2);
          } else {
            const [[n0, n1], [q0, q1]] = TWO[code];
            let a = key(n0, q0), b = key(n0, q1);
            let c = key(n1, q1), d = key(n1, q0);
            if (flip) { const t = b; b = d; d = t; }
            triKeys.push(a, b, c, a, c, d);
          }
        }
      }
    }
  }
  if (!triKeys.length) return { positions: new Float32Array(0), indices: new Uint32Array(0) };

  // one vertex per distinct lattice edge, in ascending key order
  const keys = Array.from(new Set(triKeys)).sort((a, b) => a - b);
  const slot = new Map();
  keys.forEach((k, i) => slot.set(k, i));
  const pos = new Float64Array(keys.length * 3);
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const mn = Math.floor(k / npt), mx = k - mn * npt;
    const ia = Math.floor(mn / (ny * nz)), ra = mn - ia * ny * nz;
    const ja = Math.floor(ra / nz), ka = ra - ja * nz;
    const ib = Math.floor(mx / (ny * nz)), rb = mx - ib * ny * nz;
    const jb = Math.floor(rb / nz), kb = rb - jb * nz;
    const va = val[mn], vb = val[mx];
    const t = va / (va - vb);
    pos[i * 3] = xs[ia] + t * (xs[ib] - xs[ia]);
    pos[i * 3 + 1] = ys[ja] + t * (ys[jb] - ys[ja]);
    pos[i * 3 + 2] = zs[ka] + t * (zs[kb] - zs[ka]);
  }
  let tris = new Int32Array(triKeys.length);
  for (let i = 0; i < triKeys.length; i++) tris[i] = slot.get(triKeys[i]);

  // Cosmetic second weld: distinct lattice edges can still cross at
  // nearly the same point -- systematically for the fields whose zero
  // set runs through the lattice points, where the nudge leaves a whole
  // fan of edges collapsed onto one. Merging those and dropping the
  // slivers they make cannot reopen the surface, because the edge keys
  // have already made it watertight.
  const span = Math.max(Math.max(...boxMax) - Math.min(...boxMin), 1);
  const eps = span * 1e-6;
  const qkey = new Map();
  const rep = new Int32Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    const k = `${Math.round(pos[i * 3] / eps)},${Math.round(pos[i * 3 + 1] / eps)},`
            + `${Math.round(pos[i * 3 + 2] / eps)}`;
    const hit = qkey.get(k);
    if (hit === undefined) { qkey.set(k, i); rep[i] = i; } else { rep[i] = hit; }
  }
  // keep only the vertices something still points at, in their order
  const used = new Int32Array(keys.length).fill(-1);
  const outPos = [];
  const outTri = [];
  for (let t = 0; t < tris.length; t += 3) {
    const a = rep[tris[t]], b = rep[tris[t + 1]], c = rep[tris[t + 2]];
    if (a === b || b === c || a === c) continue;        // sliver
    for (const v of [a, b, c]) {
      if (used[v] < 0) {
        used[v] = outPos.length / 3;
        outPos.push(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
      }
    }
    outTri.push(used[a], used[b], used[c]);
  }
  return {
    positions: Float32Array.from(outPos),
    indices: Uint32Array.from(outTri),
  };
}

// --------------------------------------------------------- the surfaces

/** One unit cell of a triply-periodic field, over [-pi, pi]^3. */
export function unitCell(kind, resPerCell, offset = 0) {
  const [, field, triply] = TPMS[kind];
  if (!triply) throw new Error(`${kind} is not triply periodic`);
  const f = offset === 0 ? field
    : (x, y, z) => field(x, y, z) - offset;
  const half = TAU / 2;
  const r = Math.max(4, Math.round(resPerCell));
  return marchingTets(f, [-half, -half, -half], [half, half, half], [r, r, r]);
}

/**
 * A block of cx x cy x cz cells.
 *
 * The fields are exactly 2*pi-periodic, so every cell holds a translate
 * of the first -- an identity, not an approximation -- and the block is
 * built by copying one extracted cell rather than marching the whole
 * volume. The generator measured that at 48 times faster for a 5x5x5
 * gyroid, and the saving is the same here.
 */
export function block(kind, cells, resPerCell, scale = 2, offset = 0) {
  const [cx, cy, cz] = Array.isArray(cells) ? cells : [cells, cells, cells];
  const cell = unitCell(kind, resPerCell, offset);
  const nv = cell.positions.length / 3;
  const nt = cell.indices.length / 3;
  const copies = cx * cy * cz;

  const V = new Float64Array(copies * nv * 3);
  const T = new Int32Array(copies * nt * 3);
  let c = 0;
  for (let i = 0; i < cx; i++) {
    for (let j = 0; j < cy; j++) {
      for (let k = 0; k < cz; k++) {
        const ox = (i - 0.5 * (cx - 1)) * TAU;
        const oy = (j - 0.5 * (cy - 1)) * TAU;
        const oz = (k - 0.5 * (cz - 1)) * TAU;
        const vb = c * nv;
        for (let v = 0; v < nv; v++) {
          V[(vb + v) * 3] = cell.positions[v * 3] + ox;
          V[(vb + v) * 3 + 1] = cell.positions[v * 3 + 1] + oy;
          V[(vb + v) * 3 + 2] = cell.positions[v * 3 + 2] + oz;
        }
        const tb = c * nt * 3;
        for (let t = 0; t < nt * 3; t++) T[tb + t] = cell.indices[t] + vb;
        c++;
      }
    }
  }

  // Weld the seams, and only the seams. Neighbouring cells carry the
  // crossings on their shared face twice, and a duplicated seam vertex
  // splits the surface there. Which planes are seams depends on the
  // parity of the cell count -- an odd count puts cell centres on whole
  // periods and the boundaries on half periods, an even count the other
  // way about -- so testing 2V/TAU against a whole number covers both.
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < V.length; i++) { if (V[i] < lo) lo = V[i]; if (V[i] > hi) hi = V[i]; }
  const tol = Math.max(hi - lo, 1) * 1e-6;
  const remap = new Int32Array(copies * nv);
  const seam = new Map();
  for (let v = 0; v < copies * nv; v++) {
    remap[v] = v;
    let onSeam = false;
    for (let a = 0; a < 3; a++) {
      const d = 2 * V[v * 3 + a] / TAU;
      if (Math.abs(d - Math.round(d)) < 1e-7) { onSeam = true; break; }
    }
    if (!onSeam) continue;
    const k = `${Math.round(V[v * 3] / tol)},${Math.round(V[v * 3 + 1] / tol)},`
            + `${Math.round(V[v * 3 + 2] / tol)}`;
    const hit = seam.get(k);
    if (hit === undefined) seam.set(k, v); else remap[v] = hit;
  }

  const s = scale / TAU;                 // one period -> `scale` units
  const used = new Int32Array(copies * nv).fill(-1);
  const outPos = [];
  const outTri = [];
  for (let t = 0; t < T.length; t += 3) {
    const a = remap[T[t]], b = remap[T[t + 1]], d = remap[T[t + 2]];
    if (a === b || b === d || a === d) continue;
    for (const v of [a, b, d]) {
      if (used[v] < 0) {
        used[v] = outPos.length / 3;
        outPos.push(V[v * 3] * s, V[v * 3 + 1] * s, V[v * 3 + 2] * s);
      }
    }
    outTri.push(used[a], used[b], used[d]);
  }
  return {
    positions: Float32Array.from(outPos),
    indices: Uint32Array.from(outTri),
    cells: [cx, cy, cz],
  };
}

/** The engine's label, typeset: "+-Y" reads better as "±Y". */
export function prettyLabel(kind) {
  const [label] = TPMS[kind];
  return label.replace(/\+-/g, '±');
}

// ------------------------------------------------- clipping to a ball

/**
 * Clip a mesh to the ball of radius `radius` about the origin.
 *
 * Sutherland-Hodgman against the sphere, one face at a time: corners
 * inside are kept, and where an edge crosses, the crossing is SOLVED
 * for rather than approximated -- |a + t(b - a)| = r is a quadratic in
 * t. So the cut edge lies on the sphere and comes out smooth, instead
 * of following the face boundaries in a staircase the way dropping
 * whole faces would.
 *
 * Clipping opens an edge where a periodic surface had none, which is
 * what the rim tube is for.
 *
 * Faces go in and come out as arrays of indices: the clip turns some
 * triangles into quadrilaterals and pentagons.
 */
export function clipToSphere(positions, faces, radius) {
  const r = Number(radius);
  const nv = positions.length / 3;
  if (!(r > 0) || !nv) return { positions, faces };
  const d = new Float64Array(nv);
  for (let i = 0; i < nv; i++) {
    d[i] = Math.hypot(positions[i * 3], positions[i * 3 + 1],
                      positions[i * 3 + 2]) - r;
  }
  const outV = [];
  for (let i = 0; i < nv * 3; i++) outV.push(positions[i]);
  const cache = new Map();

  const crossing = (i, j) => {
    const lo = Math.min(i, j), hi = Math.max(i, j);
    const k = lo + ',' + hi;
    const hit = cache.get(k);
    if (hit !== undefined) return hit;
    const a = [positions[lo * 3], positions[lo * 3 + 1], positions[lo * 3 + 2]];
    const b = [positions[hi * 3], positions[hi * 3 + 1], positions[hi * 3 + 2]];
    const e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const qa = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    const qb = 2 * (a[0] * e[0] + a[1] * e[1] + a[2] * e[2]);
    const qc = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] - r * r;
    let t = 0.5;
    if (Math.abs(qa) > 1e-30) {
      const disc = qb * qb - 4 * qa * qc;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        for (const cand of [(-qb - sq) / (2 * qa), (-qb + sq) / (2 * qa)]) {
          if (cand >= -1e-9 && cand <= 1 + 1e-9) {
            t = Math.min(1, Math.max(0, cand));
            break;
          }
        }
      }
    }
    outV.push(a[0] + t * e[0], a[1] + t * e[1], a[2] + t * e[2]);
    const idx = outV.length / 3 - 1;
    cache.set(k, idx);
    return idx;
  };

  const outF = [];
  for (const f of faces) {
    const n = f.length;
    if (n < 3) continue;
    let allIn = true, allOut = true;
    for (const k of f) { if (d[k] <= 0) allOut = false; else allIn = false; }
    if (allIn) { outF.push(f.slice()); continue; }
    if (allOut) continue;
    const poly = [];
    for (let k = 0; k < n; k++) {
      const a = f[k], b = f[(k + 1) % n];
      const ain = d[a] <= 0, bin = d[b] <= 0;
      if (ain) poly.push(a);
      if (ain !== bin) poly.push(crossing(a, b));
    }
    // a corner sitting exactly on the sphere can be produced twice
    const clean = [];
    for (const k of poly) {
      if (!clean.length || k !== clean[clean.length - 1]) clean.push(k);
    }
    if (clean.length > 1 && clean[0] === clean[clean.length - 1]) clean.pop();
    if (clean.length >= 3) outF.push(clean);
  }

  const used = [...new Set(outF.flat())].sort((a, b) => a - b);
  const remap = new Map(used.map((k, i) => [k, i]));
  const pos = new Float32Array(used.length * 3);
  used.forEach((k, i) => {
    pos[i * 3] = outV[k * 3];
    pos[i * 3 + 1] = outV[k * 3 + 1];
    pos[i * 3 + 2] = outV[k * 3 + 2];
  });
  return { positions: pos, faces: outF.map((f) => f.map((k) => remap.get(k))) };
}

/** Fan-split polygons into triangles. */
export function triangulate(faces) {
  const out = [];
  for (const f of faces) {
    for (let k = 1; k < f.length - 1; k++) out.push(f[0], f[k], f[k + 1]);
  }
  return Uint32Array.from(out);
}

/** Triangles as an array of index triples, which is what the clipper
 *  and the rim walker take. */
export function facesOf(indices) {
  const out = [];
  for (let i = 0; i < indices.length; i += 3) {
    out.push([indices[i], indices[i + 1], indices[i + 2]]);
  }
  return out;
}

// ------------------------------------------------------ the rim curve

/** The open edge of a mesh, as chains of vertex indices.
 *
 *  Edges used by exactly one face are the boundary. The walk assumes
 *  nothing about manifoldness -- a rim vertex can carry four boundary
 *  edges rather than two -- and takes whatever chains it finds, open
 *  or closed. */
export function boundaryIndexLoops(faces) {
  const count = new Map();
  for (const f of faces) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      count.set(a < b ? a + ',' + b : b + ',' + a,
                (count.get(a < b ? a + ',' + b : b + ',' + a) || 0) + 1);
    }
  }
  const rim = [];
  for (const [k, c] of count) {
    if (c !== 1) continue;
    const parts = k.split(',');
    rim.push([Number(parts[0]), Number(parts[1])]);
  }
  if (!rim.length) return [];

  const adj = new Map();
  for (const pair of rim) {
    const a = pair[0], b = pair[1];
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push(b);
    adj.get(b).push(a);
  }
  const used = new Set();
  const take = (p, q) => {
    const k = p < q ? p + ',' + q : q + ',' + p;
    if (used.has(k)) return false;
    used.add(k);
    return true;
  };
  const chains = [];
  for (const pair of rim) {
    const a0 = pair[0], b0 = pair[1];
    if (!take(a0, b0)) continue;
    const chain = [a0, b0];
    for (;;) {
      const cur = chain[chain.length - 1];
      let nxt = null;
      for (const cand of adj.get(cur) || []) {
        if (take(cur, cand)) { nxt = cand; break; }
      }
      if (nxt === null) break;
      chain.push(nxt);
      if (nxt === chain[0]) break;
    }
    if (chain.length < 4) continue;
    const closed = chain[chain.length - 1] === chain[0];
    if (closed) chain.pop();
    chains.push({ idx: chain, closed });
  }
  return chains;
}

const TAUBIN_LAMBDA = 0.5;
const TAUBIN_MU = -0.53;
export const RIM_SMOOTH_DEFAULT = 3;

/** Smooth a polyline WITHOUT shrinking it.
 *
 *  A plain Laplacian pass is a curve-shortening flow, and on a rim that
 *  wraps a curved surface it walks the curve off the edge it is meant
 *  to trace. Taubin's fix alternates a positive step with a slightly
 *  larger negative one, which cancels the shrinkage to first order
 *  while still attenuating the grid staircase. No point may then travel
 *  more than a quarter of the median spacing, which is what holds a
 *  coarse rim onto its corners. */
export function taubin(pts, closed, passes) {
  const n = pts.length;
  if (passes <= 0 || n < 3) return pts;
  const orig = pts.map((p) => p.slice());
  let cur = pts.map((p) => p.slice());
  const step = (p, w) => {
    const q = p.map((v) => v.slice());
    if (closed) {
      for (let i = 0; i < n; i++) {
        const a = p[(i - 1 + n) % n], b = p[(i + 1) % n];
        for (let k = 0; k < 3; k++) {
          q[i][k] = p[i][k] + w * (0.5 * (a[k] + b[k]) - p[i][k]);
        }
      }
    } else {
      for (let i = 1; i < n - 1; i++) {
        for (let k = 0; k < 3; k++) {
          q[i][k] = p[i][k] + w * (0.5 * (p[i - 1][k] + p[i + 1][k]) - p[i][k]);
        }
      }
    }
    return q;
  };
  for (let s = 0; s < passes; s++) {
    cur = step(cur, TAUBIN_LAMBDA);
    cur = step(cur, TAUBIN_MU);
  }
  const seg = [];
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const a = orig[i], b = orig[(i + 1) % n];
    seg.push(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
  }
  const sorted = seg.slice().sort((a, b) => a - b);
  // the median as numpy computes it: on an even count it is the mean
  // of the two middle values, not either of them. Taking one instead
  // moved the cap enough to shift a smoothed rim in the fourth
  // decimal, which the parity test duly caught.
  const mid = sorted.length
    ? (sorted.length % 2
      ? sorted[(sorted.length - 1) / 2]
      : 0.5 * (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]))
    : 0;
  const cap = 0.25 * mid;
  if (cap > 0) {
    for (let i = 0; i < n; i++) {
      const dx = cur[i][0] - orig[i][0];
      const dy = cur[i][1] - orig[i][1];
      const dz = cur[i][2] - orig[i][2];
      const dist = Math.hypot(dx, dy, dz);
      if (dist > cap) {
        const s = cap / dist;
        cur[i] = [orig[i][0] + dx * s, orig[i][1] + dy * s, orig[i][2] + dz * s];
      }
    }
  }
  return cur;
}


// ------------------------------------------- which way is out of a rim

/** Mean position of each vertex's face-neighbours, for every vertex.
 *
 *  A face of width k contributes, to each of its corners, the sum of
 *  the OTHER k-1 corners -- the face sum minus the corner itself -- so
 *  no per-face inner loop is needed. A neighbour shared by two faces
 *  counts twice, in the sum and in the divisor alike, which is what the
 *  engine does. */
export function neighbourMeans(positions, faces) {
  const n = positions.length / 3;
  const tot = new Float64Array(n * 3);
  const cnt = new Float64Array(n);
  for (const f of faces) {
    const k = f.length;
    if (k < 3) continue;
    let sx = 0, sy = 0, sz = 0;
    for (const v of f) { sx += positions[v * 3]; sy += positions[v * 3 + 1]; sz += positions[v * 3 + 2]; }
    for (const v of f) {
      tot[v * 3] += sx - positions[v * 3];
      tot[v * 3 + 1] += sy - positions[v * 3 + 1];
      tot[v * 3 + 2] += sz - positions[v * 3 + 2];
      cnt[v] += k - 1;
    }
  }
  for (let i = 0; i < n; i++) {
    const c = Math.max(cnt[i], 1);
    tot[i * 3] /= c; tot[i * 3 + 1] /= c; tot[i * 3 + 2] /= c;
  }
  return tot;
}

/** Unit vectors along a rim chain pointing AWAY from the surface.
 *
 *  A closed loop in space has no inside, so the rim curve alone cannot
 *  say which way is out -- but the surface can. Step from the mean of
 *  each rim vertex's neighbours back out to the vertex itself, then
 *  remove the component along the rim. What is left is the conormal:
 *  the direction lying IN the surface, across its edge, pointing out of
 *  the sheet. That is the direction to lift a rim tube along so it
 *  rests against the cut rather than straddling it. */
export function outwardField(positions, faces, idx, means) {
  const M = means || neighbourMeans(positions, faces);
  const n = idx.length;
  const P = idx.map((i) => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]);
  const out = idx.map((i, k) => [P[k][0] - M[i * 3], P[k][1] - M[i * 3 + 1],
                                 P[k][2] - M[i * 3 + 2]]);
  // the engine takes the tangent of the chain as if it were closed,
  // whether it is or not, so the two agree at the ends as well
  for (let k = 0; k < n; k++) {
    const a = P[(k - 1 + n) % n], b = P[(k + 1) % n];
    let tx = b[0] - a[0], ty = b[1] - a[1], tz = b[2] - a[2];
    const tm = Math.max(Math.hypot(tx, ty, tz), 1e-30);
    tx /= tm; ty /= tm; tz /= tm;
    const d = out[k][0] * tx + out[k][1] * ty + out[k][2] * tz;
    out[k] = [out[k][0] - d * tx, out[k][1] - d * ty, out[k][2] - d * tz];
  }
  const norm = out.map((v) => Math.hypot(v[0], v[1], v[2]));
  // a rim vertex whose neighbours average out to itself gives no
  // direction; borrow the nearest sample that has one
  const good = [];
  for (let k = 0; k < n; k++) if (norm[k] >= 1e-12) good.push(k);
  if (good.length && good.length < n) {
    for (let k = 0; k < n; k++) {
      if (norm[k] >= 1e-12) continue;
      let best = good[0];
      for (const g of good) if (Math.abs(g - k) < Math.abs(best - k)) best = g;
      out[k] = out[best].slice();
      norm[k] = Math.hypot(out[k][0], out[k][1], out[k][2]);
    }
  }
  return out.map((v, k) => {
    const m = Math.max(norm[k], 1e-30);
    return [v[0] / m, v[1] / m, v[2] / m];
  });
}


/** Thin a polyline so no two points sit closer than `spacing`, by
 *  DROPPING points -- never by moving them.
 *
 *  A swept tube folds over itself wherever the curve turns inside its
 *  own radius, and a rim traced off a mesh has points spaced by the
 *  sample grid, not by the tube. On a rim whose points sit a tenth of
 *  the tube radius apart, every small wiggle of the staircase becomes
 *  a crease, and the tube comes out looking like a caterpillar rather
 *  than a pipe.
 *
 *  Re-interpolating at equal steps of arc length is the wrong tool: it
 *  slides every point along its chords, off the corners the rim was
 *  traced from, and it does that even where the rim was already spaced
 *  comfortably wider than the tube. Choosing a subset cannot introduce
 *  that error -- a point either survives exactly where it was, or goes
 *  -- and on a rim already coarser than the tube nothing is dropped
 *  and this is the identity.
 *
 *  The gap is measured to the last KEPT point rather than the previous
 *  one, which is what handles a rim doubling back on itself, where arc
 *  length advances while the point barely moves.
 *
 *  Returns the surviving INDICES, so the caller can carry per-point
 *  data -- here the outward conormal -- through the thinning. */
export function resample(pts, closed, spacing) {
  const n = pts.length;
  if (!(spacing > 0) || n < 3) return pts.map((_, i) => i);
  const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const out = [0];
  for (let i = 1; i < n; i++) {
    if (gap(pts[i], pts[out[out.length - 1]]) >= spacing) out.push(i);
  }
  if (closed && out.length > 2
      && gap(pts[out[out.length - 1]], pts[out[0]]) < spacing) {
    out.pop();
  }
  return out.length >= 4 ? out : pts.map((_, i) => i);
}

/** The rim as smoothed polylines: {points, closed, outward}.
 *
 *  `outward` is the conormal at each point, taken from the RAW rim
 *  before smoothing -- which is what the engine does, and is right:
 *  the smoother has already rounded away the evidence of where the
 *  sheet was. */
export function boundaryLoops(positions, faces, smooth = RIM_SMOOTH_DEFAULT) {
  const chains = boundaryIndexLoops(faces);
  if (!chains.length) return [];
  const means = neighbourMeans(positions, faces);
  return chains.map((chain) => ({
    points: taubin(chain.idx.map((i) => [positions[i * 3], positions[i * 3 + 1],
                                         positions[i * 3 + 2]]),
                   chain.closed, smooth),
    closed: chain.closed,
    outward: outwardField(positions, faces, chain.idx, means),
  }));
}

// ------------------------------------------------- the whole object

// 16 sides put the flats within 2% of the radius; 8 left the tube
// visibly octagonal at the sizes this rim is used at.
const RIM_SIDES = 16;
// No two control points closer than this multiple of the tube radius:
// a rim traced off the sample grid is far finer than the tube, and
// every step of its staircase would otherwise crease the sweep.
const RIM_SPACING = 1.6;

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

/** Build the object the page draws and the exporter writes.
 *
 *  Order matters: march the field, clip the block to a ball if asked,
 *  take the rim the clip opened, and only then give the sheet a
 *  thickness. Thickening first would wall the rim shut and leave the
 *  tube with nothing to sit on.
 *
 *  Returns the surface and the rim tube separately, because the page
 *  colours them differently, and `exported` with the two merged,
 *  because a printed TPMS with a wire round its cut edge needs the
 *  wire in the file.
 */
export function buildScene({ kind, cells, res, offset = 0, scale = 2,
                             clip = 0, thickness = 0, rim = 0,
                             rimSmooth = RIM_SMOOTH_DEFAULT }) {
  const mesh = block(kind, cells, res, scale, offset);
  let positions = mesh.positions;
  let indices = mesh.indices;
  let faces = null;
  let loops = [];
  let clippedAway = false;

  // 1 is a real radius, not "off": the ball then has the block's own
  // half-width and still bites its corners away. Only 0 is off.
  if (clip > 0) {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
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
      clippedAway = true;                   // the ball missed the surface
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
  positions = positions instanceof Float32Array
    ? positions : Float32Array.from(positions);
  indices = indices instanceof Uint32Array
    ? indices : Uint32Array.from(indices);

  let tube = null;
  if (rim > 0 && loops.length) {
    const out = { pos: [], nor: [], idx: [] };
    for (const l of loops) {
      // A closed rim shorter than the tube's own circumference is not
      // an edge worth drawing -- it reads as a bead sitting on the
      // surface. The engine drops these too.
      let len = 0;
      const n = l.points.length;
      const last = l.closed ? n : n - 1;
      for (let i = 0; i < last; i++) {
        const a = l.points[i], b = l.points[(i + 1) % n];
        len += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      }
      if (l.closed && len < 2 * Math.PI * rim) continue;
      const keep = resample(l.points, l.closed, RIM_SPACING * rim);
      tubeAlong(keep.map((i) => l.points[i]), l.closed, rim, RIM_SIDES, out,
                keep.map((i) => l.outward[i]));
    }
    if (out.idx.length) {
      tube = {
        positions: Float32Array.from(out.pos),
        normals: Float32Array.from(out.nor),
        indices: Uint32Array.from(out.idx),
      };
    }
  }

  let exported = { positions, indices };
  if (tube) {
    const nv = positions.length / 3;
    const pos = new Float32Array(positions.length + tube.positions.length);
    pos.set(positions, 0);
    pos.set(tube.positions, positions.length);
    const idx = new Uint32Array(indices.length + tube.indices.length);
    idx.set(indices, 0);
    for (let i = 0; i < tube.indices.length; i++) {
      idx[indices.length + i] = tube.indices[i] + nv;
    }
    exported = { positions: pos, indices: idx };
  }

  // How far the object reaches, so a camera can be stood back far
  // enough: one period is `scale` units wide, so a four-cell block is
  // four times the size of a one-cell block.
  let extent = 0;
  for (let i = 0; i < positions.length; i += 3) {
    const r = Math.hypot(positions[i], positions[i + 1], positions[i + 2]);
    if (r > extent) extent = r;
  }

  return {
    surface: { positions, indices },
    rim: tube,
    exported,
    stats: {
      extent,
      vertices: positions.length / 3,
      triangles: indices.length / 3,
      cells: mesh.cells,
      solid,
      clippedAway,
      rimLoops: loops.length,
      rimPoints: loops.reduce((a, l) => a + l.points.length, 0),
      rimTriangles: tube ? tube.indices.length / 3 : 0,
    },
  };
}
