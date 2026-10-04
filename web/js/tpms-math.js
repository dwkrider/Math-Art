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
