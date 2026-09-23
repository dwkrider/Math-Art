// Chair44 (R44), the three-dimensional aperiodic monotile: the
// geometry, in the browser.
//
// A port of math_art/ifs/chair44.py, function for function and in the
// same order, so the two can be compared directly --
// tests/web/test_chair44.mjs runs the Python engine and checks this
// against it, vertex by vertex. Read the Python module's header for
// the mathematics; the notes here are about the port.
//
// EXACTNESS. The Python engine works in Fractions and welds vertices
// on exact equality. Here the coordinates are ordinary doubles and the
// weld key is the coordinate triple rounded to 1e-9. That is safe
// because the construction never brings two distinct vertices closer
// than 1e-4: the panel cuts sit 0.105 apart at the published feature
// width, and the smallest feature height is 1/10000. The parity test
// confirms it the only way that counts -- the same vertex count, in
// the same order, as the exact engine.
//
// After Ioannis Tsiokos, "A Strongly Aperiodic Monotile in Three
// Dimensions", arXiv:2609.19214 (2026), and Chaim Goodman-Strauss,
// "Notes on a strongly aperiodic monotile in E^3", arXiv:2609.24779
// (2026) for the arrow marking.

// ---------------------------------------------------------------- tables

// The eight in-panel feature offsets, in the reading order of Figure 3.
export const OFFSETS = [
  [-1 / 8, 1 / 4], [1 / 8, 1 / 4],
  [-1 / 4, 1 / 8], [1 / 4, 1 / 8],
  [-1 / 4, -1 / 8], [1 / 4, -1 / 8],
  [-1 / 8, -1 / 4], [1 / 8, -1 / 4],
];

// The panel recipe (Figure 3), as {ax, sg, plane, u0, v0, pid, coeffs}.
// `ax` is the normal axis, `sg` its sign, `plane` the coordinate of the
// panel's plane, and (u0, v0) the centre along the two non-normal axes
// in increasing order. Panels 4, 14 and 23 are the notch's inset faces.
export const PANELS = [
  { ax: 0, sg: -1, plane: 0, u0: 0.5, v0: 0.5, pid: 0, coeffs: [2, 4, 6, 8, 5, 7, 1, 3] },
  { ax: 0, sg: -1, plane: 0, u0: 0.5, v0: 1.5, pid: 1, coeffs: [-1, -3, -5, -7, -6, -8, -2, -4] },
  { ax: 0, sg: -1, plane: 0, u0: 1.5, v0: 0.5, pid: 2, coeffs: [8, 7, 4, 3, 2, 1, 6, 5] },
  { ax: 0, sg: -1, plane: 0, u0: 1.5, v0: 1.5, pid: 3, coeffs: [10, 12, -11, -12, -9, -10, 9, 11] },
  { ax: 0, sg: 1, plane: 1, u0: 1.5, v0: 1.5, pid: 4, coeffs: [-2, -4, -6, -8, -5, -7, -1, -3] },
  { ax: 0, sg: 1, plane: 2, u0: 0.5, v0: 0.5, pid: 5, coeffs: [-11, -9, 10, 9, 12, 11, -12, -10] },
  { ax: 0, sg: 1, plane: 2, u0: 0.5, v0: 1.5, pid: 6, coeffs: [-5, -6, -1, -2, -3, -4, -7, -8] },
  { ax: 0, sg: 1, plane: 2, u0: 1.5, v0: 0.5, pid: 7, coeffs: [4, 2, 8, 6, 7, 5, 3, 1] },
  { ax: 1, sg: -1, plane: 0, u0: 0.5, v0: 0.5, pid: 8, coeffs: [-2, -4, -6, -8, -5, -7, -1, -3] },
  { ax: 1, sg: -1, plane: 0, u0: 0.5, v0: 1.5, pid: 9, coeffs: [1, 3, 5, 7, 6, 8, 2, 4] },
  { ax: 1, sg: 1, plane: 2, u0: 0.5, v0: 0.5, pid: 10, coeffs: [11, 9, -10, -9, -12, -11, 12, 10] },
  { ax: 1, sg: 1, plane: 2, u0: 0.5, v0: 1.5, pid: 11, coeffs: [5, 6, 1, 2, 3, 4, 7, 8] },
  { ax: 1, sg: -1, plane: 0, u0: 1.5, v0: 0.5, pid: 12, coeffs: [-8, -7, -4, -3, -2, -1, -6, -5] },
  { ax: 1, sg: -1, plane: 0, u0: 1.5, v0: 1.5, pid: 13, coeffs: [-10, -12, 11, 12, 9, 10, -9, -11] },
  { ax: 1, sg: 1, plane: 1, u0: 1.5, v0: 1.5, pid: 14, coeffs: [2, 4, 6, 8, 5, 7, 1, 3] },
  { ax: 1, sg: 1, plane: 2, u0: 1.5, v0: 0.5, pid: 15, coeffs: [-4, -2, -8, -6, -7, -5, -3, -1] },
  { ax: 2, sg: -1, plane: 0, u0: 0.5, v0: 0.5, pid: 16, coeffs: [11, 9, -10, -9, -12, -11, 12, 10] },
  { ax: 2, sg: 1, plane: 2, u0: 0.5, v0: 0.5, pid: 17, coeffs: [-11, -9, 10, 9, 12, 11, -12, -10] },
  { ax: 2, sg: -1, plane: 0, u0: 0.5, v0: 1.5, pid: 18, coeffs: [-1, -3, -5, -7, -6, -8, -2, -4] },
  { ax: 2, sg: 1, plane: 2, u0: 0.5, v0: 1.5, pid: 19, coeffs: [-5, -6, -1, -2, -3, -4, -7, -8] },
  { ax: 2, sg: -1, plane: 0, u0: 1.5, v0: 0.5, pid: 20, coeffs: [8, 7, 4, 3, 2, 1, 6, 5] },
  { ax: 2, sg: 1, plane: 2, u0: 1.5, v0: 0.5, pid: 21, coeffs: [4, 2, 8, 6, 7, 5, 3, 1] },
  { ax: 2, sg: -1, plane: 0, u0: 1.5, v0: 1.5, pid: 22, coeffs: [10, 12, -11, -12, -9, -10, 9, 11] },
  { ax: 2, sg: 1, plane: 1, u0: 1.5, v0: 1.5, pid: 23, coeffs: [-11, -9, 10, 9, 12, 11, -12, -10] },
];

// The eight child poses of the doubled chair (Table 1).
export const CHILDREN = [
  ['000', [0, 1, 2], [1, 1, 1], [0, 0, 0]],
  ['001', [1, 0, 2], [1, 1, -1], [0, 0, 4]],
  ['010', [0, 2, 1], [1, -1, 1], [0, 4, 0]],
  ['011', [2, 0, 1], [1, -1, -1], [0, 4, 4]],
  ['100', [2, 1, 0], [-1, 1, 1], [4, 0, 0]],
  ['101', [1, 2, 0], [-1, 1, -1], [4, 0, 4]],
  ['110', [0, 1, 2], [-1, -1, 1], [4, 4, 0]],
  ['central', [0, 1, 2], [1, 1, 1], [1, 1, 1]],
];

// The 44-contact atlas (Figure 7): the relative poses a neighbour may
// take against a root chair in the identity pose.
export const ATLAS44 = [
  [[0, 1, 2], [-1, -1, 1], [2, 2, -2]], [[0, 1, 2], [-1, -1, 1], [2, 2, 2]],
  [[0, 1, 2], [-1, -1, 1], [3, 3, -1]], [[0, 1, 2], [-1, -1, 1], [3, 3, 1]],
  [[0, 1, 2], [-1, 1, -1], [2, -2, 2]], [[0, 1, 2], [-1, 1, -1], [2, 2, 2]],
  [[0, 1, 2], [1, -1, -1], [-2, 2, 2]], [[0, 1, 2], [1, -1, -1], [2, 2, 2]],
  [[0, 1, 2], [1, 1, 1], [-1, -1, -1]], [[0, 1, 2], [1, 1, 1], [1, 1, 1]],
  [[0, 2, 1], [-1, 1, 1], [4, 0, 0]], [[0, 2, 1], [1, -1, 1], [-1, 3, -1]],
  [[0, 2, 1], [1, -1, 1], [0, 4, 0]], [[0, 2, 1], [1, 1, -1], [0, 0, 4]],
  [[0, 2, 1], [1, 1, -1], [1, 1, 3]], [[1, 0, 2], [-1, 1, 1], [0, 0, 0]],
  [[1, 0, 2], [-1, 1, 1], [4, 0, 0]], [[1, 0, 2], [1, -1, 1], [0, 0, 0]],
  [[1, 0, 2], [1, -1, 1], [0, 4, 0]], [[1, 0, 2], [1, 1, -1], [-1, -1, 3]],
  [[1, 0, 2], [1, 1, -1], [0, 0, 0]], [[1, 0, 2], [1, 1, -1], [0, 0, 4]],
  [[1, 0, 2], [1, 1, -1], [1, 1, 3]], [[1, 2, 0], [-1, -1, 1], [2, 2, -2]],
  [[1, 2, 0], [-1, -1, 1], [2, 2, 2]], [[1, 2, 0], [-1, -1, 1], [3, 3, 1]],
  [[1, 2, 0], [-1, 1, -1], [2, -2, 2]], [[1, 2, 0], [-1, 1, -1], [2, 2, 2]],
  [[1, 2, 0], [-1, 1, -1], [3, -1, 3]], [[1, 2, 0], [1, -1, -1], [-2, 2, 2]],
  [[1, 2, 0], [1, -1, -1], [2, 2, 2]], [[2, 0, 1], [-1, -1, 1], [2, 2, -2]],
  [[2, 0, 1], [-1, -1, 1], [2, 2, 2]], [[2, 0, 1], [-1, -1, 1], [3, 3, 1]],
  [[2, 0, 1], [-1, 1, -1], [2, -2, 2]], [[2, 0, 1], [-1, 1, -1], [2, 2, 2]],
  [[2, 0, 1], [1, -1, -1], [-2, 2, 2]], [[2, 0, 1], [1, -1, -1], [-1, 3, 3]],
  [[2, 0, 1], [1, -1, -1], [2, 2, 2]], [[2, 1, 0], [-1, 1, 1], [3, -1, -1]],
  [[2, 1, 0], [-1, 1, 1], [4, 0, 0]], [[2, 1, 0], [1, -1, 1], [0, 4, 0]],
  [[2, 1, 0], [1, 1, -1], [0, 0, 4]], [[2, 1, 0], [1, 1, -1], [1, 1, 3]],
];

export const ETA_TRUE = 1 / 100;        // feature base half-width
export const HEIGHT_TRUE = 1 / 10000;   // unit of signed height
export const ETA_SHOWN = 5 / 100;       // the paper's figure convention
export const CENTROID = [13 / 14, 13 / 14, 13 / 14];

export const ARROW_COLORS = ['BLUE', 'GREEN', 'RED'];

// panel id -> arrow colour index. Blue is forced (it is the
// self-matching class); green and red are a free relabelling.
export const PANEL_COLOR = [
  2, 1, 2, 0, 1, 0, 1, 2, 1, 2, 0, 2,
  1, 0, 2, 1, 0, 0, 1, 1, 2, 2, 0, 0,
];

// the seven surviving corners of the 2-cube, plus the concave socket
export const SPECIAL_VERTICES = (() => {
  const out = [];
  for (const x of [0, 2]) {
    for (const y of [0, 2]) {
      for (const z of [0, 2]) if (!(x === 2 && y === 2 && z === 2)) out.push([x, y, z]);
    }
  }
  out.push([1, 1, 1]);
  return out;
})();
const SPECIAL_SET = new Set(SPECIAL_VERTICES.map((v) => v.join(',')));

// ---------------------------------------------------------------- helpers

const key3 = (p) => p.map((x) => Math.round(x * 1e9) / 1e9).join(',');

function welder() {
  const index = new Map();
  const verts = [];
  const vid = (p) => {
    const k = key3(p);
    let i = index.get(k);
    if (i === undefined) {
      i = verts.length;
      index.set(k, i);
      verts.push([p[0], p[1], p[2]]);
    }
    return i;
  };
  return { verts, vid };
}

/** The one special vertex among a panel's four corners, and its
 *  direction from the panel centre. */
export function panelHome(panel) {
  const { ax, plane, u0, v0 } = panel;
  const [o0, o1] = [0, 1, 2].filter((i) => i !== ax);
  const found = [];
  for (const du of [-0.5, 0.5]) {
    for (const dv of [-0.5, 0.5]) {
      const p = [0, 0, 0];
      p[ax] = plane;
      p[o0] = u0 + du;
      p[o1] = v0 + dv;
      if (SPECIAL_SET.has(p.map((x) => Math.round(x)).join(','))) {
        found.push([p.map((x) => Math.round(x)), [du, dv]]);
      }
    }
  }
  if (found.length !== 1) throw new Error(`panel ${panel.pid} owns ${found.length} vertices`);
  return found[0];
}

// arrow shape in panel coordinates, along the diagonal toward the home
// vertex: a shaft with a barbed head.
const A_TAIL = -0.50, A_TIP = 0.62;
const A_HEAD = 0.34, A_BARB = 0.20;
const A_HALF = 0.17, A_SHAFT = 0.072;
export const A_SINK = -0.005, A_RISE = 0.014;
const A_HALF_GAIN = 1.4;

/** The arrow outline in panel (u, v), pointing at (su, sv)/2.
 *  `half` = 0 is the whole (blue) arrow; +-1 the two halves, which is
 *  the same marking seen from the two sides of its panel. */
export function arrowPolygon(su, sv, half = 0) {
  const r = 0.7071067811865476;
  const du = su * r, dv = sv * r;
  const nu = -dv, nv = du;
  const at = (along, across) => [du * along + nu * across, dv * along + nv * across];
  let pts;
  if (half === 0) {
    pts = [at(A_TAIL, A_SHAFT), at(A_TIP - A_BARB, A_SHAFT),
           at(A_TIP - A_HEAD, A_HALF), at(A_TIP, 0.0),
           at(A_TIP - A_HEAD, -A_HALF), at(A_TIP - A_BARB, -A_SHAFT),
           at(A_TAIL, -A_SHAFT)];
  } else {
    const k = (half > 0 ? 1.0 : -1.0) * A_HALF_GAIN;
    pts = [at(A_TAIL, k * A_SHAFT), at(A_TIP - A_BARB, k * A_SHAFT),
           at(A_TIP - A_HEAD, k * A_HALF), at(A_TIP, 0.0), at(A_TAIL, 0.0)];
  }
  const n = pts.length;
  let area = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
  }
  return area / 2 < 0 ? pts.slice().reverse() : pts;
}

// Both outlines are concave at the barbs, so each is cut into convex
// pieces by hand rather than fanned.
const ARROW_PIECES_FULL = [[0, 1, 5, 6], [2, 3, 4], [1, 2, 4], [1, 4, 5]];
const ARROW_PIECES_HALF = [[0, 1, 4], [1, 2, 3], [1, 3, 4]];

/** Which half of the arrow a panel carries, as a sign in its own
 *  (u, v) frame: green barbs right of travel, red left, seen from
 *  outside -- so the sign is taken back through the winding flip. */
export function arrowHalf(pid, flip) {
  const colour = PANEL_COLOR[pid];
  if (colour === 0) return 0;
  const seenLeft = colour === 2 ? 1 : -1;
  return flip ? -seenLeft : seenLeft;
}

// ---------------------------------------------------------------- frames

export function frame(p, s) {
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) M[i][p[i]] = s[i];
  return M;
}

export function matMul(A, B) {
  const out = [];
  for (let i = 0; i < 3; i++) {
    const row = [];
    for (let j = 0; j < 3; j++) {
      row.push(A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]);
    }
    out.push(row);
  }
  return out;
}

export function matApply(M, v) {
  return [M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
          M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
          M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]];
}

export function matTranspose(M) {
  return [[M[0][0], M[1][0], M[2][0]],
          [M[0][1], M[1][1], M[2][1]],
          [M[0][2], M[1][2], M[2][2]]];
}

export function det(M) {
  return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
       - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
       + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

const PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
const SIGNS = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
               [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]];

/** The 24 proper signed permutation matrices, in the engine's order
 *  (itertools.permutations then itertools.product). */
export function properFrames() {
  const out = [];
  for (const p of PERMS) {
    for (const s of SIGNS) {
      const M = frame(p, s);
      if (det(M) === 1) out.push(M);
    }
  }
  return out;
}

export const PROPER_FRAMES = properFrames();
const FRAME_INDEX = new Map(PROPER_FRAMES.map((M, i) => [M.flat().join(','), i]));

/** Which of the 24 proper cubic rotations a pose uses. */
export function frameIndex(M) {
  return FRAME_INDEX.get(M.flat().join(','));
}

// ------------------------------------------------- carrier and features

/** The seven unit cells of the chair, by their lower corners. */
export function carrierCells() {
  const out = [];
  for (const x of [0, 1]) {
    for (const y of [0, 1]) {
      for (const z of [0, 1]) if (!(x === 1 && y === 1 && z === 1)) out.push([x, y, z]);
    }
  }
  return out;
}

/** The 192 features, as {centre, normal, coeff}. */
export function features(height = HEIGHT_TRUE) {
  const out = [];
  for (const panel of PANELS) {
    const { ax, sg, plane, u0, v0, coeffs } = panel;
    const [o0, o1] = [0, 1, 2].filter((i) => i !== ax);
    const n = [0, 0, 0];
    n[ax] = sg;
    OFFSETS.forEach(([du, dv], k) => {
      const c = [0, 0, 0];
      c[ax] = plane;
      c[o0] = u0 + du;
      c[o1] = v0 + dv;
      out.push({ centre: c, normal: n.slice(), coeff: coeffs[k], height });
    });
  }
  return out;
}

/** The two named in-panel axes, and whether their cross product points
 *  along +e_ax. */
function panelAxes(ax) {
  const [o0, o1] = [0, 1, 2].filter((i) => i !== ax);
  return [o0, o1, ax !== 1 ? 1 : -1];
}

// ------------------------------------------------------------ tile meshes

/** The carrier alone: 24 unit-square panels, outward-oriented. */
export function bareTileMesh() {
  const { verts, vid } = welder();
  const faces = [];
  for (const { ax, sg, plane, u0, v0 } of PANELS) {
    const [o0, o1, orient] = panelAxes(ax);
    const corners = [[u0 - 0.5, v0 - 0.5], [u0 + 0.5, v0 - 0.5],
                     [u0 + 0.5, v0 + 0.5], [u0 - 0.5, v0 + 0.5]];
    const quad = corners.map(([u, v]) => {
      const p = [0, 0, 0];
      p[ax] = plane;
      p[o0] = u;
      p[o1] = v;
      return vid(p);
    });
    if (orient !== sg) quad.reverse();
    faces.push(quad);
  }
  return { verts, faces };
}

/** The bare carrier plus Goodman-Strauss's arrow marking per panel.
 *  `colors` is parallel to `faces`: null for a body panel, else an
 *  index into ARROW_COLORS. */
export function arrowTileMesh() {
  const { verts, vid } = welder();
  const faces = [];
  const colors = [];

  const body = bareTileMesh();
  const remap = body.verts.map((q) => vid(q));
  for (const f of body.faces) {
    faces.push(f.map((i) => remap[i]));
    colors.push(null);
  }

  for (const panel of PANELS) {
    const { ax, sg, plane, u0, v0, pid } = panel;
    const [o0, o1, orient] = panelAxes(ax);
    const flip = orient !== sg;
    const [, [du, dv]] = panelHome(panel);
    const half = arrowHalf(pid, flip);
    const poly = arrowPolygon(du > 0 ? 1 : -1, dv > 0 ? 1 : -1, half);
    const pieces = half === 0 ? ARROW_PIECES_FULL : ARROW_PIECES_HALF;

    const pt = (uv, lift) => {
      const q = [0, 0, 0];
      q[ax] = plane + sg * lift;
      q[o0] = u0 + uv[0];
      q[o1] = v0 + uv[1];
      return vid(q);
    };
    const lo = poly.map((q) => pt(q, A_SINK));
    const hi = poly.map((q) => pt(q, A_RISE));
    const col = PANEL_COLOR[pid];
    for (const piece of pieces) {
      const top = piece.map((i) => hi[i]);
      const bot = piece.map((i) => lo[i]).reverse();
      faces.push(flip ? top.slice().reverse() : top);
      colors.push(col);
      faces.push(flip ? bot.slice().reverse() : bot);
      colors.push(col);
    }
    for (let i = 0; i < poly.length; i++) {
      const j = (i + 1) % poly.length;
      const side = [lo[i], lo[j], hi[j], hi[i]];
      faces.push(flip ? side.slice().reverse() : side);
      colors.push(col);
    }
  }
  return { verts, faces, colors };
}

/** Plan area of one arrow marking. */
export function arrowArea(half = 0) {
  const pts = arrowPolygon(1, 1, half);
  const n = pts.length;
  let a = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
  }
  return Math.abs(a) / 2;
}

/** One copy of Q: every panel cut into a 9x9 grid, the eight cells on
 *  the feature offsets raised (or sunk) into square pyramids. */
export function tileMesh(eta = ETA_TRUE, height = HEIGHT_TRUE) {
  if (!(eta > 0 && eta < 1 / 16)) {
    throw new Error('feature half-width must satisfy 0 < eta < 1/16');
  }
  const bases = [-1 / 4, -1 / 8, 1 / 8, 1 / 4];
  const inner = [];
  for (const b of bases) inner.push(b - eta, b + eta);
  inner.sort((a, b) => a - b);
  const cuts = [-0.5, ...inner, 0.5];
  // index of the low edge of each feature's base interval
  const slot = new Map();
  bases.forEach((b) => {
    slot.set(b, cuts.findIndex((c) => Math.abs(c - (b - eta)) < 1e-12));
  });

  const { verts, vid } = welder();
  const faces = [];

  for (const panel of PANELS) {
    const { ax, sg, plane, u0, v0, coeffs } = panel;
    const [o0, o1, orient] = panelAxes(ax);
    const flip = orient !== sg;

    const point = (u, v, lift = 0) => {
      const p = [0, 0, 0];
      p[ax] = plane + sg * lift;
      p[o0] = u0 + u;
      p[o1] = v0 + v;
      return vid(p);
    };

    const pyramids = new Map();
    OFFSETS.forEach(([du, dv], k) => {
      pyramids.set(`${slot.get(du)},${slot.get(dv)}`, coeffs[k]);
    });

    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        const uLo = cuts[i], uHi = cuts[i + 1];
        const vLo = cuts[j], vHi = cuts[j + 1];
        const quad = [point(uLo, vLo), point(uHi, vLo),
                      point(uHi, vHi), point(uLo, vHi)];
        if (flip) quad.reverse();
        const a = pyramids.get(`${i},${j}`);
        if (a === undefined) {
          faces.push(quad);
          continue;
        }
        const apex = point((uLo + uHi) / 2, (vLo + vHi) / 2, a * height);
        for (let k = 0; k < 4; k++) faces.push([quad[k], quad[(k + 1) % 4], apex]);
      }
    }
  }
  return { verts, faces };
}

/** Signed volume of a closed, outward-oriented mesh. */
export function meshVolume(verts, faces) {
  let tot = 0;
  for (const f of faces) {
    const a = verts[f[0]];
    for (let k = 1; k < f.length - 1; k++) {
      const b = verts[f[k]], c = verts[f[k + 1]];
      tot += a[0] * (b[1] * c[2] - b[2] * c[1])
           - a[1] * (b[0] * c[2] - b[2] * c[0])
           + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
  }
  return tot / 6;
}

/** Fan-split every face into triangles. */
export function triangulate(faces) {
  const out = [];
  for (const f of faces) {
    for (let k = 1; k < f.length - 1; k++) out.push([f[0], f[k], f[k + 1]]);
  }
  return out;
}

// --------------------------------------------------------- the substitution

/** The eight children of one pose (G, t). */
export function refine(G, t) {
  const out = [];
  for (const [, p, s, u] of CHILDREN) {
    const H = frame(p, s);
    const Gu = matApply(G, u);
    out.push([matMul(G, H), [2 * t[0] + Gu[0], 2 * t[1] + Gu[1], 2 * t[2] + Gu[2]]]);
  }
  return out;
}

/** A substitution patch of 8**depth chairs, as {G, t, group}, where
 *  `group` is the child taken at the FIRST refinement -- so the eight
 *  groups are the level-(n-1) supertiles. */
export function patch(depth) {
  if (depth < 0) throw new Error('depth must be >= 0');
  const identity = frame([0, 1, 2], [1, 1, 1]);
  if (depth === 0) return [{ G: identity, t: [0, 0, 0], group: 0 }];
  let tagged = refine(identity, [0, 0, 0]).map(([G, t], g) => ({ G, t, group: g }));
  for (let d = 1; d < depth; d++) {
    const next = [];
    for (const { G, t, group } of tagged) {
      for (const [H, u] of refine(G, t)) next.push({ G: H, t: u, group });
    }
    tagged = next;
  }
  return tagged;
}

/** The seven unit cells a posed chair occupies. */
export function poseCells(G, t) {
  const out = [];
  for (const a of carrierCells()) {
    const lo = [Infinity, Infinity, Infinity];
    for (const dx of [0, 1]) {
      for (const dy of [0, 1]) {
        for (const dz of [0, 1]) {
          const c = matApply(G, [a[0] + dx, a[1] + dy, a[2] + dz]);
          for (let i = 0; i < 3; i++) lo[i] = Math.min(lo[i], c[i]);
        }
      }
    }
    out.push([lo[0] + t[0], lo[1] + t[1], lo[2] + t[2]]);
  }
  return out;
}

/** For every legal contact, the sum of the two chairs' centroid
 *  distances to their shared panel plane. */
function contactSpans() {
  const root = frame([0, 1, 2], [1, 1, 1]);
  const rootCells = new Set(poseCells(root, [0, 0, 0]).map((c) => c.join(',')));
  const spans = new Set();
  for (const [p, s, off] of ATLAS44) {
    const G = frame(p, s);
    const gc = matApply(G, CENTROID);
    const nb = [gc[0] + off[0], gc[1] + off[1], gc[2] + off[2]];
    for (const b of poseCells(G, off)) {
      for (let ax = 0; ax < 3; ax++) {
        for (const sg of [-1, 1]) {
          const a = b.slice();
          a[ax] += sg;
          if (!rootCells.has(a.join(','))) continue;
          const plane = Math.max(a[ax], b[ax]);
          spans.add(Math.abs(plane - CENTROID[ax]) + Math.abs(plane - nb[ax]));
        }
      }
    }
  }
  return [...spans];
}

export const MIN_CONTACT_SPAN = Math.min(...contactSpans());

/** Largest height exaggeration that keeps neighbouring pyramids clear
 *  at this gap. A bound on a cosmetic overlap, not on correctness. */
export function maxRelief(gap) {
  if (gap >= 1.0) return Infinity;
  const clearance = (1.0 - gap) * MIN_CONTACT_SPAN;
  return clearance * 10000.0 / (2.0 * gap * 12.0);
}

/** Relative poses of every face-adjacent pair in a patch, normalized
 *  so that tile i sits in the identity pose. */
export function contacts(poses) {
  const owner = new Map();
  poses.forEach(({ G, t }, idx) => {
    for (const cell of poseCells(G, t)) owner.set(cell.join(','), idx);
  });
  const seen = new Set();
  const out = [];
  for (const [ck, i] of owner) {
    const cell = ck.split(',').map(Number);
    for (let ax = 0; ax < 3; ax++) {
      for (const sg of [-1, 1]) {
        const nb = cell.slice();
        nb[ax] += sg;
        const j = owner.get(nb.join(','));
        if (j === undefined || j === i || seen.has(`${i},${j}`)) continue;
        seen.add(`${i},${j}`);
        const Ri = matTranspose(poses[i].G);
        const d = [poses[j].t[0] - poses[i].t[0],
                   poses[j].t[1] - poses[i].t[1],
                   poses[j].t[2] - poses[i].t[2]];
        out.push({ i, j, G: matMul(Ri, poses[j].G), t: matApply(Ri, d) });
      }
    }
  }
  return out;
}

const ATLAS_SET = new Set(ATLAS44.map(([p, s, off]) => frame(p, s).flat().join(',') + '|' + off.join(',')));

/** Is a relative pose one of the 44 the features permit? */
export function inAtlas(G, t) {
  return ATLAS_SET.has(G.flat().join(',') + '|' + t.join(','));
}
