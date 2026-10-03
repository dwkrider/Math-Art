// Scherk-Collins saddle towers: the geometry, in the browser.
//
// A port of the mid-surface construction in math_art/minsurf/scherk.py,
// which is itself a re-implementation of the engine of Carlo Sequin's
// Sculpture Generator I. tests/web/test_scherk.mjs runs that engine and
// checks this against it, point by point, including which levels have
// their holes cut open.
//
// THE SURFACE. Scherk's singly-periodic minimal surface of 1835 is
//
//     sin(z) = sinh(x) sinh(y),
//
// a tower of saddles. Slice it at a height where sin(z) = c and the
// cross-section can be written down exactly:
//
//     x(s) = asinh(sqrt(c) e^{sL}),  y(s) = asinh(sqrt(c) e^{-sL}),
//     L = ln(sinh W / sqrt c),       s in [-1, 1],
//
// which puts the ends of the curve exactly on the truncation planes
// x = W and y = W, where W is the flange width. Above sinh(W)^2 the
// level has no curve at all: that is a hole in the tower, cut open by
// the truncation. Saddles with more than two arms come from squeezing
// that 90-degree wedge into a wedge of 180/b degrees, and each storey
// is turned 180/b from the one below. Bend the tower round a circle
// (warp) and the ends can be made to meet: that is the toroidal form
// Brent Collins carved and Sequin parameterised.
//
// WHAT IS PORTED, AND WHAT IS NOT. The mid-surface -- the mathematics
// above -- is ported exactly. The generator then gives the sheet a
// thickness with rims and welds of its own, which is a Blender-side
// construction of several hundred lines; the page draws the mid-surface
// and, for export, thickens it with the site's own solidifier. So an
// STL from here is a faithful offset of the same surface, not a
// byte-for-byte match of the add-on's mesh.

const { sin, cos, sqrt, sinh, asinh, atan2, exp, log, hypot, PI } = Math;

// Matches the proportions of the original program.
export const XY_SCALE = 1.5 / PI;

export const DEFAULTS = {
  branches: 2, storeys: 2, height: 1.5, flange: 1.5, thickness: 0.15,
  rimBulge: 1.5, twist: 0, azimuth: 0, warp: 0, detail: 5,
  scaleX: 1, scaleY: 1, scaleZ: 1, globalScale: 1, phase: 0.5, rimRound: 1,
};

/** The generator's presets, name and parameters, in its order. */
export const PRESETS = {
  HEX: ['Hyperbolic Hexagon', { branches: 2, storeys: 6, height: 1.2, flange: 1.1, thickness: 0.07, rimBulge: 1.06, twist: 0, azimuth: 45, warp: 360 }],
  TREFOIL: ['Minimal Trefoil', { branches: 2, storeys: 3, height: 1.5, flange: 1.3, thickness: 0.06, rimBulge: 1.06, twist: 270, azimuth: 45, warp: 360 }],
  MONKEY: ['Monkey-Saddle Trefoil', { branches: 3, storeys: 3, height: 1.75, flange: 1.5, thickness: 0.05, rimBulge: 1.0, twist: 180, azimuth: 0, warp: 360 }],
  HEPTOROID: ['Heptoroid', { branches: 4, storeys: 7, height: 1.5, flange: 1.0, thickness: 0.10, rimBulge: 0.0, twist: 135, azimuth: 0, warp: 360 }],
  TOWER: ['Scherk Tower (straight)', { branches: 2, storeys: 5, height: 1.5, flange: 1.5, thickness: 0.1, rimBulge: 1.0, twist: 0, azimuth: 0, warp: 0 }],
  DEMO1: ['Demo 1', { branches: 1, storeys: 4, height: 1.9, flange: 1.5, thickness: 0.08, rimBulge: 1.5, warp: 270, twist: 885, azimuth: 0, detail: 11 }],
  DEMO2: ['Demo 2', { branches: 1, storeys: 3, height: 1.7, flange: 1.0, thickness: 0.08, rimBulge: 1.5, warp: 360, twist: 540, azimuth: 0, detail: 10 }],
  DEMO3: ['Demo 3', { branches: 3, storeys: 5, height: 1.0, flange: 1.1, thickness: 0.04, rimBulge: 1.5, warp: 0, twist: -135, azimuth: 0, detail: 5 }],
  DEMO4: ['Demo 4', { branches: 3, storeys: 6, height: 1.0, flange: 1.1, thickness: 0.08, rimBulge: 1.01, warp: 360, twist: 0, azimuth: 0, detail: 5 }],
  DEMO6: ['Demo 6', { branches: 6, storeys: 4, height: 1.0, flange: 0.8, thickness: 0.02, rimBulge: 0.0, warp: 360, twist: 180, azimuth: 45, detail: 7 }],
  DEMO7: ['Demo 7', { branches: 4, storeys: 8, height: 1.0, flange: 0.8, thickness: 0.04, rimBulge: 0.96, warp: 360, twist: 180, azimuth: 45, detail: 6 }],
  DEMO8: ['Demo 8', { branches: 4, storeys: 12, height: 1.0, flange: 1.0, thickness: 0.06, rimBulge: 1.06, warp: 210, twist: 180, azimuth: 45, detail: 7 }],
  DEMO10: ['Demo 10', { branches: 4, storeys: 3, height: 1.0, flange: 1.0, thickness: 0.09, rimBulge: 1.06, warp: 0, twist: 180, azimuth: 45, detail: 5 }],
  DEMO11: ['Demo 11', { branches: 7, storeys: 2, height: 0.1, flange: 2.1, thickness: 0.0, rimBulge: 0.0, warp: 0, twist: 0, azimuth: 0, detail: 5 }],
  DEMO12: ['Demo 12', { branches: 7, storeys: 1, height: 0.3, flange: 2.6, thickness: 0.03, rimBulge: 0.0, warp: 0, twist: -75, azimuth: 0, detail: 8 }],
  DEMO13: ['Demo 13', { branches: 5, storeys: 3, height: 0.3, flange: 1.1, thickness: 0.2, rimBulge: 0.55, warp: 0, twist: 0, azimuth: 0, detail: 6 }],
  DEMO14: ['Demo 14', { branches: 2, storeys: 3, height: 1.5, flange: 1.5, thickness: 0.15, rimBulge: 1.5, warp: 0, twist: 0, azimuth: 0, detail: 5 }],
  DEMO15: ['Demo 15', { branches: 2, storeys: 11, height: 1.5, flange: 1.5, thickness: 0.15, rimBulge: 1.5, warp: 180, twist: 495, azimuth: 0, detail: 5 }],
  DEMO16: ['Demo 16', { branches: 1, storeys: 3, height: 1.5, flange: 1.5, thickness: 0.15, rimBulge: 1.5, warp: 0, twist: -300, azimuth: 0, detail: 5 }],
  DEMO20: ['Demo 20', { branches: 4, storeys: 7, height: 1.5, flange: 0.9, thickness: 0.11, rimBulge: 0.43, warp: 60, twist: 390, azimuth: -15, detail: 5 }],
};

/** Fill in whatever a caller left out. */
export function params(over = {}) {
  return { ...DEFAULTS, ...over };
}

/** Parameters for a preset, as the generator defines it. */
export function preset(name) {
  const hit = PRESETS[name];
  return params(hit ? hit[1] : {});
}

/** True when a warped tower's two ends meet with matching profiles.
 *  The condition is Sequin's: a full turn of warp, and the twist plus
 *  the storeys' own rotation a whole number of branch-widths. */
export function ringCloses(p) {
  const w = ((p.warp % 360) + 360) % 360;
  if (p.warp <= 0 || (Math.abs(w) > 1e-4 && Math.abs(w - 360) > 1e-4)) return false;
  const b = p.branches;
  const mismatch = ((p.twist + p.storeys * 180 / b) % (360 / b) + 360 / b) % (360 / b);
  return mismatch < 1e-3 || (360 / b) - mismatch < 1e-3;
}

/** Centre on the origin and scale the longest side to `span`. */
export function fitTransform(points, globalScale = 1, span = 2) {
  if (!points.length) return { centre: [0, 0, 0], factor: globalScale };
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const v of points) {
    for (let k = 0; k < 3; k++) {
      if (v[k] < lo[k]) lo[k] = v[k];
      if (v[k] > hi[k]) hi[k] = v[k];
    }
  }
  const centre = [0, 1, 2].map((k) => 0.5 * (lo[k] + hi[k]));
  const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return { centre, factor: (ext > 1e-9 ? span / ext : 1) * globalScale };
}

/**
 * The mid-surface, as the generator builds it.
 *
 * Returns { grids, R, m, cells, closes }, where `grids` is a Map from
 * "cell,branch" to R+1 rows, each either a list of m points or null --
 * a null row is a level whose hole the truncation has cut open.
 */
export function grids(pIn) {
  const p = params(pIn);
  const b = p.branches;
  const S = p.storeys;
  // The flange carries a +0.1 correction: the original tool's flange
  // was 0.1 narrower than its setting, so a stored flange is read as
  // the corrected value, which is what makes the demo files load right.
  const W = p.flange + 0.1;
  const d = Math.max(1, p.detail);
  const R = 4 * d;                 // rows per storey
  const m = 4 * d + 1;             // points across a branch (odd)
  const H_ex = S * PI;
  const H_out = S * p.height;
  const gs = p.globalScale;
  const warpOn = p.warp > 1e-6;
  const closes = ringCloses(p);
  // Where the saddle chain sits relative to the tower's ends. It is
  // invisible on a closed ring, so it is forced to zero there.
  const ph = closes ? 0 : Math.max(0, Math.min(0.999, p.phase));
  const zBase = ph * PI;
  // The azimuth carries a -45 degree correction, for the same reason
  // as the flange: it matches the original tool's own files.
  const az = (p.azimuth - 45) * PI / 180;
  const tw = p.twist * PI / 180;
  const wr = warpOn ? p.warp * PI / 180 : 0;
  const Rring = warpOn ? H_out / wr : 0;
  const sinhW = sinh(W);
  const cOpen = sinhW * sinhW;     // above this, the level is a hole

  const xform = (xt, yt, zEx) => {
    const xs = xt * XY_SCALE * gs;
    const ys = yt * XY_SCALE * gs;
    const zn = (zEx - zBase) / H_ex;
    const a = az + tw * zn;
    const ca = cos(a), sa = sin(a);
    const x1 = xs * ca - ys * sa;
    const y1 = xs * sa + ys * ca;
    let X, Y, Z;
    if (warpOn) {
      const th = wr * (zn - 0.5);
      const rad = Rring * gs + x1;
      X = rad * cos(th); Y = rad * sin(th); Z = y1;
    } else {
      X = x1; Y = y1; Z = (zn - 0.5) * H_out * gs;
    }
    return [X * p.scaleX, Y * p.scaleY, Z * p.scaleZ];
  };

  // cosine-spaced across the curve, with the midpoint exactly zero
  const sig = [];
  for (let k = 0; k < m; k++) sig.push(-cos(PI * k / (m - 1)));

  // The tower is the window [ph, S + ph] of the periodic chain, split
  // into cells at the saddle levels; a cell never crosses one, so its
  // branch rotation is constant.
  const tLo = ph, tHi = S + ph;
  const bounds = [tLo];
  for (let k = 1; k <= S; k++) if (tLo + 1e-9 < k && k < tHi - 1e-9) bounds.push(k);
  bounds.push(tHi);
  const cells = [];
  for (let c = 0; c < bounds.length - 1; c++) {
    cells.push([bounds[c], bounds[c + 1], Math.floor(bounds[c] + 1e-9)]);
  }

  const cellT = (ci, i) => {
    const [a, bc] = cells[ci];
    return a + (bc - a) * (0.5 - 0.5 * cos(PI * i / R));
  };

  const out = new Map();
  for (let ci = 0; ci < cells.length; ci++) {
    const sRot = cells[ci][2];
    for (let j = 0; j < b; j++) {
      const A = (2 * j + sRot) % (2 * b);
      const baseAng = A * PI / b;
      const wedge = PI / b;
      const rows = [];
      for (let i = 0; i <= R; i++) {
        const t = cellT(ci, i);
        const zEx = t * PI;
        const frac = t - sRot;
        const c = sin(PI * frac);
        // One branch is an order-one saddle, which is a plane, so every
        // level takes the flat flange cross-section; otherwise only the
        // saddle levels do.
        if (b === 1 || Math.abs(frac - Math.round(frac)) < 1e-9) {
          const row = [];
          for (let k = 0; k < m; k++) {
            const sg = sig[k];
            let ang, rad;
            if (sg > 1e-12) { ang = baseAng; rad = W * sg; }
            else if (sg < -1e-12) { ang = baseAng + wedge; rad = -W * sg; }
            else { ang = baseAng; rad = 0; }
            row.push(xform(rad * cos(ang), rad * sin(ang), zEx));
          }
          rows.push(row);
          continue;
        }
        if (c > cOpen) { rows.push(null); continue; }
        const sc = sqrt(c);
        const L = log(sinhW / sc);
        const row = [];
        for (let k = 0; k < m; k++) {
          const e = exp(sig[k] * L);
          const x = asinh(sc * e);
          const y = asinh(sc / e);
          const f = atan2(y, x) / (PI / 2);
          const ang = baseAng + f * wedge;
          const rad = hypot(x, y);
          row.push(xform(rad * cos(ang), rad * sin(ang), zEx));
        }
        rows.push(row);
      }
      out.set(`${ci},${j}`, rows);
    }
  }
  return { grids: out, R, m, cells, closes };
}

/**
 * The mid-surface as a triangle mesh, centred and scaled to the
 * project's 2 m cube.
 *
 * The tessellation is the page's own -- the generator's is bound up
 * with the thickening it does next -- but every vertex is a point of
 * the grids above, so the surface is the generator's surface.
 */
export function surfaceMesh(pIn) {
  const p = params(pIn);
  const g = grids(p);
  const { R, m } = g;

  const all = [];
  for (const rows of g.grids.values()) {
    for (const row of rows) if (row) for (const pt of row) all.push(pt);
  }
  const { centre, factor } = fitTransform(all, p.globalScale);

  const positions = [];
  const indices = [];
  for (const rows of g.grids.values()) {
    // index of each row's first vertex, or -1 where the hole is open
    const rowBase = new Array(R + 1).fill(-1);
    for (let i = 0; i <= R; i++) {
      if (!rows[i]) continue;
      rowBase[i] = positions.length / 3;
      for (const pt of rows[i]) {
        positions.push((pt[0] - centre[0]) * factor,
                       (pt[1] - centre[1]) * factor,
                       (pt[2] - centre[2]) * factor);
      }
    }
    for (let i = 0; i < R; i++) {
      if (rowBase[i] < 0 || rowBase[i + 1] < 0) continue;   // across a hole
      for (let k = 0; k < m - 1; k++) {
        const a = rowBase[i] + k, bq = a + 1;
        const c = rowBase[i + 1] + k, dq = c + 1;
        indices.push(a, c, dq, a, dq, bq);
      }
    }
  }
  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
    closes: g.closes,
    rows: R + 1,
    across: m,
    patches: g.grids.size,
  };
}
