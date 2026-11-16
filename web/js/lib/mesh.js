// Shared mesh arithmetic: the computation layer of the library.
//
// Pure maths -- no DOM, no three.js -- so it runs in node under the
// parity tests exactly as it does in a browser.
//
// These two were written for the TPMS rim tube and are wanted again
// for the Seifert knot tube. A tube swept along a mesh's boundary is
// centred on that boundary unless something tells it which way is out,
// and half of it then sits inside the surface with the surface poking
// through it. The answer is the same in both cases, which is why it
// lives here rather than in either module.
//
// Ported from math_art/rim_curve.py (`neighbour_means`, `outward_field`)
// and pinned against it by tests/web/test_tpms.mjs.

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
