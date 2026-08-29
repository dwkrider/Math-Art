// Tessellation of polyhedron faces as the database stores them.
//
// This is a port of tools/polyhedra_tessellate.py, and it has to stay a
// faithful one: the thumbnails in web/thumbs/ are rendered by the Python
// side and the live view here is drawn by this file, so if the two
// disagree the catalogue tile stops matching the model beside it.
//
// Records store each face as its true winding cycle rather than as a
// pre-triangulated patch, which is what lets a {5/2} pentagram be a
// single face. Turning such a cycle into triangles has three traps, and
// the database contains all three:
//
//   * a triangle fan is wrong for the ~7,600 concave faces;
//   * a fill keyed off the polygon's TURNING number drops the 568
//     crossed quadrilaterals of the uniform duals, whose turning number
//     is 0 though both lobes are plainly there;
//   * the stored metadata cannot find the star faces for you --
//     faces_by_type labels a pentagram "{5}", so self-intersection has
//     to be detected geometrically.
//
// One routine handles all of it: project the face to its plane, split
// every edge at its proper crossings, walk the resulting subdivision
// into bounded regions, and keep a region when the ORIGINAL cycle's
// winding number about an interior point of it is non-zero. Kept
// regions are simple polygons and get ear-clipped.
//
// The nonzero rule rather than even-odd is deliberate: even-odd would
// punch the inner pentagon out of a pentagram, which is not how these
// solids are drawn.

const EPS = 1e-9;

// -- plane projection -------------------------------------------------

function newellNormal(pts) {
  const n = [0, 0, 0];
  const m = pts.length;
  for (let i = 0; i < m; i++) {
    const p = pts[i], q = pts[(i + 1) % m];
    n[0] += (p[1] - q[1]) * (p[2] + q[2]);
    n[1] += (p[2] - q[2]) * (p[0] + q[0]);
    n[2] += (p[0] - q[0]) * (p[1] + q[1]);
  }
  let ln = Math.hypot(n[0], n[1], n[2]);
  if (ln >= EPS) return n.map((t) => t / ln);

  // Newell CANCELS on a symmetric crossed quadrilateral, whose two lobes
  // contribute equal and opposite area -- and the uniform duals are full
  // of those. The face is still planar, so take the normal from the
  // widest triple of vertices instead. Its sign is then arbitrary, which
  // is honest: a crossed quad has no consistent outward side.
  let best = null, bestLn = EPS;
  for (let i = 1; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      const a = [0, 1, 2].map((k) => pts[i][k] - pts[0][k]);
      const b = [0, 1, 2].map((k) => pts[j][k] - pts[0][k]);
      const c = [a[1] * b[2] - a[2] * b[1],
                 a[2] * b[0] - a[0] * b[2],
                 a[0] * b[1] - a[1] * b[0]];
      const cl = Math.hypot(c[0], c[1], c[2]);
      if (cl > bestLn) { best = c; bestLn = cl; }
    }
  }
  return best ? best.map((t) => t / bestLn) : null;
}

export function planeBasis(pts) {
  const m = pts.length;
  const c = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / m);
  const nrm = newellNormal(pts);
  if (!nrm) return null;
  // Aim u at the first vertex so the frame is deterministic: the same
  // face always projects to the same 2-D coordinates.
  let u = [0, 1, 2].map((k) => pts[0][k] - c[k]);
  let ln = Math.hypot(u[0], u[1], u[2]);
  if (ln < EPS) {
    const seed = Math.abs(nrm[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const d = seed[0] * nrm[0] + seed[1] * nrm[1] + seed[2] * nrm[2];
    u = [0, 1, 2].map((k) => seed[k] - nrm[k] * d);
    ln = Math.hypot(u[0], u[1], u[2]);
    if (ln < EPS) return null;
  }
  u = u.map((t) => t / ln);
  const w = [nrm[1] * u[2] - nrm[2] * u[1],
             nrm[2] * u[0] - nrm[0] * u[2],
             nrm[0] * u[1] - nrm[1] * u[0]];
  return { c, u, w, n: nrm };
}

function project(pts, f) {
  return pts.map((p) => {
    const d = [0, 1, 2].map((k) => p[k] - f.c[k]);
    return [d[0] * f.u[0] + d[1] * f.u[1] + d[2] * f.u[2],
            d[0] * f.w[0] + d[1] * f.w[1] + d[2] * f.w[2]];
  });
}

function unproject(q, f) {
  return [0, 1, 2].map((k) => f.c[k] + q[0] * f.u[k] + q[1] * f.w[k]);
}

// -- 2-D primitives ---------------------------------------------------

function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

function signedArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return s / 2;
}

// Winding number about `pt`, evaluated against the ORIGINAL cycle. This
// is what makes a pentagram's inner pentagon come out 2 and a crossed
// quad's two lobes +1 and -1.
function windingNumber(poly, pt) {
  let wn = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if (a[1] <= pt[1]) {
      if (b[1] > pt[1] && cross(a, b, pt) > 0) wn++;
    } else if (b[1] <= pt[1] && cross(a, b, pt) < 0) wn--;
  }
  return wn;
}

// Only PROPER crossings matter; shared endpoints are already shared
// vertices of the cycle and need no split.
function segIntersect(p1, p2, p3, p4) {
  const d1 = [p2[0] - p1[0], p2[1] - p1[1]];
  const d2 = [p4[0] - p3[0], p4[1] - p3[1]];
  const den = d1[0] * d2[1] - d1[1] * d2[0];
  if (Math.abs(den) < EPS) return null;
  const dx = p3[0] - p1[0], dy = p3[1] - p1[1];
  const t = (dx * d2[1] - dy * d2[0]) / den;
  const s = (dx * d1[1] - dy * d1[0]) / den;
  if (t <= EPS || t >= 1 - EPS || s <= EPS || s >= 1 - EPS) return null;
  return [t, s];
}

function adjacentInCycle(i, j, m) {
  return j === i || (j + 1) % m === i || (i + 1) % m === j;
}

export function hasSelfIntersection(poly) {
  const m = poly.length;
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      if (adjacentInCycle(i, j, m)) continue;
      if (segIntersect(poly[i], poly[(i + 1) % m],
                       poly[j], poly[(j + 1) % m])) return true;
    }
  }
  return false;
}

// -- ear clipping -----------------------------------------------------

function pointInTriangle(p, a, b, c) {
  const d1 = cross(a, b, p), d2 = cross(b, c, p), d3 = cross(c, a, p);
  const neg = d1 < -EPS || d2 < -EPS || d3 < -EPS;
  const pos = d1 > EPS || d2 > EPS || d3 > EPS;
  return !(neg && pos);
}

// Handles concave polygons, which a fan does not -- and roughly a third
// of the database's faces are concave.
export function earClip(poly) {
  const m = poly.length;
  if (m < 3) return [];
  let idx = [...Array(m).keys()];
  if (signedArea(poly) < 0) idx.reverse();   // work counter-clockwise
  const tris = [];
  let guard = 0;
  while (idx.length > 3 && guard < 4 * m * m) {
    guard++;
    let clipped = false;
    for (let k = 0; k < idx.length; k++) {
      const i0 = idx[(k - 1 + idx.length) % idx.length];
      const i1 = idx[k];
      const i2 = idx[(k + 1) % idx.length];
      const a = poly[i0], b = poly[i1], c = poly[i2];
      if (cross(a, b, c) <= EPS) continue;             // reflex
      let blocked = false;
      for (const j of idx) {
        if (j === i0 || j === i1 || j === i2) continue;
        if (pointInTriangle(poly[j], a, b, c)) { blocked = true; break; }
      }
      if (blocked) continue;
      tris.push([i0, i1, i2]);
      idx.splice(k, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;                     // numerically stuck; salvage
  }
  if (idx.length === 3) tris.push([idx[0], idx[1], idx[2]]);
  return tris;
}

// -- planar subdivision -----------------------------------------------

function dedupePoint(store, pt, tol = 1e-7) {
  for (let i = 0; i < store.length; i++) {
    if (Math.abs(store[i][0] - pt[0]) < tol &&
        Math.abs(store[i][1] - pt[1]) < tol) return i;
  }
  store.push(pt);
  return store.length - 1;
}

function subdivide(poly) {
  const m = poly.length;
  const pts = [];
  const ring = poly.map((p) => dedupePoint(pts, p));
  const splits = Array.from({ length: m }, () => []);
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      if (adjacentInCycle(i, j, m)) continue;
      const a = poly[i], b = poly[(i + 1) % m];
      const c = poly[j], d = poly[(j + 1) % m];
      const hit = segIntersect(a, b, c, d);
      if (!hit) continue;
      const [t, s] = hit;
      const p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
      const pi = dedupePoint(pts, p);
      splits[i].push([t, pi]);
      splits[j].push([s, pi]);
    }
  }
  const edges = new Set();
  for (let i = 0; i < m; i++) {
    const chain = [[0, ring[i]], ...splits[i].slice().sort((x, y) => x[0] - y[0]),
                   [1, ring[(i + 1) % m]]];
    for (let k = 0; k < chain.length - 1; k++) {
      const a = chain[k][1], b = chain[k + 1][1];
      if (a !== b) edges.add(Math.min(a, b) + ':' + Math.max(a, b));
    }
  }
  return { pts, edges: [...edges].map((s) => s.split(':').map(Number)) };
}

// Standard half-edge traversal: arriving at v from u, leave along the
// neighbour next CLOCKWISE from u around v. That keeps the region on the
// left, so bounded faces come out counter-clockwise (positive area) and
// the single outer face comes out negative.
function facesOf(pts, edges) {
  const adj = new Map();
  for (const [a, b] of edges) {
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push(b);
    adj.get(b).push(a);
  }
  const order = new Map();
  for (const [v, nbrs] of adj) {
    nbrs.sort((p, q) =>
      Math.atan2(pts[p][1] - pts[v][1], pts[p][0] - pts[v][0]) -
      Math.atan2(pts[q][1] - pts[v][1], pts[q][0] - pts[v][0]));
    order.set(v, new Map(nbrs.map((w, k) => [w, k])));
  }
  const unused = new Set();
  for (const [a, b] of edges) { unused.add(a + ':' + b); unused.add(b + ':' + a); }
  const faces = [];
  while (unused.size) {
    const start = unused.values().next().value;
    let cur = start;
    const cycle = [];
    while (unused.has(cur)) {
      unused.delete(cur);
      const [u, v] = cur.split(':').map(Number);
      cycle.push(u);
      const nbrs = adj.get(v);
      const k = order.get(v).get(u);
      const nxt = nbrs[(k - 1 + nbrs.length) % nbrs.length];
      cur = v + ':' + nxt;
    }
    if (cycle.length >= 3) faces.push(cycle);
  }
  return faces;
}

// A point strictly inside a simple polygon: the centroid of an ear.
function interiorPoint(poly) {
  const tris = earClip(poly);
  if (!tris.length) return null;
  const [i0, i1, i2] = tris[0];
  return [(poly[i0][0] + poly[i1][0] + poly[i2][0]) / 3,
          (poly[i0][1] + poly[i1][1] + poly[i2][1]) / 3];
}

export function fillPolygon2d(poly) {
  if (poly.length < 3) return { pts: poly.slice(), tris: [] };
  if (!hasSelfIntersection(poly)) {
    return { pts: poly.slice(), tris: earClip(poly) };
  }
  const { pts, edges } = subdivide(poly);
  const tris = [];
  for (const cycle of facesOf(pts, edges)) {
    const ring = cycle.map((i) => pts[i]);
    if (signedArea(ring) <= EPS) continue;        // outer face, or degenerate
    const inside = interiorPoint(ring);
    if (!inside || windingNumber(poly, inside) === 0) continue;
    for (const [a, b, c] of earClip(ring)) {
      tris.push([cycle[a], cycle[b], cycle[c]]);
    }
  }
  return { pts, tris };
}

/**
 * Triangulate one stored face cycle.
 *
 * `points3d` are the face's vertex positions in stored winding order.
 * Returns { points, tris, normal }; `points` begins with the input
 * points and is extended with any intersection points, so callers can
 * append the extras to their own vertex array and offset the indices.
 */
export function tessellateFace(points3d) {
  const flat = { points: points3d.slice(), tris: [], normal: [0, 0, 1] };
  if (points3d.length < 3) return flat;
  const frame = planeBasis(points3d);
  if (!frame) return flat;
  const poly = project(points3d, frame);
  const { pts, tris } = fillPolygon2d(poly);
  const points = points3d.slice();
  for (let i = points3d.length; i < pts.length; i++) {
    points.push(unproject(pts[i], frame));
  }
  return { points, tris, normal: frame.n };
}
