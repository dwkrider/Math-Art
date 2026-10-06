// STL export for the surface viewer.
//
// The mesh on screen is the mesh that comes out: the same baked geometry
// the viewer draws, decoded with the same decodeMesh(), so the file
// cannot describe a different surface from the picture.
//
// SCALE. The baked meshes are normalised to the project's 2 m cube, and
// their stored bounds vary from surface to surface. STL carries no
// units, and essentially every slicer reads it as millimetres, so the
// export is scaled to put the longest side of the bounding box at
// exactly the requested size and centred on the origin. "Fits in a
// 200 mm cube" is then literally true, whatever the surface's aspect
// ratio: a long thin surface reaches 200 mm along its longest axis and
// less on the others.
//
// BINARY, NOT ASCII. A 20k-triangle surface is about 1 MB binary and
// roughly seven times that as ASCII, for a file nothing reads by eye.
//
// WHAT THIS DOES NOT DO. It does not make a surface printable. Many of
// these are mathematical surfaces in the strict sense -- a sheet of zero
// thickness, often with a boundary, sometimes passing through itself or
// non-orientable. An STL of a zero-thickness sheet is not a solid, and a
// slicer will either refuse it or produce nothing. Those need thickening
// first (Blender's Solidify, or the extension's own tools). The closed
// ones -- the quadrics, the tori, the algebraic surfaces bounding a
// region -- come out ready to print. The caller says so in the UI rather
// than this module guessing.

import { decodeMesh, computeNormals } from './surface-viewer.js';

/**
 * Merge vertices that share a position.
 *
 * THE BAKE SPLITS EDGES ON PURPOSE. tools/surfdb_export.py runs
 * `bmesh.ops.split_edges` on every sharp edge so the creases survive
 * into a format that carries no crease data -- the viewer needs that,
 * because those folds ARE the shape. The cost is duplicated vertices
 * along every seam, and a seam then looks exactly like a boundary: two
 * coincident edges, each used by one triangle.
 *
 * Thickening believed those. It built a wall down the middle of 59
 * surfaces' creases, and worse, it thickened surfaces that were never
 * open at all -- the astroidal ellipsoid, the Bouguer dome, Boy's
 * surface and the bubble cluster are closed solids whose boundary count
 * only falls to zero once the seams are welded.
 *
 * Welding is right for an export and wrong for the viewer, which is why
 * it lives here: STL carries no shading, so nothing is lost, and every
 * slicer welds on import anyway.
 */
export function weld(positions, indices, eps = 1e-5) {
  // ONLY VERTICES ON A BOUNDARY EDGE ARE CANDIDATES.
  //
  // Welding everything by position also fuses sheets that merely touch,
  // and a lot of these surfaces pass through themselves on purpose --
  // the Roman surface's triple line, the k-noids' necks. Merging there
  // turns a self-intersection into an edge shared by four faces, and
  // measured over the catalogue it took non-manifold meshes from 105 to
  // 130. The crease seams this needs to repair are all on the boundary,
  // so that is where it looks.
  const onBoundary = new Set();
  for (const [a, b] of boundaryEdges(indices)) {
    onBoundary.add(a);
    onBoundary.add(b);
  }
  const bucket = new Map();
  const remap = new Uint32Array(positions.length / 3);
  const out = [];
  const q = (v) => Math.round(v / eps);
  for (let i = 0; i < positions.length / 3; i++) {
    if (!onBoundary.has(i)) {
      remap[i] = out.length / 3;
      out.push(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      continue;
    }
    const k = q(positions[i * 3]) + ',' + q(positions[i * 3 + 1]) + ','
            + q(positions[i * 3 + 2]);
    let j = bucket.get(k);
    if (j === undefined) {
      j = out.length / 3;
      bucket.set(k, j);
      out.push(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    }
    remap[i] = j;
  }
  const idx = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) idx[i] = remap[indices[i]];
  return { positions: new Float32Array(out), indices: idx };
}


/**
 * Directed edges used by exactly one triangle: the mesh's boundary.
 *
 * Measured from the geometry rather than read off the record, because it
 * is the geometry a slicer will meet. 386 of the 466 baked meshes have
 * one -- they are surfaces in the strict sense, sheets of no thickness --
 * and 80 are already closed.
 */
export function boundaryEdges(indices) {
  const count = new Map();
  const key = (a, b) => (a < b ? a + ':' + b : b + ':' + a);
  for (let t = 0; t < indices.length; t += 3) {
    const tri = [indices[t], indices[t + 1], indices[t + 2]];
    for (let e = 0; e < 3; e++) {
      const a = tri[e], b = tri[(e + 1) % 3];
      const k = key(a, b);
      const rec = count.get(k);
      if (rec) rec.n += 1;
      else count.set(k, { n: 1, a, b });        // keep the first direction
    }
  }
  const out = [];
  for (const rec of count.values()) if (rec.n === 1) out.push([rec.a, rec.b]);
  return out;
}


/**
 * Give a surface thickness, so a slicer has a solid to work with.
 *
 * A mathematical surface has none -- it is a sheet -- and a slicer given
 * one produces nothing at all. This is the same construction as
 * Blender's Solidify: offset the surface along its vertex normals in
 * both directions to make two shells, flip the winding on the inner one
 * so both face outwards, and close the gap between them with a rim built
 * along the boundary edges.
 *
 * The result is checked, not assumed: solidify() is only correct if the
 * output has NO boundary edges left and every edge is used exactly twice
 * in opposite directions. Both are asserted in the tests, because a
 * shell that looks closed and is not slices into something hollow and
 * wrong rather than failing outright.
 *
 * WHAT IT CANNOT FIX. A surface that passes through itself still does
 * after thickening, and a non-orientable one -- Boy's surface, the Klein
 * bottle -- has no consistent side to offset toward, so the two shells
 * cross where the orientation turns over. Blender's Solidify has exactly
 * the same limits. Such a model usually still prints, because slicers
 * resolve self-intersection by winding number, but it is not a clean
 * solid and this module does not pretend otherwise.
 */
/**
 * Make the winding agree across the whole mesh.
 *
 * A surface built patch by patch can be perfectly welded and still
 * have neighbouring patches wound in opposite directions: the
 * Scherk-Collins saddles alternate, and 504 of the hexagon's edges
 * disagreed. Nothing is wrong with the geometry, but every consumer
 * reads the winding as the side the surface faces, so the shading
 * breaks along those edges and a slicer sees a solid turned
 * inside-out in patches.
 *
 * This walks the faces, crossing shared edges, and flips any face
 * that disagrees with the one it was reached from -- the standard
 * flood fill -- but ONLY across edges where exactly two faces meet.
 *
 * That restriction is the whole difficulty. A saddle tower is not a
 * manifold: at every vane the sheets meet along the central axis,
 * three or four of them at once, and "the same way round" has no
 * meaning at such an edge. Walking through those junctions does not
 * merely fail, it actively spreads the disagreement -- the trefoil
 * went from 72 badly wound edges to 124 before this rule was added.
 * Stopping at them orients each manifold piece as well as it can be
 * oriented and leaves the junctions as the surface's own geometry
 * made them.
 *
 * A piece with no boundary of its own is then turned outward, which
 * is what a printer expects; an open sheet is left as it is, since
 * its signed volume means nothing.
 *
 * SOME SURFACES CANNOT BE ORIENTED AT ALL. A one-sided surface -- a
 * Mobius band, and several of the Scherk-Collins rings, whose twist
 * brings the sheet back onto itself reversed -- has no consistent
 * winding to find, and the fill ends up carrying the contradiction
 * to wherever it closes the loop. The walk notices that (it meets a
 * face it has already placed, disagreeing), reports it, and keeps
 * whichever of the two windings has fewer disagreements, so the pass
 * can never make a mesh worse than it found it.
 *
 * Returns { indices, flipped, orientable }.
 */
export function orient(positions, indices) {
  const nf = indices.length / 3;
  if (!nf) return indices;
  // edge -> the faces using it
  const byEdge = new Map();
  for (let f = 0; f < nf; f++) {
    for (let k = 0; k < 3; k++) {
      const a = indices[f * 3 + k], b = indices[f * 3 + (k + 1) % 3];
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      const hit = byEdge.get(key);
      if (hit) hit.push(f); else byEdge.set(key, [f]);
    }
  }
  const out = Int32Array.from(indices);
  const seen = new Uint8Array(nf);
  const uses = (f, a, b) => {            // does face f use edge a->b this way?
    for (let k = 0; k < 3; k++) {
      if (out[f * 3 + k] === a && out[f * 3 + (k + 1) % 3] === b) return true;
    }
    return false;
  };
  const flip = (f) => {
    const t = out[f * 3 + 1];
    out[f * 3 + 1] = out[f * 3 + 2];
    out[f * 3 + 2] = t;
  };
  const pieces = [];
  let orientable = true;
  for (let start = 0; start < nf; start++) {
    if (seen[start]) continue;
    seen[start] = 1;
    const piece = [start];
    const stack = [start];
    while (stack.length) {
      const f = stack.pop();
      for (let k = 0; k < 3; k++) {
        const a = out[f * 3 + k], b = out[f * 3 + (k + 1) % 3];
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        const faces = byEdge.get(key) || [];
        if (faces.length !== 2) continue;          // a junction, or a rim
        for (const g of faces) {
          if (g === f) continue;
          if (seen[g]) {
            // already placed: if it agrees we are fine, and if not
            // this piece admits no consistent winding at all
            if (uses(g, a, b)) orientable = false;
            continue;
          }
          // neighbours agree when they traverse the shared edge the
          // opposite way round; if g uses a->b as well, it is mirrored
          if (uses(g, a, b)) flip(g);
          seen[g] = 1;
          piece.push(g);
          stack.push(g);
        }
      }
    }
    pieces.push(piece);
  }
  // a closed piece should face outwards: the signed volume says which
  for (const piece of pieces) {
    // only if the piece really is closed -- every one of its edges
    // shared with another face of the same piece
    const inPiece = new Set(piece);
    let open = false;
    for (const f of piece) {
      for (let k = 0; k < 3 && !open; k++) {
        const a = out[f * 3 + k], b = out[f * 3 + (k + 1) % 3];
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        const shared = (byEdge.get(key) || []).filter((g) => inPiece.has(g));
        if (shared.length < 2) open = true;
      }
      if (open) break;
    }
    if (open) continue;
    let vol = 0;
    for (const f of piece) {
      const i = out[f * 3] * 3, j = out[f * 3 + 1] * 3, k = out[f * 3 + 2] * 3;
      vol += (positions[i] * (positions[j + 1] * positions[k + 2] - positions[j + 2] * positions[k + 1])
            - positions[i + 1] * (positions[j] * positions[k + 2] - positions[j + 2] * positions[k])
            + positions[i + 2] * (positions[j] * positions[k + 1] - positions[j + 1] * positions[k])) / 6;
    }
    if (vol < 0) for (const f of piece) flip(f);
  }
  // Never hand back something worse than what came in: on a surface
  // with no consistent winding the fill can concentrate the
  // disagreement along a longer seam than it started with.
  const made = indices instanceof Uint16Array ? Uint16Array.from(out)
                                              : Uint32Array.from(out);
  const was = auditMesh(indices).flipped;
  const now = auditMesh(made).flipped;
  return now <= was
    ? { indices: made, flipped: now, orientable }
    : { indices, flipped: was, orientable: false };
}

export function solidify(positions, indices, thickness) {
  const normals = computeNormals(positions, indices);
  const vn = positions.length / 3;
  const h = thickness / 2;
  const pos = new Float32Array(positions.length * 2);
  for (let i = 0; i < vn; i++) {
    for (let k = 0; k < 3; k++) {
      const p = positions[i * 3 + k], d = normals[i * 3 + k] * h;
      pos[i * 3 + k] = p + d;                       // outer shell
      pos[(vn + i) * 3 + k] = p - d;                // inner shell
    }
  }

  const rim = boundaryEdges(indices);
  const tris = indices.length / 3;
  const idx = new Uint32Array(indices.length * 2 + rim.length * 6);
  let o = 0;
  for (let t = 0; t < tris; t++) {                  // outer, as it was
    idx[o++] = indices[t * 3];
    idx[o++] = indices[t * 3 + 1];
    idx[o++] = indices[t * 3 + 2];
  }
  for (let t = 0; t < tris; t++) {                  // inner, reversed
    idx[o++] = vn + indices[t * 3 + 2];
    idx[o++] = vn + indices[t * 3 + 1];
    idx[o++] = vn + indices[t * 3];
  }
  // The wall between the shells. Each rim face must traverse its
  // boundary edge in the OPPOSITE direction to the outer face that owns
  // it -- an edge used twice the same way is two faces back to back, not
  // a closed surface. Getting this backwards still closes every hole, so
  // a boundary-edge count alone reports success; it is the winding audit
  // in the tests that catches it.
  for (const [a, b] of rim) {
    idx[o++] = b; idx[o++] = a; idx[o++] = vn + a;
    idx[o++] = b; idx[o++] = vn + a; idx[o++] = vn + b;
  }
  return { positions: pos, indices: idx };
}


/**
 * What a slicer will make of the geometry, measured rather than assumed.
 *
 * A watertight solid needs every edge used by exactly two triangles, in
 * opposite directions. Anything else -- a hole, an edge shared by four
 * faces where the surface crosses itself, a flipped patch -- is
 * something the reader should hear about before they open a slicer and
 * wonder why the result is hollow.
 *
 * This is not a defect of the thickening. 53 of the baked meshes are
 * already non-manifold before anything touches them, because the
 * surfaces themselves pass through themselves; Boy's surface and the
 * Klein bottle have no consistent side to offset toward at all.
 */
export function auditMesh(indices) {
  const nTris = Math.floor(indices.length / 3);
  if (!nTris) {
    return { holes: 0, nonManifold: 0, flipped: 0, watertight: true };
  }
  // COUNTED BY SORTING, NOT BY A MAP.
  //
  // This used to key two Maps on strings like "17:42". That works up
  // to a point and then stops dead: V8 caps a Map at 2^24 entries, so
  // a mesh past about 5.6 million triangles threw `Map maximum size
  // exceeded` instead of being audited -- and those are exactly the
  // meshes worth auditing, because they are the ones nobody can check
  // by eye. Packing each edge into one number and sorting has no such
  // ceiling, allocates two flat arrays instead of millions of strings,
  // and is faster besides.
  let maxV = 0;
  for (let i = 0; i < indices.length; i++) {
    if (indices[i] > maxV) maxV = indices[i];
  }
  const N = maxV + 1;
  // a*N + b has to stay exactly representable, or two different edges
  // could pack to the same number and the audit would quietly lie
  if (N * N > Number.MAX_SAFE_INTEGER) {
    throw new Error(`auditMesh: ${N} vertices is past exact edge packing`);
  }
  const keys = new Float64Array(nTris * 3);

  const runs = (count) => {
    keys.sort();
    let a = 0, b = 0;                 // edges seen once, and more than twice
    for (let i = 0; i < keys.length;) {
      let j = i + 1;
      while (j < keys.length && keys[j] === keys[i]) j++;
      count(j - i, (n) => { a += n; }, (n) => { b += n; });
      i = j;
    }
    return [a, b];
  };

  // Undirected: an edge used once is a hole, an edge used more than
  // twice is non-manifold.
  let n = 0;
  for (let t = 0; t < indices.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = indices[t + e], b = indices[t + (e + 1) % 3];
      keys[n++] = (a < b ? a : b) * N + (a < b ? b : a);
    }
  }
  const [holes, nonManifold] = runs((c, one, many) => {
    if (c === 1) one(1); else if (c > 2) many(1);
  });

  // Directed: the same edge traversed the same way by two faces means
  // their windings disagree.
  n = 0;
  for (let t = 0; t < indices.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      keys[n++] = indices[t + e] * N + indices[t + (e + 1) % 3];
    }
  }
  const [flipped] = runs((c, one) => { if (c > 1) one(1); });

  return { holes, nonManifold, flipped,
           watertight: holes === 0 && nonManifold === 0 && flipped === 0 };
}


/**
 * Apply the mesh's instance transforms, if it carries any.
 *
 * No baked mesh uses these today, but the format allows them and a
 * compound surface would. Dropping them silently would export one copy
 * of something the viewer draws many times, which is exactly the kind of
 * quiet disagreement between file and picture this module exists to
 * avoid.
 *
 * The matrices are column-major 4x4, as the shader consumes them.
 */
function expand(positions, indices, instances) {
  if (!instances || !instances.length) return { positions, indices };
  const copies = Math.floor(instances.length / 16);
  const vn = positions.length / 3;
  const out = new Float32Array(positions.length * copies);
  const idx = new Uint32Array(indices.length * copies);
  for (let c = 0; c < copies; c++) {
    const m = instances.subarray(c * 16, c * 16 + 16);
    for (let i = 0; i < vn; i++) {
      const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
      const o = (c * vn + i) * 3;
      out[o] = m[0] * x + m[4] * y + m[8] * z + m[12];
      out[o + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
      out[o + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
    for (let k = 0; k < indices.length; k++) {
      idx[c * indices.length + k] = indices[k] + c * vn;
    }
  }
  return { positions: out, indices: idx };
}

/**
 * Build a binary STL from a packed mesh.
 *
 * @param {object} packed  the mesh as served, before decodeMesh
 * @param {object} opts    {name, sizeMM}
 * @returns {{blob: Blob, triangles: number, scale: number, mm: number[]}}
 */
export function buildBinarySTL(packed, opts = {}) {
  const decoded = decodeMesh(packed);
  const { positions, indices } = expand(
    decoded.positions, decoded.indices, decoded.instances);
  return buildBinarySTLFromMesh(positions, indices, opts);
}

/**
 * The same export, from raw geometry rather than a baked surface.
 *
 * The surfaces module arrives with a packed mesh to decode; a module
 * that generates its geometry in the browser -- the Scherk-Collins
 * sculptures -- already has positions and indices in hand. Everything
 * that matters here (welding, the decision to thicken, scaling to
 * millimetres, the byte layout) is the same for both, so it lives in
 * one place.
 */
export function buildBinarySTLFromMesh(positions, indices, opts = {}) {
  const sizeMM = opts.sizeMM || 200;
  const thickness = Math.max(0, opts.thicknessMM || 0);

  const tris = Math.floor(indices.length / 3);
  if (!tris) return null;

  // Bounds of what will actually be written, not the stored lo/hi: the
  // instance transforms above can move geometry outside them.
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k];
      if (v < lo[k]) lo[k] = v;
      if (v > hi[k]) hi[k] = v;
    }
  }
  const ext0 = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const span = Math.max(ext0[0], ext0[1], ext0[2]) || 1;
  const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];

  // THICKENING HAPPENS IN MILLIMETRES, AFTER SCALING.
  //
  // A wall is a physical dimension -- 2 mm is 2 mm whatever the surface's
  // extent in model units -- so the geometry is brought to its final size
  // first and offset afterwards. Doing it the other way would make the
  // wall thickness depend on how big the surface happened to be.
  //
  // The surface is scaled to (size - thickness) rather than size, because
  // the shells then sit half a wall outside it on each side. Scaling to
  // the full size and then offsetting would put the model over the cube
  // it is supposed to fit in.
  // Welded first: the crease seams the bake introduced are not holes,
  // and treating them as such both walls the creases and thickens
  // surfaces that were already closed.
  const merged = thickness > 0
    ? weld(positions, indices)
    : { positions, indices };
  const open = thickness > 0 ? boundaryEdges(merged.indices).length : 0;
  const solid = thickness > 0 && open > 0;
  const s = (solid ? sizeMM - thickness : sizeMM) / span;

  let geo = { positions: new Float32Array(merged.positions.length),
              indices: merged.indices };
  for (let i = 0; i < merged.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      geo.positions[i + k] = (merged.positions[i + k] - c[k]) * s;
    }
  }
  if (solid) geo = solidify(geo.positions, geo.indices, thickness);

  const P = geo.positions, I = geo.indices;
  const nTris = I.length / 3;

  // Final bounds, measured after thickening: the offset does not add a
  // full wall on every axis, so the model is usually a little under.
  const flo = [Infinity, Infinity, Infinity];
  const fhi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (P[i + k] < flo[k]) flo[k] = P[i + k];
      if (P[i + k] > fhi[k]) fhi[k] = P[i + k];
    }
  }
  const ext = [fhi[0] - flo[0], fhi[1] - flo[1], fhi[2] - flo[2]];
  const shift = [0, 1, 2].map((k) => (flo[k] + fhi[k]) / 2);

  const buf = new ArrayBuffer(84 + nTris * 50);
  const view = new DataView(buf);

  // The 80-byte header. It must NOT begin with "solid": a reader that
  // sniffs the format sees that word and tries to parse the file as
  // ASCII, which fails on the first byte of binary payload.
  const header = `Math Art - ${opts.name || 'surface'} - ${sizeMM}mm`;
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < 80 && i < header.length; i++) {
    const code = header.charCodeAt(i);
    bytes[i] = code < 128 ? code : 63;              // '?' for anything else
  }
  view.setUint32(80, nTris, true);

  let off = 84;
  const ax = [0, 0, 0], bx = [0, 0, 0];
  for (let t = 0; t < nTris; t++) {
    const ia = I[t * 3] * 3, ib = I[t * 3 + 1] * 3, ic = I[t * 3 + 2] * 3;
    // Already in millimetres; only the post-thickening recentre remains.
    const v = [ia, ib, ic].map((i) => [
      P[i] - shift[0], P[i + 1] - shift[1], P[i + 2] - shift[2],
    ]);
    for (let k = 0; k < 3; k++) {
      ax[k] = v[1][k] - v[0][k];
      bx[k] = v[2][k] - v[0][k];
    }
    let nx = ax[1] * bx[2] - ax[2] * bx[1];
    let ny = ax[2] * bx[0] - ax[0] * bx[2];
    let nz = ax[0] * bx[1] - ax[1] * bx[0];
    const L = Math.hypot(nx, ny, nz);
    // A degenerate triangle gets a zero normal rather than being
    // dropped. Dropping it would silently change the mesh, and every
    // consumer recomputes the normal from the winding anyway.
    if (L > 1e-20) { nx /= L; ny /= L; nz /= L; } else { nx = ny = nz = 0; }
    view.setFloat32(off, nx, true);
    view.setFloat32(off + 4, ny, true);
    view.setFloat32(off + 8, nz, true);
    off += 12;
    for (let j = 0; j < 3; j++) {
      view.setFloat32(off, v[j][0], true);
      view.setFloat32(off + 4, v[j][1], true);
      view.setFloat32(off + 8, v[j][2], true);
      off += 12;
    }
    view.setUint16(off, 0, true);
    off += 2;
  }

  return {
    blob: new Blob([buf], { type: 'model/stl' }),
    triangles: nTris,
    scale: s,
    mm: ext,
    thickened: solid,
    wasOpen: open > 0,
    audit: auditMesh(I),
  };
}

/** Hand a built STL to the browser as a download. */
export function downloadSTL(built, slug) {
  const url = URL.createObjectURL(built.blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${slug}-${Math.round(Math.max(...built.mm))}mm.stl`;
  document.body.append(a);
  a.click();
  a.remove();
  // Revoked on the next turn of the event loop: revoking synchronously
  // races the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
