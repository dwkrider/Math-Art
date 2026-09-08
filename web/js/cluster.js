// Similarity layout for the surface catalogue.
//
// The grid answers "what is there"; this answers "what is near what".
// Surfaces are placed by a spring embedder so that ones the database
// describes similarly end up close together, and the families separate on
// their own rather than being drawn as labelled boxes.
//
// Everything is computed here, in the browser, from the same index.json
// the grid already loads. No layout is precomputed and shipped.
//
// THE SIMILARITY. Each surface becomes a sparse weighted vector over its
// categorical fields, and similarity is the cosine between two of them.
// The weights are a judgement about what makes two surfaces alike, so
// they are written down here rather than tuned invisibly:
//
//   family        the strongest signal, and the one a reader recognises
//   curvature     minimal / CMC / flat is a real kinship across families
//   definition    how the surface is GIVEN -- an implicit sextic and a
//                 Weierstrass minimal surface are different kinds of
//                 object even when both are, say, genus 3
//   symmetry      kind (space / rod / point / none) and periodicity rank
//   topology      ends, embedding quality, orientability
//   tags          the 56 free-form family tags: individually weak, but
//                 collectively good at pulling near-relatives together
//
// genus and orientable are deliberately light. They are populated for
// only 61 and 108 of the 473 records, so leaning on them would cluster by
// "has the field filled in" rather than by mathematics.

const W = {
  family: 3.0,
  curvature: 2.0,
  mode: 1.6,
  symKind: 1.2,
  periodicity: 1.2,
  ends: 0.8,
  embedding: 0.8,
  genus: 0.5,
  orientable: 0.4,
  tag: 0.45,
};

/** Sparse unit-length feature vector for one index entry. */
export function featureVector(e) {
  const v = new Map();
  const put = (k, w) => v.set(k, (v.get(k) || 0) + w);

  put('fam:' + e.primary_family, W.family);
  put('cur:' + (e.curvature_condition || 'none'), W.curvature);
  put('mode:' + (e.definition_mode || 'unknown'), W.mode);
  put('sym:' + (e.symmetry && e.symmetry.kind ? e.symmetry.kind : 'none'),
      W.symKind);
  put('per:' + (e.periodicity_rank ?? 0), W.periodicity);

  // Ends are counted, but "many" is not meaningfully different from
  // "slightly more many", so the tail is bucketed.
  const ends = e.ends ?? 0;
  put('ends:' + (ends > 4 ? 'many' : ends), W.ends);
  put('emb:' + (e.embedding || 'unknown'), W.embedding);

  if (e.genus !== null && e.genus !== undefined) {
    put('genus:' + (e.genus > 5 ? 'high' : e.genus), W.genus);
  }
  if (e.orientable !== null && e.orientable !== undefined) {
    put('or:' + e.orientable, W.orientable);
  }
  for (const t of e.families || []) put('tag:' + t, W.tag);

  // Unit length, so cosine is a plain dot product and a record carrying
  // more tags is not automatically "bigger" than one carrying fewer.
  let n = 0;
  for (const x of v.values()) n += x * x;
  n = Math.sqrt(n) || 1;
  for (const [k, x] of v) v.set(k, x / n);
  return v;
}

function cosine(a, b) {
  // Both are unit length, so walking the shorter one is enough.
  const s = a.size < b.size ? a : b;
  const l = a.size < b.size ? b : a;
  let d = 0;
  for (const [k, x] of s) {
    const y = l.get(k);
    if (y) d += x * y;
  }
  return d;
}

/**
 * A k-nearest-neighbour graph over the entries.
 *
 * Springs on every pair pull the whole set into one blob: 473 nodes is
 * 111k springs, and the many weak similarities swamp the few strong ones.
 * Keeping each node's k best neighbours cuts that back.
 *
 * k = 40 was measured, not guessed, and the result was the opposite of
 * what I expected. Separation -- mean distance between different-family
 * pairs over same-family pairs -- goes 1.30x at k=4, 1.22x at k=6, 1.63x
 * at k=10, 3.99x at k=28, 9.35x at k=40, then stops improving (9.28x at
 * k=60). A very sparse graph gives the layout too little to work with:
 * each node is pinned by a handful of edges and the global arrangement
 * stays near its random start. Enough edges, and the families genuinely
 * pull themselves apart.
 */
export function knnGraph(entries, k = 40) {
  const vecs = entries.map(featureVector);
  const edges = [];
  const seen = new Set();
  for (let i = 0; i < entries.length; i++) {
    const best = [];
    for (let j = 0; j < entries.length; j++) {
      if (i === j) continue;
      const s = cosine(vecs[i], vecs[j]);
      if (best.length < k) {
        best.push([s, j]);
        best.sort((p, q) => p[0] - q[0]);
      } else if (s > best[0][0]) {
        best[0] = [s, j];
        best.sort((p, q) => p[0] - q[0]);
      }
    }
    for (const pair of best) {
      const j = pair[1];
      const key = i < j ? i + ':' + j : j + ':' + i;
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push({ a: i, b: j, w: pair[0] });
    }
  }
  return edges;
}

// A seeded generator, so the same catalogue always lays out the same way.
// A picture that reshuffled on every reload would read as arbitrary,
// which is the opposite of what this view is for.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Fruchterman-Reingold, stepped so the caller can animate it.
 *
 * Attraction along the kNN edges, repulsion between all pairs. 473 nodes
 * is 112k pairs per iteration, small enough to do directly -- a
 * Barnes-Hut tree would be faster asymptotically and slower here.
 */
export class SpringLayout {
  constructor(n, edges, opts = {}) {
    const width = opts.width || 1000;
    const height = opts.height || 700;
    this.n = n;
    this.edges = edges;
    this.w = width;
    this.h = height;
    // The size a node occupies on screen. Overlap is part of the energy
    // rather than something repaired afterwards: a post-hoc shuffle moves
    // nodes away from the arrangement the layout just solved for, so what
    // is drawn is no longer what converged. Below this distance the
    // repulsion becomes stiff, so the layout settles into an arrangement
    // that has room for the tiles in the first place.
    this.minDist = opts.minDist || 0;
    this.collide = opts.collide === undefined ? 300 : opts.collide;
    // Iterations over which the collision term is brought up to strength.
    // Applied from the start it competes with the edges while the clusters
    // are still forming, and the layout settles into a more even, less
    // grouped field: measured over the full 473, separation is 5.07x with
    // no ramp against 6.21x with one, and overlap is worse as well (119
    // overlapping tiles against 2). Cluster first, then make room.
    this.ramp = opts.ramp === undefined ? 300 : opts.ramp;
    this.pad = (this.minDist / 2) + 6;
    this.x = new Float64Array(n);
    this.y = new Float64Array(n);
    this.dx = new Float64Array(n);
    this.dy = new Float64Array(n);

    const rand = rng(opts.seed || 20260907);
    for (let i = 0; i < n; i++) {
      // Start on a disc, not a square: a square's corners bias the first
      // iterations outward along the diagonals.
      const r = Math.sqrt(rand()) * Math.min(width, height) * 0.45;
      const t = rand() * Math.PI * 2;
      this.x[i] = width / 2 + r * Math.cos(t);
      this.y[i] = height / 2 + r * Math.sin(t);
    }
    this.k = Math.sqrt((width * height) / Math.max(1, n));
    this.temp = Math.min(width, height) * 0.10;
    this.iter = 0;
  }

  step(times) {
    for (let t = 0; t < (times || 1); t++) this._one();
    return this;
  }

  _one() {
    // Squared, so the term stays near zero through the early iterations
    // instead of fading in linearly from the first one.
    const r = Math.min(1, this.iter / Math.max(1, this.ramp));
    const collide = this.collide * r * r;
    const n = this.n;
    const x = this.x, y = this.y, dx = this.dx, dy = this.dy, k = this.k;
    dx.fill(0);
    dy.fill(0);

    for (let i = 0; i < n; i++) {
      const xi = x[i], yi = y[i];
      for (let j = i + 1; j < n; j++) {
        let ex = xi - x[j];
        let ey = yi - y[j];
        let d2 = ex * ex + ey * ey;
        if (d2 < 1e-6) { ex = 1e-3; ey = 1e-3; d2 = 2e-6; }
        let f = (k * k) / d2;                 // repulsion ~ k^2 / d
        // Collision term. Within a tile's width the pair is pushed apart
        // hard, and the force grows as they close, so pairs separate
        // without the whole field being blown up the way a uniform scaling
        // would. Squared in the overlap, so it dominates on contact and
        // vanishes immediately beyond it -- clusters stay clusters.
        //
        // The coefficient has to be large because of what it competes
        // with: FR attraction grows as d^2/k, so each of a node's ~40
        // edges pulls with ~28 at contact distance, while plain repulsion
        // there is ~1.5. A term of the same order as the repulsion (6 was
        // the first try) is simply not felt -- it left 441 of 473 tiles
        // overlapping. 300 is where overlap goes to 2 without the field
        // flattening: pushed further, to 1200, it beats the edges as well
        // and separation falls from 6.2x to 4.9x.
        const md = this.minDist;
        if (md > 0 && d2 < md * md) {
          const d = Math.sqrt(d2);
          const over = (md - d) / md;
          f += (k * k) * collide * over * over / Math.max(d2, 1e-4);
        }
        dx[i] += ex * f; dy[i] += ey * f;
        dx[j] -= ex * f; dy[j] -= ey * f;
      }
    }

    for (const e of this.edges) {
      const ex = x[e.a] - x[e.b];
      const ey = y[e.a] - y[e.b];
      const d = Math.hypot(ex, ey) || 1e-3;
      // Attraction scaled by similarity, so a strong pair settles closer
      // than a weak one rather than every edge having one rest length.
      const f = (d * d) / k * (0.35 + e.w);
      const ux = (ex / d) * f;
      const uy = (ey / d) * f;
      dx[e.a] -= ux; dy[e.a] -= uy;
      dx[e.b] += ux; dy[e.b] += uy;
    }

    const temp = this.temp;
    const pad = this.pad;
    const mx = this.w - pad, my = this.h - pad;
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(dx[i], dy[i]) || 1e-9;
      const m = Math.min(d, temp);
      x[i] += (dx[i] / d) * m;
      y[i] += (dy[i] / d) * m;
      // Keep the field on the canvas. Without this the collision term
      // pushes it outward indefinitely and the caller has to scale it
      // back, which is the shrinking this was meant to avoid.
      if (x[i] < pad) x[i] = pad; else if (x[i] > mx) x[i] = mx;
      if (y[i] < pad) y[i] = pad; else if (y[i] > my) y[i] = my;
    }
    // Cool slowly; stopping early leaves clusters still overlapping.
    this.temp = Math.max(temp * 0.985, 0.35);
    this.iter++;
  }

  /** Positions rescaled to fit a box, with a margin. */
  fitted(width, height, pad) {
    pad = pad === undefined ? 30 : pad;
    let lox = Infinity, loy = Infinity, hix = -Infinity, hiy = -Infinity;
    for (let i = 0; i < this.n; i++) {
      if (this.x[i] < lox) lox = this.x[i];
      if (this.y[i] < loy) loy = this.y[i];
      if (this.x[i] > hix) hix = this.x[i];
      if (this.y[i] > hiy) hiy = this.y[i];
    }
    const s = Math.min((width - 2 * pad) / Math.max(1e-6, hix - lox),
                       (height - 2 * pad) / Math.max(1e-6, hiy - loy));
    const ox = pad + (width - 2 * pad - (hix - lox) * s) / 2;
    const oy = pad + (height - 2 * pad - (hiy - loy) * s) / 2;
    const px = new Float64Array(this.n);
    const py = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      px[i] = ox + (this.x[i] - lox) * s;
      py[i] = oy + (this.y[i] - loy) * s;
    }
    return { x: px, y: py };
  }
}
