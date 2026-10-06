// Seifert surfaces of braid closures.
//
// A port of math_art/seifert/: braid word -> Kauffman state -> disk/band
// combinatorics -> quad mesh -> Catmull-Clark refinement.
// tests/web/test_seifert.mjs runs that engine and checks this against it.
//
// WHAT A SEIFERT SURFACE IS. Take a knot -- a closed loop of string in
// space. Is there a surface whose only edge is that loop, the way a soap
// film spans a wire? Seifert showed in 1934 that there always is, and
// gave a recipe for building one. Orient the knot, walk along it, and at
// every crossing take the turn that respects the direction of travel.
// The loop falls apart into a set of disjoint circles, each of which can
// be filled with a disk; glue the disks back together with one twisted
// band at each crossing, and the result is a surface bounded by exactly
// the knot you started with.
//
// Its genus -- how many handles it has -- is a measurable property of
// the knot, not an accident of the construction, and that is what makes
// the surface worth building rather than merely drawing.
//
// SEIFERT'S IS ONE CHOICE AMONG MANY. At each crossing there are two
// ways to resolve it, so a diagram with c crossings carries 2^c spanning
// surfaces. Seifert's algorithm takes the oriented resolution every
// time. The others are state surfaces too, and most of them are
// ONE-SIDED -- the trefoil's all-turnback state is a Mobius band with
// three half-twists whose edge is the trefoil. Which ones are two-sided
// has a clean answer: exactly those whose state graph is bipartite.
//
// References:
// - Herbert Seifert, "Ueber das Geschlecht von Knoten", Math. Ann. 110,
//   1934 -- the algorithm and the genus.
// - J. J. van Wijk & A. M. Cohen, "Visualization of Seifert Surfaces",
//   IEEE TVCG 12(4), 2006, 485-496 -- the stacked disk/band layout this
//   follows, sections IV and V.
// - E. Kalfagianni, "State surfaces of links", arXiv:1804.05281, Lemma
//   2.2 -- the bipartite orientability criterion.
// - W. Wang, B. Juttler, D. Zheng, Y. Liu, "Computation of Rotation
//   Minimizing Frames", ACM TOG 27(1), 2008 -- the band frames.

export const TAU = 2 * Math.PI;

// ------------------------------------------------------- braid words

const TOKEN = /([A-Za-z])(\d*)/g;

/** Parse letter notation into crossings.
 *
 *  An uppercase letter is a right-hand crossing, the matching lowercase
 *  one a left-hand crossing, and the letter picks the strand pair --
 *  A = strands 1-2, B = 2-3. A trailing integer repeats, so "A3" is
 *  three successive crossings. Crossings are stored EXPANDED, one entry
 *  each, because Seifert's algorithm gives one band per crossing.
 *
 *  THE CONVENTION TRAP, carried over from the engine: this follows van
 *  Wijk and Cohen, where uppercase is the positive crossing. The
 *  repository's `knots` package follows Gittings, where lowercase is
 *  positive, so the same string means mirror-image braids there. The
 *  two exchange signed-integer words, never strings.
 */
export function parseBraid(word, strands = null) {
  const text = String(word).trim();
  const crossings = [];
  let pos = 0;
  TOKEN.lastIndex = 0;
  let m;
  while ((m = TOKEN.exec(text)) !== null) {
    if (m.index !== pos) throw new Error(`cannot parse ${text} at index ${pos}`);
    pos = m.index + m[0].length;
    const repeats = m[2] ? parseInt(m[2], 10) : 1;
    if (repeats < 1) throw new Error(`repeat count must be >= 1 in ${m[0]}`);
    const row = m[1].toUpperCase().charCodeAt(0) - 65;
    const sign = m[1] === m[1].toUpperCase() ? 1 : -1;
    for (let k = 0; k < repeats; k++) crossings.push({ row, sign });
  }
  if (pos !== text.length) throw new Error(`trailing junk in ${text}`);
  if (!crossings.length) throw new Error(`no crossings parsed from ${word}`);
  const implied = Math.max(...crossings.map((c) => c.row)) + 2;
  if (strands !== null && strands < implied) {
    throw new Error(`${text} needs at least ${implied} strands`);
  }
  return { word: text, crossings, strands: strands === null ? implied : strands };
}

/** Signed-integer form: +i is the Artin generator s_i, -i its inverse. */
export function signedWord(braid) {
  return braid.crossings.map((c) => (c.row + 1) * c.sign);
}

/** Strand permutation of the closure. The sign is IGNORED -- only which
 *  pair each crossing transposes matters. */
export function strandPermutation(braid) {
  const perm = Array.from({ length: braid.strands }, (_, i) => i);
  for (const c of braid.crossings) {
    const i = c.row;
    [perm[i], perm[i + 1]] = [perm[i + 1], perm[i]];
  }
  return perm;
}

export function countCycles(perm) {
  const seen = new Set();
  let cycles = 0;
  for (let s = 0; s < perm.length; s++) {
    if (seen.has(s)) continue;
    cycles++;
    let cur = s;
    while (!seen.has(cur)) { seen.add(cur); cur = perm[cur]; }
  }
  return cycles;
}

/** Link components: the cycles of the strand permutation. */
export function nComponents(braid) {
  return countCycles(strandPermutation(braid));
}

/** The braid word of the (p, q) torus knot: (s1 s2 ... s_{p-1})^q. */
export function torusKnot(p, q) {
  if (p < 2 || q < 1) throw new Error('need p >= 2 and q >= 1');
  let base = '';
  for (let i = 0; i < p - 1; i++) base += String.fromCharCode(65 + i);
  return parseBraid(base.repeat(q));
}

// ------------------------------------------- Kauffman states, traced

/** Boundary components of a ribbon graph whose every band is twisted.
 *
 *  Darts are (disk, foot). Follow a dart across its band -- flipping
 *  which side you are on, because the band is twisted -- then step to
 *  the next foot around the far disk. The orbits of that permutation
 *  pair up, one pair per boundary circle. */
export function boundaryCount(feet, bands, reversed = null, twisted = true) {
  const rev = reversed || feet.map(() => false);
  const tw = typeof twisted === 'boolean' ? bands.map(() => twisted) : twisted;
  const key = (d, f) => `${d},${f}`;
  const partner = new Map();
  const flip = new Map();
  bands.forEach(([a, fa, b, fb], i) => {
    partner.set(key(a, fa), [b, fb]);
    partner.set(key(b, fb), [a, fa]);
    flip.set(key(a, fa), !!tw[i]);
    flip.set(key(b, fb), !!tw[i]);
  });
  const step = (dart, direction) => {
    const [disk, foot] = partner.get(dart);
    const turn = direction * (rev[disk] ? -1 : 1);
    return [key(disk, (((foot + turn) % feet[disk]) + feet[disk]) % feet[disk]),
            direction];
  };
  const seen = new Set();
  let orbits = 0;
  for (const dart of partner.keys()) {
    for (const start of [1, -1]) {
      if (seen.has(`${dart}|${start}`)) continue;
      orbits++;
      let cur = dart, side = start;
      while (!seen.has(`${cur}|${side}`)) {
        seen.add(`${cur}|${side}`);
        const nextSide = flip.get(cur) ? -side : side;
        const [moved, sideNow] = step(cur, nextSide);
        cur = moved;
        side = sideNow;
      }
    }
  }
  return orbits / 2;
}

/** Edges of the resolved diagram, and which edges each band attaches to.
 *
 *  Nodes are strand slots (level, row); every node has degree two, so
 *  the edges form disjoint cycles -- the state circles. */
function resolvedDiagram(braid, state) {
  const n = braid.strands;
  const c = braid.crossings.length;
  const node = (level, row) => level * n + row;
  const edges = [];
  const attach = [];
  const add = (u, v) => { edges.push([u, v]); return edges.length - 1; };
  for (let q = 1; q <= c; q++) {
    const k = braid.crossings[q - 1].row;
    const turn = state[q - 1];
    const horizontal = {};
    for (let row = 0; row < n; row++) {
      if (turn && (row === k || row === k + 1)) continue;
      horizontal[row] = add(node(q - 1, row), node(q, row));
    }
    if (turn) {
      const low = add(node(q - 1, k), node(q - 1, k + 1));   // cup
      const high = add(node(q, k), node(q, k + 1));          // cap
      attach.push([low, high]);
    } else {
      attach.push([horizontal[k], horizontal[k + 1]]);
    }
  }
  for (let row = 0; row < n; row++) add(node(c, row), node(0, row));  // closure
  return { edges, attach };
}

/** Walk each cycle; return the circles as ordered edge sequences. */
function traceCircles(nNodes, edges) {
  const incident = Array.from({ length: nNodes }, () => []);
  edges.forEach(([u, v], e) => { incident[u].push(e); incident[v].push(e); });
  const seen = new Set();
  const circles = [];
  for (let start = 0; start < edges.length; start++) {
    if (seen.has(start)) continue;
    const walk = [];
    let edge = start;
    let node = edges[start][0];
    while (!seen.has(edge)) {
      seen.add(edge);
      walk.push(edge);
      const [u, v] = edges[edge];
      node = node === u ? v : u;
      const next = incident[node].filter((e) => e !== edge);
      if (!next.length) break;
      edge = next[0];
    }
    circles.push(walk);
  }
  return circles;
}

/** Is the state graph two-colourable? That is exactly whether the
 *  surface is two-sided (Kalfagianni, Lemma 2.2). */
function bipartite(nCircles, bands) {
  const adjacency = Array.from({ length: nCircles }, () => []);
  for (const [a, , b] of bands) { adjacency[a].push(b); adjacency[b].push(a); }
  const colour = new Map();
  for (let root = 0; root < nCircles; root++) {
    if (colour.has(root)) continue;
    colour.set(root, 0);
    const stack = [root];
    while (stack.length) {
      const v = stack.pop();
      for (const w of adjacency[v]) {
        if (!colour.has(w)) { colour.set(w, 1 - colour.get(v)); stack.push(w); }
        else if (colour.get(w) === colour.get(v)) return false;
      }
    }
  }
  return true;
}

/** Eigenvectors of a small symmetric matrix, by cyclic Jacobi rotation.
 *
 *  Only needed for the Fiedler vector of a state graph, which has one
 *  node per state circle -- a handful. Returns eigenvalues ascending
 *  with their vectors as columns, which is what numpy's eigh gives. */
export function symmetricEigen(A, sweeps = 100) {
  const n = A.length;
  const a = A.map((row) => row.slice());
  const V = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    }
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1)
          / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: V.map((row) => order.map((i) => row[i])),
  };
}

/** Resolve every crossing, trace the state circles, read off the
 *  rotation system.
 *
 *  The rotation matters: the Euler characteristic depends only on the
 *  counts, but the number of boundary components depends on the cyclic
 *  order in which bands meet each circle, and that order is what makes
 *  the boundary come out as the original link. */
export function stateData(braid, state) {
  const b = typeof braid === 'string' ? parseBraid(braid) : braid;
  const c = b.crossings.length;
  const bits = Array.from(state, (s) => !!s);
  if (bits.length !== c) throw new Error(`state has ${bits.length} bits, need ${c}`);
  const n = b.strands;

  const { edges, attach } = resolvedDiagram(b, bits);
  const circles = traceCircles((c + 1) * n, edges);

  const circleOfEdge = new Map();
  circles.forEach((walk, i) => walk.forEach((e) => circleOfEdge.set(e, i)));
  const attaching = new Set();
  for (const pair of attach) for (const e of pair) attaching.add(e);
  const footOfEdge = new Map();
  const feet = circles.map(() => 0);
  circles.forEach((walk, i) => {
    for (const e of walk) {
      if (attaching.has(e)) { footOfEdge.set(e, feet[i]); feet[i]++; }
    }
  });

  const bands = attach.map(([ea, eb], q) => [
    circleOfEdge.get(ea), footOfEdge.get(ea),
    circleOfEdge.get(eb), footOfEdge.get(eb),
    b.crossings[q].sign,
  ]);

  const anchors = circles.map((walk) => {
    let rows = 0, levels = 0, count = 0;
    for (const e of walk) {
      for (const endpoint of edges[e]) {
        levels += Math.floor(endpoint / n);
        rows += endpoint % n;
        count++;
      }
    }
    return [rows / count, levels / count];
  });

  return {
    braid: b,
    state: bits,
    nCircles: circles.length,
    bands,
    feet,
    anchors,
    get eulerCharacteristic() { return this.nCircles - this.bands.length; },
    get nBoundaries() { return nComponents(this.braid); },
    get orientable() { return bipartite(this.nCircles, this.bands); },
    get eulerGenus() { return 2 - this.eulerCharacteristic - this.nBoundaries; },
    get genus() {
      if (!this.orientable) return null;
      const g = this.eulerGenus;
      return g % 2 === 0 ? g / 2 : null;
    },
    get crosscapNumber() { return this.orientable ? null : this.eulerGenus; },
    // A one-sided surface needs a twist on every band; a two-sided one
    // does not, and no twist is the nicer realisation -- a HALF
    // geometric turn per band rather than a full one, which is the
    // classical picture of a Seifert band.
    get twisted() { return !this.orientable; },
  };
}

/** Which state circles to traverse the other way round.
 *
 *  Tracing a cycle gives its band order only up to direction, and the
 *  directions have to agree with a single planar orientation or the
 *  boundary comes out wrong. Rather than reconstruct the embedding, fix
 *  them by the property that DEFINES a state surface: its boundary is
 *  the link. Reversing every circle at once is a global reflection, so
 *  circle 0 can be held fixed. */
export function rotationReversals(data) {
  const m = data.nCircles;
  const plain = data.bands.map(([a, fa, b, fb]) => [a, fa, b, fb]);
  if (m > 18) return new Array(m).fill(false);    // the search would be silly
  // The engine takes the FIRST combination that works, enumerating with
  // the last flag varying fastest (itertools.product order), so the bit
  // order here has to match or a different -- equally valid, but
  // different -- traversal is chosen.
  for (let bits = 0; bits < (1 << (m - 1)); bits++) {
    const flags = [false];
    for (let i = 0; i < m - 1; i++) flags.push(!!((bits >> (m - 2 - i)) & 1));
    if (boundaryCount(data.feet, plain, flags, data.twisted)
        === data.nBoundaries) {
      return flags;
    }
  }
  throw new Error(`no traversal gives ${data.nBoundaries} boundary component(s)`);
}

/** Order the disks along the stacking axis, by the state graph's
 *  Fiedler vector: circles joined by a band end up close together, so
 *  the bands stay short. For the Seifert state the graph is a path and
 *  this recovers the paper's row-by-row stack. */
export function spatialOrder(data) {
  const n = data.nCircles;
  if (n < 3) return Array.from({ length: n }, (_, i) => i);
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [a, , b] of data.bands) {
    if (a === b) continue;
    L[a][b] -= 1; L[b][a] -= 1; L[a][a] += 1; L[b][b] += 1;
  }
  const { values, vectors } = symmetricEigen(L);
  const fiedler = values.length > 1
    ? vectors.map((row) => row[1]) : new Array(n).fill(0);
  // AN EIGENVECTOR HAS NO INHERENT SIGN, and the sign decides which end
  // of the stack is which -- flipping it reflects the whole surface.
  // The engine takes whatever LAPACK returns, so its stack order is at
  // the mercy of a library's internal choice; here the sign is pinned
  // to the diagram instead, so the same braid always builds the same
  // way round. (When the second eigenvalue is REPEATED, as it is for a
  // symmetric state graph, the vector is not defined even up to sign;
  // nothing can pin that, and the parity test says so.)
  let corr = 0;
  for (let i = 0; i < n; i++) corr += fiedler[i] * data.anchors[i][1];
  if (corr === 0) {
    for (let i = 0; i < n; i++) {
      if (Math.abs(fiedler[i]) > 1e-12) { corr = fiedler[i]; break; }
    }
  }
  if (corr < 0) for (let i = 0; i < n; i++) fiedler[i] = -fiedler[i];
  const keys = Array.from({ length: n }, (_, i) =>
    [Number(fiedler[i].toFixed(9)), data.anchors[i][0], data.anchors[i][1]]);
  return Array.from({ length: n }, (_, i) => i).sort((i, j) => {
    for (let k = 0; k < 3; k++) {
      if (keys[i][k] !== keys[j][k]) return keys[i][k] - keys[j][k];
    }
    return 0;
  });
}

export const seifertState = (braid) =>
  new Array((typeof braid === 'string' ? parseBraid(braid) : braid)
    .crossings.length).fill(false);

export const turnbackState = (braid) =>
  new Array((typeof braid === 'string' ? parseBraid(braid) : braid)
    .crossings.length).fill(true);

/** Every one of the 2^c states. Exponential -- guard the count. */
export function* allStates(braid) {
  const b = typeof braid === 'string' ? parseBraid(braid) : braid;
  const c = b.crossings.length;
  for (let bits = 0; bits < (1 << c); bits++) {
    const s = [];
    for (let i = 0; i < c; i++) s.push(!!((bits >> i) & 1));
    yield stateData(b, s);
  }
}

/** The non-orientable state surface of smallest crosscap number -- an
 *  upper bound for the link's own crosscap number, since state surfaces
 *  are only some of its spanning surfaces. */
export function minimalCrosscapState(braid) {
  let best = null;
  for (const s of allStates(braid)) {
    if (s.orientable) continue;
    if (best === null || s.crosscapNumber < best.crosscapNumber) best = s;
  }
  if (best === null) throw new Error('no non-orientable state');
  return best;
}

// ------------------------------------------------------------- a mesh
//
// Deliberately NOT a half-edge structure: a half-edge's twin relation
// encodes a global orientation, so it cannot represent a one-sided
// surface -- and half the surfaces here are one-sided. An indexed face
// set plus an explicit orientation test costs nothing and stays honest.

export class Mesh {
  constructor(vertices = [], faces = [], faceGroups = []) {
    this.vertices = vertices;        // [[x,y,z], ...]
    this.faces = faces;              // [[i,j,k,l], ...] any arity
    this.faceGroups = faceGroups;    // "disk" / "band", carried through refinement
  }

  static empty() { return new Mesh([], [], []); }

  addVertices(points) {
    const start = this.vertices.length;
    for (const p of points) this.vertices.push([p[0], p[1], p[2]]);
    return Array.from({ length: points.length }, (_, i) => start + i);
  }

  addFace(indices, group = null) {
    this.faces.push(indices.slice());
    if (group !== null || this.faceGroups.length) {
      this.faceGroups.push(group || '');
    }
  }

  /** Undirected edge -> the faces using it, with the direction traversed. */
  edgeMap() {
    const out = new Map();
    this.faces.forEach((f, fi) => {
      const n = f.length;
      for (let i = 0; i < n; i++) {
        const a = f[i], b = f[(i + 1) % n];
        const k = a < b ? `${a},${b}` : `${b},${a}`;
        if (!out.has(k)) out.set(k, []);
        out.get(k).push([fi, a, b]);
      }
    });
    return out;
  }

  boundaryEdges() {
    const out = [];
    for (const [k, uses] of this.edgeMap()) {
      if (uses.length === 1) {
        const [a, b] = k.split(',').map(Number);
        out.push([a, b]);
      }
    }
    return out;
  }

  /** Boundary components as ordered vertex cycles -- the link itself. */
  boundaryLoops() {
    const adjacency = new Map();
    for (const [a, b] of this.boundaryEdges()) {
      if (!adjacency.has(a)) adjacency.set(a, []);
      if (!adjacency.has(b)) adjacency.set(b, []);
      adjacency.get(a).push(b);
      adjacency.get(b).push(a);
    }
    const loops = [];
    const visited = new Set();
    for (const start of adjacency.keys()) {
      if (visited.has(start)) continue;
      const loop = [start];
      let previous = null, current = start;
      visited.add(start);
      for (;;) {
        let next = (adjacency.get(current) || []).filter((v) => v !== previous);
        const unseen = next.filter((v) => !visited.has(v));
        next = unseen.length ? unseen : next;
        if (!next.length || next[0] === start) break;
        previous = current;
        current = next[0];
        visited.add(current);
        loop.push(current);
      }
      if (loop.length > 2) loops.push(loop);
    }
    return loops;
  }

  /** Propagate an orientation over the dual graph.
   *
   *  Returns the per-face flip sign and the number of conflict edges.
   *  On a one-sided surface a coherent winding does not exist, so the
   *  signs are a best-effort partition whose only inconsistency is
   *  along the true seam -- which is exactly the Mobius band's seam. */
  orientationSigns() {
    const dual = new Map();
    const push = (a, b, same) => {
      if (!dual.has(a)) dual.set(a, []);
      dual.get(a).push([b, same]);
    };
    for (const uses of this.edgeMap().values()) {
      if (uses.length !== 2) continue;
      const [[f1, a1, b1], [f2, a2, b2]] = uses;
      const same = a1 === a2 && b1 === b2;
      push(f1, f2, same);
      push(f2, f1, same);
    }
    const sign = new Map();
    let conflicts = 0;
    for (let root = 0; root < this.faces.length; root++) {
      if (sign.has(root)) continue;
      sign.set(root, 0);
      const stack = [root];
      while (stack.length) {
        const a = stack.pop();
        for (const [b, same] of dual.get(a) || []) {
          const want = sign.get(a) ^ (same ? 1 : 0);
          if (!sign.has(b)) { sign.set(b, want); stack.push(b); }
          else if (sign.get(b) !== want) conflicts++;
        }
      }
    }
    return { sign, conflicts };
  }

  isOrientable() { return this.orientationSigns().conflicts === 0; }

  /** Rewind faces to a coherent winding (best effort if one-sided).
   *
   *  The builder emits disks and bands independently, so their faces
   *  are wound at random relative to each other; with smooth shading
   *  that shows as dark creases along every disk/band seam, where the
   *  averaged vertex normal flips. */
  oriented() {
    const { sign } = this.orientationSigns();
    const faces = this.faces.map((f, i) =>
      (sign.get(i) ? f.slice().reverse() : f.slice()));
    return new Mesh(this.vertices.map((v) => v.slice()), faces,
                    this.faceGroups.slice());
  }

  /** Fan-triangulate every face. */
  triangulated() {
    const tris = [], groups = [];
    this.faces.forEach((f, fi) => {
      for (let i = 1; i < f.length - 1; i++) {
        tris.push([f[0], f[i], f[i + 1]]);
        if (this.faceGroups.length) groups.push(this.faceGroups[fi]);
      }
    });
    return new Mesh(this.vertices.map((v) => v.slice()), tris, groups);
  }

  area() {
    const tri = this.triangulated();
    let total = 0;
    for (const [i, j, k] of tri.faces) {
      const a = tri.vertices[i], b = tri.vertices[j], c = tri.vertices[k];
      const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
      const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
      total += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz,
                                ux * vy - uy * vx);
    }
    return total;
  }

  /** V, E, F, chi, boundary count, orientability -- the check that the
   *  geometry still realises the combinatorics it was built from. */
  info() {
    const edges = this.edgeMap();
    const used = new Set();
    for (const f of this.faces) for (const i of f) used.add(i);
    const chi = used.size - edges.size + this.faces.length;
    const boundaries = this.boundaryLoops().length;
    const orientable = this.isOrientable();
    const eulerGenus = 2 - chi - boundaries;
    return {
      nVertices: used.size,
      nEdges: edges.size,
      nFaces: this.faces.length,
      eulerCharacteristic: chi,
      nBoundaries: boundaries,
      orientable,
      eulerGenus,
      genus: orientable && eulerGenus % 2 === 0 ? eulerGenus / 2 : null,
    };
  }
}

// ------------------------------------------------ curves and frames

export function unit(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

/** Cubic Bezier positions and unit tangents. */
export function bezier(p0, p1, p2, p3, samples) {
  const points = [], tangents = [];
  for (let i = 0; i < samples; i++) {
    const t = samples > 1 ? i / (samples - 1) : 0;
    const s = 1 - t;
    points.push([0, 1, 2].map((k) =>
      s * s * s * p0[k] + 3 * s * s * t * p1[k]
      + 3 * s * t * t * p2[k] + t * t * t * p3[k]));
    tangents.push(unit([0, 1, 2].map((k) =>
      3 * s * s * (p1[k] - p0[k]) + 6 * s * t * (p2[k] - p1[k])
      + 3 * t * t * (p3[k] - p2[k]))));
  }
  return { points, tangents };
}

/** Propagate a normal along a curve by DOUBLE REFLECTION.
 *
 *  The paper frames its bands with the Frenet frame, which is undefined
 *  where the curvature vanishes and flips through an inflection -- a
 *  real hazard for a cubic Bezier with nearly collinear control points.
 *  Each step here is two reflections, hence a rotation, so
 *  orthonormality is exact; reflecting first across the chord makes it
 *  fourth-order accurate. Wang, Juttler, Zheng & Liu, ACM TOG 27(1). */
export function transportNormals(points, tangents, firstNormal) {
  const normals = new Array(points.length);
  normals[0] = unit(sub(firstNormal, mul(tangents[0], dot(firstNormal, tangents[0]))));
  for (let i = 0; i < points.length - 1; i++) {
    const v1 = sub(points[i + 1], points[i]);
    const c1 = dot(v1, v1);
    if (c1 < 1e-18) { normals[i + 1] = normals[i]; continue; }
    const rn = sub(normals[i], mul(v1, (2 / c1) * dot(v1, normals[i])));
    const rt = sub(tangents[i], mul(v1, (2 / c1) * dot(v1, tangents[i])));
    const v2 = sub(tangents[i + 1], rt);
    const c2 = dot(v2, v2);
    normals[i + 1] = c2 < 1e-18 ? rn : sub(rn, mul(v2, (2 / c2) * dot(v2, rn)));
  }
  return normals;
}

/** Total rotation to apply along a band.
 *
 *  The two admissible closures differ by pi, so the PARITY of
 *  `halfTurns` selects one; whole turns are then added to reach the
 *  requested twist. */
export function twistToClose(transported, target, tangent, halfTurns) {
  const reference = unit(sub(transported, mul(tangent, dot(transported, tangent))));
  const binormal = cross(tangent, reference);
  const goal = unit(sub(target, mul(tangent, dot(target, tangent))));
  let residual = Math.atan2(dot(binormal, goal), dot(reference, goal));
  if (halfTurns % 2) residual += residual <= 0 ? Math.PI : -Math.PI;
  const wanted = halfTurns * Math.PI;
  return residual + Math.round((wanted - residual) / TAU) * TAU;
}

// --------------------------------------------------- the disk/band mesh

/** Geometry knobs, in units of the disk radius.
 *
 *  The paper publishes no numeric defaults; these ratios come from a
 *  reference surface with disks of diameter 60 spaced 83 apart and
 *  bands 20 wide, so the stack is much taller and the bands much
 *  narrower than the obvious guess. */
export const DEFAULT_PARAMS = {
  diskRadius: 1.0,
  diskSpacing: 2.77,          // 83/30
  bandWidth: 0.85,
  arcSpan: 0.75,
  samplesPerSector: 12,       // must be even, so the O-grid core closes
  radialRings: 2,
  coreFraction: 0.45,
  bandSamples: 18,
  bulge: 0.85,
  bandLaunch: 1.0,
};

export function surfaceParams(overrides = {}) {
  const p = { ...DEFAULT_PARAMS, ...overrides };
  if (p.samplesPerSector % 2) throw new Error('samplesPerSector must be even');
  return p;
}

/** Rim samples spanned by one band foot, from the requested width. */
function arcSamples(p, nTheta) {
  const chord = Math.min(0.98, p.bandWidth / (2 * p.diskRadius));
  const wanted = 2 * Math.asin(chord) * nTheta / TAU;
  const ceiling = Math.max(2, Math.trunc(p.arcSpan * p.samplesPerSector));
  // Python's round() is banker's rounding; JS Math.round is not, and
  // the two differ on exact halves, which this expression can hit.
  const r = Math.abs(wanted % 1 - 0.5) < 1e-12
    ? 2 * Math.round(wanted / 2) : Math.round(wanted);
  return Math.trunc(Math.min(ceiling, Math.max(2, r)));
}

/** Transfinite (Coons) interpolation of a square grid from its boundary. */
function coons(boundary, m) {
  const grid = [];
  const bottom = boundary.slice(0, m + 1);
  const right = boundary.slice(m, 2 * m + 1);
  const top = boundary.slice(2 * m, 3 * m + 1).slice().reverse();
  const left = boundary.slice(3 * m).concat(boundary.slice(0, 1)).reverse();
  for (let i = 0; i <= m; i++) {
    const u = i / m;
    grid.push([]);
    for (let j = 0; j <= m; j++) {
      const v = j / m;
      grid[i].push([0, 1, 2].map((k) =>
        (1 - v) * bottom[i][k] + v * top[i][k]
        + (1 - u) * left[j][k] + u * right[j][k]
        - ((1 - u) * (1 - v) * bottom[0][k] + u * (1 - v) * bottom[m][k]
           + (1 - u) * v * top[0][k] + u * v * top[m][k])));
    }
  }
  return grid;
}

/** Add one disk as an O-grid; return its rim vertices in angle order.
 *
 *  A polar fan would put a vertex of valence nTheta at the centre.
 *  Both Catmull-Clark and the relaxation behave badly around a vertex
 *  of valence 70, and it shows as a radial starburst crease that no
 *  amount of smoothing removes. An O-grid -- a square patch in the
 *  middle, annular rings around it -- keeps every vertex at valence
 *  four except four corners at valence three. */
function disk(mesh, centre, nTheta, p) {
  if (nTheta % 4) throw new Error('nTheta must be a multiple of 4');
  const origin = typeof centre === 'number' ? [0, 0, centre] : centre;
  const m = nTheta / 4;
  const angles = Array.from({ length: nTheta }, (_, i) => TAU * i / nTheta);
  const inner = p.diskRadius * p.coreFraction;

  const ring = angles.map((a) => [origin[0] + inner * Math.cos(a),
                                  origin[1] + inner * Math.sin(a),
                                  origin[2]]);
  const grid = coons(ring, m);

  const ids = Array.from({ length: m + 1 }, () => new Array(m + 1).fill(-1));
  const walk = [];
  for (let i = 0; i < m; i++) walk.push([i, 0]);
  for (let j = 0; j < m; j++) walk.push([m, j]);
  for (let i = m; i > 0; i--) walk.push([i, m]);
  for (let j = m; j > 0; j--) walk.push([0, j]);
  const coreRing = mesh.addVertices(walk.map(([i, j]) => grid[i][j]));
  walk.forEach(([i, j], k) => { ids[i][j] = coreRing[k]; });
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < m; j++) ids[i][j] = mesh.addVertices([grid[i][j]])[0];
  }
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      mesh.addFace([ids[i][j], ids[i + 1][j], ids[i + 1][j + 1], ids[i][j + 1]],
                   'disk');
    }
  }

  const rings = [coreRing];
  for (let k = 1; k <= p.radialRings; k++) {
    const r = inner + (p.diskRadius - inner) * k / p.radialRings;
    rings.push(mesh.addVertices(angles.map((a) =>
      [origin[0] + r * Math.cos(a), origin[1] + r * Math.sin(a), origin[2]])));
  }
  for (let k = 0; k < rings.length - 1; k++) {
    const a = rings[k], b = rings[k + 1];
    for (let i = 0; i < nTheta; i++) {
      const j = (i + 1) % nTheta;
      mesh.addFace([a[i], b[i], b[j], a[j]], 'disk');
    }
  }
  return rings[rings.length - 1];
}

/** The width+1 rim vertices centred on an angle. Every disk is meshed
 *  at the same angular resolution, so an arc of a given sample count
 *  subtends the same chord wherever it lands -- which is what lets the
 *  two ends of a band weld one-to-one. */
function attachmentArc(rim, angle, width) {
  const n = rim.length;
  const start = Math.round(angle * n / TAU) - Math.floor(width / 2);
  return Array.from({ length: width + 1 },
                    (_, i) => rim[(((start + i) % n) + n) % n]);
}

/** Sweep a twisted ribbon between two rim arcs, reusing their vertices.
 *
 *  ON THE TWIST PARITY. Both disks face +z and the band leaves one rim
 *  radially outward and arrives at the other radially inward, so the
 *  band makes a U-turn -- which already reverses the frame once. An
 *  ODD number of geometric half-turns therefore delivers the far end
 *  face-up and an EVEN number face-down, the opposite of what the flat,
 *  coplanar picture would suggest. */
function band(mesh, arcA, arcB, angleA, angleB, halfTurns, span, p) {
  const outA = [Math.cos(angleA), Math.sin(angleA), 0];
  const outB = [Math.cos(angleB), Math.sin(angleB), 0];
  const tanA = [-Math.sin(angleA), Math.cos(angleA), 0];
  const tanB = [-Math.sin(angleB), Math.cos(angleB), 0];

  const start = arcA.map((i) => mesh.vertices[i]);
  const end = arcB.map((i) => mesh.vertices[i]);
  const mean = (pts) => {
    const s = pts.reduce((a, q) => add(a, q), [0, 0, 0]);
    return mul(s, 1 / pts.length);
  };
  const p0 = mean(start), p3 = mean(end);
  let reach = p.bulge * p.diskRadius + 0.42 * span;
  if (Math.hypot(...sub(p3, p0)) < 1e-9) reach += p.diskRadius;  // onto its own disk
  const dz = p3[2] - p0[2];
  const axis = [0, 0, dz > 0 ? 1 : (dz < 0 ? -1 : 1)];
  const lam = p.bandLaunch;
  const launchA = unit(add(mul(outA, lam), mul(axis, 1 - lam)));
  const launchB = unit(sub(mul(outB, lam), mul(axis, 1 - lam)));
  const { points, tangents } = bezier(p0, add(p0, mul(launchA, reach)),
                                      add(p3, mul(launchB, reach)), p3,
                                      p.bandSamples);
  const normals = transportNormals(points, tangents, tanA);

  const totalTwist = twistToClose(normals[normals.length - 1], tanB,
                                  tangents[tangents.length - 1], halfTurns);
  const far = (halfTurns % 2) ? arcB.slice().reverse() : arcB;

  const widthAxis = tanA;
  const normalAxis = cross(tangents[0], widthAxis);
  const profile = start.map((q) => {
    const local = sub(q, p0);
    return [dot(local, widthAxis), dot(local, normalAxis)];
  });

  const rows = [arcA.slice()];
  for (let step = 1; step < p.bandSamples; step++) {
    if (step === p.bandSamples - 1) { rows.push(far.slice()); break; }
    const t = step / (p.bandSamples - 1);
    const tangent = tangents[step];
    const u = unit(sub(normals[step], mul(tangent, dot(normals[step], tangent))));
    const v = cross(tangent, u);
    const axisU = add(mul(u, Math.cos(totalTwist * t)),
                      mul(v, Math.sin(totalTwist * t)));
    const axisV = cross(tangent, axisU);
    rows.push(mesh.addVertices(profile.map(([w, h]) =>
      add(points[step], add(mul(axisU, w), mul(axisV, h))))));
  }

  for (let k = 0; k < rows.length - 1; k++) {
    const a = rows[k], b = rows[k + 1];
    for (let i = 0; i < a.length - 1; i++) {
      mesh.addFace([a[i], a[i + 1], b[i + 1], b[i]], 'band');
    }
  }
}

/** Disks stacked along z, joined by twisted bands. */
export function spanningSurface(feet, bands, params) {
  const p = surfaceParams(params);
  const nTheta = Math.max(4, Math.max(...feet, 1) * p.samplesPerSector);
  const width = arcSamples(p, nTheta);

  const mesh = Mesh.empty();
  const rims = [], angles = [];
  feet.forEach((count, d) => {
    rims.push(disk(mesh, d * p.diskSpacing, nTheta, p));
    angles.push(Array.from({ length: count },
                           (_, j) => TAU * (j + 0.5) / Math.max(count, 1)));
  });

  for (const [diskA, footA, diskB, footB, halfTurns] of bands) {
    band(mesh,
         attachmentArc(rims[diskA], angles[diskA][footA], width),
         attachmentArc(rims[diskB], angles[diskB][footB], width),
         angles[diskA][footA], angles[diskB][footB], halfTurns,
         Math.abs(diskA - diskB) * p.diskSpacing, p);
  }
  return mesh;
}

/** Geometric half-twists per band. The band's U-turn already reverses
 *  the frame once, so an ODD count delivers the far end face-up and an
 *  even count face-down. A two-sided surface can use the odd one, which
 *  is both the smaller deformation and the classical picture of a
 *  Seifert band; a one-sided one needs the even. */
export const BAND_HALF_TURNS = { false: 1, true: 2 };

/** Spanning surface for one Kauffman state -- orientable or not.
 *
 *  `order` overrides the Fiedler stacking order, which is only defined
 *  up to the sign of an eigenvector (and not at all when the state
 *  graph is symmetric). The parity test passes the engine's own order
 *  in, so the geometry can be compared exactly without the comparison
 *  depending on which eigensolver ran. */
export function stateSurface(braid, state = null, params = null, order = null) {
  const b = typeof braid === 'string' ? parseBraid(braid) : braid;
  const data = state === null ? stateData(b, seifertState(b))
    : (state.braid ? state : stateData(b, state));

  const spatial = order || spatialOrder(data);
  const height = new Map();
  spatial.forEach((circle, level) => height.set(circle, level));
  const flip = rotationReversals(data);
  const feet = new Array(data.nCircles).fill(0);
  for (const [circle, level] of height) feet[level] = data.feet[circle];

  const place = (circle, foot) =>
    flip[circle] ? data.feet[circle] - 1 - foot : foot;

  // The geometric twist parity that realises the combinatorics depends
  // on how far the band turns on its way over, which depends on the
  // launch direction. Rather than reason about it, build both parities
  // and keep the one whose mesh matches the prediction.
  const wanted = [data.eulerCharacteristic, data.nBoundaries, data.orientable];
  let fallback = null;
  for (const extra of [0, 1]) {
    const turns = BAND_HALF_TURNS[String(data.twisted)] + extra;
    const mesh = spanningSurface(feet, data.bands.map(
      ([a, fa, bb, fb, sign]) =>
        [height.get(a), place(a, fa), height.get(bb), place(bb, fb),
         turns * sign]), params);
    const info = mesh.info();
    if (info.eulerCharacteristic === wanted[0]
        && info.nBoundaries === wanted[1]
        && info.orientable === wanted[2]) return mesh;
    fallback = fallback || mesh;
  }
  throw new Error(`neither twist parity realises `
    + `chi=${wanted[0]}, b=${wanted[1]}, orientable=${wanted[2]}`);
}

/** Seifert surface of a braid closure, in the stacked disk/band style. */
export function seifertSurface(braid, params = null, order = null) {
  const b = typeof braid === 'string' ? parseBraid(braid) : braid;
  return stateSurface(b, stateData(b, seifertState(b)), params, order);
}

// ------------------------------------------------ Catmull-Clark refinement
//
// Boundary rules follow the usual convention: a boundary edge point is
// the edge midpoint, and a boundary vertex uses the cubic B-spline
// stencil (1, 6, 1)/8 along the boundary curve, so the rim converges to
// a smooth curve independent of the interior -- which matters here,
// because that rim is the knot.

export function catmullClark(mesh, levels = 1) {
  let m = mesh;
  for (let i = 0; i < levels; i++) m = subdivideOnce(m);
  return m;
}

function subdivideOnce(mesh) {
  const V = mesh.vertices;
  const edgeFaces = new Map();
  const key = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
  mesh.faces.forEach((f, fi) => {
    for (let i = 0; i < f.length; i++) {
      const k = key(f[i], f[(i + 1) % f.length]);
      if (!edgeFaces.has(k)) edgeFaces.set(k, []);
      edgeFaces.get(k).push(fi);
    }
  });
  const vertexFaces = new Map();
  const vertexEdges = new Map();
  mesh.faces.forEach((f, fi) => {
    for (const v of f) {
      if (!vertexFaces.has(v)) vertexFaces.set(v, []);
      vertexFaces.get(v).push(fi);
    }
  });
  for (const k of edgeFaces.keys()) {
    const [a, b] = k.split(',').map(Number);
    if (!vertexEdges.has(a)) vertexEdges.set(a, []);
    if (!vertexEdges.has(b)) vertexEdges.set(b, []);
    vertexEdges.get(a).push([a, b]);
    vertexEdges.get(b).push([a, b]);
  }

  const facePoints = mesh.faces.map((f) => {
    const s = f.reduce((acc, i) => add(acc, V[i]), [0, 0, 0]);
    return mul(s, 1 / f.length);
  });

  const edgePoints = new Map();
  const boundaryNeighbours = new Map();
  for (const [k, faces] of edgeFaces) {
    const [a, b] = k.split(',').map(Number);
    const midpoint = mul(add(V[a], V[b]), 0.5);
    if (faces.length === 1) {
      edgePoints.set(k, midpoint);
      if (!boundaryNeighbours.has(a)) boundaryNeighbours.set(a, []);
      if (!boundaryNeighbours.has(b)) boundaryNeighbours.set(b, []);
      boundaryNeighbours.get(a).push(b);
      boundaryNeighbours.get(b).push(a);
    } else {
      const fp = faces.reduce((acc, fi) => add(acc, facePoints[fi]), [0, 0, 0]);
      edgePoints.set(k, mul(add(midpoint, mul(fp, 1 / faces.length)), 0.5));
    }
  }

  const newVertex = new Map();
  for (let v = 0; v < V.length; v++) {
    if (boundaryNeighbours.has(v)) {
      const nb = boundaryNeighbours.get(v);
      newVertex.set(v, nb.length === 2
        ? mul(add(add(V[nb[0]], mul(V[v], 6)), V[nb[1]]), 1 / 8)
        : V[v].slice());
      continue;
    }
    const faces = vertexFaces.get(v) || [];
    const edges = vertexEdges.get(v) || [];
    const n = faces.length;
    if (n === 0) { newVertex.set(v, V[v].slice()); continue; }
    const F = mul(faces.reduce((acc, fi) => add(acc, facePoints[fi]), [0, 0, 0]),
                  1 / n);
    const R = mul(edges.reduce((acc, [a, b]) =>
      add(acc, mul(add(V[a], V[b]), 0.5)), [0, 0, 0]), 1 / edges.length);
    newVertex.set(v, mul(add(add(F, mul(R, 2)), mul(V[v], n - 3)), 1 / n));
  }

  const out = Mesh.empty();
  const vertexId = new Map();
  for (const [v, p] of newVertex) vertexId.set(v, out.addVertices([p])[0]);
  const faceId = facePoints.map((p) => out.addVertices([p])[0]);
  const edgeId = new Map();
  for (const [k, p] of edgePoints) edgeId.set(k, out.addVertices([p])[0]);

  mesh.faces.forEach((f, fi) => {
    const n = f.length;
    const group = mesh.faceGroups.length ? mesh.faceGroups[fi] : null;
    for (let i = 0; i < n; i++) {
      const prevV = f[(i - 1 + n) % n], v = f[i], nextV = f[(i + 1) % n];
      out.addFace([vertexId.get(v), edgeId.get(key(v, nextV)), faceId[fi],
                   edgeId.get(key(prevV, v))], group);
    }
  });
  return out;
}

// ------------------------------------- handing the surface onwards
//
// The computation layer stops here. What leaves it is plain arrays --
// no three.js types, no DOM -- which is what lets the whole of the
// above run in node against the Python engine.

/** Triangulate into flat buffers for a renderer or an exporter. */
export function toTriangles(mesh) {
  const tri = mesh.triangulated();
  const positions = new Float32Array(tri.vertices.length * 3);
  tri.vertices.forEach((v, i) => {
    positions[i * 3] = v[0];
    positions[i * 3 + 1] = v[1];
    positions[i * 3 + 2] = v[2];
  });
  const indices = new Uint32Array(tri.faces.length * 3);
  tri.faces.forEach((f, i) => {
    indices[i * 3] = f[0];
    indices[i * 3 + 1] = f[1];
    indices[i * 3 + 2] = f[2];
  });
  return { positions, indices };
}

/** Centre on the origin and scale the longest side to `size`.
 *
 *  The add-on's own convention: a generator's output is centred and
 *  fits a 2-unit cube, so every surface arrives at a comparable size
 *  whatever braid produced it. */
export function fit(mesh, size = 2) {
  const V = mesh.vertices;
  if (!V.length) return mesh;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const v of V) {
    for (let k = 0; k < 3; k++) {
      if (v[k] < lo[k]) lo[k] = v[k];
      if (v[k] > hi[k]) hi[k] = v[k];
    }
  }
  const centre = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  const s = size / span;
  return new Mesh(V.map((v) => [0, 1, 2].map((k) => (v[k] - centre[k]) * s)),
                  mesh.faces.map((f) => f.slice()), mesh.faceGroups.slice());
}

/** The boundary as closed polylines -- this IS the knot, which is the
 *  whole point: the surface was built to span it. */
export function boundaryPolylines(mesh) {
  return mesh.boundaryLoops().map((loop) => ({
    points: loop.map((i) => mesh.vertices[i].slice()),
    closed: true,
  }));
}

/** A tube swept along a polyline, with a rotation-minimising frame, as
 *  plain buffers. Used to draw the link itself over the surface. */
export function tubeAlong(loops, radius, sides = 12) {
  const pos = [], nor = [], idx = [];
  for (const { points, closed } of loops) {
    const n = points.length;
    if (n < 3) continue;
    const base = pos.length / 3;
    const T = points.map((_, i) => {
      const a = points[(i - 1 + n) % n], b = points[(i + 1) % n];
      return unit([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    });
    let N = (() => {
      const t = T[0];
      const up = Math.abs(t[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
      return unit(cross(up, t));
    })();
    for (let i = 0; i < n; i++) {
      const t = T[i];
      N = unit(sub(N, mul(t, dot(N, t))));
      const B = cross(t, N);
      for (let s = 0; s < sides; s++) {
        const a = TAU * s / sides;
        const u = add(mul(N, Math.cos(a)), mul(B, Math.sin(a)));
        pos.push(points[i][0] + radius * u[0],
                 points[i][1] + radius * u[1],
                 points[i][2] + radius * u[2]);
        nor.push(u[0], u[1], u[2]);
      }
    }
    for (let i = 0; i < n; i++) {
      const i0 = base + i * sides;
      const i1 = base + ((i + 1) % n) * sides;
      for (let s = 0; s < sides; s++) {
        const s1 = (s + 1) % sides;
        idx.push(i0 + s, i1 + s, i1 + s1, i0 + s, i1 + s1, i0 + s1);
      }
    }
  }
  return {
    positions: Float32Array.from(pos),
    normals: Float32Array.from(nor),
    indices: Uint32Array.from(idx),
  };
}

/** Everything the page needs, from a braid word.
 *
 *  This is the module's "operator": parameters in, geometry and a
 *  topological summary out, with no renderer and no DOM anywhere in
 *  sight. */
export function buildSurface({
  word = 'AAA', surface = 'SEIFERT', levels = 1, params = null,
  fitSize = 2,
} = {}) {
  const braid = parseBraid(word);
  let data;
  if (surface === 'TURNBACK') data = stateData(braid, turnbackState(braid));
  else if (surface === 'CROSSCAP') data = minimalCrosscapState(braid);
  else data = stateData(braid, seifertState(braid));

  let mesh = stateSurface(braid, data, params);
  if (levels > 0) mesh = catmullClark(mesh, levels);
  mesh = fit(mesh.oriented(), fitSize);
  const info = mesh.info();
  let extent = 0;
  for (const v of mesh.vertices) {
    extent = Math.max(extent, Math.hypot(v[0], v[1], v[2]));
  }
  return {
    mesh,
    data,
    info,
    extent,
    surface: toTriangles(mesh),
    loops: boundaryPolylines(mesh),
    summary: {
      word: braid.word,
      strands: braid.strands,
      crossings: braid.crossings.length,
      disks: data.nCircles,
      bands: data.bands.length,
      components: data.nBoundaries,
      chi: info.eulerCharacteristic,
      orientable: info.orientable,
      genus: info.genus,
      crosscap: info.orientable ? null : info.eulerGenus,
      // the geometry has to realise the combinatorics it was built
      // from; if it ever does not, the mesh is wrong, not the arithmetic
      agrees: info.eulerCharacteristic === data.eulerCharacteristic
        && info.nBoundaries === data.nBoundaries
        && info.orientable === data.orientable,
    },
  };
}
