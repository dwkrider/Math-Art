// What makes two polyhedra alike.
//
// The layout itself lives in cluster.js and is shared with the surfaces:
// the kNN graph, the spring embedder, the collision term and the cooling
// schedule were measured once and are the same machinery here. This file
// is the only part that knows it is looking at solids.
//
// THE SIGNALS, and why these:
//
//   symmetry     the strongest thing a polyhedron has, and populated on
//                every one of the 471. Two solids sharing Ih really are
//                relatives whatever else differs, which is why the great
//                stellations sit with the icosahedra rather than with
//                the other stars
//   point group  the same signal one step coarser, so I and Ih land
//                near each other instead of being unrelated tokens.
//                Without it the chiral forms drift away from the full
//                groups they belong to
//   convexity    the sharpest single division in the catalogue -- 208
//                convex against 263 not -- and the one a reader sees
//                first
//   family tags  Platonic, Archimedean, Catalan, Johnson, compound,
//                zonohedron and the rest: 2.1 per solid on average, and
//                collectively what a reader would call the kind
//   size         face count, bucketed by powers of two. A tetrahedron
//                and a 720-face geodesic are not neighbours even when
//                everything else agrees
//   density      only for the stars that have one (252 of 471), so it is
//                weighted lightly: leaning on it would sort by "is a
//                star" a second time rather than by anything new
//
// Symmetry ORDER is deliberately absent as its own term. It is a
// function of the group -- Ih is always 120 -- so it would double the
// weight already given to the group while adding nothing, and it puts
// unrelated groups of equal order together.

// MEASURED, and the balance is a real choice rather than a free lunch.
// Kind and symmetry cross-cut each other -- the Platonic solids span
// Td, Oh and Ih; the Johnson solids span most of the C and D groups --
// so a two-dimensional layout has to pick which one it groups by, and
// leaning on one costs the other. Separation over the 471, by kind and
// by Schoenflies symbol:
//
//   schoenflies 2.6, tag 0.9   kind 1.79x   symmetry 2.86x
//   schoenflies 1.8, tag 1.8   kind 1.81x   symmetry 1.65x
//   schoenflies 1.4, tag 3.0   kind 4.19x   symmetry 1.18x   <- this
//
// Kind wins because it is what the reader is looking at: the tiles are
// pictures, and Johnson solids look like each other. Symmetry still
// orders things WITHIN a cluster, which is the right place for it.
//
// The held-out check agrees. Nothing in this vector knows which solids
// are duals of one another, and dual pairs land at 0.22 of the mean
// separation -- the tightest of any setting tried.
const W = {
  schoenflies: 1.4,
  pointGroup: 0.9,
  convex: 1.2,
  tag: 3.0,
  faces: 0.8,
  density: 0.5,
};

/**
 * The point-group family behind a Schoenflies symbol.
 *
 * Ih, I and Th all begin with a letter that names the family; the
 * subscript says which member. Grouping on the letter is what puts a
 * chiral solid beside the full-symmetry one it came from.
 */
export function pointGroup(sym) {
  if (!sym) return 'none';
  const c = sym[0];
  if (c === 'I') return 'icosahedral';
  if (c === 'O') return 'octahedral';
  if (c === 'T') return 'tetrahedral';
  if (c === 'D') return 'dihedral';
  if (c === 'S') return 'rotoreflection';
  if (c === 'C') return 'cyclic';
  return 'other';
}

/** Face count in powers of two, so size is a gradient and not 40 tokens. */
function sizeBucket(n) {
  if (!n) return 'unknown';
  return String(Math.round(Math.log2(n)));
}

/** Sparse unit-length feature vector for one polyhedron index entry. */
export function polyhedronVector(e) {
  const v = new Map();
  const put = (k, w) => v.set(k, (v.get(k) || 0) + w);

  const sym = e.symmetry || {};
  put('sym:' + (sym.schoenflies || 'none'), W.schoenflies);
  put('pg:' + pointGroup(sym.schoenflies), W.pointGroup);
  put('conv:' + (e.convex ? 'yes' : 'no'), W.convex);
  put('f:' + sizeBucket(e.counts && e.counts.faces), W.faces);
  for (const t of e.families || []) put('tag:' + t, W.tag);

  // Density is the number of times a star wraps its centre. Absent on
  // the convex solids, where it is 1 and says nothing.
  if (e.density != null) {
    put('dens:' + (e.density > 6 ? 'high' : e.density), W.density);
  }

  // Unit length, so cosine is a plain dot product and a solid carrying
  // more tags is not automatically "bigger" than one carrying fewer.
  let n = 0;
  for (const x of v.values()) n += x * x;
  n = Math.sqrt(n) || 1;
  for (const [k, x] of v) v.set(k, x / n);
  return v;
}

/**
 * The kind a reader would name, for labelling and for measurement.
 *
 * First match wins, most specific first: a solid tagged both `uniform`
 * and `archimedean` is an Archimedean solid to anyone looking at it.
 */
export const KINDS = [
  ['platonic', 'Platonic'],
  ['archimedean', 'Archimedean'],
  ['catalan', 'Catalan'],
  ['johnson', 'Johnson'],
  ['kepler-poinsot', 'Kepler-Poinsot'],
  ['compound', 'Compound'],
  ['uniform-dual', 'Uniform dual'],
  ['uniform', 'Uniform'],
  ['zonohedron', 'Zonohedron'],
  ['deltahedron', 'Deltahedron'],
  ['geodesic', 'Geodesic'],
  ['toroid', 'Toroid'],
  ['antiprism', 'Antiprism'],
  ['prism', 'Prism'],
  ['notable', 'Notable'],
];

export function kindOf(e) {
  const fams = e.families || [];
  for (const [tag, label] of KINDS) if (fams.includes(tag)) return label;
  return e.convex ? 'Convex' : 'Other';
}
