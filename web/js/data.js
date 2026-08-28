// Access to the polyhedron database in data/polyhedra/.
//
// The site reads the repository's real database rather than a curated
// export: index.json is 256 KB and carries enough to drive the whole
// catalogue (slug, name, families, V/E/F, symmetry), and the per-solid
// records average 22 KB and are fetched only when something actually
// needs one. So the catalogue costs a single request and a solid costs
// one more, which is why no build step or data pipeline exists here.
//
// Paths are resolved relative to THIS module's URL, never from the site
// root: GitHub Pages serves the project from /Math-Art/, so an absolute
// "/data/..." would 404 in production while working fine locally.

const DB = new URL('../data/polyhedra/', import.meta.url);

let indexPromise = null;
const records = new Map();

export function loadIndex() {
  if (!indexPromise) {
    indexPromise = fetch(new URL('index.json', DB))
      .then((r) => {
        if (!r.ok) throw new Error(`index.json: HTTP ${r.status}`);
        return r.json();
      });
  }
  return indexPromise;
}

export async function loadRecord(slug) {
  if (records.has(slug)) return records.get(slug);
  const index = await loadIndex();
  const entry = index.entries.find((e) => e.slug === slug);
  if (!entry) throw new Error(`unknown solid: ${slug}`);
  const p = fetch(new URL(entry.path, DB)).then((r) => {
    if (!r.ok) throw new Error(`${entry.path}: HTTP ${r.status}`);
    return r.json();
  });
  records.set(slug, p);
  return p;
}

export function thumbUrl(slug) {
  return new URL(`../thumbs/${slug}.png`, import.meta.url).href;
}

// -- facets -----------------------------------------------------------
//
// Every facet below is computed from index.json alone, which is what
// keeps browsing 448 solids to one request. The family tags are the
// database's own; they overlap deliberately (a cube is uniform AND
// platonic AND zonohedron AND space-filling), so families are treated
// as a set-membership filter rather than a partition.

/**
 * The family taxonomy, in the vocabulary the Blender generators use.
 *
 * The database tags a solid with every family it belongs to -- a cube is
 * uniform AND platonic AND zonohedron AND space-filling AND prism -- which
 * is correct but makes a poor filter: the counts overlap, they do not sum
 * to anything, and "which family is this?" has no single answer.
 *
 * The add-on's Family enums answer it. `regular_solids_generator.FAMILIES`
 * is a partition -- one solid, one family -- and the other polyhedron
 * operators extend the same vocabulary. So the list below is a PRECEDENCE:
 * the first rule that matches wins, and the order is from most specific
 * (the five Platonic solids) to most general (anything else uniform).
 * Every one of the 448 lands in exactly one family, so the counts sum.
 *
 * Precedence matters where the tags genuinely overlap: a pentagrammic
 * prism is both `prism-family` and `uniform`, and belongs under prisms;
 * the biscribed forms are also uniform, and belong under biscribed.
 */
const FAMILY_RULES = [
  ['Platonic', (e) => e.families.includes('platonic')],
  ['Kepler–Poinsot', (e) => e.families.includes('kepler-poinsot')],
  ['Archimedean', (e) => e.families.includes('archimedean')],
  ['Catalan', (e) => e.families.includes('catalan')],
  ['Johnson', (e) => e.families.includes('johnson')],
  ['Compounds', (e) => e.families.includes('compound')],
  ['Biscribed', (e) => e.families.includes('biscribed')
                    || e.families.includes('biscribed-dual')],
  ['Geodesic', (e) => e.families.includes('geodesic')],
  ['Toroids', (e) => e.families.includes('toroid')],
  ['Zonohedra', (e) => e.families.includes('zonohedron')],
  ['Dipyramids & Trapezohedra', (e) => e.families.includes('dipyramid')],
  ['Prisms & Antiprisms', (e) => e.families.includes('prism-family')
                              || e.families.includes('prism')
                              || e.families.includes('antiprism')],
  ['Uniform Duals', (e) => e.families.includes('uniform-dual')],
  ['Uniform', (e) => e.families.includes('uniform')],
  ['Stellations', (e) => e.families.includes('stellation')],
];

export const FAMILY_ORDER = FAMILY_RULES.map(([name]) => name);

/** The one family this solid belongs to. */
export function primaryFamily(entry) {
  for (const [name, test] of FAMILY_RULES) {
    if (test(entry)) return name;
  }
  return 'Other';
}

export function familyCounts(entries) {
  const counts = new Map();
  for (const e of entries) {
    const f = primaryFamily(e);
    counts.set(f, (counts.get(f) || 0) + 1);
  }
  return counts;
}

export function symmetryCounts(entries) {
  const counts = new Map();
  for (const e of entries) {
    const s = e.symmetry?.schoenflies;
    if (s) counts.set(s, (counts.get(s) || 0) + 1);
  }
  return counts;
}

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

/**
 * Filter the index by the current query state.
 *
 * `families` and `symmetries` are OR within a facet and AND across them,
 * which is the behaviour people expect from faceted search: ticking
 * "platonic" and "catalan" widens, ticking a symmetry as well narrows.
 */
export function filterEntries(entries, q) {
  const text = q.text ? norm(q.text) : '';
  const fams = q.families || [];
  const syms = q.symmetries || [];
  return entries.filter((e) => {
    if (fams.length && !fams.includes(primaryFamily(e))) return false;
    if (syms.length && !syms.includes(e.symmetry?.schoenflies)) return false;
    if (q.convex === 'convex' && !e.convex) return false;
    if (q.convex === 'nonconvex' && e.convex) return false;
    if (text) {
      const hay = norm(`${e.name} ${e.slug} ${Object.values(e.ids || {}).join(' ')}`);
      if (!hay.includes(text)) return false;
    }
    return true;
  });
}

export const SORTS = {
  name: (a, b) => a.name.localeCompare(b.name),
  faces: (a, b) => a.counts.faces - b.counts.faces || a.name.localeCompare(b.name),
  vertices: (a, b) => a.counts.vertices - b.counts.vertices || a.name.localeCompare(b.name),
  symmetry: (a, b) =>
    (b.symmetry?.order || 0) - (a.symmetry?.order || 0) ||
    a.name.localeCompare(b.name),
};
