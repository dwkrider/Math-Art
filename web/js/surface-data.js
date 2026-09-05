// Access to the surface database in data/surfaces/, and to the baked
// meshes in web/surfaces/.
//
// Shaped like data.js next door, with two differences that come straight
// from what a surface is:
//
//   * The family facet needs no precedence rule. Every record carries
//     `primary_family` -- the surface database chose one for each -- so
//     unlike the polyhedron tags, which overlap deliberately, this is a
//     partition already and the counts sum without help.
//   * Geometry is not in the record. data/surfaces/ deliberately stores
//     no meshes (a mesh is a rendering at a chosen resolution, not part
//     of the surface's identity), so the drawable geometry is fetched
//     separately from web/surfaces/<slug>.json, baked by
//     tools/surfdb_export.py from the add-on's own generators.

const DB = new URL('../data/surfaces/', import.meta.url);
const MESHES = new URL('../surfaces/', import.meta.url);

let manifestPromise = null;

/**
 * Which surfaces actually have a baked mesh.
 *
 * The database's `implemented` flag nearly answers this -- it is right
 * for 461 of the 462 records -- but a surface can be implemented as an
 * OPERATOR and still not be bakeable: plateau-span's operator spans a
 * selection and cannot run in an empty scene. tools/surfdb_export.py
 * writes down what it actually produced, so the page never has to keep
 * its own copy of that exception in step.
 */
export function loadMeshManifest() {
  if (!manifestPromise) {
    manifestPromise = fetch(new URL('../surface-meshes.json', import.meta.url))
      .then((r) => (r.ok ? r.json() : { meshes: [] }))
      .then((d) => new Set(d.meshes || []))
      .catch(() => new Set());
  }
  return manifestPromise;
}

let indexPromise = null;
const records = new Map();
const meshes = new Map();

export function loadIndex() {
  if (!indexPromise) {
    indexPromise = fetch(new URL('index.json', DB)).then((r) => {
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
  if (!entry) throw new Error(`unknown surface: ${slug}`);
  const p = fetch(new URL(entry.path, DB)).then((r) => {
    if (!r.ok) throw new Error(`${entry.path}: HTTP ${r.status}`);
    return r.json();
  });
  records.set(slug, p);
  return p;
}

/**
 * The baked mesh for a surface, or null when there is none.
 *
 * A missing mesh is a normal state, not an error: 61 of the 462 records
 * are not implemented by any generator, so no mesh can exist for them.
 * The module shows their mathematics with the viewer left empty.
 */
export async function loadMesh(slug) {
  if (meshes.has(slug)) return meshes.get(slug);
  const p = fetch(new URL(`${slug}.json`, MESHES))
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
  meshes.set(slug, p);
  return p;
}

export function thumbUrl(slug) {
  return new URL(`../thumbs/surfaces/${slug}.png`, import.meta.url).href;
}

// -- facets -----------------------------------------------------------

export const FAMILY_LABELS = {
  'minimal': 'Minimal',
  'minimal-periodic': 'Minimal (periodic)',
  'algebraic': 'Algebraic',
  'quadric': 'Quadric',
  'cmc': 'Constant mean curvature',
  'constant-curvature': 'Constant curvature',
  'cyclide': 'Cyclides',
  'revolution': 'Of revolution',
  'ruled': 'Ruled',
  'swept': 'Swept',
  'topological': 'Topological',
  'spectral': 'Spectral',
  'discrete': 'Discrete',
  'physical': 'Physical',
  'derived': 'Derived',
  'misc': 'Miscellaneous',
};

// Reading order: the curvature-defined families first, then the
// algebraic and classical ones, then the odds and ends.
export const FAMILY_ORDER = [
  'minimal', 'minimal-periodic', 'cmc', 'constant-curvature',
  'algebraic', 'quadric', 'cyclide', 'revolution', 'ruled', 'swept',
  'topological', 'spectral', 'discrete', 'physical', 'derived', 'misc',
];

export const MODE_LABELS = {
  parametric: 'Parametric',
  implicit: 'Implicit',
  nodal: 'Nodal',
  weierstrass: 'Weierstrass',
  derived: 'Derived',
  variational: 'Variational',
};

export function countBy(entries, pick) {
  const counts = new Map();
  for (const e of entries) {
    const k = pick(e);
    if (k === null || k === undefined) continue;
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  return counts;
}

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

export function filterEntries(entries, q) {
  const text = q.text ? norm(q.text) : '';
  const fams = q.families || [];
  const modes = q.modes || [];
  return entries.filter((e) => {
    if (fams.length && !fams.includes(e.primary_family)) return false;
    if (modes.length && !modes.includes(e.definition_mode)) return false;
    if (q.periodic === 'periodic' && !(e.periodicity_rank > 0)) return false;
    if (q.periodic === 'aperiodic' && e.periodicity_rank > 0) return false;
    if (q.drawable === 'yes' && !e.implemented) return false;
    if (q.drawable === 'no' && e.implemented) return false;
    if (text) {
      const hay = norm(`${e.name} ${e.slug} ${e.primary_family}`);
      if (!hay.includes(text)) return false;
    }
    return true;
  });
}

export const SORTS = {
  name: (a, b) => a.name.localeCompare(b.name),
  family: (a, b) =>
    FAMILY_ORDER.indexOf(a.primary_family) -
      FAMILY_ORDER.indexOf(b.primary_family) ||
    a.name.localeCompare(b.name),
  genus: (a, b) =>
    (a.genus ?? 99) - (b.genus ?? 99) || a.name.localeCompare(b.name),
  // Drawable first: a reader browsing for something to look at should not
  // have to step over the 61 records no generator implements.
  drawable: (a, b) =>
    (b.implemented ? 1 : 0) - (a.implemented ? 1 : 0) ||
    a.name.localeCompare(b.name),
};
