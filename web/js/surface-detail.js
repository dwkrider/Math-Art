// The mathematics panel for one surface.
//
// A surface record's centre of gravity is its DEFINITION -- the
// polynomial, the parametrisation, the level function, the Weierstrass
// data -- where a polyhedron's was its coordinate table. So the
// definition leads, shown as the formula it is, and the classification
// follows.
//
// Two fields are given unusual prominence because the surface database
// treats them as first-class and they are easy to misread:
//
//   * FIDELITY. A definition can be exact or an approximation, and that
//     is independent of how it is evaluated. The gyroid's nodal formula
//     is not the gyroid: measured, its mean curvature reaches 0.032
//     where a minimal surface has 0. The record says so; so does this.
//   * IMPLEMENTED. 61 records describe surfaces no generator builds.
//     Those have no mesh and the viewer stays empty, which needs saying
//     rather than looking like a failure to load.

const $ = (tag, cls, text) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
};

/** Light presentation of the record's formula strings. */
export function prettyFormula(expr) {
  if (!expr) return null;
  return String(expr)
    .replace(/\*\*/g, '^')
    .replace(/\bpi\b/g, 'π')
    .replace(/\bphi\b/g, 'φ')
    .replace(/\btheta\b/g, 'θ')
    .replace(/\bsqrt\b/g, '√')
    .replace(/\s*\*\s*/g, '·');
}

function section(title) {
  const s = $('section', 'panel-section');
  s.append($('h3', null, title));
  return s;
}

function defList(pairs) {
  const dl = $('dl', 'deflist');
  let any = false;
  for (const [k, v] of pairs) {
    if (v === null || v === undefined || v === '') continue;
    any = true;
    dl.append($('dt', null, k));
    dl.append($('dd', null, String(v)));
  }
  return any ? dl : null;
}

function formulaRow(label, expr) {
  if (!expr) return null;
  const row = $('div', 'formula');
  row.append($('span', 'formula-label', label));
  row.append($('code', null, prettyFormula(expr)));
  return row;
}

const EXTERNAL = {
  wikipedia: (v) => ['Wikipedia', `https://en.wikipedia.org/wiki/${encodeURIComponent(v)}`],
  mathworld: (v) => ['MathWorld', `https://mathworld.wolfram.com/${encodeURIComponent(v)}.html`],
};

const YES_NO = (v) => (v === undefined || v === null ? null : (v ? 'yes' : 'no'));

export function renderSurfaceDetail(rec, entry, host) {
  host.textContent = '';

  const head = $('header', 'detail-head');
  head.append($('h2', null, rec.name));
  if (rec.alternate_names?.length) {
    head.append($('p', 'alt-names', rec.alternate_names.join(' · ')));
  }
  const who = [rec.discovered_by, rec.year].filter(Boolean).join(', ');
  if (who) head.append($('p', 'alt-names', who));
  const tags = $('div', 'tag-row');
  for (const f of rec.families || []) tags.append($('span', 'tag', f));
  head.append(tags);
  host.append(head);

  if (entry && entry.hasMesh === false) {
    // Say it plainly rather than leaving an empty stage looking broken.
    // Two different reasons land here: no generator builds the surface at
    // all, or one does but it cannot be driven headlessly (an operator
    // that spans a selection has no canonical shape to bake).
    host.append($('p', 'notice',
      entry.implemented
        ? 'This surface is built by an operator that works on geometry you '
          + 'select, so it has no single shape to show here. Everything '
          + 'below is what the database records about it.'
        : 'No generator builds this surface yet, so there is no mesh to '
          + 'show. Everything below is what the database records about it.'));
  }

  const panels = $('div', 'panel-grid');
  host.append(panels);

  // -- definition, which is what a surface actually is
  const d = rec.definition || {};
  const sec = section('Definition');
  const meta = defList([
    ['Mode', d.mode],
    ['Fidelity', d.fidelity],
    ['Exactness', d.exactness],
    ['Scale', d.scale],
    ['Lattice', d.lattice],
    ['Level', d.level],
  ]);
  if (meta) sec.append(meta);
  for (const row of [
    formulaRow('x', d.x), formulaRow('y', d.y), formulaRow('z', d.z),
    formulaRow('F', d.polynomial),
    formulaRow('level', d.level_function),
    formulaRow('g', d.g), formulaRow('dh', d.dh),
  ]) if (row) sec.append(row);
  if (d.u_range) {
    sec.append(formulaRow('u', `${d.u_range[0]} … ${d.u_range[1]}`));
  }
  if (d.v_range) {
    sec.append(formulaRow('v', `${d.v_range[0]} … ${d.v_range[1]}`));
  }
  if (d.fidelity === 'approximation' && d.residual) {
    const r = d.residual;
    sec.append($('p', 'provenance',
      `This is an approximation, and the database measures how far: mean `
      + `curvature reaches ${r.max_abs_mean_curvature} on the level set `
      + `(sampled at resolution ${r.measured_at_resolution}), where a `
      + `minimal surface has 0.`));
  }
  if (d.note) sec.append($('p', 'provenance', d.note));
  panels.append(sec);

  // -- curvature
  const c = rec.curvature || {};
  const curv = defList([
    ['Condition', c.condition],
    ['Mean curvature', c.mean?.exact ?? c.mean?.value],
    ['Gauss curvature', c.gauss?.exact ?? c.gauss?.value],
    ['Total curvature', c.total_curvature?.exact ?? c.total_curvature?.value],
    ['Also satisfies', (c.also_satisfies || []).join(', ')],
  ]);
  if (curv) {
    const s = section('Curvature');
    s.append(curv);
    panels.append(s);
  }

  // -- topology
  const t = rec.topology || {};
  const ends = (t.ends || [])
    .map((e) => `${e.count} × ${e.type}${e.embedded ? '' : ' (immersed)'}`)
    .join(', ');
  const topo = defList([
    ['Genus', t.genus],
    ['Euler characteristic', t.euler_characteristic],
    ['Orientable', YES_NO(t.orientable)],
    ['One-sided', YES_NO(t.one_sided)],
    ['Complete', YES_NO(t.complete)],
    ['Compact', YES_NO(t.compact)],
    ['Boundary components', t.boundary_components],
    ['Ends', ends],
    ['Finite total curvature', YES_NO(t.finite_total_curvature)],
    ['Class', t.class],
  ]);
  if (topo) {
    const s = section('Topology');
    s.append(topo);
    panels.append(s);
  }

  // -- symmetry
  const y = rec.symmetry || {};
  const sym = defList([
    ['Kind', y.kind],
    ['Continuous', y.continuous],
    ['Point group', y.point_residual?.schoenflies],
    ['Space group', y.space_group ?? y.symbol],
    ['Periodicity rank', y.periodicity_rank],
    ['Order', y.order],
    ['Chiral', YES_NO(y.chiral)],
    ['Verified by', y.verified_by],
  ]);
  if (sym) {
    const s = section('Symmetry');
    s.append(sym);
    panels.append(s);
  }

  // -- embedding and fabrication
  const emb = rec.embedding || {};
  const fab = rec.fabrication || {};
  const eb = defList([
    ['Quality', emb.quality],
    ['Self-intersecting', YES_NO(emb.self_intersecting)],
    ['Developable', YES_NO(fab.developable)],
    ['Watertight when printed', YES_NO(fab.printable_watertight)],
    ['Needs thickening', YES_NO(fab.requires_thickening)],
  ]);
  if (eb) {
    const s = section('Embedding & fabrication');
    s.append(eb);
    panels.append(s);
  }

  // -- relations, as links: following one is the natural next move
  const rel = rec.relations || {};
  const relRows = [
    ['Conjugate', rel.conjugate],
    ['Associate family', rel.associate_family],
    ['Dual', rel.dual],
    ['Limit of', (rel.limit_of || []).join(', ')],
  ].filter(([, v]) => v);
  if (relRows.length) {
    const s = section('Relations');
    const dl = $('dl', 'deflist');
    for (const [k, v] of relRows) {
      dl.append($('dt', null, k));
      const dd = $('dd');
      for (const part of String(v).split(', ')) {
        const a = $('a', null, part);
        a.href = `#${part}`;
        dd.append(a);
        dd.append(document.createTextNode(' '));
      }
      dl.append(dd);
    }
    s.append(dl);
    panels.append(s);
  }

  // -- sources
  const prov = rec.provenance || {};
  const s = section('Sources');
  const ids = rec.ids || {};
  const links = $('p', 'ext-links');
  for (const [key, make] of Object.entries(EXTERNAL)) {
    if (!ids[key]) continue;
    const [label, href] = make(ids[key]);
    const a = $('a', null, label);
    a.href = href;
    a.target = '_blank';
    a.rel = 'noopener';
    links.append(a);
  }
  if (links.childNodes.length) s.append(links);
  if (prov.definition) s.append($('p', 'provenance', prov.definition));
  if (prov.sources?.length) {
    const ul = $('ul', 'sources');
    for (const src of prov.sources) ul.append($('li', null, src));
    s.append(ul);
  }
  host.append(s);
}
