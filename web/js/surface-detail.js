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
//   * IMPLEMENTED. 7 records have no mesh, each for a stated reason.
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

/**
 * The description block: what the surface is, and its formula.
 *
 * Both halves are built by tools/surfdb/describe.py and stored on the
 * record, so this only places them. In particular the formula is NOT
 * assembled here: it is compiled from the same expression the mesher
 * integrates, precisely so that the equation on the page and the surface
 * in the viewport cannot disagree.
 *
 * MATHML, AND NO LIBRARY. The maths is stored as MathML with the LaTeX
 * carried inside it as an <annotation>. Browsers render MathML Core
 * natively, so the alternative -- a vendored KaTeX at ~280 KB, on a site
 * whose whole JS is a fraction of that -- buys nothing except a
 * dependency. The LaTeX rides along because it is what a reader wants to
 * copy, and it is offered as a click.
 */
function renderDescription(rec, host) {
  const d = rec.description;
  if (!d) return;
  const sec = $('section', 'panel-section description');
  if (d.curated) {
    sec.append($('p', 'description-prose', d.curated));
    // The templated line still earns its place under curated prose: it
    // states the classification in the catalogue's own vocabulary, which
    // the prose deliberately does not.
    sec.append($('p', 'description-summary', d.summary));
  } else {
    sec.append($('p', 'description-prose', d.summary));
  }

  // The caption is dropped when there is only one formula: the section
  // it sits in already says "Formula", so "IMPLICIT EQUATION" underneath
  // it is the same word twice. Where a record carries several -- a Gauss
  // map and a height differential, or x, y and z -- the label is the
  // only thing telling them apart, so it stays.
  const many = (d.formulas || []).length > 1;
  for (const f of d.formulas || []) {
    const fig = $('figure', 'eqn');
    if (many) fig.append($('figcaption', 'eqn-label', f.label));
    const box = $('div', 'eqn-math');
    // The MathML is generated, not user content, and is inserted as
    // markup because that is what it is. It has to be parsed in the
    // MathML namespace or the browser builds unknown HTML elements that
    // lay out as inline text -- the equation appears as a run of loose
    // numbers and letters, which is the failure this line avoids.
    const doc = new DOMParser().parseFromString(
      f.mathml, 'application/xhtml+xml');
    const node = doc.documentElement;
    if (node && node.localName === 'math') {
      box.append(document.importNode(node, true));
    } else {
      // Unparseable: show the LaTeX rather than nothing, and never
      // innerHTML something that did not parse as MathML.
      box.append($('code', null, f.latex));
    }
    fig.append(box);

    const copy = $('button', 'eqn-copy', 'copy LaTeX');
    copy.type = 'button';
    copy.addEventListener('click', () => {
      navigator.clipboard?.writeText(f.latex).then(
        () => { copy.textContent = 'copied';
                setTimeout(() => { copy.textContent = 'copy LaTeX'; }, 1200); },
        () => { copy.textContent = 'copy failed'; });
    });
    fig.append(copy);
    sec.append(fig);
  }
  host.append(sec);
}


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

  renderDescription(rec, host);

  if (entry && entry.hasMesh === false) {
    // Say WHY, in the database's own words where it has them.
    //
    // The records that cannot be drawn are not one kind of thing. Some
    // describe surfaces proved not to exist -- the finding IS the record
    // -- and some are merely unbuilt and may become buildable. Reporting
    // both as "no generator builds this yet" states the second correctly
    // and the first not at all.
    const blocked = (rec.construction || [])
      .map((c) => c.blocked_by).filter(Boolean)[0];
    const resume = (rec.construction || [])
      .map((c) => c.resume).filter(Boolean)[0];
    const note = $('p', 'notice');
    if (blocked) {
      // Only the non-existence case gets a lead-in. The others already
      // begin by saying they are not built, and a prefix produced
      // "Not built yet. Not built, and not buildable by this add-on".
      if (/NOT TO EXIST/.test(blocked)) {
        note.append($('strong', null, 'There is nothing to draw. '));
      }
      note.append(document.createTextNode(blocked));
      if (resume) {
        note.append(document.createElement('br'));
        note.append(document.createTextNode(resume));
      }
    } else if (entry.implemented) {
      note.textContent =
        'This surface is built by an operator that works on geometry you '
        + 'select, so it has no single shape to show here.';
    } else {
      note.textContent =
        'No generator builds this surface yet, so there is no mesh to show.';
    }
    host.append(note);
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
  // The defining expressions are NOT repeated here. They are set
  // properly at the top of the panel, from the record's own
  // `description.formulas`, and printing the same polynomial again as a
  // row of ASCII a few centimetres below is the same thing twice -- the
  // second time worse. What stays is what the typeset formula does not
  // carry: the mode and fidelity above, and the parameter ranges below,
  // which bound the chart rather than define it.
  if (d.u_range) {
    sec.append(formulaRow('u', `${d.u_range[0]} … ${d.u_range[1]}`));
  }
  if (d.v_range) {
    sec.append(formulaRow('v', `${d.v_range[0]} … ${d.v_range[1]}`));
  }
  panels.append(sec);

  // Prose goes BELOW the two-column grid, at full width.
  //
  // The panel grid is right for label/value pairs and wrong for a
  // paragraph: a 13rem column turns a definition note into a ribbon of
  // three or four words a line. These notes run to a hundred words, so
  // they get the panel's whole width.
  const prose = [];
  if (d.fidelity === 'approximation' && d.residual) {
    const r = d.residual;
    prose.push(`This is an approximation, and the database measures how `
      + `far: mean curvature reaches ${r.max_abs_mean_curvature} on the `
      + `level set (sampled at resolution ${r.measured_at_resolution}), `
      + `where a minimal surface has 0.`);
  }
  if (d.note) prose.push(d.note);
  if (prose.length) {
    const notes = $('section', 'panel-section detail-notes');
    notes.append($('h3', null, 'Notes on the definition'));
    for (const t of prose) notes.append($('p', 'provenance', t));
    host.append(notes);
  }

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
