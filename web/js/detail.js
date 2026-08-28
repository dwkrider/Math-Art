// The mathematics panel for one solid.
//
// The database records far more than "vertices and faces", and the point
// of this panel is to present that as mathematics rather than as a dump
// of JSON. In particular metrics carry an `exact` algebraic expression
// alongside the float, so the panel leads with sqrt(3)/2 and keeps
// 0.8660254 as the secondary reading -- the exact form is the true
// statement, the decimal is only a measurement of it.

const $ = (tag, cls, text) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
};

/**
 * Render the database's small exact-expression language for a reader.
 *
 * The grammar is deliberately tiny -- integers, + - * /, parentheses,
 * sqrt(), acos(), pi, phi -- so this is presentation only: no parsing,
 * no evaluation, and anything unrecognised passes through untouched
 * rather than being mangled.
 */
export function prettyExact(expr) {
  if (!expr) return null;
  let s = String(expr);
  s = s.replace(/\bsqrt\(([^()]*)\)/g, '√($1)');
  s = s.replace(/√\((\d+)\)/g, '√$1');       // sqrt(3) -> √3
  s = s.replace(/\bpi\b/g, 'π');
  s = s.replace(/\bphi\b/g, 'φ');
  s = s.replace(/\*/g, '·');
  return s;
}

function metric(label, m) {
  if (!m) return null;
  const row = $('div', 'metric');
  row.append($('span', 'metric-label', label));
  const val = $('span', 'metric-value');
  const exact = prettyExact(m.exact);
  if (exact) {
    val.append($('span', 'exact', exact));
    val.append($('span', 'approx', ` ${(+m.value).toFixed(6)}`));
  } else if (m.value !== undefined && m.value !== null) {
    val.append($('span', 'exact', (+m.value).toFixed(6)));
  } else return null;
  row.append(val);
  return row;
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

const EXTERNAL = {
  wikipedia: (v) => [`Wikipedia`, `https://en.wikipedia.org/wiki/${encodeURIComponent(v)}`],
  wolfram: (v) => [`Wolfram MathWorld`, `https://mathworld.wolfram.com/${encodeURIComponent(v)}.html`],
};

export function renderDetail(rec, host, onNavigate) {
  host.textContent = '';

  const head = $('header', 'detail-head');
  head.append($('h2', null, rec.name));
  if (rec.alternate_names?.length) {
    head.append($('p', 'alt-names', rec.alternate_names.join(' · ')));
  }
  const tags = $('div', 'tag-row');
  for (const f of rec.families || []) tags.append($('span', 'tag', f));
  head.append(tags);
  host.append(head);

  // -- counts, up front: this is what a reader checks first
  const c = rec.combinatorics.counts;
  const counts = $('div', 'counts');
  for (const [k, v] of [['Vertices', c.vertices], ['Edges', c.edges],
                        ['Faces', c.faces]]) {
    const cell = $('div', 'count-cell');
    cell.append($('div', 'count-value', String(v)));
    cell.append($('div', 'count-label', k));
    counts.append(cell);
  }
  host.append(counts);

  // -- notation
  const n = rec.notation || {};
  const notation = defList([
    ['Schläfli', n.schlafli],
    ['Wythoff', n.wythoff],
    ['Coxeter diagram', n.coxeter_diagram],
    ['Conway', n.conway],
    ['Vertex configuration', (n.vertex_configuration || []).join(', ')],
    ['Face configuration', n.face_configuration],
  ]);
  if (notation) {
    const s = section('Notation');
    s.append(notation);
    host.append(s);
  }

  // -- symmetry
  const y = rec.symmetry || {};
  const t = y.transitivity || {};
  const trans = ['vertex', 'edge', 'face'].filter((k) => t[k]);
  const sym = defList([
    ['Schoenflies', y.schoenflies],
    ['Orbifold', y.orbifold],
    ['Coxeter', y.coxeter],
    ['Hermann–Mauguin', y.hermann_mauguin],
    ['Order', y.order],
    ['Rotation subgroup', y.rotation_group && `${y.rotation_group} (order ${y.rotation_order})`],
    ['Chiral', y.chiral === undefined ? null : (y.chiral ? 'yes' : 'no')],
    ['Transitive on', trans.length ? trans.join(', ') : null],
  ]);
  if (sym) {
    const s = section('Symmetry');
    s.append(sym);
    host.append(s);
  }

  // -- combinatorics
  const cb = rec.combinatorics;
  const faceTypes = (cb.faces_by_type || [])
    .map((f) => `${f.count} × ${f.schlafli || `${f.sides}-gon`}`).join(', ');
  const combi = defList([
    ['Faces by type', faceTypes],
    ['Euler characteristic', cb.euler_characteristic],
    ['Genus', cb.genus],
    ['Density', cb.density],
    ['Orientable', cb.orientable === undefined ? null : (cb.orientable ? 'yes' : 'no')],
    ['Convex', cb.convex === undefined ? null : (cb.convex ? 'yes' : 'no')],
    ['Self-dual', cb.self_dual ? 'yes' : null],
  ]);
  if (combi) {
    const s = section('Combinatorics');
    s.append(combi);
    if (cb.dual) {
      // The dual is a link, not a string: following it is the single most
      // natural move a reader makes from this panel.
      const p = $('p', 'dual-link');
      p.append(document.createTextNode('Dual: '));
      const a = $('a', null, cb.dual);
      a.href = `#${cb.dual}`;
      a.addEventListener('click', (ev) => {
        ev.preventDefault();
        onNavigate?.(cb.dual);
      });
      p.append(a);
      s.append(p);
    }
    host.append(s);
  }

  // -- metrics
  const m = rec.metrics || {};
  const s = section('Metrics');
  const note = $('p', 'metric-note',
    `Normalised to ${(m.normalization || 'edge length 1').replace(/_/g, ' ')}.`);
  s.append(note);
  const rows = [
    metric('Edge length', m.edge_length),
    metric('Circumradius', m.circumradius),
    metric('Midradius', m.midradius),
    metric('Surface area', m.surface_area),
    metric('Volume', m.volume),
  ].filter(Boolean);
  for (const r of rows) s.append(r);
  for (const ir of m.inradius || []) {
    const r = metric(`Inradius to ${ir.face || 'face'}`, ir);
    if (r) s.append(r);
  }
  for (const da of m.dihedral_angles || []) {
    const row = $('div', 'metric');
    const between = (da.between || []).join(' / ');
    row.append($('span', 'metric-label',
      between ? `Dihedral ${between}` : 'Dihedral angle'));
    const val = $('span', 'metric-value');
    val.append($('span', 'exact', `${(+da.degrees).toFixed(4)}°`));
    if (da.exact) val.append($('span', 'approx', ` ${prettyExact(da.exact)}`));
    row.append(val);
    s.append(row);
  }
  if (rows.length || (m.dihedral_angles || []).length) host.append(s);

  // -- catalogue numbers and outside references
  const ids = rec.ids || {};
  const idList = defList([
    ['Uniform', ids.uniform],
    ['Wenninger', ids.wenninger],
    ['Johnson', ids.johnson],
    ['Coxeter', ids.coxeter_clm],
    ['Netlib', ids.netlib],
    ['Bowers', ids.bowers],
  ]);
  if (idList) {
    const sec = section('Catalogue numbers');
    sec.append(idList);
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
    if (links.childNodes.length) sec.append(links);
    host.append(sec);
  }

  // -- how this record was made
  const con = rec.construction || {};
  const prov = rec.provenance || {};
  const sec = section('Construction');
  const cons = defList([
    ['Generator', con.generator],
    ['Operator', con.operator_id],
    ['Conway', con.conway_from],
    ['Wythoff', con.wythoff_from?.symbol],
  ]);
  if (cons) sec.append(cons);
  if (prov.coordinates) sec.append($('p', 'provenance', prov.coordinates));
  if (prov.sources?.length) {
    const ul = $('ul', 'sources');
    for (const src of prov.sources) ul.append($('li', null, src));
    sec.append($('h4', null, 'Sources'));
    sec.append(ul);
  }
  host.append(sec);
}
