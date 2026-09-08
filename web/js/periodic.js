// Entry point for the Periodic Table module page.
//
// The arrangement is not decided here. tools/build_periodic_table.py
// works it out from the polyhedron records -- which need
// `symmetry.transitivity` and `symmetry.orbits`, neither of which is in
// index.json -- and ships web/periodic-table.json. This file draws it,
// and opens a solid when one is clicked.
//
// The viewer and the detail panel are the polyhedra module's, unchanged:
// same Viewer, same renderDetail, same records. A solid opened from the
// table is the same object it is in the catalogue, which is the point.

import { loadRecord, thumbUrl } from './data.js';
import { Viewer, STYLES } from './viewer.js';
import { renderDetail } from './detail.js';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const STYLE_LABELS = {
  solid: 'Solid',
  leonardo: 'Leonardo',
  wireframe: 'Wireframe',
  'ball-and-stick': 'Ball and stick',
};

// Colour by classification, because that is the reading a reader brings
// to a periodic table: blocks of related things. It cuts across the
// columns on purpose -- a column is about regularity, the colour is
// about provenance, and seeing the Johnson solids spread across seven
// columns is the table earning its keep.
const KIND_CLASS = {
  Platonic: 'k-platonic',
  Archimedean: 'k-archimedean',
  Catalan: 'k-catalan',
  Johnson: 'k-johnson',
  'Kepler-Poinsot': 'k-kepler',
};

async function main() {
  const host = $('#table');
  let table;
  try {
    const r = await fetch(new URL('../periodic-table.json', import.meta.url));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    table = await r.json();
  } catch (err) {
    host.textContent = `Could not load the table (${err.message}).`;
    return;
  }

  host.textContent = '';

  // WHAT FORMS A COLUMN IS THE READER'S CHOICE.
  //
  // There is no single right answer -- kinds of face, kinds of vertex,
  // symmetry group, classification and sheer size are all defensible,
  // and they disagree interestingly. Every scheme places all 127; none
  // is a filter. tools/build_periodic_table.py works them all out and
  // ships the assignments, so switching is instant and no scheme can
  // quietly drop a solid.
  const schemes = new Map(table.schemes.map((s) => [s.key, s]));
  let schemeKey = table.default_scheme;
  let current = null;

  const bar = el('div', 'periodic-controls');
  const label = el('label', 'periodic-select');
  label.append(el('span', null, 'Columns:'));
  const sel = el('select', 'sort');
  sel.setAttribute('aria-label', 'What forms a column');
  for (const s of table.schemes) {
    const o = el('option', null, s.label);
    o.value = s.key;
    if (s.key === schemeKey) o.selected = true;
    sel.append(o);
  }
  label.append(sel);
  bar.append(label);
  const note = el('p', 'periodic-note');
  bar.append(note);
  host.append(bar);

  const legend = el('div', 'periodic-legend');
  for (const [kind, cls] of Object.entries(KIND_CLASS)) {
    const n = table.solids.filter((s) => s.kind === kind).length;
    const item = el('span', 'legend-item');
    item.append(el('span', `legend-swatch ${cls}`));
    item.append(el('span', null, `${kind} (${n})`));
    legend.append(item);
  }
  host.append(legend);

  const grid = el('div', 'periodic-grid');
  host.append(grid);

  function draw() {
    const scheme = schemes.get(schemeKey);
    note.textContent = scheme.note;
    grid.textContent = '';
    const byColumn = new Map(scheme.columns.map((c) => [c.key, []]));
    for (const s of table.solids) {
      const k = s.columns[scheme.key];
      if (byColumn.has(k)) byColumn.get(k).push(s);
    }
    for (const col of scheme.columns) {
      const column = el('div', 'periodic-column');
      // Only where the claim is true: a "noble" column is one whose
      // members cannot be made more uniform in the terms being grouped
      // by. Grouping by symmetry ends with the icosahedral solids, which
      // are the most symmetric but not closed in that sense, so that
      // scheme highlights nothing.
      if (scheme.noble && col.key === scheme.noble) {
        column.classList.add('noble');
      }
      const head = el('div', 'column-head');
      head.append(el('span', 'column-label', col.label));
      head.append(el('span', 'column-count', String(col.count)));
      if (col.note) head.title = col.note;
      column.append(head);
      for (const s of byColumn.get(col.key) || []) column.append(cell(s));
      grid.append(column);
    }
    if (current) {
      for (const c of grid.querySelectorAll('.periodic-cell')) {
        c.classList.toggle('on', c.dataset.slug === current);
      }
    }
  }

  sel.addEventListener('change', () => {
    schemeKey = sel.value;
    draw();
  });

  // -- the detail panel, opened on click ---------------------------
  const panel = $('#panel');
  const viewer = new Viewer($('#stage'));
  const detail = $('#detail');

  function cell(s) {
    const b = el('button', `periodic-cell ${KIND_CLASS[s.kind] || ''}`);
    b.type = 'button';
    b.dataset.slug = s.slug;
    b.title = `${s.name} — ${s.kind}, ${s.schoenflies}, `
            + `${s.faces} faces in ${s.orbits} `
            + `${s.orbits === 1 ? 'class' : 'classes'}`;

    const img = el('img');
    img.loading = 'lazy';
    img.decoding = 'async';
    img.width = 320;
    img.height = 320;
    img.alt = s.name;
    img.src = thumbUrl(s.slug);
    img.addEventListener('error', () => img.classList.add('missing'));
    b.append(img);
    // The face count sits where an atomic number would, because it is
    // the closest thing to one -- an ordering, not an identifier.
    b.append(el('span', 'cell-number', String(s.faces)));
    b.append(el('span', 'cell-name', s.name));
    b.append(el('span', 'cell-sym', s.schoenflies || ''));
    b.addEventListener('click', () => { location.hash = s.slug; });
    return b;
  }

  async function show(slug) {
    const s = table.solids.find((x) => x.slug === slug);
    if (!s) return;
    for (const c of grid.querySelectorAll('.periodic-cell')) {
      c.classList.toggle('on', c.dataset.slug === slug);
    }
    current = slug;
    panel.hidden = false;
    const rec = await loadRecord(slug);
    if (current !== slug) return;          // a later click won the race
    viewer.show(rec);
    viewer.resetView();
    renderDetail(rec, detail, (next) => { location.hash = next; });
    document.title = `${rec.name} — Periodic Table — Math Art`;
    $('#stage-caption').textContent = rec.name;
    panel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function close() {
    panel.hidden = true;
    current = null;
    for (const c of grid.querySelectorAll('.periodic-cell')) {
      c.classList.remove('on');
    }
    if (location.hash) history.replaceState(null, '', location.pathname);
    document.title = 'Periodic Table of Polyhedra — Math Art';
  }
  draw();

  $('#panel-close').addEventListener('click', close);
  addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });

  // -- style controls, the polyhedra module's ----------------------
  const styleBar = $('#style-bar');
  for (const s of STYLES) {
    const b = el('button', 'seg' + (s === 'solid' ? ' on' : ''),
                 STYLE_LABELS[s] || s);
    b.type = 'button';
    b.addEventListener('click', () => {
      for (const o of styleBar.children) o.classList.remove('on');
      b.classList.add('on');
      viewer.setStyle(s);
    });
    styleBar.append(b);
  }
  const colorToggle = $('#color-toggle');
  colorToggle.addEventListener('change', () => {
    viewer.setColoring(colorToggle.checked ? 'auto' : 'none');
  });
  $('#reset-view').addEventListener('click', () => viewer.resetView());

  addEventListener('hashchange', () => {
    const s = location.hash.slice(1);
    if (s) show(s).catch(() => {});
    else close();
  });
  const initial = location.hash.slice(1);
  if (initial) show(initial).catch(() => {});
}

main();
