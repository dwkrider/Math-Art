// Entry point for the Surfaces module.
//
// ONE surface is shown at a time. surface-viewer.js can drive several
// linked viewers at once -- that is what the validation report uses it
// for, and `linkViewers` is still exported for anyone who wants it --
// but comparison is a different task from browsing, and a catalogue that
// splits the stage in two halves the space each surface gets. So the
// multi-view capability stays a capability, and this module does not use
// it: `VIEWS` below is the single point where that choice lives.

import { loadIndex, loadRecord, loadMesh, loadMeshManifest, filterEntries, SORTS,
         FAMILY_ORDER, FAMILY_LABELS, MODE_LABELS, countBy, thumbUrl }
  from './surface-data.js';
import { SurfaceViewer, freshCanvas, STUDIO_VIEW, STUDIO_FOV,
         STUDIO_DISTANCE } from './surface-viewer.js';
import { renderSurfaceDetail } from './surface-detail.js';

const VIEWS = 1;

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

async function main() {
  const index = await loadIndex().catch((err) => {
    $('#catalog').textContent =
      `Could not load the surface database (${err.message}).`;
    throw err;
  });
  const entries = index.entries;
  const byslug = new Map(entries.map((e) => [e.slug, e]));
  // What was actually baked, rather than what the database predicts.
  const baked = await loadMeshManifest();

  // The studio's own camera: its orientation, its 84 mm lens and its
  // distance. Orientation alone was not enough -- with a wider, closer
  // lens a surface turned exactly right still reads as though seen from
  // somewhere else, because perspective enlarges whatever is nearest.
  // See STUDIO_FOV in surface-viewer.js.
  let viewer = new SurfaceViewer(freshCanvas($('#stage')), {
    rotation: STUDIO_VIEW,
    fov: STUDIO_FOV,
    distance: STUDIO_DISTANCE,
  });
  const detail = $('#detail');

  // -- catalogue ------------------------------------------------------
  const query = { text: '', families: [], modes: [], periodic: 'any',
                  drawable: 'any' };
  let sort = 'name';
  let selected = null;

  const head = el('div', 'catalog-head');
  const controls = el('div', 'catalog-controls');

  const search = el('input', 'search');
  search.type = 'search';
  search.placeholder = `Search ${entries.length} surfaces…`;
  search.setAttribute('aria-label', 'Search surfaces by name');
  search.addEventListener('input', () => { query.text = search.value; refresh(); });
  controls.append(search);

  const sortSel = el('select', 'sort');
  sortSel.setAttribute('aria-label', 'Sort order');
  for (const [k, label] of [['name', 'Name'], ['family', 'Family'],
                            ['genus', 'Genus'], ['drawable', 'Drawable first']]) {
    const o = el('option', null, label);
    o.value = k;
    sortSel.append(o);
  }
  sortSel.addEventListener('change', () => { sort = sortSel.value; refresh(); });
  controls.append(sortSel);

  const periodSel = el('select', 'convex');
  periodSel.setAttribute('aria-label', 'Periodicity');
  for (const [k, label] of [['any', 'Periodic or not'],
                            ['periodic', 'Periodic only'],
                            ['aperiodic', 'Aperiodic only']]) {
    const o = el('option', null, label);
    o.value = k;
    periodSel.append(o);
  }
  periodSel.addEventListener('change', () => {
    query.periodic = periodSel.value;
    refresh();
  });
  controls.append(periodSel);
  head.append(controls);

  const chipFacet = (title, values, labels, counts, key) => {
    const facet = el('div', 'facet');
    facet.append(el('h3', null, title));
    const chips = el('div', 'chips');
    for (const v of values) {
      const n = counts.get(v);
      if (!n) continue;
      const chip = el('button', 'chip');
      chip.type = 'button';
      chip.append(el('span', null, labels[v] || v));
      chip.append(el('span', 'chip-count', String(n)));
      chip.addEventListener('click', () => {
        const arr = query[key];
        const i = arr.indexOf(v);
        if (i >= 0) arr.splice(i, 1); else arr.push(v);
        chip.classList.toggle('on');
        refresh();
      });
      chips.append(chip);
    }
    facet.append(chips);
    return facet;
  };

  head.append(chipFacet('Family', FAMILY_ORDER, FAMILY_LABELS,
                        countBy(entries, (e) => e.primary_family), 'families'));
  head.append(chipFacet('Defined by', Object.keys(MODE_LABELS), MODE_LABELS,
                        countBy(entries, (e) => e.definition_mode), 'modes'));

  const status = el('p', 'catalog-status');
  head.append(status);
  $('#catalog').append(head);
  const grid = el('div', 'grid');
  $('#catalog').append(grid);

  function tile(e) {
    const card = el('button', 'tile');
    card.type = 'button';
    card.dataset.slug = e.slug;
    if (e.slug === selected) card.classList.add('on');
    if (baked.has(e.slug)) {
      const img = el('img');
      img.loading = 'lazy';
      img.decoding = 'async';
      img.width = 320;
      img.height = 320;
      img.alt = e.name;
      img.src = thumbUrl(e.slug);
      // Still guarded: a handful of implemented surfaces cannot be baked
      // (see UNDRIVEABLE in tests/test_web.py).
      img.addEventListener('error', () => img.classList.add('missing'));
      card.append(img);
    } else {
      // No generator builds this one, so there is no tile to ask for.
      // Requesting it anyway would 404 once per unimplemented record --
      // 61 of them -- on every page load, for an image we know is absent.
      card.append(el('span', 'tile-blank'));
    }
    card.append(el('span', 'tile-name', e.name));
    // Genus and periodicity are the two numbers that tell surfaces apart
    // at a glance, the way V/E/F does for solids.
    const bits = [];
    if (e.genus !== null && e.genus !== undefined) bits.push(`g${e.genus}`);
    if (e.periodicity_rank) bits.push(`${e.periodicity_rank}-periodic`);
    if (!baked.has(e.slug)) bits.push('no mesh');
    card.append(el('span', 'tile-meta', bits.join(' · ')));
    card.addEventListener('click', () => { location.hash = e.slug; });
    return card;
  }

  function refresh() {
    const found = filterEntries(entries, query).sort(SORTS[sort]);
    status.textContent = `${found.length} of ${entries.length} surfaces`;
    grid.textContent = '';
    for (const e of found) grid.append(tile(e));
    if (!found.length) {
      grid.append(el('p', 'empty', 'Nothing matches those filters.'));
    }
  }

  // -- stage ----------------------------------------------------------
  async function show(slug) {
    const entry = byslug.get(slug);
    if (!entry) return;
    selected = slug;
    for (const t of grid.querySelectorAll('.tile')) {
      t.classList.toggle('on', t.dataset.slug === slug);
    }
    const [rec, mesh] = await Promise.all([loadRecord(slug), loadMesh(slug)]);
    renderSurfaceDetail(rec, { ...entry, hasMesh: baked.has(slug) }, detail);
    document.title = `${rec.name} — Math Art`;
    $('#stage-caption').textContent = rec.name;

    const stage = $('#stage');
    if (mesh) {
      stage.classList.remove('empty');
      viewer.setMesh(mesh);
    } else {
      // A record with no generator has nothing to draw; the detail panel
      // says so, and the stage is blanked rather than left showing the
      // previous surface, which would be a quiet lie.
      stage.classList.add('empty');
      viewer.setMesh({ positions: new Float32Array(0),
                       indices: new Uint16Array(0) });
    }
  }

  for (const [id, opt] of [['opt-wireframe', 'wireframe'],
                           ['opt-rotate', 'autoRotate']]) {
    const box = $('#' + id);
    box.addEventListener('change', () => viewer.set(opt, box.checked));
  }
  for (const b of document.querySelectorAll('#view-bar button')) {
    b.addEventListener('click', () => viewer.setView(b.dataset.view));
  }

  addEventListener('hashchange', () => {
    const s = location.hash.slice(1);
    if (s) show(s).catch(() => {});
  });

  refresh();
  const initial = location.hash.slice(1)
    || (byslug.has('catenoid') ? 'catenoid' : entries[0].slug);
  await show(initial);
  grid.querySelector(`.tile[data-slug="${CSS.escape(initial)}"]`)
    ?.scrollIntoView({ block: 'nearest' });
}

main();
