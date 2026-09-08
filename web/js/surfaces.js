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
         FAMILY_ORDER, FAMILY_LABELS, countBy, thumbUrl }
  from './surface-data.js';
import { SurfaceViewer, freshCanvas, STUDIO_VIEW, STUDIO_FOV,
         STUDIO_DISTANCE } from './surface-viewer.js';
import { renderSurfaceDetail } from './surface-detail.js';
import { ClusterView, preloadAtlas } from './cluster-view.js';

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
  const query = { text: '', families: [] };
  const sort = 'name';        // no selector any more; alphabetical
  let selected = null;

  const head = el('div', 'catalog-head');

  // GRID is the default. The clustered view answers a different question
  // -- what is near what -- and is worth opting into, not landing in.
  let mode = 'grid';
  let cluster = null;

  // Fetch the tile sheet now, not when the reader first opens the
  // clustered view. The view will not start laying anything out until it
  // has arrived, so requesting it only on the switch means waiting then;
  // requesting it here means it is normally already decoded. It is one
  // image the grid does not use, so it is started after the page has
  // done its own loading rather than competing with it.
  if ('requestIdleCallback' in window) requestIdleCallback(() => preloadAtlas());
  else setTimeout(() => preloadAtlas(), 0);
  const modeBar = el('div', 'segmented view-modes');
  modeBar.setAttribute('role', 'group');
  modeBar.setAttribute('aria-label', 'Catalogue view');
  for (const [k, label] of [['grid', 'Grid'], ['cluster', 'Clusters']]) {
    const b = el('button', 'seg' + (k === 'grid' ? ' on' : ''), label);
    b.type = 'button';
    b.addEventListener('click', () => {
      for (const o of modeBar.children) o.classList.remove('on');
      b.classList.add('on');
      mode = k;
      refresh();
    });
    modeBar.append(b);
  }
  head.append(modeBar);

  // Two controls only: filter by name, pick a family. The chip banks for
  // family and definition mode ran to twenty-odd buttons and took more
  // vertical space than the catalogue they were filtering.
  const controls = el('div', 'catalog-controls');

  const search = el('input', 'search');
  search.type = 'search';
  search.placeholder = `Search ${entries.length} surfaces…`;
  search.setAttribute('aria-label', 'Search surfaces by name');
  search.addEventListener('input', () => { query.text = search.value; refresh(); });
  controls.append(search);

  const famSel = el('select', 'sort');
  famSel.setAttribute('aria-label', 'Family');
  const famCounts = countBy(entries, (e) => e.primary_family);
  const anyOpt = el('option', null, `All families (${entries.length})`);
  anyOpt.value = '';
  famSel.append(anyOpt);
  for (const f of FAMILY_ORDER) {
    const n = famCounts.get(f);
    if (!n) continue;
    const o = el('option', null, `${FAMILY_LABELS[f] || f} (${n})`);
    o.value = f;
    famSel.append(o);
  }
  famSel.addEventListener('change', () => {
    query.families = famSel.value ? [famSel.value] : [];
    refresh();
  });
  controls.append(famSel);
  head.append(controls);

  const status = el('p', 'catalog-status');
  head.append(status);
  $('#catalog').append(head);
  const grid = el('div', 'grid');
  $('#catalog').append(grid);
  const clusterWrap = el('div', 'cluster-wrap');
  const clusterCanvas = el('canvas', 'cluster-canvas');
  clusterWrap.append(clusterCanvas);
  clusterWrap.hidden = true;
  $('#catalog').append(clusterWrap);

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
    const clustered = mode === 'cluster';
    grid.hidden = clustered;
    clusterWrap.hidden = !clustered;

    if (clustered) {
      if (!cluster) {
        cluster = new ClusterView(clusterCanvas, entries, {
          thumbUrl,
          hasMesh: (slug) => baked.has(slug),
          onSelect: (slug) => { location.hash = slug; },
        });
      }
      cluster.show(found);
      if (selected) cluster.select(selected);
      return;
    }
    grid.textContent = '';
    for (const e of found) grid.append(tile(e));
    if (!found.length) {
      grid.append(el('p', 'empty', 'Nothing matches those filters.'));
    }
  }

  // -- stage ----------------------------------------------------------
  async function show(slug) {
    const entry = byslug.get(slug);
    if (!entry) {
      // Say so, rather than leaving the previous surface up as though it
      // were the one asked for. Records do get removed -- horgan-surface
      // was, being a proved non-existence -- so a stale link is a real
      // way to arrive here.
      detail.textContent = '';
      const p = document.createElement('p');
      p.className = 'notice';
      p.textContent = `No surface in the database has the name "${slug}".`;
      detail.append(p);
      $('#stage-caption').textContent = '';
      $('#stage').classList.add('empty');
      return;
    }
    selected = slug;
    for (const t of grid.querySelectorAll('.tile')) {
      t.classList.toggle('on', t.dataset.slug === slug);
    }
    if (cluster) cluster.select(slug);
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
