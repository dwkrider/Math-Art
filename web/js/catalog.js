// The catalogue: faceted browsing over all 448 solids.
//
// Every facet is derived from index.json, so filtering and sorting the
// whole database costs no further requests -- a record is fetched only
// when a solid is actually opened.
//
// Tiles use native lazy loading. Each solid has a real rendered
// thumbnail in web/thumbs/, produced by tools/render_polyhedra_thumbs.py
// from the same record the viewer draws, so the grid stays cheap and the
// tile always matches the model it opens.

import { familyCounts, symmetryCounts, filterEntries, SORTS, thumbUrl,
         FAMILY_ORDER } from './data.js';

const $ = (tag, cls, text) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
};

export class Catalog {
  constructor(host, entries, onSelect) {
    this.host = host;
    this.entries = entries;
    this.onSelect = onSelect;
    this.query = { text: '', families: [], symmetries: [], convex: 'any' };
    this.sort = 'name';
    this.selected = null;
    this.build();
    this.refresh();
  }

  build() {
    // Search and facets sit in a fixed header and only the tile grid
    // scrolls. Putting them in the same scroll box hid them the moment
    // the grid was scrolled -- including on load, since revealing the
    // selected tile scrolls the box -- so the filters were invisible
    // until the window was resized.
    const head = $('div', 'catalog-head');
    const controls = $('div', 'catalog-controls');

    const search = $('input', 'search');
    search.type = 'search';
    search.placeholder = 'Search 448 solids…';
    search.setAttribute('aria-label', 'Search solids by name');
    search.addEventListener('input', () => {
      this.query.text = search.value;
      this.refresh();
    });
    controls.append(search);

    const sortSel = $('select', 'sort');
    sortSel.setAttribute('aria-label', 'Sort order');
    for (const [k, label] of [['name', 'Name'], ['faces', 'Face count'],
                              ['vertices', 'Vertex count'],
                              ['symmetry', 'Symmetry order']]) {
      const o = $('option', null, label);
      o.value = k;
      sortSel.append(o);
    }
    sortSel.addEventListener('change', () => {
      this.sort = sortSel.value;
      this.refresh();
    });
    controls.append(sortSel);

    const convexSel = $('select', 'convex');
    convexSel.setAttribute('aria-label', 'Convexity');
    for (const [k, label] of [['any', 'Convex or not'], ['convex', 'Convex only'],
                              ['nonconvex', 'Non-convex only']]) {
      const o = $('option', null, label);
      o.value = k;
      convexSel.append(o);
    }
    convexSel.addEventListener('change', () => {
      this.query.convex = convexSel.value;
      this.refresh();
    });
    controls.append(convexSel);

    head.append(controls);

    // -- family facet, in the generators' own vocabulary. Because it is a
    // partition, these counts sum to the whole database.
    const famCounts = familyCounts(this.entries);
    const fams = $('div', 'facet');
    fams.append($('h3', null, 'Family'));
    const famList = $('div', 'chips');
    for (const f of FAMILY_ORDER) {
      const n = famCounts.get(f);
      if (!n) continue;
      const chip = $('button', 'chip');
      chip.type = 'button';
      chip.append($('span', null, f));
      chip.append($('span', 'chip-count', String(n)));
      chip.addEventListener('click', () => {
        const i = this.query.families.indexOf(f);
        if (i >= 0) this.query.families.splice(i, 1);
        else this.query.families.push(f);
        chip.classList.toggle('on');
        this.refresh();
      });
      famList.append(chip);
    }
    fams.append(famList);
    head.append(fams);

    // -- symmetry facet
    const symCounts = [...symmetryCounts(this.entries)]
      .sort((a, b) => b[1] - a[1]);
    const syms = $('div', 'facet');
    syms.append($('h3', null, 'Symmetry group'));
    const symList = $('div', 'chips');
    for (const [g, n] of symCounts) {
      const chip = $('button', 'chip');
      chip.type = 'button';
      chip.append($('span', null, g));
      chip.append($('span', 'chip-count', String(n)));
      chip.addEventListener('click', () => {
        const i = this.query.symmetries.indexOf(g);
        if (i >= 0) this.query.symmetries.splice(i, 1);
        else this.query.symmetries.push(g);
        chip.classList.toggle('on');
        this.refresh();
      });
      symList.append(chip);
    }
    syms.append(symList);
    head.append(syms);

    this.status = $('p', 'catalog-status');
    head.append(this.status);
    this.host.append(head);

    this.grid = $('div', 'grid');
    this.host.append(this.grid);
  }

  refresh() {
    const found = filterEntries(this.entries, this.query).sort(SORTS[this.sort]);
    this.status.textContent =
      `${found.length} of ${this.entries.length} solids`;
    this.grid.textContent = '';
    for (const e of found) this.grid.append(this.tile(e));
    if (!found.length) {
      this.grid.append($('p', 'empty', 'Nothing matches those filters.'));
    }
  }

  tile(e) {
    const card = $('button', 'tile');
    card.type = 'button';
    card.dataset.slug = e.slug;
    if (e.slug === this.selected) card.classList.add('on');

    const img = $('img');
    img.loading = 'lazy';
    img.decoding = 'async';
    img.width = 320;
    img.height = 320;
    img.alt = e.name;
    img.src = thumbUrl(e.slug);
    // A missing tile must not leave a broken-image glyph in the grid;
    // the name below it still identifies the solid.
    img.addEventListener('error', () => { img.classList.add('missing'); });
    card.append(img);

    card.append($('span', 'tile-name', e.name));
    const meta = `${e.counts.vertices}·${e.counts.edges}·${e.counts.faces}`;
    card.append($('span', 'tile-meta', meta));

    card.addEventListener('click', () => {
      this.select(e.slug);
      this.onSelect(e.slug);
    });
    return card;
  }

  select(slug) {
    this.selected = slug;
    for (const el of this.grid.querySelectorAll('.tile')) {
      el.classList.toggle('on', el.dataset.slug === slug);
    }
  }

  reveal(slug) {
    const el = this.grid.querySelector(`.tile[data-slug="${CSS.escape(slug)}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }
}
