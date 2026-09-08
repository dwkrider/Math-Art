// The catalogue: faceted browsing over the whole database.
//
// Every facet is derived from index.json, so filtering and sorting the
// whole database costs no further requests -- a record is fetched only
// when a solid is actually opened.
//
// Tiles use native lazy loading. Each solid has a real rendered
// thumbnail in web/thumbs/, produced by tools/render_polyhedra_thumbs.py
// from the same record the viewer draws, so the grid stays cheap and the
// tile always matches the model it opens.

import { familyCounts, filterEntries, SORTS, thumbUrl, FAMILY_ORDER }
  from './data.js';
import { ClusterView, preloadAtlas } from './cluster-view.js';
import { polyhedronVector } from './cluster-polyhedra.js';

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
    this.query = { text: '', families: [], convex: 'any' };
    this.sort = 'name';
    this.selected = null;
    this.build();
    this.refresh();
  }

  build() {
    // Search and facets go in a header that sticks to the top of the
    // viewport (see .catalog-head). The catalogue itself has no inner
    // scroll region -- the page scrolls it -- so without this the filters
    // would simply leave the screen a few rows into the grid.
    const head = $('div', 'catalog-head');
    const controls = $('div', 'catalog-controls');

    const search = $('input', 'search');
    search.type = 'search';
    search.placeholder = `Search ${this.entries.length} solids…`;
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

    // GRID is the default. The clustered view answers a different
    // question -- what is near what -- and is worth opting into rather
    // than landing in. Same arrangement as the surfaces module.
    this.mode = 'grid';
    this.cluster = null;
    const modeBar = $('div', 'segmented view-modes');
    modeBar.setAttribute('role', 'group');
    modeBar.setAttribute('aria-label', 'Catalogue view');
    for (const [k, label] of [['grid', 'Grid'], ['cluster', 'Clusters']]) {
      const b = $('button', 'seg' + (k === 'grid' ? ' on' : ''), label);
      b.type = 'button';
      b.addEventListener('click', () => {
        for (const o of modeBar.children) o.classList.remove('on');
        b.classList.add('on');
        this.mode = k;
        this.refresh();
      });
      modeBar.append(b);
    }
    head.append(modeBar);

    this.status = $('p', 'catalog-status');
    head.append(this.status);
    this.host.append(head);

    this.grid = $('div', 'grid');
    this.host.append(this.grid);

    this.clusterWrap = $('div', 'cluster-wrap');
    this.clusterCanvas = $('canvas', 'cluster-canvas');
    this.clusterWrap.append(this.clusterCanvas);
    this.clusterWrap.hidden = true;
    this.host.append(this.clusterWrap);

    // Fetch the sprite sheet now rather than on the first switch: the
    // view will not lay anything out until it arrives, so asking for it
    // late means waiting then. After the page's own loading, since the
    // grid does not use it.
    if ('requestIdleCallback' in window) requestIdleCallback(() => preloadAtlas('polyhedra'));
    else setTimeout(() => preloadAtlas('polyhedra'), 0);
  }

  refresh() {
    const found = filterEntries(this.entries, this.query).sort(SORTS[this.sort]);
    this.status.textContent =
      `${found.length} of ${this.entries.length} solids`;

    const clustered = this.mode === 'cluster';
    this.grid.hidden = clustered;
    this.clusterWrap.hidden = !clustered;

    if (clustered) {
      if (!this.cluster) {
        this.cluster = new ClusterView(this.clusterCanvas, this.entries, {
          atlas: 'polyhedra',
          vector: polyhedronVector,
          thumbUrl,
          onSelect: (slug) => {
            this.select(slug);
            this.onSelect(slug);
          },
        });
      }
      this.cluster.show(found);
      if (this.selected) this.cluster.select(this.selected);
      return;
    }

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
    if (this.cluster) this.cluster.select(slug);
  }

  reveal(slug) {
    // Only meaningful for the grid: the clustered view has no scroll
    // position to move, and it frames its whole field by construction.
    if (this.mode !== 'grid') return;
    const el = this.grid.querySelector(`.tile[data-slug="${CSS.escape(slug)}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }
}
