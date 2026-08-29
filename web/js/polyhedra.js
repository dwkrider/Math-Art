// Entry point for the Polyhedra module page.

import { loadIndex, loadRecord } from './data.js';
import { Catalog } from './catalog.js';
import { Viewer, STYLES } from './viewer.js';
import { renderDetail } from './detail.js';

const $ = (sel) => document.querySelector(sel);

const STYLE_LABELS = {
  solid: 'Solid',
  leonardo: 'Leonardo',
  wireframe: 'Wireframe',
  'ball-and-stick': 'Ball and stick',
};

async function main() {
  const index = await loadIndex().catch((err) => {
    $('#catalog').textContent =
      `Could not load the polyhedron database (${err.message}).`;
    throw err;
  });

  const viewer = new Viewer($('#stage'));
  const detail = $('#detail');

  const catalog = new Catalog($('#catalog'), index.entries, (slug) => {
    // The hash is the page's address bar: a solid is linkable, and the
    // back button walks the reader's own path through the catalogue.
    if (location.hash.slice(1) !== slug) location.hash = slug;
    else show(slug);
  });

  async function show(slug) {
    const rec = await loadRecord(slug);
    viewer.show(rec);
    viewer.resetView();
    renderDetail(rec, detail, (next) => { location.hash = next; });
    catalog.select(slug);
    document.title = `${rec.name} — Math Art`;
    $('#stage-caption').textContent = rec.name;
  }

  // -- style / coloring controls
  const styleBar = $('#style-bar');
  for (const s of STYLES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'seg' + (s === 'solid' ? ' on' : '');
    b.textContent = STYLE_LABELS[s];
    b.addEventListener('click', () => {
      for (const other of styleBar.children) other.classList.remove('on');
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
    const slug = location.hash.slice(1);
    if (slug) show(slug).catch(() => {});
  });

  const initial = location.hash.slice(1) ||
    (index.entries.find((e) => e.slug === 'cube')?.slug) ||
    index.entries[0].slug;
  await show(initial);
  catalog.reveal(initial);
}

main();
