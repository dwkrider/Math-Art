// Entry point for the Chair44 module page.
//
// The page is a rebuild over five decisions -- how deep to grow the
// patch, how to draw the matching rule, how tall to exaggerate it, how
// far to shrink each chair, and how to colour them. Nothing is
// animated, so every change rebuilds the patch from scratch; a depth-3
// patch of the bare tile takes a few milliseconds and of the featured
// tile a fraction of a second, both reported under the view.

import { maxRelief } from './chair44-math.js';
import { ChairView } from './chair44-view.js';

const $ = (sel) => document.querySelector(sel);

const DEPTHS = [0, 1, 2, 3];
const FEATURE_LABELS = {
  ARROWS: 'Arrows',
  EXAGGERATED: 'Enlarged pyramids',
  TRUE: 'True size',
  NONE: 'Bare chair',
};

function main() {
  const view = new ChairView($('#stage'));

  const state = {
    depth: 2,
    features: 'ARROWS',
    relief: 30,
    gap: 0.92,
    colorBy: 'FRAME',
  };

  const readout = $('#readout');
  const notes = $('#notes-line');
  const reliefField = $('#relief-field');
  const reliefInput = $('#relief');
  const reliefOut = $('#relief-out');
  const gapInput = $('#gap');
  const gapOut = $('#gap-out');

  function rebuild() {
    // The pyramids are drawn taller than life; past a point the tallest
    // bump on one chair would poke into the one facing it across the
    // gap. The engine's own bound is what the add-on clamps to, so the
    // two agree.
    const bound = maxRelief(state.gap);
    const relief = Math.min(state.relief, bound);
    const clamped = relief < state.relief - 1e-9;

    const stats = view.build({
      depth: state.depth,
      features: state.features,
      relief,
      gap: state.gap,
      colorBy: state.colorBy,
    });
    const rep = ChairView.report(state.depth);

    const n = (x) => x.toLocaleString();
    readout.textContent =
      `${n(stats.chairs)} chair${stats.chairs === 1 ? '' : 's'} · `
      + `${n(stats.vertices)} vertices · ${n(stats.triangles)} triangles · `
      + `built in ${stats.buildMs.toFixed(0)} ms`;

    const bits = [];
    if (rep.total) {
      bits.push(rep.inside === rep.total
        ? `All ${rep.total} face contacts in this patch are among the 44 the `
          + 'features permit.'
        : `${rep.total - rep.inside} of ${rep.total} contacts are NOT in the `
          + 'atlas, which should not happen.');
    }
    if (clamped) {
      bits.push(`Relief clamped to ${relief.toFixed(0)}× so the pyramids stay `
                + `clear at a gap of ${state.gap.toFixed(2)}.`);
    }
    if (state.features === 'TRUE') {
      bits.push('True size: the pyramids are 1/10000 of a cell tall, so the '
                + 'chair looks bare. That is the point — the rule is invisible '
                + 'at any honest scale.');
    }
    notes.textContent = bits.join(' ');
    notes.hidden = !bits.length;

    document.body.dataset.chairs = String(stats.chairs);
    document.body.dataset.buildMs = stats.buildMs.toFixed(1);
  }

  // -- depth ---------------------------------------------------------
  for (const b of document.querySelectorAll('#depth-bar button')) {
    b.addEventListener('click', () => {
      for (const o of document.querySelectorAll('#depth-bar button')) {
        o.classList.toggle('on', o === b);
      }
      state.depth = Number(b.dataset.depth);
      rebuild();
    });
  }
  for (const b of document.querySelectorAll('#depth-bar button')) {
    b.classList.toggle('on', Number(b.dataset.depth) === state.depth);
  }

  // -- the matching rule ---------------------------------------------
  const featSel = $('#features');
  for (const [k, label] of Object.entries(FEATURE_LABELS)) {
    const o = document.createElement('option');
    o.value = k;
    o.textContent = label;
    if (k === state.features) o.selected = true;
    featSel.append(o);
  }
  featSel.addEventListener('change', () => {
    state.features = featSel.value;
    reliefField.hidden = state.features !== 'EXAGGERATED';
    rebuild();
  });
  reliefField.hidden = state.features !== 'EXAGGERATED';

  reliefInput.value = String(state.relief);
  reliefOut.textContent = `${state.relief}×`;
  reliefInput.addEventListener('input', () => {
    state.relief = Number(reliefInput.value);
    reliefOut.textContent = `${state.relief}×`;
    rebuild();
  });

  // -- gap and colour ------------------------------------------------
  gapInput.value = String(state.gap);
  gapOut.textContent = state.gap.toFixed(2);
  gapInput.addEventListener('input', () => {
    state.gap = Number(gapInput.value);
    gapOut.textContent = state.gap.toFixed(2);
    rebuild();
  });

  const colourSel = $('#colour');
  colourSel.value = state.colorBy;
  colourSel.addEventListener('change', () => {
    state.colorBy = colourSel.value;
    rebuild();
  });

  // -- view ----------------------------------------------------------
  const spinBtn = $('#spin');
  spinBtn.addEventListener('click', () => {
    view.spin = view.spin ? 0 : 0.25;
    spinBtn.textContent = view.spin ? 'Stop' : 'Spin';
    spinBtn.setAttribute('aria-pressed', view.spin ? 'true' : 'false');
  });
  $('#reset').addEventListener('click', () => view.resetView());

  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    view.spin = 0;
  }

  rebuild();
  document.body.dataset.ready = '1';
}

main();
