// Entry point for the Chair44 module page.
//
// The page is a rebuild over five decisions -- how deep to grow the
// patch, how to draw the matching rule, how tall to exaggerate it, how
// far to shrink each chair, and how to colour them -- plus one that is
// not a rebuild at all: the build slider, which only changes how much
// of the finished buffer is drawn.

import { maxRelief } from './chair44-math.js';
import { ChairView, planFor, tileTriangles, MAX_TRIANGLES } from './chair44-view.js';

const $ = (sel) => document.querySelector(sel);

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
    shown: 1.0,          // fraction of the patch drawn, 0 to 1
  };

  const readout = $('#readout');
  const notes = $('#notes-line');
  const reliefField = $('#relief-field');
  const reliefInput = $('#relief');
  const reliefOut = $('#relief-out');
  const gapInput = $('#gap');
  const gapOut = $('#gap-out');
  const builtInput = $('#built');
  const builtOut = $('#built-out');
  const featSel = $('#features');

  // The contact report depends on the depth alone, so it is worked out
  // once per depth -- and never in the way of the picture. At depth 5
  // it walks 229,376 cells and takes about a second, which as part of
  // the rebuild would be a second of frozen page before anything
  // appeared. So the patch is drawn first and the line fills in when
  // the count arrives.
  const reports = new Map();
  let reportToken = 0;

  function withReport(depth, then) {
    if (reports.has(depth)) {
      then(reports.get(depth));
      return;
    }
    const mine = ++reportToken;
    then(null);
    setTimeout(() => {
      const rep = ChairView.report(depth);
      reports.set(depth, rep);
      if (mine === reportToken) then(rep);
    }, 0);
  }

  let stats = null;

  function describeShown() {
    if (!stats) return;
    const n = Math.round(state.shown * stats.chairs);
    view.setShown(n);
    builtOut.textContent = `${Math.round(state.shown * 100)}%`;
    const fmt = (x) => x.toLocaleString();
    readout.textContent =
      (n === stats.chairs
        ? `${fmt(stats.chairs)} chair${stats.chairs === 1 ? '' : 's'}`
        : `${fmt(n)} of ${fmt(stats.chairs)} chairs`)
      + ` · ${fmt(stats.vertices)} vertices · ${fmt(stats.triangles)} triangles`
      + ` · built in ${stats.buildMs.toFixed(0)} ms`;
    document.body.dataset.shown = String(n);
  }

  function rebuild() {
    // Too big to draw? Fall back along the feature modes rather than
    // refusing, and say so. The depth is what the reader asked for;
    // how finely the rule is drawn is the part that can give way.
    const plan = planFor(state.depth, state.features);
    const features = plan.mode;

    // The pyramids are drawn taller than life; past a point the tallest
    // bump on one chair would poke into the one facing it across the
    // gap. The engine's own bound is what the add-on clamps to, so the
    // two agree.
    const bound = maxRelief(state.gap);
    const relief = Math.min(state.relief, bound);
    const clamped = relief < state.relief - 1e-9 && features === 'EXAGGERATED';

    stats = view.build({
      depth: state.depth,
      features,
      relief,
      gap: state.gap,
      colorBy: state.colorBy,
    });

    builtInput.max = String(stats.chairs);
    builtInput.step = '1';
    builtInput.value = String(Math.round(state.shown * stats.chairs));
    describeShown();

    const fixed = [];
    if (plan.fellBack) {
      const want = 8 ** state.depth * tileTriangles(state.features);
      fixed.push(`${FEATURE_LABELS[state.features]} would be `
                 + `${(want / 1e6).toFixed(0)} million triangles at this size, `
                 + `past the ${(MAX_TRIANGLES / 1e6).toFixed(0)} million this page `
                 + `redraws comfortably — showing `
                 + `${FEATURE_LABELS[features].toLowerCase()} instead.`);
    }
    if (clamped) {
      fixed.push(`Relief clamped to ${relief.toFixed(0)}× so the pyramids stay `
                 + `clear at a gap of ${state.gap.toFixed(2)}.`);
    }
    if (features === 'TRUE') {
      fixed.push('True size: the pyramids are 1/10000 of a cell tall, so the '
                 + 'chair looks bare. That is the point — the rule is invisible '
                 + 'at any honest scale.');
    }
    const depthNow = state.depth;
    withReport(depthNow, (rep) => {
      if (depthNow !== state.depth) return;
      const bits = fixed.slice();
      if (rep === null) {
        if (8 ** depthNow > 4096) bits.push('Checking every face contact against the atlas…');
      } else if (rep.total) {
        bits.push(rep.inside === rep.total
          ? `All ${rep.total.toLocaleString()} face contacts in this patch are `
            + 'among the 44 the features permit.'
          : `${(rep.total - rep.inside).toLocaleString()} of `
            + `${rep.total.toLocaleString()} contacts are NOT in the atlas, `
            + 'which should not happen.');
      }
      notes.textContent = bits.join(' ');
      notes.hidden = !bits.length;
      document.body.dataset.atlas = rep ? `${rep.inside}/${rep.total}` : 'pending';
    });

    document.body.dataset.chairs = String(stats.chairs);
    document.body.dataset.buildMs = stats.buildMs.toFixed(1);
    document.body.dataset.features = features;
  }

  // -- depth ---------------------------------------------------------
  for (const b of document.querySelectorAll('#depth-bar button')) {
    b.classList.toggle('on', Number(b.dataset.depth) === state.depth);
    b.addEventListener('click', () => {
      for (const o of document.querySelectorAll('#depth-bar button')) {
        o.classList.toggle('on', o === b);
      }
      state.depth = Number(b.dataset.depth);
      rebuild();
    });
  }

  // -- the matching rule ---------------------------------------------
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

  // -- how much of it is built ---------------------------------------
  // Chairs come in the order the substitution made them, so winding
  // this back takes the patch apart supertile by supertile and winding
  // it forward puts it back one chair at a time.
  builtInput.addEventListener('input', () => {
    state.shown = stats ? Number(builtInput.value) / stats.chairs : 1;
    describeShown();
  });
  for (const b of document.querySelectorAll('.built-snap')) {
    b.addEventListener('click', () => {
      state.shown = Number(b.dataset.shown);
      if (stats) builtInput.value = String(Math.round(state.shown * stats.chairs));
      describeShown();
    });
  }

  // -- view ----------------------------------------------------------
  const spinBtn = $('#spin');
  spinBtn.addEventListener('click', () => {
    view.spin = view.spin ? 0 : 0.25;
    spinBtn.textContent = view.spin ? 'Stop' : 'Spin';
    spinBtn.setAttribute('aria-pressed', view.spin ? 'true' : 'false');
  });
  $('#reset').addEventListener('click', () => view.resetView());

  addEventListener('keydown', (e) => {
    if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    if (!stats) return;
    // one chair at a time from the keyboard, which is the only way to
    // step a 32,768-chair patch singly
    const n = Math.round(state.shown * stats.chairs) + (e.key === 'ArrowRight' ? 1 : -1);
    state.shown = Math.max(0, Math.min(stats.chairs, n)) / stats.chairs;
    builtInput.value = String(Math.round(state.shown * stats.chairs));
    describeShown();
  });

  rebuild();
  document.body.dataset.ready = '1';
}

main();
