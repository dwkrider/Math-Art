// Entry point for the Hopf fibration module page.
//
// Two scenes are kept in step: the fibres, and the base sphere they
// come from. Every base point has one fibre and one dot, in one
// colour, and the page never draws one without the other -- which is
// the whole idea of a fibration, made into an interface.
//
// Picking is the part a film cannot do. Click the base sphere and a
// circle appears; drag and you paint a curve of them; two points show
// the link. The presets are the generator's own, so anything seen here
// can be rebuilt in Blender.

import { buildFibers, basePoints, linkingNumber, projectFiber, normalize3,
         apply3, rotMatrix, TILT } from './hopf-math.js';
import { HopfView, BaseSphere } from './hopf-view.js';
import { BundleView, BUNDLES } from './bundle-view.js';

const $ = (sel) => document.querySelector(sel);

const PRESET_LABELS = {
  CUSTOM: 'Your own points',
  PAIR: 'Two fibres (a Hopf link)',
  FLOWER: 'One ring of latitude',
  LATITUDES: 'Nested tori',
  GREATCIRCLE: 'Great-circle band',
  CAP: 'Cap spiral',
  LOXODROME: 'Loxodrome',
  CURL: 'Curl',
  FIBONACCI: 'Fibonacci sphere',
  TETRA: 'Tetrahedron',
  OCTA: 'Octahedron',
  CUBE: 'Cube',
  ICOSA: 'Icosahedron',
  DODECA: 'Dodecahedron',
};

// Presets the latitude slider means something for.
const RING_LIKE = new Set(['FLOWER', 'LATITUDES', 'CAP', 'LOXODROME', 'CURL']);

function main() {
  const view = new HopfView($('#stage'));
  const state = {
    preset: 'FLOWER',
    nFiber: 24,
    nLat: 5,
    ring: 60,          // colatitude of the single ring, degrees
    latMin: 20,
    latMax: 160,
    P: 1,
    Q: 1,
    chirality: 'RIGHT',
    palette: 'RAINBOW',
    includeAxis: true,
    radius: 0.022,
    flow: 0,           // S^3 rotation, degrees
    playing: false,
    custom: [],        // hand-picked base points, already tilted
  };

  const sphere = new BaseSphere($('#base-sphere'), {
    onPick: (b, how) => {
      if (state.preset !== 'CUSTOM') {
        state.preset = 'CUSTOM';
        state.custom = [];
        presetSel.value = 'CUSTOM';
        syncFields();
      }
      if (how === 'start' || state.custom.length < 400) state.custom.push(b);
      rebuild();
    },
  });

  const readout = $('#readout');
  const linkLine = $('#link-line');
  const presetSel = $('#preset');
  const paletteSel = $('#palette');
  const chiralSel = $('#chirality');

  // ---- what to draw ------------------------------------------------
  function buildArgs() {
    if (state.preset === 'CUSTOM' || state.preset === 'PAIR') {
      // Hand-picked points are already in tilted coordinates, so they
      // go in with the tilt switched off -- otherwise clicking the top
      // of the sphere would put a fibre somewhere else.
      return {
        preset: 'FIBONACCI', nFiber: 1, samples: 200,
        _custom: state.preset === 'PAIR' ? pairPoints() : state.custom,
      };
    }
    const single = state.preset === 'FLOWER';
    return {
      preset: state.preset,
      nLat: state.nLat,
      nFiber: state.nFiber,
      samples: 200,
      P: state.P, Q: state.Q,
      latMin: single ? state.ring : state.latMin,
      latMax: single ? state.ring : state.latMax,
      chirality: state.chirality,
      includeAxis: state.includeAxis,
      s3Rot: state.flow,
      extra: { turns: 4.0, curl_lobes: 6, curl_amp: 22.0 },
    };
  }

  function pairPoints() {
    const R = rotMatrix(...TILT);
    return [apply3(R, normalize3([0.35, 0.15, 0.55])),
            apply3(R, normalize3([-0.45, 0.5, -0.25]))];
  }

  function rebuild() {
    const args = buildArgs();
    if (args._custom) {
      args.points = args._custom;
      delete args._custom;
      Object.assign(args, {
        P: state.P, Q: state.Q, chirality: state.chirality,
        s3Rot: state.flow, includeAxis: state.includeAxis,
      });
    }
    // While the flow plays the picture is rebuilt every frame, so it
    // is drawn a little coarser: fewer samples along each fibre and a
    // thinner-walled tube. Still exact -- the samples are points of
    // the real curve -- just fewer of them, and only while moving.
    const stats = view.build({
      ...args,
      samples: state.playing ? 120 : (args.samples ?? 200),
      sides: state.playing ? 6 : 8,
      palette: state.palette,
      radius: state.radius,
    });
    const bases = view.built.bases;
    sphere.show(bases, state.palette);

    const n = (x) => x.toLocaleString();
    readout.textContent =
      `${n(stats.fibers)} fibre${stats.fibers === 1 ? '' : 's'}`
      + (stats.dropped ? ` · ${stats.dropped} off the edge of the projection` : '')
      + ` · ${n(stats.vertices)} vertices · drawn in ${stats.buildMs.toFixed(0)} ms`;

    // The claim the picture makes, checked on the picture: any two
    // distinct fibres are linked exactly once.
    if (bases.length === 2) {
      const a = projectFiber(bases[0], 400);
      const b = projectFiber(bases[1], 400);
      const lk = linkingNumber(a, b);
      linkLine.textContent =
        `Linking number of these two fibres: ${lk.toFixed(3)} — computed here `
        + 'from the two curves, not assumed. Any two distinct fibres of the '
        + 'Hopf map are linked exactly once, which is why the whole family '
        + 'cannot be combed apart.';
      linkLine.hidden = false;
    } else {
      linkLine.hidden = true;
    }
    document.body.dataset.fibers = String(stats.fibers);
    document.body.dataset.buildMs = stats.buildMs.toFixed(1);
  }

  // ---- controls ----------------------------------------------------
  for (const [k, label] of Object.entries(PRESET_LABELS)) {
    const o = document.createElement('option');
    o.value = k;
    o.textContent = label;
    if (k === state.preset) o.selected = true;
    presetSel.append(o);
  }
  presetSel.addEventListener('change', () => {
    state.preset = presetSel.value;
    if (state.preset === 'CUSTOM') state.custom = state.custom.slice();
    syncFields();
    rebuild();
  });

  const slider = (id, key, fmt) => {
    const el = $(id);
    const out = $(id + '-out');
    el.value = String(state[key]);
    if (out) out.textContent = fmt(state[key]);
    el.addEventListener('input', () => {
      state[key] = Number(el.value);
      if (out) out.textContent = fmt(state[key]);
      rebuild();
    });
    return el;
  };
  slider('#fibers', 'nFiber', (v) => String(v));
  slider('#rings', 'nLat', (v) => String(v));
  slider('#ring', 'ring', (v) => `${v}°`);
  slider('#wind-p', 'P', (v) => String(v));
  slider('#wind-q', 'Q', (v) => String(v));
  const flowEl = slider('#flow', 'flow', (v) => `${v}°`);

  chiralSel.addEventListener('change', () => {
    state.chirality = chiralSel.value;
    rebuild();
  });
  paletteSel.addEventListener('change', () => {
    state.palette = paletteSel.value;
    rebuild();
  });
  $('#axis').addEventListener('change', (e) => {
    state.includeAxis = e.target.checked;
    rebuild();
  });
  $('#clear').addEventListener('click', () => {
    state.custom = [];
    state.preset = 'CUSTOM';
    presetSel.value = 'CUSTOM';
    syncFields();
    rebuild();
  });
  $('#undo').addEventListener('click', () => {
    state.custom.pop();
    rebuild();
  });
  $('#reset-view').addEventListener('click', () => view.resetView());

  const playBtn = $('#play');
  playBtn.addEventListener('click', () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? 'Pause flow' : 'Play flow';
    playBtn.setAttribute('aria-pressed', state.playing ? 'true' : 'false');
  });
  view.onFrame = (dt) => {
    if (!state.playing) return;
    state.flow = (state.flow + 18 * dt) % 360;
    flowEl.value = String(Math.round(state.flow));
    $('#flow-out').textContent = `${Math.round(state.flow)}°`;
    rebuild();
  };

  // Which controls apply depends on the preset; hiding the rest keeps
  // the panel honest about what is actually doing something.
  function syncFields() {
    const p = state.preset;
    const custom = p === 'CUSTOM' || p === 'PAIR';
    $('#fibers-field').hidden = custom || p === 'LATITUDES' ? p === 'CUSTOM' || p === 'PAIR' : false;
    $('#rings-field').hidden = p !== 'LATITUDES';
    $('#ring-field').hidden = !RING_LIKE.has(p) || p === 'LATITUDES';
    $('#pick-tools').hidden = !custom;
  }
  syncFields();

  // ---- the warm-up figure -------------------------------------------
  const bundle = new BundleView($('#bundle-stage'));
  const bundleNote = $('#bundle-note');
  let bundleKind = 'CYLINDER';
  function showBundle(kind) {
    bundleKind = kind;
    const b = BUNDLES[kind];
    bundle.show(kind, 1);
    bundleNote.innerHTML =
      `<strong>Base space:</strong> ${b.base}. <strong>Fibre:</strong> ${b.fibre}. `
      + `<strong>Total space:</strong> ${b.total}. ${b.note}`;
    for (const el of document.querySelectorAll('#bundle-bar button')) {
      el.classList.toggle('on', el.dataset.bundle === kind);
    }
  }
  for (const el of document.querySelectorAll('#bundle-bar button')) {
    el.textContent = BUNDLES[el.dataset.bundle].label;
    el.addEventListener('click', () => showBundle(el.dataset.bundle));
  }
  showBundle(bundleKind);

  rebuild();
  document.body.dataset.ready = '1';
}

main();
