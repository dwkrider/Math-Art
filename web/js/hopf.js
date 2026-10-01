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
         apply3, rotMatrix, haloAround, TILT } from './hopf-math.js';
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
  let picks = 0;
  // what the picture was last framed for: the camera is re-fitted
  // when the shape changes, not when a parameter is merely
  // animating, which would otherwise make the scene breathe
  let lastFrameKey = null;
  // The fit held still for the duration of an animation. Letting
  // buildFibers work it out per frame makes the whole picture lurch:
  // as a swept torus grows, the centre and the scale travel with it,
  // measured at 0.3 of a radius between two-degree steps. So it is
  // computed once, over the extremes the animation will reach, and
  // handed back in on every frame.
  let heldFit = null;
  let lastFitKey = null;
  const RING_MIN = 8, RING_MAX = 172;
  const state = {
    preset: 'FLOWER',
    nFiber: 36,
    nLat: 6,
    // How far the rings of latitude reach from the equator. A ring at
    // colatitude b projects to a torus of outer size 1/cos(b/2) +
    // tan(b/2), which runs away as b approaches 180 degrees: at the
    // generator's own 20-160 the outermost torus is ten times the
    // innermost, so the inner ones shrink to a knot in the middle and
    // the picture stops reading as a nest. At 40 it is under three
    // times, which is what the films show.
    spread: 50,
    ring: 60,          // colatitude of the single ring, degrees

    P: 1,
    Q: 1,
    chirality: 'RIGHT',
    palette: 'RAINBOW',
    includeAxis: true,
    radius: 0.022,
    flow: 0,           // S^3 rotation, degrees
    spin: 0,           // tumble of the base points, degrees
    halo: 0,           // ring of points around each one, degrees
    haloN: 8,
    anim: 'LATITUDE',  // what Play animates
    speed: 1,
    playing: false,
    sweepUp: true,     // which way the latitude is travelling
    custom: [],        // hand-picked base points, already tilted
  };

  const sphere = new BaseSphere($('#base-sphere'), {
    onPick: (b, how) => {
      // counted for the tests: picking is the page's whole point, and
      // a silent regression in it would not show up in any screenshot
      picks++;
      document.body.dataset.picks = String(picks);
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
  // The halo turns each chosen base point into a small ring of them.
  // Presets hand back single points, so when a halo is wanted the page
  // expands them here and passes the result in as explicit points --
  // the same door the hand-picked ones go through. Everything after
  // that is the generator's kernel, untouched.
  function expand(points) {
    if (!state.halo) return points;
    const r = state.halo * Math.PI / 180;
    const out = [];
    for (const p of points) out.push(...haloAround(p, r, state.haloN));
    return out;
  }

  function presetPoints() {
    const single = state.preset === 'FLOWER';
    const raw = basePoints(state.preset, state.nLat, state.nFiber,
                           single ? state.ring : 90 - state.spread,
                           single ? state.ring : 90 + state.spread,
                           { turns: 4.0, curl_lobes: 6, curl_amp: 22.0 });
    const R = rotMatrix(TILT[0] + state.spin * Math.PI / 180, TILT[1], TILT[2]);
    return raw.map((b) => apply3(R, b));
  }

  function buildArgs() {
    if (state.preset === 'CUSTOM' || state.preset === 'PAIR') {
      // Hand-picked points are already in tilted coordinates, so they
      // go in with the tilt switched off -- otherwise clicking the top
      // of the sphere would put a fibre somewhere else.
      return {
        preset: 'FIBONACCI', nFiber: 1, samples: 200,
        _custom: expand(spun(state.preset === 'PAIR' ? pairPoints() : state.custom)),
      };
    }
    if (state.halo) {
      return {
        preset: state.preset, samples: 200, P: state.P, Q: state.Q,
        chirality: state.chirality, includeAxis: state.includeAxis,
        s3Rot: state.flow, _custom: expand(presetPoints()),
      };
    }
    const single = state.preset === 'FLOWER';
    const lo = 90 - state.spread, hi = 90 + state.spread;
    return {
      preset: state.preset,
      // The tumble turns the base points themselves, which is the
      // motion the films show on their little grey sphere: the dots
      // travel and the fibres follow. It goes in as an extra term on
      // the orientation of S^2, so it costs nothing and stays inside
      // the generator's own parametrisation.
      sphereEuler: [TILT[0] + state.spin * Math.PI / 180, TILT[1], TILT[2]],
      nLat: state.nLat,
      nFiber: state.nFiber,
      samples: 200,
      P: state.P, Q: state.Q,
      latMin: single ? state.ring : lo,
      latMax: single ? state.ring : hi,
      chirality: state.chirality,
      includeAxis: state.includeAxis,
      s3Rot: state.flow,
      extra: { turns: 4.0, curl_lobes: 6, curl_amp: 22.0 },
    };
  }

  // hand-picked points are not covered by sphereEuler, so they are
  // turned here by the same angle, about the same axis
  function spun(points) {
    if (!state.spin) return points;
    const a = state.spin * Math.PI / 180;
    const c = Math.cos(a), si = Math.sin(a);
    return points.map(([x, y, z]) => [x, c * y - si * z, si * y + c * z]);
  }

  function pairPoints() {
    const R = rotMatrix(...TILT);
    return [apply3(R, normalize3([0.35, 0.15, 0.55])),
            apply3(R, normalize3([-0.45, 0.5, -0.25]))];
  }

  /** The fit that covers every frame of the current animation: build
   *  the extremes, take the box that contains them all, and scale to
   *  that. Costs a handful of builds, once, when something structural
   *  changes. */
  function computeHeldFit() {
    const probes = [];
    const save = { ring: state.ring, flow: state.flow, spin: state.spin };
    const a = state.anim;
    if (a === 'LATITUDE' || a === 'LATFLOW') {
      for (const r of [RING_MIN, 90, RING_MAX]) probes.push({ ring: r });
    }
    if (a === 'FLOW' || a === 'LATFLOW') {
      for (const f of [0, 90, 180, 270]) probes.push({ flow: f });
    }
    if (a === 'SPIN') {
      for (const sp of [0, 90, 180, 270]) probes.push({ spin: sp });
    }
    if (!probes.length) return null;

    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    let worst = 0;
    for (const probe of probes) {
      Object.assign(state, save, probe);
      const args = buildArgs();
      if (args._custom) { args.points = args._custom; delete args._custom; }
      // raw, with no fit of its own, so the boxes can be compared
      const b = buildFibers({ ...args, samples: 96,
                              fitCentre: [0, 0, 0], fitScale: 1 });
      for (const f of b.fibers) {
        for (let i = 0; i < f.length; i += 3) {
          for (let k = 0; k < 3; k++) {
            if (f[i + k] < lo[k]) lo[k] = f[i + k];
            if (f[i + k] > hi[k]) hi[k] = f[i + k];
          }
        }
      }
    }
    Object.assign(state, save);
    if (!Number.isFinite(lo[0])) return null;
    const centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]),
                    0.5 * (lo[2] + hi[2])];
    for (let k = 0; k < 3; k++) worst = Math.max(worst, hi[k] - lo[k]);
    // the same convention as the generator's own fit: a radius of one
    const scale = worst > 1e-9 ? 2 / worst : 1;
    return { centre, scale };
  }

  function rebuild() {
    // A new shape, or a different thing being animated, needs a new
    // fit held across it. The ring is deliberately not part of this
    // key: a latitude sweep must keep the fit it started with.
    const fitKey = [state.preset, state.nFiber, state.nLat, state.spread,
                    state.P, state.Q, state.chirality, state.includeAxis,
                    state.halo, state.haloN, state.custom.length,
                    state.anim].join('|');
    if (fitKey !== lastFitKey) {
      lastFitKey = fitKey;
      heldFit = computeHeldFit();
    }
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
    if (heldFit) {
      args.fitCentre = heldFit.centre;
      args.fitScale = heldFit.scale;
    }
    const stats = view.build({
      ...args,
      samples: state.playing ? 120 : (args.samples ?? 200),
      sides: state.playing ? 6 : 8,
      palette: state.palette,
      radius: state.radius,
    });
    const bases = view.built.bases;
    // Reframe when the shape of the picture changes, but not while a
    // parameter is merely animating: rescaling every frame would make
    // the whole scene breathe.
    const key = [state.preset, state.nFiber, state.nLat, state.spread,
                 state.P, state.Q, state.chirality, state.includeAxis,
                 state.halo, state.haloN, state.custom.length,
                 state.anim].join('|');
    if (key !== lastFrameKey) {
      lastFrameKey = key;
      // With a held fit the picture is scaled so that the LARGEST
      // frame of the animation spans two units, so the camera frames
      // for that rather than for whatever happens to be on screen
      // now -- otherwise a sweep that starts small would grow out of
      // the view.
      view.frame(heldFit ? 1.0 : stats.extent);
    }
    sphere.show(bases, state.palette);

    // A fibre over a base point near the south pole projects to a
    // circle too big for the picture, and is kept as the arc that
    // fits. Saying how many are in that state answers the obvious
    // question about the curves that sail off and never come back.
    const arcs = view.built.closed.filter((c) => !c).length;
    const n = (x) => x.toLocaleString();
    readout.textContent =
      `${n(stats.fibers)} fibre${stats.fibers === 1 ? '' : 's'}`
      + (arcs ? ` · ${arcs} too large to fit, drawn as arcs` : '')
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
    // the animated parameters, so a test can see the motion rather
    // than having to read it off a picture
    document.body.dataset.anim =
      `${state.anim} ring=${state.ring.toFixed(1)} flow=${state.flow.toFixed(1)} `
      + `spin=${state.spin.toFixed(1)}`;
    // Three attributes exist for the tests, which cannot read a
    // picture: what is being animated and where it has got to, how
    // far the picture reaches and how far back the camera stands,
    // and how many fibres are drawn.
    document.body.dataset.view =
      `extent=${(stats.extent ?? -1).toFixed(2)} cam=${view.camera.position.length().toFixed(2)}`;
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
  slider('#spread', 'spread', (v) => `±${v}°`);
  slider('#halo', 'halo', (v) => (v ? `${v}°` : 'off'));
  slider('#halo-n', 'haloN', (v) => String(v));
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

  // The base sphere turns on a drag and drops a fibre on a click;
  // Paint makes a drag a stroke of fibres instead, and Home puts the
  // sphere back where it started.
  const paintBtn = $('#paint');
  paintBtn.addEventListener('click', () => {
    sphere.paintMode = !sphere.paintMode;
    paintBtn.classList.toggle('on', sphere.paintMode);
    paintBtn.setAttribute('aria-pressed', sphere.paintMode ? 'true' : 'false');
    $('#base-sphere').style.cursor = sphere.paintMode ? 'crosshair' : 'grab';
  });
  $('#sphere-home').addEventListener('click', () => sphere.home());

  const playBtn = $('#play');
  const animSel = $('#anim');
  const speedSel = $('#speed');
  animSel.value = state.anim;
  animSel.addEventListener('change', () => {
    state.anim = animSel.value;
    // Sweeping the latitude only means something when the base points
    // are a ring of latitude, so asking for it picks a preset that is
    // one rather than quietly doing nothing.
    if ((state.anim === 'LATITUDE' || state.anim === 'LATFLOW')
        && !RING_LIKE.has(state.preset)) {
      state.preset = 'FLOWER';
      presetSel.value = 'FLOWER';
      syncFields();
      rebuild();
    }
  });
  speedSel.addEventListener('change', () => { state.speed = Number(speedSel.value); });
  function setPlaying(on) {
    state.playing = on;
    playBtn.textContent = on ? 'Pause' : 'Play';
    playBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
  playBtn.addEventListener('click', () => setPlaying(!state.playing));

  view.onFrame = (dt) => {
    if (!state.playing) return;
    const k = state.speed * dt;
    const a = state.anim;
    if (a === 'FLOW' || a === 'LATFLOW') {
      state.flow = (state.flow + 18 * k) % 360;
      flowEl.value = String(Math.round(state.flow));
      $('#flow-out').textContent = `${Math.round(state.flow)}°`;
    }
    if (a === 'SPIN') {
      state.spin = (state.spin + 22 * k) % 360;
    }
    if (a === 'LATITUDE' || a === 'LATFLOW') {
      // A ring cannot pass through a pole -- at the pole the fibre is
      // the axis and there is nothing to see -- so the sweep turns
      // round just short of each one and comes back.
      const step = 26 * k * (state.sweepUp ? 1 : -1);
      let r = state.ring + step;
      if (r >= RING_MAX) { r = RING_MAX; state.sweepUp = false; }
      if (r <= RING_MIN) { r = RING_MIN; state.sweepUp = true; }
      state.ring = r;
      $('#ring').value = String(Math.round(r));
      $('#ring-out').textContent = `${Math.round(r)}°`;
    }
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
    $('#spread-field').hidden = !(p === 'LATITUDES' || p === 'CAP' || p === 'LOXODROME');
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

  // A way for the tests to step the animation by hand. Headless
  // browsers throttle requestAnimationFrame to a couple of frames a
  // second, which is far too coarse to tell a working animation from
  // a dead one, so the clock can be supplied from outside.
  window.hopfTick = (dt, n = 1) => {
    for (let i = 0; i < n; i++) view.onFrame(dt);
    return document.body.dataset.anim;
  };

  rebuild();
  document.body.dataset.ready = '1';
}

main();
