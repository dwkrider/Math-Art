// Entry point for the Belt Trick module page.
//
// The page is a transport over one parameter: the turn, from 0 to 720
// degrees. Everything drawn at a turn is computed afresh from it --
// nothing depends on what was drawn before -- which is what makes the
// cycle exactly periodic and why scrubbing backwards is no different
// from playing forwards.

import { prepare, allBelts, SOLID_NAMES } from './belt-math.js';
import { BeltView } from './belt-view.js';

const $ = (sel) => document.querySelector(sel);

const SOLID_LABELS = {
  TWO: 'Two belts (cube)',
  TETRA: 'Tetrahedron — 4 belts',
  CUBE: 'Cube — 6 belts',
  OCTA: 'Octahedron — 8 belts',
  DODECA: 'Dodecahedron — 12 belts',
  ICOSA: 'Icosahedron — 20 belts',
  SNUB: 'Snub cube — 38 belts',
  GEO: 'Geodesic sphere — 80 belts',
};

// Speed as a multiplier of one full cycle -- two turns, 720 degrees --
// in twelve seconds at 1x.
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const DEG_PER_SEC = 720 / 12;
const FULL = 720;

function speedLabel(s) {
  return ({ 0.25: '¼×', 0.5: '½×' })[s] || `${s}×`;
}

function main() {
  const view = new BeltView($('#stage'));
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const state = {
    kind: 'CUBE',
    axis: 'Z',
    // The generator's default turn, 300 degrees, is a good still: the
    // belts are well into their coil. Moving, start from rest instead.
    turn: reduced ? 300 : 0,
    playing: !reduced,
    dir: 1,
    speed: 1,
    dirty: true,
  };
  let prep = null;

  function rebuild() {
    prep = prepare({ kind: state.kind, spinAxis: state.axis });
    view.setup(prep);
    state.dirty = true;
  }

  // -- the solid and the spin axis --------------------------------
  const solidSel = $('#solid');
  for (const k of SOLID_NAMES) {
    const o = document.createElement('option');
    o.value = k;
    o.textContent = SOLID_LABELS[k];
    if (k === state.kind) o.selected = true;
    solidSel.append(o);
  }
  solidSel.addEventListener('change', () => { state.kind = solidSel.value; rebuild(); });

  for (const b of document.querySelectorAll('#axis-bar button')) {
    b.addEventListener('click', () => {
      for (const o of document.querySelectorAll('#axis-bar button')) o.classList.remove('on');
      b.classList.add('on');
      state.axis = b.dataset.axis;
      rebuild();
    });
  }

  // -- the turn ----------------------------------------------------
  const slider = $('#turn');
  const readout = $('#turn-readout');
  const lesson = $('#lesson');
  let resumeAfterScrub = false;
  slider.addEventListener('input', () => {
    if (state.playing) { resumeAfterScrub = true; setPlaying(false); }
    state.turn = Number(slider.value);
    state.dirty = true;
  });
  slider.addEventListener('change', () => {
    if (resumeAfterScrub) { resumeAfterScrub = false; setPlaying(true); }
  });
  for (const b of document.querySelectorAll('.snap')) {
    b.addEventListener('click', () => {
      setPlaying(false);
      state.turn = Number(b.dataset.turn);
      state.dirty = true;
    });
  }

  // -- the transport -----------------------------------------------
  const playBtn = $('#play');
  const dirBtn = $('#direction');
  const speedSel = $('#speed');
  for (const s of SPEEDS) {
    const o = document.createElement('option');
    o.value = String(s);
    o.textContent = speedLabel(s);
    if (s === state.speed) o.selected = true;
    speedSel.append(o);
  }
  function setPlaying(on) {
    state.playing = on;
    playBtn.textContent = on ? 'Pause' : 'Play';
    playBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
  function setDir(d) {
    state.dir = d;
    dirBtn.textContent = d > 0 ? 'Forward' : 'Backward';
    dirBtn.setAttribute('aria-pressed', d < 0 ? 'true' : 'false');
  }
  playBtn.addEventListener('click', () => setPlaying(!state.playing));
  dirBtn.addEventListener('click', () => setDir(-state.dir));
  speedSel.addEventListener('change', () => { state.speed = Number(speedSel.value); });
  setPlaying(state.playing);
  setDir(state.dir);

  addEventListener('keydown', (e) => {
    if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) && e.key !== ' ') return;
    if (e.key === ' ') { e.preventDefault(); setPlaying(!state.playing); }
    else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      setPlaying(false);
      state.turn += (e.key === 'ArrowRight' ? 5 : -5);
      state.turn = ((state.turn % FULL) + FULL) % FULL;
      state.dirty = true;
    } else if (e.key === '[' || e.key === ']') {
      const i = SPEEDS.indexOf(state.speed) + (e.key === ']' ? 1 : -1);
      if (i >= 0 && i < SPEEDS.length) {
        state.speed = SPEEDS[i];
        speedSel.value = String(state.speed);
      }
    }
  });

  // -- what the reader is looking at -------------------------------
  function describe(turn) {
    const turns = turn / 360;
    readout.textContent = `${turn.toFixed(0)}° · ${turns.toFixed(2)} `
      + `turn${Math.abs(turns - 1) < 0.005 ? '' : 's'}`;
    // Within a degree of the two landmarks, say what is being shown.
    const near = (t) => Math.abs(turn - t) < 1.0;
    let text = '';
    if (near(360)) {
      text = 'One full turn. The solid is back where it started — watch '
           + 'the marked face — but the belts are not: they are tangled '
           + 'round it.';
    } else if (near(720) || near(0)) {
      text = 'Two full turns, or none: the solid and every belt are back '
           + 'where they started. Turning twice undoes itself; turning '
           + 'once cannot.';
    }
    lesson.textContent = text;
    lesson.hidden = !text;
  }

  // -- the frame loop ----------------------------------------------
  let computeMs = 0;
  view.onFrame = (dt) => {
    if (state.playing) {
      let t = state.turn + state.dir * DEG_PER_SEC * state.speed * dt;
      // Wrap, in both directions: the construction is periodic in 720.
      t = ((t % FULL) + FULL) % FULL;
      state.turn = t;
      state.dirty = true;
    }
    if (!state.dirty || !prep) return;
    state.dirty = false;
    const t0 = performance.now();
    const lines = allBelts(prep, 0.5 * state.turn * Math.PI / 180);
    computeMs = 0.9 * computeMs + 0.1 * (performance.now() - t0);
    view.update(lines, state.turn);
    slider.value = String(state.turn);
    describe(state.turn);
    // For tests and for the curious: how long the mathematics takes.
    document.body.dataset.computeMs = computeMs.toFixed(2);
    document.body.dataset.turn = state.turn.toFixed(1);
  };

  rebuild();
  document.body.dataset.ready = '1';
}

main();
