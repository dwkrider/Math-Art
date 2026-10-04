// Entry point for the triply-periodic minimal surface module.
//
// The controls are the generator's: which surface, how many cells, how
// finely to sample, and the level offset that sweeps each field's
// companion family. Export reuses the site's STL path, as the
// Scherk-Collins page does -- a nodal surface is a sheet with no
// thickness, so it is walled on the way out.

import { TPMS, prettyLabel, block } from './tpms-math.js';
import { TpmsView } from './tpms-view.js';
import { buildBinarySTLFromMesh, downloadSTL } from './stl.js';

const $ = (sel) => document.querySelector(sel);

function main() {
  const view = new TpmsView($('#stage'));
  const state = { kind: 'G', cells: 2, res: 64, offset: 0, wireframe: false,
                  thickness: 0, clip: 0, rim: 0 };
  let lastFrameKey = null;

  const readout = $('#readout');
  const kindSel = $('#kind');
  const note = $('#surface-note');

  // the singly-periodic Scherk tower in the engine's table is not a
  // triply-periodic surface and has no unit cell to tile, so it is not
  // offered here; the Scherk-Collins module draws it properly
  for (const kind of Object.keys(TPMS)) {
    if (!TPMS[kind][2]) continue;
    const o = document.createElement('option');
    o.value = kind;
    o.textContent = prettyLabel(kind).replace(' (nodal approximation)', '');
    if (kind === state.kind) o.selected = true;
    kindSel.append(o);
  }

  function rebuild() {
    let stats;
    try {
      stats = view.build({ ...state });
    } catch (err) {
      readout.textContent = `Could not build: ${err.message}`;
      return;
    }
    // reframe when the block changes size, not on every rebuild, so
    // the reader's own zoom survives a change of surface
    const key = `${state.cells}|${state.clip}`;
    if (key !== lastFrameKey) { lastFrameKey = key; view.frame(stats.extent); }
    const n = (x) => x.toLocaleString();
    readout.textContent =
      `${state.cells}×${state.cells}×${state.cells} cells · `
      + `${n(stats.vertices)} vertices · ${n(stats.triangles)} triangles · `
      + (stats.solid ? 'a solid' : 'a sheet with no thickness')
      + (stats.rimLoops
        ? ` · ${stats.rimLoops} rim loop${stats.rimLoops === 1 ? '' : 's'}`
        : '')
      + ` · built in ${stats.buildMs.toFixed(0)} ms`;
    if (stats.clippedAway) {
      readout.textContent = 'The ball is too small to touch the surface — '
        + 'raise Clip to a ball, or switch it off.';
    }
    const label = prettyLabel(state.kind);
    note.textContent = label.includes('nodal approximation')
      ? `${label.replace(' (nodal approximation)', '')}: drawn from its `
        + 'published nodal formula, which is a close approximation to the '
        + 'minimal surface rather than the surface itself.'
      : `${label}: drawn from its nodal formula.`;
    document.body.dataset.ready = '1';
    document.body.dataset.stats =
      `kind=${state.kind} tris=${stats.triangles} ms=${stats.buildMs.toFixed(0)}`;
  }

  kindSel.addEventListener('change', () => { state.kind = kindSel.value; rebuild(); });

  for (const [sel, key, fmt] of [
    ['#cells', 'cells', (v) => `${v}×${v}×${v}`],
    ['#res', 'res', (v) => String(v)],
    ['#offset', 'offset', (v) => (v ? v.toFixed(2) : '0 (canonical)')],
    ['#thickness', 'thickness', (v) => (v ? v.toFixed(3) : 'none (a sheet)')],
    ['#clip', 'clip', (v) => (v ? v.toFixed(2) : 'off')],
    ['#rim', 'rim', (v) => (v ? v.toFixed(3) : 'off')],
  ]) {
    const el = $(sel);
    const out = $(`${sel}-out`);
    el.value = String(state[key]);
    if (out) out.textContent = fmt(state[key]);
    el.addEventListener('input', () => {
      state[key] = Number(el.value);
      if (out) out.textContent = fmt(state[key]);
      rebuild();
    });
  }

  $('#wire').addEventListener('change', (e) => {
    state.wireframe = e.target.checked;
    rebuild();
  });
  $('#spin').addEventListener('click', (e) => {
    view.spin = view.spin ? 0 : 0.25;
    e.target.textContent = view.spin ? 'Stop' : 'Spin';
    e.target.setAttribute('aria-pressed', view.spin ? 'true' : 'false');
  });
  $('#reset-view').addEventListener('click', () => view.resetView());

  // -- export --------------------------------------------------------
  const sizeEl = $('#size');
  const wallEl = $('#wall');
  const exportNote = $('#export-note');
  const show = () => {
    $('#size-out').textContent = `${sizeEl.value} mm`;
    $('#wall-out').textContent = `${Number(wallEl.value).toFixed(1)} mm`;
  };
  sizeEl.addEventListener('input', show);
  wallEl.addEventListener('input', show);
  show();

  $('#export').addEventListener('click', () => {
    const btn = $('#export');
    btn.disabled = true;
    exportNote.textContent = 'Building…';
    setTimeout(() => {
      try {
        const mesh = view.lastGeometry;
        const built = buildBinarySTLFromMesh(mesh.positions, mesh.indices, {
          sizeMM: Number(sizeEl.value),
          thicknessMM: Number(wallEl.value),
          name: prettyLabel(state.kind),
        });
        if (!built) { exportNote.textContent = 'Nothing to export.'; return; }
        downloadSTL(built, state.kind.toLowerCase());
        const mm = built.mm.map((v) => v.toFixed(0)).join(' × ');
        exportNote.textContent =
          `${built.triangles.toLocaleString()} triangles, ${mm} mm`
          + (built.thickened
            ? `, walled at ${Number(wallEl.value).toFixed(1)} mm.`
            : ', already closed.');
      } catch (err) {
        exportNote.textContent = `Export failed: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    }, 20);
  });

  rebuild();
}

main();
