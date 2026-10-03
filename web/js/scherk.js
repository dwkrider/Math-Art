// Entry point for the Scherk-Collins module page.
//
// The controls are the generator's own, in its own units, so a shape
// found here can be typed into the add-on and come out the same. The
// page adds one thing the add-on does differently: export. The add-on
// builds its own thickened solid with rims; the page draws the
// mid-surface and, on export, offsets it with the site's solidifier,
// which is a faithful shell of the same surface rather than a copy of
// the add-on's mesh.

import { PRESETS, preset, params, ringCloses, surfaceMesh } from './scherk-math.js';
import { ScherkView } from './scherk-view.js';
import { buildBinarySTLFromMesh, downloadSTL } from './stl.js';

const $ = (sel) => document.querySelector(sel);

function main() {
  const view = new ScherkView($('#stage'));

  // The Hyperbolic Hexagon: Brent Collins' sculpture, and the one the
  // add-on renders for its own documentation.
  const state = { ...preset('HEX'), presetName: 'HEX', wireframe: false };

  const readout = $('#readout');
  const closeLine = $('#close-line');
  const presetSel = $('#preset');

  function rebuild() {
    const stats = view.build(state, { wireframe: state.wireframe });
    const n = (x) => x.toLocaleString();
    readout.textContent =
      `${n(stats.patches)} patches · ${n(stats.vertices)} vertices · `
      + `${n(stats.triangles)} triangles · `
      + (stats.solid ? 'a solid' : 'a sheet with no thickness')
      + ` · built in ${stats.buildMs.toFixed(0)} ms`;
    // A solid whose faces cannot all agree which way is out is worth
    // saying out loud: it is a fact about the sculpture, and it is
    // also what a slicer will complain about.
    const sides = $('#sides-line');
    if (stats.solid && !stats.orientable) {
      sides.textContent =
        `This one has no consistent inside and outside: following the `
        + `surface round brings you back on the other side, and `
        + `${stats.misWound.toLocaleString()} edges are left where the two `
        + `faces disagree. It will still slice, but not as a clean solid.`;
      sides.hidden = false;
    } else {
      sides.hidden = true;
    }
    closeLine.textContent = stats.closes
      ? 'The ring closes: the two ends of the tower meet with their '
        + 'saddles aligned, so this is a single closed band.'
      : (state.warp > 0
        ? 'The ring does not close at this twist — the ends meet at a '
          + 'mismatched angle. Sequin’s rule: with a full turn of warp, '
          + 'the twist plus the storeys’ own rotation has to come to a '
          + 'whole number of branch widths.'
        : 'A straight tower, not bent into a ring. Raise Warp to bend it.');
    document.body.dataset.ready = '1';
    document.body.dataset.stats =
      `patches=${stats.patches} tris=${stats.triangles} closes=${stats.closes}`;
  }

  // -- presets -------------------------------------------------------
  for (const [key, [label]] of Object.entries(PRESETS)) {
    const o = document.createElement('option');
    o.value = key;
    o.textContent = label;
    if (key === state.presetName) o.selected = true;
    presetSel.append(o);
  }
  presetSel.addEventListener('change', () => {
    Object.assign(state, preset(presetSel.value));
    state.presetName = presetSel.value;
    syncControls();
    rebuild();
  });

  // -- the generator's parameters ------------------------------------
  const FIELDS = [
    ['#branches', 'branches', (v) => String(v)],
    ['#storeys', 'storeys', (v) => String(v)],
    ['#height', 'height', (v) => v.toFixed(2)],
    ['#thickness', 'thickness', (v) => (v > 0 ? v.toFixed(3) : 'none')],
    ['#flange', 'flange', (v) => v.toFixed(2)],
    ['#twist', 'twist', (v) => `${v}°`],
    ['#azimuth', 'azimuth', (v) => `${v}°`],
    ['#warp', 'warp', (v) => `${v}°`],
    ['#phase', 'phase', (v) => v.toFixed(2)],
    ['#detail', 'detail', (v) => String(v)],
  ];
  for (const [sel, key, fmt] of FIELDS) {
    const el = $(sel);
    const out = $(`${sel}-out`);
    el.addEventListener('input', () => {
      state[key] = Number(el.value);
      if (out) out.textContent = fmt(state[key]);
      // the shape no longer matches the preset it came from
      presetSel.value = '';
      state.presetName = '';
      rebuild();
    });
  }

  function syncControls() {
    for (const [sel, key, fmt] of FIELDS) {
      const el = $(sel);
      const out = $(`${sel}-out`);
      el.value = String(state[key]);
      if (out) out.textContent = fmt(state[key]);
    }
    $('#phase-field').hidden = ringCloses(state);
  }

  $('#wire').addEventListener('change', (e) => {
    state.wireframe = e.target.checked;
    rebuild();
  });
  $('#spin').addEventListener('click', (e) => {
    view.spin = view.spin ? 0 : 0.3;
    e.target.textContent = view.spin ? 'Stop' : 'Spin';
    e.target.setAttribute('aria-pressed', view.spin ? 'true' : 'false');
  });
  $('#reset-view').addEventListener('click', () => view.resetView());

  // -- export --------------------------------------------------------
  // The surface has a boundary and no thickness, which no slicer will
  // print, so the export offsets it into a shell. The wall is in
  // millimetres of the finished object, as it is everywhere else on
  // the site.
  const sizeEl = $('#size');
  const wallEl = $('#wall');
  const exportBtn = $('#export');
  const exportNote = $('#export-note');
  const showExport = () => {
    $('#size-out').textContent = `${sizeEl.value} mm`;
    $('#wall-out').textContent = `${Number(wallEl.value).toFixed(1)} mm`;
  };
  sizeEl.addEventListener('input', showExport);
  wallEl.addEventListener('input', showExport);
  showExport();

  exportBtn.addEventListener('click', () => {
    exportBtn.disabled = true;
    exportNote.textContent = 'Building…';
    // let the button repaint before the work starts
    setTimeout(() => {
      try {
        // the geometry the view just built, thickness and all, so the
        // file cannot describe a different sculpture from the picture
        const mesh = view.lastGeometry || surfaceMesh(state);
        const built = buildBinarySTLFromMesh(mesh.positions, mesh.indices, {
          sizeMM: Number(sizeEl.value),
          thicknessMM: Number(wallEl.value),
          name: PRESETS[state.presetName] ? PRESETS[state.presetName][0]
                                          : 'scherk-collins',
        });
        if (!built) {
          exportNote.textContent = 'Nothing to export.';
          return;
        }
        const slug = (state.presetName || 'scherk-collins').toLowerCase();
        downloadSTL(built, slug);
        const mm = built.mm.map((v) => v.toFixed(0)).join(' × ');
        exportNote.textContent =
          `${built.triangles.toLocaleString()} triangles, ${mm} mm`
          + (built.thickened
            ? `, walled at ${Number(wallEl.value).toFixed(1)} mm`
            : ', solid already — no wall needed')
          + (built.audit && built.audit.nonManifold
            ? `. ${built.audit.nonManifold} non-manifold edges: a slicer may complain.`
            : '.');
      } catch (err) {
        exportNote.textContent = `Export failed: ${err.message}`;
      } finally {
        exportBtn.disabled = false;
      }
    }, 20);
  });

  syncControls();
  rebuild();
}

main();
