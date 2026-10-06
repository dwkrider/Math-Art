// The Seifert surface page: UI wiring only.
//
// Three layers meet here and nowhere else. `seifert-math.js` is the
// COMPUTATION -- pure arithmetic, no DOM, no three.js, which is what
// lets the parity test run it in node against the Python engine.
// `lib/stage.js` is the RENDERING, and knows nothing about knots.
// `lib/controls.js` is the UI, and knows no mathematics. This file is
// the thin seam that composes them, and it is deliberately short: if it
// starts growing geometry, that geometry belongs in the layer below.

import { buildSurface, tubeAlong, torusKnot } from './seifert-math.js';
import { Stage } from './lib/stage.js';
import { mountControls } from './lib/controls.js';
import { buildBinarySTLFromMesh, downloadSTL } from './stl.js';

const $ = (sel) => document.querySelector(sel);

// Braid words for the classics. A knot's braid word is not unique, so
// these are the usual presentations rather than the only ones.
// Every label here was checked by building it rather than recalled:
// the component count and genus printed by the page are what name it.
// A knot has many braid words, which is why the trefoil appears twice.
const PRESETS = [
  ['AAA', 'Trefoil — (2,3) torus, genus 1'],
  ['ABAB', 'Trefoil again, on 3 strands'],
  ['AbAb', 'Figure-eight — genus 1'],
  ['A5', 'Cinquefoil — (2,5) torus, genus 2'],
  ['A7', '(2,7) torus knot — genus 3'],
  ['ABABABAB', '(3,4) torus knot — genus 3'],
  ['AA', 'Hopf link — 2 components'],
  ['ABABAB', '(3,3) torus link — 3 components'],
  ['AbCb', 'A two-component link on 4 strands'],
  ['custom', 'Braid word below…'],
];

const SURFACES = [
  ['SEIFERT', 'Seifert (two-sided)'],
  ['TURNBACK', 'All-turnback state'],
  ['CROSSCAP', 'Smallest one-sided state'],
];

const FRONT = 0x6fb3f2;
const BACK = 0x1d3448;
const KNOT = 0xffb454;

function main() {
  const stage = new Stage($('#stage'), { distance: 5.2, maxDistance: 30 });
  const readout = document.createElement('p');
  readout.className = 'belt-hint';
  readout.id = 'readout';
  const note = document.createElement('p');
  note.className = 'belt-hint';
  note.id = 'surface-note';

  let lastFrameKey = null;
  let lastBuild = null;

  const panel = mountControls($('#controls'), [
    { type: 'row', controls: [
      { id: 'preset', type: 'select', label: 'Knot', options: PRESETS,
        value: 'AAA', title: 'A classic knot or link' },
      { id: 'surface', type: 'select', label: 'Surface', options: SURFACES,
        value: 'SEIFERT', title: 'Which spanning surface to build' },
    ] },
    { type: 'row', controls: [
      { id: 'word', type: 'text', label: 'Braid word', value: 'AAA',
        placeholder: 'e.g. AAA, AbAb, A5',
        title: 'Braid word in letter notation' },
    ] },
    { type: 'row', controls: [
      { type: 'group', controls: [
        { id: 'spin', type: 'button', label: 'Spin',
          onClick: (_v, button) => {
            const on = stage.toggleSpin();
            button.textContent = on ? 'Stop' : 'Spin';
            button.setAttribute('aria-pressed', on ? 'true' : 'false');
          } },
        { id: 'reset-view', type: 'button', label: 'Reset view',
          onClick: () => stage.resetView() },
        { id: 'wire', type: 'toggle', label: 'Wireframe', value: false },
      ] },
    ] },
    { type: 'row', controls: [
      { id: 'levels', type: 'range', label: 'Smoothing', min: 0, max: 3,
        step: 1, value: 1, title: 'Catmull-Clark subdivision levels',
        format: (v) => (v ? `${v} level${v === 1 ? '' : 's'}` : 'none (raw bands)') },
      { id: 'tube', type: 'range', label: 'Knot tube', min: 0, max: 0.06,
        step: 0.002, value: 0.018, title: 'Radius of the tube along the boundary',
        format: (v) => (v ? v.toFixed(3) : 'off') },
      { id: 'bands', type: 'range', label: 'Band width', min: 0.3, max: 1.6,
        step: 0.05, value: 0.85, title: 'Band width as a fraction of the disk radius',
        format: (v) => v.toFixed(2) },
    ] },
    { type: 'row', controls: [
      { id: 'size', type: 'range', label: 'Print size', min: 40, max: 300,
        step: 10, value: 150, title: 'Longest side of the printed object',
        format: (v) => `${v} mm` },
      { id: 'wall', type: 'range', label: 'Wall', min: 0.4, max: 6, step: 0.2,
        value: 1.6, title: 'Wall thickness in millimetres',
        format: (v) => `${v.toFixed(1)} mm` },
      { id: 'export', type: 'button', label: 'Download STL',
        onClick: () => exportSTL() },
    ] },
  ], (id) => {
    if (id === 'preset') {
      const chosen = panel.get('preset');
      if (chosen !== 'custom') panel.set('word', chosen);
    }
    if (id === 'word') panel.set('preset', 'custom');
    rebuild();
  });

  $('#controls').append(readout, note);

  function rebuild() {
    const v = panel.values;
    let out;
    try {
      out = buildSurface({
        word: v.word.trim(),
        surface: v.surface,
        levels: v.levels,
        params: { bandWidth: v.bands },
      });
    } catch (err) {
      readout.textContent = `Could not build: ${err.message}`;
      note.textContent = '';
      document.body.dataset.ready = '1';
      return;
    }
    lastBuild = out;

    const parts = [{
      ...out.surface,
      color: FRONT,
      side: 'double',
      wireframe: v.wire,
      emissive: BACK,
    }];
    if (v.tube > 0) {
      parts.push({ ...tubeAlong(out.loops, v.tube, 10), color: KNOT,
                   side: 'front', roughness: 0.4 });
    }
    stage.show(parts);

    // reframe when the object changes size, not on every rebuild, so a
    // reader's own zoom survives a change of smoothing
    const key = `${v.word}|${v.surface}`;
    if (key !== lastFrameKey) { lastFrameKey = key; stage.frame(out.extent); }

    const s = out.summary;
    const sided = s.orientable
      ? `two-sided, genus ${s.genus}`
      : `one-sided, crosscap number ${s.crosscap}`;
    readout.textContent =
      `${s.word}: ${s.strands} strands, ${s.crossings} crossings · `
      + `${s.disks} disk${s.disks === 1 ? '' : 's'}, ${s.bands} band${s.bands === 1 ? '' : 's'} · `
      + `${s.components} boundary component${s.components === 1 ? '' : 's'} · `
      + `χ = ${s.chi} · ${sided} · `
      + `${(out.surface.indices.length / 3).toLocaleString()} triangles`;
    note.textContent = s.agrees
      ? 'The finished mesh agrees with the combinatorics it was built from.'
      : 'The mesh does NOT match the predicted topology — the geometry is wrong.';
    document.body.dataset.ready = '1';
    document.body.dataset.stats =
      `word=${s.word} chi=${s.chi} genus=${s.genus} tris=${out.surface.indices.length / 3}`;
  }

  function exportSTL() {
    const button = panel.node('export');
    button.disabled = true;
    note.textContent = 'Building…';
    setTimeout(() => {
      try {
        if (!lastBuild) { note.textContent = 'Nothing to export.'; return; }
        const built = buildBinarySTLFromMesh(
          lastBuild.surface.positions, lastBuild.surface.indices, {
            sizeMM: panel.get('size'),
            thicknessMM: panel.get('wall'),
            name: `Seifert surface ${lastBuild.summary.word}`,
          });
        if (!built) { note.textContent = 'Nothing to export.'; return; }
        downloadSTL(built, `seifert-${lastBuild.summary.word.toLowerCase()}`);
        note.textContent =
          `${built.triangles.toLocaleString()} triangles, `
          + `${built.mm.map((x) => x.toFixed(0)).join(' × ')} mm`
          + (built.thickened ? `, walled at ${panel.get('wall').toFixed(1)} mm.`
                             : ', already closed.');
      } catch (err) {
        note.textContent = `Export failed: ${err.message}`;
      } finally {
        button.disabled = false;
      }
    }, 20);
  }

  rebuild();
}

main();
