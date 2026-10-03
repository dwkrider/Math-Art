// Parity: web/js/scherk-math.js against math_art/minsurf/scherk.py.
//
//     node tests/web/test_scherk.mjs [reference.json]
//
// With no argument it runs tools/scherk_reference.py itself (set
// PYTHON to choose the interpreter), so the reference is the engine as
// it stands on this checkout -- a change to the mathematics on master
// fails this until the port follows.
//
// What is compared is the MID-SURFACE: the grids the engine hands back
// from generate_sculpture(return_grids=True), point by point, in the
// same order, including which rows are null. A null row is a level
// whose hole the flange has cut open, and getting that wrong changes
// the sculpture's topology rather than its shape, so it is checked as
// a discrete fact, not a numerical one.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

const HERE = new URL('.', import.meta.url);
const ROOT = new URL('../../', HERE);
const S = await import(new URL('web/js/scherk-math.js', ROOT).href);

let ref;
if (process.argv[2]) {
  ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
} else {
  const py = process.env.PYTHON || 'python';
  const script = new URL('tools/scherk_reference.py', ROOT).pathname
    .replace(/^\/([A-Za-z]:)/, '$1');
  const r = spawnSync(py, [script], { encoding: 'utf8', maxBuffer: 512 * 1024 * 1024 });
  if (r.status !== 0) {
    console.log('  FAIL could not run tools/scherk_reference.py: ' + (r.stderr || r.error));
    console.log('\nRESULT: 1 FAILURE(S)');
    process.exit(1);
  }
  ref = JSON.parse(r.stdout);
}

const TOL = 1e-9;
let fail = 0;
const failures = [];
const note = (m) => { fail++; if (failures.length < 20) failures.push(m); };
let worst = 0;

// ---- the constant the proportions rest on
if (Math.abs(S.XY_SCALE - ref.xy_scale) > 1e-15) {
  note(`XY_SCALE ${S.XY_SCALE} vs ${ref.xy_scale}`);
}

// ---- the presets, by name and by value
{
  const mine = Object.keys(S.PRESETS).sort();
  const theirs = Object.keys(ref.presets).sort();
  if (mine.join(',') !== theirs.join(',')) {
    note(`presets differ: ${mine.length} here, ${theirs.length} there`);
  } else {
    let bad = 0;
    for (const k of mine) {
      const [label, kw] = ref.presets[k];
      if (S.PRESETS[k][0] !== label) { note(`preset ${k}: name "${S.PRESETS[k][0]}" vs "${label}"`); bad++; continue; }
      const got = S.preset(k);
      // the reference lists only what the preset overrides; the rest
      // comes from the shared defaults, which is the point of checking
      const map = { rim_bulge: 'rimBulge', rim_round: 'rimRound',
                    scale_x: 'scaleX', scale_y: 'scaleY', scale_z: 'scaleZ',
                    global_scale: 'globalScale' };
      for (const [pk, pv] of Object.entries(kw)) {
        const key = map[pk] || pk;
        if (Math.abs(got[key] - pv) > 1e-12) {
          note(`preset ${k}: ${pk} ${got[key]} vs ${pv}`);
          bad++;
          break;
        }
      }
    }
    if (!bad) console.log(`  ok   ${mine.length} presets, names and values`);
  }
}

// ---- the closure rule
{
  let bad = 0;
  for (const c of ref.closes) {
    const got = S.ringCloses(S.params({ branches: c.branches, storeys: c.storeys,
                                        twist: c.twist, warp: c.warp }));
    if (got !== c.closes) {
      note(`ringCloses b=${c.branches} S=${c.storeys} tw=${c.twist} warp=${c.warp}: ${got} vs ${c.closes}`);
      bad++;
    }
  }
  if (!bad) console.log(`  ok   the ring-closing rule agrees on ${ref.closes.length} combinations`);
}

// ---- the mid-surface itself
const KW = { rim_bulge: 'rimBulge', rim_round: 'rimRound', scale_x: 'scaleX',
             scale_y: 'scaleY', scale_z: 'scaleZ', global_scale: 'globalScale' };
for (const c of ref.cases) {
  const over = {};
  for (const [k, v] of Object.entries(c.kwargs)) over[KW[k] || k] = v;
  const p = S.params(over);
  const got = S.grids(p);
  if (got.R !== c.R || got.m !== c.m) {
    note(`${c.label}: grid ${got.R}x${got.m} vs ${c.R}x${c.m}`);
    continue;
  }
  if (got.closes !== c.closes) {
    note(`${c.label}: closes ${got.closes} vs ${c.closes}`);
  }
  const keys = Object.keys(c.grids);
  if (got.grids.size !== keys.length) {
    note(`${c.label}: ${got.grids.size} patches vs ${keys.length}`);
    continue;
  }
  let bad = false;
  let holes = 0;
  for (const key of keys) {
    const mine = got.grids.get(key);
    const theirs = c.grids[key];
    if (!mine) { note(`${c.label}: patch ${key} missing`); bad = true; break; }
    for (let i = 0; i < theirs.length && !bad; i++) {
      const rowT = theirs[i], rowM = mine[i];
      if ((rowT === null) !== (rowM === null)) {
        // a cut-open hole in one and not the other is a different solid
        note(`${c.label}: patch ${key} row ${i} ${rowM === null ? 'is' : 'is not'} `
             + `a cut hole, ${rowT === null ? 'should be' : 'should not be'}`);
        bad = true;
        break;
      }
      if (rowT === null) { holes++; continue; }
      for (let k = 0; k < rowT.length; k++) {
        for (let a = 0; a < 3; a++) {
          const e = Math.abs(rowM[k][a] - rowT[k][a]);
          if (e > worst) worst = e;
          if (e > TOL) {
            note(`${c.label}: patch ${key} row ${i} point ${k} differs by ${e.toExponential(2)}`);
            bad = true;
            break;
          }
        }
        if (bad) break;
      }
    }
    if (bad) break;
  }
  if (bad) continue;
  // the fit the generator would apply
  const all = [];
  for (const rows of got.grids.values()) {
    for (const row of rows) if (row) for (const pt of row) all.push(pt);
  }
  const fit = S.fitTransform(all, p.globalScale);
  const fe = Math.max(...fit.centre.map((v, i) => Math.abs(v - c.fit.centre[i])),
                      Math.abs(fit.factor - c.fit.factor));
  if (fe > TOL) note(`${c.label}: fit differs by ${fe.toExponential(2)}`);
  else {
    console.log(`  ok   ${c.label}: ${keys.length} patches, ${c.R + 1}x${c.m}`
                + (holes ? `, ${holes} cut rows` : ''));
  }
}

// ---- the mesh the page actually draws is built from those grids
{
  const mesh = S.surfaceMesh(S.preset('HEX'));
  const n = mesh.positions.length / 3;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < mesh.positions.length; i++) {
    lo = Math.min(lo, mesh.positions[i]);
    hi = Math.max(hi, mesh.positions[i]);
  }
  const span = hi - lo;
  if (!(n > 0 && mesh.indices.length > 0)) note('the hexagon mesh is empty');
  else if (Math.abs(span - 2) > 1e-6) note(`the hexagon mesh spans ${span}, not 2`);
  else console.log(`  ok   the hyperbolic hexagon meshes: ${n} vertices, `
                   + `${mesh.indices.length / 3} triangles, spanning 2`);
  // every index has to point at a vertex that exists
  let worstIdx = -1;
  for (const i of mesh.indices) if (i >= n) worstIdx = Math.max(worstIdx, i);
  if (worstIdx >= 0) note(`index ${worstIdx} is out of range (${n} vertices)`);
}

// ---- the STL the page offers. The surface is an open sheet, so the
// export has to wall it; a file that came out unthickened would be
// one no slicer could print, and the page would be lying about it.
{
  const { buildBinarySTLFromMesh } = await import(new URL('web/js/stl.js', ROOT).href);
  for (const name of ['HEX', 'TOWER']) {
    const p = S.preset(name);
    p.detail = 3;
    const mesh = S.surfaceMesh(p);
    const built = buildBinarySTLFromMesh(mesh.positions, mesh.indices,
                                         { sizeMM: 150, thicknessMM: 2, name });
    if (!built) { note(`STL ${name}: nothing built`); continue; }
    const buf = Buffer.from(await built.blob.arrayBuffer());
    const count = buf.readUInt32LE(80);
    const header = buf.subarray(0, 80).toString('latin1');
    if (buf.length !== 84 + count * 50) {
      note(`STL ${name}: ${buf.length} bytes for ${count} triangles`);
    } else if (header.trimStart().startsWith('solid')) {
      // a binary file whose header starts with "solid" is read as ASCII
      note(`STL ${name}: header starts with "solid"`);
    } else if (!built.thickened) {
      note(`STL ${name}: an open sheet came out unthickened`);
    } else if (Math.abs(Math.max(...built.mm) - 150) > 3) {
      note(`STL ${name}: longest side ${Math.max(...built.mm).toFixed(1)} mm, wanted 150`);
    } else {
      console.log(`  ok   STL ${name}: ${count.toLocaleString()} triangles, walled, `
                  + `${Math.max(...built.mm).toFixed(0)} mm`);
    }
  }
}

console.log(`  worst difference: ${worst.toExponential(2)} (tolerance ${TOL})`);
for (const f of failures) console.log('  FAIL ' + f);
if (fail > failures.length) console.log(`  ... and ${fail - failures.length} more`);
console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
