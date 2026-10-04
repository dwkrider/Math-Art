// Parity: web/js/tpms-math.js against math_art/minsurf/tpms.py.
//
//     node tests/web/test_tpms.mjs [reference.json]
//
// With no argument it runs tools/tpms_reference.py itself (set PYTHON
// to choose the interpreter), so the reference is the engine as it
// stands on this checkout -- a change to a nodal formula or to the
// mesher on master fails this until the port follows.
//
// Two things are checked, and the first is the one that matters most:
// the FIELDS, sampled at fixed points, which pin each published
// formula to the digit. A mistyped coefficient there would change the
// surface while leaving a perfectly plausible picture on the page.
//
// Then the MESHES, canonically: both sides march the same tetrahedra
// over the same grid and weld on the same lattice edges, so they find
// the same crossings, but they number their vertices differently (the
// engine compacts through numpy's lexicographic unique, the port in
// the order triangles reach it). Numbering is not geometry, so each
// mesh is reduced to its sorted list of sorted corner triples and the
// two are compared as surfaces.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

const HERE = new URL('.', import.meta.url);
const ROOT = new URL('../../', HERE);
const T = await import(new URL('web/js/tpms-math.js', ROOT).href);

let ref;
if (process.argv[2]) {
  ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
} else {
  const py = process.env.PYTHON || 'python';
  const script = new URL('tools/tpms_reference.py', ROOT).pathname
    .replace(/^\/([A-Za-z]:)/, '$1');
  const r = spawnSync(py, [script], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 1024 });
  if (r.status !== 0) {
    console.log('  FAIL could not run tools/tpms_reference.py: ' + (r.stderr || r.error));
    console.log('\nRESULT: 1 FAILURE(S)');
    process.exit(1);
  }
  ref = JSON.parse(r.stdout);
}

let fail = 0;
const failures = [];
const note = (m) => { fail++; if (failures.length < 20) failures.push(m); };
let worstField = 0, worstVert = 0;

const SAMPLES = [[0.3, -1.1, 2.4], [1.0, 1.0, 1.0], [-2.2, 0.7, 0.15],
                 [0.0, 0.0, 0.0], [3.0, -3.0, 1.5], [0.77, 2.71, -1.41]];

// ---- the published formulas, and the names on them
{
  const mine = Object.keys(T.TPMS).sort();
  const theirs = Object.keys(ref.fields).sort();
  if (mine.join(',') !== theirs.join(',')) {
    note(`surfaces differ: ${mine.length} here, ${theirs.length} there`
         + ` (${mine.filter((k) => !theirs.includes(k)).join(',') || 'none extra'})`);
  }
  let ok = 0;
  for (const kind of theirs) {
    if (!T.TPMS[kind]) continue;
    const [label, field, triply] = T.TPMS[kind];
    const [wantLabel, wantTriply] = ref.labels[kind];
    if (label !== wantLabel) note(`${kind}: name "${label}" vs "${wantLabel}"`);
    if (triply !== wantTriply) note(`${kind}: triply ${triply} vs ${wantTriply}`);
    let bad = false;
    SAMPLES.forEach(([x, y, z], i) => {
      const e = Math.abs(field(x, y, z) - ref.fields[kind][i]);
      if (e > worstField) worstField = e;
      if (e > 1e-12) {
        note(`${kind}: field at (${x}, ${y}, ${z}) is ${field(x, y, z)}, `
             + `engine says ${ref.fields[kind][i]}`);
        bad = true;
      }
    });
    if (!bad) ok++;
  }
  console.log(`  ok   ${ok} of ${theirs.length} nodal formulas match to 1e-12`);
}

// ---- the meshes
function canonical(positions, indices, nd = 6) {
  const r = (v) => {
    const s = Math.round(v * 10 ** nd) / 10 ** nd;
    return Object.is(s, -0) ? 0 : s;
  };
  const out = [];
  for (let t = 0; t < indices.length; t += 3) {
    const corners = [];
    for (let c = 0; c < 3; c++) {
      const i = indices[t + c] * 3;
      corners.push([r(positions[i]), r(positions[i + 1]), r(positions[i + 2])]);
    }
    // rotate to start at the smallest corner, which keeps the winding
    let k = 0;
    for (let i = 1; i < 3; i++) {
      const a = corners[i], b = corners[k];
      if (a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1]
          || (a[1] === b[1] && a[2] < b[2])))) k = i;
    }
    out.push([corners[k], corners[(k + 1) % 3], corners[(k + 2) % 3]]);
  }
  out.sort((A, B) => {
    for (let i = 0; i < 3; i++) {
      for (let k = 0; k < 3; k++) {
        if (A[i][k] !== B[i][k]) return A[i][k] - B[i][k];
      }
    }
    return 0;
  });
  return out;
}

function compare(label, got, want, wantTris, wantArea, wantVerts) {
  // The same triangles can sit on twice as many vertices if a weld
  // was missed, and the surface is split along that seam -- so the
  // count is worth comparing. It is the count of vertices a triangle
  // USES: the engine keeps the odd point that only a degenerate
  // triangle referenced before it dropped the sliver, and the port
  // compacts those away. Two orphans in the shipped set, on
  // Fischer-Koch S and the complementary gyroid.
  if (wantVerts !== undefined) {
    const used = new Set(got.indices).size;
    if (used !== wantVerts) {
      note(`${label}: ${used} vertices used vs ${wantVerts}`);
      return;
    }
  }
  const mine = canonical(got.positions, got.indices);
  if (mine.length !== wantTris) {
    note(`${label}: ${mine.length} triangles vs ${wantTris}`);
    return;
  }
  let bad = 0;
  if (!want) {
    // counts and area only: the formula behind this one is already
    // pinned exactly by the field comparison above
    let a = 0;
    for (let t = 0; t < got.indices.length; t += 3) {
      const p = [0, 1, 2].map((c) => {
        const i = got.indices[t + c] * 3;
        return [got.positions[i], got.positions[i + 1], got.positions[i + 2]];
      });
      const u = [0, 1, 2].map((k) => p[1][k] - p[0][k]);
      const v = [0, 1, 2].map((k) => p[2][k] - p[0][k]);
      a += 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1],
                            u[2] * v[0] - u[0] * v[2],
                            u[0] * v[1] - u[1] * v[0]);
    }
    if (Math.abs(a - wantArea) > 1e-6 * Math.max(1, wantArea)) {
      note(`${label}: area ${a.toFixed(6)} vs ${wantArea.toFixed(6)}`);
    } else {
      console.log(`  ok   ${label}: ${wantTris} triangles, area ${a.toFixed(4)}`);
    }
    return;
  }
  for (let i = 0; i < mine.length && bad < 1; i++) {
    for (let c = 0; c < 3; c++) {
      for (let k = 0; k < 3; k++) {
        const e = Math.abs(mine[i][c][k] - want[i][c][k]);
        if (e > worstVert) worstVert = e;
        if (e > 1e-5) {
          note(`${label}: triangle ${i} corner ${c} differs by ${e.toExponential(2)}`);
          bad++;
        }
      }
    }
  }
  if (bad) return;
  // the area, as a second opinion that does not depend on the ordering
  let a = 0;
  for (let t = 0; t < got.indices.length; t += 3) {
    const p = [0, 1, 2].map((c) => {
      const i = got.indices[t + c] * 3;
      return [got.positions[i], got.positions[i + 1], got.positions[i + 2]];
    });
    const u = [0, 1, 2].map((k) => p[1][k] - p[0][k]);
    const v = [0, 1, 2].map((k) => p[2][k] - p[0][k]);
    a += 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1],
                          u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0]);
  }
  if (Math.abs(a - wantArea) > 1e-6 * Math.max(1, wantArea)) {
    note(`${label}: area ${a.toFixed(6)} vs ${wantArea.toFixed(6)}`);
    return;
  }
  console.log(`  ok   ${label}: ${wantTris} triangles, area ${a.toFixed(4)}`);
}

for (const c of ref.cases) {
  const got = T.block(c.kind, c.cells, c.res, 2.0);
  const cells = Array.isArray(c.cells) ? c.cells.join('x') : `${c.cells}^3`;
  compare(`${c.kind} ${cells} at ${c.res}`, got, c.canonical, c.ntris,
          c.area, c.nused);
}

for (const c of ref.offsets) {
  const got = T.block(c.kind, 1, c.res, 2.0, c.offset);
  compare(`${c.kind} offset ${c.offset}`, got, c.canonical, c.ntris,
          c.area, c.nused);
}

// ---- clipping to a ball, and the rim it opens
for (const c of ref.clips) {
  const m = T.block(c.kind, c.cells, c.res, 2.0);
  // the radius is a fraction of the block's own half-extent, so it
  // keeps its meaning when the cell count changes
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < m.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (m.positions[i + k] < lo[k]) lo[k] = m.positions[i + k];
      if (m.positions[i + k] > hi[k]) hi[k] = m.positions[i + k];
    }
  }
  const half = 0.5 * Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const r = c.frac * half;
  if (Math.abs(r - c.radius) > 1e-9) {
    note(`clip ${c.kind}: radius ${r} vs ${c.radius}`);
    continue;
  }
  const clipped = T.clipToSphere(m.positions, T.facesOf(m.indices), r);
  const label = `clip ${c.kind} ${c.cells} cell(s) at ${c.frac}`;
  if (clipped.faces.length !== c.nfaces) {
    note(`${label}: ${clipped.faces.length} faces vs ${c.nfaces}`);
    continue;
  }
  const sizes = {};
  for (const f of clipped.faces) sizes[f.length] = (sizes[f.length] || 0) + 1;
  for (const [k, v] of Object.entries(c.sizes)) {
    if (sizes[k] !== v) note(`${label}: ${v} faces of ${k} corners, got ${sizes[k] || 0}`);
  }
  // the cut has to lie ON the sphere, which is the whole point of
  // solving the crossing rather than approximating it
  let maxr = 0;
  for (let i = 0; i < clipped.positions.length; i += 3) {
    maxr = Math.max(maxr, Math.hypot(clipped.positions[i], clipped.positions[i + 1],
                                     clipped.positions[i + 2]));
  }
  if (Math.abs(maxr - c.maxr) > 1e-5) {
    note(`${label}: furthest point ${maxr} vs ${c.maxr}`);
    continue;
  }
  const loops = T.boundaryLoops(clipped.positions, clipped.faces);
  if (loops.length !== c.loops.length) {
    note(`${label}: ${loops.length} rim loops vs ${c.loops.length}`);
    continue;
  }
  // loop ORDER is the walk's own, so compare them as a multiset of
  // (length, closed, perimeter)
  const key = (n, closed, len) => `${n}|${closed}|${len.toFixed(4)}`;
  const want = c.loops.map((l) => key(l.n, l.closed, l.length)).sort();
  const got = loops.map((l) => {
    let len = 0;
    const n = l.points.length;
    const last = l.closed ? n : n - 1;
    for (let i = 0; i < last; i++) {
      const a = l.points[i], b = l.points[(i + 1) % n];
      len += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
    return key(n, l.closed, len);
  }).sort();
  if (want.join(' ') !== got.join(' ')) {
    note(`${label}: rim loops differ
      want ${want.join(', ')}
      got  ${got.join(', ')}`);
    continue;
  }
  console.log(`  ok   ${label}: ${c.nfaces} faces, cut on the sphere, `
              + `${loops.length} rim loop${loops.length === 1 ? '' : 's'}`);
}

// ---- every surface the page offers has to build, and the export has
// to come out walled: a nodal surface is a sheet, and a slicer can do
// nothing with a sheet.
{
  const kinds = Object.keys(T.TPMS).filter((k) => T.TPMS[k][2]);
  let empty = 0;
  for (const k of kinds) {
    const m = T.block(k, 1, 12, 2);
    if (!m.indices.length) { note(`${k}: builds an empty mesh`); empty++; }
  }
  if (!empty) console.log(`  ok   all ${kinds.length} triply-periodic surfaces build`);

  const { buildBinarySTLFromMesh } = await import(new URL('web/js/stl.js', ROOT).href);
  const m = T.block('G', 1, 16, 2);
  const built = buildBinarySTLFromMesh(m.positions, m.indices,
                                       { sizeMM: 150, thicknessMM: 1.6, name: 'Gyroid' });
  const buf = Buffer.from(await built.blob.arrayBuffer());
  const count = buf.readUInt32LE(80);
  if (buf.length !== 84 + count * 50) note(`STL: ${buf.length} bytes for ${count} triangles`);
  else if (buf.subarray(0, 80).toString('latin1').trimStart().startsWith('solid')) {
    note('STL: header starts with "solid", which readers take for ASCII');
  } else if (!built.thickened) note('STL: the sheet came out unwalled');
  else console.log(`  ok   STL: ${count.toLocaleString()} triangles, walled, `
                   + `${Math.max(...built.mm).toFixed(0)} mm`);
}

console.log(`  worst difference: fields ${worstField.toExponential(2)}, `
            + `vertices ${worstVert.toExponential(2)}`);
for (const f of failures) console.log('  FAIL ' + f);
if (fail > failures.length) console.log(`  ... and ${fail - failures.length} more`);
console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
