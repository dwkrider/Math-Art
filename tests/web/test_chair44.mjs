// Parity: web/js/chair44-math.js against math_art/ifs/chair44.py.
//
//     node tests/web/test_chair44.mjs [reference.json]
//
// With no argument it runs tools/chair44_reference.py itself (set
// PYTHON to choose the interpreter), so the reference is the engine as
// it stands on this checkout -- a change to the mathematics on master
// fails this until the port follows.
//
// The engine is exact (Fractions) and the port is not (doubles), so
// positions are compared at 1e-12. That is not a fudge factor: the
// construction's smallest feature is 1/10000 tall and its closest
// distinct cuts are 0.105 apart, so anything a porting mistake could
// do is many orders of magnitude larger. What the test really pins
// down is discrete: the same vertices in the same order, the same
// faces with the same winding, the same poses in the same order, and
// the same contact set.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

const HERE = new URL('.', import.meta.url);
const ROOT = new URL('../../', HERE);
const C = await import(new URL('web/js/chair44-math.js', ROOT).href);

let ref;
if (process.argv[2]) {
  ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
} else {
  const py = process.env.PYTHON || 'python';
  const script = new URL('tools/chair44_reference.py', ROOT).pathname
    .replace(/^\/([A-Za-z]:)/, '$1');
  const r = spawnSync(py, [script], { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
  if (r.status !== 0) {
    console.log('  FAIL could not run tools/chair44_reference.py: ' + (r.stderr || r.error));
    console.log('\nRESULT: 1 FAILURE(S)');
    process.exit(1);
  }
  ref = JSON.parse(r.stdout);
}

const TOL = 1e-12;
let fail = 0;
const failures = [];
const note = (m) => { fail++; if (failures.length < 25) failures.push(m); };
const near = (a, b) => Math.abs(a - b) <= TOL;
const sameInts = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
let worst = 0;

function checkMesh(name, got, want, colors) {
  if (got.verts.length !== want.verts.length) {
    note(`${name}: ${got.verts.length} vertices vs ${want.verts.length}`);
    return;
  }
  for (let i = 0; i < want.verts.length; i++) {
    for (let k = 0; k < 3; k++) {
      const e = Math.abs(got.verts[i][k] - want.verts[i][k]);
      worst = Math.max(worst, e);
      if (e > TOL) {
        note(`${name}: vertex ${i} differs by ${e.toExponential(2)}`);
        return;
      }
    }
  }
  if (got.faces.length !== want.faces.length) {
    note(`${name}: ${got.faces.length} faces vs ${want.faces.length}`);
    return;
  }
  for (let i = 0; i < want.faces.length; i++) {
    // Winding matters: these faces are the outward orientation of a
    // closed solid, and a reversed one is a hole in the shading.
    if (!sameInts(got.faces[i], want.faces[i])) {
      note(`${name}: face ${i} is [${got.faces[i]}] vs [${want.faces[i]}]`);
      return;
    }
  }
  if (colors) {
    const bad = colors.findIndex((c, i) => (c === null ? want.colors[i] !== null
                                                       : c !== want.colors[i]));
    if (bad >= 0) note(`${name}: face ${bad} colour ${colors[bad]} vs ${want.colors[bad]}`);
  }
  console.log(`  ok   ${name}: ${want.verts.length} vertices, ${want.faces.length} faces`);
}

// ---- the three tile meshes the site can draw
const bare = C.bareTileMesh();
checkMesh('bare tile', bare, ref.meshes.bare);
const arrow = C.arrowTileMesh();
checkMesh('arrow tile', arrow, ref.meshes.arrow, arrow.colors);
checkMesh('featured tile (true size)', C.tileMesh(), ref.meshes.true);
checkMesh('featured tile (enlarged)',
          C.tileMesh(C.ETA_SHOWN, ref.relief * C.HEIGHT_TRUE),
          ref.meshes.enlarged);

// ---- volumes: the solid is volume 7 whatever the relief
for (const [k, want] of Object.entries(ref.volumes)) {
  const m = k === 'bare' ? bare
    : k === 'true' ? C.tileMesh()
    : C.tileMesh(C.ETA_SHOWN, ref.relief * C.HEIGHT_TRUE);
  const got = C.meshVolume(m.verts, m.faces);
  if (Math.abs(got - want) > 1e-9) note(`volume ${k}: ${got} vs ${want}`);
  else if (Math.abs(want - 7) > 1e-9) note(`volume ${k}: reference is ${want}, not 7`);
}
console.log('  ok   every tile has volume exactly 7 (bumps and dents cancel)');

// ---- scalars and tables
if (!near(C.MIN_CONTACT_SPAN, ref.min_contact_span)) {
  note(`MIN_CONTACT_SPAN ${C.MIN_CONTACT_SPAN} vs ${ref.min_contact_span}`);
}
for (const [g, want] of Object.entries(ref.max_relief)) {
  const got = C.maxRelief(Number(g));
  const ok = want === null || !isFinite(want) ? !isFinite(got)
                                              : Math.abs(got - want) <= 1e-9;
  if (!ok) note(`maxRelief(${g}) ${got} vs ${want}`);
}
for (const [k, want] of Object.entries(ref.arrow_area)) {
  const got = C.arrowArea(k === 'full' ? 0 : 1);
  if (Math.abs(got - want) > 1e-12) note(`arrowArea ${k}: ${got} vs ${want}`);
}
if (C.PROPER_FRAMES.length !== 24) note(`${C.PROPER_FRAMES.length} proper frames, not 24`);
C.PROPER_FRAMES.forEach((M, i) => {
  if (M.flat().join(',') !== ref.proper_frames[i].flat().join(',')) {
    note(`proper frame ${i} differs -- the rotation colouring would be re-indexed`);
  }
});
// the arrow rule per panel: which vertex it points at, and which half
C.PANELS.forEach((panel, i) => {
  const [home, dir] = C.panelHome(panel);
  const [wHome, wDir, wHalf] = ref.panel_home[i];
  const flip = (panel.ax !== 1 ? 1 : -1) !== panel.sg;
  if (!sameInts(home, wHome) || !near(dir[0], wDir[0]) || !near(dir[1], wDir[1])
      || C.arrowHalf(panel.pid, flip) !== wHalf) {
    note(`panel ${panel.pid}: arrow home/half differs`);
  }
});
// the 192 features, in order
const feats = C.features();
if (feats.length !== ref.features.length) {
  note(`${feats.length} features vs ${ref.features.length}`);
} else {
  for (let i = 0; i < feats.length; i++) {
    const [c, n, a] = ref.features[i];
    const f = feats[i];
    if (!c.every((x, k) => near(f.centre[k], x)) || !sameInts(f.normal, n)
        || f.coeff !== a) {
      note(`feature ${i} differs`);
      break;
    }
  }
}
console.log(`  ok   ${feats.length} features, 24 proper frames, arrow rule per panel`);

// ---- the substitution
for (const [d, want] of Object.entries(ref.patches)) {
  const got = C.patch(Number(d));
  if (got.length !== want.length) {
    note(`patch ${d}: ${got.length} chairs vs ${want.length}`);
    continue;
  }
  let bad = 0;
  for (let i = 0; i < want.length; i++) {
    const w = want[i], g = got[i];
    if (g.G.flat().join(',') !== w.G.flat().join(',') || !sameInts(g.t, w.t)
        || g.group !== w.group || C.frameIndex(g.G) !== w.frame) {
      bad++;
    }
  }
  if (bad) note(`patch ${d}: ${bad} of ${want.length} chairs differ in pose, supertile or frame`);
  else console.log(`  ok   patch depth ${d}: ${want.length} chairs, same poses in the same order`);
}

// ---- contacts, and the atlas
for (const [d, want] of Object.entries(ref.contacts)) {
  const got = C.contacts(C.patch(Number(d)));
  if (got.length !== want.length) {
    note(`contacts ${d}: ${got.length} vs ${want.length}`);
    continue;
  }
  let bad = 0;
  for (let i = 0; i < want.length; i++) {
    const w = want[i], g = got[i];
    if (g.i !== w.i || g.j !== w.j || g.G.flat().join(',') !== w.G.flat().join(',')
        || !sameInts(g.t, w.t)) bad++;
  }
  if (bad) note(`contacts ${d}: ${bad} of ${want.length} differ`);
  else console.log(`  ok   depth ${d}: ${want.length} face contacts, identical`);
}
// the property the page reports to the reader
const inAtlas = C.contacts(C.patch(2)).every((c) => C.inAtlas(c.G, c.t));
if (inAtlas !== ref.depth2_all_in_atlas) {
  note(`atlas check: port says ${inAtlas}, engine says ${ref.depth2_all_in_atlas}`);
} else if (!inAtlas) {
  note('a depth-2 patch contains a contact outside the 44-contact atlas');
} else {
  console.log('  ok   every contact of a depth-2 patch is one of the 44');
}

console.log(`  worst coordinate difference: ${worst.toExponential(2)} (tolerance ${TOL})`);
for (const f of failures) console.log('  FAIL ' + f);
if (fail > failures.length) console.log(`  ... and ${fail - failures.length} more`);
console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
