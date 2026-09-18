// Parity: web/js/belt-math.js against math_art/belt_trick_generator.py.
//
//     node tests/web/test_belt_math.mjs [reference.json]
//
// With no argument it runs tools/belt_reference.py itself (set PYTHON to
// choose the interpreter), so the reference is always the generator as
// it stands on this checkout -- a change to the mathematics on master
// fails this until the port follows.
//
// Tolerance is 1e-9 on positions in a 2 m cage. The two languages use
// different libms, so the last bit of a sine can differ, and the fit's
// banded solve amplifies that by its condition number -- to around
// 1e-12. A real porting mistake is far larger: a width that eases back
// to the wrong half turn reverses a row, which is an error the size of
// the belt, and is reported as a FLIP rather than as a number.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

const HERE = new URL('.', import.meta.url);
const ROOT = new URL('../../', HERE);
const B = await import(new URL('web/js/belt-math.js', ROOT).href);

let ref;
if (process.argv[2]) {
  ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
} else {
  const py = process.env.PYTHON || 'python';
  const r = spawnSync(py, [new URL('tools/belt_reference.py', ROOT).pathname.replace(/^\/([A-Za-z]:)/, '$1')],
                      { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
  if (r.status !== 0) {
    console.log('  FAIL could not run tools/belt_reference.py: ' + (r.stderr || r.error));
    console.log('\nRESULT: 1 FAILURE(S)');
    process.exit(1);
  }
  ref = JSON.parse(r.stdout);
}

const TOL = 1e-9;
let fail = 0;
const failures = [];
const note = (m) => { fail++; if (failures.length < 25) failures.push(m); };
const maxErr = (a, b) => {
  let e = 0;
  for (let k = 0; k < a.length; k++) e = Math.max(e, Math.abs(a[k] - b[k]));
  return e;
};
const DEG = Math.PI / 180;

const toKw = (kw) => ({
  kind: kw.kind, spinAxis: kw.spin_axis || 'Z',
  mirror: !!kw.mirror, freq: kw.freq || 2,
});

let worstLine = 0, worstScalar = 0, lines = 0;
const perTurn = new Map();

for (const c of ref.cases) {
  const prep = B.prepare(toKw(c.kwargs));
  for (const [key, jsKey] of [['half', 'half'], ['limit', 'limit'],
                              ['width', 'width'], ['fit', 'fit'],
                              ['r_min', 'rMin'], ['solid_thick', 'solidThick']]) {
    const e = Math.abs(prep[jsKey] - c[key]);
    worstScalar = Math.max(worstScalar, e);
    if (e > TOL) note(`${c.label}: prepare ${key} ${prep[jsKey]} vs ${c[key]}`);
  }
  if (prep.verts.length !== c.nverts || prep.faces.length !== c.nfaces) {
    note(`${c.label}: solid ${prep.verts.length}v ${prep.faces.length}f vs `
         + `${c.nverts}v ${c.nfaces}f`);
  }
  if (prep.belts.length !== c.belts.length) {
    note(`${c.label}: ${prep.belts.length} belts vs ${c.belts.length}`);
    continue;
  }
  // Belt ORDER matters -- it is the colours -- so compare index by index.
  for (let i = 0; i < c.belts.length; i++) {
    const [u, w, rf, hf] = prep.belts[i];
    const [ru, rw, rrf, rhf] = c.belts[i];
    const e = Math.max(maxErr(u, ru), maxErr(w, rw), Math.abs(rf - rrf), Math.abs(hf - rhf));
    worstScalar = Math.max(worstScalar, e);
    if (e > TOL) note(`${c.label}: belt ${i} differs by ${e.toExponential(2)} (order or geometry)`);
  }
  for (const t of c.turns) {
    const got = B.allBelts(prep, 0.5 * t.turn * DEG, ref.ns);
    let worst = 0, flips = 0;
    for (let b = 0; b < got.length; b++) {
      for (let i = 0; i < ref.ns; i++) {
        const eP = maxErr(got[b].P[i], t.P[b][i]);
        const d = got[b].dir[i], rd = t.dir[b][i];
        const eD = maxErr(d, rd);
        // a reversed width direction is the same strip with every row
        // flipped -- a discrete failure, not a numerical one
        const flipped = eD > TOL && maxErr(d, rd.map((x) => -x)) <= TOL;
        if (flipped) flips++;
        worst = Math.max(worst, eP, flipped ? 0 : eD);
      }
      lines++;
    }
    worstLine = Math.max(worstLine, worst);
    const key = t.turn;
    const rec = perTurn.get(key) || { worst: 0, flips: 0 };
    rec.worst = Math.max(rec.worst, worst);
    rec.flips += flips;
    perTurn.set(key, rec);
    if (flips) note(`${c.label} @ ${t.turn}°: ${flips} sample(s) with the width FLIPPED`);
    else if (worst > TOL) note(`${c.label} @ ${t.turn}°: line differs by ${worst.toExponential(2)}`);
  }
}

// The rows themselves, once: the combination of line and direction.
{
  const rc = ref.rows_check;
  const prep = B.prepare({ kind: rc.kind });
  const [u, w, , hf] = prep.belts[rc.belt];
  const line = B.beltLine(0.5 * rc.turn * DEG, u, w, 1.0, hf, prep.width,
                          prep.reach, ref.ns, prep.n, prep.m, prep.rMin, prep.smoothing);
  const rows = B.rowsOf(line, 11);
  let e = 0;
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j < 11; j++) e = Math.max(e, maxErr(rows[i][j], rc.rows[i][j]));
  }
  if (e > TOL) note(`rows: differ by ${e.toExponential(2)}`);
  else console.log(`  ok   cross-section rows match (${e.toExponential(1)})`);
}

// diagnose(): level, measurements, and which warning.
for (const d of ref.diagnose) {
  const prep = B.prepare(toKw(d.kwargs));
  const allRows = B.allBelts(prep, 0.5 * d.turn * DEG, ref.ns).map((l) => B.rowsOf(l, 11));
  const [level, meas, warning] = B.diagnose(prep, allRows);
  if (level !== d.level) note(`diagnose ${d.label}: level ${level} vs ${d.level}`);
  for (const k of Object.keys(d.meas)) {
    const e = Math.abs(meas[k] - d.meas[k]);
    if (e > 1e-7) note(`diagnose ${d.label}: ${k} ${meas[k]} vs ${d.meas[k]}`);
  }
  // Compare the words, not the formatted numbers: %.3f and toFixed(3)
  // may round an exact binary tie differently, and the numbers are
  // already compared above.
  const words = (s) => (s || '').replace(/[0-9.]+/g, '#');
  if (words(warning) !== words(d.warning)) {
    note(`diagnose ${d.label}: warning "${warning}" vs "${d.warning}"`);
  } else {
    console.log(`  ok   diagnose ${d.label}: ${level}${warning ? ' — ' + warning.slice(0, 48) + '…' : ''}`);
  }
}

console.log(`  ${ref.cases.length} configurations, ${lines} belt lines compared`);
console.log(`  worst difference: prepare/solids ${worstScalar.toExponential(2)}, `
            + `lines ${worstLine.toExponential(2)} (tolerance ${TOL})`);
const turns = [...perTurn.keys()].sort((a, b) => a - b);
console.log('  by turn: ' + turns.map((t) => {
  const r = perTurn.get(t);
  return `${t}°:${r.worst.toExponential(0)}${r.flips ? '!' + r.flips : ''}`;
}).join('  '));
for (const f of failures) console.log('  FAIL ' + f);
if (fail > failures.length) console.log(`  ... and ${fail - failures.length} more`);
console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
