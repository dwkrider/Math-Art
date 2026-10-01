// Parity: web/js/hopf-math.js against math_art/hopf_fibration_generator.py.
//
//     node tests/web/test_hopf.mjs [reference.json]
//
// With no argument it runs tools/hopf_reference.py itself (set PYTHON
// to choose the interpreter), so the reference is the generator as it
// stands on this checkout -- a change to the mathematics on master
// fails this until the port follows.
//
// Tolerance is 1e-9 on positions in a unit-radius fit. The two
// languages use different libms, so the last bit of a sine can differ;
// stereographic projection near the pole magnifies that, which is why
// the near-axis case is in the reference at all. A porting mistake is
// nothing like that small: a wrong chirality mirrors a fibre, a wrong
// half-angle puts it on the wrong torus.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

const HERE = new URL('.', import.meta.url);
const ROOT = new URL('../../', HERE);
const H = await import(new URL('web/js/hopf-math.js', ROOT).href);

let ref;
if (process.argv[2]) {
  ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
} else {
  const py = process.env.PYTHON || 'python';
  const script = new URL('tools/hopf_reference.py', ROOT).pathname
    .replace(/^\/([A-Za-z]:)/, '$1');
  const r = spawnSync(py, [script], { encoding: 'utf8', maxBuffer: 512 * 1024 * 1024 });
  if (r.status !== 0) {
    console.log('  FAIL could not run tools/hopf_reference.py: ' + (r.stderr || r.error));
    console.log('\nRESULT: 1 FAILURE(S)');
    process.exit(1);
  }
  ref = JSON.parse(r.stdout);
}

const TOL = 1e-9;
let fail = 0;
const failures = [];
const note = (m) => { fail++; if (failures.length < 25) failures.push(m); };
let worst = 0;

function cmpRows(name, got, want, stride) {
  // `got` is a flat typed array, `want` a list of rows
  if (got.length / stride !== want.length) {
    note(`${name}: ${got.length / stride} points vs ${want.length}`);
    return false;
  }
  for (let i = 0; i < want.length; i++) {
    for (let k = 0; k < stride; k++) {
      const e = Math.abs(got[i * stride + k] - want[i][k]);
      if (e > worst) worst = e;
      if (e > TOL) {
        note(`${name}: point ${i} coord ${k} differs by ${e.toExponential(2)}`);
        return false;
      }
    }
  }
  return true;
}

// ---- the tilt constant, which every picture depends on
if (H.TILT.some((x, i) => Math.abs(x - ref.tilt[i]) > 1e-15)) {
  note(`TILT ${H.TILT} vs ${ref.tilt}`);
}

// ---- the base-point providers
let okPresets = 0;
for (const [preset, want] of Object.entries(ref.base_points)) {
  const got = H.basePoints(preset, 5, 12, 20.0, 160.0,
                           { turns: 4.0, curl_lobes: 6, curl_amp: 22.0 });
  if (got.length !== want.length) {
    note(`base points ${preset}: ${got.length} vs ${want.length}`);
    continue;
  }
  let bad = false;
  for (let i = 0; i < want.length && !bad; i++) {
    for (let k = 0; k < 3; k++) {
      const e = Math.abs(got[i][k] - want[i][k]);
      if (e > worst) worst = e;
      if (e > TOL) {
        note(`base points ${preset}: point ${i} differs by ${e.toExponential(2)}`);
        bad = true;
        break;
      }
    }
  }
  if (!bad) okPresets++;
}
console.log(`  ok   ${okPresets} of ${Object.keys(ref.base_points).length} base-point providers match`);

// ---- the kernel: lift to S^3, rotate, project
let okKernel = 0;
for (const c of ref.fiber_s3) {
  let X = H.fiberS3(c.base, 32, c.P, c.Q, c.chirality);
  if (c.rot) X = H.s3Rotate(X, H.quatLeft(c.rot * Math.PI / 180));
  const a = cmpRows(`fibre ${c.base} P${c.P}Q${c.Q} ${c.chirality} rot${c.rot}`,
                    X, c.s3, 4);
  const b = cmpRows(`projected ${c.base} P${c.P}Q${c.Q} ${c.chirality} rot${c.rot}`,
                    H.stereographic(X), c.projected, 3);
  if (a && b) okKernel++;
}
console.log(`  ok   ${okKernel} of ${ref.fiber_s3.length} fibre lifts and projections match`);

// ---- the quaternion product in general, not just the flow's own
// (cos phi, sin phi, 0, 0), whose j and k terms are identically zero
let okQuat = 0;
for (const c of ref.quat_rotations) {
  const X = H.fiberS3(c.base, 24, 1, 1, 'RIGHT');
  if (cmpRows(`s3Rotate by [${c.q}]`, H.s3Rotate(X, c.q), c.s3, 4)) okQuat++;
}
console.log(`  ok   ${okQuat} of ${ref.quat_rotations.length} general quaternion rotations match`);

// ---- the flow must MOVE the family, not slide points along their own
// fibres. Multiplying on the wrong side leaves every circle exactly
// where it was, so the control looks broken while every other check
// still passes; that is what happened, and this is the guard.
{
  const hopf = (x) => [2 * (x[0] * x[2] + x[1] * x[3]),
                       2 * (x[1] * x[2] - x[0] * x[3]),
                       x[0] ** 2 + x[1] ** 2 - x[2] ** 2 - x[3] ** 2];
  let ok = 0;
  for (const c of ref.flow) {
    let X = H.fiberS3(c.base, 24, 1, 1, 'RIGHT');
    if (c.deg) X = H.s3Flow(X, H.quatFlow(c.deg * Math.PI / 180));
    if (!cmpRows(`flow ${c.deg}deg`, X, c.s3, 4)) continue;
    const got = hopf(Array.from(X.slice(0, 4)));
    const e = Math.max(...got.map((v, i) => Math.abs(v - c.moved_base[i])));
    if (e > TOL) { note(`flow ${c.deg}deg: base lands at ${got} vs ${c.moved_base}`); continue; }
    const moved = Math.hypot(...got.map((v, i) => v - c.base[i]));
    if (c.deg && moved < 0.1) {
      note(`flow ${c.deg}deg does not move the base point (moved ${moved.toExponential(1)})`);
      continue;
    }
    ok++;
  }
  console.log(`  ok   ${ok} of ${ref.flow.length} flow steps match, and they move the family`);
}

// ---- whole builds, including the fit
for (const c of ref.cases) {
  const kw = c.kwargs;
  const got = H.buildFibers({
    preset: kw.preset, nLat: kw.n_lat ?? 6, nFiber: kw.n_fiber ?? 24,
    samples: kw.samples ?? 160, P: kw.P ?? 1, Q: kw.Q ?? 1,
    latMin: kw.lat_min ?? 20.0, latMax: kw.lat_max ?? 160.0,
    s3Rot: kw.s3_rot ?? 0, chirality: kw.chirality ?? 'RIGHT',
    includeAxis: kw.include_axis ?? false, extra: kw.extra ?? {},
    adaptive: kw.adaptive ?? true,
    fitCentre: kw.fit_centre ?? null, fitScale: kw.fit_scale ?? null,
  });
  // the fit itself, which an animation holds still between frames
  if (c.centre !== undefined) {
    const e = Math.max(...got.centre.map((v, i) => Math.abs(v - c.centre[i])),
                       Math.abs(got.scale - c.scale));
    if (e > TOL) note(`${c.label}: fit centre/scale differs by ${e.toExponential(2)}`);
  }
  if (got.fibers.length !== c.fibers.length) {
    note(`${c.label}: ${got.fibers.length} fibres vs ${c.fibers.length}`);
    continue;
  }
  if (got.dropped !== c.dropped) {
    note(`${c.label}: dropped ${got.dropped} vs ${c.dropped}`);
  }
  let bad = 0;
  for (let i = 0; i < c.fibers.length; i++) {
    // fibre ORDER matters: it is the colour order, and for BOTH
    // chirality it interleaves the two rulings
    if (!cmpRows(`${c.label} fibre ${i}`, got.fibers[i], c.fibers[i], 3)) { bad++; break; }
    if (got.closed[i] !== c.closed[i]) {
      note(`${c.label} fibre ${i}: closed ${got.closed[i]} vs ${c.closed[i]}`);
      bad++;
      break;
    }
    for (let k = 0; k < 3; k++) {
      if (Math.abs(got.bases[i][k] - c.bases[i][k]) > TOL) {
        note(`${c.label} fibre ${i}: base point differs`);
        bad++;
        break;
      }
    }
  }
  if (!bad) {
    const pts = c.fibers.reduce((n, f) => n + f.length, 0);
    console.log(`  ok   ${c.label}: ${c.fibers.length} fibres, ${pts} points`);
  }
}

// ---- colour, which ties a dot on the base sphere to its fibre
let okColors = 0;
for (const c of ref.colors) {
  const rgb = H.paletteRgb(c.base, c.style);
  const par = H.paramRgb(0.3, c.style);
  const e = Math.max(...rgb.map((x, i) => Math.abs(x - c.rgb[i])),
                     ...par.map((x, i) => Math.abs(x - c.param[i])));
  if (e > worst) worst = e;
  if (e > TOL) note(`colour ${c.style} ${c.base}: differs by ${e.toExponential(2)}`);
  else okColors++;
}
console.log(`  ok   ${okColors} of ${ref.colors.length} colours match`);

// ---- what the page claims about the pictures it draws
{
  // any two distinct fibres are linked exactly once
  const b1 = H.normalize3([0.3, 0.1, 0.6]);
  const b2 = H.normalize3([-0.4, 0.5, -0.2]);
  const f1 = H.projectFiber(b1, 400);
  const f2 = H.projectFiber(b2, 400);
  const lk = H.linkingNumber(f1, f2);
  if (Math.abs(Math.abs(lk) - 1) > 0.02) {
    note(`linking number of two fibres is ${lk.toFixed(4)}, not +-1`);
  } else {
    console.log(`  ok   two fibres link exactly once (computed ${lk.toFixed(4)})`);
  }
  // a fibre lies on the unit 3-sphere
  const X = H.fiberS3(b1, 64);
  let off = 0;
  for (let i = 0; i < X.length; i += 4) {
    off = Math.max(off, Math.abs(Math.hypot(X[i], X[i + 1], X[i + 2], X[i + 3]) - 1));
  }
  if (off > 1e-12) note(`fibre leaves the 3-sphere by ${off.toExponential(2)}`);
  else console.log('  ok   every lifted point is on the unit 3-sphere');
}

// ---- INDEPENDENT checks.
//
// Everything above compares the port with the generator, which cannot
// catch a mistake both of them share -- and there has been one: the
// flow multiplied on the wrong side in both, so the control did
// nothing and every parity check still passed. These test the
// geometry against definitions and closed forms from outside this
// project.
{
  const hopf = (x) => [2 * (x[0] * x[2] + x[1] * x[3]),
                       2 * (x[1] * x[2] - x[0] * x[3]),
                       x[0] ** 2 + x[1] ** 2 - x[2] ** 2 - x[3] ** 2];
  const dist = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));

  // (a) The definition: h(z0, z1) = (2 z0 conj(z1), |z0|^2 - |z1|^2)
  // must send every point of a fibre to the one base point it was
  // asked for.
  let spread = 0, offBase = 0;
  for (const b of [[0, 0, 1], [1, 0, 0], [0.3, -0.6, 0.74162], [-0.5, 0.5, -0.70711]]) {
    const base = H.normalize3(b);
    const X = H.fiberS3(base, 64);
    const imgs = [];
    for (let i = 0; i < X.length; i += 4) imgs.push(hopf([X[i], X[i+1], X[i+2], X[i+3]]));
    for (const p of imgs) {
      spread = Math.max(spread, dist(p, imgs[0]));
      offBase = Math.max(offBase, dist(p, base));
    }
  }
  if (spread > 1e-12 || offBase > 1e-12) {
    note(`Hopf map: fibre spread ${spread.toExponential(1)}, base error ${offBase.toExponential(1)}`);
  } else {
    console.log('  ok   the standard Hopf map sends each fibre to its own base point');
  }

  // (b) Each fibre is a great circle of the 3-sphere: unit radius, and
  // its Gram matrix has rank two.
  let radius = 0, rank2 = 0;
  for (const b of [[0.3, -0.6, 0.74162], [0.1, 0.2, 0.97417]]) {
    const X = H.fiberS3(H.normalize3(b), 128);
    const P = [];
    for (let i = 0; i < X.length; i += 4) P.push([X[i], X[i+1], X[i+2], X[i+3]]);
    for (const q of P) radius = Math.max(radius, Math.abs(Math.hypot(...q) - 1));
    const G = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
    for (const q of P) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) G[i][j] += q[i]*q[j];
    const tr = G[0][0] + G[1][1] + G[2][2] + G[3][3];
    let fro = 0;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) fro += G[i][j] * G[i][j];
    rank2 = Math.max(rank2, Math.abs(fro - tr * tr / 2) / (tr * tr));
  }
  if (radius > 1e-12 || rank2 > 1e-12) {
    note(`great circle: |r-1| ${radius.toExponential(1)}, rank-2 residual ${rank2.toExponential(1)}`);
  } else {
    console.log('  ok   every fibre is a great circle of the 3-sphere');
  }

  // (c) Villarceau: projected, the fibres over a circle of latitude
  // beta lie on the torus of revolution with R = 1/cos(beta/2) and
  // r = tan(beta/2). That closed form follows from the projection and
  // owes nothing to our code.
  let offTorus = 0;
  for (const deg of [40, 70, 110]) {
    const beta = deg * Math.PI / 180;
    const R = 1 / Math.cos(beta / 2), r = Math.abs(Math.tan(beta / 2));
    for (let k = 0; k < 12; k++) {
      const lam = 2 * Math.PI * k / 12;
      const P = H.projectFiber([Math.sin(beta) * Math.cos(lam),
                                Math.sin(beta) * Math.sin(lam),
                                Math.cos(beta)], 200);
      for (let i = 0; i < P.length; i += 3) {
        const rho = Math.hypot(P[i], P[i + 1]);
        offTorus = Math.max(offTorus, Math.abs(Math.hypot(rho - R, P[i + 2]) - r));
      }
    }
  }
  if (offTorus > 1e-12) note(`Villarceau: points lie ${offTorus.toExponential(1)} off the torus`);
  else console.log('  ok   a ring of fibres lies on the torus R=1/cos(b/2), r=tan(b/2)');
}

console.log(`  worst difference: ${worst.toExponential(2)} (tolerance ${TOL})`);
for (const f of failures) console.log('  FAIL ' + f);
if (fail > failures.length) console.log(`  ... and ${fail - failures.length} more`);
console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
