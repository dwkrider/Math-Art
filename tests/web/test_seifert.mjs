// The browser port of the Seifert-surface engine, against the engine.
//
//     node tests/web/test_seifert.mjs
//
// Run by tests/test_web.py when node and python are available.
//
// The reference is produced by tools/seifert_reference.py from
// math_art/seifert/, so this fails the moment the two disagree.
//
// WHAT IS WORTH CHECKING, AND WHY. The combinatorics are the part that
// carries the mathematics -- the genus of a knot is read off the disk
// and band counts, not measured off the mesh -- so bands, feet,
// anchors, chi, boundary count, orientability, genus and the traversal
// reversals are pinned exactly. The geometry is then pinned vertex by
// vertex, because a construction that gets the counting right and the
// surface wrong would still print a plausible-looking genus.
//
// One quantity is deliberately NOT compared: the disk order. See the
// note where it is used.
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const PROJ = path.resolve(path.dirname(new URL(import.meta.url).pathname
  .replace(/^\/([A-Za-z]:)/, '$1')), '..', '..');
const ROOT = pathToFileURL(PROJ + path.sep).href;
const S = await import(new URL('web/js/seifert-math.js', ROOT).href);

let fail = 0;
const note = (m) => { fail++; console.log('  FAIL ' + m); };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const ref = JSON.parse(execFileSync('python',
  [path.join(PROJ, 'tools', 'seifert_reference.py')],
  { cwd: PROJ, maxBuffer: 1 << 29 }).toString());

// ---- braid words and their invariants
for (const c of ref.cases) {
  const b = S.parseBraid(c.word);
  if (b.strands !== c.strands) note(`${c.word}: ${b.strands} strands vs ${c.strands}`);
  if (!eq(S.signedWord(b), c.signed)) note(`${c.word}: signed word`);
  if (b.crossings.length !== c.n_bands) note(`${c.word}: band count`);
  if (S.nComponents(b) !== c.n_components) {
    note(`${c.word}: ${S.nComponents(b)} components vs ${c.n_components}`);
  }
  if (!eq(S.strandPermutation(b), c.permutation)) note(`${c.word}: permutation`);
}
console.log(`  ok   ${ref.cases.length} braid words: strands, signed form, `
  + 'permutation and component count');

// ---- the state surfaces each word carries
let ambiguous = 0;
for (const c of ref.cases) {
  const b = S.parseBraid(c.word);
  for (const st of c.states) {
    const d = S.stateData(b, st.bits);
    const tag = `${c.word}/${st.name}`;
    if (d.nCircles !== st.n_circles) { note(`${tag}: circle count`); continue; }
    if (!eq(d.bands, st.bands)) { note(`${tag}: bands`); continue; }
    if (!eq(d.feet, st.feet)) { note(`${tag}: feet`); continue; }
    const anchorsOk = d.anchors.every((a, i) =>
      Math.abs(a[0] - st.anchors[i][0]) < 1e-9
      && Math.abs(a[1] - st.anchors[i][1]) < 1e-9);
    if (!anchorsOk) { note(`${tag}: anchors`); continue; }
    if (d.eulerCharacteristic !== st.chi) { note(`${tag}: chi`); continue; }
    if (d.nBoundaries !== st.n_boundaries) { note(`${tag}: boundaries`); continue; }
    if (d.orientable !== st.orientable) {
      note(`${tag}: orientable ${d.orientable} vs ${st.orientable}`); continue;
    }
    if (d.genus !== st.genus) { note(`${tag}: genus ${d.genus} vs ${st.genus}`); continue; }
    if (d.crosscapNumber !== st.crosscap) { note(`${tag}: crosscap`); continue; }
    if (!eq(S.rotationReversals(d), st.reversals)) {
      note(`${tag}: traversal reversals ${JSON.stringify(S.rotationReversals(d))} `
        + `vs ${JSON.stringify(st.reversals)}`);
      continue;
    }
    // THE DISK ORDER IS NOT COMPARED. It comes from the Fiedler vector
    // of the state graph, and an eigenvector has no inherent sign:
    // flipping it reverses the stack. When the second eigenvalue is
    // repeated -- a symmetric state graph -- the vector is not defined
    // even up to sign. All that can be asserted is that the port's
    // order is a permutation of the circles, and, where the eigenvalue
    // is simple, that it agrees with the engine's up to reversal.
    const mine = S.spatialOrder(d);
    if (!eq(mine.slice().sort((x, y) => x - y),
            Array.from({ length: d.nCircles }, (_, i) => i))) {
      note(`${tag}: order is not a permutation`);
      continue;
    }
    // Comparing the two permutations directly is the wrong test, and
    // was tried: a sign flip alone reverses the order, but a sign flip
    // with TIED entries does not, because the tie-break then runs the
    // other way through the ties. What is actually required is the
    // PROPERTY -- that the order is a valid ascending sort by the key
    // spatial_order uses, for one of the two signs the eigenvector may
    // come back with.
    if (st.fiedler === null) {
      if (!eq(mine, st.order)) note(`${tag}: order differs with no Fiedler vector`);
    } else if (st.fiedler_gap > 1e-9) {
      const sorted = (sign) => {
        const key = (i) => [Number((sign * st.fiedler[i]).toFixed(9)),
                            d.anchors[i][0], d.anchors[i][1]];
        for (let k = 1; k < mine.length; k++) {
          const a = key(mine[k - 1]), b = key(mine[k]);
          for (let c = 0; c < 3; c++) {
            if (a[c] < b[c]) break;
            if (a[c] > b[c]) return false;
          }
        }
        return true;
      };
      if (!sorted(1) && !sorted(-1)) {
        note(`${tag}: order ${JSON.stringify(mine)} is not a sort by the `
          + `engine's Fiedler vector under either sign`);
      }
    } else {
      ambiguous++;
    }
  }
}
const nStates = ref.cases.reduce((a, c) => a + c.states.length, 0);
console.log(`  ok   ${nStates} state surfaces: bands, feet, anchors, chi, `
  + 'boundaries, orientability, genus, crosscap, traversal');
console.log(`  ok   stacking order agrees up to the sign of an eigenvector; `
  + `${ambiguous} of ${nStates} had a repeated eigenvalue, where it is not `
  + 'determined at all');

// ---- the meshes themselves
for (const g of ref.geometry) {
  const tag = `${g.word}/${g.which}/L${g.levels}`;
  const b = S.parseBraid(g.word);
  const data = S.stateData(b, g.which === 'seifert'
    ? S.seifertState(b) : S.turnbackState(b));
  const params = {
    samplesPerSector: ref.params.samples_per_sector,
    bandSamples: ref.params.band_samples,
    radialRings: ref.params.radial_rings,
  };
  let mesh;
  try {
    // the engine's own stacking order goes in, so this compares meshes
    // rather than eigensolvers
    mesh = S.stateSurface(b, data, params, g.order);
  } catch (err) { note(`${tag}: threw ${err.message}`); continue; }
  if (g.levels) mesh = S.catmullClark(mesh, g.levels);

  if (mesh.vertices.length !== g.n_vertices) {
    note(`${tag}: ${mesh.vertices.length} vertices vs ${g.n_vertices}`); continue;
  }
  if (mesh.faces.length !== g.n_faces) {
    note(`${tag}: ${mesh.faces.length} faces vs ${g.n_faces}`); continue;
  }
  const info = mesh.info();
  const want = g.info;
  if (info.eulerCharacteristic !== want.chi || info.nBoundaries !== want.boundaries
      || info.orientable !== want.orientable || info.genus !== want.genus
      || info.nEdges !== want.n_edges) {
    note(`${tag}: topology ${JSON.stringify(info)} vs ${JSON.stringify(want)}`);
    continue;
  }
  if (!eq(mesh.boundaryLoops().map((l) => l.length).sort((a, c) => a - c),
          g.loop_lengths)) {
    note(`${tag}: boundary loop lengths`); continue;
  }
  if (Math.abs(mesh.area() - g.area) > 1e-7 * Math.max(1, g.area)) {
    note(`${tag}: area ${mesh.area()} vs ${g.area}`); continue;
  }
  let worst = 0;
  if (g.vertices) {
    for (let v = 0; v < g.vertices.length; v++) {
      for (let k = 0; k < 3; k++) {
        worst = Math.max(worst, Math.abs(mesh.vertices[v][k] - g.vertices[v][k]));
      }
    }
    if (worst > 1e-8) { note(`${tag}: vertices differ by ${worst.toExponential(2)}`); continue; }
    if (!eq(mesh.faces, g.faces)) { note(`${tag}: face list differs`); continue; }
  }
  const sided = want.orientable
    ? `two-sided genus ${want.genus}` : `one-sided k=${want.euler_genus}`;
  console.log(`  ok   ${tag}: ${g.n_vertices} verts, ${g.n_faces} faces, `
    + `chi=${want.chi} b=${want.boundaries} ${sided}`);
}

// ---- torus knots, which the page offers as presets
for (const t of ref.torus) {
  const b = S.torusKnot(t.p, t.q);
  if (b.word !== t.word) note(`torus(${t.p},${t.q}): ${b.word} vs ${t.word}`);
  if (S.nComponents(b) !== t.n_components) {
    note(`torus(${t.p},${t.q}): components`);
  }
}
console.log(`  ok   ${ref.torus.length} torus knots: word and component count`);

// ---- independent of the engine: the genus the construction reports
// has to satisfy Euler's formula, and the classical knots have the
// genus the literature gives them.
{
  const known = [['AAA', 1, 1], ['AbAb', 1, 1], ['A5', 2, 1], ['A7', 3, 1],
                 ['ABABABAB', 3, 1], ['AA', 0, 2]];
  for (const [word, genus, components] of known) {
    const b = S.parseBraid(word);
    const d = S.stateData(b, S.seifertState(b));
    if (d.nBoundaries !== components) {
      note(`${word}: ${d.nBoundaries} components, the literature says ${components}`);
    }
    if (d.genus !== genus) {
      note(`${word}: genus ${d.genus}, the literature says ${genus}`);
    }
    // chi = 2 - 2g - b, independently of how it was counted
    if (d.eulerCharacteristic !== 2 - 2 * d.genus - d.nBoundaries) {
      note(`${word}: chi ${d.eulerCharacteristic} contradicts genus and boundary count`);
    }
  }
  console.log(`  ok   ${known.length} classical knots have their published `
    + 'genus, consistent with Euler\'s formula');
}

console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
