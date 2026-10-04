// Batch-export every triply-periodic minimal surface the site offers,
// as a printable STL.
//
//     node tools/tpms_batch.mjs [outdir]
//
// It drives the same `buildScene` the page draws with and the same STL
// writer the page's download button uses, so a file from here is the
// file the page would have given you with those sliders set. Nothing
// about it is browser-specific: the whole pipeline lives in
// web/js/tpms-math.js and needs no DOM.
//
// Settings are at the top and are meant to be edited.

import { writeFileSync, readFileSync, mkdirSync, existsSync, statSync,
         openSync, readSync, closeSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const HERE = pathToFileURL(join(process.cwd(), 'web/js/')).href;
const T = await import(HERE + 'tpms-math.js');
const { buildBinarySTLFromMesh } = await import(HERE + 'stl.js');
const { boundaryEdges } = await import(HERE + 'stl.js');

// ---- what to build -------------------------------------------------
// Defaults here; any of them can be overridden on the command line,
// e.g. `node tools/tpms_batch.mjs --res 128 --out some/dir`.
const flag = (name, dflt) => {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return dflt;
  const v = process.argv[i + 1];
  if (v === undefined) throw new Error(`--${name} needs a value`);
  return typeof dflt === 'number' ? Number(v) : v;
};

const CELLS = flag('cells', 2);           // 2x2x2 unit cells
const RES = flag('res', 64);              // samples per cell
const OFFSET = flag('offset', 0);         // 0 is the canonical surface
const THICKNESS = flag('thickness', 0.02);  // model units; see --wall
const CLIP = flag('clip', 1);             // ball of the block's half-width
const RIM = flag('rim', 0.025);           // tube radius along the cut
const SIZE_MM = flag('size', 75);         // longest side when printed
// ONE SCALE FOR THE WHOLE SET.
//
// Scaling each surface so its own bounding box is SIZE_MM does not
// give a set of the same size -- it gives a set of the same BOX. The
// clip is a ball, and a surface only touches the ball's extreme
// points if it happens to pass near them. Schwarz P reaches 91.9% of
// the ball's diameter and Fischer-Koch S reaches 102.3% (the rim tube
// and the wall push it past), so scaling each to its own box blows
// Schwarz P up by 11.3% relative to Fischer-Koch S: its unit cell
// comes out 40.8 mm against 36.6 mm, and its wall 0.408 mm against
// 0.366 mm, from identical settings.
//
// With --uniform (the default) every surface is scaled by the same
// factor, chosen so the largest object in the set is exactly SIZE_MM.
// Unit cell and wall thickness are then identical across the set and
// the surfaces are honestly comparable; the smaller ones simply come
// out smaller, which is the truth about them. --no-uniform restores
// per-file scaling.
const UNIFORM = !process.argv.includes('--no-uniform');
// --wall asks for a wall in MILLIMETRES and works backwards to the
// thickness in model units that produces it. That is not a division:
// thickening pushes the surface out, so it changes the very span the
// scale is derived from, and the answer depends on itself. The
// dependence is near enough linear to solve by iterating a two-point
// fit, which `calibrate` below does at a coarse resolution -- the
// span moves by 0.06% between 16 samples and 64, far less than the
// wall tolerance of any printer.
//
// With --wall the scale is fixed by the WALL rather than by the size:
// the requested wall comes out exact and the largest object lands
// within a fraction of a percent of --size. That is the right way
// round when the wall is what has to clear a nozzle.
const WALL_MM = flag('wall', 0);
const CAL_RES = flag('calres', 16);

const OUT = flag('out',
  'C:/Users/dkrid/Projects/2026_07_21_Math_Art/dev/tpms-stl');

/** The widest span in the set, in model units, at a given thickness. */
function widestSpan(kinds, thickness, res) {
  let widest = 0;
  for (const kind of kinds) {
    const out = buildSceneFor(kind, thickness, res);
    const p = out.exported.positions;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        if (p[i + k] < lo[k]) lo[k] = p[i + k];
        if (p[i + k] > hi[k]) hi[k] = p[i + k];
      }
    }
    widest = Math.max(widest, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  }
  return widest;
}

/** The model-unit thickness whose printed wall is `wallMM`, with the
 *  largest object landing on `SIZE_MM`. */
function calibrate(kinds, wallMM) {
  const probes = [0.02, 0.04];
  const spans = probes.map((t) => widestSpan(kinds, t, CAL_RES));
  const b = (spans[1] - spans[0]) / (probes[1] - probes[0]);
  const a = spans[0] - b * probes[0];
  // t * SIZE / span(t) = wall, with span(t) = a + b t
  const t = wallMM * a / (SIZE_MM - wallMM * b);
  console.log(`calibration at ${CAL_RES} samples: span = ${a.toFixed(4)} `
    + `+ ${b.toFixed(4)} x thickness; for a ${wallMM} mm wall, `
    + `thickness ${t.toFixed(6)}`);
  return t;
}
// --resume picks up where an interrupted run stopped. At 128 samples
// a surface takes a minute and half a gigabyte, so a run that dies
// three from the end should not start again from the beginning.
const RESUME = process.argv.includes('--resume');

/** A binary STL says how many triangles it holds in bytes 80..84. */
function trianglesIn(file) {
  const fd = openSync(file, 'r');
  const head = Buffer.alloc(84);
  readSync(fd, head, 0, 84, 0);
  closeSync(fd);
  return head.readUInt32LE(80);
}
mkdirSync(OUT, { recursive: true });

// The names differ by characters a filename cannot carry -- there is
// a Fischer-Koch Y and a Fischer-Koch +-Y, and a C(Y) and a C(+-Y) --
// so the sign is spelled out rather than stripped. It was stripped at
// first, and two of the twenty-two silently overwrote the other two.
const slug = (s) => s
  .replace(' (nodal approximation)', '')
  .replace(/±/g, 'pm-')
  .replace(/\*/g, 'star')
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, '-')
  .replace(/^-|-$/g, '');
const taken = new Set();
const uniqueSlug = (name, kind) => {
  let s = slug(name);
  if (taken.has(s)) s += '-' + kind.toLowerCase();
  if (taken.has(s)) throw new Error(`two surfaces want the filename ${s}`);
  taken.add(s);
  return s;
};

const kinds = Object.keys(T.TPMS).filter((k) => T.TPMS[k][2]);

function buildSceneFor(kind, thickness, res) {
  return T.buildScene({
    kind, cells: CELLS, res, offset: OFFSET,
    thickness, clip: CLIP, rim: RIM,
  });
}

const THICK = WALL_MM > 0 ? calibrate(kinds, WALL_MM) : THICKNESS;

console.log(`${kinds.length} surfaces · ${CELLS}x${CELLS}x${CELLS} cells · `
  + `${RES} samples/cell · thickness ${THICK.toFixed(6)} · clip ${CLIP} · `
  + `rim ${RIM} · ${SIZE_MM} mm\n`);

const rows = [];
for (const kind of kinds) {
  const t0 = Date.now();
  const name = T.prettyLabel(kind).replace(' (nodal approximation)', '');
  const file = join(OUT, `${uniqueSlug(name, kind)}.stl`);
  if (RESUME && existsSync(file)) {
    const r = { kind, name, file, tris: trianglesIn(file),
                mb: +(statSync(file).size / 1e6).toFixed(1), kept: true };
    rows.push(r);
    console.log(`${name.padEnd(28)} ${String(r.tris).padStart(9)} tris  `
      + `${r.mb} MB  (already on disk, kept)`);
    continue;
  }
  const out = buildSceneFor(kind, THICK, RES);
  // thicknessMM 0: the sheet already has a thickness, so there is
  // nothing for the exporter to wall -- it would only double it
  const built = buildBinarySTLFromMesh(out.exported.positions,
                                       out.exported.indices,
                                       { sizeMM: SIZE_MM, thicknessMM: 0, name });
  const buf = Buffer.from(await built.blob.arrayBuffer());
  writeFileSync(file, buf);

  // How thick the wall actually comes out, in millimetres: the model
  // is scaled so its longest side is SIZE_MM, and the thickness rides
  // along with it. This is the number that decides printability.
  const mmPerUnit = Math.max(...built.mm) / Math.max(
    ...(() => {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      const p = out.exported.positions;
      for (let i = 0; i < p.length; i += 3) {
        for (let k = 0; k < 3; k++) {
          if (p[i + k] < lo[k]) lo[k] = p[i + k];
          if (p[i + k] > hi[k]) hi[k] = p[i + k];
        }
      }
      return [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    })());
  const openEdges = boundaryEdges(out.exported.indices).length;

  rows.push({
    kind, name, file,
    span: Math.max(...built.mm) / mmPerUnit,   // in model units
    tris: built.triangles,
    mm: built.mm.map((v) => +v.toFixed(1)),
    wallMM: +(THICK * mmPerUnit).toFixed(4),
    wireMM: +(2 * RIM * mmPerUnit).toFixed(2),
    rimLoops: out.stats.rimLoops,
    openEdges,
    mb: +(buf.length / 1e6).toFixed(1),
    secs: +((Date.now() - t0) / 1000).toFixed(1),
  });
  const r = rows[rows.length - 1];
  console.log(`${name.padEnd(28)} ${String(r.tris).padStart(9)} tris  `
    + (UNIFORM ? ''
               : `${r.mm.join('x').padEnd(18)} wall ${r.wallMM} mm  `
                 + `wire ${r.wireMM} mm  `)
    + `${r.rimLoops} loops  ${r.openEdges ? r.openEdges + ' OPEN EDGES' : 'closed'}  `
    + `${r.mb} MB  ${r.secs}s`);
}

// The common scale is applied afterwards, by rescaling the files in
// place: a binary STL is a flat array of floats, so this is a pass of
// arithmetic over bytes rather than a second meshing run, which at
// these resolutions would be another twenty-five minutes.
if (UNIFORM && rows.length) {
  const spans = rows.map((r) => r.span).filter((v) => v > 0);
  if (spans.length !== rows.length) {
    console.log('\nnot rescaling: some rows came from --resume and carry '
      + 'no span. Re-run without --resume for one scale across the set.');
  } else {
    const biggest = Math.max(...spans);
    // mm per model unit, from the wall when one was asked for
    const S = WALL_MM > 0 ? WALL_MM / THICK : SIZE_MM / biggest;
    console.log(`\none scale for the set: ${S.toFixed(6)} mm per model unit`
      + ` (widest span ${biggest.toFixed(4)}, so the largest object is `
      + `${(biggest * S).toFixed(2)} mm)`);
    for (const r of rows) {
      const f = S * r.span / SIZE_MM;       // each file is at SIZE_MM/span
      if (Math.abs(f - 1) < 1e-9) continue;
      const buf = Buffer.from(readFileSync(r.file));
      const n = buf.readUInt32LE(80);
      for (let t = 0; t < n; t++) {
        const off = 84 + t * 50 + 12;       // past the normal
        for (let k = 0; k < 9; k++) {
          buf.writeFloatLE(buf.readFloatLE(off + k * 4) * f, off + k * 4);
        }
      }
      writeFileSync(r.file, buf);
      r.mm = r.mm.map((v) => +(v * f).toFixed(2));
      r.wallMM = +(r.wallMM * f).toFixed(4);
      r.wireMM = +(r.wireMM * f).toFixed(3);
    }
    const w = rows.map((r) => r.wallMM);
    console.log(`  wall ${Math.min(...w).toFixed(4)} mm on every surface; `
      + `largest object ${Math.max(...rows.map((r) => Math.max(...r.mm))).toFixed(2)} mm`);
  }
}

writeFileSync(join(OUT, 'index.json'), JSON.stringify(rows, null, 2));
const total = rows.reduce((a, r) => a + r.mb, 0);
console.log(`\n${rows.length} files, ${total.toFixed(0)} MB, in ${OUT}`);
const bad = rows.filter((r) => r.openEdges);
console.log(bad.length ? `${bad.length} with open edges: `
  + bad.map((r) => r.name).join(', ') : 'every mesh is closed');
