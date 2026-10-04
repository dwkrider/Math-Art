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

import { writeFileSync, mkdirSync, existsSync, statSync,
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
const THICKNESS = flag('thickness', 0.02);  // model units, before scaling
const CLIP = flag('clip', 1);             // ball of the block's half-width
const RIM = flag('rim', 0.025);           // tube radius along the cut
const SIZE_MM = flag('size', 75);         // longest side when printed

const OUT = flag('out',
  'C:/Users/dkrid/Projects/2026_07_21_Math_Art/dev/tpms-stl');
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
console.log(`${kinds.length} surfaces · ${CELLS}x${CELLS}x${CELLS} cells · `
  + `${RES} samples/cell · thickness ${THICKNESS} · clip ${CLIP} · `
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
  const out = T.buildScene({
    kind, cells: CELLS, res: RES, offset: OFFSET,
    thickness: THICKNESS, clip: CLIP, rim: RIM,
  });
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
    tris: built.triangles,
    mm: built.mm.map((v) => +v.toFixed(1)),
    wallMM: +(THICKNESS * mmPerUnit).toFixed(3),
    wireMM: +(2 * RIM * mmPerUnit).toFixed(2),
    rimLoops: out.stats.rimLoops,
    openEdges,
    mb: +(buf.length / 1e6).toFixed(1),
    secs: +((Date.now() - t0) / 1000).toFixed(1),
  });
  const r = rows[rows.length - 1];
  console.log(`${name.padEnd(28)} ${String(r.tris).padStart(9)} tris  `
    + `${r.mm.join('x').padEnd(18)} wall ${r.wallMM} mm  wire ${r.wireMM} mm  `
    + `${r.rimLoops} loops  ${r.openEdges ? r.openEdges + ' OPEN EDGES' : 'closed'}  `
    + `${r.mb} MB  ${r.secs}s`);
}

writeFileSync(join(OUT, 'index.json'), JSON.stringify(rows, null, 2));
const total = rows.reduce((a, r) => a + r.mb, 0);
console.log(`\n${rows.length} files, ${total.toFixed(0)} MB, in ${OUT}`);
const bad = rows.filter((r) => r.openEdges);
console.log(bad.length ? `${bad.length} with open edges: `
  + bad.map((r) => r.name).join(', ') : 'every mesh is closed');
