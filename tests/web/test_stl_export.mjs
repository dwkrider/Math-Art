// Headless checks for the STL export.
//
//     node tests/web/test_stl_export.mjs
//
// Run by tests/test_web.py when node is available.
//
// WHAT NEEDS CHECKING, AND WHY IT IS NOT OBVIOUS. A binary STL is a
// blob: nothing about it looks wrong until a slicer says so. Three
// properties are worth asserting, and each has already failed here:
//
//   * the file's own header count must match its length, or every
//     reader disagrees about how many triangles there are;
//   * the model must fit the cube it claims to -- the thickening adds
//     half a wall on each side, so the surface is scaled to
//     (size - wall) before it is offset;
//   * a thickened sheet must come out CLOSED and consistently wound.
//     The first version closed every hole and still wound the rim
//     backwards, so a boundary-edge count alone reported success. Only
//     the direction audit caught it.
import fs from 'node:fs';

globalThis.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
globalThis.Blob = class {
  constructor(parts) {
    this.parts = parts;
    this.size = parts.reduce((n, p) => n + (p.byteLength ?? p.length), 0);
  }
};
globalThis.window = globalThis;
globalThis.document = { createElement: () => ({ style: {} }) };

const HERE = new URL('.', import.meta.url);
const STL = await import(new URL('../../web/js/stl.js', HERE).href);
const V = await import(new URL('../../web/js/surface-viewer.js', HERE).href);
const MESH = new URL('../../web/surfaces/', HERE);

let fail = 0;
const ok = (c, m) => { if (!c) { fail++; console.log('  FAIL ' + m); }
                       else console.log('  ok   ' + m); };

const read = (slug) => {
  const p = new URL(`${slug}.json`, MESH);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null;
};

const SIZE = 200, WALL = 2;

console.log('1. file structure');
{
  const packed = read('sphere');
  const b = STL.buildBinarySTL(packed, { name: 'sphere', sizeMM: SIZE, thicknessMM: WALL });
  const buf = Buffer.concat(b.blob.parts.map((x) => Buffer.from(x)));
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  ok(buf.length === 84 + 50 * dv.getUint32(80, true), 'length matches the header count');
  ok(dv.getUint32(80, true) === b.triangles, 'header agrees with the reported count');
  ok(!buf.toString('ascii', 0, 5).startsWith('solid'),
     'header does not start with "solid" (which would be read as ASCII STL)');
}

console.log('2. it fits the cube it claims');
for (const slug of ['sphere', 'catenoid', 'gyroid', 'enneper-surface']) {
  const packed = read(slug);
  if (!packed) { console.log('  skip ' + slug); continue; }
  const b = STL.buildBinarySTL(packed, { name: slug, sizeMM: SIZE, thicknessMM: WALL });
  ok(Math.max(...b.mm) <= SIZE + 0.05,
     `${slug}: ${b.mm.map((v) => v.toFixed(1)).join(' x ')} mm fits ${SIZE}`);
}

console.log('3. thickening closes a sheet, and winds it consistently');
for (const slug of ['catenoid', 'enneper-surface', 'gyroid']) {
  const packed = read(slug);
  if (!packed) { console.log('  skip ' + slug); continue; }
  const raw = V.decodeMesh(packed);
  const m = STL.weld(raw.positions, raw.indices);
  const before = STL.auditMesh(m.indices);
  if (!before.holes) { console.log(`  skip ${slug} (already closed)`); continue; }
  const sol = STL.solidify(m.positions, m.indices, 0.02);
  const after = STL.auditMesh(sol.indices);
  ok(after.holes === 0, `${slug}: ${before.holes} open edges -> 0`);
  ok(after.flipped === 0, `${slug}: winding consistent (${after.flipped} bad)`);
  ok(after.nonManifold === 0, `${slug}: no edge shared by >2 faces`);
}

console.log('4. a closed surface is not thickened');
{
  const b = STL.buildBinarySTL(read('sphere'), { name: 's', sizeMM: SIZE, thicknessMM: WALL });
  ok(!b.thickened, 'the sphere is exported as the solid it already is');
  ok(b.audit.watertight, 'and it is watertight');
}

console.log('5. the seam weld does not fuse touching sheets');
{
  // Welding by position alone merges self-intersections into
  // non-manifold edges; it must only consider boundary vertices.
  const raw = V.decodeMesh(read('roman-surface') || read('boys-surface'));
  const before = STL.auditMesh(raw.indices);
  const w = STL.weld(raw.positions, raw.indices);
  const after = STL.auditMesh(w.indices);
  ok(after.nonManifold <= before.nonManifold + 2,
     `welding did not manufacture non-manifold edges (${before.nonManifold} -> ${after.nonManifold})`);
}

console.log('6. the audit counts what a Map-based count would');
{
  // auditMesh used to key two Maps on strings like "17:42", which V8
  // caps at 2^24 entries: a mesh past about 5.6 million triangles
  // threw instead of being audited, and those are precisely the
  // meshes nobody can check by eye. It now packs each edge into one
  // number and sorts. The obvious implementation is kept HERE, as the
  // thing to agree with -- a different algorithm, not a copy.
  const byMap = (indices) => {
    const und = new Map(), dir = new Map();
    for (let t = 0; t < indices.length; t += 3) {
      const tri = [indices[t], indices[t + 1], indices[t + 2]];
      for (let e = 0; e < 3; e++) {
        const a = tri[e], b = tri[(e + 1) % 3];
        const k = a < b ? a + ':' + b : b + ':' + a;
        und.set(k, (und.get(k) || 0) + 1);
        const d = a + '>' + b;
        dir.set(d, (dir.get(d) || 0) + 1);
      }
    }
    let holes = 0, nonManifold = 0, flipped = 0;
    for (const c of und.values()) {
      if (c === 1) holes++; else if (c > 2) nonManifold++;
    }
    for (const c of dir.values()) if (c > 1) flipped++;
    return { holes, nonManifold, flipped,
             watertight: holes === 0 && nonManifold === 0 && flipped === 0 };
  };
  const cases = [
    ['nothing at all', []],
    ['one open sheet', [0, 1, 2, 1, 3, 2]],
    ['two faces wound the same way along a shared edge', [0, 1, 2, 0, 1, 3]],
    ['four faces on one edge', [0, 1, 2, 0, 1, 3, 1, 0, 4, 1, 0, 5]],
  ];
  // random soups, which hit holes, non-manifold edges and reversed
  // windings all at once and in quantity
  let seed = 7;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (const [n, verts] of [[60, 12], [400, 40], [4000, 500]]) {
    const idx = [];
    for (let i = 0; i < n * 3; i++) idx.push(Math.floor(rnd() * verts));
    cases.push([n + ' random triangles over ' + verts + ' vertices', idx]);
  }
  let agree = 0;
  for (const [name, list] of cases) {
    const idx = Uint32Array.from(list);
    const got = JSON.stringify(STL.auditMesh(idx));
    const want = JSON.stringify(byMap(idx));
    if (got === want) agree++;
    else ok(false, name + ': audit says ' + got + ', a Map count says ' + want);
  }
  ok(agree === cases.length,
     'the audit agrees with a Map count on all ' + cases.length + ' meshes');
}

console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
