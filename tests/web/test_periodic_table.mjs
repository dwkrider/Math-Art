// Headless checks for the periodic table of polyhedra.
//
//     node tests/web/test_periodic_table.mjs
//
// The table's authority is COMPLETENESS -- it claims to be the union of
// five closed classifications, so a solid quietly missing from it is the
// one failure that matters and the one nothing else would report. The
// page would look perfectly fine with 120 of the 127.
//
// Also checks the table has not gone stale: it is generated from the
// records by tools/build_periodic_table.py, and a database rebuild that
// adds or reclassifies a solid leaves the shipped JSON describing the
// previous state, which is silent.
import fs from 'node:fs';

const HERE = new URL('.', import.meta.url);
const table = JSON.parse(fs.readFileSync(
  new URL('../../web/periodic-table.json', HERE), 'utf8'));
const index = JSON.parse(fs.readFileSync(
  new URL('../../data/polyhedra/index.json', HERE), 'utf8')).entries;

let fail = 0;
const ok = (c, m) => { if (!c) { fail++; console.log('  FAIL ' + m); }
                       else console.log('  ok   ' + m); };

console.log('1. it is the complete union of the closed classifications');
const FAMS = table.families;
const want = index.filter((e) => (e.families || []).some((f) => FAMS.includes(f)));
const have = new Set(table.solids.map((s) => s.slug));
ok(want.length === table.solids.length,
   `${want.length} solids in those families, ${table.solids.length} in the table`);
const missing = want.filter((e) => !have.has(e.slug));
ok(missing.length === 0,
   `none missing${missing.length ? ': ' + missing.slice(0, 5).map((e) => e.slug) : ''}`);
const extra = [...have].filter((s) => !want.some((e) => e.slug === s));
ok(extra.length === 0, `no solid in the table that is not in a family`);

console.log('2. the counts each classification is famous for');
const byKind = {};
for (const s of table.solids) byKind[s.kind] = (byKind[s.kind] || 0) + 1;
for (const [kind, n] of [['Platonic', 5], ['Archimedean', 13],
                         ['Catalan', 13], ['Johnson', 92],
                         ['Kepler-Poinsot', 4]]) {
  ok(byKind[kind] === n, `${kind}: ${byKind[kind]} (expected ${n})`);
}

console.log('3. EVERY scheme places EVERY solid');
// A scheme is a rearrangement, never a filter. A column key missing from
// a scheme's order would drop its solids off the page silently -- the
// table would simply look shorter.
ok(table.schemes.length >= 2, `${table.schemes.length} column schemes`);
for (const sc of table.schemes) {
  const keys = new Set(sc.columns.map((c) => c.key));
  const placed = table.solids.filter((s) => keys.has(s.columns[sc.key]));
  ok(placed.length === table.solids.length,
     `${sc.label}: places all ${table.solids.length} (${placed.length})`);
  const sum = sc.columns.reduce((n, c) => n + c.count, 0);
  ok(sum === table.solids.length, `${sc.label}: counts sum to ${sum}`);
  for (const c of sc.columns) {
    const n = table.solids.filter((s) => s.columns[sc.key] === c.key).length;
    if (n !== c.count) {
      fail++;
      console.log(`  FAIL ${sc.label}/${c.key}: ${n} vs stated ${c.count}`);
    }
  }
}
ok(table.schemes.some((s) => s.key === table.default_scheme),
   `default scheme ${table.default_scheme} exists`);

console.log('4. a noble column, where claimed, really is closed');
// The claim is that nothing in it can be made more uniform in the terms
// the scheme groups by. Only checked where a scheme makes it.
for (const sc of table.schemes.filter((s) => s.noble)) {
  const noble = table.solids.filter((s) => s.columns[sc.key] === sc.noble);
  ok(noble.length > 0, `${sc.label}: the noble column is populated`);
  ok(noble.every((s) => s.transitivity.every(Boolean)),
     `${sc.label}: all of them transitive on vertices, edges AND faces`);
  const outside = table.solids.filter(
    (s) => s.columns[sc.key] !== sc.noble && s.transitivity.every(Boolean));
  ok(outside.length === 0,
     `${sc.label}: no fully-transitive solid left outside it (${outside.length})`);
}
const reg = table.solids.filter((s) => s.transitivity.every(Boolean));
ok(reg.length === 9, `9 fully-transitive solids in all (${reg.length})`);
ok(new Set(reg.map((s) => s.kind)).size === 2,
   'exactly the Platonic and Kepler-Poinsot solids');

console.log('5. every cell can draw a thumbnail');
let noThumb = 0;
for (const s of table.solids) {
  const p = new URL(`../../web/thumbs/polyhedra/${s.slug}.png`, HERE);
  if (!fs.existsSync(p)) { noThumb++; if (noThumb <= 3) console.log('    ' + s.slug); }
}
ok(noThumb === 0, `all ${table.solids.length} have a tile`);

console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
