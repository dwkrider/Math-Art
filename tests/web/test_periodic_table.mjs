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

console.log('3. every solid has a column, and the columns add up');
ok(table.solids.every((s) => s.column), 'every solid placed in a column');
const sum = table.columns.reduce((n, c) => n + c.count, 0);
ok(sum === table.solids.length, `column counts sum to ${sum}`);
for (const c of table.columns) {
  const n = table.solids.filter((s) => s.column === c.key).length;
  if (n !== c.count) { fail++; console.log(`  FAIL ${c.key}: ${n} vs ${c.count}`); }
}

console.log('4. the noble gases are the regular solids, and only those');
const noble = table.solids.filter((s) => s.column === 'regular');
ok(noble.length === 9, `9 in the regular column (${noble.length})`);
ok(noble.every((s) => s.transitivity.every(Boolean)),
   'all transitive on vertices, edges AND faces');
const regularElsewhere = table.solids.filter(
  (s) => s.column !== 'regular' && s.transitivity.every(Boolean));
ok(regularElsewhere.length === 0,
   `no fully-transitive solid left outside it (${regularElsewhere.length})`);
ok(new Set(noble.map((s) => s.kind)).size === 2,
   'they are exactly the Platonic and Kepler-Poinsot solids');

console.log('5. every cell can draw a thumbnail');
let noThumb = 0;
for (const s of table.solids) {
  const p = new URL(`../../web/thumbs/polyhedra/${s.slug}.png`, HERE);
  if (!fs.existsSync(p)) { noThumb++; if (noThumb <= 3) console.log('    ' + s.slug); }
}
ok(noThumb === 0, `all ${table.solids.length} have a tile`);

console.log(fail ? `\nRESULT: ${fail} FAILURE(S)` : '\nRESULT: OK');
process.exit(fail ? 1 : 0);
