# Polyhedron database — handoff

Single entry point for anyone (human or agent) picking this up. Read this,
then `README.md` for the format spec. Everything below is re-runnable.

## State

**314 records, 0 errors, 0 warnings.**

```sh
python tools/polydb_build.py      # rebuild everything, ~1 min
python tools/polydb_validate.py   # the gate
```

| Stage | N | Notes |
|---|---|---|
| `uniform` | 75 | U1–U75; Platonic, Archimedean, Kepler-Poinsot included |
| `dual` | 56 | Catalans among them |
| `johnson` | 92 | J1–J92 |
| `prism` | 36 | prisms, antiprisms, dipyramids, trapezohedra, n = 3..12 |
| `biscribed` | 39 | McCooey's chiral + non-chiral |
| `toroid` | 6 | genus 1 |
| `zonohedron` | 1 | the rest are already records |
| `geodesic` | 9 | class-I, frequency 2–4 |
| `crosscheck` | — | re-verify against sources; needs no geometry rebuild |
| `link` | — | resolves convex-hull and biscribed-form cross-references |

Cross-check: **201 of 203 checkable records confirmed against McCooey**;
111 have no published source. The 2 disagreements (`great-rhombidodecacron`,
`small-dodecicosacron`) are recorded honestly, not forced to agree.

## Decisions already made — do not re-litigate

- Exact values: **published tables first**, cross-validated against our own
  build; PSLQ only as fallback. Every value carries a `source`.
- Published tables are an **oracle**, never republished. McCooey's site is
  "All rights reserved", so his *compilation* stays out; the stored vertex
  tables are our own derivation, and he is cited for scalar closed forms
  (mathematical facts). `.polydb_cache/` is gitignored for this reason —
  **never commit it.**
- The database is **reference-only**: `data/` sits outside `math_art/` and is
  not packaged into the extension zip.
- Delivery is staged per family, validated at each stage.

## Traps that have already cost time

- Import generator modules **flat** (`sys.path.insert(0, 'math_art')`). The
  package `__init__` imports `bpy` and fails outside Blender.
- `research/**` is gitignored, so **Grep returns zero hits there** unless you
  pass `--no-ignore`. This silently poisoned several searches.
- **Python cannot fetch** — the project guard blocks outbound connections.
  Use `tools/polydb_fetch.sh` (Git Bash curl; the Windows system curl fails
  with exit 61 on these servers).
- Generic **sympy cannot do the Wythoff orbit** symbolically; radical trees
  blow up and it will not finish a tetrahedron.
- **Counts do not identify a solid.** Stellation preserves V/E/F, so the great
  icosahedron has the icosahedron's counts *and* its vertices. Hull matching
  must require convexity.
- **Vertices do not identify a solid either.** Compare faces and dihedral
  angles too, or the rhombicosidodecahedron "matches" several 60-vertex star
  uniforms.
- **Newell normals vanish on crossed faces** (antiparallelograms). Use the
  least-squares plane fallback, or star duals look degenerate.
- **`acos` is stationary at 0 and pi** — compare dihedral exacts in cosine
  space, not angle space.
- **`slugify` strips parenthesised text** (so "Square Pyramid (J1)" slugs
  cleanly). Names differing *only* by a parenthesised number collide; `emit`
  now raises rather than overwriting.
- **Symmetry must be detected with faces.** `find_group(V)` alone gives the
  symmetry of the vertex *arrangement*, which can be strictly larger than the
  solid's — the compound of 5 tetrahedra reads I_h (120) instead of the
  correct chiral I (60). Pass `F=`.
- **Weld coincident vertices before indexing compounds**: 10 tetrahedra
  occupy 40 slots but 20 distinct points.

## Deliberately absent — do not "fix"

- Duals of the 9 hemipolyhedra and U75 (unbounded — vertices at infinity).
- Anything duplicating an existing record (square prism = cube, triangular
  antiprism = octahedron).
- The knotted toroid (faces 5.7e-7 out of plane: a polyhedral *surface*).
- Exact vertex tables for the snubs and J84–J92 (roots of irreducible cubics
  and higher; no radical form exists).

## Open work

**Compounds** — designed, not built. A compound is a *set* of polyhedra, so
the one-solid-one-record model needs extending: a `components` array, the
manifold check applied per component, chi as a documented sum. Note the
taxonomy has three incompatible kinds, so a catalogue number cannot be the
key:

1. *Closed enumerations* — Coxeter's 5 regular; Skilling (1976) 75 uniform,
   complete **for uniform compounds only** (dual pairs are not vertex-transitive
   and are absent by definition).
2. *Rule-generated* — Harman's ~40, identified by his `nX/mY` notation
   (n-fold axis of component group X on m-fold axis of compound group Y;
   i/c/t = icosahedral/cuboctahedral/tetrahedral). **Unpublished, 1974**;
   known via Hart, and the citation must say so.
3. *Continuously parameterised* — spin/cube-compound families with a free
   angle. No enumeration exists even in principle.

So: **identity = construction** (component + group + axis pair + phase), with
`enumeration` a cross-reference *array*. Completeness claims belong at family
level, always qualified. `research/plans/hart-polyhedra-implementation-plan.md`
§3.1 scopes the implementation.

**Stellations** — engine exists (`math_art/general_stellation.py`), but
literature counts are unstable (triakis tetrahedron: 21 / 28 / 138 by
different rules) and some legitimate stellations are not manifolds. Needs a
per-source citation model, not a number.

**More cross-checks** — `provenance.cross_checked` is populated only where a
source file is cached. `bash tools/polydb_fetch.sh mccooey <Stem> ...` then
`python tools/polydb_build.py crosscheck`.

**Unresolved literature conflict** — net counts: 43,380 is assigned to the
dodecahedron in `research/priorities/bridges-archive-review-detail.md` and to
the icosahedron on Wikipedia. Resolve before storing.

## Generator bugs this database found (all fixed)

Cross-validation against published tables found **12 wrong Johnson solids**
and 2 wrong compounds in `math_art/`. See the git log for
`regular_solids_generator.py` and `polyhedra/compounds.py`. The root cause in
every Johnson case was one thing: an orientation `shift` counted from a face's
arbitrary first-listed vertex. If you touch cupola placement, re-run
`python tools/polydb_build.py crosscheck` — it is the regression test.
