# Polyhedron database

A JSON database of polyhedra: one document per solid, plus an `index.json`
that enumerates them. It exists because the established public databases each
carry a *different* slice of what a polyhedron is, and none carries all of it —
netlib in particular has excellent geometry and almost no classification.

```
data/polyhedra/
  index.json                     enumerates every record; enough to search without opening files
  README.md                      this file
  schema/
    polyhedron.schema.json       the record schema (JSON Schema draft 2020-12)
    index.schema.json            the index schema
  solids/
    platonic/cube.json
    archimedean/truncated-icosahedron.json
    kepler-poinsot/small-stellated-dodecahedron.json
    catalan/  johnson/  uniform/  uniform-dual/  prism/  zonohedron/  ...
```

Family folders are for humans; `slug` is unique across the **whole** database,
and `index.json` is the only thing that resolves a slug to a path.

## What the existing databases carry

| | netlib | McCooey | Hart / `finnp` | Wolfram | Polytope Wiki | **here** |
|---|---|---|---|---|---|---|
| vertices, faces | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| exact algebraic values | ✅ (`bc`) | ✅ | ❌ float | ✅ | ✅ | ✅ |
| net / unfolding | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ (out of scope) |
| dihedral angles | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| circum/mid/inradius | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| volume, area | ❌ | ✅ (V) | ❌ | ✅ | ✅ | ✅ |
| **symmetry group** | ❌ | ✅ (Schoenflies only) | ❌ | ✅ | ✅ | ✅ (5 notations) |
| **orbifold notation** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Schläfli / Wythoff | partial | ❌ | ❌ | ✅ | ✅ | ✅ |
| Coxeter diagram | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Conway notation | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| χ, genus, density | ❌ | ❌ | ❌ | partial | ✅ | ✅ |
| transitivity / orbits | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| cross-reference IDs | ❌ | ❌ | ❌ | ❌ | partial | ✅ |
| machine-readable | ✅ own format | ❌ HTML/txt | ✅ JSON | ❌ proprietary | ❌ wiki | ✅ JSON |

The gap this database fills is the middle block: **classification and symmetry
as first-class, machine-readable fields**, attached to geometry that is exact
and verified.

## The attribute subset

Not everything the sources carry is worth keeping. Nets and Wolfram's
`MeanCylindricalRadius` are omitted; the following eight groups are kept.

**1. Identity** — `slug`, `name`, `alternate_names`, `families`.

**2. `ids`** — cross-references, so a record can be reconciled against every
source above: `wenninger` (W1–W119), `uniform` (U1–U75), `johnson` (J1–J92),
`netlib` (0–141), `coxeter_clm` (the 1954 figure numbers), `bowers`,
`wikipedia`, `wolfram`, `mccooey`. This is what makes the database auditable
rather than another undocumented dump.

**3. `notation`** — `schlafli`, `wythoff`, `coxeter_diagram`, `conway`,
`vertex_configuration`, `face_configuration`. Compact and generative: the
Wythoff symbol plus a Schwarz triangle *reconstructs* a uniform solid, so this
group doubles as a build recipe.

**4. `symmetry`** — the headline gap in netlib. Carried in five notations
because the sources disagree on which to use, and Cromwell's Appendix I is the
concordance between them: `schoenflies` (`Ih`), `orbifold` (`*532`), `coxeter`
(`[5,3]`), `hermann_mauguin` (`m-3̄-5`), plus `order`, the rotation subgroup,
`chiral`, `transitivity` (isogonal/isotoxal/isohedral), and `orbits`.
Orbifold notation is included deliberately: the Pattern Engine
(`math_art/patterns/groups.py`) already keys its groups that way, so a record
joins directly to the code.

**5. `combinatorics`** — `counts`, `faces_by_type`, `vertex_types`,
`euler_characteristic`, `orientable`, `genus`, `density`, `convex`,
`self_dual`, `dual`, `chiral_partner`. χ, genus and density are what let star
and toroidal solids sit in the same table as the Platonics instead of needing a
separate one.

**6. `metrics`** — `normalization`, edge length(s), `circumradius`,
`midradius`, `inradius` (per face type), `surface_area`, `volume`,
`isoperimetric_quotient`, `dihedral_angles` (per face-type pair). Every scalar
is a *measure object* — see below.

**7. `geometry`** — `centroid`, `orientation`, named `constants`, `vertices`,
`vertices_exact` (the exact table, written against those constants in McCooey's
style), `faces`, and `face_angles` — the interior angles of each face orbit.
Face angles are what Wenninger's *Dual Models* tabulates for every solid, and
they are what distinguishes the Catalans, the duals and the Johnson solids,
whose faces are not regular. Edges are derivable from face cycles and omitted.

**8. `provenance`** — `coordinates` (how this file's numbers were obtained),
`sources` (real mathematical attribution), `verified` (recomputed claims),
`cross_checked` (which independent sources confirmed the geometry).

**9. Grouping** — `face_groups`, `vertex_groups`, `edge_groups`.
`faces_by_type` says how *many* faces of each kind there are; these say
*which*: by symmetry orbit (the canonical partition), by polygon type, by
coplanarity, and by antipodal pairing. All computed from the detected group.

**10. `space_filling`** — whether and how the solid tiles space, with
parameters. A boolean alone would be wrong: the octahedron does not tile space
by itself but does with tetrahedra, and tiling by *translations* alone (a
parallelohedron — Fedorov proved there are exactly five) is strictly stronger.
The lattice basis makes the claim falsifiable, and the validator checks that
`|det(basis)| == volume × cells_per_lattice_point`.

Also carried, each computed rather than curated: `metrics.angular_defect` and
`metrics.solid_angle` per vertex orbit (per-orbit deliberately — Descartes
makes the *total* 4π for every genus-0 solid, so a total says nothing);
`combinatorics.vertex_density` (Har'El's *d*, which discriminates only among
the star uniforms, where *d* = 0 marks exactly those Wythoff cannot build
directly); `combinatorics.convex_hull`; `combinatorics.dual_unbounded`; and
`metrics.biscribed`.

Plus a small **`construction`** block pointing back at the `math_art/` module
and operator that build the solid.

## Scope: reference-only

This database is a research and documentation resource. `data/` sits outside
`math_art/`, so it is **not** packaged into the extension zip, and the
generators do not read from it — they keep their own data modules. The
`construction` block is a cross-reference back to the code, not a dependency in
either direction. Revisit only if having a single source of truth becomes worth
a build step that bakes the JSON into a Python module.

## Conventions

**Measure objects.** Every scalar is `{"exact": "...", "value": 1.234}`.
Following netlib and McCooey, the closed form is kept — but in a small fixed
language (integers, `+ - * /`, parentheses, `sqrt` `acos` `asin` `atan` `sin`
`cos` `tan`, `pi`, `phi`) rather than netlib's embedded `bc` or McCooey's prose.
The validator evaluates every `exact` string and fails if it disagrees with its
own `value`, so the two can never drift apart.

**Normalization.** `edge_length_1` where all edges are equal — this is
McCooey's convention and it keeps the radicals clean. Where edges differ
(Catalans, canonical forms) use `midradius_1`, because reciprocating in the
unit sphere then makes a dual pair *exactly* dual. Hart normalizes everything
to midradius 1; netlib does neither consistently. The field is explicit in
every record, and the recorded radii make rescaling one multiplication.

This is a *storage* convention. It is not the repo's "fit a 2×2×2 box" render
convention, which is a display transform applied downstream.

**Orientation.** Recorded in prose per record (icosahedral solids use the
standard golden-ratio frame with 2-fold axes on the coordinate axes). Pinning
it is what makes two records comparable and compounds constructible.

**Star faces** are stored as their true winding cycle — a `{5/2}` is a 5-cycle
in step-2 order, never a convex outline — and windings must be *coherent*:
every edge traversed once in each direction. This is the single easiest thing
to get wrong; the validator checks it.

**Area and volume of self-intersecting solids** are `null`, with a
`measures_note` saying why. Net, density-weighted and outer-solid-region
readings all differ, and quietly picking one is how a database becomes wrong.

**Coordinates are derived, not transcribed.** Hart's data is noncommercial-use;
McCooey's and netlib's carry their own terms. This repository is published
publicly, so records are computed from the mathematics, cross-checked against
those sources, and cite them. The existing `math_art/polyhedra/` data modules
already work this way.

## Maintenance

```
python tools/polydb_validate.py
```

Recomputes, per record: counts against the stored arrays; V − E + F against the
stated χ; χ against the stated genus; that every edge is shared by exactly two
faces with opposite orientation; face planarity; the centroid; circumradius,
midradius, inradius, edge length and every dihedral angle against the geometry;
every `exact` string against its `value`; and that dual pointers are mutual with
reciprocal V/E/F. It also checks the index agrees with each record.

The `provenance.verified` block records what passed. It is **not** hand-written
— it is what the validator confirms, and it caught three real errors (an
incoherent star winding and two wrong closed forms) in the three seed records
below.

## The corpus

**314 records**, all passing the validator. Built by `tools/polydb_build.py`
in stages:

| Stage | Records | What |
|---|---|---|
| `uniform` | 75 | every uniform polyhedron U1-U75; Platonic, Archimedean and Kepler-Poinsot included |
| `dual` | 56 | their duals, the 13 Catalans among them |
| `johnson` | 92 | all of J1-J92 |
| `prism` | 36 | prisms, antiprisms, dipyramids, trapezohedra, n = 3..12 |
| `biscribed` | 39 | McCooey's biscribed solids, chiral and non-chiral |
| `toroid` | 6 | genus-1 solids, Csaszar and Szilassi among them |
| `zonohedron` | 1 | the rhombic enneacontahedron (the others are already records) |
| `geodesic` | 9 | class-I geodesic spheres, frequency 2-4, on three bases |
| `link` | -- | post-pass resolving cross-references |

Four kinds of record are deliberately absent, each for a stated reason:

- **Duals of the 9 hemipolyhedra, and of U75.** Their faces pass through the
  centre, so the polar dual has vertices at infinity and is not a bounded
  polyhedron. The parent names its dual and sets `dual_unbounded`.
- **Anything that would duplicate an existing record.** The square prism *is*
  the cube; the triangular antiprism *is* the octahedron; three of the four
  zonohedra built here are already present as a Platonic or a Catalan. Slugs
  are unique, and `emit` now raises rather than overwriting.
- **The knotted toroid**, whose faces are 5.7e-7 out of plane. That is a
  polyhedral *surface*, not a polyhedron, and this database certifies
  planarity.
- **Exact vertex tables for the snubs and J84-J92**, whose coordinates are
  roots of irreducible cubics and higher. Those carry `vertices_exact: null`
  rather than a fabricated value.

### Not attempted, and why

- **Compounds.** A compound is a *set* of polyhedra, not one: its surface is
  disconnected, so the closed-manifold check and the whole
  one-solid-one-record model do not apply. Supporting them means a
  `components` model first.
- **Stellations.** The engine exists (`math_art/general_stellation.py`), but
  the counts are unstable in the literature -- the triakis tetrahedron is
  variously 21, 28 or 138 stellations depending on whose rules are applied,
  and Cromwell notes some legitimate stellations are not even manifolds. This
  needs a per-source citation model, not a number.

## Building it

```sh
bash tools/polydb_fetch.sh netlib $(seq 0 141)     # populate the source cache
bash tools/polydb_fetch.sh mccooey Cube Icosahedron ...
python tools/polydb_build.py                       # every stage
python tools/polydb_build.py uniform dual          # selected stages
python tools/polydb_build.py crosscheck            # re-verify against sources
python tools/polydb_validate.py                    # the gate
```

The `crosscheck` stage is separate from the build on purpose: it depends only
on what is in the source cache, so more sources can be fetched and the
verification re-run without rebuilding any geometry.

Source page names do NOT follow from ours mechanically — McCooey tags
chirality (`BiscribedLsnubCube`), writes *Dipyramid* where Johnson's list says
*Bipyramid*, suffixes Johnson numbers (`ElongatedPentagonalDipyramidJ16`), and
paginates his index behind `window.location.href` buttons rather than plain
links. The mapping is therefore built once by crawling his index and matching
normalised names, and cached in `.polydb_cache/_match.json`, rather than
guessed per record.

`tools/polydb/` holds the pieces: `symmetry.py` (point-group detection and
orbits), `measure.py` (metrics, vertex figures, grouping schemes), `exact.py`
(closed-form recovery and minimal polynomials), `refine.py` (re-solving a
vertex-transitive solid to machine precision), `crosscheck.py` (reading the
cached sources), `curation.py` (the literature facts), `harel.py` (Har'El's
tables).

Two things that will bite anyone extending this:

- The generator modules must be imported **flat** (`sys.path.insert(0,
  'math_art')`). The package `__init__` imports `bpy` and will fail outside
  Blender.
- `research/**` is gitignored, so **Grep silently returns nothing there**
  unless you pass `--no-ignore`. This is not a missing corpus; it is a missing
  flag.

## Two corrections worth remembering

- **The stored snub coordinates carried only six decimals**, leaving faces
  ~1e-7 out of plane. A snub is determined by its generator point, so
  `refine.py` re-solves it as an orbit under the exact rotation group; all
  twelve now sit at ~3e-16.
- **Counts do not identify a solid.** Stellation preserves V, E and F, so the
  great icosahedron has the icosahedron's counts *and* its vertex set. Hull
  matching that ignores convexity picks the wrong twin, and marks star solids
  as their own hull. The convexity constraint is what makes it right.

## Local sources

Beyond the online databases, the metadata design draws on this repo's
(gitignored) `research/` corpus:

- **Cromwell, *Polyhedra* (1997), Appendix I** — the concordance between
  Schoenflies, Coxeter, Shubnikov, Conway–Thurston orbifold and Fejes Tóth
  notations. This table is the direct source for the five-notation `symmetry`
  block.
- **Har'El, "Uniform Solution for Uniform Polyhedra", *Geom. Dedicata* 47
  (1993), Appendix II, Tables 4–8** — per-polyhedron Wythoff symbol, vertex
  configuration, χ, density, dual name, and cross-references to Coxeter et al.
  and Wenninger. The closest existing thing to a record layout, and the model
  for the `notation` + `combinatorics` + `ids` grouping.
- **Coxeter, Longuet-Higgins & Miller (1954)** — the canonical enumeration and
  figure numbering.
- **Wenninger, *Polyhedron Models* (1971) / *Dual Models* (1983)** — W-numbers,
  with the Coxeter et al. numbers cross-listed in the model list.
- **Conway, Burgiel & Goodman-Strauss, *The Symmetries of Things*** — orbifold
  notation, already the primary key in `math_art/patterns/`.
