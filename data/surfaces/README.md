# Surface database

A JSON database of surfaces: one document per surface, plus an
`index.json` that enumerates them. Sibling to `data/polyhedra/`, and built on
the same principles — classification and symmetry as first-class,
machine-readable fields, attached to a definition that is exact and verified.

```
data/surfaces/
  index.json                     enumerates every record; enough to search without opening files
  README.md                      this file
  schema/
    surface.schema.json          the record schema (JSON Schema draft 2020-12)
    index.schema.json            the index schema
  surfaces/
    minimal/catenoid.json
    minimal-periodic/gyroid.json
    algebraic/barth-sextic.json
    quadric/  cmc/  constant-curvature/  ruled/  topological/  ...
```

Family folders are for humans; `slug` is unique across the **whole** database,
and `index.json` is the only thing that resolves a slug to a path.

**363 records, 298 implemented, 0 validator errors.**

Every record carries a real citation; 65 are independently cross-checked
against the local Ferréol mirror; and 296 of the 298 implemented records have
been built by *running their actual operator* and comparing the resulting mesh
against what the record claims.

## Why a surface database is not a polyhedron database

Six differences drive every design decision here. They are worth stating,
because several of them are places where copying `data/polyhedra/` would be
actively wrong.

**1. A polyhedron is a point; a surface is usually a family.** The cube is one
object. Costa–Hoffman–Meeks is a sequence indexed by genus; the Dupin cyclide's
ring, horn and spindle types are *regimes of one formula*; Schwarz P, the
gyroid and Schwarz D are three points of one Bonnet associate family. So
parameters are first-class (`definition.parameters`), and distinguished members
live in `specimens`.

**2. There is no exact coordinate table.** The polyhedron schema's centre of
gravity is `geometry.vertices_exact`. A surface has no finite vertex set — its
identity is the polynomial, the parametrisation, the Weierstrass data, the
nodal level function, or the energy-plus-boundary. **Meshes are not stored.**
A mesh is a rendering at a chosen resolution; the record carries a reproducible
recipe (`mesh_recipe`) and *measured* invariants instead. This is the same
argument by which `data/polyhedra` omits nets, one level up.

**3. Fidelity is a separate axis from exactness.** A definition can be
*evaluated* exactly, by numerical integral, or through a solved parameter —
and, orthogonally, it either **is** the surface or **approximates** it. The 23
nodal TPMS are approximations: `sin x cos y + sin y cos z + sin z cos x = 0`
defines a surface that is **not** the gyroid, and whose mean curvature is wrong
at the percent level. Recording `fidelity` is what lets the validator hold the
exact rows to 10⁻⁸ while holding the nodal rows to their own published residual.

**4. Symmetry needs five kinds of group, not one.** Point groups for bounded
surfaces, continuous groups for surfaces of revolution, rod groups for
singly-periodic ones, layer groups for doubly-periodic, and **space groups**
for the TPMS (the gyroid is *Ia*3̄*d*, No. 230). `symmetry.periodicity_rank`
(0–3) selects which notation applies. For `kind: "point"` the block is
field-for-field identical to polydb's, so the two databases join on symmetry.

**5. Topology is richer, and it is the field's primary organiser.** Beyond
χ/genus/orientability: the number and **type** of ends (planar, catenoidal,
helicoidal, Enneper-of-order-*k*), one-sidedness, completeness, and finite
total curvature. The minimal-surface literature indexes almost entirely by
(periodicity, genus, ends).

**6. Area and volume are frequently infinite.** A complete minimal surface has
infinite area; a triply-periodic one has finite area only per cell. So
`metrics` carries `area_per_cell` / `area_of_piece` with the piece named, and
`null` plus a `measures_note` otherwise — polydb's convention, needed far more
often.

## The taxonomy: facets, not a tree

Every source surveyed is a facet system pretending to be a tree. MathWorld
presents eleven sibling "topics", and its entries multiply-list constantly:
Enneper is Algebraic ∩ Minimal, the torus is Algebraic ∩ Closed ∩ Revolution,
the Möbius strip and Cayley cubic are Algebraic ∩ Ruled, the unduloid sits
under Revolution while the CMC topic has *one* entry. Two of the eleven are not
surface classes at all, and the second-largest is *Miscellaneous*. Mathcurve is
alphabetical with cross-links; the Virtual Math Museum organises by
construction method; the minimal-surface literature by (periodicity, genus,
ends). Four different questions, and a single tree answers one.

So: **five curated facets**, with everything else computed.

| | facet | values |
|---|---|---|
| F1 | `definition.mode` | implicit · parametric · weierstrass · nodal · variational · swept · derived · discrete |
| F2 | `curvature.condition` | minimal · cmc · cmc1-bryant · k-const-negative · k-const-positive · flat · willmore · constrained-willmore · weingarten · none |
| F3 | `symmetry.periodicity_rank` | 0 · 1 · 2 · 3 |
| F4 | `embedding.quality` | embedded · immersed · self-intersecting · branched · singular |
| F5 | `tradition` | classical · gallery · sculptural · physical-model · architectural · crystallographic · physical |

**F2 needs a precedence rule**, because the values are not disjoint: every
minimal, CMC, flat and constant-*K* surface *is* a Weingarten surface. A record
stores the strongest condition, and `curvature.also_satisfies` carries the
rest, generated.

**Generated, never hand-written:** `families[]`, `curvature.also_satisfies`,
`topology.class`, and the `fabrication` block. The validator regenerates each
and compares, so a hand-edited generated field is an error. An early draft let
`families[]` be a free tag space; three records in it had already accreted tags
from no declared vocabulary.

**A field that varies across a family** carries the literal string `"varies"`
at family level and its real value on **every** specimen. The validator
enforces that, so `"varies"` can never be used to avoid stating a value. An
explicit `null` on a specimen is a *resolution* ("this member has no space
group"), not a dodge — presence is what is checked, not truthiness.

## Identity

**One record per surface, up to congruence and reparametrisation.** Alternate
constructions live in `alternate_definitions`, not in separate records:
Catalan's minimal surface *is* the Björling surface of a cycloid; the catenoid
*is* the θ = 0 associate of the helicoid; a Dupin cyclide *is* an inverted
torus.

**One record per family**, with `specimens`. Exception: a member with its own
name, its own literature and its own symmetry group is *promoted* — Schwarz P,
the gyroid and Schwarz D have three different space groups and three separate
bodies of work, so they are three records plus one family record. The same rule
promotes the unduloid and the nodoid out of the Delaunay family.

**Chirality gets the opposite default from polydb.** A chiral surface is one
record with a handedness parameter, not two enantiomorph records, because a
surface family is presented as one object with a handedness rather than as two
catalogued solids. There is one `gyroid` record.

**Suspected identities are recorded, not guessed.** `relations.same_surface_as`
carries the claim with a `confidence` and the check that would decide it — and
then someone runs the check. Fischer–Koch C(S) and C(Y) were "later recognised
as the P surface and the D surface" (Brakke); measuring χ settled it as
**refuted**, and the entries say so with the numbers. See "Settled" below.

## The row → record mapping

The workhorse, and the part that is genuinely curation rather than derivation.
The generator registries are keyed per **operator row**; identity is per
**surface**. `tools/surfdb/mapping.py` declares four dispositions — `emit`,
`merge`, `specimen`, `promote` — and `--mapping-report` prints what happened to
every row.

This is not a marginal correction. Nine records were being silently destroyed
before the collision check existed (below), and 34 rows are specimens of 8
family records rather than surfaces in their own right.

## What the validator proves

```
python tools/surfdb_validate.py            # structure, expressions, relations, index
python tools/surfdb_validate.py --slow     # + curvature measured, symmetry proved
python tools/surfdb_validate.py --coverage # the gap ledger
python tools/surfdb_validate.py --stale    # records older than their generator
```

Beyond the structural checks, it does four things a polyhedron database
cannot, because a surface's defining property is a *predicate*:

- **The curvature condition, measured — from an implicit *F* or from a
  parametric chart.** For a level set, *H* and *K* come from the derivatives of
  *F* with points Newton-projected onto the surface; for a chart r(u,v) they
  come from the fundamental forms. Both matter, because the *conditions* and
  the *implicit polynomials* are almost disjoint sets: most algebraic surfaces
  satisfy no condition at all, while every minimal, CMC and constant-curvature
  record is parametric or Weierstrass. Adding the chart path took the number of
  proved conditions from 7 to 24. The self-tests confirm Scherk's surface
  measures minimal and a 15 %-perturbed one does not, and likewise for the
  catenoid, helicoid, pseudosphere and cylinder.
- **The symmetry group, proved symbolically.** For an implicit surface the
  claimed group's generators are substituted into the polynomial and it must be
  unchanged — exact, and stronger than detecting symmetry from a point cloud.
- **Total curvature quantised, and cross-checked against the Gauss map.** For a
  complete minimal surface of finite total curvature, ∫*K* d*A* is an integer
  multiple of 4π — and it equals −4π·deg(*g*), where the degree is the winding
  number of the stored Gauss map. Where a record carries both a *g* and a
  stated total curvature they were put there for different reasons, so their
  agreement is a real cross-check rather than a restatement.

**The counts are reported honestly.** `--slow` prints how many conditions were
actually proved (7) and how many groups (17), because "0 errors" over 320
records would otherwise read as "everything checked out" when most records
carry no stored polynomial to test. That distinction belongs to the reader.

`provenance.verified` is written by the validator and must never be hand-set.

## Conventions

**Polynomials are verified against an oracle, never trusted.** A mistyped
coefficient does not error — it silently defines a *different* surface with the
wrong node count. So every closed form is checked numerically against the
shipped implementation in `math_art/` over 240 sample points, up to a single
scalar multiple, and a candidate that disagrees is **discarded in favour of a
null**. This caught two wrong equations on the first run (see below).

**Measure objects.** Every scalar is `{"exact": "...", "value": 1.234}` in a
small fixed language — integers, `+ - * / **`, parentheses, `sqrt acos asin
atan sin cos tan cosh sinh tanh exp log`, and `pi`/`phi` — extended for
surfaces with the hyperbolics and with a record's own declared parameters.
Parsing is `ast`-restricted, not `eval`: a record is data, not code.

**Scale is a closed vocabulary**, shared by `definition.scale` and
`metrics.normalization`: `published` · `unit_cell` · `unit_waist` ·
`unit_radius` · `unit_scale_parameter` · `unit_total_curvature`. Left as free
text this reintroduces exactly the ambiguity polydb solved with
`edge_length_1` / `midradius_1`. The repo's "fit a 2×2×2 box" is a **display**
transform applied downstream and is never a storage normalization.

**Derived, not transcribed.** Mathcurve, the Virtual Math Museum and Hauser's
gallery are *oracles*: cited, never republished. Equations are mathematical
facts; compilations are not.

**`null` with a stated reason beats a fabricated value.** Sarti's dodecic
carries `polynomial: null` and a `blocked_by` explaining that the pencil's
singular members are not reconstructible from the shipped machinery — strictly
more useful than either an absent record or a guessed equation.

## Building it

```sh
python tools/surfdb_build.py                  # every stage
python tools/surfdb_build.py algebraic tpms   # selected stages
python tools/surfdb_build.py --mapping-report # every row's disposition
python tools/surfdb_validate.py               # the gate
```

The strategy is **derive what the code knows, curate what the literature
knows**. The registries carry labels, families, enum keys, clip regions and
which operator reaches a surface; `tools/surfdb/curation.py` carries the
discoverer, the year, the cross-references, the symmetry and the topology.

Two things that will bite anyone extending this:

- Import the engine packages **flat** (`sys.path.insert(0, 'math_art')`). The
  `math_art` package `__init__` imports `bpy` and fails outside Blender.
- The **flat generator modules cannot be imported at all** outside Blender.
  Their enum tables are read as *source* with `ast`
  (`tools/surfdb/registry.py`), which is why the whole build runs with no
  Blender: `math_art/minsurf/` and `math_art/surfaces/` are deliberately
  bpy-free, and everything else is parsed rather than executed.

## Five corrections worth remembering

Each of these was found by a check, not by inspection, and each would have
shipped silently.

1. **Nine records were being destroyed by slug collisions.** `slugify` in
   `data/polyhedra` strips parenthesised text, and it is right to — its
   parentheticals are catalogue numbers (`Square Pyramid (J1)`). For surfaces
   the parenthetical *is* the mathematics: "Chen–Gackstatter" and
   "Chen–Gackstatter (higher genus)" are different surfaces, as are
   "Six-Ended Scherk Tower" and "… (genus 1)". Stripping them collapsed nine
   distinct records onto five slugs, and because the later row overwrote the
   earlier, **Wolf Barth's 65-nodal sextic ended up carrying a Goursat family
   member's coefficients while still looking entirely plausible.** Comparison
   operators collapse too: `(k < 0)`, `(k = 0)` and `(k > 4)` all reduce to
   `k-0` unless mapped to words.
2. **Two hand-written polynomials were wrong**, and the oracle caught both. The
   heart surface had a different circulating variant, `(2x² + y² + z² − 1)³ −
   x²z³/10 − y²z³`, rather than the Taubin form `math_art` ships; Cartan's
   umbrella had an extra `+ 3xy²` conflating it with the monkey saddle.
3. **The tetrahedral generator table was wrong.** `(y, x, −z)` is an element of
   *O*h, not *T*d — it negates *xyz*, which is exactly what fixes the
   tetrahedral orientation. Every tetrahedral record failed until the mirror
   was corrected to the plain transposition `(y, x, z)`.
4. **The icosahedral 5-fold axes are (±φ, ±1, 0), not (±1, ±φ, 0).** A rotation
   about the second is a perfectly good order-5 rotation — of a *different*
   icosahedron. Barth's sextic fails against it and passes against the first.
   Order 5 alone is not enough to make a generator correct.
5. **Symmetry claims can be right in the wrong frame.** The Goursat
   dodecahedral sextic is built from the six planes whose normals are the
   five-fold axes, which puts a C₅ axis on *z* rather than 2-fold axes on the
   coordinate axes. Same group, different orientation — so `symmetry` carries a
   `generator_set` naming which substitution table applies, and those records
   verify to ~10⁻¹⁶ on D₅d with the partiality reported rather than hidden.

## Scope

Reference-only. `data/` sits outside `math_art/`, so it is **not** packaged into
the extension zip, and the generators do not read from it. `construction` is a
cross-reference back to the code, not a dependency in either direction.

Deliberately out of scope, each for a stated reason: meshes (recipes instead);
nets and unfoldings; abstract classification objects (Enriques and K3 surfaces
are *classes* — named specimens like Kummer's quartic get records, the classes
do not); the eight Thurston geometries (visualising Nil or Sol means in-space
raytracing, not a mesh); Riemann surfaces without a named immersion in ℝ³ (the
Klein quartic is in, a generic hyperelliptic curve is not); surfaces known only
from an image; and fractal surfaces, which belong to the Fractals branch of
`research/taxonomy.md`.

## Ownership

**`data/surfaces/` belongs to the integrator**, on the same terms as `docs/`:
it funnels through shared files (`index.json`, the mapping table), it batches
naturally at integration time, and defaults change between branch and merge.
Feature agents do not write records. `tools/surfdb_validate.py` joins the
pre-merge gate, and `--stale` reports records whose generator module has a
newer commit — the direct analogue of the docs figure-age warning.

Without an owner and a gate, the coverage ledger below is an unenforced habit,
which is precisely what let `research/missing-surfaces-catalog.md` go stale.

## Coverage

`--coverage` is the gap ledger, and it is **computed**, not curated:

```
COVERAGE  363 records, 298 implemented, 65 not (82%)
```

**That figure went DOWN from 93% when the tail was added, and that is the
point.** Before it, one placeholder record (`schoen-batwing`) stood in for
roughly eighteen second-tier triply-periodic minimal surfaces, and Weber's
(p,q,r) series — about forty more — had no record at all. The surfaces were
always missing; only the records were. A ledger that under-counts what is
absent is not a ledger, and an honest 82% is worth more than a flattering 93%.

| family | total | impl. | | family | total | impl. |
|---|---|---|---|---|---|---|
| algebraic | 126 | 111 | | ruled | 12 | 12 |
| minimal-periodic | 89 | 63 | | constant-curvature | 10 | 9 |
| minimal | 46 | 46 | | cmc | 8 | 7 |
| topological | 21 | 14 | | swept | 8 | 7 |
| quadric | 13 | 3 | | misc | 7 | 7 |
| revolution | 7 | 6 | | discrete | 4 | 4 |
| physical | 4 | 4 | | spectral | 3 | 3 |
| derived | 3 | 1 | | cyclide | 2 | 1 |

**The quadrics are the honest surprise.** Ten of thirteen are *not*
implemented: only the hyperboloid of one sheet and the hyperbolic paraboloid
ship (both as *ruled* surfaces), plus the circular cylinder as the Delaunay
family's member. An "ellipsoid" exists only as a superellipsoid special case,
which is a different family. Blender's own primitives are not `math_art`
constructions and are not counted as coverage.

This replaces `research/missing-surfaces-catalog.md`, which listed 19 headline
gaps of which **13 had silently closed** — cyclides, the Hauser block, all
three record nodal surfaces, Zindler, Sievert, K = +1 revolution,
non-orientable genus-*k*, Morin, and Schwarz H/CLP (shipped as exact
Weierstrass rather than the nodal form the file said was blocked). A computed
field cannot go stale.

## Provenance

Every record states **how** its definition was obtained and **what** the
validator confirmed. Beyond that:

| | coverage |
|---|---|
| a real citation (no placeholders) | **363 / 363** |
| harvested from a module's own `References:` block | 311 |
| `discovered_by` or `named_after` | 109 |
| an external ID that resolves | 135 |
| independently cross-checked | 65 |

Citations are **harvested, not retyped.** CLAUDE.md already requires every
generator module to credit the mathematics it implements; those blocks are
kept current by whoever writes the generator, and `docs/scaffold_pages.py`
already lifts them onto the doc pages. Retyping is how a citation goes subtly
wrong and then looks authoritative, so `tools/surfdb/references.py` reads them
instead — taking the **nearest** block to each row, because `weierstrass.py`
carries fifteen of them and `zoo.py` eight, one per family section, and taking
only the header would attribute Karcher's saddle towers to the entire zoo.

Three modules had no extractable block and gained one:
`curiosity_surface_generator`, `oloid_generator` and `minsurf/orbital`. In
every case the citations already existed in the header prose — von Seggern,
Robin and Trott, Knill, Dupin, Tannery, Zoll, Schwarz, Bouguer, Neile,
Dirnböck and Stachel, Schrödinger — and were reformatted into the block form
CLAUDE.md asks for. Nothing was invented, and those modules are now compliant.

## Cross-checking

`python tools/surfdb_crosscheck.py` compares records against the local Ferréol
mirror (181 surface chapters). It is kept **separate from the build**, as
`data/polyhedra` keeps its own crosscheck stage separate, so more sources can
be cached and the verification re-run without rebuilding any geometry.

**106 agreements, 4 disagreements.** The disagreements are recorded, not
reconciled — all four are discovery years where Ferréol and the record's cited
source differ (the catenoid: 1740 against Euler's 1744; Cayley's cubic 1850
against 1869; Gabriel's horn 1641 against 1643; Barth's sextic 1994 against
1996). `data/polyhedra` records two disagreements against McCooey rather than
forcing them to agree; this is the same standard.

Silence is never confirmation. A page that does not discuss curvature produces
no entry rather than an agreement, and a single passing mention of another
surface's property is not a contradiction — Ferréol's helicoid chapter says
"developable" once, about the *developable helicoid*, a different surface.

The cross-check also caught two wrong IDs of this database's own:
`schwarz-lantern` pointed at `ch1340_sinus_2`, which is Ferréol's **Sine
surface** (the mirror has no lantern chapter at all), and `genus-g-surface`
pointed at a chapter *defining* genus rather than describing an object. Both
were removed — a wrong cross-reference is worse than an absent one, because it
sends a reader to the wrong page and looks authoritative doing it.

## Driving the generators

`blender --background --python tools/surfdb_drive.py -- --write` runs each
implemented record's operator and compares the mesh with the record. Before
it, `implemented: true` meant "the module file exists on disk" — not a check
at all.

**328 invocations across 298 records: 327 agreed.** This is the analogue of
polydb's crosscheck, the stage that found twelve wrong Johnson solids, and it
earned its place immediately by finding four things:

1. **It caught a flaw in itself first.** The first version never passed the
   record's enum key, so every record on a multi-surface operator drove that
   operator's *default*: Boy's surface, the Klein bottle and the genus-g
   surface were all silently measured as the same shape, and the resulting
   "agreement" meant nothing. Selecting the right surface needs more than
   introspection — several of these enums are built by a callback and report
   no items until a context supplies them, and the selector is filtered by
   another enum whose vocabulary differs from the registry's (`periodicity` is
   `TRIPLY` where the registry family says `TPMS`).
2. **A fabricated operator ID, in this database.** `pearce-saddle-surface`
   named `mesh.pearce_surface_add`, which does not exist:
   `math_art/pearce_surface.py` is a library used by the saddle-polyhedron
   generator, not a generator itself.
3. **The Klein bottle mesh is not closed.** It has the right Euler
   characteristic — χ = 0, as the module header claims — but 96 edges bound
   only one face, forming two open loops: the combinatorial identification
   closes one direction of the parameter grid and seams the other. χ being
   right anyway is a coincidence, since an open cylinder also has χ = 0, which
   is precisely why an Euler check alone would not have caught it.
4. **`SCHERKT` is unreachable from the interface.** It is a row in the TPMS
   registry and builds fine through `minsurf.build_tpms`, but
   `mesh.periodic_minimal_add` omits it from the surface list under all five
   of its periodicity settings.

Both findings about the code are recorded as `notes.known_issue` on the
records concerned. Boundary components are compared only for **compact**
surfaces: a complete one is necessarily rendered as a truncated patch, so its
mesh has rims the mathematical surface does not, and comparing those failed
correct records until it was fixed.

## Settled: Fischer–Koch C(S) and C(Y)

Brakke lists Fischer–Koch C(S) and C(Y) as "later recognised as the P surface
/ the D surface", while this repo ships them as separate rows. The records
carried `same_surface_as` with `confidence: "suspected"` and the check that
would decide it. That check has now been run.

Meshing each nodal level set over a common one-cell block at resolution 64 and
counting V − E + F gives **χ = −128 for C(S) against −4 for Schwarz P**, and
**−15 for C(Y) against −10 for D**. χ scales roughly with the number of cells
in the block, so even allowing for C(S)'s nodal function having half the period
of P's — its leading terms are cos 2x, cos 2y, cos 2z — an eight-fold cell
count would predict χ near −32, not −128.

**Refuted**, and recorded as such. The likely explanation is that Brakke
indexes the Fischer–Koch family differently from the sources the shipped rows
follow (Koch & Fischer 1988 for C(S), von Schnering & Nesper 1991 for C(Y)), so
the remark may well be true of surfaces those names denote elsewhere. That
caveat is recorded rather than resolved, because resolving it would mean
transcribing Brakke's own definitions.

## Local sources

Beyond the shipped registries, the metadata design draws on this repo's
(gitignored) `research/` corpus and the mirrors under
`S:/data/math_art/references/websites/`:

- **Ferréol, *Encyclopédie des formes mathématiques remarquables*** — 181
  surface chapters, mirrored locally, so `ids.mathcurve` resolves offline.
- **The 3DXM Virtual Math Museum** — organised by *construction method*, which
  is where `definition.mode` comes from.
- **Hauser's *Bildergalerie algebraischer Flächen*** — the 63 named gallery
  surfaces, whose equations `math_art/surfaces/algebraic.py` already carries.
- **Weber's minimal-surface archive** (dead; recovered via the Wayback CDX
  index, 280 folders) and `research/minimal_surfaces_status.md` — the
  per-surface invariants for the Weierstrass zoo.
- **Schoen, NASA TN D-5541 (1970)** and Fischer–Koch for the TPMS space groups.
- **MathWorld's `topics/Surfaces.html`** — the eleven-topic tree whose overlaps
  are the evidence for a faceted scheme.
