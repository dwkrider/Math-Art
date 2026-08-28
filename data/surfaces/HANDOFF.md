# Surface database — handoff

Single entry point for anyone (human or agent) picking this up. Read this, then
`README.md` for the format spec. Everything below is re-runnable.

## State

**363 records, 298 implemented, 0 validator errors, 8 warnings.**

```sh
python tools/surfdb_build.py            # rebuild everything, ~40 s, no Blender
python tools/surfdb_validate.py         # the gate, ~4 s
python tools/surfdb_validate.py --slow  # + curvature and symmetry, ~5 min
python tools/surfdb_crosscheck.py       # against the local Ferreol mirror
python tests/test_surfdb.py             # 11 tooling self-tests

blender --background --python tools/surfdb_drive.py -- --write   # the only
                                        # step that needs Blender
```

**Never pass `--factory-startup` to the drive step.** It disables extensions,
so every Math Art operator appears unregistered and you will conclude the
add-on is broken when it is merely switched off.

| Stage | N | What |
|---|---|---|
| `algebraic` | 126 | 10 classical, 63 Hauser, 22 Goursat, 7 Mathcurve, 7 VMM, 3 record-nodal, 14 absent |
| `minimal-periodic` | 89 | 25 singly, 9 doubly, 23 nodal TPMS, 6 exact, 25 absent |
| `minimal` | 46 | the rank-0 Weierstrass/Bjorling zoo |
| `topological` | 21 | Klein x2, Boy, Roman, cross-cap, Morin, genus-g, non-orientable genus-k |
| `quadric` | 13 | the degree-2 classics; **only 3 implemented** |
| `constant-curvature` | 10 | pseudosphere, Dini, Kuen, Minding, breather, Amsler, Sievert, K=+1 |
| `cmc` | 8 | Delaunay family + promoted unduloid/nodoid, Wente, bubbleton, Bryant |
| others | 50 | ruled, swept, revolution, misc, discrete, physical, spectral, derived, cyclide |

The 8 warnings are all one informative kind: the Goursat dodecahedral records
verify their symmetry only on D5d, a proper subgroup of *I*h, and say so rather
than overclaiming.

## Completion state of Part 1

| criterion | status |
|---|---|
| `--slow` validation, 0 errors | **yes** |
| every record has a real citation | **363 / 363** |
| definitions stated (datum or explicit null-with-reason) | **363 / 363** |
| `cross_checked` where a source page exists | **65 / 65 resolvable** |
| `mesh_recipe.measured` for implemented records | **296 / 298** |
| `polyhedral_analogue` where a join exists | **26** |
| tail records added | **43** |
| Fischer-Koch C(S)/C(Y) settled | **refuted, with numbers** |

The two implemented records without a measurement are the two the drive stage
found problems with (see below); `plateau-span` is skipped because
`object.minimal_span` transforms a *selection* and cannot be driven from an
empty scene.

## Decisions already made — do not re-litigate

- **Facets, not a tree.** Five curated facets; `families[]`,
  `curvature.also_satisfies`, `topology.class` and `fabrication` are
  *generated* and the validator regenerates and compares. Do not hand-edit them.
- **Meshes are not stored.** A recipe plus measured invariants.
- **`fidelity` is separate from `exactness`.** The nodal TPMS are
  `approximation`s gated at their own residual. Do not "fix" this by gating
  everything at one tolerance — that either fails all 23 or blinds the exact rows.
- **One record per surface, one per family.** Alternate constructions go in
  `alternate_definitions`; regimes go in `specimens`; promotion needs its own
  name, literature *and* symmetry group.
- **Polynomials are verified against the shipped implementation** before they
  land, up to a scalar multiple, over 240 points. A candidate that disagrees
  becomes a `null`.
- **Citations are harvested from module `References:` blocks, not retyped**
  (`tools/surfdb/references.py`), taking the *nearest* block to each row.
- **Coverage is allowed to look worse.** Adding the tail dropped it from 93% to
  82%; the surfaces were always missing, only the records were.
- **The database is reference-only**: outside `math_art/`, not packaged into
  the extension zip, and no generator reads from it.

## Traps that have already cost time

- **`slugify` must KEEP parentheses.** Copying polydb's version (which strips
  them, correctly, for `(J1)`) collapsed nine records onto five slugs and let a
  Goursat row silently overwrite Wolf Barth's sextic. Comparison operators need
  words too: `(k < 0)` and `(k = 0)` both reduce to `k-0` otherwise.
- **The flat generator modules cannot be imported** — they `import bpy`. Read
  their tables as source (`tools/surfdb/registry.py`). Only `math_art/minsurf/`
  and `math_art/surfaces/` import flat, and even those need
  `sys.path.insert(0, 'math_art')`.
- **`(y, x, -z)` is not a Td element.** It is in *O*h; it negates *xyz*.
- **Icosahedral 5-fold axes are (±φ, ±1, 0)**, not (±1, ±φ, 0). A rotation
  about the latter is order 5 but belongs to a different icosahedron.
- **A symmetry claim can be right in the wrong frame.** Use
  `symmetry.generator_set` to name the substitution table.
- **The Clebsch cubic is not Td.** Its S5 symmetry is *projective*; the point
  group is C3v about the body diagonal.
- **`ast` node whitelisting alone does not secure the expression language.**
  Function names and literal types must be checked at parse time too.
- **Dynamic enums report no items.** Several operators build their surface list
  with an `items=` callback, so introspection sees nothing until a context
  supplies it — and the selector is filtered by another enum whose vocabulary
  differs from the registry's (`periodicity` is `TRIPLY` where the registry
  family says `TPMS`). The drive stage tries filter/target combinations.
- **Boundary components are comparable only for COMPACT surfaces.** A complete
  one is rendered as a truncated patch, so its mesh has rims the surface does
  not; comparing them failed correct records.

## What this database found in math_art

Recorded as `notes.known_issue` on the records concerned.

1. **The Klein bottle mesh is not closed.** χ = 0 as the module header claims,
   but 96 edges bound one face, in two open loops: the combinatorial
   identification closes one direction of the parameter grid and seams the
   other. χ being right is a coincidence — an open cylinder also has χ = 0 —
   which is why an Euler check alone would not catch it.
2. **`SCHERKT` is unreachable from the interface.** It is a TPMS registry row
   that builds fine through `minsurf.build_tpms`, but
   `mesh.periodic_minimal_add` omits it under all five periodicity settings.
3. **Three modules lacked the `References:` block CLAUDE.md requires**
   (`curiosity_surface_generator`, `oloid_generator`, `minsurf/orbital`). Their
   citations existed in header prose and have been reformatted into the
   extractable block; nothing was invented.

## Deliberately absent — do not "fix"

- Meshes, nets, unfoldings.
- Enriques and K3 as classes; named specimens only.
- The eight Thurston geometries — wrong output kind.
- Blender primitives counted as coverage, which is why ten of thirteen quadrics
  read `implemented: false`.
- Fischer-Koch C(S)/C(Y) merged into P/D — **refuted by measurement**, see the
  README.

## Open work — Part 2 and beyond

Part 1 is complete. `research/plans/surface-gaps-implementation-plan.md` Part 2
covers the generators, in order: quadrics (11-12 records, cheapest, verification
already written) → the nearly-free named surfaces (Bianchi-Pinkall, spherical
helicoid, Darboux cyclide) → transcription (Sarti dodecic, multi-solitons) →
the TPMS tail → derived transforms → Klein quartic.

**Rename the branch at that boundary**, not before: Part 1 is squarely database
work, so `surface-database` stays correct, and Part 1 produced no builds to
strand.

Database work that remains, none of it blocking:

- **Defining data is stored for 105 of 363 records.** The rest are
  `weierstrass` or `parametric` rows built by dedicated functions with no
  extractable closed form; each says so explicitly. Curating them by hand — and
  verifying each against the shipped builder, as the implicit block already is
  — would lift the curvature check beyond its current 7 records and the
  symmetry check beyond 17. Both counts are *reported* by `--slow` so the
  number is never mistaken for "everything was proved".
- **`discovered_by`/`named_after` is on 109 records and external IDs on 135.**
  Both are "where known" rather than gaps, but the Weierstrass zoo would repay
  a pass: its row labels name Karcher, Wohlgemuth, Wei, Callahan-Hoffman-Meeks
  and others whose individual papers are not yet cited per row.
- **Only Ferreol is cross-checked.** The VMM and minimalsurfaces mirrors sit
  beside it under `S:/data/math_art/references/websites/` and would extend
  `tools/surfdb/crosscheck.py` with no new machinery.
- **`mesh_recipe.measured` records no orientability or one-sidedness** — the
  drive stage leaves both `null`, because a Blender mesh carries no global
  sign. `math_art`'s own self-tests measure one-sidedness for the
  non-orientable rows and consuming that is the intended route.
