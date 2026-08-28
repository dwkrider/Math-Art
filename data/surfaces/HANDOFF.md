# Surface database — handoff

Single entry point for anyone (human or agent) picking this up. Read this, then
`README.md` for the format spec. Everything below is re-runnable.

## State

**320 records, 298 implemented, 0 errors, 8 warnings.**

```sh
python tools/surfdb_build.py       # rebuild everything, ~30 s, no Blender
python tools/surfdb_validate.py    # the gate, ~3 s
python tools/surfdb_validate.py --slow   # + curvature and symmetry, ~4 min
```

| Stage | N | What |
|---|---|---|
| `algebraic` | 112 | 10 classical, 63 Hauser gallery, 22 Goursat, 7 Mathcurve, 7 VMM, 3 record-nodal |
| `minimal` | 46 | the rank-0 Weierstrass/Björling zoo |
| `minimal-periodic` | 64 | 25 singly, 9 doubly, 23 nodal TPMS, 6 exact |
| `quadric` | 13 | the degree-2 classics; **only 3 implemented** |
| `topological` | 17 | Klein ×2, Boy, Roman, cross-cap, Morin, genus-g, non-orientable genus-k |
| `constant-curvature` | 10 | pseudosphere, Dini, Kuen, Minding, breather, Amsler, Sievert, K = +1 |
| `cmc` | 8 | Delaunay family + promoted unduloid/nodoid, Wente, bubbleton, Bryant |
| `ruled` / `swept` / `revolution` | 27 | conoids, Whitney, Gaudí, Darboux motions, Tannery, Zoll |
| `misc` / `discrete` / `physical` / `spectral` / `derived` / `cyclide` | 23 | |
| `missing` | 10 | not implemented, each with `blocked_by` + `resume` |

The 8 warnings are all the same informative kind: the Goursat dodecahedral
records verify their symmetry only on D₅d, a proper subgroup of *I*h, and say
so rather than overclaiming.

## Decisions already made — do not re-litigate

- **Facets, not a tree.** Five curated facets; `families[]`,
  `curvature.also_satisfies`, `topology.class` and `fabrication` are
  *generated* and the validator regenerates and compares. Do not hand-edit them.
- **Meshes are not stored.** A recipe plus measured invariants. Storing
  vertices would be storing one rendering.
- **`fidelity` is separate from `exactness`.** The nodal TPMS are
  `approximation`s pointing at an exact definition by index, and are gated at
  their own residual. Do not "fix" them by gating everything at one tolerance —
  that was tried in design and it either fails all 23 or blinds the exact rows.
- **One record per surface, one per family.** Alternate constructions go in
  `alternate_definitions`; regimes go in `specimens`; promotion needs its own
  name, literature *and* symmetry group.
- **Polynomials are verified against the shipped implementation** before they
  land, up to a scalar multiple, over 240 points. A candidate that disagrees
  becomes a `null`, not a record.
- **The database is reference-only**: outside `math_art/`, not packaged into
  the extension zip, and no generator reads from it.

## Traps that have already cost time

- **`slugify` must KEEP parentheses.** Copying polydb's version (which strips
  them, correctly, for `(J1)`) collapsed nine records onto five slugs and let a
  Goursat row silently overwrite Wolf Barth's sextic. Comparison operators need
  words too: `(k < 0)` and `(k = 0)` both reduce to `k-0` otherwise.
- **The flat generator modules cannot be imported** — they `import bpy`. Read
  their enum tables as source with `tools/surfdb/registry.py`. Only
  `math_art/minsurf/` and `math_art/surfaces/` import flat, and even those need
  `sys.path.insert(0, 'math_art')` because the package `__init__` imports `bpy`.
- **`(y, x, -z)` is not a Td element.** It is in *O*h; it negates *xyz*, which
  is the very thing that fixes the tetrahedral orientation. Td's mirror is the
  plain transposition.
- **Icosahedral 5-fold axes are (±φ, ±1, 0)**, not (±1, ±φ, 0). A rotation
  about the latter is order 5 but belongs to a different icosahedron.
- **A symmetry claim can be right in the wrong frame.** The Goursat
  dodecahedral sextic has its C₅ axis on *z*; the standard-frame generators
  fail against it. Use `symmetry.generator_set` to name the substitution table.
- **The Clebsch cubic is not Td.** Its celebrated S₅ symmetry is *projective*.
  In the affine embedding the point group is only the coordinate permutations —
  C₃v about the body diagonal.
- **Curvature at a singular point is undefined**, and `sample_curvature`
  returns `None` when too few points project onto the surface. That is "not
  measurable here", not a pass; treat it as such.
- **`ast` node whitelisting alone does not secure the expression language.**
  `__import__('os')` is structurally a `Call` on a `Name` with a `Constant`
  argument — every node whitelisted. Function names and literal types must be
  checked at parse time too.

## Deliberately absent — do not "fix"

- **Meshes, nets, unfoldings.**
- **Enriques and K3 as classes.** Named specimens only (Kummer's quartic is a
  K3 and has a record; the class does not).
- **The eight Thurston geometries.** Wrong output kind — raytracing, not a mesh.
- **Blender primitives counted as coverage.** A sphere added by
  `mesh.primitive_uv_sphere_add` is not a `math_art` construction, which is why
  ten of thirteen quadrics read `implemented: false`.
- **Fischer–Koch C(S)/C(Y) merged into P/D.** Brakke says they were "later
  recognised as" those surfaces; that is recorded as
  `same_surface_as` with `confidence: "suspected"` and the check that would
  decide it. Do not merge on the strength of the remark alone.

## Open work

**Polynomial coverage.** 100 of 320 records carry a stored closed form. The
rest are `weierstrass`, `parametric` or `nodal` modes whose defining data is
not yet transcribed into the record — the machinery is there
(`definition.gauss_map`, `height_differential`, `period_conditions`), the data
is not. Highest value: the 46 `minimal` records, where filling `g` and d`h`
would let §7.5's total-curvature check run across the whole zoo.

**Curvature checks reach 7 records** and **symmetry 17**, because those are the
records with both a polynomial and a testable claim. Both counts rise directly
with polynomial coverage. The validator prints them so the number is never
mistaken for "everything was proved".

**Cross-checks.** `provenance.cross_checked` is empty everywhere. The Mathcurve
mirror is local (`S:/data/math_art/references/websites/mathcurve/`, 181 surface
chapters) and `ids.mathcurve` already resolves into it, so a `crosscheck` stage
comparing stored invariants against those pages is the obvious next stage —
this is what found 12 wrong Johnson solids on the polyhedron side.

**Driving the generators.** `construction[].implemented` is set from module
existence, not from running the operator. The design calls for invoking each
one headlessly and comparing measured invariants (the polydb `crosscheck` that
found real bugs). That needs Blender and is not built.

**`polyhedral_analogue` is unpopulated.** The joins are real and interesting —
the Petrie–Coxeter apeirohedra to the TPMS, saddle polyhedra to the minimal
surfaces they span, Schwarz's lantern to the cylinder. The field exists and the
validator resolves it against `data/polyhedra/index.json`.

**Weierstrass records carry no `mesh_recipe`.** Nothing measures V/E/F, χ,
components or one-sidedness yet, though `math_art`'s own self-tests already
compute all of them — consuming those rather than duplicating them is the
intended route.

## What this found in the shipped code

Nothing wrong, which is worth stating: every one of the 22 Goursat polynomials
and all 63 Hauser equations reproduced the shipped implementations exactly, and
the Goursat symmetry claims verified to ~10⁻¹⁶. The five corrections in the
README were all errors in *this* database's own tables and transcriptions,
caught before they shipped.
