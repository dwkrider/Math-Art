# Backlog — incomplete & resumable work

Work that was **designed and/or partly executed but not finished** — including
plans stopped because they turned out infeasible or blocked. This is distinct
from [TODO.md](TODO.md), which is a broad idea list; entries here are things
already scoped enough to resume, each with a pointer back into the detailed
plan in `research/`.

Status tags: **BLOCKED** (stopped for a stated reason — needs a decision or
missing input) · **PARTIAL** (started, unfinished) · **PLANNED** (designed,
not started).

Maintenance: update this file whenever a requested plan is left incomplete or
stopped, and prune/mark items as they get finished. (See CLAUDE.md.)

---

## ⛔ Blocked — needs your decision or missing input

These are the "figure it out later" items. Each is stalled on something
external, not on effort.

- **Hauer Design 1–5 faithful facsimiles** — needs source material: the
  *Continua* book plates / good photographs. Current Modular Screen presets
  are only approximations "after Hauer." → `research/hauer-carlberg-screens-research.md`, "Deferred (and why)".
- **Scherk-graph screen module** — *dropped as infeasible as approached*: the
  surface "comes out disconnected when clipped to a slab (the saddle bridges
  only meet at the divergent necks that clipping removes)." Needs a different
  construction if revived. → same doc, "Deferred (and why)".
- **Biscribed-solids solver** — fully designed but unbuilt, and the blocker is
  conceptual, not code: **a biscribed form does not exist for every solid**
  (naive alternating projection diverges on e.g. the truncated dodecahedron).
  First requirement is an *existence test* that reports "no biscribed form"
  gracefully (rectified solids, truncated tetra/cube/dodeca,
  rhombicuboctahedron) rather than looping forever. → `research/biscribed-solver-research.md`.
- **dMcCooey polyhedra program — licensing gate** — the plan mandates **do
  NOT bulk-copy McCooey's ~680 coordinate tables**; everything must be
  procedurally derived, or an email-for-permission hybrid for the hardest
  cases (Johnson J84–J92, higher-genus maps). This constrains how much of the
  polyhedra backlog below can be done. → `research/dmccooey-polyhedra-plan.md`.
- **From-scratch "no public code" items** — must be reimplemented from papers,
  no reference repo: Séquin Sculpture Generator 1/SLIDE (from SIGGRAPH '97),
  Rinus Roelofs weaving (from Bridges papers), **Norman Carlberg modules**
  ("original implementation territory"), and a parametric **Star Ball**
  generator ("would be genuinely novel"). → `research/github-math-art-repos.md`, §6 "Known gaps".

## 🔧 In flight (this session)

- **Over-Under Screen — bridge twist** — currently at v1.12.122 (Hauer-style
  relief; monotone 90° quarter-twist necks, CW-from-one-hub/CCW-from-other).
  Awaiting a live verdict on whether the twist sense/amount reads right; a
  mirror is a single-sign flip. Generator: `math_art/over_under_screen_generator.py`.
- **Remove `if __name__ == "__main__":` self-test blocks** — a Blender
  extension code-review request (94 of 97 modules). Removal has **zero runtime
  effect** (the blocks never run when imported by Blender). Decision pending:
  *migrate* the pure-numpy invariant checks into `tests/` first, or *delete*
  the blocks + their `_run_selftest` bodies outright. → review link in TODO.md
  ("Remove `if __name__ …`").

## Pattern Engine (unifying `math_art/patterns/` package)

A 6-phase roadmap; individual generators exist (Orbifold Sphere, Hyperbolic
Tiling, Celtic, Islamic, Wallpaper) but the unifying signature-driven engine
is not built. Reconcile against the live menu before resuming — Phases 0–1 may
be partly overtaken. → `research/pattern-engine-plan.md`.

- **PLANNED** — Orbifold-signature engine (Phase 0): `orbifold.py` parser +
  `tiling_wrapper.py` common placed-tiles format; refactor existing generators
  onto it.
- **PLANNED** — Substitution/aperiodic backend (Phase 2): Penrose P3 →
  hat/spectre with Tile(a,b) slider; optional cut-and-project tie to
  `polytope4d_generator.py`.
- **PLANNED** — Interlace backend (Phase 3): mirror-curve tracer → medial-graph
  "interlace any tiling." Open question: generator vs. post-process modifier.
- **PLANNED** — Mode B continuous Hauer/Carlberg modules (Phase 4). Open risk:
  **adjacency fidelity** — how much edge/corner metadata the common format must
  carry for robust C¹ boundary matching on hyperbolic/aperiodic (not just
  wallpaper) cases.
- **PLANNED** — Wrap-onto-surface + SVG/net export (Phase 5); shares the SVG
  writer with stacked-slices.

## Fathauer additions (R1–R7)

All open recommendations (none built yet). Top two by payoff are R1 and R2.
→ `research/fathauer-review.md`, §3.

- **PLANNED** — R1 **Phyllotaxis / Fibonacci spirals** (golden-angle placement
  on disk/dome/sphere). The one clear missing headline; a genuine coverage gap.
- **PLANNED** — R2 Crochet branching lobes + graded/crest growth (enhance the
  current single-ratio crochet).
- **PLANNED** — R3 Fractal knots: substitution method + decorated-tiling links
  (the decorated-tiling sub-part is a near-free win).
- **PLANNED** — R4 Fractal-tiling dissection families (verify `fractal_reptile`
  coverage first), R5 Haüy fractal-polyhedron variants, R6 Goldberg GP(1,2)
  "spiked" ceramic preset, R7 n-fold monkey saddle `z=ρⁿcos(nφ)` (trivial add).

## Hauer / Carlberg screens

Modular Screen shipped. Remaining: → `research/hauer-carlberg-screens-research.md`.

- **BLOCKED** — Design 1–5 facsimiles (source material) — see top section.
- **BLOCKED/dropped** — Scherk-graph screen — see top section.
- **PLANNED** — Custom module from a mesh/`.obj` file: bundle cells in
  `math_art/modules/`, array on grid, optional weld/solidify.

## Knots & KnotPlot

Knot Carpets and Celtic-2D are built; the rest of the Tier catalog is open.
→ `research/knotplot_research.md`, §5.

- **PLANNED (Tier 1)** — Harmonic/Lissajous/Chebyshev/billiard knots; Petal
  knots; Turk's-head/decorative rope knots; Rational/2-bridge/pretzel/twist
  from Conway notation.
- **PLANNED (Tier 2)** — real knot-energy/ropelength relaxation (current relaxer
  is "smoothing+repulsion, not a knot energy"); DT/Gauss-code input + 11+
  crossing catalog; custom braid word / Brunnian links; satellite/cable knots;
  Seifert surface from a scene curve (see Seifert section); Antoine's necklace
  enhancements.
- **PARTIAL (Tier 3 infra)** — consolidate the **four near-duplicate curve→tube
  helpers** into one shared RMF-tube module, and add a shared knot-energy
  module ("the two biggest consolidation opportunities"); invariant-readout
  panel.
- **PLANNED (Tier 4)** — 4D spun/twist-spun knotted spheres; Lorenz knots;
  hyperbolic census presentation.

## Polyhedra — dMcCooey completeness program

Large planned program (gated by the licensing constraint above).
→ `research/dmccooey-polyhedra-plan.md`, §5.

- **PLANNED** — Uniform **star** polyhedra + duals (~51 + ~47), "biggest single
  gap": `uniform_star_generator.py` via Wythoff (incl. snubs + Great
  Dirhombicosidodecahedron), duals by polar reciprocation + Dorman Luke.
- **PARTIAL** — Johnson solids: J1–J48 shipped; **J49–J83** (augment/diminish/
  gyrate recipe table) and **J84–J92** (per-solid numeric solver) remain.
- **MISSING** — Star prisms/antiprisms & star dipyramids/trapezohedra ({p/q}
  step parameter, none exists yet).
- **PARTIAL** — Geodesic Class III + geodesic cubes/RTs (only Class I/II,
  sphere-projected, exist).
- **MISSING** — Toroids (Szilassi/Császár, regular {6,3}/{3,6}/{4,4},
  non-regular, iris) — high showpiece value.
- **MISSING/PARTIAL** — classical compounds beyond the 3; "Other" gallery
  (Dürer, Schönhardt, Jessen, Bilinski, Escher's Solid, Associahedron).
- **PLANNED, low-effort** — expose reachable-but-hidden presets (Dipyramids/
  Trapezohedra `dPn/dAn`, Archimedean–Catalan hulls `j`, Propellor, Chamfered,
  Truncated/Rectified Archimedean) — op strings + canonicalize, no new math.
- **BUILT (done)** — icosahedron stellation engine (`stellation_engine.py`,
  59 Crennell presets). Residual: the superseded 2-D DCEL exact-build path was
  abandoned (not fixed), and a **general** stellation engine (RT stellations,
  full diagram enumeration beyond the icosahedron) is still unbuilt.
  → `research/icosahedron-stellations.md`.

## Stacked slices & fabrication export

Implementation plan, not yet built. → `research/stacked-slices-plan.md`.

- **PLANNED** — Flavor A: Geometry-Nodes solid-stack asset (Repeat Zone +
  boolean-intersect slabs). Can't export SVG / laser-accurate prisms.
- **PLANNED** — Flavor B (the fabrication-useful half): Python/bmesh
  `bisect_plane` → island detection → per-region SVG (`fill-rule:evenodd`, mm
  units) with a hand-rolled ~30–40-line SVG writer. First prototype target: the
  **woven polyhedron** output (exercises multi-loop/hole slices). Shares the
  SVG writer with Pattern Engine Phase 5.

## Curvilinear interlace

Designed this session; not built. → `research/curvilinear-interlace.md`.

- **PLANNED** — Stage-1 curve-family front-end (stages 2–4 reuse the Knot
  Carpet crossing/alternation + `pattern_common` strapwork). Roadmap: Wavy
  Plaid MVP → boundary frame → Warped Grid + Polar → Guilloché → graduate to
  `curvilinear_interlace_generator.py`. Recommended: prototype as a new
  `source` mode on the Knot Carpet (~80% reuse). Open risks: 2-colorability
  seams in dense guilloché, O(n²) crossing perf, grazing crossings, watertight
  caps for open-strand relief.

## Seifert surfaces

- **PARTIAL** — free-boundary "membrane" smoother (`dynamic_relax`, the
  `MEMBRANE` method) is implemented and numerically stable but "does not yet
  reach SeifertView's finish"; the pinned-boundary `AREA` method is the shipped
  default and works. Also open: **Seifert surface from an arbitrary scene
  curve** (currently braid-input only). → `research/knotplot_research.md` and
  `math_art/seifert_surface_generator.py` (§"Dynamic relaxation" ~line 262,
  `MEMBRANE` enum ~line 905). Not a "kill or fix" break — a resumable finish.

## Minimal surfaces (Weierstrass zoo)

Catalog built from the minimalsurfaces.blog harvest (`research/msblog_harvest/`,
160 surfaces indexed, 114 with clean g/dh). M1/M2 and a chunk of M3 shipped;
these are the flagged residuals and the next tiers.

- **PARTIAL** — **Tilted Scherk (doubly periodic) tiled appearance not exactly
  right.** Shipped (trim-and-snap truncation → open wall rims, no blocky clamp
  plateaus), built as a horizontal reparametrization of the Scherk graph
  (`_tilt_scherk_doubly`, `Rho=1` == `SCHERK1`, single connected component). But
  at Cells 3×3 / radius 1.2 it reads as a **boxy lattice of open cells with
  near-flat vertical walls**, not a smooth leaning/organic tilted corrugation.
  Note the surface is **non-embedded by design** — the López–Ros tilt makes the
  half-planes self-intersect (Weber's page), so it can't be a clean embedded
  surface. Follow-ups: make the walls read as the true leaning tilted sheets
  (currently flat/rectangular); and/or present only the single fundamental cell
  to sidestep the boxy tiling; residual fine ribbing at strong tilt (radius
  ~1.5). → https://minimalsurfaces.blog/home/repository/doubly-periodic/tilted-scherk/
- **PLANNED** — **Higher-genus Chen–Gackstatter (g2/4/5).** Harvested branch
  constants are captured and correct, but the engine's only hyperelliptic
  assembler is hard-specialized to the Costa–Hoffman–Meeks cut structure (gives
  wrong-χ non-manifold meshes on CG data). Needs a **dedicated cyclic-cover
  assembler** (integrate one angular wedge with an Enneper-aware trim, then
  rotate/reflect/weld). Also blocks `symm-chen-gackstatter-gn`. →
  `research/msblog_harvest/higher_genus.json`.
- **PLANNED** — **Doubly-periodic block (~20): Wei g2/g3, Karcher–Scherk
  g2/g3/g4, KMR-2/3, RTW.** All share one reciprocal-paired hyperelliptic form
  (`dh=1/z`) with harvested solved constants — a single abstraction + a
  doubly-periodic tiler (mind the connectivity lessons from Scherk) unlocks the
  set. → `research/msblog_harvest/doubly_periodic.json`.
- **PLANNED (Tier 3 exotics)** — **genus-1 helicoid**, **Callahan–Hoffman–Meeks**,
  Costa–Wohlgemuth, Catenoid–Enneper g2/3/4 — solved constants/seeds harvested;
  each needs a period solve or the cyclic-cover assembler. →
  `research/msblog_harvest/{singly,higher_genus}_periodic.json`.
- **PLANNED (Tier 4)** — Weber–Wolf g3/g4, Kapouleas, Connor, higher Wei/RTW —
  research-grade (orthodisk / flat-structure / multi-dim Newton); attempt-all
  per the plan, gate on period residual.
- **BLOCKED (small)** — **Björling clothoid** (needs a complex-Fresnel
  evaluator); **plane-with-catenoids** (branched `g=Rho/√z`; needs an
  upper-half-plane + Schwarz-reflection integrator, not radial disk rays);
  **antiprismatic k-noid general-n** (harvested numeric (a,ρ) don't close the
  period beyond nn=5 — only nn=5 shipped).
- **PARTIAL** — **Exact-WE TPMS Bonnet family (`PGD`).** Presets show the clean
  nodal P/Gyroid/D; Custom sweeps the exact Bonnet morph. Exact **tiled** cells:
  Schwarz P (mirror-plane) is clean; Schwarz D is watertight-ish (~6%
  non-manifold, rounder than nodal); the **chiral Gyroid could not be tiled
  watertight** (kept as the fundamental piece). Follow-up: a θ-tracking /
  chiral-aware space-group assembler for the Gyroid cell.

## Other from-scratch / infrastructure gaps (from the repo survey)

Flagged in `research/github-math-art-repos.md` as not-yet-in-repo (reimplement
from papers; no-license repos are reference-only):

- **PLANNED** — geometry-nodes emission mode for generators; perforated-shell /
  congruent-shell splitting modifier (voronoizer + manifold3d + libigl offset);
  TPMS generator; Hopf fibration; 4D polytope slicing/projection; Wythoff
  across all three curvatures; escape-time fractal mesher; quasicrystal suite;
  isohedral Escher-tile + Farris wallpaper displacement; differential-growth /
  reaction-diffusion on mesh.
