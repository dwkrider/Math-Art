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

*   **Hauer Design 1–5 faithful facsimiles** — needs source material: the  
    _Continua_ book plates / good photographs. Current Modular Screen presets  
    are only approximations "after Hauer." → `research/hauer-carlberg-screens-research.md`, "Deferred (and why)".
*   **Scherk-graph screen module** — _dropped as infeasible as approached_: the  
    surface "comes out disconnected when clipped to a slab (the saddle bridges  
    only meet at the divergent necks that clipping removes)." Needs a different  
    construction if revived. → same doc, "Deferred (and why)".
*   **Biscribed-solids solver** — **BUILT** (`math_art/biscribed_solids_generator.py`):  
    exact symmetry-reduced solvers with an existence test (reports "no biscribed  
    form" for the rectified/truncated cases that have none), the non-chiral set,  
    and 27/31 chiral. **Residual blocker:** 4 chiral solids (#20/#24, #27/#31)  
    resist the solver — conceptual, not code; see the Polyhedra section below  
    for the attempts and the reciprocal/homotopy direction.  
    → `research/biscribed-solver-research.md`.
*   **dMcCooey polyhedra program — licensing gate** — the plan mandates **do**  
    **NOT bulk-copy McCooey's ~680 coordinate tables**; everything must be  
    procedurally derived, or an email-for-permission hybrid for the hardest  
    cases (Johnson J84–J92, higher-genus maps). This constrains how much of the  
    polyhedra backlog below can be done. → `research/dmccooey-polyhedra-plan.md`.
*   **From-scratch "no public code" items** — must be reimplemented from papers,  
    no reference repo: Séquin Sculpture Generator 1/SLIDE (from SIGGRAPH '97),  
    Rinus Roelofs weaving (from Bridges papers), **Norman Carlberg modules**  
    ("original implementation territory"), and a parametric **Star Ball**  
    generator ("would be genuinely novel"). → `research/github-math-art-repos.md`, §6 "Known gaps".

## 🔧 In flight (this session)

*   **Over-Under Screen — bridge twist** — currently at v1.12.122 (Hauer-style  
    relief; monotone 90° quarter-twist necks, CW-from-one-hub/CCW-from-other).  
    Awaiting a live verdict on whether the twist sense/amount reads right; a  
    mirror is a single-sign flip. Generator: `math_art/over_under_screen_generator.py`.
*   **Remove** `**if __name__ == "__main__":**` **self-test blocks** — a Blender  
    extension code-review request (94 of 97 modules). Removal has **zero runtime**  
    **effect** (the blocks never run when imported by Blender). Decision pending:  
    _migrate_ the pure-numpy invariant checks into `tests/` first, or _delete_  
    the blocks + their `_run_selftest` bodies outright. → review link in TODO.md  
    ("Remove `if __name__ …`").

## Pattern Engine (unifying `math_art/patterns/` package)

A 6-phase roadmap; individual generators exist (Orbifold Sphere, Hyperbolic  
Tiling, Celtic, Islamic, Wallpaper) but the unifying signature-driven engine  
is not built. Reconcile against the live menu before resuming — Phases 0–1 may  
be partly overtaken. → `research/pattern-engine-plan.md`.

*   **PLANNED** — Orbifold-signature engine (Phase 0): `orbifold.py` parser +  
    `tiling_wrapper.py` common placed-tiles format; refactor existing generators  
    onto it.
*   **PLANNED** — Substitution/aperiodic backend (Phase 2): Penrose P3 →  
    hat/spectre with Tile(a,b) slider; optional cut-and-project tie to  
    `polytope4d_generator.py`.
*   **PLANNED** — Interlace backend (Phase 3): mirror-curve tracer → medial-graph  
    "interlace any tiling." Open question: generator vs. post-process modifier.
*   **PLANNED** — Mode B continuous Hauer/Carlberg modules (Phase 4). Open risk:  
    **adjacency fidelity** — how much edge/corner metadata the common format must  
    carry for robust C¹ boundary matching on hyperbolic/aperiodic (not just  
    wallpaper) cases.
*   **PLANNED** — Wrap-onto-surface + SVG/net export (Phase 5); shares the SVG  
    writer with stacked-slices.

## Fathauer additions (R1–R7)

All open recommendations (none built yet). Top two by payoff are R1 and R2.  
→ `research/fathauer-review.md`, §3.

*   **PLANNED** — R1 **Phyllotaxis / Fibonacci spirals** (golden-angle placement  
    on disk/dome/sphere). The one clear missing headline; a genuine coverage gap.
*   **PLANNED** — R2 Crochet branching lobes + graded/crest growth (enhance the  
    current single-ratio crochet).
*   **PLANNED** — R3 Fractal knots: substitution method + decorated-tiling links  
    (the decorated-tiling sub-part is a near-free win).
*   **PLANNED** — R4 Fractal-tiling dissection families (verify `fractal_reptile`  
    coverage first), R5 Haüy fractal-polyhedron variants, R6 Goldberg GP(1,2)  
    "spiked" ceramic preset, R7 n-fold monkey saddle `z=ρⁿcos(nφ)` (trivial add).

## Hauer / Carlberg screens

Modular Screen shipped. Remaining: → `research/hauer-carlberg-screens-research.md`.

*   **BLOCKED** — Design 1–5 facsimiles (source material) — see top section.
*   **BLOCKED/dropped** — Scherk-graph screen — see top section.
*   **PLANNED** — Custom module from a mesh/`.obj` file: bundle cells in  
    `math_art/modules/`, array on grid, optional weld/solidify.

## Knots & KnotPlot

Knot Carpets and Celtic-2D are built; the rest of the Tier catalog is open.  
→ `research/knotplot_research.md`, §5.

*   **PLANNED (Tier 1)** — Harmonic/Lissajous/Chebyshev/billiard knots; Petal  
    knots; Turk's-head/decorative rope knots; Rational/2-bridge/pretzel/twist  
    from Conway notation.
*   **PLANNED (Tier 2)** — real knot-energy/ropelength relaxation (current relaxer  
    is "smoothing+repulsion, not a knot energy"); DT/Gauss-code input + 11+  
    crossing catalog; custom braid word / Brunnian links; satellite/cable knots;  
    Seifert surface from a scene curve (see Seifert section); Antoine's necklace  
    enhancements.
*   **PARTIAL (Tier 3 infra)** — consolidate the **four near-duplicate curve→tube**  
    **helpers** into one shared RMF-tube module, and add a shared knot-energy  
    module ("the two biggest consolidation opportunities"); invariant-readout  
    panel.
*   **PLANNED (Tier 4)** — 4D spun/twist-spun knotted spheres; Lorenz knots;  
    hyperbolic census presentation.

## Polyhedra — dMcCooey completeness program

**Essentially complete.** → `research/dmccooey-polyhedra-plan.md`, §5.
Done since the plan: all Platonic/Kepler-Poinsot/Archimedean/Catalan; the 75
uniform polyhedra + ~47 duals; star prisms/antiprisms/dipyramids/trapezohedra;
all 92 Johnson solids; the Conway families (hulls/propellor/chamfer/truncated/
rectified/dipyramids) via the Regular Solid families; geodesics Class I/II/III
+ geodesic cubes/RTs + Goldberg duals; compounds; the exact biscribed solids
(27/31, see below); the **full 59-stellation engine** (`stellation_engine.py`);
the toroidal generator (Császár/Szilassi, Regular-Faced Toroid, Knotted
Dodecahedron, polygon-ring, and the tiled-torus mode wrapping 18 uniform
tilings); the Notable gallery (Escher, Dürer, Jessen, Bilinski, Schönhardt,
Tetrated Dodecahedron, 132-Pentagon, Associahedron, Klein {3,7}₈, genus-6
{6,4}/{4,6} maps); and the **53 Canonical Polyhedra** (11 Greater Self-Dual +
42 Simplest-Canonical-per-symmetry, `canonical_polyhedra_generator.py`).

Genuinely remaining (specialist individual pieces):

*   **PARTIAL / BLOCKED** — **Biscribed chiral: 4 of 31.** #20/#24 (Propello
    Truncated Icosidodecahedron + dual) and #27/#31 (L-Propello L-Snub Cube +
    dual) DO biscribe (McCooey lists their r/R) but the current solver cannot
    reach their root. Attempts, all documented in
    `math_art/biscribed_solids_generator.py` (`TODO(biscribed-solver)`):
    LM multi-start (`solve_chiral_g`, fails to find the basin); iterative-
    projection relaxation (globally convergent but to a DEGENERATE r→0 collapse
    — all faces through the centre — which trivially zeroes the face-distance
    variance). Next: a convexity-preserving / **reciprocal** projection (primal
    AND dual verts on spheres, sending the r→0 collapse to poles at infinity),
    or a homotopy from McCooey's construction.
*   **PARTIAL** — **Higher-genus regular maps: ~11 of 14.** Klein {3,7}₈ (genus
    3) and the {6,4}/{4,6} pair (genus 6) shipped; the rest (heptagonal
    dodecahedra, other genus 3–11 Schulte-Wills maps) each need their own
    literature coordinates (canonicalize the abstract map, or re-derive).
*   **PLANNED** — **A few named individual toroids** the parametric generators
    don't produce directly: **Borromean Rings** and the "**iris**" toroids
    (dmccooey categories 38–41). The tiled-torus + polygon-ring + named-toroid
    coverage handles the archetypes and most instances, not every named model.
*   **PLANNED** — **General stellation engine** beyond the icosahedron (RT
    stellations, full arbitrary-solid diagram enumeration). The icosahedron's
    59 are done via robust 3-D cell enumeration; the superseded 2-D DCEL
    exact-plane path was abandoned, not fixed. → `research/icosahedron-stellations.md`.
*   **N/A** — Symmetry-census pages (dmccooey categories 45–47) are statistics
    tables (millions of canonical solids counted by symmetry) — nothing to
    generate.

## Stacked slices & fabrication export

Implementation plan, not yet built. → `research/stacked-slices-plan.md`.

*   **PLANNED** — Flavor A: Geometry-Nodes solid-stack asset (Repeat Zone +  
    boolean-intersect slabs). Can't export SVG / laser-accurate prisms.
*   **PLANNED** — Flavor B (the fabrication-useful half): Python/bmesh  
    `bisect_plane` → island detection → per-region SVG (`fill-rule:evenodd`, mm  
    units) with a hand-rolled ~30–40-line SVG writer. First prototype target: the  
    **woven polyhedron** output (exercises multi-loop/hole slices). Shares the  
    SVG writer with Pattern Engine Phase 5.

## Curvilinear interlace

Designed this session; not built. → `research/curvilinear-interlace.md`.

*   **PLANNED** — Stage-1 curve-family front-end (stages 2–4 reuse the Knot  
    Carpet crossing/alternation + `pattern_common` strapwork). Roadmap: Wavy  
    Plaid MVP → boundary frame → Warped Grid + Polar → Guilloché → graduate to  
    `curvilinear_interlace_generator.py`. Recommended: prototype as a new  
    `source` mode on the Knot Carpet (~80% reuse). Open risks: 2-colorability  
    seams in dense guilloché, O(n²) crossing perf, grazing crossings, watertight  
    caps for open-strand relief.

## Seifert surfaces

*   **DONE** — the surface finish is resolved. `seifert_surface_generator.py` was  
    rewritten to build and smooth via a vendored port of van Wijk & Cohen's  
    SeifertView (`math_art/seifert/`, from `research/sierfert/`): braid →  
    disk/band combinatorics → light KnotPlot relaxation → Catmull-Clark →  
    **implicit mean-curvature fairing** (Desbrun et al., rim pinned → discrete  
    minimal surface). Topology invariants (χ, boundaries, orientability) are  
    computed combinatorially and verified against the finished mesh. The old  
    hand-rolled `build_seifert` / `dynamic_relax` / `_relax_free` (which "did not  
    reach SeifertView's finish") are gone. The port needs only NumPy: `fair.py`'s  
    SciPy `spsolve` was reimplemented as a NumPy Jacobi solve on the identical  
    matrix; `solid.py` (SciPy + scikit-image thickening) and the KD-tree  
    `self_clearance` were dropped (Blender does its own solidify). Bonus: the  
    non-orientable **Turnback** and **Min-crosscap** state surfaces are now  
    exposed too.  
*   **PLANNED** — still open: **Seifert surface from an arbitrary scene curve**  
    (input is braid-word only). A new input path only has to produce  
    `(feet, bands)` + a predicted `(chi, b, orientable)`; the port's realisers  
    and finishing pipeline then apply unchanged (see `research/sierfert/ALGORITHM.md`  
    §10 "Extension points"). → `research/knotplot_research.md`.

## Minimal surfaces (Weierstrass zoo)

Catalog built from the minimalsurfaces.blog harvest (`research/msblog_harvest/`,  
160 surfaces indexed, 114 with clean g/dh). M1/M2 and a chunk of M3 shipped;  
these are the flagged residuals and the next tiers.

*   **PARTIAL** — **Tilted Scherk (doubly periodic) tiled appearance not exactly**  
    **right.** Shipped (trim-and-snap truncation → open wall rims, no blocky clamp  
    plateaus), built as a horizontal reparametrization of the Scherk graph  
    (`_tilt_scherk_doubly`, `Rho=1` == `SCHERK1`, single connected component). But  
    at Cells 3×3 / radius 1.2 it reads as a **boxy lattice of open cells with**  
    **near-flat vertical walls**, not a smooth leaning/organic tilted corrugation.  
    Note the surface is **non-embedded by design** — the López–Ros tilt makes the  
    half-planes self-intersect (Weber's page), so it can't be a clean embedded  
    surface. Follow-ups: make the walls read as the true leaning tilted sheets  
    (currently flat/rectangular); and/or present only the single fundamental cell  
    to sidestep the boxy tiling; residual fine ribbing at strong tilt (radius  
    ~1.5). → https://minimalsurfaces.blog/home/repository/doubly-periodic/tilted-scherk/
*   **PLANNED** — **Higher-genus Chen–Gackstatter (g2/4/5).** Harvested branch  
    constants are captured and correct, but the engine's only hyperelliptic  
    assembler is hard-specialized to the Costa–Hoffman–Meeks cut structure (gives  
    wrong-χ non-manifold meshes on CG data). Needs a **dedicated cyclic-cover**  
    **assembler** (integrate one angular wedge with an Enneper-aware trim, then  
    rotate/reflect/weld). Also blocks `symm-chen-gackstatter-gn`. →  
    `research/msblog_harvest/higher_genus.json`.
*   **PLANNED** — **Doubly-periodic block (~20): Wei g2/g3, Karcher–Scherk**  
    **g2/g3/g4, KMR-2/3, RTW.** All share one reciprocal-paired hyperelliptic form  
    (`dh=1/z`) with harvested solved constants — a single abstraction + a  
    doubly-periodic tiler (mind the connectivity lessons from Scherk) unlocks the  
    set. → `research/msblog_harvest/doubly_periodic.json`.
*   **PLANNED (Tier 3 exotics)** — **genus-1 helicoid**, **Callahan–Hoffman–Meeks**,  
    Costa–Wohlgemuth, Catenoid–Enneper g2/3/4 — solved constants/seeds harvested;  
    each needs a period solve or the cyclic-cover assembler. →  
    `research/msblog_harvest/{singly,higher_genus}_periodic.json`.
*   **PLANNED (Tier 4)** — Weber–Wolf g3/g4, Kapouleas, Connor, higher Wei/RTW —  
    research-grade (orthodisk / flat-structure / multi-dim Newton); attempt-all  
    per the plan, gate on period residual.
*   **BLOCKED (small)** — **Björling clothoid** (needs a complex-Fresnel  
    evaluator); **plane-with-catenoids** (branched `g=Rho/√z`; needs an  
    upper-half-plane + Schwarz-reflection integrator, not radial disk rays);  
    **antiprismatic k-noid general-n** (harvested numeric (a,ρ) don't close the  
    period beyond nn=5 — only nn=5 shipped).
*   **PARTIAL** — **Exact-WE TPMS Bonnet family (**`**PGD**`**).** Presets show the clean  
    nodal P/Gyroid/D; Custom sweeps the exact Bonnet morph. Exact **tiled** cells:  
    Schwarz P (mirror-plane) is clean; Schwarz D is watertight-ish (~6%  
    non-manifold, rounder than nodal); the **chiral Gyroid could not be tiled**  
    **watertight** (kept as the fundamental piece). Follow-up: a θ-tracking /  
    chiral-aware space-group assembler for the Gyroid cell.

## Other from-scratch / infrastructure gaps (from the repo survey)

Flagged in `research/github-math-art-repos.md` as not-yet-in-repo (reimplement  
from papers; no-license repos are reference-only):

*   **PLANNED** — geometry-nodes emission mode for generators; perforated-shell /  
    congruent-shell splitting modifier (voronoizer + manifold3d + libigl offset);  
    TPMS generator; Hopf fibration; 4D polytope slicing/projection; Wythoff  
    across all three curvatures; escape-time fractal mesher; quasicrystal suite;  
    isohedral Escher-tile + Farris wallpaper displacement; differential-growth /  
    reaction-diffusion on mesh.