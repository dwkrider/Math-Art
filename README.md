# Math Art — Blender Add-ons

Three single-file Blender add-ons for mathematical sculpture:

1. **Scherk-Collins Sculpture Generator**
   (`src/scherk_collins_generator.py`) — a re-implementation of the
   geometry engine of Carlo Séquin's *Sculpture Generator I*.
2. **Minimal Surface Toolkit**
   (`src/minimal_surface_toolkit.py`) — classic parametric minimal
   surfaces, triply-periodic minimal surfaces, and a Plateau solver
   that spans minimal surfaces across arbitrary curves.
3. **Seifert Surface Generator**
   (`src/seifert_surface_generator.py`) — Seifert surfaces for knots
   and links from braid words, after van Wijk & Cohen's *SeifertView*.

Only **geometry** is generated. Materials, textures and rendering are
left to Blender. Both add-ons can emit dense meshes or compact NURBS
surfaces.

---

# 1. Scherk-Collins Sculpture Generator

A Blender re-implementation of the geometry engine of Carlo Séquin's
**Sculpture Generator I** (the 1996/97 program that designed the
Scherk-Collins saddle-chain sculptures created with Brent Collins —
*Hyperbolic Hexagon II*, *Heptoroid*, the *Minimal Trefoil*, etc.).

![Hyperbolic Hexagon](renders/preset_hex.png)

## Install

1. Blender → `Edit > Preferences > Add-ons > Install from Disk…`
2. Pick `src/scherk_collins_generator.py`
3. Enable **"Scherk-Collins Sculpture Generator"**

Tested on Blender 5.1 (needs ≥ 4.2).

## Use

- `Add > Mesh > Scherk-Collins Sculpture` — choose a preset
  (Hyperbolic Hexagon, Minimal Trefoil, Monkey-Saddle Trefoil,
  Heptoroid, straight Scherk Tower) or the plain default.
- Open the **N-panel → "Scherk" tab** to edit parameters. With
  *Auto Update* on, the mesh regenerates live as you drag sliders —
  the original program's interactive workflow.
- **Load / Save** reads and writes the original program's spec-file
  format. All 20 `demo*.txt` files shipped with the original
  (`Sculpture Generator\bin\demo\`) load directly, and saved files can
  be loaded back into `generator.exe`. Display-related fields
  (textures, colors, view matrix) are ignored on load and written with
  neutral defaults on save.
- Export for 3D printing with Blender's built-in STL/3MF exporters.
  Generated solids are watertight, manifold meshes.

## Parameters (same meaning as the original sliders)

| Slider | Range | Effect |
|---|---|---|
| Branches | 1–10 | Order of the saddles (2 = classic Scherk, 3 = monkey saddle…) |
| Storeys | 1–16 | Number of hole/saddle units in the chain |
| Storey Height | 0.1–5 | Vertical stretch of each storey (1.5 ≈ natural proportions) |
| Flange Width | 0.7–5 | Truncation half-width; below ≈0.88 the holes break open |
| Thickness | 0–0.5 | Wall thickness (0 = zero-thickness surface, no rims) |
| Rim Bulge | 0–4 | Outward bulge of the rounded rim beads |
| Twist | −900–1080° | Total axial twist along the chain |
| Azimuth | −360–360° | Fixed turn of the profile about the tower axis |
| Warp | 0–1080° | Bend into an arch/toroid; 360° closes the ring, >360 wraps multiply |
| Detail | 1–16 | Tessellation density |
| Stretch X/Y/Z | 0.2–5 | Affine stretch (the "Totem" trick) |
| Overall Scale | — | Uniform output scale (Blender-side extra) |

The panel shows whether the current twist closes the ring smoothly.
A ring closes when `(twist + storeys·180/branches) mod (360/branches) = 0`
— consecutive storeys of a Scherk tower are naturally rotated by
180°/branches, and the toroidal warp must reconcile that at the seam.
(All closed-ring demo files of the original satisfy this.)

## How the geometry works

The tower template is the exact singly-periodic Scherk minimal surface
`sin z = sinh x · sinh y`, parametrized per height level `c = sin z` as

    x(σ) = asinh(√c · e^{σΛ}),  y(σ) = asinh(√c · e^{−σΛ}),
    Λ = ln(sinh W / √c),  σ ∈ [−1, 1]

which places the curve ends exactly on the truncation planes
(`W` = flange). Higher-order saddles compress the 90° wedge of this
curve into a 180°/branches wedge. Storeys stack with the natural
180°/branches alternation; then azimuth + linear twist rotate the
cross-section, and the warp bends the tower around a circle whose
radius preserves arc length. Thickness is applied as a two-sided
offset with semicircular rim tubes (elongated by rim bulge) welded on
— including correctly through Möbius (single-sided) configurations.

Differences from the original: Séquin's C code is not public, so
tessellation layout and the exact flange/bulge profiles are
re-derivations, not byte-identical copies — shapes match the published
sculptures and the demo files visually. Thickness is uniform (applied
after the affine stretch). Storey-joint creases are C1 like the
original's; add a Subdivision Surface modifier if you want extra
smoothness.

**NURBS output**: enable *NURBS Output* to get a compact NURBS surface
instead of a mesh — one clamped patch per half-wedge of each storey,
with far fewer control points (lower *Detail* for even fewer). NURBS
mode outputs the smooth mid-surface only: thickness and rim bulge do
not apply (convert to mesh and add Solidify if needed).

---

# 2. Minimal Surface Toolkit

![Gyroid lattice](renders/min_tpms_g_2cells.png)

Install `src/minimal_surface_toolkit.py` the same way. Everything is
under `Add > Mesh > Minimal Surfaces` and the N-panel **MinSurf** tab.

## Classic parametric surfaces

Enneper (any order), Catenoid, Helicoid, Henneberg, Catalan, Bour,
Richmond, and Scherk's doubly-periodic surface — after Jürgen Meier's
gallery (3d-meier.de, tut25). Output as mesh or as a single NURBS
patch (`Output: NURBS`, ~24×24 control points instead of ~10k
vertices), with correct periodic closure (catenoid seam etc.).

## Triply-periodic minimal surfaces (TPMS)

Schwarz P & D, the Gyroid, Neovius, Schoen's I-WP & F-RD, the
Lidinoid, Split P, and the singly-periodic Scherk tower — the members
of Brakke's TPMS inventory that have standard nodal (level-set)
approximations. Meshed by a vectorized marching-tetrahedra extractor
with gradient-oriented winding; choose cells per axis, resolution and
an optional Solidify thickness for 3D-printable lattices (mesh output
only — an implicit surface has no NURBS parametrization).

## Plateau solver ("Span Minimal Surface")

A lightweight, in-Blender take on what Ken Brakke's **Surface
Evolver** does: pin boundary curves, minimize area. The solver is the
Pinkall-Polthier cotangent-Laplacian iteration with a conjugate-
gradient core (numpy), validated against exact solutions — a planar
circle relaxes to a flat disk (area π to 0.07%), and two parallel
rings relax to a catenoid (waist radius correct to 0.06%).

- Select **one closed curve** (Curve object or closed mesh edge loop)
  → disk-type minimal surface spanning it.
- Select **two closed curves** → annulus-type minimal surface between
  them (loops are auto-aligned).
- `Add > Mesh > Minimal Surfaces > Circle to Torus Knot` builds the
  classic *minimal surface with a trefoil knot as inner edge and a
  circle as outer edge* (any (p,q) torus knot; trefoil = (2,3)).

Span operators can also emit NURBS (`NURBS Output`); note that where
the surface curls very tightly (e.g. hugging a knotted edge) the NURBS
approximation may ripple — increase rings/samples or use mesh output.

![Trefoil-circle span](renders/min_trefoil_circle.png)

---

# 3. Seifert Surface Generator

![Borromean rings Seifert surface](renders/seifert_borromean.png)

Install `src/seifert_surface_generator.py`; find it under
`Add > Mesh > Seifert Surface` and in the **MinSurf** N-panel tab.

Enter a knot or link as a **braid word** — letters (`a` = σ₁,
`A` = σ₁⁻¹, `aBaB` = figure-8) or integers (`1 -2 1 -2`) — or pick a
preset: trefoil, figure-8, cinquefoil, granny & square knots, Hopf
link, Solomon's link, Borromean rings, or any (p,q) torus knot.
Seifert's algorithm on the braid closure gives one disk per strand and
one half-twisted band per crossing; the add-on stacks the disks like a
wedding cake and joins their rims with twisted ribbons — the classic
SeifertView presentation. The surface boundary *is* the knot, and is
optionally emitted as a bevelled tube curve.

The operator reports (and stores on the object) the braid, strand and
crossing counts, number of link components, and the surface **genus**
(from χ = strands − crossings). The headless tests verify that the
generated mesh's Euler characteristic and boundary-loop count match
the braid combinatorics exactly for every preset.

*Relax Iterations* smooths the surface with the Plateau solver from
the Minimal Surface Toolkit (soft dependency), pinning the knot — a
soap-film look while keeping the correct topology.

## Files

- `src/scherk_collins_generator.py` — Scherk-Collins add-on (the
  geometry core also runs standalone:
  `python src/scherk_collins_generator.py` prints a smoke test)
- `src/minimal_surface_toolkit.py` — Minimal Surface Toolkit add-on
  (standalone run validates the Plateau solver; needs numpy)
- `src/seifert_surface_generator.py` — Seifert Surface Generator
  (standalone run checks Euler characteristics of all presets)
- `tests/test_scherk.py`, `tests/test_minimal.py`, `tests/test_nurbs.py`,
  `tests/test_seifert.py` — headless tests, e.g.
  `blender --background --factory-startup --python tests/test_scherk.py`;
  they check mesh integrity (watertight/manifold), validate the solver,
  and render to `renders/`
- `renders/` — sample renders

## References

- [Scherk-Collins Sculpture Generator](https://people.eecs.berkeley.edu/~sequin/SCULPTS/scherk.html) (C. H. Séquin)
- C. H. Séquin, *Virtual Prototyping of Scherk-Collins Saddle Rings*, Leonardo 30(2), 1997
- C. H. Séquin, H. Meshkin, L. Downs, *Interactive Generation of Scherk-Collins Sculptures*, I3D '97
- C. H. Séquin, *15 Years of Scherk-Collins Saddle Chains*, UCB/EECS-2010-41
- [Ken Brakke — Triply Periodic Minimal Surfaces](https://kenbrakke.com/evolver/examples/periodic/periodic.html) and the [Surface Evolver](https://kenbrakke.com/evolver/evolver.html)
- [Jürgen Meier — Minimal surface gallery](http://www.3d-meier.de/tut25/Seite0.html)
- [Mathematica SE #69131 — minimal surface with trefoil knot inner edge](https://mathematica.stackexchange.com/questions/69131/)
- U. Pinkall, K. Polthier, *Computing Discrete Minimal Surfaces and Their Conjugates*, Exp. Math. 2(1), 1993
- [SeifertView](https://vanwijk.win.tue.nl/seifertview/) — J. J. van Wijk, A. M. Cohen, *Visualization of Seifert Surfaces*, IEEE TVCG 12(4), 2006
