# Math Art — Blender Extension

Mathematical sculpture generators for Blender, packaged as a single
modern **extension** (Blender 4.2+):

**Install:** build the zip (below) or grab `dist/math_art-*.zip`, then
in Blender: `Edit > Preferences > Get Extensions > ⌄ (top-right) >
Install from Disk…` — or just drag the zip into the Blender window.
Everything appears under **Add > Mesh > Math Art** plus the Scherk /
Minimal Surfaces N-panel tabs. (If you previously installed the individual
files as legacy add-ons, remove those first.)

**Build the extension:**

    blender --command extension build --source-dir math_art --output-filepath dist/math_art-1.0.0.zip

Each module in `math_art/` is also still installable on its own as a
classic single-file add-on via `Install from Disk`.

The modules:

1. **Scherk-Collins Sculpture Generator**
   (`math_art/scherk_collins_generator.py`) — a re-implementation of the
   geometry engine of Carlo Séquin's *Sculpture Generator I*.
2. **Minimal Surface Toolkit**
   (`math_art/minimal_surface_toolkit.py`) — classic parametric minimal
   surfaces, triply-periodic minimal surfaces, and a Plateau solver
   that spans minimal surfaces across arbitrary curves.
3. **Seifert Surface Generator**
   (`math_art/seifert_surface_generator.py`) — Seifert surfaces for knots
   and links from braid words, after van Wijk & Cohen's *SeifertView*.
4. **Polyhedron suite**, after Adrian Rossiter's
   [Antiprism](https://www.antiprism.com) tools — five independent
   add-ons:
   - `math_art/conway_operators.py` — **Conway notation** (`dkC`, `taD`,
     `k3sT`…) with seeds T C O D I / Pn / An / Yn, operators
     d a k g c r + t j e o b m s n z, and Hart-style
     **canonicalization** (edges tangent to the sphere). Verified
     against 17 textbook polyhedra (snub cube 24/38, truncated
     icosahedron 60/32, …). Styles: Solid, Leonardo (da Vinci)
     open-faced panels, or Wireframe.
   - `math_art/zonohedra_generator.py` — **zonohedra** from vector stars
     (rhombic dodeca/triaconta/enneacontahedron, random stars), plus
     **polar zonohedra** and Russell Towle's **rhombic
     spirallohedra** via a direct port of Antiprism's
     `make_polar_zonohedron` (the preset equals `zono -P 12,4`).
     Styles: **Solid**, **Leonardo (da Vinci)** open-faced panels
     (via the shared Leonardo Style modifier below), or
     **Wireframe** edge struts.
   - `math_art/waterman_generator.py` — **Waterman polyhedra** (hulls of
     FCC lattice points; W1 = cuboctahedron, high roots approach a
     sphere). Styles: Solid, Leonardo (da Vinci), or Wireframe.
   - `math_art/rotegrity_generator.py` — **rotegrity / nexorade** strap
     spheres over Platonic and geodesic seeds (twist + extension
     sliders). Coloring **by strap length class** (as in physical
     rotegrity kits; default), per strap, or none — with
     `strap_index` / `length_class` face attributes.
   - `math_art/weave_generator.py` — **woven-strand models** driven by
     poly_weave's **pattern language**: a program
     `[C|L][bV,bE,bF][:up,side,along…]steps[tl|tr|tb]` walked over the
     seed's flag triangles (steps `V E F` pivot, `v e f` cross, `R`
     reverse, `-` stay). The default `vfe` is the classic
     straight-ahead weave (cube → 4 hexagons, icosahedron → 6
     decagons); `FEV` gives corner circuits (icosahedron → 12
     interwoven pentagons), plus vertex/face rings and raised/wavy
     variants. Strands are coloured individually by default
     (`strand_index` face attribute; Coloring = None to disable).
5. **Sculptural forms**, after constructions on George W. Hart's
   pages — five more modules:
   - `math_art/polylinks_generator.py` — **regular polylinks**:
     interlocked polygon frames on Platonic face planes (presets for
     the classic 4-triangle, 6-square, 6/12-pentagon and 20-triangle
     tangles; size/rotation/offset sliders control the linking).
   - `math_art/platonic_twist_generator.py` — **Platonic twist
     sculptures**: shrunken, separated faces reconnected by ribbons
     with any number of half twists (exactly welded surface; optional
     Solidify thickness).
   - `math_art/fractal_polyhedron_generator.py` — **fractal
     polyhedra**: recursive copies at vertices/faces/edges (vertex
     mode at scale ½ on a tetrahedron = the Sierpinski tetrahedron).
   - `math_art/symmetrohedron_generator.py` — **symmetrohedra** after
     Kaplan & Hart: regular m·(axis order)-gons placed covariantly on
     the T/O/I symmetry axes (inscribed in the unit sphere), convex
     hull filling the gaps; per-class multiplier/size/phase sliders.
   - `math_art/twisted_torus_generator.py` — **twisted torus**: an
     n-gon revolving around a ring while twisting in exact 360/n
     steps, giving gcd(n, steps) helical bands; corner-rounding
     slider. *Triangle Shrink* < 1 shrinks each side about its
     midpoint and emits **one mesh object per helical band** (solid
     ribbons with sharp border edges); at 1 the classic contiguous
     tube is produced. Coloring: **Per Strip** (default) gives each
     of the n visible helical strips its own material; **Per Band**
     colours by connected band — gcd(n, twist) of them, so coprime
     twists give a single colour; `strip_index` / `band_index` face
     attributes are always written.
6. **Polyhedral tangles** (math_art/tangle_generator.py) — compounds
   of interwoven polyhedron frames: 2/5/10 tetrahedra (the classic
   *Tetra Tangle*), 3 and 5 cubes, 5 octahedra. Default style is
   **Hollow Faces (da Vinci)**: solid face panels with openings,
   built as the shell between the polyhedron scaled out and in by
   half the thickness so panels share the scaled polyhedron
   vertices — joints along edges and at vertices are exact,
   watertight mitres, as in Leonardo's *De divina proportione*
   models. **Edge Struts** (Lang polypolyhedra style) are square
   sticks trimmed back from each vertex with flat caps and a faceted
   convex-hull knuckle filling every joint. Per-component rotation,
   colouring and a `component_index` face attribute.
7. **4D polytopes** (`math_art/polytope4d_generator.py`) — all six
   regular convex 4-polytopes (5-cell, tesseract, 16-cell, 24-cell,
   600-cell, 120-cell) as strut-and-sphere edge frameworks. Edges are
   **straight** (4D perspective; small distances approach a Schlegel
   diagram) or **curved**: vertices on the 3-sphere, edges as
   great-circle arcs, stereographically projected so every edge is a
   circular arc. 4D rotation sliders (XW/YW/ZW/XY), and struts can
   taper with the local projection scale. Vertex/edge counts verified
   (120-cell: 600 vertices, 1200 edges). A **Leonardo (da Vinci)**
   style renders one flat open panel per 2D face of the polytope
   instead of edge struts — projected faces are planar even in
   curved mode, because stereographic projection maps the circle
   through a face's vertices to a circle in space (face counts
   verified: 10/24/32/96/720/1200).
8. **Symmetric Sculpture designer**
   (`math_art/symmetric_sculpture_generator.py`) — an interactive
   Blender adaptation of George Hart's sculpture design software
   (*Symmetric Sculpture*, J. of Mathematics and the Arts 1(1), 2007
   — the tool behind *Twisted Rivers*, *Tumbleweed*, *Frabjous*,
   *Spaghetti Code*). Pick a symmetry group (icosahedral 60 /
   octahedral 24 / tetrahedral 12) and a **plane family** — the
   extended face planes of an icosahedron (20×3-fold), dodecahedron
   (12×5-fold), rhombic triacontahedron (30×2-fold) or
   hexecontahedron (60×1). Presets reproduce the setups of three of
   Hart's pieces — **Twisted Rivers** (C-shaped rivers, 20 icosahedral
   planes), **Tumbleweed** (five-armed pinwheels, 12 dodecahedral
   planes) and **Frabjous** (30 S-shaped parts in the triacontahedral
   planes, each S welded from two half-arms by the in-plane 2-fold
   symmetry) — or pick **Custom** to choose the group and family
   yourself. The operator creates three linked objects: a flat **Motif** mesh in one representative plane, a
   wireframe **Guides** object showing the stellation pattern (the
   lines where the other planes cut this one — Hart's 2D editor
   background), and a **SymSculpt** object whose Geometry Nodes
   modifier instances the motif under every rotation of the group.
   Edit the motif in edit mode — or grab/rotate the whole motif
   object — and all copies update **live**, exactly Hart's
   workflow. In the default **design view** the motif is opaque
   (orange) while the replicated copies are ghosted with a shared
   translucent material (and the copy coinciding with the motif is
   hidden), so the editable part always stands out; tick the
   modifier's **Full Sculpture** checkbox to show all copies with
   the motif's own material for export or final rendering. The modifier's **Shell** input radially extrudes the
   result by a fraction of its distance from the origin (Hart's ~4%
   extrusion, planarity-preserving) for a printable solid; **Weld**
   merges copies that meet exactly on the plane-intersection lines.
   A `copy_index` attribute distinguishes copies for shading.
9. **Leonardo Style modifier** (`math_art/leonardo_style.py`) — a
   reusable Geometry Nodes group (applied by *Object > Leonardo
   Style* or the Math Art menu) that turns **any closed mesh** into
   a Leonardo da Vinci open-faced model, as in his polyhedron
   illustrations for Pacioli's *De divina proportione*: every face
   is inset individually, the centre is cut out, and the frame
   surface is extruded into a solid shell. Border and Thickness are
   live modifier inputs; the zonohedron generator's Leonardo style
   uses the same shared node group.
10. **Fractal sponges** (`math_art/sponge_generator.py`) — Menger
    sponge, Vicsek fractal and Sierpinski carpet as single
    watertight exterior surfaces (only faces between solid and
    empty cells are emitted), plus the Sierpinski tetrahedron and
    octahedron as point-contact solids. Replication counts and
    manifoldness verified per level.
11. **Space-filling curves**
    (`math_art/space_curve_generator.py`) — Hilbert and Moore
    curves in 2D and 3D (after the Wolfram *Hilbert and Moore 3D
    Fractal Curves* demonstration). Hilbert points via Skilling's
    transpose algorithm; the Moore loop chains 4 / 8 rotated
    Hilbert blocks around a Gray-code ring into a single closed
    circuit (unit-step continuity and closure verified for every
    order). Output is a poly curve with bevel radius and optional
    Chaikin corner rounding.
12. **Oloid & ruled surfaces** (`math_art/oloid_generator.py`) —
    the oloid from the exact Dirnböck–Stachel ruling (every ruling
    has length √3; watertight), the two-circle roller (hull), and
    ruled strips between two perpendicular circles after Kit
    Wallace's *ruled Möbius strip* posts — including his true
    one-edged Möbius surface (χ = 0, single boundary loop,
    verified).
13. **Regular solids** (`math_art/regular_solids_generator.py`) — a
    complete *Add Regular Solid*, organised by family: the 5
    **Platonic** solids; the 4 **Kepler–Poinsot** star polyhedra as
    true intersecting faces (pentagrams star-triangulated; the
    great stellated dodecahedron's pentagram points are the second
    ring of each dodecahedron face); all 13 **Archimedean** and all
    13 **Catalan** solids via Conway notation + canonicalization
    (V/F counts verified for all 26); uniform **prisms and
    antiprisms**; and the **Johnson solids J1–J48** — every
    pyramid/cupola/rotunda solid and their elongated,
    gyroelongated and bi- (ortho/gyro) combinations, composed with
    exact unit-edge coordinates (all 47 verified to machine
    precision, ortho/gyro pairings confirmed by like-meets-like
    contact tests; the augmented/diminished J49+ are not included).
    Options: generic **stellation** (pyramid to the intersection of
    neighbour planes — octahedron → stella octangula), Solid /
    Leonardo (da Vinci) / Wireframe styles, coloring by face
    size, and **Congruent Pieces**: split the shell into N
    congruent, connected pieces (rotated copies of one another,
    one object each, for printing and reassembly). The rotation
    group is derived from the mesh itself; a free subgroup of
    order N supplies the pieces, grown as compact connected
    fundamental domains (icosahedron → 2/4/5/10, cube → 2/3/6,
    rhombic triacontahedron → 3/5, …); an Explode slider
    separates them visually.
14. **Prime knots** (`math_art/prime_knot_generator.py`) — **all
    249 prime knots up to 10 crossings** (Rolfsen table) from the
    minimum braid words of T. Gittings. The braid closure is laid
    around a circle and relaxed with smoothing + self-repulsion
    into a rounded rope presentation. Every braid word is verified
    programmatically: single-component closure, and the Alexander
    polynomial (computed via the reduced Burau representation)
    matches Gittings' published value at t=10 for all 249 entries.
    Output styles as in *Torus Knot +*: Bézier / Poly / NURBS curve
    with bevel radius, or a mesh tube (seam-corrected
    parallel-transport frames); custom braid words, mirror option,
    relax controls.
15. **Strange attractors** (`math_art/attractor_generator.py`) —
    **all 38 chaotic attractors of Chaotic Atmospheres' *MATHRULES*
    series** (Lorenz, Aizawa, Thomas, Halvorsen, Dadras, Burke-Shaw,
    Chua, Dequan Li, the three-scroll unified systems, …, plus
    Zhou-Chen as a bonus), each a preset with the ODE system and
    parameters after Jürgen Meier's 3d-meier.de compilation — the
    artist's stated source — transcribed from the original formula
    sheets and posters. The 4D systems (Qi, Lorenz-Stenflo) and the
    6D coupled Lorenz pair are drawn as 3D projections. Fixed-step
    RK4 integration (Euler for Lorenz Mod 1, whose reference render
    depends on Euler's numerical dissipation) with the transient
    discarded, emitted as a Poly/Bézier curve with bevel; options:
    even arc-length resampling, speed-based taper (slow flow
    thickens the tube, the look of the original renders), and a
    polygonal cross-section profile for printable tubes. Every
    preset is verified by the module's self-test to stay bounded
    without collapsing to a fixed point.
16. **Stellated Surface Weave**
    (`math_art/stellated_weave_generator.py`) — twelve folded
    strips interlocking along the pentagram planes of a small
    stellated dodecahedron, a faithful port of Shengyi Wang's
    construction (exactly 2-manifold at every strip width;
    per-strip coloring).
17. **Stereographic projection spheres**
    (`math_art/stereographic_projection_generator.py`) — Segerman-
    style shadow spheres: a pattern (grid / polar / tiling /
    beachball) on the sphere, projected from the north pole so a
    point light there casts the flat pattern as its shadow.
18. **Hyperbolic honeycombs**
    (`math_art/hyperbolic_honeycomb_generator.py`) — {p,q,r}
    honeycombs (including paracompact ones) drawn in the Poincaré
    ball, edges as geodesic arcs.
19. **Algebraic surfaces**
    (`math_art/algebraic_surface_generator.py`) — classical
    nodal/smooth algebraic surfaces (Clebsch cubic, Cayley, Kummer,
    Barth sextic, Togliatti, …) via marching cubes on their
    polynomials.
20. **Space-filling solids** (`math_art/spacefill_generator.py`) —
    honeycomb blocks of space-fillers: cubic, octet truss
    (tet+oct), truncated octahedra, rhombic dodecahedra.
21. **Links & connect sums** (`math_art/link_generator.py`) —
    Hopf, Borromean, Whitehead, torus links, chains and connect
    sums as tube curves.
22. **Topological surfaces**
    (`math_art/topological_surface_generator.py`) — Klein bottles
    (classic and figure-8), Boy's and Roman surfaces, cross-cap,
    genus-g handlebodies and twisted strips (χ = 2−2g verified).
23. **Hyperbolic tilings**
    (`math_art/hyperbolic_tiling_generator.py`) — {p,q} tilings on
    the Poincaré disk, hemisphere, or pseudosphere.
24. **Geodesic spheres & domes**
    (`math_art/geodesic_generator.py`) — Class I/II geodesic
    subdivisions with dome cuts and four render styles.
25. **Orbifold symmetry spheres**
    (`math_art/orbifold_sphere_generator.py`) — all 14 spherical
    symmetry types as decorated spheres (Conway orbifold
    notation).
26. **Fractal trees & mobiles**
    (`math_art/fractal_tree_generator.py`) — recursive branching
    trees and Calder-style hanging mobiles as curve skeletons.
27. **Curvature Color** (`math_art/curvature_color.py`) — a Styles
    operator coloring any mesh by discrete Gaussian curvature
    (angle defect), saddle regions vs caps.
    The 4D polytope generator also gained **half cutaways, dual
    compounds, and Hopf-fibration cell rings** (`rings` /
    `rings_only`) from the same batch.

![120-cell](renders/form_120cell.png)

![120-cell, Leonardo style](renders/form_120cell_leonardo.png)

![Symmetric sculpture](renders/form_symmetric_sculpture.png)

![Knot 8_18](renders/form_knot_8_18.png)

![Tetra tangle, da Vinci style](renders/form_tangle_t5.png)

![Great stellated dodecahedron](renders/form_gsd.png)

![Snub dodecahedron](renders/form_snub_dodecahedron.png)

![Zonohedron, Leonardo style](renders/form_zonohedron_leonardo.png)

![Tumbleweed preset](renders/form_symsculpt_tumbleweed.png)

![Frabjous preset](renders/form_symsculpt_frabjous.png)

![Platonic twist](renders/form_platonic_twist.png)

![Weave](renders/poly_weave.png)

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
2. Pick `math_art/scherk_collins_generator.py`
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

Install `math_art/minimal_surface_toolkit.py` the same way. Everything is
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

Span operators can also emit NURBS (`NURBS Output`): the solved grid is
faired (per-column arc-length resampling + light net smoothing + a
normal-only mean-curvature polish) and becomes the control net of a
single cyclic NURBS patch — the trefoil span evaluates to within ~4%
of the mesh solution's area with a fraction of the data.

![Trefoil-circle span](renders/min_trefoil_circle.png)

---

# 3. Seifert Surface Generator

![Borromean rings Seifert surface](renders/seifert_borromean.png)

Install `math_art/seifert_surface_generator.py`; find it under
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

**Smoothing.** Two sliders, both using the Plateau solver from the
Minimal Surface Toolkit (soft dependency):

- *Relax Iterations* — pins the knot exactly where the schematic
  construction put it and relaxes only the membrane (soap film on the
  cake-shaped knot).
- *Shape Relax Rounds* — relaxes the **whole shape**: each round the
  knot boundary evolves as an elastic curve (Laplacian smoothing at
  constant total length, with self-repulsion so strands cannot pass
  through each other, which preserves the knot type), then the surface
  re-relaxes to the moved boundary. 40–80 rounds turn the wedding cake
  into the organic, rounded form (see
  `renders/seifert_trefoil_shaped_top.png`). The dark slits at the
  crossings are the genuine twist tunnels of the surface, not defects.
  Note: with large *Band Width* the disks' rims are mostly band
  chords, and the relaxed shape correctly stays plate-like — use the
  default band width for the organic look.

## Files

- `math_art/scherk_collins_generator.py` — Scherk-Collins add-on (the
  geometry core also runs standalone:
  `python math_art/scherk_collins_generator.py` prints a smoke test)
- `math_art/minimal_surface_toolkit.py` — Minimal Surface Toolkit add-on
  (standalone run validates the Plateau solver; needs numpy)
- `math_art/seifert_surface_generator.py` — Seifert Surface Generator
  (standalone run checks Euler characteristics of all presets)
- `tests/test_scherk.py`, `tests/test_minimal.py`, `tests/test_nurbs.py`,
  `tests/test_seifert.py` — headless tests, e.g.
  `blender --background --factory-startup --python tests/test_scherk.py`;
  they check mesh integrity (watertight/manifold), validate the solver,
  and render to `renders/`
- `renders/` — sample renders

## References & Attribution

This project re-implements, ports, or draws on the following prior
work. All credit for the underlying mathematics, algorithms and
original software belongs to their authors.

**Scherk-Collins sculptures** — Carlo H. Séquin & Brent Collins

- [Scherk-Collins Sculpture Generator](https://people.eecs.berkeley.edu/~sequin/SCULPTS/scherk.html), C. H. Séquin, UC Berkeley
- C. H. Séquin, *Virtual Prototyping of Scherk-Collins Saddle Rings*, Leonardo 30(2), 1997
- C. H. Séquin, H. Meshkin, L. Downs, *Interactive Generation of Scherk-Collins Sculptures*, Proc. I3D '97
- C. H. Séquin, *15 Years of Scherk-Collins Saddle Chains*, UCB/EECS-2010-41
- Parameter semantics validated against the demo files shipped with the original *Sculpture Generator* Windows program

**Minimal surfaces**

- [Ken Brakke — Triply Periodic Minimal Surfaces](https://kenbrakke.com/evolver/examples/periodic/periodic.html) and the [Surface Evolver](https://kenbrakke.com/evolver/evolver.html) (the Plateau solver here is a lightweight in-Blender analogue); see also [SE-FIT](https://www.se-fit.com/)
- [Jürgen Meier — parametric surface gallery](http://www.3d-meier.de/tut25/Seite0.html) (the classic parametric surface inventory)
- U. Pinkall, K. Polthier, *Computing Discrete Minimal Surfaces and Their Conjugates*, Experimental Math. 2(1), 1993 (the area-minimization algorithm)
- [Paul Nylander — Scherk-Collins surface notes](https://nylander.wordpress.com/2009/02/10/scherk-collins-surface/) (twist/warp transformations)
- [Mathematica SE #69131](https://mathematica.stackexchange.com/questions/69131/) — the circle-to-trefoil-knot minimal surface construction
- Nodal TPMS approximations after H. G. von Schnering & R. Nesper and subsequent literature
- [xyzdims — TPMS 3D-printing experiments](https://xyzdims.com/tag/triply-periodic-minimal-surface/) (printable-lattice inspiration)

**Seifert surfaces**

- [SeifertView](https://vanwijk.win.tue.nl/seifertview/) — J. J. van Wijk & A. M. Cohen, *Visualization of Seifert Surfaces*, IEEE TVCG 12(4), 2006 (braid-word input, wedding-cake layout, relaxed presentation)
- H. Seifert, *Über das Geschlecht von Knoten*, Math. Annalen 110, 1934 (the algorithm itself)

**Prime knots**

- T. A. Gittings, *Minimum Braids: A Complete Invariant of Knots and
  Links*, [arXiv:math/0401051](https://arxiv.org/abs/math/0401051)
  (Table 1: the minimum braid words for all knots to 10 crossings)
- D. Rolfsen, *Knots and Links*, Publish or Perish 1976 (the knot
  numbering)
- Relaxed rope presentation inspired by R. Scharein's
  [KnotPlot](https://knotplot.com/)

**Regular solids**

- N. W. Johnson, *Convex Polyhedra with Regular Faces*, Canad. J.
  Math. 18 (1966) - the Johnson solid enumeration
- Conway/Hart operator notation and canonicalization as in the
  Conway generator above

**Polyhedra** — after [Antiprism](https://www.antiprism.com) by Adrian
Rossiter ([source](https://github.com/antiprism/antiprism), GPL):

- `conway` — Conway polyhedron notation: J. H. Conway's operator
  notation, extended and popularized by
  [George W. Hart](https://www.georgehart.com/virtual-polyhedra/conway_notation.html);
  canonicalization after G. Hart, *Calculating Canonical Polyhedra*,
  Mathematica in Educ. & Research 6(3), 1997
- `zono` — zonohedra; polar zonohedra & **rhombic spirallohedra**
  after [Russell Towle](http://www.zonohedra.com/) (the spirallohedron
  builder is a direct port of Antiprism's `make_polar_zonohedron`,
  `base/zonohedron.cc`)
- `waterman` — [Waterman polyhedra](http://watermanpolyhedron.com/)
  by Steve Waterman
- `rotegrity` — rotegrity / nexorade strap spheres (Antiprism's
  `rotegrity` program; terminology after Dick Fischbeck and the
  nexorade work of O. Baverel et al.)
- `poly_weave` — polyhedral weaves and the weave **pattern language**
  ([examples](https://www.antiprism.com/examples/200_programs/700_poly_weave/imagelist.html))

**Sculptural forms** — after constructions by
[George W. Hart](https://www.georgehart.com/) (see his
[vibe-coded program examples](https://www.georgehart.com/vibecode/)):

- Stellated Surface Weave and the wavy-circle / torus-knot
  polylink variants: Shengyi Wang ([txyyss](https://github.com/txyyss),
  [math-art](https://txyyss.github.io/math-art/) and the
  [polylink](https://github.com/txyyss/polylink) add-on)
- Regular polylinks: G. Hart, *Orderly Tangles Revisited*
  ([georgehart.com](https://www.georgehart.com/orderly-tangles-revisited/tangles.htm)),
  building on Alan Holden's *Orderly Tangles* (1983)
- Platonic twist sculptures, fractal polyhedra and the twisted torus:
  after the generators on Hart's vibecode page
- Symmetrohedra: C. S. Kaplan & G. W. Hart, *Symmetrohedra: Polyhedra
  from Symmetric Placement of Regular Polygons*, Bridges 2001
- Symmetric Sculpture designer: G. W. Hart, *Symmetric sculpture*,
  Journal of Mathematics and the Arts 1(1), 2007, pp. 21–28
  ([PDF](https://www.georgehart.com/sculpture/Symmetric-Sculpture.pdf))
  — the plane-family replication, stellation-pattern guides and
  radial extrusion are modelled on the software described there
- Leonardo style: after Leonardo da Vinci's open-faced ("vacuus")
  polyhedron illustrations for Luca Pacioli's *De divina
  proportione* (1509); see also G. W. Hart,
  [Leonardo da Vinci's Polyhedra](https://www.georgehart.com/virtual-polyhedra/leonardo.html)

Thanks also to the Blender Foundation — everything here builds on the
Blender Python API and the 4.2+ extensions platform.
