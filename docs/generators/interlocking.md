# Topological Interlocking

## Overview

Assemblies of rigid blocks that **kinematically lock**: once a
peripheral **frame** of blocks is held fixed, no non-empty subset of
the interior blocks can be slid or rotated out without colliding with
its neighbours. The blocks are never glued — the interlocking is
purely geometric. This is the idea introduced by Dyskin, Estrin,
Kanel-Belov and Pasternak (2001) and developed since across
architecture, fracture-resistant materials, and 3D-printed puzzles.

The generator offers several families, from a rigorous single-layer
interlock to modular space-filling blocks and a spherical shell:

- **Interlocking Tetrahedra** — the canonical layer: regular
  tetrahedra in two 90°-rotated orientations on a checkerboard. Each
  cell's horizontal mid-section is a unit square that tiles the plane;
  the x-neighbours block a cell from rising and the y-neighbours block
  it from sinking, so the framed interior locks.
- **Escher / Osteomorphic** — a square fundamental domain has its
  four edges deformed by one profile (the *Escher trick*), is turned
  90° at mid-height, and is lofted back. Every horizontal section
  tiles the plane, so copies interlock. A **sine** profile reproduces
  Estrin's *osteomorphic* saddle block; **tent**, **S-curve** and
  **step** profiles give Versatile-style blocks.
- **Versatile Blocks** — the exact block of Akpanya, Goertzen,
  Wiesenhütter, Niemeyer & Nönnig (Bridges 2023): the interpolation
  between a square and a 1×2 rectangle. It is tiled by translation
  over the diamond lattice into a framed **interlocking layer**
  (checkerboard-coloured for the Truchet look), one block per
  fundamental domain.
- **Interlocking Cubes / Octahedra** — the Kanel-Belov / Dyskin
  *moving-cross-section* layer: cubes (or octahedra) with a body-
  diagonal 3-fold axis vertical, placed as identical translates on a
  honeycomb. Each cell's faces tilt ±35.26° (cube) or ±19.47°
  (octahedron), and because neighbours meet across an edge that
  reverses the tilt sense, adjacent cells share every inclined plane —
  so the framed layer locks.
- **Tetroctahedrille Kitten** — a non-convex block (one octahedron +
  two tetrahedra) glued on the octet-truss (tetrahedral-octahedral)
  lattice, after Akpanya, Goertzen & Niemeyer (2024). It **tiles
  space by pure translation** over the FCC basis and is assembled into
  a space-filling patch.
- **Tetroctahedrille UFO / Cushion** (single) — one octahedron + four
  tetrahedra; shown as a single block, since they admit no integer
  translation lattice.
- **SL Blocks** — self-interlocking octocubes (Shih 2018): an
  S-tetracube fused to an L-tetracube (each a contiguous unit cell).
  Two blocks form a conjugate pair (one turned 180° about a y-axis
  through a shared S corner); pairs chain by the *a*-engagement
  (Rz(−90) then T(1,−1,0)) into a periodic strand — four pairs close
  the a⁴ loop. Verified cube-disjoint.

The result is centred at the origin and fit within a cube of the
chosen **Size**. The Tetrahedra, Escher, Versatile and moving-cross-
section (Cubes / Octahedra) layers are verified framed-interior
interlocks; the Kitten patch, the Bisquare layer, the SL strand, the
interlocking dome and the hendecahedron cluster are verified
non-overlapping assemblies; the Rhom, Rhom Obverse, UFO and Cushion
families are single reference blocks.

### Using it

1. **Add it** from *Add ▸ Mesh ▸ Math Art ▸ Polyhedra ▸ Topological Interlocking*.
2. **Pick the Family.** The genuinely locking layers are **Interlocking Tetrahedra** (the default), **Escher / Osteomorphic**, **Versatile Blocks**, and **Interlocking Cubes / Octahedra** — for these the operator reports *interlocking (framed interior)*. **Bisquare Blocks**, **Tetroctahedrille Kitten**, **SL Blocks**, **Interlocking Dome** and **Bisymmetric Hendecahedron** are further verified assemblies, and **Rhom / Rhom Obverse / UFO / Cushion** are shown as single reference blocks.
3. **Set the extent.** The layer families take *Cells X* and *Cells Y* (the Kitten also uses *Cells Z* for stacked layers); more cells make a larger framed patch.
4. **Dial the mode-specific controls** that appear only for the family you chose: the **Escher** *Profile* (sine → osteomorphic saddle, tent → Versatile-style, plus S-curve and step), its *Deform Depth* (how deep the interlock bites), *Edge Samples* and *Block Height*; the **SL** *Mode* (single block, engaged pair, or the periodic square strand) with its *Engagement* and *Frame Order*; the **Dome** *Seed* (icosahedron / dodecahedron), *Edge Deform* and *Shell Thickness*; and the **Hendecahedron** *Cells* count.
5. **Finish the look.** *Gap Factor* shrinks each block about its centroid (1.0 = blocks touch); *Scale* fits a 2 m cube; *Colouring* is *By Block Type* (Truchet two-tone), *Highlight Frame* (paints the fixed peripheral ring apart, so you can see what must be held), or *None*; and *Separate Objects* outputs every block as its own mesh.
6. **Read the report.** Each build prints the family with a rigour tag — *interlocking (framed interior)* for the four locking layers, *modular blocks* otherwise — followed by the vertex and face counts.

## Options


<!-- options: generated by docs/scaffold_pages.py -->

| Option | Default | Description |
| --- | --- | --- |
| Family | Interlocking Tetrahedra | Interlocking Tetrahedra, Escher / Osteomorphic, Versatile Blocks, Interlocking Cubes, Interlocking Octahedra, Bisquare Blocks, Rhom Block (single), Rhom Block Obverse (single), and 6 more. |
| Cells X | 4 | Range 1-16. |
| Cells Y | 4 | Range 1-16. |
| Cells Z | 2 | Layers (tetroctahedrille only) Range 1-8. |
| Profile | Sine (Osteomorphic) | Sine (Osteomorphic), Tent (Versatile), S-Curve, Step. |
| Deform Depth | 0.18 | Escher edge-deformation depth (interlock depth) Range 0-0.45. |
| Edge Samples | 8 | Range 2-24. |
| Block Height | 1 | Range 0.2-4. |
| SL Mode | Square Strand | Single Block, Engaged Pair, Square Strand. |
| Engagement | a  (Rz-90, T 1 -1 0) | Engagement used for the Engaged Pair mode. a  (Rz-90, T 1 -1 0), h  (Rx180, T 2 0 0), s  (Rx180 Rz-90, T 1 1 -1), t  (Rx180 Rz90, T 1 -1 1), d  (T 2 0 -1), y  (Rz90, T 1 1 -2). |
| Frame Order | 0 | Square strand (a h^2n)^4: n=0 is the tight a4 loop, n>=1 nests larger square frames Range 0-4. |
| Seed | Icosahedron | Icosahedron, Dodecahedron. |
| Edge Deform | 0.18 | Tangential push of each shared edge at the middle shell (the interlock depth) Range 0-0.5. |
| Shell Thickness | 0.15 | Inner/outer shell offset about the sphere Range 0.03-0.5. |
| Cells | 32 | Number of hendecahedra in the interlocking cluster Range 1-200. |
| Gap Factor | 0.94 | Scale of each block about its centroid (1.0 = blocks touch) Range 0.3-1. |
| Scale | 1 | Overall scale (1.0 fits a 2 m cube) Range 0.01-100. |
| Colouring | By Block Type | By Block Type, Highlight Frame, None. |
| Separate Objects | Off | Output each element of the pattern as its own mesh object |

<!-- /options -->

## Notes

- **Frame.** A finite interlocking patch only locks if its peripheral
  ring is held fixed; that ring is what the *Highlight Frame*
  colouring marks. Without a frame any finite assembly can be taken
  apart from its boundary.
- **Assemblies vs. single blocks.** The Tetrahedra layer, the Escher
  loft, the Versatile layer, the Kitten and the SL strand are verified
  non-overlapping space-filling / interlocking assemblies; the Escher
  tiling is checked to cover the plane exactly once at every height,
  and the SL strand is checked to be cube-disjoint. The UFO and
  Cushion families are shown as single reference blocks, because their
  space-filling assemblies require rotation-based grammars (see the
  project backlog).

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/interlocking__TETRA.png" width="200"><br><sub>Interlocking Tetrahedra</sub></td>
<td align="center"><img src="../images/variants/interlocking__ESCHER.png" width="200"><br><sub>Escher / Osteomorphic</sub></td>
<td align="center"><img src="../images/variants/interlocking__VERSATILE.png" width="200"><br><sub>Versatile Blocks</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/interlocking__MCSCUBE.png" width="200"><br><sub>Interlocking Cubes</sub></td>
<td align="center"><img src="../images/variants/interlocking__MCSOCTA.png" width="200"><br><sub>Interlocking Octahedra</sub></td>
<td align="center"><img src="../images/variants/interlocking__BISQUARE.png" width="200"><br><sub>Bisquare Blocks</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/interlocking__RHOM.png" width="200"><br><sub>Rhom Block (single)</sub></td>
<td align="center"><img src="../images/variants/interlocking__RHOM_OBV.png" width="200"><br><sub>Rhom Block Obverse (single)</sub></td>
<td align="center"><img src="../images/variants/interlocking__KITTEN.png" width="200"><br><sub>Tetroctahedrille Kitten</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/interlocking__UFO.png" width="200"><br><sub>Tetroctahedrille UFO (single)</sub></td>
<td align="center"><img src="../images/variants/interlocking__CUSHION.png" width="200"><br><sub>Tetroctahedrille Cushion (single)</sub></td>
<td align="center"><img src="../images/variants/interlocking__SL.png" width="200"><br><sub>SL Blocks</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/interlocking__DOME.png" width="200"><br><sub>Interlocking Dome</sub></td>
<td align="center"><img src="../images/variants/interlocking__HENDECA.png" width="200"><br><sub>Bisymmetric Hendecahedron</sub></td>
</tr>
</table>

## How it works

**In plain terms.** *Topological interlocking* is the trick of making solid blocks hold one another in place by shape alone — no glue, no screws, no notch that pins them. Think of a flat ring of stones in an arch: hold the outer ring fixed and no inner stone can fall out, because every direction it might move is blocked by a neighbour. These blocks do the same in a slab. The secret is that each block is shaped so that a horizontal slice through it, taken at the right height, is a tile that covers the floor — so at that height the block is completely ringed by its neighbours. A block cannot rise, because its slanted top is trapped under a neighbour's overhang; it cannot sink, because a *different* neighbour traps it from the other side. Fix the outer frame and the whole interior is locked. Everything below makes "shaped so a slice tiles" precise for each family.

**Why a frame.** A finite patch only locks if its peripheral ring is held: without the frame you could always slide a boundary block off sideways and unzip the assembly from the edge. So the rigorous families flag their outer ring as a *frame* (the **Highlight Frame** colouring paints it), and the locking claim is always about the *framed interior*. The module reserves the report tag *interlocking (framed interior)* for the families it proves lock this way, and calls the rest *modular blocks*.

**Interlocking tetrahedra.** Take a regular tetrahedron and set it so that its two opposite edges are both horizontal and mutually perpendicular — the top edge (at height $+H$) along $x$, the bottom edge (at $-H$) along $y$. For a regular tetrahedron whose opposite edges have length $2$ the slant edges force

$$\sqrt{2 + 4H^2} = 2 \quad\Longrightarrow\quad H = \tfrac{1}{\sqrt2},$$

and — the key fact — its horizontal mid-section at $z=0$ is exactly a **unit square**. Put type A on the white squares of a checkerboard and type B (A turned $90°$) on the black, and those unit-square mid-sections tile the plane, so every cell is surrounded. Now the two directions do different jobs: a cell's $x$-neighbours sit under its rising top edge and block it from lifting, while its $y$-neighbours block it from sinking — the asymmetry of the tetrahedron is precisely what makes "up" and "down" both dead ends.

**Moving cross-section: cubes and octahedra.** A second route (Kanel-Belov / Dyskin) starts from the common horizontal section — a regular hexagon — and erects a tilted plane over each of its six edges, alternating the tilt sense $+,-,+,-,+,-$ around the ring. Intersecting those six half-spaces closes the solid, and the tilt angle chooses which Platonic solid you get:

$$\beta = \arcsin\tfrac{1}{\sqrt3} \approx 35.26° \ \text{(cube)}, \qquad \beta = \arcsin\tfrac13 \approx 19.47° \ \text{(octahedron, with horizontal caps)} .$$

The hexagon is the section normal to the solid's 3-fold (body-diagonal) axis, held vertical. All cells are *identical* pure translates on the triangular lattice $t_1=(\tfrac32,\tfrac{\sqrt3}{2}),\ t_2=(0,\sqrt3)$; the interlock needs no two-colour scheme because a neighbour sits across a hexagon *edge*, mapping that edge to the neighbour's *opposite* edge — which carries the opposite tilt sense — so the two cells automatically share each inclined face-plane. That shared slanted wall is exactly the overhang that traps its neighbour.

**The Escher trick: osteomorphic and Versatile lofts.** Here the tile does the work directly. Take the unit square $[0,1]^2$, a $p4$ fundamental domain, and deform its four edges with a single profile — because the four edges are $90°$ images of one another, one profile fixes all four (this is the *Escher trick*). Place the deformed square $M$ at $z=0$, its $90°$ rotation $M'$ at $z=\tfrac12$, and $M$ again at $z=1$, and loft between them. Every horizontal section is then a copy of a deformed-square tiling, so copies on the integer lattice fill each slab — and the half-height $90°$ twist is what makes the loft interlock rather than merely stack (the generator checks this by confirming each section tiles with coverage exactly one). Checkerboard parity flips the twist sense and yields the two Truchet colours. A **sine** profile is Estrin's *osteomorphic* saddle block; a **tent** profile a Versatile-style block. The exact literature blocks are built the same way but from published vertex tables: the **Versatile** block interpolates a square into a $1\times2$ rectangle and tiles by translation over the diamond lattice, and the **Bisquare** block (Frézier, 1737) rises from a square base to a two-tent roof, a $p4$ layer of volume $\tfrac32$ per cell — the missing quarter being the valleys between the roofs, so the layer has a flat floor and a tented top rather than filling solid.

**SL octocubes.** The self-interlocking blocks are built from cubes. Fuse an S-shaped tetracube to an L-shaped tetracube, sharing three faces, and you get one contiguous **octocube** spanning three $z$-levels. Blocks engage by one of six rigid motions $a,h,s,t,d,y$, each a world-frame affine map $p\mapsto Rp+t$ (for instance $a$ is $R_z(-90°)$ then $T(1,-1,0)$). A *strand* is a word in these letters; it is a periodic, self-locking loop exactly when the product of its transforms is the identity, so the block returns to where it started. The simplest such loop is $a^4$ — the tight square strand — and the general square frames are $(a\,h^{2n})^4$. Because the pieces are integer polycubes, "no overlap" is an *exact* question about shared unit cubes, and the strand is verified globally cube-disjoint.

**Verification.** None of these locking claims are taken on faith. The module's self-test rebuilds every family and gates it: the convex-primitive layers (tetrahedra, cubes, octahedra, kitten) pass an interior-penetration test, the SL polycubes an exact integer-disjointness test, the Bisquare layer a ray-cast coverage test, and the space-fillers (Escher sections, the hendecahedron boat-and-lattice) a coverage-exactly-one test — printing `RESULT: OK` only when no assembly interpenetrates and every tiling covers space once.

## References

- A. V. Dyskin, Y. Estrin, A. J. Kanel-Belov, E. Pasternak, "A new concept in design of materials and structures: assemblies of interlocked tetrahedron-shaped elements", Scripta Materialia 44 (2001) 2689-2694.
- A. J. Kanel-Belov, A. V. Dyskin, Y. Estrin, E. Pasternak, I. A. Ivanov-Pogodaev, "Interlocking of convex polyhedra: towards a geometric theory of fragmented solids", Moscow Math. J. 10(2) (2010) 337-342 (arXiv:0812.5089).
- Y. Estrin, A. V. Dyskin, E. Pasternak et al., "Fracture resistant structures based on topological interlocking with non-planar contacts", Adv. Eng. Materials 5(3) (2003) 116-119 (osteomorphic blocks).
- R. Akpanya, T. Goertzen, S. Wiesenhuetter, A. C. Niemeyer, J. Noennig, "Topological Interlocking, Truchet Tiles and Self-Assemblies", Bridges 2023, 61-68 (Versatile block).
- R. Akpanya, T. Goertzen, A. C. Niemeyer, "From Tilings of Orientable Surfaces to Topological Interlocking Assemblies", Applied Sciences 14(16):7276 (2024) (the Escher-loft framework and the interlocking dome).
- R. Akpanya, T. Goertzen, A. C. Niemeyer, "Topologically Interlocking Blocks inside the Tetroctahedrille", arXiv:2405.01944 (2024) (kitten / UFO / cushion).
- M. Weiss, A. C. Niemeyer, "Construction Methods for Space-Filling Heterogeneous Topological Interlocking Assemblies", arXiv:2604.22475 (2026) (Bisquare and Rhom blocks; A.-F. Frezier, 1737, for the original Bisquare block).
- Shen-Guan Shih, "The Art and Mathematics of Self-Interlocking SL Blocks", Bridges 2018, 107-114.
- G. Inchbald, "Five Space-filling Polyhedra", The Mathematical Gazette 80 (489) (1996) 466-475; J. Wu & G. Inchbald, "Folding the Space-Filling Bisymmetric Hendecahedron for a Large-Scale Art Installation", Bridges 2018, 483-486 (hendecahedron).

