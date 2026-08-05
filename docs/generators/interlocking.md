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
- **Versatile Block** (single) — the exact block of Akpanya,
  Goertzen, Wiesenhütter, Niemeyer & Nönnig (Bridges 2023): the
  interpolation between a square and a 1×2 rectangle. Its
  space-filling assembly mixes translations and rotations classified
  by Truchet tiles, so the block is shown on its own.
- **Tetroctahedrille Kitten** — a non-convex block (one octahedron +
  two tetrahedra) glued on the octet-truss (tetrahedral-octahedral)
  lattice, after Akpanya, Goertzen & Niemeyer (2024). It **tiles
  space by pure translation** over the FCC basis and is assembled into
  a space-filling patch.
- **Tetroctahedrille UFO / Cushion** (single) — one octahedron + four
  tetrahedra (volume 8/3); shown as a single block, since they admit
  no integer translation lattice.
- **SL Blocks** — self-interlocking octocubes (Shih 2018): an
  S-tetracube fused to an L-tetracube (each a contiguous unit cell).
  Two blocks form a conjugate pair (one turned 180° about a y-axis
  through a shared S corner); pairs chain by the *a*-engagement
  (Rz(−90) then T(1,−1,0)) into a periodic strand — four pairs close
  the a⁴ loop. Verified cube-disjoint.
- **Polyhedral Shell Dome** — one block per face of an icosahedron or
  dodecahedron, each the region of the spherical shell between an
  inner and outer radius cut by the radial walls through that face's
  edges. Adjacent blocks share their radial walls, so the shell is
  tiled with no gaps or overlaps (after the radial construction of
  Akpanya, Goertzen & Niemeyer 2024).

The result is centred at the origin and fit within a cube of the
chosen **Size**. TETRA, Escher and Kitten are verified
non-overlapping assemblies; the other families are single reference
blocks or a shell dissection.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Family | Interlocking Tetrahedra | Which construction to build (see above). |
| Cells X / Y | 4 | Grid extent — Tetrahedra, Escher and Kitten only. |
| Cells Z | 2 | Layers — Kitten only. |
| Profile | Sine (Osteomorphic) | Escher edge profile: Sine, Tent, S-Curve, Step. |
| Deform Depth | 0.18 | Escher edge-deformation depth — the interlock depth (min 0, max 0.45). |
| Edge Samples | 8 | Samples along each deformed edge (min 2, max 24). |
| Block Height | 1.0 | Height of the lofted Escher block (min 0.2, max 4.0). |
| Dome Seed | Icosahedron | Seed polyhedron for the dome: Icosahedron (20 blocks) or Dodecahedron (12). |
| Dome Bulge | 0.18 | Outward push of the middle ring for a domed profile (min 0, max 0.8). |
| Shell Thickness | 0.12 | Inner/outer shell offset about the unit sphere (min 0.02, max 0.4). |
| SL Mode | Strand | SL layout: Single Block, Conjugate Pair, or Strand. |
| Strand Pairs | 4 | Conjugate pairs in the SL strand; 4 closes the a⁴ loop (min 1, max 12). |
| Gap Factor | 0.94 | Scale of each block about its own centroid; 1.0 = blocks touch (min 0.3, max 1.0). |
| Size | 2.0 | The whole assembly is fit within this cube at the origin (min 0.1, max 100). |
| Colouring | By Block Type | By Block Type (two-tone Truchet / block role), Highlight Frame (colour the fixed peripheral frame apart), or None. |

## Notes

- **Frame.** A finite interlocking patch only locks if its peripheral
  ring is held fixed; that ring is what the *Highlight Frame*
  colouring marks. Without a frame any finite assembly can be taken
  apart from its boundary.
- **Assemblies vs. single blocks.** The Tetrahedra layer, the Escher
  loft, the Kitten and the SL strand are verified non-overlapping
  space-filling / interlocking assemblies; the Escher tiling is
  checked to cover the plane exactly once at every height, and the SL
  strand is checked to be cube-disjoint. The Polyhedral Shell Dome is
  a gap-free shell dissection. The Versatile, UFO and Cushion families
  are shown as single reference blocks, because their space-filling
  assemblies require rotation-based grammars (see the project
  backlog).

## References

- A. V. Dyskin, Y. Estrin, A. J. Kanel-Belov, E. Pasternak, "A new
  concept in design of materials and structures: assemblies of
  interlocked tetrahedron-shaped elements", *Scripta Materialia* 44
  (2001) 2689–2694.
- A. J. Kanel-Belov, A. V. Dyskin, Y. Estrin, E. Pasternak,
  I. A. Ivanov-Pogodaev, "Interlocking of convex polyhedra: towards a
  geometric theory of fragmented solids", *Moscow Math. J.* 10(2)
  (2010) 337–342 (arXiv:0812.5089).
- Y. Estrin, A. V. Dyskin, E. Pasternak et al., "Fracture resistant
  structures based on topological interlocking with non-planar
  contacts", *Adv. Eng. Materials* 5(3) (2003) 116–119.
- R. Akpanya, T. Goertzen, S. Wiesenhütter, A. C. Niemeyer,
  J. Nönnig, "Topological Interlocking, Truchet Tiles and
  Self-Assemblies", *Bridges 2023*, 61–68.
- R. Akpanya, T. Goertzen, A. C. Niemeyer, "From Tilings of Orientable
  Surfaces to Topological Interlocking Assemblies", *Applied Sciences*
  14(16):7276 (2024).
- R. Akpanya, T. Goertzen, A. C. Niemeyer, "Topologically Interlocking
  Blocks inside the Tetroctahedrille", arXiv:2405.01944 (2024).
- S.-G. Shih, "The Art and Mathematics of Self-Interlocking SL
  Blocks", *Bridges 2018*, 107–114.
