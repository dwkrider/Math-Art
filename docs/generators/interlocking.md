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
- **Versatile Block** — the exact block of Akpanya, Goertzen,
  Wiesenhütter, Niemeyer & Nönnig (Bridges 2023): the interpolation
  between a square and a 1×2 rectangle, two-coloured by the Truchet
  rule.
- **Tetroctahedrille Kitten / UFO / Cushion** — non-convex blocks
  built by gluing regular tetrahedra and octahedra on the
  octet-truss (tetrahedral-octahedral) lattice, after Akpanya,
  Goertzen & Niemeyer (2024). They tile space by translation.
- **SL Blocks** — self-interlocking octocubes (Shih 2018): an
  S-tetracube fused to an L-tetracube, joined by the *a*-engagement
  into a closed strand (four blocks close the loop).
- **Interlocking Dome** — one block per face of an icosahedron or
  dodecahedron, each lofted radially from an inner shell through a
  rotated middle shell to an outer shell; the blocks tile a spherical
  shell and interlock (Akpanya, Goertzen & Niemeyer 2024).

The result is centred at the origin and fit within a cube of the
chosen **Size**.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Family | Interlocking Tetrahedra | Which construction to build (see above). |
| Cells X / Y | 4 | Grid extent for the planar families. |
| Cells Z | 2 | Layers, tetroctahedrille families only. |
| Profile | Sine (Osteomorphic) | Escher edge profile: Sine, Tent, S-Curve, Step. |
| Deform Depth | 0.18 | Escher edge-deformation depth — the interlock depth (min 0, max 0.45). |
| Edge Samples | 8 | Samples along each deformed edge (min 2, max 24). |
| Block Height | 1.0 | Height of the lofted Escher block (min 0.2, max 4.0). |
| Strand Blocks | 4 | SL blocks in the strand; 4 closes the loop (min 1, max 16). |
| Dome Seed | Icosahedron | Seed polyhedron for the dome: Icosahedron (20 blocks) or Dodecahedron (12). |
| Dome Bulge | 0.25 | Outward dilation of the middle ring (min 0, max 0.8). |
| Dome Twist | 0.5 | In-plane rotation of the middle ring — what interlocks the dome (min 0, max 1.2). |
| Shell Thickness | 0.12 | Inner/outer shell offset about the unit sphere (min 0.02, max 0.4). |
| Gap Factor | 0.94 | Scale of each block about its own centroid; 1.0 = blocks touch (min 0.3, max 1.0). |
| Size | 2.0 | The whole assembly is fit within this cube at the origin (min 0.1, max 100). |
| Colouring | By Block Type | By Block Type (two-tone Truchet / block role), Highlight Frame (colour the fixed peripheral frame apart), or None. |

## Notes

- **Frame.** A finite interlocking patch only locks if its peripheral
  ring is held fixed; that ring is what the *Highlight Frame*
  colouring marks. Without a frame any finite assembly can be taken
  apart from its boundary.
- **Rigorous vs. modular.** The Tetrahedra, Escher/Osteomorphic and
  Versatile families are space-filling/interlocking constructions with
  an established frame criterion. The tetroctahedrille, SL and dome
  families are modular block assemblies (translation-tiling, strand,
  and spherical-shell respectively).

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
