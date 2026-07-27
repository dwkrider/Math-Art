# Space-filling Solids

![Space-filling Solids](../images/spacefill_solids.png)

## Overview

Blocks of the classical space-filling honeycombs, with every cell shrunk by a gap factor about its own centroid so the packing reads as a stack of separate solids — after Henry Segerman, *Visualizing Mathematics with 3D Printing* (figs 4-17 – 4-20). Choices are cubes, the octet truss (regular octahedra plus tetrahedra), truncated octahedra on the BCC lattice, rhombic dodecahedra on the FCC lattice, and Russell Towle's three- and four-armed rhombic spirallohedra. At gap $=1$ adjacent cells share their faces exactly; the default gap $<1$ gives the printed-separately look.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Honeycomb | Octahedra + Tetrahedra | Cell type: Cubes, Octahedra + Tetrahedra (octet truss), Truncated Octahedra (BCC), Rhombic Dodecahedra (FCC), or 3-/4-armed Rhombic Spirallohedra |
| Cells X | 3 | Lattice cells along X |
| Cells Y | 3 | Lattice cells along Y |
| Cells Z | 2 | Lattice cells along Z |
| Gap Factor | 0.92 | Scale of each cell about its own centroid (1.0 = cells share faces exactly) |
| Cell Size | 1.0 | Length of one lattice step per axis |
| Style | Solid | Plain solid cells, or Leonardo open-faced da Vinci panels (Geometry Nodes modifier on the whole block) |
| Border | 0.3 | Leonardo face-frame width (fraction of the face) |
| Thickness | 0.06 | Leonardo panel thickness |
| Two Materials | True | Distinct materials for octahedra vs tetrahedra (Octet), or alternating by lattice parity (Spirallohedra); other honeycombs have no honest 2-colouring |
| Spiral Segments | 12 | Star vectors of the spirallohedron cell, rounded down to a multiple of the arm count; more segments = finer, longer spiral cells |
| Spiral Pitch | 55.0 | Polar-star pitch angle from the axis (degrees); higher is squatter, wider spiral cells |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/spacefill_solids__CUBIC.png" width="200"><br><sub>Cubes</sub></td>
<td align="center"><img src="../images/variants/spacefill_solids__OCTET.png" width="200"><br><sub>Octahedra + Tetrahedra</sub></td>
<td align="center"><img src="../images/variants/spacefill_solids__TRUNCOCT.png" width="200"><br><sub>Truncated Octahedra</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/spacefill_solids__RHOMBDODEC.png" width="200"><br><sub>Rhombic Dodecahedra</sub></td>
<td align="center"><img src="../images/variants/spacefill_solids__SPIRAL3.png" width="200"><br><sub>Spirallohedra (3-arm)</sub></td>
<td align="center"><img src="../images/variants/spacefill_solids__SPIRAL4.png" width="200"><br><sub>Spirallohedra (4-arm)</sub></td>
</tr>
</table>

## How it works

**Space-filling honeycombs.** Each preset is a solid that tiles $\mathbb{R}^3$ by translations (and, for the octet, two orientations). The generator stores each cell in integer *canonical* lattice coordinates, so at gap $=1$ the faces shared between neighbours coincide to the bit; a self-test checks that the numeric mesh volume equals the analytic total cell volume for every honeycomb.

- **Cubic** — unit cubes on the integer lattice $\mathbb{Z}^3$; one cell per lattice point.
- **Octet (alternated cubic honeycomb).** Regular octahedra sit on one FCC sublattice and regular tetrahedra (in two mirror orientations, chosen by cube parity) fill the gaps, in the bulk ratio $1$ octahedron $:2$ tetrahedra. The octahedron has vertices on the axes; the tetrahedra are the even- and odd-parity corners of a unit cube. Per canonical cell the volumes are $\tfrac{4}{3}$ (octahedron) and $\tfrac13$ (tetrahedron).
- **Truncated octahedra (bitruncated cubic honeycomb).** The permutohedron with vertices at all permutations of $(0,\pm1,\pm2)$ — six squares and eight hexagons — is the Voronoi cell of the BCC lattice. Cells sit on a primary grid plus a second grid at the cube centres, spacing $4$.
- **Rhombic dodecahedra.** Twelve rhombi normal to the $\langle1,1,0\rangle$ neighbour directions form the Voronoi cell of the FCC lattice; cells occupy the even-parity integer points of a box, lattice spacing $2$.

**Rhombic spirallohedra (Russell Towle).** These cells are cut from the *polar zonohedron* construction: a polar star of $n$ vectors evenly spaced on a cone of pitch angle $\vartheta$ generates a zonohedron, and taking $w=n/\text{arms}$ layers yields the spiralling bundle of rhombi $S(n,w)$ — the three-armed cell is $S(12,4)$, the four-armed $S(12,3)$. Rather than hardcode a lattice, the generator *derives* the tiling from the cell itself: for each rhombic face it finds the translation carrying it onto the opposite (anti-normal) face of the same cell,

$$t = c_{\text{face}} - c_{\text{opp}},\qquad \mathbf{n}_{\text{face}} + \mathbf{n}_{\text{opp}} = 0,$$

collects those translations, and picks three whose determinant equals the cell volume,

$$\bigl|\det[\,t_1\ t_2\ t_3\,]\bigr| = V_{\text{cell}},$$

as the lattice basis $B$ (verified so that every face translation is an integer combination of $B$). The cell tiles $\mathbb{R}^3$ by pure translations over $B$; a `ValueError` is raised if a face has no translation partner.

**Assembling the block.** Cells are laid out at $i\,B_0+j\,B_1+k\,B_2$ (or the fixed lattice for the classical honeycombs). Every cell's vertices are then scaled about its own centroid by the gap factor,

$$v \ \longmapsto\ c + g\,(v-c),$$

and the whole block is centred at the origin and scaled so one lattice step maps to the chosen Cell Size. The optional Leonardo style replaces each solid face with an open panel frame via a shared Geometry Nodes modifier.

## References

- Henry Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 (figs 4-17 – 4-20, space-filling solids). ISBN 978-1-4214-2035-6.
- Russell Towle, "Spirallohedra" and polar-zonohedron notebooks — rhombic spirallohedra $S(n,w)$ from the polar zonohedron construction. See the archived Mathematica material and the *Wolfram Demonstrations* entry "Zonohedra": <https://demonstrations.wolfram.com/Zonohedra/>
- Adrian Rossiter, *Antiprism* polyhedron-modelling software — `make_polar_zonohedron` / `zono -P` (the spirallohedron preset equals `zono -P 12,4`): <https://www.antiprism.com>
- H. S. M. Coxeter, *Regular Polytopes*, 3rd ed., Dover, 1973 — the alternated cubic (octet) and bitruncated cubic honeycombs and the Voronoi cells of the cubic, BCC and FCC lattices.
