# Polyhedron Vertex Vortices

![Polyhedron Vertex Vortices](../images/vertex_vortices.png)

## Overview
Each face of a polyhedron is divided by spokes from its centre to each vertex, the spokes are bent within the face plane with equal chirality (a pinwheel), the original edges are deleted, and a minimal surface spans each group of four bent spokes -- inspired by Robert Fathauer's "Vertices Vortices" sculpture. Every original edge yields exactly one saddle patch, and every spoke borders exactly two patches, so the welded result is a closed surface with swirling vortices meeting at the polyhedron's vertices.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Seed | Dodecahedron | Base polyhedron: Tetrahedron, Cube, Octahedron, Dodecahedron (the Vertices Vortices), Icosahedron, Cuboctahedron, Truncated Octahedron, Snub Cube, Icosidodecahedron, Truncated Icosahedron (soccer ball), Rhombic Dodecahedron, or Rhombic Triacontahedron. |
| Bend | 0.2 | Sideways bend of each spoke, as a fraction of its length (about 0.11 in the original sculpture). Range 0-0.6. |
| Spoke Samples | 12 | Subdivisions of each bent spoke. Range 3-48. |
| Solver Iterations | 30 | Plateau area-minimization iterations per patch. Range 0-200. |
| Reverse Swirl | Off | Mirror the vortex chirality. |
| Smooth Shading | Off | Shade the mesh smooth. |
| Thickness | 0.0 | Solidify modifier thickness (0 = raw surface). Range 0-1. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/vertex_vortices__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/vertex_vortices__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__CO.png" width="200"><br><sub>Cuboctahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/vertex_vortices__TO.png" width="200"><br><sub>Truncated Octahedron</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__SC.png" width="200"><br><sub>Snub Cube</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__ID.png" width="200"><br><sub>Icosidodecahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/vertex_vortices__TI.png" width="200"><br><sub>Truncated Icosahedron</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__RD.png" width="200"><br><sub>Rhombic Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/vertex_vortices__RT.png" width="200"><br><sub>Rhombic Triacontahedron</sub></td>
</tr>
</table>

## How it works

**Bent spokes.** For each face, its outward Newell normal $\hat n$ and centroid $C$ are computed. Each vertex $v$ of the face defines a spoke $C\!\to\!v$ with direction $\mathbf d = V_v - C$ of length $L$. The spoke is bent sideways in the face plane, along the perpendicular $\hat p = \hat n \times \hat{\mathbf d}\cdot\text{sgn}$ (sgn flips with `Reverse Swirl`), by a $4t(1-t)$ profile:
$$P(t) = C + \mathbf d\,t + \hat p\,\big(\text{bend}\cdot L\cdot 4t(1-t)\big),\quad t=\tfrac{j}{m}.$$
All spokes on a face turn the same way -- a pinwheel of common chirality.

**One patch per edge.** For each undirected edge $\{a,b\}$ shared by faces $f_a$ (which runs $a\!\to\!b$) and $f_b$ (which runs $b\!\to\!a$), the four bent spokes $C_a\!\to\!a$, $C_a\!\to\!b$, $C_b\!\to\!a$, $C_b\!\to\!b$ form a curved quadrilateral frame straddling the deleted edge. A **Coons patch** interpolates the four boundary curves into an initial membrane grid:
$$P(u,v) = (1-v)\,\text{bot}(u) + v\,\text{top}(u) + (1-u)\,\text{lef}(v) + u\,\text{rig}(v) - \big[\text{bilinear blend of the four corners}\big].$$
The four boundary rows/columns are pinned and the interior is relaxed toward minimal area by the toolkit's `minimize_area` (Pinkall-Polthier cotangent-Laplacian, conjugate-gradient solve), with a uniform-Laplacian fallback. The seed must be a closed manifold mesh with consistent winding (checked via the directed-edge set). After all patches are built, the whole surface is oriented outward by checking its signed volume, then centered, fit within a $2\times\text{scale}$ cube, tagged with an `edge_index` attribute, and optionally given a Solidify shell.

## References

- R. Fathauer, "Vertices Vortices" sculpture; Robert Fathauer, https://robertfathauer.com/
- S. A. Coons, "Surfaces for Computer-Aided Design of Space Forms," MIT Project MAC report MAC-TR-41, 1967 (the Coons patch).
- U. Pinkall and K. Polthier, "Computing Discrete Minimal Surfaces and Their Conjugates," *Experimental Mathematics* 2(1), 1993, pp. 15-36 (the area-minimization solver).
- K. Brakke, *Surface Evolver*, https://kenbrakke.com/evolver (Plateau-problem area minimization).
