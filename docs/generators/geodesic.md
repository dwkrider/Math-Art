# Geodesic Sphere / Dome

![Geodesic Sphere / Dome](../images/geodesic.png)

## Overview

Geodesic spheres and domes built by subdividing a triangular Platonic seed (icosahedron, octahedron or tetrahedron) and projecting the result to the sphere — Class I $(f,0)$ or Class II $(f,f)$ breakdowns, oriented vertex-up and optionally cut to a hemisphere or 5/8 dome. An optional **Dual (Goldberg)** switch outputs the geodesic's dual: hexagons and twelve pentagons, i.e. a Goldberg polyhedron. Styles include a welded shell (with optional Solidify thickness), a strut-and-node frame, Leonardo open panels, and inset dome panels. After Segerman, *Visualizing Mathematics with 3D Printing* (figs 4-5 / 4-6).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Base | Icosahedron | Seed polyhedron to subdivide: Icosahedron, Octahedron or Tetrahedron |
| Class | Class I (f,0) | Class I: each base triangle split into $f^2$ triangles and projected. Class II: mid-face (kis) split followed by a Class I subdivision at $f$, giving the $(f,f)$ triangulation (effective frequency $2f$) |
| Frequency | 3 | Breakdown frequency $f$: Class I gives the $(f,0)$ subdivision, Class II the $(f,f)$ |
| Dual (Goldberg) | False | Output the dual polyhedron — hexagons and twelve pentagons (a Goldberg polyhedron / the geodesic's honeycomb) instead of triangles |
| Cut | Full Sphere | Full sphere, Hemisphere Dome (centroid $z\ge0$), or 5/8 Dome (centroid $z\ge-0.25R$) |
| Base Ring | False | Thicken the open rim of a dome into a flat ring band |
| Ring Width | 0.1 | Radial width of the base-ring band |
| Style | Shell | Shell (welded surface, Solidify if thickness $>0$), Struts & Nodes (cylinder edges + sphere vertices), Leonardo open panels, or inset Panels |
| Radius | 1.0 | Sphere radius $R$ |
| Thickness | 0.05 | Shell / panel thickness (0 = single surface) |
| Border | 0.3 | Leonardo face-frame width |
| Strut Radius | 0.02 | Struts style: cylinder radius |
| Node Radius | 0.035 | Struts style: node sphere radius |
| Panel Gap | 0.15 | Panels style: fraction each triangle is shrunk about its centroid |


## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/geodesic__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
<td align="center"><img src="../images/variants/geodesic__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
<td align="center"><img src="../images/variants/geodesic__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/geodesic__GOLDBERG.png" width="200"><br><sub>Goldberg Dual</sub></td>
</tr>
</table>

## How it works

**Seed and orientation.** The seed is one of the three triangular-faced Platonic solids, each vertex normalized to the unit sphere. The icosahedron's twelve vertices are the cyclic coordinates $(0,\pm1,\pm\varphi)$, $\varphi=\tfrac{1+\sqrt5}{2}$; its faces are recovered as all vertex triples at minimal mutual distance. The seed is then rotated so its topmost vertex sits exactly on $+Z$ (vertex-up).

**Class I breakdown $(f,0)$.** Each triangle $ABC$ is filled with a barycentric grid of frequency $f$ and every grid point is projected radially to the sphere:

$$p_{ijk}=\frac{i\,A+j\,B+k\,C}{f},\qquad i+j+k=f,\qquad \hat p=\frac{p_{ijk}}{\lVert p_{ijk}\rVert}.$$

Each base triangle becomes $f^2$ small triangles. Vertices shared along an edge are deduped exactly, since the barycentric sums are bitwise identical from both adjacent faces. For the icosahedron this gives $10f^2+2$ vertices and $20f^2$ triangles.

**Class II breakdown $(f,f)$.** First a *kis* (mid-face) operation splits every triangle into three about its projected centroid; a Class I subdivision at frequency $f$ then follows. The composition yields the triacon-style $(f,f)$ triangulation — $3f^2$ triangles per original face, effective frequency $2f$ (for the icosahedron $30f^2+2$ vertices).

**Goldberg dual.** The dual places a vertex at each triangle's spherical centroid and builds one polygon around each original vertex from its incident face-centroids, ordered by angle in the tangent plane. A degree-6 vertex gives a hexagon and the twelve original degree-5 vertices give pentagons — exactly a Goldberg polyhedron $GP(m,n)$, the "honeycomb" dual of the geodesic sphere (and, at frequency 1 on the icosahedron, the truncated icosahedron / soccer ball). Boundary vertices on a cut dome have no closed ring and are skipped.

**Dome cutting and base ring.** A dome keeps only faces whose centroid lies above a cut height: $z\ge0$ for a hemisphere, $z\ge-0.25R$ for the 5/8 dome (a clean strut ring on the 3v icosahedron). The open rim loops are found from the boundary edges (edges belonging to a single face); the optional base ring widens each rim loop outward in $XY$ by the ring width into a flat band.

**Output styles.** *Shell* welds the triangles into one surface, adding a Solidify modifier for thickness. *Struts & Nodes* renders every edge as a cylinder and every vertex as a small sphere. *Leonardo* opens each face into a framed panel via the shared Leonardo modifier. *Panels* insets each triangle about its centroid by the gap and extrudes it into a thin prism along its outward normal.

## References

- Henry Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 (figs 4-5, 4-6, geodesic spheres and domes). ISBN 978-1-4214-2035-6.
- Michael Goldberg, "A class of multi-symmetric polyhedra," *Tôhoku Mathematical Journal* 43 (1937), 104–108 — the Goldberg polyhedra $GP(m,n)$, duals of the geodesic spheres.
- Hugh Kenner, *Geodesic Math and How to Use It*, University of California Press, 1976 — Class I and Class II (alternate and triacon) breakdown frequencies.
- R. Buckminster Fuller, U.S. Patent 2,682,235, "Building Construction" (geodesic dome), 1954.
