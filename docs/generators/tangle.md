# Polyhedral Tangle

![Polyhedral Tangle](../images/tangle.png)

## Overview

Interwoven compounds of polyhedron frames — "tangles" in the spirit of Alan Holden's *Orderly Tangles*: several rotated copies of a seed polyhedron interlocked into a symmetric compound. Supported compounds are 2 tetrahedra (the stella octangula), 5 and 10 tetrahedra, 3 cubes (Escher's compound), 5 cubes, and 5 octahedra. Each component is rendered either as hollow face panels (the Leonardo da Vinci open-faced style, mitred exactly along shared edges) or as square edge struts with faceted knuckles at the vertices (the Lang polypolyhedra style).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Compound | 5 Tetrahedra | Which compound to build: 5 Tetrahedra (the classic Tetra Tangle), 2 Tetrahedra (stella octangula), 10 Tetrahedra (both chiralities), 3 Cubes (Escher's compound), 5 Cubes (in a dodecahedron), 5 Octahedra (icosahedral). |
| Style | Hollow Faces (da Vinci) | Hollow Faces — solid face panels with openings, mitred along shared edges and vertices; or Edge Struts — square sticks along the edges with flat caps and faceted vertex knuckles. |
| Frame Width | 0.22 | Hollow-face ring width, as a fraction of the face (min 0.02, max 0.9). |
| Frame Thickness | 0.10 | Panel / strut thickness (min 0.01, max 1.0). |
| Cap Size | 1.0 | How far the edge struts stop short of each vertex, in strut thicknesses; the faceted knuckle fills the joint (min 0.3, max 3.0). |
| Component Rotation | 0.0 | Rotate every component about its own symmetry axis, exploring Lang-style polypolyhedra variants (degrees, min -180, max 180). |
| Component Size | 1.0 | Scale of each component about the centre (min 0.5, max 2.0). |
| Spin | 0.0 | Extra rotation of the whole compound about its main symmetry axis (degrees, min -180, max 180). |
| Coloring | Per Component | Per Component (one material per polyhedron) or None. |
| Scale | 1.0 | Uniform output scale; the result is fit within a 2×Scale cube at the origin (min 0.01, max 100.0). |





## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/tangle__T5.png" width="200"><br><sub>5 Tetrahedra</sub></td>
<td align="center"><img src="../images/variants/tangle__T2.png" width="200"><br><sub>2 Tetrahedra</sub></td>
<td align="center"><img src="../images/variants/tangle__T10.png" width="200"><br><sub>10 Tetrahedra</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/tangle__C3.png" width="200"><br><sub>3 Cubes</sub></td>
<td align="center"><img src="../images/variants/tangle__C5.png" width="200"><br><sub>5 Cubes</sub></td>
<td align="center"><img src="../images/variants/tangle__O5.png" width="200"><br><sub>5 Octahedra</sub></td>
</tr>
</table>

## How it works

Each compound is generated from a seed polyhedron by **exact symmetry rotations**. Every rotation uses the axis–angle (Rodrigues) matrix

$$R(\hat a,\theta) = I\cos\theta + [\hat a]_\times\sin\theta + \hat a\hat a^{\!\top}(1-\cos\theta).$$

- **2 tetrahedra** — a tetrahedron and its mirror image (stella octangula).
- **5 / 10 tetrahedra** — five copies of the seed tetrahedron rotated by $\tfrac{2\pi k}{5},\;k=0,\dots,4$ about a 5-fold axis of the icosahedral group, $\mathbf a = (0,1,\varphi)$ with $\varphi = \tfrac{1+\sqrt5}{2}$. The 10-compound adds the five mirror copies.
- **5 cubes / 5 octahedra** — five copies rotated by $\tfrac{2\pi k}{5}$ about the same 5-fold axis.
- **3 cubes** — one axis-aligned cube plus two more turned by $45°$ about the $x$- and $y$-axes (Escher's compound).

Each component also carries its own symmetry axis, so **Component Rotation** turns every copy about that axis to explore Lang-style polypolyhedra variants, and **Spin** rotates the assembled compound about its main axis.

Each component is then drawn in one of two styles.

**Hollow Faces (da Vinci).** Two scaled copies of the polyhedron are made, one at radius $1+\tfrac{\text{thickness}}{2}$ and one at $1-\tfrac{\text{thickness}}{2}$; adjacent panels share these scaled vertices, so the joints along edges and at vertices are exact watertight mitres. Within each face a hole is inset by the factor $(1-\text{width})$ toward the face centroid, and the panel is the shell between the outer surface, the inner surface and the hole rim — the open "vacuus" panels Leonardo drew for Pacioli's *De divina proportione*.

**Edge Struts.** Each edge becomes a square-section stick trimmed back from both endpoints by $s = \min(\text{thickness}\cdot\text{cap\_size},\,0.35\,\ell)$ (with $\ell$ the edge length) and closed with flat end caps. At every vertex the surrounding strut-cap corners are collected and filled with their **convex hull**, giving a clean faceted knuckle at the joint.

Every emitted face is tagged with its component index (a `component_index` face attribute), which also drives **Per Component** colouring (metallic materials from a fixed palette). A built-in self-test confirms each compound is a closed 2-manifold with the expected component count and positive volume.

## References

- Alan Holden, *Orderly Tangles: Cloverleafs, Gordian Knots, and Regular Polylinks*, Columbia University Press, 1983.
- Robert J. Lang, *Polypolyhedra* — designs of interwoven edge-strut polyhedral compounds (the Edge Struts style).
- Leonardo da Vinci, open-faced ("vacuus") polyhedron illustrations for Luca Pacioli, *De divina proportione*, 1509; see also George W. Hart, *Leonardo da Vinci's Polyhedra* — <https://www.georgehart.com/virtual-polyhedra/leonardo.html> (the Hollow Faces style).
- M. C. Escher, *Waterfall* / compound of three cubes (the 3 Cubes compound).
