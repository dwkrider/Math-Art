# Rotegrity

![Rotegrity](../images/rotegrity.png)

## Overview

Rotegrity (also **nexorade**) sphere models, after Antiprism's `rotegrity`: every edge of a spherical polyhedron becomes a curved strap lying on the sphere, rotated about its midpoint and lengthened so that neighbouring strap ends overlap in a woven, tensegrity-like arrangement. Twist and extension are free sliders — tune them until the strap ends meet cleanly. Straps are coloured by length class, as in physical rotegrity kits, since a geodesic breakdown produces only a few distinct edge lengths.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Seed | Icosahedron | Seed spherical polyhedron: Icosahedron, Octahedron, Tetrahedron, Cube, Dodecahedron. |
| Geodesic Frequency | 2 | Geodesic subdivision of triangular seeds (Tetrahedron, Octahedron, Icosahedron; hidden for Cube / Dodecahedron; min 1, max 8). |
| Twist | 18.0 | Rotation of each strap about its midpoint, in degrees (min -60, max 60). |
| Extension | 0.35 | Lengthening of each strap beyond its edge, as a fraction (min 0.0, max 1.0). |
| Strap Width | 0.06 | Width of each strap (min 0.005, max 0.4). |
| Strap Thickness | 0.025 | Radial thickness of each strap (min 0.002, max 0.2). |
| Segments | 12 | Arc segments per strap (min 4, max 64). |
| Coloring | By Strap Length | By Strap Length (one material per length class, as in kits), Per Strap (cycle the palette), or None. |
| Radius | 1.0 | Sphere radius; the result is fit within a 2×Radius cube at the origin (min 0.01, max 100.0). |





## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/rotegrity__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
<td align="center"><img src="../images/variants/rotegrity__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
<td align="center"><img src="../images/variants/rotegrity__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/rotegrity__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/rotegrity__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
</tr>
</table>

## How it works

The seed polyhedron's vertices are projected to the unit sphere, and triangular seeds are optionally refined by a **Class-I geodesic** subdivision of frequency $\nu$: each triangle $ABC$ is split into $\nu^2$ triangles at barycentric points $\tfrac{iA+jB+kC}{\nu}$ re-projected to the sphere. The set of undirected edges of the resulting spherical mesh is collected; **one strap is built per edge.**

For an edge with unit endpoints $A,B$, the midpoint direction is $\hat m = \widehat{A+B}$. Both endpoints are rotated about $\hat m$ by the **Twist** angle $\theta$ using Rodrigues' formula,

$$A' = A\cos\theta + (\hat m\times A)\sin\theta + \hat m\,(\hat m\cdot A)(1-\cos\theta),$$

and likewise for $B'$. This turn about the strap centre is what makes each strap ride over one neighbour and under the next, the defining move of a nexorade. The strap's "pole" direction (across its width) is $\hat p = \widehat{A'\times B'}$.

The strap centreline is a great-circle arc through $A'$ and $B'$, sampled by spherical linear interpolation and **extended past both endpoints** so the ends reach their neighbours:

$$t = -\text{extension} + (1 + 2\,\text{extension})\,\frac{i}{n},\qquad P_i = \operatorname{slerp}(A',B',t),\quad i=0,\dots,n.$$

At each sample the point is offset by $\pm\tfrac{\text{width}}{2}$ along $\hat p$ and scaled to the inner and outer radii $1\mp\tfrac{\text{thickness}}{2}$, giving a four-sided ring; consecutive rings are bridged into a closed solid strap with flat end caps. Because **Twist** and **Extension** are independent, they are tuned together until the overlapping ends interlock (there is no automatic solve).

Each strap's arc length $\arccos(A\cdot B)$ is recorded, and straps are grouped into **length classes**; faces carry `strap_index` and `length_class` attributes. **By Strap Length** colouring assigns one material per class (matching how physical kits ship a few strap lengths in distinct colours), while **Per Strap** simply cycles the palette.

## References

- Adrian Rossiter, *Antiprism* and its `rotegrity` program — <https://www.antiprism.com>, <https://github.com/antiprism/antiprism> (GPL; the reference implementation this follows).
- Dick Fischbeck, *Rotegrity* — the term and physical strap-sphere models.
- O. Baverel et al., nexorade / reciprocal-frame structures (the "nexorade" terminology).
