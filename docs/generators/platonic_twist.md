# Platonic Twist

![Platonic Twist](../images/platonic_twist.png)

## Overview
After George W. Hart's *Platonic Twist* construction: the faces of a Platonic solid are shrunk and pushed radially outward, then every original edge is spanned by a ribbon that makes a half turn (or several) on its way between the two neighbouring face plates. The result is a single, exactly welded surface -- an airy, paper-sculpture-like form -- that can be given a printable wall via the Thickness (Solidify) option. Odd numbers of half twists produce non-orientable, Mobius-like ribbons, handled by a non-manifold solidify mode.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Solid | Cube | Seed Platonic solid: Tetrahedron, Cube, Octahedron, Dodecahedron, or Icosahedron. |
| Face Shrink | 0.55 | Shrink of each face plate about its (pushed) centre. Range 0.1-0.95. |
| Face Push | 0.35 | Separation of the faces from the solid (radial outward displacement). Range 0.0-2.0. |
| Half Twists | 1 | Half turns of each connecting ribbon. Range 0-4. |
| Ribbon Bulge | 0.35 | Outward billow of the ribbon centreline (Hermite tangent magnitude / smoothstep bulge). Range 0.0-1.5. |
| Smooth Joins (C1) | On | Ribbons leave and arrive within the plate planes (tangent-continuous, no crease); off reverts to creased joins marked with sharp edges. |
| Ribbon Rows | 16 | Lengthwise subdivisions of each ribbon. Range 4-64. |
| Ribbon Columns | 6 | Crosswise subdivisions of each edge / ribbon. Range 2-24. |
| Thickness | 0.04 | Solidify modifier thickness (0 = pure surface). Range 0.0-0.5. |
| Scale | 1.0 | Output is centred and fitted so its largest extent is $2\,\text{m}\times$ Scale. Range 0.01-100. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/platonic_twist__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/platonic_twist__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/platonic_twist__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/platonic_twist__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/platonic_twist__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
</tr>
</table>

## How it works

**Seed and plates.** Each Platonic solid is generated from exact coordinates (the icosahedron from the golden-ratio $\varphi=(1+\sqrt5)/2$ cyclic triples; the dodecahedron as the dual of the icosahedron, its vertices the icosa face centroids ordered by angle). For a face with vertex centroid $\mathbf c$ and unit normal $\mathbf n=\mathbf c/\lVert\mathbf c\rVert$, the plate centre is pushed out radially,
$$\mathbf c' = \mathbf n\,(\lVert\mathbf c\rVert + \text{push}),$$
and each corner is contracted toward it,
$$\mathbf v' = \mathbf c' + \text{shrink}\,(\mathbf v - \mathbf c).$$
Each plate edge is subdivided into *Ribbon Columns* segments, so the plate rim vertices are shared exactly by the ribbons (the surface is welded, not merely coincident).

**The antiprism-like twist.** A shrunken-and-pushed Platonic solid, with its faces reconnected, is topologically the same move that turns a prism into an *antiprism*: adjacent faces are offset in rotation and bridged by a band that carries a relative twist. Here the band spans one original edge between face plates with unit normals $\mathbf n_1,\mathbf n_2$. Two rotations are composed along the ribbon parameter $t\in[0,1]$, both driven by the smoothstep $\sigma(t)=t^2(3-2t)$ (whose zero end-derivatives keep the joins $C^1$):

1. a *bend* that carries $\mathbf n_1$ into $\mathbf n_2$ -- the geodesic (slerp) rotation about their common perpendicular $\mathbf a$ through the full angle $\Phi=\arccos(\mathbf n_1\!\cdot\!\mathbf n_2)$, applied as $\Phi\,\sigma(t)$;
2. an extra *twist* about the ribbon spine of $k\pi\,\sigma(t)$, where $k$ is *Half Twists*.

A profile point offset $\mathbf q_0$ from the start midpoint is transported as
$$\mathbf q(t) = R_{\text{spine}}\!\big(k\pi\sigma\big)\,R_{\mathbf a}\!\big(\Phi\sigma\big)\,\mathbf q_0 .$$

**Centreline.** With *Smooth Joins* on, the ribbon centreline is a cubic Hermite curve between the two edge midpoints $\mathbf m_1,\mathbf m_2$ whose end tangents lie **in** the plate planes (directions $\mathbf o_1,\mathbf o_2$ pointing outward from each plate centre), with magnitude $\tau = L\,(0.3 + 0.8\,\text{bulge})$, $L=\lVert\mathbf m_2-\mathbf m_1\rVert$:
$$\mathbf c(t) = h_{00}\mathbf m_1 + h_{10}\tau\,\mathbf o_1 + h_{01}\mathbf m_2 - h_{11}\tau\,\mathbf o_2 ,$$
using the standard Hermite basis $h_{00},h_{10},h_{01},h_{11}$. This makes each ribbon meet its plates tangentially (no crease). With smooth joins off, the centreline is a straight smoothstep interpolation bulged outward by $\text{bulge}\cdot4\sigma(1-\sigma)$, and the plate/ribbon boundary edges are marked sharp.

**Closure and orientation.** The final ribbon row is forced to coincide with the neighbour's edge vertices; with an odd number of half twists the ribbon arrives reversed, so the shared vertex list is flipped, which is exactly what makes odd-twist ribbons non-orientable. A `NON_MANIFOLD` Solidify handles thickening those Mobius-like sheets.

## References

- G. W. Hart, *Platonic Twist* and related constructions, https://www.georgehart.com/ (see the vibe-coded program examples, https://www.georgehart.com/vibecode/).
- Platonic-twist / fractal-polyhedra / twisted-torus generators after the programs on Hart's vibecode page.
- Rodrigues rotation and cubic Hermite interpolation: standard computer-graphics geometry (e.g. G. Farin, *Curves and Surfaces for CAGD*, Academic Press).
