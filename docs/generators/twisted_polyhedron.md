# Twisted Polyhedron

![Twisted Polyhedron](../images/twisted_polyhedron.png)

## Overview

A dual-polyhedron sculpture: an inset polygon sits on every face of a chosen **outer solid**, an inset polygon sits on every face of its **dual inner solid**, and a twisted bezier ribbon swirls from each outer-polygon edge to a neighbouring inner-polygon edge. The outer solid can be any of the five Platonic solids, all thirteen Archimedean solids (woven to their Catalan duals), or an icosahedral geodesic sphere (woven to its Goldberg dual). Everything welds into one closed shell so an optional Solidify + Subdivision pair can thicken it and round the corners. It lives under the **Weaves & Tangles** menu.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Outer Solid | Dodecahedron | The solid on the outside; its dual is woven on the inside. Choices: the 5 Platonic solids, the 13 Archimedean solids (each woven to its Catalan dual), and Geodesic Sphere (woven to its Goldberg dual of hexagons + 12 pentagons). Switching loads that solid's default parameters. |
| Frequency | 2 | Geodesic breakdown frequency — Geodesic Sphere only; higher = more, smaller faces (min 1, max 8). |
| Inner Spin | 90.0 | Spin of every inner polygon about its face normal, in degrees; folds with the inner polygon's rotational symmetry and also selects the outer-to-inner edge pairing (min -720, max 720). |
| Outer Spin | -18.66 | Spin of every outer polygon about its face normal, in degrees (min -720, max 720). |
| Outer Size | 0.32 | Inset outer-polygon size as a fraction of its face (min 0.05, max 1.0). |
| Inner Size | 0.15 | Inset inner-polygon size as a fraction of its face (min 0.05, max 1.0). |
| Inner Scale | 0.76 | Radius of the inner (dual) solid; the outer solid is fixed at 1 (min 0.2, max 2.5). |
| Outer Curvature | 1.05 | Dome each outer polygon toward a sphere through its corners (0 = flat; min 0.0, max 2.0). |
| Inner Curvature | 0.0 | Dome each inner polygon toward a sphere (0 = flat; min 0.0, max 2.0). |
| Stiffness (Outer) | 1.5 | Length of the bezier handle leaving the outer edge; higher = straighter exit (min 0.0, max 3.0). |
| Stiffness (Inner) | 0.4 | Length of the bezier handle leaving the inner edge (min 0.0, max 3.0). |
| Middle Width | 0.29 | Ribbon width at the middle as a fraction of the endpoint width sum (0.5 = linear midpoint, less = a waist; min 0.0, max 1.5). |
| Taper (Outer) | 50.65 | How fast the outer half reaches the middle width (0 = linear; min 0.0, max 100.0). |
| Taper (Inner) | 0.0 | How fast the inner half reaches the middle width (0 = linear; min 0.0, max 100.0). |
| Ribbon Segments | 14 | Lengthwise segments per ribbon (min 2, max 40). |
| Thickness | 0.03 | Solidify shell thickness (0 = no Solidify modifier; min 0.0, max 0.5). |
| Smoothing | 2 | Subdivision-Surface levels rounding the corners (0 = no Subdivision modifier; min 0, max 4). |
| Smooth Shading | On | Shade the mesh smooth. |
| Scale | 1.0 | Uniform output scale (min 0.01, max 100.0). |




## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/twisted_polyhedron__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/twisted_polyhedron__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__TT.png" width="200"><br><sub>Truncated Tetrahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/twisted_polyhedron__CO.png" width="200"><br><sub>Cuboctahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__TC.png" width="200"><br><sub>Truncated Cube</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__TO.png" width="200"><br><sub>Truncated Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/twisted_polyhedron__RCO.png" width="200"><br><sub>Rhombicuboctahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__TCO.png" width="200"><br><sub>Truncated Cuboctahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__SC.png" width="200"><br><sub>Snub Cube</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/twisted_polyhedron__ID.png" width="200"><br><sub>Icosidodecahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__TD.png" width="200"><br><sub>Truncated Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__TI.png" width="200"><br><sub>Truncated Icosahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/twisted_polyhedron__RID.png" width="200"><br><sub>Rhombicosidodecahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__TID.png" width="200"><br><sub>Truncated Icosidodecahedron</sub></td>
<td align="center"><img src="../images/variants/twisted_polyhedron__SD.png" width="200"><br><sub>Snub Dodecahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/twisted_polyhedron__GEODESIC.png" width="200"><br><sub>Geodesic Sphere</sub></td>
</tr>
</table>

## How it works

**Solid and dual.** The chosen outer solid is built with unit-radius vertices and faces oriented CCW as seen from outside. Its **dual** is computed directly: one dual vertex per face (the unit face centroid), and, per outer vertex, the ring of incident faces ordered CCW — that ring *is* the dual face at the vertex. The dual of a Platonic solid is another Platonic; the dual of an Archimedean solid is its Catalan solid; the dual of the icosahedral geodesic sphere is the Goldberg solid of hexagons and twelve pentagons. The outer solid is scaled to just fill a 2 m cube ($\max|coord| = 1$), and the dual is placed at radius **Inner Scale**.

| Outer solid | Inner (dual) | Outer faces | Inner polygon (vertex degree $m$) |
|-------------|--------------|-------------|-----------------------------------|
| Tetrahedron | Tetrahedron (self-dual) | triangles | triangles ($m=3$) |
| Cube | Octahedron | squares | triangles ($m=3$) |
| Octahedron | Cube | triangles | squares ($m=4$) |
| Dodecahedron | Icosahedron | pentagons | triangles ($m=3$) |
| Icosahedron | Dodecahedron | triangles | pentagons ($m=5$) |
| Archimedean | Catalan dual | mixed | equal-sided (= vertex degree) |
| Geodesic | Goldberg | triangles | pentagons & hexagons |

**Inset, spin, cap.** An inset polygon is built on every outer face (scaled toward its centroid by **Outer Size**) and every inner face (by **Inner Size**), then each is spun about its face normal — inner by +**Inner Spin** (CCW), outer by −**Outer Spin**. Each inset polygon is filled with a **subdivision-friendly cap** that avoids a high-valence pole: triangles use a 3-triangle fan to a domed valence-3 apex, and $n\ge4$ polygons use an outer band of quads around a central $n$-gon. The **Outer / Inner Curvature** blends the cap interior toward the sphere through the corners ($0$ = flat), while the corners stay pinned so the ribbon weld is preserved.

**Ribbon pairing.** Each outer face fans out one ribbon per edge. The pairing of an outer edge to an inner edge is **topological, and therefore rotation-proof**: outer face $f$'s edge $(v_i \to v_{i+1})$ binds to the inner polygon at the **forward vertex** $v_{i+1}$, at edge index

$$(j + \text{step}) \bmod m,\qquad \text{step} = \operatorname{round}\!\left(\frac{\text{Inner Spin}}{360/m}\right),$$

where $j$ is $f$'s position in the ring of faces around that vertex and $m$ is the vertex degree. Because the inner polygon is $m$-fold symmetric, `step` folds the pairing with that symmetry: `step = 0` binds each outer edge to the dual edge directly below it, and every $360/m$ of **Inner Spin** advances it by one. This is a clean bijection — every inner edge receives exactly one ribbon — for every solid and at every spin angle.

**Twisted ribbon.** Each ribbon is a pair of cubic bezier rails from the two outer-edge endpoints to the two inner-edge endpoints:

$$\mathbf B(t) = (1-t)^3\mathbf p_0 + 3(1-t)^2 t\,\mathbf p_1 + 3(1-t)t^2\,\mathbf p_2 + t^3\mathbf p_3,$$

with the interior handles leaving each edge along its tangent, at lengths **Stiffness (Outer)** and **Stiffness (Inner)** times the span. The centreline and twist come from the two rails, but the half-width is driven by an eased **width profile**: it runs from the outer half-width to the inner half-width through a middle width $\text{mid}\cdot(w_0+w_1)$ (so **Middle Width** $<0.5$ pinches a waist), reaching that middle width at rates set by **Taper (Outer/Inner)** ($0$ = linear); endpoints stay exact. The ribbon is tessellated across **Ribbon Segments** lengthwise steps.

**Rail pairing and orientability.** For each ribbon the two rails can be paired two ways. The default `"dist"` choice is the geometric non-crossing pairing (swaps the inner endpoints when that shortens the total rail length) — it looks best, but on a few solids (truncated tetrahedron, cuboctahedron, icosidodecahedron) it is globally inconsistent and makes the shell **non-orientable**. The operator therefore builds with `"dist"`, counts non-contiguous manifold seams, and only when the result is non-orientable rebuilds with the always-consistent `"fwd"` pairing — so every other solid keeps the nicer pairing.

**Welding and modifiers.** All ribbon ends coincide with the outer/inner polygon edge vertices, so everything fuses into one welded vertex pool; bmesh merges doubles and unifies windings. A **Thickness** > 0 adds a Solidify modifier and **Smoothing** > 0 adds a Subdivision-Surface modifier, so the whole shell can be thickened and its corners rounded. Switching the **Outer Solid** loads that solid's tuned default parameters (few large faces vs. many small faces want different insets, spins and curvature); the Geodesic Sphere additionally exposes **Frequency**.

## References

- **Bathsheba Grossman**, *[Quintrino](https://www.bathsheba.com/sculpt/quintrino/)* — the direct inspiration for this generator. Quintrino is a metal sculpture built on a dodecahedron, with strap-like arms swirling around its several fivefold and twofold rotational-symmetry axes; it belongs to Grossman's *Ora* family of dual-symmetry pieces. The twisted ribbons weaving an outer solid to its dual here are a generalization of that idea to every Platonic, Archimedean and geodesic solid.
- The generalized twisted dual-weave construction (arbitrary outer solid woven to its dual, with topological edge pairing and orientability repair) is original to the Math Art project; it extends the classic "Platonic solid woven to its dual" sculpture to all Platonic, Archimedean (with Catalan duals) and geodesic (with Goldberg duals) solids.
- Archimedean geometry reuses the extension's Conway/Hart canonicalized `regular_solids` builder; the geodesic sphere reuses the extension's geodesic generator.
