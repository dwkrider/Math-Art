# Polylinks

![Polylinks](../images/polylinks.png)

## Overview

Orderly tangles of regular polygons, after George W. Hart's **regular polylinks**: one polygonal frame is placed on every face plane of a Platonic solid, each rotated about its face normal, scaled, and pushed along the normal so the frames interlock. Classic examples are 4 triangles (tetrahedron), 6 squares (cube), 8 or 20 triangles (octahedron / icosahedron), and 6 or 12 pentagons (dodecahedron). Beyond the flat polygon frame, each link can also be a radius-modulated wavy circle or a torus knot swept as a tube (both after Shengyi Wang's polylink add-on).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Preset | 4 Triangles (T4) | Ready-made tangle (4 triangles, 6 squares, 8 triangles, 6 or 12 pentagons, 20 triangles) that fills in the solid/size/rotation/offset fields; Custom leaves them as-is. |
| Solid | Tetrahedron | Seed Platonic solid whose faces host the frames: Tetrahedron, Cube, Octahedron, Dodecahedron, Icosahedron. |
| Frame Size | 1.55 | Scale of each polygon frame (min 0.5, max 4.0). |
| Rotation | 30.0 | Turn of each frame about its face normal, in degrees (min -90, max 90). |
| Plane Offset | -0.45 | Push of each frame along its normal; negative moves it toward the centre (min -2.0, max 2.0). |
| Link Shape | Polygon Frame | Polygon Frame (flat frames, Hart's polylinks), Wavy Circle (radius-modulated rings, after Wang), or Torus Knot (a (p, q×sides) torus knot about each face axis). |
| Wave Amplitude | 0.35 | Radial wave amplitude (wavy circle) or knot minor radius (min 0.0, max 2.0). |
| Wave Factor | 1 | Wave frequency as a multiple of the face's side count (min 1, max 8). |
| Knot p | 2 | Windings around the face axis (min 1, max 8). |
| Knot q Factor | 1 | q = factor × face side count (min 1, max 8). |
| Tube Sides | 8 | Cross-section sides of the swept tube for wavy / knot links (min 3, max 24). |
| Link Segments | 128 | Samples along wavy / knot centerlines (min 24, max 512). |
| Frame Width | 0.14 | Width of the flat polygon frame, as a fraction (min 0.02, max 0.9). |
| Frame Thickness | 0.10 | Thickness of the frame / tube (min 0.01, max 1.0). |
| Antipodal Half | Off | Use only one face of each antipodal pair (e.g. 6 pentagons instead of 12). |
| Coloring | Per Link | Per Link (one material per frame), Per Parallel Pair (iso-colour antipodal frames, as in Hart's paper models), or None. |
| Scale | 1.0 | Uniform output scale; the result is fit within a 2×Scale cube at the origin (min 0.01, max 100.0). |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/polylinks__T4.png" width="200"><br><sub>4 Triangles</sub></td>
<td align="center"><img src="../images/variants/polylinks__C6.png" width="200"><br><sub>6 Squares</sub></td>
<td align="center"><img src="../images/variants/polylinks__O8.png" width="200"><br><sub>8 Triangles</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/polylinks__D6.png" width="200"><br><sub>6 Pentagons</sub></td>
<td align="center"><img src="../images/variants/polylinks__D12.png" width="200"><br><sub>12 Pentagons</sub></td>
<td align="center"><img src="../images/variants/polylinks__I20.png" width="200"><br><sub>20 Triangles</sub></td>
</tr>
</table>

## How it works

The seed Platonic solid gives vertices $V$ and faces $F$. For each face $f$ the centroid is

$$c = \frac{1}{m}\sum_{i \in f} V_i,\qquad n = \frac{c}{\lVert c\rVert},$$

where $m$ is the number of sides. Because the seed is a Platonic solid, the outward face normal $n$ is radial. The frame is centred at $c_{\text{en}} = n\,(\lVert c\rVert + \text{offset})$, so a negative **Plane Offset** slides it inward toward the origin.

**Polygon frame.** Each corner vector $d = V_i - c$ is rotated about the normal $n$ by the **Rotation** angle $\theta$ and scaled by **Frame Size** $s$ using Rodrigues' rotation formula:

$$d' = s\big(d\cos\theta + (n\times d)\sin\theta + n\,(n\cdot d)(1-\cos\theta)\big).$$

A ring of these rotated corners, together with an inner ring shrunk by $(1-\text{width})$ and duplicated at $\pm\tfrac{\text{thickness}}{2}$ along $n$, forms a solid mitred frame with top, bottom, outer-wall and inner-wall quads. Rotating and offsetting every face's frame the same way is exactly what makes neighbouring frames pass through one another and lock.

**Wavy circle.** Instead of the polygon, a circle of radius $r = s\,\lVert V_{f_0}-c\rVert$ is drawn in the face plane with a radially modulated radius

$$r(t) = r + A\cos(\text{frq}\cdot t),\qquad \text{frq} = \text{wave\_factor}\cdot m,$$

so the ring has $m$ (or a multiple of $m$) lobes matching the polygon's symmetry, and neighbouring wavy rings weave through each other.

**Torus knot.** A $(p,q)$ torus knot is traced about the face axis, with $q = \text{knot\_q\_factor}\cdot m$ and minor radius $r_2 = A$:

$$P(t) = c_{\text{en}} + (r + r_2\cos qt)\big(\cos pt\,\hat x_N + \sin pt\,\hat y_N\big) + r_2\sin qt\;n,$$

where $\hat x_N,\hat y_N$ are the in-plane axes after the rotation. The wavy and knot centrelines are swept into a closed circular tube using **rotation-minimizing frames** (the double-reflection method): a normal is propagated along the loop by reflecting it across each segment and across the tangent change, and the residual closure twist is measured and distributed uniformly ($-\text{ang}\cdot i/n$ at sample $i$) so the tube joins seamlessly end to end.

Each frame's face membership is written as a `link_index` face attribute. **Per Parallel Pair** colouring groups frames by the axis of their normal (identifying $n$ with $-n$), reproducing Hart's convention of, e.g., colouring 6 squares in 3 colours.

## References

- George W. Hart, *Orderly Tangles Revisited* — <https://www.georgehart.com/orderly-tangles-revisited/tangles.htm>
- Alan Holden, *Orderly Tangles: Cloverleafs, Gordian Knots, and Regular Polylinks*, Columbia University Press, 1983.
- Shengyi Wang (txyyss), *polylink* Blender add-on — <https://github.com/txyyss/polylink> (the wavy-circle and torus-knot link variants and the rotation-minimizing tube sweep).
