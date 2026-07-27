# Zonohedra

![Zonohedra](../images/zonohedra.png)

## Overview

Zonohedra are convex polyhedra whose faces are all centrally symmetric — the Minkowski sum of a set of line segments (a "vector star"). This generator builds them two ways: a robust general mode that takes the convex hull of all subset sums of a star (giving the rhombic dodecahedron, triacontahedron, enneacontahedron, cube, or a random star), and a direct port of Antiprism's `make_polar_zonohedron` for **polar zonohedra** and Russell Towle's **rhombic spirallohedra**. The preset `Rhombic Spirallohedron` reproduces `zono -P 12,4`.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Star | Rhombic Spirallohedron | Which vector star / construction: Polar Zonohedron, Rhombic Spirallohedron, Rhombic Dodecahedron (4 cube diagonals), Rhombic Triacontahedron (6 icosahedral axes), Rhombic Enneacontahedron (10 dodecahedral axes), Cube (3 orthogonal vectors), or Random Star. |
| Vectors | 12 | Number of star vectors, for the polar / spiral / random stars (min 3, max 64). |
| Spiral Width | 4 | Spirallohedron spiral width $w$ (as in `zono -P n,w`); must not be a multiple of the vector count (min 1, max 31). |
| Pitch | 55.0 | Polar-star pitch angle from the axis, in degrees (min 5.0, max 85.0). |
| Random Seed | 1 | Seed for the random unit-vector star (min 0). |
| Style | Solid | Plain closed zonohedron, Leonardo (da Vinci) open-faced panels, or Wireframe struts along the zone edges. |
| Border | 0.3 | Leonardo face-frame width, as a fraction of the face (min 0.02, max 0.95). |
| Thickness | 0.05 | Panel / strut thickness for the Leonardo and Wireframe styles (min 0.001, max 1.0). |
| Scale | 1.0 | Uniform output scale (min 0.01, max 100.0). |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/zonohedra__POLAR.png" width="200"><br><sub>Polar Zonohedron</sub></td>
<td align="center"><img src="../images/variants/zonohedra__SPIRAL.png" width="200"><br><sub>Rhombic Spirallohedron</sub></td>
<td align="center"><img src="../images/variants/zonohedra__RHOMBIC_DODECA.png" width="200"><br><sub>Rhombic Dodecahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/zonohedra__TRIACONTA.png" width="200"><br><sub>Rhombic Triacontahedron</sub></td>
<td align="center"><img src="../images/variants/zonohedra__ENNEACONTA.png" width="200"><br><sub>Rhombic Enneacontahedron</sub></td>
<td align="center"><img src="../images/variants/zonohedra__CUBE.png" width="200"><br><sub>Cube</sub></td>
</tr>
</table>

## How it works

A **zonohedron** is the Minkowski sum of a star of line segments $\{\pm\tfrac12\mathbf v_i\}$. Equivalently it is the set of all vectors $\sum_i t_i \mathbf v_i$ with $t_i \in [0,1]$, and its extreme points are exactly the $2^{n}$ **subset sums** — each choice of "include $\mathbf v_i$ or not." Every face is a **zonogon** (a centrally symmetric polygon); when the star is in general position all faces are rhombi, one per pair of star vectors.

**General mode.** The generator forms all subset sums of the star and takes their **convex hull**. A limit-dissolve then merges the hull's coplanar triangles back into the zonohedron's true rhombi / zonogons. The built-in stars are:

- **Cube** — the three coordinate axes $\mathbf e_x, \mathbf e_y, \mathbf e_z$.
- **Rhombic dodecahedron** — the four cube diagonals $(\pm1,\pm1,\pm1)$ (an even set), each normalized.
- **Rhombic triacontahedron** — the six icosahedral 5-fold axes, from cyclic permutations of $(0, 1, \varphi)$.
- **Rhombic enneacontahedron** — the ten dodecahedral 3-fold axes.
- **Random star** — $n$ Gaussian unit vectors (capped at 13 for the hull mode).

**Polar zonohedra and spirallohedra.** A *polar* star places $n$ unit vectors at equal azimuth spacing around a common axis, all sharing one pitch angle $p$ from the axis:

$$\mathbf v_k = \big(\sin p\,\cos\tfrac{2\pi k}{n},\; -\sin p\,\sin\tfrac{2\pi k}{n},\; \cos p\big),\qquad k = 0,\dots,n-1.$$

Rather than hull the $2^n$ sums (astronomically many for large $n$), the port of Antiprism's `make_polar_zonohedron` builds the faces **directly**. It walks the rhombi in spiral chains: with $\gcd(n, \text{step})$ parts and a per-part period $P = n/\gcd$, it accumulates partial sums $A$ (down one spiral) and $B$ (across the spiral width) and emits each rhombic face as four indexed corners via a closed-form index map, closing the top and bottom apex points. With **spiral width 0** this produces the ordinary polar zonohedron; a **nonzero spiral width** $w$ produces a **rhombic spirallohedron** of that width — the family Russell Towle studied, in which the rhombi wind up the solid in helical bands. The width must not be a multiple of $n$ (that would degenerate the spiral).

Finally the mesh is welded (`remove_doubles`), its normals recalculated, and it is recentred on the origin and scaled to fit within a $2\times$ `Scale` cube regardless of what the star produced.

## References

- Russell Towle, *Zonohedra* and his rhombic spirallohedron work — <http://www.zonohedra.com/>
- Adrian Rossiter, *Antiprism* and its `zono` program (`base/zonohedron.cc`, `make_polar_zonohedron`) — <https://www.antiprism.com>, <https://github.com/antiprism/antiprism> (GPL; the direct port source).
- H. S. M. Coxeter, *Regular Polytopes*, 3rd ed., Dover, 1973 (zonohedra as Minkowski sums / zones).
- G. C. Shephard, *Space-filling zonotopes*, Mathematika 21, pp. 261–269, 1974.
