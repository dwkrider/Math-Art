# Curiosity Surface

![Curiosity Surface](../images/curiosity_surface.png)

## Overview
Three classic surfaces from the geometry literature: Fresnel's elasticity surface (a quartic radial surface from optics), the paper bag surface (a crimped inflated-bag plot), and the trihyperboloid (the boundary of the solid enclosed by three mutually perpendicular hyperboloids, with exact volume $\ln 256 = 8\ln 2$). The two radial surfaces are meshed as displaced UV spheres and come out watertight; the paper bag stays open like a real bag.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Surface | Fresnel Elasticity Surface | Which surface: Fresnel Elasticity Surface, Paper Bag Surface, or Trihyperboloid. |
| Semi-Axis A | 1.0 | Fresnel semi-axis along X. Range 0.01-10. |
| Semi-Axis B | 1.5 | Fresnel semi-axis along Y. Range 0.01-10. |
| Semi-Axis C | 2.0 | Fresnel semi-axis along Z. Range 0.01-10. |
| Height Coefficient | 2.47 | Paper bag coefficient $a$ in $z = a v^2$ (2.47 in the classic plot). Range 0.01-10. |
| Crimp Coefficient | -1.26 | Paper bag coefficient $b$ in $y = (v + b u)\sin u$ (-1.26 in the classic plot). Range -10-10. |
| Resolution | 48 | Rings across the surface (twice as many segments around). Range 6-256. |
| Smooth Shading | On | Shade the mesh smooth. |
| Thickness | 0.0 | Solidify modifier thickness (0 = raw surface). Range 0-1. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |





## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/curiosity_surface__FRESNEL.png" width="200"><br><sub>Fresnel Elasticity</sub></td>
<td align="center"><img src="../images/variants/curiosity_surface__PAPERBAG.png" width="200"><br><sub>Paper Bag</sub></td>
<td align="center"><img src="../images/variants/curiosity_surface__TRIHYPERBOLOID.png" width="200"><br><sub>Trihyperboloid</sub></td>
</tr>
</table>

## How it works

**Radial surfaces.** Both the Fresnel surface and the trihyperboloid are *radial graphs* over the sphere of directions: a radius function $r(l,m,n)$ evaluated on unit directions $(l,m,n)$ and meshed as a displaced UV sphere (`build_radial`) with poles and the $\theta$ seam welded, so the mesh is watertight and winds outward.

**Fresnel's elasticity surface** (von Seggern 1993, p. 304) is the quartic
$$(x^2+y^2+z^2)^2 = a^2 x^2 + b^2 y^2 + c^2 z^2,$$
equivalently the radial surface $r(l,m,n) = \sqrt{a^2 l^2 + b^2 m^2 + c^2 n^2}$ where $(l,m,n)$ is the unit direction of the radius vector. The three coordinate axes hit $r = a, b, c$.

**Paper bag surface** (Robin 2004; plotted after Trott 2004, p. 103) is the crimped-bag parametrization
$$x = v\cos u,\quad y = (v + b u)\sin u,\quad z = a v^2,$$
for $u\in[0,2\pi]$, $v\in[0,\text{depth}]$ (depth = 2), with the classic constants $a = 2.47$ (height coefficient) and $b = -1.26$ (crimp coefficient). The $u=0$ and $u=2\pi$ boundary curves coincide in space and are welded; the surface stays **open** at $v=0$ and $v=\text{depth}$ like a real bag.

**Trihyperboloid** (Knill 2017; Villarino and Varilly 2024) is the boundary of the solid enclosed by the three hyperboloids
$$x^2+y^2-z^2\le 1,\quad y^2+z^2-x^2\le 1,\quad z^2+x^2-y^2\le 1,$$
shaped like a stella octangula with webs hung across adjacent faces. Along a unit direction $(l,m,n)$ the largest of the three quadratic forms
$$q_{\max} = \max\big(l^2+m^2-n^2,\; m^2+n^2-l^2,\; n^2+l^2-m^2\big)$$
is at least $1/3$, so the boundary is the radial graph $r = 1/\sqrt{q_{\max}}$ with $1\le r\le\sqrt3$. The enclosed volume is exactly $\ln 256 = 8\ln 2$.

In every case the mesh is centered and fit within a $2\times\text{scale}$ cube, then optionally given a Solidify shell.

## References

- D. H. von Seggern, *CRC Standard Curves and Surfaces*, CRC Press, 1993, p. 304 (Fresnel's elasticity surface).
- V. Robin, paper bag surface (2004); M. Trott, *The Mathematica GuideBook for Graphics*, Springer, 2004, p. 103 (the crimped-bag plot).
- O. Knill, "The volume of the intersection of three cylinders / hyperboloids" and related notes (2017); see http://people.math.harvard.edu/~knill/
- M. B. Villarino and J. C. Varilly, work on the trihyperboloid / intersection-of-hyperboloids volume ($8\ln 2$), 2024.
