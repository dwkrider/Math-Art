# Topological Surface

![Topological Surface](../images/topological.png)

## Overview
The classic topology menagerie: Klein bottles, the two RP² immersions (cross-cap and Steiner's Roman surface), Boy's surface, orientable genus-$g$ handlebodies, and solid closed strips with $n$ half-twists (the $n=1$ Mobius band is printable). The presets follow chapter 6 of Henry Segerman's *Visualizing Mathematics with 3D Printing*. Non-orientable surfaces cannot embed in 3-space, so the Klein and RP² presets are immersions with self-intersections; boundary identifications are made by vertex index so each mesh carries the correct Euler characteristic.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Surface | Klein Bottle | Which surface: Klein Bottle, Klein Bottle (Figure-8), Cross-Cap, Roman Surface, Boy's Surface, Genus-g Surface, or Twisted Strip (solid). |
| Resolution U | 96 | Samples along $u$ (around); for the genus surface, implicit grid density. Range 8-512. |
| Resolution V | 48 | Samples along $v$ (across / radial). Range 4-512. |
| Genus | 2 | Number of handles for the genus-g surface (verified for 1-5). Range 1-5. |
| Half-Twists | 1 | Half-twists per revolution of the twisted strip; 1 = Mobius band. Range 0-12. |
| Strip Width | 0.6 | Width of the twisted strip's cross-section. Range 0.05-2.0. |
| Strip Thickness | 0.18 | Thickness of the twisted strip's cross-section. Range 0.01-1.0. |
| Center Ridge | Off | Raised ridge along the strip center line (as in Segerman fig 6-1). |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |
| Thickness | 0.0 | Immersed surfaces only: 0 = raw surface, > 0 = Solidify modifier of this thickness. Range 0-1. |
| Smooth Shading | On | Shade the mesh smooth. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/topological__KLEIN.png" width="200"><br><sub>Klein Bottle</sub></td>
<td align="center"><img src="../images/variants/topological__KLEIN8.png" width="200"><br><sub>Klein Bottle (Figure-8)</sub></td>
<td align="center"><img src="../images/variants/topological__SUDANESE.png" width="200"><br><sub>Sudanese Mobius Band</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/topological__CROSSCAP.png" width="200"><br><sub>Cross-Cap</sub></td>
<td align="center"><img src="../images/variants/topological__ROMAN.png" width="200"><br><sub>Roman Surface</sub></td>
<td align="center"><img src="../images/variants/topological__BOY.png" width="200"><br><sub>Boy's Surface</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/topological__GENUS.png" width="200"><br><sub>Genus-g Surface</sub></td>
<td align="center"><img src="../images/variants/topological__TWIST_STRIP.png" width="200"><br><sub>Twisted Strip (solid)</sub></td>
</tr>
</table>

## How it works

Each preset is a distinct construction; all are welded (or index-identified) so their combinatorial topology is exact.

**Klein bottle** (the iconic bottle shape) uses the standard smooth closed-form immersion over $u\in[0,\pi]$, $v\in[0,2\pi]$: a degree-high trigonometric parametrization $x(u,v),y(u,v),z=\tfrac{2}{15}\sin v\,(3+5\cos u\sin u)$. The $u=\pi$ rim coincides with $u=0$ under $v\mapsto\pi-v$, but the seam is left **split** (coincident duplicate vertices, no index gluing): welding it would flip the winding there and darken the shading crease. Cut along that rim the surface is an orientable cylinder, so $\chi=0$.

**Klein bottle (figure-8)** is the twisted-torus immersion: the cross-section is a figure-8 that makes a half-turn per revolution,
$$r = R + \cos\tfrac{u}{2}\sin v - \sin\tfrac{u}{2}\sin 2v,\quad x=r\cos u,\; y=r\sin u,\; z=\sin\tfrac{u}{2}\sin v + \cos\tfrac{u}{2}\sin 2v.$$
$v$ samples sit at half-steps so no column lands on the crossing point.

**Cross-cap** and **Roman surface** are meshed by `_rp2_quotient`: a hemisphere parametrization $(\theta,\phi)$, $\phi\in(0,\tfrac\pi2]$, with the pole collapsed to one vertex and the equator glued to itself by $\theta\mapsto\theta+\pi$ (the RP² quotient), giving $\chi=1$ by construction. The cross-cap maps antipodes of the sphere together; the Roman surface is the sphere through $(x,y,z)\mapsto(yz,zx,xy)$.

**Boy's surface** uses the Bryant-Kusner parametrization on the unit disk (polar grid) with the boundary circle glued antipodally. For a complex disk coordinate $z$, with $w=z^6+\sqrt5\,z^3-1$,
$$g_1=-\tfrac32\,\mathrm{Im}\!\frac{z(1-z^4)}{w},\quad g_2=-\tfrac32\,\mathrm{Re}\!\frac{z(1+z^4)}{w},\quad g_3=\mathrm{Im}\!\frac{1+z^6}{w}-\tfrac12,$$
and the point is $(g_1,g_2,g_3)/\lVert g\rVert^2$. The three poles of $w$ invert to the triple point at the origin; samples landing on them are nudged off. $\chi=1$.

**Genus-$g$ handlebody** is implicit, meshed by marching tetrahedra. In the plane, $q(x,y)$ is the product over $g+1$ overlapping circles in a row of the **normalized** factors $\dfrac{\rho_i^2-r^2}{\rho_i^2+r^2}\in(-1,1)$, $\rho_i$ the distance to center $i$. Normalization keeps the saddle-channel width $O(\sqrt{\varepsilon})$ independent of $g$. Adding $k z^2$ keeps every vertical fiber an interval, so $\{q + k z^2 \le \varepsilon\}$ is a genus-$g$ handlebody whose boundary has $\chi = 2-2g$.

**Twisted strip** sweeps a rectangular cross-section (optionally with a raised center ridge) around a circle, rotating it by $\tfrac12 n\,t$ over one revolution $t\in[0,2\pi]$. The profile point list is symmetric under a half-turn, so for odd $n$ the seam closes with an index shift and the result is a single **watertight solid** (printable). $n=1$ is the Mobius band.

All presets are centered and fit within a 2 m cube, then Scaled; immersed presets can be given a Solidify shell.

## References

- H. Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 (chapter 6, figs 6-1..6-7).
- R. Bryant and R. Kusner parametrization of Boy's surface; see R. Kusner, "Conformal geometry and complete minimal surfaces," *Bull. Amer. Math. Soc.* 17 (1987), pp. 291-295.
- W. Boy, "Über die Curvatura integra und die Topologie geschlossener Flächen," *Mathematische Annalen* 57 (1903), pp. 151-184.
- J. Steiner, Roman surface; see D. Hilbert and S. Cohn-Vossen, *Geometry and the Imagination*, Chelsea, 1952.
- Weierstrass/nodal genus construction and marching tetrahedra as used by the sibling Minimal Surface Toolkit (Ken Brakke's periodic-surface pages, https://kenbrakke.com/evolver).
