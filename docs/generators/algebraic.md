# Algebraic Surface

![Algebraic Surface](../images/algebraic.png)

## Overview
Classical algebraic surfaces -- celebrated cubics, quartics, quintics and sextics -- built as implicit level sets $f(x,y,z)=0$ and meshed with the marching-tetrahedra extractor from the sibling Minimal Surface Toolkit. The presets are the famous node-record surfaces (Clebsch, Cayley, Kummer, Barth, Togliatti) together with a handful of visually striking shapes (Taubin heart, Ding-dong, Chmutov sextic, tangle cube). Geometry only; materials and rendering are left to Blender.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Preset | Clebsch Diagonal Cubic | Which classical surface to build: Clebsch cubic, Cayley nodal cubic, Kummer quartic, Barth sextic, Togliatti quintic, Taubin heart, Ding-dong, Chmutov sextic, or tangle cube. |
| Resolution | 80 | Sample grid resolution per axis (algebraic surfaces need more than TPMS). Range 16-256. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |
| Kummer Mu | 1.3 | Kummer quartic parameter (node sharpness); used by the Kummer preset only. Range 1.05-2.0. |
| Clip Override | 0.0 | Clip ball radius / box half-extent; 0 uses the preset default. Range 0-20. |
| Thickness | 0.0 | If > 0, add a Solidify modifier with this thickness. Range 0-1. |
| Smooth Shading | On | Shade the mesh smooth. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/algebraic__CLEBSCH.png" width="200"><br><sub>Clebsch Diagonal Cubic</sub></td>
<td align="center"><img src="../images/variants/algebraic__CAYLEY.png" width="200"><br><sub>Cayley Nodal Cubic</sub></td>
<td align="center"><img src="../images/variants/algebraic__KUMMER.png" width="200"><br><sub>Kummer Quartic</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/algebraic__BARTH.png" width="200"><br><sub>Barth Sextic</sub></td>
<td align="center"><img src="../images/variants/algebraic__TOGLIATTI.png" width="200"><br><sub>Togliatti Quintic</sub></td>
<td align="center"><img src="../images/variants/algebraic__HEART.png" width="200"><br><sub>Taubin Heart</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/algebraic__DINGDONG.png" width="200"><br><sub>Ding-dong</sub></td>
<td align="center"><img src="../images/variants/algebraic__CHMUTOV.png" width="200"><br><sub>Chmutov Sextic</sub></td>
<td align="center"><img src="../images/variants/algebraic__TANGLE.png" width="200"><br><sub>Tangle Cube</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/algebraic__MONKEY.png" width="200"><br><sub>Monkey Saddle (n-fold)</sub></td>
</tr>
</table>

## How it works

Each preset is a real polynomial $f(x,y,z)$ whose **zero level set** $\{f=0\}$ is the surface. The generator samples $f$ on a regular $\text{res}\times\text{res}\times\text{res}$ grid over a cube and extracts the isosurface with **marching tetrahedra** (`marching_tets` from the Minimal Surface Toolkit): each grid cube is split into tetrahedra, and where $f$ changes sign along a tetrahedron edge a vertex is placed by linear interpolation, with triangle winding oriented along the field gradient.

The polynomials (as commented in the code):

- **Clebsch diagonal cubic** -- the smooth cubic on which all 27 lines are real:
$$81(x^3+y^3+z^3) - 189\!\!\sum_{\text{sym}}\!\! x^2 y + 54xyz + 126(xy+xz+yz) - 9(x^2+y^2+z^2) - 9(x+y+z) + 1 = 0$$
- **Cayley nodal cubic** -- four conical nodes, the maximum on a cubic:
$$4(x^2+y^2+z^2) + 16xyz - 1 = 0$$
- **Kummer quartic** -- 16 nodes for generic $\mu$, with $\lambda=\dfrac{3\mu^2-1}{3-\mu^2}$ and the four tetrahedral tangent planes $p,q,r,s$:
$$(x^2+y^2+z^2-\mu^2)^2 - \lambda\,pqrs = 0,$$
$$p=1-z-\sqrt2\,x,\; q=1-z+\sqrt2\,x,\; r=1+z+\sqrt2\,y,\; s=1+z-\sqrt2\,y$$
- **Barth sextic** -- 65 nodes, the maximum for a sextic; $\varphi$ the golden ratio:
$$4(\varphi^2 x^2 - y^2)(\varphi^2 y^2 - z^2)(\varphi^2 z^2 - x^2) - (1+2\varphi)(x^2+y^2+z^2-1)^2 = 0$$
- **Togliatti quintic** -- 31 nodes, the maximum for a quintic; with $a=\sqrt{5-\sqrt5}$ (see the code for the full quartic factor).
- **Taubin heart** (z up):
$$\left(x^2 + \tfrac94 y^2 + z^2 - 1\right)^3 - x^2 z^3 - \tfrac{9}{80} y^2 z^3 = 0$$
- **Ding-dong** -- a droplet on a cone: $x^2+y^2-(1-z)z^2 = 0$.
- **Chmutov sextic** -- $T_6(x)+T_6(y)+T_6(z)=0$ with the degree-6 Chebyshev polynomial $T_6(t)=32t^6-48t^4+18t^2-1$.
- **Tangle cube** -- $x^4-5x^2+y^4-5y^2+z^4-5z^2+11.8=0$.

Each preset carries its own **clip region** framing the interesting part of the (usually unbounded) surface: a `BOX` with half-extent $r$ (the sample box itself supplies the clip, leaving the level set open where it crosses the boundary), or a `BALL` of radius $r$ (the bounding cube is sampled and then triangles whose centroid falls outside the ball are culled, leaving an open, even rim; orphaned vertices are dropped). Finally the mesh is centered on the origin and fit within a 2 m cube before the Scale factor is applied. If Thickness > 0 a Solidify modifier gives it a shell.

## References

- W. Fischer (ed.), *Mathematical Models: From the Collections of Universities and Museums*, Vieweg, 1986 (Clebsch, Cayley, Kummer surfaces).
- A. Cayley, "A Memoir on Cubic Surfaces," *Philosophical Transactions of the Royal Society of London*, 159 (1869), pp. 231-326.
- W. Barth, "Two projective surfaces with many nodes, admitting the symmetries of the icosahedron," *Journal of Algebraic Geometry* 5 (1996), pp. 173-186.
- E. G. Togliatti, "Una notevole superficie di 5° ordine con soli punti doppi isolati," *Vierteljschr. Naturforsch. Ges. Zürich* 85 (1940), pp. 127-132.
- G. Taubin, "Distance approximations for rasterizing implicit curves," *ACM Transactions on Graphics* 13(1), 1994 (the heart surface).
- V. Chmutov, surfaces with many singularities; see the algebraic-surface gallery at https://homepage.univie.ac.at/herwig.hauser/gallery.html
- Imaginary / Herwig Hauser algebraic surface gallery: https://imaginary.org/gallery/herwig-hauser-classic
- A. B. Bloomenthal et al., *Introduction to Implicit Surfaces*, Morgan Kaufmann, 1997 (marching-cell isosurface extraction).
