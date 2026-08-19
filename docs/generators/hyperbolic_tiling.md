# Hyperbolic Tiling

![Hyperbolic Tiling](../images/hyperbolic_tiling.png)

## Overview
Regular $\{p,q\}$ tilings of the hyperbolic plane -- $q$ regular $p$-gons meeting at every vertex -- realized as printable relief models after Henry Segerman. The tiling is generated in the hyperboloid model of $\mathbb H^2$ via the $(p,q,2)$ reflection group, two-coloured by word parity, and then mapped into one of three pictures: the conformal **Poincare disk** plaque, the **hemisphere** model, or the **pseudosphere** (tractricoid), onto which a strip of $\mathbb H^2$ is wrapped genuinely isometrically.

## Options

| Option | Default | Description |
| --- | --- | --- |
| p | 3 | Sides of each tile ($\{p,q\}$: $q$ $p$-gons meet at every vertex). Range 3-20. |
| q | 7 | Tiles meeting at each vertex (hyperbolic requires $1/p+1/q<1/2$). Range 3-24. |
| Model | Poincare Disk | Poincare Disk (relief plaque on a backing slab), Hemisphere (Klein disk lifted onto the upper unit hemisphere), or Pseudosphere (wrapped isometrically onto the tractricoid). |
| Depth | 8 | Reflection word length explored (how far the tiling spreads). Range 1-14. |
| Max Triangles | 4000 | Cap on the number of fundamental triangles generated. Range 10-20000. |
| Subdivision | 2 | Geodesic subdivisions per fundamental triangle (so tile edges curve correctly in the model). Range 0-4. |
| Relief Height | 0.05 | Height of the raised-parity tiles. Range 0.005-0.5. |
| Base Thickness | 0.1 | Backing slab thickness (Poincare disk plaque). Range 0.01-1.0. |
| Shell Thickness | 0.08 | Shell wall thickness (Hemisphere / Pseudosphere). Range 0.01-0.5. |
| Cusp Cap | 6.0 | Clip the pseudosphere cusp at upper-half-plane height $y$ (the horn gets thin). Range 1.5-50. |
| Include Base Disk | On | Backing slab under the raised tiles (disk plaque); off = only the raised tiles, flat on $z=0$. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/hyperbolic_tiling__POINCARE.png" width="200"><br><sub>Poincare Disk</sub></td>
<td align="center"><img src="../images/variants/hyperbolic_tiling__KLEIN.png" width="200"><br><sub>Klein Disk</sub></td>
<td align="center"><img src="../images/variants/hyperbolic_tiling__HEMISPHERE.png" width="200"><br><sub>Hemisphere</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/hyperbolic_tiling__PSEUDOSPHERE.png" width="200"><br><sub>Pseudosphere</sub></td>
</tr>
</table>

## How it works

**When is $\{p,q\}$ hyperbolic?** The angle sum condition. A regular $p$-gon with vertex angle $2\pi/q$ tiles the sphere, plane, or hyperbolic plane according as
$$\tfrac1p+\tfrac1q \;>,\;=,\;<\;\tfrac12 .$$
Equivalently, in terms of the $(p,q,2)$ triangle, the tiling is hyperbolic iff $\cos(\pi/q)>\sin(\pi/p)$.

**Hyperboloid model and the reflection group.** The tiling is built in the hyperboloid model of $\mathbb H^2$ sitting in Minkowski space $\mathbb R^{2,1}$ with inner product $\langle x,y\rangle = x_1y_1+x_2y_2-x_3y_3$; points of $\mathbb H^2$ satisfy $\langle x,x\rangle=-1$, $t>0$. The $(p,q,2)$ group is realized by three mirrors with unit spacelike normals $\mathbf n_0,\mathbf n_1,\mathbf n_2$ whose pairwise Gram products encode the triangle angles:
$$\langle\mathbf n_0,\mathbf n_1\rangle=-\cos\tfrac\pi p,\quad \langle\mathbf n_1,\mathbf n_2\rangle=-\cos\tfrac\pi q,\quad \langle\mathbf n_0,\mathbf n_2\rangle=0\ (\text{the right angle}).$$
Reflection in a mirror is $x\mapsto x-2\langle x,\mathbf n\rangle\,\mathbf n$. A breadth-first search over reflection words generates all tiles up to *Depth*, each represented by its group matrix; duplicates are pruned by the rounded ambient centroid, and the word length modulo 2 gives the two-colouring. Every fundamental triangle is barycentrically subdivided ($2^{\text{subdiv}}$) and each point reprojected to $\langle x,x\rangle=-1$, so edges follow geodesics after mapping into the target model.

**Poincare disk.** The hyperboloid maps to the conformal disk by $(x,y,t)\mapsto(x,y)/(1+t)$. Tiles of one parity are raised to `relief` above a backing slab of radius $\approx1$.

**Hemisphere.** Klein-disk coordinates $H_{xy}/H_t$ are lifted vertically onto the upper unit hemisphere $z=\sqrt{1-x^2-y^2}$, and tiles are offset radially by parity, giving a shell.

**Pseudosphere isometry.** The tractricoid
$$\sigma(u,v)=(\operatorname{sech}u\cos v,\ \operatorname{sech}u\sin v,\ u-\tanh u),\quad u\ge0,$$
has first fundamental form $ds^2=\tanh^2u\,du^2+\operatorname{sech}^2u\,dv^2$. Substituting $x=v,\ y=\cosh u$ into the upper-half-plane metric $(dx^2+dy^2)/y^2$ yields the **same** form, so $y=\cosh u$ (not $y=e^u$, which gives a different, horocyclic chart) wraps the UHP region $y\ge1$, $x$ taken mod $2\pi$, isometrically onto the pseudosphere -- a genuinely isometric embedding of a strip of $\mathbb H^2$. The tiling is carried from the disk to the UHP by the Cayley map, then Moebius-shifted to centre the view; tiles are clipped to $1\le y\le y_{\text{cap}}$ (the $u=0$ rim to the thinning cusp) over one seam period $0\le x\le2\pi$.

Every emitted tile is its own watertight closed shell (top, reversed bottom, and wall quads along the boundary); for printing the shells overlap so a slicer/boolean union is trivial.

## References

- H. Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 (chapter 4, figs 4-2, 4-8..4-14 -- disk plaque, hemisphere and pseudosphere hyperbolic tilings).
- The hyperboloid / Minkowski model of $\mathbb H^2$ and $(p,q,2)$ triangle reflection groups: standard hyperbolic geometry (e.g. J. G. Ratcliffe, *Foundations of Hyperbolic Manifolds*, Springer).
- Pseudosphere (tractricoid) as a surface of constant curvature $-1$; the $y=\cosh u$ isometry with the upper half plane is derived in full in the module header.
