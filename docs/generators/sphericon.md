# Sphericon

![Sphericon](../images/sphericon.png)

## Overview

A **sphericon** is a developable roller built from a regular polygon. Take the solid of revolution of a regular $n$-gon about one of its axes of symmetry, slice it in half with a plane through that axis, rotate one half so the polygon cross-section lands back on itself, and rejoin the two halves. The result rolls with a lurching, meandering motion whose contact point traces the whole surface — a cousin of the [Oloid](oloid.md). The classic sphericon is the $n=4$ case (a square swept into a bicone, halved and given a quarter turn); this generator makes the whole **$(n,k)$ polysphericon** family, including the **heptagonal sphericon** ($n=7$).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Polygon Sides | 4 | $n$: sides of the regular polygon revolved into the roller. **4** = the classic sphericon, **7** = the heptagonal sphericon; 3–24 supported. |
| Rotation Steps | 1 | $k$: the second half is turned by $k\cdot360/n$ degrees so the cross-section maps onto itself. $k=1$ is the fundamental sphericon; other $k$ (coprime to $n$) give distinct polysphericons. |
| Segments | 96 | Angular resolution over each half turn (smoothness around the axis). |
| Coloring | Per Conical Band | Give each conical band its own colour, or use a single plain material. |
| Smooth Shading | On | Shade the surface smooth; the band boundaries and cut seam are creased so the cones stay crisp while the circumference stays round. |
| Scale | 1.0 | Uniform output scale (1.0 = fit a 2 m cube, centered on the origin). |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/sphericon__3.png" width="200"><br><sub>Triangular (3)</sub></td>
<td align="center"><img src="../images/variants/sphericon__4.png" width="200"><br><sub>Sphericon (4)</sub></td>
<td align="center"><img src="../images/variants/sphericon__5.png" width="200"><br><sub>Pentagonal (5)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/sphericon__6.png" width="200"><br><sub>Hexagonal (6)</sub></td>
<td align="center"><img src="../images/variants/sphericon__7.png" width="200"><br><sub>Heptagonal (7)</sub></td>
<td align="center"><img src="../images/variants/sphericon__8.png" width="200"><br><sub>Octagonal (8)</sub></td>
</tr>
</table>

## How it works

**Solid of revolution.** A regular $n$-gon is spun about one of its axes of symmetry to make a spindle of **conical bands** (each polygon edge sweeps a cone or frustum). Which axis depends on the parity of $n$:

- **Even $n$:** the axis runs through two opposite vertices, so both ends of the profile are cone apexes. For $n=4$ this is a **bicone** (two cones base to base) — a square standing on its diagonal.
- **Odd $n$:** a regular polygon with an odd number of sides has no diameter through two vertices; its only symmetry axes pass through one vertex and the **midpoint of the opposite edge**. So one end of the spindle is a pointed apex and the other a disc. The heptagon ($n=7$) is this case.

**Cut and rotate.** The spindle is cut in half by a plane containing the axis; the cut face of each half is the whole $n$-gon. One half is turned by

$$\theta = k\cdot\frac{360^\circ}{n}$$

about the normal of the cut plane. Because the regular $n$-gon has $n$-fold rotational symmetry, this angle carries the cross-section exactly onto itself, so the halves rejoin into a closed surface — but the conical bands of the two halves no longer line up, which is what gives the sphericon its characteristic offset silhouette. The standard sphericon is $(n,k)=(4,1)$, a $90^\circ$ turn.

**Developability and rolling.** Every piece of the surface is part of a cone, so the whole surface is **developable** (it unrolls flat — Swart's paper unwraps it as a "snake" of circular arcs). As the sphericon rolls, its centre of mass stays at a constant height and the contact point sweeps the entire surface. Not every $(n,k)$ rolls smoothly: the clean rollers are those whose polygon has an even number of vertices with the axis through two corners and $\gcd(n/2,k)=1$; the odd-$n$ pieces (like the heptagonal one) are still valid developable solids with the same family of conical bands.

**Watertight seam.** The one subtlety in building the mesh is the weld along the cut. The turn $k\cdot360/n$ maps each polygon **vertex** onto the next vertex, and (for odd $n$) the bottom **edge-midpoint** onto the next edge-midpoint. So the profile is sampled at every polygon vertex *and* every edge midpoint; the rotated half's seam then coincides vertex-for-vertex with the other half's, and the two weld into a single closed, manifold surface ($\chi = 2$) for every $n$ and $k$.

## References

- David Swart, *Arcs on Spheres and Snakes on Planes*, Bridges 2024 Conference Proceedings, pp. 353–356 — <https://archive.bridgesmathart.org/2024/bridges2024-353.html> (the sphericon as the solid of revolution of a regular polygon, halved with its cross-section rotated onto itself; the developed surface as a "snake" of arcs).
- **Polysphericons**, Heidelberg Institute for Theoretical Studies (H-ITS) — <https://www.h-its.org/projects/polysphericons/> (the $(n,k)$ family, rolling condition, odd-vs-even axis).
- C. J. Roberts, *sphericon*; see also Wikipedia, *Sphericon* — <https://en.wikipedia.org/wiki/Sphericon> (history and the classic $(4,1)$ construction).
- Complements the extension's [Oloid](oloid.md), the other developable roller in *Odds & Ends*.
