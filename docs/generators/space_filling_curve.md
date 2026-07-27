# Space-filling Curve

![Space-filling Curve](../images/space_filling_curve.png)

## Overview

This generator builds space-filling curves in 2D and 3D: the open **Hilbert** curve, the closed **Moore** curve, and the **Gilbert** generalized Hilbert curve that fills an arbitrary (not just power-of-two) rectangle or box. The curve visits every cell of a grid exactly once with unit steps, then can be rounded with Chaikin corner-cutting and emitted as a bevelled poly curve or plain wire.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Curve | Hilbert 3D | Which curve to build: Hilbert 3D (open, fills a cube), Moore 3D (closed loop filling a cube), Hilbert 2D (open, fills a square), Moore 2D (closed loop filling a square), Gilbert 3D (generalized Hilbert filling an arbitrary W×H×D box, Cerveny), Gilbert 2D (generalized Hilbert filling an arbitrary W×H rectangle, Cerveny) |
| Cells W | 12 | Gilbert grid width |
| Cells H | 8 | Gilbert grid height |
| Cells D | 4 | Gilbert grid depth (even sizes recommended in 3D) |
| Order | 3 | Recursion depth; 3D point count is $8^{\text{order}}$ (3D capped at 5) |
| Tube Radius | 0.03 | Curve bevel depth (0 = wire only) |
| Corner Rounding | 1 | Chaikin corner-cutting passes |
| Bevel Resolution | 4 | Bevel profile resolution |
| Size | 2.0 | Overall size of the curve |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/space_filling_curve__HILBERT3D.png" width="200"><br><sub>Hilbert 3D</sub></td>
<td align="center"><img src="../images/variants/space_filling_curve__MOORE3D.png" width="200"><br><sub>Moore 3D</sub></td>
<td align="center"><img src="../images/variants/space_filling_curve__HILBERT2D.png" width="200"><br><sub>Hilbert 2D</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/space_filling_curve__MOORE2D.png" width="200"><br><sub>Moore 2D</sub></td>
<td align="center"><img src="../images/variants/space_filling_curve__GILBERT3D.png" width="200"><br><sub>Gilbert 3D</sub></td>
<td align="center"><img src="../images/variants/space_filling_curve__GILBERT2D.png" width="200"><br><sub>Gilbert 2D</sub></td>
</tr>
</table>

## How it works

A space-filling curve is a single path through every cell of a discrete grid, adjacent cells differing by one unit step. On a grid of side $2^k$ in $d$ dimensions the path has $2^{kd}$ points.

**Hilbert curve.** The points come from Skilling's transpose algorithm, which maps a linear index $d \in \{0,\dots,2^{kd}-1\}$ to grid coordinates $X \in \mathbb{Z}^d$. The index bits are first scattered into the transposed coordinate words, then a Gray-code decode is applied,

$$X_i \mathrel{{\oplus}{=}} X_{i-1}, \qquad X_0 \mathrel{{\oplus}{=}} \lfloor X_{d-1}/2 \rfloor,$$

and finally an "undo excess work" pass walks bit-planes $Q = 2, 4, \dots, 2^{k}$ exchanging/inverting coordinates so that consecutive indices land on grid-adjacent cells. The result is the classic Hilbert ordering, an **open** curve.

**Moore curve.** The Moore curve is the **closed** variant: $2^d$ copies of an order-$(k{-}1)$ Hilbert block are placed in the orthants of a Gray-code ring around the cube. Each block is mapped by a signed axis permutation (base traversal axis $\alpha$ sent to that block's assigned axis $\tau$, and coordinates flipped per the block's entry-side vector $\sigma$) so that every block's exit cell is grid-adjacent to the next block's entry, and the final block closes back onto the first — a single closed loop filling the cube. Entry sides propagate as $\sigma_\tau \gets 1-\sigma_\tau$ (flip along the exit axis) and $\sigma_{\text{axis}} \gets$ crossing side toward the next orthant.

**Gilbert curve.** For arbitrary rectangle/box sizes the module ports Jakub Červený's `gilbert2d`/`gilbert3d`. The region is described by generator vectors ($\mathbf{a}$ = major extent, $\mathbf{b}$, $\mathbf{c}$) and recursively bisected: a $1$-thick slab is traversed linearly; otherwise the dominant extent is split — halved with a parity correction ($w_2$ odd and $w>2$ nudges the split by one cell) so that the two (or more) sub-blocks join with unit steps. For **even** sizes the whole path is unit-step; **odd** sizes make a few unavoidable diagonal hops, a documented property of the algorithm, while still covering every cell exactly once.

**Rounding and placement.** Integer cells are centered on the origin and scaled so the largest extent equals *Size*. Optional **Chaikin** corner-cutting replaces each edge by the two points

$$p' = \tfrac34 a + \tfrac14 b, \qquad p'' = \tfrac14 a + \tfrac34 b,$$

per pass (endpoints preserved for open curves, wrapped for closed ones), giving rounded corners. The curve is a POLY spline, cyclic for Moore, with a round bevel of the requested radius.

## References

- John Skilling, *Programming the Hilbert curve*, AIP Conf. Proc. 707, 381 (2004). DOI: <https://doi.org/10.1063/1.1751381>
- Wolfram Demonstrations Project, *Hilbert and Moore 3D Fractal Curves*: <https://demonstrations.wolfram.com/HilbertAndMoore3DFractalCurves/>
- Jakub Červený, *gilbert* — generalized Hilbert space-filling curve for rectangular grids (BSD-2-Clause): <https://github.com/jakubcerveny/gilbert>
- E. H. Moore, *On certain crinkly curves*, Trans. Amer. Math. Soc. 1 (1900), 72–90.
- G. Chaikin, *An algorithm for high-speed curve generation*, Computer Graphics and Image Processing 3 (1974), 346–349.
