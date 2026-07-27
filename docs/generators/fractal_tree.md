# Fractal Tree

![Fractal Tree](../images/fractal_tree.png)

## Overview

This generator builds parametric $n$-ary fractal trees and hanging mobiles, after the Mahler–Segerman ternary tree mobile (fig. 7-2 in Segerman's *Visualizing Mathematics with 3D Printing*). In **Tree** mode a trunk splits recursively into `arity` children per tip, each tilted by the branch angle, spread evenly in azimuth, and scaled by `ratio` per generation, with per-point radii producing solidly tapered printable limbs and optional icosphere leaves. In **Mobile** mode the same branching structure hangs as strings, spreader bars and sphere weights. Everything is deterministic (no random seed), fit to a 2 m cube, and capped at 8000 segments.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Mode | Tree | Tree (recursive branching with tapering limbs) or Mobile (hanging mobile: strings, spreader bars and sphere weights) |
| Arity | 3 | Branches (or spreader arms) per node |
| Depth | 5 | Branching generations (Tree mode); clamped to keep the total under 8000 segments |
| Levels | 4 | Spreader levels (Mobile mode); clamped to keep the total under 8000 segments |
| Branch Angle | 35° | Tilt of each child branch away from its parent's direction |
| Ratio | 0.67 | Length and radius scale factor per generation |
| Azimuth Twist | 0° | Extra rotation of each generation's azimuth spread (try the golden angle 137.5° for non-aligned crowns). Mobile mode uses 60° while this is left untouched |
| Trunk Length | 2.0 | Length of the trunk (Tree) or of the level-0 string and arms (Mobile) |
| Trunk Radius | 0.06 | Bevel radius at the trunk; limbs taper by Ratio per generation (0 = wire) |
| Leaf Spheres | Off | Merge a small icosphere at each final tip (Tree mode; forces mesh output) |
| Sphere Size | 0.12 | Radius of the leaf spheres / mobile weights |
| Output | Mesh | Convert the bevelled curve to a mesh (forced whenever spheres are present), or keep a Curve object with bevel |

## How it works

**Tree mode.** Starting from the origin with the trunk along $+Z$, the tree is built by a recursion that, at each tip of a branch of level $\ell$, emits a segment and then spawns $a$ = *arity* children. A child's direction is the parent direction $\mathbf{d}$ tilted by the branch angle $\phi$ and rotated to azimuth $\psi$ about $\mathbf{d}$:

$$\mathbf{d}_{\text{child}} = \sin\phi\,(\cos\psi\,\mathbf{u} + \sin\psi\,\mathbf{v}) + \cos\phi\,\mathbf{d},$$

where $(\mathbf{u},\mathbf{v})$ is a deterministic orthonormal frame perpendicular to $\mathbf{d}$. The $k$-th child of generation $g$ uses azimuth

$$\psi_k = \frac{2\pi k}{a} + g\,\tau,$$

with $\tau$ the azimuth twist — a golden-angle-ish $\tau$ makes successive crowns non-aligning. Lengths and radii scale geometrically: a level-$\ell$ segment has length $\propto \text{ratio}^{\ell}$ and per-point radii $\text{ratio}^{\ell} \to \text{ratio}^{\ell+1}$, so a round bevel at the trunk radius yields continuously tapering limbs. A full tree has

$$\sum_{g=0}^{\text{depth}} a^{g} = \frac{a^{\,\text{depth}+1}-1}{a-1}$$

segments; Depth is clamped down (with a warning) to stay under the 8000-segment budget. Optional leaves place a subdivided **icosphere** (unit icosahedron subdivided once → 42 verts, 80 faces) at each of the $a^{\text{depth}}$ final tips.

**Mobile mode.** The same $n$-ary structure hangs downward. Each hang point drops a vertical string of length $\text{trunk\_len}\cdot\text{ratio}^{g}$ to a horizontal `arity`-armed spreader (arm length equal to the drop), generation $g$'s arms rotated by $g\,\tau$ in azimuth (default $\tau = 60°$ while the twist slider is untouched, so alternating spreaders do not align). Final tips drop a half-length string to a sphere weight. Bars taper slightly along their length.

**Fit and output.** All segment endpoints (and sphere extents) are bounding-boxed; the model is centered and uniformly scaled so the largest extent is 2 m, with the same `fit_scale` applied to the bevel so limb thickness stays in proportion. The curve is one POLY spline per segment carrying per-point radii; whenever spheres are present (Mobile, or Tree with leaves) the bevelled curve is converted to a mesh and the icosphere geometry merged in.

## References

- Henry Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press (2016) — the Mahler–Segerman ternary tree mobile, fig. 7-2. <https://www.3dprintmath.com/>
- Sabetta Matsumoto, Henry Segerman, Laura Taalman (Mahler & Segerman, "Ternary tree mobile") — the printed sculpture the Mobile mode reproduces.
