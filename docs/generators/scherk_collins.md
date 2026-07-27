# Scherk-Collins Sculpture

![Scherk-Collins Sculpture](../images/scherk_collins.png)

## Overview

The Scherk-Collins generator builds the saddle-chain toroid sculptures that Carlo H. Séquin designed with the sculptor Brent Collins in his *Sculpture Generator I* program — pieces such as *Hyperbolic Hexagon II*, the *Heptoroid*, and the *Minimal Trefoil*. Each sculpture is a stack of "storeys" cut from the singly-periodic Scherk minimal surface, optionally twisted and warped into a closed ring. It is a faithful re-implementation of the original program's geometry engine, and reads and writes the original's spec/demo files.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Preset | Default | Program defaults, or one of the classic sculptures (Hyperbolic Hexagon, Minimal Trefoil, Monkey-Saddle Trefoil, Heptoroid, straight Scherk Tower) copied into the sliders |
| Branches | 2 | Order of the saddles (number of branches; 2 = classic Scherk saddle, 3 = monkey saddle) |
| Storeys | 2 | Number of hole/saddle storeys stacked in the chain |
| Storey Height | 1.5 | Height of one storey (1.5 ≈ the natural aspect ratio) |
| Flange Width | 1.5 | Width of the flanges; holes break open below ≈ 0.88 |
| Thickness | 0.15 | Thickness of the vanes (0 = surface only, no rims) |
| Rim Bulge | 1.5 | Amount of bulge on the rounded rim beads |
| Twist | 0.0 | Overall axial twist along the chain (degrees) |
| Azimuth | 0.0 | Fixed turn of the profile around the tower axis (degrees) |
| Warp | 0.0 | Bend of the tower toward an arch/toroid (degrees; 360 = closed ring, > 360 wraps multiply) |
| Detail | 5 | Grid detail (tessellation density) |
| Stretch X | 1.0 | Affine stretch along X (the "Totem" trick) |
| Stretch Y | 1.0 | Affine stretch along Y |
| Stretch Z | 1.0 | Affine stretch along Z |
| Overall Scale | 1.0 | Uniform output scale (Blender-side extra) |
| NURBS Output | Off | Emit a compact NURBS surface (mid-surface only; thickness and rims do not apply) |
| NURBS Detail | 2 | Control-point density used for NURBS output (stays smooth at low values) |




## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/scherk_collins__HEX.png" width="200"><br><sub>Hyperbolic Hexagon</sub></td>
<td align="center"><img src="../images/variants/scherk_collins__TREFOIL.png" width="200"><br><sub>Minimal Trefoil</sub></td>
<td align="center"><img src="../images/variants/scherk_collins__MONKEY.png" width="200"><br><sub>Monkey-Saddle Trefoil</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/scherk_collins__HEPTOROID.png" width="200"><br><sub>Heptoroid</sub></td>
<td align="center"><img src="../images/variants/scherk_collins__TOWER.png" width="200"><br><sub>Scherk Tower</sub></td>
</tr>
</table>

## How it works

The tower template is the exact singly-periodic **Scherk minimal surface**

$$\sin z = \sinh x \,\sinh y.$$

Each height level $c = \sin z$ contributes a cross-section curve, parametrized as

$$x(\sigma) = \operatorname{asinh}\!\big(\sqrt{c}\; e^{\,\sigma\Lambda}\big), \qquad
y(\sigma) = \operatorname{asinh}\!\big(\sqrt{c}\; e^{-\sigma\Lambda}\big), \qquad
\Lambda = \ln\!\frac{\sinh W}{\sqrt{c}}, \qquad \sigma \in [-1, 1],$$

where $W$ is the flange (truncation) half-width. This choice of $\Lambda$ lands the curve ends exactly on the truncation planes $x = W$ and $y = W$. When $c > \sinh^2 W$ the level lies inside an opened hole and is omitted, which is why holes break open once $W$ drops below $\operatorname{asinh}(1) \approx 0.881$ (the original slider bottoms out near 0.7).

Rows within a storey are **cosine-clustered** in $z$ so extra resolution concentrates near the saddle levels where curvature is highest; the saddle levels themselves ($c = 0$) degenerate the cross-section to the two vane segments $[0, W]$.

**Higher-order saddles** (branches $b$) are produced by compressing the $90°$ wedge of the base curve into a $180°/b$ wedge, using the angular map $f = \operatorname{atan2}(y, x)/(\pi/2)$ to place each point within that wedge. Consecutive storeys are rotated by $180°/b$, so a warped ring joins smoothly precisely when

$$\big(\text{twist} + \text{storeys}\cdot 180/b\big) \bmod (360/b) = 0.$$

The N-panel reports whether the current twist closes the ring. After the template cross-section point is placed, an affine pipeline applies azimuth + linear twist (rotation growing with normalized height $z_n$), then the warp bends the tower around a circle of radius $R = H_{\text{out}}/\text{warp}$ that preserves arc length, and finally the per-axis stretches and overall scale.

**Thickness** is applied as a two-sided normal offset of $\pm t/2$ about the mid-surface. Boundary (cut/flange) edges get semicircular **rim tubes**, elongated by the rim-bulge factor, welded on via a shared canonical-normal registry so neighbouring grids join watertight — including correctly through Möbius (single-sided) configurations, where grid windings flip across the closure seam. The finished mesh is welded (`remove_doubles`) and its normals recalculated, yielding a watertight, manifold solid suitable for 3D printing.

**NURBS output** instead emits one clamped NURBS patch per half-wedge of each storey (split at the wedge bisector so adjacent patches share identical edge control sequences), giving the smooth mid-surface with far fewer control points.

Because Séquin's C source is not public, the tessellation layout and the exact flange/bulge profiles here are re-derivations rather than byte-identical copies; the shapes match the published sculptures and all 20 demo files shipped with the original program.

## References

- C. H. Séquin, *Virtual Prototyping of Scherk-Collins Saddle Rings*, Leonardo, Vol. 30, No. 2, pp. 89–96, 1997.
- C. H. Séquin, H. Meshkin, L. Downs, *Interactive Generation of Scherk-Collins Sculptures*, Proc. I3D '97.
- C. H. Séquin, *15 Years of Scherk-Collins Saddle Chains*, Technical Report UCB/EECS-2010-41.
- C. H. Séquin, *Sculpture Generator I*, UC Berkeley — <https://people.eecs.berkeley.edu/~sequin/SCULPTS/scherk.html>
- Parameter semantics validated against the `demo*.txt` files shipped with the original *Sculpture Generator* Windows program.
- Paul Nylander, *Scherk-Collins surface notes* (twist/warp transformations) — <https://nylander.wordpress.com/2009/02/10/scherk-collins-surface/>
