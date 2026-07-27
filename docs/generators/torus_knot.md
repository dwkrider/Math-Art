# Torus Knot

![Torus Knot](../images/torus_knot.png)

## Overview

This generator builds the $(p,q)$ torus knots and links as rope curves lying on a torus of configurable major/minor radius. The curve winds $p$ times around the torus axis while looping $q$ times through the hole; when $\gcd(p,q)=d>1$ the result is a torus **link** of $d$ components (e.g. $(2,2)$ is the Hopf link, $(2,4)$ Solomon's link, $(3,3)$ three fibres of the Hopf fibration), emitted as $d$ splines or merged tube meshes with optional per-component colouring. Output styles match the Prime Knots generator: Bézier / Poly / NURBS curve or swept tube mesh.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| p | 2 | Windings around the torus axis |
| q | 3 | Loops through the torus hole; gcd(p, q) > 1 gives a link with that many components |
| Major Radius | 0.7 | Torus centre-line radius |
| Minor Radius | 0.3 | Torus tube radius the knot lies on |
| Samples | 192 | Points per component |
| Output | Bezier Curve | Bezier (auto-smoothed), Poly, NURBS, or Mesh Tube (swept tube mesh) |
| Tube Radius | 0.08 | Curve bevel depth / tube radius |
| Bevel Resolution | 6 | Bevel profile resolution (curve output) |
| Tube Sides | 12 | Cross-section sides (mesh output) |
| Color Components | On | One material with a distinct color per link component (links only) |
| Scale | 1.0 | Overall size multiplier |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/torus_knot__2_3.png" width="200"><br><sub>Trefoil (2, 3)</sub></td>
<td align="center"><img src="../images/variants/torus_knot__2_5.png" width="200"><br><sub>Cinquefoil (2, 5)</sub></td>
<td align="center"><img src="../images/variants/torus_knot__2_7.png" width="200"><br><sub>(2, 7)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/torus_knot__3_4.png" width="200"><br><sub>(3, 4)</sub></td>
<td align="center"><img src="../images/variants/torus_knot__3_5.png" width="200"><br><sub>(3, 5)</sub></td>
<td align="center"><img src="../images/variants/torus_knot__5_2.png" width="200"><br><sub>(5, 2)</sub></td>
</tr>
</table>

## How it works

A $(p,q)$ torus knot lies on the surface of a torus of major radius $R$ and minor radius $r$, winding $p$ times the long way (around the axis) and $q$ times the short way (through the hole). It is parametrized by

$$\big((R + r\cos qt)\cos pt,\;\; (R + r\cos qt)\sin pt,\;\; r\sin qt\big),\qquad t\in[0,2\pi).$$

Every sampled point satisfies $\big(\sqrt{x^2+y^2}-R\big)^2 + z^2 = r^2$, i.e. it lies exactly on the torus.

**Links.** When $d=\gcd(p,q)>1$ the closed curve breaks into $d$ disjoint components, each a reduced $(p/d,\,q/d)$ torus knot. On the flat torus the components are parallel lines of slope $q'/p'$ (with $p'=p/d,\ q'=q/d$); spacing them by $2\pi/p$ in the tube angle gives distinct disjoint curves. Component $k$ therefore uses a phase offset $\varphi_k = 2\pi k/p$:

$$r_k(t) = R + r\cos(q' t + \varphi_k),\quad
P_k(t) = \big(r_k\cos p't,\; r_k\sin p't,\; r\sin(q' t + \varphi_k)\big).$$

(Spacing by $2\pi/d$ instead would re-trace the same curve when $p/d>1$.) Thus $(2,2)$ gives the Hopf link, $(2,4)$ Solomon's link $4^2_1$, $(3,3)$ three Hopf fibres, and so on.

**Colouring.** For links, one Principled-BSDF material per component is created with hue $k/d$ (HSV, saturation 0.72, value 0.9) and assigned by spline (or per-face `material_index` for meshes). Because a spline's `material_index` is clamped to the current material count at assignment time, the indices are set only after all materials exist.

**Mesh tube.** Mesh output reuses the Prime Knots parallel-transport tube sweep, with the seam holonomy distributed so the cross-section closes without a twist artifact; per-component face material indices carry the colouring.

## References

- *Torus knot* — parametrization and $\gcd(p,q)$ component count: <https://en.wikipedia.org/wiki/Torus_knot>
- Regular Polytopes / *Add Torus Knot* — the classic Blender Torus Knot Plus add-on whose output styles this generator follows.
