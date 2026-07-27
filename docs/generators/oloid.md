# Oloid & Ruled Surfaces

![Oloid & Ruled Surfaces](../images/oloid.png)

## Overview
A small family of surfaces built from two perpendicular circles. The **oloid** (Paul Schatz, 1929) is the convex hull of two unit circles in perpendicular planes, each passing through the other's centre; its lateral surface is a *developable* (flat-unrollable) ruled surface, meshed here from the exact ruling of Dirnboeck & Stachel. Also included: the **two-circle roller**, the non-developable **anti-oloid**, and Kit Wallace's straight-ruled **circle strips** and true one-edged **ruled Mobius strip**.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Shape | Oloid | Oloid, Two-Circle Roller, Anti-Oloid, Ruled Circle Strip, or Ruled Mobius Strip. |
| Segments | 96 | Number of rulings / samples around the construction. Range 12-512. |
| Separation | 1.0 | Distance between the circle centres (Ruled Circle Strip only). Range 0.0-3.0. |
| Inclination | 0.0 | Tilt of the second circle in degrees (Ruled Circle Strip only). Range -90 to 90. |
| Phase | 0.0 | Ruling offset around the second circle as a fraction of a turn (Ruled Circle Strip and Anti-Oloid). Range -0.5 to 0.5. |
| Scale | 1.0 | Output is centred and fitted so its largest extent is $2\,\text{m}\times$ Scale. Range 0.01-100. |


## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/oloid__OLOID.png" width="200"><br><sub>Oloid</sub></td>
<td align="center"><img src="../images/variants/oloid__ROLLER.png" width="200"><br><sub>Two-Circle Roller</sub></td>
<td align="center"><img src="../images/variants/oloid__ANTIOLOID.png" width="200"><br><sub>Anti-Oloid</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/oloid__RULED.png" width="200"><br><sub>Ruled Circle Strip</sub></td>
<td align="center"><img src="../images/variants/oloid__MOBIUS.png" width="200"><br><sub>Ruled Mobius Strip</sub></td>
</tr>
</table>

## How it works

**The oloid (developable ruling).** Take two unit circles: one in the $z=0$ plane centred at $(0,-\tfrac12,0)$, one in the $x=0$ plane centred at $(0,\tfrac12,0)$, so each passes through the other's centre. Dirnboeck & Stachel give the boundary of the developable lateral surface as a pair of curves joined by straight rulings,
$$A(t) = (\sin t,\; -\tfrac12 - \cos t,\; 0),\qquad t\in[-\tfrac{2\pi}{3},\tfrac{2\pi}{3}],$$
$$B(t) = \Big(0,\; \tfrac12 - \frac{\cos t}{1+\cos t},\; \pm\frac{\sqrt{1+2\cos t}}{1+\cos t}\Big).$$
Every ruling segment $A(t)\,B(t)$ has the **constant length $\sqrt3$** -- the hallmark of the oloid, verified to machine precision in the self-test. Because the rulings have constant length and turn without stretching, the surface is *developable*: it can be unrolled flat, and physically an oloid rolls developing its whole surface across the ground. Two mirror strips ($\pm$ branch) are welded along the two circular arcs into a watertight solid.

**Two-circle roller.** The same two circles but with centres $\sqrt2$ apart; the convex hull is taken in Blender. At that separation the solid's centre of mass stays at constant height as it rolls -- a "wobbler".

**Anti-oloid.** The ruled band between the *same* two circles as the oloid, but with rulings connecting points travelling around the **full** circles in step (a half-turn phase offset), sweeping through the interior. Parametrized by
$$A(t)=(\sin t,\,-\tfrac12-\cos t,\,0),\quad B(t)=(0,\,\tfrac12+\cos t,\,\sin t),$$
with $B$ shifted by $\pi+2\pi\,\text{phase}$. Unlike the Mobius strip it is two-sided, and unlike the oloid it is **not** developable.

**Ruled circle strips.** Straight rulings from circle 1 in the $xy$-plane to circle 2 in the $xz$-plane, offset by `separation` along $x$, optionally inclined about $z$ by `incline`, connecting sample $i$ to sample $i+\text{phase}$. This is Kit Wallace's experimental family; varying separation, inclination and phase sweeps between disc-like, saddle-like and twisted ruled surfaces.

**Ruled Mobius strip.** Wallace's genuine one-edged ruled surface: a single double-loop edge curve
$$f(x) = \big(r\cos 4\pi x,\; r\sin 4\pi x,\; 0.2\cos 2\pi x\big),\quad r = 1 - 0.15\sin(2\pi x + 30^\circ),$$
with rulings joining $f(x)$ to $f(x+\tfrac12)$. The result has Euler characteristic $\chi=0$ and a **single** boundary loop, confirmed by the standalone boundary-walk test.

## References

- P. Schatz, *Rhythmusforschung und Technik* (the oloid, 1929); see also the Schatz "invertible cube" work.
- H. Dirnboeck and H. Stachel, "The Development of the Oloid," *Journal for Geometry and Graphics* 1 (1997), pp. 105-118 (the exact ruling and the $\sqrt3$ constant-length property).
- Kit Wallace, "ruled Mobius strip" experiments, https://kitwallace.tumblr.com/post/85762927079 (the ruled circle strips and one-edged Mobius surface).
- Anti-oloid after the Matter Collection piece of the same name.
