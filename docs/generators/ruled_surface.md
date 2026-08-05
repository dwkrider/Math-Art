# Ruled Surfaces

![Ruled Surfaces](../images/ruled_surface.png)

## Overview

The family of surfaces swept by a **single straight line** — a *ruling* —
moving through space. Every mode is built from the general ruled-surface
recipe $S(u,v) = b(u) + v\,d(u)$ (a directrix $b$ plus $v$ times a ruling
direction $d$), from the right-conoid variant $S(u,v)=(v\cos u,\,v\sin u,\,h(u))$,
or from a bilinear patch spanning four skew points. Because the generator
is always a straight segment, each surface can be rendered three ways: as
the filled **surface**, as its literal **rulings turned into rods** (the
look of string / stick sculptures), or as **bare curves** (a wireframe of
the rulings only).

The star of the family is the **stick hyperboloid** (George Hart, Bridges
2023): a hyperboloid of one sheet strung as straight rods between two
coaxial circles, the top circle twisted relative to the bottom. The twist
alone sets the waist — $a = R\cos(\tfrac{\text{twist}}{2})$ — from a
cylinder (0°) down to a pinched double cone (180°). It is *doubly ruled*:
two opposite-handed families of rods cross to weave the same surface. This
complements the ruled surfaces already in the extension (the developable
[oloid](oloid.md) strips, the [modular screen](../generators/) hypar
saddles, the [helical surfaces](helical_surface.md)) with the missing
straight-ruling core.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Surface | Stick Hyperboloid | Which mode: Stick Hyperboloid, Compound Helical Cone, Spiral Ruled, Conoid, Tangent Developable, Helicoid, Twisted Strip, or Hyperbolic Paraboloid. |
| Conoid | Plucker Cylindroid | (Conoid mode) Plucker's cylindroid, n-Fold, Wallis conical edge, or Whitney umbrella. |
| Radius | 1.0 | Circle / envelope / helix radius. Range 0.01–20. |
| Half Height | 1.0 | Half the axial height of the stick hyperboloid (rings sit at $\pm$ this). Range 0.05–20. |
| Twist | 120° | Rotation of the hyperboloid's top ring; the waist radius is $R\cos(\text{twist}/2)$. Range 0–179. |
| Ruling Family | Both | Which hyperboloid ruling family to draw as rods/curves; Both gives the crossing string sculpture. |
| Height / Flutes / Flute Depth / Spiral Twist / Taper / Orbit Amount / Orbit Turns | — | Compound helical cone: envelope height, number of helical flutes and their depth, the flute winding rate, the envelope profile exponent, and the amplitude/turns of the optional second (planetary) helix. |
| Spiral Tightness / Ruling Slope / Turns / Symmetry / Rosette Amount | — | Spiral ruled surface: log-spiral growth $k$ (0 = circle), ruling steepness, number of turns, and the fold count / amount of an optional rosette base curve. |
| Amplitude / Folds / Wallis a / Wallis b | — | Conoid profile controls (amplitude of $h(u)$; fold count for n-Fold; $a,b$ for the Wallis conical edge). |
| Pitch / Turns / Edge Gap / Inner Radius / Ruling Slope | — | Helix-based modes: axial rise per radian, number of turns, the inner offset from a tangent developable's cuspidal edge, and the helicoid's inner radius and ruling tilt. |
| Strip Half-Width / Half Twists | — | Twisted strip: strip half-width and number of half-twists (odd = one-sided Mobius band). |
| a / b / Saddle Height / From 4 Corner Points | — | Hyperbolic paraboloid as $z=c((x/a)^2-(y/b)^2)$, or the bilinear patch through four user-set corner points P00, P10, P01, P11. |
| Output | Surface | Surface, Rulings as Rods, or Bare Curves (straight-ruled modes only). |
| Rod Count / Rod Radius | 48 / 0.02 | Number of rulings to draw, and rod thickness (Rods output). |
| Resolution U / V | 120 / 20 | Samples along the directrix and along the ruling. |
| Smooth Shading / Thickness / Scale | — | Smooth normals; Solidify thickness (surface output); output fitted to a $2\,\text{m}\times$ Scale cube at the origin. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/ruled_surface__HYPERBOLOID.png" width="200"><br><sub>Stick Hyperboloid</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__HYPERBOLOID_RODS.png" width="200"><br><sub>Stick Hyperboloid (Rulings)</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__HELICAL_CONE.png" width="200"><br><sub>Compound Helical Cone</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ruled_surface__SPIRAL.png" width="200"><br><sub>Spiral Ruled</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__SPIRAL_ROSETTE.png" width="200"><br><sub>Spiral Ruled (Rosette)</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__PLUCKER.png" width="200"><br><sub>Plucker Cylindroid</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ruled_surface__WALLIS.png" width="200"><br><sub>Wallis Conical Edge</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__WHITNEY.png" width="200"><br><sub>Whitney Umbrella</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__TANGENT_DEV.png" width="200"><br><sub>Tangent Developable</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ruled_surface__HELICOID.png" width="200"><br><sub>Helicoid</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__TWIST_STRIP.png" width="200"><br><sub>Twisted Strip (Mobius)</sub></td>
<td align="center"><img src="../images/variants/ruled_surface__HYPAR.png" width="200"><br><sub>Hyperbolic Paraboloid</sub></td>
</tr>
</table>

## How it works

**Stick hyperboloid.** Two coaxial circles of radius $R$ sit at $z=\pm H$;
the top circle is rotated by the twist angle $2\varphi$. A straight ruling
joins the bottom point at angle $\theta-\varphi$ to the top point at
$\theta+\varphi$, so linearly interpolating gives the surface point
$S(\theta,v) = (1-v)\,b(\theta) + v\,t(\theta)$. Every such rod lies on the
hyperboloid of one sheet whose waist radius is $a = R\cos\varphi$: at
$\varphi=0$ the rods are vertical (a cylinder), and as $\varphi\to90°$ the
waist pinches to a point (a double cone). The two families — top rotated
$+2\varphi$ versus $-2\varphi$ — are the left- and right-handed rulings of
the **doubly ruled** surface; drawn together as rods they reproduce Hart's
crossing string sculptures. The self-test verifies $a = R\cos(\text{twist}/2)$
to $2\times10^{-3}$.

**Compound helical cone.** A conical envelope $R(t)=R_0(1-t)^{\text{taper}}$
is fluted by $N$ ridges $r(\theta,t)=R(t)\,(1+\varepsilon\cos(N\theta+2\pi\,\text{twist}\,t))$
that wind helically as $t$ climbs; the flute depth $\varepsilon$ fades to
zero at the apex so the ridges pass from ornament (rounded flutes) at the
base to arrises (sharp edges) near the top, exactly as Jannasch & Macnab
describe the Solomonic column. An optional planetary helix sweeps the whole
section's centre around a second, larger spiral. The apex is emitted as a
single welded point.

**Spiral ruled surface (Farris).** The hyperboloid's generating circle is
replaced by a base curve $b(u)=e^{ku}\,(\cos u + A\cos(pu),\,\sin u + A\sin(pu))$
— a logarithmic spiral for $p=1,A=0$, or an $n$-fold rosette otherwise —
and swept with a tangent-plus-vertical ruling $S = b(u) + v\,b'(u)$ with
$z = \text{slope}\cdot v$. Tightness $k=0$ recovers a hyperboloid.

**Conoids.** Right conoids take the form $S(u,v)=(v\cos u,\,v\sin u,\,h(u))$:
the height profile $h(u)$ is the whole design freedom. $h=\text{amp}\,\sin 2u$
is **Plücker's cylindroid** (two leaves); $h=\text{amp}\,\sin(nu)$ its
$n$-leaved generalization; $h=\text{amp}\sqrt{a^2-b^2\cos^2u}$ the **Wallis
conical edge**. The **Whitney umbrella** $S=(uv,\,u,\,v^2)$ is the classic
pinch-point ruled surface (a self-intersection line along the axis above
its singular point).

**Tangent developable.** $T(u,v)=c(u)+v\,c'(u)$ of the circular helix
$c(u)=(R\cos u, R\sin u, \text{pitch}\cdot u)$. Because the ruling is the
curve's own tangent, the surface is **developable** — it unrolls flat
without distortion ($K\equiv0$). The helix itself ($v=0$) is the cuspidal
**edge of regression** where the surface folds, so $v$ starts at a small
positive Edge Gap. A shared *developability predicate*
$\det[\,b'(u),\,d(u),\,d'(u)\,]\stackrel{?}{=}0$ distinguishes such
flat-unrollable surfaces from the doubly-curved ones (the self-test
confirms it vanishes for the tangent developable and equals the pitch for
the right helicoid).

**Helicoid.** A radial ruling $S(u,v)=(v\cos u,\,v\sin u,\,\text{pitch}\cdot u+\text{slope}\cdot v)$
screws up the axis; slope 0 is the **right helicoid** (a minimal surface),
non-zero slope tilts the rulings into an **oblique helicoid**.

**Twisted strip.** $b(u)=(R\cos u,R\sin u,0)$ with ruling
$d(u)=(\cos\tfrac{nu}{2}\cos u,\,\cos\tfrac{nu}{2}\sin u,\,\sin\tfrac{nu}{2})$
gives an $n$-half-twist band. Odd $n$ is a one-sided **Möbius band**; even
$n$ an orientable twisted annulus.

**Hyperbolic paraboloid.** The doubly-ruled saddle, either as the graph
$z=c((x/a)^2-(y/b)^2)$ or as the **bilinear (Coons) patch** through four
skew corner points, $S(s,t)=(1-s)(1-t)P_{00}+s(1-t)P_{10}+(1-s)t\,P_{01}+st\,P_{11}$
— whose $s$- and $t$-edges are straight, so the surface spanning *any* four
points in general position is a hypar.

## References

- G. W. Hart, "Curved, yet Straight: Stick Hyperboloids," *Bridges 2023
  Conference Proceedings*, pp. 251–258. <https://archive.bridgesmathart.org/2023/bridges2023-251.html>
- E. Jannasch and J. Macnab, "The Compound Helical Cone as Kinematic
  Trace," *Bridges 2023 Conference Proceedings*, pp. 15–22.
- F. A. Farris, "Spiral Ruled Surfaces," *Bridges 2022 Conference
  Proceedings*, pp. 289–292; and F. A. Farris, *Creating Symmetry*
  (Princeton Univ. Press, 2015).
- J. Plücker, *On a New Geometry of Space* (1865); H. Whitney, "The general
  type of singularity of a set of $2n-1$ smooth functions of $n$ variables,"
  *Duke Math. J.* 10 (1943).
- S. A. Coons, "Surfaces for Computer-Aided Design of Space Forms," MIT
  Project MAC TR-41 (1967) — the bilinear patch.
- Classical background: M. do Carmo, *Differential Geometry of Curves and
  Surfaces* (1976); A. Gray, *Modern Differential Geometry of Curves and
  Surfaces* (1997); D. Struik, *Lectures on Classical Differential
  Geometry* (1950).
