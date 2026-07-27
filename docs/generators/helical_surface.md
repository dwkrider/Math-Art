# Helical Surface

![Helical Surface](../images/helical_surface.png)

## Overview
Three classic helical / spiral parametric surfaces: the hyperbolic helicoid (a twisted band), a conical seashell shell, and the corkscrew "twisted sphere." Each is built by a pure-Python parametrization; seams where a parameter wraps are welded by index so the meshes are seamless, and the seashell's apex is emitted as a single vertex with a triangle fan.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Surface | Hyperbolic Helicoid | Which surface: Hyperbolic Helicoid, Seashell, or Corkscrew. |
| Torsion | 2.5 | Torsion $\tau$ of the hyperbolic helicoid (turns per unit of the spine parameter). Range 0-20. |
| Extent | 4.0 | Half-range of both parameters of the hyperbolic helicoid. Range 0.5-10. |
| Whorls | 2 | Number of turns of the shell spiral. Range 1-8. |
| Tube Aspect | 1.0 | Vertical stretch of the tube cross-section ($a$). Range 0-2. |
| Height | 4.0 | Total rise of the shell from mouth to apex ($b$). Range 0-10. |
| Opening | 0.37 | Offset radius that keeps the shell mouth open ($c$). Range 0-2. |
| Sphere Radius | 0.5 | Radius $a$ of the twisted sphere (corkscrew). Range 0.01-10. |
| Twist Rise | 0.3 | Axial rise $b$ per radian of twist (corkscrew). Range 0-5. |
| Resolution | 96 | Segments along each parametric direction. Range 8-512. |
| Smooth Shading | On | Shade the mesh smooth. |
| Thickness | 0.0 | Solidify modifier thickness (0 = raw surface). Range 0-1. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |


## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/helical_surface__HYPERBOLIC_HELICOID.png" width="200"><br><sub>Hyperbolic Helicoid</sub></td>
<td align="center"><img src="../images/variants/helical_surface__SEASHELL.png" width="200"><br><sub>Seashell</sub></td>
<td align="center"><img src="../images/variants/helical_surface__CORKSCREW.png" width="200"><br><sub>Corkscrew</sub></td>
</tr>
</table>

## How it works

Each surface is a direct parametric mesh over a $u,v$ grid.

**Hyperbolic helicoid** -- the helicoid's hyperbolic cousin, a twisted band whose points all share the denominator $1+\cosh u\cosh v$:
$$x = \frac{\sinh v\,\cos(\tau u)}{1+\cosh u\cosh v},\quad y = \frac{\sinh v\,\sin(\tau u)}{1+\cosh u\cosh v},\quad z = \frac{\cosh v\,\sinh u}{1+\cosh u\cosh v},$$
over $u,v\in[-\text{extent},\text{extent}]$. The torsion $\tau$ sets how fast the band winds around its axis. No parameter wraps, so there is no seam; the shared denominator is $\ge 2$ everywhere, so every point is finite.

**Seashell** -- a conical spiral shell: a circular tube whose radius shrinks linearly to an apex while it winds $n$ whorls around the axis. With taper $= 1 - v/2\pi$,
$$r(u,v) = \text{taper}\,(1+\cos u) + c,\quad x = r\cos(nv),\quad y = r\sin(nv),\quad z = \frac{b\,v}{2\pi} + a\sin(u)\,\text{taper},$$
for $u,v\in[0,2\pi]$, where $n$ = whorls, $a$ = tube aspect, $b$ = height, $c$ = opening. The tube direction $u$ wraps and is welded by index; at $v=2\pi$ the whole tube collapses to the apex $(c\cos 2\pi n,\, c\sin 2\pi n,\, b)$, emitted once and fanned with triangles.

**Corkscrew** -- the "twisted sphere": a sphere stretched along a diameter and sheared upward while it turns, so each meridian circle climbs by $b$ per radian:
$$x = a\cos u\cos v,\quad y = a\sin u\cos v,\quad z = a\sin v + b\,u,$$
for $u,v\in[0,2\pi]$, with $a$ the sphere radius and $b$ the axial rise per radian of twist. Each meridian circle (fixed $u$, $v$ wrapping) is welded by index; $u$ is left open because the twist offsets its two ends by $2\pi b$ in $z$.

In all three, the mesh is centered and fit within a $2\times\text{scale}$ cube, then optionally given a Solidify shell.

## References

- J. Meier, minimal-surface and parametric-surface gallery, https://www.3d-meier.de/ (hyperbolic helicoid and related parametrizations).
- A. Gray, E. Abbena, S. Salamon, *Modern Differential Geometry of Curves and Surfaces with Mathematica*, 3rd ed., Chapman & Hall/CRC, 2006 (seashell and helicoidal parametrizations).
- Wolfram MathWorld, "Seashell" and "Helicoid," https://mathworld.wolfram.com/
