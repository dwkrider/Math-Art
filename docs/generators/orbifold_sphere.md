# Orbifold Symmetry Sphere

![Orbifold Symmetry Sphere](../images/orbifold_sphere.png)

## Overview
After the "comma symmetry spheres" of Henry Segerman's *Visualizing Mathematics with 3D Printing*: a sphere decorated with a comma-shaped motif in raised (or carved) relief, replicated under one of the **14 types of spherical symmetry**, selected by Conway orbifold signature. The comma is deliberately chiral and placed at a generic point -- off every mirror plane and rotation axis -- so the pattern of commas and their mirror images makes the symmetry type visible at a glance.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Signature | 532 - icosahedral (I) | Spherical symmetry type by Conway orbifold signature: the seven infinite families $nn$ (Cn), $\ast nn$ (Cnv), $n\ast$ (Cnh), $n\times$ (S2n), $22n$ (Dn), $2\ast n$ (Dnd), $\ast22n$ (Dnh); and the seven "oddities" 332 (T), $\ast$332 (Td), 3$\ast$2 (Th), 432 (O), $\ast$432 (Oh), 532 (I), $\ast$532 (Ih). |
| n | 6 | Order of the main axis for the seven infinite families (ignored by the seven oddities). Range 1-32. |
| Sphere Radius | 1.0 | Radius of the sphere. Range 0.01-100. |
| Sphere Resolution | 48 | Longitudinal segments of the sphere shell. Range 8-256. |
| Shell Thickness | 0.03 | Wall thickness of the hollow sphere (0 = solid ball); negative relief deeper than the wall cuts comma-shaped holes right through the shell. Range 0.0-10. |
| Motif Size | 0.25 | Overall size of the comma motif. Range 0.01-10. |
| Motif Relief | -0.05 | Negative carves the commas into the sphere (boolean difference); positive raises them above the surface. Range -10 to 10. |
| Color Reflected Copies | Off | Second material on the mirror-image commas (orientation-reversing copies). |

## How it works

**Orbifold symmetry groups.** The finite subgroups of $O(3)$ -- the symmetry types of patterns on a sphere -- number exactly 14, named by Conway's orbifold notation. Seven are infinite families parametrized by a main-axis order $n$:

| Signature | Schoenflies | Description | Order |
| --- | --- | --- | --- |
| $nn$ | $C_n$ | cyclic | $n$ |
| $\ast nn$ | $C_{nv}$ | pyramidal (vertical mirrors) | $2n$ |
| $n\ast$ | $C_{nh}$ | rotation + horizontal mirror | $2n$ |
| $n\times$ | $S_{2n}$ | rotoreflection | $2n$ |
| $22n$ | $D_n$ | dihedral | $2n$ |
| $2\ast n$ | $D_{nd}$ | antiprismatic | $4n$ |
| $\ast22n$ | $D_{nh}$ | prismatic | $4n$ |

and seven are the "oddities" tied to the Platonic solids: $332$ ($T$, 12), $\ast332$ ($T_d$, 24), $3\ast2$ ($T_h$, 24), $432$ ($O$, 24), $\ast432$ ($O_h$, 48), $532$ ($I$, 60), $\ast532$ ($I_h$, 120).

**Group construction by closure.** Each group is built explicitly as a set of $3\times3$ orthogonal matrices. A short list of generators -- rotations $R_{\hat a}(\vartheta)$ about an axis (Rodrigues' formula), plus the mirrors $\sigma_h,\sigma_v$, the swap, and the central inversion $-I$ -- is closed under multiplication with a rounding-based dedupe. For example the pure rotation groups use $C_z=R_{\hat z}(2\pi/n)$, a horizontal 2-fold $R_{\hat x}(\pi)$, and the tetrahedral/octahedral/icosahedral seeds $R_{(1,1,1)}(2\pi/3)$, $R_{\hat z}(\pi/2)$, $R_{(0,1,\varphi)}(2\pi/5)$; adding $-I$ or a diagonal mirror produces the reflection extensions. The rotoreflection $S_{2n}=R_{\hat z}(\pi/n)\,\sigma_h$. The realized order is checked against the theoretical order and a warning is issued on any float-drift mismatch.

**The comma motif.** `comma_outline` builds a closed 2D polygon of a comma / teardrop: a circular head of radius $r_0$ blended into a tapering tail swept along a circular-arc centreline of radius $\rho$ through a curl angle of $150^\circ$, with tapering half-width $w(t)=r_0(1-t)^{1.4}$. The tail curls clockwise, making the motif chiral so its mirror images are distinguishable. The polygon is triangulated, placed tangent to the sphere at the generic point (spherical coordinates $\theta=63^\circ$, $\phi=21^\circ$, chosen off every axis and mirror of all 14 groups), and projected radially onto the sphere.

**Relief and replication.** For raised relief ($h>0$) the motif is extruded between radii $R$ and $R+h$ into a closed bump and unioned with the sphere shell. For carved relief ($h<0$) the commas become cutter solids from below the carve depth up through the surface, boolean-differenced out of a hollow shell (relief deeper than the wall pierces right through). Each group element $M$ maps the base motif by $\mathbf p\mapsto M\mathbf p$; elements with $\det M<0$ are orientation-reversing, so their copies are the mirror commas (windings flipped, optionally given a second material). In the larger reflective groups adjacent commas overlap, so the cutter is first resolved into one clean solid by a self-union before the difference, and an EXACT/FLOAT solver cascade accepts the first watertight non-empty result.

## References

- H. Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 (chapter 1, figs 1.19-1.29 -- the comma symmetry spheres).
- J. H. Conway, H. Burgiel, C. Goodman-Strauss, *The Symmetries of Things*, A K Peters, 2008 (orbifold notation and the enumeration of spherical symmetry groups).
- Point groups $C_n, C_{nv}, C_{nh}, S_{2n}, D_n, D_{nd}, D_{nh}, T, T_d, T_h, O, O_h, I, I_h$: standard group theory / crystallography.
