# Spherical Harmonic

![Spherical Harmonic](../images/spherical_harmonic.png)

## Overview

The spherical harmonics $Y_\ell^m$ are the eigenfunctions of the Laplace-Beltrami operator on the sphere — the angular part of every separable solution of Laplace's equation, and so the shape language shared by gravitational and electrostatic multipoles, the vibration modes of a spherical shell, and atomic orbitals. This generator draws them as radial surfaces $r = f(\theta,\varphi)$ in three forms (a smoothly deformed sphere, the classic lobed balloon, and a sign-split lobe picture), plus a fourth: the eight-integer trigonometric family popularised by Paul Bourke, which is not a spherical harmonic but a rich sculptural relative of one.

For the radial part that turns an angular lobe picture into an actual wavefunction, see the sibling [Atomic & Molecular Orbital](orbital.md) generator.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Form | Offset Sphere | Offset ($r = r_0 + a\,Y$), Absolute ($r=\|Y\|$), Signed (absolute, with the lobes separated by the sign of $Y$), or the Bourke family. |
| Degree l | 3 | Degree $\ell$ of the harmonic: the surface has $\ell$ nodal circles in total. Range 0-12. |
| Order m | 2 | Order $m$; clamped to $\|m\| \le \ell$. Negative $m$ selects the $\sin$ partner, positive the $\cos$ partner. Range -12 to 12. |
| Base Radius | 1.0 | Offset form: the undeformed sphere radius $r_0$. Range 0.05-10. |
| Amplitude | 0.6 | Offset form: how strongly $Y_\ell^m$ deforms the sphere. Range -5 to 5. |
| Nodal Gap | 0.02 | Absolute/signed forms: radius added at the nodal circles so they do not pinch to a point. Range 0-0.5. |
| Split Lobes | Off | Signed form: separate the lobes into loose parts instead of joining them at the nodes. |
| Bourke Set | Bourke 4-1-4-1 | Which of the six shipped integer tuples to use, or Custom for the m0..m7 fields. |
| m0 … m7 | 4,1,4,1,4,1,4,1 | The eight exponents/frequencies of the Bourke form. Range 0-8 each. |
| Absolute Radius | Off | Bourke form: use $\|r\|$, so the surface cannot fold through the origin. |
| Resolution (polar) | 128 | Grid rows in $\theta$. Range 8-1024. |
| Resolution (azimuth) | 256 | Grid columns in $\varphi$. Range 8-1024. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |
| Thickness | 0.0 | If > 0, add a Solidify modifier with this thickness. Range 0-1. |
| Smooth Shading | On | Shade the mesh smooth. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/spherical_harmonic__OFFSET.png" width="200"><br><sub>Offset Sphere</sub></td>
<td align="center"><img src="../images/variants/spherical_harmonic__ABS.png" width="200"><br><sub>Absolute Lobes</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/spherical_harmonic__SIGNED.png" width="200"><br><sub>Signed Lobes</sub></td>
<td align="center"><img src="../images/variants/spherical_harmonic__BOURKE.png" width="200"><br><sub>Bourke Family</sub></td>
</tr>
</table>

## How it works

### The real harmonics

Blender ships no scipy, so the associated Legendre functions are built from the classical three-term recurrences, evaluated on whole numpy arrays at once:

$$P_m^m(x) = (-1)^m (2m-1)!!\,(1-x^2)^{m/2}, \qquad P_{m+1}^m(x) = x(2m+1)P_m^m(x),$$

$$(\ell-m)\,P_\ell^m(x) = x(2\ell-1)P_{\ell-1}^m(x) - (\ell+m-1)P_{\ell-2}^m(x).$$

The $(-1)^m$ is the **Condon-Shortley phase**. The real harmonics are then

$$Y_\ell^0 = \sqrt{\tfrac{2\ell+1}{4\pi}}\,P_\ell(\cos\theta),$$

$$Y_\ell^{m} = \sqrt{2}\,N_{\ell m}\,P_\ell^{m}(\cos\theta)\cos(m\varphi), \qquad Y_\ell^{-m} = \sqrt{2}\,N_{\ell m}\,P_\ell^{m}(\cos\theta)\sin(m\varphi)$$

for $m>0$, with $N_{\ell m} = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}$ computed through `lgamma` so that large $\ell$ cannot overflow. This is the basis in which $(\ell,m) = (1,0)$ is $p_z$, $(1,1)$ is $p_x$ and $(1,-1)$ is $p_y$.

$Y_\ell^m$ has $\ell - |m|$ nodal circles of latitude and $2|m|$ nodal meridians; the module's self-test counts both and checks orthonormality on the sphere to $10^{-6}$.

### The four forms

- **Offset** — $r = r_0 + a\,Y_\ell^m$. While $a < r_0 / \max|Y_\ell^m|$ the radius stays positive, so the surface is star-shaped, hence embedded and printable. This is the safe default.
- **Absolute** — $r = |Y_\ell^m| + \varepsilon$. The classic balloon picture. At a nodal circle $Y$ vanishes and the surface pinches; the Nodal Gap $\varepsilon$ keeps that pinch from collapsing to coincident vertices, but the surface is still an immersion rather than an embedding, and Solidify on it self-intersects.
- **Signed** — as Absolute, with each face assigned to one of two material slots by the sign of $Y$ at its centre. With Split Lobes on, the band of faces straddling a nodal line is deleted and the lobes fall apart into separate loose parts.
- **Bourke** — the eight-integer family

$$r = \sin^{m_1}(m_0\varphi) + \cos^{m_3}(m_2\varphi) + \sin^{m_5}(m_4\theta) + \cos^{m_7}(m_6\theta)$$

with $\varphi$ the polar angle and $\theta$ the azimuth, following Bourke's own convention (his renders are $y$-up; here the result is remapped to Blender's $z$-up). The exponents are forced to Python integers before use: numpy raising a negative float base to a *float* power yields NaN, which is the classic way to get an empty mesh out of this family.

### Meshing

The radius is evaluated on a $(\theta,\varphi)$ grid, the two poles are collapsed to single vertices and the azimuth seam is glued **by index, never by coordinate proximity** — so the result is a closed sphere-topology mesh with $\chi = 2$ and no boundary edges. Finally the mesh is centred on its bounding box and fitted to a 2 m cube before Scale is applied.

## References

- P. S. Laplace, "Théorie des attractions des sphéroïdes et de la figure des planètes," *Mémoires de l'Académie royale des Sciences*, 1785.
- A.-M. Legendre, "Recherches sur l'attraction des sphéroïdes homogènes," *Mémoires de Mathématique et de Physique*, 1785.
- E. U. Condon and G. H. Shortley, *The Theory of Atomic Spectra*, Cambridge University Press, 1935 (the real form, normalisation and the phase convention).
- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Dover, 1965, chapter 8 (the associated Legendre recurrences used here).
- P. Bourke, "Spherical Harmonics," February 1990, http://paulbourke.net/geometry/sphericalh/ — the source of the eight-parameter form. The parameter sets shipped as presets here are project-chosen, not his.
