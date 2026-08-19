# Classic Minimal Surface

![Classic Minimal Surface](../images/parametric_minimal.png)

## Overview

This generator produces the classic gallery of parametric minimal surfaces — Enneper (of any order), the catenoid, helicoid, Henneberg, Catalan, Bour, Richmond, Scherk's doubly-periodic graph, plus the Weierstrass-representation surfaces Costa (genus 1) and Chen-Gackstatter, the Jorge-Meeks $k$-noid, and the catenoid–helicoid associate family. Each is meshed directly from its closed-form parametrization and fit to a 2 m cube, and can be emitted as a dense polygon mesh or as a single compact NURBS patch with correct periodic closure.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Surface | Enneper | Which minimal surface to build (Enneper, Catenoid, Helicoid, Henneberg, Catalan, Bour, Richmond, Scherk doubly-periodic, Costa, Chen-Gackstatter, Jorge-Meeks k-noid, Catenoid-Helicoid associate) |
| Output | Mesh | Dense polygon mesh, or a compact NURBS surface patch (control grid = Control Points U × V) |
| Resolution U | 48 | Mesh grid resolution along U |
| Resolution V | 48 | Mesh grid resolution along V |
| Control Points U | 24 | NURBS control-grid size in U |
| Control Points V | 24 | NURBS control-grid size in V |
| Order / Count | 1 | Enneper order; helicoid half-turns; Jorge-Meeks end count $n$ ($\ge 3$); ignored for the rest |
| Domain Radius | 1.2 | Extent of the parameter domain (for the k-noid, how close the disk reaches its ends; for Costa/Chen-Gackstatter, scales the end-rim disk) |
| Associate Angle | 0.0 | Bonnet associate family angle: 0 = catenoid, $\pi/2$ = helicoid (Catenoid-Helicoid surface only) |
| Scale | 1.0 | Multiplier on the normalized size (1.0 = a 2 m cube centered on the origin) |

## Variants

Renders of each selectable option:

### Classical

<table>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__ENNEPER.png" width="200"><br><sub>Enneper</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__CATENOID.png" width="200"><br><sub>Catenoid</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__HELICOID.png" width="200"><br><sub>Helicoid</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__HENNEBERG.png" width="200"><br><sub>Henneberg</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__CATALAN.png" width="200"><br><sub>Catalan</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__BOUR.png" width="200"><br><sub>Bour</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__RICHMOND.png" width="200"><br><sub>Richmond</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__CATHEL.png" width="200"><br><sub>Catenoid-Helicoid (associate)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__RICHMOND_K.png" width="200"><br><sub>Richmond (generalized, g = z^k)</sub></td>
</tr>
</table>

### Tori (genus 1)

<table>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__COSTA.png" width="200"><br><sub>Costa (genus 1)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__CHEN_GACK.png" width="200"><br><sub>Chen-Gackstatter</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__COSTA_HM.png" width="200"><br><sub>Costa-Hoffman-Meeks (genus k)</sub></td>
</tr>
</table>

### Spheres (genus 0)

<table>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__KNOID.png" width="200"><br><sub>Jorge-Meeks k-noid</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__DOUBLE_ENNEPER.png" width="200"><br><sub>Double Enneper</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__ENNK.png" width="200"><br><sub>Enneper-ended k-noid</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__M3_PYR.png" width="200"><br><sub>Pyramidal k-noid</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__M3_PRISM.png" width="200"><br><sub>Prismatic k-noid</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__M3_BIPYR.png" width="200"><br><sub>Bipyramidal k-noid</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__M3_ANTI5.png" width="200"><br><sub>Antiprismatic k-noid (nn=5)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__M3_LOPEZ.png" width="200"><br><sub>Lopez sphere (2 ends of index 2)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__M3_CATENN.png" width="200"><br><sub>Sphere: catenoid + Enneper end</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__M3_FRIEM.png" width="200"><br><sub>Finite Riemann (plane + 2 catenoids)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__ENNEPER_NCAT.png" width="200"><br><sub>Enneper with n Catenoids</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__SYMM_FRIEM.png" width="200"><br><sub>Symmetrized Finite Riemann (2m catenoids)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__SYMM_DBLENN.png" width="200"><br><sub>Symmetrized Double Enneper</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__KNOID_ENN_ENDS.png" width="200"><br><sub>k-Noid with Enneper Ends</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__ANTIPRISM_KNOID.png" width="200"><br><sub>Antiprismatic k-noid (full family)</sub></td>
</tr>
</table>

### Non-Orientable

<table>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__MEEKS_MOBIUS.png" width="200"><br><sub>Meeks Mobius Strip</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__HENNEBERG_RP2.png" width="200"><br><sub>Henneberg (one-sided)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__KUSNER_RP2.png" width="200"><br><sub>Kusner Projective Plane (p planar ends)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__LOPEZ_KLEIN.png" width="200"><br><sub>Lopez Minimal Klein Bottle</sub></td>
</tr>
</table>

### Higher Genus

<table>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__CG_HIGHER.png" width="200"><br><sub>Chen-Gackstatter (higher genus)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__SYMM_CG.png" width="200"><br><sub>Symmetrized Chen-Gackstatter (k-fold, genus k-1)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__SYMM_CG_G2N.png" width="200"><br><sub>Symmetrized Chen-Gackstatter (2-level, genus 2(k-1))</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__SYMM_CG_G3K.png" width="200"><br><sub>Symmetrized Chen-Gackstatter (3-level, genus 3(k-1))</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__CATENOID_ENNEPER.png" width="200"><br><sub>Catenoid-Enneper (higher genus)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__COSTA_WOHLGEMUTH.png" width="200"><br><sub>Costa-Wohlgemuth (4 ends)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__WOHLGEMUTH_G3.png" width="200"><br><sub>Wohlgemuth Second Surface (genus 3)</sub></td>
</tr>
</table>

### Bjorling (curve-seeded)

<table>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__BJ_CYCLOID.png" width="200"><br><sub>Bjorling: Cycloid (Catalan)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__BJ_CIRCLE.png" width="200"><br><sub>Bjorling: Twisted Band (Mobius)</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__BJ_HELIX.png" width="200"><br><sub>Bjorling: Helix Ribbon</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/parametric_minimal__BJ_TREFOIL.png" width="200"><br><sub>Bjorling: Trefoil Ribbon</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__BJ_ARCH_SPIRAL.png" width="200"><br><sub>Bjorling: Archimedean Spiral</sub></td>
<td align="center"><img src="../images/variants/parametric_minimal__BJ_LOG_SPIRAL.png" width="200"><br><sub>Bjorling: Logarithmic Spiral</sub></td>
</tr>
</table>

## How it works

A minimal surface has zero mean curvature everywhere; each surface here is realized by evaluating its standard parametrization on a $(n_u, n_v)$ grid, then centering and scaling the result so its largest extent is 2 units.

**Elementary surfaces** are sampled from their explicit maps. For example the **Enneper surface** of order $n$ uses the complex chart $z = u\,e^{iv}$ with

$$x = \operatorname{Re}\!\Big(z - \tfrac{z^{2n+1}}{2n+1}\Big),\quad
y = -\operatorname{Im}\!\Big(z + \tfrac{z^{2n+1}}{2n+1}\Big),\quad
w = \tfrac{2}{n+1}\operatorname{Re}\!\big(z^{n+1}\big),$$

the **catenoid** as $x = \cosh v \cos u,\; y = \cosh v \sin u,\; z = v$, and the **helicoid** as $x = v\cos u,\; y = v\sin u,\; z \propto u$. Scherk's doubly-periodic surface is the graph $w = \ln(\cos u / \cos v)$. The **Catenoid–Helicoid associate** interpolates the Bonnet family with angle $\theta$: $\theta = 0$ gives the catenoid, $\theta = \pi/2$ the helicoid, and every intermediate value a complete minimal surface isometric to both.

**Weierstrass-representation surfaces** (Costa, Chen-Gackstatter) live on the square (lemniscatic) torus with half-periods $\omega_1 = \tfrac12,\ \omega_3 = \tfrac{i}{2}$, $\tau = i$. The module includes a small Weierstrass engine computing $\wp$, $\wp'$ and $\zeta$ from the Jacobi theta functions ($\theta_1$ $q$-series, nome $q = e^{i\pi\tau}$; about a dozen terms reach $10^{-15}$). Costa's surface (genus 1, three ends) uses the Gray/Nylander closed form built from $\zeta(z)$, $\zeta(z-\tfrac12)$, $\zeta(z-\tfrac{i}{2})$ and $\ln\big|(\wp - e_1)/(\wp + e_1)\big|$; Chen-Gackstatter (genus 1, a single order-3 Enneper end, total curvature $-8\pi$) is built from $\wp$, $\wp'$, $\zeta$ with $g_2 = 4e_1^2$. These are meshed **periodically** on the torus with small disks punctured out around the ends, so the only mesh boundaries are clean circular end rims.

The **Jorge-Meeks $k$-noid** (genus 0 with $n\ge 3$ catenoid ends) has no closed form; it is built by radial Weierstrass integration of $g = z^{n-1}$, $dh = z^{n-1}/(z^n-1)^2\,dz$ outward from the pole-free base point $z=0$, spanning both sides of $|z|=1$, with disks around the ends removed.

Surfaces whose ends run to infinity are trimmed: a parameter-space validity mask gives clean circular rims where available, otherwise faces past a distance percentile (and faces bridging an excluded end) are dropped and the resulting staircase rim is smoothed by a boundary-only Laplacian pass. The body is fit to the 2 m cube using an inlier reference so runaway ends don't shrink it. NURBS output drops the duplicated periodic endpoint and builds a single patch with cyclic-U/V flags matching the surface's true periodicity.

## References

- Jürgen Meier, *Minimalflächen* — parametric surface gallery, 3d-meier.de tut25: <http://www.3d-meier.de/tut25/Seite0.html>
- DLMF (NIST Digital Library of Mathematical Functions), §23.6 (Weierstrass elliptic functions in terms of theta functions) and §20.2 (theta-function $q$-series): <https://dlmf.nist.gov/>
- Alfred Gray / Paul Nylander — closed form for Costa's minimal surface.
- L. P. Jorge, W. H. Meeks III, *The topology of complete minimal surfaces of finite total Gaussian curvature*, Topology 22 (1983) — the $k$-noid / $n$-noid family.
- C. C. Chen, F. Gackstatter, *Elliptische und hyperelliptische Funktionen und vollständige Minimalflächen vom Enneperschen Typ*, Math. Ann. 259 (1982).
