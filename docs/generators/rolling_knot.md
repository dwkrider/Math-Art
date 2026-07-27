# Rolling Knot

![Rolling Knot](../images/rolling_knot.png)

## Overview

This generator builds a smooth-rolling $(p,2)$ torus knot, after Brodeur, Vidulis, Dandy & Pauly's *Smooth-Rolling Knots* (Bridges 2025), building on Morton's tritangentless trefoils, the rolling analysis of Eget, Lucas & Taalman, and the smooth-rolling Two-Disk Rollers of Engelhardt & Ucke. Morton's $(p,2)$ knots roll on a plane but their centre of mass bobs up and down; following the paper, the knot is stretched and its two exterior lobes are pinned onto the two orthogonal ellipses of a Two-Disk Roller, with the interior morphed smoothly inside the roller's convex hull, so the centre of mass stays at constant height while rolling. Crucially, this implementation also optimizes for the **actual tube thickness**: where strands fuse it measures the resulting centre-of-mass shift and rebalances the knot so the physical solid still rolls smoothly.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Lobes p | 3 | Odd (p,2) torus knot parameter (3 = trefoil; even values are rounded down) |
| Shape a | 0.5 | Morton shape parameter |
| Mode | Smooth-Rolling | Smooth-Rolling (morph onto the two-disk roller, constant-height rolling), Stretched (only the optimal vertical stretch), or Morton (the raw Morton knot) |
| Tube Radius | 0.05 | Tube radius, in units of the (unit-scale) curve |
| Optimize For Thickness | On | Rebalance the interior so the THICK solid's centre of mass (fused strands weigh once) sits at the rolling centre |
| Min Gap | 0.02 | Minimum surface-to-surface gap between strands of the thick knot (0 = strands may touch and fuse) |
| Smoothness | 600.0 | Interior fairing weight: higher gives wider, calmer interior curves at the cost of knot-shape fidelity |
| Curve Samples | 512 | Points along the centreline |
| Tube Sides | 16 | Cross-section sides |
| Smooth Shading | On | Shade the tube smooth |
| Scale | 1.0 | Overall size (fit within a 2 × Scale cube) |

## How it works

**Morton's knot.** A $(p,2)$ torus knot in Morton's tritangentless form (trefoil for $p=3$) is, with $b=\sqrt{1-a^2}$, $c=a/(1+b)$ and $\text{den}=1-b\sin 2t$,

$$K(t) = \frac{c}{\text{den}}\Big(a\cos pt,\;\; a\sin pt,\;\; z_s\, b\cos 2t\Big),\qquad t\in[0,2\pi),$$

where $z_s$ is a vertical stretch. Such a knot rolls on a plane, but its centre of mass rises and falls — the roughness is measured by

$$\rho = \frac{\max h_{\text{com}} - \min h_{\text{com}}}{\text{mean } h_{\text{com}}},$$

the range-to-average ratio of the centre-of-mass height over the rolling motion ($\rho=0$ is perfect smooth rolling). The resting states are the physical **bitangent** directions where the two exterior lobes support equally; `rho_eval` finds them by Newton-refining Fibonacci-sampled directions onto the equal-support band.

**Two-Disk Roller.** A smooth-rolling Two-Disk Roller (Engelhardt & Ucke) is two identical ellipses in perpendicular planes sharing a centre line, half-axis $\alpha$ along the line and $\beta$ across, centres a distance $\gamma$ apart with

$$\gamma^2 = 4\alpha^2 - 2\beta^2,$$

the condition under which its centre of mass stays at constant height as it rolls. The generator first identifies the knot's two exterior **lobes** (maximal runs of samples appearing on the convex hull, found via support maxima over Fibonacci-distributed directions), then fits a TDR $(z_s,\alpha,\beta)$ to them by Nelder–Mead, minimizing the mean squared distance from each lobe to its ellipse (each ellipse's closest point found by Newton iteration).

**Morphing.** With the roller fixed, the grown lobes are pinned onto the roller's contact arcs by arc length — only the **core** of each lobe (spanning the ~240° the roller actually touches while rolling) is clamped to the ellipse; hinge zones at both ends stay free with a tapering data weight so the curve peels off smoothly instead of kinking. The interior is then solved as a weighted least-squares fair: minimize a Laplacian bending energy $\lVert D K\rVert^2$ (with $D$ the discrete second-difference operator) plus a data term pulling toward the targets, subject to the pinned points as hard constraints, and clamped to stay inside the roller's support hull.

**Thickness optimization.** The paper treats the knot as an ideal curve. For a physical tube of radius $r$ two exact facts keep the ideal solution robust: the support function of the thickened solid is the curve's support plus $r$ in every direction (Minkowski sum with a ball), and the centre of mass of a closed circular tube equals the curve's arc-length centroid exactly (the curvature term integrates to zero around a closed loop). What breaks smooth rolling is **strand fusion**: where tubes overlap the merged volume is counted once, shifting the solid's centre of mass. `com_thick` measures that shift exactly — the arc centroid minus a fine local grid correction restricted to the near-contact zones (distant strands closer than $2r$) — and a balance loop iteratively translates and re-solves the interior to drive the thick solid's centre of mass back to the rolling centre. An optional **minimum-clearance** rope relaxation pushes distant strands apart to keep a surface-to-surface gap, alternating repulsion of violating pairs with light fairing. The operator reports the achieved $\rho$ for both the ideal curve and the thick solid, plus the overlap fraction and the minimum strand gap.

**Mesh.** The optimized centreline is swept into a closed circular tube using parallel-transport frames with the seam holonomy untwisted, then fit to a $2\times\text{Scale}$ cube.

## References

- Yingying Ren / Julian Brodeur, Michele Vidulis, Aleksia Dandy, Mark Pauly, *Smooth-Rolling Knots*, Bridges 2025 Conference Proceedings: <https://archive.bridgesmathart.org/>
- Hugh R. Morton, *Trefoil knots without tritangent planes*, Bull. London Math. Soc. 23 (1991), 78–80.
- Chloe Eget, Steven Lucas, Laura Taalman, *Rolling Knots*, Bridges 2020 Conference Proceedings: <https://archive.bridgesmathart.org/2020/bridges2020-367.html>
- Christian Ucke, Hans-Joachim Schlichting (and A. Engelhardt), *Two-disk roller* / *Zwei-Scheiben-Roller* — smooth-rolling two-disk roller: <https://www.ucke.de/christian/physik/ftp/lectures/Wobbler_2009.pdf>
