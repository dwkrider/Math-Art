# Prime Knot

![Prime Knot](../images/prime_knot.png)

## Overview

This generator builds all 249 prime knots with up to 10 crossings (the Rolfsen table) from the minimum braid words of Thomas Gittings. Each braid closure is laid out around a circle — strands at radii by braid level, crossings as over/under bumps — and then relaxed by curve smoothing plus self-repulsion into a rounded, KnotPlot-like presentation. Every braid word in the table is verified programmatically: the closure permutation must be a single cycle and the Alexander polynomial (via the reduced Burau representation) must match Gittings' published value. Output styles follow the classic Torus Knot add-on: Bézier / Poly / NURBS curve or a swept tube mesh.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Knot | 3.1 | Which knot to build: Custom braid, or any Rolfsen knot up to 10 crossings (each listed with its crossing number and minimum braid word) |
| Braid Word | AAA | Letters a..z are braid generators, A..Z their inverses (used for Custom) |
| Curve Samples | 240 | Number of points along the final curve |
| Relax Iterations | 150 | Smoothing + self-repulsion rounds (0 shows the raw braid closure) |
| Strand Clearance | 0.35 | Self-repulsion distance while relaxing |
| Mirror | Off | Mirror image of the knot |
| Output | Bezier Curve | Bezier (auto-smoothed), Poly, NURBS, or Mesh Tube (swept tube mesh) |
| Tube Radius | 0.08 | Curve bevel depth / tube radius |
| Bevel Resolution | 6 | Bevel profile resolution (curve output) |
| Tube Sides | 12 | Cross-section sides (mesh output) |
| Scale | 1.0 | Overall size (max extent = 2.0 × Scale) |

## How it works

**Braid words.** A braid on $n$ strands is a word in the Artin generators $\sigma_1,\dots,\sigma_{n-1}$ and their inverses; Gittings' letter notation uses `a..z` for $\sigma_i$ and `A..Z` for $\sigma_i^{-1}$. The **closure** of a braid joins each strand's top to its bottom, producing a knot or link. The closure is a knot (one component) iff the induced permutation — the product of transpositions $(i\,\;i{+}1)$ for each generator $\sigma_i^{\pm}$ — is a single $n$-cycle; the generator rejects words whose closure is a link.

**Verification.** For every table entry the module recomputes the Alexander polynomial from the **reduced Burau representation** $\rho$ of the braid group: each generator maps to a matrix over $\mathbb{Z}[t,t^{-1}]$, the word's image $R$ is formed, and

$$\Delta(t) \;\propto\; \frac{\det\!\big(R - t^{k} I\big)}{1 + t + \cdots + t^{\,n-1}},$$

where $k$ counts inverse generators. Evaluated at $t=10$ this must equal Gittings' published invariant; the full table passes.

**Braid-closure embedding.** The closure is drawn around a circle of $L = $ word-length angular steps. Strand at level $\ell$ sits at radius $r = \text{ring} + \big(\ell - \tfrac{n-1}{2}\big)\,\text{spacing}$, i.e. the $n$ strands are concentric rings. At each letter $\sigma_i^{\pm}$ the two strands at levels $i,\,i{+}1$ swap, passing through a mid-angle point lifted to $z = \pm\text{bump}$ (sign from the generator's sign) so one goes over and the other under; the remaining strands advance flat. Following the closure permutation from strand 0 concatenates the per-strand paths into one closed polyline.

**Relaxation.** The raw closure is resampled to evenly spaced points, then relaxed for `iters` rounds. Each round applies Laplacian curve **smoothing**

$$P_i \leftarrow P_i + s\left(\tfrac{P_{i-1}+P_{i+1}}{2} - P_i\right)$$

and **self-repulsion**: any two points closer than `repel` that are not near-neighbours along the strand push apart with strength $(\text{repel}-d)/d$. After each round the curve is recentered and its mean radius renormalized, so for conservative settings the strand cannot pass through itself — the knot type is preserved while it rounds into shape. The final curve is resampled to `samples` points and fit to a 2 m cube.

**Tube mesh.** Mesh output sweeps a circular cross-section along the curve using **parallel-transport frames**; the frame's closure holonomy (residual twist after going once around) is measured and distributed uniformly, $\text{corr}_i = -\text{ang}\cdot i/m$, so the seam matches with no visible twist discontinuity.

## References

- Thomas A. Gittings, *Minimum braids: a complete invariant of knots and links*, arXiv:math/0401051, Table 1: <https://arxiv.org/abs/math/0401051>
- Dale Rolfsen, *Knots and Links*, Publish or Perish (1976) — the prime knot table.
- Joan S. Birman, *Braids, Links, and Mapping Class Groups*, Annals of Math. Studies 82, Princeton (1974) — braid groups, Burau representation, closures.
- KnotPlot (Robert Scharein) — reference for the relaxed, rounded knot presentation: <https://knotplot.com/>
