# Links & Connected Sums

![Links & Connected Sums](../images/link.png)

## Overview

This generator builds multi-component links and knot connected sums as rope curves, after the knot chapter of Henry Segerman's *Visualizing Mathematics with 3D Printing*. Presets include the Hopf link, Solomon's link, the Whitehead link, Borromean rings (two constructions), $(p,q)$ torus links, interlocked chains, and the connected sum of two prime knots (square vs. granny via mirroring). It reuses the braid-word, resampling and rope-relaxation machinery of the Prime Knots generator: multi-component braid closures are split into one loop per component and relaxed jointly with cross-component repulsion, so linked components stay clear of each other and cannot pass through (which would unlink them).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Preset | Hopf Link | Hopf Link (two circles through each other's centers), Solomon's Link (the (2,4) torus link 4²₁, linking number 2), Whitehead Link (closure of (s1 s2⁻¹)² s1, rope-relaxed; linking number 0), Borromean Rings (three mutually perpendicular ellipses; pairwise unlinked yet inseparable), Borromean Crest (three equal circles at 120° woven over/under), Torus Link (p,q) (gcd(p,q) components on a common torus), Chain (a row of interlocked rings), Connect Sum (two prime knots spliced into one rope) |
| p | 2 | Windings around the torus axis (Torus preset) |
| q | 4 | Windings around the torus tube (Torus preset) |
| Rings | 5 | Rings in the chain (Chain preset) |
| Knot A | 3.1 | First summand (Connect Sum preset) |
| Knot B | 3.1 | Second summand (Connect Sum preset) |
| Mirror Second Knot | Off | Mirror the second summand (e.g. square knot instead of granny knot) |
| Axis Ratio a/b | 1.618 | Ellipse semi-axis ratio; larger gives more clearance between the rings (Borromean Rings) |
| Weave Depth | 0.35 | Height of the over/under weave bumps (Borromean Crest) |
| Samples / Component | 160 | Points per component |
| Relax Iterations | 100 | Smoothing + repulsion rounds for the rope-relaxed presets (0 shows the raw construction) |
| Strand Clearance | 0.35 | Repulsion distance while relaxing |
| Color Components | On | One material with a distinct color per link component |
| Output | Bezier Curve | Bezier (auto-smoothed), Poly, NURBS, or Mesh Tube |
| Rope Radius | 0.08 | Curve bevel depth / tube radius |
| Bevel Resolution | 6 | Bevel profile resolution (curve output) |
| Tube Sides | 12 | Cross-section sides (mesh output) |
| Scale | 1.0 | Overall size (combined bbox ≈ 2 m × Scale) |


## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/link__HOPF.png" width="200"><br><sub>Hopf Link</sub></td>
<td align="center"><img src="../images/variants/link__SOLOMON.png" width="200"><br><sub>Solomon's Link</sub></td>
<td align="center"><img src="../images/variants/link__WHITEHEAD.png" width="200"><br><sub>Whitehead Link</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/link__BORROMEAN.png" width="200"><br><sub>Borromean Rings</sub></td>
<td align="center"><img src="../images/variants/link__BORROMEAN_CREST.png" width="200"><br><sub>Borromean Crest</sub></td>
<td align="center"><img src="../images/variants/link__TORUS.png" width="200"><br><sub>Torus Link</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/link__CHAIN.png" width="200"><br><sub>Chain</sub></td>
<td align="center"><img src="../images/variants/link__CONNECT_SUM.png" width="200"><br><sub>Connect Sum</sub></td>
</tr>
</table>

## How it works

Each preset produces a list of closed component curves; all components share one bounding box, are centered on the origin, and scaled so the combined extent is ~2 m.

**Direct constructions.** The **Hopf link** is two unit circles, each passing through the other's center (one in the $xy$-plane, one in a perpendicular plane). The **$(p,q)$ torus link** places $\gcd(p,q)$ components on a common torus, component $k$ phase-offset by $2\pi k/p$ in the tube angle:

$$P_k(t) = \big((R+r\cos\varphi)\cos\theta,\;(R+r\cos\varphi)\sin\theta,\; r\sin\varphi\big),\quad
\theta = \tfrac{p}{g}t,\;\; \varphi = \tfrac{q}{g}t + \tfrac{2\pi k}{p}.$$

**Solomon's link** is the $(2,4)$ case. A **chain** is $n$ interlocked circles in a row, alternating between the $xy$- and $xz$-planes like physical links.

**Borromean rings** come in two forms. The *ellipse* construction uses three mutually perpendicular $a\times b$ ellipses (in the $xy$, $yz$, $zx$ planes) with $b=a/\text{ratio}$: each pair is unlinked, yet the three are collectively inseparable for any $a/b>1$, with minimum pairwise clearance $\sim a-b$. The *crest* construction uses three equal circles centered at $120°$ offsets, woven by the cyclic rule "circle $i$ passes over circle $i{+}1$": at each planar crossing angle a smooth $\pm z$ Gaussian bump $z \mathrel{+}= s\,e^{-(\delta/\sigma)^2}$ lifts one strand over the other, so every pair crosses over/over (pairwise unlinked) while the whole is the Borromean rings.

**Braid closures.** The **Whitehead link** is the closure of the minimum braid $(\sigma_1\sigma_2^{-1})^2\sigma_1$ — the unique connected reduced alternating 5-crossing 2-component link with linking number 0. (For comparison, the figure-eight knot is $(\sigma_1\sigma_2^{-1})^2$ and Borromean rings $(\sigma_1\sigma_2^{-1})^3$.) A multi-component version of the braid-closure embedding splits the closure into one loop per component, which are then rope-relaxed jointly.

**Joint relaxation.** `relax_link` extends the Prime Knots relaxation to several components: each component is Laplacian-smoothed, and repulsion acts between **all** strand pairs — both self and cross-component — with a global renormalization. Because repulsion never lets strands cross, linked components stay linked; being linked, they also cannot drift apart. A Gauss linking-integral routine (midpoint rule) verifies the linking numbers of the results.

**Connected sum.** Two prime knots are each built and relaxed by the Prime Knots engine, cut open at their outermost ($\max x$) point, placed side by side with the openings facing (the second rotated $180°$), spliced by two short straight bridges, and rope-relaxed into a single closed rope. Mirroring the second summand distinguishes the **square** knot ($3_1 \# \overline{3_1}$) from the **granny** knot ($3_1 \# 3_1$).

**Output.** Each component is one cyclic spline (or a merged tube mesh). Distinct per-component Principled-BSDF materials are created with hue $k/d$; because a spline's `material_index` is clamped to the material count when set, the indices are reassigned after the materials exist.

## References

- Henry Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press (2016) — knot/link chapter, figs 5-7, 5-10, 5-12. <https://www.3dprintmath.com/>
- *Whitehead link*, *Borromean rings*, *Solomon's knot* — link tables and constructions: <https://en.wikipedia.org/wiki/Whitehead_link>, <https://en.wikipedia.org/wiki/Borromean_rings>
- The braid-closure, resampling and rope-relaxation machinery is shared with the Prime Knots generator (`math_art/prime_knot_generator.py`); see its references for Gittings' minimum braids and the Burau/Alexander verification.
