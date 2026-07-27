# Seifert Surface

> ⚠️ **Experimental.** This generator is a work in progress — its
> options and output may change in future versions.

![Seifert Surface](../images/seifert.png)

## Overview

The Seifert generator builds the orientable spanning surface (Seifert surface) of any knot or link, entered as a **braid word**. Seifert's algorithm applied to a braid closure gives one disk per strand and one half-twisted band per crossing; the add-on stacks the disks like a wedding cake (largest at the bottom) and joins consecutive rims with ±180° twisted ribbons — the SeifertView presentation. The surface boundary *is* the knot (optionally emitted as a bevelled tube curve), and the operator reports the strand, crossing, link-component and **genus** counts computed straight from the braid.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Preset | Trefoil (3_1) | Custom braid, one of the classic examples (trefoil, figure-8, cinquefoil, granny, square, Hopf, Solomon, Borromean), or a $(p,q)$ torus knot |
| Braid Word | aaa | Braid word: letters ($a = \sigma_1$, $A = \sigma_1^{-1}$) or integers (`1 -2 1`) |
| Torus p | 3 | $p$ of the $(p,q)$ torus-knot braid (Torus preset) |
| Torus q | 4 | $q$ of the $(p,q)$ torus-knot braid (Torus preset) |
| Base Radius | 2.0 | Radius of the bottom (largest) disk |
| Taper | 0.5 | How much smaller the top disk is than the bottom |
| Disk Spacing | 0.75 | Vertical spacing between stacked disks |
| Band Width | 0.5 | Band width as a fraction of the free rim arc |
| Resolution | 96 | Angular samples around the disks |
| Disk Rings | 6 | Concentric rings meshing each disk |
| Band Rows | 16 | Rows along each twisted band |
| Shape Relax Rounds | 0 | Relax the whole shape: the knot boundary evolves as an elastic curve (constant length, with self-repulsion so strands cannot cross) while the surface re-relaxes each round (needs the Minimal Surface Toolkit add-on) |
| Relax Iterations | 0 | Smooth the surface with the Plateau solver, keeping the knot boundary pinned (needs the Minimal Surface Toolkit add-on) |
| Add Knot Curve | On | Also create a bevelled curve along the knot/link (the surface boundary) |
| Knot Tube Radius | 0.04 | Bevel radius of the knot curve |





## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/seifert__TREFOIL.png" width="200"><br><sub>Trefoil (3_1)</sub></td>
<td align="center"><img src="../images/variants/seifert__FIGURE8.png" width="200"><br><sub>Figure-8 (4_1)</sub></td>
<td align="center"><img src="../images/variants/seifert__CINQUEFOIL.png" width="200"><br><sub>Cinquefoil (5_1)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/seifert__GRANNY.png" width="200"><br><sub>Granny</sub></td>
<td align="center"><img src="../images/variants/seifert__SQUARE.png" width="200"><br><sub>Square</sub></td>
<td align="center"><img src="../images/variants/seifert__HOPF.png" width="200"><br><sub>Hopf link</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/seifert__SOLOMON.png" width="200"><br><sub>Solomon's link</sub></td>
<td align="center"><img src="../images/variants/seifert__BORROMEAN.png" width="200"><br><sub>Borromean rings</sub></td>
<td align="center"><img src="../images/variants/seifert__TORUS.png" width="200"><br><sub>Torus knot</sub></td>
</tr>
</table>

## How it works

A **braid word** on $n$ strands is a sequence of generators $\sigma_i$ / $\sigma_i^{-1}$ (adjacent-strand exchanges), given as letters (`a` = $\sigma_1$, `A` = $\sigma_1^{-1}$) or signed integers. Its **closure** is a knot or link, and **Seifert's algorithm** on that closure produces one Seifert circle per strand and one twisted band per crossing.

**Combinatorics.** For a word of length $L$ on $n$ strands the generator computes the closure permutation, counts its cycles $\mu$ (the number of boundary/link components), and reports the genus of the (connected) surface from the Euler characteristic $\chi = n - L$:

$$\chi = n - L, \qquad g = \frac{2 - \mu - \chi}{2}.$$

The headless tests confirm the generated mesh's own $V - E + F$ equals $n - L$ for every preset.

**Geometry.** The $n$ disks are stacked as a wedding cake: disk $i$ has radius $r_i = R\,(1 - \tau\, i/(n-1))$ at height $z_i = i\cdot\text{spacing}$ (largest at the bottom). For each crossing at braid position $k$, a band is placed at azimuth $\theta = 2\pi(k+\tfrac12)/L$ connecting the rims of disks $i$ and $i+1$. The band's centerline follows a smoothstep profile $s = t^2(3 - 2t)$ in both radius and height, and its cross-section is rotated by $\varphi = \pm\pi\, s(t)$ about the local frame — a **half-twist** whose sign is the crossing sign — so the ribbon carries exactly the $\pm 180°$ turn Seifert's algorithm demands. Each disk is meshed from its rim (with the band-chord attachment points inserted in angular order) inward through concentric rings to a centre point. Every mesh vertex on the knot boundary is flagged as fixed for the optional relaxation steps.

**Smoothing.** Two paths are offered, both soft-depending on the Minimal Surface Toolkit's Plateau solver:

- *Relax Iterations* pins the knot exactly where the schematic put it and relaxes only the membrane (Pinkall-Polthier area minimization) — a taut soap film on the cake-shaped knot.
- *Shape Relax Rounds* relaxes the **whole shape**: each round the knot boundary itself evolves as an elastic curve (Laplacian smoothing at constant total length, with self-repulsion so strands cannot pass through each other, preserving the knot type), then the surface re-relaxes to the moved boundary. 40–80 rounds turn the wedding cake into the organic rounded form.

**Minimize Surface operator** (`mesh.seifert_minimize`). A one-click companion, active on any object carrying a `braid` property, that repeats the smoothing with tuned defaults and reports the area reduction. It offers two methods: **Area (pinned boundary)** — the elastic-boundary evolution plus a final pinned Pinkall-Polthier polish; and **Membrane (free boundary, experimental)** — a SeifertView-style inflating elastic membrane (`dynamic_relax`), where Laplacian fairing (interior by the umbrella Laplacian; boundary only along its own loop, so the knot stays a fair curve rather than retracting) is balanced by a uniform pressure force along the consistently-oriented surface normal, integrated dynamically with damping and a per-step displacement clamp. The membrane path keeps the boundary free — the key difference from the pinned area-minimizer, which can only ever hang a taut film on the frozen, tangled knot.

## References

- J. J. van Wijk, A. M. Cohen, *Visualization of Seifert Surfaces*, IEEE Transactions on Visualization and Computer Graphics 12(4), 2006 — SeifertView: <https://vanwijk.win.tue.nl/seifertview/> (braid-word input, wedding-cake layout, dynamic relaxation).
- H. Seifert, *Über das Geschlecht von Knoten*, Mathematische Annalen 110, 1934 (the algorithm itself).
- R. Scharein, *KnotPlot* — <https://knotplot.com/> (the physical-model smoothing SeifertView's dynamic relaxation is based on).
- U. Pinkall, K. Polthier, *Computing Discrete Minimal Surfaces and Their Conjugates*, Experimental Mathematics 2(1), 1993 (the pinned area-minimization used for smoothing).
