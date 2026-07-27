# 4D Polytopes

![4D Polytopes](../images/polytope4d.png)

## Overview

The six regular convex 4-polytopes — the 5-cell, tesseract (8-cell), 16-cell, 24-cell, 600-cell and 120-cell — rendered in 3D as strut-and-sphere edge frameworks. Edges can be **straight** (a perspective projection from 4-space that approaches a Schlegel diagram at close range) or **curved** (vertices placed on the unit 3-sphere, edges drawn as great-circle arcs and mapped by exact stereographic projection, so every edge becomes a circular arc). It adds 4D rotation sliders, a Leonardo open-panel style, equatorial half-cutaways, primal+dual compounds, and the 120-cell's twelve Hopf-fibration rings of dodecahedral cells.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Polytope | Tesseract (8-cell) | Which regular 4-polytope: 5-cell (5 V, 10 E), Tesseract (16 V, 32 E), 16-cell (8 V, 24 E), 24-cell (24 V, 96 E), 120-cell (600 V, 1200 E), or 600-cell (120 V, 720 E). |
| Edges | Curved (stereographic) | Curved (vertices on the 3-sphere, great-circle arcs, stereographic projection → circular arcs) or Straight (direct 4D perspective; small distance ≈ Schlegel diagram). |
| Projection Distance | 1.05 | Eye distance along the $w$ axis for straight edges; near 1 gives a Schlegel diagram (min 1.001, max 10.0). Curved mode always uses the exact stereographic projection. |
| Rotate XW / YW / ZW / XY | 0.0 | 4D rotation in each named coordinate plane, in degrees (min -180, max 180). |
| Arc Segments | 12 | Samples per edge, for curved arcs and tapering (min 1, max 48). |
| Strut Radius | 0.03 | Radius of the edge struts (min 0.002, max 0.5). |
| Strut Sides | 6 | Cross-section sides of each strut tube (min 3, max 16). |
| Taper With Projection | On | Scale strut thickness by the local projection factor (features near the pole appear fatter). |
| Vertex Spheres | On | Cap each projected vertex with a small sphere. |
| Sphere Size | 1.6 | Vertex-sphere radius as a multiple of the strut radius (min 1.0, max 4.0). |
| Style | Edge Struts | Edge Struts (tubes along the projected edges) or Leonardo (da Vinci) (a flat open panel per 2D face). |
| Border | 0.35 | Leonardo panel frame width, as a fraction of the face (min 0.02, max 0.95). |
| Panel Thickness | 0.03 | Leonardo panel thickness (min 0.002, max 0.5). |
| Scale | 1.0 | Uniform output scale; the framework is fit to a 2 m cube times this (min 0.01, max 100.0). |
| Half (Cutaway) | Off | Keep only elements on one side of the equatorial hyperplane ($w \le 0$, equator included) before projection — Segerman's legible half-models. |
| Dual Compound | Off | Also build the dual polytope (5↔5, 8↔16, 24↔24, 120↔600), its vertices at this polytope's cell centres, in a second material slot. |
| Hopf Rings | 0 | 120-cell only: render N of the 12 rings of 10 dodecahedral cells that partition the 120-cell along Hopf fibers, as solid shrunken cells, one material per ring (min 0, max 12). |
| Ring Cell Scale | 0.9 | Shrink factor of each ring cell toward its 4D centroid, so gaps make the rings legible (min 0.1, max 1.0). |
| Rings Only | Off | Drop the edge framework and show just the rings of solid cells (the printable, fig. 3-29 look). |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/polytope4d__CELL5.png" width="200"><br><sub>5-cell</sub></td>
<td align="center"><img src="../images/variants/polytope4d__CELL8.png" width="200"><br><sub>Tesseract</sub></td>
<td align="center"><img src="../images/variants/polytope4d__CELL16.png" width="200"><br><sub>16-cell</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/polytope4d__CELL24.png" width="200"><br><sub>24-cell</sub></td>
<td align="center"><img src="../images/variants/polytope4d__CELL120.png" width="200"><br><sub>120-cell</sub></td>
<td align="center"><img src="../images/variants/polytope4d__CELL600.png" width="200"><br><sub>600-cell</sub></td>
</tr>
</table>

## How it works

**Vertices.** Each polytope is built from its standard coordinate set — the tesseract as $(\pm1,\pm1,\pm1,\pm1)$; the 16-cell as all sign changes of $(1,0,0,0)$; the 24-cell from $(\pm1,\pm1,0,0)$; the 600-cell and 120-cell from sign/permutation orbits involving the golden ratio $\varphi$ (e.g. the 600-cell includes the even permutations of $(\varphi/2, \tfrac12, 1/(2\varphi), 0)$). All vertices are normalized onto the unit 3-sphere. **Edges** are the vertex pairs at the minimal nonzero distance; **2D faces** (needed for the Leonardo style) are found by walking edge cycles that stay within a common 2-flat of $\mathbb R^4$. Vertex/edge counts are verified against the known values (120-cell: 600 vertices, 1200 edges; 600-cell: 120/720).

**4D rotation.** Before projecting, every point set is rotated by the same 4D rotation, composed of independent rotations in the XW, YW, ZW and XY planes (a rotation in plane $(a,b)$ sends $v_a \mapsto v_a\cos t - v_b\sin t$, $v_b \mapsto v_a\sin t + v_b\cos t$).

**Projection.** A point $\mathbf v = (x,y,z,w)$ is projected to $\mathbb R^3$ by central projection from the eye $(0,0,0,d)$:

$$\pi(\mathbf v) = \frac{1}{d - w}\,(x, y, z),\qquad \text{local scale } s = \frac{1}{d-w}.$$

- **Straight** edges use eye distance $d = \max(\text{proj\_dist}, 1.001)$ and draw each edge as a line segment; small $d$ pulls the near cell forward into a **Schlegel diagram**.
- **Curved** edges set $d = 1$, which — for $\mathbf v$ on the unit 3-sphere — is exactly the **stereographic projection** from the pole $(0,0,0,1)$. Each edge is sampled along the great-circle arc between its endpoints by 4D spherical interpolation (slerp), and every projected sample is joined into a tube. Because stereographic projection is conformal and maps circles to circles, each great-circle edge becomes a **circular arc** in $\mathbb R^3$, and each 2D face's circumscribing circle maps to a circle, so projected faces stay planar (which is what lets the Leonardo panels be flat in both modes). When a vertex sits at the pole ($w \approx 1$) it would project to infinity, so a small deterministic extra 4D rotation is applied first to clear the pole, and the same rotation is applied to every auxiliary point set so compounds stay aligned.

**Struts and spheres.** Each projected edge becomes a closed tube with a parallel-transport frame (a stable per-ring normal); if *taper* is on, the per-point radius scales with the local projection factor $s$, so features near the pole render fatter and far features thinner. Vertices are optionally capped with small UV spheres of radius `radius × sphere_factor × s`.

**Leonardo panels.** In the Leonardo style each projected 2D face becomes a flat open frame: the face is inset by `Border`, a hole is cut, and a solid shell is extruded. Every polytope vertex is offset once, along a least-squares mitre direction solving $\mathbf m\cdot\mathbf n_f = \text{thickness}$ over all panels meeting there, so adjacent panels share inner-boundary vertices and the joints along edges and at corners are exact and watertight.

**Cutaway, compound and rings.**

- **Half** keeps only vertices with $w \le 0$ (plus the equator) and the edges/faces they span, in the polytope's own coordinates before any rotation — Segerman's legible half-models.
- **Dual compound** adds the dual polytope. In curved mode both frameworks lie on the 3-sphere; in straight mode the dual is scaled to the primal's cell inradius, so its vertices sit at the primal's cell centres.
- **Hopf rings** (120-cell only) exploit the fact that the 120 cell centres are the 600-cell vertices — the unit **icosians**, the binary icosahedral group $2I$. Left cosets of a cyclic order-10 subgroup partition them into 12 rings of 10 cells that follow the fibers of the Hopf fibration $S^3 \to S^2$. Each ring's dodecahedral cells (the 20 nearest polytope vertices, shrunk toward their 4D centroid) are projected and hulled into solid cells, one material per ring. Two rings give the classic pair of interlocked orthogonal rings; *Rings Only* drops the strut framework for the printable presentation.

## References

- H. Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 (figs. 3-20/22/23, 3-25, 3-29 — stereographic 4-polytope projections, half-models, dual compounds, and the Hopf-ring construction reproduced here).
- H. S. M. Coxeter, *Regular Polytopes*, 3rd ed., Dover, 1973 (the six regular 4-polytopes and their coordinates).
- J. H. Conway and D. A. Smith, *On Quaternions and Octonions*, A K Peters, 2003 (the icosians / binary icosahedral group $2I$ underlying the Hopf rings).
