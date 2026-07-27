# Spiked & Hyperbolic Polyhedra

![Spiked & Hyperbolic Polyhedra](../images/spiked_polyhedron.png)

## Overview

A family of augmented and folded Platonic sculptures. **Spiked** raises a pyramid on every face of a Platonic seed — an icosahedron at pyramid height $\sqrt6/3$ becomes a 60-face deltahedron of regular tetrahedra. **Hyperbolic** sags each face toward the centre while spiking the vertices outward. **Folded / Rhombic Hexecontahedron** builds the 60 golden rhombi of the rhombic hexecontahedron (the union of 20 acute golden rhombohedra) and can fold each rhombus gently along its long diagonal.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Form | Folded Rhombic Hexecontahedron | Spiked Polyhedron, Hyperbolic Polyhedron, Folded Rhombic Hexecontahedron, or the flat Rhombic Hexecontahedron |
| Seed | Icosahedron | Base Platonic solid for the Spiked form (Tetrahedron, Cube, Octahedron, Dodecahedron, Icosahedron) |
| Seed (Hyper) | Dodecahedron | Base Platonic solid for the Hyperbolic form |
| Spike Height | $\sqrt6/3 \approx 0.816$ | Pyramid height in units of edge length; $\sqrt6/3$ makes regular tetrahedra on triangular faces (Spiked) |
| Spike Length | 1.5 | Radial factor at the vertices (Hyperbolic) |
| Sharpness | 3.0 | How abruptly the spikes rise from the sagging faces (Hyperbolic) |
| Resolution | 12 | Subdivision of each face (Hyperbolic) |
| Mid Scale | 1.0 | Radial scale of the rhombus mid vertices (1 = rhombic hexecontahedron) |
| Top Scale | 1.05 | Radial scale of the rhombus tips; 1 is flat, ~1.05 a gentle fold |
| Coloring | None | By Face Group (one material per seed face, 12-colour palette) or None |
| Style | Solid | Solid, Leonardo open-faced panels, or Wireframe modifier |
| Border | 0.3 | Leonardo face-frame width |
| Thickness | 0.04 | Leonardo / Wireframe thickness |
| Smooth Shading | True | Smooth shading (Hyperbolic form only) |
| Scale | 1.0 | Overall size multiplier |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/spiked_polyhedron__SPIKED.png" width="200"><br><sub>Spiked</sub></td>
<td align="center"><img src="../images/variants/spiked_polyhedron__HYPER.png" width="200"><br><sub>Hyperbolic</sub></td>
<td align="center"><img src="../images/variants/spiked_polyhedron__MODERN.png" width="200"><br><sub>Folded Rhombic Hexecontahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/spiked_polyhedron__RHOMBIC.png" width="200"><br><sub>Rhombic Hexecontahedron</sub></td>
</tr>
</table>

## How it works

**Platonic seeds.** Each seed is generated at circumradius 1. The icosahedron uses the cyclic coordinates $(0,\pm1,\pm\varphi)$ with $\varphi=\tfrac{1+\sqrt5}{2}$, normalized by $1/\sqrt{1+\varphi^2}$; the dodecahedron is its dual (a vertex at each icosahedral face centroid, faces ordered around each icosahedral vertex).

**Spiked (face augmentation).** Every face $f$ with centroid $c$, unit outward normal $\hat n$ and edge length $e$ gains an apex

$$a = c + h\,e\,\hat n,$$

and the face is replaced by the pyramid of triangles from each edge to $a$. On the icosahedron the special height

$$h=\frac{\sqrt6}{3}\approx0.8165$$

makes each pyramid a *regular* tetrahedron, so the 20 faces become 60 equilateral triangles — a closed 60-face deltahedron (the self-test verifies uniform edge length and a watertight surface).

**Hyperbolic (concave faces, spiked vertices).** Each face is subdivided into a barycentric grid over its (centroid, $A$, $B$) sub-triangles, and every point is pushed radially by a factor that stays near 1 across the face interiors and rises to the spike length at the vertices. With inradius $r_{\min}$ (centre-to-face distance), radius $r$, spike $\sigma$ and sharpness $s$,

$$w=\left(\operatorname{clip}\frac{r-r_{\min}}{1-r_{\min}},\,0,\,1\right)^{s},\qquad g=1+(\sigma-1)\,w,\qquad \mathbf{p}\mapsto g\,\mathbf{p}.$$

The exponent $s$ controls how abruptly the spikes rise from the sagging faces ($s=1$ is nearly the flat solid); every vertex reaches radius exactly $\sigma$.

**Folded & flat rhombic hexecontahedron.** For each icosahedron face with vertex vectors $a,b,c$, the construction places a tent whose mid vertices sit along $a+b$ (scaled by *mid*) and whose top sits along $a+b+c$ (scaled by *top*), forming rhombi $(a,\,a+b,\,a+b+c,\,a+c)$. At $\text{mid}=\text{top}=1$ the 120 triangles fuse in pairs into the **60 golden rhombi** of the rhombic hexecontahedron — the union of 20 acute golden rhombohedra sharing the centre, whose diagonals are in the golden ratio

$$\frac{d_{\text{long}}}{d_{\text{short}}}=\varphi.$$

Raising *top* above 1 lifts each rhombus tip, folding the rhombus along its long diagonal into two slightly non-planar triangles (the "folded" family, default $\text{top}=1.05$). The flat **Rhombic Hexecontahedron** preset emits the rhombi as planar quads. Self-tests confirm 60 planar golden rhombi, Euler characteristic $\chi=2$, and a closed surface.

## References

- Stan Wagon, "Rhombic hexecontahedron," and Eric W. Weisstein, "Rhombic Hexecontahedron," *MathWorld*: <https://mathworld.wolfram.com/RhombicHexecontahedron.html> — the 60-golden-rhombus solid as a stellation of the rhombic triacontahedron / union of 20 acute golden rhombohedra.
- Eric W. Weisstein, "Golden Rhombohedron," *MathWorld*: <https://mathworld.wolfram.com/GoldenRhombohedron.html>
- H. Martyn Cundy and A. P. Rollett, *Mathematical Models*, 2nd ed., Oxford University Press, 1961 — deltahedra and face-augmented (pyramid-capped) Platonic solids.
- H. S. M. Coxeter, *Regular Polytopes*, 3rd ed., Dover, 1973 — the golden ratio in the icosahedron/dodecahedron and the rhombic zonohedra.
