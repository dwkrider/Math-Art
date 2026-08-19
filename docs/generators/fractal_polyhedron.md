# Fractal Polyhedron

![Fractal Polyhedron](../images/fractal_polyhedron.png)

## Overview

Recursive clusters of Platonic solids, after the fractal-polyhedron construction on George W. Hart's pages: every generation places a scaled copy of the solid at each vertex, face centre or edge midpoint of the current copies. Vertex mode with child scale $\tfrac12$ on a tetrahedron, parents removed, is the classic Sierpinski tetrahedron; face mode grows spiky, coral-like clusters. Twist and per-child rotation sliders make the growth chiral, and each generation can be coloured separately.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Solid | Cube | Seed Platonic solid: Tetrahedron, Cube, Octahedron, Dodecahedron, Icosahedron |
| Anchors | Vertices | Where children are placed: Vertices, Faces (face centres), or Edges (edge midpoints) |
| Generations | 3 | Number of recursion generations |
| Child Scale | 0.5 | Size of each child relative to its parent |
| Spread | 1.27 | Distance multiplier from parent to children |
| Keep Parents | True | Keep every generation (off = only the last, e.g. the Sierpinski gasket) |
| Push | 0.0 | Extra shift of each child along its anchor direction (in parent-size units) |
| Twist | 0.0 | Rotation of each child about its anchor direction, cumulative per generation (degrees) |
| Child Rotate X | 0.0 | Cumulative X rotation applied to each child (degrees) |
| Child Rotate Y | 0.0 | Cumulative Y rotation applied to each child (degrees) |
| Child Rotate Z | 0.0 | Cumulative Z rotation applied to each child (degrees) |
| Coloring | Per Generation | One material per generation, or None |
| Scale | 1.0 | Overall size (the cluster is fit to a 2 m cube then scaled) |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/fractal_polyhedron__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/fractal_polyhedron__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/fractal_polyhedron__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/fractal_polyhedron__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
<td align="center"><img src="../images/variants/fractal_polyhedron__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
</tr>
</table>

## How it works

**Seeds and anchors.** The five Platonic solids are generated at unit scale (the dodecahedron as the dual of the icosahedron — a vertex at each icosahedral face centroid, faces ordered around each icosahedral vertex). The chosen anchor mode gives the set of local points where children are attached:

- **Vertices** — the seed's vertices,
- **Edges** — the midpoints $\tfrac12(V_a+V_b)$ of its distinct edges,
- **Faces** — the centroids $\tfrac1{|f|}\sum_{i\in f}V_i$ of its faces.

**The iterated function system.** Each copy carries an origin $o$, a size $s$, an orientation matrix $R$ and a generation index. For every anchor $a$ (with unit direction $\hat a$) a child is created at

$$o' = o + R\,a\,\cdot s\,\rho \;+\; R\,\hat a\,\cdot p\,s,\qquad s' = s\,\kappa,$$

where $\rho$ is the **spread**, $p$ the radial **push**, and $\kappa$ the **child scale**. The orientation is updated cumulatively — first a **twist** of angle $\theta$ about the (world-space) anchor direction $R\hat a$, then a fixed Euler rotation $R_e$ from the three child-rotate angles:

$$R' = R_e\;\bigl(\text{Rot}(R\hat a,\theta)\,R\bigr).$$

This is a self-affine IFS: the attractor is the fixed set of the family of contractions $\{x\mapsto o'+R'x\,s'\}$. With Keep Parents off, only the finest generation survives; vertex mode, $\kappa=\tfrac12$, spread $1$ on a tetrahedron reproduces the **Sierpinski tetrahedron**, since the four vertex-anchored half-scale maps are exactly its defining contractions. Keeping parents accumulates every generation into one dense cluster; face and edge modes with spread $>1$ throw children outward into spiky growths.

**Assembly.** Copies are expanded into mesh geometry — each generation's copies transformed by their $R$, $s$, $o$ — with a per-face `generation` integer attribute and an optional one-material-per-generation palette. The whole cluster is recentred on its bounding-box centre and fit to a 2 m cube before the final scale is applied. Copy counts are capped (30 000) to keep meshes tractable, so deep generations of high-anchor-count seeds raise an error rather than explode.

## References

- George W. Hart, "Fractal / recursive polyhedra," *Virtual Polyhedra* and sculpture pages: <https://www.georgehart.com/virtual-polyhedra/vp.html> and <https://www.georgehart.com/>
- Michael F. Barnsley, *Fractals Everywhere*, 2nd ed., Academic Press, 1993 — iterated function systems and self-affine attractors.
- Wacław Sierpiński (1916) / the Sierpinski tetrahedron as the vertex-mode, scale-$\tfrac12$ IFS on a tetrahedron; see Eric W. Weisstein, "Tetrix," *MathWorld*: <https://mathworld.wolfram.com/Tetrix.html>
