# Polyhedron Squeeze Surface

![Polyhedron Squeeze Surface](../images/squeeze.png)

## Overview
A polyhedron whose edges are replaced by curves bending inward along one of their two faces, with minimal surfaces spanning the bent frames -- inspired by Robert Fathauer's "Cubic Squeeze" sculpture. Every edge is claimed by exactly one of its two faces and bows toward that face's centre; around each face the claimed edges alternate with edges claimed by neighbours. On a square face the claimed pair are opposite edges squeezing toward each other (the "squeeze"), which is only possible when every face has an even number of edges.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Seed | Cube | Base polyhedron: Cube (the Cubic Squeeze), Rhombic Dodecahedron, Truncated Octahedron, Hexagonal Prism, or Active Object (all faces must have an even number of edges). |
| Bend | 0.45 | Inward bend of each edge, as a fraction of the distance from the edge midpoint to its face centre. Range 0-0.95. |
| Edge Samples | 12 | Subdivisions of each bent edge. Range 3-48. |
| Rings | 10 | Concentric rings toward each face centre. Range 2-48. |
| Solver Iterations | 30 | Plateau area-minimization iterations per face. Range 0-200. |
| Alternate Pattern | Off | Flip which edge pair each face claims. |
| Smooth Shading | Off | Shade the mesh smooth. |
| Thickness | 0.0 | Solidify modifier thickness (0 = raw surface). Range 0-1. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |


## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/squeeze__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/squeeze__RHOMBIC.png" width="200"><br><sub>Rhombic Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/squeeze__TRUNCOCT.png" width="200"><br><sub>Truncated Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/squeeze__HEXPRISM.png" width="200"><br><sub>Hexagonal Prism</sub></td>
</tr>
</table>

## How it works

**Parity assignment.** The core problem is choosing, for each edge, which of its two faces claims it, so that (a) every edge is claimed exactly once and (b) claims alternate around every face. The generator assigns each face a phase $p_f\in\{0,1\}$ and declares slot $i$ claimed iff $(i+p_f)$ is even. For an edge shared by slot $i$ of face $f$ and slot $j$ of face $g$, "claimed once" requires
$$(i+p_f) + (j+p_g) \equiv 1 \pmod 2.$$
This is solved by **breadth-first parity propagation** over the face graph. It requires every face to have an even number of edges (odd faces are rejected immediately), a closed manifold mesh (each edge in exactly two faces), and no incompatible cycle; otherwise a `ValueError` is reported. `Alternate Pattern` flips all phases.

**Bending.** Each claimed edge $A\!\to\!B$ bows toward its face centre. Let $\mathbf d$ be the centre-minus-midpoint vector, projected in-plane orthogonal to the edge direction. The bent samples are
$$P(t) = A + (B-A)\,t + \mathbf d\,\big(\text{bend}\cdot 4t(1-t)\big),\quad t=\tfrac{j}{m},$$
a smooth $4t(1-t)$ profile peaking at the midpoint. Edge samples are shared verbatim between neighbouring faces, so the welded result is watertight with crisp creases along the bent edges.

**Spanning membrane.** For each face, the bent edges are concatenated into a closed boundary loop $B$ of $n_b$ points. An initial fan of `rings` concentric rings is built by scaling the loop toward its centroid, plus a single centre vertex, and triangulated. The boundary ring is pinned (`fixed`) and the interior is relaxed by the toolkit's `minimize_area` -- a Plateau area minimizer (Pinkall-Polthier cotangent-Laplacian iteration solved by conjugate gradients). If the toolkit is unavailable it falls back to uniform Laplacian smoothing ($P \leftarrow P + 0.6(\overline{P}-P)$ on free vertices). Finally the mesh is centered, fit within a $2\times\text{scale}$ cube, tagged with a `face_index` attribute, and optionally given a Solidify shell.

## References

- R. Fathauer, "Cubic Squeeze" and related sculpture; Robert Fathauer, https://robertfathauer.com/
- U. Pinkall and K. Polthier, "Computing Discrete Minimal Surfaces and Their Conjugates," *Experimental Mathematics* 2(1), 1993, pp. 15-36 (the area-minimization solver).
- K. Brakke, *Surface Evolver* and periodic-surface pages, https://kenbrakke.com/evolver (the Plateau-problem approach the solver is modeled on).
