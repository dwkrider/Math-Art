# Bubble Cluster

![Bubble Cluster](../images/bubble_cluster.png)

## Overview
A soap-bubble cluster with one bubble at every vertex of a seed mesh (the Platonic solids, or any mesh via Active Object). Radii are uniform or follow each point's mean distance to its neighbours. Where bubbles overlap, the interfaces obey soap-film physics (Young-Laplace): equal bubbles meet on a flat film, unequal ones on a curved film bulging into the larger, lower-pressure bubble, with natural triple junctions where three films meet. By default each bubble is emitted as its own closed mesh (its outer cap plus its films), and interior bubbles fully enclosed by films can also emit their flat-walled foam-cell polyhedron.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Seed | Icosahedron | Where the bubble centres sit: Tetrahedron, Cube, Octahedron, Dodecahedron, Icosahedron, or Active Object (one bubble per vertex of the active mesh). |
| Radii | Same Radius | Same Radius (all bubbles from the global mean neighbour distance) or From Neighbour Distance (each radius follows its own mean neighbour distance). |
| Radius Factor | 0.62 | Bubble radius as a fraction of the mean neighbour distance (above 0.5 neighbouring bubbles merge). Range 0.1-2.0. |
| Subdivisions | 2 | Icosphere subdivisions per bubble. Range 1-5. |
| Interior Films | On | Include the soap films between touching bubbles (merged mesh only; separate bubbles always carry their walls). |
| Separate Bubble Meshes | On | One closed mesh object per bubble (its outer cap plus its films), parented to an empty. |
| Color Bubbles | Off | Give each bubble its own material colour. |
| Intersection Polyhedra | Off | For every bubble fully enclosed by films, add its flat-walled cell polyhedron as a separate mesh (needs interior points, e.g. a lattice seed via Active Object). |
| Smooth Shading | On | Shade the bubble caps and films smooth (analytic normals; creases stay sharp). |
| Scale | 1.0 | The seed polyhedron is fitted roughly within a $2\times$ Scale cube at the origin. Range 0.01-100. |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/bubble_cluster__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/bubble_cluster__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/bubble_cluster__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/bubble_cluster__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/bubble_cluster__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
</tr>
</table>

## How it works

**Radii.** Each point's local scale is its mean edge length to its mesh neighbours (or, with no edges, its nearest-point distance). *Same Radius* uses the global mean; *From Neighbour Distance* uses the per-point value. The radius is `factor` times that scale -- above $0.62$-ish, neighbours overlap enough to merge.

**Young-Laplace films.** Two spheres of radii $R_i,R_j$ with centres distance $d$ apart intersect in a circle at signed offset $a=(d^2+R_i^2-R_j^2)/(2d)$ from centre $i$ along the axis $\mathbf u$, of radius $\rho=\sqrt{R_i^2-a^2}$. The soap film through that circle depends on the pressures. By Laplace's law the pressure jump across a film of mean curvature $\kappa$ is $\Delta P=2\gamma\kappa$; mechanical equilibrium of the two spherical caps and the interface fixes the interface curvature as the **difference** of the bubble curvatures:
$$\frac1r=\frac1{r_{\text{small}}}-\frac1{r_{\text{large}}}.$$
- **Equal radii** ($R_i=R_j$): $1/r=0$, so the film is a flat **plane** through the intersection circle.
- **Unequal radii**: the film is a sphere of radius $r_f=\dfrac{R_iR_j}{|R_j-R_i|}$ centred so the film **bulges into the larger** (lower-pressure) bubble, since the smaller bubble has the higher internal pressure.

Each pair carries sign factors so that a bubble's own centre is on the positive side of every film touching it.

**Cells and trimming.** A bubble's cell is the region on the positive side of *all* its films: `_side_min` takes the minimum signed value over every (film, sign) pair, and a point is kept when that minimum is $\ge0$. Each bubble's outer surface is its sphere (an icosphere of the chosen subdivision) trimmed to its own cell; triangles crossing the boundary are split at the zero contour (bisection, re-projected onto the sphere). Interior films are likewise trimmed where a *third* bubble's cell takes over -- which is exactly where three films meet, producing the **triple junctions** seen in real foam.

**Closed per-bubble solids.** With *Separate Bubble Meshes*, each bubble is built as one closed, star-shaped solid: an icosphere is displaced radially outward to the nearest boundary (its own sphere, label 0, or a film, label $k$), triangles spanning different surfaces are split along the crease (and a three-surface facet at a junction is fanned to the junction point), a repair pass catches creases that cross a face without changing its vertex labels, and finally every vertex is snapped exactly onto all the surfaces its facets touch by alternating projection -- so each facet lies exactly on one plane or sphere up to its edges. Faces are oriented outward (valid because the cell is star-shaped about the centre). Analytic split normals -- radial on caps, the film's own normal on films -- give perfectly smooth surfaces with perfectly sharp creases; cap/film and film/film boundary edges are marked sharp.

**Foam cells.** With *Intersection Polyhedra*, a bubble fully surrounded by films has its film **planes** (each intersection-circle plane) bound a convex polyhedron -- the flat-walled foam cell. It is computed by intersecting plane triples, keeping vertices inside all half-spaces, and is returned only when the faces close up (every edge shared by two faces). A BCC-style centre bubble surrounded by its eight corner neighbours yields the regular octahedron (verified in the self-test).

## References

- Young-Laplace law and soap-film / foam geometry: see C. Isenberg, *The Science of Soap Films and Soap Bubbles*, Dover, 1992; and D. Weaire and S. Hutzler, *The Physics of Foams*, Oxford University Press, 1999.
- Plateau's laws (triple junctions at $120^\circ$, and the equilibrium of films) underlie the trimming that produces the triple-junction network.
- The interface-curvature rule $1/r = 1/r_{\text{small}} - 1/r_{\text{large}}$ for the film between two unequal bubbles follows from equating the Laplace pressure jumps across the two caps and the interface.
