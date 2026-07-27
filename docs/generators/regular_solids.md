# Regular Solids

![Regular Solids](../images/regular_solids.png)

## Overview

A complete *Add Regular Solid* organised by family: the five **Platonic** solids, the four **Kepler–Poinsot** regular star polyhedra (built as their true intersecting faces), all thirteen **Archimedean** semiregular solids and all thirteen **Catalan** duals, uniform **prisms and antiprisms**, and the **Johnson solids J1–J48**. Options add generic stellation, Solid / Leonardo / Wireframe styles, coloring by face size, a handedness switch for the nine chiral solids, and a "congruent pieces" splitter that carves the shell into rotated identical parts for printing and reassembly.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Family | Platonic | Which family to build: Platonic, Kepler–Poinsot (true intersecting faces), Archimedean (all 13), Catalan (all 13 duals), Prisms & Antiprisms, or Johnson (J1–J48). |
| Solid | (first in family) | The specific solid within the chosen family. |
| Sides | 6 | Base sides of a prism / antiprism (min 3, max 32). |
| Handedness | Right-Handed | For chiral solids only: build the shape as constructed (right) or its mirror image (left, the other enantiomorph). |
| Stellated | Off | Replace each face with a pyramid to the intersection of its neighbours' planes (octahedron → stella octangula, dodecahedron → small stellated dodecahedron). Disabled where neighbour planes never meet. |
| Style | Solid | Solid closed mesh, Leonardo (da Vinci) open-faced panels (shared Leonardo Style modifier), or Wireframe struts. |
| Border | 0.3 | Leonardo face-frame width, as a fraction of the face (min 0.02, max 0.95). |
| Thickness | 0.05 | Panel / strut thickness for the Leonardo and Wireframe styles (min 0.001, max 1.0). |
| Coloring | By Face Size | One material per face size (shared Conway palette), or None. |
| Congruent Pieces | 1 | Split the shell into this many congruent, connected pieces (rotated copies, one object each). 1 = a single object (min 1, max 60). |
| Explode | 0.0 | Move each piece outward along its centroid direction so the split is visible (min 0.0, max 5.0). |
| Scale | 1.0 | Uniform output scale (min 0.01, max 100.0). |


## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/regular_solids__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/regular_solids__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
</tr>
</table>

## How it works

The generator draws on several distinct constructions, one per family.

**Platonic, Archimedean and Catalan** solids are produced through **Conway operator notation** (see the Conway Operators generator). Each solid carries a notation string — the cuboctahedron is `aC` (ambo cube), the truncated icosahedron `tI`, the rhombic triacontahedron `daD` (dual of ambo dodecahedron), and so on. The raw operator output is then run through George Hart's **canonicalization**, which iterates two corrections until convergence:

- edge midpoints are pushed toward tangency with the unit sphere, and the whole solid is recentred on the average edge-tangent point;
- every face is planarized by projecting its vertices onto their best-fit plane.

The result is the canonical form in which every edge is tangent to a common sphere and faces are exactly planar — reproducing the correct Archimedean / Catalan geometry. Vertex/face counts are verified for all 26 (snub cube 24/38, truncated icosahedron 60/32, snub dodecahedron 60/92, …).

**Prisms and antiprisms** are built from exact unit-edge coordinates. An $n$-gon of circumradius $r_n = \tfrac{1}{2\sin(\pi/n)}$ is placed at $z=\pm\tfrac12$ (prism) or rotated by half a step at antiprism height $h = \sqrt{1 - \big(2 r_n \sin\tfrac{\pi}{2n}\big)^2}$.

**Johnson solids J1–J48** are composed from exact unit-edge building blocks: pyramids, cupolas, and the pentagonal rotunda (sliced from a canonical icosidodecahedron), stacked on a common ring with optional prism or antiprism middles. A pyramid apex sits at height $\sqrt{1 - r_n^2}$ above the ring; a cupola rises by the height at which its slant edges are unit length; ortho vs. gyro pairings differ by a whole-step twist of the bottom cap. All 47 are verified to machine precision (all edges unit length, Euler characteristic $\chi = V - E + F = 2$), and ortho/gyro pairings are confirmed by like-meets-like contact tests. J49 and above (augmented / diminished forms) are not included.

**Kepler–Poinsot** star polyhedra are built as their *true self-intersecting faces*, not as convex approximations. The great dodecahedron $\{5, \tfrac52\}$ and great icosahedron $\{3, \tfrac52\}$ take their faces directly from the neighbour rings of an icosahedron. The small $\{5/2, 5\}$ and great $\{5/2, 3\}$ stellated dodecahedra are made of genuine pentagrams: each pentagram $\{5/2\}$ is star-triangulated by finding the five inner crossing points where the extended chords $i \to i{+}2$ meet, then filling the central pentagon and five points as triangles. For the great stellated dodecahedron the five pentagram points are the *second* ring of each dodecahedron face (the vertices one step out), so the face planes cut deep and interpenetrate into the spiky star.

**Generic stellation** (the Stellated option) replaces every face with a pyramid whose apex is the least-squares intersection of the planes of that face's edge-neighbours. For each neighbour plane $(\mathbf n_g, d_g)$ the apex $\mathbf a$ solves

$$\Big(\textstyle\sum_g \mathbf n_g \mathbf n_g^{\!\top}\Big)\,\mathbf a = \sum_g d_g\, \mathbf n_g,$$

via Cramer's rule. When the neighbour planes are all parallel to a common axis (cube, prisms, flat-capped elongated Johnson solids) the system is rank-deficient and the option is disabled. The octahedron yields the stella octangula, the dodecahedron the small stellated dodecahedron, the icosahedron the small triambic icosahedron.

**Congruent Pieces.** The solid's own rotation group is recovered *from the mesh*: candidate rotations align face 0's local frame (normal plus an in-plane axis) with every same-size face at every cyclic offset, and each candidate is kept only if it maps the whole set of face centroids onto itself. A **free subgroup** of order $N$ (one whose non-identity elements fix no face) then supplies $N$ pieces. One piece is grown as a compact, connected fundamental domain by greedily annexing the adjacent unassigned face nearest the running centroid; the other pieces are its group translates. A hill-climbing pass de-jags the partition by re-choosing which group copy of each face-orbit lands in which piece, minimizing cut perimeter plus a hinge-face penalty while keeping every piece edge-connected. Icosahedron → 2/4/5/10 pieces, cube → 2/3/6, rhombic triacontahedron → 3/5, and so on. Because the domain search is not mirror-symmetric, the left-handed form of a chiral solid takes exactly the mirrored right-handed split.

## References

- N. W. Johnson, *Convex Polyhedra with Regular Faces*, Canadian Journal of Mathematics 18, pp. 169–200, 1966 (the Johnson solid enumeration).
- G. W. Hart, *Calculating Canonical Polyhedra*, Mathematica in Education and Research 6(3), pp. 5–10, 1997 (the canonicalization algorithm).
- G. W. Hart, *Conway Notation for Polyhedra* — <https://www.georgehart.com/virtual-polyhedra/conway_notation.html>
- Adrian Rossiter, *Antiprism* polyhedron modelling software — <https://www.antiprism.com> (the Conway / canonicalization approach followed here).
- L. Poinsot, *Mémoire sur les polygones et les polyèdres*, Journal de l'École Polytechnique 10, 1810 (the four regular star polyhedra).
