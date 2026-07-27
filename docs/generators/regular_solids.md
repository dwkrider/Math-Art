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

### Platonic

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

### Archimedean

<table>
<tr>
<td align="center"><img src="../images/variants/regular_solids__TT.png" width="200"><br><sub>Truncated Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__CO.png" width="200"><br><sub>Cuboctahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__TC.png" width="200"><br><sub>Truncated Cube</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__TO.png" width="200"><br><sub>Truncated Octahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__RCO.png" width="200"><br><sub>Rhombicuboctahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__TCO.png" width="200"><br><sub>Truncated Cuboctahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__SC.png" width="200"><br><sub>Snub Cube</sub></td>
<td align="center"><img src="../images/variants/regular_solids__ID.png" width="200"><br><sub>Icosidodecahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__TD.png" width="200"><br><sub>Truncated Dodecahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__TI.png" width="200"><br><sub>Truncated Icosahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__RID.png" width="200"><br><sub>Rhombicosidodecahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__TID.png" width="200"><br><sub>Truncated Icosidodecahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__SD.png" width="200"><br><sub>Snub Dodecahedron</sub></td>
</tr>
</table>

### Catalan

<table>
<tr>
<td align="center"><img src="../images/variants/regular_solids__KTT.png" width="200"><br><sub>Triakis Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__RD.png" width="200"><br><sub>Rhombic Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__KTC.png" width="200"><br><sub>Triakis Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__KTO.png" width="200"><br><sub>Tetrakis Hexahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__DIT.png" width="200"><br><sub>Deltoidal Icositetrahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__DDD.png" width="200"><br><sub>Disdyakis Dodecahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__PIT.png" width="200"><br><sub>Pentagonal Icositetrahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__RT.png" width="200"><br><sub>Rhombic Triacontahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__KTD.png" width="200"><br><sub>Triakis Icosahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__PKD.png" width="200"><br><sub>Pentakis Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__DHX.png" width="200"><br><sub>Deltoidal Hexecontahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__DDT.png" width="200"><br><sub>Disdyakis Triacontahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__PHX.png" width="200"><br><sub>Pentagonal Hexecontahedron</sub></td>
</tr>
</table>

### Kepler-Poinsot

<table>
<tr>
<td align="center"><img src="../images/variants/regular_solids__SSD.png" width="200"><br><sub>Small Stellated Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__GD.png" width="200"><br><sub>Great Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/regular_solids__GSD.png" width="200"><br><sub>Great Stellated Dodecahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__GI.png" width="200"><br><sub>Great Icosahedron</sub></td>
</tr>
</table>

### Prisms & Antiprisms

<table>
<tr>
<td align="center"><img src="../images/variants/regular_solids__PRISM.png" width="200"><br><sub>Hexagonal Prism</sub></td>
<td align="center"><img src="../images/variants/regular_solids__ANTIPRISM.png" width="200"><br><sub>Hexagonal Antiprism</sub></td>
</tr>
</table>

### Johnson

<table>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J1.png" width="200"><br><sub>Square Pyramid (J1)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J2.png" width="200"><br><sub>Pentagonal Pyramid (J2)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J3.png" width="200"><br><sub>Triangular Cupola (J3)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J4.png" width="200"><br><sub>Square Cupola (J4)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J5.png" width="200"><br><sub>Pentagonal Cupola (J5)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J6.png" width="200"><br><sub>Pentagonal Rotunda (J6)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J7.png" width="200"><br><sub>Elongated Triangular Pyramid (J7)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J8.png" width="200"><br><sub>Elongated Square Pyramid (J8)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J9.png" width="200"><br><sub>Elongated Pentagonal Pyramid (J9)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J10.png" width="200"><br><sub>Gyroelongated Square Pyramid (J10)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J11.png" width="200"><br><sub>Gyroelongated Pentagonal Pyramid (J11)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J12.png" width="200"><br><sub>Triangular Bipyramid (J12)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J13.png" width="200"><br><sub>Pentagonal Bipyramid (J13)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J14.png" width="200"><br><sub>Elongated Triangular Bipyramid (J14)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J15.png" width="200"><br><sub>Elongated Square Bipyramid (J15)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J16.png" width="200"><br><sub>Elongated Pentagonal Bipyramid (J16)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J17.png" width="200"><br><sub>Gyroelongated Square Bipyramid (J17)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J18.png" width="200"><br><sub>Elongated Triangular Cupola (J18)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J19.png" width="200"><br><sub>Elongated Square Cupola (J19)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J20.png" width="200"><br><sub>Elongated Pentagonal Cupola (J20)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J21.png" width="200"><br><sub>Elongated Pentagonal Rotunda (J21)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J22.png" width="200"><br><sub>Gyroelongated Triangular Cupola (J22)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J23.png" width="200"><br><sub>Gyroelongated Square Cupola (J23)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J24.png" width="200"><br><sub>Gyroelongated Pentagonal Cupola (J24)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J25.png" width="200"><br><sub>Gyroelongated Pentagonal Rotunda (J25)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J26.png" width="200"><br><sub>Gyrobifastigium (J26)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J27.png" width="200"><br><sub>Triangular Orthobicupola (J27)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J28.png" width="200"><br><sub>Square Orthobicupola (J28)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J29.png" width="200"><br><sub>Square Gyrobicupola (J29)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J30.png" width="200"><br><sub>Pentagonal Orthobicupola (J30)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J31.png" width="200"><br><sub>Pentagonal Gyrobicupola (J31)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J32.png" width="200"><br><sub>Pentagonal Orthocupolarotunda (J32)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J33.png" width="200"><br><sub>Pentagonal Gyrocupolarotunda (J33)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J34.png" width="200"><br><sub>Pentagonal Orthobirotunda (J34)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J35.png" width="200"><br><sub>Elongated Triangular Orthobicupola (J35)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J36.png" width="200"><br><sub>Elongated Triangular Gyrobicupola (J36)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J37.png" width="200"><br><sub>Elongated Square Gyrobicupola (J37)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J38.png" width="200"><br><sub>Elongated Pentagonal Orthobicupola (J38)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J39.png" width="200"><br><sub>Elongated Pentagonal Gyrobicupola (J39)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J40.png" width="200"><br><sub>Elongated Pentagonal Orthocupolarotunda (J40)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J41.png" width="200"><br><sub>Elongated Pentagonal Gyrocupolarotunda (J41)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J42.png" width="200"><br><sub>Elongated Pentagonal Orthobirotunda (J42)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J43.png" width="200"><br><sub>Elongated Pentagonal Gyrobirotunda (J43)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J44.png" width="200"><br><sub>Gyroelongated Triangular Bicupola (J44)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J45.png" width="200"><br><sub>Gyroelongated Square Bicupola (J45)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/regular_solids__J46.png" width="200"><br><sub>Gyroelongated Pentagonal Bicupola (J46)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J47.png" width="200"><br><sub>Gyroelongated Pentagonal Cupolarotunda (J47)</sub></td>
<td align="center"><img src="../images/variants/regular_solids__J48.png" width="200"><br><sub>Gyroelongated Pentagonal Birotunda (J48)</sub></td>
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
