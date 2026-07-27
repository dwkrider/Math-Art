# Polyhedral Weave

![Polyhedral Weave](../images/weave.png)

## Overview

Woven-strand spheres built from a seed polyhedron, a Blender take on Antiprism's `poly_weave` including its **weave pattern language**. The seed's flag (barycentric) subdivision is walked by a small pattern program; the visited pattern points become closed weave strands that are swept as ribbons over the sphere. A short pattern string chooses everything from a plain over-under weave to rings around vertices or faces, curved segments, and raised or wavy weaves.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Seed | Cube | Seed polyhedron: Cube, Icosahedron, Octahedron, Tetrahedron, Dodecahedron. |
| Geodesic Frequency | 1 | Geodesic subdivision of triangular seeds (Tetrahedron, Octahedron, Icosahedron only; min 1, max 6). |
| Pattern Preset | Classic Weave (vfe) | Ready-made pattern that fills the Pattern field: Classic Weave (vfe), Corner Weave (FEV), Vertex Rings (1,1,1V), Face Rings (1,1,1F), Face Pairs (1,1,1FFE), Curved, Raised Weave, Wavy, Knots (1,3,1EV), or Custom. |
| Pattern | vfe | The pattern program: `[C\|L][bV,bE,bF][:up,side,along...]steps[tl\|tr\|tb]`, with steps drawn from `V E F v e f R -`. |
| Strand Width | 0.10 | Ribbon width across the strand (min 0.005, max 0.5). |
| Strand Thickness | 0.03 | Ribbon thickness (min 0.002, max 0.2). |
| Weave Amplitude | 0.05 | Alternating radial offset applied at pattern points to make strands weave over and under; set 0 when the pattern's own `up` values already weave (min 0.0, max 0.3). |
| Path Subdivision | 6 | Number of segments per step along curved paths (min 1, max 24). |
| Smoothing | 2 | Rounds of corner smoothing on curved paths (min 0, max 10). |
| Coloring | Per Strand | Per Strand (one material per woven strand) or None. |
| Radius | 1.0 | Output radius; the result is fit within a 2×Radius cube at the origin (min 0.01, max 100.0). |





## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/weave__CUBE.png" width="200"><br><sub>Cube</sub></td>
<td align="center"><img src="../images/variants/weave__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
<td align="center"><img src="../images/variants/weave__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/weave__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/weave__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
</tr>
</table>

## How it works

The seed polyhedron is projected onto the unit sphere; triangular seeds are optionally refined by a **Class-I geodesic** subdivision of frequency $\nu$, splitting each triangle $ABC$ into $\nu^2$ smaller triangles at barycentric points $\tfrac{iA+jB+kC}{\nu}$ (with $i+j+k=\nu$) re-projected to the sphere.

**Flags.** The weave is walked over the polyhedron's **flag (barycentric) subdivision**. A flag is a triple (face $i$, corner $i$, side $s$) — essentially one of the small triangles you get by cutting each face from its centre to each vertex and edge midpoint. Three elementary crossings move to the neighbouring flag that shares everything except one element: `v` changes the vertex, `e` changes the edge, `f` crosses to the flag on the adjacent face.

**Pattern language.** A pattern string is parsed into:

- an optional leading `C` or `L` — a smoothed **curve** or a **linear** path (default `C`);
- a barycentric start point `bV,bE,bF` giving the point in each flag as $\tfrac{b_V\,v + b_E\,e + b_F\,f}{b_V+b_E+b_F}$ over the flag's vertex $v$, edge midpoint $e$ and face centroid $f$ (default `0,1,0` = the edge centre);
- zero or more intermediate points `:up[,side[,along]]` inserted along each step, offset radially by `up` and sideways by `side`;
- a sequence of **steps**. Lowercase `v e f` are single crossings; uppercase `V E F` are a full **rotation click** about that element — two crossings landing on a flag of the same handedness. `R` reverses the pivot sense and `-` stays put. A trailing `tl`, `tr` or `tb` starts circuits from the left, right or both flags.

Each rotation click is implemented as two elementary crossings whose order sets the sense (`_CLICK = {F:(e,v), E:(v,f), V:(e,f)}`); `V`'s default sense is chosen opposite to `F`'s so that the classic `FEV` pattern advances as a travelling weave strand instead of closing on itself.

**Circuits.** Starting from each start flag, the pattern steps are applied cyclically, emitting a pattern point per step (plus any intermediates), until the flag/phase/direction state repeats — a closed circuit. Duplicate circuits (the same loop reached from another flag or traversed backward) are removed by comparing point sets.

**Ribbon sweep.** Each circuit is resampled (`subdiv` segments per step for curves, with a cosine-eased radial modulation of amplitude `amplitude` alternating $\pm$ at successive points so the strand weaves over and under), optionally smoothed by the stencil $\tfrac{a+2b+c}{4}$, and swept into a rectangular ribbon. At each sample the frame is the path tangent, the radial direction $\hat p$, and their cross product; the four corners are offset by $\pm\tfrac{\text{width}}{2}$ sideways and $\pm\tfrac{\text{thickness}}{2}$ radially. Each face records its strand index as a `strand_index` attribute for per-strand colouring.

## References

- Adrian Rossiter, *Antiprism* and its `poly_weave` program and pattern language — <https://www.antiprism.com>, examples at <https://www.antiprism.com/examples/200_programs/700_poly_weave/imagelist.html> (GPL; the reference implementation this follows).
