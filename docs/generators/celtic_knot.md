# Celtic Knot

![Celtic Knot](../images/celtic_knot.png)

## Overview

Strands weaving over and under along the edges of a framework mesh — a port of Adam Newgas's *Celtic Knot* Blender add-on (BorisTheBrave/celtic-knot, MIT license). The framework can be one of the built-in Platonic seeds (so the operator works straight from the Add menu) or the active mesh object. Two weave types are offered: **Celtic** (plain weaving where every crossing alternates) and **Twill** (over-two, under-two, via a heuristic edge colouring after Akleman et al.). Optional remeshing refines the pattern, and strands come out as bezier curves, beveled pipes, or flat ribbon meshes.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Framework | Icosahedron | Mesh the strands weave along: Icosahedron, Dodecahedron, Cube, Octahedron, Tetrahedron, or Active Object (weave over the active mesh). |
| Remesh | None | Pre-process the framework: None, Edge Subdivide (subdivide every edge), or Medial (replace every vertex with a fan of faces). |
| Weave | Celtic | Celtic (plain weaving: alternating crossings) or Twill (over two then under two, heuristic). |
| Twist Proportion | 1.0 | Fraction of edges that cross (Celtic only); the rest pass straight through (min 0.0, max 1.0). |
| Output | Pipes | Bezier Strands (bare curves), Pipes (curves beveled into round tubes), or Ribbons (flat ribbon mesh). |
| Weave Up | 0.12 | Lift of a strand crossing over, along the surface normal (min 0.0, max 2.0). |
| Weave Down | 0.12 | Drop of a strand crossing under (min 0.0, max 2.0). |
| Handles | Auto | Bezier handle style: Auto (automatic control points) or Aligned (fixed crossing-angle control points). |
| Crossing Angle | 45° (π/4) | Aligned handles: angle between strands at a crossing (min 0, max π/2). |
| Crossing Strength | 0.25 | Aligned handles: control-point length (min 0.0, max 5.0). |
| Pipe Radius | 0.05 | Bevel radius of the pipes (Pipes output; min 0.001, max 1.0). |
| Bevel Resolution | 4 | Bevel resolution of the pipes (min 1, max 12). |
| Ribbon Length | 0.9 | Fraction along faces the ribbon runs (Ribbons output; min 0.05, max 1.0). |
| Ribbon Breadth | 0.5 | Ribbon width as a fraction across faces (min 0.05, max 1.0). |
| Coloring | Per Strand | Per Strand (a distinct material per strand), Per Braid (fewest materials with no two crossing strands sharing one), or None. |
| Scale | 1.0 | Built-in frameworks only: half-size of the bounding cube (min 0.01, max 100.0). |



## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/celtic_knot__ICOSA.png" width="200"><br><sub>Icosahedron</sub></td>
<td align="center"><img src="../images/variants/celtic_knot__DODECA.png" width="200"><br><sub>Dodecahedron</sub></td>
<td align="center"><img src="../images/variants/celtic_knot__CUBE.png" width="200"><br><sub>Cube</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/celtic_knot__OCTA.png" width="200"><br><sub>Octahedron</sub></td>
<td align="center"><img src="../images/variants/celtic_knot__TETRA.png" width="200"><br><sub>Tetrahedron</sub></td>
</tr>
</table>

## How it works

The knot is a **midpoint weave** over a framework mesh. Every strand threads from edge midpoint to edge midpoint, and at each edge it either passes straight or twists (crosses over/under its neighbour). The whole pattern is produced by walking the mesh's **half-edge loops**.

**Loops and turning.** Blender's bmesh gives each face-edge a *loop*. A `DirectedLoop` is a loop plus a facing. Two moves drive the walk: `next_face_loop` advances around the current face to the next non-boundary edge, and `next_edge_loop` swaps to the loop on the other side of the edge (used when a crossing turns the strand). Walking with these moves, marking loops entered/exited so each is visited once, decomposes the surface into closed **strands**.

**Twists.** Each edge is assigned a twist — `STRAIGHT`, `TWIST_CW`, `TWIST_CCW`, or `IGNORE` (boundary):

- **Celtic** weaves assign `TWIST_CW` to a random fraction (**Twist Proportion**) of edges and `STRAIGHT` to the rest, seeded deterministically so the result is reproducible.
- **Twill** weaves need an over-two/under-two edge colouring. On a **Medial**-remeshed framework this is exact (edges of the original faces get one twist, the rest the other). Otherwise a **heuristic voting** colouring after Akleman et al. is grown from a seed edge: uncoloured frontier edges accumulate CW/CCW votes from local edge-, face-, and vertex-conditions, and the highest-voted edge is coloured next until the whole mesh is consistent.

**Weaving offset.** At each crossed edge midpoint the strand is displaced along the averaged loop normal by a signed offset:

$$\text{offset} = \begin{cases} -\text{weave\_down} \text{ or } +\text{weave\_up}, & \text{twisting (which one depends on facing/sense)}\\[2pt] \tfrac{\text{weave\_up}-\text{weave\_down}}{2}, & \text{straight.}\end{cases}$$

so the strand passing **over** lifts by `weave_up` and the one passing **under** drops by `weave_down`.

**Remeshing.** *Edge Subdivide* inserts a vertex at every edge midpoint (each face becomes a $2n$-gon through vertex and edge-midpoint points); *Medial* replaces each vertex with a fan of faces, turning faces into edge-midpoint polygons plus corner triangles. Both refine how densely the strands weave.

**Output.** Strands become closed bezier splines (`Bezier Strands`), optionally beveled into round tubes (`Pipes` — set **Pipe Radius** and **Bevel Resolution**), or a flat quad **ribbon** whose corners run **Ribbon Length** along and **Ribbon Breadth** across each face. With `Aligned` handles the control points are placed at the fixed **Crossing Angle** and **Crossing Strength**. Per-strand colouring gives each strand its own material; **Per Braid** partitions strands with a greedy graph colouring so no two crossing strands share a material.

## References

- Adam Newgas, *Celtic Knot* Blender add-on — <https://github.com/BorisTheBrave/celtic-knot> (MIT license; the add-on this ports).
- E. Akleman, J. Chen, Q. Xing, J. L. Gross, *Cyclic Twill-Woven Objects*, Computers & Graphics 35(3), 2011, pp. 623–631 (the over-two/under-two twill colouring heuristic).
