# Symmetric Sculpture

> ⚠️ **Experimental.** This generator is a work in progress — its
> presets, options and output may change in future versions.

![Symmetric Sculpture](../images/symmetric_sculpture.png)

## Overview

A Blender adaptation of George W. Hart's sculpture design program (the tool behind *Twisted Rivers*, *Tumbleweed*, *Frabjous*, *Spaghetti Code*, ...). A flat motif drawn in **one** representative plane is replicated live into a whole family of symmetrically arranged planes — the extended face planes of an icosahedron, dodecahedron, rhombic triacontahedron or hexecontahedron (or their octahedral / tetrahedral analogues). The replication is done with a Geometry Nodes modifier, so editing the motif object (in edit mode, or by moving/rotating it) updates every copy in real time. The operator ships four presets after Hart's pieces plus a fully custom mode.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Preset | Twisted Rivers | Sculpture setup; the motif stays editable afterwards |
| — Twisted Rivers | | After *Twisted Rivers, Knotted Sea* (2001): C-shaped rivers in the 20 icosahedral planes, three per plane |
| — Tumbleweed | | After *Tumbleweed* (2006): swirling five-armed pinwheels in the 12 dodecahedral planes |
| — Frabjous | | After *Frabjous* (2006): 30 S-shaped parts in the face planes of the great rhombic triacontahedron — ends meeting in threes at the 3-fold corners ($\varphi^2 \times$ the plane distance), five-fold vortices at the 5-fold corners ($\varphi \times$) |
| — Whimsy | | After *Whimsy* (2014): 60 flat blades in the face planes of the pentagonal hexecontahedron (snub dodecahedron dual), five to a hub, meeting in threes at the 3-fold corners |
| — Custom | | Choose the group and plane family yourself; starts from the demo arc motif |
| Symmetry Group | Icosahedral (60) | Rotation group: Icosahedral (60), Octahedral (24), or Tetrahedral (12). Shown only in Custom mode |
| Plane Family | 3-fold planes | Which orbit of planes to fill. Shown only in Custom mode |
| — 5-fold planes (dodecahedral) | | 12 planes, 5-fold symmetry in each (icosahedral group only) |
| — 4-fold planes (cube) | | 6 planes, 4-fold each (octahedral group only) |
| — 3-fold planes (icosahedral) | | Planes perpendicular to the 3-fold axes: 20 icosahedral, 8 octahedral, 4 tetrahedral |
| — 2-fold planes (triacontahedral) | | Planes perpendicular to the 2-fold axes: 30 icosahedral, 12 octahedral, 6 tetrahedral |
| — General planes (hexecontahedral) | | A full-orbit family with no in-plane symmetry: 60 / 24 / 12 planes |
| Plane Distance | 1.0 | Distance of the plane family from the origin |
| Shell | 0.04 | Radial extrusion fraction (Hart used ~4%); 0 keeps the flat planes |
| Guide Extent | 2.2 | Radius of the guide pattern, in units of the plane distance |
| Use Active Object as Motif | Off | Place the active mesh object into the representative plane instead of building the demo arc motif (draw it flat in its local XY plane) |

The Geometry Nodes modifier exposes further live inputs on the result object: **Shell** (radial extrusion fraction), **Weld** (merge distance where copies meet, default $10^{-4}$), **Full Sculpture** (show all copies with the motif's real material for export/render, off = translucent design view), **Copy Material**, and the **Plane Rotation** / **Plane Offset** vectors that map the motif's XY plane onto the representative plane.




## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/symmetric_sculpture__TWISTED_RIVERS.png" width="200"><br><sub>Twisted Rivers</sub></td>
<td align="center"><img src="../images/variants/symmetric_sculpture__TUMBLEWEED.png" width="200"><br><sub>Tumbleweed</sub></td>
<td align="center"><img src="../images/variants/symmetric_sculpture__FRABJOUS.png" width="200"><br><sub>Frabjous</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/symmetric_sculpture__WHIMSY.png" width="200"><br><sub>Whimsy</sub></td>
</tr>
</table>

## How it works

**The point group.** Each symmetry choice is the chiral (rotation-only) tetrahedral, octahedral, or icosahedral group, of order 12, 24, and 60 respectively. The group is built by closure: starting from the identity and two generator rotations $G_1, G_2$ (Rodrigues matrices), every product is generated until no new $3\times3$ matrix appears, giving all $N$ rotations $\{R_k\}$.

**Plane families.** A plane family is fixed by a single axis $\mathbf a$ (a symmetry axis of the group); the representative plane sits at distance $d$ perpendicular to $\mathbf a$. The distinct plane normals are the **orbit** of $\mathbf a$ under the group,

$$\{\, \hat{R_k\,\mathbf a} \,\}, \qquad k = 1,\dots,N,$$

with duplicates removed. A $k$-fold axis has a stabiliser of order $k$, so the orbit has $N/k$ distinct normals — exactly $N/k$ planes, each carrying $k$-fold in-plane symmetry. This is Hart's plane-family construction: a motif placed perpendicular to a $k$-fold axis and instanced under all $N$ rotations lands in $N/k$ planes with $k$-fold symmetry inside each.

**Live replication (Geometry Nodes).** One point is created per rotation $R_k$, carrying $R_k$ as a Euler-angle attribute (`sym_rot`) and a flag marking the identity copy (`is_motif`). The node group instances the editable motif object on those points, applies each point's rotation, realises the instances, welds coincident vertices, and (in design view) hides the copy that coincides with the editable motif so it stands out translucently. Because the instance source is the live motif object, any edit propagates instantly.

**Radial shell.** With Shell $s>0$ each face is extruded along the position vector by $s$ times its distance from the origin:

$$\mathbf p \;\mapsto\; \mathbf p + s\,\mathbf p .$$

Scaling every point radially preserves planarity of the flat panels while giving the surface thickness, yielding a watertight solid for 3D printing (Hart's ~4% radial extrusion).

**Guide pattern (stellation diagram).** To draw against, the operator also emits the wireframe of lines where the *other* planes of the family cut the representative plane — the pattern in Hart's 2D editor window. For a neighbouring normal $\mathbf b$, the intersection line in the plane's local $(u,v)$ coordinates has foot

$$\mathbf f = \frac{c}{\lVert (\mathbf b\!\cdot\!\mathbf u,\ \mathbf b\!\cdot\!\mathbf v)\rVert^2}\,(\mathbf b\!\cdot\!\mathbf u,\ \mathbf b\!\cdot\!\mathbf v), \qquad c = d\,(1 - \mathbf a\!\cdot\!\mathbf b),$$

clipped to a disc of radius (Guide Extent $\times d$). Motifs whose corners land on the piercings of neighbouring axes (e.g. Frabjous's rhombus corners at $\pm\varphi^2 d$ and $\pm\varphi d$, $\varphi=\tfrac{1+\sqrt5}{2}$) join their neighbours cleanly across these lines.

## References

- G. W. Hart, *Symmetric Sculpture*, Journal of Mathematics and the Arts 1(1), 2007, pp. 21–28 — <https://www.georgehart.com/sculpture/Symmetric-Sculpture.pdf> (the plane-family replication, stellation-pattern guides, and radial extrusion are modelled on the software described there).
- George W. Hart, sculpture gallery — <https://www.georgehart.com/> (the pieces referenced: *Twisted Rivers, Knotted Sea* 2001, *Tumbleweed* 2006, *Frabjous* 2006, *Whimsy* 2014).
