# Symmetric Sculpture

![Symmetric Sculpture](../images/symmetric_sculpture.png)

## Overview

A Blender adaptation of George W. Hart's sculpture-design program — the tool behind pieces such as *Frabjous*, *Whimsy*, *Twisted Rivers* and *Tumbleweed*. You draw a flat **motif** in **one** representative plane, and it is replicated live into a whole family of symmetrically arranged planes: the extended face planes of an icosahedron, dodecahedron, rhombic triacontahedron or hexecontahedron (or their octahedral / tetrahedral analogues). The replication runs in a Geometry Nodes modifier, so editing the motif object — in edit mode, or by moving/rotating it — updates every copy in real time.

It ships three ready presets (**Frabjous**, **Krull**, **Whimsy**) plus a fully **Custom** mode. Each preset drops in a starting motif together with the matching symmetry group and plane family; you then reshape that one motif to taste and the sculpture follows.

### Using it

1. **Add it** from *Add ▸ Mesh ▸ Math Art ▸ Surfaces ▸ Symmetric Sculpture*. The operator builds a small scene: the editable **motif** object plus the instanced result carrying the Geometry Nodes modifier.
2. **Pick a preset** in the redo panel (*Adjust Last Operation*). Frabjous, Krull and Whimsy each set the group, plane family and a starting motif; **Custom** lets you choose the **Symmetry Group** (icosahedral 60 / octahedral 24 / tetrahedral 12) and **Plane Family** yourself and starts from a plain demo arc.
3. **Shape the motif.** The whole sculpture is instanced from that single motif object — select it and edit its vertices (or move/rotate the object) and every symmetric copy updates instantly. A faint **guide pattern** is drawn in the motif's plane showing where the neighbouring planes cut through; land the motif's edges and corners on those lines and the parts meet cleanly across planes. (This is Hart's 2-D stellation-diagram editor, reproduced live.)
4. **Give it thickness.** Raise **Shell** (a radial extrusion fraction — Hart used ~4%) to turn the flat panels into a watertight solid for 3-D printing; leave it at 0 for flat planes.
5. **Bring your own motif.** Set **Motif Object** to the name of an existing flat mesh (drawn in its local XY plane) to instance it instead of the demo arc.
6. **Render or export.** Leave **Translucent Copies** off to show the sculpture in its real material (on is the ghost *design view* that keeps the one editable copy distinct). For fabrication, turn on **Build Machinable Part** to drop a single solid part — edges mitred to half the dihedral — laid out flat below the plane, ready to cut or print. **Show Defining Polyhedron** adds a design aid: the solid whose face planes this family extends, with every guide crossing marked so you can see where parts converge.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Preset | Frabjous | Sculpture setup; the motif stays editable afterwards |
| — Frabjous | | After *Frabjous* (2003): 30 S-shaped parts in the face planes of the great rhombic triacontahedron — ends meeting in threes at the 3-fold corners ($\varphi^2 \times$ the plane distance), five-fold vortices at the 5-fold corners ($\varphi \times$) |
| — Krull | | A small stellated dodecahedron $\{5/2, 5\}$: 12 five-armed stars in the dodecahedral planes, traced from the cutting template. The arms end on the crossings at twice the plane distance, where five planes meet, so each of the solid's 12 spikes is built from five arms |
| — Whimsy | | After *Whimsy* (2014): 60 flat blades in the face planes of the pentagonal hexecontahedron (snub dodecahedron dual) — a curved band with two teardrop openings, five to a hub and three at each 3-fold corner |
| — Custom | | Choose the group and plane family yourself; starts from the demo arc motif |
| Symmetry Group | Icosahedral (60) | Rotation group: Icosahedral (60), Octahedral (24), or Tetrahedral (12). Shown only in Custom mode |
| Plane Family | 3-fold planes | Which orbit of planes to fill. Shown only in Custom mode |
| — 5-fold planes (dodecahedral) | | 12 planes, 5-fold symmetry in each (icosahedral group only) |
| — 4-fold planes (cube) | | 6 planes, 4-fold each (octahedral group only) |
| — 3-fold planes (icosahedral) | | Planes perpendicular to the 3-fold axes: 20 icosahedral, 8 octahedral, 4 tetrahedral |
| — 2-fold planes (triacontahedral) | | Planes perpendicular to the 2-fold axes: 30 icosahedral, 12 octahedral, 6 tetrahedral |
| — General planes (hexecontahedral) | | A full-orbit family with no in-plane symmetry: 60 / 24 / 12 planes |
| Plane Distance | 1.0 | Distance of the plane family from the origin |
| Shell | 0.04 | Radial extrusion fraction (Hart used ~4%); 0 keeps the flat planes, negative extrudes the other way, towards the origin |
| Guide Extent | 3.2 | Radius of the guide pattern, in units of the plane distance. 3.2 covers every line of the 5-, 4-, 3- and 2-fold families (the outermost sits at 3.078), so the outer corners a motif snaps to — e.g. Frabjous's tips at $\varphi^2\approx2.618$ — have guide lines crossing there. The general (P1) families run much farther out and need a bigger radius |
| Guide Rings | 0 | Thin the guide pattern to its innermost N rings of lines (0 = draw them all); the lines sit at a few distinct distances, each ring one symmetry-equivalent set, so dropping outer rings clears clutter |
| Lift Sculpture Clear of Motif | On | Raise the sculpture up $+Z$ so it stops surrounding the motif and guides (which stay at the origin to edit); the height scales with the sculpture, and is a live **Lift** input on the modifier afterwards |
| Build Machinable Part | Off | Add one part as a solid of the Shell thickness, every edge that beds against a neighbour cut to half the dihedral so the two butt cleanly; laid flat below the XY plane, ready to export for cutting/printing (the dihedral angles are reported) |
| Show Defining Polyhedron | Off | Add the semi-transparent solid whose extended face planes are this plane family, marking each guide-line crossing with a coloured disc (a crossing of $k$ lines is where $k{+}1$ planes and parts meet); colour = the orbit under the group. Design aid, excluded from renders |
| Translucent Copies | Off | Draw the replicated copies in a ghost material so the editable motif reads through them; off = the sculpture's real material |
| Motif Object | *(none)* | Name a mesh object to replicate instead of the preset motif — draw it flat on the XY plane. Empty builds the preset's own motif |

The Geometry Nodes modifier keeps several of these **live** on the result object, so you can tune them after the fact without re-running the operator: **Shell** (radial extrusion fraction), **Lift** (height the sculpture is raised to clear the motif and guides), **Weld** (merge distance where copies meet, default $10^{-4}$), the **Translucent Copies** design-view toggle, **Copy Material**, and the **Plane Rotation** / **Plane Offset** vectors that map the motif's XY plane onto the representative plane.

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/symmetric_sculpture__FRABJOUS.png" width="200"><br><sub>Frabjous</sub></td>
<td align="center"><img src="../images/variants/symmetric_sculpture__KRULL.png" width="200"><br><sub>Krull</sub></td>
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

**The preset motifs.** Each named preset is a single flat motif traced onto that guide pattern, then handed to the same replication machinery as a Custom motif — nothing about a preset is special once it is built, so you can freely reshape it. **Frabjous** places one S-curve whose two ends reach the neighbouring-plane crossings at $\varphi^2 d$ and $\varphi d$, so under the icosahedral group the 30 copies interlock three-to-a-corner and five-to-a-vortex. **Krull** traces the outline of a small stellated dodecahedron $\{5/2,5\}$ directly from its stellation diagram in the dodecahedral plane: a five-pointed star whose arms reach the crossings at $2d$, so the 12 copies assemble the solid's 12 five-arm spikes. **Whimsy** is a curved band with two teardrop cut-outs, sized so five meet at each 5-fold hub and three at each 3-fold corner of the hexecontahedral family. Because all three are built by the same "draw on the guide lines" method Hart used, **Custom** is not a lesser mode — it is the general case, and the presets are just worked examples to start from.

## References

- G. W. Hart, "Symmetric sculpture", J. Mathematics and the Arts 1(1), 2007, 21-28.  doi:10.1080/17513470701228040
- G. W. Hart, "Sculpture from Symmetrically Arranged Planar Components", Bridges 2003, 315-322.
- H. S. M. Coxeter, P. Du Val, H. T. Flather, J. F. Petrie, "The Fifty-Nine Icosahedra", U. Toronto Press, 1938 -- the stellation pattern drawn as the editing guides.

