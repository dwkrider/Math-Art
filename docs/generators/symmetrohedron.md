# Symmetrohedron

![Symmetrohedron](../images/symmetrohedron.png)

## Overview

Symmetrohedra, introduced by Craig Kaplan and George Hart, are polyhedra assembled by placing regular polygons symmetrically on the rotation axes of a symmetry group and letting the convex hull fill the gaps between them. This generator inscribes each polygon in the unit sphere, gives every axis class (5/3/2 for icosahedral, 4/3/2 for octahedral, 3/3/2 for tetrahedral) its own polygon multiplier, size, and phase, and hulls the whole arrangement. Sweeping the size sliders passes through the exact edge-to-edge Kaplan–Hart solutions as well as a continuum of decorative in-between forms.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Symmetry | Icosahedral (5,3,2) | The symmetry group whose axes carry the polygons: Icosahedral, Octahedral (4,3,2), or Tetrahedral (3,3,2). |
| Axis-1 Multiplier | 1 | Polygon on the first (highest-order) axis class has `multiplier × axis-order` sides; 0 = no polygon there (min 0, max 4). |
| Axis-2 Multiplier | 1 | Same, for the second axis class (min 0, max 4). |
| Axis-3 Multiplier | 0 | Same, for the third axis class (min 0, max 4). |
| Axis-1 Size | 0.45 | Radius of the first class's polygons (fraction of the unit sphere; min 0.05, max 0.98). |
| Axis-2 Size | 0.31 | Radius of the second class's polygons (min 0.05, max 0.98). |
| Axis-3 Size | 0.3 | Radius of the third class's polygons (min 0.05, max 0.98). |
| Axis-1 Phase | 0.0 | In-plane rotation of the first class's polygons, in degrees (min -90, max 90). |
| Axis-2 Phase | 0.0 | In-plane rotation of the second class's polygons (min -90, max 90). |
| Axis-3 Phase | 0.0 | In-plane rotation of the third class's polygons (min -90, max 90). |
| Coloring | By Face Size | One material per face size (shared Conway palette), or None. |
| Style | Solid | Plain closed polyhedron, Leonardo (da Vinci) open-faced panels, or Wireframe struts. |
| Border | 0.3 | Leonardo face-frame width, as a fraction of the face (min 0.02, max 0.95). |
| Thickness | 0.05 | Panel / strut thickness for the Leonardo and Wireframe styles (min 0.001, max 1.0). |
| Scale | 1.0 | Uniform output scale (min 0.01, max 100.0). |

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/symmetrohedron__I.png" width="200"><br><sub>Icosahedral</sub></td>
<td align="center"><img src="../images/variants/symmetrohedron__O.png" width="200"><br><sub>Octahedral</sub></td>
<td align="center"><img src="../images/variants/symmetrohedron__T.png" width="200"><br><sub>Tetrahedral</sub></td>
</tr>
</table>

## How it works

Each of the three polyhedral rotation groups has three classes of rotation axis, named by their order:

- **Tetrahedral T** — four 3-fold axes (cube diagonals) and three 2-fold axes (coordinate axes).
- **Octahedral O** — three 4-fold, four 3-fold, six 2-fold axes.
- **Icosahedral I** — six 5-fold, ten 3-fold, fifteen 2-fold axes.

The axis directions are generated from standard coordinates (the 5-fold axes from cyclic permutations of $(0,1,\varphi)$, and so on) and de-duplicated so each $\pm$-axis is represented once, then both ends are returned.

On every axis of a class with a nonzero **multiplier** $m$, the generator places a regular polygon with $m \times (\text{axis order})$ sides — e.g. multiplier 1 on the 5-fold axes of $I$ gives pentagons; multiplier 2 gives decagons. Each polygon is **inscribed in the unit sphere**: with a chosen radius $r$ (the size slider), its centre sits at height $h = \sqrt{1 - r^2}$ along the axis $\mathbf u$, so its vertices

$$\mathbf p_i = h\,\mathbf u + r\big(\cos\alpha_i\,\mathbf e_1 + \sin\alpha_i\,\mathbf e_2\big),\qquad \alpha_i = \phi + \tfrac{2\pi i}{m\cdot\text{order}},$$

all lie on the sphere. The in-plane frame $(\mathbf e_1, \mathbf e_2)$ is chosen **covariantly**: vertex 0 is aimed at the projection of a neighbouring symmetry axis onto the polygon's plane, so that all copies of the polygon are related by the group and the assembly is genuinely symmetric. (A polygon whose side count is a multiple of the axis order is invariant under that axis's stabilizer, so this alignment is consistent whichever tied neighbour is picked.) The `Phase` slider adds an extra in-plane twist $\phi$.

All the placed polygon vertices are then fed to a **convex hull**. The polygons themselves survive as faces of the hull; the gaps between them close up with the incidental faces the hull creates. A limit-dissolve merges coplanar hull triangles so genuine polygons read as single faces. At special size ratios the polygon edges meet exactly edge-to-edge — these are the true Kaplan–Hart symmetrohedra — while intermediate sizes give the smoothly varying family the sliders explore. Faces are tagged with an `ngon_sides` attribute and (optionally) colored by side count using the shared Conway palette.

## References

- C. S. Kaplan and G. W. Hart, *Symmetrohedra: Polyhedra from Symmetric Placement of Regular Polygons*, Bridges 2001 Conference Proceedings, pp. 21–28 — <https://www.georgehart.com/> / <http://archive.bridgesmathart.org/2001/bridges2001-21.html>
- G. W. Hart, *Virtual Polyhedra* — <https://www.georgehart.com/virtual-polyhedra/vp.html> (background on symmetry groups and their axes).
