# Leonardo Style

![Leonardo Style](../images/leonardo.png)

## Overview

Leonardo Style is a **modifier applied to a selected existing object**, not a standalone add-mesh generator: with a mesh active, `Object > Leonardo Style` attaches a reusable Geometry Nodes group that turns any (closed) mesh into a Leonardo da Vinci style open-faced model, as in the polyhedron illustrations Leonardo drew for Luca Pacioli's *De divina proportione* (1509). Every face becomes a solid panel with a polygonal opening, joined to its neighbours along the shared edges. The render image was produced by applying the style to a base polyhedron. The `Border` and `Thickness` inputs stay editable on the modifier afterwards, and the same shared node group is reused by other Math Art generators (e.g. the zonohedron generator's Leonardo style).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Border | 0.3 | Width of the face frame as a fraction of the face (the opening is what remains); range 0.02–0.95 |
| Thickness | 0.06 | Panel thickness, extruded along the face normals; range 0.001–1.0 |

## How it works

The construction is pure Geometry Nodes, performed per face:

1. **Inset every face.** Each face is extruded individually with zero offset, producing a coincident top copy of every face; that top copy is then scaled about its own centre by a uniform factor $1 - b$, where $b$ is the **Border** fraction. The gap between the original face rim and the shrunken copy is the solid frame of width $b$ (as a fraction of the face); the shrunken interior is the future opening.

2. **Cut the openings.** The inset centre faces (the `Top` selection from the extrude) are deleted, leaving only the ring-shaped frame surface of each face.

3. **Thicken into a shell.** The remaining frame surface is extruded (non-individually) along its normals by **Thickness**, giving each frame a solid slab with an outer and inner surface plus side walls around both the outer edge and the opening.

4. **Flip the interior.** The original, untouched frame faces are now the *interior* surface of the shell, so their normals point inward. Selecting everything that is **not** a newly created `Top` or `Side` face and flipping it makes all normals point out of the solid, yielding a clean, two-sided watertight panel.

Because neighbouring faces share edges and each keeps the same border fraction, the frames meet along mitred edges, reproducing the open "vacuus" polyhedra Leonardo illustrated. Setting Border small leaves a large opening (a thin wire-frame cage); Border near 1 leaves an almost solid panel with a small hole.

## References

- Leonardo da Vinci, open-faced ("vacuus") polyhedron illustrations for Luca Pacioli, *De divina proportione*, 1509.
- G. W. Hart, *Leonardo da Vinci's Polyhedra* — <https://www.georgehart.com/virtual-polyhedra/leonardo.html>.
