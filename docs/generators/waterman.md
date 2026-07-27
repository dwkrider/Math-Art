# Waterman Polyhedra

![Waterman Polyhedra](../images/waterman.png)

## Overview

Waterman polyhedra, discovered by Steve Waterman, are the convex hulls of the clusters of spheres in a face-centred cubic (FCC) packing that fit inside a ball of a given radius about the origin. Indexed by an integer "root," they form an endless family of many-faced solids: root 1 is the cuboctahedron, and as the root grows the hull acquires more and more small facets and approaches a sphere. This generator takes the FCC lattice points within radius $\sqrt{2\cdot\text{root}}$ and returns their convex hull, with Solid / Leonardo / Wireframe styles.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Root | 10 | The radius parameter: the hull uses all FCC points with $\lvert\mathbf p\rvert^2 \le 2\cdot\text{root}$ (min 1, max 1000). |
| Style | Solid | Plain closed polyhedron, Leonardo (da Vinci) open-faced panels, or Wireframe struts along the edges. |
| Border | 0.3 | Leonardo face-frame width, as a fraction of the face (min 0.02, max 0.95). |
| Thickness | 0.05 | Panel / strut thickness for the Leonardo and Wireframe styles (min 0.001, max 1.0). |
| Scale | 1.0 | Uniform output scale (min 0.001, max 100.0). |

## How it works

The **face-centred cubic (FCC) lattice** is the set of integer points $\mathbf p = (x, y, z)$ whose coordinate sum is even:

$$x + y + z \equiv 0 \pmod 2.$$

These are the sphere centres of cubic close packing. A Waterman polyhedron of **root** $n$ collects every FCC point lying within the ball of squared radius $2n$,

$$W_n = \operatorname{conv}\{\,\mathbf p \in \text{FCC} : x^2 + y^2 + z^2 \le 2n\,\},$$

and takes the **convex hull** of that finite point set. The factor of 2 comes from the FCC packing: choosing $r^2 = 2n$ makes the shells land on the natural coordination radii of the lattice, so each integer root corresponds to a complete spherical shell of packing spheres. Root 1 keeps the twelve nearest neighbours of the origin (plus the origin), whose hull is the **cuboctahedron**; larger roots enclose more shells and yield hulls with progressively more, and smaller, faces that converge toward a sphere.

Implementation: the generator enumerates candidate integer points in the bounding box $[-m, m]^3$ with $m = \lfloor\sqrt{2n}\rfloor + 1$, keeps those satisfying both the radius and the even-sum conditions, and hands them to Blender's `bmesh` convex-hull operator. Interior points that end up unused are deleted, a limit-dissolve merges coplanar hull triangles into the true polygonal faces, normals are recalculated, and the solid is scaled so its circumradius is `Scale` (dividing coordinates by $\sqrt{2n}$).

## References

- Steve Waterman, *Waterman Polyhedra* — <http://watermanpolyhedron.com/>
- Adrian Rossiter, *Antiprism* and its `waterman` program — <https://www.antiprism.com>, <https://github.com/antiprism/antiprism> (GPL; the construction followed here).
- P. Bourke, *Waterman Polyhedra* — <http://paulbourke.net/geometry/waterman/> (illustrated description of the FCC-hull construction).
