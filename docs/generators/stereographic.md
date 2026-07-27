# Stereographic Projection Sphere

![Stereographic Projection Sphere](../images/stereographic.png)

## Overview
After Henry Segerman's stereographic-projection sculptures: a perforated spherical shell that, lit by a point light at its north pole, casts a shadow reproducing a chosen flat planar pattern (square grid, polar grid, $\{p,q\}$ tiling, beach-ball gores, or a flower lattice). The shell is watertight and boolean-free, and its pattern boundaries are exact -- meshed on strip-aligned divisions or by marching squares on a signed field, so there is no staircasing at any resolution. A **Bowl** option stops the shell just above the pattern with a solid rim and an open top.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Pattern | Square Grid | Shadow pattern: Square Grid, Polar Grid, $\{p,q\}$ Tiling, Beach Ball (lune gores), or Flower Lattice (daisies). |
| Bowl | Off | Open bowl instead of a full sphere: the shell stops just above the pattern with a solid rim band and an open top. |
| Petals | 8 | Petals per flower (Flower Lattice). Range 3-16. |
| Sphere Radius | 1.0 | Radius $R$ of the shell. Range 0.05-100. |
| Shell Thickness | 0.05 | Radial wall thickness. Range 0.001-1.0. |
| Strip Width | 0.35 | Material strip width as a fraction of the pattern cell (grid spacing, ring gap, tile edge, lune slot). Range 0.05-0.95. |
| Pattern Extent | 3.0 | Radius of the plane pattern disc in sphere radii; the shell is solid above the matching latitude (the cap holding the projection point). Range 1.0-10. |
| Grid Spacing | 0.6 | Grid line spacing in sphere radii (Square Grid / Flower). Range 0.1-5. |
| Rings | 4 | Concentric rings (Polar Grid). Range 1-24. |
| Rays | 8 | Radial rays (Polar Grid). Range 1-64. |
| p | 4 | Tile polygon sides (Tiling; 2 = hosohedron). Range 2-5. |
| q | 3 | Tiles per vertex (Tiling; needs $1/p+1/q>1/2$, clamped if not). Range 3-8. |
| Gores | 8 | Number of solid lunes (Beach Ball). Range 2-64. |
| Resolution | 48 | Latitude rows in the pattern zone (longitudes $=3\times$). Range 16-192. |
| Add Point Light | Off | Add a small point light at the north pole to preview the shadow. |
| Add Floor Plane | Off | Large plane under the sphere so the projected shadow is visible. |
| Floor Size | 40.0 | Floor plane side length, in sphere radii. Range 4-400. |

## How it works

**Stereographic projection $S^2\to$ plane.** The sphere of radius $R$ rests with its south pole at the origin, so its centre is $C=(0,0,R)$ and its north pole (the projection point) is $N=(0,0,2R)$. Stereographic projection from $N$ onto the plane $z=0$ sends a sphere point at polar angle $\theta$ (measured from $N$) to plane radius
$$r = 2R\cot\!\frac{\theta}{2},\qquad\text{inversely}\qquad \theta = 2\arctan\!\frac{2R}{r}.$$
This is the conformal map behind the sculptures: with a point light at $N$, a ray through a hole in the shell continues to exactly the projected plane point, so the shadow *is* the plane pattern. A flat pattern clipped to a disc of radius $R_{\max}=\text{extent}\cdot R$ becomes a perforation pattern below the latitude $\theta_{\min}=2\arctan(2R/R_{\max})$; everything above stays solid -- the cap that surrounds the projection point, without which the shell would fall apart.

**Watertight construction.** The kept (material) faces are duplicated at radii $R\pm t/2$ from $C$, and every boundary edge of the kept region is closed with a side-wall quad, so the solid is closed by construction -- no boolean is needed. Two guards keep it 2-manifold: pole fan rows are forced uniform, and a cleanup pass fills "checkerboard" cells (material touching only diagonally, which would put four walls on one vertical edge). Because the map is conformal but not equiareal, the latitude rows for the plane patterns are spaced uniformly in *plane radius* $r$, so cells are evenly resolved where the pattern actually lives.

**Pattern classification.**
- *Square grid* is meshed on a Cartesian grid whose division lines fall exactly on the strip edges, so every cell is wholly material or wholly hole -- exact boundaries, no staircase.
- *Polar grid* and *beach ball* use latitude/longitude divisions aligned to the ring/ray/lune edges; a small solid cap at the south pole keeps converging rays and gore tips joined.
- *$\{p,q\}$ tiling* keeps material along the great-circle edges of a spherical Platonic edge graph. Only spherical Schlaefli symbols are valid ($1/p+1/q>1/2$); the code clamps otherwise. The signed field is the angular distance from each sphere point to the nearest tiling arc, minus half the strip width.
- *Flower lattice* is a hexagonal lattice of petal-shaped elliptical holes; its signed field (material between petals) is marched on a Cartesian grid at a feature-sized step.

The tiling and flower patterns are extracted by **marching squares** on their signed fields, clipping cells at the zero crossings so the hole boundaries come out as smooth level sets. For bowls, the solid rim band $r\in[R_{\max},1.15\,R_{\max}]$ and the open trim are folded into the field as additional level sets.

## References

- H. Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 (chapter 3, figs 3-11..3-14 -- the stereographic-projection shadow sculptures).
- H. Segerman, "Stereographic projection" sculptures, https://www.segerman.org/.
- Stereographic projection as a conformal map $S^2\setminus\{N\}\to\mathbb R^2$: standard complex analysis / differential geometry.
