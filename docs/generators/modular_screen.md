# Modular Screen (Hauer-Carlberg)

![Modular Screen](../images/modular_screen.png)

## Overview
A perforated architectural screen wall in the modular-constructivist tradition of **Erwin Hauer** and **Norman Carlberg**. One curvilinear saddle module is tiled seamlessly across a rectangle; because neighbouring cells share their boundary curves, the tiled midsurface is a single continuous undulating sheet -- the "undulating webbing" of a Hauer screen. That sheet is thickened into a solid slab and perforated with a smooth aperture through each module. The result is a single watertight, orientable manifold suitable for rendering or 3D printing. The saddle surface is the essential ingredient: its opposite curvatures let a module's boundary match its neighbour's on all four sides, so a single unit propagates without ever closing the form. The interwoven **Bilayer Weave (after Design 1)** -- which no single-valued height field can express -- is built instead as two families of undulating ribbons woven over and under one another. A finished screen can be left flat, **curved** onto a cylinder, or wrapped into a **column**, as in the built installations.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Design | Saddle Lattice (after Design 5) | Saddle Lattice (`cos·cos` egg-crate), **Pinwheel (chiral four-fold swirl)**, **Bilayer Weave (after Design 1 -- woven ribbons)**, Woven Undulation (`cos x + cos y`), Diagonal Brace (`cos(x+y)·cos(x-y)`), Relief Wall (after Design 6 -- solid, no perforations), or Hypar (Carlberg -- ruled hyperbolic-paraboloid modules with sharp creases). |
| Swirl | 1.0 | How strongly the pinwheel arms curl (Pinwheel only). Range 0-2.5. |
| Handedness | Right | Chirality of the pinwheel swirl -- Right or Left (mirror) (Pinwheel only). Set Aperture to 0 for the pure relief. |
| Cells X / Cells Y | 5 / 5 | Modules across and down. Range 2-24 each. |
| Relief Depth | 0.5 | Undulation amplitude of the module. Range 0.02-2.0. |
| Thickness | 0.14 | Wall / ribbon thickness. Range 0.02-0.6. |
| Aperture | 0.34 | Aperture radius per module in cell units (0 = solid wall). Range 0-0.47. Ignored by the Relief Wall and Bilayer Weave designs. |
| Aperture Squareness | 0.0 | Morph each aperture from a circle (0) to a rounded square (1) via a superellipse -- rounded-square openings, and polygonal openings for the Carlberg design. |
| Solid Border | On | Leave the outer ring of modules unperforated as a frame. |
| Ribbon Width | 0.7 | Width of the woven ribbons in cell units (Bilayer Weave only). Range 0.1-0.95. |
| Resolution | 6 | Samples per cell; higher gives rounder apertures and smoother webbing. Range 3-12. |
| Rim Bulge | 0.6 | Round the aperture and border edges into a bull-nose (0 = square-cut vertical rim). Range 0-1. |
| Rim Segments | 3 | Facets across the rounded rim. Range 2-6. |
| Form | Flat Panel | Flat Panel, Curved Screen (bent onto a cylinder through the Wrap Angle), or Column (wrapped a full turn -- join the seam with Merge by Distance for a watertight tube). |
| Wrap Angle | 120 | Arc the Curved Screen spans, in degrees. Range 10-350. |
| Smooth Shading | On | Shade the slab smooth. |
| Scale | 1.0 | The screen is fitted within a $2\times$ Scale cube at the origin. Range 0.01-100. |

## Variants

Renders of each Design:

<table>
<tr>
<td align="center"><img src="../images/variants/modular_screen__DESIGN5.png" width="200"><br><sub>Saddle Lattice (Design 5)</sub></td>
<td align="center"><img src="../images/variants/modular_screen__PINWHEEL.png" width="200"><br><sub>Pinwheel (chiral)</sub></td>
<td align="center"><img src="../images/variants/modular_screen__DESIGN1.png" width="200"><br><sub>Bilayer Weave (Design 1)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/modular_screen__WEAVE.png" width="200"><br><sub>Woven Undulation</sub></td>
<tr>
<td align="center"><img src="../images/variants/modular_screen__DIAGONAL.png" width="200"><br><sub>Diagonal Brace</sub></td>
<td align="center"><img src="../images/variants/modular_screen__DESIGN6.png" width="200"><br><sub>Relief Wall (Design 6)</sub></td>
<td align="center"><img src="../images/variants/modular_screen__HYPAR.png" width="200"><br><sub>Hypar (Carlberg)</sub></td>
</tr>
</table>

## How it works

**Tiled saddle midsurface.** The midsurface is a periodic height field $h(x,y)$ of unit period. For the smooth designs it is a doubly-periodic saddle field -- $h=\text{amp}\cdot\cos\pi x\,\cos\pi y$ (Saddle Lattice), $\tfrac{\text{amp}}2(\cos\pi x+\cos\pi y)$ (Weave), or $\text{amp}\cdot\cos\pi(x{+}y)\,\cos\pi(x{-}y)$ (Diagonal). Each is $C^1$ and continuous across cell boundaries, so tiling it produces a single seamless undulating sheet.

**Hypar (Carlberg) modules.** The Hypar design replaces the smooth field with a ruled **hyperbolic paraboloid** per cell, $h=\text{amp}\cdot\xi\eta$ on local coordinates $\xi,\eta\in[-1,1]$, multiplied by a checkerboard sign $(-1)^{i+j}$. The sign makes adjacent cells meet exactly along their shared edge while leaving a **sharp crease** there -- Carlberg's harder-edged, straight-ruled manner, as opposed to Hauer's smooth membranes.

**Bilayer Weave (Design 1).** Hauer's Design 1 interweaves *two* saddle layers, which no single-valued height field $h(x,y)$ can represent. It is built instead as a plain weave of ribbons: **warp** ribbons run along $x$ at $y=j+\tfrac12$ with height $\text{amp}\,(-1)^j\sin\pi x$, **weft** ribbons run along $y$ at $x=i+\tfrac12$ with the opposite phase $-\text{amp}\,(-1)^i\sin\pi y$. At every crossing the two land at $\pm\text{amp}$, so one rides over and the other under, alternating across the lattice. Each ribbon is a closed swept tube (rectangular section), so the woven screen is watertight by construction; the layers stay clear of one another as long as $2\,\text{amp}>t$, which the builder enforces. Wider ribbons close the openings; the width control sets the openness.

**Pinwheel (chiral).** The pinwheel module is a chiral height field of wallpaper group **p4** (four-fold rotation, no mirror): a mirror-symmetric base $\cos2\pi x+\cos2\pi y$ plus a **swirl** term $\sin4\pi x\,\sin2\pi y-\sin4\pi y\,\sin2\pi x$ that is *antisymmetric* under reflection. Because no mirror can undo the swirl, the arms all curl one way; the **Handedness** flips the sign of the swirl term (right/left), and **Swirl** scales it. Left at its default relief (Aperture 0) it is a solid undulating wall of interlocking pinwheels; raising the aperture opens the deep valleys at the cell centres. Unlike the achiral saddle styles, this is the module that reproduces the swirling one-handed relief of chiral screen facades.

**Apertures: circle to square.** Each aperture is traced along the ray to every perimeter sample as a **superellipse** $|X/r|^{p}+|Y/r|^{p}=1$, with $p=2+6\cdot(\text{squareness})$ running from a circle ($p=2$) to a rounded square ($p=8$) -- giving the softer rounded-square openings of some Hauer screens and the polygonal openings of the Carlberg modules.

**Flat, curved, column.** After the flat screen is built it can be bent onto a cylinder whose axis is vertical: the width $x\in[0,n_x]$ wraps through the chosen angle and the slab's relief $z$ becomes radial, at arc radius $R=n_x/\theta$. A full turn wraps the panel into a **column** (the seam is left coincident -- Merge by Distance closes it into a watertight tube). The wrap is a per-vertex map, so it preserves the mesh topology and watertightness.

**Traced apertures.** Each unit cell is meshed as a radial fan: its square perimeter is sampled uniformly (so adjacent cells share their common-edge samples exactly and weld with no seam), and rings are interpolated inward along the ray to each sample -- ending either on a central **aperture circle** of the chosen radius, or, for a solid cell, at the cell centre (a triangle fan). Because the aperture is traced as a true circle rather than cut out of a square grid, the perforations stay smooth at any resolution.

**Thickening and rims.** Two offset copies of the meshed sheet at $h\pm t/2$ form the slab's top and bottom faces. Every boundary edge of the top sheet -- the panel perimeter and every aperture -- raises a wall down to the matching bottom edge, closing the shell. With **Rim Bulge** on, each wall is rounded into a bull-nose: a per-boundary-vertex outward normal (the mean of its incident boundary-edge normals) drives an arc of intermediate loops, $z=h+\tfrac t2\cos\varphi$ offset outward by $\text{bulge}\cdot\tfrac t2\sin\varphi$. Because the intermediate vertices are shared between a boundary vertex's two wall strips, the rounded rim stays watertight.

**Watertightness.** Perimeter samples are deduplicated across cells, so the tiled sheet has no internal seams; every interior edge is shared by exactly two faces, and every boundary edge receives exactly one wall. The self-test confirms that each Design builds a closed 2-manifold (no edge shared by other than two faces) whose Euler characteristic gives the expected number of through-holes.

## References

- Erwin Hauer, *Continua -- Architectural Screens and Walls*, Princeton Architectural Press, 2004 -- the screen designs (numbered Design 1-7, 1950-57) and Hauer's saddle-surface method.
- Norman Carlberg and Erwin Hauer -- co-originators of Modular Constructivism (units designed to be multiplied); Carlberg's ruled, hard-edged saddle modules.
- H. F. Scherk, "Bemerkungen über die kleinste Fläche innerhalb gegebener Grenzen", *J. reine angew. Math.* **13** (1835), 185-208 -- the doubly-periodic minimal surface behind the saddle lattice.
- A. H. Schoen, *Infinite Periodic Minimal Surfaces Without Self-Intersections*, NASA TN D-5541, 1970 -- the I-WP surface later identified with Hauer's fully three-dimensional modules.
