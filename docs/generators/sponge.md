# Fractal Sponge

![Fractal Sponge](../images/sponge.png)

## Overview

Subtractive volume fractals: the Menger sponge and its relatives, built by recursively keeping a subset of grid subcells, plus the point-contact Sierpinski solids.

The eight kinds fall into two families. **Grid sponges** (Menger, Vicsek, Sierpinski carpet, both Mosely snowflakes, and Cantor dust) divide a cube into a $3\times3\times3$ grid, keep a fixed subset of the 27 subcells, and repeat inside every survivor; they emit **only their exterior faces**, so the result is a single watertight surface even at high recursion levels — Cantor dust is the lone exception, its corner-only cells never touch, so it comes out as a cloud of separate cubes. **Corner fractals** (Sierpinski tetrahedron and octahedron) instead place shrinking copies at a solid's *vertices*, so neighbouring copies meet only at points.

### Using it

1. **Add it** from *Add ▸ Mesh ▸ Math Art ▸ Fractals ▸ Fractal Sponge*.
2. **Pick the Fractal.** The grid sponges are **Menger Sponge** (drill out the body-centre and the six face-centres — the classic sponge), **Vicsek Fractal** (keep the centre plus its six face neighbours, a 3-D plus sign), **Sierpinski Carpet** (the flat, one-cell-thick $3\times3$ carpet), **Mosely Snowflake** (drop the eight corners), **Mosely Snowflake (light)** (drop the corners *and* the body centre, leaving a hollow), and **Cantor Dust** (keep only the eight corners — a scattering of cubes). The corner fractals are **Sierpinski Tetrahedron** (4 corner copies) and **Sierpinski Octahedron** (6 corner copies).
3. **Set the Level** — the recursion depth, the one shape knob; each step replaces every cell (or copy) with a smaller block of the same rule, so detail multiplies fast. Each kind has its own cap so the mesh stays manageable (Menger 4, Vicsek 5, Carpet 6, the two Mosely kinds 3, Cantor 4, Tetra 6, Octa 5); ask for more and it is clamped down with a warning.
4. **Set the Size** — the overall extent of the fractal. There are no mode-specific controls: every kind is driven by just these three properties.
5. **Read the output.** Grid sponges arrive as one watertight surface (Cantor dust as separate cubes); the corner fractals arrive as many closed solids touching at points. The operator reports the level it actually built and the final vertex/face counts, and prints a warning line whenever it had to cap the level you asked for.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Fractal | Menger Sponge | Menger Sponge (3×3×3, centre and face-centre cells removed), Sierpinski Tetrahedron (4 corner copies), Sierpinski Octahedron (6 corner copies), Vicsek Fractal (3D plus sign: centre + 6 face neighbours), or Sierpinski Carpet (flat 3×3, one cell thick) |
| Level | 3 | Recursion depth (grid sponges are capped per kind to keep meshes manageable) |
| Size | 2.0 | Overall size of the fractal |

*Per-kind level caps: Menger 4, Vicsek 5, Carpet 6, Tetra 6, Octa 5.*

## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/sponge__MENGER.png" width="200"><br><sub>Menger Sponge</sub></td>
<td align="center"><img src="../images/variants/sponge__TETRA.png" width="200"><br><sub>Sierpinski Tetrahedron</sub></td>
<td align="center"><img src="../images/variants/sponge__OCTA.png" width="200"><br><sub>Sierpinski Octahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/sponge__VICSEK.png" width="200"><br><sub>Vicsek Fractal</sub></td>
<td align="center"><img src="../images/variants/sponge__CARPET.png" width="200"><br><sub>Sierpinski Carpet</sub></td>
<td align="center"><img src="../images/variants/sponge__MOSELY.png" width="200"><br><sub>Mosely Snowflake</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/sponge__MOSELYL.png" width="200"><br><sub>Mosely Snowflake (light)</sub></td>
<td align="center"><img src="../images/variants/sponge__CANTOR.png" width="200"><br><sub>Cantor Dust</sub></td>
</tr>
</table>

## How it works

**In plain terms.** Picture a Rubik's cube: one big cube built from 27 little ones. Now pick a *rule* for which little cubes to keep — the Menger rule, for instance, says "throw away the one in the very middle and the one in the middle of each face, keep the other twenty." Then treat every little cube you kept as a fresh Rubik's cube and apply the same rule again, and again, forever. What you end up with is riddled with square holes at every scale — the closer you look, the more holes you see, exactly like the holes you already saw but smaller. That "same pattern at every zoom level" is what makes it a *fractal*, and it is why a sponge can have almost no solid volume yet an endless amount of surface. The Sierpinski tetrahedron and octahedron use the opposite trick: instead of drilling holes, they shrink the whole shape and glue copies of it to the *corners* of a pyramid, so the copies kiss at single points. Everything below turns "which little cubes to keep" into an exact rule the code can iterate, and counts how much survives — which is what pins down the dimension.

**The digit rule.** Divide the unit cube into a $3\times3\times3$ grid and index each subcell by $(x,y,z)$ with each coordinate in $\lbrace0,1,2\rbrace$. Let $\text{mid}=(x{=}1,\,y{=}1,\,z{=}1)$ record which coordinates sit at the centre. Every grid sponge is then one predicate on that digit vector — a hand-listed cell set would obscure the pattern, but as a predicate the whole family is a single rule applied recursively, each surviving cell subdividing into its own $3\times3\times3$ block:

- **Menger sponge:** keep a subcell unless it shares two or more central coordinates ($\text{sum}(\text{mid})<2$) — drop the body-centre and the six face-centres. $20$ of $27$ survive, so the level-$n$ sponge has $20^n$ cells; its Hausdorff dimension is $\dfrac{\ln 20}{\ln 3}\approx2.727$, just under a solid's $3$, which is the arithmetic statement of "almost all volume drilled away."
- **Vicsek fractal:** keep only the cells with two or more central coordinates ($\text{sum}(\text{mid})\ge2$) — the centre plus the six face neighbours, a 3-D plus sign. $7$ cells, dimension $\dfrac{\ln 7}{\ln 3}\approx1.771$: the thin cross keeps far less than the sponge, so its dimension drops below $2$.
- **Sierpinski carpet:** one cube thick ($z=0$), keeping the eight cells that are not the planar centre, $8$ per step, dimension $\dfrac{\ln 8}{\ln 3}\approx1.893$.
- **Mosely snowflake:** drop the eight corners ($\text{sum}(\text{mid})\ge1$), $19$ cells, dimension $\dfrac{\ln 19}{\ln 3}\approx2.679$; the **light** variant also drops the body centre ($1\le\text{sum}(\text{mid})\le2$), leaving $18$ cells with a hollow at every stage, dimension $\dfrac{\ln 18}{\ln 3}\approx2.631$.
- **Cantor dust:** keep only the eight corners ($\text{sum}(\text{mid})=0$), the 3-fold product of the Cantor middle-thirds set, dimension $\dfrac{\ln 8}{\ln 3}\approx1.893$. Because corner cells are never face-adjacent, the copies never touch — the sponge disintegrates into a cloud of separate cubes.

At level $n$ the cells are integer coordinates in a $3^n$ grid. To keep the mesh watertight the generator emits **only exterior faces** — a face of a solid cell is written only when the neighbouring cell in that direction is empty — and dedupes shared corner vertices, so two abutting cells fuse into one surface rather than burying a wall between them. The self-test confirms every edge is shared by exactly two faces (manifold). Cantor dust is the honest exception: with no cell ever adjacent to another, *every* face is exterior, and the "surface" is genuinely a set of disjoint boxes.

**Corner sponges (Sierpinski tetrahedron / octahedron).** These are point-contact fractals: at each step every existing copy is replaced by half-scale copies placed at the *vertices* of the seed solid, so where a grid sponge removes material a corner fractal instead clusters shrunken selves at the extremities. Starting from a centre set $\lbrace0\rbrace$ and halving the scale each step,

$$c \ \longmapsto\ \bigl\lbrace\, c + s\,v \ :\ v\in\text{vertices}\,\bigr\rbrace,\qquad s\to s/2,$$

gives $4$ copies per step for the tetrahedron (its 4 vertices) and $6$ for the octahedron (its 6). Because the copies meet only at the tangent points where two vertices coincide, each smallest copy is emitted as its own closed solid: the level-$n$ tetrahedron has $4^n$ solids and $4\cdot4^n$ faces, the octahedron $6^n$ solids and $8\cdot6^n$ faces. The Sierpinski tetrahedron has Hausdorff dimension $\dfrac{\ln 4}{\ln 2}=2$ exactly — four half-size copies — so despite living in 3-D it is, in the fractal sense, precisely a surface.

## References

- Menger sponge: Karl Menger, "Allgemeine Raeume und Cartesische Raeume", Proc. Akad. Wetensch. Amsterdam 29, 1926, pp. 1125-1128.
- Sierpinski carpet (and its 3D tetra/octa analogues): Waclaw Sierpinski, C. R. Acad. Sci. Paris 162, 1916, pp. 629-632.
- Vicsek fractal: Tamas Vicsek, "Fractal models for diffusion controlled aggregation", J. Phys. A 16, 1983, pp. L647-L652.
- Mosely snowflake: named for Jeannine Mosely (of business-card Menger-sponge fame); the Sierpinski-Menger snowflake family is surveyed in M. Kalinski, "On the variations of the Sierpinski and Menger sponges and the Mosely snowflake" (2017).
- Cantor dust: the 3-fold Cartesian product of Georg Cantor's set, "Ueber unendliche, lineare Punktmannigfaltigkeiten V", Math. Ann. 21, 1883, pp. 545-591.
- Self-similar dimension: Benoit B. Mandelbrot, "The Fractal Geometry of Nature", W. H. Freeman, 1982.

