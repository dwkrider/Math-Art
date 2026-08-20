# Fractal Sponge

![Fractal Sponge](../images/sponge.png)

## Overview

Subtractive volume fractals: the Menger sponge and its relatives, built by recursively keeping a subset of grid subcells, plus the point-contact Sierpinski solids. Grid sponges (Menger, Vicsek, Sierpinski carpet) emit only their exterior faces, so the result is a single watertight surface even at high recursion levels; the corner fractals (Sierpinski tetrahedron and octahedron) place shrinking copies at a solid's vertices, meeting at points.

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

**Grid sponges (Menger / Vicsek / carpet).** A unit cube is divided into a $3\times3\times3$ grid and a fixed rule decides which of the 27 subcells survive one step; the rule is applied recursively, each surviving cell subdividing into its own $3\times3\times3$ block. Writing $\text{mid}=(x{=}1,\,y{=}1,\,z{=}1)$ for whether each coordinate is central,

- **Menger sponge:** keep a subcell unless it shares two or more central coordinates — drop the body-centre and the six face-centres. $20$ of $27$ survive, so the level-$n$ sponge has $20^n$ cells; its Hausdorff dimension is $\dfrac{\ln 20}{\ln 3}\approx2.727$.
- **Vicsek fractal:** keep the centre plus the six face neighbours (a 3D plus sign), $7$ cells, dimension $\dfrac{\ln 7}{\ln 3}\approx1.771$.
- **Sierpinski carpet:** one cube thick ($z=0$), keeping the eight cells that are not the planar centre, $8$ per step, dimension $\dfrac{\ln 8}{\ln 3}\approx1.893$.

At level $n$ the cells are integer coordinates in a $3^n$ grid. To keep the mesh watertight the generator emits **only exterior faces** — a face of a solid cell is written only when the neighbouring cell in that direction is empty — with shared vertices deduped. The self-test confirms every edge is shared by exactly two faces (manifold).

**Corner sponges (Sierpinski tetrahedron / octahedron).** These are point-contact fractals: at each step every existing copy is replaced by half-scale copies placed at the *vertices* of the seed solid. Starting from a centre set $\{0\}$ and halving the scale each step,

$$c \ \longmapsto\ \bigl\{\, c + s\,v \ :\ v\in\text{vertices}\,\bigr\},\qquad s\to s/2,$$

gives $4$ copies per step for the tetrahedron and $6$ for the octahedron. The final copies are emitted as separate closed solids that touch only at points, so the level-$n$ tetrahedron has $4^n$ solids and $4\cdot4^n$ faces, the octahedron $6^n$ solids and $8\cdot6^n$ faces. The Sierpinski tetrahedron has Hausdorff dimension $\dfrac{\ln 4}{\ln 2}=2$ exactly.

## References

- Karl Menger, "Allgemeine Räume und Cartesische Räume. Zweite Mitteilung: Über umfassendste $n$-dimensionale Mengen," *Proc. Akad. Wetensch. Amsterdam* 29 (1926), 1125–1128 — the Menger sponge (Menger curve in three dimensions).
- Wacław Sierpiński, "Sur une courbe cantorienne qui contient une image biunivoque et continue de toute courbe donnée," *C. R. Acad. Sci. Paris* 162 (1916), 629–632 — the Sierpiński carpet; the tetrahedron/octahedron are its 3D analogues.
- Tamás Vicsek, "Fractal models for diffusion controlled aggregation," *J. Phys. A: Math. Gen.* 16 (1983), L647–L652 — the Vicsek (box) fractal.
- Benoît B. Mandelbrot, *The Fractal Geometry of Nature*, W. H. Freeman, 1982 — self-similar dimension $\ln N/\ln s$.
- Eric W. Weisstein, "Menger Sponge," *MathWorld*: <https://mathworld.wolfram.com/MengerSponge.html>
