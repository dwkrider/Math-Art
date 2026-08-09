# 3D Iterated Function System

![3D Iterated Function System](../images/ifs3d.png)

## Overview

Attractors of iterated function systems in three dimensions, in two families that need quite different machinery.

**Self-affine lattice tiles** come from an expanding integer matrix and a digit set: exotic, crystal-like solids that tile space by the integer lattice, including Bandt's three-dimensional twindragons and the ABC tiles proved homeomorphic to a ball. These are computed *exactly* — the level-$k$ approximation is a set of integer lattice points, not a sampled cloud, so the mesh is watertight and has volume exactly 1 at every level.

**General affine IFS attractors** come from any set of contractive maps: the Sierpinski tetrahedron and octahedron, the Menger sponge, Cantor dust, and anything you care to type in, rendered as exact solid copies, as watertight voxels, or as a smooth contour.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Mode | Self-Affine Tile | A lattice tile from an expanding integer matrix and a residue digit set, or the attractor of contractive affine maps. |
| Tile | ABC tile (1,2,4) | Six ABC tiles, Bandt's seven twindragons, or the cube. |
| Level | 0 (auto) | Radix depth; 0 picks a level landing in the 30k-300k cell band. Range 0-24. |
| Holes | 0 | Drop this many digits at every level, turning the tile into a gasket. Range 0-6. |
| System | Sierpinski Tetrahedron | Sierpinski tetrahedron/octahedron, Cantor dust, Menger sponge, the embedded 2-D Barnsley fern, or Custom. |
| Output | Solid Copies | Deterministic seed copies, chaos-game voxels, or a smooth marching-tetrahedra contour. |
| Seed Solid | Tetrahedron | Which solid to place per word in Solid Copies mode. |
| Depth | 5 | Solid-copies depth; the count is maps^depth and is capped automatically. Range 1-12. |
| Points | 400000 | Chaos-game sample count. Range 10k-5M. |
| Resolution | 128 | Voxel / density grid resolution per axis. Range 16-256. |
| Cover | 0.90 | Smooth contour: the fraction of the sampled mass the surface encloses. Range 0.1-0.999. |
| Min Points per Cell | 1 | Voxel mode: cells with fewer points than this are left empty. Range 1-200. |
| Maps | (Sierpinski tetrahedron) | Custom affine maps: nine matrix entries \| three translations \| probability, one map per semicolon. |
| Seed | 0 | Chaos-game random seed; the same seed always gives the same mesh. Range 0-99999. |
| Largest Piece Only | Off | Smooth contour: discard all but the biggest connected piece. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |
| Thickness | 0.0 | If > 0, add a Solidify modifier with this thickness. Range 0-1. |
| Smooth Shading | Off | Shade the mesh smooth. |

## Variants

<table>
<tr>
<td align="center"><img src="../images/variants/ifs3d__ABC124.png" width="200"><br><sub>ABC tile (1,2,4)</sub></td>
<td align="center"><img src="../images/variants/ifs3d__ABC112.png" width="200"><br><sub>ABC tile (1,1,2)</sub></td>
<td align="center"><img src="../images/variants/ifs3d__ABC134.png" width="200"><br><sub>ABC tile (1,3,4)</sub></td>
<td align="center"><img src="../images/variants/ifs3d__TWINA.png" width="200"><br><sub>Twindragon A</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ifs3d__TWIND.png" width="200"><br><sub>Twindragon D</sub></td>
<td align="center"><img src="../images/variants/ifs3d__TWING.png" width="200"><br><sub>Twindragon G</sub></td>
<td align="center"><img src="../images/variants/ifs3d__GASKET.png" width="200"><br><sub>Cube gasket (4 holes)</sub></td>
<td align="center"><img src="../images/variants/ifs3d__SIERPTETRA.png" width="200"><br><sub>Sierpinski tetrahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ifs3d__MENGER.png" width="200"><br><sub>Menger sponge</sub></td>
<td align="center"><img src="../images/variants/ifs3d__VOXEL.png" width="200"><br><sub>Sierpinski octahedron (voxels)</sub></td>
<td align="center"><img src="../images/variants/ifs3d__ISO.png" width="200"><br><sub>Sierpinski tetrahedron (smooth)</sub></td>
<td align="center"><img src="../images/variants/ifs3d__FERN.png" width="200"><br><sub>Barnsley fern (2-D, embedded)</sub></td>
</tr>
</table>

## How it works

### Self-affine tiles, exactly

An expanding integer matrix $M$ (every eigenvalue of modulus $>1$) together with a digit set $D \subset \mathbb{Z}^3$ that is a **complete residue system** for $\mathbb{Z}^3/M\mathbb{Z}^3$ — so $|D| = |\det M| = C$ — determines a unique compact set $T$ with

$$M\,T = T + D, \qquad\text{equivalently}\qquad T = \bigcup_{d \in D} M^{-1}(T + d),$$

and that $T$ tiles $\mathbb{R}^3$ by the lattice $\mathbb{Z}^3$ (Bandt 1991).

The level-$k$ approximation is computed on the integers rather than sampled. Starting from $S_0 = \{0\}$, iterate

$$S_{j+1} = D + M\,S_j$$

as one int64 broadcast — int64 because the coordinates grow like $\|M\|^k$ and would overflow int32 within a few levels. After $k$ steps $S_k$ has **exactly** $C^k$ distinct points; the distinctness is precisely what the residue condition buys, and the self-test checks it for every preset at every level up to 6. Those unit cubes are meshed by an exterior-face walker (only faces between an occupied cell and an empty neighbour, with shared vertices), and then the single linear map $M^{-k}$ is applied. Being linear it cannot break what the walker just built, so the surface stays closed — and its volume is $C^k \cdot |\det M|^{-k} = 1$ exactly, at any level.

The presets:

- **ABC tiles** — $M$ is the companion matrix of $\lambda^3 + A\lambda^2 + B\lambda + C$ with digits $j\,e_1$, $j = 0,\dots,C-1$. Those with $1 = A \le B < C$ are proved homeomorphic to a closed ball, with the cell structure of a truncated octahedron (Thuswaldner–Zhang).
- **Twindragons** — Bandt's seven three-dimensional twindragons: $|\det M| = 2$, two digits, characteristic polynomial $\lambda^3 - a\lambda^2 - b\lambda - 2$ for the seven $(a,b)$ pairs giving distinct tiles.
- **Cube** — $M = 2I$ with the eight digits $\{0,1\}^3$: the degenerate case, and the base for the gaskets.

A **Holes** count drops the last $h$ digits at every level, giving $(C-h)^k$ cells instead of $C^k$ — the same gasket semantics the sibling 2-D Fractal Rep-Tile generator uses.

### Anisotropy is the point

In three dimensions the eigenvalues of $M$ generally have *different* moduli, so $M^{-k}$ contracts unevenly and the level-$k$ approximation is genuinely thin in the slow direction. That is not a defect — it is what "self-affine" rather than "self-similar" means. (A self-affine tile is conjugate to a self-similar one **iff** all eigenvalues of $M$ share a modulus.)

It also sets how fast the shape settles: the bounding box approaches the tile's at a rate governed by $1/\min|\lambda|$, which runs from $1/2$ for the cube to $1/1.063$ for twindragon G. That is why the auto level differs per preset, and why the self-test asserts on the *decay* of the box step rather than against any fixed threshold.

### General IFS attractors

Contractive maps $w_i(x) = A_ix + b_i$ (largest singular value $< 1$, checked and refused otherwise) have a unique compact attractor by Hutchinson's theorem. Three renderings:

- **Solid copies** — deterministic: compose all $m^d$ words and place one seed solid per word. Exact, and for the Sierpinski sets the copies meet at points.
- **Voxels** — chaos game binned into a grid, then the same exterior-face walker. Watertight and blocky, the printable option.
- **Smooth contour** — chaos game into a density grid, blurred by a separable 3-tap, then contoured by marching tetrahedra at the level enclosing the requested fraction of the sampled mass.

The chaos game does not run one long sequential orbit — thousands of walkers advance together and each step is a handful of vectorised affine maps. Same measure, far faster in Python, and reproducible: the same seed always gives the identical mesh (the self-test checks both that, and that a different seed does not).

### A note on manifoldness

These families touch at edges and corners by construction, so an edge can have four incident faces rather than two. The surface still encloses its solid — there are no boundary edges — but it is not a manifold there, and a slicer will object. The operator reports the count so you know to thicken before printing.

### What is not here

There is no "3D Barnsley fern". The four maps of the **two-dimensional** fern are published (Barnsley, *Fractals Everywhere*) and are offered embedded in the $z = 0$ plane, correctly attributed; a three-dimensional fern is folklore with no authoritative source, so none is invented.

## References

- C. Bandt, Mai The Duy and M. Mesing, "Three-Dimensional Fractals," *The Mathematical Intelligencer* 32, 2010, pp. 12-18. [doi:10.1007/s00283-009-9110-6](https://doi.org/10.1007/s00283-009-9110-6)
- C. Bandt, "Self-similar sets 5. Integer matrices and fractal tilings of $\mathbb{R}^n$," *Proceedings of the American Mathematical Society* 112, 1991, pp. 549-562 — the integer-matrix plus residue-digit-set theorem behind every radix tile here.
- C. Bandt, "Combinatorial topology of three-dimensional self-affine tiles," arXiv:1002.0710, 2010 — the seven twindragon cases.
- J. M. Thuswaldner and S.-Q. Zhang, "On self-affine tiles that are homeomorphic to a ball," arXiv:2107.12076 — the ABC normal form.
- G. Gelbrich, "Crystallographic reptiles," *Geometriae Dedicata*, 1994.
- J. E. Hutchinson, "Fractals and self similarity," *Indiana University Mathematics Journal* 30, 1981 — existence and uniqueness of the attractor of a contractive IFS.
- M. F. Barnsley, *Fractals Everywhere*, 2nd ed., Academic Press, 1993 — the chaos game, and the fern.
