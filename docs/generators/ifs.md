# Iterated Function System

![Iterated Function System](../images/ifs.png)

## Overview

Attractors of iterated function systems, in two and three dimensions, across three families that need quite different machinery.

**Self-affine lattice tiles** come from an expanding integer matrix and a digit set: exotic, crystal-like solids that tile space by the integer lattice, including Bandt's seven three-dimensional twindragons and the ABC tiles that are proved homeomorphic to a 3-ball. The level-$k$ point set is computed exactly on the integer lattice rather than sampled; how that point set becomes a mesh is the interesting part, and is discussed below.

**General affine IFS attractors** come from any set of contractive maps: the Sierpinski tetrahedron and octahedron, the Menger sponge, Cantor dust, and anything you care to type in, rendered as exact solid copies, as watertight voxels, or as a smooth contour.

**Planar systems** — the Barnsley fern, the Sierpinski triangle, the Heighway dragon, the Lévy C curve, the Koch curve — are recognised as flat and meshed as a watertight slab one cell thick, at *plane* resolution. This is not a lesser case: a 512×512 grid is a quarter of a million cells, where a 512³ volume grid is out of reach and would leave all but a sliver of it empty. The result prints, and Solidify gives it real depth.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Mode | Self-Affine Tile | A lattice tile from an expanding integer matrix and a residue digit set, or the attractor of contractive affine maps. |
| Tile | ABC tile (1,2,4) | Six ABC tiles, Bandt's seven twindragons, and the cube. Twindragon A is not a fractal — it *is* the cube, in a different basis. The operator reports what the papers prove about each tile's topology. |
| Tile Output | Voxels | Sample the attractor and mesh it as a watertight voxel solid, contour it with marching tetrahedra, or build the exact level-$k$ union of cubes. |
| Level | 0 (auto) | Exact mode only: radix depth; 0 picks a level landing in the 30k-300k cell band. Range 0-24. |
| Holes | 0 | Drop this many digits at every level, turning the tile into a gasket; clamped so the attractor stays three-dimensional. Range 0-6. |
| Dimension | 3D | Filters both the system list and the output list to what is valid. |
| System | Sierpinski Tetrahedron | 3D: Sierpinski tetrahedron/octahedron, Cantor dust, Menger sponge. 2D: Barnsley fern, Sierpinski triangle, Heighway dragon, Lévy C curve, Koch curve. Or Custom. Selecting one loads its maps into the Maps field. |
| Output | Solid Copies | Filtered by Dimension. 3D: deterministic seed copies, chaos-game voxels, or a smooth marching-tetrahedra contour. 2D: relief. |
| Plane Resolution | 512 | In-plane grid resolution for a planar system. Range 32-2048. |
| Seed Solid | Tetrahedron | Which solid to place per word in Solid Copies mode. |
| Depth | 5 | Solid-copies depth; the count is maps^depth and is capped automatically. Range 1-12. |
| Points | 400000 | Chaos-game sample count. Range 10k-5M. |
| Resolution | 128 | Voxel / density grid resolution per axis. Range 16-256. |
| Cover | 0.90 | Smooth contour: the fraction of the sampled mass the surface encloses. Range 0.1-0.999. |
| Min Points per Cell | 1 | Voxel mode: cells with fewer points than this are left empty. Range 1-200. |
| Maps | (Sierpinski tetrahedron) | The maps that will be built: nine matrix entries \| three translations \| probability, one map per semicolon. Always shown, and reloaded whenever a System is selected; editing it switches System to Custom so the edit is what gets used. |
| Seed | 0 | Chaos-game random seed; the same seed always gives the same mesh. Range 0-99999. |
| Largest Piece Only | Off | Smooth contour: discard all but the biggest connected piece. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |
| Thickness | 0.0 | If > 0, add a Solidify modifier with this thickness. Range 0-1. |
| Smooth Shading | Off | Shade the mesh smooth. |

## Variants

<table>
<tr>
<td align="center"><img src="../images/variants/ifs__ABC124.png" width="200"><br><sub>ABC tile (1,2,4)</sub></td>
<td align="center"><img src="../images/variants/ifs__ABC128.png" width="200"><br><sub>ABC tile (1,2,8), self-similar</sub></td>
<td align="center"><img src="../images/variants/ifs__ABC134.png" width="200"><br><sub>ABC tile (1,3,4)</sub></td>
<td align="center"><img src="../images/variants/ifs__TWINA.png" width="200"><br><sub>Twindragon A (non-fractal)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ifs__TWIND.png" width="200"><br><sub>Twindragon D</sub></td>
<td align="center"><img src="../images/variants/ifs__TWING.png" width="200"><br><sub>Twindragon G</sub></td>
<td align="center"><img src="../images/variants/ifs__GASKET.png" width="200"><br><sub>Cube gasket (4 holes)</sub></td>
<td align="center"><img src="../images/variants/ifs__SIERPTETRA.png" width="200"><br><sub>Sierpinski tetrahedron</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ifs__MENGER.png" width="200"><br><sub>Menger sponge</sub></td>
<td align="center"><img src="../images/variants/ifs__VOXEL.png" width="200"><br><sub>Sierpinski octahedron (voxels)</sub></td>
<td align="center"><img src="../images/variants/ifs__ISO.png" width="200"><br><sub>Sierpinski tetrahedron (smooth)</sub></td>
<td align="center"><img src="../images/variants/ifs__FERN.png" width="200"><br><sub>Barnsley fern (2-D)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ifs__SIERPTRI.png" width="200"><br><sub>Sierpinski triangle (2-D)</sub></td>
<td align="center"><img src="../images/variants/ifs__DRAGON.png" width="200"><br><sub>Heighway dragon (2-D)</sub></td>
<td align="center"><img src="../images/variants/ifs__LEVY.png" width="200"><br><sub>Lévy C curve (2-D)</sub></td>
<td align="center"><img src="../images/variants/ifs__KOCH.png" width="200"><br><sub>Koch curve (2-D)</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/ifs__EXACT.png" width="200"><br><sub>ABC (1,2,4), exact level-k cubes</sub></td>
<td></td>
<td></td>
<td></td>
</tr>
</table>

## How it works

### Self-affine tiles, exactly

An expanding integer matrix $M$ (every eigenvalue of modulus $>1$) together with a digit set $D \subset \mathbb{Z}^3$ that is a **complete residue system** for $\mathbb{Z}^3/M\mathbb{Z}^3$ — so $|D| = |\det M| = C$ — determines a unique compact set $T$ with

$$M\,T = T + D, \qquad\text{equivalently}\qquad T = \bigcup_{d \in D} M^{-1}(T + d),$$

and that $T$ tiles $\mathbb{R}^3$ by the lattice $\mathbb{Z}^3$ (Bandt 1991).

The level-$k$ approximation is computed on the integers rather than sampled. Starting from $S_0 = \{0\}$, iterate

$$S_{j+1} = D + M\,S_j$$

as one int64 broadcast — int64 because the coordinates grow like $\|M\|^k$ and would overflow int32 within a few levels. After $k$ steps $S_k$ has **exactly** $C^k$ distinct points; the distinctness is precisely what the residue condition buys, and the self-test checks it for every preset at every level up to 6. In **Exact Level-$k$ Cubes** mode those unit cubes are meshed by an exterior-face walker (only faces between an occupied cell and an empty neighbour, with shared vertices), and then the single linear map $M^{-k}$ is applied. Being linear it cannot break what the walker just built, so the surface stays closed — and its volume is $C^k \cdot |\det M|^{-k} = 1$ exactly, at any level. The default outputs sample the attractor instead and make no such volume claim; see below for why.

The presets:

- **ABC tiles** — $M$ is the companion matrix of $\lambda^3 + A\lambda^2 + B\lambda + C$ with digits $j\,e_1$, $j = 0,\dots,C-1$. Thuswaldner–Zhang (Theorem 1.1) prove such a tile homeomorphic to a closed 3-ball when $1 = A \le B < C$ **and** the tile has 14 neighbours. The neighbour count is a *hypothesis*, not a consequence — their Remark 1.4 conjectures that tiles with more than 14 neighbours are generally *not* balls.

Their Remark 1.3 reduces the count to arithmetic: for $1 \le A \le B < C$ there are 14 neighbours iff $1 \le A < B < C$ and either ($B \ge 2A-1$ and $C \ge 2(B-A)+2$) or ($B < 2A-1$ and $C \ge A+B-2$). Note both branches need $A < B$ strictly, so $A = B$ never qualifies. The generator evaluates this per tile and reports the result, so a Custom matrix gets the same verdict as a preset:

| preset | neighbours | verdict |
|---|---|---|
| (1,2,4), (1,2,8), (1,3,6) | 14 | **proved a 3-ball**; truncated-octahedron CW structure, 24 vertices / 36 edges / 14 faces (Thm 1.5) |
| (1,2,3), (1,3,4), (2,2,3) | >14 | ball theorem does not apply; Remark 1.4 conjectures they are not balls |

**ABC (1,2,8)** is worth singling out: $x^3+x^2+2x+8 = (x+2)(x^2-x+4)$, so all three eigenvalues have modulus exactly 2. By Bandt's Proposition 2.2 that makes it conjugate to a genuinely *self-similar* tile, and it is the one preset whose exact level-$k$ cells distort only polynomially (aspect 5:1 at level 4, 13:1 at level 12) instead of exponentially — so **Exact Level-$k$ Cubes** stays usable at depth on it.

ABC (1,1,2) is deliberately **not** shipped. It satisfies the normal form, so it is a genuine ABC tile, but it also has two collinear digits and $|\det M| = 2$, making it a twindragon as well — and $\mathrm{diag}(-1,1,-1)$ conjugates its matrix to minus twindragon C's, so it is case C turned 180°, not a distinct solid.
- **Twindragons** — Bandt's seven three-dimensional twindragons: $|\det M| = 2$, two digits, characteristic polynomial $\lambda^3 - a\lambda^2 - b\lambda - 2$ for the seven $(a,b)$ pairs giving distinct tiles. Theorem 6.2 proves these seven are the complete list up to conjugacy. Case **A** is Bandt's own non-fractal example: in this lattice basis its tile is exactly the unit cube, and Theorem 2.3(ii) makes it the *unique* self-similar three-dimensional lattice tile with two pieces — as a $1 : \sqrt[3]{2} : \sqrt[3]{4}$ box, its map is a 120° rotation composed with a homothety of ratio $\sqrt[3]{2}$.

What is known about each (Bandt, Prop. 6.4, Prop. 7.2 and §12):

| case | $(a,b)$ | neighbours | faces | point nbrs | topology |
|---|---|---|---|---|---|
| A | (0,0) | 26 | 6 | 8 | the cube; not a fractal |
| B | (−1,1) | 18 | 14 | 4 | truncated-octahedron face pattern; ball *conjectured* |
| C | (1,−1) | 20 | 12 | 8 | rhombic faces; interior *proved* connected; ball conjectured |
| D | (0,1) | 34 | 14 | 12 | interior not simply connected → **not** a ball |
| E | (2,−2) | 34 | 12 | 2 | interior not simply connected → **not** a ball |
| F | (1,0) | — | 16 | — | — |
| G | (0,2) | — | ≥22 | — | — |

F and G are left blank on purpose. Bandt prints neighbour counts for them but writes "we shall provide no details for the complicated twindragons $\mathcal{F}$ and $\mathcal{G}$", and warns that "for $\mathcal{F}$ and $\mathcal{G}$ there are rare outliers on the thin fibres, and an exact estimate is needed" — so this generator does not repeat those two numbers as settled.
- **Cube** — $M = 2I$ with the eight digits $\{0,1\}^3$: the degenerate case, and the base for the gaskets.

A **Holes** count drops the last $h$ digits at every level, giving $(C-h)^k$ cells instead of $C^k$ — the same gasket semantics the sibling 2-D Fractal Rep-Tile generator uses. The count is clamped so the survivors still fill three dimensions. That test is on the *attractor*, not on the digits: the affine hull is the span of $\{M^{-j}(d-d_0)\}$, the smallest $M^{-1}$-invariant subspace containing the digit differences, which is why the twindragons stay solid on two collinear digits while a badly ordered cube subset would collapse to a sheet. The cube's digits are ordered so that the first four are an inscribed tetrahedron, making its four-hole gasket a Sierpinski tetrahedron.

### Why the tile is sampled, not built from cubes

The obvious way to mesh the level-$k$ body is the one the mathematics hands you: take the $C^k$ unit cubes and apply $M^{-k}$. That is exactly right as a *set* — volume 1, closed surface — and exactly wrong as a *mesh*. These companion matrices are strongly non-normal, so $M^{-k}$ maps the unit cube to a parallelepiped whose aspect ratio grows like $(\max|\lambda|/\min|\lambda|)^k$. At its own default level twindragon G's cells are **4830:1 slivers**; the object renders as a stack of loose wafers, and ABC (1,3,4) renders as a hairball of needles. Raising the level improves the shape and worsens the lamination at the same time, so no level fixes it.

The tile's *own* proportions are fine — bounding-box aspect ratios across the presets run 1.0 to 5.2. It is the cells that degenerate, not the object, and compensating with a non-uniform scale would falsify the tile.

So the default output samples the **attractor** instead. Because $T$ tiles $\mathbb{R}^3$ by $\mathbb{Z}^3$, the invariant measure of $w_d(x) = M^{-1}(x+d)$ with equal weights is Lebesgue measure restricted to $T$ — the chaos game samples the tile *uniformly*, and a voxel grid over its exact bounding box recovers the solid. The transient has to be long: $\|M^{-n}\|$ decays only like $\min|\lambda|^{-n}$, which is $0.94^n$ for twindragon G, so the sampler runs 300 steps before recording anything. This is also how the published pictures of these tiles are made.

**Exact Level-$k$ Cubes** remains available, because it is the only mode with volume exactly 1 and it is genuinely exact for the cube and for twindragon A. It reports its own cell aspect ratio and warns when the body has become a laminate.

### Knowing how close it is

The true tile's bounding box is available in closed form, which makes "how good is this approximation" a measurable question rather than a guess. Every point of $T$ is a convergent radix series $\sum_{j\ge1}M^{-j}d_j$ with each digit free, so the support function separates term by term:

$$\max_{x \in T}\langle u, x\rangle = \sum_{j\ge1}\max_{d \in D}\langle u, M^{-j}d\rangle,$$

a geometric series evaluated to machine precision in a few dozen terms. The operator reports the achieved extent as a percentage of that limit.

The numbers are sobering for the thin cases, and are reported rather than hidden. Twindragon G's tips lie along fibres of vanishing measure: reaching 99% of its true extent would need on the order of $2^{77}$ cells. At the default sample it reaches about 80%, and in exact mode at level 10 only 45%. The cube, twindragon A and ABC (1,2,4) reach 97-100%.

### General IFS attractors

Contractive maps $w_i(x) = A_ix + b_i$ (largest singular value $< 1$, checked and refused otherwise) have a unique compact attractor by Hutchinson's theorem. Three renderings:

- **Solid copies** — deterministic: compose all $m^d$ words and place one seed solid per word. Exact, and for the Sierpinski sets the copies meet at points.
- **Voxels** — chaos game binned into a grid, then the same exterior-face walker. Watertight and blocky, the printable option.
- **Smooth contour** — chaos game into a density grid, blurred by a separable 3-tap, then contoured by marching tetrahedra at the level enclosing the requested fraction of the sampled mass.

The chaos game does not run one long sequential orbit — thousands of walkers advance together and each step is a handful of vectorised affine maps. Same measure, far faster in Python, and reproducible: the same seed always gives the identical mesh (the self-test checks both that, and that a different seed does not).

### A note on manifoldness

These families touch at edges and corners by construction, so an edge can have four incident faces rather than two. The surface still encloses its solid — there are no boundary edges — but it is not a manifold there, and a slicer will object. The operator reports the count so you know to thicken before printing.

### What is not here

There is no "3D Barnsley fern". The four maps of the **two-dimensional** fern are published (Barnsley, *Fractals Everywhere*) and are offered embedded in the $xz$-plane — so it stands upright in Blender's z-up world — correctly attributed; a three-dimensional fern is folklore with no authoritative source, so none is invented.

Because it is genuinely two-dimensional, all four of its maps are singular in $\mathbb{R}^3$, and the same is true of every planar preset. **Solid Copies** is therefore refused for them: a rank-deficient map flattens the seed solid to a plate, and the output would be a scatter of loose quads rather than a fern.

Planarity is **measured, not assumed** — the generator samples the attractor and takes the eigenvalues of its 3×3 covariance, so a custom map set that happens to be flat gets the relief treatment too, and a three-dimensional one is refused the relief output. The plane's two principal axes become the grid, and the result is lifted back into the $xz$-plane so it stands upright.

## References

- C. Bandt, Mai The Duy and M. Mesing, "Three-Dimensional Fractals," *The Mathematical Intelligencer* 32, 2010, pp. 12-18. [doi:10.1007/s00283-009-9110-6](https://doi.org/10.1007/s00283-009-9110-6)
- C. Bandt, "Self-similar sets 5. Integer matrices and fractal tilings of $\mathbb{R}^n$," *Proceedings of the American Mathematical Society* 112, 1991, pp. 549-562 — the integer-matrix plus residue-digit-set theorem behind every radix tile here.
- C. Bandt, "Combinatorial topology of three-dimensional self-affine tiles," arXiv:1002.0710, 2010 — the seven twindragon cases.
- J. M. Thuswaldner and S.-Q. Zhang, "On self-affine tiles that are homeomorphic to a ball," arXiv:2107.12076 — the ABC normal form.
- J. E. Hutchinson, "Fractals and self similarity," *Indiana University Mathematics Journal* 30, 1981 — existence and uniqueness of the attractor of a contractive IFS.
- M. F. Barnsley, *Fractals Everywhere*, 2nd ed., Academic Press, 1993 — the chaos game, and the fern.
