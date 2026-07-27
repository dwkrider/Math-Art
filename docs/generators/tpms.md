# Periodic Minimal Surface (TPMS)

![Periodic Minimal Surface (TPMS)](../images/tpms.png)

## Overview

This generator builds triply-periodic minimal surfaces (TPMS) — the space-filling, crystallographically periodic minimal surfaces such as Schwarz P and D, Schoen's Gyroid, I-WP and F-RD, the Neovius surface, the Lidinoid and Split P — plus the singly-periodic Scherk tower. Each is meshed from its standard nodal (level-set) approximation by a vectorized marching-tetrahedra extractor, tileable over several unit cells per axis, and can carry an optional Solidify thickness for 3D-printable lattices.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Surface | Gyroid | Which periodic surface to build (Schwarz P, Schwarz D, Gyroid, Neovius, Schoen I-WP, Schoen F-RD, Lidinoid, Split P, or the singly-periodic Scherk Tower) |
| Cells | 1 | Number of unit cells per axis (1–4) |
| Resolution / Cell | 28 | Sample-grid resolution per unit cell |
| Cell Size | 2.0 | Edge length of one unit cell in Blender units |
| Thickness | 0.0 | If > 0, add a Solidify modifier with this thickness |




## Variants

Renders of each selectable option:

<table>
<tr>
<td align="center"><img src="../images/variants/tpms__P.png" width="200"><br><sub>Schwarz P</sub></td>
<td align="center"><img src="../images/variants/tpms__D.png" width="200"><br><sub>Schwarz D</sub></td>
<td align="center"><img src="../images/variants/tpms__G.png" width="200"><br><sub>Gyroid</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/tpms__NEOVIUS.png" width="200"><br><sub>Neovius</sub></td>
<td align="center"><img src="../images/variants/tpms__IWP.png" width="200"><br><sub>Schoen I-WP</sub></td>
<td align="center"><img src="../images/variants/tpms__FRD.png" width="200"><br><sub>Schoen F-RD</sub></td>
</tr>
<tr>
<td align="center"><img src="../images/variants/tpms__LIDINOID.png" width="200"><br><sub>Lidinoid</sub></td>
<td align="center"><img src="../images/variants/tpms__SPLITP.png" width="200"><br><sub>Split P</sub></td>
<td align="center"><img src="../images/variants/tpms__SCHERKT.png" width="200"><br><sub>Scherk Tower</sub></td>
</tr>
</table>

## How it works

Each surface is approximated by its **nodal (level-set) surface** — the zero set $\{F(x,y,z) = 0\}$ of a short trigonometric polynomial that is periodic with period $2\pi$. These are the standard first-order Fourier approximations to the exact TPMS. The implemented fields include:

- **Schwarz P:** $\cos x + \cos y + \cos z = 0$
- **Schwarz D:** $\sin x\sin y\sin z + \sin x\cos y\cos z + \cos x\sin y\cos z + \cos x\cos y\sin z = 0$
- **Gyroid (Schoen G):** $\sin x\cos y + \sin y\cos z + \sin z\cos x = 0$
- **Neovius:** $3(\cos x + \cos y + \cos z) + 4\cos x\cos y\cos z = 0$
- **Schoen I-WP:** $2(\cos x\cos y + \cos y\cos z + \cos z\cos x) - (\cos 2x + \cos 2y + \cos 2z) = 0$
- **Schoen F-RD**, **Lidinoid**, and **Split P** by their respective higher-harmonic level-set polynomials.

The singly-periodic **Scherk tower** uses $\sin z - \sinh x\,\sinh y = 0$ and is periodic only in $z$, so its $x, y$ extents are clipped to a finite box rather than tiled.

The zero level set is extracted by **marching tetrahedra**: the sample box is divided into a grid of cubes, each cube split into 6 tetrahedra sharing the main diagonal. For every tetrahedron the sign pattern of $F$ at its four corners selects a crossing case (one corner isolated → one triangle; two-and-two → a quad split into two triangles), and the crossing points are found by **linear interpolation** along each edge: for corner values $v_a, v_b$ of opposite sign the crossing sits at parameter $t = v_a/(v_a - v_b)$. Triangle winding is oriented to follow the field gradient using a set of combinatorial orientation flags calibrated once on an exact linear field (so they are immune to sliver triangles). Samples landing exactly on the surface are nudged off zero to avoid degenerate crossings, and the resulting triangle soup is welded by quantizing and uniquing vertex positions.

For a triply-periodic surface the box spans $\text{cells}\times 2\pi$ per axis at resolution $\text{cells}\times(\text{res per cell})$; the mesh is finally scaled so one period equals the chosen cell size ($s = \text{scale}/2\pi$). A Solidify modifier gives the sheet a printable wall thickness when requested.

## References

- Ken Brakke, *Triply Periodic Minimal Surfaces* — <https://kenbrakke.com/evolver/examples/periodic/periodic.html> (the surface inventory) and the *Surface Evolver* — <https://kenbrakke.com/evolver/evolver.html>
- A. H. Schoen, *Infinite Periodic Minimal Surfaces Without Self-Intersections*, NASA Technical Note TN D-5541, 1970 (the Gyroid, I-WP, F-RD).
- H. A. Schwarz, *Gesammelte Mathematische Abhandlungen*, 1890 (the P and D surfaces).
- Nodal (level-set) approximations after H. G. von Schnering & R. Nesper and subsequent literature.
- W. E. Lorensen, H. E. Cline, *Marching Cubes* (and the marching-tetrahedra variant) — the isosurface-extraction method adapted here.
- *xyzdims* — TPMS 3D-printing experiments: <https://xyzdims.com/tag/triply-periodic-minimal-surface/>
