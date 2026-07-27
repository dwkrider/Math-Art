# Curvature Color

![Curvature Color](../images/curvature_color.png)

## Overview

Curvature Color is a **Styles operator applied to a selected existing object**, not a standalone add-mesh generator: with a mesh active, it colors that mesh by discrete Gaussian curvature, after the curvature illustrations in Henry Segerman's *Visualizing Mathematics with 3D Printing* (fig. 4-4) — **red** where the surface is positively curved (sphere-like), **white** where it is flat, **blue** where it is negatively curved (saddle-like). The render image was produced by applying the operator to a base surface. The mesh geometry is never altered: the curvature is computed on a triangulated bmesh copy, written to a `Curvature` vertex-color attribute, and shown through a shared `Curvature Color` material. Apply modifiers first if you want the curvature of the modified surface.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Smoothing | 2 | Laplacian smoothing passes applied to the curvature field (uniform weights); range 0–50 |
| Normalize | Percentile | How to pick the curvature value mapped to full red/blue: **Percentile** (clamp at a percentile of \|curvature\|) or **Manual** (clamp at a fixed value) |
| Percentile | 90.0% | Percentile of \|curvature\| clamped to full color (auto normalization); range 50–100 |
| Clamp | 1.0 | Curvature value mapped to full color (manual normalization); minimum $10^{-6}$ |
| Invert | Off | Swap the roles of the two ramp ends |
| Positive | ~#C83030 red (0.784, 0.188, 0.188) | Ramp end for positive curvature |
| Negative | ~#3060C8 blue (0.188, 0.376, 0.784) | Ramp end for negative curvature |

## How it works

**Discrete Gaussian curvature (angle deficit).** The mesh is triangulated on a throwaway bmesh copy. For each vertex the classical angle deficit is computed:

$$\kappa = \frac{2\pi - \sum_j \theta_j}{A} \quad \text{(interior vertex)}, \qquad \kappa = \frac{\pi - \sum_j \theta_j}{A} \quad \text{(boundary vertex)},$$

where $\theta_j$ are the interior angles of the incident triangles at that vertex and $A$ is the **mixed area**, approximated barycentrically as one third of the total area of the incident triangles. Boundary vertices use the target angle $\pi$ (a straight surface edge) instead of $2\pi$, so a flat rim reads as flat. Angles are accumulated with $\theta = \operatorname{atan2}(\lVert \mathbf u \times \mathbf v\rVert,\ \mathbf u\!\cdot\!\mathbf v)$ for the two edge vectors $\mathbf u,\mathbf v$ at each triangle corner. Loose or degenerate vertices ($A \le 10^{-12}$) are left at $\kappa = 0$.

**Smoothing.** The scalar field is optionally relaxed with uniform-weight Laplacian smoothing over the mesh edges; each pass replaces a value with the average of itself and its neighbours' mean, $v \leftarrow \tfrac12(v + \bar v_{\text{nbr}})$.

**Normalization and ramp.** A clamp value $c$ is chosen either as a percentile of $|\kappa|$ (auto) or a manual constant, and floored at $10^{-3}$ so a flat mesh stays white rather than amplifying float noise to full saturation. Each vertex maps through

$$t = \operatorname{clip}\!\left(\frac{\kappa}{c},\ -1,\ 1\right),$$

optionally negated by **Invert**. The positive part $t_+ = \max(t,0)$ blends white toward the **Positive** color and the negative part $t_- = \max(-t,0)$ blends white toward the **Negative** color:

$$\mathbf{color} = \mathbf 1 + t_+(\mathbf{c}_{\text{pos}} - \mathbf 1) + t_-(\mathbf{c}_{\text{neg}} - \mathbf 1),$$

a diverging color ramp with white at $\kappa = 0$. The result is written to a `FLOAT_COLOR` point attribute named `Curvature`; a shared material (a Principled BSDF fed by an Attribute node reading that attribute) is created or reused and made the active slot. The operator reports the measured curvature range and the clamp used.

## References

- H. Segerman, *Visualizing Mathematics with 3D Printing*, Johns Hopkins University Press, 2016 — the diverging red/white/blue Gaussian-curvature coloring (fig. 4-4).
