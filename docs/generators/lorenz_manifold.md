# Lorenz Manifold

![Lorenz Manifold](../images/lorenz_manifold.png)

## Overview

The two-dimensional **stable manifold of the origin** of the Lorenz system — the surface of all points that flow *into* the equilibrium at 0 rather than onto the butterfly attractor. It is the object Osinga and Krauskopf famously crocheted, and the one the 2026 Alternative Fields Medal trophies were 3D-printed from.

$$\dot x = \sigma(y-x), \qquad \dot y = x(\rho - z) - y, \qquad \dot z = xy - \beta z$$

## Options

| Option | Default | Description |
| --- | --- | --- |
| Sigma | 10.0 | Prandtl number; 10 is Lorenz's own value. Range 0.1-50. |
| Rho | 28.0 | Rayleigh number; 28 is Lorenz's own value. Range 1.1-100. |
| Beta | 2.667 | Geometric factor; 8/3 is Lorenz's own value. Range 0.1-10. |
| Arclength | 100.0 | How far to grow the manifold, measured along trajectories rather than in time. Range 5-250. |
| Ring Spacing | 1.0 | Arclength between recorded rings. Range 0.1-5. |
| Target Edge | 0.25 | Wanted spacing along a ring; smaller keeps the sharp folds from being cut across. Range 0.02-2. |
| Seed Radius | 0.02 | Radius of the starting circle in the stable eigenspace. Range 1e-4 to 0.5. |
| Seed Points | 96 | Points on the starting circle. Range 16-512. |
| Max Ring Points | 3000 | Cap on the points in one ring. Range 200-20000. |
| Step | 0.02 | Arclength step of the integrator. Range 0.002-0.2. |
| Thickness | 0.03 | Solidify thickness — the manifold is a surface, so it needs a shell to print. Range 0-0.5. |
| Scale | 1.0 | Uniform scale of the result. Range 0.01-100. |
| Smooth Shading | On | Shade the mesh smooth. |

## How it works

### Why the obvious method fails

At the origin the Jacobian

$$J(0) = \begin{pmatrix} -\sigma & \sigma & 0 \\ \rho & -1 & 0 \\ 0 & 0 & -\beta\end{pmatrix}$$

has eigenvalues $+11.83$, $-22.83$ and $-8/3$. The two negative ones span the stable eigenspace, and the manifold is tangent to it at the origin. Growing it outward means integrating **backward** in time.

The difficulty is that the two stable rates differ by a factor of **8.6**. A front of equal-*time* images therefore stretches by $8.6^{\,t}$ between the fast and slow directions, and the mesh degenerates within a few units of growth. Equal-time integration is not a coarse method here; it is the wrong one.

### Unit-speed growth

The cure is to integrate at unit speed,

$$\frac{dx}{d\tau} = -\frac{f(x)}{\lVert f(x)\rVert},$$

so every trajectory advances the same **arclength** rather than the same time — which is exactly what removes the eigenvalue-ratio stretching. The whole ring is stepped in lockstep by a vectorised RK4, and at each checkpoint it is resampled to evenly spaced points and allowed to gain points as it lengthens.

This is the method of the 2026 *AMS Notices* article, which is how the printed trophies were made. The alternative — Krauskopf and Osinga's geodesic level sets, where each new ring point solves a two-unknown boundary value problem — produces true geodesic circles and reaches much larger geodesic radii, at a cost this module does not need.

Note the sign: growing a *stable* manifold means integrating backward, so it is $-f/\lVert f\rVert$, not $+$. Getting that wrong silently produces the unstable manifold instead, which is a curve's worth of nonsense; the self-test checks the ring moves outward.

### The seed

The starting circle is the **linear** stable eigenspace at radius $\delta$, not a high-order parameterization of the manifold. The error is $O(\delta^2)$, which at $\delta = 0.02$ is far below the mesh resolution once the result is fitted into a 2 m cube. The parameterization method used in the *Notices* paper is the upgrade if that ever stops being true.

### Interpolation is the accuracy bottleneck

Every resampling step moves each point off the true ring by the chord-to-arc deviation, and those errors are what the forward flow later amplifies. Measured, that — not the ODE integrator — is what limits the surface's accuracy: halving `target_edge` halves the error, while halving `step` changes nothing.

So resampling uses **Catmull-Rom**, not linear interpolation, dropping the deviation from $O(h^2)$ to $O(h^4)$ at no cost in ring count. Measured against the invariance test below, that roughly halves how far the finished surface sits off the manifold: **0.049 against 0.113**.

### How it is checked

Points of a stable manifold flow into the equilibrium — but they cannot stay there numerically. The origin is a saddle, so any deviation is amplified by $e^{11.83\,t}$: a point sitting slightly off the surface dives toward 0 and then shoots away. The test therefore measures the **closest approach** as a fraction of the starting radius, and is only meaningful beside a control:

| | closest approach |
|---|---|
| points of the outer ring | **0.049** (worst 0.102) |
| the same points nudged 0.5 off the surface | 0.378 |

A surface that merely looked right would fail this. The self-test also confirms the stable eigenvalues to $10^{-3}$, that the seed frame is orthonormal and spans an invariant plane of $J$, that ring circumference and point count grow monotonically, and that the mesh is a disk ($\chi = 1$) whose single free rim is exactly the outermost ring.

### Limits worth knowing

The field vanishes at the two nontrivial equilibria $C^\pm = (\pm\sqrt{\beta(\rho-1)}, \pm\sqrt{\beta(\rho-1)}, \rho-1)$, which the manifold spirals around but never contains. Unit-speed integration divides by $\lVert f\rVert$, so the norm is clamped and the operator warns if the ring passes within 0.05 of either.

Chord-length resampling **cuts across sharp folds** as the manifold develops its helical scrolls. Past about arclength 180 you want `target_edge` at 0.1 or below; the operator warns when the outermost ring hits its point cap, which is the symptom.

## References

- E. N. Lorenz, "Deterministic nonperiodic flow," *Journal of the Atmospheric Sciences* 20(2), 1963, pp. 130-141.
- B. Krauskopf and H. M. Osinga, "Computing geodesic level sets on global (un)stable manifolds of vector fields," *SIAM Journal on Applied Dynamical Systems* 2(4), 2003, pp. 546-569.
- H. M. Osinga and B. Krauskopf, "Crocheting the Lorenz manifold," *The Mathematical Intelligencer* 26(4), 2004, pp. 25-37.
- P. R. Bishop, S. Chenoweth, E. Fleurantin, A. Ogueda-Oliva, E. Sander and J. Seay, "3D printing of invariant manifolds in dynamical systems," *Notices of the American Mathematical Society*, 2026; preprint [arXiv:2504.15884](https://arxiv.org/abs/2504.15884). The arclength-reparametrized growth used here, and the parameterization method for the local piece.
