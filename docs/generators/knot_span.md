# Knot to Knot Surface

![Knot to Knot Surface](../images/knot_span.png)

## Overview

This generator builds the classic Plateau-problem demonstration: the minimal surface (soap film) spanning between a $(p,q)$ **torus knot** as its inner boundary and a **circle** as its outer boundary — a trefoil-to-circle span by default. Because the outer circle is wound $p$ times to line up with the knot's strands, the film is a multiply-wound annulus whose sheets pass through one another. It can also span between two torus knots, and can be output as a mesh, a NURBS patch, or as separate one-winding sheet objects.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Knot p | 2 | $p$ of the inner torus knot |
| Knot q | 3 | $q$ of the inner torus knot; 0 degenerates it to a flat circle (radius 3 × Knot Scale) wound $p$ times |
| Circle Radius | 4.5 | Radius of the outer circle (when the outer boundary is a plain circle) |
| Knot Scale | 1.0 | Scale of the inner torus knot |
| Inner Height | 1.0 | Scale of the inner boundary's vertical oscillation, independent of its radius (0 flattens it into a wavy-radius ring) |
| Inner Lift | 0.0 | Shift the inner boundary up or down; with two circles this makes a catenoid-style span |
| Inner Rotation | 0.0 | Rotate the inner boundary about the vertical axis relative to the outer one, twisting the ruling |
| Outer Knot q | 0 | Make the outer boundary a $(p,q)$ torus knot too; 0 keeps the flat round circle |
| Outer Knot p | 0 | $p$ of the outer boundary; 0 matches the inner knot's $p$ (keeps the ruling lined up) |
| Outer Knot Scale | 2.0 | Scale of the outer torus knot (radii ≈ 1–3 × this) |
| Outer Height | 1.0 | Scale of the outer knot's vertical oscillation (0 flattens it into a wavy-radius ring) |
| Boundary Samples | 96 | Number of samples around each boundary loop |
| Interior Rings | 16 | Number of interior rings between the two boundaries |
| Solver Iterations | 2 | Pinkall-Polthier area-minimization outer iterations |
| NURBS Output | Off | Emit a compact NURBS surface (control grid = solver grid) instead of a dense mesh |
| Split Sheets | Off | Output $p$ separate one-winding sheet objects instead of one self-passing span (shown only when $p > 1$) |

## How it works

A **$(p,q)$ torus knot** is sampled as

$$\big(x, y, z\big) = \Big(r\cos(p\,t),\; r\sin(p\,t),\; -\sin(q\,t)\Big), \qquad r = 2 + \cos(q\,t),\quad t \in [0, 2\pi),$$

(scaled by *Knot Scale*, with the $z$-oscillation multiplied by *Inner Height* and shifted by *Inner Lift*). This is the inner boundary. The outer boundary is either a plain circle wound $p$ times, $\big(R\cos(p\,t), R\sin(p\,t), 0\big)$ — the winding matching the knot's $p$ so the ruling lines up — or a second torus knot when *Outer Knot q* > 0.

An initial **ruled annulus** is built by linearly interpolating between the two aligned loops across *Interior Rings* rows, with both boundary loops pinned. The surface is then relaxed toward minimal area by the **Pinkall-Polthier** cotangent-Laplacian iteration: each outer iteration recomputes per-edge cotangent weights

$$w_{ij} = \tfrac12\cot\theta_{ij}$$

(clamped positive to keep the linear system positive-definite) and solves the discrete Laplace equation $L\,x = 0$ for the free (interior) vertices with the boundary held fixed, using a conjugate-gradient core. The cotangent Laplacian is the gradient of discrete surface area, so fixed-point iteration drives the mesh to a discrete minimal surface. The result is centered on the origin and fit within a 2 m cube.

Because the outer boundary winds $p$ times, the span's sheets genuinely pass through one another. **Split Sheets** instead slices the solver grid into $p$ blocks of columns (neighbouring sheets sharing their seam columns) and emits each single-winding sheet as its own object. **NURBS output** fairs the solved grid first — per-column arc-length resampling to remove the solver's shear, light 2D net smoothing, and a normal-only mean-curvature polish — then uses it as the control net of a single cyclic NURBS patch.

## References

- U. Pinkall, K. Polthier, *Computing Discrete Minimal Surfaces and Their Conjugates*, Experimental Mathematics 2(1), 1993 (the area-minimization algorithm).
- Ken Brakke, *Surface Evolver* — <https://kenbrakke.com/evolver/evolver.html> (the reference Plateau-problem solver this is a lightweight analogue of).
- *The circle-to-trefoil-knot minimal surface construction*, Mathematica Stack Exchange #69131 — <https://mathematica.stackexchange.com/questions/69131/>
