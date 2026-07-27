# Stellated Surface Weave

![Stellated Surface Weave](../images/stellated_weave.png)

## Overview

A port of Shengyi Wang's **Stellated Surface Weave**: twelve folded strips interlock along the pentagram planes of a small stellated dodecahedron. Each of the twelve pentagram faces carries a strip that folds over the star's points and weaves through its neighbours; the strip width is the single free parameter, and every mitre recomputes to keep the folds sharp at constant width.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| Strip Width | 0.12 | Width of the folded strips; all mitres recompute when it changes (min 0.02, max 0.32). |
| Coloring | Per Strip | Per Strip (one material per pentagram plane; 12 strips) or None. |
| Scale | 1.0 | Uniform output scale; the result is fit within a 2×Scale cube at the origin (min 0.01, max 100.0). |

## How it works

The construction is driven entirely by the geometry of the **small stellated dodecahedron** (SSD). Starting from a regular dodecahedron (vertices built from $\varphi = \tfrac{\sqrt5+1}{2}$ and $1/\varphi$), each of the 12 pentagonal faces is capped by a pentagonal-pyramid apex raised along the face normal by

$$h = \sqrt{2 + \tfrac{2}{\sqrt5}},$$

giving the 12 star points. Each SSD pentagram face is then indexed as the five apexes of its neighbouring faces taken in star order (every second face around the pentagon).

**Skeleton arms.** Every pentagram face contributes five arms. An arm runs from a tip apex, to a **bend** placed a fraction

$$\frac{5-\sqrt5}{10}$$

of the way along the star edge, to the face centre. Where five arms meet at a shared tip, they form a pentagonal-pyramid apex; the strip's inner point is placed $2\times\text{width}$ inward from the apex toward the origin.

**Mitred folds.** Each strip has two side faces lying in the two adjacent star planes plus a parallelogram closure. To keep the strip a constant width as it folds over a bend, the two extension factors

$$\sqrt{\tfrac{5+\sqrt5}{10}}\quad\text{(outer mitre)}\qquad\text{and}\qquad\sqrt{\tfrac{5-\sqrt5}{10}}\quad\text{(inner mitre)}$$

extend the mitre points along the edge direction. Where an arm meets a face, the inward direction is the component of (face centre − apex) perpendicular to the edge, and mitre points are found as the intersection of the two offset edge lines. At each tip a **star patch** stitches the outer mitre points, the per-face mitre point and the inner apex; toward the face centres the strips stay parallel and meet in mitred pentagon junctions.

Finally all polygons are welded (coincident vertices merged at 8-decimal precision, degenerate faces dropped) and scaled so the dodecahedron vertices sit at radius $\sqrt3$. Each face is tagged with its pentagram-plane (strip) index as a `strip_index` attribute, driving the 12-colour **Per Strip** materials; face normals are recomputed with bmesh. A self-test checks 12 faces / 12 tips / 60 arms and reports any non-2-manifold edges across a range of widths.

## References

- Shengyi Wang (txyyss), *Stellated Surface Weave* — <https://txyyss.github.io/math-art/stellated-surface-weave>, source at <https://github.com/txyyss/math-art> (the construction this ports).
