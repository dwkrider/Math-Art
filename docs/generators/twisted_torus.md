# Twisted Torus

![Twisted Torus](../images/twisted_torus.png)

## Overview
A regular polygon swept around a circle while rotating about the sweep path -- the classic twisted prismatic torus from George W. Hart's pages. When the total twist is an integer number of $360/n$ steps the seam stays exact, so the polygon's faces join into long helical bands that spiral around the ring; the number of distinct bands is $\gcd(n,\text{steps})$. Corner rounding morphs the profile continuously from a crisp polygon to a circle, and a *Triangle Shrink* below 1 separates the swept faces into one solid ribbon per band.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Polygon Sides | 3 | Number of sides $n$ of the swept profile polygon. Range 2-16. |
| Twist Steps | 1 | Total twist in units of $360/n$ degrees; $\gcd(n,\text{steps})$ helical bands result. Range -8 to 8. |
| Triangle Shrink | 1.0 | Shrink of each polygon side about its midpoint. 1 = sides share edges (one contiguous mesh); less than 1 separates the swept sheets into one mesh object per helical band. Range 0.3-1.0. |
| Sheet Thickness | 0.06 | Thickness of the separated sheets (shown only when Shrink < 1). Range 0.005-0.5. |
| Ring Radius | 1.6 | Major radius of the ring the profile sweeps around. Range 0.2-20. |
| Profile Radius | 0.55 | Minor radius (circumradius of the profile polygon). Range 0.02-5. |
| Corner Rounding | 0.0 | 0 = crisp polygon, 1 = circle. Range 0.0-1.0. |
| Ring Segments | 192 | Sweep steps around the ring. Range 16-512. |
| Profile Subdivision | 2 | Points per polygon side. Range 1-16. |
| Coloring | Per Strip | Per Strip: one material per visible helical strip ($n$ colours). Per Band: one material per connected band ($\gcd(n,\text{twist})$ colours; coprime = one colour). None: no materials. |
| Scale | 1.0 | Contiguous mesh / band assembly is centred and fitted to a 2 m cube $\times$ Scale. Range 0.01-100. |

## How it works

**Profile.** The cross-section is a regular $n$-gon of circumradius `minor`, sampled at $m = n\cdot(\text{profile\_res})$ points. Corner rounding blends each polygon-edge point toward the inscribing circle: for a sample at fractional polygon-corner coordinate, with the crisp edge point $\mathbf e$ and the circle point $\mathbf{circ}=(\cos\alpha,\sin\alpha)$,
$$\mathbf p = \big((1-r)\,\mathbf e + r\,\mathbf{circ}\big)\cdot\text{minor},\qquad r=\text{rounding}.$$
$r=0$ gives the polygon, $r=1$ the circle.

**Twisted sweep.** Let $R$ be the ring radius (`major`) and $u=2\pi s/\text{segments}$ the sweep angle. As the profile travels around the ring it is rotated about the sweep tangent by the accumulated twist
$$\theta(s) = \frac{2\pi\,\text{twist\_steps}}{n}\cdot\frac{s}{\text{segments}} .$$
A profile point $(p_x,p_y)$ becomes $(x,y)=\big(p_x\cos\theta-p_y\sin\theta,\;p_x\sin\theta+p_y\cos\theta\big)$, and the surface point is
$$\big((R+x)\cos u,\;(R+x)\sin u,\;y\big).$$
Over one full revolution the profile has turned by exactly $\text{twist\_steps}\cdot(360/n)$, an integer number of $n$-gon symmetry steps, so the last ring column matches the first under an index shift of $(m/n)\cdot\text{twist\_steps}$ -- the seam is exact and the mesh closes into a genuine torus ($\chi=0$, checked in the self-test).

**Bands and strips.** Following one polygon side continuously around the ring, it returns to a side of the same polygon only after the accumulated twist is a whole number of turns, i.e. after $n/\gcd(n,\text{steps})$ revolutions. Hence there are
$$g = \gcd(n,\text{twist\_steps})$$
topologically connected **bands**, each visiting $n/g$ of the $n$ angular **strips**. *Per Strip* colouring assigns a material to each of the $n$ visible strips; *Per Band* to each of the $g$ connected bands (so when $n$ and the twist are coprime, $g=1$ and the whole torus is one band). Both `strip_index` and `band_index` face attributes are always written.

**Separated ribbons.** With *Triangle Shrink* $<1$, each polygon side is shrunk about its own midpoint and swept as a *solid* ribbon of thickness `sheet_thickness`; the ribbons chain across the twist seam into $g$ closed bands, each emitted as its own mesh object with sharp border edges. At shrink $=1$ the classic single contiguous tube is produced instead.

## References

- G. W. Hart, *Twisted Torus* example, https://www.georgehart.com/ (and the vibe-coded program examples, https://www.georgehart.com/vibecode/).
- Twisted-torus generator after the programs on Hart's vibecode page.
