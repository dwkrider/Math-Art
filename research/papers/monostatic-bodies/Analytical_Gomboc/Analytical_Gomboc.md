# An Analytical Gömböc

## Source

M. L. Sloan, *"An Analytical Gomboc"* — a six-page technical note presenting two
closed-form, infinitely-differentiable ("analytic") gömböc surfaces of the form
$r^4 = 1 + 4\beta\,\sin(\theta)\,\cos(\varphi - P(\theta))$. The note builds on,
and acknowledges the guidance of, Professor Gábor Domokos, co-inventor of the
gömböc. It cites the original existence result:

> [1] P. L. Várkonyi and G. Domokos, *"Mono-monostatic bodies: the answer to
> Arnold's question,"* **The Mathematical Intelligencer**, Volume 28, Number 4,
> pp. 34–38 (2006).

**Notation convention (important):** the note uses physics-style spherical
coordinates in which $\theta$ is the **polar** angle (measured from the $z$
axis) and $\varphi$ is the **azimuth** (measured around the $z$ axis). This is
the opposite of the common mathematics convention, so read every equation below
with $\theta$ = polar, $\varphi$ = azimuth.

---

## Abstract

Investigation of the mathematical requirements for a three-dimensional
geometrical object to qualify as a gömböc (mono-monostatic) has resulted in the
discovery of two specific, *analytical* gömböc shapes — analytical in the sense
that the function describing the gömböc surface is infinitely differentiable.
This brief note summarizes the analysis and gives the formulae for the two
specific shapes.

## 1. Introduction

A gömböc is a three-dimensional convex solid of constant density that possesses
only two equilibrium points — one stable, the other unstable — when resting on a
horizontal surface under a constant vertical gravitational field.

The question of whether such an object could exist was raised by the Russian
mathematician Vladimir Arnold in 1995, and settled in 2006 by Gábor Domokos and
Péter Várkonyi of Hungary [1]. Their solution required extreme precision in any
physical embodiment: tolerances of about $10^{-5}$ in the earliest embodiment,
and a somewhat more forgiving but still demanding $10^{-3}$ in later ones (as
discussed in various internet articles on the gömböc, e.g. en.wikipedia.org and
plus.maths.org).

This note presents two specific analytic gömböc shapes that are simple in form
and, to the extent examined, less sensitive to variation. The presentation is
expository; the many (often fruitless but instructive) research paths taken
along the way are omitted.

## 2. Overall Gömböc Requirements

In general, the bounding surface of an analytic gömböc may be specified by a
function

$$\psi(z, x, y) = \text{constant}$$

where $x, y, z$ are standard Cartesian coordinates of the three-dimensional space
in which the gömböc is embedded. The coordinate system is oriented so that the
origin ($x = y = z = 0$) is the center of mass of the homogeneous-density
gömböc.

For examining the gömböc requirements, spherical (polar) coordinates
$r, \theta, \varphi$ are appropriate, with

$$
\begin{aligned}
x &= r\,\sin(\theta)\,\cos(\varphi) \\
y &= r\,\sin(\theta)\,\sin(\varphi) \\
z &= r\,\cos(\theta)
\end{aligned}
$$

The center of mass of the gömböc is then the point $r = 0$.

In polar coordinates the bounding surface $\psi(z, x, y) = \text{constant}$ may
be inverted to yield a solution in $r$ for the boundary of the gömböc:

$$r = F(\theta, \varphi)$$

The requirement of convexity ensures that $r$ is single-valued in
$\theta, \varphi$ and moreover positive definite.

### Center of Mass Requirements

Since $r = 0$ is the center of mass of the gömböc, we require

$$
\begin{aligned}
\int d^3V\; x(r, \theta, \varphi) &= 0 \\
\int d^3V\; y(r, \theta, \varphi) &= 0 \\
\int d^3V\; z(r, \theta, \varphi) &= 0
\end{aligned}
\qquad
\begin{aligned}
&\text{where } d^3V = r^2\,\sin(\theta)\,dr\,d\theta\,d\varphi, \\
&\text{with the integration carried out} \\
&\text{over the volume of the gömböc.}
\end{aligned}
\tag{1}
$$

The boundary $r = F(\theta, \varphi)$ being single-valued in
$\theta, \varphi$, the $r$ integration is trivial and one arrives at the
following center-of-mass requirements:

$$
\begin{aligned}
\int_0^{2\pi} d\varphi \int_0^{\pi} \sin(\theta)\,\cos(\theta)\,F^4(\theta, \varphi)\,d\theta &= 0 \\[4pt]
\int_0^{2\pi} d\varphi \int_0^{\pi} \sin^2(\theta)\,\sin(\varphi)\,F^4(\theta, \varphi)\,d\theta &= 0 \\[4pt]
\int_0^{2\pi} d\varphi \int_0^{\pi} \sin^2(\theta)\,\cos(\varphi)\,F^4(\theta, \varphi)\,d\theta &= 0
\end{aligned}
\tag{2}
$$

Using complex-variable notation $\exp(i\varphi) = \cos(\varphi) + i\,\sin(\varphi)$,
the second and third equations may be combined into a more succinct statement of
the center-of-mass requirements:

$$
\int_0^{2\pi} d\varphi \int_0^{\pi} \sin(\theta)\,\cos(\theta)\,F^4(\theta, \varphi)\,d\theta = 0
\tag{3a}
$$

$$
\int_0^{2\pi} d\varphi \int_0^{\pi} \sin^2(\theta)\,\exp(i\varphi)\,F^4(\theta, \varphi)\,d\theta = 0
\tag{3b}
$$

### Equilibria Requirements

Given a gömböc shape defined by

$$r - F(\theta, \varphi) = 0 \tag{4}$$

equilibrium points are those points on the surface where the normal to the
surface,

$$\nabla\bigl(r - F(\theta, \varphi)\bigr) \tag{5}$$

lies wholly in the $\hat{r}$ unit-normal direction. Specifically, equilibria are
those $\theta, \varphi$ sets of points where the components of the gradient
$\nabla$ in the $\theta$ and $\varphi$ directions vanish:

$$\frac{1}{\sin(\theta)}\,\partial_\varphi F(\theta, \varphi) = 0 \tag{6}$$

and

$$\partial_\theta F(\theta, \varphi) = 0. \tag{7}$$

For a shape to be a gömböc, there can be only two sets of such points.

### Convexity Requirement

As with all convex gömböc shapes, the radius of curvature must be positive at
each point on the gömböc surface. For the specific analytic gömböcs exhibited
below, this is easily achieved.

## 3. Specific Analytic Gömböcs

Minimally "bumpy" shapes probably stand the best chance of meeting the gömböc
requirements, since a gömböc is required to exhibit two and only two equilibrium
points. That observation, together with many hours of research and false starts,
led the author to restrict the search for analytic gömböc geometries to the
simplest possible $\varphi$ dependence.[^1]

$$F^4(\theta, \varphi) = R(\theta) + \sin(\theta)\,A(\theta)\,\cos\bigl(\varphi - P(\theta)\bigr), \qquad A(\theta) > 0 \tag{8}$$

Among such possible solutions, the following particular embodiment has proven
particularly useful:

$$r^4 = F^4(\theta, \varphi) = 1 + 4\beta\,\sin(\theta)\,\cos\bigl(\varphi - P(\theta)\bigr), \tag{9}$$

with $\beta$ a small positive constant.[^2]

[^1]: It is well known that a purely $\varphi$-symmetric gömböc is not possible.

[^2]: This gömböc shape is essentially a sphere of unit radius with small surface
    perturbations. To lowest order in $\beta$, this particular embodiment may be
    functionally realized by slicing a unit sphere transverse to the $z$ axis into
    many (infinitesimally) thin disks and then re-assembling the sphere from the
    disks, but with the center of each disk offset from the $z$ axis by
    $(x_o, y_o) = \bigl(\beta\,\cos P(\theta),\; \beta\,\sin P(\theta)\bigr)$.

This formulation greatly simplifies the equilibria and center-of-mass
requirements. In particular, there are two and only two equilibrium points:

$$
\begin{aligned}
\theta = \pi/2, \quad \varphi &= P(\pi/2) + \pi && \text{(stable equilibrium point)} \\
\theta = \pi/2, \quad \varphi &= P(\pi/2) && \text{(unstable equilibrium point)}
\end{aligned}
\tag{10}
$$

Moreover, the center-of-mass requirement reduces to a restriction on $P(\theta)$:

$$\int_0^{\pi} \sin^3(\theta)\,\exp\bigl(i\,P(\theta)\bigr)\,d\theta = 0 \tag{11}$$

Thus Eq. (9) properly describes an analytic gömböc, provided $P(\theta)$
satisfies the center-of-mass requirement Eq. (11). Two particular examples are
now presented.

### Analytic Gömböc 1

The choice $P(\theta) = n\,\theta$ satisfies Eq. (11) for all
$n = 2p + 1$ with $p \geq 2$. Choosing $p = 2$ (i.e. $n = 5$) to provide the
simplest and least "bumpy" solution, we arrive at one embodiment of an analytic
gömböc:

$$r^4 = 1 + 4\beta\,\sin(\theta)\,\cos(\varphi - 5\theta), \tag{12}$$

with a single stable equilibrium point at

$$\theta = \pi/2,\; \varphi = 3\pi/2 \tag{13}$$

and a single unstable equilibrium point at $\theta = \pi/2,\; \varphi = \pi/2$.

The positive constant $\beta$ must, of course, be sufficiently small to satisfy
the convexity requirement; $\beta = 0.15$ or smaller appears to suffice.

### Analytic Gömböc 2

Introducing the variable

$$\eta = \frac{3\pi}{2}\left(\cos(\theta) - \frac{\cos^3(\theta)}{3}\right)$$

and expressing $P(\theta)$ in terms of $\eta$, the center-of-mass requirement
Eq. (11) reduces to the simple requirement

$$\int_{-\pi}^{\pi} \exp\bigl(i\,P(\eta)\bigr)\,d\eta = 0. \tag{14}$$

The following particularly simple solution:

$$P(\eta) = \eta = \frac{3\pi}{2}\left(\cos(\theta) - \frac{\cos^3(\theta)}{3}\right) \tag{15}$$

provides a second analytic gömböc solution:

$$r^4 = 1 + 4\beta\,\sin(\theta)\,\cos\!\left(\varphi - \frac{3\pi}{2}\left(\cos(\theta) - \frac{\cos^3(\theta)}{3}\right)\right) \tag{16}$$

with a single stable equilibrium point at

$$\theta = \pi/2,\; \varphi = \pi \tag{17}$$

and a single unstable equilibrium point at $\theta = \pi/2,\; \varphi = 0$.

Examination of the convexity requirement for this gömböc shape indicates that a
value of $\beta = 0.17$ or smaller should suffice.

Note that the Eq. (12) gömböc wraps smoothly two and one-half revolutions around
the $z$ axis ($\varphi = 0$ to $5\pi$ as $\theta$ traverses $0$ to $\pi$), while
the second, Eq. (16), gömböc exhibits only one full revolution around the $z$
axis, but with a somewhat nonlinear $\varphi$-versus-$\theta$ path.

Finally, as indicated in Footnote 2, the solutions presented amount to surface
perturbations on a unit sphere. Accordingly, the solutions may be scaled up by
any constant $r_o^4$ to achieve any specific gömböc size desired.

## Acknowledgement

Posting and dissemination of this technical note would not have been possible
without the guidance of Professor Gábor Domokos, co-inventor of the gömböc,
whose support and encouragement are greatly appreciated.

## References

[1] P. L. Várkonyi and G. Domokos, *"Mono-monostatic bodies: the answer to
Arnold's question,"* **The Mathematical Intelligencer**, Volume 28, Number 4,
pp. 34–38 (2006).
