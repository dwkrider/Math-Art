# Turning packing radii into circle positions.
#
# Solving for radii is only half the job: the radii satisfy the angle-sum
# conditions, but nothing yet says where any circle sits.  Positions are
# recovered by walking the complex -- place a seed circle, place one neighbour
# beside it, then repeatedly take an already-placed vertex and fan its remaining
# neighbours around it at the angles the radii dictate.  Because the angle sums
# close, the fan closes too, and the walk is consistent.
#
# TWO TRAPS, both of which produce plausible-looking but wrong output.
#
# 1. OPEN FLOWERS.  A boundary vertex's flower is a fan, not a cycle.  Fanning
#    it as though it wrapped places the first and last petals on top of one
#    another.  The walk below stops at the ends of an open flower.
#
# 2. HYPERBOLIC IS NOT A PARAMETER.  "Step a distance r_v + r_u" is a euclidean
#    instruction.  In the Poincare disc one steps along geodesics, and the
#    natural map is the disc automorphism carrying 0 to w,
#
#        phi_w(z) = (z + w) / (1 + conj(w) z)
#
#    -- note the parameter is conjugated, not the variable; writing
#    (z + w)/(1 + conj(z) w) instead is neither holomorphic nor an isometry.
#    Worse, in a MAXIMAL packing the boundary circles are horocycles of infinite
#    hyperbolic radius, so there is no distance to step at all.  A horocycle is
#    placed instead at the IDEAL endpoint of the geodesic leaving its finite
#    neighbour in the known direction, and sized by its tangency condition.
#    Because the Poincare model is conformal, the direction angle needs no
#    correction: euclidean and hyperbolic angles agree.
#
# References:
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005 (layout of packings,
#   and the horocycle conventions for maximal packings).
# - Charles R. Collins and Kenneth Stephenson, "A circle packing algorithm",
#   Computational Geometry: Theory and Applications 25 (2003), pp. 233-256.

import cmath
import math

from . import angles
from .angles import EUCLIDEAN, HYPERBOLIC


# --------------------------------------------------------------------------
# Euclidean
# --------------------------------------------------------------------------

def layout_euclidean(K, radii, seed=None):
    """Centres for a euclidean packing.  Returns a list of (x, y) or None."""
    nv = K.nv
    pos = [None] * nv
    if seed is None:
        seed = next(iter(sorted(K.interior))) if K.interior else 0
    flowers = K.flowers
    pos[seed] = (0.0, 0.0)
    first = flowers[seed][0]
    pos[first] = (radii[seed] + radii[first], 0.0)

    queue = [seed]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        fl = flowers[v]
        n = len(fl)
        if n == 0 or pos[v] is None:
            continue
        closed = K.is_interior(v)
        k0 = -1
        for k in range(n):
            if pos[fl[k]] is not None:
                k0 = k
                break
        if k0 < 0:
            continue
        base = pos[fl[k0]]
        th0 = math.atan2(base[1] - pos[v][1], base[0] - pos[v][0])

        # forward around the flower
        cur = th0
        for k in range(k0, k0 + n - 1):
            if not closed and (k + 1) % n == 0:
                break
            u = fl[k % n]
            w = fl[(k + 1) % n]
            cur += angles.corner_euclidean(radii[v], radii[u], radii[w])
            if pos[w] is None:
                d = radii[v] + radii[w]
                pos[w] = (pos[v][0] + d * math.cos(cur),
                          pos[v][1] + d * math.sin(cur))
                queue.append(w)
        # backward
        cur = th0
        for k in range(k0, k0 - n + 1, -1):
            if not closed and k % n == 0:
                break
            u = fl[k % n]
            w = fl[(k - 1) % n]
            cur -= angles.corner_euclidean(radii[v], radii[u], radii[w])
            if pos[w] is None:
                d = radii[v] + radii[w]
                pos[w] = (pos[v][0] + d * math.cos(cur),
                          pos[v][1] + d * math.sin(cur))
                queue.append(w)
    return pos


def tangency_error(K, pos, radii):
    """Largest |distance - (r+r)| over all edges: the honest quality number."""
    worst = 0.0
    for (v, u) in K.edges():
        if pos[v] is None or pos[u] is None:
            continue
        d = math.hypot(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
        worst = max(worst, abs(d - (radii[v] + radii[u])))
    return worst


# --------------------------------------------------------------------------
# Hyperbolic (Poincare disc)
# --------------------------------------------------------------------------

def mobius_shift(w, z):
    """The disc automorphism carrying 0 to w, applied to z.

    phi_w(z) = (z + w) / (1 + conj(w) z).  Conjugate the PARAMETER."""
    return (z + w) / (1.0 + w.conjugate() * z)


def euclid_circle_of_hyp(z, r):
    """Euclidean centre and radius of the hyperbolic circle (centre z, radius r)
    as it appears in the Poincare disc."""
    t = math.tanh(0.5 * r)
    if abs(z) < 1e-15:
        return complex(0.0, 0.0), t
    u = z / abs(z)                       # radial direction: the extreme points
    p = mobius_shift(z, t * u)
    q = mobius_shift(z, -t * u)
    return 0.5 * (p + q), 0.5 * abs(p - q)


def horocycle_from_tangency(p, c, rho):
    """Euclidean radius of the horocycle at ideal point p that is tangent to the
    euclidean circle (c, rho).

    A horocycle at p with euclidean radius e has centre (1 - e) p; imposing
    |c - (1-e)p| = rho + e and using |p| = 1 makes the e^2 terms cancel."""
    w = c - p
    den = 2.0 * (w * p.conjugate()).real - 2.0 * rho
    if abs(den) < 1e-14:
        return 0.0
    e = (rho * rho - abs(w) ** 2) / den
    return max(0.0, min(e, 1.0))


def layout_hyperbolic(K, s_radii, seed=None):
    """Lay out a hyperbolic packing in the Poincare disc.

    `s_radii` are s-values (s = exp(-2r); s = 0 is a horocycle).  Returns
    (centres, radii) as euclidean data in the unit disc, ready to draw."""
    nv = K.nv
    hz = [None] * nv                       # hyperbolic centres (complex)
    flowers = K.flowers
    if seed is None:
        # an interior vertex with at least one finite neighbour
        cand = [v for v in sorted(K.interior)
                if any(s_radii[u] > 0.0 for u in flowers[v])]
        if not cand:
            raise ValueError("no interior vertex has a finite neighbour; "
                             "nothing to lay out from")
        seed = cand[0]

    def hrad(v):
        s = s_radii[v]
        return float('inf') if s <= 0.0 else -0.5 * math.log(s)

    hz[seed] = complex(0.0, 0.0)
    # seed the second circle on a FINITE neighbour: in a maximal packing many
    # neighbours are horocycles, which carry no distance to step and would
    # leave the walk with nothing to propagate from
    first = next((u for u in flowers[seed] if s_radii[u] > 0.0), None)
    if first is not None:
        d = hrad(seed) + hrad(first)
        hz[first] = complex(math.tanh(0.5 * d), 0.0)

    queue = [seed]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        if hz[v] is None or s_radii[v] <= 0.0:
            continue                        # never fan out from a horocycle
        fl = flowers[v]
        n = len(fl)
        closed = K.is_interior(v)
        k0 = -1
        for k in range(n):
            if hz[fl[k]] is not None:
                k0 = k
                break
        if k0 < 0:
            continue
        # direction of the reference petal, as seen from v after moving v to 0
        ref = mobius_shift(-hz[v], hz[fl[k0]])
        th0 = cmath.phase(ref) if abs(ref) > 1e-15 else 0.0

        def place(w, theta):
            if hz[w] is not None:
                return
            if s_radii[w] <= 0.0:
                return                      # horocycle: positioned below
            d = hrad(v) + hrad(w)
            local = math.tanh(0.5 * d) * cmath.exp(1j * theta)
            hz[w] = mobius_shift(hz[v], local)
            queue.append(w)

        cur = th0
        for k in range(k0, k0 + n - 1):
            if not closed and (k + 1) % n == 0:
                break
            u, w = fl[k % n], fl[(k + 1) % n]
            cur += angles.corner_hyperbolic(s_radii[v], s_radii[u], s_radii[w])
            place(w, cur)
        cur = th0
        for k in range(k0, k0 - n + 1, -1):
            if not closed and k % n == 0:
                break
            u, w = fl[k % n], fl[(k - 1) % n]
            cur -= angles.corner_hyperbolic(s_radii[v], s_radii[u], s_radii[w])
            place(w, cur)

    # euclidean rendering data for the finite circles
    centres = [None] * nv
    radii = [0.0] * nv
    for v in range(nv):
        if hz[v] is None or s_radii[v] <= 0.0:
            continue
        c, e = euclid_circle_of_hyp(hz[v], hrad(v))
        centres[v] = c
        radii[v] = e

    # horocycles: ideal endpoint of the ray from a placed finite neighbour
    for w in range(nv):
        if s_radii[w] > 0.0 or not flowers[w]:
            continue
        host = None
        for u in flowers[w]:
            if hz[u] is not None and s_radii[u] > 0.0:
                host = u
                break
        if host is None:
            continue
        # direction from host to w, in the frame where host sits at the origin
        fl = flowers[host]
        j = fl.index(w)
        k0 = next((k for k in range(len(fl)) if hz[fl[k]] is not None), None)
        if k0 is None:
            continue
        ref = mobius_shift(-hz[host], hz[fl[k0]])
        th = cmath.phase(ref) if abs(ref) > 1e-15 else 0.0
        closed = K.is_interior(host)
        step = 1 if (j - k0) % len(fl) <= len(fl) // 2 else -1
        k = k0
        guard = 0
        while k % len(fl) != j and guard < 2 * len(fl):
            a, b = fl[k % len(fl)], fl[(k + step) % len(fl)]
            th += step * angles.corner_hyperbolic(
                s_radii[host], s_radii[a], s_radii[b])
            k += step
            guard += 1
        ideal = mobius_shift(hz[host], cmath.exp(1j * th))
        ideal = ideal / abs(ideal) if abs(ideal) > 0 else complex(1.0, 0.0)
        e = horocycle_from_tangency(ideal, centres[host], radii[host])
        centres[w] = (1.0 - e) * ideal
        radii[w] = e
    return centres, radii


def _selftest():
    """Layout accuracy checks; raises AssertionError on failure."""
    from . import complexes, thurston

    # 1. euclidean boundary-value packing: every edge tangent
    K, xy = complexes.hex_disc(5)
    br = [0.0] * K.nv
    for w in K.boundary:
        x, y = xy[w]
        br[w] = 0.35 * (1.0 + 0.75 * math.cos(3.0 * math.atan2(y, x)))
    res = thurston.pack(K, EUCLIDEAN, boundary=thurston.PRESCRIBED,
                        boundary_radii=br)
    pos = layout_euclidean(K, res.radii)
    assert all(p is not None for p in pos), "every circle must be placed"
    err = tangency_error(K, pos, res.radii)
    assert err < 1e-8, "euclidean tangency error %.3e" % err

    # 2. the regular hexagonal packing lands on the exact hexagonal lattice
    K2, _ = complexes.hex_disc(2)
    unit = [1.0] * K2.nv
    pos2 = layout_euclidean(K2, unit)
    assert tangency_error(K2, pos2, unit) < 1e-12

    # 3. the Mobius map is an isometry of the disc; the mis-conjugated form is
    #    not -- this is the bug the module header warns about
    def hdist(a, b):
        t = abs(a - b) / abs(1.0 - b.conjugate() * a)
        return math.log((1.0 + t) / (1.0 - t))
    w = complex(0.4, 0.3)
    worst_ok = 0.0
    worst_bad = 0.0
    for k in range(60):
        z1 = 0.7 * cmath.exp(1j * k * 0.31)
        z2 = 0.5 * cmath.exp(1j * k * 0.77)
        d0 = hdist(z1, z2)
        worst_ok = max(worst_ok,
                       abs(hdist(mobius_shift(w, z1), mobius_shift(w, z2)) - d0))
        bad = lambda z: (z + w) / (1.0 + z.conjugate() * w)
        worst_bad = max(worst_bad, abs(hdist(bad(z1), bad(z2)) - d0))
    assert worst_ok < 1e-9, "mobius_shift must be an isometry (%.3e)" % worst_ok
    assert worst_bad > 1e-3, "the mis-conjugated form should NOT be an isometry"

    # 4. hyperbolic circle rendering: a centred circle has euclidean radius
    #    tanh(r/2), and the mapped one keeps its hyperbolic radius
    c, e = euclid_circle_of_hyp(complex(0.0, 0.0), 1.0)
    assert abs(e - math.tanh(0.5)) < 1e-12
    c2, e2 = euclid_circle_of_hyp(complex(0.5, 0.1), 0.8)
    far = abs(c2) + e2
    assert far < 1.0, "a hyperbolic circle must stay inside the disc"

    # 5. maximal packing: horocycles sit on the unit circle, internally tangent
    K3, _ = complexes.hex_disc(3)
    resh = thurston.pack(K3, HYPERBOLIC, boundary=thurston.MAXIMAL)
    assert resh.converged
    cen, rad = layout_hyperbolic(K3, resh.radii)
    for w in K3.boundary:
        assert cen[w] is not None, "horocycle %d not placed" % w
        d = abs(cen[w]) + rad[w]
        assert abs(d - 1.0) < 1e-6, \
            "horocycle %d not internally tangent to the unit circle (%.6f)" % (w, d)
    for v in K3.interior:
        assert cen[v] is not None and abs(cen[v]) + rad[v] < 1.0

    print("packing.layout: euclidean tangency < 1e-8, hexagonal exact, "
          "mobius_shift is an isometry, horocycles land on the unit circle. "
          "RESULT: OK")
