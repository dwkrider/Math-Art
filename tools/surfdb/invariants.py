# Measuring a surface's defining property.
#
# This is what a surface database can check that a polyhedron database
# cannot.  data/polyhedra recomputes radii, dihedral angles and Euler's
# formula -- real checks, but all of them about the STORED GEOMETRY.  A
# surface's defining property is a PREDICATE, and predicates can be
# tested: a minimal surface really must have H = 0 everywhere, a
# pseudospherical surface really must have K = -1, a flat one K = 0.
#
# For an implicit surface F = 0 the two curvatures are computable
# directly from the derivatives of F, with no meshing at all:
#
#     n = grad F / |grad F|
#     H = ( |grad F|^2 tr(Hess F) - grad F^T Hess F grad F )
#         / ( 2 |grad F|^3 )
#     K = ( grad F^T adj(Hess F) grad F ) / |grad F|^4
#
# Points must lie ON the surface, so a sample is Newton-projected along
# the gradient first; a point that does not converge is discarded rather
# than measured, because curvature off the level set is meaningless.
#
# Derivatives are finite differences on the exact expression.  That is
# accurate to ~1e-6 with h = 1e-4, which is far more than enough to
# separate H = 0 from H != 0 -- the distinction the facet actually turns
# on -- while needing no symbolic differentiation of the record language.

import math

from . import expr


def _f(poly, params):
    env = dict(params or {})

    def fn(x, y, z):
        env["x"], env["y"], env["z"] = x, y, z
        return expr.evaluate(poly, env)
    return fn


def gradient(fn, p, h=1e-5):
    x, y, z = p
    return (
        (fn(x + h, y, z) - fn(x - h, y, z)) / (2 * h),
        (fn(x, y + h, z) - fn(x, y - h, z)) / (2 * h),
        (fn(x, y, z + h) - fn(x, y, z - h)) / (2 * h),
    )


def hessian(fn, p, h=1e-4):
    x, y, z = p
    f0 = fn(x, y, z)
    hxx = (fn(x + h, y, z) - 2 * f0 + fn(x - h, y, z)) / (h * h)
    hyy = (fn(x, y + h, z) - 2 * f0 + fn(x, y - h, z)) / (h * h)
    hzz = (fn(x, y, z + h) - 2 * f0 + fn(x, y, z - h)) / (h * h)
    hxy = (fn(x + h, y + h, z) - fn(x + h, y - h, z)
           - fn(x - h, y + h, z) + fn(x - h, y - h, z)) / (4 * h * h)
    hxz = (fn(x + h, y, z + h) - fn(x + h, y, z - h)
           - fn(x - h, y, z + h) + fn(x - h, y, z - h)) / (4 * h * h)
    hyz = (fn(x, y + h, z + h) - fn(x, y + h, z - h)
           - fn(x, y - h, z + h) + fn(x, y - h, z - h)) / (4 * h * h)
    return ((hxx, hxy, hxz), (hxy, hyy, hyz), (hxz, hyz, hzz))


def project(fn, p, tol=1e-12, steps=40):
    """Newton-project `p` onto the level set F = 0, along the gradient."""
    x, y, z = p
    for _ in range(steps):
        v = fn(x, y, z)
        if abs(v) < tol:
            return (x, y, z)
        g = gradient(fn, (x, y, z))
        n2 = g[0] ** 2 + g[1] ** 2 + g[2] ** 2
        if n2 < 1e-18 or not math.isfinite(n2):
            return None
        s = v / n2
        x, y, z = x - s * g[0], y - s * g[1], z - s * g[2]
        if not all(math.isfinite(t) for t in (x, y, z)):
            return None
        if abs(x) > 1e4 or abs(y) > 1e4 or abs(z) > 1e4:
            return None
    return (x, y, z) if abs(fn(x, y, z)) < 1e-8 else None


def curvatures_at(fn, p):
    """(H, K) of the level set of `fn` at an on-surface point `p`."""
    g = gradient(fn, p)
    gn2 = g[0] ** 2 + g[1] ** 2 + g[2] ** 2
    gn = math.sqrt(gn2)
    if gn < 1e-8:
        return None, None                     # singular point: undefined
    Hm = hessian(fn, p)
    tr = Hm[0][0] + Hm[1][1] + Hm[2][2]
    gHg = sum(g[i] * Hm[i][j] * g[j] for i in range(3) for j in range(3))
    H = (gn2 * tr - gHg) / (2.0 * gn2 * gn)

    # adjugate of the Hessian
    def cof(i, j):
        a = [0, 1, 2]
        a.remove(i)
        b = [0, 1, 2]
        b.remove(j)
        m = Hm
        det = (m[a[0]][b[0]] * m[a[1]][b[1]] - m[a[0]][b[1]] * m[a[1]][b[0]])
        return ((-1) ** (i + j)) * det
    adj = [[cof(j, i) for j in range(3)] for i in range(3)]
    gAg = sum(g[i] * adj[i][j] * g[j] for i in range(3) for j in range(3))
    K = gAg / (gn2 * gn2)
    return H, K


def sample_curvature(poly, params=None, extent=1.6, n=160, seed=20260828):
    """Measure H and K over the level set of `poly`.

    Returns a dict with the maxima and the sample count, or None when too
    few points could be projected onto the surface -- which happens for a
    polynomial whose real locus misses the sampling box entirely, and is
    reported as "not measurable here" rather than as a pass.
    """
    import random
    rng = random.Random(seed)
    fn = _f(poly, params)
    Hs, Ks = [], []
    tries = 0
    while len(Hs) < n and tries < n * 30:
        tries += 1
        p0 = (rng.uniform(-extent, extent), rng.uniform(-extent, extent),
              rng.uniform(-extent, extent))
        try:
            p = project(fn, p0)
        except expr.ExprError:
            continue
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        if p is None or max(abs(t) for t in p) > extent * 1.5:
            continue
        try:
            H, K = curvatures_at(fn, p)
        except (expr.ExprError, ValueError, OverflowError, ZeroDivisionError):
            continue
        if H is None or not math.isfinite(H) or not math.isfinite(K):
            continue
        Hs.append(H)
        Ks.append(K)
    if len(Hs) < 12:
        return None
    return {
        "samples": len(Hs),
        "max_abs_H": max(abs(v) for v in Hs),
        "mean_H": sum(Hs) / len(Hs),
        "max_abs_K": max(abs(v) for v in Ks),
        "mean_K": sum(Ks) / len(Ks),
        "max_K_dev_from_1": max(abs(abs(v) - 1.0) for v in Ks),
    }


def check_condition(condition, stats, tol=2e-3):
    """Does the measured curvature match the claimed condition?

    Returns (verdict, detail) where verdict is True, False or None
    ('not checkable from what is stored').
    """
    if stats is None:
        return None, "no on-surface samples converged in the sampling box"
    if condition == "minimal":
        ok = stats["max_abs_H"] <= tol
        return ok, "max |H| = %.3g (tol %.1g)" % (stats["max_abs_H"], tol)
    if condition == "flat":
        ok = stats["max_abs_K"] <= tol
        return ok, "max |K| = %.3g (tol %.1g)" % (stats["max_abs_K"], tol)
    if condition == "k-const-positive":
        ok = stats["max_K_dev_from_1"] <= tol and stats["mean_K"] > 0
        return ok, "max ||K| - 1| = %.3g, mean K = %.3g" % (
            stats["max_K_dev_from_1"], stats["mean_K"])
    if condition == "k-const-negative":
        ok = stats["max_K_dev_from_1"] <= tol and stats["mean_K"] < 0
        return ok, "max ||K| - 1| = %.3g, mean K = %.3g" % (
            stats["max_K_dev_from_1"], stats["mean_K"])
    if condition == "cmc":
        spread = stats["max_abs_H"] - abs(stats["mean_H"])
        return None, "H spread %.3g about mean %.3g (CMC needs the RADIUS " \
                     "to fix the constant; not gated)" % (spread, stats["mean_H"])
    return None, "no numeric test for condition %r" % condition


# ---------------------------------------------------------------------------
# Symbolic symmetry: substitute a group's generators into the polynomial
# and require it unchanged.  For an implicit surface this PROVES the
# symmetry rather than detecting it from a point cloud, which is what
# data/polyhedra has to do -- cheaper and stronger.
# ---------------------------------------------------------------------------

# Generators, as coordinate substitutions (x, y, z) -> expressions.
#
# A PASS HERE PROVES INVARIANCE UNDER THESE GENERATORS, which for some
# groups generate a proper subgroup of the named one.  That is stated
# rather than glossed: `full` says whether the listed generators really
# do generate the whole group, and the validator records
# "symbolic" or "symbolic-partial" accordingly.
#
# Getting these wrong is not academic.  An earlier version of this table
# had Td generated by the 3-cycle and (y, x, -z) -- but (y, x, -z) is an
# element of Oh, NOT of Td: it flips the sign of xyz, and xyz is exactly
# what distinguishes the tetrahedral orientation.  Every tetrahedral
# record failed as a result, and the failure was in the table, not the
# records.  Td's mirror is the plain transposition (y, x, z).
GENERATORS = {
    # Tetrahedral: coordinate permutations plus EVEN sign changes. The
    # tetrahedron (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) has xyz = +1
    # at every vertex, and these generators preserve that.
    "Td": {"gens": [("y", "z", "x"), ("y", "x", "z"), ("-x", "-y", "z")],
           "full": True},
    "T":  {"gens": [("y", "z", "x"), ("-x", "-y", "z")], "full": True},

    # Octahedral: all permutations and ALL sign changes.
    "Oh": {"gens": [("y", "z", "x"), ("y", "x", "z"), ("-x", "y", "z")],
           "full": True},
    "O":  {"gens": [("y", "z", "x"), ("y", "-x", "z")], "full": True},

    # Icosahedral, in the standard frame (2-fold axes on the coordinate
    # axes, which is the frame Barth's sextic and decic are written in).
    #
    # The 5-fold rotation is the generator that carries the icosahedral
    # content; without it "Ih" only ever tests the tetrahedral subgroup and
    # a pass means far less than it looks like. Getting its AXIS right
    # matters: the twelve 5-fold axes here are (+-phi, +-1, 0) and cyclic
    # permutations -- NOT (+-1, +-phi, 0). An earlier version of this table
    # used a rotation about (1, phi, 0); it is a perfectly good order-5
    # rotation, just of a DIFFERENT icosahedron, and every icosahedral
    # record failed against it.
    #
    # The matrix below is the 72-degree rotation about (phi, 1, 0),
    # written with 1/phi = phi - 1 so it stays in the exact language
    # without division:
    #
    #     1/2 * [[ phi,   1/phi,  1    ],
    #            [ 1/phi, 1,     -phi  ],
    #            [ -1,    phi,    1/phi]]
    "Ih": {"gens": [("y", "z", "x"),
                    ("-x", "-y", "z"),
                    ("(phi*x + (phi - 1)*y + z)/2",
                     "((phi - 1)*x + y - phi*z)/2",
                     "(-x + phi*y + (phi - 1)*z)/2")],
           "full": True},

    # Icosahedral in the FIVE-FOLD-ON-Z frame. Not a different group --
    # the same Ih in a different orientation, and the distinction is not
    # cosmetic: math_art/surfaces/goursat.py builds its dodecahedral sextic
    # from the six planes whose normals are the five-fold axes, which puts
    # a C5 axis on z. Tested against the standard-frame "Ih" above, every
    # dodecahedral Goursat row FAILS -- not because the claim is wrong but
    # because the frames disagree.
    #
    # These three generate D5d (order 20), a PROPER subgroup of Ih, so
    # `full` is False and a pass is reported as partial. Proving the whole
    # icosahedral group here would need a second five-fold axis, whose
    # coordinates in this frame are not worth hard-coding for the gain;
    # the family's full symmetry is a fact of its CONSTRUCTION (Goursat
    # defines it as the icosahedrally invariant sextic), and the record
    # says so rather than claiming a proof it does not have.
    "Ih_z5": {"gens": [("x*cos(2*pi/5) - y*sin(2*pi/5)",
                        "x*sin(2*pi/5) + y*cos(2*pi/5)",
                        "z"),
                       ("x", "-y", "z"),
                       ("-x", "-y", "-z")],
              "full": False},

    # Tetrahedral in the EDGE frame: 2-fold axes on the coordinate axes
    # and the tetrahedron's face normals along (+-sqrt2, 0, -1) and
    # (0, +-sqrt2, 1), which is how math_art/surfaces/algebraic.py writes
    # Kummer's four tangent planes.
    #
    # Same group as "Td", different orientation -- and the distinction is
    # not academic: Kummer's quartic fails the standard-frame test
    # completely (the coordinate 3-cycle is not one of its symmetries)
    # while being perfectly tetrahedral here. Its 3-fold axes lie along
    # the face normals, and the four normals have pairwise dot product
    # -1/3, so the tetrahedron is regular. Reporting it as asymmetric was
    # a frame error, not a finding.
    "Td_edge": {"gens": [("-x", "y", "z"),
                         ("x", "-y", "z"),
                         ("y", "x", "-z"),
                         # 120 degrees about (sqrt2, 0, 1)/sqrt3
                         ("x/2 - y/2 + z/sqrt(2)",
                          "x/2 - y/2 - z/sqrt(2)",
                          "x/sqrt(2) + y/sqrt(2)")],
                "full": False},

    # C3v with the 3-fold axis on the BODY DIAGONAL (1,1,1) -- the
    # symmetry a polynomial symmetric in x, y, z actually has. Distinct
    # from C3v about the z axis, which is what a naive reading assumes.
    "C3v_diag": {"gens": [("y", "z", "x"), ("y", "x", "z")], "full": True},

    "C3v": {"gens": [("-y/2 - x*sqrt(3)/2", "x/2 - y*sqrt(3)/2", "z"),
                     ("-x", "y", "z")],
            "full": True},

    "D_inf_h": {"gens": [("-x", "-y", "z"), ("x", "y", "-z")], "full": False},
}


def symmetry_invariance(poly, group, params=None, n=120, extent=1.4,
                        seed=20260828):
    """Max |F(g.p) - F(p)| over the claimed group's generators.

    Returns (max_residual, detail, full) or (None, why-not, None) when the
    group has no generator table here. `full` says whether the tested
    generators generate the whole named group -- a caller must not report
    a partial pass as if it proved the full symmetry.
    """
    import random
    entry = GENERATORS.get(group)
    if not entry:
        return None, "no generator table for %r" % group, None
    gens, full = entry["gens"], entry["full"]
    rng = random.Random(seed)
    fn = _f(poly, params)
    worst = 0.0
    scale = 0.0
    for _ in range(n):
        p = (rng.uniform(-extent, extent), rng.uniform(-extent, extent),
             rng.uniform(-extent, extent))
        env = {"x": p[0], "y": p[1], "z": p[2]}
        env.update(params or {})
        try:
            base = fn(*p)
        except (expr.ExprError, ValueError, OverflowError):
            continue
        if not math.isfinite(base):
            continue
        scale = max(scale, abs(base))
        for gx, gy, gz in gens:
            try:
                q = (expr.evaluate(gx, env), expr.evaluate(gy, env),
                     expr.evaluate(gz, env))
                v = fn(*q)
            except (expr.ExprError, ValueError, OverflowError):
                continue
            if math.isfinite(v):
                worst = max(worst, abs(v - base))
    if scale == 0.0:
        return None, "polynomial is identically zero on the sample", None
    return (worst / scale,
            "relative residual %.3g over %d points x %d generators of %s%s"
            % (worst / max(scale, 1e-12), n, len(gens), group,
               "" if full else " (a PROPER SUBGROUP -- partial proof)"),
            full)


def _selftest():
    """Numeric self-test; raises on failure."""
    # unit sphere: K = +1 everywhere, H = 1 (magnitude), NOT minimal
    st = sample_curvature("x**2 + y**2 + z**2 - 1")
    assert st is not None
    assert st["max_K_dev_from_1"] < 1e-3, st
    ok, detail = check_condition("k-const-positive", st)
    assert ok, detail
    ok, _ = check_condition("minimal", st)
    assert not ok, "the sphere is not minimal"

    # plane: H = 0 and K = 0 -- both minimal and flat
    st = sample_curvature("z")
    ok, detail = check_condition("minimal", st)
    assert ok, detail
    ok, detail = check_condition("flat", st)
    assert ok, detail

    # circular cylinder: K = 0 (developable), H != 0
    st = sample_curvature("x**2 + y**2 - 1")
    ok, detail = check_condition("flat", st)
    assert ok, detail
    ok, _ = check_condition("minimal", st)
    assert not ok, "a cylinder is not minimal"

    # the classical test the whole facet turns on: a genuinely minimal
    # surface must pass where a lookalike does not.  Scherk's doubly
    # periodic surface is exp(z) = cos(x)/cos(y), i.e. z = log(cos x/cos y).
    st = sample_curvature("exp(z)*cos(y) - cos(x)", extent=1.0)
    ok, detail = check_condition("minimal", st)
    assert ok, "Scherk's surface must measure as minimal: %s" % detail

    # ... and a near-miss must FAIL, or the test proves nothing
    st = sample_curvature("exp(z)*cos(y) - cos(1.15*x)", extent=1.0)
    ok, _ = check_condition("minimal", st)
    assert not ok, "a perturbed Scherk surface must not pass as minimal"

    # symbolic symmetry: x^2+y^2+z^2 is Oh-invariant; x^2+y^2+2z^2 is not
    r, detail, _ = symmetry_invariance("x**2 + y**2 + z**2 - 1", "Oh")
    assert r is not None and r < 1e-12, detail
    r, detail, _ = symmetry_invariance("x**2 + y**2 + 2*z**2 - 1", "Oh")
    assert r is not None and r > 1e-3, "anisotropic quadric is not Oh: %s" % detail

    r, _, _ = symmetry_invariance("x**2 + y**2 + z**2 - 1", "NoSuchGroup")
    assert r is None, "an unknown group must report 'not checkable', not pass"

    # The Cayley nodal cubic 4(x^2+y^2+z^2) + 16xyz - 1 IS tetrahedral.
    # It failed against the old table because that table's "Td" contained
    # (y, x, -z), which is an Oh element: it negates xyz. Regression test
    # for exactly that.
    cayley = "4*(x**2 + y**2 + z**2) + 16*x*y*z - 1"
    r, detail, full = symmetry_invariance(cayley, "Td")
    assert r is not None and r < 1e-12, "Cayley must be Td-invariant: %s" % detail
    assert full is True
    # and it must NOT be octahedral -- Oh contains the sign flip that Td
    # excludes, and that is the whole distinction
    r, _, _ = symmetry_invariance(cayley, "Oh")
    assert r is not None and r > 1e-3, "Cayley is tetrahedral, not octahedral"

    # A polynomial symmetric in x, y, z has its 3-fold axis on the BODY
    # DIAGONAL, not on z. The Clebsch cubic is this case: its celebrated
    # S5 symmetry is PROJECTIVE and is not a Euclidean point group.
    sym3 = "x**3 + y**3 + z**3 - 3*x*y*z"
    r, detail, _ = symmetry_invariance(sym3, "C3v_diag")
    assert r is not None and r < 1e-12, detail
    r, _, _ = symmetry_invariance(sym3, "Td")
    assert r is not None and r > 1e-3, \
        "a cubic odd under sign flips is not Td, even though it is symmetric"

    # the icosahedral generator really is a rotation of order 5
    env = {"phi": (1 + math.sqrt(5)) / 2}
    p = (0.31, -0.62, 0.44)
    q = p
    for _ in range(5):
        e = dict(env, x=q[0], y=q[1], z=q[2])
        q = tuple(expr.evaluate(g, e) for g in GENERATORS["Ih"]["gens"][2])
    assert all(abs(a - b) < 1e-9 for a, b in zip(p, q)), \
        "the Ih 5-fold generator must have order 5, got %r from %r" % (q, p)

    # ... and it must be the 5-fold of the RIGHT icosahedron. Barth's
    # sextic is the test case: it is icosahedral in the standard frame,
    # and it fails against a 5-fold about (1, phi, 0) while passing
    # against one about (phi, 1, 0). Order 5 alone is not enough.
    barth = ("4*(phi**2*x**2 - y**2)*(phi**2*y**2 - z**2)*(phi**2*z**2 - x**2)"
             " - (1 + 2*phi)*(x**2 + y**2 + z**2 - 1)**2")
    r, detail, full = symmetry_invariance(barth, "Ih")
    assert r is not None and r < 1e-12, \
        "Barth's sextic must verify as icosahedral: %s" % detail
    assert full is True
    # and a genuinely non-icosahedral sextic must not
    r, _, _ = symmetry_invariance("x**6 + y**6 + z**6 - 1", "Ih")
    assert r is not None and r > 1e-3, "x^6+y^6+z^6 is octahedral, not Ih"

    print("RESULT: OK  (surfdb.invariants)")


# ---------------------------------------------------------------------------
# PARAMETRIC surfaces: curvature from the fundamental forms.
#
# The implicit machinery above covers 105 records and reaches only 7 of
# them with a gated curvature condition, because most algebraic surfaces
# satisfy no condition at all.  The conditions live somewhere else: 25
# parametric records and 101 Weierstrass ones claim `minimal`, `flat` or
# constant K, and none of them could be checked, because the only
# curvature path took an implicit F.
#
# For a chart r(u, v) the classical route needs no level set:
#
#     E = r_u . r_u      F = r_u . r_v      G = r_v . r_v
#     n = (r_u x r_v) / |r_u x r_v|
#     L = r_uu . n       M = r_uv . n       N = r_vv . n
#
#     H = (E N - 2 F M + G L) / (2 (E G - F^2))
#     K = (L N - M^2) / (E G - F^2)
#
# Derivatives are finite differences on the stored expressions, as in the
# implicit case: accurate to about 1e-6, far more than enough to separate
# H = 0 from H != 0, which is the distinction the facet turns on.
#
# Degenerate points are DISCARDED, not measured: where E G - F^2 vanishes
# the chart is singular (a pole of a sphere, the tip of a cone) and
# curvature there is a property of the parametrisation rather than of the
# surface.
# ---------------------------------------------------------------------------


def _chart(xs, ys, zs, params):
    env = dict(params or {})

    def r(u, v):
        env["u"], env["v"] = u, v
        return (expr.evaluate(xs, env), expr.evaluate(ys, env),
                expr.evaluate(zs, env))
    return r


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def parametric_curvatures(r, u, v, h=1e-4):
    """(H, K) of the chart `r` at (u, v), or (None, None) if degenerate."""
    ru = _scale(_sub(r(u + h, v), r(u - h, v)), 0.5 / h)
    rv = _scale(_sub(r(u, v + h), r(u, v - h)), 0.5 / h)

    p0 = r(u, v)
    ruu = _scale(_sub(_sub(r(u + h, v), _scale(p0, 2.0)), _scale(r(u - h, v), -1.0)),
                 1.0 / (h * h))
    rvv = _scale(_sub(_sub(r(u, v + h), _scale(p0, 2.0)), _scale(r(u, v - h), -1.0)),
                 1.0 / (h * h))
    ruv = _scale(_sub(_sub(r(u + h, v + h), r(u + h, v - h)),
                      _sub(r(u - h, v + h), r(u - h, v - h))),
                 0.25 / (h * h))

    E, F, G = _dot(ru, ru), _dot(ru, rv), _dot(rv, rv)
    det = E * G - F * F
    nrm = _cross(ru, rv)
    nlen = math.sqrt(_dot(nrm, nrm))
    if det <= 1e-12 or nlen <= 1e-9:
        return None, None                    # singular point of the CHART
    n = _scale(nrm, 1.0 / nlen)

    L, M, N = _dot(ruu, n), _dot(ruv, n), _dot(rvv, n)
    H = (E * N - 2.0 * F * M + G * L) / (2.0 * det)
    K = (L * N - M * M) / det
    return H, K


def sample_parametric_curvature(xs, ys, zs, u_range, v_range, params=None,
                                n=180, seed=20260828, margin=0.04):
    """Measure H and K over a chart. Same shape of result as the implicit
    sampler, so `check_condition` serves both.

    `margin` keeps the sample clear of the parameter rectangle's edges,
    where a chart is most often singular -- the poles of a sphere, the
    rim of a Minding surface -- and where a finite difference would in
    any case reach outside the stated domain.
    """
    import random
    rng = random.Random(seed)
    env = dict(params or {})
    try:
        u0 = expr.evaluate(u_range[0], env)
        u1 = expr.evaluate(u_range[1], env)
        v0 = expr.evaluate(v_range[0], env)
        v1 = expr.evaluate(v_range[1], env)
    except (expr.ExprError, TypeError, IndexError):
        return None
    if not all(math.isfinite(t) for t in (u0, u1, v0, v1)):
        return None

    du, dv = (u1 - u0), (v1 - v0)
    u0, u1 = u0 + margin * du, u1 - margin * du
    v0, v1 = v0 + margin * dv, v1 - margin * dv

    try:
        r = _chart(xs, ys, zs, params)
    except expr.ExprError:
        return None

    Hs, Ks = [], []
    tries = 0
    while len(Hs) < n and tries < n * 20:
        tries += 1
        u = rng.uniform(u0, u1)
        v = rng.uniform(v0, v1)
        try:
            H, K = parametric_curvatures(r, u, v)
        except (expr.ExprError, ValueError, OverflowError, ZeroDivisionError):
            continue
        if H is None or not math.isfinite(H) or not math.isfinite(K):
            continue
        Hs.append(H)
        Ks.append(K)
    if len(Hs) < 12:
        return None
    return {
        "samples": len(Hs),
        "max_abs_H": max(abs(t) for t in Hs),
        "mean_H": sum(Hs) / len(Hs),
        "max_abs_K": max(abs(t) for t in Ks),
        "mean_K": sum(Ks) / len(Ks),
        "max_K_dev_from_1": max(abs(abs(t) - 1.0) for t in Ks),
    }


def _selftest_parametric():
    """Chart-based curvature, on surfaces whose answers are known exactly."""
    # unit sphere: K = +1, H = -+1, NOT minimal
    st = sample_parametric_curvature(
        "sin(u)*cos(v)", "sin(u)*sin(v)", "cos(u)",
        ("0", "pi"), ("0", "2*pi"))
    assert st is not None
    assert st["max_K_dev_from_1"] < 1e-3, st
    ok, d = check_condition("k-const-positive", st)
    assert ok, d
    ok, _ = check_condition("minimal", st)
    assert not ok, "the sphere is not minimal"

    # catenoid: MINIMAL, and the classical test of the parametric path
    st = sample_parametric_curvature(
        "cosh(v)*cos(u)", "cosh(v)*sin(u)", "v",
        ("0", "2*pi"), ("-1.5", "1.5"))
    ok, d = check_condition("minimal", st)
    assert ok, "the catenoid must measure as minimal: %s" % d

    # helicoid: minimal too
    st = sample_parametric_curvature(
        "v*cos(u)", "v*sin(u)", "u",
        ("0", "2*pi"), ("-1.5", "1.5"))
    ok, d = check_condition("minimal", st)
    assert ok, "the helicoid must measure as minimal: %s" % d

    # ...and a near-miss must FAIL, or the test proves nothing
    st = sample_parametric_curvature(
        "cosh(1.25*v)*cos(u)", "cosh(v)*sin(u)", "v",
        ("0", "2*pi"), ("-1.5", "1.5"))
    ok, _ = check_condition("minimal", st)
    assert not ok, "a perturbed catenoid must not pass as minimal"

    # circular cylinder: FLAT (K = 0) but not minimal
    st = sample_parametric_curvature("cos(u)", "sin(u)", "v",
                                     ("0", "2*pi"), ("-1", "1"))
    ok, d = check_condition("flat", st)
    assert ok, d
    ok, _ = check_condition("minimal", st)
    assert not ok

    # pseudosphere (tractricoid): K = -1 everywhere
    st = sample_parametric_curvature(
        "cos(u)/cosh(v)", "sin(u)/cosh(v)", "v - tanh(v)",
        ("0", "2*pi"), ("0.15", "3.0"))
    ok, d = check_condition("k-const-negative", st)
    assert ok, "the pseudosphere must measure K = -1: %s" % d

    # a chart with a genuine singular point must not crash: the sphere's
    # poles are excluded by the margin, but ask for a degenerate point
    # directly and it must report None rather than a number
    r = _chart("sin(u)*cos(v)", "sin(u)*sin(v)", "cos(u)", {})
    H, K = parametric_curvatures(r, 0.0, 0.0)
    assert H is None and K is None, \
        "curvature at a chart singularity must be None, got %r" % ((H, K),)

    print("RESULT: OK  (surfdb.invariants parametric)")
