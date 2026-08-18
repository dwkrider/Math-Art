# Alexander polynomial via the reduced Burau representation, and from
# a 3D curve via a generic planar projection.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# Two independent routes to the same invariant:
#   * `alexander_at(word)` -- from a braid word, through the reduced
#     Burau representation.
#   * `alexander_from_curve(P)` -- from a closed 3D polyline, through a
#     generic planar projection: crossings -> Wirtinger presentation ->
#     Fox free-calculus Alexander matrix -> one minor's determinant,
#     computed fraction-free over Z[t] (Bareiss).  This is what lets a
#     geometric flow assert "the knot type did not change" without any
#     braid bookkeeping: the Alexander polynomial before and after must
#     agree.
#
# References:
#   J. W. Alexander, "Topological invariants of knots and links",
#       Trans. AMS 30 (1928).
#   W. Burau, "Ueber Zopfgruppen und gleichsinnig verdrillte
#       Verkettungen", Abh. Math. Semin. Univ. Hambg. 11 (1935).
#   R. H. Fox, "Free differential calculus. I", Annals of Mathematics
#       57 (1953) -- the derivation of the Alexander matrix from the
#       Wirtinger presentation used by `alexander_from_curve`.
#   E. H. Bareiss, "Sylvester's identity and multistep
#       integer-preserving Gaussian elimination", Math. Comp. 22
#       (1968) -- the fraction-free determinant over Z[t].
#   At x = -1 the value is the knot determinant.  That is NOT what
#   `tables.py` records: its third column is Gittings' published value
#   at t = 10, this module's default evaluation point.  The trefoil's
#   determinant is 3; its table entry is 91.

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(n)]


def _pneg(a):
    return [-x for x in a]


def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _ptrim(a):
    while a and a[-1] == 0:
        a.pop()
    return a or [0]


def _pdivexact(a, b):
    a = list(a)
    b = _ptrim(list(b))
    out = [0] * (len(a) - len(b) + 1)
    for k in range(len(out) - 1, -1, -1):
        c, r = divmod(a[len(b) - 1 + k], b[-1])
        if r:
            raise ValueError("inexact division")
        out[k] = c
        for j, y in enumerate(b):
            a[j + k] -= c * y
    if _ptrim(a) != [0]:
        raise ValueError("nonzero remainder")
    return _ptrim(out)


def alexander_at(word, x=10):
    """|Alexander polynomial| of the braid closure at t=x, from the
    reduced Burau representation (quotient of the unreduced one by
    its fixed vector; inverse generators are scaled by t, giving
    t^k * rho, handled by det(P - t^k I))."""
    n = max(abs(g) for g in word) + 1
    one, t = [1], [0, 1]

    def fold(col):
        last = col[n - 1]
        return [_ptrim(_padd(col[i], _pneg(last)))
                for i in range(n - 1)]

    def gen(i, inv):
        diag = t if inv else one
        M = [[diag if r == c else [0] for c in range(n)]
             for r in range(n)]
        if not inv:
            M[i][i] = [1, -1]
            M[i][i + 1] = t
            M[i + 1][i] = one
            M[i + 1][i + 1] = [0]
        else:
            M[i][i] = [0]
            M[i][i + 1] = t
            M[i + 1][i] = one
            M[i + 1][i + 1] = [-1, 1]
        cols = [fold([M[r][j] for r in range(n)])
                for j in range(n - 1)]
        return [[cols[j][r] for j in range(n - 1)]
                for r in range(n - 1)]

    m = n - 1
    R = [[one if i == j else [0] for j in range(m)] for i in range(m)]
    for g in word:
        G = gen(abs(g) - 1, g < 0)
        R = [[_ptrim(_padd([0], _prowcol(G, R, r, c, m)))
              for c in range(m)] for r in range(m)]
    k = sum(1 for g in word if g < 0)
    tk = [0] * k + [1]
    for i in range(m):
        R[i][i] = _ptrim(_padd(R[i][i], _pneg(tk)))
    q = _det(R)
    r = _pdivexact(q, [1] * n)
    while r[0] == 0:
        r = r[1:]
    return abs(sum(c * x ** i for i, c in enumerate(r)))


def _prowcol(A, B, r, c, m):
    acc = [0]
    for k in range(m):
        acc = _padd(acc, _pmul(A[r][k], B[k][c]))
    return acc


def _det(M):
    m = len(M)
    if m == 1:
        return M[0][0]
    det = [0]
    for c in range(m):
        minor = [[M[r][cc] for cc in range(m) if cc != c]
                 for r in range(1, m)]
        term = _pmul(M[0][c], _det(minor))
        det = _padd(det, term if c % 2 == 0 else _pneg(term))
    return _ptrim(det)


# ----------------------------------------------------------------------
# Alexander polynomial straight from a closed 3D polyline
# ----------------------------------------------------------------------

def _psub(a, b):
    return _padd(a, _pneg(b))


def _pbareiss_det(M):
    """Determinant of a square matrix of Z[t] polynomials (coefficient
    lists) by fraction-free Bareiss elimination -- every division is
    exact in Z[t], so the arithmetic stays integer throughout."""
    m = len(M)
    if m == 0:
        return [1]
    A = [[_ptrim(list(x)) for x in row] for row in M]
    sign = 1
    prev = [1]
    for k in range(m - 1):
        if A[k][k] == [0]:
            for r in range(k + 1, m):
                if A[r][k] != [0]:
                    A[k], A[r] = A[r], A[k]
                    sign = -sign
                    break
            else:
                return [0]
        for i in range(k + 1, m):
            for j in range(k + 1, m):
                num = _ptrim(_psub(_pmul(A[k][k], A[i][j]),
                                   _pmul(A[i][k], A[k][j])))
                A[i][j] = ([0] if num == [0]
                           else _pdivexact(num, prev))
        prev = A[k][k]
    det = A[m - 1][m - 1]
    return det if sign == 1 else _pneg(det)


def _project_crossings(P, R):
    """Crossing data of the polyline P under the rotation R, or None if
    the projection is not generic (near-tangential crossing, crossing
    at a vertex, coincident depths or underpass parameters).

    Returns a list of (under_param, over_param, sign) with parameters
    measured along the curve in [0, n)."""
    import numpy as np
    Q = np.asarray(P, dtype=float) @ R.T
    n = len(Q)
    xy = Q[:, :2]
    z = Q[:, 2]
    d = np.roll(xy, -1, axis=0) - xy
    dz = np.roll(z, -1) - z
    scale = float(np.linalg.norm(xy.max(0) - xy.min(0))) or 1.0

    i, j = np.nonzero(np.triu(np.ones((n, n), dtype=bool), 2))
    keep = ~((i == 0) & (j == n - 1))            # wrap-adjacent pair
    i, j = i[keep], j[keep]
    w = xy[j] - xy[i]
    den = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
    num_s = w[:, 0] * d[j, 1] - w[:, 1] * d[j, 0]
    num_t = w[:, 0] * d[i, 1] - w[:, 1] * d[i, 0]
    safe = np.abs(den) > 1e-300
    s = np.where(safe, num_s / np.where(safe, den, 1.0), -1.0)
    t = np.where(safe, num_t / np.where(safe, den, 1.0), -1.0)
    hit = (s > 0.0) & (s < 1.0) & (t > 0.0) & (t < 1.0)
    ii, jj = i[hit], j[hit]
    ss, tt = s[hit], t[hit]
    # genericity: transversal, away from vertices, depths separated
    li = np.linalg.norm(d[ii], axis=1)
    lj = np.linalg.norm(d[jj], axis=1)
    if np.any(np.abs(den[hit]) < 1e-9 * li * lj):
        return None
    margin = 1e-6
    if np.any((ss < margin) | (ss > 1 - margin)
              | (tt < margin) | (tt > 1 - margin)):
        return None
    z1 = z[ii] + ss * dz[ii]
    z2 = z[jj] + tt * dz[jj]
    if np.any(np.abs(z1 - z2) < 1e-9 * scale):
        return None
    sgn_det = den[hit]
    events = []
    for k in range(len(ii)):
        p1 = float(ii[k] + ss[k])
        p2 = float(jj[k] + tt[k])
        if z1[k] > z2[k]:
            over_p, under_p = p1, p2
            # sign: orientation of (t_over, t_under) in the plane
            sgn = 1 if sgn_det[k] > 0 else -1
        else:
            over_p, under_p = p2, p1
            sgn = -1 if sgn_det[k] > 0 else 1
        events.append((under_p, over_p, sgn))
    unders = sorted(ev[0] for ev in events)
    for a, b in zip(unders, unders[1:]):
        if b - a < 1e-9:
            return None
    return events


def alexander_from_curve(P, x=10):
    """|Alexander polynomial| at t = x of the knot type of the closed
    3D polyline P, computed from a generic planar projection.

    Underpasses cut the diagram into arcs; each crossing contributes
    the Fox-calculus row of its Wirtinger relation (positive crossing
    b = c a c^-1: over c gets 1 - t, under-in a gets t, under-out b
    gets -1; negative crossing rows are the t-scaled mirror), one
    row and one column are deleted, and the Bareiss determinant gives
    Delta(t) up to +-t^k -- the +-t^k is stripped before evaluating,
    so the value is directly comparable with `alexander_at`.  Only
    single-component curves (knots) are supported."""
    import math as _math

    import numpy as np

    def _rot(ax, ay):
        ca, sa = _math.cos(ax), _math.sin(ax)
        cb, sb = _math.cos(ay), _math.sin(ay)
        Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        return Ry @ Rx

    events = None
    for ax, ay in ((0.0, 0.0), (0.31, 0.17), (0.53, 0.71),
                   (0.13, 0.41), (0.87, 0.23), (1.07, 0.61),
                   (0.29, 0.97), (0.71, 0.37), (1.31, 0.11),
                   (0.47, 1.19), (0.91, 0.83), (1.13, 1.03)):
        events = _project_crossings(P, _rot(ax, ay))
        if events is not None:
            break
    if events is None:
        raise ValueError("no generic projection found for the curve")
    c = len(events)
    if c == 0:
        return 1
    # arcs: sorted underpass parameters cut the curve into c arcs;
    # arc r runs from underpass r to underpass r+1 (cyclically)
    events.sort()
    unders = [ev[0] for ev in events]

    def _arc_of(param):
        import bisect
        m = bisect.bisect_right(unders, param) - 1
        return m % c

    one_t = [1, -1]                              # 1 - t
    t_poly = [0, 1]
    rows = []
    for r, (_up, over_p, sgn) in enumerate(events):
        row = [[0]] * c
        over = _arc_of(over_p)
        a_in = (r - 1) % c
        a_out = r
        if sgn > 0:
            add = ((over, one_t), (a_in, t_poly), (a_out, [-1]))
        else:
            add = ((over, [-1, 1]), (a_in, [1]), (a_out, [0, -1]))
        for col, poly in add:
            row[col] = _ptrim(_padd(row[col], poly))
        rows.append(row)
    minor = [row[:c - 1] for row in rows[:c - 1]]
    det = _ptrim(_pbareiss_det(minor))
    if det == [0]:
        return 0
    while det[0] == 0:
        det = det[1:]
    return abs(sum(coef * x ** k for k, coef in enumerate(det)))


def _selftest():
    ok = True

    # Polynomial arithmetic, as coefficient lists.  _pdivexact is the one
    # that can silently corrupt a result, so it is checked as the inverse
    # of _pmul on a product built here.
    a, b = [1, 2], [1, -1, 3]                 # (1 + 2x), (1 - x + 3x^2)
    prod = _pmul(a, b)
    good = (_ptrim(prod) == [1, 1, 1, 6]
            and _ptrim(_padd(a, _pneg(a))) in ([], [0])
            and _ptrim(_pdivexact(prod, a)) == _ptrim(b)
            and _ptrim(_pdivexact(prod, b)) == _ptrim(a))
    ok &= good
    print(f"alexander: poly algebra (mul/add/neg/divexact) "
          f"{'OK' if good else 'FAIL ' + str(prod)}")

    # Known values, cross-checked against the shipped table.  The trefoil
    # and the figure-eight are the two everyone can verify by hand:
    # Delta(3_1) = t - 1 + 1/t and Delta(4_1) = -t + 3 - 1/t, which at the
    # module's evaluation point are the 91 and 71 the table records.
    from .braid import parse_letters
    from .tables import KNOTS
    ref = dict((n, (w, d)) for n, w, d in KNOTS)
    checks = []
    for name in ('3_1', '4_1', '5_1', '5_2', '6_1', '7_1'):
        word, det = ref[name]
        got = alexander_at(parse_letters(word))
        checks.append((name, got, det))
    bad = [c for c in checks if c[1] != c[2]]
    good = not bad
    ok &= good
    print("alexander: " + " ".join(f"{n}={g}" for n, g, _d in checks)
          + f"  {'OK' if good else 'FAIL ' + str(bad)}")

    # The unknot's Alexander polynomial is 1, however it is presented: a
    # one-letter word closes to a single unknotted circle.
    got = alexander_at(parse_letters('a'))
    good = got == 1
    ok &= good
    print(f"alexander: unknot = {got} (exp 1) {'OK' if good else 'FAIL'}")

    # A knot and its mirror share the Alexander polynomial (it cannot
    # detect chirality) -- swapping every letter's case must not change it.
    w = '3_1'
    word, _d = ref[w]
    mirror = word.swapcase()
    good = alexander_at(parse_letters(word)) == alexander_at(
        parse_letters(mirror))
    ok &= good
    print(f"alexander: mirror invariance on {w} {'OK' if good else 'FAIL'}")

    # The geometric route must agree with the Burau route: the braid
    # closure EMBEDDING of a word, resampled (so crossings land inside
    # segments, not on vertices), read back through a generic projection.
    # Two completely independent computations of the same invariant.
    import numpy as np
    from .braid import braid_closure_points
    from .resample import resample_closed
    geo = []
    for name in ('3_1', '4_1', '5_2'):
        word, det = ref[name]
        P = resample_closed(braid_closure_points(parse_letters(word)), 200)
        geo.append((name, alexander_from_curve(P), det))
    bad = [g for g in geo if g[1] != g[2]]
    good = not bad
    ok &= good
    print("alexander: from_curve " + " ".join(f"{n}={v}" for n, v, _d
                                              in geo)
          + f" {'OK' if good else 'FAIL ' + str(bad)}")

    # A crossing-free curve is the unknot: Delta = 1.
    t2 = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    ring = np.stack([np.cos(t2), np.sin(t2), 0.1 * np.sin(2 * t2)],
                    axis=1)
    got = alexander_from_curve(ring)
    good = got == 1
    ok &= good
    print(f"alexander: from_curve unknot = {got} (exp 1) "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("alexander self-test failed")
