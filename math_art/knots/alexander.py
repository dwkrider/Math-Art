# Alexander polynomial via the reduced Burau representation.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# The Alexander polynomial evaluated through the reduced Burau
# representation of the braid group.
#
# References:
#   J. W. Alexander, "Topological invariants of knots and links",
#       Trans. AMS 30 (1928).
#   W. Burau, "Ueber Zopfgruppen und gleichsinnig verdrillte
#       Verkettungen", Abh. Math. Semin. Univ. Hambg. 11 (1935).
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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("alexander self-test failed")
