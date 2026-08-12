
# Curves read off a combinatorial word, and spirolaterals.
#
# The idea in both cases is the same and very old: take an integer
# sequence, read each term as an instruction to the turtle, and see what
# it draws.  What makes it worth a generator rather than a novelty is
# that the resulting figure is a faithful PICTURE of the sequence's
# structure -- the Fibonacci word draws a quasiperiodic curve because it
# is quasiperiodic, Thue-Morse tends to the Koch curve because of its
# substitution structure, and a spirolateral closes exactly when the
# arithmetic says it must.
#
# SPIROLATERALS.  Draw segments of length 1, 2, ... n, turning by a fixed
# angle after each, and repeat.  Odds introduced them in 1973; Krawczyk
# took up the 2**n enumeration in which any subset of the turns is
# REVERSED.  That enumeration is the reason `closure.dedupe_by_turning`
# exists: most of the 2**n variants are the same figure up to
# deformation, and Whitney-Graustein says exactly which.
#
# References:
# - Frank Odds, "Spirolaterals", Mathematics Teacher 66(2), 1973,
#   pp. 121-124.
# - Robert Krawczyk, "Spirolaterals, Complexity from Simplicity", in
#   Bridges 2000, and "The Art of Spirolaterals", Bridges 1999.
# - Alexis Monnerot-Dumaine, "The Fibonacci word fractal", HAL
#   hal-00367972, 2009.
# - Jun Ma and Judy Holdener, "When Thue-Morse meets Koch", Fractals
#   13(3), 2005, pp. 191-206.
# - Marston Morse, "Recurrent geodesics on a surface of negative
#   curvature", Trans. AMS 22, 1921 (the Thue-Morse sequence).
# - William Kolakoski, problem 5304, American Mathematical Monthly 72,
#   1965.
# - Harold Abelson and Andrea diSessa, "Turtle Geometry", MIT Press,
#   1981, ch. 1-2 (POLY) and ch. 4 (closure).

import numpy as np

from .closure import (blocks_to_close, closes_geometrically,
                      dedupe_by_turning, minimum_turn, spirolateral_closes,
                      turning_number)


# ---------------------------------------------------------------------
# The sequences
# ---------------------------------------------------------------------

def fibonacci_word(n):
    """The infinite Fibonacci word, as 0/1, truncated to length >= n.

    S(1) = "1", S(2) = "0", S(k) = S(k-1) + S(k-2).  Equivalently the
    fixed point of 0 -> 01, 1 -> 0.  Its letter frequencies are in the
    golden ratio, which is what makes the drawn curve quasiperiodic
    rather than merely irregular.
    """
    a, b = "1", "0"
    while len(b) < n:
        a, b = b, b + a
    return np.array([int(c) for c in b[:n]], dtype=int)


def thue_morse(n):
    """Thue-Morse: the parity of the number of 1 bits in the index."""
    i = np.arange(max(int(n), 0), dtype=np.int64)
    out = np.zeros(len(i), dtype=int)
    x = i.copy()
    while x.any():
        out ^= (x & 1).astype(int)
        x >>= 1
    return out


def sturmian(n, slope=None, intercept=None):
    """Sturmian word of a given slope: floor((k+1)a+b) - floor(ka+b).

    An irrational slope gives an aperiodic word with exactly n+1 factors
    of each length -- the least complex aperiodic sequences there are.

    `intercept` defaults to the slope, which is the CHARACTERISTIC
    Sturmian word of that slope.  With the default slope 1/phi**2 that
    is exactly the Fibonacci word; note it is 1/phi**2 and not 1/phi,
    and that the characteristic convention (b = a) matters -- b = 0
    gives the same word shifted by one place and complemented.
    """
    if slope is None:
        slope = (3.0 - np.sqrt(5.0)) / 2.0          # 1/phi**2
    if intercept is None:
        intercept = slope
    k = np.arange(max(int(n), 0))
    return (np.floor((k + 1) * slope + intercept)
            - np.floor(k * slope + intercept)).astype(int)


def kolakoski(n):
    """The Kolakoski word over {1,2}: its own run-length encoding.

    Returned as the 1/2 run lengths themselves (not 0/1), because that
    IS the sequence; map them to turns with `word_to_turns`.
    """
    n = max(int(n), 0)
    if n == 0:
        return np.zeros(0, dtype=int)
    out = [1, 2, 2]
    i = 2
    while len(out) < n:
        run = out[i]
        nxt = 1 if out[-1] == 2 else 2
        out.extend([nxt] * run)
        i += 1
    return np.array(out[:n], dtype=int)


def ruler(n):
    """The ruler sequence: the 2-adic valuation of each index.

    Term k is the exponent of the largest power of two dividing k.  Drawn
    as turn magnitudes it produces the self-similar tick pattern of a
    ruler, and it is the sequence underlying the dragon's folds.
    """
    k = np.arange(1, max(int(n), 0) + 1, dtype=np.int64)
    v = np.zeros(len(k), dtype=int)
    x = k.copy()
    while True:
        m = (x % 2) == 0
        if not m.any():
            break
        v += m.astype(int)
        x = np.where(m, x // 2, x)
    return v


def abacaba(n):
    """The ABACABA (Zimmermann) sequence -- the ruler sequence's letters.

    Level k inserts a new symbol between two copies of the previous
    level: A, ABA, ABACABA, ...  Returned as 0-based symbol indices.
    """
    out = []
    k = 0
    while len(out) < n:
        out = out + [k] + out
        k += 1
    return np.array(out[:n], dtype=int)


def digit_sum(n, base=10, mod=2):
    """Sum of the base-`base` digits of each index, reduced mod `mod`."""
    i = np.arange(max(int(n), 0), dtype=np.int64)
    s = np.zeros(len(i), dtype=np.int64)
    x = i.copy()
    while x.any():
        s += x % base
        x //= base
    return (s % mod).astype(int)


def prime_residue(n, mod=4):
    """Residue class of the k-th prime, mod `mod`.

    Drawn as turns this is the "prime walk"; the residues of primes mod 4
    are Chebyshev's bias made visible.
    """
    n = max(int(n), 0)
    if n == 0:
        return np.zeros(0, dtype=int)
    limit = max(16, int(n * (np.log(n + 2) + np.log(np.log(n + 3)) + 2)))
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    primes = np.flatnonzero(sieve)[:n]
    return (primes % mod).astype(int)


#: Every sequence the WORD mode offers, by id.
WORDS = {
    "FIBONACCI":  fibonacci_word,
    "THUE_MORSE": thue_morse,
    "STURMIAN":   sturmian,
    "KOLAKOSKI":  kolakoski,
    "RULER":      ruler,
    "ABACABA":    abacaba,
    "DIGIT_SUM":  digit_sum,
    "PRIME":      prime_residue,
}


#: Angle and turn rule that give each sequence its intended figure.
#: The Fibonacci entry is Monnerot-Dumaine's Fibonacci word fractal; the
#: rest are the settings under which the sequence draws a genuinely
#: two-dimensional figure rather than a zigzag along one axis, which is
#: what most angle/rule combinations degenerate into.
WORD_DEFAULTS = {
    "FIBONACCI":  (90.0, "ALTERNATE"),
    "STURMIAN":   (90.0, "ALTERNATE"),
    "THUE_MORSE": (90.0, "ALTERNATE"),
    "KOLAKOSKI":  (90.0, "ALTERNATE"),
    "RULER":      (90.0, "SCALED"),
    "ABACABA":    (90.0, "SCALED"),
    "DIGIT_SUM":  (90.0, "ALTERNATE"),
    "PRIME":      (90.0, "ALTERNATE"),
}


def word_to_turns(seq, angle=90.0, mode="ALTERNATE"):
    """Map an integer sequence to a turn per step.

    `SIGNED`     the sequence's SMALLEST value turns left, every other
                 value turns right.  Splitting on the minimum rather than
                 on zero matters: Kolakoski is a word over {1,2} and the
                 prime residues mod 4 are mostly {1,3}, so a zero test
                 would never turn left and the curve would spiral into a
                 tiny closed loop.
    `GATED`      turn only on a term above the minimum, else go straight.
    `ALTERNATE`  the Fibonacci-word rule: turn only on a zero term, and
                 alternate the direction by the term's POSITION parity.
                 This is the rule that makes the Fibonacci word draw
                 Monnerot-Dumaine's fractal rather than a spiral.
    `SCALED`     turn proportional to the term -- for ruler / ABACABA,
                 where the term is a magnitude rather than a choice.
    """
    s = np.asarray(seq, dtype=int)
    n = len(s)
    if n == 0:
        return np.zeros(0)
    lo = int(s.min())
    if mode == "SIGNED":
        return np.where(s == lo, angle, -angle).astype(float)
    if mode == "GATED":
        return np.where(s != lo, angle, 0.0).astype(float)
    if mode == "SCALED":
        m = float(s.max()) or 1.0
        return (s / m) * angle
    even = (np.arange(n) % 2) == 0
    return np.where(s == lo, np.where(even, angle, -angle), 0.0).astype(float)


def turns_to_points(turns, step=1.0, start_heading=0.0):
    """Walk a turn sequence: turn, then step, repeatedly.

    `n` turns give `n` segments and so `n + 1` points -- the turn is
    applied BEFORE each step, not between steps.
    """
    t = np.asarray(turns, dtype=float)
    if len(t) == 0:
        return np.zeros((1, 2))
    r = np.radians(start_heading + np.cumsum(t))
    d = np.stack([np.cos(r), np.sin(r)], axis=1) * step
    return np.vstack([np.zeros(2), np.cumsum(d, axis=0)])


def word_curve(name, n, angle=90.0, mode="ALTERNATE", **kw):
    """A named combinatorial word, drawn."""
    if name not in WORDS:
        raise KeyError(f"unknown word {name!r}")
    seq = WORDS[name](n, **kw) if kw else WORDS[name](n)
    return turns_to_points(word_to_turns(seq, angle, mode))


# ---------------------------------------------------------------------
# Spirolaterals
# ---------------------------------------------------------------------

def spirolateral(n, angle=90.0, reversals=(), blocks=None):
    """Segments 1..n with a fixed turn, repeated until the figure closes.

    `reversals` is the set of 1-based step indices whose turn is negated
    -- Krawczyk's enumeration.  `blocks` overrides the computed repeat
    count; by default it comes from the closure theorem, which for a turn
    of 360/q is `q / gcd(n, q)` blocks.
    """
    n = max(int(n), 1)
    q = 360.0 / float(angle) if angle else 0.0
    if blocks is None:
        qi = int(round(q))
        blocks = (blocks_to_close(n, qi)
                  if abs(q - qi) < 1e-9 and qi > 0 else 1)
    rev = {int(i) for i in reversals}
    turns, lengths = [], []
    for _b in range(int(blocks)):
        for i in range(1, n + 1):
            turns.append(-angle if i in rev else angle)
            lengths.append(float(i))

    head, pos, pts = 0.0, np.zeros(2), [np.zeros(2)]
    for t, L in zip(turns, lengths):
        r = np.radians(head)
        pos = pos + L * np.array([np.cos(r), np.sin(r)])
        pts.append(pos.copy())
        head += t
    return np.array(pts)


def spirolateral_family(n, angle=90.0, dedupe=True, limit=None):
    """Every reversal variant of an order-n spirolateral.

    There are 2**n subsets of the turns to reverse, most of which give
    the same figure up to deformation.  With `dedupe`, one representative
    per Whitney-Graustein class is returned -- that theorem is the only
    principled way to cut the enumeration down, and it takes 2**10 = 1024
    order-10 variants to a couple of dozen distinct figures.

    Returns (figures, reversal_sets).
    """
    n = max(int(n), 1)
    total = 1 << n
    if limit is not None:
        total = min(total, int(limit))
    sets = [tuple(i + 1 for i in range(n) if m & (1 << i))
            for m in range(total)]
    figs = [spirolateral(n, angle, s) for s in sets]
    if not dedupe:
        return figs, sets
    from .closure import turns_of
    keep, seen = [], set()
    for f, s in zip(figs, sets):
        k = round(turning_number(turns_of(np.hstack(
            [f, np.zeros((len(f), 1))]))), 6)
        closed = closes_geometrically(f)
        key = (k, closed)
        if key in seen:
            continue
        seen.add(key)
        keep.append((f, s))
    return [f for f, _s in keep], [s for _f, s in keep]


def _selftest():
    # --- Fibonacci word ----------------------------------------------
    f = fibonacci_word(21)
    assert "".join(map(str, f[:13])) == "0100101001001", f[:13]
    # letter counts approach the golden ratio
    f = fibonacci_word(2584)
    ratio = float((f == 0).sum()) / float((f == 1).sum())
    assert abs(ratio - (1 + np.sqrt(5)) / 2) < 1e-2, ratio
    # the Sturmian word of slope 1/phi IS the Fibonacci word
    assert np.array_equal(sturmian(200), 1 - fibonacci_word(200)) or \
        np.array_equal(sturmian(200), fibonacci_word(200)), \
        "slope 1/phi should reproduce the Fibonacci word"

    # --- Thue-Morse ---------------------------------------------------
    assert list(thue_morse(16)) == [0, 1, 1, 0, 1, 0, 0, 1,
                                    1, 0, 0, 1, 0, 1, 1, 0]
    # its defining substitution: t(2n) = t(n), t(2n+1) = 1 - t(n)
    tm = thue_morse(512)
    assert np.array_equal(tm[0::2], tm[:256])
    assert np.array_equal(tm[1::2], 1 - tm[:256])

    # --- Kolakoski ----------------------------------------------------
    k = kolakoski(20)
    assert list(k[:12]) == [1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2], list(k[:12])
    # it must equal its own run-length encoding -- the defining property
    runs, cur, cnt = [], k[0], 0
    for v in k:
        if v == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur, cnt = v, 1
    assert list(k[:len(runs)]) == runs, (list(k[:len(runs)]), runs)

    # --- ruler / ABACABA ----------------------------------------------
    assert list(ruler(8)) == [0, 1, 0, 2, 0, 1, 0, 3]
    assert list(abacaba(7)) == [0, 1, 0, 2, 0, 1, 0]
    # they are the same sequence, one as exponents and one as letters
    assert list(ruler(15)) == list(abacaba(15))

    # --- digit sum / primes -------------------------------------------
    # base 2 digit sum mod 2 IS Thue-Morse
    assert np.array_equal(digit_sum(64, base=2, mod=2), thue_morse(64))
    assert list(digit_sum(12, base=10, mod=10)) == \
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2]
    # primes mod 4 are all 1 or 3 after the first two
    p = prime_residue(50, 4)
    assert set(p[2:].tolist()) <= {1, 3}, set(p.tolist())
    assert list(p[:3]) == [2, 3, 1]

    # --- turn mappings -------------------------------------------------
    seq = np.array([0, 1, 0, 0, 1])
    assert list(word_to_turns(seq, 90.0, "SIGNED")) == [90, -90, 90, 90, -90]
    assert list(word_to_turns(seq, 90.0, "GATED")) == [0, 90, 0, 0, 90]
    # ALTERNATE turns only on a zero, alternating by POSITION parity
    assert list(word_to_turns(seq, 90.0, "ALTERNATE")) == [90, 0, 90, -90, 0]
    assert word_to_turns(np.zeros(0)).shape == (0,)

    # a drawn word must be finite and the right length
    P = word_curve("FIBONACCI", 200, 90.0)
    assert len(P) == 201 and np.all(np.isfinite(P))

    # --- spirolaterals -------------------------------------------------
    # The closure arithmetic is the whole point.  Block count is
    # q/gcd(n,q); whether the figure actually CLOSES is a separate
    # question, and the answer is "unless q divides n".
    for n, want in ((1, 4), (2, 2), (3, 4), (4, 1), (5, 4), (6, 2)):
        assert blocks_to_close(n, 4) == want, n
        P = spirolateral(n, 90.0)
        assert len(P) - 1 == n * want, (n, len(P))
        assert closes_geometrically(P) == spirolateral_closes(n, 4), n
    # order 4 at 90 is THE spiral: heading resets every block, so the
    # blocks translate rather than cancel
    assert not closes_geometrically(spirolateral(4, 90.0))
    assert closes_geometrically(spirolateral(3, 90.0))

    # ... and the same rule at 60 degrees (q=6)
    for n in range(1, 9):
        P = spirolateral(n, 60.0)
        assert closes_geometrically(P) == spirolateral_closes(n, 6), n

    # A reversal generally breaks closure -- which is exactly why the
    # enumeration is interesting rather than uniform.
    opened = [s for s in range(1, 5)
              if not closes_geometrically(spirolateral(4, 90.0, (s,)))]
    assert opened, "some reversals should open the figure"

    # Whitney-Graustein dedupe must actually reduce the enumeration
    figs, sets = spirolateral_family(6, 90.0)
    assert len(figs) < (1 << 6), (len(figs),)
    assert len(figs) == len(sets)
    raw, _ = spirolateral_family(6, 90.0, dedupe=False)
    assert len(raw) == 64
    assert len(figs) < len(raw)

    print(f"words: OK -- Fibonacci/Sturmian agree and tend to phi, "
          f"Thue-Morse satisfies its substitution and equals base-2 "
          f"digit sum, Kolakoski is its own run-lengths, ruler==ABACABA, "
          f"spirolaterals close per q/gcd(n,q), 64 reversals -> "
          f"{len(figs)} classes")
