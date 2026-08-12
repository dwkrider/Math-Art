
# Predicting how big a derivation will get, WITHOUT deriving it.
#
# A runaway L-system is the classic way to hang a modelling session: the
# word length of `F -> FF` doubles every step, so generation 24 is 16.7
# million modules and the UI is gone.  Guessing an iteration cap per
# preset is unsatisfying; for the deterministic context-free
# non-parametric class (D0L) the size is available in closed form, so
# this module computes it instead.
#
# Let k = |alphabet|.  The GROWTH MATRIX M has M[i][j] = the number of
# occurrences of letter i in the successor of letter j, pi is the Parikh
# vector of the axiom (its letter counts) and eta the all-ones column.
# Then the growth function is
#
#     f(n) = eta^T . M^n . pi
#
# which is the module count at generation n, obtained by k-by-k matrix
# powers and never by building a word.  (With M indexed as above, the
# letter-count vector advances as v <- M v, so the sum of M^n pi is the
# word length.  Getting that multiplication the wrong way round is easy
# and silent -- both the doubling system and the Fibonacci system have
# SYMMETRIC growth matrices, so they pass either way.  The (n+1)^2 case
# in `_selftest` is asymmetric precisely to catch it.)
#
# Three results from Rozenberg & Salomaa make this a usable UI feature
# rather than a curiosity:
#
#   * Theorem 3.4 -- because M satisfies its own characteristic
#     equation, f obeys a linear recurrence
#         f(n+k) = c_{k-1} f(n+k-1) + ... + c_0 f(n)
#     with coefficients obtained effectively from the grammar.  After k
#     seed values the count advances in O(k) per generation, so a live
#     read-out in the redo panel costs nothing.
#
#   * Theorem 3.7 -- every D0L growth function is either EXPONENTIAL or
#     POLYNOMIALLY BOUNDED.  That dichotomy is the right thing to tell a
#     user: "this grammar grows polynomially, go as deep as you like"
#     versus "this one is exponential, n = 24 is 16.7 M modules".
#
#   * Exercise 3.14 -- growth is exponential IF AND ONLY IF some power
#     M^p with p <= 2^k + k - 1 has a diagonal element greater than 1.
#     For the small alphabets of every shipped preset this is instant,
#     and it is a proof rather than a heuristic.
#
# SCOPE.  All of the above is D0L-only.  Parametric, stochastic and
# context-sensitive grammars have no such closed form -- for those the
# honest answer is a hard budget plus clamp-and-warn, and `predict`
# returns None so the caller can say so rather than pretend.
#
# References:
# - Grzegorz Rozenberg and Arto Salomaa, "The Mathematical Theory of
#   L Systems", Academic Press, 1980 -- section I.3 (growth matrix,
#   Theorems 3.1-3.8) and Exercise 3.14.
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, section 1.9 (growth functions).

import numpy as np

EXPONENTIAL, POLYNOMIAL, UNKNOWN = "exponential", "polynomial", "unknown"


def growth_matrix(grammar):
    """(M, pi, letters) for a D0L grammar, or (None, None, letters).

    M[i][j] counts occurrences of letters[i] in the successor of
    letters[j].  A letter with no production is its own successor (the
    identity production), which is what keeps terminals from vanishing
    out of the count.
    """
    letters = grammar.alphabet()
    if not grammar.is_dol():
        return None, None, letters
    idx = {s: i for i, s in enumerate(letters)}
    k = len(letters)
    M = np.zeros((k, k), dtype=np.int64)
    succ = {p.strict_sym: [s for s, _ in p.succ] for p in grammar.productions}
    for j, sym in enumerate(letters):
        for s in succ.get(sym, [sym]):
            i = idx.get(s)
            if i is not None:
                M[i, j] += 1
    pi = np.zeros(k, dtype=np.int64)
    for mod in grammar.axiom:
        i = idx.get(mod.sym)
        if i is not None:
            pi[i] += 1
    return M, pi, letters


def counts(grammar, n):
    """Module count at generation `n`, or None when not predictable.

    Uses f(n) = pi . M^n . eta directly; `n` is small in practice and
    integer matrix powers are exact, so there is no drift.
    """
    M, pi, _ = growth_matrix(grammar)
    if M is None:
        return None
    if n <= 0:
        return int(pi.sum())
    v = pi.astype(object)                 # exact: counts get big fast
    Mo = M.astype(object)
    for _ in range(int(n)):
        v = Mo.dot(v)
    return int(v.sum())


def sequence(grammar, upto):
    """[f(0), f(1), ... f(upto)] or None -- for plotting a growth curve
    or for populating the recurrence's seed values."""
    M, pi, _ = growth_matrix(grammar)
    if M is None:
        return None
    out, v = [], pi.astype(object)
    Mo = M.astype(object)
    for _ in range(int(upto) + 1):
        out.append(int(v.sum()))
        v = Mo.dot(v)
    return out


def recurrence(grammar):
    """Coefficients (c_0 ... c_{k-1}) of Theorem 3.4's linear recurrence

        f(n+k) = c_{k-1} f(n+k-1) + ... + c_0 f(n)

    read off the characteristic polynomial of M.  Returns None when the
    grammar is not D0L.

    This is what makes a live read-out cheap: with the coefficients and k
    seed values, each further generation is O(k) additions instead of a
    matrix power.
    """
    M, _pi, _ = growth_matrix(grammar)
    if M is None:
        return None
    k = M.shape[0]
    if k == 0:
        return []
    # numpy gives the characteristic polynomial as
    #   lambda^k + a_{k-1} lambda^{k-1} + ... + a_0
    # and Cayley-Hamilton turns that into the recurrence with
    #   c_i = -a_i.
    coeffs = np.poly(M.astype(float))      # [1, a_{k-1}, ..., a_0]
    return [float(-c) for c in coeffs[1:]][::-1]


def is_exponential(grammar):
    """Decide exponential vs polynomial growth, exactly.

    Rozenberg & Salomaa Exercise 3.14: growth is exponential iff some
    power M^p with p <= 2^k + k - 1 has a diagonal element > 1.  A
    diagonal element of M^p greater than 1 means some letter derives at
    least two copies of itself in p steps, which compounds.

    The stated bound is astronomically loose for real alphabets; the
    practical bound below is min(that, 64), which is unreachable in
    practice because a compounding letter shows up within a couple of
    powers.  Returns None when the grammar is not D0L.
    """
    M, _pi, _ = growth_matrix(grammar)
    if M is None:
        return None
    k = M.shape[0]
    if k == 0:
        return False
    limit = min(2 ** min(k, 20) + k - 1, 64)
    P = np.eye(k, dtype=object)
    Mo = M.astype(object)
    for _ in range(limit):
        P = P.dot(Mo)
        if any(P[i, i] > 1 for i in range(k)):
            return True
        if not P.any():
            return False              # everything died: bounded
    return False


def classify(grammar):
    """EXPONENTIAL / POLYNOMIAL / UNKNOWN -- the string a UI should show.

    Theorem 3.7 guarantees the dichotomy for D0L; UNKNOWN is returned for
    parametric, stochastic or context-sensitive grammars, where no closed
    form exists and the caller should fall back to a hard budget.
    """
    e = is_exponential(grammar)
    if e is None:
        return UNKNOWN
    return EXPONENTIAL if e else POLYNOMIAL


def safe_iterations(grammar, budget, hard_cap=32):
    """Largest n with counts(n) <= budget, or None when unpredictable.

    This is the number the operator clamps to, so a preset that would
    produce 16 million modules quietly stops at the last generation that
    fits and reports the clamp instead of freezing the session.
    """
    M, _pi, _ = growth_matrix(grammar)
    if M is None:
        return None
    n = 0
    while n < hard_cap:
        if counts(grammar, n + 1) > budget:
            return n
        n += 1
    return hard_cap


def describe(grammar, n):
    """One line for the redo panel: predicted size and growth class."""
    c = counts(grammar, n)
    if c is None:
        return "size not predictable (parametric/stochastic/context-sensitive)"
    return f"{c:,} modules at n={n} ({classify(grammar)} growth)"


def _selftest():
    from .core import Grammar

    # --- the classic doubling system ---------------------------------
    g = Grammar.parse("axiom: F\np1: F -> FF")
    assert [counts(g, n) for n in range(5)] == [1, 2, 4, 8, 16]
    assert classify(g) == EXPONENTIAL
    assert counts(g, 24) == 2 ** 24

    # --- Fibonacci: the textbook non-trivial D0L ---------------------
    # a -> b, b -> ab gives word lengths 1,1,2,3,5,8,...
    fib = Grammar.parse("axiom: a\np1: a -> b\np2: b -> ab")
    got = [counts(fib, n) for n in range(9)]
    assert got == [1, 1, 2, 3, 5, 8, 13, 21, 34], got
    assert classify(fib) == EXPONENTIAL      # golden-ratio growth

    # --- a polynomially growing system -------------------------------
    # ABOP-style: a -> abc^2, b -> bc^2, c -> c gives f(n) = (n+1)^2.
    quad = Grammar.parse("axiom: a\np1: a -> abcc\np2: b -> bcc\np3: c -> c")
    got = [counts(quad, n) for n in range(6)]
    assert got == [(n + 1) ** 2 for n in range(6)], got
    assert classify(quad) == POLYNOMIAL, \
        "no letter compounds, so growth must be polynomial"

    # --- a bounded system --------------------------------------------
    flat = Grammar.parse("axiom: ab\np1: a -> a\np2: b -> b")
    assert [counts(flat, n) for n in range(4)] == [2, 2, 2, 2]
    assert classify(flat) == POLYNOMIAL

    # --- prediction agrees with actually deriving ---------------------
    for src in ("axiom: F\np1: F -> F+F-F",
                "axiom: X\np1: X -> F[+X]F[-X]+X\np2: F -> FF",
                "axiom: a\np1: a -> b\np2: b -> ab"):
        gg = Grammar.parse(src)
        for n in range(6):
            assert counts(gg, n) == len(gg.derive(n)), \
                f"prediction != derivation for {src!r} at n={n}"

    # --- Theorem 3.4: the recurrence reproduces the sequence ---------
    seq = sequence(fib, 10)
    cs = recurrence(fib)
    k = len(cs)
    for n in range(len(seq) - k):
        pred = sum(cs[i] * seq[n + i] for i in range(k))
        assert abs(pred - seq[n + k]) < 1e-6, \
            f"recurrence broke at n={n}: {pred} vs {seq[n + k]}"

    # --- clamping ----------------------------------------------------
    n = safe_iterations(g, budget=1000)
    assert counts(g, n) <= 1000 < counts(g, n + 1), n

    # --- non-D0L systems decline to predict, rather than guessing ----
    par = Grammar.parse("axiom: A(1)\np1: A(x) -> A(x+1)")
    sto = Grammar.parse("axiom: A\np1: A -> AA : 0.5\np2: A -> B : 0.5")
    ctx = Grammar.parse("axiom: BA\np1: B < A -> B")
    for bad in (par, sto, ctx):
        assert counts(bad, 3) is None
        assert classify(bad) == UNKNOWN
        assert safe_iterations(bad, 100) is None
        assert "not predictable" in describe(bad, 3)

    print("growth: OK -- growth matrix, exact counts vs derivation, "
          "Fibonacci and (n+1)^2 sequences, Thm 3.4 recurrence, "
          "exponential/polynomial classifier, clamping")
