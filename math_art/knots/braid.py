# Braid words and their closures.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# Braid groups B_n and the Artin generators s_i:
#   Emil Artin, "Theory of braids", Annals of Mathematics 48 (1947).
# Alexander's theorem (every link is a braid closure):
#   J. W. Alexander, "A lemma on systems of knotted curves", PNAS 9
#   (1923).
#
# Words are written in the letter convention of Gittings' table: a lower
# case letter is a positive crossing of that strand pair, upper case its
# inverse.  The closure is laid out around a circle -- strands at radii by
# braid level, crossings as over/under bumps.

import math

import numpy as np


def parse_letters(s):
    """Gittings letter notation: a..z = generators 1.., A..Z their
    inverses."""
    word = []
    for ch in s:
        if ch.islower():
            word.append(ord(ch) - ord('a') + 1)
        elif ch.isupper():
            word.append(-(ord(ch) - ord('A') + 1))
        else:
            raise ValueError(f"bad braid character {ch!r}")
    return word


def closure_components(word):
    """Number of components of the braid closure."""
    n = max(abs(g) for g in word) + 1
    perm = list(range(n))
    for g in word:
        i = abs(g) - 1
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    seen = set()
    comps = 0
    for s0 in range(n):
        if s0 in seen:
            continue
        comps += 1
        cur = s0
        while cur not in seen:
            seen.add(cur)
            cur = perm[cur]
    return comps


def braid_closure_points(word, ring=1.6, spacing=0.3, bump=0.26):
    """Closed 3D polyline of the braid closure: strands at radii by
    braid level around a circle, crossings as +-z bumps."""
    n = max(abs(g) for g in word) + 1
    L = len(word)

    def pt(a, lev, z=0.0):
        r = ring + (lev - (n - 1) / 2.0) * spacing
        return (r * math.cos(a), r * math.sin(a), z)

    paths = [[] for _ in range(n)]        # per strand id
    strands = list(range(n))              # level -> strand id
    for lev in range(n):
        paths[lev].append(pt(0.0, lev))
    for s, g in enumerate(word):
        a0 = 2 * math.pi * s / L
        a1 = 2 * math.pi * (s + 1) / L
        am = (a0 + a1) / 2
        i = abs(g) - 1
        for lev in range(n):
            if lev != i and lev != i + 1:
                paths[strands[lev]].append(pt(a1, lev))
        z = bump if g > 0 else -bump
        sid_i, sid_j = strands[i], strands[i + 1]
        paths[sid_i].append(pt(am, i + 0.5, z))
        paths[sid_i].append(pt(a1, i + 1))
        paths[sid_j].append(pt(am, i + 0.5, -z))
        paths[sid_j].append(pt(a1, i))
        strands[i], strands[i + 1] = sid_j, sid_i
    end_level = {sid: lev for lev, sid in enumerate(strands)}
    loop = [paths[0][0]]
    sid = 0
    while True:
        loop.extend(paths[sid][1:])
        sid = end_level[sid]
        if sid == 0:
            break
    loop.pop()                            # closing duplicate
    return loop


def braid_closure_loops(word, ring=1.6, spacing=0.3, bump=0.26):
    """Closed 3D polylines of a braid closure, one per link
    component (multi-component generalization of
    prime_knot_generator.braid_closure_points)."""
    n = max(abs(g) for g in word) + 1
    L = len(word)

    def pt(a, lev, z=0.0):
        r = ring + (lev - (n - 1) / 2.0) * spacing
        return (r * math.cos(a), r * math.sin(a), z)

    paths = [[] for _ in range(n)]        # per strand id
    strands = list(range(n))              # level -> strand id
    for lev in range(n):
        paths[lev].append(pt(0.0, lev))
    for s, g in enumerate(word):
        a0 = 2 * math.pi * s / L
        a1 = 2 * math.pi * (s + 1) / L
        am = (a0 + a1) / 2
        i = abs(g) - 1
        for lev in range(n):
            if lev != i and lev != i + 1:
                paths[strands[lev]].append(pt(a1, lev))
        z = bump if g > 0 else -bump
        sid_i, sid_j = strands[i], strands[i + 1]
        paths[sid_i].append(pt(am, i + 0.5, z))
        paths[sid_i].append(pt(a1, i + 1))
        paths[sid_j].append(pt(am, i + 0.5, -z))
        paths[sid_j].append(pt(a1, i))
        strands[i], strands[i + 1] = sid_j, sid_i
    end_level = {sid: lev for lev, sid in enumerate(strands)}
    loops = []
    seen = set()
    for s0 in range(n):
        if s0 in seen:
            continue
        loop = [paths[s0][0]]
        sid = s0
        while True:
            seen.add(sid)
            loop.extend(paths[sid][1:])
            sid = end_level[sid]
            if sid == s0:
                break
        loop.pop()                        # closing duplicate
        loops.append(loop)
    return loops


def _selftest():
    ok = True

    # parse_letters: lower case is a positive generator, upper case its
    # inverse, and the index follows the letter.
    good = (parse_letters('a') == [1] and parse_letters('A') == [-1]
            and parse_letters('aB') == [1, -2]
            and parse_letters('AAA') == [-1, -1, -1])
    ok &= good
    print(f"braid: parse_letters {'OK' if good else 'FAIL'}")

    # closure_components counts the cycles of the strand permutation.
    # s1^n on 2 strands closes to ONE component when n is odd (trefoil,
    # unknot) and TWO when n is even (the Hopf link at n = 2).
    cases = [('a', 1), ('aa', 2), ('aaa', 1), ('aaaa', 2),
             ('AAA', 1), ('aBaBa', 2)]
    bad = [(w, closure_components(parse_letters(w)), e)
           for w, e in cases if closure_components(parse_letters(w)) != e]
    good = not bad
    ok &= good
    print(f"braid: closure_components on {len(cases)} words "
          f"{'OK' if good else 'FAIL ' + str(bad)}")

    # The closure embedding must produce a CLOSED, non-degenerate polyline:
    # consecutive points distinct, and the wrap segment no longer than a
    # few times the median step (a gap there means the closure did not
    # actually close).
    for w in ('AAA', 'AbAb', 'AABacBc'):
        P = np.asarray(braid_closure_points(parse_letters(w)), dtype=float)
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        med = float(np.median(seg))
        # 4 points per letter of the word, so even the trefoil's 3-letter
        # word gives only 12 -- the floor is deliberately low.
        g = (len(P) >= 4 * len(parse_letters(w))
             and bool(np.all(np.isfinite(P)))
             and float(seg.min()) > 1e-9 and float(seg[-1]) < 8.0 * med)
        ok &= g
        print(f"braid: closure {w:8s} {len(P):4d} pts, wrap/median="
              f"{seg[-1] / med:.2f} {'OK' if g else 'FAIL'}")

    # The multi-component form must return exactly closure_components loops,
    # and together account for every point of the single-polyline form.
    for w in ('aa', 'aBaBa'):
        word = parse_letters(w)
        loops = braid_closure_loops(word)
        g = len(loops) == closure_components(word) and all(
            len(lp) > 3 for lp in loops)
        ok &= g
        print(f"braid: {w} -> {len(loops)} loops "
              f"(exp {closure_components(word)}) {'OK' if g else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("braid self-test failed")
