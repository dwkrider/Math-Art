# Conway polyhedron notation: seeds, the operator words, and the parser.
#
# Part of the Math Art polyhedra engine (`math_art/polyhedra/`).  Python
# and stdlib only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operator stays in `conway_operators.py`.
#
# A Conway word is read RIGHT TO LEFT: `dkC` is "take the cube, kis it,
# then dualise".  Seeds are the five Platonic solids plus the prism (P),
# antiprism (A) and pyramid (Y) families, which take a side count.
#
# The operators themselves are NOT here.  They live in `flags.py`, which
# implements all eight as rewrites of the flag complex -- the (vertex,
# edge, face) incidence triples with their three involutions -- rather
# than as eight separate constructions.  This module is only the
# notation: what the letters mean, which derived letters expand to which
# primitive sequence, and the order to apply them in.
#
# References:
# - J. H. Conway and M. J. T. Guy, the notation itself; described in
#   G. W. Hart, "Conway Notation for Polyhedra", 1998,
#   http://www.georgehart.com/virtual-polyhedra/conway_notation.html
# - J. H. Conway, H. Burgiel and C. Goodman-Strauss, "The Symmetries of
#   Things", A K Peters, 2008, chapter 21.
# - A. Rossiter, "Antiprism" (antiprism.com) -- the reference
#   implementation this catalogue was checked against.

import itertools
import math
import re

from . import flags


PHI = (1 + 5 ** 0.5) / 2


def _seed(sym, n):
    if sym == 'T':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    elif sym == 'C':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    elif sym == 'O':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    elif sym == 'I':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _hull_faces(V)
    elif sym == 'D':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        for a in (-1 / PHI, 1 / PHI):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _hull_faces(V)
    elif sym == 'P':          # n-prism
        V = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), z)
             for z in (-1, 1) for i in range(n)]
        F = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        F += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    elif sym == 'A':          # n-antiprism
        V = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n),
              -0.5) for i in range(n)]
        V += [(math.cos(2 * math.pi * (i + 0.5) / n),
               math.sin(2 * math.pi * (i + 0.5) / n), 0.5) for i in range(n)]
        F = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        for i in range(n):
            F.append((i, (i + 1) % n, n + i))
            F.append(((i + 1) % n, n + (i + 1) % n, n + i))
    elif sym == 'Y':          # n-pyramid
        V = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n),
              0.0) for i in range(n)] + [(0.0, 0.0, 1.0)]
        F = [tuple(range(n - 1, -1, -1))]
        F += [(i, (i + 1) % n, n) for i in range(n)]
    else:
        raise ValueError(f"unknown seed {sym!r}")
    return [list(map(float, v)) for v in V], [list(f) for f in F]


from .hull import hull_faces as _hull_faces
import numpy as np


def _centroid(V, f):
    return [sum(V[i][c] for i in f) / len(f) for c in range(3)]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _newell(V, f):
    n = [0.0, 0.0, 0.0]
    for i in range(len(f)):
        p = V[f[i]]
        q = V[f[(i + 1) % len(f)]]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return n


def orient_outward(V, F):
    """Flip faces whose Newell normal points toward the centroid.
    Valid for star-shaped (in particular convex-ish) solids."""
    cen = [sum(v[c] for v in V) / len(V) for c in range(3)]
    for f in F:
        n = _newell(V, f)
        fc = _centroid(V, f)
        d = _sub(fc, cen)
        if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] < 0:
            f.reverse()
    return V, F


DERIVED = {'t': 'dk{n}d', 'j': 'da', 'e': 'aa', 'o': 'dada', 'b': 'dk{n}da',
           'm': 'k{n}da', 's': 'dg', 'n': 'k{n}d', 'z': 'dk{n}'}


PRIMS = set('dakgcrpw')


def parse_conway(text):
    """Parse a Conway string into (ops_right_to_left, seed, seed_n).
    ops is a list of (op_char, n) with n=0 meaning unfiltered."""
    text = text.strip().replace(' ', '')
    m = re.fullmatch(r'([a-z0-9]*)([TCODIPAY])(\d*)', text)
    if not m:
        raise ValueError(
            "cannot parse: expected operators then a seed, e.g. 'dkC', "
            "'taD', 'k3sT', 'eP5'")
    opstr, seed, seed_n = m.group(1), m.group(2), m.group(3)
    n = int(seed_n) if seed_n else 0
    if seed in 'PAY' and n < 3:
        raise ValueError(f"seed {seed} needs a number >= 3, e.g. {seed}5")
    toks = re.findall(r'([a-z])(\d*)', opstr)
    # expand derived operators (right-to-left application order)
    prim = []
    for ch, dig in toks:
        nn = int(dig) if dig else 0
        if ch in PRIMS:
            if nn and ch != 'k':
                raise ValueError(f"only k (and t) take a number, not {ch}")
            prim.append((ch, nn))
        elif ch in DERIVED:
            sub = DERIVED[ch].format(n='')
            subtoks = re.findall(r'([a-z])', sub)
            for sch in subtoks:
                prim.append((sch, nn if sch == 'k' else 0))
        else:
            raise ValueError(f"unknown operator {ch!r}")
    return prim, seed, n


def apply_conway(text, kis_height=0.25, chamfer_t=0.35):
    """Build the polyhedron named by a Conway word.

    The operators come from `flags`, which derives all eight from one
    mechanism -- rewriting the flag complex -- instead of eight bespoke
    constructions.  This module contributes only the dispatch: which
    letter means which rewrite, and the right-to-left reading order.

    Note `kis` is called by KEYWORD.  The bespoke version took
    (V, F, only_n, height) and the flag version takes
    (V, F, height, only_n): the same two numbers in the opposite order,
    which positional arguments would silently swap.
    """
    prim, seed, n = parse_conway(text)
    V, F = _seed(seed, n)
    V, F = orient_outward(V, F)
    for ch, nn in reversed(prim):
        if ch == 'd':
            V, F = flags.dual(V, F)
        elif ch == 'a':
            V, F = flags.ambo(V, F)
        elif ch == 'k':
            V, F = flags.kis(V, F, height=kis_height, only_n=nn)
        elif ch == 'g':
            V, F = flags.gyro(V, F)
        elif ch == 'c':
            V, F = flags.chamfer(V, F, t=chamfer_t)
        elif ch == 'p':
            V, F = flags.propellor(V, F)
        elif ch == 'w':
            V, F = flags.whirl(V, F)
        elif ch == 'r':
            V, F = flags.reflect(V, F)
        V, F = orient_outward(V, F)
    return V, F


CATALOG = [
    # Archimedean-Catalan hulls: convex hull of an Archimedean solid and its
    # Catalan dual == the Conway join of the Archimedean solid.
    ('jtT', 'Hull', "Joined Truncated Tetrahedron"),
    ('jaC', 'Hull', "Joined Cuboctahedron"),
    ('jtO', 'Hull', "Joined Truncated Octahedron"),
    ('jtC', 'Hull', "Joined Truncated Cube"),
    ('jeC', 'Hull', "Joined Rhombicuboctahedron"),
    ('jsC', 'Hull', "Joined Snub Cube (laevo)"),
    ('rjsC', 'Hull', "Joined Snub Cube (dextro)"),
    ('jaD', 'Hull', "Joined Icosidodecahedron"),
    ('jbC', 'Hull', "Joined Truncated Cuboctahedron"),
    ('jtI', 'Hull', "Joined Truncated Icosahedron"),
    ('jtD', 'Hull', "Joined Truncated Dodecahedron"),
    ('jeD', 'Hull', "Joined Rhombicosidodecahedron"),
    ('jsD', 'Hull', "Joined Snub Dodecahedron (laevo)"),
    ('rjsD', 'Hull', "Joined Snub Dodecahedron (dextro)"),
    ('jbD', 'Hull', "Joined Truncated Icosidodecahedron"),
    # Propellor solids (George Hart's propellor operator).
    ('pT', 'Propellor', "Propello Tetrahedron"),
    ('pC', 'Propellor', "Propello Cube"),
    ('pO', 'Propellor', "Propello Octahedron"),
    ('pD', 'Propellor', "Propello Dodecahedron"),
    ('pI', 'Propellor', "Propello Icosahedron"),
    ('ptO', 'Propellor', "Propello Truncated Octahedron"),
    ('pkC', 'Propellor', "Propello Tetrakis Hexahedron"),
    ('psC', 'Propellor', "Propello Snub Cube"),
    ('prgC', 'Propellor', "Propello Pentagonal Icositetrahedron"),
    ('pbC', 'Propellor', "Propello Truncated Cuboctahedron"),
    ('pmC', 'Propellor', "Propello Disdyakis Dodecahedron"),
    ('ptI', 'Propellor', "Propello Truncated Icosahedron"),
    ('pkD', 'Propellor', "Propello Pentakis Dodecahedron"),
    ('pbD', 'Propellor', "Propello Truncated Icosidodecahedron"),
    ('pmD', 'Propellor', "Propello Disdyakis Triacontahedron"),
    # Hexpropellor (whirl) solids: central n-gon + hexagons per face.
    ('wT', 'Propellor', "Hexpropello Tetrahedron"),
    ('wC', 'Propellor', "Hexpropello Cube"),
    ('wO', 'Propellor', "Hexpropello Octahedron"),
    ('wD', 'Propellor', "Hexpropello Dodecahedron"),
    ('wI', 'Propellor', "Hexpropello Icosahedron"),
    # Truncated Archimedean solids (truncate all vertices, then canonicalize).
    ('ttT', 'Truncated Archimedean', "Truncated Truncated Tetrahedron"),
    ('ttO', 'Truncated Archimedean', "Truncated Truncated Octahedron"),
    ('ttC', 'Truncated Archimedean', "Truncated Truncated Cube"),
    ('teC', 'Truncated Archimedean', "Truncated Rhombicuboctahedron"),
    ('tsC', 'Truncated Archimedean', "Truncated Snub Cube"),
    ('tbC', 'Truncated Archimedean', "Truncated Truncated Cuboctahedron"),
    ('ttI', 'Truncated Archimedean', "Truncated Truncated Icosahedron"),
    ('ttD', 'Truncated Archimedean', "Truncated Truncated Dodecahedron"),
    ('teD', 'Truncated Archimedean', "Truncated Rhombicosidodecahedron"),
    ('tsD', 'Truncated Archimedean', "Truncated Snub Dodecahedron"),
    ('tbD', 'Truncated Archimedean', "Truncated Truncated Icosidodecahedron"),
    # Rectified Archimedean solids (ambo, then canonicalize).
    ('atT', 'Rectified Archimedean', "Rectified Truncated Tetrahedron"),
    ('atO', 'Rectified Archimedean', "Rectified Truncated Octahedron"),
    ('atC', 'Rectified Archimedean', "Rectified Truncated Cube"),
    ('aeC', 'Rectified Archimedean', "Rectified Rhombicuboctahedron"),
    ('asC', 'Rectified Archimedean', "Rectified Snub Cube"),
    ('abC', 'Rectified Archimedean', "Rectified Truncated Cuboctahedron"),
    ('atI', 'Rectified Archimedean', "Rectified Truncated Icosahedron"),
    ('atD', 'Rectified Archimedean', "Rectified Truncated Dodecahedron"),
    ('aeD', 'Rectified Archimedean', "Rectified Rhombicosidodecahedron"),
    ('asD', 'Rectified Archimedean', "Rectified Snub Dodecahedron"),
    ('abD', 'Rectified Archimedean', "Rectified Truncated Icosidodecahedron"),
    # Chamfered solids (chamfer, then canonicalize).
    ('cT', 'Chamfered', "Chamfered Tetrahedron"),
    ('cO', 'Chamfered', "Chamfered Octahedron"),
    ('cC', 'Chamfered', "Chamfered Cube"),
    ('cI', 'Chamfered', "Chamfered Icosahedron"),
    ('cD', 'Chamfered', "Chamfered Dodecahedron"),
    ('ctI', 'Chamfered', "Chamfered Truncated Icosahedron"),
    # Dipyramids and trapezohedra (duals of the uniform prisms/antiprisms).
    ('dP3', 'Dipyramid / Trapezohedron', "Triangular Dipyramid"),
    ('dP4', 'Dipyramid / Trapezohedron', "Square Dipyramid (Octahedron)"),
    ('dP5', 'Dipyramid / Trapezohedron', "Pentagonal Dipyramid"),
    ('dP6', 'Dipyramid / Trapezohedron', "Hexagonal Dipyramid"),
    ('dP7', 'Dipyramid / Trapezohedron', "Heptagonal Dipyramid"),
    ('dP8', 'Dipyramid / Trapezohedron', "Octagonal Dipyramid"),
    ('dA3', 'Dipyramid / Trapezohedron', "Trigonal Trapezohedron (Cube)"),
    ('dA4', 'Dipyramid / Trapezohedron', "Tetragonal Trapezohedron"),
    ('dA5', 'Dipyramid / Trapezohedron', "Pentagonal Trapezohedron"),
    ('dA6', 'Dipyramid / Trapezohedron', "Hexagonal Trapezohedron"),
    ('dA7', 'Dipyramid / Trapezohedron', "Heptagonal Trapezohedron"),
    ('dA8', 'Dipyramid / Trapezohedron', "Octagonal Trapezohedron"),
    # Truncated Catalan solids: a Catalan solid with only the vertices of a
    # given degree truncated (Conway tN = dk(N)d truncates degree-N
    # vertices; a bare t truncates them all).
    ('t6dtT', 'Truncated Catalan', "6-Truncated Triakis Tetrahedron"),
    ('t6dtO', 'Truncated Catalan', "6-Truncated Tetrakis Hexahedron"),
    ('t8dtC', 'Truncated Catalan', "8-Truncated Triakis Octahedron"),
    ('tdtO', 'Truncated Catalan', "Truncated Tetrakis Hexahedron"),
    ('t4deC', 'Truncated Catalan', "4-Truncated Deltoidal Icositetrahedron"),
    ('t6dtI', 'Truncated Catalan', "6-Truncated Pentakis Dodecahedron"),
    ('t10dtD', 'Truncated Catalan', "10-Truncated Triakis Icosahedron"),
    ('tdbC', 'Truncated Catalan', "Truncated Disdyakis Dodecahedron"),
    ('tdtI', 'Truncated Catalan', "Truncated Pentakis Dodecahedron"),
    ('t5deD', 'Truncated Catalan', "5-Truncated Deltoidal Hexecontahedron"),
    ('t4t5deD', 'Truncated Catalan',
     "4-5-Truncated Deltoidal Hexecontahedron"),
    ('tdbD', 'Truncated Catalan', "Truncated Disdyakis Triacontahedron"),
]


def _selftest():
    """The notation's contract: what parses, what it expands to, and
    that every word builds a genuine polyhedron."""
    ok = True

    # Euler's formula is the sharpest cheap test available: any operator
    # that drops or double-counts an element breaks V - E + F = 2, and
    # it holds for every seed and every word.
    bad = []
    for seed in ('T', 'C', 'O', 'I', 'D', 'P5', 'A5', 'Y5'):
        for word in ('', 'd', 'a', 'k', 'g', 'p', 'w', 'c', 'r',
                     'ak', 'kd', 'gd', 'dad', 'akd'):
            V, F = apply_conway(word + seed)
            E = len({tuple(sorted((f[i], f[(i + 1) % len(f)])))
                     for f in F for i in range(len(f))})
            if len(V) - E + len(F) != 2:
                bad.append(f"{word}{seed}:{len(V)}-{E}+{len(F)}")
    good = not bad
    ok &= good
    print(f"conway: V - E + F = 2 for 112 words "
          f"{'OK' if good else 'FAIL ' + ','.join(bad[:6])}")

    # Every edge is shared by exactly two faces -- the surface is closed.
    bad = []
    for text in ('kC', 'aD', 'gI', 'pO', 'wT', 'cC', 'akC', 'A6'):
        V, F = apply_conway(text)
        cnt = {}
        for f in F:
            for i in range(len(f)):
                e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
                cnt[e] = cnt.get(e, 0) + 1
        odd = [e for e, c in cnt.items() if c != 2]
        if odd:
            bad.append(f"{text}:{len(odd)}")
    good = not bad
    ok &= good
    print(f"conway: every edge borders exactly two faces "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Classical identities.  These are the notation's own claims, and
    # they are what a wrong operator gets caught by: the ambo of a
    # tetrahedron IS the octahedron, and dual is an involution.
    def shape(text):
        V, F = apply_conway(text)
        A = np.asarray(V, float) if 'np' in dir() else None
        pts = sorted(tuple(round(x, 6) for x in v) for v in V)
        return len(V), len(F), sorted(len(f) for f in F), pts

    pairs = [('aT', 'O'), ('dC', 'O'), ('dO', 'C'), ('dT', 'T'),
             ('dD', 'I'), ('dI', 'D'), ('ddC', 'C'), ('aC', 'aO')]
    bad = [f"{a}!={b}" for a, b in pairs if shape(a)[:3] != shape(b)[:3]]
    good = not bad
    ok &= good
    print(f"conway: classical identities (aT=O, dC=O, ddC=C, aC=aO) "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Right-to-left reading: 'dk' means kis THEN dual, not the reverse.
    # If the order were flipped these two would agree, and they must not.
    good = shape('dkC')[:3] != shape('kdC')[:3]
    ok &= good
    print(f"conway: words read right to left (dkC != kdC) "
          f"{'OK' if good else 'FAIL'}")

    # The parser must reject what it cannot mean.  Note the unknown
    # letter is 'q': 'z' looks unknown but is the real derived operator
    # zip (= dk), and using it here would have made this check pass for
    # the wrong reason.
    rejected = 0
    for bad_text in ('', 'C5x', 'qT', 'k', 'P', 'P2', 'a3C'):
        try:
            parse_conway(bad_text)
        except ValueError:
            rejected += 1
    good = rejected == 7
    ok &= good
    print(f"conway: the parser rejects {rejected}/7 malformed words "
          f"{'OK' if good else 'FAIL'}")

    # Derived letters must expand to their primitive sequence.
    good = all(all(c in PRIMS for c in v.format(n=''))
               for v in DERIVED.values())
    ok &= good
    print(f"conway: all {len(DERIVED)} derived operators expand to "
          f"primitives {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("conway self-test failed")
