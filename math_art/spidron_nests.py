# The 34 spidron nests, constructed from their boundary specification.
#
# A NEST is a spidronised polygon.  Van Ballegooijen, Gailiunas and
# Erdely's "Spidronised Space-fillers" (Bridges 2009) enumerate the 34
# that occur as faces of Pearce's saddle polyhedra, and identify each by
# a code (n3a ... n12d) giving its polygon size, its group, and the
# sequence of angles between adjacent edges.
#
# NEST IS THE PRECISE TERM, ROSETTE IS A SPECIAL CASE.  Only 7 of the 34
# are FLAT; the other 27 are skew -- which is exactly why they bound
# saddle polyhedra rather than lying in a plane.  A planar arrangement
# of spidron arms about a centre (what this add-on has called a rosette)
# is the flat case; the general object is the nest.
#
# HOW EACH ONE IS BUILT.  Every angle in the table is a Universal Node
# angle, so a nest's boundary is a closed circuit of branch vectors
# whose successive included angles are the tabulated sequence.  That is
# a small search, and it needs no coordinates from the paper -- only the
# angle sequence, which the paper prints.
#
# THE ANGLE SEQUENCE ALONE IS NOT ENOUGH, and this is the trap: a planar
# and a skew polygon can share one angle sequence.  The equiangular
# decagon n10a has both realisations, and the planar one is NOT the
# decatrihedron's face.  The paper's Group column is the discriminator:
# `flat` means planar, anything else means skew, and `regular`
# additionally means equilateral.  Constructing without that constraint
# silently returns the wrong polygon for several codes.
#
# References:
# - Walt van Ballegooijen, Paul Gailiunas & Daniel Erdely, "Spidronised
#   Space-fillers", Bridges 2009 Conference Proceedings, pp. 271-278 --
#   Table 1 and Figure 1, the 34 nests, their groups and G-angles.
# - Daniel Erdely, "Some Surprising New Properties of the Spidrons",
#   Bridges 2005 Conference Proceedings, pp. 179-186 -- the spidron and
#   the hexagonal nest the family generalises.
# - Peter Pearce, "Structure in Nature is a Strategy for Design", The
#   MIT Press, 1978, ch. 8 -- the saddle polyhedra whose faces these are.

import numpy as np

try:
    from . import pearce_net as pnet
except Exception:                       # legacy single-file / CLI use
    import pearce_net as pnet


#: code -> (polygon size, group, symmetry, angle sequence in degrees).
#: Transcribed from Table 1; every angle is a Universal Node angle.
NESTS = {
    'n3a': (3, 'flat', '3-fold', (60, 60, 60)),
    'n3b': (3, 'flat', 'none', (90, 54.7356, 35.2644)),
    'n3c': (3, 'flat', 'mirror', (54.7356, 70.5288, 54.7356)),
    'n3d': (3, 'flat', 'mirror', (45, 90, 45)),
    'n4a': (4, 'flat', '4-fold', (90, 90, 90, 90)),
    'n4b': (4, 'regular', '2-fold', (70.5288,) * 4),
    'n4c': (4, 'regular', '2-fold', (60, 60, 60, 60)),
    'n4d': (4, 'enantio', '2-fold', (45, 90, 45, 90)),
    'n4e': (4, 'mirror', '2-fold', (60, 90, 60, 90)),
    'n4f': (4, 'enantio', '2-fold', (54.7356,) * 4),
    'n4g': (4, 'mirror', 'mirror', (60, 90, 90, 90)),
    'n4h': (4, 'mirror', 'mirror', (109.4712, 54.7356, 90, 54.7356)),
    'n4i': (4, 'enantio', 'none', (90, 54.7356, 54.7356, 90)),
    'n4j': (4, 'enantio', 'none', (90, 45, 54.7356, 54.7356)),
    'n5a': (5, 'flat', 'mirror', (90, 90, 180, 90, 90)),
    'n5b': (5, 'mirror', 'mirror', (90,) * 5),
    'n6a': (6, 'flat', '6-fold', (120,) * 6),
    'n6b': (6, 'regular', '3-fold', (109.4712,) * 6),
    'n6c': (6, 'regular', '3-fold', (60,) * 6),
    'n6d': (6, 'regular', '3-fold', (90,) * 6),
    'n6e': (6, 'mirror', '2-fold', (90, 120, 120, 90, 120, 120)),
    'n6f': (6, 'mirror', '2-fold', (109.4712,) * 6),
    'n6g': (6, 'mirror', 'mirror', (90,) * 6),
    'n6h': (6, 'mirror', 'mirror', (90, 90, 120, 90, 90, 120)),
    'n6i': (6, 'mirror', 'mirror',
            (70.5288, 70.5288, 109.4712, 70.5288, 70.5288, 109.4712)),
    'n8a': (8, 'mirror', '4-fold', (60, 90) * 4),
    'n8b': (8, 'mirror', '4-fold', (70.5288, 109.4712) * 4),
    'n8c': (8, 'mirror', '2-fold', (120,) * 8),
    'n8d': (8, 'mirror', '2-fold',
            (90, 144.7356, 109.4712, 144.7356) * 2),
    'n10a': (10, 'enantio', '2-fold', (120,) * 10),
    'n12a': (12, 'mirror', '4-fold', (90, 120, 120) * 4),
    'n12b': (12, 'mirror', '4-fold',
             (70.5288, 144.7356, 144.7356) * 4),
    'n12c': (12, 'mirror', '3-fold', (120,) * 12),
    'n12d': (12, 'mirror', '3-fold', (144.7356,) * 12),
}

CODES = tuple(sorted(NESTS, key=lambda c: (int(c[1:-1]), c)))

_CACHE = {}


def _is_planar(pts, tol=1e-9):
    P = np.asarray(pts, float)
    return float(np.linalg.svd(P - P.mean(axis=0),
                               compute_uv=False)[2]) < tol


def _is_equilateral(pts, tol=1e-9):
    P = np.asarray(pts, float)
    m = len(P)
    L = [float(np.linalg.norm(P[(i + 1) % m] - P[i])) for i in range(m)]
    return max(L) - min(L) < tol


def _corner(u, v):
    """Interior angle between an incoming edge u and outgoing edge v."""
    return pnet.included_angle(tuple(-x for x in u), v)


def _points(edges):
    pts = [(0, 0, 0)]
    for e in edges[:-1]:
        pts.append(tuple(pts[-1][i] + e[i] for i in range(3)))
    return pts


def boundary(code, tol=0.02):
    """The nest's boundary polygon, as integer eighth-coordinates.

    Searched as a closed circuit of Universal Node branches matching the
    tabulated angle sequence, then filtered by the group so the planar
    and skew realisations of one sequence are not confused."""
    if code in _CACHE:
        return _CACHE[code]
    n, group, _sym, angles = NESTS[code]
    # Try SINGLE (class, modulus) sets first, which force every edge to
    # the same length, and only then widen to mixtures.  Without that
    # ordering the search happily returns a nest with unequal edges when
    # an equilateral one exists -- n10a came out skew and equiangular
    # but NOT equilateral, which is not the decatrihedron's decagon.
    kinds = [(c, m) for c in pnet.CLASSES for m in ('FULL', 'HALF')]
    ladders = ([[k] for k in kinds]
               + [[a, b] for i, a in enumerate(kinds) for b in kinds[i + 1:]]
               + [kinds])

    def walk(edges):
        k = len(edges)
        if k == n:
            if [sum(e[i] for e in edges) for i in range(3)] != [0, 0, 0]:
                return None
            if abs(_corner(edges[-1], edges[0]) - angles[0]) > tol:
                return None
            pts = _points(edges)
            flat = _is_planar(pts)
            if (group == 'flat') != flat:
                return None
            if group == 'regular' and not _is_equilateral(pts):
                return None
            return pts
        for v in vecs:
            if k and abs(_corner(edges[-1], v) - angles[k]) > tol:
                continue
            got = walk(edges + [v])
            if got:
                return got
        return None

    for ladder in ladders:
        vecs = pnet.branch_vectors(ladder)
        for v0 in vecs:
            got = walk([v0])
            if got:
                _CACHE[code] = tuple(got)
                return _CACHE[code]
    _CACHE[code] = None
    return None


def info(code):
    n, group, sym, angles = NESTS[code]
    return dict(code=code, n=n, group=group, symmetry=sym,
                angles=tuple(angles), flat=(group == 'flat'))


def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-58s %s %s" % (name, "OK" if cond else "BAD", extra))

    print("spidron_nests: the 34 nests of Bridges 2009")
    chk("34 nests tabulated", len(NESTS) == 34, "%d" % len(NESTS))
    legal = set(pnet.TABULATED)
    bad = set()
    for code, (n, _g, _s, angs) in NESTS.items():
        if len(angs) != n:
            bad.add(code + ":len")
        for a in angs:
            if pnet.angle_label(a) not in legal:
                bad.add("%s:%s" % (code, pnet.angle_label(a)))
    chk("every angle is a Universal Node angle, one per corner",
        not bad, " ".join(sorted(bad)))

    flat = [c for c in CODES if NESTS[c][1] == 'flat']
    chk("7 flat nests, 27 skew", len(flat) == 7,
        "%d flat: %s" % (len(flat), " ".join(flat)))

    built = 0
    for code in CODES:
        pts = boundary(code)
        if pts is None:
            chk("%s: constructed" % code, False)
            continue
        built += 1
        n, group, _sym, angs = NESTS[code]
        if len(pts) != n:
            chk("%s: %d corners" % (code, n), False, str(len(pts)))
            continue
        got = pnet.circuit_angles(pts)
        good = all(abs(g - w) < 0.05 for g, w in zip(got, angs))
        planar = _is_planar(pts)
        chk("%s %2d-gon %-8s planar=%-5s angles match"
            % (code, n, group, planar), good and (planar == (group == 'flat')))
        # every edge must be a branch, and the circuit must close exactly
        if not pnet.closes(pts):
            chk("  %s closes exactly" % code, False)
        for k in range(n):
            v = tuple(pts[(k + 1) % n][i] - pts[k][i] for i in range(3))
            if pnet.branch_class(v) is None:
                chk("  %s edge %d is a branch" % (code, k), False, str(v))
                break
    chk("all 34 constructed", built == 34, "%d" % built)

    # the decatrihedron's face is n10a, and it is NOT the planar decagon
    pts = boundary('n10a')
    chk("n10a is skew (the decatrihedron's decagon, not a flat decagon)",
        pts is not None and not _is_planar(pts))
    chk("n10a is equilateral and equiangular (a true nest)",
        pts is not None and pnet.is_equilateral_equiangular(pts))

    print("RESULT:", "OK" if ok else "BAD")
    if not ok:
        raise AssertionError("spidron_nests self-test failed")
