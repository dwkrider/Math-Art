# Hexagonal refinement of a packing complex.
#
# Subdividing every triangle into four by adding edge midpoints, then re-packing,
# drives the discrete packing toward the true conformal structure of the region:
# this is the Rodin-Sullivan theorem, and it is the mechanism by which circle
# packings approximate the Riemann map.  It is also what makes conformal tilings
# computable -- each refinement level is a better conformal picture of the same
# combinatorics.
#
# Cost is x4 circles per level, so the caller must cap the depth.
#
# References:
# - Burt Rodin and Dennis Sullivan, "The convergence of circle packings to the
#   Riemann mapping", Journal of Differential Geometry 26 (1987), pp. 349-360.
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005.

from .complexes import PackingComplex


def hex_refine(K, xy=None):
    """One level of hexagonal refinement.

    Each triangle becomes four; original vertices keep their indices, so a
    caller can carry data forward.  Returns (complex, positions or None)."""
    mid = {}
    nv = K.nv
    new_xy = list(xy) if xy is not None else None

    def midpoint(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in mid:
            mid[key] = len(mid) + nv
            if new_xy is not None:
                ax, ay = xy[a]
                bx, by = xy[b]
                new_xy.append((0.5 * (ax + bx), 0.5 * (ay + by)))
        return mid[key]

    faces = []
    for (a, b, c) in K.faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        faces.append((a, ab, ca))
        faces.append((ab, b, bc))
        faces.append((ca, bc, c))
        faces.append((ab, bc, ca))
    return PackingComplex(nv + len(mid), faces), new_xy


def refine_n(K, xy=None, levels=1):
    for _ in range(max(0, levels)):
        K, xy = hex_refine(K, xy)
    return K, xy


def _selftest():
    from . import complexes

    K, xy = complexes.hex_disc(2)
    f0, v0, b0 = len(K.faces), K.nv, len(K.boundary)

    K1, xy1 = hex_refine(K, xy)
    assert len(K1.faces) == 4 * f0, (len(K1.faces), f0)
    assert len(K1.boundary) == 2 * b0, (len(K1.boundary), b0)
    assert len(xy1) == K1.nv
    ne = len(list(K1.edges()))
    assert K1.nv - ne + len(K1.faces) == 1, "refinement must stay a disc"
    for v in range(v0):
        assert abs(xy1[v][0] - xy[v][0]) < 1e-12, "original vertices must not move"

    # interior vertices of the original stay degree 6; new edge vertices are
    # degree 6 in the interior too
    for v in K1.interior:
        assert K1.degree(v) >= 4

    K2, _ = refine_n(K, xy, 2)
    assert len(K2.faces) == 16 * f0
    K2.validate()

    print("packing.refine: x4 faces per level, disc preserved, original "
          "vertices fixed. RESULT: OK")
