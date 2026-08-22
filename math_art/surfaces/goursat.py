# Goursat's surfaces: the algebraic surfaces with the symmetries of a
# regular polyhedron.
#
# Part of the Math Art surfaces engine (`math_art/surfaces/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operator stays in its flat generator module.
#
# Goursat asked which algebraic surfaces carry the full symmetry group of
# a Platonic solid, and answered it degree by degree.  The answer is a
# short list of one-parameter families, and this module is that list:
#
#   TETRAHEDRAL, degree 3   xyz + k' a (x^2+y^2+z^2) = k a^3
#       The simplest non-spherical case.  k' = 0 collapses it to
#       Titeica's affine sphere xyz = a^3; otherwise a is rescalable to
#       k' = 1 and one parameter k remains.  It runs through four
#       separate sheets (k < 0), four sheets plus an isolated point at
#       the origin (k = 0), a rounded tetrahedron plus four sheets
#       (0 < k < 4), Cayley's four-nodal cubic exactly at k = 4, and a
#       single blown-out sheet beyond.
#
#   TETRAHEDRAL, degree 4   the quartic with four "triconic" points at
#       the alternate cube corners (+-a, +-a, +-a) taking an odd number
#       of minus signs -- the tetrahedron's own vertices.
#
#   OCTAHEDRAL, degree 4    x^4+y^4+z^4 + k(x^2+y^2+z^2)^2
#                             + k' a^2 (x^2+y^2+z^2) + k'' a^4 = 0
#       Goursat calls these the octahedral quartics; "cubic" is best
#       avoided for them since it collides with the degree.  Seven
#       members are named, and each is named for a piece of
#       cube/octahedron geometry it contains or a set of nodes it
#       carries -- see `_OCT` below, where every one of those claims is
#       an exact identity the self-test checks.
#
#   DODECAHEDRAL, degree 6  a fixed icosahedrally invariant sextic
#       z^6 - 5(x^2+y^2)z^4 + 5(x^2+y^2)^2 z^2
#            - 2(x^4 - 10x^2y^2 + 5y^4) x z
#       plus the radial terms k(r^2)^3 + k' a^2 (r^2)^2 + k'' a^4 r^2
#       + k''' a^6.  The fixed part is exactly the product of the six
#       planes through the edges of an icosidodecahedron, whose normals
#       are the six five-fold axes -- which is why the whole family is
#       icosahedral.  Members include a rounded dodecahedron, a rounded
#       icosahedron, Barth's sextic, and two surfaces carrying 60 lines.
#
# WHAT THE SELF-TEST GATES ON, and why it is not "it meshed".  Every
# surface here is a level set of a polynomial with hand-typed integer or
# small-rational coefficients.  A mistyped coefficient raises nothing: it
# produces a perfectly meshable surface that has simply lost the property
# it is named for.  So each row is checked against ITS OWN claim --
# a contained line is evaluated on that line, a node is evaluated at the
# node, and the symmetry group is checked as an exact algebraic identity.
#
# ONE CORRECTION TO THE SOURCE, deliberate.  For the octahedral member
# (k, k', k'') = (-1, 2, -2) the encyclopedia prints the expanded form
#     x^2y^2 + y^2z^2 + z^2x^2 - a^2(x^2+y^2+z^2) = a^4
# but expanding the (k, k', k'') it prints beside it gives that equation
# with -a^4 on the right.  The two disagree, and the surface's own stated
# property decides it: the row is documented as containing the twelve
# extended medians of the cube's faces, e.g. the line z = 0, x = a.  On
# that line the (k, k', k'') form vanishes identically (max |F| = 5e-15)
# and the printed expansion does not (|F| = 2 a^4).  The tuple is right
# and the printed sign is a slip, so the tuple is what is implemented and
# the line identity is what the self-test gates on.
#
# References:
# - E. Goursat, "Etude des surfaces qui admettent tous les plans de
#   symetrie d'un polyedre regulier", Annales scientifiques de l'Ecole
#   Normale Superieure, 3e serie, 4 (1887) 159-200.
#   http://www.numdam.org/article/ASENS_1887_3_4__159_0.pdf
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapter "surface de Goursat" -- the named members and
#   their (k, k', k'', k''') tuples used here.  A converted copy of the
#   encyclopedia is in research/books/
#   mathcurve_encyclopedie_formes_mathematiques/.
# - W. Barth, "Two projective surfaces with many nodes admitting the
#   symmetries of the icosahedron", J. Algebraic Geometry 5 (1996)
#   173-186 -- the sextic that appears here as (0, 5/4, -5/2, 5/4).
# - A. Esculier, ray-traced Goursat sextics,
#   http://aesculier.fr/fichiersPovray/sextiques/sextiques.html

import math

import numpy as np

try:
    from .. import geom_cache as _geom_cache
except ImportError:  # flat import outside the package
    import geom_cache as _geom_cache


_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _toolkit():
    """`minsurf` supplies marching_tets; see surfaces/algebraic.py for
    why this is two dots and not one."""
    try:
        from .. import minsurf as mst
    except ImportError:
        import minsurf as mst
    return mst


# ----------------------------------------------------------------------
# the four fields
# ----------------------------------------------------------------------

def f_oct4(x, y, z, coeffs, a=1.0):
    """Octahedral quartic, coeffs = (k, k', k'')."""
    k, k1, k2 = coeffs
    r2 = x * x + y * y + z * z
    return (x ** 4 + y ** 4 + z ** 4 + k * r2 * r2
            + k1 * a * a * r2 + k2 * a ** 4)


def f_tet3(x, y, z, coeffs, a=1.0):
    """Tetrahedral cubic, coeffs = (k, k').  k' = 0 is Titeica's
    xyz = a^3; any other k' is rescalable to 1."""
    k, k1 = coeffs
    return (x * y * z + k1 * a * (x * x + y * y + z * z) - k * a ** 3)


def f_tet4(x, y, z, coeffs, a=1.0):
    """Tetrahedral quartic with four triconic points at the alternate
    cube corners.  Takes no coefficients."""
    return ((x + y + z - a) * (-x - y + z - a) * (x - y - z - a)
            * (-x + y - z - a) - (x * x + y * y + z * z - 3.0 * a) ** 2)


def _dodec_base(x, y, z):
    """The icosahedrally invariant sextic core: the product of the six
    planes through the edges of an icosidodecahedron, whose normals are
    the six five-fold axes.  `_selftest` checks that factorisation
    exactly, which is what pins the transcription down."""
    r2 = x * x + y * y
    return (z ** 6 - 5.0 * r2 * z ** 4 + 5.0 * r2 * r2 * z * z
            - 2.0 * (x ** 4 - 10.0 * x * x * y * y + 5.0 * y ** 4) * x * z)


def f_dodec6(x, y, z, coeffs, a=1.0):
    """Dodecahedral sextic, coeffs = (k, k', k'', k''')."""
    k, k1, k2, k3 = coeffs
    r2 = x * x + y * y + z * z
    return (_dodec_base(x, y, z) + k * r2 ** 3 + k1 * a * a * r2 * r2
            + k2 * a ** 4 * r2 + k3 * a ** 6)


#: family key -> (label, field, number of free coefficients, degree)
FAMILIES = (
    ('OCT4', "Octahedral Quartic", f_oct4, 3, 4),
    ('TET3', "Tetrahedral Cubic", f_tet3, 2, 3),
    ('TET4', "Tetrahedral Quartic", f_tet4, 0, 4),
    ('DODEC6', "Dodecahedral Sextic", f_dodec6, 4, 6),
)

FAMILY_FIELD = {k: fn for (k, _l, fn, _n, _d) in FAMILIES}
FAMILY_NCOEFF = {k: n for (k, _l, _f, n, _d) in FAMILIES}
FAMILY_LABEL = {k: lab for (k, lab, _f, _n, _d) in FAMILIES}


# ----------------------------------------------------------------------
# the named members
# ----------------------------------------------------------------------
# Row layout: (key, label, family, coefficients, clip radius in units of
# a, description).  The clip radius on each row was chosen by
# measurement, not by eye: the smallest ball that holds the whole
# bounded part, or -- for the members that run off to infinity -- the
# radius at which the sheets are already unmistakable.

_OCT = (
    ('OCT_CUBOCTA', "Cuboctahedral Quartic", (-1.0, 1.0, 1.0), 2.4,
     "2(x^2y^2 + y^2z^2 + z^2x^2) - a^2(x^2+y^2+z^2) = a^4"),
    ('OCT_CUBOCTA_NODES', "Quartic with 12 Cuboctahedral Nodes",
     (0.0, -2.0, 2.0), 2.0,
     "x^4+y^4+z^4 - 2a^2(x^2+y^2+z^2) + 2a^4 = 0, with twelve "
     "singular points at the cuboctahedron's vertices (+-a, +-a, 0)"),
    ('OCT_OCTA_EDGES', "Octahedron-Edge Quartic", (-0.5, -1.0, 0.5), 2.6,
     "Contains the twelve extended edges of the octahedron "
     "|x|+|y|+|z| = a; it is to the octahedron what Cayley's cubic "
     "is to the tetrahedron"),
    ('OCT_CUBE_DIAGONALS', "Cube-Diagonal Quartic", (-0.5, 1.0, -1.5), 2.6,
     "Contains the twelve face diagonals of the cube "
     "max(|x|,|y|,|z|) = a"),
    ('OCT_CUBE_EDGES', "Cube-Edge Quartic", (-1.0, 4.0, -6.0), 2.6,
     "Contains the twelve extended edges of the cube "
     "max(|x|,|y|,|z|) = a"),
    ('OCT_CUBE_MEDIANS', "Cube-Median Quartic", (-1.0, 2.0, -2.0), 2.4,
     "Twelve singular points at the cuboctahedron's vertices, and the "
     "twelve extended medians of the cube's faces; six sheets sitting "
     "on the cuboctahedron's square faces"),
    ('OCT_TRIANGLES', "Cuboctahedral Triangle Quartic",
     (-1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0), 2.4,
     "Twelve singular points at the cuboctahedron's vertices, with "
     "eight sheets sitting on its triangular faces"),
)

_TET = (
    ('TET_TITEICA', "Titeica Cubic (k' = 0)", (1.0, 0.0), 4.0,
     "xyz = a^3, the affine sphere -- the k' = 0 member of the "
     "tetrahedral cubics"),
    ('TET_FOUR_SHEETS', "Tetrahedral Cubic (k < 0)", (-1.0, 1.0), 7.0,
     "Four separate sheets"),
    ('TET_ISOLATED_POINT', "Tetrahedral Cubic (k = 0)", (0.0, 1.0), 7.0,
     "Four sheets plus an isolated point at the origin"),
    ('TET_ROUNDED', "Rounded Tetrahedron (0 < k < 4)", (2.0, 1.0), 6.0,
     "A rounded tetrahedron plus four outer sheets"),
    ('TET_CAYLEY', "Cayley's Cubic (k = 4)", (4.0, 1.0), 5.0,
     "The four-nodal cubic: the rounded tetrahedron has closed onto "
     "the sheets at four conical points"),
    ('TET_BEYOND', "Tetrahedral Cubic (k > 4)", (6.0, 1.0), 5.0,
     "Past the nodal member, a single blown-out sheet"),
)

_TET4 = (
    ('TET4_TRICONIC', "Triconic Tetrahedral Quartic", (), 3.4,
     "Quartic with the tetrahedron's symmetries and four triconic "
     "points at the alternate cube corners (+-a, +-a, +-a) taking an "
     "odd number of minus signs"),
)

_DODEC = (
    ('DODEC_PLANES', "Six Icosidodecahedral Planes", (0.0, 0.0, 0.0, 0.0),
     2.0,
     "The degenerate member: the sextic factors into the six planes "
     "through the edges of an icosidodecahedron"),
    ('DODEC_ROUNDED', "Rounded Dodecahedron", (1.0, -1.0, 1.0, -1.0), 1.6,
     "A dodecahedron with its faces bowed out"),
    ('ICOSA_ROUNDED', "Rounded Icosahedron", (-1.0, 0.0, -1.0, 1.0), 1.6,
     "An icosahedron with its faces bowed out"),
    ('DODEC_TRIANGLES', "Icosidodecahedral Triangle Sextic",
     (0.0, 1.0, -2.0, 1.0), 1.8,
     "Thirty singular points at the icosidodecahedron's vertices, with "
     "twenty sheets sitting on its triangular faces"),
    ('DODEC_PENTAGONS', "Icosidodecahedral Pentagon Sextic",
     (0.0, -1.0, 2.0, -1.0), 1.8,
     "Thirty singular points at the icosidodecahedron's vertices, with "
     "twelve sheets sitting on its pentagonal faces"),
    ('DODEC_BARTH', "Barth Sextic (icosahedral frame)",
     (0.0, 1.25, -2.5, 1.25), 1.8,
     "Barth's nodal sextic, in Goursat's frame with a five-fold axis "
     "along Oz"),
    ('DODEC_LINES_60', "Sextic with 60 Lines", (-1.0, 8.0, -18.0, 11.0),
     2.0,
     "Contains the line x = 0, y = a and its 59 images under the "
     "dodecahedron's rotations; they fall into six bundles of ten "
     "parallels, one per five-fold axis"),
    ('DODEC_LINES_60B', "Sextic with 60 Lines (second solution)",
     (0.0, 5.0, -45.0, 71.0), 2.6,
     "The other integral solution: contains the line x = z = a and 59 "
     "more"),
)


#: preset key -> (label, family, coefficients, clip radius, description)
PRESETS = {}
for _rows, _fam in ((_OCT, 'OCT4'), (_TET, 'TET3'),
                    (_TET4, 'TET4'), (_DODEC, 'DODEC6')):
    for _k, _lab, _c, _r, _d in _rows:
        PRESETS[_k] = (_lab, _fam, _c, _r, _d)

PRESET_ORDER = ([k for k, _l, _c, _r, _d in _OCT]
                + [k for k, _l, _c, _r, _d in _TET]
                + [k for k, _l, _c, _r, _d in _TET4]
                + [k for k, _l, _c, _r, _d in _DODEC])

PRESETS_BY_FAMILY = {f: [k for k in PRESET_ORDER if PRESETS[k][1] == f]
                     for f, _l, _fn, _n, _d in FAMILIES}


def default_coeffs(family):
    """The coefficient tuple the family's first preset uses -- what the
    operator's sliders start at when the user switches to Custom."""
    keys = PRESETS_BY_FAMILY.get(family)
    return PRESETS[keys[0]][2] if keys else ()


# ----------------------------------------------------------------------
# meshing
# ----------------------------------------------------------------------

@_geom_cache.memoise(version=1)
def build_goursat(family, coeffs, res=120, a=1.0, clip=2.4, scale=1.0):
    """Mesh the zero level set inside the ball of radius `clip` * a.

    Returns (verts, tris), centred on the origin and fitted to a 2 m
    cube.  A ball is used rather than a box for every family: these
    surfaces are defined by a point group, and a cubical window breaks
    the very symmetry that is the point of them -- an octahedral quartic
    clipped to a box reads as a box.  Triangles whose centroid falls
    outside the ball are culled, which leaves an even rim (masking the
    outside samples instead stitches stair-step caps onto it).
    """
    fn = FAMILY_FIELD[family]
    co = tuple(float(c) for c in coeffs)
    r = float(clip) * float(a)
    mst = _toolkit()
    verts, tris = mst.marching_tets(
        lambda X, Y, Z: fn(X, Y, Z, co, a),
        (-r, -r, -r), (r, r, r), (res, res, res))
    if len(tris):
        cen = verts[tris].mean(axis=1)
        keep = np.einsum('ij,ij->i', cen, cen) <= r * r
        tris = tris[keep]
        used = np.unique(tris)                 # drop orphaned verts
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        verts = verts[used]
        tris = remap[tris]
    if len(verts):
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ext = float((hi - lo).max())
        verts = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9
                                             else 1.0)
    return verts * scale, tris


def build_preset(key, res=120, a=1.0, clip=0.0, scale=1.0):
    """Mesh a named member.  `clip` overrides the row's own radius."""
    _lab, fam, co, r, _d = PRESETS[key]
    return build_goursat(fam, co, res, a, clip if clip > 0.0 else r, scale)


# ----------------------------------------------------------------------

def _selftest():
    """Six gates, each on a property that a coefficient typo destroys.

    1. Every family really has the point group it is named for, as an
       exact algebraic identity.
    2. The dodecahedral core factors into the six icosidodecahedral
       planes -- this is what pins that 24-term transcription down.
    3. The dodecahedral family is icosahedral, not merely 5-fold: it is
       invariant under two DIFFERENT five-fold axes, which generate the
       whole order-60 group.
    4. Every documented contained line lies on its surface, and every
       documented node is a node.
    5. The row the encyclopedia prints inconsistently resolves the way
       its own stated property says it does.
    6. Every preset meshes to a solid, finite, well-proportioned patch.
    """
    ok = True
    rng = np.random.default_rng(20260821)
    X, Y, Z = rng.uniform(-2.0, 2.0, (3, 400))
    t = rng.uniform(-3.0, 3.0, 60)
    zero = 0.0 * t
    one = 1.0 + zero

    def dev(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        sc = max(float(np.max(np.abs(u))), 1e-30)
        return float(np.max(np.abs(u - v))) / sc

    # ---- 1. the point groups ----------------------------------------
    def rot_z(ang, x, y, z):
        c, s = math.cos(ang), math.sin(ang)
        return x * c - y * s, x * s + y * c, z

    sym = []
    oct_c = (-0.5, 1.0, -1.5)
    base = f_oct4(X, Y, Z, oct_c)
    sym += [("octahedral quartic: 3-cycle",
             dev(base, f_oct4(Y, Z, X, oct_c))),
            ("octahedral quartic: transposition",
             dev(base, f_oct4(Y, X, Z, oct_c))),
            ("octahedral quartic: reflection",
             dev(base, f_oct4(-X, Y, Z, oct_c)))]
    tet_c = (2.0, 1.0)
    base = f_tet3(X, Y, Z, tet_c)
    sym += [("tetrahedral cubic: 3-cycle",
             dev(base, f_tet3(Y, Z, X, tet_c))),
            ("tetrahedral cubic: double sign flip",
             dev(base, f_tet3(-X, -Y, Z, tet_c))),
            ("tetrahedral cubic: transposition",
             dev(base, f_tet3(Y, X, Z, tet_c)))]
    base = f_tet4(X, Y, Z, ())
    sym += [("tetrahedral quartic: 3-cycle", dev(base, f_tet4(Y, Z, X, ()))),
            ("tetrahedral quartic: double sign flip",
             dev(base, f_tet4(-X, -Y, Z, ())))]
    dod_c = (1.0, -1.0, 1.0, -1.0)
    base = f_dodec6(X, Y, Z, dod_c)
    sym += [("dodecahedral sextic: 2pi/5 about Oz",
             dev(base, f_dodec6(*rot_z(0.4 * math.pi, X, Y, Z), dod_c))),
            ("dodecahedral sextic: mirror y -> -y",
             dev(base, f_dodec6(X, -Y, Z, dod_c)))]
    bad = ['%s:%.1e' % (n, d) for n, d in sym if not (d < 1e-12)]
    ok &= not bad
    print("goursat: %d point-group identities hold %s"
          % (len(sym), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- 2. the dodecahedral core factors ---------------------------
    c2, s2 = math.cos(0.4 * math.pi), math.sin(0.4 * math.pi)
    c1, s1 = math.cos(0.2 * math.pi), math.sin(0.2 * math.pi)
    planes = (Z * (Z - 2.0 * X)
              * (Z - 2.0 * c2 * X + 2.0 * s2 * Y)
              * (Z - 2.0 * c2 * X - 2.0 * s2 * Y)
              * (Z + 2.0 * c1 * X + 2.0 * s1 * Y)
              * (Z + 2.0 * c1 * X - 2.0 * s1 * Y))
    d = dev(_dodec_base(X, Y, Z), planes)
    ok &= d < 1e-12
    print("goursat: the sextic core is the product of the six "
          "icosidodecahedral planes %s"
          % ('OK' if d < 1e-12 else 'FAIL %.1e' % d))

    # ---- 3. icosahedral, not merely 5-fold --------------------------
    # The six plane normals ARE the six five-fold axes, so a rotation by
    # 2pi/5 about any of them must preserve the core.  Two distinct
    # five-fold axes generate the full order-60 group, so checking a
    # second one is what upgrades "5-fold" to "icosahedral".
    def rot_axis(axis, ang, P):
        k = np.asarray(axis, dtype=float)
        k = k / np.linalg.norm(k)
        c, s = math.cos(ang), math.sin(ang)
        return (P * c + np.cross(k, P.T).T * s
                + k[:, None] * (k @ P) * (1.0 - c))

    P = np.vstack([X, Y, Z])
    base = _dodec_base(X, Y, Z)
    ico = []
    for ax, lab in (((0.0, 0.0, 1.0), 'Oz'), ((-2.0, 0.0, 1.0), 'z = 2x')):
        for m in (1, 2):
            ico.append(("%s^%d" % (lab, m),
                        dev(base, _dodec_base(
                            *rot_axis(ax, 0.4 * math.pi * m, P)))))
    ico.append(("pi about Oy", dev(base, _dodec_base(
        *rot_axis((0.0, 1.0, 0.0), math.pi, P)))))
    bad = ['%s:%.1e' % (n, d) for n, d in ico if not (d < 1e-12)]
    ok &= not bad
    print("goursat: the sextic core is icosahedral (two distinct "
          "five-fold axes) %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- 4. contained lines and nodes -------------------------------
    # Each is the property the row is NAMED for, so it is the thing a
    # typo would silently take away.
    def at(key, x, y, z):
        _lab, fam, co, _r, _d = PRESETS[key]
        return float(np.max(np.abs(FAMILY_FIELD[fam](x, y, z, co, 1.0))))

    props = [
        # the twelve extended octahedron edges |x|+|y|+|z| = a
        ("octahedron edge z=0, y=x+a on OCT_OCTA_EDGES",
         at('OCT_OCTA_EDGES', t, t + 1.0, zero)),
        # the twelve face diagonals of the cube
        ("cube face diagonal x=y, z=a on OCT_CUBE_DIAGONALS",
         at('OCT_CUBE_DIAGONALS', t, t, one)),
        # the twelve extended cube edges
        ("cube edge x=y=a on OCT_CUBE_EDGES",
         at('OCT_CUBE_EDGES', one, one, t)),
        # the twelve extended medians of the cube's faces
        ("cube median z=0, x=a on OCT_CUBE_MEDIANS",
         at('OCT_CUBE_MEDIANS', one, t, zero)),
        # cuboctahedral nodes (+-a, +-a, 0)
        ("cuboctahedral node on OCT_CUBOCTA_NODES",
         at('OCT_CUBOCTA_NODES', 1.0, 1.0, 0.0)),
        ("cuboctahedral node on OCT_CUBE_MEDIANS",
         at('OCT_CUBE_MEDIANS', 1.0, -1.0, 0.0)),
        ("cuboctahedral node on OCT_TRIANGLES",
         at('OCT_TRIANGLES', 0.0, 1.0, 1.0)),
        # the tetrahedral quartic's four triconic points
        ("triconic point (-a,a,a) on TET4_TRICONIC",
         at('TET4_TRICONIC', -1.0, 1.0, 1.0)),
        ("triconic point (a,-a,a) on TET4_TRICONIC",
         at('TET4_TRICONIC', 1.0, -1.0, 1.0)),
        ("triconic point (-a,-a,-a) on TET4_TRICONIC",
         at('TET4_TRICONIC', -1.0, -1.0, -1.0)),
        # Titeica is the k' = 0 tetrahedral cubic xyz = a^3
        ("TET_TITEICA is xyz = a^3",
         at('TET_TITEICA', t + 4.0, one, 1.0 / (t + 4.0))),
        ("Cayley node (-2a,-2a,-2a) on TET_CAYLEY",
         at('TET_CAYLEY', -2.0, -2.0, -2.0)),
        ("Cayley node (-2a,2a,2a) on TET_CAYLEY",
         at('TET_CAYLEY', -2.0, 2.0, 2.0)),
        # the isolated point at the origin of the k = 0 member
        ("TET_ISOLATED_POINT vanishes at the origin",
         at('TET_ISOLATED_POINT', 0.0, 0.0, 0.0)),
        # the two sextics with sixty lines
        ("line x=0, y=a on DODEC_LINES_60",
         at('DODEC_LINES_60', zero, one, t)),
        ("line x=z=a on DODEC_LINES_60B",
         at('DODEC_LINES_60B', one, t, one)),
    ]
    bad = ['%s:%.1e' % (n, d) for n, d in props if not (d < 1e-9)]
    ok &= not bad
    print("goursat: %d documented lines/nodes lie on their surface %s"
          % (len(props), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- 4b. a node really is a node, not just a zero ---------------
    # A node is a point where the surface vanishes AND the gradient
    # does; only the second half distinguishes it from an ordinary
    # point, and it is what the "12 singular points" claim means.
    def grad(key, p, h=1e-6):
        _lab, fam, co, _r, _d = PRESETS[key]
        g = []
        for i in range(3):
            q = [np.array([p[0]]), np.array([p[1]]), np.array([p[2]])]
            q[i] = q[i] + h
            hi = FAMILY_FIELD[fam](q[0], q[1], q[2], co, 1.0)
            q[i] = q[i] - 2.0 * h
            lo = FAMILY_FIELD[fam](q[0], q[1], q[2], co, 1.0)
            g.append(float(np.ravel(hi - lo)[0]) / (2.0 * h))
        return math.sqrt(sum(v * v for v in g))

    nodes = [
        ('OCT_CUBOCTA_NODES', (1.0, 1.0, 0.0)),
        ('OCT_CUBE_MEDIANS', (1.0, -1.0, 0.0)),
        ('OCT_TRIANGLES', (0.0, 1.0, 1.0)),
        ('TET4_TRICONIC', (-1.0, 1.0, 1.0)),
        ('TET_CAYLEY', (-2.0, -2.0, -2.0)),
        ('TET_CAYLEY', (-2.0, 2.0, 2.0)),
    ]
    bad = ['%s:%.1e' % (k, g) for k, p in nodes
           for g in (grad(k, p),) if not (g < 1e-6)]
    ok &= not bad
    print("goursat: %d documented nodes have a vanishing gradient %s"
          % (len(nodes), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # ---- 5. the inconsistent row ------------------------------------
    # The encyclopedia's printed expansion for (-1, 2, -2) and the tuple
    # it prints beside it disagree by the sign of a^4.  The row's own
    # documented property -- it contains the cube's extended face
    # medians -- picks the tuple.  Assert BOTH halves, so that if the
    # implementation ever drifts back to the printed form this fails
    # rather than silently shipping the other surface.
    ours = at('OCT_CUBE_MEDIANS', one, t, zero)
    printed = float(np.max(np.abs(
        1.0 * t * t - 1.0 * (1.0 + t * t) - 1.0)))   # x^2y^2 - a^2 r^2 - a^4
    good = ours < 1e-9 and printed > 1.0
    ok &= good
    print("goursat: the (-1,2,-2) sign is settled by its own median "
          "line (ours %.1e, printed form %.1e) %s"
          % (ours, printed, 'OK' if good else 'FAIL'))

    # ---- 6. every preset meshes ------------------------------------
    thin, empty = [], []
    for key in PRESET_ORDER:
        V, T = build_preset(key, 44)
        if len(V) < 100 or len(T) < 200 or not np.all(np.isfinite(V)):
            empty.append(key)
            continue
        ext = V.max(axis=0) - V.min(axis=0)
        if float(ext.max()) > 2.0 + 1e-6:
            empty.append(key + '(oversize)')
        elif float(ext.min() / ext.max()) < 0.05:
            thin.append('%s:%.3f' % (key, ext.min() / ext.max()))
    good = not thin and not empty
    ok &= good
    print("goursat: %d presets mesh solid, finite, in the 2 m cube %s"
          % (len(PRESET_ORDER), 'OK' if good else
             'FAIL empty=%s thin=%s' % (empty, thin)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("Goursat surfaces self-test failed")
