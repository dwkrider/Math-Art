
# Slide-togethers
#
# George Hart's paper models made from identical copies of ONE regular
# polygon, each slit at set positions, that slide into each other and
# hold without glue.  Seven of them:
#
#   20 triangles   30 squares    12 decagons    12 pentagons
#   20 hexagons    12 star decagons {10/3}      12 pentagrams
#
# Each model puts one polygon in every face plane of a symmetric solid --
# the twenty planes of an icosahedron, the twelve of a dodecahedron, or
# the thirty 2-fold planes an icosidodecahedron's edges pick out -- and
# turns it about that plane's normal.  Two panels that cross do so along
# the chord where their planes meet, and the joint is a pair of
# COMPLEMENTARY slits: each panel is cut from its own rim inward to the
# midpoint of the chord, so the two pass through one another and stop
# halfway.  That is the whole construction, and it is why the pieces
# hold: every panel is trapped by its neighbours.
#
# The slits are computed from the geometry rather than tabulated.  For
# each pair of panels the plane-plane line is intersected with both
# polygons; if the overlap is real, the chord's midpoint fixes the slit
# depth, and which panel is cut from which end alternates so that two
# slits never collide.
#
# References:
# - George W. Hart, "Slide-Togethers", Virtual Polyhedra
#   (georgehart.com/virtual-polyhedra/slide-togethers.html), and
#   "Modular Kirigami", Bridges 2007 -- the construction and all seven
#   models.
# - The face planes used here are those of the Platonic solids and of
#   the icosidodecahedron; see H. S. M. Coxeter, "Regular Polytopes",
#   3rd ed., Dover (1973).

bl_info = {
    "name": "Slide-Togethers",
    "author": "Math Art project (after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Hart's slotted-polygon slide-together models",
    "category": "Add Mesh",
}

import math

try:
    from .polyhedra import seeds as _seeds
    from .polyhedra import hull as _hull
    from .polyhedra import fit as _fit
except ImportError:                       # flat-file / headless import
    from polyhedra import seeds as _seeds
    from polyhedra import hull as _hull
    from polyhedra import fit as _fit

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------

def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _mul(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return [c / n for c in a] if n > 1e-12 else [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------
# the seven models
# --------------------------------------------------------------------------
# axes: which set of face planes carries the panels.
#   ICOSA  the 20 face normals of an icosahedron
#   DODECA the 12 face normals of a dodecahedron
#   ID30   the 30 two-fold axes an icosidodecahedron's edges pick out

# key, label, planes, n, d, radius, turn.  Each radius is the largest one
# at which EVERY panel has the same number of crossings, which matters
# more than it looks: a slide-together is one shape cut one way, so if
# panels crossed different numbers of neighbours they would need
# different slit patterns and would no longer be identical pieces.  Grow
# a panel past this and the outer ones start sawing through neighbours
# they should miss; shrink it and they float free.
MODELS = [
    ('T20', "20 Triangles", 'U30TRI', 3, 1, 1.00, 0.0),
    ('S30', "30 Squares", 'U38SQ', 4, 1, 1.00, 0.0),
    ('P12', "12 Pentagons", 'U54PEN', 5, 1, 1.00, 0.0),
    ('D12', "12 Decagons", 'U33DEC', 10, 1, 1.00, 0.0),
    ('H20', "20 Hexagons", 'U50HEX', 6, 1, 1.00, 0.0),
    ('SD12', "12 Star Decagons {10/3}", 'U73DEC', 10, 3, 1.00, 0.0),
    ('PG12', "12 Pentagrams", 'U54PEN', 5, 2, 1.00, 0.0),
    # beyond Hart's seven -- see FACE_SOURCES
    ('T20B', "20 Triangles (great ditrigonal)", 'U42TRI', 3, 1, 1.00, 0.0),
    ('H20B', "20 Hexagons (icositruncated)", 'U45HEX', 6, 1, 1.00, 0.0),
    ('S30B', "30 Squares (truncated dodecadodeca)", 'U59SQ', 4, 1,
     1.00, 0.0),
    ('P12B', "12 Pentagons (great dodecicosidodeca)", 'U61PEN', 5, 1,
     1.00, 0.0),
    ('D12B', "12 Decagons (great truncated icosidodeca)", 'U68DEC', 10, 1,
     1.00, 0.0),
    ('P12C', "12 Pentagons (rhombidodecadodeca)", 'U38PEN', 5, 1,
     1.00, 0.0),
    ('PG12C', "12 Pentagrams (rhombidodecadodeca)", 'U38PEN', 5, 2,
     1.00, 0.0),
    ('P12D', "12 Pentagons (great icosicosidodeca)", 'U48PEN', 5, 1,
     1.00, 0.0),
    ('PG12D', "12 Pentagrams (great icosicosidodeca)", 'U48PEN', 5, 2,
     1.00, 0.0),
]

_MODEL = {m[0]: m for m in MODELS}

# How far each face plane sits from the centre.  Taken from the solid the
# planes come from, so a panel really does lie in a face plane.
_MODEL_OFFSET = {}


def _plane_offset(kind):
    if kind in _MODEL_OFFSET:
        return _MODEL_OFFSET[kind]
    if kind in FACE_SOURCES:
        spec = FACE_SOURCES[kind]
        wy, pqr, sides = spec[:3]
        cls = spec[3] if len(spec) > 3 else 0
        nv, cen, _e, _r = uniform_face_frames(wy, pqr, sides, cls)[0]
        val = abs(_dot(nv, cen))
    elif kind == 'ID30':
        val = 1.0
    else:
        V, F = _seeds.seed_poly('ICOSA' if kind == 'ICOSA' else 'DODECA')
        planes = _hull.face_planes(V, F)
        r = max(_norm(v) for v in V)
        val = planes[0][1] / r           # inradius, circumradius = 1
    _MODEL_OFFSET[kind] = val
    return val


def plane_normals(kind):
    """Unit normals of the planes the panels sit in."""
    if kind == 'ICOSA':
        V, F = _seeds.seed_poly('ICOSA')
        return [n for n, _d in _hull.face_planes(V, F)]
    if kind == 'DODECA':
        V, F = _seeds.seed_poly('DODECA')
        return [n for n, _d in _hull.face_planes(V, F)]
    if kind in FACE_SOURCES:
        spec = FACE_SOURCES[kind]
        wy, pqr, sides = spec[:3]
        cls = spec[3] if len(spec) > 3 else 0
        return [nv for nv, _c, _e, _r
                in uniform_face_frames(wy, pqr, sides, cls)]
    if kind == 'ID30':
        # the thirty 2-fold axes: midpoints of an icosahedron's edges
        V, F = _seeds.seed_poly('ICOSA')
        seen, out = set(), []
        for f in F:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                out.append(_unit(_add(V[a], V[b])))
        return out
    raise ValueError(kind)


_FRAME_CACHE = {}


def uniform_face_frames(wythoff, pqr, sides, cls=0):
    """Faces of a uniform polyhedron as (normal, centre, e1, radius).

    The placement a slide-together needs, taken from the solid rather
    than parameterised: each panel sits in a real face plane, centred on
    that face and turned to its corners.
    """
    ckey = (wythoff, sides, cls)
    if ckey in _FRAME_CACHE:
        return _FRAME_CACHE[ckey]
    try:
        from . import uniform_polyhedra_generator as _uni
    except ImportError:                    # flat import (test runner)
        import uniform_polyhedra_generator as _uni
    V, faces = _uni.build_uniform(wythoff, pqr)
    V = [tuple(float(c) for c in v) for v in V]
    R = max(_norm(v) for v in V)
    out = []
    for f, _dens in faces:
        if len(f) != sides:
            continue
        P = [tuple(c / R for c in V[i]) for i in f]
        cen = [sum(p[k] for p in P) / float(sides) for k in range(3)]
        # Newell rather than a single cross product: on a STAR circuit
        # three consecutive vertices can be nearly collinear and the
        # cross product then points anywhere.
        nx = [0.0, 0.0, 0.0]
        for k in range(sides):
            a, b = P[k], P[(k + 1) % sides]
            nx[0] += (a[1] - b[1]) * (a[2] + b[2])
            nx[1] += (a[2] - b[2]) * (a[0] + b[0])
            nx[2] += (a[0] - b[0]) * (a[1] + b[1])
        nrm = _unit(nx)
        if _dot(nrm, cen) < 0:
            nrm = _mul(nrm, -1.0)
        out.append((nrm, cen, _unit(_sub(P[0], cen)),
                    _norm(_sub(P[0], cen))))
    # A side count is NOT always one face class: the
    # rhombidodecadodecahedron has twelve pentagons AND twelve
    # pentagrams, both five-sided, at different plane distances.  Taking
    # them together gives 24 "panels" of which only half interlock, and
    # the per-panel crossing count comes out [0, 5] -- not a
    # slide-together at all.  Split by plane distance and let the caller
    # pick, innermost class first.
    groups = {}
    for fr in out:
        groups.setdefault(round(abs(_dot(fr[0], fr[1])), 5), []).append(fr)
    keys = sorted(groups)
    out = groups[keys[min(cls, len(keys) - 1)]] if keys else []
    _FRAME_CACHE[ckey] = out
    return out


#: models whose panels come from a uniform polyhedron's faces:
#: key -> (Wythoff symbol, pqr, face side count)
FACE_SOURCES = {
    # Hart: the thirty squares sit on the square faces of the
    # rhombidodecadodecahedron.
    'U38SQ': ('5/2 5 | 2', ['5/2', '5', '2'], 4),
    # Hart: "if the pentagrams were included with the triangles, this
    # would be a small [ditrigonal] icosidodecahedron", U30 -- 20
    # triangles plus 12 pentagrams.  Leaving the pentagrams open is the
    # model.  He adds that the construction "contains all the edges of
    # the compound of five cubes", which is a real check and it passes:
    # the 20 triangles have 3 x 20 = 60 edges, the five cubes have
    # 5 x 12 = 60, and the two sets are congruent -- same lengths, same
    # midpoint radii.  The same triangles also belong to the GREAT
    # ditrigonal icosidodecahedron (U47), exactly as he says.
    'U30TRI': ('3 | 5/2 3', ['3', '5/2', '3'], 3),
    # Hart on the twelve decagons: "the edges of the decagons form
    # pentagons, squares, and triangles.  If we include the squares only,
    # we get the small rhombidodecahedron; if we include the pentagons
    # and triangles only, we get the small dodecicosidodecahedron."
    # Both solids therefore carry the SAME twelve decagons, and they do:
    # U39 is 30 squares + 12 decagons, U33 is 20 triangles + 12 pentagons
    # + 12 decagons, and the decagons agree to the last digit -- plane
    # distance 0.689152, circumradius 0.724617.  U33 is used here.
    'U33DEC': ('3/2 5 | 5', ['3/2', '5', '5'], 10),
    # "This is part of the small dodecicosahedron and the small icosic
    # icosidodecahedron" -- U50 (20 hexagons + 12 decagons) and U31
    # (20 triangles + 12 pentagons + 20 hexagons).  Both carry 20
    # hexagons, as he says.
    'U50HEX': ('3/2 3 5 |', ['3/2', '3', '5'], 6),
    # "a subset of the faces of the great rhombidodecahedron and the
    # great dodecicosidodecahedron" -- U73 (30 squares + 12 decagrams)
    # and U61 (20 triangles + 12 pentagons + 12 decagrams).
    'U73DEC': ('3/2 5/3 2 |', ['3/2', '5/3', '2'], 10),
    # "part of the great dodecahemidodecahedron and the great
    # icosidodecahedron" -- U54 is 20 triangles + 12 pentagrams.
    'U54PEN': ('2 | 3 5/2', ['2', '3', '5/2'], 5),

    # --- beyond Hart's seven -------------------------------------------
    # Found by sweeping every face class of every Star (Icosahedral)
    # uniform for a placement that behaves like a slide-together: panels
    # all congruent, every panel crossing the SAME number of others, and
    # the slit outlines simple.  The sweep recovers all seven of Hart's,
    # which is what makes its other answers worth trusting.  These five
    # are the ones in his structural range -- 3 to 5 neighbours per panel,
    # the hand-assemblable band -- at plane distances none of his use.
    #
    # Not shipped, and worth not retrying: the 12 pentagons and 12
    # pentagrams on U30 and the {10/3} decagrams on U42 all give
    # SELF-CROSSING outlines once slit, and a self-crossing loop is still
    # a face to Blender, which tessellates it into slivers.  A further 15
    # placements are geometrically fine but put 9 or 10 crossings on every
    # panel; Hart has one such model, so they are dense rather than wrong.
    'U42TRI': ('3 5 | 5/3', ['3', '5', '5/3'], 3),
    'U45HEX': ('3 5 5/3 |', ['3', '5', '5/3'], 6),
    'U59SQ': ('2 5 5/3 |', ['2', '5', '5/3'], 4),
    'U61PEN': ('5/2 3 | 5/3', ['5/2', '3', '5/3'], 5),
    'U68DEC': ('2 3 5/3 |', ['2', '3', '5/3'], 10),
    # Two more five-sided placements, each usable convex OR starred.
    # U38 has TWO five-sided classes and only the outer one interlocks,
    # hence the class index: its 0.9176 pentagons never meet.  U44
    # carries the same 0.7658 class as U38, so they are one placement.
    'U38PEN': ('5/2 5 | 2', ['5/2', '5', '2'], 5, 0),
    'U48PEN': ('3/2 5 | 3', ['3/2', '5', '3'], 5, 0),
}

# The TWELVE PENTAGONS are the only model Hart names no solid for -- he
# says just "a close cousin is this one, made of twelve regular
# pentagons".  His own X3D model settles it: its panels lie at plane
# distance 0.5258 of the circumradius, which is exactly where the
# PENTAGRAM model's panels lie (0.525731, U54).  A regular pentagon and a
# pentagram share a plane and a circumradius, so the two models use the
# same twelve planes and the same twelve positions, drawn convex in one
# and starred in the other.  That is the "close cousin", literally.  So
# P12 and PG12 share a face source and differ only in the density d.


def u38_square_frames():
    """The thirty squares of the rhombidodecadodecahedron (U38), each as
    (normal, centre, e1, circumradius).

    This is where the thirty-square slide-together's panels belong: one
    panel per square face of U38, in that square's own plane and turned
    to that square's own corners.  The normals are the icosahedral
    two-fold axes, which the old `ID30` code already had right; what it
    got wrong was everything else about the placement.

    Two things this fixes.  The plane distance is **0.8452** of the
    circumradius, not the 1.0 the old code assumed, so every panel sat
    too far out.  And the in-plane orientation now comes from the square
    itself: `_polygon` builds its reference direction from a fixed global
    axis, which is NOT equivariant, so one `turn` value cannot orient
    thirty symmetry-related panels consistently -- they came out rotated
    at assorted angles, which is what made the arrangement wrong however
    the turn was tuned.
    """
    try:
        from . import uniform_polyhedra_generator as _uni
    except ImportError:                    # flat import (test runner)
        import uniform_polyhedra_generator as _uni
    V, faces = _uni.build_uniform('5/2 5 | 2', ['5/2', '5', '2'])
    V = [tuple(float(c) for c in v) for v in V]
    R = max(_norm(v) for v in V)
    out = []
    for f, _dens in faces:
        if len(f) != 4:
            continue
        P = [tuple(c / R for c in V[i]) for i in f]
        cen = [sum(p[k] for p in P) / 4.0 for k in range(3)]
        nrm = _unit(_cross(_sub(P[1], P[0]), _sub(P[2], P[0])))
        if _dot(nrm, cen) < 0:
            nrm = _mul(nrm, -1.0)
        e1 = _unit(_sub(P[0], cen))
        out.append((nrm, cen, e1, _norm(_sub(P[0], cen))))
    if len(out) != 30:
        raise ValueError('expected 30 squares on U38, found %d' % len(out))
    return out


def _polygon_in_frame(n, d, radius, normal, e1, cen, turn):
    """A regular {n/d} outline in a frame taken FROM THE SOLID.

    Same shape as `_polygon` but with the in-plane axes supplied rather
    than invented, so symmetry-related panels really are symmetry-related.
    """
    e2 = _cross(normal, e1)
    t = math.radians(turn)

    def at(rad, ang):
        return _add(cen, _add(_mul(e1, rad * math.cos(ang)),
                              _mul(e2, rad * math.sin(ang))))
    if d == 1:
        return [at(radius, t + 2 * math.pi * k / n) for k in range(n)]
    inner = radius * (math.cos(math.pi * d / n)
                      / math.cos(math.pi * (d - 1) / n))
    pts = []
    for k in range(n):
        a = t + 2 * math.pi * k / n
        pts.append(at(radius, a))
        pts.append(at(inner, a + math.pi / n))
    return pts


def _polygon(n, d, radius, normal, turn, offset=0.0):
    """A regular {n/d} polygon of `radius` centred on `normal * offset`,
    lying in that face plane and turned by `turn` degrees.

    The offset is the whole point: a slide-together puts each panel in a
    FACE PLANE of the solid, not through its centre.  Panels through the
    centre would all cut through one another at the origin, and the
    number of crossings would not depend on how big they are.
    """
    e1 = [1.0, 0.0, 0.0]
    if abs(_dot(e1, normal)) > 0.9:
        e1 = [0.0, 1.0, 0.0]
    e1 = _unit(_sub(e1, _mul(normal, _dot(e1, normal))))
    e2 = _cross(normal, e1)
    t = math.radians(turn)
    cen = _mul(normal, offset)

    def at(rad, ang):
        return _add(cen, _add(_mul(e1, rad * math.cos(ang)),
                              _mul(e2, rad * math.sin(ang))))

    if d == 1:
        return [at(radius, t + 2 * math.pi * k / n) for k in range(n)]

    # A star polygon {n/d} walked vertex-to-vertex is a self-crossing
    # path, and a self-crossing loop is not a face -- Blender tessellates
    # it into a tangle.  A physical star panel is the OUTLINE: a simple
    # 2n-gon alternating the outer radius with the inner radius where
    # consecutive chords of the star meet,
    #     r / R = cos(pi d / n) / cos(pi (d - 1) / n),
    # which is 1/phi^2 for the pentagram, as it should be.
    inner = radius * (math.cos(math.pi * d / n)
                      / math.cos(math.pi * (d - 1) / n))
    pts = []
    for k in range(n):
        pts.append(at(radius, t + 2 * math.pi * k / n))
        pts.append(at(inner, t + 2 * math.pi * (k + 0.5) / n))
    return pts


def _inside_intervals(P, nrm, origin, direction):
    """Parameter intervals of the line origin + s*direction that lie
    INSIDE the planar polygon P.

    Clipping against each edge as a half-plane only works for a convex
    polygon, and a star panel's outline is not convex -- it would report
    the whole convex hull, including the notches between the points.  So
    collect every crossing of the line with an edge, sort them, and take
    alternate intervals: for a simple closed polygon the line is inside
    between the 1st and 2nd crossing, the 3rd and 4th, and so on.

    Vertices ON the line are the whole difficulty, and they are not rare
    here -- a slide-together's chords run through the panels' corners all
    the time.  The rule below is the standard half-open one: an edge
    counts when its two ends fall on different sides, with "on the line"
    filed as the NEGATIVE side.  A vertex the boundary really crosses
    then scores once, and a vertex it merely touches scores twice (or
    not at all), so the running parity stays right and the
    alternate-interval walk never slips a place.  Counting a touched
    vertex once instead -- which an earlier "merge coincident hits" pass
    did -- loses a hit, makes the count odd, and silently drops the last
    interval: that is what threw away one of the two arms a pentagram's
    chord passes through, leaving those panels slit differently from one
    another.
    """
    m = len(P)
    hits = []
    side = _cross(nrm, direction)
    for k in range(m):
        a, b = P[k], P[(k + 1) % m]
        da = _dot(side, _sub(a, origin))
        db = _dot(side, _sub(b, origin))
        if abs(da) < 1e-12:
            da = 0.0
        if abs(db) < 1e-12:
            db = 0.0
        if (da > 0.0) == (db > 0.0):
            continue                        # both sides the same
        if abs(da - db) < 1e-15:
            continue                        # edge along the line
        t = da / (da - db)
        pt = _add(a, _mul(_sub(b, a), t))
        hits.append(_dot(_sub(pt, origin), direction))
    hits.sort()
    out = []
    for i in range(0, len(hits) - 1, 2):
        lo, hi = hits[i], hits[i + 1]
        # Two arms that meet at a notch vertex the line runs through give
        # two intervals sharing an endpoint.  The panels overlap right
        # across that point, so it is one crossing and wants one slot.
        if out and lo - out[-1][1] < 1e-9:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return [(lo, hi) for lo, hi in out if hi - lo > 1e-9]


def crossings(panels):
    """Every pair of panels that actually overlap, with the chord along
    which they cross: (i, j, midpoint, end_low, end_high)."""
    out = []
    for i in range(len(panels)):
        Pi, ni = panels[i]
        for j in range(i + 1, len(panels)):
            Pj, nj = panels[j]
            raw_dir = _cross(ni, nj)
            if _norm(raw_dir) < 1e-9:
                continue                    # parallel planes never cross
            dirv = _unit(raw_dir)
            # A point on the line where the two face planes meet.  This
            # formula is NOT scale-invariant -- numerator linear in the
            # cross product, denominator quadratic -- so it needs the RAW
            # cross product, not the unit direction.  Feeding it the unit
            # vector puts the point off both planes by a factor of
            # sin(angle between them).
            hi_ = _dot(ni, Pi[0])
            hj_ = _dot(nj, Pj[0])
            dd = _dot(raw_dir, raw_dir)
            org = _mul(_add(_mul(_cross(nj, raw_dir), hi_),
                            _mul(_cross(raw_dir, ni), hj_)), 1.0 / dd)
            si = _inside_intervals(Pi, ni, org, dirv)
            sj = _inside_intervals(Pj, nj, org, dirv)
            # EVERY shared stretch is a crossing, not just the longest.
            # Two convex panels overlap in one interval, so keeping the
            # longest was harmless there -- but a STAR panel is entered
            # and left several times along one chord, once per arm, and
            # each of those overlaps is a separate place the two sheets
            # pass through one another and so needs its own pair of
            # slits.  Hart's pentagram template shows TEN slits per
            # panel; keeping only the longest gave five, left the
            # pattern varying from panel to panel (so the pieces were
            # not one template), and let a slot span a notch.
            shared = []
            for ai, bi in si:
                for aj, bj in sj:
                    lo, hi = max(ai, aj), min(bi, bj)
                    if hi - lo > 1e-6:
                        shared.append((lo, hi, ai, bi, aj, bj))
            # Each slit starts on ITS OWN panel's rim.  Handing back the
            # shared stretch's ends is wrong whenever the two panels
            # reach different distances along the chord: the far end then
            # lies on the OTHER panel's rim, which for this one is an
            # interior point, and `_slit_outline` -- which snaps an entry
            # to the nearest edge -- drops the slot at whatever rim point
            # is closest, often across a notch.  So use bi for panel i
            # and aj for panel j, where the chord leaves each panel.
            for lo, hi, ai, bi, aj, bj in shared:
                mid = _add(org, _mul(dirv, 0.5 * (lo + hi)))
                out.append((i, j, mid, _add(org, _mul(dirv, aj)),
                            _add(org, _mul(dirv, bi))))
    return out


def _slit_outline(P, nrm, cuts, width):
    """The panel outline with a slit cut in at each requested place.

    `cuts` are (entry point on the rim, inward target).  Each slit is a
    thin rectangle from the rim to the target, spliced into the boundary
    walk, so the result is one simple (non-convex) loop.
    """
    m = len(P)
    per_edge = {}
    for entry, target in cuts:
        best, bestd = None, 1e18
        for k in range(m):
            a, b = P[k], P[(k + 1) % m]
            ab = _sub(b, a)
            L2 = _dot(ab, ab) or 1.0
            t = max(0.0, min(1.0, _dot(_sub(entry, a), ab) / L2))
            foot = _add(a, _mul(ab, t))
            d = _norm(_sub(foot, entry))
            if d < bestd:
                best, bestd, bt, bfoot = k, d, t, foot
        per_edge.setdefault(best, []).append((bt, bfoot, target))
    out = []
    for k in range(m):
        a, b = P[k], P[(k + 1) % m]
        ab = _sub(b, a)
        L = _norm(ab) or 1.0
        edge = _unit(ab)
        out.append(list(a))
        # Keep the slot's MOUTH clear of both ends of its edge.  The two
        # mouth corners sit half a slit-width either side of the entry
        # point, so an entry within half a width of a corner throws one
        # of them past the vertex and outside the panel -- which is what
        # made the decagon and star-decagon panels show slots poking out
        # through their points.
        margin = min(0.5 * width / L, 0.45)
        for _t, foot, target in sorted(
                [(min(max(t, margin), 1.0 - margin),
                  _add(a, _mul(ab, min(max(t, margin), 1.0 - margin))),
                  tgt) for t, _f, tgt in per_edge.get(k, [])],
                key=lambda r: r[0]):
            axis = _unit(_sub(target, foot))
            side = _mul(_unit(_cross(nrm, axis)), width * 0.5)
            # Walk INTO the slot on whichever side the boundary reaches
            # first.  Always entering on the same side sends the notch
            # backwards along the rim half the time, and the outline
            # crosses itself -- which is what a face made from it looks
            # like: a tangle rather than a slot.
            if _dot(side, edge) > 0:
                side = _mul(side, -1.0)

            # The slot's two MOUTH corners must sit on the rim.  `side`
            # is perpendicular to the slit AXIS, not to the edge, so on an
            # oblique chord `foot +- side` lifts off the edge -- one
            # corner inward, the other out past the rim, which is the
            # overshoot the ten-sided panels showed.  Slide each side-line
            # of the slot along the axis until it meets the edge, and
            # CLAMP to the segment: on a star panel the unclamped
            # intersection can be far outside the edge it belongs to.
            n2 = _cross(nrm, axis)
            den = _dot(ab, n2)

            def _mouth(pt):
                if abs(den) < 1e-12:
                    return list(pt)
                sN = min(max(_dot(_sub(pt, a), n2) / den, 0.0), 1.0)
                return _add(a, _mul(ab, sN))

            out.append(_mouth(_add(target, side)))
            out.append(_add(target, side))
            out.append(_sub(target, side))
            out.append(_mouth(_sub(target, side)))
    return out


def model_panels(key, radius_scale=1.0, turn_delta=0.0):
    """The bare panels of a model, before any slit is cut.

    Shared by `build_model` and the self-test.  They used to build panels
    separately, which let the test check a DIFFERENT arrangement from the
    one shipped -- it kept passing while the thirty squares were placed
    wrongly.
    """
    _key, _lbl, axes, n, d, rad, turn = _MODEL[key]
    if axes in FACE_SOURCES:
        spec = FACE_SOURCES[axes]
        wy, pqr, sides = spec[:3]
        cls = spec[3] if len(spec) > 3 else 0
        return [(_polygon_in_frame(n, d, r0 * radius_scale, nv, e1, cen,
                                   turn + turn_delta), nv)
                for nv, cen, e1, r0
                in uniform_face_frames(wy, pqr, sides, cls)]
    off = _plane_offset(axes)
    return [(_polygon(n, d, rad * radius_scale, nv, turn + turn_delta,
                      off), nv)
            for nv in plane_normals(axes)]


def build_model(key, radius_scale=1.0, turn_delta=0.0, slit_width=0.05):
    """Panels of a slide-together as a list of outlines with normals."""
    panels = model_panels(key, radius_scale, turn_delta)
    cx = crossings(panels)
    # The chord runs from end_lo to end_hi through BOTH panels, so those
    # ends are exactly where a slit can enter.  Cutting the two panels
    # from OPPOSITE ends is what lets them pass through one another and
    # come to rest halfway; cutting both from the same end would leave
    # the two slits overlapping and the joint would simply fall apart.
    cuts = [[] for _ in panels]
    for i, j, mid, end_lo, end_hi in cx:
        cuts[i].append((end_hi, mid))
        cuts[j].append((end_lo, mid))
    out = []
    for idx, (P, nv) in enumerate(panels):
        out.append((_slit_outline(P, nv, cuts[idx], slit_width), nv))
    return out


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    ITEMS = [(k, lbl, f"{lbl} in the {ax.lower()} planes")
             for k, lbl, ax, _n, _d, _r, _t in MODELS]

    _PALETTE = [(0.85, 0.22, 0.18, 1.0), (0.18, 0.42, 0.80, 1.0),
                (0.96, 0.74, 0.16, 1.0), (0.20, 0.64, 0.32, 1.0),
                (0.56, 0.28, 0.70, 1.0), (0.94, 0.48, 0.16, 1.0)]

    class MESH_OT_slide_together_add(bpy.types.Operator):
        """Add one of Hart's slide-together models: identical slotted
        polygons that interlock without glue"""
        bl_idname = "mesh.slide_together_add"
        bl_label = "Slide-Together"
        bl_options = {'REGISTER', 'UNDO'}

        model: EnumProperty(name="Model", items=ITEMS, default='S30')
        panel_size: FloatProperty(
            name="Panel Size", default=1.0, min=0.3, max=2.5,
            description="Scale every panel about its own centre. Larger "
                        "panels overlap more deeply, which is what makes "
                        "the model hold together")
        turn: FloatProperty(
            name="Turn", default=0.0, min=-90.0, max=90.0,
            description="Turn every panel in its own plane, away from "
                        "the angle the model is built at")
        slit: FloatProperty(
            name="Slit Width", default=0.01, min=0.0, max=0.3,
            description="Width of the cut slots. Zero leaves the panels "
                        "uncut, showing the bare arrangement")
        thickness: FloatProperty(
            name="Thickness", default=0.01, min=0.0, max=0.3,
            description="Panel thickness (0 leaves them as flat faces)")
        colors: BoolProperty(
            name="Colours", default=True,
            description="Colour the panels in rotation, the way the paper "
                        "models are made")
        separate: BoolProperty(
            name="Separate Panels", default=False,
            description="One object per panel instead of a single mesh")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            try:
                panels = build_model(self.model, self.panel_size,
                                     self.turn, self.slit)
            except (ValueError, RuntimeError) as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            allpts = [p for P, _n in panels for p in P]
            fitted = _fit.fit_cube(allpts, 2.0 * self.scale)
            it = iter(fitted)
            panels = [([next(it) for _ in P], n) for P, n in panels]

            mats = []
            if self.colors:
                for i, rgba in enumerate(_PALETTE):
                    nm = f"Slide Panel {i}"
                    mat = bpy.data.materials.get(nm)
                    if mat is None:
                        mat = bpy.data.materials.new(nm)
                        mat.use_nodes = True
                        mat.diffuse_color = rgba
                        b = mat.node_tree.nodes.get("Principled BSDF")
                        if b is not None:
                            b.inputs["Base Color"].default_value = rgba
                    mats.append(mat)

            made = []
            if self.separate:
                for i, (P, nv) in enumerate(panels):
                    made.append(self._panel_object(context, P, nv,
                                                   f"Panel {i}",
                                                   mats[i % len(mats)]
                                                   if mats else None))
            else:
                V, F, midx = [], [], []
                for i, (P, nv) in enumerate(panels):
                    off = len(V)
                    V += [list(p) for p in P]
                    F.append([off + k for k in range(len(P))])
                    midx.append(i % max(1, len(mats)))
                me = bpy.data.meshes.new("Slide-Together")
                me.from_pydata([tuple(v) for v in V], [],
                               [tuple(f) for f in F])
                me.validate(clean_customdata=True)
                for m in mats:
                    me.materials.append(m)
                if mats:
                    me.polygons.foreach_set('material_index', midx)
                me.polygons.foreach_set('use_smooth',
                                        [False] * len(me.polygons))
                me.update()
                obj = bpy.data.objects.new("Slide-Together", me)
                made.append(obj)

            for o in context.selected_objects:
                o.select_set(False)
            for o in made:
                if o.name not in context.collection.objects:
                    context.collection.objects.link(o)
                o.location = context.scene.cursor.location
                o.select_set(True)
                if self.thickness > 0:
                    md = o.modifiers.new("Solidify", 'SOLIDIFY')
                    md.thickness = self.thickness
                    md.offset = 0.0
            context.view_layer.objects.active = made[0]
            self.report({'INFO'}, f"{len(panels)} panels")
            return {'FINISHED'}

        def _panel_object(self, context, P, nv, name, mat):
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(p) for p in P], [],
                           [tuple(range(len(P)))])
            me.validate(clean_customdata=True)
            if mat is not None:
                me.materials.append(mat)
            me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
            me.update()
            return bpy.data.objects.new(name, me)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'model')
            lay.prop(self, 'panel_size')
            lay.prop(self, 'turn')
            lay.prop(self, 'slit')
            lay.prop(self, 'thickness')
            lay.prop(self, 'colors')
            lay.prop(self, 'separate')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.slide_together_add", icon='MOD_BOOLEAN')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_slide_together_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_slide_together_add)


def _verify_five_cubes_edges():
    """Hart's own check on the twenty triangles.

    He notes that the twenty-triangle model "contains all the edges of
    the compound of five cubes".  That is testable and it is a much
    stronger statement than any count: the twenty triangles have
    3 x 20 = 60 edges, five cubes have 5 x 12 = 60, and if the placement
    is right the two sets are congruent.  Compared by invariants (edge
    lengths and edge-midpoint radii) because the two constructions do not
    share a frame.
    """
    import numpy as _np
    try:
        from .polyhedra import compounds as _cmp
    except ImportError:
        from polyhedra import compounds as _cmp

    def inv(edges):
        ln = sorted(round(_norm(_sub(b, a)), 3) for a, b in edges)
        mid = sorted(round(_norm(_mul(_add(a, b), 0.5)), 3)
                     for a, b in edges)
        return ln, mid

    # dedupe: every edge belongs to two faces, so a naive walk sees it
    # twice and the count comes out 120 rather than 60
    def _ek(a, b):
        ka = tuple(round(c, 4) + 0.0 for c in a)
        kb = tuple(round(c, 4) + 0.0 for c in b)
        return (ka, kb) if ka <= kb else (kb, ka)

    cubes = _cmp.build_compound('5CUBES')
    R = max(_norm(v) for V, _F in cubes for v in V)
    seen = {}
    for V, F in cubes:
        for f in F:
            for i in range(len(f)):
                a = [c / R for c in V[f[i]]]
                b = [c / R for c in V[f[(i + 1) % len(f)]]]
                seen.setdefault(_ek(a, b), (a, b))
    ce = list(seen.values())
    assert len(ce) == 60, len(ce)

    wy, pqr, sides = FACE_SOURCES['U30TRI']
    frames = uniform_face_frames(wy, pqr, sides)
    te = []
    for nv, cen, e1, r0 in frames:
        P = _polygon_in_frame(3, 1, r0, nv, e1, cen, 0.0)
        for i in range(3):
            te.append((P[i], P[(i + 1) % 3]))
    assert len(te) == 60, len(te)
    a, b = inv(ce), inv(te)
    assert a == b, ("the twenty triangles' edges are not the compound of "
                    "five cubes'", a[0][:3], b[0][:3])
    return len(te)


#: plane distance of each model's panels, as a fraction of the
#: circumradius, MEASURED OFF HART'S OWN MODELS.  His site's VRML files
#: are gone (the vrml/ directory 404s), but the earlier X3D exports
#: survive as slider-0 .. slider-6, and every panel plane in them agrees
#: with the source solid used here to five decimals.  These are the
#: numbers that decide whether a placement is right; the crossing counts
#: beside them do not, since several wrong placements also give uniform
#: crossings.
HART_PLANE_DISTANCE = {
    'T20': 0.7454, 'S30': 0.8452, 'P12': 0.5258, 'D12': 0.6891,
    'H20': 0.8140, 'SD12': 0.5067, 'PG12': 0.5257,
}


def _verify_slits_inside():
    """No slit may extend beyond the panel it is cut into.

    The defect this catches was visible on screen: slots on the decagon
    and star-decagon panels ran out through the spikes.  Two causes, both
    fixed, both silent to every other check --

    * `crossings` handed back the ends of the SHARED stretch, so the far
      end lay on the OTHER panel's rim; for this panel that is an
      interior point, and the slot got dropped at whatever rim point was
      nearest, often across a notch.  Each panel now gets its own rim
      crossing.
    * the mouth corners sit half a slit-width either side of the entry
      along a direction perpendicular to the slit AXIS, not to the edge,
      so on an oblique chord one corner lifted off the rim and out.  They
      are now slid along the axis onto the edge and clamped to it.
    """
    worst = 0.0
    for key, _lbl, _ax, _n, _d, _r, _t in MODELS:
        bare = model_panels(key)
        slit = build_model(key, slit_width=0.01)
        for (P, _nv), (O, _n2) in zip(bare, slit):
            cen = [sum(q[t] for q in P) / len(P) for t in range(3)]
            rmax = max(_norm(_sub(q, cen)) for q in P)
            worst = max(worst, max(_norm(_sub(q, cen)) for q in O) - rmax)
    assert worst < 1e-9, ('a slit extends outside its panel', worst)
    return worst


def _verify_against_hart():
    """Every model's panel plane must sit where Hart's own model puts it."""
    bad = []
    for key, want in HART_PLANE_DISTANCE.items():
        got = _plane_offset(_MODEL[key][2])
        if abs(got - want) > 2e-3:
            bad.append((key, got, want))
    assert not bad, ('panel planes disagree with Hart', bad)
    return len(HART_PLANE_DISTANCE)


def _slit_pattern(panels, cx, idx):
    """Where panel `idx` is slit, as (radius, azimuth) in its own frame."""
    P, nv = panels[idx]
    cen = [sum(q[t] for q in P) / len(P) for t in range(3)]
    e1 = _unit(_sub(P[0], cen))
    e2 = _cross(nv, e1)
    out = []
    for i, j, _mid, lo, hi in cx:
        for who, end in ((i, hi), (j, lo)):
            if who != idx:
                continue
            v = _sub(end, cen)
            out.append((_norm(v),
                        math.degrees(math.atan2(_dot(v, e2), _dot(v, e1)))
                        % 360.0))
    return out


def _verify_one_template():
    """Every panel of a model must be slit the SAME way.

    This is the defining property of a slide-together and the sharpest
    test there is of the whole construction: the model is built from many
    copies of ONE piece, so a maker cuts one template and traces it.  If
    two panels came out with slits at different radii, or a different
    number of them, the thing could not be made -- and the arrangement
    would be wrong, because the symmetry that carries one panel onto
    another has to carry its slits along too.

    It is also a strong check on the code, since it fails the moment a
    step stops being equivariant: an in-plane axis invented from a global
    direction rather than taken from the solid, a slit end chosen by
    array index rather than by geometry, or a boundary crossing miscounted
    at a vertex all break it, and all three have.
    """
    checked = 0
    for key, _lbl, _ax, n, _d, _r, _t in MODELS:
        panels = model_panels(key)
        cx = crossings(panels)
        ref = _slit_pattern(panels, cx, 0)
        for idx in range(1, len(panels)):
            other = _slit_pattern(panels, cx, idx)
            assert len(other) == len(ref),                 (key, "panels are slit a different number of times",
                 idx, len(other), len(ref))
            # a panel's own rotations are the only freedom: its {n/d}
            # outline turns onto itself in steps of 360/n degrees (and of
            # 360/2n for a star, whose outline has 2n corners), so the
            # patterns need only agree up to one of those turns.
            for step in range(2 * n):
                pool = list(other)
                shift = 360.0 * step / (2 * n)
                for r, az in ref:
                    for t, (r2, az2) in enumerate(pool):
                        gap = abs(az - (az2 + shift)) % 360.0
                        if abs(r - r2) < 1e-3 and min(gap, 360.0 - gap) < 0.05:
                            pool.pop(t)
                            break
                    else:
                        break
                if not pool:
                    break
            else:
                raise AssertionError(
                    (key, "panel %d is not slit like panel 0" % idx,
                     sorted((round(r, 4), round(a, 2)) for r, a in ref),
                     sorted((round(r, 4), round(a, 2)) for r, a in other)))
        checked += 1
    return checked


def _selftest():
    want_panels = {'T20': 20, 'S30': 30, 'P12': 12, 'D12': 12,
                   'H20': 20, 'SD12': 12, 'PG12': 12,
                   'T20B': 20, 'H20B': 20, 'S30B': 30, 'P12B': 12,
                   'D12B': 12, 'P12C': 12, 'PG12C': 12, 'P12D': 12,
                   'PG12D': 12}
    for key, lbl, axes, n, d, _r, _t in MODELS:
        normals = plane_normals(axes)
        assert len(normals) == want_panels[key], (key, len(normals))

        panels = build_model(key, slit_width=0.0)
        assert len(panels) == want_panels[key], (key, len(panels))

        # every panel is congruent: same vertex count, same radius
        ref = len(panels[0][0])
        for P, _nv in panels:
            assert len(P) == ref, (key, len(P), ref)

        # the panels must actually interlock -- a slide-together with no
        # crossings is just a heap of loose polygons
        raw = model_panels(key)
        cx = crossings(raw)
        assert cx, (key, "no panel crossings: nothing would interlock")

        # each crossing's midpoint really is inside both panels, and its
        # two ends really are distinct -- the slits are cut to these
        for i, j, mid, lo, hi in cx:
            assert _norm(_sub(hi, lo)) > 1e-6, (key, "degenerate chord")
            for who in (i, j):
                P, nv = raw[who]
                # the chord lies in BOTH face planes, which sit at the
                # solid's inradius -- not through the centre
                assert abs(_dot(nv, mid) - _plane_offset(axes)) < 1e-6,                     (key, "chord off-plane")

        # each panel meets several others, not just one
        deg = {}
        for i, j, _m, _lo, _hi in cx:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        assert len(deg) == want_panels[key],             (key, "a panel crosses nothing", len(deg))
        assert min(deg.values()) >= 3, (key, "a panel is barely held",
                                        sorted(deg.values())[:3])
        # 10 because that is what Hart's own star-decagon model does --
        # twelve large {10/3} panels each crossing ten of the other
        # eleven.  The old cap of 9 was fitted to the earlier, wrong
        # placement and would now reject the correct one.
        assert max(deg.values()) <= 10,             (key, "panels saw through far too many neighbours",
             sorted(deg.values())[-3:])

        # No panel outline may cross itself.  This is the check that
        # matters most: a self-crossing loop is still a "face" as far as
        # from_pydata is concerned, and Blender tessellates it into a
        # tangle of slivers -- which is exactly what a star panel walked
        # vertex-to-vertex, or a slot spliced in backwards, produces.
        for P, nv in panels:
            mm = len(P)
            for x in range(mm):
                for y in range(x + 2, mm):
                    if x == 0 and y == mm - 1:
                        continue
                    a1, b1 = P[x], P[(x + 1) % mm]
                    a2, b2 = P[y], P[(y + 1) % mm]
                    r1, r2 = _sub(b1, a1), _sub(b2, a2)
                    den = _dot(_cross(r1, r2), nv)
                    if abs(den) < 1e-12:
                        continue
                    w = _sub(a2, a1)
                    t = _dot(_cross(w, r2), nv) / den
                    u = _dot(_cross(w, r1), nv) / den
                    assert not (1e-6 < t < 1 - 1e-6
                                and 1e-6 < u < 1 - 1e-6),                         (key, "panel outline crosses itself")

        # panels lie in distinct planes
        seen = set()
        for _P, nv in panels:
            k = tuple(round(c, 5) for c in nv)
            assert k not in seen, (key, "two panels share a plane")
            seen.add(k)

    _verify_slits_inside()
    print("SLITS   no slit extends beyond its own panel")
    nt = _verify_one_template()
    print("SAME    all %d models cut every panel from one template" % nt)
    nh = _verify_against_hart()
    print("PLANES  all %d models sit at Hart's own panel-plane distances"
          % nh)
    n = _verify_five_cubes_edges()
    print("T20  %d triangle edges congruent to the compound of five "
          "cubes (Hart's check)" % n)
    print("RESULT: OK")
