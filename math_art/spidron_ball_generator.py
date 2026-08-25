# Spidroball -- spidronised solids.
#
# Take a polyhedron and replace every face with a spidron nest.  The
# result is what Daniel Erdely, Amina Buhler-Allen and Marc Pelletier
# called a spidroball; Paul Nylander's dodecahedral version is the
# widely reproduced picture of it, and the rhombic triacontahedron
# assembled from thirty spidron pairs is the one the press showed.
#
# THE GENERAL OPERATION.  Van Ballegooijen, Gailiunas and Erdely
# generalised spidronisation from the hexagon to ANY polygon, skew ones
# included, and their construction is the one used here: copy the
# polygon, scale it toward its centre, rotate it about the face normal,
# triangulate the annulus between the two by joining every vertex to the
# images of its two neighbours, and repeat.  Every vertex's images then
# lie on a logarithmic spiral.
#
# FLAT OR IN RELIEF.  With Relief at zero each face is decorated in its
# own plane -- always well defined, and the safe starting point.  Raising
# Relief alternates the face's boundary points above and below the face
# plane first, turning it into a regular skew polygon, and it is the
# skew polygon that gives the deep, folded spidroball look.  A face with
# an odd number of sides is refined with edge midpoints first so the
# alternation can close.
#
# HOW the boundary is displaced decides whether the result is Nylander's
# interlocked ball or a heap of separate spikes, and the difference is
# worth stating plainly.  A solid's vertex belongs to three faces and an
# edge midpoint to two.  Displace each along ITS OWN FACE'S NORMAL
# (Relief Style: Per Face) and the three copies of a shared vertex fly
# apart in three directions, tearing the surface into disconnected
# shards.  Displace RADIALLY instead (Relief Style: Woven, the default)
# -- vertices pushed out, midpoints pulled in -- and every face computes
# the same position for a shared point, because the displacement depends
# only on the point and the centre.  The nests then meet exactly and the
# ball reads as one continuous corrugated surface of interlocking
# spirals.  The parities agree for free: on the refined boundary the
# solid's vertices always land on even indices and the midpoints on odd
# ones, in every face.
#
# WHICH WAY the radial displacement goes matters as much as the fact
# that it is radial.  The solid's VERTICES move IN and its EDGE
# MIDPOINTS move OUT.  Pushing the vertices out instead raises a spike
# where three faces already converge on a point, and the ball comes out
# a hedgehog; raising the edge midpoints lifts a ridge along every
# edge, and the arms of the two faces that share it sweep up over that
# ridge together, which is what makes the surface read as woven.
#
# Nylander reached the same interlocking by a different route, and his
# 2010 source is reproduced in `nylander_dodeca_nest` as a check.  He
# lifts along the face normal -- the Per Face style -- but SOLVES the
# lift from the dihedral angle, dz = a*b/sqrt(a*b^2 + 1) with
# a = sin^2(pi/n) and b = tan(alpha/2), so that the raised points of
# neighbouring faces land on one another regardless.  The self-test
# confirms his twelve faces do share every boundary point, to 3e-17.
# His solution is exact for the dodecahedron; the radial displacement
# used here is not tuned to any particular solid, so it holds on all
# seven seeds and at any relief.  His boundary is in fact this module's
# Woven boundary at relief tan^2(pi/10) = 0.105573 exactly -- the value
# `regular_relief` derives as the one relief that puts all ten boundary
# points of a face at the SAME distance from the face's axis, i.e. that
# makes the skew decagon regular.
#
# THE TWIST THAT MAKES THE PETALS.  Nylander's ring recursion is
# Rz(dtheta) * (2/3) per ring with dtheta = 36.53 degrees -- one FULL
# node step of the ten-point boundary (36 deg) plus half a degree.  The
# full step is the load-bearing part, and it is not a planar-spidron
# habit: the skew decagon's corrugation has period TWO nodes, so
# rotating by one step maps the boundary onto its parity mirror --
# raised points land where lowered ones were.  Each ring therefore
# advances the corrugation phase by one node, an arm climbs over the
# adjacent fold as it spirals in, and the outermost triangles stay wide
# and flat -- the broad petals and deep five-arm vortices of his
# renders.  A small fixed twist (this generator's old default) never
# flips the parity, and the same kernel yields a shallow rosette
# instead.  Twist Style: Advance builds the full step in for any face
# size (2*pi/m for an m-point boundary, so pentagons and hexagons each
# advance by their own step on the truncated icosahedron); the Twist
# knob is then only the excess on top.  Nylander's own excess, 0.53 deg,
# comes from a formula his source comments doubt ("not sure if this is
# right"), and at 8 rings it curves each arm by all of 4 degrees; zero
# excess is visually the same object.  With Advance, scale 2/3, 8
# rings, relief 0.10557 and uniform chirality, this module reproduces
# his published ball exactly (vertex-for-vertex, to 5e-16 -- asserted
# in `_selftest`).
#
# TWO HONEST LIMITS.  First, the general construction is always
# DRAWABLE but not generally FOLDABLE: only regular skew polygons
# reliably admit a folding, and the two degrees of freedom that case has
# are not parameterised in the literature.  What this generator makes is
# the drawn surface, not a claim that a paper model folds flat.  Second,
# chirality has to alternate across a shared face in any aggregation of
# spidronised solids -- a rule that is invisible on a single solid, so
# Alternate two-colours the faces and says so when the face graph has an
# odd cycle and no such colouring exists.
#
# That obstruction is the common case, not the exception.  A solid's
# faces two-colour exactly when its DUAL's edge graph is bipartite, and
# among the seeds offered here only the OCTAHEDRON qualifies: its dual
# is the cube, whose faces are quadrilaterals, while every other dual
# has triangular or pentagonal faces and hence an odd cycle.  On the
# other solids Alternate is a best effort -- on the dodecahedron it
# leaves 12 of the 30 face joins sharing a chirality, against all 30 for
# a uniform winding -- and the operator reports that it could not do
# better.  This is the same fact the Bridges paper states as "a
# polyhedron with an odd number of identical faces needs two different
# spidronised forms" in a space filling.
#
# References:
# - Walt van Ballegooijen, Paul Gailiunas & Daniel Erdely,
#   "Spidronised Space-fillers", Bridges 2009 Conference Proceedings,
#   pp. 271-278 -- spidronisation of an arbitrary, possibly skew,
#   polygon; the catalogue of nests; and the rule that a clockwise face
#   must meet an anticlockwise one across a shared face.
# - Daniel Erdely & Marc Pelletier, "Spidron Domain: The Expanding
#   Spidron Universe", Bridges 2006 Conference Proceedings,
#   pp. 549-550 -- the spidroball family.
# - Daniel Erdely, "Some Surprising New Properties of the Spidrons",
#   Bridges 2005 Conference Proceedings, pp. 179-186 -- the hexagonal
#   nest the face decoration generalises.
# - Peter Pearce, "Structure in Nature is a Strategy for Design" (MIT
#   Press, 1978), ch. 8 -- the saddle polyhedra and space-filling
#   systems whose faces the Bridges catalogue spidronises.
# - Paul Nylander, "Dodeca-Spidroball" (bugman123.com, 2010), AutoLisp
#   and POV-Ray sources; the dihedral-solved lift reproduced in
#   `nylander_dodeca_nest` and checked in the self-test.

bl_info = {
    "name": "Spidroball",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Polyhedra",
    "description": "Spidronised solids: every face replaced by a "
                   "spiral nest of triangles",
    "category": "Add Mesh",
}

from math import (cos, sin, tan, acos, atan2, pi, sqrt, radians,
                  degrees)

import numpy as np

try:
    from . import spidron_math as sm
    from .polyhedra import seeds as _seeds
    from .polyhedra import hull as _hull
    from .polyhedra import fit as _fit
    from .patterns import common as pc
    from . import sharp_creases as _sc
except Exception:                       # legacy single-file / CLI use
    import spidron_math as sm
    from polyhedra import seeds as _seeds
    from polyhedra import hull as _hull
    from polyhedra import fit as _fit
    from patterns import common as pc
    import sharp_creases as _sc

PHI = 0.5 * (1.0 + sqrt(5.0))

SEED_ITEMS = [
    ('TETRA', "Tetrahedron", "Four triangles"),
    ('CUBE', "Cube", "Six squares"),
    ('OCTA', "Octahedron", "Eight triangles"),
    ('DODECA', "Dodecahedron",
     "Twelve pentagons -- the classical spidroball, five spiral arms "
     "to a face"),
    ('ICOSA', "Icosahedron", "Twenty triangles"),
    ('TRUNC_ICOSA', "Truncated Icosahedron",
     "Twelve pentagons and twenty hexagons, so five- and six-armed "
     "nests mixed on one solid"),
    ('RHOMB_TRIACONTA', "Rhombic Triacontahedron",
     "Thirty rhombi -- the solid the spidroball was first shown as"),
]


def _cyclic(v):
    x, y, z = v
    return [(x, y, z), (y, z, x), (z, x, y)]


def _trunc_icosa_verts():
    """All even permutations of the standard coordinates."""
    V = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            V += _cyclic((0.0, s1 * 1.0, s2 * 3.0 * PHI))
            for s3 in (1, -1):
                V += _cyclic((s1 * 1.0, s2 * (2.0 + PHI), s3 * 2.0 * PHI))
                V += _cyclic((s1 * PHI, s2 * 2.0, s3 * (2.0 * PHI + 1.0)))
    return _dedupe(V)


def _icosidodeca_verts():
    W = []
    for s in (1, -1):
        W += _cyclic((0.0, 0.0, s * PHI))
    for s1 in (1, -1):
        for s2 in (1, -1):
            for s3 in (1, -1):
                W += _cyclic((s1 * 0.5, s2 * PHI / 2.0,
                              s3 * PHI * PHI / 2.0))
    return _dedupe(W)


def _rhombic_triaconta_verts():
    """The rhombic triacontahedron as the POLAR DUAL of the
    icosidodecahedron: one vertex per face plane, at n/d.

    Taking instead the convex hull of a dodecahedron and an icosahedron
    in their usual shared coordinates does NOT give the RT -- the
    icosahedral vertices sit too far out and raise pyramids, yielding
    24 triangles and 12 pentagons.  Only one relative scale makes the
    four vertices of each rhombus coplanar, and the dual construction
    lands on it exactly instead of approximating it.
    """
    W = np.asarray(_icosidodeca_verts(), float)
    out = []
    for f in _hull.hull_faces([tuple(v) for v in W]):
        p = W[list(f)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n = n / np.linalg.norm(n)
        d = float(n @ p[0])
        if d < 0.0:
            n, d = -n, -d
        out.append(tuple(n / d))
    return _dedupe(out)


def _dedupe(V, tol=1e-7):
    out = []
    for v in V:
        if not any(abs(v[0] - w[0]) < tol and abs(v[1] - w[1]) < tol
                   and abs(v[2] - w[2]) < tol for w in out):
            out.append(tuple(float(c) for c in v))
    return out


def seed_solid(kind):
    """(verts, faces) of the base solid, as python lists."""
    if kind == 'TRUNC_ICOSA':
        V = _trunc_icosa_verts()
        return V, _hull.hull_faces(V)
    if kind == 'RHOMB_TRIACONTA':
        V = _rhombic_triaconta_verts()
        return V, _hull.hull_faces(V)
    V, F = _seeds.seed_poly(kind)
    return [tuple(float(c) for c in v) for v in V], [list(f) for f in F]


def face_adjacency(faces):
    """Face index pairs sharing an edge."""
    owner = {}
    adj = [set() for _ in faces]
    for i, f in enumerate(faces):
        for k in range(len(f)):
            e = tuple(sorted((f[k], f[(k + 1) % len(f)])))
            if e in owner:
                j = owner[e]
                adj[i].add(j)
                adj[j].add(i)
            else:
                owner[e] = i
    return adj


def two_colour(faces):
    """Greedy 2-colouring of the face adjacency graph.  Returns
    (colours, ok); ok is False when an odd cycle makes the alternating
    chirality rule impossible -- which is exactly the case the Bridges
    paper says needs two different spidronised forms."""
    adj = face_adjacency(faces)
    col = [-1] * len(faces)
    ok = True
    for start in range(len(faces)):
        if col[start] != -1:
            continue
        col[start] = 0
        stack = [start]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if col[j] == -1:
                    col[j] = 1 - col[i]
                    stack.append(j)
                elif col[j] == col[i]:
                    ok = False
    return col, ok


def regular_relief(n):
    """The relief that makes an n-gon face's woven boundary a REGULAR
    skew 2n-gon: tan^2(pi/(2n)).

    The woven boundary pulls the face's vertices in by (1 - e) and
    pushes its edge midpoints out by (1 + e).  Seen from the face's
    axis the vertices sit at the pentagon circumradius r_c and the
    midpoints at its apothem r_a = r_c cos(pi/n), so one relief
    equalises the two: (1 - e) r_c = (1 + e) r_a, i.e.
    e = (1 - cos(pi/n)) / (1 + cos(pi/n)) = tan^2(pi/(2n)).  For the
    pentagon that is 0.105573, and the resulting skew decagon is
    node-for-node Nylander's boundary (see `nylander_dodeca_nest`).
    """
    return tan(pi / (2.0 * n)) ** 2


def nylander_dodeca_nest(rings=0):
    """Paul Nylander's dodeca-spidroball, reproduced exactly from his
    2010 AutoLisp/POV-Ray sources, for validation.

    His route to an interlocking ball differs from this module's: he
    lifts along the FACE NORMAL, like Relief Style: Per Face, but solves
    the lift `dz` from the solid's dihedral angle so that the raised
    points of neighbouring faces land on each other anyway.  The Woven
    style instead displaces radially, which makes neighbours meet by
    construction on any solid at any relief.  Both are correct; his is
    exact for the dodecahedron, this module's generalises.

    With `rings` = 0 the returned nodes are his ten boundary points.
    With `rings` > 0 they also contain his spiral: his recursion is
    Rtrans = Rz(dtheta) * scale applied in the face's local frame
    (about the face axis, toward the face-plane centre, so the +/-dz
    corrugation shrinks with each ring), giving (rings + 1) * 10 nodes
    ring by ring.  His published renders use rings = 8.

    Returns (nodes, face_frames, constants); constants carries his
    dtheta (36.53 deg -- one 36-deg node step plus 0.53) and scale 2/3.
    """
    n_arm = 5
    n = 2 * n_arm
    alpha = acos(-sqrt(5.0) / 5.0)          # dihedral, 116.565 deg
    a = sin(pi / n) ** 2
    b = tan(alpha / 2.0)
    dz = a * b / sqrt(a * b * b + 1.0)
    r = sqrt(1.0 - dz * dz / a)
    z0 = r * b * cos(pi / n_arm) + dz
    # his ring recursion constants; the sources flag both as "not sure
    # if this is right", but they are what rendered the reference
    # images: dtheta = acos((1 + 2 sqrt3 + 1/(dz^2/a - 1))/4) = 36.532
    # degrees, scale = 2/3.
    dtheta = acos((1.0 + 2.0 * sqrt(3.0)
                   + 1.0 / (dz * dz / a - 1.0)) / 4.0)
    scale = 2.0 / 3.0
    ring0 = np.array([[r * cos(i * pi / n_arm), r * sin(i * pi / n_arm),
                       (2 * (i % 2) - 1) * dz] for i in range(n)])
    c, s = cos(dtheta), sin(dtheta)
    Rtrans = np.array([[c, -s, 0.0], [s, c, 0.0],
                       [0.0, 0.0, 1.0]]) * scale
    loops = [ring0]
    for _ in range(rings):
        loops.append((Rtrans @ loops[-1].T).T)
    nodes = np.vstack(loops) + np.array([0.0, 0.0, z0])

    def rz(t):
        c, s = cos(t), sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def ry(t):
        c, s = cos(t), sin(t)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def euler(phi, th, psi):
        return rz(psi) @ ry(th) @ rz(phi)

    frames = [euler(0.0, 0.0, 0.0), euler(0.0, pi, 0.0)]
    th = 0.0
    for _ in range(5):
        frames.append(euler(0.2 * pi, alpha, th))
        frames.append(euler(0.0, pi - alpha, th + 0.2 * pi))
        th += 0.4 * pi
    return nodes, frames, dict(alpha=alpha, a=a, b=b, dz=dz, r=r, z0=z0,
                               dtheta=dtheta, scale=scale)


def arm_boundary_edges(faces, labels):
    """Edges where two DIFFERENT spiral arms meet.

    Shading the ball smooth blurs the whole face into one soft blob;
    what should stay crisp is the fold between one spiral arm and the
    next, which is where the real geometric discontinuity is.  Within an
    arm the surface is genuinely smooth as it winds inward, so those
    edges are left alone.  Faces of the seed solid do not share mesh
    vertices at all -- each nest is built independently -- so no edge
    ever spans two faces and the labels only ever separate arms.
    """
    owner = {}
    out = []
    for f, lab in zip(faces, labels):
        lab = tuple(lab)[:2]            # (face, arm) -- mesh adjacency
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            e = (a, b) if a < b else (b, a)
            if e in owner:
                if owner[e] != lab:
                    out.append(e)
            else:
                owner[e] = lab
    return out


def woven_boundary(face, V, relief):
    """The skew 2n-gon a face contributes to the WOVEN ball.

    A face's boundary is refined to its own vertices plus its edge
    midpoints, and those points are then displaced RADIALLY -- the
    solid's VERTICES PULLED IN, its EDGE MIDPOINTS PUSHED OUT -- rather
    than along the face's own normal.

    That direction is not a free choice: it is what Nylander's
    dodeca-spidroball does, and reversing it is what made this
    generator's early output a hedgehog instead of a spidroball.
    Raising the vertices puts a spike where three faces already meet at
    a point; raising the edge midpoints instead lifts a ridge along
    every edge, and the arms of the two faces sharing that edge sweep up
    over it together.  `_selftest` pins this against his own numbers:
    at relief 0.10557 the boundary reproduces his 50 points exactly.

    That distinction is the whole difference between a heap of separate
    spiky decorations and Nylander's interlocked spidroball.  A solid's
    vertex belongs to three faces and an edge midpoint to two; displace
    each along its own face's normal and the three copies fly apart in
    three different directions, leaving the surface torn.  Displace
    along the radius instead and every face computes the *same* position
    for a shared point, because the displacement depends only on the
    point and the centre -- so the nests meet exactly and the ball reads
    as one continuous corrugated surface.  The parities agree for free:
    on the refined boundary the solid's vertices all land on even
    indices and the midpoints on odd ones, in every face.
    """
    n = len(face)
    pts = []
    for k in range(n):
        a = V[face[k]]
        b = V[face[(k + 1) % n]]
        pts.append(tuple(a * (1.0 - relief)))
        pts.append(tuple(0.5 * (a + b) * (1.0 + relief)))
    return pts


COLOR_ITEMS = [
    ('PAIR', "Joined Arms",
     "Give the two arms that meet across an edge of the solid the same "
     "colour, so a petal spanning two faces reads as one shape rather "
     "than as two halves"),
    ('ARM', "Arm",
     "One colour per spiral arm, so each face reads as a pinwheel -- "
     "the colouring Nylander's renderings use"),
    ('RING', "Ring",
     "One colour per ring, banding each face from its rim to its "
     "centre and showing how fast the spiral shrinks"),
    ('FACE', "Face", "One colour per face of the underlying solid"),
    ('SHAPE', "Triangle Shape",
     "One colour per triangle shape, so the two triangles of each "
     "spiral step read differently"),
    ('CHIRALITY', "Chirality",
     "One colour per winding direction, showing which faces are wound "
     "which way"),
    ('UNIFORM', "Uniform", "A single material"),
]

NPAL = 12


def arm_keys(face, mpoly):
    """A global key per arm, shared by the arms that meet across an edge.

    An arm spans two consecutive points of the face's boundary.  When
    the boundary has been refined -- vertices and edge midpoints, so
    2n points for an n-sided face -- each arm runs from one of the
    solid's VERTICES to the midpoint of one of its EDGES, which is
    exactly a half-edge of the solid.  A half-edge belongs to both
    faces that share its edge, and both of them lay an arm along it, so
    keying on (vertex, edge) pairs those two arms and nothing else.

    Without relief the boundary is the raw face, one arm per edge, and
    the edge itself is the key -- the two faces sharing that edge each
    contribute one arm to it.

    Returns a list of `mpoly` hashable keys, or None if the boundary is
    neither of those two shapes.
    """
    n = len(face)

    def edge(i, j):
        a, b = int(face[i % n]), int(face[j % n])
        return (a, b) if a < b else (b, a)

    if mpoly == 2 * n:                  # refined: vertex, midpoint, ...
        keys = []
        for k in range(mpoly):
            i = k // 2
            if k % 2 == 0:              # vertex face[i] -> midpoint of i,i+1
                keys.append((int(face[i]), edge(i, i + 1)))
            else:                       # midpoint of i,i+1 -> vertex i+1
                keys.append((int(face[(i + 1) % n]), edge(i, i + 1)))
        return keys
    if mpoly == n:                      # unrefined: one arm per edge
        return [edge(k, k + 1) for k in range(n)]
    return None


def limb_index(arm_i, ring_i, mpoly, ch, twist_style):
    """Which spiral LIMB a triangle belongs to.

    With Twist Style Advance each ring is turned by a whole node step,
    so the triangle at (ring t, arm k) does not sit radially inward of
    (ring t-1, arm k) -- it sits one step around from it.  A limb that
    reads as one continuous band sweeping from the rim into the vortex
    therefore has an arm index that DRIFTS by one per ring, in the
    direction the rings are wound.  Colouring by `arm_i` alone slices
    that band into differently coloured pieces, one per ring, which is
    not what a viewer means by an arm.

    Verified against the geometry: grouping by this index holds each
    group to about a 23-degree azimuth band on a pentagon face, against
    345 degrees -- i.e. the whole face -- for the raw arm index.
    """
    if twist_style != 'ADVANCE':
        return arm_i % mpoly
    return (arm_i + ch * ring_i) % mpoly


def _material_index(color_by, face_i, ring_i, arm_i, kind, ch, pair_i=0,
                    npal=NPAL):
    if color_by == 'PAIR':
        return pair_i % npal
    if color_by == 'ARM':
        return arm_i % npal
    if color_by == 'RING':
        return ring_i % npal
    if color_by == 'FACE':
        return face_i % npal
    if color_by == 'SHAPE':
        return kind
    if color_by == 'CHIRALITY':
        return 0 if ch > 0 else 1
    return 0


def build(seed='DODECA', rings=8, scale=2.0 / 3.0, twist=0.0,
          relief=regular_relief(5), chirality='CW', open_center=False,
          relief_style='WOVEN', color_by='PAIR', twist_style='ADVANCE',
          colors=5):
    """Spidronise every face of the seed solid.

    With `twist_style` = 'ADVANCE' each ring turns by one node step of
    the face's boundary (2*pi/m for an m-point boundary) PLUS `twist`;
    on a woven boundary the step flips the corrugation parity, which is
    what weaves the arms (see the module header).  'FIXED' turns by
    exactly `twist`, the planar-rosette behaviour.  The defaults
    reproduce Nylander's dodeca-spidroball (his own excess over the
    step is 0.53 deg; zero is visually identical).
    """
    SV, SF = seed_solid(seed)
    A = np.asarray(SV, float)
    A = A / float(np.linalg.norm(A, axis=1).max())      # unit circumradius
    col, colour_ok = two_colour(SF)

    verts, faces, mats, labels = [], [], [], []
    pair_index = {}
    for fi, f in enumerate(SF):
        poly = [tuple(A[i]) for i in f]
        C = np.mean([A[i] for i in f], axis=0)
        N = sm._best_fit_normal(np.asarray(poly, float))
        if float(N @ (C - A.mean(axis=0))) < 0.0:
            N = -N                       # outward
        if relief > 0.0:
            if relief_style == 'WOVEN':
                poly = woven_boundary(f, A, relief)
            else:
                span = float(np.linalg.norm(
                    np.asarray(poly) - C, axis=1).mean())
                poly = sm.skew_lift(poly, relief * span, normal=N)
        if chirality == 'CW':
            ch = 1
        elif chirality == 'CCW':
            ch = -1
        else:
            ch = 1 if col[fi] == 0 else -1
        tw = twist
        if twist_style == 'ADVANCE':
            # one node step of THIS face's boundary, so mixed solids
            # (pentagons and hexagons on the truncated icosahedron)
            # each advance by their own step
            tw = 2.0 * pi / len(poly) + twist
        v, fc, mt = sm.spidronise(poly, scale, tw, rings,
                                  chirality=ch, centre=C, normal=N,
                                  cap=not open_center)
        o = len(verts)
        verts.extend(v)
        faces.extend([tuple(i + o for i in tri) for tri in fc])
        # `spidronise` lays the annulus down ring by ring, and within a
        # ring arm by arm, two triangles at a time; any cap triangles
        # follow.  That fixed order is what lets the ring and arm of
        # each triangle be recovered here without threading extra
        # bookkeeping through the kernel.
        mpoly = len(poly)
        akeys = arm_keys(f, mpoly)
        n_ann = rings * 2 * mpoly
        for ti in range(len(fc)):
            if ti < n_ann:
                ring_i = ti // (2 * mpoly)
                arm_i = (ti % (2 * mpoly)) // 2
                kind = ti % 2
            else:
                ring_i, arm_i, kind = rings, ti - n_ann, 2
            limb = limb_index(arm_i, ring_i, mpoly, ch, twist_style)
            if akeys is None:
                pair_i = limb
            else:
                key = akeys[limb % len(akeys)]
                if key not in pair_index:
                    pair_index[key] = len(pair_index)
                pair_i = pair_index[key]
            mats.append(_material_index(color_by, fi, ring_i, limb,
                                        kind, ch, pair_i, colors))
            # Creases follow MESH ADJACENCY, not the colour grouping.
            # The ribbon a triangle belongs to physically is the chain
            # of constant arm index -- that is what shares edges ring to
            # ring -- whereas `limb` groups by constant azimuth, which
            # under Advance drifts one step per ring.  Labelling the
            # creases with `limb` marked every ring-to-ring edge sharp
            # as well, creasing the ribbon along its own length and
            # undoing the smooth shading entirely.
            # (face, arm, limb): `arm` is mesh adjacency and drives the
            # creases; `limb` is the colour grouping.  Under Advance the
            # two differ by the per-ring drift.
            labels.append((fi, arm_i, limb))
    return verts, faces, mats, colour_ok, labels


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_spidron_ball_add(bpy.types.Operator, AddObjectHelper):
        """Add a spidronised solid: every face replaced by a spiral
        nest of triangles"""
        bl_idname = "mesh.spidron_ball_add"
        bl_label = "Spidroball"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Solid", items=SEED_ITEMS, default='DODECA',
            description="Base polyhedron whose faces are spidronised")
        rings: IntProperty(
            name="Rings", default=8, min=1, max=20,
            description="How many times the spiral step repeats on each "
                        "face. Nylander's ball uses eight")
        scale_step: FloatProperty(
            name="Ring Scale", default=2.0 / 3.0, min=0.3, max=0.97,
            description="How much each ring shrinks toward the centre "
                        "of its face. Two thirds is Nylander's value; "
                        "larger makes a shallower, denser vortex")
        twist_style: EnumProperty(
            name="Twist Style", default='ADVANCE',
            items=[('ADVANCE', "Advance",
                    "Each ring turns one node step of the face's "
                    "boundary plus Twist, so the corrugation phase "
                    "advances by a node per ring and the arms weave "
                    "over the folds -- the broad-petalled look of "
                    "Nylander's spidroball"),
                   ('FIXED', "Fixed",
                    "Each ring turns by exactly Twist, so the arms "
                    "stay on fixed nodes -- a shallower rosette "
                    "decoration")],
            description="What one ring's rotation is measured against")
        twist: FloatProperty(
            name="Twist", default=0.0,
            min=radians(-90.0), max=radians(90.0), subtype='ANGLE',
            description="Extra rotation per ring: on top of the node "
                        "step when the style is Advance (Nylander's "
                        "own excess is half a degree), the whole "
                        "rotation when it is Fixed")
        relief: FloatProperty(
            name="Relief", default=regular_relief(5), min=0.0, max=0.8,
            description="Lift alternate boundary points out of the face "
                        "plane before spiralling, so the nests stand "
                        "proud of the solid. Zero decorates each face "
                        "flat; 0.106 makes a pentagon's skew boundary "
                        "regular, which is exactly Nylander's ball")
        chirality: EnumProperty(
            name="Chirality", default='CW',
            items=[('CW', "Clockwise",
                    "Every face wound the same way, as Nylander's "
                    "renders are"),
                   ('CCW', "Anticlockwise",
                    "Every face wound the other way"),
                   ('ALTERNATE', "Alternate",
                    "Neighbouring faces wound oppositely, the pairing "
                    "an assembly of these solids requires")],
            description="Which way each face's spiral winds")
        relief_style: EnumProperty(
            name="Relief Style", default='WOVEN',
            items=[('WOVEN', "Woven",
                    "Displace shared boundary points radially, so "
                    "neighbouring faces meet exactly and the ball reads "
                    "as one interlocked surface"),
                   ('FACE', "Per Face",
                    "Displace each face's boundary along its own "
                    "normal, leaving the faces as separate raised "
                    "decorations")],
            description="How the relief lifts each face's boundary out "
                        "of the solid")
        color_by: EnumProperty(
            name="Color", items=COLOR_ITEMS, default='PAIR',
            description="How materials are assigned across the "
                        "spidronised faces")
        smooth: BoolProperty(
            name="Smooth Shading", default=True,
            description="Shade the spiral surfaces smooth instead of "
                        "faceted")
        sharp_edges: BoolProperty(
            name="Sharp Creases", default=True,
            description="Keep the fold between neighbouring spiral "
                        "arms crisp under smooth shading, and creased "
                        "under a Subdivision Surface, while the "
                        "surface stays smooth along each arm")
        colors: IntProperty(
            name="Colors", default=5, min=1, max=12,
            description="How many materials the palette is cut down to. "
                        "Five is enough to keep neighbouring limbs "
                        "apart on most solids without the ball turning "
                        "into a colour chart")
        open_center: BoolProperty(
            name="Open Centres", default=False,
            description="Leave the small hole at the centre of each "
                        "face open instead of closing it")

        def execute(self, context):
            V, F, M, colour_ok, labels = build(
                self.seed, int(self.rings), float(self.scale_step),
                float(self.twist), float(self.relief), self.chirality,
                self.open_center, self.relief_style,
                self.color_by, self.twist_style, int(self.colors))
            if not F:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}
            obj = pc.build_object(context, "Spidroball", V, F, M,
                                  span=2.0, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}
            me = obj.data
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                me.update()
            ncrease = 0
            if self.sharp_edges:
                ncrease = _sc.mark_sharp(me, arm_boundary_edges(F, labels))
                if ncrease == 0:
                    # marking nothing looks exactly like success in a
                    # render until you go hunting for the crease
                    self.report({'WARNING'},
                                "no arm boundaries found to crease")
            warn = ("" if colour_ok or self.chirality != 'ALTERNATE'
                    else "  (face graph has an odd cycle: alternating "
                         "chirality is impossible on this solid)")
            self.report({'INFO'}, "%s  V=%d F=%d  creases=%d%s"
                        % (self.seed.title(), len(me.vertices),
                           len(me.polygons), ncrease, warn))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for p in ('seed', 'rings', 'scale_step', 'twist_style',
                      'twist', 'relief', 'relief_style', 'chirality',
                      'color_by', 'open_center', 'smooth',
                      'sharp_edges'):
                lay.prop(self, p)
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.spidron_ball_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_spidron_ball_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_spidron_ball_add)


# --------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------

def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-52s %s %s" % (name, "OK" if cond else "BAD", extra))

    print("spidron_ball: seed solids")
    want = {'TETRA': (4, 4), 'CUBE': (8, 6), 'OCTA': (6, 8),
            'DODECA': (20, 12), 'ICOSA': (12, 20),
            'TRUNC_ICOSA': (60, 32), 'RHOMB_TRIACONTA': (32, 30)}
    for kind, (nv, nf) in want.items():
        V, F = seed_solid(kind)
        chk("%-16s V=%d F=%d" % (kind, len(V), len(F)),
            len(V) == nv and len(F) == nf, "want %d/%d" % (nv, nf))
        # Euler characteristic of a sphere
        ne = sum(len(f) for f in F) // 2
        chk("%-16s Euler = 2" % kind, len(V) - ne + len(F) == 2,
            "V-E+F=%d" % (len(V) - ne + len(F)))

    print("spidron_ball: two-colouring")
    # A solid's faces are 2-colourable exactly when its DUAL's edge
    # graph is bipartite -- and for every seed here the dual has
    # triangular or pentagonal faces, hence odd cycles.  So perfect
    # alternation is impossible on all of them, which is the same fact
    # the Bridges paper states as "a polyhedron with an odd number of
    # identical faces needs two different spidronised forms".  The
    # colouring is therefore a best effort and the operator says so.
    for kind in want:
        _, F = seed_solid(kind)
        col, good = two_colour(F)
        adj = face_adjacency(F)
        clash = sum(1 for i in range(len(F)) for j in adj[i]
                    if col[i] == col[j]) // 2
        chk("%-16s colouring is consistent with its verdict" % kind,
            good == (clash == 0), "%d same-chirality joins" % clash)
    _, F = seed_solid('TETRA')
    _, good = two_colour(F)
    chk("tetrahedron reports the odd-cycle obstruction", not good,
        "every pair of its 4 faces is adjacent")
    # A solid's faces 2-colour exactly when its dual's edge graph is
    # bipartite.  Among these seeds only the octahedron qualifies -- its
    # dual is the cube, whose faces are quadrilaterals; every other dual
    # here has triangular or pentagonal faces and so an odd cycle.
    good_seeds = set(k for k in want if two_colour(seed_solid(k)[1])[1])
    chk("octahedron alone admits perfect alternation",
        good_seeds == {'OCTA'}, "2-colourable: %s"
        % (sorted(good_seeds) or "none"))

    print("spidron_ball: build")
    for kind in want:
        V, F, M, cok, _lb = build(kind, rings=4, scale=0.62,
                             twist=radians(30.0))
        chk("%-16s builds" % kind, len(F) > 0,
            "V=%d F=%d" % (len(V), len(F)))
        A = np.array(V)
        bad = 0
        for f in F:
            p = A[list(f)]
            a = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
            if a < 1e-14:
                bad += 1
        chk("%-16s no degenerate faces" % kind, bad == 0, "%d bad" % bad)

    V, F, M, _, _ = build('DODECA', rings=8, scale=0.62,
                       twist=radians(30.0), relief=0.35)
    Vf = _fit.fit_cube(V, 2.0)
    A = np.array(Vf)
    ext = A.max(axis=0) - A.min(axis=0)
    ctr = 0.5 * (A.max(axis=0) + A.min(axis=0))
    chk("fits the 2 m cube", abs(ext.max() - 2.0) < 1e-9
        and np.abs(ctr).max() < 1e-9, "extent %.6f" % ext.max())
    chk("relief keeps the solid three-dimensional",
        min(ext) > 0.5 * max(ext), "aspect %.3f" % (min(ext) / max(ext)))

    Vf0, _, _, _, _ = build('DODECA', rings=6, scale=0.62,
                         twist=radians(30.0), relief=0.0)
    Vf1, _, _, _, _ = build('DODECA', rings=6, scale=0.62,
                         twist=radians(30.0), relief=0.4)
    r0 = np.linalg.norm(np.array(Vf0), axis=1)
    r1 = np.linalg.norm(np.array(Vf1), axis=1)
    chk("relief pushes vertices off the face planes",
        r1.max() > r0.max() + 1e-6,
        "rmax %.4f -> %.4f" % (r0.max(), r1.max()))

    print("spidron_ball: joined-arm colouring")
    # Every arm key must be claimed by exactly two arms, and those two
    # must sit on DIFFERENT faces -- that is what makes a petal spanning
    # a fold read as one shape.  The count follows: one key per
    # half-edge with relief (2E), one per edge without it (E).
    for kind in want:
        _SV, SF2 = seed_solid(kind)
        E = sum(len(f) for f in SF2) // 2
        for refined, target in ((True, 2 * E), (False, E)):
            owners = {}
            for fi2, f2 in enumerate(SF2):
                mp = 2 * len(f2) if refined else len(f2)
                for k in arm_keys(f2, mp):
                    owners.setdefault(k, []).append(fi2)
            ok_n = len(owners) == target
            ok_pair = all(len(v) == 2 and v[0] != v[1]
                          for v in owners.values())
            chk("%-16s %s arm keys pair across faces"
                % (kind, "relief " if refined else "flat   "),
                ok_n and ok_pair,
                "%d keys, want %d" % (len(owners), target))
    # and the paired arms really do get the same material.  `build`
    # emits ring-major (all arms of ring 0, then ring 1, ...), so read
    # the (face, arm) labels it returns rather than trying to stride
    # through the material list.
    _V4, _F4, M4, _o4, lb4 = build('DODECA', rings=4, color_by='PAIR')
    _SVd, SFd = seed_solid('DODECA')
    by_key = {}
    for (fi2, _ai2, li2), m in zip(lb4, M4):
        ks = arm_keys(SFd[fi2], 2 * len(SFd[fi2]))
        by_key.setdefault(ks[li2 % len(ks)], set()).add(m)
    mixed = sum(1 for v in by_key.values() if len(v) != 1)
    chk("paired arms share one material", mixed == 0,
        "%d of %d keys mixed" % (mixed, len(by_key)))
    # and the pairing is not vacuous: neighbouring arms within a face
    # must still differ, or "same colour" would just mean "one colour"
    per_face = {}
    for (fi2, _ai2, li2), m in zip(lb4, M4):
        per_face.setdefault(fi2, {})[li2] = m
    varied = sum(1 for d in per_face.values() if len(set(d.values())) > 2)
    chk("faces still carry several colours", varied == len(per_face),
        "%d of %d faces" % (varied, len(per_face)))

    print("spidron_ball: limbs follow the spiral")
    # The claim that makes Joined Arms mean what a viewer expects: a
    # limb index holds a group of triangles inside a narrow azimuth
    # band, sweeping from rim to vortex, where the raw arm index
    # smears the same group right round the face.
    SVl, SFl = seed_solid('DODECA')
    Al = np.asarray(SVl, float)
    Al = Al / float(np.linalg.norm(Al, axis=1).max())
    fl = SFl[0]
    polyl = woven_boundary(fl, Al, regular_relief(5))
    Cl = np.mean([Al[i] for i in fl], axis=0)
    Nl = sm._best_fit_normal(np.asarray(polyl, float))
    if float(Nl @ (Cl - Al.mean(axis=0))) < 0.0:
        Nl = -Nl
    ml, rgs = len(polyl), 6
    Vl, Fl, _Ml = sm.spidronise(polyl, 2.0 / 3.0, 2.0 * pi / ml, rgs,
                                chirality=1, centre=Cl, normal=Nl,
                                cap=False)
    Vl = np.asarray(Vl, float)
    e1 = np.asarray(polyl[0], float) - Cl
    e1 = e1 - float(e1 @ Nl) * Nl
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(Nl, e1)

    def _spread(by_limb):
        grp = {}
        for ti in range(len(Fl)):
            rt, ar = ti // (2 * ml), (ti % (2 * ml)) // 2
            d = Vl[list(Fl[ti])].mean(axis=0) - Cl
            a = degrees(atan2(float(d @ e2), float(d @ e1))) % 360.0
            key = limb_index(ar, rt, ml, 1, 'ADVANCE') if by_limb else ar
            grp.setdefault(key, []).append(a)
        out = []
        for v in grp.values():
            v = np.asarray(v)
            v = (v - v[0] + 180.0) % 360.0 - 180.0
            out.append(float(v.max() - v.min()))
        return float(np.mean(out))
    s_limb, s_arm = _spread(True), _spread(False)
    chk("limb index holds one azimuth band",
        s_limb < 40.0 and s_arm > 300.0,
        "limb %.0f deg vs raw arm %.0f deg" % (s_limb, s_arm))
    # and under FIXED there is no drift to compensate
    chk("no drift compensation under Twist Style Fixed",
        all(limb_index(a, r, ml, 1, 'FIXED') == a
            for a in range(ml) for r in range(rgs)))

    print("spidron_ball: colour modes and creases")
    for cb, lo, hi in (('PAIR', 5, 5), ('ARM', 5, 5), ('RING', 5, 5),
                       ('FACE', 5, 5), ('SHAPE', 2, 3),
                       ('CHIRALITY', 1, 2), ('UNIFORM', 1, 1)):
        _, _, M, _, _ = build('DODECA', rings=6, color_by=cb)
        used = len(set(M))
        chk("colour by %-9s uses %2d materials" % (cb, used),
            lo <= used <= hi and max(M) < NPAL)
    for n in (1, 3, 5, 12):
        _, _, M, _, _ = build('DODECA', rings=6, color_by='PAIR', colors=n)
        chk("colors=%-2d honoured" % n, len(set(M)) == n,
            "%d used" % len(set(M)))
    V, F, M, _, labels = build('DODECA', rings=6)
    ce = arm_boundary_edges(F, labels)
    chk("arm boundaries found to crease", len(ce) > 0, "%d edges" % len(ce))
    # every creased edge must separate two different arms, and no edge
    # inside a single arm may be creased
    owner = {}
    for f, lab in zip(F, labels):
        lab = tuple(lab)[:2]
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            owner.setdefault((a, b) if a < b else (b, a), []).append(lab)
    bad = sum(1 for e in ce if len(set(owner[e])) < 2)
    chk("every crease separates two arms", bad == 0, "%d bad" % bad)
    inner = sum(1 for e, l in owner.items()
                if len(l) == 2 and len(set(l)) == 2)
    chk("creases are exactly the arm-to-arm joins",
        len(set(ce)) == inner, "%d vs %d" % (len(set(ce)), inner))

    print("spidron_ball: Nylander's dodeca-spidroball (literature check)")
    nodes, frames, K = nylander_dodeca_nest()
    chk("dihedral = arccos(-1/sqrt5) = 116.565 deg",
        abs(degrees(K['alpha']) - 116.56505117707799) < 1e-9)
    chk("b = tan(alpha/2) is the golden ratio",
        abs(K['b'] - PHI) < 1e-12, "%.12f" % K['b'])
    chk("his constants reproduce", abs(K['dz'] - 0.138197) < 1e-6
        and abs(K['r'] - 0.894427) < 1e-6
        and abs(K['z0'] - 1.309017) < 1e-6,
        "dz=%.6f r=%.6f z0=%.6f" % (K['dz'], K['r'], K['z0']))
    pts = [(M @ nodes.T).T for M in frames]
    chk("his construction places 12 faces", len(pts) == 12)
    shared = []
    for j in range(1, len(pts)):
        d = np.linalg.norm(pts[0][:, None, :] - pts[j][None, :, :], axis=2)
        shared.append(float(d.min()))
    chk("his neighbours share boundary points exactly",
        min(shared) < 1e-12, "closest %.1e" % min(shared))
    # every raised point of his nest is shared with the two faces that
    # meet there -- the same interlocking the Woven style guarantees by
    # construction, reached by solving dz from the dihedral instead.
    allp = np.vstack(pts)
    hits = 0
    for p in pts[0]:
        d = np.linalg.norm(allp - p, axis=1)
        hits += int((d < 1e-9).sum() > 1)
    chk("every one of his boundary points is shared", hits == len(pts[0]),
        "%d of %d" % (hits, len(pts[0])))
    # THE reconciliation: his boundary is the Woven boundary.  His 50
    # distinct points split 20 + 30 -- one per dodecahedron vertex and
    # one per edge -- and the vertices sit at 0.894 of the circumradius
    # while the midpoints sit at 1.106 of the midradius.  Both are
    # 1 -/+ 0.10557, so it is a symmetric radial relief with the
    # VERTICES PULLED IN.  Reproduce it exactly.
    uni = []
    for q in np.vstack(pts):
        if not any(np.linalg.norm(q - w) < 1e-9 for w in uni):
            uni.append(q)
    rh = np.sort(np.linalg.norm(np.array(uni), axis=1))
    chk("his 50 points are 20 vertices + 30 edge midpoints",
        len(uni) == 50 and (np.abs(rh - rh[0]) < 1e-6).sum() == 20
        and (np.abs(rh - rh[-1]) < 1e-6).sum() == 30,
        "%d points" % len(uni))
    Aw = np.asarray(seed_solid('DODECA')[0], float)
    Aw = Aw / float(np.linalg.norm(Aw, axis=1).max()) * 1.6472782070926637
    mine = []
    for f2 in seed_solid('DODECA')[1]:
        for q in woven_boundary(f2, Aw, 0.1055728090000841):
            if not any(np.linalg.norm(np.asarray(q) - np.asarray(w)) < 1e-9
                       for w in mine):
                mine.append(q)
    rm = np.sort(np.linalg.norm(np.array(mine), axis=1))
    chk("Woven at relief 0.10557 reproduces his boundary",
        len(mine) == 50 and np.allclose(rh, rm, atol=1e-6),
        "%d points, radii %.6f/%.6f" % (len(mine), rm[0], rm[-1]))
    # and that relief is not a fitted number: it is the one value that
    # makes the skew decagon REGULAR (all boundary points equidistant
    # from the face axis), tan^2(pi/10)
    chk("his relief is derived: tan^2(pi/10) = 0.105573",
        abs(regular_relief(5) - 0.1055728090000841) < 1e-13
        and abs(regular_relief(5)
                - (1.0 - cos(pi / 5.0)) / (1.0 + cos(pi / 5.0))) < 1e-15,
        "%.12f" % regular_relief(5))
    A5 = np.asarray(seed_solid('DODECA')[0], float)
    A5 = A5 / float(np.linalg.norm(A5, axis=1).max())
    f5 = seed_solid('DODECA')[1][0]
    C5 = A5[list(f5)].mean(axis=0)
    N5 = sm._best_fit_normal(A5[list(f5)])
    if float(N5 @ C5) < 0.0:
        N5 = -N5
    dax = []
    for q in woven_boundary(f5, A5, regular_relief(5)):
        w = np.asarray(q) - C5
        w = w - float(w @ N5) * N5
        dax.append(float(np.linalg.norm(w)))
    chk("regular relief equalises distance from the face axis",
        max(dax) - min(dax) < 1e-12,
        "spread %.1e" % (max(dax) - min(dax)))
    # HIS SPIRAL, not just his boundary.  His ring recursion
    # Rz(dtheta) * (2/3) has dtheta = one 36-degree node step of the
    # decagon plus 0.53 -- the full step is what flips the corrugation
    # parity each ring and weaves the arms; it is this module's
    # ADVANCE twist style with a 0.53-degree excess.
    chk("his dtheta = 36.532 deg = node step + 0.53",
        abs(degrees(K['dtheta']) - 36.531915275721104) < 1e-9
        and abs(K['scale'] - 2.0 / 3.0) < 1e-15,
        "dtheta %.6f deg" % degrees(K['dtheta']))
    # THE reproduction of his published ball: ADVANCE + his excess +
    # his scale + the derived relief + uniform chirality rebuilds his
    # spiral NODE FOR NODE (face 0 in its local face frame, all nine
    # rings), and by face congruence the whole ball.
    his90 = nylander_dodeca_nest(rings=8)[0]
    his90 = his90 - np.array([0.0, 0.0, K['z0']])
    his90 = his90 / np.linalg.norm(his90[0][:2])
    V9, _, _, _, _ = build('DODECA', rings=8, scale=2.0 / 3.0,
                           twist=float(K['dtheta'] - pi / 5.0),
                           relief=regular_relief(5), chirality='CW',
                           open_center=True, twist_style='ADVANCE')
    blk = np.asarray(V9[:90], float)
    P0 = blk[0] - C5
    e3 = N5
    e1 = P0 - float(P0 @ e3) * e3
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    loc = (blk - C5) @ np.stack([e1, e2, e3], axis=1)
    loc = loc / np.linalg.norm(loc[0][:2])
    err9 = float(np.abs(loc - his90).max())
    chk("ADVANCE rebuilds his spiral node for node", err9 < 1e-12,
        "max dev %.1e" % err9)
    # the two twist styles are the same kernel: ADVANCE(t) is exactly
    # FIXED(t + 2*pi/m) on an m-point boundary
    Va2, _, _, _, _ = build('DODECA', rings=3, scale=0.7, twist=0.3,
                            twist_style='ADVANCE')
    Vf2, _, _, _, _ = build('DODECA', rings=3, scale=0.7,
                            twist=0.3 + 2.0 * pi / 10.0,
                            twist_style='FIXED')
    chk("ADVANCE(t) == FIXED(t + step)",
        float(np.abs(np.asarray(Va2) - np.asarray(Vf2)).max()) < 1e-12)
    # and the defaults are that regime: they must weave, not rosette
    chk("defaults are the Nylander regime",
        build.__defaults__[1] == 8 and abs(build.__defaults__[2]
                                           - 2.0 / 3.0) < 1e-15
        and abs(build.__defaults__[4] - regular_relief(5)) < 1e-15
        and build.__defaults__[5] == 'CW'
        and build.__defaults__[9] == 'ADVANCE')

    print("spidron_ball: relief styles")
    # THE invariant that separates the interlocked ball from a heap of
    # spikes: neighbouring faces must agree exactly on every boundary
    # point they share.
    for style, want_ok in (('WOVEN', True), ('FACE', False)):
        for kind in ('DODECA', 'CUBE', 'ICOSA'):
            SV, SF = seed_solid(kind)
            A = np.asarray(SV, float)
            A = A / float(np.linalg.norm(A, axis=1).max())
            bnds = []
            for f in SF:
                if style == 'WOVEN':
                    bnds.append(np.asarray(woven_boundary(f, A, 0.22),
                                           float))
                else:
                    poly = [tuple(A[i]) for i in f]
                    C = np.mean([A[i] for i in f], axis=0)
                    N = sm._best_fit_normal(np.asarray(poly, float))
                    if float(N @ (C - A.mean(axis=0))) < 0.0:
                        N = -N
                    span = float(np.linalg.norm(
                        np.asarray(poly) - C, axis=1).mean())
                    bnds.append(np.asarray(
                        sm.skew_lift(poly, 0.22 * span, normal=N), float))
            adj = face_adjacency(SF)
            worst = 0.0
            for i in range(len(SF)):
                for j in adj[i]:
                    if j <= i:
                        continue
                    dm = np.linalg.norm(
                        bnds[i][:, None, :] - bnds[j][None, :, :], axis=2)
                    # two faces share an edge: 3 boundary points
                    # (two solid vertices and their midpoint)
                    worst = max(worst, float(np.sort(dm.min(axis=1))[2]))
            got = worst < 1e-9
            chk("%-5s %-7s neighbours agree on shared points"
                % (style, kind), got == want_ok, "%.1e" % worst)

    Vcw, _, _, _, _ = build('CUBE', rings=4, chirality='CW')
    Vcc, _, _, _, _ = build('CUBE', rings=4, chirality='CCW')
    chk("chirality changes the geometry",
        np.abs(np.array(Vcw) - np.array(Vcc)).max() > 1e-6)
    # ALTERNATE cannot oppose EVERY neighbour on these solids (above),
    # but it must do far better than a uniform winding, which opposes
    # none at all.
    _, SF = seed_solid('DODECA')
    col, _ = two_colour(SF)
    adj = face_adjacency(SF)
    joins = sum(len(a) for a in adj) // 2
    clash = sum(1 for i in range(len(SF)) for j in adj[i]
                if col[i] == col[j]) // 2
    chk("alternate beats a uniform winding on the dodecahedron",
        0 < clash < joins // 2, "%d of %d joins share chirality "
        "(uniform would share all %d)" % (clash, joins, joins))
    colo, oko = two_colour(seed_solid('OCTA')[1])
    adjo = face_adjacency(seed_solid('OCTA')[1])
    chk("alternate opposes EVERY neighbour on the octahedron",
        oko and all(colo[i] != colo[j] for i in range(len(colo))
                    for j in adjo[i]))

    print("RESULT:", "OK" if ok else "BAD")
    return ok
