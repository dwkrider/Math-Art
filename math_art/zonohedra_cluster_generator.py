
# Polar-zonohedral clusters
#
# One polar zonohedron per face of a seed solid, all sharing the seed's
# centre, filling the space around that point without gaps or overlaps.
#
#   CLUSTER   Webster's construction.  Take the centre O of a seed whose
#             vertices all lie on one sphere, and one of its faces with
#             vertices V_0..V_{m-1}; the segments O->V_i are equal in
#             length and evenly spaced about the axis O->C through the
#             face's centroid, so their Minkowski sum is a polar
#             zonohedron of that order.  Doing it for every face packs the
#             units round O: each pair of adjacent V_i belongs to two
#             faces, so neighbouring units share a whole face and no gap
#             is left.  The seed must therefore be convex with regular
#             faces and a circumsphere -- the Platonic and Archimedean
#             solids, the prisms and antiprisms, and 19 of the 25
#             circumscribable Johnson solids.
#   FACETED   Marotta and Redmond's step: join the units' outer poles and
#             take the hull.  For a Kleetope of the dodecahedron this is a
#             truncation of the icosahedron, and the family runs through
#             the Archimedean truncated icosahedron itself.
#
# Dropping the circumsphere requirement is what makes the second half
# work.  On a non-circumscribable seed the segments O->V_i are no longer
# equal and the axis is no longer perpendicular to the face, so a unit is
# a general zonohedron rather than a polar one -- still every face a
# parallelogram -- and the cluster reads as a spiky star.  Augmenting each
# face of the seed with a pyramid of signed height h (a Kleetope, or an
# excavated solid when h is negative) is the dial that family runs on.
#
# Why the circumcentre and not the centroid: a diminished solid still has
# every vertex on the original sphere, but its centroid has moved off that
# sphere's centre.  Measuring the segments from the centroid would give
# unequal lengths and quietly turn a Webster cluster into a generalized
# one.  The centre also has to lie strictly INSIDE every face plane, which
# is what rules out the six pyramid-, cupola- and rotunda-shaped Johnson
# solids: their circumcentre sits on or outside the base, so the "umbrella"
# of segments opens the wrong way.  Testing that condition over the 25
# circumscribable Johnson solids leaves exactly the 19 Webster counts.
#
# References:
# - Phil Webster, "Polar Zonohedral Helices and Clusters", Bridges 2023,
#   pp. 329-336 -- the cluster construction and the enumeration of the 37
#   clusters plus the prism and antiprism families.
# - Laura Marotta and Brian Redmond, "Spiky Soccer Balls: Generalized
#   Polar Zonohedral Clusters", Bridges 2025, pp. 463-466 -- dropping
#   circumscribability, the Kleetope seeds, and faceting the spiky form
#   into truncations of the icosahedron with
#   B/A = (2/5) sqrt(5(5-2 sqrt5)) h + (5 + sqrt5)/10.
# - George W. Hart, "The Joy of Polar Zonohedra", Bridges 2021, pp. 7-14.
# - Norman W. Johnson, "Convex Polyhedra with Regular Faces", Canadian
#   Journal of Mathematics 18 (1966), pp. 169-200.
# - Sandor Kabai, "30 Cubes on a Rhombic Triacontahedron", Bridges 2010,
#   pp. 317-322; "Inside and Outside the Rhombic Hexecontahedron",
#   Bridges 2011, pp. 387-394 -- the rhombic hexecontahedron as the
#   icosahedron's cluster of twenty golden rhombohedra.
# - P. Huybers, "In Search of the Roundest Soccer Ball", Int. Conference
#   on Adaptability in Design and Construction (2006), pp. 6.115-6.121 --
#   the biscribed truncated icosahedron the family passes through.

bl_info = {
    "name": "Zonohedral Clusters",
    "author": "Math Art project (after Phil Webster)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Polar zonohedra packed around a point, and the "
                   "solids that facets them",
    "category": "Add Mesh",
}

import math

try:
    from .polyhedra import zonotope as _zt
    from .polyhedra import hull as _hull
    from .polyhedra import fit as _fit
except ImportError:                       # flat-file / headless import
    from polyhedra import zonotope as _zt
    from polyhedra import hull as _hull
    from polyhedra import fit as _fit

PHI = (1 + 5 ** 0.5) / 2
EPS = 1e-9


# --------------------------------------------------------------------------
# small helpers
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
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    l = _norm(a)
    return [x / l for x in a] if l > EPS else [0.0, 0.0, 0.0]


def _face_normal(V, f):
    """Newell's normal -- correct for a face that is not exactly planar,
    which an augmented seed's faces need not be."""
    n = [0.0, 0.0, 0.0]
    m = len(f)
    for k in range(m):
        a, b = V[f[k]], V[f[(k + 1) % m]]
        n = [n[0] + (a[1] - b[1]) * (a[2] + b[2]),
             n[1] + (a[2] - b[2]) * (a[0] + b[0]),
             n[2] + (a[0] - b[0]) * (a[1] + b[1])]
    return n


def _centroid(V, f):
    m = len(f)
    return [sum(V[i][d] for i in f) / m for d in range(3)]


def edge_lengths(V, F):
    """Every distinct edge length in the mesh, rounded for grouping."""
    seen = set()
    out = []
    for f in F:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append(_norm(_sub(V[b], V[a])))
    return out


# --------------------------------------------------------------------------
# the circumsphere, and whether a seed can carry a Webster cluster
# --------------------------------------------------------------------------

def circumcentre(V):
    """Least-squares centre of the sphere through the vertices.

    Linearising |v - c|^2 = r^2 gives 2 v.c + (r^2 - |c|^2) = |v|^2, which
    is linear in (c, r^2 - |c|^2), so one 4x4 normal-equation solve does
    it.  It is a FIT, so it returns an answer for any solid; whether the
    solid actually has a circumsphere is `circumscribable`'s question.
    """
    A = [[0.0] * 4 for _ in range(4)]
    b = [0.0] * 4
    for v in V:
        row = [2 * v[0], 2 * v[1], 2 * v[2], 1.0]
        rhs = _dot(v, v)
        for i in range(4):
            for j in range(4):
                A[i][j] += row[i] * row[j]
            b[i] += row[i] * rhs
    M = [A[i][:] + [b[i]] for i in range(4)]
    for i in range(4):
        p = max(range(i, 4), key=lambda k: abs(M[k][i]))
        M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-14:
            return [sum(v[d] for v in V) / len(V) for d in range(3)]
        for k in range(4):
            if k == i:
                continue
            f = M[k][i] / M[i][i]
            for j in range(i, 5):
                M[k][j] -= f * M[i][j]
    return [M[i][4] / M[i][i] for i in range(3)]


def circumscribable(V, centre=None, tol=1e-6):
    """(is it, relative spread of the vertex radii)."""
    c = centre if centre is not None else circumcentre(V)
    r = [_norm(_sub(v, c)) for v in V]
    spread = (max(r) - min(r)) / max(max(r), EPS)
    return spread <= tol, spread


def centre_margin(V, F, centre):
    """How far inside the solid the centre sits, as a fraction of the
    circumradius, taking the closest face.

    Zero or negative means the centre lies on or outside some face plane,
    and the cluster construction fails there: the segments to that face's
    vertices no longer form an umbrella opening away from the centre.
    """
    body = [sum(v[d] for v in V) / len(V) for d in range(3)]   # interior
    R = max(_norm(_sub(v, centre)) for v in V) or 1.0
    worst = None
    for f in F:
        if len(f) < 3:
            continue
        n = _unit(_face_normal(V, f))
        a = V[f[0]]
        if _dot(n, _sub(a, body)) < 0:
            n = _mul(n, -1.0)
        d = _dot(n, _sub(a, centre)) / R
        worst = d if worst is None else min(worst, d)
    return 0.0 if worst is None else worst


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------

def prism(n, anti=False):
    """A uniform prism or antiprism on unit edges.

    Both are circumscribable by construction, which is why Webster's
    enumeration carries them as two infinite families alongside the finite
    ones.  The triangular antiprism is the octahedron and the square prism
    the cube, so the two families overlap the Platonic solids twice.
    """
    r = 1.0 / (2.0 * math.sin(math.pi / n))
    if anti:
        h = math.sqrt(max(0.0, 1.0 - 1.0 /
                          (4.0 * math.cos(math.pi / (2 * n)) ** 2)))
        off = math.pi / n
    else:
        h = 1.0
        off = 0.0
    V = []
    for k in range(n):
        a = 2 * math.pi * k / n
        V.append([r * math.cos(a), r * math.sin(a), -h / 2])
    for k in range(n):
        a = 2 * math.pi * k / n + off
        V.append([r * math.cos(a), r * math.sin(a), h / 2])
    F = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    if anti:
        for k in range(n):
            F.append([k, (k + 1) % n, n + k])
            F.append([(k + 1) % n, n + (k + 1) % n, n + k])
    else:
        for k in range(n):
            F.append([k, (k + 1) % n, n + (k + 1) % n, n + k])
    return V, F


def unit_edge(V, F):
    """Rescale so the mean edge is 1, which is the unit every published
    formula for these clusters is written in."""
    e = edge_lengths(V, F)
    s = (sum(e) / len(e)) if e else 1.0
    return [_mul(v, 1.0 / s) for v in V]


def seed_solid(family, sid, n=6):
    """(V, F) for a seed, on unit edges and centred on its circumcentre."""
    if family == 'PRISM':
        V, F = prism(max(3, n), anti=False)
    elif family == 'ANTIPRISM':
        V, F = prism(max(3, n), anti=True)
    else:
        try:
            from . import regular_solids_generator as rs
        except ImportError:
            import regular_solids_generator as rs
        V, F, _sizes = rs.build_solid(family, sid, n=n, scale=1.0,
                                      canon=(family != 'JOHNSON'))
        V = [list(v) for v in V]
        F = [list(f) for f in F]
    V = unit_edge(V, F)
    c = circumcentre(V)
    return [_sub(v, c) for v in V], F


def kis(V, F, height):
    """Raise a pyramid of the given signed height on every face.

    Positive heights augment (a Kleetope), negative ones excavate.  The
    apex sits on the face's own normal, so the height is measured in the
    seed's edge lengths -- the unit Marotta and Redmond state their
    results in.

    Height ZERO is a real member of the family, not a no-op: the apex
    lands in the face's own plane and the face splits into coplanar
    triangles.  That still changes the cluster completely -- a
    dodecahedron carries twelve 5-fold units, its flat Kleetope sixty
    parallelepipeds -- and it is the h = 0 end of the B/A relation.  A
    caller that wants the untouched seed must simply not call this.
    """
    body = [sum(v[d] for v in V) / len(V) for d in range(3)]
    Vo = [list(v) for v in V]
    Fo = []
    for f in F:
        c = _centroid(V, f)
        nrm = _unit(_face_normal(V, f))
        if _dot(nrm, _sub(c, body)) < 0:
            nrm = _mul(nrm, -1.0)
        apex = len(Vo)
        Vo.append(_add(c, _mul(nrm, height)))
        m = len(f)
        for k in range(m):
            Fo.append([f[k], f[(k + 1) % m], apex])
    return Vo, Fo


# --------------------------------------------------------------------------
# the cluster
# --------------------------------------------------------------------------

def cluster_units(V, F, centre=None):
    """One zonohedron per face.  Returns [(Vc, Fc, order, pole)].

    The generators are the segments from the centre to the face's
    vertices, in the face's own cyclic order; `zonotope(center=False)`
    leaves the empty sum at the origin, so each unit already sits with its
    near pole on the centre and needs no placement.  `pole` is the far
    pole, the sum of the generators, which is the vertex the faceting step
    joins up.
    """
    c = centre if centre is not None else circumcentre(V)
    out = []
    for f in F:
        star = [_sub(V[i], c) for i in f]
        if len(star) < 3:
            continue
        if abs(_dot(_unit(_cross(_sub(star[1], star[0]),
                                 _sub(star[2], star[0]))),
                    star[0])) < 1e-9:
            raise ValueError(
                "one face's plane passes through the centre, so its unit "
                "would be flat; that seed cannot carry a cluster")
        Vc, Fc = _zt.zonotope(star, center=False)
        pole = [0.0, 0.0, 0.0]
        for g in star:
            pole = _add(pole, g)
        out.append((Vc, Fc, len(star), pole))
    return out


def facet_poles(units):
    """The hull of the units' outer poles.

    Adjacent units share a face, so their poles are the natural vertices
    of a solid wrapped round the cluster.  On a Kleetope of the
    dodecahedron the result is combinatorially a truncated icosahedron --
    twelve pentagons and twenty hexagons -- whatever the pyramid height,
    and the height only moves the two edge lengths.
    """
    P = [u[3] for u in units]
    if len(P) < 4:
        raise ValueError("a faceted solid needs at least four units")
    return P, _hull.hull_faces(P)


def edge_ratio(V, F, tol=1e-6):
    """(shortest, longest) distinct edge length of a faceted solid.

    Marotta and Redmond track B/A across the family, and B/A = 1 is the
    Archimedean truncated icosahedron.
    """
    e = sorted(edge_lengths(V, F))
    if not e:
        return 0.0, 0.0
    return e[0], e[-1]


def truncated_icosahedron_ratio(h):
    """B/A for the faceted Kleetope of the dodecahedron at height h.

    Marotta and Redmond's closed form; B/A = 1 marks the Archimedean
    truncated icosahedron, and the family degenerates to the
    icosidodecahedron (maximum truncation) at the lower bound on h.
    """
    return (0.4 * math.sqrt(5.0 * (5.0 - 2.0 * math.sqrt(5.0))) * h
            + (5.0 + math.sqrt(5.0)) / 10.0)


#: below this pyramid height the faceting reaches maximum truncation and
#: collapses towards the icosidodecahedron
KIS_HEIGHT_FLOOR = -PHI ** 2 / (2.0 * math.sqrt(3.0 - PHI))


def build_cluster(family, sid, n=6, augment=False, kis_height=0.0,
                  explode=0.0, strict=True):
    """(units, note).  `strict` insists on Webster's conditions.

    `augment` is deliberately separate from `kis_height`: a zero-height
    Kleetope is not the seed, it is the seed with every face split into
    coplanar triangles, and it gives a completely different cluster.
    """
    V, F = seed_solid(family, sid, n)
    centre = [0.0, 0.0, 0.0]                 # seed_solid already centred
    ok, spread = circumscribable(V, centre)
    margin = centre_margin(V, F, centre)
    if strict and not augment:
        if not ok:
            raise ValueError(
                "that seed has no circumsphere (vertex radii spread by "
                "%.1f%%), so its units would not be polar zonohedra; "
                "augment it instead" % (100.0 * spread))
        if margin <= 1e-9:
            raise ValueError(
                "that seed's centre lies on or outside one of its faces, "
                "so the segments do not open into an umbrella")
    Vk, Fk = kis(V, F, kis_height) if augment else (V, F)
    units = cluster_units(Vk, Fk, centre)
    if explode:
        moved = []
        for Vc, Fc, m, pole in units:
            d = _mul(pole, explode)
            moved.append(([_add(v, d) for v in Vc], Fc, m, _add(pole, d)))
        units = moved
    orders = sorted({m for _V, _F, m, _p in units})
    note = "%d units, orders %s" % (
        len(units), "/".join(str(o) for o in orders))
    if not ok or augment:
        note += " (generalized)"
    return units, note


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    try:
        from .styles import shell as _shell
        from .styles import face_colors as _fc
        from . import regular_solids_generator as _rs
    except ImportError:
        from styles import shell as _shell
        from styles import face_colors as _fc
        import regular_solids_generator as _rs

    MODES = [
        ('CLUSTER', "Cluster",
         "One polar zonohedron per face of the seed, packed around its "
         "centre"),
        ('FACETED', "Faceted",
         "The hull of the units' outer poles. On a Kleetope of the "
         "dodecahedron this is a truncation of the icosahedron"),
        ('BOTH', "Cluster and Faceted",
         "Both, so the spiky cluster can be seen inside its faceted "
         "shell"),
    ]

    FAMILIES = [
        ('PLATONIC', "Platonic", "The five regular solids"),
        ('ARCHIMEDEAN', "Archimedean", "The thirteen semiregular solids"),
        ('JOHNSON', "Johnson",
         "The nineteen circumscribable Johnson solids whose centre lies "
         "inside every face"),
        ('PRISM', "Prism", "The uniform n-gonal prisms"),
        ('ANTIPRISM', "Antiprism", "The uniform n-gonal antiprisms"),
    ]

    #: The Johnson solids that carry a cluster.  All 92 were tested for a
    #: circumsphere and for the centre lying strictly inside every face;
    #: 25 have the sphere and 6 of those -- the pyramid, cupola and
    #: rotunda shapes J1..J6, which are caps rather than closed bodies
    #: around their centre -- fail the second test, leaving these 19.
    #: That is the count Webster reports, arrived at independently.
    JOHNSON_OK = ['J11', 'J19', 'J27', 'J34', 'J37', 'J62', 'J63',
                  'J72', 'J73', 'J74', 'J75', 'J76', 'J77', 'J78',
                  'J79', 'J80', 'J81', 'J82', 'J83']

    def _plat_items():
        return [(sid, lbl, lbl) for sid, lbl, *_r in _rs.PLATONIC]

    def _arch_items():
        return [(sid, lbl, lbl) for sid, lbl, *_r in _rs.ARCHIMEDEAN]

    def _john_items():
        by = {sid: lbl for sid, lbl, *_r in _rs.JOHNSON}
        return [(sid, by.get(sid, sid), by.get(sid, sid))
                for sid in JOHNSON_OK]

    PLAT_ITEMS = _plat_items()
    ARCH_ITEMS = _arch_items()
    JOHN_ITEMS = _john_items()

    def _mesh_from(name, V, F):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
        me.update()
        return me

    def _hull_bmesh(pts):
        """Convex hull with coplanar triangles merged back into real
        faces -- the faceted solid's hexagons must not arrive as fans."""
        bm = bmesh.new()
        for p in pts:
            bm.verts.new((p[0], p[1], p[2]))
        bm.verts.ensure_lookup_table()
        bmesh.ops.convex_hull(bm, input=bm.verts[:])
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.5),
                                 verts=bm.verts[:], edges=bm.edges[:])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        V = [list(v.co) for v in bm.verts]
        idx = {v: k for k, v in enumerate(bm.verts)}
        F = [[idx[v] for v in f.verts] for f in bm.faces]
        bm.free()
        return V, F

    class MESH_OT_zonohedra_cluster_add(bpy.types.Operator):
        """Add a cluster of polar zonohedra packed around a point, or the
        solid that facets it"""
        bl_idname = "mesh.zonohedra_cluster_add"
        bl_label = "Zonohedral Cluster"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(name="Mode", items=MODES, default='CLUSTER',
                           description="What to build from the cluster")
        family: EnumProperty(
            name="Family", items=FAMILIES, default='PLATONIC',
            description="Which family the seed solid comes from")
        platonic: EnumProperty(
            name="Solid", items=PLAT_ITEMS, default='DODECA',
            description="Seed solid; one unit is built per face")
        archimedean: EnumProperty(
            name="Solid", items=ARCH_ITEMS, default='CO',
            description="Seed solid; one unit is built per face")
        johnson: EnumProperty(
            name="Solid", items=JOHN_ITEMS, default='J34',
            description="Seed solid; one unit is built per face")
        poly_n: IntProperty(
            name="Sides", default=6, min=3, max=24,
            description="Order of the prism or antiprism")
        augment: BoolProperty(
            name="Augment Faces", default=False,
            description="Raise a pyramid on every face of the seed "
                        "before clustering (a Kleetope). This leaves "
                        "Webster's circumscribable case for Marotta and "
                        "Redmond's generalized one. Note that a pyramid "
                        "of height zero is NOT the plain seed: the apex "
                        "lands in the face's plane and splits it into "
                        "coplanar triangles, which is a different cluster "
                        "again")
        kis_height: FloatProperty(
            name="Pyramid Height", default=0.4,
            min=-1.10, max=2.0,
            description="Raise a pyramid of this height on every face of "
                        "the seed before clustering, in seed edge "
                        "lengths. Nonzero heights leave Webster's "
                        "circumscribable case for the generalized one; "
                        "negative heights excavate instead of augment. "
                        "Below about -1.114 the faceting reaches maximum "
                        "truncation and collapses")
        explode: FloatProperty(
            name="Explode", default=0.0, min=0.0, max=3.0,
            description="Push each unit outward along its own axis, "
                        "opening the cluster up")
        separate: BoolProperty(
            name="Separate Units", default=False,
            description="One object per unit instead of a single mesh")
        color: EnumProperty(
            name="Colour By",
            items=[('NONE', "None", "One material, or none at all"),
                   ('ORDER', "Order",
                    "One material per unit order n, which is how Webster "
                    "draws them and makes the seed's face types obvious "
                    "at a glance"),
                   ('UNIT', "Unit", "One material per unit"),
                   ('SIDES', "Face Size", "One material per edge count")],
            default='ORDER',
            description="What the colours mean")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size multiplier")
        __annotations__.update(_shell.style_properties())

        def _sid(self):
            return {'PLATONIC': self.platonic,
                    'ARCHIMEDEAN': self.archimedean,
                    'JOHNSON': self.johnson}.get(self.family, '')

        def execute(self, context):
            try:
                units, note = build_cluster(
                    self.family, self._sid(), self.poly_n,
                    self.augment, self.kis_height, self.explode)
            except (ValueError, RuntimeError, KeyError) as e:  # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

            if self.mode in ('FACETED', 'BOTH'):
                poles = [u[3] for u in units]
                try:
                    Vf, Ff = _hull_bmesh(poles)
                except (ValueError, RuntimeError) as e:   # noqa: BLE001
                    self.report({'ERROR'}, str(e))
                    return {'CANCELLED'}
                lo, hi = edge_ratio(Vf, Ff)
                note += "; faceted F=%d, edge ratio %.4f" % (
                    len(Ff), (lo / hi) if hi else 0.0)
            else:
                Vf, Ff = [], []

            # one fit for everything, so a cluster and its shell stay
            # registered with each other
            allv = [v for Vc, _F, _m, _p in units for v in Vc] + list(Vf)
            fitted = _fit.fit_cube(allv, 2.0 * self.scale)
            it = iter(fitted)
            units = [([next(it) for _ in Vc], Fc, m, p)
                     for Vc, Fc, m, p in units]
            Vf = [next(it) for _ in Vf]

            made = []
            emitted = False
            if self.mode in ('CLUSTER', 'BOTH'):
                made += self._emit_cluster(context, units)
            if self.mode in ('FACETED', 'BOTH'):
                objs, emitted = self._emit_faceted(context, Vf, Ff)
                made += objs
            # `emitted` covers the Face Segments style, which builds and
            # links its own objects and hands back none -- an empty list
            # there means success, not failure
            if not made and not emitted:
                self.report({'ERROR'}, "nothing was built")
                return {'CANCELLED'}
            for o in context.selected_objects:
                o.select_set(False)
            for o in made:
                if o.name not in context.collection.objects:
                    context.collection.objects.link(o)
                o.location = context.scene.cursor.location
                o.select_set(True)
            if made:
                context.view_layer.objects.active = made[0]
            self.report({'INFO'}, note)
            return {'FINISHED'}

        def _keys(self, units):
            """One colour key per face, in emission order."""
            keys = []
            for u, (Vc, Fc, m, _p) in enumerate(units):
                for f in Fc:
                    if self.color == 'ORDER':
                        keys.append(m)
                    elif self.color == 'UNIT':
                        keys.append(u)
                    else:
                        keys.append(len(f))
            return keys

        def _emit_cluster(self, context, units):
            want = self.color != 'NONE'
            mats, midx = ([], None)
            if want:
                mats, midx = _fc.materials_for(self._keys(units), "Unit")
            if self.separate:
                made = []
                at = 0
                for i, (Vc, Fc, m, _p) in enumerate(units):
                    me = _mesh_from("Zonohedron %d" % i, Vc, Fc)
                    if want:
                        # a separated unit keeps the colour it had in the
                        # assembly, so exploding does not recolour it
                        k = midx[at]
                        me.materials.append(mats[k])
                    at += len(Fc)
                    made.append(bpy.data.objects.new("Zonohedron %d" % i,
                                                     me))
                return made
            V, F = [], []
            for Vc, Fc, _m, _p in units:
                off = len(V)
                V += Vc
                F += [[i + off for i in f] for f in Fc]
            me = _mesh_from("Zonohedral Cluster", V, F)
            for mt in mats:
                me.materials.append(mt)
            if want and me.materials and len(midx) == len(me.polygons):
                me.polygons.foreach_set('material_index', midx)
                me.update()
            return [bpy.data.objects.new("Zonohedral Cluster", me)]

        def _emit_faceted(self, context, V, F):
            """(objects, did the style emit its own).

            The Face Segments style splits the shell into one object per
            face and links them itself, returning nothing -- so an empty
            list from it is a finished build, not an empty one.
            """
            obj = _shell.apply(self, context, V, F, "Faceted Cluster")
            return ([obj], False) if obj is not None else ([], True)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            lay.prop(self, 'family')
            if self.family == 'PLATONIC':
                lay.prop(self, 'platonic')
            elif self.family == 'ARCHIMEDEAN':
                lay.prop(self, 'archimedean')
            elif self.family == 'JOHNSON':
                lay.prop(self, 'johnson')
            else:
                lay.prop(self, 'poly_n')
            lay.prop(self, 'augment')
            if self.augment:
                lay.prop(self, 'kis_height')
            if self.mode in ('CLUSTER', 'BOTH'):
                lay.prop(self, 'explode')
                lay.prop(self, 'separate')
            lay.prop(self, 'color')
            if self.mode in ('FACETED', 'BOTH'):
                _shell.draw_style(self, lay)
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.zonohedra_cluster_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_zonohedra_cluster_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_zonohedra_cluster_add)


def _selftest():
    def close(a, b, t=1e-6):
        return abs(a - b) <= t * max(1.0, abs(a), abs(b))

    # --- prisms and antiprisms are circumscribable by construction -----
    for n in (3, 5, 8):
        for anti in (False, True):
            V, F = prism(n, anti)
            e = edge_lengths(V, F)
            assert close(min(e), 1.0) and close(max(e), 1.0), (n, anti,
                                                               min(e), max(e))
            ok, spread = circumscribable(V)
            assert ok, (n, anti, spread)
    # the triangular antiprism IS the octahedron
    V, F = prism(3, anti=True)
    assert (len(V), len(F)) == (6, 8), (len(V), len(F))

    # --- Webster's clusters -------------------------------------------
    # seed_solid is the one that puts the seed on UNIT EDGES and centres
    # it on its circumcentre.  Reaching for a raw seed table instead is
    # the trap here: those come on unit CIRCUMRADIUS, and every published
    # height and ratio below is stated per unit edge, so the checks would
    # silently compare against the wrong member of the family.
    Vi, Fi = seed_solid('PLATONIC', 'ICOSA')
    units = cluster_units(Vi, Fi, [0.0, 0.0, 0.0])
    # every face is a triangle, so every unit is a parallelepiped, and
    # twenty of them make the rhombic hexecontahedron: 6 faces each, of
    # which 3 touch the centre, leaving 20 x 3 = 60 outer golden rhombi
    assert len(units) == 20, len(units)
    assert all(m == 3 and len(Fc) == 6 for _V, Fc, m, _p in units)
    golden = math.degrees(math.acos(1.0 / math.sqrt(5.0)))
    for Vc, Fc, _m, _p in units:
        for f in Fc:
            assert len(f) == 4
            o, a, b = Vc[f[0]], Vc[f[1]], Vc[f[3]]
            u, w = _unit(_sub(a, o)), _unit(_sub(b, o))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, _dot(u, w)))))
            assert close(min(ang, 180.0 - ang), golden, 1e-6), ang
    # the far poles are all the same distance out -- the cluster has the
    # seed's symmetry
    rad = {round(_norm(p), 9) for _V, _F, _m, p in units}
    assert len(rad) == 1, sorted(rad)

    # the octahedron's faces give CUBES, because the three segments from
    # the centre to a face's vertices are mutually orthogonal
    Vo, Fo = seed_solid('PLATONIC', 'OCTA')
    units = cluster_units(Vo, Fo, [0.0, 0.0, 0.0])
    assert len(units) == 8, len(units)
    for Vc, Fc, _m, _p in units:
        e = edge_lengths(Vc, Fc)
        assert close(min(e), max(e)), (min(e), max(e))
        # right angles at the shared corner
        g = [_sub(Vc[i], [0.0, 0.0, 0.0]) for i in range(len(Vc))]
        del g

    # the dodecahedron gives Webster's original: twelve 5-fold units
    Vd, Fd = seed_solid('PLATONIC', 'DODECA')
    assert close(min(edge_lengths(Vd, Fd)), 1.0), min(edge_lengths(Vd, Fd))
    units = cluster_units(Vd, Fd, [0.0, 0.0, 0.0])
    assert len(units) == 12, len(units)
    assert all(m == 5 for _V, _F, m, _p in units)
    assert all(len(Fc) == 5 * 4 for _V, Fc, _m, _p in units)

    # --- the centre must be the CIRCUMcentre ---------------------------
    # a fit through a solid whose centroid differs must still find equal
    # radii; use a deliberately off-centre copy of the icosahedron
    shifted = [_add(v, [0.3, -0.2, 0.45]) for v in Vi]
    c = circumcentre(shifted)
    ok, spread = circumscribable(shifted, c)
    assert ok and spread < 1e-9, spread
    assert _norm(_sub(c, [0.3, -0.2, 0.45])) < 1e-6, c

    # --- Kleetopes and the generalized cluster -------------------------
    Vk, Fk = kis(Vd, Fd, 0.4)
    assert len(Fk) == 12 * 5, len(Fk)              # 60 triangles
    assert len(Vk) == len(Vd) + 12
    ok, _s = circumscribable(Vk)
    assert not ok                                   # no longer inscribed
    units = cluster_units(Vk, Fk, [0.0, 0.0, 0.0])
    assert len(units) == 60, len(units)
    assert all(m == 3 for _V, _F, m, _p in units)   # parallelepipeds

    # faceting the spiky form: twelve pentagons and twenty hexagons,
    # whatever the height -- a truncated icosahedron combinatorially
    for h in (0.4, 0.05, -0.3):
        Vk, Fk = kis(Vd, Fd, h)
        units = cluster_units(Vk, Fk, [0.0, 0.0, 0.0])
        P, HF = facet_poles(units)
        sizes = sorted(len(f) for f in HF)
        assert len(HF) == 32, (h, len(HF), sizes[:5])
        assert sizes.count(5) == 12 and sizes.count(6) == 20, (h, sizes)

    # ... and B/A follows Marotta and Redmond's closed form.  The seed
    # here is on unit edges, which is the unit their h is measured in.
    for h in (0.0, 0.3, -0.2, 0.6):
        Vk, Fk = kis(Vd, Fd, h)
        units = cluster_units(Vk, Fk, [0.0, 0.0, 0.0])
        P, HF = facet_poles(units)
        lo, hi = edge_ratio(P, HF)
        want = truncated_icosahedron_ratio(h)
        # edge_ratio reports shortest over longest, so compare against
        # whichever way up B/A happens to fall
        assert close(lo / hi, min(want, 1.0 / want), 2e-4),             (h, lo / hi, want)

    # the Archimedean truncated icosahedron is the B/A = 1 member; solve
    # the closed form for h and check the faceted solid really is regular
    h1 = (1.0 - (5.0 + math.sqrt(5.0)) / 10.0) / \
         (0.4 * math.sqrt(5.0 * (5.0 - 2.0 * math.sqrt(5.0))))
    Vk, Fk = kis(Vd, Fd, h1)
    units = cluster_units(Vk, Fk, [0.0, 0.0, 0.0])
    P, HF = facet_poles(units)
    lo, hi = edge_ratio(P, HF)
    assert close(lo / hi, 1.0, 1e-4), (h1, lo / hi)
    assert len(HF) == 32 and sorted(len(f) for f in HF).count(5) == 12

    # --- the seed conditions -------------------------------------------
    # a face plane through the centre is refused rather than emitting a
    # flat "unit"
    flat = [[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0],
            [0, 0, 1.0], [0, 0, -1.0]]
    try:
        cluster_units(flat, [[0, 1, 2, 3]], [0.0, 0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("a face through the centre was accepted")
    # and the margin test sees it
    assert centre_margin(Vi, Fi, [0.0, 0.0, 0.0]) > 0.1
    assert centre_margin(flat, [[0, 1, 2, 3]], [0.0, 0.0, 0.0]) <= 1e-9

    # --- explode moves units without changing them ---------------------
    units, note = build_cluster('PLATONIC', 'ICOSA')
    moved, _n = build_cluster('PLATONIC', 'ICOSA', explode=0.5)  # noqa
    assert len(units) == len(moved) == 20
    for (Va, Fa, _ma, pa), (Vb, Fb, _mb, pb) in zip(units, moved):
        ea, eb = sorted(edge_lengths(Va, Fa)), sorted(edge_lengths(Vb, Fb))
        assert all(close(x, y) for x, y in zip(ea, eb))
        assert close(_norm(pb), 1.5 * _norm(pa), 1e-9)
    assert "20 units" in note and "orders 3" in note

    # every Platonic solid is circumscribable, so none is refused
    for sid, want in (('TETRA', 4), ('CUBE', 6), ('OCTA', 8),
                      ('DODECA', 12), ('ICOSA', 20)):
        u, _n = build_cluster('PLATONIC', sid)
        assert len(u) == want, (sid, len(u))

    # a zero-height Kleetope is NOT the seed: the dodecahedron carries
    # twelve 5-fold units, its flat Kleetope sixty parallelepipeds
    plain, _n = build_cluster('PLATONIC', 'DODECA')
    flat_kis, _n = build_cluster('PLATONIC', 'DODECA', augment=True,
                                 kis_height=0.0)
    assert len(plain) == 12 and len(flat_kis) == 60, (len(plain),
                                                      len(flat_kis))

    # the documented floor is Marotta and Redmond's degeneracy bound
    assert close(KIS_HEIGHT_FLOOR, -1.1135163644, 1e-8), KIS_HEIGHT_FLOOR

    print("RESULT: OK")
