# Spiral-Faced Polyhedron -- spidronised solids, the "spidroball".
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
# Nylander reached the same interlocking by a different route, and his
# 2010 source is reproduced in `nylander_dodeca_nest` as a check.  He
# lifts along the face normal -- the Per Face style -- but SOLVES the
# lift from the dihedral angle, dz = a*b/sqrt(a*b^2 + 1) with
# a = sin^2(pi/n) and b = tan(alpha/2), so that the raised points of
# neighbouring faces land on one another regardless.  The self-test
# confirms his twelve faces do share every boundary point, to 3e-17.
# His solution is exact for the dodecahedron; the radial displacement
# used here is not tuned to any particular solid, so it holds on all
# seven seeds and at any relief.
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
    "name": "Spiral-Faced Polyhedron",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Polyhedra",
    "description": "Spidronised solids: every face replaced by a "
                   "spiral nest of triangles",
    "category": "Add Mesh",
}

from math import cos, sin, tan, acos, pi, sqrt, radians, degrees

import numpy as np

try:
    from . import spidron_math as sm
    from .polyhedra import seeds as _seeds
    from .polyhedra import hull as _hull
    from .polyhedra import fit as _fit
    from .patterns import common as pc
except Exception:                       # legacy single-file / CLI use
    import spidron_math as sm
    from polyhedra import seeds as _seeds
    from polyhedra import hull as _hull
    from polyhedra import fit as _fit
    from patterns import common as pc

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


def nylander_dodeca_nest():
    """Paul Nylander's dodeca-spidroball boundary, reproduced exactly
    from his 2010 AutoLisp source, for validation.

    His route to an interlocking ball differs from this module's: he
    lifts along the FACE NORMAL, like Relief Style: Per Face, but solves
    the lift `dz` from the solid's dihedral angle so that the raised
    points of neighbouring faces land on each other anyway.  The Woven
    style instead displaces radially, which makes neighbours meet by
    construction on any solid at any relief.  Both are correct; his is
    exact for the dodecahedron, this module's generalises.

    Returns (boundary, face_frames, constants).
    """
    n_arm = 5
    n = 2 * n_arm
    alpha = acos(-sqrt(5.0) / 5.0)          # dihedral, 116.565 deg
    a = sin(pi / n) ** 2
    b = tan(alpha / 2.0)
    dz = a * b / sqrt(a * b * b + 1.0)
    r = sqrt(1.0 - dz * dz / a)
    z0 = r * b * cos(pi / n_arm) + dz
    nodes = np.array([[r * cos(i * pi / n_arm), r * sin(i * pi / n_arm),
                       (2 * (i % 2) - 1) * dz + z0] for i in range(n)])

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
    return nodes, frames, dict(alpha=alpha, a=a, b=b, dz=dz, r=r, z0=z0)


def woven_boundary(face, V, relief):
    """The skew 2n-gon a face contributes to the WOVEN ball.

    A face's boundary is refined to its own vertices plus its edge
    midpoints, and those points are then displaced RADIALLY -- vertices
    pushed out, midpoints pulled in -- rather than along the face's own
    normal.

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
        pts.append(tuple(a * (1.0 + relief)))
        pts.append(tuple(0.5 * (a + b) * (1.0 - relief)))
    return pts


def build(seed='DODECA', rings=14, scale=0.82, twist=radians(22.0),
          relief=0.22, chirality='ALTERNATE', open_center=False,
          relief_style='WOVEN'):
    """Spidronise every face of the seed solid."""
    SV, SF = seed_solid(seed)
    A = np.asarray(SV, float)
    A = A / float(np.linalg.norm(A, axis=1).max())      # unit circumradius
    col, colour_ok = two_colour(SF)

    verts, faces, mats = [], [], []
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
        v, fc, mt = sm.spidronise(poly, scale, twist, rings,
                                  chirality=ch, centre=C, normal=N,
                                  cap=not open_center)
        o = len(verts)
        verts.extend(v)
        faces.extend([tuple(i + o for i in t) for t in fc])
        mats.extend([m + (0 if ch > 0 else 3) for m in mt])
    return verts, faces, mats, colour_ok


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
        bl_label = "Spiral-Faced Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Solid", items=SEED_ITEMS, default='DODECA',
            description="Base polyhedron whose faces are spidronised")
        rings: IntProperty(
            name="Rings", default=14, min=1, max=20,
            description="How many times the spiral step repeats on each "
                        "face")
        scale_step: FloatProperty(
            name="Ring Scale", default=0.82, min=0.3, max=0.97,
            description="How much each ring shrinks toward the centre "
                        "of its face")
        twist: FloatProperty(
            name="Twist", default=radians(22.0),
            min=radians(-90.0), max=radians(90.0), subtype='ANGLE',
            description="How far each ring turns about the face normal. "
                        "Thirty degrees is the classical hexagonal value")
        relief: FloatProperty(
            name="Relief", default=0.22, min=0.0, max=0.8,
            description="Lift alternate boundary points out of the face "
                        "plane before spiralling, so the nests stand "
                        "proud of the solid. Zero decorates each face "
                        "flat")
        chirality: EnumProperty(
            name="Chirality", default='ALTERNATE',
            items=[('CW', "Clockwise", "Every face wound the same way"),
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
        open_center: BoolProperty(
            name="Open Centres", default=False,
            description="Leave the small hole at the centre of each "
                        "face open instead of closing it")

        def execute(self, context):
            V, F, M, colour_ok = build(
                self.seed, int(self.rings), float(self.scale_step),
                float(self.twist), float(self.relief), self.chirality,
                self.open_center, self.relief_style)
            if not F:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}
            V = _fit.fit_cube(V, 2.0)
            me = bpy.data.meshes.new("Spidron Ball")
            me.from_pydata([tuple(v) for v in V], [], F)
            cols = pc.PALETTE_RGBA
            nmat = (max(M) + 1) if M else 1
            for i in range(nmat):
                mat = bpy.data.materials.new("Spidron Ball %d" % i)
                mat.use_nodes = False
                mat.diffuse_color = cols[i % len(cols)]
                me.materials.append(mat)
            if M:
                me.polygons.foreach_set('material_index', M)
            me.validate(clean_customdata=True)
            me.update()
            import bpy_extras.object_utils as _ou
            _ou.object_data_add(context, me, operator=self)
            warn = ("" if colour_ok or self.chirality != 'ALTERNATE'
                    else "  (face graph has an odd cycle: alternating "
                         "chirality is impossible on this solid)")
            self.report({'INFO'}, "%s  V=%d F=%d%s"
                        % (self.seed.title(), len(V), len(F), warn))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for p in ('seed', 'rings', 'scale_step', 'twist', 'relief',
                      'relief_style', 'chirality', 'open_center'):
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
        V, F, M, cok = build(kind, rings=4, scale=0.62,
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

    V, F, M, _ = build('DODECA', rings=8, scale=0.62,
                       twist=radians(30.0), relief=0.35)
    Vf = _fit.fit_cube(V, 2.0)
    A = np.array(Vf)
    ext = A.max(axis=0) - A.min(axis=0)
    ctr = 0.5 * (A.max(axis=0) + A.min(axis=0))
    chk("fits the 2 m cube", abs(ext.max() - 2.0) < 1e-9
        and np.abs(ctr).max() < 1e-9, "extent %.6f" % ext.max())
    chk("relief keeps the solid three-dimensional",
        min(ext) > 0.5 * max(ext), "aspect %.3f" % (min(ext) / max(ext)))

    Vf0, _, _, _ = build('DODECA', rings=6, scale=0.62,
                         twist=radians(30.0), relief=0.0)
    Vf1, _, _, _ = build('DODECA', rings=6, scale=0.62,
                         twist=radians(30.0), relief=0.4)
    r0 = np.linalg.norm(np.array(Vf0), axis=1)
    r1 = np.linalg.norm(np.array(Vf1), axis=1)
    chk("relief pushes vertices off the face planes",
        r1.max() > r0.max() + 1e-6,
        "rmax %.4f -> %.4f" % (r0.max(), r1.max()))

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

    Vcw, _, _, _ = build('CUBE', rings=4, chirality='CW')
    Vcc, _, _, _ = build('CUBE', rings=4, chirality='CCW')
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
