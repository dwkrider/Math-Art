
# Transpolyhedra -- the forms between a polyhedron and its dual
#
# Haresh Lalvani's transpolyhedron is a transitional solid built from
# three pieces at once: every face of the original polyhedron, every
# face of its dual, and one connecting rectangle per edge.  A single
# dial takes it continuously from the one to the other, which is the
# point -- duality stops being a statement and becomes a motion.
#
# Construction.  Put the seed in CANONICAL form first, so all its edges
# are tangent to a common midsphere, and read off each face's pole
# p_f = n_f / d_f (the point where that face's plane touches).  Then one
# vertex per FLAG -- a (vertex, face) incidence -- placed at
#
#     x(v, f) = (1 - t) v + t p_f
#
# and three families of faces built from those flags:
#
#   * one per original face, its own vertices' flags, which is the face
#     shrunk toward its pole;
#   * one per original vertex, the flags of the faces around it in
#     cyclic order, which grows into the dual's face;
#   * one per original edge, over the four flags of its two endpoints
#     and two faces.
#
# That third family is a parallelogram for free and for any seed, since
#
#     x(b, f) - x(a, f) = (1 - t)(b - a) = x(b, g) - x(a, g),
#
# and it is a RECTANGLE exactly when the seed is canonical, because an
# edge and its dual edge cross at right angles at the midsphere tangency
# point.  So the "connecting rectangles" of Lalvani's description are a
# property of the canonical form rather than something to solve for.
#
# Counts follow immediately: V = 2E, F = F0 + V0 + E0, and Euler is
# 2E - 3E + (F0 + V0 + E0) = F0 + V0 - E0 = 2.
#
# References:
# - Haresh Lalvani, "Transpolyhedra: Dual Transformations by
#   Explosion-Implosion", Lalvani (1977); and his structural-morphology
#   work on continuous polyhedral transformations.
# - George W. Hart, "Transpolyhedra", Virtual Polyhedra
#   (georgehart.com/virtual-polyhedra/transpolyhedra.html), which is the
#   source of the twenty named examples built here.
# - Canonical form: George W. Hart, "Calculating Canonical Polyhedra",
#   Mathematica in Education and Research 6(3) (1997).

bl_info = {
    "name": "Transpolyhedra",
    "author": "Math Art project (after Haresh Lalvani)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Transitional forms between a polyhedron and its dual",
    "category": "Add Mesh",
}

import math

try:
    from .polyhedra import canonical as _canon
    from .polyhedra import seeds as _seeds
    from .polyhedra import fit as _fit
except ImportError:                       # flat-file / headless import
    from polyhedra import canonical as _canon
    from polyhedra import seeds as _seeds
    from polyhedra import fit as _fit


# --------------------------------------------------------------------------
# seeds -- Hart's twenty
# --------------------------------------------------------------------------
# A polyhedron and its dual give the SAME transpolyhedron (the two just
# swap which family of faces grows), so the five Platonic solids
# contribute three entries, not five -- which is exactly the list Hart
# publishes.

SEEDS = [
    ('TETRA', "Tetrahedron", 'PLATONIC', None),
    ('CUBE', "Cube", 'PLATONIC', None),
    ('DODECA', "Dodecahedron", 'PLATONIC', None),
    ('TT', "Truncated Tetrahedron", 'ARCHIMEDEAN', None),
    ('CO', "Cuboctahedron", 'ARCHIMEDEAN', None),
    ('TC', "Truncated Cube", 'ARCHIMEDEAN', None),
    ('TO', "Truncated Octahedron", 'ARCHIMEDEAN', None),
    ('RCO', "Rhombicuboctahedron", 'ARCHIMEDEAN', None),
    ('TCO', "Truncated Cuboctahedron", 'ARCHIMEDEAN', None),
    ('SC', "Snub Cube", 'ARCHIMEDEAN', None),
    ('ID', "Icosidodecahedron", 'ARCHIMEDEAN', None),
    ('TD', "Truncated Dodecahedron", 'ARCHIMEDEAN', None),
    ('TI', "Truncated Icosahedron", 'ARCHIMEDEAN', None),
    ('RID', "Rhombicosidodecahedron", 'ARCHIMEDEAN', None),
    ('TID', "Truncated Icosidodecahedron", 'ARCHIMEDEAN', None),
    ('SD', "Snub Dodecahedron", 'ARCHIMEDEAN', None),
    ('P5', "Pentagonal Prism", 'PRISM', ('PRISM', 5)),
    ('P7', "Heptagonal Prism", 'PRISM', ('PRISM', 7)),
    ('A5', "Pentagonal Antiprism", 'PRISM', ('ANTIPRISM', 5)),
    ('A7', "Heptagonal Antiprism", 'PRISM', ('ANTIPRISM', 7)),
]

_SEED = {sid: (fam, extra) for sid, _l, fam, extra in SEEDS}


def seed_solid(sid):
    """(V, F) for a seed, in canonical (midsphere) form.

    The prisms are the one place a solve really runs: a UNIFORM prism has
    no midsphere at all, so it has to be canonicalized into one before it
    has face poles to interpolate toward.
    """
    fam, extra = _SEED[sid]
    if fam == 'PLATONIC':
        V, F = _seeds.seed_poly(sid, unit=True)
        V = _canon.canonicalize_best(V, F)
        return [list(v) for v in V], [list(f) for f in F]
    try:
        from . import regular_solids_generator as rs
    except ImportError:
        import regular_solids_generator as rs
    if fam == 'PRISM':
        kind, n = extra
        V, F = rs.build_prism(kind, n, 1.0)
        V = _canon.canonicalize_best([list(v) for v in V],
                                     [list(f) for f in F])
        return [list(v) for v in V], [list(f) for f in F]
    V, F, _sizes = rs.build_solid('ARCHIMEDEAN', sid, scale=1.0, canon=True)
    V = _canon.canonicalize_best([list(v) for v in V],
                                 [list(f) for f in F])
    return [list(v) for v in V], [list(f) for f in F]


# --------------------------------------------------------------------------
# the construction
# --------------------------------------------------------------------------

def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def face_poles(V, F):
    """Each face's pole n/d -- the point where its plane touches the
    midsphere, once the seed is canonical."""
    out = []
    for f in F:
        # Newell normal, robust for the non-planar case mid-iteration
        n = [0.0, 0.0, 0.0]
        for k in range(len(f)):
            a, b = V[f[k]], V[f[(k + 1) % len(f)]]
            n[0] += (a[1] - b[1]) * (a[2] + b[2])
            n[1] += (a[2] - b[2]) * (a[0] + b[0])
            n[2] += (a[0] - b[0]) * (a[1] + b[1])
        ln = _norm(n)
        if ln < 1e-12:
            raise ValueError("degenerate face in the seed")
        n = [c / ln for c in n]
        c = [sum(V[i][k] for i in f) / len(f) for k in range(3)]
        d = _dot(n, c)
        if abs(d) < 1e-9:
            raise ValueError("a face plane passes through the centre")
        out.append([c / d for c in n])
    return out


def _faces_around(V, F):
    """For each vertex, its incident faces in cyclic order about the
    vertex direction."""
    inc = [[] for _ in V]
    for fi, f in enumerate(F):
        for vi in f:
            inc[vi].append(fi)
    poles = face_poles(V, F)
    out = []
    for vi, fis in enumerate(inc):
        if len(fis) < 3:
            out.append(list(fis))
            continue
        nrm = V[vi]
        ln = _norm(nrm) or 1.0
        nrm = [c / ln for c in nrm]
        ref = _sub(poles[fis[0]], V[vi])
        ref = _sub(ref, [nrm[k] * _dot(ref, nrm) for k in range(3)])
        rl = _norm(ref) or 1.0
        ref = [c / rl for c in ref]
        side = _cross(nrm, ref)

        def ang(fi):
            d = _sub(poles[fi], V[vi])
            return math.atan2(_dot(d, side), _dot(d, ref))

        out.append(sorted(fis, key=ang))
    return out


def build_transpolyhedron(V, F, t=0.5):
    """The transpolyhedron of (V, F) at blend `t`.

    t -> 0 is the seed, t -> 1 the dual; both ends are degenerate (all
    the flags at a vertex, or all those on a face, coincide), so the
    caller should keep t strictly inside.
    """
    poles = face_poles(V, F)
    around = _faces_around(V, F)
    flag = {}
    P = []
    for fi, f in enumerate(F):
        for vi in f:
            flag[(vi, fi)] = len(P)
            p, q = V[vi], poles[fi]
            P.append([(1.0 - t) * p[k] + t * q[k] for k in range(3)])

    out = []
    for fi, f in enumerate(F):                      # the seed's faces
        out.append([flag[(vi, fi)] for vi in f])
    for vi, fis in enumerate(around):               # the dual's faces
        if len(fis) >= 3:
            out.append([flag[(vi, fi)] for fi in fis])

    # one rectangle per edge, over the two endpoints' flags in its two
    # faces -- found by walking each face's directed edges and pairing
    # them with the reverse direction in the neighbouring face
    half = {}
    for fi, f in enumerate(F):
        for k in range(len(f)):
            half[(f[k], f[(k + 1) % len(f)])] = fi
    seen = set()
    for (a, b), fi in half.items():
        if (b, a) not in half:
            continue                                # open edge: no rectangle
        gi = half[(b, a)]
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append([flag[(a, fi)], flag[(b, fi)],
                    flag[(b, gi)], flag[(a, gi)]])

    return P, _outward(P, out)


def _outward(V, F):
    """Wind every face so its normal points away from the origin.  A
    transpolyhedron is star-shaped about its centre, so the centroid
    direction is a safe reference."""
    out = []
    for f in F:
        c = [sum(V[i][k] for i in f) / len(f) for k in range(3)]
        n = _cross(_sub(V[f[1]], V[f[0]]), _sub(V[f[2]], V[f[0]]))
        out.append(list(reversed(f)) if _dot(n, c) < 0 else list(f))
    return out


def build_seed_transpolyhedron(sid, t=0.5):
    V, F = seed_solid(sid)
    return build_transpolyhedron(V, F, t)


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

    def _seed_items():
        out = []
        last = None
        for sid, lbl, fam, _x in SEEDS:
            if fam != last:
                out.append(('', fam.title(), ''))
                last = fam
            out.append((sid, lbl, lbl))
        return out

    SEED_ITEMS = _seed_items()

    class MESH_OT_transpolyhedron_add(bpy.types.Operator):
        """Add a transpolyhedron: the transitional form between a
        polyhedron and its dual"""
        bl_idname = "mesh.transpolyhedron_add"
        bl_label = "Transpolyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Seed", items=SEED_ITEMS, default='CO',
            description="Polyhedron to transform. A solid and its dual "
                        "give the same transpolyhedron, so only one of "
                        "each dual pair is listed")
        blend: FloatProperty(
            name="Blend", default=0.5, min=0.02, max=0.98,
            description="Where to stop between the seed and its dual. "
                        "Near 0 the seed's faces dominate and the dual's "
                        "are specks; near 1 the reverse. Both ends are "
                        "degenerate, so the range stops short of them")
        colors: BoolProperty(
            name="Colours", default=True,
            description="One material for the seed's faces, one for the "
                        "dual's and one for the connecting rectangles, so "
                        "the three families the construction puts together "
                        "can be told apart")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            try:
                V, F = build_seed_transpolyhedron(self.seed, self.blend)
            except (ValueError, RuntimeError) as e:     # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            Vs, Fs = seed_solid(self.seed)
            nf, nv = len(Fs), len(Vs)
            V = _fit.fit_cube(V, 2.0 * self.scale)
            me = bpy.data.meshes.new("Transpolyhedron")
            me.from_pydata([tuple(v) for v in V], [],
                           [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
            if self.colors:
                for name, rgba in (("Trans Seed", (0.85, 0.30, 0.20, 1.0)),
                                   ("Trans Dual", (0.20, 0.45, 0.80, 1.0)),
                                   ("Trans Edge", (0.92, 0.78, 0.25, 1.0))):
                    mat = bpy.data.materials.get(name)
                    if mat is None:
                        mat = bpy.data.materials.new(name)
                        mat.use_nodes = True
                        mat.diffuse_color = rgba
                        b = mat.node_tree.nodes.get("Principled BSDF")
                        if b is not None:
                            b.inputs["Base Color"].default_value = rgba
                    me.materials.append(mat)
                idx = [0] * nf + [1] * nv + [2] * (len(F) - nf - nv)
                me.polygons.foreach_set('material_index', idx)
            me.update()
            obj = bpy.data.objects.new("Transpolyhedron", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"V={len(me.vertices)} F={len(me.polygons)}"
                                  f" ({nf} seed + {nv} dual + "
                                  f"{len(F) - nf - nv} rectangles)")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'seed')
            lay.prop(self, 'blend')
            lay.prop(self, 'colors')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.transpolyhedron_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_transpolyhedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_transpolyhedron_add)


def _selftest():
    for sid, _lbl, _fam, _x in SEEDS:
        V, F = seed_solid(sid)
        E = set()
        for f in F:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                E.add((min(a, b), max(a, b)))
        nE = len(E)
        P, G = build_transpolyhedron(V, F, 0.5)

        # V = 2E, F = F0 + V0 + E0, and Euler holds
        assert len(P) == 2 * nE, (sid, len(P), 2 * nE)
        assert len(G) == len(F) + len(V) + nE, (sid, len(G))
        GE = set()
        for f in G:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                GE.add((min(a, b), max(a, b)))
        assert len(P) - len(GE) + len(G) == 2, (sid, len(P), len(GE), len(G))

        # the edge faces are parallelograms exactly, and rectangles
        # because the seed is canonical
        for f in G[len(F) + len(V):]:
            a, b, c, d = (P[i] for i in f)
            s1, s2 = _sub(b, a), _sub(c, b)
            s3 = _sub(c, d)
            for k in range(3):
                assert abs(s1[k] - s3[k]) < 1e-6, (sid, "not a parallelogram")
            # The canonical form makes these exact, not approximate: the
            # worst seed here lands at ~2e-14, so the tolerance is set
            # where it actually guards something rather than where any
            # roughly-square quad would pass.
            cosang = _dot(s1, s2) / (_norm(s1) * _norm(s2))
            assert abs(cosang) < 1e-9, (sid, "not a rectangle", cosang)

        # the ends really are the seed and its dual: at small t every
        # flag sits near its own vertex, at large t near its own pole
        poles = face_poles(V, F)
        P0, _ = build_transpolyhedron(V, F, 0.02)
        P1, _ = build_transpolyhedron(V, F, 0.98)
        far0 = max(min(_norm(_sub(p, v)) for v in V) for p in P0)
        far1 = max(min(_norm(_sub(p, q)) for q in poles) for p in P1)
        assert far0 < 0.1, (sid, far0)
        assert far1 < 0.1, (sid, far1)

    print("RESULT: OK")
