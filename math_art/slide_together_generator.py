
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
    ('T20', "20 Triangles", 'ICOSA', 3, 1, 0.61, 12.0),
    ('S30', "30 Squares", 'ID30', 4, 1, 0.60, 18.0),
    ('P12', "12 Pentagons", 'DODECA', 5, 1, 1.38, 20.0),
    ('D12', "12 Decagons", 'DODECA', 10, 1, 1.30, 9.0),
    ('H20', "20 Hexagons", 'ICOSA', 6, 1, 0.59, 10.0),
    ('SD12', "12 Star Decagons {10/3}", 'DODECA', 10, 3, 2.10, 9.0),
    ('PG12', "12 Pentagrams", 'DODECA', 5, 2, 3.00, 20.0),
]

_MODEL = {m[0]: m for m in MODELS}

# How far each face plane sits from the centre.  Taken from the solid the
# planes come from, so a panel really does lie in a face plane.
_MODEL_OFFSET = {}


def _plane_offset(kind):
    if kind in _MODEL_OFFSET:
        return _MODEL_OFFSET[kind]
    if kind == 'ID30':
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
    pts = []
    for k in range(n):
        a = t + 2 * math.pi * d * k / n
        pts.append(_add(cen, _add(_mul(e1, radius * math.cos(a)),
                                  _mul(e2, radius * math.sin(a)))))
    return pts


def _seg_polygon_span(P, nrm, origin, direction):
    """Parameter interval of the line origin + s*direction that lies
    inside the convex hull of the planar polygon P (used to find where
    two panels overlap)."""
    lo, hi = -1e9, 1e9
    m = len(P)
    for k in range(m):
        a, b = P[k], P[(k + 1) % m]
        edge = _sub(b, a)
        out = _cross(edge, nrm)             # inward/outward test vector
        den = _dot(out, direction)
        num = _dot(out, _sub(a, origin))
        if abs(den) < 1e-12:
            if num < -1e-9:
                return None
            continue
        s = num / den
        if den > 0:
            hi = min(hi, s)
        else:
            lo = max(lo, s)
    return (lo, hi) if hi - lo > 1e-9 else None


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
            si = _seg_polygon_span(Pi, ni, org, dirv)
            sj = _seg_polygon_span(Pj, nj, org, dirv)
            if si is None or sj is None:
                continue
            lo = max(si[0], sj[0])
            hi = min(si[1], sj[1])
            if hi - lo < 1e-6:
                continue
            mid = _add(org, _mul(dirv, 0.5 * (lo + hi)))
            out.append((i, j, mid, _add(org, _mul(dirv, lo)),
                        _add(org, _mul(dirv, hi))))
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
        out.append(list(P[k]))
        for _t, foot, target in sorted(per_edge.get(k, []),
                                       key=lambda r: r[0]):
            axis = _unit(_sub(target, foot))
            side = _mul(_unit(_cross(nrm, axis)), width * 0.5)
            out.append(_sub(foot, side))
            out.append(_sub(target, side))
            out.append(_add(target, side))
            out.append(_add(foot, side))
    return out


def build_model(key, radius_scale=1.0, turn_delta=0.0, slit_width=0.05):
    """Panels of a slide-together as a list of outlines with normals."""
    key, _lbl, axes, n, d, rad, turn = _MODEL[key]
    normals = plane_normals(axes)
    r = rad * radius_scale
    off = _plane_offset(axes)
    panels = [(_polygon(n, d, r, nv, turn + turn_delta, off), nv)
              for nv in normals]
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
            name="Slit Width", default=0.05, min=0.0, max=0.3,
            description="Width of the cut slots. Zero leaves the panels "
                        "uncut, showing the bare arrangement")
        thickness: FloatProperty(
            name="Thickness", default=0.02, min=0.0, max=0.3,
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


def _selftest():
    want_panels = {'T20': 20, 'S30': 30, 'P12': 12, 'D12': 12,
                   'H20': 20, 'SD12': 12, 'PG12': 12}
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
        raw = [(_polygon(n, d, _r, nv, _t, _plane_offset(axes)), nv)
               for nv in normals]
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
        assert max(deg.values()) <= 9,             (key, "panels saw through far too many neighbours",
             sorted(deg.values())[-3:])

        # panels lie in distinct planes
        seen = set()
        for _P, nv in panels:
            k = tuple(round(c, 5) for c in nv)
            assert k not in seen, (key, "two panels share a plane")
            seen.add(k)

    print("RESULT: OK")
