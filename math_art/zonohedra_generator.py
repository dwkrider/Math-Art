
# Zonohedra Generator for Blender
#
# Zonohedra (Minkowski sums of line segments) from vector stars, after
# Antiprism's `zono` program:
#
# - General mode: the faces are enumerated directly from the star by
#   `polyhedra/zonotope.py` -- one face per pair of zones, opening into a
#   2m-gon where m zones are coplanar. That replaced a convex hull of all
#   2^n subset sums, which was exact but capped this path at thirteen
#   vectors. Classic stars included: cube axes, cube diagonals (rhombic
#   dodecahedron), icosahedral axes (rhombic triacontahedron),
#   dodecahedral axes (rhombic enneacontahedron).
# - Polar zonohedra and rhombic spirallohedra: a direct port of
#   Antiprism's make_polar_zonohedron (base/zonohedron.cc), including
#   the spiral-width option -- `zono -P 12,4` is the Rhombic
#   Spirallohedron preset (after Russell Towle).
#
# References:
# - Zonohedra / zonotopes: E. S. Fedorov (1885).
# - Zonohedra as Minkowski sums / zones: H. S. M. Coxeter, "Regular
#   Polytopes", 3rd ed., Dover, 1973.
# - Polar zonohedra and rhombic spirallohedra: Russell Towle
#   (zonohedra.com).
# - Antiprism (Adrian Rossiter), the `zono` program.
# - Rhombic rose (the flat, two-dimensional case): Alan H. Schoen,
#   "Rhombic rosettes" (schoengeometry.com); construction and counts from
#   Robert Ferreol, "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), "rosace rhombique".  Its order-5 pair of rhombi are
#   the ones in Roger Penrose's rhomb tilings.

bl_info = {
    "name": "Zonohedra Generator",
    "author": "Math Art project (after Antiprism's zono)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Zonohedron",
    "description": "Zonohedra, polar zonohedra and rhombic spirallohedra",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, gcd

try:
    from .polyhedra import zonotope as _zt
except ImportError:                       # flat-file / headless import
    from polyhedra import zonotope as _zt

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------
# Stars
# --------------------------------------------------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / l for x in v]


def star_vectors(kind, n=7, pitch=45.0, seed=0):
    if kind == 'CUBE':
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if kind == 'RHOMBIC_DODECA':          # 4 cube diagonals
        return [_unit([1, 1, 1]), _unit([1, -1, -1]),
                _unit([-1, 1, -1]), _unit([-1, -1, 1])]
    if kind == 'TRIACONTA':               # 6 icosahedral (5-fold) axes
        vs = []
        for a in (1, -1):
            vs.append(_unit([0, a, PHI]))
            vs.append(_unit([a, PHI, 0]))
            vs.append(_unit([PHI, 0, a]))
        return vs
    if kind == 'ENNEACONTA':              # 10 dodecahedral (3-fold) axes
        vs = [_unit([1, 1, 1]), _unit([1, -1, -1]),
              _unit([-1, 1, -1]), _unit([-1, -1, 1])]
        for a in (1, -1):
            vs.append(_unit([0, a / PHI, PHI]))
            vs.append(_unit([a / PHI, PHI, 0]))
            vs.append(_unit([PHI, 0, a / PHI]))
        return vs
    if kind == 'POLAR':
        return polar_star(n, pitch)
    if kind == 'RANDOM':
        import random
        rng = random.Random(seed)
        vs = []
        while len(vs) < n:
            v = [rng.gauss(0, 1) for _ in range(3)]
            if math.sqrt(sum(x * x for x in v)) > 1e-3:
                vs.append(_unit(v))
        return vs
    raise ValueError(kind)


def polar_star(n, pitch=45.0):
    """Antiprism -P star: unit vectors, equal azimuth spacing, common
    pitch (45 degrees in the original)."""
    p = math.radians(pitch)
    return [[sin(p) * cos(2 * pi * k / n), -sin(p) * sin(2 * pi * k / n),
             cos(p)] for k in range(n)]


def subset_sums(star):
    pts = [[0.0, 0.0, 0.0]]
    for v in star:
        pts = pts + [[p[0] + v[0], p[1] + v[1], p[2] + v[2]] for p in pts]
    return pts


# --------------------------------------------------------------------------
# Polar zonohedron / spirallohedron: port of Antiprism base/zonohedron.cc
# --------------------------------------------------------------------------

def _pos_mod(a, b):
    return a % b if b else 0


def _get_idx(P, s, s_step, num_spirals, i, j, V):
    if i < 0:
        if j == 0:
            idx = -2
        elif j > P - s_step:
            idx = ((_pos_mod(s - 1, num_spirals) + 1)
                   * (P - s_step) * s_step) + j - P
        else:
            idx = (_pos_mod(s - 1, num_spirals) * (P - s_step) * s_step) \
                + (j - 1) * s_step
    elif j == s_step:
        if i == P - s_step - 1:
            idx = -1
        elif i >= P - 2 * s_step:
            idx = (_pos_mod(s - 1, num_spirals) * (P - s_step) * s_step) \
                + (P - s_step) * s_step + i - (P - s_step - 1)
        else:
            idx = (_pos_mod(s - 1, num_spirals) * (P - s_step) * s_step) \
                + (i + j) * s_step
    else:
        idx = (s * (P - s_step) * s_step) + (i * s_step) + j
    return V + idx + 2


def _get_face(P, s, s_step, num_spirals, i, j, V):
    return [_get_idx(P, s, s_step, num_spirals, i - 1, j, V),
            _get_idx(P, s, s_step, num_spirals, i, j, V),
            _get_idx(P, s, s_step, num_spirals, i, j + 1, V),
            _get_idx(P, s, s_step, num_spirals, i - 1, j + 1, V)]


def make_polar_zonohedron(star, step=1, spiral_step=0):
    """Direct port of Antiprism's make_polar_zonohedron. Returns
    (verts, faces). spiral_step=0 gives the polar zonohedron; nonzero
    gives a rhombic spirallohedron of that spiral width."""
    verts = []
    faces = []
    N = len(star)
    D = step
    num_parts = gcd(N, D)
    P = N // num_parts
    P_spiral_step = 1 if spiral_step == 0 else \
        _pos_mod(spiral_step // num_parts, P)
    if P_spiral_step == 0:
        raise ValueError("invalid spiral width for this star")
    P_num_spirals = P // gcd(P, P_spiral_step)

    for p in range(num_parts):
        V = len(verts)
        star_part = [star[(p * P // P_num_spirals + i * D) % N]
                     for i in range(P)]
        verts.append([0.0, 0.0, 0.0])       # initial point
        verts.append([0.0, 0.0, 0.0])       # final point, set later
        A = [0.0, 0.0, 0.0]
        B = [0.0, 0.0, 0.0]
        for s in range(P_num_spirals):
            A = [0.0, 0.0, 0.0]
            for i in range(P - P_spiral_step):
                i_idx = (s * P_spiral_step + i) % P
                A = [A[c] + star_part[i_idx][c] for c in range(3)]
                B = [0.0, 0.0, 0.0]
                for j in range(P_spiral_step):
                    verts.append([A[c] + B[c] for c in range(3)])
                    j_idx = _pos_mod((s - 1) * P_spiral_step + j, P)
                    B = [B[c] + star_part[j_idx][c] for c in range(3)]
                    faces.append(_get_face(P, s, P_spiral_step,
                                           P_num_spirals, i, j, V))
        verts[V + 1] = [A[c] + B[c] for c in range(3)]
    return verts, faces


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    KINDS = [
        ('POLAR', "Polar Zonohedron", "n equal-pitch star vectors"),
        ('SPIRAL', "Rhombic Spirallohedron",
         "Polar star with spiral width (zono -P n,w; after Russell Towle)"),
        ('RHOMBIC_DODECA', "Rhombic Dodecahedron", "4 cube diagonals"),
        ('TRIACONTA', "Rhombic Triacontahedron", "6 icosahedral axes"),
        ('ENNEACONTA', "Rhombic Enneacontahedron", "10 dodecahedral axes"),
        # The cube is a zonohedron -- three orthogonal generators, the
        # simplest parallelohedron there is -- but offering it here only
        # duplicates Add > Polyhedra > Regular Solid, which builds it
        # exactly and with the whole Platonic family beside it. build()
        # still accepts kind='CUBE' as the degenerate base case.
        ('RANDOM', "Random Star", "n random unit vectors"),
        ('ROSETTE', "Rhombic Rose",
         "The two-dimensional member of the family: rings of rhombi "
         "filling a regular polygon.  At order 5 its two rhombi are the "
         "ones the Penrose tilings are made of"),
    ]

    try:
        from .styles import net_style as _net_style
    except ImportError:
        from styles import net_style as _net_style

    class MESH_OT_zonohedron_add(bpy.types.Operator,
                                 _net_style.NetStyleProps):
        """Add a zonohedron, polar zonohedron or rhombic spirallohedron"""
        bl_idname = "mesh.zonohedron_add"
        bl_label = "Zonohedron"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(name="Star", items=KINDS, default='SPIRAL',
                           description="Vector star the zonohedron is "
                                       "built from")
        n: IntProperty(
            name="Vectors", default=12, min=3, max=64,
            description="Number of star vectors (polar/spiral/random)")
        spiral_width: IntProperty(
            name="Spiral Width", default=4, min=1, max=31,
            description="Spirallohedron spiral width (zono -P n,w)")
        pitch: FloatProperty(
            name="Pitch", default=55.0, min=5.0, max=85.0,
            description="Polar star pitch angle from the axis (degrees)")
        rand_seed: IntProperty(name="Random Seed", default=1, min=0,
                               description="Seed for the random star "
                                           "vectors")
        style: EnumProperty(
            name="Style",
            description="How the zonohedron is rendered",
            items=[('SOLID', "Solid", "Plain closed zonohedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo "
                    "Style Geometry Nodes modifier (Border and "
                    "Thickness stay editable on the modifier)"),
                   ('WIRE', "Struts",
                    "Struts along the zone edges (Wireframe "
                    "modifier)"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face"),
            _net_style.net_enum_item()],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.06, min=0.005, max=1.0,
            description="Leonardo face frame width, the same on every "
                        "face whatever its size")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness for the Leonardo "
                        "and Wireframe styles")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
        facet_depth: FloatProperty(name="Depth", default=0.15, min=0.01,
                                   max=2.0,
                                   description="Face Segments inward depth")
        facet_gap: FloatProperty(name="Bevel Gap", default=0.0, min=0.0,
                                 max=0.5,
                                 description="Gap between face segments")
        facet_explode: FloatProperty(name="Explode", default=0.1, min=0.0,
                                     max=5.0,
                                     description="Move segments outward")
        facet_separate: BoolProperty(
            name="Separate Meshes", default=False,
            description="Each face segment as its own object")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size multiplier")

        def execute(self, context):
            kind = self.kind
            try:
                if kind in ('POLAR', 'SPIRAL'):
                    star = polar_star(self.n, self.pitch)
                    w = self.spiral_width if kind == 'SPIRAL' else 0
                    if w and gcd(self.n, w) == self.n:
                        raise ValueError(
                            "spiral width must not be a multiple of n")
                    verts, faces = make_polar_zonohedron(star, 1, w)
                    me = bpy.data.meshes.new("Zonohedron")
                    me.from_pydata([tuple(v) for v in verts],
                                   [], [tuple(f) for f in faces])
                    me.validate(clean_customdata=True)
                    bm = bmesh.new()
                    bm.from_mesh(me)
                    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
                    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                    bm.to_mesh(me)
                    bm.free()
                elif kind == 'ROSETTE':
                    # flat: the zonogon rather than the zonotope
                    rv, rf, _rings = _zt.rhombic_rosette(self.n)
                    me = bpy.data.meshes.new("Rhombic Rose")
                    me.from_pydata([(x, y, 0.0) for (x, y) in rv], [],
                                   [tuple(f) for f in rf])
                    me.validate(clean_customdata=True)
                else:
                    # The faces come straight out of the star (see
                    # polyhedra/zonotope.py), so there is no 2^n subset-sum
                    # enumeration and no cap on the number of vectors --
                    # this path used to stop at thirteen.
                    star = star_vectors(kind, n=self.n,
                                        seed=self.rand_seed)
                    verts, faces = _zt.zonotope(star)
                    me = bpy.data.meshes.new("Zonohedron")
                    me.from_pydata([tuple(v) for v in verts], [],
                                   [tuple(f) for f in faces])
                    me.validate(clean_customdata=True)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # normalize: centre on the origin and fit (roughly)
            # within a 2 x scale cube, whatever the star produced
            lo = [min(v.co[k] for v in me.vertices)
                  for k in range(3)]
            hi = [max(v.co[k] for v in me.vertices)
                  for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0
                       for k in range(3)) or 1.0
            f = self.scale / half
            for v in me.vertices:
                v.co = [(v.co[k] - (lo[k] + hi[k]) / 2.0) * f
                        for k in range(3)]
            me.polygons.foreach_set('use_smooth',
                                    [False] * len(me.polygons))
            me.update()
            if self.style in ('FACETS', 'NET'):
                Vf = [tuple(v.co) for v in me.vertices]
                Ff = [list(p.vertices) for p in me.polygons]
                bpy.data.meshes.remove(me)
                if self.style == 'NET':
                    return _net_style.emit_net_from_operator(
                        self, context, Vf, Ff, "Zonohedron",
                        hint=("the Rhombic Rose is already a flat "
                              "sheet; pick a three-dimensional star"
                              if self.kind == 'ROSETTE' else None))
                try:
                    from .styles import facet_style
                except ImportError:
                    from styles import facet_style
                facet_style.emit_facets(
                    context, Vf, Ff, "Zonohedron", self.facet_depth,
                    self.facet_gap, self.facet_explode,
                    self.facet_separate)
                self.report({'INFO'}, f"{len(Ff)} face segments")
                return {'FINISHED'}
            obj = bpy.data.objects.new("Zonohedron", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness
                mod.use_even_offset = False
            elif self.style == 'BALLSTICK':
                try:
                    from .styles import ball_and_stick
                except ImportError:
                    from styles import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'},
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind in ('POLAR', 'SPIRAL', 'RANDOM', 'ROSETTE'):
                lay.prop(self, 'n')
            if self.kind == 'SPIRAL':
                lay.prop(self, 'spiral_width')
            if self.kind in ('POLAR', 'SPIRAL'):
                lay.prop(self, 'pitch')
            if self.kind == 'RANDOM':
                lay.prop(self, 'rand_seed')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            if self.style == 'NET':
                _net_style.draw_net_props(lay, self)
            if self.style == 'FACETS':
                lay.prop(self, 'facet_depth')
                lay.prop(self, 'facet_gap')
                lay.prop(self, 'facet_explode')
                lay.prop(self, 'facet_separate')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.zonohedron_add", icon='MESH_UVSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_zonohedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_zonohedron_add)


def _selftest():
    for n, w in ((12, 0), (12, 4), (7, 0), (50, 5)):
        star = polar_star(n)
        V, F = make_polar_zonohedron(star, 1, w)
        print(f"polar n={n} w={w}: raw V={len(V)} F={len(F)} "
              f"(expect F={'n(n-1)=' + str(n * (n - 1)) if w == 0 else '?'})")
