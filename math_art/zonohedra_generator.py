
# Zonohedra Generator for Blender
#
# Zonohedra (Minkowski sums of line segments) from vector stars, after
# Antiprism's `zono` program:
#
# - General mode: convex hull of all subset sums of the star vectors
#   (robust for any star; coplanar hull triangles are merged into the
#   zonohedron's rhombi/zonogons). Classic stars included: cube axes,
#   cube diagonals (rhombic dodecahedron), icosahedral axes (rhombic
#   triacontahedron), dodecahedral axes (rhombic enneacontahedron).
# - Polar zonohedra and rhombic spirallohedra: a direct port of
#   Antiprism's make_polar_zonohedron (base/zonohedron.cc), including
#   the spiral-width option -- `zono -P 12,4` is the Rhombic
#   Spirallohedron preset (after Russell Towle).

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
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
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
        ('CUBE', "Cube", "3 orthogonal vectors"),
        ('RANDOM', "Random Star", "n random unit vectors"),
    ]

    class MESH_OT_zonohedron_add(bpy.types.Operator):
        """Add a zonohedron, polar zonohedron or rhombic spirallohedron"""
        bl_idname = "mesh.zonohedron_add"
        bl_label = "Zonohedron"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(name="Star", items=KINDS, default='SPIRAL')
        n: IntProperty(
            name="Vectors", default=12, min=3, max=64,
            description="Number of star vectors (polar/spiral/random)")
        spiral_width: IntProperty(
            name="Spiral Width", default=4, min=1, max=31,
            description="Spirallohedron spiral width (zono -P n,w)")
        pitch: FloatProperty(
            name="Pitch", default=55.0, min=5.0, max=85.0,
            description="Polar star pitch angle from the axis (degrees)")
        rand_seed: IntProperty(name="Random Seed", default=1, min=0)
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed zonohedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo "
                    "Style Geometry Nodes modifier (Border and "
                    "Thickness stay editable on the modifier)"),
                   ('WIRE', "Wireframe",
                    "Struts along the zone edges (Wireframe "
                    "modifier)")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width (fraction of "
                        "the face)")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness for the Leonardo "
                        "and Wireframe styles")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

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
                else:
                    nv = self.n if kind == 'RANDOM' else None
                    star = star_vectors(kind, n=min(self.n, 13),
                                        seed=self.rand_seed)
                    if kind == 'RANDOM' and self.n > 13:
                        self.report({'WARNING'},
                                    "random star capped at 13 vectors")
                    pts = subset_sums(star)
                    me = bpy.data.meshes.new("Zonohedron")
                    bm = bmesh.new()
                    vlist = [bm.verts.new(tuple(p)) for p in pts]
                    out = bmesh.ops.convex_hull(bm, input=vlist)
                    inner = [v for v in bm.verts
                             if v not in set(out["geom"])
                             and isinstance(v, bmesh.types.BMVert)]
                    unused = [v for v in bm.verts if not v.link_faces]
                    bmesh.ops.delete(bm, geom=unused, context='VERTS')
                    bmesh.ops.dissolve_limit(
                        bm, angle_limit=math.radians(0.5),
                        verts=bm.verts[:], edges=bm.edges[:])
                    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                    bm.to_mesh(me)
                    bm.free()
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
            self.report({'INFO'},
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind in ('POLAR', 'SPIRAL', 'RANDOM'):
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
            if self.style != 'SOLID':
                lay.prop(self, 'thickness')
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


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for n, w in ((12, 0), (12, 4), (7, 0), (50, 5)):
            star = polar_star(n)
            V, F = make_polar_zonohedron(star, 1, w)
            print(f"polar n={n} w={w}: raw V={len(V)} F={len(F)} "
                  f"(expect F={'n(n-1)=' + str(n * (n - 1)) if w == 0 else '?'})")
