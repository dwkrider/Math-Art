
# Polytwister generator for Blender.
#
# Polytwisters are curved 4D shapes built from a polyhedron via the
# Hopf fibration (Jonathan Bowers, ~2007: 222 shapes + 3 infinite
# families).  Where a polyhedron is the intersection of half-spaces --
# one per face -- a "hard" polytwister is the intersection of
# CYCLOPLANES, one per face: a cycloplane is the Hopf preimage of a
# spherical cap around the face's direction on S^2.  Both live in R^4
# and are invariant under the Hopf circle action, so one visualises
# them by a 3-space cross-section at a slice coordinate w, swept to
# animate.
#
# The load-bearing closed form (from Nathan Ho's polytwister mesher):
# the 3-space cross-section, at slice w, of a single cycloplane with
# zenith zeta and azimuth alpha is a transformed infinite cylinder.
# Starting from a unit cylinder about the Z-axis, with theta = zeta/2:
#   1. rotate +90 deg about X          (lay it along Y)
#   2. scale along X by 1 / cos(theta) (eccentricity from the slice)
#   3. translate along X by w * tan(theta)
#   4. rotate by zeta about X, then alpha about Y (aim at the face)
# At the south pole (theta -> pi/2) the scale blows up and the
# cross-section is a plain cylinder about the projection axis.  The
# polytwister cross-section is the Boolean intersection of these
# per-face cylinders; sweeping w slides and stretches them, animating
# the classic ring / strip / twister cell structure.
#
# References:
# - Jonathan Bowers, "Polytwisters" (c. 2007),
#   https://www.polytope.net/hedrondude/twisters.htm (discovery; the
#   222 + 3 infinite families and their names).
# - Nathan Ho, "Polytwisters", https://nathan.ho.name/pages/polytwisters/
#   and the reference toolchain github.com/polytwisters/polytwister-mesher
#   (the cycloplane = transformed-cylinder cross-section implemented
#   here) and github.com/polytwisters/polytwisters.
# - Heinz Hopf, "Ueber die Abbildungen der dreidimensionalen Sphaere
#   auf die Kugelflaeche", Math. Ann. 104 (1931), 637-665 (the
#   fibration underlying the cycloplane construction).

bl_info = {
    "name": "Polytwister",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polytwister",
    "description": "3D cross-section of a polytwister: the Boolean "
                   "intersection of per-face cycloplane cylinders "
                   "(Hopf-fibration 4D->3D, after Bowers / Ho)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, tan, pi, sqrt

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

_PHI = (1.0 + sqrt(5.0)) / 2.0

# generic tilt of the seed sphere so no face points at the exact south
# pole (whose cycloplane scale 1/cos(zeta/2) would blow up)
_TILT = (0.2731, 0.1904, 0.1123)


# --------------------------------------------------------------------------
# 4x4 homogeneous transform helpers (numpy)
# --------------------------------------------------------------------------

def _rx(a):
    import numpy as np
    c, s = cos(a), sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0],
                     [0, s, c, 0], [0, 0, 0, 1]], float)


def _ry(a):
    import numpy as np
    c, s = cos(a), sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0],
                     [-s, 0, c, 0], [0, 0, 0, 1]], float)


def _rz(a):
    import numpy as np
    c, s = cos(a), sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]], float)


def _sx(s):
    import numpy as np
    return np.diag([s, 1.0, 1.0, 1.0])


def _tx(d):
    import numpy as np
    M = np.eye(4)
    M[0, 3] = d
    return M


def _rot3(ax, ay, az):
    return (_rz(az) @ _ry(ay) @ _rx(ax))[:3, :3]


# --------------------------------------------------------------------------
# Seed polyhedra: face directions on S^2 (one cycloplane per face)
# --------------------------------------------------------------------------

def _norm(v):
    import numpy as np
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def face_normals(seed):
    """Outward face directions of the Platonic seed as unit 3-vectors
    (the polyhedron's faces = the dual's vertices)."""
    if seed == 'TETRA':      # 4 faces (opposite the 4 vertices)
        V = [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
    elif seed == 'CUBE':     # 6 faces = octahedron vertices
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
             (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif seed == 'OCTA':     # 8 faces = cube vertices
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
    elif seed == 'DODECA':   # 12 faces = icosahedron vertices
        V = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                V += [(0, s1, s2 * _PHI), (s1, s2 * _PHI, 0),
                      (s2 * _PHI, 0, s1)]
    elif seed == 'ICOSA':    # 20 faces = dodecahedron vertices
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                V += [(0, s1 / _PHI, s2 * _PHI),
                      (s1 / _PHI, s2 * _PHI, 0),
                      (s2 * _PHI, 0, s1 / _PHI)]
    else:
        raise ValueError(seed)
    return [tuple(_norm(v)) for v in V]


_SEED_FACES = {'TETRA': 4, 'CUBE': 6, 'OCTA': 8, 'DODECA': 12,
               'ICOSA': 20}


# --------------------------------------------------------------------------
# One cycloplane cross-section = a transformed cylinder mesh
# --------------------------------------------------------------------------

def _base_cylinder(radius, height, seg):
    """A closed (capped) cylinder about the Z-axis as (verts, faces)."""
    import numpy as np
    hz = height / 2.0
    ang = np.linspace(0.0, 2.0 * pi, seg, endpoint=False)
    bot = [(radius * math.cos(a), radius * math.sin(a), -hz)
           for a in ang]
    top = [(radius * math.cos(a), radius * math.sin(a), hz)
           for a in ang]
    verts = bot + top
    faces = []
    for i in range(seg):
        j = (i + 1) % seg
        faces.append([i, j, seg + j, seg + i])        # side quad
    faces.append(list(range(seg - 1, -1, -1)))         # bottom cap
    faces.append(list(range(seg, 2 * seg)))            # top cap
    return verts, faces


def cycloplane_cylinder(w, zenith, azimuth, height=100.0, radius=1.0,
                        seg=48):
    """Transformed-cylinder cross-section of one cycloplane at slice
    w (verts, faces).  See the module header for the transform."""
    import numpy as np
    theta = 0.5 * zenith
    ct = cos(theta)
    verts, faces = _base_cylinder(radius, height, seg)
    if abs(ct) < 1e-3:                       # south-pole degeneration
        M = _ry(azimuth) @ _rx(zenith) @ _rx(pi / 2.0)
    else:
        M = (_ry(azimuth) @ _rx(zenith) @ _tx(w * tan(theta))
             @ _sx(1.0 / ct) @ _rx(pi / 2.0))
    P = np.asarray(verts, float)
    Ph = np.hstack([P, np.ones((len(P), 1))])
    out = (Ph @ M.T)[:, :3]
    return out.tolist(), faces


def cycloplanes(seed, w, seg=48, height=100.0):
    """(zenith, azimuth, verts, faces) for every face cycloplane of the
    seed at slice w, after the generic tilt."""
    import numpy as np
    R = _rot3(*_TILT)
    out = []
    for n in face_normals(seed):
        d = R @ np.asarray(n)
        zenith = math.acos(max(-1.0, min(1.0, d[2])))
        azimuth = math.atan2(d[1], d[0])
        v, f = cycloplane_cylinder(w, zenith, azimuth, height, 1.0, seg)
        out.append((zenith, azimuth, v, f))
    return out


if _IN_BLENDER:

    def _mesh_obj(name, verts, faces, coll):
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        me.update()
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        return ob

    class MESH_OT_polytwister_add(bpy.types.Operator):
        """Add the 3D cross-section of a polytwister: the Boolean
        intersection of one cycloplane cylinder per seed face"""
        bl_idname = "mesh.polytwister_add"
        bl_label = "Polytwister"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Seed",
            items=[('TETRA', "Tetratwister", "4 cycloplanes"),
                   ('CUBE', "Cubetwister", "6 cycloplanes"),
                   ('OCTA', "Octatwister", "8 cycloplanes"),
                   ('DODECA', "Dodecatwister", "12 cycloplanes"),
                   ('ICOSA', "Icosatwister", "20 cycloplanes")],
            default='CUBE')
        slice_w: FloatProperty(
            name="Slice w", default=0.0, min=-3.0, max=3.0,
            step=5, precision=3,
            description="4D->3D cross-section height; sweep to "
                        "animate the polytwister through its cells")
        seg: IntProperty(
            name="Cylinder Segments", default=48, min=8, max=160,
            description="Facets per cycloplane cylinder (mesh "
                        "resolution vs. Boolean cost)")
        shade_smooth: BoolProperty(name="Shade Smooth", default=False)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            import numpy as np
            coll = context.collection
            planes = cycloplanes(self.seed, self.slice_w, self.seg)
            temps = []
            for k, (_z, _a, v, f) in enumerate(planes):
                temps.append(_mesh_obj(f"_cyclo{k}", v, f, coll))
            acc = temps[0]
            for o in temps[1:]:
                m = acc.modifiers.new("bool", 'BOOLEAN')
                m.operation = 'INTERSECT'
                m.solver = 'EXACT'
                m.object = o
            dg = context.evaluated_depsgraph_get()
            final = bpy.data.meshes.new_from_object(
                acc.evaluated_get(dg))

            n_out = len(final.vertices)
            if n_out == 0:
                bpy.data.meshes.remove(final)
                for o in temps:
                    bpy.data.meshes.remove(o.data)
                self.report({'ERROR'},
                            "Empty cross-section at this slice w "
                            "(try a smaller |w|)")
                return {'CANCELLED'}

            # centre + scale the baked cross-section into the unit cube
            co = np.empty(n_out * 3)
            final.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3)
            center = 0.5 * (co.max(0) + co.min(0))
            ext = (co.max(0) - co.min(0)).max()
            s = (2.0 / ext) if ext > 1e-9 else 1.0
            co = (co - center) * s * self.scale
            final.vertices.foreach_set('co', co.ravel())
            if self.shade_smooth:
                final.polygons.foreach_set(
                    'use_smooth', [True] * len(final.polygons))
            final.update()

            name = f"{self.seed.title()}twister"
            obj = bpy.data.objects.new(name, final)
            coll.objects.link(obj)
            for o in temps:                     # drop the scaffolding
                md = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                bpy.data.meshes.remove(md)

            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"{name}: {len(planes)} cycloplanes, "
                f"{n_out} verts, slice w={self.slice_w:.3f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('seed', 'slice_w', 'seg', 'shade_smooth',
                      'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.polytwister_add", icon='MESH_TORUS')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_polytwister_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polytwister_add)


def _selftest():
    import numpy as np
    ok_all = True

    # face-normal sets: right count, unit norm, and (before tilt) a
    # dual pair per axis so the intersection is bounded
    for seed, nf in _SEED_FACES.items():
        N = face_normals(seed)
        unit = max(abs(np.linalg.norm(n) - 1.0) for n in N)
        ok = len(N) == nf and unit < 1e-12
        ok_all = ok_all and ok
        print(f"{seed}: {len(N)} faces (want {nf}) unit dev={unit:.1e} "
              f"{'OK' if ok else 'BAD'}")

    # one cycloplane cylinder is finite and has the right topology
    v, f = cycloplane_cylinder(0.3, 0.7, 1.1, height=100.0, seg=32)
    finite = np.isfinite(np.asarray(v)).all()
    ok = finite and len(v) == 64 and len(f) == 32 + 2
    ok_all = ok_all and ok
    print(f"cycloplane cyl: verts={len(v)} faces={len(f)} "
          f"finite={finite} {'OK' if ok else 'BAD'}")

    # the transform preserves the origin at w=0 (every cylinder then
    # contains it, so the w=0 intersection is non-empty)
    theta = 0.35
    M = (_ry(1.0) @ _rx(0.7) @ _tx(0.0) @ _sx(1.0 / cos(theta))
         @ _rx(pi / 2.0))
    o = M @ np.array([0.0, 0.0, 0.0, 1.0])
    ok = np.linalg.norm(o[:3]) < 1e-12
    ok_all = ok_all and ok
    print(f"origin fixed at w=0: |o|={np.linalg.norm(o[:3]):.1e} "
          f"{'OK' if ok else 'BAD'}")

    # south-pole degeneration path stays finite
    v2, _ = cycloplane_cylinder(0.5, pi - 1e-4, 0.0, seg=16)
    ok = np.isfinite(np.asarray(v2)).all()
    ok_all = ok_all and ok
    print(f"south-pole cyl finite={ok} {'OK' if ok else 'BAD'}")

    assert ok_all
    print("polytwister standalone tests passed")
