
# Frieze Group Generator for Blender
#
# The 7 frieze groups: strip patterns that repeat in ONE direction.
# A frieze group is a wallpaper group restricted to a single
# translation axis, so every symmetry combination of translation,
# glide, 180 rotation, and vertical/horizontal mirror collapses into
# exactly 7 types (hop, step, sidle, jump, spinning hop/sidle/jump).
#
# Same machinery as the wallpaper generator: a chiral motif (or the
# active mesh) is replicated along the strip, colored so the symmetry
# is evident, optionally extruded into a relief.  Part of the Pattern
# Engine (see pattern_common.py).
#
# References:
#   The 7 frieze (strip) groups -- classical crystallographic result.
#   John H. Conway, Heidi Burgiel & Chaim Goodman-Strauss, "The
#     Symmetries of Things" (2008) -- the orbifold signature notation
#     (and the hop/step/sidle/jump naming).

bl_info = {
    "name": "Frieze Group",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "The 7 frieze (strip) groups as colored motif "
                   "patterns",
    "category": "Add Mesh",
}

import numpy as np

try:
    from . import pattern_common as pc
except Exception:
    import pattern_common as pc


def build(group='p2mm', motif_kind='ARROW', reps=6, color_by='COPY',
          height=0.0, margin=0.0):
    """Merged (verts, faces, mats) for a frieze strip of `reps` cells."""
    return pc.merge_cells(build_cells(group, motif_kind, reps,
                                      color_by, height, margin))


def build_cells(group='p2mm', motif_kind='ARROW', reps=6,
                color_by='COPY', height=0.0, margin=0.0):
    """One cell per replicated motif copy along the strip."""
    cosets = pc.frieze_group(group)
    polys = pc.motif(motif_kind)
    if margin > 0.0:
        f = 1.0 / (1.0 + margin)
        polys = [0.5 + (p - 0.5) * f for p in polys]
    cells = []
    for i in range(reps):
        base = pc.T(float(i), 0.0)
        for gi, g in enumerate(cosets):
            M = base @ g
            kind = pc.kind_of(color_by, M, g, i, gi)
            cv, cf, cm = [], [], []
            for p in polys:
                q = pc.apply(M, p)
                if height > 0.0:
                    pc.prisms(cv, cf, cm, [q], height, 0.0, kind)
                else:
                    b0 = len(cv)
                    for x, y in q:
                        cv.append((float(x), float(y), 0.0))
                    cf.append(tuple(range(b0, b0 + len(q))))
                    cm.append(kind)
            cells.append((cv, cf, cm))
    return cells


def build_active(group, verts3, faces, reps=6, color_by='COPY',
                 margin=0.0):
    """Merged active mesh tiled along the frieze strip."""
    return pc.merge_cells(build_active_cells(
        group, verts3, faces, reps, color_by, margin))


def build_active_cells(group, verts3, faces, reps=6, color_by='COPY',
                       margin=0.0):
    """One cell per copy of the active mesh along the strip at its
    ORIGINAL size; frieze operations are axis-aligned, so a rectangular
    mesh tiles flush along x."""
    V = np.asarray(verts3, dtype=float)
    lo = V[:, :2].min(axis=0)
    wx, wy = np.maximum(V[:, :2].max(axis=0) - lo, 1e-6)
    sx, sy = wx * (1.0 + margin), wy
    U = np.column_stack([(V[:, 0] - lo[0]) / sx, (V[:, 1] - lo[1]) / sy])
    cosets = pc.frieze_group(group)
    cells = []
    for i in range(reps):
        base = pc.T(float(i), 0.0)
        for gi, g in enumerate(cosets):
            M = base @ g
            kind = pc.kind_of(color_by, M, g, i, gi)
            uxy = pc.apply(M, U)
            cv = [(uxy[k, 0] * sx, uxy[k, 1] * sy, float(V[k, 2]))
                  for k in range(len(U))]
            cells.append((cv, [tuple(f) for f in faces],
                          [kind] * len(faces)))
    return cells


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_frieze_add(bpy.types.Operator, AddObjectHelper):
        """Add a frieze (strip) pattern"""
        bl_idname = "mesh.frieze_add"
        bl_label = "Frieze Group"
        bl_options = {'REGISTER', 'UNDO'}

        group: EnumProperty(
            name="Group",
            items=[(k, "%s  (%s)" % (pc.FRIEZE_NAMES[k][0], k),
                    "Frieze %s, orbifold %s" %
                    (k, pc.FRIEZE_NAMES[k][1]))
                   for k in pc.FRIEZE_ORDER],
            default='p2mm')
        motif_kind: EnumProperty(
            name="Motif",
            items=[('ARROW', "Arrow", "Arrow with unequal barbs"),
                   ('F', "F", "Asymmetric F"),
                   ('L', "L", "L-tromino"),
                   ('COMMA', "Comma", "Chiral paisley"),
                   ('ZIG', "Zig", "Z / zigzag"),
                   ('TRIANGLE', "Triangle", "Scalene triangle"),
                   ('ACTIVE', "Active Mesh",
                    "Use the selected mesh as the unit; falls back to "
                    "the default motif if no mesh is selected")],
            default='ARROW')
        reps: IntProperty(name="Repeats", default=6, min=1, max=60)
        color_by: EnumProperty(
            name="Color By",
            items=[('COPY', "Symmetry Copy",
                    "Each frieze copy a distinct color"),
                   ('OP', "Operation Type",
                    "Identity / rotation / reflection / glide"),
                   ('HAND', "Handedness",
                    "Direct vs. mirrored copies"),
                   ('CELL', "Repeat",
                    "One color per repeat along the strip")],
            default='COPY')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=3.0,
            description="Spacing around each unit, as a fraction of "
                        "its size (0 = flush)")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=1.0,
            description="0 = flat 2D mesh; > 0 extrudes into a relief")
        separate: BoolProperty(
            name="Separate Cells", default=False,
            description="Output each unit as its own mesh object "
                        "(parented to an empty) so cells can be edited "
                        "individually")

        def execute(self, context):
            src = None
            if self.motif_kind == 'ACTIVE':
                for o in ([context.active_object]
                          + list(context.selected_objects)):
                    if (o and o.type == 'MESH' and o.data.vertices
                            and o.data.polygons
                            and not o.get("math_art_pattern")):
                        src = o
                        break
            fit = True
            if self.motif_kind == 'ACTIVE' and src is not None:
                v3 = [(vv.co.x, vv.co.y, vv.co.z)
                      for vv in src.data.vertices]
                fc = [tuple(p.vertices) for p in src.data.polygons]
                cells = build_active_cells(
                    self.group, v3, fc, self.reps, self.color_by,
                    self.margin)
                fit = False
            else:
                kind = 'ARROW' if self.motif_kind == 'ACTIVE' \
                    else self.motif_kind
                cells = build_cells(
                    self.group, kind, self.reps, self.color_by,
                    self.height, self.margin)
            obj = pc.emit(context, "Frieze %s" % self.group, cells,
                          self.separate, fit=fit, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no pattern generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s  V=%d F=%d" %
                            (self.group, len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d cells" %
                            (self.group, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for p in ('group', 'motif_kind', 'reps', 'color_by',
                      'margin'):
                lay.prop(self, p)
            if self.motif_kind != 'ACTIVE':
                lay.prop(self, 'height')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.frieze_add", icon='MOD_ARRAY')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_frieze_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_frieze_add)


def _selftest():
    bad = []
    for g in pc.FRIEZE_ORDER:
        v, f, m = build(g, 'ARROW', 4)
        z0 = all(abs(vt[2]) < 1e-12 for vt in v)
        ok = len(f) > 0 and len(m) == len(f) and z0
        print("%-5s V=%d F=%d copies=%d %s" %
              (g, len(v), len(f), max(m) + 1, "OK" if ok else "BAD"))
        if not ok:
            bad.append(g)
    sq_v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0.3)]
    sq_f = [(0, 1, 2, 3)]
    v, f, m = build_active('p2mg', sq_v, sq_f, 3)
    print("active p2mg V=%d F=%d %s" %
          (len(v), len(f), "OK" if f else "BAD"))
    print("RESULT:", "OK" if not bad else "BAD %s" % bad)
    assert not bad
