
# Layer Group Generator for Blender
#
# The 3D extension of the wallpaper groups.  A layer group is a pattern
# periodic in two directions (a slab) whose symmetries may involve the
# THIRD axis -- a horizontal mirror through the layer plane, a 2-fold
# axis lying in the plane, or a centre of inversion.  There are 80 in
# full; this generator takes the tractable "z-augmentation" route:
# pick one of the 17 in-plane wallpaper groups and add a z-symmetry,
# so copies can flip up and down, not just rotate/reflect in-plane.
#
# Because the whole point is the third dimension, a layer pattern only
# reads in 3D: use a relief height (built-in motifs) or the active mesh
# as the unit.  A z-mirror then produces a top/bottom-symmetric screen.
# Part of the Pattern Engine (see the `patterns` package).
#
# References:
# - The 80 layer (diperiodic) groups -- "International Tables for
#   Crystallography", Volume E: Subperiodic Groups.
# - John H. Conway, Heidi Burgiel & Chaim Goodman-Strauss, "The
#   Symmetries of Things" (2008) -- the wallpaper-group / orbifold
#   foundation this z-augmentation builds on.

bl_info = {
    "name": "Layer Group",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Wallpaper groups augmented with a z-symmetry "
                   "(the 3D layer groups)",
    "category": "Add Mesh",
}

from math import pi

import numpy as np

try:
    from .patterns import common as pc
except Exception:
    from patterns import common as pc


# groups whose in-plane operations keep x and y separate (rectangular
# lattice), so a rectangular active mesh tiles flush
_AXIS_ALIGNED = {'p1', 'p2', 'pm', 'pg', 'pmm', 'pmg', 'pgg'}

ZSYMS = ('NONE', 'MIRROR', 'TWOFOLD', 'INVERSION')


def _z_copies(zsym):
    """The z-augmentation as a list of (in-plane extra transform,
    z-sign): the identity copy plus, optionally, a z-inverted copy
    composed with an in-plane operation."""
    if zsym == 'MIRROR':                       # horizontal mirror m_z
        return [(pc.I(), 1.0), (pc.I(), -1.0)]
    if zsym == 'TWOFOLD':                      # 2-fold axis along x
        return [(pc.I(), 1.0), (pc.Mir(0.0), -1.0)]
    if zsym == 'INVERSION':                    # centre of inversion
        return [(pc.I(), 1.0), (pc.Rot(pi), -1.0)]
    return [(pc.I(), 1.0)]                      # NONE


def _kind(color_by, M, g, cell, gi, zsign):
    if color_by == 'HAND':                     # 3D chirality: include z
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        return 0 if det * zsign > 0 else 1
    if color_by == 'OP':
        return pc.iso_type(g)
    if color_by == 'CELL':
        return cell % len(pc.PALETTE_RGBA)   # wrap: avoid a material per cell
    return gi


def build(group='p4', zsym='MIRROR', motif_kind='ARROW', nx=3, ny=3,
          color_by='COPY', height=0.3, margin=0.0):
    """Merged (verts, faces, mats): an in-plane wallpaper relief
    augmented with a z-symmetry."""
    return pc.merge_cells(build_cells(group, zsym, motif_kind, nx, ny,
                                      color_by, height, margin))


def build_cells(group='p4', zsym='MIRROR', motif_kind='ARROW', nx=3,
                ny=3, color_by='COPY', height=0.3, margin=0.0):
    """One cell per (in-plane copy x z-copy)."""
    b1, b2, cosets = pc.wallpaper_group(pc.SIG_OF[group])
    polys = pc.motif(motif_kind)
    if margin > 0.0:
        f = 1.0 / (1.0 + margin)
        polys = [0.5 + (p - 0.5) * f for p in polys]
    zc = _z_copies(zsym)
    cells = []
    for i in range(nx):
        for j in range(ny):
            cell = i * ny + j
            latt = pc.T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for gi, g in enumerate(cosets):
                M0 = latt @ g
                for xe, zsign in zc:
                    M = xe @ M0
                    kind = _kind(color_by, M, g, cell, gi, zsign)
                    cv, cf, cm = [], [], []
                    for p in polys:
                        q = pc.apply(M, p)
                        pc.prisms(cv, cf, cm, [q],
                                  height * zsign, 0.0, kind)
                    cells.append((cv, cf, cm))
    return cells


def build_active(group, zsym, verts3, faces, nx=3, ny=3,
                 color_by='COPY', margin=0.0):
    """Merged active mesh tiled by the in-plane group and its
    z-augmentation."""
    return pc.merge_cells(build_active_cells(
        group, zsym, verts3, faces, nx, ny, color_by, margin))


def build_active_cells(group, zsym, verts3, faces, nx=3, ny=3,
                       color_by='COPY', margin=0.0):
    """One cell per (in-plane copy x z-copy) of the active mesh at its
    ORIGINAL size; a z-inverted copy flips the mesh through the plane."""
    V = np.asarray(verts3, dtype=float)
    lo = V[:, :2].min(axis=0)
    wx, wy = np.maximum(V[:, :2].max(axis=0) - lo, 1e-6)
    if group in _AXIS_ALIGNED:
        sx, sy = wx, wy
    else:
        sx = sy = max(wx, wy)
    sx *= 1.0 + margin
    sy *= 1.0 + margin
    U = np.column_stack([(V[:, 0] - lo[0]) / sx, (V[:, 1] - lo[1]) / sy])
    vz = V[:, 2] - V[:, 2].min()               # rest the mesh on z = 0
    b1, b2, cosets = pc.wallpaper_group(pc.SIG_OF[group])
    zc = _z_copies(zsym)
    cells = []
    for i in range(nx):
        for j in range(ny):
            cell = i * ny + j
            latt = pc.T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for gi, g in enumerate(cosets):
                M0 = latt @ g
                for xe, zsign in zc:
                    M = xe @ M0
                    kind = _kind(color_by, M, g, cell, gi, zsign)
                    uxy = pc.apply(M, U)
                    cv = [(uxy[k, 0] * sx, uxy[k, 1] * sy,
                           float(vz[k] * zsign)) for k in range(len(U))]
                    cells.append((cv, [tuple(f) for f in faces],
                                  [kind] * len(faces)))
    return cells


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, StringProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_layer_add(bpy.types.Operator, AddObjectHelper):
        """Add a layer-group pattern: a wallpaper group augmented with
        a z-symmetry (shows only in 3D -- use relief or an active mesh)"""
        bl_idname = "mesh.layer_add"
        bl_label = "Layer Group"
        bl_options = {'REGISTER', 'UNDO'}

        group: EnumProperty(
            name="In-plane Group",
            items=[(g, "%s  (%s)" % (g, pc.SIG_OF[g]),
                    "In-plane wallpaper group %s" % g)
                   for g in pc.IUC_ORDER],
            default='p4')
        zsym: EnumProperty(
            name="Z-Symmetry",
            items=[('NONE', "None",
                    "In-plane pattern only (no third-axis symmetry)"),
                   ('MIRROR', "Horizontal Mirror",
                    "Mirror through the layer plane (top/bottom "
                    "symmetric)"),
                   ('TWOFOLD', "2-fold In-plane Axis",
                    "180 rotation about an in-plane axis"),
                   ('INVERSION', "Inversion Centre",
                    "Centre of symmetry")],
            default='MIRROR')
        motif_kind: EnumProperty(
            name="Motif",
            items=[('ARROW', "Arrow", "Arrow with unequal barbs"),
                   ('F', "F", "Asymmetric F"),
                   ('L', "L", "L-tromino"),
                   ('COMMA', "Comma", "Chiral paisley"),
                   ('ZIG', "Zig", "Z / zigzag"),
                   ('TRIANGLE', "Triangle", "Scalene triangle"),
                   ('ACTIVE', "Object",
                    "Use another object's mesh as the unit. Pick it below; "
                    "with nothing picked the active or selected mesh is "
                    "used, and failing that the default motif")],
            default='ARROW')
        source: StringProperty(
            name="Object",
            description="Object whose mesh is used as the motif. Leave "
                        "empty to use the active or selected mesh")
        nx: IntProperty(name="Cells X", default=3, min=1, max=30)
        ny: IntProperty(name="Cells Y", default=3, min=1, max=30)
        color_by: EnumProperty(
            name="Color By",
            items=[('COPY', "Symmetry Copy",
                    "Each point-group copy a distinct color"),
                   ('OP', "Operation Type",
                    "Identity / rotation / reflection / glide"),
                   ('HAND', "Handedness",
                    "3D chirality (includes the z-flip)"),
                   ('CELL', "Lattice Cell",
                    "One color per unit cell")],
            default='COPY')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=3.0,
            description="Spacing around each unit, as a fraction of "
                        "its size (0 = flush)")
        height: FloatProperty(
            name="Relief Height", default=0.3, min=0.02, max=2.0,
            description="Relief depth for built-in motifs (a layer "
                        "pattern needs real 3D height to show)")
        separate: BoolProperty(
            name="Separate Cells", default=False,
            description="Output each unit as its own mesh object "
                        "(parented to an empty) so cells can be edited "
                        "individually")

        def invoke(self, context, event):
            # Start from whatever is active, so choosing "Object" does
            # something immediately rather than needing a pick first.
            if not self.source and context.active_object is not None:
                src = pc.source_mesh(context, "")
                if src is not None:
                    self.source = src.name
            return self.execute(context)

        def execute(self, context):
            src = None
            if self.motif_kind == 'ACTIVE':
                src = pc.source_mesh(context, self.source)
                if src is None and self.source:
                    self.report({'WARNING'},
                                "%r is not a usable mesh; using the default "
                                "motif" % self.source)
            fit = True
            if self.motif_kind == 'ACTIVE' and src is not None:
                v3 = [(vv.co.x, vv.co.y, vv.co.z)
                      for vv in src.data.vertices]
                fc = [tuple(p.vertices) for p in src.data.polygons]
                cells = build_active_cells(
                    self.group, self.zsym, v3, fc, self.nx, self.ny,
                    self.color_by, self.margin)
                fit = False
            else:
                kind = 'ARROW' if self.motif_kind == 'ACTIVE' \
                    else self.motif_kind
                cells = build_cells(
                    self.group, self.zsym, kind, self.nx, self.ny,
                    self.color_by, self.height, self.margin)
            obj = pc.emit(
                context, "Layer %s %s" % (self.group, self.zsym),
                cells, self.separate, fit=fit, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no pattern generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s/%s  V=%d F=%d" %
                            (self.group, self.zsym,
                             len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s/%s  %d cells" %
                            (self.group, self.zsym, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for p in ('group', 'zsym', 'motif_kind', 'nx', 'ny',
                      'color_by', 'margin'):
                lay.prop(self, p)
            if self.motif_kind == 'ACTIVE':
                lay.prop_search(self, 'source', bpy.data, 'objects')
            if self.motif_kind != 'ACTIVE':
                lay.prop(self, 'height')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.layer_add", icon='MOD_SOLIDIFY')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_layer_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_layer_add)


def _selftest():
    bad = []
    for z in ZSYMS:
        v, f, m = build('p4', z, 'ARROW', 2, 2)
        zspan = (max(p[2] for p in v) - min(p[2] for p in v)) if v else 0
        # NONE keeps one side; the z-augmentations span both sides
        ok = len(f) > 0 and (zspan > 0)
        print("%-10s V=%d F=%d zspan=%.3f %s" %
              (z, len(v), len(f), zspan, "OK" if ok else "BAD"))
        if not ok:
            bad.append(z)
    cube_v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    cube_f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
              (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    v, f, m = build_active('p2', 'MIRROR', cube_v, cube_f, 2, 2)
    print("active p2/MIRROR V=%d F=%d %s" %
          (len(v), len(f), "OK" if f else "BAD"))
    print("RESULT:", "OK" if not bad else "BAD %s" % bad)
    assert not bad
