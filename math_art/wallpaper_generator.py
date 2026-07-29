
# Wallpaper Group Generator for Blender
#
# Euclidean plane-symmetry patterns driven by Conway-Thurston ORBIFOLD
# signatures.  A small asymmetric motif in one fundamental domain is
# replicated by every isometry of a chosen wallpaper group to fill an
# Nx x Ny block of the lattice.
#
# Output is a FLAT 2D mesh by default: each replicated copy of the
# motif is one flat face, and the faces are colored so the symmetry
# is evident -- either by "unit" (each of the point group's copies a
# distinct color, repeated identically in every cell, so the repeat
# unit reads at a glance) or by orientation (direct vs. mirrored, so
# reflections and glides pop).  An optional height extrudes the flat
# faces into a relief.
#
# The 17 wallpaper groups are selected by orbifold signature; the same
# signature language (pattern_common.geometry_of) routes to the
# spherical and hyperbolic worlds -- one grammar, three geometries.
# Part of the Pattern Engine (see pattern_common.py).
#
# References:
#   E. S. Fedorov (1891); George Polya and Paul Niggli (1924) -- the
#     classification of the 17 wallpaper (plane crystallographic) groups.
#   John H. Conway, Heidi Burgiel & Chaim Goodman-Strauss, "The
#     Symmetries of Things" (2008) -- the orbifold signature notation
#     used here.

bl_info = {
    "name": "Wallpaper Group",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "The 17 wallpaper groups as flat colored motif "
                   "patterns, chosen by orbifold signature",
    "category": "Add Mesh",
}

import numpy as np

try:
    from . import pattern_common as pc
except Exception:
    import pattern_common as pc


# The motif library, color classifier and group tables live in
# pattern_common so the wallpaper, frieze and layer generators share
# one implementation.  Keep the familiar local names as thin aliases.
motif = pc.motif
MOTIFS = pc.MOTIFS
_kind = pc.kind_of
_SIG = pc.SIG_OF
IUC_ORDER = pc.IUC_ORDER


def build(group='p4m', motif_kind='F', nx=3, ny=3, color_by='COPY',
          height=0.0, margin=0.0):
    """Build (verts, faces, mats) for a wallpaper pattern.

    color_by 'COPY' -> by point-group coset (symmetry copy)
             'OP'   -> by isometry type (identity/rotation/mirror/glide)
             'HAND' -> by chirality (direct vs mirrored)
             'CELL' -> by lattice cell (shows the translation lattice)
    height   0.0    -> flat 2D faces at z = 0
             > 0.0  -> extrude each face into a relief prism
    margin          -> extra space around each unit, as a fraction of
                       its size (0 = the motif at its drawn size)
    """
    return pc.merge_cells(build_cells(group, motif_kind, nx, ny,
                                      color_by, height, margin))


def build_cells(group='p4m', motif_kind='F', nx=3, ny=3,
                color_by='COPY', height=0.0, margin=0.0):
    """One (verts, faces, mats) cell per replicated motif copy."""
    sig = _SIG[group]
    b1, b2, cosets = pc.wallpaper_group(sig)
    polys = motif(motif_kind)
    if margin > 0.0:
        f = 1.0 / (1.0 + margin)              # shrink about the cell centre
        polys = [0.5 + (p - 0.5) * f for p in polys]
    cells = []
    for i in range(nx):
        for j in range(ny):
            cell = i * ny + j
            base = pc.T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for gi, g in enumerate(cosets):
                M = base @ g
                kind = _kind(color_by, M, g, cell, gi)
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


# groups whose symmetry operations keep the x and y axes separate, so
# the cell can be scaled independently per axis -- a rectangular active
# mesh then tiles perfectly flush regardless of its aspect ratio
_AXIS_ALIGNED = {'p1', 'p2', 'pm', 'pg', 'pmm', 'pmg', 'pgg'}


def build_active(group, verts3, faces, nx=3, ny=3, color_by='COPY',
                 margin=0.0):
    """Merged (verts, faces, mats) of the active mesh tiled by the
    group at its original size."""
    return pc.merge_cells(build_active_cells(
        group, verts3, faces, nx, ny, color_by, margin))


def build_active_cells(group, verts3, faces, nx=3, ny=3,
                       color_by='COPY', margin=0.0):
    """One cell per replicated copy of the active mesh, tiled at its
    ORIGINAL size (no rescaling); the lattice spacing is the mesh's own
    XY bounding box (times 1 + margin), so with margin=0 the copies
    tile edge to edge.  Z is carried through, so reflections become
    real mirror-image copies."""
    V = np.asarray(verts3, dtype=float)
    lo = V[:, :2].min(axis=0)
    wx, wy = np.maximum(V[:, :2].max(axis=0) - lo, 1e-6)
    if group in _AXIS_ALIGNED:
        sx, sy = wx, wy                       # flush at any aspect ratio
    else:
        sx = sy = max(wx, wy)                 # square/hex cell for rotations
    sx *= 1.0 + margin
    sy *= 1.0 + margin
    # express the mesh in unit-cell coordinates (bbox min -> origin);
    # scaling back by (sx, sy) at the end restores the original size
    U = np.column_stack([(V[:, 0] - lo[0]) / sx, (V[:, 1] - lo[1]) / sy])
    sig = _SIG[group]
    b1, b2, cosets = pc.wallpaper_group(sig)
    cells = []
    for i in range(nx):
        for j in range(ny):
            cell = i * ny + j
            base = pc.T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for gi, g in enumerate(cosets):
                M = base @ g
                kind = _kind(color_by, M, g, cell, gi)
                uxy = pc.apply(M, U)
                cv = [(uxy[k, 0] * sx, uxy[k, 1] * sy, float(V[k, 2]))
                      for k in range(len(U))]
                cf = [tuple(f) for f in faces]
                cells.append((cv, cf, [kind] * len(faces)))
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

    class MESH_OT_wallpaper_add(bpy.types.Operator, AddObjectHelper):
        """Add a wallpaper-group pattern (flat colored motif, chosen
        by orbifold signature)"""
        bl_idname = "mesh.wallpaper_add"
        bl_label = "Wallpaper Group"
        bl_options = {'REGISTER', 'UNDO'}

        group: EnumProperty(
            name="Group",
            items=[(g, "%s  (%s)" % (g, _SIG[g]),
                    "Wallpaper group %s, orbifold %s" % (g, _SIG[g]))
                   for g in IUC_ORDER],
            default='p4m')
        motif_kind: EnumProperty(
            name="Motif",
            items=[('ARROW', "Arrow", "Arrow with unequal barbs"),
                   ('F', "F", "Asymmetric F"),
                   ('L', "L", "L-tromino"),
                   ('COMMA', "Comma", "Chiral paisley"),
                   ('ZIG', "Zig", "Z / zigzag"),
                   ('TRIANGLE', "Triangle", "Scalene triangle"),
                   ('ACTIVE', "Active Mesh",
                    "Use the selected mesh object as the unit, tiled "
                    "in 3D by the group; falls back to the default "
                    "motif if no mesh is selected")],
            default='ARROW')
        nx: IntProperty(name="Cells X", default=3, min=1, max=30)
        ny: IntProperty(name="Cells Y", default=3, min=1, max=30)
        color_by: EnumProperty(
            name="Color By",
            items=[('COPY', "Symmetry Copy",
                    "Each point-group copy a distinct color "
                    "(pinwheel around rotation centres)"),
                   ('OP', "Operation Type",
                    "Identity / rotation / reflection / glide -- "
                    "shows the group's generators"),
                   ('HAND', "Handedness",
                    "Direct vs. mirrored copies (chirality)"),
                   ('CELL', "Lattice Cell",
                    "One color per unit cell (the translation "
                    "lattice)")],
            default='COPY')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=3.0,
            description="Spacing around each unit, as a fraction of "
                        "its size (0 = flush)")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=1.0,
            description="0 = flat 2D mesh; > 0 extrudes the faces "
                        "into a relief")
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
                    self.group, v3, fc, self.nx, self.ny,
                    self.color_by, self.margin)
                fit = False                   # keep the mesh's true size
            else:
                # built-in motif, or the default fallback when ACTIVE
                # is chosen with no suitable mesh selected
                kind = 'ARROW' if self.motif_kind == 'ACTIVE' \
                    else self.motif_kind
                cells = build_cells(
                    self.group, kind, self.nx, self.ny,
                    self.color_by, self.height, self.margin)
            obj = pc.emit(context, "Wallpaper %s" % self.group, cells,
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
            for p in ('group', 'motif_kind', 'nx', 'ny', 'color_by',
                      'margin'):
                lay.prop(self, p)
            if self.motif_kind != 'ACTIVE':     # active mesh is 3D
                lay.prop(self, 'height')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')             # World / 3D Cursor / View

    def _menu_func(self, context):
        self.layout.operator("mesh.wallpaper_add", icon='MOD_MIRROR')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_wallpaper_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_wallpaper_add)


if __name__ == "__main__":
    bad = []
    for g in IUC_ORDER:
        v, f, m = build(g, 'F', 2, 2)
        z0 = all(abs(vt[2]) < 1e-12 for vt in v)
        ok = len(f) > 0 and len(m) == len(f) and z0
        print("%-5s V=%d F=%d units=%d %s" %
              (g, len(v), len(f), max(m) + 1, "OK" if ok else "BAD"))
        if not ok:
            bad.append(g)
    # color modes and the active-mesh path (a unit square motif)
    for cb in ('COPY', 'OP', 'HAND', 'CELL'):
        v, f, m = build('p4m', 'COMMA', 2, 2, color_by=cb)
        print("color %-5s p4m colors=%d %s" %
              (cb, max(m) + 1, "OK" if f else "BAD"))
    sq_v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0.3)]
    sq_f = [(0, 1, 2, 3)]
    v, f, m = build_active('p6', sq_v, sq_f, 2, 2)
    print("active-mesh p6 V=%d F=%d %s" %
          (len(v), len(f), "OK" if f else "BAD"))
    print("RESULT:", "OK" if not bad else "BAD %s" % bad)
