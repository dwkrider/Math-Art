# The classical crease patterns, and folding them.
#
# Two operators over `math_art/crease/`:
#
#   mesh.crease_pattern_add   builds a classical pattern flat
#   object.fold_solve         folds the active pattern, caching the whole
#                             fold path as shape keys so it animates
#
# ONE OPERATOR, ONE DIAL.  The seven classical patterns are one entry
# with a Pattern selector rather than seven menu entries, matching the
# way CMC surfaces and the Hopf/Willmore tori are each a single entry:
# they are one construction with one dial, and seven entries for seven
# parameterisations of "a periodic crease pattern" would be clutter.
#
# WHY SHIP PATTERNS AT ALL, when the add-on does not design crease
# patterns?  Because the scope rule is about substitutability, and these
# are folk mathematics -- no author to credit, no copyright to infringe,
# nothing to download.  Without them the add-on could open every crease
# pattern in the world and produce none.
#
# HOW THE ANIMATION WORKS.  A fold is not a cheap function of a
# parameter: each state costs a Newton solve.  So the path is solved
# ONCE by continuation, every state is stored as a shape key, and a
# single custom property `fold_t` drives them through hat-function
# drivers.  Keyframe `fold_t` and the model folds; nothing re-solves at
# render time.
#
# References:
#   K. Miura, "Method of Packaging and Deployment of Large Membranes in
#       Space," ISAS report 618, 1985.
#   M. Schenk, S. D. Guest, "Geometry of Miura-folded metamaterials,"
#       PNAS 110(9), 2013.
#   T. Tachi, "Simulation of Rigid Origami," Origami^4, 2009.
#   E. D. Demaine, M. L. Demaine, V. Hart, G. N. Price, T. Tachi,
#       "(Non)existence of Pleated Folds," Graphs and Combinatorics
#       27(3), 2011 -- why the hypar folds only faceted.

import numpy as np

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty
from mathutils import Vector

try:
    from . import crease
except ImportError:                                   # headless import
    import crease

_ASSIGN_CODE = {"M": 0, "V": 1, "F": 2, "U": 3, "B": 4}

_PATTERN_ITEMS = (
    ('MIURA', "Miura-ori",
     "Rigid-foldable parallelogram corrugation; flat-foldable at any panel angle"),
    ('ACCORDION', "Accordion Pleat",
     "Parallel pleats: no interior vertices, so it folds unconditionally"),
    ('WATERBOMB', "Waterbomb",
     "The magic-ball tessellation: squares split by both diagonals and a mid-line"),
    ('YOSHIMURA', "Yoshimura Diamond",
     "The diamond buckle pattern of a cylinder"),
    ('HYPAR', "Pleated Hypar",
     "Concentric pleats and diagonals; folds only faceted, never smooth"),
)


def _frame_to_mesh(name, frame, positions, size):
    """Build a mesh from a flat frame, optionally placed in 3-D."""
    xy = frame.verts[:, :2]
    extent = float(max(xy.max(axis=0) - xy.min(axis=0))) or 1.0
    scale = size / extent
    centre = np.append(0.5 * (xy.max(axis=0) + xy.min(axis=0)), 0.0)

    src = positions if positions is not None else np.hstack(
        [xy, np.zeros((len(xy), 1))])
    co = [tuple((p - centre) * scale) for p in src]

    faces = [list(f) for f in (frame.faces or [])]
    edges = [] if faces else [tuple(int(i) for i in e) for e in frame.edges]

    me = bpy.data.meshes.new(name)
    me.from_pydata(co, edges, faces)
    me.update()

    if frame.assignment is not None:
        # from_pydata may reorder edges when faces are supplied, so map
        # by vertex pair rather than trusting index order.
        want = {}
        for k, (a, b) in enumerate(frame.edges):
            want[(int(a), int(b))] = str(frame.assignment[k])
            want[(int(b), int(a))] = str(frame.assignment[k])
        attr = me.attributes.new("crease_assignment", 'INT', 'EDGE')
        vals = []
        for e in me.edges:
            a, b = e.vertices
            vals.append(_ASSIGN_CODE.get(want.get((a, b), "U"), 3))
        attr.data.foreach_set("value", vals)
    me.update()
    return me, scale, centre


class MESH_OT_crease_pattern_add(bpy.types.Operator):
    """Add a classical origami crease pattern, flat"""

    bl_idname = "mesh.crease_pattern_add"
    bl_label = "Crease Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    pattern: EnumProperty(
        name="Pattern", items=_PATTERN_ITEMS, default='MIURA',
        description="Which classical pattern to build")
    rows: IntProperty(
        name="Rows", default=4, min=1, max=64,
        description="Panels across the sheet, one way")
    cols: IntProperty(
        name="Columns", default=6, min=1, max=64,
        description="Panels across the sheet, the other way")
    panel_angle: FloatProperty(
        name="Panel Angle", default=np.deg2rad(60.0),
        min=np.deg2rad(5.0), max=np.deg2rad(85.0), subtype='ANGLE',
        description="Acute angle of the parallelogram (Miura only)")
    size: FloatProperty(
        name="Sheet Size", default=2.0, min=0.001, max=1000.0,
        unit='LENGTH', description="Longest side of the flat sheet")
    check: bpy.props.BoolProperty(
        name="Report Checks", default=True,
        description="Check Maekawa and Kawasaki after building and "
                    "report any vertices that fail")

    def execute(self, context):
        kw = dict(rows=self.rows, cols=self.cols, alpha=self.panel_angle,
                  count=max(2, self.cols), sides=max(3, min(8, self.cols)),
                  rings=max(2, self.rows))
        try:
            frame = crease.patterns.build(self.pattern, **kw)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        frame.faces = crease.build_faces(frame.verts, frame.edges)
        me, _scale, _c = _frame_to_mesh(self.pattern.title(), frame,
                                        None, self.size)
        obj = bpy.data.objects.new(me.name, me)
        context.collection.objects.link(obj)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        obj["fold_is_flat"] = True

        msg = f"{frame.n_verts} vertices, {frame.n_edges} creases, " \
              f"{frame.n_faces} panels"
        level = {'INFO'}
        if self.check:
            rep = crease.check(frame)
            msg += "; " + rep.summary()
            if not rep:
                level = {'WARNING'}
        if self.pattern == 'HYPAR':
            msg += "; note the pleated hypar has no planar-facet folding " \
                   "(Demaine et al. 2011) -- it folds only faceted"
        self.report(level, msg)
        return {'FINISHED'}


class OBJECT_OT_fold_solve(bpy.types.Operator):
    """Rigidly fold the active crease pattern, caching the path to shape keys"""

    bl_idname = "object.fold_solve"
    bl_label = "Fold Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    fold_angle: FloatProperty(
        name="Fold Angle", default=np.deg2rad(70.0),
        min=np.deg2rad(-179.0), max=np.deg2rad(179.0), subtype='ANGLE',
        description="Dihedral angle to drive the pattern to")
    steps: IntProperty(
        name="Steps", default=12, min=1, max=120,
        description="States solved along the fold path; each becomes a "
                    "shape key, so this is also the animation resolution")
    animate: bpy.props.BoolProperty(
        name="Animate", default=True,
        description="Cache the whole path as shape keys driven by a "
                    "single Fold property, instead of only the end state")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def _frame_from_object(self, obj):
        """Rebuild a crease.Frame from the mesh and its edge attributes."""
        me = obj.data
        verts = np.array([(v.co.x, v.co.y, v.co.z) for v in me.vertices],
                         dtype=float)
        edges = np.array([tuple(e.vertices) for e in me.edges],
                         dtype=np.int64).reshape(-1, 2)
        code_to_char = {0: "M", 1: "V", 2: "F", 3: "U", 4: "B"}
        attr = me.attributes.get("crease_assignment")
        if attr is None:
            raise crease.FoldError(
                "this mesh carries no crease_assignment attribute, so "
                "there is nothing to fold; build or import a crease "
                "pattern first")
        assign = np.array(
            [code_to_char.get(int(d.value), "U") for d in attr.data],
            dtype="<U1")
        faces = [list(p.vertices) for p in me.polygons]
        fr = crease.Frame(verts=verts, edges=edges, assignment=assign,
                          faces=faces or None)
        return fr

    def execute(self, context):
        obj = context.active_object
        try:
            frame = self._frame_from_object(obj)
            if frame.faces is None:
                frame.faces = crease.build_faces(frame.verts, frame.edges)
            folder = crease.rigid.RigidFolder(frame)
            if not folder.n_vars:
                raise crease.rigid.FoldFailure(
                    "no foldable creases: every edge is boundary or flat")
            path = folder.fold_path(0, float(self.fold_angle),
                                    steps=int(self.steps))
            states = [folder.place(r) for r in path]
        except (crease.FoldError, crease.rigid.FoldFailure) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        me = obj.data
        if not self.animate:
            for v, p in zip(me.vertices, states[-1]):
                v.co = Vector(p)
            me.update()
            self.report({'INFO'},
                        f"folded to {np.rad2deg(self.fold_angle):.1f} deg; "
                        f"{folder.dof(path[-1])} degree(s) of freedom")
            return {'FINISHED'}

        # Cache the path: one shape key per solved state, blended by a
        # hat function of a single property so the fold is one dial.
        if obj.data.shape_keys:
            obj.shape_key_clear()
        basis = obj.shape_key_add(name="Flat", from_mix=False)
        for i, p in enumerate(states):
            for v, co in zip(basis.data, states[0]):
                pass
            key = obj.shape_key_add(name=f"Fold {i:02d}", from_mix=False)
            for kv, co in zip(key.data, p):
                kv.co = Vector(co)
            key.slider_min, key.slider_max = 0.0, 1.0

        obj["fold_t"] = 0.0
        obj.id_properties_ui("fold_t").update(
            min=0.0, max=1.0, description="Fold progress, 0 flat to 1 folded")
        n = len(states) - 1
        for i, key in enumerate(obj.data.shape_keys.key_blocks[1:]):
            fc = key.driver_add("value")
            drv = fc.driver
            drv.type = 'SCRIPTED'
            var = drv.variables.new()
            var.name = "t"
            var.type = 'SINGLE_PROP'
            var.targets[0].id = obj
            var.targets[0].data_path = '["fold_t"]'
            drv.expression = f"max(0.0, 1.0 - abs(t*{n} - {i}))"

        obj["fold_is_flat"] = False
        self.report(
            {'INFO'},
            f"folded to {np.rad2deg(self.fold_angle):.1f} deg over "
            f"{n} steps; keyframe the Fold T property to animate; "
            f"{folder.dof(path[-1])} degree(s) of freedom")
        return {'FINISHED'}


_CLASSES = (MESH_OT_crease_pattern_add, OBJECT_OT_fold_solve)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
