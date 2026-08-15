
# Scherk-Collins Sculpture Generator for Blender
#
# A re-implementation of the geometry engine of Carlo Sequin's
# "Sculpture Generator I" (1996/97) as a Blender add-on.
#
#   C. H. Sequin, "Virtual Prototyping of Scherk-Collins Saddle Rings",
#       Leonardo, Vol 30, No 2, pp 89-96, 1997.
#   C. H. Sequin, H. Meshkin, L. Downs, "Interactive Generation of
#       Scherk-Collins Sculptures", Proc. I3D '97.
#
# Only the sculpture *geometry* is reproduced (per project scope);
# materials, textures and backgrounds are left to Blender.
#
# The tower is based on the exact singly-periodic Scherk minimal surface
#   sin(z) = sinh(x) * sinh(y)
# parametrized per height-level c = sin(z) as
#   x(s) = asinh(sqrt(c) * e^( s*L)),   y(s) = asinh(sqrt(c) * e^(-s*L)),
#   L = ln(sinh(W)/sqrt(c)),   s in [-1, 1]
# which lands the curve ends exactly on the truncation planes x = W / y = W
# (W = "flange" slider; holes break open for W < asinh(1) ~ 0.881, which is
# why the original slider bottoms out at 0.7).
# Higher-order saddles ("branches" b) come from compressing the 90-degree
# wedge of that curve into a 180/b-degree wedge; consecutive storeys are
# rotated by 180/b degrees (so a closed warp=360 ring joins smoothly iff
# (twist + storeys*180/b) mod (360/b) == 0 -- verified against all demo
# files shipped with the original program).
#
# References:
#   H. F. Scherk, "Bemerkungen ueber die kleinste Flaeche innerhalb
#       gegebener Grenzen", J. reine angew. Math. (Crelle) 13, 1835
#       -- the singly-periodic saddle-tower minimal surface.
#   Sculptural form after the collaboration of sculptor Brent Collins
#       and Carlo H. Sequin (see the Leonardo and I3D papers above).

bl_info = {
    "name": "Scherk-Collins Sculpture Generator",
    "author": "Math Art project (after Carlo H. Sequin's Sculpture Generator I)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Scherk-Collins Sculpture / N-panel 'Scherk'",
    "description": "Generate Scherk-Collins saddle-chain toroid sculptures",
    "category": "Add Mesh",
}
# The mathematics lives in the sibling `minsurf` engine package;
# this module is the Blender layer over it.
try:
    from .minsurf.scherk import (PRESETS, Params, XY_SCALE,
                                     fit_transform, generate_sculpture,
                                     parse_spec_text, ring_closes,
                                     spec_text_from, weld_epsilon)
except ImportError:  # flat import outside the package
    from minsurf.scherk import (PRESETS, Params, XY_SCALE,
                                    fit_transform, generate_sculpture,
                                    parse_spec_text, ring_closes,
                                    spec_text_from, weld_epsilon)








# --------------------------------------------------------------------------
# Pure-python geometry core (no bpy - testable standalone)
# --------------------------------------------------------------------------











# --------------------------------------------------------------------------
# Spec-file I/O (the original program's save/demo format)
# --------------------------------------------------------------------------









# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from mathutils import Matrix
    from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                           PointerProperty, StringProperty, EnumProperty)
    from bpy_extras.io_utils import ImportHelper, ExportHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _params_from_props(st):
        detail = st.nurbs_detail if st.output_nurbs else st.detail
        return Params(branches=st.branches, storeys=st.storeys,
                      height=st.height, flange=st.flange,
                      thickness=st.thickness, rim_bulge=st.rim_bulge,
                      twist=st.twist, azimuth=st.azimuth, warp=st.warp,
                      phase=st.phase, rim_round=st.rim_round, detail=detail,
                      scale_x=st.scale_x, scale_y=st.scale_y,
                      scale_z=st.scale_z, global_scale=st.global_scale)

    def _nurbs_patches(p):
        """Contiguous mid-surface row blocks -> NURBS control patches.
        Each block is split at the wedge bisector column so that patch
        edges coincide with single vane legs; adjacent storeys' patches
        then share identical edge control sequences and join cleanly."""
        grids, R, m = generate_sculpture(p, return_grids=True)
        mid = (m - 1) // 2
        patches = []
        for key in sorted(grids.keys()):
            block = []
            for row in grids[key] + [None]:
                if row is not None:
                    block.append(row)
                    continue
                if len(block) >= 2:
                    patches.append([r[:mid + 1] for r in block])
                    patches.append([r[mid:] for r in block])
                block = []
        return patches

    def _build_nurbs_data(obj, p):
        """Replace obj's data with a NURBS surface (one clamped patch per
        storey/branch grid; thickness and rims do not apply)."""
        su = bpy.data.curves.new("ScherkCollins", 'SURFACE')
        su.dimensions = '3D'
        old = obj.data
        obj.data = su
        if old is not None and old.users == 0:
            if isinstance(old, bpy.types.Mesh):
                bpy.data.meshes.remove(old)
            else:
                bpy.data.curves.remove(old)
        view = bpy.context.view_layer
        prev_active = view.objects.active
        view.objects.active = obj

        def _grid_fps():
            fps = []
            for sp in su.splines:
                if sp.point_count_u > 1 and sp.point_count_v > 1:
                    c0 = sp.points[0].co
                    fps.append((sp.point_count_u, sp.point_count_v,
                                round(c0[0], 4), round(c0[1], 4),
                                round(c0[2], 4)))
            return fps

        # Fit the mid-surface control net into the 2 m cube (same
        # convention as the mesh path) before laying down the splines.
        patches = _nurbs_patches(p)
        allpts = [pt for patch in patches for row in patch for pt in row]
        center, f = fit_transform(allpts, p.global_scale)
        patches = [[[((pt[0] - center[0]) * f,
                      (pt[1] - center[1]) * f,
                      (pt[2] - center[2]) * f) for pt in row]
                    for row in patch] for patch in patches]
        for patch in patches:
            prev_fps = _grid_fps()
            for sp in su.splines:
                for pt in sp.points:
                    pt.select = False
            for row in patch:
                sp = su.splines.new('NURBS')
                sp.points.add(len(row) - 1)
                flat = []
                for (x, y, z) in row:
                    flat.extend((x, y, z, 1.0))
                sp.points.foreach_set('co', flat)
                for pt in sp.points:
                    pt.select = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.curve.make_segment()
            bpy.ops.object.mode_set(mode='OBJECT')
            # make_segment chains rows in an arbitrary order: find the
            # newly created grid spline and rewrite its control points
            # into the intended row-major layout
            remaining = list(prev_fps)
            new_sp = None
            for sp in su.splines:
                if sp.point_count_u > 1 and sp.point_count_v > 1:
                    c0 = sp.points[0].co
                    fp = (sp.point_count_u, sp.point_count_v,
                          round(c0[0], 4), round(c0[1], 4), round(c0[2], 4))
                    if fp in remaining:
                        remaining.remove(fp)
                    else:
                        new_sp = sp
            if new_sp is not None:
                pu, pv = new_sp.point_count_u, new_sp.point_count_v
                nrows, ncols = len(patch), len(patch[0])
                flat = []
                if (pu, pv) == (nrows, ncols):
                    # storage is u-fastest: S[v][u] = patch[u][v]
                    for vv in range(pv):
                        for uu in range(pu):
                            x, y, z = patch[uu][vv]
                            flat.extend((x, y, z, 1.0))
                else:
                    for vv in range(pv):
                        for uu in range(pu):
                            x, y, z = patch[vv][uu]
                            flat.extend((x, y, z, 1.0))
                new_sp.points.foreach_set('co', flat)
            for sp in su.splines:
                if sp.point_count_u > 1 and sp.point_count_v > 1:
                    sp.order_u = min(4, sp.point_count_u)
                    sp.order_v = min(4, sp.point_count_v)
                    sp.use_endpoint_u = True
                    sp.use_endpoint_v = True
                    sp.resolution_u = 4
                    sp.resolution_v = 4
        su.update_tag()
        if prev_active is not None:
            view.objects.active = prev_active

    _PROP_COPY_KEYS = ('is_scherk', 'branches', 'storeys',
                       'height', 'flange', 'thickness', 'rim_bulge',
                       'rim_round', 'twist', 'azimuth', 'warp', 'phase',
                       'detail', 'scale_x',
                       'scale_y', 'scale_z', 'global_scale', 'output_nurbs',
                       'nurbs_detail')

    def _swap_object_type(old_obj, to_surface):
        """Mesh <-> Surface object types cannot be changed in place;
        recreate the object, carrying over transform and parameters."""
        name = old_obj.name
        saved = {k: getattr(old_obj.scherk_collins, k)
                 for k in _PROP_COPY_KEYS}
        mw = old_obj.matrix_world.copy()
        colls = list(old_obj.users_collection)
        if to_surface:
            data = bpy.data.curves.new(name, 'SURFACE')
            data.dimensions = '3D'
        else:
            data = bpy.data.meshes.new(name)
        bpy.data.objects.remove(old_obj, do_unlink=True)
        new_obj = bpy.data.objects.new(name, data)
        for coll in (colls or [bpy.context.collection]):
            coll.objects.link(new_obj)
        new_obj.matrix_world = mw
        st = new_obj.scherk_collins
        _SUSPEND[0] = True
        for k in saved:
            setattr(st, k, saved[k])
        _SUSPEND[0] = False
        new_obj.select_set(True)
        bpy.context.view_layer.objects.active = new_obj
        return new_obj

    def rebuild_object(obj):
        st = obj.scherk_collins
        p = _params_from_props(st)
        if st.output_nurbs != (obj.type == 'SURFACE'):
            obj = _swap_object_type(obj, st.output_nurbs)
            st = obj.scherk_collins
        if st.output_nurbs:
            _build_nurbs_data(obj, p)
            return obj
        verts, faces, vert_uv = generate_sculpture(p, with_uv=True)
        me = bpy.data.meshes.new(obj.data.name if obj.data else "ScherkCollins")
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        bm = bmesh.new()
        bm.from_mesh(me)
        # UV map from the parametric grid, set before welding so the
        # per-loop coordinates survive remove_doubles (bmesh vert index
        # still matches the pre-weld vertex order / vert_uv here)
        if len(vert_uv) == len(me.vertices):
            bm.verts.ensure_lookup_table()
            uvl = bm.loops.layers.uv.new("UVMap")
            for f in bm.faces:
                for loop in f.loops:
                    loop[uvl].uv = vert_uv[loop.vert.index]
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld_epsilon(p))
        bmesh.ops.dissolve_degenerate(bm, dist=weld_epsilon(p) * 0.1,
                                      edges=bm.edges)
        if p.thickness * XY_SCALE * p.global_scale > 1e-6:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        # Centre at the origin and fit the whole sculpture (thickness and
        # rim beads included) into the 2 m cube, then apply Overall Scale.
        # Done after welding so remove_doubles still runs at the tuned raw
        # scale; a uniform positive scale preserves the recalculated normals.
        center, f = fit_transform(verts, p.global_scale)
        me.transform(Matrix.Diagonal((f, f, f, 1.0)) @
                     Matrix.Translation((-center[0], -center[1], -center[2])))
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        old = obj.data
        obj.data = me
        if old and old.users == 0:
            bpy.data.meshes.remove(old)
        return obj

    # Set while properties are being copied in bulk -- loading a spec, or
    # duplicating a sculpture -- so the half-copied state does not rebuild
    # once per property.  This used to be done by switching `auto_update`
    # off and back on, which meant the guard and a user-facing toggle were
    # the same switch; with the toggle gone the guard says what it is.
    _SUSPEND = [False]

    def _prop_update(self, context):
        if _SUSPEND[0] or not self.is_scherk:
            return
        obj = self.id_data
        if obj is None:
            return
        if self.output_nurbs or obj.type == 'SURFACE':
            # NURBS rebuilds use edit-mode operators and may replace the
            # object; defer out of the property-update callback
            name = obj.name
            def _deferred():
                o = bpy.data.objects.get(name)
                if o is not None and o.scherk_collins.is_scherk:
                    rebuild_object(o)
                return None
            bpy.app.timers.register(_deferred, first_interval=0.05)
        else:
            rebuild_object(obj)

    class ScherkCollinsProps(bpy.types.PropertyGroup):
        is_scherk: BoolProperty(default=False, options={'HIDDEN'})
        branches: IntProperty(
            name="Branches", description="Order of the saddles (# of branches)",
            default=2, min=1, max=10, update=_prop_update)
        storeys: IntProperty(
            name="Storeys", description="Number of hole/saddle storeys",
            default=2, min=1, max=16, update=_prop_update)
        height: FloatProperty(
            name="Storey Height", description="Height of one storey",
            default=1.5, min=0.1, max=5.0, update=_prop_update)
        flange: FloatProperty(
            name="Flange Width",
            description="Width of the flanges (holes break open below ~0.88)",
            default=1.5, min=0.7, max=5.0, update=_prop_update)
        thickness: FloatProperty(
            name="Thickness", description="Thickness of the vanes (0 = surface only)",
            default=0.15, min=0.0, max=0.5, update=_prop_update)
        rim_bulge: FloatProperty(
            name="Rim Bulge", description="Amount of bulge on the rim beads",
            default=1.5, min=0.0, max=4.0, update=_prop_update)
        rim_round: FloatProperty(
            name="Rim Round",
            description="Roundness of the edge: 1 = a rounded bull-nose, "
                        "0 = a flat/square edge",
            default=1.0, min=0.0, max=1.0, step=10, update=_prop_update)
        twist: FloatProperty(
            name="Twist", description="Overall axial twist (degrees)",
            default=0.0, min=-900.0, max=1080.0, step=1500, update=_prop_update)
        azimuth: FloatProperty(
            name="Azimuth", description="Turn of the profile around the tower axis (degrees)",
            default=0.0, min=-360.0, max=360.0, step=500, update=_prop_update)
        warp: FloatProperty(
            name="Warp", description="Bend of the tower towards an arch/toroid (degrees; 360 = closed ring)",
            default=0.0, min=0.0, max=1080.0, step=1000, update=_prop_update)
        phase: FloatProperty(
            name="Phase",
            description="Shift the holes along an open tower, in storeys "
                        "(0 = a flange at each end; 0.5 = a half-hole at "
                        "each end). No effect on closed rings.",
            default=0.5, min=0.0, max=0.999, step=10, update=_prop_update)
        detail: IntProperty(
            name="Detail", description="Grid detail (tessellation density)",
            default=5, min=1, max=16, update=_prop_update)
        scale_x: FloatProperty(
            name="Stretch X", default=1.0, min=0.2, max=5.0, update=_prop_update)
        scale_y: FloatProperty(
            name="Stretch Y", default=1.0, min=0.2, max=5.0, update=_prop_update)
        scale_z: FloatProperty(
            name="Stretch Z", default=1.0, min=0.2, max=5.0, update=_prop_update)
        global_scale: FloatProperty(
            name="Overall Scale", default=1.0, min=0.05, max=10.0,
            update=_prop_update)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Output a compact NURBS surface (mid-surface only; "
                        "thickness and rim bulge do not apply)",
            update=_prop_update)
        nurbs_detail: IntProperty(
            name="NURBS Detail",
            description="Control-point density used for NURBS output "
                        "(the NURBS surface stays smooth at low values)",
            default=2, min=1, max=16, update=_prop_update)

    def _apply_param_dict(st, d):
        _SUSPEND[0] = True
        for k, v in d.items():
            if hasattr(st, k):
                setattr(st, k, v)
        _SUSPEND[0] = False

    _RESET_KEYS = ('branches', 'storeys', 'height', 'flange', 'thickness',
                   'rim_bulge', 'rim_round', 'twist', 'azimuth', 'warp',
                   'phase', 'detail', 'scale_x', 'scale_y', 'scale_z',
                   'global_scale')

    def _preset_chosen(self, context):
        """Copy the chosen preset's values into the operator's own
        properties so the redo-panel sliders start from them and remain
        freely tweakable afterwards.  Choosing Default resets every
        parameter to the program defaults."""
        if self.preset == 'CUSTOM':
            for k in _RESET_KEYS:
                prop = self.bl_rna.properties.get(k)
                if prop is not None and hasattr(self, k):
                    setattr(self, k, prop.default)
        else:
            for k, v in PRESETS[self.preset][1].items():
                if hasattr(self, k):
                    setattr(self, k, v)
        self.preset_applied = True

    class MESH_OT_scherk_collins_add(bpy.types.Operator):
        """Add a Scherk-Collins sculpture (tweak all parameters live in
        the redo panel, or later in the N-panel 'Scherk' tab)"""
        bl_idname = "mesh.scherk_collins_add"
        bl_label = "Scherk-Collins Sculpture"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Default", "Program defaults")] +
                  [(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='CUSTOM', update=_preset_chosen)
        branches: IntProperty(name="Branches", default=2, min=1, max=10)
        storeys: IntProperty(name="Storeys", default=2, min=1, max=16)
        height: FloatProperty(name="Storey Height", default=1.5,
                              min=0.1, max=5.0)
        flange: FloatProperty(name="Flange Width", default=1.5,
                              min=0.7, max=5.0)
        thickness: FloatProperty(name="Thickness", default=0.15,
                                 min=0.0, max=0.5)
        rim_bulge: FloatProperty(name="Rim Bulge", default=1.5,
                                 min=0.0, max=4.0)
        rim_round: FloatProperty(
            name="Rim Round", default=1.0, min=0.0, max=1.0,
            description="Roundness of the edge: 1 = rounded bull-nose, "
                        "0 = flat/square edge")
        twist: FloatProperty(name="Twist", default=0.0,
                             min=-900.0, max=1080.0)
        azimuth: FloatProperty(name="Azimuth", default=0.0,
                               min=-360.0, max=360.0)
        warp: FloatProperty(name="Warp", default=0.0, min=0.0, max=1080.0)
        phase: FloatProperty(
            name="Phase", default=0.5, min=0.0, max=0.999,
            description="Shift the holes along an open tower, in storeys "
                        "(0 = flange at each end, 0.5 = half-hole at each "
                        "end); no effect on closed rings")
        detail: IntProperty(name="Detail", default=5, min=1, max=16)
        scale_x: FloatProperty(name="Stretch X", default=1.0,
                               min=0.2, max=5.0)
        scale_y: FloatProperty(name="Stretch Y", default=1.0,
                               min=0.2, max=5.0)
        scale_z: FloatProperty(name="Stretch Z", default=1.0,
                               min=0.2, max=5.0)
        global_scale: FloatProperty(name="Overall Scale", default=1.0,
                                    min=0.05, max=10.0)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Compact NURBS surface instead of a mesh "
                        "(mid-surface only; no thickness/rims)")
        nurbs_detail: IntProperty(
            name="NURBS Detail", default=2, min=1, max=16,
            description="Control-point density used for NURBS output")
        # set once the preset values have been copied into the sliders,
        # so redo-panel tweaks are not overwritten on re-execute
        preset_applied: BoolProperty(default=False, options={'HIDDEN'})

        _PARAM_KEYS = ('branches', 'storeys', 'height', 'flange',
                       'thickness', 'rim_bulge', 'rim_round', 'twist',
                       'azimuth', 'warp', 'phase', 'detail', 'scale_x',
                       'scale_y', 'scale_z',
                       'global_scale', 'output_nurbs', 'nurbs_detail')

        def execute(self, context):
            # menu/scripted invocation sets `preset` without firing its
            # update callback -- apply it here exactly once
            if self.preset != 'CUSTOM' and not self.preset_applied:
                _preset_chosen(self, context)
            me = bpy.data.meshes.new("ScherkCollins")
            obj = bpy.data.objects.new("ScherkCollins", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            st = obj.scherk_collins
            st.is_scherk = True
            _apply_param_dict(st, {k: getattr(self, k)
                                   for k in self._PARAM_KEYS})
            rebuild_object(obj)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "preset")
            col = lay.column(align=True)
            skip = 'detail' if self.output_nurbs else 'nurbs_detail'
            for k in self._PARAM_KEYS:
                if k != skip:
                    col.prop(self, k)

    class SCHERK_OT_regenerate(bpy.types.Operator):
        """Rebuild the sculpture mesh from its parameters"""
        bl_idname = "scherk.regenerate"
        bl_label = "Regenerate"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            o = context.object
            return o is not None and o.scherk_collins.is_scherk

        def execute(self, context):
            rebuild_object(context.object)
            return {'FINISHED'}

    class SCHERK_OT_load_spec(bpy.types.Operator, ImportHelper):
        """Load a Sculpture Generator spec/demo file (.txt)"""
        bl_idname = "scherk.load_spec"
        bl_label = "Load Spec File"
        bl_options = {'REGISTER', 'UNDO'}
        filename_ext = ".txt"
        filter_glob: StringProperty(default="*.txt;*.param", options={'HIDDEN'})

        def execute(self, context):
            try:
                with open(self.filepath, 'r', encoding='utf-8',
                          errors='replace') as f:
                    d = parse_spec_text(f.read())
            except OSError as e:
                self.report({'ERROR'}, f"Cannot read file: {e}")
                return {'CANCELLED'}
            if not d:
                self.report({'ERROR'}, "No sculpture parameters found in file")
                return {'CANCELLED'}
            obj = context.object
            if obj is None or not obj.scherk_collins.is_scherk:
                bpy.ops.mesh.scherk_collins_add()
                obj = context.object
            _apply_param_dict(obj.scherk_collins, d)
            rebuild_object(obj)
            self.report({'INFO'}, f"Loaded {len(d)} parameters")
            return {'FINISHED'}

    class SCHERK_OT_save_spec(bpy.types.Operator, ExportHelper):
        """Save parameters as a Sculpture Generator spec file (.txt)"""
        bl_idname = "scherk.save_spec"
        bl_label = "Save Spec File"
        filename_ext = ".txt"
        filter_glob: StringProperty(default="*.txt", options={'HIDDEN'})

        @classmethod
        def poll(cls, context):
            o = context.object
            return o is not None and o.scherk_collins.is_scherk

        def execute(self, context):
            p = _params_from_props(context.object.scherk_collins)
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.write(spec_text_from(p))
            except OSError as e:
                self.report({'ERROR'}, f"Cannot write file: {e}")
                return {'CANCELLED'}
            return {'FINISHED'}

    class VIEW3D_PT_scherk_collins(bpy.types.Panel):
        bl_label = "Scherk-Collins"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Math Art"
        bl_order = 20

        @classmethod
        def poll(cls, context):
            # Only for a Scherk-Collins sculpture.  The panel used to offer
            # an Add button when none was selected, which forced it to stay
            # visible in every scene; creating one belongs in Add > Mesh,
            # where every other generator's entry already is.
            o = context.object
            return o is not None and o.scherk_collins.is_scherk

        def draw(self, context):
            lay = self.layout
            obj = context.object
            st = obj.scherk_collins
            col = lay.column(align=True)
            col.use_property_split = True
            col.prop(st, "branches")
            col.prop(st, "storeys")
            col.prop(st, "height")
            col.prop(st, "flange")
            col.prop(st, "thickness")
            col.prop(st, "rim_bulge")
            col.prop(st, "rim_round")
            col.separator()
            col.prop(st, "twist")
            col.prop(st, "azimuth")
            col.prop(st, "warp")
            col.prop(st, "phase")
            col.separator()
            col.prop(st, "nurbs_detail" if st.output_nurbs else "detail")
            col.prop(st, "scale_x")
            col.prop(st, "scale_y")
            col.prop(st, "scale_z")
            col.prop(st, "global_scale")
            col.prop(st, "output_nurbs")
            if st.output_nurbs:
                col.label(text="NURBS: thickness/rims not applied",
                          icon='INFO')
            p = _params_from_props(st)
            if p.warp > 0:
                if ring_closes(p):
                    lay.label(text="Ring closes smoothly", icon='CHECKMARK')
                else:
                    b = p.branches
                    need = (-p.storeys * 180.0 / b) % (360.0 / b)
                    lay.label(text=f"Seam at ring closure (twist "
                                   f"{need:.0f} + k*{360.0 / b:.0f} closes)",
                              icon='ERROR')
            # No Auto Update toggle, no Regenerate, no Add Another: every
            # property below rebuilds the mesh as it changes, so the first two
            # had nothing to do, and making a new sculpture belongs in
            # Add > Mesh like every other generator's.  Load and Save remain
            # as operators (F3), off the panel.

    def _menu_func(self, context):
        self.layout.operator_menu_enum("mesh.scherk_collins_add", "preset",
                                       text="Scherk-Collins Sculpture",
                                       icon='MESH_TORUS')

    _classes = (ScherkCollinsProps, MESH_OT_scherk_collins_add,
                SCHERK_OT_regenerate, SCHERK_OT_load_spec,
                SCHERK_OT_save_spec, VIEW3D_PT_scherk_collins)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        bpy.types.Object.scherk_collins = PointerProperty(
            type=ScherkCollinsProps)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        del bpy.types.Object.scherk_collins
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def _selftest():
    # standalone smoke test of the geometry core
    for name, kw in [("defaults", {}),
                     ("trefoil", PRESETS['TREFOIL'][1]),
                     ("heptoroid", PRESETS['HEPTOROID'][1]),
                     ("open arc b1", dict(branches=1, storeys=4, height=1.9,
                                          flange=1.5, thickness=0.08,
                                          rim_bulge=1.5, warp=270,
                                          twist=885, detail=6)),
                     ("open holes", dict(branches=6, storeys=4, height=1.0,
                                         flange=0.8, thickness=0.02,
                                         rim_bulge=0.0, warp=360,
                                         twist=180, azimuth=45, detail=7)),
                     ("thin sheet", dict(thickness=0.0, warp=360,
                                         storeys=4))]:
        p = Params(**kw)
        v, f = generate_sculpture(p)
        xs = [q[0] for q in v]; ys = [q[1] for q in v]; zs = [q[2] for q in v]
        print(f"{name:12s}: verts={len(v):7d} faces={len(f):7d} "
              f"closes={ring_closes(p)} "
              f"bbox=({min(xs):.2f}..{max(xs):.2f}, "
              f"{min(ys):.2f}..{max(ys):.2f}, {min(zs):.2f}..{max(zs):.2f})")
