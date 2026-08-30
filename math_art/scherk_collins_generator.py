
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
    "location": "View3D > Add > Mesh > Scherk-Collins Sculpture",
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
                           StringProperty, EnumProperty)
    from bpy_extras.io_utils import ImportHelper, ExportHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _live_settings(obj):
        """This object's Scherk-Collins settings, or None.

        The parameters live where the Math Art sidebar keeps every
        generator's -- on the object, in a settings group derived from
        the Add operator's own properties.  Reading them from there
        rather than from a private PropertyGroup of this module's own is
        the point of folding this generator into that framework: there
        is one set of properties now, not two to keep in step.
        """
        if obj is None:
            return None
        try:
            from .live.registry import settings_for_object
        except ImportError:              # flat import outside the package
            from live.registry import settings_for_object
        info, pg = settings_for_object(obj)
        if info is None or info.idname != "mesh.scherk_collins_add":
            return None
        return pg

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

    def write_geometry(obj, st):
        """Build the sculpture described by `st` into `obj`.

        `st` is anything carrying the parameter properties: the operator
        while it runs, or the settings the Math Art sidebar stores on the
        object.  The two are the same set of properties, so this is the
        one place the sculpture is built and there is no second
        implementation to keep in step.

        The caller owns the object's TYPE.  A mesh and a NURBS surface
        cannot be swapped in place, so choosing between them belongs to
        whoever creates the object -- `execute` when the sculpture is
        added, and the sidebar's rebuild (which replaces the object) when
        NURBS Output is toggled later.
        """
        p = _params_from_props(st)
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
        the redo panel, or later in the N-panel 'Math Art' tab)"""
        bl_idname = "mesh.scherk_collins_add"
        bl_label = "Scherk-Collins Sculpture"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            description="Start from a named sculpture, or Default for "
                        "the program defaults",
            items=[('CUSTOM', "Default", "Program defaults")] +
                  [(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='CUSTOM', update=_preset_chosen)
        branches: IntProperty(
            name="Branches", default=2, min=1, max=10,
            description="Order of the saddles (# of branches)")
        storeys: IntProperty(
            name="Storeys", default=2, min=1, max=16,
            description="Number of hole/saddle storeys")
        height: FloatProperty(
            name="Storey Height", default=1.5, min=0.1, max=5.0,
            description="Height of one storey")
        flange: FloatProperty(
            name="Flange Width", default=1.5, min=0.7, max=5.0,
            description="Width of the flanges (holes break open "
                        "below ~0.88)")
        thickness: FloatProperty(
            name="Thickness", default=0.15, min=0.0, max=0.5,
            description="Thickness of the vanes (0 = surface only)")
        rim_bulge: FloatProperty(
            name="Rim Bulge", default=1.5, min=0.0, max=4.0,
            description="Amount of bulge on the rim beads")
        rim_round: FloatProperty(
            name="Rim Round", default=1.0, min=0.0, max=1.0, step=10,
            description="Roundness of the edge: 1 = a rounded bull-nose, "
                        "0 = a flat/square edge")
        twist: FloatProperty(
            name="Twist", default=0.0, min=-900.0, max=1080.0, step=1500,
            description="Overall axial twist (degrees)")
        azimuth: FloatProperty(
            name="Azimuth", default=0.0, min=-360.0, max=360.0, step=500,
            description="Turn of the profile around the tower axis "
                        "(degrees)")
        warp: FloatProperty(
            name="Warp", default=0.0, min=0.0, max=1080.0, step=1000,
            description="Bend of the tower towards an arch/toroid "
                        "(degrees; 360 = closed ring)")
        phase: FloatProperty(
            name="Phase", default=0.5, min=0.0, max=0.999, step=10,
            description="Shift the holes along an open tower, in storeys "
                        "(0 = a flange at each end; 0.5 = a half-hole at "
                        "each end). No effect on closed rings.")
        detail: IntProperty(
            name="Detail", default=5, min=1, max=16,
            description="Grid detail (tessellation density)")
        scale_x: FloatProperty(name="Stretch X", default=1.0,
                               min=0.2, max=5.0,
                               description="Stretch factor along the X axis")
        scale_y: FloatProperty(name="Stretch Y", default=1.0,
                               min=0.2, max=5.0,
                               description="Stretch factor along the Y axis")
        scale_z: FloatProperty(name="Stretch Z", default=1.0,
                               min=0.2, max=5.0,
                               description="Stretch factor along the Z axis")
        global_scale: FloatProperty(name="Overall Scale", default=1.0,
                                    min=0.05, max=10.0,
                                    description="Overall size of the "
                                                "sculpture")
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Output a compact NURBS surface (mid-surface "
                        "only; thickness and rim bulge do not apply)")
        nurbs_detail: IntProperty(
            name="NURBS Detail", default=2, min=1, max=16,
            description="Control-point density used for NURBS output "
                        "(the NURBS surface stays smooth at low values)")
        # set once the preset values have been copied into the sliders,
        # so redo-panel tweaks are not overwritten on re-execute
        preset_applied: BoolProperty(default=False, options={'HIDDEN'})

        _PARAM_KEYS = ('branches', 'storeys', 'height', 'flange',
                       'thickness', 'rim_bulge', 'rim_round', 'twist',
                       'azimuth', 'warp', 'phase', 'detail', 'scale_x',
                       'scale_y', 'scale_z',
                       'global_scale', 'output_nurbs', 'nurbs_detail')

        # Read by the Math Art sidebar: rebuild on a timer rather than
        # inside the property-update callback.  The NURBS output is built
        # with `bpy.ops.object.mode_set` and `bpy.ops.curve.make_segment`,
        # and edit-mode operators need a settled context that a
        # half-finished property write does not provide.
        math_art_live_defer = True

        def execute(self, context):
            # menu/scripted invocation sets `preset` without firing its
            # update callback -- apply it here exactly once
            if self.preset != 'CUSTOM' and not self.preset_applied:
                _preset_chosen(self, context)
            # The object's type is fixed by the data it is created with,
            # so the choice is made here rather than by swapping a mesh
            # for a surface afterwards.
            if self.output_nurbs:
                data = bpy.data.curves.new("ScherkCollins", 'SURFACE')
                data.dimensions = '3D'
            else:
                data = bpy.data.meshes.new("ScherkCollins")
            obj = bpy.data.objects.new("ScherkCollins", data)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            write_geometry(obj, self)
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
            # A spec file describes a whole sculpture, so loading one
            # adds that sculpture.  Going through the Add operator means
            # the parameters are recorded on the new object exactly as any
            # other sculpture's are, and stay editable in the sidebar.
            keys = set(MESH_OT_scherk_collins_add._PARAM_KEYS)
            bpy.ops.mesh.scherk_collins_add(
                **{k: v for k, v in d.items() if k in keys})
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
            return _live_settings(context.object) is not None

        def execute(self, context):
            st = _live_settings(context.object)
            if st is None:
                self.report({'ERROR'}, "not a Scherk-Collins sculpture")
                return {'CANCELLED'}
            p = _params_from_props(st)
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.write(spec_text_from(p))
            except OSError as e:
                self.report({'ERROR'}, f"Cannot write file: {e}")
                return {'CANCELLED'}
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator_menu_enum("mesh.scherk_collins_add", "preset",
                                       text="Scherk-Collins Sculpture",
                                       icon='MESH_TORUS')

    _classes = (MESH_OT_scherk_collins_add, SCHERK_OT_load_spec,
                SCHERK_OT_save_spec)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
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
