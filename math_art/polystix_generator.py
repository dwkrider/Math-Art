
# Polystix -- non-intersecting cubic rod packings for Blender
#
# Symmetric bundles of straight rods running in 3 or 4 directions that
# interlock through space WITHOUT touching -- the "polystix" family of
# invariant cubic cylinder/rod packings. Distinct from linked-loop
# tangles (see polylinks_generator): here the elements are infinite
# straight rods on a crystallographic lattice, clipped to a finite
# sculptural cell.
#
# Nomenclature (Conway): the prefix counts the SIDES of the rod's
# cross-section prism, not the number of directions.
#   * tetrastix -- square rods in 3 directions (the cube axes, <100>),
#     primitive-cubic packing (O'Keeffe's Pi*); square prisms fill 3/4.
#   * hemistix  -- square rods in the same 3 directions, the second
#     (and only other) cubic square-rod packing: chiral, space group
#     I4_1 32, six rods per cell, filling 3/8 -- exactly half of
#     tetrastix (the "hemi").
#   * hexastix  -- hexagonal rods in 4 directions (the body diagonals,
#     <111>), the garnet packing (O'Keeffe's Gamma, space group Ia-3d);
#     hexagonal prisms fill exactly 3/4 -- the "bundle of pencils" form.
#   * tristix   -- triangular rods in the same 4 <111> directions, a
#     chiral packing (O'Keeffe's +Omega, space group I4_1 32).
#   * +Sigma    -- a second chiral 4-direction <111> packing.
# Every <111> packing shares the same four direction vectors and
# differs only in the registration (offset) of the four families.
#
# The four <111> rods meet pairwise at the tetrahedral angle
# arccos(1/3) = 70.5288 deg. In the plane perpendicular to each
# direction the rod centres form a triangular (hexagonal) lattice; the
# other families thread through the gaps. For the garnet (hexastix)
# packing with cube side a, nearest inter-family axis distance is
# a*sqrt(2)/4, so round rods just touch at radius r = a*sqrt(2)/8 and
# the round-cylinder packing fraction is pi*sqrt(3)/8 ~ 0.6802.
#
# References:
#   - M. O'Keeffe & S. Andersson, "Rod Packings and Crystal Chemistry",
#     Acta Cryst. A33 (1977) 914-923.
#   - M. O'Keeffe, J. Plevert, Y. Teshima, Y. Watanabe, T. Ogama,
#     "The Invariant Cubic Rod (Cylinder) Packings: Symmetries and
#     Coordinates", Acta Cryst. A57 (2001) 110-111.
#   - J. H. Conway, H. Burgiel, C. Goodman-Strauss, "The Symmetries of
#     Things" (A K Peters, 2008) -- coins polystix/tetrastix/hexastix.
#   - A. Holden, "Shapes, Space and Symmetry" (Columbia U.P., 1971) --
#     earliest rod-packing sculptures.
#   - A. Widmark, "Sculpture Design with Hexastix and Related
#     Non-Intersecting Cylinder Packings", Bridges 2021, pp. 293-296.
#     https://archive.bridgesmathart.org/2021/bridges2021-293.html
#   - A. Widmark, "Polystix Sculpture Design Revisited", Bridges 2022,
#     pp. 379-382.
#     https://archive.bridgesmathart.org/2022/bridges2022-379.html
#   - K. Hui & J. S. Purcell, "On the geometry of rod packings in the
#     3-torus", arXiv:2212.04662 (2023) -- explicit O'Keeffe coordinates.

bl_info = {
    "name": "Polystix",
    "author": "Math Art project (after O'Keeffe, Conway & Widmark)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polystix",
    "description": "Non-intersecting cubic rod packings "
                   "(hexastix, tetrastix, tristix)",
    "category": "Add Mesh",
}



try:                                  # inside the math_art package
    from .curve_frames import closed_tube as _shared_closed_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import closed_tube as _shared_closed_tube
# The mathematics lives in the sibling `weaving` engine package;
# this module is the Blender layer over it.
try:
    from .weaving.sticks import (PACKINGS, build_polystix)
except ImportError:  # flat import outside the package
    from weaving.sticks import (PACKINGS, build_polystix)






# ----------------------------------------------------------------------
# small vector helpers (plain tuples, no numpy dependency)















# ----------------------------------------------------------------------
# packing geometry

















# ----------------------------------------------------------------------
# end-connecting knots/links (after Widmark 2021): join the protruding
# rod ends in pairs -- "half loops" between neighbours on the same axis,
# "twists" between crossing neighbours on different axes -- so the rods
# fuse into continuous embedded loops (the torch-worked glass knots).











try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, BoolProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PRESETS = {
        'PENCILS': ("Hexagonal Pencils (hexastix)",
                    dict(packing='HEXASTIX', cross_section='PRISM',
                         fill=1.0, clip='RHOMBIC_DODECA')),
        'HEXCYL': ("Cylinder Bundle (hexastix)",
                   dict(packing='HEXASTIX', cross_section='CYLINDER',
                        fill=0.95, clip='SPHERE')),
        'TETRA': ("Tetrastix Cubes",
                  dict(packing='TETRASTIX', cross_section='PRISM',
                       fill=0.9, clip='CUBE')),
        'TRI': ("Tristix (chiral triangles)",
                dict(packing='TRISTIX', cross_section='PRISM',
                     fill=1.0, clip='TRUNC_OCTA')),
        'SIGMA': ("+Sigma bundle (chiral)",
                  dict(packing='SIGMA', cross_section='CYLINDER',
                       fill=0.95, clip='RHOMBIC_DODECA')),
        'KNOT': ("Hexastix Knot (linked loops)",
                 dict(packing='HEXASTIX', cross_section='CYLINDER',
                      fill=0.65, clip='RHOMBIC_DODECA',
                      connect_loops=True)),
    }

    _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                (0.62, 0.40, 0.75), (0.25, 0.72, 0.72)]

    class MESH_OT_polystix_add(bpy.types.Operator):
        """Non-intersecting cubic rod packing (hexastix / tetrastix /
        tristix), after O'Keeffe, Conway and Widmark"""
        bl_idname = "mesh.polystix_add"
        bl_label = "Polystix"
        bl_options = {'REGISTER', 'UNDO'}

        def _preset_chosen(self, context):
            if self.preset != 'CUSTOM':
                for k, v in PRESETS[self.preset][1].items():
                    setattr(self, k, v)

        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Custom", "")] +
                  [(k, v[0], "") for k, v in PRESETS.items()],
            default='PENCILS', update=_preset_chosen)
        packing: EnumProperty(
            name="Packing",
            items=[(k, PACKINGS[k]['label'], "")
                   for k in ('TETRASTIX', 'HEMISTIX', 'HEXASTIX',
                             'TRISTIX', 'SIGMA')],
            default='HEXASTIX')
        cross_section: EnumProperty(
            name="Cross Section",
            items=[('PRISM', "Native Prism",
                    "The canonical space-filling polygon for this "
                    "packing (square / hexagon / triangle), oriented "
                    "so flats face the interpenetrating rods"),
                   ('CYLINDER', "Round Cylinder",
                    "Circular rods (n-gon tube)")],
            default='PRISM')
        fill: FloatProperty(
            name="Fill", default=0.98, min=0.02, max=1.2,
            description="Rod radius as a fraction of the just-touching "
                        "radius for this packing (1.0 = rods touch)")
        extent: IntProperty(
            name="Extent", default=4, min=1, max=10,
            description="Number of lattice cells the arrangement spans "
                        "(more = more rods)")
        clip: EnumProperty(
            name="Clip Volume",
            items=[('CUBE', "Cube", ""),
                   ('RHOMBIC_DODECA', "Rhombic Dodecahedron", ""),
                   ('TRUNC_OCTA', "Truncated Octahedron", ""),
                   ('SPHERE', "Sphere", "")],
            default='RHOMBIC_DODECA')
        handedness: EnumProperty(
            name="Handedness",
            items=[('RIGHT', "Right", ""), ('LEFT', "Left (mirror)", "")],
            default='RIGHT',
            description="Mirror the packing (flips the chiral "
                        "enantiomorph of tristix / +Sigma)")
        tube_sides: IntProperty(name="Tube Sides", default=16,
                                min=3, max=48)
        overhang: FloatProperty(
            name="Overhang", default=0.0, min=-4.0, max=8.0,
            description="Extend each rod past the interleaved core by "
                        "this many cells at both ends (negative "
                        "retracts the rods to expose the weave)")
        connect_loops: BoolProperty(
            name="Connect Loops", default=False,
            description="Fuse the protruding rod ends into continuous "
                        "knots/links: 'half loops' join same-axis "
                        "neighbours, 'twists' join crossing neighbours "
                        "(after Widmark 2021). Rods become round tubes; "
                        "overhang is forced on so the ends can meet")
        loop_segments: IntProperty(
            name="Loop Smoothness", default=12, min=2, max=48,
            description="Samples along each connecting arc")
        cap_ends: BoolProperty(name="Cap Ends", default=True)
        coloring: EnumProperty(
            name="Coloring",
            items=[('DIRECTION', "Per Direction",
                    "One material per rod direction (3 or 4 colours)"),
                   ('NONE', "None", "")],
            default='DIRECTION')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        @classmethod
        def _material_for(cls, i):
            name = f"Polystix {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[i] if i < len(cls._PALETTE) else \
                    __import__('colorsys').hsv_to_rgb(
                        (i * 0.618034) % 1.0, 0.6, 0.8)
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.4
            return mat

        _PALETTE = _PALETTE

        def execute(self, context):
            if self.preset != 'CUSTOM':
                self._preset_chosen(context)
            (verts, faces, face_dir, ncols,
             n_rods, truncated) = build_polystix(
                self.packing, self.cross_section, self.fill,
                self.extent, self.clip, self.handedness,
                self.tube_sides, self.cap_ends, self.overhang,
                self.connect_loops, self.loop_segments)
            if not verts:
                self.report({'WARNING'},
                            "No rods in the clip volume; raise Extent")
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Polystix")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # centre and fit within a 2 x scale cube at the origin
            lo = [min(v.co[k] for v in me.vertices) for k in range(3)]
            hi = [max(v.co[k] for v in me.vertices) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) or 1.0
            f = self.scale / half
            for v in me.vertices:
                v.co = [(v.co[k] - (lo[k] + hi[k]) / 2.0) * f
                        for k in range(3)]
            if (self.coloring == 'DIRECTION'
                    and len(me.polygons) == len(faces)):
                for i in range(ncols):
                    me.materials.append(self._material_for(i))
                me.polygons.foreach_set('material_index', face_dir)
                attr = me.attributes.new("dir_index", 'INT', 'FACE')
                attr.data.foreach_set('value', face_dir)
            me.update()
            obj = bpy.data.objects.new("Polystix", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.connect_loops:
                msg = f"{ncols} loop(s) from {n_rods} rods"
            else:
                msg = f"{n_rods} rods"
            if truncated:
                msg += " (capped; lower Extent)"
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        # properties a non-Custom preset overwrites on execute -- these
        # are greyed out (locked) while a preset is active.
        _PRESET_DRIVEN = {'packing', 'cross_section', 'fill', 'clip'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            locked = self.preset != 'CUSTOM'
            loops = self.connect_loops
            keys = ['packing', 'cross_section', 'fill', 'extent', 'clip',
                    'handedness']
            if self.cross_section == 'CYLINDER' or loops:
                keys.append('tube_sides')
            keys += ['overhang', 'connect_loops']
            if loops:
                keys.append('loop_segments')
            keys += ['cap_ends', 'coloring', 'scale']
            for k in keys:
                row = lay.row()
                dis_preset = locked and k in self._PRESET_DRIVEN
                # loop mode forces round tubes and closed loops, so the
                # cross-section and cap-ends controls have no effect
                dis_loop = loops and k in ('cross_section', 'cap_ends')
                row.enabled = not (dis_preset or dis_loop)
                row.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.polystix_add", icon='MESH_CYLINDER')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polystix_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polystix_add)
