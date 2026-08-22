
# Polyhedral Weave Generator for Blender
#
# A Blender take on Antiprism's `poly_weave`, including its weave
# PATTERN language. The seed polyhedron's flag (barycentric) subdivision
# is walked by a pattern program; the visited pattern points become
# closed weave strands, swept as flat ribbons or as round rope tubes
# (optionally relaxed so the ropes round their clasps and keep clear
# of each other).
#
# Pattern syntax (after poly_weave -p):
#   [C|L]                 curve (smoothed) or linear path   (default C)
#   bV,bE,bF              barycentric start point in each flag triangle
#                         over (Vertex, Edge centre, Face centre)
#                         (default 0,1,0 = edge centre)
#   :up[,side[,along]]    intermediate point on each step; repeatable.
#                         up = radial offset, side = sideways offset,
#                         along = position along step (default: spaced)
#   steps                 V E F  pivot about vertex/edge/face corner
#                         v e f  cross to the flag on the other side
#                         R      reverse pivot direction
#                         -      stay on the current flag
#   t[l|r|b]              start circuits from left/right/both flags
#
# Examples: "FEV" (default weave), "1,1,1V" (rings around vertices),
# "1,1,1F" (rings around faces), "1,1,1FFE" (squares on a cube),
# "1,1,1::::::FEV" (curved segments), "0,1,0:0.1FEV" (raised weave).
#
# References:
#   - Adrian Rossiter, Antiprism and its `poly_weave` program and
#     pattern language (https://www.antiprism.com) -- the reference
#     implementation this follows.  The over-under plain/twill weave
#     over a surface is an otherwise classical basketry construction.

bl_info = {
    "name": "Polyhedral Weave Generator",
    "author": "Math Art project (after Antiprism's poly_weave)",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polyhedral Weave",
    "description": "Woven-strand spheres from polyhedra (pattern language)",
    "category": "Add Mesh",
}


try:                                  # inside the math_art package
    from .curve_frames import welded_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import welded_tube



try:                                  # inside the math_art package
    from .polyhedra.seeds import icosa_faces as _icosa_faces
    from .polyhedra.seeds import seed_poly as _shared_seed
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly as _shared_seed
import math

# The mathematics lives in the sibling `weaving` engine package;
# this module is the Blender layer over it.
try:
    from .weaving.polyhedral import (_strand_path, build_weave,
                                         build_weave_tubes, geodesic, parse_pattern, relax_strands,
                                         seed_poly, weave_circuits)
except ImportError:  # flat import outside the package
    from weaving.polyhedral import (_strand_path, build_weave,
                                        build_weave_tubes, geodesic, parse_pattern, relax_strands,
                                        seed_poly, weave_circuits)






# ---- seeds ---------------------------------------------------------------









# ---- pattern parsing -----------------------------------------------------





# ---- flag machinery ------------------------------------------------------
# flag = (face index, corner index i, side s); s=0: edge (f[i], f[i+1]),
# s=1: edge (f[i-1], f[i]). The three adjacencies (cross to the flag
# sharing everything except one element):
#   'v' change vertex, 'e' change edge, 'f' change face.













# ---- strand sweeps -------------------------------------------------------







# ---- rope (round tube) sweep ---------------------------------------------







# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PATTERNS = [
        ('CUSTOM', "Custom", ""),
        ('vfe', "Classic Weave (vfe)",
         "straight-ahead weave: cube 4 hexagons, icosa 6 decagons"),
        ('FEV', "Corner Weave (FEV)", "circuits turning around vertices"),
        ('1,1,1V', "Vertex Rings (1,1,1V)", "circuits around vertices"),
        ('1,1,1F', "Face Rings (1,1,1F)", "circuits around faces"),
        ('1,1,1FFE', "Face Pairs (1,1,1FFE)", "squares on a cube"),
        ('1,1,1::::::FEV', "Curved (1,1,1::::::FEV)",
         "six intermediate points"),
        ('0,1,0:0.12FEV', "Raised Weave (0,1,0:0.12FEV)",
         "intermediate point lifted radially"),
        ('0,1,0:0.1,0.06:-0.1,0.06FEV', "Wavy (up/side intermediates)",
         ""),
        ('1,3,1EV', "Knots (1,3,1EV)", "small rings at each edge"),
    ]

    class MESH_OT_weave_add(bpy.types.Operator):
        """Add a woven-strand model driven by a poly_weave-style
        pattern walked over the seed polyhedron's flag triangles"""
        bl_idname = "mesh.poly_weave_add"
        bl_label = "Polyhedral Weave"
        bl_options = {'REGISTER', 'UNDO'}

        def _pattern_chosen(self, context):
            if self.pattern_preset != 'CUSTOM':
                self.pattern = self.pattern_preset

        kind: EnumProperty(
            name="Seed",
            items=[('CUBE', "Cube", ""), ('ICOSA', "Icosahedron", ""),
                   ('OCTA', "Octahedron", ""), ('TETRA', "Tetrahedron", ""),
                   ('DODECA', "Dodecahedron", "")],
            description="Seed polyhedron whose flags the weave is walked over",
            default='CUBE')
        freq: IntProperty(
            name="Geodesic Frequency", default=1, min=1, max=6,
            description="Subdivision of triangular seeds")
        pattern_preset: EnumProperty(name="Pattern Preset", items=PATTERNS,
                                     default='vfe',
                                     description="Ready-made weave pattern; "
                                                 "Custom keeps the pattern "
                                                 "field below",
                                     update=_pattern_chosen)
        pattern: StringProperty(
            name="Pattern", default="vfe",
            description="[C|L][bV,bE,bF][:up,side,along...]steps[tl|tr|tb]"
                        " -- steps from V E F v e f R -")
        output: EnumProperty(
            name="Output",
            items=[('RIBBON', "Ribbon", "Flat rectangular strands"),
                   ('TUBE', "Tube (Rope)",
                    "Round tubes -- the weave as rope with rounded "
                    "clasps")],
            description="Sweep each strand as a flat ribbon or a round "
                        "rope tube",
            default='RIBBON')
        width: FloatProperty(name="Strand Width", default=0.10,
                             min=0.005, max=0.5,
                             description="Width of the flat ribbon strands")
        thickness: FloatProperty(name="Strand Thickness", default=0.03,
                                 min=0.002, max=0.2,
                                 description="Thickness of the flat ribbon "
                                             "strands")
        tube_radius: FloatProperty(name="Tube Radius", default=0.03,
                                   min=0.005, max=0.2,
                                   description="Radius of the round rope "
                                               "tubes")
        tube_sides: IntProperty(name="Tube Sides", default=10,
                                min=3, max=32,
                                description="Number of sides around each "
                                            "rope tube")
        relax_iters: IntProperty(
            name="Relax Iterations", default=0, min=0, max=500,
            description="Relax the strand centerlines together so the "
                        "ropes round their clasps and keep clear of "
                        "each other (0 = off)")
        rope_clearance: FloatProperty(
            name="Rope Clearance", default=0.0, min=0.0, max=0.5,
            description="Target centre-to-centre strand clearance when "
                        "relaxing (0 = auto, 2.2 x tube radius)")
        amplitude: FloatProperty(
            name="Weave Amplitude", default=0.05, min=0.0, max=0.3,
            description="Alternating radial offset at pattern points "
                        "(set 0 when the pattern's 'up' values weave)")
        subdiv: IntProperty(name="Path Subdivision", default=6,
                            min=1, max=24,
                            description="How finely each strand path is "
                                        "subdivided")
        smooth_rounds: IntProperty(name="Smoothing", default=2,
                                   min=0, max=10,
                                   description="Number of smoothing passes "
                                               "over the strand paths")
        coloring: EnumProperty(
            name="Coloring",
            items=[('STRAND', "Per Strand",
                    "One material per woven strand (view with "
                    "Material Preview or Solid shading set to "
                    "Material color)"),
                   ('NONE', "None", "No materials")],
            description="Whether to give each strand its own material",
            default='STRAND')
        scale: FloatProperty(name="Radius", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall radius of the woven model")

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.45, 0.45, 0.85), (0.80, 0.50, 0.30),
                    (0.35, 0.60, 0.55), (0.75, 0.35, 0.45)]

        @classmethod
        def _material_for(cls, i):
            name = f"Weave Strand {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[i % len(cls._PALETTE)]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb,
                                                               1.0)
                    bsdf.inputs["Roughness"].default_value = 0.45
            return mat

        def execute(self, context):
            try:
                if self.output == 'TUBE':
                    verts, faces, ns, face_strand = build_weave_tubes(
                        self.kind, self.freq, self.pattern,
                        self.tube_radius, self.tube_sides,
                        self.amplitude, self.subdiv,
                        self.smooth_rounds, self.scale,
                        self.relax_iters, self.rope_clearance)
                else:
                    verts, faces, ns, face_strand = build_weave(
                        self.kind, self.freq, self.pattern, self.width,
                        self.thickness, self.amplitude, self.subdiv,
                        self.smooth_rounds, self.scale)
            except (ValueError, RuntimeError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Weave")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # fit (roughly) within a 2 x scale cube at the origin
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
                                    [True] * len(me.polygons))
            if len(me.polygons) == len(face_strand):
                attr = me.attributes.new("strand_index", 'INT',
                                         'FACE')
                attr.data.foreach_set('value', face_strand)
                if self.coloring == 'STRAND':
                    for i in range(ns):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set('material_index',
                                            face_strand)
            me.update()
            obj = bpy.data.objects.new("Weave", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{ns} strands")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind in ('TETRA', 'OCTA', 'ICOSA'):
                lay.prop(self, 'freq')
            lay.prop(self, 'pattern_preset')
            lay.prop(self, 'pattern')
            lay.prop(self, 'output')
            if self.output == 'TUBE':
                keys = ('tube_radius', 'tube_sides', 'relax_iters',
                        'rope_clearance')
            else:
                keys = ('width', 'thickness')
            for k in keys + ('amplitude', 'subdiv', 'smooth_rounds',
                             'coloring', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.poly_weave_add", icon='MOD_LATTICE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_weave_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_weave_add)


def _selftest():
    for kind, freq, pattern in (
            ('CUBE', 1, 'FEV'), ('OCTA', 1, 'FEV'),
            ('ICOSA', 1, 'FEV'), ('ICOSA', 2, 'FEV'),
            ('CUBE', 1, '1,1,1V'), ('CUBE', 1, '1,1,1F'),
            ('CUBE', 1, '1,1,1FFE'), ('DODECA', 1, 'FEV'),
            ('ICOSA', 2, '1,1,1::::::FEV'),
            ('CUBE', 1, '0,1,0:0.12FEV'),
            ('CUBE', 1, '1,3,1EV')):
        V, F = seed_poly(kind)
        if kind in ('TETRA', 'OCTA', 'ICOSA'):
            V, F = geodesic(V, F, freq)
        pat = parse_pattern(pattern)
        cir = weave_circuits(V, F, pat)
        lens = sorted(set(len(c) for c in cir))
        print(f"{kind} f{freq} '{pattern}': strands={len(cir)} "
              f"point-counts={lens}")

    # tube (rope) output
    for kind, freq, relax in (('CUBE', 1, 0), ('CUBE', 1, 40),
                              ('ICOSA', 1, 0), ('ICOSA', 1, 40)):
        verts, faces, ns, fs = build_weave_tubes(
            kind, freq, 'FEV', tube_radius=0.03, tube_sides=8,
            amplitude=0.05, subdiv=4, relax_iters=relax)
        assert verts and faces, "empty tube geometry"
        assert all(len(f) == 4 for f in faces), "non-quad tube face"
        assert len(fs) == len(faces), "face_strand length mismatch"
        assert all(math.isfinite(c) for v in verts for c in v), \
            "non-finite tube vertex"
        print(f"{kind} f{freq} TUBE relax={relax}: strands={ns} "
              f"verts={len(verts)} faces={len(faces)}")

    # relaxation separates the strand centerlines
    def _min_inter(paths):
        best = None
        for a in range(len(paths)):
            for b in range(a + 1, len(paths)):
                for p in paths[a]:
                    for q in paths[b]:
                        d = math.dist(p, q)
                        if best is None or d < best:
                            best = d
        return best

    tr = 0.03
    clr = 2.2 * tr
    V, F = seed_poly('ICOSA')
    cir = weave_circuits(V, F, parse_pattern('FEV'))
    paths0 = [_strand_path(c, 0.05, 4, 2, True) for c in cir]
    paths1 = relax_strands(paths0, tr, 80)
    assert all(math.isfinite(c) for p in paths1 for q in p
               for c in q), "non-finite relaxed bead"
    d0, d1 = _min_inter(paths0), _min_inter(paths1)
    print(f"ICOSA f1 relax: min inter-strand dist "
          f"{d0:.4f} -> {d1:.4f} (target clearance {clr:.4f})")
    assert d1 >= 0.5 * clr or d1 >= d0, \
        "relaxation did not keep strands apart"
    print("RESULT: OK")
