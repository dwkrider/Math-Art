
# D-Form Generator for Blender
#
# A D-form is made by cutting two flat pieces whose outlines have the
# SAME PERIMETER and gluing them edge to edge.  The pieces cannot lie
# flat once joined, so the join pops into a closed solid that is curved
# but never stretched: it is paper everywhere except along the seam,
# which carries all of the Gaussian curvature.  Tony Wills invented them
# (Bridges 2006) and John Sharp brought them to the mathematical-art
# community (Bridges 2005, and the Tarquin book).
#
# Demaine and O'Rourke show the resulting solid is the convex hull of its
# own seam curve, which is what makes the shape well posed at all: the
# two outlines and the point at which they are joined determine it
# uniquely.  That join point is the interesting control -- it does not
# add or remove curvature, it only slides it around the seam, since
# Gauss-Bonnet pins the total at 4*pi (Orduno et al., Bridges 2016).
# The "Seam Curvature" colour attribute paints exactly that quantity.
#
# The generator also emits the exact flat development -- the two pieces
# to cut out -- because for a D-form the flat pieces are the input, so
# unrolling is free and exact rather than an approximation.
#
# The mathematics lives in the sibling `dform` engine package; this
# module is the Blender layer over it.
#
# References:
#   T. Wills, "D-Forms: 3D Forms from Two 2D Sheets," Bridges 2006,
#       pp. 503-510.
#   J. Sharp, "D-Forms and Developable Surfaces," Bridges 2005,
#       pp. 121-128; "D-Forms" (Tarquin, 2009) -- also the source of the
#       anti-D-form.
#   R. R. Orduno, N. Winard, S. Bierwagen, D. Shell, N. Kalantar,
#       A. Borhani, E. Akleman, "A Mathematical Approach to Obtain
#       Isoperimetric Shapes for D-Form Construction," Bridges 2016,
#       pp. 277-284.
#   E. Demaine, J. O'Rourke, "Geometric Folding Algorithms" (Cambridge,
#       2007), ch. 25 -- Alexandrov gluings.
#   H. Pottmann, J. Wallner, "Computational Line Geometry" (Springer,
#       2001) -- poses the D-form shape question.

bl_info = {
    "name": "D-Forms",
    "author": "Math Art project (after Wills / Sharp / Orduno et al.)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > D-Form",
    "description": "D-forms: solids glued from two equal-perimeter sheets",
    "category": "Add Mesh",
}

try:
    from .dform import build_dform
    from .dform.curves import CURVE_KINDS
    from .sharp_creases import mark_sharp
except ImportError:  # flat import outside the package
    from dform import build_dform
    from dform.curves import CURVE_KINDS
    from sharp_creases import mark_sharp

try:
    import bpy
    import bmesh
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


_CURVE_ITEMS = [
    ('ELLIPSE', "Ellipse", "Circle or ellipse; the classic D-form pair"),
    ('SUPERELLIPSE', "Superellipse",
     "Lame curve: squarer or pincushioned as the exponent moves"),
    ('EGG', "Egg / Oval",
     "An oval fattened at one end, so the widest point is off-axis"),
    ('ROUNDED_POLYGON', "Rounded Polygon",
     "Regular polygon offset by a disc -- exactly convex at every "
     "corner radius"),
    ('CASSINI', "Cassinian Oval",
     "Product-of-distances oval; convex only for b/a >= sqrt(2), "
     "waisted below it"),
]


if _IN_BLENDER:

    class MESH_OT_dform_add(bpy.types.Operator):
        """Add a D-form: a solid glued from two flat equal-perimeter sheets"""
        bl_idname = "mesh.dform_add"
        bl_label = "D-Form"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('SEAM', "Seam (two sheets)",
                    "The classic D-form: two flat pieces of equal "
                    "perimeter glued edge to edge. Solved for"),
                   ('ANTI', "Anti (holes joined)",
                    "John Sharp's anti-D-form: two annuli glued along "
                    "their HOLE edges with the outer rims left free -- "
                    "an open saddle collar, not convex, carrying -4*pi "
                    "on the seam. Solved for"),
                   ('VESICA', "Folded Vesica Piscis",
                    "Two overlapping discs folded so their boundaries "
                    "meet, giving one cylindrical and two conical "
                    "patches with curved creases (Mundilova and Wills). "
                    "Exact -- no solver"),
                   ('KOMAN_SPIRAL', "Koman Spiral Sculpture",
                    "Ilhan Koman's spiral developable: a coiling spine "
                    "with flat blades springing off it, growing as a "
                    "logarithmic spiral. Exact -- no solver"),
                   ('KOMAN_CURL', "Koman Curled Strip",
                    "Ilhan Koman's spiral developable: planar fins in "
                    "one plane with arched panels between them, each "
                    "sliding under the next. Exact -- no solver")],
            default='SEAM')
        vesica_u: FloatProperty(
            name="Disc Offset u", default=0.5, min=0.06, max=0.94,
            description="Half the distance between the two disc "
                        "centres. u = 0.5 is the true Vesica Piscis, "
                        "where each circle passes through the other's "
                        "centre")
        vesica_h: FloatProperty(
            name="Apex Height", default=-1.0, min=-1.0, max=1.0,
            description="Height of the cone apexes. Leave at -1 for the "
                        "height whose unrolled creases are exactly "
                        "circular; the fold only exists up to "
                        "h = 1 - u^2")
        panels: IntProperty(
            name="Panels", default=32, min=5, max=160,
            description="Tongues around the ring. With Close Ring on "
                        "they always make exactly one full turn")
        close_ring: BoolProperty(
            name="Close Ring", default=True,
            description="Derive the slide so the panels close a full "
                        "turn (a = 2*pi/n). Off, the slide below is "
                        "used as given and the ring stops wherever it "
                        "happens to -- 24 panels at d = 0.06 only get "
                        "164 degrees round")
        slide: FloatProperty(
            name="Slide d", default=0.10, min=0.005, max=0.5,
            description="How far each tongue slides under its "
                        "neighbour, when Close Ring is off. The turn "
                        "per tongue is arctan(2d/h)")
        skew: FloatProperty(
            name="Pinwheel Skew", default=0.5, min=-1.4, max=1.4,
            description="Swing each tongue away from radial, which is "
                        "what makes the ring read as a pinwheel")
        tilt: FloatProperty(
            name="Tongue Lean", default=0.42, min=0.0, max=1.2,
            description="How far each tongue leans as it rides up on "
                        "the ones beneath it. At zero they lie coplanar "
                        "and read as a solid disc")
        hole: FloatProperty(
            name="Hole", default=0.45, min=0.05, max=2.0,
            description="Inner radius, as a fraction of the tongue "
                        "length")
        lean: FloatProperty(
            name="Blade Lean", default=1.0, min=0.0, max=1.57,
            description="Tilt of each blade out of the coil plane")
        blade_len: FloatProperty(
            name="Blade Length", default=1.7, min=0.1, max=4.0,
            description="Blade length as a multiple of the local "
                        "spiral radius, so the form stays self-similar")
        blade_width: FloatProperty(
            name="Blade Width", default=0.6, min=0.05, max=2.0,
            description="Blade width, also relative to the local radius")
        twist: FloatProperty(
            name="Blade Twist", default=0.0, min=0.0, max=1.5,
            description="Turn each blade's far edge against its root, "
                        "for the look of Koman's metal pieces. Note "
                        "this forfeits developability -- a twisted "
                        "blade can no longer be cut from flat stock")
        growth: FloatProperty(
            name="Spiral Growth", default=0.0, min=-0.12, max=0.12,
            description="Shrink or grow the tongues geometrically along "
                        "the ribbon. The ring radius goes as h/2, so "
                        "this is what opens it into a spiral -- and it "
                        "is the parameter Koman himself varied")

        kind_a: EnumProperty(name="Piece A", items=_CURVE_ITEMS,
                             default='ELLIPSE')
        kind_b: EnumProperty(name="Piece B", items=_CURVE_ITEMS,
                             default='EGG')
        aspect_a: FloatProperty(
            name="A Aspect", default=0.6, min=0.1, max=1.0,
            description="Width-to-length ratio of the first outline")
        aspect_b: FloatProperty(
            name="B Aspect", default=0.8, min=0.1, max=1.0,
            description="Width-to-length ratio of the second outline")
        super_n: FloatProperty(
            name="Superellipse n", default=3.0, min=1.05, max=8.0,
            description="Lame exponent: 2 is an ellipse, higher is squarer")
        egg: FloatProperty(
            name="Egg Bias", default=0.35, min=0.0, max=0.8,
            description="How far the widest point moves off the short axis")
        sides_a: IntProperty(name="A Sides", default=3, min=3, max=12)
        sides_b: IntProperty(name="B Sides", default=5, min=3, max=12)
        corner: FloatProperty(
            name="Corner Radius", default=0.35, min=0.02, max=0.95,
            description="Rounding of the polygon corners")
        cassini: FloatProperty(
            name="Cassini b/a", default=1.6, min=1.02, max=3.0,
            description="Below sqrt(2) = 1.414 the oval is waisted, so "
                        "it is no longer convex and the D-form theorems "
                        "no longer apply")

        hole_kind_a: EnumProperty(
            name="Hole A", items=_CURVE_ITEMS, default='ELLIPSE',
            description="Outline of the first piece's hole. In ANTI the "
                        "HOLES are glued, so these two are the pair "
                        "whose perimeters get matched")
        hole_kind_b: EnumProperty(
            name="Hole B", items=_CURVE_ITEMS, default='ELLIPSE')
        hole_aspect_a: FloatProperty(
            name="Hole A Aspect", default=0.6, min=0.1, max=1.0)
        hole_aspect_b: FloatProperty(
            name="Hole B Aspect", default=1.0, min=0.1, max=1.0)
        hole_scale: FloatProperty(
            name="Hole Size", default=0.4, min=0.1, max=0.8,
            description="Hole size relative to the rim. Clamped so the "
                        "hole always fits inside its own piece")

        segments: IntProperty(
            name="Seam Segments", default=72, min=16, max=240,
            description="Vertices shared by the two pieces along the seam")
        join_offset: FloatProperty(
            name="Join Offset", default=0.25, min=0.0, max=1.0,
            description="Where the two outlines first meet. This is the "
                        "shape control: it cannot change the total "
                        "curvature (always 4*pi) but it slides it around "
                        "the seam, often changing the form completely")
        flip: BoolProperty(
            name="Flip Second Piece", default=False,
            description="Glue the second outline the other way round, "
                        "giving a more twisted form from the same pair")
        quality: IntProperty(
            name="Solver Quality", default=900, min=120, max=4000,
            description="Solver iteration budget. Raise it if the form "
                        "looks creased or the report shows high strain. "
                        "Above 72 segments the solve is done coarse-first "
                        "and resampled up, so raising the segment count "
                        "costs resolution but not convergence")

        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        make_net: BoolProperty(
            name="Flat Development", default=False,
            description="Also add the two pieces laid out flat -- the "
                        "exact shapes to cut out and glue")
        sharp_seam: BoolProperty(
            name="Sharp Seam", default=True,
            description="Mark the seam as a sharp edge (and a subdivision "
                        "crease). A D-form is smooth everywhere EXCEPT "
                        "the seam, which is where all of its curvature "
                        "sits, so shading it smooth across the join is "
                        "wrong -- it is a real crease in the paper")
        colour_mu: BoolProperty(
            name="Seam Curvature Attribute", default=True,
            description="Store the angle defect at each seam vertex as a "
                        "'mu' colour attribute; it is zero away from the "
                        "seam and sums to 4*pi along it")
        shade_smooth: BoolProperty(name="Shade Smooth", default=True)

        def execute(self, context):
            try:
                d = build_dform(
                    mode=self.mode, kind_a=self.kind_a, kind_b=self.kind_b,
                    aspect_a=self.aspect_a, aspect_b=self.aspect_b,
                    super_n=self.super_n, egg=self.egg,
                    sides_a=self.sides_a, sides_b=self.sides_b,
                    corner=self.corner, cassini=self.cassini,
                    segments=self.segments, join_offset=self.join_offset,
                    flip=self.flip, quality=self.quality, scale=self.scale,
                    hole_kind_a=self.hole_kind_a,
                    hole_kind_b=self.hole_kind_b,
                    hole_aspect_a=self.hole_aspect_a,
                    hole_aspect_b=self.hole_aspect_b,
                    hole_scale=self.hole_scale,
                    vesica_u=self.vesica_u, vesica_h=self.vesica_h,
                    panels=self.panels,
                    slide=(-1.0 if self.close_ring else self.slide),
                    growth=self.growth, skew=self.skew, hole=self.hole,
                    tilt=self.tilt, lean=self.lean,
                    blade_len=self.blade_len,
                    blade_width=self.blade_width, twist=self.twist)
            except Exception as exc:  # noqa: BLE001 - report, never crash
                self.report({'ERROR'}, f"D-form failed: {exc}")
                return {'CANCELLED'}

            name = {'VESICA': "Vesica Fold", 'ANTI': "Anti-D-Form",
                    'KOMAN_SPIRAL': "Koman Spiral",
                    'KOMAN_CURL': "Koman Curl"}.get(self.mode, "D-Form")
            obj = self._mesh_object(context, name, d.verts, d.faces)
            if self.sharp_seam:
                self._mark_seam(obj.data, d.sharp_edges)
            if self.colour_mu and len(d.seam):
                self._paint_mu(obj.data, d)

            if self.make_net and len(d.dev_faces):
                span = float(d.verts[:, 0].max() - d.verts[:, 0].min())
                net = self._mesh_object(context, "D-Form Net",
                                        d.dev_verts, d.dev_faces,
                                        smooth=False)
                net.location = (net.location[0],
                                net.location[1] - 1.2 * max(span, 0.1),
                                net.location[2])
                obj.select_set(True)
                context.view_layer.objects.active = obj

            s = d.stats
            if self.mode == 'SEAM':
                self.report(
                    {'INFO'},
                    f"V={len(d.verts)} F={len(d.faces)}  strain "
                    f"{100*s['strain']:.2f}%  seam curvature "
                    f"{s['mu_total']/3.14159265:.2f}pi (4pi ideal)  "
                    f"{s['iterations']} its")
            elif self.mode == 'ANTI':
                # `crossings` is the honest number: the two sheets are
                # zero-thickness here and real paper rests in contact,
                # so they pass through each other in places
                self.report(
                    {'INFO'},
                    f"V={len(d.verts)} F={len(d.faces)}  strain "
                    f"{100*s['strain']:.2f}%  seam curvature "
                    f"{s['mu_total']/3.14159265:.2f}pi (-4pi ideal)  "
                    f"folds p99 {s.get('fold_p99', 0.0):.0f} deg  "
                    f"{s.get('crossings', 0)} sheet crossings")
            elif self.mode == 'VESICA':
                self.report(
                    {'INFO'},
                    f"V={len(d.verts)} F={len(d.faces)}  exact fold, "
                    f"u={s['u']:.2f}  h_max={s['h_max']:.3f}  "
                    f"circular creases at h={s['h_circular']:.3f}")
            elif self.mode == 'KOMAN_SPIRAL':
                self.report(
                    {'INFO'},
                    f"V={len(d.verts)} F={len(d.faces)}  exact, "
                    f"{s['blades']} blades, growth {s['growth']:.3f}"
                    + ("  (twisted: not developable)"
                       if s['twist'] > 0 else "  (blades planar)"))
            else:
                self.report(
                    {'INFO'},
                    f"V={len(d.verts)} F={len(d.faces)}  exact, "
                    f"{s['panels']} panels turning "
                    f"{s['turn_per_panel']*57.2958:.1f} deg each")
            return {'FINISHED'}

        def _mesh_object(self, context, name, verts, faces, smooth=None):
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(v) for v in verts], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            if smooth is None:
                smooth = self.shade_smooth
            me.polygons.foreach_set('use_smooth',
                                    [bool(smooth)] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            return obj

        def _mark_seam(self, me, sharp_edges):
            """Make the folds real edges: sharp for shading, creased
            for subdivision.

            The seam carries ALL of a D-form's curvature -- the two
            sheets meet there at a genuine angle, which is exactly what
            makes it a D-form -- so smoothing normals across it reads as
            a soft bulge instead of a folded edge.  The closed-form
            modes fold the paper as well, and their creases are listed
            here too.
            """
            return mark_sharp(me, sharp_edges)

        def _paint_mu(self, me, d):
            """Angle defect per vertex: the seam's curvature, as colour."""
            mu = [0.0] * len(me.vertices)
            peak = max((abs(float(x)) for x in d.mu), default=0.0)
            if peak <= 1e-12:
                return
            for i, v in zip(d.seam, d.mu):
                if 0 <= int(i) < len(mu):
                    mu[int(i)] = max(0.0, float(v) / peak)
            att = me.color_attributes.new(name="mu", type='FLOAT_COLOR',
                                          domain='POINT')
            for i, m in enumerate(mu):
                att.data[i].color = (m, 0.35 * m, 1.0 - m, 1.0)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')

            if self.mode in ('SEAM', 'ANTI'):
                head = "Rims" if self.mode == 'ANTI' else "Outlines"
                col = lay.column(heading=head)
                col.prop(self, 'kind_a')
                self._curve_knobs(col, self.kind_a, 'a')
                col.separator()
                col.prop(self, 'kind_b')
                self._curve_knobs(col, self.kind_b, 'b')

                if self.mode == 'ANTI':
                    col = lay.column(heading="Holes (these are glued)")
                    col.prop(self, 'hole_kind_a')
                    col.prop(self, 'hole_aspect_a')
                    col.separator()
                    col.prop(self, 'hole_kind_b')
                    col.prop(self, 'hole_aspect_b')
                    col.prop(self, 'hole_scale')

                lay.separator()
                lay.prop(self, 'join_offset')
                lay.prop(self, 'flip')
                lay.prop(self, 'segments')
                lay.prop(self, 'quality')
            elif self.mode == 'VESICA':
                lay.separator()
                lay.prop(self, 'vesica_u')
                lay.prop(self, 'vesica_h')
                lay.prop(self, 'segments')
            elif self.mode == 'KOMAN_SPIRAL':
                lay.separator()
                lay.prop(self, 'panels', text="Blades")
                lay.prop(self, 'growth')
                lay.prop(self, 'skew')
                lay.prop(self, 'lean')
                lay.prop(self, 'blade_len')
                lay.prop(self, 'blade_width')
                lay.prop(self, 'twist')
                lay.prop(self, 'segments')
            else:
                lay.separator()
                lay.prop(self, 'panels')
                lay.prop(self, 'close_ring')
                if not self.close_ring:
                    lay.prop(self, 'slide')
                lay.prop(self, 'skew')
                lay.prop(self, 'tilt')
                lay.prop(self, 'hole')
                lay.prop(self, 'growth')
                lay.prop(self, 'segments')

            lay.separator()
            lay.prop(self, 'scale')
            if self.mode in ('SEAM', 'ANTI'):
                lay.prop(self, 'make_net')
                lay.prop(self, 'colour_mu')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'sharp_seam')

        def _curve_knobs(self, col, kind, which):
            col.prop(self, f'aspect_{which}')
            if kind == 'SUPERELLIPSE':
                col.prop(self, 'super_n')
            elif kind == 'EGG':
                col.prop(self, 'egg')
            elif kind == 'ROUNDED_POLYGON':
                col.prop(self, f'sides_{which}')
                col.prop(self, 'corner')
            elif kind == 'CASSINI':
                col.prop(self, 'cassini')

    def _menu_func(self, context):
        self.layout.operator("mesh.dform_add", icon='MOD_CLOTH')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_dform_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_dform_add)


def _selftest():
    import numpy as np

    ok = True

    # the operator's own defaults have to produce a usable D-form -- the
    # panel is where every user starts, so a bad default is a bad feature
    d = build_dform(kind_a='ELLIPSE', kind_b='EGG', aspect_a=0.6,
                    aspect_b=0.8, egg=0.35, segments=56, join_offset=0.25,
                    quality=700)
    good = (len(d.faces) > 0 and d.stats['strain'] < 0.01
            and abs(d.stats['mu_total'] - 4 * np.pi) < 0.02 * 4 * np.pi)
    ok &= good
    print(f"dform_generator: default parameters build a D-form "
          f"(strain {d.stats['strain']:.4f}, seam {d.stats['mu_total']:.3f}) "
          f"{'OK' if good else 'FAIL'}")

    # every curve in the menu must be offerable as either piece
    bad = []
    for kind in CURVE_KINDS:
        try:
            x = build_dform(kind_a=kind, kind_b='ELLIPSE', segments=40,
                            quality=200)
            if len(x.faces) == 0:
                bad.append(kind)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{kind}({type(exc).__name__})")
    good = not bad
    ok &= good
    print(f"dform_generator: every menu curve builds "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # the net must carry the same faces as the form, or the cut sheet
    # does not correspond to the object the user is looking at
    good = len(d.dev_faces) == len(d.faces)
    ok &= good
    print(f"dform_generator: net face count matches the form "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform_generator self-test failed")
