
# Relief Panel generator for Blender.
#
# A relief panel is a surface whose height is a scalar field h(x, y): a
# plate carved by mathematics rather than assembled from parts.  The
# field is built compositionally -- pattern, orientation, warp, transfer
# -- so that the organic and the geometric can be mixed rather than
# chosen between.  A drumhead mode pushed through a smooth domain warp
# and a crest-sharpening transfer becomes drapery without ceasing to be
# a drumhead mode.
#
# The design idea the generator turns on: ridges that fan, split and
# sweep across a panel are an ORIENTATION phenomenon, not an amplitude
# one.  A wave with a single global direction cannot produce them
# however its amplitude is modulated.  So the direction field, and the
# phase that follows it, are core controls rather than refinements.
#
# The mathematics is in the sibling `relief` engine package (NumPy
# only, no bpy); this module is the Blender layer over it.
#
# References:
# - Robert T. Frankot and Rama Chellappa, "A Method for Enforcing
#   Integrability in Shape from Shading Algorithms", IEEE Transactions
#   on Pattern Analysis and Machine Intelligence 10(4), 1988, 439-451
#   -- the spectral least-squares phase integration that lets a wave
#   follow a direction field.
# - Felix Knoeppel, Keenan Crane, Ulrich Pinkall and Peter Schroeder,
#   "Stripe Patterns on Surfaces", ACM Transactions on Graphics 34(4)
#   (SIGGRAPH 2015), article 39 -- the complex-phase formulation whose
#   branch points are the bifurcations seen in carved drapery.
# - Ken Perlin, "An Image Synthesizer", Computer Graphics 19(3)
#   (SIGGRAPH 1985), 287-296 -- gradient noise and phase perturbation.
# - Franz Josef von Gerstner (1802); Alain Fournier and William T.
#   Reeves, "A Simple Model of Ocean Waves", Computer Graphics 20(4)
#   (SIGGRAPH 1986), 75-84; Jerry Tessendorf, "Simulating Ocean Water",
#   SIGGRAPH course notes, 1999-2004 -- the trochoidal profile with
#   sharp crests and broad troughs that the Steepness control emulates.
# - Benoit B. Mandelbrot, "The Fractal Geometry of Nature", Freeman,
#   1982; Dietmar Saupe, "Algorithms for random fractals", in Peitgen
#   and Saupe (eds.), "The Science of Fractal Images", Springer, 1988.
# - Mary D. Waller, "Chladni Figures: A Study in Symmetry", G. Bell and
#   Sons, 1961 -- sand collects on the nodal lines, fine powder at the
#   antinodes: the physical reading of the Valleys/Ridges transfer pair.

bl_info = {
    "name": "Relief Panel",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Patterns > Relief Panel",
    "description": "Panels whose height is a stack of pattern fields: "
                   "waves, fractals and ripples, steered by an "
                   "orientation field and warped",
    "category": "Add Mesh",
}

import numpy as np

try:
    from .relief import (FIELDS, FITS, FORMS, ORIENTATIONS, PRESETS, SHAPES,
                         build_relief)
    from .relief import transfer as _transfer
except ImportError:                       # flat import outside the package
    from relief import (FIELDS, FITS, FORMS, ORIENTATIONS, PRESETS, SHAPES,
                        build_relief)
    from relief import transfer as _transfer


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PRESET_ITEMS = [
        ('CUSTOM', "Custom", "Use the controls below"),
        ('DRAPERY', "Drapery",
         "Wave train steered by a curl field, ridged -- carved cloth"),
        ('DUNES', "Dunes", "Warped fractal surface, ridged"),
        ('POND', "Pond", "Interfering circular wavefronts"),
        ('TERRAIN', "Terrain", "Weierstrass-Mandelbrot relief"),
        ('BANDS', "Bands", "Spiral-steered wave with contrast"),
        ('CHLADNI', "Chladni Plate",
         "A real free-plate mode, sand-figure profile"),
        ('CHLADNI_FLOW', "Chladni (Melted)",
         "The same plate mode warped -- geometric structure, organic surface"),
        ('DRUMHEAD', "Drumhead", "Circular membrane mode on a disc"),
        ('ZERNIKE', "Zernike", "Optical aberration mode on a disc"),
        ('LASER', "Laser Mode", "Hermite-Gauss TEM transverse mode"),
    ]

    _FIELD_ITEMS = [(k, v[0], v[1]) for k, v in sorted(FIELDS.items())]

    _ORIENT_ITEMS = [
        ('CONSTANT', "Constant", "One global direction"),
        ('RADIAL', "Radial", "Directions point away from the centre"),
        ('TANGENT', "Tangent", "Directions circle the centre"),
        ('SPIRAL', "Spiral", "Radially twisting directions"),
        ('CURL', "Curl (flow)",
         "Divergence-free swirl -- the drapery default"),
        ('GRADIENT', "Gradient", "Follows the slope of a smooth field"),
    ]

    _CURVE_ITEMS = [
        ('NONE', "None", "Leave the profile alone"),
        ('RIDGE', "Ridges", "Zero set becomes sharp crests"),
        ('ABS', "Valleys", "Zero set becomes sharp valleys"),
        ('GAMMA', "Gamma", "Broaden or sharpen crests"),
        ('SCURVE', "S-Curve", "Flatten troughs, plateau crests"),
        ('TERRACE', "Terrace", "Quantise into contour steps"),
        ('CLAMP', "Clamp", "Flat-topped mesas"),
    ]

    class MESH_OT_relief_panel_add(bpy.types.Operator):
        """Add a relief panel whose height is a pattern field"""
        bl_idname = "mesh.relief_panel_add"
        bl_label = "Relief Panel"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset", items=_PRESET_ITEMS, default='DRAPERY',
            description="Named starting point; choose Custom to drive the "
                        "controls below directly")

        # -- panel ----------------------------------------------------
        shape: EnumProperty(
            name="Shape",
            items=[(s, s.replace('_', ' ').title(), "") for s in SHAPES],
            default='RECT')
        width: FloatProperty(
            name="Width", default=2.0, min=0.01, max=100.0, unit='LENGTH')
        aspect: FloatProperty(
            name="Aspect", default=1.0, min=0.05, max=20.0,
            description="Height / width. The vertical sample count is "
                        "derived from this so cells stay square")
        resolution: IntProperty(
            name="Resolution", default=256, min=8, max=1024,
            description="Samples across the width; the vertical count "
                        "follows from Aspect")
        border: FloatProperty(
            name="Border", default=0.0, min=0.0, max=0.5,
            description="Raised-cosine margin fading the relief to a flat "
                        "rim (fraction of the short side)")

        # -- pattern --------------------------------------------------
        field: EnumProperty(name="Pattern", items=_FIELD_ITEMS,
                            default='WAVE_TRAIN')
        wavelength: FloatProperty(
            name="Wavelength", default=0.5, min=0.005, max=20.0,
            unit='LENGTH')
        angle: FloatProperty(name="Angle", default=0.0, min=-6.2832,
                             max=6.2832, unit='ROTATION')
        steepness: FloatProperty(
            name="Steepness", default=0.0, min=0.0, max=1.0,
            description="Sharpen crests and broaden troughs, toward the "
                        "trochoidal profile of a real wave")
        count: IntProperty(name="Waves", default=3, min=1, max=24)
        spread: FloatProperty(name="Spread", default=0.4, min=0.0, max=3.1416,
                              unit='ROTATION')
        sources: IntProperty(name="Sources", default=3, min=1, max=32)
        seed: IntProperty(name="Seed", default=1, min=0, max=100000)

        method: EnumProperty(
            name="Fractal",
            items=[('FBM', "Fractional Brownian", "Hurst exponent is the knob"),
                   ('WEIERSTRASS', "Weierstrass-Mandelbrot",
                    "Fractal dimension is the knob")],
            default='FBM')
        hurst: FloatProperty(name="Hurst", default=0.7, min=0.05, max=0.99)
        dim: FloatProperty(name="Fractal Dimension", default=2.3, min=2.01,
                           max=2.95)
        octaves: IntProperty(name="Octaves", default=8, min=1, max=16)

        # -- vibration modes & optics ---------------------------------
        exact: BoolProperty(
            name="Exact Plate Solve", default=True,
            description="Solve the real free-plate eigenproblem "
                        "(Rayleigh-Ritz). Off uses Rayleigh's cosine "
                        "approximation with a freely mixable blend")
        mode_index: IntProperty(
            name="Mode", default=6, min=1, max=40,
            description="Which plate mode, counting from the first "
                        "non-rigid-body figure")
        poisson: FloatProperty(
            name="Poisson Ratio", default=0.225, min=0.0, max=0.49,
            description="Material constant. Ritz used 0.225 to match "
                        "Chladni's glass plates; metals are near 0.33")
        ritz: IntProperty(
            name="Basis Size", default=10, min=4, max=16,
            description="Ritz basis order; the eigenproblem is "
                        "(n+1)^2 square")
        mode_m: IntProperty(name="Mode m", default=2, min=0, max=24)
        mode_n: IntProperty(name="Mode n", default=3, min=0, max=24)
        chi: FloatProperty(
            name="Blend", default=1.0, min=-1.0, max=1.0,
            description="Mix of the swapped cosine pair. For a real plate "
                        "this is not free: equal-parity pairs occur only at "
                        "+1 and -1, at two different frequencies")
        zern_n: IntProperty(name="Zernike n", default=4, min=0, max=20)
        zern_m: IntProperty(name="Zernike m", default=2, min=-20, max=20)
        waist: FloatProperty(name="Beam Waist", default=0.5, min=0.05,
                             max=2.0)

        # -- orientation & warp ---------------------------------------
        orient: EnumProperty(name="Orientation", items=_ORIENT_ITEMS,
                             default='CONSTANT')
        orient_freq: FloatProperty(name="Field Scale", default=0.5, min=0.05,
                                   max=8.0)
        swirl: FloatProperty(name="Swirl", default=1.0, min=-8.0, max=8.0)
        warp: FloatProperty(
            name="Warp", default=0.0, min=0.0, max=2.0,
            description="Displace the coordinates the pattern is evaluated "
                        "at -- what turns a regular field into flowing cloth")
        warp_iters: IntProperty(name="Warp Steps", default=2, min=1, max=4)

        # -- transfer -------------------------------------------------
        curve: EnumProperty(name="Profile", items=_CURVE_ITEMS, default='NONE')
        curve_amount: FloatProperty(name="Amount", default=1.0, min=0.05,
                                    max=4.0)
        levels: IntProperty(name="Levels", default=6, min=2, max=64)

        # -- output ---------------------------------------------------
        depth: FloatProperty(name="Relief Depth", default=0.25, min=0.0,
                             max=10.0, unit='LENGTH')
        form: EnumProperty(
            name="Form",
            items=[('SLAB', "Slab", "Watertight panel with a flat back"),
                   ('SHEET', "Sheet", "Open surface, no back")],
            default='SLAB')
        base_thickness: FloatProperty(name="Base", default=0.1, min=0.0,
                                      max=10.0, unit='LENGTH')
        fit: EnumProperty(
            name="Fit",
            items=[('FOOTPRINT', "Footprint",
                    "Fit the panel outline; relief depth stays proportional"),
                   ('CUBE', "2 m Cube", "Fit the whole bounding box"),
                   ('NONE', "None", "Literal metres")],
            default='FOOTPRINT')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def _params(self):
            p = dict(
                shape=self.shape, width=self.width, aspect=self.aspect,
                resolution=self.resolution, border=self.border,
                field=self.field, wavelength=self.wavelength,
                angle=self.angle, steepness=self.steepness,
                count=self.count, spread=self.spread, sources=self.sources,
                seed=self.seed, method=self.method, hurst=self.hurst,
                dim=self.dim, octaves=self.octaves,
                exact=self.exact, mode_index=self.mode_index,
                poisson=self.poisson, ritz=self.ritz,
                mode_m=self.mode_m, mode_n=self.mode_n, chi=self.chi,
                zern_n=self.zern_n, zern_m=self.zern_m, waist=self.waist,
                orient=self.orient, orient_freq=self.orient_freq,
                swirl=self.swirl, warp=self.warp,
                warp_iters=self.warp_iters,
                curve=self.curve, curve_amount=self.curve_amount,
                levels=self.levels,
                depth=self.depth, form=self.form,
                base_thickness=self.base_thickness,
                fit=self.fit, scale=self.scale)
            if self.preset != 'CUSTOM':
                # A preset overrides only what it names; panel size,
                # resolution and output form stay under the user's control.
                # `shape` is NOT kept -- a drumhead mode on a rectangular
                # panel is simply the wrong domain, so a preset may set it.
                keep = ('width', 'aspect', 'resolution', 'form',
                        'base_thickness', 'fit', 'scale', 'border')
                over = dict(PRESETS[self.preset])
                for k in keep:
                    over.pop(k, None)
                p.update(over)
            return p

        def execute(self, context):
            try:
                verts, faces, info = build_relief(**self._params())
            except (ValueError, MemoryError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            if not len(faces):
                self.report({'ERROR'}, "empty panel: the mask removed "
                                       "every sample")
                return {'CANCELLED'}

            me = bpy.data.meshes.new("Relief Panel")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))

            # Height as a vertex attribute, so materials can read the relief
            # without recomputing it.
            try:
                attr = me.attributes.new("height", 'FLOAT', 'POINT')
                z = np.asarray(verts, dtype=float)[:, 2]
                rng = float(z.max() - z.min())
                attr.data.foreach_set(
                    'value', ((z - z.min()) / rng if rng > 1e-12
                              else z * 0.0).tolist())
            except (RuntimeError, AttributeError):
                pass                      # attribute is a nicety, not a gate

            me.update()
            obj = bpy.data.objects.new("Relief Panel", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            msg = ("%dx%d samples, V=%d F=%d"
                   % (info['nx'], info['ny'], len(me.vertices),
                      len(me.polygons)))
            if info.get('aliasing'):
                self.report({'WARNING'},
                            msg + " -- wavelength is only %.1f samples; "
                            "raise Resolution or Wavelength or the pattern "
                            "will alias" % info['wavelength_cells'])
            else:
                self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')

            box = lay.box()
            box.label(text="Panel")
            box.prop(self, 'shape')
            box.prop(self, 'width')
            box.prop(self, 'aspect')
            box.prop(self, 'resolution')
            box.prop(self, 'border')

            custom = self.preset == 'CUSTOM'
            box = lay.box()
            box.label(text="Pattern" if custom else "Pattern (from preset)")
            col = box.column()
            col.enabled = custom
            col.prop(self, 'field')
            if self.field in ('WAVE', 'WAVE_TRAIN'):
                col.prop(self, 'wavelength')
                col.prop(self, 'angle')
                col.prop(self, 'steepness')
                if self.field == 'WAVE_TRAIN':
                    col.prop(self, 'count')
                    col.prop(self, 'spread')
            elif self.field == 'RIPPLE':
                col.prop(self, 'wavelength')
                col.prop(self, 'sources')
            elif self.field == 'FBM':
                col.prop(self, 'method')
                if self.method == 'FBM':
                    col.prop(self, 'hurst')
                else:
                    col.prop(self, 'dim')
                col.prop(self, 'octaves')
            elif self.field == 'CHLADNI':
                col.prop(self, 'exact')
                if self.exact:
                    col.prop(self, 'mode_index')
                    col.prop(self, 'poisson')
                    col.prop(self, 'ritz')
                else:
                    col.prop(self, 'mode_m')
                    col.prop(self, 'mode_n')
                    col.prop(self, 'chi')
            elif self.field in ('DRUMHEAD', 'MEMBRANE'):
                col.prop(self, 'mode_m')
                col.prop(self, 'mode_n')
                if self.field == 'DRUMHEAD' and self.shape != 'DISC':
                    col.label(text="Drumhead wants the Disc shape",
                              icon='INFO')
            elif self.field == 'ZERNIKE':
                col.prop(self, 'zern_n')
                col.prop(self, 'zern_m')
                if self.shape != 'DISC':
                    col.label(text="Zernike is defined on the disc",
                              icon='INFO')
            elif self.field == 'HERMITE':
                col.prop(self, 'mode_m')
                col.prop(self, 'mode_n')
                col.prop(self, 'waist')
            col.prop(self, 'seed')

            box = lay.box()
            box.label(text="Orientation & Warp")
            col = box.column()
            col.enabled = custom
            col.prop(self, 'orient')
            if self.orient in ('CURL', 'GRADIENT'):
                col.prop(self, 'orient_freq')
            if self.orient == 'SPIRAL':
                col.prop(self, 'swirl')
            col.prop(self, 'warp')
            if self.warp > 0.0:
                col.prop(self, 'warp_iters')

            box = lay.box()
            box.label(text="Profile")
            col = box.column()
            col.enabled = custom
            col.prop(self, 'curve')
            if self.curve in ('GAMMA', 'SCURVE', 'CLAMP'):
                col.prop(self, 'curve_amount')
            if self.curve == 'TERRACE':
                col.prop(self, 'levels')

            box = lay.box()
            box.label(text="Output")
            box.prop(self, 'depth')
            box.prop(self, 'form')
            if self.form == 'SLAB':
                box.prop(self, 'base_thickness')
            box.prop(self, 'fit')
            box.prop(self, 'scale')
            box.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.relief_panel_add", icon='MOD_DISPLACE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_relief_panel_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_relief_panel_add)


def _selftest():
    """Build every preset through the same path the operator uses."""
    ok = True
    for name in sorted(PRESETS):
        p = dict(PRESETS[name])
        p.setdefault('resolution', 81)
        p['resolution'] = 81
        verts, faces, info = build_relief(**p)
        finite = np.isfinite(np.asarray(verts)).all()
        print("relief_generator: %-8s V=%-6d F=%-6d %dx%d finite=%s"
              % (name, len(verts), len(faces), info['nx'], info['ny'], finite))
        ok = ok and finite and len(faces) > 0

    # Every profile curve survives the full pipeline.
    from .relief import transfer as t
    for curve in t.CURVES:
        v, f, _ = build_relief(resolution=49, curve=curve)
        ok = ok and len(f) > 0 and np.isfinite(np.asarray(v)).all()
    print("relief_generator: all %d profile curves build" % len(t.CURVES))

    # Every orientation field survives it too.
    from .relief import warp as w
    for kind in w.ORIENTATIONS:
        v, f, _ = build_relief(resolution=49, field='WAVE', orient=kind)
        ok = ok and len(f) > 0 and np.isfinite(np.asarray(v)).all()
    print("relief_generator: all %d orientation fields build"
          % len(w.ORIENTATIONS))

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
