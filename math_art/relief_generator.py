
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
                         build_relief, ordered_fields)
    from .relief.kernels import KERNELS
    from .relief import transfer as _transfer
except ImportError:                       # flat import outside the package
    from relief import (FIELDS, FITS, FORMS, ORIENTATIONS, PRESETS, SHAPES,
                        build_relief, ordered_fields)
    from relief.kernels import KERNELS
    from relief import transfer as _transfer


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _evaluated_mesh(context, obj):
        """The object's mesh with modifiers applied, plus its world matrix."""
        deps = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(deps)
        me = ev.to_mesh()
        if me is None:
            return None, None, None
        V = np.empty((len(me.vertices), 3))
        me.vertices.foreach_get('co', V.ravel())
        tris = []
        me.calc_loop_triangles()
        for t in me.loop_triangles:
            tris.append(tuple(t.vertices))
        M = np.array(obj.matrix_world.to_4x4())
        V = (np.c_[V, np.ones(len(V))] @ M.T)[:, :3]
        ev.to_mesh_clear()
        return V, np.array(tris, dtype=int) if tris else None, me

    def _area_weighted_samples(V, tris, count, seed=1):
        """Uniform samples over the surface, area-weighted.

        Raw vertices over-weight densely subdivided regions, so a smooth
        sphere and a faceted one would imprint differently for no reason.
        The square root is what makes the barycentric draw area-uniform:

            P = (1 - sqrt(r1)) A + sqrt(r1)(1 - r2) B + sqrt(r1) r2 C

        *Osada, Funkhouser, Chazelle and Dobkin, "Shape Distributions",
        ACM TOG 21(4), 2002.*
        """
        if tris is None or not len(tris):
            return V
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        A, B, C = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        tot = float(area.sum())
        if tot <= 0.0:
            return V
        cum = np.cumsum(area) / tot
        pick = np.searchsorted(cum, rng.random(int(count)))
        r1 = np.sqrt(rng.random(int(count)))[:, None]
        r2 = rng.random(int(count))[:, None]
        return ((1.0 - r1) * A[pick] + r1 * (1.0 - r2) * B[pick]
                + r1 * r2 * C[pick])

    def _fit_to_panel(P, width, height, margin=0.9):
        """Map the object's XY footprint into the panel, keeping aspect."""
        if P is None or not len(P):
            return P
        lo = P[:, :2].min(axis=0)
        hi = P[:, :2].max(axis=0)
        ext = np.maximum(hi - lo, 1e-9)
        s = margin * min(width / ext[0], height / ext[1])
        out = P.copy()
        out[:, :2] = (P[:, :2] - 0.5 * (lo + hi)) * s
        if out.shape[1] > 2:
            out[:, 2] = (P[:, 2] - P[:, 2].min()) * s
        return out

    def _depth_map(context, obj, X, Y):
        """Ray-cast the object from above; returns (depth, hit-mask).

        A height field is a graph, so one downward ray per sample is the whole
        story -- no undercut can be represented anyway.
        """
        from mathutils.bvhtree import BVHTree
        from mathutils import Vector
        V, tris, _ = _evaluated_mesh(context, obj)
        if V is None or tris is None or not len(tris):
            return None, None
        V = _fit_to_panel(V, float(np.abs(X).max()) * 2.0,
                          float(np.abs(Y).max()) * 2.0)
        bvh = BVHTree.FromPolygons([Vector(v) for v in V],
                                   [tuple(t) for t in tris])
        top = float(V[:, 2].max()) + 1.0
        down = Vector((0.0, 0.0, -1.0))
        depth = np.zeros(X.shape)
        hit = np.zeros(X.shape, dtype=bool)
        for j in range(X.shape[0]):
            for i in range(X.shape[1]):
                loc, _n, _idx, _d = bvh.ray_cast(
                    Vector((float(X[j, i]), float(Y[j, i]), top)), down)
                if loc is not None:
                    depth[j, i] = loc.z
                    hit[j, i] = True
        if hit.any():
            depth[~hit] = depth[hit].min()
        return depth, hit


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

    _FIELD_ITEMS = list(ordered_fields())

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

    _TILING_ITEMS = [
        ('NONE', "None", "Do not constrain the pattern"),
        ('TORUS', "Translation",
         "Tiles by repeating; wavevectors snap to the panel's dual lattice"),
        ('MIRROR', "Mirror",
         "Tiles by reflection; built from cosine modes, so the joint is "
         "smooth to every order"),
        ('ANTIMIRROR', "Alternating",
         "Neighbouring panels are the negative relief, meeting at the base "
         "plane"),
    ]

    class MESH_OT_relief_panel_add(bpy.types.Operator):
        """Add a relief panel whose height is a pattern field"""
        bl_idname = "mesh.relief_panel_add"
        bl_label = "Relief Panel"
        bl_options = {'REGISTER', 'UNDO'}

        def _preset_update(self, context):
            """Write a chosen preset's values into the real properties.

            The properties are the single source of truth; a preset is a
            *starting point* that fills them in, not a parallel setting that
            overrides them at build time.  That way switching to Custom
            inherits whatever the preset just set, instead of snapping back
            to the factory defaults -- which is what happens if the preset is
            only consulted while building.
            """
            if self.preset == 'CUSTOM':
                return
            for key, value in PRESETS.get(self.preset, {}).items():
                if not hasattr(self, key):
                    continue
                try:
                    setattr(self, key, value)
                except (TypeError, ValueError):
                    pass          # a preset key the property cannot hold

        preset: EnumProperty(
            name="Preset", items=_PRESET_ITEMS, default='DRAPERY',
            update=_preset_update,
            description="Named starting point. Choosing one fills in the "
                        "controls below, so switching to Custom keeps what "
                        "you were looking at")

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
        tiling: EnumProperty(
            name="Tiling", items=_TILING_ITEMS, default='NONE',
            description="Constrain the pattern so the panel abuts copies of "
                        "itself invisibly. Suppresses Warp, which cannot be "
                        "made periodic")

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

        # -- object layer ---------------------------------------------
        # An operator property cannot be a PointerProperty to an ID, so the
        # object is named and resolved at execute time.
        source: StringProperty(
            name="Object",
            description="Scene object to imprint (defaults to the active "
                        "object)")
        obj_mode: EnumProperty(
            name="Mode",
            items=[('SPLAT', "Splat",
                    "Sum a kernel at every point -- a density field"),
                   ('MAX', "Merge",
                    "Union of bumps rather than their sum, metaball-like"),
                   ('INTERPOLATE', "Drape",
                    "Interpolate a surface through the points"),
                   ('IMPRINT', "Press",
                    "Ray-cast the object's depth and compress it to read"),
                   ('ENGRAVE', "Engrave",
                    "Cut a groove along the object's projected outline")],
            default='SPLAT')
        sample: EnumProperty(
            name="Points",
            items=[('FACES', "Surface samples",
                    "Area-weighted samples -- mesh density does not bias the "
                    "result"),
                   ('VERTS', "Vertices", "The object's vertices as they are")],
            default='FACES')
        samples: IntProperty(name="Sample Count", default=2000, min=8,
                             max=200000)
        kernel: EnumProperty(
            name="Kernel",
            items=[(k, v[0], "%s support, %s at the edge" % (v[1], v[2]))
                   for k, v in sorted(KERNELS.items())],
            default='GAUSSIAN')
        sigma: FloatProperty(
            name="Radius", default=0.0, min=0.0, max=10.0, unit='LENGTH',
            description="Kernel radius; 0 uses the density-estimation rule "
                        "sigma = spread * N^(-1/6)")
        merge: FloatProperty(
            name="Merge", default=0.0, min=0.0, max=1.0,
            description="Round the seam where merged bumps meet (Merge mode)")
        power: FloatProperty(name="Falloff Power", default=2.0, min=0.5,
                             max=8.0)
        groove: FloatProperty(name="Groove Width", default=0.08, min=0.002,
                              max=1.0, unit='LENGTH')
        compress: EnumProperty(
            name="Compression",
            items=[('AHE', "Histogram",
                    "Adaptive histogram equalisation -- no solver, robust"),
                   ('GRADIENT', "Gradient domain",
                    "Attenuate gradients and reintegrate -- better on "
                    "organic shapes"),
                   ('LINEAR', "None",
                    "Raw depth; usually reads as a stepped blob")],
            default='AHE')
        alpha: FloatProperty(name="Attenuation", default=0.1, min=0.001,
                             max=2.0)
        beta: FloatProperty(name="Compression", default=0.85, min=0.1,
                            max=1.0)

        # -- scatter layer --------------------------------------------
        process: EnumProperty(
            name="Point Process",
            items=[('BLUE', "Blue noise",
                    "Poisson-disk: no clumping under the kernel's low-pass"),
                   ('HALTON', "Halton", "Low-discrepancy, deterministic"),
                   ('UNIFORM', "Uniform random", "Clumps, for comparison")],
            default='BLUE')
        points_n: IntProperty(name="Point Count", default=120, min=2,
                              max=20000)

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
                tiling=self.tiling,
                field=self.field, wavelength=self.wavelength,
                angle=self.angle, steepness=self.steepness,
                count=self.count, spread=self.spread, sources=self.sources,
                seed=self.seed, method=self.method, hurst=self.hurst,
                dim=self.dim, octaves=self.octaves,
                exact=self.exact, mode_index=self.mode_index,
                poisson=self.poisson, ritz=self.ritz,
                mode_m=self.mode_m, mode_n=self.mode_n, chi=self.chi,
                zern_n=self.zern_n, zern_m=self.zern_m, waist=self.waist,
                obj_mode=self.obj_mode, kernel=self.kernel,
                sigma=self.sigma, merge=self.merge, power=self.power,
                groove=self.groove, compress=self.compress,
                alpha=self.alpha, beta=self.beta,
                process=self.process, points_n=self.points_n,
                orient=self.orient, orient_freq=self.orient_freq,
                swirl=self.swirl, warp=self.warp,
                warp_iters=self.warp_iters,
                curve=self.curve, curve_amount=self.curve_amount,
                levels=self.levels,
                depth=self.depth, form=self.form,
                base_thickness=self.base_thickness,
                fit=self.fit, scale=self.scale)
            # No preset override here: choosing a preset has already written
            # its values into the properties (see `_preset_update`), so the
            # properties are the only source of truth and what the panel
            # shows is exactly what gets built.
            return p

        def invoke(self, context, event):
            if not self.source and context.active_object is not None:
                self.source = context.active_object.name
            # The update callback fires when the enum is *changed*, not for
            # its default, so the first invocation has to seed it explicitly
            # or the default preset would build from factory values.
            self._preset_update(context)
            return self.execute(context)

        def _object_inputs(self, context, params):
            """Resolve the source object into plain arrays for the engine."""
            obj = bpy.data.objects.get(self.source)
            if obj is None:
                obj = context.active_object
            if obj is None or obj.type not in {'MESH', 'CURVE', 'SURFACE',
                                               'FONT', 'META'}:
                self.report({'ERROR'},
                            "Object pattern needs a mesh-like source object; "
                            "pick one in the Object field")
                return False

            half_w = 0.5 * self.width
            half_h = 0.5 * self.width * self.aspect
            if self.obj_mode == 'IMPRINT':
                from .relief import make_grid
                X, Y, _ = make_grid(self.width, self.aspect, self.resolution)
                depth, hit = _depth_map(context, obj, X, Y)
                if depth is None:
                    self.report({'ERROR'}, "could not evaluate that object "
                                           "into a mesh")
                    return False
                params['depth_map'] = depth
                params['obj_mask'] = hit
                return True

            V, tris, _ = _evaluated_mesh(context, obj)
            if V is None or not len(V):
                self.report({'ERROR'}, "could not evaluate that object "
                                       "into a mesh")
                return False
            if self.sample == 'FACES' and tris is not None and len(tris):
                P = _area_weighted_samples(V, tris, self.samples, self.seed)
            else:
                P = V
            P = _fit_to_panel(P, 2.0 * half_w, 2.0 * half_h)
            params['points'] = P
            params['weights'] = None
            return True

        def execute(self, context):
            params = self._params()
            if params.get('field') == 'OBJECT':
                if not self._object_inputs(context, params):
                    return {'CANCELLED'}
            try:
                verts, faces, info = build_relief(**params)
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

            # Seed the object's layer stack from what was just built, so the
            # sidebar can carry on editing it.  Without this an Add-menu
            # panel is a dead end: the stack UI would offer only "New
            # Layered Panel", as though the thing just made were not one.
            _seed_stack(obj, self)

            msg = ("%dx%d samples, V=%d F=%d"
                   % (info['nx'], info['ny'], len(me.vertices),
                      len(me.polygons)))
            if info.get('tiling', 'NONE') != 'NONE':
                if info.get('untileable'):
                    self.report({'WARNING'},
                                "this pattern cannot tile: %s"
                                % info['untileable'])
                msg += ("  seam x%.2f/x%.2f"
                        % (info.get('seam_step', 0.0),
                           info.get('seam_curvature', 0.0)))
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
            box.prop(self, 'tiling')
            if self.tiling != 'NONE' and self.warp > 0.0:
                box.label(text="Warp is suppressed while tiling",
                          icon='INFO')

            custom = self.preset == 'CUSTOM'
            box = lay.box()
            box.label(text="Pattern" if custom
                      else "Pattern - switch to Custom to edit")
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
            elif self.field == 'OBJECT':
                col.prop_search(self, 'source', bpy.data, 'objects')
                col.prop(self, 'obj_mode')
                if self.obj_mode == 'IMPRINT':
                    col.prop(self, 'compress')
                    if self.compress == 'GRADIENT':
                        col.prop(self, 'alpha')
                        col.prop(self, 'beta')
                elif self.obj_mode == 'ENGRAVE':
                    col.prop(self, 'groove')
                else:
                    col.prop(self, 'sample')
                    if self.sample == 'FACES':
                        col.prop(self, 'samples')
                    col.prop(self, 'kernel')
                    col.prop(self, 'sigma')
                    if self.obj_mode == 'MAX':
                        col.prop(self, 'merge')
                    if self.obj_mode == 'INTERPOLATE':
                        col.prop(self, 'power')
            elif self.field == 'SCATTER':
                col.prop(self, 'process')
                col.prop(self, 'points_n')
                col.prop(self, 'kernel')
                col.prop(self, 'sigma')
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

    # ==================================================================
    # Multi-layer stack: a live, editable panel on the object itself
    # ==================================================================
    #
    # The redo panel is fine for one pattern but cannot host an
    # add/remove list, so the stack lives on the object as a property
    # group with a UIList and an explicit Rebuild -- the same shape the
    # Scherk-Collins generator uses.

    _BLEND_ITEMS = [
        ('ADD', "Add", "Sum -- the usual choice"),
        ('SUB', "Subtract", "Carve this layer out of what is below"),
        ('MUL', "Multiply", "Modulate the layers below"),
        ('SCREEN', "Screen", "Lighten; the inverse of multiply"),
        ('MAX', "Max", "Keep whichever is higher"),
        ('MIN', "Min", "Keep whichever is lower"),
    ]

    _MASK_ITEMS = [
        ('NONE', "None", "Apply everywhere"),
        ('RADIAL', "Radial", "Fade out from a centre"),
        ('LINEAR', "Linear", "Fade across the panel"),
        ('LAYER', "Layer", "Use an earlier layer as the mask"),
    ]

    class ReliefLayer(bpy.types.PropertyGroup):
        """One entry in a panel's pattern stack."""
        kind: EnumProperty(name="Pattern", items=_FIELD_ITEMS,
                           default='WAVE_TRAIN')
        amplitude: FloatProperty(name="Amount", default=1.0, min=-4.0,
                                 max=4.0)
        blend: EnumProperty(name="Blend", items=_BLEND_ITEMS, default='ADD')
        enabled: BoolProperty(name="Enabled", default=True)

        wavelength: FloatProperty(name="Wavelength", default=0.5, min=0.005,
                                  max=20.0, unit='LENGTH')
        angle: FloatProperty(name="Angle", default=0.0, min=-6.2832,
                             max=6.2832, unit='ROTATION')
        steepness: FloatProperty(name="Steepness", default=0.0, min=0.0,
                                 max=1.0)
        count: IntProperty(name="Waves", default=3, min=1, max=24)
        sources: IntProperty(name="Sources", default=3, min=1, max=32)
        seed: IntProperty(name="Seed", default=1, min=0, max=100000)
        mode_index: IntProperty(name="Mode", default=6, min=1, max=40)
        mode_m: IntProperty(name="m", default=2, min=0, max=24)
        mode_n: IntProperty(name="n", default=3, min=0, max=24)
        zern_n: IntProperty(name="Zernike n", default=4, min=0, max=20)
        zern_m: IntProperty(name="Zernike m", default=2, min=-20, max=20)
        hurst: FloatProperty(name="Hurst", default=0.7, min=0.05, max=0.99)

        orient: EnumProperty(name="Orientation", items=_ORIENT_ITEMS,
                             default='CONSTANT')
        orient_freq: FloatProperty(name="Field Scale", default=0.5, min=0.05,
                                   max=8.0)

        offset_x: FloatProperty(name="Offset X", default=0.0, min=-10.0,
                                max=10.0, unit='LENGTH')
        offset_y: FloatProperty(name="Offset Y", default=0.0, min=-10.0,
                                max=10.0, unit='LENGTH')
        rotation: FloatProperty(name="Rotation", default=0.0, min=-6.2832,
                                max=6.2832, unit='ROTATION')
        scale_x: FloatProperty(name="Scale X", default=1.0, min=0.01,
                               max=100.0)
        scale_y: FloatProperty(name="Scale Y", default=1.0, min=0.01,
                               max=100.0)

        mask: EnumProperty(name="Mask", items=_MASK_ITEMS, default='NONE')
        mask_width: FloatProperty(name="Mask Width", default=0.5, min=0.01,
                                  max=2.0)
        mask_angle: FloatProperty(name="Mask Angle", default=0.0, min=-6.2832,
                                  max=6.2832, unit='ROTATION')
        mask_layer: IntProperty(
            name="Mask Layer", default=0, min=0, max=63,
            description="Index of an EARLIER layer to use as the mask; a "
                        "layer can only reference ones below it")
        curve: EnumProperty(name="Profile", items=_CURVE_ITEMS,
                            default='NONE')
        curve_amount: FloatProperty(name="Amount", default=1.0, min=0.05,
                                    max=4.0)

    class ReliefPanelProps(bpy.types.PropertyGroup):
        """The whole panel: outline, output settings, and the layer stack."""
        shape: EnumProperty(
            name="Shape",
            items=[(s, s.replace('_', ' ').title(), "") for s in SHAPES],
            default='RECT')
        width: FloatProperty(name="Width", default=2.0, min=0.01, max=100.0,
                             unit='LENGTH')
        aspect: FloatProperty(name="Aspect", default=1.0, min=0.05, max=20.0)
        resolution: IntProperty(name="Resolution", default=192, min=8,
                                max=1024)
        border: FloatProperty(name="Border", default=0.0, min=0.0, max=0.5)
        tiling: EnumProperty(name="Tiling", items=_TILING_ITEMS,
                             default='NONE')
        warp: FloatProperty(name="Warp", default=0.0, min=0.0, max=2.0)
        warp_iters: IntProperty(name="Warp Steps", default=2, min=1, max=4)
        seed: IntProperty(name="Seed", default=1, min=0, max=100000)
        depth: FloatProperty(name="Relief Depth", default=0.25, min=0.0,
                             max=10.0, unit='LENGTH')
        form: EnumProperty(
            name="Form",
            items=[('SLAB', "Slab", "Watertight panel with a flat back"),
                   ('SHEET', "Sheet", "Open surface")],
            default='SLAB')
        base_thickness: FloatProperty(name="Base", default=0.1, min=0.0,
                                      max=10.0, unit='LENGTH')
        fit: EnumProperty(
            name="Fit",
            items=[('FOOTPRINT', "Footprint", ""), ('CUBE', "2 m Cube", ""),
                   ('NONE', "None", "")],
            default='FOOTPRINT')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        layers: bpy.props.CollectionProperty(type=ReliefLayer)
        active: IntProperty(name="Active Layer", default=0, min=0)
        is_panel: BoolProperty(default=False)

    class RELIEF_UL_layers(bpy.types.UIList):
        def draw_item(self, context, layout, data, item, icon, active_data,
                      active_prop, index):
            row = layout.row(align=True)
            row.prop(item, 'enabled', text="", emboss=False,
                     icon='CHECKBOX_HLT' if item.enabled
                     else 'CHECKBOX_DEHLT')
            label = dict(_FIELD_ITEMS_MAP).get(item.kind, item.kind)
            row.label(text="%d. %s" % (index, label))
            sub = row.row(align=True)
            sub.alignment = 'RIGHT'
            sub.label(text="%s  x%.2f" % (item.blend.title(), item.amplitude))

    # Properties copied from the single-shot operator into the first stack
    # layer, so an Add-menu panel and a sidebar panel are the same thing.
    _SEED_LAYER_KEYS = (
        'wavelength', 'angle', 'steepness', 'count', 'sources', 'seed',
        'mode_index', 'mode_m', 'mode_n', 'zern_n', 'zern_m', 'hurst',
        'orient', 'orient_freq', 'curve', 'curve_amount')
    _SEED_PANEL_KEYS = (
        'shape', 'width', 'aspect', 'resolution', 'border', 'tiling',
        'warp',
        'warp_iters', 'seed', 'depth', 'form', 'base_thickness', 'fit',
        'scale', 'smooth')

    def _seed_stack(obj, op):
        """Fill an object's layer stack from the single-shot operator."""
        props = obj.relief_panel
        props.is_panel = True
        props.layers.clear()
        for key in _SEED_PANEL_KEYS:
            if hasattr(op, key):
                try:
                    setattr(props, key, getattr(op, key))
                except (TypeError, ValueError):
                    pass
        lay = props.layers.add()
        lay.kind = op.field
        lay.amplitude = 1.0
        lay.blend = 'ADD'
        for key in _SEED_LAYER_KEYS:
            if hasattr(op, key) and hasattr(lay, key):
                try:
                    setattr(lay, key, getattr(op, key))
                except (TypeError, ValueError):
                    pass
        props.active = 0

    def _panel_params(props):
        """Turn the property group into a `build_relief` argument dict."""
        layers = []
        for lay in props.layers:
            if not lay.enabled:
                continue
            layers.append(dict(
                kind=lay.kind, amplitude=lay.amplitude, blend=lay.blend,
                wavelength=lay.wavelength, angle=lay.angle,
                steepness=lay.steepness, count=lay.count,
                sources=lay.sources, seed=lay.seed,
                mode_index=lay.mode_index, mode_m=lay.mode_m,
                mode_n=lay.mode_n, zern_n=lay.zern_n, zern_m=lay.zern_m,
                hurst=lay.hurst, orient=lay.orient,
                orient_freq=lay.orient_freq,
                offset_x=lay.offset_x, offset_y=lay.offset_y,
                rotation=lay.rotation, scale_x=lay.scale_x,
                scale_y=lay.scale_y,
                mask=lay.mask, mask_width=lay.mask_width,
                mask_angle=lay.mask_angle, mask_layer=lay.mask_layer,
                curve=lay.curve, curve_amount=lay.curve_amount))
        return dict(
            shape=props.shape, width=props.width, aspect=props.aspect,
            resolution=props.resolution, border=props.border,
            tiling=props.tiling, warp=props.warp, warp_iters=props.warp_iters, seed=props.seed,
            depth=props.depth, form=props.form,
            base_thickness=props.base_thickness, fit=props.fit,
            scale=props.scale, layers=layers)

    def _write_mesh(obj, verts, faces, smooth):
        """Replace the object's geometry IN PLACE.

        The mesh datablock is reused rather than swapped out: replacing it
        invalidates every reference anything else still holds and surfaces
        later as a stale-`Mesh` ReferenceError.
        """
        me = obj.data
        me.clear_geometry()
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        if smooth and len(me.polygons):
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        try:
            attr = me.attributes.get("height") or me.attributes.new(
                "height", 'FLOAT', 'POINT')
            z = np.asarray(verts, dtype=float)[:, 2]
            rng = float(z.max() - z.min())
            attr.data.foreach_set('value',
                                  ((z - z.min()) / rng if rng > 1e-12
                                   else z * 0.0).tolist())
        except (RuntimeError, AttributeError):
            pass
        me.update()

    class RELIEF_OT_panel_new(bpy.types.Operator):
        """Create a new layered relief panel"""
        bl_idname = "relief.panel_new"
        bl_label = "New Layered Panel"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            me = bpy.data.meshes.new("Relief Panel")
            obj = bpy.data.objects.new("Relief Panel", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            props = obj.relief_panel
            props.is_panel = True
            base = props.layers.add()
            base.kind = 'WAVE_TRAIN'
            base.amplitude = 1.0
            base.orient = 'CURL'
            base.steepness = 0.5
            props.active = 0
            bpy.ops.relief.rebuild()
            return {'FINISHED'}

    class RELIEF_OT_layer_add(bpy.types.Operator):
        """Add a layer to the stack"""
        bl_idname = "relief.layer_add"
        bl_label = "Add Layer"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            props = context.active_object.relief_panel
            lay = props.layers.add()
            lay.amplitude = 0.4
            lay.seed = len(props.layers)
            props.active = len(props.layers) - 1
            bpy.ops.relief.rebuild()
            return {'FINISHED'}

    class RELIEF_OT_layer_remove(bpy.types.Operator):
        """Remove the active layer"""
        bl_idname = "relief.layer_remove"
        bl_label = "Remove Layer"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            props = context.active_object.relief_panel
            if not len(props.layers):
                return {'CANCELLED'}
            props.layers.remove(props.active)
            props.active = max(0, min(props.active, len(props.layers) - 1))
            bpy.ops.relief.rebuild()
            return {'FINISHED'}

    class RELIEF_OT_layer_move(bpy.types.Operator):
        """Move the active layer up or down the stack"""
        bl_idname = "relief.layer_move"
        bl_label = "Move Layer"
        bl_options = {'REGISTER', 'UNDO'}

        direction: EnumProperty(
            items=[('UP', "Up", ""), ('DOWN', "Down", "")], default='UP')

        def execute(self, context):
            props = context.active_object.relief_panel
            i = props.active
            j = i - 1 if self.direction == 'UP' else i + 1
            if j < 0 or j >= len(props.layers):
                return {'CANCELLED'}
            props.layers.move(i, j)
            props.active = j
            bpy.ops.relief.rebuild()
            return {'FINISHED'}

    class RELIEF_OT_rebuild(bpy.types.Operator):
        """Rebuild the panel from its layer stack"""
        bl_idname = "relief.rebuild"
        bl_label = "Rebuild Panel"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            obj = context.active_object
            if obj is None or not obj.relief_panel.is_panel:
                self.report({'ERROR'}, "active object is not a relief panel")
                return {'CANCELLED'}
            props = obj.relief_panel
            params = _panel_params(props)
            if not params['layers']:
                self.report({'WARNING'}, "no enabled layers; panel is flat")
            try:
                verts, faces, info = build_relief(**params)
            except (ValueError, MemoryError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            _write_mesh(obj, verts, faces, props.smooth)
            self.report({'INFO'}, "%d layers, %dx%d samples, V=%d F=%d"
                        % (len(params['layers']), info['nx'], info['ny'],
                           len(obj.data.vertices), len(obj.data.polygons)))
            return {'FINISHED'}

    class VIEW3D_PT_relief_panel(bpy.types.Panel):
        bl_label = "Relief Panel"
        bl_idname = "VIEW3D_PT_relief_panel"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Relief Panel"

        def draw(self, context):
            lay = self.layout
            obj = context.active_object
            if obj is None or not getattr(obj, 'relief_panel', None) \
                    or not obj.relief_panel.is_panel:
                lay.operator("relief.panel_new", icon='ADD')
                col = lay.column(align=True)
                col.label(text="or select a relief panel,")
                col.label(text="or use Add > Mesh > Relief Panel")
                return
            props = obj.relief_panel
            lay.operator("relief.rebuild", icon='FILE_REFRESH')

            box = lay.box()
            box.label(text="Panel")
            col = box.column(align=True)
            col.prop(props, 'shape')
            col.prop(props, 'width')
            col.prop(props, 'aspect')
            col.prop(props, 'resolution')
            col.prop(props, 'border')
            col.prop(props, 'tiling')

            box = lay.box()
            box.label(text="Layers")
            row = box.row()
            row.template_list("RELIEF_UL_layers", "", props, "layers",
                              props, "active", rows=4)
            side = row.column(align=True)
            side.operator("relief.layer_add", icon='ADD', text="")
            side.operator("relief.layer_remove", icon='REMOVE', text="")
            side.separator()
            side.operator("relief.layer_move", icon='TRIA_UP',
                          text="").direction = 'UP'
            side.operator("relief.layer_move", icon='TRIA_DOWN',
                          text="").direction = 'DOWN'

            if 0 <= props.active < len(props.layers):
                lay_ = props.layers[props.active]
                sub = box.box()
                sub.prop(lay_, 'kind')
                sub.prop(lay_, 'amplitude')
                if props.active > 0:
                    sub.prop(lay_, 'blend')
                if lay_.kind in ('WAVE', 'WAVE_TRAIN'):
                    sub.prop(lay_, 'wavelength')
                    sub.prop(lay_, 'angle')
                    sub.prop(lay_, 'steepness')
                    if lay_.kind == 'WAVE_TRAIN':
                        sub.prop(lay_, 'count')
                elif lay_.kind == 'RIPPLE':
                    sub.prop(lay_, 'wavelength')
                    sub.prop(lay_, 'sources')
                elif lay_.kind == 'FBM':
                    sub.prop(lay_, 'hurst')
                elif lay_.kind == 'CHLADNI':
                    sub.prop(lay_, 'mode_index')
                elif lay_.kind in ('DRUMHEAD', 'MEMBRANE', 'HERMITE'):
                    sub.prop(lay_, 'mode_m')
                    sub.prop(lay_, 'mode_n')
                elif lay_.kind == 'ZERNIKE':
                    sub.prop(lay_, 'zern_n')
                    sub.prop(lay_, 'zern_m')
                sub.prop(lay_, 'orient')
                sub.prop(lay_, 'seed')
                sub.prop(lay_, 'curve')

                place = sub.box()
                place.label(text="Place")
                r = place.row(align=True)
                r.prop(lay_, 'offset_x')
                r.prop(lay_, 'offset_y')
                r = place.row(align=True)
                r.prop(lay_, 'scale_x')
                r.prop(lay_, 'scale_y')
                place.prop(lay_, 'rotation')

                msk = sub.box()
                msk.label(text="Mask")
                msk.prop(lay_, 'mask', text="")
                if lay_.mask == 'RADIAL':
                    msk.prop(lay_, 'mask_width')
                elif lay_.mask == 'LINEAR':
                    msk.prop(lay_, 'mask_width')
                    msk.prop(lay_, 'mask_angle')
                elif lay_.mask == 'LAYER':
                    msk.prop(lay_, 'mask_layer')
                    if lay_.mask_layer >= props.active:
                        msk.label(text="Must be an earlier layer",
                                  icon='ERROR')

            box = lay.box()
            box.label(text="Warp & Output")
            col = box.column(align=True)
            col.prop(props, 'warp')
            if props.warp > 0.0:
                col.prop(props, 'warp_iters')
            col.prop(props, 'seed')
            col.separator()
            col.prop(props, 'depth')
            col.prop(props, 'form')
            if props.form == 'SLAB':
                col.prop(props, 'base_thickness')
            col.prop(props, 'fit')
            col.prop(props, 'scale')
            col.prop(props, 'smooth')

    _FIELD_ITEMS_MAP = [(k, lbl) for k, lbl, _d in _FIELD_ITEMS]

    _STACK_CLASSES = (ReliefLayer, ReliefPanelProps, RELIEF_UL_layers,
                      RELIEF_OT_panel_new, RELIEF_OT_layer_add,
                      RELIEF_OT_layer_remove, RELIEF_OT_layer_move,
                      RELIEF_OT_rebuild, VIEW3D_PT_relief_panel)

    def _menu_func(self, context):
        self.layout.operator("mesh.relief_panel_add", icon='MOD_DISPLACE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_relief_panel_add)
        for c in _STACK_CLASSES:
            bpy.utils.register_class(c)
        bpy.types.Object.relief_panel = bpy.props.PointerProperty(
            type=ReliefPanelProps)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        del bpy.types.Object.relief_panel
        for c in reversed(_STACK_CLASSES):
            bpy.utils.unregister_class(c)
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
