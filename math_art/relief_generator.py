
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
    from .relief import fields as _fields, grid as _grid
    from .relief import reaction as _reaction, symmetry as _symmetry
except ImportError:                       # flat import outside the package
    from relief import (FIELDS, FITS, FORMS, ORIENTATIONS, PRESETS, SHAPES,
                        build_relief, ordered_fields)
    from relief.kernels import KERNELS
    from relief import transfer as _transfer
    from relief import fields as _fields, grid as _grid
    from relief import reaction as _reaction, symmetry as _symmetry


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
        ('CUSTOM', "Custom", "Leave the controls below alone"),
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
        ('SCATTER', "Scatter",
         "Blue-noise points, each raised by a smooth kernel"),
        ('ELLIPTIC_TORUS', "Elliptic Torus",
         "Weierstrass P on a period lattice -- doubly periodic, so it tiles "
         "exactly by definition rather than by snapping"),
        ('TRUCHET', "Truchet Lanes",
         "Randomly oriented arc tiles whose arcs always join tangentially"),
        ('SEIGAIHA', "Seigaiha",
         "The Japanese overlapping-wave motif on a staggered grid"),
        ('TRIANGLE_DRUM', "Triangle Drum",
         "An equilateral membrane mode -- Lame's closed-form spectrum"),
        ('ISOSPECTRAL_A', "Isospectral A",
         "Half of Kac's counterexample; build B at the same mode"),
        ('ISOSPECTRAL_B', "Isospectral B",
         "A different outline carrying the same spectrum"),
        ('TURING_SKIN', "Turing Skin",
         "Reaction-diffusion grown on a torus -- spots, worms, mazes, coral"),
        ('CELLULAR', "Cellular",
         "Worley cell walls: setts, dried mud, reptile skin"),
        ('WALLPAPER_RELIEF', "Wallpaper",
         "A function invariant under a plane symmetry group"),
        ('QUASICRYSTAL', "Quasicrystal",
         "Ten-fold order that never repeats; approximant when tiled"),
        ('OCEAN', "Ocean",
         "Wind-driven sea by Phillips-spectrum synthesis, with choppiness"),
        ('WEAVE', "Gabor Weave",
         "Band-limited noise: a fibrous weave at a chosen pitch"),
        ('SCREEN', "Pierced Screen",
         "Quasicrystal ground opened right through the panel"),
    ]

    _FIELD_ITEMS = list(ordered_fields())

    # Every property any preset touches.  Built from the presets themselves,
    # so a key added to one of them is reset by all the others automatically
    # rather than leaking the next time someone switches.
    _PRESET_KEYS = sorted({k for v in PRESETS.values() for k in v})

    # Built from the engine's own tuples rather than retyped, so a family
    # added there cannot go missing here.
    _CELL_ITEMS = [
        ('CRACK', "Cell Walls", "F2 - F1: zero exactly on the Voronoi "
                                "boundary, so it draws the walls"),
        ('F1', "Domes", "Distance to the nearest feature point"),
        ('F2', "Second Nearest", "Distance to the second nearest"),
        ('F2F1_SUM', "Mounds", "F1 + F2: broader, softer cells"),
        ('CELL_ID', "Flat Cells", "Each Voronoi region a plateau"),
    ]
    _REGIME_ITEMS = [(k, k.title(), "feed %.3f, kill %.3f"
                      % _reaction.REGIMES[k])
                     for k in _reaction.REGIME_ORDER]
    _GROUP_ITEMS = [(g, g, "Plane symmetry group %s" % g)
                    for g in _symmetry.GROUP_ORDER]

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

            **Anything a preset can set is reset first.**  A preset is a
            partial dict, so simply writing its keys leaves every setting it
            does not mention at whatever the last preset put there.  Pierced
            Screen switches piercing on; choosing Dunes afterwards left the
            panel full of holes, because Dunes has no opinion about piercing
            and so said nothing.  Silence has to mean "the default", not "keep
            whatever was there", or presets stop being independent of the
            order they are visited in.
            """
            if self.preset == 'CUSTOM':
                return
            values = PRESETS.get(self.preset, {})
            for key in _PRESET_KEYS:
                if key in values or not hasattr(self, key):
                    continue
                prop = self.bl_rna.properties.get(key)
                if prop is None:
                    continue
                try:
                    setattr(self, key, prop.default)
                except (TypeError, ValueError):
                    pass
            for key, value in values.items():
                if not hasattr(self, key):
                    continue
                try:
                    setattr(self, key, value)
                except (TypeError, ValueError):
                    pass          # a preset key the property cannot hold

        preset: EnumProperty(
            name="Load Preset", items=_PRESET_ITEMS, default='DRAPERY',
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
        tile_x: IntProperty(
            name="Tile Across", default=1, min=1, max=16,
            description="Copies of the panel to lay out. The run is one mesh "
                        "with no joint in it, which is what the seamless "
                        "basis was for, and each copy keeps the size a "
                        "single panel would have")
        tile_y: IntProperty(name="Tile Down", default=1, min=1, max=16)
        antialias: IntProperty(
            name="Prefilter", default=1, min=1, max=4,
            description="Average this many sub-samples per axis. Point "
                        "sampling shows a pattern only what lands exactly on "
                        "the grid, so a feature finer than the sample pitch "
                        "snaps to the nearest sample and staircases. Costs "
                        "its square in evaluations")

        method: EnumProperty(
            name="Fractal",
            items=[('FBM', "Fractional Brownian",
                    "Hurst exponent is the knob"),
                   ('WEIERSTRASS', "Weierstrass-Mandelbrot",
                    "Fractal dimension is the knob")],
            default='FBM')
        hurst: FloatProperty(name="Hurst", default=0.7, min=0.05, max=0.99)
        dim: FloatProperty(name="Fractal Dimension", default=2.3, min=2.01,
                           max=2.95)
        octaves: IntProperty(name="Octaves", default=8, min=1, max=16)
        anisotropy: FloatProperty(
            name="Wind Anisotropy", default=0.0, min=0.0, max=1.0,
            description="Concentrate the fractal spectrum around one "
                        "direction, so crests run transverse to it as a dune "
                        "field's do. 0 is isotropic")
        wind: FloatProperty(
            name="Wind Direction", default=0.0, min=-6.2832, max=6.2832,
            unit='ROTATION')

        # -- solved membrane drums ------------------------------------
        drum_shape: EnumProperty(
            name="Drum",
            items=[('TRIANGLE', "Equilateral Triangle",
                    "Lame solved this one; its spectrum is proportional to "
                    "the Loeschian numbers"),
                   ('ELLIPSE', "Ellipse (Mathieu)",
                    "The elliptical membrane, whose modes are Mathieu "
                    "functions"),
                   ('L_SHAPE', "L-Shape",
                    "The reentrant corner that makes this hard"),
                   ('CIRCLE', "Circle", "Bessel modes, solved numerically"),
                   ('ISO_ONE', "Isospectral A",
                    "Half of the Gordon-Webb-Wolpert pair"),
                   ('ISO_TWO', "Isospectral B",
                    "The other half: a different outline, the same "
                    "spectrum")],
            default='TRIANGLE')
        drum_ratio: FloatProperty(
            name="Ellipse Ratio", default=0.6, min=0.1, max=1.0)
        drum_res: IntProperty(
            name="Solve Grid", default=48, min=20, max=96,
            description="Grid the eigenproblem is solved on. The cost is "
                        "cubic in the sample count, so this is the "
                        "expensive number, not the panel resolution")

        # -- Phase 5 families -----------------------------------------
        cell_mode: EnumProperty(name="Cells", items=_CELL_ITEMS,
                                default='CRACK')
        jitter: FloatProperty(
            name="Jitter", default=1.0, min=0.0, max=1.0,
            description="0 puts the feature points on a regular lattice, "
                        "giving identical cells")
        cell_sharp: FloatProperty(name="Wall Profile", default=1.0, min=0.2,
                                  max=4.0)
        regime: EnumProperty(
            name="Regime", items=_REGIME_ITEMS, default='MAZE',
            description="Where in the feed/kill plane the chemistry sits. "
                        "Outside a narrow diagonal band every seed decays "
                        "and the panel comes out blank")
        rd_steps: IntProperty(
            name="Steps", default=6000, min=100, max=40000,
            description="Reaction-diffusion is grown, not evaluated; this is "
                        "how long it runs")
        rd_scale: FloatProperty(
            name="Blobs Across", default=12.0, min=3.0, max=50.0,
            description="How many pattern features span the panel. Sets the "
                        "simulation lattice; the panel resolution only "
                        "decides how finely the finished skin is sampled")
        rd_init: EnumProperty(
            name="Seeding",
            items=[('SCATTER', "Scattered",
                    "Seed patches across the whole panel"),
                   ('CENTRE', "Central (Pearson)",
                    "One central patch, and watch the front spread")],
            default='SCATTER')
        group: EnumProperty(name="Group", items=_GROUP_ITEMS, default='P4M')
        waves: IntProperty(name="Waves", default=6, min=1, max=24)
        sym_cells: FloatProperty(name="Cells Across", default=2.0, min=0.5,
                                 max=12.0)
        freq_max: IntProperty(name="Detail", default=3, min=1, max=8)
        fold: IntProperty(
            name="Wave Directions", default=5, min=2, max=17,
            description="Directions spread over a half-turn, so five give a "
                        "ten-fold pattern. Five-fold order is impossible for "
                        "any lattice, which is why it cannot repeat")
        qc_cells: FloatProperty(name="Scale", default=6.0, min=1.0, max=30.0)
        qc_sharp: FloatProperty(name="Terrace", default=0.0, min=0.0,
                                max=6.0)
        patch: FloatProperty(
            name="Patch", default=100.0, min=5.0, max=1000.0, unit='LENGTH',
            description="Physical size of the sea being simulated. With the "
                        "wind speed this decides how many waves cross it")
        wind_speed: FloatProperty(name="Wind", default=8.0, min=1.0,
                                  max=30.0)
        choppy: FloatProperty(
            name="Choppiness", default=0.0, min=0.0, max=3.0,
            description="Measured in cusp limits: 1 puts the steepest crest "
                        "exactly at the Stokes cusp, above that the sharpest "
                        "crests break")
        sea_sim: IntProperty(name="Sea Grid", default=256, min=64, max=512)
        gabor_freq: FloatProperty(name="Pitch", default=8.0, min=1.0,
                                  max=60.0)
        gabor_band: FloatProperty(
            name="Bandwidth", default=0.3, min=0.05, max=1.0,
            description="Envelope width as a fraction of the carrier. Well "
                        "under 1, or the band is as wide as the frequency it "
                        "is centred on")

        # -- pierced output -------------------------------------------
        pierce: BoolProperty(
            name="Pierce", default=False,
            description="Open the panel right through where the field is "
                        "low, turning a relief into a screen")
        pierce_level: FloatProperty(name="Open Below", default=0.35, min=0.0,
                                    max=1.0)
        pierce_invert: BoolProperty(name="Invert", default=False)
        pierce_min: IntProperty(
            name="Smallest Hole", default=8, min=1, max=200,
            description="Holes smaller than this many samples are filled in; "
                        "a threshold near the mean otherwise opens pinholes "
                        "nothing can manufacture")

        # -- elliptic functions ---------------------------------------
        ell_kind: EnumProperty(
            name="Function",
            items=[('WP', "Weierstrass P", "Doubly periodic: tiles exactly"),
                   ('WP_PRIME', "Weierstrass P'",
                    "Its derivative; also doubly periodic"),
                   ('ZETA', "Weierstrass zeta",
                    "Quasi-periodic: gains a constant per period, so it "
                    "does NOT tile"),
                   ('THETA', "Jacobi theta",
                    "Quasi-periodic: gains a factor per period, so it "
                    "does NOT tile")],
            default='WP')
        ell_part: EnumProperty(
            name="Height From",
            items=[('SPHERE', "Riemann Sphere",
                    "Stereographic height: smooth THROUGH the poles, so a "
                    "pole becomes a rounded summit"),
                   ('RE', "Real Part", "Unbounded at the poles"),
                   ('IM', "Imaginary Part", "Unbounded at the poles"),
                   ('ABS', "Modulus", "Unbounded at the poles")],
            default='SPHERE')
        tau_re: FloatProperty(name="Lattice Skew", default=0.0, min=-2.0,
                              max=2.0)
        tau_im: FloatProperty(name="Lattice Ratio", default=1.0, min=0.05,
                              max=4.0)
        ell_cells: FloatProperty(name="Periods", default=1.0, min=0.25,
                                 max=8.0)

        # -- tile patterns --------------------------------------------
        tile_cells: IntProperty(name="Cells", default=6, min=1, max=64)
        lane: FloatProperty(
            name="Lane Width", default=0.25, min=0.02, max=1.0,
            description="Width of the Truchet arc lane, as a fraction of "
                        "the cell")
        straight: FloatProperty(
            name="Straight Tiles", default=0.0, min=0.0, max=1.0,
            description="Fraction of cells drawn with crossing straight "
                        "lanes instead of arcs. A second tile on the same "
                        "edge convention, so the lanes still join smoothly")
        rings: IntProperty(name="Arcs", default=3, min=1, max=12)
        crown: FloatProperty(
            name="Crown", default=0.55, min=0.1, max=1.0,
            description="Fraction of each circle left visible by the row "
                        "in front")
        rim: FloatProperty(
            name="Rim", default=0.08, min=0.0, max=0.5,
            description="Width of the edge where a scale laps over the one "
                        "behind, as a fraction of its radius. Zero is a "
                        "vertical wall, which no grid can place cleanly")

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
                cell_mode=self.cell_mode, jitter=self.jitter,
                cell_sharp=self.cell_sharp,
                regime=self.regime, rd_steps=self.rd_steps,
                rd_scale=self.rd_scale, rd_init=self.rd_init,
                group=self.group, waves=self.waves,
                sym_cells=self.sym_cells, freq_max=self.freq_max,
                fold=self.fold, qc_cells=self.qc_cells,
                qc_sharp=self.qc_sharp,
                patch=self.patch, wind_speed=self.wind_speed,
                choppy=self.choppy, sea_sim=self.sea_sim,
                gabor_freq=self.gabor_freq, gabor_band=self.gabor_band,
                tile_x=self.tile_x, tile_y=self.tile_y,
                pierce=self.pierce, pierce_level=self.pierce_level,
                pierce_invert=self.pierce_invert,
                pierce_min=self.pierce_min,
                seed=self.seed, antialias=self.antialias,
                method=self.method, hurst=self.hurst,
                dim=self.dim, octaves=self.octaves,
                anisotropy=self.anisotropy, wind=self.wind,
                drum_shape=self.drum_shape, drum_ratio=self.drum_ratio,
                drum_res=self.drum_res,
                ell_kind=self.ell_kind, ell_part=self.ell_part,
                tau_re=self.tau_re, tau_im=self.tau_im,
                ell_cells=self.ell_cells,
                tile_cells=self.tile_cells, lane=self.lane,
                straight=self.straight, rings=self.rings,
                crown=self.crown, rim=self.rim,
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
                msg += ("  seam x%.2f/x%.2f"
                        % (info.get('seam_step', 0.0),
                           info.get('seam_curvature', 0.0)))
                if not info.get('seam_ok', True):
                    self.report(
                        {'WARNING'},
                        "the panel does NOT tile: %s (joint x%.1f step, "
                        "x%.1f curvature; 1.0 would be invisible)"
                        % (info.get('seam_reason') or "measured at the joint",
                           info.get('seam_step', 0.0),
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
            if self.tiling != 'NONE':
                # One per row.  Side by side, Blender splits the width
                # between them and truncates the second label to "Til...",
                # which reads as a bug rather than as a layout.
                box.prop(self, 'tile_x')
                box.prop(self, 'tile_y')
                if self.tile_x > 1 or self.tile_y > 1:
                    box.label(text="%d copies as one mesh, %.1f x %.1f m"
                                   % (self.tile_x * self.tile_y,
                                      self.width * self.tile_x,
                                      self.width * self.aspect * self.tile_y),
                              icon='INFO')
            if self.tiling != 'NONE' and self.warp > 0.0:
                box.label(text="Warp is suppressed while tiling",
                          icon='INFO')

            box = lay.box()
            box.label(text="Pattern")
            col = box.column()
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
                col.prop(self, 'anisotropy')
                if self.anisotropy > 0.0:
                    col.prop(self, 'wind')
                    if self.tiling != 'NONE':
                        col.label(text="Tiling overrides the wind",
                                  icon='INFO')
            elif self.field == 'ELLIPTIC':
                col.prop(self, 'ell_kind')
                col.prop(self, 'ell_part')
                col.prop(self, 'ell_cells')
                col.prop(self, 'tau_re')
                col.prop(self, 'tau_im')
                if self.ell_kind not in ('WP', 'WP_PRIME'):
                    col.label(text="Only P and P' are doubly periodic",
                              icon='INFO')
            elif self.field == 'TRUCHET':
                col.prop(self, 'tile_cells')
                col.prop(self, 'lane')
                col.prop(self, 'straight')
            elif self.field == 'SEIGAIHA':
                col.prop(self, 'tile_cells')
                col.prop(self, 'rings')
                col.prop(self, 'crown')
                col.prop(self, 'rim')
            elif self.field == 'WORLEY':
                col.prop(self, 'cell_mode')
                col.prop(self, 'points_n')
                col.prop(self, 'jitter')
                col.prop(self, 'cell_sharp')
            elif self.field == 'TURING':
                col.prop(self, 'regime')
                col.prop(self, 'rd_scale')
                col.prop(self, 'rd_steps')
                col.prop(self, 'rd_init')
                col.label(text="Grown, not evaluated: cost is the step count",
                          icon='INFO')
            elif self.field == 'WALLPAPER':
                col.prop(self, 'group')
                col.prop(self, 'sym_cells')
                col.prop(self, 'waves')
                col.prop(self, 'freq_max')
                if self.group in ('P3', 'P3M1', 'P31M', 'P6', 'P6M') \
                        and abs(self.aspect - 1.7320508) > 0.02:
                    col.label(text="Hex groups tile at aspect 1.732",
                              icon='INFO')
            elif self.field == 'QUASI':
                col.prop(self, 'fold')
                col.prop(self, 'qc_cells')
                col.prop(self, 'qc_sharp')
                if self.tiling != 'NONE':
                    col.label(text="Tiling uses the periodic approximant",
                              icon='INFO')
            elif self.field == 'OCEAN':
                col.prop(self, 'wind_speed')
                col.prop(self, 'patch')
                col.prop(self, 'choppy')
                col.prop(self, 'sea_sim')
                col.prop(self, 'angle', text="Wind Direction")
            elif self.field == 'GABOR':
                col.prop(self, 'gabor_freq')
                col.prop(self, 'gabor_band')
                col.prop(self, 'angle')
                col.prop(self, 'spread')
                col.prop(self, 'points_n')
            elif self.field == 'DRUM':
                col.prop(self, 'drum_shape')
                if self.drum_shape == 'ELLIPSE':
                    col.prop(self, 'drum_ratio')
                col.prop(self, 'mode_index')
                col.prop(self, 'drum_res')
                if self.drum_shape in ('ISO_ONE', 'ISO_TWO'):
                    col.label(text="Build both at one mode to compare",
                              icon='INFO')
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

    # Set while a rebuild is running.  An autobuild writes `built_sig` and
    # `built_quality` back onto the property group; those two carry no update
    # callback of their own, but the guard makes the no-reentry rule explicit
    # rather than dependent on that staying true.
    _BUILDING = [False]

    def _rebuild(obj, preview=True):
        """Rebuild `obj`'s geometry from its layer stack.  Returns `info`.

        The single place a relief panel's mesh is written, shared by the
        Rebuild operators and by autobuild -- an autobuild cannot go through
        `bpy.ops`, which would act on the active object rather than on the
        one whose property actually changed.
        """
        props = obj.relief_panel
        params = _panel_params(props, preview=preview)
        verts, faces, info = build_relief(**params)
        _write_mesh(obj, verts, faces, props.smooth)
        props.built_sig = _params_signature(props)
        props.built_quality = 'PREVIEW' if preview else 'FULL'
        return info

    def _auto(self, context):
        """Property update hook: rebuild at preview resolution if asked to.

        Attached to every setting that changes the geometry.  It builds at
        PREVIEW resolution unconditionally -- autobuild is only affordable
        because of that, and an exact plate solve on a 1024 grid is not
        drag-speed.  The full-resolution build stays an explicit action.
        """
        if _BUILDING[0]:
            return
        obj = getattr(self, 'id_data', None)
        props = getattr(obj, 'relief_panel', None)
        if props is None or not props.is_panel or not props.autobuild:
            return
        _BUILDING[0] = True
        try:
            _rebuild(obj, preview=True)
        except (ValueError, MemoryError, RuntimeError):
            # A half-typed number is a normal intermediate state while
            # dragging; failing loudly on every one of them would make the
            # option unusable.  The stale marker still tells the truth.
            pass
        finally:
            _BUILDING[0] = False

    class ReliefLayer(bpy.types.PropertyGroup):
        """One entry in a panel's pattern stack."""
        kind: EnumProperty(name="Pattern", items=_FIELD_ITEMS,
                           default='WAVE_TRAIN', update=_auto)
        amplitude: FloatProperty(name="Amount", default=1.0, min=-4.0,
                                 max=4.0, update=_auto)
        blend: EnumProperty(name="Blend", items=_BLEND_ITEMS, default='ADD',
                            update=_auto)
        enabled: BoolProperty(name="Enabled", default=True, update=_auto)

        wavelength: FloatProperty(name="Wavelength", default=0.5, min=0.005,
                                  max=20.0, unit='LENGTH', update=_auto)
        angle: FloatProperty(name="Angle", default=0.0, min=-6.2832,
                             max=6.2832, unit='ROTATION', update=_auto)
        steepness: FloatProperty(name="Steepness", default=0.0, min=0.0,
                                 max=1.0, update=_auto)
        count: IntProperty(name="Waves", default=3, min=1, max=24,
                           update=_auto)
        sources: IntProperty(name="Sources", default=3, min=1, max=32,
                             update=_auto)
        seed: IntProperty(name="Seed", default=1, min=0, max=100000,
                          update=_auto)
        mode_index: IntProperty(name="Mode", default=6, min=1, max=40,
                                update=_auto)
        mode_m: IntProperty(name="m", default=2, min=0, max=24, update=_auto)
        mode_n: IntProperty(name="n", default=3, min=0, max=24, update=_auto)
        zern_n: IntProperty(name="Zernike n", default=4, min=0, max=20,
                            update=_auto)
        zern_m: IntProperty(name="Zernike m", default=2, min=-20, max=20,
                            update=_auto)
        hurst: FloatProperty(name="Hurst", default=0.7, min=0.05, max=0.99,
                             update=_auto)
        method: EnumProperty(
            name="Fractal",
            items=[('FBM', "Fractional Brownian",
                    "Hurst exponent is the knob"),
                   ('WEIERSTRASS', "Weierstrass-Mandelbrot",
                    "Fractal dimension is the knob")],
            default='FBM', update=_auto)
        dim: FloatProperty(name="Fractal Dimension", default=2.3, min=2.01,
                           max=2.95, update=_auto)
        octaves: IntProperty(name="Octaves", default=8, min=1, max=16,
                             update=_auto)
        anisotropy: FloatProperty(name="Wind Anisotropy", default=0.0,
                                  min=0.0, max=1.0, update=_auto)
        wind: FloatProperty(name="Wind Direction", default=0.0, min=-6.2832,
                            max=6.2832, unit='ROTATION', update=_auto)

        cell_mode: EnumProperty(name="Cells", items=_CELL_ITEMS,
                                default='CRACK', update=_auto)
        jitter: FloatProperty(name="Jitter", default=1.0, min=0.0, max=1.0,
                              update=_auto)
        cell_sharp: FloatProperty(name="Wall Profile", default=1.0, min=0.2,
                                  max=4.0, update=_auto)
        regime: EnumProperty(name="Regime", items=_REGIME_ITEMS,
                             default='MAZE', update=_auto)
        rd_steps: IntProperty(name="Steps", default=6000, min=100,
                              max=40000, update=_auto)
        rd_scale: FloatProperty(name="Blobs Across", default=12.0, min=3.0,
                                max=50.0, update=_auto)
        rd_init: EnumProperty(
            name="Seeding",
            items=[('SCATTER', "Scattered", ""), ('CENTRE', "Central", "")],
            default='SCATTER', update=_auto)
        group: EnumProperty(name="Group", items=_GROUP_ITEMS, default='P4M',
                            update=_auto)
        waves: IntProperty(name="Waves", default=6, min=1, max=24,
                           update=_auto)
        sym_cells: FloatProperty(name="Cells Across", default=2.0, min=0.5,
                                 max=12.0, update=_auto)
        freq_max: IntProperty(name="Detail", default=3, min=1, max=8,
                              update=_auto)
        fold: IntProperty(name="Wave Directions", default=5, min=2, max=17,
                          update=_auto)
        qc_cells: FloatProperty(name="Scale", default=6.0, min=1.0, max=30.0,
                                update=_auto)
        qc_sharp: FloatProperty(name="Terrace", default=0.0, min=0.0,
                                max=6.0, update=_auto)
        patch: FloatProperty(name="Patch", default=100.0, min=5.0,
                             max=1000.0, unit='LENGTH', update=_auto)
        wind_speed: FloatProperty(name="Wind", default=8.0, min=1.0,
                                  max=30.0, update=_auto)
        choppy: FloatProperty(name="Choppiness", default=0.0, min=0.0,
                              max=3.0, update=_auto)
        sea_sim: IntProperty(name="Sea Grid", default=256, min=64, max=512,
                             update=_auto)
        gabor_freq: FloatProperty(name="Pitch", default=8.0, min=1.0,
                                  max=60.0, update=_auto)
        gabor_band: FloatProperty(name="Bandwidth", default=0.3, min=0.05,
                                  max=1.0, update=_auto)

        ell_kind: EnumProperty(
            name="Function",
            items=[('WP', "Weierstrass P", "Doubly periodic: tiles exactly"),
                   ('WP_PRIME', "Weierstrass P'", "Also doubly periodic"),
                   ('ZETA', "Weierstrass zeta",
                    "Quasi-periodic: does NOT tile"),
                   ('THETA', "Jacobi theta", "Quasi-periodic: does NOT tile")],
            default='WP', update=_auto)
        ell_part: EnumProperty(
            name="Height From",
            items=[('SPHERE', "Riemann Sphere", "Smooth through the poles"),
                   ('RE', "Real Part", ""), ('IM', "Imaginary Part", ""),
                   ('ABS', "Modulus", "")],
            default='SPHERE', update=_auto)
        tau_re: FloatProperty(name="Lattice Skew", default=0.0, min=-2.0,
                              max=2.0, update=_auto)
        tau_im: FloatProperty(name="Lattice Ratio", default=1.0, min=0.05,
                              max=4.0, update=_auto)
        ell_cells: FloatProperty(name="Periods", default=1.0, min=0.25,
                                 max=8.0, update=_auto)

        tile_cells: IntProperty(name="Cells", default=6, min=1, max=64,
                                update=_auto)
        lane: FloatProperty(name="Lane Width", default=0.25, min=0.02,
                            max=1.0, update=_auto)
        straight: FloatProperty(name="Straight Tiles", default=0.0, min=0.0,
                                max=1.0, update=_auto)
        drum_shape: EnumProperty(
            name="Drum",
            items=[('TRIANGLE', "Equilateral Triangle", ""),
                   ('ELLIPSE', "Ellipse (Mathieu)", ""),
                   ('L_SHAPE', "L-Shape", ""),
                   ('CIRCLE', "Circle", ""),
                   ('ISO_ONE', "Isospectral A", ""),
                   ('ISO_TWO', "Isospectral B", "")],
            default='TRIANGLE', update=_auto)
        drum_ratio: FloatProperty(name="Ellipse Ratio", default=0.6,
                                  min=0.1, max=1.0, update=_auto)
        drum_res: IntProperty(name="Solve Grid", default=48, min=20,
                              max=96, update=_auto)
        rings: IntProperty(name="Arcs", default=3, min=1, max=12,
                           update=_auto)
        crown: FloatProperty(name="Crown", default=0.55, min=0.1, max=1.0,
                             update=_auto)
        rim: FloatProperty(name="Rim", default=0.08, min=0.0, max=0.5,
                           update=_auto)

        orient: EnumProperty(name="Orientation", items=_ORIENT_ITEMS,
                             default='CONSTANT', update=_auto)
        orient_freq: FloatProperty(name="Field Scale", default=0.5, min=0.05,
                                   max=8.0, update=_auto)

        offset_x: FloatProperty(name="Offset X", default=0.0, min=-10.0,
                                max=10.0, unit='LENGTH', update=_auto)
        offset_y: FloatProperty(name="Offset Y", default=0.0, min=-10.0,
                                max=10.0, unit='LENGTH', update=_auto)
        rotation: FloatProperty(name="Rotation", default=0.0, min=-6.2832,
                                max=6.2832, unit='ROTATION', update=_auto)
        scale_x: FloatProperty(name="Scale X", default=1.0, min=0.01,
                               max=100.0, update=_auto)
        scale_y: FloatProperty(name="Scale Y", default=1.0, min=0.01,
                               max=100.0, update=_auto)

        mask: EnumProperty(name="Mask", items=_MASK_ITEMS, default='NONE',
                           update=_auto)
        mask_width: FloatProperty(name="Mask Width", default=0.5, min=0.01,
                                  max=2.0, update=_auto)
        mask_angle: FloatProperty(name="Mask Angle", default=0.0, min=-6.2832,
                                  max=6.2832, unit='ROTATION', update=_auto)
        mask_layer: IntProperty(
            name="Mask Layer", default=0, min=0, max=63,
            description="Index of an EARLIER layer to use as the mask; a "
                        "layer can only reference ones below it", update=_auto)
        curve: EnumProperty(name="Profile", items=_CURVE_ITEMS,
                            default='NONE', update=_auto)
        curve_amount: FloatProperty(name="Amount", default=1.0, min=0.05,
                                    max=4.0, update=_auto)

    class ReliefPanelProps(bpy.types.PropertyGroup):
        """The whole panel: outline, output settings, and the layer stack."""
        shape: EnumProperty(
            name="Shape",
            items=[(s, s.replace('_', ' ').title(), "") for s in SHAPES],
            default='RECT', update=_auto)
        width: FloatProperty(name="Width", default=2.0, min=0.01, max=100.0,
                             unit='LENGTH', update=_auto)
        aspect: FloatProperty(name="Aspect", default=1.0, min=0.05, max=20.0,
                              update=_auto)
        resolution: IntProperty(name="Resolution", default=192, min=8,
                                max=1024, update=_auto)
        border: FloatProperty(name="Border", default=0.0, min=0.0, max=0.5,
                              update=_auto)
        tiling: EnumProperty(name="Tiling", items=_TILING_ITEMS,
                             default='NONE', update=_auto)
        warp: FloatProperty(name="Warp", default=0.0, min=0.0, max=2.0,
                            update=_auto)
        warp_iters: IntProperty(name="Warp Steps", default=2, min=1, max=4,
                                update=_auto)
        seed: IntProperty(name="Seed", default=1, min=0, max=100000,
                          update=_auto)
        depth: FloatProperty(name="Relief Depth", default=0.25, min=0.0,
                             max=10.0, unit='LENGTH', update=_auto)
        form: EnumProperty(
            name="Form",
            items=[('SLAB', "Slab", "Watertight panel with a flat back"),
                   ('SHEET', "Sheet", "Open surface")],
            default='SLAB', update=_auto)
        base_thickness: FloatProperty(name="Base", default=0.1, min=0.0,
                                      max=10.0, unit='LENGTH', update=_auto)
        fit: EnumProperty(
            name="Fit",
            items=[('FOOTPRINT', "Footprint", ""), ('CUBE', "2 m Cube", ""),
                   ('NONE', "None", "")],
            default='FOOTPRINT', update=_auto)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             update=_auto)
        smooth: BoolProperty(name="Smooth Shading", default=True, update=_auto)
        tile_x: IntProperty(name="Tile Across", default=1, min=1, max=16,
                            update=_auto)
        tile_y: IntProperty(name="Tile Down", default=1, min=1, max=16,
                            update=_auto)
        pierce: BoolProperty(
            name="Pierce", default=False,
            description="Open the panel right through where the field is "
                        "low, turning a relief into a screen",
            update=_auto)
        pierce_level: FloatProperty(name="Open Below", default=0.35, min=0.0,
                                    max=1.0, update=_auto)
        pierce_invert: BoolProperty(name="Invert", default=False,
                                    update=_auto)
        pierce_min: IntProperty(name="Smallest Hole", default=8, min=1,
                                max=200, update=_auto)
        autobuild: BoolProperty(
            name="Autobuild", default=True,
            description="Rebuild the panel at PREVIEW resolution whenever a "
                        "setting changes. The full-resolution build stays an "
                        "explicit action, so autobuild costs preview time, "
                        "not build time")
        antialias: IntProperty(
            name="Prefilter", default=1, min=1, max=4,
            description="Average this many sub-samples per axis, to keep "
                        "features finer than the sample pitch from "
                        "staircasing. Costs its square in evaluations",
            update=_auto)
        preview_resolution: IntProperty(
            name="Preview Resolution", default=128, min=8, max=512,
            description="Sampling used by autobuild and Rebuild. The surface "
                        "is the same one the final build makes, only sampled "
                        "more coarsely",
            update=_auto)

        layers: bpy.props.CollectionProperty(type=ReliefLayer)
        active: IntProperty(name="Active Layer", default=0, min=0)
        is_panel: BoolProperty(default=False)
        # Signature of the settings the current mesh was built from.  Layer
        # edits do not rebuild automatically -- an exact plate solve at full
        # resolution is not drag-speed -- so the panel compares this against
        # the live settings and says when the mesh is stale.  Auto-rebuilding
        # some actions (add, remove, reorder) and not others is the worst of
        # both worlds: you cannot tell which state you are looking at.
        built_sig: StringProperty(default="")
        # Which of the two resolutions produced the current mesh.  Kept apart
        # from `built_sig` on purpose: a preview is not a settings change, and
        # folding the two together would make every preview read as stale.
        built_quality: StringProperty(default="")

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
        """Fill an object's layer stack from the single-shot operator.

        Seeding assigns dozens of properties in a row.  Each assignment would
        otherwise fire the autobuild hook and rebuild the panel from a
        half-copied state, so the whole transfer happens under the guard --
        the operator has already built the mesh it is seeding from.
        """
        props = obj.relief_panel
        props.is_panel = True
        _BUILDING[0] = True
        try:
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
        finally:
            _BUILDING[0] = False

    def _panel_params(props, preview=False):
        """Turn the property group into a `build_relief` argument dict.

        `preview` swaps in the preview resolution.  Nothing else changes, so
        a preview is the same surface sampled more coarsely -- not a different
        or simplified one -- and what you tune at preview resolution is what
        you get at full.
        """
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
                cell_mode=lay.cell_mode, jitter=lay.jitter,
                cell_sharp=lay.cell_sharp,
                regime=lay.regime, rd_steps=lay.rd_steps,
                rd_scale=lay.rd_scale, rd_init=lay.rd_init,
                group=lay.group, waves=lay.waves,
                sym_cells=lay.sym_cells, freq_max=lay.freq_max,
                fold=lay.fold, qc_cells=lay.qc_cells,
                qc_sharp=lay.qc_sharp,
                patch=lay.patch, wind_speed=lay.wind_speed,
                choppy=lay.choppy, sea_sim=lay.sea_sim,
                gabor_freq=lay.gabor_freq, gabor_band=lay.gabor_band,
                hurst=lay.hurst, method=lay.method, dim=lay.dim,
                octaves=lay.octaves, anisotropy=lay.anisotropy,
                wind=lay.wind,
                ell_kind=lay.ell_kind, ell_part=lay.ell_part,
                tau_re=lay.tau_re, tau_im=lay.tau_im,
                ell_cells=lay.ell_cells,
                tile_cells=lay.tile_cells, lane=lay.lane,
                drum_shape=lay.drum_shape, drum_ratio=lay.drum_ratio,
                drum_res=lay.drum_res,
                straight=lay.straight, rings=lay.rings,
                crown=lay.crown, rim=lay.rim,
                orient=lay.orient,
                orient_freq=lay.orient_freq,
                offset_x=lay.offset_x, offset_y=lay.offset_y,
                rotation=lay.rotation, scale_x=lay.scale_x,
                scale_y=lay.scale_y,
                mask=lay.mask, mask_width=lay.mask_width,
                mask_angle=lay.mask_angle, mask_layer=lay.mask_layer,
                curve=lay.curve, curve_amount=lay.curve_amount))
        res = props.resolution
        if preview:
            # Never preview at more than the final resolution: below that the
            # preview would be the slower of the two, which is absurd.
            res = min(int(props.preview_resolution), int(props.resolution))
        return dict(
            shape=props.shape, width=props.width, aspect=props.aspect,
            resolution=res, border=props.border,
            tiling=props.tiling, antialias=props.antialias,
            tile_x=props.tile_x, tile_y=props.tile_y,
            pierce=props.pierce, pierce_level=props.pierce_level,
            pierce_invert=props.pierce_invert,
            pierce_min=props.pierce_min,
            warp=props.warp,
            warp_iters=props.warp_iters, seed=props.seed,
            depth=props.depth, form=props.form,
            base_thickness=props.base_thickness, fit=props.fit,
            scale=props.scale, layers=layers)

    def _feature_samples(kind, params, props):
        """Finest feature of one layer, in samples, at preview resolution."""
        prev = min(int(props.preview_resolution), int(props.resolution))
        nx, ny = _grid.clamp_resolution(prev, props.aspect)
        info = {'dx': props.width / max(nx - 1, 1), 'width': props.width}
        try:
            return _fields.feature_samples(kind, params, info)
        except (KeyError, TypeError, ValueError):
            return None

    def _resolution_for(props, have, want):
        """Resolution at which the finest feature would span `want` samples."""
        prev = min(int(props.preview_resolution), int(props.resolution))
        return int(max(8, min(1024, round(prev * want / max(have, 1e-6)))))

    def _params_signature(props):
        """A short string identifying the SETTINGS a mesh was built from.

        Always taken at full resolution, so it describes the design and not
        the sampling.  Which of the two resolutions actually produced the
        current mesh is recorded separately in `built_quality` -- conflating
        them would make every preview look like a settings change.
        """
        p = _panel_params(props, preview=False)
        return repr(sorted((k, repr(v)) for k, v in p.items()))

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
        # Editing the datablock in place does not by itself mark the object
        # dirty, so the viewport keeps drawing the previous geometry until
        # something else happens to trigger a refresh.
        obj.update_tag()
        me.update_tag()
        if bpy.context.view_layer is not None:
            bpy.context.view_layer.update()
        for area in getattr(bpy.context.screen, 'areas', ()) or ():
            if area.type == 'VIEW_3D':
                area.tag_redraw()

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
            _BUILDING[0] = True
            try:
                base = props.layers.add()
                base.kind = 'WAVE_TRAIN'
                base.amplitude = 1.0
                base.orient = 'CURL'
                base.steepness = 0.5
                props.active = 0
            finally:
                _BUILDING[0] = False
            # A new panel always builds, autobuild or not: an empty object
            # would be a worse answer than a coarse one.
            _rebuild(obj, preview=True)
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
            _auto(props, context)
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
            _auto(props, context)
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
            _auto(props, context)
            return {'FINISHED'}

    class RELIEF_OT_rebuild(bpy.types.Operator):
        """Rebuild the panel from its layer stack"""
        bl_idname = "relief.rebuild"
        bl_label = "Rebuild Panel"
        bl_options = {'REGISTER', 'UNDO'}

        preview: BoolProperty(
            name="Preview", default=True,
            description="Build at the preview resolution rather than the "
                        "full one")

        def execute(self, context):
            obj = context.active_object
            if obj is None or not obj.relief_panel.is_panel:
                self.report({'ERROR'}, "active object is not a relief panel")
                return {'CANCELLED'}
            props = obj.relief_panel
            if not _panel_params(props)['layers']:
                self.report({'WARNING'}, "no enabled layers; panel is flat")
            try:
                info = _rebuild(obj, preview=self.preview)
            except (ValueError, MemoryError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            self.report({'INFO'}, "%s: %dx%d samples, V=%d F=%d"
                        % ("preview" if self.preview else "full",
                           info['nx'], info['ny'],
                           len(obj.data.vertices), len(obj.data.polygons)))
            return {'FINISHED'}

    class RELIEF_OT_export_heightmap(bpy.types.Operator):
        """Write the panel's height field to an image"""
        bl_idname = "relief.export_heightmap"
        bl_label = "Export Heightmap"
        bl_options = {'REGISTER'}

        filepath: StringProperty(subtype='FILE_PATH')
        file_format: EnumProperty(
            name="Format",
            items=[('OPEN_EXR', "OpenEXR",
                    "Float: no visible banding on a shallow relief"),
                   ('PNG', "PNG",
                    "Integer: smaller, and banded where the relief is "
                    "shallow")],
            default='OPEN_EXR')
        resolution: IntProperty(name="Resolution", default=512, min=8,
                                max=4096)

        def invoke(self, context, event):
            if not self.filepath:
                self.filepath = "//relief_height.exr"
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def execute(self, context):
            obj = context.active_object
            if obj is None or not obj.relief_panel.is_panel:
                self.report({'ERROR'}, "active object is not a relief panel")
                return {'CANCELLED'}
            props = obj.relief_panel
            params = _panel_params(props)
            params['resolution'] = int(self.resolution)
            # A heightmap is the field, not the mesh: ask for the open sheet
            # so no slab underside or wall is folded into the image.
            params['form'] = 'SHEET'
            params['fit'] = 'NONE'
            try:
                verts, faces, info = build_relief(**params)
            except (ValueError, MemoryError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

            nx, ny = info['nx'], info['ny']
            z = np.asarray(verts, dtype=float)[:nx * ny, 2].reshape(ny, nx)
            lo, hi = float(z.min()), float(z.max())
            span = (hi - lo) or 1.0

            # **The map is normalised to 0..1, in both formats.**  Writing
            # metres would be better and is not available: Blender's image
            # save clamps to the display range, so a float buffer holding
            # -0.125..0.125 comes back as 0..1 with every negative flattened
            # to zero -- verified by round-tripping a -2..2 ramp, which
            # returned 0..1.  The mapping is reported instead, and it is
            # exactly what a Displace modifier wants: Strength = the reported
            # span, Midlevel = -lo / span.
            v = (z - lo) / span

            img = bpy.data.images.new("relief_height", width=nx, height=ny,
                                      float_buffer=True, alpha=False)
            # Colourspace BEFORE the pixels.  Setting it afterwards discards
            # the buffer -- the image saves as a uniform zero, and the only
            # visible symptom is a suspiciously small file.
            img.file_format = self.file_format
            img.colorspace_settings.name = 'Non-Color'
            px = np.empty((ny, nx, 4), dtype=np.float32)
            px[:, :, 0] = v
            px[:, :, 1] = v
            px[:, :, 2] = v
            px[:, :, 3] = 1.0
            img.pixels.foreach_set(px.ravel())
            # **Written with `save_render`, not `save`.**  `Image.save()`
            # ignores `file_format` on a generated image -- asking for
            # OpenEXR produced a PNG with an .exr name -- and it clamps the
            # buffer to the display range on the way out.  `save_render` goes
            # through the scene's image settings, which do honour the format,
            # and a Raw view transform keeps the numbers untouched: the
            # round trip is exact for EXR and 16-bit-quantised for PNG.
            scene = context.scene
            st = scene.render.image_settings
            keep = (st.file_format, st.color_mode, st.color_depth,
                    scene.view_settings.view_transform)
            try:
                st.file_format = self.file_format
                st.color_mode = 'BW'
                st.color_depth = '32' if self.file_format == 'OPEN_EXR'                     else '16'
                scene.view_settings.view_transform = 'Raw'
                img.save_render(filepath=bpy.path.abspath(self.filepath),
                                scene=scene)
            except (RuntimeError, TypeError) as exc:
                self.report({'ERROR'}, "could not write image: %s" % exc)
                return {'CANCELLED'}
            finally:
                # The render settings belong to the user's scene, not to this
                # operator; leaving them changed would silently alter their
                # next render.
                (st.file_format, st.color_mode, st.color_depth,
                 scene.view_settings.view_transform) = keep
                bpy.data.images.remove(img)
            self.report({'INFO'},
                        "%dx%d %s: 0..1 maps to %.4f..%.4f m "
                        "(Displace strength %.4f, midlevel %.3f)"
                        % (nx, ny, self.file_format, lo, hi, span,
                           -lo / span))
            return {'FINISHED'}

    class VIEW3D_PT_relief_panel(bpy.types.Panel):
        bl_label = "Relief Panel"
        bl_idname = "VIEW3D_PT_relief_panel"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Math Art"
        bl_order = 10

        @classmethod
        def poll(cls, context):
            # The tab edits the selected object and nothing else: objects are
            # created from Add > Mesh, and this panel appears when one of
            # them is what you have selected.
            o = context.active_object
            return (o is not None and getattr(o, 'relief_panel', None)
                    and o.relief_panel.is_panel)

        def draw(self, context):
            lay = self.layout
            obj = context.active_object
            props = obj.relief_panel
            stale = _params_signature(props) != props.built_sig
            prev_res = min(int(props.preview_resolution),
                           int(props.resolution))
            coarse = (props.built_quality == 'PREVIEW'
                      and prev_res < int(props.resolution))

            head = lay.row(align=True)
            head.prop(props, 'autobuild', toggle=True,
                      icon='AUTO' if props.autobuild else 'FILE_REFRESH')

            row = lay.row(align=True)
            row.scale_y = 1.4 if stale else 1.0
            row.alert = stale
            # Autobuild already keeps the preview current, so offering a
            # manual preview rebuild alongside it is just a button that does
            # nothing; the final build is the one that still has work to do.
            if not props.autobuild:
                row.operator("relief.rebuild", icon='FILE_REFRESH',
                             text="Rebuild *" if stale else "Rebuild"
                             ).preview = True
            fin = row.operator("relief.rebuild", icon='SHADERFX',
                               text="Build Final")
            fin.preview = False

            lay.operator("relief.export_heightmap", icon='IMAGE_DATA')

            if stale and not props.autobuild:
                lay.label(text="Settings changed since the last build",
                          icon='INFO')
            elif coarse:
                lay.label(text="Preview at %d; final is %d"
                               % (prev_res, int(props.resolution)),
                          icon='INFO')

            box = lay.box()
            box.label(text="Panel")
            col = box.column(align=True)
            col.prop(props, 'shape')
            col.prop(props, 'width')
            col.prop(props, 'aspect')
            col.prop(props, 'resolution')
            col.prop(props, 'preview_resolution')
            col.prop(props, 'antialias')
            col.prop(props, 'border')

            # How well the sampling resolves the finest thing being drawn.
            # A mesh interpolates linearly between samples, so a crest given
            # only a few of them comes out as a facet -- which is what the
            # eye reads as a jagged relief line.
            worst = None
            for lay_i in props.layers:
                if not lay_i.enabled:
                    continue
                q = _panel_params(props, preview=True)
                q.update({k: getattr(lay_i, k) for k in
                          ('tile_cells', 'rings', 'lane', 'wavelength',
                           'octaves', 'mode_m', 'mode_n', 'zern_n',
                           'mode_index', 'ell_cells')})
                f = _feature_samples(lay_i.kind, q, props)
                if f is not None:
                    worst = f if worst is None else min(worst, f)
            if worst is not None and worst < 8.0:
                warn = box.column(align=True)
                warn.alert = worst < 6.0
                warn.label(text="Finest feature spans %.1f samples" % worst,
                           icon='ERROR' if worst < 6.0 else 'INFO')
                warn.label(text="Raise Resolution to about %d for a smooth "
                                "crest" % _resolution_for(props, worst, 10.0))
            col.prop(props, 'tiling')
            if props.tiling != 'NONE':
                col.prop(props, 'tile_x')
                col.prop(props, 'tile_y')
                if props.tile_x > 1 or props.tile_y > 1:
                    col.label(text="%d copies, one mesh, no joint"
                                   % (props.tile_x * props.tile_y),
                              icon='INFO')
            elif props.tile_x > 1 or props.tile_y > 1:
                col.label(text="Tiling is off; copies would show a joint",
                          icon='ERROR')

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
                    sub.prop(lay_, 'method')
                    if lay_.method == 'FBM':
                        sub.prop(lay_, 'hurst')
                    else:
                        sub.prop(lay_, 'dim')
                    sub.prop(lay_, 'octaves')
                    sub.prop(lay_, 'anisotropy')
                    if lay_.anisotropy > 0.0:
                        sub.prop(lay_, 'wind')
                        if props.tiling != 'NONE':
                            sub.label(text="Tiling overrides the wind",
                                      icon='INFO')
                elif lay_.kind == 'ELLIPTIC':
                    sub.prop(lay_, 'ell_kind')
                    sub.prop(lay_, 'ell_part')
                    sub.prop(lay_, 'ell_cells')
                    sub.prop(lay_, 'tau_re')
                    sub.prop(lay_, 'tau_im')
                    if lay_.ell_kind not in ('WP', 'WP_PRIME'):
                        sub.label(text="Only P and P' are doubly periodic",
                                  icon='INFO')
                elif lay_.kind == 'TRUCHET':
                    sub.prop(lay_, 'tile_cells')
                    sub.prop(lay_, 'lane')
                    sub.prop(lay_, 'straight')
                elif lay_.kind == 'SEIGAIHA':
                    sub.prop(lay_, 'tile_cells')
                    sub.prop(lay_, 'rings')
                    sub.prop(lay_, 'crown')
                    sub.prop(lay_, 'rim')
                elif lay_.kind == 'WORLEY':
                    sub.prop(lay_, 'cell_mode')
                    sub.prop(lay_, 'points_n')
                    sub.prop(lay_, 'jitter')
                elif lay_.kind == 'TURING':
                    sub.prop(lay_, 'regime')
                    sub.prop(lay_, 'rd_scale')
                    sub.prop(lay_, 'rd_steps')
                elif lay_.kind == 'WALLPAPER':
                    sub.prop(lay_, 'group')
                    sub.prop(lay_, 'sym_cells')
                    sub.prop(lay_, 'waves')
                elif lay_.kind == 'QUASI':
                    sub.prop(lay_, 'fold')
                    sub.prop(lay_, 'qc_cells')
                    sub.prop(lay_, 'qc_sharp')
                elif lay_.kind == 'OCEAN':
                    sub.prop(lay_, 'wind_speed')
                    sub.prop(lay_, 'choppy')
                elif lay_.kind == 'GABOR':
                    sub.prop(lay_, 'gabor_freq')
                    sub.prop(lay_, 'gabor_band')
                    sub.prop(lay_, 'spread')
                elif lay_.kind == 'DRUM':
                    sub.prop(lay_, 'drum_shape')
                    sub.prop(lay_, 'mode_index')
                    sub.prop(lay_, 'drum_res')
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
            col.prop(props, 'pierce')
            if props.pierce:
                col.prop(props, 'pierce_level')
                col.prop(props, 'pierce_invert')
                col.prop(props, 'pierce_min')
            if props.form == 'SLAB':
                col.prop(props, 'base_thickness')
            col.prop(props, 'fit')
            col.prop(props, 'scale')
            col.prop(props, 'smooth')

    _FIELD_ITEMS_MAP = [(k, lbl) for k, lbl, _d in _FIELD_ITEMS]

    _STACK_CLASSES = (ReliefLayer, ReliefPanelProps, RELIEF_UL_layers,
                      RELIEF_OT_export_heightmap,
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
