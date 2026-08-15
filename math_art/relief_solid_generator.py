# -----------------------------------------------------------------------
# Solid Relief: relief carved onto closed surfaces -- spheres, tori,
# cylinders.
#
# A separate generator from the Relief Panel, and deliberately so.  The panel
# answers h(x, y) on a rectangle; a closed surface has no rectangle to answer
# on.  By Gauss's Theorema Egregium a sphere cannot be flattened without
# distortion, so a projected planar field is not merely awkward there but
# wrong in a way no better projection repairs: measured on a lat-long sphere,
# faces near the pole were a tenth the area of those at the equator and 130
# vertices crowded into a five-degree cap that should hold fifteen.
#
# So the field is evaluated in 3D at the surface's own points and the mesh
# carries no poles.  Spatial continuity then costs nothing: a function of
# space restricted to a surface is continuous wherever the surface is,
# whichever way it wraps.
#
# The mathematics lives in `relief/solid.py`; this module is the Blender
# layer over it.
#
# References:
#   Carl Friedrich Gauss, "Disquisitiones generales circa superficies
#     curvas", 1827 -- the Theorema Egregium.
#   J. P. Lewis, "Algorithms for Solid Noise Synthesis", SIGGRAPH 1989,
#     263-270 -- noise defined in space, so it restricts to any surface.
#   Steven Worley, "A Cellular Texture Basis Function", SIGGRAPH 1996,
#     291-294.
#   Benoit B. Mandelbrot, "The Fractal Geometry of Nature", Freeman, 1982;
#     Dietmar Saupe, "Algorithms for random fractals", in "The Science of
#     Fractal Images", Springer, 1988 -- spectral synthesis.
#   E. W. Hobson, "The Theory of Spherical and Ellipsoidal Harmonics",
#     Cambridge University Press, 1931 -- spherical harmonics, the sphere's
#     own vibration modes.

bl_info = {
    "name": "Solid Relief",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Solid Relief",
    "description": "Relief on closed surfaces: geodesic sphere, torus, "
                   "cylinder, with the field evaluated in 3D",
    "category": "Add Mesh",
}

import numpy as np

try:
    from .relief import solid as _solid
    from .relief.solid import BASES, build_solid, FIELDS3D
    from .relief import transfer as _transfer
except ImportError:                       # flat import outside the package
    from relief import solid as _solid
    from relief.solid import BASES, build_solid, FIELDS3D
    from relief import transfer as _transfer

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PRESETS = {
        'PLANET': dict(
            base='SPHERE', sphere_res=5, field='FRACTAL', method='FBM', hurst=0.75,
            octaves=9, curve='NONE', depth=0.26, seed=3),
        'ASTEROID': dict(
            base='SPHERE', sphere_res=5, field='FRACTAL', method='WEIERSTRASS',
            dim=2.4, octaves=8, curve='NONE', depth=0.45, seed=7),
        'HARMONIC_BALL': dict(
            # The sphere's own eigenmode: a Chladni figure for a ball.
            base='SPHERE', sphere_res=5, field='HARMONIC', sph_l=6, sph_m=3,
            curve='NONE', depth=0.3),
        'SHAGREEN': dict(
            # Cellular texture with the feature points in space, so the cells
            # meet across every wrap without being matched up.
            base='SPHERE', sphere_res=5, field='CELLULAR', points_n=220,
            cell_mode='CRACK', cell_sharp=0.7, curve='NONE', depth=0.2,
            seed=11),
        'TURING_SPHERE': dict(
            base='SPHERE', sphere_res=5, field='TURING', regime='MAZE',
            rd_steps=5000, rd_scale=0.35, curve='NONE', depth=0.2, seed=3),
        'TURING_TORUS': dict(
            base='TORUS', grid_res=192, field='TURING', regime='SPOTS',
            rd_steps=5000, rd_scale=0.35, curve='NONE', depth=0.16, seed=5),
        'WOVEN_TORUS': dict(
            base='TORUS', grid_res=192, field='GABOR', gabor_freq=9.0,
            points_n=260, spread=0.4, curve='NONE', depth=0.16, seed=5),
        'CORAL_TORUS': dict(
            base='TORUS', grid_res=192, field='CELLULAR', points_n=260,
            cell_mode='CRACK', curve='RIDGE', depth=0.18, seed=2),
        'CARVED_COLUMN': dict(
            # A cylinder is developable, so a pattern on it keeps every
            # distance it had -- the one base here with no distortion at all.
            base='CYLINDER', grid_res=192, field='GABOR', gabor_freq=7.0,
            points_n=240, spread=0.2, curve='NONE', depth=0.14, seed=4),
    }

    _PRESET_KEYS = sorted({k for v in _PRESETS.values() for k in v})

    _PRESET_ITEMS = [
        ('CUSTOM', "Custom", "Leave the controls below alone"),
        ('PLANET', "Planet", "Fractional Brownian relief on a geodesic ball"),
        ('ASTEROID', "Asteroid",
         "Weierstrass-Mandelbrot relief, deep and crumpled"),
        ('HARMONIC_BALL', "Harmonic Ball",
         "A spherical harmonic -- the sphere's own vibration mode"),
        ('SHAGREEN', "Shagreen", "Cellular texture, cells cut out of space"),
        ('TURING_SPHERE', "Turing Sphere",
         "A reaction-diffusion skin grown on the ball itself"),
        ('TURING_TORUS', "Turing Torus",
         "The same chemistry on a torus, spotted"),
        ('WOVEN_TORUS', "Woven Torus", "Band-limited weave around a torus"),
        ('CORAL_TORUS', "Coral Torus", "Ridged cellular growth on a torus"),
        ('CARVED_COLUMN', "Carved Column", "A fluted, woven cylinder"),
    ]

    _FIELD_ITEMS = [
        ('FRACTAL', "Fractal",
         "Spectral synthesis in space: fBm or Weierstrass-Mandelbrot"),
        ('CELLULAR', "Cellular",
         "Worley cells cut out of space by feature points"),
        ('GABOR', "Gabor Weave",
         "Band-limited sparse convolution noise, at a chosen pitch"),
        ('HARMONIC', "Spherical Harmonic",
         "Y_lm: the sphere's own eigenfunction. Meant for the Sphere base"),
        ('TURING', "Turing Skin",
         "Gray-Scott grown on the surface itself, with no grid to map it "
         "onto and so no seam"),
        ('LATTICE', "Lattice Wave",
         "A triply periodic wave, for bases with no mode family"),
    ]

    def _preset_update(self, context):
        """Copy a preset into the live properties.

        A module-level function rather than a method: the `update` callback
        is handed an RNA-wrapped operator, which does not carry the Python
        class's own methods, so `self._apply_preset()` raises AttributeError
        the moment a preset is chosen.

        The properties stay the single source of truth, so switching to
        Custom afterwards inherits whatever the preset just set rather than
        snapping back to defaults.
        """
        if self.preset == 'CUSTOM':
            return
        values = _PRESETS[self.preset]
        # Reset anything any preset can set, so a key one preset turns on is
        # not left on by the next preset that has no opinion about it.
        for key in _PRESET_KEYS:
            if key in values or not hasattr(self, key):
                continue
            prop = self.bl_rna.properties.get(key)
            if prop is not None:
                try:
                    setattr(self, key, prop.default)
                except (TypeError, ValueError):
                    pass
        for key, value in values.items():
            if hasattr(self, key):
                try:
                    setattr(self, key, value)
                except (TypeError, ValueError):
                    pass

    class MESH_OT_relief_solid_add(bpy.types.Operator):
        """Add relief carved onto a closed surface"""
        bl_idname = "mesh.relief_solid_add"
        bl_label = "Solid Relief"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset", items=_PRESET_ITEMS, default='PLANET',
            update=_preset_update)

        base: EnumProperty(
            name="Base",
            items=[('SPHERE', "Sphere (geodesic)",
                    "Subdivided icosahedron: no poles, no seam, faces within "
                    "a third of one another"),
                   ('TORUS', "Torus",
                    "Doubly periodic, closed by index arithmetic rather than "
                    "by welding a seam"),
                   ('CYLINDER', "Cylinder",
                    "Developable, so a pattern on it keeps every distance "
                    "it had")],
            default='SPHERE')
        sphere_res: IntProperty(
            name="Subdivisions", default=5, min=1, max=7,
            description="Icosphere subdivisions. Each step quadruples the "
                        "faces: 5 is about 10k vertices, 7 is 164k")
        grid_res: IntProperty(
            name="Detail", default=128, min=12, max=512,
            description="Grid resolution around the torus or along the "
                        "cylinder")
        ring: FloatProperty(name="Ring Radius", default=1.0, min=0.1,
                            max=10.0)
        tube: FloatProperty(name="Tube Radius", default=0.4, min=0.02,
                            max=5.0)
        height: FloatProperty(name="Height", default=2.0, min=0.1, max=20.0)
        radius: FloatProperty(name="Radius", default=0.7, min=0.05, max=10.0)

        field: EnumProperty(name="Pattern", items=_FIELD_ITEMS,
                            default='FRACTAL')
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
        lacunarity: FloatProperty(name="Lacunarity", default=2.0, min=1.2,
                                  max=4.0)
        modes: IntProperty(name="fBm Modes", default=240, min=16, max=1200)
        field_scale: FloatProperty(name="Feature Scale", default=1.0,
                                   min=0.05, max=20.0)

        points_n: IntProperty(name="Points", default=120, min=4, max=2000)
        cell_mode: EnumProperty(
            name="Cells",
            items=[('CRACK', "Cell Walls", "F2 - F1: the Voronoi boundaries"),
                   ('F1', "Domes", "Distance to the nearest point"),
                   ('F2', "Second Nearest", "Distance to the second")],
            default='CRACK')
        cell_sharp: FloatProperty(name="Wall Profile", default=1.0, min=0.2,
                                  max=4.0)
        gabor_freq: FloatProperty(name="Pitch", default=6.0, min=0.5,
                                  max=40.0)
        gabor_band: FloatProperty(name="Bandwidth", default=0.35, min=0.05,
                                  max=1.0)
        spread: FloatProperty(name="Direction Spread", default=3.1416,
                              min=0.0, max=3.1416, unit='ROTATION')

        regime: EnumProperty(
            name="Regime",
            items=[(k, k.title(), "Gray-Scott feed %.3f, kill %.3f"
                    % _solid.SURFACE_REGIMES[k])
                   for k in _solid.SURFACE_REGIME_ORDER],
            default='MAZE',
            description="Where in the feed/kill plane the chemistry sits. "
                        "These pairs are the ones that live on a MESH, which "
                        "is not the region that lives on a grid")
        rd_steps: IntProperty(
            name="Steps", default=4000, min=100, max=30000,
            description="Reaction-diffusion is grown, not evaluated")
        rd_scale: FloatProperty(
            name="Feature Size", default=0.35, min=0.05, max=1.2,
            description="Diffusion per step; larger spreads the pattern "
                        "into bigger blobs")
        sph_l: IntProperty(name="Degree l", default=4, min=0, max=24)
        sph_m: IntProperty(name="Order m", default=2, min=-24, max=24)
        mode_m: IntProperty(name="m", default=3, min=0, max=16)
        mode_n: IntProperty(name="n", default=2, min=0, max=16)
        mode_k: IntProperty(name="k", default=1, min=0, max=16)
        phase: FloatProperty(name="Phase", default=0.0, min=-6.2832,
                             max=6.2832, unit='ROTATION')

        curve: EnumProperty(
            name="Profile",
            items=[(c, c.title(), "") for c in _transfer.CURVES],
            default='NONE')
        curve_amount: FloatProperty(name="Amount", default=1.0, min=0.05,
                                    max=4.0)
        depth: FloatProperty(name="Relief Depth", default=0.25, min=0.0,
                             max=4.0, unit='LENGTH')
        seed: IntProperty(name="Seed", default=1, min=0, max=100000)
        fit: EnumProperty(
            name="Fit",
            items=[('CUBE', "2 m Cube", ""), ('NONE', "None", "")],
            default='CUBE')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def _params(self):
            return dict(
                base=self.base, sphere_res=self.sphere_res,
                grid_res=self.grid_res, ring=self.ring, tube=self.tube,
                height=self.height, radius=self.radius,
                field=self.field, method=self.method, hurst=self.hurst,
                dim=self.dim, octaves=self.octaves,
                lacunarity=self.lacunarity, modes=self.modes,
                field_scale=self.field_scale,
                points_n=self.points_n, cell_mode=self.cell_mode,
                cell_sharp=self.cell_sharp, gabor_freq=self.gabor_freq,
                gabor_band=self.gabor_band, spread=self.spread,
                regime=self.regime, rd_steps=self.rd_steps,
                rd_scale=self.rd_scale,
                sph_l=self.sph_l, sph_m=self.sph_m, mode_m=self.mode_m,
                mode_n=self.mode_n, mode_k=self.mode_k, phase=self.phase,
                curve=self.curve, curve_amount=self.curve_amount,
                depth=self.depth, seed=self.seed, fit=self.fit,
                scale=self.scale)

        def execute(self, context):
            try:
                verts, faces, info = build_solid(**self._params())
            except (ValueError, MemoryError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

            me = bpy.data.meshes.new("SolidRelief")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.smooth and len(me.polygons):
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Solid Relief", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        "%s / %s: V=%d F=%d, base face areas vary %.2fx"
                        % (self.base.title(), self.field.title(),
                           len(me.vertices), len(me.polygons),
                           info['area_ratio']))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.prop(self, 'preset')

            box = lay.box()
            box.label(text="Base")
            col = box.column()
            col.prop(self, 'base')
            if self.base == 'SPHERE':
                col.prop(self, 'sphere_res')
                if self.sphere_res > 6:
                    col.label(text="Each step quadruples the faces",
                              icon='INFO')
            else:
                col.prop(self, 'grid_res')
            if self.base == 'TORUS':
                col.prop(self, 'ring')
                col.prop(self, 'tube')
            elif self.base == 'CYLINDER':
                col.prop(self, 'height')
                col.prop(self, 'radius')

            box = lay.box()
            box.label(text="Pattern")
            col = box.column()
            col.prop(self, 'field')
            if self.field == 'FRACTAL':
                col.prop(self, 'method')
                if self.method == 'FBM':
                    col.prop(self, 'hurst')
                    col.prop(self, 'modes')
                else:
                    col.prop(self, 'dim')
                col.prop(self, 'octaves')
                col.prop(self, 'lacunarity')
                col.prop(self, 'field_scale')
            elif self.field == 'CELLULAR':
                col.prop(self, 'cell_mode')
                col.prop(self, 'points_n')
                col.prop(self, 'cell_sharp')
            elif self.field == 'GABOR':
                col.prop(self, 'gabor_freq')
                col.prop(self, 'gabor_band')
                col.prop(self, 'spread')
                col.prop(self, 'points_n')
            elif self.field == 'TURING':
                col.prop(self, 'regime')
                col.prop(self, 'rd_scale')
                col.prop(self, 'rd_steps')
                col.label(text="Grown on the mesh; cost is the step count",
                          icon='INFO')
            elif self.field == 'HARMONIC':
                col.prop(self, 'sph_l')
                col.prop(self, 'sph_m')
                col.prop(self, 'phase')
                if abs(self.sph_m) > self.sph_l:
                    col.label(text="Order m is clamped to |m| <= l",
                              icon='INFO')
                if self.base != 'SPHERE':
                    col.label(text="Spherical harmonics want the Sphere base",
                              icon='INFO')
            elif self.field == 'LATTICE':
                col.prop(self, 'mode_m')
                col.prop(self, 'mode_n')
                col.prop(self, 'mode_k')
                col.prop(self, 'phase')
            col.prop(self, 'seed')

            box = lay.box()
            box.label(text="Output")
            col = box.column()
            col.prop(self, 'curve')
            if self.curve != 'NONE':
                col.prop(self, 'curve_amount')
            col.prop(self, 'depth')
            col.prop(self, 'fit')
            col.prop(self, 'scale')
            col.prop(self, 'smooth')

    def menu_func(self, context):
        self.layout.operator("mesh.relief_solid_add", icon='MATSPHERE')

    _CLASSES = (MESH_OT_relief_solid_add,)

    def register():
        for cls in _CLASSES:
            bpy.utils.register_class(cls)

    def unregister():
        for cls in reversed(_CLASSES):
            bpy.utils.unregister_class(cls)


def _selftest():
    ok = True
    from relief.solid import BASES as _B

    for name in ('PLANET', 'ASTEROID', 'HARMONIC_BALL', 'SHAGREEN',
                 'WOVEN_TORUS', 'CORAL_TORUS', 'CARVED_COLUMN'):
        preset = {
            'PLANET': dict(base='SPHERE', sphere_res=3, field='FRACTAL'),
            'ASTEROID': dict(base='SPHERE', sphere_res=3, field='FRACTAL',
                             method='WEIERSTRASS'),
            'HARMONIC_BALL': dict(base='SPHERE', sphere_res=3, field='HARMONIC',
                                  sph_l=6, sph_m=3),
            'SHAGREEN': dict(base='SPHERE', sphere_res=3, field='CELLULAR',
                             points_n=80),
            'WOVEN_TORUS': dict(base='TORUS', grid_res=64, field='GABOR',
                                points_n=80),
            'CORAL_TORUS': dict(base='TORUS', grid_res=64, field='CELLULAR',
                                points_n=80),
            'CARVED_COLUMN': dict(base='CYLINDER', grid_res=64, field='GABOR',
                                  points_n=80),
        }[name]
        verts, faces, info = build_solid(depth=0.25, seed=2, **preset)
        finite = bool(np.isfinite(np.asarray(verts)).all())
        print("relief_solid: %-14s V=%-6d F=%-6d area ratio %.2f finite=%s"
              % (name, len(verts), len(faces), info['area_ratio'], finite))
        ok = ok and finite and len(faces) > 0

    # Every field runs on every base -- the combinations are what a user
    # will actually reach for, and a field that assumes one base would fail
    # here rather than in front of them.
    for base in _B:
        for fld in ('FRACTAL', 'CELLULAR', 'GABOR', 'HARMONIC',
                    'LATTICE', 'TURING'):
            v, f, _ = build_solid(
                base=base, sphere_res=3, grid_res=48, field=fld,
                points_n=60, seed=1, rd_steps=300)
            ok = ok and np.isfinite(np.asarray(v)).all() and len(f) > 0
    print("relief_solid: all %d fields build on all %d bases"
          % (6, len(_B)))

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
