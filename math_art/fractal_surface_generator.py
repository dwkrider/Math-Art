
# Fractal Relief / Terrain Generator for Blender
#
# Smooth-but-fractal surfaces: a base surface (flat plate, sphere or
# torus) displaced by a self-affine height field that is the SUM of a
# geometric series of cosine "octaves". Truncated to a finite number
# of octaves the field is a smooth, rolling, wavy sheet; in the limit
# it is continuous but nowhere differentiable, and its roughness is a
# single tunable knob. Two field models are offered:
#
#   WEIERSTRASS  the Weierstrass-Mandelbrot fractal surface (Ausloos-
#                Berman): octave n has frequency lacunarity^n and
#                amplitude lacunarity^((D-3) n), so the fractal
#                dimension D in (2, 3) IS the roughness -- D near 2 is
#                silky, D near 3 is crumpled. Deterministic, no seed.
#   FBM          fractional Brownian motion by spectral synthesis: a
#                cloud of random Fourier modes with amplitude
#                |k|^-(H+1), the Hurst exponent H in (0, 1) tuning
#                smooth-rolling (H -> 1) to rugged (H -> 0). Seeded.
#
# On a sphere the same field displaces the radius (a lumpy "fractal
# planet"); on a torus it displaces the surface normal. Output is
# centred on the origin and fitted to a 2 m cube; pair it with the
# Curvature Colour operator for a striking read of the relief.
#
# References:
# - Weierstrass-Mandelbrot function: Michael V. Berry and Z. V. Lewis,
#   "On the Weierstrass-Mandelbrot fractal function", Proc. R. Soc.
#   Lond. A 370, 1980, pp. 459-484. Surface generalisation: Marcel
#   Ausloos and D. H. Berman, "A multivariate Weierstrass-Mandelbrot
#   function", Proc. R. Soc. Lond. A 400, 1985, pp. 331-350.
# - Fractional Brownian surfaces / spectral synthesis: Benoit B.
#   Mandelbrot, "The Fractal Geometry of Nature", Freeman, 1982; Alain
#   Fournier, Don Fussell and Loren Carpenter, "Computer rendering of
#   stochastic models", Comm. ACM 25(6), 1982, pp. 371-384; Dietmar
#   Saupe, "Algorithms for random fractals", in "The Science of
#   Fractal Images", Springer, 1988.

bl_info = {
    "name": "Fractal Relief / Terrain",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Fractal Relief",
    "description": "Weierstrass-Mandelbrot and fBm fractal surfaces "
                   "on a plate, sphere or torus",
    "category": "Add Mesh",
}

import math

import numpy as np
# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.spectral import (build_fractal_surface)
except ImportError:  # flat import outside the package
    from ifs.spectral import (build_fractal_surface)



# ==========================================================================
# Fractal height field: a sum of cosine modes cos(K.P + phase) * amp
# ==========================================================================


# ==========================================================================
# Base surfaces -> (points, displace-direction, faces)
# ==========================================================================


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_fractal_surface_add(bpy.types.Operator):
        """Add a Weierstrass-Mandelbrot or fBm fractal surface on a
        plate, sphere or torus"""
        bl_idname = "mesh.fractal_surface_add"
        bl_label = "Fractal Relief"
        bl_options = {'REGISTER', 'UNDO'}

        base: EnumProperty(
            name="Base",
            items=[('PLATE', "Plate", "Flat sheet, displaced in z"),
                   ('SPHERE', "Sphere (Planet)",
                    "Radius displaced -- a lumpy fractal planet"),
                   ('TORUS', "Torus", "Normal-displaced torus")],
            default='PLATE')
        method: EnumProperty(
            name="Field",
            items=[('WEIERSTRASS', "Weierstrass-Mandelbrot",
                    "Deterministic; fractal dimension is the knob"),
                   ('FBM', "Fractional Brownian (fBm)",
                    "Random spectral synthesis; Hurst is the knob")],
            default='WEIERSTRASS')
        resolution: IntProperty(
            name="Resolution", default=128, min=16, max=512,
            description="Base surface grid resolution per axis")
        dim: FloatProperty(
            name="Fractal Dimension", default=2.3, min=2.01, max=2.95,
            description="Weierstrass surface dimension D in (2,3): "
                        "low = smooth, high = crumpled")
        hurst: FloatProperty(
            name="Hurst Exponent", default=0.7, min=0.05, max=0.99,
            description="fBm roughness: high = smooth rolling, "
                        "low = rugged")
        octaves: IntProperty(
            name="Octaves", default=9, min=1, max=16,
            description="Number of frequency octaves summed")
        lacunarity: FloatProperty(
            name="Lacunarity", default=2.0, min=1.2, max=4.0,
            description="Frequency ratio between octaves")
        amplitude: FloatProperty(
            name="Amplitude", default=0.3, min=0.0, max=2.0,
            description="Displacement strength (before the fit to "
                        "2 m)")
        count: IntProperty(
            name="fBm Modes", default=240, min=16, max=1200,
            description="Number of random Fourier modes (fBm only)")
        seed: IntProperty(
            name="Seed", default=1, min=0, max=100000,
            description="Random seed (fBm only)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            verts, faces = build_fractal_surface(
                base=self.base, method=self.method,
                res=self.resolution, octaves=self.octaves,
                lacunarity=self.lacunarity, dim=self.dim,
                hurst=self.hurst, amplitude=self.amplitude,
                count=self.count, seed=self.seed, scale=self.scale)
            me = bpy.data.meshes.new("FractalRelief")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Fractal Relief", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{self.base}/{self.method}: "
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'base')
            lay.prop(self, 'method')
            lay.prop(self, 'resolution')
            if self.method == 'WEIERSTRASS':
                lay.prop(self, 'dim')
            else:
                lay.prop(self, 'hurst')
                lay.prop(self, 'count')
                lay.prop(self, 'seed')
            lay.prop(self, 'octaves')
            lay.prop(self, 'lacunarity')
            lay.prop(self, 'amplitude')
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.fractal_surface_add",
                             icon='RNDCURVE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_fractal_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_fractal_surface_add)


def _selftest():
    from collections import Counter
    for base in ('PLATE', 'SPHERE', 'TORUS'):
        for method in ('WEIERSTRASS', 'FBM'):
            verts, faces = build_fractal_surface(
                base=base, method=method, res=48)
            # closed check only meaningful for sphere/torus
            edges = Counter()
            for f in faces:
                for k in range(len(f)):
                    a, b = f[k], f[(k + 1) % len(f)]
                    edges[(min(a, b), max(a, b))] += 1
            bnd = sum(1 for c in edges.values() if c != 2)
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            finite = np.isfinite(verts).all()
            print(f"{base:7s}/{method:11s}: V={len(verts)} "
                  f"F={len(faces)} openEdges={bnd} "
                  f"bbox={np.round(hi - lo, 2)} "
                  f"{'OK' if len(faces) and finite else 'BAD'}")
            assert len(faces) and finite
