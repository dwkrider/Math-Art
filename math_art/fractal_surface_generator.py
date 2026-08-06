
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


# ==========================================================================
# Fractal height field: a sum of cosine modes cos(K.P + phase) * amp
# ==========================================================================

def _fib_dirs(count):
    """`count` near-uniform directions on the unit sphere (deterministic
    Fibonacci spiral), so the Weierstrass field is close to isotropic."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = np.empty((count, 3))
    for i in range(count):
        z = 1.0 - 2.0 * (i + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        t = ga * i
        out[i] = (r * math.cos(t), r * math.sin(t), z)
    return out


def weierstrass_modes(octaves, lacunarity, dim, dirs_per_oct=4,
                      k0=math.pi):
    """Weierstrass-Mandelbrot modes: geometric frequencies with
    amplitude lacunarity^((D-3) n), several ridge directions per
    octave. Deterministic."""
    dirs = _fib_dirs(octaves * dirs_per_oct)
    K, A, P = [], [], []
    idx = 0
    for n in range(octaves):
        freq = k0 * lacunarity ** n
        amp = lacunarity ** ((dim - 3.0) * n)
        for _ in range(dirs_per_oct):
            K.append(dirs[idx] * freq)
            A.append(amp)
            P.append((idx * 0.6180339887498949) % 1.0 * 2.0 * math.pi)
            idx += 1
    return np.array(K), np.array(A), np.array(P)


def fbm_modes(count, octaves, lacunarity, hurst, seed, k0=math.pi):
    """Fractional Brownian modes by spectral synthesis: random
    directions, log-uniform frequencies over the octave band,
    amplitude |k|^-(H+1). Seeded."""
    rng = np.random.default_rng(seed)
    kmax = k0 * lacunarity ** octaves
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    f = k0 * (kmax / k0) ** rng.random(count)
    K = d * f[:, None]
    A = f ** (-(hurst + 1.0))
    P = rng.random(count) * 2.0 * math.pi
    return K, A, P


def eval_field(points, modes):
    """Evaluate sum_i A_i cos(K_i . P + phase_i) at each point, then
    normalise to zero mean and unit standard deviation."""
    K, A, P = modes
    ang = points @ K.T + P[None, :]
    f = (np.cos(ang) * A[None, :]).sum(axis=1)
    f -= f.mean()
    sd = f.std()
    return f / sd if sd > 1e-9 else f


# ==========================================================================
# Base surfaces -> (points, displace-direction, faces)
# ==========================================================================

def _grid_faces(nu, nv, wrap_u=False, wrap_v=False):
    faces = []
    for i in range(nu - (0 if wrap_u else 1)):
        for j in range(nv - (0 if wrap_v else 1)):
            i1 = (i + 1) % nu
            j1 = (j + 1) % nv
            faces.append([i * nv + j, i1 * nv + j,
                          i1 * nv + j1, i * nv + j1])
    return faces


def base_plate(res):
    xs = np.linspace(-1.0, 1.0, res)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=-1)
    D = np.tile((0.0, 0.0, 1.0), (P.shape[0], 1))
    return P, D, _grid_faces(res, res)


def base_sphere(res):
    nu, nv = res, res
    u = np.linspace(0.0, math.pi, nu)               # polar
    v = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    P = np.stack([(np.sin(U) * np.cos(Vv)).ravel(),
                  (np.sin(U) * np.sin(Vv)).ravel(),
                  np.cos(U).ravel()], axis=-1)
    return P, P.copy(), _grid_faces(nu, nv, wrap_v=True)


def base_torus(res, ring=1.0, tube=0.4):
    nu, nv = res, res
    u = np.linspace(0.0, 2.0 * math.pi, nu, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    cu, su, cv, sv = np.cos(U), np.sin(U), np.cos(Vv), np.sin(Vv)
    P = np.stack([((ring + tube * cv) * cu).ravel(),
                  ((ring + tube * cv) * su).ravel(),
                  (tube * sv).ravel()], axis=-1)
    D = np.stack([(cv * cu).ravel(), (cv * su).ravel(), sv.ravel()],
                 axis=-1)
    return P, D, _grid_faces(nu, nv, wrap_u=True, wrap_v=True)


BASES = {'PLATE': base_plate, 'SPHERE': base_sphere,
         'TORUS': base_torus}


def _weld(V, faces, eps=1e-5):
    """Merge coincident vertices (e.g. the sphere's collapsed pole
    rows) and drop the degenerate faces that result, so the closed
    bases stay watertight."""
    keys = np.round(V / eps).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    newV = np.zeros((len(uniq), 3))
    newV[inv] = V
    out = []
    for f in faces:
        nf = [int(inv[i]) for i in f]
        dd = [nf[k] for k in range(len(nf))
              if nf[k] != nf[(k + 1) % len(nf)]]
        if len(dd) >= 3:
            out.append(dd)
    return newV, out


def build_fractal_surface(base='PLATE', method='WEIERSTRASS', res=128,
                          octaves=9, lacunarity=2.0, dim=2.3,
                          hurst=0.7, amplitude=0.3, count=240,
                          seed=1, scale=1.0):
    """Displace a base surface by a Weierstrass-Mandelbrot or fBm
    field. Returns (verts, faces) centred and fit to a 2 m cube."""
    P, D, faces = BASES[base](res)
    if method == 'WEIERSTRASS':
        modes = weierstrass_modes(octaves, lacunarity, dim)
    else:
        modes = fbm_modes(count, octaves, lacunarity, hurst, seed)
    h = eval_field(P, modes)
    V = P + (amplitude * h)[:, None] * D
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    V = (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
    if base != 'PLATE':                 # close sphere poles / weld
        V, faces = _weld(V, faces)
    return V * scale, faces


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
