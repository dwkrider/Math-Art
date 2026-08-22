# Willmore surface generator: bending-energy minimizers as printable
# art.  Three families, all produced by genuine constrained descent on
# the curvature energy E = int (H - h0)^2 dA (solver/willmore) rather
# than closed-form drawing:
#
#   * Willmore torus -- start from ANY torus of revolution and flow the
#     Willmore energy (h0 = 0, unconstrained: the energy is scale-
#     invariant).  The minimizer among all tori is the stereographically
#     projected Clifford torus -- aspect ratio sqrt(2), energy 2 pi^2 --
#     which was the Willmore conjecture (1965) until Marques and Neves
#     proved it (2014).  The generator reports the achieved energy
#     against 2 pi^2 and the fitted aspect ratio against sqrt(2).
#
#   * Vesicle -- fix the AREA and the VOLUME at reduced volume v < 1
#     and minimize bending energy: the Helfrich membrane problem, whose
#     equilibria are the shapes of red blood cells.  An oblate start
#     descends to the biconcave DISCOCYTE (the red-cell shape); a
#     prolate start at high v to a capsule.  Optional spontaneous
#     curvature h0 biases the branch.
#
#   * Inflated ring -- a torus at fixed enclosed volume with PRESCRIBED
#     mean curvature h0 > 0: the tube thins toward H ~ h0 and the ring
#     widens, giving slender prescribed-curvature loops.
#
# The energy, its exact analytic first variation, the normal-only
# stabilized L-BFGS descent, and the volume/area constraint machinery
# all live in math_art/solver/willmore.py (see its header for the
# derivation and the measured stability story).
#
# References:
# - T. J. Willmore, "Note on embedded surfaces", An. Sti. Univ. "Al.
#   I. Cuza" Iasi Sect. I a Mat. 11B (1965), 493-496.
# - F. C. Marques and A. Neves, "Min-max theory and the Willmore
#   conjecture", Annals of Mathematics 179(2) (2014), 683-782.
# - W. Helfrich, "Elastic properties of lipid bilayers: theory and
#   possible experiments", Z. Naturforsch. C 28 (1973), 693-703.
# - U. Seifert, K. Berndl, R. Lipowsky, "Shape transformations of
#   vesicles", Physical Review A 44 (1991), 1182-1202 -- the
#   discocyte/prolate/stomatocyte phase diagram at reduced
#   volume < 1.
# - C. Zhu, C. T. Lee, P. Rangamani, "Mem3DG: Modeling membrane
#   mechanochemical dynamics in 3D using discrete differential
#   geometry", Biophysical Reports 2(3), 100062 (2022).

bl_info = {
    "name": "Willmore Surface",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Willmore tori (Clifford shape), red-blood-cell "
                   "vesicles, and prescribed-curvature rings by "
                   "bending-energy minimization",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    import bpy
    from bpy.props import IntProperty, FloatProperty, EnumProperty, \
        BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from .solver import willmore as _swil
    from .solver import volume as _svol
except ImportError:                      # flat import outside the package
    from solver import willmore as _swil     # type: ignore
    from solver import volume as _svol       # type: ignore


# --------------------------------------------------------------------
# Pure-math core (no bpy)
# --------------------------------------------------------------------

def _closed_labels(T):
    lab = np.zeros((len(T), 2), dtype=np.int64)
    lab[:, 1] = 1
    return lab


def _icosphere(subdiv):
    try:
        from .surfaces.primitives import icosphere
    except ImportError:
        from surfaces.primitives import icosphere    # type: ignore
    SV, SF = icosphere(subdiv, 'per_level')
    return np.asarray(SV, float), np.asarray(SF, dtype=np.int64)


def make_willmore_torus(nu=48, initial_ratio=3.0, iters=400,
                        groom_every=10):
    """Flow a torus of revolution (major/minor ratio `initial_ratio`)
    to the Willmore minimizer.  Returns (V, T, info) with info carrying
    E, the 2 pi^2 target, and the fitted aspect ratio."""
    nv = max(8, nu // 2)
    V, T = _swil.torus_mesh(nu, nv, R=float(initial_ratio), r=1.0)
    T = T.copy()
    info = _swil.minimize_willmore(V, T, iters=iters,
                                   groom_every=groom_every, step_cap=0.5)
    R, rm, rcv, ratio = _swil.fit_torus_of_revolution(V)
    info.update({"ratio": ratio, "tube_cv": rcv,
                 "E_target": 2.0 * math.pi * math.pi})
    return V, T, info


def make_vesicle(subdiv=3, reduced_volume=0.65, h0=0.0, seed="OBLATE",
                 iters=400, groom_every=10):
    """Fixed-area, fixed-volume bending minimizer at the given reduced
    volume v = V / ((4 pi/3)(A/4 pi)^(3/2)) -- v = 1 is the sphere,
    v ~ 0.65 the classic discocyte.  `seed` picks the descent branch:
    OBLATE squashes the seed sphere (discocyte branch), PROLATE
    stretches it (capsule branch)."""
    v = float(reduced_volume)
    if not (0.5 <= v <= 1.0):
        raise ValueError("reduced volume must be in [0.5, 1]")
    SV, SF = _icosphere(subdiv)
    area0 = _svol.mesh_area(SV, SF)
    vol0 = float(_svol.region_volumes(SV, SF, _closed_labels(SF), 1)[0])
    if seed == "PROLATE":
        s = 1.0 / math.sqrt(v)
        V = SV * np.array([1.0 / s, 1.0 / s, s * s])
    else:
        V = SV * np.array([1.0, 1.0, v])
    T = SF.copy()
    info = _swil.minimize_willmore(V, T, h0=float(h0), iters=iters,
                                   groom_every=groom_every, step_cap=0.5,
                                   vol_target=v * vol0, area_target=area0)
    info["E_target"] = 4.0 * math.pi
    return V, T, info


def make_inflated_ring(nu=48, initial_ratio=2.5, h0=1.0, iters=400,
                       groom_every=10):
    """Torus at fixed enclosed volume with prescribed mean curvature
    h0: minimizes int (H - h0)^2 dA, thinning the tube toward H ~ h0."""
    nv = max(8, nu // 2)
    V, T = _swil.torus_mesh(nu, nv, R=float(initial_ratio), r=1.0)
    T = T.copy()
    vol0 = float(_svol.region_volumes(V, T, _closed_labels(T), 1)[0])
    info = _swil.minimize_willmore(V, T, h0=float(h0), iters=iters,
                                   groom_every=groom_every, step_cap=0.5,
                                   vol_target=vol0)
    R, rm, rcv, ratio = _swil.fit_torus_of_revolution(V)
    info.update({"ratio": ratio, "tube_cv": rcv})
    return V, T, info


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_willmore_add(bpy.types.Operator):
        """Minimize the bending energy of a closed surface: Willmore
        tori converging to the Clifford shape, red-blood-cell vesicles
        at reduced volume, and prescribed-curvature rings"""
        bl_idname = "mesh.willmore_add"
        bl_label = "Willmore Surface"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            description="Which Willmore-energy surface to build",
            items=[
                ('TORUS', "Willmore Torus (Clifford)",
                 "Flow a torus to the Willmore minimizer: the Clifford "
                 "torus, aspect ratio sqrt(2), energy 2 pi squared "
                 "(Marques-Neves 2014)"),
                ('VESICLE', "Vesicle (red blood cell)",
                 "Minimize bending energy at fixed area and volume: "
                 "the Helfrich membrane problem whose equilibria are "
                 "red-blood-cell discocytes"),
                ('RING', "Inflated Ring",
                 "Torus at fixed volume with prescribed mean "
                 "curvature: the tube thins toward H = h0 and the "
                 "ring widens"),
            ],
            default='TORUS')
        resolution: IntProperty(
            name="Resolution", default=48, min=24, max=96,
            description="Vertices around the major circle (tube gets "
                        "half); vesicles use the sphere subdivision "
                        "below instead")
        subdiv: IntProperty(
            name="Sphere Subdivision", default=3, min=2, max=4,
            description="Icosphere subdivision for the vesicle seed "
                        "(162 / 642 / 2562 vertices)")
        initial_ratio: FloatProperty(
            name="Start Ratio", default=3.0, min=1.6, max=6.0,
            description="Major/minor ratio of the starting torus -- "
                        "the flow forgets it, which is the point; "
                        "watch a fat or thin torus find sqrt(2)")
        reduced_volume: FloatProperty(
            name="Reduced Volume", default=0.65, min=0.5, max=0.99,
            description="Enclosed volume relative to the sphere of the "
                        "same area: 1 is the sphere, ~0.65 the classic "
                        "biconcave discocyte; lower deepens the "
                        "dimples")
        seed_shape: EnumProperty(
            name="Seed",
            items=[('OBLATE', "Oblate (discocyte)",
                    "Squashed-sphere start: descends to the biconcave "
                    "red-cell branch"),
                   ('PROLATE', "Prolate (capsule)",
                    "Stretched-sphere start: descends to the "
                    "cigar/capsule branch")],
            default='OBLATE',
            description="Which side of the oblate/prolate energy "
                        "barrier the descent starts on")
        h0: FloatProperty(
            name="Spontaneous Curvature", default=1.0, min=0.0, max=2.5,
            description="Prescribed mean curvature h0 of the "
                        "(H - h0)^2 energy (RING mode; also biases "
                        "VESICLE branches)")
        iterations: IntProperty(
            name="Iterations", default=400, min=20, max=2000,
            description="Descent budget; the flow stops early at its "
                        "stationary point")
        smooth: BoolProperty(name="Smooth Shading", default=True,
                             description="Shade the surface smooth")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size multiplier")

        def execute(self, context):
            try:
                if self.mode == 'TORUS':
                    V, T, info = make_willmore_torus(
                        nu=self.resolution,
                        initial_ratio=self.initial_ratio,
                        iters=self.iterations)
                    name = "Willmore Torus"
                    extra = (f"E {info['E']:.4f} vs 2pi^2 "
                             f"{info['E_target']:.4f}, aspect "
                             f"{info['ratio']:.4f} vs sqrt2 "
                             f"{math.sqrt(2):.4f}")
                elif self.mode == 'VESICLE':
                    V, T, info = make_vesicle(
                        subdiv=self.subdiv,
                        reduced_volume=self.reduced_volume,
                        h0=(self.h0 if self.h0 > 0.0 else 0.0),
                        seed=self.seed_shape, iters=self.iterations)
                    name = "Vesicle"
                    extra = (f"E {info['E']:.3f} (sphere 4pi = "
                             f"{info['E_target']:.3f}), v = "
                             f"{self.reduced_volume:.2f}, constraint "
                             f"drift {info['drift_max']:.1e}")
                else:
                    V, T, info = make_inflated_ring(
                        nu=self.resolution,
                        initial_ratio=self.initial_ratio,
                        h0=self.h0, iters=self.iterations)
                    name = "Inflated Ring"
                    extra = (f"E {info['E']:.3f}, ring aspect "
                             f"{info['ratio']:.2f}")
            except Exception as e:
                self.report({'ERROR'}, f"willmore flow failed: {e}")
                return {'CANCELLED'}

            lo = V.min(axis=0)
            hi = V.max(axis=0)
            ctr = 0.5 * (lo + hi)
            half = float(np.max(hi - lo)) / 2.0 or 1.0
            s = self.scale / half
            Vt = (V - ctr) * s

            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(p) for p in Vt], [],
                           [list(t) for t in T])
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [self.smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {info['iters_run']} iterations, "
                        f"{extra}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'TORUS':
                lay.prop(self, 'initial_ratio')
                lay.prop(self, 'resolution')
            elif self.mode == 'VESICLE':
                lay.prop(self, 'reduced_volume')
                lay.prop(self, 'seed_shape')
                lay.prop(self, 'subdiv')
                lay.prop(self, 'h0')
            else:
                lay.prop(self, 'h0')
                lay.prop(self, 'initial_ratio')
                lay.prop(self, 'resolution')
            for k in ('iterations', 'smooth', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.willmore_add", icon='MESH_TORUS')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_willmore_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_willmore_add)


# --------------------------------------------------------------------
# self-test (headless; the fine-resolution battery is tests/bench)
# --------------------------------------------------------------------

def _selftest():
    ok = True
    twopi2 = 2.0 * math.pi * math.pi

    # Willmore torus at coarse preview resolution: lands near the
    # Clifford shape and terminates before the budget
    V, T, info = make_willmore_torus(nu=24, initial_ratio=3.0, iters=300)
    good = (abs(info["ratio"] - math.sqrt(2.0)) < 0.08
            and abs(info["E"] - twopi2) / twopi2 < 0.05
            and info["rise_max"] <= 1e-12 and info["iters_run"] < 300)
    ok &= good
    print(f"willmore_gen: torus ratio {info['ratio']:.4f} (sqrt2 "
          f"{math.sqrt(2):.4f}), E {info['E']:.3f} vs 2pi^2 "
          f"{twopi2:.3f}, stopped at {info['iters_run']} "
          f"{'OK' if good else 'FAIL'}")

    # Vesicle: discocyte branch is oblate with dimples, E above 4 pi,
    # constraints held to solver tolerance.  v = 0.60, not 0.65: the
    # Seifert-Berndl-Lipowsky diagram puts the discocyte/prolate
    # transition AT v ~ 0.65, and the 162-vertex mesh's discretization
    # bias is enough to hop the branch there (measured); 0.60 is firmly
    # on the discocyte side at every resolution
    V, T, info = make_vesicle(subdiv=2, reduced_volume=0.60, iters=200)
    ext = V.max(axis=0) - V.min(axis=0)
    rho = np.hypot(V[:, 0] - V[:, 0].mean(), V[:, 1] - V[:, 1].mean())
    zc = V[:, 2] - V[:, 2].mean()
    axis_pts = rho < 0.15 * ext[0]
    dimple = (float(zc[axis_pts].max()) / float(zc.max())
              if axis_pts.any() else 1.0)
    good = (info["E"] > 4.0 * math.pi and info["drift_max"] < 1e-9
            and info["rise_max"] <= 1e-12
            and ext[2] / ext[0] < 0.6 and dimple < 0.95)
    ok &= good
    print(f"willmore_gen: vesicle v=0.60 E {info['E']:.3f} > 4pi, "
          f"z/x {ext[2] / ext[0]:.3f}, dimple {dimple:.3f}, drift "
          f"{info['drift_max']:.1e} {'OK' if good else 'FAIL'}")

    # Inflated ring: fixed volume held, ring widens past its start
    V, T, info = make_inflated_ring(nu=32, initial_ratio=2.5, h0=1.0,
                                    iters=200)
    good = (info["drift_max"] < 1e-9 and info["rise_max"] <= 1e-12
            and info["ratio"] > 2.5 and info["E"] < info["E0"])
    ok &= good
    print(f"willmore_gen: ring h0=1 aspect {info['ratio']:.2f} "
          f"(start 2.5), E {info['E0']:.3f} -> {info['E']:.3f}, drift "
          f"{info['drift_max']:.1e} {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("willmore_generator self-test failed")
