
# Growth Generator -- form decided by the environment, not by a grammar.
#
# The L-system operators rewrite a string and read it as turtle
# commands: the productions decide the form.  These four decide it the
# other way round -- the structure grows into a space, and its shape is
# a consequence of that space.  See `lsystem/growth3d.py` for the
# algorithms and what each one is really claiming.
#
# The two tree modes reach the same excurrent-to-decurrent range from
# opposite directions, which is why both are here rather than one:
# COLONIZE gets it from the ENVELOPE (change the shape the attractors
# fill and the habit changes, with no habit parameter at all), SELFORG
# from a single resource-allocation constant lambda.
#
# Timed growth is available on every mode: `growth_t` freezes a
# developmental stage, and because every element is born with zero size
# AND zero rate, a bake over the sequence is fixed-topology and
# pop-free.  See `lsystem/timed.py`.
#
# References: as `lsystem/growth3d.py` and `lsystem/timed.py` --
# Runions, Lane and Prusinkiewicz (2007); Palubicki et al., SIGGRAPH
# 2009; Borchert and Honda (1984); Witten and Sander (1981); Bosman
# (1942); ABOP chapter 6, equation (6.11).

bl_info = {
    "name": "Growth",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Plants > Growth",
    "description": "Space colonization, self-organizing trees, DLA and "
                   "the Pythagoras tree",
    "category": "Add Curve",
}

import numpy as np

try:
    from .lsystem import growth3d as g3
    from .lsystem import timed
except ImportError:                       # headless / direct execution
    from lsystem import growth3d as g3
    from lsystem import timed


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _tree_curve(nodes, parents, radii, name, radius, resolution):
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        rmax = float(radii.max()) or 1.0
        for i, p in enumerate(parents):
            if p < 0:
                continue
            sp = cu.splines.new('POLY')
            sp.points.add(1)
            sp.points[0].co = (*nodes[p], 1.0)
            sp.points[1].co = (*nodes[i], 1.0)
            sp.points[0].radius = float(radii[p] / rmax)
            sp.points[1].radius = float(radii[i] / rmax)
        cu.bevel_depth = float(radius)
        cu.bevel_resolution = int(resolution)
        return bpy.data.objects.new(name, cu)

    def _points_mesh(pts, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for p in pts], [], [])
        me.update()
        return bpy.data.objects.new(name, me)

    def _quads_mesh(quads, depths, name):
        verts, faces = [], []
        for q in quads:
            b = len(verts)
            verts.extend(tuple(float(c) for c in v) for v in q)
            faces.append((b, b + 1, b + 2, b + 3))
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate()
        me.update()
        try:
            a = me.attributes.new(name="level", type='FLOAT', domain='FACE')
            a.data.foreach_set("value", [float(d) for d in depths])
        except Exception:
            pass
        return bpy.data.objects.new(name, me)

    class CURVE_OT_growth_add(bpy.types.Operator):
        """Add a structure grown into a space: space colonization,
        self-organizing tree, DLA cluster or Pythagoras tree"""
        bl_idname = "curve.growth_add"
        bl_label = "Growth"
        bl_options = {'REGISTER', 'UNDO', 'PRESET'}

        mode: EnumProperty(
            name="Mode",
            items=[('COLONIZE', "Space Colonization", "Grow toward "
                                "attractors in an envelope. Habit comes "
                                "from the ENVELOPE alone"),
                   ('SELFORG', "Self-Organizing", "Buds compete for "
                               "light; one lambda sweeps excurrent to "
                               "decurrent"),
                   ('DLA', "Diffusion-Limited", "Random walkers stick "
                           "where they first touch"),
                   ('PYTHAGORAS', "Pythagoras Tree", "Squares on the "
                                  "legs of a right triangle")],
            default='COLONIZE')

        # -- COLONIZE
        envelope: EnumProperty(
            name="Envelope",
            items=[(k, k.title(), f"Attractors fill a {k.lower()}")
                   for k in g3.ENVELOPES],
            default='CROWN')
        attractors: IntProperty(name="Attractors", default=700,
                                min=20, max=6000)
        trunk: FloatProperty(
            name="Trunk", default=0.5, min=0.0, max=2.0,
            description="How far the crown sits above the root. Without "
                        "a gap the tree forks from the ground and the "
                        "envelope's effect is washed out")
        influence: FloatProperty(
            name="Influence", default=0.75, min=0.05, max=3.0,
            description="How far an attractor can reach. It has to "
                        "exceed the trunk gap, or nothing in the crown "
                        "can see the root and the tree never starts")
        kill: FloatProperty(name="Kill Radius", default=0.14,
                            min=0.01, max=1.0)
        gravity: FloatProperty(
            name="Gravity", default=0.0, min=-2.0, max=2.0,
            description="Bias along the axis. Negative gives the "
                        "mycelial variant -- the same algorithm growing "
                        "down and out")

        # -- SELFORG
        steps: IntProperty(name="Steps", default=10, min=1, max=18)
        lam: FloatProperty(
            name="Lambda", default=0.5, min=0.05, max=0.95,
            description="Borchert-Honda resource split. Above 0.5 the "
                        "main axis wins and the tree is excurrent (a "
                        "conifer); below, the laterals win and it is "
                        "decurrent (a spreading crown)")
        branch_angle: FloatProperty(name="Branch Angle", default=42.0,
                                    min=5.0, max=89.0)

        # -- DLA
        walkers: IntProperty(name="Walkers", default=700, min=20,
                             max=6000)
        dla_dim: EnumProperty(
            name="Dimension",
            items=[('2', "2-D", "Planar cluster"),
                   ('3', "3-D", "Spatial cluster")],
            default='3')
        seed_kind: EnumProperty(
            name="Seed",
            items=[(k, k.title(), f"Nucleate on a {k.lower()}")
                   for k in g3.SEEDS],
            default='POINT')
        stickiness: FloatProperty(
            name="Stickiness", default=1.0, min=0.02, max=1.0,
            description="Below 1 a walker may refuse and keep going, so "
                        "it packs deeper: wispy dendrite at 1, compact "
                        "moss well below")
        lattice: IntProperty(name="Lattice", default=44, min=12, max=140)

        # -- PYTHAGORAS
        depth: IntProperty(name="Depth", default=9, min=1, max=14)
        alpha: FloatProperty(name="Apex Angle", default=45.0,
                             min=5.0, max=85.0)
        roll: FloatProperty(
            name="Roll", default=0.0, min=0.0, max=90.0,
            description="Rotate each generation out of its parent's "
                        "plane, turning the relief into a spatial figure")
        jitter: FloatProperty(name="Angle Jitter", default=0.0,
                              min=0.0, max=30.0)

        # -- shared
        seed: IntProperty(name="Seed", default=0, min=0, max=10000)
        radius: FloatProperty(name="Branch Radius", default=0.02,
                              min=0.0005, max=0.5)
        resolution: IntProperty(name="Bevel Resolution", default=2,
                                min=0, max=16)
        murray: FloatProperty(
            name="Pipe Exponent", default=g3.MURRAY, min=1.5, max=4.0,
            description="r^n = r1^n + r2^n. 2 conserves cross-sectional "
                        "area; measured trees run nearer 2.5")
        grow_t: FloatProperty(
            name="Growth Time", default=1.0, min=0.0, max=1.0,
            description="Freeze a developmental stage. Every element is "
                        "born with zero size AND zero rate, so sweeping "
                        "this is continuous -- keyframe it")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            name = f"Growth {self.mode.title()}"
            try:
                if self.mode == 'PYTHAGORAS':
                    quads, depths = g3.pythagoras(
                        depth=self.depth, alpha=self.alpha, roll=self.roll,
                        jitter=self.jitter, seed=self.seed)
                    allp = g3.fit(np.vstack(quads), 2.0 * self.scale)
                    quads = [allp[i * 4:(i + 1) * 4]
                             for i in range(len(quads))]
                    ob = _quads_mesh(quads, depths, name)
                elif self.mode == 'DLA':
                    # An aggregate is a tree, so it skins exactly like
                    # the grown ones.  Emitting bare points made the
                    # mode look empty: a vertex-only mesh is not drawn
                    # at all in Object Mode.
                    pts, par = g3.dla(
                        count=self.walkers, dim=int(self.dla_dim),
                        seed_kind=self.seed_kind,
                        stickiness=self.stickiness,
                        size=self.lattice, seed=self.seed)
                    if len(pts) < 2:
                        self.report({'ERROR'}, "no particles stuck")
                        return {'CANCELLED'}
                    radii = g3.pipe_radii(par, exponent=self.murray)
                    ob = _tree_curve(g3.fit(pts, 2.0 * self.scale), par,
                                     radii, name, self.radius,
                                     self.resolution)
                else:
                    if self.mode == 'COLONIZE':
                        A = g3.envelope_points(
                            self.envelope, n=self.attractors,
                            seed=self.seed, base=self.trunk)
                        nodes, parents = g3.colonize(
                            A, step=0.07, influence=self.influence,
                            kill=self.kill, gravity=self.gravity)
                    else:
                        nodes, parents = g3.selforg(
                            steps=self.steps, lam=self.lam,
                            angle=self.branch_angle, seed=self.seed)
                    if len(nodes) < 2:
                        self.report({'ERROR'}, "produced no structure")
                        return {'CANCELLED'}
                    # Fit on the FINAL structure, then stage.  Fitting
                    # each stage independently renormalises it, so a
                    # sweep of the slider makes the tree breathe in and
                    # out at constant size instead of growing -- which
                    # defeats the only reason the slider exists.
                    full = g3.fit(nodes, 2.0 * self.scale)
                    if self.grow_t < 1.0:
                        full, _alive = timed.stage(
                            full, parents, self.grow_t * _span(parents))
                    radii = g3.pipe_radii(parents, exponent=self.murray)
                    ob = _tree_curve(full, parents, radii, name,
                                     self.radius, self.resolution)
            except (ValueError, KeyError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

            context.collection.objects.link(ob)
            for o in context.selected_objects:
                o.select_set(False)
            ob.select_set(True)
            context.view_layer.objects.active = ob
            n = (len(ob.data.vertices) if ob.type == 'MESH'
                 else len(ob.data.splines))
            self.report({'INFO'}, f"{name}: {n} elements")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "mode")
            if self.mode == 'COLONIZE':
                for k in ("envelope", "attractors", "trunk", "influence",
                          "kill", "gravity"):
                    lay.prop(self, k)
            elif self.mode == 'SELFORG':
                for k in ("steps", "lam", "branch_angle"):
                    lay.prop(self, k)
                box = lay.box()
                box.label(text="excurrent (conifer)" if self.lam > 0.55
                          else ("decurrent (spreading)" if self.lam < 0.45
                                else "intermediate"), icon='INFO')
            elif self.mode == 'DLA':
                for k in ("walkers", "dla_dim", "seed_kind", "stickiness",
                          "lattice"):
                    lay.prop(self, k)
            else:
                for k in ("depth", "alpha", "roll", "jitter"):
                    lay.prop(self, k)
            lay.separator()
            lay.prop(self, "seed")
            if self.mode in ('COLONIZE', 'SELFORG', 'DLA'):
                for k in ("radius", "resolution", "murray"):
                    lay.prop(self, k)
            if self.mode in ('COLONIZE', 'SELFORG'):
                lay.prop(self, "grow_t")
            lay.prop(self, "scale")

    def register():
        bpy.utils.register_class(CURVE_OT_growth_add)

    def unregister():
        bpy.utils.unregister_class(CURVE_OT_growth_add)


def _span(parents):
    """Total developmental time of a structure, for the growth slider."""
    b = timed.birth_times(parents)
    return float(b.max()) + 1.0


def _selftest():
    # Every mode the enum offers must build something finite.
    A = g3.envelope_points("CROWN", n=400, seed=0, base=0.5)
    nodes, parents = g3.colonize(A, influence=0.75, max_iters=150)
    assert len(nodes) > 10 and len(parents) == len(nodes)

    nodes2, parents2 = g3.selforg(steps=8, seed=0)
    assert len(nodes2) > 3

    pts, dpar = g3.dla(count=200, dim=3, size=30, seed=0)
    assert len(pts) > 10 and pts.shape[1] == 3
    # the aggregate skins through the same path as the trees
    assert len(g3.pipe_radii(dpar)) == len(pts)

    quads, depths = g3.pythagoras(depth=6)
    assert len(quads) == 2 ** 7 - 1

    # The growth slider must be usable across its whole range, and the
    # structure must only ever get bigger as it advances.  Staging is
    # applied AFTER the fit, so the extent really is comparable between
    # stages -- staging a re-fitted structure would renormalise every
    # frame and the tree would breathe rather than grow.
    nodes = g3.fit(nodes, 2.0)
    span = _span(parents)
    assert span > 1.0
    ext = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        P, _a = timed.stage(nodes, parents, f * span)
        assert P.shape == nodes.shape, f
        assert np.all(np.isfinite(P)), f
        ext.append(float(np.linalg.norm(P - P[0], axis=1).max()))
    assert all(b >= a - 1e-9 for a, b in zip(ext, ext[1:])), ext
    assert ext[0] < 1e-9 and ext[-1] > 0.1, ext

    # pipe radii must be defined for whatever the tree modes produce
    for par in (parents, parents2):
        r = g3.pipe_radii(par)
        assert len(r) == len(par) and r.min() > 0.0
        assert r[0] == r.max()

    print(f"growth_generator: OK -- all four modes build, growth slider "
          f"spans {span:.1f} plastochrons monotonically from nothing to "
          f"{ext[-1]:.2f}")
