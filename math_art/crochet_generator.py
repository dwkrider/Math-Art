
# Crochet (Hyperbolic Plane) Generator for Blender
#
# The crocheted hyperbolic plane, after Daina Taimina: crochet
# increases stitches at a constant ratio (one increase every N
# stitches), so each row's circumference grows exponentially and the
# fabric -- unable to lie flat (Hilbert's theorem) -- buckles into
# self-similar ruffles. The increase ratio N sets the curvature radius
# R = h / ln(1 + 1/N) ~ N*h: small N crochets a tightly folded, very
# "bendy" plane; large N a gently wavy one.
#
# The mesh is the actual crochet: each row's stitch count grows with
# the hyperbolic circumference 2*pi*R*sinh(rho/R), every stitch the
# same size. Each ring is pre-seeded with a phase-coherent period-
# doubling cascade of ruffles (wavenumbers m0, 2*m0, 4*m0, ... switched
# on as the excess length grows) whose doubling radii are the surface's
# "distributed branch points" -- the source of the self-similar,
# kale/lettuce morphology. The sheet is then relaxed (position-based
# stitch-length + bending constraints, self-repulsion, and a fast
# BVHTree vertex-triangle self-collision) into its buckled shape,
# centred and fit to a 2 m cube.
#
# References:
# - Daina Taimina, "Crocheting Adventures with Hyperbolic Planes",
#   A K Peters, 2009; David W. Henderson and Daina Taimina,
#   "Crocheting the hyperbolic plane", Math. Intelligencer 23(2),
#   2001, pp. 17-28.
# - Self-similar buckling / distributed branch points of constant-
#   negative-curvature elastic sheets: John A. Gemmer, E. Sharon,
#   S. C. Venkataramani et al., "Distributed branch points and the
#   shape of elastic surfaces with constant negative curvature",
#   arXiv:2006.14461.
# - Hilbert's theorem (no complete C^2 isometric immersion of H^2 in
#   R^3): D. Hilbert, Trans. AMS 2, 1901, pp. 87-99.

bl_info = {
    "name": "Crochet/Coral",
    "author": "Math Art project",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Crochet/Coral",
    "description": "Ruffled crocheted hyperbolic planes, from gently "
                   "wavy to tightly folded; graded (cristate) and "
                   "multi-lobed forms",
    "category": "Add Mesh",
}
























# ==========================================================================
# Blender layer
# ==========================================================================

import math
import numpy as np

# The mathematics lives in the sibling `hyperbolic` engine package;
# this module is the Blender layer over it.
try:
    from .hyperbolic.crochet import (PRESETS, _bvh_decollide, _center,
                                         _crochet_mesh, _relax, _repel,
                                         crochet_c2f, mean_curvature)
except ImportError:  # flat import outside the package
    from hyperbolic.crochet import (PRESETS, _bvh_decollide, _center,
                                        _crochet_mesh, _relax, _repel,
                                        crochet_c2f, mean_curvature)


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# These stay in the Blender layer: both branch on `_IN_BLENDER`,
# and a function that asks whether Blender is present is
# presentation rather than mathematics.
def _pack(P, E0, E1, REST, nbr, faces, pin, pin_pos, iters, pull,
          stitch, repel_r, smooth=0.12):
    """Collision-packing: pull every free vertex weakly toward the
    centroid (a self-weight/gravity that compacts the sheet) while the
    stitch springs preserve the metric and BVHTree self-collision stops
    the folds passing through each other -- so the smooth hyperbolic
    disk folds up into a dense ball, like real crochet gathered in the
    hand. Blender only (needs the BVHTree collision)."""
    n = len(P)
    valence = np.zeros(n)
    np.add.at(valence, E0, 1.0)
    np.add.at(valence, E1, 1.0)
    valence = np.maximum(valence, 1.0)[:, None]
    free = np.ones(n, dtype=bool)
    free[pin] = False
    for it in range(iters):
        C = P[free].mean(axis=0)
        P[free] += pull * (C - P[free])           # gravity toward centre
        for _ in range(3):                        # keep stitch lengths
            d = P[E1] - P[E0]
            L = np.linalg.norm(d, axis=1)
            L = np.where(L < 1e-9, 1e-9, L)
            corr = np.clip(1.0 - REST / L, -3.0, 1.0)[:, None] * d
            dP = np.zeros_like(P)
            np.add.at(dP, E0, corr)
            np.add.at(dP, E1, -corr)
            P = P + dP / valence
            P[pin] = pin_pos
        if smooth > 0.0:                           # keep the folds smooth
            nsum = np.zeros_like(P)
            np.add.at(nsum, E0, P[E1])
            np.add.at(nsum, E1, P[E0])
            P = P + smooth * (nsum / valence - P)
        P = P + _repel(P, nbr, repel_r, 0.5)
        if _IN_BLENDER:                            # self-collision each step
            P = _bvh_decollide(P, faces, nbr, 0.9 * stitch, 0.8)
        P[pin] = pin_pos
    # final polish: smooth out collision roughness while holding the
    # stitch lengths, so the ball reads as folded fabric not crumple
    for _ in range(8):
        nsum = np.zeros_like(P)
        np.add.at(nsum, E0, P[E1])
        np.add.at(nsum, E1, P[E0])
        P = P + 0.25 * (nsum / valence - P)
        d = P[E1] - P[E0]
        L = np.linalg.norm(d, axis=1)
        L = np.where(L < 1e-9, 1e-9, L)
        corr = np.clip(1.0 - REST / L, -3.0, 1.0)[:, None] * d
        dP = np.zeros_like(P)
        np.add.at(dP, E0, corr)
        np.add.at(dP, E1, -corr)
        P = P + dP / valence
        P[pin] = pin_pos
    return P

def build_ruffle(ratio_n=4, rows=18, stitch=0.09, max_stitches=600,
                 iters=340, smooth=0.06, repel=0.5, collide=3,
                 anneal=False, stiff=1.0, pack=0, pack_pull=0.03,
                 grade=0.0, lobes=3, refine_levels=0, bend=0.0):
    # `anneal`/`stiff` grow the curvature smoothly from flat and damp
    # the springs (needed for the tight-curvature presets to stay
    # smooth); `pack` folds the relaxed sheet into a ball via gravity +
    # self-collision. `refine_levels` > 0 switches to the coarse-to-fine
    # continuation (relax coarse, subdivide, re-relax -- larger,
    # smoother ruffles at less cost) and `bend` adds thin-plate
    # bending resistance against short-wavelength crumple. Defaults
    # (all off) reproduce the plain build, so Wavy/Ruffled are
    # unchanged.
    ce = max(15, iters // (collide * 4 + 1)) if collide > 0 else 10 ** 9
    if refine_levels > 0:
        cff = None
        if collide > 0 and _IN_BLENDER:
            def cff(fcs, nb, h):
                return lambda pp: _bvh_decollide(pp, fcs, nb,
                                                 0.75 * h, 0.6)
        res = crochet_c2f(ratio_n=ratio_n, rows=rows, stitch=stitch,
                          max_stitches=max_stitches, iters=iters,
                          smooth=smooth, repel=repel, bend=bend,
                          levels=refine_levels,
                          seed_scale=0.2 if anneal else 1.0,
                          grade=grade, lobes=lobes, anneal=anneal,
                          stiff=stiff, collide_fn_factory=cff,
                          collide_every=ce)
        P, faces = res["P"], res["tris"]
        if pack > 0:
            P = _pack(P, res["E0"], res["E1"], res["REST"], res["nbr"],
                      faces, res["pin"], P[res["pin"]].copy(), pack,
                      pack_pull, stitch, 0.4 * stitch)
        return _center(P), faces
    P, E0, E1, REST, nbr, faces, pin, pin_pos = _crochet_mesh(
        ratio_n, rows, stitch, max_stitches,
        seed_scale=0.2 if anneal else 1.0, grade=grade, lobes=lobes)
    L0 = None
    if anneal:
        flat = P.copy()
        flat[:, 2] = 0.0
        L0 = np.linalg.norm(flat[E1] - flat[E0], axis=1)
    cf = None
    if collide > 0 and _IN_BLENDER:
        cf = lambda pp: _bvh_decollide(pp, faces, nbr,
                                       0.75 * stitch, 0.6)
    P = _relax(P, E0, E1, REST, nbr, pin, pin_pos, iters, smooth,
               0.4 * stitch, repel, collide_fn=cf, collide_every=ce,
               anneal_L0=L0, stiff=stiff, bend=bend)
    if pack > 0:
        P = _pack(P, E0, E1, REST, nbr, faces, pin, pin_pos, pack,
                  pack_pull, stitch, 0.4 * stitch)
    return _center(P), faces


if _IN_BLENDER:

    class MESH_OT_crochet_add(bpy.types.Operator):
        """Add a crocheted hyperbolic plane -- a ruffled, negatively
        curved surface; graded for crested/cristate coral forms"""
        bl_idname = "mesh.crochet_add"
        bl_label = "Crochet/Coral"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            items=[('WAVY', "Wavy", "Gently wavy (loose curvature)"),
                   ('RUFFLED', "Ruffled", "Clearly ruffled"),
                   ('BENDY', "Bendy", "Tightly folded, very 3D "
                    "(slower)"),
                   ('TAIMINA', "Taimina (ball)", "Densely folded, "
                    "ball-like (slowest)"),
                   ('CUSTOM', "Custom", "Use the parameters below")],
            default='WAVY')
        # Custom defaults mirror the WAVY preset
        ratio_n: IntProperty(
            name="Increase Ratio N", default=6, min=2, max=24,
            description="Increase 1 stitch every N. Small N = tight "
                        "curvature, more bend")
        rows: IntProperty(
            name="Rows", default=14, min=3, max=30,
            description="Crochet rows; more = larger and more folded")
        stitch: FloatProperty(
            name="Stitch Size", default=0.10, min=0.02, max=0.5,
            description="Size of one (square) stitch")
        max_stitches: IntProperty(
            name="Max Stitches/Row", default=400, min=24, max=2000,
            description="Cap on the (growing) stitch count per row; "
                        "raise it at small N so the edge keeps ruffling")
        iters: IntProperty(
            name="Relax Steps", default=280, min=0, max=3000,
            description="Buckling relaxation iterations")
        smooth: FloatProperty(
            name="Bending", default=0.08, min=0.0, max=1.0,
            description="Higher = larger, smoother ruffles; lower = "
                        "finer, crisper folds. Above ~0.5 it strongly "
                        "over-smooths and can shrink the sheet")
        repel: FloatProperty(
            name="Self-Repulsion", default=0.5, min=0.0, max=1.0,
            description="Push apart ruffles that get too close")
        collide: IntProperty(
            name="Collision Passes", default=2, min=0, max=10,
            description="BVHTree vertex-triangle self-collision passes "
                        "that stop the fabric passing through itself")
        physics: EnumProperty(
            name="Fold By",
            items=[('PBD', "Built-in Packing",
                    "Fold with the internal collision-packing"),
                   ('CLOTH', "Blender Cloth",
                    "Build the flat sheet and fold it with a Cloth "
                    "modifier (self-collision + a gather field)")],
            default='PBD')
        cloth_bake: IntProperty(
            name="Cloth Bake Frames", default=45, min=0, max=400,
            description="Cloth mode: frames to auto-simulate and "
                        "freeze. 0 = just set up the modifier and let "
                        "you press Play")
        cloth_gather: FloatProperty(
            name="Gather Strength", default=170.0, min=0.0, max=2000.0,
            description="Cloth mode: inward force that gathers the "
                        "sheet into a ball. Lower = looser, less "
                        "tightly folded (0 = drape under gravity)")
        cloth_prep: IntProperty(
            name="Sheet Prep Steps", default=40, min=0, max=400,
            description="Cloth mode: relaxation steps to tidy the "
                        "procedurally-ruffled sheet before handing it "
                        "to cloth (the cloth solver does the rest, so "
                        "this can be small or 0)")
        cloth_bend: IntProperty(
            name="Fold Softness", default=12, min=0, max=60,
            description="Cloth mode: smoothing passes applied after "
                        "folding, to round the sharp creases. Higher = "
                        "softer, rounder folds; 0 = raw cloth")
        cloth_thick: FloatProperty(
            name="Fabric Thickness", default=0.7, min=0.2, max=2.0,
            description="Cloth mode: self-collision distance as a "
                        "fraction of stitch size; higher keeps folds "
                        "from pinching into sharp seams")
        cloth_detail: IntProperty(
            name="Cloth Detail", default=1500, min=300, max=12000,
            description="Cloth mode: target vertex count for the SIM "
                        "(the sheet is decimated to this before "
                        "folding). Self-collision cost grows fast with "
                        "vertex count, so lower = much faster; the "
                        "fold smoothing hides the coarseness")
        grade: FloatProperty(
            name="Cristate Grade", default=0.0, min=-0.9, max=2.0,
            description="Grade the curvature across the sheet: > 0 = a "
                        "flatter centre with the ruffling concentrated "
                        "at the rim (a crested / cristate form, as in a "
                        "crested cactus or brain coral); < 0 = ruffled "
                        "centre, calmer rim. 0 = uniform curvature")
        lobes: IntProperty(
            name="Lobes", default=3, min=2, max=12,
            description="Coarsest ruffle count: the rim breaks into this "
                        "many primary lobes, for a multi-lobed "
                        "hyperbolic form")
        refine_levels: IntProperty(
            name="Coarse-to-Fine", default=0, min=0, max=3,
            description="Continuation levels: relax a coarse sheet "
                        "first (large waves form there), then "
                        "subdivide and re-relax. Larger, smoother "
                        "ruffles at less cost than one-shot; 0 = off "
                        "(classic single-resolution relax). 1 with "
                        "Stiffness ~0.03 is the measured sweet spot; "
                        "2 makes still larger waves but wants a "
                        "Collision pass to keep the deep folds apart")
        bend: FloatProperty(
            name="Stiffness", default=0.0, min=0.0, max=0.35,
            description="Thin-plate bending resistance: damps "
                        "stitch-scale crumple while leaving the large "
                        "waves intact. Best combined with Coarse-to-"
                        "Fine (~0.02-0.06; measured best 0.03), 0 = "
                        "off")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0)
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            if self.preset == 'CUSTOM':
                p = dict(ratio_n=self.ratio_n, rows=self.rows,
                         stitch=self.stitch,
                         max_stitches=self.max_stitches,
                         iters=self.iters, smooth=self.smooth,
                         repel=self.repel, collide=self.collide)
            else:
                p = dict(PRESETS[self.preset])
            # cristate grading + lobe count + continuation/stiffness
            # apply on top of any preset
            p['grade'] = self.grade
            p['lobes'] = self.lobes
            p['refine_levels'] = self.refine_levels
            p['bend'] = self.bend
            if self.physics == 'CLOTH':
                # hand cloth the procedurally-ruffled flat sheet (full
                # wave seed, no PBD packing/collision) with only a light
                # cleanup relax; the cloth solver refines + folds it.
                # Cap the resolution -- self-collision cost is superlinear
                # in vertex count and the fold smoothing hides coarseness.
                p['pack'] = 0
                p['anneal'] = False
                p['collide'] = 0
                p['iters'] = self.cloth_prep
                p['refine_levels'] = 0    # cloth folds the sheet itself
            verts, faces = build_ruffle(**p)
            verts = np.asarray(verts) * self.scale
            me = bpy.data.meshes.new("Crochet Coral")
            me.from_pydata([tuple(v) for v in verts], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Crochet Coral", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            note = ""
            if self.physics == 'CLOTH':
                note = self._setup_cloth(context, obj, verts, faces)
            self.report({'INFO'},
                        f"Crochet/Coral ({self.preset.title()}): "
                        f"V={len(obj.data.vertices)} "
                        f"F={len(obj.data.polygons)}{note}")
            return {'FINISHED'}

        def _setup_cloth(self, context, obj, verts, faces):
            """Attach a Cloth modifier (self-collision + optional inward
            gather field) that folds the flat sheet into a ball, and
            optionally bake+freeze it."""
            context.view_layer.objects.active = obj
            # decimate to the target sim resolution first -- self-
            # collision cost is superlinear in vertex count
            if len(obj.data.vertices) > self.cloth_detail:
                dm = obj.modifiers.new("PreDecimate", 'DECIMATE')
                dm.ratio = self.cloth_detail / len(obj.data.vertices)
                try:
                    bpy.ops.object.modifier_apply(modifier=dm.name)
                except RuntimeError:
                    pass
            V = np.array([v.co[:] for v in obj.data.vertices])
            faces = [tuple(p.vertices) for p in obj.data.polygons]
            cen = V.mean(axis=0)
            k = max(6, len(V) // 150)
            pin = np.argsort(np.linalg.norm(V[:, :2] - cen[:2],
                                            axis=1))[:k]
            vg = obj.vertex_groups.new(name="crochet_pin")
            vg.add([int(i) for i in pin], 1.0, 'REPLACE')
            med = float(np.median(
                [np.linalg.norm(V[f[0]] - V[f[1]]) for f in faces]))

            mod = obj.modifiers.new("Cloth", 'CLOTH')
            s = mod.settings
            s.quality = 7
            s.mass = 0.3
            s.tension_stiffness = 15.0
            s.compression_stiffness = 15.0
            s.shear_stiffness = 5.0
            try:
                s.bending_model = 'ANGULAR'
            except (AttributeError, TypeError):
                pass
            s.bending_stiffness = 3.0
            s.bending_damping = 2.0
            s.vertex_group_mass = "crochet_pin"     # pin group
            s.pin_stiffness = 5.0
            col = mod.collision_settings
            col.use_self_collision = True
            col.self_distance_min = min(0.1, max(0.003,
                                                 self.cloth_thick * med))
            col.self_impulse_clamp = 0.0
            col.collision_quality = 3

            fld = None
            if self.cloth_gather > 0.0:
                s.effector_weights.gravity = 0.0    # gather, don't droop
                fld = bpy.data.objects.new("Crochet Gather", None)
                context.collection.objects.link(fld)
                fld.location = tuple(float(c) for c in cen)
                context.view_layer.objects.active = fld
                bpy.ops.object.forcefield_toggle()  # enables fld.field
                fld.field.type = 'FORCE'
                fld.field.strength = -abs(self.cloth_gather)
                fld.field.falloff_type = 'SPHERE'
                context.view_layer.objects.active = obj

            # post-fold smoothing rounds the sharp cloth creases
            sm = None
            if self.cloth_bend > 0:
                sm = obj.modifiers.new("Fold Smooth", 'SMOOTH')
                sm.iterations = int(self.cloth_bend)
                sm.factor = 0.6

            if self.cloth_bake > 0:
                scene = context.scene
                start = scene.frame_start
                for fnum in range(start, start + self.cloth_bake):
                    scene.frame_set(fnum)
                context.view_layer.objects.active = obj
                try:
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                    if sm is not None:
                        bpy.ops.object.modifier_apply(modifier=sm.name)
                except RuntimeError:
                    pass
                if fld is not None:
                    bpy.data.objects.remove(fld, do_unlink=True)
                scene.frame_set(start)
                return f" (cloth baked {self.cloth_bake}f)"
            return " (cloth set up -- press Play)"

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                for k in ('ratio_n', 'rows', 'stitch', 'max_stitches',
                          'iters', 'smooth', 'repel', 'collide'):
                    lay.prop(self, k)
            lay.prop(self, 'grade')
            lay.prop(self, 'lobes')
            if self.physics != 'CLOTH':
                lay.prop(self, 'refine_levels')
                lay.prop(self, 'bend')
            lay.prop(self, 'physics')
            if self.physics == 'CLOTH':
                lay.prop(self, 'cloth_detail')
                lay.prop(self, 'cloth_prep')
                lay.prop(self, 'cloth_bake')
                lay.prop(self, 'cloth_gather')
                lay.prop(self, 'cloth_bend')
                lay.prop(self, 'cloth_thick')
            lay.prop(self, 'scale')
            lay.prop(self, 'shade_smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.crochet_add", icon='MOD_CLOTH')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_crochet_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_crochet_add)


def _selftest():
    import time
    h0 = 0.09
    Rc = h0 / math.log(1.0 + 1.0 / 4)
    t0 = time.time()
    V, E0, E1, REST, nbr0, F, pin, pinp = _crochet_mesh(4, 18, h0,
                                                        600)
    V = _relax(V, E0, E1, REST, nbr0, pin, pinp, 340, 0.06,
               0.4 * h0, 0.5)
    dt = time.time() - t0
    strain = float(np.mean(np.linalg.norm(V[E1] - V[E0], axis=1)
                           / REST))
    zext = float(V[:, 2].max() - V[:, 2].min())
    K = mean_curvature(V, F)
    finite = np.isfinite(V).all()
    ok = finite and zext > 0.05 and K < 0
    print(f"RUFFLED: V={len(V)} F={len(F)} z_extent={zext:.3f} "
          f"meanK={K:+.2f} strain={strain:.3f} "
          f"targetK={-1.0 / Rc ** 2:+.2f} time={dt:.1f}s "
          f"{'OK' if ok else 'CHECK'}")
    assert ok
