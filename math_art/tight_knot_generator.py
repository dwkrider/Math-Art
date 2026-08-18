# Tight Knot Generator for Blender
#
# Tight (ideal-style) knots and untangling by minimizing the
# tangent-point energy under length and barycenter constraints, with a
# fractional Sobolev (H^s) preconditioner -- the `knots.tangent_point`
# engine.  The energy blows up as the rope approaches self-contact, so
# the flow cannot change the knot type: the braid-table seed relaxes to
# a canonical tight shape (and a tangled unknot -- e.g. the braid
# abaBcBC -- untangles to a round circle).  This is the
# topology-guaranteed upgrade of the Prime Knot generator's smoothing +
# repulsion relaxer, which remains available as the fast preview.
#
# The rope radius can be set automatically from the measured
# Gonzalez-Maddocks thickness, so the swept tube is exactly the
# maximal-thickness rope of the tight shape.
#
# Links (multi-component curves) get their own operator, Tight Link:
# the same energy's inter-component terms keep components from passing
# through each other, each component keeps its own length, and the
# pairwise linking numbers are asserted unchanged.  Above 400 total
# samples the flow switches to the lagged-factorization solver
# (measured against the exact dense path; see knots/tangent_point.py).
#
# References:
# - G. Buck and J. Orloff, "A simple energy function for knots",
#   Topology Appl. 61 (1995) -- the tangent-point energy.
# - C. Yu, H. Schumacher, K. Crane, "Repulsive Curves", ACM Trans.
#   Graph. 40(2) (2021) -- the fractional Sobolev-Slobodeckij
#   preconditioned flow implemented in `knots/tangent_point.py`.
# - O. Gonzalez and J. H. Maddocks, "Global curvature, thickness, and
#   the ideal shapes of knots", PNAS 96 (1999) -- thickness/ropelength.
# - Thomas A. Gittings, "Minimum braids: a complete invariant of knots
#   and links", arXiv:math/0401051, 2004 (Table 1 braid words).
# - J. Cantarella, R. B. Kusner, J. M. Sullivan, "On the minimum
#   ropelength of knots and links", Invent. Math. 150 (2002) -- the
#   tight Hopf link and chain geometry the link presets relax toward.

bl_info = {
    "name": "Tight Knots",
    "author": "Math Art project (braid table after Thomas Gittings)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Tight Knot",
    "description": "Tight knots and untangling via the tangent-point "
                   "energy with Sobolev preconditioning",
    "category": "Add Curve",
}

import numpy as np

from .knots import (KNOTS, alexander_link_from_curves,
                    braid_closure_loops, braid_closure_points,
                    closed_tube, closure_components,
                    gm_ropelength, gm_ropelength_link, gm_thickness,
                    gm_thickness_link, linking_matrix, parse_letters,
                    resample_closed, resample_loops, tighten,
                    tighten_link)


def build_tight_knot(braid, samples=140, iters=60, mirror=False,
                     scale=1.0):
    """Seed the braid closure, run the tangent-point flow, and
    normalize the result to fit the standard 2 m cube (centered at the
    origin, max coordinate 1) times `scale`.

    Returns (P, info) with info the flow record plus the thickness and
    ropelength of the NORMALIZED curve."""
    word = parse_letters(braid)
    if closure_components(word) != 1:
        raise ValueError(f"braid {braid!r} closes to a link, not a "
                         "knot")
    P = resample_closed(braid_closure_points(word), samples)
    if mirror:
        P = P * np.array([1.0, 1.0, -1.0])
    if samples > 400:
        # beyond the dense operator ceiling: the lagged-factorization
        # solver (same math, factorization reused across iterations;
        # measured ~2x per iteration, converged energy equal to the
        # dense path to ~1e-6 -- see the tp_scale bench case).  At or
        # below 400 the exact dense path runs, byte-identical to the
        # original operator behaviour.
        comps, info = tighten_link([P], iters=iters, solver="lagged")
        P = comps[0]
    else:
        P, info = tighten(P, iters=iters)
    P = P - P.mean(axis=0)
    m = float(np.abs(P).max())
    if m > 1e-12:
        P = P * (scale / m)
    info = dict(info)
    info["thickness"] = gm_thickness(P)
    info["ropelength"] = gm_ropelength(P)
    return P, info


# ---- links -------------------------------------------------------------

# preset -> braid word (multi-component closures); chains are built
# geometrically (circles in alternating perpendicular planes)
LINK_PRESETS = {
    'HOPF': 'aa',
    'TORUS24': 'aaaa',
    'WHITEHEAD': 'aBaBa',
    'BORROMEAN': 'aBaBaB',
}


def _chain_circles(k, spacing=1.4, samples=80):
    """A chain of k unit circles in alternating perpendicular planes,
    centers spaced along x: adjacent pairs are Hopf-linked (lk = +-1),
    non-adjacent pairs unlinked."""
    t = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    c, s = np.cos(t), np.sin(t)
    z = np.zeros_like(t)
    comps = []
    for i in range(int(k)):
        if i % 2 == 0:
            Q = np.stack([i * spacing + c, s, z], axis=1)
        else:
            Q = np.stack([i * spacing + c, z, s], axis=1)
        comps.append(Q)
    return comps


def build_tight_link(seed, samples=80, iters=100, solver='AUTO',
                     scale=1.0, chain_length=3):
    """Seed a link (preset braid word, custom braid word, or geometric
    chain), run the multi-component tangent-point flow, and normalize
    the whole link to the standard 2 m cube times `scale`.

    `solver='AUTO'` uses the exact dense path up to 400 total samples
    and the lagged-factorization path above (measured to agree with
    dense on converged energy and to preserve topology; see
    knots.tangent_point).  Returns (comps, info); info records the
    linking matrix before/after (they must be equal -- checked here),
    the link thickness/ropelength of the normalized link, and the flow
    record."""
    if seed == 'CHAIN':
        comps = _chain_circles(chain_length, samples=samples)
    else:
        word = parse_letters(LINK_PRESETS.get(seed, seed))
        ncomp = closure_components(word)
        if ncomp < 2:
            raise ValueError(f"braid {seed!r} closes to a knot, not a "
                             "link -- use Tight Knot for it")
        comps = resample_loops(braid_closure_loops(word), samples)
    total = sum(len(Q) for Q in comps)
    if total > 1600:
        raise ValueError(f"{total} total samples is beyond the "
                         "supported range (1600)")
    if solver == 'AUTO':
        solver_name = 'dense' if total <= 400 else 'lagged'
    else:
        solver_name = solver.lower()
    lk0, _dev0 = linking_matrix(comps)
    comps, info = tighten_link(comps, iters=iters, solver=solver_name)
    lk1, _dev1 = linking_matrix(comps)
    if not np.array_equal(lk0, lk1):
        raise RuntimeError("linking matrix changed during the flow -- "
                           "this should be impossible (tunneling cap)")
    X = np.concatenate(comps, axis=0)
    ctr = X.mean(axis=0)
    comps = [Q - ctr for Q in comps]
    m = max(float(np.abs(Q).max()) for Q in comps)
    if m > 1e-12:
        comps = [Q * (scale / m) for Q in comps]
    info = dict(info)
    info["solver"] = solver_name
    info["linking_matrix"] = lk1.tolist()
    info["thickness"] = gm_thickness_link(comps)
    info["ropelength"] = gm_ropelength_link(comps)
    return comps, info


# ---- Blender layer -----------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           StringProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _ITEMS = [('CUSTOM', "Custom braid",
               "Closure of the braid word entered below (the default "
               "abaBcBC is a 7-crossing tangled UNKNOT: watch it "
               "untangle to a circle)")]
    _ITEMS += [(name, name.replace('_', '.'),
                f"{name.split('_')[0]} crossings, minimum braid "
                f"{braid}")
               for (name, braid, _ap) in KNOTS]
    _BRAIDS = {name: braid for (name, braid, _ap) in KNOTS}

    class CURVE_OT_tight_knot_add(bpy.types.Operator):
        """Add a tight knot: the braid-table seed relaxed under the
        self-avoiding tangent-point energy (knot type provably kept)"""
        bl_idname = "curve.tight_knot_add"
        bl_label = "Tight Knot"
        bl_options = {'REGISTER', 'UNDO'}

        knot: EnumProperty(name="Knot", items=_ITEMS, default='3_1')
        braid: StringProperty(
            name="Braid Word", default="abaBcBC",
            description="Letters a..z are braid generators, A..Z "
                        "their inverses (used for Custom)")
        samples: IntProperty(
            name="Curve Samples", default=140, min=48, max=800,
            description="Polyline resolution; up to 400 the exact "
                        "dense solver runs, above that the "
                        "accelerated (lagged-factorization) solver")
        iters: IntProperty(
            name="Tighten Iterations", default=60, min=0, max=500,
            description="Tangent-point flow steps (the flow usually "
                        "converges in a few tens; 0 shows the seed)")
        mirror: BoolProperty(
            name="Mirror", default=False,
            description="Mirror image of the knot")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='BEZIER')
        auto_radius: BoolProperty(
            name="Radius From Thickness", default=True,
            description="Set the rope radius to the measured "
                        "Gonzalez-Maddocks thickness of the tight "
                        "shape (the maximal embedded tube)")
        radius: FloatProperty(
            name="Tube Radius", default=0.08, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius (when not "
                        "taken from the thickness)")
        resolution: IntProperty(name="Bevel Resolution", default=6,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            braid = (self.braid if self.knot == 'CUSTOM'
                     else _BRAIDS[self.knot])
            try:
                P, info = build_tight_knot(
                    braid, self.samples, self.iters,
                    mirror=self.mirror, scale=self.scale)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            radius = (0.98 * info["thickness"] if self.auto_radius
                      else self.radius)
            name = ("Tight Knot " + self.knot.replace('_', '.')
                    if self.knot != 'CUSTOM' else "Tight Knot custom")
            if self.output == 'MESH':
                verts, faces = closed_tube(P, radius,
                                           self.tube_sides)
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                if self.output == 'BEZIER':
                    sp = cu.splines.new('BEZIER')
                    sp.bezier_points.add(len(P) - 1)
                    for i, p in enumerate(P):
                        bp = sp.bezier_points[i]
                        bp.co = p
                        bp.handle_left_type = 'AUTO'
                        bp.handle_right_type = 'AUTO'
                else:
                    sp = cu.splines.new(self.output)
                    sp.points.add(len(P) - 1)
                    for i, p in enumerate(P):
                        sp.points[i].co = (p[0], p[1], p[2], 1.0)
                    if self.output == 'NURBS':
                        sp.order_u = 4
                sp.use_cyclic_u = True
                cu.bevel_depth = radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"{name}: braid {braid}, E {info['E0']:.1f} -> "
                f"{info['E']:.1f} in {info['iters_run']} steps, "
                f"ropelength {info['ropelength']:.2f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'knot')
            if self.knot == 'CUSTOM':
                lay.prop(self, 'braid')
            for k in ('samples', 'iters', 'mirror', 'output',
                      'auto_radius'):
                lay.prop(self, k)
            if not self.auto_radius:
                lay.prop(self, 'radius')
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            else:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    class CURVE_OT_tight_link_add(bpy.types.Operator):
        """Add a tight link: several closed components relaxed under
        the multi-component tangent-point energy (linking numbers
        provably kept; components cannot pass through each other)"""
        bl_idname = "curve.tight_link_add"
        bl_label = "Tight Link"
        bl_options = {'REGISTER', 'UNDO'}

        link: EnumProperty(
            name="Link",
            items=[('HOPF', "Hopf Link",
                    "Two circles, linking number 1 -- relaxes to two "
                    "round circles in perpendicular planes"),
                   ('TORUS24', "Torus Link (2,4)",
                    "Two components, linking number 2"),
                   ('WHITEHEAD', "Whitehead Link",
                    "Linking number 0 yet inseparable"),
                   ('BORROMEAN', "Borromean Rings",
                    "Three rings, no two linked, yet inseparable"),
                   ('CHAIN', "Chain",
                    "A chain of Hopf-linked rings"),
                   ('CUSTOM', "Custom Braid",
                    "Closure of the braid word entered below (must "
                    "close to 2+ components)")],
            default='HOPF')
        braid: StringProperty(
            name="Braid Word", default="aa",
            description="Letters a..z are braid generators, A..Z "
                        "their inverses (used for Custom; must close "
                        "to a link)")
        chain_length: IntProperty(
            name="Chain Rings", default=3, min=2, max=8,
            description="Number of rings in the chain")
        samples: IntProperty(
            name="Samples Per Component", default=80, min=32, max=400,
            description="Polyline resolution per component (larger "
                        "links switch to the accelerated solver "
                        "automatically)")
        iters: IntProperty(
            name="Tighten Iterations", default=100, min=0, max=500,
            description="Tangent-point flow steps (links typically "
                        "converge in a few tens; 0 shows the seed)")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='BEZIER')
        auto_radius: BoolProperty(
            name="Radius From Thickness", default=True,
            description="Set the rope radius to the measured "
                        "Gonzalez-Maddocks thickness of the tight "
                        "link (the maximal embedded tube)")
        radius: FloatProperty(
            name="Tube Radius", default=0.08, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius (when not "
                        "taken from the thickness)")
        resolution: IntProperty(name="Bevel Resolution", default=6,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            seed = (self.braid if self.link == 'CUSTOM' else self.link)
            try:
                comps, info = build_tight_link(
                    seed, self.samples, self.iters,
                    scale=self.scale, chain_length=self.chain_length)
            except (ValueError, KeyError, RuntimeError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            radius = (0.98 * info["thickness"] if self.auto_radius
                      else self.radius)
            name = ("Tight Link " + self.link.capitalize()
                    if self.link != 'CUSTOM' else "Tight Link custom")
            if self.output == 'MESH':
                me = bpy.data.meshes.new(name)
                verts = []
                faces = []
                for Q in comps:
                    v, f = closed_tube(Q, radius, self.tube_sides)
                    base = len(verts)
                    verts.extend(v)
                    faces.extend([[i + base for i in face]
                                  for face in f])
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                for Q in comps:
                    if self.output == 'BEZIER':
                        sp = cu.splines.new('BEZIER')
                        sp.bezier_points.add(len(Q) - 1)
                        for i, p in enumerate(Q):
                            bp = sp.bezier_points[i]
                            bp.co = p
                            bp.handle_left_type = 'AUTO'
                            bp.handle_right_type = 'AUTO'
                    else:
                        sp = cu.splines.new(self.output)
                        sp.points.add(len(Q) - 1)
                        for i, p in enumerate(Q):
                            sp.points[i].co = (p[0], p[1], p[2], 1.0)
                        if self.output == 'NURBS':
                            sp.order_u = 4
                    sp.use_cyclic_u = True
                cu.bevel_depth = radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            lk = np.asarray(info["linking_matrix"])
            iu = np.triu_indices(len(comps), 1)
            self.report(
                {'INFO'},
                f"{name}: {len(comps)} components, lk "
                f"{lk[iu].tolist()} (kept), E {info['E0']:.1f} -> "
                f"{info['E']:.1f} in {info['iters_run']} steps "
                f"({info['solver']}), ropelength "
                f"{info['ropelength']:.2f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'link')
            if self.link == 'CUSTOM':
                lay.prop(self, 'braid')
            if self.link == 'CHAIN':
                lay.prop(self, 'chain_length')
            for k in ('samples', 'iters', 'output', 'auto_radius'):
                lay.prop(self, k)
            if not self.auto_radius:
                lay.prop(self, 'radius')
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            else:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.tight_knot_add",
                             icon='FORCE_VORTEX')

    def _menu_func_link(self, context):
        self.layout.operator("curve.tight_link_add",
                             icon='LINKED')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_tight_knot_add)
        bpy.utils.register_class(CURVE_OT_tight_link_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func_link)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func_link)
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_tight_link_add)
        bpy.utils.unregister_class(CURVE_OT_tight_knot_add)


def _selftest():
    ok = True
    from .knots.alexander import alexander_from_curve

    # The pipeline must keep the knot type (Alexander gate), reduce the
    # energy, and emit a curve normalized into the 2 m cube.
    P, info = build_tight_knot('AAA', samples=96, iters=15)
    alex = alexander_from_curve(P)
    good = (alex == 91 and info["E"] < info["E0"]
            and abs(float(np.abs(P).max()) - 1.0) < 1e-9
            and float(np.linalg.norm(P.mean(axis=0))) < 1e-9
            and info["thickness"] > 0.0)
    ok &= good
    print(f"tight_knot: trefoil alex={alex} (exp 91), E "
          f"{info['E0']:.1f}->{info['E']:.1f}, thickness "
          f"{info['thickness']:.3f} {'OK' if good else 'FAIL'}")

    # A link-closing braid must be rejected, not silently emitted.
    try:
        build_tight_knot('aa', samples=64, iters=0)
        good = False
    except ValueError:
        good = True
    ok &= good
    print(f"tight_knot: link braid rejected {'OK' if good else 'FAIL'}")

    # The link pipeline: Hopf preset keeps its linking matrix, lands
    # normalized in the cube, and reduces the energy.
    comps, li = build_tight_link('HOPF', samples=48, iters=8)
    lk = np.asarray(li["linking_matrix"])
    X = np.concatenate(comps, axis=0)
    good = (len(comps) == 2 and abs(int(lk[0, 1])) == 1
            and li["E"] < li["E0"]
            and abs(float(np.abs(X).max()) - 1.0) < 1e-9
            and li["thickness"] > 0.0
            and li["inter_gap_min"] > 0.0)
    ok &= good
    print(f"tight_link: Hopf lk={int(lk[0, 1]):+d} kept, E "
          f"{li['E0']:.1f}->{li['E']:.1f}, thickness "
          f"{li['thickness']:.3f} {'OK' if good else 'FAIL'}")

    # A chain seed has the chain linking pattern (adjacent +-1, ends
    # 0), and a knot-closing braid is rejected by the link builder.
    comps, lc = build_tight_link('CHAIN', samples=40, iters=4,
                                 chain_length=3)
    lkc = np.asarray(lc["linking_matrix"])
    good = (len(comps) == 3 and abs(int(lkc[0, 1])) == 1
            and abs(int(lkc[1, 2])) == 1 and int(lkc[0, 2]) == 0)
    try:
        build_tight_link('AAA', samples=48, iters=0)
        good = False
    except ValueError:
        pass
    ok &= good
    print(f"tight_link: chain lk pattern {lkc[np.triu_indices(3, 1)].tolist()}"
          f", knot braid rejected {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("tight_knot self-test failed")
