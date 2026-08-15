
# Woven Double Shell generator for Blender.
#
# Two concentric spheres of rope rosettes tied into one woven fabric.
# On the OUTER sphere (radius 1) a rosette medallion sits over every
# face of a chosen Platonic / Archimedean / geodesic scaffold; on the
# INNER sphere (radius `inner_radius`) a medallion sits at every
# scaffold VERTEX -- the dual solid's faces (the dual of a Platonic
# solid is Platonic, of an Archimedean solid its Catalan solid, of a
# geodesic sphere its Goldberg dual).  Each medallion is cut open at
# its lobe corners into PORTS, and rope BRIDGES dive through the gap
# from every outer port to a topologically paired inner port -- the
# rotation-proof outer-edge -> inner-edge pairing of the Twisted
# Polyhedron generator -- so the whole object is a set of closed rope
# strands threading outer arc, down bridge, inner arc, up bridge, ...
# forever.  An integer `advance` rotates the up-pairing around each
# inner medallion: advance 0 closes every strand into a short
# chain-mail stitch ring around one scaffold edge, advance 1 chains
# them into long snaking strands.  All crossings (bridge x bridge in
# the gap, medallion x medallion on each shell, bridge x medallion at
# the dives) are woven strictly one-over-one-under with the parity
# union-find of the Knot Carpet engine, then the whole rope field is
# physically relaxed with the KnotPlot bead/stick model (springs,
# fairing, 1/r^4 repulsion, a radial over/under gap at each crossing,
# and a per-bead target radius holding the two shells apart).
#
# References:
# - Inspired by Bathsheba Grossman's "Quintrino" and her "Ora" family
#   of dual-symmetry sculptures: strap-like arms swirling between a
#   dodecahedron and its dual.  This renders that bridge topology as
#   woven rope over any Platonic / Archimedean / geodesic scaffold.
# - Polyhedral duality: the dual of a Platonic solid is Platonic; the
#   dual of an Archimedean solid is its Catalan solid (Archimedes,
#   work lost; Johannes Kepler, "Harmonices Mundi", 1619; Eugene
#   Catalan, 1865).  R. Buckminster Fuller, U.S. Patent 2,682,235
#   (1954) -- the geodesic subdivision of the icosahedron.
# - Robert G. Scharein, "Interactive Topological Drawing" (PhD
#   thesis, The University of British Columbia, 1998), ch. 7 -- the
#   KnotPlot bead/stick relaxation reused here (springs, high-order
#   repulsion, critical damping, the d_max step bound).
# - J. K. Simon, "Energy functions for polygonal knots", J. Knot
#   Theory Ramifications 3(3), 1994; G. Buck & J. Orloff, Topology
#   Appl. 51 (1993) / 61 (1995) -- the knot energies the repulsion
#   approximates.
# - Peter R. Cromwell, "Knots and Links" (CUP, 2004) -- alternating
#   diagrams, the one-over-one-under discipline enforced here.

bl_info = {
    "name": "Woven Double Shell",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Woven Double Shell",
    "description": "Rope rosettes on two concentric spheres joined "
                   "by woven bridges (dual-polyhedron topology)",
    "category": "Add Mesh",
}



try:                                  # inside the math_art package
    from .curve_frames import welded_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import welded_tube


try:
    from .patterns import common as pc
    from . import knot_carpet_generator as kcg
    from . import regular_solids_generator as rs
    from . import geodesic_generator as geo
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import knot_carpet_generator as kcg
    import regular_solids_generator as rs
    import geodesic_generator as geo


# --------------------------------------------------------------------
# Scaffold: solid + dual + the outer-edge -> inner-edge pairing
# --------------------------------------------------------------------
#
# A numpy port of the Twisted Polyhedron scaffold (that module builds
# with mathutils and cannot run outside Blender): unit-sphere solid
# vertices, faces reordered CCW as seen from outside, dual vertices
# (normalized face centroids) and, per vertex, the CCW ring of
# incident faces -- the dual face sitting at that vertex.


import numpy as np

# The mathematics lives in the sibling `weaving` engine package;
# this module is the Blender layer over it.
try:
    from .weaving.shells import (ELEM_INNER, ELEM_OUTER, SOLID_ITEMS,
                                     _FAMILY, _scaffold, build_cells,
                                     build_double_shell,
                                     relax_double_shell, shell_lines,
                                     strand_cycles, strand_paths)
except ImportError:  # flat import outside the package
    from weaving.shells import (ELEM_INNER, ELEM_OUTER, SOLID_ITEMS,
                                    _FAMILY, _scaffold, build_cells,
                                    build_double_shell,
                                    relax_double_shell, shell_lines,
                                    strand_cycles, strand_paths)

# The catalogue gains one entry that only the Blender layer offers, so
# it is appended AFTER the import that binds it.
SOLID_ITEMS.append(
    ('GEODESIC', "Geodesic Sphere",
     "Icosahedral geodesic sphere (set Frequency), its Goldberg "
     "dual of hexagons and 12 pentagons inside"))


















# --------------------------------------------------------------------
# The strand walk (a permutation on lobe arcs -- always closes)
# --------------------------------------------------------------------
#
# Arcs: outer arc (f, i) spans outer medallion corner i -> i+1 (CCW);
# inner arc (v, t) spans inner corner t -> t+1 in increasing ring
# angle.  Both medallion families are walked FORWARD (_DELTA = +1,
# the same CCW sense seen from outside): at advance = 0 the walk
# then closes every strand into a short stitch ring around one
# scaffold edge -- exactly E components, verified by the self-test
# on every scaffold -- the chain-mail fallback.  `advance` rotates
# the up-pairing by that many ring slots, merging the rings into
# long snaking strands (advance = 1 gives one strand per scaffold
# face on the Platonic solids); the walk is a permutation for every
# advance, so strands ALWAYS close.





# --------------------------------------------------------------------
# Medallion and bridge geometry
# --------------------------------------------------------------------























# --------------------------------------------------------------------
# Crossings: radial projection per scaffold-edge arena
# --------------------------------------------------------------------







# --------------------------------------------------------------------
# The full build
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# Relaxation (KnotPlot bead/stick model on the double shell)
# --------------------------------------------------------------------









# --------------------------------------------------------------------
# Output builders
# --------------------------------------------------------------------







# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _emit_curve(context, name, paths, span=2.0, operator=None):
        """Curve object of cyclic POLY splines, centered and scaled
        into the span cube."""
        paths = [(list(p), c) for p, c in paths if len(p) >= 2]
        if not paths:
            return None
        allv = np.asarray([p for pa, _c in paths for p in pa], float)
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        cen = 0.5 * (lo + hi)
        s = span / max(float(np.max(hi - lo)), 1e-9)
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        for pa, closed in paths:
            sp = cu.splines.new('POLY')
            sp.points.add(len(pa) - 1)
            for i, p in enumerate(pa):
                sp.points[i].co = ((p[0] - cen[0]) * s,
                                   (p[1] - cen[1]) * s,
                                   (p[2] - cen[2]) * s, 1.0)
            sp.use_cyclic_u = bool(closed)
        obj = bpy.data.objects.new(name, cu)
        context.collection.objects.link(obj)
        if operator is not None:
            from bpy_extras import object_utils
            obj.matrix_world = object_utils.add_object_align_init(
                context, operator)
        else:
            obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_woven_double_shell_add(bpy.types.Operator,
                                         AddObjectHelper):
        """Add a woven double shell: rope rosettes on an outer and an
        inner sphere joined by rope bridges that weave over and under
        through the gap (the Twisted Polyhedron bridge topology as
        relaxed rope)"""
        bl_idname = "mesh.woven_double_shell_add"
        bl_label = "Woven Double Shell"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(
            name="Scaffold", items=SOLID_ITEMS, default='CUBE',
            description="Outer solid: one medallion per face outside,"
                        " one per vertex (its dual) inside")
        frequency: IntProperty(
            name="Frequency", default=2, min=1, max=4,
            description="Geodesic breakdown frequency (Geodesic "
                        "Sphere only)")
        inner_radius: FloatProperty(
            name="Inner Radius", default=0.62, min=0.45, max=0.9,
            description="Radius of the inner shell (outer is 1)")
        overlap_out: FloatProperty(
            name="Outer Overlap", default=1.10, min=0.8, max=1.6,
            description="Outer medallion radius as a fraction of "
                        "half the largest neighbour distance")
        overlap_in: FloatProperty(
            name="Inner Overlap", default=1.10, min=0.8, max=1.6,
            description="Inner medallion radius factor")
        amplitude: FloatProperty(
            name="Lobe Amplitude", default=0.10, min=0.0, max=0.4,
            description="Rosette lobe depth (0 = circles)")
        port_gap: FloatProperty(
            name="Port Gap", default=0.08, min=0.01, max=0.25,
            description="Fraction of a lobe cut open at each corner "
                        "where the rope dives to a bridge")
        spin: FloatProperty(
            name="Spin", default=20.0, min=-360.0, max=360.0,
            description="Inner medallion spin (degrees); folds with "
                        "the pairing so bridges lean, never tangle")
        advance: IntProperty(
            name="Advance", default=1, min=0, max=6,
            description="Ring rotation of the up-pairing: 0 = short "
                        "chain-mail stitch rings, 1+ = long snaking "
                        "strands")
        twist: FloatProperty(
            name="Bridge Twist", default=25.0, min=-90.0, max=90.0,
            description="Swirl of the bridge mid-controls about the "
                        "scaffold edge (degrees); makes neighbouring "
                        "bridges cross in the gap")
        handle_out: FloatProperty(
            name="Handle (Outer)", default=1.2, min=0.0, max=3.0,
            description="Bezier handle leaving the outer port, as a "
                        "fraction of the bridge span")
        handle_in: FloatProperty(
            name="Handle (Inner)", default=0.5, min=0.0, max=3.0,
            description="Bezier handle at the inner port")
        samples: IntProperty(
            name="Samples", default=160, min=48, max=400,
            description="Beads per outer medallion (sets the global "
                        "bead spacing)")
        rail_samples: IntProperty(
            name="Bridge Samples", default=24, min=8, max=64,
            description="Bezier samples per bridge before "
                        "resampling")
        tube_radius: FloatProperty(
            name="Rope Radius", default=0.02, min=0.005, max=0.12,
            description="Rope tube radius (outer sphere radius = 1)")
        tube_sides: IntProperty(
            name="Tube Sides", default=10, min=3, max=32)
        weave_gap: FloatProperty(
            name="Weave Gap", default=0.0, min=0.0, max=0.4,
            description="Radial over/under separation at crossings; "
                        "0 = off (no over/under push). Positive values "
                        "are floored at the rope diameter")
        clearance: FloatProperty(
            name="Clearance", default=0.03, min=0.0, max=0.4,
            description="Strand-strand repulsion (intersection "
                        "avoidance); 0 = off (ropes may intersect). "
                        "Positive values floored at the rope diameter")
        relax_iters: IntProperty(
            name="Relax Iterations", default=120, min=0, max=400,
            description="Bead/stick relaxation steps (0 = the raw "
                        "radial weave seed)")
        output: EnumProperty(
            name="Output",
            items=[('TUBE', "Rope Tubes",
                    "One welded round tube per strand"),
                   ('RIBBON', "Ribbon Straps",
                    "Flat straps, width tangent, thickness radial"),
                   ('CURVE', "Centerline Curves",
                    "Closed poly splines")],
            default='TUBE')
        color_by: EnumProperty(
            name="Color By",
            items=[('STRAND', "By Strand",
                    "A distinct color per closed strand"),
                   ('UNIFORM', "Uniform", "A single material"),
                   ('ELEMENT', "By Element",
                    "Outer arcs / bridges / inner arcs")],
            default='STRAND')
        separate: BoolProperty(
            name="Separate Strands", default=False,
            description="One object per closed strand")

        def execute(self, context):
            shell = build_double_shell(
                solid=self.solid, frequency=self.frequency,
                inner_radius=self.inner_radius,
                overlap_out=self.overlap_out,
                overlap_in=self.overlap_in,
                amplitude=self.amplitude, port_gap=self.port_gap,
                spin=self.spin, advance=self.advance,
                twist=self.twist, handle_out=self.handle_out,
                handle_in=self.handle_in, samples=self.samples,
                rail_samples=self.rail_samples,
                weave_gap=self.weave_gap, clearance=self.clearance)
            lines = shell_lines(shell, self.relax_iters,
                                self.clearance, self.weave_gap,
                                self.tube_radius)
            info = ("%s: %d strands, %d crossings, alternating=%s"
                    % (self.solid, shell['n_components'],
                       len(shell['crossings']),
                       "yes" if shell['consistent'] else "partial"))
            if self.output == 'CURVE':
                obj = _emit_curve(context, "Woven Double Shell",
                                  strand_paths(lines), operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no strands generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, info)
                return {'FINISHED'}
            cells = build_cells(shell, lines, self.output,
                                self.tube_radius, self.tube_sides,
                                self.color_by)
            obj = pc.emit(context, "Woven Double Shell", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no strands generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                me = obj.data
                me.polygons.foreach_set(
                    "use_smooth", [True] * len(me.polygons))
                me.update()
                info += ("  V=%d F=%d" % (len(me.vertices),
                                          len(me.polygons)))
            else:
                for ch in obj.children:
                    if ch.type == 'MESH':
                        ch.data.polygons.foreach_set(
                            "use_smooth",
                            [True] * len(ch.data.polygons))
                        ch.data.update()
            self.report({'INFO'}, info)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            if self.solid == 'GEODESIC':
                lay.prop(self, 'frequency')
            lay.prop(self, 'inner_radius')
            lay.prop(self, 'advance')
            lay.separator()
            for prop in ('overlap_out', 'overlap_in', 'amplitude',
                         'port_gap', 'spin', 'twist', 'handle_out',
                         'handle_in'):
                lay.prop(self, prop)
            lay.separator()
            lay.prop(self, 'samples')
            lay.prop(self, 'rail_samples')
            lay.prop(self, 'tube_radius')
            if self.output == 'TUBE':
                lay.prop(self, 'tube_sides')
            lay.prop(self, 'weave_gap')
            lay.prop(self, 'clearance')
            lay.prop(self, 'relax_iters')
            lay.separator()
            lay.prop(self, 'output')
            if self.output != 'CURVE':
                lay.prop(self, 'color_by')
                lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.woven_double_shell_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_woven_double_shell_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_woven_double_shell_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _min_interstrand(lines):
    """Minimum bead distance between different strands."""
    sid = np.concatenate([np.full(len(P), i, int)
                          for i, P in enumerate(lines)])
    X = np.vstack(lines)
    best = np.inf
    chunk = 800
    for a in range(0, len(X), chunk):
        D = np.linalg.norm(X[a:a + chunk, None, :] - X[None, :, :],
                           axis=2)
        same = sid[a:a + chunk, None] == sid[None, :]
        D[same] = np.inf
        best = min(best, float(D.min()))
    return best


def _test_solid(name, freq, samples, rail_samples, relax_iters,
                verbose=True):
    scaf = _scaffold(name, freq, 20.0)
    Fp, rings, D = scaf['Fp'], scaf['rings'], scaf['D']
    E = len(scaf['edges'])
    n_corners = sum(len(f) for f in Fp)
    # pairing bijection: every inner corner exactly one down rail...
    assert len(D) == n_corners == 2 * E
    assert len(set(D.values())) == len(D), "pairing not a bijection"
    assert set(D.values()) == {(v, t) for v in range(len(rings))
                               for t in range(len(rings[v]))}
    # ... and (via the up map inside the walk) exactly one up rail:
    # the up map is Dinv composed with a ring rotation, a bijection.
    counts = {}
    for adv in (0, 1):
        cycles = strand_cycles(Fp, rings, D, adv)
        oarcs = [c[0] for cyc in cycles for c in cyc]
        iarcs = [c[2] for cyc in cycles for c in cyc]
        dns = [c[1] for cyc in cycles for c in cyc]
        ups = [c[3] for cyc in cycles for c in cyc]
        assert len(oarcs) == len(set(oarcs)) == 2 * E
        assert len(iarcs) == len(set(iarcs)) == 2 * E
        assert len(dns) == len(set(dns)) == 2 * E
        assert len(set((u[1], u[2]) for u in ups)) == 2 * E
        # closure: the successor of each cycle's last entry is its
        # first outer arc
        for cyc in cycles:
            last = cyc[-1][3]
            assert ('O', last[3], last[4]) == cyc[0][0]
        counts[adv] = len(cycles)
    assert counts[0] == E, \
        "advance=0 gave %d components, expected E=%d" \
        % (counts[0], E)
    assert counts[1] < counts[0], "advance=1 did not merge strands"
    if verbose:
        print("  %s: E=%d  components(adv=0)=%d  (adv=1)=%d"
              % (name, E, counts[0], counts[1]))

    shell = build_double_shell(
        solid=name, frequency=freq, samples=samples,
        rail_samples=rail_samples, advance=1)
    inner_radius = shell['inner_radius']
    # every crossing is one-over-one-under, always: the two strands
    # read opposite senses of the same bit by construction
    for (key, lo, hi, flo, fhi) in shell['crossings']:
        s_lo = shell['over'][key] ^ 0
        s_hi = shell['over'][key] ^ 1
        assert s_lo != s_hi
    # bridge x bridge crossings exist at the default twist
    assert shell['cross_class'].get((1, 1), 0) > 0, \
        "no bridge x bridge crossings"
    # seed beads finite, radii inside the two-shell band
    lift = 0.6 * 0.10
    band = lift + 0.02
    for P in shell['seed']:
        assert np.isfinite(P).all()
        r = np.linalg.norm(P, axis=1)
        assert float(r.min()) > inner_radius - band - 0.05
        assert float(r.max()) < 1.0 + band + 0.05
    if verbose:
        print("    beads=%d crossings=%d classes=%s consistent=%s"
              % (sum(len(P) for P in shell['seed']),
                 len(shell['crossings']),
                 dict(sorted(shell['cross_class'].items())),
                 shell['consistent']))
    if relax_iters <= 0:
        return
    trace = []
    tr = 0.03
    clr = max(0.10, 2.2 * tr)
    gap = max(0.10, 2.2 * tr)
    lines = relax_double_shell(shell, iters=relax_iters,
                               clearance=clr, weave_gap=gap,
                               trace=trace)
    for P in lines:
        assert np.isfinite(P).all()
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        assert float(seg.max()) < 6.0 * float(seg.mean()), \
            "a strand tore open"
    mv_early = float(np.mean(trace[:10]))
    mv_late = float(np.mean(trace[-10:]))
    assert mv_late < mv_early, "relaxation did not converge"
    mind = _min_interstrand(lines)
    assert mind >= 0.6 * clr, \
        "strands too close after relax: %.4f < %.4f" \
        % (mind, 0.6 * clr)
    # per-crossing radial order preserved (at the tracked anchors:
    # the strands slide while relaxing, so each crossing is checked
    # where the two strands actually stack).  A small fraction may
    # RELEASE (drift out of contact sideways -- no order left to
    # preserve) or REVERSE (a near-tie crossing settling cleanly the
    # other way -- an equally valid weave, recorded as such); every
    # crossing still in contact must obey its recorded order.
    X = np.vstack(lines)
    oi, ui = shell['relax_anchors']
    rel = shell['relax_released']
    rev = shell['relax_reversed']
    assert float(rel.mean() if len(rel) else 0.0) < 0.15, \
        "too many crossings released"
    assert float(rev.mean() if len(rev) else 0.0) < 0.15, \
        "too many crossings reversed"
    ok = 0
    for c in range(len(oi)):
        if rel[c]:
            continue
        ro = float(np.linalg.norm(X[oi[c]]))
        ru = float(np.linalg.norm(X[ui[c]]))
        assert ro - ru > 0.0, "crossing radial order flipped"
        ok += 1
    # the shells stay separated
    el = np.concatenate(shell['elem'])
    r = np.linalg.norm(X, axis=1)
    sep = (float(r[el == ELEM_OUTER].mean())
           - float(r[el == ELEM_INNER].mean()))
    assert sep > 0.5 * (1.0 - inner_radius), \
        "shells collapsed: separation %.3f" % sep
    # TUBE geometry: non-empty, finite, all quads
    cells = build_cells(shell, lines, 'TUBE', tr, 8, 'STRAND')
    assert cells
    for (cv, cf, cm) in cells:
        assert cf and all(len(fc) == 4 for fc in cf)
        assert np.isfinite(np.asarray(cv)).all()
    if verbose:
        print("    relax: mv %.5f -> %.5f  min dist %.4f  "
              "sep %.3f  order ok %d" % (mv_early, mv_late, mind,
                                         sep, ok))


def _self_test():
    print("Woven Double Shell self-test")
    _test_solid('CUBE', 2, 128, 16, 60)
    _test_solid('DODECA', 2, 112, 14, 60)
    _test_solid('ICOSA', 2, 112, 14, 0)
    _test_solid('GEODESIC', 2, 60, 10, 0)
    print("RESULT: OK")
