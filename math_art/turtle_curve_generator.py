
# Turtle Curves -- one operator for the whole family of 2-D line-art
# constructions, lifted into printable geometry.
#
# Six modes, because there are genuinely six different ways to produce a
# turtle path and collapsing them into one would misrepresent all of
# them:
#
#   NODE   node-rewriting L-systems -- the classic path, delegated to the
#          `lsystem` engine, where a module's placement depends on the
#          whole chain from the base of the structure to it.
#   EDGE   the Koch construction -- a segment is replaced by lines whose
#          position, orientation and scale come from THAT SEGMENT alone.
#          Signed teragon generators on an n-gon initiator.
#   FOLD   paper folding, with a CONTINUOUS unfolding parameter so the
#          curve visibly unfolds between generations (keyframeable).
#   FASS   space-Filling, self-Avoiding, Simple, self-Similar curves.
#   WORD   the turn sequence read off a combinatorial word.
#   SPIRO  spirolaterals, with the reversal enumeration reduced by the
#          Whitney-Graustein invariant.
#
# NODE and EDGE are separate modes rather than a style flag because the
# 2003 course notes show the same production giving DIFFERENT figures
# under the two composition rules.  See `lsystem/koch.py` for the full
# statement.
#
# Output lifts turn the path into an object: a bevelled tube, a filled
# flat island, or an extruded prism plate.  Closure decides which are
# offered -- an open arc cannot be filled -- and that decision comes from
# the Closed-Path Theorem in `lsystem/closure.py` rather than a guess.
#
# References:
# - Helge von Koch (1904); Ernesto Cesaro (1906); Paul Levy (1938).
# - Benoit Mandelbrot, "The Fractal Geometry of Nature", 1982, ch. 6-7.
# - Przemyslaw Prusinkiewicz, Aristid Lindenmayer and F. David Fracchia,
#   "Synthesis of Space-Filling Curves on the Square Grid", 1991.
# - Frank Odds, "Spirolaterals", Mathematics Teacher 66(2), 1973.
# - Robert Krawczyk, "Spirolaterals, Complexity from Simplicity",
#   Bridges 2000.
# - Harold Abelson and Andrea diSessa, "Turtle Geometry", MIT Press,
#   1981, ch. 4 -- the Closed-Path Theorem and Whitney-Graustein.
# - Alexis Monnerot-Dumaine, "The Fibonacci word fractal", 2009.
# - Jun Ma and Judy Holdener, "When Thue-Morse meets Koch", Fractals
#   13(3), 2005.

bl_info = {
    "name": "Turtle Curves",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Fractals > Turtle Curve",
    "description": "Koch teragons, folded dragons, FASS curves, word "
                   "curves and spirolaterals",
    "category": "Add Curve",
}

import numpy as np

try:
    from . import lsystem as ls
    from .lsystem import closure, koch, lifts, words
except ImportError:                       # headless / direct execution
    import lsystem as ls
    from lsystem import closure, koch, lifts, words


MODES = ("NODE", "EDGE", "FOLD", "FASS", "WORD", "SPIRO")


def build_points(mode, **kw):
    """Every mode reduces to one thing: an (N, 2) array of points.

    Keeping that contract means the output lifts, the closure test and
    the normalisation are written once rather than six times.
    """
    if mode == 'EDGE':
        return koch.teragon_preset(kw.get("teragon", "KOCH"),
                                   iters=kw.get("iters", 4),
                                   alpha=kw.get("alpha"),
                                   sides=kw.get("sides"),
                                   sign=kw.get("sign"))
    if mode == 'FOLD':
        base = 3 if kw.get("fold", "DRAGON") in ("TERDRAGON",
                                                 "FUDGEFLAKE") else 2
        _b, ang = koch.FOLDS.get(kw.get("fold", "DRAGON"), (base, 90.0))
        return koch.fold_curve(kw.get("iters", 10),
                               angle=kw.get("angle", ang) or ang,
                               base=base,
                               fraction=kw.get("fraction", 1.0))
    if mode == 'FASS':
        return koch.fass_points(kw.get("fass", "HILBERT"),
                                kw.get("iters", 4))
    if mode == 'WORD':
        n = max(int(kw.get("count", 800)), 2)
        name = kw.get("word", "FIBONACCI")
        ang, rule = words.WORD_DEFAULTS.get(name, (90.0, "ALTERNATE"))
        if not kw.get("word_defaults", True):
            ang, rule = kw.get("angle", ang), kw.get("map", rule)
        return words.word_curve(name, n, angle=ang, mode=rule)
    if mode == 'SPIRO':
        rev = kw.get("reversals", ())
        return words.spirolateral(kw.get("order", 5),
                                  angle=kw.get("angle", 90.0),
                                  reversals=rev,
                                  blocks=kw.get("blocks"))
    # NODE -- the classic engine, flattened to 2-D
    g = ls.preset(kw.get("preset", "KOCH"))
    out = ls.interpret(g.derive(kw.get("iters", g.iters or 4)),
                       angle=g.angle, draw=g.draw or None)
    return np.vstack([s.points[:, :2] for s in out.strands])


def normalise(points, size=2.0):
    """Centre on the origin and fit the 2 m cube, as every generator does."""
    P = np.asarray(points, dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    ext = float((hi - lo).max())
    P = P - (lo + hi) / 2.0
    return P * (size / ext) if ext > 1e-12 else P


def describe(points):
    """A one-line report: segments, closure and turning number."""
    P3 = koch.to3d(points)
    t = closure.turns_of(P3)
    shut = closure.closes_geometrically(points)
    tn = closure.turning_number(t)
    return (f"{len(points) - 1} segments, "
            f"{'closed' if shut else 'open'}, "
            f"turning number {tn:+.3f}")


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    from .lsystem import polyline as _poly

    def _mesh_object(points, name, closed):
        """The path as a plain MESH of vertices and edges.

        The raw figure, with no fillet and no bevel: this is the exact
        polyline the mathematics produced, which is what you want as a
        Skin/Curve modifier input, for measuring, or for any downstream
        mesh operation.  The fillet is a rendering fix for Blender's
        bevel and would falsify the geometry here, so it is not applied.
        """
        pts = np.asarray(points, dtype=float)[:, :2]
        verts = [(float(x), float(y), 0.0) for x, y in pts]
        edges = [(i, i + 1) for i in range(len(verts) - 1)]
        if closed and len(verts) > 2:
            # a closed path repeats its seam vertex; weld it rather than
            # leaving a doubled vertex and a zero-length edge
            if abs(verts[0][0] - verts[-1][0]) < 1e-12 and \
                    abs(verts[0][1] - verts[-1][1]) < 1e-12:
                verts.pop()
                edges = [(i, i + 1) for i in range(len(verts) - 1)]
            edges.append((len(verts) - 1, 0))
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, edges, [])
        me.update()
        return bpy.data.objects.new(name, me)

    def _curve_object(points, name, output, radius, resolution,
                      height, closed, fillet, fillet_segments):
        """One CURVE datablock, configured for the requested lift.

        All three lifts are curve settings rather than mesh operations:
        a round bevel gives the tube, `extrude` with a 2-D fill gives the
        prism, and extrude 0 with the same fill gives the flat island.
        That keeps the result editable and the operator small.
        """
        pts = np.asarray(points, dtype=float)
        w = np.ones(len(pts))
        if output == 'CURVE':
            pts2, _w = _poly.round_corners(pts[:, :2], w, fillet,
                                           fillet_segments, closed=closed)
        else:
            pts2 = pts[:, :2]

        cu = bpy.data.curves.new(name, 'CURVE')
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts2) - 1)
        for i, p in enumerate(pts2):
            sp.points[i].co = (float(p[0]), float(p[1]), 0.0, 1.0)
        sp.use_cyclic_u = bool(closed)

        if output == 'CURVE':
            cu.dimensions = '3D'
            cu.bevel_depth = float(radius)
            cu.bevel_resolution = int(resolution)
        else:
            # A 2-D curve is what Blender will FILL; the fill is what
            # makes an island or a prism rather than a wire loop.
            cu.dimensions = '2D'
            cu.fill_mode = 'BOTH'
            # Blender's `extrude` goes BOTH ways from the curve plane, so
            # the finished thickness is twice it.  Halve, so that the
            # Height property means the thickness you actually get.
            cu.extrude = 0.0 if output == 'ISLAND' else float(height) / 2.0
        return bpy.data.objects.new(name, cu)

    class CURVE_OT_turtle_curve_add(bpy.types.Operator):
        """Add a turtle curve: Koch teragons, folded dragons, FASS
        curves, curves from combinatorial words, or spirolaterals"""
        bl_idname = "curve.turtle_curve_add"
        bl_label = "Turtle Curve"
        bl_options = {'REGISTER', 'UNDO', 'PRESET'}

        mode: EnumProperty(
            name="Mode",
            description="Which turtle-path construction to build",
            items=[('EDGE', "Koch Teragon", "Edge rewriting: a segment is "
                            "replaced by a scaled copy of a generator"),
                   ('FOLD', "Paper Fold", "Dragon family, with a "
                            "continuous unfolding parameter"),
                   ('FASS', "FASS Curve", "Space-filling, self-avoiding, "
                            "simple and self-similar"),
                   ('WORD', "Word Curve", "Turns read off a combinatorial "
                            "sequence"),
                   ('SPIRO', "Spirolateral", "Segments 1..n at a fixed "
                             "turn, repeated until it closes"),
                   ('NODE', "Node Rewriting", "The classic L-system path, "
                            "for comparison with edge rewriting")],
            default='EDGE')

        # -- EDGE
        teragon: EnumProperty(
            name="Generator",
            description="Koch generator replacing each segment (Koch "
                        "Teragon mode)",
            items=[(k, k.replace("_", " ").title(), k)
                   for k in sorted(koch.TERAGONS)],
            default='KOCH')
        sides: IntProperty(
            name="Initiator Sides", default=3, min=2, max=24,
            description="Regular polygon the generator is applied to; "
                        "2 gives a single open segment")
        alpha: FloatProperty(
            name="Bump Angle", default=60.0, min=1.0, max=89.0,
            description="Cesaro's free apex angle. 60 gives von Koch's "
                        "original; approaching 90 makes it a spike")
        use_alpha: BoolProperty(
            name="Override Bump Angle", default=False,
            description="Use the Bump Angle below instead of the "
                        "generator's built-in apex angle")
        outward: BoolProperty(
            name="Outward", default=True,
            description="Signed teragon: outward gives the snowflake, "
                        "inward the antisnowflake")

        # -- FOLD
        fold: EnumProperty(
            name="Fold",
            description="Which paper-folding dragon curve to build "
                        "(Paper Fold mode)",
            items=[(k, k.replace("_", " ").title(), k)
                   for k in sorted(koch.FOLDS)],
            default='DRAGON')
        fraction: FloatProperty(
            name="Unfold", default=1.0, min=0.0, max=1.0,
            description="Angle of the newest fold only, as a fraction. "
                        "0 is the previous generation, 1 this one -- "
                        "sweep it to animate the curve unfolding")

        # -- FASS
        fass: EnumProperty(
            name="Curve",
            description="Which space-filling FASS curve to build",
            items=[(k, k.replace("_", " ").title(), k)
                   for k in sorted(koch.FASS)],
            default='HILBERT')

        # -- WORD
        word: EnumProperty(
            name="Sequence",
            description="Which combinatorial sequence drives the turns "
                        "(Word Curve mode)",
            items=[(k, k.replace("_", " ").title(), k)
                   for k in sorted(words.WORDS)],
            default='FIBONACCI')
        mapping: EnumProperty(
            name="Turn Rule",
            description="How each term of the sequence is turned into a "
                        "turn",
            items=[('ALTERNATE', "Alternate", "Turn on a zero term, "
                                "alternating by position parity -- the "
                                "Fibonacci word fractal rule"),
                   ('SIGNED', "Signed", "Left on 0, right otherwise"),
                   ('GATED', "Gated", "Turn only on a non-zero term"),
                   ('SCALED', "Scaled", "Turn proportional to the term")],
            default='ALTERNATE')
        count: IntProperty(
            name="Terms", default=800, min=4, max=200000,
            description="Number of sequence terms to draw (Word Curve "
                        "mode)")
        word_defaults: BoolProperty(
            name="Recommended Settings", default=True,
            description="Use the angle and turn rule under which this "
                        "sequence draws its intended figure. Most other "
                        "combinations degenerate into a zigzag along one "
                        "axis; turn this off to explore anyway")

        # -- SPIRO
        order: IntProperty(
            name="Order", default=5, min=1, max=24,
            description="Segments of length 1..n in each block")
        reversals: StringProperty(
            name="Reversals", default="",
            description="Comma-separated 1-based step numbers whose turn "
                        "is reversed (Krawczyk's enumeration)")

        # -- NODE
        preset: EnumProperty(
            name="Preset",
            description="Which L-system preset to interpret (Node "
                        "Rewriting mode)",
            items=lambda self, ctx: _NODE_ITEMS,
            default=None)

        # -- shared
        iters: IntProperty(name="Iterations", default=4, min=0, max=20,
                           description="Number of rewriting / subdivision "
                                       "iterations")
        angle: FloatProperty(
            name="Angle", default=90.0, min=-180.0, max=180.0,
            description="Turn angle at each step, in degrees")

        output: EnumProperty(
            name="Output",
            description="How to lift the flat path into geometry",
            items=[('CURVE', "Bevelled Curve", "A round tube along the "
                             "path"),
                   ('ISLAND', "Filled Island", "The enclosed area as a "
                              "flat filled face -- closed paths only"),
                   ('PRISM', "Extruded Prism", "The island given "
                             "thickness"),
                   ('MESH', "Plain Mesh", "Vertices and edges along the "
                            "exact path -- no bevel, no fillet"),
                   ('RELIEF', "Mitred Relief", "A prism along the path "
                              "with MITRED joints, terracing in height "
                              "at every turn"),
                   ('SCAFFOLD', "Scaffold on Columns", "The path raised "
                                "on columns over a base plate -- what "
                                "makes an open figure printable")],
            default='CURVE')
        radius: FloatProperty(name="Tube Radius", default=0.01,
                              min=0.0001, max=1.0,
                              description="Radius of the round tube along "
                                          "the path (Bevelled Curve "
                                          "output)")
        resolution: IntProperty(name="Bevel Resolution", default=2,
                                min=0, max=16,
                                description="Smoothness of the tube's "
                                            "round cross-section")
        height: FloatProperty(name="Height", default=0.01,
                              min=0.0001, max=5.0,
                              description="Finished thickness of the "
                                          "prism, plate to plate")
        fillet: FloatProperty(
            name="Round Corners", default=0.25, min=0.0, max=0.5,
            description="Blender places a bevel profile in the plane "
                        "BISECTING two segments rather than mitring it, "
                        "so a sharp corner pinches the tube to "
                        "cos(angle/2) of its width. Rounding keeps the "
                        "cross-section even; 0 keeps corners exact")
        fillet_segments: IntProperty(name="Corner Segments", default=4,
                                     min=1, max=16,
                                     description="Number of segments used "
                                                 "to round each corner")

        # -- mitred relief
        band: FloatProperty(
            name="Band Width", default=0.06, min=0.002, max=1.0,
            description="Width of the ribbon. The joints are MITRED: the "
                        "offset at a corner runs along the bisector at "
                        "w/2 / cos(angle/2), so there is no notch")
        relief_step: FloatProperty(
            name="Height Step", default=0.02, min=0.0, max=0.5,
            description="Height added per right-angle of turning. This "
                        "is what makes it a relief rather than an "
                        "extrusion -- the surface terraces as the curve "
                        "winds. 0 gives a constant-height prism")
        miter_limit: FloatProperty(
            name="Miter Limit", default=4.0, min=1.0, max=20.0,
            description="Cap on the mitre extension. It diverges at a "
                        "hairpin, so without a limit one corner flies "
                        "off to infinity")

        # -- scaffold
        gap: FloatProperty(name="Stand-off", default=0.35,
                           min=0.01, max=2.0,
                           description="Height of the curve above the plate")
        plate: FloatProperty(name="Plate Thickness", default=0.04,
                             min=0.005, max=0.5,
                             description="Thickness of the base plate "
                                         "under the scaffold")
        column_radius: FloatProperty(name="Column Radius", default=0.012,
                                     min=0.002, max=0.2,
                                     description="Radius of the columns "
                                                 "holding the path above "
                                                 "the plate")
        column_every: IntProperty(
            name="Column Every", default=8, min=1, max=200,
            description="One column every N path points")

        # -- assembly
        assembly: EnumProperty(
            name="Assembly",
            description="How to repeat the figure into a solid "
                        "arrangement",
            items=[('NONE', "Single", "One copy"),
                   ('AXIS', "About an Axis", "N copies rotated about Z"),
                   ('CUBE', "Cube Faces", "One copy on each face of a "
                            "cube -- what turns a flat figure into a "
                            "solid object")],
            default='NONE')
        copies: IntProperty(name="Copies", default=6, min=2, max=64,
                            description="Number of copies rotated about "
                                        "the axis (About an Axis "
                                        "assembly)")
        spin: FloatProperty(name="Phase", default=0.0,
                            min=-180.0, max=180.0,
                            description="Starting rotation of the axis "
                                        "assembly, in degrees")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size; 1 fits the 2 m "
                                         "cube")

        def _kwargs(self):
            rev = []
            for tok in self.reversals.replace(";", ",").split(","):
                tok = tok.strip()
                if tok.isdigit():
                    rev.append(int(tok))
            return dict(
                teragon=self.teragon, sides=self.sides,
                alpha=self.alpha if self.use_alpha else None,
                sign=+1 if self.outward else -1,
                fold=self.fold, fraction=self.fraction,
                fass=self.fass, word=self.word, map=self.mapping,
                count=self.count, order=self.order, reversals=tuple(rev),
                word_defaults=self.word_defaults,
                preset=self.preset or 'KOCH',
                iters=self.iters, angle=self.angle)

        def execute(self, context):
            try:
                pts = build_points(self.mode, **self._kwargs())
            except (KeyError, ValueError, ls.GrammarError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            if len(pts) < 2:
                self.report({'ERROR'}, "produced no geometry")
                return {'CANCELLED'}

            shut = closure.closes_geometrically(pts)
            pts = normalise(pts, 2.0 * self.scale)

            out = self.output
            if out in ('ISLAND', 'PRISM') and not shut:
                # An open arc has no interior, so there is nothing to
                # fill.  Say so rather than emitting a silently wrong
                # face closed by an arbitrary chord.
                self.report({'WARNING'},
                            "path is open -- filling needs a closed path; "
                            "drawing the bevelled curve instead")
                out = 'CURVE'

            name = f"Turtle {self.mode.title()}"
            if out in ('RELIEF', 'SCAFFOLD') or self.assembly != 'NONE':
                ob = self._solid(pts, name, out, shut)
                if ob is None:
                    self.report({'ERROR'}, "produced no geometry")
                    return {'CANCELLED'}
            elif out == 'MESH':
                ob = _mesh_object(pts, name, shut)
            else:
                ob = _curve_object(pts, name, out, self.radius,
                                   self.resolution, self.height, shut,
                                   self.fillet, self.fillet_segments)
            context.collection.objects.link(ob)
            for o in context.selected_objects:
                o.select_set(False)
            ob.select_set(True)
            context.view_layer.objects.active = ob
            self.report({'INFO'}, f"{name}: {describe(pts)}")
            return {'FINISHED'}

        def _solid(self, pts, name, out, shut):
            """The mesh lifts, and any assembly applied on top.

            A relief or a scaffold is already a mesh, and an assembly has
            to repeat a mesh, so anything reaching here is built as
            (verts, faces) rather than as a curve.
            """
            if out == 'SCAFFOLD':
                verts, faces = lifts.scaffold(
                    pts, closed=shut, plate_thickness=self.plate,
                    gap=self.gap, column_radius=self.column_radius,
                    every=self.column_every)
                rv, rf = lifts.mitred_relief(
                    pts, width=self.band, closed=shut,
                    height=max(self.band * 0.4, 1e-3),
                    step=self.relief_step, miter_limit=self.miter_limit)
                base = len(verts)
                verts = list(verts) + list(rv)
                faces = list(faces) + [tuple(i + base for i in q)
                                       for q in rf]
            else:
                verts, faces = lifts.mitred_relief(
                    pts, width=self.band, closed=shut,
                    height=max(self.band * 0.6, 1e-3),
                    step=self.relief_step, miter_limit=self.miter_limit)
            if self.assembly != 'NONE':
                verts, faces = lifts.assemble(
                    verts, faces, mode=self.assembly,
                    count=self.copies, size=2.0 * self.scale,
                    spin=self.spin)
            if not faces:
                return None
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(float(c) for c in v) for v in verts],
                           [], faces)
            me.validate()
            me.update()
            return bpy.data.objects.new(name, me)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "mode")
            if self.mode == 'EDGE':
                for k in ("teragon", "sides", "outward", "use_alpha"):
                    lay.prop(self, k)
                if self.use_alpha:
                    lay.prop(self, "alpha")
                lay.prop(self, "iters")
            elif self.mode == 'FOLD':
                for k in ("fold", "iters", "fraction"):
                    lay.prop(self, k)
            elif self.mode == 'FASS':
                for k in ("fass", "iters"):
                    lay.prop(self, k)
            elif self.mode == 'WORD':
                lay.prop(self, "word")
                lay.prop(self, "count")
                lay.prop(self, "word_defaults")
                if self.word_defaults:
                    ang, rule = words.WORD_DEFAULTS.get(
                        self.word, (90.0, "ALTERNATE"))
                    lay.box().label(text=f"{ang:g} deg, {rule.title()}",
                                    icon='INFO')
                else:
                    lay.prop(self, "mapping")
                    lay.prop(self, "angle")
            elif self.mode == 'SPIRO':
                for k in ("order", "angle", "reversals"):
                    lay.prop(self, k)
                q = 360.0 / self.angle if self.angle else 0.0
                if abs(q - round(q)) < 1e-9 and round(q) > 0:
                    qi = int(round(q))
                    box = lay.box()
                    box.label(text=f"{closure.blocks_to_close(self.order, qi)}"
                                   f" blocks to close (q/gcd(n,q))",
                              icon='INFO')
                    if not closure.spirolateral_closes(self.order, qi):
                        box.label(text="spirals: q divides n",
                                  icon='ERROR')
            else:
                for k in ("preset", "iters"):
                    lay.prop(self, k)
            lay.separator()
            lay.prop(self, "output")
            if self.output == 'CURVE':
                for k in ("radius", "resolution", "fillet"):
                    lay.prop(self, k)
            elif self.output == 'PRISM':
                lay.prop(self, "height")
            elif self.output == 'RELIEF':
                for k in ("band", "relief_step", "miter_limit"):
                    lay.prop(self, k)
            elif self.output == 'SCAFFOLD':
                for k in ("band", "gap", "plate", "column_radius",
                          "column_every"):
                    lay.prop(self, k)
            lay.separator()
            lay.prop(self, "assembly")
            if self.assembly == 'AXIS':
                lay.prop(self, "copies")
                lay.prop(self, "spin")
            lay.prop(self, "scale")

    _NODE_ITEMS = [(k, ls.title(k), k) for k in sorted(ls.CURVES)]

    def register():
        bpy.utils.register_class(CURVE_OT_turtle_curve_add)

    def unregister():
        bpy.utils.unregister_class(CURVE_OT_turtle_curve_add)


def _selftest():
    # Every mode must produce a finite, non-trivial planar path.
    cases = [
        ('EDGE',  dict(teragon="KOCH", iters=3)),
        ('EDGE',  dict(teragon="MINKOWSKI", iters=2, sides=4)),
        ('FOLD',  dict(fold="DRAGON", iters=8)),
        ('FOLD',  dict(fold="TERDRAGON", iters=4)),
        ('FASS',  dict(fass="HILBERT", iters=3)),
        ('FASS',  dict(fass="GOSPER", iters=2)),
        ('WORD',  dict(word="FIBONACCI", count=400)),
        ('WORD',  dict(word="THUE_MORSE", count=400)),
        ('WORD',  dict(word="KOLAKOSKI", count=400)),
        ('WORD',  dict(word="PRIME", count=400)),
        ('SPIRO', dict(order=5, angle=90.0)),
        ('NODE',  dict(preset="KOCH", iters=3)),
    ]
    for mode, kw in cases:
        P = build_points(mode, **kw)
        assert P.ndim == 2 and P.shape[1] == 2, (mode, P.shape)
        assert len(P) > 2, (mode, len(P))
        assert np.all(np.isfinite(P)), mode
        N = normalise(P, 2.0)
        ext = N.max(axis=0) - N.min(axis=0)
        assert abs(float(ext.max()) - 2.0) < 1e-9, (mode, kw, ext)
        # a real 2-D figure, not a collapsed line
        assert float(ext.min()) > 1e-9, (mode, kw, ext)
        assert isinstance(describe(P), str)

    # Closure must be reported correctly, because the fill lifts depend
    # on it: a Koch snowflake on a triangle is closed, a dragon is not.
    assert closure.closes_geometrically(
        build_points('EDGE', teragon="KOCH", sides=3, iters=3))
    assert not closure.closes_geometrically(
        build_points('FOLD', fold="DRAGON", iters=8))
    assert closure.closes_geometrically(
        build_points('SPIRO', order=3, angle=90.0))
    assert not closure.closes_geometrically(
        build_points('SPIRO', order=4, angle=90.0))     # the spiral

    # The signed teragon really does flip: outward and inward differ.
    a = build_points('EDGE', teragon="KOCH", iters=3, sign=+1)
    b = build_points('EDGE', teragon="KOCH", iters=3, sign=-1)
    assert a.shape == b.shape and not np.allclose(a, b)

    # The unfolding parameter is continuous, which is what makes it
    # keyframeable rather than a discrete iteration count.
    mids = [build_points('FOLD', fold="DRAGON", iters=6, fraction=f)
            for f in (0.0, 0.5, 1.0)]
    assert not np.allclose(mids[0], mids[1])
    assert not np.allclose(mids[1], mids[2])

    # normalise must be exact and must survive a degenerate input
    n2 = normalise(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert abs(float(n2[:, 0].max() - n2[:, 0].min()) - 2.0) < 1e-12
    flat = normalise(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert np.all(np.isfinite(flat))

    # --- the output lifts ---------------------------------------------
    from .lsystem import lifts as _lf  # noqa: F401  (import shape check)

    print(f"turtle_curve_generator: OK -- {len(cases)} mode/parameter "
          f"combinations produce finite planar figures fitting the 2 m "
          f"cube; closure reported correctly for snowflake, dragon and "
          f"the order-4 spiral")
