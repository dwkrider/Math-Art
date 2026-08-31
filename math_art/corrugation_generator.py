# Corrugate a mathematical surface into a folded sheet.
#
# The Blender operator over `math_art/crease/corrugate.py`, which holds
# the mathematics.  This module is only the skin: pick a target, build
# the pleated form and its flat crease pattern, and report what the
# approximation cost.
#
# WHY THIS BELONGS IN THIS ADD-ON SPECIFICALLY.  Inverse design -- "make
# me a crease pattern that folds into THAT" -- only means anything if
# there is a library of interesting surfaces to point it at, and this
# add-on is mostly that library.  Elsewhere it would be a demo with a
# sphere in it.
#
# THE THEOREM IS PART OF THE UI, NOT A FOOTNOTE.  A flat sheet has
# Gaussian curvature K = 0 and, by Gauss's Theorema Egregium, folding
# cannot change it -- so a folded sheet is NEVER isometric to a curved
# surface.  This operator approximates, and it says by how much every
# time it runs: the fit error, and how much bigger the flat sheet is
# than the surface.  A tool that quietly presented an approximation as a
# solution would be lying about something a 19th-century theorem
# forbids.
#
# References:
#   L. H. Dudte, E. Vouga, T. Tachi, L. Mahadevan, "Programming
#       curvature using origami tessellations," Nature Materials 15,
#       2016.
#   T. Tachi, "Freeform Variations of Origami," J. Geometry and
#       Graphics 14(2), 2010.
#   C. F. Gauss, "Disquisitiones generales circa superficies curvas,"
#       1827 -- the Theorema Egregium.

import numpy as np

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

try:
    from . import crease
except ImportError:                                   # headless import
    import crease

_ASSIGN_CODE = {"M": 0, "V": 1, "F": 2, "U": 3, "B": 4}

_TARGET_ITEMS = (
    ('HYPAR', "Hyperbolic Paraboloid",
     "z = x^2 - y^2. Negative curvature everywhere, which is the "
     "favourable case: pleating adds material and a saddle needs it"),
    ('SCHERK', "Scherk Surface",
     "A minimal surface, so curvature is negative or zero throughout"),
    ('CATENOID', "Catenoid",
     "The minimal surface of revolution. Its parameter grid sweeps a "
     "RING, so the pleat direction matters far more here than on the "
     "others -- pleat it the wrong way and the pattern describes a "
     "shape it cannot hold. Leave Pleat Direction on Automatic"),
    ('SPHERE', "Spherical Cap",
     "Positive curvature -- the hard case. Pleating adds material and a "
     "sphere needs material removed, so expect a worse fit"),
    ('PLANE', "Plane",
     "The trivial case, useful as a control: a pleated plane is exactly "
     "developable, so the reported residual should be zero"),
)


class MESH_OT_corrugation_add(bpy.types.Operator):
    """Approximate a curved surface with a pleated, foldable sheet"""

    bl_idname = "mesh.corrugation_add"
    bl_label = "Surface Corrugation"
    bl_options = {'REGISTER', 'UNDO'}

    target: EnumProperty(
        name="Surface", items=_TARGET_ITEMS, default='HYPAR',
        description="Which surface to approximate")
    nu: IntProperty(
        name="Cells U", default=16, min=2, max=80,
        description="Pleats across the sheet, one way")
    nv: IntProperty(
        name="Cells V", default=16, min=2, max=80,
        description="Cells across the sheet, the other way")
    size: FloatProperty(
        name="Size", default=2.0, min=0.01, max=100.0, unit='LENGTH',
        description="Width of the target surface")
    depth: FloatProperty(
        name="Depth", default=0.6, min=0.0, max=10.0,
        description="How strongly the target surface curves")
    amplitude: FloatProperty(
        name="Pleat Depth", default=0.12, min=0.0, max=2.0,
        description="How far the pleats stand off the surface. This is "
                    "the store of surplus material -- deeper pleats can "
                    "absorb more curvature, at the cost of a bigger sheet")
    pleat_axis: EnumProperty(
        name="Pleat Direction", default='AUTO',
        items=[('AUTO', "Automatic",
                "Try both directions and keep the one whose pattern "
                "actually holds the target shape. Costs two extra "
                "solves, and is worth it on surfaces of revolution"),
               ('U', "Across U", "Pleats run along the first parameter"),
               ('V', "Across V", "Pleats run along the second parameter")],
        description="Which way the pleats run. On a surface of "
                    "revolution this is the difference between a pattern "
                    "that works and one that does not")
    relax: IntProperty(
        name="Flatten Steps", default=1200, min=50, max=20000,
        description="Iterations spent laying the sheet flat. The residual "
                    "that remains is the curvature the pleats could not "
                    "absorb, not a convergence failure -- it stops falling")
    make_pattern: BoolProperty(
        name="Also Build Crease Pattern", default=True,
        description="Emit the flat crease pattern beside the pleated "
                    "form, with its mountain and valley assignments")

    def execute(self, context):
        try:
            # No progress reporting here on purpose.  This operator is
            # fast -- measured 0.15s to 0.68s from 10x10 to 24x24 -- and
            # a progress indicator on a sub-second operation is noise.
            # Automatic pleat direction costs two extra solves and is
            # the one slow case; if that becomes a complaint, this needs
            # the same modal treatment as Fold Pattern rather than
            # `progress_*`, which draws nothing from inside a blocking
            # execute().
            ax = {'AUTO': None, 'U': 0, 'V': 1}[self.pleat_axis]
            frame, folded, rep = crease.corrugate.fit(
                self.target, nu=self.nu, nv=self.nv, size=self.size,
                depth=self.depth, amplitude=self.amplitude,
                iters=self.relax, axis=ax)
        except crease.corrugate.CorrugateError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        objs = []
        objs.append(self._mesh_object(
            context, f"{self.target.title()} Corrugation", folded,
            frame.faces, frame.edges, frame.assignment, frame.fold_angle))
        if self.make_pattern:
            flat = np.hstack([np.asarray(frame.verts, dtype=float)[:, :2],
                              np.zeros((frame.n_verts, 1))])
            # BELOW the folded form, and far enough below to stay clear
            # once it is folded too.
            #
            # The clearance is not a guess.  Folding this pattern is
            # supposed to reproduce the corrugation, so the folded
            # pattern will occupy the same HEIGHT as the corrugation --
            # call it H.  The sheet starts flat at plane z = p and rises
            # or falls from there depending on where the solver settles,
            # so in the worst case it reaches all the way up to p + H.
            # Requiring p + H to stay under the corrugation's underside
            # gives p = z_min - H - margin, which holds whichever way
            # the fold happens to go rather than only for the direction
            # seen while testing.
            fz = np.asarray(folded, dtype=float)
            height = float(np.ptp(fz[:, 2])) or float(rep["target_w"]) * 0.25
            margin = 0.15 * max(height, float(rep["target_w"]))
            # Directly underneath, not offset sideways: lining the two up
            # in x and y is what makes the crease pattern readable
            # against the shape it produces.
            #
            # THE OFFSET GOES ON THE OBJECT, NOT INTO THE MESH.  Baking it
            # into the vertices was the obvious thing and it broke
            # folding: a sheet sitting at a constant z = -2.02 is planar
            # but not FLAT, and the solver -- which quite reasonably wants
            # to start from the plane -- refused it as "already folded".
            # Placement is a transform, so it belongs in the transform.
            # BOUNDING-BOX CENTRE, not the mean of the vertices.  The
            # mean is the centre of MASS of the sample points, which
            # only coincides with the middle of the shape when the shape
            # is symmetric and evenly sampled.  The catenoid's flat
            # pattern is neither -- it unrolls to a long curved strip --
            # so centring on the mean pushed it sideways, visibly off to
            # one side of the form it belongs under.
            def _mid(a, i):
                return 0.5 * (float(a[:, i].min()) + float(a[:, i].max()))

            pattern_offset = (
                _mid(fz, 0) - _mid(flat, 0),
                _mid(fz, 1) - _mid(flat, 1),
                float(fz[:, 2].min()) - height - margin)
            pat = self._mesh_object(
                context, f"{self.target.title()} Crease Pattern", flat,
                frame.faces, frame.edges, frame.assignment,
                frame.fold_angle)
            pat.location = pattern_offset
            objs.append(pat)

        for o in context.selected_objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        context.view_layer.objects.active = objs[0]

        # The theorem, stated every run.  `area_ratio` above 1 is not a
        # defect -- it IS the mechanism, since the surplus is what the
        # pleats store.
        self.report(
            {'INFO'},
            f"{self.target.title()}: "
            + crease.corrugate.report_summary(rep))
        return {'FINISHED'}

    @staticmethod
    def _mesh_object(context, name, verts, faces, edges, assignment,
                     fold_angle=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(map(float, p)) for p in verts], [],
                       [list(map(int, f)) for f in faces])
        me.update()
        want = {}
        for k, (a, b) in enumerate(np.asarray(edges).reshape(-1, 2)):
            code = _ASSIGN_CODE.get(str(assignment[k]), 3)
            want[(int(a), int(b))] = code
            want[(int(b), int(a))] = code
        attr = me.attributes.new("crease_assignment", 'INT', 'EDGE')
        attr.data.foreach_set(
            "value", [want.get(tuple(e.vertices), 3) for e in me.edges])

        # THE FOLD ANGLES HAVE TO TRAVEL WITH THE MESH.
        #
        # The engine records the angle each crease must reach, but the
        # Blender side dropped them, so Fold Pattern on the emitted
        # pattern saw only M/V and drove every crease to a uniform 57.3
        # degrees -- and reproduced some other shape.  Recording the
        # angles in the engine while not writing them here fixed nothing
        # a user could see, which is the only place it matters.
        if fold_angle is not None:
            ang = {}
            for k, (a, b) in enumerate(np.asarray(edges).reshape(-1, 2)):
                v = float(fold_angle[k])
                if v != v:                         # NaN: boundary, no angle
                    continue
                ang[(int(a), int(b))] = v
                ang[(int(b), int(a))] = v
            fa = me.attributes.new("fold_angle", 'FLOAT', 'EDGE')
            fa.data.foreach_set(
                "value", [ang.get(tuple(e.vertices), 0.0) for e in me.edges])
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        return obj


_CLASSES = (MESH_OT_corrugation_add,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
