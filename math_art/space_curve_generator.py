
# Space-Filling Curve Generator for Blender
#
# Hilbert and Moore curves in 2D and 3D (after the Wolfram
# demonstration "Hilbert and Moore 3D Fractal Curves"). The Hilbert
# points come from Skilling's transpose algorithm; the Moore curve is
# the closed variant: 4 (2D) or 8 (3D) rotated Hilbert blocks whose
# ends chain around a Gray-code ring into a single closed loop.
#
# Output is a poly curve with bevel (optionally Chaikin-rounded
# corners) or a plain mesh wire.
#
# References:
# - Hilbert curve: David Hilbert, "Ueber die stetige Abbildung einer
#   Linie auf ein Flaechenstueck", Math. Ann. 38, 1891, pp. 459-460.
# - Moore (closed) curve: E. H. Moore, "On certain crinkly curves",
#   Trans. Amer. Math. Soc. 1, 1900, pp. 72-90.
# - Hilbert index/transpose algorithm: John Skilling, "Programming the
#   Hilbert curve", AIP Conf. Proc. 707, 2004, p. 381.
# - Corner rounding: G. Chaikin, "An algorithm for high-speed curve
#   generation", Computer Graphics and Image Processing 3, 1974,
#   pp. 346-349.

bl_info = {
    "name": "Space-Filling Curves",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Space-Filling Curve",
    "description": "Hilbert and Moore curves, 2D and 3D",
    "category": "Add Curve",
}
# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.curves import (build_curve, gilbert2d, gilbert3d,
                                 hilbert_points, moore_points)
except ImportError:  # flat import outside the package
    from ifs.curves import (build_curve, gilbert2d, gilbert3d,
                                hilbert_points, moore_points)












# ------------------------------------------------------------------
# Gilbert: generalized Hilbert curve for ARBITRARY rectangular
# grids -- a port of Jakub Cerveny's gilbert2d / gilbert3d
# (github.com/jakubcerveny/gilbert, BSD-2-Clause).
# ------------------------------------------------------------------















try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class CURVE_OT_space_curve_add(bpy.types.Operator):
        """Add a space-filling curve (Hilbert / Moore, 2D / 3D)"""
        bl_idname = "curve.space_filling_add"
        bl_label = "Space-Filling Curve"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Curve",
            items=[('HILBERT3D', "Hilbert 3D", "open, fills a cube"),
                   ('MOORE3D', "Moore 3D",
                    "closed loop filling a cube"),
                   ('HILBERT2D', "Hilbert 2D",
                    "open, fills a square"),
                   ('MOORE2D', "Moore 2D",
                    "closed loop filling a square"),
                   ('GILBERT3D', "Gilbert 3D",
                    "generalized Hilbert curve filling an "
                    "arbitrary W x H x D box (Cerveny)"),
                   ('GILBERT2D', "Gilbert 2D",
                    "generalized Hilbert curve filling an "
                    "arbitrary W x H rectangle (Cerveny)")],
            default='HILBERT3D')
        gw: IntProperty(name="Cells W", default=12, min=1, max=64,
                        description="Gilbert grid width")
        gh: IntProperty(name="Cells H", default=8, min=1, max=64,
                        description="Gilbert grid height")
        gd: IntProperty(name="Cells D", default=4, min=1, max=64,
                        description="Gilbert grid depth (even "
                                    "sizes recommended in 3D)")
        order: IntProperty(
            name="Order", default=3, min=1, max=6,
            description="Recursion depth; 3D point count is 8^order "
                        "(3D capped at 5)")
        radius: FloatProperty(
            name="Tube Radius", default=0.03, min=0.0, max=0.5,
            step=1, precision=3,
            description="Curve bevel depth (0 = wire only)")
        rounds: IntProperty(
            name="Corner Rounding", default=1, min=0, max=3,
            description="Chaikin corner-cutting passes")
        resolution: IntProperty(name="Bevel Resolution", default=4,
                                min=1, max=12)
        size: FloatProperty(name="Size", default=2.0, min=0.05,
                            max=100.0)

        def execute(self, context):
            order = self.order
            if self.kind in ('HILBERT3D', 'MOORE3D'):
                order = min(order, 5)
                if order != self.order:
                    self.report({'WARNING'}, "3D order capped at 5")
            pts, closed = build_curve(self.kind, order, self.size,
                                      self.rounds, self.gw,
                                      self.gh, self.gd)
            cu = bpy.data.curves.new("SpaceCurve", 'CURVE')
            cu.dimensions = '3D'
            sp = cu.splines.new('POLY')
            sp.points.add(len(pts) - 1)
            for i, p in enumerate(pts):
                sp.points[i].co = (p[0], p[1], p[2], 1.0)
            sp.use_cyclic_u = closed
            cu.bevel_depth = self.radius
            cu.bevel_resolution = self.resolution
            if self.radius > 0:
                cu.use_fill_caps = not closed
            obj = bpy.data.objects.new("SpaceCurve", cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{len(pts)} points"
                        + (", closed" if closed else ""))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind.startswith('GILBERT'):
                lay.prop(self, 'gw')
                lay.prop(self, 'gh')
                if self.kind == 'GILBERT3D':
                    lay.prop(self, 'gd')
            else:
                lay.prop(self, 'order')
            for k in ('radius', 'resolution', 'rounds', 'size'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("curve.space_filling_add",
                             icon='CURVE_DATA')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_space_curve_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_space_curve_add)


def _selftest():
    def unit_steps(pts, closed):
        n = len(pts)
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            a, b = pts[i], pts[(i + 1) % n]
            if sum(abs(a[k] - b[k]) for k in range(len(a))) != 1:
                return False
        return True
    for dim in (2, 3):
        for order in (1, 2, 3, 4):
            p = hilbert_points(order, dim)
            ok = (len(p) == 2 ** (order * dim)
                  and len(set(p)) == len(p)
                  and unit_steps(p, False))
            print(f"hilbert{dim}d o{order}: {len(p)} pts "
                  f"{'OK' if ok else 'BAD'}")
    for dim in (2, 3):
        for order in (2, 3, 4):
            p = moore_points(order, dim)
            ok = (len(p) == 2 ** (order * dim)
                  and len(set(p)) == len(p)
                  and unit_steps(p, True))
            print(f"moore{dim}d o{order}: {len(p)} pts closed "
                  f"{'OK' if ok else 'BAD'}")
    for w, h in ((5, 3), (12, 8), (7, 11), (16, 6), (1, 9)):
        p = gilbert2d(w, h)
        ok = (len(p) == w * h and len(set(p)) == len(p)
              and unit_steps(p, False))
        print(f"gilbert2d {w}x{h}: {len(p)} pts "
              f"{'OK' if ok else 'BAD'}")
        assert ok
    # 3D: full coverage always; unit steps guaranteed only for
    # even sizes (odd sizes make a few diagonal hops -- a
    # documented property of the algorithm)
    for w, h, d in ((4, 4, 4), (12, 8, 4), (6, 10, 2),
                    (8, 6, 6), (5, 4, 3), (16, 2, 2)):
        p = gilbert3d(w, h, d)
        even = w % 2 == 0 and h % 2 == 0 and d % 2 == 0
        ok = (len(p) == w * h * d and len(set(p)) == len(p)
              and (unit_steps(p, False) or not even))
        print(f"gilbert3d {w}x{h}x{d}: {len(p)} pts "
              f"{'OK' if ok else 'BAD'}")
        assert ok
    print("space curve standalone tests passed")
