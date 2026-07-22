
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

bl_info = {
    "name": "Space-Filling Curves",
    "author": "David Krider (Math Art project)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Space-Filling Curve",
    "description": "Hilbert and Moore curves, 2D and 3D",
    "category": "Add Curve",
}


def hilbert_points(order, dim):
    """Points of the Hilbert curve on the 2^order grid (Skilling's
    transpose algorithm), as integer tuples starting at the origin."""
    n = dim
    npts = 2 ** (order * n)
    pts = []
    for d in range(npts):
        # index -> transposed coordinates
        X = [0] * n
        for i in range(order * n):
            if d >> i & 1:
                X[n - 1 - i % n] |= 1 << (i // n)
        # Gray decode
        t = X[n - 1] >> 1
        for i in range(n - 1, 0, -1):
            X[i] ^= X[i - 1]
        X[0] ^= t
        # undo excess work
        Q = 2
        while Q != 1 << order:
            P = Q - 1
            for i in range(n - 1, -1, -1):
                if X[i] & Q:
                    X[0] ^= P
                else:
                    t = (X[0] ^ X[i]) & P
                    X[0] ^= t
                    X[i] ^= t
            Q <<= 1
        pts.append(tuple(X))
    return pts


def _axis_diff(a, b):
    """(axis, sign) of the single-coordinate difference b - a."""
    for k in range(len(a)):
        if a[k] != b[k]:
            return k, (1 if b[k] > a[k] else -1)
    raise ValueError("identical corners")


# per dimension: ring of orthants, entry sides of block 0, and each
# block's traversal axis. Solved once from the side-propagation
# constraints (each block's Hilbert may traverse any axis; its exit
# corner must sit on the face toward the next orthant).
_MOORE = {
    2: ([(0, 0), (1, 0), (1, 1), (0, 1)],
        (0, 1), (0, 0, 0, 0)),
    3: ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
        (0, 0, 1), (0, 1, 1, 0, 0, 1, 1, 0)),
}


def moore_points(order, dim):
    """Closed Moore curve: 2^dim order-(order-1) Hilbert blocks in
    the orthants of a Gray-code ring, each mapped by a signed axis
    permutation so every exit meets the next entry and the loop
    closes."""
    ring, entry0, taus = _MOORE[dim]
    if order < 2:
        return list(ring)
    s = 2 ** (order - 1)                 # sub-block grid size
    base = hilbert_points(order - 1, dim)
    alpha, _ = _axis_diff(base[0], base[-1])   # base runs along alpha
    pts = []
    sigma = list(entry0)                 # entry corner sides (0/1)
    for k in range(len(ring)):
        d_axis, d_sign = _axis_diff(ring[k],
                                    ring[(k + 1) % len(ring)])
        tau = taus[k]                    # this block's traversal axis
        origin = tuple(c * s for c in ring[k])
        # signed permutation: base axis alpha -> block axis tau
        perm = [None] * dim              # block axis -> base axis
        perm[tau] = alpha
        rest = iter(a for a in range(dim) if a != alpha)
        for a in range(dim):
            if perm[a] is None:
                perm[a] = next(rest)
        for p in base:
            pts.append(tuple(
                origin[a] + (s - 1 - p[perm[a]] if sigma[a]
                             else p[perm[a]])
                for a in range(dim)))
        # next entry sides: flip along tau (exit), cross along d_axis
        sigma[tau] = 1 - sigma[tau]
        sigma[d_axis] = 1 - (1 if d_sign > 0 else 0)
    return pts


def chaikin(pts, rounds, closed):
    """Corner-cutting smoothing (keeps endpoints of open curves)."""
    for _ in range(rounds):
        new = []
        n = len(pts)
        rng = range(n) if closed else range(n - 1)
        if not closed:
            new.append(pts[0])
        for i in rng:
            a, b = pts[i], pts[(i + 1) % n]
            new.append(tuple(0.75 * a[k] + 0.25 * b[k]
                             for k in range(3)))
            new.append(tuple(0.25 * a[k] + 0.75 * b[k]
                             for k in range(3)))
        if not closed:
            new.append(pts[-1])
        pts = new
    return pts


def build_curve(kind='HILBERT3D', order=3, size=2.0, rounds=1):
    """Returns (points3d, closed)."""
    dim = 3 if kind.endswith('3D') else 2
    if kind.startswith('HILBERT'):
        ipts = hilbert_points(order, dim)
        closed = False
    else:
        ipts = moore_points(order, dim)
        closed = True
    n = 2 ** order
    s = size / n
    off = size / 2.0 - s / 2.0
    pts = [(p[0] * s - off, p[1] * s - off,
            (p[2] * s - off) if dim == 3 else 0.0) for p in ipts]
    return chaikin(pts, rounds, closed), closed


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
                    "closed loop filling a square")],
            default='HILBERT3D')
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
            order = min(self.order, 5) if self.kind.endswith('3D') \
                else self.order
            if order != self.order:
                self.report({'WARNING'}, "3D order capped at 5")
            pts, closed = build_curve(self.kind, order, self.size,
                                      self.rounds)
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
            for k in ('kind', 'order', 'radius', 'resolution',
                      'rounds', 'size'):
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


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
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
