
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


# ------------------------------------------------------------------
# Gilbert: generalized Hilbert curve for ARBITRARY rectangular
# grids -- a port of Jakub Cerveny's gilbert2d / gilbert3d
# (github.com/jakubcerveny/gilbert, BSD-2-Clause).
# ------------------------------------------------------------------

def _sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def _gilbert2d(x, y, ax, ay, bx, by):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = _sgn(ax), _sgn(ay)
    dbx, dby = _sgn(bx), _sgn(by)
    if h == 1:
        for _ in range(w):
            yield (x, y)
            x, y = x + dax, y + day
        return
    if w == 1:
        for _ in range(h):
            yield (x, y)
            x, y = x + dbx, y + dby
        return
    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)
    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            ax2, ay2 = ax2 + dax, ay2 + day
        yield from _gilbert2d(x, y, ax2, ay2, bx, by)
        yield from _gilbert2d(x + ax2, y + ay2,
                              ax - ax2, ay - ay2, bx, by)
    else:
        if (h2 % 2) and (h > 2):
            bx2, by2 = bx2 + dbx, by2 + dby
        yield from _gilbert2d(x, y, bx2, by2, ax2, ay2)
        yield from _gilbert2d(x + bx2, y + by2, ax, ay,
                              bx - bx2, by - by2)
        yield from _gilbert2d(x + (ax - dax) + (bx2 - dbx),
                              y + (ay - day) + (by2 - dby),
                              -bx2, -by2,
                              -(ax - ax2), -(ay - ay2))


def gilbert2d(width, height):
    """Every cell of a width x height grid exactly once, with unit
    steps, for arbitrary (not just power-of-two) sizes."""
    if width >= height:
        return list(_gilbert2d(0, 0, width, 0, 0, height))
    return list(_gilbert2d(0, 0, 0, height, width, 0))


def _gilbert3d(x, y, z, ax, ay, az, bx, by, bz, cx, cy, cz):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)
    dax, day, daz = _sgn(ax), _sgn(ay), _sgn(az)
    dbx, dby, dbz = _sgn(bx), _sgn(by), _sgn(bz)
    dcx, dcy, dcz = _sgn(cx), _sgn(cy), _sgn(cz)
    if h == 1 and d == 1:
        for _ in range(w):
            yield (x, y, z)
            x, y, z = x + dax, y + day, z + daz
        return
    if w == 1 and d == 1:
        for _ in range(h):
            yield (x, y, z)
            x, y, z = x + dbx, y + dby, z + dbz
        return
    if w == 1 and h == 1:
        for _ in range(d):
            yield (x, y, z)
            x, y, z = x + dcx, y + dcy, z + dcz
        return
    ax2, ay2, az2 = ax // 2, ay // 2, az // 2
    bx2, by2, bz2 = bx // 2, by // 2, bz // 2
    cx2, cy2, cz2 = cx // 2, cy // 2, cz // 2
    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)
    if (w2 % 2) and (w > 2):
        ax2, ay2, az2 = ax2 + dax, ay2 + day, az2 + daz
    if (h2 % 2) and (h > 2):
        bx2, by2, bz2 = bx2 + dbx, by2 + dby, bz2 + dbz
    if (d2 % 2) and (d > 2):
        cx2, cy2, cz2 = cx2 + dcx, cy2 + dcy, cz2 + dcz
    if (2 * w > 3 * h) and (2 * w > 3 * d):
        # wide case: split in w only
        yield from _gilbert3d(x, y, z, ax2, ay2, az2,
                              bx, by, bz, cx, cy, cz)
        yield from _gilbert3d(x + ax2, y + ay2, z + az2,
                              ax - ax2, ay - ay2, az - az2,
                              bx, by, bz, cx, cy, cz)
    elif 3 * h > 4 * d:
        # do not split in d
        yield from _gilbert3d(x, y, z, bx2, by2, bz2,
                              cx, cy, cz, ax2, ay2, az2)
        yield from _gilbert3d(x + bx2, y + by2, z + bz2,
                              ax, ay, az,
                              bx - bx2, by - by2, bz - bz2,
                              cx, cy, cz)
        yield from _gilbert3d(x + (ax - dax) + (bx2 - dbx),
                              y + (ay - day) + (by2 - dby),
                              z + (az - daz) + (bz2 - dbz),
                              -bx2, -by2, -bz2,
                              cx, cy, cz,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2))
    elif 3 * d > 4 * h:
        # do not split in h
        yield from _gilbert3d(x, y, z, cx2, cy2, cz2,
                              ax2, ay2, az2, bx, by, bz)
        yield from _gilbert3d(x + cx2, y + cy2, z + cz2,
                              ax, ay, az, bx, by, bz,
                              cx - cx2, cy - cy2, cz - cz2)
        yield from _gilbert3d(x + (ax - dax) + (cx2 - dcx),
                              y + (ay - day) + (cy2 - dcy),
                              z + (az - daz) + (cz2 - dcz),
                              -cx2, -cy2, -cz2,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2),
                              bx, by, bz)
    else:
        # regular case: split in all of w/h/d
        yield from _gilbert3d(x, y, z, bx2, by2, bz2,
                              cx2, cy2, cz2, ax2, ay2, az2)
        yield from _gilbert3d(x + bx2, y + by2, z + bz2,
                              cx, cy, cz, ax2, ay2, az2,
                              bx - bx2, by - by2, bz - bz2)
        yield from _gilbert3d(x + (bx2 - dbx) + (cx - dcx),
                              y + (by2 - dby) + (cy - dcy),
                              z + (bz2 - dbz) + (cz - dcz),
                              ax, ay, az, -bx2, -by2, -bz2,
                              -(cx - cx2), -(cy - cy2),
                              -(cz - cz2))
        yield from _gilbert3d(x + (ax - dax) + bx2 + (cx - dcx),
                              y + (ay - day) + by2 + (cy - dcy),
                              z + (az - daz) + bz2 + (cz - dcz),
                              -cx, -cy, -cz,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2),
                              bx - bx2, by - by2, bz - bz2)
        yield from _gilbert3d(x + (ax - dax) + (bx2 - dbx),
                              y + (ay - day) + (by2 - dby),
                              z + (az - daz) + (bz2 - dbz),
                              -bx2, -by2, -bz2,
                              cx2, cy2, cz2,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2))


def gilbert3d(width, height, depth):
    """Every cell of a width x height x depth cuboid exactly once,
    with unit steps, for arbitrary sizes (even sizes recommended)."""
    if width >= height and width >= depth:
        return list(_gilbert3d(0, 0, 0, width, 0, 0,
                               0, height, 0, 0, 0, depth))
    if height >= width and height >= depth:
        return list(_gilbert3d(0, 0, 0, 0, height, 0,
                               width, 0, 0, 0, 0, depth))
    return list(_gilbert3d(0, 0, 0, 0, 0, depth,
                           width, 0, 0, 0, height, 0))


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


def build_curve(kind='HILBERT3D', order=3, size=2.0, rounds=1,
                gw=12, gh=8, gd=4):
    """Returns (points3d, closed)."""
    dim = 3 if kind.endswith('3D') else 2
    if kind.startswith('GILBERT'):
        if dim == 3:
            ipts = gilbert3d(gw, gh, gd)
            dims = (gw, gh, gd)
        else:
            ipts = gilbert2d(gw, gh)
            dims = (gw, gh, 1)
        s = size / max(dims)
        pts = [tuple((p[k] - (dims[k] - 1) / 2.0) * s
                     if k < dim else 0.0
                     for k in range(3))
               for p in (q + (0,) * (3 - len(q)) for q in ipts)]
        return chaikin(pts, rounds, False), False
    if kind.startswith('HILBERT'):
        ipts = hilbert_points(order, dim)
        n = 2 ** order
        closed = False
    else:
        ipts = moore_points(order, dim)
        n = 2 ** order
        closed = True
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
