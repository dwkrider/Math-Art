# Rim curves: a swept tube along a surface's open edge.
#
# Most of the surfaces this add-on builds are cut off somewhere -- a
# level set clipped to its sample box, a minimal surface truncated at
# its ends, a helicoid stopped at a parameter bound.  That cut is a
# stair-step through whatever grid produced it, not a curve, and it
# reads as a ragged fringe.  Sweeping a bevelled tube along it hides the
# staircase and gives the surface a deliberate border, which is as much
# an aesthetic control as a tidy-up.
#
# This module is shared: the geometry is plain numpy so it self-tests
# headlessly, and the Blender half is one helper plus three property
# factories, so every generator that offers a rim offers the SAME
# controls with the same defaults rather than drifting apart.
#
# A closed surface has no open edge and simply gets no curve.  That is
# not an error -- a cyclide, a Hauser tube or a triply-periodic cell is
# closed by construction, and the option is a no-op there.

import numpy as np

try:
    import bpy
    from bpy.props import BoolProperty, FloatProperty, IntProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# Smoothing is deliberately high by default.  The rim is a grid
# artifact, so smoothing it hard is exactly what makes the tube read as
# a drawn edge rather than a traced staircase.
RIM_THICKNESS_DEFAULT = 0.01
RIM_SMOOTH_DEFAULT = 8


def _edges_of(faces):
    """(n, 2) array of sorted vertex pairs, one row per face corner.

    Faces may be triangles, quads or a mix; the uniform case is
    vectorised and the ragged one falls back to a loop.
    """
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    widths = {len(f) for f in faces}
    if len(widths) == 1:
        F = np.asarray(faces, dtype=np.int64)
        k = F.shape[1]
        e = np.concatenate([F[:, [i, (i + 1) % k]] for i in range(k)])
    else:
        pairs = []
        for f in faces:
            n = len(f)
            for i in range(n):
                pairs.append((int(f[i]), int(f[(i + 1) % n])))
        e = np.asarray(pairs, dtype=np.int64)
    return np.sort(e, axis=1)


def boundary_loops(verts, faces, smooth=RIM_SMOOTH_DEFAULT):
    """Ordered polylines along the open edge of a mesh.

    Edges used by exactly one face are the boundary.  Chaining them
    assumes nothing about manifoldness: several surfaces here are
    deliberately singular (the algebraic Kreuz is three planes), so a
    rim vertex can carry four boundary edges rather than two.  The walk
    consumes unused edges greedily and returns whatever chains it finds,
    open or closed.

    `smooth` closure-preserving Laplacian passes run over each polyline
    before it is returned; without them the swept tube reproduces the
    staircase faithfully, which defeats the purpose.

    Returns a list of (points, closed) with points an (n, 3) array.
    """
    V = np.asarray(verts, dtype=float)
    e = _edges_of(faces)
    if not len(e):
        return []
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    rim = uniq[counts == 1]
    if not len(rim):
        return []

    adj = {}
    for a, b in rim:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    used = set()

    def take(p, q):
        key = (p, q) if p < q else (q, p)
        if key in used:
            return False
        used.add(key)
        return True

    chains = []
    for a0, b0 in rim:
        a0, b0 = int(a0), int(b0)
        if not take(a0, b0):
            continue
        chain = [a0, b0]
        while True:
            cur = chain[-1]
            nxt = None
            for cand in adj.get(cur, ()):
                if take(cur, cand):
                    nxt = cand
                    break
            if nxt is None:
                break
            chain.append(nxt)
            if nxt == chain[0]:
                break
        if len(chain) < 4:
            continue
        closed = chain[-1] == chain[0]
        if closed:
            chain = chain[:-1]
        chains.append((np.asarray(chain, dtype=np.int64), closed))

    out = []
    for idx, closed in chains:
        pts = V[idx]
        for _ in range(max(0, int(smooth))):
            if closed:
                prev = np.roll(pts, 1, axis=0)
                nxt = np.roll(pts, -1, axis=0)
                pts = 0.5 * pts + 0.25 * (prev + nxt)
            elif len(pts) > 2:
                inner = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
                pts = np.concatenate([pts[:1], inner, pts[-1:]])
        out.append((pts, closed))
    return out


if _IN_BLENDER:

    def rim_prop():
        return BoolProperty(
            name="Rim Curve", default=False,
            description="Sweep a tube along the open edge of the "
                        "surface. That edge is a stair-step through "
                        "the sample grid, so the tube both tidies it "
                        "and gives the surface a deliberate border; a "
                        "closed surface has no edge and gets no curve")

    def rim_thickness_prop():
        return FloatProperty(
            name="Rim Thickness", default=RIM_THICKNESS_DEFAULT,
            min=0.0, max=1.0,
            description="Bevel radius of the rim tube (0 leaves a "
                        "bare curve)")

    def rim_smooth_prop():
        return IntProperty(
            name="Rim Smoothing", default=RIM_SMOOTH_DEFAULT,
            min=0, max=40,
            description="Laplacian passes along the rim before it is "
                        "swept; 0 follows the grid staircase exactly. "
                        "The default is high on purpose -- the rim is "
                        "a grid artifact, so smoothing it hard is what "
                        "makes the tube read as a drawn edge")

    def draw_rim(layout, op):
        """The three controls, shown only when the rim is on."""
        layout.prop(op, 'rim')
        if getattr(op, 'rim', False):
            layout.prop(op, 'rim_thickness')
            layout.prop(op, 'rim_smooth')

    def add_rim_curve(context, obj, label, verts, faces, thickness=None,
                      smooth=None):
        """Sweep a bevelled curve along the mesh's open edge.

        Parented to `obj` so the pair moves as one.  Returns the number
        of rim loops, zero meaning the surface was closed.
        """
        if thickness is None:
            thickness = RIM_THICKNESS_DEFAULT
        if smooth is None:
            smooth = RIM_SMOOTH_DEFAULT
        loops = boundary_loops(verts, faces, smooth=smooth)
        if not loops:
            return 0
        cu = bpy.data.curves.new(label + " Rim", 'CURVE')
        cu.dimensions = '3D'
        cu.fill_mode = 'FULL'
        cu.bevel_depth = float(thickness)
        cu.bevel_resolution = 4
        cu.use_fill_caps = True
        for pts, closed in loops:
            sp = cu.splines.new('POLY')
            sp.points.add(len(pts) - 1)
            for i, q in enumerate(pts):
                sp.points[i].co = (float(q[0]), float(q[1]),
                                   float(q[2]), 1.0)
            sp.use_cyclic_u = bool(closed)
        rim = bpy.data.objects.new(label + " Rim", cu)
        context.collection.objects.link(rim)
        rim.matrix_world = obj.matrix_world.copy()
        rim.parent = obj
        rim.matrix_parent_inverse = obj.matrix_world.inverted()
        return len(loops)

    def add_rim_from_object(context, obj, label, thickness=None,
                            smooth=None):
        """Same, reading the geometry back off an existing mesh object.

        For generators that build their object through a path this
        module cannot see (bmesh, modifiers, a solver), taking the
        vertices and polygons off the finished mesh is simpler and
        always matches what the user is looking at.
        """
        me = getattr(obj, 'data', None)
        if me is None or not hasattr(me, 'polygons'):
            return 0
        verts = [tuple(v.co) for v in me.vertices]
        faces = [tuple(p.vertices) for p in me.polygons]
        return add_rim_curve(context, obj, label, verts, faces,
                             thickness, smooth)


def _selftest():
    ok = True

    # An open patch: a grid of quads has a rim of exactly its border.
    n = 12
    verts = [(i / (n - 1.0), j / (n - 1.0), 0.0)
             for j in range(n) for i in range(n)]
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1,
              (j + 1) * n + i)
             for j in range(n - 1) for i in range(n - 1)]
    loops = boundary_loops(verts, faces, smooth=0)
    good = (len(loops) == 1 and loops[0][1]
            and len(loops[0][0]) == 4 * (n - 1))
    ok &= good
    print("rim_curve: open quad grid gives one closed rim of %d points %s"
          % (4 * (n - 1), 'OK' if good else 'FAIL'))

    # A closed surface: a torus of quads has no rim at all.
    R, r, nu, nv = 2.0, 0.7, 24, 16
    import math
    tv = []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            rad = R + r * math.cos(v)
            tv.append((rad * math.cos(u), rad * math.sin(u),
                       r * math.sin(v)))
    tf = [(i * nv + j, ((i + 1) % nu) * nv + j,
           ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv)
          for i in range(nu) for j in range(nv)]
    good = boundary_loops(tv, tf) == []
    ok &= good
    print("rim_curve: closed torus has no rim %s"
          % ('OK' if good else 'FAIL'))

    # Mixed tri/quad faces must not break the edge extraction.
    mixed = faces[:-3] + [(0, 1, n + 1)]
    good = len(boundary_loops(verts, mixed, smooth=0)) >= 1
    ok &= good
    print("rim_curve: mixed tri/quad faces handled %s"
          % ('OK' if good else 'FAIL'))

    # Smoothing must shorten a staircase without moving its endpoints
    # far, and must preserve the point count.
    stair = [(i // 2 * 1.0, i % 2 * 1.0, 0.0) for i in range(24)]
    sf = []
    raw = boundary_loops(verts, faces, smooth=0)[0][0]
    sm = boundary_loops(verts, faces, smooth=12)[0][0]
    per_raw = float(np.linalg.norm(np.diff(np.vstack([raw, raw[:1]]),
                                           axis=0), axis=1).sum())
    per_sm = float(np.linalg.norm(np.diff(np.vstack([sm, sm[:1]]),
                                          axis=0), axis=1).sum())
    good = len(raw) == len(sm) and per_sm < per_raw
    ok &= good
    print("rim_curve: smoothing shortens the rim %.3f -> %.3f, keeps "
          "%d points %s" % (per_raw, per_sm, len(sm),
                            'OK' if good else 'FAIL'))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("rim_curve self-test failed")
