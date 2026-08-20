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

import math

import numpy as np

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty,
                           FloatProperty, IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# Smoothing is deliberately LOW.  The visual rounding is done by the
# Bezier handles, which interpolate the rim points instead of moving
# them, so the smoother only has to take the zigzag out of a genuinely
# stair-stepped edge.  Asking it to do the rounding as well is what
# pulled the curve off the surface in the first place.
RIM_THICKNESS_DEFAULT = 0.01
RIM_SMOOTH_DEFAULT = 3


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


def boundary_index_loops(faces):
    """The rim as chains of VERTEX INDICES, before any smoothing.

    Separate from `boundary_loops` because a caller that wants to
    refresh an existing curve after the mesh moves needs the indices,
    not the positions -- the Seifert generator does exactly that after
    it minimises a surface.

    Returns a list of (index array, closed).
    """
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
    return chains


def boundary_loops(verts, faces, smooth=RIM_SMOOTH_DEFAULT):
    """Ordered polylines along the open edge of a mesh.

    Edges used by exactly one face are the boundary.  Chaining them
    assumes nothing about manifoldness: several surfaces here are
    deliberately singular (the algebraic Kreuz is three planes), so a
    rim vertex can carry four boundary edges rather than two.  The walk
    consumes unused edges greedily and returns whatever chains it finds,
    open or closed.

    `smooth` Taubin passes run over each polyline before it is
    returned; without them the swept tube reproduces the staircase
    faithfully, which defeats the purpose, and with a shrinking
    smoother it would leave the edge instead.

    Returns a list of (points, closed) with points an (n, 3) array.
    """
    V = np.asarray(verts, dtype=float)
    chains = boundary_index_loops(faces)
    out = []
    for idx, closed in chains:
        out.append((_taubin(V[idx], closed, int(smooth)), closed))
    return out


# Taubin's lambda/mu pair.  mu is slightly larger in magnitude than
# lambda and negative, so each shrinking pass is followed by an
# expanding one; the pair removes high-frequency wobble while leaving
# the curve where it was.
_TAUBIN_LAMBDA = 0.5
_TAUBIN_MU = -0.53


def _taubin(pts, closed, passes):
    """Smooth a polyline WITHOUT shrinking it.

    A plain Laplacian pass moves every point toward the midpoint of its
    neighbours, which is a curve-shortening flow: on a rim that wraps a
    curved surface it walks the curve off the edge it is supposed to
    trace, visibly so after a few passes.  Taubin's fix is to alternate
    a positive step with a slightly larger negative one, which cancels
    the shrinkage to first order while still attenuating the
    grid staircase.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if passes <= 0 or n < 3:
        return pts

    def step(p, w):
        if closed:
            lap = 0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) - p
            return p + w * lap
        q = p.copy()
        q[1:-1] = p[1:-1] + w * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
        return q

    orig = pts
    for _ in range(passes):
        pts = step(pts, _TAUBIN_LAMBDA)
        pts = step(pts, _TAUBIN_MU)

    # Cap how far any point may travel -- but scale the budget to how
    # ragged the curve actually is, not to its point spacing.
    #
    # A fixed fraction of the spacing cannot serve both cases here, and
    # trying it broke one to fix the other.  A marching-tetrahedra rim
    # zigzags by about a grid cell perpendicular to its own direction,
    # so flattening it needs a budget of that order; a woven-polyhedron
    # rim is a coarse polygon with genuine corners, where the same
    # budget lets the smoother cut them and lift off the surface.
    #
    # The discriminator is the high-frequency content itself.  The
    # Laplacian residual |0.5(prev + next) - p| is exactly the local
    # zigzag amplitude: large on a staircase, near zero on a smooth
    # polygon however sharply it turns overall.  Twice its median is
    # therefore enough to flatten the ragged case and almost nothing in
    # the coarse one.
    if closed:
        lap0 = 0.5 * (np.roll(orig, 1, axis=0)
                      + np.roll(orig, -1, axis=0)) - orig
    else:
        lap0 = np.zeros_like(orig)
        lap0[1:-1] = 0.5 * (orig[:-2] + orig[2:]) - orig[1:-1]
    rough = float(np.median(np.linalg.norm(lap0, axis=1)))
    cap = 2.0 * rough
    if cap > 0.0:
        d = pts - orig
        dist = np.linalg.norm(d, axis=1)
        over = dist > cap
        if np.any(over):
            d[over] *= (cap / dist[over])[:, None]
            pts = orig + d
    return pts



def resample(pts, closed, spacing):
    """Re-space a polyline by arc length.

    A swept tube self-intersects wherever the curve turns inside its own
    bevel radius, and a rim traced off a mesh has points spaced by the
    grid, not by the tube.  At a thickness of 0.04 on a rim whose points
    sit 0.005 apart, every small wiggle folds the sweep over itself and
    the tube comes out lumpy -- the failure looks like a caterpillar
    rather than a pipe.  Re-spacing the control points to roughly the
    tube diameter removes the cause instead of hiding it.
    """
    P = np.asarray(pts, dtype=float)
    if spacing <= 0.0 or len(P) < 3:
        return P
    Q = np.vstack([P, P[:1]]) if closed else P
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0:
        return P
    n = int(round(total / spacing))
    n = max(8, min(n, len(P)))          # never ADD detail, only thin it
    t = (np.linspace(0.0, total, n, endpoint=False) if closed
         else np.linspace(0.0, total, n))
    out = np.empty((len(t), 3))
    for k in range(3):
        out[:, k] = np.interp(t, s, Q[:, k])

    # Equal steps in ARC LENGTH are not equal steps in space: where the
    # rim doubles back on itself the path advances while the point
    # barely moves, so a few gaps come out far shorter than the target
    # and the tube folds there anyway.  Drop those directly.
    keep = [0]
    lo = 0.5 * spacing
    for i in range(1, len(out)):
        if float(np.linalg.norm(out[i] - out[keep[-1]])) >= lo:
            keep.append(i)
    if closed and len(keep) > 2:
        if float(np.linalg.norm(out[keep[-1]] - out[keep[0]])) < lo:
            keep.pop()
    return out[keep] if len(keep) >= 4 else out


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
            description="Taubin smoothing passes along the rim before "
                        "it is swept. Unlike a plain Laplacian this "
                        "does not shrink the curve, so the tube stays "
                        "on the edge however many passes you use; 0 "
                        "follows the sample grid exactly")

    def rim_profile_prop():
        return EnumProperty(
            name="Rim Profile",
            items=[('ROUND', "Circular",
                    "Round tube -- the curve's own bevel depth"),
                   ('SQUARE', "Square",
                    "Square tube, swept from a four-point bevel "
                    "object that is created alongside and hidden")],
            default='ROUND',
            description="Cross-section swept along the rim")

    def _square_bevel(name, half):
        """A closed square used as a curve's bevel object.

        Blender sweeps a round tube from `bevel_depth` alone, but any
        other cross-section has to be an actual curve object, so one is
        made per rim and hidden.  It is parented to the rim so deleting
        the surface takes the whole assembly with it.
        """
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '2D'
        sp = cu.splines.new('POLY')
        sp.points.add(3)
        for i, (x, y) in enumerate(((-half, -half), (half, -half),
                                    (half, half), (-half, half))):
            sp.points[i].co = (x, y, 0.0, 1.0)
        sp.use_cyclic_u = True
        return bpy.data.objects.new(name, cu)

    def draw_rim(layout, op):
        """The three controls, shown only when the rim is on."""
        layout.prop(op, 'rim')
        if getattr(op, 'rim', False):
            layout.prop(op, 'rim_thickness')
            if hasattr(op, 'rim_profile'):
                layout.prop(op, 'rim_profile')
            layout.prop(op, 'rim_smooth')

    def add_rim_curve(context, obj, label, verts, faces, thickness=None,
                      smooth=None, profile='ROUND'):
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
        # space the control points to the tube, not to the sample grid
        loops = [(resample(pts, closed, 1.6 * float(thickness)), closed)
                 for pts, closed in loops]
        loops = [(p, c) for p, c in loops if len(p) >= 4]
        if not loops:
            return 0
        cu = bpy.data.curves.new(label + " Rim", 'CURVE')
        cu.dimensions = '3D'
        cu.fill_mode = 'FULL'
        cu.use_fill_caps = True
        if profile == 'SQUARE':
            bev = _square_bevel(label + " Rim Profile", float(thickness))
            context.collection.objects.link(bev)
            bev.hide_viewport = True
            bev.hide_render = True
            cu.bevel_mode = 'OBJECT'
            cu.bevel_object = bev
        else:
            bev = None
            cu.bevel_depth = float(thickness)
            cu.bevel_resolution = 4
        # BEZIER with AUTO handles, not POLY: the handles round the
        # corners between samples without moving the samples, so the
        # curve still passes exactly through the rim it was built from.
        # A POLY spline would render the staircase; a NURBS one would
        # approximate rather than interpolate and drift off the edge in
        # the same way the old shrinking smoother did.
        cu.resolution_u = 6
        for pts, closed in loops:
            sp = cu.splines.new('BEZIER')
            sp.bezier_points.add(len(pts) - 1)
            for i, q in enumerate(pts):
                bp = sp.bezier_points[i]
                bp.co = (float(q[0]), float(q[1]), float(q[2]))
                bp.handle_left_type = 'AUTO'
                bp.handle_right_type = 'AUTO'
            sp.use_cyclic_u = bool(closed)
        rim = bpy.data.objects.new(label + " Rim", cu)
        context.collection.objects.link(rim)
        if profile == 'SQUARE':
            # A square tube shaded smooth is a round tube: the shading
            # averages across the four corners and erases the only
            # thing that makes the profile square.  Splitting edges
            # above a threshold creases those corners while leaving the
            # tube smooth ALONG its length, which is what flat shading
            # would throw away.  30 degrees clears the 90-degree
            # corners comfortably and stays well above the angle
            # between successive segments of a swept curve.
            sharp = rim.modifiers.new("Sharpen", 'EDGE_SPLIT')
            sharp.split_angle = math.radians(30.0)
            sharp.use_edge_angle = True
            sharp.use_edge_sharp = False
        rim.matrix_world = obj.matrix_world.copy()
        rim.parent = obj
        rim.matrix_parent_inverse = obj.matrix_world.inverted()
        if bev is not None:
            bev.parent = rim
        return len(loops)

    def add_rim_from_object(context, obj, label, thickness=None,
                            smooth=None, profile=None):
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
        if profile is None:
            profile = 'ROUND'
        return add_rim_curve(context, obj, label, verts, faces,
                             thickness, smooth, profile)


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

    # Smoothing must NOT shrink the rim.  This is the property the
    # first implementation got wrong: a plain Laplacian is a
    # curve-shortening flow, so on a rim that wraps a curved surface it
    # migrated off the edge, visibly, at the default number of passes.
    # A square loop is the sharpest test -- Taubin should hold its
    # perimeter where a Laplacian collapses it toward the centroid.
    raw = boundary_loops(verts, faces, smooth=0)[0][0]
    sm = boundary_loops(verts, faces, smooth=20)[0][0]

    def perim(p):
        return float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]),
                                            axis=0), axis=1).sum())

    shrink = 1.0 - perim(sm) / perim(raw)
    drift = float(np.max(np.linalg.norm(sm - raw, axis=1)))
    good = len(raw) == len(sm) and abs(shrink) < 0.06 and drift < 0.12
    ok &= good
    print("rim_curve: 20 smoothing passes shrink the rim %.1f%% and "
          "move it at most %.3f %s"
          % (100.0 * shrink, drift, 'OK' if good else 'FAIL'))

    # And for contrast, the shrinking flow it replaced: same loop, same
    # pass count, run as a pure Laplacian.
    lap = raw.copy()
    for _ in range(20):
        lap = 0.5 * lap + 0.25 * (np.roll(lap, 1, axis=0)
                                  + np.roll(lap, -1, axis=0))
    print("rim_curve:   (a plain Laplacian would shrink it %.1f%% and "
          "move it %.3f)"
          % (100.0 * (1.0 - perim(lap) / perim(raw)),
             float(np.max(np.linalg.norm(lap - raw, axis=1)))))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("rim_curve self-test failed")
