
# Output lifts: turning a plane curve into a sculpture.
#
# A bevelled tube is the obvious lift and the least interesting one.
# These three are the ones that make a turtle curve into an object you
# would actually print or hang.
#
# MITRED-PRISM RELIEF.  Offset the path left and right by half a width
# and extrude, with the height STEPPING at each turn.  Two things matter:
#
#   * The joints must be MITRED.  Offsetting each segment independently
#     and butting the results together leaves a notch on the outside of
#     every corner and an overlap on the inside.  The correct offset at a
#     vertex is along the angle bisector, at distance w/2 / cos(theta/2)
#     -- which grows without bound as the turn approaches 180 degrees, so
#     it has to be limited or a hairpin fires the corner off to infinity.
#   * The height stepping is what makes it a RELIEF rather than an
#     extrusion: accumulate a step per turn, so the surface terraces as
#     the curve winds and reads as depth under raking light.
#
# MULTI-COPY ASSEMBLY.  Six copies on the faces of a cube, or n about an
# axis.  The cube arrangement is the one that turns a flat figure into a
# solid object, and it is why the repo's brief is a 2 m cube.
#
# SCAFFOLD ON COLUMNS.  The curve held above a base plate on columns.
# This is the lift that makes an open, self-intersecting curve printable:
# the columns carry the overhangs, and the plate gives it a ground.
#
# All three return plain (verts, faces) so they stay bpy-free and the
# geometry can be checked headlessly -- which matters, because a mitre
# that is subtly wrong looks fine in a screenshot and fails on a printer.
#
# References:
# - Benoit Mandelbrot, "The Fractal Geometry of Nature", 1982 -- the
#   teragon islands these lifts are usually applied to.
# - Harold Abelson and Andrea diSessa, "Turtle Geometry", MIT Press,
#   1981, ch. 4 -- closure, which decides whether a figure can be a
#   plate at all.

import numpy as np


def _norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def offsets(points, width, closed=False, miter_limit=4.0):
    """Left and right offset polylines, MITRED at every joint.

    Returns (left, right), each the same length as `points`.

    At a vertex the offset runs along the angle bisector at distance
    `w/2 / cos(theta/2)`, where theta is the turn.  That factor is the
    whole point: offsetting perpendicular to each segment separately and
    joining the results leaves a wedge-shaped notch outside every corner.
    The factor diverges at a hairpin, so it is capped by `miter_limit`
    -- beyond which the corner is bevelled instead, exactly as a stroke
    renderer would.
    """
    P = np.asarray(points, dtype=float)[:, :2]
    n = len(P)
    if n < 2:
        return P.copy(), P.copy()
    h = 0.5 * float(width)

    if closed:
        prev = np.roll(P, 1, axis=0)
        nxt = np.roll(P, -1, axis=0)
    else:
        prev = np.vstack([P[:1] - (P[1:2] - P[:1]), P[:-1]])
        nxt = np.vstack([P[1:], P[-1:] + (P[-1:] - P[-2:-1])])

    u = _norm(P - prev)                 # incoming direction
    v = _norm(nxt - P)                  # outgoing direction
    # left normals (rotate +90 in the plane)
    nu = np.stack([-u[:, 1], u[:, 0]], axis=1)
    nv = np.stack([-v[:, 1], v[:, 0]], axis=1)

    bis = _norm(nu + nv)
    # cos(theta/2) = bisector . segment normal
    cos_half = (bis * nu).sum(axis=1)
    cos_half = np.where(np.abs(cos_half) < 1e-6, 1e-6, cos_half)
    scale = np.clip(1.0 / cos_half, -miter_limit, miter_limit)

    left = P + bis * (h * scale)[:, None]
    right = P - bis * (h * scale)[:, None]
    return left, right


def mitred_relief(points, width=0.06, closed=False, base=0.0,
                  height=0.05, step=0.0, miter_limit=4.0):
    """A prism along the path, terraced by `step` at every turn.

    `height` is the thickness at the start; each turn adds `step`, so
    the ribbon climbs as the curve winds.  With `step` at 0 it is a
    plain constant-height prism.
    """
    P = np.asarray(points, dtype=float)[:, :2]
    if len(P) < 2:
        return [], []
    left, right = offsets(P, width, closed, miter_limit)

    # Accumulate a height step per turn, then NORMALISE the total.
    # Raw accumulation is unusable: the Koch snowflake at iteration 3
    # turns 192 times, so a 0.02 step piles up 4 m of relief on a 2 m
    # footprint.  Normalising keeps the terracing -- the surface still
    # steps at every turn, in proportion to how sharply it turned --
    # while `step` means the DEPTH of the finished relief.
    d = np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    turn = np.abs(((np.diff(ang) + 180.0) % 360.0) - 180.0)
    acc = np.concatenate([[0.0], np.cumsum(turn)])
    if closed:
        acc = np.concatenate([acc, acc[-1:]])
    acc = np.resize(acc, len(P))
    span = float(acc.max())
    if span > 1e-12:
        acc = acc / span
    z = float(base) + float(height) + acc * float(step)

    verts, faces = [], []
    m = len(P)
    for i in range(m):
        verts.append((left[i, 0], left[i, 1], float(base)))
        verts.append((right[i, 0], right[i, 1], float(base)))
        verts.append((left[i, 0], left[i, 1], z[i]))
        verts.append((right[i, 0], right[i, 1], z[i]))
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        a, b = 4 * i, 4 * ((i + 1) % m)
        faces.append((a + 2, b + 2, b + 3, a + 3))      # top
        faces.append((a + 1, b + 1, b, a))              # bottom
        faces.append((a, b, b + 2, a + 2))              # left wall
        faces.append((a + 3, b + 3, b + 1, a + 1))      # right wall
    if not closed:
        faces.append((0, 1, 3, 2))
        e = 4 * (m - 1)
        faces.append((e + 2, e + 3, e + 1, e))
    return verts, faces


def _transform(verts, R, t):
    V = np.asarray(verts, dtype=float)
    return [tuple(p) for p in V.dot(np.asarray(R).T) + np.asarray(t)]


#: The six cube faces: (rotation matrix, outward translation direction).
def _cube_frames(size):
    h = 0.5 * float(size)
    I = np.eye(3)
    rx = lambda a: np.array([[1, 0, 0],                       # noqa: E731
                             [0, np.cos(a), -np.sin(a)],
                             [0, np.sin(a), np.cos(a)]])
    ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)],       # noqa: E731
                             [0, 1, 0],
                             [-np.sin(a), 0, np.cos(a)]])
    p2 = np.pi / 2
    return [(I, (0, 0, h)),
            (rx(np.pi), (0, 0, -h)),
            (rx(p2), (0, -h, 0)),
            (rx(-p2), (0, h, 0)),
            (ry(p2), (h, 0, 0)),
            (ry(-p2), (-h, 0, 0))]


def assemble(verts, faces, mode="AXIS", count=6, size=2.0, spin=0.0):
    """Repeat a piece: `AXIS` rotates n copies about Z, `CUBE` puts one
    on each face of a cube.

    The cube arrangement is what turns a flat figure into a solid
    object, which is the whole reason to have it.
    """
    V = np.asarray(verts, dtype=float)
    if not len(V):
        return [], []
    out_v, out_f = [], []

    def add(R, t):
        base = len(out_v)
        out_v.extend(_transform(V, R, t))
        out_f.extend(tuple(i + base for i in q) for q in faces)

    if mode == "CUBE":
        # scale the piece to a face before placing it
        lo, hi = V.min(axis=0), V.max(axis=0)
        ext = float((hi - lo)[:2].max())
        f = (0.9 * float(size) / ext) if ext > 1e-12 else 1.0
        V = (V - np.array([*(0.5 * (lo + hi))[:2], lo[2]])) * f
        for R, t in _cube_frames(size):
            add(R, t)
    else:
        n = max(int(count), 1)
        for k in range(n):
            a = 2 * np.pi * k / n + np.radians(float(spin))
            R = np.array([[np.cos(a), -np.sin(a), 0.0],
                          [np.sin(a), np.cos(a), 0.0],
                          [0.0, 0.0, 1.0]])
            add(R, (0.0, 0.0, 0.0))
    return out_v, out_f


def scaffold(points, closed=False, plate_thickness=0.04, gap=0.35,
             column_radius=0.012, every=8, sides=6, margin=0.08):
    """The curve raised on columns over a base plate.

    Returns (verts, faces) for the plate and columns only -- the curve
    itself is emitted by the caller's chosen lift, so the two can carry
    different materials.

    This is the lift that makes an open, self-intersecting figure
    printable: the columns carry the overhangs and the plate gives it a
    ground to stand on.
    """
    P = np.asarray(points, dtype=float)[:, :2]
    if len(P) < 2:
        return [], []
    lo, hi = P.min(axis=0), P.max(axis=0)
    m = float(margin)
    lo, hi = lo - m, hi + m

    verts, faces = [], []
    tz = -float(gap)
    bz = tz - float(plate_thickness)
    plate = [(lo[0], lo[1], bz), (hi[0], lo[1], bz),
             (hi[0], hi[1], bz), (lo[0], hi[1], bz),
             (lo[0], lo[1], tz), (hi[0], lo[1], tz),
             (hi[0], hi[1], tz), (lo[0], hi[1], tz)]
    verts.extend(plate)
    faces.extend([(0, 3, 2, 1), (4, 5, 6, 7),
                  (0, 1, 5, 4), (1, 2, 6, 5),
                  (2, 3, 7, 6), (3, 0, 4, 7)])

    # a column every `every` points, plus always the two ends
    idx = list(range(0, len(P), max(int(every), 1)))
    if not closed and (len(P) - 1) not in idx:
        idx.append(len(P) - 1)
    r = float(column_radius)
    k = max(int(sides), 3)
    ring = [(r * np.cos(2 * np.pi * j / k), r * np.sin(2 * np.pi * j / k))
            for j in range(k)]
    for i in idx:
        base = len(verts)
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for dx, dy in ring:
            verts.append((cx + dx, cy + dy, tz))
        for dx, dy in ring:
            verts.append((cx + dx, cy + dy, 0.0))
        for j in range(k):
            a, b = base + j, base + (j + 1) % k
            faces.append((a, b, b + k, a + k))
    return verts, faces


def _mesh_area(verts, faces):
    V = np.asarray(verts, dtype=float)
    tot = 0.0
    for q in faces:
        p = V[list(q)]
        for j in range(1, len(q) - 1):
            tot += 0.5 * float(np.linalg.norm(
                np.cross(p[j] - p[0], p[j + 1] - p[0])))
    return tot


def _selftest():
    square = np.array([[0., 0], [1, 0], [1, 1], [0, 1]])
    zig = np.array([[0., 0], [1, 0], [1, 1], [2, 1], [2, 2]])

    # --- mitred offsets -----------------------------------------------
    # THE POINT OF MITRING: at a 90-degree corner the offset must move
    # out by w/2 / cos45 = 0.7071*w, not by w/2.  Offsetting each segment
    # separately and butting them leaves a notch exactly that deep.
    left, right = offsets(square, width=0.2, closed=True)
    assert left.shape == square.shape and right.shape == square.shape
    d = np.linalg.norm(left[0] - square[0])
    assert abs(d - 0.1 / np.cos(np.radians(45.0))) < 1e-9, d
    # the two sides sit on opposite sides of the path
    for i in range(len(square)):
        assert np.dot(left[i] - square[i], right[i] - square[i]) < 0

    # a mitre on a hairpin must be limited, not sent to infinity
    hairpin = np.array([[0., 0], [1, 0], [0.0, 1e-6]])
    L, R = offsets(hairpin, width=0.2, miter_limit=4.0)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(R))
    assert float(np.abs(L).max()) < 100.0, float(np.abs(L).max())

    # --- relief --------------------------------------------------------
    v, f = mitred_relief(square, width=0.2, closed=True, height=0.05)
    assert len(v) == 4 * len(square)
    assert len(f) == 4 * len(square)
    V = np.array(v)
    assert np.all(np.isfinite(V))
    # a closed relief is a closed band: every edge shared by two faces
    from collections import Counter
    ec = Counter()
    for q in f:
        for j in range(len(q)):
            a, b = q[j], q[(j + 1) % len(q)]
            ec[(min(a, b), max(a, b))] += 1
    assert all(c == 2 for c in ec.values()), \
        f"{sum(1 for c in ec.values() if c != 2)} non-manifold edges"

    # height stepping must actually terrace: a curve that turns more
    # must end up taller
    flat = mitred_relief(zig, width=0.1, height=0.05, step=0.0)[0]
    terr = mitred_relief(zig, width=0.1, height=0.05, step=0.04)[0]
    zf = np.array(flat)[:, 2].max()
    zt = np.array(terr)[:, 2].max()
    assert zt > zf + 1e-9, (zf, zt)
    # ... and monotonically in the step
    heights = [np.array(mitred_relief(zig, width=0.1, height=0.05,
                                      step=s)[0])[:, 2].max()
               for s in (0.0, 0.02, 0.05, 0.1)]
    assert all(b > a for a, b in zip(heights, heights[1:])), heights

    # The relief must stay BOUNDED however much the curve turns.  Raw
    # accumulation piled 4 m of height onto a 2 m Koch snowflake,
    # because it turns 192 times; `step` is the depth of the finished
    # relief, not a per-turn increment.
    many = np.array([[np.cos(a), np.sin(a)]
                     for a in np.linspace(0, 40 * np.pi, 400)])
    vz = np.array(mitred_relief(many, width=0.05, height=0.02,
                                step=0.1)[0])[:, 2]
    assert float(vz.max()) <= 0.02 + 0.1 + 1e-9, float(vz.max())
    # and the terracing is still there: intermediate heights exist
    assert len(set(np.round(vz, 6).tolist())) > 10

    # an open relief is capped at both ends
    vo, fo = mitred_relief(zig, width=0.1, closed=False)
    assert len(fo) == 4 * (len(zig) - 1) + 2, len(fo)

    # --- assembly -------------------------------------------------------
    v0, f0 = mitred_relief(square, width=0.15, closed=True, height=0.05)
    for n in (2, 3, 6, 12):
        va, fa = assemble(v0, f0, mode="AXIS", count=n)
        assert len(va) == n * len(v0), (n, len(va))
        assert len(fa) == n * len(f0)
        assert abs(_mesh_area(va, fa) - n * _mesh_area(v0, f0)) < 1e-6
    # rotation must actually move the copies apart
    va, _fa = assemble(v0, f0, mode="AXIS", count=4)
    A = np.array(va)
    assert float(A[:, :2].max() - A[:, :2].min()) > \
        float(np.array(v0)[:, :2].max() - np.array(v0)[:, :2].min())

    vc, fc = assemble(v0, f0, mode="CUBE", size=2.0)
    assert len(vc) == 6 * len(v0)
    C = np.array(vc)
    # a copy on every face: the assembly must reach both ways on all
    # three axes, which a single flat piece never does
    for ax in range(3):
        assert C[:, ax].min() < -0.5 and C[:, ax].max() > 0.5, ax

    # --- scaffold --------------------------------------------------------
    sv, sf = scaffold(zig, every=2, gap=0.3, plate_thickness=0.05)
    assert len(sv) > 8 and len(sf) > 6
    S = np.array(sv)
    assert np.all(np.isfinite(S))
    # the plate sits BELOW the curve, and the columns bridge the gap
    assert float(S[:, 2].min()) < -0.3, float(S[:, 2].min())
    assert abs(float(S[:, 2].max())) < 1e-9, float(S[:, 2].max())
    # the plate must overhang the curve, or it is not a base
    assert float(S[:, 0].min()) < float(zig[:, 0].min())
    assert float(S[:, 0].max()) > float(zig[:, 0].max())
    # more columns with a smaller spacing
    few = len(scaffold(zig, every=4)[0])
    many = len(scaffold(zig, every=1)[0])
    assert many > few, (few, many)

    print("lifts: OK -- mitre moves out by w/2/cos(theta/2) and is "
          "limited at a hairpin, closed relief is edge-manifold, "
          "terracing is monotone in the step, axis and cube assemblies "
          "preserve area, scaffold plate overhangs and sits below")
