# Plane sections of a triangle mesh: the front half of the fabrication
# slicer.
#
# Slicing a mesh with a plane is elementary; slicing it so the answer is
# TRUSTWORTHY is where the work is, and three decisions carry that.
#
# 1. WELD BY EDGE IDENTITY, NOT BY COORDINATE.  Every crossing point
#    lies on one specific mesh edge, so the pair (min(i,j), max(i,j))
#    names it exactly.  Two triangles sharing that edge produce the same
#    crossing with the same name -- no distance test, no quantisation
#    grid, no "points closer than epsilon are the same point" guess.
#    The chaining that follows is therefore combinatorial and exact, and
#    a vertex whose neighbours disagree is a genuine topological fault
#    rather than a tolerance that wanted tuning.  (The reference free
#    add-on hashes coordinates and then chains with an O(n^2) scan; both
#    of those problems disappear here.)
#
# 2. HALF-OPEN SIGN CONVENTION.  A vertex exactly on the plane is
#    counted as BELOW it (`above = s > 0`).  That single rule makes the
#    crossing count per triangle exactly 0 or 2 -- never 1, never 3 --
#    so a triangle that merely grazes the plane at one vertex
#    contributes nothing instead of a spurious dangling segment.
#
# 3. THE JITTERED OFFSET IS THE OFFSET.  Coplanar faces defeat any sign
#    convention, so each plane is nudged off the exact value and, if
#    chaining still leaves dangling ends, re-nudged.  Whatever offset
#    finally worked is COMMITTED and returned: every later stage --
#    crossing lines, slots, the 3-D preview, the exported sheets --
#    reads that number.  Slicing both families to completion before any
#    slot is computed is what keeps the two halves of a joint talking
#    about the same plane.
#
# Numpy for the per-vertex arithmetic, plain Python for the chaining.
# No bpy: this self-tests headlessly.

import math

import numpy as np


class SectionFault(Exception):
    """The mesh is not a closed surface at this height, so the section
    is not a set of closed outlines."""


def plane_frame(normal):
    """An orthonormal (u, v) spanning the plane, with u x v = normal.

    Built once per family and shared by every plane in it -- slices in
    different frames would not stack up.
    """
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    # pick the axis least aligned with n, so the cross product is stable
    a = np.zeros(3)
    a[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(a, n)
    u /= (np.linalg.norm(u) or 1.0)
    v = np.cross(n, u)
    return u, v, n


def triangulate(faces):
    """Fan-triangulate a mixed polygon list into triangles."""
    tris = []
    for f in faces:
        if len(f) < 3:
            continue
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    return tris


def _bucket(tmin, tmax, offsets, margin=0.0):
    """For each plane, the triangles whose span straddles it.

    `margin` widens every triangle's span before bucketing.  It has to:
    the buckets are built from the NOMINAL offsets but membership is
    later decided against the NUDGED ones, so a triangle whose extreme
    lands exactly on a nominal plane would otherwise be filed under no
    plane at all and its section would come apart -- a crossing point of
    degree one.  Meshes with vertices exactly on a plane are not a rare
    case; a sphere built from an icosahedron has a whole equator of
    them.  Widening by more than the largest nudge makes the bucket a
    conservative superset, which is all it needs to be.
    """
    order = np.argsort(offsets)
    sorted_off = np.asarray(offsets, dtype=float)[order]
    lo = np.searchsorted(sorted_off, tmin - margin, side='right')
    hi = np.searchsorted(sorted_off, tmax + margin, side='left')
    buckets = [[] for _ in offsets]
    for t in range(len(tmin)):
        for k in range(lo[t], hi[t]):
            buckets[order[k]].append(t)
    return buckets


def _chain(segments):
    """Ordered closed loops from unordered (keyA, keyB) segments.

    Every crossing point must be shared by exactly two segments -- one
    arriving, one leaving.  Any other degree means the surface was not
    closed here, which is reported rather than papered over.
    """
    adj = {}
    for a, b in segments:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for key, nbrs in adj.items():
        if len(nbrs) != 2:
            raise SectionFault(
                f"crossing point of degree {len(nbrs)} (not a closed "
                f"surface at this height)")

    loops = []
    unseen = set(adj)
    while unseen:
        start = next(iter(unseen))
        loop = [start]
        unseen.discard(start)
        prev, cur = None, start
        while True:
            a, b = adj[cur]
            nxt = a if a != prev else b
            if nxt == start:
                break
            if nxt not in unseen:
                raise SectionFault("section graph is not a union of "
                                   "simple cycles")
            loop.append(nxt)
            unseen.discard(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def section_at(V, tris, n, d, u, v, tri_ids=None):
    """One plane section.  Returns a list of loops, each a list of
    (x, y) in the plane's own (u, v) frame.

    Raises SectionFault if the section does not close up.
    """
    s = V @ n - d
    above = s > 0.0                      # on-plane counts as below

    pts = {}                             # edge key -> (x, y)
    segments = []
    ids = range(len(tris)) if tri_ids is None else tri_ids
    for t in ids:
        i, j, k = tris[t]
        ai, aj, ak = above[i], above[j], above[k]
        cnt = int(ai) + int(aj) + int(ak)
        if cnt == 0 or cnt == 3:
            continue
        ends = []
        for p, q in ((i, j), (j, k), (k, i)):
            if above[p] == above[q]:
                continue
            key = (p, q) if p < q else (q, p)
            if key not in pts:
                sp, sq = s[p], s[q]
                f = sp / (sp - sq)
                P = V[p] + f * (V[q] - V[p])
                rel = P - d * n
                pts[key] = (float(rel @ u), float(rel @ v))
            ends.append(key)
        if len(ends) != 2:
            raise SectionFault(
                f"triangle produced {len(ends)} crossings (expected 2)")
        segments.append((ends[0], ends[1]))

    if not segments:
        return []
    return [[pts[key] for key in loop] for loop in _chain(segments)]


def section_family(verts, faces, normal, offsets, jitter=None,
                   tries=4):
    """Section a mesh by a family of parallel planes.

    Returns (slices, planes) where `slices[k]` is the list of loops for
    plane k and `planes[k]` is its COMMITTED offset -- the nudged value
    that actually worked, which every later stage must use.

    A plane that cannot be made to close after `tries` nudges yields an
    empty slice and is recorded in `faults`, reachable as an attribute
    on the returned list.  Slicing never raises for one bad layer: one
    unusable height out of sixty is a report, not a dead operator.
    """
    V = np.asarray(verts, dtype=float)
    tris = faces if (faces and len(faces[0]) == 3) else triangulate(faces)
    tris = [tuple(t) for t in tris]
    u, v, n = plane_frame(normal)

    T = np.asarray(tris, dtype=np.int64)
    proj = V @ n
    tmin = proj[T].min(axis=1)
    tmax = proj[T].max(axis=1)

    offsets = [float(o) for o in offsets]
    if jitter is None:
        span = float(proj.max() - proj.min()) or 1.0
        jitter = 1e-7 * span
    # deterministic nudges: same input, same output, every run
    nudges = (0.5, -1.37, 2.11, -3.29)
    margin = jitter * max(abs(x) for x in nudges) * 2.0

    buckets = _bucket(tmin, tmax, offsets, margin)

    slices, committed, faults = [], [], []
    for k, d0 in enumerate(offsets):
        got, used = None, d0
        for a in range(tries):
            d = d0 + nudges[a % len(nudges)] * jitter
            # re-bucket only when the nudge could change membership
            ids = [t for t in buckets[k] if tmin[t] < d < tmax[t]]
            if not ids:
                ids = [t for t in range(len(tris))
                       if tmin[t] < d < tmax[t]]
            try:
                got = section_at(V, tris, n, d, u, v, ids)
                used = d
                break
            except SectionFault as exc:
                last = exc
        if got is None:
            faults.append((k, str(last)))
            got = []
        slices.append(got)
        committed.append(used)

    slices = list(slices)
    return slices, committed, faults, (u, v, n)


def section_planes(verts, faces, planes, jitter=None, tries=4):
    """Section a mesh by planes that need NOT be parallel and need not
    share a frame.

    `planes` is a sequence of (normal, offset, u, v).  The frame is
    ORTHONORMALISED rather than trusted: u is projected off the normal
    and rescaled, and v is re-derived as n x u.  A caller who passes a
    v of length sqrt(2) -- easy to do when writing an axis pair by hand
    -- would otherwise get every part in that plane stretched by 41%
    along one direction, with nothing anywhere to complain about it.
    On a fabrication tool that is a scrapped sheet, so the frame is
    made correct here instead of being assumed correct.

    Radial fins and
    ribs along a curve both need this: every plane has its own normal,
    so there is no shared projection to bucket triangles by and each
    plane is simply done in turn.  Parallel stacks should keep using
    `section_family`, which does bucket and is much faster for the
    sixty-plane case.

    Returns (slices, committed_offsets, faults) exactly as
    `section_family` does, so callers can treat the two alike.
    """
    V = np.asarray(verts, dtype=float)
    tris = faces if (faces and len(faces[0]) == 3) else triangulate(faces)
    tris = [tuple(t) for t in tris]
    T = np.asarray(tris, dtype=np.int64)

    if jitter is None:
        lo = V.min(axis=0)
        hi = V.max(axis=0)
        jitter = 1e-7 * (float(np.linalg.norm(hi - lo)) or 1.0)
    nudges = (0.5, -1.37, 2.11, -3.29)

    slices, committed, faults = [], [], []
    for k, (normal, offset, u, v) in enumerate(planes):
        n = np.asarray(normal, dtype=float)
        n = n / (np.linalg.norm(n) or 1.0)
        u = np.asarray(u, dtype=float)
        u = u - n * float(u @ n)                  # drop any out-of-plane part
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:                            # u was parallel to n
            u, v, n = plane_frame(n)
        else:
            u = u / nu
            v = np.cross(n, u)                    # so that u x v = n
        proj = V @ n
        tmin = proj[T].min(axis=1)
        tmax = proj[T].max(axis=1)

        got, used, last = None, float(offset), None
        for a in range(tries):
            d = float(offset) + nudges[a % len(nudges)] * jitter
            ids = np.nonzero((tmin < d) & (tmax > d))[0].tolist()
            try:
                got = section_at(V, tris, n, d, u, v, ids)
                used = d
                break
            except SectionFault as exc:
                last = exc
        if got is None:
            faults.append((k, str(last)))
            got = []
        slices.append(got)
        committed.append(used)
    return slices, committed, faults


def to_world(pt2, u, v, n, d):
    """Lift a plane-frame (x, y) back into 3-D."""
    return (d * n[0] + pt2[0] * u[0] + pt2[1] * v[0],
            d * n[1] + pt2[0] * u[1] + pt2[1] * v[1],
            d * n[2] + pt2[0] * u[2] + pt2[1] * v[2])


def layer_offsets(lo, hi, thickness):
    """Plane offsets for a stacked model, one per material layer.

    Each plane sits at the CENTRE of the layer it represents, so the
    finished stack is as tall as the object instead of overshooting it
    by a layer, and neither end slice is a near-empty sliver.
    """
    span = hi - lo
    if thickness <= 0.0 or span <= 0.0:
        return []
    count = max(1, int(math.floor(span / thickness + 1e-9)))
    start = lo + 0.5 * (span - count * thickness)
    return [start + thickness * (k + 0.5) for k in range(count)]


def spread_offsets(lo, hi, count):
    """`count` planes spread evenly through (lo, hi), excluding the
    extremes -- a plane exactly on a tangent point sections nothing."""
    if count < 1:
        return []
    step = (hi - lo) / (count + 1)
    return [lo + step * (k + 1) for k in range(count)]


# ------------------------------------------------------------------ #

def _icosphere(subdiv=3, radius=1.0):
    """A unit sphere as a closed triangle mesh, for the checks below."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [list(p) for p in verts]
    for _ in range(subdiv):
        mid, new = {}, []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                pa, pb = verts[a], verts[b]
                verts.append([(pa[i] + pb[i]) * 0.5 for i in range(3)])
                mid[key] = len(verts) - 1
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    out = []
    for p in verts:
        L = math.sqrt(sum(c * c for c in p)) or 1.0
        out.append([c * radius / L for c in p])
    return out, faces


def _cube(half=1.0):
    V = [(-half, -half, -half), (half, -half, -half),
         (half, half, -half), (-half, half, -half),
         (-half, -half, half), (half, -half, half),
         (half, half, half), (-half, half, half)]
    Q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return list(V), triangulate(Q)


def _torus(nu=48, nv=24, R=1.0, r=0.35):
    V, F = [], []
    for i in range(nu):
        a = 2 * math.pi * i / nu
        for j in range(nv):
            b = 2 * math.pi * j / nv
            V.append(((R + r * math.cos(b)) * math.cos(a),
                      (R + r * math.cos(b)) * math.sin(a),
                      r * math.sin(b)))
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            F.append((a, b, c))
            F.append((a, c, d))
    return V, F


def _selftest():
    from . import polyclip as pc

    # --- a cube sections to an exact square -----------------------
    V, F = _cube(1.0)
    sl, off, faults, (u, v, n) = section_family(V, F, (0, 0, 1), [0.0])
    assert not faults, f"cube section faulted: {faults}"
    assert len(sl[0]) == 1, f"one loop, got {len(sl[0])}"
    got = pc.area(sl[0][0])
    assert abs(got - 4.0) < 1e-6, f"cube square area {got}"

    # --- a sphere's equator converges to the circle ---------------
    # The section of a faceted sphere is a polygon INSCRIBED in the
    # true circle, so it is short of pi r^2 by an O(1/n^2) deficit that
    # no tolerance can be tightened past.  Refining and watching the
    # residual shrink is the honest check -- a fixed epsilon here would
    # only be measuring how coarse the test sphere is.
    prev_a = prev_p = None
    for subdiv in (2, 3, 4):
        V, F = _icosphere(subdiv, 1.0)
        sl, off, faults, _ = section_family(V, F, (0, 0, 1), [0.0])
        assert not faults, f"sphere section faulted: {faults}"
        assert len(sl[0]) == 1, "sphere equator is one loop"
        ea = abs(pc.area(sl[0][0]) - math.pi)
        ep = abs(pc.perimeter(sl[0][0]) - 2 * math.pi)
        if prev_a is not None:
            assert ea < prev_a * 0.5, \
                f"equator area error must fall with refinement: {prev_a} -> {ea}"
            assert ep < prev_p * 0.5, \
                f"equator perimeter error: {prev_p} -> {ep}"
        prev_a, prev_p = ea, ep
    assert prev_a < 5e-3, f"finest equator area error {prev_a}"
    assert prev_p < 5e-3, f"finest equator perimeter error {prev_p}"

    # --- a torus at its equator is TWO loops (the annulus) --------
    # a chainer that merges loops passes every single-loop test and
    # fails here, which is why this case is in the suite
    V, F = _torus()
    sl, off, faults, _ = section_family(V, F, (0, 0, 1), [0.0])
    assert not faults, f"torus section faulted: {faults}"
    assert len(sl[0]) == 2, f"torus equator is two loops, got {len(sl[0])}"
    areas = sorted(pc.area(L) for L in sl[0])
    # inscribed again, so compare relatively rather than absolutely
    for got, want in zip(areas, (math.pi * 0.65 ** 2, math.pi * 1.35 ** 2)):
        assert abs(got - want) < 0.01 * want, \
            f"torus equator loop area {got}, expected about {want}"

    # --- Cavalieri: sum(area) * spacing converges to the volume ---
    V, F = _icosphere(4, 1.0)
    exact = 4.0 / 3.0 * math.pi
    errs = []
    for count in (20, 40, 80):
        offs = spread_offsets(-1.0, 1.0, count)
        step = 2.0 / (count + 1)
        sl, _, faults, _ = section_family(V, F, (0, 0, 1), offs)
        assert not faults, f"faults at count={count}: {faults}"
        vol = sum(sum(pc.area(L) for L in s) for s in sl) * step
        errs.append(abs(vol - exact))
    assert errs[0] > errs[1] > errs[2], \
        f"Cavalieri sum must converge, got errors {errs}"
    assert errs[-1] < 0.02 * exact, f"final volume error {errs[-1]}"

    # --- layer offsets centre the stack on the object -------------
    offs = layer_offsets(-1.0, 1.0, 0.25)
    assert len(offs) == 8, f"2.0 / 0.25 = 8 layers, got {len(offs)}"
    assert abs(offs[0] + 0.875) < 1e-12, f"first layer centre {offs[0]}"
    assert abs(offs[-1] - 0.875) < 1e-12, f"last layer centre {offs[-1]}"
    assert abs((offs[-1] - offs[0]) - 1.75) < 1e-12, "layer pitch"

    # --- the committed offset is the nudged one, not the request --
    sl, off, faults, _ = section_family(V, F, (0, 0, 1), [0.0])
    assert off[0] != 0.0, \
        "the committed offset must be the nudged value actually used"
    assert abs(off[0]) < 1e-5, "the nudge stays tiny"

    # --- non-parallel planes go through section_planes ------------
    # two planes at right angles through the same sphere must agree
    # about its size; this is the entry point radial fins and ribs use
    V, F = _icosphere(3, 1.0)
    pl = [((0, 0, 1), 0.0, (1, 0, 0), (0, 1, 0)),
          ((1, 0, 0), 0.0, (0, 1, 0), (0, 0, 1)),
          ((1, 1, 0), 0.0, (0, 0, 1), (1, -1, 0))]
    sl, off, faults = section_planes(V, F, pl)
    assert not faults, f"section_planes faulted: {faults}"
    got = [pc.area(s[0]) for s in sl]
    for a in got:
        assert abs(a - math.pi) < 2e-2, \
            f"every great-circle section of a unit sphere has area ~pi: {got}"
    assert off[0] != 0.0, "section_planes commits its nudge too"

    # --- an open surface is a fault, not a silent broken outline --
    V, F = _cube(1.0)
    F_open = [f for f in F if 2 not in f]      # tear a hole in it
    sl, off, faults, _ = section_family(V, F_open, (0, 0, 1), [0.0])
    assert faults, "an open surface must be reported, not chained anyway"

    return True
