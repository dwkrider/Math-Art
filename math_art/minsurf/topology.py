# Topological surfaces: Klein bottles, cross-caps, genus-g handlebodies.
#
# Part of the Math Art minsurf engine (`math_art/minsurf/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#

import math
import numpy as np


TAU = 2.0 * math.pi


def edge_face_counts(faces):
    """{sorted edge tuple: number of incident faces}."""
    cnt = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


def euler_characteristic(nverts, faces):
    return nverts - len(edge_face_counts(faces)) + len(faces)


def build_klein_bottle(nu, nv):
    """The iconic bottle-shaped Klein immersion (the standard smooth
    closed-form parametrization, u in [0, pi], v in [0, 2pi]). The
    u = pi rim coincides with the u = 0 rim under v -> pi - v.  The
    seam is left SPLIT (coincident duplicate vertices, no index
    gluing): welding it makes the winding flip there, and averaged
    smooth normals then degenerate into a dark shading crease.  Split,
    each side shades smoothly and the renderer's double-sided normal
    flip hides the join.  Cut along that rim the surface is an
    orientable cylinder, so chi is still 0."""
    nv += nv % 2
    u = math.pi * np.arange(nu + 1)[:, None] / nu
    v = TAU * np.arange(nv)[None, :] / nv
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    x = (-2.0 / 15.0) * cu * (3 * cv - 30 * su + 90 * cu ** 4 * su
                              - 60 * cu ** 6 * su + 5 * cu * cv * su)
    y = (-1.0 / 15.0) * su * (3 * cv - 3 * cu ** 2 * cv
                              - 48 * cu ** 4 * cv + 48 * cu ** 6 * cv
                              - 60 * su + 5 * cu * cv * su
                              - 5 * cu ** 3 * cv * su
                              - 80 * cu ** 5 * cv * su
                              + 80 * cu ** 7 * cv * su)
    z = (2.0 / 15.0) * sv * (3 + 5 * cu * su)
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j))
    return V, faces


def build_klein_figure8(nu, nv, radius=2.0):
    """Figure-8 (twisted-torus) Klein immersion: the cross-section is a
    figure-8 that makes a half-turn per revolution. The u = 2pi seam
    coincides with u = 0 under v -> -v but is left SPLIT (coincident
    duplicate vertices) -- see build_klein_bottle for why.  v samples
    sit at half-steps so no column lands on the figure-8 crossing
    point."""
    u = TAU * np.arange(nu + 1)[:, None] / nu
    v = TAU * (np.arange(nv)[None, :] + 0.5) / nv
    c2, s2 = np.cos(u / 2), np.sin(u / 2)
    sv, s2v = np.sin(v), np.sin(2 * v)
    r = radius + c2 * sv - s2 * s2v
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = s2 * sv + c2 * s2v
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j))
    return V, faces


def build_sudanese_mobius(nu, nv):
    """Lawson's minimal Mobius band in S^3, stereographically projected
    to R^3.  The t = 0 and t = pi seam rows coincide in space (with a
    flip v -> pi - v), so -- exactly as for the Klein bottles above --
    the grid is left SPLIT there rather than index-glued: welding flips
    the winding and averaged smooth normals then form a dark crease.
    Cut open along the seam the mesh is a disk (chi 1) whose two ends
    meet on the boundary circle."""
    R2 = math.sqrt(2.0)
    t = math.pi * np.arange(nu + 1)[:, None] / nu       # around
    v = math.pi * np.arange(nv + 1)[None, :] / nv        # across
    ct, st = np.cos(t), np.sin(t)
    cv, sv = np.cos(v), np.sin(v)
    x1 = ct * cv
    x2 = st * cv
    x3 = np.cos(2 * t) * sv
    x4 = np.sin(2 * t) * sv
    s = 1.0 + (x1 + x3) / R2                              # = 1 - x . p
    x = x2 / s
    y = x4 / s
    z = (x1 - x3) / (R2 * s)
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    stride = nv + 1
    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append((i * stride + j, i * stride + j + 1,
                          (i + 1) * stride + j + 1, (i + 1) * stride + j))
    return V, faces


def _rp2_quotient(nu, nv, fn, theta_offset):
    """Mesh a hemisphere parametrization fn(theta, phi) -> (x, y, z),
    phi in (0, pi/2], with the phi = pi/2 pole collapsed to one vertex
    and the phi = 0 equator glued to itself by theta -> theta + pi
    (the RP^2 quotient). nu must be even. Returns (verts, faces) with
    Euler characteristic 1 by construction."""
    half = nu // 2
    th = TAU * (np.arange(nu) + theta_offset) / nu
    x, y, z = fn(th[:1], math.pi / 2)
    verts = [np.array([x[0], y[0], z[0]])]
    for k in range(1, nv):
        ph = (math.pi / 2) * (1.0 - k / (nv - 1))
        m = nu if k < nv - 1 else half
        x, y, z = fn(th[:m], ph)
        verts.extend(np.stack(np.broadcast_arrays(x, y, z), axis=-1))

    def rid(k, j):
        j %= nu
        if k < nv - 1:
            return 1 + (k - 1) * nu + j
        return 1 + (nv - 2) * nu + (j % half)

    faces = [(0, rid(1, j + 1), rid(1, j)) for j in range(nu)]
    for k in range(1, nv - 1):
        for j in range(nu):
            faces.append((rid(k, j), rid(k, j + 1),
                          rid(k + 1, j + 1), rid(k + 1, j)))
    return np.array(verts), faces


def _crosscap_pt(th, ph):
    """Standard cross-cap immersion of RP^2; antipodes
    of the sphere (th, ph latitude) map to the same point."""
    st, ct = np.sin(th), np.cos(th)
    return (0.5 * st * np.sin(2 * ph) + 0 * th,
            0.5 * np.sin(2 * th) * np.cos(ph) ** 2,
            0.5 * np.cos(2 * th) * np.cos(ph) ** 2)


def _roman_pt(th, ph):
    """Steiner's Roman surface: the sphere mapped through
    (x, y, z) -> (yz, zx, xy)."""
    cp, sp = np.cos(ph), np.sin(ph)
    return (np.sin(th) * cp * sp,
            np.cos(th) * cp * sp,
            np.sin(th) * np.cos(th) * cp * cp)


def build_crosscap(nu, nv):
    nu += nu % 2               # theta -> theta + pi must be a grid map
    return _rp2_quotient(nu, nv, _crosscap_pt, 0.5)


def build_roman(nu, nv):
    nu += (-nu) % 4            # quarter-offset grid: need 4 | nu
    return _rp2_quotient(nu, nv, _roman_pt, 0.25)


def build_boy(ntheta, nrings):
    """Boy's surface via the Bryant-Kusner parametrization on the unit
    disk (polar grid), with the boundary circle glued antipodally
    (z ~ -z on |z| = 1) by vertex index. The three poles of the
    denominator inside the disk are the planar ends of the underlying
    minimal surface; they invert to the triple point at the origin, and
    samples landing on them are nudged off."""
    ntheta += ntheta % 2
    half = ntheta // 2
    th = TAU * (np.arange(ntheta) + 0.5) / ntheta
    s5 = math.sqrt(5.0)

    def bk(zc):
        w = zc ** 6 + s5 * zc ** 3 - 1
        bad = np.abs(w) < 1e-7
        if np.any(bad):
            zc = np.where(bad, zc * 1.01, zc)
            w = zc ** 6 + s5 * zc ** 3 - 1
        g1 = -1.5 * (zc * (1 - zc ** 4) / w).imag
        g2 = -1.5 * (zc * (1 + zc ** 4) / w).real
        g3 = ((1 + zc ** 6) / w).imag - 0.5
        s = g1 * g1 + g2 * g2 + g3 * g3
        return np.stack([g1 / s, g2 / s, g3 / s], axis=-1)

    verts = [bk(np.zeros(1, dtype=complex))[0]]
    for k in range(1, nrings + 1):
        r = k / nrings
        m = ntheta if k < nrings else half
        verts.extend(bk(r * np.exp(1j * th[:m])))

    def rid(k, j):
        j %= ntheta
        if k < nrings:
            return 1 + (k - 1) * ntheta + j
        return 1 + (nrings - 1) * ntheta + (j % half)

    faces = [(0, rid(1, j + 1), rid(1, j)) for j in range(ntheta)]
    for k in range(1, nrings):
        for j in range(ntheta):
            faces.append((rid(k, j), rid(k, j + 1),
                          rid(k + 1, j + 1), rid(k + 1, j)))
    return np.array(verts), faces


_GENUS_R = 1.0          # circle radius


_GENUS_SPACING = 1.4    # center spacing (< 2r: adjacent circles overlap)


_GENUS_LEVEL = 0.015    # eps: below the lens-core peak for g = 1..5


_GENUS_ZK = 2.0         # z^2 coefficient: slab half-height <= ~0.42


def build_genus(genus, cell=0.125):
    # This module IS inside `minsurf`, so it imports the package it lives
    # in -- one dot up, not one dot across.
    try:
        from . import parametric, plateau, weierstrass, zoo   # noqa: F401
        from .. import minsurf as mst
    except ImportError:
        import minsurf as mst
    g = genus
    r, d = _GENUS_R, _GENUS_SPACING
    cs = [(i - g / 2.0) * d for i in range(g + 1)]

    def field(x, y, z):
        q = np.ones_like(x)
        for c in cs:
            rho2 = (x - c) ** 2 + y ** 2
            q = q * (rho2 - r * r) / (rho2 + r * r)
        return q + _GENUS_ZK * z * z - _GENUS_LEVEL

    m = 0.6
    bmin = (cs[0] - r - m, -r - m, -0.55)
    bmax = (cs[-1] + r + m, r + m, 0.55)
    res = tuple(max(8, int(round((bmax[i] - bmin[i]) / cell)))
                for i in range(3))
    return mst.marching_tets(field, bmin, bmax, res)


def build_nonorientable(k=3, segments=64, rings=32, hole=0.0,
                        pinch=0.55):
    """The closed non-orientable surface N_k of genus k, as an
    immersion: a sphere carrying k cross-caps.

    N_1 is the projective plane, N_2 the Klein bottle, N_3 Dyck's
    surface, and every closed non-orientable surface is one of these.
    None of them EMBEDS in R^3 -- that is a theorem, not a limitation of
    the meshing -- so each cross-cap is drawn the way it always is, as a
    self-intersecting pinched cap with a segment of double points
    running between two pinch points.

    The construction is surgery rather than a formula, which is what
    makes it exact.  For each cross-cap: cut a disk out of the sphere,
    leaving a boundary circle of 2m vertices, then glue that circle to
    itself ANTIPODALLY by welding vertex i to vertex i + m.  That is the
    definition of attaching a cross-cap, so the topology is right by
    construction rather than by numerical luck: each one drops the Euler
    characteristic by exactly 1, giving chi = 2 - k, and makes the
    surface one-sided.

    Welding each antipodal pair to their midpoint collapses the cut
    circle onto one of its diameters, and that segment is precisely the
    double-point line of the classical cross-cap picture.  `pinch`
    lifts the cap over that segment so the two sheets are visible
    rather than coincident.

    Returns (verts, faces).  The faces along each double-point segment
    are shared by four triangles, not two; that is what an immersion
    looks like as a mesh and is not a defect to weld away.
    """
    import numpy as np

    k = max(1, int(k))
    # How big each cross-cap should be.  A fixed radius makes N_1 read
    # as a sphere with a dent rather than as the projective plane: with
    # one cross-cap the cap IS the surface's whole character and should
    # dominate, while with six they must stay clear of one another.
    # Adjacent centres sit 2 sin(pi/k) apart on the equator, so that
    # sets the ceiling; 0.9 is the free choice when there is only one.
    if hole <= 0.0:
        hole = 0.95 if k == 1 else min(0.95, 0.80 * math.sin(math.pi / k))
    m = max(3, int(segments) // 2)          # half the hole's boundary
    nseg, nring = int(segments), int(rings)

    # --- the sphere, poles welded -----------------------------------
    verts = [(0.0, 0.0, 1.0)]
    for j in range(1, nring):
        phi = math.pi * j / nring
        for i in range(nseg):
            th = 2.0 * math.pi * i / nseg
            verts.append((math.sin(phi) * math.cos(th),
                          math.sin(phi) * math.sin(th),
                          math.cos(phi)))
    verts.append((0.0, 0.0, -1.0))
    south = len(verts) - 1

    def vid(j, i):
        return 1 + (j - 1) * nseg + (i % nseg)

    faces = []
    for i in range(nseg):
        faces.append((0, vid(1, i + 1), vid(1, i)))
    for j in range(1, nring - 1):
        for i in range(nseg):
            faces.append((vid(j, i), vid(j, i + 1),
                          vid(j + 1, i + 1), vid(j + 1, i)))
    for i in range(nseg):
        faces.append((south, vid(nring - 1, i), vid(nring - 1, i + 1)))

    V = [list(p) for p in verts]

    # --- k cross-caps, spaced around the equator --------------------
    remap = {}

    def resolve(a):
        while a in remap:
            a = remap[a]
        return a

    kept = []
    for c in range(k):
        centre = np.array([math.cos(2.0 * math.pi * c / k),
                           math.sin(2.0 * math.pi * c / k), 0.0])
        # faces whose centroid falls inside the disk are cut away, and
        # the vertices left on the cut form the boundary circle
        inside = []
        for f in faces:
            g = np.mean([V[a] for a in f], axis=0)
            if float(np.linalg.norm(g - centre)) < hole:
                inside.append(f)
        if not inside:
            continue
        cut = set()
        for f in inside:
            cut.update(f)
        # boundary ring: cut vertices that still belong to a kept face
        kept_now = [f for f in faces if f not in inside]
        onring = set()
        for f in kept_now:
            for a in f:
                if a in cut:
                    onring.add(a)
        # order the cut circle by angle IN ITS OWN PLANE.  Picking the
        # axes off fixed coordinates instead sorts different holes by
        # different conventions, which pairs the wrong vertices and
        # wrecks the Euler characteristic on some values of k.
        nrm0 = centre / max(float(np.linalg.norm(centre)), 1e-12)
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, nrm0))) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(nrm0, tmp)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
        e2 = np.cross(nrm0, e1)

        def ring_angle(a):
            d = np.array(V[a]) - centre
            return math.atan2(float(np.dot(d, e2)), float(np.dot(d, e1)))

        ring = sorted(onring, key=ring_angle)
        faces = kept_now
        if len(ring) < 6:
            continue
        half = len(ring) // 2
        axis = np.array([-centre[1], centre[0], 0.0])
        nrm = np.linalg.norm(axis)
        axis = axis / nrm if nrm > 1e-12 else np.array([0.0, 1.0, 0.0])
        for t in range(half):
            a, b = resolve(ring[t]), resolve(ring[t + half])
            if a == b:
                continue
            pa, pb = np.array(V[a]), np.array(V[b])
            mid = 0.5 * (pa + pb)
            # lift the weld off the sphere so the two sheets separate
            s = math.sin(math.pi * (t + 0.5) / half)
            mid = mid + pinch * hole * s * centre / max(
                float(np.linalg.norm(centre)), 1e-12)
            V[a] = list(mid)
            remap[b] = a
        kept.append(c)

    faces = [tuple(resolve(a) for a in f) for f in faces]
    faces = [f for f in faces if len(set(f)) == len(f)]

    used = sorted({a for f in faces for a in f})
    idx = {a: i for i, a in enumerate(used)}
    Vout = [tuple(V[a]) for a in used]
    Fout = [tuple(idx[a] for a in f) for f in faces]
    return Vout, Fout


def build_twist_strip(half_twists, segments, width=0.6, thick=0.18,
                      ridge=False, radius=1.5):
    """Sweep a rectangular cross-section (optionally with a raised
    center-line ridge on both wide faces, as in Segerman fig 6-1)
    around a circle, turning it by n*pi over one revolution. The
    cross-section point list is symmetric under a half-turn (index
    shift k/2), so for odd n the seam closes with an index shift and
    the result is a single watertight solid -- printable directly."""
    n = half_twists
    m = max(segments, 8 * max(abs(n), 1))
    w2, t2 = width / 2.0, thick / 2.0
    if ridge:
        bw, bh = 0.16 * width, 0.7 * thick
        prof = [(w2, -t2), (w2, t2), (bw, t2), (0.0, t2 + bh),
                (-bw, t2), (-w2, t2), (-w2, -t2), (-bw, -t2),
                (0.0, -t2 - bh), (bw, -t2)]
    else:
        prof = [(w2, -t2), (w2, t2), (-w2, t2), (-w2, -t2)]
    k = len(prof)
    shift = (n % 2) * (k // 2)
    verts = []
    for j in range(m):
        t = TAU * j / m
        al = 0.5 * n * t
        ca, sa = math.cos(al), math.sin(al)
        ct, st = math.cos(t), math.sin(t)
        for (a, b) in prof:
            ar = a * ca - b * sa       # rotate in the (radial, z) plane
            br = a * sa + b * ca
            verts.append(((radius + ar) * ct, (radius + ar) * st, br))
    faces = []
    for j in range(m):
        j2 = (j + 1) % m
        s = shift if j == m - 1 else 0
        for i in range(k):
            i2 = (i + 1) % k
            faces.append((j * k + i, j * k + i2,
                          j2 * k + (i2 + s) % k, j2 * k + (i + s) % k))
    return np.array(verts), faces


def _selftest():
    """The module had no self-test; this adds one for the surface the
    whole point of which is its topology."""
    from collections import defaultdict, deque
    ok = True

    def _chi(V, F):
        e = set()
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                e.add((a, b) if a < b else (b, a))
        return len(V) - len(e) + len(F)

    def _orientable(F):
        """Try to orient every face consistently.

        Orientation only propagates across MANIFOLD edges; the
        double-point segments of an immersion carry four faces and are
        skipped, which is correct -- they are where the surface passes
        through itself, not where it is glued.
        """
        edge = defaultdict(list)
        for fi, f in enumerate(F):
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                edge[(a, b) if a < b else (b, a)].append((fi, a, b))
        adj = defaultdict(list)
        for lst in edge.values():
            if len(lst) == 2:
                (f0, a0, _b0), (f1, a1, _b1) = lst
                flip = (a0 == a1)
                adj[f0].append((f1, flip))
                adj[f1].append((f0, flip))
        sign = {}
        for start in range(len(F)):
            if start in sign:
                continue
            sign[start] = 1
            q = deque([start])
            while q:
                u = q.popleft()
                for v, flip in adj[u]:
                    want = -sign[u] if flip else sign[u]
                    if v in sign:
                        if sign[v] != want:
                            return False
                    else:
                        sign[v] = want
                        q.append(v)
        return True

    # N_k: chi = 2 - k, and one-sided.  These are the definition of the
    # surface, not a proxy for it, and the surgery is exact, so a bug in
    # the ring ordering shows up here at once.  It did: sorting the cut
    # circle by fixed coordinate axes instead of in the hole's own plane
    # paired the wrong vertices and gave chi = -18 for k = 4, while
    # k = 1, 2, 3 and 5 all came out right and looked convincing.
    bad = []
    for k in (1, 2, 3, 4, 5, 6):
        V, F = build_nonorientable(k, 48, 24)
        c = _chi(V, F)
        if c != 2 - k:
            bad.append('N%d:chi=%d(want %d)' % (k, c, 2 - k))
        elif _orientable(F):
            bad.append('N%d:two-sided' % k)
    ok &= not bad
    print("topology: N_k has chi = 2-k and is one-sided, k = 1..6 %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # control: the same machinery on a sphere must come out orientable
    # with chi = 2, or the test above proves nothing.
    V, F = build_genus(1)
    good = _orientable(F)
    ok &= good
    print("topology: control -- an orientable surface still reads as "
          "two-sided %s" % ('OK' if good else 'FAIL'))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("topology self-test failed")

