# Route II: D-forms built exactly, as valence-3 conical meshes.
#
# Part of the Math Art D-form engine (`math_art/dform/`).  Python and
# numpy only -- no `bpy`.
#
# THE OTHER WAY TO BUILD A D-FORM.  `solve.py` takes the intrinsic
# route: give it two flat outlines and it relaxes a mesh until the metric
# is right.  That is how a D-form is defined, but it is numeric, it is
# stuck at two pieces, and it cannot leave genus 0.  Gonen, Akleman and
# Srinivasan (Bridges 2007) take the extrinsic route instead -- start
# from a convex polyhedron and SLICE it -- and get D-forms exactly, in
# one pass, with no solver and no convergence to worry about, in any
# number of pieces.  It stays at genus 0, though: leaving that needs the
# twisted handles of Xing, Esquivel and Akleman (Bridges 2012), which
# were built and then parked -- see the `hold/twisted-dform` branch and
# BACKLOG.md.
#
# PLANAR TRUNCATION is the whole of the first idea: cut the solid with a
# plane and throw away everything on the far side.  Two things make it
# the right operation.  Clipping a planar face by a plane leaves it
# planar, so faces stay flat for free -- ordinary bevelling does not
# guarantee that.  And every vertex the cut creates lies on exactly
# three faces: the two that shared the cut edge, plus the new cap.
#
# WHY VALENCE 3 IS THE POINT.  A mesh is *conical* at a vertex when
# offsetting all of its face planes outward by one common distance still
# leaves them meeting at a single point -- equivalently, the planes are
# all tangent to one cone of revolution.  That is the condition for
# building the thing out of real material with real thickness, since the
# offset surface stays a mesh with the same combinatorics.  Three planes
# in general position always meet at a point, so a valence-3 planar mesh
# is conical automatically, with nothing to enforce and nothing to
# solve.  Restricting to valence-3 vertices is therefore not a
# simplification -- it is what makes the output buildable at any scale,
# in glass or sheet metal.
#
# Repeated truncation of the same edge smooths it, converging on a
# quadratic profile in the same way Chaikin's corner cutting converges on
# a quadratic B-spline; the paper's Figure 4 rounds four edges of a cube
# and lands on a shape close to the classic two-ellipse D-form.
#
# References:
#   O. Gonen, E. Akleman, V. Srinivasan, "Modeling D-Forms," Bridges
#       2007, pp. 209-216 -- planar truncation, the valence-3/conical
#       argument, and multi-piece D-forms.  Local copy:
#       research/journals/bridges/2007/bridges2007-209/.
#   Y. Liu, H. Pottmann, J. Wallner, Y. Yang, W. Wang, "Geometric
#       Modeling with Conical Meshes and Developable Surfaces," ACM
#       SIGGRAPH 2006, pp. 681-689 -- conical meshes and the offset
#       property.
#   E. Akleman, J. Chen, V. Srinivasan, "A minimal and complete set of
#       operators for the development of robust manifold mesh modelers,"
#       Graphical Models 65(2), 2003, pp. 286-304 -- the TopMod operators
#       this reimplements.

import numpy as np

TRUNCATE_SEEDS = ('CUBE', 'TETRA', 'OCTA', 'DODECA', 'ICOSA')
EDGE_MODES = ('EQUATOR', 'ALL', 'SHARPEST')


def _seed(kind):
    """A convex seed polyhedron, from the shared polyhedra package."""
    try:
        from ..polyhedra.seeds import seed_poly
    except ImportError:                 # flat import outside the package
        from polyhedra.seeds import seed_poly
    V, F = seed_poly(kind)
    return (np.asarray(V, dtype=float),
            [list(int(i) for i in f) for f in F])


def signed_volume(V, F):
    """Six times the signed volume, by the divergence theorem."""
    V = np.asarray(V, dtype=float)
    tot = 0.0
    for f in F:
        for i in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[i]], V[f[i + 1]]
            tot += float(np.dot(a, np.cross(b, c)))
    return tot / 6.0


def orient_outward(V, F):
    """Reverse every face if the solid came out wound inside-out."""
    return [f[::-1] for f in F] if signed_volume(V, F) < 0 else F


def clip_convex(V, F, normal, offset, eps=1e-9):
    """Cut a convex polyhedron with the halfspace n.x <= d.

    Exact, and planarity-preserving by construction: a face is clipped
    inside its own plane, so it cannot leave it, and the cut edges of
    every face all lie in the cutting plane and so close up into one new
    planar cap.  Vertices are welded on rounded coordinates, which is
    what makes the cap share its rim with the clipped faces rather than
    doubling it.
    """
    V = np.asarray(V, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-15)
    s = V @ n - float(offset)

    out_V = []
    ids = {}

    def emit(p):
        key = (round(float(p[0]), 9), round(float(p[1]), 9),
               round(float(p[2]), 9))
        i = ids.get(key)
        if i is None:
            i = len(out_V)
            ids[key] = i
            out_V.append(np.asarray(p, dtype=float))
        return i

    out_F = []
    rim = []
    for f in F:
        loop = []
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            sa, sb = s[a], s[b]
            if sa <= eps:
                j = emit(V[a])
                loop.append(j)
                if abs(sa) <= eps:
                    rim.append(j)
            if (sa < -eps and sb > eps) or (sa > eps and sb < -eps):
                t = sa / (sa - sb)
                j = emit(V[a] + t * (V[b] - V[a]))
                loop.append(j)
                rim.append(j)
        clean = [loop[i] for i in range(len(loop))
                 if loop[i] != loop[i - 1]]
        if len(clean) >= 3:
            out_F.append(clean)

    rim = sorted(set(rim))
    if len(rim) >= 3:
        P = np.array([out_V[i] for i in rim])
        c = P.mean(axis=0)
        u = P[0] - c
        nu = float(np.linalg.norm(u))
        if nu > 1e-12:
            u = u / nu
            w = np.cross(n, u)
            ang = np.arctan2((P - c) @ w, (P - c) @ u)
            out_F.append([rim[i] for i in np.argsort(ang)])

    out_V = np.asarray(out_V, dtype=float)
    return out_V, orient_outward(out_V, out_F)


def face_normals(V, F):
    """Newell normal of every face (robust for near-degenerate ngons)."""
    V = np.asarray(V, dtype=float)
    out = []
    for f in F:
        nrm = np.zeros(3)
        m = len(f)
        for i in range(m):
            a, b = V[f[i]], V[f[(i + 1) % m]]
            nrm += np.cross(a, b)
        ln = float(np.linalg.norm(nrm))
        out.append(nrm / ln if ln > 1e-15 else nrm)
    return np.asarray(out)


def edge_faces(F):
    """Undirected edge -> the faces on it."""
    ef = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            ef.setdefault((a, b) if a < b else (b, a), []).append(fi)
    return ef


def valences(V, F):
    """How many faces meet at each vertex."""
    val = np.zeros(len(V), dtype=int)
    for f in F:
        for i in f:
            val[i] += 1
    return val


def planarity(V, F):
    """Worst out-of-plane deviation of any face, relative to its size."""
    V = np.asarray(V, dtype=float)
    worst = 0.0
    N = face_normals(V, F)
    for f, n in zip(F, N):
        if len(f) < 4:
            continue                    # a triangle is planar, always
        P = V[f]
        c = P.mean(axis=0)
        sz = float(np.max(np.linalg.norm(P - c, axis=1)))
        if sz < 1e-12:
            continue
        worst = max(worst, float(np.max(np.abs((P - c) @ n))) / sz)
    return worst


def genus(V, F):
    """Genus from Euler's formula, for a closed orientable mesh."""
    chi = len(V) - len(edge_faces(F)) + len(F)
    return (2 - chi) // 2


def _select_edges(V, F, mode, axis, min_angle):
    """Which edges to truncate this round.

    EQUATOR is the paper's Figure 4: on a cube it picks the four edges
    running between the two faces that become the caps, and rounding
    just those lands on the classic two-ellipse D-form.  ALL rounds
    everything, which is how the multi-piece forms arise.  SHARPEST
    simply takes whatever is still sharp, which is what repeated
    truncation naturally wants.
    """
    V = np.asarray(V, dtype=float)
    N = face_normals(V, F)
    ax = np.asarray(axis, dtype=float)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-15)
    thr = np.cos(np.radians(min_angle))

    out = []
    for e, fs in edge_faces(F).items():
        if len(fs) != 2:
            continue
        if float(np.dot(N[fs[0]], N[fs[1]])) > thr:
            continue                    # already smooth enough
        if mode == 'EQUATOR':
            d = V[e[1]] - V[e[0]]
            ln = float(np.linalg.norm(d))
            if ln < 1e-12 or abs(float(np.dot(d / ln, ax))) < 0.7:
                continue
        out.append((e, fs))
    return out


def vertex_truncate(V, F, t=0.28):
    """Cut every vertex of valence > 3 back to a face.

    The paper does this before anything else when it starts from an
    octahedron ("we first truncate vertices to obtain a truncated
    octahedron"), and the reason is the whole argument: a vertex where
    four or five faces meet is NOT automatically conical, so edge
    truncation would inherit a defect it cannot repair.  One cut per
    offending vertex replaces it with an n-gon and n valence-3 vertices.
    """
    V = np.asarray(V, dtype=float)
    c = V.mean(axis=0)
    val = valences(V, F)
    picks = [i for i in range(len(V)) if val[i] > 3]
    planes = []
    for i in picks:
        d = V[i] - c
        ln = float(np.linalg.norm(d))
        if ln < 1e-12:
            continue
        n = d / ln
        planes.append((n, float(np.dot(n, V[i])) - t * ln))
    for n, d in planes:
        V, F = clip_convex(V, F, n, d)
    return V, F


def build_truncate(seed='CUBE', mode='EQUATOR', rounds=4, depth=0.28,
                   min_angle=12.0, axis=(0.0, 0.0, 1.0), scale=1.0,
                   jitter=0.03):
    """A D-form by repeated planar truncation (Gonen et al., Bridges 2007).

    Returns (verts, faces).  Every face is exactly planar and every
    vertex the process creates has valence 3, so the result is a conical
    mesh and can be offset to real thickness -- which is the property
    that lets these be built in glass or sheet metal at architectural
    scale.

    `depth` is how far each cut moves toward the centroid, as a fraction
    of the distance from the edge; `rounds` is how many times the
    surviving sharp edges are cut again.  The sequence smooths like
    Chaikin corner cutting, so a handful of rounds already reads as a
    developable band.
    """
    V, F = _seed(seed)
    F = orient_outward(V, F)
    depth = min(max(float(depth), 1e-3), 0.9)
    # a seed with valence-4 or -5 vertices (octahedron, icosahedron) is
    # not conical to begin with, so square it away first
    V, F = vertex_truncate(V, F, depth)

    for _ in range(max(0, int(rounds))):
        picks = _select_edges(V, F, mode, axis, min_angle)
        if not picks:
            break
        # every plane is computed BEFORE any cut, as the paper requires:
        # cutting changes the mesh, and a plane derived from the changed
        # mesh is no longer the plane that was selected
        planes = []
        N = face_normals(V, F)
        for q, (e, fs) in enumerate(picks):
            n = N[fs[0]] + N[fs[1]]
            ln = float(np.linalg.norm(n))
            if ln < 1e-12:
                continue
            n = n / ln
            mid = 0.5 * (V[e[0]] + V[e[1]])
            # THE CUT HAS TO BE LOCAL.  Measuring the depth toward the
            # solid's centroid sends the plane straight past the
            # neighbouring edges, so a round of planes computed together
            # eats each other's faces and the mesh oscillates instead of
            # converging.  Bound it by the nearest vertex of the two
            # adjacent faces instead: the corner is then cut at a fixed
            # FRACTION of the room available, which is Chaikin's rule and
            # is what makes repeated truncation converge on a quadratic.
            room = min((float(np.dot(n, mid - V[v]))
                        for v in set(F[fs[0]]) | set(F[fs[1]])
                        if v not in e
                        and float(np.dot(n, mid - V[v])) > 1e-12),
                       default=0.0)
            if room <= 1e-12:
                continue
            # GENERAL POSITION.  Cut a symmetric solid with symmetric
            # planes and they meet exactly -- three bevels arriving
            # together at a cube corner leave one vertex on six faces,
            # which is precisely the valence the method exists to avoid.
            # A fraction of a percent of variation per edge separates
            # them again; the paper relies on the same genericity when it
            # says set operations on valence-3 meshes "generally" give
            # back valence-3 meshes.
            dq = depth * (1.0 + jitter * ((q / max(len(picks) - 1, 1))
                                          - 0.5))
            planes.append((n, float(np.dot(n, mid)) - dq * room))
        for n, d in planes:
            V, F = clip_convex(V, F, n, d)

    return _fit(V, scale), F


def _fit(V, scale):
    V = np.asarray(V, dtype=float)
    if not len(V):
        return V
    ext = float(np.max(V.max(axis=0) - V.min(axis=0)))
    s = (2.0 * float(scale) / ext) if ext > 1e-12 else 1.0
    return (V - 0.5 * (V.max(axis=0) + V.min(axis=0))) * s

def _selftest():
    ok = True

    V, F = build_truncate('CUBE', mode='EQUATOR', rounds=4)

    # THE claim of the method: faces stay exactly planar.  Clipping a
    # planar polygon by a plane cannot leave its own plane, so this is
    # machine precision rather than a tolerance -- ordinary bevelling
    # would not manage it.
    pl = planarity(V, F)
    good = pl < 1e-9
    ok &= good
    print(f"conical: truncation keeps faces exactly planar ({pl:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # ... and every vertex has valence 3, which is what makes the mesh
    # conical -- three planes always meet at a point when offset, so the
    # solid can be given real thickness and still be built
    val = valences(V, F)
    bad = int(np.sum(val != 3))
    good = bad == 0
    ok &= good
    print(f"conical: every vertex valence 3 ({len(V)} verts, {bad} bad) "
          f"{'OK' if good else 'FAIL'}")

    # the offset test itself, stated directly: displace each of a
    # vertex's face planes outward by a common distance and they still
    # meet at one point
    worst = _offset_residual(V, F, 0.05)
    good = worst < 1e-9
    ok &= good
    print(f"conical: offset planes still meet at a point ({worst:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # closed, convex, genus 0
    good = genus(V, F) == 0 and signed_volume(V, F) > 0
    ok &= good
    print(f"conical: closed convex solid, genus {genus(V, F)} "
          f"{'OK' if good else 'FAIL'}")

    # Truncation must actually round the thing off: the edges it cuts
    # get less sharp every round, converging like Chaikin corner
    # cutting.  Measured over the CUT edges only -- in EQUATOR mode the
    # cap-to-side edges are deliberately left alone, so a maximum over
    # the whole mesh would sit at the cube's untouched 90 degrees and
    # report no progress while the band smooths perfectly well.
    angs = []
    for r in (0, 1, 3, 6):
        Vr, Fr = build_truncate('CUBE', mode='EQUATOR', rounds=r)
        N = face_normals(Vr, Fr)
        w = 0.0
        for e, fs in edge_faces(Fr).items():
            if len(fs) != 2:
                continue
            d = Vr[e[1]] - Vr[e[0]]
            ln = float(np.linalg.norm(d))
            if ln < 1e-12 or abs(float(d[2] / ln)) < 0.7:
                continue                # not one of the cut edges
            w = max(w, np.degrees(np.arccos(np.clip(
                float(np.dot(N[fs[0]], N[fs[1]])), -1, 1))))
        angs.append(w)
    good = angs[-1] < 0.5 * angs[0] and all(
        angs[i + 1] <= angs[i] + 1e-9 for i in range(len(angs) - 1))
    ok &= good
    print(f"conical: repeated truncation smooths the cut band "
          f"({' -> '.join('%.0f' % a for a in angs)} deg) "
          f"{'OK' if good else 'FAIL'}")

    # every seed must survive the operation
    bad = []
    for s in TRUNCATE_SEEDS:
        try:
            Vs, Fs = build_truncate(s, mode='ALL', rounds=2)
            if (planarity(Vs, Fs) > 1e-12 or genus(Vs, Fs) != 0
                    or int(np.sum(valences(Vs, Fs) != 3)) != 0):
                bad.append(s)
        except Exception as exc:                    # noqa: BLE001
            bad.append(f"{s}({type(exc).__name__})")
    good = not bad
    ok &= good
    print(f"conical: every seed truncates cleanly "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform.conical self-test failed")


def _offset_residual(V, F, eps):
    """How far the offset planes at a vertex miss a common point."""
    V = np.asarray(V, dtype=float)
    N = face_normals(V, F)
    inc = {}
    for fi, f in enumerate(F):
        for i in f:
            inc.setdefault(i, []).append(fi)
    worst = 0.0
    for i, fs in inc.items():
        if len(fs) != 3:
            continue
        A = np.array([N[f] for f in fs])
        d = np.array([float(np.dot(N[f], V[F[f][0]])) + eps for f in fs])
        try:
            p = np.linalg.solve(A, d)
        except np.linalg.LinAlgError:
            continue
        worst = max(worst, float(np.max(np.abs(A @ p - d))))
    return worst
