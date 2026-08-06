
# Exact biscribed forms of the symmetric (Archimedean-type) solids.
#
# A biscribed solid has ALL vertices on a circumsphere and ALL faces tangent
# to a CONCENTRIC insphere.  For a vertex-transitive symmetric family the
# circumsphere is automatic, so biscribing reduces to equalizing the
# face-orbit plane distances with the family's shape parameter(s):
#
#   * Truncation family (tT tC tO tD tI): one parameter t, the truncation
#     depth (new vertex = (1-t)A + t.B along each base edge; t = 1/2 is full
#     rectification).  The base-face plane distance is constant in t, the
#     vertex-truncation-face distance varies; a biscribed form exists iff
#     g(t) = d_vertexface(t) - d_baseface has a root in (0, 1/2).  Solved by
#     bisection; no sign change => NO biscribed form exists.
#   * Rectified solids (cuboctahedron, icosidodecahedron): zero parameters,
#     two unequal face-orbit distances => NO biscribed form.
#   * Omnitruncates (truncated cuboctahedron / icosidodecahedron): faces lie
#     in planes perpendicular to the 2-, 3- and n-fold symmetry axes, with
#     the three plane offsets (d2 : d3 : dn) as the shape parameters (mod
#     scale).  Each vertex is the meet of one plane of each type, so setting
#     d2 = d3 = dn = 1 IS the biscribed form, exactly.
#   * Snubs (snub cube, snub dodecahedron): vertex orbit of a point p on the
#     sphere under the pure rotation group (2 parameters mod scale); three
#     face orbits (n-gons, axis triangles, generic triangles) give two
#     equalities g(p) = 0, solved by a damped Newton iteration.
#
# Duals: polar reciprocation about the sphere of radius rho = sqrt(R*r)
# carries a biscribed solid to its dual, again biscribed with the same
# (R, r) -- so tetrakis hexahedron, pentakis dodecahedron, disdyakis solids
# and the pentagonal icositetra-/hexecontahedron come for free.
#
# The chiral biscribed solids (propello / hexpropello families) use a
# general K-orbit generalization of the snub solver (see solve_chiral);
# solve_chiral_g extends this to non-Platonic seeds (propello / snub of the
# truncated forms, and orthokis-propello), whose slow roundest-root solves are
# precomputed into _biscribed_chiral_data and re-derivable via
# _regen_chiral_data.  Covered: all 31 of McCooey's chiral biscribed solids.
#
# References:
# - D. McCooey, "Biscribed (Non-)Chiral Solids", Visual Polyhedra,
#   dmccooey.com/polyhedra/BiscribedNonChiral.html and BiscribedChiral.html
#   (catalog matched here; Propello Cube circumradius reproduced to 1e-10).
# - G. W. Hart, "Calculating Canonical Polyhedra", Mathematica in Education
#   and Research 6(3), 1997 (canonical/midscribed background); and Hart's
#   propellor operator.  Whirl / hexpropellor = Goldberg-Coxeter c(2,1):
#   M. Goldberg, "A class of multi-symmetric polyhedra", Tohoku Math. J. 43
#   (1937).
# - I. Rivin, "A characterization of ideal polyhedra in hyperbolic 3-space"
#   (inscribability), arXiv:math/9210218.

bl_info = {
    "name": "Biscribed Solids",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "The biscribed Archimedean-type solids and their duals "
                   "(vertices on a circumsphere, faces tangent to a "
                   "concentric insphere)",
    "category": "Add Mesh",
}

import itertools
import math

try:
    import numpy as np
except ImportError:
    np = None

try:
    from . import _biscribed_chiral_data as _chiral_data
except ImportError:
    try:
        import _biscribed_chiral_data as _chiral_data
    except ImportError:
        _chiral_data = None
# precomputed coords for the slow non-Platonic-seed chiral solids (loaded so
# the operator is instant; re-derivable via _regen_chiral_data / solve_chiral_g)
_CHIRAL_DATA = getattr(_chiral_data, 'CHIRAL_DATA', {})

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------
# Platonic seeds
# --------------------------------------------------------------------------

def _platonic(sym):
    if sym == 'T':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    elif sym == 'C':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    elif sym == 'O':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
             [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    elif sym == 'I':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = convex_hull_faces(np.array(V, float))
    elif sym == 'D':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        for a in (-1 / PHI, 1 / PHI):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = convex_hull_faces(np.array(V, float))
    else:
        raise ValueError(f"unknown Platonic {sym!r}")
    return np.array(V, float), [list(f) for f in F]


# --------------------------------------------------------------------------
# Convex hull (plane enumeration, vectorized; fine up to ~100 points)
# --------------------------------------------------------------------------

def convex_hull_faces(V, tol=1e-7):
    """Faces of the convex hull of centred points V, merged into maximal
    planar polygons, each wound CCW seen from outside."""
    V = np.asarray(V, float)
    n = len(V)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    A, B, C = V[idx[:, 0]], V[idx[:, 1]], V[idx[:, 2]]
    N = np.cross(B - A, C - A)
    ln = np.linalg.norm(N, axis=1)
    ok = ln > 1e-9
    N, A = N[ok] / ln[ok, None], A[ok]
    d = np.einsum('ij,ij->i', N, A)
    flip = d < 0
    N[flip] *= -1
    d[flip] *= -1
    sup = (N @ V.T).max(axis=1) <= d + tol
    N, d = N[sup], d[sup]
    keys = np.round(np.column_stack([N, d]), 6)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    faces = []
    for i in uniq:
        on = np.where(np.abs(V @ N[i] - d[i]) < tol)[0]
        faces.append(_order_around(V, list(on), N[i]))
    return faces


def _order_around(V, ids, u):
    """Order vertex ids CCW (seen from outside) about outward direction u."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(u @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)          # (e1, e2, u) right-handed
    c = V[ids].mean(axis=0)
    ang = [math.atan2((V[i] - c) @ e2, (V[i] - c) @ e1) for i in ids]
    ordered = [i for _, i in sorted(zip(ang, ids))]
    # (e1, e2, u) right-handed and increasing atan2 => CCW from outside
    return ordered


# --------------------------------------------------------------------------
# Mesh measurements
# --------------------------------------------------------------------------

def _face_planes(V, F):
    """(unit outward Newell normal, centroid, plane distance) per face."""
    out = []
    for f in F:
        P = V[list(f)]
        nrm = np.zeros(3)
        m = len(f)
        for i in range(m):
            p, q = P[i], P[(i + 1) % m]
            nrm += np.cross(p, q)
        nrm /= np.linalg.norm(nrm)
        c = P.mean(axis=0)
        if nrm @ c < 0:
            nrm = -nrm
        out.append((nrm, c, nrm @ c))
    return out


def verify(V, F):
    """(std vertex radii, std face-plane distances, max non-planarity)."""
    V = np.asarray(V, float)
    rad = np.linalg.norm(V, axis=1)
    planes = _face_planes(V, F)
    dist = np.array([d for _, _, d in planes])
    planar = max(np.max(np.abs((V[list(f)] - c) @ n))
                 for f, (n, c, _) in zip(F, planes))
    return rad.std(), dist.std(), planar


# --------------------------------------------------------------------------
# Truncation family  tT tC tO tD tI  (and rectified at t = 1/2)
# --------------------------------------------------------------------------

def _truncation_mesh(sym, t):
    """Truncate Platonic `sym` at depth t.  Returns (V, F_orig, F_vert):
    the shrunken base faces and the vertex-truncation faces."""
    V0, F0 = _platonic(sym)
    nv, verts = {}, []
    for f in F0:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            for e in ((a, b), (b, a)):
                if e not in nv:
                    nv[e] = len(verts)
                    verts.append((1 - t) * V0[e[0]] + t * V0[e[1]])
    F_orig = []
    nbrs = {}
    for f in F0:
        m = len(f)
        face = []
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            face += [nv[(a, b)], nv[(b, a)]]
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
        F_orig.append(face)
    V = np.array(verts)
    F_vert = [_order_around(V, [nv[(a, b)] for b in nbrs[a]], V0[a])
              for a in range(len(V0))]
    return V, F_orig, F_vert


def _trunc_distances(sym, t):
    V, Fo, Fv = _truncation_mesh(sym, t)
    d_o = np.array([d for _, _, d in _face_planes(V, Fo)])
    d_v = np.array([d for _, _, d in _face_planes(V, Fv)])
    return V, Fo, Fv, d_o.mean(), d_v.mean()


def solve_truncation(sym):
    """Bisection on g(t) = d_vertexface - d_baseface over (0, 1/2)."""
    def g(t):
        _, _, _, do, dv = _trunc_distances(sym, t)
        return dv - do

    lo, hi = 1e-6, 0.499
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:
        _, _, _, do, dv = _trunc_distances(sym, 0.25)
        return {'exists': False, 'why': (
            f"no root: d_vertexface stays {'above' if glo > 0 else 'below'} "
            f"d_baseface over t in (0, 1/2)  (at t=1/4: {dv:.4f} vs {do:.4f})")}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if glo * gm <= 0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    t = 0.5 * (lo + hi)
    V, Fo, Fv, do, dv = _trunc_distances(sym, t)
    R = np.linalg.norm(V, axis=1).max()
    V = V / R
    r = 0.5 * (do + dv) / R
    return {'exists': True, 'param': f"t*={t:.6f}", 't': t,
            'verts': V, 'faces': [list(f) for f in Fo + Fv], 'r_over_R': r}


def solve_rectified(sym):
    """Rectified solid (t = 1/2 fixed): zero shape parameters, two face
    orbits.  Report the two (unequal) distances after scaling R = 1."""
    V, Fo, Fv, do, dv = _trunc_distances(sym, 0.5)
    R = np.linalg.norm(V, axis=1).max()
    return {'exists': False, 'why': (
        f"rectified: no shape parameter; face-orbit distances differ "
        f"({do / R:.5f} vs {dv / R:.5f} at R=1)")}


# --------------------------------------------------------------------------
# Symmetry groups and axes
# --------------------------------------------------------------------------

def _rot(axis, angle):
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _closure(gens, expect):
    def key(M):
        return tuple(np.round(M, 6).ravel())
    mats = [np.eye(3)]
    seen = {key(mats[0])}
    grew = True
    while grew:
        grew = False
        for A in list(mats):
            for G in gens:
                M = G @ A
                k = key(M)
                if k not in seen:
                    seen.add(k)
                    mats.append(M)
                    grew = True
    assert len(mats) == expect, f"group closure gave {len(mats)}"
    return mats


def _sym_data(sym):
    """For octahedral ('O') or icosahedral ('I') symmetry: rotation group,
    n-/3-/2-fold axis directions, and one mutually adjacent corner triple
    (cn, c3, c2) of the fundamental spherical triangle."""
    V, F = _platonic(sym)
    nfold = 4 if sym == 'O' else 5
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    dirs_n = list(Vn)
    dirs_3 = []
    for f in F:
        c = V[list(f)].mean(axis=0)
        dirs_3.append(c / np.linalg.norm(c))
    edges = {tuple(sorted((f[i], f[(i + 1) % len(f)])))
             for f in F for i in range(len(f))}
    dirs_2 = []
    for a, b in sorted(edges):
        m = 0.5 * (V[a] + V[b])
        dirs_2.append(m / np.linalg.norm(m))
    f0 = next(f for f in F if 0 in f)
    cn = Vn[0]
    c3 = V[list(f0)].mean(axis=0)
    c3 = c3 / np.linalg.norm(c3)
    b = f0[(f0.index(0) + 1) % len(f0)]
    c2 = 0.5 * (V[0] + V[b])
    c2 = c2 / np.linalg.norm(c2)
    order = 24 if sym == 'O' else 60
    rots = _closure([_rot(cn, 2 * math.pi / nfold),
                     _rot(c3, 2 * math.pi / 3)], order)
    return {'rots': rots, 'nfold': nfold, 'dirs_n': dirs_n,
            'dirs_3': dirs_3, 'dirs_2': dirs_2, 'corners': (cn, c3, c2)}


# --------------------------------------------------------------------------
# Omnitruncates (truncated cuboctahedron / icosidodecahedron) -- exact
# --------------------------------------------------------------------------

def solve_omnitruncate(sym):
    """Faces sit in planes u.x = d perpendicular to the 2-, 3- and n-fold
    axes; the offsets (d2:d3:dn) are the two shape parameters (mod scale).
    d2 = d3 = dn = 1 is the biscribed form, exactly."""
    S = _sym_data(sym)
    cn, c3, c2 = S['corners']
    v = np.linalg.solve(np.array([cn, c3, c2]), np.ones(3))
    full = S['rots'] + [-R for R in S['rots']]        # Oh / Ih
    pts = np.array([R @ v for R in full])
    uniq = np.unique(np.round(pts, 9), axis=0)
    n_expected = len(full)
    assert len(uniq) == n_expected, "omnitruncate orbit not free"
    V = pts
    faces = []
    for u, size in ([(d, 2 * S['nfold']) for d in S['dirs_n']] +
                    [(d, 6) for d in S['dirs_3']] +
                    [(d, 4) for d in S['dirs_2']]):
        on = np.where(V @ u > 1 - 1e-6)[0]
        assert len(on) == size, f"face size {len(on)} != {size}"
        assert np.max(V @ u) <= 1 + 1e-9        # supporting plane
        faces.append(_order_around(V, list(on), u))
    R = np.linalg.norm(v)
    return {'exists': True, 'param': "d2=d3=dn (exact)",
            'verts': V / R, 'faces': faces, 'r_over_R': 1.0 / R}


# --------------------------------------------------------------------------
# Snubs (snub cube / snub dodecahedron) -- 2-parameter Newton
# --------------------------------------------------------------------------

def _snub_orbit(S, p):
    return np.array([R @ p for R in S['rots']])


def _classify_snub_faces(S, faces, V):
    """Split hull faces into (n-gons, axis triangles, generic triangles)."""
    big, ax, gen = [], [], []
    d3 = np.array(S['dirs_3'])
    for f in faces:
        if len(f) > 3:
            big.append(f)
            continue
        c = V[list(f)].mean(axis=0)
        c = c / np.linalg.norm(c)
        ax.append(f) if np.max(np.abs(d3 @ c)) > 1 - 1e-9 else gen.append(f)
    return big, ax, gen


def solve_snub(sym):
    """Vertex orbit of p (2 sphere parameters) under the rotation group;
    equalize the three face-orbit distances by damped Newton on g(p)."""
    S = _sym_data(sym)
    cn, c3, c2 = S['corners']
    nbig = 6 if sym == 'O' else 12
    nax = 8 if sym == 'O' else 20
    ngen = 24 if sym == 'O' else 60
    hist_want = {S['nfold']: nbig, 3: nax + ngen}

    # --- initial guess: barycentric grid over the fundamental triangle,
    #     keep the sample with snub combinatorics and smallest |g|
    best = None
    n = 7
    for i in range(1, n - 1):
        for j in range(1, n - i):
            k = n - i - j
            p = i * cn + j * c3 + k * c2
            p = p / np.linalg.norm(p)
            V = _snub_orbit(S, p)
            faces = convex_hull_faces(V)
            hist = {}
            for f in faces:
                hist[len(f)] = hist.get(len(f), 0) + 1
            if hist != hist_want:
                continue
            big, ax, gen = _classify_snub_faces(S, faces, V)
            if (len(big), len(ax), len(gen)) != (nbig, nax, ngen):
                continue
            g = _snub_g(S, p, (big, ax, gen))
            if best is None or np.linalg.norm(g) < best[0]:
                best = (np.linalg.norm(g), p, (big, ax, gen))
    if best is None:
        return {'exists': False, 'why': "no snub-type region found"}
    _, p, orbits = best

    # --- damped Newton with central differences, topology fixed
    h = 1e-6
    for _ in range(100):
        g0 = _snub_g(S, p, orbits)
        if np.max(np.abs(g0)) < 1e-13:
            break
        ref = np.array([1.0, 0, 0]) if abs(p[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(p, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(p, e1)
        J = np.empty((2, 2))
        for c, e in enumerate((e1, e2)):
            pp = p + h * e
            pm = p - h * e
            gp = _snub_g(S, pp / np.linalg.norm(pp), orbits)
            gm = _snub_g(S, pm / np.linalg.norm(pm), orbits)
            J[:, c] = (gp - gm) / (2 * h)
        step = np.linalg.solve(J, -g0)
        s = 1.0
        while s > 1e-8:
            pn = p + s * (step[0] * e1 + step[1] * e2)
            pn /= np.linalg.norm(pn)
            if np.linalg.norm(_snub_g(S, pn, orbits)) < np.linalg.norm(g0):
                p = pn
                break
            s *= 0.5
        else:
            return {'exists': False, 'why': "Newton stalled"}
    else:
        return {'exists': False, 'why': "Newton did not converge"}

    V = _snub_orbit(S, p)                       # |p| = 1 -> R = 1 exactly
    big, ax, gen = orbits
    faces = big + ax + gen
    # confirm the fixed topology still is the convex hull
    for nrm, _, d in _face_planes(V, faces):
        assert np.max(V @ nrm) <= d + 1e-9, "left the snub topology cell"
    r = np.mean([d for _, _, d in _face_planes(V, faces)])
    return {'exists': True, 'param': "Newton on 2 sphere params",
            'verts': V, 'faces': faces, 'r_over_R': r}


def _snub_g(S, p, orbits):
    V = _snub_orbit(S, p)
    dm = [np.mean([d for _, _, d in _face_planes(V, fs)]) for fs in orbits]
    return np.array([dm[0] - dm[1], dm[0] - dm[2]])


# --------------------------------------------------------------------------
# Chiral solids (propello / hexpropello) -- general K-orbit exact biscriber
#
# Generalizes solve_snub from ONE vertex orbit to K.  A propello (Conway p)
# or hexpropello / whirl (Conway w) of a Platonic seed has its vertices in
# several orbits under the pure rotation group; each orbit is the group
# image of one representative point on the unit sphere (2 DOF, or 0 DOF when
# the representative sits on a symmetry axis).  The face planes fall into
# face orbits; the biscribed condition is: every face planar AND all
# face-orbit plane distances equal (all vertices are on |v|=1 by
# construction).  Solved by damped Levenberg-Marquardt (central-difference
# Jacobian) with the topology held fixed -- the multi-orbit analogue of
# solve_snub.  Propello Cube reproduces McCooey's circumradius to 1e-10.
# --------------------------------------------------------------------------

def _cw():
    try:
        from . import conway_operators as cw
    except ImportError:
        import conway_operators as cw
    return cw


def _frame(a, b):
    e1 = a / np.linalg.norm(a)
    t = b - (b @ e1) * e1
    e2 = t / np.linalg.norm(t)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def _rot_group(seed):
    """All proper rotations permuting a Platonic seed's vertex directions
    (order 12 for T, 24 for C/O, 60 for D/I)."""
    V, _ = _cw()._seed(seed, 0)
    P = np.array(V, float)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    n = len(P)
    i1 = next(i for i in range(1, n) if abs(P[0] @ P[i]) < 0.99)
    Fs = _frame(P[0], P[i1])
    ang0 = P[0] @ P[i1]
    Rs, seen = [], set()
    for qi in range(n):
        for qj in range(n):
            if qi == qj or abs(P[qi] @ P[qj] - ang0) > 1e-6:
                continue
            R = _frame(P[qi], P[qj]) @ Fs.T
            if abs(np.linalg.det(R) - 1) > 1e-6:
                continue
            if all(np.linalg.norm(P - (P @ R.T)[k], axis=1).min() < 1e-6
                   for k in range(n)):
                key = tuple(np.round(R, 5).ravel())
                if key not in seen:
                    seen.add(key)
                    Rs.append(R)
    return Rs


def _vertex_orbits(G, U):
    """Assign each vertex direction U[i] to a G-orbit with a group index g
    (G[g]@U[rep]=U[i]); reps on a symmetry axis (stabilizer>1) are locked."""
    N = len(U)
    idI = next(g for g, R in enumerate(G)
               if np.allclose(R, np.eye(3), atol=1e-6))
    orbit = [-1] * N
    assign = [None] * N
    reps = []
    for i in range(N):
        if orbit[i] >= 0:
            continue
        o = len(reps)
        reps.append(i)
        orbit[i] = o
        assign[i] = (o, idI)
        for g, R in enumerate(G):
            Ui = R @ U[i]
            for j in range(N):
                if orbit[j] < 0 and np.linalg.norm(U[j] - Ui) < 1e-6:
                    orbit[j] = o
                    assign[j] = (o, g)
                    break
    locked = [sum(1 for R in G if np.linalg.norm(R @ U[r] - U[r]) < 1e-6) > 1
              for r in reps]
    return assign, reps, locked


def _face_orbit_reps(G, Vc, F):
    """One representative face index per face orbit (by size + centroid
    direction orbit)."""
    cd = [Vc[list(f)].mean(0) / np.linalg.norm(Vc[list(f)].mean(0)) for f in F]
    done = [False] * len(F)
    reps = []
    for fi in range(len(F)):
        if done[fi]:
            continue
        reps.append(fi)
        done[fi] = True
        for R in G:
            img = R @ cd[fi]
            for fj in range(len(F)):
                if (not done[fj] and len(F[fj]) == len(F[fi])
                        and np.linalg.norm(cd[fj] - img) < 1e-6):
                    done[fj] = True
    return reps


def _fib_sphere(n):
    pts, ga = [], math.pi * (3 - math.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = math.sqrt(max(0.0, 1 - z * z))
        pts.append(np.array([rr * math.cos(ga * i), rr * math.sin(ga * i), z]))
    return pts


def _tangent(p):
    ref = np.array([1.0, 0, 0]) if abs(p[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(p, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(p, e1)


def _min_edge(V, F):
    m = np.inf
    for f in F:
        k = len(f)
        for i in range(k):
            m = min(m, np.linalg.norm(V[f[i]] - V[f[(i + 1) % k]]))
    return m


def _min_vertex_gap(V):
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return D.min()


def _chiral_convex(V, F, tol=1e-7):
    """Fixed face set is exactly the convex hull AND non-degenerate (no
    collapsed edges / coincident vertices) -- rejects the spurious LM minima
    where all vertices collapse onto the seed corners."""
    if _min_edge(V, F) < 1e-4 or _min_vertex_gap(V) < 1e-4:
        return False
    for n, _, d in _face_planes(V, F):
        if np.max(V @ n) > d + tol:
            return False
    return True


def _chiral_lm(residuals, rep_pts, free, iters=400):
    rep_pts = [p.copy() for p in rep_pts]
    lam, h, ok = 1e-3, 1e-7, False
    for _ in range(iters):
        r0 = residuals(rep_pts)
        if np.max(np.abs(r0)) < 1e-13:
            ok = True
            break
        bases = {o: _tangent(rep_pts[o]) for o in free}
        cols = []
        for o in free:
            for e in bases[o]:
                pp = [p.copy() for p in rep_pts]
                pp[o] = rep_pts[o] + h * e
                pp[o] /= np.linalg.norm(pp[o])
                pm = [p.copy() for p in rep_pts]
                pm[o] = rep_pts[o] - h * e
                pm[o] /= np.linalg.norm(pm[o])
                cols.append((residuals(pp) - residuals(pm)) / (2 * h))
        J = np.array(cols).T
        JtJ, Jtr = J.T @ J, J.T @ r0
        n0 = np.linalg.norm(r0)
        improved = False
        for _try in range(40):
            try:
                step = np.linalg.solve(JtJ + lam * np.eye(len(Jtr)), -Jtr)
            except np.linalg.LinAlgError:
                lam *= 4
                continue
            newp = [p.copy() for p in rep_pts]
            k = 0
            for o in free:
                e1, e2 = bases[o]
                newp[o] = rep_pts[o] + step[k] * e1 + step[k + 1] * e2
                newp[o] /= np.linalg.norm(newp[o])
                k += 2
            if np.linalg.norm(residuals(newp)) < n0:
                rep_pts, lam, improved = newp, max(lam * 0.5, 1e-13), True
                break
            lam *= 4
        if not improved:
            break
        if np.max(np.abs(residuals(rep_pts))) < 1e-13:
            ok = True
            break
    return rep_pts, ok


def solve_chiral(op, seed, restarts=60):
    """Exact biscribed form of a chiral Conway solid: op in {'p','w'} on a
    Platonic seed in {'T','C','O','D','I'}.  Standard {'exists', ...}
    contract; exists=False (with a reason) when only a degenerate
    vertex-collapsing solution exists."""
    cw = _cw()
    V0, F = cw.apply_conway(op + seed)
    V0, F = cw.orient_outward(V0, [list(f) for f in F])
    V0 = np.array(V0, float)
    N = len(V0)
    U = V0 / np.linalg.norm(V0, axis=1, keepdims=True)
    G = _rot_group(seed)
    assign, reps, locked = _vertex_orbits(G, U)
    rep0 = [U[r].copy() for r in reps]
    free = [o for o in range(len(reps)) if not locked[o]]

    def build_V(rp):
        Vout = np.empty((N, 3))
        for i, (o, g) in enumerate(assign):
            Vout[i] = G[g] @ rp[o]
        return Vout

    face_reps = _face_orbit_reps(G, build_V(rep0), F)

    def residuals(rp):
        Vc = build_V(rp)
        planar, dist = [], []
        for fi in face_reps:
            Q = Vc[list(F[fi])]
            c = Q.mean(0)
            _, _, Vt = np.linalg.svd(Q - c)
            n = Vt[2]
            if n @ c < 0:
                n = -n
            planar.extend(((Q - c) @ n).tolist())
            dist.append(c @ n)
        return np.array(planar + [d - dist[0] for d in dist[1:]])

    fib = _fib_sphere(max(restarts, 8) * max(len(free), 1))
    inits = [rep0]
    for k in range(restarts):
        rp = [p.copy() for p in rep0]
        for oi, o in enumerate(free):
            rp[o] = fib[(k * len(free) + oi) % len(fib)]
        inits.append(rp)

    best = None
    for rp0 in inits:
        rp, ok = _chiral_lm(residuals, rp0, free)
        V = build_V(rp)
        rmax = np.max(np.abs(residuals(rp)))
        if ok and rmax < 1e-11 and _chiral_convex(V, F):
            best = (rp, True)
            break
        if best is None or rmax < np.max(np.abs(residuals(best[0]))):
            best = (rp, False)

    rp, converged = best
    V = build_V(rp)
    if not (converged and _chiral_convex(V, F)):
        return {'exists': False, 'why': (
            "only a degenerate (vertex-collapsing) solution exists -- no "
            "non-degenerate biscribed form")}
    r = float(np.mean([d for _, _, d in _face_planes(V, F)]))
    return {'exists': True,
            'param': f"{len(free)} free orbit(s), {2 * len(free)} params",
            'verts': V, 'faces': [list(f) for f in F], 'r_over_R': r}


# --------------------------------------------------------------------------
# Generalized chiral biscriber for NON-Platonic seeds (propello / snub of the
# truncated / orthokis forms).  Same K-orbit machinery, but the solid comes
# from a full Conway string (or a prebuilt mesh), the rotation group is that
# of the base symmetry (Conway ops preserve symmetry), and -- since these can
# have several biscribed roots -- it returns the LARGEST-r/R (roundest) one,
# which is McCooey's realization.  These solves are slow, so results are
# precomputed into _biscribed_chiral_data.CHIRAL_DATA (see biscribe_exact).
# --------------------------------------------------------------------------

def solve_chiral_g(spec, sym, restarts=24, canon_iters=400):
    """Exact biscribed form of a chiral solid given by `spec` (a Conway
    notation string built with apply_conway, or a prebuilt (V0, F) mesh) and
    base-symmetry Platonic char `sym` (whose proper rotation group is the
    solid's).  Returns the roundest biscribed root among seed + canonical +
    Fibonacci-restart inits.  Standard {'exists', ...} contract."""
    cw = _cw()
    V0, F = cw.apply_conway(spec) if isinstance(spec, str) else spec
    V0, F = cw.orient_outward([list(v) for v in V0], [list(f) for f in F])
    V0 = np.array(V0, float)
    N = len(V0)
    U = V0 / np.linalg.norm(V0, axis=1, keepdims=True)
    G = _rot_group(sym)
    assign, reps, locked = _vertex_orbits(G, U)
    rep0 = [U[r].copy() for r in reps]
    free = [o for o in range(len(reps)) if not locked[o]]

    def build_V(rp):
        Vout = np.empty((N, 3))
        for i, (o, g) in enumerate(assign):
            Vout[i] = G[g] @ rp[o]
        return Vout

    face_reps = _face_orbit_reps(G, build_V(rep0), F)

    def residuals(rp):
        Vc = build_V(rp)
        planar, dist = [], []
        for fi in face_reps:
            Q = Vc[list(F[fi])]
            c = Q.mean(0)
            _, _, Vt = np.linalg.svd(Q - c)
            n = Vt[2]
            if n @ c < 0:
                n = -n
            planar.extend(((Q - c) @ n).tolist())
            dist.append(c @ n)
        return np.array(planar + [d - dist[0] for d in dist[1:]])

    inits = [rep0]
    try:                                             # canonical-form init
        Vc = np.array(cw.canonicalize([list(v) for v in V0],
                                      [list(f) for f in F], iters=canon_iters))
        Uc = Vc / np.linalg.norm(Vc, axis=1, keepdims=True)
        inits.append([Uc[r].copy() for r in reps])
    except Exception:
        pass
    fib = _fib_sphere(max(restarts, 8) * max(len(free), 1))
    for k in range(restarts):
        rp = [p.copy() for p in rep0]
        for oi, o in enumerate(free):
            rp[o] = fib[(k * len(free) + oi) % len(fib)]
        inits.append(rp)

    best = None                                      # (r/R, rep) of largest r
    for rp0 in inits:
        rp, ok = _chiral_lm(residuals, rp0, free, iters=800)
        V = build_V(rp)
        if np.max(np.abs(residuals(rp))) < 1e-11 and _chiral_convex(V, F):
            r = float(np.mean([d for _, _, d in _face_planes(V, F)]))
            if best is None or r > best[0]:
                best = (r, [p.copy() for p in rp])
    if best is None:
        return {'exists': False,
                'why': "no non-degenerate convex biscribed form found"}
    r, rp = best
    V = build_V(rp)
    return {'exists': True,
            'param': f"{len(free)} free orbit(s), {2 * len(free)} params",
            'verts': V, 'faces': [list(f) for f in F], 'r_over_R': r}


def _orthokis_mesh(seed):
    """kis ONLY the central n-gon faces (those on the n-fold axes) of the
    biscribed propello solid pX -- seeds solve_chiral_g in the basin of the
    biscribed orthokis-propello root."""
    cw = _cw()
    base = {'C': 'propello_cube', 'D': 'propello_dodecahedron'}[seed]
    b = biscribe_exact(base)
    V, F = cw.orient_outward([list(v) for v in b[0]], [list(f) for f in b[1]])
    V = np.array(V, float)
    sV, sF = cw._seed(seed, 0)
    sV = np.array(sV, float)
    axes = np.array([sV[list(f)].mean(0) / np.linalg.norm(sV[list(f)].mean(0))
                     for f in sF])
    Rmean = float(np.linalg.norm(V, axis=1).mean())
    NV, NF = [list(v) for v in V], []
    for f in F:
        cn = V[list(f)].mean(0)
        cn = cn / np.linalg.norm(cn)
        if np.max(axes @ cn) > 1 - 1e-6:             # central axis face -> kis
            ai = len(NV)
            NV.append(list(cn * Rmean))
            m = len(f)
            for i in range(m):
                NF.append([f[i], f[(i + 1) % m], ai])
        else:
            NF.append(list(f))
    return np.array(NV, float), [list(f) for f in NF]


# non-Platonic-seed chiral solids that biscribe: (builder, base symmetry)
_CHIRAL_SPECS = {
    'orthokis_propello_cube':           (lambda: _orthokis_mesh('C'), 'C'),
    'orthokis_propello_dodecahedron':   (lambda: _orthokis_mesh('D'), 'D'),
    'propello_truncated_octahedron':    (lambda: 'ptO', 'O'),
    'propello_truncated_cuboctahedron': (lambda: 'pbC', 'C'),
    'propello_truncated_icosahedron':   (lambda: 'ptI', 'I'),
    'snub_truncated_octahedron':        (lambda: 'stO', 'O'),
    'snub_truncated_icosahedron':       (lambda: 'stI', 'I'),
}

# ---- The 4 "hard" chiral biscribed solids (SOLVED, reciprocal solver) -------
# #20 Propello Truncated Icosidodecahedron ('pbD', 480 verts, I) + #24 its dual
# (Propello Disdyakis Triacontahedron); #27 L-Propello L-Snub Cube (propello of
# the MIRRORED snub cube -- the matching-handedness class; the opposite class
# 'psC' has no convex biscribed root) + #31 its dual.  History: solve_chiral_g's
# roundest-root LM never found their basin, and a plain projection biscriber
# (push faces to r = mean(d_f)) collapses to the DEGENERATE r->0 fixed point.
# The fix is biscribe_reciprocal below: reset all face-plane distances to a
# FIXED r > 0 each step (the "dual poles on a sphere of radius 1/r" constraint
# -- r never moves, so the r->0 collapse is impossible), rebuild each vertex
# from its incident planes by least squares, project onto the symmetric
# subspace, then polish with the orbit LM (r free) so the final coordinates are
# self-derived roots of the pure equal-distance equations.  Init that selects
# the right basin: for #20, propello of the EXACT biscribed truncated
# icosidodecahedron; for #27, propello of the MIRRORED snub cube.  All four
# verify std|v|/std(face-dist)/planar ~1e-13 with r/R matching McCooey; coords
# are in _biscribed_chiral_data (#24/#31 via polar reciprocation).  This
# completes the biscribed chiral catalog at 31/31.
# ---------------------------------------------------------------------------


def _regen_chiral_data():
    """Re-derive _biscribed_chiral_data.CHIRAL_DATA from the solver (the
    embedded coords are exactly this).  Returns {key: {'r','V','F'}}."""
    data = {}
    for key, (mk, sym) in _CHIRAL_SPECS.items():
        res = solve_chiral_g(mk(), sym)
        if res['exists']:
            data[key] = {'r': float(res['r_over_R']),
                         'V': [[float(c) for c in v] for v in res['verts']],
                         'F': [list(map(int, f)) for f in res['faces']]}
    return data


# --------------------------------------------------------------------------
# Reciprocal (dual-sphere) biscriber -- solves the 4 "hard" chiral solids that
# the LM roundest-root search misses and the plain projection collapses on
# (see the note above).  Resetting every face distance to a FIXED r each step
# is the "dual poles on a sphere of radius 1/r" constraint, which makes the
# r->0 collapse impossible; a final orbit-LM polish (r free) yields the exact
# self-derived root.
# --------------------------------------------------------------------------

def _face_groups(F):
    gr = {}
    for fi, f in enumerate(F):
        gr.setdefault(len(f), []).append(fi)
    return {k: (np.array(v), np.array([F[fi] for fi in v]))
            for k, v in gr.items()}


def _fit_planes(V, F, FG):
    n = np.empty((len(F), 3))
    d = np.empty(len(F))
    for k, (idx, fk) in FG.items():
        Q = V[fk]
        c = Q.mean(1)
        nrm = np.cross(Q, np.roll(Q, -1, axis=1)).sum(1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        flip = np.einsum('ij,ij->i', nrm, c) < 0
        nrm[flip] *= -1
        n[idx] = nrm
        d[idx] = np.einsum('ij,ij->i', nrm, c)
    return n, d


def _vert_face_incidence(nV, F):
    vf = [[] for _ in range(nV)]
    for fi, f in enumerate(F):
        for v in f:
            vf[v].append(fi)
    deg = [len(x) for x in vf]
    groups = {}
    for v in range(nV):
        groups.setdefault(deg[v], []).append(v)
    return {k: (np.array(vs), np.array([vf[v] for v in vs]))
            for k, vs in groups.items()}


def _reconstruct(nV, VF, n, r):
    """v = argmin sum_f (n_f.v - r)^2 over incident faces (batched 3x3)."""
    V = np.empty((nV, 3))
    for k, (vs, fidx) in VF.items():
        N = n[fidx]
        M = np.einsum('gki,gkj->gij', N, N)
        b = r * N.sum(1)
        V[vs] = np.linalg.solve(M, b[:, :, None])[:, :, 0]
    return V


def biscribe_reciprocal(V0, F, sym, r_target, iters=6000, alpha=0.7,
                        tol=1e-13):
    """Reciprocal dual-sphere biscriber + orbit-LM polish.  V0 is a near-target
    symmetric init; r_target selects the basin (the true r is found free by the
    polish).  Returns the standard {'exists', ...} dict."""
    G = _rot_group(sym)
    U = V0 / np.linalg.norm(V0, axis=1, keepdims=True)
    assign, reps, locked = _vertex_orbits(G, U)
    Ga = np.array(G)
    ov = np.array([a[0] for a in assign])
    gv = np.array([a[1] for a in assign])
    n_orb = len(reps)
    FG = _face_groups(F)
    VF = _vert_face_incidence(len(V0), F)

    def project(V):
        back = np.einsum('ikj,ik->ij', Ga[gv], V)     # G^T @ v
        rep = np.zeros((n_orb, 3))
        cnt = np.zeros(n_orb)
        np.add.at(rep, ov, back)
        np.add.at(cnt, ov, 1)
        rep /= cnt[:, None]
        return np.einsum('ijk,ik->ij', Ga[gv], rep[ov])

    V = U.copy()
    for it in range(iters):
        n, d = _fit_planes(V, F, FG)
        Vn = _reconstruct(len(V), VF, n, r_target)
        V = (1 - alpha) * V + alpha * Vn
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        V = project(V)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        if it % 50 == 0:
            _, sd, pl = verify(V, F)
            if sd < tol and pl < tol:
                break
    # orbit-LM polish (r free) -> exact self-derived root
    rep0 = [V[r].copy() for r in reps]
    free = [o for o in range(len(reps)) if not locked[o]]
    N = len(V)

    def build_V(rp):
        Vo = np.empty((N, 3))
        for i, (o, g) in enumerate(assign):
            Vo[i] = G[g] @ rp[o]
        return Vo

    face_reps = _face_orbit_reps(G, build_V(rep0), F)

    def residuals(rp):
        Vc = build_V(rp)
        planar, dist = [], []
        for fi in face_reps:
            Q = Vc[list(F[fi])]
            c = Q.mean(0)
            _, _, Vt = np.linalg.svd(Q - c)
            nn = Vt[2]
            if nn @ c < 0:
                nn = -nn
            planar.extend(((Q - c) @ nn).tolist())
            dist.append(c @ nn)
        return np.array(planar + [dd - dist[0] for dd in dist[1:]])

    rp, ok = _chiral_lm(residuals, rep0, free, iters=800)
    Vf = build_V(rp)
    if not (np.max(np.abs(residuals(rp))) < 1e-9 and _chiral_convex(Vf, F)):
        return {'exists': False, 'why': "reciprocal polish did not converge"}
    r = float(np.mean([dd for _, _, dd in _face_planes(Vf, F)]))
    return {'exists': True, 'param': "reciprocal + orbit-LM polish",
            'verts': Vf, 'faces': [list(f) for f in F], 'r_over_R': r}


def _hard_mesh(key):
    """Basin-selecting init for the two hard chiral primals."""
    cw = _cw()
    if key == 'propello_truncated_icosidodecahedron':   # propello of biscribed bD
        b = biscribe_exact('truncated_icosidodecahedron')
        V, F = cw.op_propellor([list(map(float, v)) for v in b[0]],
                               [list(f) for f in b[1]])
    elif key == 'propello_l_snub_cube':                 # propello of MIRRORED sC
        V, F = cw.apply_conway('sC')
        V = [[v[0], v[1], -v[2]] for v in V]
        F = [list(f[::-1]) for f in F]
        V, F = cw.op_propellor(V, F)
    else:
        raise ValueError(key)
    V, F = cw.orient_outward([list(v) for v in V], [list(f) for f in F])
    return np.array(V, float), [list(f) for f in F]


# key -> (init builder, base symmetry, r_target basin selector)
_CHIRAL_HARD = {
    'propello_truncated_icosidodecahedron': ('D', 0.986706162392544289),
    'propello_l_snub_cube':                 ('C', 0.970598205840632),
}


def _regen_chiral_hard():
    """Re-derive the 2 hard chiral primals via biscribe_reciprocal (their
    embedded coords are exactly this; #24/#31 follow by polar reciprocation)."""
    data = {}
    for key, (sym, rt) in _CHIRAL_HARD.items():
        V0, F = _hard_mesh(key)
        res = biscribe_reciprocal(V0, F, sym, rt)
        if res['exists']:
            data[key] = {'r': float(res['r_over_R']),
                         'V': [[float(c) for c in v] for v in res['verts']],
                         'F': [list(map(int, f)) for f in res['faces']]}
    return data


# --------------------------------------------------------------------------
# Duals by polar reciprocation (rho^2 = R*r; biscribed -> biscribed dual)
# --------------------------------------------------------------------------

def polar_dual(V, F, r_over_R):
    """Biscribed solid scaled to R=1 -> its dual, also biscribed with the
    same R=1 and r.  Dual vertices are the poles of the face planes; dual
    faces are the ordered face cycles around each vertex."""
    V = np.asarray(V, float)
    planes = _face_planes(V, F)
    rho2 = 1.0 * r_over_R                       # R * r with R = 1
    DV = np.array([n * (rho2 / d) for n, _, d in planes])
    e2f, nxt = {}, {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            e2f[(a, b)] = fi
            nxt[(fi, a)] = b
    v2f = {}
    for fi, f in enumerate(F):
        for a in f:
            v2f.setdefault(a, fi)
    DF = []
    for a in range(len(V)):
        cyc, fi = [], v2f[a]
        while True:
            cyc.append(fi)
            fi = e2f[(nxt[(fi, a)], a)]
            if fi == v2f[a]:
                break
        DF.append(cyc)
    # rescale so the dual circumsphere is 1 (it already is: |DV| = rho2/r = 1)
    R = np.linalg.norm(DV, axis=1).max()
    return DV / R, [list(f) for f in DF], rho2 / R


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

_SOLVERS = {
    'truncated_tetrahedron':       lambda: solve_truncation('T'),
    'truncated_cube':              lambda: solve_truncation('C'),
    'truncated_octahedron':        lambda: solve_truncation('O'),
    'truncated_dodecahedron':      lambda: solve_truncation('D'),
    'truncated_icosahedron':       lambda: solve_truncation('I'),
    'cuboctahedron':               lambda: solve_rectified('C'),
    'icosidodecahedron':           lambda: solve_rectified('D'),
    'truncated_cuboctahedron':     lambda: solve_omnitruncate('O'),
    'truncated_icosidodecahedron': lambda: solve_omnitruncate('I'),
    'snub_cube':                   lambda: solve_snub('O'),
    'snub_dodecahedron':           lambda: solve_snub('I'),
    'propello_tetrahedron':        lambda: solve_chiral('p', 'T'),
    'propello_cube':               lambda: solve_chiral('p', 'C'),
    'propello_dodecahedron':       lambda: solve_chiral('p', 'D'),
    'hexpropello_cube':            lambda: solve_chiral('w', 'C'),
    'hexpropello_dodecahedron':    lambda: solve_chiral('w', 'D'),
}
# non-Platonic-seed chiral solvers (the slow, roundest-root ones); their
# results are normally served from the embedded _CHIRAL_DATA cache, with these
# as the reproducible fallback.
_SOLVERS.update({
    _k: (lambda mk=_mk, sym=_sym: solve_chiral_g(mk(), sym))
    for _k, (_mk, _sym) in _CHIRAL_SPECS.items()
})
# the 2 "hard" chiral primals -- served from _CHIRAL_DATA, this reciprocal
# solver is the reproducible fallback
_SOLVERS.update({
    _k: (lambda key=_k, sym=_s, rt=_r:
         biscribe_reciprocal(*_hard_mesh(key), sym, rt))
    for _k, (_s, _r) in _CHIRAL_HARD.items()
})


def _norm_name(name):
    return name.strip().lower().replace(' ', '_').replace('-', '_')


_CACHE = {}          # solver results memoized (the chiral solves are slow)


def biscribe_exact(name):
    """Exact biscribed form of the named solid, or None if none exists.
    Returns (verts, faces, r_over_R) with circumradius scaled to 1:
    verts = list of (x, y, z), faces = list of vertex-index tuples.
    Results are memoized (the chiral propello/hexpropello solves are slow)."""
    key = _norm_name(name)
    if key in _CACHE:
        return _CACHE[key]
    if key in _CHIRAL_DATA:                 # precomputed roundest-root coords
        d = _CHIRAL_DATA[key]
        out = ([tuple(map(float, v)) for v in d['V']],
               [tuple(f) for f in d['F']], float(d['r']))
        _CACHE[key] = out
        return out
    if key not in _SOLVERS:
        raise ValueError(f"unknown solid {name!r}; know {sorted(_SOLVERS)}")
    res = _SOLVERS[key]()
    if not res['exists']:
        _CACHE[key] = None
        return None
    V, F = res['verts'], res['faces']
    out = ([tuple(map(float, v)) for v in V],
           [tuple(f) for f in F], float(res['r_over_R']))
    _CACHE[key] = out
    return out


def biscribe_exact_dual(name):
    """Dual of the biscribed solid (polar reciprocation about rho^2 = R*r),
    itself biscribed with the same r/R.  None if the base has no
    biscribed form."""
    base = biscribe_exact(name)
    if base is None:
        return None
    V, F, r = polar_dual(np.array(base[0]), [list(f) for f in base[1]],
                         base[2])
    return ([tuple(map(float, v)) for v in V],
            [tuple(f) for f in F], float(r))


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

# (id, label, base-solver name, is_dual) -- the 8 non-chiral biscribed
# solids + the 2 chiral snubs + their duals (McCooey's catalog).
_BISCRIBED = [
    ('tO', "Biscribed Truncated Octahedron", 'truncated_octahedron', False),
    ('tI', "Biscribed Truncated Icosahedron", 'truncated_icosahedron', False),
    ('tCO', "Biscribed Truncated Cuboctahedron",
     'truncated_cuboctahedron', False),
    ('tID', "Biscribed Truncated Icosidodecahedron",
     'truncated_icosidodecahedron', False),
    ('sC', "Biscribed Snub Cube", 'snub_cube', False),
    ('sD', "Biscribed Snub Dodecahedron", 'snub_dodecahedron', False),
    ('kH', "Biscribed Tetrakis Hexahedron", 'truncated_octahedron', True),
    ('kD', "Biscribed Pentakis Dodecahedron", 'truncated_icosahedron', True),
    ('mD', "Biscribed Disdyakis Dodecahedron",
     'truncated_cuboctahedron', True),
    ('mT', "Biscribed Disdyakis Triacontahedron",
     'truncated_icosidodecahedron', True),
    ('pI', "Biscribed Pentagonal Icositetrahedron", 'snub_cube', True),
    ('pH', "Biscribed Pentagonal Hexecontahedron",
     'snub_dodecahedron', True),
    # Chiral: propello family (p commutes with dual, so pO = dual(pC),
    # pI = dual(pD), and the tetrahedral form is self-dual)
    ('ppT', "Biscribed Propello Tetrahedron", 'propello_tetrahedron', False),
    ('ppC', "Biscribed Propello Cube", 'propello_cube', False),
    ('ppO', "Biscribed Propello Octahedron", 'propello_cube', True),
    ('ppD', "Biscribed Propello Dodecahedron", 'propello_dodecahedron', False),
    ('ppI', "Biscribed Propello Icosahedron", 'propello_dodecahedron', True),
    # Chiral: hexpropello (whirl) family -- only C and D biscribe
    # non-degenerately (whirl does not commute with dual, so the duals are
    # distinct solids)
    ('whC', "Biscribed Hexpropello Cube", 'hexpropello_cube', False),
    ('dwC', "Biscribed Dual Hexpropello Cube", 'hexpropello_cube', True),
    ('whD', "Biscribed Hexpropello Dodecahedron",
     'hexpropello_dodecahedron', False),
    ('dwD', "Biscribed Dual Hexpropello Dodecahedron",
     'hexpropello_dodecahedron', True),
    # Chiral on non-Platonic seeds (propello / snub of the truncated /
    # orthokis forms) -- coords served from _CHIRAL_DATA, duals via polar
    # reciprocation
    ('okpC', "Biscribed Orthokis Propello Cube",
     'orthokis_propello_cube', False),
    ('otpO', "Biscribed Orthotruncated Propello Octahedron",
     'orthokis_propello_cube', True),
    ('okpD', "Biscribed Orthokis Propello Dodecahedron",
     'orthokis_propello_dodecahedron', False),
    ('otpI', "Biscribed Orthotruncated Propello Icosahedron",
     'orthokis_propello_dodecahedron', True),
    ('ptrO', "Biscribed Propello Truncated Octahedron",
     'propello_truncated_octahedron', False),
    ('pkH', "Biscribed Propello Tetrakis Hexahedron",
     'propello_truncated_octahedron', True),
    ('ptrCO', "Biscribed Propello Truncated Cuboctahedron",
     'propello_truncated_cuboctahedron', False),
    ('pmD', "Biscribed Propello Disdyakis Dodecahedron",
     'propello_truncated_cuboctahedron', True),
    ('ptrI', "Biscribed Propello Truncated Icosahedron",
     'propello_truncated_icosahedron', False),
    ('pkD', "Biscribed Propello Pentakis Dodecahedron",
     'propello_truncated_icosahedron', True),
    ('sntO', "Biscribed Snub Truncated Octahedron",
     'snub_truncated_octahedron', False),
    ('dsntO', "Biscribed Dual Snub Truncated Octahedron",
     'snub_truncated_octahedron', True),
    ('sntI', "Biscribed Snub Truncated Icosahedron",
     'snub_truncated_icosahedron', False),
    ('dsntI', "Biscribed Dual Snub Truncated Icosahedron",
     'snub_truncated_icosahedron', True),
    # the 4 "hard" chiral solids (reciprocal solver) -- completes 31/31
    ('ptID', "Biscribed Propello Truncated Icosidodecahedron",
     'propello_truncated_icosidodecahedron', False),
    ('pmT', "Biscribed Propello Disdyakis Triacontahedron",
     'propello_truncated_icosidodecahedron', True),
    ('plsC', "Biscribed L-Propello L-Snub Cube",
     'propello_l_snub_cube', False),
    ('plpI', "Biscribed L-Propello R-Pentagonal Icositetrahedron",
     'propello_l_snub_cube', True),
]

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PALETTE = {3: (0.90, 0.36, 0.23), 4: (0.27, 0.52, 0.79),
                5: (0.30, 0.69, 0.42), 6: (0.95, 0.77, 0.29),
                8: (0.25, 0.72, 0.72), 10: (0.55, 0.60, 0.29)}

    def _material_for(n):
        name = f"Biscribed {n}-gon"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            import colorsys
            rgb = _PALETTE.get(n, colorsys.hsv_to_rgb((n * 0.618) % 1.0,
                                                      0.55, 0.8))
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        return mat

    class MESH_OT_biscribed_solid_add(bpy.types.Operator):
        """Add a biscribed solid: vertices on a circumsphere AND faces
        tangent to a concentric insphere (exact symmetric construction)"""
        bl_idname = "mesh.biscribed_solid_add"
        bl_label = "Biscribed Solid"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(
            name="Solid",
            items=[(sid, lbl, "") for sid, lbl, _b, _d in _BISCRIBED])
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "By Face Size", ""), ('NONE', "None", "")],
            default='SIDES')
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed polyhedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "modifier"),
                   ('WIRE', "Struts", "Wireframe modifier"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness")
        facet_depth: FloatProperty(name="Depth", default=0.15, min=0.01,
                                   max=2.0,
                                   description="Face Segments inward depth")
        facet_gap: FloatProperty(name="Bevel Gap", default=0.0, min=0.0,
                                 max=0.5,
                                 description="Gap between face segments")
        facet_explode: FloatProperty(name="Explode", default=0.1, min=0.0,
                                     max=5.0,
                                     description="Move segments outward")
        facet_separate: BoolProperty(
            name="Separate Meshes", default=False,
            description="Each face segment as its own object")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            lay.prop(self, 'coloring')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'FACETS':
                lay.prop(self, 'facet_depth')
                lay.prop(self, 'facet_gap')
                lay.prop(self, 'facet_explode')
                lay.prop(self, 'facet_separate')
            lay.prop(self, 'scale')

        def execute(self, context):
            if np is None:
                self.report({'ERROR'}, "biscribed solids need NumPy")
                return {'CANCELLED'}
            row = next(r for r in _BISCRIBED if r[0] == self.solid)
            _sid, label, base, is_dual = row
            res = (biscribe_exact_dual(base) if is_dual
                   else biscribe_exact(base))
            if res is None:
                self.report({'ERROR'}, "no biscribed form")
                return {'CANCELLED'}
            V, F, _r = res
            if self.style == 'FACETS':
                try:
                    from . import facet_style
                except ImportError:
                    import facet_style
                Vf = [tuple(c * self.scale for c in v) for v in V]
                mat = (_material_for
                       if self.coloring == 'SIDES' else None)
                facet_style.emit_facets(
                    context, Vf, [list(f) for f in F], label,
                    self.facet_depth, self.facet_gap,
                    self.facet_explode, self.facet_separate, mat)
                self.report({'INFO'}, f"{label}: {len(F)} face segments")
                return {'FINISHED'}
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            if self.coloring == 'SIDES' and len(me.polygons) == len(F):
                sizes = sorted({len(f) for f in F})
                slot = {n: i for i, n in enumerate(sizes)}
                for n in sizes:
                    me.materials.append(_material_for(n))
                me.polygons.foreach_set('material_index',
                                        [slot[len(f)] for f in F])
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness
                mod.use_even_offset = False
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'}, f"{label}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.biscribed_solid_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_biscribed_solid_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_biscribed_solid_add)


# --------------------------------------------------------------------------
# Self-test / validation table
# --------------------------------------------------------------------------

def _selftest():
    order = ['truncated_tetrahedron', 'truncated_cube',
             'truncated_octahedron', 'truncated_dodecahedron',
             'truncated_icosahedron', 'cuboctahedron', 'icosidodecahedron',
             'truncated_cuboctahedron', 'truncated_icosidodecahedron',
             'snub_cube', 'snub_dodecahedron']
    expected = {                       # (t*, r/R) from prior research
        'truncated_octahedron':  (0.42265, 0.8069),
        'truncated_icosahedron': (0.37147, 0.9226),
    }
    must_fail = {'truncated_tetrahedron', 'truncated_cube',
                 'truncated_dodecahedron', 'cuboctahedron',
                 'icosidodecahedron'}
    print(f"{'solid':30s} {'param':22s} {'r/R':>8s}  verdict")
    print('-' * 96)
    failures = []
    for name in order:
        res = _SOLVERS[name]()
        if not res['exists']:
            print(f"{name:30s} {'--':22s} {'--':>8s}  "
                  f"NO biscribed form  ({res['why']})")
            if name not in must_fail:
                failures.append(f"{name}: expected biscribed form")
            continue
        V, F, r = res['verts'], res['faces'], res['r_over_R']
        sR, sd, pl = verify(V, F)
        ok = sR < 1e-10 and sd < 1e-10 and pl < 1e-10
        print(f"{name:30s} {res['param']:22s} {r:8.4f}  BISCRIBED  "
              f"V={len(V)} F={len(F)}  std|v|={sR:.1e} "
              f"std d_f={sd:.1e} planar={pl:.1e}")
        if not ok:
            failures.append(f"{name}: verification stds too large")
        if name in must_fail:
            failures.append(f"{name}: should have NO biscribed form")
        if name in expected:
            t_exp, r_exp = expected[name]
            if abs(res['t'] - t_exp) > 5e-5:
                failures.append(f"{name}: t*={res['t']:.6f} != {t_exp}")
            if abs(r - r_exp) > 5e-5:
                failures.append(f"{name}: r/R={r:.6f} != {r_exp}")
        # dual, via polar reciprocation
        DV, DF, dr = polar_dual(V, F, r)
        sR, sd, pl = verify(DV, DF)
        print(f"{'  dual':30s} {'polar rho^2=R*r':22s} {dr:8.4f}  BISCRIBED  "
              f"V={len(DV)} F={len(DF)}  std|v|={sR:.1e} "
              f"std d_f={sd:.1e} planar={pl:.1e}")
        if not (sR < 1e-10 and sd < 1e-10 and pl < 1e-10):
            failures.append(f"{name} dual: verification stds too large")
    print('-' * 96)
    print("chiral (propello / hexpropello):")
    chiral = [("Propello Tetrahedron", 'p', 'T'),
              ("Propello Cube", 'p', 'C'),
              ("Propello Dodecahedron", 'p', 'D'),
              ("Hexpropello Cube", 'w', 'C'),
              ("Hexpropello Dodecahedron", 'w', 'D')]
    for nm, op, sd in chiral:
        res = solve_chiral(op, sd)
        if not res['exists']:
            failures.append(f"{nm}: expected biscribed form ({res['why']})")
            print(f"{nm:30s} NO biscribed form  (UNEXPECTED)")
            continue
        V, F, r = res['verts'], res['faces'], res['r_over_R']
        sR, sd2, pl = verify(V, F)
        print(f"{nm:30s} {res['param']:22s} {r:8.4f}  BISCRIBED  "
              f"V={len(V)} F={len(F)}  std|v|={sR:.1e} std d_f={sd2:.1e} "
              f"planar={pl:.1e}")
        if not (sR < 1e-9 and sd2 < 1e-9 and pl < 1e-9):
            failures.append(f"{nm}: verification stds too large")
        DV, DF, dr = polar_dual(V, F, r)
        dR, dd, dp = verify(DV, DF)
        if not (dR < 1e-9 and dd < 1e-9 and dp < 1e-9):
            failures.append(f"{nm} dual: verification stds too large")
    # honest non-coverage: whirl of T/O/I has no non-degenerate biscribed form
    res = solve_chiral('w', 'O')
    print(f"{'Hexpropello Octahedron':30s} "
          f"{'NO form (correct)' if not res['exists'] else 'HAS FORM (BAD)'}")
    if res['exists']:
        failures.append("hexpropello_octahedron: should have NO biscribed form")
    print('-' * 96)
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        raise AssertionError("biscribed self-test failed")
    print("All validation checks passed.")
