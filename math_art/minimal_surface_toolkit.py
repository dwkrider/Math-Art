
# Minimal Surface Toolkit for Blender
#
# Three generators in one add-on:
#
# 1. Classic parametric minimal surfaces (Enneper & higher orders,
#    catenoid, helicoid, Henneberg, Catalan, Bour, Richmond, Scherk's
#    doubly-periodic graph) -- after Juergen Meier's gallery
#    (3d-meier.de, tut25).
#
# 2. Triply-periodic minimal surfaces via their standard nodal
#    (level-set) approximations, meshed by marching tetrahedra:
#    Schwarz P & D, Schoen's Gyroid / I-WP / F-RD, Neovius, Lidinoid,
#    Split P, plus the singly-periodic Scherk tower. Inventory after
#    Ken Brakke's periodic-surface pages (kenbrakke.com/evolver).
#
# 3. A Plateau-problem solver -- a lightweight, in-Blender take on what
#    Brakke's Surface Evolver does: pin one or two boundary curves and
#    minimize surface area (Pinkall-Polthier cotangent-Laplacian
#    iteration solved with conjugate gradients). Includes the classic
#    "minimal surface between a circle and a torus knot" construction
#    (trefoil by default).
#
# Geometry only; materials and rendering are left to Blender.

bl_info = {
    "name": "Minimal Surface Toolkit",
    "author": "David Krider (Math Art project)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Minimal Surfaces / N-panel 'Minimal Surfaces'",
    "description": "Parametric & triply-periodic minimal surfaces, and a "
                   "Plateau solver spanning minimal surfaces on curves",
    "category": "Add Mesh",
}

import math
import numpy as np

TAU = 2.0 * math.pi


# ==========================================================================
# 1. Parametric classic surfaces
# ==========================================================================
# Each entry: (label, builder(params) -> (V (n,3) ndarray, quads list,
# wrap_u, wrap_v)); u along axis 0, v along axis 1 of the grid.

def _grid(nu, nv, u0, u1, v0, v1):
    u = np.linspace(u0, u1, nu)
    v = np.linspace(v0, v1, nv)
    return np.meshgrid(u, v, indexing='ij')


def _enneper(nu, nv, order, radius):
    U, V = _grid(nu, nv, 1e-4, radius, 0.0, TAU)
    z = U * np.exp(1j * V)
    n = order
    x = np.real(z - z ** (2 * n + 1) / (2 * n + 1))
    y = -np.imag(z + z ** (2 * n + 1) / (2 * n + 1))
    w = (2.0 / (n + 1)) * np.real(z ** (n + 1))
    return x, y, w, False, True


def _catenoid(nu, nv, order, radius):
    c = 1.0
    U, V = _grid(nu, nv, 0.0, TAU, -radius, radius)
    x = c * np.cosh(V / c) * np.cos(U)
    y = c * np.cosh(V / c) * np.sin(U)
    return x, y, V, True, False


def _helicoid(nu, nv, order, radius):
    turns = max(1, order)
    U, V = _grid(nu, nv, -turns * math.pi, turns * math.pi, -radius, radius)
    return V * np.cos(U), V * np.sin(U), 0.6 * U, False, False


def _henneberg(nu, nv, order, radius):
    U, V = _grid(nu, nv, 1e-3, 0.4 * radius, 0.0, TAU)
    x = 2 * np.sinh(U) * np.cos(V) - (2.0 / 3.0) * np.sinh(3 * U) * np.cos(3 * V)
    y = 2 * np.sinh(U) * np.sin(V) + (2.0 / 3.0) * np.sinh(3 * U) * np.sin(3 * V)
    w = 2 * np.cosh(2 * U) * np.cos(2 * V)
    return x, y, w, False, True


def _catalan(nu, nv, order, radius):
    U, V = _grid(nu, nv, -math.pi, 3 * math.pi, -radius, radius)
    x = U - np.sin(U) * np.cosh(V)
    y = 1 - np.cos(U) * np.cosh(V)
    w = 4 * np.sin(U / 2) * np.sinh(V / 2)
    return x, y, w, False, False


def _bour(nu, nv, order, radius):
    U, V = _grid(nu, nv, 1e-3, radius, 0.0, 2 * TAU)   # double cover closes it
    x = U * np.cos(V) - 0.5 * U ** 2 * np.cos(2 * V)
    y = -U * np.sin(V) - 0.5 * U ** 2 * np.sin(2 * V)
    w = (4.0 / 3.0) * U ** 1.5 * np.cos(1.5 * V)
    return x, y, w, False, True


def _richmond(nu, nv, order, radius):
    U, V = _grid(nu, nv, 0.25, radius + 0.25, 0.0, TAU)
    z = U * np.exp(1j * V)
    x = np.real(-1.0 / (2 * z) - z ** 3 / 6.0)
    y = np.imag(-1.0 / (2 * z) + z ** 3 / 6.0)
    w = np.real(z)
    return x, y, w, False, True


def _scherk_graph(nu, nv, order, radius):
    lim = 0.47 * math.pi
    U, V = _grid(nu, nv, -lim, lim, -lim, lim)
    w = np.log(np.cos(U) / np.cos(V))
    return U, V, w, False, False


PARAMETRIC = {
    'ENNEPER': ("Enneper", _enneper),
    'CATENOID': ("Catenoid", _catenoid),
    'HELICOID': ("Helicoid", _helicoid),
    'HENNEBERG': ("Henneberg", _henneberg),
    'CATALAN': ("Catalan", _catalan),
    'BOUR': ("Bour", _bour),
    'RICHMOND': ("Richmond", _richmond),
    'SCHERK1': ("Scherk (doubly periodic)", _scherk_graph),
}


def build_parametric_grid(kind, nu, nv, order, radius, scale):
    """(nu, nv, 3) point grid plus wrap flags."""
    x, y, w, wrap_u, wrap_v = PARAMETRIC[kind][1](nu, nv, order, radius)
    return np.stack([x, y, w], axis=-1) * scale, wrap_u, wrap_v


def build_parametric(kind, nu, nv, order, radius, scale):
    G, wrap_u, wrap_v = build_parametric_grid(kind, nu, nv, order, radius,
                                              scale)
    V = G.reshape(-1, 3)
    quads = []
    def vid(i, j):
        return i * nv + j
    iu = nu - 1
    for i in range(nu if wrap_u else nu - 1):
        i2 = (i + 1) % nu
        for j in range(nv if wrap_v else nv - 1):
            j2 = (j + 1) % nv
            quads.append((vid(i, j), vid(i2, j), vid(i2, j2), vid(i, j2)))
    return V, quads


# ==========================================================================
# 2. Triply-periodic minimal surfaces (nodal approximations)
# ==========================================================================

def _f_p(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)

def _f_d(x, y, z):
    return (np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z))

def _f_g(x, y, z):
    return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z)
            + np.sin(z) * np.cos(x))

def _f_neovius(x, y, z):
    return (3 * (np.cos(x) + np.cos(y) + np.cos(z))
            + 4 * np.cos(x) * np.cos(y) * np.cos(z))

def _f_iwp(x, y, z):
    return (2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z)
                 + np.cos(z) * np.cos(x))
            - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))

def _f_frd(x, y, z):
    return (4 * np.cos(x) * np.cos(y) * np.cos(z)
            - (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
               + np.cos(2 * z) * np.cos(2 * x)))

def _f_lidinoid(x, y, z):
    return (0.5 * (np.sin(2 * x) * np.cos(y) * np.sin(z)
                   + np.sin(2 * y) * np.cos(z) * np.sin(x)
                   + np.sin(2 * z) * np.cos(x) * np.sin(y)
                   - np.cos(2 * x) * np.cos(2 * y)
                   - np.cos(2 * y) * np.cos(2 * z)
                   - np.cos(2 * z) * np.cos(2 * x)) + 0.15)

def _f_splitp(x, y, z):
    return (1.1 * (np.sin(2 * x) * np.sin(z) * np.cos(y)
                   + np.sin(2 * y) * np.sin(x) * np.cos(z)
                   + np.sin(2 * z) * np.sin(y) * np.cos(x))
            - 0.2 * (np.cos(2 * x) * np.cos(2 * y)
                     + np.cos(2 * y) * np.cos(2 * z)
                     + np.cos(2 * z) * np.cos(2 * x))
            - 0.4 * (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))

def _f_scherk_tower(x, y, z):
    return np.sin(z) - np.sinh(x) * np.sinh(y)

TPMS = {
    'P': ("Schwarz P", _f_p, True),
    'D': ("Schwarz D", _f_d, True),
    'G': ("Gyroid", _f_g, True),
    'NEOVIUS': ("Neovius", _f_neovius, True),
    'IWP': ("Schoen I-WP", _f_iwp, True),
    'FRD': ("Schoen F-RD", _f_frd, True),
    'LIDINOID': ("Lidinoid", _f_lidinoid, True),
    'SPLITP': ("Split P", _f_splitp, True),
    'SCHERKT': ("Scherk Tower (singly periodic)", _f_scherk_tower, False),
}

# cube corners (i,j,k offsets) and a 6-tetrahedra decomposition sharing
# the 0-6 diagonal
_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_TETS = [(0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
         (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6)]
# sign-pattern cases for one tetrahedron: bit set = corner value < 0
_ONE = {1: (0, (1, 2, 3)), 2: (1, (0, 2, 3)),
        4: (2, (0, 1, 3)), 8: (3, (0, 1, 2)),
        14: (0, (1, 2, 3)), 13: (1, (0, 2, 3)),
        11: (2, (0, 1, 3)), 7: (3, (0, 1, 2))}
_TWO = {3: ((0, 1), (2, 3)), 5: ((0, 2), (1, 3)), 9: ((0, 3), (1, 2)),
        6: ((1, 2), (0, 3)), 10: ((1, 3), (0, 2)), 12: ((2, 3), (0, 1))}


def _orientation_flags():
    """Whether each (tet, sign-case) emits triangles wound against
    the field gradient. Calibrated on an exact linear field, where
    the crossing polygon is exactly perpendicular to the gradient --
    so the flags are combinatorial and immune to sliver triangles."""
    flags = {}
    cube = np.array(_CUBE, dtype=float)
    for ti, tet in enumerate(_TETS):
        P = cube[list(tet)]                       # (4,3) corners
        M = P[1:] - P[0]                          # for gradient solve
        for cd in list(_ONE) + list(_TWO):
            f = np.where([cd >> i & 1 for i in range(4)], -1.0, 1.0)
            g = np.linalg.solve(M, f[1:] - f[0])  # exact gradient

            def x(ci, cj):
                t = f[ci] / (f[ci] - f[cj])
                return P[ci] + t * (P[cj] - P[ci])

            if cd in _ONE:
                lone, (o0, o1, o2) = _ONE[cd]
                p0, p1, p2 = x(lone, o0), x(lone, o1), x(lone, o2)
            else:
                (n0, n1), (q0, q1) = _TWO[cd]
                p0, p1, p2 = x(n0, q0), x(n0, q1), x(n1, q1)
            n = np.cross(p1 - p0, p2 - p0)
            flags[(ti, cd)] = float(np.dot(n, g)) < 0.0
    return flags


_ORIENT = None


def marching_tets(field, box_min, box_max, res):
    """Extract the zero level set of `field` on a res[0]xres[1]xres[2]
    sample grid over the box. Returns (verts (n,3), tris (m,3)) with
    triangle winding oriented along the field gradient."""
    nx, ny, nz = (r + 1 for r in res)
    xs = np.linspace(box_min[0], box_max[0], nx)
    ys = np.linspace(box_min[1], box_max[1], ny)
    zs = np.linspace(box_min[2], box_max[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    vals = field(X, Y, Z).ravel()
    # samples landing exactly on the surface (e.g. Schwarz P at the
    # lattice points) produce degenerate crossings; nudge them off
    vals = np.where(np.abs(vals) < 1e-9, 1e-9, vals)
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    ii, jj, kk = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1),
                             np.arange(nz - 1), indexing='ij')
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    def flat(i, j, k):
        return (i * ny + j) * nz + k

    corner = [flat(ii + o[0], jj + o[1], kk + o[2]) for o in _CUBE]

    global _ORIENT
    if _ORIENT is None:
        _ORIENT = _orientation_flags()

    tri_pts = []          # list of (3, ntri, 3) blocks
    for ti, (a, b, c, d) in enumerate(_TETS):
        A, B, C, D = corner[a], corner[b], corner[c], corner[d]
        fa, fb, fc, fd = vals[A], vals[B], vals[C], vals[D]
        code = ((fa < 0).astype(np.int8) | ((fb < 0) << 1)
                | ((fc < 0) << 2) | ((fd < 0) << 3))
        tet = np.stack([A, B, C, D], axis=0)

        def interp(sel, ci, cj):
            ia, ib = tet[ci][sel], tet[cj][sel]
            va, vb = vals[ia], vals[ib]
            t = va / (va - vb)
            return pos[ia] + t[:, None] * (pos[ib] - pos[ia])

        for cd, (lone, others) in _ONE.items():
            sel = np.nonzero(code == cd)[0]
            if len(sel) == 0:
                continue
            p0 = interp(sel, lone, others[0])
            p1 = interp(sel, lone, others[1])
            p2 = interp(sel, lone, others[2])
            if _ORIENT[(ti, cd)]:
                p1, p2 = p2, p1
            tri_pts.append(np.stack([p0, p1, p2], axis=1))
        for cd, ((n0, n1), (pp0, pp1)) in _TWO.items():
            sel = np.nonzero(code == cd)[0]
            if len(sel) == 0:
                continue
            q0 = interp(sel, n0, pp0)
            q1 = interp(sel, n0, pp1)
            q2 = interp(sel, n1, pp1)
            q3 = interp(sel, n1, pp0)
            if _ORIENT[(ti, cd)]:
                q1, q3 = q3, q1
            tri_pts.append(np.stack([q0, q1, q2], axis=1))
            tri_pts.append(np.stack([q0, q2, q3], axis=1))

    if not tri_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    tris_xyz = np.concatenate(tri_pts, axis=0)      # (ntri, 3, 3)

    # weld: quantize and unique
    flatv = tris_xyz.reshape(-1, 3)
    eps = max(np.max(box_max) - np.min(box_min), 1.0) * 1e-6
    keys = np.round(flatv / eps).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_index=False,
                          return_inverse=True)
    order = np.zeros(len(uniq), dtype=np.int64)
    order[inv] = np.arange(len(flatv))              # a representative
    verts = flatv[order]
    tris = inv.reshape(-1, 3)
    good = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2]))
    tris = tris[good]

    return verts, tris


def build_tpms(kind, cells, res_per_cell, scale):
    label, field, triply = TPMS[kind]
    if triply:
        span = cells * TAU
        box_min = (-span / 2, -span / 2, -span / 2)
        box_max = (span / 2, span / 2, span / 2)
        res = (cells * res_per_cell,) * 3
    else:  # Scherk tower: periodic in z only, clip x/y
        w = 2.2
        box_min = (-w, -w, -cells * math.pi)
        box_max = (w, w, cells * math.pi)
        rxy = int(res_per_cell * 1.4)
        res = (rxy, rxy, cells * res_per_cell)
    verts, tris = marching_tets(field, box_min, box_max, res)
    s = scale / TAU  # one period -> `scale` Blender units
    return verts * s, tris


# ==========================================================================
# 3. Plateau solver (area minimization with pinned boundaries)
# ==========================================================================

def resample_loop(pts, m):
    """Uniform-arclength resample of a closed polyline (k,3) -> (m,3)."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    closed = np.vstack([pts, pts[:1]])
    t = np.linspace(0.0, total, m, endpoint=False)
    out = np.empty((m, 3))
    j = 0
    for i, ti in enumerate(t):
        while s[j + 1] < ti:
            j += 1
        f = (ti - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out[i] = closed[j] * (1 - f) + closed[j + 1] * f
    return out


def align_loops(A, B):
    """Cyclic shift + optional reversal of B minimizing sum |A_i - B_i|^2."""
    m = len(A)
    best = (None, 1e30)
    for Bc in (B, B[::-1]):
        # brute-force cyclic shifts (m <= a few hundred)
        for sft in range(m):
            d = np.sum((A - np.roll(Bc, -sft, axis=0)) ** 2)
            if d < best[1]:
                best = (np.roll(Bc, -sft, axis=0), d)
    return best[0]


def _cotan_weights(V, T):
    """Per-edge cotangent weights. Returns (edges (e,2), w (e,))."""
    ijk = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    E = []
    W = []
    for (a, b, c) in ijk:
        u = V[T[:, a]] - V[T[:, c]]
        v = V[T[:, b]] - V[T[:, c]]
        cross = np.cross(u, v)
        denom = np.linalg.norm(cross, axis=1)
        cot = np.einsum('ij,ij->i', u, v) / np.maximum(denom, 1e-12)
        # clamp to positive: keeps the system positive-definite (maximum
        # principle), so the CG solve cannot blow up on degenerate fans
        cot = np.clip(cot, 0.01, 20.0)
        E.append(np.stack([T[:, a], T[:, b]], axis=1))
        W.append(0.5 * cot)
    return np.concatenate(E), np.concatenate(W)


def minimize_area(V, T, fixed, outer_iters=30, cg_tol=1e-8, cg_iters=400,
                  uniform=False):
    """Pinkall-Polthier: repeatedly solve the cotan-Laplace equation for
    the interior vertices (boundary pinned). V modified in place.
    With uniform=True, unit weights are used instead of cotangents --
    a Tutte-style fairing solve that untangles folded regions (at the
    cost of exact minimality; follow with a cotan pass)."""
    n = len(V)
    free = ~fixed
    nfree = int(np.sum(free))
    if nfree == 0:
        return V
    for _ in range(outer_iters):
        E, W = _cotan_weights(V, T)
        if uniform:
            W = np.ones_like(W)
        deg = np.zeros(n)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)

        def matvec_full(Xf):
            y = deg[:, None] * Xf
            np.add.at(y, E[:, 0], -W[:, None] * Xf[E[:, 1]])
            np.add.at(y, E[:, 1], -W[:, None] * Xf[E[:, 0]])
            return y

        Xb = np.where(fixed[:, None], V, 0.0)
        b = -matvec_full(Xb)[free]

        def matvec(xf):
            full = np.zeros((n, 3))
            full[free] = xf
            return matvec_full(full)[free]

        x = V[free].copy()
        r = b - matvec(x)
        p = r.copy()
        rs = np.sum(r * r, axis=0)
        b_norm = max(np.max(np.sum(b * b, axis=0)), 1e-30)
        for _cg in range(cg_iters):
            Ap = matvec(p)
            pAp = np.sum(p * Ap, axis=0)
            alpha = rs / np.where(np.abs(pAp) > 1e-30, pAp, 1e-30)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.sum(r * r, axis=0)
            if np.max(rs_new) < cg_tol * b_norm:
                break
            p = r + (rs_new / np.maximum(rs, 1e-30)) * p
            rs = rs_new
        move = np.max(np.linalg.norm(x - V[free], axis=1))
        V[free] = x
        if move < 1e-6 * max(1.0, np.max(np.abs(V))):
            break
    return V


def relax_normal_flow(V, T, fixed, iters=60, lam=0.4):
    """Mean-curvature flow restricted to the surface normal: pulls a
    (slightly perturbed) net back toward the minimal surface without
    tangential sliding, so a fair control net stays fair. Only suitable
    as a polish -- it cannot perform global reorganization."""
    free = ~fixed
    n = len(V)
    for _ in range(iters):
        E, W = _cotan_weights(V, T)
        lap = np.zeros((n, 3))
        d = V[E[:, 1]] - V[E[:, 0]]
        np.add.at(lap, E[:, 0], W[:, None] * d)
        np.add.at(lap, E[:, 1], -W[:, None] * d)
        wsum = np.zeros(n)
        np.add.at(wsum, E[:, 0], W)
        np.add.at(wsum, E[:, 1], W)
        umb = lap / np.maximum(wsum, 1e-12)[:, None]
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        vn = np.zeros((n, 3))
        np.add.at(vn, T[:, 0], fn)
        np.add.at(vn, T[:, 1], fn)
        np.add.at(vn, T[:, 2], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        move = np.sum(umb * vn, axis=1, keepdims=True) * vn
        V[free] += lam * move[free]
    return V


def fair_grid_2d(G, iters=8, step=0.5):
    """Light Laplacian fairing of an (rows, m, 3) net, cyclic in m,
    end rows pinned. Restores row/column coherence after per-column
    resampling; follow with relax_normal_flow to restore minimality."""
    G = G.copy()
    for _ in range(iters):
        up = np.roll(G, 1, axis=0)
        dn = np.roll(G, -1, axis=0)
        up[0] = G[0]
        dn[-1] = G[-1]
        lt = np.roll(G, 1, axis=1)
        rt = np.roll(G, -1, axis=1)
        avg = (up + dn + lt + rt) / 4.0
        G[1:-1] += step * (avg[1:-1] - G[1:-1])
    return G


def fair_grid_columns(G):
    """Re-sample every column (axis 0) of an (rows, m, 3) grid uniformly
    by arc length. Points stay on their column polylines -- i.e. on the
    solved surface -- but the severe bunching/shear the area solver
    introduces (which makes a NURBS control net ring) is equalized."""
    rows, m, _ = G.shape
    out = np.empty_like(G)
    for i in range(m):
        P = G[:, i, :]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-12:
            out[:, i, :] = P
            continue
        t = np.linspace(0.0, total, rows)
        for a in range(3):
            out[:, i, a] = np.interp(t, s, P[:, a])
    return out


def mesh_area(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return 0.5 * float(np.sum(np.linalg.norm(n, axis=1)))


def _quads_to_tris(quads):
    T = []
    for q in quads:
        if len(q) == 4:
            T.append((q[0], q[1], q[2]))
            T.append((q[0], q[2], q[3]))
        else:
            T.append(tuple(q))
    return np.array(T, dtype=np.int64)


def build_disk_grid(boundary, rings):
    """Initial disk spanning one loop: concentric rings toward centroid."""
    m = len(boundary)
    cen = boundary.mean(axis=0)
    verts = [boundary]
    for r in range(1, rings):
        f = r / rings
        verts.append(boundary * (1 - f) + cen * f)
    V = np.concatenate(verts + [cen[None, :]], axis=0)
    quads = []
    for r in range(rings - 1):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    last = (rings - 1) * m
    cidx = rings * m
    for i in range(m):
        quads.append((last + i, last + (i + 1) % m, cidx))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    return V, quads, fixed


def build_annulus_grid(loopA, loopB, rows):
    """Initial ruled surface between two aligned loops of equal length."""
    m = len(loopA)
    verts = []
    for r in range(rows + 1):
        f = r / rows
        verts.append(loopA * (1 - f) + loopB * f)
    V = np.concatenate(verts, axis=0)
    quads = []
    for r in range(rows):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    fixed[rows * m:] = True
    return V, quads, fixed


def torus_knot(p, q, m, scale=1.0, tube=1.0):
    t = np.linspace(0, TAU, m, endpoint=False)
    r = np.cos(q * t) * tube + 2.0
    return np.stack([r * np.cos(p * t), r * np.sin(p * t),
                     -np.sin(q * t) * tube], axis=1) * scale


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _nurbs_grid_object(context, name, grid, cyclic_u=False,
                           cyclic_v=False, order=4):
        """Create a NURBS surface object from an (nu, nv, 3) control grid.
        `u` runs across rows, `v` along each row. Grids must be built
        row-by-row as separate splines and merged with make_segment (the
        only scriptable way to produce a surface grid in Blender)."""
        grid = np.asarray(grid)
        nu, nv = grid.shape[:2]
        su = bpy.data.curves.new(name, 'SURFACE')
        su.dimensions = '3D'
        for i in range(nu):
            sp = su.splines.new('NURBS')
            sp.points.add(nv - 1)
            flat = np.concatenate(
                [grid[i], np.ones((nv, 1))], axis=1).ravel()
            sp.points.foreach_set('co', flat)
        obj = bpy.data.objects.new(name, su)
        context.collection.objects.link(obj)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.make_segment()
        bpy.ops.object.mode_set(mode='OBJECT')
        sp = su.splines[0]
        # make_segment chains the selected splines in an arbitrary
        # (geometry-dependent) order, so only the resulting GRID
        # STRUCTURE is trustworthy. Rewrite every control point into the
        # intended layout (storage is u-fastest: flat = v*pu + u).
        pu, pv = sp.point_count_u, sp.point_count_v
        if (pu, pv) == (nu, nv):
            ordered = grid.transpose(1, 0, 2)   # S[v][u] = grid[u][v]
            cu, cv = cyclic_u, cyclic_v
        else:                                   # (pu, pv) == (nv, nu)
            ordered = grid                      # S[v][u] = grid[v][u]
            cu, cv = cyclic_v, cyclic_u
        flat = np.concatenate(
            [ordered.reshape(-1, 3), np.ones((pu * pv, 1))],
            axis=1).ravel()
        sp.points.foreach_set('co', flat)
        sp.use_cyclic_u = cu
        sp.use_cyclic_v = cv
        sp.use_endpoint_u = not cu
        sp.use_endpoint_v = not cv
        sp.order_u = min(order, pu)
        sp.order_v = min(order, pv)
        sp.resolution_u = 6
        sp.resolution_v = 6
        su.update_tag()
        obj.location = context.scene.cursor.location
        return obj

    def _new_object(context, name, verts, faces, weld=0.0, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        if weld > 0:
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
            bm.to_mesh(me)
            bm.free()
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    def _extract_loop(obj, depsgraph):
        """Ordered closed polyline (world space) from a curve or mesh
        object. Returns ndarray (k,3) or raises ValueError."""
        ev = obj.evaluated_get(depsgraph)
        me = ev.to_mesh()
        try:
            if len(me.vertices) == 0:
                raise ValueError(f"{obj.name}: no geometry")
            adj = {}
            for e in me.edges:
                a, b = e.vertices
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            if not adj or any(len(v) != 2 for v in adj.values()):
                raise ValueError(
                    f"{obj.name}: must be a single closed loop "
                    f"(every point joining exactly 2 segments)")
            start = next(iter(adj))
            loop = [start]
            prev, cur = None, start
            while True:
                nxt = [v for v in adj[cur] if v != prev]
                if not nxt:
                    raise ValueError(f"{obj.name}: open curve")
                prev, cur = cur, nxt[0]
                if cur == start:
                    break
                loop.append(cur)
                if len(loop) > len(me.vertices):
                    raise ValueError(f"{obj.name}: not a single loop")
            mw = np.array(obj.matrix_world)
            pts = np.array([me.vertices[i].co[:] for i in loop])
            pts = pts @ mw[:3, :3].T + mw[:3, 3]
            return pts
        finally:
            ev.to_mesh_clear()

    class MESH_OT_parametric_minimal_add(bpy.types.Operator):
        """Add a classic parametric minimal surface"""
        bl_idname = "mesh.parametric_minimal_add"
        bl_label = "Classic Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            items=[(k, v[0], v[0]) for k, v in PARAMETRIC.items()],
            default='ENNEPER')
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh", "Dense polygon mesh"),
                   ('NURBS', "NURBS", "Compact NURBS surface patch "
                                      "(control grid = Resolution U x V)")],
            default='MESH')
        res_u: IntProperty(name="Resolution U", default=48, min=8, max=512)
        res_v: IntProperty(name="Resolution V", default=48, min=8, max=512)
        ctrl_u: IntProperty(
            name="Control Points U", default=24, min=6, max=128,
            description="NURBS control grid size in U")
        ctrl_v: IntProperty(
            name="Control Points V", default=24, min=6, max=128,
            description="NURBS control grid size in V")
        order: IntProperty(
            name="Order / Turns", default=1, min=1, max=8,
            description="Enneper order; helicoid half-turns; ignored otherwise")
        radius: FloatProperty(
            name="Domain Radius", default=1.2, min=0.2, max=4.0,
            description="Extent of the parameter domain")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            label = PARAMETRIC[self.surface][0]
            if self.output == 'NURBS':
                G, wrap_u, wrap_v = build_parametric_grid(
                    self.surface, self.ctrl_u, self.ctrl_v,
                    self.order, self.radius, self.scale)
                if wrap_u:          # drop duplicated periodic endpoint
                    G = G[:-1]
                if wrap_v:
                    G = G[:, :-1]
                _nurbs_grid_object(context, label, G,
                                   cyclic_u=wrap_u, cyclic_v=wrap_v)
            else:
                V, quads = build_parametric(self.surface, self.res_u,
                                            self.res_v, self.order,
                                            self.radius, self.scale)
                _new_object(context, label, V, quads,
                            weld=1e-5 * max(1.0, self.scale))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'surface')
            lay.prop(self, 'output')
            if self.output == 'NURBS':
                lay.prop(self, 'ctrl_u')
                lay.prop(self, 'ctrl_v')
            else:
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
            for k in ('order', 'radius', 'scale'):
                lay.prop(self, k)

    class MESH_OT_tpms_add(bpy.types.Operator):
        """Add a triply-periodic minimal surface (nodal approximation)"""
        bl_idname = "mesh.tpms_add"
        bl_label = "Periodic Minimal Surface (TPMS)"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            items=[(k, v[0], v[0]) for k, v in TPMS.items()],
            default='G')
        cells: IntProperty(
            name="Cells", default=1, min=1, max=4,
            description="Number of unit cells per axis")
        resolution: IntProperty(
            name="Resolution / Cell", default=28, min=8, max=80,
            description="Sample grid resolution per unit cell")
        cell_size: FloatProperty(
            name="Cell Size", default=2.0, min=0.1, max=100.0,
            description="Edge length of one unit cell in Blender units")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this thickness")

        def execute(self, context):
            verts, tris = build_tpms(self.surface, self.cells,
                                     self.resolution, self.cell_size)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty level set")
                return {'CANCELLED'}
            label = TPMS[self.surface][0]
            obj = _new_object(context, label, verts, tris)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('surface', 'cells', 'resolution', 'cell_size',
                      'thickness'):
                lay.prop(self, k)

    class OBJECT_OT_minimal_span(bpy.types.Operator):
        """Span a minimal surface across the selected curve (1 object:
        disk) or between two selected curves (2 objects: annulus)"""
        bl_idname = "object.minimal_span"
        bl_label = "Span Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        samples: IntProperty(
            name="Boundary Samples", default=128, min=16, max=512)
        rings: IntProperty(
            name="Interior Rings", default=24, min=3, max=128)
        iterations: IntProperty(
            name="Solver Iterations", default=40, min=1, max=200)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Emit a compact NURBS surface (control grid = the "
                        "solver grid) instead of a dense mesh. Where the "
                        "surface curls tightly (e.g. near a knot) the NURBS "
                        "may ripple; raise rings/samples or use mesh output")

        @classmethod
        def poll(cls, context):
            sel = [o for o in context.selected_objects
                   if o.type in ('CURVE', 'MESH')]
            return len(sel) in (1, 2)

        def execute(self, context):
            deps = context.evaluated_depsgraph_get()
            sel = [o for o in context.selected_objects
                   if o.type in ('CURVE', 'MESH')]
            try:
                loops = [_extract_loop(o, deps) for o in sel]
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            m = self.samples
            if len(loops) == 1:
                A = resample_loop(loops[0], m)
                V, quads, fixed = build_disk_grid(A, self.rings)
                rows = self.rings   # + center point row
            else:
                A = resample_loop(loops[0], m)
                B = align_loops(A, resample_loop(loops[1], m))
                V, quads, fixed = build_annulus_grid(A, B, self.rings)
                rows = self.rings + 1
            T = _quads_to_tris(quads)
            minimize_area(V, T, fixed, outer_iters=self.iterations)
            if self.output_nurbs:
                G = fair_grid_columns(V[:rows * m].reshape(rows, m, 3))
                G = fair_grid_2d(G)
                V[:rows * m] = G.reshape(-1, 3)
                relax_normal_flow(V, T, fixed)
                G = V[:rows * m].reshape(rows, m, 3)
                if len(loops) == 1:   # close the pole with the center point
                    cen = np.tile(V[-1], (1, m, 1))
                    G = np.concatenate([G, cen], axis=0)
                obj = _nurbs_grid_object(context, "MinimalSpan", G,
                                         cyclic_u=False, cyclic_v=True)
            else:
                obj = _new_object(context, "MinimalSpan", V, quads)
            obj.location = (0, 0, 0)   # loops are in world space
            self.report({'INFO'},
                        f"area = {mesh_area(V, T):.4f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('samples', 'rings', 'iterations', 'output_nurbs'):
                lay.prop(self, k)

    class MESH_OT_knot_span_add(bpy.types.Operator):
        """Minimal surface between a circle and a (p,q) torus knot
        (trefoil by default) -- the classic Plateau demonstration"""
        bl_idname = "mesh.minimal_knot_span_add"
        bl_label = "Minimal Surface: Circle to Torus Knot"
        bl_options = {'REGISTER', 'UNDO'}

        p: IntProperty(name="Knot p", default=2, min=1, max=8)
        q: IntProperty(name="Knot q", default=3, min=1, max=9)
        circle_radius: FloatProperty(
            name="Circle Radius", default=4.5, min=1.0, max=20.0)
        knot_scale: FloatProperty(
            name="Knot Scale", default=1.0, min=0.1, max=5.0)
        samples: IntProperty(
            name="Boundary Samples", default=96, min=32, max=512)
        rings: IntProperty(name="Interior Rings", default=16, min=4, max=128)
        iterations: IntProperty(
            name="Solver Iterations", default=30, min=1, max=200)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Emit a compact NURBS surface (control grid = the "
                        "solver grid) instead of a dense mesh. Where the "
                        "surface curls tightly (e.g. near a knot) the NURBS "
                        "may ripple; raise rings/samples or use mesh output")

        def execute(self, context):
            m = self.samples
            knot = torus_knot(self.p, self.q, m, scale=self.knot_scale)
            t = np.linspace(0, TAU, m, endpoint=False)
            # circle wound p times so the ruling lines up with the knot
            circ = np.stack([self.circle_radius * np.cos(self.p * t),
                             self.circle_radius * np.sin(self.p * t),
                             np.zeros(m)], axis=1)
            V, quads, fixed = build_annulus_grid(knot, circ, self.rings)
            T = _quads_to_tris(quads)
            minimize_area(V, T, fixed, outer_iters=self.iterations)
            name = f"Knot({self.p},{self.q})Span"
            if self.output_nurbs:
                G = fair_grid_columns(V.reshape(self.rings + 1, m, 3))
                G = fair_grid_2d(G)
                V = G.reshape(-1, 3).copy()
                relax_normal_flow(V, T, fixed)
                G = V.reshape(self.rings + 1, m, 3)
                obj = _nurbs_grid_object(context, name, G,
                                         cyclic_u=False, cyclic_v=True)
            else:
                obj = _new_object(context, name, V, quads)
            self.report({'INFO'}, f"area = {mesh_area(V, T):.4f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('p', 'q', 'circle_radius', 'knot_scale', 'samples',
                      'rings', 'iterations', 'output_nurbs'):
                lay.prop(self, k)

    class VIEW3D_PT_minimal_surfaces(bpy.types.Panel):
        bl_label = "Minimal Surfaces"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            col = lay.column(align=True)
            col.operator("mesh.parametric_minimal_add", icon='SURFACE_NSPHERE')
            col.operator("mesh.tpms_add", icon='MESH_ICOSPHERE')
            col.separator()
            col.operator("mesh.minimal_knot_span_add", icon='MESH_TORUS')
            col.label(text="Select 1-2 closed curves, then:")
            col.operator("object.minimal_span", icon='OUTLINER_OB_SURFACE')

    class VIEW3D_MT_minimal_add(bpy.types.Menu):
        bl_idname = "VIEW3D_MT_minimal_add"
        bl_label = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            lay.operator("mesh.parametric_minimal_add")
            lay.operator("mesh.tpms_add")
            lay.operator("mesh.minimal_knot_span_add")
            lay.operator("object.minimal_span")

    def _menu_func(self, context):
        self.layout.menu("VIEW3D_MT_minimal_add", icon='SURFACE_DATA')

    _classes = (MESH_OT_parametric_minimal_add, MESH_OT_tpms_add,
                OBJECT_OT_minimal_span, MESH_OT_knot_span_add,
                VIEW3D_PT_minimal_surfaces, VIEW3D_MT_minimal_add)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # standalone smoke tests of the numeric core
        for kind in PARAMETRIC:
            V, Q = build_parametric(kind, 48, 48, 1, 1.2, 1.0)
            print(f"parametric {kind:10s}: {len(V):6d} verts {len(Q):6d} quads")
        for kind in TPMS:
            V, T = build_tpms(kind, 1, 20, 2.0)
            print(f"tpms {kind:10s}: {len(V):6d} verts {len(T):6d} tris")
        # flat-disk validation: minimal surface on a planar circle is flat
        t = np.linspace(0, TAU, 96, endpoint=False)
        circle = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        V, quads, fixed = build_disk_grid(circle, 12)
        V[~fixed] += np.random.default_rng(1).normal(0, 0.1, V[~fixed].shape)
        T = _quads_to_tris(quads)
        minimize_area(V, T, fixed)
        print("flat disk: max|z| =", float(np.max(np.abs(V[:, 2]))),
              " area =", mesh_area(V, T), " (pi =", math.pi, ")")
        # catenoid validation: two rings radius 1 at z = +/-0.4
        z0 = 0.4
        ringA = np.stack([np.cos(t), np.sin(t), np.full_like(t, z0)], axis=1)
        ringB = np.stack([np.cos(t), np.sin(t), np.full_like(t, -z0)], axis=1)
        V, quads, fixed = build_annulus_grid(ringA, ringB, 20)
        T = _quads_to_tris(quads)
        minimize_area(V, T, fixed)
        waist = np.min(np.linalg.norm(V[:, :2], axis=1))
        print("catenoid: waist =", float(waist), "(analytic ~0.9098)")
