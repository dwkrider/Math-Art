# Volume-constrained surface evolution: bodies, pressures, projection.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# The missing capability class this module creates: gradient descent on
# surface area (or any vertex-gradient energy) that conserves the volume
# of every enclosed body exactly, so soap-bubble clusters can be
# *relaxed* into genuine equilibria -- Plateau's 120-degree borders
# emerge from the minimization instead of being drawn in -- and constant
# mean curvature / capillary shapes become reachable.
#
# Representation (multi-material, after Da-Batty-Grinspun): every
# triangle carries an ordered region-pair label `(front, back)` --
# `front` is the region its oriented normal points into, `back` the
# region behind it.  Region 0 is the unconstrained ambient; bodies are
# regions 1..R.  A face is outward-oriented for its `back` region, so
# by the divergence theorem
#
#     V_b = sum_f s_{f,b} det(x0, x1, x2) / 6,
#     s = +1 where back == b, -1 where front == b, 0 elsewhere.
#
# Films are label classes; Plateau borders are simply the non-manifold
# edges where three films meet -- no special casing anywhere.
#
# Constrained descent (after Brakke's Surface Evolver, fixvol.c):
#
#   * project_velocity -- solve the small k x k Gram system
#     [grad V_i . M grad V_j] lam = [grad V_i . v] and subtract
#     M * sum_i lam_i grad V_i, so the motion is tangent to every
#     volume-constraint level set (M is the per-vertex mobility metric;
#     with it, the fixed point satisfies the true Karush-Kuhn-Tucker
#     condition grad E = sum p_i grad V_i, not an M-skewed one).
#
#   * restore_volumes -- after a step, a damped Newton loop on the
#     constraint values: solve the same Gram system against
#     (target - V) and move along the volume gradients, halving on
#     regression, <= 10 rounds (Evolver's volume_restore/project_all).
#
#   * The Lagrange multipliers ARE the body pressures: at equilibrium
#     grad A = sum_i p_i grad V_i, and for a sphere p = dA/dV = 2/r --
#     the Young-Laplace law, which doubles as an independent check that
#     the constraint algebra is right.
#
# `evolve` composes this with the shared machinery: mollified-cotan
# area gradients (solver/cotan), the Evolver parabola line search
# (solver/descent) evaluated at volume-restored trial positions, and
# label-aware grooming (solver/groom) with triple-line vertices pinned.
#
# References:
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- the Gram-matrix Lagrange scheme and the damped Newton
#       volume restore (fixvol.c, iterate.c).
#   F. Da, C. Batty, E. Grinspun, "Multimaterial Mesh-Based Surface
#       Tracking", ACM Trans. Graph. 33(4) (2014) -- per-face ordered
#       region-pair labels for multi-body clusters.
#   J. E. Taylor, "The structure of singularities in soap-bubble-like
#       and soap-film-like minimal surfaces", Annals of Mathematics
#       103(3) (1976) -- proof of Plateau's laws (films meet in threes
#       at 120 degrees).
#   M. Hutchings, F. Morgan, M. Ritore, A. Ros, "Proof of the double
#       bubble conjecture", Annals of Mathematics 155 (2002).
#   J. Plateau, "Statique experimentale et theorique des liquides
#       soumis aux seules forces moleculaires" (1873).

import numpy as np

try:
    from . import cotan as _cotan
    from . import descent as _descent
    from . import groom as _groom
    from . import walls as _walls
except ImportError:                      # flat (path-based) headless import
    import cotan as _cotan               # type: ignore
    import descent as _descent           # type: ignore
    import groom as _groom               # type: ignore
    import walls as _walls               # type: ignore


# --------------------------------------------------------------------------
# volumes and their gradients
# --------------------------------------------------------------------------

def face_det6(V, T):
    """det(x0, x1, x2)/6 per face: the divergence-theorem volume element."""
    return np.einsum('ij,ij->i', V[T[:, 0]],
                     np.cross(V[T[:, 1]], V[T[:, 2]])) / 6.0


def region_volumes(V, T, labels, nregions=None):
    """Volumes of regions 1..nregions from the label-signed divergence
    sum.  labels: (ntri, 2) ints, (front, back); region 0 = ambient."""
    if nregions is None:
        nregions = int(labels.max())
    det6 = face_det6(V, T)
    vols = np.empty(nregions)
    for r in range(1, nregions + 1):
        s = (labels[:, 1] == r).astype(float) - (labels[:, 0] == r)
        vols[r - 1] = float(s @ det6)
    return vols


def volume_gradients(V, T, labels, nregions=None, free=None):
    """d V_b / d x_v for every body: (nregions, nverts, 3).  Per face
    (x0,x1,x2): d(det/6)/dx0 = (x1 x x2)/6 etc., scattered with the
    label sign.  Rows of non-free vertices are zeroed so both the
    projection and the restore leave them untouched."""
    if nregions is None:
        nregions = int(labels.max())
    n = len(V)
    C = (np.cross(V[T[:, 1]], V[T[:, 2]]) / 6.0,
         np.cross(V[T[:, 2]], V[T[:, 0]]) / 6.0,
         np.cross(V[T[:, 0]], V[T[:, 1]]) / 6.0)
    G = np.zeros((nregions, n, 3))
    for r in range(1, nregions + 1):
        s = (labels[:, 1] == r).astype(float) - (labels[:, 0] == r)
        if not np.any(s):
            continue
        for k in range(3):
            np.add.at(G[r - 1], T[:, k], s[:, None] * C[k])
    if free is not None:
        G[:, ~free] = 0.0
    return G


def gram_matrix(grads, weights=None):
    """G[i, j] = sum_v w_v grad V_i(v) . grad V_j(v)  (k x k dense)."""
    if weights is None:
        return np.einsum('inj,mnj->im', grads, grads)
    return np.einsum('inj,n,mnj->im', grads, weights, grads)


def _solve_gram(A, rhs):
    """Solve the k x k Gram system; a zero diagonal entry (empty or
    fully-pinned body) gets Evolver's unit-diagonal treatment so the
    system stays invertible and that body simply receives no force."""
    A = np.array(A, float)
    rhs = np.array(rhs, float)
    dead = np.abs(np.diag(A)) < 1e-300
    if dead.any():
        A[dead, :] = 0.0
        A[:, dead] = 0.0
        A[dead, dead] = 1.0
        rhs[dead] = 0.0
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, rhs, rcond=None)[0]


# --------------------------------------------------------------------------
# projection and restore
# --------------------------------------------------------------------------

def project_velocity(vel, grads, weights=None):
    """Make `vel` tangent to every volume constraint: subtract
    (w *) sum_i lam_i grad V_i with lam from the (weighted) Gram solve.
    Returns (vel_tangent, lam).  With weights = the mobility metric M
    and vel = -M grad E, the fixed point vel_tangent = 0 is exactly the
    KKT condition grad E = sum_i (-lam_i) grad V_i, i.e. the body
    pressures are p = -lam."""
    if len(grads) == 0:                  # no bodies (open film): no-op
        return vel, np.zeros(0)
    A = gram_matrix(grads, weights)
    rhs = np.einsum('inj,nj->i', grads, vel)
    lam = _solve_gram(A, rhs)
    corr = np.einsum('i,inj->nj', lam, grads)
    if weights is not None:
        corr *= weights[:, None]
    return vel - corr, lam


def body_pressures(area_grad, grads, weights=None):
    """Least-squares pressures p solving [grad V_i . w grad V_j] p =
    [grad V_i . w grad E]: at equilibrium grad E = sum p_i grad V_i
    exactly, and p obeys Young-Laplace (2/r for a unit-tension sphere)."""
    if len(grads) == 0:                  # no bodies (open film): no-op
        return np.zeros(0)
    A = gram_matrix(grads, weights)
    if weights is None:
        rhs = np.einsum('inj,nj->i', grads, area_grad)
    else:
        rhs = np.einsum('inj,n,nj->i', grads, weights, area_grad)
    return _solve_gram(A, rhs)


def restore_volumes(V, T, labels, targets, free=None, max_rounds=10,
                    tol=1e-12, project=None, tangent=None):
    """Damped Newton on the constraint values, in place (Evolver's
    volume_restore wrapped in project_all's damping): solve
    G mu = (target - V_b), move every vertex by sum_i mu_i grad V_i,
    halve the move if the worst relative deficit grew, recompute the
    gradients each round, give up after max_rounds.

    Wall-constraint hooks (both default None, changing nothing):
    `tangent(grads, V) -> grads` projects the volume gradients onto the
    wall tangent spaces so the Newton move slides along the walls, and
    `project(V)` (in place) re-projects each trial position onto the
    walls before its volumes are measured -- so the accepted state
    satisfies the walls at projection tolerance AND the volumes at
    `tol` (the O(step^2) volume change of the wall re-projection is
    absorbed by the next Newton round).

    Returns (rel_before, rel_after, rounds): worst |V - target|/|target|
    entering and leaving, and the Newton rounds spent."""
    targets = np.asarray(targets, float)
    nb = len(targets)
    if nb == 0:                          # no bodies (open film): no-op
        return 0.0, 0.0, 0
    scale = np.maximum(np.abs(targets), 1e-300)
    rel_before = None
    rounds = 0
    for rounds in range(max_rounds):
        vols = region_volumes(V, T, labels, nb)
        deficit = targets - vols
        rel = float(np.max(np.abs(deficit) / scale))
        if rel_before is None:
            rel_before = rel
        if rel < tol:
            return rel_before, rel, rounds
        grads = volume_gradients(V, T, labels, nb, free)
        if tangent is not None:
            grads = tangent(grads, V)
        A = gram_matrix(grads)
        mu = _solve_gram(A, deficit)
        step = np.einsum('i,inj->nj', mu, grads)
        s = 1.0
        accepted = False
        for _ in range(8):
            V_n = V + s * step
            if project is not None:
                project(V_n)
            vols_n = region_volumes(V_n, T, labels, nb)
            rel_n = float(np.max(np.abs(targets - vols_n) / scale))
            if rel_n < rel:
                accepted = True
                break
            s *= 0.5
        if not accepted:
            return rel_before, rel, rounds     # no direction improves
        V[:] = V_n
    vols = region_volumes(V, T, labels, nb)
    rel = float(np.max(np.abs(targets - vols) / scale))
    return rel_before, rel, max_rounds


# --------------------------------------------------------------------------
# area, its gradient, mobility
# --------------------------------------------------------------------------

def mesh_area(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return 0.5 * float(np.sum(np.linalg.norm(n, axis=1)))


def area_gradient(V, T, cotan_mode="mollify"):
    """grad(total area) at every vertex, assembled from the cotangent
    identity grad_i A = sum_j w_ij (x_i - x_j) -- exactly the classical
    cross-product form, but built from solver.cotan so degenerate
    triangles stay finite under intrinsic mollification."""
    E, W = _cotan.edge_cotan_weights(V, T, mode=cotan_mode)
    g = np.zeros_like(V)
    contrib = W[:, None] * (V[E[:, 0]] - V[E[:, 1]])
    np.add.at(g, E[:, 0], contrib)
    np.add.at(g, E[:, 1], -contrib)
    return g


def vertex_star_areas(V, T):
    """Barycentric vertex areas (star/3): the Evolver mobility that makes
    velocity = mean curvature and the stable step resolution-free."""
    fa = 0.5 * np.linalg.norm(
        np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    astar = np.zeros(len(V))
    for k in range(3):
        np.add.at(astar, T[:, k], fa / 3.0)
    return astar


# --------------------------------------------------------------------------
# cluster combinatorics helpers
# --------------------------------------------------------------------------

def face_groups(labels):
    """Map ordered label pairs to consecutive film ids (ntri,) -- the
    groups within which grooming flips are allowed."""
    _, inv = np.unique(labels, axis=0, return_inverse=True)
    return inv


def nonmanifold_vertices(T, n):
    """Mask of vertices on any edge not shared by exactly two faces --
    the Plateau borders (3 films) and any open boundary.  These are
    pinned for tangential smoothing (a film-averaged vertex normal is
    meaningless on a triple line)."""
    de = np.sort(np.concatenate(
        [T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(de, axis=0, return_counts=True)
    bad = uniq[counts != 2]
    mask = np.zeros(n, dtype=bool)
    mask[bad.ravel()] = True
    return mask


def triple_line_angles(V, T):
    """Dihedral angles (degrees) between the films around every
    non-manifold 3-film edge: per edge, the three sector angles between
    the in-film directions perpendicular to the edge (they sum to 360).
    Plateau/Taylor: at equilibrium each is 120.  Returns a flat array,
    3 angles per triple edge (empty if the mesh is manifold)."""
    de = np.concatenate([T[:, [1, 2]], T[:, [2, 0]], T[:, [0, 1]]])
    opp = np.concatenate([T[:, 0], T[:, 1], T[:, 2]])
    key = np.sort(de, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True,
                                  return_counts=True)
    order = np.argsort(inv, kind='stable')
    starts = np.zeros(len(uniq), dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    angles = []
    for e in np.nonzero(counts == 3)[0]:
        a, b = uniq[e]
        ed = V[b] - V[a]
        ed = ed / max(np.linalg.norm(ed), 1e-300)
        mid = 0.5 * (V[a] + V[b])
        # orthonormal frame perpendicular to the edge
        f1 = np.cross(ed, [1.0, 0.0, 0.0])
        if np.linalg.norm(f1) < 1e-6:
            f1 = np.cross(ed, [0.0, 1.0, 0.0])
        f1 /= np.linalg.norm(f1)
        f2 = np.cross(ed, f1)
        th = []
        for k in range(3):
            # in-film direction perpendicular to the edge, from the
            # face plane (normal x edge), signed toward the face
            j = order[starts[e] + k]
            c = opp[j]
            f = j % len(T)
            tri = T[f]
            nf = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
            w = np.cross(nf, ed)
            if (w @ (V[c] - mid)) < 0.0:
                w = -w
            th.append(np.arctan2(w @ f2, w @ f1))
        th = np.sort(th)
        secs = [th[1] - th[0], th[2] - th[1],
                2.0 * np.pi - (th[2] - th[0])]
        angles.extend(np.degrees(secs))
    return np.asarray(angles)


# --------------------------------------------------------------------------
# the evolver
# --------------------------------------------------------------------------

def evolve(V, T, labels, targets=None, iters=200, fixed=None,
           mobility="star", cotan_mode="mollify", cg=True,
           groom_every=0, groom_smooth=0.25,
           area_tol=1e-10, vol_tol=1e-12, s_init_frac=0.2,
           walls=None, wall_tol=1e-12, ext_energy=None, ext_grad=None,
           groom_hook=None):
    """Minimize film area at fixed body volumes.  V is modified in
    place; T is modified in place only when grooming flips edges.

    V (n,3) float, T (m,3) int, labels (m,2) int region pairs (front,
    back; 0 = ambient).  targets: per-body volumes (default: the seed's
    own).  fixed: bool mask of pinned vertices.  mobility "star" divides
    the force by the barycentric vertex area (motion by mean curvature);
    "none" uses the raw gradient.  cg=True runs Polak-Ribiere conjugate
    gradient over the projected velocities (Evolver uses CG together
    with the optimizing-scale line search, never with a fixed step) --
    plain projected descent stalls on the soft long-wavelength modes,
    whose decay the stiff short-wavelength stability limit makes
    excruciating; the safeties are Evolver's: gamma outside [0, 10]
    resets to 0, and any rejected step or groom cycle restarts the
    direction.  groom_every > 0 runs label-aware Delaunay flips +
    tangential smoothing (triple-line and fixed vertices pinned) before
    iterations g, 2g, ..., re-restoring volumes afterwards.

    Wall constraints (solver/walls): `walls` is a list of (wall, mask)
    pairs -- boolean vertex masks flagged to level-set walls f(x) = 0.
    Each step the velocity is wall-tangent-projected, the volume
    gradients entering the Gram solve are tangent-projected against the
    two-sided walls (so the Lagrange correction slides along them and
    the fixed point is the true constrained KKT state), and positions
    are Newton-projected back onto the walls both before the volume
    restore and inside its damping loop -- walls first, volumes second,
    because the restore respects the walls by construction while wall
    projection knows nothing of volumes.  One-sided walls (f >= 0)
    project positions only when violating and constrain only on-wall,
    inward-pointing velocities; their volume gradients are left free
    (the position projection catches the violation and the restore loop
    absorbs it), which is the survey's "simple version" by design.
    History entries then also carry wall_resid (post-step max |f|) and
    vt_normal_max (tangency residual of the projected velocity over the
    two-sided rows).

    `groom_hook(V, T)` (default None) runs right after each groom
    cycle, before the wall projection and volume restore -- the hook
    for caller-side mesh maintenance the pinned-vertex groom cannot do
    (e.g. redistributing free-boundary vertices along their boundary
    curve).

    Extra energy terms: `ext_energy(V) -> float` is added to the film
    area in the line search (e.g. the wetting energy
    -cos(theta) * wetted area of a sessile drop) and `ext_grad(V) ->
    (n,3)` to the area gradient.  With an ext term the monotone
    quantity is E = area + ext (recorded per iteration as "E" with
    "E_rise"); without one it is the area itself, exactly as before.

    Every accepted solver step is monotone in the energy BY
    CONSTRUCTION: the line search evaluates it at wall-projected,
    volume-restored trial positions, so a rise in the history (outside
    groom iterations) means a bug.

    Returns a dict: iters_run, area, volumes, pressures, targets,
    grooms_run, history (per-iteration area / step / drift / rise)."""
    V = np.asarray(V)
    T = np.asarray(T)
    labels = np.asarray(labels)
    nb = int(labels.max())
    free = np.ones(len(V), dtype=bool)
    if fixed is not None:
        free &= ~np.asarray(fixed, dtype=bool)
    if targets is None:
        targets = region_volumes(V, T, labels, nb)
    targets = np.asarray(targets, float)

    groups = face_groups(labels)
    smooth_pin = ~free | nonmanifold_vertices(T, len(V))

    wall_list = None
    wall_ts = None                       # two-sided subset
    if walls:
        wall_list = [(w, None if m is None else np.asarray(m, bool))
                     for w, m in walls]
        wall_ts = [(w, m) for w, m in wall_list if not w.one_sided]

    def _wproject(Varr):
        _walls.project_all(Varr, wall_list, tol=wall_tol)

    def _gtangent(G, Varr):
        # volume gradients slide along the two-sided walls only; the
        # one-sided walls are enforced by position projection instead
        for i in range(len(G)):
            _walls.tangent_all(G[i], Varr, wall_ts)
        return G

    def _restore(Varr):
        if wall_list is None:
            return restore_volumes(Varr, T, labels, targets, free=free,
                                   tol=vol_tol)
        return restore_volumes(Varr, T, labels, targets, free=free,
                               tol=vol_tol, project=_wproject,
                               tangent=_gtangent)

    def _E_of(Varr, A):
        return A if ext_energy is None else A + float(ext_energy(Varr))

    if wall_list is not None:
        _wproject(V)                     # walls first, volumes second
    _restore(V)                          # start ON the constraint manifold
    A_prev = mesh_area(V, T)
    E_prev = _E_of(V, A_prev)
    history = []
    grooms_run = 0
    s_prev = None
    lam = np.zeros(nb)
    flat_streak = 0
    d_prev = None                        # CG state
    v_prev = None
    v_prev_ip = 1.0
    it = 0
    for it in range(1, iters + 1):
        groomed = False
        if groom_every and (it - 1) and (it - 1) % groom_every == 0:
            _groom.groom(V, T, fixed=smooth_pin, smooth_lam=groom_smooth,
                         tri_groups=groups)
            if groom_hook is not None:
                # caller-supplied groom-time maintenance (e.g. a film
                # generator redistributing its free-boundary vertices
                # ALONG the boundary curve, which the pinned-vertex
                # groom deliberately leaves alone); runs before the
                # wall projection + restore so its motion is cleaned
                # up exactly like the groom's own
                groom_hook(V, T)
            if wall_list is not None:
                _wproject(V)
            _restore(V)
            grooms_run += 1
            groomed = True
            d_prev = None                # full CG restart on mesh change
            A_prev = mesh_area(V, T)
            E_prev = _E_of(V, A_prev)
        g = area_gradient(V, T, cotan_mode=cotan_mode)
        if ext_grad is not None:
            g = g + ext_grad(V)
        if mobility == "star":
            M = 3.0 / np.maximum(vertex_star_areas(V, T), 1e-300)
        elif mobility == "none":
            M = np.ones(len(V))
        else:
            raise ValueError(f"unknown mobility {mobility!r}")
        M[~free] = 0.0
        v = -M[:, None] * g
        if wall_list is not None:
            _walls.tangent_all(v, V, wall_list)
        grads = volume_gradients(V, T, labels, nb, free=free)
        if wall_list is not None:
            _gtangent(grads, V)
        vt, lam = project_velocity(v, grads, weights=M)
        vt_normal_max = (0.0 if wall_list is None
                         else _walls.normal_component_max(vt, V, wall_ts))
        vmax = float(np.max(np.linalg.norm(vt, axis=1)))
        if vmax < 1e-300:
            break
        # Polak-Ribiere direction over the projected velocities, in the
        # energy inner product <a, b> = sum a.b/M (velocity . force)
        Minv = np.where(M > 0.0, 1.0 / np.maximum(M, 1e-300), 0.0)
        vt_ip = float(np.einsum('nj,n,nj->', vt, Minv, vt))
        d = vt
        gamma = 0.0
        if cg and d_prev is not None and not groomed:
            gamma = (vt_ip - float(np.einsum('nj,n,nj->', vt, Minv,
                                             v_prev))) / v_prev_ip
            if not (0.0 <= gamma <= 10.0):
                gamma = 0.0              # Evolver's safeties
            if gamma > 0.0:
                d = vt + gamma * d_prev
                # the previous direction was tangent to the PREVIOUS
                # constraint gradients; re-project onto the current ones
                # (walls first: their normals rotated with the surface)
                if wall_list is not None:
                    _walls.tangent_all(d, V, wall_list)
                d, _ = project_velocity(d, grads, weights=M)
        Lmean = float(np.mean(np.linalg.norm(
            V[T[:, 1]] - V[T[:, 0]], axis=1)))
        dmax = float(np.max(np.linalg.norm(d, axis=1)))
        s_max = 4.0 * Lmean / max(dmax, 1e-300)
        if s_prev is None or not (0.0 < s_prev):
            s_prev = s_init_frac * Lmean / vmax
        s0 = min(s_prev, s_max)

        def energy(x):
            Vc = np.array(x, float)
            if wall_list is not None:
                _wproject(Vc)
            _restore(Vc)
            return _E_of(Vc, mesh_area(Vc, T))

        x1, s, E1, nev = _descent.parabola_line_search(
            energy, V, d, s0, s_max=s_max)
        if s == 0.0 and gamma > 0.0:
            # CG direction rejected: restart with pure projected descent
            d = vt
            gamma = 0.0
            s_max = 4.0 * Lmean / vmax
            x1, s, E1, nev = _descent.parabola_line_search(
                energy, V, d, min(s0, s_max), s_max=s_max)
        if s == 0.0:
            break                        # no downhill scale: converged
        v_prev = vt
        v_prev_ip = vt_ip
        d_prev = d
        V[:] = x1
        if wall_list is not None:
            _wproject(V)
        drift_pre, drift_post, _rounds = _restore(V)
        A = mesh_area(V, T)
        E = _E_of(V, A)
        entry = {
            "it": it, "area": A, "s": s, "n_evals": nev,
            "rise": (A - A_prev) / max(A_prev, 1e-300),
            "drift_pre": drift_pre, "drift_post": drift_post,
            "groomed": groomed,
        }
        if wall_list is not None:
            entry["wall_resid"] = _walls.wall_residual(V, wall_list)
            entry["vt_normal_max"] = vt_normal_max
        if ext_energy is not None:
            entry["E"] = E
            entry["E_rise"] = (E - E_prev) / max(abs(E_prev), 1e-300)
        history.append(entry)
        if abs(E - E_prev) < area_tol * abs(E_prev):
            flat_streak += 1
            if flat_streak >= 3:
                A_prev = A
                E_prev = E
                break
        else:
            flat_streak = 0
        A_prev = A
        E_prev = E
        s_prev = s

    g = area_gradient(V, T, cotan_mode=cotan_mode)
    if ext_grad is not None:
        g = g + ext_grad(V)
    grads = volume_gradients(V, T, labels, nb, free=free)
    if wall_list is not None:
        _gtangent(grads, V)
    if mobility == "star":
        M = 3.0 / np.maximum(vertex_star_areas(V, T), 1e-300)
        M[~free] = 0.0
    else:
        M = None
    press = body_pressures(g, grads, weights=M)
    return {
        "iters_run": it, "area": mesh_area(V, T),
        "volumes": region_volumes(V, T, labels, nb),
        "targets": targets, "pressures": press,
        "grooms_run": grooms_run, "history": history,
    }


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _icosphere_mesh(subdiv):
    try:
        from ..surfaces.primitives import icosphere
    except ImportError:
        from surfaces.primitives import icosphere    # type: ignore
    SV, SF = icosphere(subdiv, 'per_level')
    return np.asarray(SV, float), np.asarray(SF, dtype=np.int64)


def _two_cube_mesh():
    """Two unit cubes sharing the x = 1 face: bodies 1 and 2, welded,
    with the shared wall labeled (2, 1).  A minimal genuine multi-body
    labeled mesh (non-manifold edges around the shared square)."""
    xs, ys, zs = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0],
                             indexing='ij')
    V = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)

    def idx(i, j, k):
        return i * 4 + j * 2 + k

    quads = []   # (4 verts CCW seen from OUTSIDE the named body, front, back)

    def add_quad(a, b, c, d, front, back):
        quads.append(((a, b, c, d), front, back))

    for i0, body in ((0, 1), (1, 2)):
        i1 = i0 + 1
        # -y / +y walls
        add_quad(idx(i0, 0, 0), idx(i1, 0, 0), idx(i1, 0, 1),
                 idx(i0, 0, 1), 0, body)
        add_quad(idx(i0, 1, 0), idx(i0, 1, 1), idx(i1, 1, 1),
                 idx(i1, 1, 0), 0, body)
        # -z / +z walls
        add_quad(idx(i0, 0, 0), idx(i0, 1, 0), idx(i1, 1, 0),
                 idx(i1, 0, 0), 0, body)
        add_quad(idx(i0, 0, 1), idx(i1, 0, 1), idx(i1, 1, 1),
                 idx(i0, 1, 1), 0, body)
    # outer x walls
    add_quad(idx(0, 0, 0), idx(0, 0, 1), idx(0, 1, 1), idx(0, 1, 0), 0, 1)
    add_quad(idx(2, 0, 0), idx(2, 1, 0), idx(2, 1, 1), idx(2, 0, 1), 0, 2)
    # shared wall at x=1: normal +x, outward for body 1, into body 2
    add_quad(idx(1, 0, 0), idx(1, 1, 0), idx(1, 1, 1), idx(1, 0, 1), 2, 1)

    T, labels = [], []
    for (a, b, c, d), front, back in quads:
        T.append((a, b, c))
        T.append((a, c, d))
        labels.append((front, back))
        labels.append((front, back))
    return V, np.asarray(T, dtype=np.int64), np.asarray(labels,
                                                        dtype=np.int64)


def _selftest():
    rng = np.random.default_rng(11)
    ok = True

    # --- volumes: two welded cubes must measure exactly 1 each --------
    V, T, labels = _two_cube_mesh()
    vols = region_volumes(V, T, labels)
    good = np.allclose(vols, [1.0, 1.0], atol=1e-14)
    ok &= good
    print(f"volume: two-cube volumes {vols} {'OK' if good else 'FAIL'}")

    # --- gradients: directional finite-difference check ---------------
    grads = volume_gradients(V, T, labels)
    d = rng.normal(size=V.shape)
    h = 1e-6
    fd = (region_volumes(V + h * d, T, labels)
          - region_volumes(V - h * d, T, labels)) / (2 * h)
    an = np.einsum('inj,nj->i', grads, d)
    err = float(np.max(np.abs(fd - an) / np.maximum(np.abs(fd), 1e-30)))
    good = err < 1e-8
    ok &= good
    print(f"volume: gradient vs central difference rel err {err:.2e} "
          f"{'OK' if good else 'FAIL'}")

    # --- projection: tangency and weighted-metric tangency ------------
    vel = rng.normal(size=V.shape)
    vt, lam = project_velocity(vel, grads)
    dV = np.einsum('inj,nj->i', grads, vt)
    good = float(np.max(np.abs(dV))) < 1e-10
    ok &= good
    print(f"volume: projected velocity tangent (|dV/dt| max "
          f"{np.max(np.abs(dV)):.2e}) {'OK' if good else 'FAIL'}")
    w = rng.uniform(0.5, 2.0, len(V))
    vt2, _ = project_velocity(vel, grads, weights=w)
    dV2 = np.einsum('inj,nj->i', grads, vt2)
    good = float(np.max(np.abs(dV2))) < 1e-10
    ok &= good
    print(f"volume: weighted projection tangent (max {np.max(np.abs(dV2)):.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # --- restore: perturbed cubes come back to volume 1 exactly -------
    Vp = V + rng.normal(0.0, 0.03, V.shape)
    b4, aft, rounds = restore_volumes(Vp, T, labels, [1.0, 1.0])
    good = aft < 1e-12 and b4 > 1e-3
    ok &= good
    print(f"volume: restore {b4:.2e} -> {aft:.2e} in {rounds} rounds "
          f"{'OK' if good else 'FAIL'}")

    # --- area gradient matches finite differences ---------------------
    SV, SF = _icosphere_mesh(1)
    g = area_gradient(SV, SF)
    d = rng.normal(size=SV.shape)
    fd = (mesh_area(SV + h * d, SF) - mesh_area(SV - h * d, SF)) / (2 * h)
    an = float(np.sum(g * d))
    err = abs(fd - an) / max(abs(fd), 1e-30)
    good = err < 1e-7
    ok &= good
    print(f"volume: area gradient vs FD rel err {err:.2e} "
          f"{'OK' if good else 'FAIL'}")

    # --- triple-line angles: the two-cube wall corners are 90/90/180 --
    ang = triple_line_angles(V, T)
    good = (len(ang) > 0
            and np.allclose(np.sort(ang.reshape(-1, 3), axis=1),
                            [90.0, 90.0, 180.0], atol=1e-9))
    ok &= good
    print(f"volume: two-cube triple-edge sectors 90/90/180 "
          f"{'OK' if good else 'FAIL'}")

    # --- the headline check: a squashed sphere relaxes to a sphere ----
    # and its Lagrange multiplier reproduces Young-Laplace p = 2/r.
    SV, SF = _icosphere_mesh(2)              # 162 vertices
    Vb = SV * np.array([1.25, 0.8, 1.0])
    lab = np.zeros((len(SF), 2), dtype=np.int64)
    lab[:, 1] = 1
    target = 4.0 * np.pi / 3.0
    info = evolve(Vb, SF, lab, targets=[target], iters=120)
    r_star = (3.0 * target / (4.0 * np.pi)) ** (1.0 / 3.0)      # = 1
    A_star = (36.0 * np.pi * target ** 2) ** (1.0 / 3.0)
    c = Vb.mean(axis=0)
    rad = np.linalg.norm(Vb - c, axis=1)
    rad_cv = float(np.std(rad) / np.mean(rad))
    area_exc = (info["area"] - A_star) / A_star
    p_err = abs(info["pressures"][0] - 2.0 / r_star) / (2.0 / r_star)
    drift = max(hh["drift_post"] for hh in info["history"])
    rise = max(hh["rise"] for hh in info["history"]
               if not hh["groomed"])
    # NOTE the discrete area minimizer at fixed volume on icosahedral
    # combinatorics is NOT an equal-radius vertex set (valence-5 and
    # valence-6 vertices settle at slightly different radii), so the
    # radius spread converges to a small mesh-dependent value, not 0;
    # 5e-3 comfortably covers the 162-vertex value (~2e-3) while still
    # catching any real failure to round up.  tests/bench measures the
    # resolution scaling.
    good = (rad_cv < 5e-3 and 0.0 < area_exc < 5e-3 and p_err < 2e-2
            and drift < 1e-10 and rise <= 1e-12)
    ok &= good
    print(f"volume: squashed sphere -> sphere: radius cv {rad_cv:.2e}, "
          f"area excess {area_exc:.2e} (discrete>=0), pressure err "
          f"{p_err:.2e} vs 2/r, drift {drift:.1e}, max rise {rise:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # --- walls compose with the restore: perturbed two-cube with its
    # outer x = 2 face constrained to the plane, volumes AND wall both
    # recovered to tolerance ------------------------------------------
    V, T, labels = _two_cube_mesh()
    m = np.abs(V[:, 0] - 2.0) < 1e-9
    pw = _walls.PlaneWall([2.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    wl = [(pw, m)]
    Vp = V + rng.normal(0.0, 0.05, V.shape)
    _walls.project_all(Vp, wl)

    def _proj(Varr):
        _walls.project_all(Varr, wl)

    def _tang(G, Varr):
        for i in range(len(G)):
            _walls.tangent_all(G[i], Varr, wl)
        return G

    b4, aft, rounds = restore_volumes(Vp, T, labels, [1.0, 1.0],
                                      project=_proj, tangent=_tang)
    wres = _walls.wall_residual(Vp, wl)
    good = aft < 1e-12 and b4 > 1e-3 and wres < 1e-12
    ok &= good
    print(f"volume: restore with plane wall {b4:.2e} -> {aft:.2e} "
          f"({rounds} rounds), wall residual {wres:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # --- one-sided floor: the squashed bubble dips below z = -0.8;
    # evolution must end violation-free, volume-exact, monotone, and
    # still reach the Young-Laplace pressure (the floor goes inactive
    # as the sphere rounds up and lifts off) ---------------------------
    SV, SF = _icosphere_mesh(2)
    Vb = SV * np.array([1.25, 0.8, 1.0])
    lab = np.zeros((len(SF), 2), dtype=np.int64)
    lab[:, 1] = 1
    floor = _walls.PlaneWall([0.0, 0.0, -0.8], [0.0, 0.0, 1.0],
                             one_sided=True)
    target = 4.0 * np.pi / 3.0
    info = evolve(Vb, SF, lab, targets=[target], iters=120,
                  walls=[(floor, np.ones(len(Vb), dtype=bool))])
    zmin = float(Vb[:, 2].min())
    p_err = abs(info["pressures"][0] - 2.0) / 2.0
    drift = max(hh["drift_post"] for hh in info["history"])
    rise = max(hh["rise"] for hh in info["history"] if not hh["groomed"])
    wres = max(hh["wall_resid"] for hh in info["history"])
    good = (zmin >= -0.8 - 1e-9 and p_err < 2e-2 and drift < 1e-10
            and rise <= 1e-12 and wres < 1e-9)
    ok &= good
    print(f"volume: one-sided floor evolve: min z {zmin:.4f} (>= -0.8), "
          f"pressure err {p_err:.2e}, drift {drift:.1e}, max rise "
          f"{rise:.1e}, wall resid {wres:.1e} {'OK' if good else 'FAIL'}")

    # --- zero bodies (open film, labels all (0,0)): evolve degrades to
    # pure area descent -- 0 x 0 Gram, empty pressures, no restore.
    # A pinned-rim cylinder lathe must shrink toward the catenoid
    # monotonically with the wall machinery untouched -----------------
    nring, nrow, hh2 = 24, 9, 0.5
    th = np.linspace(0.0, 2 * np.pi, nring, endpoint=False)
    zs = np.linspace(-hh2, hh2, nrow)
    Vf = np.array([[np.cos(t), np.sin(t), z] for z in zs for t in th])
    Tf = []
    for j in range(nrow - 1):
        for i in range(nring):
            a0 = j * nring + i
            b0 = j * nring + (i + 1) % nring
            Tf.append([a0, b0, a0 + nring])
            Tf.append([b0, b0 + nring, a0 + nring])
    Tf = np.asarray(Tf, dtype=np.int64)
    labf = np.zeros((len(Tf), 2), dtype=np.int64)
    fixf = np.zeros(len(Vf), dtype=bool)
    fixf[:nring] = True
    fixf[-nring:] = True
    A0 = mesh_area(Vf, Tf)
    info = evolve(Vf, Tf, labf, iters=100, fixed=fixf)
    rise = max(hh["rise"] for hh in info["history"] if not hh["groomed"])
    waist = float(np.mean(np.hypot(
        Vf[(nrow // 2) * nring:(nrow // 2) * nring + nring, 0],
        Vf[(nrow // 2) * nring:(nrow // 2) * nring + nring, 1])))
    # analytic bound: the catenoid between these rings has area ~5.99
    # = 0.956 * the cylinder's 6.265, so require < 0.97 * A0 and the
    # waist near the catenoid's c ~ 0.848
    good = (len(info["pressures"]) == 0 and info["area"] < 0.97 * A0
            and rise <= 1e-12 and 0.8 < waist < 0.9)
    ok &= good
    print(f"volume: zero-body film (cylinder -> catenoid): area "
          f"{A0:.4f} -> {info['area']:.4f}, waist {waist:.4f}, "
          f"pressures empty, max rise {rise:.1e} "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.volume self-test failed")
