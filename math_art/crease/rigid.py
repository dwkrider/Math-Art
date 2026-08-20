# Rigid origami: folding a crease pattern with flat, undeformed panels.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy`.
#
# THE FORMULATION.  The unknowns are the FOLD ANGLES, one per interior
# crease -- not the vertex positions.  A rigid folding is exactly an
# assignment of fold angles for which the paper closes up around every
# interior vertex.  Working in angle space makes the panel rigidity
# automatic (the panels never appear in the unknowns, so they can never
# stretch) and shrinks the problem from 3V unknowns to E.
#
# THE CONSTRAINT.  Walk once around an interior vertex.  Between
# consecutive creases you turn by the sector angle; at each crease you
# fold by its dihedral angle.  Returning to the start must be the
# identity:
#
#     M(rho) = prod_i [ Rx(rho_i) . Rz(alpha_i) ] = I
#
# with the creases in counter-clockwise order and alpha_i the sector
# between crease i and crease i+1.  Two properties make this the right
# object to solve:
#
#   * at rho = 0 it reduces to Rz(sum of alpha_i) = Rz(2*pi) = I, so the
#     UNFOLDED sheet satisfies it exactly when the vertex is developable;
#   * it is a rotation, so the residual is three numbers, not nine -- we
#     take the axis-angle (vee of the skew part), which vanishes iff
#     M = I.
#
# SOLVING IT.  Newton on the residual, with the step taken by
# least squares rather than a square solve.  This matters: the constraint
# Jacobian of a real pattern is RANK DEFICIENT.  A quad-mesh Miura is
# overconstrained -- redundant constraints are precisely why it has one
# degree of freedom -- so forming J.J^T and inverting it, as a naive
# reading of the projection method suggests, hits a singular matrix on
# the very family the solver exists to fold.  `np.linalg.lstsq` returns
# the minimum-norm solution and does not care about the rank.
#
# The nullity of J is also the honest DOF count.  The textbook
# bookkeeping formula DOF = N - 3M holds only at full rank, and goes
# NEGATIVE for the Miura, so it is reported here as a diagnostic and
# never used as a test.
#
# DRIVING AND CONTINUATION.  One crease is nominated the driver and its
# angle appended to the residual as a hard row.  The fold path is walked
# by continuation -- step the driver a little, re-converge, repeat --
# which both keeps Newton in its basin and produces the sequence of
# states an animation needs.
#
# PLACING THE PAPER.  Fold angles do not by themselves give coordinates.
# A breadth-first walk over the face adjacency graph does: start a face
# at the identity, and cross each shared crease by rotating about that
# crease's line by its fold angle.  The vertex constraints are exactly
# the condition that this walk is path-independent, so on a converged
# solution every route to a face agrees.
#
# References:
#   T. Tachi, "Simulation of Rigid Origami," Origami^4 (A K Peters,
#       2009), pp. 175-187 -- the projection method this follows.
#   s. belcastro, T. C. Hull, "Modelling the folding of paper into three
#       dimensions using affine transformations," Linear Algebra and its
#       Applications 348, 2002, pp. 273-282 -- the product-of-rotations
#       condition at a vertex.
#   M. Schenk, S. D. Guest, "Geometry of Miura-folded metamaterials,"
#       PNAS 110(9), 2013 -- the closed-form Miura state used as the
#       oracle in the self-test.
#   H. Akitaya, E. D. Demaine, T. Horiyama, T. C. Hull, J. S. Ku,
#       T. Tachi, "Rigid Foldability is NP-Hard," 2018 -- why nothing
#       here promises to fold an arbitrary pattern.

import numpy as np

from .fold_io import BOUNDARY, FLAT
from .graph import vertex_rings
from .validate import sector_angles


class FoldFailure(RuntimeError):
    """Newton did not converge, or the pattern is not rigidly foldable."""


def _rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _vee(M):
    """Axis-angle vector of the skew part; zero iff M is the identity."""
    return 0.5 * np.array([M[2, 1] - M[1, 2],
                           M[0, 2] - M[2, 0],
                           M[1, 0] - M[0, 1]], dtype=float)


def _axis_rotation(axis, angle):
    """Rodrigues rotation about a unit axis."""
    n = axis / (np.linalg.norm(axis) or 1.0)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return (np.eye(3) + np.sin(angle) * K +
            (1.0 - np.cos(angle)) * (K @ K))


def _anchor_to_flat(folded, flat_xy):
    """Rigidly best-fit a folded state back onto the flat pattern.

    WHY THIS IS NEEDED.  A fold is only determined up to a rigid motion:
    the constraints say how the panels sit relative to EACH OTHER and
    nothing about where the sheet is in space.  The breadth-first walk
    resolves that arbitrarily, by pinning whichever face it happened to
    start from -- so the model appears to pivot about that face's corner
    as the fold progresses, and worse, the apparent pivot drifts along
    the fold path, which reads as the paper swinging rather than folding.

    Fixing it is the orthogonal Procrustes problem: find the rotation R
    and translation t minimising |R q + t - p| over the flat positions p
    and folded positions q.  Kabsch's solution is the SVD of the
    cross-covariance, with a sign guard so a reflection is never
    returned -- a reflected "fit" would turn the paper inside out and is
    a worse answer than a poor rotation.

    The effect is that the sheet stays centred and its mid-surface stays
    in the plane it started in, contracting and rising rather than
    swinging.  For a Miura that is exactly the familiar picture.

    References:
      W. Kabsch, "A solution for the best rotation to relate two sets of
          vectors," Acta Crystallographica A32, 1976, pp. 922-923.
    """
    P = np.hstack([np.asarray(flat_xy, dtype=float),
                   np.zeros((len(flat_xy), 1))])
    Q = np.asarray(folded, dtype=float)
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (Q - qc).T @ (P - pc)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    return (R @ (Q - qc).T).T + pc


class RigidFolder:
    """A crease pattern prepared for folding.

    Construction does the once-only work: find the interior vertices,
    order their creases counter-clockwise, measure the sector angles,
    and index the free fold angles.
    """

    def __init__(self, frame, angle_tol=1e-7):
        if frame.verts is None or frame.edges is None:
            raise FoldFailure("a rigid folding needs vertices and edges")
        if not frame.is_flat:
            raise FoldFailure(
                "the frame is already folded; rigid folding starts from "
                "the flat crease pattern")
        if frame.faces is None:
            raise FoldFailure(
                "no faces: call graph.build_faces before folding")

        self.frame = frame
        self.verts0 = np.asarray(frame.verts, dtype=float)[:, :2]
        self.edges = np.asarray(frame.edges, dtype=np.int64)
        self.faces = [list(f) for f in frame.faces]

        # An edge index for a vertex pair, both ways round.
        self.edge_of = {}
        for k, (a, b) in enumerate(self.edges):
            self.edge_of[(int(a), int(b))] = k
            self.edge_of[(int(b), int(a))] = k

        # Free variables: every crease that is not boundary and not flat.
        assign = frame.assignment
        self.free = []
        for k in range(len(self.edges)):
            s = None if assign is None else str(assign[k])
            if s in (BOUNDARY, FLAT):
                continue
            self.free.append(k)
        self.free = np.array(self.free, dtype=np.int64)
        self.var_of = {int(k): i for i, k in enumerate(self.free)}

        # Interior vertices, with creases in CCW order.
        rings = vertex_rings(frame.n_verts, self.edges, frame.verts)
        on_boundary = np.zeros(frame.n_verts, dtype=bool)
        if assign is not None:
            b = assign == BOUNDARY
            if b.any():
                on_boundary[self.edges[b].ravel()] = True

        self.vertices = []
        for v in range(frame.n_verts):
            ring = rings[v]
            if on_boundary[v] or len(ring) < 3:
                continue
            ang = sector_angles(frame.verts, ring, v)
            if abs(float(ang.sum()) - 2 * np.pi) > 1e-6:
                continue                     # not developable: not foldable
            self.vertices.append((v, ring, ang))

        self.n_vars = len(self.free)
        self.n_rows = 3 * len(self.vertices)

    # -- the constraint ---------------------------------------------
    def residual(self, rho, driver=None, target=0.0):
        """Closure residual at every interior vertex, plus the driver row."""
        out = np.empty(self.n_rows + (1 if driver is not None else 0))
        for n, (v, ring, ang) in enumerate(self.vertices):
            M = np.eye(3)
            for u, a in zip(ring, ang):
                k = self.edge_of[(v, int(u))]
                i = self.var_of.get(k)
                M = M @ _rx(0.0 if i is None else rho[i]) @ _rz(a)
            out[3 * n:3 * n + 3] = _vee(M)
        if driver is not None:
            out[-1] = rho[driver] - target
        return out

    def jacobian(self, rho, driver=None, target=0.0, h=1e-6):
        """Central differences.

        The analytic derivative of a product of rotations is available,
        but the vertex rings are short (degree 4 to 6) and patterns are
        small, so the O(n_vars) extra products cost less than the risk of
        a sign slip in a hand-derived Jacobian.
        """
        rows = self.n_rows + (1 if driver is not None else 0)
        J = np.zeros((rows, self.n_vars))
        for i in range(self.n_vars):
            step = np.zeros(self.n_vars)
            step[i] = h
            J[:, i] = (self.residual(rho + step, driver, target) -
                       self.residual(rho - step, driver, target)) / (2 * h)
        return J

    def solve(self, rho, driver, target, iters=40, tol=1e-10):
        """Newton-project onto the constraint manifold at a driver angle."""
        rho = np.array(rho, dtype=float)
        for _ in range(iters):
            r = self.residual(rho, driver, target)
            err = float(np.linalg.norm(r))
            if err < tol:
                return rho, err
            J = self.jacobian(rho, driver, target)
            # Least squares, not a square solve: J is rank deficient for
            # any overconstrained pattern, the Miura included.
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            # A short damping guard keeps the first steps out of trouble
            # when the driver is stepped coarsely.
            scale = min(1.0, 0.5 / (np.abs(step).max() + 1e-12))
            rho = rho + scale * step
        r = self.residual(rho, driver, target)
        raise FoldFailure(
            f"did not converge: residual {np.linalg.norm(r):.3e} after "
            f"{iters} iterations")

    def fold_path(self, driver, target, steps=12, rho0=None):
        """Continuation from flat to `target`, returning every state.

        The intermediate states are the point, not a side effect: they
        are what an animation caches, and stepping is also what keeps
        Newton inside its basin of attraction.
        """
        rho = np.zeros(self.n_vars) if rho0 is None else np.array(rho0, float)
        out = [rho.copy()]
        for s in range(1, steps + 1):
            t = target * s / steps
            rho, _ = self.solve(rho, driver, t)
            out.append(rho.copy())
        return out

    def dof(self, rho):
        """Numerical nullity of the constraint Jacobian at `rho`.

        This -- not N - 3M -- is the usable degree-of-freedom count: it
        is computed from the actual rank, so redundant constraints do not
        make it negative.
        """
        if not self.n_rows:
            return self.n_vars
        J = self.jacobian(rho)
        if J.size == 0:
            return self.n_vars
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int((sv > max(J.shape) * np.finfo(float).eps * sv[0]).sum()) \
            if sv[0] > 0 else 0
        return self.n_vars - rank

    # -- placing the paper ------------------------------------------
    def place(self, rho, anchor=True):
        """Vertex positions in 3-D for a converged set of fold angles.

        With `anchor` (the default) the result is rigidly best-fit back
        onto the flat pattern.  Without it the walk leaves the sheet
        hanging off whichever face happened to be placed first, so the
        model appears to swing about that face's corner as it folds --
        which is an artefact of the traversal, not of the fold.
        """
        n_faces = len(self.faces)
        if not n_faces:
            raise FoldFailure("no faces to place")

        # face adjacency across shared edges
        face_of_edge = {}
        for fi, f in enumerate(self.faces):
            for n in range(len(f)):
                p, q = int(f[n]), int(f[(n + 1) % len(f)])
                face_of_edge.setdefault(self.edge_of[(p, q)], []).append(
                    (fi, p, q))

        placed = np.zeros(n_faces, dtype=bool)
        T = [np.eye(4) for _ in range(n_faces)]
        order = [0]
        placed[0] = True
        head = 0
        while head < len(order):
            fi = order[head]
            head += 1
            f = self.faces[fi]
            for n in range(len(f)):
                p, q = int(f[n]), int(f[(n + 1) % len(f)])
                k = self.edge_of[(p, q)]
                for (gj, _p2, _q2) in face_of_edge.get(k, ()):
                    if gj == fi or placed[gj]:
                        continue
                    i = self.var_of.get(k)
                    ang = 0.0 if i is None else float(rho[i])
                    # Rotate about the shared crease, oriented by the
                    # PARENT's winding so the two faces disagree in sign
                    # exactly as they should.
                    a3 = np.append(self.verts0[p], 0.0)
                    b3 = np.append(self.verts0[q], 0.0)
                    R = _axis_rotation(b3 - a3, ang)
                    A = np.eye(4)
                    A[:3, :3] = R
                    A[:3, 3] = a3 - R @ a3
                    T[gj] = T[fi] @ A
                    placed[gj] = True
                    order.append(gj)

        if not placed.all():
            raise FoldFailure(
                f"{int((~placed).sum())} face(s) are not connected to the "
                "rest of the sheet")

        out = np.zeros((len(self.verts0), 3))
        seen = np.zeros(len(self.verts0), dtype=bool)
        for fi in order:
            M = T[fi]
            for v in self.faces[fi]:
                if seen[v]:
                    continue
                p = np.append(self.verts0[v], 0.0)
                out[v] = (M[:3, :3] @ p) + M[:3, 3]
                seen[v] = True
        return _anchor_to_flat(out, self.verts0) if anchor else out


def fold(frame, target, driver=None, steps=12):
    """Fold a flat pattern to a driver angle; return (positions, rho, path).

    `target` is the driving crease's fold angle in radians.  `driver` is
    an index into the free creases; the first free crease is used when it
    is not given.
    """
    f = RigidFolder(frame)
    if not f.n_vars:
        raise FoldFailure("this pattern has no foldable creases")
    d = 0 if driver is None else int(driver)
    path = f.fold_path(d, float(target), steps=steps)
    return f.place(path[-1]), path[-1], path


def _selftest():
    from . import patterns
    from .graph import build_faces

    # --- the accordion: no interior vertices, so folding is free -----
    ac = patterns.accordion(count=4, spacing=0.5, length=1.0)
    ac.faces = build_faces(ac.verts, ac.edges)
    f = RigidFolder(ac)
    assert f.n_rows == 0, "an accordion has no interior vertices"
    P, rho, path = fold(ac, np.deg2rad(60.0), steps=4)
    assert P.shape == (ac.n_verts, 3)
    assert np.abs(P[:, 2]).max() > 1e-3, "the accordion did not leave the plane"
    # panels stayed rigid: every crease keeps its flat length
    for (a, b) in ac.edges:
        d0 = np.linalg.norm(ac.verts[a][:2] - ac.verts[b][:2])
        d1 = np.linalg.norm(P[a] - P[b])
        assert abs(d0 - d1) < 1e-9, (d0, d1)

    # --- the Miura, against Schenk and Guest's closed form -----------
    a_len, b_len, gamma = 1.0, 1.0, np.deg2rad(60.0)
    rows = cols = 4
    mi = patterns.miura(rows=rows, cols=cols, panel_a=a_len,
                        panel_b=b_len, alpha=gamma)
    mi.faces = build_faces(mi.verts, mi.edges)
    folder = RigidFolder(mi)
    assert len(folder.vertices) == 9
    # flat is a solution: the residual vanishes at rho = 0
    assert np.linalg.norm(folder.residual(np.zeros(folder.n_vars))) < 1e-12

    # Nullity is the usable DOF count.  The bookkeeping formula
    # N - 3M would be 24 - 27 = -3 here, which is why it is not used.
    assert folder.n_vars - 3 * len(folder.vertices) < 0

    P, rho, path = fold(mi, np.deg2rad(70.0), steps=10)
    assert len(path) == 11

    # -- EVERY crease must fold.  KNOWN FAILURE, and the point of this
    # -- assertion is that it fails: see BRANCH SELECTION below.
    #
    # The flat state is a bifurcation point -- the Jacobian's nullity
    # here is 6, not 1 -- and a least-norm Newton step from it slides
    # onto a DEGENERATE BUT VALID branch: every straight row line folds
    # like an accordion while the zigzag creases stay dead flat.  That
    # really is a rigid folding of this crease pattern, because the two
    # row-line segments at a Miura vertex are collinear, so folding them
    # alone is just bending the sheet along one straight line.  It is
    # simply not the Miura.
    #
    # The distance oracle below does NOT catch it: it samples a few
    # vertex pairs that happen to stay consistent, which is why this
    # check is stated separately and in terms of the creases themselves.
    zig = np.array([abs(mi.verts[p][1] - mi.verts[q][1]) > 1e-9
                    for (p, q) in mi.edges[folder.free]])
    folded_zig = int((np.abs(rho[zig]) > 1e-6).sum())
    assert folded_zig == int(zig.sum()), (
        f"only {folded_zig} of {int(zig.sum())} zigzag creases folded -- "
        "the solver picked the accordion branch, not the Miura")

    # rigidity: no crease changed length
    for (p, q) in mi.edges:
        d0 = np.linalg.norm(mi.verts[p][:2] - mi.verts[q][:2])
        d1 = np.linalg.norm(P[p] - P[q])
        assert abs(d0 - d1) < 1e-8, (d0, d1)

    # it actually left the plane
    assert np.abs(P[:, 2]).max() > 1e-3

    # -- the oracle.  Schenk and Guest give the folded Miura in closed
    # -- form; two independent measured distances must agree through it
    # -- for a single consistent fold parameter theta.
    def vid(i, j):
        want = (j * a_len + (i % 2) * b_len * np.cos(gamma),
                i * b_len * np.sin(gamma))
        hit = np.nonzero((np.abs(mi.verts[:, 0] - want[0]) < 1e-9) &
                         (np.abs(mi.verts[:, 1] - want[1]) < 1e-9))[0]
        assert len(hit) == 1, (i, j)
        return int(hit[0])

    # 2S across two rows, 2L across two columns (both are exact grid
    # spacings in Schenk's p(i, j), free of the alternating offsets)
    S_meas = 0.5 * np.linalg.norm(P[vid(2, 0)] - P[vid(0, 0)])
    L_meas = 0.5 * np.linalg.norm(P[vid(0, 2)] - P[vid(0, 0)])

    # L = a sqrt(1 - sin^2(theta) sin^2(gamma))  ->  recover theta
    s2 = (1.0 - (L_meas / a_len) ** 2) / (np.sin(gamma) ** 2)
    assert -1e-9 <= s2 <= 1.0 + 1e-9, s2
    theta = np.arcsin(np.sqrt(max(0.0, min(1.0, s2))))
    assert theta > 1e-3, "the solver returned the flat state"

    # S = b cos(theta) tan(gamma) / sqrt(1 + cos^2(theta) tan^2(gamma))
    ct, tg = np.cos(theta), np.tan(gamma)
    S_pred = b_len * ct * tg / np.sqrt(1.0 + ct * ct * tg * tg)
    assert abs(S_pred - S_meas) < 1e-6, (S_pred, S_meas, np.rad2deg(theta))

    # H and V, the other two closed-form lengths, from the same theta
    H_pred = a_len * np.sin(theta) * np.sin(gamma)
    V_pred = b_len / np.sqrt(1.0 + ct * ct * tg * tg)
    d01 = np.linalg.norm(P[vid(1, 0)] - P[vid(0, 0)])
    assert abs(d01 - np.hypot(S_pred, V_pred)) < 1e-6, (d01, S_pred, V_pred)
    d10 = np.linalg.norm(P[vid(0, 1)] - P[vid(0, 0)])
    assert abs(d10 - np.hypot(L_meas, H_pred)) < 1e-6, (d10, H_pred)

    # --- continuation really is a path, not a jump ------------------
    mags = [float(np.abs(r).max()) for r in path]
    assert mags[0] == 0.0
    assert all(x <= y + 1e-9 for x, y in zip(mags, mags[1:])), mags

    # --- anchoring: the sheet stays put instead of swinging ---------
    # Unanchored, the walk pins whichever face it started from, so the
    # centroid wanders as the fold proceeds.  Anchored, it does not.
    raw = [folder.place(r, anchor=False) for r in path]
    fix = [folder.place(r, anchor=True) for r in path]

    # the anchored states are all centred on the flat sheet's centroid
    flat_c = np.append(mi.verts[:, :2].mean(axis=0), 0.0)
    for p in fix:
        assert np.allclose(p.mean(axis=0), flat_c, atol=1e-9)
    # ... while the raw ones drift away from it as the fold proceeds
    drift = [float(np.linalg.norm(p.mean(axis=0) - flat_c)) for p in raw]
    assert drift[-1] > 1e-3, drift[-1]

    # anchoring is a RIGID motion, so it cannot change the shape: every
    # pairwise distance is preserved exactly
    for r, fx in ((raw[-1], fix[-1]), (raw[len(raw) // 2],
                                       fix[len(fix) // 2])):
        for _ in range(20):
            i, j = np.random.default_rng(3).integers(0, len(r), 2)
            if i != j:
                assert abs(np.linalg.norm(r[i] - r[j]) -
                           np.linalg.norm(fx[i] - fx[j])) < 1e-9
    # and it never reflects: a reflected fit would turn the paper inside
    # out, so check the flat state comes back as itself
    flat_again = folder.place(np.zeros(folder.n_vars))
    assert np.allclose(flat_again[:, :2], mi.verts[:, :2], atol=1e-9)
    assert np.abs(flat_again[:, 2]).max() < 1e-9

    print("RESULT: OK  crease.rigid")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
