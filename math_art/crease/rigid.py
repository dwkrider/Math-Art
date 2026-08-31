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
# THE CONSTRAINT, AND WHY IT IS NOT WRITTEN SEPARATELY FROM THE
# PLACEMENT.  A rigid folding is one where the paper closes up around
# every interior vertex.  Walk the faces around a vertex, stepping over
# each incident crease, and the accumulated rotation must be the
# identity.  Every crease at the vertex passes through it, so the
# translations drop out and the product is a pure rotation -- three
# numbers, taken as the axis-angle of its skew part.
#
# Crucially, `residual` and `place` step across a crease using the SAME
# primitive, `_cross`.  Earlier versions derived the two independently
# and they silently disagreed: a configuration could satisfy the
# constraint to 1e-16 and still TEAR THE SHEET when placed, with edge
# lengths off by 0.57 on unit panels.  Sharing the primitive makes "the
# residual vanishes" and "the placement walk closes" the same statement
# by construction.
#
# THREE CONVENTION TRAPS, all of which bit:
#
#   * An edge's two sides must be named by its STORED direction a -> b,
#     never by a face's winding.  Winding is how a face happened to be
#     written down; left and right are properties of the edge, and only
#     those survive being reached from another direction.
#   * The internal angle's sign is defined against that stored direction,
#     which is arbitrary, so it does NOT agree with the FOLD convention
#     (valley positive) edge by edge.  `_valley_sign` measures the
#     relation per crease rather than assuming a global one.
#   * "At rho = 0 the residual vanishes" is TRUE FOR EVERY ordering of
#     the vertex walk, so it cannot distinguish a correct convention from
#     a wrong one.  It is necessary and worthless alone.  The self-test
#     therefore checks a FOLDED state against Schenk and Guest's closed
#     form, which is the only thing that actually settled this.
#
# THE DERIVATIVE IS EXACT, AND THAT IS WHERE THE SPEED IS.  Each factor
# in the vertex product turns about a FIXED axis -- the crease direction
# in the flat pattern, which does not move as the sheet folds -- so
#
#     dM/drho_j = R_1 ... R_{j-1} (sigma_j K_j R_j) R_{j+1} ... R_n
#
# and dr/drho_j = vee(dM/drho_j), since vee is linear.  Prefix and suffix
# products make a whole vertex cost O(degree) instead of O(n_vars *
# degree).  The finite-difference version this replaced needed
# 2*n_vars residual evaluations and was 99% of the solve time: on an
# 8x10 Miura the Jacobian went from 813 ms to 2.9 ms (279x, matching the
# predicted 2*n_vars) and a full fold from 64 s to 0.67 s.
#
# `jacobian_fd` is kept deliberately.  It is an INDEPENDENT reference for
# the analytic one, and the self-test compares them along a fold path --
# not merely at rho = 0, which is where every wrong convention in this
# module's history still looked right.
#
# SOLVING IT.  Newton on the residual, with the step taken by least
# squares rather than a square solve, because the Jacobian is RANK
# DEFICIENT: a quad-mesh Miura is overconstrained -- redundant
# constraints are precisely why it has one degree of freedom -- so
# forming J.J^T and inverting it hits a singular matrix on the very
# family the solver exists to fold.
#
# The nullity of J is the honest DOF count.  The bookkeeping formula
# DOF = N - 3M holds only at full rank and is NEGATIVE for the Miura, so
# it is a diagnostic here and never a test.
#
# LEAVING THE FLAT STATE.  Flat is a bifurcation point: the nullity there
# is 8 for a 4x6 Miura and drops to 1 the instant the sheet moves.  A
# least-norm step from flat slides onto a degenerate but VALID branch --
# every straight row line folding like an accordion while the zigzags
# stay dead flat -- because the two row-line creases at a Miura vertex
# are collinear, so folding them alone just bends the sheet along a
# line.  Escaping needs three things: a seed direction with the right
# relative magnitudes (mountain/valley signs alone are not enough, since
# the two families fold at a ratio of cos(alpha)); a correction step
# taken ORTHOGONAL to the tangent, or it undoes the step it is
# correcting; and a step scaled by the tangent's leading component, so
# a large pattern advances as far per step as a small one.
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

from .fold_io import BOUNDARY, FLAT, MOUNTAIN
from .graph import vertex_rings
from .validate import sector_angles


class FoldFailure(RuntimeError):
    """Newton did not converge, or the pattern is not rigidly foldable."""


#: Largest closure residual a state may carry and still count as folded.
#: Converged states in this module sit at 1e-16 to 1e-13, so this is
#: several orders of margin -- it separates "solved" from "diverged",
#: not "accurate" from "less accurate".
_PATH_TOL = 1e-6


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


def _expm_skew(K, angle):
    """Rodrigues from a precomputed unit-axis skew matrix."""
    return (np.eye(3) + np.sin(angle) * K +
            (1.0 - np.cos(angle)) * (K @ K))


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

        # ONE CONVENTION, SHARED.  Every crossing of a crease -- whether
        # done by `place` to lay the paper out, or by `residual` to ask
        # whether the paper closes up around a vertex -- goes through
        # `_cross`.  That is deliberate and it is the whole point of this
        # arrangement: an earlier version derived the two independently
        # and they disagreed, so a configuration could satisfy the
        # constraint exactly (residual 1e-16) and still tear the sheet
        # when placed (edge lengths off by 0.57).  Sharing the primitive
        # makes "the residual vanishes" and "the walk closes" the same
        # statement by construction rather than by hope.
        #
        # An edge's two sides are named by its STORED direction a -> b,
        # not by any face's winding: the face whose third vertex lies to
        # the left of a -> b is `left_of[k]`, the other `right_of[k]`.
        # Winding is a property of how a face was written down; left and
        # right are properties of the edge itself, and only the latter
        # survives being reached from a different direction.
        self.left_of = {}
        self.right_of = {}
        for fi, fc in enumerate(self.faces):
            m = len(fc)
            for n in range(m):
                a, b = int(fc[n]), int(fc[(n + 1) % m])
                k = self.edge_of[(a, b)]
                ka, kb = (int(self.edges[k][0]), int(self.edges[k][1]))
                other = [w for w in fc if w not in (ka, kb)]
                if not other:
                    continue
                u = self.verts0[kb] - self.verts0[ka]
                w = self.verts0[other[0]] - self.verts0[ka]
                if u[0] * w[1] - u[1] * w[0] > 0.0:
                    self.left_of[k] = fi
                else:
                    self.right_of[k] = fi

        # Fixed per-crossing data, so the residual and its derivative do
        # no geometry work per call.
        self.terms = self._vertex_terms()

    def _vertex_terms(self):
        """Per interior vertex, the fixed data every crossing needs.

        The rotation axis of a crease is its direction in the FLAT
        pattern, which does not depend on the fold angles at all -- so
        the axis, its skew matrix, and the traversal sign can all be
        computed once here rather than rebuilt on every residual call.
        That is what makes an analytic derivative cheap: the only thing
        varying is the angle.
        """
        terms = []
        for (v, ring, _ang) in self.vertices:
            row = []
            for u in ring:
                k = self.edge_of[(v, int(u))]
                to_left = int(self.edges[k][0]) == v
                a, b = int(self.edges[k][0]), int(self.edges[k][1])
                axis = np.append(self.verts0[b] - self.verts0[a], 0.0)
                n = axis / (np.linalg.norm(axis) or 1.0)
                K = np.array([[0.0, -n[2], n[1]],
                              [n[2], 0.0, -n[0]],
                              [-n[1], n[0], 0.0]])
                sigma = -1.0 if to_left else 1.0
                row.append((k, self.var_of.get(k), n, K, sigma))
            terms.append(row)
        return terms

    def _cross(self, k, rho_k, to_left):
        """3x3 rotation for stepping across crease `k`.

        Positive `rho_k` rotates the RIGHT side about the edge's stored
        direction a -> b by +rho; going the other way undoes it.  Both
        `place` and `residual` step with this and nothing else.
        """
        a, b = int(self.edges[k][0]), int(self.edges[k][1])
        axis = np.append(self.verts0[b] - self.verts0[a], 0.0)
        return _axis_rotation(axis, -rho_k if to_left else rho_k)

    def _angle_of(self, rho, k):
        i = self.var_of.get(k)
        return 0.0 if i is None else float(rho[i])

    # -- the constraint ---------------------------------------------
    def residual(self, rho, driver=None, target=0.0):
        """How far the paper fails to close up around each interior vertex.

        Walk the faces around the vertex, stepping across each incident
        crease with `_cross`, and come back to where you started.  The
        accumulated rotation must be the identity.  Every crease at the
        vertex passes through it, so the translations drop out and the
        product is a pure rotation -- three numbers, taken as the
        axis-angle of the skew part.

        Because the steps are literally the ones `place` takes, a zero
        residual here MEANS the placement walk is path-independent
        around that vertex.  There is no second convention to get wrong.
        """
        out = np.empty(self.n_rows + (1 if driver is not None else 0))
        for n, (v, ring, _ang) in enumerate(self.vertices):
            M = np.eye(3)
            for u in ring:
                k = self.edge_of[(v, int(u))]
                # Going counter-clockwise about v, each crease is crossed
                # from its right side to its left when the edge points
                # away from v, and the other way when it points back.
                to_left = int(self.edges[k][0]) == v
                M = M @ self._cross(k, self._angle_of(rho, k), to_left)
            out[3 * n:3 * n + 3] = _vee(M)
        if driver is not None:
            out[-1] = rho[driver] - target
        return out

    def jacobian_fd(self, rho, driver=None, target=0.0, h=1e-6):
        """Central-difference Jacobian.

        Superseded by the analytic one for real work, and kept because
        it is an INDEPENDENT reference: the two must agree elementwise,
        not only at the flat state but along a fold path.  A derivative
        that is right only at rho = 0 is exactly the failure mode that
        produced three convention bugs in this module's history, so the
        self-test checks both.
        """
        rows = self.n_rows + (1 if driver is not None else 0)
        J = np.zeros((rows, self.n_vars))
        for i in range(self.n_vars):
            step = np.zeros(self.n_vars)
            step[i] = h
            J[:, i] = (self.residual(rho + step, driver, target) -
                       self.residual(rho - step, driver, target)) / (2 * h)
        return J

    def jacobian(self, rho, driver=None, target=0.0):
        """Exact derivative of the closure residual.

        The vertex residual is r = vee(M) with M = R_1 R_2 ... R_n, and
        each R_t = exp(sigma_t rho_t K_t) turns about a FIXED axis -- the
        crease direction in the flat pattern, which does not move as the
        sheet folds.  So

            dM/drho_j = R_1 ... R_{j-1} (sigma_j K_j R_j) R_{j+1} ... R_n

        and, since vee is linear, dr/drho_j = vee(dM/drho_j).

        Evaluated with prefix and suffix products, one vertex costs
        O(degree) matrix products rather than the O(n_vars * degree) the
        finite-difference version needs -- the whole Jacobian drops from
        2*n_vars residual evaluations to a single sweep.

        Note what is NOT here: no 1/cos(theta) anywhere.  Schenk and
        Guest's constraint carries that factor and it is singular at
        theta = +/- pi/2, in the middle of an ordinary fold; working in
        rotations rather than sines avoids it by construction.
        """
        rows = self.n_rows + (1 if driver is not None else 0)
        J = np.zeros((rows, self.n_vars))
        for n, row in enumerate(self.terms):
            m = len(row)
            # the individual rotations, in ring order
            Rs = []
            for (_k, idx, _axis, K, sigma) in row:
                ang = 0.0 if idx is None else float(rho[idx])
                Rs.append(_expm_skew(K, sigma * ang))
            # prefix[t] = R_0 ... R_{t-1};  suffix[t] = R_{t+1} ... R_{m-1}
            prefix = [np.eye(3)] * (m + 1)
            for t in range(m):
                prefix[t + 1] = prefix[t] @ Rs[t]
            suffix = [np.eye(3)] * (m + 1)
            for t in range(m - 1, -1, -1):
                suffix[t] = Rs[t] @ suffix[t + 1]
            for t, (_k, idx, _axis, K, sigma) in enumerate(row):
                if idx is None:
                    continue                      # boundary or flat crease
                dM = prefix[t] @ (sigma * K @ Rs[t]) @ suffix[t + 1]
                J[3 * n:3 * n + 3, idx] += _vee(dM)
        if driver is not None:
            J[-1, driver] = 1.0
        return J

    def _valley_sign(self, k):
        """+1 if a positive internal angle at crease `k` is a VALLEY.

        The internal angle is defined against each edge's STORED
        direction a -> b, and that order is whatever the pattern builder
        happened to emit -- so the relation between it and the FOLD
        convention (valley positive) flips from edge to edge.  Measuring
        it is one cheap test: rotate the right-hand face a little and see
        which way its far vertex goes.  Up is a valley.
        """
        lo, ro = self.left_of.get(k), self.right_of.get(k)
        if lo is None or ro is None:
            return 1.0
        a, b = int(self.edges[k][0]), int(self.edges[k][1])
        far = [w for w in self.faces[ro] if w not in (a, b)]
        if not far:
            return 1.0
        R = self._cross(k, 1e-3, to_left=False)
        p = np.append(self.verts0[far[0]], 0.0)
        a3 = np.append(self.verts0[a], 0.0)
        return 1.0 if float((R @ (p - a3))[2]) > 0.0 else -1.0

    def seed_direction(self):
        """The direction to leave the flat state along.

        A pattern that knows its own kinematics records relative fold
        magnitudes in `meta["fold_seed"]`; otherwise fall back to the
        mountain/valley signs, which fix every sign but assume every
        crease folds at the same rate.  For the Miura that assumption is
        wrong -- the two families differ by cos(alpha) -- and picking the
        wrong direction here is exactly what sends the solver onto the
        accordion branch.
        """
        seed = self.frame.meta.get("fold_seed")
        if seed is not None and len(seed) == len(self.edges):
            fold = np.asarray(seed, dtype=float)[self.free]
        else:
            assign = self.frame.assignment
            if assign is None:
                return np.ones(self.n_vars)
            fold = np.array([-1.0 if str(assign[k]) == MOUNTAIN else 1.0
                             for k in self.free])
        # The seed is stated in the FOLD convention (valley positive);
        # translate it edge by edge into the internal one.
        sgn = np.array([self._valley_sign(int(k)) for k in self.free])
        return fold * sgn

    def tangent(self, rho, prev=None, bias=None):
        """Unit tangent to the constraint manifold at `rho`.

        Chosen by projecting `prev` (or `bias` at the start) onto the
        null space, which is what keeps a continuation on ONE branch
        rather than hopping between the several that meet at the flat
        state.
        """
        ref = prev if prev is not None else bias
        J = self.jacobian(rho)
        if J.size == 0 or not self.n_rows:
            # No interior vertices means no closure conditions at all --
            # an accordion is the canonical case.  Every fold is valid,
            # so the seed direction IS the tangent.
            return ref / (np.linalg.norm(ref) or 1.0)
        U, S, Vt = np.linalg.svd(J)
        if not len(S) or S[0] <= 0:
            return ref / (np.linalg.norm(ref) or 1.0)
        tol = max(max(J.shape) * np.finfo(float).eps * S[0], 1e-9)
        null = Vt[int((S > tol).sum()):]
        if null.shape[0] == 0:
            return None
        t = null.T @ (null @ ref)
        if np.linalg.norm(t) < 1e-12:
            t = null[0]
        t = t / np.linalg.norm(t)
        if np.dot(t, ref) < 0:
            t = -t
        return t

    def _correct(self, rho, t, iters=30, tol=1e-12):
        """Newton back onto the manifold, ORTHOGONAL to the tangent.

        A plain minimum-norm correction is orthogonal to the null space
        at the current point -- which at the flat state is 8-dimensional
        and includes the very mode being followed, so the correction
        undoes the step.  Constraining it against the tangent keeps the
        progress that the step just made.
        """
        for _ in range(iters):
            r = self.residual(rho)
            if np.linalg.norm(r) < tol:
                break
            A = np.vstack([self.jacobian(rho), t[None, :]])
            rhs = np.concatenate([-r, [0.0]])
            d, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            rho = rho + d
        return rho

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

    def fold_path(self, target, steps=24):
        """Continuation from flat until the largest fold angle hits `target`.

        Returns every state along the way -- those are the point, not a
        side effect: they are what an animation caches, and stepping is
        also what keeps Newton in its basin.

        `target` is the biggest dihedral angle anywhere in the model,
        which is the quantity a user can actually see, rather than the
        angle of one arbitrarily chosen crease.
        """
        target = abs(float(target))
        rho = np.zeros(self.n_vars)
        out = [rho.copy()]
        t = self.tangent(rho, bias=self.seed_direction())
        if t is None:
            return out
        # The step is an arclength in fold-angle space, but the caller
        # asked for an ANGLE.  A unit tangent spreads its length over
        # every crease, so its largest component shrinks as the pattern
        # grows -- a fixed arclength step would advance a 6x8 sheet far
        # less per step than a 2x3 one, and larger patterns would stop
        # short of the target.  Scaling by the tangent's largest
        # component makes each step advance the biggest angle by the same
        # amount whatever the size.
        # ADAPTIVE STEP.  A CORRECTION THAT DID NOT CONVERGE IS NOT A
        # STATE.
        #
        # `_correct` returns whatever Newton reached, converged or not,
        # and this loop used to append that unconditionally -- so a
        # diverged step entered the path, and because its magnitude is
        # enormous it immediately satisfied the `>= target` test and
        # became the state the caller animated.  The pleated hypar did
        # exactly this near 179 degrees: fold angles of 2.3e5 degrees,
        # closure residual 0.88, and the paper stretched by 400 per cent
        # -- returned silently as if it were a fold.
        #
        # Rejecting the step is necessary but not sufficient: simply
        # stopping made the deep targets WORSE than the shallow ones,
        # because the step is target/steps, so asking for 179 takes a
        # bigger first stride than asking for 150 -- and the first
        # stride leaves the flat state, which is a bifurcation point and
        # the most delicate part of the whole path.  A single failure
        # there returned a flat sheet for the deepest fold requested.
        #
        # So halve on failure and grow back on success, which is what a
        # continuation method is supposed to do: the step shrinks
        # through the bifurcation and through any tight spot later,
        # and the path stops only when even a very small step cannot be
        # corrected -- a real limit of the pattern rather than an
        # artefact of how far the caller asked to go.
        want = target / max(1, steps)
        trial = want
        floor = want / 64.0
        taken = 0
        while taken < steps * 8:
            lead = float(np.abs(t).max()) or 1.0
            step = self._correct(rho + (trial / lead) * t, t)
            # `res` is EMPTY for a pattern with no interior vertices --
            # the accordion has none, so there is nothing to close and
            # every state is trivially valid.  `.max()` on an empty
            # array raises rather than returning a harmless zero, so the
            # size has to be checked before the magnitude.
            res = self.residual(step)
            if not np.isfinite(step).all() or (
                    res.size and float(np.abs(res).max()) > _PATH_TOL):
                trial *= 0.5
                if trial < floor:
                    break            # a real limit, not a step-size problem
                continue
            rho = step
            out.append(rho.copy())
            taken += 1
            trial = min(want, trial * 1.5)
            if float(np.abs(rho).max()) >= target:
                break
            nxt = self.tangent(rho, prev=t)
            if nxt is None:
                break
            t = nxt
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

        Breadth-first over the faces, stepping across each shared crease
        with `_cross` -- the same primitive the residual uses, so a
        configuration that satisfies the constraint lays out without
        tearing.

        With `anchor` (the default) the result is rigidly best-fit back
        onto the flat pattern; without it the sheet hangs off whichever
        face was placed first and appears to swing as it folds.
        """
        n_faces = len(self.faces)
        if not n_faces:
            raise FoldFailure("no faces to place")

        placed = np.zeros(n_faces, dtype=bool)
        T = [np.eye(4) for _ in range(n_faces)]
        order = [0]
        placed[0] = True
        head = 0
        while head < len(order):
            fi = order[head]
            head += 1
            fc = self.faces[fi]
            m = len(fc)
            for n in range(m):
                k = self.edge_of[(int(fc[n]), int(fc[(n + 1) % m]))]
                lo, ro = self.left_of.get(k), self.right_of.get(k)
                if lo is None or ro is None:
                    continue                       # boundary crease
                gj = ro if fi == lo else lo
                if gj == fi or placed[gj]:
                    continue
                R = self._cross(k, self._angle_of(rho, k),
                                to_left=(gj == lo))
                a3 = np.append(self.verts0[int(self.edges[k][0])], 0.0)
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
                q = np.append(self.verts0[v], 0.0)
                out[v] = (M[:3, :3] @ q) + M[:3, 3]
                seen[v] = True
        return _anchor_to_flat(out, self.verts0) if anchor else out


def fold(frame, target, steps=24):
    """Fold a flat pattern; return (positions, rho, path).

    `target` is the largest dihedral angle to reach, in radians.
    """
    f = RigidFolder(frame)
    if not f.n_vars:
        raise FoldFailure("this pattern has no foldable creases")
    path = f.fold_path(float(target), steps=steps)
    return f.place(path[-1]), path[-1], path


def _selftest():
    from . import patterns
    from .graph import build_faces

    # --- the accordion: no interior vertices, so folding is free -----
    ac = patterns.accordion(count=4, spacing=0.5, length=1.0)
    ac.faces = build_faces(ac.verts, ac.edges)
    f = RigidFolder(ac)
    assert f.n_rows == 0, "an accordion has no interior vertices"
    P, rho, path = fold(ac, np.deg2rad(60.0), steps=8)
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

    P, rho, path = fold(mi, np.deg2rad(70.0), steps=24)
    assert len(path) > 2

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

    # -- THE ANALYTIC JACOBIAN AGAINST ITS FREE ORACLE.
    #
    # The finite-difference Jacobian it replaced is an INDEPENDENT
    # reference, so there is no excuse for not checking against it -- and
    # checking at the flat state alone would not do.  rho = 0 is where
    # every wrong convention in this module's history still looked right,
    # so the comparison is made along an actual fold path as well.
    for (rr, cc) in ((4, 4), (4, 6)):
        mj = patterns.miura(rows=rr, cols=cc, alpha=np.deg2rad(60.0))
        mj.faces = build_faces(mj.verts, mj.edges)
        fj = RigidFolder(mj)
        pth = fj.fold_path(np.deg2rad(50.0), steps=8)
        states = [np.zeros(fj.n_vars),
                  np.linspace(-0.4, 0.4, fj.n_vars),
                  pth[len(pth) // 3], pth[-1]]
        for x in states:
            d = float(np.abs(fj.jacobian(x) - fj.jacobian_fd(x)).max())
            assert d < 1e-6, (
                f"{rr}x{cc}: analytic Jacobian differs from finite "
                f"differences by {d:.2e}")
        # the driver row is a plain unit entry, and must survive too
        d = float(np.abs(fj.jacobian(pth[-1], driver=0, target=0.2) -
                         fj.jacobian_fd(pth[-1], driver=0, target=0.2)).max())
        assert d < 1e-6, f"driver row mismatch: {d:.2e}"

    # -- THE ORACLE.  Compare the WHOLE folded state, vertex by vertex,
    # -- against Schenk and Guest's closed form.
    #
    # This is the check that matters, and its absence is what let three
    # separate convention errors survive.  Sampling a few distances did
    # not catch them; neither did "the flat state is a solution", which
    # is true for every ordering of the vertex walk.  Only a FOLDED state
    # measured against an independent source distinguishes right from
    # plausible.
    def _schenk(th, R, C, av, bv, gv):
        H = av * np.sin(th) * np.sin(gv)
        ct, tg = np.cos(th), np.tan(gv)
        S = bv * ct * tg / np.sqrt(1 + ct * ct * tg * tg)
        L = av * np.sqrt(1 - np.sin(th) ** 2 * np.sin(gv) ** 2)
        Vv = bv / np.sqrt(1 + ct * ct * tg * tg)
        return np.array([[i * S,
                          j * L + (Vv / 2) * (1 - (-1) ** i),
                          (H / 2) * (1 - (-1) ** j)]
                         for i in range(R + 1) for j in range(C + 1)])

    def _vid(frame, i, j, av, bv, gv):
        w = (j * av + (i % 2) * bv * np.cos(gv), i * bv * np.sin(gv))
        hit = np.nonzero((np.abs(frame.verts[:, 0] - w[0]) < 1e-9) &
                         (np.abs(frame.verts[:, 1] - w[1]) < 1e-9))[0]
        return int(hit[0])

    for (rr, cc, gd) in ((4, 4, 60.0), (4, 6, 60.0), (3, 5, 45.0)):
        gv = np.deg2rad(gd)
        mm = patterns.miura(rows=rr, cols=cc, panel_a=1.0, panel_b=1.0,
                            alpha=gv)
        mm.faces = build_faces(mm.verts, mm.edges)
        PP, _rr2, _pp = fold(mm, np.deg2rad(70.0), steps=16)
        idx = [_vid(mm, i, j, 1.0, 1.0, gv)
               for i in range(rr + 1) for j in range(cc + 1)]
        Q = PP[idx]

        def _dev(th):
            T = _schenk(th, rr, cc, 1.0, 1.0, gv)
            qc, tc = Q.mean(0), T.mean(0)
            U2, _s2, Vt2 = np.linalg.svd((Q - qc).T @ (T - tc))
            dd = np.sign(np.linalg.det(Vt2.T @ U2.T))
            Rt = Vt2.T @ np.diag([1.0, 1.0, dd]) @ U2.T
            return np.linalg.norm(((Rt @ (Q - qc).T).T + tc) - T,
                                  axis=1).max()

        grid = np.linspace(0.01, 1.55, 200)
        best = min(float(_dev(t)) for t in grid)
        assert best < 5e-3, (
            f"{rr}x{cc} alpha={gd}: best fit to the Schenk closed form is "
            f"{best:.2e} -- the solver is not producing a Miura")

    # --- the Yoshimura must CURL ------------------------------------
    #
    # It is the buckle pattern of a cylinder, so the direction along its
    # rows has to contract as it folds: that is the whole behaviour.  A
    # wrong-neighbour triangulation shipped here once and folded into a
    # straight trough instead -- corrugating in z while its width stayed
    # put to within 0.02 across the entire fold -- and no test noticed,
    # because the paper still did not stretch and every local
    # flat-foldability condition still held.  Width is the observable
    # that separates a cylinder from a channel, so measure it.
    ym = patterns.yoshimura(rows=4, cols=6, cell=1.0, height=0.8)
    ym.faces = build_faces(ym.verts, ym.edges)
    w0 = float(ym.verts[:, 0].max() - ym.verts[:, 0].min())
    widths = []
    for tgt in (20.0, 45.0, 70.0):
        PY, _r3, _p3 = fold(ym, np.deg2rad(tgt), steps=16)
        e0 = np.linalg.norm(ym.verts[ym.edges[:, 1]] -
                            ym.verts[ym.edges[:, 0]], axis=1)
        e1 = np.linalg.norm(PY[ym.edges[:, 1]] - PY[ym.edges[:, 0]], axis=1)
        assert float(np.abs(e1 / e0 - 1.0).max()) < 1e-9, (
            "yoshimura: the paper stretched")
        # width across the curling direction, measured on the principal
        # axes so it does not depend on how `place` happened to orient it
        Q = PY - PY.mean(0)
        span = sorted(float(np.ptp(Q @ v))
                      for v in np.linalg.svd(Q, full_matrices=False)[2])
        widths.append(span[-1])
    assert widths[0] < 0.85 * w0 and widths[-1] < 0.5 * w0, (
        f"yoshimura does not curl: widest extent went {w0:.2f} -> "
        f"{[round(w, 2) for w in widths]}; a cylinder buckle pattern must "
        f"contract, a straight trough keeps its width")
    assert widths[0] > widths[-1], (
        f"yoshimura should keep curling as it folds: {widths}")

    # --- no pattern may be handed back torn -------------------------
    #
    # `fold_path` used to append whatever `_correct` reached, converged
    # or not.  The pleated hypar near 179 degrees came back with fold
    # angles of 2.3e5 degrees, closure residual 0.88 and edges stretched
    # by 400 per cent -- silently, as a fold.  Stretch is the check that
    # cannot be argued with: paper does not stretch, so any state on the
    # path with a stretched edge is not a folding of this sheet.
    #
    # Every pattern, across the whole angle range the operator allows,
    # including the deep end where the step control has to shrink.
    for name, kw in (('MIURA', dict(rows=4, cols=6)),
                     ('ACCORDION', dict(count=8)),
                     ('WATERBOMB', dict(rows=3, cols=4)),
                     ('YOSHIMURA', dict(rows=4, cols=6)),
                     ('HYPAR', dict(rings=4, sides=6))):
        pf = patterns.build(name, **kw)
        pf.faces = build_faces(pf.verts, pf.edges)
        pfs = RigidFolder(pf)
        f0 = np.linalg.norm(pf.verts[pfs.edges[:, 1]] -
                            pf.verts[pfs.edges[:, 0]], axis=1)
        for tgt in (20.0, 43.0, 70.0, 120.0, 179.0):
            pp = pfs.fold_path(np.deg2rad(tgt), steps=12)
            # EVERY state, not just the last: a torn intermediate is a
            # torn frame of the animation.
            for si, pr in enumerate(pp):
                pv = pfs.place(pr)
                f1 = np.linalg.norm(pv[pfs.edges[:, 1]] -
                                    pv[pfs.edges[:, 0]], axis=1)
                assert float(np.abs(f1 / f0 - 1.0).max()) < 1e-8, (
                    f"{name} at {tgt} deg: state {si} of {len(pp)} stretches "
                    f"the paper by "
                    f"{float(np.abs(f1 / f0 - 1.0).max()):.2e}")
            pres = pfs.residual(pp[-1])
            assert not pres.size or float(np.abs(pres).max()) < 1e-6, (
                f"{name} at {tgt} deg: final state does not close, "
                f"residual {float(np.abs(pres).max()):.2e}")
        # and it must actually get somewhere -- a guard that returns the
        # flat state for every request would pass everything above
        deep = pfs.fold_path(np.deg2rad(179.0), steps=12)[-1]
        assert float(np.rad2deg(np.abs(deep).max())) > 150.0, (
            f"{name}: asked to fold to 179 deg, reached only "
            f"{float(np.rad2deg(np.abs(deep).max())):.1f}")

    print("RESULT: OK  crease.rigid")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
