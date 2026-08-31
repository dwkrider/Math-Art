# Compliant origami: folding paper that is allowed to bend.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy`.
#
# WHY THIS EXISTS ALONGSIDE `rigid.py`.  The rigid solver treats every
# panel as a flat plate hinged at the creases, and for a Miura that is
# exactly right -- it IS rigid-foldable, and the panels genuinely do not
# bend.  For most of the visually interesting patterns it is wrong, and
# provably so: Demaine, Demaine, Hart, Price and Tachi (2011) prove the
# pleated hypar has NO proper folding with planar facets at all.  It
# folds only because the paper bends between the creases.  A rigid
# solver cannot express that, at any step size or tolerance, because the
# configuration it is looking for does not exist in its model.
#
# THE MODEL is Ghassaei, Demaine and Gershenfeld's -- the one behind
# Origami Simulator -- because it is explicit, it needs no global
# stiffness matrix, and it is the method whose output most people have
# actually seen.  The sheet becomes a pin-jointed truss of the
# triangulated crease pattern, carrying three kinds of linear-elastic
# constraint:
#
#     axial    each edge resists changing length      k_axial = EA / l0
#     crease   each interior edge pulls its two faces toward a target
#              fold angle                             k = l0 * k_fold
#     face     each triangle corner resists shear     k_face
#
# and it is integrated explicitly with a small step.
#
# TWO CORRECTIONS TO THE PAPER, both recorded in this repo's conversion
# of it and both load bearing here:
#
#   * The update is labelled "forward Euler" but as written it is
#     SEMI-IMPLICIT (symplectic) Euler -- the new velocity advances the
#     position.  Implementing the label instead of the equations gives a
#     scheme that pumps energy and blows up; implementing the equations
#     gives one that is stable at the paper's own step size.
#   * The printed face-constraint partial derivatives are the reciprocal
#     of what the force expression needs.
#
# THE STIFFNESS ORDERING IS THE WHOLE TRICK.  k_axial >> k_crease means
# the paper stretches almost not at all while the creases do the work,
# which is what makes an inextensible-sheet model out of a spring
# network.  Push k_crease too close to k_axial and the sheet stretches
# to reach the target angle -- and then it is not paper.
#
# References:
#   A. Ghassaei, E. D. Demaine, N. Gershenfeld, "Fast, Interactive
#       Origami Simulation using GPU Computation," 7OSME / Origami^7,
#       2018 -- the constraint formulation and the explicit solver.
#   M. Schenk, S. D. Guest, "Origami Folding: A Structural Engineering
#       Approach," Origami^5, 2011 -- the bar-and-hinge model the crease
#       stiffnesses follow.
#   E. D. Demaine, M. L. Demaine, V. Hart, G. N. Price, T. Tachi,
#       "(Non)existence of Pleated Folds: How Paper Folds Between
#       Creases," Graphs and Combinatorics 27(3), 2011 -- why a rigid
#       solver cannot do this job.

import numpy as np

from .fold_io import BOUNDARY, FLAT, MOUNTAIN, VALLEY


class CompliantFailure(RuntimeError):
    """The simulation did not produce a usable state."""


def _unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


class CompliantFolder:
    """Fold a triangulated crease pattern as a compliant spring network.

    `frame` must carry triangular faces.  Quads are held flat by the
    rigid solver and would be held flat here too -- there is no bending
    degree of freedom inside a triangle, so a quad panel cannot bend
    unless it is split.
    """

    #: Defaults TUNED BY MEASUREMENT, not taken from the paper, because
    #: the paper's application drives its own interactive slider while
    #: this one has to produce a settled shape unattended.  Sweeping
    #: them on the pleated hypar -- the pattern that needs bending most
    #: -- gives, at fold amount 0.8:
    #:
    #:     k_facet 0.70  ->  strain 1.1%, fold depth 0.118
    #:     k_facet 0.15  ->  strain 1.3%, fold depth 0.225
    #:     k_facet 0.03  ->  strain 1.5%, fold depth 0.327
    #:
    #: A SOFT FACET IS THE POINT.  With stiff facets the sheet cannot
    #: bend, so the only way left to reach the target angles is to
    #: stretch -- and a stretching sheet is not paper.  Softening the
    #: facet creases lets the panels curve instead, which triples the
    #: fold depth at essentially unchanged strain.  k_axial stays far
    #: above everything else so the edges keep their length.
    def __init__(self, frame, k_axial=200.0, k_fold=0.7, k_facet=0.03,
                 k_face=0.05, damping=0.45, dt_safety=0.4):
        if frame.faces is None:
            raise CompliantFailure(
                "no faces: call graph.build_faces before folding")
        bad = [f for f in frame.faces if len(f) != 3]
        if bad:
            raise CompliantFailure(
                f"{len(bad)} face(s) are not triangles. The compliant "
                f"model bends paper BETWEEN creases, and a triangle has "
                f"no interior bending freedom -- triangulate the pattern "
                f"first (the importer's Triangulate Cells option does "
                f"this, adding unassigned diagonals)")

        self.frame = frame
        V = np.asarray(frame.verts, dtype=float)
        if V.shape[1] == 2:
            V = np.hstack([V, np.zeros((len(V), 1))])
        self.rest = V.copy()
        self.pos = V.copy()
        self.vel = np.zeros_like(V)
        self.faces = np.asarray([list(f) for f in frame.faces], dtype=np.int64)
        self.edges = np.asarray(frame.edges, dtype=np.int64).reshape(-1, 2)
        self.assign = (np.asarray(frame.assignment, dtype="<U1")
                       if frame.assignment is not None
                       else np.array([""] * len(self.edges), dtype="<U1"))
        # PER-CREASE TARGET ANGLES, when the pattern knows them.
        #
        # Without these every mountain is driven to -1 radian and every
        # valley to +1, which is fine for "fold this pattern up" and
        # WRONG for "fold this pattern into that specific shape".  A
        # corrugation fitted to a surface knows the angle each crease
        # must reach -- it was measured off the fitted 3-D form -- and
        # ignoring it means folding a different object: measured on a
        # corrugated plane, the intended form ranges -100 to +100
        # degrees while the uniform target drives everything to 57.3.
        self.target_angle = (np.asarray(frame.fold_angle, dtype=float)
                             if frame.fold_angle is not None else None)

        self.l0 = np.linalg.norm(self.rest[self.edges[:, 1]] -
                                 self.rest[self.edges[:, 0]], axis=1)
        self.l0 = np.maximum(self.l0, 1e-12)
        # k_axial = EA/l0.  Only the RATIO of the stiffnesses matters
        # for a quasi-static fold, so EA is folded into k_axial and each
        # beam gets its own value from its own rest length -- a short
        # beam is stiffer, which is what keeps a fine triangulation from
        # behaving like a slack one.
        self.k_ax = k_axial / self.l0

        self._build_creases(k_fold, k_facet)

        self.k_face = float(k_face)
        self.damping = float(damping)
        # Stability bound (Eqs. 7-8), TIMES A SAFETY FACTOR.  The bound
        # accounts for the axial springs alone; the crease and face
        # springs add stiffness on top, and running exactly at the bound
        # is unstable in a way that is easy to misread -- energy creeps
        # up, so a LONGER run comes out worse than a short one.  That was
        # measured here: at the bare bound, 6000 steps left twice the
        # strain of 2400.
        omega = float(np.sqrt(self.k_ax.max()))
        self.dt = (dt_safety / (2.0 * np.pi * omega)) if omega > 0 else 1e-3

        self._face_rest = self._corner_angles(self.rest)

    # -- topology ---------------------------------------------------
    def _build_creases(self, k_fold, k_facet):
        """Per interior edge: its two faces, their apexes, and its target.

        A crease needs FOUR nodes, not two -- the two ends and the two
        opposite corners -- because the fold angle is a function of all
        four, and so is the force it applies.
        """
        edge_faces = {}
        for fi, f in enumerate(self.faces):
            for t in range(3):
                a, b = int(f[t]), int(f[(t + 1) % 3])
                edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)

        self.cr_edge, self.cr_face, self.cr_apex = [], [], []
        self.cr_k, self.cr_target = [], []
        for k, (a, b) in enumerate(self.edges):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            fl = edge_faces.get(key, [])
            code = str(self.assign[k])
            if len(fl) != 2 or code == BOUNDARY:
                continue
            apex = []
            for fi in fl:
                rest = [int(v) for v in self.faces[fi] if v not in key]
                if len(rest) != 1:
                    break
                apex.append(rest[0])
            if len(apex) != 2:
                continue
            self.cr_edge.append(key)
            self.cr_face.append((fl[0], fl[1]))
            self.cr_apex.append((apex[0], apex[1]))
            # Stiffness and target BOTH follow the assignment.  An
            # unassigned crease is a place the paper bends rather than a
            # fold anybody designed -- it gets the facet stiffness and a
            # target of zero, so it resists bending but is not driven.
            want = None
            if self.target_angle is not None and k < len(self.target_angle):
                a = float(self.target_angle[k])
                want = None if a != a else a            # NaN means unknown
            if code in (MOUNTAIN, VALLEY):
                self.cr_k.append(self.l0[k] * k_fold)
                self.cr_target.append(
                    want if want is not None
                    else (-1.0 if code == MOUNTAIN else 1.0))
            else:                       # FLAT, UNASSIGNED, or unlabelled
                self.cr_k.append(self.l0[k] * k_facet)
                # An unassigned crease with a KNOWN angle is still known:
                # the corrugation's added diagonals are exactly that.
                self.cr_target.append(want if want is not None else 0.0)

        self.cr_edge = np.array(self.cr_edge, dtype=np.int64).reshape(-1, 2)
        self.cr_apex = np.array(self.cr_apex, dtype=np.int64).reshape(-1, 2)
        self.cr_k = np.array(self.cr_k, dtype=float)
        self.cr_target = np.array(self.cr_target, dtype=float)

    # -- geometry ---------------------------------------------------
    def fold_angles(self, P=None):
        """Signed fold angle of every crease, valley positive."""
        P = self.pos if P is None else P
        if not len(self.cr_edge):
            return np.zeros(0)
        p3 = P[self.cr_edge[:, 0]]
        p4 = P[self.cr_edge[:, 1]]
        p1 = P[self.cr_apex[:, 0]]
        p2 = P[self.cr_apex[:, 1]]
        e = _unit(p4 - p3)
        n1 = np.cross(p4 - p3, p1 - p3)
        n2 = np.cross(p2 - p3, p4 - p3)
        n1, n2 = _unit(n1), _unit(n2)
        s = np.einsum('ij,ij->i', np.cross(n1, n2), e)
        c = np.einsum('ij,ij->i', n1, n2)
        return np.arctan2(s, c)

    def _corner_angles(self, P):
        """Interior angle at each corner of each triangle."""
        out = np.zeros((len(self.faces), 3))
        for t in range(3):
            i = self.faces[:, t]
            j = self.faces[:, (t + 1) % 3]
            k = self.faces[:, (t + 2) % 3]
            u = _unit(P[j] - P[i])
            v = _unit(P[k] - P[i])
            out[:, t] = np.arccos(np.clip(
                np.einsum('ij,ij->i', u, v), -1.0, 1.0))
        return out

    # -- forces -----------------------------------------------------
    def _axial(self, P, F):
        a, b = self.edges[:, 0], self.edges[:, 1]
        d = P[b] - P[a]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        dirn = d / L[:, None]
        f = (self.k_ax * (L - self.l0))[:, None] * dirn
        np.add.at(F, a, f)
        np.add.at(F, b, -f)

    def _crease(self, P, F, drive):
        """Torsional springs on the creases (Eqs. 2-6)."""
        if not len(self.cr_edge):
            return
        i3, i4 = self.cr_edge[:, 0], self.cr_edge[:, 1]
        i1, i2 = self.cr_apex[:, 0], self.cr_apex[:, 1]
        p3, p4, p1, p2 = P[i3], P[i4], P[i1], P[i2]

        theta = self.fold_angles(P)
        err = self.cr_k * (theta - drive * self.cr_target)

        e = p4 - p3
        eL = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        eu = e / eL[:, None]
        n1 = _unit(np.cross(p4 - p3, p1 - p3))
        n2 = _unit(np.cross(p2 - p3, p4 - p3))

        # Lever arms: the perpendicular distance from each apex to the
        # crease LINE, not to its nearer endpoint.  Using the endpoint
        # distance is a plausible-looking mistake that makes obtuse
        # triangles rotate the wrong amount.
        def lever(p):
            w = p - p3
            along = np.einsum('ij,ij->i', w, eu)[:, None] * eu
            return np.maximum(np.linalg.norm(w - along, axis=1), 1e-9)

        # SIGN.  `n/h` is the magnitude of the apex gradient, but whether
        # it is +grad(theta) or -grad(theta) depends on how theta was
        # oriented -- and this module measures theta with `fold_angles`,
        # valley positive, which is the codebase's convention rather
        # than the paper's.  Determined against a central difference of
        # the crease energy, not argued from the figure: with the
        # opposite sign every force here comes out exactly negated, so
        # the sheet drives itself away from the target angle and folds
        # inside out.  The self-test pins it.
        h1, h2 = lever(p1), lever(p2)
        d1 = -n1 / h1[:, None]
        d2 = -n2 / h2[:, None]

        # The endpoint gradients, in the BARYCENTRIC form rather than
        # the paper's cotangent one.  The two are equivalent, but this
        # one is checkable by inspection: the coefficients of n1/h1
        # across the four nodes sum to 1 + w1 + u1 = 0 exactly, so the
        # gradient is translation invariant by construction -- and a
        # dihedral gradient that is not translation invariant applies a
        # net force to a rigid body, which is the failure the cotangent
        # version was silently producing here.
        #
        # (Grinspun et al., "Discrete Shells", 2003, is the usual
        # source; Ghassaei et al. Eqs. 5-6 are the same quantity.)
        E2 = np.maximum(np.einsum('ij,ij->i', e, e), 1e-18)
        w1 = np.einsum('ij,ij->i', p1 - p4, e) / E2
        w2 = np.einsum('ij,ij->i', p2 - p4, e) / E2
        u1 = -np.einsum('ij,ij->i', p1 - p3, e) / E2
        u2 = -np.einsum('ij,ij->i', p2 - p3, e) / E2

        g1 = d1
        g2 = d2
        g3 = w1[:, None] * d1 + w2[:, None] * d2
        g4 = u1[:, None] * d1 + u2[:, None] * d2

        np.add.at(F, i1, -(err[:, None] * g1))
        np.add.at(F, i2, -(err[:, None] * g2))
        np.add.at(F, i3, -(err[:, None] * g3))
        np.add.at(F, i4, -(err[:, None] * g4))

    def _face(self, P, F):
        """Corner-angle springs: cheap shear resistance inside a face."""
        if self.k_face <= 0.0:
            return
        ang = self._corner_angles(P)
        for t in range(3):
            i = self.faces[:, t]
            j = self.faces[:, (t + 1) % 3]
            k = self.faces[:, (t + 2) % 3]
            err = self.k_face * (ang[:, t] - self._face_rest[:, t])
            u = P[j] - P[i]
            v = P[k] - P[i]
            lu = np.maximum(np.linalg.norm(u, axis=1), 1e-12)
            lv = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
            n = _unit(np.cross(u, v))
            # d(angle)/dp for the two arms: perpendicular to each arm,
            # in the face plane, scaled by 1/|arm|.
            gj = np.cross(n, u / lu[:, None]) / lu[:, None]
            gk = -np.cross(n, v / lv[:, None]) / lv[:, None]
            np.add.at(F, j, -(err[:, None] * gj))
            np.add.at(F, k, -(err[:, None] * gk))
            np.add.at(F, i, err[:, None] * (gj + gk))

    # -- integration ------------------------------------------------
    def step(self, drive=1.0, dt=None):
        """One semi-implicit Euler step.

        Semi-implicit, NOT the "forward Euler" the paper's text names:
        the position is advanced by the NEW velocity.  Forward Euler on
        an undamped spring network gains energy every step and diverges;
        this does not.
        """
        dt = self.dt if dt is None else dt
        F = np.zeros_like(self.pos)
        self._axial(self.pos, F)
        self._crease(self.pos, F, drive)
        self._face(self.pos, F)
        F -= self.damping * self.vel          # viscous drag toward rest
        self.vel += F * dt                    # unit nodal mass
        self.pos += self.vel * dt
        return float(np.abs(F).max())

    def run(self, drive=1.0, steps=12000, ramp=0.35, tol=1e-6):
        """Fold to `drive` (0 flat, 1 the full target angles).

        The drive is RAMPED rather than applied at once.  Slamming every
        crease to its target on step one is the reliable way to turn a
        sheet inside out: the network has no time to sort out which way
        each panel should swing, and it settles into a tangle that is a
        perfectly good energy minimum and not the fold anyone wanted.
        """
        n_ramp = max(1, int(steps * ramp))
        last = 0.0
        for i in range(steps):
            d = drive * min(1.0, (i + 1) / n_ramp)
            last = self.step(d)
            if i > n_ramp and last < tol and \
                    float(np.abs(self.vel).max()) < tol:
                break
        if not np.isfinite(self.pos).all():
            raise CompliantFailure(
                "the simulation diverged. The usual cause is stiffness: "
                "k_axial must stay far above k_fold, or the sheet "
                "stretches instead of folding")
        return self.pos

    # -- readouts ---------------------------------------------------
    def edge_strain(self, P=None):
        """Signed axial strain per edge: (l - l0) / l0.

        This is the quantity worth colouring.  Paper does not stretch,
        so a well-behaved fold shows strain near zero everywhere, and a
        band of high strain is the model telling you where the pattern
        is fighting itself -- a jam, an over-constrained vertex, or a
        target angle the sheet cannot reach.
        """
        P = self.pos if P is None else P
        L = np.linalg.norm(P[self.edges[:, 1]] - P[self.edges[:, 0]], axis=1)
        return (L - self.l0) / self.l0

    def vertex_strain(self, P=None):
        """Edge strain averaged onto vertices, for a per-vertex colour."""
        s = np.abs(self.edge_strain(P))
        acc = np.zeros(len(self.pos))
        cnt = np.zeros(len(self.pos))
        for col in (0, 1):
            np.add.at(acc, self.edges[:, col], s)
            np.add.at(cnt, self.edges[:, col], 1.0)
        return acc / np.maximum(cnt, 1.0)


def fold(frame, drive=1.0, steps=2000, **kw):
    """Convenience: build a folder, run it, return (positions, folder)."""
    cf = CompliantFolder(frame, **kw)
    return cf.run(drive=drive, steps=steps), cf


def _selftest():
    from . import patterns
    from .graph import build_faces, triangulate
    from .fold_io import UNASSIGNED

    def tri_frame(name, **kw):
        fr = patterns.build(name, **kw)
        fr.faces = build_faces(fr.verts, fr.edges)
        if any(len(f) != 3 for f in fr.faces):
            tris, diags = triangulate(fr.verts, fr.faces)
            fr.faces = tris
            if diags:
                fr.edges = np.vstack([fr.edges,
                                      np.array(diags, dtype=np.int64)])
                fr.assignment = np.concatenate(
                    [fr.assignment,
                     np.array([UNASSIGNED] * len(diags), dtype="<U1")])
        return fr

    # --- the fold-angle derivative, against finite differences -------
    #
    # Equations 3-6 are the part of this that cannot be checked by
    # looking at the result: a wrong lever arm or a swapped cotangent
    # still folds something, just not the right shape.  So compare the
    # analytic gradient with a central difference of the angle itself.
    fr = tri_frame('MIURA', rows=2, cols=2)
    cf = CompliantFolder(fr)
    rng = np.random.default_rng(0)
    cf.pos = cf.rest + 0.05 * rng.standard_normal(cf.rest.shape)
    P0 = cf.pos.copy()

    F = np.zeros_like(P0)
    cf._crease(P0, F, drive=0.0)          # target 0 => force = -k*theta*grad
    h = 1e-6
    worst = 0.0
    for v in range(len(P0)):
        for ax in range(3):
            Pp, Pm = P0.copy(), P0.copy()
            Pp[v, ax] += h
            Pm[v, ax] -= h
            # energy 0.5*k*theta^2 summed over creases
            ep = 0.5 * float(np.sum(cf.cr_k * cf.fold_angles(Pp) ** 2))
            em = 0.5 * float(np.sum(cf.cr_k * cf.fold_angles(Pm) ** 2))
            fd = -(ep - em) / (2 * h)
            worst = max(worst, abs(fd - F[v, ax]))
    scale = max(1.0, float(np.abs(F).max()))
    assert worst / scale < 2e-3, (
        f"crease force disagrees with the energy gradient by {worst:.2e} "
        f"(scale {scale:.2e}) -- Eqs. 3-6 are wrong")

    # --- a Miura folds, and the paper does not stretch ---------------
    fr = tri_frame('MIURA', rows=4, cols=6)
    P, cf = fold(fr, drive=0.55, steps=6000)
    z = float(np.ptp(P[:, 2]))
    assert z > 0.15, f"the Miura did not fold: z-extent {z:.3f}"
    st = float(np.abs(cf.edge_strain()).max())
    assert st < 0.03, (
        f"the Miura stretched by {st:.3f}. It is rigid-foldable, so the "
        f"panels should barely work at all -- this means k_axial is not "
        f"dominating k_fold")

    # --- THE POINT OF THIS MODULE: the hypar folds ------------------
    #
    # It has no proper folding with planar facets (Demaine et al. 2011,
    # Cor. 14), so this is the case the rigid solver cannot do by
    # construction, and the reason the compliant model exists.
    fr = tri_frame('HYPAR', rings=6, sides=4)
    P, cf = fold(fr, drive=0.8, steps=15000)
    r = np.linalg.norm(fr.verts[:, :2], axis=1)
    rim = [i for i in range(len(P)) if abs(r[i] - r.max()) < 1e-9]
    ang = np.degrees(np.arctan2(fr.verts[rim, 1], fr.verts[rim, 0])) % 360
    zc = [float(z) for _a, z in sorted(zip(ang, P[rim, 2]))]
    assert len(zc) == 4, zc
    assert (zc[0] - zc[1]) * (zc[1] - zc[2]) < 0 and \
           (zc[0] - zc[1]) * (zc[2] - zc[3]) > 0, (
        f"the compliant hypar rim does not alternate up/down: "
        f"{[round(v, 3) for v in zc]}")

    # and the bending it needs shows up as strain, which is exactly
    # what P3b colours
    s = cf.edge_strain()
    assert float(np.abs(s).max()) > 1e-4, (
        "a hypar that folds with no strain anywhere would mean the "
        "sheet is not bending, which is the thing it cannot do")

    # --- an untriangulated pattern is refused, with the remedy -------
    q = patterns.build('MIURA', rows=2, cols=2)
    q.faces = build_faces(q.verts, q.edges)
    try:
        CompliantFolder(q)
    except CompliantFailure as exc:
        assert "triangulate" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("quad faces should be refused")

    print("RESULT: OK  crease.compliant")
