# Approximating a curved surface with a folded sheet.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy`.
#
# THE CONSTRAINT THIS WHOLE MODULE LIVES UNDER.  A flat sheet has
# Gaussian curvature K = 0, and by Gauss's *Theorema Egregium* K is
# intrinsic: bending and folding cannot change it.  So no folded sheet
# is ever isometric to a sphere, or to any surface with K != 0.  This
# module therefore does NOT claim to fold a surface; it APPROXIMATES
# one, and it reports how badly.  Both numbers it reports exist because
# of that theorem:
#
#   fit error   how far the folded form sits from the target
#   sheet size  how much bigger the flat sheet is than the target
#
# The second is not a curiosity.  The surplus area is the whole
# mechanism: pleats store material, and unfolding gives it back.  A
# corrugation that claims a flat sheet the same size as the target
# surface has got something wrong.
#
# WHY MINIMAL SURFACES ARE THE FAVOURABLE CASE.  K <= 0 everywhere means
# angle EXCESS at every point, and pleating naturally adds material --
# so a saddle is what this technique is good at, and a sphere is what it
# is bad at.  The same asymmetry drives hyperbolic crochet.
#
# HOW IT WORKS, and what is honest about it.  The target is sampled on a
# grid; alternate rows are displaced along the surface normal, which
# turns the smooth surface into a pleated one whose ridges follow the
# grid.  The pleated form is then FLATTENED by relaxing it in the plane
# under its own edge lengths -- and the residual of that relaxation is
# not a nuisance, it is the measurement: the amount by which the sheet
# refuses to lie flat is exactly the Gaussian curvature the corrugation
# failed to absorb.
#
# References:
#   L. H. Dudte, E. Vouga, T. Tachi, L. Mahadevan, "Programming
#       curvature using origami tessellations," Nature Materials 15,
#       2016 -- generalised Miura fitting to a target surface.
#   T. Tachi, "Freeform Variations of Origami," J. Geometry and
#       Graphics 14(2), 2010; and "Designing Freeform Origami
#       Tessellations by Generalizing Resch's Patterns," ASME JMD 135,
#       2013.
#   S. J. P. Callens, A. A. Zadpoor, "From flat sheets to curved
#       geometries: Origami and kirigami approaches," Materials Today,
#       2018 -- the developability constraint and the Gauss-map argument.
#   C. F. Gauss, "Disquisitiones generales circa superficies curvas,"
#       1827 -- the Theorema Egregium itself.

import numpy as np

from .fold_io import BOUNDARY, MOUNTAIN, VALLEY, Frame


class CorrugateError(ValueError):
    """The target cannot be corrugated as asked."""


#: The built-in targets.  Chosen to span the sign of K, because that is
#: what decides whether this technique works: the saddles are the
#: favourable case and the sphere is the adversarial one, and a tool
#: that only ever showed the favourable case would be misleading.
TARGETS = ("HYPAR", "SCHERK", "SPHERE", "CATENOID", "PLANE")


def sample_target(kind, nu=16, nv=16, size=2.0, depth=0.6):
    """Sample a target surface on an (nu+1) x (nv+1) grid.

    Returns an array of shape (nu+1, nv+1, 3).
    """
    u = np.linspace(-1.0, 1.0, nu + 1)
    v = np.linspace(-1.0, 1.0, nv + 1)
    return sample_target_uv(kind, *np.meshgrid(u, v, indexing='ij'),
                            size=size, depth=depth)


def sample_target_uv(kind, U, V, size=2.0, depth=0.6):
    """Evaluate a target at arbitrary (u, v) in [-1, 1]^2.

    Split out from `sample_target` so the fit error can be measured
    BETWEEN the samples -- see `fit`, where that turns out to be the
    only place the error is not zero by construction.
    """
    kind = str(kind).upper()
    if kind not in TARGETS:
        raise CorrugateError(f"unknown target {kind!r}; expected one of "
                             f"{', '.join(TARGETS)}")
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    h = size * 0.5

    if kind == "PLANE":
        Z = np.zeros_like(U)
        X, Y = U * h, V * h
    elif kind == "HYPAR":
        # z = x^2 - y^2: K < 0 everywhere, the classical saddle
        X, Y = U * h, V * h
        Z = depth * (U ** 2 - V ** 2)
    elif kind == "SCHERK":
        # Scherk's first surface, z = ln(cos y / cos x), clipped away
        # from its asymptotes.  A minimal surface, so K <= 0.
        a = 0.85 * (np.pi / 2)
        X, Y = U * h, V * h
        Z = depth * np.log(np.clip(np.cos(V * a), 1e-3, None) /
                           np.clip(np.cos(U * a), 1e-3, None))
    elif kind == "SPHERE":
        # A spherical cap: K > 0, the case the theorem forbids doing
        # well.  Included precisely so the fit error can show it.
        X, Y = U * h, V * h
        r2 = np.clip(U ** 2 + V ** 2, 0.0, 1.0)
        Z = depth * np.sqrt(1.0 - r2)
    else:                                   # CATENOID
        # A catenoid strip: minimal, K < 0.
        th = V * np.pi
        c = np.cosh(U * 1.2)
        X = h * c * np.cos(th) / np.cosh(1.2)
        Y = h * c * np.sin(th) / np.cosh(1.2)
        Z = depth * U * 1.2
    return np.stack([X, Y, Z], axis=-1)


def _grid_normals(P):
    """Unit normals at grid samples, from central differences."""
    du = np.gradient(P, axis=0)
    dv = np.gradient(P, axis=1)
    n = np.cross(du, dv)
    L = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(L, 1e-12)


def corrugate(P, amplitude=0.12, axis=0):
    """Pleat a sampled surface along `axis`.

    Alternate grid lines are pushed out along the surface normal, which
    replaces the smooth surface with a zigzag one that touches it on
    every line.  The pleat is what stores the extra material, and the
    amplitude is how much it can store -- which is why a bigger
    amplitude fits a more strongly curved target and costs a bigger
    sheet.

    Returns `(verts, faces, assignment, edges)` for the pleated form.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 3 or P.shape[2] != 3:
        raise CorrugateError("target must be an (nu+1, nv+1, 3) grid")
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    if nu < 1 or nv < 1:
        raise CorrugateError("the grid needs at least one cell each way")

    N = _grid_normals(P)
    sign = np.ones(P.shape[:2])
    if axis == 0:
        sign *= (-1.0) ** np.arange(P.shape[0])[:, None]
    else:
        sign *= (-1.0) ** np.arange(P.shape[1])[None, :]
    V = P + amplitude * sign[..., None] * N

    def vid(i, j):
        return i * (nv + 1) + j

    verts = V.reshape(-1, 3)
    faces, edges, assign = [], [], []
    seen = {}

    def edge(a, b, kind):
        key = (min(a, b), max(a, b))
        if key in seen:
            if kind == BOUNDARY:
                assign[seen[key]] = BOUNDARY
            return
        seen[key] = len(edges)
        edges.append(key)
        assign.append(kind)

    for i in range(nu):
        for j in range(nv):
            a, b = vid(i, j), vid(i, j + 1)
            c, d = vid(i + 1, j + 1), vid(i + 1, j)
            # Triangulate every cell: the pleated form is not developable
            # and a quad panel would have to be non-planar, which no
            # folded sheet can have.
            faces.append([a, b, c])
            faces.append([a, c, d])
            edge(a, c, MOUNTAIN if (i + j) % 2 else VALLEY)

    # Ridge lines carry the pleat, so their assignment alternates with
    # the pleat itself; the cross lines are the panel boundaries.
    for i in range(nu + 1):
        for j in range(nv + 1):
            if j < nv:
                rim = i in (0, nu)
                edge(vid(i, j), vid(i, j + 1),
                     BOUNDARY if rim else (MOUNTAIN if i % 2 else VALLEY))
            if i < nu:
                rim = j in (0, nv)
                edge(vid(i, j), vid(i + 1, j),
                     BOUNDARY if rim else (VALLEY if i % 2 else MOUNTAIN))

    return (verts, faces,
            np.array(assign, dtype="<U1"),
            np.array(edges, dtype=np.int64).reshape(-1, 2))


def angle_defects(verts, faces):
    """2*pi minus the angle sum at every vertex.

    Zero at an interior vertex means the cone there opens out flat, so
    the surface can be developed through it.  This is the discrete
    Gaussian curvature, and it is the quantity the Theorema Egregium
    says a flat sheet cannot create.
    """
    V = np.asarray(verts, dtype=float)
    total = np.zeros(len(V))
    F = np.asarray(faces, dtype=np.int64)
    for t in range(3):
        i = F[:, t]
        j = F[:, (t + 1) % 3]
        k = F[:, (t + 2) % 3]
        u = V[j] - V[i]
        w = V[k] - V[i]
        cu = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        cw = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
        ang = np.arccos(np.clip(np.einsum('ij,ij->i', cu, cw), -1.0, 1.0))
        np.add.at(total, i, ang)
    return 2.0 * np.pi - total


def make_developable(verts, faces, target, interior, rate=0.35, pull=0.05,
                     iters=200):
    """Nudge a pleated mesh toward developability, without leaving the target.

    OFF BY DEFAULT, AND MEASURED TO BE WORSE.  This was written to fix
    the residual described below and does not: on a 16x16 grid at
    amplitude 0.12, turning it on takes the hyperbolic paraboloid from
    0.42 flattening residual and 0.006 fit to 2.04 and 0.166, and the
    same happens on Scherk and the sphere.  The normal-direction nudge
    is too crude -- moving a vertex along its own normal changes the
    defect at all its neighbours too, so the iteration fights itself.
    It is kept, off, with these numbers attached, so the next attempt
    starts from "this heuristic is known not to work" rather than
    rediscovering it.  A real fix is Dudte et al.'s constrained
    optimisation over the panel geometry, which the plan flags as the
    hardest solver in it.

    WHY THIS STEP EXISTS.  Simply displacing grid samples along the
    surface normal makes a bumpy mesh, not a foldable one: every
    interior vertex keeps an angle defect, so the result has Gaussian
    curvature and no flat sheet can be folded into it.  Measured before
    this step was added, a corrugated hyperbolic paraboloid flattened
    with 9.4% RMS edge error and stayed there however long the
    flattener ran -- the residual was real, not unconverged.

    So each interior vertex is moved along its own normal to reduce its
    defect (raising a cone vertex toward its ring's plane flattens it),
    with a weak spring back to the target so the surface does not simply
    dissolve into a plane.  That trade is the honest one: the tighter
    the developability, the further from the target, and both numbers
    are reported.

    This is Dudte et al.'s idea in miniature -- they solve the same
    trade as a constrained optimisation over a generalised Miura.
    """
    V = np.asarray(verts, dtype=float).copy()
    T = np.asarray(target, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    interior = np.asarray(interior, dtype=bool)

    # Vertex normals, area-weighted, recomputed as the shape moves.
    for _ in range(iters):
        d = angle_defects(V, F)
        d = np.where(interior, d, 0.0)
        n = np.zeros_like(V)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        for t in range(3):
            np.add.at(n, F[:, t], fn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        scale = float(np.mean(np.linalg.norm(
            V[F[:, 1]] - V[F[:, 0]], axis=1)))
        V += rate * scale * (d[:, None] / (2.0 * np.pi)) * n
        V += pull * (T - V)
    return V


def flatten(verts, edges, iters=600, seed=None):
    """Lay a 3-D mesh flat, preserving edge lengths as far as possible.

    Plain spring relaxation in the plane, started from the target's own
    (x, y).  The RESIDUAL is the point of the exercise: a sheet that
    flattens with no residual was developable to begin with, and the
    part that will not flatten is the Gaussian curvature the corrugation
    could not absorb.  Reporting it is the honest alternative to
    presenting an approximation as an isometry.

    Returns `(xy, report)`.
    """
    V = np.asarray(verts, dtype=float)
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    L0 = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)
    L0 = np.maximum(L0, 1e-12)

    xy = V[:, :2].copy()
    if seed is not None:
        xy = np.asarray(seed, dtype=float).copy()
    # Nudge any exactly-coincident start apart, or the first step has no
    # direction to push along.
    rng = np.random.default_rng(0)
    xy += 1e-9 * rng.standard_normal(xy.shape)

    deg = np.zeros(len(V))
    for col in (0, 1):
        np.add.at(deg, E[:, col], 1.0)
    deg = np.maximum(deg, 1.0)

    for _ in range(iters):
        d = xy[E[:, 1]] - xy[E[:, 0]]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        corr = (0.5 * (L - L0) / L)[:, None] * d
        step = np.zeros_like(xy)
        np.add.at(step, E[:, 0], corr)
        np.add.at(step, E[:, 1], -corr)
        xy += step / deg[:, None]

    L = np.linalg.norm(xy[E[:, 1]] - xy[E[:, 0]], axis=1)
    err = np.abs(L / L0 - 1.0)
    report = {
        "max_edge_error": float(err.max()),
        "rms_edge_error": float(np.sqrt(np.mean(err ** 2))),
        "sheet_w": float(np.ptp(xy[:, 0])),
        "sheet_h": float(np.ptp(xy[:, 1])),
    }
    return xy, report


def fit(kind="HYPAR", nu=16, nv=16, size=2.0, depth=0.6, amplitude=0.12,
        iters=600, develop=0):
    """Corrugate a target and flatten it; return everything measured.

    Returns `(frame, folded, report)` where `frame` is the FLAT crease
    pattern, `folded` the pleated 3-D positions, and `report` carries
    the numbers the Theorema Egregium forces this operator to admit to.
    """
    P = sample_target(kind, nu=nu, nv=nv, size=size, depth=depth)
    verts, faces, assign, edges = corrugate(P, amplitude=amplitude)

    if develop:
        nu_g, nv_g = P.shape[0] - 1, P.shape[1] - 1
        inside = np.zeros(P.shape[:2], dtype=bool)
        inside[1:-1, 1:-1] = True
        verts = make_developable(verts, faces, verts.copy(),
                                 inside.reshape(-1), iters=develop)
    # SEED THE FLATTENING BY UNROLLING, not by projecting.
    #
    # Projecting the pleated form onto its own xy plane starts every
    # pleat COMPRESSED -- the material hidden in the folds is missing --
    # and the relaxation then has to push the whole sheet outwards
    # against itself.  It does not get there: measured on a pleated
    # PLANE, which is exactly developable and so has a perfect answer
    # available, the projected start converged to 5.0% RMS edge error
    # and stayed there from 6000 iterations to 100000.  It was stuck,
    # not slow.
    #
    # Unrolling instead walks the grid accumulating true 3-D edge
    # lengths, so the pleats arrive already opened out and the sheet
    # starts near its answer rather than folded over on itself.
    G = verts.reshape(P.shape[0], P.shape[1], 3)
    du = np.linalg.norm(np.diff(G, axis=0), axis=2)
    dv = np.linalg.norm(np.diff(G, axis=1), axis=2)
    ys = np.zeros(P.shape[:2])
    xs = np.zeros(P.shape[:2])
    ys[1:, :] = np.cumsum(du, axis=0)
    xs[:, 1:] = np.cumsum(dv, axis=1)
    seed = np.stack([xs.ravel(), ys.ravel()], axis=1)

    xy, rep = flatten(verts, edges, iters=iters, seed=seed)

    # HOW WELL THE PLEATED FORM TRACKS THE SURFACE -- measured BETWEEN
    # the samples, not at them.
    #
    # At a sample the answer is `amplitude` exactly, by construction:
    # every grid vertex was pushed off the surface by precisely that
    # much.  Reporting it looks like a fit error and is really just the
    # amplitude echoed back, identical for every target and unchanged by
    # refining the grid -- which quietly makes the refinement test pass
    # on nothing.  That was the first version of this and it measured
    # nothing at all.
    #
    # The meaningful quantity is the pleat's MID-SURFACE: average the
    # four corners of each cell and the +/- offsets cancel, leaving the
    # chord across the cell.  Its distance from the true surface is the
    # approximation error, it scales with curvature times cell size
    # squared, and it therefore actually falls when the grid is refined.
    target = P.reshape(-1, 3)
    nu_, nv_ = P.shape[0] - 1, P.shape[1] - 1
    G = verts.reshape(nu_ + 1, nv_ + 1, 3)
    mid = 0.25 * (G[:-1, :-1] + G[1:, :-1] + G[:-1, 1:] + G[1:, 1:])
    cu = np.linspace(-1.0, 1.0, nu_ + 1)
    cv = np.linspace(-1.0, 1.0, nv_ + 1)
    cu = 0.5 * (cu[:-1] + cu[1:])
    cv = 0.5 * (cv[:-1] + cv[1:])
    CU, CV = np.meshgrid(cu, cv, indexing='ij')
    truth = sample_target_uv(kind, CU, CV, size=size, depth=depth)
    fit_err = np.linalg.norm(mid.reshape(-1, 3) - truth.reshape(-1, 3),
                             axis=1)

    tgt_w = float(np.ptp(target[:, 0]))
    tgt_h = float(np.ptp(target[:, 1]))
    rep.update({
        "fit_max": float(fit_err.max()),
        "fit_rms": float(np.sqrt(np.mean(fit_err ** 2))),
        "target_w": tgt_w,
        "target_h": tgt_h,
        # The surplus: how much bigger the flat sheet is than the
        # surface it will become.  Always positive for a real
        # corrugation -- the pleats have to come from somewhere.
        "area_ratio": float((rep["sheet_w"] * rep["sheet_h"]) /
                            max(tgt_w * tgt_h, 1e-12)),
    })

    frame = Frame(
        verts=xy,
        edges=edges,
        assignment=assign,
        faces=faces,
        face_orders=None,
        meta={"frame_title": f"{kind.title()} corrugation",
              "corrugation": rep},
    )
    return frame, verts, rep


def report_summary(rep):
    """One line stating what the approximation actually cost."""
    return (f"fit {rep['fit_rms']:.3g} rms / {rep['fit_max']:.3g} max; "
            f"flat sheet {rep['sheet_w']:.2f} x {rep['sheet_h']:.2f} "
            f"for a {rep['target_w']:.2f} x {rep['target_h']:.2f} target "
            f"({rep['area_ratio']:.2f}x the area); "
            f"unfolding residual {rep['max_edge_error']:.2g}")


def _selftest():
    # --- the flattener is exact on something already flat ------------
    P = sample_target("PLANE", nu=6, nv=6)
    verts, faces, assign, edges = corrugate(P, amplitude=0.0)
    _xy, rep = flatten(verts, edges, iters=400)
    assert rep["max_edge_error"] < 1e-6, rep
    assert len(faces) == 6 * 6 * 2, len(faces)

    # --- A PLEATED PLANE MUST FLATTEN EXACTLY: the control -----------
    #
    # Pleating a plane is an isometry -- the pleats are only folds, and
    # the Gaussian curvature is zero before and after -- so a perfect
    # flat pattern EXISTS and any residual at all is the flattener's
    # fault, not the surface's.  This is the one case where the answer
    # is known, which makes it the test that matters.
    #
    # Run at a realistic grid size on purpose.  An earlier version of
    # this check used a 6x6 grid, passed, and hid a flattener that
    # stalled at 5% RMS error on a 14x14 one -- small enough to look
    # converged, large enough to be wrong.
    _fr, _f, rep = fit("PLANE", nu=14, nv=14, amplitude=0.12, iters=1500)
    assert rep["max_edge_error"] < 1e-6, (
        f"a pleated plane is developable and must flatten exactly, so "
        f"this residual is a flattener bug: {rep}")

    # --- the sheet is always BIGGER than the target ------------------
    _fr, _f, rep = fit("HYPAR", nu=12, nv=12, amplitude=0.12)
    assert rep["area_ratio"] > 1.0, (
        f"the pleats have to come from somewhere: {rep}")

    # --- K <= 0 is the favourable case, K > 0 the hard one -----------
    #
    # Not an arbitrary comparison: pleating ADDS material, which is what
    # a saddle's angle excess needs and the opposite of what a sphere's
    # deficit needs.  The numbers should show the asymmetry the theorem
    # predicts.
    errs = {}
    for kind in ("HYPAR", "SCHERK", "SPHERE"):
        _fr, _f, r = fit(kind, nu=14, nv=14, amplitude=0.12, iters=1500)
        errs[kind] = r["max_edge_error"]
    assert errs["SPHERE"] > errs["HYPAR"], (
        f"a spherical cap should be HARDER to corrugate than a saddle "
        f"(positive curvature needs material removed, pleating adds it): "
        f"{errs}")

    # --- the developability projection is known NOT to help ---------
    #
    # Pinned deliberately.  It is off by default because it was measured
    # to make both numbers worse, and a test that records that stops the
    # next person turning it on hopefully.
    base = fit("HYPAR", nu=12, nv=12, amplitude=0.12, iters=800,
               develop=0)[2]
    tried = fit("HYPAR", nu=12, nv=12, amplitude=0.12, iters=800,
                develop=150)[2]
    assert tried["max_edge_error"] > base["max_edge_error"], (
        "the developability nudge has started helping -- if that is real, "
        "re-measure it across targets and change the default, and update "
        "the comment in make_developable")

    # --- finer cells fit better, as a trend --------------------------
    #
    # The plan is explicit that this must not be asserted as strict
    # monotonicity: different cell sizes land in different local minima
    # of a nonconvex fit, so a monotone test would be flaky.  Compare
    # the ends of the sweep instead.
    coarse = fit("HYPAR", nu=6, nv=6, amplitude=0.12, iters=500)[2]
    fine = fit("HYPAR", nu=20, nv=20, amplitude=0.12, iters=500)[2]
    assert fine["fit_rms"] <= coarse["fit_rms"] * 1.05, (
        f"refining the grid should not fit worse: coarse "
        f"{coarse['fit_rms']:.4g}, fine {fine['fit_rms']:.4g}")

    # --- the frame is usable downstream ------------------------------
    fr, folded, rep = fit("HYPAR", nu=8, nv=8)
    assert fr.is_flat, "the emitted crease pattern must be flat"
    assert len(fr.edges) == len(fr.assignment)
    assert set(fr.assignment.tolist()) <= {"M", "V", "B"}
    assert len(folded) == fr.n_verts
    assert "x" in report_summary(rep)

    try:
        sample_target("NOPE")
    except CorrugateError as exc:
        assert "unknown target" in str(exc)
    else:
        raise AssertionError("an unknown target should raise")

    print("RESULT: OK  crease.corrugate")
