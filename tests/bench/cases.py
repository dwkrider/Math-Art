# Benchmark cases: fixed, deterministic inputs for each solver in scope.
#
# Every case is a function `fn(config: dict) -> dict` returning
#     {"metrics": {...}, "trace": [...], "time_s": float, ...}
# The `config` dict carries opt-in solver flags (empty = baseline: the
# solvers are called exactly as current callers call them).  All inputs
# are fixed seeds / fixed meshes -- same command, same numbers (wall
# time excepted).
#
# Convergence traces are collected by exact chunking where re-entering
# the solver is provably identical to continuing it (minimize_area
# recomputes all state from V each outer iteration; relax_knot's only
# cross-iteration state, `target`, is re-derived to the same value).
# Solvers where re-entry differs (canonicalize/biscribe renormalise on
# entry; crochet has an annealing schedule) are traced via an opt-in
# `trace` callback or reported endpoint-only.
import math
import time

import numpy as np

from . import _boot  # noqa: F401  (installs the synthetic math_art package)
from . import metrics as M


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _timer():
    return time.perf_counter()


def catenoid_area_analytic(h=0.5, R=1.0):
    """Area of the minimal catenoid between coaxial radius-R rings at
    z = +-h: r(z) = c cosh(z/c) with c cosh(h/c) = R, taking the deep
    (stable) root; A = 2 pi c (h + (c/2) sinh(2h/c))."""
    lo, hi = 0.5, 1.0            # deep root for h=0.5, R=1 is c ~ 0.848
    f = lambda c: c * math.cosh(h / c) - R
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    c = 0.5 * (lo + hi)
    return 2.0 * math.pi * c * (h + 0.5 * c * math.sinh(2.0 * h / c)), c


def cylinder_grid(nring, nrow, h=0.5, R=1.0):
    th = np.linspace(0.0, 2 * np.pi, nring, endpoint=False)
    zs = np.linspace(-h, h, nrow)
    V = np.array([[R * math.cos(t), R * math.sin(t), z]
                  for z in zs for t in th])
    T = []
    for j in range(nrow - 1):
        for i in range(nring):
            a0 = j * nring + i
            b0 = j * nring + (i + 1) % nring
            T.append([a0, b0, a0 + nring])
            T.append([b0, b0 + nring, a0 + nring])
    T = np.array(T, dtype=np.int64)
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:nring] = True
    fixed[-nring:] = True
    return V, T, fixed


def _plateau_mesh_metrics(V, T, fixed):
    out = {"area": M.mesh_area(V, T)}
    out.update(M.tri_quality(V, T))
    out.update(M.cotan_clamp_fraction(V, T))
    out.update(M.mean_curvature_residual(V, T, free=~fixed))
    out["edge_cv"] = M.edge_len_cv(V, T)
    out["valence_rms_dev"] = M.valence_rms_dev(T, len(V), boundary=fixed)
    return out


def _minimize_area_traced(V, T, fixed, outer_iters, kwargs):
    """Run plateau.minimize_area one outer iteration at a time (exact:
    each outer iteration recomputes weights from V), recording area vs
    iteration and vs elapsed wall seconds, with the solver's own
    convergence break replicated.

    With groom_every=g, one solver call grooms before iterations
    g, 2g, ...; chunking per-iteration would reset that counter and
    silently disable grooming, so here the groom cycle is invoked
    explicitly on the same schedule -- provably the same sequence of
    operations as a single call.  (An artifact generated before this
    replay existed reported a groomed config with grooming silently
    off; the returned `effective` dict now records what actually ran,
    and is written into every result as provenance.)

    Returns (trace, effective)."""
    from math_art.minsurf import plateau
    import inspect
    kwargs = dict(kwargs)
    groom_every = int(kwargs.pop("groom_every", 0) or 0)
    groom_smooth = kwargs.pop("groom_smooth", 0.25)
    sig = inspect.signature(plateau.minimize_area)
    unknown = [k for k in kwargs if k not in sig.parameters]
    if unknown:
        raise ValueError(f"plateau_kwargs not accepted by minimize_area: "
                         f"{unknown}")
    default_mode = sig.parameters["cotan_mode"].default
    effective = {
        "cotan_mode": kwargs.get("cotan_mode", default_mode),
        "groom_every": groom_every,
        "grooms_run": 0,
    }
    if groom_every:
        from math_art.solver import groom as _sg
    trace = [{"iter": 0, "t": 0.0, "E": M.mesh_area(V, T)}]
    t0 = _timer()
    for it in range(1, outer_iters + 1):
        V0 = V.copy()
        if groom_every and (it - 1) and (it - 1) % groom_every == 0:
            _sg.groom(V, T, fixed=fixed, smooth_lam=groom_smooth)
            effective["grooms_run"] += 1
        plateau.minimize_area(V, T, fixed, outer_iters=1, **kwargs)
        trace.append({"iter": it, "t": _timer() - t0, "E": M.mesh_area(V, T)})
        move = np.max(np.linalg.norm(V - V0, axis=1))
        if move < 1e-6 * max(1.0, np.max(np.abs(V))):
            break
    if groom_every and effective["grooms_run"] == 0:
        raise RuntimeError("groom_every was configured but no groom cycle "
                           "ran -- refusing to report a null result")
    return trace, effective


# --------------------------------------------------------------------------
# Plateau cases
# --------------------------------------------------------------------------

def case_catenoid(config, nring=64, nrow=17):
    V, T, fixed = cylinder_grid(nring, nrow)
    A_exact, c = catenoid_area_analytic()
    kwargs = dict(config.get("plateau_kwargs", {}))
    trace, effective = _minimize_area_traced(V, T, fixed, 40, kwargs)
    mets = _plateau_mesh_metrics(V, T, fixed)
    mets["area_exact"] = A_exact
    mets["area_rel_err"] = (mets["area"] - A_exact) / A_exact
    mid = (nrow // 2) * nring
    mets["waist_rel_err"] = abs(float(np.mean(np.hypot(
        V[mid:mid + nring, 0], V[mid:mid + nring, 1]))) - c) / c
    mets["selfx"] = M.selfx_count(V, T)
    mets["iters"] = trace[-1]["iter"]
    return {"metrics": mets, "trace": trace, "time_s": trace[-1]["t"],
            "n_verts": len(V), "n_tris": len(T), "effective": effective}


def case_catenoid_fine(config):
    return case_catenoid(config, nring=96, nrow=25)


def case_seifert_span(config, q=5, m=140, rings=24):
    from math_art.minsurf import plateau
    V, quads, fixed = plateau.build_seifert_span_grid(q, m, rings)
    T = plateau._quads_to_tris(quads)
    kwargs = dict(config.get("plateau_kwargs", {}))
    iters = int(config.get("seifert_iters", plateau._SEIFERT_MAX_ITERS))
    rim0 = V[fixed].copy()
    trace, effective = _minimize_area_traced(V, T, fixed, iters, kwargs)
    mets = _plateau_mesh_metrics(V, T, fixed)
    mets["selfx"] = M.selfx_count(V, T)
    mets["rim_max_move"] = float(np.max(np.abs(V[fixed] - rim0)))
    mets["iters"] = trace[-1]["iter"]
    return {"metrics": mets, "trace": trace, "time_s": trace[-1]["t"],
            "n_verts": len(V), "n_tris": len(T), "genus": (q - 1) // 2,
            "effective": effective}


def case_seifert_sweep(config):
    """Embeddedness gate: the (2,q) Seifert span across the q x samples
    grid, relaxed the capped 8 passes; the only metric that matters here
    is the self-intersection count staying 0 everywhere (the documented
    reason for the clamp + iteration cap)."""
    from math_art.minsurf import plateau
    import inspect
    kwargs = dict(config.get("plateau_kwargs", {}))
    kwargs.pop("groom_every", None)
    kwargs.pop("groom_smooth", None)
    groom_every = int(config.get("plateau_kwargs", {}).get(
        "groom_every", 0) or 0)
    sig = inspect.signature(plateau.minimize_area)
    effective = {
        "cotan_mode": kwargs.get("cotan_mode",
                                 sig.parameters["cotan_mode"].default),
        "groom_every": groom_every,
        "grooms_run": 0,
    }
    per = {}
    t_all = _timer()
    worst = 0
    grid = ([(q, m, 24) for q in (1, 3, 5, 7) for m in (96, 140, 200)]
            + [(q, 48, 8) for q in (1, 3, 5, 7, 9)])   # toolkit selftest grid
    for q, m, rings in grid:
        V, quads, fixed = plateau.build_seifert_span_grid(q, m, rings)
        T = plateau._quads_to_tris(quads)
        if groom_every:
            from math_art.solver import groom as _sg
            for it in range(plateau._SEIFERT_MAX_ITERS):
                if it and it % groom_every == 0:
                    _sg.groom(V, T, fixed=fixed)
                    effective["grooms_run"] += 1
                plateau.minimize_area(V, T, fixed, outer_iters=1,
                                      **kwargs)
        else:
            plateau.minimize_area(
                V, T, fixed,
                outer_iters=plateau._SEIFERT_MAX_ITERS, **kwargs)
        sx = M.selfx_count(V, T)
        worst = max(worst, sx)
        per[f"q{q}_m{m}_r{rings}"] = {
            "selfx": sx,
            "min_angle_deg": M.tri_quality(V, T)["min_angle_deg"]}
    if groom_every and effective["grooms_run"] == 0:
        raise RuntimeError("groom_every was configured but no groom cycle "
                           "ran -- refusing to report a null result")
    mets = {"selfx_worst": worst,
            "n_embedded": sum(1 for p in per.values() if p["selfx"] == 0),
            "n_total": len(per)}
    return {"metrics": mets, "per_solid": per, "trace": [],
            "time_s": _timer() - t_all, "effective": effective}


def case_seifert_span_q3(config):
    return case_seifert_span(config, q=3)


def case_seifert_span_q5(config):
    return case_seifert_span(config, q=5)


# --------------------------------------------------------------------------
# Seifert pipeline (relax + fair)
# --------------------------------------------------------------------------

def case_seifert_fair(config, word="AAA"):
    from math_art.seifert.build import seifert_surface
    from math_art.seifert.relax import relax
    from math_art.seifert.subdivide import catmull_clark
    from math_art.seifert import fair

    mesh = seifert_surface(word)
    genus0 = mesh.info().genus
    t0 = _timer()
    mesh = relax(mesh, iterations=100)
    t_relax = _timer() - t0
    mesh = catmull_clark(mesh, 2)
    V_before = mesh.vertices.copy()
    area_before = mesh.area()
    fair_kwargs = dict(config.get("fair_kwargs", {}))
    t0 = _timer()
    faired = fair.minimal_surface(mesh, strength=2.0, iterations=10,
                                  **fair_kwargs)
    t_fair = _timer() - t0
    info = faired.info()
    tri = faired.triangulated()
    T = np.asarray(tri.faces, dtype=np.int64)
    V = tri.vertices
    boundary = np.zeros(len(V), dtype=bool)
    bidx = [v for loop in faired.boundary_loops() for v in loop]
    boundary[bidx] = True
    mets = {
        "area_before_fair": area_before,
        "area": faired.area(),
        "area_shrink_frac": 1.0 - faired.area() / max(area_before, 1e-30),
        "genus_preserved": int(info.genus == genus0),
        "rim_max_move": M.max_move(V_before, faired.vertices, boundary),
        "interior_mean_move": M.mean_move(V_before, faired.vertices,
                                          ~boundary),
        "selfx": M.selfx_count(V, T),
        "time_relax_s": t_relax,
        "time_fair_s": t_fair,
    }
    mets.update(M.mean_curvature_residual(V, T, free=~boundary))
    mets.update(M.tri_quality(V, T))
    mets.update(M.cotan_clamp_fraction(V, T))
    return {"metrics": mets, "trace": [], "time_s": t_relax + t_fair,
            "n_verts": len(V), "n_tris": len(T)}


# --------------------------------------------------------------------------
# canonical / biscribed polyhedra
# --------------------------------------------------------------------------

_CANON_SOLIDS = ("gC", "pC", "wC", "tI", "kD")
_BISCRIBE_SOLIDS = (("C", True), ("D", True), ("kC", True), ("kD", True),
                    ("tO", True), ("aC", False), ("aD", False),
                    ("tC", False))


def case_canonical(config):
    from math_art.polyhedra import canonical
    from math_art.polyhedra.conway import apply_conway
    mode = config.get("canonical_mode", "hart")
    kwargs = dict(config.get("canonical_kwargs", {}))
    per = {}
    total_t = 0.0
    for text in _CANON_SOLIDS:
        V, F = apply_conway(text)
        snaps = []
        if mode == "bd":
            fn = lambda V, F, **kw: canonical.canonicalize_bd(V, F, **kw)
        else:
            fn = canonical.canonicalize
        # timing run (no trace)
        t0 = _timer()
        Vc = fn(V, F, iters=400, **kwargs)
        dt = _timer() - t0
        total_t += dt
        # trace run (same path, counts iterations via the opt-in hook)
        iters_used = None
        try:
            fn(V, F, iters=400, trace=lambda it, P: snaps.append(it),
               **kwargs)
            iters_used = (snaps[-1] + 1) if snaps else 0
        except TypeError:
            pass                       # baseline signature without trace
        per[text] = {
            "tangent_spread": M.tangent_spread(Vc, F),
            "planarity_max": M.planarity_max(Vc, F),
            "iters": iters_used,
            "time_s": dt,
        }
    agg = {
        "tangent_spread_worst": max(p["tangent_spread"] for p in per.values()),
        "tangent_spread_mean": float(np.mean(
            [p["tangent_spread"] for p in per.values()])),
        "planarity_worst": max(p["planarity_max"] for p in per.values()),
        "iters_mean": (float(np.mean([p["iters"] for p in per.values()]))
                       if all(p["iters"] is not None for p in per.values())
                       else None),
    }
    return {"metrics": agg, "per_solid": per, "trace": [],
            "time_s": total_t}


def case_biscribe(config):
    from math_art.polyhedra import canonical
    from math_art.polyhedra.conway import apply_conway
    kwargs = dict(config.get("biscribe_kwargs", {}))
    per = {}
    total_t = 0.0
    correct = 0
    for text, exists in _BISCRIBE_SOLIDS:
        V, F = apply_conway(text)
        t0 = _timer()
        Vb, conv = canonical.biscribe(V, F, iters=2500, **kwargs)
        dt = _timer() - t0
        total_t += dt
        rs, fs = M.sphere_spreads(Vb, F)
        snaps = []
        iters_used = None
        try:
            canonical.biscribe(V, F, iters=2500,
                               trace=lambda it, P: snaps.append(it),
                               **kwargs)
            iters_used = (snaps[-1] + 1) if snaps else 0
        except TypeError:
            pass
        per[text] = {"exists": exists, "converged": bool(conv),
                     "correct": int(bool(conv) == exists),
                     "r_spread": rs, "f_spread": fs,
                     "iters": iters_used, "time_s": dt}
        correct += per[text]["correct"]
    succ = [p for p in per.values() if p["exists"] and p["converged"]]
    agg = {
        "n_correct": correct,
        "n_total": len(_BISCRIBE_SOLIDS),
        "r_spread_worst_ok": max((p["r_spread"] for p in succ), default=None),
        "f_spread_worst_ok": max((p["f_spread"] for p in succ), default=None),
        "iters_mean_ok": (float(np.mean([p["iters"] for p in succ]))
                          if succ and all(p["iters"] is not None
                                          for p in succ) else None),
    }
    return {"metrics": agg, "per_solid": per, "trace": [],
            "time_s": total_t}


# --------------------------------------------------------------------------
# knots
# --------------------------------------------------------------------------

def case_knot_trefoil(config):
    from math_art.knots import relax as krelax
    from math_art.knots.braid import braid_closure_points, parse_letters
    from math_art.knots.resample import resample_closed
    P = resample_closed(braid_closure_points(parse_letters('AAA')), 120)
    kwargs = dict(config.get("knot_kwargs", {}))
    iters_total = int(config.get("knot_iters", 150))
    chunk = 10
    trace = [{"iter": 0, "t": 0.0,
              "E": M.ropelength_proxy(P), "gap": M.min_far_gap(P)}]
    t0 = _timer()
    it = 0
    while it < iters_total:
        n = min(chunk, iters_total - it)
        P = krelax.relax_knot(P, iters=n, **kwargs)
        it += n
        trace.append({"iter": it, "t": _timer() - t0,
                      "E": M.ropelength_proxy(P), "gap": M.min_far_gap(P)})
    mets = {
        "min_far_gap": M.min_far_gap(P),
        "length": M.curve_length(P),
        "ropelength_proxy": M.ropelength_proxy(P),
        "turning_rms_deg": M.turning_rms_deg(P),
    }
    return {"metrics": mets, "trace": trace, "time_s": trace[-1]["t"],
            "n_verts": len(P)}


def case_knot_hopf(config):
    from math_art.knots import relax as krelax
    t = np.linspace(0.0, 2 * np.pi, 100, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    zero = np.zeros_like(t)
    A = np.stack([c, s, zero], axis=1)
    B = np.stack([1.0 + c, zero, s], axis=1)
    lk0 = krelax.linking_number(A, B)
    kwargs = dict(config.get("knot_kwargs", {}))
    t0 = _timer()
    out = krelax.relax_link([A.copy(), B.copy()], iters=100, **kwargs)
    dt = _timer() - t0
    lk1 = krelax.linking_number(out[0], out[1])
    inter = float(np.min(np.linalg.norm(
        out[0][:, None, :] - out[1][None, :, :], axis=2)))
    mets = {
        "lk_before": lk0, "lk_after": lk1,
        "lk_preserved": int(abs(lk1 - lk0) < 0.1),
        "inter_component_gap": inter,
        "length": M.curve_length(out[0]) + M.curve_length(out[1]),
    }
    return {"metrics": mets, "trace": [], "time_s": dt,
            "n_verts": len(out[0]) + len(out[1])}


# --------------------------------------------------------------------------
# hyperbolic crochet sheet
# --------------------------------------------------------------------------

def case_crochet(config):
    from math_art.hyperbolic import crochet
    stitch = 0.10
    P, E0, E1, REST, nbr, tris, pin, pin_pos = crochet._crochet_mesh(
        6, 10, stitch, 300)
    kwargs = dict(config.get("crochet_kwargs", {}))
    iters = int(config.get("crochet_iters", 150))
    t0 = _timer()
    P = crochet._relax(P, E0, E1, REST, nbr, pin, pin_pos, iters,
                       0.08, 0.4 * stitch, 0.5, **kwargs)
    dt = _timer() - t0
    mets = dict(M.edge_metric_error(P, E0, E1, REST))
    mets["selfx"] = M.selfx_count(P, np.asarray(tris, dtype=np.int64))
    mets["median_K"] = crochet.mean_curvature(P, tris)
    mets["z_extent"] = float(P[:, 2].max() - P[:, 2].min())
    return {"metrics": mets, "trace": [], "time_s": dt,
            "n_verts": len(P), "n_tris": len(tris)}


# --------------------------------------------------------------------------
# volume-constrained bubble evolution (solver/volume.py)
# --------------------------------------------------------------------------

def _evolve_checked(V, T, labels, targets, kwargs):
    """Run solver.volume.evolve with the same config discipline as the
    plateau cases: unknown kwargs raise, and the returned `effective`
    dict records what actually ran (a configured-but-never-run groom
    refuses to report a null result)."""
    from math_art.solver import volume as sv
    import inspect
    sig = inspect.signature(sv.evolve)
    unknown = [k for k in kwargs if k not in sig.parameters
               or k in ("V", "T", "labels", "targets")]
    if unknown:
        raise ValueError(f"kwargs not accepted by evolve: {unknown}")
    info = sv.evolve(V, T, labels, targets=targets, **kwargs)
    effective = {
        "iters": kwargs.get("iters", sig.parameters["iters"].default),
        "iters_run": info["iters_run"],
        "groom_every": kwargs.get("groom_every",
                                  sig.parameters["groom_every"].default),
        "grooms_run": info["grooms_run"],
        "mobility": kwargs.get("mobility",
                               sig.parameters["mobility"].default),
        "cotan_mode": kwargs.get("cotan_mode",
                                 sig.parameters["cotan_mode"].default),
    }
    if effective["groom_every"] and info["grooms_run"] == 0:
        raise RuntimeError("groom_every was configured but no groom cycle "
                           "ran -- refusing to report a null result")
    return info, effective


def _evolution_metrics(info):
    """Drift / monotonicity metrics shared by all bubble cases.
    area_max_rise is over NON-groom steps: the line search evaluates
    area at volume-restored trial positions, so any rise there is a
    solver bug (groom steps may legitimately bump the area a little)."""
    hist = info["history"]
    solver_rises = [h["rise"] for h in hist if not h["groomed"]]
    groom_rises = [h["rise"] for h in hist if h["groomed"]]
    return {
        "vol_drift_max": max((h["drift_post"] for h in hist), default=0.0),
        "vol_drift_pre_max": max((h["drift_pre"] for h in hist),
                                 default=0.0),
        "area_max_rise": max(solver_rises, default=0.0),
        "groom_max_rise": max(groom_rises, default=0.0),
        "iters": info["iters_run"],
    }


def case_bubble_single(config, subdiv=3):
    """A squashed ellipsoid under a volume constraint must relax to the
    round sphere: area against the isoperimetric (36 pi V^2)^(1/3), the
    Lagrange pressure against Young-Laplace 2/r, exact volume
    conservation, monotone area.  The honest discrete reference is the
    same-combinatorics icosphere scaled to the target volume: the
    relaxed area may not beat the smooth bound and should land at
    (or slightly below) the reference's excess."""
    from math_art.bubble_generator import build_single_bubble_mesh
    from math_art.solver import volume as sv
    kwargs = dict(config.get("bubble_kwargs", {}))
    kwargs.setdefault("iters", 300)
    V, T, labels = build_single_bubble_mesh(1.0, subdiv)
    target = 4.0 * math.pi / 3.0
    # discrete reference: perfect icosphere scaled to the exact target
    # DISCRETE volume (the inscribed mesh's volume is below 4 pi/3)
    v0 = sv.region_volumes(V, T, labels)[0]
    s = (target / v0) ** (1.0 / 3.0)
    A_ref = sv.mesh_area(s * V, T)
    V *= np.array([1.25, 0.8, 1.0])
    t0 = _timer()
    info, effective = _evolve_checked(V, T, labels, [target], kwargs)
    dt = _timer() - t0
    r_star = (3.0 * target / (4.0 * math.pi)) ** (1.0 / 3.0)
    A_star = (36.0 * math.pi * target ** 2) ** (1.0 / 3.0)
    c = V.mean(axis=0)
    rad = np.linalg.norm(V - c, axis=1)
    mets = {
        "area": info["area"],
        "area_rel_excess": (info["area"] - A_star) / A_star,
        "area_rel_excess_ref": (A_ref - A_star) / A_star,
        "radius_cv": float(np.std(rad) / np.mean(rad)),
        "radius_max_rel_dev": float(np.max(np.abs(rad / np.mean(rad)
                                                  - 1.0))),
        "pressure": float(info["pressures"][0]),
        "pressure_rel_err": abs(float(info["pressures"][0])
                                - 2.0 / r_star) / (2.0 / r_star),
    }
    mets.update(_evolution_metrics(info))
    trace = [{"iter": h["it"], "E": h["area"]} for h in info["history"]]
    return {"metrics": mets, "trace": trace, "time_s": dt,
            "n_verts": len(V), "n_tris": len(T), "effective": effective}


def case_bubble_single_fine(config):
    """Same at one more subdivision: the discretization errors (area
    excess, radius spread, pressure error) must shrink ~4x."""
    return case_bubble_single(config, subdiv=4)


def _case_bubble_double(config, r1, r2, nphi, iters_default=600,
                        perturb=(1.10, 0.95, 1.0)):
    """Perturbed standard double bubble relaxing back: dihedral angles
    at the Plateau border (Taylor: 120 degrees), per-film sphericity,
    the Young-Laplace curvature relation 1/r3 = 1/r1 - 1/r2, pressures,
    exact volumes, monotone area.  Targets are the ANALYTIC lobe
    volumes, so the area comparison is against the closed-form standard
    double bubble for exactly those volumes (the proven minimizer --
    Hutchings-Morgan-Ritore-Ros 2002); the discrete excess must be
    positive and O(h^2)."""
    from math_art.bubble_generator import (build_double_bubble_mesh,
                                           double_bubble_geometry)
    from math_art.solver import volume as sv
    kwargs = dict(config.get("bubble_kwargs", {}))
    kwargs.setdefault("iters", iters_default)
    V, T, labels = build_double_bubble_mesh(r1, r2, nphi=nphi)
    geo = double_bubble_geometry(r1, r2)
    ang_seed = sv.triple_line_angles(V, T)
    V *= np.array(perturb)
    t0 = _timer()
    info, effective = _evolve_checked(V, T, labels,
                                      [geo["V1"], geo["V2"]], kwargs)
    dt = _timer() - t0
    ang_raw = sv.triple_line_angles(V, T)
    fits = M.film_fits(V, T, labels)
    ang_fit = M.fitted_triple_angles(V, T, labels, fits)
    r1f = fits[(0, 1)]["r"]
    r2f = fits[(0, 2)]["r"]
    r3f = fits[(2, 1)]["r"]
    curv_resid = abs((0.0 if math.isinf(r3f) else 1.0 / r3f)
                     - (1.0 / r1f - 1.0 / r2f))
    p1, p2 = (float(p) for p in info["pressures"])
    dp_star = 0.0 if math.isinf(geo["r3"]) else 2.0 / geo["r3"]
    mets = {
        "area": info["area"],
        "area_analytic": geo["A"],
        "area_rel_excess": (info["area"] - geo["A"]) / geo["A"],
        "angle_rms_fit": float(np.sqrt(np.mean((ang_fit - 120.0) ** 2))),
        "angle_min_fit": float(np.min(ang_fit)),
        "angle_max_fit": float(np.max(ang_fit)),
        "angle_rms_raw": float(np.sqrt(np.mean((ang_raw - 120.0) ** 2))),
        "angle_rms_raw_seed": float(np.sqrt(np.mean(
            (ang_seed - 120.0) ** 2))),
        "fit_rms_worst": max(f["rms"] for f in fits.values()),
        "r1_fit": r1f, "r2_fit": r2f,
        "r3_fit": (r3f if not math.isinf(r3f) else None),
        "curv_resid": curv_resid,
        "p1": p1, "p2": p2,
        "dp_err": abs((p1 - p2) - dp_star),
        "pressure_rel_err_worst": max(abs(p1 - 2.0 / r1) * r1 / 2.0,
                                      abs(p2 - 2.0 / r2) * r2 / 2.0),
    }
    mets.update(_evolution_metrics(info))
    trace = [{"iter": h["it"], "E": h["area"]} for h in info["history"]]
    return {"metrics": mets, "trace": trace, "time_s": dt,
            "n_verts": len(V), "n_tris": len(T), "effective": effective}


def case_bubble_double(config):
    return _case_bubble_double(config, 1.0, 1.0, 48, iters_default=1200)


def case_bubble_double_unequal(config):
    """Converges more slowly than the equal case (the curved interface
    couples the lobes), hence the deeper default budget: at 600
    iterations the fitted angle rms is still ~0.7 deg, at 1800 it
    reaches ~0.1 deg."""
    return _case_bubble_double(config, 0.8, 1.2, 48, iters_default=1800)


def case_bubble_double_fine(config):
    """The equal double bubble at doubled rim resolution, measuring
    DISCRETIZATION-ERROR scaling: angle spread, area excess, and film
    fit rms must shrink vs bubble_double (expected ~h^2 for area).
    The perturbation is mild because this case's job is the scaling at
    the discrete equilibrium; recovery from a large perturbation is
    proven by the coarse cases, and at nphi=96 the large-perturbation
    transient needs several thousand iterations (measured: still ~4 deg
    angle rms after 2400) -- an honest solver limitation recorded in
    the plan write-up, not hidden by a huge budget here."""
    return _case_bubble_double(config, 1.0, 1.0, 96, iters_default=600,
                               perturb=(1.02, 0.99, 1.0))


# --------------------------------------------------------------------------
# CMC / capillary family (solver/walls.py + cmc_generator.py)
# --------------------------------------------------------------------------

def _cmc_kwargs(config, defaults):
    """Merge config['cmc_kwargs'] over per-case defaults, with the same
    discipline as bubble_kwargs: unknown keys raise in _evolve_checked."""
    kwargs = dict(defaults)
    kwargs.update(config.get("cmc_kwargs", {}))
    return kwargs


def _bridge_case(config, volume_factor, nring, nrow, iters_default,
                 squash=0.0, targets_override=None):
    """Liquid bridge between pinned rims at a prescribed volume.  The
    defining CMC checks: H constant over the free surface (vertex cv,
    plus the row-mean cv on the intact lathe rows), the Young-Laplace
    cross-check p = 2H between two independently computed quantities
    (Lagrange pressure vs cotan curvature), and the conserved axial
    force F(z) = 2 pi r / sqrt(1 + r'^2) - p pi r^2 of a CMC surface
    of revolution, measured per row.  Row metrics are only meaningful
    while the lathe structure survives, so they are skipped (None) if
    a groom config rewired the mesh."""
    from math_art import cmc_generator as cg
    from math_art.solver import volume as sv
    h, R = 0.5, 1.0
    kwargs = _cmc_kwargs(config, {"iters": iters_default})
    V, T, labels, fixed = cg.build_bridge_mesh(nring, nrow, h, R)
    v_lat0 = sv.region_volumes(V, T, labels)[0]
    if targets_override is not None:
        target = targets_override
    else:
        v_true0 = cg.bridge_true_volume(v_lat0, R, h, nring)
        target = cg.bridge_lateral_target(volume_factor * v_true0,
                                          R, h, nring)
    if squash:
        fac = 1.0 + squash * np.cos(np.pi * V[:, 2] / (2.0 * h))
        fac[fixed] = 1.0
        V[:, 0] *= fac
        V[:, 1] *= fac
    kwargs["fixed"] = fixed
    t0 = _timer()
    info, effective = _evolve_checked(V, T, labels, [target], kwargs)
    dt = _timer() - t0
    free = ~fixed
    Hs = cg.signed_mean_curvature(V, T)
    Hin = Hs[free]
    p = float(info["pressures"][0])
    mets = {
        "area": info["area"],
        "pressure": p,
        "H_mean": float(np.mean(Hin)),
        "H_cv": float(np.std(Hin) / max(abs(np.mean(Hin)), 1e-300)),
        "p_minus_2H": abs(p - 2.0 * float(np.mean(Hin))),
    }
    if volume_factor == 1.0:
        # cylinder ground truth: radius constant at R, p = 1/R
        rr = np.hypot(V[free, 0], V[free, 1])
        mets["radius_cv"] = float(np.std(rr) / np.mean(rr))
        mets["pressure_rel_err"] = abs(p - 1.0 / R) / (1.0 / R)
    if info["grooms_run"] == 0:
        # rows intact: row-mean H profile and the axial-force invariant
        Hrow = Hs.reshape(nrow, nring).mean(axis=1)[2:-2]
        r = np.hypot(V[:, 0], V[:, 1]).reshape(nrow, nring).mean(axis=1)
        z = V[:, 2].reshape(nrow, nring).mean(axis=1)
        rp = np.gradient(r, z)
        F = (2.0 * np.pi * r / np.sqrt(1.0 + rp * rp)
             - p * np.pi * r * r)[1:-1]
        mets["H_row_cv"] = float(np.std(Hrow)
                                 / max(abs(np.mean(Hrow)), 1e-300))
        mets["F_cv"] = float(np.std(F) / max(abs(np.mean(F)), 1e-300))
    else:
        mets["H_row_cv"] = None
        mets["F_cv"] = None
    mets.update(_evolution_metrics(info))
    trace = [{"iter": hh["it"], "E": hh["area"]}
             for hh in info["history"]]
    return {"metrics": mets, "trace": trace, "time_s": dt,
            "n_verts": len(V), "n_tris": len(T), "effective": effective}


def case_cmc_bridge_cyl(config):
    """Radially bulged cylinder at the cylinder's own volume must relax
    back to the cylinder (the CMC family's exact member): p ~ 1 = 2/R,
    H ~ 1/(2R), radius constant (radius_cv, pressure_rel_err)."""
    return _bridge_case(config, 1.0, 48, 17, 400, squash=0.08)


def case_cmc_bridge_fat(config):
    """Unduloid barrel at 1.25 x the cylinder volume (measured budget:
    interior rows constant to ~5e-4 by 1200 iterations; at 400 the
    transient still reads ~3e-2 vertex H cv)."""
    return _bridge_case(config, 1.25, 48, 17, 1200)


def case_cmc_bridge_thin(config):
    """Nodoid-side neck at 0.75 x the cylinder volume (just below the
    catenoid volume ~0.809 x): the pressure must come out NEGATIVE."""
    return _bridge_case(config, 0.75, 48, 17, 600)


def case_cmc_bridge_fat_fine(config):
    """The unduloid at doubled resolution: p_minus_2H must shrink ~4x
    (measured 5.5e-3 -> 1.3e-3).  The vertex H cv does NOT shrink --
    it is limited by tangential mesh disorder (sliding within the
    surface is area-neutral, so vertex positions are underdetermined)
    amplified by the pointwise cotan estimator; the row-mean cv and
    the p=2H cross-check are the converging instruments.  Recorded
    honestly rather than tuned away."""
    return _bridge_case(config, 1.25, 96, 33, 2400)


def _catenoid_limit_case(config, nring, nrow, iters_default):
    """As the bridge volume approaches the catenoid's own volume the
    relaxed surface must BE the minimal catenoid: analytic area, and a
    Lagrange pressure indistinguishable from zero at the discretization
    scale (p = dA/dV = 0 at the minimal surface) -- both O(h^2)."""
    from math_art import cmc_generator as cg
    h, R = 0.5, 1.0
    A_cat, c = cg.catenoid_area(h, R)
    V_cat = cg.catenoid_volume(h, R)
    target = cg.bridge_lateral_target(V_cat, R, h, nring)
    res = _bridge_case(config, None, nring, nrow, iters_default,
                       targets_override=target)
    mets = res["metrics"]
    mets["area_exact"] = A_cat
    mets["area_rel_err"] = abs(mets["area"] - A_cat) / A_cat
    mets["pressure_abs"] = abs(mets["pressure"])
    # a cv against a mean of ~zero is not a number worth reporting:
    # at the catenoid H -> 0, so the informative spread measure is the
    # ABSOLUTE mean (both instruments must vanish together, O(h^2))
    mets["H_abs_mean"] = abs(mets.pop("H_mean"))
    mets.pop("H_cv")
    mets.pop("H_row_cv")
    return res


def case_cmc_catenoid(config):
    return _catenoid_limit_case(config, 48, 17, 400)


def case_cmc_catenoid_fine(config):
    return _catenoid_limit_case(config, 96, 33, 1200)


def _drop_case(config, theta_deg, nring, iters_default, seed_theta_deg,
               groom_default=4):
    """Sessile drop: contact line sliding on the floor (two-sided wall
    on the rim, one-sided on the interior), wetting energy
    -cos(theta) * wetted area, volume constrained.  The zero-gravity
    ground truth is the spherical cap: achieved contact angle (sphere
    fit; the seed starts at a DIFFERENT angle so the line demonstrably
    slides), contact radius, free area, total energy, p = 2/R = 2H.
    Wall residual and velocity tangency must sit at solver tolerance;
    the monotone quantity is E = area - cos(theta) A_wet."""
    from math_art import cmc_generator as cg
    from math_art.solver import walls as sw
    kwargs = _cmc_kwargs(config, {"iters": iters_default,
                                  "groom_every": groom_default})
    th = math.radians(theta_deg)
    Vt = 2.0 * math.pi / 3.0
    geo = cg.cap_geometry(th, Vt)
    geo_seed = cg.cap_geometry(math.radians(seed_theta_deg), Vt)
    V, T, labels, rim = cg.build_cap_mesh(seed_theta_deg,
                                          geo_seed["R"], nring)
    T = np.ascontiguousarray(T)
    interior = ~rim
    loop = cg.rim_loop(V, nring)
    ext_e, ext_g = cg.drop_energy_terms(th, loop)
    plane = sw.PlaneWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    floor = sw.PlaneWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                         one_sided=True)
    kwargs.update(walls=[(plane, rim), (floor, interior)],
                  ext_energy=ext_e, ext_grad=ext_g)
    t0 = _timer()
    info, effective = _evolve_checked(V, T, labels, [Vt], kwargs)
    dt = _timer() - t0
    ang, fit_rms = cg.measured_contact_angle(V, T)
    a_ach = float(np.mean(np.hypot(V[loop, 0], V[loop, 1])))
    Hs = cg.signed_mean_curvature(V, T)[interior]
    p = float(info["pressures"][0])
    E_final = info["area"] + ext_e(V)
    hist = info["history"]
    mets = {
        "angle_achieved_deg": ang,
        "angle_err_deg": abs(ang - theta_deg),
        "contact_r": a_ach,
        "contact_r_rel_err": abs(a_ach - geo["a"]) / geo["a"],
        "area": info["area"],
        "area_rel_err": abs(info["area"] - geo["A_free"])
        / geo["A_free"],
        "E_rel_err": abs(E_final - geo["E"]) / abs(geo["E"]),
        "pressure": p,
        "pressure_rel_err": abs(p - 2.0 / geo["R"]) / (2.0 / geo["R"]),
        "H_mean": float(np.mean(Hs)),
        "H_cv": float(np.std(Hs) / max(abs(np.mean(Hs)), 1e-300)),
        "p_minus_2H": abs(p - 2.0 * float(np.mean(Hs))),
        "fit_rms": fit_rms,
        "wall_resid_max": max(hh["wall_resid"] for hh in hist),
        "vt_normal_max": max(hh["vt_normal_max"] for hh in hist),
        "min_interior_z": float(V[interior, 2].min()),
        "E_max_rise": max((hh["E_rise"] for hh in hist
                           if not hh["groomed"]), default=0.0),
        "vol_drift_max": max((hh["drift_post"] for hh in hist),
                             default=0.0),
        "iters": info["iters_run"],
    }
    trace = [{"iter": hh["it"], "E": hh["E"]} for hh in hist]
    return {"metrics": mets, "trace": trace, "time_s": dt,
            "n_verts": len(V), "n_tris": len(T), "effective": effective}


def case_cmc_drop45(config):
    """45-degree (wetting) drop seeded as a hemisphere: the contact
    line slides OUT by ~45% in radius."""
    return _drop_case(config, 45.0, 48, 1500, 90.0)


def case_cmc_drop135(config):
    """135-degree (beading) drop seeded at 120 degrees: the contact
    line slides IN and the free surface bulges past vertical.  Slower
    to converge than the wetting case (measured: the angle error is
    still ~1.3 deg at 600 iterations, ~0.2-0.3 deg by 3000)."""
    return _drop_case(config, 135.0, 48, 3000, 120.0)


def case_cmc_drop45_fine(config):
    """The 45-degree drop at doubled contact-line resolution: the
    achieved-angle error and contact-radius error must shrink ~4x
    (measured -0.014 -> +0.004 deg and 1.7e-3 -> 3.6e-4)."""
    return _drop_case(config, 45.0, 96, 3000, 90.0)


# --------------------------------------------------------------------------
# high-genus embedder planarize (research script, not shipped code)
# --------------------------------------------------------------------------

def case_planarize(config):
    import importlib.util
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    # the embedder lives in the (gitignored) research tree of the MAIN
    # checkout; fall back to it when absent in a worktree
    cands = [
        os.path.join(root, "research", "highgenus_embedder"),
        r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\research\highgenus_embedder",
    ]
    home = next((c for c in cands
                 if os.path.exists(os.path.join(c, "roundembed.py"))), None)
    if home is None:
        return {"metrics": {"skipped": 1}, "trace": [], "time_s": 0.0,
                "note": "research/highgenus_embedder not found"}
    spec = importlib.util.spec_from_file_location(
        "roundembed", os.path.join(home, "roundembed.py"))
    re_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(re_mod)
    data = json.load(open(os.path.join(home, "maps_combinatorics.json")))
    name = config.get("planarize_map") or "Heptagonal Dodecahedron B"
    rec = data[name]
    Vn = len(rec["V"]) if not isinstance(rec["V"], int) else rec["V"]
    F = [list(f) for f in rec["F"]]
    Lap = re_mod.laplacian(Vn, F)
    w, vec = np.linalg.eigh(Lap)
    cl = re_mod.clusters(w)
    sub = None
    for c in cl:
        if len(c) >= 3 and w[c[0]] > 1e-6:
            sub = c[:3]
            break
    if sub is None:
        nz = [i for i in range(len(w)) if w[i] > 1e-6]
        sub = nz[:3]
    X0 = vec[:, sub].copy()
    kwargs = dict(config.get("planarize_kwargs", {}))
    total = int(config.get("planarize_iters", 500))
    chunk = 50
    trace = [{"iter": 0, "t": 0.0, "E": re_mod.planar_dev(X0, F)}]
    X = X0
    t0 = _timer()
    it = 0
    while it < total:
        X = re_mod.planarize(X, F, iters=min(chunk, total - it), **kwargs)
        it += min(chunk, total - it)
        trace.append({"iter": it, "t": _timer() - t0,
                      "E": re_mod.planar_dev(X, F)})
    mets = {
        "planar_dev": re_mod.planar_dev(X, F),
        "aspect": re_mod.aspect(X),
        "crossings": re_mod.crossings(X, F),
        "map": name,
    }
    return {"metrics": mets, "trace": trace, "time_s": trace[-1]["t"],
            "n_verts": Vn}


# Every config key any case reads.  run.py rejects a config containing
# anything else, so a typo cannot silently benchmark a null change.
# (Keys irrelevant to a given case are legitimately ignored by that
# case -- one config file drives the whole suite -- which is why the
# check is a union at the runner level, plus per-case `effective`
# provenance in the results for the flags that actually ran.)
KNOWN_CONFIG_KEYS = {
    "plateau_kwargs", "seifert_iters", "fair_kwargs", "canonical_mode",
    "canonical_kwargs", "biscribe_kwargs", "knot_kwargs", "knot_iters",
    "crochet_kwargs", "crochet_iters", "planarize_map",
    "planarize_kwargs", "planarize_iters", "bubble_kwargs",
    "cmc_kwargs",
}

CASES = {
    "catenoid": case_catenoid,
    "catenoid_fine": case_catenoid_fine,
    "seifert_span_q3": case_seifert_span_q3,
    "seifert_span_q5": case_seifert_span_q5,
    "seifert_sweep": case_seifert_sweep,
    "seifert_fair": case_seifert_fair,
    "canonical": case_canonical,
    "biscribe": case_biscribe,
    "knot_trefoil": case_knot_trefoil,
    "knot_hopf": case_knot_hopf,
    "crochet": case_crochet,
    "planarize": case_planarize,
    "bubble_single": case_bubble_single,
    "bubble_single_fine": case_bubble_single_fine,
    "bubble_double": case_bubble_double,
    "bubble_double_unequal": case_bubble_double_unequal,
    "bubble_double_fine": case_bubble_double_fine,
    "cmc_bridge_cyl": case_cmc_bridge_cyl,
    "cmc_bridge_fat": case_cmc_bridge_fat,
    "cmc_bridge_thin": case_cmc_bridge_thin,
    "cmc_bridge_fat_fine": case_cmc_bridge_fat_fine,
    "cmc_catenoid": case_cmc_catenoid,
    "cmc_catenoid_fine": case_cmc_catenoid_fine,
    "cmc_drop45": case_cmc_drop45,
    "cmc_drop135": case_cmc_drop135,
    "cmc_drop45_fine": case_cmc_drop45_fine,
}
