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
    operations as a single call."""
    from math_art.minsurf import plateau
    kwargs = dict(kwargs)
    groom_every = int(kwargs.pop("groom_every", 0) or 0)
    groom_smooth = kwargs.pop("groom_smooth", 0.25)
    if groom_every:
        from math_art.solver import groom as _sg
    trace = [{"iter": 0, "t": 0.0, "E": M.mesh_area(V, T)}]
    t0 = _timer()
    for it in range(1, outer_iters + 1):
        V0 = V.copy()
        if groom_every and (it - 1) and (it - 1) % groom_every == 0:
            _sg.groom(V, T, fixed=fixed, smooth_lam=groom_smooth)
        plateau.minimize_area(V, T, fixed, outer_iters=1, **kwargs)
        trace.append({"iter": it, "t": _timer() - t0, "E": M.mesh_area(V, T)})
        move = np.max(np.linalg.norm(V - V0, axis=1))
        if move < 1e-6 * max(1.0, np.max(np.abs(V))):
            break
    return trace


# --------------------------------------------------------------------------
# Plateau cases
# --------------------------------------------------------------------------

def case_catenoid(config, nring=64, nrow=17):
    V, T, fixed = cylinder_grid(nring, nrow)
    A_exact, c = catenoid_area_analytic()
    kwargs = dict(config.get("plateau_kwargs", {}))
    trace = _minimize_area_traced(V, T, fixed, 40, kwargs)
    mets = _plateau_mesh_metrics(V, T, fixed)
    mets["area_exact"] = A_exact
    mets["area_rel_err"] = (mets["area"] - A_exact) / A_exact
    mid = (nrow // 2) * nring
    mets["waist_rel_err"] = abs(float(np.mean(np.hypot(
        V[mid:mid + nring, 0], V[mid:mid + nring, 1]))) - c) / c
    mets["selfx"] = M.selfx_count(V, T)
    mets["iters"] = trace[-1]["iter"]
    return {"metrics": mets, "trace": trace, "time_s": trace[-1]["t"],
            "n_verts": len(V), "n_tris": len(T)}


def case_catenoid_fine(config):
    return case_catenoid(config, nring=96, nrow=25)


def case_seifert_span(config, q=5, m=140, rings=24):
    from math_art.minsurf import plateau
    V, quads, fixed = plateau.build_seifert_span_grid(q, m, rings)
    T = plateau._quads_to_tris(quads)
    kwargs = dict(config.get("plateau_kwargs", {}))
    iters = int(config.get("seifert_iters", plateau._SEIFERT_MAX_ITERS))
    rim0 = V[fixed].copy()
    trace = _minimize_area_traced(V, T, fixed, iters, kwargs)
    mets = _plateau_mesh_metrics(V, T, fixed)
    mets["selfx"] = M.selfx_count(V, T)
    mets["rim_max_move"] = float(np.max(np.abs(V[fixed] - rim0)))
    mets["iters"] = trace[-1]["iter"]
    return {"metrics": mets, "trace": trace, "time_s": trace[-1]["t"],
            "n_verts": len(V), "n_tris": len(T), "genus": (q - 1) // 2}


def case_seifert_sweep(config):
    """Embeddedness gate: the (2,q) Seifert span across the q x samples
    grid, relaxed the capped 8 passes; the only metric that matters here
    is the self-intersection count staying 0 everywhere (the documented
    reason for the clamp + iteration cap)."""
    from math_art.minsurf import plateau
    kwargs = dict(config.get("plateau_kwargs", {}))
    kwargs.pop("groom_every", None)
    groom_every = int(config.get("plateau_kwargs", {}).get(
        "groom_every", 0) or 0)
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
    mets = {"selfx_worst": worst,
            "n_embedded": sum(1 for p in per.values() if p["selfx"] == 0),
            "n_total": len(per)}
    return {"metrics": mets, "per_solid": per, "trace": [],
            "time_s": _timer() - t_all}


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
}
