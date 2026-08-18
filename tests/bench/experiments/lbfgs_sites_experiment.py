# Experiment: per-site L-BFGS adoption evaluation for the remaining
# S2 candidate sites (the survey's list), measured rather than assumed:
#
#   A. minsurf/plateau.py -- direct L-BFGS(+Laplacian H0) area descent
#      vs the bespoke Pinkall-Polthier linearization on the catenoid.
#      (relax_normal_flow shares the verdict site: it is a POLISH pass
#      whose contract is "no tangential sliding, a fair net stays
#      fair" -- part B measures what an energy optimizer does to that
#      contract.)
#   B. relax_normal_flow polish on a normally-perturbed catenoid vs an
#      L-BFGS area polish: mean-curvature residual, wall time, AND the
#      tangential drift the polish contract forbids.
#   C. research/highgenus_embedder/roundembed.py::planarize -- the
#      fixed-point projection-averaging loop vs L-BFGS on the
#      least-squares-plane deviation energy (exact envelope-theorem
#      gradient), under the same centroid/inertia normalization.
#   D. hyperbolic/crochet.py::_relax -- budget-doubling probe: if the
#      documented 6-7% edge-length error does not move when the budget
#      doubles, the limiter is the smooth-vs-isometric tradeoff (the
#      S5 coarse-to-fine story), not step-size control, and no outer
#      optimizer is the cure.
#
#   python tests/bench/experiments/lbfgs_sites_experiment.py
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # tests/
from bench import _boot  # noqa: F401,E402
from bench import cases as C  # noqa: E402
from bench import metrics as M  # noqa: E402
import numpy as np  # noqa: E402
from math_art.minsurf import plateau  # noqa: E402
from math_art.solver import descent as sd  # noqa: E402
from math_art.solver import volume as sv  # noqa: E402


# ---------------------------------------------------------------- A --

def lbfgs_area_descent(V, T, fixed, iters=200, tol=1e-10):
    """Direct constrained-free area minimization: L-BFGS over the free
    vertices, Laplacian H0, Armijo from unit step."""
    free = ~fixed
    lb = sd.LBFGS(m=8)
    h0 = sd.LaplacianH0(eps=1e-3, tol=1e-2, max_iters=100)
    x_prev = None
    g_prev = None
    E_prev = sv.mesh_area(V, T)
    trace = [(0, 0.0, E_prev)]
    t0 = time.perf_counter()
    flat = 0
    for it in range(1, iters + 1):
        g = sv.area_gradient(V, T)
        g[fixed] = 0.0
        if x_prev is not None:
            lb.push((V - x_prev).ravel(), (g - g_prev).ravel())
        h0.update(V, T, free=free)
        d = lb.direction(g.ravel(), h0=h0).reshape(-1, 3)
        d = -d
        d[fixed] = 0.0
        slope = float(np.einsum('nj,nj->', g, d))
        if not (slope < 0.0):
            lb.reset()
            d = -h0(g.ravel()).reshape(-1, 3)
            d[fixed] = 0.0
            slope = float(np.einsum('nj,nj->', g, d))
        Lmean = float(np.mean(np.linalg.norm(
            V[T[:, 1]] - V[T[:, 0]], axis=1)))
        dmax = float(np.max(np.linalg.norm(d, axis=1)))
        s_max = 4.0 * Lmean / max(dmax, 1e-300)

        def energy(x):
            return sv.mesh_area(x, T)

        x1, s, E1, _ne = sd.armijo_backtrack(
            energy, V, d, slope, s0=min(1.0, s_max), E0=E_prev,
            s_max=s_max)
        if s == 0.0:
            break
        x_prev = V.copy()
        g_prev = g
        V[:] = x1
        trace.append((it, time.perf_counter() - t0, E1))
        if abs(E1 - E_prev) < tol * E_prev:
            flat += 1
            if flat >= 3:
                E_prev = E1
                break
        else:
            flat = 0
        E_prev = E1
    return V, trace


def part_a():
    A_exact, c_exact = C.catenoid_area_analytic()
    out = {}
    for nring, nrow in ((64, 17), (96, 25)):
        rec = {}
        # bespoke Pinkall-Polthier (exactly the bench's traced runner)
        V, T, fixed = C.cylinder_grid(nring, nrow)
        trace, _eff = C._minimize_area_traced(V, T, fixed, 40, {})
        mid = (nrow // 2) * nring
        rec["pp"] = {
            "iters": trace[-1]["iter"], "t": round(trace[-1]["t"], 3),
            "area_rel_err": (M.mesh_area(V, T) - A_exact) / A_exact,
            "waist_rel_err": abs(float(np.mean(np.hypot(
                V[mid:mid + nring, 0], V[mid:mid + nring, 1])))
                - c_exact) / c_exact,
            "H_rms": M.mean_curvature_residual(V, T, free=~fixed)["H_rms"],
        }
        # L-BFGS direct area descent
        V, T, fixed = C.cylinder_grid(nring, nrow)
        V, tr = lbfgs_area_descent(V, T, fixed, iters=300)
        rec["lbfgs"] = {
            "iters": tr[-1][0], "t": round(tr[-1][1], 3),
            "area_rel_err": (M.mesh_area(V, T) - A_exact) / A_exact,
            "waist_rel_err": abs(float(np.mean(np.hypot(
                V[mid:mid + nring, 0], V[mid:mid + nring, 1])))
                - c_exact) / c_exact,
            "H_rms": M.mean_curvature_residual(V, T, free=~fixed)["H_rms"],
        }
        out[f"catenoid_{nring}x{nrow}"] = rec
        print(f"A catenoid {nring}x{nrow}:", json.dumps(rec), flush=True)
    return out


# ---------------------------------------------------------------- B --

def part_b():
    # converge a catenoid, then perturb along vertex normals with a
    # smooth bump and polish it back two ways
    V, T, fixed = C.cylinder_grid(64, 17)
    plateau.minimize_area(V, T, fixed, outer_iters=40)
    vn = np.zeros_like(V)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3):
        np.add.at(vn, T[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    bump = 0.02 * np.cos(2.0 * V[:, 2] * np.pi) * np.cos(
        2.0 * np.arctan2(V[:, 1], V[:, 0]))
    Vp = V + bump[:, None] * vn
    Vp[fixed] = V[fixed]
    out = {}
    for name in ("rnf", "lbfgs"):
        W = Vp.copy()
        t0 = time.perf_counter()
        if name == "rnf":
            plateau.relax_normal_flow(W, T, fixed, iters=60)
        else:
            lbfgs_area_descent(W, T, fixed, iters=60)
        dt = time.perf_counter() - t0
        # tangential drift: displacement from the perturbed state
        # minus its component along the (reference) vertex normal
        disp = W - Vp
        tang = disp - np.einsum('nj,nj->n', disp, vn)[:, None] * vn
        out[name] = {
            "t": round(dt, 3),
            "H_rms": M.mean_curvature_residual(W, T, free=~fixed)["H_rms"],
            "area_rel_err": (M.mesh_area(W, T)
                             - C.catenoid_area_analytic()[0])
            / C.catenoid_area_analytic()[0],
            "tangential_drift_max": float(
                np.max(np.linalg.norm(tang, axis=1))),
        }
        print(f"B polish {name}:", json.dumps(out[name]), flush=True)
    return out


# ---------------------------------------------------------------- C --

def _load_roundembed():
    import importlib.util
    cands = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            HERE))), "research", "highgenus_embedder"),
        r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\research\highgenus_embedder",
    ]
    home = next((p for p in cands
                 if os.path.exists(os.path.join(p, "roundembed.py"))),
                None)
    if home is None:
        return None, None
    spec = importlib.util.spec_from_file_location(
        "roundembed", os.path.join(home, "roundembed.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, home


def part_c():
    re_mod, home = _load_roundembed()
    if re_mod is None:
        return {"skipped": "research/highgenus_embedder not found"}
    data = json.load(open(os.path.join(home, "maps_combinatorics.json")))
    rec = data["Heptagonal Dodecahedron B"]
    Vn = len(rec["V"]) if not isinstance(rec["V"], int) else rec["V"]
    F = [list(f) for f in rec["F"]]
    Lap = re_mod.laplacian(Vn, F)
    w, vec = np.linalg.eigh(Lap)
    cl = re_mod.clusters(w)
    sub = next((c[:3] for c in cl if len(c) >= 3 and w[c[0]] > 1e-6),
               None)
    X0 = vec[:, sub].copy()

    def normalize(X):
        Xc = X - X.mean(0)
        return Xc / np.sqrt((Xc ** 2).sum(1).mean())

    def energy_grad(X):
        """Sum of squared least-squares-plane deviations and its exact
        gradient (envelope theorem: the SVD plane is the per-face
        minimizer, so its variation contributes nothing)."""
        E = 0.0
        G = np.zeros_like(X)
        for f in F:
            P = X[f]
            c, nrm = re_mod.face_plane(P)
            d = (P - c) @ nrm
            E += float(d @ d)
            G[f] += 2.0 * d[:, None] * nrm[None, :]
        return E, G

    out = {}
    # baseline fixed-point loop, planar_dev vs iteration/time
    X = X0.copy()
    t0 = time.perf_counter()
    tr = []
    for chunk in range(10):
        X = re_mod.planarize(X, F, iters=50)
        tr.append((50 * (chunk + 1), round(time.perf_counter() - t0, 3),
                   re_mod.planar_dev(X, F)))
    out["baseline"] = {"trace": tr, "final_dev": tr[-1][2],
                       "aspect": re_mod.aspect(X)}
    print("C planarize baseline:", json.dumps(out["baseline"]),
          flush=True)

    # L-BFGS on the plane-deviation energy with the same normalization
    # retraction; gradient projected off the scaling direction
    X = normalize(X0.copy())
    lb = sd.LBFGS(m=8)
    x_prev = None
    g_prev = None
    E, G = energy_grad(X)
    t0 = time.perf_counter()
    tr = []
    for it in range(1, 501):
        Xf = X.ravel()
        xdir = Xf / np.linalg.norm(Xf)
        g = G.ravel()
        g = g - (g @ xdir) * xdir            # scaling mode is fixed
        g = g.reshape(-1, 3)
        g -= g.mean(0)                       # translation mode too
        g = g.ravel()
        if x_prev is not None:
            lb.push(Xf - x_prev, g - g_prev)
        d = -lb.direction(g)
        slope = float(g @ d)
        if not (slope < 0.0):
            lb.reset()
            d = -g
            slope = float(g @ d)

        def efun(xf):
            return energy_grad(normalize(xf.reshape(-1, 3)))[0]

        x1, s, E1, _ne = sd.armijo_backtrack(efun, Xf, d, slope, E0=E)
        if s == 0.0:
            lb.reset()
            x1, s, E1, _ne = sd.parabola_line_search(efun, Xf, -g,
                                                     1e-2)
            if s == 0.0:
                break
        x_prev = Xf.copy()
        g_prev = g
        X = normalize(x1.reshape(-1, 3))
        E, G = energy_grad(X)
        if it % 50 == 0:
            tr.append((it, round(time.perf_counter() - t0, 3),
                       re_mod.planar_dev(X, F)))
        if E < 1e-30:
            tr.append((it, round(time.perf_counter() - t0, 3),
                       re_mod.planar_dev(X, F)))
            break
    out["lbfgs"] = {"trace": tr, "final_dev": re_mod.planar_dev(X, F),
                    "aspect": re_mod.aspect(X),
                    "crossings": re_mod.crossings(X, F)}
    print("C planarize lbfgs:", json.dumps(out["lbfgs"]), flush=True)
    return out


# ---------------------------------------------------------------- D --

def part_d():
    out = {}
    for iters in (150, 300, 600):
        res = C.case_crochet({"crochet_iters": iters})
        out[f"iters_{iters}"] = {
            "edge_err_rms": res["metrics"]["edge_err_rms"],
            "edge_err_max": res["metrics"]["edge_err_max"],
            "t": round(res["time_s"], 2),
        }
        print(f"D crochet {iters}:", json.dumps(out[f'iters_{iters}']),
              flush=True)
    return out


def main():
    results = {"A_plateau": part_a(), "B_polish": part_b(),
               "C_planarize": part_c(), "D_crochet": part_d()}
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
