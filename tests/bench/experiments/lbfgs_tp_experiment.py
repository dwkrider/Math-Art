# Experiment: what does the L-BFGS core buy the tangent-point knot
# flow?  MEASUREMENT ONLY -- knots/tangent_point.py is not modified;
# this script drives its public pieces (tp_energy / tp_gradient /
# _saddle_inverse / _constraint_rows) from an outer L-BFGS loop and
# compares against the reference tighten() flow.
#
# Two questions, answered separately because they are different kinds
# of "convergence":
#
# 1. DISCRETIZATION order ("first-order energy convergence" in the S7
#    plan's honest-limits list): the discrete energy's O(1/n) error
#    against the continuum value comes from the vertex quadrature's
#    omitted diagonal band.  No outer optimizer appears anywhere in
#    that number -- the committed conv_order metric is computed from
#    tp_energy() of the round SEED, without running any flow at all --
#    so the expected answer is "unchanged", and this experiment
#    verifies it end-to-end by comparing the CONVERGED trefoil energies
#    at n and 2n under both optimizers.
#
# 2. OPTIMIZATION convergence: iterations / wall seconds to reach
#    given energy levels on the tight trefoil, reference H^s descent
#    vs L-BFGS seeded with the same H^s solve (the saddle inverse as
#    H0, curvature pairs from raw gradient differences).  The
#    reference step is already Newton-like (the survey's point), so
#    the honest prior is a modest gain at best.
#
#   python tests/bench/experiments/lbfgs_tp_experiment.py
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # tests/
from bench import _boot  # noqa: F401,E402
import numpy as np  # noqa: E402
from math_art.knots import tangent_point as tpm  # noqa: E402
from math_art.knots.braid import (braid_closure_points,  # noqa: E402
                                  parse_letters)
from math_art.knots.resample import resample_closed  # noqa: E402
from math_art.solver import descent as sd  # noqa: E402


def tighten_lbfgs(P, iters=300, m=8, length_mode="edge",
                  backproj_tol=1e-6, grad_tol=1e-10):
    """The reference flow's step machinery with an L-BFGS model on
    top: H0 = the current bordered H^s solve, curvature pairs from raw
    gradient differences, Armijo from the capped unit step, and the
    reference's own Newton backprojection (reusing the factorization).
    Structured exactly like tighten() so per-iteration cost is
    comparable; nothing in knots/tangent_point.py is changed."""
    x = np.asarray(P, dtype=float).copy()
    n = len(x)
    _C0, targets = tpm._constraint_rows(x, length_mode)
    lb = sd.LBFGS(m=m)
    hist = []
    E = tpm.tp_energy(x)
    x_prev = None
    g_prev = None
    t0 = time.perf_counter()
    for it in range(int(iters)):
        dE = tpm.tp_gradient(x).ravel()
        Ainv, mrows = tpm._saddle_inverse(x, 3.0, 6.0, length_mode,
                                          True, 0.0)

        def h0(q):
            rhs = np.zeros(3 * n + mrows)
            rhs[:3 * n] = q
            return (Ainv @ rhs)[:3 * n]

        if x_prev is not None:
            lb.push(x.ravel() - x_prev, dE - g_prev)
        g_sob = h0(dE)
        if float(np.linalg.norm(g_sob)) < grad_tol:
            break
        d = -lb.direction(dE, h0=h0)
        slope = float(dE @ d)
        if not (slope < 0.0):
            lb.reset()
            d = -g_sob
            slope = float(dE @ d)
        dmax = float(np.max(np.linalg.norm(d.reshape(n, 3), axis=1)))
        gap = tpm._min_gap(x)
        s_max = 0.45 * gap / max(dmax, 1e-300)   # the tunneling cap

        def energy_flat(xf):
            return tpm.tp_energy(xf.reshape(n, 3))

        xf = x.ravel()
        _xn, s_used, _En, _ne = sd.armijo_backtrack(
            energy_flat, xf, d, slope, s0=min(1.0, s_max), E0=E,
            s_max=s_max)
        if s_used == 0.0:
            lb.reset()
            _xn, s_used, _En, _ne = sd.parabola_line_search(
                energy_flat, xf, -g_sob,
                min(1.0, 0.45 * gap
                    / max(float(np.max(np.linalg.norm(
                        g_sob.reshape(n, 3), axis=1))), 1e-300)))
            d = -g_sob
        if s_used == 0.0:
            break
        # reference backprojection, verbatim logic
        viol = np.inf
        for _attempt in range(8):
            y = (xf + s_used * d).reshape(n, 3).copy()
            for _newton in range(3):
                _Cy, vals = tpm._constraint_rows(y, length_mode)
                phi = targets - vals
                viol = float(np.max(np.abs(phi)))
                if viol < backproj_tol:
                    break
                rhs2 = np.zeros(3 * n + mrows)
                rhs2[3 * n:] = phi
                y += (Ainv @ rhs2)[:3 * n].reshape(n, 3)
            _Cy, vals = tpm._constraint_rows(y, length_mode)
            viol = float(np.max(np.abs(targets - vals)))
            if viol < backproj_tol:
                break
            s_used *= 0.5
            if s_used < 1e-15:
                y = x.copy()
                break
        x_prev = xf.copy()
        g_prev = dE
        E_new = tpm.tp_energy(y)
        hist.append({"it": it + 1, "E": E_new, "s": s_used,
                     "viol": viol, "t": time.perf_counter() - t0,
                     "rise": max(0.0, E_new - E)})
        x = y
        E = E_new
    return x, {"history": hist, "E": E, "iters_run": len(hist),
               "skips": lb.skips, "resets": lb.resets}


def trefoil(n):
    return resample_closed(braid_closure_points(parse_letters('AAA')), n)


def run_ref(P, iters):
    t0 = time.perf_counter()
    times = []
    x, info = tpm.tighten(P, iters=iters,
                          callback=lambda it, xx: times.append(
                              time.perf_counter() - t0))
    for k, h in enumerate(info["history"]):
        h["t"] = times[k]
    return x, info


def levels_table(hist, levels):
    out = {}
    for lev in levels:
        hit = next((h for h in hist if h["E"] <= lev), None)
        out[str(lev)] = (None if hit is None
                         else {"it": hit["it"], "t": round(hit["t"], 2)})
    return out


def main():
    results = {}
    levels = (100.0, 60.0, 47.2, 47.18216)
    for n in (96, 192):
        P = trefoil(n)
        x_r, info_r = run_ref(P.copy(), 300)
        x_l, info_l = tighten_lbfgs(P.copy(), iters=300)
        results[f"trefoil_n{n}"] = {
            "ref": {"E": info_r["E"], "iters": info_r["iters_run"],
                    "t": round(info_r["history"][-1]["t"], 2),
                    "rise_max": max(h["rise"]
                                    for h in info_r["history"]),
                    "levels": levels_table(info_r["history"], levels)},
            "lbfgs": {"E": info_l["E"], "iters": info_l["iters_run"],
                      "t": round(info_l["history"][-1]["t"], 2),
                      "rise_max": max(h["rise"]
                                      for h in info_l["history"]),
                      "skips": info_l["skips"],
                      "levels": levels_table(info_l["history"], levels)},
        }
        print(f"n={n}: ref E={info_r['E']:.6f} "
              f"({info_r['iters_run']} it, "
              f"{results[f'trefoil_n{n}']['ref']['t']}s)  "
              f"lbfgs E={info_l['E']:.6f} ({info_l['iters_run']} it, "
              f"{results[f'trefoil_n{n}']['lbfgs']['t']}s)", flush=True)
    # discretization order from the CONVERGED energies: the continuum
    # tight-trefoil energy is unknown, so quote the n -> 2n energy gap
    # (first-order band => the gap halves when n doubles; identical
    # between optimizers => the order is an optimizer-independent
    # quadrature property).  Plus the committed circle-seed orders.
    E96r = results["trefoil_n96"]["ref"]["E"]
    E192r = results["trefoil_n192"]["ref"]["E"]
    E96l = results["trefoil_n96"]["lbfgs"]["E"]
    E192l = results["trefoil_n192"]["lbfgs"]["E"]
    E_exact = math.pi ** 2 / 2.0
    circ = {}
    for n in (32, 64, 128, 256):
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        C = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        circ[n] = abs(tpm.tp_energy(C) - E_exact) / E_exact
    results["discretization"] = {
        "trefoil_E_gap_ref": E96r - E192r,
        "trefoil_E_gap_lbfgs": E96l - E192l,
        "circle_rel_err": circ,
        "circle_order_last": math.log2(circ[128] / circ[256]),
        "note": "the circle numbers involve NO optimizer at all: "
                "first-order convergence is the quadrature's, and no "
                "outer loop can change it",
    }
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
