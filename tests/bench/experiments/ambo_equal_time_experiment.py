# Experiment: antiprism circle-packing ("ambo") canonicalization vs the
# shipped adaptive-gain Hart relaxation, judged ON THE WALL-CLOCK CURVE
# (the rule that rejected the parabola line search in this same file's
# sibling canon_ls_experiment.py: per-iteration wins are worthless if
# each iteration is slower).
#
#   python tests/bench/experiments/ambo_equal_time_experiment.py
#
# Protocol per catalog solid (gC pC wC tI kD):
#   1. incumbent: canonicalize(adaptive=True), the bench's 400-iter
#      budget -> wall time T_hart, tangency spread S_hart.
#   2. candidate: canonicalize_ambo run to its own convergence
#      (tol 1e-12) -> T_ambo, S_ambo, iterations used.
#   3. equal wall time, both directions:
#      - ambo given T_hart: it converges long before the budget, so its
#        equal-time spread IS S_ambo (stated, not extrapolated);
#      - Hart given T_ambo: iteration budget round(400 * T_ambo/T_hart)
#        (>= 1), the same budget-matching scheme canon_ls used.
# Quality numbers are deterministic; times are quoted for ratios.
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "math_art"))
pkg = types.ModuleType('math_art')
pkg.__path__ = [sys.path[0]]
sys.modules['math_art'] = pkg

import numpy as np
from math_art.polyhedra import canonical
from math_art.polyhedra.conway import apply_conway

SOLIDS = ("gC", "pC", "wC", "tI", "kD")


def edge_points(V, F):
    P = np.asarray(V, float)
    pts = []
    for a, b in sorted({tuple(sorted((f[i], f[(i + 1) % len(f)])))
                        for f in F for i in range(len(f))}):
        A, B = P[a], P[b]
        AB = B - A
        t = min(1.0, max(0.0,
                         -np.dot(A, AB) / max(np.dot(AB, AB), 1e-30)))
        pts.append(A + t * AB)
    return np.asarray(pts)


def tangent_spread(V, F):
    d = np.linalg.norm(edge_points(V, F), axis=1)
    return float((d.max() - d.min()) / max(d.mean(), 1e-12))


def abs_tangency_resid(V, F):
    """max |dist(edge nearest point -> origin) - 1| (unit midsphere)."""
    d = np.linalg.norm(edge_points(V, F), axis=1)
    return float(np.max(np.abs(d - 1.0)))


def centroid_off(V, F):
    return float(np.linalg.norm(edge_points(V, F).mean(axis=0)))


def planarity_max(V, F):
    P = np.asarray(V, float)
    worst = 0.0
    for f in F:
        Q = P[list(f)]
        c = Q.mean(axis=0)
        Qc = Q - c
        _w, vv = np.linalg.eigh(Qc.T @ Qc)
        worst = max(worst, float(np.abs(Qc @ vv[:, 0]).max()))
    return worst


def main():
    rows = []
    for text in SOLIDS:
        V, F = apply_conway(text)

        t0 = time.perf_counter()
        Vh = canonical.canonicalize(V, F, iters=400)
        t_hart = time.perf_counter() - t0
        s_hart = tangent_spread(Vh, F)

        snaps = []
        t0 = time.perf_counter()
        Va = canonical.canonicalize_ambo(
            V, F, iters=10000, trace=lambda it, P: snaps.append(it))
        t_ambo = time.perf_counter() - t0
        s_ambo = tangent_spread(Va, F)
        n_ambo = len(snaps)

        # Hart at ambo's wall budget (iteration-budget matching)
        n_eq = max(1, round(400 * t_ambo / max(t_hart, 1e-9)))
        Vh_eq = canonical.canonicalize(V, F, iters=n_eq)
        s_hart_eq = tangent_spread(Vh_eq, F)

        rows.append({
            "solid": text,
            "s_hart": s_hart, "t_hart": t_hart,
            "s_ambo": s_ambo, "t_ambo": t_ambo, "n_ambo": n_ambo,
            "n_hart_eq": n_eq, "s_hart_eq": s_hart_eq,
            "abs_resid_ambo": abs_tangency_resid(Va, F),
            "cen_ambo": centroid_off(Va, F),
            "plan_ambo": planarity_max(Va, F),
            "plan_hart": planarity_max(Vh, F),
        })

    print("\nUnlimited (each to its own budget/convergence):\n")
    print("| solid | adaptive spread (time) | ambo spread (time, iters) |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['solid']} | {r['s_hart']:.2e} ({r['t_hart']:.2f}s) "
              f"| {r['s_ambo']:.2e} ({r['t_ambo']:.3f}s, "
              f"{r['n_ambo']}) |")

    print("\nAt EQUAL WALL TIME (the deciding curve):\n")
    print("| solid | Hart @ ambo budget (iters) | ambo | verdict |")
    print("|---|---|---|---|")
    for r in rows:
        ratio = r["s_hart_eq"] / max(r["s_ambo"], 1e-300)
        print(f"| {r['solid']} | {r['s_hart_eq']:.2e} "
              f"({r['n_hart_eq']}) | {r['s_ambo']:.2e} "
              f"| ambo {ratio:.1e}x better |")

    print("\nGround-truth residuals of the ambo results "
          "(unit midsphere):\n")
    print("| solid | max|d-1| | centroid offset | planarity "
          "| Hart planarity |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['solid']} | {r['abs_resid_ambo']:.2e} "
              f"| {r['cen_ambo']:.2e} | {r['plan_ambo']:.2e} "
              f"| {r['plan_hart']:.2e} |")


if __name__ == "__main__":
    main()
