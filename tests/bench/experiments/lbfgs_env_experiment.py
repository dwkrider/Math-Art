# Experiment: does the L-BFGS branch close the ENVIRONMENT MARGINALITY
# of the perturbed nphi=96 triple bubble?
#
# Background (gravity-walls-triple-bubble plan section 2.3, finding 2):
# under the default CG optimizer, the mildly perturbed (1.02, 0.99)
# fine triple bubble at 2400 iterations reproducibly reached DIFFERENT
# near-equilibria under different BLAS-threading configurations
# (0.0233 deg vs 0.2165 deg fitted 120-rms) -- sub-ulp reduction-order
# differences amplified through groom flip decisions.  The committed
# bubble_triple_fine case dodges this by starting unperturbed.
#
# This experiment runs the PERTURBED fine triple under each optimizer
# in child processes whose BLAS thread count is forced to 1 / 4
# before numpy is imported (the controllable analogue of the two
# historical environments), and reports the fitted angle rms and a
# bitwise hash of the final vertex array per environment.
#
#   python tests/bench/experiments/lbfgs_env_experiment.py            # driver
#   python ... lbfgs_env_experiment.py --child <optimizer> <iters>    # child
#
# Interpretation contract, fixed before running: the marginality is
# CLOSED by an optimizer if every environment lands within the
# committed unperturbed floor (~0.023 deg) of the same equilibrium;
# bitwise-identical hashes across thread counts are NOT expected of
# either optimizer (threaded BLAS reductions are not order-stable) and
# are reported for completeness only.
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # tests/


def child(optimizer, iters):
    sys.path.insert(0, ROOT)
    from bench import cases as C
    from bench import metrics as M
    import numpy as np
    from math_art.bubble_generator import (build_triple_bubble_mesh,
                                           triple_bubble_geometry)
    import time

    geo = triple_bubble_geometry(1.0)
    V, T, labels = build_triple_bubble_mesh(1.0, nphi=96)
    T = np.ascontiguousarray(T)
    V *= np.array([1.02, 0.99, 1.0])
    kwargs = {"iters": iters, "groom_every": 4}
    if optimizer == "lbfgs":
        kwargs["optimizer"] = "lbfgs"
    t0 = time.perf_counter()
    info, _eff = C._evolve_checked(V, T, labels,
                                   [geo["V_cell"]] * 3, kwargs)
    dt = time.perf_counter() - t0
    fits = M.film_fits(V, T, labels)
    ang = M.fitted_triple_angles(V, T, labels, fits)
    out = {
        "optimizer": optimizer,
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "iters_run": info["iters_run"],
        "time_s": dt,
        "angle_rms_fit": float(np.sqrt(np.mean((ang - 120.0) ** 2))),
        "area": float(info["area"]),
        "vhash": hashlib.sha256(np.ascontiguousarray(V).tobytes())
                 .hexdigest()[:16],
        "vol_drift_max": max(h["drift_post"] for h in info["history"]),
        "rise_max": max(h["rise"] for h in info["history"]
                        if not h["groomed"]),
    }
    print("CHILD_RESULT " + json.dumps(out), flush=True)


def main():
    results = []
    for optimizer, iters in (("cg", 2400), ("lbfgs", 2400)):
        for threads in (1, 4):
            env = dict(os.environ)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS"):
                env[k] = str(threads)
            print(f"[env-exp] {optimizer} threads={threads} ...",
                  flush=True)
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 "--child", optimizer, str(iters)],
                env=env, capture_output=True, text=True)
            line = next((ln for ln in p.stdout.splitlines()
                         if ln.startswith("CHILD_RESULT ")), None)
            if line is None:
                print(p.stdout)
                print(p.stderr)
                raise SystemExit(f"child failed ({optimizer}, {threads})")
            rec = json.loads(line[len("CHILD_RESULT "):])
            rec["threads"] = threads
            results.append(rec)
            print(f"[env-exp]   rms={rec['angle_rms_fit']:.4f} deg  "
                  f"iters={rec['iters_run']}  t={rec['time_s']:.1f}s  "
                  f"vhash={rec['vhash']}", flush=True)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]))
    else:
        main()
