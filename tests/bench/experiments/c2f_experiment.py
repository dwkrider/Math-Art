# S5 coarse-to-fine continuation: the wall-clock evidence.
#
# Three site studies, each writing a markdown table under
# tests/bench/results/ (gitignored; regeneration = re-run this script):
#
#   1. crochet   -- one-shot budget sweep vs continuation (+ thin-plate
#                   bending): does ANY one-shot budget reach the c2f
#                   quality?  (results/c2f_crochet_sweep.md)
#   2. plateau   -- external composition (coarse minimize_area ->
#                   solver.refine.subdivide with boundary projection ->
#                   short fine re-solve) vs the direct fine solve, on
#                   the analytic catenoid.  plateau.py itself is NOT
#                   modified.  (results/c2f_plateau.md)
#   3. untangle  -- the high-genus embedder's hard maps, planarize
#                   alone vs + roundembed.untangle cycles.
#                   (results/c2f_untangle_maps.md; the 671-second
#                   genus-5 row only with --slow)
#
# Deterministic apart from wall clock; decisions quote ratios.
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # tests/
from bench import _boot  # noqa: F401,E402
from bench import cases as C  # noqa: E402
from bench import metrics as M  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS, exist_ok=True)


def crochet_sweep():
    from math_art.hyperbolic import crochet
    ratio_n, rows, stitch, max_st = 4, 16, 0.09, 600
    R0 = stitch / math.log(1.0 + 1.0 / ratio_n)
    Kt = -1.0 / R0 ** 2

    def measure(P, E0, E1, REST, tris, dt, label):
        rel = M.edge_metric_error(P, E0, E1, REST)
        dih = M.dihedral_rms_deg(P, tris)
        K = crochet.mean_curvature(P, tris)
        sx = M.selfx_count(P, np.asarray(tris, np.int64))
        z = float(P[:, 2].max() - P[:, 2].min())
        return dict(label=label, t=dt, err=rel["edge_err_rms"],
                    emax=rel["edge_err_max"], dih=dih,
                    Krel=abs(K - Kt) / abs(Kt), selfx=sx, z=z,
                    nv=len(P))

    rowsout = []
    for iters in (170, 340, 680, 1360):
        for bend in (0.0, 0.02):
            t0 = time.perf_counter()
            P, E0, E1, REST, nbr, tris, pin, pinp = crochet._crochet_mesh(
                ratio_n, rows, stitch, max_st)
            P = crochet._relax(P, E0, E1, REST, nbr, pin, pinp, iters,
                               0.06, 0.4 * stitch, 0.5, bend=bend)
            rowsout.append(measure(P, E0, E1, REST, tris,
                                   time.perf_counter() - t0,
                                   f"oneshot it={iters} bend={bend}"))
    for levels in (1, 2):
        for bend in (0.0, 0.02, 0.04):
            t0 = time.perf_counter()
            r = crochet.crochet_c2f(
                ratio_n=ratio_n, rows=rows, stitch=stitch,
                max_stitches=max_st, iters=340, smooth=0.06, repel=0.5,
                bend=bend, levels=levels)
            rowsout.append(measure(r["P"], r["E0"], r["E1"], r["REST"],
                                   r["tris"], time.perf_counter() - t0,
                                   f"c2f L={levels} it=340 bend={bend}"))
    path = os.path.join(RESULTS, "c2f_crochet_sweep.md")
    with open(path, "w") as f:
        f.write("# Crochet: one-shot budget sweep vs coarse-to-fine\n\n"
                "Sheet: ratio 4, rows 16, stitch 0.09, cap 600 "
                "(RUFFLED-scale, the documented buckler); smooth 0.06, "
                "repel 0.5 everywhere.  err = hyperbolic edge "
                "distortion rms vs the mesh's exact rest lengths; dih "
                "= RMS dihedral (deg, crumple signature); Krel = "
                "|median K - Ktarget|/|Ktarget|.\n\n"
                "| config | t (s) | err_rms | err_max | dih_rms | "
                "K_rel | selfx | z | V |\n"
                "|---|---|---|---|---|---|---|---|---|\n")
        for r in rowsout:
            f.write(f"| {r['label']} | {r['t']:.2f} | {r['err']:.4f} | "
                    f"{r['emax']:.3f} | {r['dih']:.1f} | "
                    f"{r['Krel']:.2f} | {r['selfx']} | {r['z']:.2f} | "
                    f"{r['nv']} |\n")
    print(f"wrote {path}")
    for r in rowsout:
        print(f"  {r['label']:28s} t={r['t']:6.2f} err={r['err']:.4f} "
              f"dih={r['dih']:5.1f} Krel={r['Krel']:.2f} "
              f"selfx={r['selfx']} z={r['z']:.2f}")


def plateau_composition():
    from math_art.minsurf import plateau
    from math_art.solver import refine
    A_true, c_true = C.catenoid_area_analytic()
    h, R = 0.5, 1.0

    def waist_err(V):
        z = V[:, 2]
        band = np.abs(z) < 0.02
        r = np.linalg.norm(V[band, :2], axis=1)
        return abs(float(np.mean(r)) - c_true) / c_true

    out = []
    # A: direct fine solve (96 x 25), default settings as in the bench
    V, T, fixed = C.cylinder_grid(96, 25)
    t0 = time.perf_counter()
    plateau.minimize_area(V, T, fixed, outer_iters=40)
    tA = time.perf_counter() - t0
    mets = M.mean_curvature_residual(V, T, free=~fixed)
    out.append(("direct fine 96x25, 40 outer", tA, waist_err(V),
                mets["H_rms"], M.mesh_area(V, T)))

    # B: coarse 48x13 solve -> subdivide (boundary ring projection) ->
    # short fine re-solve
    t0 = time.perf_counter()
    Vc, Tc, fc = C.cylinder_grid(48, 13)
    plateau.minimize_area(Vc, Tc, fc, outer_iters=30)

    def project_rings(Pn, parents):
        both_fixed = fc[parents[:, 0]] & fc[parents[:, 1]]
        same_ring = np.abs(Vc[parents[:, 0], 2]
                           - Vc[parents[:, 1], 2]) < 1e-12
        sel = both_fixed & same_ring
        if np.any(sel):
            xy = Pn[sel, :2]
            nrm = np.linalg.norm(xy, axis=1, keepdims=True)
            Pn[sel, :2] = xy / np.maximum(nrm, 1e-30) * R
            Pn[sel, 2] = np.sign(Pn[sel, 2]) * h
        return Pn

    Vf, Tf, parents = refine.subdivide(Vc, Tc, project=project_rings)
    ff = np.concatenate([fc, fc[parents[:, 0]] & fc[parents[:, 1]]
                         & (np.abs(Vc[parents[:, 0], 2]
                                   - Vc[parents[:, 1], 2]) < 1e-12)])
    for it_fine in (6,):
        Vf2 = Vf.copy()
        plateau.minimize_area(Vf2, Tf, ff, outer_iters=it_fine)
        tB = time.perf_counter() - t0
        mets = M.mean_curvature_residual(Vf2, Tf, free=~ff)
        out.append((f"c2f 48x13(30) -> refine -> fine({it_fine})", tB,
                    waist_err(Vf2), mets["H_rms"], M.mesh_area(Vf2, Tf)))

    path = os.path.join(RESULTS, "c2f_plateau.md")
    with open(path, "w") as f:
        f.write("# Plateau (catenoid): direct fine vs external "
                "coarse-to-fine composition\n\n"
                f"Analytic area {A_true:.6f}, waist c {c_true:.6f}.  "
                "plateau.py unmodified; the composition uses "
                "solver.refine.subdivide with the boundary rings "
                "projected back onto the pinned circles.\n\n"
                "| pipeline | t (s) | waist rel err | H_rms | area |\n"
                "|---|---|---|---|---|\n")
        for lab, t, we, hr, ar in out:
            f.write(f"| {lab} | {t:.2f} | {we:.2e} | {hr:.2e} | "
                    f"{ar:.6f} |\n")
    print(f"wrote {path}")
    for lab, t, we, hr, ar in out:
        print(f"  {lab:38s} t={t:5.2f} waist={we:.2e} H_rms={hr:.2e}")


def untangle_maps(slow=False):
    re_mod, home = C._load_roundembed()
    if re_mod is None:
        print("roundembed not found; skipping")
        return
    import json
    data = json.load(open(os.path.join(home, "maps_combinatorics.json")))
    names = ["Overarching Octagonal Dodecahedron",
             "Overarching Hendecagonal Dodecahedron",
             "Klein Map {7,3}_8 (Heptagonal)"]
    if slow:
        names.append("Locally Regular Map {7,3} Genus 5")
    rowsout = []
    for name in names:
        rec = data[name]
        Vn = len(rec["V"]) if not isinstance(rec["V"], int) else rec["V"]
        F = [list(f) for f in rec["F"]]
        Lap = re_mod.laplacian(Vn, F)
        w, vec = np.linalg.eigh(Lap)
        sub = None
        for c in re_mod.clusters(w):
            if len(c) >= 3 and w[c[0]] > 1e-6:
                sub = c[:3]
                break
        if sub is None:
            sub = [i for i in range(len(w)) if w[i] > 1e-6][:3]
        X = re_mod.planarize(vec[:, sub].copy(), F)
        cr0 = re_mod.crossings(X, F)
        t0 = time.perf_counter()
        Xb, crb = re_mod.untangle(X, F)
        dt = time.perf_counter() - t0
        rowsout.append((name, Vn, cr0, crb,
                        re_mod.planar_dev(Xb, F),
                        re_mod.aspect(Xb), dt))
        print(f"  {name}: {cr0} -> {crb} crossings "
              f"(pd={rowsout[-1][4]:.1e}, asp={rowsout[-1][5]:.3f}, "
              f"{dt:.0f}s)")
    path = os.path.join(RESULTS, "c2f_untangle_maps.md")
    with open(path, "w") as f:
        f.write("# High-genus embedder: crossings, planarize alone vs "
                "+ untangle cycles\n\n"
                "Deterministic first-cluster eigenseed; default "
                "untangle schedule ((1,100)x2, (2,40)x3) + terminal "
                "planarize polish.\n\n"
                "| map | V | crossings before | after | planar_dev | "
                "aspect | t (s) |\n|---|---|---|---|---|---|---|\n")
        for name, Vn, cr0, crb, pd, asp, dt in rowsout:
            f.write(f"| {name} | {Vn} | {cr0} | {crb} | {pd:.2e} | "
                    f"{asp:.3f} | {dt:.0f} |\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"crochet", "plateau", "untangle"}
    if "crochet" in which:
        crochet_sweep()
    if "plateau" in which:
        plateau_composition()
    if "untangle" in which or "untangle-slow" in which:
        untangle_maps(slow="untangle-slow" in which or "--slow" in which)
