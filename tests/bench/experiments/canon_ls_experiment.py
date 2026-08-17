# Experiment: parabola line search on the Hart canonicalize update,
# vs the adaptive-gain default.  Prototype only -- shipped iff it wins.
import sys, time, types
sys.path.insert(0, r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\.claude\worktrees\solver-unification-bench\math_art")
pkg = types.ModuleType('math_art'); pkg.__path__ = [sys.path[0]]
sys.modules['math_art'] = pkg
import numpy as np
from math_art.polyhedra import canonical
from math_art.polyhedra.conway import apply_conway
from math_art.solver import descent


def tangent_spread(V, F):
    P = np.asarray(V, float)
    d = []
    for a, b in sorted({tuple(sorted((f[i], f[(i + 1) % len(f)])))
                        for f in F for i in range(len(f))}):
        A, B = P[a], P[b]
        AB = B - A
        t = min(1.0, max(0.0, -np.dot(A, AB) / max(np.dot(AB, AB), 1e-30)))
        d.append(np.linalg.norm(A + t * AB))
    d = np.asarray(d)
    return float((d.max() - d.min()) / max(d.mean(), 1e-12))


def canon_energy(P, E, Fi):
    A, B = P[E[:, 0]], P[E[:, 1]]
    d = B - A
    t = np.clip(-np.einsum('ij,ij->i', A, d)
                / np.maximum(np.einsum('ij,ij->i', d, d), 1e-12), 0, 1)
    C = A + t[:, None] * d
    cl = np.linalg.norm(C, axis=1)
    Ee = float(np.sum((1.0 - cl) ** 2)) + float(np.sum(C.mean(0) ** 2))
    for f in Fi:
        Q = P[f]
        c = Q.mean(0)
        Qc = Q - c
        n = np.zeros(3)
        for i in range(len(f)):
            n += np.cross(Qc[i], Qc[(i + 1) % len(f)])
        ln = np.linalg.norm(n)
        if ln > 1e-12:
            off = Qc @ (n / ln)
            Ee += float(np.sum(off * off))
    return Ee


def canonicalize_ls(V, F, iters=400):
    P = np.array(V, float)
    P -= P.mean(0)
    P /= np.mean(np.linalg.norm(P, axis=1))
    E = np.array(sorted({tuple(sorted((f[i], f[(i + 1) % len(f)])))
                         for f in F for i in range(len(f))}))
    Fi = [np.array(f) for f in F]
    s_prev = 1.0
    for it in range(iters):
        # Hart update direction at unit gains
        A, B = P[E[:, 0]], P[E[:, 1]]
        d = B - A
        t = np.clip(-np.einsum('ij,ij->i', A, d)
                    / np.maximum(np.einsum('ij,ij->i', d, d), 1e-12), 0, 1)
        C = A + t[:, None] * d
        cen = C.mean(0)
        C = C - cen
        cl = np.linalg.norm(C, axis=1, keepdims=True)
        corr = C / np.maximum(cl, 1e-9) * (1.0 - cl)
        adj = np.zeros_like(P)
        cnt = np.zeros(len(P))
        np.add.at(adj, E[:, 0], corr)
        np.add.at(adj, E[:, 1], corr)
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        D = -np.tile(cen, (len(P), 1)) + adj / np.maximum(cnt, 1)[:, None]
        for f in Fi:
            Q = P[f]
            c = Q.mean(0)
            Qc = Q - c
            n = np.zeros(3)
            for i in range(len(f)):
                n += np.cross(Qc[i], Qc[(i + 1) % len(f)])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n /= ln
            D[f] -= (Qc @ n)[:, None] * n
        P, s_used, En, _ = descent.parabola_line_search(
            lambda Q: canon_energy(Q, E, Fi), P, D, s0=max(s_prev, 1e-6),
            s_max=2.0)
        if s_used == 0.0:
            break
        s_prev = s_used
        if s_used * float(np.max(np.abs(D))) < 1e-9:
            break
    return [list(map(float, p)) for p in P], it + 1


for text in ('gC', 'pC', 'wC', 'tI', 'kD'):
    V, F = apply_conway(text)
    t0 = time.perf_counter()
    Va = canonical.canonicalize(V, F, iters=400)          # adaptive default
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    Vl, nl = canonicalize_ls(V, F, iters=400)
    tl = time.perf_counter() - t0
    print(f"{text}: adaptive spread={tangent_spread(Va, F):.3e} ({ta:.2f}s)"
          f"  LS spread={tangent_spread(Vl, F):.3e} ({tl:.2f}s, {nl} it)")
