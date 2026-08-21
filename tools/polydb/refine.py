# Refine a vertex-transitive solid to machine precision.
#
# The snub uniform polyhedra reach this database through stored coordinate
# tables that carry only six decimal places, which leaves their faces out of
# plane by ~1e-7 -- far too coarse for a database that claims exactness, and
# too coarse for the exact-form recovery to work at all.
#
# They do not need to be stored precisely, because they are DETERMINED. A snub
# is vertex-transitive under a purely rotational group, so the whole solid is
# the orbit of a single generator point, and the snub condition is simply that
# every edge come out the same length. That is three unknowns and many
# equations, so it can be solved to machine precision from a rough start.
#
# The one subtlety is that the rotation group recovered from six-digit
# coordinates is itself only good to six digits. Composing 60 such matrices
# would not converge. So each matrix is SNAPPED to its exact form first: the
# octahedral group's matrices have entries in {0, +-1}, and the icosahedral
# group's in {0, +-1/2, +-phi/2, +-1/(2phi), +-1}. Snapping is only accepted
# when every entry is close to one of those values and the result is still
# orthogonal, so a bad snap fails loudly instead of silently biasing the solve.

import math

import numpy as np

from . import symmetry as SY

PHI = (1.0 + math.sqrt(5.0)) / 2.0

# the entries an exact octahedral or icosahedral rotation matrix can have
_OCTA = (0.0, 1.0, -1.0)
_ICOSA = (0.0, 0.5, -0.5, PHI / 2, -PHI / 2, 1.0 / (2 * PHI), -1.0 / (2 * PHI),
          1.0, -1.0)


def snap_group(G, tol=1e-4):
    """Snap rotation matrices to their exact entries. Returns None if any
    matrix cannot be snapped, or if snapping breaks orthogonality."""
    for palette in (_OCTA, _ICOSA):
        out = []
        ok = True
        for R in G:
            S = np.empty((3, 3))
            for i in range(3):
                for j in range(3):
                    best = min(palette, key=lambda v: abs(v - R[i, j]))
                    if abs(best - R[i, j]) > tol:
                        ok = False
                        break
                    S[i, j] = best
                if not ok:
                    break
            if not ok:
                break
            if np.max(np.abs(S.T @ S - np.eye(3))) > 1e-9:
                ok = False
                break
            out.append(S)
        if ok and out:
            return out
    return None


def _orbit(v0, G, nd=6):
    """Orbit of v0, deduplicated, in a deterministic order."""
    seen, pts = {}, []
    for R in G:
        p = R @ v0
        k = tuple(round(float(x), nd) + 0.0 for x in p)
        if k in seen:
            continue
        seen[k] = len(pts)
        pts.append(p)
    return np.array(pts)


def refine_vertex_transitive(V, F, max_nfev=400):
    """Re-solve a vertex-transitive solid so every edge has the same length.

    Returns (V_refined, info) or (None, reason). The refined vertices are in
    the SAME index order as the input, so the face table stays valid.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None, "scipy unavailable"

    A = np.asarray([[float(c) for c in v] for v in V])
    A = A - A.mean(axis=0)

    # The input carries only six good decimals, so the group must be detected
    # at a matching tolerance: at the default 1e-9 key most elements fail to
    # match and the group silently comes back as a subgroup of itself.
    S = None
    for nd, tol in ((4, 1e-4), (3, 1e-3), (5, 1e-5)):
        G = SY.find_group(A, nd=nd, tol=tol)
        rot = [R for R in G if np.linalg.det(R) > 0]
        cand = snap_group(rot)
        if cand is not None and len(cand) >= len(A):
            S = cand
            break
    if S is None:
        return None, "rotation group not recovered (best order %d for %d vertices)" % (
            len(snap_group(rot) or []), len(A))

    # edges, from the face table
    E = set()
    for f in F:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            E.add((min(a, b), max(a, b)))
    E = sorted(E)

    v0 = A[0].copy()

    def rebuild(p):
        pts = _orbit(p, S)
        if len(pts) != len(A):
            return None
        # match regenerated points back to the original index order
        idx = []
        used = set()
        for target in A:
            d = np.linalg.norm(pts - target, axis=1)
            order = np.argsort(d)
            pick = next((int(k) for k in order if int(k) not in used), None)
            if pick is None:
                return None
            used.add(pick)
            idx.append(pick)
        return pts[idx]

    def resid(p):
        W = rebuild(p)
        if W is None:
            return np.full(len(E) + 1, 1e3)
        L = np.array([np.linalg.norm(W[a] - W[b]) for a, b in E])
        return np.append(L - L.mean(), (np.linalg.norm(p) - 1.0) * 10.0)

    sol = least_squares(resid, v0, xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        max_nfev=max_nfev)
    W = rebuild(sol.x)
    if W is None:
        return None, "orbit lost the vertex count during refinement"
    L = np.array([np.linalg.norm(W[a] - W[b]) for a, b in E])
    spread = float((L.max() - L.min()) / L.max())
    if spread > 1e-12:
        return None, "did not converge (edge spread %.2e)" % spread
    return [tuple(float(x) for x in p) for p in W], {
        "edge_spread": spread, "group_order": len(S)}


def planarity(V, F):
    worst = 0.0
    for f in F:
        vs = [V[i] for i in f]
        n = [0.0, 0.0, 0.0]
        for p, q in zip(vs, vs[1:] + vs[:1]):
            n[0] += (p[1] - q[1]) * (p[2] + q[2])
            n[1] += (p[2] - q[2]) * (p[0] + q[0])
            n[2] += (p[0] - q[0]) * (p[1] + q[1])
        ln = math.sqrt(sum(x * x for x in n))
        if ln < 1e-14:
            continue
        n = [c / ln for c in n]
        d = sum(n[i] * vs[0][i] for i in range(3))
        worst = max(worst, max(abs(sum(n[i] * p[i] for i in range(3)) - d)
                               for p in vs))
    return worst
