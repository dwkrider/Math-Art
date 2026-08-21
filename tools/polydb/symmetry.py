# Point-group detection for the polyhedron database.
#
# Given a vertex table and its faces, find the full symmetry group as an
# explicit set of orthogonal matrices, classify it, and report the orbits of
# vertices, edges and faces.
#
# Everything here is COMPUTED from the geometry rather than transcribed from a
# table.  That is the point: the symmetry metadata is the part netlib lacks and
# the part most easily got wrong by hand, so it is derived and checkable.
#
# The classification follows the standard concordance between notations, for
# which the reference in this repository's own library is:
#
#   P. R. Cromwell, "Polyhedra", Cambridge (1997), Appendix I -- the table
#   relating Schoenflies, Coxeter, Shubnikov, Conway-Thurston orbifold and
#   Fejes Toth symbols.  The orbifold strings below are that table's column 4.
#   J. H. Conway and W. P. Thurston, orbifold notation for the spherical
#   groups; see J. H. Conway, H. Burgiel, C. Goodman-Strauss, "The Symmetries
#   of Things", A K Peters (2008), ch. 3.

import math
from collections import defaultdict

import numpy as np

TOL = 1e-7


def _centred(V):
    A = np.asarray(V, dtype=float)
    return A - A.mean(axis=0)


def _key(v, nd=6):
    return tuple(0.0 + round(float(x), nd) for x in v)


def find_group(V, max_elements=240, nd=6, tol=TOL, F=None):
    """Orthogonal matrices mapping the (centred) solid onto itself.

    Pass `F` whenever the faces are known. Without them this finds the
    symmetry of the VERTEX ARRANGEMENT, which can be strictly larger than the
    symmetry of the solid: the compound of five tetrahedra sits on a
    dodecahedron's twenty vertices, so its vertex set is fixed by all of Ih
    (order 120) while the compound itself is chiral (I, order 60) -- a
    reflection preserves the points but exchanges the compound with its mirror
    image. Requiring faces to map to faces is what tells the two apart.

    `nd` is the number of decimals the vertex-matching key keeps, and `tol`
    the inner-product tolerance. The defaults suit coordinates good to about
    1e-9. For COARSER input -- the stored snub tables carry only six decimals,
    which is exactly the default key's noise floor -- pass a smaller nd, or
    most group elements are silently rejected and the group comes back a
    subgroup of itself.
    """
    A = _centred(V)
    n = len(A)
    lookup = {_key(v, nd): i for i, v in enumerate(A)}
    if len(lookup) != n:
        # duplicate vertices -- fall back to a coarser key rather than lying
        lookup = {_key(v, nd - 1): i for i, v in enumerate(A)}
        nd = nd - 1

    faceset = None
    if F:
        faceset = {frozenset(f) for f in F}

    radii = np.linalg.norm(A, axis=1)
    by_r = defaultdict(list)
    for i, r in enumerate(radii):
        by_r[round(float(r), nd)].append(i)

    # A reference triple of linearly independent vertices. Prefer vertices
    # from small radius classes: fewer candidate images to try.
    order = sorted(range(n), key=lambda i: len(by_r[round(float(radii[i]), nd)]))
    ref = []
    for i in order:
        cand = ref + [i]
        if np.linalg.matrix_rank(A[cand], tol=1e-8) == len(cand):
            ref = cand
            if len(ref) == 3:
                break
    if len(ref) < 3:
        return [np.eye(3)]                      # degenerate / planar
    M = A[ref].T
    Minv = np.linalg.inv(M)
    v1, v2, v3 = A[ref[0]], A[ref[1]], A[ref[2]]
    r1, r2, r3 = (round(float(np.linalg.norm(x)), nd) for x in (v1, v2, v3))
    d12, d13, d23 = float(v1 @ v2), float(v1 @ v3), float(v2 @ v3)

    out = []
    for i in by_r.get(r1, []):
        w1 = A[i]
        for j in by_r.get(r2, []):
            if abs(float(w1 @ A[j]) - d12) > tol:
                continue
            w2 = A[j]
            for k in by_r.get(r3, []):
                if abs(float(w1 @ A[k]) - d13) > tol:
                    continue
                if abs(float(w2 @ A[k]) - d23) > tol:
                    continue
                O = np.column_stack((w1, w2, A[k])) @ Minv
                if np.max(np.abs(O.T @ O - np.eye(3))) > max(1e-6, 10 * tol):
                    continue
                img = A @ O.T
                if not all(_key(p, nd) in lookup for p in img):
                    continue
                if faceset is not None:
                    perm = [lookup[_key(p, nd)] for p in img]
                    if any(frozenset(perm[i] for i in f) not in faceset
                           for f in F):
                        continue          # preserves the points, not the solid
                out.append(O)
                if len(out) >= max_elements:
                    return out
    if not out:
        out = [np.eye(3)]
    return out


def _axis_and_order(R):
    """Rotation axis (unit, sign-normalised) and order n, for a proper R."""
    c = (np.trace(R) - 1.0) / 2.0
    ang = math.acos(max(-1.0, min(1.0, c)))
    if ang < 1e-9:
        return None, 1
    w, v = np.linalg.eig(R)
    ax = None
    for idx in range(3):
        if abs(w[idx].real - 1.0) < 1e-6 and abs(w[idx].imag) < 1e-6:
            ax = np.real(v[:, idx])
            break
    if ax is None:
        return None, 1
    ax = ax / np.linalg.norm(ax)
    for s in ax:                                # sign-normalise
        if abs(s) > 1e-8:
            if s < 0:
                ax = -ax
            break
    n = int(round(2 * math.pi / ang))
    return ax, max(1, n)


def classify(G):
    """Schoenflies symbol and friends for a group given as matrices."""
    g = len(G)
    proper = [R for R in G if np.linalg.det(R) > 0]
    r = len(proper)
    has_inv = any(np.max(np.abs(R + np.eye(3))) < 1e-6 for R in G)

    axes = {}
    for R in proper:
        ax, n = _axis_and_order(R)
        if ax is None or n < 2:
            continue
        k = _key(ax, 5)
        axes[k] = max(axes.get(k, 1), n)
    nmax = max(axes.values()) if axes else 1

    # rotation subgroup
    if r == 60 and nmax == 5:
        rot, nprin = "I", None
    elif r == 24 and nmax == 4 and len([1 for v in axes.values() if v == 3]) == 4:
        rot, nprin = "O", None
    elif r == 12 and nmax == 3:
        rot, nprin = "T", None
    elif r == 1:
        rot, nprin = "C1", 1
    elif r == nmax:
        rot, nprin = "C%d" % r, r
    elif r == 2 * nmax:
        rot, nprin = "D%d" % nmax, nmax
    else:
        rot, nprin = "C%d" % nmax, nmax           # fallback

    if g == r:
        schoen = rot
    else:
        principal = None
        if nprin and nprin > 1:
            for k, v in axes.items():
                if v == nmax:
                    principal = np.array(k, dtype=float)
                    break
        if rot == "T":
            schoen = "Th" if has_inv else "Td"
        elif rot == "O":
            schoen = "Oh"
        elif rot == "I":
            schoen = "Ih"
        elif rot == "C1":
            schoen = "Ci" if has_inv else "Cs"
        else:
            sigma_h = None
            if principal is not None:
                sigma_h = np.eye(3) - 2.0 * np.outer(principal, principal)
            has_sh = sigma_h is not None and any(
                np.max(np.abs(R - sigma_h)) < 1e-6 for R in G)
            if rot.startswith("D"):
                schoen = ("D%dh" if has_sh else "D%dd") % nprin
            else:
                if has_sh:
                    schoen = "C%dh" % nprin
                elif any(np.linalg.det(R) < 0 and
                         abs(float(np.trace(R)) - 1.0) < 1e-6 for R in G):
                    schoen = "C%dv" % nprin       # a mirror (trace +1) exists
                else:
                    schoen = "S%d" % (2 * nprin)

    return {"schoenflies": schoen, "order": g, "rotation_group": rot,
            "rotation_order": r, "chiral": g == r, "inversion": has_inv,
            "orbifold": orbifold(schoen), "coxeter": coxeter(schoen),
            "hermann_mauguin": hermann_mauguin(schoen)}


def _n_of(s):
    d = "".join(c for c in s if c.isdigit())
    return int(d) if d else 1


def orbifold(s):
    """Conway-Thurston orbifold signature. Cromwell (1997) Appendix I, col. 4."""
    table = {"T": "332", "Td": "*332", "Th": "3*2", "O": "432", "Oh": "*432",
             "I": "532", "Ih": "*532", "C1": "11", "Cs": "*11", "Ci": "1x"}
    if s in table:
        return table[s]
    n = _n_of(s)
    if s.startswith("S"):
        return "%dx" % (n // 2)
    if s.startswith("D"):
        if s.endswith("h"):
            return "*22%d" % n
        if s.endswith("d"):
            return "2*%d" % n
        return "22%d" % n
    if s.startswith("C"):
        if s.endswith("v"):
            return "*%d%d" % (n, n)
        if s.endswith("h"):
            return "%d*" % n
        return "%d%d" % (n, n)
    return None


def coxeter(s):
    """Coxeter bracket notation. Cromwell (1997) Appendix I, col. 2."""
    table = {"T": "[3,3]+", "Td": "[3,3]", "Th": "[3+,4]", "O": "[3,4]+",
             "Oh": "[3,4]", "I": "[3,5]+", "Ih": "[3,5]", "C1": "[]+",
             "Cs": "[]", "Ci": "[2+,2+]"}
    if s in table:
        return table[s]
    n = _n_of(s)
    if s.startswith("S"):
        return "[2+,%d+]" % n
    if s.startswith("D"):
        if s.endswith("h"):
            return "[2,%d]" % n
        if s.endswith("d"):
            return "[2+,%d]" % (2 * n)
        return "[2,%d]+" % n
    if s.startswith("C"):
        if s.endswith("v"):
            return "[%d]" % n
        if s.endswith("h"):
            return "[2,%d+]" % n
        return "[%d]+" % n
    return None


def hermann_mauguin(s):
    """International (crystallographic) symbol, for the groups that occur here."""
    table = {"T": "23", "Td": "-43m", "Th": "m-3", "O": "432", "Oh": "m-3m",
             "I": "532", "Ih": "m-3-5", "C1": "1", "Cs": "m", "Ci": "-1"}
    if s in table:
        return table[s]
    n = _n_of(s)
    if s.startswith("S"):
        # S_m is -m when 4 | m, else -m/2 (S6 = C3i is -3, not -6).
        return "-%d" % (n if n % 4 == 0 else n // 2)
    if s.startswith("D"):
        if s.endswith("h"):
            return "%d/mmm" % n if n % 2 == 0 else "-%dm2" % (2 * n)
        if s.endswith("d"):
            # even n: -(2n)2m  (D2d = -42m);  odd n: -n m  (D3d = -3m)
            return "-%d2m" % (2 * n) if n % 2 == 0 else "-%dm" % n
        return "%d22" % n
    if s.startswith("C"):
        if s.endswith("v"):
            return "%dmm" % n
        if s.endswith("h"):
            return "%d/m" % n
        return "%d" % n
    return None


# -- orbits -----------------------------------------------------------------

def _perm(A, G_elem, lookup, nd=6):
    img = A @ G_elem.T
    out = []
    for p in img:
        j = lookup.get(_key(p, nd))
        if j is None:
            return None
        out.append(j)
    return out


def orbits(V, F, G):
    """Orbit partitions of vertices, edges and faces under G.

    Returns (vertex_groups, edge_groups, face_groups) as lists of index lists,
    each sorted, the whole list ordered by first member.
    """
    A = _centred(V)
    lookup = {_key(v): i for i, v in enumerate(A)}
    perms = []
    for R in G:
        p = _perm(A, R, lookup)
        if p is not None:
            perms.append(p)

    face_key = {}
    for i, f in enumerate(F):
        face_key.setdefault(frozenset(f), i)
    edge_list = []
    seen_e = {}
    for f in F:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            k = frozenset((a, b))
            if k not in seen_e:
                seen_e[k] = len(edge_list)
                edge_list.append(k)

    def union(nelem, image):
        parent = list(range(nelem))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in perms:
            for i in range(nelem):
                j = image(i, p)
                if j is None:
                    continue
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
        groups = defaultdict(list)
        for i in range(nelem):
            groups[find(i)].append(i)
        return sorted((sorted(v) for v in groups.values()), key=lambda g: g[0])

    vg = union(len(A), lambda i, p: p[i])
    eg = union(len(edge_list),
               lambda i, p: seen_e.get(frozenset(p[x] for x in edge_list[i])))
    fg = union(len(F), lambda i, p: face_key.get(frozenset(p[x] for x in F[i])))
    return vg, eg, fg
