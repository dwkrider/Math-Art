# Multi-source cross-validation for the polyhedron database.
#
# Published polyhedron tables are used here as an ORACLE, not as the stored
# data. For each solid we build the geometry independently (from this repo's
# own generators), then check it against every source we can reach:
#
#   * D. McCooey, "Visual Polyhedra" -- exact closed forms and 30-digit
#     decimals for the radii, volume and dihedral angles.
#   * netlib polyhedra database -- face/vertex summaries and dihedral angles.
#   * Z. Har'El (1993) Appendix II -- Euler characteristic, density, Wythoff
#     symbol, vertex configuration, Wenninger and Coxeter et al. numbers.
#
# A record's stored coordinates remain our own derivation, so the repository
# publishes no third party's compilation; what the sources contribute is
# CONFIRMATION, recorded in `provenance.cross_checked`, plus exact closed forms
# for scalar measures, which are mathematical facts and are cited as such.
#
# Fetching is deliberately polite: one request at a time, with a pause, and
# everything cached on disk so a rebuild does not re-hit the servers.

import math
import os
import re
from collections import defaultdict

# This module NEVER opens a network connection: the project's python guard
# forbids it, and separating retrieval from analysis is better practice
# anyway. `tools/polydb_fetch.sh` populates the cache; everything here is a
# pure read of what is already on disk.
CACHE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), ".polydb_cache")


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if not text.strip() or "<html" in text[:200].lower():
        return None
    return text


def fetch_mccooey(stem):
    """Cached McCooey data file, or None if it was never downloaded."""
    if not stem:
        return None
    return _read(os.path.join(CACHE, "mccooey", stem + ".txt"))


def fetch_netlib(num):
    """Cached netlib record, or None if it was never downloaded."""
    if num is None:
        return None
    return _read(os.path.join(CACHE, "netlib", "%d.txt" % num))


# -- McCooey parsing --------------------------------------------------------

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_mccooey(text):
    """Return {'constants': {name: (decimal, exact_str)}, 'vertices': [...],
    'faces': [[...]], 'name': str} -- or None."""
    if not text:
        return None
    consts, verts, faces = {}, [], []
    name = text.strip().splitlines()[0].strip() if text.strip() else None
    for line in text.splitlines():
        s = line.strip()
        # Layout A: "C0 = 0.809... = (1 + sqrt(5)) / 4" on one line.
        m = re.match(r"^(C\d+)\s*=\s*(%s)\s*=\s*(.+)$" % _NUM, s)
        if m:
            consts[m.group(1)] = (float(m.group(2)), m.group(3).strip())
            continue
        # Layout B: the decimals are listed first, then the exact forms are
        # repeated further down, one per line. Pages for the harder solids use
        # this; treating it as unparseable loses the whole file, which is why
        # several duals looked like they had "no cached source".
        m = re.match(r"^(C\d+)\s*=\s*(.+)$", s)
        if m:
            key, rhs = m.group(1), m.group(2).strip()
            try:
                consts[key] = (float(rhs), consts.get(key, (None, None))[1])
            except ValueError:
                dec = consts.get(key, (None, None))[0]
                consts[key] = (dec, rhs)
            continue
        m = re.match(r"^V(\d+)\s*=\s*\(([^)]*)\)\s*$", s)
        if m:
            parts = [p.strip() for p in m.group(2).split(",")]
            verts.append((int(m.group(1)), parts))
            continue
        m = re.match(r"^\{([\d,\s]+)\}\s*$", s)
        if m:
            faces.append([int(x) for x in m.group(1).split(",")])
    verts.sort()
    return {"name": name, "constants": consts,
            "vertices": [v for _i, v in verts], "faces": faces}


def mccooey_numeric(parsed):
    """Evaluate McCooey's symbolic vertex table to floats."""
    if not parsed:
        return None
    env = {k: v[0] for k, v in parsed["constants"].items()}
    out = []
    for row in parsed["vertices"]:
        vals = []
        for tok in row:
            t = tok.strip()
            neg = t.startswith("-")
            t2 = t[1:].strip() if neg else t
            if t2 in env:
                val = env[t2]
            else:
                try:
                    val = float(t2)
                except ValueError:
                    return None
            vals.append(-val if neg else val)
        if len(vals) != 3:
            return None
        out.append(tuple(vals))
    return out


def to_language(expr):
    """McCooey's expression syntax -> the schema's exact language."""
    if expr is None:
        return None
    s = expr.strip().rstrip(".")
    s = s.replace("^", "**")
    s = re.sub(r"\s+", "", s)
    if not re.match(r"^[0-9+\-*/(){}\[\].a-zA-Z_]+$", s):
        return None
    s = s.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    # the language has no powers; expand small integer ones
    while "**" in s:
        i = s.index("**")
        j = i + 2
        k = j
        while k < len(s) and s[k].isdigit():
            k += 1
        try:
            e = int(s[j:k])
        except ValueError:
            return None
        if e < 1 or e > 6:
            return None
        if s[i - 1] == ")":
            depth, b = 0, i - 1
            while b >= 0:
                if s[b] == ")":
                    depth += 1
                elif s[b] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                b -= 1
            base = s[b:i]
        else:
            b = i - 1
            while b >= 0 and (s[b].isalnum() or s[b] == "."):
                b -= 1
            b += 1
            base = s[b:i]
        rep = "*".join([base] * e)
        s = s[:b] + ("(" + rep + ")" if e > 1 else rep) + s[k:]
    return s


# -- shape comparison, independent of orientation and vertex order ----------

def _invariants(V):
    """Orientation- and order-independent fingerprint of a vertex set:
    the sorted multiset of radii and of pairwise distances (rounded)."""
    A = [tuple(float(x) for x in v) for v in V]
    c = [sum(p[i] for p in A) / len(A) for i in range(3)]
    A = [tuple(p[i] - c[i] for i in range(3)) for p in A]
    radii = sorted(round(math.sqrt(sum(x * x for x in p)), 9) for p in A)
    ds = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            ds.append(round(math.dist(A[i], A[j]), 9))
    return radii, sorted(ds)


def _fit_normal(vs):
    """Unit normal by least squares, for faces whose Newell normal cancels."""
    n = len(vs)
    if n < 3:
        return None
    c = [sum(v[i] for v in vs) / n for i in range(3)]
    m = [[0.0] * 3 for _ in range(3)]
    for v in vs:
        d = [v[i] - c[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                m[i][j] += d[i] * d[j]
    try:
        import numpy as np
        _w, vec = np.linalg.eigh(np.array(m))
        out = [float(x) for x in vec[:, 0]]
    except Exception:
        return None
    L = math.sqrt(sum(x * x for x in out))
    return [x / L for x in out] if L > 1e-14 else None


def _dihedral_multiset(V, F, nd=6):
    """Sorted dihedral angles (degrees), one per edge."""
    inc = defaultdict(list)
    for i, f in enumerate(F):
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            inc[frozenset((a, b))].append(i)
    normals = []
    for f in F:
        vs = [V[i] for i in f]
        n = [0.0, 0.0, 0.0]
        for p, q in zip(vs, vs[1:] + vs[:1]):
            n[0] += (p[1] - q[1]) * (p[2] + q[2])
            n[1] += (p[2] - q[2]) * (p[0] + q[0])
            n[2] += (p[0] - q[0]) * (p[1] + q[1])
        L = math.sqrt(sum(x * x for x in n))
        if L > 1e-14:
            normals.append([c / L for c in n])
            continue
        # The Newell normal vanishes on a CROSSED face -- the star duals are
        # full of them -- so fall back to a least-squares plane. Dropping
        # those faces instead shortens this list on one side only, which then
        # reads as an edge-count mismatch between two identical solids.
        normals.append(_fit_normal(vs))
    out = []
    for _e, fis in inc.items():
        if len(fis) != 2 or normals[fis[0]] is None or normals[fis[1]] is None:
            continue
        c = sum(normals[fis[0]][i] * normals[fis[1]][i] for i in range(3))
        c = max(-1.0, min(1.0, c))
        deg = math.degrees(math.pi - math.acos(c))
        # Fold to the UNORIENTED angle. A face normal's sign follows the
        # winding, and on a non-orientable solid -- every hemipolyhedron here
        # -- no coherent winding exists, so the same edge reads as theta on one
        # source and 180-theta on another. Comparing unfolded angles reports
        # that convention difference as a disagreement, which it is not.
        out.append(round(min(deg, 180.0 - deg), nd))
    return sorted(out)


def _weld(V, F=None, nd=7):
    """Merge coincident vertices, remapping faces if given."""
    key, remap, out = {}, [], []
    for v in V:
        k = tuple(round(float(c), nd) + 0.0 for c in v)
        if k not in key:
            key[k] = len(out)
            out.append(tuple(float(c) for c in v))
        remap.append(key[k])
    if F is None:
        return out, None
    faces = []
    for f in F:
        g = [remap[i] for i in f]
        h = [g[i] for i in range(len(g)) if g[i] != g[(i - 1) % len(g)]]
        if len(h) >= 3:
            faces.append(h)
    return out, faces


def same_shape(V1, V2, F1=None, F2=None, tol=1e-7):
    """True when two solids are the same polyhedron up to rotation, reflection,
    vertex order and uniform scale.

    Vertices alone are NOT enough, and assuming otherwise gives false matches:
    distinct polyhedra routinely share a vertex arrangement -- the
    rhombicosidodecahedron with several 60-vertex star uniforms, the
    icosahedron with the great icosahedron. When faces are supplied for both
    sides the face-size and dihedral-angle multisets are compared too, which
    separates them. Without faces the result is only a vertex-arrangement
    match, and says so.
    """
    # Weld both sides first. Some star duals genuinely have COINCIDENT
    # vertices -- the great hexagonal hexecontahedron lists 104 vertices at 92
    # distinct positions, and McCooey's published table shows the same -- so a
    # welded table and an unwelded one describe the same solid and must not be
    # reported as a count mismatch.
    V1, F1 = _weld(V1, F1)
    V2, F2 = _weld(V2, F2)
    if len(V1) != len(V2):
        return False, "vertex count %d vs %d" % (len(V1), len(V2))
    r1, d1 = _invariants(V1)
    r2, d2 = _invariants(V2)
    if not d1 or not d2:
        return False, "degenerate"
    s = (sum(d1) / len(d1)) / (sum(d2) / len(d2))
    if s <= 0:
        return False, "bad scale"
    r2s = [x * s for x in r2]
    d2s = [x * s for x in d2]
    dr = max(abs(a - b) for a, b in zip(r1, r2s))
    dd = max(abs(a - b) for a, b in zip(d1, d2s))
    scale = max(1.0, max(d1))
    if not (dr < tol * scale and dd < tol * scale):
        return False, ("vertex arrangement differs: radii dev %.2e, distance "
                       "dev %.2e (scale %.6f)" % (dr, dd, s))

    if F1 is None or F2 is None:
        return True, ("vertex arrangement matches (radii dev %.2e, distance "
                      "dev %.2e); faces not compared" % (dr, dd))

    sz1 = sorted(len(f) for f in F1)
    sz2 = sorted(len(f) for f in F2)
    if sz1 != sz2:
        return False, ("vertices match but faces differ: %d vs %d faces, "
                       "size multisets disagree" % (len(F1), len(F2)))
    a1 = _dihedral_multiset(V1, F1)
    a2 = _dihedral_multiset(V2, F2)
    if len(a1) != len(a2):
        return False, "vertices and face sizes match but edge counts differ"
    da = max((abs(x - y) for x, y in zip(a1, a2)), default=0.0)
    if da > 1e-4:
        return False, ("vertices and face sizes match but dihedral angles "
                       "differ by up to %.4f deg -- a different solid on the "
                       "same vertex arrangement" % da)
    return True, ("full match: radii dev %.2e, distance dev %.2e, dihedral dev "
                  "%.1e deg (scale %.6f)" % (dr, dd, da, s))
