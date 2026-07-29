
# Seifert Surface Generator for Blender
#
# Builds Seifert surfaces for knots and links given as braid words,
# inspired by SeifertView (J. J. van Wijk & A. M. Cohen, "Visualization
# of Seifert Surfaces", IEEE TVCG 12(4), 2006; vanwijk.win.tue.nl/seifertview).
#
# A braid word on n strands is a sequence of crossings s_i / s_i^-1
# (adjacent-strand exchanges). Seifert's algorithm applied to the braid
# closure yields one disk per strand and one half-twisted band per
# crossing. The add-on stacks the disks like a wedding cake (largest at
# the bottom) and connects consecutive rims with +-180-degree twisted
# ribbons -- the SeifertView picture. The surface boundary is exactly
# the knot/link (optionally emitted as a bevelled curve object), and
# genus / boundary-component counts are computed from the braid.
#
# Optional smoothing relaxes the surface interior with the Plateau
# solver from minimal_surface_toolkit.py (if installed), pinning the
# knot boundary -- a "soap film bounded by the knot" look.
#
# Braid word syntax: letters ("aAbB..": a = s1, A = s1^-1) or integers
# ("1 -2 1 -2"). Presets cover the classic examples.
#
# References:
# - Seifert's algorithm and the genus of a knot: Herbert Seifert,
#   "Ueber das Geschlecht von Knoten", Math. Ann. 110, 1934.
# - Visualization approach after J. J. van Wijk & A. M. Cohen,
#   "Visualization of Seifert Surfaces", IEEE TVCG 12(4), 2006
#   (SeifertView; vanwijk.win.tue.nl/seifertview).
# - Braid generators s_i after Emil Artin (braid groups B_n).

bl_info = {
    "name": "Seifert Surface Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Seifert Surface / N-panel 'Minimal Surfaces'",
    "description": "Seifert surfaces for knots and links from braid words",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi, asin, sqrt

TAU = 2.0 * math.pi


# --------------------------------------------------------------------------
# Braid combinatorics
# --------------------------------------------------------------------------

def parse_braid(text):
    """Parse a braid word. Letters: a..z = generators 1.., A..Z their
    inverses. Or whitespace/comma separated signed integers. Returns a
    list of nonzero ints (sign = crossing sign)."""
    text = text.strip()
    if not text:
        return []
    if any(c.isalpha() for c in text):
        word = []
        for c in text:
            if c.isspace():
                continue
            if not c.isalpha():
                raise ValueError(f"bad braid character {c!r}")
            if c.islower():
                word.append(ord(c) - ord('a') + 1)
            else:
                word.append(-(ord(c) - ord('A') + 1))
        return word
    toks = text.replace(',', ' ').split()
    word = []
    for t in toks:
        v = int(t)
        if v == 0:
            raise ValueError("0 is not a braid generator")
        word.append(v)
    return word


def torus_braid(p, q):
    """Braid word of the (p,q) torus link: (s1 s2 .. s_{p-1})^q."""
    return [i for _ in range(q) for i in range(1, p)]


def braid_stats(word):
    """(strands, crossings, boundary components mu, genus). The braid
    closure has Seifert circles = strands, so Euler characteristic
    chi = strands - crossings and genus = (2 - mu - chi) / 2 per
    connected component sum (reported for the whole surface)."""
    n = max((abs(w) for w in word), default=1) + 1 if word else 1
    L = len(word)
    perm = list(range(n))
    for w in word:
        i = abs(w) - 1
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    seen = [False] * n
    mu = 0
    for s in range(n):
        if seen[s]:
            continue
        mu += 1
        cur = s
        while not seen[cur]:
            seen[cur] = True
            cur = perm[cur]
    chi = n - L
    genus = (2 - mu - chi) / 2  # for a connected surface
    return n, L, mu, genus


PRESETS = {
    'TREFOIL': ("Trefoil (3_1)", "aaa"),
    'FIGURE8': ("Figure-8 (4_1)", "aBaB"),
    'CINQUEFOIL': ("Cinquefoil (5_1)", "aaaaa"),
    'GRANNY': ("Granny knot", "aaabbb"),
    'SQUARE': ("Square knot", "aaaBBB"),
    'HOPF': ("Hopf link", "aa"),
    'SOLOMON': ("Solomon's link", "aaaa"),
    'BORROMEAN': ("Borromean rings", "aBaBaB"),
}


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def build_seifert(word, m_res=96, rings=6, band_rows=16, band_seg=8,
                  radius=2.0, taper=0.5, spacing=0.75, band_width=0.5):
    """Build the Seifert surface mesh for a braid word.

    Returns (verts, faces, fixed, n, L) where `fixed` marks the surface
    boundary (the knot/link) for the optional relaxation step."""
    n = max((abs(w) for w in word), default=1) + 1 if word else 1
    L = len(word)
    radii = [radius * (1.0 - taper * i / max(1, n - 1)) for i in range(n)] \
        if n > 1 else [radius]
    zs = [i * spacing for i in range(n)]

    verts = []
    faces = []
    fixed = []

    def add_vert(p, on_boundary):
        verts.append(p)
        fixed.append(on_boundary)
        return len(verts) - 1

    # metric band width, kept below the free arc between crossing slots
    if L:
        W = band_width * min(2.0 * r * sin(pi / L) for r in radii)
    else:
        W = 0.0

    # ---- bands ----------------------------------------------------------
    # per disk: list of (theta, [rim vertex indices CCW]) attachments
    attach = [[] for _ in range(n)]
    for k, w in enumerate(word):
        i = abs(w) - 1
        sgn = 1.0 if w > 0 else -1.0
        th = TAU * (k + 0.5) / L
        U = (-sin(th), cos(th), 0.0)
        r0, r1 = radii[i], radii[i + 1]
        z0, z1 = zs[i], zs[i + 1]
        rows = []
        for j in range(band_rows + 1):
            t = j / band_rows
            s = _smooth(t)
            rho = r0 + (r1 - r0) * s
            zz = z0 + (z1 - z0) * s
            # centerline tangent (in the rho-z plane through angle th)
            ds = 6.0 * t * (1.0 - t)
            drho = (r1 - r0) * ds
            dz = (z1 - z0) * ds
            if abs(drho) + abs(dz) < 1e-9:
                drho, dz = (r1 - r0), (z1 - z0)
            tl = sqrt(drho * drho + dz * dz)
            # V = T x U where T = (drho*cos, drho*sin, dz)/tl
            ct, st2 = cos(th), sin(th)
            Tv = (drho * ct / tl, drho * st2 / tl, dz / tl)
            V = (Tv[1] * U[2] - Tv[2] * U[1],
                 Tv[2] * U[0] - Tv[0] * U[2],
                 Tv[0] * U[1] - Tv[1] * U[0])
            phi = sgn * pi * _smooth(t)
            cph, sph = cos(phi), sin(phi)
            C = (rho * ct, rho * st2, zz)
            row = []
            for l in range(band_seg + 1):
                a = (l / band_seg - 0.5) * W
                d = (cph * U[0] + sph * V[0],
                     cph * U[1] + sph * V[1],
                     cph * U[2] + sph * V[2])
                onb = l == 0 or l == band_seg
                row.append(add_vert((C[0] + a * d[0], C[1] + a * d[1],
                                     C[2] + a * d[2]), onb))
            rows.append(row)
        for j in range(band_rows):
            for l in range(band_seg):
                faces.append((rows[j][l], rows[j][l + 1],
                              rows[j + 1][l + 1], rows[j + 1][l]))
        attach[i].append((th, rows[0]))                 # lower rim, CCW
        attach[i + 1].append((th, rows[-1][::-1]))      # upper rim, CCW
        # band end verts sit on the disk rims: they are interior to the
        # surface except the two corners, handled via the rim pass below

    # ---- disks ----------------------------------------------------------
    for d in range(n):
        r = radii[d]
        z = zs[d]
        att = sorted(attach[d], key=lambda e: e[0])
        alpha = asin(min(0.999, (W / 2.0) / r)) if W > 0 else 0.0
        spans = [(th - alpha, th + alpha) for th, _ in att]
        rim = []            # list of vertex indices, CCW
        if att:
            for idx, (th, chord) in enumerate(att):
                rim.extend(chord)
                a_from = th + alpha
                a_to = att[(idx + 1) % len(att)][0] - alpha
                if idx == len(att) - 1:
                    a_to += TAU
                k0 = int(math.ceil(a_from / (TAU / m_res)))
                k1 = int(math.floor(a_to / (TAU / m_res)))
                for kk in range(k0, k1 + 1):
                    ang = (TAU / m_res) * kk
                    rim.append(add_vert((r * cos(ang), r * sin(ang), z),
                                        True))
        else:
            for kk in range(m_res):
                ang = (TAU / m_res) * kk
                rim.append(add_vert((r * cos(ang), r * sin(ang), z), True))
        # interior rings shrinking to the center
        nrim = len(rim)
        prev = rim
        for rg in range(1, rings):
            f = rg / rings
            cur = []
            for vi in prev if rg == 1 else []:
                pass
            for ii in range(nrim):
                px, py, pz = verts[rim[ii]]
                cur.append(add_vert((px * (1 - f), py * (1 - f), z), False))
            for ii in range(nrim):
                faces.append((prev[ii], prev[(ii + 1) % nrim],
                              cur[(ii + 1) % nrim], cur[ii]))
            prev = cur
        cen = add_vert((0.0, 0.0, z), False)
        for ii in range(nrim):
            faces.append((prev[ii], prev[(ii + 1) % nrim], cen))

    # band end rows lie on the rims but are interior points of the
    # surface (band + disk meet there); only their outermost corners are
    # on the knot. Fix up the flags: chord verts were marked boundary at
    # l=0/band_seg only, which is exactly the knot -- nothing more to do.
    return verts, faces, fixed, n, L


# --------------------------------------------------------------------------
# Dynamic relaxation  (SeifertView / KnotPlot-style physical smoothing)
# --------------------------------------------------------------------------
#
# Reverse-engineered from seifertview1.1.exe (J. van Wijk & A. Cohen; the
# "physical model for smoothing ... based on the work of Robert Scharein",
# i.e. KnotPlot). The binary's "Dynamic smoothing" panel exposes exactly
# these controls -- attraction, repulsion (with a range scale), a bend
# force, a time step, damping ("decay") and alpha/beta smoothing weights,
# integrated dynamically over many iterations. Crucially the knot boundary
# is NOT pinned: the whole surface, its boundary included, evolves as a
# self-avoiding elastic membrane. That is the key difference from the
# Plateau/area-minimization path below (`_relax`), which pins the boundary
# and only ever produces a taut soap film on a frozen, tangled wire.
#
# The smoother models the surface as an inflating elastic membrane and
# integrates it dynamically with a FREE boundary -- the key difference
# from the pinned area-minimizer (`_relax`) below, which can only ever
# hang a taut soap film on the frozen, tangled knot. Forces per step:
#
#   * smoothing -- Laplacian (mean-curvature) fairing pulls each vertex
#                  toward the umbrella mean of its neighbours, removing
#                  crinkle. Boundary vertices smooth only ALONG the knot
#                  loop, so the boundary stays a fair curve on the
#                  surface instead of retracting.
#   * pressure  -- a uniform push along the consistently-oriented surface
#                  normal. Smoothing alone shrinks a free sheet to a
#                  point; pressure balances it, so the stacked disks
#                  billow out into smooth embedded caps (rather than
#                  buckling into pleats). Seifert surfaces are orientable,
#                  so `_orient_faces` first gives every face a consistent
#                  winding and hence a coherent normal field. Pressure --
#                  not pairwise vertex repulsion, which shatters the mesh
#                  into shards -- is what inflates a membrane smoothly.
#
# Integration is damped and Euler-style with a per-step displacement
# clamp, robust even at the high-valence pole at each disk centre.


def _topology(faces, nv):
    """Directed neighbour arrays, unique edges, and up-to-2-ring
    exclusion sets (for repulsion)."""
    nbr = [set() for _ in range(nv)]
    edges = set()
    for f in faces:
        k = len(f)
        for a in range(k):
            i, j = f[a], f[(a + 1) % k]
            nbr[i].add(j)
            nbr[j].add(i)
            edges.add((i, j) if i < j else (j, i))
    src, dst = [], []
    for i, ns in enumerate(nbr):
        for j in ns:
            src.append(i)
            dst.append(j)
    return nbr, src, dst, edges


def _kring_exclude(nbr, nv, hops):
    """For each vertex, the set of vertices within `hops` graph steps
    (BFS), so repulsion never fires between on-sheet neighbours."""
    excl = []
    for s in range(nv):
        seen = {s}
        frontier = {s}
        for _ in range(hops):
            nxt = set()
            for u in frontier:
                nxt |= nbr[u]
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        seen.discard(s)
        excl.append(seen)
    return excl


def _orient_faces(faces):
    """Return a copy of `faces` with a globally consistent winding
    (BFS over the face-adjacency graph, flipping faces so shared edges
    are traversed in opposite directions). Works for an orientable
    manifold such as a Seifert surface; components handled separately."""
    faces = [list(f) for f in faces]
    # edge -> list of (face index, position a within face)
    e2f = {}
    for fi, f in enumerate(faces):
        k = len(f)
        for a in range(k):
            e = (f[a], f[(a + 1) % k])
            e2f.setdefault((min(e), max(e)), []).append((fi, a))
    adj = [[] for _ in faces]
    for (u, v), incid in e2f.items():
        if len(incid) == 2:
            (f0, a0), (f1, a1) = incid
            adj[f0].append((f1, u, v))
            adj[f1].append((f0, u, v))

    def directed(f, u, v):
        k = len(f)
        for a in range(k):
            if f[a] == u and f[(a + 1) % k] == v:
                return True
            if f[a] == v and f[(a + 1) % k] == u:
                return False
        return None

    seen = [False] * len(faces)
    for s in range(len(faces)):
        if seen[s]:
            continue
        seen[s] = True
        stack = [s]
        while stack:
            fi = stack.pop()
            for (fj, u, v) in adj[fi]:
                if seen[fj]:
                    continue
                # consistent orientation: the shared edge must run the
                # opposite way in the two faces
                di = directed(faces[fi], u, v)
                dj = directed(faces[fj], u, v)
                if di is not None and dj is not None and di == dj:
                    faces[fj].reverse()
                seen[fj] = True
                stack.append(fj)
    return faces


def _vertex_normals(P, ftris, np):
    """Area-weighted vertex normals from a triangle index array."""
    N = np.zeros_like(P)
    a = P[ftris[:, 1]] - P[ftris[:, 0]]
    b = P[ftris[:, 2]] - P[ftris[:, 0]]
    fn = np.cross(a, b)
    for k in range(3):
        np.add.at(N, ftris[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(ln, 1e-12)


def _repulsion_pairs(P, radius, excl):
    """Candidate vertex pairs within `radius`, via a spatial hash; each
    unordered pair once, topological 2-ring neighbours excluded."""
    import numpy as np
    inv = 1.0 / radius
    keys = np.floor(P * inv).astype(np.int64)
    buckets = {}
    for idx in range(len(P)):
        buckets.setdefault(tuple(keys[idx]), []).append(idx)
    I, J = [], []
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)]
    for cell, members in buckets.items():
        neigh = []
        for o in offs:
            c = (cell[0] + o[0], cell[1] + o[1], cell[2] + o[2])
            b = buckets.get(c)
            if b:
                neigh.extend(b)
        for a in members:
            ea = excl[a]
            for b in neigh:
                if b > a and b not in ea:
                    I.append(a)
                    J.append(b)
    if not I:
        return (np.empty(0, np.int64), np.empty(0, np.int64))
    return np.array(I, np.int64), np.array(J, np.int64)


def _boundary_from_faces(faces, nv):
    """(is_boundary mask, ordered boundary loops) from a face list."""
    cnt = {}
    for f in faces:
        k = len(f)
        for a in range(k):
            i, j = f[a], f[(a + 1) % k]
            e = (i, j) if i < j else (j, i)
            cnt[e] = cnt.get(e, 0) + 1
    adj = {}
    mask = [False] * nv
    for (a, b), c in cnt.items():
        if c == 1:
            mask[a] = mask[b] = True
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    loops = []
    seen = set()
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxts = [v for v in adj[cur] if v != prev]
            nxt = None
            for v in nxts:
                if v not in seen:
                    nxt = v
                    break
            if nxt is None:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return mask, loops


def dynamic_relax(verts, faces, iterations=600, smooth=0.6,
                  pressure=0.35, decay=0.5, callback=None):
    """SeifertView-style inflating-membrane smoothing with a FREE
    boundary.

    Laplacian fairing (interior by the umbrella Laplacian, boundary
    along its own loop) balanced by a pressure force along the
    consistently-oriented surface normal, integrated with damping.
    `smooth` in (0,1] is the fairing strength; `pressure` is the
    inflation as a fraction of the mean edge length per step; `decay`
    is the velocity damping. Returns a new vertex list (numpy
    required; None otherwise). `callback(i, P)` runs each iteration."""
    try:
        import numpy as np
    except ImportError:
        return None
    P = np.asarray(verts, dtype=np.float64).copy()
    nv = len(P)
    nbr, src, dst, edges = _topology(faces, nv)
    src = np.asarray(src, np.int64)
    dst = np.asarray(dst, np.int64)
    deg = np.zeros(nv)
    np.add.at(deg, src, 1.0)
    deg = np.maximum(deg, 1.0)
    bmask, loops = _boundary_from_faces(faces, nv)
    bmask = np.asarray(bmask, bool)
    ofaces = _orient_faces(faces)
    ftris = np.asarray([(f[0], f[t], f[t + 1]) for f in ofaces
                        for t in range(1, len(f) - 1)], np.int64)

    bprev = np.arange(nv)
    bnext = np.arange(nv)
    for lp in loops:
        L = len(lp)
        for a, v in enumerate(lp):
            bprev[v] = lp[(a - 1) % L]
            bnext[v] = lp[(a + 1) % L]

    def umbrella(X):
        acc = np.zeros_like(X)
        np.add.at(acc, src, X[dst])
        lap = acc / deg[:, None] - X
        if bmask.any():                 # boundary: along the loop only
            lap[bmask] = 0.5 * (X[bprev[bmask]] + X[bnext[bmask]]) \
                - X[bmask]
        return lap

    ei = np.asarray([e[0] for e in edges], np.int64)
    ej = np.asarray([e[1] for e in edges], np.int64)
    mean_edge = float(np.linalg.norm(P[ei] - P[ej], axis=1).mean()) \
        or 1.0
    smax = 0.5 * mean_edge
    kp = pressure * mean_edge
    Vel = np.zeros_like(P)
    for it in range(iterations):
        F = smooth * umbrella(P)                    # fairing (shrinks)
        N = _vertex_normals(P, ftris, np)
        F[~bmask] += kp * N[~bmask]                 # pressure (inflates)
        Vel = (Vel + F) * decay
        sl = np.linalg.norm(Vel, axis=1)
        big = sl > smax
        if big.any():
            Vel[big] *= (smax / sl[big])[:, None]
        P += Vel
        if callback is not None:
            callback(it, P)
    return [tuple(p) for p in P]


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _relax(me_verts, faces, fixed, iterations):
        """Optional smoothing using the Plateau solver from the Minimal
        Surface Toolkit (soft dependency)."""
        try:
            import numpy as np
            try:
                from . import minimal_surface_toolkit as mst
            except ImportError:
                import minimal_surface_toolkit as mst
        except ImportError:
            return None
        V = np.array(me_verts, dtype=np.float64)
        T = []
        for f in faces:
            if len(f) == 4:
                T.append((f[0], f[1], f[2]))
                T.append((f[0], f[2], f[3]))
            else:
                T.append(tuple(f))
        T = np.array(T, dtype=np.int64)
        fx = np.array(fixed, dtype=bool)
        mst.minimize_area(V, T, fx, outer_iters=iterations)
        return [tuple(v) for v in V]

    def _boundary_loops_from_faces(faces):
        """Ordered boundary vertex loops from a face list."""
        count = {}
        for vs in faces:
            for a in range(len(vs)):
                e = tuple(sorted((vs[a], vs[(a + 1) % len(vs)])))
                count[e] = count.get(e, 0) + 1
        adj = {}
        for (a, b), c in count.items():
            if c == 1:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        loops = []
        unvisited = set(adj.keys())
        while unvisited:
            start = unvisited.pop()
            loop = [start]
            prev = None
            cur = start
            while True:
                nxts = [v for v in adj[cur] if v != prev and v in unvisited] \
                    or [v for v in adj[cur] if v != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
                if cur == start:
                    break
                loop.append(cur)
                unvisited.discard(cur)
            loops.append(loop)
        return loops

    def _relax_free(me_verts, faces, fixed, rounds, surf_iters=6,
                    step=0.22):
        """Whole-shape relaxation: the knot boundary evolves like an
        elastic curve (Laplacian smoothing at constant total length,
        with self-repulsion so strands cannot pass through each other),
        while the surface interior re-relaxes to the moving boundary
        each round. Returns new verts or None if numpy/toolkit absent."""
        try:
            import numpy as np
            try:
                from . import minimal_surface_toolkit as mst
            except ImportError:
                import minimal_surface_toolkit as mst
        except ImportError:
            return None
        V = np.array(me_verts, dtype=np.float64)
        T = []
        for f in faces:
            if len(f) == 4:
                T.append((f[0], f[1], f[2]))
                T.append((f[0], f[2], f[3]))
            else:
                T.append(tuple(f))
        T = np.array(T, dtype=np.int64)
        fx = np.array(fixed, dtype=bool)
        loops = _boundary_loops_from_faces(faces)
        loop_idx = [np.array(lp, dtype=np.int64) for lp in loops]
        ball = np.concatenate(loop_idx)
        # arc positions for excluding near-neighbours from repulsion
        arc_id = np.concatenate([np.full(len(lp), li)
                                 for li, lp in enumerate(loop_idx)])
        arc_pos = np.concatenate([np.arange(len(lp)) for lp in loop_idx])
        loop_len = {li: len(lp) for li, lp in enumerate(loop_idx)}

        def total_length(P):
            tot = 0.0
            o = 0
            for lp in loop_idx:
                Q = P[o:o + len(lp)]
                tot += float(np.sum(np.linalg.norm(
                    np.roll(Q, -1, axis=0) - Q, axis=1)))
                o += len(lp)
            return tot

        B = V[ball]
        len0 = total_length(B)
        seg0 = len0 / len(ball)
        d0 = 4.0 * seg0            # repulsion range
        for _ in range(rounds):
            B = V[ball]
            # curve Laplacian smoothing per loop
            newB = B.copy()
            o = 0
            for lp in loop_idx:
                Q = B[o:o + len(lp)]
                lap = 0.5 * (np.roll(Q, 1, axis=0)
                             + np.roll(Q, -1, axis=0)) - Q
                newB[o:o + len(lp)] = Q + step * lap
                o += len(lp)
            # self-repulsion (all boundary pairs, skipping arc-neighbours)
            D = newB[:, None, :] - newB[None, :, :]
            dist = np.linalg.norm(D, axis=2)
            near = dist < d0
            same = arc_id[:, None] == arc_id[None, :]
            gap = np.abs(arc_pos[:, None] - arc_pos[None, :])
            wrap = np.array([loop_len[a] for a in arc_id])
            gap = np.minimum(gap, wrap[:, None] - gap)
            near &= ~(same & (gap <= 4))
            np.fill_diagonal(near, False)
            ii, jj = np.nonzero(near)
            if len(ii):
                d = np.maximum(dist[ii, jj], 1e-9)
                push = (D[ii, jj].T / d * (d0 - d) / d0).T
                acc = np.zeros_like(newB)
                np.add.at(acc, ii, push)
                newB += (0.5 * step * seg0) * \
                    (acc / np.maximum(np.linalg.norm(acc, axis=1,
                                                     keepdims=True), 1e-9)) \
                    * (np.linalg.norm(acc, axis=1, keepdims=True) > 1e-9)
            # rescale about the centroid to preserve total length
            cen = newB.mean(axis=0)
            cur_len = total_length(newB)
            if cur_len > 1e-9:
                newB = cen + (newB - cen) * (len0 / cur_len)
            V[ball] = newB
            # surface follows the moved boundary
            mst.minimize_area(V, T, fx, outer_iters=surf_iters)
        return [tuple(v) for v in V]

    def _boundary_loops(me):
        """Ordered boundary vertex loops of a mesh."""
        count = {}
        for p in me.polygons:
            vs = p.vertices
            for a in range(len(vs)):
                e = tuple(sorted((vs[a], vs[(a + 1) % len(vs)])))
                count[e] = count.get(e, 0) + 1
        adj = {}
        for (a, b), c in count.items():
            if c == 1:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        loops = []
        unvisited = set(adj.keys())
        while unvisited:
            start = unvisited.pop()
            loop = [start]
            prev = None
            cur = start
            while True:
                nxts = [v for v in adj[cur] if v != prev and v in unvisited] \
                    or [v for v in adj[cur] if v != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
                if cur == start:
                    break
                loop.append(cur)
                unvisited.discard(cur)
            loops.append(loop)
        return loops

    class MESH_OT_seifert_add(bpy.types.Operator):
        """Build the Seifert surface of a knot/link given as braid word"""
        bl_idname = "mesh.seifert_surface_add"
        bl_label = "Seifert Surface"
        bl_options = {'REGISTER', 'UNDO'}

        def _preset_chosen(self, context):
            if self.preset == 'TORUS':
                self.braid = " ".join(map(str, torus_braid(self.torus_p,
                                                           self.torus_q)))
            elif self.preset != 'CUSTOM':
                self.braid = PRESETS[self.preset][1]

        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Custom braid", "Use the braid word below")] +
                  [(k, v[0], f"braid {v[1]}") for k, v in PRESETS.items()] +
                  [('TORUS', "Torus knot (p,q)", "Torus-knot braid")],
            default='TREFOIL', update=_preset_chosen)
        braid: StringProperty(
            name="Braid Word", default="aaa",
            description="Letters (a=s1, A=s1 inverse) or integers '1 -2 1'")
        torus_p: IntProperty(name="Torus p", default=3, min=2, max=8,
                             update=_preset_chosen)
        torus_q: IntProperty(name="Torus q", default=4, min=2, max=12,
                             update=_preset_chosen)
        radius: FloatProperty(name="Base Radius", default=2.0,
                              min=0.5, max=10.0)
        taper: FloatProperty(
            name="Taper", default=0.5, min=0.1, max=0.85,
            description="How much smaller the top disk is than the bottom")
        spacing: FloatProperty(name="Disk Spacing", default=0.75,
                               min=0.1, max=3.0)
        band_width: FloatProperty(
            name="Band Width", default=0.5, min=0.1, max=0.9,
            description="Band width as a fraction of the free rim arc")
        resolution: IntProperty(
            name="Resolution", default=96, min=24, max=384,
            description="Angular samples around the disks")
        rings: IntProperty(name="Disk Rings", default=6, min=2, max=32)
        band_rows: IntProperty(name="Band Rows", default=16, min=4, max=64)
        relax: IntProperty(
            name="Relax Iterations", default=0, min=0, max=100,
            description="Smooth the surface with the Plateau solver "
                        "(needs the Minimal Surface Toolkit add-on), "
                        "keeping the knot boundary pinned")
        shape_relax: IntProperty(
            name="Shape Relax Rounds", default=0, min=0, max=200,
            description="Relax the WHOLE shape: the knot boundary evolves "
                        "as an elastic curve (constant length, with "
                        "self-repulsion so strands cannot cross) while the "
                        "surface re-relaxes each round. Needs the Minimal "
                        "Surface Toolkit add-on")
        add_knot_curve: BoolProperty(
            name="Add Knot Curve", default=True,
            description="Also create a bevelled curve along the "
                        "knot/link (the surface boundary)")
        knot_radius: FloatProperty(name="Knot Tube Radius", default=0.04,
                                   min=0.0, max=0.5)

        def execute(self, context):
            if self.preset == 'TORUS':
                word = torus_braid(self.torus_p, self.torus_q)
            else:
                try:
                    word = parse_braid(self.braid)
                except ValueError as e:
                    self.report({'ERROR'}, str(e))
                    return {'CANCELLED'}
                if self.preset != 'CUSTOM':
                    word = parse_braid(PRESETS[self.preset][1])
            n, L, mu, genus = braid_stats(word)
            verts, faces, fixedm, n, L = build_seifert(
                word, m_res=self.resolution, rings=self.rings,
                band_rows=self.band_rows, radius=self.radius,
                taper=self.taper, spacing=self.spacing,
                band_width=self.band_width)
            if self.shape_relax > 0:
                relaxed = _relax_free(verts, faces, fixedm,
                                      self.shape_relax)
                if relaxed is None:
                    self.report({'WARNING'},
                                "Shape relax needs the Minimal Surface "
                                "Toolkit add-on (numpy) -- skipped")
                else:
                    verts = relaxed
            if self.relax > 0:
                relaxed = _relax(verts, faces, fixedm, self.relax)
                if relaxed is None:
                    self.report({'WARNING'},
                                "Relax needs the Minimal Surface Toolkit "
                                "add-on (numpy) -- skipped")
                else:
                    verts = relaxed
            me = bpy.data.meshes.new("SeifertSurface")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("SeifertSurface", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            obj["braid"] = " ".join(map(str, word))
            obj["strands"] = n
            obj["crossings"] = L
            obj["boundary_components"] = mu
            obj["genus"] = genus
            if self.add_knot_curve:
                loops = _boundary_loops(me)
                cu = bpy.data.curves.new("Knot", 'CURVE')
                cu.dimensions = '3D'
                cu.bevel_depth = self.knot_radius
                cu.use_fill_caps = True
                for loop in loops:
                    sp = cu.splines.new('POLY')
                    sp.points.add(len(loop) - 1)
                    flat = []
                    for vi in loop:
                        x, y, z = verts[vi]
                        flat.extend((x, y, z, 1.0))
                    sp.points.foreach_set('co', flat)
                    sp.use_cyclic_u = True
                knot = bpy.data.objects.new("Knot", cu)
                context.collection.objects.link(knot)
                knot.location = obj.location
                knot.parent = obj
                knot.matrix_parent_inverse.identity()
                knot.location = (0, 0, 0)
            self.report({'INFO'},
                        f"strands={n} crossings={L} components={mu} "
                        f"genus={genus:g}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'TORUS':
                lay.prop(self, 'torus_p')
                lay.prop(self, 'torus_q')
            else:
                lay.prop(self, 'braid')
            for k in ('radius', 'taper', 'spacing', 'band_width',
                      'resolution', 'rings', 'band_rows', 'shape_relax',
                      'relax', 'add_knot_curve', 'knot_radius'):
                lay.prop(self, k)

    class MESH_OT_seifert_minimize(bpy.types.Operator):
        """Minimize the active Seifert surface (one click, tuned
        defaults): the knot boundary evolves as an elastic curve of
        constant length with self-repulsion, the membrane re-relaxes
        each round (Pinkall-Polthier area minimization, the solver
        validated on the flat disk and catenoid), then a final pinned
        polish. Click again to continue minimizing"""
        bl_idname = "mesh.seifert_minimize"
        bl_label = "Minimize Surface"
        bl_options = {'REGISTER', 'UNDO'}

        method: EnumProperty(
            name="Method",
            items=[
                ('AREA', "Area (pinned boundary)",
                 "Pinkall-Polthier area minimization -- a soap film "
                 "on the fixed knot (the original behaviour)"),
                ('MEMBRANE', "Membrane (free boundary, experimental)",
                 "SeifertView-style: the whole surface AND its "
                 "boundary evolve as an inflating elastic membrane "
                 "(dynamic_relax). Numerically stable but does not "
                 "yet reach SeifertView's finish -- see module notes"),
            ],
            default='AREA')
        rounds: IntProperty(
            name="Evolve Boundary Rounds", default=0, min=0, max=400,
            description="AREA method: gentle boundary evolution "
                        "rounds (0 keeps the knot fixed and only "
                        "relaxes the membrane)")
        polish: IntProperty(
            name="Polish Iterations", default=60, min=0, max=300,
            description="AREA method: final area-minimization with "
                        "the boundary pinned")
        mem_iterations: IntProperty(
            name="Membrane Iterations", default=600, min=10,
            max=5000,
            description="MEMBRANE method: dynamic relaxation steps")
        mem_smooth: FloatProperty(
            name="Membrane Fairing", default=0.6, min=0.0, max=1.0,
            description="MEMBRANE method: Laplacian fairing strength")
        mem_pressure: FloatProperty(
            name="Membrane Pressure", default=0.2, min=0.0, max=1.0,
            description="MEMBRANE method: inflation along the surface "
                        "normal that stops the free sheet buckling")

        @classmethod
        def poll(cls, context):
            o = context.object
            return (o is not None and o.type == 'MESH'
                    and "braid" in o)

        @staticmethod
        def _area(verts, faces):
            import numpy as np
            V = np.asarray(verts)
            tot = 0.0
            for f in faces:
                for i in range(1, len(f) - 1):
                    a = V[f[0]]
                    b = V[f[i]]
                    c = V[f[i + 1]]
                    tot += 0.5 * float(np.linalg.norm(
                        np.cross(b - a, c - a)))
            return tot

        def execute(self, context):
            obj = context.object
            me = obj.data
            verts = [tuple(v.co) for v in me.vertices]
            faces = [list(p.vertices) for p in me.polygons]
            # boundary verts = ends of edges used by a single face
            cnt = {}
            for f in faces:
                for i in range(len(f)):
                    e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
                    cnt[e] = cnt.get(e, 0) + 1
            fixed = [False] * len(verts)
            for (a, b), c in cnt.items():
                if c == 1:
                    fixed[a] = True
                    fixed[b] = True
            a0 = None
            try:
                a0 = self._area(verts, faces)
            except ImportError:
                pass
            if self.method == 'MEMBRANE':
                out = dynamic_relax(
                    verts, faces, iterations=self.mem_iterations,
                    smooth=self.mem_smooth,
                    pressure=self.mem_pressure)
                if out is None:
                    self.report({'ERROR'}, "membrane method needs "
                                "numpy")
                    return {'CANCELLED'}
            else:
                if self.rounds > 0:
                    out = _relax_free(verts, faces, fixed,
                                      self.rounds, surf_iters=12,
                                      step=0.12)
                else:
                    out = _relax(verts, faces, fixed, 1)
                if out is None:
                    self.report({'ERROR'},
                                "minimization needs numpy (Minimal "
                                "Surface Toolkit)")
                    return {'CANCELLED'}
                if self.polish > 0:
                    polished = _relax(out, faces, fixed, self.polish)
                    if polished is not None:
                        out = polished
            flat = [c for v in out for c in v]
            me.vertices.foreach_set('co', flat)
            me.update()
            # keep the bevelled knot curve child in sync
            for ch in obj.children:
                if ch.type == 'CURVE':
                    loops = _boundary_loops(me)
                    if len(loops) == len(ch.data.splines):
                        for sp, loop in zip(ch.data.splines, loops):
                            if len(sp.points) != len(loop):
                                continue
                            fl = []
                            for vi in loop:
                                x, y, z = out[vi]
                                fl.extend((x, y, z, 1.0))
                            sp.points.foreach_set('co', fl)
                    ch.data.update_tag()
            if a0:
                a1 = self._area(out, faces)
                self.report({'INFO'},
                            f"area {a0:.3f} -> {a1:.3f} "
                            f"({100 * (1 - a1 / a0):.1f}% less)")
            return {'FINISHED'}

    class VIEW3D_PT_seifert(bpy.types.Panel):
        bl_label = "Seifert Surfaces"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            lay.operator("mesh.seifert_surface_add", icon='MOD_SIMPLIFY')
            obj = context.object
            if obj is not None and "braid" in obj:
                lay.operator("mesh.seifert_minimize",
                             icon='MOD_SMOOTH')
                box = lay.box()
                box.label(text=f"Braid: {obj['braid']}")
                box.label(text=f"Strands {obj['strands']}, "
                               f"crossings {obj['crossings']}")
                box.label(text=f"Components {obj['boundary_components']}, "
                               f"genus {obj['genus']:g}")

    def _menu_func(self, context):
        self.layout.operator("mesh.seifert_surface_add",
                             icon='MOD_SIMPLIFY')

    _classes = (MESH_OT_seifert_add, MESH_OT_seifert_minimize,
                VIEW3D_PT_seifert)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for name, (label, braid) in PRESETS.items():
            word = parse_braid(braid)
            n, L, mu, g = braid_stats(word)
            verts, faces, fixed, n2, L2 = build_seifert(word)
            # Euler characteristic check: V - E + F == strands - crossings
            edges = set()
            for f in faces:
                for a in range(len(f)):
                    edges.add(tuple(sorted((f[a], f[(a + 1) % len(f)]))))
            chi = len(verts) - len(edges) + len(faces)
            ok = (chi == n - L)
            print(f"{label:18s} braid={braid:8s} strands={n} crossings={L} "
                  f"mu={mu} genus={g:g}  chi_mesh={chi} chi_expect={n - L} "
                  f"{'OK' if ok else 'MISMATCH'}")

        # dynamic_relax stability / free-boundary check (needs numpy)
        try:
            import numpy as np
            print("\n-- dynamic_relax (free-boundary membrane) --")
            for name, (label, braid) in PRESETS.items():
                word = parse_braid(braid)
                verts, faces, fixed, n, L = build_seifert(
                    word, m_res=48, rings=4, band_rows=8)
                V0 = np.array(verts)
                out = dynamic_relax(verts, faces, iterations=120)
                P = np.array(out)
                bmask, _ = _boundary_from_faces(faces, len(P))
                bmask = np.array(bmask, bool)
                bmove = float(np.linalg.norm(
                    (P - V0)[bmask], axis=1).mean())
                el = np.linalg.norm(
                    V0[np.array([e[0] for e in
                                 _topology(faces, len(P))[3]])]
                    - V0[np.array([e[1] for e in
                                   _topology(faces, len(P))[3]])],
                    axis=1).mean()
                finite = bool(np.isfinite(P).all())
                free = bmove > 0.1 * el
                print(f"{label:18s} finite={finite} "
                      f"boundary_moved={bmove/el:.2f} edges "
                      f"{'OK' if finite and free else 'FAIL'}")
        except ImportError:
            print("(numpy absent -- skipped dynamic_relax check)")
