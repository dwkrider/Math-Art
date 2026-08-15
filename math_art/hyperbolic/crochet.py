# Crocheted hyperbolic surfaces.
#
# Part of the Math Art hyperbolic engine (`math_art/hyperbolic/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Daina Taimina's models: crochet a row, then increase stitches at a
# fixed ratio every row, and the fabric is forced into a surface of
# constant negative curvature -- the first physical models of the
# hyperbolic plane anyone could handle.  The increase ratio sets the
# radius of curvature, so the whole family is one parameter.
#
# References:
# - D. W. Henderson and D. Taimina, "Crocheting the hyperbolic plane",
#   The Mathematical Intelligencer 23, 2001, pp. 17-28.

import math
import numpy as np


def _center(V):
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    return (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)


def _join_rings(sa, na, sb, nb):
    """Bridge two concentric rings of different vertex counts with a
    triangle strip, advancing whichever ring is 'behind' in angle."""
    faces = []
    ia = ib = 0
    while ia < na or ib < nb:
        a0 = sa + ia % na
        b0 = sb + ib % nb
        if ib >= nb or (ia < na and (ia + 1) / na <= (ib + 1) / nb):
            faces.append((a0, b0, sa + (ia + 1) % na))
            ia += 1
        else:
            faces.append((a0, b0, sb + (ib + 1) % nb))
            ib += 1
    return faces


def _crochet_mesh(ratio_n, rows, stitch, max_stitches, seed_scale=1.0,
                  grade=0.0, lobes=3):
    """Build the crochet mesh: exponentially growing stitch count per
    row (constant stitch size h), cascade-seeded ruffles, and true
    hyperbolic edge rest lengths.

    `grade` grades the curvature radius across the rows: grade>0 makes a
    flatter centre and a tighter, more ruffled rim -- Fathauer's cristate
    / crested-cactus look; grade<0 does the reverse (ruffled centre, calm
    rim). `lobes` sets the coarsest ruffle wavenumber, so the rim breaks
    into that many primary lobes (a multi-lobed hyperbolic form)."""
    h = stitch
    R0 = h / math.log(1.0 + 1.0 / ratio_n)
    rho_max = max(1e-6, rows * h)

    def _R(rho):
        # graded curvature radius: grade>0 flattens the CENTRE (larger R)
        # while the rim stays at the base curvature R0, so the ruffling
        # concentrates at the rim -- Fathauer's flat-centred, ruffled-rim
        # cristate form -- without over-stitching (spiking) the rim.
        # grade<0 does the reverse (ruffled centre, calmer rim).
        f = 1.0 + grade * (1.0 - rho / rho_max)
        return R0 * np.maximum(f, 0.4)

    rng = np.random.default_rng(0)
    m0 = max(2, int(lobes))                       # cascade base = lobes
    # index 0 is a centre vertex closing the magic-ring hole
    P, faces, rings, prev = [[0.0, 0.0, 0.0]], [], [], None
    for i in range(rows):
        rho = (i + 1) * h
        R = float(_R(rho))
        circ = 2.0 * math.pi * R * math.sinh(rho / R)
        n = int(max(6, min(max_stitches, round(circ / h))))
        start = len(P)
        theta = 2.0 * math.pi * np.arange(n) / n
        excess = circ / (2.0 * math.pi * rho)     # > 1
        z = np.zeros(n)
        if excess > 1.005:
            # arclength-matching target wavenumber (small-slope):
            # m_t = (2 rho / a) sqrt(excess - 1)
            m_t = (2.0 * rho / (0.7 * h)) * math.sqrt(excess - 1.0)
            ks = []
            k = 0
            while m0 * 2 ** k <= max(m_t, m0):
                ks.append(k)
                k += 1
            for k in ks:
                mk = m0 * 2 ** k
                if mk > n // 2 - 1:
                    break
                ak = 0.7 * h if k == ks[-1] else 0.35 * h
                z = z + ak * np.sin(mk * theta)
        z = seed_scale * z + 0.03 * rho + rng.normal(0.0, 0.02 * h, n)
        for j in range(n):
            P.append([rho * math.cos(theta[j]),
                      rho * math.sin(theta[j]), float(z[j])])
        if prev is not None:
            faces += _join_rings(prev[0], prev[1], start, n)
        else:                                     # fan-fill the centre
            for j in range(n):
                faces.append((0, start + j, start + (j + 1) % n))
        rings.append((start, n))
        prev = (start, n)
        if len(P) > 80000:
            break
    P = np.array(P)
    eset = set()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            eset.add((a, b) if a < b else (b, a))
    edges = list(eset)
    E0 = np.array([e[0] for e in edges])
    E1 = np.array([e[1] for e in edges])
    rr = np.linalg.norm(P[:, :2], axis=1)
    th = np.arctan2(P[:, 1], P[:, 0])
    Re = _R(0.5 * (rr[E0] + rr[E1]))              # per-edge graded R
    ch = (np.cosh(rr[E0] / Re) * np.cosh(rr[E1] / Re)
          - np.sinh(rr[E0] / Re) * np.sinh(rr[E1] / Re)
          * np.cos(th[E0] - th[E1]))
    REST = Re * np.arccosh(np.maximum(ch, 1.0))
    nbr = [set() for _ in range(len(P))]
    for a, b in edges:
        nbr[a].add(int(b))
        nbr[b].add(int(a))
    pin = np.arange(rings[0][1] + 1)              # centre + inner ring
    tris = np.array(faces, dtype=np.int64)
    return P, E0, E1, REST, nbr, tris, pin, P[pin].copy()


def _repel(P, nbr, radius, strength):
    """Short-range self-repulsion between non-adjacent vertices via a
    uniform spatial hash."""
    from collections import defaultdict
    keys = np.floor(P / radius).astype(np.int64)
    buckets = defaultdict(list)
    for idx in range(len(P)):
        buckets[(keys[idx, 0], keys[idx, 1], keys[idx, 2])].append(idx)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)]
    dP = np.zeros_like(P)
    r2 = radius * radius
    for idx in range(len(P)):
        kx, ky, kz = keys[idx]
        ni = nbr[idx]
        pi = P[idx]
        for dx, dy, dz in offs:
            for j in buckets.get((kx + dx, ky + dy, kz + dz), ()):
                if j <= idx or j in ni:
                    continue
                d = pi - P[j]
                l2 = float(d @ d)
                if 1e-12 < l2 < r2:
                    L = math.sqrt(l2)
                    f = strength * (radius - L) / radius * (d / L)
                    dP[idx] += f
                    dP[j] -= f
    return dP


def _bvh_decollide(P, faces, nbr, thick, strength):
    """One vertex-triangle self-collision pass using Blender's C-level
    BVHTree. Vertices within `thick` of a non-adjacent triangle are
    pushed out along the separation normal (with a reaction on the
    triangle). Blender only (uses mathutils)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    verts = [Vector((float(p[0]), float(p[1]), float(p[2])))
             for p in P]
    polys = [(int(f[0]), int(f[1]), int(f[2])) for f in faces]
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=True)
    dP = np.zeros_like(P)
    for v in range(len(P)):
        nv = nbr[v]
        for loc, nrm, idx, dist in tree.find_nearest_range(verts[v],
                                                           thick):
            if dist is None or dist >= thick:
                continue
            f = polys[idx]
            if v in f or f[0] in nv or f[1] in nv or f[2] in nv:
                continue
            dvec = P[v] - np.array([loc.x, loc.y, loc.z])
            L = float(np.linalg.norm(dvec))
            direction = (dvec / L if L > 1e-9
                         else np.array([nrm.x, nrm.y, nrm.z]))
            push = strength * (thick - dist) * direction
            dP[v] += push
            dP[f[0]] -= push / 3.0
            dP[f[1]] -= push / 3.0
            dP[f[2]] -= push / 3.0
    return P + 0.5 * dP


def _relax(P, E0, E1, REST, nbr, pin, pin_pos, iters, smooth,
           repel_r, repel_s, collide_fn=None, collide_every=40,
           anneal_L0=None, anneal_frac=0.75, stiff=1.0):
    """Jacobi stitch-length constraints + Laplacian bending + periodic
    self-repulsion (and optional BVHTree collision); inner ring pinned.
    With `anneal_L0` (flat starting edge lengths) the target rest grows
    smoothly from flat to hyperbolic over `anneal_frac` of the run, so
    the sheet buckles as a continuous deformation; `stiff` damps the
    compression term to stop compressed edges flinging into spikes.
    Defaults reproduce the plain relaxation exactly."""
    n = len(P)
    valence = np.zeros(n)
    np.add.at(valence, E0, 1.0)
    np.add.at(valence, E1, 1.0)
    valence = np.maximum(valence, 1.0)[:, None]
    for it in range(iters):
        if anneal_L0 is not None:
            t = min(1.0, it / max(1.0, anneal_frac * iters))
            rest = (1.0 - t) * anneal_L0 + t * REST
        else:
            t, rest = 1.0, REST
        d = P[E1] - P[E0]
        L = np.linalg.norm(d, axis=1)
        L = np.where(L < 1e-9, 1e-9, L)
        ratio = np.clip(1.0 - rest / L, -3.0, 1.0)
        corr = (stiff * ratio)[:, None] * d
        dP = np.zeros_like(P)
        np.add.at(dP, E0, corr)
        np.add.at(dP, E1, -corr)
        P = P + dP / valence
        if smooth > 0.0:
            nsum = np.zeros_like(P)
            np.add.at(nsum, E0, P[E1])
            np.add.at(nsum, E1, P[E0])
            P = P + smooth * (nsum / valence - P)
        if t > 0.9:
            if repel_s > 0.0 and it % 3 == 0 and it > 0:
                P = P + _repel(P, nbr, repel_r, repel_s)
            if collide_fn is not None and it % collide_every == 0:
                P = collide_fn(P)
        P[pin] = pin_pos
    return P






# named presets from gently wavy to tightly folded ("bendy")
PRESETS = {
    'WAVY': dict(ratio_n=6, rows=14, stitch=0.10, max_stitches=400,
                 iters=280, smooth=0.08, repel=0.5, collide=2),
    'RUFFLED': dict(ratio_n=4, rows=18, stitch=0.09, max_stitches=600,
                    iters=340, smooth=0.06, repel=0.5, collide=3),
    # tight-curvature presets: curvature-annealed + damped smooth base
    # (uncapped so outer rows are not truncated into spikes); Taimina
    # then collision-PACKS it into a ball.
    'BENDY': dict(ratio_n=3, rows=16, stitch=0.08, max_stitches=6000,
                  iters=380, smooth=0.14, repel=0.3, collide=0,
                  anneal=True, stiff=0.55, pack=45, pack_pull=0.010),
    'TAIMINA': dict(ratio_n=3, rows=16, stitch=0.08, max_stitches=6000,
                    iters=360, smooth=0.13, repel=0.3, collide=0,
                    anneal=True, stiff=0.55, pack=90, pack_pull=0.012),
}


def mean_curvature(V, faces):
    n = len(V)
    ang = np.zeros(n)
    area = np.zeros(n)
    deg = np.zeros(n)
    tris = [(f[0], f[1], f[2]) for f in faces]
    for a, b, c in tris:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u = V[j] - V[i]
            w = V[k] - V[i]
            cs = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)
                                 + 1e-12)
            ang[i] += math.acos(max(-1.0, min(1.0, cs)))
        ar = 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
        for i in (a, b, c):
            area[i] += ar / 3.0
            deg[i] += 1
    interior = deg >= 6
    defect = 2 * math.pi - ang
    ki = defect[interior] / np.maximum(area[interior], 1e-12)
    return float(np.median(ki)) if ki.size else 0.0
