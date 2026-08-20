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
# - Thin-plate (bi-Laplacian) bending via the umbrella operator:
#   L. Kobbelt, S. Campagna, J. Vorsatz, H.-P. Seidel, "Interactive
#   multi-resolution modeling on arbitrary meshes", SIGGRAPH 1998,
#   pp. 105-114; M. Botsch, L. Kobbelt, "An intuitive framework for
#   real-time freeform modeling", ACM Trans. Graph. 23(3), 2004.
# - Coarse-to-fine continuation for buckling-unstable relaxation:
#   W. L. Briggs, V. E. Henson, S. F. McCormick, "A Multigrid
#   Tutorial", 2nd ed., SIAM, 2000 (nested iteration); K. A. Brakke,
#   "The Surface Evolver", Experimental Mathematics 1(2), 1992
#   (converge-then-refine workflow).

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


def _graded_R(rho, R0, grade, rho_max):
    """Graded curvature radius: grade>0 flattens the CENTRE (larger R)
    while the rim stays at the base curvature R0, so the ruffling
    concentrates at the rim -- Fathauer's flat-centred, ruffled-rim
    cristate form -- without over-stitching (spiking) the rim.
    grade<0 does the reverse (ruffled centre, calmer rim)."""
    f = 1.0 + grade * (1.0 - rho / rho_max)
    return R0 * np.maximum(f, 0.4)


def _rest_lengths(UV, E0, E1, R0, grade, rho_max):
    """Exact hyperbolic rest length per edge from the intrinsic flat
    chart UV (the crochet layout: polar radius = hyperbolic distance
    from the centre, angle = crochet angle), using the graded per-edge
    curvature radius.  This is the hyperbolic law of cosines
    cosh(d/R) = cosh(r0/R) cosh(r1/R) - sinh(r0/R) sinh(r1/R) cos(dth)
    evaluated at the edge's mean radius."""
    rr = np.linalg.norm(UV, axis=1)
    th = np.arctan2(UV[:, 1], UV[:, 0])
    Re = _graded_R(0.5 * (rr[E0] + rr[E1]), R0, grade, rho_max)
    ch = (np.cosh(rr[E0] / Re) * np.cosh(rr[E1] / Re)
          - np.sinh(rr[E0] / Re) * np.sinh(rr[E1] / Re)
          * np.cos(th[E0] - th[E1]))
    return Re * np.arccosh(np.maximum(ch, 1.0))


def _crochet_mesh(ratio_n, rows, stitch, max_stitches, seed_scale=1.0,
                  grade=0.0, lobes=3, R0=None):
    """Build the crochet mesh: exponentially growing stitch count per
    row (constant stitch size h), cascade-seeded ruffles, and true
    hyperbolic edge rest lengths.

    `grade` grades the curvature radius across the rows: grade>0 makes a
    flatter centre and a tighter, more ruffled rim -- Fathauer's cristate
    / crested-cactus look; grade<0 does the reverse (ruffled centre, calm
    rim). `lobes` sets the coarsest ruffle wavenumber, so the rim breaks
    into that many primary lobes (a multi-lobed hyperbolic form).

    `R0` overrides the base curvature radius (default: derived from the
    stitch size and increase ratio as h / ln(1 + 1/N)).  The override is
    what lets a COARSE mesh (large stitch) target the FINE metric during
    coarse-to-fine continuation -- without it, halving the resolution
    would also halve the curvature of the surface being solved."""
    h = stitch
    if R0 is None:
        R0 = h / math.log(1.0 + 1.0 / ratio_n)
    rho_max = max(1e-6, rows * h)

    def _R(rho):
        return _graded_R(rho, R0, grade, rho_max)

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
    REST = _rest_lengths(P[:, :2], E0, E1, R0, grade, rho_max)
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


def _bend_forces(P, E0, E1, valence):
    """Thin-plate (bi-Laplacian) bending energy and its EXACT gradient.

    E = 1/2 sum_i || (L P)_i ||^2   with the umbrella operator
    L = I - D^-1 A  (A the adjacency matrix, D the valence diagonal),
    so  dE/dP = L^T (L P) = (I - A D^-1)(L P).

    Unlike the plain Laplacian `smooth` term (gradient descent on the
    Dirichlet energy, which shrinks and flattens the sheet -- measured
    in research/hyperbolic-mesh-embedding-lessons.md), this penalizes
    the *curvature proxy* |L P| itself: short-wavelength crumple is
    damped ~ wavenumber^4 while large-scale waves and area are left
    nearly intact -- the standard thin-plate discretization (Kobbelt et
    al. 1998; Botsch-Kobbelt 2004).  Returns (E, G)."""
    nsum = np.zeros_like(P)
    np.add.at(nsum, E0, P[E1])
    np.add.at(nsum, E1, P[E0])
    Lp = P - nsum / valence
    E = 0.5 * float(np.sum(Lp * Lp))
    Q = Lp / valence
    asum = np.zeros_like(P)
    np.add.at(asum, E0, Q[E1])
    np.add.at(asum, E1, Q[E0])
    return E, Lp - asum


def _relax(P, E0, E1, REST, nbr, pin, pin_pos, iters, smooth,
           repel_r, repel_s, collide_fn=None, collide_every=40,
           anneal_L0=None, anneal_frac=0.75, stiff=1.0, bend=0.0):
    """Jacobi stitch-length constraints + Laplacian bending + periodic
    self-repulsion (and optional BVHTree collision); inner ring pinned.
    With `anneal_L0` (flat starting edge lengths) the target rest grows
    smoothly from flat to hyperbolic over `anneal_frac` of the run, so
    the sheet buckles as a continuous deformation; `stiff` damps the
    compression term to stop compressed edges flinging into spikes.
    `bend` > 0 adds thin-plate (bi-Laplacian) bending resistance --
    explicit gradient descent on `_bend_forces`, stable for bend < ~0.4
    (the umbrella's spectrum is [0, 2], so ||L^T L|| <= 4).
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
        if bend > 0.0:
            _eb, G = _bend_forces(P, E0, E1, valence)
            P = P - bend * G
        if t > 0.9:
            if repel_s > 0.0 and it % 3 == 0 and it > 0:
                P = P + _repel(P, nbr, repel_r, repel_s)
            if collide_fn is not None and it % collide_every == 0:
                P = collide_fn(P)
        P[pin] = pin_pos
    return P


def _nbr_from_edges(E0, E1, n):
    nbr = [set() for _ in range(n)]
    for a, b in zip(E0.tolist(), E1.tolist()):
        nbr[a].add(int(b))
        nbr[b].add(int(a))
    return nbr


def crochet_c2f(ratio_n=4, rows=18, stitch=0.09, max_stitches=600,
                iters=340, smooth=0.06, repel=0.5, bend=0.03,
                levels=1, sched=None, fine_frac=0.5, seed_scale=1.0,
                grade=0.0, lobes=3, anneal=False, stiff=1.0,
                collide_fn_factory=None, collide_every=40):
    """Coarse-to-fine continuation build+relax of the crochet sheet
    (S5 of research/geometric-solver-survey.md).

    Builds the crochet mesh at 1/2**levels of the target resolution --
    with the curvature radius R0 pinned to the TARGET stitch size, so
    the coarse solve relaxes toward the same hyperbolic surface -- and
    relaxes it there, where short-wavelength crumple modes simply do
    not exist.  Then, per level: midpoint 1->4 subdivision (positions
    AND the intrinsic chart UV), exact hyperbolic rest lengths
    recomputed from the refined chart (the metric constraint is
    re-projected, never interpolated), thresholds rescaled
    (repel radius follows the stitch: 0.4 * h_level), and a shorter
    re-relax that adds fine detail on top of the inherited
    long-wavelength waves.

    `sched` (levels+1 entries, coarse -> fine) overrides the default
    iteration schedule [iters, iters//2, ..., iters * fine_frac]; the
    coarse iterations are nearly free (16x fewer vertices two levels
    down), the fine ones dominate the wall clock.  `anneal` applies the
    flat->hyperbolic rest-length ramp at the COARSEST level only (the
    finer levels continue from an already-buckled state).
    `collide_fn_factory(tris, nbr, stitch_level)` may return a per-level
    collision pass (Blender layer).

    Measured defaults (tests/bench/results/c2f_crochet_sweep.md, the
    RUFFLED-scale sheet): levels=1 + bend=0.03 + fine_frac=0.5 reaches
    dihedral crumple 21.8 deg (one-shot: 44.4), hyperbolic edge error
    0.061 (one-shot: 0.057), and ZERO self-intersections (one-shot: 91)
    at 0.9x the one-shot wall time.  levels=2 gives still larger,
    smoother waves (dih ~14-16) but the deep lobes come into contact --
    use it with a collision pass (the Blender layer's BVH decollide).

    Returns a dict: P, tris, E0, E1, REST, nbr, pin, UV, schedule."""
    if levels < 1:
        raise ValueError("levels must be >= 1; use _relax for one-shot")
    try:
        from ..solver import refine as _refine
    except ImportError:                    # flat import outside the package
        from solver import refine as _refine
    R0 = stitch / math.log(1.0 + 1.0 / ratio_n)
    rho_max = max(1e-6, rows * stitch)
    rows_c = max(3, int(round(rows / 2 ** levels)))
    stitch_c = rho_max / rows_c            # preserve the domain exactly
    ms_c = max(12, int(math.ceil(max_stitches / 2 ** levels)))
    P, E0, E1, REST, nbr, tris, pin, pin_pos = _crochet_mesh(
        ratio_n, rows_c, stitch_c, ms_c, seed_scale=seed_scale,
        grade=grade, lobes=lobes, R0=R0)
    UV = P[:, :2].copy()                   # intrinsic chart of the layout
    if sched is None:
        sched = ([int(iters)] + [max(20, int(iters) // 2)] * (levels - 1)
                 + [max(20, int(round(iters * fine_frac)))])
    sched = [int(s) for s in sched]
    if len(sched) != levels + 1:
        raise ValueError(f"sched needs {levels + 1} entries "
                         f"(coarse->fine), got {len(sched)}")
    pinm = np.zeros(len(P), dtype=bool)
    pinm[pin] = True
    h_k = stitch_c
    schedule = []
    for lev in range(levels + 1):
        it_k = sched[lev]
        L0 = None
        if anneal and lev == 0:
            L0 = np.linalg.norm(UV[E1] - UV[E0], axis=1)
        cf = (collide_fn_factory(tris, nbr, h_k)
              if collide_fn_factory is not None else None)
        pin_k = np.where(pinm)[0]
        P = _relax(P, E0, E1, REST, nbr, pin_k, P[pin_k].copy(),
                   it_k, smooth, 0.4 * h_k, repel, collide_fn=cf,
                   collide_every=collide_every, anneal_L0=L0,
                   stiff=stiff, bend=bend)
        schedule.append(dict(level=lev, n_verts=len(P),
                             n_tris=len(tris), iters=it_k,
                             stitch=float(h_k)))
        if lev == levels:
            break
        P, tris, parents = _refine.subdivide(P, tris)
        UV = _refine.interp(UV, parents)
        E = _refine.edges_of(tris)
        E0, E1 = E[:, 0], E[:, 1]
        REST = _rest_lengths(UV, E0, E1, R0, grade, rho_max)
        nbr = _nbr_from_edges(E0, E1, len(P))
        pinm = np.concatenate(
            [pinm, pinm[parents[:, 0]] & pinm[parents[:, 1]]])
        h_k *= 0.5
    return dict(P=P, tris=tris, E0=E0, E1=E1, REST=REST, nbr=nbr,
                pin=np.where(pinm)[0], UV=UV, schedule=schedule)




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


def _selftest():
    rng = np.random.default_rng(5)

    # 1) thin-plate bending gradient vs central finite differences.
    #    The energy is QUADRATIC in P, so the central difference is
    #    exact up to roundoff; a large step (1e-2) makes roundoff --
    #    not truncation -- the only error source.
    n = 40
    P = rng.normal(size=(n, 3))
    E0 = np.arange(n)
    E1 = (np.arange(n) + 1) % n
    ex0 = rng.integers(0, n, 30)
    ex1 = (ex0 + rng.integers(2, n - 2, 30)) % n
    E0 = np.concatenate([E0, ex0])
    E1 = np.concatenate([E1, ex1])
    keep = E0 != E1
    E0, E1 = E0[keep], E1[keep]
    val = np.zeros(n)
    np.add.at(val, E0, 1.0)
    np.add.at(val, E1, 1.0)
    val = np.maximum(val, 1.0)[:, None]
    _e, G = _bend_forces(P, E0, E1, val)
    gscale = float(np.abs(G).max())
    eps, worst = 1e-2, 0.0
    for _ in range(30):
        i, c = int(rng.integers(0, n)), int(rng.integers(0, 3))
        Pp = P.copy()
        Pp[i, c] += eps
        Pm = P.copy()
        Pm[i, c] -= eps
        fd = (_bend_forces(Pp, E0, E1, val)[0]
              - _bend_forces(Pm, E0, E1, val)[0]) / (2 * eps)
        worst = max(worst, abs(fd - G[i, c]) / gscale)
    assert worst < 1e-9, f"bend gradient vs FD: {worst:.2e}"

    # 2) coarse-to-fine continuation invariants on a small sheet
    r = crochet_c2f(ratio_n=4, rows=8, stitch=0.12, max_stitches=200,
                    iters=120, smooth=0.06, repel=0.5, bend=0.03,
                    levels=2)
    P2, T2 = r["P"], r["tris"]
    assert np.isfinite(P2).all()
    # vertex/triangle counts grew ~4x per level
    sch = r["schedule"]
    assert len(sch) == 3
    assert sch[1]["n_tris"] == 4 * sch[0]["n_tris"]
    assert sch[2]["n_tris"] == 4 * sch[1]["n_tris"]
    # rest lengths positive and halving with the stitch
    assert (r["REST"] > 0).all()
    med = float(np.median(r["REST"]))
    assert 0.3 * sch[2]["stitch"] < med < 3.0 * sch[2]["stitch"]
    # the relaxed sheet buckled (3D) with negative median curvature
    K = mean_curvature(P2, T2)
    zext = float(P2[:, 2].max() - P2[:, 2].min())
    assert zext > 0.05 and K < 0
    # metric distortion no worse than the historical one-shot floor
    L = np.linalg.norm(P2[r["E1"]] - P2[r["E0"]], axis=1)
    rel = (L - r["REST"]) / r["REST"]
    err = float(np.sqrt(np.mean(rel * rel)))
    assert err < 0.12, f"c2f edge_err_rms {err:.3f}"
    print(f"crochet: bend-grad FD worst={worst:.2e}  c2f V={len(P2)} "
          f"err_rms={err:.4f} z={zext:.2f} K={K:+.2f} RESULT: OK")
