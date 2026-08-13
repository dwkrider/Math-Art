# Rope relaxation of knots and links.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# Rope relaxation: a knot diagram embedded from its braid closure is
# geometrically ugly, so the polyline is relaxed under a repulsion +
# smoothing flow that preserves the knot type (no strand may pass through
# another).  This is the discrete cousin of the "ropelength" / ideal-knot
# literature:
#   J. Cantarella, R. B. Kusner and J. M. Sullivan, "On the minimum
#       ropelength of knots and links", Inventiones Math. 150 (2002).
#   Y. Diao, C. Ernst and E. J. Janse van Rensburg, on thickness and
#       energy of knots (1990s-2000s).
# The exclusion window skips near neighbours along the strand, so
# repulsion acts only between genuinely distinct passes of the rope.

import math

import numpy as np


def relax_knot(P, iters=150, repel=0.35, step=0.25, smooth=0.5,
               excl=6):
    """Rounds the closure into a smooth knot: curve smoothing plus
    self-repulsion below `repel`, size renormalized each step so the
    strand cannot pass through itself for conservative settings."""
    P = np.asarray(P, dtype=float).copy()
    m = len(P)
    target = np.linalg.norm(P - P.mean(0), axis=1).mean()
    idx = np.arange(m)
    ring_d = np.abs(idx[:, None] - idx[None, :])
    ring_d = np.minimum(ring_d, m - ring_d)
    near = ring_d <= excl
    for _ in range(iters):
        Pn = (np.roll(P, 1, 0) + np.roll(P, -1, 0)) / 2
        P = P + smooth * (Pn - P)
        D = P[:, None, :] - P[None, :, :]
        d = np.linalg.norm(D, axis=-1)
        np.fill_diagonal(d, 1.0)
        mask = (d < repel) & ~near
        if mask.any():
            w = np.where(mask, (repel - d) / d, 0.0)
            P = P + step * (D * w[:, :, None]).sum(axis=1)
        c = P.mean(0)
        P -= c
        r = np.linalg.norm(P, axis=1).mean()
        if r > 1e-9:
            P *= target / r
    return P


def relax_link(comps, iters=100, repel=0.35, step=0.25, smooth=0.5,
               excl=6):
    """Multi-component version of prime_knot_generator.relax_knot:
    per-component curve smoothing plus repulsion between all strand
    pairs (both self and cross-component), globally renormalized.
    Repulsion never lets strands cross, so linked components stay
    linked; being linked, they also cannot drift apart."""
    comps = [np.asarray(c, dtype=float).copy() for c in comps]
    if iters <= 0:
        return comps
    sizes = [len(c) for c in comps]
    off = [0]
    for s in sizes:
        off.append(off[-1] + s)
    P = np.concatenate(comps)
    cid = np.concatenate([np.full(s, k, dtype=int)
                          for k, s in enumerate(sizes)])
    lidx = np.concatenate([np.arange(s) for s in sizes])
    same = cid[:, None] == cid[None, :]
    ring = np.abs(lidx[:, None] - lidx[None, :])
    msz = np.array(sizes)[cid]
    ring = np.minimum(ring, msz[:, None] - ring)
    near = same & (ring <= excl)          # neighbors along a strand
    target = np.linalg.norm(P - P.mean(0), axis=1).mean()
    for _ in range(iters):
        for k in range(len(sizes)):
            seg = P[off[k]:off[k + 1]]
            Pn = (np.roll(seg, 1, 0) + np.roll(seg, -1, 0)) / 2
            P[off[k]:off[k + 1]] = seg + smooth * (Pn - seg)
        D = P[:, None, :] - P[None, :, :]
        d = np.linalg.norm(D, axis=-1)
        np.fill_diagonal(d, 1.0)
        mask = (d < repel) & ~near
        if mask.any():
            w = np.where(mask, (repel - d) / d, 0.0)
            P += step * (D * w[:, :, None]).sum(axis=1)
        P -= P.mean(0)
        r = np.linalg.norm(P, axis=1).mean()
        if r > 1e-9:
            P *= target / r
    return [P[off[k]:off[k + 1]].copy() for k in range(len(sizes))]


def linking_number(A, B):
    """Gauss linking integral of two closed polylines (midpoint
    rule); rounds to +-lk for reasonably sampled curves."""
    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    da = np.roll(a, -1, 0) - a
    db = np.roll(b, -1, 0) - b
    R = (a + 0.5 * da)[:, None, :] - (b + 0.5 * db)[None, :, :]
    r3 = ((R * R).sum(-1)) ** 1.5
    cr = np.cross(da[:, None, :], db[None, :, :])
    return float(((cr * R).sum(-1) / r3).sum() / (4 * math.pi))


def _selftest():
    ok = True

    # linking_number on the two textbook cases.  Two circles in orthogonal
    # planes offset so they interlock are the Hopf link (lk = +-1); two
    # disjoint coplanar circles are the unlink (lk = 0).
    t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    zero = np.zeros_like(t)
    hopf_a = np.stack([c, s, zero], axis=1)
    hopf_b = np.stack([1.0 + c, zero, s], axis=1)
    lk = linking_number(hopf_a, hopf_b)
    good = abs(abs(lk) - 1.0) < 0.05
    ok &= good
    print(f"relax: linking_number Hopf = {lk:+.4f} (exp +-1) "
          f"{'OK' if good else 'FAIL'}")

    far_b = np.stack([4.0 + c, s, zero], axis=1)
    lk0 = linking_number(hopf_a, far_b)
    good = abs(lk0) < 0.05
    ok &= good
    print(f"relax: linking_number unlink = {lk0:+.4f} (exp 0) "
          f"{'OK' if good else 'FAIL'}")

    # Relaxation must SPREAD the rope: the closest approach between
    # non-adjacent points has to increase, while the curve stays closed,
    # finite and of comparable length (it must not collapse to a point).
    from .braid import braid_closure_points, parse_letters
    from .resample import resample_closed
    P0 = resample_closed(braid_closure_points(parse_letters('AAA')), 120)

    def min_far_gap(P, excl=6):
        n = len(P)
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        idx = np.arange(n)
        sep = np.abs(idx[:, None] - idx[None, :])
        sep = np.minimum(sep, n - sep)
        return float(d[sep > excl].min())

    def length(P):
        return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())

    g0, l0 = min_far_gap(P0), length(P0)
    P1 = relax_knot(P0.copy(), iters=60)
    g1, l1 = min_far_gap(P1), length(P1)
    good = (g1 > g0 and bool(np.all(np.isfinite(P1)))
            and 0.3 * l0 < l1 < 3.0 * l0)
    ok &= good
    print(f"relax: trefoil clearance {g0:.4f} -> {g1:.4f}, length "
          f"{l0:.2f} -> {l1:.2f} {'OK' if good else 'FAIL'}")

    # The link version must preserve the linking number -- that is the
    # topological gate: a relaxation that let a strand pass through
    # another would change it.
    comps = [hopf_a.copy(), hopf_b.copy()]
    before = linking_number(*comps)
    out = relax_link(comps, iters=40)
    after = linking_number(out[0], out[1])
    good = abs(after - before) < 0.1 and len(out) == 2
    ok &= good
    print(f"relax: link lk preserved {before:+.3f} -> {after:+.3f} "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("relax self-test failed")
