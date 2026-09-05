# The Ambrus-Dancso node warp: softening a polyhedral tiling in closed form.
#
# Part of the Math Art soft-cell engine (`math_art/softcell/`).  Numpy only.
#
# Where `cell.py` softens ONE named family by solving for half-tangents, this
# module implements the general theorem: EVERY locally polyhedral tiling of
# space can be completely softened, and the softening is an explicit
# displacement field -- no solver, no equation to satisfy, just a formula
# applied to the mesh near each node.
#
# The construction, per node V:
#
#   1. Take the spherical vertex graph at V: one vertex per edge leaving V,
#      one face per cell meeting at V.  Two-colour its vertices with signs
#      +1/-1 so that NO FACE IS MONOCHROMATIC.  Such a colouring always
#      exists for a polyhedral graph -- via Thomassen's 2-list-colouring
#      result, or by four-colouring a triangulation and merging the colours
#      in pairs.  This is the only combinatorial work, and it is what lets
#      the theorem drop the Hamiltonian-circuit precondition of the earlier
#      edge-bending algorithm.
#   2. Pick an axis t_V through V that is generic: parallel to no edge
#      direction and lying in no plane spanned by two of them.
#   3. In cylindrical coordinates about that axis, displace every point
#      PARALLEL TO THE AXIS by
#
#          Phi(eps, alpha, h) = (eps, alpha, h + mu(h) T(eps, alpha)) .
#
# The profile phi(x) = x^(1/3) (1-x)^3 is what does the geometric work: its
# slope is infinite at x = 0, which drags each bent edge's half-tangent onto
# the axis, and phi''/(phi')^3 -> 0 there, which is the stronger condition
# that makes the union of two oppositely-coloured bent edges a C2 curve
# rather than merely a continuous one.  At the other end phi'(1) = phi''(1)
# = 0, so the deformation joins the untouched geometry smoothly.
#
# The displacement is bounded by r/4 and supported strictly inside disjoint
# balls of radius r about the nodes, so most of every face is left alone.
# That is a property of the theorem, not a shortcoming of this
# implementation: the result is a polyhedron whose corners have melted, not
# a droplet.
#
# References:
# - G. Ambrus and D. Dancso, "Softening locally polyhedral tilings",
#   arXiv:2604.18545 (2026).  Equation (5) is the profile phi, (8) the
#   translation tau, (10) the cutoff mu and (11) the local homeomorphism;
#   Lemma 2 is the two-colouring and Lemma 3 the properties of phi.
#   https://arxiv.org/abs/2604.18545
# - M. Karpinski and K. Piecuch, "Linear-time algorithm for vertex
#   2-coloring without monochromatic triangles on planar graphs", arXiv
#   preprint (2021) -- the practical route to the colouring for large
#   vertex figures; the exhaustive search below is enough for the small
#   ones that occur in a honeycomb.

import itertools
import math

import numpy as np

TAU = 2.0 * math.pi


def profile(x):
    """phi(x) = x^(1/3) (1-x)^3 on [0,1], zero beyond.  Equation (5)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > 0.0) & (x < 1.0)
    out[m] = np.cbrt(x[m]) * (1.0 - x[m]) ** 3
    return out


def blend(x):
    """psi(x) = x - sin(2 pi x)/(2 pi): flat to second order at both ends."""
    x = np.asarray(x, float)
    return x - np.sin(TAU * x) / TAU


def cutoff(h, r):
    """mu(h) of equation (10): 1 near the node, 0 beyond 3r/4."""
    a = np.abs(np.asarray(h, float))
    out = np.zeros_like(a)
    out[a < r / 4.0] = 1.0
    m = (a >= r / 4.0) & (a <= 3.0 * r / 4.0)
    out[m] = 1.5 - 2.0 * a[m] / r
    return out


def two_colour(n_edges, cells):
    """Signs +1/-1 on the edges so no cell at the node is monochromatic.

    `cells` lists, for each cell meeting the node, the indices of the edges
    on its boundary there.  Exhaustive over 2^n; vertex figures in a
    honeycomb have at most a handful of edges, so this is instant and
    removes any doubt about the colouring being valid.
    """
    for bits in range(1, 1 << n_edges):
        sig = [1 if (bits >> i) & 1 else -1 for i in range(n_edges)]
        if all(any(sig[e] > 0 for e in c) and any(sig[e] < 0 for e in c)
               for c in cells):
            return np.array(sig, float)
    raise ValueError("no valid two-colouring of the vertex figure: the "
                     "tiling is not locally polyhedral at this node")


def generic_axis(dirs, rng=None):
    """An axis parallel to no edge and in no plane spanned by two of them."""
    rng = rng or np.random.default_rng(0)
    for _ in range(200):
        t = rng.normal(size=3)
        t /= np.linalg.norm(t)
        if any(abs(abs(float(t @ d)) - 1.0) < 1e-3 for d in dirs):
            continue
        bad = False
        for u, v in itertools.combinations(dirs, 2):
            nrm = np.cross(u, v)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9:
                continue
            if abs(float(t @ (nrm / ln))) < 1e-3:
                bad = True
                break
        if not bad:
            return t
    raise ValueError("could not find a generic axis")


def node_warp(P, node, dirs, signs, axis, r, kappa, delta=0.35, depth=1.0):
    """Displace the points `P` that lie near one node.  Equation (11).

    Returns a new array; points outside the bending neighbourhood are
    returned unchanged, so the caller can apply this node by node.
    """
    P = np.asarray(P, float)
    z = axis / np.linalg.norm(axis)
    # an orthonormal frame with z along the axis
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(z @ tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, tmp)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    Q = P - node
    h = Q @ z
    px, py = Q @ x, Q @ y
    eps = np.hypot(px, py)
    alpha = np.arctan2(py, px) % TAU

    # edge azimuths, sorted
    ea = np.array([math.atan2(float(d @ y), float(d @ x)) % TAU
                   for d in dirs])
    order = np.argsort(ea)
    ea, sg = ea[order], np.asarray(signs, float)[order]
    n = len(ea)

    amp = (r / 4.0) * depth * profile(np.clip(eps / kappa, 0.0, 1.0))

    # tau: piecewise linear in alpha between consecutive edge azimuths
    idx = np.searchsorted(ea, alpha) - 1
    a0 = ea[idx % n]
    a1 = ea[(idx + 1) % n]
    span = (a1 - a0) % TAU
    span = np.where(span < 1e-12, TAU, span)
    lam = ((alpha - a0) % TAU) / span
    s0, s1 = sg[idx % n], sg[(idx + 1) % n]
    tau = amp * ((1.0 - lam) * s0 + lam * s1)

    # T: smooth tau in alpha while pinning its values on the edge rays
    dist0 = np.minimum((alpha - a0) % TAU, (a0 - alpha) % TAU)
    dist1 = np.minimum((alpha - a1) % TAU, (a1 - alpha) % TAU)
    near0 = dist0 <= delta
    near1 = dist1 <= delta
    T = tau.copy()
    w0 = blend(np.clip(dist0 / delta, 0.0, 1.0))
    w1 = blend(np.clip(dist1 / delta, 0.0, 1.0))
    T = np.where(near0, amp * s0 + w0 * (tau - amp * s0), T)
    T = np.where(near1, amp * s1 + w1 * (tau - amp * s1), T)

    T = np.where(eps > kappa, 0.0, T)
    disp = cutoff(h, r) * T
    return P + disp[:, None] * z[None, :]


def cube_vertex_figure():
    """The six edge directions and eight cells at a node of a cubic grid."""
    dirs = [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
            np.array([0, 1.0, 0]), np.array([0, -1.0, 0]),
            np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]
    cells = []
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                cells.append([0 + sx, 2 + sy, 4 + sz])
    return dirs, cells


def soften_cubic(n=1, subdiv=12, bend_radius=0.9, depth=1.0):
    """A softened cubic grid: `n`^3 cubes, every corner melted away.

    The worked example of the paper (its Figures 9 and 10), and the one
    case where the vertex figure is known exactly without needing cell
    adjacency from a mesh.  Faces are subdivided with a grading toward the
    nodes, because the profile's slope is infinite there and a uniform grid
    renders the tangency as a visible crease.
    """
    r = 0.5 * bend_radius
    kappa = 0.7 * r
    dirs, cells = cube_vertex_figure()
    signs = two_colour(len(dirs), cells)

    # graded samples on [0,1] with extra density at both ends
    t = np.linspace(0.0, 1.0, subdiv + 1)
    s = 0.5 - 0.5 * np.cos(math.pi * t)

    verts = []
    faces = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                o = np.array([i, j, k], float)
                base = len(verts)
                grids = []
                for ax in range(3):
                    for side in (0.0, 1.0):
                        g = np.empty((subdiv + 1, subdiv + 1, 3))
                        for p in range(subdiv + 1):
                            for q in range(subdiv + 1):
                                c = [0.0, 0.0, 0.0]
                                c[ax] = side
                                c[(ax + 1) % 3] = s[p]
                                c[(ax + 2) % 3] = s[q]
                                g[p, q] = o + np.array(c)
                        grids.append(g)
                for g in grids:
                    b = base + len(verts) - base
                    b = len(verts)
                    for p in range(subdiv + 1):
                        for q in range(subdiv + 1):
                            verts.append(g[p, q])
                    for p in range(subdiv):
                        for q in range(subdiv):
                            faces.append((b + p * (subdiv + 1) + q,
                                          b + p * (subdiv + 1) + q + 1,
                                          b + (p + 1) * (subdiv + 1) + q + 1,
                                          b + (p + 1) * (subdiv + 1) + q))
    V = np.array(verts, float)

    # warp about every lattice node the block touches
    rng = np.random.default_rng(3)
    axis = generic_axis(dirs, rng)
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                node = np.array([i, j, k], float)
                V = node_warp(V, node, dirs, signs, axis, r, kappa,
                              depth=depth)
    V = V - V.mean(axis=0)
    return V, faces


def _selftest():
    # phi: the properties Lemma 3 states, checked numerically
    xs = np.array([1e-9, 1e-6, 1e-3, 0.05, 0.1, 0.5, 0.9, 0.999])
    p = profile(xs)
    assert (p >= 0).all() and abs(float(profile(np.array([1.0]))[0])) < 1e-15
    assert abs(float(profile(np.array([0.0]))[0])) < 1e-15
    # phi' -> +infinity as x -> 0+
    slopes = []
    for x in (1e-4, 1e-6, 1e-8):
        d = (profile(np.array([x + x * 1e-3]))[0]
             - profile(np.array([x]))[0]) / (x * 1e-3)
        slopes.append(d)
    assert slopes[0] < slopes[1] < slopes[2], slopes
    assert slopes[-1] > 1e4, slopes
    print(f"warp: phi'(x) -> infinity as x -> 0 ({slopes[-1]:.1e})  OK")
    # phi peaks at 1/10, as the paper notes
    grid = np.linspace(1e-6, 1.0, 200001)
    assert abs(grid[int(np.argmax(profile(grid)))] - 0.1) < 1e-3
    print("warp: phi peaks at x = 1/10  OK")

    # psi is flat to second order at both ends
    for x0 in (0.0, 1.0):
        h = 1e-5
        d1 = (blend(np.array([x0 + h])) - blend(np.array([x0 - h])))[0] / (2 * h)
        assert abs(d1) < 1e-8, (x0, d1)
    assert abs(float(blend(np.array([1.0]))[0]) - 1.0) < 1e-12
    print("warp: psi(0)=0, psi(1)=1, psi'=0 at both ends  OK")

    # cutoff
    r = 1.0
    assert abs(float(cutoff(np.array([0.0]), r)[0]) - 1.0) < 1e-15
    assert abs(float(cutoff(np.array([r]), r)[0])) < 1e-15
    assert abs(float(cutoff(np.array([r / 2]), r)[0]) - 0.5) < 1e-12
    print("warp: cutoff mu is 1 at the node, 0 beyond 3r/4  OK")

    # the two-colouring of a cubic node exists and leaves no cell
    # monochromatic -- and note both colour classes come out disconnected,
    # the case the older edge-bending algorithm could not handle
    dirs, cells = cube_vertex_figure()
    sg = two_colour(len(dirs), cells)
    for c in cells:
        assert any(sg[e] > 0 for e in c) and any(sg[e] < 0 for e in c), c
    print(f"warp: cubic vertex figure two-coloured {sg.astype(int).tolist()}, "
          f"no monochromatic cell  OK")

    # the displacement respects the paper's bound |T| <= (r/4) phi(eps/kappa)
    rng = np.random.default_rng(5)
    P = rng.uniform(-0.6, 0.6, size=(4000, 3))
    node = np.zeros(3)
    axis = generic_axis(dirs, rng)
    r, kappa = 0.5, 0.35
    Q = node_warp(P, node, dirs, sg, axis, r, kappa)
    d = np.linalg.norm(Q - P, axis=1)
    assert d.max() <= r / 4.0 + 1e-12, d.max()
    # and it is the identity outside the bending neighbourhood
    far = np.linalg.norm(P - node, axis=1) > 1.2 * r
    assert np.abs(Q[far] - P[far]).max() < 1e-15
    print(f"warp: displacement bounded by r/4 ({d.max():.4f} <= "
          f"{r / 4:.4f}) and identity outside the neighbourhood  OK")

    # a softened cube block builds, stays closed in count, and actually
    # moved something
    V, F = soften_cubic(n=1, subdiv=6)
    assert len(V) == 6 * 49 and len(F) == 6 * 36, (len(V), len(F))
    V0, _F = soften_cubic(n=1, subdiv=6, depth=0.0)
    moved = float(np.abs(V - V0).max())
    assert moved > 1e-3, moved
    print(f"warp: softened cube built, max corner displacement "
          f"{moved:.4f}  OK")

    print("softcell.warp standalone tests passed")
