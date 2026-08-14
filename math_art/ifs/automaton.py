# Snowflake growth as a cellular automaton on the triangular lattice.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Reiter's model: each hexagonal cell holds a real amount of water.
# Cells with at least one frozen neighbour form the boundary and receive
# an addition constant gamma; the rest diffuse towards the local mean.  A
# cell freezes when its water reaches 1.  Two parameters -- the
# background vapour level beta and the addition gamma -- reproduce the
# whole morphology diagram, from stellar dendrites to solid plates, which
# is why the presets differ only in those numbers.
#
# References:
# - C. A. Reiter, "A local cellular model for snow crystal growth",
#   Chaos, Solitons and Fractals 23, 2005, pp. 1111-1119.
# - K. G. Libbrecht, "The physics of snow crystals", Reports on Progress
#   in Physics 68, 2005, pp. 855-895 -- the morphology diagram the
#   presets are named after.

import math

import numpy as np


# neighbour offsets in axial coords, ordered by direction angle 60*m
# (m = 0..5): east, 60, 120, 180, 240, 300 degrees
_DIR_OFF = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def simulate_reiter(alpha=1.0, beta=0.4, gamma=0.0001, radius=70,
                    max_steps=6000):
    """Run Reiter's snow-crystal automaton. Returns (s, frozen,
    hexdist): the water field, the frozen (crystal) mask, and the
    hex-distance-from-centre of every cell on a (2R'+1)^2 axial grid
    (R' = radius + 2, a margin so diffusion never wraps into the
    crystal)."""
    ar = radius + 2
    n = 2 * ar + 1
    q = np.arange(n)[:, None] - ar
    r = np.arange(n)[None, :] - ar
    hexdist = (np.abs(q) + np.abs(r) + np.abs(q + r)) // 2
    domain = hexdist <= radius
    s = np.where(domain, beta, 0.0).astype(np.float64)
    s[ar, ar] = 1.0                                  # frozen seed

    def neigh_sum(a):
        tot = np.zeros_like(a)
        for dq, dr in _DIR_OFF:
            tot += np.roll(np.roll(a, dq, axis=0), dr, axis=1)
        return tot

    def neigh_any(mask):
        tot = np.zeros_like(mask)
        for dq, dr in _DIR_OFF:
            tot |= np.roll(np.roll(mask, dq, axis=0), dr, axis=1)
        return tot

    steps_done = 0
    for step in range(max_steps):
        frozen = s >= 1.0
        receptive = (frozen | neigh_any(frozen)) & domain
        u = np.where(receptive, 0.0, s)
        v = np.where(receptive, s + gamma, 0.0)
        u = u + (alpha / 2.0) * (neigh_sum(u) / 6.0 - u)
        s = v + u
        s = np.where(domain, s, beta)                # outer reservoir
        steps_done = step + 1
        if step % 25 == 0 and np.any((s >= 1.0)
                                     & (hexdist >= radius - 1)):
            break
    return s, (s >= 1.0) & domain, hexdist, steps_done


def build_snowflake(frozen, s, spacing=1.0, base=0.4, relief=0.6,
                    s_cap=3.0, scale=1.0):
    """Turn a frozen mask + water field into a welded, two-sided
    hexagonal-relief plate. Each hexagon corner carries a single
    top height set by the ice mass of the frozen cells meeting there,
    so caps of neighbouring cells share corner vertices and the plate
    is one watertight surface; only the outline gets vertical walls.
    Returns (verts, faces)."""
    n = frozen.shape[0]
    ar = n // 2
    rc = spacing / math.sqrt(3.0)                    # hex circumradius
    corner = [(rc * math.cos(math.radians(60 * k + 30)),
               rc * math.sin(math.radians(60 * k + 30)))
              for k in range(6)]
    cells = [(int(i), int(j)) for i, j in np.argwhere(frozen)]

    def cell_xy(i, j):
        qq, rr = i - ar, j - ar
        return (qq + rr * 0.5) * spacing, rr * math.sqrt(3.0) / 2.0 * spacing

    def cell_mass(i, j):
        return min(max(float(s[i, j]), 1.0), s_cap)

    # pass 1: accumulate ice mass at each shared corner, so its top
    # height is the mean over the frozen cells that meet there
    acc = {}
    for i, j in cells:
        cx, cy = cell_xy(i, j)
        m = cell_mass(i, j)
        for k in range(6):
            key = (round(cx + corner[k][0], 4),
                   round(cy + corner[k][1], 4))
            tot, cnt = acc.get(key, (0.0, 0))
            acc[key] = (tot + m, cnt + 1)

    def top_z(key):
        tot, cnt = acc[key]
        return 0.5 * (base + relief * (tot / cnt - 1.0))

    verts = []
    faces = []
    vid = {}

    def V(x, y, z):
        vk = (round(x, 4), round(y, 4), round(z, 4))
        idx = vid.get(vk)
        if idx is None:
            idx = len(verts)
            vid[vk] = idx
            verts.append((x, y, z))
        return idx

    for i, j in cells:
        cx, cy = cell_xy(i, j)
        keys = [(round(cx + corner[k][0], 4),
                 round(cy + corner[k][1], 4)) for k in range(6)]
        pts = [(cx + corner[k][0], cy + corner[k][1]) for k in range(6)]
        zt = [top_z(keys[k]) for k in range(6)]
        top = [V(pts[k][0], pts[k][1], zt[k]) for k in range(6)]
        bot = [V(pts[k][0], pts[k][1], -zt[k]) for k in range(6)]
        faces.append(top[:])                         # +z cap
        faces.append(bot[::-1])                      # -z cap
        for e in range(6):
            k0, k1 = e, (e + 1) % 6
            di, dj = _DIR_OFF[(e + 1) % 6]            # cell across edge e
            ni, nj = i + di, j + dj
            nf = (0 <= ni < n and 0 <= nj < n and frozen[ni, nj])
            if not nf:                               # outline wall
                faces.append([top[k0], top[k1], bot[k1], bot[k0]])

    verts = np.asarray(verts, dtype=np.float64)
    if len(verts):
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ext = float((hi - lo).max())
        verts = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9
                                             else 1.0)
    return verts * scale, faces


# per-preset (alpha, beta, gamma) picked for distinct real habits
PRESETS = {
    'DENDRITE': ("Stellar Dendrite", 1.0, 0.4, 0.0001),
    'FERN': ("Fern Dendrite", 1.0, 0.35, 0.0001),
    'SECTORED': ("Sectored Plate", 1.0, 0.6, 0.001),
    'PLATE': ("Hexagonal Plate", 1.0, 0.9, 0.01),
    'STELLAR': ("Stellar Plate", 2.0, 0.5, 0.001),
}


def build_preset(kind='DENDRITE', radius=70, max_steps=6000,
                 base=0.4, relief=0.6, scale=1.0):
    label, alpha, beta, gamma = PRESETS[kind]
    s, frozen, _, steps = simulate_reiter(alpha, beta, gamma, radius,
                                          max_steps)
    verts, faces = build_snowflake(frozen, s, base=base,
                                   relief=relief, scale=scale)
    return verts, faces, int(frozen.sum()), steps
