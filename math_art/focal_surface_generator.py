# Focal surfaces: the surface of centres (the caustic by reflection of
# the normals).
#
# At every point of a surface, each principal curvature kappa_i defines a
# centre of curvature x + N / kappa_i along the normal.  The locus of
# those centres -- one sheet per principal curvature --
#
#     f_i(u, v) = x(u, v) + N(u, v) / kappa_i(u, v)
#
# is the FOCAL SURFACE, classically the "surface of centres"
# (centro-surface).  It is where the normal lines of the source surface
# focus, so it is the caustic of the surface's normal congruence, and it
# is singular exactly where the source is interesting: a sheet runs to
# infinity where kappa_i -> 0 (parabolic points), and the two sheets
# MEET at umbilics, where the principal curvatures coincide.  Those
# cuspidal edges and swallowtails are the point of drawing it.
#
# SOURCES ARE EXACT CHARTS ONLY, by design.  Principal curvatures are
# second derivatives; estimating them from a triangle mesh amplifies
# noise twice over, and the caustic is worst exactly where the estimate
# is worst (kappa near 0, and near umbilics).  A focal surface of a
# coarse mesh is noise wearing the shape of mathematics, so this
# generator evaluates the shape operator analytically from charts whose
# first and second fundamental forms are written out exactly -- the same
# reasoning by which mesh.quadric_add builds from charts rather than by
# contouring.
#
# Degeneracies worth knowing, each asserted by the self-test rather than
# hoped for:
#   - the SPHERE's sheets both collapse to its centre (every point is an
#     umbilic);
#   - the TORUS's sheets collapse to its centre circle and its axis --
#     the classical example of both sheets degenerating to curves, which
#     is Dupin's characterisation territory: a surface with ONE sheet a
#     curve is a canal surface (mesh.canal_surface_add), and one with
#     BOTH degenerate is a Dupin cyclide;
#   - the ELLIPSOID's sheets meet at its four umbilics: Cayley's
#     centro-surface, the classical showpiece;
#   - a minimal surface (the CATENOID here) has kappa_2 = -kappa_1, so
#     the source lies exactly midway between its two focal sheets.
#
# References:
# - G. Monge, "Application de l'analyse a la geometrie" (1807) -- lines
#   of curvature and the centres of curvature of a surface.
# - A. Cayley, "On the centro-surface of an ellipsoid", Trans. Cambridge
#   Phil. Soc. 12 (1873) -- the focal surface of the ellipsoid.
# - L. P. Eisenhart, "A Treatise on the Differential Geometry of Curves
#   and Surfaces" (1909), ch. on the surface of centres -- the classical
#   treatment of the two sheets and their tangency to the normals.
# - D. Hilbert and S. Cohn-Vossen, "Anschauliche Geometrie" (1932) --
#   centres of curvature and umbilics, read geometrically.
# - I. R. Porteous, "Geometric Differentiation" (1994) -- the modern
#   singularity-theory reading of focal sets, umbilics and ridges.
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), "Surface focale (developpee d'une surface)".

bl_info = {
    "name": "Focal Surface",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "The surface of centres: both sheets of principal "
                   "curvature centres of an exactly-known source "
                   "surface, singular at umbilics and parabolic points",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    from .curve_frames.sweep import closed_tube, sweep
    from .quadric_generator import fit
except ImportError:
    from curve_frames.sweep import closed_tube, sweep
    from quadric_generator import fit

TAU = 2.0 * math.pi

#: key -> (label, description); read by tools/surfdb (as source, via ast)
FOCAL_SOURCES = (
    ('ELLIPSOID', "Ellipsoid",
     "Cayley's centro-surface: the two sheets of the ellipsoid's "
     "centres of curvature, meeting at its four umbilics"),
    ('TORUS', "Torus",
     "Both sheets degenerate: the tube curvature focuses onto the "
     "centre circle and the ring curvature onto the axis -- the "
     "boundary case Dupin's cyclides generalise"),
    ('SPHERE', "Sphere",
     "Every point is an umbilic, so both sheets collapse to the "
     "centre point: the degenerate control case"),
    ('PARABOLOID', "Elliptic Paraboloid",
     "The mirror-dish caustic: two sheets with cuspidal edges over "
     "the dish's two principal directions"),
    ('SADDLE', "Hyperbolic Paraboloid",
     "Opposite-sign curvatures put one focal sheet on each side of "
     "the saddle"),
    ('CATENOID', "Catenoid",
     "A minimal surface: the principal curvatures are opposite, so "
     "the source sits exactly midway between its two focal sheets"),
    ('MONKEY_SADDLE', "Monkey Saddle",
     "A planar umbilic (both curvatures vanish at the origin) sends "
     "both sheets to infinity there; the clip distance cuts the "
     "resulting flare"),
)


# ---------------------------------------------------------------------------
# charts, with exact first and second derivatives


def _chart(source, u, v, a=1.0, b=0.85, c=0.7, ring=1.0, tube=0.35):
    """(P, Pu, Pv, Puu, Puv, Pvv) for one source chart, all exact."""
    cu, su = math.cos(u), math.sin(u)
    if source == 'TORUS':
        cv, sv = math.cos(v), math.sin(v)
        w = ring + tube * cv
        return ((w * cu, w * su, tube * sv),
                (-w * su, w * cu, 0.0),
                (-tube * sv * cu, -tube * sv * su, tube * cv),
                (-w * cu, -w * su, 0.0),
                (tube * sv * su, -tube * sv * cu, 0.0),
                (-tube * cv * cu, -tube * cv * su, -tube * sv))
    if source in ('SPHERE', 'ELLIPSOID'):
        if source == 'SPHERE':
            A = B = C = 1.0
        else:
            A, B, C = a, b, c
        cv, sv = math.cos(v), math.sin(v)
        return ((A * sv * cu, B * sv * su, C * cv),
                (-A * sv * su, B * sv * cu, 0.0),
                (A * cv * cu, B * cv * su, -C * sv),
                (-A * sv * cu, -B * sv * su, 0.0),
                (-A * cv * su, B * cv * cu, 0.0),
                (-A * sv * cu, -B * sv * su, -C * cv))
    if source == 'CATENOID':
        ch, sh = math.cosh(v), math.sinh(v)
        return ((ch * cu, ch * su, v),
                (-ch * su, ch * cu, 0.0),
                (sh * cu, sh * su, 1.0),
                (-ch * cu, -ch * su, 0.0),
                (-sh * su, sh * cu, 0.0),
                (ch * cu, ch * su, 0.0))
    # the graph charts z = f(u, v)
    if source == 'PARABOLOID':
        f = 0.5 * u * u / a + 0.5 * v * v / b
        fu, fv = u / a, v / b
        fuu, fuv, fvv = 1.0 / a, 0.0, 1.0 / b
    elif source == 'SADDLE':
        f = 0.5 * u * u / a - 0.5 * v * v / b
        fu, fv = u / a, -v / b
        fuu, fuv, fvv = 1.0 / a, 0.0, -1.0 / b
    elif source == 'MONKEY_SADDLE':
        k = 0.6
        f = k * (u ** 3 - 3.0 * u * v * v)
        fu, fv = 3.0 * k * (u * u - v * v), -6.0 * k * u * v
        fuu, fuv, fvv = 6.0 * k * u, -6.0 * k * v, -6.0 * k * u
    else:
        raise ValueError("unknown source %r" % source)
    return ((u, v, f), (1.0, 0.0, fu), (0.0, 1.0, fv),
            (0.0, 0.0, fuu), (0.0, 0.0, fuv), (0.0, 0.0, fvv))


#: (u0, u1, v0, v1, wrap_u, wrap_v) per source
_DOMAIN = {
    'TORUS':         (0.0, TAU, 0.0, TAU, True, True),
    'SPHERE':        (0.0, TAU, 0.12 * math.pi, 0.88 * math.pi,
                      True, False),
    'ELLIPSOID':     (0.0, TAU, 0.10 * math.pi, 0.90 * math.pi,
                      True, False),
    'PARABOLOID':    (-1.6, 1.6, -1.6, 1.6, False, False),
    'SADDLE':        (-1.2, 1.2, -1.2, 1.2, False, False),
    'CATENOID':      (0.0, TAU, -1.25, 1.25, True, False),
    'MONKEY_SADDLE': (-1.0, 1.0, -1.0, 1.0, False, False),
}


def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _dot(p, q):
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


def _cross(p, q):
    return (p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0])


def principal(source, u, v, **kw):
    """(P, N, kappa1, kappa2) at one parameter point, all analytic.

    kappa1 >= kappa2 with respect to the chart normal N; the shape
    operator is diagonalised through the fundamental forms, so no mesh
    estimate enters anywhere.
    """
    P, Pu, Pv, Puu, Puv, Pvv = _chart(source, u, v, **kw)
    E, F, G = _dot(Pu, Pu), _dot(Pu, Pv), _dot(Pv, Pv)
    n = _cross(Pu, Pv)
    m = math.sqrt(_dot(n, n))
    if m < 1e-15:
        return P, (0.0, 0.0, 1.0), 0.0, 0.0
    N = (n[0] / m, n[1] / m, n[2] / m)
    L, M, Q = _dot(Puu, N), _dot(Puv, N), _dot(Pvv, N)
    den = E * G - F * F
    if abs(den) < 1e-18:
        return P, N, 0.0, 0.0
    K = (L * Q - M * M) / den
    H = (E * Q - 2.0 * F * M + G * L) / (2.0 * den)
    disc = math.sqrt(max(H * H - K, 0.0))
    return P, N, H + disc, H - disc


def _reduce_locus(pts, NU, NV, wrap_u, wrap_v, tol):
    """Reduce a DEGENERATE sheet's grid points to their 1-D locus.

    `pts` maps (i, j) -> focal point of a sheet whose image has (near)
    zero area.  Coincident points are merged (rounding dedupe, then a
    union-find sweep so a value straddling a rounding boundary cannot
    split one geometric point into two nodes), grid adjacency collapses
    to a graph on the merged points, and the graph is walked into
    ordered chains.

    Returns (chains, isolated, ok): chains is a list of
    (ordered points, closed), isolated a list of lone points, and ok is
    False when the graph is not a clean union of paths and cycles -- in
    which case the caller falls back to emitting the raw grid.
    """
    reps, rep_pos = {}, []
    for p in pts.values():
        key = tuple(int(round(c / tol)) for c in p)
        if key not in reps:
            reps[key] = len(rep_pos)
            rep_pos.append(p)

    # union-find over the representatives, merging within 2 * tol
    parent = list(range(len(rep_pos)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(rep_pos)):
        for b in range(a + 1, len(rep_pos)):
            if max(abs(rep_pos[a][t] - rep_pos[b][t])
                   for t in range(3)) < 2.0 * tol:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    def node(p):
        return find(reps[tuple(int(round(c / tol)) for c in p)])

    edges, deg = set(), {}
    for (i, j) in pts:
        for (i2, j2) in (((i + 1) % NU, j), (i, (j + 1) % NV)):
            if i2 == 0 and i + 1 == NU and not wrap_u:
                continue
            if j2 == 0 and j + 1 == NV and not wrap_v:
                continue
            if (i2, j2) not in pts:
                continue
            a, b = node(pts[(i, j)]), node(pts[(i2, j2)])
            e = (min(a, b), max(a, b))
            if a != b and e not in edges:
                edges.add(e)
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1

    nodes = sorted({find(t) for t in range(len(rep_pos))})
    if any(deg.get(n, 0) > 2 for n in nodes):
        return [], [], False

    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    chains, isolated, seen = [], [], set()
    for start in nodes:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack.extend(adj[n])
        seen |= comp
        if len(comp) == 1:
            isolated.append(rep_pos[start])
            continue
        ends = [n for n in comp if deg.get(n, 0) == 1]
        cur = ends[0] if ends else min(comp)
        chain, prev = [cur], None
        while True:
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt or nxt[0] == chain[0]:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
        chains.append(([rep_pos[n] for n in chain], not ends))
    return chains, isolated, True


def _tube_chain(points, radius, sides=10, closed=False):
    """(verts, faces) -- a thin round tube along an ordered chain."""
    P = np.asarray(points, dtype=float)
    if closed and len(P) >= 3:
        vs, fs = closed_tube(P, radius, sides)
        return [tuple(v) for v in vs], [tuple(f) for f in fs]
    if len(P) < 2:
        return _marker(tuple(P[0]), radius)
    prof = [(radius * math.cos(TAU * k / sides),
             radius * math.sin(TAU * k / sides)) for k in range(sides)]
    vs, fs = sweep(P, prof)
    verts = [tuple(v) for v in vs]
    faces = [tuple(f) for f in fs]
    # flat disc caps so the rod is watertight
    for ring, pt, flip in ((0, P[0], True),
                           ((len(P) - 1) * sides, P[-1], False)):
        c = len(verts)
        verts.append(tuple(pt))
        for k in range(sides):
            k2 = (k + 1) % sides
            tri = (c, ring + k, ring + k2)
            faces.append(tri if flip else (c, ring + k2, ring + k))
    return verts, faces


def _marker(p, radius):
    """(verts, faces) -- a small octahedron marking a point locus."""
    r = radius
    verts = [(p[0] + r, p[1], p[2]), (p[0] - r, p[1], p[2]),
             (p[0], p[1] + r, p[2]), (p[0], p[1] - r, p[2]),
             (p[0], p[1], p[2] + r), (p[0], p[1], p[2] - r)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return verts, faces


def build_focal(source, nu=96, nv=64, sheets=(0, 1), clip=4.0,
                include_source=False, **kw):
    """(verts, faces, stats) -- focal sheets of one source chart.

    A grid point contributes to sheet i only while its focal distance
    |1 / kappa_i| stays within `clip`; faces are emitted where all four
    corners are valid, which is what trims the flares at parabolic
    points instead of meshing to infinity.

    THE CLIP CUTS, IT DOES NOT DROP.  Where a sheet crosses the clip
    radius -- and a sheet that flares to infinity crosses it
    diagonally through the grid cells -- each straddling cell is cut
    along the true crossing: the crossing parameter is found by
    bisection on the analytic chart, so the inserted rim vertex sits
    exactly at focal distance = clip, and the partial cell is emitted
    as the resulting polygon.  Dropping whole cells instead leaves a
    sawtooth rim of cell-sized teeth, which is how the monkey saddle
    first shipped.

    A DEGENERATE sheet -- one whose image has (near) zero area, like
    BOTH sheets of the torus, which are its centre circle and its axis
    -- is not emitted as its zero-area quad grid: that is invisible in
    the viewport, and it shipped that way once.  The sheet is reduced
    to the curve or point it actually is and drawn as a thin tube or
    an octahedral marker, and `stats["degenerate"]` records it so the
    operator reports what happened instead of staying silent.

    stats: {"cover": {sheet: fraction of the grid within clip},
            "degenerate": {sheet: "curve" | "point" | "unreduced"},
            "rim": {sheet: (crossing count, worst relative deviation
                            of a rim vertex's focal distance from
                            clip)}}.
    """
    u0, u1, v0, v1, wrap_u, wrap_v = _DOMAIN[source]
    NU = nu if wrap_u else nu + 1
    NV = nv if wrap_v else nv + 1
    us = [u0 + (u1 - u0) * i / nu for i in range(NU)]
    vs = [v0 + (v1 - v0) * j / nv for j in range(NV)]

    # evaluate the whole grid once: focal candidates per sheet, plus
    # the source's own extent, which scales the locus thickness
    F = {0: {}, 1: {}}
    lo, hi = [1e30] * 3, [-1e30] * 3
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            P, N, k1, k2 = principal(source, u, v, **kw)
            for t in range(3):
                lo[t] = min(lo[t], P[t])
                hi[t] = max(hi[t], P[t])
            for sheet, k in ((0, k1), (1, k2)):
                if abs(k) * clip > 1.0:
                    d = 1.0 / k
                    F[sheet][(i, j)] = (P[0] + d * N[0],
                                        P[1] + d * N[1],
                                        P[2] + d * N[2])
    diag = max(math.sqrt(sum((hi[t] - lo[t]) ** 2 for t in range(3))),
               1e-9)

    verts, faces = [], []

    def _emit(sub):
        vs2, fs2 = sub
        base = len(verts)
        verts.extend(vs2)
        faces.extend(tuple(base + t for t in f) for f in fs2)

    total = NU * NV
    cover = {s: 0.0 for s in sheets}
    degenerate = {}
    rim = {}
    duu = (u1 - u0) / nu
    dvv = (v1 - v0) / nv
    for sheet in sheets:
        pts = F[sheet]
        cover[sheet] = len(pts) / float(total)
        if not pts:
            continue
        quads = []
        for i in range(NU if wrap_u else NU - 1):
            i2 = (i + 1) % NU
            for j in range(NV if wrap_v else NV - 1):
                j2 = (j + 1) % NV
                q = [(i, j), (i2, j), (i2, j2), (i, j2)]
                if all(c in pts for c in q):
                    quads.append(q)
        area = 0.0
        for q in quads:
            a, b, c, d = (np.asarray(pts[t]) for t in q)
            area += 0.5 * float(np.linalg.norm(np.cross(c - a, d - b)))

        def _emit_grid():
            idx = {}
            for gc, p in pts.items():
                idx[gc] = len(verts)
                verts.append(p)
            faces.extend(tuple(idx[c] for c in q) for q in quads)

        if area > 1e-7 * diag * diag:
            # A genuine 2-D sheet.  Emit it cell by cell, CUTTING every
            # cell the clip locus crosses: dropping whole cells would
            # leave a sawtooth rim of cell-sized teeth.  The crossing
            # on each straddling grid edge is found by bisection in
            # PARAMETER space against the analytic chart, so the
            # inserted rim vertex sits on the clip locus itself.
            idx = {}
            for gc, p in pts.items():
                idx[gc] = len(verts)
                verts.append(p)
            cache = {}
            rim_n, rim_err = 0, 0.0

            def _crossing(ca_raw, cb_raw):
                """Rim vertex on the grid edge inside->outside."""
                nonlocal rim_n, rim_err
                ka = (ca_raw[0] % NU, ca_raw[1] % NV)
                kb = (cb_raw[0] % NU, cb_raw[1] % NV)
                ckey = (min(ka, kb), max(ka, kb))
                if ckey in cache:
                    return cache[ckey]
                ua, va = u0 + duu * ca_raw[0], v0 + dvv * ca_raw[1]
                ub, vb = u0 + duu * cb_raw[0], v0 + dvv * cb_raw[1]
                ta, tb = 0.0, 1.0
                for _ in range(40):
                    tm = 0.5 * (ta + tb)
                    um, vm = ua + (ub - ua) * tm, va + (vb - va) * tm
                    _P, _N, k1, k2 = principal(source, um, vm, **kw)
                    if abs((k1, k2)[sheet]) * clip > 1.0:
                        ta = tm
                    else:
                        tb = tm
                um, vm = ua + (ub - ua) * ta, va + (vb - va) * ta
                P, N, k1, k2 = principal(source, um, vm, **kw)
                d = 1.0 / (k1, k2)[sheet]
                rim_n += 1
                rim_err = max(rim_err, abs(abs(d) - clip) / clip)
                vi = len(verts)
                verts.append((P[0] + d * N[0], P[1] + d * N[1],
                              P[2] + d * N[2]))
                cache[ckey] = vi
                return vi

            for i in range(NU if wrap_u else NU - 1):
                for j in range(NV if wrap_v else NV - 1):
                    raw = [(i, j), (i + 1, j), (i + 1, j + 1),
                           (i, j + 1)]
                    keys = [(a % NU, b % NV) for a, b in raw]
                    ins = [kk in pts for kk in keys]
                    if not any(ins):
                        continue
                    poly = []
                    for t in range(4):
                        t2 = (t + 1) % 4
                        if ins[t]:
                            poly.append(idx[keys[t]])
                        if ins[t] != ins[t2]:
                            poly.append(_crossing(raw[t], raw[t2])
                                        if ins[t]
                                        else _crossing(raw[t2], raw[t]))
                    if len(poly) >= 3:
                        faces.append(tuple(poly))
            if rim_n:
                rim[sheet] = (rim_n, rim_err)
            continue

        # the sheet is 1-dimensional (or a point): draw its locus
        chains, isolated, ok = _reduce_locus(
            pts, NU, NV, wrap_u, wrap_v, tol=1e-5 * diag)
        if not ok:
            _emit_grid()   # better the raw grid than nothing; and say so
            degenerate[sheet] = "unreduced"
            continue
        kind = "point"
        for chain, closed in chains:
            if len(chain) >= 2:
                kind = "curve"
            _emit(_tube_chain(chain, 0.02 * diag, closed=closed))
        for p in isolated:
            _emit(_marker(p, 0.035 * diag))
        degenerate[sheet] = kind

    if include_source:
        base = len(verts)
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                P, _Pu, _Pv, _a, _b, _c = _chart(source, u, v, **kw)
                verts.append(P)
        for i in range(NU if wrap_u else NU - 1):
            i2 = (i + 1) % NU
            for j in range(NV if wrap_v else NV - 1):
                j2 = (j + 1) % NV
                faces.append((base + i * NV + j, base + i2 * NV + j,
                              base + i2 * NV + j2, base + i * NV + j2))

    return verts, faces, {"cover": cover, "degenerate": degenerate,
                          "rim": rim}


# ---------------------------------------------------------------------------


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_focal_surface_add(bpy.types.Operator):
        """Add a focal surface: the two sheets of centres of principal
        curvature of an exactly-known source surface -- the caustic of
        its normals, singular at umbilics and parabolic points"""
        bl_idname = "mesh.focal_surface_add"
        bl_label = "Focal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        source: EnumProperty(
            name="Source",
            items=[(k, lab, desc) for k, lab, desc in FOCAL_SOURCES],
            default='ELLIPSOID',
            description="The surface whose centres of curvature are "
                        "traced. Charts are exact, so the curvatures "
                        "are analytic, never mesh estimates")
        sheets: EnumProperty(
            name="Sheets",
            items=[('BOTH', "Both", "Both sheets of centres"),
                   ('FIRST', "First",
                    "Only the sheet of the larger principal curvature"),
                   ('SECOND', "Second",
                    "Only the sheet of the smaller principal "
                    "curvature")],
            default='BOTH',
            description="Which sheet of the surface of centres to build")
        clip: FloatProperty(
            name="Clip Distance", default=3.0, min=0.2, max=50.0,
            description="Largest focal distance kept. Near parabolic "
                        "points 1/curvature runs to infinity; this cuts "
                        "the flare instead of meshing it")
        a: FloatProperty(
            name="Semi-axis a", default=1.0, min=0.1, max=4.0,
            description="First semi-axis of the ellipsoid, or the "
                        "first curvature scale of the graph sources")
        b: FloatProperty(
            name="Semi-axis b", default=0.85, min=0.1, max=4.0,
            description="Second semi-axis of the ellipsoid, or the "
                        "second curvature scale of the graph sources")
        c: FloatProperty(
            name="Semi-axis c", default=0.7, min=0.1, max=4.0,
            description="Polar semi-axis of the ellipsoid. Distinct "
                        "semi-axes keep the four umbilics visible")
        ring_radius: FloatProperty(
            name="Ring Radius", default=1.0, min=0.2, max=4.0,
            description="Centre-circle radius of the torus source")
        tube_radius: FloatProperty(
            name="Tube Radius", default=0.35, min=0.05, max=2.0,
            description="Tube radius of the torus source")
        segments_u: IntProperty(
            name="Segments U", default=128, min=8, max=512)
        segments_v: IntProperty(
            name="Segments V", default=96, min=8, max=512)
        include_source: BoolProperty(
            name="Source Surface", default=False,
            description="Also build the source surface in the same "
                        "mesh, for reading the sheets against it")
        size: FloatProperty(
            name="Size", default=1.0, min=0.01, max=100.0,
            description="Half the largest extent of the finished object")

        def execute(self, context):
            sheets = {'BOTH': (0, 1), 'FIRST': (0,),
                      'SECOND': (1,)}[self.sheets]
            kw = {}
            if self.source == 'ELLIPSOID':
                kw = dict(a=self.a, b=self.b, c=self.c)
            elif self.source in ('PARABOLOID', 'SADDLE'):
                kw = dict(a=self.a, b=self.b)
            elif self.source == 'TORUS':
                kw = dict(ring=self.ring_radius, tube=self.tube_radius)
            verts, faces, stats = build_focal(
                self.source, self.segments_u, self.segments_v,
                sheets=sheets, clip=self.clip,
                include_source=self.include_source, **kw)
            if not verts or not faces:
                self.report({'ERROR'},
                            "every focal point lies beyond the clip "
                            "distance; raise Clip Distance")
                return {'CANCELLED'}
            verts = fit(verts, self.size)

            me = bpy.data.meshes.new("Focal Surface")
            me.from_pydata(verts, [], faces)
            me.validate()
            me.update()
            obj = bpy.data.objects.new("Focal Surface", me)
            context.collection.objects.link(obj)
            context.view_layer.objects.active = obj
            obj.select_set(True)

            cover = ", ".join("sheet %d: %d%% within clip"
                              % (s + 1, round(100 * stats["cover"][s]))
                              for s in sorted(stats["cover"]))
            note = ""
            for s in sorted(stats["degenerate"]):
                kind = stats["degenerate"][s]
                if kind == "curve":
                    note += ("; sheet %d is DEGENERATE -- a curve, not "
                             "a surface -- drawn as a thin tube" % (s + 1))
                elif kind == "point":
                    note += ("; sheet %d is DEGENERATE -- a single "
                             "point -- drawn as a small marker" % (s + 1))
                else:
                    note += ("; sheet %d is degenerate and could not "
                             "be reduced to a curve; raw (zero-area) "
                             "grid emitted" % (s + 1))
            self.report({'INFO'},
                        "Focal surface of %s: %d verts, %d faces (%s%s)"
                        % (dict((k, l) for k, l, _d in
                                FOCAL_SOURCES)[self.source],
                           len(verts), len(faces), cover, note))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'source')
            lay.prop(self, 'sheets')
            lay.prop(self, 'clip')
            if self.source == 'ELLIPSOID':
                for k in ('a', 'b', 'c'):
                    lay.prop(self, k)
            elif self.source in ('PARABOLOID', 'SADDLE'):
                lay.prop(self, 'a')
                lay.prop(self, 'b')
            elif self.source == 'TORUS':
                lay.prop(self, 'ring_radius')
                lay.prop(self, 'tube_radius')
            for k in ('segments_u', 'segments_v', 'include_source',
                      'size'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator(MESH_OT_focal_surface_add.bl_idname,
                             text="Focal Surface",
                             icon='SURFACE_NSURFACE')

    def register():
        bpy.utils.register_class(MESH_OT_focal_surface_add)
        if hasattr(bpy.types, "VIEW3D_MT_mesh_add"):
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if hasattr(bpy.types, "VIEW3D_MT_mesh_add"):
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_focal_surface_add)


# ---------------------------------------------------------------------------


def _selftest():
    """Numeric self-test; raises on failure.

    The gates are the closed forms: the sphere's and torus's degenerate
    sheets, the ellipsoid's vertex curvatures and umbilics, and the
    catenoid's minimality -- never a vertex count or a look.
    """
    ok = True

    # 1. sphere: both sheets collapse to the centre, and the BUILDER
    # must say so -- emitting small point markers there, not an
    # invisible zero-area grid.  (At an umbilic disc = sqrt(H^2 - K)
    # takes the square root of pure roundoff, so kappa carries ~1e-8 of
    # noise by construction; the marker radius dwarfs that.)
    verts, faces, st = build_focal('SPHERE', 48, 32, clip=10.0)
    worst = max(math.sqrt(x * x + y * y + z * z) for x, y, z in verts)
    good = (verts and faces and worst < 0.15
            and st["degenerate"] == {0: "point", 1: "point"})
    ok &= good
    print("focal: the sphere's sheets collapse to its centre, drawn "
          "as point markers (worst |f| = %.3f) %s"
          % (worst, "OK" if good else "FAIL"))

    # 2. the CLOSED-FORM surface of revolution: torus R = 1, r = 0.35.
    # First the mathematics, at the principal() level: one sheet's
    # focal points must be the centre circle (radius R in z = 0), the
    # other's the axis.  Which sheet is which depends on the normal's
    # sign, so each point may match either oracle -- but every point
    # must match one, and BOTH oracles must be hit.
    hit_circle = hit_axis = 0
    worst = 0.0
    for i in range(64):
        for j in range(48):
            u, v = TAU * i / 64.0, TAU * j / 48.0
            P, N, k1, k2 = principal('TORUS', u, v)
            for k in (k1, k2):
                if abs(k) < 0.1:
                    continue
                x, y, z = (P[t] + N[t] / k for t in range(3))
                d_circle = abs(math.hypot(x, y) - 1.0) + abs(z)
                d_axis = math.hypot(x, y)
                if d_circle < d_axis:
                    hit_circle += 1
                    worst = max(worst, d_circle)
                else:
                    hit_axis += 1
                    worst = max(worst, d_axis)
    good = worst < 1e-9 and hit_circle > 0 and hit_axis > 0
    ok &= good
    print("focal: the torus's focal points are its centre circle "
          "(%d pts) and its axis (%d pts), worst deviation %.1e %s"
          % (hit_circle, hit_axis, worst, "OK" if good else "FAIL"))

    # 2b. ...and then the MESH: both sheets are 1-dimensional, so the
    # builder must flag them degenerate and emit thin tubes AROUND
    # those loci -- a non-empty, visible mesh.  The torus once shipped
    # as a soup of zero-area quads that looked like an empty object:
    # the oracle above passed while the operator produced nothing
    # visible, so this check holds the MESH to the loci, not just the
    # mathematics.
    verts, faces, st = build_focal('TORUS', 64, 48, clip=10.0)
    tube_r = 0.02 * math.sqrt(2.7 ** 2 + 2.7 ** 2 + 0.7 ** 2)
    near_circle = near_axis = 0
    worst = 0.0
    for x, y, z in verts:
        d_circle = math.sqrt((math.hypot(x, y) - 1.0) ** 2 + z * z)
        d_axis = math.hypot(x, y)
        d = min(d_circle, d_axis)
        worst = max(worst, d)
        if d_circle < d_axis:
            near_circle += 1
        else:
            near_axis += 1
    good = (verts and faces and worst < tube_r + 1e-6
            and near_circle > 0 and near_axis > 0
            and st["degenerate"] == {0: "curve", 1: "curve"})
    ok &= good
    print("focal: the torus's degenerate sheets are drawn as tubes on "
          "the circle (%d verts) and the axis (%d verts), worst "
          "off-locus %.4f <= tube radius %.4f %s"
          % (near_circle, near_axis, worst, tube_r,
             "OK" if good else "FAIL"))

    # 3. ellipsoid: at the end of the major axis (u = 0, v = pi/2) the
    # principal curvatures are a/b^2 and a/c^2 in closed form
    a, b, c = 1.0, 0.85, 0.7
    _P, _N, k1, k2 = principal('ELLIPSOID', 0.0, 0.5 * math.pi,
                               a=a, b=b, c=c)
    want = sorted((a / (b * b), a / (c * c)))
    got = sorted((abs(k1), abs(k2)))
    err = max(abs(g - w) for g, w in zip(got, want))
    good = err < 1e-12
    ok &= good
    print("focal: ellipsoid vertex curvatures match a/b^2, a/c^2 "
          "(error %.1e) %s" % (err, "OK" if good else "FAIL"))

    # 4. ellipsoid umbilics: the sheets MEET.  min |k1 - k2| over the
    # grid must approach zero (the umbilics lie in the x-z plane), and
    # the corresponding focal points must coincide.
    best, best_gap = None, None
    u0, u1, v0, v1, _wu, _wv = _DOMAIN['ELLIPSOID']
    for i in range(160):
        for j in range(80):
            u = u0 + (u1 - u0) * i / 160.0
            v = v0 + (v1 - v0) * j / 80.0
            P, N, k1, k2 = principal('ELLIPSOID', u, v, a=a, b=b, c=c)
            if abs(k1) < 1e-9 or abs(k2) < 1e-9:
                continue
            spread = abs(k1 - k2)
            if best is None or spread < best:
                best = spread
                f1 = tuple(P[t] + N[t] / k1 for t in range(3))
                f2 = tuple(P[t] + N[t] / k2 for t in range(3))
                best_gap = math.dist(f1, f2)
    good = best is not None and best < 0.02 and best_gap < 0.02
    ok &= good
    print("focal: the ellipsoid's sheets meet at an umbilic "
          "(min |k1-k2| = %.4f, sheet gap %.4f) %s"
          % (best, best_gap, "OK" if good else "FAIL"))

    # 5. catenoid: H = 0 analytically, so the source lies exactly
    # midway between its focal sheets
    worst_h = worst_mid = 0.0
    for i in range(24):
        for j in range(16):
            u = TAU * i / 24.0
            v = -1.2 + 2.4 * j / 15.0
            P, N, k1, k2 = principal('CATENOID', u, v)
            worst_h = max(worst_h, abs(k1 + k2))
            if abs(k1) > 1e-9 and abs(k2) > 1e-9:
                mid = tuple(P[t] + 0.5 * (N[t] / k1 + N[t] / k2)
                            for t in range(3))
                worst_mid = max(worst_mid, math.dist(mid, P))
    good = worst_h < 1e-12 and worst_mid < 1e-9
    ok &= good
    print("focal: the catenoid is minimal (max |k1+k2| = %.1e) and "
          "sits midway between its sheets (%.1e) %s"
          % (worst_h, worst_mid, "OK" if good else "FAIL"))

    # 6. clipping: the monkey saddle's planar umbilic flares to
    # infinity; with a finite clip every emitted vertex must stay
    # finite and within clip of its source point by construction
    verts, faces, stats = build_focal('MONKEY_SADDLE', 48, 48, clip=2.5)
    good = (all(all(math.isfinite(t) for t in v) for v in verts)
            and faces and 0.0 < stats["cover"][0] < 1.0)
    ok &= good
    print("focal: the monkey saddle's planar umbilic is clipped, "
          "finite mesh (%d%% of sheet 1 within clip) %s"
          % (round(100 * stats["cover"][0]), "OK" if good else "FAIL"))

    # 6b. clipped rims are CUT along the clip locus, not
    # stair-stepped.  Dropping whole cells at the clip left a sawtooth
    # rim of cell-sized teeth (observed on the monkey saddle and the
    # elliptic paraboloid), with long spikes wherever the sheet
    # crossed the clip radius at a shallow angle.  Every source whose
    # sheets run to infinity must now insert rim vertices, and every
    # inserted rim vertex must sit AT focal distance = clip.
    for src in ('MONKEY_SADDLE', 'PARABOLOID', 'SADDLE', 'CATENOID'):
        _v, _f, stats = build_focal(src, 48, 48, clip=2.5)
        rims = stats["rim"]
        n = sum(cnt for cnt, _e in rims.values())
        worst = max((e for _c, e in rims.values()), default=1.0)
        good = bool(rims) and n > 20 and worst < 1e-3
        ok &= good
        print("focal: %-14s rim is cut on the clip locus "
              "(%d rim verts, worst |d - clip|/clip = %.1e) %s"
              % (src, n, worst, "OK" if good else "FAIL"))

    # 7. EVERY source in the enum yields a non-empty mesh with real
    # area through the builder.  The torus taught this lesson: a gate
    # that checks the focal points are in the right PLACE says nothing
    # about whether the emitted mesh is visible.  Zero-area output must
    # never ship again for any source, present or future.
    for key, _lab, _desc in FOCAL_SOURCES:
        verts, faces, stats = build_focal(key, 48, 40, clip=3.0)
        area = 0.0
        V = [np.asarray(v) for v in verts]
        for f in faces:
            for t in range(1, len(f) - 1):
                area += 0.5 * float(np.linalg.norm(
                    np.cross(V[f[t]] - V[f[0]], V[f[t + 1]] - V[f[0]])))
        good = bool(verts) and bool(faces) and area > 1e-6
        ok &= good
        print("focal: %-14s non-empty mesh with real area "
              "(%d verts, %d faces, area %.4f%s) %s"
              % (key, len(verts), len(faces), area,
                 "".join(", sheet %d %s" % (s + 1, k)
                         for s, k in sorted(stats["degenerate"].items())),
                 "OK" if good else "FAIL"))

    # 8. paraboloid apex: kappa = 1/a and 1/b in closed form
    _P, _N, k1, k2 = principal('PARABOLOID', 0.0, 0.0, a=0.8, b=1.5)
    err = max(abs(max(abs(k1), abs(k2)) - 1.0 / 0.8),
              abs(min(abs(k1), abs(k2)) - 1.0 / 1.5))
    good = err < 1e-12
    ok &= good
    print("focal: paraboloid apex curvatures are 1/a and 1/b "
          "(error %.1e) %s" % (err, "OK" if good else "FAIL"))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("focal surface self-test failed")
