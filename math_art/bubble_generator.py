
# Bubble Cluster generator for Blender: a soap-bubble cluster
# with one bubble at every point of a seed mesh (the Platonic
# solids, or any mesh's vertices).  Radii are either uniform or
# proportional to each point's mean distance to its neighbours.
#
# Intersections follow soap-film physics (Young-Laplace): where
# two bubbles meet, the film through their intersection circle
# is planar for equal radii, and for unequal radii is a sphere
# of curvature 1/r = 1/r_small - 1/r_large bulging into the
# larger (lower-pressure) bubble.  Each bubble's outer surface
# is its sphere trimmed to its own cell, and the interior films
# are trimmed where a third bubble's cell takes over, so triple
# junctions appear where three films meet, as in nature.
#
# References:
#   - Joseph Plateau, "Statique experimentale et theorique des
#     liquides soumis aux seules forces moleculaires" (1873) --
#     Plateau's laws for soap films (120-degree triple junctions).
#   - Young-Laplace law for the pressure jump across a curved film.
#   - Double Bubble theorem: M. Hutchings, F. Morgan, M. Ritore,
#     A. Ros, "Proof of the double bubble conjecture", Annals of
#     Mathematics 155 (2002), pp. 459-489.
#   - Triple Bubble theorem: E. Milman, J. Neeman, "The structure
#     of isoperimetric bubbles on R^n and S^n", arXiv:2205.09102
#     (2022) -- proof of the triple bubble conjecture in R^3 (the
#     standard triple bubble is the least-area partition).
#   - C. Isenberg, "The Science of Soap Films and Soap Bubbles"
#     (Dover, 1992); D. Weaire & S. Hutzler, "The Physics of Foams"
#     (Oxford, 1999).
#
# The RELAXED mode (single / double bubble) evolves a welded,
# region-pair-labeled seed mesh by volume-constrained area descent
# (`solver/volume.py`), so the equilibrium -- Plateau's 120-degree
# triple line included -- emerges from the minimization instead of
# being drawn in closed form.  Additional references for that mode:
#   - K. A. Brakke, "The Surface Evolver", Experimental Mathematics
#     1(2) (1992) -- the volume-constraint scheme.
#   - J. E. Taylor, "The structure of singularities in soap-bubble-like
#     and soap-film-like minimal surfaces", Annals of Mathematics
#     103(3) (1976) -- proof of Plateau's laws.

bl_info = {
    "name": "Bubble Cluster",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Odds & Ends",
    "description": "Soap-bubble clusters on the points of a "
                   "seed mesh, with physical film interfaces",
    "category": "Add Mesh",
}

import colorsys
import math
from math import sqrt

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


try:
    from .surfaces.primitives import icosphere as _icosphere_shared
except ImportError:  # flat import outside the package
    from surfaces.primitives import icosphere as _icosphere_shared

try:
    from .solver import volume as _svol
except ImportError:  # flat import outside the package
    from solver import volume as _svol


def _icosphere(subdiv=0):
    """Unit icosphere: the icosahedron subdivided `subdiv` times.
    Six modules built one of these; the shared one is in
    `surfaces.primitives`.  Same surface as before -- identical vertex
    set and identical triangles -- but the shared builder numbers the
    vertices in the standard order rather than this module's rotated
    one, so index order changes while the mesh does not.
    """
    return _icosphere_shared(subdiv, 'per_level')


def bubble_radii(P, edges, factor=0.62, uniform=True):
    """Radii for the bubbles at points P: `factor` x the mean
    distance to the neighbours (mesh edges, or the nearest point
    when there are none); `uniform` uses the global mean so all
    bubbles match."""
    P = np.asarray(P, float)
    n = len(P)
    loc = np.zeros(n)
    if edges:
        cnt = np.zeros(n)
        for a, b in edges:
            L = np.linalg.norm(P[a] - P[b])
            loc[a] += L
            loc[b] += L
            cnt[a] += 1
            cnt[b] += 1
        have = cnt > 0
        loc[have] /= cnt[have]
        if not have.all():
            loc[~have] = loc[have].mean() if have.any() else 1.0
    else:
        for i in range(n):
            d = np.linalg.norm(P - P[i], axis=1)
            d[i] = np.inf
            loc[i] = d.min()
    if uniform:
        return np.full(n, factor * loc.mean())
    return factor * loc


def _interfaces(P, R):
    """Soap-film interface for every overlapping pair: planar
    for equal radii, else the Young-Laplace sphere of curvature
    1/r_small - 1/r_large through the intersection circle,
    bulging into the larger bubble.  Each entry carries sign
    factors so that side(pair, bubble) > 0 at that bubble's
    centre."""
    P = np.asarray(P, float)
    pairs = {}
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            dv = P[j] - P[i]
            d = np.linalg.norm(dv)
            if (d < 1e-12 or d >= R[i] + R[j]
                    or d <= abs(R[i] - R[j]) + 1e-12):
                continue          # separate, or one engulfed
            u = dv / d
            a = (d * d + R[i] ** 2 - R[j] ** 2) / (2 * d)
            q = P[i] + a * u
            rho = sqrt(max(R[i] ** 2 - a * a, 0.0))
            if abs(R[i] - R[j]) < 1e-9 * max(R[i], R[j]):
                pr = dict(kind='P', q=q, u=u, rho=rho)
            else:
                rf = R[i] * R[j] / abs(R[j] - R[i])
                t = sqrt(max(rf * rf - rho * rho, 0.0))
                w = u if R[i] < R[j] else -u   # into the larger
                pr = dict(kind='S', q=q, u=u, rho=rho,
                          cf=q - w * t, rf=rf, w=w)
            vi = _raw(pr, P[i][None, :])[0]
            vj = _raw(pr, P[j][None, :])[0]
            if vi * vj >= 0:
                continue          # degenerate: no clean sides
            pr['sign'] = {i: (1.0 if vi > 0 else -1.0),
                          j: (1.0 if vj > 0 else -1.0)}
            pairs[(i, j)] = pr
    return pairs


def _raw(pr, arr):
    if pr['kind'] == 'P':
        return (arr - pr['q']) @ pr['u']
    return np.linalg.norm(arr - pr['cf'], axis=1) - pr['rf']


def _side_min(prs, arr):
    """min over (interface, bubble-sign) of the signed side
    value; > 0 keeps the point in that bubble's cell."""
    if not prs:
        return np.full(len(arr), np.inf)
    return np.min([sgn * _raw(pr, arr) for pr, sgn in prs],
                  axis=0)


def _clip(V, tris, prs, project=None):
    """Trim a triangle mesh to the region side >= 0, splitting
    crossing triangles at the (bisected) zero contour."""
    V = [np.asarray(v, float) for v in V]
    gv = _side_min(prs, np.array(V))
    cache = {}

    def root(ia, ib):
        key = (ia, ib) if ia < ib else (ib, ia)
        if key in cache:
            return cache[key]
        a, b = V[ia].copy(), V[ib].copy()
        if gv[ia] < gv[ib]:
            a, b = b, a
        for _ in range(30):        # g(a) >= 0 > g(b)
            m = 0.5 * (a + b)
            if project is not None:
                m = project(m)     # keep iterates on-surface
            if _side_min(prs, m[None, :])[0] >= 0.0:
                a = m
            else:
                b = m
        p = 0.5 * (a + b)
        if project is not None:
            p = project(p)
        cache[key] = len(V)
        V.append(p)
        return cache[key]

    out = []
    for t in tris:
        inside = [gv[i] >= 0 for i in t]
        n_in = sum(inside)
        if n_in == 3:
            out.append(tuple(t))
        elif n_in == 1:
            k = inside.index(True)
            a, b, c = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
            out.append((a, root(a, b), root(a, c)))
        elif n_in == 2:
            k = inside.index(False)
            c, a, b = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
            rbc = root(b, c)
            rca = root(a, c)
            out.append((a, b, rbc))
            out.append((a, rbc, rca))
    return V, out


def _film_mesh(pr, h):
    """Cap of the interface surface bounded by the intersection
    circle: rows of rings from the rim (exactly on the circle)
    to the apex."""
    u = pr['u']
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(u, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    rho = pr['rho']
    nphi = max(12, int(math.ceil(2 * math.pi * rho / h)))
    if pr['kind'] == 'S':
        tb = math.acos(min(1.0, max(
            -1.0, np.dot(pr['q'] - pr['cf'], pr['w'])
            / pr['rf'])))
        arc = tb * pr['rf']
    else:
        arc = rho
    K = max(2, int(math.ceil(arc / h)))
    V = []
    for k in range(K):
        fr = 1.0 - k / K
        for p in range(nphi):
            phi = 2 * math.pi * p / nphi
            dirv = math.cos(phi) * e1 + math.sin(phi) * e2
            if pr['kind'] == 'P':
                V.append(pr['q'] + rho * fr * dirv)
            else:
                th = tb * fr
                V.append(pr['cf'] + pr['rf']
                         * (math.cos(th) * pr['w']
                            + math.sin(th) * dirv))
    apex = (pr['q'] if pr['kind'] == 'P'
            else pr['cf'] + pr['rf'] * pr['w'])
    V.append(apex)
    tris = []
    for k in range(K - 1):
        for p in range(nphi):
            p2 = (p + 1) % nphi
            a, b = k * nphi + p, k * nphi + p2
            c, d = (k + 1) * nphi + p, (k + 1) * nphi + p2
            tris.append((a, b, c))
            tris.append((b, d, c))
    last = (K - 1) * nphi
    ctr = len(V) - 1
    for p in range(nphi):
        tris.append((last + p, last + (p + 1) % nphi, ctr))
    return V, tris


def build_parts(P, R, subdiv=2, films=True):
    """(caps, filmparts, pairs): one (verts, tris) cap mesh per
    bubble, and one per overlapping pair (keyed (i, j)) when
    `films`."""
    P = np.asarray(P, float)
    R = np.asarray(R, float)
    pairs = _interfaces(P, R)
    SV, SF = _icosphere(subdiv)
    caps = []
    for i in range(len(P)):
        prs = [(pr, pr['sign'][i]) for key, pr in pairs.items()
               if i in key]
        ci, ri = P[i], R[i]
        V0 = [ci + ri * v for v in SV]
        if prs:
            def proj(p, ci=ci, ri=ri):
                d = p - ci
                return ci + ri * d / (np.linalg.norm(d) or 1.0)
            V1, F1 = _clip(V0, SF, prs, proj)
        else:
            V1, F1 = V0, SF
        caps.append((V1, F1))

    filmparts = {}
    if films and pairs:
        h = 1.0515 * R.min() / (2 ** subdiv)
        for (i, j), pr in pairs.items():
            V0, F0 = _film_mesh(pr, h)
            prs = ([(p2, p2['sign'][i]) for key, p2
                    in pairs.items() if i in key and p2 is not pr]
                   + [(p2, p2['sign'][j]) for key, p2
                      in pairs.items()
                      if j in key and p2 is not pr])
            if prs:
                if pr['kind'] == 'S':
                    def proj(p, pr=pr):
                        d = p - pr['cf']
                        return (pr['cf'] + pr['rf'] * d
                                / (np.linalg.norm(d) or 1.0))
                else:
                    proj = None
                V1, F1 = _clip(V0, F0, prs, proj)
            else:
                V1, F1 = V0, F0
            if F1:
                filmparts[(i, j)] = (V1, F1)
    return caps, filmparts, pairs


def build_bubbles(P, R, subdiv=2, films=True):
    """(verts, faces, ids): the merged bubble cluster.  ids:
    bubble index for outer caps, n + pair-index for films."""
    caps, filmparts, pairs = build_parts(P, R, subdiv, films)
    pi_of = {k: idx for idx, k in enumerate(pairs)}
    verts = []
    faces = []
    ids = []

    def emit(v, f, idx):
        base = len(verts)
        verts.extend(tuple(p) for p in v)
        for t in f:
            faces.append([base + i for i in t])
            ids.append(idx)

    for i, (V, F) in enumerate(caps):
        emit(V, F, i)
    for key, (V, F) in filmparts.items():
        emit(V, F, len(caps) + pi_of[key])
    return verts, faces, ids


def _ray_t_one(ci, v, ri, cons, lab):
    """Distance along the ray ci + t v to constraint `lab`
    (0 = the bubble's own sphere, else film lab-1); inf when
    the ray never leaves the cell through it."""
    if lab == 0:
        return ri
    pr, sgn = cons[lab - 1]
    if pr['kind'] == 'P':
        du = np.dot(v, pr['u'])
        if sgn * du >= -1e-15:
            return np.inf
        t = -np.dot(ci - pr['q'], pr['u']) / du
        return t if t > 0 else np.inf
    w = ci - pr['cf']
    b = np.dot(v, w)
    disc = b * b - (np.dot(w, w) - pr['rf'] ** 2)
    if sgn < 0:                    # inside the film ball: exit
        return -b + sqrt(max(disc, 0.0))
    if disc <= 0 or b >= 0:        # outside: may enter
        return np.inf
    t = -b - sqrt(disc)
    return t if t > 0 else np.inf


def _ray_t(ci, v, ri, cons):
    """(t, label) of the nearest boundary along the ray."""
    best, lab = ri, 0
    for k in range(len(cons)):
        t = _ray_t_one(ci, v, ri, cons, k + 1)
        if t < best:
            best, lab = t, k + 1
    return best, lab


def build_bubble_solid(ci, ri, cons, subdiv=2):
    """One bubble as a CLOSED mesh: the cell is star-shaped
    around the centre, so an icosphere is displaced radially to
    the nearest boundary (own sphere or a film), and triangles
    whose vertices lie on different surfaces are split at the
    crease so every facet sits on one surface.  Returns (verts,
    tris, labels): label 0 = spherical cap, k = film cons[k-1]."""
    ci = np.asarray(ci, float)
    SV, SF = _icosphere(subdiv)
    tl = [_ray_t(ci, v, ri, cons) for v in SV]
    verts = [ci + t * v for (t, _), v in zip(tl, SV)]
    labs = [l for _, l in tl]
    dirs = list(SV)
    cache = {}

    def crossing(a, b):
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        # walk the direction arc to where vertex a's surface
        # stops being the nearest -- lands exactly on a crease
        # even when a third surface intrudes between a and b
        la = labs[a]
        p, q = dirs[a].copy(), dirs[b].copy()
        for _ in range(30):
            m = p + q
            m /= np.linalg.norm(m)
            if _ray_t(ci, m, ri, cons)[1] == la:
                p = m
            else:
                q = m
        v = p + q
        v /= np.linalg.norm(v)
        t, _ = _ray_t(ci, v, ri, cons)
        cache[key] = len(verts)
        verts.append(ci + t * v)
        dirs.append(v)
        return cache[key]

    tris = []
    tlabs = []
    for a, b, c in SF:
        la, lb, lc = labs[a], labs[b], labs[c]
        if la == lb == lc:
            tris.append((a, b, c))
            tlabs.append(la)
            continue
        if la != lb and lb != lc and lc != la:
            pab, pbc, pca = (crossing(a, b), crossing(b, c),
                             crossing(c, a))
            vj = dirs[pab] + dirs[pbc] + dirs[pca]
            vj /= np.linalg.norm(vj)
            tj, _ = _ray_t(ci, vj, ri, cons)
            J = len(verts)
            verts.append(ci + tj * vj)
            dirs.append(vj)
            for t3, l3 in (((a, pab, J), la), ((pab, b, J), lb),
                           ((b, pbc, J), lb), ((pbc, c, J), lc),
                           ((c, pca, J), lc), ((pca, a, J), la)):
                tris.append(t3)
                tlabs.append(l3)
            continue
        # exactly two labels: rotate the lone vertex first
        if lb != la and lb != lc:
            a, b, c = b, c, a
        elif lc != la and lc != lb:
            a, b, c = c, a, b
        la, lb = labs[a], labs[b]
        pab, pca = crossing(a, b), crossing(c, a)
        tris.append((a, pab, pca))
        tlabs.append(la)
        tris.append((pab, b, c))
        tlabs.append(lb)
        tris.append((pab, c, pca))
        tlabs.append(lb)
    # repair pass: a crease can cross a triangle without
    # changing its vertex labels, leaving a flap assigned to the
    # wrong surface.  Faces whose centroid ray lands on a
    # different surface are split at the centroid (conforming --
    # no edges touched) and the children relabelled by their own
    # centroid rays, iterated so flaps shrink geometrically.
    def _cen_label(t):
        cen = (np.asarray(verts[t[0]]) + verts[t[1]]
               + verts[t[2]]) / 3.0
        v = cen - ci
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return None
        v /= nv
        t_hit, lab = _ray_t(ci, v, ri, cons)
        return t_hit, lab, v

    for _ in range(3):
        changed = False
        nt, nl = [], []
        for t, l in zip(tris, tlabs):
            hit = _cen_label(t)
            if hit is None or hit[1] == l:
                nt.append(t)
                nl.append(l)
                continue
            changed = True
            t_hit, lab, v = hit
            J = len(verts)
            verts.append(ci + t_hit * v)
            for sub in ((t[0], t[1], J), (t[1], t[2], J),
                        (t[2], t[0], J)):
                h2 = _cen_label(sub)
                nt.append(sub)
                nl.append(h2[1] if h2 else l)
        tris, tlabs = nt, nl
        if not changed:
            break

    # snap every vertex exactly onto all the surfaces its
    # facets lie on (alternating projection): plain verts onto
    # their surface, crease verts onto both -- the rim curve --
    # and junction verts onto all three, so each facet is
    # exactly planar / spherical up to its edges
    vlabs = [set() for _ in verts]
    for t, l in zip(tris, tlabs):
        for k in t:
            vlabs[k].add(l)

    def _proj(p, l):
        if l == 0:
            d = p - ci
            return ci + ri * d / (np.linalg.norm(d) or 1.0)
        pr, _ = cons[l - 1]
        if pr['kind'] == 'P':
            return p - np.dot(p - pr['q'], pr['u']) * pr['u']
        d = p - pr['cf']
        return pr['cf'] + pr['rf'] * d / (np.linalg.norm(d)
                                          or 1.0)
    for k, ls in enumerate(vlabs):
        if not ls:
            continue
        p = np.asarray(verts[k], float)
        for _ in range(1 if len(ls) == 1 else 30):
            for l in sorted(ls):
                p = _proj(p, l)
        verts[k] = p

    # orient every facet outward (valid because the solid is
    # star-shaped around the centre) so crease slivers cannot
    # end up with inverted normals
    A = np.array(verts)
    for k, t in enumerate(tris):
        nrm = np.cross(A[t[1]] - A[t[0]], A[t[2]] - A[t[0]])
        cen = (A[t[0]] + A[t[1]] + A[t[2]]) / 3.0 - ci
        if np.dot(nrm, cen) < 0:
            tris[k] = (t[0], t[2], t[1])
    return ([tuple(p) for p in verts], tris, tlabs)


def film_cell(P, R, i, pairs=None):
    """The polyhedron bounded by bubble i's film planes (each
    film's intersection-circle plane): (verts, faces), or None
    when bubble i is not fully enclosed by films.  Only interior
    bubbles -- surrounded on all sides by neighbours -- close
    up; their cells are the flat-walled foam cells."""
    P = np.asarray(P, float)
    R = np.asarray(R, float)
    if pairs is None:
        pairs = _interfaces(P, R)
    C = P[i]
    planes = []
    for (a, b), pr in pairs.items():
        if i not in (a, b):
            continue
        nrm = np.array(pr['u'])    # points a -> b
        if i == b:
            nrm = -nrm             # outward from bubble i
        planes.append((np.array(pr['q']), nrm))
    if len(planes) < 4 or len(planes) > 60:
        return None
    big = 10.0 * (np.linalg.norm(P - C, axis=1).max() + R.max())
    tol = 1e-7 * big
    pts = []
    m = len(planes)
    for a in range(m):
        for b in range(a + 1, m):
            for c in range(b + 1, m):
                A = np.array([planes[a][1], planes[b][1],
                              planes[c][1]])
                if abs(np.linalg.det(A)) < 1e-9:
                    continue
                rhs = [np.dot(pl[1], pl[0])
                       for pl in (planes[a], planes[b],
                                  planes[c])]
                x = np.linalg.solve(A, rhs)
                if np.linalg.norm(x - C) > big:
                    continue
                if all(np.dot(x - q, nn) <= tol
                       for q, nn in planes):
                    pts.append(x)
    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - u) < 10 * tol
                   for u in uniq):
            uniq.append(p)
    if len(uniq) < 4:
        return None
    faces = []
    for q, nrm in planes:
        on = [k for k, p in enumerate(uniq)
              if abs(np.dot(p - q, nrm)) < 10 * tol]
        if len(on) < 3:
            continue
        fc = np.mean([uniq[k] for k in on], axis=0)
        e1 = uniq[on[0]] - fc
        e1 -= np.dot(e1, nrm) * nrm
        e1 /= np.linalg.norm(e1) or 1.0
        e2 = np.cross(nrm, e1)
        on.sort(key=lambda k: math.atan2(
            np.dot(uniq[k] - fc, e2),
            np.dot(uniq[k] - fc, e1)))
        faces.append(on)
    cnt = {}
    for f in faces:
        for k in range(len(f)):
            e = frozenset((f[k], f[(k + 1) % len(f)]))
            cnt[e] = cnt.get(e, 0) + 1
    if not cnt or any(v != 2 for v in cnt.values()):
        return None                # open: films don't close up
    return [tuple(p) for p in uniq], faces


def _seed(name):
    try:
        from . import spiked_polyhedron_generator as sp
    except ImportError:
        import spiked_polyhedron_generator as sp
    V, F = sp._seed(name)
    edges = set()
    for f in F:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            edges.add((a, b) if a < b else (b, a))
    return [tuple(v) for v in V], sorted(edges)


# --------------------------------------------------------------------
# Relaxed (evolved) clusters: analytic geometry, welded labeled seed
# meshes, and the volume-constrained relaxation.
# --------------------------------------------------------------------

def _cap_volume(R, h):
    """Volume of a spherical cap of height h on a sphere of radius R."""
    return math.pi * h * h * (3.0 * R - h) / 3.0


def double_bubble_geometry(r1, r2, d=None):
    """Exact geometry of the two-bubble cluster with sphere radii r1,
    r2 at centre separation d (default: the 120-degree equilibrium of
    the standard double bubble, d = sqrt(r1^2 + r2^2 - r1 r2)).

    The interface through the intersection circle is the Young-Laplace
    sphere 1/r3 = 1/r_small - 1/r_large (a plane when r1 == r2),
    bulging into the larger bubble.  Returns a dict:
      d, a (plane offset from centre 1 along the axis), rho (rim
      circle radius), r3 (math.inf when flat), h3 (interface bulge
      height), V1, V2 (exact lobe volumes), A1, A2, A3, A (film areas).
    """
    r1 = float(r1)
    r2 = float(r2)
    if d is None:
        d = math.sqrt(r1 * r1 + r2 * r2 - r1 * r2)
    d = float(d)
    if not (abs(r1 - r2) < d < r1 + r2):
        raise ValueError("spheres do not overlap at this separation")
    a = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    rho = math.sqrt(max(r1 * r1 - a * a, 0.0))
    # sphere portions on their own side of the rim plane
    V1 = 4.0 * math.pi * r1 ** 3 / 3.0 - _cap_volume(r1, r1 - a)
    V2 = 4.0 * math.pi * r2 ** 3 / 3.0 - _cap_volume(r2, r2 - (d - a))
    A1 = 2.0 * math.pi * r1 * (r1 + a)
    A2 = 2.0 * math.pi * r2 * (r2 + (d - a))
    if abs(r1 - r2) < 1e-12 * max(r1, r2):
        r3 = math.inf
        h3 = 0.0
        A3 = math.pi * rho * rho
    else:
        r3 = r1 * r2 / abs(r2 - r1)
        if rho > r3:
            raise ValueError("Young-Laplace interface sphere smaller "
                             "than the rim circle")
        t = math.sqrt(r3 * r3 - rho * rho)
        h3 = r3 - t
        A3 = 2.0 * math.pi * r3 * h3
        v3 = _cap_volume(r3, h3)
        if r1 < r2:          # bulge into bubble 2: lobe 1 gains the cap
            V1 += v3
            V2 -= v3
        else:
            V1 -= v3
            V2 += v3
    return {"d": d, "a": a, "rho": rho, "r3": r3, "h3": h3,
            "V1": V1, "V2": V2, "A1": A1, "A2": A2, "A3": A3,
            "A": A1 + A2 + A3}


def build_single_bubble_mesh(radius=1.0, subdiv=3):
    """One closed bubble as a labeled mesh: (V, T, labels), region 1
    inside, ambient 0 outside, outward-wound icosphere."""
    SV, SF = _icosphere(subdiv)
    V = float(radius) * np.asarray(SV, float)
    T = np.asarray(SF, dtype=np.int64)
    labels = np.zeros((len(T), 2), dtype=np.int64)
    labels[:, 1] = 1
    return V, T, labels


def _lathe_patch(V, tris, rim_idx, pos, K, nphi):
    """Rings-to-apex patch: interior rings k = 1..K-1 of nphi points
    from pos(k, phi), closed by the apex pos(K, .); ring 0 is the given
    shared rim.  Appends to V/tris (consistent winding)."""
    rings = [list(rim_idx)]
    for k in range(1, K):
        base = len(V)
        for p in range(nphi):
            V.append(pos(k, 2.0 * math.pi * p / nphi))
        rings.append(list(range(base, base + nphi)))
    apex = len(V)
    V.append(pos(K, 0.0))
    t0 = len(tris)
    for k in range(K - 1):
        prev, nxt = rings[k], rings[k + 1]
        for p in range(nphi):
            p2 = (p + 1) % nphi
            tris.append((prev[p], prev[p2], nxt[p]))
            tris.append((prev[p2], nxt[p2], nxt[p]))
    last = rings[-1]
    for p in range(nphi):
        tris.append((last[p], last[(p + 1) % nphi], apex))
    return t0, len(tris)


def _orient_patch(V, tris, t0, t1, want):
    """Flip the (consistently wound) patch slice so its normals agree
    with the desired direction field `want(centroid) -> vec`."""
    A = np.asarray(V[tris[t0][0]], float)
    B = np.asarray(V[tris[t0][1]], float)
    C = np.asarray(V[tris[t0][2]], float)
    n = np.cross(B - A, C - A)
    if float(np.dot(n, want((A + B + C) / 3.0))) < 0.0:
        for k in range(t0, t1):
            a, b, c = tris[k]
            tris[k] = (a, c, b)


def triple_bubble_geometry(r=1.0):
    """Closed-form geometry of the STANDARD EQUAL triple bubble (the
    proven minimizer for three equal volumes -- Milman-Neeman 2022):
    three spheres of radius r with centers on an equilateral triangle
    of side d = r (each pair meets at 120 degrees exactly as in the
    equal double bubble), three PLANAR films in the pairwise bisector
    planes (equal pressures), the films meeting along the vertical
    axis segment between the two tetrahedral points
    X+- = (0, 0, +-sqrt(2/3) r), where four triple lines meet at
    arccos(-1/3).  The seed this describes satisfies Plateau's laws
    EXACTLY (the rim-arc tangents at X+- make arccos(-1/3) with the
    axis and each other; verified in closed form: cos = -1/3).

    Areas are closed form (Gauss-Bonnet for the doubly-cut spherical
    bigon); the cell volume is the z-integral of exact slice areas
    (disk-wedge intersection), evaluated by Gauss-Legendre quadrature
    on the analytic pieces -- deterministic to ~1e-14, and cross-
    checked against the mesh volume's O(h^2) convergence in the
    self-test.

    Returns dict: centers (3,3), z0, rho (rim circle radius), theta0
    (= arccos(1/3), the rim-arc half-angle parameter), A_cap, A_film,
    A (total), V_cell."""
    r = float(r)
    d = r
    rho_c = d / math.sqrt(3.0)
    centers = np.array([[rho_c * math.cos(2.0 * math.pi * i / 3.0),
                         rho_c * math.sin(2.0 * math.pi * i / 3.0),
                         0.0] for i in range(3)])
    z0 = d * math.sqrt(2.0 / 3.0)
    rho = math.sqrt(3.0) * d / 2.0
    theta0 = math.acos(1.0 / 3.0)
    # cap: sphere minus two zones (cut at distance d/2, height r-d/2
    # = r/2 -> zone area pi r^2 each) plus the doubly-cut bigon,
    # A_bigon = (2 pi - 4 theta0) r^2 by Gauss-Bonnet (the two rim
    # arcs have geodesic curvature cot(60)/r and length rho*2*theta0
    # -> integral theta0 each; exterior angles at X+- are theta0)
    A_cap = (4.0 * math.pi - 2.0 * math.pi
             + 2.0 * math.pi - 4.0 * theta0) * r * r
    # film: major segment of the rho-disk cut by the axis chord at
    # distance d/(2 sqrt 3) from its center (half-angle theta0 again)
    s0c0 = 2.0 * math.sqrt(2.0) / 9.0          # sin(theta0) cos(theta0)
    A_film = rho * rho * (math.pi - theta0 + s0c0)
    # cell volume: slices are disk(center rho_c, radius s(z)) cut to
    # the 120-degree wedge around its own center direction
    from numpy.polynomial.legendre import leggauss

    def slice_area(z):
        s2 = r * r - z * z
        if s2 <= 0.0:
            return 0.0
        s = math.sqrt(s2)

        def F(phi):                      # antider. of 2 rc cos sqrt()
            u = math.sin(phi)
            return (rho_c * u * math.sqrt(max(s2 - rho_c ** 2 * u * u,
                                              0.0))
                    + s2 * math.asin(max(-1.0, min(1.0,
                                                   rho_c * u / s))))

        if s >= rho_c:                   # origin inside the disk
            def G(phi):
                return (rho_c ** 2 / 4.0 * math.sin(2.0 * phi)
                        + s2 / 2.0 * phi + 0.5 * F(phi))
            return 2.0 * (G(math.pi / 3.0) - G(0.0))
        phim = min(math.pi / 3.0, math.asin(min(1.0, s / rho_c)))
        return 2.0 * (F(phim) - F(0.0))

    xs, ws = leggauss(64)
    V_cell = 0.0
    for a, b in ((0.0, z0), (z0, math.sqrt(3.0) / 2.0 * r),
                 (math.sqrt(3.0) / 2.0 * r, r)):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        V_cell += half * sum(w * slice_area(mid + half * x)
                             for x, w in zip(xs, ws))
    V_cell *= 2.0                        # z-symmetry
    return {"centers": centers, "z0": z0, "rho": rho,
            "theta0": theta0, "A_cap": A_cap, "A_film": A_film,
            "A": 3.0 * (A_cap + A_film), "V_cell": V_cell}


def _cone_patch(V, tris, loop_ids, ring_of, K):
    """Disk patch: rings k = 1..K-1 between the boundary loop (ring 0,
    existing shared vertices) and the apex (ring_of(1.0) collapses to
    one point), triangulated with consistent winding.  ring_of(t) ->
    (L, 3) points for t in (0, 1].  Returns the (t0, t1) slice of tris
    added (for orientation fixing)."""
    L = len(loop_ids)
    rings = [list(loop_ids)]
    for k in range(1, K):
        pts = ring_of(k / K)
        base = len(V)
        V.extend(np.asarray(p, float) for p in pts)
        rings.append(list(range(base, base + L)))
    apex = len(V)
    V.append(np.asarray(ring_of(1.0)[0], float))
    t0 = len(tris)
    for k in range(K - 1):
        prev, nxt = rings[k], rings[k + 1]
        for p in range(L):
            p2 = (p + 1) % L
            tris.append((prev[p], prev[p2], nxt[p]))
            tris.append((prev[p2], nxt[p2], nxt[p]))
    last = rings[-1]
    for p in range(L):
        tris.append((last[p], last[(p + 1) % L], apex))
    return t0, len(tris)


def build_triple_bubble_mesh(r=1.0, nphi=48):
    """Welded labeled triangle mesh of the standard equal triple
    bubble: three outer spherical caps, three planar films, all six
    patches sharing the rim-arc / axis-segment vertex pools, with two
    tetrahedral vertices X+- where four triple lines (the axis and the
    three rim arcs) meet -- the combinatorics Plateau's second law is
    about.  Every patch is a disk meshed by rings shrinking from its
    boundary loop toward an interior apex (linear for the planar
    films, spherical slerp toward the far point for the caps).

    Labels: caps (0, i+1); film between bubbles i < j is (j+1, i+1)
    with its normal along c_j - c_i, matching the double bubble's
    convention.  Returns (V, T, labels)."""
    geo = triple_bubble_geometry(r)
    C = geo["centers"]
    z0, rho, theta0 = geo["z0"], geo["rho"], geo["theta0"]
    h = 2.0 * math.pi * rho / nphi
    M_c = max(3, int(round(2.0 * z0 / h)))
    M_a = max(6, int(round(rho * (2.0 * math.pi - 2.0 * theta0) / h)))

    V = [np.array([0.0, 0.0, -z0]), np.array([0.0, 0.0, z0])]
    seg = []
    for k in range(1, M_c):
        seg.append(len(V))
        V.append(np.array([0.0, 0.0, -z0 + 2.0 * z0 * k / M_c]))

    def arc_point(i, j, t):
        """Outer rim arc of pair (i, j), t in [0,1]: t=0 -> X+,
        t=1 -> X-."""
        m = 0.5 * (C[i] + C[j])
        e1 = -m / np.linalg.norm(m)
        e2 = np.array([0.0, 0.0, 1.0])
        phi = theta0 + t * (2.0 * math.pi - 2.0 * theta0)
        return m + rho * (math.cos(phi) * e1 + math.sin(phi) * e2)

    arcs = {}
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        ids = []
        for k in range(1, M_a):
            ids.append(len(V))
            V.append(arc_point(i, j, k / M_a))
        arcs[(i, j)] = ids

    tris = []
    lab = []

    def add_film(i, j):
        loop = [0] + seg + [1] + arcs[(i, j)]
        P0 = np.array([V[v] for v in loop])
        m = 0.5 * (C[i] + C[j])
        e1 = -m / np.linalg.norm(m)
        apex = m + 0.5 * (np.linalg.norm(m) - rho) * e1
        K = max(2, int(round(float(np.mean(np.linalg.norm(
            P0 - apex, axis=1))) / h)))
        t0, t1 = _cone_patch(V, tris, loop,
                             lambda t: (1.0 - t) * P0 + t * apex, K)
        _orient_patch(V, tris, t0, t1, lambda cen: C[j] - C[i])
        lab.extend([(j + 1, i + 1)] * (t1 - t0))

    def add_cap(i, j, k):
        aij = arcs[(min(i, j), max(i, j))]
        aik = arcs[(min(i, k), max(i, k))]
        loop = [1] + aij + [0] + aik[::-1]
        P0 = np.array([V[v] for v in loop])
        far = C[i] + float(r) * C[i] / np.linalg.norm(C[i])
        a = (P0 - C[i]) / float(r)
        b = (far - C[i]) / float(r)
        dots = np.clip(a @ b, -1.0, 1.0)
        Om = np.arccos(dots)
        sO = np.maximum(np.sin(Om), 1e-12)

        def ring_of(t):
            return C[i] + float(r) * (
                np.sin((1.0 - t) * Om)[:, None] * a
                + np.sin(t * Om)[:, None] * b[None, :]) / sO[:, None]

        K = max(3, int(round(float(np.mean(Om)) * float(r) / h)))
        t0, t1 = _cone_patch(V, tris, loop, ring_of, K)
        _orient_patch(V, tris, t0, t1, lambda cen: cen - C[i])
        lab.extend([(0, i + 1)] * (t1 - t0))

    add_film(0, 1)
    add_film(0, 2)
    add_film(1, 2)
    add_cap(0, 1, 2)
    add_cap(1, 0, 2)
    add_cap(2, 0, 1)
    return (np.asarray(V, float), np.asarray(tris, dtype=np.int64),
            np.asarray(lab, dtype=np.int64))


def build_double_bubble_mesh(r1=1.0, r2=1.0, nphi=48, d=None):
    """Welded labeled triangle mesh of the two-bubble cluster: the two
    outer spherical caps and the Young-Laplace interface share the
    nphi rim vertices on the intersection circle, so the Plateau border
    is a genuine non-manifold triple line.  Centre 1 at the origin,
    centre 2 at (d, 0, 0); default d is the 120-degree equilibrium.

    Returns (V, T, labels): labels per face are (front, back) region
    pairs -- caps (0, 1) and (0, 2), interface (2, 1)."""
    geo = double_bubble_geometry(r1, r2, d)
    d, a, rho, r3 = geo["d"], geo["a"], geo["rho"], geo["r3"]
    u = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    c1 = np.zeros(3)
    c2 = d * u
    q = a * u                              # rim-circle centre
    h = 2.0 * math.pi * rho / nphi         # target edge length

    V = []
    for p in range(nphi):
        phi = 2.0 * math.pi * p / nphi
        V.append(q + rho * (math.cos(phi) * e1 + math.sin(phi) * e2))
    rim = list(range(nphi))
    tris = []
    lab = []

    def cap_pos(center, R, axis, theta_rim):
        def pos(k, phi, K):
            th = theta_rim * (1.0 - k / K)
            dirv = math.cos(phi) * e1 + math.sin(phi) * e2
            return center + R * (math.cos(th) * axis + math.sin(th) * dirv)
        return pos

    def add_patch(pos_of, K, want, front, back):
        t0, t1 = _lathe_patch(V, tris,
                              rim, lambda k, phi: pos_of(k, phi, K),
                              K, nphi)
        _orient_patch(V, tris, t0, t1, want)
        lab.extend([(front, back)] * (t1 - t0))

    # outer cap of bubble 1 (major cap, axis away from bubble 2)
    th1 = math.acos(max(-1.0, min(1.0, -a / r1)))
    K1 = max(2, int(round(th1 * r1 / h)))
    add_patch(cap_pos(c1, r1, -u, th1), K1,
              lambda cen: cen - c1, 0, 1)
    # outer cap of bubble 2
    th2 = math.acos(max(-1.0, min(1.0, (a - d) / r2)))
    K2 = max(2, int(round(th2 * r2 / h)))
    add_patch(cap_pos(c2, r2, u, th2), K2,
              lambda cen: cen - c2, 0, 2)
    # interface: plane when equal, Young-Laplace cap into the larger
    if math.isinf(r3):
        K3 = max(1, int(round(rho / h)))

        def flat_pos(k, phi, K):
            dirv = math.cos(phi) * e1 + math.sin(phi) * e2
            return q + rho * (1.0 - k / K) * dirv

        add_patch(flat_pos, K3, lambda cen: u, 2, 1)
    else:
        w = u if r1 < r2 else -u           # bulge into the larger bubble
        sgn = 1.0 if r1 < r2 else -1.0     # normal must point into body 2
        t = math.sqrt(r3 * r3 - rho * rho)
        cf = q - t * w
        th3 = math.acos(max(-1.0, min(1.0, t / r3)))
        K3 = max(1, int(round(th3 * r3 / h)))
        add_patch(cap_pos(cf, r3, w, th3), K3,
                  lambda cen: sgn * (cen - cf), 2, 1)

    return (np.asarray(V, float), np.asarray(tris, dtype=np.int64),
            np.asarray(lab, dtype=np.int64))


def relax_cluster(V, T, labels, targets=None, iters=150, groom_every=0):
    """Volume-constrained area relaxation of a labeled cluster mesh
    (in place); returns the solver.volume.evolve info dict (final
    area / volumes / pressures / per-iteration history)."""
    return _svol.evolve(V, T, labels, targets=targets, iters=iters,
                        groom_every=groom_every)


if _IN_BLENDER:

    def _mat(name, col):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = next(nd for nd in mat.node_tree.nodes
                    if nd.type == 'BSDF_PRINCIPLED')
        bsdf.inputs["Base Color"].default_value = col
        mat.diffuse_color = col
        return mat

    def _color_mat(i):
        hue = (i * 0.61803398875) % 1.0
        return _mat(f"Bubble {i + 1:03d}",
                    colorsys.hsv_to_rgb(hue, 0.55, 0.9)
                    + (1.0,))

    def _plain_mat():
        return _mat("Bubble Films", (0.8, 0.8, 0.8, 1.0))

    class MESH_OT_bubble_cluster_add(bpy.types.Operator):
        """Cluster of soap bubbles at the points of a seed mesh,
        with physically correct film interfaces (planar between
        equal bubbles, curved into the larger one otherwise)"""
        bl_idname = "mesh.bubble_cluster_add"
        bl_label = "Bubble Cluster"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Seed",
            items=[('TETRA', "Tetrahedron", ""),
                   ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", ""),
                   ('ACTIVE', "Active Object",
                    "One bubble per vertex of the active mesh")],
            default='ICOSA')
        radius_mode: EnumProperty(
            name="Radii",
            items=[('UNIFORM', "Same Radius",
                    "All bubbles get the same radius (from the "
                    "global mean neighbour distance)"),
                   ('LOCAL', "From Neighbour Distance",
                    "Each bubble's radius follows its own mean "
                    "distance to its neighbours")],
            default='UNIFORM')
        factor: FloatProperty(
            name="Radius Factor", default=0.62, min=0.1, max=2.0,
            description="Bubble radius as a fraction of the "
                        "mean neighbour distance (above 0.5 "
                        "neighbouring bubbles merge)")
        subdiv: IntProperty(
            name="Subdivisions", default=2, min=1, max=5,
            description="Icosphere subdivisions per bubble")
        films: BoolProperty(
            name="Interior Films", default=True,
            description="Include the soap films between "
                        "touching bubbles (merged mesh only; "
                        "separate bubbles always carry their "
                        "walls, keeping each one closed)")
        separate: BoolProperty(
            name="Separate Bubble Meshes", default=True,
            description="One mesh object per bubble (its outer "
                        "cap plus its films), parented to an "
                        "empty")
        color: BoolProperty(
            name="Color Bubbles", default=False,
            description="Give each bubble its own material "
                        "color")
        cell: BoolProperty(
            name="Intersection Polyhedra", default=False,
            description="For every bubble fully enclosed by "
                        "films, add its flat-walled cell "
                        "polyhedron as a separate mesh (needs "
                        "interior points, e.g. a lattice seed "
                        "via Active Object)")
        smooth: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.seed == 'ACTIVE':
                src = context.active_object
                if src is None or src.type != 'MESH':
                    self.report({'ERROR'},
                                "no active mesh object; pick a "
                                "built-in seed instead")
                    return {'CANCELLED'}
                deps = context.evaluated_depsgraph_get()
                me0 = src.evaluated_get(deps).to_mesh()
                P = [tuple(v.co) for v in me0.vertices]
                edges = [tuple(e.vertices) for e in me0.edges]
                src.evaluated_get(deps).to_mesh_clear()
                name = f"{src.name} Bubbles"
            else:
                P, edges = _seed(self.seed)
                name = f"Bubbles ({self.seed.title()})"
            if len(P) < 1:
                self.report({'ERROR'}, "seed has no points")
                return {'CANCELLED'}
            R = bubble_radii(P, edges, self.factor,
                             self.radius_mode == 'UNIFORM')
            n = len(P)
            Pa = np.asarray(P, float)
            pairs = _interfaces(Pa, np.asarray(R, float))
            pi_of = {k: idx for idx, k in enumerate(pairs)}
            # (name, verts, faces, ids, origin, bubble idx)
            parts = []
            if self.separate:
                # each bubble is one CLOSED solid: its sphere
                # radially capped by its films
                for i in range(n):
                    cons = []
                    consk = []
                    for key, pr in pairs.items():
                        if i in key:
                            cons.append((pr, pr['sign'][i]))
                            consk.append(key)
                    V, F, labs = build_bubble_solid(
                        Pa[i], R[i], cons, self.subdiv)
                    ids = [i if l == 0
                           else n + pi_of[consk[l - 1]]
                           for l in labs]
                    parts.append([f"Bubble {i + 1:03d}", V,
                                  [list(f) for f in F], ids,
                                  Pa[i], i])
            else:
                verts, faces, ids = build_bubbles(
                    P, R, self.subdiv, self.films)
                parts.append([name, verts, faces, ids,
                              None, None])
            if self.cell:
                found = 0
                for i in range(n):
                    cm = film_cell(P, R, i, pairs)
                    if cm is None:
                        continue
                    found += 1
                    parts.append([f"Cell {i + 1:03d}", cm[0],
                                  [list(f) for f in cm[1]],
                                  [-1] * len(cm[1]),
                                  Pa[i], None])
                if not found:
                    self.report({'WARNING'},
                                "no bubble is fully enclosed "
                                "by films, so they bound no "
                                "polyhedron (interior points "
                                "are needed)")
            # fit the underlying polyhedron -- the bubble centres
            # (the seed mesh's vertices) -- roughly within a
            # 2 x scale cube at the origin; the same centre and
            # scale go to every emitted mesh, so bubbles just bulge
            # beyond it by their radii, which is fine
            lo = [float(Pa[:, k].min()) for k in range(3)]
            hi = [float(Pa[:, k].max()) for k in range(3)]
            ctr = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            s = self.scale / half
            cur = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            films_mat = None
            objs = []
            for pname, V, F, ids, origin, bi in parts:
                off = ([0.0, 0.0, 0.0] if origin is None
                       else [(origin[k] - ctr[k]) * s
                             for k in range(3)])
                Vt = [tuple((v[k] - ctr[k]) * s - off[k]
                            for k in range(3)) for v in V]
                me = bpy.data.meshes.new(pname)
                me.from_pydata(Vt, [], F)
                me.validate(clean_customdata=True)
                sm = self.smooth and not pname.startswith("Cell")
                me.polygons.foreach_set(
                    'use_smooth', [sm] * len(me.polygons))
                # analytic normals: radial on the caps, the
                # film's own normal on the films -- perfectly
                # smooth surfaces, perfectly sharp creases
                if sm and len(me.polygons) == len(ids) \
                        and min(ids, default=-1) >= 0:
                    keys_l = list(pairs)
                    loops = []
                    for poly, d in zip(me.polygons, ids):
                        for vi in poly.vertices:
                            p = np.asarray(V[vi], float)
                            if d < n:
                                nn = p - Pa[d]
                            else:
                                key = keys_l[d - n]
                                pr = pairs[key]
                                sb = (bi if bi is not None
                                      else key[0])
                                sg = -pr['sign'][sb]
                                if pr['kind'] == 'P':
                                    nn = sg * pr['u']
                                else:
                                    nn = sg * (p - pr['cf'])
                            ln = np.linalg.norm(nn)
                            nn = (nn / ln if ln > 1e-12
                                  else np.array((0.0, 0.0, 1.0)))
                            loops.append(tuple(nn))
                    me.normals_split_custom_set(loops)
                attr = me.attributes.new("bubble_index", 'INT',
                                         'FACE')
                if len(me.polygons) == len(ids):
                    attr.data.foreach_set('value', ids)
                    # crease edges (cap-film and film-film
                    # boundaries) stay sharp under smooth
                    # shading: mark edges whose two faces lie
                    # on different surfaces
                    eids = {}
                    for poly, d in zip(me.polygons, ids):
                        for ek in poly.edge_keys:
                            eids.setdefault(ek, set()).add(d)
                    me.edges.foreach_set(
                        'use_edge_sharp',
                        [len(eids.get(e.key, ())) > 1
                         for e in me.edges])
                if self.color:
                    if bi is not None:
                        me.materials.append(_color_mat(bi))
                    elif pname == "Intersection Cell":
                        me.materials.append(_plain_mat())
                    else:
                        for i in range(n):
                            me.materials.append(_color_mat(i))
                        me.materials.append(_plain_mat())
                        if len(me.polygons) == len(ids):
                            me.polygons.foreach_set(
                                'material_index',
                                [min(d, n) for d in ids])
                me.update()
                obj = bpy.data.objects.new(pname, me)
                context.collection.objects.link(obj)
                obj.location = off
                obj.select_set(True)
                objs.append(obj)
            if len(objs) > 1:
                root = bpy.data.objects.new(name, None)
                root.empty_display_size = 0.2
                context.collection.objects.link(root)
                root.location = cur
                for obj in objs:
                    obj.parent = root
                root.select_set(True)
            else:
                objs[0].location = cur
            context.view_layer.objects.active = objs[0]
            self.report({'INFO'},
                        f"{name}: {n} bubbles, "
                        f"{len(pairs)} films, "
                        f"{len(objs)} object(s)")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('seed', 'radius_mode', 'factor', 'subdiv',
                      'films', 'separate', 'color', 'cell',
                      'smooth', 'scale'):
                lay.prop(self, k)

    class MESH_OT_relaxed_bubble_add(bpy.types.Operator):
        """Single bubble or double bubble relaxed by genuine
        volume-constrained area minimization: Plateau's 120-degree
        triple line and the Young-Laplace pressures emerge from the
        evolution rather than being drawn in closed form"""
        bl_idname = "mesh.relaxed_bubble_add"
        bl_label = "Relaxed Bubble (Evolved)"
        bl_options = {'REGISTER', 'UNDO'}

        bubbles: EnumProperty(
            name="Bubbles",
            items=[('SINGLE', "Single Bubble",
                    "One bubble relaxing to the round sphere"),
                   ('DOUBLE', "Double Bubble",
                    "Two bubbles with their Young-Laplace interface "
                    "and a genuine Plateau border"),
                   ('TRIPLE', "Triple Bubble",
                    "Three bubbles: three films meeting along a "
                    "central Plateau border that ends in two "
                    "tetrahedral points where four triple lines "
                    "meet at arccos(-1/3) ~ 109.47 degrees")],
            default='DOUBLE')
        ratio: FloatProperty(
            name="Volume Ratio", default=0.75, min=0.4, max=1.0,
            description="Double bubble: small/large radius ratio "
                        "(1 = symmetric, flat interface).  Triple "
                        "bubble: volume grading V1:V2:V3 = "
                        "ratio : 1 : (2 - ratio); 1 = equal cells "
                        "with planar films")
        squash: FloatProperty(
            name="Seed Squash", default=0.15, min=0.0, max=0.4,
            description="Anisotropic distortion of the seed before "
                        "relaxing -- the evolution visibly pulls it "
                        "back to the equilibrium (0 starts at the "
                        "closed form)")
        resolution: IntProperty(
            name="Resolution", default=48, min=16, max=128,
            description="Vertices around the Plateau border (double "
                        "bubble) / icosphere resolution (single)")
        iterations: IntProperty(
            name="Evolve Iterations", default=150, min=0, max=2000,
            description="Volume-constrained area-descent iterations "
                        "(0 shows the raw seed)")
        groom_every: IntProperty(
            name="Groom Every", default=4, min=0, max=50,
            description="Run label-aware mesh grooming (edge flips + "
                        "tangential smoothing) every N iterations "
                        "(0 = off; the default 4 measurably tightens "
                        "the Plateau angles at equal cost)")
        color: BoolProperty(
            name="Color by Pressure", default=True,
            description="One material per film, colored by the "
                        "computed pressure jump across it")
        smooth: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.bubbles == 'SINGLE':
                subdiv = max(2, min(5, int(round(
                    math.log2(max(self.resolution, 16) / 3.0)))))
                V, T, labels = build_single_bubble_mesh(1.0, subdiv)
                targets = _svol.region_volumes(V, T, labels)
            elif self.bubbles == 'TRIPLE':
                V, T, labels = build_triple_bubble_mesh(
                    1.0, nphi=self.resolution)
                geo = triple_bubble_geometry(1.0)
                targets = [geo["V_cell"] * self.ratio,
                           geo["V_cell"],
                           geo["V_cell"] * (2.0 - self.ratio)]
            else:
                r1, r2 = self.ratio, 1.0
                V, T, labels = build_double_bubble_mesh(
                    r1, r2, nphi=self.resolution)
                geo = double_bubble_geometry(r1, r2)
                targets = [geo["V1"], geo["V2"]]
            if self.squash > 0.0:
                V *= np.array([1.0 + self.squash,
                               1.0 - 0.5 * self.squash, 1.0])
            info = relax_cluster(V, T, labels, targets=targets,
                                 iters=self.iterations,
                                 groom_every=self.groom_every)
            press = list(info["pressures"])

            # centre and fit within a 2*scale cube
            lo = V.min(axis=0)
            hi = V.max(axis=0)
            ctr = 0.5 * (lo + hi)
            half = float(np.max(hi - lo)) / 2.0 or 1.0
            s = self.scale / half
            Vt = (V - ctr) * s

            name = {"SINGLE": "Relaxed Bubble",
                    "DOUBLE": "Relaxed Double Bubble",
                    "TRIPLE": "Relaxed Triple Bubble"}[self.bubbles]
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(p) for p in Vt], [],
                           [list(t) for t in T])
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [self.smooth] * len(me.polygons))
            npoly = len(me.polygons)
            if npoly == len(T):
                a_front = me.attributes.new("region_front", 'INT', 'FACE')
                a_front.data.foreach_set('value',
                                         [int(x) for x in labels[:, 0]])
                a_back = me.attributes.new("region_back", 'INT', 'FACE')
                a_back.data.foreach_set('value',
                                        [int(x) for x in labels[:, 1]])
                pj = me.attributes.new("pressure_jump", 'FLOAT', 'FACE')

                def _p(r):
                    return press[r - 1] if r >= 1 else 0.0

                pj.data.foreach_set('value',
                                    [_p(int(b)) - _p(int(f))
                                     for f, b in labels])
                if self.color:
                    films = sorted({(int(f), int(b))
                                    for f, b in labels})
                    fidx = {fb: i for i, fb in enumerate(films)}
                    pmax = max(abs(_p(b) - _p(f))
                               for f, b in films) or 1.0
                    for f, b in films:
                        dp = _p(b) - _p(f)
                        hue = 0.62 - 0.5 * dp / pmax   # blue -> warm
                        me.materials.append(_mat(
                            f"Film {f}|{b} dp={dp:.3f}",
                            colorsys.hsv_to_rgb(hue % 1.0, 0.55, 0.9)
                            + (1.0,)))
                    me.polygons.foreach_set(
                        'material_index',
                        [fidx[(int(f), int(b))] for f, b in labels])
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            ptxt = ", ".join(f"p{i + 1}={p:.3f}"
                             for i, p in enumerate(press))
            self.report({'INFO'},
                        f"{name}: {info['iters_run']} iterations, "
                        f"area {info['area']:.4f}, {ptxt}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'bubbles')
            if self.bubbles in ('DOUBLE', 'TRIPLE'):
                lay.prop(self, 'ratio')
            for k in ('squash', 'resolution', 'iterations',
                      'groom_every', 'color', 'smooth', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.bubble_cluster_add",
                             icon='SPHERE')
        self.layout.operator("mesh.relaxed_bubble_add",
                             icon='META_BALL')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_bubble_cluster_add)
        bpy.utils.register_class(MESH_OT_relaxed_bubble_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_relaxed_bubble_add)
        bpy.utils.unregister_class(MESH_OT_bubble_cluster_add)


def _selftest():
    # two equal bubbles: the film must be planar
    P = np.array([(0.0, 0, 0), (1.4, 0, 0)])
    R = np.array([1.0, 1.0])
    pairs = _interfaces(P, R)
    assert list(pairs) == [(0, 1)]
    pr = pairs[(0, 1)]
    assert pr['kind'] == 'P'
    verts, faces, ids = build_bubbles(P, R, subdiv=2)
    film = [np.array(verts[i]) for f, d in zip(faces, ids)
            if d == 2 for i in f]
    assert film, "no film emitted"
    planar = max(abs(p[0] - pr['q'][0]) for p in film)
    print(f"equal pair: film planarity residual "
          f"{planar:.2e}")
    assert planar < 1e-9
    # caps stay in their own cell
    for f, d in zip(faces, ids):
        if d == 0:
            for i in f:
                assert verts[i][0] < pr['q'][0] + 1e-6
    # unequal bubbles: Young-Laplace sphere into the larger
    P2 = np.array([(0.0, 0, 0), (1.5, 0, 0)])
    R2 = np.array([0.8, 1.2])
    pr2 = _interfaces(P2, R2)[(0, 1)]
    assert pr2['kind'] == 'S'
    rf_want = 0.8 * 1.2 / (1.2 - 0.8)
    assert abs(pr2['rf'] - rf_want) < 1e-12
    assert np.dot(pr2['w'], [1, 0, 0]) > 0  # into larger
    v2, f2, d2 = build_bubbles(P2, R2, subdiv=2)
    filmv = [np.array(v2[i]) for f, d in zip(f2, d2)
             if d == 2 for i in f]
    offr = max(abs(np.linalg.norm(p - pr2['cf'])
                   - pr2['rf']) for p in filmv)
    print(f"unequal pair: rf={pr2['rf']:.3f} "
          f"(want {rf_want:.3f}), film sphericity "
          f"residual {offr:.2e}")
    assert offr < 1e-9
    # cube cluster: 8 bubbles, 12 films, all finite
    Pc, Ec = ([np.array((x, y, z)) for x in (0, 1)
               for y in (0, 1) for z in (0, 1)], None)
    Pc = np.array(Pc)
    Rc = bubble_radii(Pc, [], factor=0.62)
    assert abs(Rc[0] - 0.62) < 1e-12   # NN distance is 1
    vc, fc, dc = build_bubbles(Pc, Rc, subdiv=2)
    nfilm = len(set(d for d in dc if d >= 8))
    finite = all(all(math.isfinite(c) for c in v)
                 for v in vc)
    print(f"cube cluster: V={len(vc)} F={len(fc)} "
          f"films={nfilm} finite={finite}")
    assert nfilm == 12 and finite
    # every cap vertex lies in its bubble's cell
    pairs_c = _interfaces(Pc, Rc)
    Vc = np.array(vc)
    for i in range(8):
        prs = [(pr, pr['sign'][i]) for key, pr
               in pairs_c.items() if i in key]
        idxs = sorted({k for f, d in zip(fc, dc)
                       if d == i for k in f})
        g = _side_min(prs, Vc[idxs])
        assert g.min() > -1e-6, g.min()
    # corner bubbles are not enclosed: no cell polyhedra
    assert all(film_cell(Pc, Rc, i) is None for i in range(8))
    # separate solids: every bubble mesh must be CLOSED
    # (each edge in exactly two triangles), finite, with
    # positive volume, and its crease vertices must sit on
    # both surfaces (sphere and film)
    pairs_k = list(pairs_c)
    for i in range(8):
        cons = [(pairs_c[k], pairs_c[k]['sign'][i])
                for k in pairs_k if i in k]
        sv, sf, sl = build_bubble_solid(Pc[i], Rc[i], cons,
                                        subdiv=2)
        ec = {}
        for f in sf:
            for k in range(3):
                e = frozenset((f[k], f[(k + 1) % 3]))
                ec[e] = ec.get(e, 0) + 1
        assert all(c == 2 for c in ec.values()), \
            f"bubble {i} not closed"
        A = np.array(sv)
        vol = sum(np.dot(A[f[0]],
                         np.cross(A[f[1]], A[f[2]]))
                  for f in sf) / 6.0
        assert vol > 0 and np.isfinite(A).all()
        if i == 0:
            onboth = 0
            for k, p in enumerate(A):
                rs = abs(np.linalg.norm(p - Pc[0]) - Rc[0])
                rq = min(abs(sgn * _raw(pr, p[None, :])[0])
                         for pr, sgn in cons)
                if rs < 1e-5 and rq < 1e-5:
                    onboth += 1
            assert onboth > 10, onboth   # crease verts exist
            print(f"bubble solid: V={len(sv)} F={len(sf)} "
                  f"closed, vol={vol:.4f}, "
                  f"{onboth} crease verts on both surfaces")
    # deep overlap (factor 0.8): every film-label facet of
    # the solid must be exactly planar -- all its vertices
    # on the film plane, including the crease rim
    Rd = bubble_radii(Pc, [], factor=0.8)
    pairs_d = _interfaces(Pc, Rd)
    cons = [(pairs_d[k], pairs_d[k]['sign'][0])
            for k in pairs_d if 0 in k]
    sv, sf, sl = build_bubble_solid(Pc[0], Rd[0], cons,
                                    subdiv=3)
    A = np.array(sv)
    worst = 0.0
    for f, l in zip(sf, sl):
        if l == 0:
            continue
        pr, sgn = cons[l - 1]
        for k in f:
            worst = max(worst,
                        abs(np.dot(A[k] - pr['q'],
                                   pr['u'])))
    ec = {}
    for f in sf:
        for k in range(3):
            e = frozenset((f[k], f[(k + 1) % 3]))
            ec[e] = ec.get(e, 0) + 1
    closed = all(c == 2 for c in ec.values())
    print(f"deep-overlap solid: facet planarity residual "
          f"{worst:.2e}, closed={closed}")
    assert worst < 1e-6 and closed
    # BCC: a centre bubble surrounded by the 8 corners is
    # enclosed -- its film planes (normal to the body
    # diagonals) bound an octahedron
    Pb = np.vstack([[[0.5, 0.5, 0.5]], Pc])
    Rb = np.full(9, 0.62)
    cellm = film_cell(Pb, Rb, 0)
    assert cellm is not None
    cv, cfs = cellm
    print(f"BCC centre cell: V={len(cv)} F={len(cfs)} "
          f"(want the octahedron: 6/8)")
    assert len(cv) == 6 and len(cfs) == 8
    assert film_cell(Pb, Rb, 1) is None
    # ---- relaxed mode: analytic geometry, welded seeds, evolution ----
    # equal standard double bubble closed forms: V = 9 pi r^3 / 8,
    # A = 27 pi r^2 / 4 (per lobe / total), d = r
    g = double_bubble_geometry(1.0, 1.0)
    assert abs(g["d"] - 1.0) < 1e-14
    assert abs(g["V1"] - 9 * math.pi / 8) < 1e-12
    assert abs(g["A"] - 27 * math.pi / 4) < 1e-12
    for r1, r2 in ((1.0, 1.0), (0.8, 1.2)):
        Vd, Td, Ld = build_double_bubble_mesh(r1, r2, nphi=48)
        geo = double_bubble_geometry(r1, r2)
        vols = _svol.region_volumes(Vd, Td, Ld)
        # inscribed discretization deficit is O(h^2) ~ 0.6% at nphi=48
        verr = max(abs(vols[0] - geo["V1"]) / geo["V1"],
                   abs(vols[1] - geo["V2"]) / geo["V2"])
        assert verr < 0.02, verr
        # every region's outward-oriented faces form a closed surface:
        # each directed edge appears exactly once with its reverse
        for reg in (1, 2):
            sgn = ((Ld[:, 1] == reg).astype(int)
                   - (Ld[:, 0] == reg))
            ecnt = {}
            for f, sf in zip(Td, sgn):
                if sf == 0:
                    continue
                tri = f if sf > 0 else f[::-1]
                for k in range(3):
                    e = (int(tri[k]), int(tri[(k + 1) % 3]))
                    ecnt[e] = ecnt.get(e, 0) + 1
            assert all(c == 1 and ecnt.get((b, a), 0) == 1
                       for (a, b), c in ecnt.items()), \
                f"region {reg} of ({r1},{r2}) not closed"
        ang = _svol.triple_line_angles(Vd, Td)
        assert len(ang) == 3 * 48
        # per-edge sector sums are 360 so the mean is trivially 120;
        # the informative number is the spread, which at nphi=48 is
        # dominated by the O(h) face-secant tilt (~5 deg)
        rms = float(np.sqrt(np.mean((ang - 120.0) ** 2)))
        assert rms < 8.0, rms
        print(f"welded double bubble ({r1},{r2}): vol err {verr:.2e}, "
              f"regions closed, seed angle rms(120) {rms:.2f} deg")
    # ---- triple bubble: analytic constants, welded seed, closure ----
    geo3 = triple_bubble_geometry(1.0)
    # the analytic seed is Plateau-exact: rim-arc tangents at X+ make
    # arccos(-1/3) with the axis and each other (closed form: the
    # tangent is (-2 sqrt2/3) v_ij + (1/3) z with v_ij.v_ik = -1/2)
    t12 = np.array([-2.0 * math.sqrt(2.0) / 3.0, 0.0, 1.0 / 3.0])
    assert abs(np.dot(t12, [0, 0, -1]) - (-1.0 / 3.0)) < 1e-12
    for nphi, tol in ((24, 0.02), (48, 0.006)):
        V3, T3, L3 = build_triple_bubble_mesh(1.0, nphi=nphi)
        vols3 = _svol.region_volumes(V3, T3, L3)
        verr3 = float(np.max(np.abs(vols3 - geo3["V_cell"])
                             / geo3["V_cell"]))
        assert verr3 < tol, (nphi, verr3)
        # every region closed (each directed edge once with reverse)
        for reg in (1, 2, 3):
            sgn = ((L3[:, 1] == reg).astype(int) - (L3[:, 0] == reg))
            ecnt = {}
            for f, sf in zip(T3, sgn):
                if sf == 0:
                    continue
                tri = f if sf > 0 else f[::-1]
                for k in range(3):
                    e = (int(tri[k]), int(tri[(k + 1) % 3]))
                    ecnt[e] = ecnt.get(e, 0) + 1
            assert all(c == 1 and ecnt.get((b, a), 0) == 1
                       for (a, b), c in ecnt.items()), \
                f"triple region {reg} not closed at nphi={nphi}"
    # two tetra points: exactly 2 vertices carry 4 triple (3-face)
    # edges each; all other non-manifold vertices carry 2
    de3 = np.sort(np.concatenate([T3[:, [0, 1]], T3[:, [1, 2]],
                                  T3[:, [2, 0]]]), axis=1)
    uq3, cn3 = np.unique(de3, axis=0, return_counts=True)
    tri_edges = uq3[cn3 == 3]
    deg = np.zeros(len(V3), dtype=int)
    for a, b in tri_edges:
        deg[a] += 1
        deg[b] += 1
    n4 = int(np.sum(deg == 4))
    assert n4 == 2 and set(np.unique(deg[deg > 0])) == {2, 4}, \
        (n4, np.unique(deg))
    # area: the meshed seed converges to the closed-form total
    a24 = abs(_svol.mesh_area(*build_triple_bubble_mesh(1.0, 24)[:2])
              - geo3["A"]) / geo3["A"]
    a48 = abs(_svol.mesh_area(V3, T3) - geo3["A"]) / geo3["A"]
    assert a48 < a24 / 3.0 and a48 < 6e-3, (a24, a48)
    print(f"triple bubble seed: V_cell={geo3['V_cell']:.6f} "
          f"(mesh err {verr3:.1e}), A={geo3['A']:.6f} (mesh err "
          f"{a48:.1e}, x{a24 / a48:.1f} under refinement), regions "
          f"closed, 2 tetra points with 4 triple lines each")
    # evolution smoke: a perturbed equal triple bubble relaxes with
    # monotone area, exact volumes, and near-equal pressures ~ 2/r
    V3, T3, L3 = build_triple_bubble_mesh(1.0, nphi=24)
    V3 *= np.array([1.05, 0.97, 1.0])
    info3 = relax_cluster(V3, T3, L3, targets=[geo3["V_cell"]] * 3,
                          iters=80)
    drift3 = max(h["drift_post"] for h in info3["history"])
    rise3 = max(h["rise"] for h in info3["history"] if not h["groomed"])
    p3 = np.asarray(info3["pressures"])
    assert drift3 < 1e-10, drift3
    assert rise3 <= 1e-12, rise3
    assert np.all(np.abs(p3 - 2.0) < 0.1), p3
    print(f"relaxed triple bubble (nphi=24, 80 iters): drift "
          f"{drift3:.1e}, max rise {rise3:.1e}, p={p3.round(3)} "
          f"(want ~2.0)")
    # evolution smoke test (the full ground-truth battery lives in
    # tests/bench): a perturbed equal double bubble must relax with
    # monotone area, bounded volume drift, and Young-Laplace pressures
    Vd, Td, Ld = build_double_bubble_mesh(1.0, 1.0, nphi=24)
    geo = double_bubble_geometry(1.0, 1.0)
    Vd *= np.array([1.08, 0.96, 1.0])
    info = relax_cluster(Vd, Td, Ld, targets=[geo["V1"], geo["V2"]],
                         iters=60)
    drift = max(h["drift_post"] for h in info["history"])
    rise = max(h["rise"] for h in info["history"] if not h["groomed"])
    dp = abs(info["pressures"][0] - info["pressures"][1])
    perr = abs(info["pressures"][0] - 2.0) / 2.0
    assert drift < 1e-10, drift
    assert rise <= 1e-12, rise
    assert dp < 0.05, dp          # equal bubbles: no pressure jump
    assert perr < 0.05, perr      # Young-Laplace p = 2/r, r = 1
    print(f"relaxed double bubble (nphi=24, 60 iters): drift "
          f"{drift:.1e}, max rise {rise:.1e}, p={info['pressures'][0]:.3f}"
          f" (want 2.0), |p1-p2|={dp:.1e}")
    print("bubble standalone tests passed")
