
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


def _icosphere(subdiv):
    t = (1 + sqrt(5)) / 2
    V = [np.array(v, float) for v in
         [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
          (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
          (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    V = [v / np.linalg.norm(v) for v in V]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10),
         (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache = {}
        nf = []

        def mid(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in cache:
                m = V[i] + V[j]
                cache[key] = len(V)
                V.append(m / np.linalg.norm(m))
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (ab, b, bc), (ca, bc, c),
                   (ab, bc, ca)]
        F = nf
    return np.array(V), F


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
            # fit (roughly) within a 2 x scale cube at the origin
            allv = [v for p in parts for v in p[1]]
            lo = [min(v[k] for v in allv) for k in range(3)]
            hi = [max(v[k] for v in allv) for k in range(3)]
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

    def _menu_func(self, context):
        self.layout.operator("mesh.bubble_cluster_add",
                             icon='SPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_bubble_cluster_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_bubble_cluster_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
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
        print("bubble standalone tests passed")
