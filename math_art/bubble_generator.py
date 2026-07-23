
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


def build_bubbles(P, R, subdiv=3, films=True):
    """(verts, faces, ids): the bubble cluster.  ids: bubble
    index for outer caps, n + pair-index for interior films."""
    P = np.asarray(P, float)
    R = np.asarray(R, float)
    pairs = _interfaces(P, R)
    SV, SF = _icosphere(subdiv)
    verts = []
    faces = []
    ids = []

    def emit(v, f, idx):
        base = len(verts)
        verts.extend(tuple(p) for p in v)
        used = set()
        for t in f:
            faces.append([base + i for i in t])
            ids.append(idx)
            used.update(t)

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
        emit(V1, F1, i)

    if films:
        h = 1.0515 * R.min() / (2 ** subdiv)
        for pi, ((i, j), pr) in enumerate(pairs.items()):
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
                emit(V1, F1, len(P) + pi)
    return verts, faces, ids


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
            name="Subdivisions", default=3, min=1, max=5,
            description="Icosphere subdivisions per bubble")
        films: BoolProperty(
            name="Interior Films", default=True,
            description="Include the soap films between "
                        "touching bubbles")
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
            verts, faces, ids = build_bubbles(
                P, R, self.subdiv, self.films)
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set(
                'use_smooth', [self.smooth] * len(me.polygons))
            attr = me.attributes.new("bubble_index", 'INT',
                                     'FACE')
            if len(me.polygons) == len(ids):
                attr.data.foreach_set('value', ids)
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {len(P)} bubbles "
                        f"V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('seed', 'radius_mode', 'factor', 'subdiv',
                      'films', 'smooth', 'scale'):
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
        print("bubble standalone tests passed")
