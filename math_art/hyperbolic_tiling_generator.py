
# Hyperbolic Tiling Generator for Blender
#
# Regular {p,q} tilings of the hyperbolic plane -- q p-gons around
# every vertex, hyperbolic whenever 1/p + 1/q < 1/2 -- realized as
# printable relief models after Henry Segerman, "Visualizing
# Mathematics with 3D Printing" (figs 4-2, 4-8..4-14):
#
#   DISK    Poincare disk plaque: the conformal disk picture with
#           tiles of one orientation raised from a backing slab
#   HEMI    hemisphere model: Klein disk coordinates lifted
#           vertically onto the upper unit hemisphere, a shell
#           with tiles offset radially by parity
#   PSEUDO  pseudosphere (tractricoid): the tiling wrapped onto
#           the surface of constant curvature -1 via the upper
#           half plane -- a genuinely isometric embedding of a
#           strip of H^2
#
# The tiling is generated in the hyperboloid model of H^2 sitting
# in Minkowski R^{2,1}, <x,y> = x1 y1 + x2 y2 - x3 y3.  The
# (p,q,2) triangle group is realized by three mirrors with unit
# spacelike normals whose pairwise Gram products are -cos(pi/p)
# (at the p-fold vertex), -cos(pi/q), and 0 (the right angle);
# reflection in a mirror is x -> x - 2<x,n>n.  The plane is tiled
# by BFS over reflection words, 2-coloured by word parity, and
# every fundamental triangle is barycentrically subdivided and
# reprojected to <x,x> = -1 so its edges follow geodesics after
# mapping into the target model.
#
# Pseudosphere isometry (derivation): the tractricoid
#   sigma(u,v) = (sech u cos v, sech u sin v, u - tanh u), u >= 0
# has sigma_u = (-sech u tanh u cos v, -sech u tanh u sin v,
# tanh^2 u), sigma_v = (-sech u sin v, sech u cos v, 0), giving
# the first fundamental form
#   ds^2 = tanh^2 u du^2 + sech^2 u dv^2.
# Substituting x = v, y = cosh u into the upper-half-plane metric
# (dx^2 + dy^2)/y^2 gives (dv^2 + sinh^2 u du^2)/cosh^2 u =
# sech^2 u dv^2 + tanh^2 u du^2 -- the same form.  So y = cosh u
# (NOT y = e^u, which gives du^2 + e^{-2u} dv^2, a horocyclic
# chart of H^2 but not this surface's metric) wraps the UHP
# region y >= 1, x taken mod 2 pi, isometrically onto the
# pseudosphere.  The u = 0 rim corresponds to y = 1 and the cusp
# to y -> infinity, so tiles are clipped to 1 <= y <= y_cap and
# one seam period 0 <= x <= 2 pi.

bl_info = {
    "name": "Hyperbolic Tiling",
    "author": "David Krider (Math Art project, after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Tiling",
    "description": "{p,q} hyperbolic tilings as disk plaque, "
                   "hemisphere, or pseudosphere reliefs",
    "category": "Add Mesh",
}

from math import cos, sin, pi, sqrt, hypot
from collections import deque

import numpy as np

_J = np.diag([1.0, 1.0, -1.0])


def _mdot(a, b):
    return a[0] * b[0] + a[1] * b[1] - a[2] * b[2]


def fundamental_domain(p, q):
    """Mirror normals and fundamental triangle of the (p,q,2)
    reflection group in the hyperboloid model.  Returns
    (normals, corners) with corners rows [A, B, C]: A the p-fold
    vertex (0,0,1), B the q-fold vertex, C the right-angled one;
    each corner satisfies <c,c> = -1, t > 0."""
    sp, cp = sin(pi / p), cos(pi / p)
    cq = cos(pi / q)
    # hyperbolic iff cos(pi/q) > sin(pi/p)  <=>  1/p + 1/q < 1/2
    if cq <= sp + 1e-12:
        raise ValueError("{%d,%d} is not hyperbolic "
                         "(need 1/p + 1/q < 1/2)" % (p, q))
    n0 = np.array([0.0, 1.0, 0.0])          # mirror: y = 0
    n1 = np.array([sp, -cp, 0.0])           # <n0,n1> = -cos(pi/p)
    a = -cq / sp                            # <n1,n2> = -cos(pi/q)
    c = sqrt(a * a - 1.0)                   # <n2,n2> = 1
    n2 = np.array([a, 0.0, c])              # <n0,n2> = 0
    A = np.array([0.0, 0.0, 1.0])
    C = np.array([-c, 0.0, -a])             # on mirrors 0 and 2
    B = _J @ np.cross(n1, n2)               # Mink.-orth. to n1, n2
    B = B / sqrt(-_mdot(B, B))
    if B[2] < 0.0:
        B = -B
    return (n0, n1, n2), np.array([A, B, C])


def enumerate_tiles(p, q, depth=8, max_tiles=4000):
    """BFS over reflection words.  Returns (corners, tiles) where
    tiles is a list of (matrix, parity, word length): the image
    triangle is matrix @ corner for each fundamental corner, and
    parity (word length mod 2) 2-colours the tiling.  Duplicate
    group elements are pruned by the rounded ambient centroid of
    the image triangle."""
    normals, corners = fundamental_domain(p, q)
    refl = [np.eye(3) - 2.0 * np.outer(n, _J @ n) for n in normals]
    cen0 = corners.mean(axis=0)
    tiles, seen = [], set()
    queue = deque([(np.eye(3), 0, 0)])
    while queue and len(tiles) < max_tiles:
        M, par, d = queue.popleft()
        key = tuple(np.round(M @ cen0, 5))
        if key in seen:
            continue
        seen.add(key)
        tiles.append((M, par, d))
        if d < depth:
            for R in refl:
                queue.append((M @ R, par ^ 1, d + 1))
    return corners, tiles


def _grid(corners, subdiv):
    """Barycentric 2^subdiv subdivision of the fundamental
    triangle, reprojected to the hyperboloid (convex combinations
    of points on H^2 are timelike, so normalizing to <x,x> = -1
    is safe).  Returns (points, tris, boundary loop)."""
    n = 1 << subdiv
    pts, idx = [], {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            P = (i * corners[0] + j * corners[1]
                 + (n - i - j) * corners[2]) / n
            pts.append(P / sqrt(-_mdot(P, P)))
            idx[(i, j)] = len(pts) - 1
    tris = []
    for i in range(n):
        for j in range(n - i):
            a, b, c = idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]
            tris.append((a, b, c))
            if i + j < n - 1:
                tris.append((b, idx[(i + 1, j + 1)], c))
    loop = ([idx[(i, 0)] for i in range(n, 0, -1)]
            + [idx[(0, j)] for j in range(n)]
            + [idx[(i, n - i)] for i in range(n)])
    return np.array(pts), tris, loop


def _to_disk(H):
    """Hyperboloid -> Poincare disk: (x, y, t) -> (x, y)/(1+t)."""
    return H[:, :2] / (1.0 + H[:, 2:3])


def _cayley(z):
    """Poincare disk -> upper half plane, 0 -> i, 1 -> 0."""
    return 1j * (1.0 - z) / (1.0 + z)


def _pseudosphere(x, y):
    """UHP (x, y >= 1) -> pseudosphere point and outward unit
    normal via the isometry u = arccosh(y), v = x derived in the
    header.  sech u = 1/y, tanh u = sqrt(y^2-1)/y; the unit
    normal (tanh u cos v, tanh u sin v, sech u) comes from
    sigma_u x sigma_v = -sech u tanh u * that vector."""
    y = np.maximum(y, 1.0)
    se = 1.0 / y
    th = np.sqrt(y * y - 1.0) / y
    cx, sx = np.cos(x), np.sin(x)
    S = np.column_stack([se * cx, se * sx, np.arccosh(y) - th])
    N = np.column_stack([th * cx, th * sx, se])
    return S, N


def _prism(verts, faces, mats, top, bot, tris, loop, mat):
    """Closed shell between two copies of the tile grid: top
    faces, reversed bottom faces, and wall quads along the
    boundary loop.  Winding is unified later by a face-normal
    recalculation, per closed shell."""
    b0, m = len(verts), len(top)
    verts.extend(map(tuple, top))
    verts.extend(map(tuple, bot))
    for a, b, c in tris:
        faces.append((b0 + a, b0 + b, b0 + c))
        faces.append((b0 + m + c, b0 + m + b, b0 + m + a))
    for t in range(len(loop)):
        a, b = loop[t], loop[(t + 1) % len(loop)]
        faces.append((b0 + a, b0 + b, b0 + m + b, b0 + m + a))
    mats.extend([mat] * (2 * len(tris) + len(loop)))


def _slab(verts, faces, mats, radius, height, mat, seg=128):
    """Closed backing cylinder for the disk plaque, top at z=0."""
    b0 = len(verts)
    for z in (0.0, -height):
        for k in range(seg):
            a = 2.0 * pi * k / seg
            verts.append((radius * cos(a), radius * sin(a), z))
    faces.append(tuple(b0 + k for k in range(seg)))
    faces.append(tuple(b0 + seg + k for k in reversed(range(seg))))
    for k in range(seg):
        k2 = (k + 1) % seg
        faces.append((b0 + k, b0 + seg + k,
                      b0 + seg + k2, b0 + k2))
    mats.extend([mat] * (seg + 2))


def build_tiling(p, q, model='DISK', depth=8, max_tiles=4000,
                 subdiv=2, relief=0.05, base=0.1, thickness=0.08,
                 y_cap=6.0):
    """Build the relief mesh.  Returns (verts, faces, mat_ids)
    with mat_ids 0 (raised parity) / 1 (other parity or base) per
    face.  Every emitted tile is its own watertight closed shell;
    for printing the shells overlap (prisms sink into the slab,
    neighbouring shells share full side walls) so a boolean or
    slicer union is trivial."""
    corners, tiles = enumerate_tiles(p, q, depth, max_tiles)
    gpts, gtris, gloop = _grid(corners, subdiv)
    verts, faces, mats = [], [], []

    if model == 'PSEUDO':
        # centre the view: Moebius-shift the UHP picture so the
        # q-fold vertex B lands at x = pi, y = 2 -- on the fat
        # part of the horn, in the middle of the seam period
        d = _to_disk(corners[1].reshape(1, 3))[0]
        w0 = _cayley(complex(d[0], d[1]))
        shift, mag = w0.real, 2.0 / w0.imag

    for M, par, _ in tiles:
        H = gpts @ M.T
        if model == 'DISK':
            if par:                  # one orientation raised only
                continue
            D = _to_disk(H)
            if np.max(D[:, 0] ** 2 + D[:, 1] ** 2) > 0.995 ** 2:
                continue             # sub-sliver boundary tiles
            top = np.column_stack([D, np.full(len(D), relief)])
            # sink prism bottoms into the slab for a solid union
            bot = np.column_stack([D, np.full(len(D),
                                              -0.45 * base)])
        elif model == 'HEMI':
            K = H[:, :2] / H[:, 2:3]        # Klein coordinates
            zz = np.sqrt(np.maximum(
                0.0, 1.0 - K[:, 0] ** 2 - K[:, 1] ** 2))
            S = np.column_stack([K, zz])    # unit hemisphere
            top = S * (1.0 + (relief if par == 0 else 0.0))
            bot = S * (1.0 - thickness)
        else:                               # PSEUDO
            D = _to_disk(H)
            w = _cayley(D[:, 0] + 1j * D[:, 1])
            w = (w - shift) * mag + pi      # UHP isometry
            x, y = w.real, w.imag
            if (y.min() < 1.0 or y.max() > y_cap
                    or x.min() < 0.0 or x.max() > 2.0 * pi):
                continue             # rim / cusp / seam clipping
            S, N = _pseudosphere(x, y)
            top = S + N * (relief if par == 0 else 0.0)
            bot = S - N * thickness
        _prism(verts, faces, mats, top, bot, gtris, gloop, par)

    if model == 'DISK':
        _slab(verts, faces, mats, 0.999, base, 1)
    return verts, faces, mats


try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _material(name, color):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = color
            mat.use_nodes = True
            node = mat.node_tree.nodes.get("Principled BSDF")
            if node:
                node.inputs[0].default_value = color
        return mat

    class MESH_OT_hyperbolic_tiling_add(bpy.types.Operator):
        """Add a {p,q} hyperbolic tiling relief (Poincare disk
        plaque, hemisphere shell, or pseudosphere)"""
        bl_idname = "mesh.hyperbolic_tiling_add"
        bl_label = "Hyperbolic Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        p: IntProperty(
            name="p", default=3, min=3, max=20,
            description="Sides of each tile ({p,q}: q p-gons "
                        "meet at every vertex)")
        q: IntProperty(
            name="q", default=7, min=3, max=24,
            description="Tiles meeting at each vertex "
                        "(hyperbolic requires 1/p + 1/q < 1/2)")
        model: EnumProperty(
            name="Model",
            items=[('DISK', "Poincare Disk",
                    "Relief plaque of the conformal disk "
                    "picture: raised tiles on a backing slab"),
                   ('HEMI', "Hemisphere",
                    "Klein disk lifted onto the upper unit "
                    "hemisphere, tiles offset radially"),
                   ('PSEUDO', "Pseudosphere",
                    "Tiling wrapped isometrically onto the "
                    "tractricoid via the upper half plane")],
            default='DISK')
        depth: IntProperty(
            name="Depth", default=8, min=1, max=14,
            description="Reflection word length explored")
        max_tiles: IntProperty(
            name="Max Triangles", default=4000, min=10,
            max=20000,
            description="Cap on fundamental triangles generated")
        subdiv: IntProperty(
            name="Subdivision", default=2, min=0, max=4,
            description="Geodesic subdivisions per triangle "
                        "(edges curve correctly in the model)")
        relief: FloatProperty(
            name="Relief Height", default=0.05, min=0.005,
            max=0.5,
            description="Height of the raised-parity tiles")
        base: FloatProperty(
            name="Base Thickness", default=0.1, min=0.01,
            max=1.0,
            description="Backing slab thickness (disk plaque)")
        thickness: FloatProperty(
            name="Shell Thickness", default=0.08, min=0.01,
            max=0.5,
            description="Shell thickness (hemisphere / "
                        "pseudosphere)")
        y_cap: FloatProperty(
            name="Cusp Cap", default=6.0, min=1.5, max=50.0,
            description="Clip the pseudosphere cusp at UHP "
                        "height y (the horn gets thin)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            try:
                verts, faces, mats = build_tiling(
                    self.p, self.q, self.model, self.depth,
                    self.max_tiles, self.subdiv, self.relief,
                    self.base, self.thickness, self.y_cap)
            except ValueError as err:
                self.report({'ERROR'}, str(err))
                return {'CANCELLED'}
            if not faces:
                self.report({'ERROR'}, "no tiles survived "
                                       "clipping")
                return {'CANCELLED'}
            if self.scale != 1.0:
                s = self.scale
                verts = [(x * s, y * s, z * s)
                         for x, y, z in verts]
            me = bpy.data.meshes.new("Hyperbolic Tiling")
            me.from_pydata(verts, [], faces)
            me.materials.append(_material(
                "Hyperbolic Tile A", (0.55, 0.13, 0.08, 1.0)))
            me.materials.append(_material(
                "Hyperbolic Tile B", (0.87, 0.80, 0.66, 1.0)))
            me.polygons.foreach_set('material_index', mats)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.update()
            obj = bpy.data.objects.new(
                "Hyperbolic Tiling {%d,%d}" % (self.p, self.q),
                me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'p')
            lay.prop(self, 'q')
            lay.prop(self, 'model')
            lay.prop(self, 'depth')
            lay.prop(self, 'max_tiles')
            lay.prop(self, 'subdiv')
            lay.prop(self, 'relief')
            if self.model == 'DISK':
                lay.prop(self, 'base')
            else:
                lay.prop(self, 'thickness')
            if self.model == 'PSEUDO':
                lay.prop(self, 'y_cap')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_tiling_add",
                             icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_tiling_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(
            MESH_OT_hyperbolic_tiling_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # {3,7}: every q-fold tiling vertex whose ring must be
        # complete at this BFS depth (a ring needs at most 7 more
        # letters than its nearest triangle) has exactly 2q = 14
        # incident fundamental triangles
        DEPTH = 10
        corners, tiles = enumerate_tiles(3, 7, depth=DEPTH,
                                         max_tiles=100000)
        cnt, dmin = {}, {}
        for M, par, d in tiles:
            b = M @ corners[1]
            dk = b[:2] / (1.0 + b[2])
            k = (round(dk[0], 5), round(dk[1], 5))
            cnt[k] = cnt.get(k, 0) + 1
            dmin[k] = min(dmin.get(k, 99), d)
        full = [k for k in cnt if dmin[k] + 7 <= DEPTH]
        bad = [k for k in full if cnt[k] != 14]
        v, f, mt = build_tiling(3, 7, 'DISK')
        rmax = max(hypot(x, y) for x, y, z in v)
        ok = full and not bad and rmax < 1.0
        print(f"{{3,7}}: tiles={len(tiles)} "
              f"full q-vertices={len(full)} bad={len(bad)} "
              f"disk V={len(v)} F={len(f)} rmax={rmax:.4f} "
              f"{'OK' if ok else 'BAD'}")
