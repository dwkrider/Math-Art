
# Toroidal Polyhedra for Blender
#
# Genus-1 (toroidal) polyhedra.  The Csaszar polyhedron is the unique
# (besides the tetrahedron) polyhedron with no diagonals: 7 vertices, 21
# edges forming the complete graph K7, and 14 triangular faces embedded on
# a torus.  Its dual, the Szilassi polyhedron, has 7 hexagonal faces every
# pair of which shares an edge -- the toroidal analogue of the tetrahedron.
# Both are stored from their original published coordinates and centred /
# fit to a 2 m cube on build.
#
# This module also builds parametric regular toroids -- a ring of congruent
# polygon cross-sections (prism / antiprism ring) -- and tiled toroids, where
# any of the uniform plane tilings (regular triangular/square/hexagonal, the
# Archimedean tilings, and their Laves duals) from the tiling engine is
# wrapped seamlessly onto a torus: the tiling's two lattice vectors map to
# the major and minor circles, giving a genus-1 polyhedron (V-E+F=0).
#
# References:
# - Akos Csaszar, "A polyhedron without diagonals", Acta Sci. Math.
#   Szeged 13 (1949-50), 140-142.
# - Lajos Szilassi, "Regular toroids", Structural Topology 13 (1986),
#   69-80; and "On three classes of regular toroids".
# - B. M. Stewart, "Adventures Among the Toroids" (1970/1980), for the
#   toroidal-polyhedron tradition.
# - Johannes Kepler, "Harmonices Mundi" (1619); Branko Grunbaum & G. C.
#   Shephard, "Tilings and Patterns" (1987) -- the uniform tilings wrapped
#   onto the torus here (see tiling_generator.py).

bl_info = {
    "name": "Toroidal Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Toroidal (genus-1) polyhedra: Csaszar & Szilassi, "
                   "polygon-ring toroids, and uniform tilings wrapped "
                   "onto a torus",
    "category": "Add Mesh",
}

import math

# Original published coordinates (Szilassi's tables); not unit-scaled.
TOROIDS = {
    "CSASZAR": {
        "name": "Csaszar Polyhedron",
        "V": [(3.0, -3.0, 0.0), (3.0, 3.0, 1.0), (1.0, 2.0, 3.0),
              (-1.0, -2.0, 3.0), (-3.0, -3.0, 1.0), (-3.0, 3.0, 0.0),
              (0.0, 0.0, 15.0)],
        "F": [[0, 5, 1], [0, 1, 3], [4, 1, 2], [3, 2, 0], [1, 5, 6],
              [2, 1, 6], [0, 2, 6], [5, 0, 4], [5, 4, 2], [1, 4, 3],
              [2, 3, 5], [4, 0, 6], [3, 4, 6], [5, 3, 6]],
    },
    "SZILASSI": {
        "name": "Szilassi Polyhedron",
        "V": [(-12.0, 0.0, 12.0), (-2.0, 5.0, -8.0), (0.0, -12.6, -12.0),
              (3.75, 3.75, -3.0), (-7.0, -2.5, 2.0), (-7.0, 0.0, 2.0),
              (-4.5, 2.5, 2.0), (12.0, 0.0, 12.0), (2.0, -5.0, -8.0),
              (0.0, 12.6, -12.0), (-3.75, -3.75, -3.0), (7.0, 2.5, 2.0),
              (7.0, 0.0, 2.0), (4.5, -2.5, 2.0)],
        "F": [[7, 11, 6, 3, 1, 0], [0, 1, 9, 2, 5, 4], [8, 10, 3, 6, 5, 2],
              [1, 3, 10, 13, 12, 9], [9, 12, 11, 7, 8, 2],
              [0, 4, 13, 10, 8, 7], [4, 5, 6, 11, 12, 13]],
    },
    # A genus-1 toroid all of whose faces are REGULAR polygons (every edge the
    # same length): 6 triangles + 9 squares + 9 hexagons, C3v -- the kind of
    # regular-faced toroid catalogued by B. M. Stewart.  Edge-equality (which
    # this satisfies exactly, spread ~1e-8) determines the solid, so these
    # clean coordinates are the canonical object, independently verified here.
    "REGULAR": {
        "name": "Regular-Faced Toroid",
        "V": [(0.5, 0.0, 1.0), (0.5, 0.0, -1.0), (-0.5, 0.0, 1.0),
              (-0.5, 0.0, -1.0), (1.0, 0.5, 0.0), (1.0, -0.5, 0.0),
              (-1.0, 0.5, 0.0), (-1.0, -0.5, 0.0), (0.0, 1.0, 0.5),
              (0.0, 1.0, -0.5), (0.0, -1.0, 0.5), (0.0, -1.0, -0.5),
              (0.0, 0.5, 1.0), (0.0, 0.5, -1.0), (0.0, -0.5, 1.0),
              (0.0, -0.5, -1.0), (1.0, 0.0, 0.5), (1.0, 0.0, -0.5),
              (-1.0, 0.0, 0.5), (-1.0, 0.0, -0.5), (0.5, 1.0, 0.0),
              (0.5, -1.0, 0.0), (-0.5, 1.0, 0.0), (-0.5, -1.0, 0.0),
              (-1 / 6, -1 / 6, -5 / 6), (5 / 6, -1 / 6, 1 / 6),
              (-1 / 6, 5 / 6, 1 / 6), (-1 / 3, 1 / 6, 1 / 3),
              (-1 / 3, -1 / 3, -1 / 6), (1 / 6, -1 / 3, 1 / 3)],
        "F": [[0, 14, 10, 21, 5, 16], [0, 16, 4, 20, 8, 12],
              [1, 24, 28, 29, 25, 17], [1, 17, 5, 21, 11, 15],
              [2, 12, 8, 22, 6, 18], [3, 15, 11, 23, 7, 19],
              [3, 19, 6, 22, 9, 13], [4, 25, 29, 27, 26, 20],
              [9, 26, 27, 28, 24, 13], [0, 12, 2, 14], [1, 15, 3, 13],
              [2, 27, 29, 14], [4, 16, 5, 17], [6, 19, 7, 18],
              [7, 28, 27, 18], [8, 20, 9, 22], [10, 29, 28, 23],
              [10, 23, 11, 21], [24, 1, 13], [25, 4, 17], [26, 9, 20],
              [27, 2, 18], [28, 7, 23], [29, 10, 14]],
    },
    # A genus-1 toroid whose 12 faces (6 quadrilaterals + 6 hexagons) wind
    # in a knotted band around the hole (D3 symmetry).  Coordinates are an
    # independent realization: relaxed to exactly planar faces under a free
    # D3 rotation action (9 DOF) and selected non-self-intersecting.
    "KNOTTED": {
        "name": "Knotted Dodecahedron",
        "V": [
            (0.417045, 0.180484, -0.494398), (-0.364826, 0.270929, -0.494398),
            (-0.052218, -0.451413, -0.494398), (0.417045, -0.180484, 0.494398),
            (-0.052218, 0.451413, 0.494398), (-0.364826, -0.270929, 0.494398),
            (0.417045, -0.180484, -0.849910), (-0.052218, 0.451413, -0.849910),
            (-0.364826, -0.270929, -0.849910), (0.417045, 0.180484, 0.849910),
            (-0.364826, 0.270929, 0.849910), (-0.052218, -0.451413, 0.849910),
            (0.923864, -0.399820, -0.393777), (0.923864, 0.399820, 0.393777),
            (-0.115677, 1.0, -0.393777), (-0.808187, 0.600180, 0.393777),
            (-0.808187, -0.600180, -0.393777), (-0.115677, -1.0, 0.393777)],
        "F": [[0, 6, 3, 9], [0, 9, 4, 7], [1, 7, 4, 10], [1, 10, 5, 8],
              [2, 8, 5, 11], [2, 11, 3, 6], [12, 6, 0, 7, 14, 13],
              [12, 13, 9, 3, 11, 17], [12, 17, 16, 8, 2, 6],
              [15, 10, 4, 9, 13, 14], [15, 14, 7, 1, 8, 16],
              [15, 16, 17, 11, 5, 10]],
    },
    "BORROMEAN": {
        "name": 'Borromean Rings',
        "V": [
            (0.735128291, 0.353162279, 0.0), (0.735128291, -0.353162279, 0.0), (-0.735128291, 0.353162279, 0.0),
            (-0.735128291, -0.353162279, 0.0), (1.0, 0.618033989, 0.194239254), (1.0, -0.618033989, 0.194239254),
            (-1.0, 0.618033989, 0.194239254), (-1.0, -0.618033989, 0.194239254), (1.0, 0.618033989, -0.194239254),
            (1.0, -0.618033989, -0.194239254), (-1.0, 0.618033989, -0.194239254), (-1.0, -0.618033989, -0.194239254),
            (0.0, 0.735128291, 0.353162279), (0.0, 0.735128291, -0.353162279), (0.0, -0.735128291, 0.353162279),
            (0.0, -0.735128291, -0.353162279), (0.194239254, 1.0, 0.618033989), (0.194239254, 1.0, -0.618033989),
            (0.194239254, -1.0, 0.618033989), (0.194239254, -1.0, -0.618033989), (-0.194239254, 1.0, 0.618033989),
            (-0.194239254, 1.0, -0.618033989), (-0.194239254, -1.0, 0.618033989), (-0.194239254, -1.0, -0.618033989),
            (0.353162279, 0.0, 0.735128291), (-0.353162279, 0.0, 0.735128291), (0.353162279, 0.0, -0.735128291),
            (-0.353162279, 0.0, -0.735128291), (0.618033989, 0.194239254, 1.0), (-0.618033989, 0.194239254, 1.0),
            (0.618033989, 0.194239254, -1.0), (-0.618033989, 0.194239254, -1.0), (0.618033989, -0.194239254, 1.0),
            (-0.618033989, -0.194239254, 1.0), (0.618033989, -0.194239254, -1.0), (-0.618033989, -0.194239254, -1.0),
        ],
        "F": [
            [4, 6, 2, 0], [1, 5, 4, 0], [8, 9, 1, 0], [2, 10, 8, 0], [11, 10, 2, 3], [1, 9, 11, 3],
            [7, 5, 1, 3], [2, 6, 7, 3], [8, 10, 6, 4], [5, 9, 8, 4], [11, 9, 5, 7], [6, 10, 11, 7],
            [16, 18, 14, 12], [13, 17, 16, 12], [20, 21, 13, 12], [14, 22, 20, 12], [23, 22, 14, 15], [13, 21, 23, 15],
            [19, 17, 13, 15], [14, 18, 19, 15], [20, 22, 18, 16], [17, 21, 20, 16], [23, 21, 17, 19], [18, 22, 23, 19],
            [28, 30, 26, 24], [25, 29, 28, 24], [32, 33, 25, 24], [26, 34, 32, 24], [35, 34, 26, 27], [25, 33, 35, 27],
            [31, 29, 25, 27], [26, 30, 31, 27], [32, 34, 30, 28], [29, 33, 32, 28], [35, 33, 29, 31], [30, 34, 35, 31],
        ],
    },
    "IRIS7": {
        "name": 'Heptagonal Iris Toroid',
        "V": [
            (1.0, 0.0, 0.433883739), (1.0, -0.0, -0.433883739), (-0.900968868, -0.433883739, 0.433883739),
            (-0.900968868, -0.433883739, -0.433883739), (0.623489802, 0.781831482, 0.433883739), (0.623489802, 0.781831482, -0.433883739),
            (-0.900968868, 0.433883739, 0.433883739), (-0.900968868, 0.433883739, -0.433883739), (0.623489802, -0.781831482, 0.433883739),
            (0.623489802, -0.781831482, -0.433883739), (-0.222520934, -0.974927912, 0.433883739), (-0.222520934, -0.974927912, -0.433883739),
            (-0.222520934, 0.974927912, 0.433883739), (-0.222520934, 0.974927912, -0.433883739),
        ],
        "F": [
            [0, 1, 5, 4], [4, 5, 13, 12], [12, 13, 7, 6], [6, 7, 3, 2], [2, 3, 11, 10], [10, 11, 9, 8],
            [8, 9, 1, 0], [0, 11, 3], [0, 3, 8], [2, 13, 5], [2, 5, 6], [4, 9, 11],
            [4, 11, 0], [6, 5, 1], [6, 1, 12], [8, 3, 7], [8, 7, 10], [10, 7, 13],
            [10, 13, 2], [12, 1, 9], [12, 9, 4],
        ],
    },
    "IRIS8": {
        "name": 'Octagonal Iris Toroid',
        "V": [
            (1.0, -0.0, 0.382683432), (1.0, 0.0, -0.382683432), (0.707106781, -0.707106781, 0.382683432),
            (0.707106781, -0.707106781, -0.382683432), (-0.707106781, 0.707106781, 0.382683432), (-0.707106781, 0.707106781, -0.382683432),
            (-1.0, 0.0, 0.382683432), (-1.0, 0.0, -0.382683432), (0.707106781, 0.707106781, 0.382683432),
            (0.707106781, 0.707106781, -0.382683432), (-0.0, -1.0, 0.382683432), (-0.0, -1.0, -0.382683432),
            (0.0, 1.0, 0.382683432), (-0.0, 1.0, -0.382683432), (-0.707106781, -0.707106781, 0.382683432),
            (-0.707106781, -0.707106781, -0.382683432),
        ],
        "F": [
            [0, 1, 9, 8], [8, 9, 13, 12], [12, 13, 5, 4], [4, 5, 7, 6], [6, 7, 15, 14], [14, 15, 11, 10],
            [10, 11, 3, 2], [2, 3, 1, 0], [0, 11, 15], [0, 15, 2], [2, 15, 7], [2, 7, 10],
            [4, 9, 1], [4, 1, 12], [6, 13, 9], [6, 9, 4], [8, 3, 11], [8, 11, 0],
            [10, 7, 5], [10, 5, 14], [12, 1, 3], [12, 3, 8], [14, 13, 6], [14, 5, 13],
        ],
    },
}

TOROID_ITEMS = [("CSASZAR", "Csaszar Polyhedron", "7 vertices, 14 "
                 "triangles, K7 (no diagonals)"),
                ("SZILASSI", "Szilassi Polyhedron", "7 hexagons, every "
                 "pair sharing an edge (dual of Csaszar)"),
                ("REGULAR", "Regular-Faced Toroid", "genus-1 toroid with "
                 "all regular faces (6 triangles + 9 squares + 9 hexagons)"),
                ("KNOTTED", "Knotted Dodecahedron", "genus-1 toroid whose 12 "
                 "faces wind in a knotted band (6 quads + 6 hexagons)"),
                ("BORROMEAN", "Borromean Rings", "three orthogonal genus-1 "
                 "rectangular rings; the standard Borromean link"),
                ("IRIS7", "Heptagonal Iris Toroid", "genus-1 D7 band of "
                 "7 squares + 14 triangles"),
                ("IRIS8", "Octagonal Iris Toroid", "genus-1 D8 band of "
                 "8 squares + 16 triangles")]


def build_toroid(kind):
    """Vertices/faces of the toroid, centred and fit to a 2 m cube."""
    S = TOROIDS[kind]
    V = [list(v) for v in S["V"]]
    c = [sum(v[i] for v in V) / len(V) for i in range(3)]
    V = [[v[i] - c[i] for i in range(3)] for v in V]
    mx = max(abs(x) for v in V for x in v) or 1.0
    V = [tuple(x / mx for x in v) for v in V]
    return V, [list(f) for f in S["F"]]


def build_polyhedral_torus(m, k, R, r, twist):
    """Ring of m k-gon cross-sections swept around a torus: twist=0 gives a
    prism ring (quad faces), twist=0.5 an antiprism ring (triangles).  A
    genus-1 regular toroid (V-E+F=0)."""
    verts = []
    for i in range(m):
        phi = 2 * math.pi * i / m
        off = twist * 2 * math.pi / k * i
        for j in range(k):
            a = 2 * math.pi * j / k + off
            verts.append((R * math.cos(phi) + r * math.cos(a) * math.cos(phi),
                          R * math.sin(phi) + r * math.cos(a) * math.sin(phi),
                          r * math.sin(a)))
    faces = []

    def idx(i, j):
        return (i % m) * k + (j % k)
    if abs(twist) < 1e-9:
        for i in range(m):
            for j in range(k):
                faces.append([idx(i, j), idx(i, j + 1),
                              idx(i + 1, j + 1), idx(i + 1, j)])
    else:
        for i in range(m):
            for j in range(k):
                faces.append([idx(i, j), idx(i, j + 1), idx(i + 1, j)])
                faces.append([idx(i, j + 1), idx(i + 1, j + 1),
                              idx(i + 1, j)])
    return verts, faces


def build_tiled_torus(name, nu, nv, R, r):
    """Wrap one of the uniform planar tilings from the tiling engine onto a
    torus: the tiling's lattice vector b1 maps to the major circle (nu
    periods) and b2 to the minor circle (nv periods), so the flat periodic
    pattern closes seamlessly into a genus-1 tiled toroid (V-E+F=0).  `name`
    is any key of tiling_generator.TILING_ITEMS (e.g. 'TRI', 'HEX',
    'TRIHEX', 'CAIRO', ...)."""
    try:
        from . import tiling_generator as tg
    except Exception:
        import tiling_generator as tg
    parent = tg.LAVES_OF.get(name, name)
    b1, b2, _ = tg._base_cell(parent)
    a11, a12, a21, a22 = b1[0], b2[0], b1[1], b2[1]   # B = [[a11,a12],[a21,a22]]
    det = a11 * a22 - a12 * a21

    def latt(x, y):                                    # (x,y) -> lattice (s,t)
        return ((a22 * x - a12 * y) / det, (-a21 * x + a11 * y) / det)

    polys = tg.substrate_faces(name, nu, nv)
    verts, index = [], {}

    def vid(s, t):                                     # lattice -> torus, welded
        u = 2.0 * math.pi * s / nu
        v = 2.0 * math.pi * t / nv
        x = (R + r * math.cos(v)) * math.cos(u)
        y = (R + r * math.cos(v)) * math.sin(u)
        z = r * math.sin(v)
        key = (round(x, 6), round(y, 6), round(z, 6))
        i = index.get(key)
        if i is None:
            i = len(verts)
            index[key] = i
            verts.append((x, y, z))
        return i

    seen, faces = set(), []
    for p in polys:
        pts = [(float(x), float(y)) for x, y in p]
        cx = sum(q[0] for q in pts) / len(pts)
        cy = sum(q[1] for q in pts) / len(pts)
        cs, ct = latt(cx, cy)
        key = (round(cs % nu, 4), round(ct % nv, 4))   # one tile per period
        if key in seen:
            continue
        seen.add(key)
        ring = [vid(*latt(x, y)) for (x, y) in pts]
        clean = [ring[i] for i in range(len(ring))
                 if ring[i] != ring[(i - 1) % len(ring)]]
        if len(clean) >= 3:
            faces.append(clean)
    return verts, faces


def _self_test():
    for kind, S in TOROIDS.items():
        V, F = build_toroid(kind)
        E = {}
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                k = (min(a, b), max(a, b))
                E[k] = E.get(k, 0) + 1
        chi = len(V) - len(E) + len(F)
        e2 = all(v == 2 for v in E.values())
        # planarity
        maxpl = 0.0
        for f in F:
            pts = [V[i] for i in f]
            cen = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
            nx = [0.0, 0.0, 0.0]
            for i in range(len(f)):
                p, q = pts[i], pts[(i + 1) % len(f)]
                nx[0] += (p[1] - q[1]) * (p[2] + q[2])
                nx[1] += (p[2] - q[2]) * (p[0] + q[0])
                nx[2] += (p[0] - q[0]) * (p[1] + q[1])
            ln = math.sqrt(sum(x * x for x in nx)) or 1.0
            nx = [x / ln for x in nx]
            for p in pts:
                maxpl = max(maxpl, abs(sum(nx[i] * (p[i] - cen[i])
                                          for i in range(3))))
        print(f"{kind:9s} V={len(V):2d} E={len(E):2d} F={len(F):2d} "
              f"chi={chi} edge-in-2={e2} planar={maxpl:.1e}")
    for name in ('TRI', 'HEX', 'TRIHEX', 'CAIRO'):
        V, F = build_tiled_torus(name, 12, 6, 1.0, 0.4)
        E = {}
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                kk = (min(a, b), max(a, b))
                E[kk] = E.get(kk, 0) + 1
        chi = len(V) - len(E) + len(F)
        e2 = all(v == 2 for v in E.values())
        print(f"tiled:{name:8s} V={len(V):3d} E={len(E):3d} F={len(F):3d} "
              f"chi={chi} edge-in-2={e2} (genus 1 -> chi 0)")


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_toroidal_polyhedron_add(bpy.types.Operator):
        """Add a toroidal (genus-1) polyhedron: the Csaszar polyhedron
        (no diagonals) or its dual the Szilassi polyhedron"""
        bl_idname = "mesh.toroidal_polyhedron_add"
        bl_label = "Toroidal Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(name="Solid", items=TOROID_ITEMS)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            V, F = build_toroid(self.solid)
            me = bpy.data.meshes.new(TOROIDS[self.solid]["name"])
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(TOROIDS[self.solid]["name"], me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{TOROIDS[self.solid]['name']}: "
                        f"V={len(V)} F={len(F)} (genus 1)")
            return {'FINISHED'}

    from bpy.props import IntProperty, BoolProperty

    # Tilings that wrap cleanly onto a torus (all 19 uniform tilings except
    # Rhombille, whose Laves-dual patch does not fold to a single period).
    try:
        from . import tiling_generator as _tg
    except Exception:
        import tiling_generator as _tg
    _TORUS_TILING_EXCLUDE = {'RHOMBILLE'}
    TORUS_TILING_ITEMS = [it for it in _tg.TILING_ITEMS
                          if it[0] not in _TORUS_TILING_EXCLUDE]

    class MESH_OT_polyhedral_torus_add(bpy.types.Operator):
        """Add a regular polyhedral torus: a ring of congruent polygon
        cross-sections (prism / antiprism ring), or a uniform tiling
        (triangles, hexagons, and the other Archimedean/Laves patterns
        from the tiling engine) wrapped onto the torus"""
        bl_idname = "mesh.polyhedral_torus_add"
        bl_label = "Polyhedral Torus"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Pattern",
            items=[('RING', "Polygon Ring",
                    "A ring of congruent polygon cross-sections"),
                   ('TILING', "Uniform Tiling",
                    "A uniform plane tiling wrapped onto the torus")],
            default='RING')
        segments: IntProperty(name="Segments", default=12, min=3, max=64,
                              description="Sections around the major circle")
        sides: IntProperty(name="Cross-section Sides", default=4, min=3,
                           max=32)
        antiprism: BoolProperty(
            name="Antiprism Ring", default=False,
            description="Half-step twist between sections (triangular "
                        "faces) instead of a prism ring (quads)")
        tiling: EnumProperty(name="Tiling", items=TORUS_TILING_ITEMS,
                             default='HEX')
        cells_u: IntProperty(
            name="Cells Around", default=12, min=3, max=64,
            description="Tiling periods around the major circle")
        cells_v: IntProperty(
            name="Cells Through", default=6, min=2, max=48,
            description="Tiling periods around the minor circle (tube)")
        major: FloatProperty(name="Major Radius", default=1.0, min=0.1,
                             max=10.0)
        minor: FloatProperty(name="Minor Radius", default=0.4, min=0.02,
                             max=5.0)
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed torus"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "modifier"),
                   ('WIRE', "Struts", "Wireframe modifier"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
        facet_depth: FloatProperty(name="Depth", default=0.1, min=0.01,
                                   max=1.0,
                                   description="Face Segments inward depth")
        facet_gap: FloatProperty(name="Bevel Gap", default=0.0, min=0.0,
                                 max=0.5,
                                 description="Gap between face segments")
        facet_explode: FloatProperty(name="Explode", default=0.0, min=0.0,
                                     max=5.0,
                                     description="Move segments outward")
        facet_separate: BoolProperty(
            name="Separate Meshes", default=False,
            description="Each face segment as its own object")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            if self.mode == 'TILING':
                V, F = build_tiled_torus(self.tiling, self.cells_u,
                                         self.cells_v, self.major, self.minor)
                name = "Tiled Torus %s" % self.tiling
            else:
                V, F = build_polyhedral_torus(
                    self.segments, self.sides, self.major, self.minor,
                    0.5 if self.antiprism else 0.0)
                name = "Polyhedral Torus"
            mx = max((abs(c) for v in V for c in v), default=1.0) or 1.0
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(c / mx * self.scale for c in v)
                            for v in V], [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            # consistent outward normals (needed for Face Segments)
            try:
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                bm.to_mesh(me)
                bm.free()
            except Exception:
                pass
            me.update()
            if self.style == 'FACETS':
                Vf = [tuple(v.co) for v in me.vertices]
                Ff = [list(p.vertices) for p in me.polygons]
                bpy.data.meshes.remove(me)
                try:
                    from . import facet_style
                except ImportError:
                    import facet_style
                # explode away from the major (tube-centre) circle,
                # which lies in the z=0 plane at the scaled major radius
                import math as _m
                rmaj = self.major / mx * self.scale

                def _edir(cen):
                    rl = _m.hypot(cen[0], cen[1])
                    if rl < 1e-9:
                        return (0.0, 0.0, 1.0 if cen[2] >= 0 else -1.0)
                    dx = cen[0] - rmaj * cen[0] / rl
                    dy = cen[1] - rmaj * cen[1] / rl
                    dz = cen[2]
                    L = _m.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
                    return (dx / L, dy / L, dz / L)
                facet_style.emit_facets(
                    context, Vf, Ff, name, self.facet_depth,
                    self.facet_gap, self.facet_explode,
                    self.facet_separate, None, _edir)
                self.report({'INFO'}, "%s: %d face segments" %
                            (name, len(Ff)))
                return {'FINISHED'}
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness
                mod.use_even_offset = False
            elif self.style == 'BALLSTICK':
                try:
                    from . import ball_and_stick
                except ImportError:
                    import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'}, "%s: V=%d F=%d (genus 1)" %
                        (name, len(V), len(F)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'TILING':
                lay.prop(self, 'tiling')
                lay.prop(self, 'cells_u')
                lay.prop(self, 'cells_v')
            else:
                lay.prop(self, 'segments')
                lay.prop(self, 'sides')
                lay.prop(self, 'antiprism')
            lay.prop(self, 'major')
            lay.prop(self, 'minor')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            if self.style == 'FACETS':
                lay.prop(self, 'facet_depth')
                lay.prop(self, 'facet_gap')
                lay.prop(self, 'facet_explode')
                lay.prop(self, 'facet_separate')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.toroidal_polyhedron_add",
                             icon='MESH_TORUS')
        self.layout.operator("mesh.polyhedral_torus_add", icon='MESH_TORUS')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_toroidal_polyhedron_add)
        bpy.utils.register_class(MESH_OT_polyhedral_torus_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polyhedral_torus_add)
        bpy.utils.unregister_class(MESH_OT_toroidal_polyhedron_add)
