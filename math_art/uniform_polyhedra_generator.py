
# Uniform Polyhedra Generator for Blender
#
# Builds the uniform polyhedra by Wythoff's kaleidoscopic construction:
# each solid comes from a Schwarz triangle (p q r) -- a spherical triangle
# with angles pi/p, pi/q, pi/r whose three sides are mirrors.  The three
# reflections generate a finite group (tetrahedral, octahedral or
# icosahedral).  A generator point is placed by the Wythoff symbol (the
# point equidistant from the "active" mirrors, on the others), its orbit
# under the group gives the vertices, and each face is the cyclic or
# dihedral orbit of that point about a rotation axis -- a {n/d} polygon,
# a star polygon when d > 1.  This yields the convex uniform solids, the
# regular star (Kepler-Poinsot) polyhedra, and the non-convex star
# uniforms including the hemipolyhedra (faces through the centre).
#
# This build covers the 63 non-snub uniform polyhedra (Wythoff symbols
# with the bar not first); the 11 snubs and the great
# dirhombicosidodecahedron use a separate generator-point solve and are
# added in a later revision.
#
# Star faces are rendered by fanning each {n/d} polygon from its centre,
# so the star outline shows as a solid.
#
# References:
# - W. A. Wythoff, "A relation between the polytopes of the C600-family",
#   Koninklijke Akademie van Wetenschappen te Amsterdam (1918).
# - H. S. M. Coxeter, M. S. Longuet-Higgins, J. C. P. Miller, "Uniform
#   polyhedra", Phil. Trans. Royal Soc. A 246 (1954), 401-450.
# - S. P. Skilling, "The complete set of uniform polyhedra", Phil. Trans.
#   Royal Soc. A 278 (1975) (completeness).
# - Zvi Har'El, "Uniform Solution for Uniform Polyhedra", Geometriae
#   Dedicata 47 (1993), 57-110 (the construction algorithm followed here).
# - Magnus Wenninger, "Polyhedron Models", Cambridge (1971).
# - Numbering (U1-U75) after Roman Maeder's Kaleido / Mathematica package.

bl_info = {
    "name": "Uniform Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "The uniform polyhedra (Wythoff construction): convex, "
                   "Kepler-Poinsot and non-convex star uniforms",
    "category": "Add Mesh",
}

import math

try:
    import numpy as np
except ImportError:
    np = None


# u, name, wythoff, pqr, symmetry, V, E, F  (all 75; snubs/U75 flagged
# by a bar-first Wythoff symbol or a 4-entry pqr -- not built yet)
UNIFORMS = [
    (1, "Tetrahedron", "3 | 2 3", ["3", "2", "3"], 4, 6, 4),
    (2, "Truncated Tetrahedron", "2 3 | 3", ["2", "3", "3"], 12, 18, 8),
    (3, "Octahemioctahedron", "3/2 3 | 3", ["3/2", "3", "3"], 12, 24, 12),
    (4, "Tetrahemihexahedron", "3/2 3 | 2", ["3/2", "3", "2"], 6, 12, 7),
    (5, "Octahedron", "4 | 2 3", ["4", "2", "3"], 6, 12, 8),
    (6, "Cube", "3 | 2 4", ["3", "2", "4"], 8, 12, 6),
    (7, "Cuboctahedron", "2 | 3 4", ["2", "3", "4"], 12, 24, 14),
    (8, "Truncated Octahedron", "2 4 | 3", ["2", "4", "3"], 24, 36, 14),
    (9, "Truncated Cube", "2 3 | 4", ["2", "3", "4"], 24, 36, 14),
    (10, "Rhombicuboctahedron", "3 4 | 2", ["3", "4", "2"], 24, 48, 26),
    (11, "Truncated Cuboctahedron", "2 3 4 |", ["2", "3", "4"], 48, 72, 26),
    (12, "Snub Cube", "| 2 3 4", ["2", "3", "4"], 24, 60, 38),
    (13, "Small Cubicuboctahedron", "3/2 4 | 4", ["3/2", "4", "4"],
     24, 48, 20),
    (14, "Great Cubicuboctahedron", "3 4 | 4/3", ["3", "4", "4/3"],
     24, 48, 20),
    (15, "Cubohemioctahedron", "4/3 4 | 3", ["4/3", "4", "3"], 12, 24, 10),
    (16, "Cubitruncated Cuboctahedron", "4/3 3 4 |", ["4/3", "3", "4"],
     48, 72, 20),
    (17, "Nonconvex Great Rhombicuboctahedron", "3/2 4 | 2",
     ["3/2", "4", "2"], 24, 48, 26),
    (18, "Small Rhombihexahedron", "3/2 2 4 |", ["3/2", "2", "4"],
     24, 48, 18),
    (19, "Stellated Truncated Hexahedron", "2 3 | 4/3", ["2", "3", "4/3"],
     24, 36, 14),
    (20, "Great Truncated Cuboctahedron", "4/3 2 3 |", ["4/3", "2", "3"],
     48, 72, 26),
    (21, "Great Rhombihexahedron", "4/3 3/2 2 |", ["4/3", "3/2", "2"],
     24, 48, 18),
    (22, "Icosahedron", "5 | 2 3", ["5", "2", "3"], 12, 30, 20),
    (23, "Dodecahedron", "3 | 2 5", ["3", "2", "5"], 20, 30, 12),
    (24, "Icosidodecahedron", "2 | 3 5", ["2", "3", "5"], 30, 60, 32),
    (25, "Truncated Icosahedron", "2 5 | 3", ["2", "5", "3"], 60, 90, 32),
    (26, "Truncated Dodecahedron", "2 3 | 5", ["2", "3", "5"], 60, 90, 32),
    (27, "Rhombicosidodecahedron", "3 5 | 2", ["3", "5", "2"], 60, 120, 62),
    (28, "Truncated Icosidodecahedron", "2 3 5 |", ["2", "3", "5"],
     120, 180, 62),
    (29, "Snub Dodecahedron", "| 2 3 5", ["2", "3", "5"], 60, 150, 92),
    (30, "Small Ditrigonal Icosidodecahedron", "3 | 5/2 3",
     ["3", "5/2", "3"], 20, 60, 32),
    (31, "Small Icosicosidodecahedron", "5/2 3 | 3", ["5/2", "3", "3"],
     60, 120, 52),
    (32, "Small Snub Icosicosidodecahedron", "| 5/2 3 3",
     ["5/2", "3", "3"], 60, 180, 112),
    (33, "Small Dodecicosidodecahedron", "3/2 5 | 5", ["3/2", "5", "5"],
     60, 120, 44),
    (34, "Small Stellated Dodecahedron", "5 | 2 5/2", ["5", "2", "5/2"],
     12, 30, 12),
    (35, "Great Dodecahedron", "5/2 | 2 5", ["5/2", "2", "5"], 12, 30, 12),
    (36, "Dodecadodecahedron", "2 | 5 5/2", ["2", "5", "5/2"], 30, 60, 24),
    (37, "Truncated Great Dodecahedron", "2 5/2 | 5", ["2", "5/2", "5"],
     60, 90, 24),
    (38, "Rhombidodecadodecahedron", "5/2 5 | 2", ["5/2", "5", "2"],
     60, 120, 54),
    (39, "Small Rhombidodecahedron", "2 5/2 5 |", ["2", "5/2", "5"],
     60, 120, 42),
    (40, "Snub Dodecadodecahedron", "| 2 5/2 5", ["2", "5/2", "5"],
     60, 150, 84),
    (41, "Ditrigonal Dodecadodecahedron", "3 | 5/3 5", ["3", "5/3", "5"],
     20, 60, 24),
    (42, "Great Ditrigonal Dodecicosidodecahedron", "3 5 | 5/3",
     ["3", "5", "5/3"], 60, 120, 44),
    (43, "Small Ditrigonal Dodecicosidodecahedron", "5/3 3 | 5",
     ["5/3", "3", "5"], 60, 120, 44),
    (44, "Icosidodecadodecahedron", "5/3 5 | 3", ["5/3", "5", "3"],
     60, 120, 44),
    (45, "Icositruncated Dodecadodecahedron", "3 5 5/3 |",
     ["3", "5", "5/3"], 120, 180, 44),
    (46, "Snub Icosidodecadodecahedron", "| 5/3 3 5", ["5/3", "3", "5"],
     60, 180, 104),
    (47, "Great Ditrigonal Icosidodecahedron", "3/2 | 3 5",
     ["3/2", "3", "5"], 20, 60, 32),
    (48, "Great Icosicosidodecahedron", "3/2 5 | 3", ["3/2", "5", "3"],
     60, 120, 52),
    (49, "Small Icosihemidodecahedron", "3/2 3 | 5", ["3/2", "3", "5"],
     30, 60, 26),
    (50, "Small Dodecicosahedron", "3/2 3 5 |", ["3/2", "3", "5"],
     60, 120, 32),
    (51, "Small Dodecahemidodecahedron", "5/4 5 | 5", ["5/4", "5", "5"],
     30, 60, 18),
    (52, "Great Stellated Dodecahedron", "3 | 2 5/2", ["3", "2", "5/2"],
     20, 30, 12),
    (53, "Great Icosahedron", "5/2 | 2 3", ["5/2", "2", "3"], 12, 30, 20),
    (54, "Great Icosidodecahedron", "2 | 3 5/2", ["2", "3", "5/2"],
     30, 60, 32),
    (55, "Truncated Great Icosahedron", "2 5/2 | 3", ["2", "5/2", "3"],
     60, 90, 32),
    (56, "Rhombicosahedron", "2 5/2 3 |", ["2", "5/2", "3"], 60, 120, 50),
    (57, "Great Snub Icosidodecahedron", "| 2 5/2 3", ["2", "5/2", "3"],
     60, 150, 92),
    (58, "Small Stellated Truncated Dodecahedron", "2 5 | 5/3",
     ["2", "5", "5/3"], 60, 90, 24),
    (59, "Truncated Dodecadodecahedron", "2 5 5/3 |", ["2", "5", "5/3"],
     120, 180, 54),
    (60, "Inverted Snub Dodecadodecahedron", "| 5/3 2 5",
     ["5/3", "2", "5"], 60, 150, 84),
    (61, "Great Dodecicosidodecahedron", "5/2 3 | 5/3",
     ["5/2", "3", "5/3"], 60, 120, 44),
    (62, "Small Dodecahemicosahedron", "5/3 5/2 | 3", ["5/3", "5/2", "3"],
     30, 60, 22),
    (63, "Great Dodecicosahedron", "5/3 5/2 3 |", ["5/3", "5/2", "3"],
     60, 120, 32),
    (64, "Great Snub Dodecicosidodecahedron", "| 5/3 5/2 3",
     ["5/3", "5/2", "3"], 60, 180, 104),
    (65, "Great Dodecahemicosahedron", "5/4 5 | 3", ["5/4", "5", "3"],
     30, 60, 22),
    (66, "Great Stellated Truncated Dodecahedron", "2 3 | 5/3",
     ["2", "3", "5/3"], 60, 90, 32),
    (67, "Nonconvex Great Rhombicosidodecahedron", "5/3 3 | 2",
     ["5/3", "3", "2"], 60, 120, 62),
    (68, "Great Truncated Icosidodecahedron", "2 3 5/3 |",
     ["2", "3", "5/3"], 120, 180, 62),
    (69, "Great Inverted Snub Icosidodecahedron", "| 5/3 2 3",
     ["5/3", "2", "3"], 60, 150, 92),
    (70, "Great Dodecahemidodecahedron", "5/3 5/2 | 5/3",
     ["5/3", "5/2", "5/3"], 30, 60, 18),
    (71, "Great Icosihemidodecahedron", "3/2 3 | 5/3", ["3/2", "3", "5/3"],
     30, 60, 26),
    (72, "Small Retrosnub Icosicosidodecahedron", "| 3/2 3/2 5/2",
     ["3/2", "3/2", "5/2"], 60, 180, 112),
    (73, "Great Rhombidodecahedron", "3/2 5/3 2 |", ["3/2", "5/3", "2"],
     60, 120, 42),
    (74, "Great Retrosnub Icosidodecahedron", "| 2 3/2 5/3",
     ["2", "3/2", "5/3"], 60, 150, 92),
    (75, "Great Dirhombicosidodecahedron", "| 3/2 5/3 3 5/2",
     ["3/2", "5/3", "3", "5/2"], 60, 240, 124),
]


def _frac(x):
    s = str(x)
    if '/' in s:
        p, q = s.split('/')
        return float(p) / float(q), (int(p), int(q))
    return float(x), (int(float(x)), 1)


def _mirrors(pqr):
    o = [_frac(x)[0] for x in pqr]
    n0 = np.array([1.0, 0.0, 0.0])
    c01 = -math.cos(math.pi / o[2])
    n1 = np.array([c01, math.sqrt(max(0.0, 1 - c01 * c01)), 0.0])
    c02 = -math.cos(math.pi / o[1])
    c12 = -math.cos(math.pi / o[0])
    x2 = c02
    y2 = (c12 - n1[0] * x2) / n1[1]
    z2 = math.sqrt(abs(1 - x2 * x2 - y2 * y2))
    return [n0, n1, np.array([x2, y2, z2])]


def _key(v, p=5):
    return (round(float(v[0]), p), round(float(v[1]), p),
            round(float(v[2]), p))


def _mkey(M, p=4):
    return tuple(round(float(x), p) for x in M.ravel())


def _refl_group(ns, maxn=4000):
    refl = [np.eye(3) - 2 * np.outer(n, n) for n in ns]
    seen = {_mkey(np.eye(3)): np.eye(3)}
    frontier = [np.eye(3)]
    while frontier:
        nf = []
        for M in frontier:
            for r in refl:
                N = r @ M
                k = _mkey(N)
                if k not in seen:
                    seen[k] = N
                    nf.append(N)
                    if len(seen) > maxn:
                        raise RuntimeError("symmetry group too large")
        frontier = nf
    return list(seen.values())


def _orbit(P, G):
    d = {}
    for M in G:
        q = M @ P
        d.setdefault(_key(q), q)
    return list(d.values())


def _rot(ax, a):
    x, y, z = ax
    c = math.cos(a)
    s = math.sin(a)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _ring_mask(wythoff):
    toks = wythoff.split()
    bar = toks.index('|')
    mask = []
    idx = 0
    for t in toks:
        if t == '|':
            continue
        mask.append(1.0 if idx < bar else 0.0)
        idx += 1
    return mask


def build_uniform(wythoff, pqr):
    """(verts, faces) where each face is (vertex-index list, density d).
    Non-snub Wythoff symbols only."""
    if np is None:
        raise RuntimeError("uniform polyhedra need NumPy")
    ns = _mirrors(pqr)
    G = _refl_group(ns)
    rm = _ring_mask(wythoff)
    P = np.linalg.solve(np.array([ns[0], ns[1], ns[2]]),
                        np.array(rm, dtype=float))
    P = P / np.linalg.norm(P)
    verts = _orbit(P, G)
    index = {_key(v): i for i, v in enumerate(verts)}

    def vidx(q):
        return index.get(_key(q))
    faces = []
    seen = set()
    for i in range(3):
        j, k = [t for t in range(3) if t != i]
        rc = int(rm[j] + rm[k])
        if rc == 0:
            continue
        _, (num, den) = _frac(pqr[i])
        Vi = np.cross(ns[j], ns[k])
        Vi = Vi / np.linalg.norm(Vi)
        if rc == 2:
            reflj = np.eye(3) - 2 * np.outer(ns[j], ns[j])
            raw = []
            for kk in range(num):
                R = _rot(Vi, 2 * math.pi * kk / num)
                raw.append(R @ P)
                raw.append(R @ (reflj @ P))
            nv = 2 * num
        else:
            raw = [_rot(Vi, 2 * math.pi * kk / num) @ P for kk in range(num)]
            nv = num
        if nv < 3:
            continue
        u = raw[0] - np.dot(raw[0], Vi) * Vi
        u = u / np.linalg.norm(u)
        w = np.cross(Vi, u)

        def az(t):
            d = raw[t] - np.dot(raw[t], Vi) * Vi
            return math.atan2(np.dot(d, w), np.dot(d, u))
        order = sorted(range(len(raw)), key=az)
        rawo = [raw[t] for t in order]
        seed = [rawo[(den * t) % nv] for t in range(nv)]
        for M in G:
            f = [vidx(M @ p) for p in seed]
            if None in f or len(set(f)) != len(f):
                continue
            key = frozenset(f)
            if key not in seen:
                seen.add(key)
                faces.append((f, den))
    return verts, faces


# solids this build can construct: non-snub (bar not first), 3-triangle
BUILDABLE = [row for row in UNIFORMS
             if not row[2].strip().startswith('|') and len(row[3]) == 3]


# --- families (to organise the UI) --------------------------------------
# Convex (Platonic + Archimedean), the regular stars (Kepler-Poinsot), the
# hemipolyhedra (faces through the centre), and the remaining star uniforms
# split by symmetry (octahedral vs icosahedral).
_KEPLER_POINSOT = {34, 35, 52, 53}

_FAMILIES = [
    ('CONVEX', "Convex (Platonic & Archimedean)"),
    ('KEPLER', "Regular Star (Kepler-Poinsot)"),
    ('HEMI', "Hemipolyhedra"),
    ('STAR_O', "Star (Octahedral)"),
    ('STAR_I', "Star (Icosahedral)"),
]


def _symmetry(pqr):
    toks = ' '.join(pqr)
    if '5' in toks:
        return 'I'
    if '4' in toks:
        return 'O'
    return 'T'


def _category(u, name, pqr):
    if 'hemi' in name.lower():
        return 'HEMI'
    if u in _KEPLER_POINSOT:
        return 'KEPLER'
    if not any('/' in x for x in pqr):
        return 'CONVEX'
    return 'STAR_O' if _symmetry(pqr) == 'O' else 'STAR_I'


_BY_CAT = {cat: [] for cat, _lbl in _FAMILIES}
for _row in BUILDABLE:
    _BY_CAT[_category(_row[0], _row[1], _row[3])].append(_row)


def _self_test():
    ok = 0
    bad = 0
    for (u, name, wy, pqr, Ve, Ee, Fe) in BUILDABLE:
        V, F = build_uniform(wy, pqr)
        E = set()
        for f, _d in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                E.add((min(a, b), max(a, b)))
        got = (len(V), len(E), len(F))
        if got == (Ve, Ee, Fe):
            ok += 1
        else:
            bad += 1
            print(f"U{u} {name}: got {got} expected {(Ve, Ee, Fe)}")
    print(f"uniform polyhedra: {ok}/{ok + bad} correct")


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _family_items(self, context):
        return [(cat, lbl, "") for cat, lbl in _FAMILIES if _BY_CAT[cat]]

    _SOLID_CACHE = {}

    def _solid_items(self, context):
        cat = self.family
        if cat not in _SOLID_CACHE:
            _SOLID_CACHE[cat] = [(str(u), f"U{u}  {name}", wy)
                                 for (u, name, wy, pqr, V, E, F)
                                 in _BY_CAT.get(cat, [])]
        return _SOLID_CACHE[cat]

    def _family_update(self, context):
        ids = [it[0] for it in _solid_items(self, context)]
        if ids and self.solid not in ids:
            self.solid = ids[0]

    _PALETTE = {3: (0.90, 0.36, 0.23), 4: (0.27, 0.52, 0.79),
                5: (0.30, 0.69, 0.42), 6: (0.95, 0.77, 0.29),
                8: (0.25, 0.72, 0.72), 10: (0.55, 0.60, 0.29)}

    def _material_for(nn):
        name = f"Uniform {nn}-gon"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            import colorsys
            rgb = _PALETTE.get(nn, colorsys.hsv_to_rgb((nn * 0.618) % 1.0,
                                                       0.55, 0.8))
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.5
        return mat

    class MESH_OT_uniform_polyhedron_add(bpy.types.Operator):
        """Add a uniform polyhedron by Wythoff construction: convex,
        Kepler-Poinsot and non-convex star uniforms.  Star faces are
        triangulated from their centre so the star shows as a solid."""
        bl_idname = "mesh.uniform_polyhedron_add"
        bl_label = "Uniform Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(name="Family", items=_family_items,
                             update=_family_update)
        solid: EnumProperty(name="Solid", items=_solid_items)
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "By Face Size", ""), ('NONE', "None", "")],
            default='SIDES')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            lay.prop(self, 'solid')
            lay.prop(self, 'coloring')
            lay.prop(self, 'scale')

        def execute(self, context):
            ids = [it[0] for it in _solid_items(self, context)]
            sid = self.solid if self.solid in ids else (ids[0] if ids
                                                       else self.solid)
            row = next((r for r in BUILDABLE if str(r[0]) == sid),
                       BUILDABLE[0])
            u, name, wy, pqr, Ve, Ee, Fe = row
            try:
                V, F = build_uniform(wy, pqr)
            except Exception as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            s = self.scale
            verts = [tuple(c * s for c in v) for v in V]
            faces = []
            fsize = []
            for f, d in F:
                if d == 1:               # convex polygon: keep as n-gon
                    faces.append(list(f))
                    fsize.append(len(f))
                else:                    # star: fan from the centre
                    c = [sum(verts[i][k] for i in f) / len(f)
                         for k in range(3)]
                    ci = len(verts)
                    verts.append(tuple(c))
                    for i in range(len(f)):
                        faces.append([f[i], f[(i + 1) % len(f)], ci])
                        fsize.append(len(f))
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if self.coloring == 'SIDES' and len(me.polygons) == len(faces):
                sizes = sorted(set(fsize))
                slot = {n: i for i, n in enumerate(sizes)}
                for n in sizes:
                    me.materials.append(_material_for(n))
                me.polygons.foreach_set('material_index',
                                        [slot[s] for s in fsize])
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{name}: V={len(V)} E={Ee} F={len(F)}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.uniform_polyhedron_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_uniform_polyhedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_uniform_polyhedron_add)


if __name__ == "__main__":
    if not _IN_BLENDER:
        _self_test()
