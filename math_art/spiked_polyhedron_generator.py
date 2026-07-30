
# Spiked & Hyperbolic Polyhedra generator for Blender.
#
#   SPIKED    an icosahedron (or any Platonic seed) augmented with
#             a pyramid on every face.  At the default height
#             sqrt(6)/3 x edge the pyramids on the icosahedron are
#             regular tetrahedra and the result is a 60-face
#             deltahedron.  A GOLDBERG seed instead spikes every
#             face of a Goldberg polyhedron (12 pentagons plus
#             hexagons), after Robert Fathauer's "Pollen Form" /
#             "Fungus Form" ceramics.
#   HYPER     hyperbolic polyhedra: faces sag toward the centre
#             while the vertices spike outward.  Any Platonic seed
#             with sharpness / spike-length / resolution controls.
#   MODERN    the folded rhombic-hexecontahedron family.  Each
#             icosahedron face carries a tent whose mid vertices
#             sit along a+b and whose top sits along a+b+c (a, b,
#             c the face's icosahedral vertex vectors); at
#             mid = top = 1 the 120 triangles fuse in pairs into
#             the 60 golden rhombi of the rhombic hexecontahedron,
#             and lifting the tops (~5%) folds each rhombus gently
#             along its long diagonal.
#   RHOMBIC   the flat rhombic hexecontahedron itself, emitted as
#             60 planar golden rhombi -- the union of 20 acute
#             golden rhombohedra sharing the centre.
#
# References:
# - Deltahedra and face-augmented (pyramid-capped) Platonic solids:
#   H. Martyn Cundy and A. P. Rollett, "Mathematical Models", 2nd ed.,
#   Oxford University Press, 1961.
# - Rhombic hexecontahedron (60 golden rhombi, a stellation of the
#   rhombic triacontahedron / union of 20 acute golden rhombohedra):
#   see Stan Wagon; E. W. Weisstein, "Rhombic Hexecontahedron",
#   MathWorld.
# - Golden ratio in the icosahedron/dodecahedron and the rhombic
#   zonohedra: H. S. M. Coxeter, "Regular Polytopes", 3rd ed., 1973.
# - Goldberg polyhedra: Michael Goldberg, "A class of multi-symmetric
#   polyhedra", Tohoku Mathematical Journal 43 (1937), 104-108.
# - Spiked Goldberg forms after Robert Fathauer's "Pollen Form" and
#   "Fungus Form" ceramic sculptures.

bl_info = {
    "name": "Spiked & Hyperbolic Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Spiked & Hyperbolic",
    "description": "Spiked, hyperbolic and rhombic polyhedra",
    "category": "Add Mesh",
}

import math
from math import sqrt

import numpy as np

PHI = (1 + sqrt(5)) / 2


# ---------------------------------------------------------------- #
#  Platonic seeds (circumradius 1)                                 #
# ---------------------------------------------------------------- #

def _icosa():
    V = []
    for a in (-1.0, 1.0):
        for b in (-PHI, PHI):
            V += [(0.0, a, b), (a, b, 0.0), (b, 0.0, a)]
    V = np.array(V) / sqrt(1 + PHI * PHI)
    d2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    emin = np.min(d2[d2 > 1e-9])
    adj = d2 < emin + 1e-6
    faces = []
    n = len(V)
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if adj[i][k] and adj[j][k]:
                    f = [i, j, k]
                    c = V[f].mean(0)
                    if np.dot(np.cross(V[j] - V[i], V[k] - V[i]),
                              c) < 0:
                        f = [i, k, j]
                    faces.append(f)
    return V, faces


def _dual(V, F):
    """Dual polyhedron: vertex per face (normalized centroid),
    face per vertex (incident faces ordered around it)."""
    C = np.array([V[f].mean(0) for f in F])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    faces = []
    for i in range(len(V)):
        inc = [fi for fi, f in enumerate(F) if i in f]
        nrm = V[i] / np.linalg.norm(V[i])
        ref = C[inc[0]] - np.dot(C[inc[0]], nrm) * nrm
        ref /= np.linalg.norm(ref)
        sid = np.cross(nrm, ref)
        inc.sort(key=lambda fi: math.atan2(
            np.dot(C[fi], sid), np.dot(C[fi], ref)))
        c = C[inc].mean(0)
        if np.dot(np.cross(C[inc[1]] - C[inc[0]],
                           C[inc[2]] - C[inc[0]]), c) < 0:
            inc.reverse()
        faces.append(inc)
    return C, faces


def _seed(name):
    if name == 'TETRA':
        V = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1),
                      (-1, -1, 1)]) / sqrt(3)
        F = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        return V, F
    if name == 'CUBE':
        V = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1)
                      for z in (-1, 1)]) / sqrt(3)
        F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        return V, F
    if name == 'OCTA':
        V = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                      (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
        F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
             [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
        return V, F
    if name == 'ICOSA':
        return _icosa()
    if name == 'DODECA':
        return _dual(*_icosa())
    raise ValueError(name)


def _goldberg(freq=2):
    """Goldberg polyhedron GP(freq, 0): the dual of the Class-I
    frequency-`freq` icosahedral geodesic sphere -- 12 pentagons plus
    10(freq^2 - 1) hexagons (Michael Goldberg, 1937).  Robert
    Fathauer's "Pollen Form" / "Fungus Form" ceramics spike a chiral
    Class-III Goldberg, GP(1,2); the sibling geodesic module only
    builds Class I (f,0) and Class II (f,f) breakdowns, so GP(2,0) --
    the same hex-plus-12-pentagon sphere minus the chiral twist --
    stands in for it here."""
    try:
        from . import geodesic_generator as geo
    except ImportError:
        import geodesic_generator as geo
    V, F = geo.build_sphere('ICOSA', freq, 'I')
    Vd, Fd = geo.goldberg_dual(V, F)
    return np.asarray(Vd, float), Fd


# ---------------------------------------------------------------- #
#  the three constructions                                         #
# ---------------------------------------------------------------- #

def build_spiked(seed='ICOSA', height=sqrt(6.0) / 3.0, freq=2):
    """Every face of the seed replaced by a pyramid of the given
    height (in units of the edge length).  ICOSA at sqrt(6)/3
    gives a 60-face deltahedron of regular tetrahedra.  The GOLDBERG
    seed pyramids the hexagons and 12 pentagons of GP(freq, 0)
    instead -- the Pollen / Fungus form of Fathauer's ceramics."""
    if seed == 'GOLDBERG':
        V, F = _goldberg(freq)
    else:
        V, F = _seed(seed)
    verts = [tuple(v) for v in V]
    faces = []
    groups = []
    for gi, f in enumerate(F):
        c = V[f].mean(0)
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        n /= np.linalg.norm(n)
        if np.dot(n, c) < 0.0:          # keep the spikes outward
            n = -n
        e = np.linalg.norm(V[f[1]] - V[f[0]])
        apex = len(verts)
        verts.append(tuple(c + n * height * e))
        for i in range(len(f)):
            faces.append((f[i], f[(i + 1) % len(f)], apex))
            groups.append(gi)
    return verts, faces, groups


def build_hyper(seed='DODECA', spike=1.5, sharpness=3.0, res=12):
    """Hyperbolic polyhedron: the flat surface is subdivided and
    every point is pushed radially by a factor that stays near 1
    over the face interiors and rises to `spike` at the vertices,
    with `sharpness` controlling how abruptly the spikes rise
    (1 = nearly the flat solid)."""
    V, F = _seed(seed)
    # distance from centre to the face planes (the inradius)
    rmin = min(abs(np.dot(V[f[0]],
                          np.cross(V[f[1]] - V[f[0]],
                                   V[f[2]] - V[f[0]])
                          / np.linalg.norm(
                              np.cross(V[f[1]] - V[f[0]],
                                       V[f[2]] - V[f[0]]))))
               for f in F)
    vid = {}
    verts = []
    faces = []
    groups = []

    def key(p):
        return tuple(np.round(p, 9))

    def emit(p):
        k = key(p)
        if k not in vid:
            vid[k] = len(verts)
            verts.append(np.asarray(p, float))
        return vid[k]

    for gi, f in enumerate(F):
        c = V[f].mean(0)
        for i in range(len(f)):
            A, B = V[f[i]], V[f[(i + 1) % len(f)]]
            # barycentric grid over triangle (c, A, B)
            P = {}
            for r in range(res + 1):
                for s in range(res + 1 - r):
                    t = res - r - s
                    P[(r, s)] = emit((r * c + s * A + t * B) / res)
            for r in range(res):
                for s in range(res - r):
                    a = P[(r, s)]
                    b = P[(r + 1, s)]
                    d = P[(r, s + 1)]
                    faces.append((a, d, b))
                    groups.append(gi)
                    if s < res - r - 1:
                        e = P[(r + 1, s + 1)]
                        faces.append((b, d, e))
                        groups.append(gi)
    W = np.array(verts)
    rf = np.linalg.norm(W, axis=1)
    w = np.clip((rf - rmin) / (1.0 - rmin), 0.0, 1.0) ** sharpness
    g = 1.0 + (spike - 1.0) * w
    W = W * g[:, None]
    return [tuple(p) for p in W], faces, groups


def build_modern(mid=1.0, top=1.05, quads=False):
    """The rhombic hexecontahedron and its folded family.  Per
    icosahedron face with vertex vectors a, b, c: rhombi
    (a, a+b, a+b+c, a+c) with the mid vertices (a+b) scaled by
    `mid` and the tops (a+b+c) by `top`.  mid = top = 1 is exactly
    the rhombic hexecontahedron (60 planar golden rhombi, the
    union of 20 acute golden rhombohedra); top > 1 folds each
    rhombus along its long diagonal into two slightly non-planar
    triangles.  quads=True emits rhombi as quads (only sensible
    when they are planar)."""
    V, F = _icosa()
    vid = {}
    verts = []
    faces = []
    groups = []

    def emit(p):
        k = tuple(np.round(p, 9))
        if k not in vid:
            vid[k] = len(verts)
            verts.append(tuple(p))
        return vid[k]

    def wind(idx):
        P = np.array([verts[i] for i in idx])
        n = np.zeros(3)
        for i in range(1, len(idx) - 1):
            n += np.cross(P[i] - P[0], P[i + 1] - P[0])
        return idx if np.dot(n, P.mean(0)) > 0 \
            else tuple(reversed(idx))

    for gi, f in enumerate(F):
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        t = emit((a + b + c) * top)
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            vp = emit(p)
            m1 = emit((p + q) * mid)
            m2 = emit((p + r) * mid)
            if quads:
                faces.append(wind((vp, m1, t, m2)))
                groups.append(gi)
            else:
                faces.append(wind((vp, m1, t)))
                faces.append(wind((vp, t, m2)))
                groups.extend((gi, gi))
    # normalize so the rhombus tips sit at radius ~ top
    r = np.linalg.norm(V[F[0][0]] + V[F[0][1]] + V[F[0][2]])
    verts = [tuple(np.asarray(v) / r) for v in verts]
    return verts, faces, groups


def edge_face_counts(faces):
    cnt = {}
    for f in faces:
        for i in range(len(f)):
            e = frozenset((f[i], f[(i + 1) % len(f)]))
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


# ---------------------------------------------------------------- #
#  Blender operator                                                #
# ---------------------------------------------------------------- #

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


SEED_ITEMS = [('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
              ('OCTA', "Octahedron", ""),
              ('DODECA', "Dodecahedron", ""),
              ('ICOSA', "Icosahedron", "")]

# The SPIKED preset additionally accepts a Goldberg polyhedron seed
# (12 pentagons + hexagons, Michael Goldberg 1937); spiked, it gives
# the Pollen / Fungus form of Robert Fathauer's ceramics.
SPIKED_SEED_ITEMS = SEED_ITEMS + [
    ('GOLDBERG', "Goldberg Sphere",
     "Goldberg polyhedron GP(f,0): 12 pentagons plus hexagons; a "
     "spike on every face gives the Pollen / Fungus form")]


if _IN_BLENDER:

    _PALETTE = [(0.85, 0.30, 0.20), (0.25, 0.50, 0.78),
                (0.32, 0.68, 0.40), (0.94, 0.76, 0.28),
                (0.60, 0.40, 0.74), (0.27, 0.71, 0.71),
                (0.90, 0.55, 0.70), (0.55, 0.60, 0.30),
                (0.80, 0.47, 0.22), (0.42, 0.44, 0.78),
                (0.68, 0.68, 0.68), (0.52, 0.45, 0.40)]

    def _material(idx):
        name = f"Spiked Group {idx}"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            rgb = _PALETTE[idx % len(_PALETTE)]
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb,
                                                           1.0)
                bsdf.inputs["Roughness"].default_value = 0.5
        return mat

    class MESH_OT_spiked_polyhedron_add(bpy.types.Operator):
        """Add a spiked, hyperbolic or rhombic polyhedron: pyramid-
        augmented Platonic solids, hyperbolic polyhedra with spiked
        vertices, or the rhombic hexecontahedron and its folded
        variant"""
        bl_idname = "mesh.spiked_polyhedron_add"
        bl_label = "Spiked & Hyperbolic"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Form",
            items=[('SPIKED', "Spiked Polyhedron",
                    "Pyramid on every face; the icosahedron at "
                    "height sqrt(6)/3 gives a 60-face deltahedron "
                    "of regular tetrahedra"),
                   ('HYPER', "Hyperbolic Polyhedron",
                    "Concave faces, spiked vertices, any Platonic "
                    "seed"),
                   ('MODERN', "Folded Rhombic Hexecontahedron",
                    "Each golden rhombus gently folded along its "
                    "long diagonal (adjustable mid / tip scales)"),
                   ('RHOMBIC', "Rhombic Hexecontahedron",
                    "The flat 60-golden-rhombus solid, a "
                    "stellation of the rhombic triacontahedron")],
            default='MODERN')
        seed: EnumProperty(name="Seed", items=SPIKED_SEED_ITEMS,
                           default='ICOSA',
                           description="Base solid (Platonic, or a "
                                       "Goldberg polyhedron)")
        goldberg_freq: IntProperty(
            name="Frequency", default=2, min=1, max=8,
            description="Goldberg seed only: GP(f,0) frequency; f=2 "
                        "gives 12 pentagons + 30 hexagons")
        hyper_seed: EnumProperty(name="Seed", items=SEED_ITEMS,
                                 default='DODECA',
                                 description="Base Platonic solid")
        height: FloatProperty(
            name="Spike Height", default=sqrt(6.0) / 3.0,
            min=0.01, max=10.0,
            description="Pyramid height in units of the edge "
                        "length; sqrt(6)/3 = 0.816 makes regular "
                        "tetrahedra on triangular faces")
        spike: FloatProperty(
            name="Spike Length", default=1.5, min=1.0, max=5.0,
            description="Radial factor at the vertices")
        sharpness: FloatProperty(
            name="Sharpness", default=3.0, min=1.0, max=12.0,
            description="How abruptly the spikes rise from the "
                        "sagging faces")
        resolution: IntProperty(
            name="Resolution", default=12, min=2, max=48,
            description="Subdivision of each face")
        mid: FloatProperty(
            name="Mid Scale", default=1.0, min=0.5, max=2.0,
            description="Radial scale of the rhombus mid "
                        "vertices (1 = rhombic hexecontahedron)")
        top: FloatProperty(
            name="Top Scale", default=1.05, min=0.5, max=2.0,
            description="Radial scale of the rhombus tips; 1 is "
                        "the flat rhombic hexecontahedron, ~1.05 "
                        "a gentle fold")
        coloring: EnumProperty(
            name="Coloring",
            items=[('GROUP', "By Face Group",
                    "One material per seed face (12-color "
                    "palette)"),
                   ('NONE', "None", "")],
            default='NONE')
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels"),
                   ('WIRE', "Wireframe", "Wireframe modifier")],
            default='SOLID')
        border: FloatProperty(name="Border", default=0.3,
                              min=0.02, max=0.95)
        thickness: FloatProperty(name="Thickness", default=0.04,
                                 min=0.001, max=1.0)
        smooth: BoolProperty(
            name="Smooth Shading", default=True,
            description="Hyperbolic preset only")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            p = self.preset
            if p == 'SPIKED':
                verts, faces, groups = build_spiked(
                    self.seed, self.height, self.goldberg_freq)
                name = f"Spiked {self.seed.title()}"
            elif p == 'HYPER':
                verts, faces, groups = build_hyper(
                    self.hyper_seed, self.spike, self.sharpness,
                    self.resolution)
                name = f"Hyperbolic {self.hyper_seed.title()}"
            elif p == 'MODERN':
                verts, faces, groups = build_modern(self.mid,
                                                    self.top)
                name = "Folded Rhombic Hexecontahedron"
            else:
                verts, faces, groups = build_modern(1.0, 1.0,
                                                    quads=True)
                name = "Rhombic Hexecontahedron"
            s = self.scale
            me = bpy.data.meshes.new(name)
            me.from_pydata([(x * s, y * s, z * s)
                            for x, y, z in verts], [],
                           [list(f) for f in faces])
            me.validate(clean_customdata=True)
            if p == 'HYPER' and self.smooth:
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
            if self.coloring == 'GROUP' \
                    and len(me.polygons) == len(groups):
                for i in range(len(_PALETTE)):
                    me.materials.append(_material(i))
                me.polygons.foreach_set(
                    'material_index',
                    [g % len(_PALETTE) for g in groups])
                attr = me.attributes.new("face_group", 'INT',
                                         'FACE')
                attr.data.foreach_set('value', groups)
            me.update()
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
            self.report({'INFO'},
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            p = self.preset
            if p == 'SPIKED':
                lay.prop(self, 'seed')
                if self.seed == 'GOLDBERG':
                    lay.prop(self, 'goldberg_freq')
                lay.prop(self, 'height')
            elif p == 'HYPER':
                lay.prop(self, 'hyper_seed')
                lay.prop(self, 'spike')
                lay.prop(self, 'sharpness')
                lay.prop(self, 'resolution')
                lay.prop(self, 'smooth')
            elif p == 'MODERN':
                lay.prop(self, 'mid')
                lay.prop(self, 'top')
            lay.prop(self, 'coloring')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style != 'SOLID':
                lay.prop(self, 'thickness')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator_menu_enum("mesh.spiked_polyhedron_add",
                                       "preset",
                                       text="Spiked & Hyperbolic",
                                       icon='LIGHT_SUN')

    _classes = (MESH_OT_spiked_polyhedron_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # spiked icosahedron: 60-face deltahedron of regular
        # tetrahedra
        V, F, G = build_spiked('ICOSA')
        E = [np.linalg.norm(np.array(V[f[i]])
                            - np.array(V[f[(i + 1) % 3]]))
             for f in F for i in range(3)]
        spread = (max(E) - min(E)) / max(E)
        cnt = edge_face_counts(F)
        closed = all(c == 2 for c in cnt.values())
        print(f"spiked icosa: {len(F)} faces, edge spread "
              f"{spread:.2e} (deltahedron), closed={closed}")
        assert len(F) == 60 and spread < 1e-9 and closed

        # spiked Goldberg GP(2,0): 12 pentagons + 30 hexagons, each
        # pyramided -> 12*5 + 30*6 = 240 triangles, closed sphere
        V, F, G = build_spiked('GOLDBERG', freq=2)
        cnt = edge_face_counts(F)
        closed = all(c == 2 for c in cnt.values())
        chi = len(V) - len(cnt) + len(F)
        finite = all(all(math.isfinite(c) for c in v) for v in V)
        print(f"spiked goldberg GP(2,0): V={len(V)} F={len(F)} "
              f"closed={closed} chi={chi} finite={finite}")
        assert (len(F) == 240 and closed and chi == 2 and finite
                and len(set(G)) == 42)

        # rhombic hexecontahedron: 60 planar golden rhombi
        V, F, G = build_modern(1.0, 1.0, quads=True)
        ok_planar = ok_golden = True
        for f in F:
            P = np.array([V[i] for i in f])
            n = np.cross(P[1] - P[0], P[2] - P[0])
            n /= np.linalg.norm(n)
            if abs(np.dot(P[3] - P[0], n)) > 1e-9:
                ok_planar = False
            d1 = np.linalg.norm(P[2] - P[0])
            d2 = np.linalg.norm(P[3] - P[1])
            ratio = max(d1, d2) / min(d1, d2)
            if abs(ratio - PHI) > 1e-9:
                ok_golden = False
        cnt = edge_face_counts(F)
        closed = all(c == 2 for c in cnt.values())
        chi = len(V) - len(cnt) + len(F)
        print(f"rhombic hexecontahedron: {len(F)} rhombi, "
              f"planar={ok_planar} golden={ok_golden} "
              f"closed={closed} chi={chi}")
        assert (len(F) == 60 and ok_planar and ok_golden
                and closed and chi == 2)

        # folded hexecontahedron: 120 triangles, closed, chi = 2
        V, F, G = build_modern(1.0, 1.05)
        cnt = edge_face_counts(F)
        closed = all(c == 2 for c in cnt.values())
        chi = len(V) - len(cnt) + len(F)
        print(f"folded hexecontahedron: {len(F)} triangles, "
              f"closed={closed} chi={chi}")
        assert len(F) == 120 and closed and chi == 2

        # hyperbolic solids: closed, chi = 2, spikes reach `spike`
        for sd in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
            V, F, G = build_hyper(sd, spike=1.5, res=6)
            cnt = edge_face_counts(F)
            closed = all(c == 2 for c in cnt.values())
            chi = len(V) - len(cnt) + len(F)
            rmax = max(np.linalg.norm(v) for v in V)
            print(f"hyperbolic {sd}: F={len(F)} closed={closed} "
                  f"chi={chi} rmax={rmax:.6f}")
            assert closed and chi == 2 and abs(rmax - 1.5) < 1e-6
        print("spikey standalone tests passed")
