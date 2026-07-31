
# Polyhedron Compounds for Blender
#
# Classic uniform polyhedron compounds built as the orbit of a seed solid
# under a rotation group: the stella octangula (two tetrahedra), the
# icosahedral compounds of five and ten tetrahedra, five cubes and five
# octahedra, and the dual pairs (cube + octahedron, dodecahedron +
# icosahedron).  The rotation groups are generated in the standard cube /
# dodecahedron frame so an axis-aligned seed lands on the compound's
# shared vertices; each component keeps its own colour.
#
# References:
# - Stella octangula: Johannes Kepler, "Harmonices Mundi" (1619).
# - The regular compounds (5 tetrahedra, 5 cubes, ...): Edmund Hess
#   (1876); Max Bruckner (1900); catalogued in H. S. M. Coxeter,
#   "Regular Polytopes" (1948).
# - Full enumeration of uniform compounds: John Skilling, "Uniform
#   compounds of uniform polyhedra", Math. Proc. Camb. Phil. Soc. 79
#   (1976).

bl_info = {
    "name": "Polyhedron Compounds",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Compounds of regular polyhedra (stella octangula, "
                   "5/10 tetrahedra, 5 cubes, dual pairs)",
    "category": "Add Mesh",
}

import math
import itertools

try:
    import numpy as np
except ImportError:
    np = None

PHI = (1 + 5 ** 0.5) / 2


# ---- seeds (V, F), axis-aligned in the standard frame --------------------

def _seeds():
    tetra = ([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
             [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    cube = ([(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)],
            [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
    octa = ([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)],
            [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
             [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return tetra, cube, octa


def _hull_faces(V):
    """Merged planar faces of the convex hull (small vertex sets only)."""
    n = len(V)
    planes = []
    seen = set()
    for a, b, c in itertools.combinations(range(n), 3):
        A, B, C = V[a], V[b], V[c]
        nx = ((B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1]),
              (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]),
              (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))
        ln = math.sqrt(sum(x * x for x in nx))
        if ln < 1e-9:
            continue
        nx = tuple(x / ln for x in nx)
        d = sum(nx[i] * A[i] for i in range(3))
        if d < 0:
            nx = tuple(-x for x in nx)
            d = -d
        key = tuple(round(x, 5) for x in nx) + (round(d, 5),)
        if key in seen:
            continue
        if all(sum(nx[i] * V[j][i] for i in range(3)) <= d + 1e-7
               for j in range(n)):
            seen.add(key)
            onv = [j for j in range(n)
                   if abs(sum(nx[i] * V[j][i] for i in range(3)) - d) < 1e-6]
            cen = [sum(V[j][i] for j in onv) / len(onv) for i in range(3)]
            ux = [V[onv[0]][i] - cen[i] for i in range(3)]
            uy = [nx[1] * ux[2] - nx[2] * ux[1],
                  nx[2] * ux[0] - nx[0] * ux[2],
                  nx[0] * ux[1] - nx[1] * ux[0]]

            def ang(j):
                dx = [V[j][i] - cen[i] for i in range(3)]
                return math.atan2(sum(dx[i] * uy[i] for i in range(3)),
                                  sum(dx[i] * ux[i] for i in range(3)))
            planes.append(sorted(onv, key=ang))
    return planes


def _dodeca():
    V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for a in (-1 / PHI, 1 / PHI):
        for b in (-PHI, PHI):
            V += [(0, a, b), (a, b, 0), (b, 0, a)]
    return V, _hull_faces(V)


def _icosa():
    V = []
    for a in (-1, 1):
        for b in (-PHI, PHI):
            V += [(0, a, b), (a, b, 0), (b, 0, a)]
    return V, _hull_faces(V)


# ---- rotation groups in the standard frame -------------------------------

def _octa_rotations():
    G = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = signs[i]
            if np.linalg.det(M) > 0:
                G.append(M)
    return G


def _rot(axis, ang):
    a = np.array(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _icosa_rotations():
    gens = [_rot((1, 1, 1), 2 * math.pi / 3),
            _rot((0, 1, PHI), 2 * math.pi / 5)]
    seen = {}
    def key(M):
        return tuple(np.round(M.ravel(), 5))
    ident = np.eye(3)
    seen[key(ident)] = ident
    frontier = [ident]
    while frontier:
        nf = []
        for M in frontier:
            for g in gens:
                N = g @ M
                k = key(N)
                if k not in seen:
                    seen[k] = N
                    nf.append(N)
        frontier = nf
        if len(seen) > 120:
            break
    return list(seen.values())


def _orbit(seed, G):
    """Distinct copies of the seed (V, F) under the rotation group."""
    V0, F0 = seed
    seen = set()
    out = []
    for M in G:
        W = [tuple(M @ np.array(v, float)) for v in V0]
        k = frozenset(tuple(np.round(w, 4)) for w in W)
        if k not in seen:
            seen.add(k)
            out.append((W, [list(f) for f in F0]))
    return out


COMPOUNDS = [
    ('STELLA', "Stella Octangula (2 Tetrahedra)"),
    ('5TETRA', "Compound of 5 Tetrahedra"),
    ('10TETRA', "Compound of 10 Tetrahedra"),
    ('5CUBES', "Compound of 5 Cubes"),
    ('5OCTA', "Compound of 5 Octahedra"),
    ('CUBE_OCTA', "Cube + Octahedron"),
    ('DODECA_ICOSA', "Dodecahedron + Icosahedron"),
]


def build_compound(kind):
    """Return a list of components, each (V, F)."""
    if np is None:
        raise RuntimeError("compounds need NumPy")
    tetra, cube, octa = _seeds()

    def scaled(seed, s):
        V, F = seed
        return ([tuple(s * c for c in v) for v in V], F)
    if kind == 'STELLA':
        return _orbit(tetra, _octa_rotations())
    if kind == '5TETRA':
        return _orbit(tetra, _icosa_rotations())
    if kind == '10TETRA':
        I = _icosa_rotations()
        mirr = ([(-v[0], v[1], v[2]) for v in tetra[0]],
                [list(reversed(f)) for f in tetra[1]])
        return _orbit(tetra, I) + _orbit(mirr, I)
    if kind == '5CUBES':
        return _orbit(cube, _icosa_rotations())
    if kind == '5OCTA':
        return _orbit(scaled(octa, math.sqrt(3)), _icosa_rotations())
    if kind == 'CUBE_OCTA':
        return [cube, scaled(octa, 1.5)]
    if kind == 'DODECA_ICOSA':
        return [_dodeca(), scaled(_icosa(), PHI)]
    raise ValueError(kind)


# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                (0.30, 0.69, 0.42), (0.95, 0.77, 0.29),
                (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                (0.80, 0.45, 0.30), (0.45, 0.55, 0.80)]

    def _mat(i):
        name = f"Compound {i}"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            rgb = _PALETTE[i % len(_PALETTE)]
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        return mat

    class MESH_OT_polyhedron_compound_add(bpy.types.Operator):
        """Add a compound of regular polyhedra (each component coloured
        separately)"""
        bl_idname = "mesh.polyhedron_compound_add"
        bl_label = "Polyhedron Compound"
        bl_options = {'REGISTER', 'UNDO'}

        compound: EnumProperty(
            name="Compound",
            items=[(k, lbl, "") for k, lbl in COMPOUNDS])
        separate: BoolProperty(
            name="Separate Objects", default=False,
            description="One object per component instead of a single "
                        "coloured mesh")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            try:
                comps = build_compound(self.compound)
            except Exception as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # normalise the whole compound to fit a 2 m cube
            mx = max(abs(c) for V, _F in comps for v in V for c in v) or 1.0
            s = self.scale / mx
            label = dict(COMPOUNDS)[self.compound]
            for o in context.selected_objects:
                o.select_set(False)
            first = None
            if self.separate:
                for i, (V, F) in enumerate(comps):
                    me = bpy.data.meshes.new(f"{label} {i + 1}")
                    me.from_pydata([tuple(c * s for c in v) for v in V],
                                   [], [tuple(f) for f in F])
                    me.materials.append(_mat(i))
                    me.update()
                    obj = bpy.data.objects.new(f"{label} {i + 1}", me)
                    context.collection.objects.link(obj)
                    obj.location = context.scene.cursor.location
                    obj.select_set(True)
                    first = first or obj
            else:
                verts = []
                faces = []
                fmat = []
                for i, (V, F) in enumerate(comps):
                    base = len(verts)
                    verts += [tuple(c * s for c in v) for v in V]
                    for f in F:
                        faces.append([base + j for j in f])
                        fmat.append(i)
                me = bpy.data.meshes.new(label)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                for i in range(len(comps)):
                    me.materials.append(_mat(i))
                if len(me.polygons) == len(fmat):
                    me.polygons.foreach_set('material_index', fmat)
                me.update()
                first = bpy.data.objects.new(label, me)
                context.collection.objects.link(first)
                first.location = context.scene.cursor.location
                first.select_set(True)
            context.view_layer.objects.active = first
            self.report({'INFO'}, f"{label}: {len(comps)} components")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.polyhedron_compound_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polyhedron_compound_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polyhedron_compound_add)


if __name__ == "__main__" and not _IN_BLENDER:
    for k, lbl in COMPOUNDS:
        comps = build_compound(k)
        nv = len(set(tuple(round(c, 4) for c in v)
                     for V, _F in comps for v in V))
        print(f"{k:14s} {len(comps):2d} components, {nv:3d} distinct verts"
              f"  ({lbl})")
