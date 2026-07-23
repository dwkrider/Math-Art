
# Space-Filling Solids generator for Blender
#
# Blocks of the classical space-filling honeycombs, every cell
# shrunk by a gap factor about its own centroid so the packing
# reads as a stack of separate solids -- after Henry Segerman,
# "Visualizing Mathematics with 3D Printing", figs 4-17..4-20.
#
#   CUBIC       cubes on the cubic lattice
#   OCTET       alternated cubic honeycomb: regular octahedra on
#               one FCC sublattice + regular tetrahedra (two
#               orientations) filling the gaps, ratio 1:2 in bulk
#   TRUNCOCT    bitruncated cubic: truncated octahedra on BCC
#   RHOMBDODEC  rhombic dodecahedra on FCC
#
# At gap = 1.0 adjacent cells share their faces exactly (the
# lattices below are in integer canonical coordinates, so shared
# faces coincide to the bit); the default gap < 1 gives the
# printed-separately look.

bl_info = {
    "name": "Space-Filling Solids",
    "author": "David Krider (Math Art project, after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Space-Filling Solids",
    "description": "Honeycomb blocks: cubes, octet truss, "
                   "truncated octahedra, rhombic dodecahedra",
    "category": "Add Mesh",
}

import itertools

import numpy as np


def _plane_faces(verts, planes):
    """One convex face per plane (normal, offset): the vertices
    with v . normal == offset, wound counterclockwise seen from
    outside (all plane normals point away from the cell centre)."""
    V = np.asarray(verts, float)
    faces = []
    for nrm, off in planes:
        n = np.asarray(nrm, float)
        sel = np.where(np.abs(V @ n - off) < 1e-9)[0]
        nn = n / np.linalg.norm(n)
        a = (np.array((1.0, 0.0, 0.0)) if abs(nn[0]) < 0.9
             else np.array((0.0, 1.0, 0.0)))
        u = np.cross(nn, a)
        u /= np.linalg.norm(u)
        v = np.cross(nn, u)            # (u, v, nn) right-handed
        d = V[sel] - V[sel].mean(axis=0)
        ang = np.arctan2(d @ v, d @ u)
        faces.append([int(i) for i in sel[np.argsort(ang)]])
    return faces


def _tet_faces(V):
    """Outward-wound faces of the tetrahedron with vertices V."""
    c = V.mean(axis=0)
    faces = []
    for tri in itertools.combinations(range(4), 3):
        i, j, k = tri
        n = np.cross(V[j] - V[i], V[k] - V[i])
        if np.dot(n, V[i] - c) < 0:
            tri = (i, k, j)
        faces.append(list(tri))
    return faces


def _axis_planes(off):
    return [((s, 0.0, 0.0), off) for s in (-1.0, 1.0)] + \
           [((0.0, s, 0.0), off) for s in (-1.0, 1.0)] + \
           [((0.0, 0.0, s), off) for s in (-1.0, 1.0)]


def _diag_planes(off):
    return [(s, off)
            for s in itertools.product((-1.0, 1.0), repeat=3)]


# unit cube about its centre
_CUBE_V = np.array([(x - .5, y - .5, z - .5)
                    for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                   float)
_CUBE_F = _plane_faces(_CUBE_V, _axis_planes(0.5))

# regular octahedron, vertices on the axes (edge sqrt2)
_OCT_V = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                   (0, 0, 1), (0, 0, -1)], float)
_OCT_F = _plane_faces(_OCT_V, _diag_planes(1.0))

# octet tetrahedra: the even-parity corners of a unit cube, about
# the cube centre; odd cube parity gives the mirror orientation
_TET_V = (np.array([(-.5, -.5, -.5), (.5, .5, -.5),
                    (.5, -.5, .5), (-.5, .5, .5)]),)
_TET_V = _TET_V + (-_TET_V[0],)
_TET_F = (_tet_faces(_TET_V[0]), _tet_faces(_TET_V[1]))

# truncated octahedron: all permutations of (0, +-1, +-2); the
# BCC Voronoi cell, 6 squares + 8 hexagons, width 4 along an axis
_TO_V = np.array(sorted({p for s1 in (-1, 1) for s2 in (-2, 2)
                         for p in itertools.permutations(
                             (0, s1, s2))}), float)
_TO_F = _plane_faces(_TO_V, _axis_planes(2.0) + _diag_planes(3.0))

# rhombic dodecahedron: the FCC Voronoi cell, 12 rhombi normal to
# the (1,1,0)-type neighbour directions, width 2 along an axis
_RD_V = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)]
                 + [(x * .5, y * .5, z * .5)
                    for x in (-1, 1) for y in (-1, 1)
                    for z in (-1, 1)], float)
_RD_P = ([((a, b, 0.0), 1.0)
          for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
         + [((a, 0.0, b), 1.0)
            for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
         + [((0.0, a, b), 1.0)
            for a in (-1.0, 1.0) for b in (-1.0, 1.0)])
_RD_F = _plane_faces(_RD_V, _RD_P)


def build_block(kind, nx, ny, nz):
    """Cells of the block in canonical lattice coordinates.

    Returns (cells, pitch): cells is a list of tuples
    (centroid, local_verts, faces, tag) with local_verts about the
    centroid, tag 0 for the primary cell (octahedra for OCTET) and
    1 for the secondary (tetrahedra); pitch is the canonical length
    the Cell Size property maps to (one lattice step per axis)."""
    P = itertools.product
    cells = []
    if kind == 'CUBIC':
        for i, j, k in P(range(nx), range(ny), range(nz)):
            cells.append((np.array((i + .5, j + .5, k + .5)),
                          _CUBE_V, _CUBE_F, 0))
        return cells, 1.0
    if kind == 'OCTET':
        # one tetrahedron per unit cube of an nx x ny x nz box
        # (orientation by cube parity); octahedra at the odd-parity
        # integer points strictly inside, so the block stays boxy
        for i, j, k in P(range(nx), range(ny), range(nz)):
            p = (i + j + k) % 2
            cells.append((np.array((i + .5, j + .5, k + .5)),
                          _TET_V[p], _TET_F[p], 1))
        for x, y, z in P(range(1, nx), range(1, ny), range(1, nz)):
            if (x + y + z) % 2:
                cells.append((np.array((x, y, z), float),
                              _OCT_V, _OCT_F, 0))
        return cells, 1.0
    if kind == 'TRUNCOCT':
        # BCC: a primary grid plus a second grid at the cube centres
        for i, j, k in P(range(nx), range(ny), range(nz)):
            cells.append((4.0 * np.array((i, j, k)),
                          _TO_V, _TO_F, 0))
        for i, j, k in P(range(nx - 1), range(ny - 1),
                         range(nz - 1)):
            cells.append((4.0 * np.array((i, j, k)) + 2.0,
                          _TO_V, _TO_F, 0))
        return cells, 4.0
    if kind == 'RHOMBDODEC':
        # FCC: even-parity integer points of a box, pitch 2
        for x, y, z in P(range(2 * nx - 1), range(2 * ny - 1),
                         range(2 * nz - 1)):
            if (x + y + z) % 2 == 0:
                cells.append((np.array((x, y, z), float),
                              _RD_V, _RD_F, 0))
        return cells, 2.0
    raise ValueError(kind)


def build_mesh(kind='OCTET', nx=3, ny=3, nz=2, gap=0.92, size=1.0):
    """Whole block as (verts, faces, face_tags), centred at the
    origin; gap scales every cell about its own centroid and size
    scales one lattice step to that length."""
    cells, pitch = build_block(kind, nx, ny, nz)
    s = size / pitch
    lo = np.full(3, np.inf)
    hi = -lo
    for c, V, F, t in cells:
        lo = np.minimum(lo, c + V.min(axis=0))
        hi = np.maximum(hi, c + V.max(axis=0))
    mid = (lo + hi) / 2.0
    verts, faces, tags = [], [], []
    for c, V, F, t in cells:
        base = len(verts)
        verts.extend(map(tuple, (c - mid + gap * V) * s))
        faces.extend([base + i for i in f] for f in F)
        tags.extend([t] * len(F))
    return verts, faces, tags


_CELL_VOL = {'CUBIC': {0: 1.0},               # per canonical cell
             'OCTET': {0: 4.0 / 3.0, 1: 1.0 / 3.0},
             'TRUNCOCT': {0: 32.0},
             'RHOMBDODEC': {0: 2.0}}


def block_volume(kind, nx, ny, nz, size=1.0):
    """Analytic total volume of all cells at gap = 1."""
    cells, pitch = build_block(kind, nx, ny, nz)
    s3 = (size / pitch) ** 3
    return sum(_CELL_VOL[kind][t] for *_, t in cells) * s3


def _mesh_volume(verts, faces):
    """Signed volume of a from_pydata-style mesh (outward faces)."""
    V = np.asarray(verts, float)
    tot = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):
            tot += np.linalg.det(V[[f[0], f[i], f[i + 1]]])
    return tot / 6.0


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _LABEL = {'CUBIC': "Cubes", 'OCTET': "Octet",
              'TRUNCOCT': "Truncated Octahedra",
              'RHOMBDODEC': "Rhombic Dodecahedra"}

    def _material(name, rgb):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.5
        return mat

    class MESH_OT_spacefill_add(bpy.types.Operator):
        """Add a block of a space-filling honeycomb, each cell
        shrunk about its centroid so the packing reads as a stack
        of separate solids (after Segerman)"""
        bl_idname = "mesh.spacefill_add"
        bl_label = "Space-Filling Solids"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Honeycomb",
            items=[('CUBIC', "Cubes",
                    "Cubes on the cubic lattice"),
                   ('OCTET', "Octahedra + Tetrahedra",
                    "Alternated cubic honeycomb: regular octahedra "
                    "and tetrahedra (the octet truss)"),
                   ('TRUNCOCT', "Truncated Octahedra",
                    "Bitruncated cubic honeycomb on the BCC "
                    "lattice"),
                   ('RHOMBDODEC', "Rhombic Dodecahedra",
                    "Rhombic dodecahedra on the FCC lattice")],
            default='OCTET')
        nx: IntProperty(name="Cells X", default=3, min=1, max=12,
                        description="Lattice cells along X")
        ny: IntProperty(name="Cells Y", default=3, min=1, max=12,
                        description="Lattice cells along Y")
        nz: IntProperty(name="Cells Z", default=2, min=1, max=12,
                        description="Lattice cells along Z")
        gap: FloatProperty(
            name="Gap Factor", default=0.92, min=0.05, max=1.0,
            description="Scale of each cell about its own centroid "
                        "(1.0 = cells share faces exactly)")
        size: FloatProperty(
            name="Cell Size", default=1.0, min=0.01, max=100.0,
            description="Length of one lattice step per axis")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain solid cells"),
                   ('LEONARDO', "Leonardo",
                    "Open-faced da Vinci panels (Geometry Nodes "
                    "modifier on the whole block)")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width (fraction of "
                        "the face)")
        thickness: FloatProperty(
            name="Thickness", default=0.06, min=0.001, max=1.0,
            description="Leonardo panel thickness")
        two_materials: BoolProperty(
            name="Two Materials", default=True,
            description="Distinct materials for octahedra and "
                        "tetrahedra (Octet only; the other "
                        "honeycombs have no honest 2-colouring)")

        def execute(self, context):
            verts, faces, tags = build_mesh(self.kind, self.nx,
                                            self.ny, self.nz,
                                            self.gap, self.size)
            me = bpy.data.meshes.new("Spacefill")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if (self.kind == 'OCTET' and self.two_materials
                    and len(me.polygons) == len(tags)):
                me.materials.append(_material(
                    "Spacefill Octahedron", (0.9, 0.45, 0.12)))
                me.materials.append(_material(
                    "Spacefill Tetrahedron", (0.15, 0.35, 0.8)))
                me.polygons.foreach_set('material_index', tags)
            me.update()
            obj = bpy.data.objects.new(
                f"Spacefill {_LABEL[self.kind]}", me)
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
            self.report({'INFO'},
                        f"{_LABEL[self.kind]}: "
                        f"V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'nx', 'ny', 'nz', 'gap', 'size'):
                lay.prop(self, k)
            if self.kind == 'OCTET':
                lay.prop(self, 'two_materials')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
                lay.prop(self, 'thickness')

    def _menu_func(self, context):
        self.layout.operator("mesh.spacefill_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_spacefill_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_spacefill_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # at gap = 1 the numeric mesh volume must equal the
        # analytic total cell volume for every honeycomb
        for kind in ('CUBIC', 'OCTET', 'TRUNCOCT', 'RHOMBDODEC'):
            v, f, t = build_mesh(kind, 3, 3, 2, gap=1.0)
            num = _mesh_volume(v, f)
            ana = block_volume(kind, 3, 3, 2)
            ok = abs(num - ana) < 1e-9 * max(1.0, ana)
            print(f"{kind}: vol={num:.6f} ({ana:.6f}) "
                  f"{'OK' if ok else 'BAD'}")
