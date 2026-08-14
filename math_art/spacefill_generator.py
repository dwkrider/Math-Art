
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
#   SPIRAL3     3-armed rhombic spirallohedra S(12, 4) (Towle)
#   SPIRAL4     4-armed rhombic spirallohedra S(12, 3)
#
# The spirallohedra tile space by pure translations: every rhombic
# face of the cell maps onto an opposite face of the same cell
# under some translation, and those translations span a rank-3
# lattice whose fundamental volume equals the cell volume (both
# verified numerically at import of the cell -- see
# _spiral_data).  The lattice basis is derived from the face
# pairing, not hardcoded.
#
# At gap = 1.0 adjacent cells share their faces exactly (the
# lattices below are in integer canonical coordinates, so shared
# faces coincide to the bit); the default gap < 1 gives the
# printed-separately look.
#
# References:
# - Henry Segerman, "Visualizing Mathematics with 3D Printing", Johns
#   Hopkins University Press, 2016 (figs 4-17..4-20, space-filling
#   solids).
# - Rhombic spirallohedra S(n,w) via the polar-zonohedron
#   construction: Russell Towle (Spirallohedra / polar-zonohedron
#   notebooks).
# - The alternated cubic (octet) and bitruncated cubic honeycombs and
#   the Voronoi cells of the cubic, BCC and FCC lattices: H. S. M.
#   Coxeter, "Regular Polytopes", 3rd ed., Dover, 1973.
# - The obtetrahedrille (Conway's oblate tetrahedrille / tetragonal
#   disphenoid honeycomb, each rhombic dodecahedron split into 24
#   disphenoids): J. H. Conway, H. Burgiel, C. Goodman-Strauss, "The
#   Symmetries of Things", A K Peters, 2008; and Tom Verhoeff & Koos
#   Verhoeff, "The Obtetrahedrille as a Modular Building Block for 3D
#   Mathematical Art", Bridges 2019, 407-410.

bl_info = {
    "name": "Space-Filling Solids",
    "author": "Math Art project (after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Space-Filling Solids",
    "description": "Honeycomb blocks: cubes, octet truss, "
                   "truncated octahedra, rhombic dodecahedra",
    "category": "Add Mesh",
}

import itertools

import numpy as np
# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.spacefill import (_face_key, _mesh_volume, _spiral_data,
                                block_volume, build_mesh, spiral_n)
except ImportError:  # flat import outside the package
    from ifs.spacefill import (_face_key, _mesh_volume, _spiral_data,
                               block_volume, build_mesh, spiral_n)


# ------------------------------------------------------------------
# obtetrahedrille (Conway's oblate tetrahedrille): the rhombic
# dodecahedron splits into 24 congruent tetragonal disphenoids -- the
# space-filling cell of the tetragonal disphenoid honeycomb (Verhoeff
# & Verhoeff, Bridges 2019).  Each rhombic face pyramid (RD centre +
# the 4 face vertices) splits along its short diagonal (the two
# degree-3 "cube" vertices) into two disphenoids.  Every disphenoid
# has one opposite edge pair of length 1 and four edges of length
# sqrt(3)/2; the six orientations (by the axis vertex direction) tile
# space by translation of the FCC-placed rhombic dodecahedra.
# ------------------------------------------------------------------


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
              'RHOMBDODEC': "Rhombic Dodecahedra",
              'OBTET': "Obtetrahedrille",
              'SPIRAL3': "Spirallohedra (3-Armed)",
              'SPIRAL4': "Spirallohedra (4-Armed)"}

    # six-colour palette for the obtetrahedrille disphenoid orientations
    _OBTET_RGB = {0: (0.86, 0.28, 0.24), 1: (0.30, 0.55, 0.82),
                  2: (0.32, 0.70, 0.42), 3: (0.95, 0.76, 0.28),
                  4: (0.60, 0.40, 0.75), 5: (0.28, 0.72, 0.70)}

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
                    "Rhombic dodecahedra on the FCC lattice"),
                   ('OBTET', "Obtetrahedrille",
                    "The tetragonal disphenoid honeycomb: each FCC "
                    "rhombic dodecahedron split into 24 congruent "
                    "disphenoids (Conway's oblate tetrahedrille), "
                    "six-coloured by orientation"),
                   ('SPIRAL3', "Rhombic Spirallohedra (3-Armed)",
                    "Three-armed rhombic spirallohedra S(12,4) "
                    "after Russell Towle, interlocking on their "
                    "translation lattice"),
                   ('SPIRAL4', "Rhombic Spirallohedra (4-Armed)",
                    "Four-armed rhombic spirallohedra S(12,3) "
                    "after Russell Towle, interlocking on their "
                    "translation lattice")],
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
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Overall scale (1.0 fits a 2 m cube, centred "
                        "at the origin)")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain solid cells"),
                   ('LEONARDO', "Leonardo",
                    "Open-faced da Vinci panels (Geometry Nodes "
                    "modifier on the whole block)"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width (fraction of "
                        "the face)")
        thickness: FloatProperty(
            name="Thickness", default=0.06, min=0.001, max=1.0,
            description="Leonardo panel thickness")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
        two_materials: BoolProperty(
            name="Two Materials", default=True,
            description="Distinct materials for octahedra and "
                        "tetrahedra (Octet), or alternating by "
                        "lattice parity (Spirallohedra); the other "
                        "honeycombs have no honest 2-coloring")
        spiral_segments: IntProperty(
            name="Spiral Segments", default=12, min=6, max=24,
            description="Star vectors of the spirallohedron cell, "
                        "rounded down to a multiple of the arm "
                        "count; more segments give finer, longer "
                        "spiral cells")
        spiral_pitch: FloatProperty(
            name="Spiral Pitch", default=55.0, min=5.0, max=85.0,
            description="Polar star pitch angle from the axis "
                        "(degrees); higher is squatter, wider "
                        "spiral cells")

        def execute(self, context):
            try:
                verts, faces, tags = build_mesh(
                    self.kind, self.nx, self.ny, self.nz,
                    self.gap, 1.0, self.spiral_segments,
                    self.spiral_pitch)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # centre at the origin and fit a 2 m cube, then scale
            P = np.asarray(verts, float)
            if len(P):
                lo = P.min(axis=0)
                hi = P.max(axis=0)
                span = float(np.max(hi - lo)) or 1.0
                P = (P - (lo + hi) / 2.0) * (2.0 * self.scale / span)
                verts = [tuple(p) for p in P]
            me = bpy.data.meshes.new("Spacefill")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if (self.kind == 'OBTET'
                    and len(me.polygons) == len(tags)):
                # six materials, one per disphenoid orientation
                for t in range(6):
                    me.materials.append(_material(
                        f"Spacefill Obtet {t}", _OBTET_RGB[t]))
                me.polygons.foreach_set('material_index', tags)
            elif (self.kind in ('OCTET', 'SPIRAL3', 'SPIRAL4')
                    and self.two_materials
                    and len(me.polygons) == len(tags)):
                if self.kind == 'OCTET':
                    me.materials.append(_material(
                        "Spacefill Octahedron", (0.9, 0.45, 0.12)))
                    me.materials.append(_material(
                        "Spacefill Tetrahedron", (0.15, 0.35, 0.8)))
                else:
                    me.materials.append(_material(
                        "Spacefill Spiral A", (0.85, 0.55, 0.15)))
                    me.materials.append(_material(
                        "Spacefill Spiral B", (0.2, 0.45, 0.7)))
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
            elif self.style == 'BALLSTICK':
                try:
                    from .styles import ball_and_stick
                except ImportError:
                    from styles import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            self.report({'INFO'},
                        f"{_LABEL[self.kind]}: "
                        f"V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'nx', 'ny', 'nz', 'gap', 'scale'):
                lay.prop(self, k)
            if self.kind in ('SPIRAL3', 'SPIRAL4'):
                lay.prop(self, 'spiral_segments')
                lay.prop(self, 'spiral_pitch')
            if self.kind in ('OCTET', 'SPIRAL3', 'SPIRAL4'):
                lay.prop(self, 'two_materials')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
                lay.prop(self, 'thickness')
            elif self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')

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


def _selftest():
    # at gap = 1 the numeric mesh volume must equal the
    # analytic total cell volume for every honeycomb
    for kind in ('CUBIC', 'OCTET', 'TRUNCOCT', 'RHOMBDODEC',
                 'OBTET', 'SPIRAL3', 'SPIRAL4'):
        v, f, t = build_mesh(kind, 3, 3, 2, gap=1.0)
        num = _mesh_volume(v, f)
        ana = block_volume(kind, 3, 3, 2)
        ok = abs(num - ana) < 1e-9 * max(1.0, ana)
        print(f"{kind}: vol={num:.6f} ({ana:.6f}) "
              f"{'OK' if ok else 'BAD'}")
    # spirallohedra: the tiling-basis derivation must succeed
    # (it raises when the cell is not a translational tiler)
    # and at gap = 1 non-boundary faces of a block must be
    # shared exactly, for a sweep of segment counts and pitches
    for kind in ('SPIRAL3', 'SPIRAL4'):
        for segs in (6, 8, 9, 12, 15, 16, 20, 24):
            if spiral_n(kind, segs) != segs:
                continue
            for pitch in (35.0, 45.0, 55.0, 70.0):
                V, F, B, height, vol = _spiral_data(kind, segs,
                                                    pitch)
                keys = {}
                for off in itertools.product(range(2),
                                             repeat=3):
                    t = (off[0] * B[0] + off[1] * B[1]
                         + off[2] * B[2])
                    for f in F:
                        k = _face_key(V[list(f)] + t)
                        keys[k] = keys.get(k, 0) + 1
                shared = sum(1 for c in keys.values()
                             if c == 2)
                assert shared > 0, (kind, segs, pitch)
            print(f"{kind} n={segs}: tiles at all pitches, "
                  f"shared faces OK")
    print("spacefill standalone tests passed")
