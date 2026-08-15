
# Topological Interlocking & Modular Blocks generator for Blender
#
# Assemblies of rigid blocks that, once a peripheral "frame" is held
# fixed, kinematically lock: no non-empty subset of the interior
# blocks can be slid or rotated out without collision.  The blocks
# are never glued -- interlocking is purely geometric.
#
# Families implemented here (all verified non-overlapping; the
# space-filling ones fill with coverage exactly one):
#
#   TETRA        the canonical interlocking layer of regular tetrahedra
#                (two 90-deg orientations on a checkerboard), after
#                Dyskin, Estrin, Kanel-Belov & Pasternak (2001).
#   MCSCUBE /    cubes / octahedra with a 3-fold (body-diagonal) axis
#   MCSOCTA      vertical, placed as identical translates on a
#                honeycomb; each cell's six faces tilt +/-alpha and
#                adjacent cells share every inclined plane -- the
#                Kanel-Belov / Dyskin moving-cross-section layer.
#   ESCHER       the Escher-trick / osteomorphic family: a square (p4)
#                fundamental domain is edge-deformed, rotated 90 deg at
#                the mid-plane and lofted, so every section tiles.  A
#                sine profile gives Estrin's osteomorphic block, a tent
#                profile a Versatile-style block (Truchet colouring).
#   VERSATILE    the exact Versatile block tiled over the diamond
#                lattice (Akpanya et al., Bridges 2023).
#   BISQUARE     the exact Bisquare block (Frezier 1737); 1x1 a single
#                block, larger a p4 interlocking layer (Weiss &
#                Niemeyer 2026).
#   RHOM /       the exact Rhom block and its obverse, single p3
#   RHOM_OBV     reference blocks (Weiss & Niemeyer 2026).
#   KITTEN /     tetroctahedrille blocks: the kitten tiles by
#   UFO /        translation; UFO and cushion are single reference
#   CUSHION      blocks (Akpanya, Goertzen & Niemeyer 2024).
#   SL           self-interlocking octocubes with the six-engagement
#                grammar and the periodic square strand (Shih 2018).
#   DOME         one radially-lofted block per polyhedron face, the
#                spherical Escher loft (Akpanya et al. 2024).
#   HENDECA      the bisymmetric hendecahedron space-filler: a 4-cell
#                "boat" translated over a body-centred-tetragonal
#                lattice (Inchbald 1996; Wu & Inchbald 2018).
#
# References:
# - A. V. Dyskin, Y. Estrin, A. J. Kanel-Belov, E. Pasternak, "A new
#   concept in design of materials and structures: assemblies of
#   interlocked tetrahedron-shaped elements", Scripta Materialia 44
#   (2001) 2689-2694.
# - A. J. Kanel-Belov, A. V. Dyskin, Y. Estrin, E. Pasternak,
#   I. A. Ivanov-Pogodaev, "Interlocking of convex polyhedra: towards
#   a geometric theory of fragmented solids", Moscow Math. J. 10(2)
#   (2010) 337-342 (arXiv:0812.5089).
# - Y. Estrin, A. V. Dyskin, E. Pasternak et al., "Fracture resistant
#   structures based on topological interlocking with non-planar
#   contacts", Adv. Eng. Materials 5(3) (2003) 116-119 (osteomorphic
#   blocks).
# - R. Akpanya, T. Goertzen, S. Wiesenhuetter, A. C. Niemeyer,
#   J. Noennig, "Topological Interlocking, Truchet Tiles and
#   Self-Assemblies", Bridges 2023, 61-68 (Versatile block).
# - R. Akpanya, T. Goertzen, A. C. Niemeyer, "From Tilings of
#   Orientable Surfaces to Topological Interlocking Assemblies",
#   Applied Sciences 14(16):7276 (2024) (the Escher-loft framework and
#   the interlocking dome).
# - R. Akpanya, T. Goertzen, A. C. Niemeyer, "Topologically
#   Interlocking Blocks inside the Tetroctahedrille", arXiv:2405.01944
#   (2024) (kitten / UFO / cushion).
# - M. Weiss, A. C. Niemeyer, "Construction Methods for Space-Filling
#   Heterogeneous Topological Interlocking Assemblies", arXiv:2604.22475
#   (2026) (Bisquare and Rhom blocks; A.-F. Frezier, 1737, for the
#   original Bisquare block).
# - Shen-Guan Shih, "The Art and Mathematics of Self-Interlocking SL
#   Blocks", Bridges 2018, 107-114.
# - G. Inchbald, "Five Space-filling Polyhedra", The Mathematical
#   Gazette 80 (489) (1996) 466-475; J. Wu & G. Inchbald, "Folding the
#   Space-Filling Bisymmetric Hendecahedron for a Large-Scale Art
#   Installation", Bridges 2018, 483-486 (hendecahedron).

bl_info = {
    "name": "Topological Interlocking Blocks",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Topological Interlocking",
    "description": "Interlocking block assemblies: tetrahedra "
                   "layers, Escher/osteomorphic blocks, "
                   "tetroctahedrille and SL blocks",
    "category": "Add Mesh",
}
import math
import numpy as np

# The mathematics lives in the sibling `interlocking` engine package;
# this module is the Blender layer over it.
try:
    from .interlocking.blocks import (_HENDECA_F, _HENDECA_L,
                                          _HENDECA_MOTIF, _HENDECA_V, _OCT,
                                          _RIGOROUS, _T1, _T2, _TETRA_A,
                                          _TETRA_B, _TETRA_FA, _TETRA_FB,
                                          _VERSATILE_F, _VERSATILE_V,
                                          _convex_faces, _convex_overlap,
                                          _deformed_square, _mcs_cell,
                                          _mesh_volume, _orient_outward,
                                          _rot2, _sl_frames, _sl_octocube,
                                          build_bisquare, build_cells,
                                          build_dome, build_escher,
                                          build_hendeca, build_mcs,
                                          build_rhom, build_sl, build_tetra,
                                          build_tetrocta, build_versatile,
                                          cells_to_mesh, cells_to_meshes)
except ImportError:  # flat import outside the package
    from interlocking.blocks import (_HENDECA_F, _HENDECA_L,
                                         _HENDECA_MOTIF, _HENDECA_V, _OCT,
                                         _RIGOROUS, _T1, _T2, _TETRA_A,
                                         _TETRA_B, _TETRA_FA, _TETRA_FB,
                                         _VERSATILE_F, _VERSATILE_V,
                                         _convex_faces, _convex_overlap,
                                         _deformed_square, _mcs_cell,
                                         _mesh_volume, _orient_outward,
                                         _rot2, _sl_frames, _sl_octocube,
                                         build_bisquare, build_cells,
                                         build_dome, build_escher,
                                         build_hendeca, build_mcs,
                                         build_rhom, build_sl, build_tetra,
                                         build_tetrocta, build_versatile,
                                         cells_to_mesh, cells_to_meshes)






# ==================================================================
# generic mesh helpers
# ==================================================================











# ==================================================================
# FAMILY: TETRA -- interlocking regular-tetrahedron layer
# ==================================================================
#
# A regular tetrahedron with two opposite edges mutually
# perpendicular and horizontal: the top edge (z = +H) runs along one
# axis, the bottom edge (z = -H) along the perpendicular axis.  Its
# horizontal mid-section is a unit square.  Placing type A (top edge
# along x) on one colour of the checkerboard and type B = A turned 90
# deg on the other colour makes every cell's square mid-section tile
# the plane; the x-neighbours block a cell from rising and the
# y-neighbours block it from sinking, so the framed interior locks.






# ==================================================================
# FAMILY: MCS -- moving-cross-section cubes & octahedra
# ==================================================================
#
# Erect tilted planes over the six edges of a regular hexagon (the
# common middle section) and intersect their half-spaces.  Around the
# hexagon the tilt sense alternates (+ - + - + -); at the tilt angle
# beta = asin(1/sqrt3) the six planes close into a cube, at
# beta = asin(1/3) (plus horizontal caps) into an octahedron.  The
# hexagon is the section of the solid normal to a 3-fold axis.
#
# Neighbouring hexagons share an edge, and a shared edge must carry
# opposite tilt senses for the two cells (outward for one is inward
# for the other); this yields exactly two cell types, placed on the
# two sublattices of the honeycomb's brick colouring.



















# ==================================================================
# FAMILY: ESCHER -- Escher-trick / osteomorphic / Versatile loft
# ==================================================================
#
# Take the unit square [0,1]^2 (a p4 fundamental domain).  Deform its
# four edges by a single profile (the Escher trick: the four edges
# are 90-deg images of one another, so one profile fixes all four).
# Put the deformed square M at z=0, its 90-deg rotation M' at z=1/2,
# and M again at z=1, and loft.  Every horizontal section is a
# deformed-square tiling, so copies on the integer lattice fill the
# slab; checkerboard parity flips the mid rotation sense and gives the
# two Truchet colours.  A sine profile is Estrin's osteomorphic
# block; a tent profile is a Versatile-style block.













# ==================================================================
# small convex hull (faces of a convex point set, outward wound)
# ==================================================================



# ==================================================================
# FAMILY: VERSATILE / BISQUARE -- exact Escher-trick literature blocks
# ==================================================================
#
# The Versatile Block (Akpanya, Goertzen, Wiesenhuetter, Niemeyer,
# Noennig, Bridges 2023): the interpolation between a square (as a
# diamond at z=0) and a 1x2 rectangle (at z=1), both p4 fundamental
# domains.  Copies tile the slab and interlock.  Exact vertices and
# triangulation as published.




















# ==================================================================
# FAMILY: TETROCTA -- interlocking blocks in the tetroctahedrille
# ==================================================================
#
# Regular tetrahedra and octahedra of edge sqrt2 glued on the FCC
# lattice (the tetrahedral-octahedral honeycomb).  Kitten = 1 oct + 2
# tets, UFO = 1 oct + 4 tets, cushion = 1 oct + 4 tets in a strip.
# Every vertex is integer, so blocks tile space by lattice
# translation.  After Akpanya, Goertzen & Niemeyer (arXiv:2405.01944).









# ==================================================================
# FAMILY: SL -- self-interlocking octocube blocks (Shih 2018)
# ==================================================================
#
# An S-shaped tetracube fused to an L-shaped tetracube gives the SL
# octocube: eight unit cubes spanning three z-levels, the S and L
# tetracubes sharing three faces so their union is one contiguous
# solid.  Blocks engage by one of six rigid "engagements" a,h,s,t,d,y
# (world-frame affine maps p -> R p + t); a strand is a word over
# those letters and is a periodic (self-locking) loop iff its
# transform product is the identity.  Frames compose by post-
# multiplication (G <- G . M_e) with exactly one octocube placed per
# letter.  Simplest loop: a4 (the square strand); general square
# frames are (a h^{2n})4.  All verified cube-disjoint.
#
# References: Shen-Guan Shih, "The Art and Mathematics of Self-
# Interlocking SL Blocks," Bridges 2018, pp. 107-114; and "On the
# Hierarchical Construction of SL Blocks," Advances in Architectural
# Geometry 2016.




















# ==================================================================
# FAMILY: DOME -- interlocking spherical shell (Escher on a sphere)
# ==================================================================
#
# One block per face of a seed polyhedron, lofted radially from an
# inner shell to an outer shell.  Every shared edge is displaced
# tangentially at the middle shell by a single globally-consistent
# vector, so the two blocks meeting at an edge share the same wavy
# radial wall -- the spherical analogue of the planar Escher loft.
# The blocks tile the shell with no gaps or overlaps (verified) and
# interlock through the deformed walls.  After Akpanya, Goertzen &
# Niemeyer, "From Tilings of Orientable Surfaces to TIA" (2024).











# ==================================================================
# FAMILY: HENDECA -- bisymmetric hendecahedron space-filler
# ==================================================================
#
# The bisymmetric hendecahedron (Inchbald 1996): a space-filling
# 11-hedron with 2 large rhombi, 1 small rhombus, 4 isosceles
# triangles and 4 kites (7 quads + 4 triangles), volume 16 in these
# coordinates.  It is NOT a single-cell translation tiler and NOT a
# reflection tiler.  Four cells make a "hexagonal boat" that stacks by
# pure translation on a body-centred-tetragonal (BCC-scaled-
# vertically) lattice -- Inchbald's construction, here given exactly
# as a four-cell motif plus three lattice vectors (see _HENDECA_MOTIF
# and _HENDECA_L below).  The tiling is verified to fill space with no
# gaps, no overlaps and face-to-face contact everywhere.
#
# References: Guy Inchbald, "Five Space-filling Polyhedra," The
# Mathematical Gazette 80 (489), 1996, pp. 466-475; Jiangmei Wu &
# Guy Inchbald, "Folding the Space-Filling Bisymmetric
# Hendecahedron," Bridges 2018, pp. 483-486.











# ==================================================================
# assembly -> mesh
# ==================================================================





# ==================================================================
# family dispatch
# ==================================================================





# ==================================================================
# Blender operator
# ==================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _LABEL = {'TETRA': "Interlocking Tetrahedra",
              'ESCHER': "Escher / Osteomorphic Blocks",
              'VERSATILE': "Versatile Block",
              'MCSCUBE': "Interlocking Cubes",
              'MCSOCTA': "Interlocking Octahedra",
              'BISQUARE': "Bisquare Blocks",
              'RHOM': "Rhom Block",
              'RHOM_OBV': "Rhom Block (Obverse)",
              'KITTEN': "Tetroctahedrille Kitten",
              'UFO': "Tetroctahedrille UFO",
              'CUSHION': "Tetroctahedrille Cushion",
              'SL': "SL Block",
              'DOME': "Interlocking Dome",
              'HENDECA': "Bisymmetric Hendecahedron"}

    _COLOURS = {0: (0.85, 0.55, 0.15), 1: (0.20, 0.45, 0.70),
                2: (0.55, 0.65, 0.25)}
    _FRAME_RGB = (0.72, 0.10, 0.12)

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

    class MESH_OT_interlocking_add(bpy.types.Operator):
        """Add a topological-interlocking block assembly: a layer or
        shell of blocks that lock once the peripheral frame is fixed"""
        bl_idname = "mesh.interlocking_add"
        bl_label = "Topological Interlocking"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Family",
            items=[
                ('TETRA', "Interlocking Tetrahedra",
                 "The canonical layer of regular tetrahedra in two "
                 "90-deg orientations on a checkerboard (Dyskin et "
                 "al.); the framed interior kinematically locks"),
                ('ESCHER', "Escher / Osteomorphic",
                 "A square domain edge-deformed, turned 90 deg at "
                 "mid-height and lofted back; sine gives Estrin's "
                 "osteomorphic saddle block, tent a Versatile-style "
                 "block; every layer tiles so copies interlock"),
                ('VERSATILE', "Versatile Blocks",
                 "The exact Versatile block (Akpanya et al., Bridges "
                 "2023): a square lofted to a rectangle, tiled into a "
                 "layer over the diamond lattice (every section tiles "
                 "so the framed layer interlocks)"),
                ('MCSCUBE', "Interlocking Cubes",
                 "Cubes with a body-diagonal (3-fold) axis vertical, "
                 "placed as identical translates on a honeycomb "
                 "lattice; each cube's faces tilt +/-35.26 deg and "
                 "adjacent cubes share every inclined plane, so the "
                 "framed layer locks (Kanel-Belov / Dyskin moving "
                 "cross-section)"),
                ('MCSOCTA', "Interlocking Octahedra",
                 "Octahedra with a 3-fold axis vertical on the same "
                 "honeycomb; faces tilt +/-19.47 deg (Kanel-Belov "
                 "moving cross-section); the framed layer locks"),
                ('BISQUARE', "Bisquare Blocks",
                 "The exact Bisquare block (Frezier 1737): a square "
                 "base rising to a two-tent roof.  1x1 is a single "
                 "block; larger sizes give the p4 interlocking layer "
                 "(90-deg checkerboard, flat floor, tented top)"),
                ('RHOM', "Rhom Block (single)",
                 "The exact Rhom block (Goertzen 2024): a p3 lozenge "
                 "block, convex, height sqrt6/3"),
                ('RHOM_OBV', "Rhom Block Obverse (single)",
                 "The obverse of the Rhom block; nests with it in "
                 "heterogeneous p3 assemblies"),
                ('KITTEN', "Tetroctahedrille Kitten",
                 "One octahedron + two tetrahedra glued on the "
                 "octet-truss lattice; tiles space by translation "
                 "(Akpanya et al. 2024)"),
                ('UFO', "Tetroctahedrille UFO (single)",
                 "One octahedron + four tetrahedra; a single "
                 "tetroctahedrille block"),
                ('CUSHION', "Tetroctahedrille Cushion (single)",
                 "One octahedron + four tetrahedra in a strip; a "
                 "single tetroctahedrille block"),
                ('SL', "SL Blocks",
                 "Self-interlocking octocubes (Shih 2018): an "
                 "S-tetracube fused to an L-tetracube; conjugate "
                 "pairs chain by the a-engagement into a periodic "
                 "strand (four pairs close the a4 loop)"),
                ('DOME', "Interlocking Dome",
                 "One radially-lofted block per polyhedron face; each "
                 "shared edge is displaced tangentially at the middle "
                 "shell so adjacent blocks share a wavy wall and "
                 "interlock (Escher on a sphere; Akpanya et al. 2024)"),
                ('HENDECA', "Bisymmetric Hendecahedron",
                 "A space-filling 11-hedron (Inchbald 1996): 7 quads "
                 "+ 4 triangles, grown into an interlocking cluster by "
                 "its face-adjacency motions (four cells make a "
                 "translation 'boat'); guaranteed non-overlapping")],
            default='TETRA')

        nx: IntProperty(name="Cells X", default=4, min=1, max=16)
        ny: IntProperty(name="Cells Y", default=4, min=1, max=16)
        nz: IntProperty(name="Cells Z", default=2, min=1, max=8,
                        description="Layers (tetroctahedrille only)")

        profile: EnumProperty(
            name="Profile",
            items=[('SINE', "Sine (Osteomorphic)",
                    "Sinusoidal edge -> saddle block"),
                   ('TENT', "Tent (Versatile)", "Tented edge"),
                   ('ZIGZAG', "S-Curve", "S-shaped edge"),
                   ('STEP', "Step", "Stepped edge")],
            default='SINE')
        deform: FloatProperty(
            name="Deform Depth", default=0.18, min=0.0, max=0.45,
            description="Escher edge-deformation depth (interlock "
                        "depth)")
        samples: IntProperty(name="Edge Samples", default=8, min=2,
                             max=24)
        height: FloatProperty(name="Block Height", default=1.0,
                              min=0.2, max=4.0)

        sl_mode: EnumProperty(
            name="SL Mode",
            items=[('BLOCK', "Single Block",
                    "One SL octocube (S + L tetracubes)"),
                   ('ENGAGEMENT', "Engaged Pair",
                    "The octocube plus one partner engaged by the "
                    "chosen engagement"),
                   ('STRAND', "Square Strand",
                    "The periodic square strand (a h^2n)^4 -- a "
                    "self-locking loop of octocubes")],
            default='STRAND')
        sl_engage: EnumProperty(
            name="Engagement",
            items=[('a', "a  (Rz-90, T 1 -1 0)",
                    "The square-strand engagement; a4 = identity"),
                   ('h', "h  (Rx180, T 2 0 0)", "Flip to the row "
                    "below, half-period along x"),
                   ('s', "s  (Rx180 Rz-90, T 1 1 -1)", "Flip + "
                    "quarter turn"),
                   ('t', "t  (Rx180 Rz90, T 1 -1 1)", "Flip + "
                    "quarter turn (opposite)"),
                   ('d', "d  (T 2 0 -1)", "Pure translation to the "
                    "next block"),
                   ('y', "y  (Rz90, T 1 1 -2)", "Quarter turn, two "
                    "levels down")],
            default='a',
            description="Engagement used for the Engaged Pair mode")
        sl_frame: IntProperty(
            name="Frame Order", default=0, min=0, max=4,
            description="Square strand (a h^2n)^4: n=0 is the tight "
                        "a4 loop, n>=1 nests larger square frames")

        dome_seed: EnumProperty(
            name="Seed",
            items=[('ICOSA', "Icosahedron", "20 triangular blocks"),
                   ('DODECA', "Dodecahedron",
                    "12 pentagonal blocks")],
            default='ICOSA')
        dome_depth: FloatProperty(
            name="Edge Deform", default=0.18, min=0.0, max=0.5,
            description="Tangential push of each shared edge at the "
                        "middle shell (the interlock depth)")
        dome_thick: FloatProperty(
            name="Shell Thickness", default=0.15, min=0.03, max=0.5,
            description="Inner/outer shell offset about the sphere")

        hendeca_count: IntProperty(
            name="Cells", default=32, min=1, max=200,
            description="Number of hendecahedra in the interlocking "
                        "cluster")

        gap: FloatProperty(
            name="Gap Factor", default=0.94, min=0.3, max=1.0,
            description="Scale of each block about its centroid "
                        "(1.0 = blocks touch)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall scale (1.0 fits a "
                                         "2 m cube)")
        colour_mode: EnumProperty(
            name="Colouring",
            items=[('TYPE', "By Block Type",
                    "Two-tone by Truchet colour / block role"),
                   ('FRAME', "Highlight Frame",
                    "Colour the fixed peripheral frame apart"),
                   ('NONE', "None", "Single material")],
            default='TYPE')
        separate: BoolProperty(
            name="Separate Objects", default=False,
            description="Output each element of the pattern as its "
                        "own mesh object")

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            fam = self.family
            if fam in ('TETRA', 'ESCHER', 'VERSATILE', 'MCSCUBE',
                       'MCSOCTA', 'BISQUARE', 'KITTEN'):
                lay.prop(self, 'nx')
                lay.prop(self, 'ny')
            if fam == 'KITTEN':
                lay.prop(self, 'nz')
            if fam == 'ESCHER':
                lay.prop(self, 'profile')
                lay.prop(self, 'deform')
                lay.prop(self, 'samples')
                lay.prop(self, 'height')
            if fam == 'SL':
                lay.prop(self, 'sl_mode')
                if self.sl_mode == 'ENGAGEMENT':
                    lay.prop(self, 'sl_engage')
                if self.sl_mode == 'STRAND':
                    lay.prop(self, 'sl_frame')
            if fam == 'DOME':
                lay.prop(self, 'dome_seed')
                lay.prop(self, 'dome_depth')
                lay.prop(self, 'dome_thick')
            if fam == 'HENDECA':
                lay.prop(self, 'hendeca_count')
            lay.prop(self, 'gap')
            lay.prop(self, 'scale')
            lay.prop(self, 'colour_mode')
            lay.prop(self, 'separate')

        def _emit(self, context, name, verts, faces, cols, frames):
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if self.colour_mode != 'NONE' and \
                    len(me.polygons) == len(cols):
                if self.colour_mode == 'FRAME':
                    me.materials.append(_material("Interlock Body",
                                                  _COLOURS[0]))
                    me.materials.append(_material("Interlock Frame",
                                                  _FRAME_RGB))
                    me.polygons.foreach_set('material_index', frames)
                else:
                    used = sorted(set(cols))
                    for c in used:
                        me.materials.append(_material(
                            f"Interlock {c}",
                            _COLOURS.get(c, (0.6, 0.6, 0.6))))
                    remap = {c: idx for idx, c in enumerate(used)}
                    me.polygons.foreach_set(
                        'material_index', [remap[c] for c in cols])
            me.update()
            try:
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                bm.to_mesh(me)
                bm.free()
                me.update()
            except Exception:
                pass
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            obj.select_set(True)
            return obj

        def execute(self, context):
            try:
                cells = build_cells(
                    self.family, self.nx, self.ny, self.nz,
                    self.profile, self.deform, self.samples,
                    self.height, self.sl_mode, self.sl_engage,
                    self.sl_frame, self.dome_seed, self.dome_depth,
                    self.dome_thick, self.hendeca_count)
            except Exception as e:               # bad parameter combo
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            for o in context.selected_objects:
                o.select_set(False)
            label = _LABEL[self.family]
            if self.separate:
                parts = cells_to_meshes(cells, 2.0 * self.scale,
                                        self.gap)
                first = None
                for k, (v, f, cols, frames) in enumerate(parts):
                    obj = self._emit(context, f"{label} {k + 1}",
                                     v, f, cols, frames)
                    if first is None:
                        first = obj
                context.view_layer.objects.active = first
                self.report({'INFO'},
                            f"{label}: {len(parts)} separate objects")
                return {'FINISHED'}
            verts, faces, cols, frames = cells_to_mesh(
                cells, 2.0 * self.scale, self.gap)
            obj = self._emit(context, f"Interlocking {label}",
                             verts, faces, cols, frames)
            context.view_layer.objects.active = obj
            tag = ("interlocking (framed interior)"
                   if self.family in _RIGOROUS else "modular blocks")
            self.report({'INFO'},
                        f"{label}: {tag}, V={len(obj.data.vertices)} "
                        f"F={len(obj.data.polygons)}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.interlocking_add",
                             icon='MOD_BUILD')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_interlocking_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_interlocking_add)


# ==================================================================
# standalone self-test
# ==================================================================

def _selftest():
    from collections import Counter

    def closed_bad_edges(V, F):
        ec = Counter()
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                ec[(min(a, b), max(a, b))] += 1
        return sum(1 for c in ec.values() if c != 2)

    ok = True

    # regular tetrahedra
    for nm, V, F in (('A', _TETRA_A, _TETRA_FA),
                     ('B', _TETRA_B, _TETRA_FB)):
        el = [np.linalg.norm(V[f[i]] - V[f[(i + 1) % 3]])
              for f in F for i in range(3)]
        reg = max(el) - min(el) < 1e-9
        print(f"tetra {nm}: regular={reg} vol={_mesh_volume(V, F):.4f}")
        ok = ok and reg

    # MCS cells are exact platonic (kept for reuse / backlog)
    for kind, nv, nf in (('CUBE', 8, 6), ('OCTA', 6, 8)):
        V, F = _mcs_cell(kind)
        el = [np.linalg.norm(np.asarray(V)[f[i]] -
              np.asarray(V)[f[(i + 1) % len(f)]])
              for f in F for i in range(len(f))]
        reg = max(el) - min(el) < 1e-6 and len(V) == nv and \
            len(F) == nf
        print(f"mcs {kind}: regular={reg} V={len(V)} F={len(F)}")
        ok = ok and reg

    # Versatile exact block: closed manifold, unit-height volume 2
    V0 = _VERSATILE_V - _VERSATILE_V.mean(axis=0)
    F = _orient_outward(V0, _VERSATILE_F)
    be = closed_bad_edges(V0, F)
    vol = abs(_mesh_volume(V0, F))
    print(f"versatile: bad_edges={be} vol={vol:.4f}")
    ok = ok and be == 0 and abs(vol - 2.0) < 1e-6

    # tetroctahedrille primitives are exact regular tet/oct
    for nm, prim, ev in (('OCT', _OCT, 4.0 / 3.0),
                         ('T1', _T1, 1.0 / 3.0),
                         ('T2', _T2, 1.0 / 3.0)):
        Fp = _convex_faces(prim)
        v = abs(_mesh_volume(prim, Fp))
        good = closed_bad_edges(prim, Fp) == 0 and abs(v - ev) < 1e-9
        print(f"tetrocta {nm}: closed+vol_ok={good} vol={v:.4f}")
        ok = ok and good

    # every family builds a non-degenerate mesh that fits the cube
    for name, cells in (
            ('TETRA', build_tetra(4, 4)),
            ('ESCHER', build_escher('SINE', 4, 4, 0.18, 8, 1.0)),
            ('VERSATILE', build_versatile(4, 4)),
            ('MCSCUBE', build_mcs('CUBE', 4, 4)),
            ('MCSOCTA', build_mcs('OCTA', 4, 4)),
            ('BISQUARE', build_bisquare()),
            ('RHOM', build_rhom(False)),
            ('RHOM_OBV', build_rhom(True)),
            ('KITTEN', build_tetrocta('KITTEN', 2, 2, 2)),
            ('UFO', build_tetrocta('UFO', 1, 1, 1)),
            ('CUSHION', build_tetrocta('CUSHION', 1, 1, 1)),
            ('SL', build_sl('STRAND', 'a', 0)),
            ('DOME_I', build_dome('ICOSA', 0.18, 0.15)),
            ('DOME_D', build_dome('DODECA', 0.18, 0.15)),
            ('HENDECA', build_hendeca(20))):
        v, f, cols, fr = cells_to_mesh(cells, 2.0, 0.94)
        Vv = np.asarray(v)
        deg = sum(1 for ff in f
                  if np.linalg.norm(np.cross(
                      Vv[ff[1]] - Vv[ff[0]],
                      Vv[ff[2]] - Vv[ff[0]])) < 1e-9)
        fit = (Vv.max(axis=0) - Vv.min(axis=0)).max() <= 2.0 + 1e-6
        good = deg == 0 and fit and len(v) > 0
        print(f"family {name}: mesh_ok={good} V={len(v)} F={len(f)} "
              f"degen={deg}")
        ok = ok and good

    # OVERLAP GATE: assembled families must not interpenetrate.
    # convex-primitive families use an interior-penetration test;
    # the Escher tiling is verified by a section-partition test.
    def _face_planes(V, F):
        V = np.asarray(V)
        c = V.mean(axis=0)
        pls = []
        for f in F:
            p = V[f]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n = n / ln
            d = float(np.dot(n, p[0]))
            if np.dot(n, c) > d:
                n, d = -n, -d
            pls.append((n, d))
        return pls

    def _maxpen(A, B):
        Va, Fa = A
        plB = _face_planes(*B)
        Va = np.asarray(Va)
        ca = Va.mean(axis=0)
        S = [ca] + [0.5 * Va[i] + 0.5 * ca for i in range(len(Va))]
        for f in Fa:
            S.append(0.5 * ca + 0.5 * Va[f].mean(axis=0))
        return max(min(d - np.dot(n, x) for n, d in plB) for x in S)

    def _assembly_overlap(cells):
        sol = [(np.asarray(c) + np.asarray(V), F)
               for c, V, F, fr, col in cells]
        cen = [s[0].mean(axis=0) for s in sol]
        worst = -9.0
        for a in range(len(sol)):
            for b in range(a + 1, len(sol)):
                if np.linalg.norm(cen[a] - cen[b]) < 3.0:
                    worst = max(worst, min(_maxpen(sol[a], sol[b]),
                                           _maxpen(sol[b], sol[a])))
        return worst

    for name, cells in (
            ('TETRA', build_tetra(5, 5)),
            ('MCSCUBE', build_mcs('CUBE', 4, 4)),
            ('MCSOCTA', build_mcs('OCTA', 4, 4)),
            ('KITTEN', build_tetrocta('KITTEN', 2, 2, 2))):
        w = _assembly_overlap(cells)
        good = w < 1e-3
        print(f"overlap {name}: max_penetration={w:+.4f} "
              f"{'ok' if good else 'OVERLAP'}")
        ok = ok and good

    # SL blocks are integer polycubes -> overlap is exact (a cube
    # shared by two blocks).  Each square strand (a h^{2n})^4 must
    # close to the identity and be globally cube-disjoint, and each of
    # the six single engagements must give a disjoint 2-block pair.
    octo = [c for o, _ in _sl_octocube() for c in map(tuple, o)]
    for n in range(4):
        frames = _sl_frames(('a' + 'h' * (2 * n)) * 4)
        # closure: the word's full product returns to the start, so the
        # dropped frame count equals one full loop (len(word) frames ->
        # len(word) distinct blocks, the (len+1)-th coincident)
        closes = len(frames) == 4 * (1 + 2 * n)
        cubes = []
        for R, t in frames:
            cubes += [tuple(np.round(R @ np.array(p, float) + t)
                            .astype(int)) for p in octo]
        disjoint = len(cubes) == len(set(cubes))
        good = closes and disjoint
        print(f"overlap SL (a h^{2 * n})^4: blocks={len(frames)} "
              f"closes={closes} disjoint={disjoint} "
              f"{'ok' if good else 'OVERLAP'}")
        ok = ok and good
    for e in 'ahstdy':
        b0, b1 = _sl_frames(e)
        c0 = {tuple(np.round(b0[0] @ np.array(p, float) + b0[1])
                    .astype(int)) for p in octo}
        c1 = {tuple(np.round(b1[0] @ np.array(p, float) + b1[1])
                    .astype(int)) for p in octo}
        good = len(c0 & c1) == 0
        print(f"overlap SL engagement {e}: 2-block disjoint={good}")
        ok = ok and good

    # Bisquare p4 layer: non-convex roofs, so overlap is tested by a
    # ray-cast point-in-mesh coverage sample -- the interlocking layer
    # must have NO interpenetration (gaps are allowed: the roof
    # valleys make it a tented layer, not a solid fill).
    def _inside(P, V, tri, d):
        cnt = np.zeros(len(P), int)
        for f in tri:
            a, e1, e2 = V[f[0]], V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]
            h = np.cross(d, e2)
            det = e1 @ h
            if abs(det) < 1e-12:
                continue
            s = P - a
            u = (s @ h) / det
            q = np.cross(s, e1)
            v = (q @ d) / det
            tt = (q @ e2) / det
            cnt += ((u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) &
                    (u + v <= 1 + 1e-9) & (tt > 1e-9))
        return cnt % 2 == 1

    bcells = build_bisquare(4, 4)
    d = np.array((0.5123, 0.311, 0.8007))
    d = d / np.linalg.norm(d)
    lo = np.min([c[1].min(axis=0) for c in bcells], axis=0)
    hi = np.max([c[1].max(axis=0) for c in bcells], axis=0)
    rng2 = np.random.RandomState(1)
    P = lo + (hi - lo) * rng2.uniform(size=(1500, 3))
    P[:, 2] = rng2.uniform(0.05, 0.95, 1500)
    cov = np.zeros(len(P), int)
    for _, V, F, _fr, _c in bcells:
        cov += _inside(P, np.asarray(V), F, d)
    overlaps = int((cov > 1).sum())
    print(f"overlap BISQUARE p4 layer: interpenetrations={overlaps} "
          f"{'ok' if overlaps == 0 else 'OVERLAP'}")
    ok = ok and overlaps == 0

    # Bisymmetric hendecahedron: the cell is a closed 11-hedron of
    # volume 16, and the grown cluster must not interpenetrate.
    be = closed_bad_edges(_HENDECA_V, _HENDECA_F)

    def _convex_volume(V, F):           # winding-independent (convex)
        V = np.asarray(V)
        c = V.mean(axis=0)
        vol = 0.0
        for f in F:
            p = V[f]
            fc = p.mean(axis=0)
            n = np.cross(p[1] - p[0], p[2] - p[0])
            n = n / np.linalg.norm(n)
            if np.dot(n, fc - c) < 0:
                n = -n
            cs = np.zeros(3)             # positive face area (magnitude)
            for i in range(len(f)):
                cs = cs + np.cross(p[i] - fc, p[(i + 1) % len(f)] - fc)
            area = np.linalg.norm(cs) / 2
            vol += area * np.dot(n, fc) / 3
        return abs(vol)

    hv = _convex_volume(_HENDECA_V, _HENDECA_F)
    good = be == 0 and abs(hv - 16.0) < 1e-6 and len(_HENDECA_F) == 11
    print(f"hendecahedron cell: bad_edges={be} faces={len(_HENDECA_F)} "
          f"vol={hv:.3f} {'ok' if good else 'BAD'}")
    ok = ok and good
    hcells = build_hendeca(48)
    hverts = [np.asarray(V) for _, V, _f, _fr, _c in hcells]
    hbad = 0
    for a in range(len(hverts)):
        for b in range(a + 1, len(hverts)):
            if np.linalg.norm(hverts[a].mean(0) - hverts[b].mean(0)) \
                    < 5.0 and _convex_overlap(hverts[a], hverts[b],
                                              _HENDECA_F, _HENDECA_F):
                hbad += 1
    print(f"overlap HENDECA cluster: cells={len(hcells)} "
          f"interpenetrations={hbad} {'ok' if hbad == 0 else 'OVERLAP'}")
    ok = ok and hbad == 0

    # the boat motif + lattice must fill space (coverage exactly one)

    def _in_cell(p, W):
        cc = W.mean(axis=0)
        for f in _HENDECA_F:
            q = W[f]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            n = n / np.linalg.norm(n)
            d = np.dot(n, q[0])
            if np.dot(n, cc) > d:
                n, d = -n, -d
            if np.dot(n, p) - d > 1e-9:
                return False
        return True

    tiles = []
    for R, t in _HENDECA_MOTIF:
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    tiles.append(_HENDECA_V @ R.T + (t + i * _HENDECA_L[0]
                                 + j * _HENDECA_L[1] + k * _HENDECA_L[2]))
    tcen = [W.mean(axis=0) for W in tiles]
    rngh = np.random.RandomState(7)
    Ph = rngh.uniform(-4.0, 4.0, (2000, 3))
    hg = ho = 0
    for p in Ph:
        cnt = sum(1 for W, c in zip(tiles, tcen)
                  if np.linalg.norm(c - p) < 2.5 and _in_cell(p, W))
        if cnt == 0:
            hg += 1
        elif cnt > 1:
            ho += 1
    good = hg == 0 and ho == 0
    print(f"tiling HENDECA boat+lattice: gaps={hg} overlaps={ho} "
          f"{'ok' if good else 'FAIL'}")
    ok = ok and good

    # Escher: every plane section must tile (coverage exactly 1)
    def _pip(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    rng = np.random.RandomState(0)
    for prof in ('SINE', 'TENT', 'ZIGZAG', 'STEP'):
        base = _deformed_square(prof, 0.18, 10)
        mid = _rot2(base, math.pi / 2.0)
        good = True
        for tile in (base, mid):
            for pt in rng.uniform(-0.45, 0.45, (200, 2)):
                cov = sum(1 for i in range(-2, 3) for j in range(-2, 3)
                          if _pip(pt - np.array([i, j]), tile))
                if cov != 1:
                    good = False
                    break
        print(f"escher tiling {prof}: coverage_one={good}")
        ok = ok and good

    if not ok:
        raise AssertionError("interlocking self-test failed")
    print("RESULT: OK")
    return ok
