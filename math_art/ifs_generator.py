
# Iterated Function System Generator for Blender
#
# Attractors of iterated function systems in three dimensions, in two
# families that need quite different machinery.  Both live in the
# sibling `ifs` engine package, which is where the mathematics and its
# references are documented:
#
#   ifs.radix    self-affine lattice tiles -- an expanding integer matrix
#                M with a complete residue digit set D, giving a tile T
#                with M T = T + D that tiles R^3 by Z^3.  Exact at every
#                level, not sampled.  Presets ABC (Thuswaldner-Zhang
#                normal form), TWINDRAGON (Bandt's seven) and CUBE.
#   ifs.affine   general affine attractors: contractive maps
#                w_i(x) = A_i x + b_i, unique compact attractor by
#                Hutchinson's theorem, rendered as deterministic solid
#                copies, or by the chaos game into a voxel grid (blocky
#                and printable) or a density field (smooth).
#   ifs.voxel    the surfacing both families share -- exterior-face
#                walker, pinhole repair, outward orientation, and the
#                fit to the 2 m cube.
#
# This module is the Blender layer over that engine: properties, the
# operator, and the menu entry.  Geometry only; materials and rendering
# are left to Blender.
#
# References:
# - C. Bandt, "Self-similar sets 5. Integer matrices and fractal
#   tilings of R^n", Proceedings of the American Mathematical Society
#   112, 1991, pp. 549-562.
# - C. Bandt, Mai The Duy and M. Mesing, "Three-Dimensional Fractals",
#   The Mathematical Intelligencer 32, 2010, pp. 12-18.
#   doi:10.1007/s00283-009-9110-6
# - J. M. Thuswaldner and S.-Q. Zhang, "On self-affine tiles that are
#   homeomorphic to a ball", arXiv:2107.12076.
# - J. E. Hutchinson, "Fractals and self similarity", Indiana
#   University Mathematics Journal 30, 1981.
# - M. F. Barnsley, "Fractals Everywhere", 2nd ed., Academic Press, 1993.

bl_info = {
    "name": "Iterated Function System Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Iterated Function System",
    "description": "Self-affine lattice tiles from an expanding integer "
                   "matrix, and affine IFS attractors in two and three "
                   "dimensions",
    "category": "Add Mesh",
}

import math

import numpy as np

# The mathematics lives in the sibling `ifs` engine package; this module
# is the Blender layer over it.  The private names on the second import
# are reached for by the self-test below, which checks the engine's
# internals (the BMM map families and the volume sign) and not just its
# public surface.
try:
    from .ifs import (IFS_FACTS, IFS_PRESETS, RADIX_PRESETS, SEEDS,
                      abc_has_14_neighbours, abc_is_ball, attractor_rank,
                      build_ifs, build_radix, chaos_game, companion,
                      contractive, edge_stats, format_maps, is_expanding,
                      is_residue_system, max_holes, max_level, parse_maps,
                      radix_points, radix_topology, tile_support_bbox)
    from .ifs.affine import _BMM_TETRA_V, _bmm_sierpinski, _rot
    from .ifs.voxel import _signed_volume
except ImportError:  # flat import outside the package (headless runners)
    from ifs import (IFS_FACTS, IFS_PRESETS, RADIX_PRESETS, SEEDS,
                     abc_has_14_neighbours, abc_is_ball, attractor_rank,
                     build_ifs, build_radix, chaos_game, companion,
                     contractive, edge_stats, format_maps, is_expanding,
                     is_residue_system, max_holes, max_level, parse_maps,
                     radix_points, radix_topology, tile_support_bbox)
    from ifs.affine import _BMM_TETRA_V, _bmm_sierpinski, _rot
    from ifs.voxel import _signed_volume


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    # Blender hands dynamic enum items back to C, which does not keep
    # the Python strings alive; anything returned from an items
    # callback has to be held in a module-level cache or the labels
    # turn to garbage.
    _ENUM_CACHE = {}

    # Selecting a system loads its maps into the field; editing the
    # field flips the system to Custom so the edit is what gets built.
    # The flag stops the two callbacks from calling each other.
    _SYNC = {'busy': False}

    def _on_system(self, context):
        if _SYNC['busy'] or self.ifs_preset == 'CUSTOM':
            return
        entry = IFS_PRESETS.get(self.ifs_preset)
        if entry is None:
            return
        _SYNC['busy'] = True
        try:
            self.maps = format_maps(entry[1]())
        finally:
            _SYNC['busy'] = False

    def _on_maps(self, context):
        if _SYNC['busy'] or self.ifs_preset == 'CUSTOM':
            return
        _SYNC['busy'] = True
        try:
            self.ifs_preset = 'CUSTOM'
        finally:
            _SYNC['busy'] = False

    def _system_items(self, context):
        dim = int(getattr(self, 'dimension', '3'))
        items = [(k, v[0], v[0]) for k, v in IFS_PRESETS.items()
                 if v[2] == dim]
        items.append(('CUSTOM', "Custom", "Use the maps field below"))
        _ENUM_CACHE[f'sys{dim}'] = items
        return items

    def _output_items(self, context):
        dim = int(getattr(self, 'dimension', '3'))
        if dim == 2:
            # a planar attractor has no solid image, and a volume grid
            # over it would be almost entirely empty
            items = [('RELIEF', "Relief",
                      "A watertight slab one cell thick, meshed at "
                      "plane resolution")]
        else:
            items = [('SOLIDS', "Solid Copies",
                      "Deterministic: one seed solid per word"),
                     ('VOXEL', "Voxels",
                      "Chaos game binned into a watertight voxel grid"),
                     ('ISO', "Smooth Contour",
                      "Chaos game contoured by marching tetrahedra")]
        _ENUM_CACHE[f'out{dim}'] = items
        return items

    class MESH_OT_ifs_add(bpy.types.Operator):
        """Add a three-dimensional self-affine tile or the attractor of
        an affine iterated function system"""
        bl_idname = "mesh.ifs_add"
        bl_label = "Iterated Function System"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('IFS', "IFS Attractor", "The attractor of a set of "
                                            "contractive affine maps"),
                   ('RADIX', "Self-Affine Tile", "A lattice tile from "
                                                 "an expanding integer "
                                                 "matrix and a residue "
                                                 "digit set")],
            default='IFS')
        preset: EnumProperty(
            name="Tile",
            items=[(k, v[0], v[0]) for k, v in RADIX_PRESETS.items()],
            default='ABC_124')
        tile_output: EnumProperty(
            name="Tile Output",
            items=[('VOXEL', "Voxels", "Sample the attractor and mesh "
                                       "it as a watertight voxel "
                                       "solid"),
                   ('SMOOTH', "Smooth Contour", "Sample the attractor "
                                                "and contour it with "
                                                "marching tetrahedra"),
                   ('EXACT', "Exact Level-k Cubes", "The exact union "
                                                    "of C^k cells, "
                                                    "volume exactly 1 "
                                                    "-- but the cells "
                                                    "flatten into "
                                                    "plates as the "
                                                    "level rises")],
            default='VOXEL')
        level: IntProperty(
            name="Level", default=0, min=0, max=24,
            description="Exact mode only: radix depth; 0 picks a level "
                        "landing in the 30k-300k cell band")
        holes: IntProperty(
            name="Holes", default=0, min=0, max=6,
            description="Drop this many digits at every level, turning "
                        "the tile into a gasket")
        dimension: EnumProperty(
            name="Dimension",
            items=[('3', "3D", "Systems whose attractor fills three "
                               "dimensions"),
                   ('2', "2D", "Planar systems, meshed as a relief")],
            default='3')
        # both of these are filtered by the dimension above, so they
        # take an items callback rather than a fixed list; a dynamic
        # enum cannot carry a default, so the first entry is it
        ifs_preset: EnumProperty(name="System", items=_system_items,
                                 update=_on_system)
        output: EnumProperty(name="Output", items=_output_items)
        seed_solid: EnumProperty(
            name="Seed Solid",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", "")],
            default='TETRA')
        depth: IntProperty(
            name="Depth", default=5, min=1, max=12,
            description="Solid-copies depth; the count is maps^depth "
                        "and is capped automatically")
        points: IntProperty(
            name="Points", default=400000, min=10000, max=5000000,
            description="Chaos-game sample count")
        resolution: IntProperty(
            name="Resolution", default=128, min=16, max=256,
            description="Voxel / density grid resolution per axis")
        plane_resolution: IntProperty(
            name="Plane Resolution", default=512, min=32, max=2048,
            description="In-plane grid resolution for a planar "
                        "system; a plane affords far more of it than a "
                        "volume can")
        cover: FloatProperty(
            name="Cover", default=0.90, min=0.1, max=0.999,
            description="Smooth contour: the fraction of the sampled "
                        "mass the surface encloses")
        min_count: IntProperty(
            name="Min Points per Cell", default=1, min=1, max=200,
            description="Voxel mode: cells with fewer points than this "
                        "are left empty")
        maps: StringProperty(
            name="Maps", update=_on_maps,
            default="0.5 0 0 0 0.5 0 0 0 0.5 | 0.5 0.5 0.5 | 1; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | -0.5 -0.5 0.5 | 1; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | 0.5 -0.5 -0.5 | 1; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | -0.5 0.5 -0.5 | 1",
            description="Custom affine maps: nine matrix entries | "
                        "three translations | probability, one map per "
                        "semicolon")
        seed: IntProperty(
            name="Seed", default=0, min=0, max=99999,
            description="Chaos-game random seed; the same seed always "
                        "gives the same mesh")
        poly_sides: IntProperty(
            name="Polygon Sides", default=3, min=3, max=10,
            description="The n-gon the Sierpinski-in-3D construction "
                        "is built on; the paper derives its ratio for "
                        "the triangle and notes the construction "
                        "applies to every n >= 3")
        poly_ratio: FloatProperty(
            name="Polygon Ratio", default=2.0 / 3.0, min=0.35,
            max=0.95,
            description="Contraction ratio toward each vertex; 2/3 is "
                        "the triangle value at which the pieces meet, "
                        "and 1/2 would give a Cantor set")
        reverse: BoolProperty(
            name="Reverse", default=False,
            description="Replace every map f by -f: the neighbour maps "
                        "are unchanged, so the dimension and the "
                        "boundary structure survive, but the shape "
                        "does not")
        largest_only: BoolProperty(
            name="Largest Piece Only", default=False,
            description="Smooth contour: discard all but the biggest "
                        "connected piece")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(name="Smooth Shading", default=False)

        def execute(self, context):
            try:
                if self.mode == 'RADIX':
                    verts, faces, info = build_radix(
                        preset=self.preset, level=self.level,
                        holes=self.holes, output=self.tile_output,
                        resolution=self.resolution, points=self.points,
                        seed=self.seed,
                        largest_only=self.largest_only,
                        scale=self.scale)
                    label = RADIX_PRESETS[self.preset][0]
                    if info['holes']:
                        label += f" gasket -{info['holes']}"
                else:
                    if (self.ifs_preset != 'CUSTOM'
                            and self.ifs_preset in IFS_PRESETS
                            and IFS_PRESETS[self.ifs_preset][2]
                            != int(self.dimension)):
                        raise ValueError(
                            f"{IFS_PRESETS[self.ifs_preset][0]} is a "
                            f"{IFS_PRESETS[self.ifs_preset][2]}D "
                            f"system; switch the Dimension to match")
                    if self.ifs_preset == 'CUSTOM':
                        mp = parse_maps(self.maps)
                    elif self.ifs_preset == 'BMM_SIERP':
                        mp = _bmm_sierpinski(self.poly_sides,
                                             self.poly_ratio)
                    else:
                        mp = None
                    verts, faces, info = build_ifs(
                        preset=self.ifs_preset, output=self.output,
                        maps=mp, depth=self.depth,
                        seed_solid=self.seed_solid, points=self.points,
                        resolution=self.resolution,
                        plane_resolution=self.plane_resolution,
                        cover=self.cover,
                        seed=self.seed, min_count=self.min_count,
                        largest_only=self.largest_only,
                        reverse=self.reverse,
                        scale=self.scale)
                    label = (IFS_PRESETS[self.ifs_preset][0]
                             if self.ifs_preset != 'CUSTOM'
                             else "Custom IFS")
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

            # edge-to-edge contact is intrinsic to these families, but
            # a slicer will choke on it, so say so -- only when the
            # count is small enough that the check is cheap
            if len(faces) <= 200000 and self.output != 'ISO':
                nb, nm = edge_stats(faces)
                if nb:
                    self.report({'WARNING'},
                                f"{label}: {nb} boundary edges -- the "
                                f"surface is not closed")
                elif nm:
                    self.report({'WARNING'},
                                f"{label}: closed, but {nm} edges have "
                                f"cells meeting only edge to edge; "
                                f"non-manifold, so thicken it before "
                                f"printing")

            obj = _new_object(context, label, verts, faces,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            if self.mode == 'RADIX':
                if info.get('topology'):
                    self.report({'INFO'},
                                f"{label}: {info['topology']}")
                if info.get('holes_clamped'):
                    self.report({'WARNING'},
                                f"{label}: dropping more than "
                                f"{info['max_holes']} digits leaves "
                                f"them coplanar, which would collapse "
                                f"the tile to a sheet -- clamped")
                fid = info.get('fidelity', 1.0)
                if self.tile_output == 'EXACT':
                    asp = info.get('cell_aspect', 1.0)
                    if asp > 20.0:
                        self.report(
                            {'WARNING'},
                            f"{label}: at level {info['level']} the "
                            f"cells are {asp:.0f}:1 slivers, so this "
                            f"reads as a laminate rather than a solid "
                            f"-- use the Voxels or Smooth output")
                    if fid < 0.95:
                        self.report(
                            {'WARNING'},
                            f"{label}: the level-{info['level']} body "
                            f"reaches only {100 * fid:.0f}% of the "
                            f"true tile's extent on its thinnest axis")
                    self.report(
                        {'INFO'},
                        f"{label}: level {info['level']}, "
                        f"{info['cells']} cells, volume "
                        f"{info['volume']:.4f}, cell aspect "
                        f"{info['cell_aspect']:.0f}:1, "
                        f"{len(me.vertices)} verts")
                else:
                    self.report(
                        {'INFO'},
                        f"{label}: {info.get('points', 0)} attractor "
                        f"samples at resolution "
                        f"{info.get('resolution', 0)}, "
                        f"{100 * fid:.0f}% of the true extent, "
                        f"{len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces")
            else:
                note = IFS_FACTS.get(self.ifs_preset)
                if note:
                    self.report({'INFO'}, f"{label}: {note}")
                if self.output == 'SOLIDS':
                    extra = f"{info.get('copies', 0)} copies"
                elif info.get('planar'):
                    extra = (f"planar relief, {info.get('cells', 0)} "
                             f"cells at {info.get('resolution', 0)}^2")
                else:
                    extra = f"{info.get('points', 0)} points"
                self.report({'INFO'},
                            f"{label}: {extra}, {len(me.vertices)} "
                            f"verts, {len(me.polygons)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'RADIX':
                lay.prop(self, 'preset')
                lay.prop(self, 'tile_output')
                lay.prop(self, 'holes')
                if self.tile_output == 'EXACT':
                    lay.prop(self, 'level')
                else:
                    lay.prop(self, 'resolution')
                    lay.prop(self, 'points')
                    lay.prop(self, 'seed')
                    if self.tile_output == 'SMOOTH':
                        lay.prop(self, 'largest_only')
            else:
                lay.prop(self, 'dimension', expand=True)
                lay.prop(self, 'ifs_preset')
                if self.ifs_preset == 'BMM_SIERP':
                    lay.prop(self, 'poly_sides')
                    lay.prop(self, 'poly_ratio')
                lay.prop(self, 'reverse')
                lay.prop(self, 'maps')
                lay.prop(self, 'output')
                if self.output == 'SOLIDS':
                    lay.prop(self, 'seed_solid')
                    lay.prop(self, 'depth')
                else:
                    lay.prop(self, 'points')
                    lay.prop(self, 'seed')
                    if self.output == 'RELIEF':
                        lay.prop(self, 'plane_resolution')
                    else:
                        lay.prop(self, 'resolution')
                        if self.output == 'VOXEL':
                            lay.prop(self, 'min_count')
                        else:
                            lay.prop(self, 'cover')
                            lay.prop(self, 'largest_only')
            for k in ('scale', 'thickness', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.ifs_add", icon='MOD_REMESH')

    _classes = (MESH_OT_ifs_add,)

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


# ==========================================================================
# Standalone numeric self-test
# ==========================================================================



def _selftest():
    # ---- 0. normals point outward, everywhere ------------------------
    # Three separate ways to get this wrong, all of them silent: a seed
    # solid wound the wrong way; an orientation-REVERSING transform
    # (det M is negative for every ABC tile, so M^-k flips at odd
    # levels); and an isosurface left open, whose normals then mean
    # nothing at all.  The divergence theorem catches all three.
    for name, (sv, sf) in SEEDS.items():
        vol = _signed_volume(sv, sf)
        if vol <= 0.0:
            raise AssertionError(
                f"the {name} seed solid is wound inside out "
                f"(signed volume {vol:+.4f})")
    print(f"seed solids: {', '.join(sorted(SEEDS))} all wound outward")

    checks = [
        ("radix exact, odd level",
         lambda: build_radix(preset='ABC_124', level=3, output='EXACT')),
        ("radix exact, even level",
         lambda: build_radix(preset='ABC_124', level=4, output='EXACT')),
        ("radix voxels",
         lambda: build_radix(preset='ABC_124', output='VOXEL',
                             resolution=48, points=120000)),
        ("radix smooth",
         lambda: build_radix(preset='ABC_223', output='SMOOTH',
                             resolution=48, points=150000)),
        ("ifs solid tetrahedra",
         lambda: build_ifs(preset='SIERP_TETRA', output='SOLIDS',
                           depth=3)),
        ("ifs solid cubes",
         lambda: build_ifs(preset='MENGER', output='SOLIDS',
                           seed_solid='CUBE', depth=1)),
        ("ifs solid octahedra",
         lambda: build_ifs(preset='SIERP_OCTA', output='SOLIDS',
                           seed_solid='OCTA', depth=2)),
        ("ifs voxels",
         lambda: build_ifs(preset='SIERP_TETRA', output='VOXEL',
                           points=80000, resolution=40)),
        ("ifs smooth",
         lambda: build_ifs(preset='SIERP_TETRA', output='ISO',
                           points=150000, resolution=48)),
        ("planar relief",
         lambda: build_ifs(preset='SIERP_TRI', output='RELIEF',
                           points=150000, plane_resolution=192)),
    ]
    for name, fn in checks:
        V, F, info = fn()
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(
                f"{name}: {nb} boundary edges -- the surface is open, "
                f"so its normals are meaningless")
        vol = _signed_volume(V, F)
        if vol <= 0.0:
            raise AssertionError(
                f"{name}: signed volume {vol:+.4f} -- the mesh is "
                f"inside out")
        print(f"{name:24s}: closed, signed volume {vol:+.4f}")

    # ---- 1. every preset really is a radix system -------------------
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        if not is_expanding(M):
            raise AssertionError(f"{label}: matrix is not expanding, "
                                 f"eigenvalues "
                                 f"{np.linalg.eigvals(M.astype(float))}")
        if not is_residue_system(M, D):
            raise AssertionError(f"{label}: digits are not a complete "
                                 f"residue system")
        det = abs(int(round(np.linalg.det(M.astype(float)))))
        if det != len(D):
            raise AssertionError(f"{label}: |det M| = {det} but there "
                                 f"are {len(D)} digits")
    print(f"{len(RADIX_PRESETS)} radix presets: expanding, "
          f"|det M| = |D|, digits a complete residue system")

    # ---- 2. |S_k| = C^k exactly (this is what the residue condition
    #         buys, and the check that catches a wrong digit set) -----
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        C = len(D)
        kmax = min(6, max_level(C))
        for k in range(1, kmax + 1):
            S = radix_points(M, D, k)
            uniq = len(np.unique(S, axis=0))
            if uniq != C ** k or len(S) != C ** k:
                raise AssertionError(
                    f"{label}: level {k} has {uniq} distinct points of "
                    f"{len(S)}, expected {C ** k}")
    print("radix point sets: |S_k| = C^k distinct points, all presets")

    # gaskets drop exactly the digits asked for
    M, D = RADIX_PRESETS['CUBE'][1]
    for h in (1, 2, 4):
        for k in (1, 2, 3):
            S = radix_points(M, D, k, holes=h)
            if len(np.unique(S, axis=0)) != (8 - h) ** k:
                raise AssertionError(
                    f"cube gasket -{h}: level {k} has "
                    f"{len(np.unique(S, axis=0))} points, expected "
                    f"{(8 - h) ** k}")
    print("gaskets: (C-h)^k cells at every level")

    # ---- 3. the exact body: volume exactly 1, and watertight --------
    for key in ('ABC_124', 'TWIN_A', 'TWIN_G', 'CUBE'):
        label = RADIX_PRESETS[key][0]
        V, F, info = build_radix(preset=key, level=4, output='EXACT')
        if abs(info['volume'] - 1.0) > 1e-12:
            raise AssertionError(
                f"{label}: level-4 volume {info['volume']} != 1")
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(
                f"{label}: {nb} boundary edges -- the surface has a "
                f"hole in it")
        ext = float((V.max(axis=0) - V.min(axis=0)).max())
        if abs(ext - 2.0) > 1e-6:
            raise AssertionError(f"{label}: {ext:.4f} across, expected "
                                 f"a 2 m fit")
        note = (f", {nm} non-manifold edges (cubes touching edge to "
                f"edge)" if nm else "")
        print(f"{label:24s}: level {info['level']}, {info['cells']:6d} "
              f"cells, volume {info['volume']:.6f}, closed{note}")

    # ---- 3a. the papers' own arithmetic and tables -------------------
    # Thuswaldner-Zhang Remark 1.3 reduced to arithmetic, checked
    # against the neighbour counts computed for these tiles: only
    # (1,2,4) of the originally shipped six had 14 neighbours.
    for (A, B, C), want in (((1, 1, 2), False), ((1, 1, 3), False),
                            ((1, 2, 3), False), ((1, 2, 4), True),
                            ((1, 3, 4), False), ((2, 2, 3), False),
                            ((1, 2, 5), True), ((1, 2, 8), True),
                            ((1, 3, 6), True), ((1, 4, 8), True),
                            ((2, 3, 4), True)):
        if abc_has_14_neighbours(A, B, C) != want:
            raise AssertionError(
                f"ABC ({A},{B},{C}): Remark 1.3 gives "
                f"{abc_has_14_neighbours(A, B, C)}, expected {want}")
    # the ball theorem additionally needs A = 1
    if abc_is_ball(2, 3, 4):
        raise AssertionError("ABC (2,3,4) has 14 neighbours but A = 2, "
                             "so Theorem 1.1 must not claim it")
    if not abc_is_ball(1, 2, 4):
        raise AssertionError("ABC (1,2,4) is the paper's own worked "
                             "example of a ball")
    balls = [k for k, v in RADIX_PRESETS.items()
             if isinstance(v[2], tuple) and abc_is_ball(*v[2])]
    print(f"Thuswaldner-Zhang Remark 1.3 reproduced; proven 3-balls "
          f"shipped: {', '.join(sorted(balls))}")

    # (1,2,8) factors as (x+2)(x^2-x+4), so all three eigenvalues have
    # modulus exactly 2 -- by Bandt Prop 2.2 that makes it conjugate to
    # a SELF-SIMILAR tile, and its level-k cells distort only
    # polynomially instead of exponentially
    M, D = RADIX_PRESETS['ABC_128'][1]
    ev = np.abs(np.linalg.eigvals(M.astype(float)))
    if float(ev.max() / ev.min()) > 1.0 + 1e-9:
        raise AssertionError(
            f"ABC (1,2,8) should have equal eigenvalue moduli, got "
            f"{np.sort(ev)}")
    asp = []
    for k in (4, 12):
        Ak = np.linalg.inv(np.linalg.matrix_power(M.astype(float), k))
        sv = np.linalg.svd(Ak, compute_uv=False)
        asp.append(float(sv[0] / sv[-1]))
    if asp[1] > 4.0 * asp[0]:
        raise AssertionError(
            f"ABC (1,2,8) cells went {asp[0]:.1f} -> {asp[1]:.1f} from "
            f"level 4 to 12; equal moduli should keep that slow")
    print(f"ABC (1,2,8): eigenvalue moduli all 2, cell aspect "
          f"{asp[0]:.1f} -> {asp[1]:.1f} over levels 4-12")

    # every preset's topology note must come from the tables, not thin
    # air -- and F and G must stay silent, since Bandt gives no details
    for key, v in RADIX_PRESETS.items():
        note = radix_topology(v[2])
        if key in ('TWIN_F', 'TWIN_G'):
            if note:
                raise AssertionError(
                    f"{v[0]}: Bandt provides no details for F and G, so "
                    f"nothing should be claimed")
        elif not note:
            raise AssertionError(f"{v[0]}: no topology note")
    print("topology notes present for every preset but F and G")

    # ---- 3b. the closed-form bounding box -----------------------------
    # the cube tile is [0,1]^3 exactly, which pins the support series
    M, D = RADIX_PRESETS['CUBE'][1]
    lo, hi = tile_support_bbox(M, D)
    if (float(np.max(np.abs(lo))) > 1e-12
            or float(np.max(np.abs(hi - 1.0))) > 1e-12):
        raise AssertionError(f"the cube tile's support box came out "
                             f"{lo} .. {hi}, expected [0,1]^3")
    # Twindragon A is Bandt's non-fractal case, and in this lattice
    # basis its tile is exactly the unit cube: M [0,1]^3 is the box
    # spanned by Me1 = e2, Me2 = e3, Me3 = 2e1, i.e. [0,2]x[0,1]x[0,1],
    # which is precisely [0,1]^3 union ([0,1]^3 + e1) = T + D.  Support
    # box [0,1]^3 together with volume 1 pins it down.
    M, D = RADIX_PRESETS['TWIN_A'][1]
    lo, hi = tile_support_bbox(M, D)
    if (float(np.max(np.abs(lo))) > 1e-12
            or float(np.max(np.abs(hi - 1.0))) > 1e-12):
        raise AssertionError(f"twindragon A's support box came out "
                             f"{lo} .. {hi}; it should be the unit "
                             f"cube")
    print("support series: cube and twindragon A both exactly [0,1]^3 "
          "(A is Bandt's non-fractal case)")

    # ---- 3c. gasket digit sets stay three-dimensional -----------------
    # dropping digits off the end must never leave them coplanar: the
    # cube's naive i,j,k order put all four i = 0 digits last, so a
    # four-hole gasket collapsed to a flat sheet
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        hmax = max_holes(M, D)
        for h in range(0, hmax + 1):
            keep = D[:len(D) - h]
            rk = attractor_rank(M, keep)
            if rk != 3:
                raise AssertionError(
                    f"{label}: {h} holes leaves an attractor of "
                    f"dimension {rk} -- it would collapse")
        if hmax + 1 <= len(D) - 2:
            over = D[:len(D) - (hmax + 1)]
            if attractor_rank(M, over) == 3:
                raise AssertionError(
                    f"{label}: {hmax + 1} holes still fills three "
                    f"dimensions, so the clamp is too tight")
    hmax_cube = max_holes(*RADIX_PRESETS['CUBE'][1])
    if hmax_cube < 4:
        raise AssertionError(
            f"the cube should support a 4-hole gasket, got {hmax_cube}")
    V, F, info = build_radix(preset='CUBE', holes=4, output='VOXEL',
                             resolution=48, points=120000)
    span = V.max(axis=0) - V.min(axis=0)
    if float(span.max() / max(span.min(), 1e-12)) > 4.0:
        raise AssertionError(
            f"the 4-hole cube gasket has aspect "
            f"{span.max() / span.min():.1f} -- it collapsed")
    print(f"gasket digit sets: rank 3 throughout, cube takes up to "
          f"{hmax_cube} holes")

    # ---- 3d. sampled tiles reach their true extent -------------------
    # this is the check the old per-step metric could not make: compare
    # against the closed-form support box, not against the last level
    # The tips of these tiles are reached only by rare addresses along
    # thin fibres -- Bandt says as much of cases F and G -- so no finite
    # sample reaches 100%.  What must hold is that the body stays
    # INSIDE the true tile and gets closer as the sample grows.
    for key in ('ABC_124', 'ABC_134', 'TWIN_A', 'TWIN_D', 'TWIN_G',
                'CUBE'):
        label = RADIX_PRESETS[key][0]
        V, F, info = build_radix(preset=key, output='VOXEL',
                                 resolution=64, points=200000)
        fid = info['fidelity']
        if fid > 1.05:
            raise AssertionError(
                f"{label}: the sampled body is {100 * fid:.0f}% of the "
                f"true extent -- it cannot exceed the tile")
        if fid < 0.75:
            raise AssertionError(
                f"{label}: the sampled tile reaches only "
                f"{100 * fid:.0f}% of its true extent")
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(f"{label}: {nb} boundary edges in the "
                                 f"sampled tile")
        print(f"{label:34s}: sampled to {100 * fid:5.1f}% of the true "
              f"extent, {info['cells']:6d} cells, closed")

    # more samples must get closer -- the check that the shortfall is
    # sampling and not a wrong bounding box
    lean = build_radix(preset='ABC_134', output='VOXEL', resolution=64,
                       points=60000)[2]['fidelity']
    rich = build_radix(preset='ABC_134', output='VOXEL', resolution=64,
                       points=600000)[2]['fidelity']
    if rich <= lean:
        raise AssertionError(
            f"ABC (1,3,4): fidelity went {lean:.3f} -> {rich:.3f} as "
            f"the sample grew tenfold; it should improve")
    print(f"sampling converges: ABC (1,3,4) {100 * lean:.1f}% -> "
          f"{100 * rich:.1f}% on a tenfold sample")

    # ---- 4. the exact level-k body, measured against the truth -------
    # The old check compared each level with the previous one, which is
    # far too weak: the remaining error is the tail of a geometric
    # series with ratio 1/min|lambda|, so for twindragon G a per-step
    # change of 9% hides a body that has reached barely a quarter of
    # the real tile.  Measure against the closed-form support box, and
    # assert the two things that actually matter -- the level-k body
    # must approach the tile from INSIDE and must improve with k.
    for key in ('ABC_124', 'TWIN_B', 'TWIN_G', 'CUBE'):
        label = RADIX_PRESETS[key][0]
        M, D = RADIX_PRESETS[key][1]
        true_span = np.subtract(*reversed(tile_support_bbox(M, D)))
        top = min(10, max_level(len(D)))
        fids = []
        for k in (3, top):
            S = radix_points(M, D, k)
            A = np.linalg.inv(np.linalg.matrix_power(M.astype(float),
                                                     k))
            P = S.astype(float) @ A.T
            span = P.max(axis=0) - P.min(axis=0)
            if float(np.max(span / true_span)) > 1.0 + 1e-9:
                raise AssertionError(
                    f"{label}: the level-{k} body sticks out past the "
                    f"true tile ({span} vs {true_span}) -- M^-k is "
                    f"wrong")
            fids.append(float(np.min(span / true_span)))
        if fids[-1] <= fids[0]:
            raise AssertionError(
                f"{label}: fidelity went {fids[0]:.3f} -> {fids[-1]:.3f} "
                f"as the level rose; it should improve")
        A = np.linalg.inv(np.linalg.matrix_power(M.astype(float), top))
        sv = np.linalg.svd(A, compute_uv=False)
        print(f"{label:34s}: level {top:2d} reaches "
              f"{100 * fids[-1]:5.1f}% of the true extent, cells "
              f"{sv[0] / sv[-1]:8.0f}:1")

    # the aspect blow-up is real and must be REPORTED, not hidden: at
    # its own default level twindragon G's cells are thousands to one
    V, F, info = build_radix(preset='TWIN_G', output='EXACT')
    if info['cell_aspect'] < 100.0:
        raise AssertionError(
            f"twindragon G cells came out {info['cell_aspect']:.0f}:1; "
            f"the laminate warning would never fire")
    if info['fidelity'] > 0.9:
        raise AssertionError(
            f"twindragon G exact mode reached {info['fidelity']:.2f} "
            f"of the true extent; the shortfall warning would never "
            f"fire")
    print(f"exact mode on twindragon G: cells "
          f"{info['cell_aspect']:.0f}:1 and only "
          f"{100 * info['fidelity']:.0f}% of the extent -- both "
          f"correctly flagged")

    # ---- 5. IFS maps are contractions -------------------------------
    for key, (label, fn, dim) in IFS_PRESETS.items():
        mp = fn()
        if dim == 2:
            # the fern's first map is singular (it flattens to the
            # stem), which is contractive but not invertible
            if not all(float(np.linalg.svd(A, compute_uv=False)[0])
                       < 1.0 for A, _, _ in mp):
                raise AssertionError(f"{label}: not every map is a "
                                     f"contraction")
        elif not contractive(mp):
            raise AssertionError(f"{label}: not every map is a "
                                 f"contraction")
    print(f"{len(IFS_PRESETS)} IFS presets: every map a contraction")

    # ---- 5b. the Bandt-Mai-Mesing constructions ---------------------
    # Each is a homothety toward a fixed point composed with a proper
    # rotation, so the linear part must be exactly ratio x orthogonal
    # with determinant +1 -- a reflection would be a different fractal.
    for key, r, count in (('BMM_SIERP', 2.0 / 3.0, 3),
                          ('BMM_TETRA', 0.6, 4),
                          ('BMM_CUBE', 0.625, 8)):
        mp = IFS_PRESETS[key][1]()
        if len(mp) != count:
            raise AssertionError(
                f"{key}: {len(mp)} maps, the paper gives {count}")
        for A, b, _p in mp:
            R = np.asarray(A) / r
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-12):
                raise AssertionError(f"{key}: the linear part is not "
                                     f"{r} times an orthogonal matrix")
            if abs(float(np.linalg.det(R)) - 1.0) > 1e-12:
                raise AssertionError(
                    f"{key}: det {np.linalg.det(R):+.3f} -- a rotation "
                    f"must be proper, a reflection is a different set")
        # each map must fix its own centre
        for A, b, _p in mp:
            fix = np.linalg.solve(np.eye(3) - np.asarray(A), b)
            if not np.all(np.isfinite(fix)):
                raise AssertionError(f"{key}: a map has no fixed point")
    print("Bandt-Mai-Mesing maps: proper rotations at ratios 2/3, 3/5, "
          "5/8 with the stated piece counts")

    # the triangle case must reproduce the paper's f_1 verbatim:
    #   f_1(x1,x2,x3) = (r x1 + 1 - r, -r x3, r x2)
    r = 2.0 / 3.0
    A1, b1, _ = _bmm_sierpinski(3, r)[0]
    x = np.array([0.3, -0.7, 0.2])
    want = np.array([r * x[0] + 1 - r, -r * x[2], r * x[1]])
    if not np.allclose(A1 @ x + b1, want, atol=1e-12):
        raise AssertionError(f"the Sierpinski-in-3D f_1 gives "
                             f"{A1 @ x + b1}, the paper gives {want}")
    print("Sierpinski-in-3D f_1 matches the paper's formula exactly")

    # Each construction is built around a 3-fold rotation, so its
    # attractor has to be invariant under that rotation -- a strong
    # check on the axes and the composition order, which a formula
    # comparison of f_1 alone would not catch.
    def _cloud_sig(P, n=28):
        P = np.asarray(P, dtype=float)
        P = (P - P.mean(axis=0))
        P = P / max(float(np.abs(P).max()), 1e-12)
        idx = np.clip(((P + 1.0) * 0.5 * n).astype(int), 0, n - 1)
        grid = np.zeros((n, n, n), dtype=bool)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return grid

    def _cloud_overlap(a, b):
        return float((a & b).sum()) / max(int((a | b).sum()), 1)

    tetra_axis = _BMM_TETRA_V.mean(axis=0) - _BMM_TETRA_V[0]
    for key, axis in (('BMM_SIERP', (0.0, 0.0, 1.0)),
                      ('BMM_TETRA', tetra_axis),
                      ('BMM_CUBE', (1.0, 1.0, 1.0))):
        P = chaos_game(IFS_PRESETS[key][1](), points=400000, seed=2,
                       transient=300)
        R = _rot(axis, 2.0 * math.pi / 3.0)
        turned = _cloud_overlap(_cloud_sig(P), _cloud_sig(P @ R.T))
        if turned < 0.85:
            raise AssertionError(
                f"{key} should be invariant under a 120 degree turn "
                f"about {np.round(np.asarray(axis, float), 2)}, but "
                f"the overlap is only {turned:.2f}")
        # a turn that is NOT a symmetry has to score much lower, or the
        # test would pass on any blob
        Rc = _rot(axis, math.radians(50.0))
        control = _cloud_overlap(_cloud_sig(P), _cloud_sig(P @ Rc.T))
        if control > 0.75 * turned:
            raise AssertionError(
                f"{key}: a 50 degree turn scores {control:.2f} against "
                f"{turned:.2f} for the real symmetry -- the test is "
                f"not discriminating")
    print("Bandt-Mai-Mesing attractors: all three invariant under "
          "their own 3-fold rotation")

    # "When A is centrally symmetric, as in Figures 9 and 5, it
    # coincides with its reverse."  The modified cube is that case; the
    # modified tetrahedron is not (the paper's Figure 1 IS the reverse
    # of its Figure 8, and looks quite different).  Compare occupancy,
    # not bounding boxes -- a set and its mirror share their extents.
    def _occupancy(V, n=24):
        idx = np.clip(((np.asarray(V) + 1.0) * 0.5 * n).astype(int),
                      0, n - 1)
        grid = np.zeros((n, n, n), dtype=bool)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return grid

    def _overlap(a, b):
        return float((a & b).sum()) / max(int((a | b).sum()), 1)

    same = {}
    for key in ('BMM_CUBE', 'BMM_TETRA', 'BMM_SIERP'):
        fwd = build_ifs(preset=key, output='VOXEL', points=300000,
                        resolution=48, seed=5)[0]
        rev = build_ifs(preset=key, output='VOXEL', points=300000,
                        resolution=48, seed=5, reverse=True)[0]
        same[key] = _overlap(_occupancy(fwd), _occupancy(rev))
    if same['BMM_CUBE'] < 0.6:
        raise AssertionError(
            f"the modified cube is centrally symmetric, so it should "
            f"coincide with its reverse, but they overlap only "
            f"{same['BMM_CUBE']:.2f}")
    for key in ('BMM_TETRA', 'BMM_SIERP'):
        if same[key] > 0.5 * same['BMM_CUBE']:
            raise AssertionError(
                f"{key} is not centrally symmetric, so its reverse "
                f"should differ, but they overlap {same[key]:.2f} "
                f"against the cube's {same['BMM_CUBE']:.2f}")
    print(f"reverse fractals: cube overlaps its reverse "
          f"{same['BMM_CUBE']:.2f} (centrally symmetric), tetrahedron "
          f"{same['BMM_TETRA']:.2f} and triangle "
          f"{same['BMM_SIERP']:.2f} (not)")

    # ---- 6. deterministic solid copies ------------------------------
    V, F, info = build_ifs(preset='SIERP_TETRA', output='SOLIDS',
                           depth=5)
    if info['copies'] != 4 ** 5:
        raise AssertionError(f"Sierpinski tetrahedron depth 5 made "
                             f"{info['copies']} copies, expected "
                             f"{4 ** 5}")
    if len(F) != 4 ** 5 * 4:
        raise AssertionError(f"expected {4 ** 5 * 4} faces, got "
                             f"{len(F)}")
    print(f"Sierpinski tetrahedron: {info['copies']} copies, "
          f"{len(F)} faces")

    V, F, info = build_ifs(preset='MENGER', output='SOLIDS',
                           seed_solid='CUBE', depth=3)
    if info['copies'] != 20 ** 3:
        raise AssertionError(f"Menger depth 3 made {info['copies']} "
                             f"copies, expected {20 ** 3}")
    print(f"Menger sponge (solid copies): {info['copies']} copies")

    # ---- 7. chaos game: determinism and watertight voxels -----------
    a = build_ifs(preset='SIERP_TETRA', output='VOXEL', points=60000,
                  resolution=48, seed=7)
    b = build_ifs(preset='SIERP_TETRA', output='VOXEL', points=60000,
                  resolution=48, seed=7)
    if (not np.array_equal(a[0], b[0])
            or not np.array_equal(np.asarray(a[1]), np.asarray(b[1]))):
        raise AssertionError("the chaos game is not reproducible from "
                             "its seed")
    c = build_ifs(preset='SIERP_TETRA', output='VOXEL', points=60000,
                  resolution=48, seed=8)
    if np.array_equal(a[0], c[0]):
        raise AssertionError("two different seeds gave an identical "
                             "mesh")
    nb, nm = edge_stats(a[1])
    if nb:
        raise AssertionError(f"voxel attractor: {nb} boundary edges")
    print(f"chaos game: reproducible from the seed, {len(a[1])} voxel "
          f"faces, closed ({nm} non-manifold edges)")

    # the attractor stays inside a bounded region
    P = chaos_game(IFS_PRESETS['SIERP_TETRA'][1](), points=20000,
                   seed=3)
    if float(np.max(np.abs(P))) > 3.0:
        raise AssertionError("the Sierpinski attractor escaped its "
                             "bounding box")

    # ---- 8. smooth contour ------------------------------------------
    V, F, info = build_ifs(preset='SIERP_TETRA', output='ISO',
                           points=200000, resolution=64, cover=0.9,
                           largest_only=False)
    if not len(F) or not np.all(np.isfinite(V)):
        raise AssertionError("the smooth contour came out empty or "
                             "non-finite")
    ext = float((V.max(axis=0) - V.min(axis=0)).max())
    if abs(ext - 2.0) > 1e-6:
        raise AssertionError(f"smooth contour is {ext:.4f} across, "
                             f"expected a 2 m fit")
    print(f"smooth contour: {len(F)} tris, level {info['level']:.1f}")

    # ---- 9. custom map parser ---------------------------------------
    mp = parse_maps("0.5 0 0 0 0.5 0 0 0 0.5 | 1 2 3 | 0.4; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | 0 0 0")
    if len(mp) != 2:
        raise AssertionError(f"parser made {len(mp)} maps, expected 2")
    if not np.allclose(mp[0][1], [1, 2, 3]) or mp[0][2] != 0.4:
        raise AssertionError("first custom map parsed wrongly")
    if mp[1][2] != 1.0:
        raise AssertionError("a missing probability should default "
                             "to 1")
    for bad in ("", "0.5 0 0 | 1 2 3", "1 2 3 4 5 6 7 8 9 | 1 2",
                "a b c d e f g h i | 1 2 3"):
        try:
            parse_maps(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_maps({bad!r}) should have raised")

    # ---- 10. planar systems, detected and meshed as reliefs ---------
    # A two-dimensional system has singular maps in R^3, so solid
    # copies would flatten the seed to a plate.  Planarity is MEASURED
    # from the attractor, not read off the preset label, so a custom
    # flat map set is handled the same way.
    planar = [k for k, v in IFS_PRESETS.items() if v[2] == 2]
    if not planar:
        raise AssertionError("no planar presets to test")
    for key in planar:
        label = IFS_PRESETS[key][0]
        mp = IFS_PRESETS[key][1]()
        if any(abs(float(np.linalg.det(A))) > 1e-12 for A, _, _ in mp):
            raise AssertionError(f"{label}: a planar system's maps "
                                 f"should be singular in 3-D")
        try:
            build_ifs(preset=key, output='SOLIDS', depth=3)
        except ValueError as e:
            if 'planar' not in str(e):
                raise AssertionError(
                    f"{label} was refused for solid copies, but "
                    f"unhelpfully: {e}")
        else:
            raise AssertionError(
                f"solid copies of {label} should have been refused")
        V, F, info = build_ifs(preset=key, output='RELIEF',
                               points=200000, plane_resolution=256)
        if not info.get('planar'):
            raise AssertionError(f"{label} was not detected as planar")
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(f"{label}: {nb} boundary edges -- the "
                                 f"relief slab is not closed")
        if _signed_volume(V, F) <= 0.0:
            raise AssertionError(f"{label}: the relief is inside out")
        span = V.max(axis=0) - V.min(axis=0)
        thin = float(np.min(span))
        wide = float(np.max(span))
        if thin > 0.05 * wide:
            raise AssertionError(
                f"{label} should be flat, but its thinnest axis spans "
                f"{thin:.3f} against {wide:.3f}")
        # embedded in the xz-plane, so y is the thin one
        if float(np.argmin(span)) != 1:
            raise AssertionError(
                f"{label} should be flat in y (upright in a z-up "
                f"world), but the thin axis is {int(np.argmin(span))}")
        print(f"{label:28s}: planar, closed relief, "
              f"{info['cells']:6d} cells, {thin / wide:.4f} thick")

    # a three-dimensional system must refuse the relief output
    try:
        build_ifs(preset='SIERP_TETRA', output='RELIEF')
    except ValueError as e:
        if 'planar' not in str(e):
            raise AssertionError(f"unhelpful refusal: {e}")
    else:
        raise AssertionError("a relief of a 3-D system should have "
                             "been refused")
    print("relief output refused for three-dimensional systems")

    # ---- 11. the Maps field round-trips every preset ----------------
    # Selecting a system loads its maps into the editable field, so
    # format_maps has to be an exact inverse of parse_maps or a preset
    # would quietly change the moment it was displayed.
    for key, (label, fn, dim) in IFS_PRESETS.items():
        mp = fn()
        back = parse_maps(format_maps(mp))
        if len(back) != len(mp):
            raise AssertionError(
                f"{label}: the field round-tripped {len(mp)} maps into "
                f"{len(back)}")
        for (A, b, pr), (A2, b2, pr2) in zip(mp, back):
            if (not np.allclose(A, A2, atol=1e-8)
                    or not np.allclose(b, b2, atol=1e-8)
                    or abs(pr - pr2) > 1e-8):
                raise AssertionError(
                    f"{label}: a map changed on its way through the "
                    f"Maps field")
    print(f"Maps field: all {len(IFS_PRESETS)} presets round-trip "
          f"exactly")

    # a non-contractive system must be refused, not silently diverge
    try:
        build_ifs(maps=parse_maps("2 0 0 0 2 0 0 0 2 | 0 0 0"),
                  output='SOLIDS', depth=2)
    except ValueError:
        pass
    else:
        raise AssertionError("an expanding map set should have been "
                             "refused")

    # and so must a digit set that is not a residue system
    try:
        build_radix(preset='CUSTOM',
                    custom=(companion(1, 2, 4),
                            np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0),
                                      (4, 0, 0)], dtype=np.int64)))
    except ValueError:
        pass
    else:
        raise AssertionError("digits 0 and 4 are congruent mod M; the "
                             "build should have been refused")
    print("parsers and validity checks reject what they should")

    print("RESULT: OK")
