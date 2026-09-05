# Soft Cells generator for Blender.
#
# A SOFT CELL fills space without gaps or overlaps while carrying the fewest
# possible sharp corners.  In the plane that minimum is two; in space it is
# ZERO -- a cell can tile all of space with no corners at all, every point of
# its boundary lying on some smooth curve of that boundary.  Such shapes turn
# out to be common in nature, in the chambers of seashells, in muscle tissue
# and in river estuaries, and rare in the geometry that had been written down
# before 2024.
#
# Two families are offered.
#
# ANALYTIC cells have closed formulas and need no solver.  The saddle prism is
# the square prism cut above and below by z = sqrt(1-x^2) - sqrt(1-y^2) +/- 1:
# the cap's slope becomes infinite at the wall, so it meets the wall
# tangentially instead of in an edge, and the corner simply is not there.
#
# MORPHOSPACE cells soften the truncated octahedron of the body-centred cubic
# lattice.  Its 24 vertices never move; what changes is the direction each
# edge leaves them in.  The entire family is fixed by a single unit vector --
# two angles -- so the whole design space is one sphere, and the named cells
# of the literature are marked points on it.  Two of them are second-order
# equivalent to the unit cells of the Schwarz P and Schwarz D minimal
# surfaces, and another is Kelvin's foam cell, so sweeping the two angles
# carries a minimal surface continuously into a foam.
#
# SOFTEN mode instead applies the general theorem that every locally
# polyhedral tiling can be completely softened, as an explicit displacement
# field around each node.  Its effect is deliberately local -- a polyhedron
# whose corners have melted, not a droplet -- because the theorem's
# deformation is supported only inside small balls about the nodes.
#
# References:
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells and the
#   geometry of seashells", PNAS Nexus 3(9):pgae311 (2024).
#   https://doi.org/10.1093/pnasnexus/pgae311
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells, Kelvin's
#   foam and the minimal surfaces of Schwarz", arXiv:2412.04491 (2025).
#   https://arxiv.org/abs/2412.04491
# - G. Domokos, A. G. Horvath and K. Regos, "A two-vertex theorem for normal
#   tilings", Aequationes Mathematicae 97(1):185-197 (2023) -- the saddle
#   prism, the first corner-free space-filler.
# - G. Ambrus and D. Dancso, "Softening locally polyhedral tilings",
#   arXiv:2604.18545 (2026) -- every locally polyhedral tiling of space can
#   be completely softened.  https://arxiv.org/abs/2604.18545
# - L. E. Dubins, "On curves of minimal length with a constraint on average
#   curvature, and with prescribed initial and terminal positions and
#   tangents", American Journal of Mathematics 79(3):497-516 (1957) -- the
#   minimum-curvature edge curves.
# - W. Thomson (Lord Kelvin), "On the division of space with minimum
#   partitional area", Phil. Mag. 24(151):503-514 (1887) -- the Kelvin cell.

import math

import numpy as np

try:
    from . import softcell
except ImportError:
    import softcell

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


_LABEL = {
    'SADDLE': "Saddle Prism Cell",
    'TRIPRISM': "Soft Triangular Prism",
    'HEXPRISM': "Soft Hexagonal Prism",
    'E2': "Truncated Octahedron",
    'F2': "Soft Cell (f2)",
    'G2': "Schwarz P Cell (g2)",
    'H2': "Soft Cell (h2)",
    'I2': "Schwarz D Cell (i2)",
    'KELVIN': "Kelvin Cell",
    'PD': "PD Cell",
    'CUSTOM': "Custom Direction",
}


if _IN_BLENDER:

    def _material(name, rgb):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.45
        return mat

    class MESH_OT_soft_cell_add(bpy.types.Operator):
        """Add a soft cell: a shape that fills space without gaps while
        carrying the fewest possible sharp corners -- in three dimensions,
        none at all"""
        bl_idname = "mesh.soft_cell_add"
        bl_label = "Soft Cells"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('CELLS', "Named Cells",
                    "Build one of the soft cells of the literature, or a "
                    "custom point of the morphospace"),
                   ('SOFTEN', "Soften Honeycomb",
                    "Melt the corners off a polyhedral honeycomb using the "
                    "general softening theorem")],
            default='CELLS')

        cell: EnumProperty(
            name="Cell",
            items=[
                ('SADDLE', "Saddle Prism Cell",
                 "Square prism cut by two saddle surfaces; no corners at "
                 "all, and the simplest corner-free space-filler known"),
                ('TRIPRISM', "Soft Triangular Prism",
                 "Triangular prism whose caps meet the walls tangentially"),
                ('HEXPRISM', "Soft Hexagonal Prism",
                 "Hexagonal prism whose caps meet the walls tangentially. "
                 "It cannot be both corner-free and space-filling -- see "
                 "Wall Scheme, which chooses which of the two you get"),
                None,
                ('E2', "Truncated Octahedron",
                 "The unsoftened polyhedron the family is built from: the "
                 "Voronoi cell of the body-centred cubic lattice"),
                ('F2', "Soft Cell (f2)",
                 "Standard soft cell -- all half-tangents at a node are "
                 "collinear.  Softness 0.331"),
                ('G2', "Schwarz P Cell (g2)",
                 "Non-standard soft cell, second-order equivalent to the "
                 "Voronoi cell of the Schwarz P minimal surface.  "
                 "Softness 0.333"),
                ('H2', "Soft Cell (h2)",
                 "Standard soft cell with tetrahedral symmetry.  "
                 "Softness 0.464"),
                ('I2', "Schwarz D Cell (i2)",
                 "Non-standard soft cell, second-order equivalent to the "
                 "Voronoi cell of the Schwarz D minimal surface.  "
                 "Softness 0.474"),
                ('KELVIN', "Kelvin Cell",
                 "Kelvin's dry-foam cell: edges meet at the Plateau angle. "
                 "The conjectured minimum-area partition of space into "
                 "equal cells OF A SINGLE SHAPE -- the two-cell "
                 "Weaire-Phelan structure does better"),
                ('PD', "PD Cell",
                 "The shortest path between the two Schwarz cells.  Not "
                 "soft: its half-tangents are never antiparallel"),
                ('CUSTOM', "Custom Direction",
                 "Any point of the morphospace, set by the two angles "
                 "below.  Every direction gives a valid space-filling cell"),
            ],
            default='F2')

        hex_scheme: EnumProperty(
            name="Wall Scheme",
            items=[
                ('ALTERNATE', "Soft (does not tile)",
                 "Arcs alternate up and down around the hexagon, so all "
                 "six vertices are smoothed -- but neighbouring cells then "
                 "disagree about the wall they share, and the cells do not "
                 "pack"),
                ('TILING', "Space-filling (two corners remain)",
                 "Arcs repeat with period three, so neighbours agree and "
                 "the cells pack -- at the cost of two of the six vertices "
                 "staying sharp.  A SOFTENED rather than a soft cell")],
            default='ALTERNATE')

        colatitude: FloatProperty(
            name="Colatitude", description="Angle of the generating "
            "direction from the vertical axis",
            default=math.pi / 2.0, min=0.0, max=math.pi, subtype='ANGLE')

        azimuth: FloatProperty(
            name="Azimuth", description="Angle of the generating direction "
            "about the vertical axis",
            default=math.pi / 4.0, min=-math.pi, max=math.pi,
            subtype='ANGLE')

        symmetry: EnumProperty(
            name="Symmetry",
            items=[('TETRAHEDRAL', "Tetrahedral",
                    "Always available: every direction admits it"),
                   ('OCTAHEDRAL', "Octahedral",
                    "The full symmetry of the polyhedron.  Only some "
                    "directions admit it; others fall back to tetrahedral")],
            default='TETRAHEDRAL')

        nx: IntProperty(name="Cells X", default=1, min=1, max=6)
        ny: IntProperty(name="Cells Y", default=1, min=1, max=6)
        nz: IntProperty(name="Cells Z", default=1, min=1, max=6)

        gap: FloatProperty(
            name="Gap Factor", description="Shrink each cell about its own "
            "centroid; at 1.0 neighbours share their walls exactly",
            default=0.92, min=0.05, max=1.0)

        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        edge_samples: IntProperty(
            name="Edge Samples", description="Points along each bent edge",
            default=16, min=4, max=64)

        face_rings: IntProperty(
            name="Face Rings", description="Grid rings spanning each face",
            default=8, min=2, max=24)

        face_style: EnumProperty(
            name="Face Style",
            items=[('MINIMAL', "Minimal Surface",
                    "Relax each face toward a minimal surface, as the "
                    "papers specify"),
                   ('RULED', "Ruled",
                    "Leave the raw spanning grid.  The mathematics fixes "
                    "only the edge tangents, so this is an equally valid "
                    "member of the same family -- and much faster")],
            default='MINIMAL')

        relax_iterations: IntProperty(
            name="Relax Iterations", default=40, min=0, max=200)

        two_materials: BoolProperty(name="Two Materials", default=True)

        separate_objects: BoolProperty(
            name="Separate Objects", description="Make each cell its own "
            "object sharing one mesh, instead of welding the whole block "
            "into a single mesh",
            default=False)

        shade_smooth: BoolProperty(
            name="Smooth Shading", description="Shade the curved faces "
            "smoothly.  Turn off to see the mesh facets, which is the "
            "honest view of how finely the cell is sampled",
            default=True)

        crease_angle: FloatProperty(
            name="Crease Angle", description="Edges meeting at more than "
            "this angle stay sharp instead of being smoothed over.  A soft "
            "cell is smooth ACROSS its nodes but still creased ALONG its "
            "edges, so blanket smooth shading rounds off detail that is "
            "really there",
            default=math.radians(25.0), min=0.0, max=math.pi,
            subtype='ANGLE')

        honeycomb: EnumProperty(
            name="Honeycomb",
            items=[('CUBIC', "Cubes",
                    "The cubic grid -- the worked example of the softening "
                    "theorem")],
            default='CUBIC')

        subdivisions: IntProperty(
            name="Subdivisions", description="Grid density on each face",
            default=12, min=4, max=32)

        bend_radius: FloatProperty(
            name="Bend Radius", description="Size of the bending "
            "neighbourhood around each node, as a fraction of the largest "
            "that keeps them disjoint",
            default=0.9, min=0.1, max=1.0)

        bend_depth: FloatProperty(
            name="Bend Depth", description="Strength of the displacement, "
            "as a fraction of the theorem's bound",
            default=1.0, min=0.0, max=1.0)

        resolution: IntProperty(
            name="Resolution", description="Samples around an analytic cell",
            default=24, min=6, max=64)

        def _shade(self, me):
            """Smooth shading with creases kept above the crease angle."""
            for p in me.polygons:
                p.use_smooth = self.shade_smooth
            if not self.shade_smooth:
                return
            import bmesh
            bm = bmesh.new()
            bm.from_mesh(me)
            for e in bm.edges:
                if len(e.link_faces) == 2:
                    if e.calc_face_angle(0.0) > self.crease_angle:
                        e.smooth = False
                else:
                    e.smooth = False
            bm.to_mesh(me)
            bm.free()
            me.update()

        def _build_separate(self, context):
            """One object per cell, all sharing a single mesh datablock.

            Cheaper than welding the block into one mesh and far easier to
            pull apart afterwards, which is the usual reason for wanting a
            packing as separate solids in the first place.
            """
            from mathutils import Matrix

            V, faces, info = softcell.build_cell(
                self.cell, phi=self.colatitude, theta=self.azimuth,
                symmetry=self.symmetry, edge_samples=self.edge_samples,
                face_rings=self.face_rings,
                relax_iters=(self.relax_iterations
                             if self.face_style == 'MINIMAL' else 0),
                face_style=self.face_style, resolution=self.resolution,
                hex_scheme=self.hex_scheme)
            places = softcell.cell_placements(
                self.cell, self.nx, self.ny, self.nz, info['basis'])

            cen = V.mean(axis=0)
            Vg = cen + (V - cen) * self.gap
            allp = np.vstack([Vg @ np.asarray(R, float).T + t
                              for R, t, _tag in places])
            lo, hi = allp.min(axis=0), allp.max(axis=0)
            span = float(np.max(hi - lo)) or 1.0
            mid = (lo + hi) / 2.0
            s = 2.0 * self.scale / span

            me = bpy.data.meshes.new("SoftCell")
            me.from_pydata([tuple(p) for p in Vg], [], list(faces))
            me.validate(clean_customdata=True)
            me.update()
            self._shade(me)
            if self.two_materials:
                # ONE slot, not two.  The mesh is shared by every object, so
                # a polygon's material_index is shared too and writing it
                # per object just overwrites the previous one -- which is
                # why all the cells came out the same colour.  A single
                # OBJECT-linked slot is the mechanism that lets identical
                # geometry carry different materials.
                me.materials.append(_material("Soft Cell A",
                                              (0.90, 0.55, 0.20)))

            for o in context.selected_objects:
                o.select_set(False)
            origin = context.scene.cursor.location
            last = None
            for R, t, tag in places:
                M = Matrix.Identity(4)
                Rm = np.asarray(R, float)
                for i in range(3):
                    for j in range(3):
                        M[i][j] = s * float(Rm[i][j])
                    M[i][3] = s * float(t[i] - mid[i]) + origin[i]
                obj = bpy.data.objects.new(
                    f"Soft Cell {_LABEL[self.cell]}", me)
                obj.matrix_world = M
                context.collection.objects.link(obj)
                if self.two_materials and obj.material_slots:
                    sl = obj.material_slots[0]
                    sl.link = 'OBJECT'
                    sl.material = _material(
                        "Soft Cell B", (0.22, 0.45, 0.72)) if tag else                         _material("Soft Cell A", (0.90, 0.55, 0.20))
                obj.select_set(True)
                last = obj
            if last is not None:
                context.view_layer.objects.active = last
            self.report({'INFO'},
                        f"{_LABEL[self.cell]}: {len(places)} separate cells, "
                        f"V={len(me.vertices)} F={len(me.polygons)} each")
            return {'FINISHED'}

        def execute(self, context):
            notes = []
            try:
                if self.mode == 'SOFTEN':
                    V, faces = softcell.warp.soften_cubic(
                        n=max(self.nx, self.ny, self.nz),
                        subdiv=self.subdivisions,
                        bend_radius=self.bend_radius,
                        depth=self.bend_depth)
                    tags = [0] * len(faces)
                    info = {}
                    label = "Softened Cubes"
                elif self.separate_objects and self.mode == 'CELLS':
                    return self._build_separate(context)
                else:
                    V, faces, tags, info = softcell.build_block(
                        self.cell, nx=self.nx, ny=self.ny, nz=self.nz,
                        gap=self.gap,
                        phi=self.colatitude, theta=self.azimuth,
                        symmetry=self.symmetry,
                        edge_samples=self.edge_samples,
                        face_rings=self.face_rings,
                        relax_iters=(self.relax_iterations
                                     if self.face_style == 'MINIMAL' else 0),
                        face_style=self.face_style,
                        resolution=self.resolution,
                        hex_scheme=self.hex_scheme)
                    label = _LABEL[self.cell]
                    if info.get('demoted'):
                        notes.append("octahedral symmetry unavailable for "
                                     "this direction, used tetrahedral")
            except (ValueError, RuntimeError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

            P = np.asarray(V, float)
            lo, hi = P.min(axis=0), P.max(axis=0)
            span = float(np.max(hi - lo)) or 1.0
            P = (P - (lo + hi) / 2.0) * (2.0 * self.scale / span)

            me = bpy.data.meshes.new("SoftCell")
            me.from_pydata([tuple(p) for p in P], [], list(faces))
            me.validate(clean_customdata=True)
            if (self.two_materials and self.mode == 'CELLS'
                    and len(me.polygons) == len(tags) and max(tags) > 0):
                me.materials.append(_material("Soft Cell A",
                                              (0.90, 0.55, 0.20)))
                me.materials.append(_material("Soft Cell B",
                                              (0.22, 0.45, 0.72)))
                me.polygons.foreach_set('material_index', tags)
            me.update()
            # A soft cell is smooth across its NODES -- that is the whole
            # point -- but its faces still meet along its EDGES at a real
            # dihedral angle, and the polyhedral (e2) cell is creased
            # everywhere.  Smoothing the lot rounds away geometry that is
            # genuinely there, so creases are kept by angle.
            self._shade(me)

            obj = bpy.data.objects.new(f"Soft Cell {label}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            msg = f"{label}: V={len(me.vertices)} F={len(me.polygons)}"
            if info.get('area'):
                msg += f" area={info['area']:.3f}"
            if notes:
                msg += " -- " + "; ".join(notes)
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'SOFTEN':
                for k in ('honeycomb', 'nx', 'subdivisions',
                          'bend_radius', 'bend_depth', 'scale',
                          'shade_smooth'):
                    lay.prop(self, k)
                if self.shade_smooth:
                    lay.prop(self, 'crease_angle')
                return
            lay.prop(self, 'cell')
            if self.cell == 'CUSTOM':
                lay.prop(self, 'colatitude')
                lay.prop(self, 'azimuth')
                lay.prop(self, 'symmetry')
            if self.cell == 'HEXPRISM':
                lay.prop(self, 'hex_scheme')
            if self.cell in softcell.ANALYTIC:
                lay.prop(self, 'resolution')
            else:
                lay.prop(self, 'edge_samples')
            lay.prop(self, 'face_rings')
            lay.prop(self, 'face_style')
            if self.face_style == 'MINIMAL':
                lay.prop(self, 'relax_iterations')
            for k in ('nx', 'ny', 'nz', 'gap', 'scale', 'separate_objects',
                      'two_materials', 'shade_smooth'):
                lay.prop(self, k)
            if self.shade_smooth:
                lay.prop(self, 'crease_angle')

    def _menu_func(self, context):
        self.layout.operator("mesh.soft_cell_add", icon='META_BALL')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_soft_cell_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_soft_cell_add)


def _selftest():
    """Delegate to the engine package, then check the front-end's own
    invariants: every enum item the operator offers must actually build."""
    softcell._selftest()

    for kind in softcell.ANALYTIC:
        V, F, info = softcell.build_cell(kind, resolution=12, face_rings=3)
        assert len(V) and len(F), kind
    for kind in softcell.MORPHOSPACE:
        V, F, info = softcell.build_cell(kind, edge_samples=6, face_rings=3,
                                         relax_iters=0, face_style='RULED')
        assert len(V) and len(F), kind
    print(f"operator: all {len(softcell.ANALYTIC)} analytic and "
          f"{len(softcell.MORPHOSPACE)} morphospace cells build  OK")

    # a Custom direction must build under tetrahedral symmetry for any
    # angles, and must fall back rather than fail under octahedral
    rng = np.random.default_rng(2)
    demoted = 0
    for _ in range(12):
        phi = math.acos(rng.uniform(-1.0, 1.0))
        th = rng.uniform(-math.pi, math.pi)
        V, F, info = softcell.build_cell(
            'CUSTOM', phi=phi, theta=th, symmetry='OCTAHEDRAL',
            edge_samples=6, face_rings=3, relax_iters=0, face_style='RULED')
        assert len(V) and len(F)
        demoted += bool(info.get('demoted'))
    print(f"operator: 12 random Custom directions build; {demoted} needed "
          f"the tetrahedral fallback  OK")

    print("soft_cell_generator standalone tests passed")
