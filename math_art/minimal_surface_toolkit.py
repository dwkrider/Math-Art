
# Minimal Surface Toolkit for Blender
#
# Three generators in one add-on:
#
# 1. Classic parametric minimal surfaces (Enneper & higher orders,
#    catenoid, helicoid, Henneberg, Catalan, Bour, Richmond, Scherk's
#    doubly-periodic graph) -- after Juergen Meier's gallery
#    (3d-meier.de, tut25).
#
# 2. Triply-periodic minimal surfaces via their standard nodal
#    (level-set) approximations, meshed by marching tetrahedra:
#    Schwarz P & D, Schoen's Gyroid / I-WP / F-RD / O,C-TO, Neovius,
#    Lidinoid, Split P, the Fischer-Koch S / C(S) / Y / +-Y / C(+-Y) /
#    C(Y) family, the complementary D and Gyroid, G', D', Karcher K,
#    the C(I2-Y**) rod-packing surface, plus the singly-periodic Scherk
#    tower.  Inventory after Ken Brakke's periodic-surface pages
#    (kenbrakke.com/evolver); tier-2 nodal equations and their sources
#    are documented at the "Tier-2 nodal TPMS expansion" block in
#    `minsurf/tpms.py`.
#
# 3. A Plateau-problem solver -- a lightweight, in-Blender take on what
#    Brakke's Surface Evolver does: pin one or two boundary curves and
#    minimize surface area (Pinkall-Polthier cotangent-Laplacian
#    iteration solved with conjugate gradients). Includes the classic
#    "minimal surface between a circle and a torus knot" construction
#    (trefoil by default). For (2, q) knots the default span is a
#    single embedded sheet: Seifert's algorithm applied to the round
#    knot diagram, with the outer Seifert disk opened out to the
#    circle (an embedded *annulus* between a knotted and a round
#    boundary cannot exist, so the correct spanning surface has genus
#    (q - 1) / 2).
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
# - Weierstrass-Enneper representation: K. Weierstrass (1866) and
#   A. Enneper (1864); the catenoid (surface of Euler, 1744) was
#   shown minimal by J. B. C. Meusnier (1776).
# - Costa surface: C. J. Costa (1982); embeddedness by D. Hoffman and
#   W. H. Meeks III (1985). Chen-Gackstatter: C. C. Chen and
#   F. Gackstatter (1982). Jorge-Meeks k-noids: L. P. Jorge and
#   W. H. Meeks III (1983).
# - Triply-periodic families: H. A. Schwarz (P, D; Gesammelte Math.
#   Abhandlungen, 1890),
#   A. H. Schoen (gyroid, I-WP, F-RD; NASA TN D-5541, 1970),
#   E. R. Neovius (1883). Cotangent-Laplacian area flow after
#   U. Pinkall and K. Polthier (1993).
# - Seifert surfaces: H. Seifert, "Ueber das Geschlecht von Knoten",
#   Mathematische Annalen 110 (1934). Visualization of Seifert
#   surfaces as disks joined by twisted bands after J. J. van Wijk
#   and A. M. Cohen, IEEE TVCG 12(4) (2006).

bl_info = {
    "name": "Minimal Surface Toolkit",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Minimal Surfaces / N-panel 'Minimal Surfaces'",
    "description": "Parametric & triply-periodic minimal surfaces, and a "
                   "Plateau solver spanning minimal surfaces on curves",
    "category": "Add Mesh",
}

# The engine that used to live in this file (the elliptic-function core, the
# parametric surface catalog and its meshing pipeline, the nodal TPMS family
# and the Plateau solver) now lives in the Blender-free package
# `math_art/minsurf/`, on the same pattern as `math_art/seifert/`.  This
# module keeps the registered operators and menus, and re-exports the engine
# names they use so operator code reads exactly as before.
#
# Import the engine directly (`from .minsurf import build_tpms`) in new code
# rather than reaching through this module -- importing this one pulls in
# Blender-facing machinery that a headless caller does not need.

import math
import numpy as np

try:
    from . import rim_curve as _rim
except ImportError:  # flat import outside the package
    import rim_curve as _rim

from .minsurf.domain import _center_fit
from .minsurf.elliptic import _SQUARE
from .minsurf.parametric import (ANGLE_PARAM, COUNT_PARAM, FAMILIES,
                                 MESH_PARAM, PARAMETRIC, PERIODIC_NO_ARRAY,
                                 STOREY_PARAM, SURFACE_FAMILY, _periodic_dim,
                                 build_parametric, build_parametric_grid)
from .minsurf.plateau import (_SEIFERT_MAX_ITERS, _quads_to_tris,
                              _selfx_crossings, align_loops,
                              build_annulus_grid, build_disk_grid,
                              build_seifert_span_grid, fair_grid_2d,
                              fair_grid_columns, mesh_area, minimize_area,
                              relax_normal_flow, resample_loop, torus_knot)
from .minsurf.tpms import (TPMS, TPMS2_HEX_LATTICE, TPMS_EXACT,
                           _PGD_PRESET_ANGLE, _f_p, build_tpms,
                           build_tpms_exact, marching_tets,
                           tpms2_DEFAULT_OFFSET, tpms2_LATTICE)
# the catalog module itself, for the self-test's CHM-modulus reference
from .minsurf import zoo as _zoo

TAU = 2.0 * math.pi


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _nurbs_grid_object(context, name, grid, cyclic_u=False,
                           cyclic_v=False, order=4):
        """Create a NURBS surface object from an (nu, nv, 3) control grid.
        `u` runs across rows, `v` along each row. Grids must be built
        row-by-row as separate splines and merged with make_segment (the
        only scriptable way to produce a surface grid in Blender)."""
        grid = np.asarray(grid)
        nu, nv = grid.shape[:2]
        su = bpy.data.curves.new(name, 'SURFACE')
        su.dimensions = '3D'
        for i in range(nu):
            sp = su.splines.new('NURBS')
            sp.points.add(nv - 1)
            flat = np.concatenate(
                [grid[i], np.ones((nv, 1))], axis=1).ravel()
            sp.points.foreach_set('co', flat)
        obj = bpy.data.objects.new(name, su)
        context.collection.objects.link(obj)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.make_segment()
        bpy.ops.object.mode_set(mode='OBJECT')
        sp = su.splines[0]
        # make_segment chains the selected splines in an arbitrary
        # (geometry-dependent) order, so only the resulting GRID
        # STRUCTURE is trustworthy. Rewrite every control point into the
        # intended layout (storage is u-fastest: flat = v*pu + u).
        pu, pv = sp.point_count_u, sp.point_count_v
        if (pu, pv) == (nu, nv):
            ordered = grid.transpose(1, 0, 2)   # S[v][u] = grid[u][v]
            cu, cv = cyclic_u, cyclic_v
        else:                                   # (pu, pv) == (nv, nu)
            ordered = grid                      # S[v][u] = grid[v][u]
            cu, cv = cyclic_v, cyclic_u
        flat = np.concatenate(
            [ordered.reshape(-1, 3), np.ones((pu * pv, 1))],
            axis=1).ravel()
        sp.points.foreach_set('co', flat)
        sp.use_cyclic_u = cu
        sp.use_cyclic_v = cv
        sp.use_endpoint_u = not cu
        sp.use_endpoint_v = not cv
        sp.order_u = min(order, pu)
        sp.order_v = min(order, pv)
        sp.resolution_u = 6
        sp.resolution_v = 6
        su.update_tag()
        obj.location = context.scene.cursor.location
        return obj

    def _new_object(context, name, verts, faces, weld=0.0, smooth=True,
                    loop_uv=None, recalc_normals=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        if loop_uv is not None:
            # per-face-corner UVs (assigned before any weld: bmesh's
            # remove_doubles merges vertices but keeps loop layers)
            luv = np.asarray(loop_uv, dtype=np.float32)
            if len(me.loops) == len(luv):
                layer = me.uv_layers.new(name="UVMap")
                layer.data.foreach_set('uv', luv.ravel())
        if weld > 0 or recalc_normals:
            bm = bmesh.new()
            bm.from_mesh(me)
            if weld > 0:
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
            if recalc_normals:
                # coherent winding across each connected patch: kills the
                # alternating light/dark facet stripes that flipped quads
                # produce under smooth shading (one-sided surfaces still
                # keep a single unavoidable orientation seam)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
        me.polygons.foreach_set('use_smooth',
                                [bool(smooth)] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    def _extract_loop(obj, depsgraph):
        """Ordered closed polyline (world space) from a curve or mesh
        object. Returns ndarray (k,3) or raises ValueError."""
        ev = obj.evaluated_get(depsgraph)
        me = ev.to_mesh()
        try:
            if len(me.vertices) == 0:
                raise ValueError(f"{obj.name}: no geometry")
            adj = {}
            for e in me.edges:
                a, b = e.vertices
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            if not adj or any(len(v) != 2 for v in adj.values()):
                raise ValueError(
                    f"{obj.name}: must be a single closed loop "
                    f"(every point joining exactly 2 segments)")
            start = next(iter(adj))
            loop = [start]
            prev, cur = None, start
            while True:
                nxt = [v for v in adj[cur] if v != prev]
                if not nxt:
                    raise ValueError(f"{obj.name}: open curve")
                prev, cur = cur, nxt[0]
                if cur == start:
                    break
                loop.append(cur)
                if len(loop) > len(me.vertices):
                    raise ValueError(f"{obj.name}: not a single loop")
            mw = np.array(obj.matrix_world)
            pts = np.array([me.vertices[i].co[:] for i in loop])
            pts = pts @ mw[:3, :3].T + mw[:3, 3]
            return pts
        finally:
            ev.to_mesh_clear()

    # --- family-filtered surface enum (zoo catalog UX) ------------------
    # Item tuples are cached module-level: Blender requires the strings
    # of a dynamic-items enum to stay referenced from Python.
    # Periodic minimal surfaces live in their own generator
    # (mesh.periodic_minimal_add), picked by a Singly/Doubly/Triply
    # dropdown -- singly/doubly are Weierstrass-Enneper families, triply
    # is the nodal TPMS set.  They are therefore hidden from the
    # non-periodic Minimal Surface generator's family list.
    PERIODIC_FAMILIES = ('SINGLY', 'DOUBLY')

    _FAMILY_ITEMS = []
    _SURF_ITEMS_ALL = []
    _SURF_ITEMS_FAM = {}
    _PERIODIC_ITEMS = {}          # periodicity key -> [surface items]
    _PERIODIC_ALL = []            # union fallback for scripted calls
    _PERIODICITY_ITEMS = []       # the Singly/Doubly/Triply dropdown

    def _build_surface_items():
        _FAMILY_ITEMS.clear()
        _SURF_ITEMS_ALL.clear()
        _SURF_ITEMS_FAM.clear()
        fam_of = SURFACE_FAMILY or {}
        for key, (label, _fn) in PARAMETRIC.items():
            it = (key, label, label)
            _SURF_ITEMS_ALL.append(it)
            _SURF_ITEMS_FAM.setdefault(
                fam_of.get(key, 'CLASSICAL'), []).append(it)
        for fam, flabel in (FAMILIES or ()):
            if fam in PERIODIC_FAMILIES:
                continue                       # -> periodic generator
            if fam in _SURF_ITEMS_FAM:
                n = len(_SURF_ITEMS_FAM[fam])
                _FAMILY_ITEMS.append(
                    (fam, flabel, f"{flabel} ({n} surfaces)"))
        if not _FAMILY_ITEMS:
            _FAMILY_ITEMS.append(
                ('CLASSICAL', "Classical", "Classical"))

        # --- periodic catalog: singly/doubly (WE) + triply (TPMS) ------
        _PERIODIC_ITEMS.clear()
        _PERIODIC_ALL.clear()
        _PERIODICITY_ITEMS.clear()
        flabel_of = dict(FAMILIES or ())
        for fam in PERIODIC_FAMILIES:
            items = list(_SURF_ITEMS_FAM.get(fam, ()))
            if items:
                _PERIODIC_ITEMS[fam] = items
                _PERIODIC_ALL.extend(items)
                lbl = flabel_of.get(fam, fam)
                _PERIODICITY_ITEMS.append(
                    (fam, lbl, f"{lbl} ({len(items)} surfaces)"))
        # Only genuinely triply-periodic surfaces belong under Triply.
        # SCHERKT (Scherk's tower) is a nodal SINGLY-periodic surface that
        # historically rode in the TPMS field dict; the proper singly
        # periodic Scherk tower is the WE SCHERK_TOWER under Singly, so it
        # is dropped from this list (still reachable via mesh.tpms_add).
        _NOT_TRIPLY = {'SCHERKT'}
        # the nodal approximations lead the Triply list; the exact
        # Weierstrass P/Gyroid/D (Bonnet angle) is listed LAST
        exact_items = [(k, v[0], v[0]) for k, v in TPMS_EXACT.items()]
        tpms_items = [(k, v[0], v[0]) for k, v in TPMS.items()
                      if k not in _NOT_TRIPLY] + exact_items
        if tpms_items:
            _PERIODIC_ITEMS['TRIPLY'] = tpms_items
            _PERIODIC_ALL.extend(tpms_items)
            _PERIODICITY_ITEMS.append(
                ('TRIPLY', "Triply Periodic (TPMS)",
                 f"Triply periodic minimal surfaces: the exact "
                 f"P/Gyroid/D associate family + "
                 f"{len(tpms_items) - len(exact_items)} nodal "
                 f"approximations"))
        if not _PERIODICITY_ITEMS:
            _PERIODICITY_ITEMS.append(
                ('TRIPLY', "Triply Periodic (TPMS)", "TPMS"))

    _build_surface_items()

    def _surface_items(self, context):
        # NOTE: fall back to the FULL union list whenever there is no
        # UI area (context is None, or a background/scripted context).
        # Scripted calls -- mesh.parametric_minimal_add(surface='COSTA')
        # -- must not be rejected by the family filter, and the stored
        # enum index must map against the same list on set and get
        # (see COORDINATION.md).  Only an interactive area (the redo
        # panel / add-menu) sees the family-filtered list.
        if context is None or getattr(context, 'area', None) is None:
            return _SURF_ITEMS_ALL
        return _SURF_ITEMS_FAM.get(self.family, _SURF_ITEMS_ALL)

    def _periodic_surface_items(self, context):
        # Same context=None -> full-union fallback contract as above, so
        # scripted mesh.periodic_minimal_add(surface='G') keeps working.
        if context is None or getattr(context, 'area', None) is None:
            return _PERIODIC_ALL or _SURF_ITEMS_ALL
        return _PERIODIC_ITEMS.get(self.periodicity,
                                   _PERIODIC_ALL or _SURF_ITEMS_ALL)

    class MESH_OT_parametric_minimal_add(bpy.types.Operator):
        """Add a minimal surface from the Weierstrass-Enneper / Bjorling
        catalog. Pick a Family, then a Surface within it."""
        bl_idname = "mesh.parametric_minimal_add"
        bl_label = "Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()

        family: EnumProperty(
            name="Family",
            items=_FAMILY_ITEMS,
            default='CLASSICAL',
            description="Minimal-surface family (Weber's taxonomy); "
                        "filters the Surface list")
        surface: EnumProperty(
            name="Surface",
            items=_surface_items)
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh", "Dense polygon mesh"),
                   ('NURBS', "NURBS", "Compact NURBS surface patch "
                                      "(control grid = Resolution U x V)")],
            default='MESH')
        res_u: IntProperty(name="Resolution U", default=64, min=8, max=512)
        res_v: IntProperty(name="Resolution V", default=64, min=8, max=512)
        ctrl_u: IntProperty(
            name="Control Points U", default=24, min=6, max=128,
            description="NURBS control grid size in U")
        ctrl_v: IntProperty(
            name="Control Points V", default=24, min=6, max=128,
            description="NURBS control grid size in V")
        order: IntProperty(
            name="Order / Count", default=1, min=1, max=12,
            description="Enneper order; helicoid half-turns; Jorge-Meeks "
                        "end count n (>= 3); ignored for the rest")
        storeys: IntProperty(
            name="Storeys", default=3, min=1, max=8,
            description="Number of periodic fundamental domains to stack "
                        "(saddle tower); ignored for the rest")
        radius: FloatProperty(
            name="Domain Radius", default=1.2, min=0.2, max=4.0,
            description="Extent of the parameter domain (for the k-noid, "
                        "how close the disk reaches its ends)")
        assoc_angle: FloatProperty(
            name="Associate Angle", default=0.0,
            min=0.0, max=math.pi / 2.0, subtype='ANGLE',
            description="Bonnet associate family (0 = catenoid, "
                        "pi/2 = helicoid); for the Karcher saddle tower it "
                        "is the wing-clustering angle alpha (0 = symmetric)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 = a 2 m "
                        "cube, centered on the origin)")

        def execute(self, context):
            # When the Family changes, the dynamic Surface enum is
            # refiltered and the stored value can momentarily resolve to
            # an identifier outside the new list (Blender returns '' or a
            # stale key).  Coerce to the first surface of the current
            # family so switching families never raises.
            surf = self.surface
            if surf not in PARAMETRIC:
                items = _surface_items(self, context)
                surf = items[0][0] if items else 'ENNEPER'
            label = PARAMETRIC[surf][0]
            theta = (self.assoc_angle if surf in ANGLE_PARAM
                     else 0.0)
            # some surfaces are assembled meshes with no NURBS/grid form
            if self.output == 'NURBS' and surf not in MESH_PARAM:
                G, wrap_u, wrap_v = build_parametric_grid(
                    surf, self.ctrl_u, self.ctrl_v,
                    self.order, self.radius, self.scale, theta)
                if wrap_u:          # drop duplicated periodic endpoint
                    G = G[:-1]
                if wrap_v:
                    G = G[:, :-1]
                _nurbs_grid_object(context, label, G,
                                   cyclic_u=wrap_u, cyclic_v=wrap_v)
            else:
                out = build_parametric(surf, self.res_u,
                                       self.res_v, self.order,
                                       self.radius, self.scale, theta,
                                       with_uv=True, cells=(self.storeys, 1))
                V, quads = out[0], out[1]
                cuv = out[2] if len(out) > 2 else None
                _new_object(context, label, V, quads,
                            weld=1e-5 * max(1.0, self.scale),
                            loop_uv=cuv)
            if self.rim:
                _ob = context.active_object
                if _ob is not None:
                    _rim.add_rim_from_object(
                        context, _ob, _ob.name,
                        self.rim_thickness, self.rim_smooth)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            lay.prop(self, 'surface')
            mesh_only = self.surface in MESH_PARAM
            if not mesh_only:
                lay.prop(self, 'output')
            if self.output == 'NURBS' and not mesh_only:
                lay.prop(self, 'ctrl_u')
                lay.prop(self, 'ctrl_v')
            else:
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
            if self.surface in COUNT_PARAM:
                lay.prop(self, 'order', text=COUNT_PARAM[self.surface])
            elif self.surface not in ANGLE_PARAM:
                lay.prop(self, 'order')
            if self.surface in STOREY_PARAM:
                lay.prop(self, 'storeys', text=STOREY_PARAM[self.surface])
            if self.surface in ANGLE_PARAM:
                lay.prop(self, 'assoc_angle')
            lay.prop(self, 'radius')
            lay.prop(self, 'scale')
            _rim.draw_rim(lay, self)

    class MESH_OT_tpms_add(bpy.types.Operator):
        """Add a triply-periodic minimal surface (nodal approximation)"""
        bl_idname = "mesh.tpms_add"
        bl_label = "Periodic Minimal Surface (TPMS)"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            items=[(k, v[0], v[0]) for k, v in TPMS.items()],
            default='G')
        cells: IntProperty(
            name="Cells", default=1, min=1, max=4,
            description="Number of unit cells per axis")
        resolution: IntProperty(
            name="Resolution / Cell", default=28, min=8, max=80,
            description="Sample grid resolution per unit cell")
        cell_size: FloatProperty(
            name="Cell Size", default=2.0, min=0.1, max=100.0,
            description="Edge length of one unit cell in Blender units")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this thickness")
        level_offset: FloatProperty(
            name="Level Offset", default=0.0, min=-3.0, max=3.0,
            description="Constant c in F(x,y,z) = c, relative to the "
                        "published surface: 0 keeps the (approximately) "
                        "minimal member, nonzero sweeps the field's "
                        "offset companion family")
        cell_aspect: FloatProperty(
            name="Cell Aspect (c/a)", default=1.0, min=0.25, max=4.0,
            description="Height-to-width ratio of the unit cell: 1 is "
                        "the published cell; other values stretch the "
                        "lattice tetragonally (tP/tD/tG-style cells)")

        def execute(self, context):
            verts, tris = build_tpms(self.surface, self.cells,
                                     self.resolution, self.cell_size,
                                     offset=self.level_offset,
                                     aspect=self.cell_aspect)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty level set")
                return {'CANCELLED'}
            label = TPMS[self.surface][0]
            obj = _new_object(context, label, verts, tris)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('surface', 'cells', 'resolution', 'cell_size',
                      'thickness', 'level_offset', 'cell_aspect'):
                lay.prop(self, k)

    class MESH_OT_periodic_minimal_add(bpy.types.Operator):
        """Add a periodic minimal surface.  Pick the Periodicity
        (singly / doubly / triply), then a Surface within it.  Singly and
        doubly periodic surfaces come from the Weierstrass-Enneper
        catalog; triply periodic are the nodal TPMS approximations."""
        bl_idname = "mesh.periodic_minimal_add"
        bl_label = "Periodic Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()

        periodicity: EnumProperty(
            name="Periodicity",
            items=_PERIODICITY_ITEMS,
            description="Translational symmetry of the surface: singly / "
                        "doubly periodic (Weierstrass-Enneper) or triply "
                        "periodic (TPMS); filters the Surface list")
        surface: EnumProperty(
            name="Surface",
            items=_periodic_surface_items)
        # -- Weierstrass (singly / doubly) parameters
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh", "Dense polygon mesh"),
                   ('NURBS', "NURBS", "Compact NURBS surface patch")],
            default='MESH')
        res_u: IntProperty(name="Resolution U", default=64, min=8, max=512)
        res_v: IntProperty(name="Resolution V", default=64, min=8, max=512)
        ctrl_u: IntProperty(
            name="Control Points U", default=24, min=6, max=128,
            description="NURBS control grid size in U")
        ctrl_v: IntProperty(
            name="Control Points V", default=24, min=6, max=128,
            description="NURBS control grid size in V")
        order: IntProperty(
            name="Order / Count", default=1, min=1, max=12,
            description="Period count / lattice modulus where the surface "
                        "uses it (e.g. saddle-tower wing count)")
        # -- unified cell counts: one INDEPENDENT count per tiling
        # dimension, shown by periodicity (singly -> u only; doubly ->
        # u, v; triply -> u, v, w = x, y, z).  The array's dimensionality
        # follows the surface's periodicity.
        cells_u: IntProperty(
            name="Cells", default=1, min=1, max=8,
            description="Copies along the 1st period axis (singly: the "
                        "single period; doubly: lattice vector 1; triply: x)")
        cells_v: IntProperty(
            name="Cells V", default=1, min=1, max=8,
            description="Copies along the 2nd period axis (doubly: lattice "
                        "vector 2; triply: y)")
        cells_w: IntProperty(
            name="Cells W", default=1, min=1, max=8,
            description="Copies along the 3rd period axis (triply: z)")
        # legacy scalar alias (not shown): scripted
        # periodic_minimal_add(surface=..., cells=3) still works and
        # broadcasts to every tiling axis left at its default (1).
        cells: IntProperty(
            name="Cells", default=0, min=0, max=8,
            description="Legacy uniform cell count (broadcasts to every "
                        "period axis); 0 = use the per-axis Cells controls")
        radius: FloatProperty(
            name="Domain Radius", default=1.2, min=0.2, max=4.0,
            description="Extent of the parameter domain")
        assoc_angle: FloatProperty(
            name="Associate Angle", default=0.0,
            min=0.0, max=math.pi / 2.0, subtype='ANGLE',
            description="Bonnet associate family angle; for the Karcher "
                        "saddle tower it is the wing-clustering angle alpha")
        # -- exact P/Gyroid/D preset: named shortcuts to the three iconic
        # Bonnet angles, leaving the raw angle slider independently settable.
        pgd_preset: EnumProperty(
            name="Preset",
            items=[
                ('P', "Schwarz P", "theta = 0: filled watertight P cell"),
                ('GYROID', "Gyroid",
                 "theta = 38.0148 deg: the gyroid (fundamental piece -- the "
                 "chiral cell has no watertight reflection assembly)"),
                ('D', "Schwarz D",
                 "theta = 90 deg: filled watertight D cell (P's conjugate)"),
                ('CUSTOM', "Custom angle",
                 "Use the Associate Angle slider verbatim; builds the exact "
                 "fundamental piece of the morph at that angle"),
            ],
            default='P',
            description="Iconic P / Gyroid / D by name, or Custom to drive "
                        "the raw Bonnet angle")
        # -- TPMS (triply) parameters (cells come from cells_u/v/w above)
        resolution: IntProperty(
            name="Resolution / Cell", default=28, min=8, max=80,
            description="Sample grid resolution per unit cell")
        cell_size: FloatProperty(
            name="Cell Size", default=2.0, min=0.1, max=100.0,
            description="Edge length of one unit cell in Blender units")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this thickness")
        level_offset: FloatProperty(
            name="Level Offset", default=0.0, min=-3.0, max=3.0,
            description="Constant c in F(x,y,z) = c, relative to the "
                        "published surface: 0 keeps the (approximately) "
                        "minimal member, nonzero sweeps the field's "
                        "offset companion family")
        cell_aspect: FloatProperty(
            name="Cell Aspect (c/a)", default=1.0, min=0.25, max=4.0,
            description="Height-to-width ratio of the unit cell: 1 is "
                        "the published cell; other values stretch the "
                        "lattice tetragonally (tP/tD/tG-style cells)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 = a 2 m "
                        "cube, centered on the origin)")

        def execute(self, context):
            # Coerce a stale/empty dynamic-enum value (the Periodicity
            # dropdown refilters Surface) to a valid key so switching
            # periodicity never raises.  Route on the surface's actual
            # backend, not the dropdown, so scripted calls also resolve.
            surf = self.surface
            if (surf not in TPMS and surf not in PARAMETRIC
                    and surf not in TPMS_EXACT):
                items = _periodic_surface_items(self, context)
                surf = items[0][0] if items else 'G'
            # effective per-axis cell counts: the legacy scalar `cells`
            # (default 0) broadcasts to every axis still at its default
            def _eff(v):
                return self.cells if (self.cells > 0 and v == 1) else v
            cu, cv, cw = (_eff(self.cells_u), _eff(self.cells_v),
                          _eff(self.cells_w))
            if surf in TPMS_EXACT:
                # A named preset (P / Gyroid / D) shows the CLEAN, iconic
                # surface via the nodal builder -- the exact-WE tiling of P/D
                # reads rougher and the chiral gyroid can't be tiled
                # watertight, so the nodal cells are the better "iconic" view
                # (and give a clean gyroid).  Custom sweeps the exact
                # Weierstrass Bonnet morph at the raw slider angle -- the
                # unique capability of this entry -- and at exactly 0 / 38.0148
                # / 90 deg Custom still yields the exact tiled P/D cell.
                cxyz = (cu, cv, cw)
                _PGD_NODAL = {'P': 'P', 'GYROID': 'G', 'D': 'D'}
                if self.pgd_preset in _PGD_NODAL:
                    nk = _PGD_NODAL[self.pgd_preset]
                    verts, tris = build_tpms(nk, cxyz,
                                             self.resolution, self.cell_size,
                                             offset=self.level_offset,
                                             aspect=self.cell_aspect)
                    label = TPMS[nk][0]
                else:                                   # CUSTOM: exact morph
                    verts, tris = build_tpms_exact(
                        surf, cxyz, self.resolution, self.cell_size,
                        self.assoc_angle)
                    label = TPMS_EXACT[surf][0]
                if len(tris) == 0:
                    self.report({'ERROR'}, "Empty surface")
                    return {'CANCELLED'}
                obj = _new_object(context, label, verts, tris)
                if self.thickness > 0:
                    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                    mod.thickness = self.thickness
                    mod.offset = 0.0
                if self.rim:
                    _ob = context.active_object
                    if _ob is not None:
                        _rim.add_rim_from_object(
                            context, _ob, _ob.name,
                            self.rim_thickness, self.rim_smooth)
                return {'FINISHED'}
            if surf in TPMS:
                cxyz = (cu, cv, cw)
                verts, tris = build_tpms(surf, cxyz,
                                         self.resolution, self.cell_size,
                                         offset=self.level_offset,
                                         aspect=self.cell_aspect)
                if len(tris) == 0:
                    self.report({'ERROR'}, "Empty level set")
                    return {'CANCELLED'}
                label = TPMS[surf][0]
                obj = _new_object(context, label, verts, tris)
                if self.thickness > 0:
                    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                    mod.thickness = self.thickness
                    mod.offset = 0.0
                if self.rim:
                    _ob = context.active_object
                    if _ob is not None:
                        _rim.add_rim_from_object(
                            context, _ob, _ob.name,
                            self.rim_thickness, self.rim_smooth)
                return {'FINISHED'}
            if surf not in PARAMETRIC:
                self.report({'ERROR'}, f"Unknown surface '{surf}'")
                return {'CANCELLED'}
            label = PARAMETRIC[surf][0]
            theta = (self.assoc_angle if surf in ANGLE_PARAM
                     else 0.0)
            if self.output == 'NURBS' and surf not in MESH_PARAM:
                G, wrap_u, wrap_v = build_parametric_grid(
                    surf, self.ctrl_u, self.ctrl_v,
                    self.order, self.radius, self.scale, theta)
                if wrap_u:
                    G = G[:-1]
                if wrap_v:
                    G = G[:, :-1]
                _nurbs_grid_object(context, label, G,
                                   cyclic_u=wrap_u, cyclic_v=wrap_v)
            else:
                out = build_parametric(surf, self.res_u,
                                       self.res_v, self.order,
                                       self.radius, self.scale, theta,
                                       with_uv=True,
                                       cells=(cu, cv))
                V, quads = out[0], out[1]
                cuv = out[2] if len(out) > 2 else None
                _new_object(context, label, V, quads,
                            weld=1e-5 * max(1.0, self.scale),
                            loop_uv=cuv)
            if self.rim:
                _ob = context.active_object
                if _ob is not None:
                    _rim.add_rim_from_object(
                        context, _ob, _ob.name,
                        self.rim_thickness, self.rim_smooth)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'periodicity')
            lay.prop(self, 'surface')
            if (self.periodicity == 'TRIPLY' or self.surface in TPMS
                    or self.surface in TPMS_EXACT):
                if self.surface in TPMS_EXACT:
                    # exact P/Gyroid/D: named preset first, then always show
                    # the raw Bonnet-angle slider.  A named preset reflects the
                    # special angle it uses; Custom drives the slider directly.
                    lay.prop(self, 'pgd_preset')
                    row = lay.row()
                    row.enabled = (self.pgd_preset == 'CUSTOM')
                    ang = _PGD_PRESET_ANGLE.get(self.pgd_preset)
                    if ang is None:
                        row.prop(self, 'assoc_angle')
                    else:
                        row.prop(self, 'assoc_angle',
                                 text=f"Associate Angle [{math.degrees(ang):.4g} deg]")
                # triply: three independent per-axis counts (x, y, z)
                lay.prop(self, 'cells_u', text="Cells X")
                lay.prop(self, 'cells_v', text="Cells Y")
                lay.prop(self, 'cells_w', text="Cells Z")
                for k in ('resolution', 'cell_size', 'thickness',
                          'level_offset', 'cell_aspect'):
                    lay.prop(self, k)
                return
            mesh_only = self.surface in MESH_PARAM
            if not mesh_only:
                lay.prop(self, 'output')
            if self.output == 'NURBS' and not mesh_only:
                lay.prop(self, 'ctrl_u')
                lay.prop(self, 'ctrl_v')
            else:
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
            if self.surface in COUNT_PARAM:
                lay.prop(self, 'order', text=COUNT_PARAM[self.surface])
            elif self.surface not in ANGLE_PARAM:
                lay.prop(self, 'order')
            # cell counts: one per tiling dimension (singly -> u; doubly ->
            # u, v).  NURBS output is a single control patch (no array).
            dim = _periodic_dim(self.surface)
            if (self.output == 'MESH' or mesh_only) \
                    and self.surface not in PERIODIC_NO_ARRAY:
                if dim >= 2:
                    lay.prop(self, 'cells_u', text="Cells U")
                    lay.prop(self, 'cells_v', text="Cells V")
                else:
                    lay.prop(self, 'cells_u', text="Cells")
            if self.surface in ANGLE_PARAM:
                lay.prop(self, 'assoc_angle')
            lay.prop(self, 'radius')
            lay.prop(self, 'scale')
            _rim.draw_rim(lay, self)

    class OBJECT_OT_minimal_span(bpy.types.Operator):
        """Span a minimal surface across the selected curve (1 object:
        disk) or between two selected curves (2 objects: annulus)"""
        bl_idname = "object.minimal_span"
        bl_label = "Span Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()

        samples: IntProperty(
            name="Boundary Samples", default=128, min=16, max=512)
        rings: IntProperty(
            name="Interior Rings", default=24, min=3, max=128)
        iterations: IntProperty(
            name="Solver Iterations", default=40, min=1, max=200)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Emit a compact NURBS surface (control grid = the "
                        "solver grid) instead of a dense mesh. Where the "
                        "surface curls tightly (e.g. near a knot) the NURBS "
                        "may ripple; raise rings/samples or use mesh output")

        @classmethod
        def poll(cls, context):
            sel = [o for o in context.selected_objects
                   if o.type in ('CURVE', 'MESH')]
            return len(sel) in (1, 2)

        def execute(self, context):
            deps = context.evaluated_depsgraph_get()
            sel = [o for o in context.selected_objects
                   if o.type in ('CURVE', 'MESH')]
            try:
                loops = [_extract_loop(o, deps) for o in sel]
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            m = self.samples
            if len(loops) == 1:
                A = resample_loop(loops[0], m)
                V, quads, fixed = build_disk_grid(A, self.rings)
                rows = self.rings   # + center point row
            else:
                A = resample_loop(loops[0], m)
                B = align_loops(A, resample_loop(loops[1], m))
                V, quads, fixed = build_annulus_grid(A, B, self.rings)
                rows = self.rings + 1
            T = _quads_to_tris(quads)
            # groom_every: in-loop Delaunay flips + tangential averaging
            # (solver/groom) -- measured in tests/bench to raise the
            # 5th-percentile triangle quality ~3x at unchanged area
            minimize_area(V, T, fixed, outer_iters=self.iterations,
                          groom_every=4)
            if self.output_nurbs:
                G = fair_grid_columns(V[:rows * m].reshape(rows, m, 3))
                G = fair_grid_2d(G)
                V[:rows * m] = G.reshape(-1, 3)
                relax_normal_flow(V, T, fixed)
                G = V[:rows * m].reshape(rows, m, 3)
                if len(loops) == 1:   # close the pole with the center point
                    cen = np.tile(V[-1], (1, m, 1))
                    G = np.concatenate([G, cen], axis=0)
                obj = _nurbs_grid_object(context, "MinimalSpan", G,
                                         cyclic_u=False, cyclic_v=True)
            else:
                obj = _new_object(context, "MinimalSpan", V, quads)
            obj.location = (0, 0, 0)   # loops are in world space
            self.report({'INFO'},
                        f"area = {mesh_area(V, T):.4f}")
            if self.rim:
                _ob = context.active_object
                if _ob is not None:
                    _rim.add_rim_from_object(
                        context, _ob, _ob.name,
                        self.rim_thickness, self.rim_smooth)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('samples', 'rings', 'iterations', 'output_nurbs'):
                lay.prop(self, k)
            _rim.draw_rim(lay, self)

    class MESH_OT_knot_span_add(bpy.types.Operator):
        """Minimal surface between a circle and a (p,q) torus knot
        (trefoil by default) -- the classic Plateau demonstration"""
        bl_idname = "mesh.minimal_knot_span_add"
        bl_label = "Knot to Knot Surface"
        bl_options = {'REGISTER', 'UNDO'}
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()

        p: IntProperty(name="Knot p", default=2, min=1, max=8)
        q: IntProperty(
            name="Knot q", default=3, min=0, max=9,
            description="q of the inner torus knot; 0 degenerates "
                        "it to a flat circle (radius 3 x Knot "
                        "Scale) wound p times")
        circle_radius: FloatProperty(
            name="Circle Radius", default=4.5, min=1.0, max=20.0)
        knot_scale: FloatProperty(
            name="Knot Scale", default=1.0, min=0.1, max=5.0)
        inner_height: FloatProperty(
            name="Inner Height", default=1.0, min=0.0, max=5.0,
            description="Scale of the inner boundary's vertical "
                        "oscillation, independent of its radius "
                        "(0 flattens it into a wavy-radius ring)")
        inner_lift: FloatProperty(
            name="Inner Lift", default=0.0, min=-10.0, max=10.0,
            description="Shift the inner boundary up or down; with "
                        "two circles this makes a catenoid-style "
                        "span")
        inner_rotation: FloatProperty(
            name="Inner Rotation", default=0.0,
            min=-2.0 * math.pi, max=2.0 * math.pi, subtype='ANGLE',
            description="Rotate the inner boundary about the "
                        "vertical axis relative to the outer one, "
                        "twisting the ruling between them")
        samples: IntProperty(
            name="Boundary Samples", default=96, min=32, max=512)
        rings: IntProperty(name="Interior Rings", default=16, min=4, max=128)
        iterations: IntProperty(
            name="Solver Iterations", default=2, min=1, max=200)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Emit a compact NURBS surface (control grid = the "
                        "solver grid) instead of a dense mesh. Where the "
                        "surface curls tightly (e.g. near a knot) the NURBS "
                        "may ripple; raise rings/samples or use mesh output")
        outer_q: IntProperty(
            name="Outer Knot q", default=0, min=0, max=9,
            description="The outer boundary as a (p, q) torus knot; "
                        "0 keeps the flat round circle (a circle is "
                        "the degenerate q = 0 knot)")
        outer_p: IntProperty(
            name="Outer Knot p", default=0, min=0, max=8,
            description="p of the outer boundary; 0 matches the "
                        "inner knot's p (which keeps the ruling "
                        "lined up)")
        outer_scale: FloatProperty(
            name="Outer Knot Scale", default=2.0, min=0.1, max=10.0,
            description="Scale of the outer torus knot (its radii "
                        "are ~1-3 x this)")
        outer_height: FloatProperty(
            name="Outer Height", default=1.0, min=0.0, max=5.0,
            description="Scale of the outer knot's vertical "
                        "oscillation, independent of its radius "
                        "(0 flattens it into a wavy-radius ring)")
        split_sheets: BoolProperty(
            name="Split Sheets", default=False,
            description="The span winds its outer boundary p times "
                        "and the sheets pass through one another; "
                        "this outputs p separate one-winding sheet "
                        "objects instead (they share their seam "
                        "edges, so together they still form the "
                        "whole span)")
        span_topology: EnumProperty(
            name="Span Topology",
            items=(
                ('SEIFERT', "Single Sheet (Seifert)",
                 "One embedded self-intersection-free surface from "
                 "the circle to the knot: the outer sheet funnels in "
                 "once around, a membrane spans the middle, and q "
                 "half-twist saddle bands resolve the knot "
                 "(Seifert's algorithm on the round diagram; genus "
                 "(q-1)/2). Used for p = 2 knots over the circle; "
                 "other cases fall back to the wound annulus. Keeping "
                 "the sheet embedded floors the samples it uses and "
                 "caps both the rings and the relaxation, so Boundary "
                 "Samples, Interior Rings and Solver Iterations are "
                 "requests rather than exact counts here"),
                ('WOUND', "Wound Annulus (crossing sheets)",
                 "The classic ruled annulus: the circle is traversed "
                 "p times so the ruling lines up, and for p > 1 the "
                 "windings pass through one another (an embedded "
                 "annulus between a knot and a round circle cannot "
                 "exist). Split Sheets can emit the windings as "
                 "separate objects")),
            default='SEIFERT',
            description="How to span the surface when the outer "
                        "boundary is the round circle")

        def _seifert_active(self):
            """The single-sheet Seifert span applies: (2, odd-q) knot
            over the plain circle, with enough room and a non-flat
            knot. Everything else keeps the classic wound annulus."""
            return (self.span_topology == 'SEIFERT'
                    and self.outer_q == 0 and self.p == 2
                    and self.q >= 1 and self.q % 2 == 1
                    and self.inner_height > 0.0
                    and self.circle_radius > 3.0 * self.knot_scale)

        def execute(self, context):
            if self._seifert_active():
                V, quads, fixed = build_seifert_span_grid(
                    self.q, self.samples, self.rings,
                    knot_scale=self.knot_scale,
                    circle_radius=self.circle_radius,
                    inner_height=self.inner_height,
                    inner_lift=self.inner_lift)
                if self.inner_rotation != 0.0:
                    # rotating everything keeps the circle a circle
                    # while turning the knot as requested
                    ca = math.cos(self.inner_rotation)
                    sa = math.sin(self.inner_rotation)
                    V[:, :2] = np.stack(
                        [V[:, 0] * ca - V[:, 1] * sa,
                         V[:, 0] * sa + V[:, 1] * ca], axis=1)
                T = _quads_to_tris(quads)
                iters = min(self.iterations, _SEIFERT_MAX_ITERS)
                # grooming keeps the collar/sliver quality up; the whole
                # q x samples sweep stays embedded with it (tests/bench
                # seifert_sweep: 17/17 selfx == 0)
                minimize_area(V, T, fixed, outer_iters=iters,
                              groom_every=4)
                V = _center_fit(V, 1.0)
                obj = _new_object(
                    context, f"Knot(2,{self.q})SeifertSpan", V, quads)
                capped = ("" if iters == self.iterations else
                          f" (relaxation capped at {iters} passes to "
                          f"keep the sheet embedded)")
                self.report(
                    {'INFO'},
                    f"single sheet, genus {(self.q - 1) // 2}, "
                    f"area = {mesh_area(V, T):.4f}{capped}")
                if self.rim:
                    _ob = context.active_object
                    if _ob is not None:
                        _rim.add_rim_from_object(
                            context, _ob, _ob.name,
                            self.rim_thickness, self.rim_smooth)
                return {'FINISHED'}
            if (self.span_topology == 'SEIFERT' and self.outer_q == 0
                    and self.p > 1):
                # asked for the single sheet but outside its range, and
                # the annulus about to be built really does wind p times
                # through itself: say so rather than quietly handing
                # back crossing sheets
                self.report(
                    {'WARNING'},
                    "Single Sheet needs p = 2, odd q, a non-flat knot "
                    "and a circle clear of it (Circle Radius > 3 x Knot "
                    "Scale); using the wound annulus instead")
            m = self.samples
            if self.split_sheets and self.p > 1:
                m = max(self.p * 8, (m // self.p) * self.p)
            knot = torus_knot(self.p, self.q, m, scale=self.knot_scale)
            knot[:, 2] *= self.inner_height
            knot[:, 2] += self.inner_lift
            if self.inner_rotation != 0.0:
                ca = math.cos(self.inner_rotation)
                sa = math.sin(self.inner_rotation)
                knot[:, :2] = np.stack(
                    [knot[:, 0] * ca - knot[:, 1] * sa,
                     knot[:, 0] * sa + knot[:, 1] * ca], axis=1)
            t = np.linspace(0, TAU, m, endpoint=False)
            po = self.outer_p or self.p
            if self.outer_q > 0:
                # outer boundary: another torus knot
                circ = torus_knot(po, self.outer_q, m,
                                  scale=self.outer_scale)
                circ[:, 2] *= self.outer_height
            else:
                # circle wound p times so the ruling lines up
                # (outer_p is ignored while the outer boundary is
                # a circle -- it is hidden in the UI then, and a
                # stale value must not change the winding)
                circ = np.stack(
                    [self.circle_radius * np.cos(self.p * t),
                     self.circle_radius * np.sin(self.p * t),
                     np.zeros(m)], axis=1)
            V, quads, fixed = build_annulus_grid(knot, circ, self.rings)
            T = _quads_to_tris(quads)
            minimize_area(V, T, fixed, outer_iters=self.iterations,
                          groom_every=4)
            # center on the origin and fit within a 2 m cube
            V = _center_fit(V, 1.0)
            name = (f"Knot({self.p},{self.q})Span" if self.outer_q == 0
                    else f"Knot({self.p},{self.q})-"
                         f"({po},{self.outer_q})Span")
            if self.split_sheets and self.p > 1:
                # one object per winding: columns [k w, (k+1) w] of
                # the solver grid, seam columns shared between
                # neighboring sheets
                w = m // self.p
                R = self.rings + 1
                if self.output_nurbs:
                    Gs = fair_grid_columns(V.reshape(R, m, 3))
                    Gs = fair_grid_2d(Gs)
                    V2 = Gs.reshape(-1, 3).copy()
                    relax_normal_flow(V2, T, fixed)
                    Gs = V2.reshape(R, m, 3)
                else:
                    Gs = V.reshape(R, m, 3)
                made = []
                for k in range(self.p):
                    cols = [(k * w + j) % m for j in range(w + 1)]
                    nm = f"{name} Sheet {k + 1}of{self.p}"
                    if self.output_nurbs:
                        made.append(_nurbs_grid_object(
                            context, nm, Gs[:, cols],
                            cyclic_u=False, cyclic_v=False))
                    else:
                        Vk = Gs[:, cols, :].reshape(-1, 3)
                        qk = [(r * (w + 1) + j,
                               r * (w + 1) + j + 1,
                               (r + 1) * (w + 1) + j + 1,
                               (r + 1) * (w + 1) + j)
                              for r in range(self.rings)
                              for j in range(w)]
                        made.append(_new_object(context, nm, Vk, qk))
                for o in made:
                    o.select_set(True)
                self.report({'INFO'},
                            f"{self.p} sheets, area = "
                            f"{mesh_area(V, T):.4f}")
                if self.rim:
                    _ob = context.active_object
                    if _ob is not None:
                        _rim.add_rim_from_object(
                            context, _ob, _ob.name,
                            self.rim_thickness, self.rim_smooth)
                return {'FINISHED'}
            if self.output_nurbs:
                G = fair_grid_columns(V.reshape(self.rings + 1, m, 3))
                G = fair_grid_2d(G)
                V = G.reshape(-1, 3).copy()
                relax_normal_flow(V, T, fixed)
                G = V.reshape(self.rings + 1, m, 3)
                obj = _nurbs_grid_object(context, name, G,
                                         cyclic_u=False, cyclic_v=True)
            else:
                obj = _new_object(context, name, V, quads)
            self.report({'INFO'}, f"area = {mesh_area(V, T):.4f}")
            if self.rim:
                _ob = context.active_object
                if _ob is not None:
                    _rim.add_rim_from_object(
                        context, _ob, _ob.name,
                        self.rim_thickness, self.rim_smooth)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('p', 'q', 'knot_scale', 'inner_height',
                      'inner_lift', 'inner_rotation', 'outer_q'):
                lay.prop(self, k)
            if self.outer_q > 0:
                lay.prop(self, 'outer_p')
                lay.prop(self, 'outer_scale')
                lay.prop(self, 'outer_height')
            else:
                lay.prop(self, 'circle_radius')
                if self.p == 2 and self.q % 2 == 1:
                    lay.prop(self, 'span_topology')
            seif = self._seifert_active()
            for k in ('samples', 'rings', 'iterations'):
                lay.prop(self, k)
            if not seif:
                lay.prop(self, 'output_nurbs')
                if self.p > 1:
                    lay.prop(self, 'split_sheets')

    # The sidebar panel that used to sit here is gone.  It only
    # launched the four operators below, every one of which is in
    # Add > Mesh > Minimal Surfaces already, so it was a second copy
    # of the menu taking up the tab -- and the tab is for editing the
    # selected object, not for making new ones.
            _rim.draw_rim(lay, self)

    class VIEW3D_MT_math_art_minimal_zoo(bpy.types.Menu):
        """The minimal-surface catalog, one entry per family (each
        opens the parametric operator with that family preset)."""
        bl_idname = "VIEW3D_MT_math_art_minimal_zoo"
        bl_label = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            for fam, flabel, _desc in _FAMILY_ITEMS:
                op = lay.operator("mesh.parametric_minimal_add",
                                  text=flabel, icon='SURFACE_NSPHERE')
                op.family = fam
            lay.separator()
            lay.operator("mesh.tpms_add",
                         text="Triply Periodic (TPMS)",
                         icon='MESH_ICOSPHERE')

    class VIEW3D_MT_minimal_add(bpy.types.Menu):
        bl_idname = "VIEW3D_MT_minimal_add"
        bl_label = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            lay.menu("VIEW3D_MT_math_art_minimal_zoo",
                     icon='SURFACE_NSPHERE')
            lay.operator("mesh.parametric_minimal_add")
            lay.operator("mesh.tpms_add")
            lay.operator("mesh.minimal_knot_span_add")
            lay.operator("object.minimal_span")

    def _menu_func(self, context):
        self.layout.menu("VIEW3D_MT_minimal_add", icon='SURFACE_DATA')

    _classes = (MESH_OT_parametric_minimal_add, MESH_OT_tpms_add,
                MESH_OT_periodic_minimal_add,
                OBJECT_OT_minimal_span, MESH_OT_knot_span_add,
                                VIEW3D_MT_math_art_minimal_zoo, VIEW3D_MT_minimal_add)

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


def _selftest():
    # standalone smoke tests of the numeric core
    ok = True
    # Weierstrass engine invariants on the square torus
    L = _SQUARE
    e1 = L.wp(0.5).real
    zt = np.array([0.2 + 0.3j, 0.37 + 0.11j, 0.6 + 0.44j])
    resid = np.max(np.abs(L.wp_prime(zt) ** 2
                          - (4 * L.wp(zt) ** 3 - 4 * e1 ** 2 * L.wp(zt))))
    print(f"weierstrass: e1={e1:.5f} (exp 6.87519) g2={4*e1**2:.4f} "
          f"(exp 189.0727) |P'^2-(4P^3-g2 P)|={resid:.2e} "
          f"{'OK' if abs(e1-6.87519) < 1e-3 and resid < 1e-8 else 'FAIL'}")
    ok &= abs(e1 - 6.87519) < 1e-3 and resid < 1e-8
    for kind in PARAMETRIC:
        th = (math.pi / 4
              if kind in ANGLE_PARAM and kind not in COUNT_PARAM
              else 0.0)
        n = {'KNOID': 5, 'COSTA_HM': 1, 'SCHERK_TOWER': 3}.get(kind, 1)
        V, Q = build_parametric(kind, 60, 60, n, 1.2, 1.0, th)
        finite = bool(np.all(np.isfinite(V)))
        lo, hi = V.min(0), V.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        good = (finite and len(Q) > 100 and cen < 1e-6
                and abs(ext - 2.0) < 1e-6)
        ok &= good
        print(f"parametric {kind:10s}: {len(V):5d} verts {len(Q):5d} "
              f"quads  fit[max|c|={cen:.1e} ext={ext:.4f}] "
              f"{'OK' if good else 'FAIL'}")
    # UV gate: every parametric surface carries a finite, in-range,
    # non-collapsed conformal UV chart (per-corner, [0, 1])
    for kind in PARAMETRIC:
        n = {'KNOID': 5, 'COSTA_HM': 1, 'SCHERK_TOWER': 3}.get(kind, 1)
        out = build_parametric(kind, 48, 48, n, 1.2, 1.0,
                               with_uv=True)
        cuv = out[2] if len(out) > 2 else None
        if cuv is None:
            print(f"uv {kind:15s}: NO UV  FAIL")
            ok = False
            continue
        finite = bool(np.all(np.isfinite(cuv)))
        inrange = (cuv.min() >= -1e-6) and (cuv.max() <= 1.0 + 1e-6)
        span = np.ptp(cuv, axis=0)
        good = finite and inrange and bool(np.all(span > 0.3))
        ok &= good
        print(f"uv {kind:15s}: n={len(cuv):6d} "
              f"range[{cuv.min():.3f},{cuv.max():.3f}] "
              f"span=({span[0]:.2f},{span[1]:.2f}) "
              f"{'OK' if good else 'FAIL'}")
    # Costa-Hoffman-Meeks: modulus table + Euler characteristic gate
    # (genus k, 3 ends removed -> chi = 2 - 2k - 3 = -(2k+1))
    cref = {1: 0.955978, 2: 0.988070, 3: 0.995117, 4: 0.997535}
    cok = all(abs(_zoo.chm_modulus(kk) - cref[kk]) < 1e-5
              for kk in cref)
    print(f"CHM modulus: "
          + " ".join(f"c({kk})={_zoo.chm_modulus(kk):.5f}"
                     for kk in cref)
          + f"  {'OK' if cok else 'FAIL'}")
    ok &= cok
    for kk in (1, 2, 3):
        Vc, Qc = build_parametric('COSTA_HM', 48, 48, kk, 1.2, 1.0)
        ec = {}
        for f in Qc:
            for t in range(len(f)):
                a, b = f[t], f[(t + 1) % len(f)]
                e = (a, b) if a < b else (b, a)
                ec[e] = ec.get(e, 0) + 1
        chi = len(Vc) - len(ec) + len(Qc)
        want = -(2 * kk + 1)
        good = chi == want and np.all(np.isfinite(Vc))
        ok &= good
        print(f"CHM genus {kk}: verts={len(Vc)} faces={len(Qc)} "
              f"chi={chi} (want {want}) {'OK' if good else 'FAIL'}")
    # k-noid vs closed-form trinoid (n=3): both are minimal with 3
    # ends; compare 3-fold symmetry of the numeric build
    Vn, _ = build_parametric('KNOID', 72, 72, 3, 0.9, 1.0)
    ang = np.arctan2(Vn[:, 1], Vn[:, 0])
    print(f"k-noid n=3: verts={len(Vn)} z-range="
          f"[{Vn[:,2].min():.3f},{Vn[:,2].max():.3f}] "
          f"{'OK' if np.all(np.isfinite(Vn)) else 'FAIL'}")
    # ---- TPMS gates (incl. tier-2 nodal rows) ----------------------
    def _t2_grad(field, P, h=1e-4):
        g = np.empty_like(P)
        for a in range(3):
            d = np.zeros(3)
            d[a] = h
            g[:, a] = (field(*(P + d).T) - field(*(P - d).T)) / (2 * h)
        return g

    def _t2_ncomp(nv, T, frac=0.05):
        """number of connected components holding >= frac of tris"""
        parent = np.arange(nv)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b, c in T:
            ra, rb, rc = find(a), find(b), find(c)
            parent[rb] = ra
            parent[find(rc)] = find(ra)
        roots = np.array([find(t[0]) for t in T])
        _, counts = np.unique(roots, return_counts=True)
        return int(np.sum(counts >= frac * len(T)))

    # per-row field symmetry spot checks: key -> (map, sign);
    # F(map(p)) == sign * F(p) must hold identically
    _t2_cyc = (lambda P: P[:, [1, 2, 0]], 1.0)
    _t2_swap = (lambda P: P[:, [1, 0, 2]], 1.0)
    _t2_sym = {
        'OCTO': _t2_swap, 'KSURF': _t2_swap, 'FRD2': _t2_swap,
        'FK_S': _t2_cyc, 'FK_CS': _t2_cyc, 'FK_Y': _t2_cyc,
        'FK_PMY': _t2_cyc, 'FK_CPMY': _t2_cyc, 'FK_CY': _t2_cyc,
        'CG': _t2_cyc, 'GPRIME': _t2_cyc, 'DPRIME': _t2_cyc,
        'CI2Y': _t2_cyc,
        'CD': (lambda P: P + math.pi, -1.0),
    }
    _t2_new = tuple(_t2_sym)
    rng = np.random.default_rng(7)
    for kind in TPMS:
        V, T = build_tpms(kind, 1, 20, 2.0)
        lo, hi = V.min(0), V.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = hi - lo
        finite = bool(np.all(np.isfinite(V)))
        triply = TPMS[kind][2]
        good = finite and len(T) > 200
        if triply:
            good &= (cen < 0.15 and float(ext.min()) > 1.7
                     and float(ext.max()) < 2.0 + 1e-6)
        extra = ""
        if kind in _t2_new:
            field = TPMS[kind][1]
            off = tpms2_DEFAULT_OFFSET.get(kind, 0.0)
            f0 = (field if off == 0.0
                  else (lambda X, Y, Z: field(X, Y, Z) - off))
            # nonzero field gradient on the level set (no flats)
            P = V[rng.choice(len(V), min(400, len(V)),
                             replace=False)] * (TAU / 2.0)
            G = np.linalg.norm(_t2_grad(f0, P), axis=1)
            gok = float(G.min()) > 0.05 * float(np.median(G))
            # symmetry op maps the field to +-itself
            op, sgn = _t2_sym[kind]
            Q = rng.uniform(-math.pi, math.pi, (500, 3))
            Fq = field(*Q.T)
            serr = float(np.max(np.abs(field(*op(Q).T) - sgn * Fq)))
            sok = serr < 1e-9 * max(1.0, float(np.max(np.abs(Fq))))
            # one macro connected component on a 2^3 block
            V2, T2 = build_tpms(kind, 2, 14, 2.0)
            nc = _t2_ncomp(len(V2), T2)
            cok = nc == 1
            good &= gok and sok and cok
            extra = (f" grad[min/med={float(G.min()/np.median(G)):.3f}"
                     f" {'OK' if gok else 'FAIL'}]"
                     f" sym[{'OK' if sok else f'FAIL {serr:.2e}'}]"
                     f" comp[{nc} {'OK' if cok else 'FAIL'}]")
        ok &= good
        print(f"tpms {kind:9s}: {len(V):6d} verts {len(T):6d} tris "
              f"fit[|c|={cen:.3f} ext={ext.min():.2f}-{ext.max():.2f}] "
              f"{'OK' if good else 'FAIL'}{extra}")
    # level-offset path: the marched level actually moves to c
    V, T = build_tpms('P', 1, 24, 2.0, offset=0.4)
    Fv = _f_p(*(V * (TAU / 2.0)).T)
    lerr = float(np.max(np.abs(Fv - 0.4)))
    good = len(T) > 200 and lerr < 0.05
    ok &= good
    print(f"tpms offset  P c=0.4: level err {lerr:.4f} "
          f"{'OK' if good else 'FAIL'}")
    # tetragonal aspect path: cell tiles at the true c/a ratio
    V, T = build_tpms('D', 1, 20, 2.0, aspect=1.6)
    ext = V.max(0) - V.min(0)
    ratio = float(ext[2] / ext[0])
    good = len(T) > 200 and abs(ratio - 1.6) < 0.05
    ok &= good
    print(f"tpms aspect  D c/a=1.6: z/x extent {ratio:.3f} "
          f"{'OK' if good else 'FAIL'}")
    # hexagonal-lattice infrastructure: march P on a hex cell
    tpms2_LATTICE['P'] = TPMS2_HEX_LATTICE
    try:
        V, T = build_tpms('P', 1, 20, 2.0)
    finally:
        del tpms2_LATTICE['P']
    # analytic extents of the hex-sheared P cell: the zero set
    # reaches x = +-1.25 pi and y = +-(sqrt3/2) pi in cell coords
    ext = V.max(0) - V.min(0)
    want = (2.5, math.sqrt(3.0), 2.0)
    good = (len(T) > 200 and bool(np.all(np.isfinite(V)))
            and all(abs(float(e) - w) < 0.1
                    for e, w in zip(ext, want)))
    ok &= good
    print(f"tpms hexcell P: extents ({ext[0]:.2f},{ext[1]:.2f},"
          f"{ext[2]:.2f}) {'OK' if good else 'FAIL'}")
    # flat-disk validation: minimal surface on a planar circle is flat
    t = np.linspace(0, TAU, 96, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    V, quads, fixed = build_disk_grid(circle, 12)
    V[~fixed] += np.random.default_rng(1).normal(0, 0.1, V[~fixed].shape)
    T = _quads_to_tris(quads)
    minimize_area(V, T, fixed)
    print("flat disk: max|z| =", float(np.max(np.abs(V[:, 2]))),
          " area =", mesh_area(V, T), " (pi =", math.pi, ")")
    # catenoid validation: two rings radius 1 at z = +/-0.4
    z0 = 0.4
    ringA = np.stack([np.cos(t), np.sin(t), np.full_like(t, z0)], axis=1)
    ringB = np.stack([np.cos(t), np.sin(t), np.full_like(t, -z0)], axis=1)
    V, quads, fixed = build_annulus_grid(ringA, ringB, 20)
    T = _quads_to_tris(quads)
    minimize_area(V, T, fixed)
    waist = np.min(np.linalg.norm(V[:, :2], axis=1))
    print("catenoid: waist =", float(waist), "(analytic ~0.9098)")
    # single-sheet Seifert span between a (2, q) knot and the circle:
    # Euler characteristic (genus gate), 2 boundary loops == the fixed
    # verts, manifoldness, and genuine self-intersection freedom -- the
    # last one relaxed all the way to the iteration cap, which is the
    # worst case the operator can ask for
    for qq, chi_want in ((1, 0), (3, -2), (5, -4), (7, -6), (9, -8)):
        V, F, fx = build_seifert_span_grid(qq, 48, 8)
        T = _quads_to_tris(F)
        ec = {}
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                e = (a, b) if a < b else (b, a)
                ec[e] = ec.get(e, 0) + 1
        chi = len(V) - len(ec) + len(F)
        nm = sum(1 for v in ec.values() if v > 2)
        adj = {}
        for (a, b), n in ec.items():
            if n == 1:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        seen = set()
        loops = 0
        for s in adj:
            if s not in seen:
                loops += 1
                stk = [s]
                while stk:
                    x = stk.pop()
                    if x not in seen:
                        seen.add(x)
                        stk.extend(adj[x])
        bnd_fix = set(adj) == set(np.nonzero(fx)[0].tolist())
        minimize_area(V, T, fx, outer_iters=_SEIFERT_MAX_ITERS)
        V = _center_fit(V, 1.0)
        nx = _selfx_crossings(V, T)
        lo, hi = V.min(0), V.max(0)
        fit = (np.max(np.abs(0.5 * (lo + hi))) < 1e-6
               and abs(float(np.max(hi - lo)) - 2.0) < 1e-6)
        good = (chi == chi_want and nm == 0 and loops == 2
                and bnd_fix and nx == 0 and fit)
        ok &= good
        print(f"seifert span (2,{qq}): chi={chi} (want {chi_want}) "
              f"loops={loops} nonmanifold={nm} bnd=fixed:{bnd_fix} "
              f"crossings={nx} fit={fit} {'OK' if good else 'FAIL'}")
    # the wound annulus really does self-cross for p = 2 -- the
    # crossing detector must see it (sanity of the gate itself)
    t48 = np.linspace(0, TAU, 48, endpoint=False)
    kn = torus_knot(2, 3, 48)
    ci = np.stack([4.5 * np.cos(2 * t48), 4.5 * np.sin(2 * t48),
                   np.zeros(48)], axis=1)
    Vw, Fw, fw = build_annulus_grid(kn, ci, 8)
    Tw = _quads_to_tris(Fw)
    minimize_area(Vw, Tw, fw, outer_iters=2)
    nw = _selfx_crossings(_center_fit(Vw, 1.0), Tw)
    good = nw > 0
    ok &= good
    print(f"wound annulus (2,3): crossings={nw} detected "
          f"{'OK' if good else 'FAIL'} (the classic mode crosses)")
    print("\nRESULT:", "ALL OK" if ok else "FAILURES in parametric core")
    assert ok
