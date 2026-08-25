# Saddle Polyhedron -- Pearce's Table 8.1, as buildable geometry.
#
# A saddle polyhedron is bounded not by flat faces but by SADDLE
# POLYGONS: skew circuits spanned by minimal surfaces.  Peter Pearce
# catalogued 53 of them in "Structure in Nature is a Strategy for
# Design", every one a closed circuit-graph in a single net -- the
# Universal Node, whose branches run in just 26 directions of a cubic
# lattice (6 of <100>, 12 of <110>, 8 of <111>).  Because the net is so
# constrained, each solid's row in Table 8.1 -- node valences, branch
# counts by direction class, face count, face types, included angles,
# face-plane directions, symmetry axes -- is a complete CHECKSUM on the
# geometry.
#
# WHERE THE GEOMETRY COMES FROM.  Pearce prints no coordinates.  So the
# solids are not transcribed, they are FOUND: `pearce_net` searches the
# net for closed complexes and `pearce_data` keeps only those matching
# a row on every column it can check.  The table is the acceptance
# test, not the source.  Coverage is therefore partial and honest --
# rows with no verified geometry are listed in `pearce_data.UNRESOLVED`
# and simply are not offered, because a guess with the right face count
# would be worse than a shorter list.
#
# THE ANGLE THEOREM this all rests on: over all 325 pairs of the 26
# branch directions there are exactly 12 distinct angles, and every
# included angle Pearce prints is one of them.  Verified in
# `pearce_net._selftest`, not assumed.  (Two of the twelve, 125d16' and
# 135d, never occur as a face corner in the 53 solids; the table's lone
# "69d" is a misprint for 60d, since 69d is not a branch-pair angle at
# all and 2-fold symmetry forces that face's opposite angles equal.)
#
# FACE STYLES.  MINIMAL relaxes each face to the soap film Pearce
# defines it as.  RULED keeps the straight-line disk grid -- the ruled
# saddle his plastic panels approximate.  SPIDRON fills each face with
# a spidron nest instead, after van Ballegooijen, Gailiunas and
# Erdely's "Spidronised Space-fillers"; see the honest limitation
# below.  NET draws the branch graph alone.
#
# THE SPIDRON LIMITATION, STATED PLAINLY.  A true spidron nest -- the
# foldable kind, all triangles congruent -- exists only on an
# equilateral, equiangular skew polygon.  That is an exact test, not a
# judgement call, and the operator applies it per face: uniform edge
# lengths and one included angle throughout.  Faces that pass get a
# true nest.  Faces that fail get the general similarity
# spidronisation, which is drawable and is what the Bridges paper does
# for irregular polygons, but its annulus triangles are NOT congruent:
# decoration, not a spidron in the strict sense.  The operator reports
# the split so the distinction is never silent.
#
# References:
# - Peter Pearce, "Structure in Nature is a Strategy for Design", The
#   MIT Press, 1978 (paperback 1990), ch. 8 -- the Universal Node
#   system, saddle polygons spanned by minimal surfaces, and Table
#   8.1's inventory of 53 saddle polyhedra.
# - Walt van Ballegooijen, Paul Gailiunas & Daniel Erdely,
#   "Spidronised Space-fillers", Bridges 2009 Conference Proceedings,
#   pp. 271-278 -- spidron nests on saddle-polyhedron faces and the
#   rule that clockwise must meet counter-clockwise across a shared
#   face.
# - Daniel Erdely, "Some Surprising New Properties of the Spidrons",
#   Bridges 2005 Conference Proceedings, pp. 179-186 -- the spidron
#   nest the face decoration generalises.
# - Ulrich Pinkall & Konrad Polthier, "Computing Discrete Minimal
#   Surfaces and Their Conjugates", Experimental Mathematics 2(1),
#   1993, pp. 15-36 -- the cotangent-Laplacian area minimisation used
#   for the saddle faces.

bl_info = {
    "name": "Saddle Polyhedron",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Polyhedra",
    "description": "Pearce's saddle polyhedra: skew circuits of the "
                   "Universal Node net spanned by minimal surfaces, "
                   "optionally filled with spidron nests",
    "category": "Add Mesh",
}

from math import radians

import numpy as np

try:
    from . import pearce_net as pnet
    from . import pearce_data as pdata
    from . import pearce_surface as psurf
    from . import pearce_tiling as ptile
    from . import spidron_math as sm
    from .patterns import common as pc
    from . import sharp_creases as _sc
except Exception:                       # legacy single-file / CLI use
    import pearce_net as pnet
    import pearce_data as pdata
    import pearce_surface as psurf
    import pearce_tiling as ptile
    import spidron_math as sm
    from patterns import common as pc
    import sharp_creases as _sc


FAMILY_LABELS = {
    'TRIHEDRA': "Three faces",
    'TETRAHEDRA': "Four faces",
    'PENTAHEDRA': "Five faces",
    'HEXAHEDRA': "Six faces",
    'OCTAHEDRA': "Eight faces",
    'DECAHEDRA': "Ten faces",
    'DODECAHEDRA': "Twelve faces",
    'LARGER': "More faces",
}


# --------------------------------------------------------------------
# 1.  Face fills
# --------------------------------------------------------------------

def true_nest_faces(solid):
    """Which faces admit a TRUE (congruent-triangle) spidron nest.

    Exact test, applied per face: one edge length and one included
    angle throughout.  No per-solid judgement calls."""
    V = solid['verts']
    return [pnet.is_equilateral_equiangular([V[i] for i in f])
            for f in solid['faces']]


def spidron_faces(solid, rings, scale, twist):
    """Spidron nests on every face, with a true/generalised split."""
    V = solid['verts']
    flags = true_nest_faces(solid)
    P = np.asarray(V, float)
    centre = P.mean(axis=0)
    verts, tris, face_id = [], [], []
    for fi, cyc in enumerate(solid['faces']):
        loop = np.asarray([V[i] for i in cyc], float)
        nz = pnet.newell_normal([V[i] for i in cyc])
        mid = loop.mean(axis=0)
        if float(nz @ (mid - centre)) < 0:
            loop = loop[::-1]
        pts, polys, _mats = sm.spidronise(loop, scale, twist, rings)
        base = len(verts)
        verts.extend([tuple(p) for p in pts])
        for t in polys:
            if len(t) == 3:
                tris.append(tuple(base + i for i in t))
                face_id.append(fi)
            else:
                a, b, c, d = t
                tris.append((base + a, base + b, base + c))
                tris.append((base + a, base + c, base + d))
                face_id.append(fi)
                face_id.append(fi)
    return np.asarray(verts, float), tris, face_id, flags


# --------------------------------------------------------------------
# 2.  Build
# --------------------------------------------------------------------

def build(key=None, face_style='MINIMAL', density=3, smoothness=25,
          rings=5, scale=0.60, twist=0.0, layout='SINGLE',
          nx=1, ny=1, nz=1, gap=1.0):
    """Geometry for one saddle polyhedron, or a block of the packing.

    Returns (verts, tris, face_id, info)."""
    solid = pdata.by_key(key) if key else pdata.SOLIDS[0]

    if layout == 'BLOCK':
        return _build_block(solid, face_style, density, smoothness,
                            rings, scale, twist, nx, ny, nz, gap)

    V0, F0 = solid['verts'], solid['faces']

    if face_style == 'SPIDRON':
        V, T, fid, flags = spidron_faces(solid, rings, scale, twist)
        info = dict(true_nests=sum(1 for x in flags if x),
                    generalised=sum(1 for x in flags if not x),
                    faces=len(F0))
    elif face_style == 'NET':
        V, T, fid = _net_mesh(V0, F0)
        info = dict(true_nests=0, generalised=0, faces=len(F0))
    else:
        relax = (face_style == 'MINIMAL')
        V, T, fid = psurf.solid_surface(
            V0, F0, density=density,
            iters=smoothness if relax else 0, relax=relax)
        flags = true_nest_faces(solid)
        info = dict(true_nests=sum(1 for x in flags if x),
                    generalised=0, faces=len(F0))

    V = psurf.fit_unit(V)
    info['solid'] = solid
    info['aspect'] = psurf.aspect(V)
    return V, T, fid, info


def _cell_geometry(V0, F0, face_style, density, smoothness, rings,
                   scale, twist):
    """One cell's mesh, in its own coordinates (no fitting)."""
    if face_style == 'SPIDRON':
        cell = dict(verts=V0, faces=F0)
        V, T, fid, flags = spidron_faces(cell, rings, scale, twist)
        return V, T, fid, flags
    if face_style == 'NET':
        V, T, fid = _net_mesh(V0, F0)
        return V, T, fid, []
    relax = (face_style == 'MINIMAL')
    V, T, fid = psurf.solid_surface(
        V0, F0, density=density,
        iters=smoothness if relax else 0, relax=relax)
    return V, T, fid, pnet_flags(V0, F0)


def pnet_flags(V0, F0):
    return [pnet.is_equilateral_equiangular([V0[i] for i in f])
            for f in F0]


def _build_block(solid, face_style, density, smoothness, rings, scale,
                 twist, nx, ny, nz, gap):
    """A block of the space filling: every cell of the packing.

    The packing is verified, not assumed -- `pearce_tiling.pack`
    reports whether the cells' volume actually accounts for the block,
    and the operator passes that on rather than presenting a partial
    packing as a space filling."""
    V0, F0 = solid['verts'], solid['faces']
    copies, rep = ptile.pack(V0, F0, solid['net'], nx, ny, nz)

    verts, tris, face_id, crease_id = [], [], [], []
    nf = len(F0)
    for ci, pts in enumerate(copies):
        CV, CT, cfid, _flags = _cell_geometry(
            pts, F0, face_style, density, smoothness, rings, scale,
            twist)
        CV = np.asarray(CV, float)
        if gap < 1.0:
            c = CV.mean(axis=0)
            CV = c + (CV - c) * gap
        base = len(verts)
        verts.extend([tuple(p) for p in CV])
        for t, f in zip(CT, cfid):
            tris.append(tuple(base + i for i in t))
            # colour by cell, so the packing reads as separate solids
            face_id.append(ci)
            # crease by (cell, face): the outer edges of each polyhedron
            # are where two of ITS saddle faces meet, and grouping by
            # cell alone would leave those edges smoothed over -- the
            # solids then read as blobs under smooth shading
            crease_id.append(ci * nf + f)
    V = psurf.fit_unit(np.asarray(verts, float))
    flags = pnet_flags(V0, F0)
    info = dict(true_nests=sum(1 for x in flags if x) * len(copies),
                generalised=sum(1 for x in flags if not x) * len(copies),
                faces=len(F0) * len(copies), solid=solid,
                aspect=psurf.aspect(V), packing=rep,
                crease_id=crease_id)
    return V, tris, face_id, info


def build_cells(key=None, face_style='MINIMAL', density=3, smoothness=25,
                rings=5, scale=0.60, twist=0.0, nx=1, ny=1, nz=1,
                gap=1.0):
    """The packing as SEPARATE cells, still in register with each other.

    Returns (cells, info) where each cell is (verts, tris, face_id).
    One transform is computed for the whole block and applied to every
    cell, so the pieces stay assembled -- fitting each cell to the unit
    cube individually would scale them differently and scatter the
    packing."""
    solid = pdata.by_key(key) if key else pdata.SOLIDS[0]
    V0, F0 = solid['verts'], solid['faces']
    copies, rep = ptile.pack(V0, F0, solid['net'], nx, ny, nz)

    raw = []
    for pts in copies:
        CV, CT, cfid, _flags = _cell_geometry(
            pts, F0, face_style, density, smoothness, rings, scale, twist)
        CV = np.asarray(CV, float)
        if gap < 1.0:
            c = CV.mean(axis=0)
            CV = c + (CV - c) * gap
        raw.append((CV, CT, cfid))

    # one common fit for the whole block
    allv = np.concatenate([c[0] for c in raw], axis=0) if raw else \
        np.zeros((1, 3))
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    centre = 0.5 * (lo + hi)
    ext = float((hi - lo).max())
    s = (2.0 / ext) if ext > 1e-12 else 1.0

    cells = [((CV - centre) * s, CT, cfid) for CV, CT, cfid in raw]
    flags = pnet_flags(V0, F0)
    info = dict(solid=solid, packing=rep, faces=len(F0),
                true_nests=sum(1 for x in flags if x),
                generalised=sum(1 for x in flags if not x))
    return cells, info


def face_boundary_edges(tris, face_id):
    """Edges where two DIFFERENT saddle faces meet -- the branches.

    These are the creases: shading stays smooth across each saddle
    patch and breaks along the net, which is what makes the solid read
    as a polyhedron rather than a blob."""
    owner = {}
    for t, fi in zip(tris, face_id):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            e = (a, b) if a < b else (b, a)
            owner.setdefault(e, set()).add(fi)
    return [e for e, fs in owner.items() if len(fs) > 1]


def _net_mesh(V0, F0, radius=0.06, sides=6):
    """The branch graph as tubes -- Pearce's Universal Node model."""
    P = np.asarray(V0, float)
    verts, tris, fid = [], [], []
    for (a, b) in pnet.edge_counts(F0):
        A, B = P[a], P[b]
        d = B - A
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            continue
        d = d / L
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(d @ up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])
        u = np.cross(d, up)
        u = u / float(np.linalg.norm(u))
        w = np.cross(d, u)
        base = len(verts)
        for k in range(sides):
            th = 2.0 * np.pi * k / sides
            off = radius * L * (np.cos(th) * u + np.sin(th) * w)
            verts.append(tuple(A + off))
            verts.append(tuple(B + off))
        for k in range(sides):
            a0 = base + 2 * k
            a1 = base + 2 * ((k + 1) % sides)
            tris.append((a0, a1, a1 + 1))
            tris.append((a0, a1 + 1, a0 + 1))
            fid.append(0)
            fid.append(0)
    return np.asarray(verts, float), tris, fid


# --------------------------------------------------------------------
# 3.  Operator
# --------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except Exception:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _cell_material(name, color):
        """A shaded material for one cell.

        `patterns.common` re-exports PALETTE_RGBA but not the material
        helper (that one lives in patterns.emit), so this builds the
        material directly.  use_nodes matters: without it every cell
        renders flat grey, which this project has been bitten by twice.

        Module level, not class level: a name defined in a class body
        is not in scope inside its methods, so a bare call to it from
        execute() raises NameError.
        """
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = color
            mat.use_nodes = True
            node = mat.node_tree.nodes.get("Principled BSDF")
            if node:
                node.inputs[0].default_value = color
        return mat

    def _family_items(self, context):
        out = []
        for fam in pdata.families():
            out.append((fam, FAMILY_LABELS.get(fam, fam.title()),
                        "Saddle polyhedra with this many faces"))
        return out or [('NONE', "None", "")]

    def _solid_items(self, context):
        out = []
        for s in pdata.in_family(self.family):
            out.append((s['key'], "%d. %s" % (s['number'], s['name']),
                        "Table 8.1 entry %d" % s['number']))
        return out or [('NONE', "None", "")]

    class MESH_OT_saddle_polyhedron_add(bpy.types.Operator):
        """Add one of Pearce's saddle polyhedra"""
        bl_idname = "mesh.saddle_polyhedron_add"
        bl_label = "Saddle Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Group", items=_family_items,
            description="Pearce groups his table by number of faces")
        solid: EnumProperty(
            name="Solid", items=_solid_items,
            description="Which saddle polyhedron of Table 8.1 to build")
        face_style: EnumProperty(
            name="Faces",
            items=[('MINIMAL', "Minimal surface",
                    "Relax each face to the soap film spanning its "
                    "skew circuit -- Pearce's own definition"),
                   ('RULED', "Ruled",
                    "Straight-line patch without relaxation, the ruled "
                    "saddle a flat panel approximates"),
                   ('SPIDRON', "Spidron nest",
                    "Fill each face with a spidron nest. Only "
                    "equilateral equiangular faces take a true nest of "
                    "congruent triangles; the rest get a generalised "
                    "nest, which is decoration rather than a spidron"),
                   ('NET', "Branch network",
                    "The Universal Node net alone, as rods")],
            default='MINIMAL')
        density: IntProperty(
            name="Edge divisions", default=3, min=1, max=12,
            description="Segments per branch; also the face mesh density")
        smoothness: IntProperty(
            name="Relax steps", default=25, min=0, max=200,
            description="Area-minimisation iterations for minimal faces")
        rings: IntProperty(
            name="Nest rings", default=5, min=1, max=12,
            description="Spidron annuli per face")
        scale: FloatProperty(
            name="Nest step", default=0.60, min=0.05, max=0.95,
            description="Shrink factor between spidron annuli")
        twist: FloatProperty(
            name="Nest twist", default=0.0, min=radians(-60.0),
            max=radians(60.0), subtype='ANGLE',
            description="Rotation between spidron annuli")
        layout_kind: EnumProperty(
            name="Layout",
            items=[('SINGLE', "One solid",
                    "A single saddle polyhedron"),
                   ('BLOCK', "Space filling",
                    "Fill a block of unit cells with the packing this "
                    "solid belongs to")],
            default='SINGLE')
        nx: IntProperty(name="Cells across", default=1, min=1, max=6,
                        description="Unit cells along X")
        ny: IntProperty(name="Cells deep", default=1, min=1, max=6,
                        description="Unit cells along Y")
        nz: IntProperty(name="Cells high", default=1, min=1, max=6,
                        description="Unit cells along Z")
        separate: BoolProperty(
            name="Separate objects", default=False,
            description="Emit each cell of the packing as its own "
                        "object, grouped in a collection, instead of "
                        "one merged mesh")
        gap: FloatProperty(
            name="Shrink", default=1.0, min=0.5, max=1.0,
            description="Shrink each cell about its own centre so the "
                        "packing reads as separate solids")
        smooth: BoolProperty(
            name="Smooth shading", default=True,
            description="Shade smooth, with creases along the branches")

        def draw(self, context):
            # NB: the space-filling selector is `layout_kind`, never
            # `layout` -- an operator property called `layout` shadows
            # Operator.layout, so `self.layout` returns the enum STRING
            # and every draw call raises, leaving the panel blank.
            L = self.layout
            # house convention: labels in the left column.  Without
            # this an IntProperty draws as a slider with its name
            # inside the widget, so the numeric fields sit out of line
            # with the enums above them.
            L.use_property_split = True
            L.prop(self, "family")
            L.prop(self, "solid")
            L.prop(self, "face_style")
            L.prop(self, "layout_kind")
            if self.layout_kind == 'BLOCK':
                r = L.row(align=True)
                r.prop(self, "nx")
                r.prop(self, "ny")
                r.prop(self, "nz")
                L.prop(self, "gap")
                L.prop(self, "separate")
            if self.face_style in ('MINIMAL', 'RULED'):
                L.prop(self, "density")
            if self.face_style == 'MINIMAL':
                L.prop(self, "smoothness")
            if self.face_style == 'SPIDRON':
                L.prop(self, "rings")
                L.prop(self, "scale")
                L.prop(self, "twist")
            L.prop(self, "smooth")

        def _execute_separate(self, context, key):
            """One object per cell, grouped in their own collection."""
            try:
                cells, info = build_cells(
                    key=key, face_style=self.face_style,
                    density=self.density, smoothness=self.smoothness,
                    rings=self.rings, scale=self.scale, twist=self.twist,
                    nx=self.nx, ny=self.ny, nz=self.nz, gap=self.gap)
            except Exception as exc:
                self.report({'ERROR'}, "Build failed: %s" % exc)
                return {'CANCELLED'}
            if not cells:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}

            solid = info['solid']
            coll = bpy.data.collections.new("Saddle %s packing"
                                            % solid['name'])
            context.scene.collection.children.link(coll)

            made = 0
            pieces = []
            for i, (CV, CT, cfid) in enumerate(cells):
                me = bpy.data.meshes.new("Saddle %s cell %03d"
                                         % (solid['name'], i + 1))
                # give each cell its own origin at its centroid, so the
                # pieces can be moved apart, and put that offset back on
                # the object so the packing still reads as assembled
                origin = np.asarray(CV, float).mean(axis=0)
                local = [tuple(p) for p in (np.asarray(CV, float) - origin)]
                me.from_pydata(local, [], [tuple(t) for t in CT])
                cols = pc.PALETTE_RGBA
                col = cols[i % len(cols)]
                me.materials.append(_cell_material(
                    "Saddle %s %d" % (solid['name'], i % len(cols)), col))
                me.validate(clean_customdata=True)
                # Recalculate normals, exactly as patterns.build_object
                # does for the merged mesh.  Without this the cells are
                # built inside-out: you see through each one into its
                # interior and the packing reads as a scrambled mess,
                # which is precisely how the first separate-objects
                # build looked.
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                bm.to_mesh(me)
                bm.free()
                me.update()
                if self.smooth:
                    me.polygons.foreach_set('use_smooth',
                                            [True] * len(me.polygons))
                me.update()
                if self.smooth and self.face_style != 'NET':
                    _sc.mark_sharp(me, face_boundary_edges(CT, cfid))
                obj = bpy.data.objects.new(me.name, me)
                obj.location = tuple(float(x) for x in origin)
                coll.objects.link(obj)
                pieces.append(obj)
                made += 1

            # Parent the cells to an EMPTY root.  This is the house
            # convention for a generator that lays its output out as
            # separate objects, and the live rebuild depends on it:
            # with no root it picks one cell as the anchor and moves
            # every other cell by a delta transform relative to it, so
            # a rebuild collapses the packing onto the origin.  The
            # Empty gives the group a single unambiguous anchor.
            root = bpy.data.objects.new(
                "Saddle %s packing" % solid['name'], None)
            root.empty_display_size = 0.2
            coll.objects.link(root)
            for obj in pieces:
                obj.parent = root
            for o in context.selected_objects:
                o.select_set(False)
            root.select_set(True)
            context.view_layer.objects.active = root

            rep = info['packing']
            msg = ("Table 8.1 #%d %s: %d cells as separate objects, "
                   "%.1f%% of the block"
                   % (solid['number'], solid['name'], made,
                      100.0 * rep['ratio']))
            if not rep['fills']:
                tail = (" -- no packing found for this solid in its net; "
                        "showing one cell" if rep['copies'] <= 1
                        else " -- does NOT fill space alone")
                self.report({'WARNING'}, msg + tail)
                return {'FINISHED'}
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def execute(self, context):
            if not pdata.SOLIDS:
                self.report({'ERROR'}, "No verified saddle polyhedra")
                return {'CANCELLED'}
            key = self.solid if self.solid != 'NONE' else None
            if self.layout_kind == 'BLOCK' and self.separate:
                return self._execute_separate(context, key)
            try:
                V, T, fid, info = build(
                    key=key, face_style=self.face_style,
                    density=self.density, smoothness=self.smoothness,
                    rings=self.rings, scale=self.scale, twist=self.twist,
                    layout=self.layout_kind, nx=self.nx, ny=self.ny,
                    nz=self.nz, gap=self.gap)
            except Exception as exc:
                self.report({'ERROR'}, "Build failed: %s" % exc)
                return {'CANCELLED'}

            solid = info['solid']
            obj = pc.build_object(
                context, "Saddle %s" % solid['name'],
                [tuple(p) for p in V], [tuple(t) for t in T], list(fid),
                span=2.0, fit=True)
            if obj is None:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}

            me = obj.data
            ncrease = 0
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                me.update()
                if self.face_style != 'NET':
                    # crease along the branches: the edges where two
                    # different saddle faces meet
                    # crease on the per-FACE grouping, which in a
                    # packing is not the same as the colour grouping
                    ncrease = _sc.mark_sharp(
                        me, face_boundary_edges(
                            T, info.get('crease_id') or fid))

            msg = ("Table 8.1 #%d %s: %d faces"
                   % (solid['number'], solid['name'], info['faces']))
            if self.face_style == 'SPIDRON':
                msg += (", %d true nests, %d generalised"
                        % (info['true_nests'], info['generalised']))
            if ncrease:
                msg += ", %d branch creases" % ncrease
            rep = info.get('packing')
            if rep is not None:
                msg += (" | packing: %d cells, %.1f%% of the block"
                        % (rep['copies'], 100.0 * rep['ratio']))
                if not rep['fills']:
                    # say so rather than pass a partial packing off as
                    # a space filling -- many of Pearce's solids only
                    # fill space in combination with a partner cell
                    if rep['copies'] <= 1:
                        msg += (" -- no packing found for this solid in "
                                "its net; showing one cell")
                    else:
                        msg += " -- does NOT fill space alone"
                    self.report({'WARNING'}, msg)
                    return {'FINISHED'}
            self.report({'INFO'}, msg)
            return {'FINISHED'}

    _CLASSES = (MESH_OT_saddle_polyhedron_add,)

    def register():
        for c in _CLASSES:
            bpy.utils.register_class(c)

    def unregister():
        for c in reversed(_CLASSES):
            bpy.utils.unregister_class(c)

else:
    def register():
        pass

    def unregister():
        pass


def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-58s %s %s" % (name, "OK" if cond else "BAD", extra))

    print("saddle_polyhedron: %d solids offered" % len(pdata.SOLIDS))
    chk("at least one solid ships", bool(pdata.SOLIDS))

    for s in pdata.SOLIDS:
        tag = "#%d %s" % (s['number'], s['name'])
        for style in ('MINIMAL', 'RULED', 'SPIDRON', 'NET'):
            try:
                V, T, fid, info = build(key=s['key'], face_style=style,
                                        density=2, smoothness=6, rings=3)
                good = len(V) > 0 and len(T) > 0
            except Exception as exc:
                good, V, T = False, [], []
                chk("%s %s builds" % (tag, style), False, str(exc))
                continue
            chk("%s %s builds" % (tag, style), good,
                "%d verts %d tris" % (len(V), len(T)))
            if not good:
                continue
            chk("  fits the 2 m cube",
                abs(max(psurf.mesh_extent(V)) - 2.0) < 1e-6)
            if style != 'NET':
                chk("  not collapsed", psurf.aspect(V) >= 0.2,
                    "aspect %.3f" % psurf.aspect(V))
        # the spidron honesty gate: the reported split must match the
        # exact per-face test, not a per-solid assumption
        flags = true_nest_faces(s)
        _, _, _, info = build(key=s['key'], face_style='SPIDRON', rings=3)
        chk("%s: nest split reported truthfully" % tag,
            info['true_nests'] == sum(1 for x in flags if x)
            and info['generalised'] == sum(1 for x in flags if not x),
            "%d true / %d generalised of %d"
            % (info['true_nests'], info['generalised'], len(s['faces'])))

    # --- space filling ------------------------------------------
    print("  space filling:")
    for s in pdata.SOLIDS:
        tag = "#%d %s" % (s['number'], s['name'])
        try:
            _V, _T, fid, info = build(key=s['key'], layout='BLOCK',
                                      nx=1, ny=1, nz=1, density=2,
                                      smoothness=4)
            rep = info['packing']
        except Exception as exc:
            chk("%s packs" % tag, False, str(exc))
            continue
        chk("%s: packing built" % tag, rep['copies'] >= 1,
            "%d cells, %.3f of the block, fills=%s"
            % (rep['copies'], rep['ratio'], rep['fills']))
        chk("  overlap reported truthfully",
            rep['self_intersecting'] == (rep['overused_faces'] > 0),
            "overused=%d" % rep['overused_faces'])
        chk("  one colour group per cell",
            len(set(fid)) == rep['copies'])
        if rep['fills']:
            # a block twice as wide must hold twice as many cells, or
            # the packing is not periodic and the fill was a fluke
            _V2, _T2, _f2, i2 = build(key=s['key'], layout='BLOCK',
                                      nx=2, ny=1, nz=1, density=2,
                                      smoothness=4)
            chk("  doubling the block doubles the cells",
                i2['packing']['copies'] == 2 * rep['copies']
                and i2['packing']['fills'],
                "%d -> %d" % (rep['copies'], i2['packing']['copies']))

    # --- separate cells ------------------------------------------
    print("  separate cells:")
    for s in pdata.SOLIDS:
        tag = "#%d %s" % (s['number'], s['name'])
        cells, info = build_cells(key=s['key'], density=2, smoothness=4,
                                  nx=1, ny=1, nz=1)
        rep = info['packing']
        chk("%s: one piece per cell" % tag,
            len(cells) == rep['copies'], "%d" % len(cells))
        if not cells:
            continue
        # the pieces must stay in register: their union has to be the
        # same 2 m block the merged build produces, or the packing has
        # been scattered by per-cell fitting
        allv = np.concatenate([c[0] for c in cells], axis=0)
        ext = allv.max(axis=0) - allv.min(axis=0)
        chk("  union still fits the 2 m block",
            abs(float(ext.max()) - 2.0) < 1e-6, "%.6f" % float(ext.max()))
        chk("  cells are not individually rescaled",
            all(float(np.asarray(c[0]).max(axis=0).max()
                      - np.asarray(c[0]).min(axis=0).min()) <= 2.0 + 1e-6
                for c in cells))
        merged_v, merged_t, _f, _i = build(key=s['key'], layout='BLOCK',
                                           nx=1, ny=1, nz=1, density=2,
                                           smoothness=4)
        chk("  same triangle count as the merged block",
            sum(len(c[1]) for c in cells) == len(merged_t),
            "%d vs %d" % (sum(len(c[1]) for c in cells), len(merged_t)))

    print("RESULT:", "OK" if ok else "BAD")
    if not ok:
        raise AssertionError("saddle_polyhedron self-test failed")
