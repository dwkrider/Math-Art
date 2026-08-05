
# Symmetrohedron Generator for Blender
#
# Symmetrohedra, after C. S. Kaplan & G. W. Hart, "Symmetrohedra:
# Polyhedra from Symmetric Placement of Regular Polygons" (Bridges
# 2001): regular polygons are placed symmetrically on the rotation
# axes of a symmetry group (tetrahedral 3/3/2, octahedral 4/3/2 or
# icosahedral 5/3/2), and the convex hull fills the gaps.
#
# Here every polygon is inscribed in the unit sphere (vertices on the
# sphere), each axis class has a multiplier (polygon with
# multiplier x axis-order sides, 0 = none) and a size slider, and the
# result is the convex hull with coplanar faces merged. Exact
# edge-to-edge Kaplan-Hart solutions appear at particular size ratios
# -- explore with the sliders.

bl_info = {
    "name": "Symmetrohedra",
    "author": "Math Art project (after Kaplan & Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Symmetrohedron",
    "description": "Polyhedra from symmetric placement of regular polygons",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi

PHI = (1 + 5 ** 0.5) / 2


def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _dedup_axes(dirs):
    """Keep one representative per +-axis, both directions returned."""
    axes = []
    for d in dirs:
        d = _unit(d)
        if not any(abs(d[0] + a[0]) + abs(d[1] + a[1]) + abs(d[2] + a[2])
                   < 1e-6 or
                   abs(d[0] - a[0]) + abs(d[1] - a[1]) + abs(d[2] - a[2])
                   < 1e-6 for a in axes):
            axes.append(d)
    out = []
    for a in axes:
        out.append(a)
        out.append(tuple(-x for x in a))
    return out


def symmetry_axes(group):
    """{order: [directions (both ends)]} for T, O, I."""
    if group == 'T':
        three = _dedup_axes([(1, 1, 1), (1, -1, -1), (-1, 1, -1),
                             (-1, -1, 1)])
        two = _dedup_axes([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        return {3: three, 2: two}
    if group == 'O':
        four = _dedup_axes([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        three = _dedup_axes([(1, 1, 1), (1, -1, 1), (1, 1, -1),
                             (1, -1, -1)])
        two = _dedup_axes([(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
                           (0, 1, 1), (0, 1, -1)])
        return {4: four, 3: three, 2: two}
    if group == 'I':
        five = _dedup_axes([(0, 1, PHI), (0, 1, -PHI), (1, PHI, 0),
                            (1, -PHI, 0), (PHI, 0, 1), (-PHI, 0, 1)])
        three = _dedup_axes([(1, 1, 1), (1, -1, -1), (-1, 1, -1),
                             (-1, -1, 1),
                             (0, 1 / PHI, PHI), (0, -1 / PHI, PHI),
                             (1 / PHI, PHI, 0), (-1 / PHI, PHI, 0),
                             (PHI, 0, 1 / PHI), (PHI, 0, -1 / PHI)])
        two = _dedup_axes([(1, 0, 0), (0, 1, 0), (0, 0, 1),
                           (1, PHI, PHI + 1), (1, -PHI, PHI + 1),
                           (1, PHI, -PHI - 1), (1, -PHI, -PHI - 1),
                           (PHI, PHI + 1, 1), (-PHI, PHI + 1, 1),
                           (PHI, -PHI - 1, 1), (-PHI, -PHI - 1, 1),
                           (PHI + 1, 1, PHI), (PHI + 1, -1, PHI),
                           (PHI + 1, 1, -PHI), (PHI + 1, -1, -PHI)])
        return {5: five, 3: three, 2: two}
    raise ValueError(group)


def axial_polygons(group, mults, radii, phases):
    """Place an (mult*order)-gon on every axis end of each active
    class, inscribed in the unit sphere. Returns list of polygons
    (each a list of 3D points)."""
    axes = symmetry_axes(group)
    orders = sorted(axes.keys(), reverse=True)   # e.g. [5, 3, 2]
    all_dirs = [d for o in orders for d in axes[o]]
    polys = []
    for ci, order in enumerate(orders):
        mult = mults[ci]
        if mult <= 0:
            continue
        m = mult * order
        r = min(max(radii[ci], 0.02), 0.999)
        h = math.sqrt(1.0 - r * r)
        phase = math.radians(phases[ci])
        for u in axes[order]:
            # covariant alignment: aim vertex 0 at the projection of a
            # neighbouring axis (ties differ by the axis stabilizer, and
            # an (mult*order)-gon is invariant under it)
            best = None
            bestd = -2.0
            for v in all_dirs:
                d = sum(u[k] * v[k] for k in range(3))
                if 0.2 < d < 0.999 and d > bestd:
                    bestd = d
                    best = v
            if best is None:
                best = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0, 0)
            e1 = tuple(best[k] - bestd * u[k] for k in range(3))
            e1 = _unit(e1)
            e2 = (u[1] * e1[2] - u[2] * e1[1],
                  u[2] * e1[0] - u[0] * e1[2],
                  u[0] * e1[1] - u[1] * e1[0])
            poly = []
            for i in range(m):
                a = phase + 2 * pi * i / m
                poly.append(tuple(u[k] * h + r * (cos(a) * e1[k]
                                                  + sin(a) * e2[k])
                                  for k in range(3)))
            polys.append(poly)
    return polys


try:
    import bpy
    import bmesh
    from bpy.props import (FloatProperty, EnumProperty, IntProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_symmetrohedron_add(bpy.types.Operator):
        """Symmetrohedron: regular polygons on the symmetry axes,
        convex hull filling the gaps (after Kaplan & Hart)"""
        bl_idname = "mesh.symmetrohedron_add"
        bl_label = "Symmetrohedron"
        bl_options = {'REGISTER', 'UNDO'}

        group: EnumProperty(
            name="Symmetry",
            items=[('I', "Icosahedral (5,3,2)", ""),
                   ('O', "Octahedral (4,3,2)", ""),
                   ('T', "Tetrahedral (3,3,2)", "")],
            default='I')
        mult1: IntProperty(
            name="Axis-1 Multiplier", default=1, min=0, max=4,
            description="Polygon = multiplier x axis order sides "
                        "(0 = no polygon on this class)")
        mult2: IntProperty(name="Axis-2 Multiplier", default=1,
                           min=0, max=4)
        mult3: IntProperty(name="Axis-3 Multiplier", default=0,
                           min=0, max=4)
        r1: FloatProperty(name="Axis-1 Size", default=0.45,
                          min=0.05, max=0.98)
        r2: FloatProperty(name="Axis-2 Size", default=0.31,
                          min=0.05, max=0.98)
        r3: FloatProperty(name="Axis-3 Size", default=0.3,
                          min=0.05, max=0.98)
        ph1: FloatProperty(name="Axis-1 Phase", default=0.0,
                           min=-90.0, max=90.0)
        ph2: FloatProperty(name="Axis-2 Phase", default=0.0,
                           min=-90.0, max=90.0)
        ph3: FloatProperty(name="Axis-3 Phase", default=0.0,
                           min=-90.0, max=90.0)
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "By Face Size",
                    "One material per face size, shared with the "
                    "Conway generator's palette (view with Material "
                    "Preview or Solid shading set to Material "
                    "color)"),
                   ('NONE', "None", "No materials")],
            default='SIDES')
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed polyhedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo "
                    "Style Geometry Nodes modifier (Border and "
                    "Thickness stay editable on the modifier)"),
                   ('WIRE', "Wireframe",
                    "Struts along the edges (Wireframe modifier)"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width (fraction of "
                        "the face)")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness for the Leonardo "
                        "and Wireframe styles")
        facet_depth: FloatProperty(name="Depth", default=0.15, min=0.01,
                                   max=2.0,
                                   description="Face Segments inward depth")
        facet_gap: FloatProperty(name="Bevel Gap", default=0.0, min=0.0,
                                 max=0.5,
                                 description="Gap between face segments")
        facet_explode: FloatProperty(name="Explode", default=0.1, min=0.0,
                                     max=5.0,
                                     description="Move segments outward")
        facet_separate: BoolProperty(
            name="Separate Meshes", default=False,
            description="Each face segment as its own object")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = {3: (0.90, 0.36, 0.23), 4: (0.27, 0.52, 0.79),
                    5: (0.30, 0.69, 0.42), 6: (0.95, 0.77, 0.29),
                    7: (0.62, 0.40, 0.75), 8: (0.25, 0.72, 0.72),
                    9: (0.91, 0.56, 0.71), 10: (0.55, 0.60, 0.29),
                    12: (0.52, 0.45, 0.40)}

        @classmethod
        def _material_for(cls, n):
            name = f"Conway {n}-gon"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE.get(
                    n, (0.5 + 0.5 * math.sin(n), 0.55,
                        0.5 + 0.5 * math.cos(n)))
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb,
                                                               1.0)
                    bsdf.inputs["Roughness"].default_value = 0.5
            return mat

        def execute(self, context):
            mults = (self.mult1, self.mult2, self.mult3)
            if not any(m > 0 for m in mults):
                self.report({'ERROR'}, "at least one class needs a polygon")
                return {'CANCELLED'}
            polys = axial_polygons(self.group, mults,
                                   (self.r1, self.r2, self.r3),
                                   (self.ph1, self.ph2, self.ph3))
            bm = bmesh.new()
            vlist = []
            for poly in polys:
                for p in poly:
                    vlist.append(bm.verts.new(
                        tuple(c * self.scale for c in p)))
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
            bmesh.ops.convex_hull(bm, input=bm.verts[:])
            unused = [v for v in bm.verts if not v.link_faces]
            bmesh.ops.delete(bm, geom=unused, context='VERTS')
            bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.5),
                                     verts=bm.verts[:], edges=bm.edges[:])
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            me = bpy.data.meshes.new("Symmetrohedron")
            bm.to_mesh(me)
            bm.free()
            fsz = [len(p.vertices) for p in me.polygons]
            attr = me.attributes.new("ngon_sides", 'INT', 'FACE')
            attr.data.foreach_set('value', fsz)
            if self.coloring == 'SIDES':
                lut = {}
                for n in sorted(set(fsz)):
                    lut[n] = len(me.materials)
                    me.materials.append(self._material_for(n))
                me.polygons.foreach_set('material_index',
                                        [lut[s] for s in fsz])
            me.update()
            if self.style == 'FACETS':
                Vf = [tuple(v.co) for v in me.vertices]
                Ff = [list(p.vertices) for p in me.polygons]
                bpy.data.meshes.remove(me)
                try:
                    from . import facet_style
                except ImportError:
                    import facet_style
                mat = (self._material_for
                       if self.coloring == 'SIDES' else None)
                facet_style.emit_facets(
                    context, Vf, Ff, "Symmetrohedron",
                    self.facet_depth, self.facet_gap,
                    self.facet_explode, self.facet_separate, mat)
                self.report({'INFO'}, f"{len(Ff)} face segments")
                return {'FINISHED'}
            obj = bpy.data.objects.new("Symmetrohedron", me)
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
            sizes = {}
            for p in me.polygons:
                sizes[len(p.vertices)] = sizes.get(len(p.vertices), 0) + 1
            desc = ", ".join(f"{n}x{k}-gon"
                             for k, n in sorted(sizes.items()))
            self.report({'INFO'},
                        f"V={len(me.vertices)} F={len(me.polygons)}: {desc}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'group')
            col = lay.column(align=True)
            for k in ('mult1', 'r1', 'ph1'):
                col.prop(self, k)
            col = lay.column(align=True)
            for k in ('mult2', 'r2', 'ph2'):
                col.prop(self, k)
            col = lay.column(align=True)
            for k in ('mult3', 'r3', 'ph3'):
                col.prop(self, k)
            lay.prop(self, 'coloring')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'FACETS':
                lay.prop(self, 'facet_depth')
                lay.prop(self, 'facet_gap')
                lay.prop(self, 'facet_explode')
                lay.prop(self, 'facet_separate')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.symmetrohedron_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_symmetrohedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_symmetrohedron_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for group in ('T', 'O', 'I'):
            ax = symmetry_axes(group)
            print(group, {o: len(d) // 2 for o, d in ax.items()},
                  "axes per class")
            polys = axial_polygons(group, (1, 1, 0), (0.45, 0.38, 0.3),
                                   (0, 0, 0))
            print(f"  {len(polys)} polygons placed")
