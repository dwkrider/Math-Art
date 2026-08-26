# Slice for Fabrication: turn any Math Art object into flat parts that
# a laser cutter or knife plotter can cut, and a person can glue or slot
# back together.
#
# Four constructions, all of them old.  STACKED is the parallel-section
# model that Olaus Henrici showed at South Kensington in 1876 and that
# Alexander von Brill and Felix Klein were selling out of Munich in
# cardboard from about 1870 -- the technique Naum Gabo is thought to
# have met there around 1910 and carried into constructivist sculpture.
# INTERLOCKED is John Sharp's Sliceform: two families of parallel slices
# slotted into each other, every crossing a hinge, so a paper model
# collapses flat in two different ways and moves through a continuum of
# shapes between them.  RADIAL is the globe -- meridians and parallels
# -- because N planes through one axis all meet on that single line and
# so cannot interlock with one another at all.  RIBS are sections taken
# perpendicular to a curve.
#
# The joint in all of them is one rule, stated exactly by Luecking: cut
# each piece from its own rim inward to the MIDPOINT of the span the two
# pieces share, from opposite ends, and they pass through one another
# and stop flush.  Nothing is glued.
#
# The mathematics is in the `slicing` engine package (numpy only, no
# bpy) so it self-tests headlessly; this module is the Blender layer
# over it, and everything it produces is in real millimetres.
#
# References:
# - John Sharp, "Sliceforms: Mathematical Models from Paper Sections",
#   Tarquin Publications, Stradbroke, 1995 -- the book that named the
#   technique and works out the two-family construction.
# - John Sharp, "Sliceform Sculptures - a Bridge between Art and
#   Mathematics", Bridges 1998, pp. 275-276 -- the history from
#   Henrici and Brill's models through Gabo, and the kinetic property
#   of the crossed-slice construction.
# - Stephen Luecking, "Creating Sliceforms with 3D Modelers", Bridges
#   2006, pp. 631-638 -- the complementary half-slot rule this module
#   implements, stated in full.
# - Yongquan Lu and Erik D. Demaine, "A Pattern Tracing System for
#   Generating Paper Sliceform Artwork", Bridges 2015, pp. 367-370 --
#   slits spanning alternately the top or bottom half, and the
#   convention of separating cut from score by colour so a laser can
#   map colour to power.
# - Jace Miller and Ergun Akleman, "Edge-Based Intersected Polyhedral
#   Paper Sculptures Constructed by Interlocking Slitted Planar
#   Pieces", Bridges 2008, pp. 259-264 -- the same midpoint-slit rule
#   for planes that are not parallel, generalising Hart's
#   slide-togethers.
# - George W. Hart, "Laser-Cut Plywood and Cable-Tie Sculptures",
#   Bridges 2015, pp. 77-84 -- fabrication practice for laser-cut
#   plywood sculpture, including bevelling a butt joint by half the
#   angle between the two planes' normals.
# - Caroline Bowen, "Cut Colored Paper Sculptures of 3D Contour Plots
#   of the Real and Imaginary Parts of Complex Functions", Bridges
#   2017, pp. 375-378 -- the stacked technique applied to a
#   heightfield, which is what the Relief Panel generator produces.
# - Bonaventura Cavalieri, "Geometria indivisibilibus continuorum nova
#   quadam ratione promota", Bologna, 1635 -- the principle that
#   equally spaced parallel sections reconstruct the volume in the
#   limit, which is what a stacked model is an approximation to.

import json
import math

bl_info = {
    "name": "Slice for Fabrication",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Styles",
    "description": "Slice a mesh into flat parts for laser cutting: "
                   "stacked, interlocked sliceform, radial or ribs, "
                   "with SVG and DXF export",
    "category": "Object",
}

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty, StringProperty)
    from bpy_extras.io_utils import ExportHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from .slicing import build as _build
    from .slicing import sections as _sections
    from .slicing import svg as _svg
    from .slicing import dxf as _dxf
    from .slicing.drawing import Drawing
except ImportError:                       # headless / flat import
    from slicing import build as _build
    from slicing import sections as _sections
    from slicing import svg as _svg
    from slicing import dxf as _dxf
    from slicing.drawing import Drawing


# Blender's scene unit is the metre; the slicer works in millimetres.
# Preview and layout geometry is therefore emitted in metres, so a
# 200 mm model measures 0.2 in the viewport and reads correctly under
# Metric units -- and the exporters multiply back up.
MM = 0.001

DRAWING_KEY = 'math_art_slice_drawing'

AXIS_ITEMS = [('X', "X", "Slice along the X axis"),
              ('Y', "Y", "Slice along the Y axis"),
              ('Z', "Z", "Slice along the Z axis")]


def drawing_to_json(drawing):
    return json.dumps({
        'name': drawing.name,
        'sheet_width': drawing.sheet_width,
        'sheet_height': drawing.sheet_height,
        'sheets': [[[e.layer, e.points, e.closed, e.tag]
                    for e in s.entities] for s in drawing.sheets],
    })


def drawing_from_json(text):
    data = json.loads(text)
    d = Drawing(data['name'], data['sheet_width'], data['sheet_height'])
    for i, ents in enumerate(data['sheets']):
        sheet = d.sheet(i)
        for layer, pts, closed, tag in ents:
            sheet.add(layer, pts, closed, tag)
    return d


if _IN_BLENDER:

    # -------------------------------------------------------------- #
    #  mesh input                                                    #
    # -------------------------------------------------------------- #

    def _has_boundary(mesh):
        """True if any edge has fewer than two faces -- an open
        surface, which does not section into closed outlines."""
        count = {}
        for poly in mesh.polygons:
            for ek in poly.edge_keys:
                count[ek] = count.get(ek, 0) + 1
        return any(c < 2 for c in count.values())

    def _evaluated_mesh(context, obj, solidify):
        """World-space vertices and triangles of the evaluated object.

        The evaluated mesh is used so modifiers count -- a relief panel
        carrying a Displace, or anything with a Subdivision on it,
        would otherwise be sliced in its undisplaced form.
        """
        dg = context.evaluated_depsgraph_get()
        temp_mod = None
        if solidify:
            temp_mod = obj.modifiers.new("MathArtSliceSolidify", 'SOLIDIFY')
            temp_mod.thickness = solidify
            temp_mod.offset = 0.0
            temp_mod.use_rim = True
            temp_mod.use_rim_only = False
            dg = context.evaluated_depsgraph_get()
        try:
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            mw = obj.matrix_world
            verts = [tuple(mw @ v.co) for v in me.vertices]
            tris = []
            for poly in me.polygons:
                idx = list(poly.vertices)
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
            open_surface = _has_boundary(me)
            ev.to_mesh_clear()
        finally:
            if temp_mod is not None:
                obj.modifiers.remove(temp_mod)
        return verts, tris, open_surface

    def _curve_points(context, exclude):
        """Sampled points of a selected curve object, if there is one."""
        for obj in context.selected_objects:
            if obj is exclude or obj.type != 'CURVE':
                continue
            dg = context.evaluated_depsgraph_get()
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            mw = obj.matrix_world
            pts = [tuple(mw @ v.co) for v in me.vertices]
            ev.to_mesh_clear()
            if len(pts) >= 2:
                return pts
        return None

    # -------------------------------------------------------------- #
    #  output geometry                                               #
    # -------------------------------------------------------------- #

    def _collection(context, name):
        old = bpy.data.collections.get(name)
        if old is not None:
            for obj in list(old.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            return old
        coll = bpy.data.collections.new(name)
        context.scene.collection.children.link(coll)
        return coll

    def _plate_mesh(name, part, plane, thickness):
        """A part extruded to material thickness, in its own plane."""
        u, v, n = plane.u, plane.v, plane.normal
        d = plane.offset
        half = 0.5 * thickness
        rings = [part.outer] + part.holes
        verts, faces = [], []
        for ring in rings:
            base = len(verts)
            m = len(ring)
            for s in (-half, half):
                for x, y in ring:
                    verts.append(tuple(
                        ((d + s) * n[k] + x * u[k] + y * v[k]) * MM
                        for k in range(3)))
            for i in range(m):
                j = (i + 1) % m
                faces.append([base + i, base + j,
                              base + m + j, base + m + i])
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        me.update()
        return me

    def _sheet_mesh(name, sheet, dy):
        """One sheet's paths as an edge-only mesh, for looking at."""
        verts, edges = [], []
        for e in sheet.ordered():
            base = len(verts)
            for x, y in e.points:
                verts.append((x * MM, (y + dy) * MM, 0.0))
            for i in range(len(e.points) - 1):
                edges.append((base + i, base + i + 1))
            if e.closed and len(e.points) > 2:
                edges.append((base + len(e.points) - 1, base))
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, edges, [])
        me.validate(clean_customdata=True)
        me.update()
        return me

    # -------------------------------------------------------------- #
    #  the slicing operator                                          #
    # -------------------------------------------------------------- #

    class OBJECT_OT_fabrication_slice(bpy.types.Operator):
        """Slice the active mesh into flat parts for laser cutting.

        Builds an assembled 3-D preview and a nested, real-millimetre
        sheet layout, then reports every joint that will not work"""
        bl_idname = "object.fabrication_slice"
        bl_label = "Slice for Fabrication"
        bl_options = {'REGISTER', 'UNDO'}

        technique: EnumProperty(
            name="Technique",
            items=[
                ('STACKED', "Stacked Slices",
                 "Parallel cross sections to glue into a stack, with "
                 "dowel holes to line them up"),
                ('INTERLOCKED', "Interlocked Slices",
                 "Two crossing families of slices, slotted into one "
                 "another -- a sliceform. Uses less material"),
                ('RADIAL', "Radial Slices",
                 "Half-fins radiating from an axis, slotted into a "
                 "stack of rings. Suits round, symmetrical shapes"),
                ('RIBS', "Ribs along a Curve",
                 "Sections perpendicular to a selected curve, slotted "
                 "onto a spine. Suits long organic shapes"),
            ],
            default='STACKED',
            description="How the model is cut up and put back together")

        target_size: FloatProperty(
            name="Model Size", default=200.0, min=1.0, max=5000.0,
            description="Longest dimension of the finished model, in "
                        "millimetres. Generators build into a 2 m cube, "
                        "so this is what makes the parts a real size")
        thickness: FloatProperty(
            name="Material Thickness", default=3.0, min=0.05, max=50.0,
            description="Thickness of the sheet stock, in millimetres")
        kerf: FloatProperty(
            name="Kerf", default=0.15, min=0.0, max=3.0,
            description="Width of material the beam burns away, in "
                        "millimetres. The outlines are offset by half "
                        "of it so the cut parts come out to size")
        clearance: FloatProperty(
            name="Joint Clearance", default=0.0, min=-1.0, max=2.0,
            description="Added to the slot width, in millimetres. Zero "
                        "is a press fit; a little more gives the free "
                        "hinge that lets a paper sliceform collapse")
        flexible: BoolProperty(
            name="Flexible Material", default=False,
            description="Allow joints that can only be assembled by "
                        "flexing the parts, as paper and card can and "
                        "plywood cannot")

        axis: EnumProperty(
            name="Axis", items=AXIS_ITEMS, default='Z',
            description="The single slicing direction, for Stacked, "
                        "Radial and the Ribs spine")
        axis_a: EnumProperty(
            name="First Axis", items=AXIS_ITEMS, default='X',
            description="Direction of the first family of slices")
        axis_b: EnumProperty(
            name="Second Axis", items=AXIS_ITEMS, default='Y',
            description="Direction of the second family, which must "
                        "differ from the first")
        count_a: IntProperty(
            name="First Count", default=8, min=1, max=200,
            description="Number of slices in the first direction")
        count_b: IntProperty(
            name="Second Count", default=8, min=1, max=200,
            description="Number of slices in the second direction")
        radial_count: IntProperty(
            name="Fins", default=8, min=2, max=64,
            description="Number of radial fins around the axis")
        ring_count: IntProperty(
            name="Rings", default=4, min=1, max=64,
            description="Number of rings the fins slot into")
        rib_count: IntProperty(
            name="Ribs", default=12, min=2, max=200,
            description="Number of ribs along the curve")

        dowels: IntProperty(
            name="Dowels", default=2, min=0, max=8,
            description="Alignment dowel holes through the whole "
                        "stack, so the glued slices line up")
        dowel_diameter: FloatProperty(
            name="Dowel Diameter", default=4.0, min=0.5, max=30.0,
            description="Diameter of the dowel, in millimetres")

        flare: FloatProperty(
            name="Notch Factor", default=0.0, min=0.0, max=1.0,
            description="Widens the mouth of each slot by this "
                        "fraction of its width, so a big assembly that "
                        "has racked slightly still goes together")
        flare_angle: FloatProperty(
            name="Notch Angle", default=45.0, min=5.0, max=89.0,
            description="Angle the flared mouth narrows back down at")
        tool_diameter: FloatProperty(
            name="Tool Diameter", default=0.0, min=0.0, max=20.0,
            description="Diameter of a round cutting tool, in "
                        "millimetres. Above zero, inside corners get "
                        "dog-bone relief so a router can reach them. "
                        "Leave at zero for a laser")

        sheet_width: FloatProperty(
            name="Sheet Width", default=600.0, min=10.0, max=5000.0,
            description="Width of the stock sheet, in millimetres")
        sheet_height: FloatProperty(
            name="Sheet Height", default=400.0, min=10.0, max=5000.0,
            description="Height of the stock sheet, in millimetres")
        margin: FloatProperty(
            name="Margin", default=5.0, min=0.0, max=100.0,
            description="Gap left between parts and around the sheet "
                        "edge, in millimetres")
        label_height: FloatProperty(
            name="Label Size", default=4.0, min=0.0, max=50.0,
            description="Height of the engraved part labels, in "
                        "millimetres. Zero leaves the parts unlabelled")

        close_surface: BoolProperty(
            name="Close Open Surfaces", default=True,
            description="Give an open surface a thickness before "
                        "slicing it. A surface with a boundary has no "
                        "inside, so it cannot be sliced into parts")
        shell_thickness: FloatProperty(
            name="Shell Thickness", default=0.05, min=0.001, max=10.0,
            description="Thickness given to an open surface, in the "
                        "object's own units")
        preview: BoolProperty(
            name="Assembled Preview", default=True,
            description="Build the assembled 3-D model as well as the "
                        "flat sheet layout")
        explode: FloatProperty(
            name="Explode", default=0.0, min=0.0, max=5.0,
            description="Pull the previewed slices apart along their "
                        "own normals, to see how they fit")

        @classmethod
        def poll(cls, context):
            obj = context.active_object
            return obj is not None and obj.type == 'MESH'

        def execute(self, context):
            obj = context.active_object
            solidify = 0.0
            verts, tris, open_surface = _evaluated_mesh(context, obj, 0.0)
            if open_surface and self.close_surface:
                solidify = self.shell_thickness
                verts, tris, open_surface = _evaluated_mesh(
                    context, obj, solidify)
            if open_surface:
                self.report(
                    {'ERROR'},
                    "This surface has a boundary, so it has no inside to "
                    "slice. Turn on Close Open Surfaces, or give it "
                    "thickness with a Solidify modifier.")
                return {'CANCELLED'}
            if not tris:
                self.report({'ERROR'}, "The active mesh has no faces.")
                return {'CANCELLED'}

            settings = _build.Settings(
                technique=self.technique, target_size=self.target_size,
                thickness=self.thickness, kerf=self.kerf,
                clearance=self.clearance, flexible=self.flexible,
                sheet_width=self.sheet_width,
                sheet_height=self.sheet_height, margin=self.margin,
                axis=self.axis, axis_a=self.axis_a, axis_b=self.axis_b,
                count_a=self.count_a, count_b=self.count_b,
                radial_count=self.radial_count,
                ring_count=self.ring_count, rib_count=self.rib_count,
                dowels=self.dowels,
                dowel_diameter=self.dowel_diameter,
                flare=self.flare, flare_angle=self.flare_angle,
                tool_diameter=self.tool_diameter,
                label_height=self.label_height,
                curve=_curve_points(context, obj)
                if self.technique == 'RIBS' else None)

            try:
                drawing, families, report = _build.build(
                    verts, tris, settings, obj.name)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

            layout = _collection(context, f"{obj.name} Layout")
            dy = 0.0
            for sheet in drawing.sheets:
                me = _sheet_mesh(f"{obj.name} sheet {sheet.index + 1}",
                                 sheet, dy)
                sob = bpy.data.objects.new(me.name, me)
                layout.objects.link(sob)
                dy += sheet.height + 20.0
            layout[DRAWING_KEY] = drawing_to_json(drawing)

            if self.preview:
                prev = _collection(context, f"{obj.name} Slices")
                for fam in families:
                    for plane in fam.planes:
                        for part in plane.parts:
                            me = _plate_mesh(
                                f"{obj.name} {part.label}", part, plane,
                                self.thickness)
                            pob = bpy.data.objects.new(me.name, me)
                            if self.explode:
                                n = plane.normal
                                pob.location = tuple(
                                    n[k] * plane.offset * self.explode * MM
                                    for k in range(3))
                            prev.objects.link(pob)

            summary = _build.summarise(report)
            level = 'WARNING' if (report['nest'].get('oversize')
                                  or report.get('faults')
                                  or report['cut'].get('failed')
                                  or (report.get('interlock') or {}).get(
                                      'unassemblable')) else 'INFO'
            self.report({level}, summary)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.prop(self, 'technique')

            box = lay.box()
            box.label(text="Size and Material")
            box.prop(self, 'target_size')
            box.prop(self, 'thickness')
            box.prop(self, 'kerf')

            box = lay.box()
            box.label(text="Slices")
            if self.technique == 'INTERLOCKED':
                row = box.row()
                row.prop(self, 'axis_a')
                row.prop(self, 'axis_b')
                if self.axis_a == self.axis_b:
                    box.label(text="The two axes must differ",
                              icon='ERROR')
                row = box.row()
                row.prop(self, 'count_a')
                row.prop(self, 'count_b')
            else:
                box.prop(self, 'axis')
            if self.technique == 'STACKED':
                sub = box.column()
                sub.enabled = False
                sub.label(text="Slice count follows the thickness")
                box.prop(self, 'dowels')
                if self.dowels:
                    box.prop(self, 'dowel_diameter')
            elif self.technique == 'RADIAL':
                row = box.row()
                row.prop(self, 'radial_count')
                row.prop(self, 'ring_count')
            elif self.technique == 'RIBS':
                box.prop(self, 'rib_count')
                box.label(text="Select a curve object as well",
                          icon='INFO')

            if self.technique != 'STACKED':
                box = lay.box()
                box.label(text="Joints")
                box.prop(self, 'clearance')
                box.prop(self, 'flexible')
                box.prop(self, 'flare')
                if self.flare:
                    box.prop(self, 'flare_angle')
                box.prop(self, 'tool_diameter')

            box = lay.box()
            box.label(text="Sheet")
            row = box.row()
            row.prop(self, 'sheet_width')
            row.prop(self, 'sheet_height')
            box.prop(self, 'margin')
            box.prop(self, 'label_height')

            box = lay.box()
            box.label(text="Input and Output")
            box.prop(self, 'close_surface')
            if self.close_surface:
                box.prop(self, 'shell_thickness')
            box.prop(self, 'preview')
            if self.preview:
                box.prop(self, 'explode')

    # -------------------------------------------------------------- #
    #  the export operator                                           #
    # -------------------------------------------------------------- #

    class OBJECT_OT_fabrication_slice_export(bpy.types.Operator,
                                             ExportHelper):
        """Write the sliced sheet layout to SVG and/or DXF files,
        one file per sheet, in real millimetres"""
        bl_idname = "object.fabrication_slice_export"
        bl_label = "Export Slice Layout"
        bl_options = {'REGISTER'}

        filename_ext = ".svg"
        filter_glob: StringProperty(default="*.svg;*.dxf",
                                    options={'HIDDEN'})

        write_svg: BoolProperty(
            name="SVG", default=True,
            description="Write SVG, with one group per operation and "
                        "the operation carried as stroke colour")
        write_dxf: BoolProperty(
            name="DXF", default=True,
            description="Write DXF R12, with one named layer per "
                        "operation. One drawing unit is one millimetre")
        include_frame: BoolProperty(
            name="Sheet Outline", default=True,
            description="Include a non-cutting rectangle showing the "
                        "edge of the stock")

        @classmethod
        def poll(cls, context):
            return _find_layout(context) is not None

        def execute(self, context):
            coll = _find_layout(context)
            if coll is None:
                self.report({'ERROR'}, "No sliced layout found. Run "
                                       "Slice for Fabrication first.")
                return {'CANCELLED'}
            drawing = drawing_from_json(coll[DRAWING_KEY])

            import os
            stem, _ext = os.path.splitext(self.filepath)
            written = []
            if self.write_svg:
                written += _svg.write(
                    drawing, lambda i: f"{stem}_{i + 1:02d}.svg")
            if self.write_dxf:
                written += _dxf.write(
                    drawing, lambda i: f"{stem}_{i + 1:02d}.dxf")
            if not written:
                self.report({'ERROR'}, "Nothing to write: pick SVG, DXF "
                                       "or both.")
                return {'CANCELLED'}
            self.report({'INFO'},
                        f"Wrote {len(written)} file(s) for "
                        f"{len(drawing.sheets)} sheet(s)")
            return {'FINISHED'}

    def _find_layout(context):
        """The layout collection for the active object, or any."""
        obj = context.active_object
        if obj is not None:
            coll = bpy.data.collections.get(f"{obj.name} Layout")
            if coll is not None and DRAWING_KEY in coll:
                return coll
        for coll in bpy.data.collections:
            if DRAWING_KEY in coll:
                return coll
        return None

    _CLASSES = (OBJECT_OT_fabrication_slice,
                OBJECT_OT_fabrication_slice_export)

    def register():
        for cls in _CLASSES:
            bpy.utils.register_class(cls)

    def unregister():
        for cls in reversed(_CLASSES):
            bpy.utils.unregister_class(cls)


def _selftest():
    """Round-trip the drawing through the custom-property JSON.

    The exporter reads the layout back from a string stored on the
    collection rather than re-slicing, so this serialisation is the
    join between slicing and export -- and it has to survive a file
    save and reload, which is exactly what the round trip checks.
    """
    d = Drawing('t', 100.0, 60.0)
    s = d.sheet(0)
    s.add('CUT', [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0)], True, 'p1')
    s.add('ENGRAVE', [(1.5, 2.5), (2.0, 2.5)], False, 'p1')
    d.sheet(1).add('CUT', [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)], True, 'p2')

    back = drawing_from_json(drawing_to_json(d))
    assert back.name == d.name, "name survives"
    assert back.sheet_width == 100.0 and back.sheet_height == 60.0, \
        "sheet size survives -- getting this wrong rescales the job"
    assert len(back.sheets) == len(d.sheets) == 2, "both sheets survive"
    assert back.counts() == d.counts(), \
        f"layer counts must match: {back.counts()} vs {d.counts()}"

    a = d.sheets[0].entities[0]
    b = back.sheets[0].entities[0]
    assert b.layer == a.layer and b.closed == a.closed and b.tag == a.tag, \
        "layer, closed flag and tag all survive"
    for (x0, y0), (x1, y1) in zip(a.points, b.points):
        assert abs(x0 - x1) < 1e-12 and abs(y0 - y1) < 1e-12, \
            "coordinates survive exactly -- they are millimetres"

    # the metre/millimetre convention must be the one the docstring
    # claims, or the preview and the exported sheets disagree
    assert abs(MM * 1000.0 - 1.0) < 1e-15, "MM converts mm to metres"
    return True
