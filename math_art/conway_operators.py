
# Conway Polyhedron Notation for Blender
#
# Apply Conway/Hart operator strings ("dkC", "taD", "k3sT", ...) to seed
# polyhedra, in the spirit of Antiprism's `conway` program and George
# Hart's notation.
#
# Seeds:      T C O D I  (Platonics),  Pn prism, An antiprism, Yn pyramid
# Primitive:  d dual, a ambo, k kis (kN: only N-gon faces), g gyro,
#             c chamfer, r reflect, p propellor, w whirl (hexpropellor)
# Derived:    t truncate (=dkd, tN=dkNd), j join (=da), e expand (=aa),
#             o ortho (=jj), b bevel (=ta), m meta (=kj), s snub (=dg),
#             n needle (=kd), z zip (=dk)
#
# Optional geometry post-processing: spherize, or George Hart-style
# canonicalization (edges tangent to the unit sphere, planar faces).
#
# References:
# - Conway polyhedron notation: John H. Conway; extended and
#   popularized by George W. Hart ("Conway Notation for Polyhedra").
# - Canonicalization algorithm: George W. Hart, "Calculating
#   Canonical Polyhedra", Mathematica in Education and Research
#   6(3), 1997, pp. 5-10.
# - Antiprism (Adrian Rossiter), the `conway` program.

bl_info = {
    "name": "Conway Polyhedron Operators",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Conway Polyhedron",
    "description": "Conway notation polyhedra (dual, kis, ambo, gyro, ...)",
    "category": "Add Mesh",
}

import math
import re

try:
    import numpy as np
except ImportError:
    np = None


# The mathematics lives in the sibling `polyhedra` engine package: the
# notation and its seeds in `polyhedra.conway`, the eight operators in
# `polyhedra.flags` -- as flag-complex rewrites, which is why this module
# no longer carries eight hand-written constructions of its own -- and
# the canonical forms in `polyhedra.canonical`.
try:                                  # inside the math_art package
    from .polyhedra.fit import fit_cube as _fit_cube
    from .polyhedra.canonical import (biscribe, canonicalize,
                                      canonicalize_best, spherize)
    from .polyhedra.conway import (CATALOG, apply_conway, orient_outward,
                                   parse_conway)
except ImportError:                   # flat import (test runner)
    from polyhedra.fit import fit_cube as _fit_cube
    from polyhedra.canonical import (biscribe, canonicalize,
                                     canonicalize_best, spherize)
    from polyhedra.conway import (CATALOG, apply_conway, orient_outward,
                                  parse_conway)





# --------------------------------------------------------------------------
# Mesh helpers
# --------------------------------------------------------------------------















# --------------------------------------------------------------------------
# Primitive operators
# --------------------------------------------------------------------------

















# --------------------------------------------------------------------------
# Notation
# --------------------------------------------------------------------------







# --------------------------------------------------------------------------
# Geometry post-processing
# --------------------------------------------------------------------------







# --------------------------------------------------------------------------
# Named catalog
# --------------------------------------------------------------------------
#
# Curated named solids that are reachable from the operators above plus
# Hart canonicalization, mirroring several of the "computed" categories in
# David I. McCooey's Visual Polyhedra (dmccooey.com/polyhedra): the
# Archimedean-Catalan hulls (Conway join `j`), the propellor solids (`p`),
# the truncated (`t`) and rectified (`a`) Archimedean solids, the chamfered
# solids (`c`), and the dipyramids/trapezohedra (duals of the uniform
# prisms/antiprisms).  Every entry is a pure Conway construction on an exact
# seed followed by canonicalization -- no coordinate data is copied; the
# site is used only as a naming/verification reference.
#
# Each entry is (notation, category, name).



# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (StringProperty, EnumProperty, IntProperty,
                           FloatProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    EXAMPLES = [
        ('CUSTOM', "Custom", ""),
        ('tI', "Truncated Icosahedron (tI)", "the football"),
        ('sC', "Snub Cube (sC)", ""),
        ('eD', "Rhombicosidodecahedron (eD)", ""),
        ('bT', "Bevelled Tetrahedron (bT)", ""),
        ('gD', "Pentagonal Hexecontahedron (gD)", ""),
        ('cC', "Chamfered Cube (cC)", ""),
        ('pC', "Propellor Cube (pC)", "chiral"),
        ('pkD', "Propellor Pentakis (pkD)", ""),
        ('dkt5daD', "Ornate (dkt5daD)", ""),
        ('kD', "Pentakis Dodecahedron (kD)", ""),
        ('tkD', "Truncated Pentakis (tkD)", ""),
        ('ccO', "Twice-Chamfered Octahedron (ccO)", ""),
    ]

    try:
        from .styles import net_style as _net_style
    except ImportError:
        from styles import net_style as _net_style

    class MESH_OT_conway_add(bpy.types.Operator,
                             _net_style.NetStyleProps):
        """Build a polyhedron from Conway notation (e.g. dkC, taD, k3sT).
        Seeds: T C O D I, Pn, An, Yn; ops: d a k g c r t j e o b m s n z"""
        bl_idname = "mesh.conway_add"
        bl_label = "Conway Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        def _example_chosen(self, context):
            if self.example != 'CUSTOM':
                self.notation = self.example

        example: EnumProperty(name="Example", items=EXAMPLES,
                              default='tI', update=_example_chosen,
                              description="Ready-made notation string; "
                                          "sets Notation when chosen")
        notation: StringProperty(
            name="Notation", default="tI",
            description="Operators then seed, applied right to left")
        post: EnumProperty(
            name="Geometry",
            items=[('CANON', "Canonical",
                    "Edges tangent to sphere, planar faces (Hart)"),
                   ('BISCRIBED', "Biscribed",
                    "Vertices on a circumsphere AND faces tangent to a "
                    "concentric insphere. Not every solid has a biscribed "
                    "form (rectified solids and several truncations do "
                    "not) -- a warning is shown if it cannot converge"),
                   ('SPHERE', "Spherized", "Project vertices to a sphere"),
                   ('RAW', "Raw", "Whatever the operators produce")],
            default='CANON',
            description="Geometry post-processing applied after the "
                        "operator string")
        iterations: IntProperty(name="Canonical Iterations", default=200,
                                min=5, max=2000,
                                description="Number of Hart "
                                            "canonicalization passes")
        kis_height: FloatProperty(name="Kis Height", default=0.25,
                                  min=-1.0, max=2.0,
                                  description="How far the kis operator "
                                              "raises each face apex")
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "Colored (by face sides)",
                    "One material per face size, as in Hart's 'colored' "
                    "display (view with Material Preview or Solid "
                    "shading set to Material color)"),
                   ('NONE', "None", "No materials")],
            default='SIDES',
            description="How faces are assigned materials")
        uv_map: BoolProperty(
            name="Spherical UV Map", default=True,
            description="Smooth equirectangular UVs projected from the "
                        "centre (seam-corrected per face)")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed polyhedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "Geometry Nodes modifier (Border and Thickness "
                    "stay editable on the modifier)"),
                   ('WIRE', "Struts",
                    "Struts along the edges (Wireframe modifier)"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face"),
            _net_style.net_enum_item()],
            default='SOLID',
            description="How the polyhedron is built and displayed")
        border: FloatProperty(
            name="Border", default=0.06, min=0.005, max=1.0,
            description="Leonardo face frame width, the same on every "
                        "face whatever its size")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness for the Leonardo "
                        "and Wireframe styles")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
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
                             max=100.0,
                             description="Overall size of the result")

        # Hart-style palette per face size; golden-angle HSV fallback
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
                if n in cls._PALETTE:
                    rgb = cls._PALETTE[n]
                else:
                    import colorsys
                    rgb = colorsys.hsv_to_rgb((n * 0.618034) % 1.0,
                                              0.55, 0.8)
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.55
            return mat

        def execute(self, context):
            try:
                V, F = apply_conway(self.notation,
                                    kis_height=self.kis_height)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if self.post == 'SPHERE':
                V = spherize(V, F)
            elif self.post == 'BISCRIBED':
                V, conv = biscribe(V, F)
                if not conv:
                    self.report({'WARNING'},
                                "could not biscribe this solid -- it may "
                                "not have a biscribed form")
            elif self.post == 'CANON':
                V = canonicalize_best(V, F, hart_iters=self.iterations)
            V, F = orient_outward(V, [list(f) for f in F])
            if self.style == 'NET':
                return _net_style.emit_net_from_operator(
                    self, context,
                    [tuple(c * self.scale for c in v) for v in V],
                    [list(f) for f in F], f"Conway {self.notation}",
                    material_fn=self._material_for
                    if self.coloring == 'SIDES' else None)
            if self.style == 'FACETS':
                try:
                    from .styles import facet_style
                except ImportError:
                    from styles import facet_style
                Vf = [tuple(c * self.scale for c in v) for v in V]
                mat = (self._material_for
                       if self.coloring == 'SIDES' else None)
                facet_style.emit_facets(
                    context, Vf, [list(f) for f in F],
                    f"Conway {self.notation}", self.facet_depth,
                    self.facet_gap, self.facet_explode,
                    self.facet_separate, mat)
                self.report({'INFO'}, f"{len(F)} face segments")
                return {'FINISHED'}
            me = bpy.data.meshes.new(f"Conway {self.notation}")
            # operator strings change the size unpredictably (kis grows a
            # solid, ambo shrinks it), so fit rather than scale
            me.from_pydata(_fit_cube(V, 2.0 * self.scale),
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            if self.coloring == 'SIDES' and len(me.polygons) == len(F):
                sides = sorted({len(f) for f in F})
                slot = {n: i for i, n in enumerate(sides)}
                for n in sides:
                    me.materials.append(self._material_for(n))
                me.polygons.foreach_set(
                    'material_index', [slot[len(f)] for f in F])
            # face metadata: number of sides, usable in shaders (the
            # Attribute node) and Geometry Nodes
            if len(me.polygons) == len(F):
                attr = me.attributes.new("ngon_sides", 'INT', 'FACE')
                attr.data.foreach_set('value', [len(f) for f in F])
            if self.uv_map:
                uvl = me.uv_layers.new(name="UVMap")
                two_pi = 2 * math.pi
                for poly in me.polygons:
                    uvs = []
                    for li in poly.loop_indices:
                        vi = me.loops[li].vertex_index
                        x, y, z = me.vertices[vi].co
                        r = math.sqrt(x * x + y * y + z * z) or 1.0
                        u = math.atan2(y, x) / two_pi + 0.5
                        vv = math.asin(max(-1.0, min(1.0, z / r))) \
                            / math.pi + 0.5
                        uvs.append([u, vv, abs(z / r) > 0.999])
                    us = [u for u, vv, pole in uvs if not pole]
                    if us:                               # seam wrap
                        ref = max(us)
                        for q in uvs:
                            if not q[2] and ref - q[0] > 0.5:
                                q[0] += 1.0
                        us = [q[0] for q in uvs if not q[2]]
                    if us:                               # poles: average u
                        um = sum(us) / len(us)
                        for q in uvs:
                            if q[2]:
                                q[0] = um
                    for li, q in zip(poly.loop_indices, uvs):
                        uvl.data[li].uv = (q[0], q[1])
            me.update()
            obj = bpy.data.objects.new(f"Conway {self.notation}", me)
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
            elif self.style == 'BALLSTICK':
                try:
                    from .styles import ball_and_stick
                except ImportError:
                    from styles import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'},
                        f"{self.notation}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'example')
            lay.prop(self, 'notation')
            lay.prop(self, 'post')
            if self.post == 'CANON':
                lay.prop(self, 'iterations')
            lay.prop(self, 'kis_height')
            lay.prop(self, 'coloring')
            lay.prop(self, 'uv_map')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            if self.style == 'NET':
                _net_style.draw_net_props(lay, self)
            if self.style == 'FACETS':
                lay.prop(self, 'facet_depth')
                lay.prop(self, 'facet_gap')
                lay.prop(self, 'facet_explode')
                lay.prop(self, 'facet_separate')
            lay.prop(self, 'scale')


    def _menu_func(self, context):
        self.layout.operator("mesh.conway_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_conway_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_conway_add)


def _selftest():
    def euler(V, F):
        E = set()
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                E.add((min(a, b), max(a, b)))
        return len(V) - len(E) + len(F)
    all_ok = True
    for s, expect in [("C", (8, 6)), ("dC", (6, 8)), ("aC", (12, 14)),
                      ("tC", (24, 14)), ("tI", (60, 32)),
                      ("sC", (24, 38)), ("gC", (38, 24)),
                      ("cC", (32, 18)), ("eD", (60, 62)),
                      ("jC", (14, 12)), ("mC", (26, 48)),
                      ("bC", (48, 26)), ("kD", (32, 60)),
                      ("pC", (32, 30)), ("pT", (16, 16)),
                      ("pD", (80, 72)), ("dpC", (30, 32)),
                      ("pdC", (30, 32)),
                      ("P6", (12, 8)), ("A5", (10, 12)),
                      ("dA5", (12, 10)), ("Y4", (5, 5))]:
        V, F = apply_conway(s)
        ok = (len(V), len(F)) == expect and euler(V, F) == 2
        all_ok = all_ok and ok
        print(f"{s:6s}: V={len(V):3d} F={len(F):3d} chi=2:"
              f"{euler(V, F) == 2}  expect {expect} "
              f"{'OK' if ok else 'MISMATCH'}")
    if np is not None:
        V, F = apply_conway("sD")
        V2 = canonicalize(V, F, iters=40)
        print(f"sD canonicalized: V={len(V2)} F={len(F)} "
              f"chi2={euler(V2, F) == 2}")
    bad = 0
    for notation, cat, name in CATALOG:
        V, F = apply_conway(notation)
        if euler(V, F) != 2:
            print(f"CATALOG MISMATCH {notation} ({name}) chi="
                  f"{euler(V, F)}")
            bad += 1
    print(f"catalog: {len(CATALOG)} named solids, "
          f"{'all chi=2 OK' if bad == 0 else str(bad) + ' BAD'}")
    assert all_ok and bad == 0
