
# Canonical Polyhedra for Blender
#
# Polyhedra whose geometry is uniquely fixed by their combinatorics via the
# canonical (edge-tangent midsphere) form.  Two families:
#   * Greater Self-Dual Solids: 11 self-dual polyhedra of tetrahedral symmetry
#     (four icosioctahedra V=F=28, six tetracontahedra V=F=40, a 76-hedron).
#   * Simplest Canonical (per symmetry): the simplest canonical polyhedron of
#     each of the 42 three-dimensional point-group symmetry types.
# Each is derived independently: the abstract face-incidence is canonicalized
# by George Hart's edge-tangent-midsphere algorithm from a graph-Laplacian
# spectral seed, so the coordinates are our own derivation, not transcribed.
#
# References:
# - Catalog & naming: David I. McCooey, "Visual Polyhedra"
#   (dmccooey.com/polyhedra), "Greater Self-Dual Solids" category.
# - Canonical form: George W. Hart, "Calculating Canonical Polyhedra",
#   Mathematica in Education and Research 6(3), 1997 (edge-tangent
#   midsphere; also used by conway_operators.canonicalize).

bl_info = {
    "name": "Canonical Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Canonical polyhedra fixed by their combinatorics: the "
                   "Greater Self-Dual Solids and the simplest canonical "
                   "polyhedron of each symmetry type",
    "category": "Add Mesh",
}

try:
    from .polyhedra import _canonical_polyhedra_data as _data
except ImportError:
    try:
        from polyhedra import _canonical_polyhedra_data as _data
    except ImportError:
        _data = None

# (family key, label) in menu order; _BY_CAT maps key -> list of solid dicts
_FAMILIES = []
_BY_CAT = {}
if _data is not None:
    _FAMILIES.append(('SELF_DUAL', "Greater Self-Dual"))
    _BY_CAT['SELF_DUAL'] = _data.SELF_DUAL
    if hasattr(_data, 'SIMPLEST'):
        _FAMILIES.append(('SIMPLEST', "Simplest Canonical (per symmetry)"))
        _BY_CAT['SIMPLEST'] = _data.SIMPLEST

try:                                  # inside the math_art package
    from .polyhedra.fit import fit_cube as _fit_cube
except ImportError:                   # flat import (test runner)
    from polyhedra.fit import fit_cube as _fit_cube


def build(family, index):
    """Vertices/faces of the chosen solid, centred and fit to a 2 m cube."""
    s = _BY_CAT[family][index]
    V = [tuple(float(c) for c in v) for v in s["V"]]
    F = [list(f) for f in s["F"]]
    cen = [sum(v[i] for v in V) / len(V) for i in range(3)]
    V = [tuple(v[i] - cen[i] for i in range(3)) for v in V]
    mx = max((abs(c) for v in V for c in v), default=1.0) or 1.0
    return [tuple(c / mx for c in v) for v in V], F


def _self_test():
    for fk, flabel in _FAMILIES:
        for i, s in enumerate(_BY_CAT[fk]):
            V, F = build(fk, i)
            E = {}
            for f in F:
                for j in range(len(f)):
                    a, b = f[j], f[(j + 1) % len(f)]
                    E[(min(a, b), max(a, b))] = \
                        E.get((min(a, b), max(a, b)), 0) + 1
            chi = len(V) - len(E) + len(F)
            e2 = all(v == 2 for v in E.values())
            sd = (len(V) == len(F))          # self-dual necessary condition
            print(f"{s['name'][:34]:34s} V={len(V):3d} E={len(E):3d} "
                  f"F={len(F):3d} chi={chi} edge-in-2={e2} V=F={sd}")


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _family_items(self, context):
        return [(k, lbl, "") for k, lbl in _FAMILIES]

    _SOLID_CACHE = {}

    def _solid_items(self, context):
        cat = self.family
        if cat not in _SOLID_CACHE:
            _SOLID_CACHE[cat] = [(str(i), s["name"], "")
                                 for i, s in enumerate(_BY_CAT.get(cat, []))]
        return _SOLID_CACHE[cat]

    def _family_update(self, context):
        ids = [it[0] for it in _solid_items(self, context)]
        if ids and self.solid not in ids:
            self.solid = ids[0]

    try:
        from .styles import net_style as _net_style
    except ImportError:
        from styles import net_style as _net_style

    class MESH_OT_canonical_polyhedron_add(bpy.types.Operator,
                                           _net_style.NetStyleProps):
        """Add a canonical polyhedron (geometry fixed by its combinatorics):
        the Greater Self-Dual Solids, in their edge-tangent canonical form"""
        bl_idname = "mesh.canonical_polyhedron_add"
        bl_label = "Canonical Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(name="Family", items=_family_items,
                             update=_family_update,
                             description="Which family of canonical "
                                         "polyhedra to choose from")
        solid: EnumProperty(name="Solid", items=_solid_items,
                            description="Which polyhedron in the family "
                                        "to build")
        style: EnumProperty(
            name="Style",
            description="How the polyhedron is rendered",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo modifier"),
                   ('WIRE', "Struts", "Wireframe modifier"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face"),
            _net_style.net_enum_item()],
            default='SOLID')
        border: FloatProperty(name="Border", default=0.06, min=0.005, max=1.0,
                              description="Leonardo face frame width")
        thickness: FloatProperty(name="Thickness", default=0.05, min=0.001,
                                 max=1.0, description="Panel/strut thickness")
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
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size of the result")

        def execute(self, context):
            fam = self.family or (_FAMILIES[0][0] if _FAMILIES else '')
            ids = [it[0] for it in _solid_items(self, context)]
            sid = self.solid if self.solid in ids else (ids[0] if ids else '0')
            V, F = build(fam, int(sid))
            label = _BY_CAT[fam][int(sid)]["name"]
            if self.style == 'NET':
                return _net_style.emit_net_from_operator(
                    self, context,
                    [tuple(c * self.scale for c in v) for v in V],
                    [list(f) for f in F], label)
            if self.style == 'FACETS':
                try:
                    from .styles import facet_style
                except ImportError:
                    from styles import facet_style
                Vf = [tuple(c * self.scale for c in v) for v in V]
                facet_style.emit_facets(
                    context, Vf, [list(f) for f in F], label,
                    self.facet_depth, self.facet_gap,
                    self.facet_explode, self.facet_separate)
                self.report({'INFO'}, f"{label}: {len(F)} face segments")
                return {'FINISHED'}
            me = bpy.data.meshes.new(label)
            # canonicalisation fixes the midsphere, which leaves the bounding box a fraction under;
            # fit the bounding box so it matches every other generator
            me.from_pydata(_fit_cube(V, 2.0 * self.scale),
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(label, me)
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
                leonardo_style.add_modifier(obj, self.border, self.thickness)
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
            self.report({'INFO'}, f"{label}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            lay.prop(self, 'solid')
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
        self.layout.operator("mesh.canonical_polyhedron_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_canonical_polyhedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_canonical_polyhedron_add)
