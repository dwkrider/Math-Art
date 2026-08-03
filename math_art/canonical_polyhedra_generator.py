
# Canonical Polyhedra for Blender
#
# Polyhedra whose geometry is uniquely fixed by their combinatorics via the
# canonical (edge-tangent midsphere) form.  Currently the "Greater Self-Dual
# Solids": 11 self-dual polyhedra of tetrahedral symmetry -- four self-dual
# icosioctahedra (V=F=28), six self-dual tetracontahedra (V=F=40), and a
# self-dual 76-hedron (V=F=76).  Each is derived independently: the abstract
# face-incidence is canonicalized by George Hart's edge-tangent-midsphere
# algorithm, so the coordinates are our own derivation.  (The simplest
# canonical polyhedron of each symmetry type is a natural future addition.)
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
                   "Greater Self-Dual Solids",
    "category": "Add Mesh",
}

try:
    from . import _canonical_polyhedra_data as _data
except ImportError:
    try:
        import _canonical_polyhedra_data as _data
    except ImportError:
        _data = None

# (family key, label) in menu order; _BY_CAT maps key -> list of solid dicts
_FAMILIES = []
_BY_CAT = {}
if _data is not None:
    _FAMILIES.append(('SELF_DUAL', "Greater Self-Dual"))
    _BY_CAT['SELF_DUAL'] = _data.SELF_DUAL


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
    from bpy.props import EnumProperty, FloatProperty
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

    class MESH_OT_canonical_polyhedron_add(bpy.types.Operator):
        """Add a canonical polyhedron (geometry fixed by its combinatorics):
        the Greater Self-Dual Solids, in their edge-tangent canonical form"""
        bl_idname = "mesh.canonical_polyhedron_add"
        bl_label = "Canonical Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(name="Family", items=_family_items,
                             update=_family_update)
        solid: EnumProperty(name="Solid", items=_solid_items)
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo modifier"),
                   ('WIRE', "Wireframe", "Wireframe modifier")],
            default='SOLID')
        border: FloatProperty(name="Border", default=0.3, min=0.02, max=0.95,
                              description="Leonardo face frame width")
        thickness: FloatProperty(name="Thickness", default=0.05, min=0.001,
                                 max=1.0, description="Panel/strut thickness")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            fam = self.family or (_FAMILIES[0][0] if _FAMILIES else '')
            ids = [it[0] for it in _solid_items(self, context)]
            sid = self.solid if self.solid in ids else (ids[0] if ids else '0')
            V, F = build(fam, int(sid))
            label = _BY_CAT[fam][int(sid)]["name"]
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
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
            if self.style != 'SOLID':
                lay.prop(self, 'thickness')
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


if __name__ == "__main__" and not _IN_BLENDER:
    _self_test()
