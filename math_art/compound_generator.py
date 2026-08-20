
# Polyhedron Compounds for Blender
#
# Classic uniform polyhedron compounds built as the orbit of a seed solid
# under a rotation group: the stella octangula (two tetrahedra), the
# icosahedral compounds of five and ten tetrahedra, five cubes and five
# octahedra, and the dual pairs (cube + octahedron, dodecahedron +
# icosahedron).  The rotation groups are generated in the standard cube /
# dodecahedron frame so an axis-aligned seed lands on the compound's
# shared vertices; each component keeps its own colour.
#
# The axis-alignment rule (polyhedra/compounds.py) generalises all of
# this: align the component's n-fold axis with the compound's m-fold
# axis, turn it, replicate over the group, drop duplicates. Michael
# Harman's 1974 construction, and the Turn dial is the rotational
# freedom several of Hart's families have -- the five cubes separate
# into a generic thirty as soon as it leaves the named angle.
#
# References:
# - Stella octangula: Johannes Kepler, "Harmonices Mundi" (1619).
# - The axis-alignment construction: Michael G. Harman, unpublished
#   (1974), as described by George W. Hart, "Compounds - Harman's",
#   Virtual Polyhedra (georgehart.com/virtual-polyhedra/).
# - The regular compounds (5 tetrahedra, 5 cubes, ...): Edmund Hess
#   (1876); Max Bruckner (1900); catalogued in H. S. M. Coxeter,
#   "Regular Polytopes" (1948).
# - Full enumeration of uniform compounds: John Skilling, "Uniform
#   compounds of uniform polyhedra", Math. Proc. Camb. Phil. Soc. 79
#   (1976).

bl_info = {
    "name": "Polyhedron Compounds",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Compounds of regular polyhedra (stella octangula, "
                   "5/10 tetrahedra, 5 cubes, dual pairs)",
    "category": "Add Mesh",
}


try:
    import numpy as np
except ImportError:
    np = None
# The mathematics lives in the sibling `polyhedra` engine package;
# this module is the Blender layer over it.
try:
    from .polyhedra.compounds import (AXIS_COMPOUNDS, COMPOUNDS,
                                      build_compound)
except ImportError:  # flat import outside the package
    from polyhedra.compounds import (AXIS_COMPOUNDS, COMPOUNDS,
                                     build_compound)





# ---- seeds (V, F), axis-aligned in the standard frame --------------------









# ---- rotation groups in the standard frame -------------------------------













# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                (0.30, 0.69, 0.42), (0.95, 0.77, 0.29),
                (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                (0.80, 0.45, 0.30), (0.45, 0.55, 0.80)]

    def _mat(i):
        name = f"Compound {i}"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            rgb = _PALETTE[i % len(_PALETTE)]
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        return mat

    class MESH_OT_polyhedron_compound_add(bpy.types.Operator):
        """Add a compound of regular polyhedra (each component coloured
        separately)"""
        bl_idname = "mesh.polyhedron_compound_add"
        bl_label = "Polyhedron Compound"
        bl_options = {'REGISTER', 'UNDO'}

        compound: EnumProperty(
            name="Compound",
            items=[(k, lbl, "") for k, lbl in COMPOUNDS])
        separate: BoolProperty(
            name="Separate Objects", default=False,
            description="One object per component instead of a single "
                        "coloured mesh")
        phase: FloatProperty(
            name="Turn", default=0.0, min=-180.0, max=180.0,
            description="Turn each component about the axis it was "
                        "aligned on. The named compounds sit at one "
                        "angle; away from it the components separate "
                        "and the count usually rises, which is the "
                        "rotational freedom several of these families "
                        "have")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            try:
                axis_kind = any(self.compound == r[0] for r in AXIS_COMPOUNDS)
                comps = build_compound(
                    self.compound,
                    phase=self.phase if axis_kind and self.phase else None)
            except Exception as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # normalise the whole compound to fit a 2 m cube
            mx = max(abs(c) for V, _F in comps for v in V for c in v) or 1.0
            s = self.scale / mx
            label = dict(COMPOUNDS)[self.compound]
            for o in context.selected_objects:
                o.select_set(False)
            first = None
            if self.separate:
                for i, (V, F) in enumerate(comps):
                    me = bpy.data.meshes.new(f"{label} {i + 1}")
                    me.from_pydata([tuple(c * s for c in v) for v in V],
                                   [], [tuple(f) for f in F])
                    me.materials.append(_mat(i))
                    me.update()
                    obj = bpy.data.objects.new(f"{label} {i + 1}", me)
                    context.collection.objects.link(obj)
                    obj.location = context.scene.cursor.location
                    obj.select_set(True)
                    first = first or obj
            else:
                verts = []
                faces = []
                fmat = []
                for i, (V, F) in enumerate(comps):
                    base = len(verts)
                    verts += [tuple(c * s for c in v) for v in V]
                    for f in F:
                        faces.append([base + j for j in f])
                        fmat.append(i)
                me = bpy.data.meshes.new(label)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                for i in range(len(comps)):
                    me.materials.append(_mat(i))
                if len(me.polygons) == len(fmat):
                    me.polygons.foreach_set('material_index', fmat)
                me.update()
                first = bpy.data.objects.new(label, me)
                context.collection.objects.link(first)
                first.location = context.scene.cursor.location
                first.select_set(True)
            context.view_layer.objects.active = first
            self.report({'INFO'}, f"{label}: {len(comps)} components")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.polyhedron_compound_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polyhedron_compound_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polyhedron_compound_add)


def _selftest():
    import math as _math
    for k, lbl in COMPOUNDS:
        comps = build_compound(k)
        nv = len(set(tuple(round(c, 4) for c in v)
                     for V, _F in comps for v in V))
        print(f"{k:14s} {len(comps):2d} components, {nv:3d} distinct verts"
              f"  ({lbl})")

    # The axis-rule compounds assert their component counts: the rule
    # itself is sound, so what can go wrong is the five-tuple describing
    # a named model, and a wrong axis pair changes the count.
    for k, lbl, _c, _g, _ca, _ga, _ph, want in AXIS_COMPOUNDS:
        comps = build_compound(k)
        assert len(comps) == want, (k, len(comps), want)

        # every component must be congruent to the first -- same radius
        # spectrum and same pairwise-distance multiset, which pins the
        # vertex set to an isometry without solving for the rotation
        def spectrum(V):
            rad = sorted(round(_math.dist((0, 0, 0), v), 6) for v in V)
            pair = sorted(round(_math.dist(a, b), 6)
                          for i, a in enumerate(V) for b in V[i + 1:])
            return rad, pair

        ref = spectrum(comps[0][0])
        for V, _F in comps[1:]:
            got = spectrum(V)
            assert len(got[0]) == len(ref[0]), k
            assert max(abs(a - b) for a, b in zip(got[0], ref[0])) < 1e-6, k
            assert max(abs(a - b) for a, b in zip(got[1], ref[1])) < 1e-6, k

    # turning off the named angle is what the freedom families do: the
    # five cubes separate into a generic thirty
    assert len(build_compound('H_5CUBES', phase=17.0)) > 5
    print("RESULT: OK")
