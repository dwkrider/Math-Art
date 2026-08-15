
# Hyperbolic Honeycomb Generator for Blender
#
# The edge framework of a {p,q,r} honeycomb of hyperbolic 3-space,
# rendered inside the Poincare ball (after the figures in Segerman's
# "Visualizing Mathematics with 3D Printing", figs 4-21..4-23). Struts
# follow hyperbolic geodesics and thin toward the boundary sphere by
# the local conformal factor, so the infinite honeycomb appears to
# crowd into the ball's skin.
#
# The construction works in the hyperboloid model in Minkowski R^{3,1}
# (signature +,+,+,-): the four mirror normals of the {p,q,r} Coxeter
# orthoscheme are recovered from their Gram matrix by eigen-
# factorization, the honeycomb vertex is the dual point on mirrors
# 1,2,3, and the fundamental edge is that vertex plus its reflection
# in mirror 0. Enumerating the vertex-stabilizer subgroup (the finite
# symmetry group of the vertex figure {q,r}) gives the complete edge
# star of that vertex; a breadth-first walk vertex-to-vertex through
# the 1-skeleton (group elements as Lorentz matrices, deduped by
# rounded entries and positions) then sweeps out the framework, and
# (x,y,z,t) -> (x,y,z)/(1+t) drops it into the Poincare ball.
#
# Compact presets: {4,3,5}, {5,3,4}, {3,5,3}, {5,3,5}. Paracompact
# presets: {6,3,3} (ideal cell centers) and {3,3,6} (ideal vertices --
# every strut is a complete geodesic between boundary points).
#
# References:
# - Regular hyperbolic honeycombs via Schlafli symbols {p,q,r};
#   H. S. M. Coxeter, "Regular Honeycombs in Hyperbolic Space" (1954).
# - Visualization after Roice Nelson & Henry Segerman, "Visualizing
#   Hyperbolic Honeycombs" (2015).
# - Henry Segerman, "Visualizing Mathematics with 3D Printing"
#   (2016), figs 4-21..4-23.

bl_info = {
    "name": "Hyperbolic Honeycomb",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Honeycomb",
    "description": "{p,q,r} honeycombs of H^3 as geodesic strut "
                   "frameworks in the Poincare ball",
    "category": "Add Mesh",
}
# The mathematics lives in the sibling `hyperbolic` engine package;
# this module is the Blender layer over it.
try:
    from .hyperbolic.honeycombs import (build_honeycomb, honeycomb_edges)
except ImportError:  # flat import outside the package
    from hyperbolic.honeycombs import (build_honeycomb, honeycomb_edges)






# --------------------------------------------------------------------------
# Coxeter simplex in the hyperboloid model
# --------------------------------------------------------------------------











# --------------------------------------------------------------------------
# Group walk: edges of the honeycomb
# --------------------------------------------------------------------------















# --------------------------------------------------------------------------
# Strut / sphere mesh helpers
# --------------------------------------------------------------------------











# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, IntProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


PRESETS = {
    'H435': (4, 3, 5),
    'H534': (5, 3, 4),
    'H353': (3, 5, 3),
    'H535': (5, 3, 5),
    'H633': (6, 3, 3),
    'H336': (3, 3, 6),
}


if _IN_BLENDER:

    class MESH_OT_hyperbolic_honeycomb_add(bpy.types.Operator):
        """{p,q,r} honeycomb of hyperbolic 3-space as a geodesic
        strut framework in the Poincare ball, thinning toward the
        boundary sphere"""
        bl_idname = "mesh.hyperbolic_honeycomb_add"
        bl_label = "Hyperbolic Honeycomb"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Honeycomb",
            items=[('H435', "{4,3,5} Order-5 Cubic",
                    "Cubes, five around each edge (compact)"),
                   ('H534', "{5,3,4} Order-4 Dodecahedral",
                    "Dodecahedra, four around each edge (compact)"),
                   ('H353', "{3,5,3} Icosahedral",
                    "Icosahedra, three around each edge (compact)"),
                   ('H535', "{5,3,5} Order-5 Dodecahedral",
                    "Dodecahedra, five around each edge (compact)"),
                   ('H633', "{6,3,3} Hexagonal Tiling Cells",
                    "Paracompact: cells are {6,3} planes with ideal "
                    "centers on the boundary sphere"),
                   ('H336', "{3,3,6} Order-6 Tetrahedral",
                    "Paracompact: ideal vertices, every strut a "
                    "complete geodesic between boundary points"),
                   ('CUSTOM', "Custom {p,q,r}",
                    "Any hyperbolic Schlafli symbol (validated)")],
            default='H435')
        p: IntProperty(name="p", default=4, min=3, max=12,
                       description="Cell face size (cells are {p,q})")
        q: IntProperty(name="q", default=3, min=3, max=12,
                       description="Faces per cell vertex")
        r: IntProperty(name="r", default=5, min=3, max=12,
                       description="Cells around each edge")
        depth: IntProperty(
            name="Word Depth", default=7, min=1, max=12,
            description="Reflection word length of the group walk "
                        "(more depth, more edges, deeper into the "
                        "ball's skin)")
        max_edges: IntProperty(
            name="Edge Cap", default=15000, min=10, max=200000,
            description="Stop the group walk early past this many "
                        "edges")
        cutoff: FloatProperty(
            name="Radius Cutoff", default=0.97, min=0.5, max=0.999,
            precision=3,
            description="Cull struts whose midpoint lies outside "
                        "this ball radius")
        thickness: FloatProperty(
            name="Strut Thickness", default=0.07, min=0.005, max=0.5,
            step=1, precision=3,
            description="Strut radius before the conformal thinning "
                        "(a strut at the ball center gets half this)")
        sides: IntProperty(name="Strut Sides", default=6, min=3,
                           max=16)
        arc_segments: IntProperty(
            name="Arc Segments", default=8, min=1, max=32,
            description="Samples per geodesic strut (long "
                        "near-boundary edges need the curvature)")
        node_spheres: BoolProperty(
            name="Node Spheres", default=True,
            description="Spheres at the honeycomb vertices, scaled "
                        "by the same conformal factor (skipped for "
                        "ideal-vertex honeycombs like {3,3,6})")
        sphere_factor: FloatProperty(name="Sphere Size", default=1.6,
                                     min=1.0, max=4.0)
        ball_radius: FloatProperty(name="Ball Radius", default=1.0,
                                   min=0.01, max=100.0,
                                   description="Overall scale: radius "
                                               "of the Poincare ball")

        def execute(self, context):
            if self.preset == 'CUSTOM':
                p, q, r = self.p, self.q, self.r
            else:
                p, q, r = PRESETS[self.preset]
            try:
                verts, faces, nn, ne = build_honeycomb(
                    p, q, r, self.depth, self.max_edges, self.cutoff,
                    self.thickness, self.sides, self.arc_segments,
                    self.node_spheres, self.sphere_factor,
                    self.ball_radius)
            except ValueError as err:
                self.report({'ERROR'}, str(err))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("HyperbolicHoneycomb")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("HyperbolicHoneycomb", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{{{p},{q},{r}}}: {nn} vertices, "
                        f"{ne} edges")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                col = lay.column(align=True)
                for k in ('p', 'q', 'r'):
                    col.prop(self, k)
            lay.prop(self, 'depth')
            lay.prop(self, 'max_edges')
            lay.prop(self, 'cutoff')
            col = lay.column(align=True)
            for k in ('thickness', 'sides', 'arc_segments'):
                col.prop(self, k)
            lay.prop(self, 'node_spheres')
            if self.node_spheres:
                lay.prop(self, 'sphere_factor')
            lay.prop(self, 'ball_radius')

    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_honeycomb_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_honeycomb_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hyperbolic_honeycomb_add)


def _selftest():
    for name, (p, q, r) in PRESETS.items():
        E, ideal = honeycomb_edges(p, q, r, depth=6)
        print(f"{{{p},{q},{r}}}: {len(E):5d} edges at depth 6"
              f"{'  (ideal vertices)' if ideal else ''}")
