
# Leonardo Style modifier for Blender
#
# A reusable Geometry Nodes group that turns any (closed) mesh into a
# Leonardo da Vinci style open-faced model, as in the polyhedron
# illustrations Leonardo drew for Luca Pacioli's De divina
# proportione (1509): every face becomes a solid panel with a
# polygonal opening, joined to its neighbours along the edges.
#
# The construction is pure nodes: each face is inset individually
# (extrude with zero offset + per-face scale), the inset centre is
# deleted to cut the opening, and the remaining frame surface is
# extruded along its normals into a solid shell; the interior surface
# is flipped so the result is a clean two-sided solid.
#
# The node group is shared: the operator here applies it to the
# active object, and other Math Art generators (e.g. the zonohedron
# generator's Leonardo style) attach the same group.

bl_info = {
    "name": "Leonardo Style",
    "author": "Math Art project (after Leonardo da Vinci)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Object > Leonardo Style",
    "description": "Geometry Nodes modifier: open-faced da Vinci "
                   "panels from any mesh",
    "category": "Object",
}

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    NG_NAME = "Math Art Leonardo"

    def node_group():
        """Get or build the shared Leonardo Style node group."""
        for ng in bpy.data.node_groups:
            if (ng.name.startswith(NG_NAME) and ng.type == 'GEOMETRY'
                    and any(it.name == 'Border'
                            for it in ng.interface.items_tree)):
                return ng
        ng = bpy.data.node_groups.new(NG_NAME, 'GeometryNodeTree')
        face = ng.interface.new_socket
        face("Geometry", in_out='INPUT',
             socket_type='NodeSocketGeometry')
        s = face("Border", in_out='INPUT',
                 socket_type='NodeSocketFloat')
        s.default_value, s.min_value, s.max_value = 0.3, 0.02, 0.95
        s.description = "Width of the face frame as a fraction of " \
                        "the face (the opening is what remains)"
        s = face("Thickness", in_out='INPUT',
                 socket_type='NodeSocketFloat')
        s.default_value, s.min_value, s.max_value = 0.06, 0.001, 1.0
        s.description = "Panel thickness (extruded along the face " \
                        "normals)"
        face("Geometry", in_out='OUTPUT',
             socket_type='NodeSocketGeometry')

        n = ng.nodes.new
        gi = n('NodeGroupInput')
        # 1. inset every face: individual zero extrude + per-face
        #    scale of the top copy
        ex1 = n('GeometryNodeExtrudeMesh')
        ex1.mode = 'FACES'
        ex1.inputs['Individual'].default_value = True
        ex1.inputs['Offset Scale'].default_value = 0.0
        sub = n('ShaderNodeMath')
        sub.operation = 'SUBTRACT'
        sub.inputs[0].default_value = 1.0
        sc = n('GeometryNodeScaleElements')
        sc.domain = 'FACE'
        for attr in ('scale_mode', 'mode'):    # renamed across versions
            if hasattr(sc, attr):
                setattr(sc, attr, 'UNIFORM')
                break
        # 2. cut the openings
        dg = n('GeometryNodeDeleteGeometry')
        dg.domain = 'FACE'
        # 3. thicken the frame surface into a shell
        ex2 = n('GeometryNodeExtrudeMesh')
        ex2.mode = 'FACES'
        ex2.inputs['Individual'].default_value = False
        # 4. the untouched original frame faces are now the shell's
        #    interior -- flip them so normals point out of the solid
        bor = n('FunctionNodeBooleanMath')
        bor.operation = 'OR'
        bnot = n('FunctionNodeBooleanMath')
        bnot.operation = 'NOT'
        fl = n('GeometryNodeFlipFaces')
        go = n('NodeGroupOutput')

        ln = ng.links.new
        ln(gi.outputs['Geometry'], ex1.inputs['Mesh'])
        ln(gi.outputs['Border'], sub.inputs[1])
        ln(ex1.outputs['Mesh'], sc.inputs['Geometry'])
        ln(ex1.outputs['Top'], sc.inputs['Selection'])
        ln(sub.outputs['Value'], sc.inputs['Scale'])
        ln(sc.outputs['Geometry'], dg.inputs['Geometry'])
        ln(ex1.outputs['Top'], dg.inputs['Selection'])
        ln(dg.outputs['Geometry'], ex2.inputs['Mesh'])
        ln(gi.outputs['Thickness'], ex2.inputs['Offset Scale'])
        ln(ex2.outputs['Top'], bor.inputs[0])
        ln(ex2.outputs['Side'], bor.inputs[1])
        ln(bor.outputs['Boolean'], bnot.inputs[0])
        ln(ex2.outputs['Mesh'], fl.inputs['Mesh'])
        ln(bnot.outputs['Boolean'], fl.inputs['Selection'])
        ln(fl.outputs['Mesh'], go.inputs['Geometry'])
        for i, node in enumerate((gi, ex1, sub, sc, dg, ex2,
                                  bor, bnot, fl, go)):
            node.location = (190 * (i % 5), -230 * (i // 5))
        return ng

    def add_modifier(obj, border=None, thickness=None):
        """Attach the Leonardo modifier to obj; returns the modifier."""
        mod = obj.modifiers.new("Leonardo Style", 'NODES')
        ng = node_group()
        mod.node_group = ng
        for item in ng.interface.items_tree:
            if item.in_out != 'INPUT':
                continue
            if item.name == 'Border' and border is not None:
                mod[item.identifier] = border
            elif item.name == 'Thickness' and thickness is not None:
                mod[item.identifier] = thickness
        return mod

    class OBJECT_OT_leonardo_add(bpy.types.Operator):
        """Turn the active mesh into a Leonardo da Vinci style
        open-faced model (adds a Geometry Nodes modifier; Border and
        Thickness are editable on the modifier afterwards)"""
        bl_idname = "object.leonardo_add"
        bl_label = "Leonardo Style"
        bl_options = {'REGISTER', 'UNDO'}

        border: bpy.props.FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Face frame width (fraction of the face)")
        thickness: bpy.props.FloatProperty(
            name="Thickness", default=0.06, min=0.001, max=1.0,
            description="Panel thickness")

        @classmethod
        def poll(cls, context):
            return (context.active_object is not None
                    and context.active_object.type == 'MESH')

        def execute(self, context):
            obj = context.active_object
            add_modifier(obj, self.border, self.thickness)
            self.report({'INFO'},
                        f"Leonardo modifier added to {obj.name}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("object.leonardo_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(OBJECT_OT_leonardo_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.remove(_menu_func)
        bpy.utils.unregister_class(OBJECT_OT_leonardo_add)


if __name__ == "__main__" and _IN_BLENDER:
    register()
