
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
# The frame width is ABSOLUTE, not a fraction of the face.  Scaling
# every face by one common factor -- which is what this did at first --
# makes the frame proportional to the face, so on a solid with faces of
# different sizes (a truncated icosahedron, say, or anything Conway has
# operated on) the big faces get fat frames and the small ones get thin
# ones, and the model looks unmade rather than designed.  For a frame of
# width w the scale has to vary per face as (r - w) / r, with r that
# face's own inradius.  There is no inradius field in Geometry Nodes, so
# it is recovered from the face's area and corner count: a regular
# n-gon of inradius r has area n r^2 tan(pi/n), hence
# r = sqrt(area / (n tan(pi/n))), which is exact for the regular faces
# these generators produce and close enough for the rest.
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

import math

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    # Bumped when the group's own maths changes: the builder below
    # returns any existing group of this name, so a scene saved with the
    # older proportional-inset group would otherwise keep it forever.
    NG_NAME = "Math Art Leonardo v2"

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
        s.default_value, s.min_value, s.max_value = 0.06, 0.005, 1.0
        s.description = "Width of the face frame, in object units. The " \
                        "same on every face whatever its size (the " \
                        "opening is what remains)"
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
        # per-face inradius r = sqrt(area / (n tan(pi/n)))
        area = n('GeometryNodeInputMeshFaceArea')
        corners = n('GeometryNodeCornersOfFace')
        pi_over_n = n('ShaderNodeMath')
        pi_over_n.operation = 'DIVIDE'
        pi_over_n.inputs[0].default_value = math.pi
        tan_n = n('ShaderNodeMath')
        tan_n.operation = 'TANGENT'
        k = n('ShaderNodeMath')
        k.operation = 'MULTIPLY'
        div = n('ShaderNodeMath')
        div.operation = 'DIVIDE'
        rad = n('ShaderNodeMath')
        rad.operation = 'SQRT'
        # scale = (r - w) / r, floored so an over-wide frame closes the
        # opening instead of turning the face inside out
        minus = n('ShaderNodeMath')
        minus.operation = 'SUBTRACT'
        sub = n('ShaderNodeMath')
        sub.operation = 'DIVIDE'
        clamp = n('ShaderNodeMath')
        clamp.operation = 'MAXIMUM'
        clamp.inputs[1].default_value = 0.0
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
        ln(corners.outputs['Total'], pi_over_n.inputs[1])
        ln(pi_over_n.outputs['Value'], tan_n.inputs[0])
        ln(corners.outputs['Total'], k.inputs[0])
        ln(tan_n.outputs['Value'], k.inputs[1])
        ln(area.outputs['Area'], div.inputs[0])
        ln(k.outputs['Value'], div.inputs[1])
        ln(div.outputs['Value'], rad.inputs[0])
        ln(rad.outputs['Value'], minus.inputs[0])
        ln(gi.outputs['Border'], minus.inputs[1])
        ln(minus.outputs['Value'], sub.inputs[0])
        ln(rad.outputs['Value'], sub.inputs[1])
        ln(sub.outputs['Value'], clamp.inputs[0])
        ln(ex1.outputs['Mesh'], sc.inputs['Geometry'])
        ln(ex1.outputs['Top'], sc.inputs['Selection'])
        ln(clamp.outputs['Value'], sc.inputs['Scale'])
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
        for i, node in enumerate((gi, area, corners, pi_over_n, tan_n,
                                  k, div, rad, minus, sub, clamp, ex1,
                                  sc, dg, ex2, bor, bnot, fl, go)):
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
            name="Border", default=0.06, min=0.005, max=1.0,
            description="Face frame width, the same on every face "
                        "whatever its size")
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
