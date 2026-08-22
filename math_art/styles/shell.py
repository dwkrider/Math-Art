"""
shell.py -- the shared "Style" dial for polyhedron generators.

Every generator that emits a closed shell offers the same six finishes,
and each one used to carry its own copy of the enum, the half-dozen
properties behind it and the twenty lines that apply them.  That is why
the styles drifted: a generator added later would have four of the six,
or the same style under a different property name.

This module owns all three parts:

    STYLE_ITEMS         the EnumProperty items, in one order
    style_properties()  the annotations a generator mixes into its
                        operator class
    apply(...)          takes the built (V, F) and does the rest

`apply` returns the object it made, or None when the style produced its
own objects (Face Segments emits one per face), so a caller can still
report counts.

The FACETS style consumes (V, F) rather than a finished object, so
`apply` takes the mesh data and creates the object itself -- which also
means one place decides naming, the 2 m cube fit and the smooth flag.
"""

import math

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:                       # headless import
    _IN_BLENDER = False


STYLE_ITEMS = [
    ('SOLID', "Solid", "Plain closed solid"),
    ('LEONARDO', "Leonardo (da Vinci)",
     "Open-faced panels via the shared Leonardo Style modifier, as in "
     "the illustrations Leonardo drew for Pacioli. Border and Thickness "
     "stay editable on the modifier"),
    ('WIRE', "Struts", "Struts along the edges (Wireframe modifier)"),
    ('BALLSTICK', "Ball and Stick",
     "Edges as solid cylindrical struts and vertices as small spheres"),
    ('WIREFRAME', "Wireframe",
     "Mesh edges only, displayed as a wireframe"),
    ('FACETS', "Face Segments",
     "Split the shell into one thick plate per face, padded apart and "
     "optionally exploded outward"),
]

#: styles that need the shell's faces rather than a finished object
REBUILDS_GEOMETRY = {'FACETS'}


def style_properties():
    """The properties every styled generator needs, as annotations.

    Used as:  __annotations__.update(shell.style_properties())
    so a generator declares the dial once instead of restating it.
    """
    if not _IN_BLENDER:
        return {}
    return {
        'style': EnumProperty(
            name="Style", items=STYLE_ITEMS, default='SOLID',
            description="Finish for the shell: solid, Leonardo panels, "
                        "struts, ball-and-stick, wireframe, or face "
                        "segments"),
        'border': FloatProperty(
            name="Border", default=0.06, min=0.005, max=1.0,
            description="Leonardo face frame width, the same on every "
                        "face whatever its size"),
        'thickness': FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness for the Leonardo and "
                        "Struts styles"),
        'strut_radius': FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius"),
        'node_radius': FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius (0 = no "
                        "nodes)"),
        'facet_depth': FloatProperty(
            name="Depth", default=0.15, min=0.01, max=2.0,
            description="How far each face is extruded inward (Face "
                        "Segments style)"),
        'facet_gap': FloatProperty(
            name="Bevel Gap", default=0.0, min=0.0, max=0.5,
            description="Shift the mitre bevels inward to open a gap "
                        "between neighbouring segments (0 = flush)"),
        'facet_explode': FloatProperty(
            name="Explode", default=0.0, min=0.0, max=5.0,
            description="Move each face segment outward along its own "
                        "normal"),
        'facet_separate': BoolProperty(
            name="Separate Meshes", default=False,
            description="Output each face segment as its own mesh "
                        "instead of one combined object"),
    }


def draw_style(op, layout):
    """Draw the dial and only the properties the chosen style uses."""
    layout.prop(op, 'style')
    if op.style == 'LEONARDO':
        layout.prop(op, 'border')
    if op.style in ('LEONARDO', 'WIRE'):
        layout.prop(op, 'thickness')
    if op.style == 'BALLSTICK':
        layout.prop(op, 'strut_radius')
        layout.prop(op, 'node_radius')
    if op.style == 'FACETS':
        layout.prop(op, 'facet_depth')
        layout.prop(op, 'facet_gap')
        layout.prop(op, 'facet_explode')
        layout.prop(op, 'facet_separate')


def apply(op, context, V, F, name, materials=None, material_index=None):
    """Build `name` from (V, F) in the operator's chosen style.

    `materials` / `material_index` let a generator keep its own colouring
    (the dissection's per-block palette, the transpolyhedron's three face
    families) through the SOLID, Leonardo and wireframe styles.

    Returns the created object, or None if the style made its own.
    """
    if op.style == 'FACETS':
        from . import facet_style
        facet_style.emit_facets(context, [tuple(v) for v in V],
                                [list(f) for f in F], name,
                                op.facet_depth, op.facet_gap,
                                op.facet_explode, op.facet_separate)
        return None

    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
    me.validate(clean_customdata=True)
    me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
    if materials:
        for m in materials:
            me.materials.append(m)
        if material_index is not None and len(material_index) == \
                len(me.polygons):
            me.polygons.foreach_set('material_index', material_index)
    me.update()

    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    obj.location = context.scene.cursor.location
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj

    if op.style == 'LEONARDO':
        try:
            from .. import leonardo_style
        except ImportError:
            import leonardo_style
        leonardo_style.add_modifier(obj, op.border, op.thickness)
    elif op.style == 'WIRE':
        mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
        mod.thickness = op.thickness
        mod.use_even_offset = False
    elif op.style == 'BALLSTICK':
        from . import ball_and_stick
        ball_and_stick.rebuild(obj, op.strut_radius, op.node_radius)
    elif op.style == 'WIREFRAME':
        obj.display_type = 'WIRE'
    return obj


def _selftest():
    keys = [k for k, _l, _d in STYLE_ITEMS]
    assert keys[0] == 'SOLID', keys
    assert set(keys) == {'SOLID', 'LEONARDO', 'WIRE', 'BALLSTICK',
                         'WIREFRAME', 'FACETS'}, keys
    assert REBUILDS_GEOMETRY <= set(keys)
    print("RESULT: OK")
