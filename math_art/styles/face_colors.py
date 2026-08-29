"""
face_colors.py -- one palette and one "colour the faces by this key"
helper, shared by the generators that group faces into families.

A generator rarely wants a colour ramp; it wants to SHOW something -- which
pair of zones spans a face, which level of a polar zonohedron it sits on,
how many sides it has, which zone triple built a dissection block.  All of
those are the same operation: hand in one hashable key per face, get back
a material list and a per-face material index.

Keeping it here rather than in each generator matters because the point of
the colouring is comparison across generators: the zone-pair colouring of a
zonohedron and of a zonish polyhedron should look like the same scheme,
which they will not if each module carries its own eight colours in its own
order.
"""

try:
    import bpy
    _IN_BLENDER = True
except ImportError:                       # headless import
    _IN_BLENDER = False


#: Eight well-separated hues.  Kowalewski's colouring of the rhombic
#: triacontahedron needs only five -- each of the twenty blocks takes three
#: of them, and the ten 3-subsets appear once acute and once flat -- so the
#: first five are kept in that role and three more follow for the larger
#: stars, where a zone pair or a level count runs past five families.
PALETTE = [(0.85, 0.20, 0.18, 1.0), (0.16, 0.42, 0.78, 1.0),
           (0.95, 0.72, 0.15, 1.0), (0.20, 0.62, 0.32, 1.0),
           (0.55, 0.28, 0.68, 1.0), (0.90, 0.45, 0.15, 1.0),
           (0.30, 0.72, 0.72, 1.0), (0.75, 0.30, 0.50, 1.0)]


def material(name, rgba):
    """Fetch or create a material of this name, set to this colour."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat.diffuse_color = rgba
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
    return mat


def materials_for(keys, prefix="Face"):
    """(materials, material_index) for one hashable key per face.

    Families are numbered in first-appearance order, so a colouring stays
    stable as long as the face order does -- change the face order and the
    colours move, which is why the callers build their key lists in the
    same loop that builds the faces.
    """
    order, mats = {}, []
    for k in keys:
        if k not in order:
            order[k] = len(order)
            mats.append(material("%s %d" % (prefix, len(order) - 1),
                                 PALETTE[(len(order) - 1) % len(PALETTE)]))
    return mats, [order[k] for k in keys]


def _selftest():
    # the mapping is pure bookkeeping, so it is testable without Blender
    keys = ['a', 'b', 'a', 'c', 'b']
    order = {}
    idx = []
    for k in keys:
        if k not in order:
            order[k] = len(order)
        idx.append(order[k])
    assert idx == [0, 1, 0, 2, 1], idx
    assert len(PALETTE) == 8
    assert all(len(c) == 4 for c in PALETTE)
    assert len({c for c in PALETTE}) == 8      # no accidental duplicates
    print("RESULT: OK")
