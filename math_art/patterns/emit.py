# Blender object builders for the pattern operators.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`), split out of
# the former single-file `pattern_common.py`.  Python + numpy only -- no
# `bpy` -- so the engine imports and self-tests headlessly; the registered
# operators stay in their flat generator modules and import this package.
#
# The ONLY module here that touches `bpy`.  It is deliberately not
# re-exported from the package facade, so `import patterns` stays
# headless.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np

from .relief import center_scale, merge_cells


# Blender object builder (shared by the pattern operators)
# --------------------------------------------------------------------

try:
    import bpy
    import bmesh
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PALETTE = PALETTE_RGBA               # one source of truth (above)

    def _material(name, color):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = color
            mat.use_nodes = True
            node = mat.node_tree.nodes.get("Principled BSDF")
            if node:
                node.inputs[0].default_value = color
        return mat

    def build_object(context, name, verts, faces, mats,
                     palette=None, span=2.0, fit=True, operator=None):
        """Build the mesh, assign per-face materials from a small
        palette, recalc normals, and add it to the scene.  With
        fit=True the result is centred and scaled to the `span` cube;
        with fit=False the vertices keep their original size but are
        still centred in XY so the object origin is the pattern centre.

        The pattern is built in local space; if `operator` is an
        AddObjectHelper it is placed via object_data_add, honouring the
        operator's Align (World / 3D Cursor / View), Location and
        Rotation -- so the pattern can land on any plane and be moved
        or reoriented afterwards.  Returns the object (or None if
        empty)."""
        if not faces:
            return None
        verts = center_scale(verts, span) if fit else center_xy(verts)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        cols = palette or PALETTE
        nmat = (max(mats) + 1) if mats else 1
        for i in range(nmat):
            me.materials.append(_material(
                "%s %d" % (name, i), cols[i % len(cols)]))
        if mats:
            me.polygons.foreach_set('material_index', mats)
        me.validate(clean_customdata=True)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        me.update()
        if operator is not None:
            from bpy_extras import object_utils
            obj = object_utils.object_data_add(context, me,
                                               operator=operator)
        else:
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return obj

    def _palette_material(i, cols=None):
        """A shared palette material by color index, reused across all
        cells and generators (so separate cells don't spawn duplicate
        material datablocks)."""
        cols = cols or PALETTE
        return _material("Math Art Pattern %d" % i, cols[i % len(cols)])

    def emit(context, name, cells, separate=False, fit=True, span=2.0,
             operator=None):
        """Turn per-cell pieces into scene geometry.  separate=False
        builds one merged object (via build_object); separate=True
        builds one child object per cell under a parent empty, so each
        cell can be edited on its own.  A single global centre/scale is
        applied so every cell keeps its relative place, and the parent
        follows the operator's Align."""
        cells = [c for c in cells if c[1]]           # drop empty cells
        if not cells:
            return None
        if not separate:
            v, f, m = merge_cells(cells)
            return build_object(context, name, v, f, m, fit=fit,
                                span=span, operator=operator)
        allv = [vv for c in cells for vv in c[0]]
        p = _global_transform(allv, fit, span)
        parent = bpy.data.objects.new(name, None)    # an empty
        context.collection.objects.link(parent)
        if operator is not None:
            from bpy_extras import object_utils
            parent.matrix_world = object_utils.add_object_align_init(
                context, operator)
        else:
            parent.location = context.scene.cursor.location
        for idx, (cv, cf, cm) in enumerate(cells):
            me = bpy.data.meshes.new("%s Cell %d" % (name, idx))
            me.from_pydata(_apply_transform(cv, p), [], cf)
            for i in range((max(cm) + 1) if cm else 1):
                me.materials.append(_palette_material(i))
            if cm:
                me.polygons.foreach_set('material_index', cm)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.update()
            obj = bpy.data.objects.new(me.name, me)
            obj["math_art_pattern"] = True
            context.collection.objects.link(obj)
            obj.parent = parent
        for o in context.selected_objects:
            o.select_set(False)
        parent.select_set(True)
        context.view_layer.objects.active = parent
        return parent

    # Not a generator: contributes no menu entry / operator.
    ADD_MENU = False

    def register():
        pass

    def unregister():
        pass
