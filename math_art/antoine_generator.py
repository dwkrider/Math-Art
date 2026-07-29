
# Antoine's Necklace Generator for Blender
#
# Antoine's necklace is a wild Cantor set in space: start with a solid
# torus, replace it by a closed chain of smaller solid tori linked
# through its hole, then replace each of those by a chain of still
# smaller linked tori, and so on. The intersection over all stages is
# a Cantor set whose complement is not simply connected -- a famously
# "wild" fractal link (L. Antoine, 1921).
#
# This generator renders the rings of the deepest stage as tube tori.
# Around each parent torus's core circle it places `count` child tori
# whose axes alternate between the parent axis and the radial
# direction, so consecutive children interlock like the links of a
# chain and the whole chain threads the parent's hole. Recursing gives
# nested clusters of interlocked rings; the output is centred and fit
# to a 2 m cube.
#
# References:
# - Louis Antoine, "Sur l'homeomorphie de deux figures et de leurs
#   voisinages", Journal de Mathematiques Pures et Appliquees (8) 4,
#   1921, pp. 221-325 (Antoine's necklace).
# - R. H. Bing, "A homeomorphism between the 3-sphere and the sum of
#   two solid horned spheres", Annals of Mathematics 56, 1952 (context
#   for wild embeddings).

bl_info = {
    "name": "Antoine's Necklace",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Antoine's Necklace",
    "description": "Fractal wild link: nested chains of interlocked "
                   "tori",
    "category": "Add Mesh",
}

import math

import numpy as np


class Frame:
    """A torus placement: centre C, in-plane axes e1/e2, axis e3, and
    core radius R (the tube radius is chosen at render time)."""
    __slots__ = ('C', 'e1', 'e2', 'e3', 'R')

    def __init__(self, C, e1, e2, e3, R):
        self.C, self.e1, self.e2, self.e3, self.R = C, e1, e2, e3, R


def _children(fr, count, gap):
    """`count` child frames spaced around the parent's core circle,
    with axes alternating parent-axis / radial so neighbours link."""
    spacing = 2.0 * fr.R * math.sin(math.pi / count)
    rho = gap * spacing
    out = []
    for i in range(count):
        phi = 2.0 * math.pi * i / count
        rad = math.cos(phi) * fr.e1 + math.sin(phi) * fr.e2
        tan = -math.sin(phi) * fr.e1 + math.cos(phi) * fr.e2
        centre = fr.C + fr.R * rad
        if i % 2 == 0:
            out.append(Frame(centre, rad, tan, fr.e3, rho))
        else:
            out.append(Frame(centre, tan, fr.e3, rad, rho))
    return out


def necklace_frames(depth, count, gap=0.62):
    """Deepest-stage torus frames of the necklace."""
    frames = [Frame(np.zeros(3), np.array([1.0, 0, 0]),
                    np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 1.0)]
    for _ in range(depth):
        nxt = []
        for fr in frames:
            nxt.extend(_children(fr, count, gap))
        frames = nxt
    return frames


def _torus_mesh(fr, tube, nu, nv, vbase):
    """Vertices/faces of one tube torus in the frame `fr`."""
    verts = []
    for i in range(nu):
        u = 2.0 * math.pi * i / nu
        cu, su = math.cos(u), math.sin(u)
        for j in range(nv):
            v = 2.0 * math.pi * j / nv
            rad = fr.R + tube * math.cos(v)
            p = (fr.C + rad * (cu * fr.e1 + su * fr.e2)
                 + tube * math.sin(v) * fr.e3)
            verts.append((float(p[0]), float(p[1]), float(p[2])))
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = vbase + i * nv + j
            b = vbase + ((i + 1) % nu) * nv + j
            c = vbase + ((i + 1) % nu) * nv + (j + 1) % nv
            d = vbase + i * nv + (j + 1) % nv
            faces.append([a, b, c, d])
    return verts, faces


def build_antoine(depth=2, count=4, gap=0.62, tube_ratio=0.28,
                  ring_seg=28, tube_seg=10, scale=1.0):
    """Build the necklace mesh. `count` must be even so the alternating
    chain closes. Returns (verts, faces) centred and fit to a 2 m cube."""
    if count % 2:
        count += 1
    frames = necklace_frames(depth, count, gap)
    verts, faces = [], []
    for fr in frames:
        v, f = _torus_mesh(fr, tube_ratio * fr.R, ring_seg, tube_seg,
                           len(verts))
        verts.extend(v)
        faces.extend(f)
    V = np.asarray(verts)
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    V = (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
    return V * scale, faces, len(frames)


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_antoine_add(bpy.types.Operator):
        """Add Antoine's necklace: nested chains of interlocked tori"""
        bl_idname = "mesh.antoine_add"
        bl_label = "Antoine's Necklace"
        bl_options = {'REGISTER', 'UNDO'}

        depth: IntProperty(
            name="Depth", default=2, min=1, max=4,
            description="Recursion levels (ring count grows as "
                        "count^depth)")
        count: IntProperty(
            name="Chain Links", default=4, min=4, max=12,
            description="Tori per chain (rounded up to an even number "
                        "so the chain closes)")
        gap: FloatProperty(
            name="Link Overlap", default=0.62, min=0.45, max=0.8,
            description="Child ring radius as a fraction of the "
                        "centre spacing; higher = more overlap/linking")
        tube_ratio: FloatProperty(
            name="Tube Ratio", default=0.28, min=0.05, max=0.5,
            description="Tube radius as a fraction of ring radius")
        ring_seg: IntProperty(
            name="Ring Segments", default=28, min=6, max=96)
        tube_seg: IntProperty(
            name="Tube Segments", default=10, min=4, max=48)
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            verts, faces, n = build_antoine(
                self.depth, self.count, self.gap, self.tube_ratio,
                self.ring_seg, self.tube_seg, self.scale)
            me = bpy.data.meshes.new("Antoine")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Antoine's Necklace", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{n} rings, V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('depth', 'count', 'gap', 'tube_ratio',
                      'ring_seg', 'tube_seg', 'scale', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.antoine_add", icon='LINKED')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_antoine_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_antoine_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        from collections import Counter
        for depth in (1, 2, 3):
            verts, faces, n = build_antoine(depth=depth, count=4)
            edges = Counter()
            for f in faces:
                for k in range(len(f)):
                    a, b = f[k], f[(k + 1) % len(f)]
                    edges[(min(a, b), max(a, b))] += 1
            bnd = sum(1 for c in edges.values() if c != 2)
            finite = np.isfinite(verts).all()
            ok = n == 4 ** depth and bnd == 0 and finite
            print(f"depth={depth}: rings={n}({4 ** depth}) "
                  f"V={len(verts)} F={len(faces)} openEdges={bnd} "
                  f"{'OK' if ok else 'BAD'}")
