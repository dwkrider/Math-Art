
# Twisted Torus Generator for Blender
#
# A regular polygon swept around a circle while rotating about the
# sweep path -- the classic twisted prismatic torus (after the Twisted
# Torus example on George W. Hart's pages). An integer number of twist
# steps (in units of 360/n degrees) keeps the seam exact, so the faces
# join into long helical bands: n and the twist step control how many
# bands spiral around the ring. Optional corner rounding morphs the
# profile from crisp polygon to circle.

bl_info = {
    "name": "Twisted Torus",
    "author": "David Krider (Math Art project, after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Twisted Torus",
    "description": "Polygon revolving around a ring while twisting",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, gcd


def build_twisted_torus(n=3, major=1.6, minor=0.55, twist_steps=1,
                        segments=192, rounding=0.0, profile_res=1,
                        scale=1.0):
    """Sweep an n-gon (optionally rounded) around a circle with a total
    twist of twist_steps * (360/n) degrees. Returns (verts, faces,
    n_bands) with an exact index-shifted seam."""
    m = n * max(1, profile_res)          # points around the profile
    # profile: regular n-gon with rounded corners (rounding 0..1)
    prof = []
    for i in range(m):
        t = i / m * n                    # in polygon-corner units
        k = int(t)
        f = t - k
        a0 = 2 * pi * k / n
        a1 = 2 * pi * (k + 1) / n
        p0 = (cos(a0), sin(a0))
        p1 = (cos(a1), sin(a1))
        edge = (p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f)
        ang = a0 + (a1 - a0) * f
        circ = (cos(ang), sin(ang))
        r = rounding
        prof.append(((edge[0] * (1 - r) + circ[0] * r) * minor,
                     (edge[1] * (1 - r) + circ[1] * r) * minor))
    verts = []
    faces = []
    for s in range(segments):
        u = 2 * pi * s / segments
        tw = 2 * pi * twist_steps / n * s / segments
        cu, su = cos(u), sin(u)
        for (px, py) in prof:
            c, s2 = cos(tw), sin(tw)
            x = px * c - py * s2
            y = px * s2 + py * c
            verts.append((((major + x) * cu) * scale,
                          ((major + x) * su) * scale,
                          y * scale))
    for s in range(segments):
        s2 = (s + 1) % segments
        shift = (m // n) * twist_steps if s == segments - 1 else 0
        for i in range(m):
            i2 = (i + 1) % m
            a = s * m + i
            b = s * m + i2
            c = s2 * m + (i2 + shift) % m
            d = s2 * m + (i + shift) % m
            faces.append([a, b, c, d])
    n_bands = gcd(n, twist_steps) if twist_steps else n
    return verts, faces, n_bands


try:
    import bpy
    from bpy.props import (FloatProperty, IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_twisted_torus_add(bpy.types.Operator):
        """Twisted prismatic torus: an n-gon revolves around a ring
        while turning; twist steps of 360/n keep the seam exact"""
        bl_idname = "mesh.twisted_torus_add"
        bl_label = "Twisted Torus"
        bl_options = {'REGISTER', 'UNDO'}

        n: IntProperty(name="Polygon Sides", default=3, min=2, max=16)
        twist_steps: IntProperty(
            name="Twist Steps", default=1, min=-8, max=8,
            description="Total twist in units of 360/n degrees; "
                        "gcd(n, steps) helical bands result")
        major: FloatProperty(name="Ring Radius", default=1.6,
                             min=0.2, max=20.0)
        minor: FloatProperty(name="Profile Radius", default=0.55,
                             min=0.02, max=5.0)
        rounding: FloatProperty(
            name="Corner Rounding", default=0.0, min=0.0, max=1.0,
            description="0 = crisp polygon, 1 = circle")
        segments: IntProperty(name="Ring Segments", default=192,
                              min=16, max=512)
        profile_res: IntProperty(
            name="Profile Subdivision", default=2, min=1, max=16,
            description="Points per polygon side")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            verts, faces, nb = build_twisted_torus(
                self.n, self.major, self.minor, self.twist_steps,
                self.segments, self.rounding, self.profile_res,
                self.scale)
            me = bpy.data.meshes.new("TwistedTorus")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("TwistedTorus", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{nb} helical band(s)")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('n', 'twist_steps', 'major', 'minor', 'rounding',
                      'segments', 'profile_res', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.twisted_torus_add", icon='MESH_TORUS')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_twisted_torus_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_twisted_torus_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for n, tw in ((3, 1), (4, 1), (5, 2), (6, 2)):
            v, f, nb = build_twisted_torus(n, twist_steps=tw)
            # Euler check for a torus: V - E + F should be 0
            E = set()
            for fc in f:
                for i in range(len(fc)):
                    a, b = fc[i], fc[(i + 1) % len(fc)]
                    E.add((min(a, b), max(a, b)))
            chi = len(v) - len(E) + len(f)
            print(f"n={n} twist={tw}: verts={len(v)} chi={chi} "
                  f"bands={nb} {'OK' if chi == 0 else 'BAD'}")
