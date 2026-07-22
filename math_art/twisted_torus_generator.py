
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


def _profile_points(n, minor, rounding, m):
    """m points around the (optionally rounded) n-gon, plus outward
    2D normals per point."""
    pts = []
    for i in range(m):
        t = i / m * n
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
        pts.append(((edge[0] * (1 - r) + circ[0] * r) * minor,
                    (edge[1] * (1 - r) + circ[1] * r) * minor))
    nrms = []
    for i in range(m):
        a = pts[i - 1]
        b = pts[(i + 1) % m]
        tx, ty = b[0] - a[0], b[1] - a[1]
        l = math.sqrt(tx * tx + ty * ty) or 1.0
        nrms.append((ty / l, -tx / l))     # outward for CCW profile
    return pts, nrms


def build_twisted_torus_sheets(n=3, major=1.6, minor=0.55, twist_steps=1,
                               segments=192, rounding=0.0, profile_res=2,
                               shrink=0.8, sheet_thickness=0.06,
                               scale=1.0):
    """Each polygon side, shrunk about its midpoint, sweeps a solid
    ribbon; ribbons chain across the twist seam into gcd(n, steps)
    closed bands. Returns a list of (verts, faces) per band."""
    pr = max(1, profile_res)
    m = n * pr
    pts, nrms = _profile_points(n, minor, rounding, m)
    # per-side samples (pr+1 points, endpoints included), shrunk
    sides = []
    for k in range(n):
        idx = [k * pr + i for i in range(pr)] + [((k + 1) * pr) % m]
        P = [pts[i] for i in idx]
        N = [nrms[i] for i in idx]
        mid = (sum(p[0] for p in P) / len(P), sum(p[1] for p in P) / len(P))
        P = [(mid[0] + (p[0] - mid[0]) * shrink,
              mid[1] + (p[1] - mid[1]) * shrink) for p in P]
        sides.append((P, N))
    # a band is ONE side swept continuously; it closes after
    # n/gcd(n, twist) revolutions, when the accumulated twist is a
    # whole number of turns. gcd(n, twist) distinct bands result.
    t_eff = twist_steps % n
    g = gcd(n, t_eff) if t_eff else n
    per = n // g if t_eff else 1
    th = sheet_thickness / 2.0
    out = []
    for b in range(g if t_eff else n):
        P, N = sides[b]
        verts = []
        faces = []
        corners = set()    # ribbon border verts (sharp edges)
        rows = []          # each row: [out ids], [in ids]
        R = per * segments
        for ridx in range(R):
            sweep = 2 * pi * ridx / segments
            tw = 2 * pi * twist_steps / n * ridx / segments
            cu, su = cos(sweep), sin(sweep)
            c2, s2 = cos(tw), sin(tw)
            ro = []
            ri = []
            for (px, py), (nx2, ny2) in zip(P, N):
                for sign, acc in ((1, ro), (-1, ri)):
                    qx = px + nx2 * th * sign
                    qy = py + ny2 * th * sign
                    x = qx * c2 - qy * s2
                    y = qx * s2 + qy * c2
                    verts.append((((major + x) * cu) * scale,
                                  ((major + x) * su) * scale,
                                  y * scale))
                    acc.append(len(verts) - 1)
            corners.update((ro[0], ro[-1], ri[0], ri[-1]))
            rows.append((ro, ri))
        w = len(sides[0][0])
        for r in range(R):
            r2 = (r + 1) % R
            ro, ri = rows[r]
            qo, qi = rows[r2]
            for i in range(w - 1):
                faces.append([ro[i], ro[i + 1], qo[i + 1], qo[i]])
                faces.append([qi[i], qi[i + 1], ri[i + 1], ri[i]])
            faces.append([ro[0], qo[0], qi[0], ri[0]])
            faces.append([qo[w - 1], ro[w - 1], ri[w - 1], qi[w - 1]])
        out.append((verts, faces, corners))
    return out


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
    from bpy.props import (FloatProperty, IntProperty, EnumProperty)
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
        shrink: FloatProperty(
            name="Triangle Shrink", default=1.0, min=0.3, max=1.0,
            description="Shrink of each polygon side about its midpoint. "
                        "1 = sides share edges (one contiguous mesh); "
                        "less than 1 separates the swept sheets into one "
                        "mesh object per helical band")
        sheet_thickness: FloatProperty(
            name="Sheet Thickness", default=0.06, min=0.005, max=0.5,
            description="Thickness of the separated sheets (shrink < 1)")
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
        coloring: EnumProperty(
            name="Coloring",
            items=[('BAND', "Per Band",
                    "One material per helical band -- works in both the "
                    "contiguous mesh and separated sheets (view with "
                    "Material Preview or Solid shading set to Material "
                    "colour)"),
                   ('NONE', "None", "No materials")],
            default='BAND')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29)]

        @classmethod
        def _material_for(cls, i):
            name = f"Torus Band {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[i % len(cls._PALETTE)]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.45
            return mat

        @staticmethod
        def _make_mesh(name, verts, faces, sharp_verts):
            """Mesh with smooth polygons; edges whose both ends are in
            sharp_verts are marked sharp so normals do not interpolate
            across sheet joins."""
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            if sharp_verts:
                sharp = [(e.vertices[0] in sharp_verts
                          and e.vertices[1] in sharp_verts)
                         for e in me.edges]
                me.edges.foreach_set('use_edge_sharp', sharp)
            me.update()
            return me

        def execute(self, context):
            for o in context.selected_objects:
                o.select_set(False)
            cursor = context.scene.cursor.location
            if self.shrink >= 0.999:
                verts, faces, nb = build_twisted_torus(
                    self.n, self.major, self.minor, self.twist_steps,
                    self.segments, self.rounding, self.profile_res,
                    self.scale)
                # corner verts: profile index multiple of profile_res
                m = self.n * max(1, self.profile_res)
                pr = max(1, self.profile_res)
                sharp = set()
                for s in range(self.segments):
                    for k in range(self.n):
                        sharp.add(s * m + k * pr)
                me = self._make_mesh("TwistedTorus", verts, faces, sharp)
                # helical band id per face: the side index (mod the band
                # count) of the face's profile position; continuous
                # across the twist seam by construction
                t_eff = self.twist_steps % self.n
                g = gcd(self.n, t_eff) if t_eff else self.n
                face_band = [((i // pr) % g)
                             for s in range(self.segments)
                             for i in range(m)]
                if len(me.polygons) == len(face_band):
                    attr = me.attributes.new("band_index", 'INT', 'FACE')
                    attr.data.foreach_set('value', face_band)
                    if self.coloring == 'BAND':
                        for i in range(g):
                            me.materials.append(self._material_for(i))
                        me.polygons.foreach_set('material_index',
                                                face_band)
                obj = bpy.data.objects.new("TwistedTorus", me)
                context.collection.objects.link(obj)
                obj.location = cursor
                obj.select_set(True)
                context.view_layer.objects.active = obj
                self.report({'INFO'},
                            f"{nb} helical band(s), contiguous mesh")
            else:
                bands = build_twisted_torus_sheets(
                    self.n, self.major, self.minor, self.twist_steps,
                    self.segments, self.rounding,
                    max(2, self.profile_res), self.shrink,
                    self.sheet_thickness, self.scale)
                first = None
                for bi, (verts, faces, corners) in enumerate(bands):
                    me = self._make_mesh(f"TwistedTorusBand{bi + 1}",
                                         verts, faces, corners)
                    attr = me.attributes.new("band_index", 'INT', 'FACE')
                    attr.data.foreach_set('value',
                                          [bi] * len(me.polygons))
                    if self.coloring == 'BAND':
                        me.materials.append(self._material_for(bi))
                    obj = bpy.data.objects.new(
                        f"TwistedTorusBand{bi + 1}", me)
                    context.collection.objects.link(obj)
                    obj.location = cursor
                    obj.select_set(True)
                    if first is None:
                        first = obj
                context.view_layer.objects.active = first
                self.report({'INFO'},
                            f"{len(bands)} separate band mesh(es)")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('n', 'twist_steps', 'shrink', 'sheet_thickness',
                      'major', 'minor', 'rounding', 'segments',
                      'profile_res', 'coloring', 'scale'):
                if k == 'sheet_thickness' and self.shrink >= 0.999:
                    continue
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
