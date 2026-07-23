
# Geodesic Sphere / Dome Generator for Blender
#
# Geodesic spheres and domes (after Segerman, "Visualizing Mathematics
# with 3D Printing", figs 4-5 / 4-6): Class I (b,0) and Class II (b,b)
# breakdowns of the icosahedron, octahedron or tetrahedron projected to
# the sphere, oriented vertex-up and optionally cut to a hemisphere or
# 5/8 dome.  Styles: welded shell (optional Solidify thickness), strut
# and node frame, Leonardo-style open panels, or a panelised dome with
# inset gaps between the triangles.

bl_info = {
    "name": "Geodesic Sphere / Dome Generator",
    "author": "David Krider (Math Art project, after Segerman figs 4-5, 4-6)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Geodesic Sphere / Dome",
    "description": "Geodesic spheres and domes, Class I and II",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi

import numpy as np

PHI = (1 + 5 ** 0.5) / 2


# ---- seed polyhedra (triangular faces only) ------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def seed_poly(kind):
    if kind == 'TETRA':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    elif kind == 'OCTA':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    elif kind == 'ICOSA':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _icosa_faces(V)
    else:
        raise ValueError(kind)
    return [_unit(v) for v in V], [list(f) for f in F]


def _icosa_faces(V):
    """Triangles of the icosahedron given its 12 vertices: all vertex
    triples at minimal mutual distance."""
    n = len(V)
    emin = None
    d2 = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((V[i][k] - V[j][k]) ** 2 for k in range(3))
            d2[(i, j)] = d
            emin = d if emin is None else min(emin, d)
    adj = {i: set() for i in range(n)}
    for (i, j), d in d2.items():
        if d < emin * 1.2:
            adj[i].add(j)
            adj[j].add(i)
    F = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[i] & adj[j]:
                F.add(tuple(sorted((i, j, k))))
    faces = []
    for f in F:
        a, b, c = (V[i] for i in f)
        nx = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
              (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
              (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        cen = [(a[k] + b[k] + c[k]) / 3 for k in range(3)]
        if sum(nx[k] * cen[k] for k in range(3)) < 0:
            faces.append([f[0], f[2], f[1]])
        else:
            faces.append(list(f))
    return faces


def orient_vertex_up(V):
    """Rotate the seed so its topmost vertex sits exactly on +Z."""
    apex = max(V, key=lambda v: v[2])
    v = _unit(apex)
    if v[2] > 1.0 - 1e-12:
        return list(V)
    axis = _unit((v[1], -v[0], 0.0))       # cross(v, z)
    ang = math.acos(max(-1.0, min(1.0, v[2])))
    c, s = cos(ang), sin(ang)
    out = []
    for p in V:
        d = axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2]
        cr = (axis[1] * p[2] - axis[2] * p[1],
              axis[2] * p[0] - axis[0] * p[2],
              axis[0] * p[1] - axis[1] * p[0])
        out.append(tuple(p[k] * c + cr[k] * s + axis[k] * d * (1 - c)
                         for k in range(3)))
    return out


# ---- geodesic breakdowns -------------------------------------------------

def geodesic(V, F, freq):
    """Class-I geodesic subdivision of a triangular polyhedron,
    projected to the unit sphere.  Shared vertices are deduped exactly
    (barycentric sums along a shared edge are bitwise identical from
    both sides)."""
    if freq <= 1:
        return [_unit(v) for v in V], [list(f) for f in F]
    verts = [_unit(v) for v in V]
    key = {}
    faces = []

    def vid(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in key:
            key[k] = len(verts)
            verts.append(p)
        return key[k]

    for f in F:
        if len(f) != 3:
            raise ValueError("geodesic subdivision needs triangles")
        A, B, C = (verts[i] for i in f)
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                if (i, j) == (freq, 0):
                    grid[(i, j)] = f[0]
                    continue
                if (i, j) == (0, freq):
                    grid[(i, j)] = f[1]
                    continue
                if (i, j) == (0, 0):
                    grid[(i, j)] = f[2]
                    continue
                k = freq - i - j
                p = _unit(tuple((i * A[c] + j * B[c] + k * C[c]) / freq
                                for c in range(3)))
                grid[(i, j)] = vid(p)
        for i in range(freq):
            for j in range(freq - i):
                faces.append([grid[(i, j)], grid[(i + 1, j)],
                              grid[(i, j + 1)]])
                if j < freq - i - 1:
                    faces.append([grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                  grid[(i, j + 1)]])
    return verts, faces


def kis(V, F):
    """Mid-face split: each triangle becomes three triangles about its
    centroid (projected to the sphere).  Composing this with a Class-I
    subdivision at frequency f yields the Class II (f,f) breakdown."""
    verts = [_unit(v) for v in V]
    faces = []
    for f in F:
        c = _unit(tuple(sum(verts[i][k] for i in f) / 3
                        for k in range(3)))
        ci = len(verts)
        verts.append(c)
        for i in range(3):
            faces.append([f[i], f[(i + 1) % 3], ci])
    return verts, faces


def build_sphere(base='ICOSA', freq=3, cls='I'):
    """Vertex-up unit geodesic sphere: (verts, tri faces)."""
    V, F = seed_poly(base)
    V = orient_vertex_up(V)
    if cls == 'II':
        V, F = kis(V, F)
    return geodesic(V, F, freq)


# ---- dome cutting --------------------------------------------------------

CUT_Z = {'FULL': None, 'HEMI': 0.0, 'FIVEEIGHTHS': -0.25}


def cut_faces(V, F, zmin):
    """Keep faces whose centroid z >= zmin; compact the vertex list."""
    keep = []
    for f in F:
        cz = sum(V[i][2] for i in f) / len(f)
        if cz >= zmin - 1e-9:
            keep.append(f)
    used = sorted({i for f in keep for i in f})
    remap = {o: n for n, o in enumerate(used)}
    return ([V[i] for i in used],
            [[remap[i] for i in f] for f in keep])


def boundary_loops(F):
    """Ordered rim loops (lists of vertex indices) of an open mesh;
    consecutive entries are directed boundary edges as they appear in
    their face, so faces lie to the left of the walk."""
    count = {}
    directed = {}
    for f in F:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            k = (min(a, b), max(a, b))
            count[k] = count.get(k, 0) + 1
            directed[k] = (a, b)
    nxt = {}
    for k, n in count.items():
        if n == 1:
            a, b = directed[k]
            nxt[a] = b
    loops = []
    seen = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        loops.append(loop)
    return loops


def add_base_ring(verts, faces, loops, rim_coords, width,
                  reuse=None):
    """Append a flat ring band widening each rim loop outward (in XY)
    by `width`.  `rim_coords` maps rim vertex index -> coordinate.  If
    `reuse` is given it maps rim vertex index -> index in `verts`
    (welded band); otherwise fresh copies of the rim are appended."""
    for loop in loops:
        inner = {}
        outer = {}
        for vi in loop:
            p = rim_coords[vi]
            if reuse is not None:
                inner[vi] = reuse[vi]
            else:
                inner[vi] = len(verts)
                verts.append(tuple(p))
            r = math.hypot(p[0], p[1]) or 1.0
            q = (p[0] * (1 + width / r), p[1] * (1 + width / r), p[2])
            outer[vi] = len(verts)
            verts.append(q)
        m = len(loop)
        for i in range(m):
            a, b = loop[i], loop[(i + 1) % m]
            faces.append([inner[b], inner[a], outer[a], outer[b]])


# ---- strut / node / panel geometry ---------------------------------------

def _frame(d):
    d = np.asarray(d, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    h = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, h)
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    return d, u, w


def add_strut(verts, faces, p0, p1, radius, nseg=8):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d, u, w = _frame(p1 - p0)
    base = len(verts)
    for p in (p0, p1):
        for j in range(nseg):
            t = 2 * pi * j / nseg
            q = p + radius * (cos(t) * u + sin(t) * w)
            verts.append(tuple(q))
    for j in range(nseg):
        jn = (j + 1) % nseg
        faces.append([base + j, base + jn,
                      base + nseg + jn, base + nseg + j])
    faces.append([base + j for j in range(nseg - 1, -1, -1)])
    faces.append([base + nseg + j for j in range(nseg)])


def add_node(verts, faces, center, radius, nu=8, nv=6):
    c = np.asarray(center, dtype=float)
    rows = []
    for i in range(1, nv):
        phi = pi * i / nv
        ring = []
        for j in range(nu):
            t = 2 * pi * j / nu
            q = c + radius * np.array([sin(phi) * cos(t),
                                       sin(phi) * sin(t), cos(phi)])
            ring.append(len(verts))
            verts.append(tuple(q))
        rows.append(ring)
    top = len(verts)
    verts.append(tuple(c + np.array([0.0, 0.0, radius])))
    bot = len(verts)
    verts.append(tuple(c + np.array([0.0, 0.0, -radius])))
    for j in range(nu):
        jn = (j + 1) % nu
        faces.append([top, rows[0][j], rows[0][jn]])
        faces.append([bot, rows[-1][jn], rows[-1][j]])
    for i in range(len(rows) - 1):
        for j in range(nu):
            jn = (j + 1) % nu
            faces.append([rows[i][j], rows[i + 1][j],
                          rows[i + 1][jn], rows[i][jn]])


def add_panel(verts, faces, pts, gap, thickness):
    """One triangle shrunk by `gap` about its centroid, extruded to a
    thin prism along its outward normal."""
    pts = [np.asarray(p, dtype=float) for p in pts]
    c = sum(pts) / len(pts)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    ln = np.linalg.norm(n) or 1.0
    n = n / ln
    inner = [c + (p - c) * (1.0 - gap) for p in pts]
    lo = [q - n * (thickness / 2) for q in inner]
    hi = [q + n * (thickness / 2) for q in inner]
    base = len(verts)
    for q in lo + hi:
        verts.append(tuple(q))
    m = len(pts)
    faces.append([base + m + i for i in range(m)])            # top
    faces.append([base + i for i in range(m - 1, -1, -1)])    # bottom
    for i in range(m):
        j = (i + 1) % m
        faces.append([base + i, base + j, base + m + j, base + m + i])


# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_geodesic_add(bpy.types.Operator):
        """Add a geodesic sphere or dome"""
        bl_idname = "mesh.geodesic_add"
        bl_label = "Geodesic Sphere / Dome"
        bl_options = {'REGISTER', 'UNDO'}

        base: EnumProperty(
            name="Base",
            items=[('ICOSA', "Icosahedron", ""),
                   ('OCTA', "Octahedron", ""),
                   ('TETRA', "Tetrahedron", "")],
            default='ICOSA')
        geo_class: EnumProperty(
            name="Class",
            items=[('I', "Class I (f,0)",
                    "Alternate breakdown: each base triangle is "
                    "subdivided into f² triangles and projected "
                    "to the sphere"),
                   ('II', "Class II (f,f)",
                    "Triacon-style breakdown: mid-face (kis) split "
                    "followed by a Class I subdivision at f, giving "
                    "the (f,f) triangulation (3f² triangles per "
                    "base face, effective frequency 2f)")],
            default='I')
        frequency: IntProperty(
            name="Frequency", default=3, min=1, max=16,
            description="Breakdown frequency f: Class I gives the "
                        "(f,0) subdivision, Class II the (f,f)")
        cut: EnumProperty(
            name="Cut",
            items=[('FULL', "Full Sphere", ""),
                   ('HEMI', "Hemisphere Dome",
                    "Keep faces whose centroid lies above z = 0"),
                   ('FIVEEIGHTHS', "5/8 Dome",
                    "Keep faces whose centroid lies above "
                    "z = -0.25 R (a clean strut ring on the "
                    "3v icosahedron)")],
            default='FULL')
        base_ring: BoolProperty(
            name="Base Ring", default=False,
            description="Thicken the open rim into a flat ring band "
                        "(domes are left open like real domes)")
        ring_width: FloatProperty(
            name="Ring Width", default=0.1, min=0.005, max=1.0,
            description="Radial width of the base ring band")
        style: EnumProperty(
            name="Style",
            items=[('SHELL', "Shell",
                    "Smooth welded sphere surface (Solidify modifier "
                    "if thickness > 0)"),
                   ('STRUTS', "Struts & Nodes",
                    "Edges as cylinders, vertices as small spheres"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "modifier"),
                   ('PANELS', "Panels",
                    "Each triangle inset about its centroid with a "
                    "gap and slight thickness")],
            default='SHELL')
        radius: FloatProperty(name="Radius", default=1.0, min=0.01,
                              max=100.0)
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.0, max=1.0,
            description="Shell / panel thickness (0 = single surface)")
        border: FloatProperty(name="Border", default=0.3, min=0.02,
                              max=0.95)
        strut_radius: FloatProperty(name="Strut Radius", default=0.02,
                                    min=0.001, max=0.5)
        node_radius: FloatProperty(name="Node Radius", default=0.035,
                                   min=0.001, max=0.5)
        gap: FloatProperty(
            name="Panel Gap", default=0.15, min=0.0, max=0.9,
            description="Fraction each panel is shrunk about its "
                        "centroid")

        def execute(self, context):
            V, F = build_sphere(self.base, self.frequency,
                                self.geo_class)
            zmin = CUT_Z[self.cut]
            if zmin is not None:
                V, F = cut_faces(V, F, zmin)
            R = self.radius
            V = [(x * R, y * R, z * R) for (x, y, z) in V]
            loops = (boundary_loops(F)
                     if zmin is not None and self.base_ring else [])

            # only round struts/nodes shade smooth; the flat facets
            # of shells and panels must stay flat or the edges blur
            smooth = self.style == 'STRUTS'
            if self.style in ('SHELL', 'LEONARDO'):
                verts = list(V)
                faces = [list(f) for f in F]
                add_base_ring(verts, faces, loops, V,
                              self.ring_width * R,
                              reuse={i: i for i in range(len(V))})
            elif self.style == 'STRUTS':
                verts, faces = [], []
                edges = set()
                for f in F:
                    m = len(f)
                    for i in range(m):
                        a, b = f[i], f[(i + 1) % m]
                        edges.add((min(a, b), max(a, b)))
                for a, b in sorted(edges):
                    add_strut(verts, faces, V[a], V[b],
                              self.strut_radius * R)
                for p in V:
                    add_node(verts, faces, p, self.node_radius * R)
                add_base_ring(verts, faces, loops, V,
                              self.ring_width * R)
            else:                                        # PANELS
                smooth = False
                verts, faces = [], []
                for f in F:
                    add_panel(verts, faces, [V[i] for i in f],
                              self.gap, self.thickness * R)
                add_base_ring(verts, faces, loops, V,
                              self.ring_width * R)

            name = ("Geodesic Sphere" if self.cut == 'FULL'
                    else "Geodesic Dome")
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'SHELL' and self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness * R
                mod.offset = 0.0
            elif self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            self.report({'INFO'},
                        f"V={len(me.vertices)} E={len(me.edges)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'base')
            lay.prop(self, 'geo_class')
            lay.prop(self, 'frequency')
            lay.prop(self, 'cut')
            if self.cut != 'FULL':
                lay.prop(self, 'base_ring')
                if self.base_ring:
                    lay.prop(self, 'ring_width')
            lay.prop(self, 'style')
            if self.style in ('SHELL', 'PANELS'):
                lay.prop(self, 'thickness')
            elif self.style == 'LEONARDO':
                lay.prop(self, 'border')
                lay.prop(self, 'thickness')
            elif self.style == 'STRUTS':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            if self.style == 'PANELS':
                lay.prop(self, 'gap')
            lay.prop(self, 'radius')

    def _menu_func(self, context):
        self.layout.operator("mesh.geodesic_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_geodesic_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_geodesic_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for f in range(1, 6):
            V, F = build_sphere('ICOSA', f, 'I')
            assert len(V) == 10 * f * f + 2, (f, len(V))
            assert len(F) == 20 * f * f
        for f in range(1, 5):
            V, F = build_sphere('OCTA', f, 'I')
            assert len(V) == 4 * f * f + 2, (f, len(V))
            V, F = build_sphere('ICOSA', f, 'II')
            assert len(V) == 30 * f * f + 2, (f, len(V))
        V, F = build_sphere('ICOSA', 3, 'I')
        for cut, z in (('HEMI', 0.0), ('FIVEEIGHTHS', -0.25)):
            Vc, Fc = cut_faces(V, F, z)
            loops = boundary_loops(Fc)
            zs = sorted({round(Vc[i][2], 6)
                         for lp in loops for i in lp})
            print(f"{cut}: verts={len(Vc)} faces={len(Fc)} "
                  f"rim loops={len(loops)} rim z levels={zs}")
        print("geodesic_generator self-test OK")
