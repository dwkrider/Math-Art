
# Geodesic Sphere / Dome Generator for Blender
#
# Geodesic spheres and domes (after Segerman, "Visualizing Mathematics
# with 3D Printing", figs 4-5 / 4-6): Class I (h,0), Class II (h,h) and
# the general chiral Class III (h,k) Goldberg-Coxeter breakdowns of the
# icosahedron, octahedron or tetrahedron projected to the sphere,
# oriented vertex-up and optionally cut to a hemisphere or 5/8 dome.
# The Class III cells are placed as a triangular lattice on each base
# face and the projected lattice points convex-hulled.  Styles: welded
# shell (optional Solidify thickness), strut and node frame, Leonardo-
# style open panels, or a panelised dome with inset gaps.
#
# References:
# - Geodesic domes: R. Buckminster Fuller.
# - Geodesic/Goldberg (h,k) classification: M. Goldberg, "A class of
#   multi-symmetric polyhedra", Tohoku Math. J. 43 (1937); Caspar & Klug
#   (1962) for the triangulation number T = h²+hk+k².
# - Goldberg polyhedra (the duals): Michael Goldberg (1937).
# - Henry Segerman, "Visualizing Mathematics with 3D Printing"
#   (2016), figs 4-5, 4-6.

bl_info = {
    "name": "Geodesic Sphere / Dome Generator",
    "author": "Math Art project (after Segerman figs 4-5, 4-6)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Geodesic Sphere / Dome",
    "description": "Geodesic spheres and domes, Class I, II and III "
                   "(h,k), with Goldberg duals",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi

import numpy as np

PHI = (1 + 5 ** 0.5) / 2

try:                                  # inside the math_art package
    from .polyhedra.seeds import icosa_faces as _icosa_faces
    from .polyhedra.seeds import seed_poly as _shared_seed
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly as _shared_seed


def seed_poly(kind):
    """Platonic seed normalised to unit circumradius.

    This module and two others built their seeds this way, while
    three others used the raw coordinates.  The two sets are
    EXACTLY related by that scale (verified vertex for vertex on
    all five solids, with identical face lists), so the shared
    table carries both behind its `unit` flag and this wrapper
    keeps the call sites here reading as before.
    """
    return _shared_seed(kind, unit=True)


# ---- seed polyhedra (triangular faces only) ------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)




def quad_seed(kind):
    """Quad-faced seeds for geodesic cubes / rhombic triacontahedra."""
    if kind == 'CUBE':
        V = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        Fq = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
              [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    elif kind == 'RT':                   # rhombic triacontahedron = dual iD
        try:
            from . import conway_operators as cw
        except ImportError:
            import conway_operators as cw
        V, F = cw.apply_conway('aD')
        V = cw.canonicalize(V, F, iters=400)
        Vd, Fd = cw.op_dual(V, F)
        Vd = cw.canonicalize(Vd, Fd, iters=600)
        return [_unit(v) for v in Vd], [list(f) for f in Fd]
    else:
        raise ValueError(kind)
    return [_unit(v) for v in V], Fq


def quad_geodesic(V, Fq, freq):
    """Class-I (freq,0) quad subdivision of a quad-faced polyhedron,
    projected to the unit sphere (geodesic cube / rhombic triacontahedron).
    Shared edges dedupe exactly by rounded position."""
    if freq <= 1:
        return [_unit(v) for v in V], [list(f) for f in Fq]
    verts = {}
    faces = []

    def vid(p):
        k = (round(p[0], 8), round(p[1], 8), round(p[2], 8))
        if k not in verts:
            verts[k] = len(verts)
        return verts[k]
    for f in Fq:
        A, B, C, D = (V[i] for i in f)

        def bil(s, t):
            return _unit(tuple((1 - s) * (1 - t) * A[c] + s * (1 - t) * B[c]
                               + s * t * C[c] + (1 - s) * t * D[c]
                               for c in range(3)))
        g = {}
        for i in range(freq + 1):
            for j in range(freq + 1):
                g[(i, j)] = vid(bil(i / freq, j / freq))
        for i in range(freq):
            for j in range(freq):
                faces.append([g[(i, j)], g[(i + 1, j)],
                              g[(i + 1, j + 1)], g[(i, j + 1)]])
    inv = {v: k for k, v in verts.items()}
    return [inv[i] for i in range(len(verts))], faces




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


def geodesic_points(base, h, k):
    """Vertices of the general geodesic [h,k] breakdown (Goldberg-Coxeter),
    projected to the unit sphere.  h=k=0 excluded; k=0 is Class I, h=k is
    Class II, otherwise the chiral Class III.  Each base triangle carries a
    triangular lattice whose (h,k) cell defines the subdivision; the convex
    hull of the returned points is the geodesic sphere (V = 10T+2,
    T = h²+hk+k², for the icosahedron)."""
    V, F = seed_poly(base)
    V = orient_vertex_up(V)
    s3 = math.sqrt(3) / 2

    def latt(m, n):
        return (m + n * 0.5, n * s3)
    O, P, Q = latt(0, 0), latt(h, k), latt(-k, h + k)

    def bary(p):
        det = ((P[1] - Q[1]) * (O[0] - Q[0]) + (Q[0] - P[0]) * (O[1] - Q[1]))
        a = ((P[1] - Q[1]) * (p[0] - Q[0]) + (Q[0] - P[0]) * (p[1] - Q[1])) / det
        b = ((Q[1] - O[1]) * (p[0] - Q[0]) + (O[0] - Q[0]) * (p[1] - Q[1])) / det
        return a, b, 1 - a - b
    lo1 = min(0, h, -k) - 1
    hi1 = max(0, h, -k) + 1
    lo2 = min(0, k, h + k) - 1
    hi2 = max(0, k, h + k) + 1
    pts = {}
    for f in F:
        A, B, C = V[f[0]], V[f[1]], V[f[2]]
        for m in range(lo1, hi1 + 1):
            for n in range(lo2, hi2 + 1):
                a, b, c = bary(latt(m, n))
                if a < -1e-9 or b < -1e-9 or c < -1e-9:
                    continue
                sp = _unit(tuple(a * A[t] + b * B[t] + c * C[t]
                                 for t in range(3)))
                pts[tuple(round(x, 7) for x in sp)] = sp
    return list(pts.values())


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def goldberg_dual(V, F):
    """Dual of a triangulated geodesic sphere: a vertex at each face
    centroid (on the sphere) and a polygon around each original vertex
    -- hexagons plus twelve pentagons, i.e. the Goldberg polyhedron."""
    Vd = [_unit(tuple(sum(V[i][k] for i in f) / len(f)
                      for k in range(3))) for f in F]
    inc = [[] for _ in range(len(V))]
    for fi, f in enumerate(F):
        for vi in f:
            inc[vi].append(fi)
    Fd = []
    for vi in range(len(V)):
        ring = inc[vi]
        if len(ring) < 3:            # boundary vertex: no closed face
            continue
        n = _unit(V[vi])
        ref = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
        u = _unit(_cross(n, ref))
        w = _cross(n, u)
        ring = sorted(ring, key=lambda fi: math.atan2(_dot(Vd[fi], w),
                                                      _dot(Vd[fi], u)))
        Fd.append(ring)
    return Vd, Fd


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


# ---- panel geometry ------------------------------------------------------
# The strut (edge cylinder) and node (vertex sphere) primitives that back
# the "Ball and Stick" style now live in the shared ball_and_stick module,
# so every polyhedron generator draws struts and nodes the same way.


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
                   ('TETRA', "Tetrahedron", ""),
                   ('CUBE', "Cube (quad geodesic)",
                    "Square-grid subdivision -> quad-faced geodesic"),
                   ('RT', "Rhombic Triacontahedron (quad geodesic)",
                    "Rhombus-grid subdivision -> quad-faced geodesic")],
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
                    "base face, effective frequency 2f)"),
                   ('III', "Class III (h,k)",
                    "General chiral Goldberg-Coxeter breakdown with "
                    "h = Frequency and the k below (h != k, k > 0); "
                    "convex-hulled")],
            default='I')
        frequency: IntProperty(
            name="Frequency", default=3, min=1, max=16,
            description="Breakdown frequency f (= h for Class III): "
                        "Class I gives (f,0), Class II gives (f,f)")
        k: IntProperty(
            name="k (Class III)", default=1, min=1, max=15,
            description="Second Goldberg-Coxeter index for Class III "
                        "(the chiral (h,k) breakdown, h = Frequency)")
        dual: BoolProperty(
            name="Dual (Goldberg)", default=False,
            description="Output the dual polyhedron -- hexagons and "
                        "twelve pentagons (a Goldberg polyhedron / "
                        "geodesic's honeycomb) instead of triangles")
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
                   ('WIRE', "Struts",
                    "Edges as a wireframe frame of struts (Wireframe "
                    "modifier)"),
                   ('STRUTS', "Ball and Stick",
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
            if self.base in ('CUBE', 'RT'):
                V, F = quad_geodesic(*quad_seed(self.base), self.frequency)
            elif self.geo_class == 'III':
                import bmesh
                pts = geodesic_points(self.base, self.frequency, self.k)
                bm = bmesh.new()
                for p in pts:
                    bm.verts.new(p)
                bm.verts.ensure_lookup_table()
                res = bmesh.ops.convex_hull(bm, input=bm.verts)
                junk = res.get('geom_unused', []) + \
                    res.get('geom_interior', [])
                if junk:
                    bmesh.ops.delete(bm, geom=junk, context='VERTS')
                bm.verts.ensure_lookup_table()
                bm.faces.ensure_lookup_table()
                idx = {v: i for i, v in enumerate(bm.verts)}
                V = [tuple(v.co) for v in bm.verts]
                F = [[idx[v] for v in fc.verts] for fc in bm.faces]
                bm.free()
            else:
                V, F = build_sphere(self.base, self.frequency,
                                    self.geo_class)
            if self.dual:
                V, F = goldberg_dual(V, F)
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
            if self.style in ('SHELL', 'LEONARDO', 'WIRE'):
                verts = list(V)
                faces = [list(f) for f in F]
                add_base_ring(verts, faces, loops, V,
                              self.ring_width * R,
                              reuse={i: i for i in range(len(V))})
            elif self.style == 'STRUTS':
                try:
                    from . import ball_and_stick
                except ImportError:
                    import ball_and_stick
                edges = ball_and_stick.edges_from_faces(F)
                verts, faces = ball_and_stick.build_mesh(
                    V, edges, self.strut_radius * R,
                    self.node_radius * R)
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

            base_name = "Goldberg" if self.dual else "Geodesic"
            name = ("%s Sphere" % base_name if self.cut == 'FULL'
                    else "%s Dome" % base_name)
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
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness * R
                mod.use_even_offset = False
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
            if self.geo_class == 'III':
                lay.prop(self, 'k')
            lay.prop(self, 'dual')
            lay.prop(self, 'cut')
            if self.cut != 'FULL':
                lay.prop(self, 'base_ring')
                if self.base_ring:
                    lay.prop(self, 'ring_width')
            lay.prop(self, 'style')
            if self.style in ('SHELL', 'PANELS', 'WIRE'):
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


def _selftest():
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
