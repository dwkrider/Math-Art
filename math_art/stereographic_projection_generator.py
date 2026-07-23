
# Stereographic Projection Shadow Shell for Blender
#
# After Henry Segerman's stereographic projection sculptures
# ("Visualizing Mathematics with 3D Printing", figs 3-11..3-14):
# a perforated spherical shell that, lit by a point light at its
# north pole, casts a shadow reproducing a flat planar pattern.
#
# The sphere (radius R) rests with its south pole at the origin:
# centre C = (0,0,R), north pole N = (0,0,2R). Stereographic
# projection from N sends the sphere point at polar angle th
# (measured from N) to the plane z = 0 at radius
#
#     r = 2R cot(th/2)        inversely   th = 2 atan(2R / r)
#
# so a plane pattern clipped to a disc of radius Rmax becomes a
# perforation pattern below the latitude th_min = 2 atan(2R/Rmax);
# everything above stays solid (the cap around the projection
# point, without which the shell falls apart).
#
# Watertight construction (boolean-free): the sphere is meshed as
# a lat-long quad grid (triangle fans at the poles). Each face is
# classified material/hole by mapping its centre to the plane
# (GRID, POLAR) or testing directly on the sphere (TILING,
# BEACHBALL). Kept faces are duplicated at radii R +- t/2 from C,
# and every boundary edge of the kept region is closed with a
# side-wall quad, so the solid is closed by construction. Two
# guards keep it 2-manifold: the pole fan rows are forced uniform
# (else >2 walls would meet at a pole's vertical edge), and a
# cleanup pass fills faces where material cells touch only
# diagonally (a "checkerboard" vertex would put 4 walls on one
# vertical edge). For GRID/POLAR the latitude rows are spaced
# uniformly in plane radius r, so pattern cells are evenly
# resolved in the plane where the pattern lives.

bl_info = {
    "name": "Stereographic Projection",
    "author": "David Krider (Math Art project, after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Stereographic Projection",
    "description": "Perforated sphere whose shadow from a north-"
                   "pole light is a flat planar pattern",
    "category": "Add Mesh",
}

import math
from math import pi, sqrt

import numpy as np

_PHI = (1.0 + sqrt(5.0)) / 2.0


def _valid_pq(p, q):
    """Clamp (p, q) to a spherical tiling: 1/p + 1/q > 1/2. p = 2
    gives the hosohedron (q meridians); q is kept >= 3 so the
    dihedron (a free-floating equator band) cannot occur."""
    p = max(2, min(p, 5))
    if p == 2:
        return p, max(3, q)
    return p, max(3, min(q, {3: 5, 4: 3, 5: 3}[p]))


def _tiling_edges(p, q):
    """Edges of the spherical {p,q} tiling as pairs of unit
    vectors, plus the common edge arc length (radians)."""
    if p == 2:                        # hosohedron: q meridians,
        n = np.array((0.0, 0.0, 1.0))  # split at the equator so no
        s = -n                         # arc joins antipodal points
        eq = [np.array((math.cos(2 * pi * k / q),
                        math.sin(2 * pi * k / q), 0.0))
              for k in range(q)]
        return ([(n, e) for e in eq] + [(e, s) for e in eq],
                pi / 2)
    if (p, q) == (3, 3):
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif (p, q) == (3, 4):
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
    elif (p, q) == (4, 3):
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
    elif (p, q) == (3, 5):
        V = [v for a in (-1.0, 1.0) for b in (-_PHI, _PHI)
             for v in ((0, a, b), (a, b, 0), (b, 0, a))]
    else:                             # (5, 3) dodecahedron
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
        V += [v for a in (-1 / _PHI, 1 / _PHI)
              for b in (-_PHI, _PHI)
              for v in ((0, a, b), (a, b, 0), (b, 0, a))]
    V = [np.asarray(v, float) / np.linalg.norm(v) for v in V]
    # edges = vertex pairs at the minimal chord distance
    dmin = min(np.linalg.norm(a - b)
               for i, a in enumerate(V) for b in V[i + 1:])
    edges = [(a, b) for i, a in enumerate(V) for b in V[i + 1:]
             if np.linalg.norm(a - b) < dmin * 1.001]
    length = math.acos(max(-1.0, min(1.0, float(
        edges[0][0] @ edges[0][1]))))
    return edges, length


def _arc_dist_min(D, edges):
    """Minimal angular distance from each unit vector in D (n,3)
    to any of the given great-circle arcs."""
    best = np.full(len(D), pi)
    for a, b in edges:
        n = np.cross(a, b)
        n /= np.linalg.norm(n)
        ta = np.cross(n, a)           # tangent at a, towards b
        tb = np.cross(b, n)           # tangent at b, towards a
        inside = (D @ ta >= 0.0) & (D @ tb >= 0.0)
        s = np.abs(np.arcsin(np.clip(D @ n, -1.0, 1.0)))
        da = np.arccos(np.clip(D @ a, -1.0, 1.0))
        db = np.arccos(np.clip(D @ b, -1.0, 1.0))
        best = np.minimum(best,
                          np.where(inside, s, np.minimum(da, db)))
    return best


def build_shell(pattern='GRID', radius=1.0, thickness=0.05,
                width=0.35, extent=3.0, spacing=0.6, rings=4,
                rays=8, p=4, q=3, gores=8, res=48):
    """Return (verts, faces) of the watertight perforated shell.
    'spacing' and 'extent' are in units of the sphere radius;
    'width' is the strip width as a fraction of the pattern cell
    (grid spacing / ring gap / edge length / lune width)."""
    R = radius
    t = min(thickness, R)
    Rmax = extent * R
    th_min = 2.0 * math.atan2(2.0 * R, Rmax)
    n_cap = max(4, res // 6)
    L = max(12, 3 * res)              # longitudes

    # ring latitudes: cap rows, then pattern rows (uniform in the
    # plane radius r for the plane-defined patterns, uniform in th
    # for the sphere-defined ones); ends at the south pole th = pi
    th_cap = np.linspace(0.0, th_min, n_cap + 1)
    if pattern in ('GRID', 'POLAR'):
        rs = np.linspace(Rmax, 0.0, res + 1)[1:]
        th_pat = 2.0 * np.arctan2(2.0 * R, rs)
    else:
        th_pat = np.linspace(th_min, pi, res + 1)[1:]
    thetas = np.concatenate([th_cap, th_pat])
    F = len(thetas) - 1               # face rows

    # face-centre coordinates and their plane projection
    thc = 0.5 * (thetas[:-1] + thetas[1:])
    TH, PH = np.meshgrid(thc, 2.0 * pi * (np.arange(L) + 0.5) / L,
                         indexing='ij')
    rp = 2.0 * R / np.tan(0.5 * TH)

    if pattern == 'GRID':
        # material = strips of width w on the lines x = k s, y = k s
        s = spacing * R
        hw = 0.5 * width * s
        x = rp * np.cos(PH)
        y = rp * np.sin(PH)
        M = ((np.abs(x - s * np.round(x / s)) < hw) |
             (np.abs(y - s * np.round(y / s)) < hw))
    elif pattern == 'POLAR':
        # concentric rings every dr (k = 0 doubles as a hub disc,
        # keeping the converging rays joined) plus radial rays
        dr = Rmax / (rings + 1)
        hw = 0.5 * width * dr
        nr = max(1, rays)
        ha = 0.5 * width * (2.0 * pi / nr)
        u = PH * nr / (2.0 * pi)
        angd = np.abs(u - np.round(u)) * (2.0 * pi / nr)
        M = ((np.abs(rp - dr * np.round(rp / dr)) < hw) |
             (angd < ha) | (rp < 2.0 * hw))
    elif pattern == 'BEACHBALL':
        # n solid lunes, each 'width' of its 2 pi / n slot; a small
        # solid cap at the south pole keeps the gore tips joined
        u = (PH * gores / (2.0 * pi)) % 1.0
        M = u < np.clip(width, 0.05, 0.95)
        M |= TH > 0.9 * pi
    else:                             # TILING
        p, q = _valid_pq(p, q)
        edges, elen = _tiling_edges(p, q)
        D = np.stack((np.sin(TH) * np.cos(PH),
                      np.sin(TH) * np.sin(PH),
                      np.cos(TH)), axis=-1).reshape(-1, 3)
        M = (_arc_dist_min(D, edges).reshape(F, L) <
             0.5 * width * elen)

    M[:n_cap] = True                  # solid cap at the north pole
    if pattern == 'TILING':           # uniform south fan row
        edist = _arc_dist_min(np.array([[0.0, 0.0, -1.0]]), edges)
        M[-1, :] = edist[0] < 0.5 * width * elen
    if M[-1].any() and not M[-1].all():
        M[-1, :] = True               # pole fans must be uniform

    # fill "checkerboard" vertices (material only on one diagonal)
    # -- they would put four side walls on one vertical edge
    for _ in range(64):
        Nr, Sr = M[:-1], M[1:]        # views: writes reach M
        NW = np.roll(Nr, 1, axis=1)
        SW = np.roll(Sr, 1, axis=1)
        bad_a = NW & Sr & ~Nr & ~SW
        bad_b = Nr & SW & ~NW & ~Sr
        if not (bad_a.any() or bad_b.any()):
            break
        Nr[bad_a] = True
        Sr[bad_b] = True

    # grid vertex ids: north pole, interior rings, south pole
    Vg = 2 + (F - 1) * L

    def vid(i, j):
        if i == 0:
            return 0
        if i == F:
            return Vg - 1
        return 1 + (i - 1) * L + (j % L)

    verts = np.empty((2 * Vg, 3))     # outer 0..Vg-1, inner rest

    def setv(idx, th, ph):
        d = np.array((math.sin(th) * math.cos(ph),
                      math.sin(th) * math.sin(ph), math.cos(th)))
        c = np.array((0.0, 0.0, R))
        verts[idx] = c + (R + 0.5 * t) * d
        verts[idx + Vg] = c + (R - 0.5 * t) * d

    setv(0, thetas[0], 0.0)
    setv(Vg - 1, thetas[F], 0.0)
    for i in range(1, F):
        for j in range(L):
            setv(vid(i, j), thetas[i], 2.0 * pi * j / L)

    # outer faces of the kept region + boundary edge census
    faces = []
    ecnt = {}                         # (lo,hi) -> directed rep or
    for i in range(F):                # None once seen twice
        for j in range(L):
            if not M[i, j]:
                continue
            f = []                    # pole rows collapse to tris
            for v in (vid(i, j), vid(i + 1, j),
                      vid(i + 1, j + 1), vid(i, j + 1)):
                if v not in f:
                    f.append(v)
            faces.append(f)
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                key = (a, b) if a < b else (b, a)
                ecnt[key] = None if key in ecnt else (a, b)
    # inner surface (reversed) + side walls on boundary edges
    faces += [[v + Vg for v in reversed(f)] for f in faces]
    faces += [[e[1], e[0], e[0] + Vg, e[1] + Vg]
              for e in ecnt.values() if e is not None]
    return [tuple(v) for v in verts], faces


try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_stereographic_add(bpy.types.Operator):
        """Add a perforated sphere whose shadow from a point light
        at its north pole is a flat planar pattern (after Henry
        Segerman's stereographic projection sculptures)"""
        bl_idname = "mesh.stereographic_add"
        bl_label = "Stereographic Projection"
        bl_options = {'REGISTER', 'UNDO'}

        pattern: EnumProperty(
            name="Pattern",
            items=[('GRID', "Square Grid",
                    "Shadow is a square grid of lines"),
                   ('POLAR', "Polar Grid",
                    "Shadow is concentric circles and radial "
                    "rays"),
                   ('TILING', "{p,q} Tiling",
                    "Material along the edges of a spherical "
                    "{p,q} tiling (Platonic edge graph; p = 2 "
                    "gives q meridians)"),
                   ('BEACHBALL', "Beach Ball",
                    "Alternating open and solid lune gores")],
            default='GRID')
        radius: FloatProperty(name="Sphere Radius", default=1.0,
                              min=0.05, max=100.0)
        thickness: FloatProperty(
            name="Shell Thickness", default=0.05, min=0.001,
            max=1.0, description="Radial wall thickness")
        width: FloatProperty(
            name="Strip Width", default=0.35, min=0.05, max=0.95,
            description="Material strip width as a fraction of "
                        "the pattern cell (grid spacing, ring "
                        "gap, tile edge, lune slot)")
        extent: FloatProperty(
            name="Pattern Extent", default=3.0, min=1.0, max=10.0,
            description="Radius of the plane pattern disc in "
                        "sphere radii; the shell is solid above "
                        "the matching latitude (the cap holding "
                        "the projection point)")
        spacing: FloatProperty(
            name="Grid Spacing", default=0.6, min=0.1, max=5.0,
            description="Grid line spacing in sphere radii")
        rings: IntProperty(name="Rings", default=4, min=1, max=24)
        rays: IntProperty(name="Rays", default=8, min=1, max=64)
        tile_p: IntProperty(
            name="p", default=4, min=2, max=5,
            description="Tile polygon sides (2 = hosohedron)")
        tile_q: IntProperty(
            name="q", default=3, min=3, max=8,
            description="Tiles per vertex (needs 1/p + 1/q > "
                        "1/2; clamped if not)")
        gores: IntProperty(name="Gores", default=8, min=2, max=64,
                           description="Number of solid lunes")
        res: IntProperty(
            name="Resolution", default=48, min=16, max=192,
            description="Latitude rows in the pattern zone "
                        "(longitudes = 3x)")
        add_light: BoolProperty(
            name="Add Point Light", default=False,
            description="Add a small point light at the north "
                        "pole to preview the shadow (Cycles or "
                        "Eevee with shadows)")

        def execute(self, context):
            pq = _valid_pq(self.tile_p, self.tile_q)
            if self.pattern == 'TILING' and pq != (self.tile_p,
                                                   self.tile_q):
                self.report({'WARNING'},
                            "{%d,%d} is not spherical; using "
                            "{%d,%d}" % (self.tile_p, self.tile_q,
                                         *pq))
            verts, faces = build_shell(
                self.pattern, self.radius, self.thickness,
                self.width, self.extent, self.spacing, self.rings,
                self.rays, *pq, self.gores, self.res)
            me = bpy.data.meshes.new("Stereographic")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            loose = [v for v in bm.verts if not v.link_faces]
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.update()
            obj = bpy.data.objects.new(
                "Stereographic %s" % self.pattern.title(), me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.add_light:
                # the ideal projection point N = (0,0,2R) lies
                # inside the solid cap, so the light sits just
                # below the cap's inner wall to reach the holes
                li = bpy.data.lights.new("Stereographic Light",
                                         'POINT')
                li.energy = 100.0
                li.shadow_soft_size = 0.005 * self.radius
                lo = bpy.data.objects.new("Stereographic Light",
                                          li)
                context.collection.objects.link(lo)
                lo.parent = obj
                lo.location = (0.0, 0.0, 2.0 * self.radius -
                               min(self.thickness, self.radius))
            self.report({'INFO'},
                        "V=%d F=%d" % (len(me.vertices),
                                       len(me.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'pattern')
            lay.prop(self, 'radius')
            lay.prop(self, 'thickness')
            lay.prop(self, 'width')
            lay.prop(self, 'extent')
            if self.pattern == 'GRID':
                lay.prop(self, 'spacing')
            elif self.pattern == 'POLAR':
                lay.prop(self, 'rings')
                lay.prop(self, 'rays')
            elif self.pattern == 'TILING':
                lay.prop(self, 'tile_p')
                lay.prop(self, 'tile_q')
            elif self.pattern == 'BEACHBALL':
                lay.prop(self, 'gores')
            lay.prop(self, 'res')
            lay.prop(self, 'add_light')

    def _menu_func(self, context):
        self.layout.operator("mesh.stereographic_add",
                             icon='LIGHT_POINT')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_stereographic_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_stereographic_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        from collections import Counter
        for pat in ('GRID', 'POLAR', 'TILING', 'BEACHBALL'):
            v, f = build_shell(pat, res=32)
            cnt = Counter()
            for fc in f:
                for i in range(len(fc)):
                    a, b = fc[i], fc[(i + 1) % len(fc)]
                    cnt[(min(a, b), max(a, b))] += 1
            man = all(c == 2 for c in cnt.values())
            print("%-9s verts=%d faces=%d watertight=%s %s"
                  % (pat, len(v), len(f), man,
                     'OK' if man else 'BAD'))
