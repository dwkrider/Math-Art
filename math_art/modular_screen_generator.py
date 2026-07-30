
# Modular Screen generator for Blender -- perforated architectural
# screen walls in the modular-constructivist tradition of Erwin Hauer
# and Norman Carlberg.
#
# One curvilinear saddle module is tiled seamlessly across a rectangle.
# Because neighbouring cells share their boundary curves the tiled
# midsurface is a single continuous undulating sheet -- the "undulating
# webbing" of a Hauer screen wall -- which is then thickened into a
# solid slab and perforated with a smooth aperture through each module.
# The saddle surface is the essential ingredient: its up-in-one-way,
# down-in-the-other curvature lets the module's boundary match its
# neighbour's on all four sides, so a single unit propagates without
# closing the form.  Hauer's fully three-dimensional modules were later
# identified by Alan Schoen as pieces of triply-periodic minimal
# surfaces (the I-WP surface); the lightweight height-field route taken
# here captures the doubly-periodic, single-slab members of that family
# (Design 5's egg-crate lattice, the Design 6 relief wall) and Norman
# Carlberg's harder-edged ruled (hyperbolic-paraboloid) modules.
#
# GEOMETRY.  The midsurface is a periodic height field h(x, y) of unit
# period.  Each unit cell is meshed as a radial fan from its square
# perimeter inward to a central aperture circle (or, for a solid cell,
# to the cell centre), so the perforations are traced as smooth circles
# at any resolution instead of being cut as stair-stepped quads.  Two
# offset copies of that mesh (at h +/- t/2) form the slab's faces, and a
# wall -- optionally rounded into a bull-nose rim -- is raised on every
# boundary edge (the panel perimeter and every aperture), closing the
# shell into a single watertight, orientable manifold.  Neighbouring
# cells share their perimeter samples exactly, so the tiling welds with
# no seam.
#
# References:
#   - Erwin Hauer, "Continua -- Architectural Screens and Walls",
#     Princeton Architectural Press, 2004 -- the screen designs
#     (numbered Design 1-7, 1950-57) and Hauer's saddle-surface method.
#   - Norman Carlberg and Erwin Hauer -- co-originators of Modular
#     Constructivism (units designed to be multiplied); Carlberg's
#     ruled/hard-edged saddle modules (e.g. "Minimal Surface Form 6").
#   - H. F. Scherk, "Bemerkungen ueber die kleinste Flaeche innerhalb
#     gegebener Grenzen", J. reine angew. Math. 13 (1835), 185-208 --
#     the doubly-periodic minimal surface behind the saddle lattice.
#   - A. H. Schoen, "Infinite Periodic Minimal Surfaces Without Self-
#     Intersections", NASA TN D-5541, 1970 -- the I-WP surface later
#     identified with Hauer's fully three-dimensional modules.

bl_info = {
    "name": "Modular Screen",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Hauer/Carlberg continuous perforated screen walls: "
                   "tiled saddle modules thickened and perforated with "
                   "smooth apertures",
    "category": "Add Mesh",
}

import math
from math import pi, cos, sin, hypot

import numpy as np

try:
    from . import pattern_common as pc
except Exception:                            # legacy single-file / CLI
    import pattern_common as pc


# --------------------------------------------------------------------
# Pure-math core (no bpy)
# --------------------------------------------------------------------

def height_at(style, x, y, ci, cj, amp):
    """Midsurface height of a module at global (x, y).  ci, cj are the
    integer indices of the cell (x, y) belongs to (only the HYPAR style
    is piecewise and needs them).  Every style is C0-continuous across
    cell boundaries, so a point on a shared edge evaluates to the same
    height from either adjacent cell."""
    if style == 'HYPAR':
        # Bilinear ruled saddle per cell (a hyperbolic paraboloid,
        # z = amp * xi * eta on xi, eta in [-1, 1]) with a checkerboard
        # sign so adjacent cells match along their shared edge with a
        # sharp crease -- Carlberg's hard-edged modules.
        xi = 2.0 * (x - ci) - 1.0
        eta = 2.0 * (y - cj) - 1.0
        sign = 1.0 if (ci + cj) % 2 == 0 else -1.0
        return amp * sign * xi * eta
    if style == 'WEAVE':
        return amp * 0.5 * (cos(pi * x) + cos(pi * y))
    if style == 'DIAGONAL':
        return amp * cos(pi * (x + y)) * cos(pi * (x - y))
    return amp * cos(pi * x) * cos(pi * y)           # SADDLE / RELIEF


def _perimeter(ci, cj, K):
    """The 4*K samples around a unit cell's square boundary, in order
    (each of the four edges split into K segments, endpoints not
    duplicated) -- so adjacent cells share the K+1 points on a common
    edge exactly."""
    pts = []
    for s in range(K):
        pts.append((ci + s / K, float(cj)))         # bottom
    for s in range(K):
        pts.append((float(ci + 1), cj + s / K))      # right
    for s in range(K):
        pts.append((ci + 1 - s / K, float(cj + 1)))  # top
    for s in range(K):
        pts.append((float(ci), cj + 1 - s / K))      # left
    return pts


def build_screen(nx=5, ny=5, style='SADDLE', amp=0.5, thick=0.14,
                 hole=0.34, res=6, frame=True, rim_bulge=0.6,
                 bulge_segs=3):
    """Build (verts, faces, mats) for a thickened, perforated saddle
    screen over an nx x ny block of modules.  mats: 0 = top face,
    1 = bottom face, 2 = rim wall.  The result is a single closed,
    watertight, orientable manifold."""
    nx = max(2, int(nx))
    ny = max(2, int(ny))
    K = max(3, int(res))
    N = max(2, int(res))                              # radial rings
    M = 4 * K
    r = max(0.0, min(0.47, float(hole)))
    segs = max(2, int(bulge_segs)) if rim_bulge > 1e-9 else 1
    half = 0.5 * float(thick)

    reg = {}
    verts = []

    def vid(x, y, h, layer):
        key = (round(x, 6), round(y, 6), layer)
        i = reg.get(key)
        if i is None:
            i = len(verts)
            reg[key] = i
            verts.append((x, y, h + (half if layer == 0 else -half)))
        return i

    top_faces = []
    bot_faces = []

    for ci in range(nx):
        for cj in range(ny):
            border = ci == 0 or cj == 0 or ci == nx - 1 or cj == ny - 1
            has_ap = r > 1e-6 and not (frame and border)
            per = _perimeter(ci, cj, K)
            cx, cy = ci + 0.5, cj + 0.5
            inner = []
            for (px, py) in per:
                dx, dy = px - cx, py - cy
                d = hypot(dx, dy) or 1.0
                if has_ap:
                    inner.append((cx + r * dx / d, cy + r * dy / d))
                else:
                    inner.append((cx, cy))

            top = [[0] * M for _ in range(N + 1)]
            bot = [[0] * M for _ in range(N + 1)]
            for m in range(N + 1):
                t = m / N
                for k in range(M):
                    ox, oy = per[k]
                    ix, iy = inner[k]
                    x = ox + t * (ix - ox)
                    y = oy + t * (iy - oy)
                    h = height_at(style, x, y, ci, cj, amp)
                    top[m][k] = vid(x, y, h, 0)
                    bot[m][k] = vid(x, y, h, 1)

            for m in range(N):
                for k in range(M):
                    k2 = (k + 1) % M
                    a, b = top[m][k], top[m][k2]
                    c, d = top[m + 1][k2], top[m + 1][k]
                    if c == d:                        # collapsed centre
                        top_faces.append((a, b, c))
                        ab, bb, cb = bot[m][k], bot[m][k2], bot[m + 1][k2]
                        bot_faces.append((ab, cb, bb))
                    else:
                        top_faces.append((a, b, c, d))
                        ab, bb = bot[m][k], bot[m][k2]
                        cb, db = bot[m + 1][k2], bot[m + 1][k]
                        bot_faces.append((ab, db, cb, bb))

    # boundary edges of the top sheet: perimeter + aperture rims
    ecount = {}
    owner = {}
    for fi, f in enumerate(top_faces):
        L = len(f)
        for k in range(L):
            u, v = f[k], f[(k + 1) % L]
            key = (u, v) if u < v else (v, u)
            ecount[key] = ecount.get(key, 0) + 1
            owner.setdefault(key, fi)
    boundary = [key for key, c in ecount.items() if c == 1]

    def bottom_of(top_idx):
        x, y, _ = verts[top_idx]
        return reg[(round(x, 6), round(y, 6), 1)]

    # outward horizontal normal at every boundary vertex (mean of the
    # incident boundary-edge normals, each pointing away from its face)
    vn = {}
    for (u, v) in boundary:
        f = top_faces[owner[(u, v)]]
        fx = sum(verts[i][0] for i in f) / len(f)
        fy = sum(verts[i][1] for i in f) / len(f)
        ex = verts[v][0] - verts[u][0]
        ey = verts[v][1] - verts[u][1]
        n = (ey, -ex)
        mx = 0.5 * (verts[u][0] + verts[v][0])
        my = 0.5 * (verts[u][1] + verts[v][1])
        if n[0] * (mx - fx) + n[1] * (my - fy) < 0.0:
            n = (-ey, ex)
        for w in (u, v):
            acc = vn.setdefault(w, [0.0, 0.0])
            acc[0] += n[0]
            acc[1] += n[1]
    for w, (a, b) in vn.items():
        d = hypot(a, b)
        vn[w] = (a / d, b / d) if d > 1e-12 else (0.0, 0.0)

    # a rounded bull-nose column of intermediate vertices per boundary
    # vertex (shared between its two incident wall strips, so watertight)
    rim = {}
    if segs > 1:
        for w, (nX, nY) in vn.items():
            x, y, zt = verts[w]
            h = zt - half
            for s in range(1, segs):
                phi = pi * s / segs
                zz = h + half * cos(phi)
                off = rim_bulge * half * sin(phi)
                rim[(w, s)] = len(verts)
                verts.append((x + off * nX, y + off * nY, zz))

    def col(w, L):
        if L == 0:
            return w
        if L == segs:
            return bottom_of(w)
        return rim[(w, L)]

    wall_faces = []
    for (u, v) in boundary:
        for L in range(segs):
            wall_faces.append((col(u, L), col(v, L),
                               col(v, L + 1), col(u, L + 1)))

    faces = top_faces + bot_faces + wall_faces
    mats = ([0] * len(top_faces) + [1] * len(bot_faces)
            + [2] * len(wall_faces))
    return verts, faces, mats


# --------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------
#
# Each preset selects a module height style and its default look; the
# raw parameters below remain editable in the redo panel.  "Continua"
# is Erwin Hauer Studios' brand and is kept out of the UI labels; the
# homage designs are named descriptively.

PRESET_ITEMS = [
    ('DESIGN5', "Saddle Lattice (after Design 5)",
     "The egg-crate cos*cos saddle lattice with round apertures, in "
     "the family of Hauer's Design 5 screen"),
    ('WEAVE', "Weave",
     "A shallow woven undulation, cos x + cos y"),
    ('DIAGONAL', "Diagonal Brace",
     "A diagonally braced saddle, cos(x+y)*cos(x-y)"),
    ('DESIGN6', "Relief Wall (after Design 6)",
     "The saddle surface as a solid undulating relief wall with no "
     "perforations, in the family of Hauer's Design 6"),
    ('HYPAR', "Hypar (Carlberg)",
     "Ruled hyperbolic-paraboloid modules with sharp creases along "
     "the cell edges, in Norman Carlberg's harder-edged manner"),
]

_PRESET_STYLE = {'DESIGN5': 'SADDLE', 'WEAVE': 'WEAVE',
                 'DIAGONAL': 'DIAGONAL', 'DESIGN6': 'SADDLE',
                 'HYPAR': 'HYPAR'}


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    # top face / bottom face / rim wall
    _PALETTE = [(0.82, 0.79, 0.72, 1.0), (0.72, 0.68, 0.60, 1.0),
                (0.55, 0.51, 0.45, 1.0)]

    class MESH_OT_modular_screen_add(bpy.types.Operator, AddObjectHelper):
        """Add a Hauer/Carlberg continuous perforated screen wall: one
        saddle module tiled seamlessly, thickened into a slab and
        perforated with a smooth aperture per module"""
        bl_idname = "mesh.modular_screen_add"
        bl_label = "Modular Screen (Hauer-Carlberg)"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(name="Design", items=PRESET_ITEMS,
                             default='DESIGN5')
        nx: IntProperty(name="Cells X", default=5, min=2, max=24,
                        description="Modules across")
        ny: IntProperty(name="Cells Y", default=5, min=2, max=24,
                        description="Modules down")
        amp: FloatProperty(
            name="Relief Depth", default=0.5, min=0.02, max=2.0,
            description="Undulation amplitude of the saddle midsurface")
        thick: FloatProperty(
            name="Thickness", default=0.14, min=0.02, max=0.6,
            description="Wall thickness of the slab")
        hole: FloatProperty(
            name="Aperture", default=0.34, min=0.0, max=0.47,
            description="Aperture radius per module in cell units "
                        "(0 = solid wall; ignored by the Relief Wall "
                        "design)")
        frame: BoolProperty(
            name="Solid Border", default=True,
            description="Leave the outer ring of modules unperforated "
                        "as a frame")
        res: IntProperty(
            name="Resolution", default=6, min=3, max=12,
            description="Samples per cell edge and radial rings; higher "
                        "gives rounder apertures and smoother webbing")
        rim_bulge: FloatProperty(
            name="Rim Bulge", default=0.6, min=0.0, max=1.0,
            description="Round the aperture and border edges into a "
                        "bull-nose (0 = square-cut vertical rim)")
        bulge_segs: IntProperty(
            name="Rim Segments", default=3, min=2, max=6,
            description="Facets across the rounded rim")
        smooth_shading: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 fits a "
                        "2 m cube centered on the origin)")

        def execute(self, context):
            style = _PRESET_STYLE[self.preset]
            hole = 0.0 if self.preset == 'DESIGN6' else self.hole
            verts, faces, mats = build_screen(
                self.nx, self.ny, style, self.amp, self.thick, hole,
                self.res, self.frame, self.rim_bulge, self.bulge_segs)
            obj = pc.build_object(context, "Modular Screen", verts,
                                  faces, mats, palette=_PALETTE,
                                  span=2.0 * self.scale, fit=True,
                                  operator=self)
            if obj is None:
                self.report({'ERROR'}, "empty screen")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            me = obj.data
            me.polygons.foreach_set('use_smooth',
                                    [self.smooth_shading]
                                    * len(me.polygons))
            me.update()
            self.report({'INFO'}, "Modular Screen %s  V=%d F=%d" %
                        (self.preset, len(me.vertices),
                         len(me.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'nx')
            lay.prop(self, 'ny')
            lay.prop(self, 'amp')
            lay.prop(self, 'thick')
            if self.preset != 'DESIGN6':
                lay.prop(self, 'hole')
                lay.prop(self, 'frame')
            lay.prop(self, 'res')
            lay.prop(self, 'rim_bulge')
            if self.rim_bulge > 0.0:
                lay.prop(self, 'bulge_segs')
            lay.prop(self, 'smooth_shading')
            lay.prop(self, 'scale')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.modular_screen_add",
                             icon='MOD_WIREFRAME')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_modular_screen_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_modular_screen_add)


# --------------------------------------------------------------------
# Self-test (pure Python) -- watertight closed-manifold assertions
# --------------------------------------------------------------------

def _manifold_report(verts, faces):
    """(nonmanifold, boundary, holes): count edges shared by != 2 faces
    and estimate the genus from the Euler characteristic.  A watertight
    orientable slab with H through-holes has V - E + F = 2 - 2H."""
    ec = {}
    for f in faces:
        L = len(f)
        for k in range(L):
            u, v = f[k], f[(k + 1) % L]
            key = (u, v) if u < v else (v, u)
            ec[key] = ec.get(key, 0) + 1
    boundary = sum(1 for c in ec.values() if c == 1)
    nonman = sum(1 for c in ec.values() if c > 2)
    V, E, F = len(verts), len(ec), len(faces)
    chi = V - E + F
    holes = (2 - chi) // 2
    return nonman, boundary, holes


def _self_test():
    ok = True
    print("%-10s %6s %6s %6s  %4s %4s %5s  %s"
          % ("style", "V", "E", "F", "nonm", "bnd", "holes", ""))
    cases = [
        ('SADDLE', dict(nx=4, ny=4)),
        ('WEAVE', dict(nx=4, ny=4)),
        ('DIAGONAL', dict(nx=4, ny=4)),
        ('HYPAR', dict(nx=4, ny=4)),
        ('SADDLE', dict(nx=4, ny=4, hole=0.0)),          # relief wall
        ('SADDLE', dict(nx=3, ny=5, frame=False)),       # holes to edge
        ('SADDLE', dict(nx=4, ny=4, rim_bulge=0.0)),     # square rim
    ]
    for style, kw in cases:
        v, f, m = build_screen(style=style, **kw)
        nonman, bnd, holes = _manifold_report(v, f)
        ec = {}
        for fc in f:
            for k in range(len(fc)):
                a, b = fc[k], fc[(k + 1) % len(fc)]
                key = (a, b) if a < b else (b, a)
                ec[key] = ec.get(key, 0) + 1
        watertight = (nonman == 0 and bnd == 0)
        mats_ok = len(m) == len(f) and set(m) <= {0, 1, 2}
        # expected number of perforations
        if kw.get('hole', 0.34) <= 1e-6:
            want = 0
        elif kw.get('frame', True):
            want = max(0, (kw['nx'] - 2) * (kw['ny'] - 2))
        else:
            want = kw['nx'] * kw['ny']
        holes_ok = holes == want
        case_ok = watertight and mats_ok and holes_ok and len(f) > 0
        ok = ok and case_ok
        tag = "" if kw == dict(nx=4, ny=4) else str(kw)
        print("%-10s %6d %6d %6d  %4d %4d %5d  %s  %s"
              % (style, len(v), len(ec), len(f), nonman, bnd, holes,
                 "OK" if case_ok else "BAD", tag))
        if not holes_ok:
            print("      holes=%d want=%d" % (holes, want))

    print("RESULT:", "OK" if ok else "BAD")
    return ok


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        _self_test()
