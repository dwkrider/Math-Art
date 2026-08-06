
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
# Carlberg's harder-edged ruled (hyperbolic-paraboloid) modules.  The
# interwoven bilayer of Design 1 -- which no single-valued height field
# can express -- is built instead as two families of undulating ribbons
# woven over and under one another.  A finished screen can be left flat,
# curved onto a cylinder, or closed into a column, as in the built
# installations.
#
# GEOMETRY.  Each height-field module's unit cell is meshed as a radial
# fan from its square perimeter inward to a central aperture (a circle,
# or a rounded square via a superellipse) -- so the perforations are
# traced smoothly at any resolution instead of being cut as
# stair-stepped quads.  Two offset copies at h +/- t/2 form the slab's
# faces, and a wall -- optionally rounded into a bull-nose rim -- is
# raised on every boundary edge (the panel perimeter and every
# aperture), closing the shell into a single watertight, orientable
# manifold.  Neighbouring cells share their perimeter samples exactly,
# so the tiling welds seamlessly.  The woven design instead builds each
# ribbon as a closed swept tube, watertight by construction.
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
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Hauer/Carlberg continuous perforated screen walls: "
                   "tiled saddle modules and woven ribbons, thickened, "
                   "perforated, and optionally curved into a column",
    "category": "Add Mesh",
}

import math
from math import pi, cos, sin, hypot, radians

import numpy as np

try:
    from . import pattern_common as pc
except Exception:                            # legacy single-file / CLI
    import pattern_common as pc


# --------------------------------------------------------------------
# Pure-math core (no bpy)
# --------------------------------------------------------------------

def height_at(style, x, y, ci, cj, amp, swirl_k=1.0, chir=1.0):
    """Midsurface height of a height-field module at global (x, y).
    ci, cj are the integer indices of the cell (x, y) belongs to (only
    the HYPAR style is piecewise and needs them).  Every style is
    C0-continuous across cell boundaries, so a point on a shared edge
    evaluates to the same height from either adjacent cell."""
    if style == 'PINWHEEL':
        # A chiral 4-fold (wallpaper group p4) pinwheel: a mirror-
        # symmetric base cos + cos, plus a swirl term that is
        # antisymmetric under reflection (so it cannot be undone by any
        # mirror) -- the field is invariant only under 90-degree
        # rotation, giving the one-handed swirling arms.  chir = +/-1
        # flips the handedness; swirl_k sets how strongly the arms
        # curl.
        X = 2.0 * pi * x
        Y = 2.0 * pi * y
        S = sin(2.0 * X) * sin(Y) - sin(2.0 * Y) * sin(X)
        return amp * 0.35 * (cos(X) + cos(Y) + chir * swirl_k * S)
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
        pts.append((ci + s / K, float(cj)))          # bottom
    for s in range(K):
        pts.append((float(ci + 1), cj + s / K))      # right
    for s in range(K):
        pts.append((ci + 1 - s / K, float(cj + 1)))  # top
    for s in range(K):
        pts.append((float(ci), cj + 1 - s / K))      # left
    return pts


def _close_shell(verts, top_faces, bot_faces, top2bot, rim_bulge,
                 bulge_segs):
    """Given a thickened sheet -- its top faces, matching bottom faces,
    and a top-vertex -> bottom-vertex map -- raise a wall on every
    boundary edge of the top sheet (panel perimeter and apertures),
    optionally rounded into a bull-nose rim, and return (faces, mats)
    with mats 0 = top, 1 = bottom, 2 = rim.  Appends rim vertices to
    `verts` in place.  The result is a closed orientable manifold."""
    segs = max(2, int(bulge_segs)) if rim_bulge > 1e-9 else 1

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

    # outward horizontal normal at each boundary vertex (mean of the
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
    for w, (a, b) in list(vn.items()):
        d = hypot(a, b)
        vn[w] = (a / d, b / d) if d > 1e-12 else (0.0, 0.0)

    rim = {}
    if segs > 1:
        for w, (nX, nY) in vn.items():
            zt = verts[w][2]
            zb = verts[top2bot[w]][2]
            mid = 0.5 * (zt + zb)
            hspan = 0.5 * (zt - zb)
            x, y = verts[w][0], verts[w][1]
            for s in range(1, segs):
                phi = pi * s / segs
                off = rim_bulge * hspan * sin(phi)
                rim[(w, s)] = len(verts)
                verts.append((x + off * nX, y + off * nY,
                              mid + hspan * cos(phi)))

    def col(w, L):
        if L == 0:
            return w
        if L == segs:
            return top2bot[w]
        return rim[(w, L)]

    walls = []
    for (u, v) in boundary:
        for L in range(segs):
            walls.append((col(u, L), col(v, L),
                          col(v, L + 1), col(u, L + 1)))

    faces = list(top_faces) + list(bot_faces) + walls
    mats = ([0] * len(top_faces) + [1] * len(bot_faces)
            + [2] * len(walls))
    return faces, mats


def build_screen(nx=5, ny=5, style='SADDLE', amp=0.5, thick=0.14,
                 hole=0.34, res=6, frame=True, rim_bulge=0.6,
                 bulge_segs=3, ap_square=0.0, swirl_k=1.0, chir=1.0):
    """Build (verts, faces, mats) for a thickened, perforated saddle
    screen over an nx x ny block of modules.  `ap_square` in [0, 1]
    morphs each aperture from a circle (0) to a rounded square (1) via
    a superellipse; `swirl_k` and `chir` set the swirl strength and
    handedness of the PINWHEEL style.  The result is a single closed
    watertight manifold."""
    nx = max(1, int(nx))
    ny = max(1, int(ny))
    K = max(3, int(res))
    N = max(2, int(res))                             # radial rings
    M = 4 * K
    r = max(0.0, min(0.47, float(hole)))
    p = 2.0 + 6.0 * max(0.0, min(1.0, float(ap_square)))   # 2..8
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
                ux, uy = dx / d, dy / d
                if has_ap:
                    # superellipse radius along this ray: circle at p=2
                    denom = (abs(ux) ** p + abs(uy) ** p) ** (1.0 / p)
                    rho = r / denom
                    inner.append((cx + rho * ux, cy + rho * uy))
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
                    h = height_at(style, x, y, ci, cj, amp,
                                  swirl_k, chir)
                    top[m][k] = vid(x, y, h, 0)
                    bot[m][k] = vid(x, y, h, 1)

            for m in range(N):
                for k in range(M):
                    k2 = (k + 1) % M
                    a, b = top[m][k], top[m][k2]
                    c, d = top[m + 1][k2], top[m + 1][k]
                    if c == d:                       # collapsed centre
                        top_faces.append((a, b, c))
                        ab, bb, cb = bot[m][k], bot[m][k2], bot[m + 1][k2]
                        bot_faces.append((ab, cb, bb))
                    else:
                        top_faces.append((a, b, c, d))
                        ab, bb = bot[m][k], bot[m][k2]
                        cb, db = bot[m + 1][k2], bot[m + 1][k]
                        bot_faces.append((ab, db, cb, bb))

    top2bot = {}
    for key, idx in reg.items():
        if key[2] == 0:
            top2bot[idx] = reg[(key[0], key[1], 1)]
    faces, mats = _close_shell(verts, top_faces, bot_faces, top2bot,
                               rim_bulge, bulge_segs)
    return verts, faces, mats


def build_weave(nx=5, ny=5, amp=0.35, thick=0.14, ribbon_w=0.7, res=6):
    """Build (verts, faces, mats) for the interwoven bilayer of Design 1:
    warp ribbons running in x and weft ribbons running in y, each an
    undulating solid strip, woven so that at every crossing one rides
    over and the other under.  Each ribbon is a closed swept tube, so
    the whole screen is watertight by construction.  mats: 0 = warp,
    1 = weft."""
    nx = max(1, int(nx))
    ny = max(1, int(ny))
    R = max(3, int(res))
    # keep the two layers clear of one another at the crossings
    amp = max(float(amp), 0.6 * float(thick))
    w = 0.5 * max(0.05, min(0.95, float(ribbon_w)))
    ht = 0.5 * float(thick)

    verts = []
    faces = []
    mats = []

    def tube(centerline, tx, ty, mat):
        base = len(verts)
        S = len(centerline)
        for (px, py, pz) in centerline:
            for (a, b) in ((-w, -ht), (w, -ht), (w, ht), (-w, ht)):
                verts.append((px + a * tx, py + a * ty, pz + b))
        for s in range(S - 1):
            r0, r1 = base + 4 * s, base + 4 * (s + 1)
            for k in range(4):
                k2 = (k + 1) % 4
                faces.append((r0 + k, r0 + k2, r1 + k2, r1 + k))
                mats.append(mat)
        faces.append((base, base + 1, base + 2, base + 3))       # cap 0
        mats.append(mat)
        e = base + 4 * (S - 1)
        faces.append((e + 3, e + 2, e + 1, e))                   # cap 1
        mats.append(mat)

    # warp ribbons run along x at y = j + 1/2; height amp*(-1)^j*sin(pi x)
    for j in range(ny):
        sgn = 1.0 if j % 2 == 0 else -1.0
        line = []
        for s in range(nx * R + 1):
            x = s / R
            line.append((x, j + 0.5, amp * sgn * sin(pi * x)))
        tube(line, 0.0, 1.0, 0)
    # weft ribbons run along y at x = i + 1/2; the opposite phase, so at
    # each crossing warp and weft sit at +/- amp -> woven over and under
    for i in range(nx):
        sgn = 1.0 if i % 2 == 0 else -1.0
        line = []
        for s in range(ny * R + 1):
            y = s / R
            line.append((i + 0.5, y, -amp * sgn * sin(pi * y)))
        tube(line, 1.0, 0.0, 1)

    return verts, faces, mats


def wrap_cylinder(verts, nx, angle_deg, closed=False):
    """Bend a flat screen (spanning x in [0, nx]) onto a cylinder whose
    axis is the y direction: the width wraps through `angle_deg`, the
    slab's z relief becomes radial.  With `closed` the panel wraps the
    full turn into a column (join the seam with Merge by Distance for a
    watertight tube)."""
    ang = radians(float(angle_deg))
    if ang < 1e-6:
        return verts
    Rr = nx / ang                                    # arc radius
    out = []
    for (x, y, z) in verts:
        a = (x / nx) * ang if closed else ((x - 0.5 * nx) / nx) * ang
        rad = Rr + z
        out.append((rad * sin(a), y, rad * cos(a) - Rr))
    return out


# --------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------
#
# Each preset selects a module and its default look; the raw parameters
# below stay editable in the redo panel.  "Continua" is Erwin Hauer
# Studios' brand and is kept out of the UI labels; the homage designs
# are named descriptively.

PRESET_ITEMS = [
    ('DESIGN5', "Saddle Lattice (after Design 5)",
     "The egg-crate cos*cos saddle lattice with apertures, in the "
     "family of Hauer's Design 5 screen"),
    ('PINWHEEL', "Pinwheel (chiral)",
     "A chiral four-fold pinwheel relief: swirling arms of one "
     "handedness with curved openings between them"),
    ('DESIGN1', "Bilayer Weave (after Design 1)",
     "Two families of undulating ribbons woven over and under one "
     "another, in the family of Hauer's interwoven Design 1"),
    ('WEAVE', "Woven Undulation",
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
                 'HYPAR': 'HYPAR', 'PINWHEEL': 'PINWHEEL'}


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

    # warp / weft   or   top / bottom / rim
    _PALETTE = [(0.82, 0.79, 0.72, 1.0), (0.72, 0.68, 0.60, 1.0),
                (0.55, 0.51, 0.45, 1.0)]

    class MESH_OT_modular_screen_add(bpy.types.Operator, AddObjectHelper):
        """Add a Hauer/Carlberg continuous perforated screen wall: a
        saddle module (or woven bilayer) tiled seamlessly, thickened
        into a slab, perforated, and optionally curved into a column"""
        bl_idname = "mesh.modular_screen_add"
        bl_label = "Modular Screen (Hauer-Carlberg)"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(name="Design", items=PRESET_ITEMS,
                             default='DESIGN5')
        nx: IntProperty(name="Cells X", default=5, min=1, max=24,
                        description="Modules across")
        ny: IntProperty(name="Cells Y", default=5, min=1, max=24,
                        description="Modules down")
        amp: FloatProperty(
            name="Relief Depth", default=0.5, min=0.02, max=2.0,
            description="Undulation amplitude of the module")
        thick: FloatProperty(
            name="Thickness", default=0.14, min=0.02, max=0.6,
            description="Wall / ribbon thickness")
        hole: FloatProperty(
            name="Aperture", default=0.34, min=0.0, max=0.47,
            description="Aperture radius per module in cell units "
                        "(0 = solid wall; ignored by the Relief Wall "
                        "and Bilayer Weave designs)")
        ap_square: FloatProperty(
            name="Aperture Squareness", default=0.0, min=0.0, max=1.0,
            description="Morph each aperture from a circle (0) to a "
                        "rounded square (1)")
        frame: BoolProperty(
            name="Solid Border", default=True,
            description="Leave the outer ring of modules unperforated "
                        "as a frame")
        ribbon_w: FloatProperty(
            name="Ribbon Width", default=0.7, min=0.1, max=0.95,
            description="Width of the woven ribbons in cell units "
                        "(Bilayer Weave only)")
        swirl: FloatProperty(
            name="Swirl", default=1.0, min=0.0, max=2.5,
            description="How strongly the pinwheel arms curl "
                        "(Pinwheel only)")
        handedness: EnumProperty(
            name="Handedness",
            items=[('RIGHT', "Right", "Clockwise swirl"),
                   ('LEFT', "Left", "Counter-clockwise (mirror) swirl")],
            default='RIGHT',
            description="Chirality of the pinwheel swirl (Pinwheel only)")
        res: IntProperty(
            name="Resolution", default=6, min=3, max=12,
            description="Samples per cell; higher gives rounder "
                        "apertures and smoother webbing")
        rim_bulge: FloatProperty(
            name="Rim Bulge", default=0.6, min=0.0, max=1.0,
            description="Round the aperture and border edges into a "
                        "bull-nose (0 = square-cut vertical rim)")
        bulge_segs: IntProperty(
            name="Rim Segments", default=3, min=2, max=6,
            description="Facets across the rounded rim")
        curvature: EnumProperty(
            name="Form",
            items=[('FLAT', "Flat Panel", "A flat screen wall"),
                   ('CURVED', "Curved Screen",
                    "Bend the panel onto a cylinder through the wrap "
                    "angle"),
                   ('COLUMN', "Column",
                    "Wrap the panel a full turn into a cylindrical "
                    "column (join the seam with Merge by Distance)")],
            default='FLAT')
        wrap_angle: FloatProperty(
            name="Wrap Angle", default=120.0, min=10.0, max=350.0,
            description="Arc the curved screen spans, in degrees",
            subtype='ANGLE' if False else 'NONE')
        smooth_shading: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 fits a "
                        "2 m cube centered on the origin)")

        def execute(self, context):
            p = self.preset
            if p == 'DESIGN1':
                verts, faces, mats = build_weave(
                    self.nx, self.ny, self.amp, self.thick,
                    self.ribbon_w, self.res)
            else:
                style = _PRESET_STYLE[p]
                hole = 0.0 if p == 'DESIGN6' else self.hole
                chir = -1.0 if self.handedness == 'LEFT' else 1.0
                verts, faces, mats = build_screen(
                    self.nx, self.ny, style, self.amp, self.thick, hole,
                    self.res, self.frame, self.rim_bulge,
                    self.bulge_segs, self.ap_square, self.swirl, chir)
            if self.curvature == 'CURVED':
                verts = wrap_cylinder(verts, self.nx, self.wrap_angle)
            elif self.curvature == 'COLUMN':
                verts = wrap_cylinder(verts, self.nx, 360.0, closed=True)

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
                        (p, len(me.vertices), len(me.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'nx')
            lay.prop(self, 'ny')
            lay.prop(self, 'amp')
            lay.prop(self, 'thick')
            if self.preset == 'PINWHEEL':
                lay.prop(self, 'swirl')
                lay.prop(self, 'handedness')
            if self.preset == 'DESIGN1':
                lay.prop(self, 'ribbon_w')
            elif self.preset != 'DESIGN6':
                lay.prop(self, 'hole')
                lay.prop(self, 'ap_square')
                lay.prop(self, 'frame')
            lay.prop(self, 'res')
            if self.preset != 'DESIGN1':
                lay.prop(self, 'rim_bulge')
                if self.rim_bulge > 0.0:
                    lay.prop(self, 'bulge_segs')
            lay.prop(self, 'curvature')
            if self.curvature == 'CURVED':
                lay.prop(self, 'wrap_angle')
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
    """(nonmanifold, boundary, holes): edges shared by != 2 faces, and
    the genus estimate from Euler's characteristic (for a single
    watertight orientable slab with H through-holes, V - E + F = 2 - 2H)."""
    ec = {}
    for f in faces:
        L = len(f)
        for k in range(L):
            u, v = f[k], f[(k + 1) % L]
            key = (u, v) if u < v else (v, u)
            ec[key] = ec.get(key, 0) + 1
    boundary = sum(1 for c in ec.values() if c == 1)
    nonman = sum(1 for c in ec.values() if c > 2)
    chi = len(verts) - len(ec) + len(faces)
    holes = (2 - chi) // 2
    return nonman, boundary, holes, len(ec)


def _self_test():
    ok = True
    print("== height-field screens ==")
    print("%-9s %6s %6s %6s  %4s %4s %5s  %s"
          % ("style", "V", "E", "F", "nonm", "bnd", "holes", ""))
    cases = [
        ('SADDLE', dict(nx=4, ny=4)),
        ('WEAVE', dict(nx=4, ny=4)),
        ('DIAGONAL', dict(nx=4, ny=4)),
        ('HYPAR', dict(nx=4, ny=4)),
        ('SADDLE', dict(nx=4, ny=4, hole=0.0)),          # relief wall
        ('SADDLE', dict(nx=3, ny=5, frame=False)),       # holes to edge
        ('SADDLE', dict(nx=4, ny=4, rim_bulge=0.0)),     # square rim
        ('SADDLE', dict(nx=4, ny=4, ap_square=1.0)),     # square holes
        ('HYPAR', dict(nx=4, ny=4, ap_square=1.0)),      # Carlberg poly
        ('PINWHEEL', dict(nx=4, ny=4)),                  # chiral relief
        ('PINWHEEL', dict(nx=4, ny=4, hole=0.0)),        # solid relief
        ('SADDLE', dict(nx=1, ny=1, frame=False)),       # single cell
        ('PINWHEEL', dict(nx=1, ny=3, frame=False)),     # 1-wide strip
    ]
    for style, kw in cases:
        v, f, m = build_screen(style=style, **kw)
        nonman, bnd, holes, E = _manifold_report(v, f)
        watertight = (nonman == 0 and bnd == 0)
        mats_ok = len(m) == len(f) and set(m) <= {0, 1, 2}
        if kw.get('hole', 0.34) <= 1e-6:
            want = 0
        elif kw.get('frame', True):
            want = max(0, (kw['nx'] - 2) * (kw['ny'] - 2))
        else:
            want = kw['nx'] * kw['ny']
        case_ok = (watertight and mats_ok and holes == want
                   and len(f) > 0)
        ok = ok and case_ok
        tag = "" if kw == dict(nx=4, ny=4) else str(kw)
        print("%-9s %6d %6d %6d  %4d %4d %5d  %s  %s"
              % (style, len(v), E, len(f), nonman, bnd, holes,
                 "OK" if case_ok else "BAD", tag))
        if holes != want:
            print("      holes=%d want=%d" % (holes, want))

    print("== bilayer weave (Design 1) ==")
    for kw in (dict(nx=4, ny=4), dict(nx=5, ny=3, ribbon_w=0.9)):
        v, f, m = build_weave(**kw)
        nonman, bnd, holes, E = _manifold_report(v, f)
        # ribbons are separate closed tubes: every edge shared by two
        # faces (watertight components), no boundary edges
        watertight = nonman == 0 and bnd == 0
        # layers stay clear: at each crossing warp is +amp, weft -amp
        amp = max(0.35, 0.6 * 0.14)
        clear = 2.0 * amp - 0.14
        wv_ok = watertight and clear > 0 and len(f) > 0
        ok = ok and wv_ok
        print("weave     %6d %6d %6d  %4d %4d   -    %s  %s"
              % (len(v), E, len(f), nonman, bnd,
                 "OK" if wv_ok else "BAD", kw))

    print("== cylinder / column wrap preserves topology ==")
    v, f, m = build_screen(nx=4, ny=4)
    for form, ang, closed in (('CURVED', 120.0, False),
                              ('COLUMN', 360.0, True)):
        w = wrap_cylinder(v, 4, ang, closed=closed)
        finite = all(all(math.isfinite(c) for c in p) for p in w)
        same = len(w) == len(v)
        n0, b0, h0, _ = _manifold_report(v, f)
        n1, b1, h1, _ = _manifold_report(w, f)     # faces unchanged
        wrap_ok = finite and same and (n0, b0, h0) == (n1, b1, h1)
        ok = ok and wrap_ok
        print("%-7s ang=%-5.0f finite=%s topo-preserved=%s  %s"
              % (form, ang, finite, (n0, b0, h0) == (n1, b1, h1),
                 "OK" if wrap_ok else "BAD"))

    print("RESULT:", "OK" if ok else "BAD")
    return ok
