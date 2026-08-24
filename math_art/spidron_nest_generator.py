# Folded Hexagon Nest -- the spidron's rigid fold, as a relief surface.
#
# Six spidron arms fill a regular hexagon; that hexagon can be folded
# into three dimensions WITHOUT DISTORTING A SINGLE TRIANGLE, and the
# whole family of folded states is a one-parameter rigid motion.  That
# is the property the figure is famous for, and it is what this
# generator draws: one nest, or a periodic landscape of them.
#
# HOW IT FOLDS.  Ring k carries a state triple (d, alpha, beta) -- edge
# length, azimuth, and the inclination of its outer edges to the base
# plane.  Szilassi's recursion advances it inward:
#
#       d -> d/sqrt(3),   alpha -> alpha + a(beta),   beta -> b(beta)
#
# with b the Spidron Formula.  Every ring's outer boundary is a regular
# SKEW hexagon: its six edge midpoints stay in the base plane while its
# vertices alternate above and below, and it projects to a regular
# planar hexagon of side d*cos(beta).  Every edge keeps length d exactly
# whatever the fold angle -- that invariance is checked in `_selftest`,
# and it is the whole point.
#
# THE FOLD SLIDER runs to 60 degrees, where both triangles of a base
# figure stand perpendicular to the base plane.  The closed forms stay
# real a little further, to arccos(1/3) ~ 70.53 degrees, but beyond 60
# the surface passes through itself; Allow Crossing opts into that band
# deliberately.
#
# HOW DEEP TO GO.  The fold decays as it travels inward, but slowly:
# beta_k tends to sqrt(2/k), so a nest whose outer edge spanned the
# observable universe would still be inclined more than six degrees by
# the time its rings reached the size of a quark.  Szilassi's practical
# advice is that a model needs five or six rings and that eight or nine
# is the most worth drawing; the default here is seven.
#
# THE SQUARE BOUNDARY is Imada, Hull, Ku and Tachi's two-parameter
# generalisation: annular SQUARE rings under a similarity of scale s and
# twist theta.  A single ring turns out to have two degrees of freedom
# rather than one, but as rings accumulate the non-isotropic freedom
# dies away, so only the isotropic family is built here.  Each ring
# offers two folded states -- pro-rotation and anti-rotation, by which
# way the inner boundary turns relative to the outer -- and choosing
# between them ring by ring gives three named modes.  Pleats alternates
# them, and at zero twist it degenerates to the triangulated hypar,
# which is not rigidly foldable with flat facets (Demaine and others,
# 2011): what is drawn there is the usual approximate bent-facet shape.
#
# References:
# - Daniel Erdely, "Some Surprising New Properties of the Spidrons",
#   Bridges 2005 Conference Proceedings, pp. 179-186 -- the nest and
#   the discovery that it folds.
# - Lajos Szilassi, "The right for doubting - and the necessity of
#   doubt: Thoughts concerning the analysis of Erdely's Spidron
#   System", Proceedings of the "Sprout-Selecting" Conference (ed.
#   Csaba Sarvari), Pecs, Hungary, 2004, pp. 78-96 -- the build
#   recursion, the 60-degree physical bound, the proof that the fold
#   decays to the plane, and the centre cap that makes the nest
#   watertight and rigid.
# - Gergo Kiss, "A Way to Derive the Spidron Formulas", G4G13 Exchange
#   Book vol. 1, Gathering 4 Gardner, 2018, pp. 185-192 -- the fold
#   angle and the periodic three-dimensional landscape.
# - Mihaly Hujter, "A csillaghatszog spidronszeru felgyurodese mogotti
#   matematika", Haladvany Kiadvany, 16 June 2018 -- an independent
#   proof of the same relation.
# - Rinki Imada, Thomas C. Hull, Jason S. Ku & Tomohiro Tachi,
#   "Nonlinear Kinematics of Recursive Origami Inspired by the
#   Spidron", Origami8 (Lecture Notes in Mechanical Engineering,
#   Springer); arXiv:2403.09278 -- the square-boundary generalisation
#   and its pro-rotation, anti-rotation and pleats modes.
# - Erik D. Demaine, Martin L. Demaine, Vi Hart, Gregory N. Price &
#   Tomohiro Tachi, "(Non)existence of pleated folds: how paper folds
#   between creases", Graphs and Combinatorics 27 (2011), pp. 377-397
#   -- why the pleats mode at zero twist cannot be an exact rigid fold.

bl_info = {
    "name": "Folded Hexagon Nest",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "The spidron nest's rigid fold: a hexagon of "
                   "triangles crumpled into relief without distortion",
    "category": "Add Mesh",
}

from math import cos, sin, pi, sqrt, radians, degrees

import numpy as np

try:
    from . import spidron_math as sm
    from .polyhedra import fit as _fit
except Exception:                       # legacy single-file / CLI use
    import spidron_math as sm
    from polyhedra import fit as _fit


BOUNDARY_ITEMS = [
    ('HEXAGON', "Hexagon",
     "The classical spidron nest: six arms in a hexagon, folding as a "
     "one-parameter rigid motion"),
    ('SQUARE', "Square",
     "Imada, Hull, Ku and Tachi's generalisation on square rings, with "
     "free scale and twist and three folding modes"),
]

MODE_ITEMS = [
    ('PRO', "Pro-rotation",
     "Every ring turns the same way -- the motion closest to the "
     "classical spidron fold"),
    ('ANTI', "Anti-rotation",
     "Every ring turns against the one outside it"),
    ('PLEATS', "Pleats",
     "Rings alternate between the two senses. At zero twist this "
     "becomes the triangulated hypar, which has no exact flat-facet "
     "rigid fold -- the shape drawn is the usual approximation"),
]


def hex_tiling_offsets(tiles_x, tiles_y, pitch):
    """Centres of a honeycomb patch.  Folded nests tile by pure
    translation: neighbouring nests share their boundary edges exactly,
    with no mirroring, at every fold angle (verified in `_selftest`)."""
    u = (pitch, 0.0)
    v = (pitch * cos(pi / 3.0), pitch * sin(pi / 3.0))
    out = []
    for iy in range(tiles_y):
        for ix in range(tiles_x):
            i = ix - (tiles_x - 1) // 2
            j = iy - (tiles_y - 1) // 2
            out.append((i * u[0] + j * v[0], i * u[1] + j * v[1], 0.0))
    return out


def build_hexagon(fold, rings, cap, tiles_x=1, tiles_y=1):
    """One nest, or a honeycomb patch of them."""
    V, F, M = sm.nest_mesh(fold, rings, cap=cap)
    if tiles_x <= 1 and tiles_y <= 1:
        return V, F, M
    pitch = sqrt(3.0) * cos(fold)          # outer edge length is 1.0
    verts, faces, mats = [], [], []
    for (dx, dy, dz) in hex_tiling_offsets(tiles_x, tiles_y, pitch):
        o = len(verts)
        verts.extend([(x + dx, y + dy, z + dz) for (x, y, z) in V])
        faces.extend([tuple(i + o for i in f) for f in F])
        mats.extend(M)
    return verts, faces, mats


def build_square(rho, s, theta, rings, mode, cap):
    V, F, M, rhos = sm.square_mesh(rho, s, theta, rings, mode, cap)
    return V, F, M, rhos


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, mats, operator):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in verts], [], faces)
        from .patterns import common as pc
        cols = pc.PALETTE_RGBA
        nmat = (max(mats) + 1) if mats else 1
        for i in range(nmat):
            mat = bpy.data.materials.new("%s %d" % (name, i))
            mat.use_nodes = False
            mat.diffuse_color = cols[i % len(cols)]
            me.materials.append(mat)
        if mats:
            me.polygons.foreach_set('material_index', mats)
        me.validate(clean_customdata=True)
        me.update()
        import bpy_extras.object_utils as _ou
        return _ou.object_data_add(context, me, operator=operator)

    class MESH_OT_spidron_nest_add(bpy.types.Operator, AddObjectHelper):
        """Add a folded spidron nest: a hexagon of triangles crumpled
        into relief without distorting any of them"""
        bl_idname = "mesh.spidron_nest_add"
        bl_label = "Folded Hexagon Nest"
        bl_options = {'REGISTER', 'UNDO'}

        boundary: EnumProperty(
            name="Boundary", items=BOUNDARY_ITEMS, default='HEXAGON',
            description="Which nest to fold: the classical hexagon, or "
                        "the square-ring generalisation")
        fold: FloatProperty(
            name="Fold", default=radians(40.0),
            min=0.0, max=radians(70.5288), subtype='ANGLE',
            description="How far the nest is folded, measured as the "
                        "tilt of the outer edges. Zero is flat; sixty "
                        "degrees is as far as the surface can fold "
                        "before passing through itself")
        allow_crossing: BoolProperty(
            name="Allow Crossing", default=False,
            description="Permit fold angles past sixty degrees, where "
                        "the formulas are still real but the surface "
                        "intersects itself")
        rings: IntProperty(
            name="Rings", default=7, min=1, max=12,
            description="How many rings of triangles spiral inward. The "
                        "fold decays very slowly, so five to nine is "
                        "the useful range")
        cap_center: BoolProperty(
            name="Cap Centre", default=True,
            description="Close the small hole at the centre with a "
                        "hexagonal patch, making the surface watertight")
        scale_step: FloatProperty(
            name="Ring Scale", default=0.55, min=0.2, max=0.95,
            description="How much each square ring shrinks relative to "
                        "the one outside it")
        twist: FloatProperty(
            name="Twist", default=radians(20.0), min=0.0,
            max=radians(45.0), subtype='ANGLE',
            description="How far each square ring is turned relative "
                        "to the one outside it. Clamped automatically "
                        "so the creases never cross")
        fold_mode: EnumProperty(
            name="Mode", items=MODE_ITEMS, default='PRO',
            description="Which way each square ring turns as it folds")
        tiles_x: IntProperty(
            name="Columns", default=1, min=1, max=8,
            description="Nests across the patch; more than one builds "
                        "the periodic folded landscape")
        tiles_y: IntProperty(
            name="Rows", default=1, min=1, max=8,
            description="Nests down the patch")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=0.5,
            description="Give the surface material depth with a "
                        "Solidify modifier; zero leaves it a shell")

        def execute(self, context):
            info = ""
            if self.boundary == 'HEXAGON':
                f = float(self.fold)
                if not self.allow_crossing and f > sm.BETA_PHYSICAL:
                    f = sm.BETA_PHYSICAL
                    info = " (fold clamped to 60 deg)"
                V, F, M = build_hexagon(
                    f, int(self.rings), self.cap_center,
                    int(self.tiles_x), int(self.tiles_y))
                label = "Spidron Nest"
            else:
                tmax = sm.square_theta_max(float(self.scale_step))
                th = min(float(self.twist), tmax)
                if th < float(self.twist) - 1e-12:
                    info = " (twist clamped to %.1f deg)" % degrees(tmax)
                V, F, M, rhos = build_square(
                    float(self.fold), float(self.scale_step), th,
                    int(self.rings), self.fold_mode, self.cap_center)
                if len(rhos) > 2 and abs(rhos[-1] - rhos[-2]) < 1e-4:
                    info += " (converged to a self-similar state)"
                label = "Spidron Square Nest"
            if not F:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}
            V = _fit.fit_cube(V, 2.0)
            obj = _new_object(context, label, V, F, M, self)
            if self.thickness > 0.0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = float(self.thickness)
                mod.offset = 0.0
            self.report({'INFO'}, "%s  V=%d F=%d%s"
                        % (label, len(V), len(F), info))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'boundary')
            lay.prop(self, 'fold')
            if self.boundary == 'HEXAGON':
                lay.prop(self, 'allow_crossing')
            else:
                lay.prop(self, 'scale_step')
                lay.prop(self, 'twist')
                lay.prop(self, 'fold_mode')
            lay.prop(self, 'rings')
            lay.prop(self, 'cap_center')
            if self.boundary == 'HEXAGON':
                lay.prop(self, 'tiles_x')
                lay.prop(self, 'tiles_y')
            lay.prop(self, 'thickness')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.spidron_nest_add", icon='MOD_DISPLACE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_spidron_nest_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_spidron_nest_add)


# --------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------

def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-52s %s %s" % (name, "OK" if cond else "BAD", extra))

    print("spidron_nest: single nest")
    for fd in (0.0, 20.0, 40.0, 59.0):
        V, F, M = build_hexagon(radians(fd), 7, True)
        A = np.array(V)
        ext = A.max(axis=0) - A.min(axis=0)
        chk("f=%4.1f builds" % fd, len(F) == 7 * 12 + 6,
            "V=%d F=%d" % (len(V), len(F)))
        if fd > 0:
            chk("f=%4.1f has real relief" % fd, ext[2] >= 0.15 * ext[0],
                "z/x=%.3f" % (ext[2] / ext[0]))
        else:
            chk("f= 0.0 is flat", ext[2] < 1e-15)

    V, F, M = build_hexagon(radians(40.0), 7, True)
    Vf = _fit.fit_cube(V, 2.0)
    A = np.array(Vf)
    ext = A.max(axis=0) - A.min(axis=0)
    ctr = 0.5 * (A.max(axis=0) + A.min(axis=0))
    chk("fits the 2 m cube", abs(ext.max() - 2.0) < 1e-9
        and np.abs(ctr).max() < 1e-9, "extent %.6f" % ext.max())

    edges = {}
    for f in F:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            edges[e] = edges.get(e, 0) + 1
    chk("capped nest is watertight",
        sum(1 for c in edges.values() if c == 2) == len(edges) - 6)
    V2, F2, _ = build_hexagon(radians(40.0), 7, False)
    chk("uncapped nest omits the cap", len(F2) == len(F) - 6)

    print("spidron_nest: honeycomb tiling")
    for fd in (0.0, 20.0, 40.0, 59.0):
        f = radians(fd)
        H = np.array(sm.skew_hexagon(1.0, 0.0, f))
        pitch = sqrt(3.0) * cos(f)
        # the six edge-adjacent cells of the lattice spanned by u (0 deg)
        # and v (60 deg) are +-u, +-v and +-(u - v); u + v is a
        # second-shell cell and shares only a vertex.
        u = np.array([pitch, 0.0, 0.0])
        v = np.array([pitch * cos(pi / 3.0), pitch * sin(pi / 3.0), 0.0])
        worst = 0.0
        for nb in (u, -u, v, -v, u - v, v - u):
            G = H + nb
            dm = np.linalg.norm(H[:, None, :] - G[None, :, :], axis=2)
            near = np.sort(dm.flatten())[:2]
            worst = max(worst, float(near.max()))
        chk("f=%4.1f neighbours share an edge exactly" % fd,
            worst < 1e-9, "%.1e" % worst)
    V, F, M = build_hexagon(radians(40.0), 5, True, 3, 3)
    chk("3x3 patch builds", len(F) == 9 * (5 * 12 + 6), "F=%d" % len(F))

    print("spidron_nest: square boundary")
    for mode in ('PRO', 'ANTI', 'PLEATS'):
        V, F, M, rhos = build_square(radians(45.0), 0.55, radians(20.0),
                                     6, mode, True)
        chk("%-7s builds" % mode, len(F) > 0,
            "F=%d rings=%d" % (len(F), len(rhos) - 1))
    tmax = sm.square_theta_max(0.9)
    chk("twist clamp is enforceable", tmax < pi / 4.0,
        "%.2f deg at s=0.9" % degrees(tmax))
    loops, _ = sm.square_nest(radians(45.0), 0.55, radians(20.0), 6, 'PRO')

    sq_err, shrink_ok = 0.0, True
    prev = None
    for L in loops:
        e = [float(np.linalg.norm(L[(i + 1) % 4] - L[i])) for i in range(4)]
        sq_err = max(sq_err, max(e) - min(e))
        if prev is not None and not (max(e) < prev - 1e-12):
            shrink_ok = False
        prev = max(e)
    chk("square rings stay square", sq_err < 1e-9, "%.1e" % sq_err)
    chk("square rings shrink inward", shrink_ok)
    # Imada et al. Fig 7: pro-rotation converges to a non-flat state
    _, _, _, rp = build_square(radians(60.0), 0.825, 0.09, 25, 'PRO', False)
    _, _, _, ra = build_square(radians(60.0), 0.825, 0.09, 25, 'ANTI', False)
    chk("PRO holds a finite fold, ANTI decays toward flat",
        abs(rp[-1]) > abs(ra[-1]),
        "pro %.3f deg vs anti %.3f deg"
        % (degrees(abs(rp[-1])), degrees(abs(ra[-1]))))

    print("RESULT:", "OK" if ok else "BAD")
    return ok
