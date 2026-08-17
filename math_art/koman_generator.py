
# Ilhan Koman's developable sculptures, for Blender.
#
# NOT a D-form, despite the shared theme of shapes made from unstretched
# flat sheet, and kept apart here for that reason.  A D-form glues TWO
# flat pieces of equal perimeter edge to edge and comes out convex, with
# all of its curvature on one seam and no interior creases at all
# (Demaine and Price).  These are cut from ONE sheet, slit and coiled,
# with nothing glued boundary to boundary and no seam anywhere; the 2007
# paper cites D-forms only as neighbouring work.
#
# Koman (1921-1986) never wrote his method down.  Akgun, Kaya, Koman and
# Akleman reconstructed it from the surviving sculptures: slit a sheet
# from alternating edges so it becomes one long serpentine ribbon, then
# coil that ribbon so each segment -- a "tongue" -- lies flat and slides
# under its neighbour by 2d.  Every slide turns the ribbon by
#
#     a = arctan(2d / h)                                       (Eq. 1)
#
# and n of them close a ring of radius about h/2 (Eq. 2), the tongues
# overlapping like a fanned deck of cards.  The paper between two
# tongues cannot stay flat -- it is longer than the gap it now has to
# bridge -- so it arches, and those arches are the only curved surfaces
# in the piece.
#
# Cutting the same slits into a TRAPEZOID tapers the ribbon so the coil
# never closes: it winds outward as a logarithmic spiral, which is what
# Koman's own metal pieces are.  The paper's account of how is to hold
# the turn constant and shrink the size exponentially along the strip
# ("he shrank h exponentially"); since the radius goes as h/2, that is
# what walks the blades outward.
#
# The blades stay PLANAR.  That is the paper's own "blue pieces stay
# planar", and it is what keeps the sculpture cuttable from flat stock:
# the twisting look of the metal pieces comes from each blade being
# turned further round the coil than the last, never from bending one.
#
# References:
#   T. Akgun, I. Kaya, A. Koman, E. Akleman, "Spiral Developable
#       Sculptures of Ilhan Koman," Bridges 2007, pp. 47-52 -- Eqs. (1)
#       and (2), and Figures 3-6, which are the construction.  Local
#       copy: research/journals/bridges/2007/bridges2007-47/.
#   T. Akgun, A. Koman, E. Akleman, "Developable Sculptural Forms of
#       Ilhan Koman," Bridges 2006, pp. 343-350.  Local copy:
#       research/journals/bridges/2006/bridges2006-343/.
#   I. Koman and F. Ribeyrolles, "On My Approach to Making Nonfigurative
#       Static and Kinetic Sculpture," Leonardo 12(1), 1979, pp. 1-4.

bl_info = {
    "name": "Koman Developable Sculptures",
    "author": "Math Art project (after Ilhan Koman; reconstruction by "
              "Akgun, Kaya, Koman and Akleman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Odds & Ends",
    "description": "Koman's coiled and spiral developable sculptures",
    "category": "Add Mesh",
}

import numpy as np

try:
    from .sharp_creases import mark_sharp_by_angle
except ImportError:                     # flat import outside the package
    from sharp_creases import mark_sharp_by_angle

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

KOMAN_KINDS = ('SPIRAL', 'RING')


def koman_turn(d, h):
    """Turn per panel, a = arctan(2d/h) -- Akgun et al. Eq. (1)."""
    return float(np.arctan2(2.0 * float(d), max(float(h), 1e-9)))


def koman_slide_to_close(n, h=1.0):
    """The slide d that makes `n` panels close the ring exactly.

    Akgun et al. note that a = 2*pi/n closes the arc into a regular
    n-gon with n the number of panels; inverting their Eq. 1 gives the
    slide that does it, d = h tan(a) / 2.  Driving the sculpture from
    the panel count is much the better handle -- pick d directly and
    the ring almost never closes.
    """
    # a = 2*pi/n, and d = h tan(a) / 2.  n >= 5 keeps a under a right
    # angle, where the tangent runs away
    a = 2.0 * np.pi / max(int(n), 5)
    return 0.5 * float(h) * float(np.tan(a))


def build_koman_curl(panels=32, height=1.0, slide=-1.0, width=0.55,
                     arch_samples=10, growth=0.0, skew=0.5, hole=0.45,
                     tilt=0.42, thickness=0.004):
    """Koman's curled strip: a slit sheet coiled into an overlapping ring.

    THE SHEET IS ONE PIECE.  Akgun et al.'s Figure 3 cuts a rectangle
    with slits from alternating edges, which turns it into a single long
    serpentine ribbon; coiling that ribbon is the sculpture.  Every
    segment of it -- a "tongue" -- stays flat and is pressed into a
    common plane, each rotated from the last by
    a = arctan(2d/h) (their Eq. 1), so the tongues overlap like a fanned
    deck of cards around a central hole and `panels` of them sweep a
    ring of radius about h/2 (their Eq. 2).

    The paper between two tongues is the part that cannot stay flat.
    Sliding each tongue under its neighbour by 2d leaves that connector
    longer than the gap it has to bridge, so it buckles into an arch --
    those arches are the only genuinely curved surfaces here, and the
    ribbon being serpentine, they alternate between the outer and the
    inner rim.

    `skew` swings each tongue away from radial, which is what gives the
    pinwheel; `growth` grows the slide along the ribbon so the ring
    opens into a spiral, as Koman's own sculptures do.  `thickness`
    stacks the tongues by a sheet's worth each so the overlaps read as
    layers rather than as coincident faces -- a real coil closes
    slightly out of plane for the same reason.
    """
    n = max(3, int(panels))
    h = max(float(height), 1e-3)
    l = max(float(width), 0.05) * h        # tongue width along the ribbon
    r_in = max(float(hole), 0.02) * h
    m = max(3, int(arch_samples))
    step = float(thickness) * h
    # a negative slide means "whatever closes the ring", which is what
    # anyone actually wants; a chosen d leaves the ring open at some
    # arbitrary angle, as 24 panels at d = 0.06 does (164 degrees)
    if slide is None or float(slide) < 0.0:
        slide = koman_slide_to_close(n, h)

    V = []
    F = []
    rims = []                              # (outer edge, inner edge) per tongue
    slack = []                             # each tongue's own 2d
    ang = 0.0
    d0 = max(float(slide), 1e-5)
    g = 1.0

    for i in range(n):
        # The SPIRAL comes from shrinking h, not from d.  The paper is
        # explicit: keep a constant (which means holding h/d fixed) and
        # vary the size, and Koman himself "shrank h exponentially".
        # Since the ring radius goes as h/2, that is what walks the
        # tongues inward; scaling d alone only changes the turn rate and
        # leaves every tongue on the same circle.
        hi, li, ri, di = h * g, l * g, r_in * g, d0 * g
        c, s = np.cos(ang), np.sin(ang)
        u = np.array([np.cos(ang + skew), np.sin(ang + skew), 0.0])
        # each tongue LEANS: it is lying on top of the ones before it,
        # so its trailing edge rides up on them.  Coplanar tongues merge
        # into one flat annulus in a render and the fanned-deck reading
        # -- which is what the sculpture looks like -- disappears.
        v = np.array([-np.sin(ang + skew) * np.cos(tilt),
                      np.cos(ang + skew) * np.cos(tilt),
                      np.sin(tilt)])
        base = np.array([ri * c, ri * s, i * step])
        p0 = base
        p1 = base + hi * u
        p2 = p1 + li * v
        p3 = base + li * v
        k = len(V)
        V.extend([p0, p1, p2, p3])
        F.append((k, k + 1, k + 2, k + 3))
        rims.append(((k + 1, k + 2), (k, k + 3)))
        slack.append(2.0 * di)
        ang += koman_turn(di, hi)          # constant while h/d is
        g *= (1.0 + float(growth))

    # the connectors: tongue i hands over to tongue i+1 at the outer rim
    # when i is even and at the inner rim when it is odd, because the
    # ribbon runs out, back, out, back through the slit sheet
    for i in range(n - 1):
        side = 0 if i % 2 == 0 else 1
        a0, a1 = rims[i][side]
        b0, b1 = rims[i + 1][side]
        _arch(V, F, np.asarray(V[a0]), np.asarray(V[a1]),
              np.asarray(V[b0]), np.asarray(V[b1]), slack[i], m)

    return np.asarray(V, dtype=float), F


def build_koman_spiral(blades=80, turn=None, growth=0.034, blade_len=1.70,
                       blade_width=0.60, skew=0.70, lean=1.00,
                       twist=0.0, samples=5, core=0.055):
    """Koman's spiral developable: a coiling spine with blades off it.

    THIS IS THE SCULPTURE, as against the closed wreath of
    `build_koman_curl`.  Akgun et al.'s Figure 4 cuts the alternating
    slits into a TRAPEZOID rather than a rectangle, so the ribbon tapers
    and the coil never closes -- it winds outward as a logarithmic
    spiral, which is what Koman's own metal pieces are (their Figures 2
    and 4C).  The paper's account of how: hold the turn per panel `a`
    constant and shrink the size exponentially along the strip, which is
    what they conclude Koman did.

    The uncut part of the sheet is a SPINE that coils, and each slit
    strip is a blade springing from it.  Blades stay planar -- that is
    the paper's own "blue pieces stay planar", and it is what keeps the
    surface developable -- so the twisting look of the metal sculptures
    comes from each blade being turned a little further round the coil
    than the last, not from bending any one of them.

    Every length scales with the local spiral radius, so the form is
    self-similar: `blade_len`, `blade_width` and `spine` are all given
    as multiples of it.  `twist` rotates a blade's far edge against its
    root for the look of the metal pieces; it is off by default because
    a twisted blade is a hyperbolic paraboloid, no longer developable
    and no longer something you can cut from flat stock.
    """
    n = max(3, int(blades))
    a = (2.0 * np.pi / 30.0) if turn is None else float(turn)
    g = 1.0 + float(growth)
    m = max(1, int(samples))
    r = max(float(core), 1e-4)

    V = []
    F = []
    roots = []                              # (inner, outer) of each root edge

    for i in range(n):
        th = i * a
        er = np.array([np.cos(th), np.sin(th), 0.0])
        et = np.array([-np.sin(th), np.cos(th), 0.0])
        u = np.cos(skew) * er + np.sin(skew) * et       # blade long axis
        vp = -np.sin(skew) * er + np.cos(skew) * et     # blade width, in plane
        v = np.cos(lean) * vp + np.sin(lean) * np.array([0.0, 0.0, 1.0])
        B = r * er
        L, W = blade_len * r, blade_width * r

        base = len(V)
        for j in range(m + 1):
            t = j / m
            # the twist turns the width axis about the blade's own long
            # axis as we run out along it (Rodrigues about u)
            ang = twist * t
            vt = (v * np.cos(ang) + np.cross(u, v) * np.sin(ang)
                  + u * float(np.dot(u, v)) * (1.0 - np.cos(ang)))
            root = B + L * t * u
            V.append(root)
            V.append(root + W * vt)
        for j in range(m):
            q = base + 2 * j
            F.append((q, q + 2, q + 3, q + 1))
        roots.append((base, base + 1))
        r *= g

    # the spine: the uncut band of the sheet, joining each blade's root
    # to the next.  It is what makes the whole thing one piece of paper.
    for i in range(n - 1):
        a0, a1 = roots[i]
        b0, b1 = roots[i + 1]
        F.append((a0, b0, b1, a1))

    return np.asarray(V, dtype=float), F


def _arch(V, F, a0, a1, b0, b1, slack, m):
    """A quad strip from edge (a0,a1) to edge (b0,b1), bowed up by
    `slack` of extra paper.

    The two edges are a turn apart, so the straight bridge between them
    is shorter than the paper that has to span it; the surplus goes into
    a circular bow.  Ruling the strip straight across keeps it a
    cylinder patch, so the connector stays developable.
    """
    chord = 0.5 * (float(np.linalg.norm(b0 - a0))
                   + float(np.linalg.norm(b1 - a1)))
    rise = 0.0
    if chord > 1e-9 and slack > 1e-12:
        th = _arc_angle(chord, chord + slack)
        R = chord / (2.0 * max(np.sin(th), 1e-9))
        rise = R * (1.0 - np.cos(th))
    up = np.array([0.0, 0.0, 1.0])
    base = len(V)
    for j in range(m + 1):
        t = j / m
        lift = up * (rise * np.sin(np.pi * t))
        V.append(a0 + t * (b0 - a0) + lift)
        V.append(a1 + t * (b1 - a1) + lift)
    for j in range(m):
        q = base + 2 * j
        F.append((q, q + 2, q + 3, q + 1))


def _arc_angle(chord, length):
    """Half-angle of a circular arc of given chord and arclength.

    Solves chord/length = sin(x)/x by bisection -- it is monotone on
    (0, pi], so there is nothing clever to do and nothing to go wrong.
    """
    ratio = min(max(chord / max(length, 1e-12), 1e-6), 1.0 - 1e-12)
    lo, hi = 1e-6, np.pi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sin(mid) / mid > ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _arc_points(p_in0, p_in1, p_out0, p_out1, theta, m):
    """Sample the arch's ruling endpoints, inner rim to outer rim.

    The arch bows out of the base plane; the rulings run radially, so
    the patch is a cylinder and stays developable.
    """
    tt = np.linspace(0.0, 1.0, m)
    # a circular profile of half-angle theta, rising over the chord
    ph = (2.0 * tt - 1.0) * theta
    lift = (np.cos(ph) - np.cos(theta)) / max(1.0 - np.cos(theta), 1e-12)
    inner = p_in0[None, :] + tt[:, None] * (p_in1 - p_in0)[None, :]
    outer = p_out0[None, :] + tt[:, None] * (p_out1 - p_out0)[None, :]
    rise = np.zeros((m, 3))
    rise[:, 2] = lift * np.linalg.norm(p_out1 - p_out0) * 0.5
    return 0.5 * (inner + outer) + rise


def _fit(V, scale):
    """Centre and fit a 2*scale cube, as the repo's generators all do."""
    V = np.asarray(V, dtype=float)
    if not len(V):
        return V
    ext = float(np.max(V.max(axis=0) - V.min(axis=0)))
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    return (V - 0.5 * (V.max(axis=0) + V.min(axis=0))) * s


def _defects(V, faces):
    """Angle defect at every vertex of a quad/ngon mesh."""
    defect = np.full(len(V), 2 * np.pi)
    for f in faces:
        k = len(f)
        for i in range(k):
            a = V[f[i]]
            u1 = V[f[(i - 1) % k]] - a
            u2 = V[f[(i + 1) % k]] - a
            n1 = np.linalg.norm(u1)
            n2 = np.linalg.norm(u2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            defect[f[i]] -= np.arccos(np.clip(
                float(np.dot(u1, u2)) / (n1 * n2), -1.0, 1.0))
    return defect


if _IN_BLENDER:

    class MESH_OT_koman_add(bpy.types.Operator):
        """Add one of Ilhan Koman's developable sculptures: a slit sheet
        coiled into a ring, or wound out as a logarithmic spiral"""
        bl_idname = "mesh.koman_add"
        bl_label = "Koman Developable"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Form",
            items=[('SPIRAL', "Spiral Sculpture",
                    "A coiling spine with flat blades springing off it, "
                    "winding outward as a logarithmic spiral -- the "
                    "form of Koman's own metal pieces"),
                   ('RING', "Closed Ring",
                    "The authors' demonstration from a plain rectangle: "
                    "overlapping tongues closing a flat wreath around a "
                    "central hole")],
            default='SPIRAL')

        blades: IntProperty(
            name="Blades", default=80, min=5, max=200,
            description="Segments of the slit ribbon")
        growth: FloatProperty(
            name="Growth", default=0.034, min=-0.12, max=0.12,
            description="Shrink or grow each segment geometrically "
                        "along the ribbon. The radius goes as h/2, so "
                        "this is what opens the coil into a spiral -- "
                        "the parameter Koman himself varied")
        turn_div: IntProperty(
            name="Blades per Turn", default=30, min=6, max=90,
            description="Spiral: how many blades make one full turn")
        skew: FloatProperty(
            name="Skew", default=0.70, min=-1.4, max=1.4,
            description="Swing each blade away from radial")
        lean: FloatProperty(
            name="Lean", default=1.00, min=0.0, max=1.57,
            description="Tilt each blade out of the coil plane")
        blade_len: FloatProperty(
            name="Blade Length", default=1.70, min=0.1, max=4.0,
            description="Relative to the local radius, so the form "
                        "stays self-similar")
        blade_width: FloatProperty(
            name="Blade Width", default=0.60, min=0.05, max=2.0,
            description="Also relative to the local radius")
        twist: FloatProperty(
            name="Blade Twist", default=0.0, min=0.0, max=1.5,
            description="Turn each blade's far edge against its root, "
                        "for the look of the metal pieces. This "
                        "forfeits developability -- a twisted blade is "
                        "a hyperbolic paraboloid and can no longer be "
                        "cut from flat stock")

        close_ring: BoolProperty(
            name="Close Ring", default=True,
            description="Ring: derive the slide so the tongues close "
                        "exactly one turn (a = 2*pi/n)")
        slide: FloatProperty(
            name="Slide d", default=0.10, min=0.005, max=0.5,
            description="Ring: how far each tongue slides under its "
                        "neighbour, when Close Ring is off")
        tilt: FloatProperty(
            name="Tongue Lean", default=0.42, min=0.0, max=1.2,
            description="Ring: how far each tongue leans as it rides "
                        "up on the ones beneath it")
        hole: FloatProperty(
            name="Hole", default=0.45, min=0.05, max=2.0,
            description="Ring: inner radius as a fraction of the "
                        "tongue length")
        samples: IntProperty(
            name="Samples", default=5, min=1, max=24,
            description="Subdivisions along each blade or arch")

        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)
        shade_smooth: BoolProperty(name="Shade Smooth", default=False)
        sharp_edges: BoolProperty(
            name="Sharp Folds", default=True,
            description="Mark the folds sharp (and creased) so smooth "
                        "shading does not round them off")

        def execute(self, context):
            try:
                if self.kind == 'SPIRAL':
                    verts, faces = build_koman_spiral(
                        blades=self.blades, growth=self.growth,
                        turn=2.0 * np.pi / max(6, self.turn_div),
                        skew=self.skew, lean=self.lean,
                        blade_len=self.blade_len,
                        blade_width=self.blade_width, twist=self.twist,
                        samples=self.samples)
                else:
                    verts, faces = build_koman_curl(
                        panels=self.blades,
                        slide=(-1.0 if self.close_ring else self.slide),
                        growth=self.growth, skew=self.skew,
                        hole=self.hole, tilt=self.tilt,
                        arch_samples=max(3, self.samples * 2))
            except Exception as exc:     # noqa: BLE001 - report, not crash
                self.report({'ERROR'}, f"Koman build failed: {exc}")
                return {'CANCELLED'}

            name = ("Koman Spiral" if self.kind == 'SPIRAL'
                    else "Koman Ring")
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(v) for v in _fit(verts, self.scale)], [],
                           [list(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.polygons.foreach_set('use_smooth',
                                    [self.shade_smooth] * len(me.polygons))
            if self.sharp_edges:
                mark_sharp_by_angle(me, 35.0)
            me.update()

            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            if self.kind == 'SPIRAL':
                note = ("  (twisted: not developable)" if self.twist > 0
                        else "  (blades planar)")
                self.report(
                    {'INFO'},
                    f"V={len(me.vertices)} F={len(me.polygons)}  "
                    f"{self.blades} blades, growth {self.growth:.3f}"
                    + note)
            else:
                d = (koman_slide_to_close(self.blades) if self.close_ring
                     else self.slide)
                self.report(
                    {'INFO'},
                    f"V={len(me.vertices)} F={len(me.polygons)}  "
                    f"{self.blades} tongues turning "
                    f"{np.degrees(koman_turn(d, 1.0)):.1f} deg each")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            lay.separator()
            if self.kind == 'SPIRAL':
                lay.prop(self, 'blades')
                lay.prop(self, 'turn_div')
                lay.prop(self, 'growth')
                lay.prop(self, 'skew')
                lay.prop(self, 'lean')
                lay.prop(self, 'blade_len')
                lay.prop(self, 'blade_width')
                lay.prop(self, 'twist')
            else:
                lay.prop(self, 'blades', text="Tongues")
                lay.prop(self, 'close_ring')
                if not self.close_ring:
                    lay.prop(self, 'slide')
                lay.prop(self, 'growth')
                lay.prop(self, 'skew')
                lay.prop(self, 'tilt')
                lay.prop(self, 'hole')
            lay.prop(self, 'samples')
            lay.separator()
            lay.prop(self, 'scale')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'sharp_edges')

    def _menu_func(self, context):
        self.layout.operator("mesh.koman_add", icon='MOD_SCREW')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_koman_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_koman_add)


def _selftest():
    ok = True

    # Eq. 1: the turn per tongue is arctan(2d/h)
    good = abs(koman_turn(0.5, 1.0) - np.arctan(1.0)) < 1e-12
    ok &= good
    print(f"koman: turn a = arctan(2d/h) {'OK' if good else 'FAIL'}")

    # Eq. 2: in the small-angle regime the paper works in, n = 2pi/a
    # tongues close a ring of radius about h/2
    h, d = 1.0, 0.02
    r = (2 * np.pi / koman_turn(d, h)) * d / (2 * np.pi)
    good = abs(r - h / 2) / (h / 2) < 0.01
    ok &= good
    print(f"koman: ring radius ~ h/2 (got {r:.4f} for {h/2:.4f}) "
          f"{'OK' if good else 'FAIL'}")

    # the derived slide must CLOSE the ring for every tongue count.
    # Choosing d by hand almost never does -- 24 tongues at d = 0.06 get
    # only 164 degrees round -- so this keeps the default honest.
    worst = max(abs(koman_turn(koman_slide_to_close(n), 1.0) * n - 2 * np.pi)
                for n in (8, 16, 24, 32, 48, 90))
    good = worst < 1e-9
    ok &= good
    print(f"koman: tongues close a full turn (worst {worst:.2e} rad) "
          f"{'OK' if good else 'FAIL'}")

    # and the ring really is an annulus: a central hole, round, and a
    # flat wreath rather than a tube
    V, F = build_koman_curl(panels=24)
    rr = np.linalg.norm(V[:, :2], axis=1)
    ext = V.max(axis=0) - V.min(axis=0)
    good = (len(F) > 0 and np.all(np.isfinite(V))
            and rr.min() > 0.2 * rr.max()
            and abs(ext[0] - ext[1]) < 0.1 * ext[0]
            and ext[2] < 0.4 * ext[0])
    ok &= good
    print(f"koman: ring is a holed wreath (hole {rr.min()/rr.max():.2f} of "
          f"radius, rise {ext[2]/ext[0]:.2f}) {'OK' if good else 'FAIL'}")

    # THE SPIRAL COMES FROM h, NOT d.  Scaling the slide alone only
    # changes the turn rate and leaves every tongue on the same circle;
    # the paper holds a constant and shrinks the size, which is what
    # walks them outward.  This caught exactly that bug once.
    Vs, _ = build_koman_curl(panels=24, growth=0.05)
    good = (np.all(np.isfinite(Vs))
            and np.linalg.norm(Vs[:, :2], axis=1).max() > 1.02 * rr.max())
    ok &= good
    print(f"koman: growth opens the ring into a spiral "
          f"({rr.max():.2f} -> {np.linalg.norm(Vs[:, :2], axis=1).max():.2f}) "
          f"{'OK' if good else 'FAIL'}")

    def blade_spine_planarity(tw, nb=40, sm=5):
        V, F = build_koman_spiral(blades=nb, samples=sm, twist=tw)

        def worst_of(fs):
            w = 0.0
            for f in fs:
                P = V[list(f)]
                nrm = np.cross(P[1] - P[0], P[2] - P[0])
                ln = float(np.linalg.norm(nrm))
                if ln < 1e-14:
                    continue
                w = max(w, abs(float(np.dot(nrm / ln, P[3] - P[0])))
                        / max(float(np.linalg.norm(P[1] - P[0])), 1e-12))
            return w
        return worst_of(F[:nb * sm]), worst_of(F[nb * sm:])

    flat_b, _ = blade_spine_planarity(0.0)
    twist_b, _ = blade_spine_planarity(0.8)

    # The blades must be EXACTLY planar untwisted: that is the paper's
    # own "blue pieces stay planar", and it is what makes the sculpture
    # cuttable from flat stock.  The spiral look comes from each blade
    # being turned further round the coil, never from bending one.
    good = flat_b < 1e-9
    ok &= good
    print(f"koman: spiral blades exactly planar ({flat_b:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # ... and `twist` really does trade that away, which is why it is
    # off by default; asserting it keeps the docstring honest
    good = twist_b > 1e-2
    ok &= good
    print(f"koman: spiral twist forfeits developability ({twist_b:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # the coil is logarithmic: each blade sits a constant ratio further
    # out than the last (each emits 2*(samples+1) vertices, so stride 4
    # at samples=1 is its root)
    gr = 0.05
    Vr, _ = build_koman_spiral(blades=30, growth=gr, samples=1,
                               blade_len=0.0, blade_width=0.0)
    rad = np.linalg.norm(Vr[::4, :2], axis=1)
    spread = float(np.max(np.abs(rad[1:] / np.maximum(rad[:-1], 1e-15)
                                 - (1.0 + gr))))
    good = spread < 1e-9
    ok &= good
    print(f"koman: spiral is logarithmic (ratio spread {spread:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # and unlike the ring it does NOT close: it keeps winding out
    V2, F2 = build_koman_spiral()
    r2 = np.linalg.norm(V2[:, :2], axis=1)
    good = (len(F2) > 0 and np.all(np.isfinite(V2))
            and r2.max() / max(r2.min(), 1e-12) > 5.0)
    ok &= good
    print(f"koman: spiral winds open (radius x"
          f"{r2.max()/max(r2.min(),1e-12):.0f}) {'OK' if good else 'FAIL'}")

    # the arc solver inverts sin(x)/x, which sets how far each shortened
    # tongue bows
    worst = max(abs(np.sin(_arc_angle(q, 1.0)) / _arc_angle(q, 1.0) - q)
                for q in (0.99, 0.9, 0.7, 0.5))
    good = worst < 1e-9
    ok &= good
    print(f"koman: arc chord/length inversion (worst {worst:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("koman_generator self-test failed")
