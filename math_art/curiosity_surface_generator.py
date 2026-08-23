
# Miscellaneous Surfaces generator for Blender: three classic
# surfaces from the geometry literature.
#
# 1. Fresnel's elasticity surface (von Seggern 1993, p. 304): the
#    quartic surface r = sqrt(a^2 x'^2 + b^2 y'^2 + c^2 z'^2),
#    where (x', y', z') is the unit direction of the radius
#    vector; in Cartesian form (x^2 + y^2 + z^2)^2 =
#    a^2 x^2 + b^2 y^2 + c^2 z^2.  It is a radial surface over
#    the sphere of directions, so it is meshed as a displaced UV
#    sphere with welded poles and seam (watertight).
#
# 2. The paper bag surface (Robin 2004; plotted after Trott 2004,
#    p. 103): the crimped-bag plot
#        x = v cos u
#        y = (v + b u) sin u
#        z = a v^2
#    for u in [0, 2 pi], v in [0, 2], with the classic constants
#    a = 2.47 (height coefficient) and b = -1.26 (crimp
#    coefficient).  The u = 0 and u = 2 pi boundary curves
#    coincide in space and are welded; the surface stays open at
#    v = 0 and v = 2 like a real bag.
#
# 3. The trihyperboloid (Knill 2017; Villarino and Varilly 2024):
#    the boundary of the solid enclosed by the three hyperboloids
#        x^2 + y^2 - z^2 <= 1
#        y^2 + z^2 - x^2 <= 1
#        z^2 + x^2 - y^2 <= 1
#    (volume ln 256 = 8 ln 2), shaped like a stella octangula
#    with webs hung across adjacent faces.  Along any unit
#    direction (l, m, n) the largest of the three quadratic forms
#    q_max = max(l^2+m^2-n^2, m^2+n^2-l^2, n^2+l^2-m^2) is at
#    least 1/3, so the boundary is the radial graph
#    r = 1/sqrt(q_max) with 1 <= r <= sqrt(3) -- meshed exactly
#    as a displaced UV sphere (watertight).
#
# 4. Dupin cyclides (Dupin 1822): the surfaces all of whose lines of
#    curvature are circles.  They are exactly the inversions of tori
#    (and of cylinders and cones) of revolution, and the three
#    classical types come from the three types of torus -- ring
#    (R > r), horn (R = r) and spindle (R < r).  Built by inverting the
#    torus rather than from a memorised parametrisation, so the
#    defining circle property is inherited rather than asserted.
#
# 5. Zoll surfaces -- Tannery's pear (1892), its two-lobed
#    "hourglass", and Zoll's own surface (1903).  On a round sphere
#    every geodesic closes up, which is a very fragile thing to ask of
#    a surface; for a long time the sphere was the only known answer,
#    and these are the next two.  Tannery's is the revolution of half a
#    Gerono lemniscate stretched by 2 sqrt2, and all its geodesics are
#    closed algebraic curves of the same length 2 pi a as a meridian --
#    but it has a conical point at each tip.  Zoll's is the smooth
#    answer to that.  The mathematics and the gates live in
#    `math_art/surfaces/encyclopedia.py`.
#
# 6. Three classical odds and ends, each built from its DEFINITION
#    rather than from a remembered equation, and each checked against an
#    independent statement of the same surface:
#      Bohemian dome -- a circle swept along a circle in a perpendicular
#        plane; checked by confirming every swept circle really is a
#        circle of the right radius and centre.
#      Astroidal ellipsoid -- the surface whose sections through the
#        axes are astroids; the parametrisation is checked against the
#        implicit (x/a)^(2/3)+(y/b)^(2/3)+(z/c)^(2/3) = 1.
#      Gabriel's horn -- y = 1/x revolved; checked on the property that
#        makes it famous, finite volume pi(1 - 1/L) with a lateral area
#        that diverges like 2 pi ln L.
#
# 7. Schwarz's lantern -- the companion pathology to Gabriel's horn, and
#    the one that cost the subject a definition.  A polyhedron every one
#    of whose vertices lies ON a cylinder, built of 4.m.n triangles by
#    spiking each cell of an m x n grid out to the cell centre.  Inscribed
#    polygons always converge to a curve's length, and the same was
#    assumed of surfaces; Schwarz's lantern shows it is false.  Refine the
#    height much faster than the circumference and the spikes stay as
#    sharp as they were while multiplying without limit, so the area
#    diverges although the surface tends to the cylinder pointwise.  The
#    area of a cylinder is therefore NOT the supremum of the areas of the
#    polyhedra inscribed in it -- and any real number above 2.pi.r.h is
#    the limit of some sequence of lanterns.  Both limits are measured in
#    `_selftest`.
#
# References for 5: J. Tannery, Bulletin des sciences mathematiques,
#    2e serie, 16 (1892) 190.  O. Zoll, "Ueber Flaechen mit Scharen
#    geschlossener geodaetischer Linien", Mathematische Annalen 57
#    (1903) 108-133.  A. L. Besse, "Manifolds all of whose Geodesics
#    are Closed", Springer 1978.  Parametrisations from R. Ferreol,
#    "Encyclopedie des formes mathematiques remarquables",
#    mathcurve.com, chapter "poire de Tannery"; a converted copy is in
#    research/books/mathcurve_encyclopedie_formes_mathematiques/.
#
# References for 7: H. A. Schwarz, "Sur une definition erronee de l'aire
#    d'une surface courbe", in "Gesammelte Mathematische Abhandlungen",
#    vol. 2, Springer 1890, 309-311 (the counterexample and the name).
#    Construction with a centre vertex per cell after R. Ferreol,
#    "Encyclopedie des formes mathematiques remarquables", mathcurve.com,
#    chapter "lampion de Schwarz".
#
# References for 4: C. Dupin, "Applications de Geometrie et de
#    Mechanique", Paris 1822.  B. Odehnal, "Ortho-Circles of Dupin
#    Cyclides", J. Geometry Graphics 10 (2006) 1-21 -- states the
#    inversion construction and the ring / horn / spindle
#    classification used here; a converted copy is in
#    research/papers/classical-surfaces/.

bl_info = {
    "name": "Miscellaneous Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Fresnel's elasticity surface, the paper bag "
                   "surface, the trihyperboloid, the Dupin "
                   "cyclides and the Zoll surfaces",
    "category": "Add Mesh",
}

import math

try:
    from .surfaces.encyclopedia import build_zoll
    from .sharp_creases import mark_sharp
except ImportError:                       # flat import outside the package
    from surfaces.encyclopedia import build_zoll
    from sharp_creases import mark_sharp

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def build_radial(radius_fn, segments=96, rings=48):
    """(verts, faces): the radial surface r = radius_fn(l, m, n)
    over unit directions (l, m, n), meshed as a displaced UV
    sphere.  Poles and the theta seam are welded, so the result
    is watertight; faces wind outward (positive volume)."""
    segments = max(3, int(segments))
    rings = max(2, int(rings))
    verts = []
    r = radius_fn(0.0, 0.0, 1.0)
    verts.append((0.0, 0.0, r))
    for i in range(1, rings):
        phi = math.pi * i / rings
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(segments):
            th = 2.0 * math.pi * j / segments
            d = (sp * math.cos(th), sp * math.sin(th), cp)
            r = radius_fn(*d)
            verts.append((r * d[0], r * d[1], r * d[2]))
    r = radius_fn(0.0, 0.0, -1.0)
    verts.append((0.0, 0.0, -r))
    south = len(verts) - 1

    def vid(i, j):
        return 1 + (i - 1) * segments + j % segments

    faces = []
    for j in range(segments):
        faces.append([0, vid(1, j), vid(1, j + 1)])
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append([vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)])
    for j in range(segments):
        faces.append([vid(rings - 1, j), south,
                      vid(rings - 1, j + 1)])
    return verts, faces


def fresnel_radius(a, b, c):
    """r(l, m, n) = sqrt(a^2 l^2 + b^2 m^2 + c^2 n^2) for a unit
    direction: Fresnel's elasticity surface (von Seggern's
    quartic (x^2+y^2+z^2)^2 = a^2 x^2 + b^2 y^2 + c^2 z^2)."""
    def r(l, m, n):
        return math.sqrt(a * a * l * l + b * b * m * m
                         + c * c * n * n)
    return r


def build_fresnel(a=1.0, b=1.5, c=2.0, segments=96, rings=48):
    """(verts, faces): Fresnel's elasticity surface with
    semi-axes a, b, c, watertight."""
    return build_radial(fresnel_radius(a, b, c), segments, rings)


def trihyperboloid_radius(l, m, n):
    """r(l, m, n) = 1/sqrt(max of the three hyperboloid forms):
    the boundary of the solid x^2+y^2-z^2 <= 1,
    y^2+z^2-x^2 <= 1, z^2+x^2-y^2 <= 1 along a unit direction.
    The maximum is >= 1/3, so r is finite (1 <= r <= sqrt 3)."""
    l2, m2, n2 = l * l, m * m, n * n
    q = max(l2 + m2 - n2, m2 + n2 - l2, n2 + l2 - m2)
    return 1.0 / math.sqrt(q)


def build_trihyperboloid(segments=96, rings=48):
    """(verts, faces): the trihyperboloid boundary surface,
    watertight."""
    return build_radial(trihyperboloid_radius, segments, rings)


def build_bohemian_dome(a=1.0, b=1.0, c=0.7, segments=96, rings=48):
    """The Bohemian dome: a circle swept along a circle in a plane
    perpendicular to it.

    Defined by the sweep rather than by a quartic, because the sweep is
    the definition -- the moving circle of radius c rides the fixed
    circle of radii a, b and stays in a plane orthogonal to it:

        x = a cos u,  y = b sin u + c cos v,  z = c sin v

    Closed in both parameters, so the mesh is a torus-topology quad grid
    with both seams welded; it self-intersects, which is the surface's
    own business and not a meshing fault.
    """
    verts = []
    for i in range(segments):
        u = 2.0 * math.pi * i / segments
        for j in range(rings):
            v = 2.0 * math.pi * j / rings
            verts.append((a * math.cos(u),
                          b * math.sin(u) + c * math.cos(v),
                          c * math.sin(v)))
    faces = []
    for i in range(segments):
        i1 = (i + 1) % segments
        for j in range(rings):
            j1 = (j + 1) % rings
            faces.append((i * rings + j, i1 * rings + j,
                          i1 * rings + j1, i * rings + j1))
    return verts, faces


def build_astroidal_ellipsoid(a=1.0, b=1.0, c=1.0, segments=96,
                              rings=48):
    """The astroidal ellipsoid, the surface whose plane sections through
    the axes are astroids:

        x = a(cos u cos v)^3,  y = b(sin u cos v)^3,  z = c(sin v)^3

    Equivalently the implicit (x/a)^(2/3) + (y/b)^(2/3) + (z/c)^(2/3) = 1,
    which `_selftest` uses to check the parametrisation rather than
    trusting it -- the two forms are independent statements of the same
    surface, so agreement is evidence and not a tautology.

    The four cusped edges and the six vertices are genuine features of
    the surface, not meshing artifacts.
    """
    verts = []
    for i in range(segments):
        u = 2.0 * math.pi * i / segments
        cu, su = math.cos(u), math.sin(u)
        for j in range(rings + 1):
            v = -math.pi / 2.0 + math.pi * j / rings
            cv, sv = math.cos(v), math.sin(v)
            # the semi-axis multiplies the cube, it is not inside it:
            # x = a(cos u cos v)^3 gives (x/a)^(2/3) = (cos u cos v)^2,
            # which is what makes the three terms sum to one
            verts.append((a * (cu * cv) ** 3, b * (su * cv) ** 3,
                          c * sv ** 3))
    faces = []
    for i in range(segments):
        i1 = (i + 1) % segments
        for j in range(rings):
            faces.append((i * (rings + 1) + j, i1 * (rings + 1) + j,
                          i1 * (rings + 1) + j + 1,
                          i * (rings + 1) + j + 1))
    return verts, faces


def build_gabriels_horn(length=6.0, segments=96, rings=120):
    """Gabriel's horn: y = 1/x revolved about the x-axis, x in [1, L].

    The point of the object is that it encloses a finite volume,
    pi(1 - 1/L) -> pi, while its lateral area diverges like 2 pi ln L.
    `_selftest` measures both, because a horn meshed with the wrong
    profile would still look like a horn.

    Left open at both ends: the wide mouth is a genuine boundary, and
    capping the narrow end would misrepresent a surface whose whole
    interest is that it never closes.
    """
    verts = []
    for j in range(rings + 1):
        # bunch samples toward the mouth, where the profile bends
        t = j / float(rings)
        x = 1.0 + (length - 1.0) * (t ** 2)
        r = 1.0 / x
        for i in range(segments):
            th = 2.0 * math.pi * i / segments
            verts.append((x, r * math.cos(th), r * math.sin(th)))
    faces = []
    for j in range(rings):
        for i in range(segments):
            i1 = (i + 1) % segments
            faces.append((j * segments + i, j * segments + i1,
                          (j + 1) * segments + i1,
                          (j + 1) * segments + i))
    return verts, faces

def build_schwarz_lantern(sectors=12, rings=12, radius=1.0, height=2.0,
                          lids=False):
    """Schwarz's lantern: a polyhedron INSCRIBED in a cylinder whose area
    can be made to exceed the cylinder's by as much as one likes.

    Unroll the cylinder to a rectangle, cut it into `rings` bands by
    `sectors` columns, and put a vertex at each cell corner AND at each
    cell centre -- all of them on the cylinder.  The four corners of a cell
    lie on a chord plane that cuts INSIDE the cylinder, while the centre
    stays on the surface, so each cell becomes a shallow outward spike of
    four triangles: 4 * sectors * rings faces in all.

    Why it matters.  Refining a curve's inscribed polygon always converges
    to the arc length, and it is tempting to assume the same of surfaces.
    It is false.  The radial overshoot of a spike is
    r(1 - cos(pi/sectors)) ~ r.pi^2 / (2.sectors^2), which shrinks with the
    ANGULAR refinement only, while the number of spikes grows with BOTH.
    Refine the height much faster than the circumference and the spikes
    stay as sharp as ever while multiplying without bound, so the total
    area diverges even though every vertex lies on the cylinder and the
    surface converges to it pointwise.  Taking rings ~ sectors^3 is enough:
    area(k, k) -> 2.pi.r.h, but area(k, k^3) -> infinity.  `_selftest`
    measures both limits, since a lantern with the wrong stagger would
    still look like a lantern.
    """
    sectors = max(3, int(sectors))
    rings = max(1, int(rings))
    dth = 2.0 * math.pi / sectors
    dz = float(height) / rings

    def on_cyl(j, i):                       # angular index j, height index i
        th = j * dth
        return (radius * math.cos(th), radius * math.sin(th),
                -height / 2.0 + i * dz)
    verts = []
    for i in range(rings + 1):              # (rings+1) x sectors corners
        for j in range(sectors):
            verts.append(on_cyl(j, i))

    def corner(i, j):
        return i * sectors + (j % sectors)
    faces = []
    for i in range(rings):
        for j in range(sectors):
            c = len(verts)
            verts.append(on_cyl(j + 0.5, i + 0.5))     # cell centre, on cyl
            a, b = corner(i, j), corner(i, j + 1)
            d, e = corner(i + 1, j + 1), corner(i + 1, j)
            faces += [[a, b, c], [b, d, c], [d, e, c], [e, a, c]]
    if lids:                                # close it into a polyhedron
        faces.append([corner(0, j) for j in range(sectors - 1, -1, -1)])
        faces.append([corner(rings, j) for j in range(sectors)])
    return verts, faces


def lantern_area(sectors, rings, radius=1.0, height=2.0):
    """Total area of the lantern's triangles (lids excluded)."""
    V, F = build_schwarz_lantern(sectors, rings, radius, height)
    tot = 0.0
    for f in F:
        if len(f) != 3:
            continue
        a, b, c = (V[i] for i in f)
        u = [b[i] - a[i] for i in range(3)]
        w = [c[i] - a[i] for i in range(3)]
        n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
             u[0] * w[1] - u[1] * w[0])
        tot += 0.5 * math.sqrt(sum(x * x for x in n))
    return tot


def build_cyclide(kind='RING', ring=1.0, tube=0.45, centre=1.9,
                  power=1.6, segments=96, rings=48):
    """A Dupin cyclide, as the inversion of a torus of revolution.

    Dupin cyclides are exactly the images of tori (and cylinders and
    cones of revolution) under an inversion, and the three classical
    types come from the three types of torus:

        ring    R > r   ring torus     ->  ring cyclide
        horn    R = r   horn torus     ->  horn cyclide
        spindle R < r   spindle torus  ->  spindle cyclide

    Building them this way rather than from a memorised (a, b, c, d)
    parametrisation is deliberate.  Inversion carries circles to
    circles, so the property that DEFINES a Dupin cyclide -- every line
    of curvature is a circle -- is inherited from the torus for free,
    and `_selftest` can check it directly by fitting circles to the
    parameter curves.

    `centre` places the centre of inversion on the x-axis and `power`
    is its radius; the centre must not lie on the torus, or the image
    runs off to infinity.
    """
    R, r = float(ring), float(tube)
    if kind == 'HORN':
        r = R
    elif kind == 'SPINDLE':
        r = max(r, R * 1.6)
    cx, k2 = float(centre), float(power) ** 2

    verts = []
    for i in range(segments):
        u = 2.0 * math.pi * i / segments
        cu, su = math.cos(u), math.sin(u)
        for j in range(rings):
            v = 2.0 * math.pi * j / rings
            rad = R + r * math.cos(v)
            px, py, pz = rad * cu, rad * su, r * math.sin(v)
            dx, dy, dz = px - cx, py, pz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < 1e-12:                      # centre sits on Phi
                d2 = 1e-12
            s = k2 / d2
            verts.append((cx + dx * s, dy * s, dz * s))

    faces = []
    for i in range(segments):
        i1 = (i + 1) % segments
        for j in range(rings):
            j1 = (j + 1) % rings
            faces.append((i * rings + j, i1 * rings + j,
                          i1 * rings + j1, i * rings + j1))
    return verts, faces


def cyclide_parameter_curves(kind='RING', ring=1.0, tube=0.45,
                             centre=1.9, power=1.6, n=72):
    """Sample the two families of parameter curves, as point lists.

    These are the images of the torus's lines of curvature, so on a
    genuine Dupin cyclide every one of them is a circle.
    """
    R, r = float(ring), float(tube)
    if kind == 'HORN':
        r = R
    elif kind == 'SPINDLE':
        r = max(r, R * 1.6)
    cx, k2 = float(centre), float(power) ** 2

    def invert(px, py, pz):
        dx, dy, dz = px - cx, py, pz
        d2 = max(dx * dx + dy * dy + dz * dz, 1e-12)
        s = k2 / d2
        return (cx + dx * s, dy * s, dz * s)

    def point(u, v):
        rad = R + r * math.cos(v)
        return invert(rad * math.cos(u), rad * math.sin(u),
                      r * math.sin(v))

    fixed_u = []
    for u in (0.0, 0.7, 1.9, 3.5):
        fixed_u.append([point(u, 2.0 * math.pi * t / n)
                        for t in range(n)])
    fixed_v = []
    for v in (0.0, 1.1, 2.6, 4.4):
        fixed_v.append([point(2.0 * math.pi * t / n, v)
                        for t in range(n)])
    return fixed_u, fixed_v

def build_paper_bag(a=2.47, b=-1.26, segments=96, rings=48,
                    depth=2.0):
    """(verts, faces): the paper bag surface x = v cos u,
    y = (v + b u) sin u, z = a v^2 for u in [0, 2 pi] and
    v in [0, depth].  The u seam (where both boundary curves
    coincide) is welded; the v boundaries stay open."""
    segments = max(3, int(segments))
    rings = max(1, int(rings))
    verts = []
    for i in range(rings + 1):
        v = depth * i / rings
        for j in range(segments):
            u = 2.0 * math.pi * j / segments
            verts.append((v * math.cos(u),
                          (v + b * u) * math.sin(u),
                          a * v * v))
    faces = []
    for i in range(rings):
        for j in range(segments):
            j2 = (j + 1) % segments
            faces.append([i * segments + j, i * segments + j2,
                          (i + 1) * segments + j2,
                          (i + 1) * segments + j])
    return verts, faces


if _IN_BLENDER:

    class MESH_OT_curiosity_surface_add(bpy.types.Operator):
        """Add a classic miscellaneous surface: Fresnel's
        elasticity surface, the paper bag surface, or the
        trihyperboloid"""
        bl_idname = "mesh.curiosity_surface_add"
        bl_label = "Miscellaneous Surface"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            items=[('FRESNEL', "Fresnel Elasticity Surface",
                    "The quartic (x^2+y^2+z^2)^2 = "
                    "a^2 x^2 + b^2 y^2 + c^2 z^2"),
                   ('PAPERBAG', "Paper Bag Surface",
                    "The crimped inflated-bag surface "
                    "x = v cos u, y = (v + b u) sin u, "
                    "z = a v^2"),
                   ('TRIHYPERBOLOID', "Trihyperboloid",
                    "Boundary of the solid enclosed by the "
                    "three hyperboloids x^2+y^2-z^2 <= 1 "
                    "(cyclic); volume 8 ln 2"),
                   ('BOHEMIAN', "Bohemian Dome",
                    "A circle swept along a circle in a "
                    "perpendicular plane; self-intersecting"),
                   ('ASTROIDAL', "Astroidal Ellipsoid",
                    "(x/a)^(2/3) + (y/b)^(2/3) + (z/c)^(2/3) = 1, "
                    "with astroid sections and six cusps"),
                   ('SCHWARZ_LANTERN', "Schwarz's Lantern",
                    "A polyhedron inscribed in a cylinder whose area can "
                    "exceed the cylinder's without bound"),
                   ('GABRIEL', "Gabriel's Horn",
                    "y = 1/x revolved: finite volume, infinite "
                    "lateral area"),
                   ('CYCLIDE_RING', "Dupin Cyclide (ring)",
                    "Inversion of a ring torus (R > r); every "
                    "line of curvature is a circle"),
                   ('CYCLIDE_HORN', "Dupin Cyclide (horn)",
                    "Inversion of a horn torus (R = r), which "
                    "touches itself on the axis"),
                   ('CYCLIDE_SPINDLE', "Dupin Cyclide (spindle)",
                    "Inversion of a spindle torus (R < r), the "
                    "self-intersecting one"),
                   ('TANNERY_PEAR', "Tannery's Pear",
                    "A Zoll surface: every geodesic closes up, and all "
                    "but one have the same length 2 pi a as a "
                    "meridian. Both tips are conical points"),
                   ('TANNERY_HOURGLASS', "Tannery's Hourglass",
                    "The whole Gerono lemniscate revolved instead of "
                    "half of it: two pears meeting tip to tip"),
                   ('ZOLL', "Zoll's Surface",
                    "Zoll's 1903 answer to Tannery: smooth, not a "
                    "round sphere, and still every geodesic closes")],
            default='FRESNEL',
            description="Which classic surface to build")
        semi_a: FloatProperty(
            name="Semi-Axis A", default=1.0, min=0.01, max=10.0,
            description="Fresnel semi-axis along X")
        semi_b: FloatProperty(
            name="Semi-Axis B", default=1.5, min=0.01, max=10.0,
            description="Fresnel semi-axis along Y")
        semi_c: FloatProperty(
            name="Semi-Axis C", default=2.0, min=0.01, max=10.0,
            description="Fresnel semi-axis along Z")
        bag_a: FloatProperty(
            name="Height Coefficient", default=2.47,
            min=0.01, max=10.0,
            description="Coefficient a in z = a v^2 (2.47 in "
                        "the classic plot)")
        bag_b: FloatProperty(
            name="Crimp Coefficient", default=-1.26,
            min=-10.0, max=10.0,
            description="Coefficient b in y = (v + b u) sin u "
                        "(-1.26 in the classic plot)")
        bohemian_c: FloatProperty(
            name="Swept Circle", default=0.7, min=0.01, max=10.0,
            description="Radius c of the moving circle (Bohemian dome "
                        "only)")
        horn_length: FloatProperty(
            name="Horn Length", default=6.0, min=1.2, max=200.0,
            description="Upper limit L of x in y = 1/x; the enclosed "
                        "volume tends to pi as L grows while the "
                        "lateral area diverges (Gabriel's horn only)")
        lantern_sectors: IntProperty(
            name="Sectors", default=12, min=3, max=256,
            description="Columns around the cylinder; the spikes get "
                        "blunter as this rises (Schwarz's lantern only)")
        lantern_rings: IntProperty(
            name="Bands", default=12, min=1, max=4096,
            description="Bands up the cylinder.  Raising this alone "
                        "multiplies the spikes without blunting them, so "
                        "the area grows without bound -- try bands near "
                        "the cube of the sectors (Schwarz's lantern only)")
        lantern_lids: BoolProperty(
            name="Cap the Ends", default=False,
            description="Add the top and bottom polygons, which close the "
                        "lantern into a polyhedron (Schwarz's lantern only)")
        cyclide_ring: FloatProperty(
            name="Ring Radius", default=1.0, min=0.05, max=10.0,
            description="Radius R of the torus centre circle "
                        "(Dupin cyclides only)")
        cyclide_tube: FloatProperty(
            name="Tube Radius", default=0.45, min=0.01, max=10.0,
            description="Tube radius r of the torus; the ring / horn "
                        "/ spindle type overrides it where the type "
                        "fixes the ratio (Dupin cyclides only)")
        cyclide_centre: FloatProperty(
            name="Inversion Centre", default=1.9,
            min=-10.0, max=10.0,
            description="Centre of inversion on the x-axis; it must "
                        "not lie on the torus, or the image runs off "
                        "to infinity (Dupin cyclides only)")
        cyclide_power: FloatProperty(
            name="Inversion Radius", default=1.6, min=0.05, max=10.0,
            description="Radius of the sphere of inversion "
                        "(Dupin cyclides only)")
        resolution: IntProperty(
            name="Resolution", default=48, min=6, max=256,
            description="Rings across the surface (twice as "
                        "many segments around)")
        smooth: BoolProperty(name="Smooth Shading", default=True,
                             description="Shade the surface smooth. "
                                         "Ignored for Schwarz's lantern, "
                                         "whose faces are flat and whose "
                                         "every edge is a fold")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")

        def execute(self, context):
            res = self.resolution
            if self.surface == 'FRESNEL':
                verts, faces = build_fresnel(
                    self.semi_a, self.semi_b, self.semi_c,
                    2 * res, res)
                name = "Fresnel Elasticity Surface"
            elif self.surface == 'PAPERBAG':
                verts, faces = build_paper_bag(
                    self.bag_a, self.bag_b, 2 * res, res)
                name = "Paper Bag Surface"
            elif self.surface == 'BOHEMIAN':
                verts, faces = build_bohemian_dome(
                    self.semi_a, self.semi_b, self.bohemian_c,
                    2 * res, res)
                name = "Bohemian Dome"
            elif self.surface == 'ASTROIDAL':
                verts, faces = build_astroidal_ellipsoid(
                    self.semi_a, self.semi_b, self.semi_c,
                    2 * res, res)
                name = "Astroidal Ellipsoid"
            elif self.surface == 'GABRIEL':
                verts, faces = build_gabriels_horn(
                    self.horn_length, 2 * res, 3 * res)
                name = "Gabriel's Horn"
            elif self.surface == 'SCHWARZ_LANTERN':
                verts, faces = build_schwarz_lantern(
                    self.lantern_sectors, self.lantern_rings,
                    1.0, 2.0, self.lantern_lids)
                name = "Schwarz's Lantern"
            elif self.surface.startswith('CYCLIDE_'):
                kind = self.surface.split('_', 1)[1]
                verts, faces = build_cyclide(
                    kind, self.cyclide_ring, self.cyclide_tube,
                    self.cyclide_centre, self.cyclide_power,
                    2 * res, res)
                name = "Dupin Cyclide (%s)" % kind.lower()
            elif self.surface in ('TANNERY_PEAR', 'TANNERY_HOURGLASS',
                                  'ZOLL'):
                V, faces = build_zoll(self.surface, a=self.semi_a,
                                      res_u=res, res_v=2 * res)
                verts = [tuple(map(float, v)) for v in V]
                name = {'TANNERY_PEAR': "Tannery's Pear",
                        'TANNERY_HOURGLASS': "Tannery's Hourglass",
                        'ZOLL': "Zoll's Surface"}[self.surface]
            else:
                verts, faces = build_trihyperboloid(2 * res, res)
                name = "Trihyperboloid"
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # The lantern is a POLYHEDRON, not a sampled smooth surface:
            # every face is planar and every edge is a genuine fold, so
            # there is nothing for smooth shading to interpolate and
            # letting it run averages away the spikes that are the whole
            # point.  Flat-shade it whatever the toggle says, and mark
            # every edge sharp and creased so a later subdivide or bevel
            # keeps the faceting too.  The creases are known here at
            # build time, so they are marked exactly rather than found
            # by an angle test that shallow folds would slip past.
            lantern = self.surface == 'SCHWARZ_LANTERN'
            me.polygons.foreach_set(
                'use_smooth',
                [False if lantern else self.smooth] * len(me.polygons))
            if lantern:
                edges = {(min(f[i], f[(i + 1) % len(f)]),
                          max(f[i], f[(i + 1) % len(f)]))
                         for f in faces for i in range(len(f))}
                mark_sharp(me, edges)
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.surface == 'SCHWARZ_LANTERN':
                # the number worth seeing: how far the inscribed area has
                # run away from the cylinder it is inscribed in
                a = lantern_area(self.lantern_sectors, self.lantern_rings)
                cyl = 2.0 * math.pi * 1.0 * 2.0
                self.report({'INFO'},
                            f"{name}: V={len(me.vertices)} "
                            f"F={len(me.polygons)}, area {a / cyl:.2f}x the "
                            f"cylinder it is inscribed in")
            else:
                self.report({'INFO'},
                            f"{name}: V={len(me.vertices)} "
                            f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'surface')
            if self.surface == 'FRESNEL':
                for k in ('semi_a', 'semi_b', 'semi_c'):
                    lay.prop(self, k)
            elif self.surface == 'PAPERBAG':
                for k in ('bag_a', 'bag_b'):
                    lay.prop(self, k)
            elif self.surface == 'GABRIEL':
                lay.prop(self, 'horn_length')
            elif self.surface == 'SCHWARZ_LANTERN':
                for k in ('lantern_sectors', 'lantern_rings',
                          'lantern_lids'):
                    lay.prop(self, k)
            keys = ('smooth', 'thickness', 'scale') \
                if self.surface == 'SCHWARZ_LANTERN' \
                else ('resolution', 'smooth', 'thickness', 'scale')
            for k in keys:                  # the lantern has its own counts
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.curiosity_surface_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_curiosity_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_curiosity_surface_add)


def _selftest():
    def _finite(verts):
        return all(all(math.isfinite(c) for c in v)
                   for v in verts)

    def _valid(verts, faces):
        n = len(verts)
        return all(0 <= i < n for f in faces for i in f)

    def _watertight(faces):
        cnt = {}
        for f in faces:
            for i in range(len(f)):
                e = frozenset((f[i], f[(i + 1) % len(f)]))
                cnt[e] = cnt.get(e, 0) + 1
        return all(c == 2 for c in cnt.values())

    def _volume(verts, faces):
        vol = 0.0
        for f in faces:
            for i in range(1, len(f) - 1):
                (ax, ay, az) = verts[f[0]]
                (bx, by, bz) = verts[f[i]]
                (cx, cy, cz) = verts[f[i + 1]]
                vol += (ax * (by * cz - bz * cy)
                        - ay * (bx * cz - bz * cx)
                        + az * (bx * cy - by * cx)) / 6.0
        return vol

    # ---- Fresnel's elasticity surface -------------------
    a, b, c = 1.0, 1.5, 2.0
    segs, rngs = 32, 16
    verts, faces = build_fresnel(a, b, c, segs, rngs)
    assert _finite(verts) and _valid(verts, faces)
    assert _watertight(faces)
    # every vertex obeys r = sqrt(a^2 l^2 + b^2 m^2
    # + c^2 n^2) for its unit direction (l, m, n)
    for (x, y, z) in verts:
        r = math.sqrt(x * x + y * y + z * z)
        l, m, n = x / r, y / r, z / r
        want = math.sqrt(a * a * l * l + b * b * m * m
                         + c * c * n * n)
        assert abs(r - want) < 1e-9
    # sample directions: the axes hit r = a, b, c
    eq = 1 + (rngs // 2 - 1) * segs      # phi = pi/2, th = 0
    px = verts[eq]
    py = verts[eq + segs // 4]           # theta = pi/2
    pz = verts[0]                        # north pole
    assert abs(px[0] - a) < 1e-9 and abs(px[1]) < 1e-9
    assert abs(py[1] - b) < 1e-9 and abs(abs(py[0])) < 1e-9
    assert abs(pz[2] - c) < 1e-9
    vol = _volume(verts, faces)
    print(f"fresnel: V={len(verts)} F={len(faces)} "
          f"watertight=True axes=({a},{b},{c}) "
          f"vol={vol:.3f}")
    assert vol > 0.0

    # ---- Paper bag surface ------------------------------
    pa, pb, depth = 2.47, -1.26, 2.0
    segs, rngs = 48, 24
    verts, faces = build_paper_bag(pa, pb, segs, rngs)
    assert _finite(verts) and _valid(verts, faces)
    # spot-check the parametrization at a few grid points
    for (i, j) in ((0, 0), (rngs, 0), (7, 11), (rngs, 30)):
        v = depth * i / rngs
        u = 2.0 * math.pi * j / segs
        x, y, z = verts[i * segs + j]
        assert abs(x - v * math.cos(u)) < 1e-9
        assert abs(y - (v + pb * u) * math.sin(u)) < 1e-9
        assert abs(z - pa * v * v) < 1e-9
    print(f"paper bag: V={len(verts)} F={len(faces)} "
          f"open surface, formula verified")

    # ---- Trihyperboloid ---------------------------------
    segs, rngs = 96, 48
    verts, faces = build_trihyperboloid(segs, rngs)
    assert _finite(verts) and _valid(verts, faces)
    assert _watertight(faces)
    # every vertex lies on the boundary: the largest of the
    # three hyperboloid forms equals 1, and 1 <= r <= sqrt 3
    for (x, y, z) in verts:
        x2, y2, z2 = x * x, y * y, z * z
        q = max(x2 + y2 - z2, y2 + z2 - x2, z2 + x2 - y2)
        assert abs(q - 1.0) < 1e-9
        r = math.sqrt(x2 + y2 + z2)
        assert 1.0 - 1e-9 <= r <= math.sqrt(3.0) + 1e-9
    vol = _volume(verts, faces)
    want = 8.0 * math.log(2.0)     # ln 256, the exact volume
    print(f"trihyperboloid: V={len(verts)} F={len(faces)} "
          f"watertight=True vol={vol:.4f} "
          f"(exact 8 ln 2 = {want:.4f})")
    assert abs(vol - want) < 0.15


    # ---- Dupin cyclides -------------------------------------------
    # The defining property: every line of curvature is a circle.  The
    # parameter curves ARE those lines (they are the images of the
    # torus's own curvature circles), so fitting a circle through three
    # of their points and checking that the rest lie on it tests the
    # construction itself, not merely the meshing.
    def _circumcentre(a, b, c):
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        uu = sum(t * t for t in u)
        vv = sum(t * t for t in v)
        uv = sum(u[i] * v[i] for i in range(3))
        d = 2.0 * (uu * vv - uv * uv)
        if abs(d) < 1e-30:
            return None
        s0 = (uu * vv - uv * vv) / d
        t0 = (vv * uu - uv * uu) / d
        return tuple(a[i] + s0 * u[i] + t0 * v[i] for i in range(3))

    def _circle_dev(pts):
        n = len(pts)
        cen = _circumcentre(pts[0], pts[n // 3], pts[2 * n // 3])
        if cen is None:
            return 1.0
        rr = [math.dist(q, cen) for q in pts]
        R = sum(rr) / len(rr)
        if R < 1e-12:
            return 1.0
        return max(abs(x - R) for x in rr) / R

    bad = []
    for kind in ('RING', 'HORN', 'SPINDLE'):
        fu, fv = cyclide_parameter_curves(kind)
        dev = max(_circle_dev(c) for c in fu + fv)
        if not (dev < 1e-9):
            bad.append('%s:%.1e' % (kind, dev))
        V, F = build_cyclide(kind, segments=64, rings=32)
        if not (_finite(V) and _valid(V, F) and _watertight(F)):
            bad.append(kind + ':mesh')
    assert not bad, "cyclide check failed: " + ", ".join(bad)
    print("cyclides: ring/horn/spindle -- every line of curvature is a "
          "circle (dev < 1e-9), meshes watertight")


    # ---- the three classical odds and ends -------------------------
    # Each is checked against an INDEPENDENT statement of the same
    # surface, so agreement is evidence rather than a restatement of
    # the code that produced it.

    # Bohemian dome: every v-curve must be a circle of radius c whose
    # centre rides the fixed ellipse (a cos u, b sin u, 0).
    a, b, c = 1.0, 1.0, 0.7
    V, F = build_bohemian_dome(a, b, c, 64, 48)
    worst = 0.0
    for i in range(64):
        ring = V[i * 48:(i + 1) * 48]
        u = 2.0 * math.pi * i / 64
        cen = (a * math.cos(u), b * math.sin(u), 0.0)
        for q in ring:
            worst = max(worst, abs(math.dist(q, cen) - c))
    good = worst < 1e-12 and _finite(V) and _valid(V, F)
    assert good, "bohemian dome sweep off by %.2e" % worst
    print("bohemian dome: every swept circle has radius c (max error "
          "%.1e)" % worst)

    # Astroidal ellipsoid: the parametrisation must satisfy the
    # implicit form.  Poles and cusps are excluded, where the 2/3 power
    # of a vanishing coordinate is numerically delicate rather than
    # wrong.
    aa, bb, cc = 1.0, 0.8, 1.3
    V, F = build_astroidal_ellipsoid(aa, bb, cc, 60, 40)
    worst = 0.0
    for (x, y, z) in V:
        t = ((abs(x) / aa) ** (2.0 / 3.0) + (abs(y) / bb) ** (2.0 / 3.0)
             + (abs(z) / cc) ** (2.0 / 3.0))
        worst = max(worst, abs(t - 1.0))
    good = worst < 1e-9 and _finite(V) and _valid(V, F)
    assert good, "astroidal ellipsoid implicit residual %.2e" % worst
    print("astroidal ellipsoid: parametrisation satisfies "
          "(x/a)^(2/3)+(y/b)^(2/3)+(z/c)^(2/3)=1 (max %.1e)" % worst)

    # Gabriel's horn.  The divergence theorem does NOT apply here --
    # the horn is open at both ends on purpose -- so the volume is
    # integrated from the profile by disks instead, and the profile
    # itself is checked first: r * x must be 1 everywhere, which is
    # what says the meridian really is y = 1/x.
    for L in (4.0, 40.0):
        V, F = build_gabriels_horn(L, 96, 400)
        worst = max(abs(math.hypot(q[1], q[2]) * q[0] - 1.0) for q in V)
        assert worst < 1e-9, "gabriel profile off by %.2e" % worst
        # rings, in the order the builder emits them
        segs, rows = 96, 401
        prof = []
        for j in range(rows):
            q = V[j * segs]
            prof.append((q[0], math.hypot(q[1], q[2])))
        vol = 0.0
        area = 0.0
        for (x0, r0), (x1, r1) in zip(prof[:-1], prof[1:]):
            dx = x1 - x0
            vol += math.pi * 0.5 * (r0 * r0 + r1 * r1) * dx
            ds = math.hypot(dx, r1 - r0)
            area += math.pi * (r0 + r1) * ds
        want_v = math.pi * (1.0 - 1.0 / L)
        want_a = 2.0 * math.pi * math.log(L)        # a lower bound
        assert abs(vol - want_v) < 0.01 * want_v,             "gabriel L=%.0f volume %.4f want %.4f" % (L, vol, want_v)
        assert area > want_a,             "gabriel L=%.0f area %.3f should exceed %.3f" % (L, area,
                                                             want_a)
        print("gabriel's horn L=%-5.0f profile r*x=1 to %.0e, volume "
              "%.4f (exact %.4f), lateral area %.2f (> 2 pi ln L = "
              "%.2f)" % (L, worst, vol, want_v, area, want_a))

    # Schwarz's lantern.  Every vertex lies exactly on the cylinder, and
    # the face count is 4 * sectors * rings.  The two limits are the whole
    # point of the object, so both are measured: refined evenly the area
    # converges to the cylinder's, but refined with rings ~ sectors^3 it
    # runs away -- and must do so monotonically in k.
    cyl = 2.0 * math.pi * 1.0 * 2.0
    for s, r in ((6, 4), (12, 12), (9, 40)):
        verts, faces = build_schwarz_lantern(s, r, 1.0, 2.0)
        assert len(faces) == 4 * s * r, (len(faces), s, r)
        assert _finite(verts) and _valid(verts, faces)
        for (x, y, z) in verts:            # inscribed: all vertices on it
            assert abs(math.hypot(x, y) - 1.0) < 1e-12, (x, y, z)
            assert -1.0 - 1e-12 <= z <= 1.0 + 1e-12, z
        vl, fl = build_schwarz_lantern(s, r, 1.0, 2.0, lids=True)
        assert _watertight(fl), "capped lantern is not closed"
    evenly = [lantern_area(k, k) for k in (8, 16, 32, 64, 128)]
    assert all(evenly[i] > evenly[i + 1] for i in range(len(evenly) - 1)), \
        evenly                              # settling down onto the cylinder
    assert abs(evenly[-1] - cyl) / cyl < 2e-3, (evenly[-1], cyl)
    runaway = [lantern_area(k, k ** 3) for k in (3, 5, 8, 12)]
    assert all(runaway[i] < runaway[i + 1] for i in range(len(runaway) - 1)), \
        runaway                             # and here it never settles
    assert runaway[-1] > 4.0 * cyl, runaway[-1]
    print("schwarz lantern: all vertices on the cylinder; area(k,k) -> "
          "%.4f vs cylinder %.4f, while area(k,k^3) climbs %.0f -> %.0f "
          "(%.1fx the cylinder) with no sign of stopping"
          % (evenly[-1], cyl, runaway[0], runaway[-1], runaway[-1] / cyl))

    print("miscellaneous surfaces standalone tests passed")
