
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
# 8. Three surfaces of revolution defined by a differential condition
#    rather than by a formula, each gated on that condition:
#      Bouguer dome -- a^2 y'' = x sqrt(1 + y'^2), the shape a masonry
#        dome must take to stand by compression alone.  It is the
#        catenary's equation (a y'' = sqrt(1+y'^2)) with an extra factor
#        of x: what the third dimension costs.
#      Hanging drop -- 2H = z / a^2, mean curvature proportional to
#        height, which is Young-Laplace with the hydrostatic head.  This
#        is the classical closed profile; for the same physics solved
#        variationally, with a contact angle and a free contact line,
#        see cmc_generator's sessile drop.
#      Neiloid -- a rho^2 = z^3, the revolution of Neile's semicubical
#        parabola, checked against its closed-form volume.
#
# 9. Two Ricci-flat geometries, sliced.  Yau's theorem gives a
#    Ricci-flat Kaehler metric on every compact Calabi-Yau and no
#    formula for it; the non-compact ones are the exception, and these
#    are the two whose metrics are known in closed form.  Both are
#    surfaces of revolution, and both are EXACT: the geometry built is
#    isometric to the slice, not a sketch of it.
#      Eguchi-Hanson space -- the first asymptotically locally
#        Euclidean Ricci-flat Kaehler metric, on T*S^2, the resolution
#        of C^2/Z2:
#            ds^2 = (1 - (a/r)^4)^-1 dr^2
#                   + (r^2/4)[sx^2 + sy^2 + (1 - (a/r)^4) sz^2].
#        Freezing the two-sphere directions leaves the (r, psi) slice,
#        which embeds isometrically as the revolution of
#            rho(r) = sqrt(r^4 - a^4)/(2r),
#            dz/dr  = sqrt(3 r^4 + a^4)/(2 r^2),
#        because rho'^2 + z'^2 collapses to 4r^8/(4r^4(r^4 - a^4)),
#        which is exactly g_rr.  The point is the tip: drho/ds ->
#        (r^4 + a^4)/(2r^4) -> 1 as r -> a, so the circle closes off
#        smoothly -- but only if psi runs over 2 pi.  Eguchi and
#        Hanson's psi runs to 4 pi on S^3, and the metric is smooth
#        only after halving that range, which leaves RP(3) = SO(3) at
#        infinity.  Keep the full range and the tip is a cone of angle
#        4 pi, which no surface in R^3 can carry: the Z2 quotient is
#        the difference between a smooth cap and an impossible one.
#        The two-sphere the space is named for -- the "bolt" -- sits at
#        the tip with radius a/4.
#      The conifold -- the quadric cone sum (w^A)^2 = 0 in C^4, six
#        real dimensions, a cone over S^2 x S^3, whose node can be
#        repaired either by deformation (replacing it with an S^3) or
#        by a small resolution (an S^2), passing between two
#        topologically different Calabi-Yaus.  None of that fits in
#        R^3.  What does, and is the same story one complex dimension
#        down, is the A_1 surface x y = z^2, whose real points
#        u^2 - v^2 - z^2 = delta give the double cone at delta = 0, a
#        throat of radius sqrt(-delta) below it -- the real slice of
#        the sphere that replaces the node -- and two separated sheets
#        above.  Eguchi-Hanson space is the resolution of exactly this
#        singularity, so the two entries are one geometry told
#        metrically and algebraically.
#
# References for 9: T. Eguchi and A. J. Hanson, "Asymptotically flat
#   self-dual solutions to Euclidean gravity", Physics Letters 74B
#   (1978) 249-251 -- the metric (their solution II, Eqs. (9) and
#   (17)), and the "corner" footnote on the half-range of S^3;
#   T. Eguchi, P. B. Gilkey and A. J. Hanson, "Gravitation, gauge
#   theories and differential geometry", Physics Reports 66 (1980)
#   213-393 -- the review, including the T*S^2 topology;
#   P. Candelas and X. C. de la Ossa, "Comments on conifolds", Nuclear
#   Physics B342 (1990) 246-268 -- the conifold, its deformation
#   Eq. (1.2), its small resolution Eq. (1.4) and the Ricci-flat
#   metrics on both;  E. Brieskorn, "Ueber die Aufloesung gewisser
#   Singularitaeten von holomorphen Abbildungen", Mathematische
#   Annalen 166 (1966) 76-102 -- the A_1 singularity C^2/Z2 and its
#   resolution.
#
# References for 8: Pierre Bouguer (1734), studied by Charles Bossut
#    (1778); see E. Benvenuto, "An Introduction to the History of
#    Structural Mechanics", part II, 344-348.  William Neile (1637-1670)
#    for the semicubical parabola.  Both, and the hanging drop, from
#    R. Ferreol, "Encyclopedie des formes mathematiques remarquables"
#    (mathcurve.com), chapters "dome de Bouguer", "goutte d'eau" and
#    "neiloide".
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
#
# References:
# - D. von Seggern, "CRC Standard Curves and Surfaces" (1993), p. 304 --
#   Fresnel's elasticity surface.
# - A. Robin (2004), plotted after M. Trott, "The Mathematica
#   GuideBook for Graphics" (2004), p. 103 -- the paper bag surface.
# - O. Knill (2017); M. A. Villarino and J. C. Varilly (2024) -- the
#   trihyperboloid and its volume ln 256.
# - C. Dupin, "Applications de Geometrie et de Mechanique", Paris 1822;
#   B. Odehnal, "Ortho-Circles of Dupin Cyclides", J. Geometry Graphics
#   10 (2006) 1-21 -- the inversion construction and the ring / horn /
#   spindle classification.
# - J. Tannery, Bulletin des sciences mathematiques, 2e serie, 16 (1892)
#   190; O. Zoll, "Ueber Flaechen mit Scharen geschlossener
#   geodaetischer Linien", Math. Ann. 57 (1903) 108-133; A. L. Besse,
#   "Manifolds all of whose Geodesics are Closed", Springer 1978.
# - H. A. Schwarz, "Sur une definition erronee de l'aire d'une surface
#   courbe", Gesammelte Mathematische Abhandlungen vol. 2, Springer 1890,
#   309-311 -- the lantern counterexample and the name.
# - P. Bouguer (1734), studied by C. Bossut (1778); see E. Benvenuto,
#   "An Introduction to the History of Structural Mechanics", part II,
#   344-348 -- the masonry dome.
# - W. Neile (1637-1670) for the semicubical parabola whose revolution
#   is the neiloid.
# - Torus of revolution: classical (a circle revolved about an axis in
#   its plane); R. Ferreol, "Encyclopedie des formes mathematiques
#   remarquables" (mathcurve.com), chapter "tore".
# - Revolution of the catenary (alysseid), z = a cosh(rho/a) about the
#   catenary's axis of symmetry -- the catenoid's companion, and not
#   minimal: R. Ferreol, ibid., chapter "alysseide".
# - Revolution of the sinusoid, x = a cos(z/b) about Oz -- a string of
#   onion-dome beads meeting in cusp circles on the axis: R. Ferreol,
#   ibid., chapter "surface de revolution de la sinusoide" (the chapter
#   dates its study to 2012).
# - Second tractroid: the tractrix revolved about the axis
#   PERPENDICULAR to its asymptote (the pseudosphere revolves it about
#   the asymptote itself); proposed by Ludovic Schwob per R. Ferreol,
#   ibid., chapter "tractroide".  Not of constant curvature; its
#   volume involves Catalan's constant.
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), chapters "dome de Bouguer", "goutte d'eau",
#   "neiloide", "poire de Tannery" and "lampion de Schwarz" -- the
#   parametrisations checked against.

bl_info = {
    "name": "Miscellaneous Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Fresnel's elasticity surface, the paper bag "
                   "surface, the trihyperboloid, the Dupin "
                   "cyclides, the Zoll surfaces, and slices of the "
                   "Eguchi-Hanson and conifold metrics",
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

def build_torus(ring=1.0, tube=0.35, segments=96, rings=48):
    """The torus of revolution: the circle of radius `tube`, centred
    `ring` from the axis, revolved about Oz.  Both parameter seams are
    welded by index, so the mesh is watertight with chi = 0.  Every
    vertex satisfies (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2 exactly, which
    is what `_selftest` gates on."""
    m = max(3, int(segments))
    n = max(3, int(rings))
    verts = []
    for i in range(m):
        u = 2.0 * math.pi * i / m
        cu, su = math.cos(u), math.sin(u)
        for j in range(n):
            v = 2.0 * math.pi * j / n
            w = ring + tube * math.cos(v)
            verts.append((w * cu, w * su, tube * math.sin(v)))
    faces = []
    for i in range(m):
        i1 = (i + 1) % m
        for j in range(n):
            j1 = (j + 1) % n
            faces.append((i * n + j, i1 * n + j, i1 * n + j1,
                          i * n + j1))
    return verts, faces


def alysseid_profile(a=1.0, extent=1.6, steps=96):
    """[(rho, z)] of the ALYSSEID: the catenary z = a cosh(rho / a)
    revolved about its own axis of symmetry.  The companion to the
    catenoid, which revolves the catenary about its BASE instead --
    and unlike the catenoid it is not minimal."""
    n = max(8, int(steps))
    return [(extent * i / n, a * math.cosh(extent * i / (n * a)))
            for i in range(n + 1)]


def sinusoid_rev_profile(aspect=1.0, beads=3, steps=32):
    """[(rho, z)] of the REVOLUTION OF THE SINUSOID, x = a cos(z/b)
    about Oz: a string of onion-dome beads meeting in cusp points on
    the axis.  Sampled bead by bead so every meridian zero lands
    EXACTLY on a sample; `_revolve` then welds those rings to single
    axis vertices and the beads close watertight."""
    k = max(1, int(beads))
    per = max(6, int(steps))
    zeta0 = math.pi / 2.0
    span = k * math.pi
    out = []
    for i in range(k * per + 1):
        zeta = zeta0 + span * i / (k * per)
        out.append((aspect * math.cos(zeta), zeta - zeta0 - span / 2.0))
    return out


def tractroid2_profile(a=1.0, reach=3.0, steps=120):
    """[(rho, z)] of the SECOND TRACTROID: the tractrix
    (a (t - tanh t), a sech t) revolved about the axis PERPENDICULAR
    to its asymptote.  (The pseudosphere is the other revolution, about
    the asymptote itself.)  t = 0 is the tractrix's cusp, on the axis,
    so the surface closes there in an apex; the far end is an open
    rim that flares toward the asymptote plane z = 0."""
    n = max(8, int(steps))
    return [(a * (t - math.tanh(t)), a / math.cosh(t))
            for t in (reach * i / n for i in range(n + 1))]


def _thin(profile, keep):
    """Subsample a profile to about `keep` points, ends included.

    The three ODE profiles are integrated FINELY -- accuracy there is
    cheap and is what the self-tests measure -- but meshing every step
    would spend tens of thousands of vertices on a smooth curve.  So the
    integration stays fine and only the mesh is thinned.
    """
    n = len(profile)
    if n <= keep or keep < 2:
        return profile
    step = (n - 1) / float(keep - 1)
    out = [profile[int(round(i * step))] for i in range(keep)]
    return out


def _revolve(profile, segments=96, caps=False):
    """Revolve a (rho, z) profile about Oz.

    Rings whose radius has collapsed to zero become a single vertex, so
    a profile that reaches the axis closes with a fan instead of a band
    of degenerate quads.
    """
    n = max(3, int(segments))
    th = [2.0 * math.pi * i / n for i in range(n)]
    verts, rows = [], []
    for rho, z in profile:
        if abs(rho) < 1e-12:
            rows.append([len(verts)] * n)
            verts.append((0.0, 0.0, z))
        else:
            base = len(verts)
            verts.extend((rho * math.cos(t), rho * math.sin(t), z)
                         for t in th)
            rows.append([base + i for i in range(n)])
    faces = []
    for a, b in zip(rows, rows[1:]):
        for i in range(n):
            j = (i + 1) % n
            ring = [a[i], a[j], b[j], b[i]]
            clean = [ring[0]] + [q for p, q in zip(ring, ring[1:])
                                 if p != q]
            if len(clean) > 2 and clean[0] == clean[-1]:
                clean = clean[:-1]
            if len(clean) >= 3:
                faces.append(clean)
    rims = []
    if caps:
        # A flat lid on either end that has not closed on the axis.
        # Without them the surface is not a solid and no volume the
        # divergence theorem computes from it means anything.  Each lid
        # meets the wall at a genuine circular fold, and those rims are
        # returned so they can be creased: smooth shading across them
        # rounds off the cut and makes a drop look like it dissolves
        # into the air instead of hanging from a pipe.
        for row, rev in ((rows[0], True), (rows[-1], False)):
            if row[0] == row[1]:                  # closed on the axis
                continue
            faces.append(list(reversed(row)) if rev else list(row))
            rims.extend((row[i], row[(i + 1) % n]) for i in range(n))
    return verts, faces, rims


def bouguer_profile(a=1.0, extent=1.6, steps=160):
    """[(rho, z)] of the BOUGUER DOME, the dome of constant thrust.

        a^2 y'' = x sqrt(1 + y'^2),   z = f(rho)

    Bouguer asked in 1734 what shape a masonry dome must take so that
    the line of thrust runs inside the masonry everywhere -- so that it
    stands by compression alone.  The answer is this profile.  It is
    worth comparing with the CATENARY, which solves the same equation
    without the leading x, a y'' = sqrt(1 + y'^2), and is the answer to
    the corresponding question for an ARCH: the extra factor of x is
    what the third dimension costs.

    Integrated in the closed form the source gives,
    f(x) = a * integral_0^x sinh(X^2 / 2a^2) dX, by Simpson's rule.
    """
    n = max(8, int(steps))
    h = float(extent) / n

    def g(x):
        return math.sinh(x * x / (2.0 * a * a))
    # accumulate the integral one step at a time, each step by Simpson's
    # rule on its own midpoint -- so the profile is sampled at every x
    # rather than at every other one
    xs, zs, acc = [0.0], [0.0], 0.0
    for i in range(n):
        x0 = i * h
        acc += h / 6.0 * (g(x0) + 4.0 * g(x0 + 0.5 * h) + g(x0 + h))
        xs.append(x0 + h)
        zs.append(a * acc)
    top = zs[-1]
    return [(x, top - z) for x, z in zip(xs, zs)]


def pendant_drop_profile(a=1.0, apex=1.0, steps=600, span=2.6):
    """[(rho, z)] of the HANGING DROP OF WATER.

    The drop hanging at the end of a vertical pipe is the surface of
    revolution whose mean curvature at each point is proportional to the
    height, which is Young-Laplace with the hydrostatic head included:

        dphi/ds + sin(phi) / rho = z / a^2

    with rho' = cos(phi), z' = sin(phi) along the profile's arclength.
    Integrated by RK4 from the apex, where the sin(phi)/rho term is
    removable -- both principal curvatures are equal there, so
    dphi/ds = z / (2 a^2).

    This is the classical closed profile.  For the same physics solved
    variationally, with a real contact angle and a free contact line,
    see `cmc_generator`, whose sessile drop minimizes the same energy.
    """
    ds = float(span) / max(16, int(steps))
    rho, z, phi = 0.0, float(apex), 0.0

    def dphi(r, zz, ph):
        return (zz / (a * a) - (math.sin(ph) / r if r > 1e-9
                                else zz / (2.0 * a * a)))
    out = [(0.0, z)]
    for _ in range(int(steps)):
        k = []
        st = (rho, z, phi)
        for w in (0.0, 0.5, 0.5, 1.0):
            r_, z_, p_ = (st[0] + w * ds * (k[-1][0] if k else 0.0),
                          st[1] + w * ds * (k[-1][1] if k else 0.0),
                          st[2] + w * ds * (k[-1][2] if k else 0.0))
            k.append((math.cos(p_), math.sin(p_), dphi(r_, z_, p_)))
        rho += ds / 6.0 * (k[0][0] + 2 * k[1][0] + 2 * k[2][0] + k[3][0])
        z += ds / 6.0 * (k[0][1] + 2 * k[1][1] + 2 * k[2][1] + k[3][1])
        phi += ds / 6.0 * (k[0][2] + 2 * k[1][2] + 2 * k[2][2] + k[3][2])
        if rho < 0.0:
            break
        out.append((rho, z))
    return out


def eh_rho(r, a=1.0):
    """Radius of the psi-circle of the Eguchi-Hanson metric."""
    return math.sqrt(max(r ** 4 - a ** 4, 0.0)) / (2.0 * r)


def eh_drho(r, a=1.0):
    """d rho / d r."""
    return (r ** 4 + a ** 4) / (2.0 * r ** 2 *
                                math.sqrt(max(r ** 4 - a ** 4, 1e-300)))


def eh_dz(r, a=1.0):
    """d z / d r for the isometric surface of revolution.

    Fixed by rho'^2 + z'^2 = g_rr; the algebra collapses to a perfect
    square, 4 r^8 over the numerator, so no radical is left over.
    """
    return math.sqrt(3.0 * r ** 4 + a ** 4) / (2.0 * r ** 2)


def eh_grr(r, a=1.0):
    """The metric coefficient the embedding has to reproduce."""
    return 1.0 / (1.0 - (a / r) ** 4)


def eguchi_hanson_profile(a=1.0, reach=3.0, steps=160):
    """(rho, z) along the Eguchi-Hanson bolt slice, z from the tip.

    z'(r) is finite everywhere including r = a -- it is rho' that
    blows up there, and rho is known in closed form -- so the height
    integrates with the plain trapezium rule.
    """
    a = max(a, 1e-6)
    r_max = max(reach, 1.05) * a
    h = (r_max - a) / max(steps - 1, 1)
    out, z = [], 0.0
    prev = eh_dz(a, a)
    for i in range(steps):
        r = a + i * h
        cur = eh_dz(r, a)
        if i:
            z += 0.5 * h * (prev + cur)
        prev = cur
        out.append((eh_rho(r, a), z))
    return out


def conifold_profiles(delta=-0.25, extent=1.5, steps=96):
    """Profiles of the real points of u^2 - v^2 - z^2 = delta.

    Revolving about the u-axis, the radius is sqrt(u^2 - delta).
    Below zero that never reaches the axis and the surface is one
    hyperboloid with a throat of radius sqrt(-delta) -- the real slice
    of the sphere that replaces the node.  At zero the radius is |u|
    and the profile passes through the axis, which `_revolve` closes
    to the single point of the node.  Above zero the radius is real
    only for |u| >= sqrt(delta), and the surface falls into two
    separate sheets, so two profiles come back instead of one.
    """
    n = max(3, int(steps))
    if delta <= 0.0:
        return [[(math.sqrt(u * u - delta), u)
                 for u in (-extent + 2.0 * extent * i / (2 * n)
                           for i in range(2 * n + 1))]]
    u0 = math.sqrt(delta)
    span = max(extent - u0, 1e-6)
    half = [(math.sqrt(max((u0 + span * i / n) ** 2 - delta, 0.0)),
             u0 + span * i / n) for i in range(n + 1)]
    return [half, [(rho, -u) for rho, u in half]]


def neiloid_profile(a=1.0, z0=0.15, z1=1.0, steps=96):
    """[(rho, z)] of the NEILOID, a rho^2 = z^3.

    The solid of revolution of Neile's semicubical parabola.  Foresters
    use it as one of the standard idealised trunk shapes, between the
    cone and the paraboloid, and it has the tidy closed volume
    V = pi (z2^4 - z1^4) / 4a between two horizontal planes -- which is
    what `_selftest` measures the mesh against.
    """
    n = max(4, int(steps))
    return [(math.sqrt(max(0.0, (z0 + (z1 - z0) * i / n) ** 3 / a)),
             z0 + (z1 - z0) * i / n) for i in range(n + 1)]


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
                   ('BOUGUER', "Bouguer Dome",
                    "The dome of constant thrust (Bouguer 1734): the "
                    "shape a masonry dome must take to stand by "
                    "compression alone"),
                   ('PENDANT_DROP', "Hanging Drop of Water",
                    "The drop hanging from a vertical pipe: a surface "
                    "of revolution whose curvature is proportional to "
                    "the height"),
                   ('NEILOID', "Neiloid",
                    "a rho^2 = z^3, the solid of revolution of Neile's "
                    "semicubical parabola"),
                   ('SCHWARZ_LANTERN', "Schwarz's Lantern",
                    "A polyhedron inscribed in a cylinder whose area can "
                    "exceed the cylinder's without bound"),
                   ('GABRIEL', "Gabriel's Horn",
                    "y = 1/x revolved: finite volume, infinite "
                    "lateral area"),
                   ('TORUS', "Torus",
                    "The torus of revolution: a circle revolved about "
                    "an axis in its plane, the bare classical donut"),
                   ('ALYSSEID', "Revolution of the Catenary",
                    "The alysseid: the catenary z = a cosh(rho/a) "
                    "revolved about its axis of symmetry -- the "
                    "catenoid's companion (which revolves the catenary "
                    "about its base instead), and not minimal"),
                   ('SINUSOID_REV', "Revolution of the Sinusoid",
                    "The sinusoid x = a cos(z/b) revolved about Oz: a "
                    "string of onion-dome beads meeting in cusp points "
                    "on the axis"),
                   ('TRACTROID2', "Second Tractroid",
                    "The tractrix revolved about the axis "
                    "PERPENDICULAR to its asymptote; the pseudosphere "
                    "revolves it about the asymptote itself.  Unlike "
                    "the pseudosphere it is not of constant curvature"),
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
                    "round sphere, and still every geodesic closes"),
                   ('EGUCHI_HANSON', "Eguchi-Hanson Space",
                    "An exact slice of the first Ricci-flat "
                    "asymptotically locally Euclidean metric, on "
                    "T*S^2. The tip closes smoothly only because the "
                    "angle runs over half of the three-sphere; the "
                    "two-sphere it is named for sits there, of radius "
                    "a/4"),
                   ('CONIFOLD', "Conifold Transition",
                    "The two-dimensional model of the conifold, "
                    "u^2 - v^2 - z^2 = delta: a node at delta = 0, "
                    "replaced by a sphere of radius sqrt(-delta) "
                    "below it and separating into two sheets above")],
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
        torus_tube: FloatProperty(
            name="Tube Radius", default=0.35, min=0.02, max=3.0,
            description="Radius of the revolved circle, against a "
                        "centre circle of radius 1: below 1 a ring "
                        "torus, 1 a horn torus, above 1 a spindle "
                        "torus (torus only)")
        beads: IntProperty(
            name="Beads", default=3, min=1, max=12,
            description="How many onion-dome beads of the revolved "
                        "sinusoid to build, one per half-period "
                        "(revolution of the sinusoid only)")
        bead_aspect: FloatProperty(
            name="Bead Aspect", default=1.0, min=0.1, max=6.0,
            description="Amplitude-to-period ratio a/b of the meridian "
                        "x = a cos(z/b): larger is fatter beads "
                        "(revolution of the sinusoid only)")
        tract_reach: FloatProperty(
            name="Profile Reach", default=3.0, min=0.5, max=8.0,
            description="How far along the tractrix the meridian runs "
                        "from its cusp, in the curve's parameter; the "
                        "rim creeps toward the asymptote plane as it "
                        "grows (second tractroid only)")
        dome_a: FloatProperty(
            name="Scale a", default=1.0, min=0.05, max=10.0,
            description="The a in Bouguer's a^2 y'' = x sqrt(1 + y'^2), "
                        "in the neiloid's a rho^2 = z^3, in the "
                        "revolved catenary's z = a cosh(rho/a), in "
                        "the second tractroid's tractrix, and the bolt "
                        "size a of Eguchi-Hanson space")
        dome_extent: FloatProperty(
            name="Base Radius", default=1.6, min=0.2, max=6.0,
            description="How far out the profile runs (Bouguer dome "
                        "and revolution of the catenary)")
        drop_a: FloatProperty(
            name="Capillary Length", default=1.0, min=0.1, max=6.0,
            description="The a in 2H = z / a^2: large a is a nearly "
                        "spherical drop, small a a long pendant one "
                        "(hanging drop only)")
        drop_apex: FloatProperty(
            name="Apex Height", default=1.0, min=0.05, max=6.0,
            description="Height of the drop's lowest point above the "
                        "zero-curvature plane (hanging drop only)")
        drop_span: FloatProperty(
            name="Profile Length", default=2.2, min=0.3, max=8.0,
            description="How far the profile is integrated along its "
                        "own arc before the neck is cut (hanging drop "
                        "only)")
        neiloid_z0: FloatProperty(
            name="Base Height", default=0.2, min=0.01, max=0.95,
            description="Where the trunk is cut off below; the tip at "
                        "z = 0 is a cusp (neiloid only)")
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
        bolt_reach: FloatProperty(
            name="Reach", default=3.0, min=1.05, max=12.0,
            description="How far out along r to build Eguchi-Hanson "
                        "space, in units of the bolt size a. Far out "
                        "the metric is flat (Eguchi-Hanson only)")
        conifold_delta: FloatProperty(
            name="Deformation", default=-0.25, min=-1.0, max=1.0,
            description="Negative opens a throat of radius "
                        "sqrt(-delta), the real slice of the sphere "
                        "that replaces the node; zero is the singular "
                        "cone; positive separates the two sheets "
                        "(conifold only)")
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
            rims = []                     # cap rims to crease, if any
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
            elif self.surface == 'TORUS':
                verts, faces = build_torus(1.0, self.torus_tube,
                                           2 * res, res)
                name = "Torus"
            elif self.surface == 'ALYSSEID':
                verts, faces, rims = _revolve(
                    alysseid_profile(self.dome_a, self.dome_extent,
                                     res), 2 * res)
                name = "Revolution of the Catenary"
            elif self.surface == 'SINUSOID_REV':
                verts, faces, rims = _revolve(
                    sinusoid_rev_profile(self.bead_aspect, self.beads,
                                         max(8, (3 * res)
                                             // max(1, self.beads))),
                    2 * res)
                name = "Revolution of the Sinusoid"
            elif self.surface == 'TRACTROID2':
                verts, faces, rims = _revolve(
                    tractroid2_profile(self.dome_a, self.tract_reach,
                                       2 * res), 2 * res)
                name = "Second Tractroid"
            elif self.surface == 'BOUGUER':
                verts, faces, rims = _revolve(
                    _thin(bouguer_profile(self.dome_a, self.dome_extent,
                                          8 * res), res),
                    2 * res, caps=True)
                name = "Bouguer Dome"
            elif self.surface == 'PENDANT_DROP':
                verts, faces, rims = _revolve(
                    _thin(pendant_drop_profile(
                        self.drop_a, self.drop_apex, 24 * res,
                        self.drop_span), res),
                    2 * res, caps=True)
                name = "Hanging Drop"
            elif self.surface == 'NEILOID':
                verts, faces, rims = _revolve(
                    _thin(neiloid_profile(self.dome_a, self.neiloid_z0,
                                          1.0, 4 * res), res),
                    2 * res, caps=True)
                name = "Neiloid"
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
            elif self.surface == 'EGUCHI_HANSON':
                verts, faces, rims = _revolve(
                    eguchi_hanson_profile(self.dome_a, self.bolt_reach,
                                          4 * res), 2 * res)
                name = "Eguchi-Hanson Space"
            elif self.surface == 'CONIFOLD':
                verts, faces = [], []
                for prof in conifold_profiles(self.conifold_delta,
                                              self.dome_extent,
                                              2 * res):
                    v, f, _ = _revolve(prof, 2 * res)
                    off = len(verts)
                    verts.extend(v)
                    faces.extend([tuple(i + off for i in fc) for fc in f])
                name = "Conifold"
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
            # The cut rim of a capped profile is a real fold: the flat
            # lid meets the curved wall at a circle, and shading
            # smoothly across it makes the drop look like it dissolves
            # into the air rather than hanging from a pipe.
            if rims:
                mark_sharp(me, rims)
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
            elif self.surface == 'TORUS':
                lay.prop(self, 'torus_tube')
            elif self.surface == 'ALYSSEID':
                for k in ('dome_a', 'dome_extent'):
                    lay.prop(self, k)
            elif self.surface == 'SINUSOID_REV':
                for k in ('beads', 'bead_aspect'):
                    lay.prop(self, k)
            elif self.surface == 'TRACTROID2':
                for k in ('dome_a', 'tract_reach'):
                    lay.prop(self, k)
            elif self.surface == 'BOUGUER':
                for k in ('dome_a', 'dome_extent'):
                    lay.prop(self, k)
            elif self.surface == 'PENDANT_DROP':
                for k in ('drop_a', 'drop_apex', 'drop_span'):
                    lay.prop(self, k)
            elif self.surface == 'NEILOID':
                for k in ('dome_a', 'neiloid_z0'):
                    lay.prop(self, k)
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

    # ---- three surfaces of revolution defined by a differential ------
    # condition rather than a formula.  Each is checked on that
    # condition, or on a closed form the source states independently.
    a = 1.0
    P = bouguer_profile(a=a, extent=1.6, steps=4000)
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    res = 0.0
    for i in range(2, len(P) - 2):
        h = xs[i + 1] - xs[i]
        y1 = (ys[i + 1] - ys[i - 1]) / (2 * h)
        y2 = (ys[i + 1] - 2 * ys[i] + ys[i - 1]) / (h * h)
        if 0.15 < xs[i] < 1.45:
            res = max(res, abs(a * a * y2 + xs[i]
                               * math.sqrt(1.0 + y1 * y1)))
    assert res < 1e-5, res
    print("bouguer dome: profile satisfies a^2 y'' = x sqrt(1+y'^2) to "
          "%.0e -- the catenary's equation with the extra factor of x "
          "that the third dimension costs" % res)

    # The hanging drop: its DEFINITION is 2H = z / a^2, and that is
    # measured back off the finished profile rather than trusted from
    # the integrator that produced it.
    P = pendant_drop_profile(a=1.0, apex=1.0, steps=3000, span=2.2)
    rho = [p[0] for p in P]
    zz = [p[1] for p in P]
    worst = 0.0
    for i in range(3, len(P) - 6):
        if rho[i] < 0.15:
            continue
        dr = (rho[i + 1] - rho[i - 1]) / 2.0
        dz = (zz[i + 1] - zz[i - 1]) / 2.0
        ds = math.hypot(dr, dz)
        ph = math.atan2(dz, dr)
        ph0 = math.atan2((zz[i] - zz[i - 2]) / 2.0,
                         (rho[i] - rho[i - 2]) / 2.0)
        ph1 = math.atan2((zz[i + 2] - zz[i]) / 2.0,
                         (rho[i + 2] - rho[i]) / 2.0)
        twoH = (ph1 - ph0) / (2.0 * ds) + math.sin(ph) / rho[i]
        worst = max(worst, abs(twoH - zz[i] / 1.0))
    assert worst < 1e-4, worst
    print("hanging drop: mean curvature stays proportional to height, "
          "|2H - z/a^2| < %.0e measured off the profile" % worst)

    # Neiloid: the source gives the volume in closed form, so the mesh
    # is measured against it -- and refined until it agrees.
    prev = None
    for res_n in (200, 400, 800):
        V, F, _rims = _revolve(neiloid_profile(1.0, 0.2, 1.0, res_n),
                               res_n, caps=True)
        vol = abs(_volume(V, F))
        want = math.pi / 4.0 * (1.0 ** 4 - 0.2 ** 4)
        err = abs(vol - want) / want
        assert prev is None or err < prev, (res_n, err, prev)
        prev = err
    assert err < 1e-4, err
    print("neiloid: volume converges on the published "
          "pi(z2^4 - z1^4)/4a, relative error %.1e at 800 steps" % err)

    # ---- the four Ferreol revolution additions -----------------------
    # Each is gated on the closed-form identity that defines it, not on
    # "it meshed".
    R_, r_ = 1.0, 0.35
    V, F = build_torus(R_, r_, 96, 48)
    assert _finite(V) and _valid(V, F) and _watertight(F)
    for (x, y, z) in V:
        w = math.hypot(x, y) - R_
        assert abs(w * w + z * z - r_ * r_) < 1e-12
    vol = abs(_volume(V, F))
    want = 2.0 * math.pi ** 2 * R_ * r_ * r_       # Pappus
    assert abs(vol - want) / want < 5e-3, vol
    print("torus: watertight, every vertex on (sqrt(x^2+y^2)-R)^2 + "
          "z^2 = r^2, volume within 0.5 percent of Pappus' "
          "2 pi^2 R r^2")

    a_ = 1.3
    V, F, _r = _revolve(alysseid_profile(a_, 1.6, 64), 48)
    assert _finite(V) and _valid(V, F)
    for (x, y, z) in V:
        rho = math.hypot(x, y)
        assert abs(z - a_ * math.cosh(rho / a_)) < 1e-9
    print("alysseid: every vertex on z = a cosh(rho/a)")

    asp = 1.2
    prof = sinusoid_rev_profile(asp, 3, 24)
    V, F, _r = _revolve(prof, 48)
    assert _finite(V) and _valid(V, F) and _watertight(F)
    span = 3 * math.pi
    for (x, y, z) in V:
        rho = math.hypot(x, y)
        want = abs(asp * math.cos(z + span / 2.0 + math.pi / 2.0))
        assert abs(rho - want) < 1e-9, (rho, want)
    print("revolution of the sinusoid: watertight (beads weld at "
          "their cusp points), every vertex on rho = a |cos(z/b)|")

    a_ = 0.9
    V, F, _r = _revolve(tractroid2_profile(a_, 3.0, 60), 48)
    assert _finite(V) and _valid(V, F)
    for (x, y, z) in V:
        rho = math.hypot(x, y)
        if z > 1e-9 and rho > 1e-9:
            t = math.acosh(a_ / z)
            assert abs(rho - a_ * (t - math.tanh(t))) < 1e-7
    print("second tractroid: every vertex on the revolved tractrix "
          "(rho, z) = a(t - tanh t, sech t)")

    # Eguchi-Hanson: the embedding is isometric, and the tip closes.
    for a_ in (0.5, 1.0, 2.3):
        worst = 0.0
        for i in range(1, 2000):
            r = a_ * (1.000001 + 0.01 * i)
            worst = max(worst, abs(eh_drho(r, a_) ** 2
                                   + eh_dz(r, a_) ** 2
                                   - eh_grr(r, a_)) / eh_grr(r, a_))
        assert worst < 1e-9, worst
        tip = eh_drho(a_ * (1 + 1e-7), a_) / math.sqrt(
            eh_grr(a_ * (1 + 1e-7), a_))
        assert abs(tip - 1.0) < 1e-6, tip
    print("eguchi-hanson: rho'^2 + z'^2 = g_rr to 1e-9 relative, and "
          "drho/ds -> 1 at the bolt, so the tip closes smoothly for a "
          "psi period of 2 pi -- the Z2 quotient, made visible")
    # Asymptotically flat: the psi-circle tends to r/2.
    assert abs(eh_rho(1e5, 1.0) / 5e4 - 1.0) < 1e-6
    V, F, _r = _revolve(eguchi_hanson_profile(1.0, 3.0, 96), 48)
    assert _finite(V) and _valid(V, F)
    assert min(math.hypot(x, y) for x, y, _z in V) < 1e-12
    print("eguchi-hanson: mesh closes on the axis at the bolt")

    # The conifold model sits exactly on its quadric, and the throat
    # is exactly the vanishing cycle.
    for delta, sheets in ((-0.36, 1), (0.0, 1), (0.25, 2)):
        profs = conifold_profiles(delta, 1.5, 48)
        assert len(profs) == sheets, (delta, len(profs))
        allv = []
        for prof in profs:
            V, F, _r = _revolve(prof, 48)
            assert _finite(V) and _valid(V, F)
            allv.extend(V)
        res = max(abs(z * z - x * x - y * y - delta) for x, y, z in allv)
        assert res < 1e-12, res
        if delta < 0.0:
            waist = min(math.hypot(x, y) for x, y, _z in allv)
            assert abs(waist - math.sqrt(-delta)) < 1e-9, waist
    print("conifold: every vertex on u^2 - v^2 - z^2 = delta to 1e-12; "
          "one sheet at and below the node, two above; throat radius "
          "exactly sqrt(-delta)")

    print("miscellaneous surfaces standalone tests passed")
