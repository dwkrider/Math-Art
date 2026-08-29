
# Zonohedra Generator for Blender
#
# Zonohedra -- Minkowski sums of line segments -- together with the
# constructions that surround them, after Antiprism's `zono` program and
# George Hart's survey of the polar family.
#
#   POLAR        the two-parameter family PZ(n, theta): n equal ribs at a
#                common pitch.  Built from Hart's contiguous-run indexing,
#                so every face knows its LEVEL, which is what makes dome
#                truncation, per-level colouring and flat cutting
#                templates possible at all.
#   SPIRAL       Russell Towle's rhombic spirallohedra, from Antiprism's
#                make_polar_zonohedron with a nonzero spiral width.  A
#                star STEP (Antiprism's `zono -P n/d`) shares its factor
#                with n to split the model into that many lobes.
#   NAMED STARS  the classic rhombic solids: cube axes, cube diagonals
#                (rhombic dodecahedron), icosahedral axes (rhombic
#                triacontahedron), dodecahedral axes (rhombic
#                enneacontahedron).
#   OBJECT       a star read off the active mesh, with Antiprism's four
#                methods -- centre-to-vertices, all vertex pairs, edges,
#                face normals -- plus its centre and unit-length options.
#   TRANSLATION  the surface swept by translating one chain along another
#                (`zono -T`).  Every face is still a parallelogram, but
#                the result may be open, non-convex or toroidal, which no
#                Minkowski sum of segments can be.
#   ROSETTE      the flat member of the family: rings of rhombi filling a
#                regular polygon, optionally with Towle-style bites taken
#                out of the rim.
#
# Any three-dimensional star can be emitted as the solid, as its C(n,3)
# rhombohedral dissection, as its surface helices (extruded, after
# Webster), or as flat cutting templates.  Zone lengths, an axial zone,
# an elliptical cross-section, dome truncation, fused lobes and a
# proportional face opening are all independent dials on top.
#
# The mathematics lives in `polyhedra/zonotope.py` (general zonotopes,
# translation surfaces, rosettes) and `polyhedra/polar.py` (the polar
# family, its closed forms and its design presets); this module is the
# Blender layer over them.
#
# NOTE on the pitch convention: this operator's Pitch has always been
# measured from the AXIS, while every published formula measures from the
# horizontal.  The Angle From dial says which one the number means, so the
# presets below can be written the way their sources state them.
#
# References:
# - Zonohedra / zonotopes: E. S. Fedorov (1885); M. Senechal and R. V.
#   Galiulin, "An Introduction to the Theory of Figures: The Geometry of
#   E. S. Federov", Structural Topology 10 (1984), pp. 5-22.
# - Zonohedra as Minkowski sums / zones: H. S. M. Coxeter, "Regular
#   Polytopes", 3rd ed., Dover, 1973, ch. 2.
# - George W. Hart, "The Joy of Polar Zonohedra", Bridges 2021, pp. 7-14 --
#   the level structure, surface helices, the face-angle, dihedral, height
#   and radius formulas, the volume-optimal pitch, the congruent-levels
#   construction, and the axial-zone, elliptical and fused-lobe variations.
# - B. Chilton and H. S. M. Coxeter, "Polar Zonohedra", American
#   Mathematical Monthly 70(9) (1963), pp. 946-951.
# - C. H. H. Franklin, "Hypersolid Concepts and the Completeness of Things
#   and Phenomena", Mathematical Gazette 21 (1937), pp. 360-364 -- polar
#   zonohedra as shadows of n-dimensional hypercubes.
# - Phil Webster, "Polar Zonohedral Helices and Clusters", Bridges 2023,
#   pp. 329-336 -- the extruded surface helices PZ_E(n, theta, d).
# - Russell Towle, "Polar Zonohedra", Mathematica Journal 6(2) (1996),
#   pp. 8-12; "Rhombic Spirallohedra" (2003); the bitten-zonogon idea as
#   quoted in Michel Petitjean, "Spirallohedra and Space Filling. A
#   Tribute to Russell Towle", Symmetry: Culture and Science 19(1) (2008),
#   pp. 5-8.
# - Antiprism (Adrian Rossiter), the `zono` program and
#   `base/zonohedron.cc`.
# - Rhombic rose (the flat, two-dimensional case): Alan H. Schoen,
#   "Rhombic rosettes" (schoengeometry.com); construction and counts from
#   Robert Ferreol, "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), "rosace rhombique".  Its order-5 pair of rhombi are
#   the ones in Roger Penrose's rhomb tilings.
# - Dissection into C(n,3) parallelepipeds: Gerhard Kowalewski, "Der
#   Keplersche Koerper und andere Bauspiele", Koehlers, Leipzig (1938);
#   G. M. Ziegler, "Lectures on Polytopes", Springer (1995), ch. 7.
# - Jean E. Taylor, "Zonohedra and Generalized Zonohedra", American
#   Mathematical Monthly 99(2) (1992), pp. 108-111 -- Fedorov's original
#   definition is wider than the Minkowski-sum class built here.

bl_info = {
    "name": "Zonohedra Generator",
    "author": "Math Art project (after Antiprism's zono)",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Zonohedron",
    "description": "Polar zonohedra, spirallohedra, translation surfaces "
                   "and their dissections, helices and templates",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, gcd

try:
    from .polyhedra import zonotope as _zt
    from .polyhedra import polar as _pz
    from .polyhedra import fit as _fit
except ImportError:                       # flat-file / headless import
    from polyhedra import zonotope as _zt
    from polyhedra import polar as _pz
    from polyhedra import fit as _fit

PHI = (1 + 5 ** 0.5) / 2

#: Re-exported so `ifs/spacefill.py` keeps importing the spirallohedron
#: construction from here, where it has always lived.
make_polar_zonohedron = _pz.make_polar_zonohedron


# --------------------------------------------------------------------------
# Stars
# --------------------------------------------------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / l for x in v]


def star_vectors(kind, n=7, pitch=45.0, seed=0):
    if kind == 'CUBE':
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if kind == 'RHOMBIC_DODECA':          # 4 cube diagonals
        return [_unit([1, 1, 1]), _unit([1, -1, -1]),
                _unit([-1, 1, -1]), _unit([-1, -1, 1])]
    if kind == 'TRIACONTA':               # 6 icosahedral (5-fold) axes
        vs = []
        for a in (1, -1):
            vs.append(_unit([0, a, PHI]))
            vs.append(_unit([a, PHI, 0]))
            vs.append(_unit([PHI, 0, a]))
        return vs
    if kind == 'ENNEACONTA':              # 10 dodecahedral (3-fold) axes
        vs = [_unit([1, 1, 1]), _unit([1, -1, -1]),
              _unit([-1, 1, -1]), _unit([-1, -1, 1])]
        for a in (1, -1):
            vs.append(_unit([0, a / PHI, PHI]))
            vs.append(_unit([a / PHI, PHI, 0]))
            vs.append(_unit([PHI, 0, a / PHI]))
        return vs
    if kind == 'POLAR':
        return polar_star(n, pitch)
    if kind == 'RANDOM':
        import random
        rng = random.Random(seed)
        vs = []
        while len(vs) < n:
            v = [rng.gauss(0, 1) for _ in range(3)]
            if math.sqrt(sum(x * x for x in v)) > 1e-3:
                vs.append(_unit(v))
        return vs
    raise ValueError(kind)


def polar_star(n, pitch=45.0):
    """Antiprism -P star: unit vectors, equal azimuth spacing, common
    pitch measured from the AXIS (45 degrees in the original)."""
    p = math.radians(pitch)
    return [[sin(p) * cos(2 * pi * k / n), -sin(p) * sin(2 * pi * k / n),
             cos(p)] for k in range(n)]


def subset_sums(star):
    """Every subset sum of the star.  Kept because it is the textbook
    definition of the zonotope's vertex set; the generator no longer uses
    it, since enumerating faces directly avoids the 2^n blow-up."""
    pts = [[0.0, 0.0, 0.0]]
    for v in star:
        pts = pts + [[p[0] + v[0], p[1] + v[1], p[2] + v[2]] for p in pts]
    return pts


def axial_star(star, length):
    """The star with one extra zone along the polar axis.

    Adding a zone parallel to the n-fold axis keeps the n-fold rotational
    symmetry (and so still reads as a polar solid) while breaking the
    surface helices, which is the price of the extra height.  Applied to
    Fedorov's rhombic icosahedron -- the 5-fold polar zonohedron at the
    congruent pitch -- it produces Kepler's rhombic triacontahedron.
    """
    return list(star) + [[0.0, 0.0, float(length)]]


# --------------------------------------------------------------------------
# Face opening: a proportional rim, not a constant-width border
# --------------------------------------------------------------------------

def open_faces(V, F, opening):
    """Replace each face by a rim around a scaled copy of itself.

    Every corner moves a fraction `opening` of the way from the face's
    centroid back out to itself, and the ring between the two outlines is
    kept.  The inset is therefore PROPORTIONAL: a small face near a pole
    keeps the same relative hole as a big one at the equator, which is the
    difference from the Leonardo style's constant-width border and is what
    makes the openings read as one family across a polar zonohedron.

    Returns (V2, F2, source_face_index) so a per-face colouring survives.
    """
    if opening <= 0.0:
        return [list(v) for v in V], [list(f) for f in F], list(range(len(F)))
    s = min(max(float(opening), 1e-4), 0.999)
    V2, F2, src = [], [], []
    for fi, f in enumerate(F):
        m = len(f)
        c = [sum(V[i][d] for i in f) / m for d in range(3)]
        base = len(V2)
        for i in f:
            V2.append(list(V[i]))
        for i in f:
            V2.append([c[d] + s * (V[i][d] - c[d]) for d in range(3)])
        for k in range(m):
            k2 = (k + 1) % m
            F2.append([base + k, base + k2, base + m + k2, base + m + k])
            src.append(fi)
    return V2, F2, src


# --------------------------------------------------------------------------
# Builders (pure Python -- no bpy below this line until the operator)
# --------------------------------------------------------------------------

def build_polar(n, theta, ellipticity=1.0, levels=0, cap_rim=False,
                lobes=1, zone_profile='UNIFORM', zone_spread=0.0,
                zone_seed=0, axial=0.0):
    """The polar zonohedron and everything hung off its level structure.

    Returns (V, F, keys) where `keys` carries, per face, a
    (level, zone pair) label for the colourings.  An axial zone or an
    uneven zone profile leaves the level construction behind and goes
    through the general zonotope engine instead, because neither keeps the
    n ribs congruent.
    """
    star = _pz.ribs(n, theta, ellipticity)
    if zone_profile != 'UNIFORM' and zone_spread > 0.0:
        star = _zt.weighted_star(star, zone_profile, zone_spread, zone_seed)
    if axial > 0.0:
        star = axial_star(star, axial)
    if axial > 0.0 or (zone_profile != 'UNIFORM' and zone_spread > 0.0):
        # no longer a polar star: fall back to the general enumerator,
        # which does not have levels, so the level slot reports 0
        V, F = _zt.zonotope(star)
        pairs = _zt.zonotope_labels(star)
        return V, F, [(0, p) for p in pairs]

    V, F, lev = _pz.polar_solid(star, levels=levels, cap_rim=cap_rim)
    pairs = _pz.zone_pairs(n, levels=levels)
    keys = [(lev[i], pairs[i] if i < len(pairs) else ('cap',))
            for i in range(len(F))]
    if lobes > 1:
        V, F, keys = _stack_lobes(star, V, F, keys, lobes)
    return V, F, keys


def _stack_lobes(star, V, F, keys, lobes):
    """Repeat the solid pole onto pole, the finite piece of Hart's
    complete polar zonohedron."""
    Vo, Fo, Ko = [], [], []
    for off in _pz.lobe_offsets(star, lobes):
        base = len(Vo)
        Vo += [[v[d] + off[d] for d in range(3)] for v in V]
        Fo += [[i + base for i in f] for f in F]
        Ko += list(keys)
    return Vo, Fo, Ko


def build_general(star):
    """The zonohedron of an arbitrary star, plus its zone-pair labels."""
    if len(star) < 3:
        raise ValueError("a solid zonohedron needs at least three zones")
    V, F = _zt.zonotope(star)
    return V, F, [(0, p) for p in _zt.zonotope_labels(star)]


def build_dissection(star, explode=0.0):
    """The C(n,3) parallelepipeds, each as (V, F, kind, triple)."""
    n = len(star)
    if n < 3:
        raise ValueError("a dissection needs at least three zones")
    cells = n * (n - 1) * (n - 2) // 6
    if cells > 4000:
        raise ValueError(
            "%d zones would dissect into %d blocks; keep it under 4000 "
            "(about 30 zones)" % (n, cells))
    out = []
    for V, F, triple in _zt.cells_centered(star):
        kind = _zt.acute_or_flat(V, F)
        if explode:
            c = [sum(v[k] for v in V) / len(V) for k in range(3)]
            V = [[v[k] + c[k] * explode for k in range(3)] for v in V]
        out.append((V, F, kind, triple))
    return out


def build_templates(n, theta, opening=0.0, side=1.0):
    """Flat cutting templates: one rhombus per distinct level, plus the
    unfolded chain of a whole meridian, laid out in the XY plane.

    Levels m and n-m are congruent, so a builder cuts floor(n/2) shapes
    and no more; the chain below them is the strip that folds back into a
    pole-to-pole band, and is the piece to assemble first.
    """
    polys, angs, lvls = _pz.template_rhombi(n, theta, side)
    chain = _pz.template_chain(n, theta, side)
    drop = max((max(p[1] for p in q) for q in polys), default=1.0) + 1.2 * side
    V, F, keys = [], [], []
    for quad, m in zip(polys, lvls):
        base = len(V)
        for (x, y) in quad:
            V.append([x, y, 0.0])
        F.append([base, base + 1, base + 2, base + 3])
        keys.append((m, ('template', m)))
    for k, quad in enumerate(chain, start=1):
        base = len(V)
        for (x, y) in quad:
            V.append([x, y - drop, 0.0])
        F.append([base, base + 1, base + 2, base + 3])
        keys.append((min(k, n - k), ('chain', k)))
    if opening > 0.0:
        V, F, src = open_faces(V, F, opening)
        keys = [keys[i] for i in src]
    return V, F, keys, angs, lvls


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    try:
        from .styles import net_style as _net_style
        from .styles import shell as _shell
        from .styles import face_colors as _fc
    except ImportError:
        from styles import net_style as _net_style
        from styles import shell as _shell
        from styles import face_colors as _fc

    KINDS = [
        ('POLAR', "Polar Zonohedron",
         "n equal ribs at a common pitch -- the two-parameter family, "
         "with levels, helices and templates"),
        ('SPIRAL', "Rhombic Spirallohedron",
         "Polar star with a spiral width (zono -P n,w; after Russell "
         "Towle)"),
        ('RHOMBIC_DODECA', "Rhombic Dodecahedron", "4 cube diagonals"),
        ('TRIACONTA', "Rhombic Triacontahedron", "6 icosahedral axes"),
        ('ENNEACONTA', "Rhombic Enneacontahedron", "10 dodecahedral axes"),
        # The cube is a zonohedron -- three orthogonal generators, the
        # simplest parallelohedron there is -- but offering it here only
        # duplicates Add > Polyhedra > Regular Solid, which builds it
        # exactly and with the whole Platonic family beside it. build()
        # still accepts kind='CUBE' as the degenerate base case.
        ('RANDOM', "Random Star", "n random unit vectors"),
        ('OBJECT', "From Active Object",
         "Read the star off the selected mesh, the way Antiprism's zono "
         "reads it off a model"),
        ('TRANSLATION', "Translation Surface",
         "Sweep one chain along another (zono -T). Every face is still a "
         "parallelogram, but the surface can be open, non-convex or a "
         "torus, which no Minkowski sum of segments can be"),
        ('ROSETTE', "Rhombic Rose",
         "The two-dimensional member of the family: rings of rhombi "
         "filling a regular polygon.  At order 5 its two rhombi are the "
         "ones the Penrose tilings are made of"),
    ]

    #: kinds whose star is the polar one, so the level machinery applies
    _POLAR_KINDS = {'POLAR', 'SPIRAL'}
    #: kinds that produce a three-dimensional star and so can be dissected
    _STAR_KINDS = {'POLAR', 'SPIRAL', 'RHOMBIC_DODECA', 'TRIACONTA',
                   'ENNEACONTA', 'CUBE', 'RANDOM', 'OBJECT'}

    OUTPUTS = [
        ('SOLID', "Solid", "The zonohedron itself"),
        ('DISSECTION', "Dissection",
         "The C(n,3) parallelepipeds the zonohedron cuts into -- "
         "Kowalewski's twenty golden rhombohedra for the rhombic "
         "triacontahedron"),
        ('HELICES', "Extruded Helices",
         "One handedness of the surface helices, extruded down the axis "
         "into spiralling walls (Webster's PZ_E)"),
        ('TEMPLATES', "Flat Templates",
         "One flat rhombus per distinct level plus the unfolded chain of "
         "a meridian -- the parts list for a physical model"),
    ]

    COLORINGS = [
        ('NONE', "None", "One material, or none at all"),
        ('ZONES', "Zone Pair",
         "Every face lies in the plane of exactly two zones, so colouring "
         "by that pair shows the zone structure the solid is built from"),
        ('LEVEL', "Level",
         "Polar zonohedra only: the n faces of each level are congruent, "
         "and this is the colouring a builder wants, since one level is "
         "one part number"),
        ('SIDES', "Face Size", "One material per edge count"),
        ('BLOCK', "Block",
         "Dissection only: one material per zone triple, so blocks of the "
         "same shape and orientation match"),
        ('RING', "Ring",
         "Rhombic Rose only: one material per ring of rhombi, which at "
         "order 5 separates the two Penrose rhombs"),
    ]

    SOURCES = [
        ('VERTS', "Vertices", "Centre to each vertex (zono -m v)"),
        ('PAIRS', "Vertex Pairs",
         "Every vertex-to-vertex direction (zono -m a)"),
        ('EDGES', "Edges", "The mesh's edge directions (zono -m i / -m e)"),
        ('FACES', "Face Normals", "One zone per face normal"),
    ]

    CENTRES = [
        ('ORIGIN', "Object Origin", "Measure from the object's own origin"),
        ('CENTROID', "Centroid", "Measure from the mean of the vertices"),
        ('CURSOR', "3D Cursor", "Measure from the 3D cursor"),
    ]

    CHAINS = [
        ('POLYGON', "Polygon",
         "A regular or star polygon's edge vectors; they sum to zero, so "
         "the chain closes into a loop"),
        ('HELIX', "Helix",
         "A discrete helix, left open so the surface has a boundary"),
    ]

    PITCH_PRESETS = [
        ('CUSTOM', "Custom", "Use the Pitch dial as given"),
        ('VOLUME', "Volume-optimal",
         "35.264 degrees from the horizontal, for every n: the ribs form "
         "a eutactic star, so the solid is a shadow of the n-cube. Gives "
         "the cube at n=3 and the rhombic dodecahedron at n=4"),
        ('AREA', "Area-optimal",
         "The pitch of greatest surface area. It has no closed form and "
         "drifts with n, from 35.26 degrees at n=3 towards about 33.5 "
         "for large n, so it is found by search"),
        ('SQUARE45', "45 degrees",
         "A square middle level when n is even, and the limiting profile "
         "of a sinusoid of slope 1 at the poles"),
        ('CONGRUENT', "Congruent Levels",
         "The pitch that makes two chosen levels the same rhombus, "
         "cutting one shape from the parts list. At n=5 with levels 1 "
         "and 2 all twenty faces become golden rhombi -- Fedorov's "
         "rhombic icosahedron"),
    ]

    ZONE_PROFILES = [
        ('UNIFORM', "Uniform", "Every zone the same length"),
        ('RAMP', "Ramp", "Lengths rising steadily around the star"),
        ('SINE', "Wave", "Lengths following one sine period"),
        ('RANDOM', "Random", "Lengths jittered by the seed"),
    ]

    def _hull_bmesh(pts):
        bm = bmesh.new()
        for p in pts:
            bm.verts.new((p[0], p[1], p[2]))
        bm.verts.ensure_lookup_table()
        bmesh.ops.convex_hull(bm, input=bm.verts[:])
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.5),
                                 verts=bm.verts[:], edges=bm.edges[:])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        V = [list(v.co) for v in bm.verts]
        idx = {v: k for k, v in enumerate(bm.verts)}
        F = [[idx[v] for v in f.verts] for f in bm.faces]
        bm.free()
        return V, F

    def _weld(V, F, tol=1e-6):
        """Merge coincident vertices and drop degenerate faces."""
        me = bpy.data.meshes.new("__weld")
        me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
        me.validate(clean_customdata=True)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=tol)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        Vo = [list(v.co) for v in me.vertices]
        Fo = [list(p.vertices) for p in me.polygons]
        bpy.data.meshes.remove(me)
        return Vo, Fo

    class MESH_OT_zonohedron_add(bpy.types.Operator,
                                 _net_style.NetStyleProps):
        """Add a zonohedron: the polar family, spirallohedra, translation
        surfaces, dissections, helices or flat templates"""
        bl_idname = "mesh.zonohedron_add"
        bl_label = "Zonohedron"
        bl_options = {'REGISTER', 'UNDO'}

        # ---- what to build ------------------------------------------
        kind: EnumProperty(name="Star", items=KINDS, default='SPIRAL',
                           description="Vector star the zonohedron is "
                                       "built from")
        output: EnumProperty(
            name="Output", items=OUTPUTS, default='SOLID',
            description="The solid itself, its rhombohedral dissection, "
                        "its extruded surface helices, or flat templates")

        # ---- the star ------------------------------------------------
        n: IntProperty(
            name="Vectors", default=12, min=3, max=64,
            description="Number of star vectors (polar / spiral / random "
                        "/ rose)")
        step: IntProperty(
            name="Star Step", default=1, min=1, max=31,
            description="Offset polygon for the polar star (Antiprism's "
                        "zono -P n/d). A step sharing a factor with the "
                        "vector count splits the model into that many "
                        "separate lobes")
        spiral_width: IntProperty(
            name="Spiral Width", default=4, min=1, max=31,
            description="Spirallohedron spiral width (zono -P n,w)")
        pitch: FloatProperty(
            name="Pitch", default=55.0, min=1.0, max=89.0,
            description="Tilt of the star's ribs, in degrees, measured "
                        "from whichever reference Angle From selects")
        angle_ref: EnumProperty(
            name="Angle From",
            items=[('AXIS', "Axis",
                    "Measured from the polar axis, as this generator has "
                    "always done"),
                   ('HORIZON', "Horizontal",
                    "Measured from the horizontal, as Hart and every "
                    "published formula for polar zonohedra do")],
            default='AXIS',
            description="Which reference the Pitch angle is measured "
                        "from. The two differ by 90 degrees, and the "
                        "literature uses the horizontal")
        pitch_preset: EnumProperty(
            name="Pitch Preset", items=PITCH_PRESETS, default='CUSTOM',
            description="Pick the pitch for a mathematical reason rather "
                        "than by eye; anything but Custom overrides the "
                        "Pitch dial")
        congruent_a: IntProperty(
            name="Congruent Level A", default=1, min=1, max=63,
            description="First of the two levels the Congruent Levels "
                        "preset makes identical")
        congruent_b: IntProperty(
            name="Congruent Level B", default=2, min=1, max=63,
            description="Second of the two levels. The two must satisfy "
                        "A + B > n/2, or no pitch can make them congruent")
        rand_seed: IntProperty(name="Random Seed", default=1, min=0,
                               description="Seed for the random star "
                                           "vectors")

        # ---- star from the active object -----------------------------
        source: EnumProperty(
            name="Zones From", items=SOURCES, default='VERTS',
            description="Which of the object's directions become zones, "
                        "matching Antiprism's -m methods")
        centre_mode: EnumProperty(
            name="Measured From", items=CENTRES, default='ORIGIN',
            description="What the vertex directions are measured from "
                        "(Antiprism's -c); only used by the Vertices "
                        "method")
        unit_length: BoolProperty(
            name="Unit Zones", default=True,
            description="Normalise every zone to unit length "
                        "(Antiprism's -u). Turning it off keeps the "
                        "object's own lengths, which changes the solid "
                        "and not just its scale")

        # ---- polar variations ----------------------------------------
        ellipticity: FloatProperty(
            name="Ellipticity", default=1.0, min=0.1, max=5.0,
            description="Scale the star's X coordinate, turning the "
                        "circular footprint into an ellipse. It is a "
                        "linear map, so every face stays planar and every "
                        "pair of opposite edges stays parallel -- the "
                        "rhombi merely open into parallelograms")
        levels: IntProperty(
            name="Levels", default=0, min=0, max=63,
            description="Keep only this many levels of faces, leaving a "
                        "dome (0 = the whole solid)")
        cap_rim: BoolProperty(
            name="Cap Rim", default=False,
            description="Close a truncated dome's zig-zag rim, turning it "
                        "into a bell")
        lobes: IntProperty(
            name="Lobes", default=1, min=1, max=8,
            description="Stack this many copies pole onto pole. The "
                        "surface helices are congruent, so they run "
                        "straight through the seams -- a finite piece of "
                        "the complete polar zonohedron")
        axial_length: FloatProperty(
            name="Axial Zone", default=0.0, min=0.0, max=5.0,
            description="Add one extra zone along the axis, of this "
                        "length (0 = none). It keeps the n-fold symmetry "
                        "but breaks the surface helices; on the 5-fold "
                        "rhombic icosahedron it builds the rhombic "
                        "triacontahedron")

        # ---- zone lengths --------------------------------------------
        zone_profile: EnumProperty(
            name="Zone Lengths", items=ZONE_PROFILES, default='UNIFORM',
            description="Vary the zones' lengths. Unequal zones turn the "
                        "rhombi into general parallelograms")
        zone_spread: FloatProperty(
            name="Length Spread", default=0.4, min=0.0, max=0.95,
            description="How far the zone lengths stray from equal")

        # ---- translation surface -------------------------------------
        ta_n: IntProperty(name="Chain A Steps", default=10, min=2, max=64,
                          description="Vectors in the first chain")
        ta_step: IntProperty(
            name="Chain A Step", default=1, min=1, max=31,
            description="Star-polygon step for the first chain")
        ta_radius: FloatProperty(name="Chain A Radius", default=1.0,
                                 min=0.01, max=10.0,
                                 description="Size of the first chain")
        ta_tilt: FloatProperty(name="Chain A Tilt", default=0.0,
                               min=-90.0, max=90.0,
                               description="Tip the first chain's plane, "
                                           "in degrees")
        tb_kind: EnumProperty(name="Chain B", items=CHAINS,
                              default='POLYGON',
                              description="Shape of the second chain")
        tb_n: IntProperty(name="Chain B Steps", default=6, min=2, max=64,
                          description="Vectors in the second chain")
        tb_step: IntProperty(
            name="Chain B Step", default=1, min=1, max=31,
            description="Star-polygon step for the second chain")
        tb_radius: FloatProperty(name="Chain B Radius", default=0.35,
                                 min=0.01, max=10.0,
                                 description="Size of the second chain")
        tb_tilt: FloatProperty(name="Chain B Tilt", default=90.0,
                               min=-90.0, max=90.0,
                               description="Tip the second chain's plane, "
                                           "in degrees")
        tb_rise: FloatProperty(name="Chain B Rise", default=0.25,
                               min=-2.0, max=2.0,
                               description="Height gained per step when "
                                           "the second chain is a helix")
        tb_turns: FloatProperty(name="Chain B Turns", default=1.0,
                                min=0.1, max=8.0,
                                description="Revolutions the helix makes")

        # ---- rosette --------------------------------------------------
        bites: IntProperty(
            name="Bites", default=0, min=0, max=8,
            description="Take this many wedges out of the rose's rim. "
                        "Towle's bitten zonogons are the flat shadows of "
                        "his spirallohedra; the wedges here are the "
                        "general construction, not his specific "
                        "space-filling pair")
        bite_width: IntProperty(name="Bite Width", default=3, min=1, max=15,
                                description="Rhombi across each bite at "
                                            "the rim")
        bite_depth: IntProperty(name="Bite Depth", default=2, min=1, max=15,
                                description="Rings each bite cuts inward")

        # ---- helices --------------------------------------------------
        helix_hand: EnumProperty(
            name="Handedness",
            items=[('CW', "Clockwise", "One family of surface helices"),
                   ('CCW', "Counter-clockwise", "The mirror family")],
            default='CW',
            description="Which of the two mirror families of surface "
                        "helices to extrude")
        helix_depth: FloatProperty(
            name="Helix Depth", default=0.8, min=0.02, max=5.0,
            description="How far each helix is extruded down the axis, in "
                        "edge lengths. At 1 the extruded faces are "
                        "rhombi; past the rise of one edge the walls "
                        "start to collide")

        # ---- dissection ----------------------------------------------
        explode: FloatProperty(
            name="Explode", default=0.0, min=0.0, max=4.0,
            description="Push the dissection's blocks out along their own "
                        "centres, opening the solid up to show how it is "
                        "built")
        separate: BoolProperty(
            name="Separate Blocks", default=False,
            description="One object per block instead of a single mesh")

        # ---- finish ---------------------------------------------------
        opening: FloatProperty(
            name="Face Opening", default=0.0, min=0.0, max=0.95,
            description="Open a proportional hole in every face, leaving "
                        "a rim. Unlike the Leonardo border this scales "
                        "with the face, so small faces near the poles "
                        "keep the same look as big ones at the equator")
        color: EnumProperty(
            name="Colour By", items=COLORINGS, default='NONE',
            description="What the face colours mean")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size multiplier")

        style: EnumProperty(
            name="Style",
            items=_shell.STYLE_ITEMS + [_net_style.net_enum_item()],
            default='SOLID',
            description="Finish for the shell: solid, Leonardo panels, "
                        "struts, ball-and-stick, wireframe, face "
                        "segments, or an unfolded net")
        __annotations__.update(
            {k: v for k, v in _shell.style_properties().items()
             if k != 'style'})

        # ------------------------------------------------------------------
        # geometry
        # ------------------------------------------------------------------

        def _output(self):
            """The output actually in force.

            The rose and the translation surface have no star, so the
            Output dial is not drawn for them -- but the property keeps
            whatever it was last set to, and silently erroring on a value
            the user cannot see is worse than ignoring it.
            """
            if self.kind not in _STAR_KINDS:
                return 'SOLID'
            return self.output

        def _theta(self):
            """Hart's theta, in degrees above the horizontal."""
            if self.pitch_preset != 'CUSTOM':
                return _pz.preset_theta(
                    self.pitch_preset, self.n,
                    (self.congruent_a, self.congruent_b))
            return _pz.theta_of(self.pitch, self.angle_ref)

        def _object_star(self, context):
            obj = context.active_object
            if obj is None or obj.type != 'MESH':
                raise ValueError(
                    "select a mesh object to read the star from")
            me = obj.data
            V = [list(v.co) for v in me.vertices]
            F = [list(p.vertices) for p in me.polygons]
            if len(V) < 2:
                raise ValueError("that object has too few vertices")
            centre = [0.0, 0.0, 0.0]
            if self.centre_mode == 'CENTROID':
                centre = [sum(v[k] for v in V) / len(V) for k in range(3)]
            elif self.centre_mode == 'CURSOR':
                c = obj.matrix_world.inverted() @ context.scene.cursor.location
                centre = [c[0], c[1], c[2]]
            star = _zt.star_from(V, F, self.source, centre,
                                 self.unit_length)
            if len(star) < 3:
                raise ValueError(
                    "that object gives only %d distinct zone direction(s)"
                    % len(star))
            return star

        def _star(self, context):
            """The star for the current settings, or None for the kinds
            that do not go through one (rosette, translation surface)."""
            theta = self._theta()
            if self.kind in _POLAR_KINDS:
                star = _pz.ribs(self.n, theta, self.ellipticity)
            elif self.kind == 'OBJECT':
                star = self._object_star(context)
            elif self.kind in ('RANDOM',):
                star = star_vectors('RANDOM', n=self.n, seed=self.rand_seed)
            elif self.kind in _STAR_KINDS:
                star = star_vectors(self.kind)
            else:
                return None
            if self.zone_profile != 'UNIFORM' and self.zone_spread > 0.0:
                star = _zt.weighted_star(star, self.zone_profile,
                                         self.zone_spread, self.rand_seed)
            if self.axial_length > 0.0:
                star = axial_star(star, self.axial_length)
            return star

        def _chains(self):
            a = _zt.closed_polygon_chain(self.ta_n, self.ta_step,
                                         self.ta_radius, self.ta_tilt)
            if self.tb_kind == 'HELIX':
                b = _zt.helix_chain(self.tb_n, self.tb_radius,
                                    self.tb_rise, self.tb_turns)
            else:
                b = _zt.closed_polygon_chain(self.tb_n, self.tb_step,
                                             self.tb_radius, self.tb_tilt)
            return a, b

        def _build(self, context):
            """(V, F, keys, note) for everything except the dissection."""
            theta = self._theta()

            if self.kind == 'ROSETTE':
                rv, rf, rings = _zt.bitten_rosette(
                    self.n, self.bites, self.bite_width, self.bite_depth)
                V = [[x, y, 0.0] for (x, y) in rv]
                keys = [(j, ('ring', j)) for j in rings]
                return V, [list(f) for f in rf], keys, \
                    "%d rhombi in %d rings" % (len(rf), len(set(rings)))

            if self.kind == 'TRANSLATION':
                a, b = self._chains()
                V, F = _zt.translation_surface(a, b)
                keys = [(0, ('quad',))] * len(F)
                return V, F, keys, "%d parallelograms" % len(F)

            if self._output() == 'TEMPLATES':
                if self.kind not in _POLAR_KINDS:
                    raise ValueError(
                        "templates are cut per level, so they need a "
                        "polar star")
                V, F, keys, angs, _l = build_templates(
                    self.n, theta, self.opening)
                note = "%d distinct rhombi: %s" % (
                    len(angs), ", ".join("%.2f" % a for a in angs))
                return V, F, keys, note

            if self._output() == 'HELICES':
                if self.kind not in _POLAR_KINDS:
                    raise ValueError(
                        "surface helices are a property of the polar "
                        "family, so they need a polar star")
                star = _pz.ribs(self.n, theta, self.ellipticity)
                V, F = _pz.extruded_helices(star, self.helix_depth,
                                            self.helix_hand)
                keys = [(0, ('helix', i // max(1, self.n - 1)))
                        for i in range(len(F))]
                note = "%d helices, %d parallelograms" % (self.n, len(F))
                if _pz.helix_collides(self.n, theta, self.helix_depth):
                    note += " (walls overlap at this depth)"
                return V, F, keys, note

            if self.kind == 'SPIRAL':
                star = _pz.ribs(self.n, theta, self.ellipticity)
                if gcd(self.n, self.spiral_width) == self.n:
                    raise ValueError(
                        "spiral width must not be a multiple of the "
                        "vector count")
                V, F = _pz.make_polar_zonohedron(star, self.step,
                                                 self.spiral_width)
                V, F = _weld(V, F)
                keys = [(0, ('spiral',))] * len(F)
                parts = _pz.polar_parts(self.n, self.step)
                note = "V=%d F=%d" % (len(V), len(F))
                if parts > 1:
                    note += " in %d lobes" % parts
                return V, F, keys, note

            if self.kind == 'POLAR' and self.step > 1:
                # the offset polygon is Antiprism's; it only changes the
                # solid when the step shares a factor with n, which splits
                # it into that many lobes
                star = _pz.ribs(self.n, theta, self.ellipticity)
                V, F = _pz.make_polar_zonohedron(star, self.step, 0)
                V, F = _weld(V, F)
                keys = [(0, ('step',))] * len(F)
                parts = _pz.polar_parts(self.n, self.step)
                return V, F, keys, "V=%d F=%d in %d lobe(s)" % (
                    len(V), len(F), parts)

            if self.kind == 'POLAR':
                V, F, keys = build_polar(
                    self.n, theta, self.ellipticity, self.levels,
                    self.cap_rim, self.lobes, self.zone_profile,
                    self.zone_spread, self.rand_seed, self.axial_length)
                return V, F, keys, "V=%d F=%d" % (len(V), len(F))

            star = self._star(context)
            V, F, keys = build_general(star)
            return V, F, keys, "%d zones, V=%d F=%d" % (
                len(star), len(V), len(F))

        # ------------------------------------------------------------------
        # colouring
        # ------------------------------------------------------------------

        def _colors(self, V, F, keys):
            if self.color == 'NONE' or not F:
                return [], None
            if self.color == 'SIDES':
                sel = [len(f) for f in F]
            elif self.color == 'LEVEL':
                sel = [k[0] for k in keys]
            else:                                # ZONES, RING, and the
                sel = [k[1] for k in keys]       # per-kind labels
            return _fc.materials_for(sel, "Zonohedron")

        # ------------------------------------------------------------------
        # execute
        # ------------------------------------------------------------------

        def execute(self, context):
            try:
                if self._output() == 'DISSECTION':
                    return self._dissection(context)
                V, F, keys, note = self._build(context)
            except (ValueError, RuntimeError) as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if not F:
                self.report({'ERROR'}, "that combination produced no faces")
                return {'CANCELLED'}

            if self.opening > 0.0 and self._output() != 'TEMPLATES':
                V, F, src = open_faces(V, F, self.opening)
                keys = [keys[i] for i in src]

            V = _fit.fit_cube(V, 2.0 * self.scale)
            name = "Zonohedron"
            if self._output() == 'TEMPLATES':
                name = "Zonohedron Templates"
            elif self._output() == 'HELICES':
                name = "Zonohedron Helices"
            elif self.kind == 'ROSETTE':
                name = "Rhombic Rose"
            elif self.kind == 'TRANSLATION':
                name = "Translation Surface"

            if self.style == 'NET':
                hint = None
                if self.kind == 'ROSETTE' or self._output() == 'TEMPLATES':
                    hint = ("this output is already a flat sheet; pick a "
                            "three-dimensional star")
                return _net_style.emit_net_from_operator(
                    self, context, [tuple(v) for v in V],
                    [list(f) for f in F], name, hint=hint)

            mats, midx = self._colors(V, F, keys)
            obj = _shell.apply(self, context, V, F, name,
                               materials=mats, material_index=midx)
            self.report({'INFO'}, note)
            return {'FINISHED'}

        def _dissection(self, context):
            star = self._star(context)
            if star is None:
                self.report({'ERROR'},
                            "that construction has no star to dissect")
                return {'CANCELLED'}
            try:
                blocks = build_dissection(star, self.explode)
            except (ValueError, RuntimeError) as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # one common fit for every block, so the assembly stays a solid
            allv = [v for V, _F, _k, _t in blocks for v in V]
            fitted = _fit.fit_cube(allv, 2.0 * self.scale)
            it = iter(fitted)
            blocks = [([next(it) for _ in V], F, k, t)
                      for V, F, k, t in blocks]

            want = self.color != 'NONE'
            mats, order = [], {}
            if want:
                triples = sorted({t for _V, _F, _k, t in blocks})
                order = {t: i for i, t in enumerate(triples)}
                mats = [_fc.material("Zone Block %d" % i,
                                     _fc.PALETTE[i % len(_fc.PALETTE)])
                        for i in range(len(triples))]
            made = []
            if self.separate:
                for i, (V, F, kind, t) in enumerate(blocks):
                    me = _mesh_from("Block %d" % i, V, F)
                    if want:
                        me.materials.append(mats[order[t]])
                    made.append(bpy.data.objects.new("Block %d" % i, me))
            else:
                V, F, mat_of = [], [], []
                for Vb, Fb, kind, t in blocks:
                    off = len(V)
                    V += Vb
                    F += [[i + off for i in f] for f in Fb]
                    mat_of += [order[t] if want else 0] * len(Fb)
                me = _mesh_from("Dissection", V, F)
                for m in mats:
                    me.materials.append(m)
                if want and me.materials:
                    me.polygons.foreach_set('material_index', mat_of)
                    me.update()
                made.append(bpy.data.objects.new("Dissection", me))
            for o in context.selected_objects:
                o.select_set(False)
            for o in made:
                context.collection.objects.link(o)
                o.location = context.scene.cursor.location
                o.select_set(True)
            context.view_layer.objects.active = made[0]
            acute = sum(1 for _V, _F, k, _t in blocks if k == 'ACUTE')
            flat = sum(1 for _V, _F, k, _t in blocks if k == 'FLAT')
            self.report({'INFO'},
                        "%d blocks (%d acute, %d flat)"
                        % (len(blocks), acute, flat))
            return {'FINISHED'}

        # ------------------------------------------------------------------
        # UI
        # ------------------------------------------------------------------

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            polar = self.kind in _POLAR_KINDS
            lay.prop(self, 'kind')
            if self.kind in _STAR_KINDS:
                lay.prop(self, 'output')

            if self.kind == 'OBJECT':
                lay.prop(self, 'source')
                if self.source == 'VERTS':
                    lay.prop(self, 'centre_mode')
                lay.prop(self, 'unit_length')

            if self.kind == 'TRANSLATION':
                box = lay.box()
                box.prop(self, 'ta_n')
                box.prop(self, 'ta_step')
                box.prop(self, 'ta_radius')
                box.prop(self, 'ta_tilt')
                box = lay.box()
                box.prop(self, 'tb_kind')
                box.prop(self, 'tb_n')
                if self.tb_kind == 'POLYGON':
                    box.prop(self, 'tb_step')
                box.prop(self, 'tb_radius')
                if self.tb_kind == 'POLYGON':
                    box.prop(self, 'tb_tilt')
                else:
                    box.prop(self, 'tb_rise')
                    box.prop(self, 'tb_turns')

            if self.kind in ('POLAR', 'SPIRAL', 'RANDOM', 'ROSETTE'):
                lay.prop(self, 'n')
            if polar:
                lay.prop(self, 'step')
            if self.kind == 'SPIRAL':
                lay.prop(self, 'spiral_width')
            if polar:
                lay.prop(self, 'pitch_preset')
                if self.pitch_preset == 'CUSTOM':
                    lay.prop(self, 'pitch')
                    lay.prop(self, 'angle_ref')
                elif self.pitch_preset == 'CONGRUENT':
                    lay.prop(self, 'congruent_a')
                    lay.prop(self, 'congruent_b')
            if self.kind == 'RANDOM' or \
                    (self.zone_profile == 'RANDOM'):
                lay.prop(self, 'rand_seed')

            if self.kind == 'ROSETTE':
                lay.prop(self, 'bites')
                if self.bites:
                    lay.prop(self, 'bite_width')
                    lay.prop(self, 'bite_depth')

            if polar and self.output in ('SOLID', 'HELICES', 'TEMPLATES'):
                lay.prop(self, 'ellipticity')
            if polar and self.output == 'SOLID' and self.step == 1:
                lay.prop(self, 'levels')
                if self.levels:
                    lay.prop(self, 'cap_rim')
                lay.prop(self, 'lobes')
            if self.output == 'HELICES':
                lay.prop(self, 'helix_hand')
                lay.prop(self, 'helix_depth')
            if self.kind in _STAR_KINDS and self.output != 'TEMPLATES':
                lay.prop(self, 'axial_length')
                lay.prop(self, 'zone_profile')
                if self.zone_profile != 'UNIFORM':
                    lay.prop(self, 'zone_spread')

            if self.output == 'DISSECTION':
                lay.prop(self, 'explode')
                lay.prop(self, 'separate')
            else:
                lay.prop(self, 'opening')
                if self.style == 'NET':
                    # NET is this operator's own addition to the shared
                    # style enum, so shell.draw_style knows nothing about
                    # its properties -- but it DOES draw the enum itself,
                    # so the dial is drawn here once and only its
                    # sub-properties are delegated.
                    lay.prop(self, 'style')
                    _net_style.draw_net_props(lay, self)
                else:
                    _shell.draw_style(self, lay)
            lay.prop(self, 'color')
            lay.prop(self, 'scale')

    def _mesh_from(name, V, F):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
        me.update()
        return me

    def _menu_func(self, context):
        self.layout.operator("mesh.zonohedron_add", icon='MESH_UVSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_zonohedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_zonohedron_add)


def _selftest():
    def close(a, b, t=1e-7):
        return abs(a - b) <= t * max(1.0, abs(a), abs(b))

    # --- the polar builder, with and without its variations ------------
    for n in (5, 8, 12):
        V, F, keys = build_polar(n, 40.0)
        assert len(F) == n * (n - 1), (n, len(F))
        assert len(keys) == len(F)
        assert sorted({k[0] for k in keys}) == list(range(1, n))

    # truncation leaves a dome of n faces per level; capping adds the fan
    V, F, keys = build_polar(9, 40.0, levels=3)
    assert len(F) == 27, len(F)
    V, F, keys = build_polar(9, 40.0, levels=3, cap_rim=True)
    assert len(F) == 27 + 18, len(F)

    # lobes repeat the solid and its labels
    V1, F1, K1 = build_polar(7, 40.0)
    V3, F3, K3 = build_polar(7, 40.0, lobes=3)
    assert len(F3) == 3 * len(F1) and len(V3) == 3 * len(V1)
    assert K3[:len(K1)] == K1

    # --- Fedorov's rhombic icosahedron and Kepler's triacontahedron ----
    th5 = _pz.theta_congruent(5, 1, 2)
    V, F, _k = build_polar(5, th5)
    assert len(F) == 20, len(F)
    golden = math.degrees(math.acos(1.0 / math.sqrt(5.0)))
    for f in F:
        o, a, b = V[f[0]], V[f[1]], V[f[3]]
        u = _unit([a[i] - o[i] for i in range(3)])
        w = _unit([b[i] - o[i] for i in range(3)])
        ang = math.degrees(math.acos(max(-1.0, min(1.0,
                                                   sum(u[i] * w[i]
                                                       for i in range(3))))))
        assert close(min(ang, 180.0 - ang), golden, 1e-6), ang
    # the axial zone completes it to the rhombic triacontahedron: 30
    # faces, and still all golden rhombi
    star = axial_star(_pz.ribs(5, th5), 1.0)
    V, F, _k = build_general(star)
    assert len(F) == 30, len(F)
    assert len(V) == 32, len(V)
    for f in F:
        assert len(f) == 4, len(f)
        o, a, b = V[f[0]], V[f[1]], V[f[3]]
        u = _unit([a[i] - o[i] for i in range(3)])
        w = _unit([b[i] - o[i] for i in range(3)])
        ang = math.degrees(math.acos(max(-1.0, min(1.0,
                                                   sum(u[i] * w[i]
                                                       for i in range(3))))))
        assert close(min(ang, 180.0 - ang), golden, 1e-6), ang

    # --- volume-optimal preset -----------------------------------------
    V, F, _k = build_polar(3, _pz.THETA_VOLUME)
    assert (len(V), len(F)) == (8, 6), (len(V), len(F))
    e = set()
    for f in F:
        for k in range(4):
            d = [V[f[(k + 1) % 4]][i] - V[f[k]][i] for i in range(3)]
            e.add(round(math.sqrt(sum(x * x for x in d)), 9))
    assert e == {1.0}, e                       # a cube, not a rhombohedron
    V, F, _k = build_polar(4, _pz.THETA_VOLUME)
    assert len(F) == 12, len(F)                # rhombic dodecahedron

    # --- zone lengths ---------------------------------------------------
    V, F, _k = build_polar(7, 40.0, zone_profile='RAMP', zone_spread=0.5)
    assert len(F) == 7 * 6, len(F)
    lens = set()
    for f in F:
        for k in range(len(f)):
            d = [V[f[(k + 1) % len(f)]][i] - V[f[k]][i] for i in range(3)]
            lens.add(round(math.sqrt(sum(x * x for x in d)), 5))
    assert len(lens) > 1, lens                 # parallelograms, not rhombi

    # --- face opening ----------------------------------------------------
    V, F, keys = build_polar(6, 40.0)
    V2, F2, src = open_faces(V, F, 0.5)
    assert len(F2) == 4 * len(F), (len(F2), len(F))
    assert len(src) == len(F2) and max(src) == len(F) - 1
    # the rim's outer ring is the original face
    for fi in range(len(F)):
        sub = [F2[i] for i in range(len(F2)) if src[i] == fi]
        assert len(sub) == 4
    # opening 0 is a no-op, not a degenerate rim
    V3, F3, s3 = open_faces(V, F, 0.0)
    assert len(F3) == len(F) and s3 == list(range(len(F)))

    # --- templates -------------------------------------------------------
    V, F, keys, angs, lvls = build_templates(11, 38.0)
    assert len(F) == 5 + 10, len(F)            # 5 distinct + 10 chain cells
    assert len(angs) == 5 and lvls == [1, 2, 3, 4, 5]
    assert all(abs(v[2]) < 1e-12 for v in V)   # flat, ready to cut

    # --- dissection ------------------------------------------------------
    blocks = build_dissection(star_vectors('TRIACONTA'))
    assert len(blocks) == 20, len(blocks)
    assert sum(1 for _V, _F, k, _t in blocks if k == 'ACUTE') == 10
    assert sum(1 for _V, _F, k, _t in blocks if k == 'FLAT') == 10
    try:
        build_dissection(_pz.ribs(40, 40.0))
    except ValueError:
        pass
    else:
        raise AssertionError("dissection accepted an unusable block count")

    # --- the named stars still give the named solids ---------------------
    for kind, nf in (('CUBE', 6), ('RHOMBIC_DODECA', 12),
                     ('TRIACONTA', 30), ('ENNEACONTA', 90)):
        V, F, _k = build_general(star_vectors(kind))
        assert len(F) == nf, (kind, len(F))
        lo = [min(v[i] for v in V) for i in range(3)]
        hi = [max(v[i] for v in V) for i in range(3)]
        span = [hi[i] - lo[i] for i in range(3)]
        # gate on shape, not just counts: a collapsed solid passes a face
        # count and fails this
        assert min(span) > 0.3 * max(span), (kind, span)

    # --- the re-export the space-filling generator relies on -------------
    assert make_polar_zonohedron is _pz.make_polar_zonohedron
    Vp, Fp = make_polar_zonohedron(polar_star(12, 55.0), 1, 4)
    assert len(Fp) == 96, len(Fp)

    print("RESULT: OK")
