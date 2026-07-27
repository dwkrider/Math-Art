
# Woven Polyhedron generator for Blender.
#
# A dual-polyhedron sculpture generalized to any Platonic or
# Archimedean solid: an inset polygon sits on every face of the chosen
# OUTER solid, and an inset polygon on every face of its dual INNER
# solid (the dual of a Platonic solid is another Platonic; the dual of
# an Archimedean solid is its Catalan solid).  The inner faces align
# with the outer solid's vertices.  A twisted bezier ribbon bridges
# each outer-polygon edge to one inner-polygon edge, and the whole
# thing is welded into ONE closed shell so a Solidify + Subdivision
# pair can thicken it and round the corners.
#
#   OUTER      INNER (dual)   outer faces   inner polygons (vertex
#   solid                     (n-gons)      degree m)
#   ------     ------------   -----------   ---------------------
#   tetra      tetrahedron    triangles     triangles (m=3)
#   cube       octahedron     squares       triangles (m=3)
#   octa       cube           triangles     squares    (m=4)
#   dodeca     icosahedron    pentagons     triangles  (m=3)
#   icosa      dodecahedron   triangles     pentagons  (m=5)
#
# Each outer polygon fans out one ribbon per edge that swirls to a
# neighbouring inner polygon.  The outer-edge -> inner-edge pairing is
# topological (rotation-proof): outer face f edge (v_i -> v_i+1) binds
# to the inner polygon at the forward vertex v_i+1, at edge index
# (j + step) % m, where j is f's position in the ring of faces around
# that vertex and step = round(inner_spin / (360/m)) folds the pairing
# with the inner polygon's m-fold symmetry.  This is a clean bijection
# (every inner edge receives exactly one ribbon) for every solid and
# at every spin angle.

bl_info = {
    "name": "Twisted Polyhedron",
    "author": "Math Art project",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Twisted Polyhedron",
    "description": "Ribbons weaving a Platonic or Archimedean solid "
                   "to its dual",
    "category": "Add Mesh",
}

import math

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix
    from bpy.props import (FloatProperty, IntProperty, BoolProperty,
                           EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


PHI = (1 + 5 ** 0.5) / 2

# icosahedron (seed for the icosa / dodeca pair) ...
IV = [(-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0),
      (0, -1, PHI), (0, 1, PHI), (0, -1, -PHI), (0, 1, -PHI),
      (PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)]
IF = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
      (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
      (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
      (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]

_PLATONIC_ITEMS = [
    ('TETRA', "Tetrahedron", "Self-dual: triangles woven to "
                             "triangles"),
    ('CUBE', "Cube", "Cube woven to its dual octahedron"),
    ('OCTA', "Octahedron", "Octahedron woven to its dual cube"),
    ('DODECA', "Dodecahedron", "Dodecahedron woven to its dual "
                               "icosahedron"),
    ('ICOSA', "Icosahedron", "Icosahedron woven to its dual "
                             "dodecahedron"),
]

PLATONIC = {sid for sid, _, _ in _PLATONIC_ITEMS}

# the 13 Archimedean solids (id, label); geometry comes from the
# regular_solids_generator's Conway builder, the dual (inner) solid
# is the corresponding Catalan solid.  All are vertex-transitive, so
# every inner polygon has the same number of sides (the vertex
# degree) and the topological ribbon pairing stays a clean bijection.
ARCHIMEDEAN = [
    ('TT', "Truncated Tetrahedron"),
    ('CO', "Cuboctahedron"),
    ('TC', "Truncated Cube"),
    ('TO', "Truncated Octahedron"),
    ('RCO', "Rhombicuboctahedron"),
    ('TCO', "Truncated Cuboctahedron"),
    ('SC', "Snub Cube"),
    ('ID', "Icosidodecahedron"),
    ('TD', "Truncated Dodecahedron"),
    ('TI', "Truncated Icosahedron"),
    ('RID', "Rhombicosidodecahedron"),
    ('TID', "Truncated Icosidodecahedron"),
    ('SD', "Snub Dodecahedron"),
]

SOLID_ITEMS = _PLATONIC_ITEMS + [
    (sid, label, "%s woven to its Catalan dual" % label)
    for sid, label in ARCHIMEDEAN] + [
    ('GEODESIC', "Geodesic Sphere",
     "Icosahedral geodesic sphere (set Frequency) woven to its "
     "Goldberg dual of hexagons and 12 pentagons")]

# Per-solid default parameters, loaded when the Outer Solid selector
# changes.  Each solid's few large or many small faces want their own
# sizes / spins / curvature, so one shared default cannot suit them
# all.  DODECA also supplies the operator's property defaults (below).
_PRESETS = {
    'TETRA': dict(tri_rot=90.0, pen_rot=-31.29, pen_inset=0.12,
                  tri_inset=0.12, r_ico=0.64, pen_curve=0.06,
                  tri_curve=0.0, handle_pen=1.86, handle_tri=0.40,
                  mid_width=0.29, width_pen=50.65, width_tri=0.0,
                  rib_samples=14),
    'CUBE': dict(tri_rot=90.0, pen_rot=-31.29, pen_inset=0.12,
                 tri_inset=0.21, r_ico=0.64, pen_curve=1.00,
                 tri_curve=0.0, handle_pen=1.86, handle_tri=0.40,
                 mid_width=0.25, width_pen=50.65, width_tri=0.0,
                 rib_samples=14),
    'OCTA': dict(tri_rot=67.98, pen_rot=-30.00, pen_inset=0.12,
                 tri_inset=0.38, r_ico=0.55, pen_curve=0.40,
                 tri_curve=0.0, handle_pen=1.86, handle_tri=0.40,
                 mid_width=0.25, width_pen=50.65, width_tri=0.0,
                 rib_samples=14),
    'DODECA': dict(tri_rot=90.0, pen_rot=-18.66, pen_inset=0.32,
                   tri_inset=0.15, r_ico=0.76, pen_curve=1.05,
                   tri_curve=0.0, handle_pen=1.5, handle_tri=0.40,
                   mid_width=0.29, width_pen=50.65, width_tri=0.0,
                   rib_samples=14),
    'ICOSA': dict(tri_rot=67.98, pen_rot=-23.85, pen_inset=0.21,
                  tri_inset=0.38, r_ico=0.55, pen_curve=0.40,
                  tri_curve=0.0, handle_pen=1.71, handle_tri=0.40,
                  mid_width=0.25, width_pen=50.65, width_tri=0.0,
                  rib_samples=14),
}

# Archimedean solids share a provisional default (they carry many
# small faces, so modest insets, low curvature and a slim waist read
# well); tune per solid the same way as the Platonic ones.
_ARCH_DEFAULT = dict(tri_rot=67.98, pen_rot=-25.0, pen_inset=0.35,
                     tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                     tri_curve=0.0, handle_pen=1.6, handle_tri=0.40,
                     mid_width=0.28, width_pen=50.65, width_tri=0.0,
                     rib_samples=14)
for _sid, _ in ARCHIMEDEAN:
    _PRESETS.setdefault(_sid, dict(_ARCH_DEFAULT))

_PRESETS['TT'] = dict(tri_rot=56.49, pen_rot=-21.85, pen_inset=0.35,
                      tri_inset=0.23, r_ico=0.72, pen_curve=0.0,
                      tri_curve=0.0, handle_pen=1.60, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['CO'] = dict(tri_rot=67.98, pen_rot=-34.75, pen_inset=0.20,
                      tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                      tri_curve=0.0, handle_pen=1.60, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['TC'] = dict(tri_rot=35.22, pen_rot=-13.42, pen_inset=0.23,
                      tri_inset=0.14, r_ico=0.87, pen_curve=0.30,
                      tri_curve=0.0, handle_pen=0.46, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['TO'] = dict(tri_rot=98.34, pen_rot=-13.36, pen_inset=0.35,
                      tri_inset=0.23, r_ico=0.57, pen_curve=0.30,
                      tri_curve=0.0, handle_pen=1.54, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['RCO'] = dict(tri_rot=67.98, pen_rot=-25.00, pen_inset=0.35,
                       tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                       tri_curve=0.0, handle_pen=1.06, handle_tri=0.40,
                       mid_width=0.28, width_pen=50.65, width_tri=0.0,
                       rib_samples=14)
_PRESETS['TCO'] = dict(tri_rot=163.98, pen_rot=-17.98, pen_inset=0.20,
                       tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                       tri_curve=0.0, handle_pen=1.00, handle_tri=0.40,
                       mid_width=0.28, width_pen=50.65, width_tri=0.0,
                       rib_samples=14)
_PRESETS['SC'] = dict(tri_rot=-10.00, pen_rot=-20.00, pen_inset=0.22,
                      tri_inset=0.35, r_ico=0.60, pen_curve=0.03,
                      tri_curve=0.0, handle_pen=1.24, handle_tri=0.40,
                      mid_width=0.35, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['ID'] = dict(tri_rot=70.68, pen_rot=-25.00, pen_inset=0.35,
                      tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                      tri_curve=0.0, handle_pen=0.97, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['TD'] = dict(tri_rot=42.72, pen_rot=-29.26, pen_inset=0.35,
                      tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                      tri_curve=0.0, handle_pen=1.60, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['TI'] = dict(tri_rot=93.39, pen_rot=-52.63, pen_inset=0.35,
                      tri_inset=0.17, r_ico=0.60, pen_curve=0.30,
                      tri_curve=0.0, handle_pen=1.27, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)
_PRESETS['RID'] = dict(tri_rot=72.06, pen_rot=-15.16, pen_inset=0.35,
                       tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                       tri_curve=0.0, handle_pen=0.73, handle_tri=0.40,
                       mid_width=0.28, width_pen=50.65, width_tri=0.0,
                       rib_samples=14)
_PRESETS['TID'] = dict(tri_rot=67.98, pen_rot=-37.00, pen_inset=0.25,
                       tri_inset=0.35, r_ico=0.60, pen_curve=0.30,
                       tri_curve=0.0, handle_pen=1.39, handle_tri=0.40,
                       mid_width=0.28, width_pen=50.65, width_tri=0.0,
                       rib_samples=14)
_PRESETS['SD'] = dict(tri_rot=67.98, pen_rot=-30.00, pen_inset=0.20,
                      tri_inset=0.35, r_ico=0.60, pen_curve=0.00,
                      tri_curve=0.0, handle_pen=1.60, handle_tri=0.40,
                      mid_width=0.28, width_pen=50.65, width_tri=0.0,
                      rib_samples=14)

# geodesic sphere: triangular outer faces with degree-5/6 vertices,
# so the inner (Goldberg) polygons are pentagons and hexagons.  Small
# faces at higher frequency want modest insets.
_PRESETS['GEODESIC'] = dict(frequency=2, tri_rot=67.98,
                            pen_rot=-25.00, pen_inset=0.27,
                            tri_inset=0.40, r_ico=1.20,
                            pen_curve=0.35, tri_curve=0.0,
                            handle_pen=1.60, handle_tri=0.40,
                            mid_width=0.28, width_pen=50.65,
                            width_tri=0.0, rib_samples=14,
                            smooth_shading=False)


# ---------------------------------------------------------------- #
#  polyhedron + dual construction                                  #
# ---------------------------------------------------------------- #

def _orient_faces(V, F):
    """Reorder each face's vertices CCW as seen from outside (the
    face normal is the outward centroid direction), so edge traversal
    and the swirl are consistent everywhere."""
    out = []
    for f in F:
        pts = [V[i] for i in f]
        c = sum(pts, Vector()) / len(pts)
        n = c.normalized() if c.length > 1e-9 else Vector((0, 0, 1))
        ref = V[f[0]] - c
        ref = (ref - n * ref.dot(n)).normalized()
        sid = n.cross(ref)
        idx = sorted(f, key=lambda i: math.atan2(
            (V[i] - c).dot(sid), (V[i] - c).dot(ref)))
        out.append(tuple(idx))
    return out


def _dual_data(V, F):
    """Dual of the (normalized) solid (V, F): one dual vertex per
    face (the unit face centroid) and, per solid vertex, the ring of
    incident faces ordered CCW -- that ring is the dual face sitting
    at the vertex."""
    Vd = [(sum((V[i] for i in f), Vector()) / len(f)).normalized()
          for f in F]
    rings = []
    for v in range(len(V)):
        inc = [fi for fi, f in enumerate(F) if v in f]
        n = V[v].normalized()
        ref = Vd[inc[0]] - n * Vd[inc[0]].dot(n)
        ref = ref.normalized()
        sid = n.cross(ref)
        inc.sort(key=lambda fi: math.atan2(Vd[fi].dot(sid),
                                           Vd[fi].dot(ref)))
        rings.append(inc)
    return Vd, rings


def _base(name, freq=2):
    """Raw (verts, faces) for the chosen outer solid, centered at the
    origin (faces as index tuples, winding fixed later)."""
    if name == 'ICOSA':
        return [Vector(v) for v in IV], [tuple(f) for f in IF]
    if name == 'DODECA':
        Vp = [Vector(v).normalized() for v in IV]
        Fp = _orient_faces(Vp, [tuple(f) for f in IF])
        Vd, rings = _dual_data(Vp, Fp)
        return [v.copy() for v in Vd], [tuple(r) for r in rings]
    if name == 'TETRA':
        V = [Vector(v) for v in ((1, 1, 1), (1, -1, -1),
                                 (-1, 1, -1), (-1, -1, 1))]
        return V, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    if name == 'CUBE':
        V = [Vector((x, y, z)) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
        F = [tuple(i for i, v in enumerate(V) if v[ax] == sg)
             for ax in range(3) for sg in (-1, 1)]
        return V, F
    if name == 'OCTA':
        V = [Vector(v) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                 (0, -1, 0), (0, 0, 1), (0, 0, -1))]
        F = [(0 if sx > 0 else 1, 2 if sy > 0 else 3,
              4 if sz > 0 else 5)
             for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
        return V, F
    if name == 'GEODESIC':
        # icosahedral Class-I geodesic sphere (triangles); its dual is
        # the Goldberg solid of hexagons + 12 pentagons
        try:
            from . import geodesic_generator as geo
        except ImportError:
            import geodesic_generator as geo
        V, F = geo.build_sphere('ICOSA', max(1, int(freq)), 'I')
        return [Vector(v) for v in V], [tuple(f) for f in F]
    # Archimedean: reuse the extension's canonicalized Conway geometry
    try:
        from . import regular_solids_generator as rsg
    except ImportError:
        import regular_solids_generator as rsg
    V, F, _ = rsg.build_solid('ARCHIMEDEAN', name)
    return [Vector(v) for v in V], [tuple(f) for f in F]


def solid(name, freq=2):
    """(Vp, Fp, Vd, rings) for the chosen outer solid: unit-radius
    verts, CCW faces, dual verts (one per face) and vertex rings (the
    dual faces).  Vertex degree (= inner-polygon sides) is read per
    vertex from the ring lengths, so mixed-degree solids (geodesic,
    Goldberg) also work.  `freq` is the geodesic breakdown."""
    V, F = _base(name, freq)
    Vp = [v.normalized() for v in V]
    # scale so the outer solid just fills a 2 m cube centered at the
    # origin (its extreme vertex touches a cube face: max |coord| = 1)
    maxc = max(max(abs(c) for c in v) for v in Vp) or 1.0
    fill = 1.0 / maxc
    Vp = [v * fill for v in Vp]
    Fp = _orient_faces(Vp, F)
    Vd, rings = _dual_data(Vp, Fp)
    Vd = [v * fill for v in Vd]        # keep the dual in the same scale
    return Vp, Fp, Vd, rings


# ---------------------------------------------------------------- #
#  geometry helpers                                                #
# ---------------------------------------------------------------- #

def _fd(vs):
    """face-data dict {verts, center, normal} for a polygon."""
    c = sum(vs, Vector()) / len(vs)
    nrm = (vs[1] - vs[0]).cross(vs[2] - vs[0]).normalized()
    return {"verts": list(vs), "center": c, "normal": nrm}


def inset_faces(faces, factor):
    res = []
    for fd in faces:
        c = fd["center"]
        vs = [c + factor * (v - c) for v in fd["verts"]]
        res.append({"verts": vs, "center": c,
                    "normal": fd["normal"]})
    return res


def spin_faces(faces, deg):
    a = math.radians(deg)
    for fd in faces:
        c = fd["center"]
        axis = fd["normal"].normalized()
        if axis.dot(c) < 0:
            axis = -axis
        R = Matrix.Rotation(-a, 4, axis)   # -a about outward = CCW
        fd["verts"] = [c + (R @ (v - c)) for v in fd["verts"]]


def bez(cp, t):
    u = 1 - t
    return (u * u * u * cp[0] + 3 * u * u * t * cp[1]
            + 3 * u * t * t * cp[2] + t * t * t * cp[3])


def wprofile(w0, w1, mid, ap, at, t):
    """half-width w0->w1 in two eased halves meeting at the middle
    width.  `mid` = middle width as a fraction of the endpoint SUM
    (0.5 = the linear midpoint, <0.5 a waist thinner than both ends,
    >0.5 a bulge).  ap/at = taper: 0 is a straight linear taper to
    the middle width; higher reaches it sooner and holds it flat.
    Endpoints stay exact."""
    wmid = mid * (w0 + w1)
    if t <= 0.5:
        s = 2.0 * t                        # 0..1 over the outer half
        we, k = w0, ap
    else:
        s = 2.0 * (1.0 - t)                # 1..0 over the inner half
        we, k = w1, at
    e = 1.0 - max(0.0, 1.0 - s) ** (1.0 + k)   # ease: linear at k=0
    return max(0.0, we + (wmid - we) * e)


def cap_faces(fd, curve):
    """(verts, faces): fill an inset face with a subdivision-friendly
    cap that has NO high-valence fan pole (a pole makes Catmull-Clark
    shade the cap with a seam or pinch).  The interior is blended by
    `curve` toward the sphere through the corners (0 = flat); the
    corners stay put so the ribbon weld is preserved.

    A triangle cannot be filled with quads without an extraordinary
    vertex, so n=3 uses a 3-triangle fan to a domed centre -- a clean
    valence-3 apex (like a cube corner) that subdivides smoothly.  For
    n>=4 an outer band of quads around one central n-gon avoids a pole
    entirely."""
    vs = fd["verts"]
    c = fd["center"]
    n = len(vs)
    r = vs[0].length                   # corner distance = sphere R

    def blend(p):
        return p.lerp(p.normalized() * r, curve)

    corners = [blend(v) for v in vs]                     # 0 .. n-1
    if n == 3:
        verts = corners + [blend(c)]                     # domed apex
        return verts, [[i, (i + 1) % 3, 3] for i in range(3)]
    ring = [blend((vs[i] + c) * 0.5) for i in range(n)]  # n .. 2n-1
    verts = corners + ring
    # outer band: one quad per edge (boundary edges stay single, so
    # the ribbon still welds to them); then one n-gon across the ring
    faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i]
             for i in range(n)]
    faces.append([n + i for i in range(n)])
    return verts, faces


def _edge_tangents(fd, curve):
    """per edge: (midpoint, outward tangent in the CURVED surface's
    tangent plane, curved surface normal).  The tangent lies in the
    face's tangent plane at the edge so a ribbon leaving along it
    shares a normal with the (possibly domed) face -- no crease."""
    vs, c = fd["verts"], fd["center"]
    n = len(vs)
    r = vs[0].length
    mids = [((vs[i] + c) * 0.5).lerp(
                ((vs[i] + c) * 0.5).normalized() * r, curve)
            for i in range(n)]
    out = []
    for i in range(n):
        a, b = vs[i], vs[(i + 1) % n]
        mmid = (mids[i] + mids[(i + 1) % n]) * 0.5
        m = (a + b) / 2
        e = (b - a).normalized()
        nsurf = (b - a).cross(mmid - m)     # curved-surface normal
        if nsurf.dot(m) < 0:
            nsurf = -nsurf
        nsurf.normalize()
        t = e.cross(nsurf).normalized()
        if t.dot(m - c) < 0:
            t = -t
        out.append((m, t, nsurf))
    return out


def _poly_edges(fd, curve):
    """list of {m, t, n, a, b} edge records for a capped face."""
    vs = fd["verts"]
    et = _edge_tangents(fd, curve)
    return [{"m": et[i][0], "t": et[i][1], "n": et[i][2],
             "a": vs[i], "b": vs[(i + 1) % len(vs)]}
            for i in range(len(vs))]


def build(p, swap="dist"):
    """Return (verts, faces) for the woven sculpture.  Everything
    accumulates into ONE welded vertex pool: the ribbon ends coincide
    with the outer / inner polygon edge verts, so they fuse and the
    Subdivision Surface smooths across the junctions instead of
    treating the parts as loose pieces.

    `swap` picks each ribbon's rail pairing: "dist" is the geometric
    non-crossing choice (looks best, but on a few solids -- truncated
    tetrahedron, cuboctahedron, icosidodecahedron -- it is globally
    inconsistent and makes the shell non-orientable); "fwd" is a
    consistent pairing that is always orientable.  The operator builds
    with "dist" and only falls back to "fwd" when the result comes out
    non-orientable, so every other solid keeps the nicer pairing."""
    Vp, Fp, Vd, rings = solid(p.solid, getattr(p, "frequency", 2))
    r_in = p.r_ico
    # outer inset polygons, one per face; inner inset polygons, one
    # per vertex (the dual face there = the ring of face-centroids)
    outer = inset_faces([_fd([Vp[i] for i in f]) for f in Fp],
                        p.pen_inset)
    inner = inset_faces([_fd([Vd[fi] * r_in for fi in rings[v]])
                         for v in range(len(Vp))], p.tri_inset)
    spin_faces(inner, p.tri_rot)        # + = CCW
    spin_faces(outer, -p.pen_rot)       # outer spin + = CW

    verts, faces = [], []
    vid = {}

    def add(v):
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        i = vid.get(key)
        if i is None:
            i = len(verts)
            vid[key] = i
            verts.append((v[0], v[1], v[2]))
        return i

    for fd in outer:
        cv, tr = cap_faces(fd, p.pen_curve)
        idx = [add(v) for v in cv]
        for f in tr:
            faces.append([idx[i] for i in f])
    for fd in inner:
        cv, tr = cap_faces(fd, p.tri_curve)
        idx = [add(v) for v in cv]
        for f in tr:
            faces.append([idx[i] for i in f])

    inner_edges = [_poly_edges(inner[v], p.tri_curve)
                   for v in range(len(Vp))]
    N = p.rib_samples
    # the inner polygon (at the forward vertex) has m = that vertex's
    # degree sides and is m-fold symmetric, so the target edge index
    # folds by the number of full turns: step = 0 binds each outer
    # edge to the dual edge directly below it, and each 360/m of inner
    # spin steps it by one -> a flip-free bijection at any angle.
    for fidx, f in enumerate(Fp):
        oe = _poly_edges(outer[fidx], p.pen_curve)
        n = len(f)
        for i in range(n):
            vb = f[(i + 1) % n]                 # forward vertex
            m = len(rings[vb])
            step = round(p.tri_rot / (360.0 / m))
            j = rings[vb].index(fidx)
            te = inner_edges[vb][(j + step) % m]
            pe = oe[i]
            p0, tp = pe["m"], pe["t"]
            q, tq = te["m"], te["t"]
            if tq.dot(p0 - q) < 0:
                tq = -tq
            span = (q - p0).length
            Hp = p.handle_pen * span
            Ht = p.handle_tri * span
            pa, pb = pe["a"], pe["b"]
            qa, qb = te["a"], te["b"]
            if swap == "dist":
                if ((pa - qa).length + (pb - qb).length
                        > (pa - qb).length + (pb - qa).length):
                    qa, qb = qb, qa
            elif swap == "rev":
                qa, qb = qb, qa
            cpA = [pa, pa + Hp * tp, qa + Ht * tq, qa]
            cpB = [pb, pb + Hp * tp, qb + Ht * tq, qb]
            # keep the centreline + twist from the two rails, but
            # drive the half-width by its profile (endpoints pinned)
            Wp = (pa - pb).length / 2.0
            Wq = (qa - qb).length / 2.0
            rows = []
            for s in range(N + 1):
                t = s / N
                A = bez(cpA, t)
                B = bez(cpB, t)
                d = B - A
                dl = d.length
                if dl > 1e-9:
                    c = (A + B) / 2.0
                    w = wprofile(Wp, Wq, p.mid_width,
                                 p.width_pen, p.width_tri, t)
                    A = c - w * (d / dl)
                    B = c + w * (d / dl)
                rows.append((add(A), add(B)))
            for s in range(N):
                a0, b0 = rows[s]
                a1, b1 = rows[s + 1]
                faces.append([a0, b0, b1, a1])
    return verts, faces


# ---------------------------------------------------------------- #
#  Blender operator                                                #
# ---------------------------------------------------------------- #

if _IN_BLENDER:

    def _load_preset(self, context):
        """Adjust-Last-Operation callback: load a solid's default
        parameters when the Outer Solid selector changes."""
        preset = _PRESETS.get(self.solid)
        if preset:
            for key, val in preset.items():
                setattr(self, key, val)
            # smooth shading defaults on unless the preset opts out
            self.smooth_shading = preset.get("smooth_shading", True)

    class MESH_OT_woven_polyhedron_add(bpy.types.Operator):
        """Add a woven polyhedron: twisted ribbons bridging the inset
        faces of a Platonic solid to the inset faces of its dual,
        welded into one shell with Solidify + Subdivision so it can be
        thickened and smoothed"""
        bl_idname = "mesh.woven_polyhedron_add"
        bl_label = "Twisted Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(
            name="Outer Solid", items=SOLID_ITEMS, default='DODECA',
            update=_load_preset,
            description="Solid on the outside; its dual is woven on "
                        "the inside (switching loads that solid's "
                        "default parameters)")
        frequency: IntProperty(
            name="Frequency", default=2, min=1, max=8,
            description="Geodesic breakdown frequency (Geodesic "
                        "Sphere only): higher = more, smaller faces")
        tri_rot: FloatProperty(
            name="Inner Spin", default=90.0, min=-720, max=720,
            description="Spin of every inner polygon about its face "
                        "normal (degrees); folds with the inner "
                        "polygon's rotational symmetry")
        pen_rot: FloatProperty(
            name="Outer Spin", default=-18.66, min=-720, max=720,
            description="Spin of every outer polygon about its face "
                        "normal (degrees)")
        pen_inset: FloatProperty(
            name="Outer Size", default=0.32, min=0.05, max=1.0,
            description="Inset outer polygon size as a fraction of "
                        "its face")
        tri_inset: FloatProperty(
            name="Inner Size", default=0.15, min=0.05, max=1.0,
            description="Inset inner polygon size as a fraction of "
                        "its face")
        r_ico: FloatProperty(
            name="Inner Scale", default=0.76, min=0.2, max=2.5,
            description="Radius of the inner (dual) solid; the outer "
                        "solid is fixed at 1")
        pen_curve: FloatProperty(
            name="Outer Curvature", default=1.05, min=0.0, max=2.0,
            description="Dome each outer polygon toward a sphere "
                        "through its corners (0 = flat)")
        tri_curve: FloatProperty(
            name="Inner Curvature", default=0.0, min=0.0, max=2.0,
            description="Dome each inner polygon toward a sphere "
                        "(0 = flat)")
        handle_pen: FloatProperty(
            name="Stiffness (Outer)", default=1.5, min=0.0, max=3.0,
            description="Length of the bezier handle leaving the "
                        "outer edge (higher = straighter exit)")
        handle_tri: FloatProperty(
            name="Stiffness (Inner)", default=0.4, min=0.0, max=3.0,
            description="Length of the bezier handle leaving the "
                        "inner edge")
        mid_width: FloatProperty(
            name="Middle Width", default=0.29, min=0.0, max=1.5,
            description="Ribbon width at the middle as a fraction of "
                        "the endpoint sum (0.5 = linear midpoint, "
                        "less = a waist)")
        width_pen: FloatProperty(
            name="Taper (Outer)", default=50.65, min=0.0, max=100.0,
            description="How fast the outer half reaches the middle "
                        "width (0 = linear)")
        width_tri: FloatProperty(
            name="Taper (Inner)", default=0.0, min=0.0, max=100.0,
            description="How fast the inner half reaches the middle "
                        "width (0 = linear)")
        rib_samples: IntProperty(
            name="Ribbon Segments", default=14, min=2, max=40,
            description="Lengthwise segments per ribbon")
        thickness: FloatProperty(
            name="Thickness", default=0.03, min=0.0, max=0.5,
            description="Solidify shell thickness (0 = no Solidify "
                        "modifier)")
        smooth_level: IntProperty(
            name="Smoothing", default=2, min=0, max=4,
            description="Subdivision-Surface levels rounding the "
                        "corners (0 = no Subdivision modifier)")
        smooth_shading: BoolProperty(
            name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def _woven_mesh(self, swap):
            """Build + weld a woven mesh with the given rail pairing;
            return (mesh, non-orientable-seam-count)."""
            verts, faces = build(self, swap)
            s = self.scale
            me = bpy.data.meshes.new("Twisted Polyhedron")
            me.from_pydata([(x * s, y * s, z * s)
                            for x, y, z in verts], [],
                           [list(f) for f in faces])
            me.validate()
            # weld coincident verts + unify windings so Solidify
            # builds one clean-sided shell (no black flipped faces)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts,
                                     dist=1e-5 * s)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            seams = sum(1 for e in bm.edges
                        if e.is_manifold and not e.is_contiguous)
            bm.to_mesh(me)
            bm.free()
            return me, seams

        def execute(self, context):
            s = self.scale
            # the geometric ("dist") pairing looks best but is non-
            # orientable on a few solids; fall back to the consistent
            # pairing only when it actually comes out non-orientable
            me, seams = self._woven_mesh("dist")
            if seams:
                bpy.data.meshes.remove(me)
                me, seams = self._woven_mesh("fwd")
            if self.smooth_shading:
                me.polygons.foreach_set(
                    "use_smooth", [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Twisted Polyhedron", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.thickness > 0.0:
                sol = obj.modifiers.new("Thickness", "SOLIDIFY")
                sol.thickness = self.thickness * s
                sol.offset = 0.0
            if self.smooth_level > 0:
                sub = obj.modifiers.new("Smooth", "SUBSURF")
                sub.levels = self.smooth_level
                sub.render_levels = self.smooth_level
            self.report({'INFO'},
                        f"Twisted {self.solid.title()}: "
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "solid")
            if self.solid == 'GEODESIC':
                lay.prop(self, "frequency")
            lay.separator()
            for prop in ("tri_rot", "pen_rot", "pen_inset",
                         "tri_inset", "r_ico", "pen_curve",
                         "tri_curve", "handle_pen", "handle_tri",
                         "mid_width", "width_pen", "width_tri",
                         "rib_samples"):
                lay.prop(self, prop)
            lay.separator()
            lay.prop(self, "thickness")
            lay.prop(self, "smooth_level")
            lay.prop(self, "smooth_shading")
            lay.prop(self, "scale")

    def _menu_func(self, context):
        self.layout.operator("mesh.woven_polyhedron_add",
                             icon='MESH_ICOSPHERE')

    _classes = (MESH_OT_woven_polyhedron_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        print("Woven Polyhedron needs Blender (mathutils) to build; "
              "run tests/test_extension.py under Blender.")
