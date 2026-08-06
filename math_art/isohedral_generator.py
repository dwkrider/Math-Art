
# Isohedral (Escher-style) Tiling Generator for Blender
#
# An isohedral tiling is one whose symmetry group acts transitively on
# the tiles: every tile is equivalent to every other under a symmetry of
# the pattern.  Grunbaum & Shephard's classification names 81 such types
# (IH1..IH81).  This module implements a curated, coverage-verified set
# spanning the principal wallpaper symmetries.
#
# The creative feature is EDGE DEFORMATION.  A prototile is a fundamental
# domain of a wallpaper group; its boundary edges come in matched pairs,
# each pair related by a group isometry.  If both edges of a pair carry
# the SAME profile (transformed by that isometry), the deformed tile still
# tiles the plane gap-free -- this is exactly how M. C. Escher built his
# interlocking lizards, birds and fish.  The `shape` control drives the
# deformation: 0 gives a plain straight-edged polygon tiling, non-zero
# gives Escher-like curved tiles.
#
# One edge type is exempt: an edge lying on a mirror line is fixed by
# that reflection and must stay straight, else the mirror-image
# neighbour would no longer share it.  So the pure kaleidoscope tilings
# (p4m, p6m, p3m1) -- whose fundamental triangle has all three sides on
# mirrors -- are rigid: `shape` moves nothing.  Every other type here
# has one or more deformable edges.
#
# Part of the Pattern Engine (see pattern_common.py).
#
# References:
#   Branko Grunbaum & G. C. Shephard, "Tilings and Patterns" (W. H.
#     Freeman, 1987) -- the classification of the 81 isohedral tiling
#     types IH1..IH81.
#   M. C. Escher -- the interlocking-figure ("Escherization") tradition
#     of symmetry-matched edge deformation.
#   Craig S. Kaplan & David H. Salesin, "Escherization" (SIGGRAPH 2000);
#     Kaplan, "Introductory Tiling Theory for Computer Graphics" (2009)
#     and the "Tactile" isohedral-tiling library -- algorithmic edge
#     deformation.

bl_info = {
    "name": "Isohedral Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Isohedral (Escher-style) tilings with deformable, "
                   "symmetry-matched edges",
    "category": "Add Mesh",
}

from math import cos, sin, pi, hypot, sqrt
import numpy as np

try:
    from . import pattern_common as pc
    from . import tiling_generator as tg
except Exception:
    import pattern_common as pc
    import tiling_generator as tg


_S3 = sqrt(3.0)
_H = _S3 / 2.0


# --------------------------------------------------------------------
# Prototile definitions
# --------------------------------------------------------------------
#
# Each entry returns (label, note, b1, b2, cosets, proto):
#   * b1, b2   -- lattice basis vectors
#   * cosets   -- one isometry per point-group element (3x3 matrices);
#                 the full symmetry is every lattice translation composed
#                 with every coset
#   * proto    -- (N, 2) ordered vertices of a fundamental domain, whose
#                 images under the full symmetry tile the plane
#
# Edge pairing is then discovered automatically: for every prototile edge
# the unique neighbouring symmetry that shares it is found, which fixes
# how the deformation must propagate.

def _def_p1():
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.22, 1.0])
    cos_ = [pc.I()]
    proto = np.array([[0, 0], [1, 0], [1.22, 1.0], [0.22, 1.0]], float)
    return "Translation (p1)", "IH41 -- parallelogram, edges paired by translation", b1, b2, cos_, proto


def _def_p2():
    # any quadrilateral tiles by 180-deg rotations about its edge
    # midpoints (the classic monohedral p2 tiling).
    A = np.array([0.0, 0.0])
    B = np.array([1.0, -0.12])
    C = np.array([1.18, 0.82])
    D = np.array([0.14, 0.96])
    mAB = 0.5 * (A + B)
    b1 = C - A
    b2 = D - B
    cos_ = [pc.I(), pc.Rot(pi, mAB[0], mAB[1])]
    proto = np.array([A, B, C, D], float)
    return "2-fold (p2)", "IH4 -- quadrilateral, edges paired by 180-deg rotation", b1, b2, cos_, proto


def _def_pg():
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.0, 1.0])
    # glide across the line y = 0.5, translating +0.5 in x
    cos_ = [pc.I(), pc.Glide(0.0, 0.5, 0.0, 0.5)]
    # midpoints on the glide-paired top/bottom edges so each half meets a
    # single neighbour (no T-junction)
    proto = np.array([[0, 0], [0.5, 0], [1, 0], [1, 0.5],
                      [0.5, 0.5], [0, 0.5]], float)
    return "Glide (pg)", "IH43 -- glide-paired edges", b1, b2, cos_, proto


def _def_pgg():
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.0, 1.0])
    # two perpendicular glide axes; their product is the 2-fold rotation
    g1 = pc.Glide(0.0, 0.5, 0.0, 0.25)          # (x + 0.5, 0.5 - y)
    g2 = pc.Glide(pi / 2, 0.5, 0.25, 0.0)       # (0.5 - x, y + 0.5)
    cos_ = [pc.I(), g1, g2, g1 @ g2]
    proto = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)
    return "Double Glide (pgg)", "IH53 -- two perpendicular glide axes", b1, b2, cos_, proto


def _def_cmm():
    b1 = np.array([1.0, 1.0])
    b2 = np.array([1.0, -1.0])
    cos_ = [pc.I(), pc.Rot(pi, 0, 0), pc.Mir(0.0, 0, 0), pc.Mir(pi / 2, 0, 0)]
    proto = np.array([[0, 0], [1, 0], [1, 1]], float)
    return "Rhombic (cmm)", "IH26 -- 2-fold with mirrors", b1, b2, cos_, proto


def _def_p4():
    b1 = np.array([2.0, 0.0])
    b2 = np.array([0.0, 2.0])
    cos_ = [pc.Rot(k * pi / 2, 0, 0) for k in range(4)]
    proto = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    return "4-fold (p4)", "IH28 -- 4-fold pinwheel", b1, b2, cos_, proto


def _def_p4g():
    # 4-fold rotation centres (pure C4, no mirror through them) plus
    # diagonal mirror lines offset onto the 2-fold centres -> genuine
    # 4*2, leaving two deformable edges per tile.
    b1 = np.array([2.0, 0.0])
    b2 = np.array([0.0, 2.0])
    cos_ = [pc.Rot(k * pi / 2, 0, 0) for k in range(4)]
    cos_ += [pc.Rot(k * pi / 2, 0, 0) @ pc.Mir(pi / 4, 1.0, 0.0)
             for k in range(4)]
    proto = np.array([[1, 0], [1, 1], [0, 1]], float)
    return "4-fold Mirror (p4g)", "IH29 -- 4-fold with diagonal mirror lines", b1, b2, cos_, proto


def _def_p4m():
    b1 = np.array([2.0, 0.0])
    b2 = np.array([0.0, 2.0])
    cos_ = [pc.Rot(k * pi / 2, 0, 0) for k in range(4)]
    cos_ += [pc.Rot(k * pi / 2, 0, 0) @ pc.Mir(0.0, 0, 0)
             for k in range(4)]
    proto = np.array([[0, 0], [1, 0], [1, 1]], float)
    return "4-fold Kaleidoscope (p4m)", "IH61 -- full square symmetry", b1, b2, cos_, proto


def _hex_pts(n=6, r=1.0, rot=0.0):
    return np.array([[r * cos(rot + k * pi / 3), r * sin(rot + k * pi / 3)]
                     for k in range(n)])


def _def_p3():
    # three rhombi meeting at a 3-fold centre (rhombille skeleton);
    # deformed, this becomes an Escher 3-fold pinwheel.
    h = _hex_pts()
    O = np.array([0.0, 0.0])
    cos_ = [pc.Rot(k * 2 * pi / 3, 0, 0) for k in range(3)]
    b1 = np.array([1.5, _H])
    b2 = np.array([1.5, -_H])
    proto = np.array([O, h[0], h[1], h[2]], float)
    return "3-fold Pinwheel (p3)", "IH7 -- 3-fold rotation", b1, b2, cos_, proto


def _def_p6():
    h = _hex_pts()
    O = np.array([0.0, 0.0])
    cos_ = [pc.Rot(k * pi / 3, 0, 0) for k in range(6)]
    b1 = np.array([1.5, _H])
    b2 = np.array([1.5, -_H])
    proto = np.array([O, h[0], h[1]], float)
    return "6-fold Pinwheel (p6)", "IH21 -- 6-fold rotation", b1, b2, cos_, proto


def _def_p6m():
    h = _hex_pts()
    O = np.array([0.0, 0.0])
    cos_ = [pc.Rot(k * pi / 3, 0, 0) for k in range(6)]
    cos_ += [pc.Rot(k * pi / 3, 0, 0) @ pc.Mir(0.0, 0, 0)
             for k in range(6)]
    b1 = np.array([1.5, _H])
    b2 = np.array([1.5, -_H])
    m = 0.5 * (h[0] + h[1])
    proto = np.array([O, h[0], m], float)
    return "6-fold Kaleidoscope (p6m)", "IH88(dual) -- full hexagonal symmetry", b1, b2, cos_, proto


def _def_p3m1():
    h = _hex_pts()
    O = np.array([0.0, 0.0])
    cos_ = [pc.Rot(k * 2 * pi / 3, 0, 0) for k in range(3)]
    cos_ += [pc.Rot(k * 2 * pi / 3, 0, 0) @ pc.Mir(pi / 3, 0, 0)
             for k in range(3)]
    b1 = np.array([1.5, _H])
    b2 = np.array([1.5, -_H])
    proto = np.array([O, h[0], h[1]], float)
    return "3-fold Kaleidoscope (p3m1)", "IH13 -- 3-fold mirror lines", b1, b2, cos_, proto


def _def_p31m():
    # p31m (orbifold 3*3): hexagonal lattice with the mirrors running
    # ALONG the lattice directions (0, 60, 120 deg), so the mirror net
    # cuts the plane into equilateral triangles whose corners are the
    # lattice points (3-fold-on-mirror *3 corners) and whose centroids
    # are free 3-fold centres lying OFF every mirror.  Contrast p3m1,
    # where the mirrors sit at 30 deg to the lattice and every rotation
    # centre lands on a mirror.  Fundamental domain: one third of a
    # mirror triangle -- (corner, corner, centroid).  The corner-corner
    # edge lies on a mirror and stays straight; the two centroid edges
    # are swapped by the 120-deg rotation about the free centre, so they
    # deform together into an interlocking 3-armed pinwheel.
    b1 = np.array([1.5, _H])
    b2 = np.array([1.5, -_H])
    cos_ = [pc.Rot(k * 2 * pi / 3, 0, 0) for k in range(3)]
    cos_ += [pc.Mir(pi / 6 + k * pi / 3, 0, 0) for k in range(3)]
    g = (b1 + b2) / 3.0
    proto = np.array([[0.0, 0.0], b1, g], float)
    return ("3-fold Mirror Pinwheel (p31m)",
            "mirror-triangle third -- free 3-fold centres off the mirrors",
            b1, b2, cos_, proto)


_DEFS = {
    'P1': _def_p1, 'P2': _def_p2, 'PG': _def_pg, 'PGG': _def_pgg,
    'CMM': _def_cmm, 'P4': _def_p4, 'P4G': _def_p4g, 'P4M': _def_p4m,
    'P3': _def_p3, 'P6': _def_p6, 'P6M': _def_p6m, 'P3M1': _def_p3m1,
    'P31M': _def_p31m,
}


# --------------------------------------------------------------------
# Edge deformation engine
# --------------------------------------------------------------------

def _bump(ts, seed):
    """A deterministic zero-at-endpoints wiggle profile for one edge."""
    r = (int(seed) * 2654435761) & 0xffffffff
    a1 = ((r & 1023) / 1023.0) - 0.5
    a2 = (((r >> 10) & 1023) / 1023.0) - 0.5
    a3 = (((r >> 20) & 1023) / 1023.0) - 0.5
    return (0.55 * a1 * np.sin(pi * ts) +
            0.30 * a2 * np.sin(2 * pi * ts) +
            0.20 * a3 * np.sin(3 * pi * ts))


def _deform_edge(a, b, seed, shape, samples=6):
    """Polyline from a to b with a perpendicular wiggle scaled by shape
    and edge length (straight when shape == 0)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ts = np.linspace(0.0, 1.0, samples + 1)
    base = a[None, :] + ts[:, None] * (b - a)[None, :]
    if shape == 0.0:
        return base
    d = b - a
    L = hypot(d[0], d[1])
    if L < 1e-12:
        return base
    nrm = np.array([-d[1], d[0]]) / L
    off = _bump(ts, seed) * shape * L
    return base + nrm[None, :] * off[:, None]


def _match(M, pt, target, tol=1e-6):
    q = pc.apply(M, [pt])[0]
    return abs(q[0] - target[0]) < tol and abs(q[1] - target[1]) < tol


def _neighbours(b1, b2, cosets, rng=2):
    out = []
    for i in range(-rng, rng + 1):
        for j in range(-rng, rng + 1):
            t = pc.T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for c in cosets:
                M = t @ c
                if not np.allclose(M, np.eye(3), atol=1e-9):
                    out.append(M)
    return out


def _is_reflection(M):
    A = M[:2, :2]
    return (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) < 0.0


def _pairings(proto, cands):
    """Classify every prototile edge and pair it with its neighbour.

    Two tiles sharing an edge traverse it in opposite senses.  For an
    orientation-PRESERVING neighbour M (translation/rotation) the shared
    edge is the reverse image: M(edge l) == reverse(edge k) ('rev').  For
    an orientation-REVERSING neighbour M (glide/reflection) the flip of
    handedness makes the forward image align: M(edge l) == edge k ('fwd').
    An edge lying ON a mirror line is fixed pointwise ('mirror') and must
    stay straight.

    Returns (pairs, mirrors, kind) where pairs is a list of
    (mode, k, l, M): the deformed edge k is derived from the freely
    deformed source edge l.  Each edge appears once."""
    n = len(proto)
    kind = [None] * n
    for k in range(n):
        pk, pk1 = proto[k], proto[(k + 1) % n]
        # mirror line: a reflection fixing both endpoints in place
        hit = None
        for M in cands:
            if _is_reflection(M) and _match(M, pk, pk) and _match(M, pk1, pk1):
                hit = ('mirror', k, M)
                break
        if hit is None:
            for M in cands:
                for l in range(n):
                    pl, pl1 = proto[l], proto[(l + 1) % n]
                    if _match(M, pl, pk1) and _match(M, pl1, pk):
                        hit = ('rev', l, M)
                        break
                    if (_is_reflection(M) and _match(M, pl, pk)
                            and _match(M, pl1, pk1) and not (l == k)):
                        hit = ('fwd', l, M)
                        break
                if hit:
                    break
        kind[k] = hit
    pairs, mirrors = [], []
    done = [False] * n
    for k in range(n):
        if done[k] or kind[k] is None:
            continue
        mode, l, M = kind[k]
        if mode == 'mirror':
            mirrors.append(k)
            done[k] = True
        else:
            pairs.append((mode, k, l, M))
            done[k] = True
            done[l] = True
    return pairs, mirrors, kind


def _deform_proto(proto, pairs, mirrors, shape, samples=6):
    """Deform every prototile edge so matched pairs carry the same
    profile under their linking isometry -> a gap-free deformable tile.
    Mirror-line edges are left straight (they cannot move off the line
    without breaking the shared boundary with the mirror-image tile)."""
    n = len(proto)
    edges = [None] * n
    for k in mirrors:
        edges[k] = _deform_edge(proto[k], proto[(k + 1) % n], k + 1,
                                0.0, samples)
    for (mode, k, l, M) in pairs:
        if l == k:
            # self-paired edge (180-deg rotation, or reflection across
            # its perpendicular bisector): symmetrise so the edge maps
            # exactly onto its own reverse under M.
            dp = _deform_edge(proto[k], proto[(k + 1) % n], k + 1,
                              shape, samples)
            edges[k] = 0.5 * (dp + pc.apply(M, dp)[::-1])
        else:
            dp = _deform_edge(proto[l], proto[(l + 1) % n], l + 1,
                              shape, samples)
            edges[l] = dp
            img = pc.apply(M, dp)
            edges[k] = img[::-1] if mode == 'rev' else img
    ring = []
    for k in range(n):
        ring.append(edges[k][:-1])
    return np.vstack(ring)


def _lattice(b1, b2, cosets, nx, ny):
    out = []
    for i in range(nx):
        for j in range(ny):
            t = pc.T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for gi, c in enumerate(cosets):
                out.append((t @ c, gi))
    return out


def build_patch(name, nx, ny, shape):
    """Build an isohedral tiling patch: (polys, types).  A deformed
    prototile is replicated by the tiling's full symmetry over an
    nx x ny lattice block; `types` is the coset (orbit) index per tile."""
    label, note, b1, b2, cosets, proto = _DEFS[name]()
    cands = _neighbours(b1, b2, cosets)
    pairs, mirrors, _ = _pairings(proto, cands)
    ring = _deform_proto(proto, pairs, mirrors, shape)
    polys, types = [], []
    for M, gi in _lattice(b1, b2, cosets, nx, ny):
        polys.append(pc.apply(M, ring))
        types.append(gi)
    return polys, types


def _label(name):
    return _DEFS[name]()[0]


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

TILING_ITEMS = [
    ('P1', "Translation (p1)", "Parallelogram, translation-paired edges"),
    ('P2', "2-fold (p2)", "Quadrilateral, 180-deg rotation-paired edges"),
    ('PG', "Glide (pg)", "Glide-paired edges"),
    ('PGG', "Double Glide (pgg)", "Two glide axes"),
    ('CMM', "Rhombic (cmm)", "2-fold rotation with mirror lines"),
    ('P4', "4-fold (p4)", "4-fold rotation pinwheel"),
    ('P4G', "4-fold Mirror (p4g)", "4-fold with glide reflections"),
    ('P4M', "4-fold Kaleidoscope (p4m)", "Full square symmetry"),
    ('P3', "3-fold Pinwheel (p3)", "3-fold rotation"),
    ('P6', "6-fold Pinwheel (p6)", "6-fold rotation"),
    ('P6M', "6-fold Kaleidoscope (p6m)", "Full hexagonal symmetry"),
    ('P3M1', "3-fold Kaleidoscope (p3m1)", "3-fold mirror lines"),
    ('P31M', "3-fold Mirror Pinwheel (p31m)",
     "Mirror triangles with free 3-fold centres"),
]


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_isohedral_add(bpy.types.Operator, AddObjectHelper):
        """Add an isohedral (Escher-style) tiling"""
        bl_idname = "mesh.isohedral_add"
        bl_label = "Isohedral Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        tiling: EnumProperty(name="Tiling", items=TILING_ITEMS,
                             default='P2')
        nx: IntProperty(name="Cells X", default=4, min=1, max=30)
        ny: IntProperty(name="Cells Y", default=4, min=1, max=30)
        shape: FloatProperty(
            name="Edge Shape", default=0.0, min=-1.0, max=1.0,
            description="Edge deformation: 0 = straight polygon tiling, "
                        "non-zero = Escher-like interlocking curved tiles")
        color_by: EnumProperty(
            name="Color By",
            items=[('SIDES', "By Sides", "Material per polygon side count"),
                   ('TYPE', "By Tile Type",
                    "Material per symmetry orbit in the unit cell"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='TYPE')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        trim: BoolProperty(
            name="Trim Boundary", default=False,
            description="Clip the patch to a clean rectangle, removing "
                        "the ragged edge and stray boundary tiles")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty) so tiles can be edited "
                        "individually")

        def execute(self, context):
            cells = tg.cells_from_polys(
                lambda a, b: build_patch(self.tiling, a, b, self.shape),
                self.nx, self.ny, self.color_by, self.margin,
                self.height, self.trim)
            label = _label(self.tiling)
            obj = pc.emit(context, "Isohedral %s" % label, cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no tiling generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s  V=%d F=%d" %
                            (label, len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d tiles" %
                            (label, len(obj.children)))
            return {'FINISHED'}

        # full kaleidoscopes: every edge is on a mirror, so the tile is
        # rigid and the shape slider has no effect -- hide it for these
        _RIGID = {'P4M', 'P6M', 'P3M1'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for p in ('tiling', 'nx', 'ny'):
                lay.prop(self, p)
            if self.tiling in self._RIGID:
                lay.label(text="Kaleidoscope: edges are mirror-locked, "
                               "shape has no effect", icon='INFO')
            else:
                lay.prop(self, 'shape')
            for p in ('color_by', 'margin', 'height'):
                lay.prop(self, p)
            lay.prop(self, 'trim')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.isohedral_add", icon='MESH_GRID')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_isohedral_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_isohedral_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _defects(polys, n=30, half=2.0):
    """Sample a grid; count points not covered by EXACTLY one tile."""
    allv = np.vstack([np.asarray(p, float) for p in polys])
    c = allv.mean(axis=0)
    xs = np.linspace(c[0] - half, c[0] + half, n) + 0.0137
    ys = np.linspace(c[1] - half, c[1] + half, n) + 0.0071
    bad = 0
    for x in xs:
        for y in ys:
            hits = 0
            for p in polys:
                if tg._pip(p, x, y):
                    hits += 1
                    if hits > 1:
                        break
            if hits != 1:
                bad += 1
    return bad


_ORDER = [k for k, _, _ in TILING_ITEMS]


def _selftest():
    bad_types = []
    for name in _ORDER:
        label = _label(name)
        row = None
        try:
            p0, _ = build_patch(name, 6, 6, 0.0)
            p1, _ = build_patch(name, 6, 6, 0.35)
            d0 = _defects(p0)
            d1 = _defects(p1)
            ok = (d0 == 0 and d1 == 0)
            row = "%-26s tiles=%4d  defects@0=%3d  defects@0.35=%3d  %s" % (
                label, len(p0), d0, d1, "OK" if ok else "BAD")
        except Exception as exc:
            ok = False
            row = "%-26s ERROR: %s" % (label, exc)
        print(row)
        if not ok:
            bad_types.append(name)
    print("RESULT:", "OK" if not bad_types else "BAD %s" % bad_types)
    assert not bad_types
