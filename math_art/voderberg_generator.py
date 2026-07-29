
# Voderberg Spiral Tiling Generator for Blender
#
# The first spiral tiling ever devised (Heinz Voderberg, 1936) and its
# Grunbaum-Shephard generalizations, produced as finite patches of
# colored tiles fed through the shared Pattern Engine (see
# pattern_common.py) for color, grout margin and relief.
#
# THE TILE.  The Voderberg prototile is a nonconvex ENNEAGON (nine-sided
# figure) -- a single prototile, so the tiling is MONOHEDRAL (every tile
# congruent up to rotation/reflection).  It is a "bent" isosceles
# triangle: start from an isosceles triangle whose apex (vertex) angle is
# exactly 12 degrees (= 180/15, so 30 of them close up around a point)
# and replace each of its two equal legs by a congruent bent polyline of
# three segments.  The two legs are antisymmetric about their midpoints,
# which is what lets copies interlock.  Following Waldman's "Voderberg
# Deconstructed" (2014), the whole family is parameterized by one angle
# `beta` (the obtuse angle at the deep notch), valid for 111 <= beta <=
# 153 degrees; beta = 132 is the classic Voderberg.  The enneagon has
# only three distinct edge lengths (1, x, L) and its defining property is
# that two copies can completely enclose a third (or, for higher spiral
# order, a whole chain of copies).
#
# THE TILINGS.  Built by Goldberg's triangle-substitution method (the
# construction Grunbaum & Shephard use for the Voderberg spiral, and the
# picture on the cover of "Tilings and Patterns"):
#
#   RADIAL   The monohedral radial tiling -- 30 wedge-sectors of stacked
#       V/A coronas closing up into concentric rings about the centre.
#       The un-sheared parent of every spiral (Goldberg shift 0).
#
#   CLASSIC  The classic two-arm Voderberg DOUBLE spiral.  Take the
#       radial disk, cut it along a diameter and slide one half-plane by
#       one tile (Goldberg shift 1): the halves re-mesh everywhere but
#       interlock into a two-armed spiral about the centre, with the
#       famous pair-encloses-one configuration at the core.  2-fold
#       rotational symmetry.
#
#   SPIRAL   The Grunbaum-Shephard enclosure family: the same double
#       spiral with Goldberg shift g = `arms`, so a pair of tiles
#       encloses g tiles at the core (g = 1 is the classic).  Every g
#       gives a distinct valid two-arm spiral of a different pitch.
#
# NOTE ON ARM COUNT.  Every Voderberg spiral obtainable this way is a
# TWO-arm (double) spiral -- that is a property of the diametral Goldberg
# cut.  Genuine monohedral r-armed spirals for r >= 3 are not
# constructible by this method (Grunbaum & Shephard remark that no
# prototile was known to admit a monohedral r-armed spiral for odd
# r >= 5); so the `arms` control varies the verified Goldberg enclosure
# order g rather than the topological arm count, which stays 2.
#
# types: 0 / 1 mark the two spiral arms (the two half-planes of the
# Goldberg cut) for coloring.
#
# References:
#   Heinz Voderberg, "Zur Zerlegung der Umgebung eines ebenen Bereiches
#     in kongruente" (1936) -- the original spiral tiling and enneagon.
#   Michael Goldberg, "Central tessellations" (1955) -- the triangle-
#     substitution (Goldberg-shift) construction of the spirals.
#   Branko Grunbaum & G. C. Shephard, "Tilings and Patterns" (W. H.
#     Freeman, 1987) -- the enclosure family (cover illustration).
#   Stefan Waldman, "Voderberg Deconstructed & Triangle Substitution
#     Tiling" (2014) -- the beta parameterization used here.

bl_info = {
    "name": "Voderberg Spiral Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Voderberg (1936) spiral tiling of a single bent "
                   "enneagon -- radial, classic two-arm double spiral, "
                   "and the Grunbaum-Shephard enclosure family",
    "category": "Add Mesh",
}

from math import pi, sin, cos, tan

import numpy as np

try:
    from . import pattern_common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import tiling_generator as tg


# 12-degree apex -> exactly 30 sectors close up around the centre.
VERTEX_DEG = 12.0
SECTORS = 30
BETA_MIN, BETA_MAX = 111.0, 153.0
BETA_CLASSIC = 132.0


# --------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------

def _to_xy(z):
    """Complex array -> (N, 2) float array."""
    z = np.asarray(z)
    return np.column_stack([z.real, z.imag])


def _signed_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly):
    poly = np.asarray(poly, float)
    if _signed_area(poly) < 0.0:
        return poly[::-1].copy()
    return poly


# --------------------------------------------------------------------
# The Voderberg enneagon prototile (Waldman 2014)
#
# beta is the obtuse notch angle (111..153 deg); 132 is the classic.
# The tile is returned as two congruent prototiles in the complex plane:
# V ("apex down", resting on the centre) and A ("apex up", its partner in
# a corona).  A is a reflected/translated copy of V, hence congruent.
# --------------------------------------------------------------------

def voderberg_tile(beta_deg=BETA_CLASSIC):
    """Return (V, A) complex arrays of 9 vertices each: the Voderberg
    enneagon prototile and its corona partner (both congruent)."""
    alef = VERTEX_DEG * pi / 180.0          # 12 deg apex
    beth = float(beta_deg) * pi / 180.0
    # three distinct edge lengths (base = 1); L and x from the law of
    # sines closure of the bent legs (Waldman Eqs. 1)
    L = 2.0 * sin((pi - alef) / 2.0) / cos(beth - pi / 2.0)
    x = (1.0 / sin(alef / 2.0) / 2.0 - L * cos(pi - beth)) / 2.0 - sin(alef / 2.0)
    S = [1.0, x, L, x, 1.0, x, L, x, 1.0]                 # 9 edge lengths
    theta = [alef, (3 * pi - alef) / 2.0, (2 * pi - beth), beth,
             (pi + alef) / 2.0, (pi - 3 * alef) / 2.0, (2 * pi - beth),
             beth, (pi + alef) / 2.0]                     # 9 interior angles
    phi = np.cumsum([pi - t for t in theta])              # exterior turning
    z = np.cumsum([S[k] * np.exp(1j * phi[k]) for k in range(9)])
    z = np.concatenate([[0.0 + 0.0j], z])                 # 10 pts, closes
    V = z[:-1].copy()                                     # 9 vertices
    A = (-z + 0.5 + 1j * float(np.max(z.imag)))[:-1].copy()
    return V, A


# --------------------------------------------------------------------
# Goldberg triangle-substitution: radial tiling + diametral spiral shift
# --------------------------------------------------------------------

def _radial_disk(V, A, sectors, coronas):
    """One radial disk of the V/A tile: `sectors` wedges of `coronas`
    stacked coronas.  Returns (tiles, arm, role): tiles is a list of
    complex arrays, arm is 0/1 for the top/bottom half-plane, and role is
    0/1 for the prototile aspect (0 = apex-in V, 1 = apex-out A)."""
    alef = 2.0 * pi / sectors
    dely = 1.0 / (2.0 * tan(alef / 2.0))    # corona (row) height
    # build one sector column: core V, then corona r = V-A-...-V (2r+1),
    # tracking each column entry's aspect (0 = V, 1 = A)
    cols = [V.copy()]
    col_role = [0]
    for r in range(1, coronas + 1):
        for k in range(1, r + 1):
            off = -r / 2.0 + (k - 1) + 1j * r * dely
            cols.append(V + off)
            col_role.append(0)
            cols.append(A + off)
            col_role.append(1)
        cols.append(V + r / 2.0 + 1j * r * dely)          # closing V
        col_role.append(0)
    # stand the sector on the negative x-axis, then wheel it around
    base = [c * np.exp(1j * pi / 2.0) * np.exp(-1j * pi / sectors)
            for c in cols]
    tiles, arm, role = [], [], []
    half = sectors // 2
    for ksec in range(sectors):
        rot = np.exp(-1j * ksec * alef)
        for c, rr in zip(base, col_role):
            tiles.append(c * rot)
            arm.append(0 if ksec < half else 1)
            role.append(rr)
    return tiles, arm, role


def _goldberg_spiral(V, A, sectors, coronas, shift):
    """Radial disk with Goldberg shift `shift`: slide the bottom
    half-plane by `shift` leg-lengths so the two halves interlock into a
    two-arm spiral, then recentre on the 2-fold symmetry centre."""
    tiles, arm, role = _radial_disk(V, A, sectors, coronas)
    if shift == 0:
        return tiles, arm, role
    alef = 2.0 * pi / sectors
    delx = 1.0 / (2.0 * sin(alef / 2.0))    # leg length = shift unit
    ctr = 0.5 * shift * delx                # symmetry centre after shift
    out = []
    for c, a in zip(tiles, arm):
        out.append((c + shift * delx - ctr) if a == 1 else (c - ctr))
    return out, arm, role


# --------------------------------------------------------------------
# Public patch builder
# --------------------------------------------------------------------

def voderberg_patch(kind, arms=1, coils=4, beta=BETA_CLASSIC):
    """Return (polys, arm, role) for a finite Voderberg tile patch.

    polys -- list of (9, 2) CCW float arrays (one per congruent enneagon)
    arm   -- 0/1 spiral-arm (half-plane) index per tile, parallel to polys
    role  -- 0/1 prototile aspect per tile (0 = apex-in V, 1 = apex-out A)

    kind  -- 'RADIAL', 'CLASSIC' (classic two-arm double spiral,
             Goldberg shift 1) or 'SPIRAL' (double spiral of Goldberg
             enclosure order g = arms)
    arms  -- Goldberg spiral order for 'SPIRAL' (1 = classic); the
             topological arm count is 2 for every spiral (see module
             header)
    coils -- number of coronas the disk winds out to
    beta  -- deformable-tile obtuse angle in degrees (111..153)
    """
    coronas = max(1, int(coils))
    if kind == 'RADIAL':
        shift = 0
    elif kind == 'CLASSIC':
        shift = 1
    elif kind == 'SPIRAL':
        shift = max(1, min(int(arms), coronas))
    else:
        raise ValueError("unknown Voderberg kind %r" % kind)

    V, A = voderberg_tile(beta)
    tiles, arm, role = _goldberg_spiral(V, A, SECTORS, coronas, shift)

    polys, arms_out, roles_out = [], [], []
    for c, a, rr in zip(tiles, arm, role):
        polys.append(_ensure_ccw(_to_xy(c)))
        arms_out.append(int(a))
        roles_out.append(int(rr))
    return polys, arms_out, roles_out


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

KIND_ITEMS = [
    ('CLASSIC', "Classic Double Spiral",
     "The classic 1936 two-arm Voderberg spiral (Goldberg shift 1); the "
     "picture on the cover of Grunbaum & Shephard's Tilings and Patterns"),
    ('SPIRAL', "Spiral (Enclosure Order)",
     "Two-arm double spiral of Goldberg enclosure order = Arms; a pair "
     "of tiles encloses that many tiles at the core (1 = classic)"),
    ('RADIAL', "Radial Rings",
     "The monohedral radial tiling (Goldberg shift 0): concentric rings "
     "of enneagons, the un-sheared parent of the spirals"),
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

    class MESH_OT_voderberg_add(bpy.types.Operator, AddObjectHelper):
        """Add a Voderberg spiral tiling patch"""
        bl_idname = "mesh.voderberg_add"
        bl_label = "Voderberg Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(name="Tiling", items=KIND_ITEMS,
                           default='CLASSIC')
        arms: IntProperty(
            name="Arms / Order", default=2, min=1, max=8,
            description="Goldberg spiral order for the Spiral kind (a "
                        "pair of tiles encloses this many; 1 = classic). "
                        "Every Voderberg spiral here is a two-arm double "
                        "spiral -- higher r-armed monohedral spirals are "
                        "not constructible by this method")
        coils: IntProperty(
            name="Coils", default=4, min=1, max=20,
            description="Number of coronas the spiral winds out to")
        beta: FloatProperty(
            name="Tile Angle (beta)", default=BETA_CLASSIC,
            min=BETA_MIN, max=BETA_MAX,
            description="Obtuse notch angle of the deformable enneagon in "
                        "degrees (111..153); 132 is the classic Voderberg")
        color_by: EnumProperty(
            name="Color By",
            items=[('ARM', "By Arm",
                    "A material per spiral arm (the two half-planes)"),
                   ('TYPE', "By Tile Type",
                    "A material per prototile aspect -- the two "
                    "interlocking enneagon orientations (apex-in V vs "
                    "apex-out A) that reveal the tile boundaries"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='ARM')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty) so tiles can be edited "
                        "individually")

        def execute(self, context):
            polys, arm, role = voderberg_patch(self.kind, self.arms,
                                               self.coils, self.beta)
            if self.color_by == 'TYPE':
                types, cby = role, 'TYPE'
            elif self.color_by == 'ARM':
                types, cby = arm, 'TYPE'
            else:
                types, cby = arm, 'UNIFORM'
            cells = tg.cells_from_polys(
                lambda a, b: (polys, types), 1, 1, cby,
                self.margin, self.height, False)
            label = dict((k, v) for k, v, _ in KIND_ITEMS)[self.kind]
            obj = pc.emit(context, "Voderberg %s" % label, cells,
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

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind == 'SPIRAL':
                lay.prop(self, 'arms')
            lay.prop(self, 'coils')
            lay.prop(self, 'beta')
            for p in ('color_by', 'margin', 'height'):
                lay.prop(self, p)
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.voderberg_add", icon='FORCE_VORTEX')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_voderberg_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_voderberg_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _tile_fingerprint(poly):
    """Rotation/reflection-invariant shape key: the cyclic sequence of
    (edge length, signed turning angle) reduced to its lexicographic
    minimum over all cyclic shifts and the mirror image.  Two tiles are
    congruent iff their fingerprints are equal."""
    poly = np.asarray(poly, float)
    n = len(poly)
    edges = [float(np.hypot(*(poly[(k + 1) % n] - poly[k])))
             for k in range(n)]
    turns = []
    for k in range(n):
        v1 = poly[k] - poly[k - 1]
        v2 = poly[(k + 1) % n] - poly[k]
        turns.append(float(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0],
                                      v1[0] * v2[0] + v1[1] * v2[1])))
    cands = []
    for refl in (False, True):
        e = edges[::-1] if refl else edges
        t = [-a for a in turns[::-1]] if refl else turns
        for s in range(n):
            seq = tuple(round(e[(s + i) % n], 6) for i in range(n)) + \
                  tuple(round(t[(s + i) % n], 6) for i in range(n))
            cands.append(seq)
    return min(cands)


def _congruence_defects(polys):
    """Number of tiles not congruent to the first tile (the prototile)."""
    ref = _tile_fingerprint(polys[0])
    return sum(1 for p in polys if _tile_fingerprint(p) != ref)


def _coverage(polys, samples=40, half=3.0, off=(0.0137, 0.0071)):
    """Grid over an interior window centred on the patch bbox centre.
    Returns (n_covered, n_over, n_total)."""
    allv = np.vstack([np.asarray(p, float) for p in polys])
    cx = 0.5 * (allv[:, 0].min() + allv[:, 0].max())
    cy = 0.5 * (allv[:, 1].min() + allv[:, 1].max())
    n_cov = n_over = 0
    for ix in range(samples):
        x = cx - half + 2.0 * half * ix / (samples - 1) + off[0]
        for iy in range(samples):
            y = cy - half + 2.0 * half * iy / (samples - 1) + off[1]
            cnt = 0
            for p in polys:
                if tg._pip(p, x, y):
                    cnt += 1
                    if cnt >= 2:
                        break
            if cnt >= 1:
                n_cov += 1
            if cnt >= 2:
                n_over += 1
    return n_cov, n_over, samples * samples


def _enclosed_hole(polys, eps=0.01):
    """The 'two tiles enclose a third' core check: take the tile nearest
    the patch centre (the enclosed seed) and confirm it is fully
    surrounded -- every point just outside each of its edges (along the
    edge's outward normal) lies on some OTHER tile, so the seed sits in a
    hole-free core.  Returns the number of exposed (uncovered) boundary
    samples (0 = fully enclosed)."""
    allv = np.vstack([np.asarray(p, float) for p in polys])
    cx = 0.5 * (allv[:, 0].min() + allv[:, 0].max())
    cy = 0.5 * (allv[:, 1].min() + allv[:, 1].max())
    seed = min(range(len(polys)),
               key=lambda i: np.hypot(*(np.asarray(polys[i]).mean(axis=0)
                                        - (cx, cy))))
    sp = _ensure_ccw(np.asarray(polys[seed], float))       # CCW winding
    n = len(sp)
    exposed = 0
    for k in range(n):
        a, b = sp[k], sp[(k + 1) % n]
        d = b - a
        nrm = np.array([d[1], -d[0]])          # outward normal (CCW poly)
        nrm = nrm / (np.hypot(*nrm) + 1e-30)
        for t in (0.25, 0.5, 0.75):
            pt = a + t * d + nrm * eps          # nudge just outside edge
            if not any(i != seed and tg._pip(polys[i], pt[0], pt[1])
                       for i in range(len(polys))):
                exposed += 1
    return exposed


if __name__ == "__main__":
    all_ok = True
    # (kind, arms, coils, beta) cases -- classic, radial, a couple of
    # enclosure orders, and a deformed tile shape.
    cases = [
        ("CLASSIC g1  ", 'CLASSIC', 1, 5, BETA_CLASSIC),
        ("RADIAL      ", 'RADIAL',  1, 5, BETA_CLASSIC),
        ("SPIRAL g2   ", 'SPIRAL',  2, 6, BETA_CLASSIC),
        ("SPIRAL g3   ", 'SPIRAL',  3, 6, BETA_CLASSIC),
        ("CLASSIC b120", 'CLASSIC', 1, 5, 120.0),
    ]
    for name, kind, arms, coils, beta in cases:
        polys, arm, role = voderberg_patch(kind, arms, coils, beta)
        ncong = _congruence_defects(polys)
        n_cov, n_over, n_tot = _coverage(polys)
        hole = _enclosed_hole(polys)
        gapfree = (n_cov == n_tot)
        nsides = set(len(p) for p in polys)
        ok = (ncong == 0 and gapfree and n_over == 0 and hole == 0
              and nsides == {9} and len(polys) > 0)
        all_ok = all_ok and ok
        print("%s tiles=%5d sides=%s congruent=%s noncong=%d "
              "cover=%d/%d overlaps=%d enclosed=%s  %s"
              % (name, len(polys), sorted(nsides), ncong == 0, ncong,
                 n_cov, n_tot, n_over, hole == 0,
                 "OK" if ok else "BAD"))
    print("RESULT:", "OK" if all_ok else "BAD")
