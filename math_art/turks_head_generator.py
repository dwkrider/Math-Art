
# Turk's-Head Knot generator for Blender: the classic decorative
# rope knots -- a single cord wound around a surface in L "leads"
# (passes the long way) and B "bights" (scallops the other way),
# woven strictly alternately over and under itself, swept as a
# round rope tube.
#
# Combinatorially a Turk's head THK(B bights, L leads) is the
# ALTERNATING diagram of the (L, B) torus knot: the cord's shadow
# on the surface is the standard rose diagram of T(L, B) (theta =
# L t winding around, transverse oscillation cos(B t)), with every
# crossing woven one-over-one-under in strict alternation along
# the cord.  It is a single cord iff gcd(L, B) = 1; otherwise it
# splits into gcd(L, B) parallel cords (a Turk's-head link).  The
# diagram has B (L - 1) crossings.
#
# The cord can be wound on four scaffolds: a flat ANNULUS (the
# classic cylindrical Turk's head opened flat -- a woven mat or
# ring), the side of a CYLINDER (the knot as tied around a rail or
# as a woggle), the outer face of a TORUS (a Turk's head dressed
# around a ring core), and a SPHERE (the spherical Turk's head, a
# woven ball cover).  On each, the strands do NOT lie on the
# smooth surface: at every crossing the cord rides over or under
# by an offset along the surface normal, smooth-stepped along
# arclength between crossings, so the rope genuinely weaves.
#
# Crossings are exact segment-segment intersections of the wound
# cord with itself in the annulus diagram (the (theta, transverse)
# domain shared by all four surfaces maps homeomorphically onto
# the flat annulus, so the diagram's crossings are computed once
# there); the strictly alternating over/under assignment reduces
# to a parity relation between consecutive crossings along each
# cord, solved with a parity union-find -- every planar diagram
# admits such an assignment (checkerboard shading), so the weave
# always comes out consistent.
#
# References:
#   Clifford W. Ashley, "The Ashley Book of Knots" (Doubleday,
#     1944), ch. 17 -- the Turk's-head: leads, bights, the
#     gcd(L, B) = 1 single-cord rule ("the law of the common
#     divisor"), cylindrical and spherical forms.
#   Robert G. Scharein, "Interactive Topological Drawing" (PhD
#     thesis, The University of British Columbia, 1998), sec. 6.2
#     -- the Turk's head as the torus-knot shadow rewoven
#     alternately (the DT code with its minus signs removed).
#   "Turk's head knots and links: a survey", arXiv:2409.20106
#     (2024) -- THK(B, L) as alternating (L, B) torus-knot
#     diagrams, component count gcd(L, B), crossing count
#     B (L - 1).
#   Dale Rolfsen, "Knots and Links" (1976) -- torus knots T(p, q).

bl_info = {
    "name": "Turk's-Head Knots",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Turk's-Head Knot",
    "description": "Turk's-head knots (B bights x L leads) woven "
                   "over/under as rope on a ring, cylinder, torus "
                   "or sphere",
    "category": "Add Mesh",
}

from math import gcd, pi

import numpy as np

try:                                  # inside the math_art package
    from .curve_frames import welded_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import welded_tube

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from . import knot_carpet_generator as kcg
except ImportError:                     # CLI / standalone use
    import knot_carpet_generator as kcg


# --------------------------------------------------------------------
# The wound cord: (theta, psi) parameter curves
# --------------------------------------------------------------------
#
# Cord k of the (L leads, B bights) Turk's head, with d = gcd(L, B):
#   theta(t) = (L/d) t                (around the surface, the leads)
#   psi(t)   = (B/d) t + 2 pi k / L   (transverse phase; cos(psi) is
#                                      the bight oscillation)
# for t in [0, 2 pi).  d = 1 is the single-cord Turk's head; d > 1
# gives d parallel cords (each a reduced (L/d, B/d) winding), the
# same phase spacing as the (p, q) torus link's components.

def cord_components(leads, bights, samples):
    """[(theta, psi)] parameter arrays, one pair per cord (d =
    gcd(leads, bights) cords), each sampled at `samples` points with
    no duplicated seam point."""
    L = max(1, int(leads))
    B = max(1, int(bights))
    d = gcd(L, B)
    Lr, Br = L // d, B // d
    t = np.linspace(0.0, 2.0 * pi, int(samples), endpoint=False)
    return [(Lr * t, Br * t + 2.0 * pi * k / L) for k in range(d)]


_DOM_AMP = 0.35     # transverse amplitude of the crossing domain


def domain_path(theta, psi):
    """The cord in the flat-annulus DIAGRAM domain, r = 1 +
    0.35 cos(psi) at angle theta, as an (S, 2) closed polyline.
    Every surface maps the (theta, cos psi) cylinder domain
    monotonically, so crossings found here are the crossings on
    every surface (the amplitude does not change which parameter
    pairs meet)."""
    r = 1.0 + _DOM_AMP * np.cos(psi)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


# --------------------------------------------------------------------
# Crossings (segment-segment, vectorized; kcg supplies pair-vs-pair)
# --------------------------------------------------------------------

def _self_cross(P, eps=1e-12):
    """All self-intersections of one closed polyline P ((S, 2);
    segment i runs P[i] -> P[i+1 mod S]).  Returns [(fi, fj)] with
    fi < fj, the fractional arc positions of the two passes.
    Half-open [0, 1) segment parameters count each crossing once;
    adjacent segments (shared endpoints) are excluded."""
    P = np.asarray(P, float)
    n = len(P)
    r = np.roll(P, -1, axis=0) - P
    denom = (r[:, None, 0] * r[None, :, 1]
             - r[:, None, 1] * r[None, :, 0])
    CA = P[None, :, :] - P[:, None, :]
    tnum = CA[:, :, 0] * r[None, :, 1] - CA[:, :, 1] * r[None, :, 0]
    unum = CA[:, :, 0] * r[:, None, 1] - CA[:, :, 1] * r[:, None, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = tnum / denom
        u = unum / denom
    i_idx = np.arange(n)[:, None]
    j_idx = np.arange(n)[None, :]
    sep = (j_idx - i_idx) % n
    hit = ((np.abs(denom) > eps) & (j_idx > i_idx)
           & (sep >= 2) & (sep <= n - 2)
           & (t >= 0.0) & (t < 1.0) & (u >= 0.0) & (u < 1.0))
    return [(float(i + t[i, j]), float(j + u[i, j]))
            for i, j in zip(*np.nonzero(hit))]


def find_crossings(domains):
    """Every crossing of the diagram: self-crossings of each cord
    plus pairwise crossings between cords.  Returns [(key, ca, cb,
    fa, fb)]: unique key, the two cord ids (equal for a
    self-crossing) and the fractional arc position of each pass."""
    crossings = []
    for a, P in enumerate(domains):
        for fi, fj in _self_cross(P):
            crossings.append((len(crossings), a, a, fi, fj))
    for a in range(len(domains)):
        for b in range(a + 1, len(domains)):
            for fa, fb in kcg._seg_cross(domains[a], domains[b]):
                crossings.append((len(crossings), a, b, fa, fb))
    return crossings


# --------------------------------------------------------------------
# Strictly alternating over/under (parity union-find)
# --------------------------------------------------------------------

class _ParityDSU:
    """Union-find with an XOR parity on each element relative to its
    root.  union(x, y, rel) asserts parity(x) ^ parity(y) = rel and
    reports whether that was satisfiable."""

    def __init__(self):
        self.parent = {}
        self.par = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.par[x] = 0

    def find(self, x):
        self.add(x)
        chain = []
        root, p = x, 0
        while self.parent[root] != root:
            chain.append((root, p))
            p ^= self.par[root]
            root = self.parent[root]
        for node, pn in chain:          # path compression
            self.parent[node] = root
            self.par[node] = p ^ pn
        return root, p

    def union(self, x, y, rel):
        rx, px = self.find(x)
        ry, py = self.find(y)
        if rx == ry:
            return (px ^ py) == rel
        self.parent[rx] = ry
        self.par[rx] = px ^ py ^ rel
        return True


def solve_alternating(crossings):
    """One bit per crossing: over[key] = 1 means the FIRST-listed
    pass (slot 0) rides over.  Walking any cord, consecutive
    crossing passes must alternate over/under -- with o = over[k] ^
    slot that is over[k1] ^ over[k2] = 1 ^ s1 ^ s2, a pure parity
    relation.  Returns (over, per_cord, consistent); per_cord maps
    cord id -> [(f, key, slot)] sorted along the cord.  A planar
    diagram always admits the alternating assignment, so consistent
    is True for every Turk's head."""
    per = {}
    for key, ca, cb, fa, fb in crossings:
        per.setdefault(ca, []).append((fa, key, 0))
        per.setdefault(cb, []).append((fb, key, 1))
    dsu = _ParityDSU()
    consistent = True
    for ent in per.values():
        ent.sort()
        C = len(ent)
        if C < 2:
            continue
        for a in range(C):
            _f1, k1, s1 = ent[a]
            _f2, k2, s2 = ent[(a + 1) % C]
            if k1 == k2:
                # the same crossing met twice in a row: its two
                # slots already read opposite senses of one bit
                continue
            if not dsu.union(k1, k2, 1 ^ s1 ^ s2):
                consistent = False
    over = {}
    for key, _ca, _cb, _fa, _fb in crossings:
        _root, p = dsu.find(key)
        over[key] = p
    return over, per, consistent


def cord_signed(crossings, over, ncords):
    """{cord id: [(f, +1 over / -1 under)]} sorted along each cord
    -- the fractional crossing positions with the solved weave
    sense at each pass."""
    signed = {a: [] for a in range(ncords)}
    for key, ca, cb, fa, fb in crossings:
        signed[ca].append((fa, 1 if over[key] else -1))
        signed[cb].append((fb, -1 if over[key] else 1))
    for a in signed:
        signed[a].sort()
    return signed


# --------------------------------------------------------------------
# Surfaces: parameter curve -> 3D centerline + unit surface normal
# --------------------------------------------------------------------

def surface_curve(surface, theta, psi, radius=0.8, radius2=0.3,
                  spread=0.35):
    """(P, N): the (S, 3) cord centerline on the chosen surface and
    the (S, 3) unit surface normal along it (the weave-offset
    direction).  `spread` (0..0.95) sets how far the bights swing
    across the surface; `radius2` is the torus tube radius."""
    w = np.cos(psi)
    ct, st = np.cos(theta), np.sin(theta)
    S = len(w)
    if surface == 'CYLINDER':
        z = 1.6 * radius * spread * w
        P = np.column_stack([radius * ct, radius * st, z])
        N = np.column_stack([ct, st, np.zeros(S)])
    elif surface == 'TORUS':
        chi = pi * min(1.6 * spread, 0.97) * w
        cc, sc = np.cos(chi), np.sin(chi)
        rr = radius + radius2 * cc
        P = np.column_stack([rr * ct, rr * st, radius2 * sc])
        N = np.column_stack([cc * ct, cc * st, sc])
    elif surface == 'SPHERE':
        lat = 0.5 * pi * min(1.6 * spread, 0.97) * w
        cl, sl = np.cos(lat), np.sin(lat)
        N = np.column_stack([cl * ct, cl * st, sl])
        P = radius * N
    else:                                          # RING (annulus)
        rr = radius * (1.0 + min(spread, 0.95) * w)
        P = np.column_stack([rr * ct, rr * st, np.zeros(S)])
        N = np.column_stack([np.zeros(S), np.zeros(S), np.ones(S)])
    return P, N


def _weave_offsets(P, signed_f, depth):
    """Per-vertex signed offset (along the surface normal) of one
    closed cord: reaches +depth at each over-pass and -depth at each
    under-pass (anchored at the FRACTIONAL crossing positions), and
    smoothsteps along 3D arclength between consecutive crossings --
    the arclength mirror of the strapwork engines' _weave_zoff."""
    P = np.asarray(P, float)
    n = len(P)
    z = np.zeros(n)
    if not signed_f or depth <= 0.0:
        return z
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    total = float(seg.sum())

    def arc(f):
        i = int(f) % n
        return float(s[i] + (f - int(f)) * seg[i])

    anchors = sorted((arc(f), depth * (1 if sg > 0 else -1))
                     for f, sg in signed_f)
    fs, fz = anchors[0]
    ls, lz = anchors[-1]
    ext = [(ls - total, lz)] + anchors + [(fs + total, fz)]
    for i in range(n):
        si = s[i]
        for j in range(len(ext) - 1):
            s0, z0 = ext[j]
            s1, z1 = ext[j + 1]
            if si <= s1 or j == len(ext) - 2:
                span = s1 - s0
                u = 0.0 if span < 1e-12 else min(
                    1.0, max(0.0, (si - s0) / span))
                f = u * u * (3.0 - 2.0 * u)        # smoothstep
                z[i] = z0 + (z1 - z0) * f
                break
    return z


# --------------------------------------------------------------------
# The full build
# --------------------------------------------------------------------

def build_turks_head(surface='RING', leads=3, bights=5, radius=0.8,
                     radius2=0.3, spread=0.35, samples=512,
                     weave_depth=0.07):
    """The complete woven Turk's head.  Returns a dict:
      components -- gcd(leads, bights), the cord count
      paths      -- one (S, 3) woven centerline per cord (surface
                    point + normal * offset)
      offsets    -- the per-vertex weave offsets
      crossings  -- [(key, ca, cb, fa, fb)] (see find_crossings)
      over       -- {key: bit} (see solve_alternating)
      signed     -- {cord: [(f, +1/-1)]} (see cord_signed)
      consistent -- whether strict alternation was satisfiable
    """
    comps = cord_components(leads, bights, samples)
    domains = [domain_path(th, ps) for th, ps in comps]
    crossings = find_crossings(domains)
    over, _per, consistent = solve_alternating(crossings)
    signed = cord_signed(crossings, over, len(comps))
    paths, offsets = [], []
    for a, (th, ps) in enumerate(comps):
        P, N = surface_curve(surface, th, ps, radius, radius2,
                             spread)
        z = _weave_offsets(P, signed[a], weave_depth)
        paths.append(P + z[:, None] * N)
        offsets.append(z)
    return dict(components=len(comps), paths=paths, offsets=offsets,
                crossings=crossings, over=over, signed=signed,
                consistent=consistent)


def turks_head_tubes(core, tube_radius=0.055, tube_sides=12):
    """One (verts, quad faces) rope tube per cord, swept along the
    woven centerlines with the carpet engine's proximity-welded
    rotation-minimizing sweep."""
    sides = max(3, int(tube_sides))
    tr = max(0.002, float(tube_radius))
    return [welded_tube([tuple(p) for p in P], tr, sides)
            for P in core['paths']]


def fit_cells(cells, target=2.0):
    """Recenter the merged bounding box on the origin and uniformly
    scale so the largest extent equals `target` (the 2 m cube).
    cells = [(verts, faces)]; returns (new_cells, scale)."""
    allv = np.concatenate([np.asarray(v, float)
                           for v, _f in cells if len(v)])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    c = 0.5 * (lo + hi)
    ext = float((hi - lo).max())
    sc = target / ext if ext > 1e-12 else 1.0
    out = []
    for v, f in cells:
        V = (np.asarray(v, float) - c) * sc
        out.append(([tuple(p) for p in V], f))
    return out, sc


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_turks_head_add(bpy.types.Operator):
        """Add a Turk's-head knot -- a cord wound in L leads and B
        bights, woven strictly alternately over and under, as a rope
        tube on a ring, cylinder, torus or sphere (gcd(L, B) > 1
        gives that many parallel cords)"""
        bl_idname = "mesh.turks_head_add"
        bl_label = "Turk's-Head Knot"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            description="Which surface the cord is wound on",
            items=[('RING', "Ring (Flat Mat)",
                    "The classic Turk's head opened flat -- a "
                    "woven annular mat"),
                   ('CYLINDER', "Cylinder",
                    "Wound around a cylinder, as tied on a rail "
                    "(a woggle)"),
                   ('TORUS', "Torus",
                    "Dressed around the outer face of a ring "
                    "core"),
                   ('SPHERE', "Sphere",
                    "The spherical Turk's head -- a woven ball "
                    "cover")],
            default='RING')
        leads: IntProperty(
            name="Leads", default=3, min=2, max=12,
            description="Passes the long way around (the braid's "
                        "strand count); gcd(leads, bights) cords")
        bights: IntProperty(
            name="Bights", default=5, min=1, max=24,
            description="Scallops along each edge; coprime to "
                        "leads gives a single cord")
        radius: FloatProperty(
            name="Radius", default=0.8, min=0.05, max=10.0,
            description="Main radius of the surface the cord is "
                        "wound on")
        radius2: FloatProperty(
            name="Tube Radius (Torus)", default=0.3, min=0.02,
            max=10.0,
            description="Minor (tube) radius of the torus core")
        spread: FloatProperty(
            name="Spread", default=0.35, min=0.05, max=0.95,
            description="How far the bights swing across the "
                        "surface (fraction)")
        samples: IntProperty(
            name="Samples", default=512, min=96, max=2048,
            description="Points per cord")
        weave_depth: FloatProperty(
            name="Weave Depth", default=0.07, min=0.0, max=0.5,
            step=1, precision=3,
            description="Over/under amplitude along the surface "
                        "normal (floored at 1.3 x tube radius so "
                        "the strands clear each other)")
        tube_radius: FloatProperty(
            name="Rope Radius", default=0.055, min=0.005, max=0.5,
            step=1, precision=3,
            description="Radius of the swept rope tube")
        tube_sides: IntProperty(name="Rope Sides", default=12,
                                min=3, max=32,
                                description="Sides of the rope tube "
                                            "cross-section")
        color_components: BoolProperty(
            name="Color Cords", default=True,
            description="One material with a distinct color per "
                        "cord (multi-cord Turk's heads only)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size multiplier")

        def execute(self, context):
            lift = max(self.weave_depth, 1.3 * self.tube_radius)
            core = build_turks_head(
                self.surface, self.leads, self.bights, self.radius,
                self.radius2, self.spread, self.samples, lift)
            d = core['components']
            cells = turks_head_tubes(core, self.tube_radius,
                                     self.tube_sides)
            cells, _sc = fit_cells(cells, 2.0)
            name = (f"Turk's Head ({self.leads}Lx{self.bights}B)"
                    if d == 1 else
                    f"Turk's Head Link ({self.leads}Lx"
                    f"{self.bights}B, {d} cords)")
            color = self.color_components and d > 1
            verts, faces, midx = [], [], []
            for k, (v, f) in enumerate(cells):
                base = len(verts)
                verts.extend(tuple(np.asarray(p) * self.scale)
                             for p in v)
                faces.extend([[base + i for i in face]
                              for face in f])
                midx.extend([k] * len(f))
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            if color and len(me.polygons) == len(midx):
                me.polygons.foreach_set('material_index', midx)
            me.update()
            obj = bpy.data.objects.new(name, me)
            if color:
                import colorsys
                for k in range(d):
                    rgb = colorsys.hsv_to_rgb(k / d, 0.72, 0.9)
                    mat = bpy.data.materials.new(f"{name} C{k + 1}")
                    mat.diffuse_color = (*rgb, 1.0)
                    mat.use_nodes = True
                    node = mat.node_tree.nodes.get(
                        "Principled BSDF")
                    if node:
                        node.inputs["Base Color"].default_value \
                            = (*rgb, 1.0)
                    obj.data.materials.append(mat)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            ncross = len(core['crossings'])
            self.report(
                {'INFO'},
                f"{name}: {d} cord(s), {ncross} crossings, "
                f"alternating weave "
                f"{'OK' if core['consistent'] else 'INCONSISTENT'}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'surface')
            for k in ('leads', 'bights', 'radius'):
                lay.prop(self, k)
            if self.surface == 'TORUS':
                lay.prop(self, 'radius2')
            for k in ('spread', 'samples', 'weave_depth',
                      'tube_radius', 'tube_sides'):
                lay.prop(self, k)
            if gcd(self.leads, self.bights) > 1:
                lay.prop(self, 'color_components')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.turks_head_add",
                             icon='CURVE_NCIRCLE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_turks_head_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_turks_head_add)


# --------------------------------------------------------------------
# Standalone self-test (pure Python, no Blender)
# --------------------------------------------------------------------

def _interp(A, f):
    """Cyclic linear interpolation of per-vertex data A at
    fractional position f."""
    A = np.asarray(A, float)
    n = len(A)
    i = int(f) % n
    fr = f - int(f)
    return (1.0 - fr) * A[i] + fr * A[(i + 1) % n]


def _selftest():
    ok_all = True
    depth = 0.07
    for L, B in ((3, 5), (2, 3), (4, 3), (5, 4), (2, 1),
                 (2, 4), (3, 6), (4, 6)):
        d = gcd(L, B)
        core = build_turks_head('RING', L, B, samples=600,
                                weave_depth=depth)
        paths = core['paths']
        ncross = len(core['crossings'])
        ok = core['components'] == d
        ok = ok and ncross == B * (L - 1)
        ok = ok and core['consistent']
        # cords closed: the seam step is an ordinary segment
        for P in paths:
            seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
            ok = ok and seg[-1] < 4.0 * seg[:-1].mean()
            ok = ok and np.isfinite(P).all()
        # strict alternation along every cord (cyclic)
        for a, ent in core['signed'].items():
            sg = [s for _f, s in ent]
            for i in range(len(sg)):
                ok = ok and sg[i] != sg[(i + 1) % len(sg)]
        # every crossing one-over-one-under, strands separated
        for key, ca, cb, fa, fb in core['crossings']:
            za = _interp(core['offsets'][ca], fa)
            zb = _interp(core['offsets'][cb], fb)
            dz = za - zb if core['over'][key] else zb - za
            ok = ok and dz >= 1.2 * depth       # over is over
            pa = _interp(paths[ca], fa)
            pb = _interp(paths[cb], fb)
            ok = ok and (np.linalg.norm(pa - pb)
                         >= 1.2 * depth)
        ok_all = ok_all and ok
        print(f"THK {L}Lx{B}B: cords={core['components']} "
              f"(want {d}) crossings={ncross} "
              f"(want {B * (L - 1)}) "
              f"alternating={core['consistent']} "
              f"{'OK' if ok else 'BAD'}")
    # every surface: woven tubes, all-quad, finite, separated
    for surf in ('RING', 'CYLINDER', 'TORUS', 'SPHERE'):
        core = build_turks_head(surf, 3, 5, samples=600,
                                weave_depth=depth)
        for key, ca, cb, fa, fb in core['crossings']:
            pa = _interp(core['paths'][ca], fa)
            pb = _interp(core['paths'][cb], fb)
            ok_all = ok_all and (np.linalg.norm(pa - pb)
                                 >= 1.2 * depth)
        cells = turks_head_tubes(core, 0.055, 10)
        cells, sc = fit_cells(cells, 2.0)
        allv = np.concatenate([np.asarray(v)
                               for v, _f in cells])
        quads = all(len(face) == 4 for _v, f in cells
                    for face in f)
        nonempty = all(len(v) > 0 and len(f) > 0
                       for v, f in cells)
        fits = (np.abs(allv).max() <= 1.0 + 1e-9
                and abs((allv.max(0) - allv.min(0)).max()
                        - 2.0) < 1e-9)
        ok = (quads and nonempty and fits
              and np.isfinite(allv).all())
        ok_all = ok_all and ok
        print(f"{surf}: tubes quads={quads} fits 2m "
              f"cube={fits} {'OK' if ok else 'BAD'}")
    print("RESULT:", "OK" if ok_all else "FAIL")
    assert ok_all
