
# Polytwister generator for Blender.
#
# Polytwisters are curved 4D shapes built from a polyhedron via the Hopf
# fibration (Jonathan Bowers, ~2007: 222 uniform + 3 infinite families).
# Where a polyhedron is the intersection of half-spaces -- one per face --
# a "hard" polytwister is a Boolean tree over CYCLOPLANES: a cycloplane is
# the Hopf preimage of a spherical cap around a point on S^2.  Both live in
# R^4 and are invariant under the Hopf circle action, so one visualises
# them by a 3-space cross-section at a slice coordinate w, swept to animate.
#
# This module is a faithful port of Nathan Ho's reference toolchain
# (github.com/polytwisters), whose CadQuery Realizer walks a JSON-able tree
# of {cycloplane, intersection, union, difference, rotated_copies} nodes.
# The load-bearing closed form: the 3-space cross-section, at slice w, of a
# single cycloplane with zenith z, azimuth a (theta = z/2) is a transformed
# cylinder -- from a unit Z-axis cylinder: rotate +90 deg about X, scale X
# by 1/cos(theta), translate X by w*tan(theta), rotate by -theta about X and
# a about Y.  The south-pole cycloplane (theta -> pi/2) is instead a slab of
# half-height sqrt(1-w^2) and large radius (empty for |w| >= 1).  The
# polytwister cross-section is the Boolean combination of these per-node
# cylinders; sweeping w slides/stretches them, tracing the ring / strip /
# twister cells.  The exact zenith constants and per-shape trees below are
# transcribed verbatim from the reference `core/hard_polytwisters.py`.
#
# References:
# - Jonathan Bowers, "Polytwisters" (c. 2007),
#   https://www.polytope.net/hedrondude/twisters.htm (discovery; names and
#   the 222 + 3 infinite families).
# - Nathan Ho, "Polytwisters", https://nathan.ho.name/pages/polytwisters/
#   and github.com/polytwisters/{polytwisters, polytwister-mesher,
#   polytwisters.com} (the cycloplane cross-section and Boolean DSL ported
#   here).
# - Heinz Hopf, Math. Ann. 104 (1931) 637-665 (the fibration underlying the
#   cycloplane construction).

bl_info = {
    "name": "Polytwister",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polytwister",
    "description": "3D cross-section of a polytwister (Hopf-fibration "
                   "4D->3D, after Bowers / Ho): a Boolean tree over "
                   "cycloplane cylinders, sweepable in w",
    "category": "Add Mesh",
}

import math
from math import cos, sin, tan, pi, sqrt

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

PHI = (1.0 + sqrt(5.0)) / 2.0
LARGE = 100.0
EPSILON = 1e-3


# --------------------------------------------------------------------------
# Zenith constants (verbatim from reference core/hard_polytwisters.py)
# --------------------------------------------------------------------------

def _sq_dist(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2


def get_3d_angle(a, b, c):
    """Angle ABC in radians (law of cosines on squared distances)."""
    a2 = _sq_dist(b, c)
    b2 = _sq_dist(a, c)
    c2 = _sq_dist(a, b)
    return math.acos((a2 + c2 - b2) / (2.0 * sqrt(a2 * c2)))


TETRAHEDRON_ZENITH = get_3d_angle((1, 1, 1), (0, 0, 0), (1, -1, -1))
OCTAHEDRON_ZENITH = get_3d_angle((1, 0, 0), (0, 0, 0), (1, 1, 1))
DODECAHEDRON_ZENITH = get_3d_angle((0, 1, PHI), (0, 0, 0), (0, -1, PHI))
ICOSAHEDRON_ZENITH_1 = get_3d_angle((1, 1, 1), (0, 0, 0), (0, 1, PHI))
ICOSAHEDRON_ZENITH_2 = get_3d_angle((1, 1, 1), (0, 0, 0), (0, -1, PHI))


# --------------------------------------------------------------------------
# DSL node constructors (dicts, exactly as the reference)
# --------------------------------------------------------------------------

def cycloplane(zenith, azimuth):
    return {"type": "cycloplane", "zenith": zenith, "azimuth": azimuth}


def intersection(nodes):
    return {"type": "intersection", "operands": nodes}


def difference(nodes):
    return {"type": "difference", "operands": nodes}


def union(nodes):
    return {"type": "union", "operands": nodes}


def rotated_copies(node, order):
    return {"type": "rotated_copies", "order": order, "operand": node}


NORTH_POLE = cycloplane(0, 0)
SOUTH_POLE = cycloplane(pi, 0)

TETRAHEDRON_NORTH = [cycloplane(pi - TETRAHEDRON_ZENITH, i * 2 * pi / 3)
                     for i in range(3)]
CUBE_EQUATOR = [cycloplane(pi / 2, i * pi / 2) for i in range(4)]
OCTAHEDRON_NORTH = [cycloplane(OCTAHEDRON_ZENITH, i * pi / 2)
                    for i in range(4)]
OCTAHEDRON_SOUTH = [cycloplane(pi - OCTAHEDRON_ZENITH, i * pi / 2)
                    for i in range(4)]
DODECAHEDRON_NORTH = [cycloplane(DODECAHEDRON_ZENITH, i * 2 * pi / 5)
                      for i in range(5)]
DODECAHEDRON_SOUTH = [cycloplane(pi - DODECAHEDRON_ZENITH,
                                 (i + 0.5) * 2 * pi / 5) for i in range(5)]
ICOSAHEDRON_NORTH_1 = [cycloplane(ICOSAHEDRON_ZENITH_1, i * 2 * pi / 5)
                       for i in range(5)]
ICOSAHEDRON_NORTH_2 = [cycloplane(ICOSAHEDRON_ZENITH_2, i * 2 * pi / 5)
                       for i in range(5)]
ICOSAHEDRON_SOUTH_1 = [cycloplane(pi - ICOSAHEDRON_ZENITH_1,
                                  (i + 0.5) * 2 * pi / 5) for i in range(5)]
ICOSAHEDRON_SOUTH_2 = [cycloplane(pi - ICOSAHEDRON_ZENITH_2,
                                  (i + 0.5) * 2 * pi / 5) for i in range(5)]


# --------------------------------------------------------------------------
# Per-shape construction trees (verbatim from reference)
# --------------------------------------------------------------------------

def _tetratwister():
    return intersection([SOUTH_POLE] + TETRAHEDRON_NORTH)


def _cubetwister():
    return intersection([NORTH_POLE, SOUTH_POLE] + CUBE_EQUATOR)


def _octatwister():
    return intersection(OCTAHEDRON_NORTH + OCTAHEDRON_SOUTH)


def _dodecatwister():
    return intersection([NORTH_POLE, SOUTH_POLE]
                        + DODECAHEDRON_NORTH + DODECAHEDRON_SOUTH)


def _icosatwister():
    return intersection(ICOSAHEDRON_NORTH_1 + ICOSAHEDRON_NORTH_2
                        + ICOSAHEDRON_SOUTH_1 + ICOSAHEDRON_SOUTH_2)


def _quasitetratwister():
    o = TETRAHEDRON_NORTH
    ring_1 = difference([intersection(o), SOUTH_POLE])
    ring_2 = rotated_copies(
        difference([intersection([SOUTH_POLE, o[0], o[1]]), o[2]]), 3)
    return union([ring_1, ring_2])


def _bloated_tetratwister():
    o = TETRAHEDRON_NORTH
    ring_1 = difference([intersection([o[0], o[1]]), SOUTH_POLE, o[2]])
    ring_2 = difference([intersection([SOUTH_POLE, o[0]]), o[1], o[2]])
    return union([rotated_copies(ring_1, 3), rotated_copies(ring_2, 3)])


def _quasicubetwister():
    e = CUBE_EQUATOR
    north = difference([intersection([NORTH_POLE, e[0], e[1]]),
                        intersection([SOUTH_POLE, e[2], e[3]])])
    south = difference([intersection([SOUTH_POLE, e[0], e[1]]),
                        intersection([NORTH_POLE, e[2], e[3]])])
    return union([rotated_copies(north, 4), rotated_copies(south, 4)])


def _bloated_cubetwister():
    e = CUBE_EQUATOR
    return union([rotated_copies(intersection([NORTH_POLE, e[0]]), 4),
                  rotated_copies(intersection([e[0], e[1]]), 4),
                  rotated_copies(intersection([e[0], SOUTH_POLE]), 4)])


def _quasioctatwister():
    n, s = OCTAHEDRON_NORTH, OCTAHEDRON_SOUTH
    return union([intersection(n),
                  rotated_copies(
                      intersection([n[0], n[1], s[0], s[1]]), 4),
                  intersection(s)])


def _bloated_octatwister():
    n, s = OCTAHEDRON_NORTH, OCTAHEDRON_SOUTH
    return union([rotated_copies(intersection([n[0], n[1]]), 4),
                  rotated_copies(intersection([n[0], s[0]]), 4),
                  rotated_copies(intersection([s[0], s[1]]), 4)])


def _quasidodecatwister():
    n, s = DODECAHEDRON_NORTH, DODECAHEDRON_SOUTH
    r1 = intersection([NORTH_POLE, n[0], n[1]])
    r2 = intersection([n[0], n[1], s[0]])
    r3 = intersection([n[1], s[0], s[1]])
    r4 = intersection([SOUTH_POLE, s[0], s[1]])
    return union([rotated_copies(r, 5) for r in (r1, r2, r3, r4)])


def _bloated_dodecatwister():
    n, s = DODECAHEDRON_NORTH, DODECAHEDRON_SOUTH
    rings = [intersection([NORTH_POLE, n[0]]),
             intersection([n[0], n[1]]),
             intersection([n[0], s[0]]),
             intersection([s[0], n[1]]),
             intersection([s[0], s[1]]),
             intersection([SOUTH_POLE, s[0]])]
    return union([rotated_copies(r, 5) for r in rings])


def _quasicosatwister():
    n1, n2 = ICOSAHEDRON_NORTH_1, ICOSAHEDRON_NORTH_2
    s2, s1 = ICOSAHEDRON_SOUTH_2, ICOSAHEDRON_SOUTH_1
    r1 = intersection(ICOSAHEDRON_NORTH_1)
    r2 = intersection([n1[0], n1[1], n2[0], n2[1], s2[0]])
    r3 = intersection([s1[0], s1[1], s2[0], s2[1], n2[1]])
    r4 = intersection(ICOSAHEDRON_SOUTH_1)
    return union([r1, rotated_copies(r2, 5), rotated_copies(r3, 5), r4])


def _bloated_icosatwister():
    n1, n2 = ICOSAHEDRON_NORTH_1, ICOSAHEDRON_NORTH_2
    s2, s1 = ICOSAHEDRON_SOUTH_2, ICOSAHEDRON_SOUTH_1
    rings = [intersection([n1[0], n1[1]]),
             intersection([n1[0], n2[0]]),
             intersection([n2[0], s2[0]]),
             intersection([n2[1], s2[0]]),
             intersection([s1[0], s2[0]]),
             intersection([s1[0], s1[1]])]
    return union([rotated_copies(r, 5) for r in rings])


def _great_dodecahedron_edges():
    n, s = DODECAHEDRON_NORTH, DODECAHEDRON_SOUTH
    return [((n[0], n[2]), (NORTH_POLE, n[1])),
            ((NORTH_POLE, s[0]), (n[0], n[1])),
            ((n[1], s[-1]), (n[0], s[0])),
            ((n[0], s[1]), (n[1], s[0])),
            ((SOUTH_POLE, n[1]), (s[0], s[1])),
            ((s[0], s[2]), (SOUTH_POLE, s[1]))]


def _great_dodecatwister():
    rings = [difference([intersection(e1), union(list(e2))])
             for e1, e2 in _great_dodecahedron_edges()]
    return union([rotated_copies(r, 5) for r in rings])


def _great_quasidodecatwister():
    rings = [difference([intersection(e1), intersection(list(e2))])
             for e1, e2 in _great_dodecahedron_edges()]
    return union([rotated_copies(r, 5) for r in rings])


def _great_bloated_dodecatwister():
    rings = [intersection(list(e1))
             for e1, _e2 in _great_dodecahedron_edges()]
    return union([rotated_copies(r, 5) for r in rings])


def _small_stellated_spikes():
    n, s = DODECAHEDRON_NORTH, DODECAHEDRON_SOUTH
    ring_1 = (n, NORTH_POLE)
    ring_2 = ([NORTH_POLE, n[-1], n[1], s[-1], s[0]], n[0])
    ring_3 = ([SOUTH_POLE, s[-1], s[1], n[0], n[1]], s[0])
    ring_4 = (s, SOUTH_POLE)
    return {"poles": [ring_1, ring_4], "rest": [ring_2, ring_3]}


def _small_stellated_dodecatwister():
    parts = []
    spikes = _small_stellated_spikes()
    for spike in spikes["poles"]:
        parts.append(difference([intersection(list(spike[0])), spike[1]]))
    for spike in spikes["rest"]:
        parts.append(rotated_copies(
            difference([intersection(list(spike[0])), spike[1]]), 5))
    return union(parts)


def _small_quasistellated_dodecatwister():
    parts = []
    spikes = _small_stellated_spikes()
    for spike in spikes["poles"]:
        parts.append(intersection(list(spike[0])))
    for spike in spikes["rest"]:
        parts.append(rotated_copies(intersection(list(spike[0])), 5))
    return union(parts)


def _great_icosatwister():
    n1, n2 = ICOSAHEDRON_NORTH_1, ICOSAHEDRON_NORTH_2
    s2, s1 = ICOSAHEDRON_SOUTH_2, ICOSAHEDRON_SOUTH_1
    rings = [intersection([n1[0], s2[2], s1[1], s1[3]]),
             intersection([n2[0], n2[2], n2[3], s1[2]]),
             intersection([s2[0], s2[2], s2[3], n1[3]]),
             intersection([s1[0], n2[3], n1[2], n1[4]])]
    return union([rotated_copies(r, 5) for r in rings])


def dyadic_twister(n):
    return intersection([cycloplane(pi / 2, i * 2 * pi / n)
                         for i in range(n)])


# Registry: enum key -> (label, acronym, tree-builder). Order groups the
# regular family by seed, then great/stellated, matching Bowers' listing.
_SHAPES = [
    ('TETTER', "Tetratwister", "tetter", _tetratwister),
    ('QUITTER', "Quasitetratwister", "quitter", _quasitetratwister),
    ('BLITTER', "Bloated Tetratwister", "blitter", _bloated_tetratwister),
    ('CUBITER', "Cubetwister", "cubiter", _cubetwister),
    ('QUICTER', "Quasicubetwister", "quicter", _quasicubetwister),
    ('BLICTER', "Bloated Cubetwister", "blicter", _bloated_cubetwister),
    ('OCTER', "Octatwister", "octer", _octatwister),
    ('QUOTER', "Quasioctatwister", "quoter", _quasioctatwister),
    ('BLOTER', "Bloated Octatwister", "bloter", _bloated_octatwister),
    ('DOTER', "Dodecatwister", "doter", _dodecatwister),
    ('QUADOTER', "Quasidodecatwister", "quadoter", _quasidodecatwister),
    ('BLADOTER', "Bloated Dodecatwister", "bladoter",
     _bloated_dodecatwister),
    ('IKETER', "Icosatwister", "iketer", _icosatwister),
    ('QUIKETER', "Quasicosatwister", "quiketer", _quasicosatwister),
    ('BLIKETER', "Bloated Icosatwister", "bliketer",
     _bloated_icosatwister),
    ('GADITER', "Great Dodecatwister", "gaditer", _great_dodecatwister),
    ('GAQUIDITER', "Great Quasidodecatwister", "gaquiditer",
     _great_quasidodecatwister),
    ('GABLIDITER', "Great Bloated Dodecatwister", "gabliditer",
     _great_bloated_dodecatwister),
    ('SISSIDITER', "Small Stellated Dodecatwister", "sissiditer",
     _small_stellated_dodecatwister),
    ('SIQUISSIDITER', "Small Quasistellated Dodecatwister",
     "siquissiditer", _small_quasistellated_dodecatwister),
    ('GIKETER', "Great Icosatwister", "giketer", _great_icosatwister),
]
_SHAPE_BY_KEY = {k: (label, acr, fn) for k, label, acr, fn in _SHAPES}


def get_tree(key, dyadic_n=5):
    """Return the DSL tree for an enum key ('DYADIC' uses dyadic_n)."""
    if key == 'DYADIC':
        return dyadic_twister(dyadic_n)
    return _SHAPE_BY_KEY[key][2]()


# --------------------------------------------------------------------------
# Cycloplane cross-section: a transformed cylinder mesh (verts, faces)
# --------------------------------------------------------------------------

def _rx(a):
    import numpy as np
    c, s = cos(a), sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0],
                     [0, s, c, 0], [0, 0, 0, 1]], float)


def _ry(a):
    import numpy as np
    c, s = cos(a), sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0],
                     [-s, 0, c, 0], [0, 0, 0, 1]], float)


def _sx(s):
    import numpy as np
    return np.diag([s, 1.0, 1.0, 1.0])


def _tx(d):
    import numpy as np
    m = np.eye(4)
    m[0, 3] = d
    return m


def _base_cylinder(radius, height, seg):
    """Closed (capped) cylinder about the Z-axis, centred at the origin."""
    import numpy as np
    hz = height / 2.0
    ang = np.linspace(0.0, 2.0 * pi, seg, endpoint=False)
    bot = [(radius * math.cos(a), radius * math.sin(a), -hz) for a in ang]
    top = [(radius * math.cos(a), radius * math.sin(a), hz) for a in ang]
    faces = []
    for i in range(seg):
        j = (i + 1) % seg
        faces.append([i, j, seg + j, seg + i])
    faces.append(list(range(seg - 1, -1, -1)))
    faces.append(list(range(seg, 2 * seg)))
    return bot + top, faces


def create_cycloplane(w, zenith, azimuth, seg=48):
    """Cross-section at slice w of one cycloplane, as (verts, faces), or
    None when it is empty (south-pole cap with |w| >= 1)."""
    import numpy as np
    theta = 0.5 * zenith
    if abs(theta - pi / 2.0) < EPSILON:            # south-pole slab
        if abs(w) >= 1.0:
            return None
        hh = sqrt(1.0 - w * w)
        verts, faces = _base_cylinder(LARGE, 2.0 * hh, seg)
        M = _rx(pi / 2.0)
    else:
        verts, faces = _base_cylinder(1.0, LARGE, seg)
        M = (_ry(azimuth) @ _rx(-theta) @ _tx(w * tan(theta))
             @ _sx(1.0 / cos(theta)) @ _rx(pi / 2.0))
    P = np.asarray(verts, float)
    Ph = np.hstack([P, np.ones((len(P), 1))])
    return (Ph @ M.T)[:, :3].tolist(), faces


if _IN_BLENDER:

    class _Realizer:
        """Blender port of the reference CadQuery Realizer: walks the DSL
        tree, combining per-node cycloplane cylinders with EXACT Boolean
        modifiers, baking intermediate results.  All scratch objects live
        in a temporary collection that the caller deletes afterwards."""

        def __init__(self, context, coll, w, seg):
            self.ctx = context
            self.coll = coll
            self.seg = seg
            self.w = w if abs(w) > EPSILON else EPSILON

        def _obj(self, verts, faces):
            me = bpy.data.meshes.new("_pt")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.update()
            o = bpy.data.objects.new("_pt", me)
            self.coll.objects.link(o)
            return o

        def _remove(self, o):
            if o is not None:
                me = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                if me.users == 0:
                    bpy.data.meshes.remove(me)

        def _bool(self, a, b, op):
            m = a.modifiers.new("bool", 'BOOLEAN')
            m.operation = op
            m.solver = 'EXACT'
            m.object = b
            dg = self.ctx.evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(a.evaluated_get(dg))
            newo = bpy.data.objects.new("_pt", me)
            self.coll.objects.link(newo)
            self._remove(a)
            self._remove(b)
            return newo

        def _intersect(self, a, b):
            if a is None or b is None:
                self._remove(a)
                self._remove(b)
                return None
            return self._bool(a, b, 'INTERSECT')

        def _cut(self, a, b):
            if a is None:
                self._remove(b)
                return None
            if b is None:
                return a
            return self._bool(a, b, 'DIFFERENCE')

        def _union(self, a, b):
            if a is None:
                return b
            if b is None:
                return a
            return self._bool(a, b, 'UNION')

        def _rotated_copies(self, obj, n):
            import numpy as np
            if obj is None:
                return [None] * n
            me = obj.data
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3)
            faces = [list(p.vertices) for p in me.polygons]
            out = []
            for i in range(n):
                R = _ry(i * 2.0 * pi / n)[:3, :3]
                out.append(self._obj((co @ R.T).tolist(), faces))
            self._remove(obj)
            return out

        def traverse(self, node):
            t = node["type"]
            if t == "cycloplane":
                vf = create_cycloplane(self.w, node["zenith"],
                                       node["azimuth"], self.seg)
                return None if vf is None else self._obj(vf[0], vf[1])
            if t == "rotated_copies":
                return self._rotated_copies(
                    self.traverse(node["operand"]), node["order"])
            if t in ("intersection", "difference", "union"):
                result = None
                started = False
                for child in node["operands"]:
                    ops = self.traverse(child)
                    if not isinstance(ops, list):
                        ops = [ops]
                    for op in ops:
                        if not started:
                            result, started = op, True
                        elif t == "intersection":
                            result = self._intersect(result, op)
                        elif t == "difference":
                            result = self._cut(result, op)
                        else:
                            result = self._union(result, op)
                return result
            raise ValueError(t)

    class MESH_OT_polytwister_add(bpy.types.Operator):
        """Add the 3D cross-section of a polytwister -- a Boolean tree over
        cycloplanes (Hopf-fibration 4D->3D, after Bowers / Ho)"""
        bl_idname = "mesh.polytwister_add"
        bl_label = "Polytwister (Experimental)"
        bl_options = {'REGISTER', 'UNDO'}

        shape: EnumProperty(
            name="Polytwister",
            items=([('DYADIC', "Dyadic Twister (n)",
                     "Order-n dyadic twister (dyster)")]
                   + [(k, label, f"{acr} — {label}")
                      for k, label, acr, _fn in _SHAPES]),
            default='CUBITER')
        dyadic_n: IntProperty(
            name="Dyadic n", default=5, min=3, max=24,
            description="Number of cycloplanes (DYADIC only)")
        slice_w: FloatProperty(
            name="Slice w", default=0.0, min=-3.0, max=3.0,
            step=5, precision=4,
            description="4D->3D cross-section coordinate; key this to "
                        "animate the polytwister through its cells")
        seg: IntProperty(
            name="Cylinder Segments", default=24, min=8, max=160,
            description="Facets per cycloplane cylinder (resolution vs. "
                        "Boolean cost; the quasi/bloated/great twisters "
                        "union many solids and get slow at high values)")
        recenter: BoolProperty(
            name="Recenter", default=False,
            description="Recenter the cross-section on its bounding box "
                        "(off keeps a consistent centre across a w-sweep)")
        shade_smooth: BoolProperty(name="Shade Smooth", default=False)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            import numpy as np
            tree = get_tree(self.shape, self.dyadic_n)
            temp = bpy.data.collections.new("_pt_temp")
            context.scene.collection.children.link(temp)
            try:
                final = _Realizer(context, temp, self.slice_w,
                                  self.seg).traverse(tree)
                if final is None or len(final.data.vertices) == 0:
                    self.report({'ERROR'},
                                "Empty cross-section at this slice w "
                                "(try a smaller |w|)")
                    return {'CANCELLED'}
                me = final.data.copy()
            finally:
                for o in list(temp.objects):
                    m = o.data
                    bpy.data.objects.remove(o, do_unlink=True)
                    if m.users == 0:
                        bpy.data.meshes.remove(m)
                bpy.data.collections.remove(temp)

            n_out = len(me.vertices)
            co = np.empty(n_out * 3)
            me.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3)
            if self.recenter:
                co = co - 0.5 * (co.max(0) + co.min(0))
            co = co * self.scale
            me.vertices.foreach_set('co', co.ravel())
            me.polygons.foreach_set(
                'use_smooth', [self.shade_smooth] * len(me.polygons))
            me.update()

            if self.shape == 'DYADIC':
                name = f"Dyadic Twister ({self.dyadic_n})"
            else:
                name = _SHAPE_BY_KEY[self.shape][0]
            me.name = name
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {n_out} verts, slice w={self.slice_w:.3f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'shape')
            if self.shape == 'DYADIC':
                lay.prop(self, 'dyadic_n')
            lay.prop(self, 'slice_w')
            lay.prop(self, 'seg')
            lay.prop(self, 'recenter')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.polytwister_add", icon='MESH_TORUS')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_polytwister_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polytwister_add)


def _selftest():
    import numpy as np
    ok_all = True

    # zenith constants match the known closed forms
    checks = [
        ("tetra", TETRAHEDRON_ZENITH, math.acos(-1.0 / 3.0)),
        ("octa", OCTAHEDRON_ZENITH, math.acos(1.0 / sqrt(3.0))),
    ]
    for nm, got, exp in checks:
        ok = abs(got - exp) < 1e-9
        ok_all = ok_all and ok
        print(f"zenith {nm}: {got:.6f} (want {exp:.6f}) "
              f"{'OK' if ok else 'BAD'}")

    # every registered tree builds and is a well-formed node
    def walk(node):
        assert node["type"] in ("cycloplane", "intersection", "union",
                                "difference", "rotated_copies")
        if node["type"] == "cycloplane":
            return 1
        if node["type"] == "rotated_copies":
            return walk(node["operand"])
        return sum(walk(c) for c in node["operands"])

    for k, label, acr, fn in _SHAPES:
        leaves = walk(fn())
        ok = leaves > 0
        ok_all = ok_all and ok
        print(f"{acr:14s} {label:32s} leaves={leaves} "
              f"{'OK' if ok else 'BAD'}")
    for n in (3, 5, 8):
        walk(dyadic_twister(n))
    print(f"dyadic trees build for n=3,5,8 OK")

    # cycloplane cross-section: finite mesh; south-pole slab empties at |w|>=1
    v, f = create_cycloplane(0.3, OCTAHEDRON_ZENITH, 1.1, seg=24)
    finite = np.isfinite(np.asarray(v)).all() and len(v) == 48
    north = create_cycloplane(0.0, 0.0, 0.0, seg=16)         # NORTH_POLE
    south_in = create_cycloplane(0.5, pi, 0.0, seg=16)       # slab, |w|<1
    south_out = create_cycloplane(1.5, pi, 0.0, seg=16)      # empty
    ok = (finite and north is not None and south_in is not None
          and south_out is None)
    ok_all = ok_all and ok
    print(f"cycloplane: finite={finite} north={north is not None} "
          f"slab|w<1|={south_in is not None} slab|w>=1|empty="
          f"{south_out is None} {'OK' if ok else 'BAD'}")

    assert ok_all
    print(f"polytwister standalone tests passed "
          f"({len(_SHAPES)} named + dyadic family)")
