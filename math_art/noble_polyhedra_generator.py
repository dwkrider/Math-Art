
# Noble Polyhedra for Blender
#
# A polyhedron is NOBLE when its symmetries carry any face onto any other
# face AND any vertex onto any other vertex.  The nine regular polyhedra
# are noble; so, it turns out, are exactly 146 others, plus two infinite
# prismatic families.  Hess found the first non-regular ones in the
# 1870s, Bruckner more by 1907, and after a century's gap Webb, Mikloweit
# and the Stella community found dozens more -- but whether any were still
# missing stayed open until Connor Hill's 2026 enumeration settled it.
#
# Three operators, one per way of using his result:
#
#   Noble Polyhedron      the complete catalogue: all 146, by Hill's
#                         symbol (gD-19.1 is the first faceting of the
#                         19th critical orbit of the gD type), or their
#                         polar duals.
#   Noble Orbit Explorer  the method itself: pick an orbit type, move its
#                         parameters, and the faceting search runs live on
#                         that orbit.  Snap to a critical orbit to see the
#                         polyhedra Hill proved are there; anywhere else the
#                         search comes back empty, which is his theorem.
#   Crown Polyhedron      the two infinite families -- Hess's crown
#                         polyhedra (stephanoids) PC(n, p, q) and
#                         AC(n, p, q), and the disphenoids.
#
# Every catalogue entry is REBUILT by the engine (`polyhedra/noble.py`)
# from an orbit type, a location and one face, never read from a mesh; the
# engine's self-test checks all 146 against measurements of Hill's own
# models.
#
# References:
# - C. Hill, "The complete set of noble polyhedra", arXiv:2607.28711
#   (2026), with code and models at
#   https://github.com/Plasmath/noble-tools-revised (GPL-3.0).
# - E. Hess, "Ueber die zugleich gleicheckigen und gleichflaechigen
#   Polyeder" (1876), and his papers of 1875 and 1877, for the first
#   non-regular noble polyhedra and the stephanoids.
# - M. Bruckner, "Ueber die gleicheckig-gleichflaechigen diskontinuierlichen
#   und nichtkonvexen Polyeder", Nova Acta Leopoldina (1906).
# - B. Grunbaum, "Polyhedra with hollow faces", in Polytopes: Abstract,
#   Convex and Computational (1994), 43-70, which named the noble
#   polyhedra.
# - U. Mikloweit, "Exploring Noble Polyhedra With the Program Stella4D",
#   Bridges 2020 Conference Proceedings, 257-264.

bl_info = {
    "name": "Noble Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "The 146 noble polyhedra, the orbit-type search that "
                   "finds them, and the prismatic noble families",
    "category": "Add Mesh",
}

import math

try:
    from .polyhedra import noble as _nb
    from .polyhedra import _noble_data as _data
except ImportError:                        # flat import (test runner)
    from polyhedra import noble as _nb
    from polyhedra import _noble_data as _data

try:
    from .styles import shell as _shell
except ImportError:
    try:
        from styles import shell as _shell
    except ImportError:
        _shell = None


CATALOGUE = _data.NOBLE

#: orbit types in Hill's order, with the number of catalogue entries
_TYPE_ORDER = list(_nb.ORBIT_TYPES)


def _count(t):
    return sum(1 for e in CATALOGUE if e['type'] == t)


#: vertex-orbit filter for the catalogue: every type that has members
ORBIT_FILTER = [('ALL', "All (146)", "Every noble polyhedron")] + [
    (t, "%s -- %d vertices (%d)" % (t, _nb.ORBIT_TYPES[t][3], _count(t)),
     "Noble polyhedra whose vertices form a %s orbit" % t)
    for t in _TYPE_ORDER if _count(t)]

_DOF_WORD = {0: "fixed", 1: "one parameter", 2: "two parameters"}

#: every orbit type, for the explorer -- including the seven that carry
#: no noble polyhedron at all
ORBIT_TYPE_ITEMS = [
    (t, "%s -- %d vertices, %s" % (t, _nb.ORBIT_TYPES[t][3],
                                   _DOF_WORD[_nb.dof(t)]),
     "The %s orbit type: %d vertices, %s; %d noble polyhedra"
     % (t, _nb.ORBIT_TYPES[t][3], _DOF_WORD[_nb.dof(t)], _count(t)))
    for t in _TYPE_ORDER]


def catalogue_entries(orbit_filter='ALL'):
    return [e for e in CATALOGUE
            if orbit_filter == 'ALL' or e['type'] == orbit_filter]


def by_symbol(symbol):
    return next((e for e in CATALOGUE if e['symbol'] == symbol), None)


def build_entry(e, dual=False):
    """(V, F) for one catalogue entry, fitted to the 2 m cube."""
    V = _nb.orbit(e['type'], e['a'], e['b'])
    F = _nb.build_faces(V, e['group'], e['face'])
    if dual:
        V, F = _nb.dual(V, F)
    return _nb.fit(V), F


def critical_locations(t):
    """[(orbit label, a, b)] of the critical orbits of type t that carry
    noble polyhedra, in Hill's order."""
    out = []
    for e in CATALOGUE:
        if e['type'] == t and all(o[0] != e['orbit'] for o in out):
            out.append((e['orbit'], e['a'], e['b']))
    return out


_SEARCH = {}


def explore(t, a, b):
    """Noble facetings of the orbit T(a, b), cached by location.

    Returns (V, [(group, face)], note).  The note says why the list is
    empty when it is -- the orbit coincides with a more symmetric type,
    or it simply is not critical.
    """
    key = (t, round(a, 9), round(b, 9))
    if key not in _SEARCH:
        V = _nb.orbit(t, a, b)
        extra = _nb.coincides(t, a, b)
        if extra:
            found = []
            note = ("this orbit has %s symmetry, so it belongs to a more "
                    "symmetric orbit type; its facetings are counted there"
                    % extra)
        else:
            found = _nb.noble_facetings(V)
            note = "" if found else (
                "no noble faceting on this orbit -- only critical orbits "
                "carry them (showing the convex hull)")
        if len(_SEARCH) > 64:
            _SEARCH.clear()
        _SEARCH[key] = (V, found, note)
    return _SEARCH[key]


def prismatic(family, n, p, q, height, x, y, z):
    """(V, F, label, warning) for the prismatic families."""
    if family == 'DISPHENOID':
        V, F = _nb.disphenoid(x, y, z)
        kind = ('regular tetrahedron' if abs(x - y) < 1e-9 and
                abs(y - z) < 1e-9 else
                'tetragonal' if min(abs(x - y), abs(y - z),
                                    abs(x - z)) < 1e-9 else 'rhombic')
        return _nb.fit(V), F, "Disphenoid (%s)" % kind, ''
    kind = 'P' if family == 'PRISMATIC' else 'A'
    warn = ''
    if not _nb.stephanoid_ok(n, p, q, kind):
        valid = valid_pq(n, kind)
        if not valid:
            raise ValueError("no %s crown polyhedron has n = %d"
                             % ('prismatic' if kind == 'P' else
                                'antiprismatic', n))
        warn = ("(p, q) = (%d, %d) is not valid for n = %d; using (%d, %d). "
                "Valid: %s" % (p, q, n, valid[0][0], valid[0][1],
                               ", ".join("(%d,%d)" % v for v in valid)))
        p, q = valid[0]
    V, F = _nb.stephanoid(n, p, q, kind, height)
    return (_nb.fit(V), F, "%sC(%d, %d, %d)" % (kind, n, p, q), warn)


def valid_pq(n, kind):
    return [(p, q) for p in range(1, n) for q in range(1, n)
            if _nb.stephanoid_ok(n, p, q, kind)]


def polyhedron_items(orbit_filter='ALL'):
    """Dropdown items for one orbit filter: Hill's symbol, the classical
    name where there is one, and the Schlafli type.  Cached, because
    Blender keeps only references to the strings a dynamic enum returns.
    """
    if orbit_filter not in _ITEMS:
        _ITEMS[orbit_filter] = [
            (e['symbol'],
             "%s%s  {%d, %d}" % (e['symbol'],
                                 " " + e['name'] if e['name'] else "",
                                 e['p'], e['q']),
             "%s: V=%d E=%d F=%d, symmetry %s, dual %s"
             % (e['symbol'], e['V'], e['E'], e['F'], e['group'], e['dual']),
             k)
            for k, e in enumerate(catalogue_entries(orbit_filter))]
    return _ITEMS[orbit_filter]


_ITEMS = {}


def _selftest():
    import numpy as np
    assert len(CATALOGUE) == 146, len(CATALOGUE)
    counts = {}
    for e in CATALOGUE:
        counts[e['type']] = counts.get(e['type'], 0) + 1
    # Hill, Section 5.2: one each in T, O, C, tO, tC, rC; 4 in I, 6 in ID,
    # 7 in D, 17 in tI, 6 in tD, 19 in rD, 7 in sC, 3 in gC, 33 in sD and
    # 38 in gD
    want = {'T': 1, 'O': 1, 'C': 1, 'tO': 1, 'tC': 1, 'rC': 1, 'I': 4,
            'ID': 6, 'D': 7, 'tI': 17, 'tD': 6, 'rD': 19, 'sC': 7, 'gC': 3,
            'sD': 33, 'gD': 38}
    assert counts == want, counts

    # Every entry rebuilds into a closed noble polyhedron matching Hill's
    # own model: its V/E/F, and its in/circumradius ratio measured on HIS
    # file when the catalogue was built.
    syms = {e['symbol'] for e in CATALOGUE}
    for e in CATALOGUE:
        V = _nb.orbit(e['type'], e['a'], e['b'])
        F = _nb.build_faces(V, e['group'], e['face'])
        assert _nb.is_polyhedron(F, len(V)), e['symbol']
        assert _nb._planar(V, F, 1e-7), (e['symbol'], 'non-planar face')
        d = _nb.describe(V, F)
        assert (d['V'], d['E'], d['F'], d['p'], d['q']) == \
            (e['V'], e['E'], e['F'], e['p'], e['q']), (e['symbol'], d)
        assert abs(d['ratio'] - e['ratio']) < 1e-7, (e['symbol'], d, e)
        # the location really is a root of Hill's minimal polynomials
        for key, poly in (('a', e['poly_a']), ('b', e['poly_b'])):
            if poly:
                # relative to the size of the terms: gD-20's b is near 28
                # in a degree-10 polynomial, so terms reach 1e15
                r = float(np.polyval(poly, e[key]))
                mag = float(np.polyval(np.abs(poly), abs(e[key])))
                assert abs(r) < 1e-12 * mag, (e['symbol'], key, r, mag)
        assert e['dual'] in syms or e['dual'].startswith(('D-F', 'tI-F',
                                                          'rD-F')), e
    # the listed duals are consistent: dual of a dual is the entry
    for e in CATALOGUE:
        d = by_symbol(e['dual'])
        if d is not None:
            assert d['dual'] == e['symbol'], (e['symbol'], d['symbol'])
            assert (d['V'], d['F']) == (e['F'], e['V']), e['symbol']
    print('catalogue: 146 entries rebuilt; V/E/F, {p,q}, in/circumradius '
          'ratio and minimal polynomials agree')

    # the polar dual of an entry has its dual's shape (same ratio,
    # swapped counts) -- spot-check a few, including a chiral one
    for s in ('I-2', 'tO-1.1', 'sD-5.2', 'gC-1.1'):
        e = by_symbol(s)
        V, F = build_entry(e, dual=True)
        d = by_symbol(e['dual'])
        got = _nb.describe(V, F)
        assert (got['V'], got['F']) == (d['V'], d['F']), (s, got, d)
        assert abs(got['ratio'] - d['ratio']) < 1e-6, (s, got, d)

    # the explorer at a critical orbit finds what the catalogue lists
    for label, a, b in critical_locations('tI')[:2]:
        _V, found, note = explore('tI', a, b)
        n = sum(1 for e in CATALOGUE if e['orbit'] == label)
        assert len(found) == n and not note, (label, len(found), n, note)
    # ... and away from one, nothing (Hill's finiteness in action)
    _V, found, note = explore('tI', 0.5, 1.0)
    assert not found and note, note
    _V, found, note = explore('rP', _nb.PHI, 1.0)
    assert not found and 'symmetry' in note, note
    # gD-19 is a fissary orbit: each face plane holds two pentagons, and
    # without the coplanar-neighbour check the search also returned seven
    # woven hexagon/heptagon "facetings" there that Hill rightly excludes
    label, a, b = next(c for c in critical_locations('gD') if c[0] == 'gD-19')
    _V, found, note = explore('gD', a, b)
    assert len(found) == 1 and len(found[0][1]) == 5, found

    # the prismatic families, with an invalid (p, q) falling back
    V, F, lbl, warn = prismatic('PRISMATIC', 7, 5, 2, 1.0, 1, 1, 1)
    assert lbl == 'PC(7, 5, 2)' and not warn, (lbl, warn)
    V, F, lbl, warn = prismatic('ANTIPRISMATIC', 7, 1, 1, 1.0, 1, 1, 1)
    assert warn and lbl.startswith('AC(7'), (lbl, warn)
    V, F, lbl, warn = prismatic('DISPHENOID', 5, 3, 1, 1.0, 1.0, 0.6, 0.6)
    assert 'tetragonal' in lbl, lbl
    for kind in ('P', 'A'):
        for n in range(3, 10):
            for p, q in valid_pq(n, kind):
                Vs, Fs = _nb.stephanoid(n, p, q, kind)
                assert _nb._planar(Vs, Fs), (kind, n, p, q)
                assert _nb.is_polyhedron(Fs, len(Vs)), (kind, n, p, q)
    # The display mesh: every face and every solid-display cell wound
    # OUTWARD, cells flat in their face's plane, non-degenerate, and not
    # overlapping (their areas sum to the face's covered area, checked by
    # sampling on a few faces).
    def newell(P):
        n = np.zeros(3)
        for i in range(len(P)):
            n += np.cross(P[i], P[(i + 1) % len(P)])
        return 0.5 * n
    for e in CATALOGUE:
        V, F = build_entry(e)
        for f in _nb.oriented(V, F):
            P = np.asarray(V)[f]
            assert newell(P) @ P.mean(axis=0) >= -1e-12, (e['symbol'], f)
        V2, F2 = display_mesh(V, F)
        V2 = np.asarray(V2)
        for f in F2:
            P = V2[f]
            nv = newell(P)
            assert np.linalg.norm(nv) > 1e-12, (e['symbol'], 'sliver')
            assert nv @ P.mean(axis=0) > 0, (e['symbol'], 'inward cell')
    # the pentagram: 5 points + the central pentagon, total area equal to
    # the region of non-zero winding
    e = by_symbol('D-6')
    V, F = build_entry(e)
    V2, F2 = display_mesh(V, F)
    assert len(F2) == 72 and sorted({len(f) for f in F2}) == [3, 5], len(F2)
    rng = np.random.default_rng(1)
    for s in ('D-6', 'gC-3.1', 'sD-5.2'):
        V, F = build_entry(by_symbol(s))
        V = np.asarray(V)
        f = F[0]
        c, e1, e2, n = _nb._face_frame(V, f)
        B = np.array([e1, e2])
        Q = (V[f] - c) @ B.T
        cells = _nb._face_cells(Q, 1e-9)
        a_cells = sum(_nb._signed_area(P) for P in cells)
        lo, hi = Q.min(axis=0), Q.max(axis=0)
        pts = lo + (hi - lo) * rng.random((40000, 2))
        hit = sum(1 for x, y in pts if _nb._winding(Q, x, y) != 0)
        a_mc = hit / len(pts) * float(np.prod(hi - lo))
        assert abs(a_cells - a_mc) < 0.02 * a_mc, (s, a_cells, a_mc)
    print('display: faces and cells wound outward; cell areas match the '
          'covered region')
    print('RESULT: OK')


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def display_mesh(V, F, style='SOLID'):
    """What actually goes to Blender.

    Every face wound outward.  For the solid style each face is further
    cut into the regions it covers, so star and crossed faces shade as
    one clean surface instead of overlapping triangles; the frame,
    strut and segment styles keep the true faces, whose edges are the
    polyhedron's edges.
    """
    if style == 'SOLID':
        return _nb.visible_regions(V, F)
    return V, _nb.oriented(V, F)


def _emit(op, context, V, F, name):
    V, F = display_mesh(V, F, getattr(op, 'style', 'SOLID'))
    V = [tuple(float(c) * op.scale for c in v) for v in V]
    F = [list(f) for f in F]
    if _shell is not None:
        return _shell.apply(op, context, V, F, name)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], [tuple(f) for f in F])
    me.validate(clean_customdata=True)
    me.update()
    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    obj.location = context.scene.cursor.location
    context.view_layer.objects.active = obj
    return obj


if _IN_BLENDER:

    class MESH_OT_noble_polyhedron_add(bpy.types.Operator):
        """Add one of the 146 noble polyhedra -- every face alike and every
        vertex alike -- from Connor Hill's complete enumeration"""
        bl_idname = "mesh.noble_polyhedron_add"
        bl_label = "Noble Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        def _reset_polyhedron(self, context):
            # a new orbit: start at its first polyhedron rather than leave
            # a choice that is not in the new list
            self.polyhedron = polyhedron_items(self.orbit_filter)[0][0]

        def _polyhedron_items(self, context):
            return polyhedron_items(self.orbit_filter)

        orbit_filter: EnumProperty(
            name="Vertex Orbit", items=ORBIT_FILTER, default='ALL',
            update=_reset_polyhedron,
            description="Show only the noble polyhedra whose vertices form "
                        "one kind of orbit. The symbol is Hill's: T, O, C, "
                        "I, ID, D are the fixed vertex sets of the regular "
                        "and quasiregular solids; t, r, s and g types have "
                        "vertices that slide with one or two parameters")
        polyhedron: EnumProperty(
            name="Polyhedron", items=_polyhedron_items,
            description="Which noble polyhedron of the chosen orbit, by "
                        "Hill's symbol: T-x.y is the y-th faceting (largest "
                        "inradius first) of the x-th critical orbit of type "
                        "T")
        dual: BoolProperty(
            name="Dual", default=False,
            description="Build the polar dual instead -- another noble "
                        "polyhedron, except for four whose duals have "
                        "coinciding vertices")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")

        if _shell is not None:
            __annotations__.update(_shell.style_properties())

        def execute(self, context):
            ents = catalogue_entries(self.orbit_filter)
            e = next((x for x in ents if x['symbol'] == self.polyhedron),
                     ents[0])
            pos = ents.index(e)
            V, F = build_entry(e, self.dual)
            name = "Noble %s%s" % (e['symbol'], " dual" if self.dual else "")
            if e['name'] and not self.dual:
                name += " %s" % e['name']
            _emit(self, context, V, F, name)
            loc = ("" if e['poly_a'] is None else
                   ", a = %.6f" % e['a'] +
                   ("" if e['poly_b'] is None else ", b = %.6f" % e['b']))
            self.report({'INFO'},
                        "%s (%d/%d): {%d, %d}, V=%d E=%d F=%d, symmetry "
                        "%s, dual %s%s%s"
                        % (e['symbol'], pos + 1,
                           len(ents), e['p'], e['q'], e['V'], e['E'],
                           e['F'], e['group'], e['dual'], loc,
                           " -- dual shown" if self.dual else ""))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.prop(self, 'orbit_filter')
            lay.prop(self, 'polyhedron')
            lay.prop(self, 'dual')
            if _shell is not None:
                _shell.draw_style(self, lay)
            lay.prop(self, 'scale')

    class MESH_OT_noble_orbit_add(bpy.types.Operator):
        """Explore Hill's method: choose an orbit type and its parameters,
        and search that orbit live for noble polyhedra"""
        bl_idname = "mesh.noble_orbit_add"
        bl_label = "Noble Orbit Explorer"
        bl_options = {'REGISTER', 'UNDO'}

        orbit_type: EnumProperty(
            name="Orbit Type", items=ORBIT_TYPE_ITEMS, default='tI',
            description="Which family of vertex sets to search. Each is "
                        "the orbit of one point under a symmetry group; "
                        "t, r, s and g types have one or two free "
                        "parameters")
        critical: IntProperty(
            name="Critical Orbit", default=1, min=0, max=64,
            description="0 uses the Position sliders freely; 1, 2, ... "
                        "snap to the critical orbits of this type that "
                        "carry noble polyhedra, in Hill's order; stops at "
                        "the last one")
        a: FloatProperty(
            name="Position A", default=0.5, min=0.001, max=10.0,
            precision=6,
            description="First free parameter of the orbit: the generating "
                        "point's distance from the first mirror")
        b: FloatProperty(
            name="Position B", default=0.5, min=0.001, max=10.0,
            precision=6,
            description="Second free parameter, for two-parameter types")
        faceting: IntProperty(
            name="Faceting", default=0, min=0, max=64,
            description="Which noble faceting found on this orbit; stops "
                        "at the last one")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")

        if _shell is not None:
            __annotations__.update(_shell.style_properties())

        def execute(self, context):
            t = self.orbit_type
            k = _nb.dof(t)
            crit = critical_locations(t)
            label = None
            if k == 0:
                a, b = 1.0, 1.0
                label = t if crit else None
            elif self.critical > 0 and crit:
                if self.critical > len(crit):
                    self.critical = len(crit)       # bounded, not wrapped
                label, a, b = crit[self.critical - 1]
            else:
                a, b = self.a, (self.b if k == 2 else 1.0)
            V, found, note = explore(t, a, b)
            where = t if k == 0 else "%s(a=%.6f%s)" % (
                t, a, ", b=%.6f" % b if k == 2 else "")
            if found:
                if self.faceting >= len(found):
                    self.faceting = len(found) - 1
                g, face = found[self.faceting]
                F = _nb.build_faces(V, g, face)
                d = _nb.describe(V, F)
                sym = self._symbol(label, V, F)
                name = "Noble %s" % (sym or where)
                _emit(self, context, _nb.fit(V), F, name)
                self.report({'INFO'},
                            "%s: faceting %d/%d, {%d, %d}, V=%d E=%d F=%d, "
                            "symmetry %s%s"
                            % (where, self.faceting + 1,
                               len(found), d['p'], d['q'], d['V'], d['E'],
                               d['F'], g, " = " + sym if sym else ""))
            else:
                F = _nb.convex_hull(V)
                _emit(self, context, _nb.fit(V), F, "Orbit %s" % t)
                hint = (" This type has %d critical orbits with noble "
                        "polyhedra; set Critical Orbit to 1..%d."
                        % (len(crit), len(crit)) if crit else
                        " Hill proved this type carries none at all.")
                self.report({'WARNING'}, "%s: %s.%s" % (where, note, hint))
            return {'FINISHED'}

        @staticmethod
        def _symbol(label, V, F):
            """Hill's symbol for a faceting found at a catalogued orbit."""
            if label is None:
                return None
            want = frozenset(_nb._edges(f) for f in F)
            full = _nb.transitive_groups(V)[0][1]
            for e in CATALOGUE:
                if e['orbit'] != label:
                    continue
                G = _nb.build_faces(V, e['group'], e['face'])
                if any(frozenset(_nb._edges(_nb._apply(p, f)) for f in G)
                       == want for p in full):
                    return e['symbol']
            return None

        def draw(self, context):
            lay = self.layout
            lay.prop(self, 'orbit_type')
            k = _nb.dof(self.orbit_type)
            if k:
                lay.prop(self, 'critical')
                if self.critical == 0:
                    lay.prop(self, 'a')
                    if k == 2:
                        lay.prop(self, 'b')
            lay.prop(self, 'faceting')
            if _shell is not None:
                _shell.draw_style(self, lay)
            lay.prop(self, 'scale')

    class MESH_OT_noble_prismatic_add(bpy.types.Operator):
        """Add a prismatic noble polyhedron: Hess's crown polyhedra
        (stephanoids) or a disphenoid"""
        bl_idname = "mesh.noble_prismatic_add"
        bl_label = "Crown Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Family",
            items=[('PRISMATIC', "Prismatic Crown",
                    "PC(n, p, q): crossed quadrilaterals between two "
                    "aligned regular n-gons"),
                   ('ANTIPRISMATIC', "Antiprismatic Crown",
                    "AC(n, p, q): crossed quadrilaterals on the vertices "
                    "of an n-gonal antiprism; q must be odd"),
                   ('DISPHENOID', "Disphenoid",
                    "Four congruent triangles -- the only convex noble "
                    "polyhedra that are not regular")],
            default='PRISMATIC',
            description="Which infinite family of prismatic noble "
                        "polyhedra")
        n: IntProperty(name="Sides", default=7, min=3, max=48,
                       description="n: the number of sides of the base "
                                   "polygons")
        p: IntProperty(name="Step P", default=5, min=1, max=47,
                       description="p: how far round the face's long "
                                   "diagonal reaches (1 <= p < n)")
        q: IntProperty(name="Step Q", default=2, min=1, max=47,
                       description="q: how far round the face's short "
                                   "side reaches; an invalid pair falls "
                                   "back to the first valid one")
        height: FloatProperty(name="Height", default=1.0, min=0.01,
                              max=20.0,
                              description="Distance between the two "
                                          "layers of vertices, relative to "
                                          "the base circumradius -- any "
                                          "height stays noble")
        width: FloatProperty(name="Width", default=1.0, min=0.01, max=10.0,
                             description="Disphenoid box width")
        depth: FloatProperty(name="Depth", default=0.7, min=0.01, max=10.0,
                             description="Disphenoid box depth")
        tall: FloatProperty(name="Box Height", default=0.45, min=0.01,
                            max=10.0,
                            description="Disphenoid box height; two equal "
                                        "sides make it tetragonal, three "
                                        "the regular tetrahedron")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")

        if _shell is not None:
            __annotations__.update(_shell.style_properties())

        def execute(self, context):
            try:
                V, F, label, warn = prismatic(
                    self.family, self.n, self.p, self.q, self.height,
                    self.width, self.depth, self.tall)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            _emit(self, context, V, F, label)
            d = _nb.describe(V, F)
            msg = "%s: V=%d E=%d F=%d" % (label, d['V'], d['E'], d['F'])
            self.report({'WARNING'} if warn else {'INFO'},
                        msg + (" -- " + warn if warn else ""))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.prop(self, 'family')
            if self.family == 'DISPHENOID':
                lay.prop(self, 'width')
                lay.prop(self, 'depth')
                lay.prop(self, 'tall')
            else:
                lay.prop(self, 'n')
                lay.prop(self, 'p')
                lay.prop(self, 'q')
                lay.prop(self, 'height')
            if _shell is not None:
                _shell.draw_style(self, lay)
            lay.prop(self, 'scale')

    _CLASSES = (MESH_OT_noble_polyhedron_add, MESH_OT_noble_orbit_add,
                MESH_OT_noble_prismatic_add)

    def _menu_func(self, context):
        for c in _CLASSES:
            self.layout.operator(c.bl_idname, icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        for c in _CLASSES:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_CLASSES):
            bpy.utils.unregister_class(c)
