
# Noble Facetings for Blender
#
# Faceting is the operation dual to stellation: instead of extending a
# solid's face planes outward until they cut out new cells, you keep its
# VERTICES and run new faces through them.  Every stellation of a solid
# corresponds to a faceting of that solid's dual -- Anthony Smith puts it
# plainly, that "the dual of each stellation of the triakis tetrahedron is
# a solid formed by faceting the truncated tetrahedron".
#
# This generator emits the NOBLE facetings, the ones that are isohedral
# and isogonal at once -- the property that singles out the nine regular
# polyhedra, and which a handful of other solids share.  Restricting to
# noble is what makes the search finite and quick: a noble solid's
# vertices are one orbit of its symmetry group and so are its faces, so a
# candidate is fixed by a single face together with the group, and the
# engine can simply try every plane through three or more of the vertices.
#
# The classic check on all this is the icosahedron's twelve vertices,
# which carry exactly four regular and therefore noble polyhedra -- the
# icosahedron itself, the great dodecahedron, the small stellated
# dodecahedron and the great icosahedron.  Choosing the Icosahedron seed
# below walks that set.
#
# THE TWO NOBLE ICOSAHEDRAL DUALS ARE HERE, on the Dodecahedron seed.
# Beyond the regular ones, the icosahedron has exactly two noble
# stellations -- Du Val's D and H, H being the final stellation -- and
# their duals were long recorded as unbuildable, because reciprocating
# the VISIBLE BOUNDARY a stellation engine emits gives the wrong solid:
# that boundary's "vertices" are every intersection point of the
# surface, not the polyhedron's true corners.  Faceting sidesteps the
# problem entirely by never reciprocating anything.
#
# Both duals turn out to be 60-triangle facetings of the dodecahedron
# (V=20, E=90, F=60, chi=-10), and the dodecahedral vertex set carries
# exactly two such, so the match is forced rather than chosen.  They are
# told apart by the shape of that triangle, with the dodecahedron scaled
# to circumradius 1 and its edge a = 2/(sqrt(3)*phi) = 0.71364:
#
#   dual of Du Val D    isosceles (a, 2sqrt2/sqrt3, 2sqrt2/sqrt3)
#   dual of Du Val H    isosceles (2sqrt2/sqrt3, 2sqrt2/sqrt3, phi^2 a)
#
# so the two share their long pair of sides and differ in the base.
# `_verify_noble_duals` re-derives the correspondence on every run,
# rather than trusting the enumeration order.
#
# The mathematics lives in `polyhedra/faceting.py`; this module is the
# Blender layer over it.
#
# References:
# - A. G. Smith, "Stellations of the triakis tetrahedron", The
#   Mathematical Gazette 49 (1965), 135-143, for the stellation/faceting
#   duality and an early non-uniform faceting.
# - H. S. M. Coxeter, P. Du Val, H. T. Flather & J. F. Petrie, "The
#   Fifty-Nine Icosahedra" (1938).
# - B. Grunbaum, "Polyhedra with hollow faces", in Polytopes: Abstract,
#   Convex and Computational (1994), on noble polyhedra.
# - M. Bruckner, "Vielecke und Vielflache" (1900), which first pictured
#   several of the noble solids.

bl_info = {
    "name": "Noble Facetings",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Noble facetings of a vertex set -- new polyhedra "
                   "through the vertices of an existing solid",
    "category": "Add Mesh",
}

import math
import itertools as _it

try:
    from .polyhedra import compounds as _cmp
    from .polyhedra import faceting as _fac
except ImportError:                        # flat import (test runner)
    from polyhedra import compounds as _cmp
    from polyhedra import faceting as _fac

try:
    from .styles import shell as _shell
except ImportError:
    try:
        from styles import shell as _shell
    except ImportError:
        _shell = None


SEEDS = [
    ('ICOSA', "Icosahedron (12 vertices)",
     "carries the four regular polyhedra with icosahedral vertices"),
    ('DODECA', "Dodecahedron (20 vertices)",
     "carries the dodecahedron, the great stellated dodecahedron and "
     "the duals of the icosahedron's two noble stellations"),
    ('CUBE', "Cube (8 vertices)", "octahedral vertex set"),
    ('OCTA', "Octahedron (6 vertices)", "octahedral vertex set"),
    ('ICOSIDODECA', "Icosidodecahedron (30 vertices)",
     "carries the duals of the rhombic triacontahedron's two noble "
     "stellations, Hart's K and 2B"),
    ('TRUNCTETRA', "Truncated Tetrahedron (12 vertices)",
     "Smith's seed: the duals of the triakis tetrahedron's stellations, "
     "including his Fig. 2 solid at two face orbits"),
]

#: seed -> (vertices, group name)
_SEED_GROUP = {'ICOSA': 'Ih', 'DODECA': 'Ih', 'CUBE': 'Oh', 'OCTA': 'Oh',
               'ICOSIDODECA': 'Ih', 'TRUNCTETRA': 'Td'}


def seed_vertices(kind):
    """The vertex set to facet, in the frame of its symmetry group.

    The dodecahedron is built as the icosahedron's FACE CENTRES rather
    than from `compounds._dodeca()`, which is a turned copy and is not
    invariant under this package's icosahedral rotations -- feeding it in
    would make every candidate fail its orbit and the search return
    nothing at all.
    """
    if kind == 'ICOSA':
        V, _F = _cmp._icosa()
        return [tuple(v) for v in V]
    if kind == 'DODECA':
        IV, IF = _cmp._icosa()
        out = []
        for f in IF:
            m = [sum(IV[i][k] for i in f) / len(f) for k in range(3)]
            ln = math.sqrt(sum(c * c for c in m))
            out.append(tuple(c / ln for c in m))
        return out
    if kind == 'TRUNCTETRA':
        # Smith's seed: "the dual of each stellation of the triakis
        # tetrahedron is a solid formed by faceting the truncated
        # tetrahedron".  Even sign changes of (3, 1, 1), which is the
        # tetrahedral frame T_d already works in.
        out = []
        for perm in sorted(set(_it.permutations((3.0, 1.0, 1.0)))):
            for sx in (1.0, -1.0):
                for sy in (1.0, -1.0):
                    for sz in (1.0, -1.0):
                        if sx * sy * sz != 1.0:
                            continue
                        v = (sx * perm[0], sy * perm[1], sz * perm[2])
                        ln = math.sqrt(sum(c * c for c in v))
                        q = tuple(c / ln for c in v)
                        if not any(all(abs(a - b) < 1e-9
                                       for a, b in zip(q, w)) for w in out):
                            out.append(q)
        return out
    if kind == 'ICOSIDODECA':
        # the icosahedron's EDGE midpoints, for the same reason the
        # dodecahedron is its face centres: built this way it is
        # invariant under this package's icosahedral group, which the
        # faceting search requires.  This is the dual of the rhombic
        # triacontahedron, so it is the vertex set carrying the duals of
        # the RT's noble stellations.
        IV, IF = _cmp._icosa()
        seen, out = set(), []
        for f in IF:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                m = [(IV[a][j] + IV[b][j]) / 2.0 for j in range(3)]
                ln = math.sqrt(sum(c * c for c in m))
                out.append(tuple(c / ln for c in m))
        return out
    tetra, cube, octa = _cmp._seeds()
    return [tuple(float(c) for c in v)
            for v in (cube[0] if kind == 'CUBE' else octa[0])]


_CACHE = {}


def facetings_of(kind, orbits=1):
    """Facetings of a seed, cached (the search is O(n^3) in the planes
    and the operator re-runs on every redo-panel tweak).

    `orbits` is how many face orbits a faceting may use.  One means
    isohedral, and with a seed whose vertices are a single orbit that is
    the same as noble -- the default, and every result is a noble solid.
    Two admits the isogonal-but-not-isohedral facetings, which is where
    Smith's Fig. 2 solid lives: its faces are equilateral triangles AND
    alternate-sided hexagons, so no single orbit can hold them.
    """
    key = (kind, orbits)
    if key not in _CACHE:
        V = seed_vertices(kind)
        G = _cmp.GROUPS[_SEED_GROUP[kind]]()
        if orbits <= 1:
            found = _fac.noble_facetings(V, G)
        else:
            found = [(f, None, None)
                     for f in _fac.facetings(V, G, max_orbits=orbits)]
        _CACHE[key] = (V, found)
    return _CACHE[key]


def build(kind, index, orbits=1):
    """(V, F) for one faceting, centred and fitted to a 2 m cube."""
    V, found = facetings_of(kind, orbits)
    if not found:
        raise ValueError('no facetings of %s' % kind)
    faces = found[index % len(found)][0]
    mx = max((abs(c) for v in V for c in v), default=1.0) or 1.0
    return [tuple(c / mx for c in v) for v in V], [list(f) for f in faces]


#: (stellation seed, preset) -> the faceting seed its dual lives on.
#: A stellation's face planes reciprocate to the dual's vertices, so the
#: dual of a stellation of X is always a faceting of X's DUAL: the
#: icosahedron's stellations dualise onto the dodecahedron, the rhombic
#: triacontahedron's onto the icosidodecahedron.  These four are the
#: noble stellations beyond the regular ones -- Hart's D, H, K and 2B.
NOBLE_DUALS = {
    ('icosahedron', 'duval_d'): 'DODECA',
    ('icosahedron', 'final'): 'DODECA',
    ('rhombic_triacontahedron', 'noble_suw'): 'ICOSIDODECA',
    ('rhombic_triacontahedron', 'final'): 'ICOSIDODECA',
}

_SEED_BUILDER = {
    'icosahedron': '_icosahedron_V',
    'rhombic_triacontahedron': '_rhombic_triacontahedron_V',
}


def _frame_map(src, dst):
    """Indices j with dst[j] == R @ src[i], for the rotation R carrying
    one copy of a vertex set onto another.

    Needed because the stellation engine and this package build their
    solids in different frames -- the same solid, turned.  Rather than
    hunt for R in closed form, take one ordered pair of vertices in each
    set with matching norms and angle, build an orthonormal frame from
    each, and test the resulting map on the whole set.
    """
    import numpy as _np
    S = _np.array(src, float)
    T = _np.array(dst, float)
    key = {tuple(_np.round(v, 5)): j for j, v in enumerate(T)}

    def frame(a, b):
        e1 = a / _np.linalg.norm(a)
        e2 = b - (b @ e1) * e1
        n2 = _np.linalg.norm(e2)
        if n2 < 1e-9:
            return None
        e2 = e2 / n2
        return _np.array([e1, e2, _np.cross(e1, e2)])

    a0, b0 = S[0], None
    for v in S[1:]:
        if _np.linalg.norm(_np.cross(a0, v)) > 1e-6:
            b0 = v
            break
    F0 = frame(a0, b0)
    d0 = float(a0 @ b0)
    for a in T:
        if abs(_np.linalg.norm(a) - _np.linalg.norm(a0)) > 1e-6:
            continue
        for b in T:
            if abs(float(a @ b) - d0) > 1e-6:
                continue
            F1 = frame(a, b)
            if F1 is None:
                continue
            R = F1.T @ F0
            out = []
            for v in S:
                k = tuple(_np.round(R @ v, 5))
                if k not in key:
                    out = None
                    break
                out.append(key[k])
            if out is not None and len(set(out)) == len(S):
                return out
    return None


def noble_dual_index(preset, seed='icosahedron'):
    """Which faceting is the dual of stellation `preset` of `seed`, as
    (faceting seed, index), or None.

    Derived, not tabulated.  A noble stellation is isogonal, so all its
    true vertices lie on one sphere -- and the engine's extra boundary
    points lie strictly inside it, so the OUTERMOST emitted points are
    exactly the true vertices and nothing else is.  Each of those sits on
    three of the seed's face planes, so under polar reciprocation it
    becomes a triangle on the three dual vertices those planes
    reciprocate to.  Collect those triangles and the dual's whole face
    set is in hand, with no reciprocation of the boundary mesh anywhere
    in it -- which is the step that had always given the wrong solid.
    """
    try:
        from . import general_stellation as _gs
    except ImportError:                      # flat import (test runner)
        import general_stellation as _gs
    import numpy as _np
    kind = NOBLE_DUALS.get((seed, preset))
    if kind is None:
        return None
    eng = _gs.StellationEngine(getattr(_gs, _SEED_BUILDER[seed])(), name=seed)
    code = next((c for k, _t, c, _n in _gs.named_presets(seed)
                 if k == preset), None)
    if code is None:
        return None
    V, _F = eng.build(_gs.expand_code(eng, code), scale=False)
    V = _np.array(V)
    r = _np.linalg.norm(V, axis=1)
    true = V[r.max() - r < 1e-6]

    DV, found = facetings_of(kind)
    pmap = _frame_map([tuple(float(c) for c in n) for n in eng.N], DV)
    if pmap is None:
        return None
    proj = true @ eng.N.T
    want = {frozenset(pmap[i] for i in
                      _np.nonzero(_np.abs(proj[k] - eng.D) < 1e-6)[0])
            for k in range(len(true))}
    for j, (faces, _nx, _k) in enumerate(found):
        if {frozenset(f) for f in faces} == want:
            return kind, j
    return None


def smith_fig2_index():
    """Which two-orbit faceting of the truncated tetrahedron is the solid
    of Smith's Fig. 2, or None.

    Picked out by his own description rather than by an index.  He
    reports it as the dual of stellation A + B + C + E of the triakis
    tetrahedron and says its faces are "of two types, equilateral
    triangles and alternate-sided hexagons, three of whose sides are
    diameters" -- and, crucially, that "these vertices can be joined in
    only one non-trivial way to form a solid whose faces are regular and
    semi-regular polygons".  So the test is his uniqueness claim: apply
    the description and exactly one candidate beyond the truncated
    tetrahedron itself should survive.

    "Diameters" means diameters of the hexagon's OWN circumcircle, not of
    the solid -- the truncated tetrahedron has no antipodal vertices at
    all, so no face side can pass through the centre.  A hexagon whose
    alternate sides are three such diameters closes only by winding twice
    about its centre, which is why the face is semi-regular rather than
    regular and why the solid is not among the uniform polyhedra.
    """
    V, found = facetings_of('TRUNCTETRA', orbits=2)
    hits = []
    for j, entry in enumerate(found):
        faces = entry[0]
        sides = set(len(f) for f in faces)
        if sides != {3, 6}:
            continue
        tri = [f for f in faces if len(f) == 3][0]
        hexa = [f for f in faces if len(f) == 6][0]

        def lengths(f):
            return [math.dist(V[f[i]], V[f[(i + 1) % len(f)]])
                    for i in range(len(f))]

        tl = lengths(tri)
        if max(tl) - min(tl) > 1e-6:
            continue                       # triangles must be equilateral
        hl = lengths(hexa)
        if abs(hl[0] - hl[2]) > 1e-6 or abs(hl[1] - hl[3]) > 1e-6 \
                or abs(hl[0] - hl[1]) < 1e-6:
            continue                       # hexagon must alternate
        P = [V[i] for i in hexa]
        cen = [sum(q[k] for q in P) / 6.0 for k in range(3)]
        R = sum(math.dist(q, cen) for q in P) / 6.0
        if sum(1 for L in hl if abs(L - 2.0 * R) < 1e-6) != 3:
            continue                       # three sides must be diameters
        hits.append(j)
    return hits[0] if len(hits) == 1 else None


def _verify_smith_fig2():
    """Smith's uniqueness claim, as a test.

    He asserts the truncated tetrahedron's vertices "can be joined in
    only one non-trivial way to form a solid whose faces are regular and
    semi-regular polygons".  Six two-orbit facetings have triangles and
    hexagons; his description leaves exactly one, and it is not the
    truncated tetrahedron itself.
    """
    j = smith_fig2_index()
    assert j is not None, "Smith's Fig. 2 solid is not uniquely picked out"
    V, found = facetings_of('TRUNCTETRA', orbits=2)
    faces = found[j][0]
    n3 = sum(1 for f in faces if len(f) == 3)
    n6 = sum(1 for f in faces if len(f) == 6)
    assert (n3, n6) == (4, 4), ('expected four triangles and four '
                                'hexagons', n3, n6)
    nv, ne, nf, chi = _fac.describe(V, faces)
    assert chi == 2, ('Smith Fig. 2 should close with chi = 2', chi)
    # and it must not be the truncated tetrahedron, whose hexagons are
    # regular -- the "non-trivial" in his sentence
    hexa = [f for f in faces if len(f) == 6][0]
    hl = [math.dist(V[hexa[i]], V[hexa[(i + 1) % 6]]) for i in range(6)]
    assert max(hl) - min(hl) > 1e-6, \
        'that is the truncated tetrahedron itself, not Fig. 2'
    return j, (nv, ne, nf, chi)


def _verify_noble_duals():
    """All four non-regular noble stellations have their duals here.

    Two of them, on the icosahedron, are forced by counting: both duals
    are 60-triangle facetings of the dodecahedron and that vertex set
    carries exactly two such.  The rhombic triacontahedron's are not --
    the icosidodecahedron carries four 120-triangle noble facetings --
    so those are matched face-set for face-set instead, and the check
    that they land on different ones is doing real work.
    """
    _DV, found = facetings_of('DODECA')
    tri = [j for j, (faces, _n, _k) in enumerate(found)
           if len(faces) == 60 and len(faces[0]) == 3]
    assert len(tri) == 2, ('expected exactly two 60-triangle noble '
                           'facetings of the dodecahedron', tri)
    idx = {}
    for (seed, preset), kind in sorted(NOBLE_DUALS.items()):
        got = noble_dual_index(preset, seed)
        assert got is not None, (seed, preset, 'dual not found')
        assert got[0] == kind, (seed, preset, got[0], kind)
        idx[(seed, preset)] = got[1]
    for seed in ('icosahedron', 'rhombic_triacontahedron'):
        pair = [v for (s, _p), v in idx.items() if s == seed]
        assert len(set(pair)) == 2, \
            (seed, 'its two noble stellations must have different duals',
             pair)
    return idx


def _selftest():
    for kind, _lbl, _d in SEEDS:
        V, found = facetings_of(kind)
        sizes = sorted((len(f[0]), len(f[0][0])) for f in found)
        print('%-7s %2d vertices -> %d noble facetings %s'
              % (kind, len(V), len(found), sizes))

    # the icosahedral vertex set must yield the four regulars
    _V, found = facetings_of('ICOSA')
    assert len(found) == 4, ('expected the four regular polyhedra',
                             len(found))

    # every emitted mesh must close up: each edge in exactly two faces,
    # and every vertex used, which is what makes it a polyhedron rather
    # than an arrangement of polygons
    for kind, _lbl, _d in SEEDS:
        _v, found = facetings_of(kind)
        for i in range(len(found)):
            V, F = build(kind, i)
            mult = {}
            for f in F:
                for j in range(len(f)):
                    e = tuple(sorted((f[j], f[(j + 1) % len(f)])))
                    mult[e] = mult.get(e, 0) + 1
            assert set(mult.values()) == {2}, (kind, i, 'edge multiplicity')
            assert {j for f in F for j in f} == set(range(len(V))), \
                (kind, i, 'a vertex went unused')
            assert max(abs(c) for v in V for c in v) <= 1.0 + 1e-9, \
                (kind, i, 'not fitted to the 2 m cube')

    j, stats = _verify_smith_fig2()
    print('SMITH   Fig. 2 solid = Truncated Tetrahedron faceting %d at two '
          'orbits, V=%d E=%d F=%d chi=%d' % (j, *stats))

    idx = _verify_noble_duals()
    for (seed, preset), j in sorted(idx.items()):
        print('DUALS   %-24s %-10s -> %s faceting %d'
              % (seed, preset, NOBLE_DUALS[(seed, preset)], j))
    print('RESULT: OK')


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, IntProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_noble_faceting_add(bpy.types.Operator):
        """Add a noble faceting: a new polyhedron through the vertices of
        an existing solid, isohedral and isogonal at once"""
        bl_idname = "mesh.noble_faceting_add"
        bl_label = "Noble Faceting"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Vertices", items=SEEDS, default='ICOSA',
            description="Which solid's vertices to run new faces through")
        index: IntProperty(
            name="Faceting", default=0, min=0, max=63,
            description="Which of the noble facetings of that vertex "
                        "set to build; they are found in plane order and "
                        "the count differs per seed, so this wraps")
        orbits: IntProperty(
            name="Face Orbits", default=1, min=1, max=2,
            description="How many orbits of faces a faceting may use. "
                        "One keeps every result noble -- isohedral and "
                        "isogonal at once. Two also finds the isogonal "
                        "solids whose faces come in two shapes, such as "
                        "Smith's, whose faces are equilateral triangles "
                        "and alternate-sided hexagons")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        if _shell is not None:
            __annotations__.update(_shell.style_properties())

        def execute(self, context):
            try:
                V, F = build(self.seed, self.index, self.orbits)
            except Exception as e:          # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            V = [tuple(c * self.scale for c in v) for v in V]
            # count from the SAME list the mesh was built from -- asking
            # for the one-orbit list here labelled a two-orbit faceting
            # "4/4" when there are sixteen, and wrapped the number
            # against the wrong total
            _v, found = facetings_of(self.seed, self.orbits)
            name = "%s %d/%d" % (
                "Noble Faceting" if self.orbits <= 1 else "Faceting",
                self.index % len(found) + 1, len(found))
            if _shell is not None:
                obj = _shell.apply(self, context, V, F, name)
            else:
                obj = None
            if obj is None and _shell is None:
                me = bpy.data.meshes.new(name)
                me.from_pydata(V, [], [tuple(f) for f in F])
                me.validate(clean_customdata=True)
                me.update()
                obj = bpy.data.objects.new(name, me)
                context.collection.objects.link(obj)
                obj.location = context.scene.cursor.location
                context.view_layer.objects.active = obj
            self.report({'INFO'},
                        "%s: %d faces of %d sides"
                        % (name, len(F), len(F[0])))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.prop(self, 'seed')
            lay.prop(self, 'index')
            if _shell is not None:
                _shell.draw_style(self, lay)
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.noble_faceting_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_noble_faceting_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_noble_faceting_add)
