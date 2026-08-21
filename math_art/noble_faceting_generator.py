
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
     "carries the dodecahedron and great stellated dodecahedron"),
    ('CUBE', "Cube (8 vertices)", "octahedral vertex set"),
    ('OCTA', "Octahedron (6 vertices)", "octahedral vertex set"),
]

#: seed -> (vertices, group name)
_SEED_GROUP = {'ICOSA': 'Ih', 'DODECA': 'Ih', 'CUBE': 'Oh', 'OCTA': 'Oh'}


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
    tetra, cube, octa = _cmp._seeds()
    return [tuple(float(c) for c in v)
            for v in (cube[0] if kind == 'CUBE' else octa[0])]


_CACHE = {}


def facetings_of(kind):
    """Every noble faceting of a seed, cached (the search is O(n^3) in
    the planes and the operator re-runs on every redo-panel tweak)."""
    if kind not in _CACHE:
        V = seed_vertices(kind)
        G = _cmp.GROUPS[_SEED_GROUP[kind]]()
        _CACHE[kind] = (V, _fac.noble_facetings(V, G))
    return _CACHE[kind]


def build(kind, index):
    """(V, F) for one faceting, centred and fitted to a 2 m cube."""
    V, found = facetings_of(kind)
    if not found:
        raise ValueError('no noble facetings of %s' % kind)
    faces, _nx, _k = found[index % len(found)]
    mx = max((abs(c) for v in V for c in v), default=1.0) or 1.0
    return [tuple(c / mx for c in v) for v in V], [list(f) for f in faces]


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
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        if _shell is not None:
            __annotations__.update(_shell.style_properties())

        def execute(self, context):
            try:
                V, F = build(self.seed, self.index)
            except Exception as e:          # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            V = [tuple(c * self.scale for c in v) for v in V]
            _v, found = facetings_of(self.seed)
            name = "Noble Faceting %d/%d" % (
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
