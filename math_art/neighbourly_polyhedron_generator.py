# Neighbourly Polyhedra for Blender
#
# Two classical toroids pose two extremal questions.  In the Szilassi
# polyhedron every face shares an edge with every other face; in its dual,
# the Csaszar polyhedron, every vertex is joined to every other and the
# torus is built on the fewest vertices any triangulated torus can have.
# Both are genus 1, and both live with the other toroids (see
# toroidal_polyhedron_generator.py).  This generator carries the same two
# questions to higher genus.
#
# FACE-NEIGHBOURLY.  If every pair of faces shares exactly one edge and
# three faces meet at every vertex, Euler's formula forces the genus to be
# (n-3)(n-4)/12 for n faces, so after the tetrahedron (4) and Szilassi (7)
# the next case is 12 faces of genus 6, which is open.  Letting a pair of
# faces meet along TWO collinear edges opens up eight faces, and two such
# polyhedra are known, both of genus 3 with eight non-convex nonagons, 24
# vertices and 36 edges, 20 face pairs meeting once and 8 twice:
#
#   * Mizhaev's polyhedron -- integer vertices, a fourfold rotoreflection;
#     its doubled pairs form one 8-cycle.
#   * The Rost-Vigh polyhedron -- rational vertices, three half-turn axes,
#     chiral; its doubled pairs form two 4-cycles, which is what shows the
#     two are combinatorially different.
#
# Mizhaev's first variant of the same size is included for contrast: it has
# the same counts and symmetry, but four pairs of its faces never meet.
#
# FEWEST VERTICES.  Heawood's inequality bounds the vertices of a
# triangulated surface from below; genus 2 and genus 3 both need ten.  The
# two ten-vertex polyhedra here are realisations with the smallest integer
# coordinates known, found by exhaustive search -- in a 4x4x4 and a 5x5x5
# cube.
#
# Painting a solid as a map -- faces sharing an edge get different colours
# -- shows the face property directly: a face-neighbourly solid needs one
# colour per face (eight here), Mizhaev's first variant only four.
#
# The exact data and every check (incidence, closed and oriented, genus,
# embedded, symmetry, combinatorial type) are in polyhedra/neighbourly.py
# and polyhedra/vertex_minimal.py, in integer and rational arithmetic.
#
# References:
# - Gergely Rost and Viktor Vigh, "A second eight-faced polyhedron in which
#   every two faces share an edge", arXiv:2609.32998 (2026).
# - Ruslan Mizhaev, "Integer realization of an equivelar octahedron of
#   genus 3", arXiv:2609.17700 (2026).
# - Ruslan Mizhaev, "Equivelar octahedron of genus 3 in 3-space", OSF
#   Preprints (2020), doi:10.31219/osf.io/hvtey -- both of his variants.
# - Stefan Hougardy, Frank H. Lutz and Mariano Zelke, "Polyhedra of genus 2
#   with 10 vertices and minimal coordinates", Electronic Geometry Models
#   No. 2005.08.001 (2007), arXiv:math/0507592.
# - Stefan Hougardy, Frank H. Lutz and Mariano Zelke, "Polyhedra of genus 3
#   with 10 vertices and minimal coordinates", Electronic Geometry Models
#   No. 2006.02.001 (2007), arXiv:math/0604017.
# - Lajos Szilassi, "Regular toroids", Structural Topology 13 (1986),
#   69-80; Akos Csaszar, "A polyhedron without diagonals", Acta Sci. Math.
#   Szeged 13 (1949-50), 140-142 -- the genus-1 cases.
# - Percy J. Heawood, "Map-colour theorem", Quarterly Journal of Pure and
#   Applied Mathematics 24 (1890), 332-338 -- the vertex bound, and the
#   colouring bound the face-neighbourly solids illustrate.
# - John Philip Huneke, "A minimum-vertex triangulation", Journal of
#   Combinatorial Theory, Series B 24 (1978), 258-266 -- genus 2 needs ten.

bl_info = {
    "name": "Neighbourly Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Higher-genus successors of the Szilassi and Csaszar "
                   "polyhedra: genus-3 solids whose faces all meet, and "
                   "polyhedra of genus 2 and 3 on the fewest vertices",
    "category": "Add Mesh",
}

try:
    from .polyhedra import neighbourly as _nb
    from .polyhedra import vertex_minimal as _vm
    from .polyhedra.fit import fit_cube as _fit_cube
except ImportError:
    from polyhedra import neighbourly as _nb
    from polyhedra import vertex_minimal as _vm
    from polyhedra.fit import fit_cube as _fit_cube


SOLIDS = {}
for _kind, _meta in _nb.NEIGHBOURLY.items():
    _V, _F = _nb.build(_kind)
    SOLIDS[_kind] = {"name": _meta["name"], "V": _V, "F": _F, "note": None}
for _kind, _meta in _vm.VERTEX_MINIMAL.items():
    _V, _F = _vm.build(_kind)
    SOLIDS[_kind] = {"name": _meta["name"], "V": _V, "F": _F,
                     "note": "the fewest vertices a polyhedron of genus "
                             "%d can have" % _meta["genus"]}

SOLID_ITEMS = [
    ("ROST_VIGH", "Rost-Vigh Polyhedron", "Genus 3: 8 nonagons, every pair "
     "sharing an edge (8 pairs sharing two); three half-turn axes, chiral"),
    ("MIZHAEV", "Mizhaev Polyhedron", "Genus 3: 8 nonagons, every pair "
     "sharing an edge (8 pairs sharing two); integer vertices, fourfold "
     "rotoreflection"),
    ("MIZHAEV_V1", "Mizhaev Polyhedron V1", "Genus 3: 8 nonagons like the "
     "two above, but 4 pairs of faces never meet (12 pairs share two "
     "edges); Mizhaev's first variant, for contrast"),
    ("MINIMAL_G2", "Ten-Vertex Genus-2 Polyhedron", "Genus 2 on the fewest "
     "possible vertices: 10 vertices, 24 triangles, integer coordinates in "
     "a 4x4x4 cube"),
    ("MINIMAL_G3", "Ten-Vertex Genus-3 Polyhedron", "Genus 3 on the fewest "
     "possible vertices: 10 vertices, 28 triangles, integer coordinates in "
     "a 5x5x5 cube"),
]


def build_neighbourly(kind):
    """(V, F): the solid centred on the origin, fitting a 2 m cube."""
    S = SOLIDS[kind]
    return ([tuple(v) for v in _fit_cube(S["V"])],
            [list(f) for f in S["F"]])


def summary(kind):
    """One line for the operator's report."""
    S = SOLIDS[kind]
    return _nb.summary_text(S["name"], S["V"], S["F"], S["note"])


def map_colouring(F):
    """(colour per face, colours used): faces sharing an edge differ."""
    try:
        from .styles import face_colors
    except ImportError:
        from styles import face_colors
    col, k = face_colors.proper_coloring(len(F), _nb.face_adjacency(F),
                                         kmin=1)
    return [col[f] for f in range(len(F))], k


def _selftest():
    assert {k for k, _l, _d in SOLID_ITEMS} == set(SOLIDS)
    want = {"ROST_VIGH": (24, 36, 8, 3, 8), "MIZHAEV": (24, 36, 8, 3, 8),
            "MIZHAEV_V1": (24, 36, 8, 3, 4),
            "MINIMAL_G2": (10, 36, 24, 2, None),
            "MINIMAL_G3": (10, 42, 28, 3, None)}
    for kind, (nV, nE, nF, g, ncol) in want.items():
        V, F = build_neighbourly(kind)
        E, comps, genus = _nb.surface_topology(F)
        assert (len(V), E, len(F), comps, genus) == (nV, nE, nF, 1, g), kind
        # the 2 m cube convention: centred box, largest extent 2
        for i in range(3):
            lo, hi = min(v[i] for v in V), max(v[i] for v in V)
            assert abs(lo + hi) < 1e-9, (kind, i)
        span = max(max(v[i] for v in V) - min(v[i] for v in V)
                   for i in range(3))
        assert abs(span - 2.0) < 1e-9, (kind, span)
        col, k = map_colouring(F)
        adj = _nb.face_adjacency(F)
        assert all(col[f] != col[h] for f in adj for h in adj[f]), kind
        if ncol is not None:
            assert k == ncol, (kind, k)
        text = summary(kind)
        assert "(genus %d)" % g in text, text
        print(f"{kind:10s} {text}; map colours {k}")
    assert "every two faces share an edge (8 pairs share two)" in \
        summary("ROST_VIGH")
    assert "share an edge" not in summary("MIZHAEV_V1")
    assert "fewest vertices a polyhedron of genus 2" in summary("MINIMAL_G2")
    assert len(_nb.doubled_edges(SOLIDS["MIZHAEV_V1"]["F"])) == 24
    print("RESULT: OK")


try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    try:
        from .styles import net_style as _net_style
        from .styles import plate_style as _plate_style
    except ImportError:
        from styles import net_style as _net_style
        from styles import plate_style as _plate_style

    class MESH_OT_neighbourly_polyhedron_add(bpy.types.Operator,
                                             _net_style.NetStyleProps,
                                             _plate_style.PlateStyleProps):
        """Add a higher-genus successor of the Szilassi or Csaszar
        polyhedron: a genus-3 solid in which every two faces share an edge,
        or a polyhedron of genus 2 or 3 on the fewest possible vertices"""
        bl_idname = "mesh.neighbourly_polyhedron_add"
        bl_label = "Neighbourly Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(name="Solid", items=SOLID_ITEMS,
                            default='ROST_VIGH',
                            description="Which polyhedron to build")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed polyhedron"),
                   _net_style.net_enum_item(),
                   _plate_style.plate_enum_item()],
            default='SOLID',
            description="How the polyhedron is rendered as geometry")
        map_colours: BoolProperty(
            name="Map Colours", default=True,
            description="Colour the faces as a map, so that faces sharing "
                        "an edge differ. Where every two faces share an "
                        "edge each face needs its own colour")
        mark_doubled: BoolProperty(
            name="Mark Doubled Edges", default=False,
            description="On the eight-nonagon solids, draw the edges along "
                        "which a pair of faces meets twice as red rods (16 "
                        "on Mizhaev and Rost-Vigh, 24 on V1)")
        rod_radius: FloatProperty(
            name="Rod Radius", default=0.012, min=0.001, max=0.2,
            description="Radius of the rods marking the doubled edges")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size (1.0 fits a 2 m "
                                         "cube)")

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            lay.prop(self, 'style')
            if self.style == 'NET':
                _net_style.draw_net_props(lay, self)
            if self.style == 'PLATES':
                _plate_style.draw_plate_props(lay, self)
            if self.style == 'SOLID':
                lay.prop(self, 'map_colours')
                if self.solid in _nb.NEIGHBOURLY:
                    lay.prop(self, 'mark_doubled')
                    if self.mark_doubled:
                        lay.prop(self, 'rod_radius')
            lay.prop(self, 'scale')

        def _add_doubled_rods(self, context, obj, V, F):
            """The doubled edges as a bevelled curve parented to the
            solid."""
            cu = bpy.data.curves.new(obj.name + " Doubled Edges", 'CURVE')
            cu.dimensions = '3D'
            cu.bevel_depth = self.rod_radius * self.scale
            cu.bevel_resolution = 3
            cu.use_fill_caps = True
            for a, b in _nb.doubled_edges(F):
                sp = cu.splines.new('POLY')
                sp.points.add(1)
                for pt, v in zip(sp.points, (V[a], V[b])):
                    pt.co = (v[0] * self.scale, v[1] * self.scale,
                             v[2] * self.scale, 1.0)
            try:
                from .styles import face_colors
            except ImportError:
                from styles import face_colors
            cu.materials.append(face_colors.material(
                "Doubled Edge", (0.85, 0.05, 0.10, 1.0)))
            rods = bpy.data.objects.new(cu.name, cu)
            context.collection.objects.link(rods)
            rods.parent = obj
            return rods

        def execute(self, context):
            V, F = build_neighbourly(self.solid)
            name = SOLIDS[self.solid]["name"]
            Vs = [tuple(c * self.scale for c in v) for v in V]
            if self.style == 'NET':
                return _net_style.emit_net_from_operator(
                    self, context, Vs, [list(f) for f in F], name)
            if self.style == 'PLATES':
                return _plate_style.emit_plates_from_operator(
                    self, context, Vs, [list(f) for f in F], name)
            me = bpy.data.meshes.new(name)
            me.from_pydata(Vs, [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            text = summary(self.solid)
            if self.map_colours and len(me.polygons) == len(F):
                try:
                    from .styles import face_colors
                except ImportError:
                    from styles import face_colors
                col, k = map_colouring(F)
                mats, idx = face_colors.materials_for(col, "Map Colour")
                for m in mats:
                    me.materials.append(m)
                me.polygons.foreach_set('material_index', idx)
                text += "; map coloured with %d colours" % k
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.mark_doubled and self.solid in _nb.NEIGHBOURLY:
                self._add_doubled_rods(context, obj, V, F)
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, text)
            return {'FINISHED'}

    def register():
        bpy.utils.register_class(MESH_OT_neighbourly_polyhedron_add)

    def unregister():
        bpy.utils.unregister_class(MESH_OT_neighbourly_polyhedron_add)
