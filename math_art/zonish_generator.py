
# Zonohedrification, zonish polyhedra and rhombohedral dissections
#
# Three constructions that all start from a convex seed and a STAR of
# directions taken from it, after George Hart's zonohedrification work:
#
#   ZONOHEDRON   the zonotope of the star -- Hart's zonohedrification.
#                Taking a seed's vertex directions as zones sends the
#                tetrahedron and the cube alike to the rhombic
#                dodecahedron, the icosahedron to the rhombic
#                triacontahedron, the truncated cuboctahedron to a
#                24-zone solid of 552 parallelograms.
#   ZONISH       the seed PLUS those zones (Hart's zonish polyhedra):
#                a convex seed expanded by adding zones one at a time,
#                which unlike a true zonohedron keeps the seed's
#                odd-sided faces.  Zone Length dials it from the bare
#                seed out to a zonohedron-dominated solid.
#   DISSECTION   the same zonotope cut into its C(n,3) parallelepipeds.
#                For the rhombic triacontahedron that is Kowalewski's
#                twenty golden rhombohedra -- ten acute, ten flat -- and
#                for the rhombic enneacontahedron 120 blocks.  Explode
#                pushes the blocks apart along their own centres.
#
# The mathematics lives in `polyhedra/zonotope.py`, which enumerates the
# faces directly from the star rather than hulling 2^n subset sums, so
# the 24- and 30-zone models are ordinary rather than impossible.
#
# References:
# - George W. Hart, "Zonohedrification", Mathematica Journal 7(3) (1999),
#   201-224; and the Virtual Polyhedra pages on zonohedrification and on
#   zonish polyhedra (georgehart.com/virtual-polyhedra/).
# - Zonohedra: E. S. Fedorov (1885); H. S. M. Coxeter, "Regular
#   Polytopes", 3rd ed., Dover (1973), ch. 2.
# - Dissection of the rhombic triacontahedron into twenty golden
#   rhombohedra: Gerhard Kowalewski, "Der Keplersche Koerper und andere
#   Bauspiele", Koehlers, Leipzig (1938).
# - Zonotope tilings and the C(n,3) cell count: G. M. Ziegler, "Lectures
#   on Polytopes", Springer (1995), ch. 7.

bl_info = {
    "name": "Zonish Polyhedra",
    "author": "Math Art project (after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Zonohedrification, zonish polyhedra and the "
                   "rhombohedral dissections",
    "category": "Add Mesh",
}

import math

try:
    from .polyhedra import zonotope as zt
    from .polyhedra import seeds as _seeds
    from .polyhedra import fit as _fit
    from .styles import shell as _shell
except ImportError:                       # flat-file / headless import
    from polyhedra import zonotope as zt
    from polyhedra import seeds as _seeds
    from polyhedra import fit as _fit
    from styles import shell as _shell


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------
# Hart draws his zonohedrification examples from the Platonic,
# Archimedean and Catalan solids; the first five come straight from the
# shared seed table and the rest through the Conway pipeline in
# `regular_solids_generator`, which already canonicalises them.

SEEDS = [
    ('TETRA', "Tetrahedron", 'PLATONIC'),
    ('CUBE', "Cube", 'PLATONIC'),
    ('OCTA', "Octahedron", 'PLATONIC'),
    ('DODECA', "Dodecahedron", 'PLATONIC'),
    ('ICOSA', "Icosahedron", 'PLATONIC'),
    ('TT', "Truncated Tetrahedron", 'ARCHIMEDEAN'),
    ('CO', "Cuboctahedron", 'ARCHIMEDEAN'),
    ('TC', "Truncated Cube", 'ARCHIMEDEAN'),
    ('TO', "Truncated Octahedron", 'ARCHIMEDEAN'),
    ('RCO', "Rhombicuboctahedron", 'ARCHIMEDEAN'),
    ('TCO', "Truncated Cuboctahedron", 'ARCHIMEDEAN'),
    ('SC', "Snub Cube", 'ARCHIMEDEAN'),
    ('ID', "Icosidodecahedron", 'ARCHIMEDEAN'),
    ('TD', "Truncated Dodecahedron", 'ARCHIMEDEAN'),
    ('TI', "Truncated Icosahedron", 'ARCHIMEDEAN'),
    ('RID', "Rhombicosidodecahedron", 'ARCHIMEDEAN'),
    ('TID', "Truncated Icosidodecahedron", 'ARCHIMEDEAN'),
    ('SD', "Snub Dodecahedron", 'ARCHIMEDEAN'),
    ('RD', "Rhombic Dodecahedron", 'CATALAN'),
    ('RT', "Rhombic Triacontahedron", 'CATALAN'),
    ('KTC', "Triakis Octahedron", 'CATALAN'),
    ('PKD', "Pentakis Dodecahedron", 'CATALAN'),
]

_FAMILY = {sid: fam for sid, _lbl, fam in SEEDS}


def seed_solid(sid):
    """(V, F) for a seed, unit circumradius."""
    fam = _FAMILY.get(sid)
    if fam == 'PLATONIC':
        return _seeds.seed_poly(sid, unit=True)
    try:
        from . import regular_solids_generator as rs
    except ImportError:
        import regular_solids_generator as rs
    V, F, _sizes = rs.build_solid(fam, sid, scale=1.0, canon=True)
    return [list(v) for v in V], [list(f) for f in F]


def seed_star(sid, source='VERTS'):
    """The star a seed contributes: its vertex, face-normal or edge
    directions, with parallel and antiparallel duplicates collapsed."""
    V, F = seed_solid(sid)
    if source == 'FACES':
        return zt.normal_star(V, F)
    if source == 'EDGES':
        return zt.edge_star(V, F)
    return zt.vertex_star(V)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def build_zonohedron(sid, source='VERTS'):
    star = seed_star(sid, source)
    if len(star) < 3:
        raise ValueError("that star has fewer than three distinct zones")
    return zt.zonotope(star)


def build_zonish(sid, source='VERTS', length=0.5, hull=None):
    """Seed + zones.  `hull` is the caller's convex-hull function, so the
    Blender layer can hand in bmesh's and the self-test a slow exact one.
    """
    V, F = seed_solid(sid)
    star = zt.scaled_star(seed_star(sid, source), length)
    if len(star) < 1:
        raise ValueError("that seed contributes no zones")
    pts = zt.minkowski_points(V, star)
    return hull(pts) if hull is not None else pts


def build_dissection(sid, source='VERTS', explode=0.0):
    """The zonotope's C(n,3) blocks, each as (V, F, kind, triple)."""
    star = seed_star(sid, source)
    if len(star) < 3:
        raise ValueError("a dissection needs at least three zones")
    out = []
    for V, F, triple in zt.cells_centered(star):
        kind = zt.acute_or_flat(V, F)
        if explode:
            c = [sum(v[k] for v in V) / len(V) for k in range(3)]
            V = [[v[k] + c[k] * explode for k in range(3)] for v in V]
        out.append((V, F, kind, triple))
    return out


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    MODES = [
        ('ZONOHEDRON', "Zonohedron",
         "Hart's zonohedrification: the zonohedron whose zones are the "
         "seed's own directions"),
        ('ZONISH', "Zonish",
         "The seed itself expanded by those zones, which keeps its "
         "odd-sided faces instead of forcing every face to be centrally "
         "symmetric"),
        ('DISSECTION', "Dissection",
         "The zonohedron cut into its parallelepipeds -- Kowalewski's "
         "twenty golden rhombohedra for the rhombic triacontahedron"),
    ]

    SOURCES = [
        ('VERTS', "Vertices", "One zone per vertex direction"),
        ('FACES', "Face Normals", "One zone per face normal"),
        ('EDGES', "Edges", "One zone per edge direction"),
    ]

    def _seed_items():
        """Static items, with a heading per family.

        Built once at registration rather than from a callback: an
        EnumProperty whose items come from a callback cannot take a
        string default, and an integer default would index into a list
        that contains the headings -- so the default seed would move
        whenever a solid was added.
        """
        out = []
        last = None
        for sid, lbl, fam in SEEDS:
            if fam != last:
                out.append(('', fam.title(), ''))
                last = fam
            out.append((sid, lbl, lbl))
        return out

    SEED_ITEMS = _seed_items()

    def _hull_bmesh(pts):
        """Convex hull with coplanar triangles merged back into the real
        faces -- the zonish sum's facets are mostly not triangles."""
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

    def _mesh_from(name, V, F):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
        me.update()
        return me

    # Five colours, and every block takes three of them: Kowalewski's
    # scheme for the rhombic triacontahedron, where the ten 3-subsets of
    # five colours each appear once as an acute block and once as a flat
    # one.  Applied to any dissection, a block's colour comes from its own
    # zone triple, so blocks generated by the same three zones match.
    # Base colours, not surface colours: the studio lighting the docs and
    # icon renderers use is bright enough to wash a mid-value palette out
    # to pastel at 64 px, so these sit deliberately dark and saturated.
    _PALETTE = [(0.52, 0.05, 0.04, 1.0), (0.03, 0.13, 0.42, 1.0),
                (0.62, 0.36, 0.02, 1.0), (0.04, 0.30, 0.10, 1.0),
                (0.26, 0.06, 0.38, 1.0), (0.58, 0.18, 0.02, 1.0),
                (0.04, 0.34, 0.34, 1.0), (0.42, 0.06, 0.22, 1.0)]

    def _material(name, rgba):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            mat.diffuse_color = rgba
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = rgba
        return mat

    class MESH_OT_zonish_add(bpy.types.Operator):
        """Add a zonohedrification, a zonish polyhedron or a rhombohedral
        dissection"""
        bl_idname = "mesh.zonish_add"
        bl_label = "Zonish Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(name="Mode", items=MODES, default='ZONOHEDRON')
        seed: EnumProperty(
            name="Seed", items=SEED_ITEMS, default='ICOSA',
            description="Convex solid whose directions become the zones")
        source: EnumProperty(
            name="Zones From", items=SOURCES, default='VERTS',
            description="Which of the seed's directions become zones. "
                        "Hart's zonohedrification uses the vertices")
        length: FloatProperty(
            name="Zone Length", default=0.5, min=0.02, max=4.0,
            description="Length of each added zone relative to the seed. "
                        "Small values barely bevel the seed; large ones "
                        "swamp it and the result tends to the zonohedron")
        explode: FloatProperty(
            name="Explode", default=0.0, min=0.0, max=4.0,
            description="Push the dissection's blocks out along their own "
                        "centres, opening the solid up to show how it is "
                        "built")
        color: BoolProperty(
            name="Colours", default=True,
            description="Colour the dissection's blocks by the zone triple "
                        "that generated each one, so blocks of the same "
                        "shape and orientation match")
        separate: BoolProperty(
            name="Separate Blocks", default=False,
            description="One object per block instead of a single mesh")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        __annotations__.update(_shell.style_properties())

        def execute(self, context):
            try:
                if self.mode == 'DISSECTION':
                    return self._dissection(context)
                if self.mode == 'ZONISH':
                    V, F = build_zonish(self.seed, self.source,
                                        self.length, hull=_hull_bmesh)
                    name = "Zonish"
                else:
                    V, F = build_zonohedron(self.seed, self.source)
                    name = "Zonohedron"
            except (ValueError, RuntimeError) as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            V = _fit.fit_cube(V, 2.0 * self.scale)
            obj = _shell.apply(self, context, V, F, name)
            if obj is None:
                self.report({'INFO'}, f"{len(F)} face segments")
            else:
                self.report({'INFO'}, f"V={len(V)} F={len(F)}")
            return {'FINISHED'}

        def _dissection(self, context):
            blocks = build_dissection(self.seed, self.source, self.explode)
            # one common fit for every block, so the assembly stays a solid
            allv = [v for V, _F, _k, _t in blocks for v in V]
            fitted = _fit.fit_cube(allv, 2.0 * self.scale)
            it = iter(fitted)
            blocks = [([next(it) for _ in V], F, k, t)
                      for V, F, k, t in blocks]
            mats = []
            if self.color:
                triples = sorted({t for _V, _F, _k, t in blocks})
                order = {t: i for i, t in enumerate(triples)}
                mats = [_material(f"Zone Block {i}",
                                  _PALETTE[i % len(_PALETTE)])
                        for i in range(len(triples))]
            made = []
            if self.separate:
                for i, (V, F, kind, t) in enumerate(blocks):
                    me = _mesh_from(f"Block {i}", V, F)
                    if self.color:
                        me.materials.append(mats[order[t]])
                    made.append(bpy.data.objects.new(f"Block {i}", me))
            else:
                V, F = [], []
                mat_of = []
                for Vb, Fb, kind, t in blocks:
                    off = len(V)
                    V += Vb
                    F += [[i + off for i in f] for f in Fb]
                    mat_of += [order[t] if self.color else 0] * len(Fb)
                me = _mesh_from("Dissection", V, F)
                for m in mats:
                    me.materials.append(m)
                if self.color and me.materials:
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
                        f"{len(blocks)} blocks ({acute} acute, {flat} flat)")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            lay.prop(self, 'seed')
            lay.prop(self, 'source')
            if self.mode == 'ZONISH':
                lay.prop(self, 'length')
            if self.mode == 'DISSECTION':
                lay.prop(self, 'explode')
                lay.prop(self, 'color')
                lay.prop(self, 'separate')
            else:
                # the dissection is already an assembly of solid blocks;
                # the shell finishes only make sense on a single shell
                _shell.draw_style(self, lay)
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.zonish_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_zonish_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_zonish_add)


def _selftest():
    # Hart's headline zonohedrifications, by vertex star
    want = {
        'TETRA': (14, 12),        # -> rhombic dodecahedron
        'CUBE': (14, 12),         # -> rhombic dodecahedron
        'ICOSA': (32, 30),        # -> rhombic triacontahedron
        'DODECA': (92, 90),       # -> rhombic enneacontahedron
    }
    for sid, (nv, nf) in want.items():
        V, F = build_zonohedron(sid, 'VERTS')
        assert (len(V), len(F)) == (nv, nf), (sid, len(V), len(F))

    # a 24-zone star: the one the 2^n enumerator could never reach
    star = seed_star('TCO', 'VERTS')
    assert len(star) == 24, len(star)
    V, F = build_zonohedron('TCO', 'VERTS')
    assert len(F) == 24 * 23, len(F)
    assert len(V) == 24 * 24 - 24 + 2, len(V)

    # the snub cube's 24 vertices are chiral -- no antipodal pairs to
    # collapse -- so it too gives 24 zones
    assert len(seed_star('SC', 'VERTS')) == 24, len(seed_star('SC', 'VERTS'))

    # dissections: C(n,3) blocks, and the RT's twenty split 10 acute / 10 flat
    blocks = build_dissection('ICOSA', 'VERTS')
    assert len(blocks) == 20, len(blocks)
    assert sum(1 for _V, _F, k, _t in blocks if k == 'ACUTE') == 10
    assert sum(1 for _V, _F, k, _t in blocks if k == 'FLAT') == 10
    assert len(build_dissection('DODECA', 'VERTS')) == 120

    # zonish: the point set is the seed's vertices summed with the
    # zonotope's, and it must reach further than the bare seed
    pts = build_zonish('ID', 'VERTS', 0.4)
    Vs, _Fs = seed_solid('ID')
    assert max(zt._norm(p) for p in pts) > max(zt._norm(v) for v in Vs)

    print("RESULT: OK")
