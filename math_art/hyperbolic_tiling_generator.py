
# Hyperbolic Tiling Generator for Blender
#
# The hyperbolic analogue of the Euclidean uniform-tiling generator:
# it goes beyond the regular {p,q} tilings to the whole Wythoff
# family -- truncated, rectified, cantellated, omnitruncated and the
# chiral snub -- built on a general (p, q, r) triangle reflection group,
# and renders each tiling either as flat colored polygons in the unit
# disk or wrapped onto a curved 3D surface model.
#
# Machinery.  Everything lives in the hyperboloid model of H^2 inside
# Minkowski R^{2,1}, <x,y> = x1 y1 + x2 y2 - x3 y3.  The (p, q, r)
# triangle group is three mirrors with unit spacelike normals whose
# pairwise Gram products are the corner angles:
#
#     <n0, n1> = -cos(pi/p)    corner A (mirrors 0,1), order p
#     <n1, n2> = -cos(pi/q)    corner B (mirrors 1,2), order q
#     <n0, n2> = -cos(pi/r)    corner C (mirrors 0,2), order r
#
# hyperbolic exactly when 1/p + 1/q + 1/r < 1 (r defaults to 2, the
# right triangle that gives the classic regular families).  Reflection
# in a mirror is x -> x - 2<x,n>n; the group is enumerated by BFS over
# reflection words to a tile cap.
#
# Wythoff construction.  A uniform tiling is the orbit of a single
# SEED point in the fundamental triangle.  Which of the three mirrors
# are "active" (ringed Coxeter nodes) fixes the seed: it lies on every
# inactive mirror and is equidistant from the active ones (equal-length
# edges -> a uniform, vertex-transitive tiling).  With the linear Gram
# picture this seed is the null direction of two linear constraints,
# renormalized to the hyperboloid:
#
#     REGULAR   {p,q}    rings {0}      seed at corner B
#     TRUNCATED t{p,q}   rings {0,1}    seed on edge BC
#     RECTIFIED r{p,q}   rings {1}      seed at corner C
#     BITRUNC.  t{q,p}   rings {1,2}    seed on edge AB
#     CANTEL.   rr{p,q}  rings {0,2}    seed on edge CA
#     OMNITRUNC tr{p,q}  rings {0,1,2}  seed at the incentre
#
# Faces.  Each corner of order n whose two mirrors are k-of-them active
# contributes a face centred there: an n-gon when k = 1, a 2n-gon when
# k = 2, and no face (that corner is a vertex) when k = 0.  A face is
# the orbit of the seed under the finite dihedral stabilizer of its
# corner -- always the COMPLETE local ring, so no partial faces are
# ever emitted -- pushed out to every image of that corner by the group
# and de-duplicated by centre.  A degenerate 2-gon (an order-2 corner
# with one active mirror, i.e. r = 2) is just an edge and is dropped.
# SNUB is the odd one out (the alternation of the omnitruncation: the
# rotation subgroup only, chiral); see snub_faces.
#
# Every face is kept as its ordered ring of hyperboloid corner points,
# so it can be projected into any output model.  The two flat disk
# pictures go through the shared 2D tiling pipeline (color / grout
# margin / relief / backing); the two curved surfaces are built as 3D
# cells directly:
#   KLEIN     (x,y,t) -> (x/t, y/t).  Geodesics are straight chords, so
#             each face is a straight-edged polygon of its corners.
#   POINCARE  (x,y,t) -> (x/(1+t), y/(1+t)).  Geodesic edges are arcs,
#             so every edge is subdivided by sampling the hyperboloid
#             geodesic between its endpoints before projecting.
#   HEMISPHERE  Klein disk K = (x/t, y/t) lifted onto the upper unit
#             hemisphere z = sqrt(1 - |K|^2); edges subdivided.
#   PSEUDOSPHERE  the conformal disk carried Poincare -> Cayley -> upper
#             half plane -> tractricoid (the isometry y = cosh u, x = v),
#             clipped to one seam period 0 <= x <= 2pi and 1 <= y <= cap;
#             edges subdivided.
#
# Color is by side count, tile type, uniform, or PARITY.  Parity is
# the classic alternating two-tone: the faces are 2-colored so the
# color flips across every shared edge (a proper 2-coloring of the
# face-adjacency graph, seeded by reflection-word parity).  It
# alternates cleanly only when the tiling is BIPARTITE (q even --
# {5,4}, {6,4}, ...); for odd q ({3,7}, {8,3}, ...) the faces are not
# 2-colorable and the mode shows the unavoidable frustration seam
# (the snub, full of triangles, is likewise not 2-colorable).
#
# References:
# - W. A. Wythoff -- the Wythoff (kaleidoscopic) construction of uniform
#   tilings from a triangle reflection group.
# - H. S. M. Coxeter, "Regular Polytopes" (1948) and his work on
#   reflection groups and the {p,q} Schlafli symbol.
# - Henri Poincare -- the conformal disk model of the hyperbolic plane.
# - Henry Segerman, "Visualizing Mathematics with 3D Printing" (2016) --
#   hyperbolic surface models.

bl_info = {
    "name": "Hyperbolic Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Tiling",
    "description": "Uniform (Wythoff) hyperbolic tilings on (p,q,r) "
                   "triangle groups, as flat disk polygons or curved "
                   "3D surface models",
    "category": "Add Mesh",
}



try:
    from .patterns import common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import tiling_generator as tg
# The mathematics lives in the sibling `hyperbolic` engine package;
# this module is the Blender layer over it.
try:
    from .hyperbolic.tilings import (FORM_ITEMS, _kind_for, _slab_cell,
                                         build_surface_cells, build_uniform)
except ImportError:  # flat import outside the package
    from hyperbolic.tilings import (FORM_ITEMS, _kind_for, _slab_cell,
                                        build_surface_cells, build_uniform)








# --------------------------------------------------------------------
# Minkowski helpers
# --------------------------------------------------------------------



















# --------------------------------------------------------------------
# (p, q, r) triangle group
# --------------------------------------------------------------------









# --------------------------------------------------------------------
# Wythoff seed
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# Face assembly
# --------------------------------------------------------------------



































# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------



try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


MODEL_ITEMS = [
    ('POINCARE', "Poincare Disk",
     "Conformal disk: geodesic edges curve (flat 2D)"),
    ('KLEIN', "Klein Disk",
     "Projective disk: geodesics are straight chords (flat 2D)"),
    ('HEMISPHERE', "Hemisphere",
     "Klein disk lifted onto the upper unit hemisphere (3D)"),
    ('PSEUDOSPHERE', "Pseudosphere",
     "Wrapped isometrically onto the tractricoid via the upper "
     "half plane (3D)"),
]

_FLAT = {'POINCARE', 'KLEIN'}


if _IN_BLENDER:

    class MESH_OT_hyperbolic_tiling_add(bpy.types.Operator,
                                        AddObjectHelper):
        """Add a hyperbolic tiling: any Wythoff form on a (p,q,r)
        triangle group, as a flat disk or a curved 3D surface"""
        bl_idname = "mesh.hyperbolic_tiling_add"
        bl_label = "Hyperbolic Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        form: EnumProperty(name="Form", items=FORM_ITEMS,
                           default='REGULAR',
                           description="Which Wythoff uniform form to "
                                       "build")
        p: IntProperty(
            name="p", default=7, min=3, max=12,
            description="Order of the first corner (p-gon symmetry)")
        q: IntProperty(
            name="q", default=3, min=3, max=12,
            description="Order of the second corner (q around a vertex)")
        r: IntProperty(
            name="r", default=2, min=2, max=12,
            description="Order of the third corner (2 = right triangle, "
                        "the classic regular families)")
        model: EnumProperty(name="Model", items=MODEL_ITEMS,
                            default='POINCARE',
                            description="Disk or curved 3D model the "
                                        "tiling is drawn in")
        depth: IntProperty(
            name="Depth", default=14, min=1, max=60,
            description="Reflection word length explored")
        max_tiles: IntProperty(
            name="Max Group Elements", default=8000, min=10,
            max=40000,
            description="Cap on triangle-group elements generated")
        color_by: EnumProperty(
            name="Color By",
            items=[('SIDES', "By Sides",
                    "Material per polygon side count (exact in Klein)"),
                   ('TYPE', "By Tile Type",
                    "Material per face side count / orbit"),
                   ('PARITY', "Parity (Alternating)",
                    "Classic two-tone by reflection-word parity; "
                    "alternates cleanly only for even q (bipartite), "
                    "shows a frustration seam for odd q"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='SIDES',
            description="How the tiles are colored")
        hide_off_parity: BoolProperty(
            name="Hide Off-Parity Tiles", default=False,
            description="In Parity color mode, omit the 'off' (second) "
                        "parity class so only the 'on' tiles render, "
                        "leaving a sparse alternating pattern")
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles (flat models)")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat tiling; > 0 extrudes each tile into a "
                        "raised plaque (flat models)")
        backing: BoolProperty(
            name="Backing Disk", default=False,
            description="Add a backing slab under the tiles, for a solid "
                        "disk plaque (flat models)")
        base: FloatProperty(
            name="Base Thickness", default=0.05, min=0.01, max=1.0,
            description="Backing slab thickness (flat models)")
        thickness: FloatProperty(
            name="Shell Thickness", default=0.0, min=0.0, max=0.5,
            description="0 = a single zero-thickness surface; > 0 makes "
                        "each tile a solid shell (curved models)")
        y_cap: FloatProperty(
            name="Cusp Cap", default=6.0, min=1.5, max=50.0,
            description="Clip the pseudosphere cusp at upper-half-plane "
                        "height y (the horn gets thin)")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty)")

        def execute(self, context):
            if (1.0 / self.p + 1.0 / self.q + 1.0 / self.r
                    >= 1.0 - 1e-9):
                self.report({'ERROR'},
                            "(%d,%d,%d) is not hyperbolic: need "
                            "1/p + 1/q + 1/r < 1" %
                            (self.p, self.q, self.r))
                return {'CANCELLED'}
            label = dict((k, v) for k, v, _ in FORM_ITEMS)[self.form]
            name = "Hyperbolic %s (%d,%d,%d)" % (
                label, self.p, self.q, self.r)
            try:
                if self.model in _FLAT:
                    polys, sides, parities = build_uniform(
                        self.p, self.q, self.r, self.form, self.model,
                        self.depth, self.max_tiles)
                    if not polys:
                        self.report({'ERROR'}, "no tiles generated")
                        return {'CANCELLED'}
                    if self.color_by == 'PARITY' and self.hide_off_parity:
                        keep = [i for i, pa in enumerate(parities)
                                if pa == 0]
                        polys = [polys[i] for i in keep]
                        sides = [sides[i] for i in keep]
                        parities = [parities[i] for i in keep]
                        if not polys:
                            self.report({'ERROR'},
                                        "no tiles left after hiding "
                                        "off-parity")
                            return {'CANCELLED'}
                    # Precompute the material index per tile from the
                    # TRUE side count and drive cells_from_polys via
                    # TYPE, so colors are correct even in Poincare
                    # (where edge subdivision inflates len(poly)).
                    mats = [_kind_for(s, pa, self.color_by)
                            for s, pa in zip(sides, parities)]
                    cells = tg.cells_from_polys(
                        lambda a, b: (polys, mats), 1, 1, 'TYPE',
                        self.margin, self.height, trim=False)
                    if self.backing and not self.separate:
                        cells.append(_slab_cell(1.0, self.base, 1))
                else:
                    cells = build_surface_cells(
                        self.p, self.q, self.r, self.form, self.model,
                        self.depth, self.max_tiles, self.color_by,
                        self.thickness, self.y_cap,
                        hide_off=(self.color_by == 'PARITY'
                                  and self.hide_off_parity))
            except ValueError as err:
                self.report({'ERROR'}, str(err))
                return {'CANCELLED'}
            if not cells:
                self.report({'ERROR'}, "no tiles survived clipping")
                return {'CANCELLED'}
            obj = pc.emit(context, name, cells, self.separate,
                          fit=True, operator=self)
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
            for prop in ('form', 'p', 'q', 'r', 'model', 'depth',
                         'max_tiles', 'color_by'):
                lay.prop(self, prop)
            if self.color_by == 'PARITY':
                lay.prop(self, 'hide_off_parity')
            if self.model in _FLAT:
                lay.prop(self, 'margin')
                lay.prop(self, 'height')
                lay.prop(self, 'backing')
                if self.backing:
                    lay.prop(self, 'base')
            else:
                lay.prop(self, 'thickness')
                if self.model == 'PSEUDOSPHERE':
                    lay.prop(self, 'y_cap')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_tiling_add",
                             icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_tiling_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hyperbolic_tiling_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _coverage_disk(polys, radius=0.72, samples=41, off=(0.0137, 0.0071)):
    """Point-coverage over an interior sub-disk (Klein).  Sample a grid,
    keep points inside `radius`, and count how many tiles cover each.
    Returns (tested, gaps, overlaps): points covered by 0, and by >= 2,
    tiles.  A per-tile bounding box prefilters the point-in-polygon
    tests so the sweep stays fast on dense patches."""
    boxes = [(p, p[:, 0].min(), p[:, 0].max(),
              p[:, 1].min(), p[:, 1].max()) for p in polys]
    r2 = radius * radius
    tested = gaps = overs = 0
    for ix in range(samples):
        x = -radius + 2.0 * radius * ix / (samples - 1) + off[0]
        for iy in range(samples):
            y = -radius + 2.0 * radius * iy / (samples - 1) + off[1]
            if x * x + y * y > r2:
                continue
            cnt = 0
            for p, x0, x1, y0, y1 in boxes:
                if x < x0 or x > x1 or y < y0 or y > y1:
                    continue
                if tg._pip(p, x, y):
                    cnt += 1
                    if cnt >= 2:
                        break
            tested += 1
            if cnt == 0:
                gaps += 1
            elif cnt >= 2:
                overs += 1
    return tested, gaps, overs


def _parity_defects(polys, parities, radius=0.9):
    """Count edge-adjacent face pairs (in an interior window) whose
    parity colors clash.  0 for a bipartite (even-q) tiling; positive
    where odd q forces a frustration seam.  Adjacency = two faces that
    share two nearly-coincident boundary vertices (an edge)."""
    import itertools
    cents = [p.mean(axis=0) for p in polys]
    keep = [i for i, c in enumerate(cents)
            if c[0] * c[0] + c[1] * c[1] < radius * radius]
    vsets = {i: {(round(x, 3), round(y, 3)) for x, y in polys[i]}
             for i in keep}
    clash = adj = 0
    for i, j in itertools.combinations(keep, 2):
        if len(vsets[i] & vsets[j]) >= 2:            # share an edge
            adj += 1
            if parities[i] == parities[j]:
                clash += 1
    return adj, clash


def _selftest():
    CASES = [(7, 3, 2), (5, 4, 2), (8, 3, 2), (4, 3, 3)]
    FORMS = ['REGULAR', 'TRUNCATED', 'RECTIFIED', 'BITRUNCATED',
             'CANTELLATED', 'OMNITRUNCATED', 'SNUB']
    DEPTH, MAXT = 48, 1200
    all_ok = True
    passed_forms = set()
    for form in FORMS:
        form_ok = True
        for (p, q, r) in CASES:
            polys, sides, parities = build_uniform(p, q, r, form,
                                                   'KLEIN', DEPTH, MAXT)
            tested, gaps, overs = _coverage_disk(polys)
            ok = (tested > 0 and gaps == 0 and overs == 0
                  and len(polys) > 0)
            form_ok = form_ok and ok
            print("%-14s {%d,%d,%d}  tiles=%5d sides=%-14s "
                  "sampled=%4d gaps=%3d overlaps=%3d  %s"
                  % (form, p, q, r, len(polys), str(sorted(set(sides))),
                     tested, gaps, overs, "OK" if ok else "BAD"))
        if form_ok:
            passed_forms.add(form)
        all_ok = all_ok and form_ok
    print("forms verified:",
          ", ".join(f for f in FORMS if f in passed_forms))

    # every output model builds a non-empty valid mesh for a couple forms
    print("--- models ---")
    models_ok = True
    for model in ('POINCARE', 'KLEIN', 'HEMISPHERE', 'PSEUDOSPHERE'):
        for form in ('REGULAR', 'TRUNCATED'):
            if model in ('POINCARE', 'KLEIN'):
                polys, _s, _pa = build_uniform(7, 3, 2, form, model,
                                               16, 1500)
                nf = len(polys)
                nv = sum(len(p) for p in polys)
            else:
                cells = build_surface_cells(7, 3, 2, form, model, 16,
                                            1500)
                nf = sum(len(cf) for _cv, cf, _cm in cells)
                nv = sum(len(cv) for cv, _cf, _cm in cells)
            good = nf > 0 and nv > 0
            models_ok = models_ok and good
            print("%-13s %-10s  cells/polys ok  V~%-6d F~%-6d  %s"
                  % (model, form, nv, nf, "OK" if good else "BAD"))
    all_ok = all_ok and models_ok

    # PARITY 2-coloring: clean checkerboard for even q, seam for odd q
    print("--- parity ---")
    parity_ok = True
    for (p, q, r), expect in [((5, 4, 2), 'clean'), ((6, 4, 2), 'clean'),
                              ((7, 3, 2), 'seam'), ((8, 3, 2), 'seam')]:
        polys, sides, parities = build_uniform(p, q, r, 'REGULAR',
                                               'KLEIN', DEPTH, MAXT)
        adj, clash = _parity_defects(polys, parities)
        want_clean = (expect == 'clean')
        good = adj > 0 and ((clash == 0) == want_clean)
        parity_ok = parity_ok and good
        print("REGULAR {%d,%d,%d} q%s  edges=%3d clashes=%3d expect=%-5s  %s"
              % (p, q, r, "even" if q % 2 == 0 else "odd ", adj, clash,
                 expect, "OK" if good else "BAD"))
    all_ok = all_ok and parity_ok

    # By-Sides coloring must survive edge subdivision: a mixed form
    # (14-gons + triangles) must yield >1 distinct material index in
    # BOTH flat models, not collapse to one.
    print("--- color ---")
    color_ok = True
    for model in ('KLEIN', 'POINCARE'):
        polys, sides, parities = build_uniform(7, 3, 2, 'TRUNCATED',
                                               model, 16, 1500)
        mats = [_kind_for(s, pa, 'SIDES') for s, pa in zip(sides,
                                                           parities)]
        ndistinct = len(set(mats))
        good = ndistinct > 1
        color_ok = color_ok and good
        print("TRUNCATED {7,3,2} %-9s SIDES  distinct materials=%d  %s"
              % (model, ndistinct, "OK" if good else "BAD"))
    all_ok = all_ok and color_ok

    print("RESULT:", "OK" if all_ok else "BAD")
    assert all_ok
