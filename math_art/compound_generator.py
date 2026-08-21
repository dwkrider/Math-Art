
# Polyhedron Compounds for Blender
#
# Classic uniform polyhedron compounds built as the orbit of a seed solid
# under a rotation group: the stella octangula (two tetrahedra), the
# icosahedral compounds of five and ten tetrahedra, five cubes and five
# octahedra, and the dual pairs (cube + octahedron, dodecahedron +
# icosahedron).  The rotation groups are generated in the standard cube /
# dodecahedron frame so an axis-aligned seed lands on the compound's
# shared vertices; each component keeps its own colour.
#
# The axis-alignment rule (polyhedra/compounds.py) generalises all of
# this: align the component's n-fold axis with the compound's m-fold
# axis, turn it, replicate over the group, drop duplicates. Michael
# Harman's 1974 construction, and the Turn dial is the rotational
# freedom several of Hart's families have -- the five cubes separate
# into a generic thirty as soon as it leaves the named angle.
#
# References:
# - Stella octangula: Johannes Kepler, "Harmonices Mundi" (1619).
# - The axis-alignment construction: Michael G. Harman, unpublished
#   (1974), as described by George W. Hart, "Compounds - Harman's",
#   Virtual Polyhedra (georgehart.com/virtual-polyhedra/).
# - The regular compounds (5 tetrahedra, 5 cubes, ...): Edmund Hess
#   (1876); Max Bruckner (1900); catalogued in H. S. M. Coxeter,
#   "Regular Polytopes" (1948).
# - Full enumeration of uniform compounds: John Skilling, "Uniform
#   compounds of uniform polyhedra", Math. Proc. Camb. Phil. Soc. 79
#   (1976).

bl_info = {
    "name": "Polyhedron Compounds",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Compounds of regular polyhedra (stella octangula, "
                   "5/10 tetrahedra, 5 cubes, dual pairs)",
    "category": "Add Mesh",
}


try:
    import numpy as np
except ImportError:
    np = None
# The mathematics lives in the sibling `polyhedra` engine package;
# this module is the Blender layer over it.
try:
    from .polyhedra import compounds as _cmp
    from .polyhedra.compounds import (AXIS_COMPOUNDS, COMPOUNDS,
                                      build_compound)
except ImportError:  # flat import outside the package
    from polyhedra import compounds as _cmp
    from polyhedra.compounds import (AXIS_COMPOUNDS, COMPOUNDS,
                                     build_compound)





# ---- seeds (V, F), axis-aligned in the standard frame --------------------









# ---- rotation groups in the standard frame -------------------------------













# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _grouped_enum_items():
        """COMPOUNDS as enum items, grouped under family headings."""
        items = []
        for heading, rows in _cmp.compound_groups():
            if items:
                items.append(None)                 # separator
            items.append(("", heading, ""))        # heading
            items.extend((k, lbl, "") for k, lbl in rows)
        return items

    _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                (0.30, 0.69, 0.42), (0.95, 0.77, 0.29),
                (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                (0.80, 0.45, 0.30), (0.45, 0.55, 0.80)]

    def _mat(i):
        name = f"Compound {i}"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            rgb = _PALETTE[i % len(_PALETTE)]
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        return mat

    class MESH_OT_polyhedron_compound_add(bpy.types.Operator):
        """Add a compound of regular polyhedra (each component coloured
        separately)"""
        bl_idname = "mesh.polyhedron_compound_add"
        bl_label = "Polyhedron Compound"
        bl_options = {'REGISTER', 'UNDO'}

        # Grouped, not flat.  117 entries in the order the source tables
        # happened to be concatenated ran Skilling 62, 67, 18, 48 down
        # consecutive columns; an empty identifier gives a heading and a
        # None a separator, so the families read as families.  The
        # default has to be named explicitly now, since the first item is
        # a heading rather than a compound.
        compound: EnumProperty(
            name="Compound", default='STELLA',
            items=_grouped_enum_items())
        separate: BoolProperty(
            name="Separate Objects", default=False,
            description="One object per component instead of a single "
                        "coloured mesh")
        phase: FloatProperty(
            name="Turn", default=0.0, min=-180.0, max=180.0,
            description="Turn each component about the axis it was "
                        "aligned on. The named compounds sit at one "
                        "angle; away from it the components separate "
                        "and the count usually rises, which is the "
                        "rotational freedom several of these families "
                        "have")
        sides: IntProperty(
            name="Sides", default=5, min=3, max=24,
            description="Side count of the prism or antiprism, for the "
                        "two compounds built from one. Hart draws prisms "
                        "for 3 to 10 sides and antiprisms for 4 to 10")
        repeat: IntProperty(
            name="Repeat", default=2, min=1, max=12,
            description="Repeat count r for Skilling's four infinite "
                        "prism and antiprism families, which are "
                        "parametric in both the side count and r")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            try:
                axis_kind = any(self.compound == r[0] for r in AXIS_COMPOUNDS)
                comps = build_compound(
                    self.compound, sides=self.sides,
                    repeat=self.repeat,
                    phase=self.phase if axis_kind and self.phase else None)
            except Exception as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # normalise the whole compound to fit a 2 m cube
            mx = max(abs(c) for V, _F in comps for v in V for c in v) or 1.0
            s = self.scale / mx
            label = dict(COMPOUNDS)[self.compound]
            for o in context.selected_objects:
                o.select_set(False)
            first = None
            if self.separate:
                for i, (V, F) in enumerate(comps):
                    me = bpy.data.meshes.new(f"{label} {i + 1}")
                    me.from_pydata([tuple(c * s for c in v) for v in V],
                                   [], [tuple(f) for f in F])
                    me.materials.append(_mat(i))
                    me.update()
                    obj = bpy.data.objects.new(f"{label} {i + 1}", me)
                    context.collection.objects.link(obj)
                    obj.location = context.scene.cursor.location
                    obj.select_set(True)
                    first = first or obj
            else:
                verts = []
                faces = []
                fmat = []
                for i, (V, F) in enumerate(comps):
                    base = len(verts)
                    verts += [tuple(c * s for c in v) for v in V]
                    for f in F:
                        faces.append([base + j for j in f])
                        fmat.append(i)
                me = bpy.data.meshes.new(label)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                for i in range(len(comps)):
                    me.materials.append(_mat(i))
                if len(me.polygons) == len(fmat):
                    me.polygons.foreach_set('material_index', fmat)
                me.update()
                first = bpy.data.objects.new(label, me)
                context.collection.objects.link(first)
                first.location = context.scene.cursor.location
                first.select_set(True)
            context.view_layer.objects.active = first
            self.report({'INFO'}, f"{label}: {len(comps)} components")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.polyhedron_compound_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polyhedron_compound_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polyhedron_compound_add)


def _selftest():
    import math as _math
    for k, lbl in COMPOUNDS:
        comps = build_compound(k)
        nv = len(set(tuple(round(c, 4) for c in v)
                     for V, _F in comps for v in V))
        print(f"{k:14s} {len(comps):2d} components, {nv:3d} distinct verts"
              f"  ({lbl})")

    # The axis-rule compounds assert their component counts: the rule
    # itself is sound, so what can go wrong is the five-tuple describing
    # a named model, and a wrong axis pair changes the count.
    for k, lbl, _c, _g, _ca, _ga, _ph, want in AXIS_COMPOUNDS:
        comps = build_compound(k)
        assert len(comps) == want, (k, len(comps), want)

        # every component must be congruent to the first -- same radius
        # spectrum and same pairwise-distance multiset, which pins the
        # vertex set to an isometry without solving for the rotation
        def spectrum(V):
            rad = sorted(round(_math.dist((0, 0, 0), v), 6) for v in V)
            pair = sorted(round(_math.dist(a, b), 6)
                          for i, a in enumerate(V) for b in V[i + 1:])
            return rad, pair

        ref = spectrum(comps[0][0])
        for V, _F in comps[1:]:
            got = spectrum(V)
            assert len(got[0]) == len(ref[0]), k
            assert max(abs(a - b) for a, b in zip(got[0], ref[0])) < 1e-6, k
            assert max(abs(a - b) for a, b in zip(got[1], ref[1])) < 1e-6, k

    # turning off the named angle is what the freedom families do: the
    # five cubes separate into a generic thirty
    assert len(build_compound('H_5CUBES', phase=17.0)) > 5

    # --- every declared axis must really BE one -------------------------
    # COMPONENT_AXES and GROUP_AXES are asserted geometry, and a wrong
    # entry is close to invisible: the dodecahedron's five-fold axis was
    # recorded as (0, 1, PHI) -- the ICOSAHEDRON's -- and every component
    # count still came out right, because a bogus axis and an honest
    # five-fold-on-three-fold alignment both leave the stabilizer
    # trivial.  Only turning the solid about the axis and checking it
    # lands on itself finds it.
    def _fixes(V, axis, order):
        R = _cmp._rot(axis, 2 * _math.pi / order)
        S = {tuple(np.round(v, 4)) for v in V}
        return {tuple(np.round(R @ np.array(v, float), 4)) for v in V} == S

    for comp, axes in _cmp.COMPONENT_AXES.items():
        V, _F = _cmp._component(comp)
        for order, axis in axes.items():
            assert _fixes(V, axis, order), \
                (comp, order, axis, 'not a symmetry axis of the component')
    for grp, axes in _cmp.GROUP_AXES.items():
        G = _cmp.GROUPS[grp]()
        for order, axis in axes.items():
            R = _cmp._rot(axis, 2 * _math.pi / order)
            assert any(np.allclose(R, M, atol=1e-9) for M in G), \
                (grp, order, axis, 'not a rotation axis of the group')
    print("AXES  every declared component and group axis verified against "
          "the solid it belongs to")

    # --- Skilling 20-25, the four infinite families ---------------------
    # Not models but families, parametric in the side count n AND the
    # repeat count r: entry 20 gives 2r constituents and 21 gives r, for
    # every n and r.  Assert that across a grid rather than at one point,
    # because a construction that happened to be right at r = 2 and wrong
    # elsewhere would be the easy mistake here.
    for k, lbl in _cmp.FAMILY_LABELS:
        _anti, reflect, star_m = _cmp.FAMILIES[k]
        # The m-even star antiprism (entries 24/25) exists only where
        # Coxeter's crossed trapezoid does, 2 < p < 3, i.e. n < 3m.  With
        # m = 2 that leaves n = 5 alone, so {5/2} is the only base these
        # two families have -- they are "infinite" in the repeat count r,
        # not in n.
        ns = (5,) if star_m else (3, 5, 6, 8)
        for n in ns:
            for r in (1, 2, 3, 4):
                got = len(build_compound(k, sides=n, repeat=r))
                want = 2 * r if reflect else r
                assert got == want, (k, n, r, got, want)
    print("FAMILIES  Skilling 20-23 over n=3,5,6,8 and r=1..4: "
          "2r and r constituents as tabulated")

    # --- Skilling 46-67, tetrahedral symmetry embedded ------------------
    # The constituent is placed in its own standard frame and orbited;
    # the count is whatever the two groups share.  What has to be right
    # is the FRAME, and getting it wrong is silent -- an unaligned
    # constituent still builds, still looks like a compound, and returns
    # 15, 24, 30 or 60 copies where Skilling says 5, 2, 5 or 10.  So
    # assert his counts, and separately assert that the aligner really
    # put an orthogonal pair of rotation axes on the coordinate axes.
    for k, lbl, comp, grp, want in _cmp.SUBGROUP_COMPOUNDS:
        comps = build_compound(k)
        assert len(comps) == want, (k, len(comps), want)
        V, F = _cmp._component(comp)
        for axis in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
            R = _cmp._rot(axis, _math.pi)
            S = {tuple(np.round(v, 4)) for v in V}
            assert {tuple(np.round(R @ np.array(v, float), 4))
                    for v in V} == S, (k, comp, axis, 'frame not aligned')
    # --- Skilling's special positions, 13-16 and 19 ---------------------
    # Counts alone cannot separate these: 13 and 14 are both 20 copies,
    # 15 and 16 both 10.  What separates them is his CONSTITUENTS PER
    # VERTEX column, so assert that instead -- it is the quantity the
    # special angles exist to produce.
    per_vertex = {'S13_OCTA': 1, 'S14_OCTA': 2, 'S15_OCTA': 1,
                  'S16_OCTA': 1, 'S19_TETRAHEMIHEX': 2}
    for k, want in per_vertex.items():
        cnt = {}
        for V, _F in build_compound(k):
            for v in V:
                key = tuple(round(c, 5) for c in v)
                cnt[key] = cnt.get(key, 0) + 1
        assert set(cnt.values()) == {want}, \
            (k, 'constituents per vertex', sorted(set(cnt.values())), want)

    # Entry 15's angle has two independent derivations -- the half-turn
    # axis rule it has always used, and the mirror-plane rule Skilling's
    # own footnote describes.  They must agree; two routes to one angle
    # is worth more than either alone.
    mirror = _cmp.vertex_mirror_phases('OCTA', 'Ih', 3, 3)
    older = _cmp.perp_phase_from('OCTA', 'Ih', 3, 3,
                                 _cmp.perp_half_turn_axis('OCTA', 3))
    assert abs(mirror[1] - older) < 1e-6, (mirror[1], older)

    # 15 and 16 are Skilling's figure 5(a) and 5(b): two DIFFERENT
    # compounds, not one solid turned.  Not sharing a single vertex is
    # what stands in for reading that figure.
    def vset(k):
        pts = [v for V, _F in build_compound(k) for v in V]
        R = max(_math.dist((0, 0, 0), v) for v in pts)
        return {tuple(round(c / R, 5) for c in v) for v in pts}

    assert not (vset('S15_OCTA') & vset('S16_OCTA')), \
        'Skilling 15 and 16 should share no vertex'

    # Skilling records that entry 19's 60 vertices and 240 edges are
    # shared with the compound of 20 octahedra, entry 14.  That is a
    # cross-check from outside either construction: different
    # constituents, different groups (I against I_h) and different angle
    # rules, landing on one skeleton.
    def eset(k):
        comps = build_compound(k)
        pts = [v for V, _F in comps for v in V]
        R = max(_math.dist((0, 0, 0), v) for v in pts)
        out = set()
        for V, F in comps:
            for f in F:
                for i in range(len(f)):
                    a = tuple(round(c / R, 5) for c in V[f[i]])
                    b = tuple(round(c / R, 5)
                              for c in V[f[(i + 1) % len(f)]])
                    out.add((min(a, b), max(a, b)))
        return out

    assert vset('S14_OCTA') == vset('S19_TETRAHEMIHEX'), \
        'Skilling 19 should share entry 14 vertices'
    assert eset('S14_OCTA') == eset('S19_TETRAHEMIHEX'), \
        'Skilling 19 should share entry 14 edges'
    assert len(vset('S19_TETRAHEMIHEX')) == 60, 'entry 19 wants 60 vertices'
    assert len(eset('S19_TETRAHEMIHEX')) == 240, 'entry 19 wants 240 edges'

    # Entry 19's group is I, not I_h, so Skilling says it "exists in two
    # enantiomorphous forms".  Its angle rule returns exactly two roots
    # giving 20 copies, and they are mirror images of one another.
    pair = _cmp.vertex_pairing_phases('U4', 'I', 3, 3)[-2:]
    faces = []
    for ph in pair:
        faces.append({frozenset(tuple(round(c, 5) for c in V[i])
                                for i in f)
                      for V, F in _cmp.build_axis_compound('U4', 'I', 3, 3,
                                                           ph)
                      for f in F})
    assert faces[0] != faces[1], 'entry 19 roots should be distinct'
    flip = {frozenset((-x, y, z) for (x, y, z) in f) for f in faces[0]}
    assert flip == faces[1], 'entry 19 roots should be enantiomorphs'
    print("SPECIAL   Skilling 13-16 and 19: per-vertex counts as "
          "tabulated, 19 shares 14's 60 vertices and 240 edges")

    print("SUBGROUP  Skilling 46-67 sample: constituents aligned and "
          "orbited, all counts as tabulated")

    # Skilling's REMARKS column is an independent check, and a sharp one:
    # he brackets 62, 63, 64 as "shared vertices and edge length", and
    # likewise 65, 66, 67.  Constituent counts cannot catch a wrong
    # constituent here -- all six rows are 5 copies -- but a shared
    # vertex set can, since it holds only if the right solid is in the
    # right frame at the right scale.  This is what confirms that his
    # `2 4 3/2 4/2 |` and `2 4/3 3/2 4/2 |` are the small and great
    # rhombihexahedra (U18, U21), written with the repeated generator
    # that records Wythoff's construction traversing each face twice.
    for band in (('S62_RHOMBICUBOCTA', 'S63_SM_RHOMBIHEX',
                  'S64_SM_CUBICUBOCTA'),
                 ('S65_GT_CUBICUBOCTA', 'S66_GT_RHOMBIHEX',
                  'S67_NONCONVEX_GRCO')):
        ref_v = ref_e = None
        for k in band:
            pts = [v for vv, _ff in build_compound(k) for v in vv]
            R = max(_math.sqrt(sum(x * x for x in q)) for q in pts)
            vs = sorted(tuple(round(x / R, 5) for x in q) for q in pts)
            es = set()
            for vv, ff in build_compound(k):
                for f in ff:
                    for i in range(len(f)):
                        a, b = vv[f[i]], vv[f[(i + 1) % len(f)]]
                        es.add(round(_math.dist(a, b) / R, 5))
            if ref_v is None:
                ref_v, ref_e = vs, sorted(es)
            assert vs == ref_v, (k, band[0], 'vertices not shared')
            assert sorted(es) == ref_e, (k, band[0], 'edge length differs')
    print("SKILLING  62/63/64 and 65/66/67 share vertices and edge "
          "length, as his remarks column says")

    # --- Hart's central-freedom cubes -----------------------------------
    # Nothing is aligned, so the stabilizer is trivial and the count is
    # the whole group order.  Checking that it holds at SEVERAL angles is
    # the point: a count that changed with the angle would mean the
    # "general position" had accidentally hit a special one.
    for k, lbl, grp, want in _cmp.FREE_COMPOUNDS:
        for ph in (17.0, 31.0, 43.0):
            comps = build_compound(k, phase=ph)
            assert len(comps) == want, (k, ph, len(comps), want)

    # --- Skilling 68-75, duplication of enantiomorphs -------------------
    # Two constituents is true by construction here, so the assertion
    # that carries weight is the CHIRALITY: each of these eight must
    # mirror onto a different vertex set, and the two achiral snubs the
    # data also holds must mirror onto themselves.  That asymmetry is
    # what validates the Wythoff-symbol-to-U-number mapping, since an
    # achiral solid cannot appear in Skilling's band at all.
    for k, lbl, u, grp, want in _cmp.ENANTIOMORPHS:
        comps = build_compound(k)
        assert len(comps) == want, (k, len(comps), want)
        a = {tuple(round(c, 4) for c in v) for v in comps[0][0]}
        b = {tuple(round(c, 4) for c in v) for v in comps[1][0]}
        assert a != b, (k, 'the mirror image is not distinct')
        assert len(a) == len(b), (k, len(a), len(b))
    for u in (32, 72):                    # the achiral snubs in the data
        assert len(_cmp.enantiomorph_pair(u)) == 1, \
            (u, 'an achiral snub produced a two-part compound')
    print("ENANTIOMORPHS  Skilling 68-75, all chiral, 2 constituents; "
          "U32 and U72 correctly collapse to 1")

    # --- Hart's inscribed pieces ----------------------------------------
    # The claim these models make is a containment, so that is what gets
    # checked -- component counts would pass on two solids merely sitting
    # near each other.
    for k in ('INS_TETRA_CUBE', 'INS_2TETRA_CUBE', 'INS_CUBE_DODECA',
              'INS_TETRA_DODECA'):
        comps = build_compound(k)
        host = {tuple(round(x, 7) for x in v) for v in comps[-1][0]}
        for V, _F in comps[:-1]:
            assert all(tuple(round(x, 7) for x in v) in host for v in V), \
                (k, 'an inscribed vertex is not a vertex of the host')

    # The icosahedron-in-octahedron is the one that is not a selection:
    # each of its vertices must sit ON an octahedron EDGE, one per edge,
    # cutting it in the golden ratio.  Landing on a FACE instead would
    # still look plausible and is the easy way to get this wrong.
    for k in ('INS_ICOSA_OCTA', 'INS_2ICOSA_OCTA'):
        comps = build_compound(k)
        OV, OF = comps[-1]
        E = set()
        for f in OF:
            for i in range(len(f)):
                E.add(tuple(sorted((f[i], f[(i + 1) % len(f)]))))
        assert len(E) == 12, (k, len(E))
        used, ratios = set(), set()
        for V, _F in comps[:-1]:
            for v in V:
                on = None
                for a, b in E:
                    A, B = OV[a], OV[b]
                    d = [B[i] - A[i] for i in range(3)]
                    L = sum(x * x for x in d)
                    t = sum((v[i] - A[i]) * d[i] for i in range(3)) / L
                    if not -1e-9 <= t <= 1 + 1e-9:
                        continue
                    q = [A[i] + t * d[i] for i in range(3)]
                    if _math.dist(q, v) < 1e-9:
                        on = ((a, b), round(t, 6))
                        break
                assert on, (k, v, 'vertex is not on any octahedron edge')
                used.add(on[0])
                ratios.add(on[1])
            assert len(used) == 12, (k, len(used))
            used.clear()
        phi = (1 + 5 ** 0.5) / 2
        assert ratios == {round(1 / phi, 6), round(1 / (phi * phi), 6)}, \
            (k, ratios, 'the division is not the golden ratio')
    print("INSCRIBED  6 models; icosahedron cuts all 12 octahedron edges "
          "in the golden ratio")

    # --- star prisms ----------------------------------------------------
    # These carry Skilling's rows 36, 37 and 41.  Their vertices are an
    # ordinary prism's, only rescaled, so the compound COUNTS cannot tell
    # a correct star circuit from a wrong one -- the count check above
    # would pass on a {5/2} face wound as a pentagon.  Check the faces
    # directly: the circuit must visit every vertex, and every edge must
    # come out unit, which happens only at the star's own circumradius.
    for n, m, star in ((5, 2, 5), (10, 3, 10)):
        V, F = _cmp.star_prism_solid(n, m)
        assert len(V) == 2 * n and len(F) == n + 2, (n, m, len(V), len(F))
        rings = [f for f in F if len(f) == star]
        assert len(rings) == 2, (n, m, sorted(len(f) for f in F))
        for r in rings:
            assert sorted(i % n for i in r) == list(range(n)), (n, m, r)
        lens = [_math.dist(V[f[i]], V[f[(i + 1) % len(f)]])
                for f in F for i in range(len(f))]
        assert max(lens) - min(lens) < 1e-9, (n, m, min(lens), max(lens))
        assert abs(lens[0] - 1.0) < 1e-9, (n, m, lens[0])

    # The star ANTIprism is groundwork for Skilling's rows 28/29/44/45,
    # which are NOT shipped -- see BACKLOG.md.  The solid itself is
    # correct and checked here: it closes, every edge is unit and lies in
    # exactly two faces.  What it is not is the constituent Skilling
    # means, whose symmetry includes a horizontal mirror that this one
    # lacks; keeping the check makes that groundwork honest rather than
    # dead.
    # Both antiprisms on the pentagram: Coxeter's 'first' (aligned bases,
    # D_5h, Skilling 44/45) and 'second' (staggered, D_5d, 28/29).  They
    # are distinguished by the horizontal mirror, and by the circumradius
    # CLM's rho^2 = 1/(3 -+ a) predicts -- 0.656431 against 0.587785.
    import numpy as _np
    for fn, want_r, want_mirror in ((_cmp.star_antiprism_second, 0.656431,
                                     True),
                                    (_cmp.star_antiprism_solid, 0.587785,
                                     False)):
        V, F = fn(5, 2 if fn is _cmp.star_antiprism_second else 3)
        assert abs(_math.dist((0, 0, 0), V[0]) - want_r) < 1e-6, \
            (fn.__name__, _math.dist((0, 0, 0), V[0]), want_r)
        S = {tuple(_np.round(v, 4)) for v in V}
        mir = {tuple(_np.round((v[0], v[1], -v[2]), 4)) for v in V}
        assert (mir == S) is want_mirror, \
            (fn.__name__, 'horizontal mirror', mir == S, want_mirror)

    V, F = _cmp.star_antiprism_solid(5, 3)
    assert len(V) == 10 and len(F) == 12, (len(V), len(F))
    assert sorted(len(f) for f in F) == [3] * 10 + [5] * 2, \
        sorted(len(f) for f in F)
    ecount = {}
    for f in F:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            ecount[e] = ecount.get(e, 0) + 1
    assert len(ecount) == 20 and set(ecount.values()) == {2}, \
        (len(ecount), sorted(set(ecount.values())))
    lens = [_math.dist(V[a], V[b]) for a, b in ecount]
    assert max(lens) - min(lens) < 1e-9 and abs(lens[0] - 1.0) < 1e-9, \
        (min(lens), max(lens))

    # --- prism / antiprism with its dual --------------------------------
    # `prism_and_dual` raises if the solid has no midsphere, so the whole
    # of Hart's range exercising cleanly is itself the edge-tangency
    # claim being checked -- for prisms n = 3..10 and antiprisms n = 4..10,
    # plus the ends of the operator's wider slider.
    for n in list(range(3, 11)) + [24]:
        for anti in (False, True):
            (V, F), (P, G) = build_compound(
                'ANTIPRISM_DUAL' if anti else 'PRISM_DUAL', sides=n)
            # the prism has 2n vertices and n+2 faces, and its dual
            # swaps the two; the antiprism has 2n vertices and 2n+2
            assert len(V) == 2 * n, (n, anti, len(V))
            assert len(F) == (2 * n + 2 if anti else n + 2), (n, anti,
                                                              len(F))
            assert len(P) == len(F) and len(G) == len(V), (n, anti)

            # The point of reciprocating in the MIDSPHERE rather than in
            # any sphere: each dual edge must cross its primal edge, at
            # right angles, at the point where both touch that sphere.
            #
            # Compare TANGENT points, not midpoints.  A prism's edges are
            # symmetric about their closest point so the two coincide
            # there, but a dipyramid's are not -- its apex and equator sit
            # at different radii -- and the midpoint version of this test
            # fails on a perfectly correct compound.
            def touch(VV, FF):
                out = {}
                for f in FF:
                    for i in range(len(f)):
                        a, b = VV[f[i]], VV[f[(i + 1) % len(f)]]
                        m = tuple(round(c, 7) + 0.0
                                  for c in _cmp.edge_touch(a, b))
                        out[m] = tuple(b[k] - a[k] for k in range(3))
                return out

            mv, mp = touch(V, F), touch(P, G)
            assert set(mv) == set(mp), (n, anti, len(mv), len(mp))
            rho = _math.dist((0, 0, 0), next(iter(mv)))
            for m, d1 in mv.items():
                assert abs(_math.dist((0, 0, 0), m) - rho) < 1e-7, (n, anti, m)
                d2 = mp[m]
                c = sum(d1[k] * d2[k] for k in range(3))
                n1 = _math.sqrt(sum(x * x for x in d1))
                n2 = _math.sqrt(sum(x * x for x in d2))
                assert abs(c) / (n1 * n2) < 1e-7, (n, anti, m, c)
    print("PRISM_DUAL/ANTIPRISM_DUAL  n=3..10,24: dual edges cross primal "
          "edges perpendicularly at the shared midsphere tangent points")
    print("RESULT: OK")
