
# Orbifold Symmetry Sphere generator for Blender
#
# After the "comma symmetry spheres" of Henry Segerman's Visualizing
# Mathematics with 3D Printing, ch. 1, figs 1.19-1.29: a sphere
# decorated with a comma-shaped motif in raised relief, replicated
# under one of the 14 types of spherical symmetry, selected by
# Conway orbifold signature.  The comma is deliberately chiral and
# sits at a generic point (off every mirror plane and rotation
# axis), so the pattern of commas -- and their mirror images under
# orientation-reversing elements -- makes the symmetry type visible
# at a glance.
#
# The 14 types: seven infinite families
#
#   nn    Cn    cyclic                          order n
#   *nn   Cnv   pyramidal                       order 2n
#   n*    Cnh   rotation + horizontal mirror    order 2n
#   nx    S2n   rotoreflection                  order 2n
#   22n   Dn    dihedral                        order 2n
#   2*n   Dnd   antiprismatic                   order 4n
#   *22n  Dnh   prismatic                       order 4n
#
# and seven oddities
#
#   332   T   12    *332  Td  24    3*2  Th  24
#   432   O   24    *432  Oh  48
#   532   I   60    *532  Ih  120
#
# Every group is built explicitly as a set of 3x3 orthogonal
# matrices: generators closed under multiplication with a
# rounding-dedupe (all orders <= 120).
#
# References:
#   - Henry Segerman, "Visualizing Mathematics with 3D Printing"
#     (Johns Hopkins University Press, 2016), ch. 1, figs 1.19-1.29
#     -- the comma symmetry spheres.
#   - John H. Conway, Heidi Burgiel, Chaim Goodman-Strauss, "The
#     Symmetries of Things" (A K Peters, 2008) -- Conway-Thurston
#     orbifold notation, the magic theorem, and the enumeration of
#     the 14 spherical symmetry types.

bl_info = {
    "name": "Symmetry Sphere (Orbifolds)",
    "author": "Math Art project (after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Symmetry Sphere",
    "description": "Comma motif in relief on a sphere under any of "
                   "the 14 spherical symmetry types (orbifolds)",
    "category": "Add Mesh",
}
import numpy as np

# The mathematics lives in the sibling `patterns` engine package;
# this module is the Blender layer over it.
try:
    from .patterns.spherical import (_FAMILIES, _GENERIC_PHI,
                                         _GENERIC_THETA, _det, build_group,
                                         comma_outline, cos, expected_order,
                                         sin, sphere_shell)
except ImportError:  # flat import outside the package
    from patterns.spherical import (_FAMILIES, _GENERIC_PHI,
                                        _GENERIC_THETA, _det, build_group,
                                        comma_outline, cos, expected_order,
                                        sin, sphere_shell)







# ---------------------------------------------------------------- #
#  point groups by closure                                         #
# ---------------------------------------------------------------- #




















# ---------------------------------------------------------------- #
#  the comma motif and the sphere shell                            #
# ---------------------------------------------------------------- #











try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _material(name, rgb):
        mat = bpy.data.materials.get(name)
        if mat is not None:
            return mat
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*rgb, 1.0)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.5
        return mat

    class MESH_OT_orbifold_sphere_add(bpy.types.Operator):
        """Add a sphere with a comma motif in raised relief,
        replicated under one of the 14 spherical symmetry types
        (after Segerman's comma symmetry spheres)"""
        bl_idname = "mesh.orbifold_sphere_add"
        bl_label = "Symmetry Sphere"
        bl_options = {'REGISTER', 'UNDO'}

        signature: EnumProperty(
            name="Signature",
            items=[('NN', "nn - cyclic (Cn)",
                    "n-fold rotation about the vertical axis; "
                    "order n"),
                   ('STAR_NN', "*nn - pyramidal (Cnv)",
                    "n-fold rotation with vertical mirrors; "
                    "order 2n"),
                   ('N_STAR', "n* - Cnh",
                    "n-fold rotation with a horizontal mirror; "
                    "order 2n"),
                   ('NX', "nx - rotoreflection (S2n)",
                    "2n-fold rotoreflection about the vertical "
                    "axis; order 2n"),
                   ('22N', "22n - dihedral (Dn)",
                    "n-fold rotation with horizontal 2-fold axes; "
                    "order 2n"),
                   ('2STAR_N', "2*n - antiprismatic (Dnd)",
                    "Dihedral with diagonal mirrors, the symmetry "
                    "of an antiprism; order 4n"),
                   ('STAR_22N', "*22n - prismatic (Dnh)",
                    "Dihedral with horizontal and vertical "
                    "mirrors, the symmetry of a prism; order 4n"),
                   ('332', "332 - tetrahedral (T)",
                    "Rotations of the tetrahedron; order 12"),
                   ('STAR_332', "*332 - full tetrahedral (Td)",
                    "All symmetries of the tetrahedron; order 24"),
                   ('3STAR_2', "3*2 - pyritohedral (Th)",
                    "Tetrahedral rotations plus central "
                    "inversion; order 24"),
                   ('432', "432 - octahedral (O)",
                    "Rotations of the cube/octahedron; order 24"),
                   ('STAR_432', "*432 - full octahedral (Oh)",
                    "All symmetries of the cube/octahedron; "
                    "order 48"),
                   ('532', "532 - icosahedral (I)",
                    "Rotations of the icosahedron; order 60"),
                   ('STAR_532', "*532 - full icosahedral (Ih)",
                    "All symmetries of the icosahedron; "
                    "order 120")],
            default='532',
            description="Spherical symmetry type by Conway "
                        "orbifold signature")
        n: IntProperty(
            name="n", default=6, min=1, max=32,
            description="Order of the main axis for the seven "
                        "infinite families (ignored by the seven "
                        "oddities)")
        radius: FloatProperty(
            name="Sphere Radius", default=1.0, min=0.01, max=100.0,
            description="Radius of the decorated sphere")
        resolution: IntProperty(
            name="Sphere Resolution", default=48, min=8, max=256,
            description="Longitudinal segments of the sphere shell")
        thickness: FloatProperty(
            name="Shell Thickness", default=0.03, min=0.0, max=10.0,
            description="Wall thickness of the hollow sphere "
                        "(0 = solid ball); negative relief deeper "
                        "than the wall cuts comma-shaped holes "
                        "right through the shell")
        motif_size: FloatProperty(
            name="Motif Size", default=0.25, min=0.01, max=10.0,
            description="Overall size of the comma motif")
        relief: FloatProperty(
            name="Motif Relief", default=-0.05, min=-10.0, max=10.0,
            description="Negative carves the commas into the "
                        "sphere (boolean difference); positive "
                        "raises them above the surface")
        color_reflected: BoolProperty(
            name="Color Reflected Copies", default=False,
            description="Second material on the mirror-image "
                        "commas (orientation-reversing copies)")

        def execute(self, context):
            G = build_group(self.signature, self.n)
            want = expected_order(self.signature, self.n)
            if len(G) != want:
                self.report({'WARNING'},
                            f"Group closure gave {len(G)} elements "
                            f"(theory: {want})")
            R = self.radius
            h = self.relief

            # comma polygon, triangulated with bmesh
            outline = comma_outline(self.motif_size)
            bm = bmesh.new()
            bvs = [bm.verts.new((x, y, 0.0)) for x, y in outline]
            face = bm.faces.new(bvs)
            bmesh.ops.triangulate(bm, faces=[face])
            bm.verts.index_update()
            tris = [tuple(v.index for v in f.verts)
                    for f in bm.faces]
            bm.free()
            m = len(outline)

            # place the motif tangent to the sphere at the generic
            # point, curve it onto the sphere (radial projection),
            # and raise it by the relief height (closed bump)
            th, ph = _GENERIC_THETA, _GENERIC_PHI
            p0 = np.array((sin(th) * cos(ph), sin(th) * sin(ph),
                           cos(th)))
            eu = np.array((cos(th) * cos(ph), cos(th) * sin(ph),
                           -sin(th)))
            ev = np.array((-sin(ph), cos(ph), 0.0))
            xy = np.array(outline)
            pts = R * p0 + np.outer(xy[:, 0], eu) \
                + np.outer(xy[:, 1], ev)
            dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            carve = h < 0
            if carve:
                # cutter solids: from below the carve depth up to
                # just above the surface, removed by boolean
                depth = min(-h, 0.9 * R)
                lo, hi = R - depth, R * 1.001 + 0.1 * depth
                base = None
            else:
                base = np.vstack((dirs * R, dirs * (R + h)))
            bump_faces = []
            for a, b, c in tris:
                bump_faces.append((m + a, m + b, m + c))  # top cap
                bump_faces.append((c, b, a))              # bottom
            for i in range(m):                            # sides
                j = (i + 1) % m
                bump_faces.append((i, j, m + j, m + i))

            # replicate under every group element; the det<0
            # (orientation-reversing) copies are mirror commas
            verts = []
            faces = []
            mats = []
            n_mirror = 0
            for k, M in enumerate(G):
                A = np.array(M)
                mirror = _det(M) < 0
                n_mirror += mirror
                off = len(verts)
                if carve:
                    # tiny per-copy radial jitter: where commas
                    # overlap, identical floor/cap radii would make
                    # their boundary surfaces exactly coincident,
                    # which degenerates the exact boolean
                    eps = 1e-4 * (k + 1) / len(G)
                    bk = np.vstack((dirs * (lo * (1.0 - eps)),
                                    dirs * (hi * (1.0 + eps))))
                else:
                    bk = base
                verts.extend(map(tuple, bk @ A.T))
                if mirror:
                    fs = [tuple(off + i for i in reversed(f))
                          for f in bump_faces]
                else:
                    fs = [tuple(off + i for i in f)
                          for f in bump_faces]
                faces.extend(fs)
                mats.extend([1 if (mirror and self.color_reflected)
                             else 0] * len(fs))
            n_bump = len(faces)

            thick = min(self.thickness, 0.9 * R)
            if carve:
                # hollow shell (or solid ball at thickness 0)
                # minus the comma cutters; relief deeper than the
                # wall pierces right through.  The windings are set
                # by construction -- a blanket recalc would flip
                # the inner wall away from the cavity.
                sv, sf = sphere_shell(R, thick, self.resolution,
                                      max(4, self.resolution // 2))
                me = bpy.data.meshes.new("Symmetry Sphere")
                me.from_pydata(sv, [], sf)
                me.validate(clean_customdata=True)
                me.materials.append(
                    _material("Symmetry Sphere", (0.85, 0.82, 0.75)))
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
                me.update()
                cme = bpy.data.meshes.new("SymmetryCutter")
                cme.from_pydata(verts, [], faces)
                cme.validate(clean_customdata=True)
                cbm = bmesh.new()
                cbm.from_mesh(cme)
                bmesh.ops.recalc_face_normals(cbm, faces=cbm.faces)
                cbm.to_mesh(cme)
                cbm.free()
                cme.materials.append(
                    _material("Symmetry Sphere", (0.85, 0.82, 0.75)))
                cme.materials.append(
                    _material("Symmetry Sphere Mirror",
                              (0.75, 0.30, 0.20)))
                if self.color_reflected \
                        and len(cme.polygons) == len(mats):
                    cme.polygons.foreach_set('material_index', mats)
                cme.update()
                obj = bpy.data.objects.new("Symmetry Sphere", me)
                context.collection.objects.link(obj)
                cutter = bpy.data.objects.new("SymmetryCutter", cme)
                context.collection.objects.link(cutter)

                # In the larger reflective groups adjacent commas
                # overlap, and a difference against overlapping
                # solids drops the doubly-covered regions (even-odd
                # rule) or leaks.  So first resolve the cutter into
                # one clean solid: a self-union (use_self) against a
                # far-away dummy cube.  The dummy never touches the
                # sphere, so it can stay in the cutter.
                dme = bpy.data.meshes.new("SymmetryDummy")
                dme.from_pydata(
                    [(50.0 + x, 50.0 + y, 50.0 + z)
                     for x in (0, 1) for y in (0, 1)
                     for z in (0, 1)], [],
                    [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                     (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
                dummy = bpy.data.objects.new("SymmetryDummy", dme)
                context.collection.objects.link(dummy)
                um = cutter.modifiers.new("SelfUnion", 'BOOLEAN')
                um.operation = 'UNION'
                um.object = dummy
                try:
                    um.solver = 'EXACT'
                    um.use_self = True
                except (TypeError, AttributeError):
                    pass
                deps = context.evaluated_depsgraph_get()
                clean = bpy.data.meshes.new_from_object(
                    cutter.evaluated_get(deps))
                cutter.modifiers.remove(um)
                cutter.data = clean
                bpy.data.meshes.remove(cme)
                cme = clean
                bpy.data.objects.remove(dummy, do_unlink=True)
                bpy.data.meshes.remove(dme)

                mod = obj.modifiers.new("Carve", 'BOOLEAN')
                mod.operation = 'DIFFERENCE'
                mod.object = cutter
                # With the cutter resolved, plain EXACT handles
                # every signature; keep the cascade as a safety
                # net, accepting the first watertight non-empty
                # result and a leaky one only as a last resort.
                def _open_edges(mesh):
                    cnt = {}
                    for p in mesh.polygons:
                        vs = p.vertices
                        for i in range(len(vs)):
                            a, b = vs[i], vs[(i + 1) % len(vs)]
                            k = (a, b) if a < b else (b, a)
                            cnt[k] = cnt.get(k, 0) + 1
                    return sum(1 for c in cnt.values() if c == 1)

                carved = None
                fallback = None
                for solver, self_x in (('EXACT', False),
                                       ('EXACT', True),
                                       ('FLOAT', False)):
                    try:
                        mod.solver = solver
                        mod.use_self = self_x
                    except (TypeError, AttributeError):
                        pass
                    deps = context.evaluated_depsgraph_get()
                    ev = obj.evaluated_get(deps)
                    result = bpy.data.meshes.new_from_object(ev)
                    if len(result.polygons) == 0:
                        bpy.data.meshes.remove(result)
                    elif _open_edges(result) == 0:
                        carved = result
                        break
                    elif fallback is None:
                        fallback = result
                    else:
                        bpy.data.meshes.remove(result)
                if carved is None and fallback is not None:
                    carved = fallback
                    fallback = None
                if fallback is not None:
                    bpy.data.meshes.remove(fallback)
                if carved is None:      # give back the plain sphere
                    self.report(
                        {'WARNING'},
                        "Boolean carve failed; sphere left uncut")
                    carved = me.copy()
                obj.modifiers.remove(mod)
                old = obj.data
                obj.data = carved
                bpy.data.meshes.remove(old)
                bpy.data.objects.remove(cutter, do_unlink=True)
                bpy.data.meshes.remove(cme)
                me = obj.data
            else:
                # merge with the sphere shell (overlapping union);
                # the inner wall goes in last so it can be re-flipped
                # toward the cavity after the normal recalc
                sv, sf = sphere_shell(R, thick, self.resolution,
                                      max(4, self.resolution // 2))
                n_outer = (len(sf) + 1) // 2 if thick > 0 else len(sf)
                off = len(verts)
                verts.extend(sv)
                faces.extend([tuple(off + i for i in f)
                              for f in sf])
                mats.extend([0] * len(sf))
                n_inner = len(sf) - n_outer

                me = bpy.data.meshes.new("Symmetry Sphere")
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                if n_inner:
                    bm.faces.ensure_lookup_table()
                    bmesh.ops.reverse_faces(
                        bm, faces=[bm.faces[i] for i in
                                   range(len(faces) - n_inner,
                                         len(faces))])
                bm.to_mesh(me)
                bm.free()
                if len(me.polygons) == len(mats):
                    if self.color_reflected:
                        me.materials.append(
                            _material("Symmetry Sphere",
                                      (0.85, 0.82, 0.75)))
                        me.materials.append(
                            _material("Symmetry Sphere Mirror",
                                      (0.75, 0.30, 0.20)))
                        me.polygons.foreach_set('material_index',
                                                mats)
                    smooth = ([False] * n_bump
                              + [True] * (len(mats) - n_bump))
                    me.polygons.foreach_set('use_smooth', smooth)
                me.update()
                obj = bpy.data.objects.new("Symmetry Sphere", me)
                context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"|G|={len(G)} ({n_mirror} mirrored) "
                        f"V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'signature')
            if self.signature in _FAMILIES:
                lay.prop(self, 'n')
            for k in ('radius', 'resolution', 'thickness',
                      'motif_size', 'relief', 'color_reflected'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.orbifold_sphere_add",
                             icon='SPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_orbifold_sphere_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_orbifold_sphere_add)


def _selftest():
    sigs = list(_FAMILIES) + ['332', 'STAR_332', '3STAR_2',
                              '432', 'STAR_432', '532',
                              'STAR_532']
    ok = True
    for n in (1, 4, 5, 6):
        for sig in sigs:
            G = build_group(sig, n)
            want = expected_order(sig, n)
            good = len(G) == want
            ok = ok and good
            dets = sorted(round(_det(A)) for A in G)
            plus = dets.count(1)
            if not good:
                print(f"{sig} n={n}: {len(G)} (want {want}) "
                      f"BAD")
            elif n == 6:
                print(f"{sig} n={n}: order {len(G)} "
                      f"({plus} rotations, "
                      f"{len(G) - plus} reversing) OK")
    pts = comma_outline()
    print(f"comma outline: {len(pts)} points")
    print("ALL OK" if ok else "FAILURES")
    assert ok
