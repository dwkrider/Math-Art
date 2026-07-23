
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

bl_info = {
    "name": "Symmetry Sphere (Orbifolds)",
    "author": "David Krider (Math Art project, after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Symmetry Sphere",
    "description": "Comma motif in relief on a sphere under any of "
                   "the 14 spherical symmetry types (orbifolds)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, sqrt, radians

import numpy as np

PHI = (1 + sqrt(5)) / 2


# ---------------------------------------------------------------- #
#  point groups by closure                                         #
# ---------------------------------------------------------------- #

def _rot(axis, ang):
    """3x3 rotation about axis by ang (Rodrigues)."""
    x, y, z = axis
    l = sqrt(x * x + y * y + z * z)
    x, y, z = x / l, y / l, z / l
    c, s = cos(ang), sin(ang)
    C = 1 - c
    return ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C))


def _mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _key(A):
    return tuple(round(A[i][j], 9) for i in range(3)
                 for j in range(3))


def _det(A):
    return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))


_SIGMA_H = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
_SIGMA_V = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
_SWAP_XY = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
_MINUS_I = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))

# the seven infinite families:
# (generator builders taking n, expected order as a function of n)
_FAMILIES = ('NN', 'STAR_NN', 'N_STAR', 'NX', '22N', '2STAR_N',
             'STAR_22N')


def _s2n(n):
    """S2n rotoreflection about Z: rotate pi/n, reflect in z=0."""
    return _mul(_rot((0, 0, 1), pi / n), _SIGMA_H)


def _generators(signature, n):
    cz = _rot((0, 0, 1), 2 * pi / n)
    c2x = _rot((1, 0, 0), pi)
    if signature == 'NN':                       # Cn
        return (cz,)
    if signature == 'STAR_NN':                  # Cnv
        return (cz, _SIGMA_V)
    if signature == 'N_STAR':                   # Cnh
        return (cz, _SIGMA_H)
    if signature == 'NX':                       # S2n
        return (_s2n(n),)
    if signature == '22N':                      # Dn
        return (cz, c2x)
    if signature == '2STAR_N':                  # Dnd
        return (cz, c2x, _s2n(n))
    if signature == 'STAR_22N':                 # Dnh
        return (cz, c2x, _SIGMA_H)
    t = (_rot((1, 1, 1), 2 * pi / 3), _rot((0, 0, 1), pi))
    o = (_rot((0, 0, 1), pi / 2), _rot((1, 1, 1), 2 * pi / 3))
    i = (_rot((0, 1, PHI), 2 * pi / 5), _rot((1, 1, 1), 2 * pi / 3))
    if signature == '332':                      # T
        return t
    if signature == 'STAR_332':                 # Td
        return t + (_SWAP_XY,)
    if signature == '3STAR_2':                  # Th
        return t + (_MINUS_I,)
    if signature == '432':                      # O
        return o
    if signature == 'STAR_432':                 # Oh
        return o + (_MINUS_I,)
    if signature == '532':                      # I
        return i
    if signature == 'STAR_532':                 # Ih
        return i + (_MINUS_I,)
    raise ValueError(signature)


def expected_order(signature, n=1):
    """Theoretical group order for each orbifold signature."""
    fam = {'NN': n, 'STAR_NN': 2 * n, 'N_STAR': 2 * n, 'NX': 2 * n,
           '22N': 2 * n, '2STAR_N': 4 * n, 'STAR_22N': 4 * n}
    odd = {'332': 12, 'STAR_332': 24, '3STAR_2': 24, '432': 24,
           'STAR_432': 48, '532': 60, 'STAR_532': 120}
    return fam[signature] if signature in fam else odd[signature]


def build_group(signature, n=1):
    """All matrices of the group, generated by closure under
    multiplication with rounding-dedupe."""
    gens = _generators(signature, n)
    I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    seen = {_key(I): I}
    frontier = [I]
    while frontier:
        nxt = []
        for A in frontier:
            for G in gens:
                B = _mul(G, A)
                k = _key(B)
                if k not in seen:
                    seen[k] = B
                    nxt.append(B)
        frontier = nxt
        if len(seen) > 1000:            # runaway float drift guard
            break
    return list(seen.values())


# ---------------------------------------------------------------- #
#  the comma motif and the sphere shell                            #
# ---------------------------------------------------------------- #

def comma_outline(size=1.0, steps=22, cap_steps=14):
    """Closed 2D polygon of a comma / teardrop: a circular head
    blended into a curved, tapering tail (an arc of a circle for
    the head cap plus offset curves of a circular-spiral tail
    centerline).  Deliberately chiral: the tail curls clockwise."""
    r0 = 0.30 * size                    # head radius
    rho = 0.55 * size                   # tail centerline radius
    sweep = radians(150.0)              # tail curl angle
    cy = -rho                           # centerline circle centre

    def cn(t):
        """Centerline point and outward normal at t in [0, 1]."""
        psi = pi / 2 - sweep * t
        return ((rho * cos(psi), cy + rho * sin(psi)),
                (cos(psi), sin(psi)))

    def w(t):
        return r0 * (1.0 - t) ** 1.4    # tapering half-width

    pts = []
    for i in range(steps):              # upper edge, head -> tip
        t = i / steps
        (cx, cyy), (nx, ny) = cn(t)
        ww = w(t)
        pts.append((cx + ww * nx, cyy + ww * ny))
    (cx, cyy), _ = cn(1.0)
    pts.append((cx, cyy))               # tail tip
    for i in range(steps - 1, -1, -1):  # lower edge, tip -> head
        t = i / steps
        (cx, cyy), (nx, ny) = cn(t)
        ww = w(t)
        pts.append((cx - ww * nx, cyy - ww * ny))
    for k in range(1, cap_steps):       # head cap, the long way round
        a = -pi / 2 - pi * k / cap_steps
        pts.append((r0 * cos(a), r0 * sin(a)))
    return pts


def uv_sphere(radius=1.0, segs=48, rings=24):
    """Plain UV sphere shell (verts, faces)."""
    verts = [(0.0, 0.0, radius)]
    for i in range(1, rings):
        th = pi * i / rings
        st, ct = sin(th), cos(th)
        for j in range(segs):
            ph = 2 * pi * j / segs
            verts.append((radius * st * cos(ph),
                          radius * st * sin(ph), radius * ct))
    verts.append((0.0, 0.0, -radius))
    last = len(verts) - 1
    faces = []
    for j in range(segs):
        faces.append([0, 1 + j, 1 + (j + 1) % segs])
    for i in range(rings - 2):
        a = 1 + i * segs
        b = a + segs
        for j in range(segs):
            j1 = (j + 1) % segs
            faces.append([a + j, a + j1, b + j1, b + j])
    a = 1 + (rings - 2) * segs
    for j in range(segs):
        faces.append([last, a + (j + 1) % segs, a + j])
    return verts, faces


# a generic point: spherical coords theta=63, phi=21 degrees --
# off every mirror plane and rotation axis of all 14 groups
_GENERIC_THETA = radians(63.0)
_GENERIC_PHI = radians(21.0)


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
            name="Sphere Radius", default=1.0, min=0.01, max=100.0)
        resolution: IntProperty(
            name="Sphere Resolution", default=48, min=8, max=256,
            description="Longitudinal segments of the sphere shell")
        motif_size: FloatProperty(
            name="Motif Size", default=0.3, min=0.01, max=10.0,
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
                base = np.vstack((dirs * (R - depth),
                                  dirs * (R * 1.001 + 0.1 * depth)))
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
            for M in G:
                A = np.array(M)
                mirror = _det(M) < 0
                n_mirror += mirror
                off = len(verts)
                verts.extend(map(tuple, base @ A.T))
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

            if carve:
                # sphere object minus the comma cutters
                sv, sf = uv_sphere(R, self.resolution,
                                   max(4, self.resolution // 2))
                me = bpy.data.meshes.new("Symmetry Sphere")
                me.from_pydata(sv, [], sf)
                me.validate(clean_customdata=True)
                sbm = bmesh.new()
                sbm.from_mesh(me)
                bmesh.ops.recalc_face_normals(sbm, faces=sbm.faces)
                sbm.to_mesh(me)
                sbm.free()
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
                mod = obj.modifiers.new("Carve", 'BOOLEAN')
                mod.operation = 'DIFFERENCE'
                mod.object = cutter
                # In the larger reflective groups adjacent comma
                # cutters overlap; plain EXACT then yields an empty
                # mesh.  EXACT with use_self resolves the cutter's
                # self-intersections; FLOAT is the last resort.
                carved = None
                for solver, self_x in (('EXACT', True),
                                       ('EXACT', False),
                                       ('FLOAT', False)):
                    try:
                        mod.solver = solver
                        mod.use_self = self_x
                    except (TypeError, AttributeError):
                        pass
                    deps = context.evaluated_depsgraph_get()
                    ev = obj.evaluated_get(deps)
                    result = bpy.data.meshes.new_from_object(ev)
                    if len(result.polygons):
                        carved = result
                        break
                    bpy.data.meshes.remove(result)
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
                # merge with the sphere shell (overlapping union)
                sv, sf = uv_sphere(R, self.resolution,
                                   max(4, self.resolution // 2))
                off = len(verts)
                verts.extend(sv)
                faces.extend([tuple(off + i for i in f)
                              for f in sf])
                mats.extend([0] * len(sf))

                me = bpy.data.meshes.new("Symmetry Sphere")
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
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
            for k in ('radius', 'resolution', 'motif_size',
                      'relief', 'color_reflected'):
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


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
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
