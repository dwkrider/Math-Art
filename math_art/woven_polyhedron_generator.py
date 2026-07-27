
# Woven Polyhedron generator for Blender.
#
# A dual-polyhedron sculpture: an inset pentagon sits on every face
# of a dodecahedron and an inset triangle on every face of its dual
# icosahedron (the icosahedron built as the exact dual, so its faces
# align with the dodecahedron's vertices).  A twisted bezier ribbon
# bridges each pentagon edge to one triangle edge, and the whole
# thing is welded into ONE closed shell so a Solidify + Subdivision
# pair can thicken it and round the corners.
#
# Each pentagon fans out five ribbons that swirl to the surrounding
# triangles; the pentagon/triangle spins, sizes, curvatures and the
# ribbon width profile are all adjustable.  The ribbon-to-triangle
# edge pairing is topological (rotation-proof) and folds with the
# triangle's 120-degree symmetry, so the weave stays consistent at
# any spin angle.

bl_info = {
    "name": "Woven Polyhedron",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Woven Polyhedron",
    "description": "Ribbons weaving a dodecahedron to its dual "
                   "icosahedron",
    "category": "Add Mesh",
}

import math

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix
    from bpy.props import (FloatProperty, IntProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


PHI = (1 + 5 ** 0.5) / 2
R_DOD = 1.0


# ---------------------------------------------------------------- #
#  parameter-independent base data (icosahedron + its dual)        #
# ---------------------------------------------------------------- #

# icosahedron verts / faces (the seed) ...
IV = [(-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0),
      (0, -1, PHI), (0, 1, PHI), (0, -1, -PHI), (0, 1, -PHI),
      (PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)]
IF = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
      (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
      (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
      (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]


if _IN_BLENDER:
    IVv = [Vector(v) for v in IV]
    ICIRC = IVv[0].length
    # dodecahedron vertices are the icosahedron's face centroids
    # (unit sphere); its faces are the centroids around each icosa
    # vertex ordered by angle -> exact dual, faces meet the
    # icosahedron's vertices.
    CENT = [((IVv[a] + IVv[b] + IVv[c]) / 3.0) for a, b, c in IF]
    CDIR = [c.normalized() for c in CENT]
    DOD_VERTS = [d * R_DOD for d in CDIR]
    DOD_FACES = []
    for _vi in range(12):
        _ring = [fi for fi, f in enumerate(IF) if _vi in f]
        _n = IVv[_vi].normalized()
        _ref = Vector((0, 0, 1)) if abs(_n.z) < 0.9 \
            else Vector((1, 0, 0))
        _u = _n.cross(_ref).normalized()
        _w = _n.cross(_u)
        _ring.sort(key=lambda fi: math.atan2(CDIR[fi].dot(_w),
                                             CDIR[fi].dot(_u)))
        DOD_FACES.append(_ring)


# ---------------------------------------------------------------- #
#  geometry helpers                                                #
# ---------------------------------------------------------------- #

def poly_data(verts, faces):
    out = []
    for f in faces:
        vs = [Vector(verts[i]) for i in f]
        c = sum(vs, Vector()) / len(vs)
        nrm = (vs[1] - vs[0]).cross(vs[2] - vs[0]).normalized()
        out.append({"verts": vs, "center": c, "normal": nrm})
    return out


def inset_faces(faces, factor):
    res = []
    for fd in faces:
        c = fd["center"]
        vs = [c + factor * (v - c) for v in fd["verts"]]
        res.append({"verts": vs, "center": c,
                    "normal": fd["normal"]})
    return res


def spin_faces(faces, deg):
    a = math.radians(deg)
    for fd in faces:
        c = fd["center"]
        axis = fd["normal"].normalized()
        if axis.dot(c) < 0:
            axis = -axis
        R = Matrix.Rotation(-a, 4, axis)   # -a about outward = CCW
        fd["verts"] = [c + (R @ (v - c)) for v in fd["verts"]]


def bez(cp, t):
    u = 1 - t
    return (u * u * u * cp[0] + 3 * u * u * t * cp[1]
            + 3 * u * t * t * cp[2] + t * t * t * cp[3])


def wprofile(w0, w1, mid, ap, at, t):
    """half-width w0->w1 in two eased halves meeting at the middle
    width.  `mid` = middle width as a fraction of the endpoint SUM
    (0.5 = the linear midpoint, <0.5 a waist thinner than both ends,
    >0.5 a bulge).  ap/at = taper: 0 is a straight linear taper to
    the middle width; higher reaches it sooner and holds it flat.
    Endpoints stay exact."""
    wmid = mid * (w0 + w1)
    if t <= 0.5:
        s = 2.0 * t                        # 0..1 over pentagon half
        we, k = w0, ap
    else:
        s = 2.0 * (1.0 - t)                # 1..0 over triangle half
        we, k = w1, at
    e = 1.0 - max(0.0, 1.0 - s) ** (1.0 + k)   # ease: linear at k=0
    return max(0.0, we + (wmid - we) * e)


def cap_faces(fd, curve):
    """(verts, tris): tessellate a flat inset face into a spherical
    cap blended by `curve` (0 = flat, 1 = on the sphere through its
    corners).  Corners stay put so the ribbon weld is preserved --
    only the interior bulges toward the sphere."""
    vs = fd["verts"]
    c = fd["center"]
    n = len(vs)
    r = vs[0].length                   # corner distance = sphere R

    def blend(p):
        return p.lerp(p.normalized() * r, curve)

    verts = [blend(v) for v in vs]                  # corners 0..n-1
    verts += [blend((vs[i] + c) * 0.5) for i in range(n)]   # mids
    verts.append(blend(c))                          # centre  2n
    ci = 2 * n
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append([i, j, n + j])
        tris.append([i, n + j, n + i])
        tris.append([n + i, n + j, ci])
    return verts, tris


def _edge_tangents(fd, curve):
    """per edge: (midpoint, outward tangent in the CURVED surface's
    tangent plane, curved surface normal).  The tangent lies in the
    face's tangent plane at the edge so a ribbon leaving along it
    shares a normal with the (possibly domed) face -- no crease."""
    vs, c = fd["verts"], fd["center"]
    n = len(vs)
    r = vs[0].length
    mids = [((vs[i] + c) * 0.5).lerp(
                ((vs[i] + c) * 0.5).normalized() * r, curve)
            for i in range(n)]
    out = []
    for i in range(n):
        a, b = vs[i], vs[(i + 1) % n]
        mmid = (mids[i] + mids[(i + 1) % n]) * 0.5
        m = (a + b) / 2
        e = (b - a).normalized()
        nsurf = (b - a).cross(mmid - m)     # curved-surface normal
        if nsurf.dot(m) < 0:
            nsurf = -nsurf
        nsurf.normalize()
        t = e.cross(nsurf).normalized()
        if t.dot(m - c) < 0:
            t = -t
        out.append((m, t, nsurf))
    return out


def pent_edges(faces, corner_idx, curve):
    out = []
    for fi, fd in enumerate(faces):
        vs = fd["verts"]
        et = _edge_tangents(fd, curve)
        for i in range(len(vs)):
            m, t, nsurf = et[i]
            out.append({"m": m, "t": t, "n": nsurf,
                        "left": corner_idx[fi][(i + 1) % len(vs)],
                        "a": vs[i], "b": vs[(i + 1) % len(vs)],
                        "pent": fi})
    return out


def tri_face_edges(faces, curve):
    per = []
    for fd in faces:
        vs = fd["verts"]
        et = _edge_tangents(fd, curve)
        es = [{"m": et[i][0], "t": et[i][1], "n": et[i][2],
               "a": vs[i], "b": vs[(i + 1) % len(vs)]}
              for i in range(len(vs))]
        per.append(es)
    return per


def build(p):
    """Return (verts, faces) for the woven sculpture described by the
    parameter bag `p`.  Everything accumulates into ONE welded vertex
    pool: the ribbon ends coincide with the pentagon / triangle edge
    verts, so they fuse and the Subdivision Surface smooths across the
    junctions instead of treating the parts as loose pieces."""
    ico_verts = [v * (p.r_ico / ICIRC) for v in IVv]
    pen = inset_faces(poly_data(DOD_VERTS, DOD_FACES), p.pen_inset)
    tri = inset_faces(poly_data(ico_verts, IF), p.tri_inset)
    spin_faces(tri, p.tri_rot)          # + = CCW
    spin_faces(pen, -p.pen_rot)         # pen_rot + = CW
    verts, faces = [], []
    vid = {}

    def add(v):
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        i = vid.get(key)
        if i is None:
            i = len(verts)
            vid[key] = i
            verts.append((v[0], v[1], v[2]))
        return i

    for fd in pen:
        cv, tr = cap_faces(fd, p.pen_curve)
        idx = [add(v) for v in cv]
        for f in tr:
            faces.append([idx[i] for i in f])
    for fd in tri:
        cv, tr = cap_faces(fd, p.tri_curve)
        idx = [add(v) for v in cv]
        for f in tr:
            faces.append([idx[i] for i in f])

    pen_ed = pent_edges(pen, DOD_FACES, p.pen_curve)
    tri_per = tri_face_edges(tri, p.tri_curve)
    N = p.rib_samples
    # the triangle is 120-symmetric, so the target edge index folds
    # by the number of 120 turns: default spin 90 reproduces the
    # confirmed layout (edge shift -1, verified numerically) and each
    # extra 120 steps it by one -> correct at any angle.
    step = -1 + round((p.tri_rot - 90.0) / 120.0)
    for pe in pen_ed:
        k = pe["left"]
        j = IF[k].index(pe["pent"])
        te = tri_per[k][(j - 1 + step) % 3]
        p0, tp = pe["m"], pe["t"]
        q, tq = te["m"], te["t"]
        if tq.dot(p0 - q) < 0:
            tq = -tq
        span = (q - p0).length
        Hp = p.handle_pen * span
        Ht = p.handle_tri * span
        pa, pb = pe["a"], pe["b"]
        qa, qb = te["a"], te["b"]
        if ((pa - qa).length + (pb - qb).length
                > (pa - qb).length + (pb - qa).length):
            qa, qb = qb, qa
        cpA = [pa, pa + Hp * tp, qa + Ht * tq, qa]
        cpB = [pb, pb + Hp * tp, qb + Ht * tq, qb]
        # keep the centreline + twist from the two rails, but drive
        # the half-width by its own profile (endpoints stay pinned)
        Wp = (pa - pb).length / 2.0
        Wq = (qa - qb).length / 2.0
        rows = []
        for s in range(N + 1):
            t = s / N
            A = bez(cpA, t)
            B = bez(cpB, t)
            d = B - A
            dl = d.length
            if dl > 1e-9:
                c = (A + B) / 2.0
                w = wprofile(Wp, Wq, p.mid_width,
                             p.width_pen, p.width_tri, t)
                A = c - w * (d / dl)
                B = c + w * (d / dl)
            rows.append((add(A), add(B)))
        for s in range(N):
            a0, b0 = rows[s]
            a1, b1 = rows[s + 1]
            faces.append([a0, b0, b1, a1])
    return verts, faces


# ---------------------------------------------------------------- #
#  Blender operator                                                #
# ---------------------------------------------------------------- #

if _IN_BLENDER:

    class _Bag:
        """lightweight parameter bag so build() is decoupled from the
        operator (also handy for headless tests)."""
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class MESH_OT_woven_polyhedron_add(bpy.types.Operator):
        """Add a woven polyhedron: twisted ribbons bridging the inset
        pentagons of a dodecahedron to the inset triangles of its dual
        icosahedron, welded into one shell with Solidify + Subdivision
        so it can be thickened and smoothed"""
        bl_idname = "mesh.woven_polyhedron_add"
        bl_label = "Woven Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        tri_rot: FloatProperty(
            name="Triangle Spin", default=90.0, min=-720, max=720,
            description="Spin of every triangle about its face "
                        "normal (degrees); folds with the triangle's "
                        "120-degree symmetry")
        pen_rot: FloatProperty(
            name="Pentagon Spin", default=-18.66, min=-720, max=720,
            description="Spin of every pentagon about its face "
                        "normal (degrees)")
        pen_inset: FloatProperty(
            name="Pentagon Size", default=0.32, min=0.05, max=1.0,
            description="Inset pentagon size as a fraction of the "
                        "dodecahedron face")
        tri_inset: FloatProperty(
            name="Triangle Size", default=0.15, min=0.05, max=1.0,
            description="Inset triangle size as a fraction of the "
                        "icosahedron face")
        r_ico: FloatProperty(
            name="Icosahedron Size", default=0.76, min=0.2, max=2.5,
            description="Icosahedron circumradius (dodecahedron is "
                        "fixed at 1)")
        pen_curve: FloatProperty(
            name="Pentagon Curvature", default=1.05, min=0.0,
            max=2.0,
            description="Dome each pentagon toward a sphere through "
                        "its corners (0 = flat)")
        tri_curve: FloatProperty(
            name="Triangle Curvature", default=0.0, min=0.0,
            max=2.0,
            description="Dome each triangle toward a sphere (0 = "
                        "flat)")
        handle_pen: FloatProperty(
            name="Stiffness (Pentagon)", default=1.5, min=0.0,
            max=3.0,
            description="Length of the bezier handle leaving the "
                        "pentagon edge (higher = straighter exit)")
        handle_tri: FloatProperty(
            name="Stiffness (Triangle)", default=0.4, min=0.0,
            max=3.0,
            description="Length of the bezier handle leaving the "
                        "triangle edge")
        mid_width: FloatProperty(
            name="Middle Width", default=0.29, min=0.0, max=1.5,
            description="Ribbon width at the middle as a fraction of "
                        "the endpoint sum (0.5 = linear midpoint, "
                        "less = a waist)")
        width_pen: FloatProperty(
            name="Taper (Pentagon)", default=50.65, min=0.0,
            max=100.0,
            description="How fast the pentagon half reaches the "
                        "middle width (0 = linear)")
        width_tri: FloatProperty(
            name="Taper (Triangle)", default=0.0, min=0.0, max=100.0,
            description="How fast the triangle half reaches the "
                        "middle width (0 = linear)")
        rib_samples: IntProperty(
            name="Ribbon Segments", default=14, min=2, max=40,
            description="Lengthwise segments per ribbon")
        thickness: FloatProperty(
            name="Thickness", default=0.03, min=0.0, max=0.5,
            description="Solidify shell thickness (0 = no Solidify "
                        "modifier)")
        smooth_level: IntProperty(
            name="Smoothing", default=2, min=0, max=4,
            description="Subdivision-Surface levels rounding the "
                        "corners (0 = no Subdivision modifier)")
        smooth_shading: BoolProperty(
            name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            verts, faces = build(self)
            s = self.scale
            me = bpy.data.meshes.new("Woven Polyhedron")
            me.from_pydata([(x * s, y * s, z * s)
                            for x, y, z in verts], [],
                           [list(f) for f in faces])
            me.validate()
            # weld coincident verts + unify windings so Solidify
            # builds one clean-sided shell (no black flipped faces)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts,
                                     dist=1e-5 * s)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            if self.smooth_shading:
                me.polygons.foreach_set(
                    "use_smooth", [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Woven Polyhedron", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.thickness > 0.0:
                sol = obj.modifiers.new("Thickness", "SOLIDIFY")
                sol.thickness = self.thickness * s
                sol.offset = 0.0
            if self.smooth_level > 0:
                sub = obj.modifiers.new("Smooth", "SUBSURF")
                sub.levels = self.smooth_level
                sub.render_levels = self.smooth_level
            self.report({'INFO'},
                        f"Woven Polyhedron: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for prop in ("tri_rot", "pen_rot", "pen_inset",
                         "tri_inset", "r_ico", "pen_curve",
                         "tri_curve", "handle_pen", "handle_tri",
                         "mid_width", "width_pen", "width_tri",
                         "rib_samples"):
                lay.prop(self, prop)
            lay.separator()
            lay.prop(self, "thickness")
            lay.prop(self, "smooth_level")
            lay.prop(self, "smooth_shading")
            lay.prop(self, "scale")

    def _menu_func(self, context):
        self.layout.operator("mesh.woven_polyhedron_add",
                             icon='MESH_ICOSPHERE')

    _classes = (MESH_OT_woven_polyhedron_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        print("Woven Polyhedron needs Blender (mathutils) to build; "
              "run tests/test_extension.py under Blender.")
