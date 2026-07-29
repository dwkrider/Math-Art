
# Escape-time Volumetric Fractal Generator for Blender
#
# Solid 3D escape-time fractals -- the Mandelbulb, quaternion Julia
# sets and the Mandelbox -- meshed as watertight objects. Each point
# of a sample grid is iterated under the fractal map; points whose
# orbit stays bounded are INSIDE the set, points whose orbit escapes
# are OUTSIDE. A distance estimate (Hart-Sandin-Kauffman) gives the
# outside points a sub-voxel distance to the boundary, so the zero
# level set of the resulting signed field -- extracted by the sibling
# Minimal Surface Toolkit's marching_tets -- is a smooth, printable
# surface rather than a blocky voxel shell. An optional largest-
# connected-component pass drops the isolated specks these sets throw
# off, leaving a single closed solid centred in a 2 m cube.
#
#   MANDELBULB   z -> z^p + c in White-Nylander spherical form, the
#                classic "power 8" bulb (p is exposed)
#   JULIA        quaternion Julia set z -> z^2 + c for a fixed
#                quaternion c, drawn as its w = const 3D slice
#   MANDELBOX    z -> scale * boxFold(ballFold(z)) + c
#
# References:
# - Mandelbulb: Daniel White and Paul Nylander (2009), the spherical
#   power-p map popularised at fractalforums.com.
# - Quaternion Julia sets: Alan Norton, "Generation and display of
#   geometric fractals in 3-D", Computer Graphics (SIGGRAPH) 16(3),
#   1982, pp. 61-67.
# - Mandelbox: Tom Lowe (2010), fractalforums.com.
# - Distance estimation of escape-time fractals: John C. Hart, Daniel
#   J. Sandin and Louis H. Kauffman, "Ray tracing deterministic 3-D
#   fractals", Computer Graphics (SIGGRAPH) 23(3), 1989, pp. 289-296.

bl_info = {
    "name": "Escape-time Fractals",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Escape-time Fractal",
    "description": "Mandelbulb, quaternion Julia and Mandelbox solids",
    "category": "Add Mesh",
}

import math

import numpy as np


def _toolkit():
    """The sibling Minimal Surface Toolkit supplies marching_tets.
    Relative import inside the Math Art package, absolute when this
    file runs standalone next to the toolkit."""
    try:
        from . import minimal_surface_toolkit as mst
    except ImportError:
        import minimal_surface_toolkit as mst
    return mst


# ==========================================================================
# Signed distance-estimator fields.  Each returns a callable
# field(X, Y, Z) -> array that is negative inside the set and a
# positive distance estimate outside, ready for marching_tets.
# `cell` is the grid spacing, used as the interior sentinel so the
# zero crossing lands about a cell-fraction outside the last solid
# sample.
# ==========================================================================

def _mandelbulb_field(power, iters, bailout, cell):
    def field(X, Y, Z):
        shape = X.shape
        cx, cy, cz = X.ravel(), Y.ravel(), Z.ravel()
        zx = np.zeros_like(cx)
        zy = np.zeros_like(cy)
        zz = np.zeros_like(cz)
        dr = np.ones_like(cx)
        escaped = np.zeros(cx.shape, dtype=bool)
        for _ in range(iters):
            r = np.sqrt(zx * zx + zy * zy + zz * zz)
            escaped |= r > bailout
            active = ~escaped
            rr = np.clip(r, 1e-9, None)
            theta = np.arccos(np.clip(zz / rr, -1.0, 1.0))
            phi = np.arctan2(zy, zx)
            zr = rr ** power
            new_dr = power * rr ** (power - 1.0) * dr + 1.0
            t, p = theta * power, phi * power
            sin_t = np.sin(t)
            nx = zr * sin_t * np.cos(p) + cx
            ny = zr * sin_t * np.sin(p) + cy
            nz = zr * np.cos(t) + cz
            zx = np.where(active, nx, zx)
            zy = np.where(active, ny, zy)
            zz = np.where(active, nz, zz)
            dr = np.where(active, new_dr, dr)
        r = np.sqrt(zx * zx + zy * zy + zz * zz)
        rr = np.clip(r, 1e-9, None)
        de = 0.5 * np.log(rr) * rr / np.maximum(dr, 1e-9)
        f = np.where(escaped, de, -0.5 * cell)
        return f.reshape(shape)
    return field


def _quat_mul(a, b):
    """Hamilton product of two quaternion arrays (each a 4-tuple of
    grids), returned as a 4-tuple of grids."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)


def _julia_field(c, w_slice, iters, bailout, cell):
    c0, c1, c2, c3 = c
    def field(X, Y, Z):
        shape = X.shape
        z0 = X.ravel().copy()
        z1 = Y.ravel().copy()
        z2 = Z.ravel().copy()
        z3 = np.full_like(z0, w_slice)
        dr = np.ones_like(z0)
        escaped = np.zeros(z0.shape, dtype=bool)
        for _ in range(iters):
            r = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3)
            escaped |= r > bailout
            active = ~escaped
            new_dr = 2.0 * r * dr
            s0, s1, s2, s3 = _quat_mul((z0, z1, z2, z3),
                                       (z0, z1, z2, z3))
            z0 = np.where(active, s0 + c0, z0)
            z1 = np.where(active, s1 + c1, z1)
            z2 = np.where(active, s2 + c2, z2)
            z3 = np.where(active, s3 + c3, z3)
            dr = np.where(active, new_dr, dr)
        r = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3)
        rr = np.clip(r, 1e-9, None)
        de = 0.5 * np.log(rr) * rr / np.maximum(dr, 1e-9)
        f = np.where(escaped, de, -0.5 * cell)
        return f.reshape(shape)
    return field


def _mandelbox_field(scale, min_r, fixed_r, iters, bailout, cell):
    min_r2, fixed_r2 = min_r * min_r, fixed_r * fixed_r
    def field(X, Y, Z):
        shape = X.shape
        cx, cy, cz = X.ravel(), Y.ravel(), Z.ravel()
        zx, zy, zz = cx.copy(), cy.copy(), cz.copy()
        dr = np.ones_like(cx)
        escaped = np.zeros(cx.shape, dtype=bool)
        for _ in range(iters):
            active = ~escaped

            def box_fold(v):
                v = np.where(v > 1.0, 2.0 - v, v)
                return np.where(v < -1.0, -2.0 - v, v)

            bx, by, bz = box_fold(zx), box_fold(zy), box_fold(zz)
            m2 = bx * bx + by * by + bz * bz
            factor = np.where(m2 < min_r2, fixed_r2 / min_r2,
                              np.where(m2 < fixed_r2,
                                       fixed_r2 / np.clip(m2, 1e-9, None),
                                       1.0))
            nx = scale * bx * factor + cx
            ny = scale * by * factor + cy
            nz = scale * bz * factor + cz
            new_dr = dr * factor * abs(scale) + 1.0
            zx = np.where(active, nx, zx)
            zy = np.where(active, ny, zy)
            zz = np.where(active, nz, zz)
            dr = np.where(active, new_dr, dr)
            r = np.sqrt(zx * zx + zy * zy + zz * zz)
            escaped |= r > bailout
        r = np.sqrt(zx * zx + zy * zy + zz * zz)
        de = r / np.maximum(np.abs(dr), 1e-9)
        f = np.where(escaped, de, -0.5 * cell)
        return f.reshape(shape)
    return field


# ==========================================================================
# Presets and driver
# ==========================================================================
# (label, box half-extent, default iterations, bailout)
PRESETS = {
    'MANDELBULB': ("Mandelbulb", 1.25, 10, 2.0),
    'JULIA': ("Quaternion Julia", 1.6, 12, 4.0),
    'MANDELBOX': ("Mandelbox", 6.0, 11, 6.0),
}

# a fixed quaternion constant that gives a well-known, richly detailed
# Julia set (drawn on the w = 0 slice)
_JULIA_C = (-0.2, 0.6, 0.2, 0.2)


def _largest_component(verts, tris):
    """Keep only the triangles of the largest edge-connected component
    (union-find over the mesh), dropping isolated specks. Returns
    remapped (verts, tris)."""
    if len(tris) == 0:
        return verts, tris
    n = len(verts)
    parent = np.arange(n)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:      # path compression
            parent[a], a = root, parent[a]
        return root

    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[rb] = ra
        if ra != rc:
            parent[rc] = ra
    roots = np.array([find(i) for i in range(n)])
    labels, counts = np.unique(roots, return_counts=True)
    keep_root = labels[np.argmax(counts)]
    vkeep = roots == keep_root
    tkeep = vkeep[tris[:, 0]]
    tris = tris[tkeep]
    used = np.unique(tris)
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tris]


def _field_for(kind, res, power, iters, julia_w):
    label, r, def_iters, bailout = PRESETS[kind]
    it = iters if iters > 0 else def_iters
    cell = 2.0 * r / res
    if kind == 'MANDELBULB':
        return _mandelbulb_field(power, it, bailout, cell), r
    if kind == 'JULIA':
        c = (_JULIA_C[0], _JULIA_C[1], _JULIA_C[2], _JULIA_C[3])
        return _julia_field(c, julia_w, it, bailout, cell), r
    return _mandelbox_field(2.0, 0.5, 1.0, it, bailout, cell), r


def build_escape_fractal(kind='MANDELBULB', res=96, power=8.0,
                         iters=0, julia_w=0.0, scale=1.0,
                         largest_only=True):
    """Mesh the boundary of an escape-time fractal. Returns
    (verts, tris) centred on the origin and fit to a 2 m cube."""
    field, r = _field_for(kind, res, power, iters, julia_w)
    mst = _toolkit()
    verts, tris = mst.marching_tets(field, (-r, -r, -r), (r, r, r),
                                    (res, res, res))
    if largest_only and len(tris):
        verts, tris = _largest_component(verts, tris)
    if len(verts):
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ext = float((hi - lo).max())
        verts = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9
                                             else 1.0)
    return verts * scale, tris


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_mandelbulb_add(bpy.types.Operator):
        """Add an escape-time volumetric fractal (Mandelbulb,
        quaternion Julia or Mandelbox) as a solid mesh"""
        bl_idname = "mesh.mandelbulb_add"
        bl_label = "Escape-time Fractal"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Fractal",
            items=[('MANDELBULB', "Mandelbulb",
                    "White-Nylander power-p bulb"),
                   ('JULIA', "Quaternion Julia",
                    "Quaternion Julia set, w = const slice"),
                   ('MANDELBOX', "Mandelbox",
                    "Box/ball-folding Mandelbox (scale 2)")],
            default='MANDELBULB')
        resolution: IntProperty(
            name="Resolution", default=96, min=24, max=256,
            description="Sample grid cells per axis (cost grows as "
                        "the cube; 96-160 is a good range)")
        power: FloatProperty(
            name="Bulb Power", default=8.0, min=2.0, max=16.0,
            description="Exponent p of the Mandelbulb map (8 is the "
                        "classic bulb); ignored by other presets")
        iterations: IntProperty(
            name="Iterations", default=0, min=0, max=32,
            description="Orbit iterations (0 = per-preset default); "
                        "more sharpens the boundary but adds cost")
        julia_w: FloatProperty(
            name="Julia Slice (w)", default=0.0, min=-1.5, max=1.5,
            description="Which w = const 3D slice of the 4D "
                        "quaternion Julia set to mesh")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        largest_only: BoolProperty(
            name="Largest Part Only", default=True,
            description="Keep only the largest connected component, "
                        "dropping isolated specks")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier of this "
                        "thickness (for a shell / printing)")
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            label = PRESETS[self.preset][0]
            verts, tris = build_escape_fractal(
                self.preset, self.resolution, power=self.power,
                iters=self.iterations, julia_w=self.julia_w,
                scale=self.scale, largest_only=self.largest_only)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty set at these settings")
                return {'CANCELLED'}
            obj = _new_object(context, label, verts, tris,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            self.report({'INFO'},
                        f"{label}: {len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'resolution')
            if self.preset == 'MANDELBULB':
                lay.prop(self, 'power')
            if self.preset == 'JULIA':
                lay.prop(self, 'julia_w')
            lay.prop(self, 'iterations')
            lay.prop(self, 'scale')
            lay.prop(self, 'largest_only')
            lay.prop(self, 'thickness')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.mandelbulb_add",
                             icon='META_BALL')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_mandelbulb_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_mandelbulb_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # standalone self-test: every preset must mesh to a non-empty,
        # closed (no boundary edges) single component
        from collections import Counter
        for kind in ('MANDELBULB', 'JULIA', 'MANDELBOX'):
            verts, tris = build_escape_fractal(kind, res=64)
            edges = Counter()
            for tri in tris:
                for i in range(3):
                    a, b = int(tri[i]), int(tri[(i + 1) % 3])
                    edges[(min(a, b), max(a, b))] += 1
            boundary = sum(1 for c in edges.values() if c != 2)
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            print(f"{kind}: V={len(verts)} F={len(tris)} "
                  f"boundary_edges={boundary} "
                  f"bbox={np.round(hi - lo, 3)} "
                  f"{'OK' if len(tris) and boundary == 0 else 'CHECK'}")
