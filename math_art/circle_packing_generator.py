
# Circle Packing Generator for Blender
#
# A circle packing realises a prescribed pattern of tangencies: given a
# triangulation, assign a radius to every vertex so that circles at the ends of
# each edge touch.  The radii are not free -- the combinatorics determines the
# geometry, up to the ambient normalisation.  That rigidity is the discrete
# analogue of the Riemann mapping theorem: prescribe the boundary and the
# interior follows conformally, so a packing computes conformal structure rather
# than merely drawing circles.
#
# Radii come from Thurston's relaxation in the Collins-Stephenson formulation:
# sweep the interior vertices resetting each radius to the one that closes its
# own flower, accelerated by their super-step.  The per-vertex update is the
# Uniform Neighbour Model, a closed form, so there is no inner root-find.
# Positions then come from walking the complex and fanning each vertex's
# neighbours around it.
#
# Modes:
#   MAXIMAL       the hyperbolic maximal packing of the disc -- the discrete
#                 Riemann map, with the boundary circles horocycles internally
#                 tangent to the unit circle
#   PRESCRIBED    boundary radii given by a lobed function of angle, giving a
#                 discrete conformal map onto that shape
#   FREE          boundary radii held; the cheapest mode
#   SPHERE        a packing of the sphere, computed by removing a face, packing
#                 the resulting disc hyperbolically and projecting
#
# Hexagonal refinement subdivides and re-packs; under refinement the packing
# converges to the true conformal structure (Rodin-Sullivan).
#
# The engine is in `math_art/packing/` and is Blender-free.
#
# References:
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005.
# - Charles R. Collins and Kenneth Stephenson, "A circle packing algorithm",
#   Computational Geometry: Theory and Applications 25 (2003), pp. 233-256.
# - Charles Collins, Gerald L. Orick and Kenneth Stephenson, "A linearized
#   circle packing algorithm", Computational Geometry: Theory and Applications
#   64 (2017), pp. 13-29.
# - Burt Rodin and Dennis Sullivan, "The convergence of circle packings to the
#   Riemann mapping", Journal of Differential Geometry 26 (1987), pp. 349-360.
# - William P. Thurston, "The Geometry and Topology of Three-Manifolds",
#   Princeton lecture notes, 1980, chapter 13.

bl_info = {
    "name": "Circle Packing",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Circle Packing",
    "description": "Circle packings: discrete Riemann maps, maximal packings "
                   "of the disc, and packings of the sphere",
    "category": "Add Mesh",
}

import math

from . import packing
from .packing import (EUCLIDEAN, FREE, HYPERBOLIC, MAXIMAL, PRESCRIBED,
                      hex_disc, layout_euclidean, layout_hyperbolic, pack,
                      pack_sphere, refine_n, square_grid, tangency_error)

_PALETTE = [
    (0.90, 0.91, 0.94, 1.0), (0.16, 0.22, 0.61, 1.0),
    (0.29, 0.38, 0.78, 1.0), (0.46, 0.56, 0.90, 1.0),
    (0.60, 0.42, 0.09, 1.0), (0.84, 0.64, 0.29, 1.0),
    (0.35, 0.40, 0.47, 1.0), (0.12, 0.14, 0.18, 1.0),
]
_NPAL = len(_PALETTE)


def _center_fit(verts, scale=1.0):
    """Centre on the bounding box and fit the largest extent to a 2 m cube --
    the project-wide convention -- then apply `scale`.

    A hyperbolic packing comes out inside the unit disc and a euclidean one at
    whatever size the radii happened to settle on, so without this the output
    size is an accident of the solve."""
    if not verts:
        return verts
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    ext = max(hi[i] - lo[i] for i in range(3))
    k = (2.0 / ext if ext > 1e-9 else 1.0) * scale
    mid = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    return [tuple((v[i] - mid[i]) * k for i in range(3)) for v in verts]


def _ring(cx, cy, r, tube, seg, tseg):
    """A torus of centreline radius r in the z = 0 plane."""
    verts = []
    faces = []
    for i in range(seg):
        a = 2.0 * math.pi * i / seg
        ca, sa = math.cos(a), math.sin(a)
        for j in range(tseg):
            b = 2.0 * math.pi * j / tseg
            rr = r + tube * math.cos(b)
            verts.append((cx + rr * ca, cy + rr * sa, tube * math.sin(b)))
    for i in range(seg):
        for j in range(tseg):
            a = i * tseg + j
            b = i * tseg + (j + 1) % tseg
            c = ((i + 1) % seg) * tseg + (j + 1) % tseg
            d = ((i + 1) % seg) * tseg + j
            faces.append((a, b, c, d))
    return verts, faces


def _disc(cx, cy, r, seg, z=0.0):
    verts = [(cx, cy, z)]
    for i in range(seg):
        a = 2.0 * math.pi * i / seg
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
    faces = [(0, i + 1, (i + 1) % seg + 1) for i in range(seg)]
    return verts, faces


def _spherical_cap(axis, ang, res=2):
    """A spherical cap as a dome on the unit sphere: the set of points within
    angular radius `ang` of `axis`.

    Built as concentric rings on the sphere itself.  Squashing a ball into a
    lens -- the obvious shortcut -- gives ellipsoid blobs that do not lie on the
    sphere and do not meet their neighbours where the packing says they should."""
    rings = max(2, 2 + 2 * res)
    seg = max(8, 8 * (res + 1))
    up = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    ex = (up[1] * axis[2] - up[2] * axis[1],
          up[2] * axis[0] - up[0] * axis[2],
          up[0] * axis[1] - up[1] * axis[0])
    m = math.sqrt(sum(c * c for c in ex)) or 1.0
    ex = tuple(c / m for c in ex)
    ey = (axis[1] * ex[2] - axis[2] * ex[1],
          axis[2] * ex[0] - axis[0] * ex[2],
          axis[0] * ex[1] - axis[1] * ex[0])

    verts = [tuple(axis)]                       # the pole
    for i in range(1, rings + 1):
        a = ang * i / rings
        ca, sa = math.cos(a), math.sin(a)
        for j in range(seg):
            t = 2.0 * math.pi * j / seg
            c, sn = math.cos(t), math.sin(t)
            verts.append(tuple(ca * axis[k] + sa * (c * ex[k] + sn * ey[k])
                               for k in range(3)))
    faces = []
    for j in range(seg):                        # the polar fan
        faces.append((0, 1 + j, 1 + (j + 1) % seg))
    for i in range(rings - 1):                  # the quad bands
        b0 = 1 + i * seg
        b1 = 1 + (i + 1) * seg
        for j in range(seg):
            k = (j + 1) % seg
            faces.append((b0 + j, b1 + j, b1 + k, b0 + k))
    return verts, faces


def _icosphere(level=2):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [tuple(c / math.sqrt(1 + t * t) for c in v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache = {}
        nf = []

        def mid(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                p = tuple((verts[a][i] + verts[b][i]) * 0.5 for i in range(3))
                n = math.sqrt(sum(c * c for c in p))
                cache[key] = len(verts)
                verts.append(tuple(c / n for c in p))
            return cache[key]

        for (a, b, c) in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = nf
    return verts, faces


def _mat_index(color_by, v, K, radii, rmin, rmax):
    if color_by == 'UNIFORM':
        return 1
    if color_by == 'BOUNDARY':
        return 4 if not K.is_interior(v) else 2
    if color_by == 'VALENCE':
        return 1 + (K.degree(v) % (_NPAL - 1))
    r = radii[v]
    if rmax <= rmin:
        return 2
    t = (math.log(max(r, 1e-12)) - rmin) / (rmax - rmin)
    return 1 + int(min(0.999, max(0.0, t)) * (_NPAL - 1))


def build_packing(combinatorics='HEX', rings=5, grid=6, geometry='EUCLIDEAN',
                  boundary='PRESCRIBED', lobes=3, amplitude=0.75,
                  base_radius=0.35, refine=0, output='DISCS', tube_ratio=0.14,
                  ring_seg=24, tube_seg=8, sphere_res=2, color_by='RADIUS',
                  scale=1.0):
    """Build the packing and return (verts, faces, mats, report)."""
    if combinatorics == 'SPHERE':
        K = (packing.tetrahedron() if rings <= 1 else packing.octahedron())
        if refine:
            K, _ = refine_n(K, None, min(refine, 3))
        axes, ang, info = pack_sphere(K)
        verts, faces, mats = [], [], []
        for v in range(K.nv):
            if axes[v] is None or ang[v] <= 1e-9:
                continue
            cv, cf = _spherical_cap(axes[v], ang[v], sphere_res)
            base = len(verts)
            verts.extend(cv)
            for f in cf:
                faces.append(tuple(base + i for i in f))
                mats.append(1 + (v % (_NPAL - 1)))
        report = ("sphere: %d caps, %d sweeps, angle error %.2e"
                  % (K.nv, info['sweeps'], info['angle_error']))
        return _center_fit(verts, scale), faces, mats, report

    if combinatorics == 'GRID':
        K, xy = square_grid(grid, grid)
    else:
        K, xy = hex_disc(rings)
    if refine:
        K, xy = refine_n(K, xy, min(refine, 4))

    geom = HYPERBOLIC if geometry == 'HYPERBOLIC' else EUCLIDEAN
    if boundary == 'MAXIMAL':
        geom = HYPERBOLIC
        bmode = MAXIMAL
        br = None
    elif boundary == 'PRESCRIBED':
        bmode = PRESCRIBED
        br = [0.0] * K.nv
        for w in K.boundary:
            x, y = xy[w]
            a = math.atan2(y, x)
            br[w] = base_radius * (1.0 + amplitude * math.cos(lobes * a))
            if geom == HYPERBOLIC:
                br[w] = min(0.95, max(0.05, br[w]))
    else:
        bmode = FREE
        br = None

    res = pack(K, geom, boundary=bmode, boundary_radii=br)
    if geom == HYPERBOLIC:
        cen, rad = layout_hyperbolic(K, res.radii)
        pos = [None if c is None else (c.real, c.imag) for c in cen]
        terr = None
    else:
        pos = layout_euclidean(K, res.radii)
        rad = list(res.radii)
        terr = tangency_error(K, pos, rad)

    fin = [r for r in rad if r > 0]
    rmin = math.log(min(fin)) if fin else 0.0
    rmax = math.log(max(fin)) if fin else 1.0

    verts, faces, mats = [], [], []
    for v in range(K.nv):
        if pos[v] is None or rad[v] <= 0.0:
            continue
        x, y = pos[v][0], pos[v][1]
        r = rad[v]
        mi = _mat_index(color_by, v, K, rad, rmin, rmax)
        if output == 'DISCS':
            dv, df = _disc(x, y, r, ring_seg)
        else:
            dv, df = _ring(x, y, r, max(1e-5, r * tube_ratio), ring_seg,
                           tube_seg)
        base = len(verts)
        verts.extend(dv)
        for f in df:
            faces.append(tuple(base + i for i in f))
            mats.append(mi)

    if output == 'GRAPH':
        verts, faces, mats = [], [], []
        idx = {}
        for v in range(K.nv):
            if pos[v] is None:
                continue
            idx[v] = len(verts)
            verts.append((pos[v][0], pos[v][1], 0.0))
        for (a, b) in K.edges():
            if a in idx and b in idx:
                faces.append((idx[a], idx[b], idx[a]))
                mats.append(1)

    verts = _center_fit(verts, scale)
    circles = sum(1 for v in range(K.nv) if pos[v] is not None)
    report = ("%d circles, %d sweeps, angle error %.2e"
              % (circles, res.sweeps, res.max_angle_error))
    if terr is not None:
        report += ", tangency %.2e" % terr
    if not res.converged:
        report += " (STALLED)"
    return verts, faces, mats, report


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _packing_mesh(name, verts, faces, mats, smooth):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in verts], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        for idx, rgba in enumerate(_PALETTE):
            nm = "CirclePacking_%d" % idx
            m = bpy.data.materials.get(nm)
            if m is None:
                m = bpy.data.materials.new(nm)
                m.diffuse_color = rgba
            me.materials.append(m)
        if mats and len(mats) == len(me.polygons):
            me.polygons.foreach_set('material_index', mats)
        if smooth:
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        return me

    class MESH_OT_circle_packing_add(bpy.types.Operator):
        """Add a circle packing: a discrete conformal map built from a
        prescribed pattern of tangencies"""
        bl_idname = "mesh.circle_packing_add"
        bl_label = "Circle Packing"
        bl_options = {'REGISTER', 'UNDO'}

        combinatorics: EnumProperty(
            name="Pattern",
            items=[('HEX', "Hexagonal Disc",
                    "Rings of circles around a centre"),
                   ('GRID', "Square Grid",
                    "A triangulated rectangular grid"),
                   ('SPHERE', "Sphere",
                    "Pack the sphere, by removing a face and projecting")],
            default='HEX')
        rings: IntProperty(
            name="Rings", default=5, min=1, max=40,
            description="Rings of circles around the centre; the circle count "
                        "grows as three times this squared")
        grid: IntProperty(
            name="Grid Size", default=6, min=1, max=60,
            description="Cells across the square grid")
        geometry: EnumProperty(
            name="Geometry",
            items=[('EUCLIDEAN', "Euclidean", "Pack in the plane"),
                   ('HYPERBOLIC', "Hyperbolic",
                    "Pack in the Poincare disc")],
            default='EUCLIDEAN')
        boundary: EnumProperty(
            name="Boundary",
            items=[('PRESCRIBED', "Prescribed",
                    "Boundary radii follow a lobed function of angle, giving "
                    "a conformal map onto that shape"),
                   ('MAXIMAL', "Maximal Disc",
                    "The discrete Riemann map: boundary circles become "
                    "horocycles tangent to the unit circle"),
                   ('FREE', "Free",
                    "Hold the boundary radii and relax only the interior")],
            default='PRESCRIBED')
        lobes: IntProperty(
            name="Lobes", default=3, min=0, max=12,
            description="How many lobes the prescribed boundary has")
        amplitude: FloatProperty(
            name="Lobe Depth", default=0.75, min=0.0, max=0.95,
            description="How far the prescribed boundary radius swings")
        base_radius: FloatProperty(
            name="Boundary Size", default=0.35, min=0.02, max=3.0,
            description="Average radius of the boundary circles")
        refine: IntProperty(
            name="Refinement", default=0, min=0, max=4,
            description="Hexagonal refinement passes; each multiplies the "
                        "circle count by about four and converges toward the "
                        "true conformal structure")
        output: EnumProperty(
            name="Output",
            items=[('DISCS', "Discs", "Each circle as a flat disc"),
                   ('RINGS', "Rings", "Each circle as a torus"),
                   ('GRAPH', "Tangency Graph",
                    "Edges joining the centres of touching circles")],
            default='DISCS')
        tube_ratio: FloatProperty(
            name="Ring Thickness", default=0.14, min=0.01, max=0.5,
            description="Tube radius as a fraction of each circle's radius")
        ring_seg: IntProperty(name="Ring Segments", default=24, min=6, max=96)
        tube_seg: IntProperty(name="Tube Segments", default=8, min=3, max=32)
        sphere_res: IntProperty(
            name="Cap Resolution", default=2, min=0, max=4,
            description="Subdivision of each spherical cap")
        color_by: EnumProperty(
            name="Material By",
            items=[('RADIUS', "Radius", "Shade by circle size"),
                   ('VALENCE', "Neighbours",
                    "Shade by how many circles each one touches"),
                   ('BOUNDARY', "Boundary",
                    "Distinguish boundary circles from interior ones"),
                   ('UNIFORM', "Uniform", "One material")],
            default='RADIUS')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Shade Smooth", default=True)

        def execute(self, context):
            try:
                verts, faces, mats, report = build_packing(
                    self.combinatorics, self.rings, self.grid, self.geometry,
                    self.boundary, self.lobes, self.amplitude,
                    self.base_radius, self.refine, self.output,
                    self.tube_ratio, self.ring_seg, self.tube_seg,
                    self.sphere_res, self.color_by, self.scale)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            me = _packing_mesh("CirclePacking", verts, faces, mats,
                               self.smooth)
            obj = bpy.data.objects.new("CirclePacking", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, report)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'combinatorics')
            if self.combinatorics == 'SPHERE':
                lay.prop(self, 'rings', text="Seed Solid")
                lay.prop(self, 'refine')
                lay.prop(self, 'sphere_res')
            else:
                if self.combinatorics == 'GRID':
                    lay.prop(self, 'grid')
                else:
                    lay.prop(self, 'rings')
                lay.prop(self, 'geometry')
                lay.prop(self, 'boundary')
                if self.boundary == 'PRESCRIBED':
                    lay.prop(self, 'lobes')
                    lay.prop(self, 'amplitude')
                    lay.prop(self, 'base_radius')
                lay.prop(self, 'refine')
                lay.prop(self, 'output')
                if self.output == 'RINGS':
                    lay.prop(self, 'tube_ratio')
                    lay.prop(self, 'tube_seg')
                if self.output != 'GRAPH':
                    lay.prop(self, 'ring_seg')
            lay.prop(self, 'color_by')
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.circle_packing_add", icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_circle_packing_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_circle_packing_add)


def _selftest():
    """Build every mode headlessly; raises AssertionError on failure."""
    for mode in ('RINGS', 'DISCS', 'GRAPH'):
        v, f, m, rep = build_packing(rings=3, output=mode)
        assert v and f, (mode, len(v), len(f))
        assert len(m) == len(f)
        assert 'STALLED' not in rep, rep

    v, f, m, rep = build_packing(rings=3, boundary='MAXIMAL',
                                 geometry='HYPERBOLIC')
    assert v and 'STALLED' not in rep, rep
    xs = [abs(complex(p[0], p[1])) for p in v]
    assert max(xs) < 1.6, "a maximal packing should sit near the unit disc"

    v, f, m, rep = build_packing(combinatorics='GRID', grid=4,
                                 boundary='FREE')
    assert v and f

    v, f, m, rep = build_packing(combinatorics='SPHERE', rings=2,
                                 sphere_res=1)
    assert v and f, rep
    rr = [math.sqrt(sum(c * c for c in p)) for p in v]
    assert max(rr) < 1.3, "spherical caps should lie on the unit sphere"

    v, f, m, rep = build_packing(rings=2, refine=1)
    assert v and 'STALLED' not in rep, rep

    print("circle_packing_generator: rings/discs/graph, maximal, grid, "
          "sphere and refinement all build. RESULT: OK")
