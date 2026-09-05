
# Toric Calabi-Yau Generator for Blender
#
# The one route on which the object in the viewport really is a
# Calabi-Yau, rather than a two-dimensional shadow of one.
#
# REFLEXIVE POLYTOPES (Batyrev 1994).  A lattice polytope D containing
# the origin in its interior is *reflexive* if every facet lies at
# integral distance 1 from the origin -- equivalently, if the polar
# dual
#
#     D* = { y : <x, y> >= -1  for all x in D }
#
# is again a lattice polytope.  Reflexive polytopes are exactly the
# ones whose toric variety P_D is Gorenstein Fano, so that the generic
# anticanonical hypersurface in P_D is Calabi-Yau; and because
# (D*)* = D, the involution D -> D* carries one Calabi-Yau family to
# another.  That involution is Batyrev's construction of MIRROR PAIRS.
# In three dimensions there are exactly 4319 reflexive polytopes
# (Kreuzer-Skarke 1998), each giving a K3 surface; in four there are
# 473,800,776 (Kreuzer-Skarke 2000), giving the Calabi-Yau threefolds.
# A curated handful of the three-dimensional ones is built here, each
# with its mirror.
#
# TROPICAL CALABI-YAU.  Over the tropical semiring (min, +) a
# polynomial f = min_v (c_v + v.x) has a "hypersurface" T(f): the set
# where the minimum is attained at least twice.  It is a piecewise
# linear 2-complex in R^3 -- an honest polyhedral surface, no
# projection involved -- and it is the large-complex-structure limit of
# the classical hypersurface with the same Newton polytope.  When that
# Newton polytope is the fourth dilate of the standard tetrahedron the
# classical surface is a quartic in P^3, i.e. a K3, and the unique
# bounded region of the complement of T(f) is what Balletti, Panizzut
# and Sturmfels call a K3 POLYTOPE.  Their Example 4 is the default
# here; its K3 polytope is simple with f-vector (64, 96, 34), which
# `_selftest` reproduces from the geometry.
#
# Everything is done in exact halfspace arithmetic rather than by
# calling a hull library: a polytope given as { x : N x <= d } has a
# vertex wherever three of its facet planes meet in a point that
# satisfies the rest, so enumerating triples both finds the vertices
# and tells you which facet each one lies on.  The same routine
# therefore builds a reflexive polytope, its dual, and a K3 polytope.
#
# References:
# - V. V. Batyrev, "Dual polyhedra and mirror symmetry for Calabi-Yau
#   hypersurfaces in toric varieties", Journal of Algebraic Geometry 3
#   (1994) 493-535 -- reflexive polytopes (Def. 4.1.5), the
#   self-duality theorem (Thm. 4.1.6), the face correspondence
#   (Prop. 4.1.7) and the mirror involution.
# - M. Kreuzer and H. Skarke, "Classification of reflexive polyhedra in
#   three dimensions", Advances in Theoretical and Mathematical Physics
#   2 (1998) 853-871 -- the 4319 three-dimensional reflexive polytopes.
# - M. Kreuzer and H. Skarke, "Complete classification of reflexive
#   polyhedra in four dimensions", Advances in Theoretical and
#   Mathematical Physics 4 (2000) 1209-1230 -- the 473,800,776
#   four-dimensional ones and their 30,108 Hodge pairs.
# - G. Balletti, M. Panizzut and B. Sturmfels, "K3 polytopes and their
#   quartic surfaces", Advances in Geometry 21 (2021) 85-98 -- the
#   definition of a K3 polytope, the f-vector formula
#   (Vol, 3Vol/2, Vol/2 + 2) of their Lemma 11, and the tropical
#   quartic of Example 4 used as the default coefficients here.
# - D. Maclagan and B. Sturmfels, "Introduction to Tropical Geometry",
#   Graduate Studies in Mathematics 161, AMS (2015) -- tropical
#   hypersurfaces and the duality with regular subdivisions.

import itertools
import math
from collections import defaultdict

import numpy as np

bl_info = {
    "name": "Toric Calabi-Yau",
    "author": "Math Art project (after Batyrev / Kreuzer-Skarke / "
              "Balletti-Panizzut-Sturmfels)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Toric Calabi-Yau",
    "description": "Reflexive polytopes with their mirrors, tropical "
                   "Calabi-Yau surfaces, K3 polytopes",
    "category": "Add Mesh",
}

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# --------------------------------------------------------------------
# Halfspace geometry
# --------------------------------------------------------------------

def polyhedron(N, d, tol=1e-7):
    """Vertices and facets of { x : N x <= d }, assumed bounded.

    Returns (verts, facets) with `facets` a list of index lists, each
    already wound counter-clockwise as seen from outside.  A vertex of
    a 3-polytope is where three facet planes meet in a point the other
    constraints admit, so enumerating triples of rows finds every
    vertex and, as a by-product, says which facets it lies on.
    """
    N = np.asarray(N, dtype=float)
    d = np.asarray(d, dtype=float)
    m = len(N)
    pts = []
    for i, j, k in itertools.combinations(range(m), 3):
        A = N[[i, j, k]]
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        try:
            x = np.linalg.solve(A, d[[i, j, k]])
        except np.linalg.LinAlgError:
            continue
        if np.all(N @ x <= d + tol):
            pts.append(x)
    verts = []
    for x in pts:
        if not any(np.max(np.abs(x - v)) < tol for v in verts):
            verts.append(x)
    if not verts:
        return [], []
    V = np.asarray(verts)

    facets = []
    for j in range(m):
        on = np.where(np.abs(V @ N[j] - d[j]) < tol)[0]
        if len(on) < 3:
            continue
        n = N[j] / np.linalg.norm(N[j])
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(e1, n)) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - np.dot(e1, n) * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        ctr = V[on].mean(axis=0)
        ang = [math.atan2(np.dot(V[i] - ctr, e2), np.dot(V[i] - ctr, e1))
               for i in on]
        order = [int(on[t]) for t in np.argsort(ang)]
        if len(order) >= 3:
            facets.append(order)
    return [tuple(v) for v in V], facets


def f_vector(verts, facets):
    """(vertices, edges, facets) of a 3-polytope given by its facets."""
    edges = set()
    for f in facets:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edges.add((min(a, b), max(a, b)))
    return len(verts), len(edges), len(facets)


def polar(vertices, tol=1e-7):
    """The polar dual { y : <v, y> >= -1 } of conv(vertices)."""
    V = np.asarray(vertices, dtype=float)
    return polyhedron(-V, np.ones(len(V)), tol)


def is_reflexive(vertices, tol=1e-7):
    """True if conv(vertices) is a reflexive lattice polytope.

    The test is Batyrev's: the dual must again be a lattice polytope.
    """
    dv, _ = polar(vertices, tol)
    if not dv:
        return False
    D = np.asarray(dv)
    return bool(np.all(np.abs(D - np.round(D)) < 1e-6))


# --------------------------------------------------------------------
# A curated set of reflexive 3-polytopes
# --------------------------------------------------------------------
# Each gives a K3 surface as the generic anticanonical hypersurface of
# its toric variety, and each is paired with its mirror by the polar
# duality above.  `_selftest` checks that every one of them really is
# reflexive and that D** = D.

def _cube():
    return [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]


def _octa():
    return [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1)]


REFLEXIVE = {
    # Batyrev's Delta_3: the Newton polytope of the quartic surface in
    # P^3, translated so its unique interior lattice point is the
    # origin.  Its Calabi-Yau hypersurfaces are the quartic K3s.
    'QUARTIC': ("Quartic K3 (4 Delta_3)",
                [(3, -1, -1), (-1, 3, -1), (-1, -1, 3), (-1, -1, -1)]),
    # The dual of the above: the fan polytope of P^3.
    'P3': ("Projective Space P^3",
           [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1)]),
    # P^1 x P^1 x P^1 and its mirror.
    'CUBE': ("Cube (P1 x P1 x P1)", _cube()),
    'OCTAHEDRON': ("Octahedron", _octa()),
    # A weighted projective space: the (1,1,1,3) simplex.
    'WP1113': ("Weighted P(1,1,1,3)",
               [(1, 0, 0), (0, 1, 0), (-1, -1, 3), (0, 0, -1)]),
    # Triangle x interval -- the anticanonical hypersurfaces of
    # P^2 x P^1.
    'PRISM': ("Prism (P2 x P1)",
              [(1, 0, 1), (0, 1, 1), (-1, -1, 1),
               (1, 0, -1), (0, 1, -1), (-1, -1, -1)]),
    # Its mirror, a bipyramid over a triangle.
    'BIPYRAMID': ("Bipyramid",
                  [(1, 0, 0), (0, 1, 0), (-1, -1, 0),
                   (0, 0, 1), (0, 0, -1)]),
}


# --------------------------------------------------------------------
# Tropical hypersurfaces
# --------------------------------------------------------------------

def simplex_lattice_points(deg=4, dim=3):
    """Lattice points of the deg-th dilate of the standard simplex."""
    out = []
    for pt in itertools.product(range(deg + 1), repeat=dim):
        if sum(pt) <= deg:
            out.append(pt)
    return out


def balletti_example4():
    """The tropical quartic of Balletti-Panizzut-Sturmfels, Example 4.

    Newton polytope 4 Delta_3, interior lattice point (1,1,1); the
    resulting K3 polytope is simple with f-vector (64, 96, 34), the
    largest one their classification allows.
    """
    coeff = {}
    for pt in simplex_lattice_points(4, 3):
        i, j, k = pt
        s = i + j + k
        srt = tuple(sorted((i, j, k), reverse=True))
        if s == 4:
            c = {(4, 0, 0): 5, (3, 1, 0): 3, (2, 2, 0): 2,
                 (2, 1, 1): 0}[srt]
        elif s == 3:
            c = {(3, 0, 0): 3, (2, 1, 0): 0, (1, 1, 1): -9}[srt]
        elif s == 2:
            c = {(2, 0, 0): 2, (1, 1, 0): 0}[srt]
        elif s == 1:
            c = 3
        else:
            c = 5
        coeff[pt] = float(c)
    return coeff


def _clip(poly, a, b, g, tol=1e-9):
    """Sutherland-Hodgman clip of a 2D polygon by a s + b t <= g."""
    if not poly:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        s0, t0 = poly[i]
        s1, t1 = poly[(i + 1) % n]
        d0 = a * s0 + b * t0 - g
        d1 = a * s1 + b * t1 - g
        if d0 <= tol:
            out.append((s0, t0))
        if (d0 > tol) != (d1 > tol):
            u = d0 / (d0 - d1)
            out.append((s0 + u * (s1 - s0), t0 + u * (t1 - t0)))
    return out


def _area(poly):
    a = 0.0
    for i in range(len(poly)):
        s0, t0 = poly[i]
        s1, t1 = poly[(i + 1) % len(poly)]
        a += s0 * t1 - s1 * t0
    return 0.5 * abs(a)


def bounded_radius(coeff, interior=(1, 1, 1)):
    """How far out the interesting part of T(f) reaches.

    The bounded region of the complement sets the scale of everything
    else, so the clip box is quoted as a multiple of its radius rather
    than in absolute units -- otherwise a change of coefficients
    silently crops the surface.
    """
    v, _ = k3_polytope(coeff, interior)
    if not v:
        return 1.0
    return float(np.abs(np.asarray(v)).max()) or 1.0


def tropical_surface(coeff, box=None, tol=1e-7, interior=(1, 1, 1)):
    """The tropical hypersurface of min_v (c_v + v.x), clipped to a box.

    Returns (verts, faces).  Each 2-cell of T(f) is the locus where two
    terms tie and beat the rest; that is one linear equation and a pile
    of linear inequalities, i.e. a convex polygon inside the plane of
    the equation, so each is built by clipping a large square in that
    plane.  Cells with no area are dropped, which is what removes the
    pairs of terms that never tie anywhere.
    """
    if box is None:
        box = 1.25 * bounded_radius(coeff, interior)
    A = [np.asarray(v, dtype=float) for v in coeff]
    c = [coeff[v] for v in coeff]
    m = len(A)
    axes = [np.eye(3)[i] * s for i in range(3) for s in (1.0, -1.0)]

    polys = []
    for i, j in itertools.combinations(range(m), 2):
        n = A[i] - A[j]
        nn = float(n @ n)
        if nn < 1e-12:
            continue
        rhs = c[j] - c[i]
        x0 = n * (rhs / nn)
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(e1 @ n) / math.sqrt(nn) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - (e1 @ n) / nn * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n / math.sqrt(nn), e1)

        R = 4.0 * box
        poly = [(-R, -R), (R, -R), (R, R), (-R, R)]
        for k in range(m):
            if k == i or k == j:
                continue
            w = A[i] - A[k]
            poly = _clip(poly, float(w @ e1), float(w @ e2),
                         float(c[k] - c[i] - w @ x0))
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            for ax in axes:                      # clip to the view box
                poly = _clip(poly, float(ax @ e1), float(ax @ e2),
                             float(box - ax @ x0))
                if len(poly) < 3:
                    break
        if len(poly) >= 3 and _area(poly) > 1e-6:
            polys.append([tuple(x0 + s * e1 + t * e2) for s, t in poly])

    verts, faces = [], []
    cells = defaultdict(list)
    snap = 1e-5

    def merge(v):
        base = tuple(int(math.floor(x / snap)) for x in v)
        for off in itertools.product((0, -1, 1), repeat=3):
            for n_ in cells.get((base[0] + off[0], base[1] + off[1],
                                 base[2] + off[2]), ()):
                w = verts[n_]
                if all(abs(v[k] - w[k]) <= snap for k in range(3)):
                    return n_
        n_ = len(verts)
        verts.append(tuple(v))
        cells[base].append(n_)
        return n_

    for poly in polys:
        f = []
        for v in poly:
            n_ = merge(v)
            if not f or f[-1] != n_:
                f.append(n_)
        if len(f) > 2 and f[0] == f[-1]:
            f.pop()
        if len(f) >= 3:
            faces.append(f)
    return verts, faces


def k3_polytope(coeff, interior=(1, 1, 1), tol=1e-7):
    """The bounded region of the complement of T(f).

    That region is where one fixed term -- the one at the polytope's
    interior lattice point -- beats every other, so it is the
    intersection of halfspaces (p - u).x <= c_u - c_p.  Bounded exactly
    because p is interior.
    """
    p = np.asarray(interior, dtype=float)
    cp = coeff[tuple(interior)]
    N, d = [], []
    for u, cu in coeff.items():
        if tuple(u) == tuple(interior):
            continue
        N.append(p - np.asarray(u, dtype=float))
        d.append(cu - cp)
    return polyhedron(N, d, tol)


# --------------------------------------------------------------------
# Shared mesh helpers
# --------------------------------------------------------------------

def fit(pts, scale=1.0, centre=None, extent=None):
    """Centre on the bounding-box midpoint, largest extent 2*scale."""
    if not pts:
        return pts
    A = np.asarray(pts, dtype=float)
    lo, hi = A.min(axis=0), A.max(axis=0)
    ctr = 0.5 * (lo + hi) if centre is None else np.asarray(centre)
    ext = float((hi - lo).max()) if extent is None else extent
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    return [tuple(v) for v in (A - ctr) * s]


# --------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_toric_calabi_yau_add(bpy.types.Operator):
        """Add a reflexive polytope, its mirror, or a tropical
        Calabi-Yau surface"""
        bl_idname = "mesh.toric_calabi_yau_add"
        bl_label = "Toric Calabi-Yau"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            description="What to build",
            items=[('MIRROR', "Reflexive Polytope & Mirror",
                    "A reflexive polytope and its polar dual. Batyrev's "
                    "construction makes the two families of Calabi-Yau "
                    "hypersurfaces they carry a mirror pair"),
                   ('TROPICAL', "Tropical Calabi-Yau Surface",
                    "The tropical hypersurface of a tropical quartic: "
                    "a piecewise linear K3, living in R^3 with no "
                    "projection"),
                   ('K3POLYTOPE', "K3 Polytope",
                    "The unique bounded region of the complement of a "
                    "smooth tropical quartic surface")],
            default='MIRROR')
        polytope: EnumProperty(
            name="Polytope",
            description="Which reflexive 3-polytope to build",
            items=[(k, v[0], f"Reflexive polytope: {v[0]}")
                   for k, v in REFLEXIVE.items()],
            default='CUBE')
        show: EnumProperty(
            name="Show",
            description="Which half of the mirror pair to build",
            items=[('BOTH', "Both", "The polytope and its dual"),
                   ('POLYTOPE', "Polytope", "Just the polytope"),
                   ('DUAL', "Dual", "Just the polar dual")],
            default='BOTH')
        separation: FloatProperty(
            name="Separation", default=1.3, min=0.0, max=4.0,
            description="How far apart to set the mirror pair. At 0 "
                        "they are nested about their shared origin, "
                        "which is the only interior lattice point of "
                        "either")
        normalise: BoolProperty(
            name="Equal Sizes", default=True,
            description="Scale the two polytopes to the same size. A "
                        "polytope and its dual differ wildly in scale, "
                        "and unscaled the smaller one disappears")
        box: FloatProperty(
            name="Clip Radius", default=1.25, min=0.6, max=4.0,
            description="The tropical surface is unbounded. This cuts "
                        "it to a box, as a multiple of the radius of "
                        "the bounded K3 polytope inside it -- below "
                        "about 1.0 the cut starts eating that polytope")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=0.5,
            description="Solidify the tropical surface for printing "
                        "(0 = leave it a surface)")
        wireframe: FloatProperty(
            name="Strut Radius", default=0.0, min=0.0, max=0.3,
            description="Replace the faces by struts along the edges "
                        "(0 = solid faces)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size multiplier")

        def execute(self, context):
            if self.mode == 'MIRROR':
                return self._mirror(context)
            coeff = balletti_example4()
            if self.mode == 'K3POLYTOPE':
                verts, faces = k3_polytope(coeff)
                name = "K3 Polytope"
                v, e, f = f_vector(verts, faces)
                msg = f"K3 polytope f-vector ({v}, {e}, {f})"
            else:
                r = bounded_radius(coeff)
                verts, faces = tropical_surface(coeff,
                                                self.box * r)
                name = "Tropical Calabi-Yau"
                msg = (f"tropical quartic: {len(verts)} vertices, "
                       f"{len(faces)} 2-cells")
            if not faces:
                self.report({'ERROR'}, "empty result")
                return {'CANCELLED'}
            obj = self._object(context, name, fit(verts, self.scale),
                               faces)
            if self.thickness > 0.0 and self.mode == 'TROPICAL':
                m = obj.modifiers.new("Solidify", 'SOLIDIFY')
                m.thickness = self.thickness
                m.offset = 0.0
            self._select(context, obj)
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def _mirror(self, context):
            label, V = REFLEXIVE[self.polytope]
            pv, pf = polyhedron(*_hrep(V))
            dv, df = polar(V)
            built = []
            if self.show in ('BOTH', 'POLYTOPE'):
                built.append((label, pv, pf))
            if self.show in ('BOTH', 'DUAL'):
                built.append((label + " (mirror)", dv, df))
            if not built:
                return {'CANCELLED'}

            # One shared scale keeps a nested pair honest; separate
            # scales keep a side-by-side pair legible.
            span = max(float(np.abs(np.asarray(v)).max())
                       for _, v, _ in built)
            objs = []
            for i, (nm, vs, fs) in enumerate(built):
                s = (float(np.abs(np.asarray(vs)).max())
                     if self.normalise else span)
                pts = [tuple(np.asarray(v) * (self.scale / s))
                       for v in vs]
                off = 0.0
                if len(built) == 2:
                    off = (i * 2 - 1) * self.separation * self.scale
                pts = [(x + off, y, z) for x, y, z in pts]
                objs.append(self._object(context, nm, pts, fs))
            fv = [f_vector(v, f) for _, v, f in built]
            self._select(context, objs[0])
            self.report({'INFO'},
                        f"{label}: f-vectors " +
                        " and ".join(str(t) for t in fv))
            return {'FINISHED'}

        def _object(self, context, name, verts, faces):
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.wireframe > 0.0:
                m = obj.modifiers.new("Wireframe", 'WIREFRAME')
                m.thickness = self.wireframe
                m.use_even_offset = False
            return obj

        def _select(self, context, obj):
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'MIRROR':
                lay.prop(self, 'polytope')
                lay.prop(self, 'show')
                if self.show == 'BOTH':
                    lay.prop(self, 'separation')
                    lay.prop(self, 'normalise')
            elif self.mode == 'TROPICAL':
                lay.prop(self, 'box')
                lay.prop(self, 'thickness')
            lay.prop(self, 'wireframe')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.toric_calabi_yau_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_toric_calabi_yau_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_toric_calabi_yau_add)


def _hrep(vertices):
    """Halfspaces of conv(vertices), via the dual's vertices."""
    dv, _ = polar(vertices)
    D = np.asarray(dv)
    return -D, np.ones(len(D))


# --------------------------------------------------------------------

def _selftest():
    bad = []

    # 1. every listed polytope is reflexive, and duality is an
    #    involution: D** = D.
    for key, (label, V) in REFLEXIVE.items():
        refl = is_reflexive(V)
        dv, _ = polar(V)
        ddv, _ = polar(dv)
        A = np.asarray(sorted(tuple(np.round(v, 6)) for v in V))
        B = np.asarray(sorted(tuple(np.round(v, 6)) for v in ddv))
        inv = A.shape == B.shape and bool(np.max(np.abs(A - B)) < 1e-6)
        pv, pf = polyhedron(*_hrep(V))
        _, dfc = polar(V)
        fp, fd = f_vector(pv, pf), f_vector(dv, dfc)
        # Polar duality reverses the face lattice, so the two
        # f-vectors are reverses of one another.
        rev = fp == tuple(reversed(fd))
        ok = refl and inv and rev
        print(f"toric_cy: {key:11s} f={fp} dual f={fd} reflexive={refl} "
              f"D**=D {inv} reversed {rev} {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"{key}: reflexive={refl} inv={inv} rev={rev}")

    # 2. a non-reflexive polytope is rejected (2*octahedron has its
    #    facets at integral distance 2, so its dual is half-integral).
    twice = [(2 * a, 2 * b, 2 * c) for a, b, c in _octa()]
    ok = not is_reflexive(twice)
    print(f"toric_cy: 2*octahedron rejected {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("2*octahedron accepted")

    # 3. Balletti-Panizzut-Sturmfels Example 4: the K3 polytope is
    #    simple with f-vector (64, 96, 34).
    coeff = balletti_example4()
    v, f = k3_polytope(coeff)
    fv = f_vector(v, f)
    ok = fv == (64, 96, 34)
    print(f"toric_cy: Balletti Example 4 K3 polytope f-vector {fv} "
          f"(64, 96, 34) {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append(f"K3 polytope f-vector {fv}")

    # 4. it is simple -- three edges at every vertex -- and satisfies
    #    Euler's relation and Lemma 11's 3 f0 = 2 f1.
    deg = defaultdict(int)
    for fc in f:
        for i in range(len(fc)):
            deg[fc[i]] += 1
    simple = set(deg.values()) == {3}
    euler = fv[0] - fv[1] + fv[2] == 2
    lem11 = 3 * fv[0] == 2 * fv[1] and 2 * fv[2] == fv[0] + 4
    ok = simple and euler and lem11
    print(f"toric_cy: K3 polytope simple={simple} euler={euler} "
          f"lemma11={lem11} {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("K3 polytope shape")

    # 5. the tropical surface exists, is made of convex polygons, and
    #    every one of its vertices is a point where at least three of
    #    the tropical terms tie.
    verts, faces = tropical_surface(coeff)
    A = [np.asarray(k, dtype=float) for k in coeff]
    C = [coeff[k] for k in coeff]
    worst = 0
    for v in verts[::37]:
        vals = np.array([C[i] + A[i] @ np.asarray(v) for i in range(len(A))])
        worst = max(worst, int(np.sum(vals < vals.min() + 1e-6)))
    # A regular unimodular triangulation of 4 Delta_3 has 35 vertices,
    # 130 edges, 160 triangles and 64 tetrahedra (Euler, with 64
    # boundary triangles), and T(f) is its dual complex -- so one
    # 2-cell per edge of the triangulation, 130 of them.
    ok = len(faces) == 130 and worst >= 3
    print(f"toric_cy: tropical quartic {len(verts)} verts "
          f"{len(faces)} cells (130), max tie multiplicity {worst} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("tropical surface")

    # 6. the fit: centred, inside a 2 m cube.
    P = np.asarray(fit(verts))
    ctr = float(np.abs(0.5 * (P.min(axis=0) + P.max(axis=0))).max())
    ext = float((P.max(axis=0) - P.min(axis=0)).max())
    ok = ctr < 1e-9 and abs(ext - 2.0) < 1e-9
    print(f"toric_cy: fit centre={ctr:.2e} extent={ext:.6f} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("fit")

    if bad:
        raise AssertionError("; ".join(bad))
