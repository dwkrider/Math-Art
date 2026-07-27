
# Link / Connect Sum Generator for Blender
#
# Multi-component links and knot connect sums as rope curves, after
# the knot chapter of Henry Segerman's "Visualizing Mathematics with
# 3D Printing" (figs 5-7, 5-10, 5-12):
#
#   Hopf link          two round circles through each other's centers
#   Solomon's link     the (2,4) torus link 4^2_1, linking number 2
#   Whitehead link     closure of (s1 s2^-1)^2 s1, rope-relaxed
#   Borromean rings    three mutually perpendicular a x b ellipses
#   Borromean crest    the classic three-circle emblem, circles woven
#                      over/under by a smooth z-perturbation
#   (p,q) torus link   gcd(p,q) components on a common torus
#   chain              a row of interlocked rings
#   connect sum        two prime knots cut open at their outermost
#                      points, spliced, and rope-relaxed (square vs
#                      granny via mirroring the second summand)
#
# Reuses the braid-word, resampling and rope-relaxation machinery of
# prime_knot_generator. Braid closures with several components are
# split into one loop per component and relaxed jointly with
# cross-component repulsion, so linked components stay clear of each
# other and cannot pass through (which would unlink them).

bl_info = {
    "name": "Links & Connect Sums",
    "author": "Math Art project (after Henry Segerman's "
              "'Visualizing Mathematics with 3D Printing')",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Link / Connect Sum",
    "description": "Multi-component links (Hopf, Solomon, Whitehead, "
                   "Borromean, torus links, chains) and prime-knot "
                   "connect sums as rope curves",
    "category": "Add Curve",
}

import math
from math import gcd

try:                                    # sibling module (extension)
    from . import prime_knot_generator as pk
except ImportError:                     # standalone install / script
    import prime_knot_generator as pk

# (s1 s2^-1)^2 s1: a connected reduced alternating 5-crossing closed
# braid with 2 components and linking number 0 -- the only such link
# is the Whitehead link 5^2_1.  (Cf. figure-eight = (s1 s2^-1)^2 and
# Borromean rings = (s1 s2^-1)^3.)
WHITEHEAD_BRAID = "aBaBa"


# ---- generic machinery -------------------------------------------------

def braid_closure_loops(word, ring=1.6, spacing=0.3, bump=0.26):
    """Closed 3D polylines of a braid closure, one per link
    component (multi-component generalization of
    prime_knot_generator.braid_closure_points)."""
    n = max(abs(g) for g in word) + 1
    L = len(word)

    def pt(a, lev, z=0.0):
        r = ring + (lev - (n - 1) / 2.0) * spacing
        return (r * math.cos(a), r * math.sin(a), z)

    paths = [[] for _ in range(n)]        # per strand id
    strands = list(range(n))              # level -> strand id
    for lev in range(n):
        paths[lev].append(pt(0.0, lev))
    for s, g in enumerate(word):
        a0 = 2 * math.pi * s / L
        a1 = 2 * math.pi * (s + 1) / L
        am = (a0 + a1) / 2
        i = abs(g) - 1
        for lev in range(n):
            if lev != i and lev != i + 1:
                paths[strands[lev]].append(pt(a1, lev))
        z = bump if g > 0 else -bump
        sid_i, sid_j = strands[i], strands[i + 1]
        paths[sid_i].append(pt(am, i + 0.5, z))
        paths[sid_i].append(pt(a1, i + 1))
        paths[sid_j].append(pt(am, i + 0.5, -z))
        paths[sid_j].append(pt(a1, i))
        strands[i], strands[i + 1] = sid_j, sid_i
    end_level = {sid: lev for lev, sid in enumerate(strands)}
    loops = []
    seen = set()
    for s0 in range(n):
        if s0 in seen:
            continue
        loop = [paths[s0][0]]
        sid = s0
        while True:
            seen.add(sid)
            loop.extend(paths[sid][1:])
            sid = end_level[sid]
            if sid == s0:
                break
        loop.pop()                        # closing duplicate
        loops.append(loop)
    return loops


def resample_loops(loops, samples):
    """Resample each closed loop, distributing points by arc length
    so an average component gets `samples` points."""
    import numpy as np
    lens = []
    for lp in loops:
        Q = np.asarray(lp, dtype=float)
        lens.append(np.linalg.norm(np.roll(Q, -1, 0) - Q,
                                   axis=1).sum())
    avg = sum(lens) / len(lens)
    return [pk.resample_closed(lp, max(32, int(round(samples
                                                     * ln / avg))))
            for lp, ln in zip(loops, lens)]


def relax_link(comps, iters=100, repel=0.35, step=0.25, smooth=0.5,
               excl=6):
    """Multi-component version of prime_knot_generator.relax_knot:
    per-component curve smoothing plus repulsion between all strand
    pairs (both self and cross-component), globally renormalized.
    Repulsion never lets strands cross, so linked components stay
    linked; being linked, they also cannot drift apart."""
    import numpy as np
    comps = [np.asarray(c, dtype=float).copy() for c in comps]
    if iters <= 0:
        return comps
    sizes = [len(c) for c in comps]
    off = [0]
    for s in sizes:
        off.append(off[-1] + s)
    P = np.concatenate(comps)
    cid = np.concatenate([np.full(s, k, dtype=int)
                          for k, s in enumerate(sizes)])
    lidx = np.concatenate([np.arange(s) for s in sizes])
    same = cid[:, None] == cid[None, :]
    ring = np.abs(lidx[:, None] - lidx[None, :])
    msz = np.array(sizes)[cid]
    ring = np.minimum(ring, msz[:, None] - ring)
    near = same & (ring <= excl)          # neighbors along a strand
    target = np.linalg.norm(P - P.mean(0), axis=1).mean()
    for _ in range(iters):
        for k in range(len(sizes)):
            seg = P[off[k]:off[k + 1]]
            Pn = (np.roll(seg, 1, 0) + np.roll(seg, -1, 0)) / 2
            P[off[k]:off[k + 1]] = seg + smooth * (Pn - seg)
        D = P[:, None, :] - P[None, :, :]
        d = np.linalg.norm(D, axis=-1)
        np.fill_diagonal(d, 1.0)
        mask = (d < repel) & ~near
        if mask.any():
            w = np.where(mask, (repel - d) / d, 0.0)
            P += step * (D * w[:, :, None]).sum(axis=1)
        P -= P.mean(0)
        r = np.linalg.norm(P, axis=1).mean()
        if r > 1e-9:
            P *= target / r
    return [P[off[k]:off[k + 1]].copy() for k in range(len(sizes))]


def linking_number(A, B):
    """Gauss linking integral of two closed polylines (midpoint
    rule); rounds to +-lk for reasonably sampled curves."""
    import numpy as np
    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    da = np.roll(a, -1, 0) - a
    db = np.roll(b, -1, 0) - b
    R = (a + 0.5 * da)[:, None, :] - (b + 0.5 * db)[None, :, :]
    r3 = ((R * R).sum(-1)) ** 1.5
    cr = np.cross(da[:, None, :], db[None, :, :])
    return float(((cr * R).sum(-1) / r3).sum() / (4 * math.pi))


# ---- preset geometry ---------------------------------------------------

def _circle(center, u, v, radius, samples):
    import numpy as np
    t = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    c = np.asarray(center, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return (c + radius * (np.cos(t)[:, None] * u
                          + np.sin(t)[:, None] * v))


def hopf_components(samples):
    """Two unit circles, each through the other's center."""
    return [_circle((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, samples),
            _circle((1, 0, 0), (1, 0, 0), (0, 0, 1), 1.0, samples)]


def torus_link_components(p, q, samples, R=1.25, r=0.55):
    """(p,q) torus link: gcd(p,q) components, each a (p/g, q/g)-type
    curve, phase-offset by 2*pi*k/p in the tube angle (the distinct
    cosets k mod g give the g disjoint components)."""
    import numpy as np
    g = gcd(p, q)
    s = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    comps = []
    for k in range(g):
        th = (p // g) * s
        ph = (q // g) * s + 2 * math.pi * k / p
        rad = R + r * np.cos(ph)
        comps.append(np.stack([rad * np.cos(th), rad * np.sin(th),
                               r * np.sin(ph)], axis=1))
    return comps


def whitehead_components(samples, iters=100, repel=0.35):
    """Whitehead link from its minimum braid closure, rope-relaxed."""
    loops = braid_closure_loops(pk.parse_letters(WHITEHEAD_BRAID))
    return relax_link(resample_loops(loops, samples),
                      iters=iters, repel=repel)


def borromean_ellipses(samples, ratio=1.618, a=1.0):
    """Three mutually perpendicular a x b ellipses, cyclically
    oriented; pairwise unlinked but collectively inseparable for any
    a/b > 1 (min pairwise clearance ~ a - b)."""
    import numpy as np
    b = a / ratio
    t = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    ca, sb = a * np.cos(t), b * np.sin(t)
    z = np.zeros(samples)
    return [np.stack([ca, sb, z], axis=1),      # xy-plane
            np.stack([z, ca, sb], axis=1),      # yz-plane
            np.stack([sb, z, ca], axis=1)]      # zx-plane


def borromean_crest(samples, depth=0.35, radius=1.0, spread=0.55):
    """Classic three-circle crest: equal circles centered at 120
    degree offsets, woven with the cyclic rule 'circle i passes over
    circle i+1' -- smooth +-z Gaussian bumps at the planar crossing
    angles.  Each pair crosses over/over, hence pairwise unlinked,
    yet the whole is the Borromean rings."""
    import numpy as np
    betas = [math.pi / 2 + 2 * math.pi * k / 3 for k in range(3)]
    centers = [np.array([spread * math.cos(b),
                         spread * math.sin(b), 0.0]) for b in betas]
    cross = [[] for _ in range(3)]        # per circle: (angle, other)
    for i in range(3):
        for j in range(i + 1, 3):
            dvec = centers[j] - centers[i]
            D = float(np.linalg.norm(dvec))
            if D >= 2 * radius:
                raise ValueError("crest circles do not overlap")
            mid = (centers[i] + centers[j]) / 2
            h = math.sqrt(radius * radius - (D / 2) ** 2)
            perp = np.array([-dvec[1], dvec[0], 0.0]) / D
            for sgn in (1.0, -1.0):
                p = mid + sgn * h * perp
                for a, c in ((i, centers[i]), (j, centers[j])):
                    ang = math.atan2(p[1] - c[1], p[0] - c[0])
                    cross[a].append((ang % (2 * math.pi),
                                     j if a == i else i))
    t = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    comps = []
    for i in range(3):
        cs = sorted(cross[i])
        angs = np.array([c[0] for c in cs])
        signs = np.array([1.0 if c[1] == (i + 1) % 3 else -1.0
                          for c in cs])
        gaps = np.diff(np.concatenate([angs,
                                       [angs[0] + 2 * math.pi]]))
        sigma = gaps.min() / 3.0
        z = np.zeros(samples)
        for ang, s in zip(angs, signs):
            d = np.angle(np.exp(1j * (t - ang)))
            z += s * np.exp(-(d / sigma) ** 2)
        comps.append(np.stack(
            [centers[i][0] + radius * np.cos(t),
             centers[i][1] + radius * np.sin(t),
             depth * z], axis=1))
    return comps


def chain_components(n, samples, spacing=1.3, radius=1.0):
    """N interlocked rings in a row, alternating between the xy and
    xz planes like a physical chain."""
    comps = []
    x0 = -(n - 1) * spacing / 2.0
    for i in range(n):
        c = (x0 + i * spacing, 0.0, 0.0)
        v = (0, 1, 0) if i % 2 == 0 else (0, 0, 1)
        comps.append(_circle(c, (1, 0, 0), v, radius, samples))
    return comps


def _open_loop(P, w):
    """Center a closed loop, cut out a w-point window around its
    outermost (max-x) point, return the open polyline."""
    import numpy as np
    P = np.asarray(P, dtype=float)
    P = P - P.mean(0)
    i = int(np.argmax(P[:, 0]))
    P = np.roll(P, -(i - w // 2), axis=0)  # window sits at 0..w-1
    return P[w:]


def connect_sum(braid_a, braid_b, samples=160, iters=120,
                repel=0.35, mirror_b=False, gap=0.6):
    """Connect sum of two prime knots: each summand is built and
    relaxed by prime_knot_generator, cut open at its outermost
    point, placed side by side with the openings facing, spliced by
    two short bridges, and rope-relaxed into one closed rope.
    mirror_b mirrors the second summand (square vs granny)."""
    import numpy as np
    Pa = pk.build_knot(braid_a, samples=samples, iters=iters,
                       repel=repel)
    Pb = pk.build_knot(braid_b, samples=samples, iters=iters,
                       repel=repel, mirror=mirror_b)
    w = max(3, samples // 20)
    A = _open_loop(Pa, w)
    B = _open_loop(Pb, w) * np.array([-1.0, -1.0, 1.0])  # 180deg
    ds = np.linalg.norm(np.diff(A, axis=0), axis=1).mean()
    A = A + np.array([-A[:, 0].max() - gap / 2, 0.0, 0.0])
    B = B + np.array([-B[:, 0].min() + gap / 2, 0.0, 0.0])
    if (np.linalg.norm(A[-1] - B[0])
            > np.linalg.norm(A[-1] - B[-1])):
        B = B[::-1]

    def bridge(pnt, qnt):
        k = max(1, int(np.linalg.norm(qnt - pnt) / ds))
        ts = np.linspace(0.0, 1.0, k + 2)[1:-1, None]
        return pnt + ts * (qnt - pnt)

    loop = np.concatenate([A, bridge(A[-1], B[0]),
                           B, bridge(B[-1], A[0])])
    loop = pk.resample_closed(loop, 2 * samples)
    return [pk.relax_knot(loop, iters=iters, repel=repel)]


# ---- Blender layer -----------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _KNOT_ITEMS = [(name, name.replace('_', '.'),
                    f"{name.split('_')[0]}-crossing prime knot, "
                    f"minimum braid {braid}")
                   for (name, braid, _ap) in pk.KNOTS]
    _BRAIDS = {name: braid for (name, braid, _ap) in pk.KNOTS}

    _PRESETS = [
        ('HOPF', "Hopf Link",
         "Two circles through each other's centers"),
        ('SOLOMON', "Solomon's Link",
         "The (2,4) torus link 4^2_1, linking number 2"),
        ('WHITEHEAD', "Whitehead Link",
         "Closure of (s1 s2^-1)^2 s1, rope-relaxed; linking "
         "number 0"),
        ('BORROMEAN', "Borromean Rings",
         "Three mutually perpendicular ellipses; pairwise unlinked "
         "yet inseparable"),
        ('BORROMEAN_CREST', "Borromean Crest",
         "Three equal circles at 120 degrees woven over/under"),
        ('TORUS', "Torus Link (p,q)",
         "gcd(p,q) components on a common torus"),
        ('CHAIN', "Chain",
         "A row of interlocked rings"),
        ('CONNECT_SUM', "Connect Sum",
         "Two prime knots spliced into one rope (square vs granny "
         "via mirror)"),
    ]

    _RELAX_PRESETS = {'WHITEHEAD', 'BORROMEAN_CREST', 'CONNECT_SUM'}

    class CURVE_OT_math_link_add(bpy.types.Operator):
        """Add a multi-component link or a prime-knot connect sum
        as rope curves (one spline per component)"""
        bl_idname = "curve.math_link_add"
        bl_label = "Link / Connect Sum"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(name="Preset", items=_PRESETS,
                             default='HOPF')
        p: IntProperty(name="p", default=2, min=1, max=16,
                       description="Windings around the torus axis")
        q: IntProperty(name="q", default=4, min=1, max=32,
                       description="Windings around the torus tube")
        links: IntProperty(name="Rings", default=5, min=2, max=50,
                           description="Rings in the chain")
        knot_a: EnumProperty(name="Knot A", items=_KNOT_ITEMS,
                             default='3_1')
        knot_b: EnumProperty(name="Knot B", items=_KNOT_ITEMS,
                             default='3_1')
        mirror_b: BoolProperty(
            name="Mirror Second Knot", default=False,
            description="Mirror the second summand (e.g. square "
                        "knot instead of granny knot)")
        axis_ratio: FloatProperty(
            name="Axis Ratio a/b", default=1.618, min=1.05, max=4.0,
            description="Ellipse semi-axis ratio; larger gives more "
                        "clearance between the rings")
        crest_depth: FloatProperty(
            name="Weave Depth", default=0.35, min=0.05, max=1.0,
            description="Height of the over/under weave bumps")
        samples: IntProperty(name="Samples / Component", default=160,
                             min=40, max=1000)
        iters: IntProperty(
            name="Relax Iterations", default=100, min=0, max=600,
            description="Smoothing + repulsion rounds for the "
                        "presets that are rope-relaxed (0 shows the "
                        "raw construction)")
        repel: FloatProperty(
            name="Strand Clearance", default=0.35, min=0.05, max=1.0,
            description="Repulsion distance while relaxing")
        color_components: BoolProperty(
            name="Color Components", default=True,
            description="One material with a distinct color per "
                        "link component")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='BEZIER')
        radius: FloatProperty(
            name="Rope Radius", default=0.08, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius")
        resolution: IntProperty(name="Bevel Resolution", default=6,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def _build(self):
            s = self.samples
            pr = self.preset
            if pr == 'HOPF':
                return hopf_components(s), "Hopf Link"
            if pr == 'SOLOMON':
                return torus_link_components(2, 4, s), "Solomon Link"
            if pr == 'WHITEHEAD':
                return (whitehead_components(s, self.iters,
                                             self.repel),
                        "Whitehead Link")
            if pr == 'BORROMEAN':
                return (borromean_ellipses(s, self.axis_ratio),
                        "Borromean Rings")
            if pr == 'BORROMEAN_CREST':
                comps = borromean_crest(s, self.crest_depth)
                if self.iters:
                    comps = relax_link(comps, iters=self.iters,
                                       repel=self.repel)
                return comps, "Borromean Crest"
            if pr == 'TORUS':
                return (torus_link_components(self.p, self.q, s),
                        f"Torus Link ({self.p},{self.q})")
            if pr == 'CHAIN':
                return (chain_components(self.links, s),
                        f"Chain {self.links}")
            comps = connect_sum(_BRAIDS[self.knot_a],
                                _BRAIDS[self.knot_b],
                                samples=s, iters=self.iters,
                                repel=self.repel,
                                mirror_b=self.mirror_b)
            la = self.knot_a.replace('_', '.')
            lb = self.knot_b.replace('_', '.')
            tag = "m" if self.mirror_b else ""
            return comps, f"Knot {la} # {lb}{tag}"

        def execute(self, context):
            import numpy as np
            try:
                comps, name = self._build()
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            comps = [np.asarray(c, dtype=float) for c in comps]
            # Center all components on the origin and uniformly scale
            # so the combined bounding box spans ~2 m (times Scale),
            # using ONE bbox shared by every component.
            pts = np.concatenate(comps) if comps else np.zeros((0, 3))
            if len(pts):
                lo = pts.min(0)
                hi = pts.max(0)
                cen = (lo + hi) / 2.0
                ext = float((hi - lo).max())
                s = (2.0 * self.scale / ext) if ext > 1e-9 else 1.0
                comps = [(c - cen) * s for c in comps]
            if self.output == 'MESH':
                obj = self._make_mesh(name, comps)
            else:
                obj = self._make_curve(name, comps)
            if self.color_components:
                self._add_materials(obj, len(comps), name)
                if self.output != 'MESH':
                    # spline.material_index is CLAMPED to the
                    # material count at assignment time, so the
                    # values set while the curve had no materials
                    # all collapsed to 0 -- set them again now
                    for k, sp in enumerate(obj.data.splines):
                        sp.material_index = k
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {len(comps)} component(s)")
            return {'FINISHED'}

        def _make_curve(self, name, comps):
            cu = bpy.data.curves.new(name, 'CURVE')
            cu.dimensions = '3D'
            for k, P in enumerate(comps):
                if self.output == 'BEZIER':
                    sp = cu.splines.new('BEZIER')
                    sp.bezier_points.add(len(P) - 1)
                    for i, pnt in enumerate(P):
                        bp = sp.bezier_points[i]
                        bp.co = pnt
                        bp.handle_left_type = 'AUTO'
                        bp.handle_right_type = 'AUTO'
                else:
                    sp = cu.splines.new(self.output)
                    sp.points.add(len(P) - 1)
                    for i, pnt in enumerate(P):
                        sp.points[i].co = (pnt[0], pnt[1], pnt[2],
                                           1.0)
                    if self.output == 'NURBS':
                        sp.order_u = 4
                sp.use_cyclic_u = True
                sp.material_index = k
            cu.bevel_depth = self.radius
            cu.bevel_resolution = self.resolution
            return bpy.data.objects.new(name, cu)

        def _make_mesh(self, name, comps):
            import numpy as np
            tube = pk.CURVE_OT_prime_knot_add._tube
            verts, faces, midx = [], [], []
            for k, P in enumerate(comps):
                v, f = tube(np.asarray(P), self.radius,
                            self.tube_sides)
                base = len(verts)
                verts.extend(v)
                faces.extend([[base + i for i in face]
                              for face in f])
                midx.extend([k] * len(f))
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            if len(me.polygons) == len(midx):
                me.polygons.foreach_set('material_index', midx)
            me.update()
            return bpy.data.objects.new(name, me)

        @staticmethod
        def _add_materials(obj, ncomp, name):
            import colorsys
            for k in range(ncomp):
                rgb = colorsys.hsv_to_rgb(k / max(ncomp, 1),
                                          0.72, 0.9)
                mat = bpy.data.materials.new(f"{name} C{k + 1}")
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                node = mat.node_tree.nodes.get("Principled BSDF")
                if node:
                    node.inputs["Base Color"].default_value = \
                        (*rgb, 1.0)
                obj.data.materials.append(mat)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            pr = self.preset
            if pr == 'TORUS':
                lay.prop(self, 'p')
                lay.prop(self, 'q')
            elif pr == 'CHAIN':
                lay.prop(self, 'links')
            elif pr == 'CONNECT_SUM':
                lay.prop(self, 'knot_a')
                lay.prop(self, 'knot_b')
                lay.prop(self, 'mirror_b')
            elif pr == 'BORROMEAN':
                lay.prop(self, 'axis_ratio')
            elif pr == 'BORROMEAN_CREST':
                lay.prop(self, 'crest_depth')
            if pr in _RELAX_PRESETS:
                lay.prop(self, 'iters')
                lay.prop(self, 'repel')
            lay.prop(self, 'samples')
            lay.prop(self, 'color_components')
            lay.prop(self, 'output')
            lay.prop(self, 'radius')
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.math_link_add", icon='LINKED')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_math_link_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_math_link_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        word = pk.parse_letters(WHITEHEAD_BRAID)
        nc = pk.closure_components(word)
        print(f"Whitehead braid {WHITEHEAD_BRAID}: "
              f"{nc} components", "OK" if nc == 2 else "FAIL")
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            h = hopf_components(120)
            print("Hopf lk =", round(linking_number(*h), 3))
            s = torus_link_components(2, 4, 160)
            print("Solomon lk =", round(linking_number(*s), 3))
            b = borromean_ellipses(120)
            print("Borromean pairwise lk =",
                  [round(linking_number(b[i], b[j]), 3)
                   for i in range(3) for j in range(i + 1, 3)])
            c = borromean_crest(120)
            print("Crest pairwise lk =",
                  [round(linking_number(c[i], c[j]), 3)
                   for i in range(3) for j in range(i + 1, 3)])
            cs = connect_sum(pk.KNOTS[0][1], pk.KNOTS[0][1],
                             samples=120, iters=40)
            print("Connect sum: 1 loop of", len(cs[0]), "points")
