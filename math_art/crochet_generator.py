
# Crochet (Hyperbolic Plane) Generator for Blender
#
# The crocheted hyperbolic plane, after Daina Taimina: crochet
# increases stitches at a constant ratio (one increase every N
# stitches), so each row's circumference grows exponentially and the
# fabric -- unable to lie flat (Hilbert's theorem) -- buckles into
# self-similar ruffles. The increase ratio N sets the curvature radius
# R = h / ln(1 + 1/N) ~ N*h: small N crochets a tightly folded, very
# "bendy" plane; large N a gently wavy one.
#
# The mesh is the actual crochet: each row's stitch count grows with
# the hyperbolic circumference 2*pi*R*sinh(rho/R), every stitch the
# same size. Each ring is pre-seeded with a phase-coherent period-
# doubling cascade of ruffles (wavenumbers m0, 2*m0, 4*m0, ... switched
# on as the excess length grows) whose doubling radii are the surface's
# "distributed branch points" -- the source of the self-similar,
# kale/lettuce morphology. The sheet is then relaxed (position-based
# stitch-length + bending constraints, self-repulsion, and a fast
# BVHTree vertex-triangle self-collision) into its buckled shape,
# centred and fit to a 2 m cube.
#
# References:
# - Daina Taimina, "Crocheting Adventures with Hyperbolic Planes",
#   A K Peters, 2009; David W. Henderson and Daina Taimina,
#   "Crocheting the hyperbolic plane", Math. Intelligencer 23(2),
#   2001, pp. 17-28.
# - Self-similar buckling / distributed branch points of constant-
#   negative-curvature elastic sheets: John A. Gemmer, E. Sharon,
#   S. C. Venkataramani et al., "Distributed branch points and the
#   shape of elastic surfaces with constant negative curvature",
#   arXiv:2006.14461.
# - Hilbert's theorem (no complete C^2 isometric immersion of H^2 in
#   R^3): D. Hilbert, Trans. AMS 2, 1901, pp. 87-99.

bl_info = {
    "name": "Crochet",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Crochet",
    "description": "Ruffled crocheted hyperbolic planes, from gently "
                   "wavy to tightly folded",
    "category": "Add Mesh",
}

import math

import numpy as np


def _center(V):
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    return (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)


def _join_rings(sa, na, sb, nb):
    """Bridge two concentric rings of different vertex counts with a
    triangle strip, advancing whichever ring is 'behind' in angle."""
    faces = []
    ia = ib = 0
    while ia < na or ib < nb:
        a0 = sa + ia % na
        b0 = sb + ib % nb
        if ib >= nb or (ia < na and (ia + 1) / na <= (ib + 1) / nb):
            faces.append((a0, b0, sa + (ia + 1) % na))
            ia += 1
        else:
            faces.append((a0, b0, sb + (ib + 1) % nb))
            ib += 1
    return faces


def _crochet_mesh(ratio_n, rows, stitch, max_stitches, seed_scale=1.0):
    """Build the crochet mesh: exponentially growing stitch count per
    row (constant stitch size h), cascade-seeded ruffles, and true
    hyperbolic edge rest lengths."""
    h = stitch
    R = h / math.log(1.0 + 1.0 / ratio_n)
    rng = np.random.default_rng(0)
    m0 = 3                                        # cascade base
    P, faces, rings, prev = [], [], [], None
    for i in range(rows):
        rho = (i + 1) * h
        circ = 2.0 * math.pi * R * math.sinh(rho / R)
        n = int(max(6, min(max_stitches, round(circ / h))))
        start = len(P)
        theta = 2.0 * math.pi * np.arange(n) / n
        excess = circ / (2.0 * math.pi * rho)     # > 1
        z = np.zeros(n)
        if excess > 1.005:
            # arclength-matching target wavenumber (small-slope):
            # m_t = (2 rho / a) sqrt(excess - 1)
            m_t = (2.0 * rho / (0.7 * h)) * math.sqrt(excess - 1.0)
            ks = []
            k = 0
            while m0 * 2 ** k <= max(m_t, m0):
                ks.append(k)
                k += 1
            for k in ks:
                mk = m0 * 2 ** k
                if mk > n // 2 - 1:
                    break
                ak = 0.7 * h if k == ks[-1] else 0.35 * h
                z = z + ak * np.sin(mk * theta)
        z = seed_scale * z + 0.03 * rho + rng.normal(0.0, 0.02 * h, n)
        for j in range(n):
            P.append([rho * math.cos(theta[j]),
                      rho * math.sin(theta[j]), float(z[j])])
        if prev is not None:
            faces += _join_rings(prev[0], prev[1], start, n)
        rings.append((start, n))
        prev = (start, n)
        if len(P) > 80000:
            break
    P = np.array(P)
    eset = set()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            eset.add((a, b) if a < b else (b, a))
    edges = list(eset)
    E0 = np.array([e[0] for e in edges])
    E1 = np.array([e[1] for e in edges])
    rr = np.linalg.norm(P[:, :2], axis=1)
    th = np.arctan2(P[:, 1], P[:, 0])
    ch = (np.cosh(rr[E0] / R) * np.cosh(rr[E1] / R)
          - np.sinh(rr[E0] / R) * np.sinh(rr[E1] / R)
          * np.cos(th[E0] - th[E1]))
    REST = R * np.arccosh(np.maximum(ch, 1.0))
    nbr = [set() for _ in range(len(P))]
    for a, b in edges:
        nbr[a].add(int(b))
        nbr[b].add(int(a))
    pin = np.arange(rings[0][1])
    tris = np.array(faces, dtype=np.int64)
    return P, E0, E1, REST, nbr, tris, pin, P[pin].copy()


def _repel(P, nbr, radius, strength):
    """Short-range self-repulsion between non-adjacent vertices via a
    uniform spatial hash."""
    from collections import defaultdict
    keys = np.floor(P / radius).astype(np.int64)
    buckets = defaultdict(list)
    for idx in range(len(P)):
        buckets[(keys[idx, 0], keys[idx, 1], keys[idx, 2])].append(idx)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)]
    dP = np.zeros_like(P)
    r2 = radius * radius
    for idx in range(len(P)):
        kx, ky, kz = keys[idx]
        ni = nbr[idx]
        pi = P[idx]
        for dx, dy, dz in offs:
            for j in buckets.get((kx + dx, ky + dy, kz + dz), ()):
                if j <= idx or j in ni:
                    continue
                d = pi - P[j]
                l2 = float(d @ d)
                if 1e-12 < l2 < r2:
                    L = math.sqrt(l2)
                    f = strength * (radius - L) / radius * (d / L)
                    dP[idx] += f
                    dP[j] -= f
    return dP


def _bvh_decollide(P, faces, nbr, thick, strength):
    """One vertex-triangle self-collision pass using Blender's C-level
    BVHTree. Vertices within `thick` of a non-adjacent triangle are
    pushed out along the separation normal (with a reaction on the
    triangle). Blender only (uses mathutils)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    verts = [Vector((float(p[0]), float(p[1]), float(p[2])))
             for p in P]
    polys = [(int(f[0]), int(f[1]), int(f[2])) for f in faces]
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=True)
    dP = np.zeros_like(P)
    for v in range(len(P)):
        nv = nbr[v]
        for loc, nrm, idx, dist in tree.find_nearest_range(verts[v],
                                                           thick):
            if dist is None or dist >= thick:
                continue
            f = polys[idx]
            if v in f or f[0] in nv or f[1] in nv or f[2] in nv:
                continue
            dvec = P[v] - np.array([loc.x, loc.y, loc.z])
            L = float(np.linalg.norm(dvec))
            direction = (dvec / L if L > 1e-9
                         else np.array([nrm.x, nrm.y, nrm.z]))
            push = strength * (thick - dist) * direction
            dP[v] += push
            dP[f[0]] -= push / 3.0
            dP[f[1]] -= push / 3.0
            dP[f[2]] -= push / 3.0
    return P + 0.5 * dP


def _relax(P, E0, E1, REST, nbr, pin, pin_pos, iters, smooth,
           repel_r, repel_s, collide_fn=None, collide_every=40,
           anneal_L0=None, anneal_frac=0.75, stiff=1.0):
    """Jacobi stitch-length constraints + Laplacian bending + periodic
    self-repulsion (and optional BVHTree collision); inner ring pinned.
    With `anneal_L0` (flat starting edge lengths) the target rest grows
    smoothly from flat to hyperbolic over `anneal_frac` of the run, so
    the sheet buckles as a continuous deformation; `stiff` damps the
    compression term to stop compressed edges flinging into spikes.
    Defaults reproduce the plain relaxation exactly."""
    n = len(P)
    valence = np.zeros(n)
    np.add.at(valence, E0, 1.0)
    np.add.at(valence, E1, 1.0)
    valence = np.maximum(valence, 1.0)[:, None]
    for it in range(iters):
        if anneal_L0 is not None:
            t = min(1.0, it / max(1.0, anneal_frac * iters))
            rest = (1.0 - t) * anneal_L0 + t * REST
        else:
            t, rest = 1.0, REST
        d = P[E1] - P[E0]
        L = np.linalg.norm(d, axis=1)
        L = np.where(L < 1e-9, 1e-9, L)
        ratio = np.clip(1.0 - rest / L, -3.0, 1.0)
        corr = (stiff * ratio)[:, None] * d
        dP = np.zeros_like(P)
        np.add.at(dP, E0, corr)
        np.add.at(dP, E1, -corr)
        P = P + dP / valence
        if smooth > 0.0:
            nsum = np.zeros_like(P)
            np.add.at(nsum, E0, P[E1])
            np.add.at(nsum, E1, P[E0])
            P = P + smooth * (nsum / valence - P)
        if t > 0.9:
            if repel_s > 0.0 and it % 3 == 0 and it > 0:
                P = P + _repel(P, nbr, repel_r, repel_s)
            if collide_fn is not None and it % collide_every == 0:
                P = collide_fn(P)
        P[pin] = pin_pos
    return P


def _pack(P, E0, E1, REST, nbr, faces, pin, pin_pos, iters, pull,
          stitch, repel_r, smooth=0.12):
    """Collision-packing: pull every free vertex weakly toward the
    centroid (a self-weight/gravity that compacts the sheet) while the
    stitch springs preserve the metric and BVHTree self-collision stops
    the folds passing through each other -- so the smooth hyperbolic
    disk folds up into a dense ball, like real crochet gathered in the
    hand. Blender only (needs the BVHTree collision)."""
    n = len(P)
    valence = np.zeros(n)
    np.add.at(valence, E0, 1.0)
    np.add.at(valence, E1, 1.0)
    valence = np.maximum(valence, 1.0)[:, None]
    free = np.ones(n, dtype=bool)
    free[pin] = False
    for it in range(iters):
        C = P[free].mean(axis=0)
        P[free] += pull * (C - P[free])           # gravity toward centre
        for _ in range(3):                        # keep stitch lengths
            d = P[E1] - P[E0]
            L = np.linalg.norm(d, axis=1)
            L = np.where(L < 1e-9, 1e-9, L)
            corr = np.clip(1.0 - REST / L, -3.0, 1.0)[:, None] * d
            dP = np.zeros_like(P)
            np.add.at(dP, E0, corr)
            np.add.at(dP, E1, -corr)
            P = P + dP / valence
            P[pin] = pin_pos
        if smooth > 0.0:                           # keep the folds smooth
            nsum = np.zeros_like(P)
            np.add.at(nsum, E0, P[E1])
            np.add.at(nsum, E1, P[E0])
            P = P + smooth * (nsum / valence - P)
        P = P + _repel(P, nbr, repel_r, 0.5)
        if _IN_BLENDER:                            # self-collision each step
            P = _bvh_decollide(P, faces, nbr, 0.9 * stitch, 0.8)
        P[pin] = pin_pos
    # final polish: smooth out collision roughness while holding the
    # stitch lengths, so the ball reads as folded fabric not crumple
    for _ in range(8):
        nsum = np.zeros_like(P)
        np.add.at(nsum, E0, P[E1])
        np.add.at(nsum, E1, P[E0])
        P = P + 0.25 * (nsum / valence - P)
        d = P[E1] - P[E0]
        L = np.linalg.norm(d, axis=1)
        L = np.where(L < 1e-9, 1e-9, L)
        corr = np.clip(1.0 - REST / L, -3.0, 1.0)[:, None] * d
        dP = np.zeros_like(P)
        np.add.at(dP, E0, corr)
        np.add.at(dP, E1, -corr)
        P = P + dP / valence
        P[pin] = pin_pos
    return P


def build_ruffle(ratio_n=4, rows=18, stitch=0.09, max_stitches=600,
                 iters=340, smooth=0.06, repel=0.5, collide=3,
                 anneal=False, stiff=1.0, pack=0, pack_pull=0.03):
    # `anneal`/`stiff` grow the curvature smoothly from flat and damp
    # the springs (needed for the tight-curvature presets to stay
    # smooth); `pack` folds the relaxed sheet into a ball via gravity +
    # self-collision. Defaults (all off) reproduce the plain build, so
    # Wavy/Ruffled are unchanged.
    P, E0, E1, REST, nbr, faces, pin, pin_pos = _crochet_mesh(
        ratio_n, rows, stitch, max_stitches,
        seed_scale=0.2 if anneal else 1.0)
    L0 = None
    if anneal:
        flat = P.copy()
        flat[:, 2] = 0.0
        L0 = np.linalg.norm(flat[E1] - flat[E0], axis=1)
    cf = None
    if collide > 0 and _IN_BLENDER:
        cf = lambda pp: _bvh_decollide(pp, faces, nbr,
                                       0.75 * stitch, 0.6)
    ce = max(15, iters // (collide * 4 + 1)) if collide > 0 else 10 ** 9
    P = _relax(P, E0, E1, REST, nbr, pin, pin_pos, iters, smooth,
               0.4 * stitch, repel, collide_fn=cf, collide_every=ce,
               anneal_L0=L0, stiff=stiff)
    if pack > 0:
        P = _pack(P, E0, E1, REST, nbr, faces, pin, pin_pos, pack,
                  pack_pull, stitch, 0.4 * stitch)
    return _center(P), faces


# named presets from gently wavy to tightly folded ("bendy")
PRESETS = {
    'WAVY': dict(ratio_n=6, rows=14, stitch=0.10, max_stitches=400,
                 iters=280, smooth=0.08, repel=0.5, collide=2),
    'RUFFLED': dict(ratio_n=4, rows=18, stitch=0.09, max_stitches=600,
                    iters=340, smooth=0.06, repel=0.5, collide=3),
    # tight-curvature presets: curvature-annealed + damped smooth base
    # (uncapped so outer rows are not truncated into spikes); Taimina
    # then collision-PACKS it into a ball.
    'BENDY': dict(ratio_n=3, rows=16, stitch=0.08, max_stitches=6000,
                  iters=380, smooth=0.14, repel=0.3, collide=0,
                  anneal=True, stiff=0.55, pack=45, pack_pull=0.010),
    'TAIMINA': dict(ratio_n=3, rows=16, stitch=0.08, max_stitches=6000,
                    iters=360, smooth=0.13, repel=0.3, collide=0,
                    anneal=True, stiff=0.55, pack=90, pack_pull=0.012),
}


def mean_curvature(V, faces):
    n = len(V)
    ang = np.zeros(n)
    area = np.zeros(n)
    deg = np.zeros(n)
    tris = [(f[0], f[1], f[2]) for f in faces]
    for a, b, c in tris:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u = V[j] - V[i]
            w = V[k] - V[i]
            cs = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)
                                 + 1e-12)
            ang[i] += math.acos(max(-1.0, min(1.0, cs)))
        ar = 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
        for i in (a, b, c):
            area[i] += ar / 3.0
            deg[i] += 1
    interior = deg >= 6
    defect = 2 * math.pi - ang
    ki = defect[interior] / np.maximum(area[interior], 1e-12)
    return float(np.median(ki)) if ki.size else 0.0


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

    class MESH_OT_crochet_add(bpy.types.Operator):
        """Add a crocheted hyperbolic plane -- a ruffled surface of
        constant negative curvature"""
        bl_idname = "mesh.crochet_add"
        bl_label = "Crochet"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            items=[('WAVY', "Wavy", "Gently wavy (loose curvature)"),
                   ('RUFFLED', "Ruffled", "Clearly ruffled"),
                   ('BENDY', "Bendy", "Tightly folded, very 3D "
                    "(slower)"),
                   ('TAIMINA', "Taimina (ball)", "Densely folded, "
                    "ball-like (slowest)"),
                   ('CUSTOM', "Custom", "Use the parameters below")],
            default='WAVY')
        # Custom defaults mirror the WAVY preset
        ratio_n: IntProperty(
            name="Increase Ratio N", default=6, min=2, max=24,
            description="Increase 1 stitch every N. Small N = tight "
                        "curvature, more bend")
        rows: IntProperty(
            name="Rows", default=14, min=3, max=30,
            description="Crochet rows; more = larger and more folded")
        stitch: FloatProperty(
            name="Stitch Size", default=0.10, min=0.02, max=0.5,
            description="Size of one (square) stitch")
        max_stitches: IntProperty(
            name="Max Stitches/Row", default=400, min=24, max=2000,
            description="Cap on the (growing) stitch count per row; "
                        "raise it at small N so the edge keeps ruffling")
        iters: IntProperty(
            name="Relax Steps", default=280, min=0, max=3000,
            description="Buckling relaxation iterations")
        smooth: FloatProperty(
            name="Bending", default=0.08, min=0.0, max=1.0,
            description="Higher = larger, smoother ruffles; lower = "
                        "finer, crisper folds. Above ~0.5 it strongly "
                        "over-smooths and can shrink the sheet")
        repel: FloatProperty(
            name="Self-Repulsion", default=0.5, min=0.0, max=1.0,
            description="Push apart ruffles that get too close")
        collide: IntProperty(
            name="Collision Passes", default=2, min=0, max=10,
            description="BVHTree vertex-triangle self-collision passes "
                        "that stop the fabric passing through itself")
        physics: EnumProperty(
            name="Fold By",
            items=[('PBD', "Built-in Packing",
                    "Fold with the internal collision-packing"),
                   ('CLOTH', "Blender Cloth",
                    "Build the flat sheet and fold it with a Cloth "
                    "modifier (self-collision + a gather field)")],
            default='PBD')
        cloth_bake: IntProperty(
            name="Cloth Bake Frames", default=60, min=0, max=400,
            description="Cloth mode: frames to auto-simulate and "
                        "freeze. 0 = just set up the modifier and let "
                        "you press Play")
        cloth_gather: FloatProperty(
            name="Gather Strength", default=170.0, min=0.0, max=2000.0,
            description="Cloth mode: inward force that gathers the "
                        "sheet into a ball. Lower = looser, less "
                        "tightly folded (0 = drape under gravity)")
        cloth_prep: IntProperty(
            name="Sheet Prep Steps", default=40, min=0, max=400,
            description="Cloth mode: relaxation steps to tidy the "
                        "procedurally-ruffled sheet before handing it "
                        "to cloth (the cloth solver does the rest, so "
                        "this can be small or 0)")
        cloth_bend: IntProperty(
            name="Fold Softness", default=12, min=0, max=60,
            description="Cloth mode: smoothing passes applied after "
                        "folding, to round the sharp creases. Higher = "
                        "softer, rounder folds; 0 = raw cloth")
        cloth_thick: FloatProperty(
            name="Fabric Thickness", default=0.7, min=0.2, max=2.0,
            description="Cloth mode: self-collision distance as a "
                        "fraction of stitch size; higher keeps folds "
                        "from pinching into sharp seams")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0)
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            if self.preset == 'CUSTOM':
                p = dict(ratio_n=self.ratio_n, rows=self.rows,
                         stitch=self.stitch,
                         max_stitches=self.max_stitches,
                         iters=self.iters, smooth=self.smooth,
                         repel=self.repel, collide=self.collide)
            else:
                p = dict(PRESETS[self.preset])
            if self.physics == 'CLOTH':
                # hand cloth the procedurally-ruffled flat sheet (full
                # wave seed, no PBD packing/collision) with only a light
                # cleanup relax; the cloth solver refines + folds it
                p['pack'] = 0
                p['anneal'] = False
                p['collide'] = 0
                p['iters'] = self.cloth_prep
            verts, faces = build_ruffle(**p)
            verts = np.asarray(verts) * self.scale
            me = bpy.data.meshes.new("Crochet")
            me.from_pydata([tuple(v) for v in verts], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Crochet", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            note = ""
            if self.physics == 'CLOTH':
                note = self._setup_cloth(context, obj, verts, faces)
            self.report({'INFO'},
                        f"Crochet ({self.preset.title()}): "
                        f"V={len(obj.data.vertices)} "
                        f"F={len(obj.data.polygons)}{note}")
            return {'FINISHED'}

        def _setup_cloth(self, context, obj, verts, faces):
            """Attach a Cloth modifier (self-collision + optional inward
            gather field) that folds the flat sheet into a ball, and
            optionally bake+freeze it."""
            V = np.asarray(verts)
            cen = V.mean(axis=0)
            k = max(6, len(V) // 150)
            pin = np.argsort(np.linalg.norm(V[:, :2] - cen[:2],
                                            axis=1))[:k]
            vg = obj.vertex_groups.new(name="crochet_pin")
            vg.add([int(i) for i in pin], 1.0, 'REPLACE')
            med = float(np.median(
                [np.linalg.norm(V[f[0]] - V[f[1]]) for f in faces]))

            mod = obj.modifiers.new("Cloth", 'CLOTH')
            s = mod.settings
            s.quality = 10
            s.mass = 0.3
            s.tension_stiffness = 15.0
            s.compression_stiffness = 15.0
            s.shear_stiffness = 5.0
            try:
                s.bending_model = 'ANGULAR'
            except (AttributeError, TypeError):
                pass
            s.bending_stiffness = 3.0
            s.bending_damping = 2.0
            s.vertex_group_mass = "crochet_pin"     # pin group
            s.pin_stiffness = 5.0
            col = mod.collision_settings
            col.use_self_collision = True
            col.self_distance_min = min(0.1, max(0.003,
                                                 self.cloth_thick * med))
            col.self_impulse_clamp = 0.0
            col.collision_quality = 5

            fld = None
            if self.cloth_gather > 0.0:
                s.effector_weights.gravity = 0.0    # gather, don't droop
                fld = bpy.data.objects.new("Crochet Gather", None)
                context.collection.objects.link(fld)
                fld.location = tuple(float(c) for c in cen)
                context.view_layer.objects.active = fld
                bpy.ops.object.forcefield_toggle()  # enables fld.field
                fld.field.type = 'FORCE'
                fld.field.strength = -abs(self.cloth_gather)
                fld.field.falloff_type = 'SPHERE'
                context.view_layer.objects.active = obj

            # post-fold smoothing rounds the sharp cloth creases
            sm = None
            if self.cloth_bend > 0:
                sm = obj.modifiers.new("Fold Smooth", 'SMOOTH')
                sm.iterations = int(self.cloth_bend)
                sm.factor = 0.6

            if self.cloth_bake > 0:
                scene = context.scene
                start = scene.frame_start
                for fnum in range(start, start + self.cloth_bake):
                    scene.frame_set(fnum)
                context.view_layer.objects.active = obj
                try:
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                    if sm is not None:
                        bpy.ops.object.modifier_apply(modifier=sm.name)
                except RuntimeError:
                    pass
                if fld is not None:
                    bpy.data.objects.remove(fld, do_unlink=True)
                scene.frame_set(start)
                return f" (cloth baked {self.cloth_bake}f)"
            return " (cloth set up -- press Play)"

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                for k in ('ratio_n', 'rows', 'stitch', 'max_stitches',
                          'iters', 'smooth', 'repel', 'collide'):
                    lay.prop(self, k)
            lay.prop(self, 'physics')
            if self.physics == 'CLOTH':
                lay.prop(self, 'cloth_prep')
                lay.prop(self, 'cloth_bake')
                lay.prop(self, 'cloth_gather')
                lay.prop(self, 'cloth_bend')
                lay.prop(self, 'cloth_thick')
            lay.prop(self, 'scale')
            lay.prop(self, 'shade_smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.crochet_add", icon='MOD_CLOTH')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_crochet_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_crochet_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        import time
        h0 = 0.09
        Rc = h0 / math.log(1.0 + 1.0 / 4)
        t0 = time.time()
        V, E0, E1, REST, nbr0, F, pin, pinp = _crochet_mesh(4, 18, h0,
                                                            600)
        V = _relax(V, E0, E1, REST, nbr0, pin, pinp, 340, 0.06,
                   0.4 * h0, 0.5)
        dt = time.time() - t0
        strain = float(np.mean(np.linalg.norm(V[E1] - V[E0], axis=1)
                               / REST))
        zext = float(V[:, 2].max() - V[:, 2].min())
        K = mean_curvature(V, F)
        finite = np.isfinite(V).all()
        ok = finite and zext > 0.05 and K < 0
        print(f"RUFFLED: V={len(V)} F={len(F)} z_extent={zext:.3f} "
              f"meanK={K:+.2f} strain={strain:.3f} "
              f"targetK={-1.0 / Rc ** 2:+.2f} time={dt:.1f}s "
              f"{'OK' if ok else 'CHECK'}")
