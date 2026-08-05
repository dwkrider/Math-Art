
# Realistic Snowflake Generator for Blender
#
# Snow-crystal growth by Reiter's cellular automaton on a triangular
# (hexagonal) lattice. Each cell holds an amount of water s; a cell is
# "frozen" (part of the crystal) once s >= 1 and "receptive" if it is
# frozen or touches a frozen cell. Each step the water is split into a
# receptive part (which stays put and gains a constant vapour input
# gamma) and a non-receptive part (which diffuses to the six
# neighbours with coefficient alpha), after which the two are
# recombined. From a single frozen seed this grows the full range of
# real snow-crystal habits -- hexagonal plates, sectored plates,
# stellar and fern-like dendrites -- all with exact six-fold symmetry
# because the lattice update is isotropic.
#
# The crystal is turned into geometry by extruding each frozen cell to
# a hexagonal prism whose height tracks its ice mass, so the plates,
# ribs and ridges of the real crystal appear as surface relief; the
# plate is symmetric about z = 0 (two-sided) and centred in a 2 m
# cube. Shared hexagon corners are welded, so a plate of equal-height
# cells is a single watertight surface, with risers stitching the
# height steps.
#
# References:
# - Clifford A. Reiter, "A local cellular model for snow crystal
#   growth", Chaos, Solitons & Fractals 23, 2005, pp. 1111-1119.
# - Related, physically richer models: Janko Gravner and David
#   Griffeath, "Modeling snow crystal growth II: A mesoscopic lattice
#   map with plausible dynamics", Physica D 237, 2008, pp. 385-404.
# - Snow-crystal morphology (the temperature/supersaturation habit
#   diagram): Ukichiro Nakaya, "Snow Crystals: Natural and
#   Artificial", Harvard University Press, 1954.

bl_info = {
    "name": "Snowflake (Reiter CA)",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Snowflake",
    "description": "Realistic snow crystals via Reiter's cellular "
                   "automaton",
    "category": "Add Mesh",
}

import math

import numpy as np

# neighbour offsets in axial coords, ordered by direction angle 60*m
# (m = 0..5): east, 60, 120, 180, 240, 300 degrees
_DIR_OFF = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def simulate_reiter(alpha=1.0, beta=0.4, gamma=0.0001, radius=70,
                    max_steps=6000):
    """Run Reiter's snow-crystal automaton. Returns (s, frozen,
    hexdist): the water field, the frozen (crystal) mask, and the
    hex-distance-from-centre of every cell on a (2R'+1)^2 axial grid
    (R' = radius + 2, a margin so diffusion never wraps into the
    crystal)."""
    ar = radius + 2
    n = 2 * ar + 1
    q = np.arange(n)[:, None] - ar
    r = np.arange(n)[None, :] - ar
    hexdist = (np.abs(q) + np.abs(r) + np.abs(q + r)) // 2
    domain = hexdist <= radius
    s = np.where(domain, beta, 0.0).astype(np.float64)
    s[ar, ar] = 1.0                                  # frozen seed

    def neigh_sum(a):
        tot = np.zeros_like(a)
        for dq, dr in _DIR_OFF:
            tot += np.roll(np.roll(a, dq, axis=0), dr, axis=1)
        return tot

    def neigh_any(mask):
        tot = np.zeros_like(mask)
        for dq, dr in _DIR_OFF:
            tot |= np.roll(np.roll(mask, dq, axis=0), dr, axis=1)
        return tot

    steps_done = 0
    for step in range(max_steps):
        frozen = s >= 1.0
        receptive = (frozen | neigh_any(frozen)) & domain
        u = np.where(receptive, 0.0, s)
        v = np.where(receptive, s + gamma, 0.0)
        u = u + (alpha / 2.0) * (neigh_sum(u) / 6.0 - u)
        s = v + u
        s = np.where(domain, s, beta)                # outer reservoir
        steps_done = step + 1
        if step % 25 == 0 and np.any((s >= 1.0)
                                     & (hexdist >= radius - 1)):
            break
    return s, (s >= 1.0) & domain, hexdist, steps_done


# ==========================================================================
# Geometry: extrude each frozen cell to a mass-scaled hexagonal prism
# ==========================================================================

def build_snowflake(frozen, s, spacing=1.0, base=0.4, relief=0.6,
                    s_cap=3.0, scale=1.0):
    """Turn a frozen mask + water field into a welded, two-sided
    hexagonal-relief plate. Each hexagon corner carries a single
    top height set by the ice mass of the frozen cells meeting there,
    so caps of neighbouring cells share corner vertices and the plate
    is one watertight surface; only the outline gets vertical walls.
    Returns (verts, faces)."""
    n = frozen.shape[0]
    ar = n // 2
    rc = spacing / math.sqrt(3.0)                    # hex circumradius
    corner = [(rc * math.cos(math.radians(60 * k + 30)),
               rc * math.sin(math.radians(60 * k + 30)))
              for k in range(6)]
    cells = [(int(i), int(j)) for i, j in np.argwhere(frozen)]

    def cell_xy(i, j):
        qq, rr = i - ar, j - ar
        return (qq + rr * 0.5) * spacing, rr * math.sqrt(3.0) / 2.0 * spacing

    def cell_mass(i, j):
        return min(max(float(s[i, j]), 1.0), s_cap)

    # pass 1: accumulate ice mass at each shared corner, so its top
    # height is the mean over the frozen cells that meet there
    acc = {}
    for i, j in cells:
        cx, cy = cell_xy(i, j)
        m = cell_mass(i, j)
        for k in range(6):
            key = (round(cx + corner[k][0], 4),
                   round(cy + corner[k][1], 4))
            tot, cnt = acc.get(key, (0.0, 0))
            acc[key] = (tot + m, cnt + 1)

    def top_z(key):
        tot, cnt = acc[key]
        return 0.5 * (base + relief * (tot / cnt - 1.0))

    verts = []
    faces = []
    vid = {}

    def V(x, y, z):
        vk = (round(x, 4), round(y, 4), round(z, 4))
        idx = vid.get(vk)
        if idx is None:
            idx = len(verts)
            vid[vk] = idx
            verts.append((x, y, z))
        return idx

    for i, j in cells:
        cx, cy = cell_xy(i, j)
        keys = [(round(cx + corner[k][0], 4),
                 round(cy + corner[k][1], 4)) for k in range(6)]
        pts = [(cx + corner[k][0], cy + corner[k][1]) for k in range(6)]
        zt = [top_z(keys[k]) for k in range(6)]
        top = [V(pts[k][0], pts[k][1], zt[k]) for k in range(6)]
        bot = [V(pts[k][0], pts[k][1], -zt[k]) for k in range(6)]
        faces.append(top[:])                         # +z cap
        faces.append(bot[::-1])                      # -z cap
        for e in range(6):
            k0, k1 = e, (e + 1) % 6
            di, dj = _DIR_OFF[(e + 1) % 6]            # cell across edge e
            ni, nj = i + di, j + dj
            nf = (0 <= ni < n and 0 <= nj < n and frozen[ni, nj])
            if not nf:                               # outline wall
                faces.append([top[k0], top[k1], bot[k1], bot[k0]])

    verts = np.asarray(verts, dtype=np.float64)
    if len(verts):
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ext = float((hi - lo).max())
        verts = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9
                                             else 1.0)
    return verts * scale, faces


# per-preset (alpha, beta, gamma) picked for distinct real habits
PRESETS = {
    'DENDRITE': ("Stellar Dendrite", 1.0, 0.4, 0.0001),
    'FERN': ("Fern Dendrite", 1.0, 0.35, 0.0001),
    'SECTORED': ("Sectored Plate", 1.0, 0.6, 0.001),
    'PLATE': ("Hexagonal Plate", 1.0, 0.9, 0.01),
    'STELLAR': ("Stellar Plate", 2.0, 0.5, 0.001),
}


def build_preset(kind='DENDRITE', radius=70, max_steps=6000,
                 base=0.4, relief=0.6, scale=1.0):
    label, alpha, beta, gamma = PRESETS[kind]
    s, frozen, _, steps = simulate_reiter(alpha, beta, gamma, radius,
                                          max_steps)
    verts, faces = build_snowflake(frozen, s, base=base,
                                   relief=relief, scale=scale)
    return verts, faces, int(frozen.sum()), steps


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

    class MESH_OT_snowflake_add(bpy.types.Operator):
        """Grow a realistic snow crystal with Reiter's cellular
        automaton and build it as a two-sided hexagonal-relief plate"""
        bl_idname = "mesh.snowflake_add"
        bl_label = "Snowflake"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Habit",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='DENDRITE')
        radius: IntProperty(
            name="Radius", default=70, min=16, max=180,
            description="Lattice radius in cells; growth stops when "
                        "the crystal nears this edge")
        max_steps: IntProperty(
            name="Max Steps", default=6000, min=100, max=40000,
            description="Cap on automaton steps (growth usually stops "
                        "earlier when it reaches the radius)")
        base: FloatProperty(
            name="Base Thickness", default=0.4, min=0.02, max=3.0,
            description="Prism height of a just-frozen cell (relative "
                        "to cell size, before the fit to 2 m)")
        relief: FloatProperty(
            name="Mass Relief", default=0.6, min=0.0, max=3.0,
            description="Extra height per unit of ice mass above 1, "
                        "giving the ridges and plateaus")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=False)

        def execute(self, context):
            label = PRESETS[self.preset][0]
            verts, faces, ncells, steps = build_preset(
                self.preset, radius=self.radius,
                max_steps=self.max_steps, base=self.base,
                relief=self.relief, scale=self.scale)
            if len(faces) == 0:
                self.report({'ERROR'}, "Crystal did not grow")
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Snowflake")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(f"Snowflake {label}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{label}: {ncells} cells in {steps} steps, "
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('preset', 'radius', 'max_steps', 'base',
                      'relief', 'scale', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.snowflake_add", icon='FREEZE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_snowflake_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_snowflake_add)


def _selftest():
    from collections import Counter

    def rot60(idx, ar):
        # axial 60-degree rotation (q, r) -> (-r, q + r)
        i, j = idx
        q, r = i - ar, j - ar
        return (-r + ar, (q + r) + ar)

    bad = []
    for kind in PRESETS:
        s, frozen, hexdist, steps = simulate_reiter(
            *PRESETS[kind][1:], radius=40, max_steps=4000)
        ar = frozen.shape[0] // 2
        cells = np.argwhere(frozen)
        # six-fold symmetry: rotating the frozen set by 60 degrees
        # maps it onto itself
        fset = set(map(tuple, cells))
        rset = {rot60(c, ar) for c in fset}
        sym = len(fset & rset) / max(len(fset), 1)
        verts, faces = build_snowflake(frozen, s)
        edges = Counter()
        for f in faces:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                edges[(min(a, b), max(a, b))] += 1
        boundary = sum(1 for c in edges.values() if c != 2)
        ok = len(cells) and sym > 0.99 and boundary == 0
        print(f"{kind}: cells={len(cells)} steps={steps} "
              f"6fold={sym:.3f} V={len(verts)} F={len(faces)} "
              f"boundary_edges={boundary} "
              f"{'OK' if ok else 'CHECK'}")
        if not ok:
            bad.append(kind)
    assert not bad
