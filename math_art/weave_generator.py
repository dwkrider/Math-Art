
# Polyhedral Weave Generator for Blender
#
# A Blender take on Antiprism's `poly_weave`, including its weave
# PATTERN language. The seed polyhedron's flag (barycentric) subdivision
# is walked by a pattern program; the visited pattern points become
# closed weave strands, swept as ribbons.
#
# Pattern syntax (after poly_weave -p):
#   [C|L]                 curve (smoothed) or linear path   (default C)
#   bV,bE,bF              barycentric start point in each flag triangle
#                         over (Vertex, Edge centre, Face centre)
#                         (default 0,1,0 = edge centre)
#   :up[,side[,along]]    intermediate point on each step; repeatable.
#                         up = radial offset, side = sideways offset,
#                         along = position along step (default: spaced)
#   steps                 V E F  pivot about vertex/edge/face corner
#                         v e f  cross to the flag on the other side
#                         R      reverse pivot direction
#                         -      stay on the current flag
#   t[l|r|b]              start circuits from left/right/both flags
#
# Examples: "FEV" (default weave), "1,1,1V" (rings around vertices),
# "1,1,1F" (rings around faces), "1,1,1FFE" (squares on a cube),
# "1,1,1::::::FEV" (curved segments), "0,1,0:0.1FEV" (raised weave).

bl_info = {
    "name": "Polyhedral Weave Generator",
    "author": "David Krider (Math Art project, after Antiprism's poly_weave)",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polyhedral Weave",
    "description": "Woven-strand spheres from polyhedra (pattern language)",
    "category": "Add Mesh",
}

import math
import re
from math import cos

PHI = (1 + 5 ** 0.5) / 2


# ---- seeds ---------------------------------------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _icosa_faces(V):
    n = len(V)
    emin = None
    d2 = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((V[i][k] - V[j][k]) ** 2 for k in range(3))
            d2[(i, j)] = d
            emin = d if emin is None else min(emin, d)
    adj = {i: set() for i in range(n)}
    for (i, j), d in d2.items():
        if d < emin * 1.2:
            adj[i].add(j)
            adj[j].add(i)
    fs = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[i] & adj[j]:
                fs.add(tuple(sorted((i, j, k))))
    faces = []
    for f in fs:
        a, b, c = (V[i] for i in f)
        nx = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
              (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
              (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        cen = [(a[k] + b[k] + c[k]) / 3 for k in range(3)]
        if sum(nx[k] * cen[k] for k in range(3)) < 0:
            faces.append([f[0], f[2], f[1]])
        else:
            faces.append(list(f))
    return faces


def seed_poly(kind):
    if kind == 'TETRA':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    elif kind == 'OCTA':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    elif kind == 'CUBE':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    elif kind == 'ICOSA':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _icosa_faces(V)
    elif kind == 'DODECA':
        IV = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                IV += [(0, a, b), (a, b, 0), (b, 0, a)]
        IF = _icosa_faces(IV)
        V = [tuple(sum(IV[i][k] for i in f) / 3 for k in range(3))
             for f in IF]
        F = []
        for vi in range(len(IV)):
            adj = [fi for fi, f in enumerate(IF) if vi in f]
            nrm = _unit(IV[vi])
            u0 = V[adj[0]]
            u = [u0[k] - sum(u0[j] * nrm[j] for j in range(3)) * nrm[k]
                 for k in range(3)]
            w = (nrm[1] * u[2] - nrm[2] * u[1],
                 nrm[2] * u[0] - nrm[0] * u[2],
                 nrm[0] * u[1] - nrm[1] * u[0])
            def ang(fi):
                d = V[fi]
                return math.atan2(sum(d[k] * w[k] for k in range(3)),
                                  sum(d[k] * u[k] for k in range(3)))
            F.append(sorted(adj, key=ang))
    else:
        raise ValueError(kind)
    return [_unit(v) for v in V], [list(f) for f in F]


def geodesic(V, F, freq):
    if freq <= 1:
        return V, F
    verts = list(V)
    key = {}
    faces = []

    def vid(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in key:
            key[k] = len(verts)
            verts.append(p)
        return key[k]

    for f in F:
        A, B, C = (V[i] for i in f)
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                k = freq - i - j
                grid[(i, j)] = vid(_unit(tuple(
                    (i * A[c] + j * B[c] + k * C[c]) / freq
                    for c in range(3))))
        for i in range(freq):
            for j in range(freq - i):
                faces.append([grid[(i, j)], grid[(i + 1, j)],
                              grid[(i, j + 1)]])
                if j < freq - i - 1:
                    faces.append([grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                  grid[(i, j + 1)]])
    return verts, faces


# ---- pattern parsing -----------------------------------------------------

class Pattern:
    def __init__(self):
        self.curve = True
        self.bary = (0.0, 1.0, 0.0)
        self.inters = []          # list of (up, side, along-or-None)
        self.steps = []           # tokens from V E F v e f R -
        self.start = 'l'


def parse_pattern(text):
    p = Pattern()
    s = text.strip().replace(' ', '')
    if not s:
        s = 'FEV'
    if s[0] in 'CL':
        p.curve = s[0] == 'C'
        s = s[1:]
    m = re.match(r'^(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)', s)
    if m:
        p.bary = tuple(float(m.group(i)) for i in (1, 2, 3))
        if sum(p.bary) <= 0:
            raise ValueError("barycentric start must have positive sum")
        s = s[m.end():]
    while s.startswith(':'):
        s = s[1:]
        m = re.match(r'^(-?[\d.]*)(?:,(-?[\d.]+))?(?:,(-?[\d.]+))?', s)
        up = float(m.group(1)) if m.group(1) not in ('', '-', None) else 0.0
        side = float(m.group(2)) if m.group(2) else 0.0
        along = float(m.group(3)) if m.group(3) else None
        p.inters.append((up, side, along))
        s = s[m.end():]
    if s.endswith(('tl', 'tr', 'tb')):
        p.start = s[-1]
        s = s[:-2]
    for ch in s:
        if ch in 'VEFvefR-':
            p.steps.append(ch)
        else:
            raise ValueError(f"bad pattern step {ch!r}")
    if not any(c in 'VEFvef' for c in p.steps):
        raise ValueError("pattern needs at least one step from V E F v e f")
    return p


# ---- flag machinery ------------------------------------------------------
# flag = (face index, corner index i, side s); s=0: edge (f[i], f[i+1]),
# s=1: edge (f[i-1], f[i]). The three adjacencies (cross to the flag
# sharing everything except one element):
#   'v' change vertex, 'e' change edge, 'f' change face.

def build_flags(V, F):
    e2fd = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            e2fd[(f[i], f[(i + 1) % m])] = (fi, i)
    return e2fd


def flag_cross(F, e2fd, flag, tok):
    fi, i, s = flag
    f = F[fi]
    m = len(f)
    if tok == 'v':
        return (fi, (i + 1) % m, 1) if s == 0 else (fi, (i - 1) % m, 0)
    if tok == 'e':
        return (fi, i, 1 - s)
    # 'f': cross the edge to the neighbouring face, keep vertex and edge
    if s == 0:
        a, b = f[i], f[(i + 1) % m]
    else:
        a, b = f[(i - 1) % m], f[i]
    gj = e2fd.get((b, a))
    if gj is None:
        raise ValueError("open surface: cannot cross a boundary edge")
    g, j = gj
    # in face g the directed edge is (b, a); flag vertex must stay f[i]
    if s == 0:      # vertex a = start of (a,b): in g it is end of (b,a)
        return (g, (j + 1) % len(F[g]), 1)
    return (g, j, 0)


# A rotation click about an element is two elementary crossings; the
# order sets the rotation sense (R reverses it). V's default sense is
# chosen opposite to F's so that the classic FEV pattern advances as a
# weave strand instead of closing on itself.
_CLICK = {'F': ('e', 'v'), 'E': ('v', 'f'), 'V': ('e', 'f')}


def flag_step(F, e2fd, flag, tok, dirn):
    """One pattern step. Uppercase pivots are a full rotation click
    about the element: two flag crossings, landing on a flag of the
    same handedness."""
    if tok == '-':
        return flag, dirn
    if tok == 'R':
        return flag, -dirn
    if tok in 'vef':
        return flag_cross(F, e2fd, flag, tok), dirn
    a, b = _CLICK[tok]
    if dirn < 0:
        a, b = b, a
    mid = flag_cross(F, e2fd, flag, a)
    return flag_cross(F, e2fd, mid, b), dirn


def flag_point(V, F, flag, bary):
    fi, i, s = flag
    f = F[fi]
    m = len(f)
    vp = V[f[i]]
    if s == 0:
        a, b = f[i], f[(i + 1) % m]
    else:
        a, b = f[(i - 1) % m], f[i]
    ep = tuple((V[a][k] + V[b][k]) / 2 for k in range(3))
    fp = tuple(sum(V[j][k] for j in f) / m for k in range(3))
    bV, bE, bF = bary
    tot = bV + bE + bF
    return tuple((bV * vp[k] + bE * ep[k] + bF * fp[k]) / tot
                 for k in range(3))


def weave_circuits(V, F, pat):
    """Run the pattern over all start flags. Returns a list of circuits,
    each a list of 3D points (pattern points with intermediates)."""
    e2fd = build_flags(V, F)
    flags = []
    for fi, f in enumerate(F):
        for i in range(len(f)):
            flags.append((fi, i, 0))
            flags.append((fi, i, 1))
    if pat.start == 'l':
        starts = [fl for fl in flags if fl[2] == 0]
    elif pat.start == 'r':
        starts = [fl for fl in flags if fl[2] == 1]
    else:
        starts = flags
    visited = set()
    circuits = []
    L = len(pat.steps)
    for f0 in starts:
        state0 = (f0, 0, 1)
        if state0 in visited:
            continue
        pts = []
        flag, dirn = f0, 1
        phase = 0
        guard = 0
        while True:
            st = (flag, phase, dirn)
            if st in visited and pts:
                break
            visited.add(st)
            prev_pt = flag_point(V, F, flag, pat.bary)
            flag2, dirn2 = flag_step(F, e2fd, flag, pat.steps[phase], dirn)
            next_pt = flag_point(V, F, flag2, pat.bary)
            pts.append(prev_pt)
            if pat.inters:
                K = len(pat.inters)
                seg = tuple(next_pt[k] - prev_pt[k] for k in range(3))
                for kk, (up, side, along) in enumerate(pat.inters):
                    t = along if along is not None else (kk + 1) / (K + 1)
                    base = tuple(prev_pt[k] + seg[k] * t for k in range(3))
                    rad = _unit(base)
                    sd = _unit((seg[1] * rad[2] - seg[2] * rad[1],
                                seg[2] * rad[0] - seg[0] * rad[2],
                                seg[0] * rad[1] - seg[1] * rad[0])) \
                        if any(abs(x) > 1e-12 for x in seg) else (0, 0, 0)
                    pts.append(tuple(base[k] + rad[k] * up + sd[k] * side
                                     for k in range(3)))
            flag, dirn = flag2, dirn2
            phase = (phase + 1) % L
            guard += 1
            if guard > 100 * len(flags) * L:
                raise RuntimeError("pattern circuit failed to close")
        # drop consecutive duplicates
        clean = []
        for q in pts:
            if not clean or math.dist(q, clean[-1]) > 1e-9:
                clean.append(q)
        if len(clean) >= 3:
            circuits.append(clean)
    # drop geometric duplicates (the same circuit reached from a
    # different flag state, e.g. traversed in the opposite direction)
    seen_sets = set()
    unique = []
    for c in circuits:
        key = frozenset((round(p[0], 6), round(p[1], 6), round(p[2], 6))
                        for p in c)
        if key not in seen_sets:
            seen_sets.add(key)
            unique.append(c)
    return unique


# ---- ribbon sweep --------------------------------------------------------

def sweep_ribbons(circuits, width, thickness, amplitude, subdiv,
                  smooth_rounds, scale, curve=True):
    verts = []
    faces = []
    for pts in circuits:
        L = len(pts)
        path = []
        for i in range(L):
            p0, p1 = pts[i], pts[(i + 1) % L]
            r0 = 1.0 + amplitude * (1 if i % 2 == 0 else -1)
            r1 = 1.0 + amplitude * (1 if (i + 1) % 2 == 0 else -1)
            nseg = subdiv if curve else 1
            for s in range(nseg):
                t = s / nseg
                base = tuple(p0[k] + (p1[k] - p0[k]) * t for k in range(3))
                if amplitude > 0:
                    r = r0 + (r1 - r0) * (1 - cos(math.pi * t)) / 2
                    ln = math.sqrt(sum(x * x for x in base)) or 1.0
                    base = tuple(x / ln * (ln * r) for x in base)
                path.append(base)
        if curve:
            for _ in range(smooth_rounds):
                n = len(path)
                new = []
                for i in range(n):
                    a, b, c = path[i - 1], path[i], path[(i + 1) % n]
                    new.append(tuple((a[k] + 2 * b[k] + c[k]) / 4
                                     for k in range(3)))
                path = new
        n = len(path)
        base_i = len(verts)
        for i in range(n):
            p = path[i]
            nxt = path[(i + 1) % n]
            prv = path[i - 1]
            tang = _unit(tuple(nxt[k] - prv[k] for k in range(3)))
            rad = _unit(p)
            side = _unit((tang[1] * rad[2] - tang[2] * rad[1],
                          tang[2] * rad[0] - tang[0] * rad[2],
                          tang[0] * rad[1] - tang[1] * rad[0]))
            for (sw, sr) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                verts.append(tuple(
                    (p[k] + side[k] * sw * width / 2
                     + rad[k] * sr * thickness / 2) * scale
                    for k in range(3)))
        for i in range(n):
            i2 = (i + 1) % n
            for j in range(4):
                faces.append([base_i + 4 * i + j,
                              base_i + 4 * i + (j + 1) % 4,
                              base_i + 4 * i2 + (j + 1) % 4,
                              base_i + 4 * i2 + j])
    return verts, faces


def build_weave(kind='CUBE', freq=1, pattern='FEV', width=0.10,
                thickness=0.03, amplitude=0.05, subdiv=6,
                smooth_rounds=2, scale=1.0):
    V, F = seed_poly(kind)
    if kind in ('TETRA', 'OCTA', 'ICOSA'):
        V, F = geodesic(V, F, freq)
    pat = parse_pattern(pattern)
    circuits = weave_circuits(V, F, pat)
    verts, faces = sweep_ribbons(circuits, width, thickness, amplitude,
                                 subdiv, smooth_rounds, scale,
                                 curve=pat.curve)
    return verts, faces, len(circuits)


# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PATTERNS = [
        ('CUSTOM', "Custom", ""),
        ('vfe', "Classic Weave (vfe)",
         "straight-ahead weave: cube 4 hexagons, icosa 6 decagons"),
        ('FEV', "Corner Weave (FEV)", "circuits turning around vertices"),
        ('1,1,1V', "Vertex Rings (1,1,1V)", "circuits around vertices"),
        ('1,1,1F', "Face Rings (1,1,1F)", "circuits around faces"),
        ('1,1,1FFE', "Face Pairs (1,1,1FFE)", "squares on a cube"),
        ('1,1,1::::::FEV', "Curved (1,1,1::::::FEV)",
         "six intermediate points"),
        ('0,1,0:0.12FEV', "Raised Weave (0,1,0:0.12FEV)",
         "intermediate point lifted radially"),
        ('0,1,0:0.1,0.06:-0.1,0.06FEV', "Wavy (up/side intermediates)",
         ""),
        ('1,3,1EV', "Knots (1,3,1EV)", "small rings at each edge"),
    ]

    class MESH_OT_weave_add(bpy.types.Operator):
        """Add a woven-strand model driven by a poly_weave-style
        pattern walked over the seed polyhedron's flag triangles"""
        bl_idname = "mesh.poly_weave_add"
        bl_label = "Polyhedral Weave"
        bl_options = {'REGISTER', 'UNDO'}

        def _pattern_chosen(self, context):
            if self.pattern_preset != 'CUSTOM':
                self.pattern = self.pattern_preset

        kind: EnumProperty(
            name="Seed",
            items=[('CUBE', "Cube", ""), ('ICOSA', "Icosahedron", ""),
                   ('OCTA', "Octahedron", ""), ('TETRA', "Tetrahedron", ""),
                   ('DODECA', "Dodecahedron", "")],
            default='CUBE')
        freq: IntProperty(
            name="Geodesic Frequency", default=1, min=1, max=6,
            description="Subdivision of triangular seeds")
        pattern_preset: EnumProperty(name="Pattern Preset", items=PATTERNS,
                                     default='vfe',
                                     update=_pattern_chosen)
        pattern: StringProperty(
            name="Pattern", default="vfe",
            description="[C|L][bV,bE,bF][:up,side,along...]steps[tl|tr|tb]"
                        " -- steps from V E F v e f R -")
        width: FloatProperty(name="Strand Width", default=0.10,
                             min=0.005, max=0.5)
        thickness: FloatProperty(name="Strand Thickness", default=0.03,
                                 min=0.002, max=0.2)
        amplitude: FloatProperty(
            name="Weave Amplitude", default=0.05, min=0.0, max=0.3,
            description="Alternating radial offset at pattern points "
                        "(set 0 when the pattern's 'up' values weave)")
        subdiv: IntProperty(name="Path Subdivision", default=6,
                            min=1, max=24)
        smooth_rounds: IntProperty(name="Smoothing", default=2,
                                   min=0, max=10)
        scale: FloatProperty(name="Radius", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            try:
                verts, faces, ns = build_weave(
                    self.kind, self.freq, self.pattern, self.width,
                    self.thickness, self.amplitude, self.subdiv,
                    self.smooth_rounds, self.scale)
            except (ValueError, RuntimeError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Weave")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Weave", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{ns} strands")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind in ('TETRA', 'OCTA', 'ICOSA'):
                lay.prop(self, 'freq')
            lay.prop(self, 'pattern_preset')
            lay.prop(self, 'pattern')
            for k in ('width', 'thickness', 'amplitude', 'subdiv',
                      'smooth_rounds', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.poly_weave_add", icon='MOD_LATTICE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_weave_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_weave_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for kind, freq, pattern in (
                ('CUBE', 1, 'FEV'), ('OCTA', 1, 'FEV'),
                ('ICOSA', 1, 'FEV'), ('ICOSA', 2, 'FEV'),
                ('CUBE', 1, '1,1,1V'), ('CUBE', 1, '1,1,1F'),
                ('CUBE', 1, '1,1,1FFE'), ('DODECA', 1, 'FEV'),
                ('ICOSA', 2, '1,1,1::::::FEV'),
                ('CUBE', 1, '0,1,0:0.12FEV'),
                ('CUBE', 1, '1,3,1EV')):
            V, F = seed_poly(kind)
            if kind in ('TETRA', 'OCTA', 'ICOSA'):
                V, F = geodesic(V, F, freq)
            pat = parse_pattern(pattern)
            cir = weave_circuits(V, F, pat)
            lens = sorted(set(len(c) for c in cir))
            print(f"{kind} f{freq} '{pattern}': strands={len(cir)} "
                  f"point-counts={lens}")
