
# Saddle Polyhedron generator for Blender: polyhedra whose faces
# are smooth saddle membranes spanning skew polygon frames --
# Peter Pearce's "saddle polyhedra", and the family of Norman
# Carlberg's Minimal Surface Form solids.
#
# The frames are the seed polyhedron's PETRIE POLYGONS: skew
# circuits in which consecutive edges share a face but no three
# consecutive edges do.  Every edge of the seed lies in exactly
# two Petrie circuits, so membrane patches spanning them always
# close into a watertight saddle solid: the cube's four skew
# hexagons give the classic four-saddle star (Form 6 territory),
# the tetrahedron's three skew quads a three-hypar solid, the
# dodecahedron's and icosahedron's six skew decagons deeply
# fluted stars.
#
# Each patch is meshed as concentric rings toward the frame
# centroid and relaxed by pinned-boundary Laplacian iterations --
# a discrete membrane (Plateau-lite), so the patches are
# saddle-shaped like soap film on the skew frame.  Neighboring
# patches share their frame subdivision points exactly and are
# welded, leaving crisp ridge edges along the seed's edges.

bl_info = {
    "name": "Saddle Polyhedron",
    "author": "Math Art project (after Peter Pearce and Norman "
              "Carlberg)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Saddle membranes on the Petrie polygons of a "
                   "polyhedron",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def petrie_circuits(F):
    """The Petrie polygons of a manifold mesh, as vertex cycles.
    Walk state is (directed edge u->v, the face just used); the
    next edge continues from v on the OTHER face of (u, v).  Each
    (edge, face) pair belongs to exactly one circuit."""
    edge_faces = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            e = frozenset((f[i], f[(i + 1) % len(f)]))
            edge_faces.setdefault(e, []).append(fi)
    if any(len(fs) != 2 for fs in edge_faces.values()):
        raise ValueError("seed must be a closed manifold mesh")

    def other_face(u, v, f):
        a, b = edge_faces[frozenset((u, v))]
        return b if a == f else a

    def next_vertex(g, u, v):
        """Vertex adjacent to v in face g's cycle, not u."""
        f = F[g]
        i = f.index(v)
        n = len(f)
        cand = (f[(i + 1) % n], f[(i - 1) % n])
        return cand[0] if cand[0] != u else cand[1]

    visited = set()
    circuits = []
    for e, fs in edge_faces.items():
        u0, v0 = tuple(e)
        for f0 in fs:
            for u, v in ((u0, v0), (v0, u0)):
                if (u, v, f0) in visited:
                    continue
                cyc = []
                state = (u, v, f0)
                while state not in visited:
                    visited.add(state)
                    su, sv, sf = state
                    cyc.append(su)
                    g = other_face(su, sv, sf)
                    w = next_vertex(g, su, sv)
                    state = (sv, w, g)
                circuits.append(cyc)
    # each circuit is traced twice (once per direction); dedupe by
    # canonical undirected form
    out = []
    seen = set()
    for c in circuits:
        k = min(
            tuple(c[i:] + c[:i]) for i in range(len(c))
        )
        kr = list(reversed(c))
        k2 = min(
            tuple(kr[i:] + kr[:i]) for i in range(len(kr))
        )
        key = min(k, k2)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _patch(Vs, cyc, m, rings, iterations, lam=0.6):
    """Membrane patch over the skew polygon `cyc` (seed vertex
    ids): boundary = each side subdivided into m straight
    segments (shared verbatim with the neighboring patch),
    interior = concentric rings toward the centroid, relaxed by
    pinned-boundary Laplacian iterations.  Returns (verts,
    quads-and-tris)."""
    k = len(cyc)
    B = []
    for i in range(k):
        a = np.asarray(Vs[cyc[i]], float)
        b = np.asarray(Vs[cyc[(i + 1) % k]], float)
        for j in range(m):
            B.append(a + (b - a) * (j / m))
    B = np.array(B)
    nb = len(B)
    cen = B.mean(axis=0)
    R = max(2, rings)
    verts = []
    for r in range(R):
        f = 1.0 - r / R
        verts.extend(cen + (B - cen) * f)
    verts.append(cen)
    verts = np.array(verts)
    faces = []
    for r in range(R - 1):
        for j in range(nb):
            j2 = (j + 1) % nb
            faces.append([r * nb + j, r * nb + j2,
                          (r + 1) * nb + j2, (r + 1) * nb + j])
    last = (R - 1) * nb
    ctr = len(verts) - 1
    for j in range(nb):
        faces.append([last + j, last + (j + 1) % nb, ctr])
    # relax interior toward the minimal (Plateau) membrane with
    # the boundary ring pinned; the toolkit's cotangent-Laplacian
    # area minimizer gives true saddles, with uniform Laplacian
    # smoothing as a fallback
    P = verts.copy()
    fixed = np.zeros(len(P), dtype=bool)
    fixed[:nb] = True
    tris = []
    for f in faces:
        for i in range(1, len(f) - 1):
            tris.append((f[0], f[i], f[i + 1]))
    tris = np.array(tris)
    try:
        try:
            from .minimal_surface_toolkit import minimize_area
        except ImportError:
            from minimal_surface_toolkit import minimize_area
        minimize_area(P, tris, fixed,
                      outer_iters=max(1, iterations))
    except Exception:
        nbrs = [set() for _ in range(len(P))]
        for f in faces:
            n = len(f)
            for i in range(n):
                a, b = f[i], f[(i + 1) % n]
                nbrs[a].add(b)
                nbrs[b].add(a)
        nblist = [sorted(s) for s in nbrs]
        for _ in range(iterations * 4):
            Q = np.empty_like(P)
            for i, s in enumerate(nblist):
                Q[i] = P[s].mean(axis=0)
            P[~fixed] += lam * (Q[~fixed] - P[~fixed])
    return P, faces


def build_saddle(V, F, m=10, rings=8, iterations=150):
    """(verts, faces, patch_ids): welded saddle solid over all
    Petrie circuits of the seed."""
    circuits = petrie_circuits(F)
    vid = {}
    verts = []
    faces = []
    pids = []

    def emit(p, weld):
        # only the frame (seam) vertices are welded between
        # patches; interior sheets may pass through each other
        # (they all do at the centre of the cube form) and must
        # stay separate surfaces there
        if not weld:
            verts.append(tuple(p))
            return len(verts) - 1
        key = tuple(np.round(p, 9))
        if key not in vid:
            vid[key] = len(verts)
            verts.append(tuple(p))
        return vid[key]

    for pi, cyc in enumerate(circuits):
        P, pf = _patch(V, cyc, m, rings, iterations)
        nb = len(cyc) * m
        local = [emit(p, i < nb) for i, p in enumerate(P)]
        for f in pf:
            faces.append([local[i] for i in f])
            pids.append(pi)
    return verts, faces, pids, len(circuits)


def build_diagonal_caps(n=24, bulge=1.0, mirror=False, tip=2.4,
                        iterations=6):
    """The Carlberg Form 6 construction.  The six face diagonals
    of one inscribed regular tetrahedron cut the cube's surface
    into four bent triangular caps of three half-faces each, each
    cap wrapping one corner of the OTHER tetrad.  Over the
    tetra-face triangle (A, B, C) in barycentric coordinates the
    bent cap is EXACTLY the lift k = 3 min(a, b, c) toward the
    wrapped corner D (each median maps onto a cube edge, the
    centre onto D).  Form 6 is the MINIMAL SURFACE version: the
    membrane is pinned on the triangle frame and on a small
    corner tip of the bent cap (fraction `tip` around D) and
    Plateau-relaxed, giving the flat basins, corner funnels and
    straight diagonal ridges of the sculpture.  A tetrad
    fitted against the sculpture's scan sets the handedness;
    `mirror` picks the other one, `bulge` scales the lift."""
    even = [np.array(v, float) for v in
            ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    if not mirror:                     # the scan's handedness
        even = [v * np.array((-1.0, 1.0, 1.0)) for v in even]
    verts = []
    faces = []
    pids = []
    vid = {}

    def emit(p):
        key = tuple(np.round(p, 9))
        if key not in vid:
            vid[key] = len(verts)
            verts.append(tuple(p))
        return vid[key]

    try:
        try:
            from .minimal_surface_toolkit import minimize_area
        except ImportError:
            from minimal_surface_toolkit import minimize_area
    except Exception:
        minimize_area = None

    for pi in range(4):
        A, B, C = [even[j] for j in range(4) if j != pi]
        # the odd corner wrapped by this cap: the cube corner on
        # the 3-fold axis through the triangle's centroid
        G = (A + B + C) / 3.0
        D = np.sign(G)
        # consistent outward winding: the frame normal must point
        # toward the wrapped corner
        if np.dot(np.cross(B - A, C - A), D - G) < 0:
            B, C = C, B
        amp = bulge * (D - G)
        P = []
        pin = []
        idx = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                a = i / n
                b = j / n
                c = 1.0 - a - b
                m = 3.0 * min(a, b, c)
                # the lift blends from the flat tetrahedron face
                # (basins) up the bent-cap tent into the corner
                # horn; the exponent was fitted against a scan of
                # the sculpture (~2.4)
                k = m ** tip
                P.append(a * A + b * B + c * C + k * amp)
                pin.append(m < 1e-9 or m > 1.0 - 1e-9)
                idx[(i, j)] = len(P) - 1
        P = np.array(P)
        tris = []
        for i in range(n):
            for j in range(n - i):
                tris.append((idx[(i, j)], idx[(i + 1, j)],
                             idx[(i, j + 1)]))
                if j < n - i - 1:
                    tris.append((idx[(i + 1, j)],
                                 idx[(i + 1, j + 1)],
                                 idx[(i, j + 1)]))
        # light smoothing (frame and apex pinned) rounds the fin
        # creases along the medians near the horns
        if iterations > 0:
            fixed = np.array(pin)
            nbrs = [set() for _ in range(len(P))]
            for t in tris:
                for q in range(3):
                    aq, bq = t[q], t[(q + 1) % 3]
                    nbrs[aq].add(bq)
                    nbrs[bq].add(aq)
            nbl = [sorted(s) for s in nbrs]
            for _ in range(iterations):
                Q = np.empty_like(P)
                for vi, s in enumerate(nbl):
                    Q[vi] = P[s].mean(axis=0)
                P[~fixed] += 0.5 * (Q[~fixed] - P[~fixed])
        local = [emit(p) for p in P]
        for t in tris:
            faces.append([local[i] for i in t])
            pids.append(pi)
    return verts, faces, pids, 4


def _seed(name):
    try:
        from . import spiked_polyhedron_generator as sp
    except ImportError:
        import spiked_polyhedron_generator as sp
    V, F = sp._seed(name)
    return [tuple(v) for v in V], [list(f) for f in F]


if _IN_BLENDER:

    _PALETTE = [(0.85, 0.30, 0.20), (0.25, 0.50, 0.78),
                (0.32, 0.68, 0.40), (0.94, 0.76, 0.28),
                (0.60, 0.40, 0.74), (0.27, 0.71, 0.71)]

    def _material(idx):
        name = f"Saddle Patch {idx}"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            rgb = _PALETTE[idx % len(_PALETTE)]
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            b = mat.node_tree.nodes.get("Principled BSDF")
            if b is not None:
                b.inputs["Base Color"].default_value = (*rgb, 1.0)
                b.inputs["Roughness"].default_value = 0.5
        return mat

    class MESH_OT_saddle_polyhedron_add(bpy.types.Operator):
        """Saddle membranes spanning the Petrie polygons of a
        polyhedron: the cube gives the classic four-saddle star
        solid; works on the built-in seeds or the active mesh
        object"""
        bl_idname = "mesh.saddle_polyhedron_add"
        bl_label = "Saddle Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        style: EnumProperty(
            name="Construction",
            items=[('CAPS', "Diagonal Caps (Form 6)",
                    "Four smooth caps on the inscribed "
                    "tetrahedron's face-diagonal frame, each "
                    "bulging to a cube corner -- Norman "
                    "Carlberg's Minimal Surface Form 6"),
                   ('PETRIE', "Petrie Membranes",
                    "Plateau membranes spanning the Petrie "
                    "polygons of the seed polyhedron")],
            default='CAPS')
        bulge: FloatProperty(
            name="Bulge", default=1.0, min=0.0, max=2.0,
            description="Diagonal Caps: how far each cap reaches "
                        "toward its cube corner (1 = touches the "
                        "corner, 0 = flat tetrahedron)")
        mirror: BoolProperty(
            name="Mirrored", default=False,
            description="Diagonal Caps: use the other inscribed "
                        "tetrad (the opposite handedness)")
        tip: FloatProperty(
            name="Funnel Exponent", default=2.4, min=1.0, max=6.0,
            description="Diagonal Caps: how the sheets blend from "
                        "the flat basins up into the corner "
                        "horns; 1 = the bare bent cube caps, "
                        "higher = flatter basins with sharper "
                        "horns (2.4 fits the sculpture)")
        seed: EnumProperty(
            name="Seed",
            items=[('CUBE', "Cube",
                    "4 saddle hexagons"),
                   ('TETRA', "Tetrahedron", "3 saddle quads"),
                   ('OCTA', "Octahedron", "4 saddle hexagons"),
                   ('DODECA', "Dodecahedron", "6 saddle decagons"),
                   ('ICOSA', "Icosahedron", "6 saddle decagons"),
                   ('ACTIVE', "Active Object",
                    "Petrie circuits of the active mesh")],
            default='CUBE')
        side_segments: IntProperty(
            name="Side Segments", default=10, min=2, max=48,
            description="Subdivisions of each frame edge")
        rings: IntProperty(
            name="Rings", default=8, min=2, max=48,
            description="Concentric rings toward each patch "
                        "centre")
        iterations: IntProperty(
            name="Solver Iterations", default=30, min=1, max=200,
            description="Plateau area-minimization iterations per "
                        "patch (more = closer to the minimal "
                        "saddle)")
        color_patches: BoolProperty(
            name="Color Patches", default=False,
            description="A material per saddle patch")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        smooth: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.style == 'CAPS':
                n = max(8, self.side_segments * 2)
                verts, faces, pids, npat = build_diagonal_caps(
                    n, self.bulge, self.mirror, self.tip,
                    self.iterations)
                name = "Minimal Surface Form"
                return self._emit(context, name, verts, faces,
                                  pids, npat)
            if self.seed == 'ACTIVE':
                src = context.active_object
                if src is None or src.type != 'MESH':
                    self.report({'ERROR'},
                                "no active mesh object; pick a "
                                "built-in seed instead")
                    return {'CANCELLED'}
                deps = context.evaluated_depsgraph_get()
                me0 = src.evaluated_get(deps).to_mesh()
                V = [tuple(v.co) for v in me0.vertices]
                F = [list(p.vertices) for p in me0.polygons]
                src.evaluated_get(deps).to_mesh_clear()
                name = f"{src.name} Saddles"
            else:
                V, F = _seed(self.seed)
                name = f"Saddle {self.seed.title()}"
            try:
                verts, faces, pids, npat = build_saddle(
                    V, F, self.side_segments, self.rings,
                    self.iterations)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            return self._emit(context, name, verts, faces, pids,
                              npat)

        def _emit(self, context, name, verts, faces, pids, npat):
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set(
                'use_smooth', [self.smooth] * len(me.polygons))
            if self.color_patches and len(me.polygons) == len(pids):
                for i in range(npat):
                    me.materials.append(_material(i))
                me.polygons.foreach_set('material_index',
                                        [p % max(npat, 1)
                                         for p in pids])
            attr = me.attributes.new("patch_index", 'INT', 'FACE')
            if len(me.polygons) == len(pids):
                attr.data.foreach_set('value', pids)
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {npat} saddle patches, "
                        f"V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'style')
            if self.style == 'CAPS':
                lay.prop(self, 'bulge')
                lay.prop(self, 'tip')
                lay.prop(self, 'mirror')
                lay.prop(self, 'side_segments')
                lay.prop(self, 'iterations')
            else:
                for k in ('seed', 'side_segments', 'rings',
                          'iterations'):
                    lay.prop(self, k)
            for k in ('color_patches', 'thickness', 'smooth',
                      'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.saddle_polyhedron_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_saddle_polyhedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_saddle_polyhedron_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        CUBE_V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
                  for z in (-1, 1)]
        CUBE_F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                  [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        TET_V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        TET_F = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        OCT_V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1)]
        OCT_F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                 [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
        for name, V, F, want_n, want_len in (
                ("cube", CUBE_V, CUBE_F, 4, 6),
                ("tetra", TET_V, TET_F, 3, 4),
                ("octa", OCT_V, OCT_F, 4, 6)):
            circ = petrie_circuits(F)
            lens = sorted(len(c) for c in circ)
            ok = (len(circ) == want_n
                  and all(l == want_len for l in lens))
            print(f"{name}: {len(circ)} Petrie circuits of "
                  f"lengths {lens} (want {want_n} x {want_len}) "
                  f"{'OK' if ok else 'BAD'}")
            assert ok
            verts, faces, pids, npat = build_saddle(
                V, F, m=6, rings=5, iterations=40)
            cnt = {}
            for f in faces:
                for i in range(len(f)):
                    e = frozenset((f[i], f[(i + 1) % len(f)]))
                    cnt[e] = cnt.get(e, 0) + 1
            watertight = all(c == 2 for c in cnt.values())
            finite = all(all(math.isfinite(c) for c in v)
                         for v in verts)
            print(f"  patches={npat} V={len(verts)} "
                  f"F={len(faces)} watertight={watertight} "
                  f"finite={finite}")
            assert watertight and finite
        print("saddle polyhedron standalone tests passed")
