
# Minimal Surface Form generator for Blender, after Norman
# Carlberg's "Minimal Surface Form" sculpture series (e.g. Form 6,
# 1960, Hirshhorn): a polyhedron whose faces are pierced and
# relaxed into a smooth soap-film-like slab.
#
# The construction generalizes the classic cube tutorial to any
# polyhedron:
#   1. linearly subdivide every face into quads (Catmull-Clark
#      topology without the smoothing), `levels` times;
#   2. open a hole in each selected original face by deleting the
#      sub-faces near its centre;
#   3. relax with Laplacian smoothing: the free hole rims tighten
#      like a soap film with free edges, while the original corner
#      vertices can stay pinned so the form keeps reading as its
#      polyhedron;
#   4. a live Solidify modifier gives the sculptural thickness.
#
# Works on the built-in Platonic seeds or the active mesh object,
# so any polyhedron (Conway, Johnson, Catalan, ...) can be pierced
# the same way.

bl_info = {
    "name": "Minimal Surface Form",
    "author": "Math Art project (after Norman Carlberg)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Pierced, soap-film-relaxed polyhedra after "
                   "Carlberg's Minimal Surface Form series",
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


def linear_subdivide(V, F, origin=None):
    """One linear Catmull-Clark-style split: every n-gon becomes n
    quads (corner, edge-mid, centre, edge-mid).  Returns
    (V2, F2, origin2) where origin maps each face to the ORIGINAL
    seed face it descends from."""
    V = [tuple(v) for v in V]
    if origin is None:
        origin = list(range(len(F)))
    emid = {}
    fcen = {}
    verts = list(V)

    def edge_mid(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in emid:
            emid[key] = len(verts)
            verts.append(tuple(
                (V[a][k] + V[b][k]) / 2.0 for k in range(3)))
        return emid[key]

    F2 = []
    origin2 = []
    for fi, f in enumerate(F):
        n = len(f)
        c = len(verts)
        verts.append(tuple(sum(V[i][k] for i in f) / n
                           for k in range(3)))
        fcen[fi] = c
        for i in range(n):
            a = f[i]
            m1 = edge_mid(f[i - 1], a)
            m2 = edge_mid(a, f[(i + 1) % n])
            F2.append([m1, a, m2, c])
            origin2.append(origin[fi])
    return verts, F2, origin2


def build_form(V, F, levels=3, hole=0.55, pattern='ALL',
               iterations=40, pin='EDGES', smooth_lambda=0.5,
               corner_soft=0.35):
    """Pierced and relaxed form from seed polyhedron (V, F).
    Returns (verts, faces)."""
    V0 = [tuple(v) for v in V]
    n_corner = len(V0)
    # original face data for the hole test
    fdat = []
    for f in F:
        P = np.array([V0[i] for i in f], float)
        c = P.mean(axis=0)
        nrm = np.zeros(3)
        for i in range(1, len(f) - 1):
            nrm += np.cross(P[i] - P[0], P[i + 1] - P[0])
        ln = np.linalg.norm(nrm) or 1.0
        nrm /= ln
        # inradius ~ min distance from centre to an edge
        rin = min(
            np.linalg.norm(np.cross(P[(i + 1) % len(f)] - P[i],
                                    c - P[i]))
            / (np.linalg.norm(P[(i + 1) % len(f)] - P[i]) or 1.0)
            for i in range(len(f)))
        fdat.append((c, nrm, rin))

    verts, faces, origin = [tuple(v) for v in V0], \
        [list(f) for f in F], None
    for _ in range(max(1, levels)):
        verts, faces, origin = linear_subdivide(verts, faces,
                                                origin)

    def opened(fi):
        if pattern == 'NONE':
            return False
        if pattern == 'ALTERNATE':
            return fi % 2 == 0
        return True

    keep = []
    korig = []
    if hole > 0:
        Va = np.asarray(verts)
        for f, o in zip(faces, origin):
            c = Va[f].mean(axis=0)
            fc, nrm, rin = fdat[o]
            d = c - fc
            d = d - np.dot(d, nrm) * nrm      # in-plane distance
            if opened(o) and np.linalg.norm(d) < hole * rin:
                continue
            keep.append(f)
            korig.append(o)
        faces = keep
    if not faces:
        raise ValueError("hole size leaves no material")

    # drop unused vertices
    used = sorted({i for f in faces for i in f})
    remap = {i: k for k, i in enumerate(used)}
    P = np.array([verts[i] for i in used], float)
    faces = [[remap[i] for i in f] for f in faces]

    # soft pinning: relaxation strength ramps from 0 at the
    # pinned skeleton (the seed's corners, or its whole edge
    # frame -- the "cast in the mold" look) to 1 over a fraction
    # of the edge length
    weight = np.ones(len(P))
    if pin != 'NONE' and n_corner:
        C = np.array(V0, float)
        edges = {frozenset((f[i], f[(i + 1) % len(f)]))
                 for f in F for i in range(len(f))}
        elen = np.mean([np.linalg.norm(C[a] - C[b])
                        for a, b in map(tuple, edges)])
        soft = max(0.02, corner_soft) * elen
        if pin == 'CORNERS':
            d = np.min(np.linalg.norm(
                P[:, None, :] - C[None, :, :], axis=-1), axis=1)
        else:                          # EDGES: seed edge frame
            d = np.full(len(P), np.inf)
            for a, b in map(tuple, edges):
                A, B = C[a], C[b]
                AB = B - A
                L2 = float(AB @ AB) or 1.0
                u = np.clip((P - A) @ AB / L2, 0.0, 1.0)
                d = np.minimum(d, np.linalg.norm(
                    P - (A + u[:, None] * AB), axis=1))
        weight = np.clip(d / max(soft, 1e-9), 0.0, 1.0)

    # uniform Laplacian relaxation, free boundaries
    nb = [set() for _ in range(len(P))]
    for f in faces:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            nb[a].add(b)
            nb[b].add(a)
    nbl = [sorted(s) for s in nb]
    lam = smooth_lambda * weight[:, None]
    for _ in range(iterations):
        Q = np.empty_like(P)
        for i, s in enumerate(nbl):
            Q[i] = P[s].mean(axis=0) if s else P[i]
        P = P + lam * (Q - P)
    return [tuple(p) for p in P], faces


def _seed(name):
    try:
        from . import spiked_polyhedron_generator as sp
    except ImportError:
        import spiked_polyhedron_generator as sp
    return sp._seed(name)


if _IN_BLENDER:

    class MESH_OT_minimal_form_add(bpy.types.Operator):
        """Pierce and soap-film-relax a polyhedron into a smooth
        sculptural form (after Norman Carlberg's Minimal Surface
        Form series); works on the built-in seeds or the active
        mesh object"""
        bl_idname = "mesh.minimal_form_add"
        bl_label = "Minimal Surface Form"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Seed",
            items=[('CUBE', "Cube", "the classic Form 6 setup"),
                   ('TETRA', "Tetrahedron", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", ""),
                   ('ACTIVE', "Active Object",
                    "Pierce the active mesh object's faces")],
            default='CUBE')
        levels: IntProperty(
            name="Subdivisions", default=3, min=1, max=5,
            description="Linear face subdivisions before piercing")
        hole: FloatProperty(
            name="Hole Size", default=0.55, min=0.0, max=0.95,
            description="Opening radius as a fraction of each "
                        "face's inradius (0 = no holes)")
        pattern: EnumProperty(
            name="Openings",
            items=[('ALL', "Every Face", ""),
                   ('ALTERNATE', "Every Other Face", ""),
                   ('NONE', "None", "just relax the solid")],
            default='ALL')
        iterations: IntProperty(
            name="Relax Iterations", default=40, min=0, max=400,
            description="Laplacian soap-film relaxation steps; "
                        "hole rims tighten as they smooth")
        pin: EnumProperty(
            name="Hold",
            items=[('EDGES', "Edge Frame",
                    "Hold the seed's whole edge frame -- the "
                    "cast-in-the-mold Carlberg look"),
                   ('CORNERS', "Corners Only",
                    "Hold only the seed's vertices: spiky points "
                    "with deeply relaxed webs"),
                   ('NONE', "Nothing",
                    "Free relaxation; the form slumps toward a "
                    "blob")],
            default='EDGES',
            description="Which part of the seed stays fixed "
                        "while the surface relaxes")
        corner_soft: FloatProperty(
            name="Corner Softness", default=0.35, min=0.02,
            max=1.5,
            description="Radius (in edge lengths) over which the "
                        "relaxation fades out toward the pinned "
                        "corners: small = sharp Carlberg points, "
                        "large = rounded tips")
        thickness: FloatProperty(
            name="Thickness", default=0.08, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
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
                name = f"{src.name} Form"
            else:
                V, F = _seed(self.seed)
                V = [tuple(c) for c in V]
                name = f"Minimal Form ({self.seed.title()})"
            try:
                verts, faces = build_form(
                    V, F, self.levels, self.hole, self.pattern,
                    self.iterations, self.pin,
                    corner_soft=self.corner_soft)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            fscale = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * fscale
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness * fscale \
                    if self.seed == 'ACTIVE' else self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('seed', 'levels', 'hole', 'pattern',
                      'iterations', 'pin'):
                lay.prop(self, k)
            if self.pin != 'NONE':
                lay.prop(self, 'corner_soft')
            lay.prop(self, 'thickness')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.minimal_form_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_minimal_form_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_minimal_form_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # standalone: exercise the numeric core on a cube and an
        # icosahedron without bpy
        CUBE_V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
                  for z in (-1, 1)]
        CUBE_F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                  [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        for name, V, F, holes in (("cube", CUBE_V, CUBE_F, 6),):
            verts, faces = build_form(V, F, levels=3, hole=0.5,
                                      iterations=30)
            # boundary loops = number of holes
            cnt = {}
            for f in faces:
                for i in range(len(f)):
                    e = frozenset((f[i], f[(i + 1) % len(f)]))
                    cnt[e] = cnt.get(e, 0) + 1
            bed = [e for e, c in cnt.items() if c == 1]
            # count boundary loops by walking
            import collections
            adj = collections.defaultdict(list)
            for e in bed:
                a, b = tuple(e)
                adj[a].append(b)
                adj[b].append(a)
            seen = set()
            loops = 0
            for e in bed:
                a, _ = tuple(e)
                if a in seen:
                    continue
                loops += 1
                stack = [a]
                while stack:
                    u = stack.pop()
                    if u in seen:
                        continue
                    seen.add(u)
                    stack.extend(adj[u])
            finite = all(all(math.isfinite(c) for c in v)
                         for v in verts)
            print(f"{name}: V={len(verts)} F={len(faces)} "
                  f"holes={loops} (want {holes}) finite={finite}")
            assert loops == holes and finite
        print("minimal form standalone tests passed")
