
# Planar Simplify generator for Blender -- curvature-aware simplification
# of a surface mesh into a chosen number of near-planar pieces, as the
# front end of a lasercut / papercraft workflow: run this, then unfold
# the result with Blender's built-in "Export: Paper Model" add-on (which
# adds glue tabs, lays out the nets, and writes SVG/PDF).
#
# The clustering is Variational Shape Approximation (Cohen-Steiner,
# Alliez & Desbrun, SIGGRAPH 2004): the faces are partitioned into k
# planar "proxies" so as to minimize the total L2,1 error -- the area-
# weighted squared deviation of each face normal from its proxy normal.
# It is solved by Lloyd relaxation: (1) grow regions from k seeds,
# always claiming the lowest-error face next; (2) refit each proxy normal
# to the area-weighted average of its faces; (3) reseed each region at its
# best-fit face; repeat.  Each converged region is then merged (its
# interior edges dissolved) into one nearly-flat polygon, so the output
# is a low-poly control mesh whose faces approximate the surface with
# bounded normal deviation -- exactly what unfolds into clean flat pieces.
#
# References:
#   - D. Cohen-Steiner, P. Alliez, M. Desbrun, "Variational Shape
#     Approximation", ACM SIGGRAPH 2004 -- the k-proxy L2,1 clustering
#     and Lloyd relaxation used here.
#   - Gauss's Theorema Egregium (1827): a curved surface cannot be
#     flattened without cuts/distortion, so it is approximated by flat
#     facets and cut into a net for reassembly.
#   - Downstream unfolding: J. Mitani & H. Suzuki, "Making Papercraft
#     Toys from Meshes" (SIGGRAPH 2004); Blender's "Export: Paper Model".

bl_info = {
    "name": "Planar Simplify",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Construction > Simplify",
    "description": "Curvature-aware planar simplification (VSA) into flat "
                   "pieces for papercraft / lasercut unfolding",
    "category": "Add Mesh",
}

import numpy as np


# --------------------------------------------------------------------
# Pure-math core (no bpy): Variational Shape Approximation
# --------------------------------------------------------------------

def face_geometry(V, tris):
    """Per-face unit normal, area and centroid for a triangle list."""
    V = np.asarray(V, float)
    A = np.asarray(tris)
    P0 = V[A[:, 0]]
    e1 = V[A[:, 1]] - P0
    e2 = V[A[:, 2]] - P0
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / np.maximum(np.linalg.norm(cr, axis=1)[:, None], 1e-12)
    cen = (P0 + V[A[:, 1]] + V[A[:, 2]]) / 3.0
    return nrm, area, cen


def face_adjacency(tris):
    """Face adjacency across shared edges (dict: face -> [faces])."""
    from collections import defaultdict
    e2f = defaultdict(list)
    for ti, (a, b, c) in enumerate(tris):
        for x, y in ((a, b), (b, c), (c, a)):
            e2f[(x, y) if x < y else (y, x)].append(ti)
    adj = defaultdict(list)
    for fl in e2f.values():
        if len(fl) == 2:
            adj[fl[0]].append(fl[1])
            adj[fl[1]].append(fl[0])
    return adj


def _spread_seeds(N, k):
    """k farthest-point seeds in normal space (diverse orientations)."""
    n = len(N)
    seeds = [0]
    d = np.sum((N - N[0]) ** 2, axis=1)
    for _ in range(1, min(k, n)):
        s = int(np.argmax(d))
        seeds.append(s)
        d = np.minimum(d, np.sum((N - N[s]) ** 2, axis=1))
    return seeds


def vsa_cluster(V, tris, k, iters=15):
    """Partition faces into k planar proxies (VSA / Lloyd).  Returns
    (label per face, proxy normals (k,3))."""
    import heapq
    N, area, _cen = face_geometry(V, tris)
    n = len(tris)
    k = max(1, min(int(k), n))
    adj = face_adjacency(tris)
    seeds = _spread_seeds(N, k)
    prox = N[seeds].copy()
    label = np.full(n, -1, dtype=int)
    for _ in range(max(1, iters)):
        label[:] = -1
        heap = []
        for pid, s in enumerate(seeds):
            if label[s] == -1:
                label[s] = pid
            for nb in adj[s]:
                if label[nb] == -1:
                    err = area[nb] * float(np.sum((N[nb] - prox[pid]) ** 2))
                    heapq.heappush(heap, (err, nb, pid))
        while heap:
            err, ti, pid = heapq.heappop(heap)
            if label[ti] != -1:
                continue
            label[ti] = pid
            for nb in adj[ti]:
                if label[nb] == -1:
                    e = area[nb] * float(np.sum((N[nb] - prox[pid]) ** 2))
                    heapq.heappush(heap, (e, nb, pid))
        # islands unreachable via adjacency: nearest proxy by normal
        un = np.where(label == -1)[0]
        for ti in un:
            label[ti] = int(np.argmin(np.sum((prox - N[ti]) ** 2, axis=1)))
        # refit proxy normals (area-weighted average)
        acc = np.zeros((k, 3))
        for ti in range(n):
            acc[label[ti]] += area[ti] * N[ti]
        for pid in range(k):
            ln = np.linalg.norm(acc[pid])
            if ln > 1e-12:
                prox[pid] = acc[pid] / ln
        # reseed at each region's lowest-error face
        for pid in range(k):
            reg = np.where(label == pid)[0]
            if len(reg) == 0:
                continue
            errs = area[reg] * np.sum((N[reg] - prox[pid]) ** 2, axis=1)
            seeds[pid] = int(reg[np.argmin(errs)])
    return label, prox


def cluster_error(V, tris, label, prox):
    """Mean and max per-face normal deviation (degrees) from the proxy --
    a proxy of how flat each piece is."""
    N, area, _ = face_geometry(V, tris)
    dots = np.clip(np.sum(N * prox[label], axis=1), -1.0, 1.0)
    deg = np.degrees(np.arccos(dots))
    return float(np.average(deg, weights=area)), float(deg.max())


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, BoolProperty, FloatProperty,
                           EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PALETTE = [
        (0.90, 0.75, 0.30, 1), (0.82, 0.45, 0.28, 1), (0.55, 0.68, 0.35, 1),
        (0.35, 0.55, 0.62, 1), (0.72, 0.38, 0.45, 1), (0.48, 0.42, 0.66, 1),
        (0.86, 0.62, 0.42, 1), (0.40, 0.62, 0.50, 1),
    ]

    class OBJECT_OT_planar_simplify(bpy.types.Operator):
        """Simplify the active surface into a chosen number of near-flat
        pieces (Variational Shape Approximation), ready to unfold with
        Export Paper Model for lasercut / papercraft reassembly"""
        bl_idname = "object.planar_simplify"
        bl_label = "Planar Simplify (for Unfolding)"
        bl_options = {'REGISTER', 'UNDO'}

        pieces: IntProperty(
            name="Target Pieces", default=40, min=2, max=2000,
            description="Number of planar pieces (granularity): fewer = "
                        "coarser/flatter, more = closer to the surface")
        iterations: IntProperty(
            name="Iterations", default=15, min=1, max=60,
            description="Lloyd relaxation passes (higher = better-shaped, "
                        "lower-distortion pieces)")
        faces: EnumProperty(
            name="Faces",
            items=[('TRI', "Triangles",
                    "Re-tessellate each piece into triangles -- always "
                    "flat and simple (best for cutting)"),
                   ('QUAD', "Quads",
                    "Triangles merged into quads where they stay near-"
                    "planar (fewer, cleaner faces; some triangles remain)"),
                   ('NGON', "N-gons",
                    "One polygon per piece (fewest faces, but can be "
                    "concave / non-planar)")],
            default='TRI')
        color_pieces: BoolProperty(
            name="Color Pieces", default=True,
            description="Give each piece its own material color")
        keep_source: BoolProperty(
            name="Keep Original", default=True,
            description="Keep the source object and add the simplified "
                        "mesh as a new object")

        def execute(self, context):
            src = context.active_object
            if src is None or src.type != 'MESH':
                self.report({'ERROR'}, "select a mesh object")
                return {'CANCELLED'}
            deps = context.evaluated_depsgraph_get()
            ev = src.evaluated_get(deps)
            me = ev.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(me)
            ev.to_mesh_clear()
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.verts.index_update()
            bm.faces.index_update()
            V = [tuple(v.co) for v in bm.verts]
            tris = [tuple(v.index for v in f.verts) for f in bm.faces]
            if len(tris) < 2:
                bm.free()
                self.report({'ERROR'}, "mesh has too few faces")
                return {'CANCELLED'}

            k = min(self.pieces, len(tris))
            label, prox = vsa_cluster(V, tris, k, self.iterations)
            amean, amax = cluster_error(V, tris, label, prox)

            # tag each face with its proxy (as a material index) BEFORE
            # dissolving, so per-piece color survives the merge and the
            # re-tessellation below
            npal = len(_PALETTE)
            for f in bm.faces:
                f.material_index = int(label[f.index]) % npal
            # dissolve interior edges (both faces in the same proxy) so
            # each region collapses to one near-planar polygon
            inter = [e for e in bm.edges
                     if len(e.link_faces) == 2
                     and label[e.link_faces[0].index]
                     == label[e.link_faces[1].index]]
            bmesh.ops.dissolve_edges(bm, edges=inter, use_verts=True,
                                     use_face_split=False)
            # re-tessellate the merged pieces into simple, flat, cuttable
            # primitives (dissolved n-gons can be concave / non-planar)
            if self.faces in ('TRI', 'QUAD'):
                bmesh.ops.triangulate(bm, faces=list(bm.faces),
                                      quad_method='BEAUTY',
                                      ngon_method='BEAUTY')
                if self.faces == 'QUAD':
                    bmesh.ops.join_triangles(
                        bm, faces=list(bm.faces),
                        angle_face_threshold=0.35,
                        angle_shape_threshold=0.7,
                        cmp_seam=False, cmp_sharp=False,
                        cmp_uvs=False, cmp_materials=True)
            nm = bpy.data.meshes.new("%s Simplified" % src.name)
            bm.to_mesh(nm)
            bm.free()
            nm.update()

            if self.color_pieces:
                for i in range(npal):
                    mat = bpy.data.materials.new("Piece %d" % i)
                    mat.use_nodes = True
                    b = next(nd for nd in mat.node_tree.nodes
                             if nd.type == 'BSDF_PRINCIPLED')
                    b.inputs["Base Color"].default_value = _PALETTE[i]
                    mat.diffuse_color = _PALETTE[i]
                    nm.materials.append(mat)
                # material_index was set per piece on the bmesh faces
                nm.update()

            obj = bpy.data.objects.new("%s Simplified" % src.name, nm)
            context.collection.objects.link(obj)
            obj.matrix_world = src.matrix_world.copy()
            for o in context.selected_objects:
                o.select_set(False)
            if not self.keep_source:
                bpy.data.objects.remove(src, do_unlink=True)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                "Planar Simplify: %d pieces, %d faces; normal dev "
                "mean %.1f deg max %.1f deg. Now run Export Paper Model."
                % (k, len(nm.polygons), amean, amax))
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("object.planar_simplify", icon='MOD_REMESH')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(OBJECT_OT_planar_simplify)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(OBJECT_OT_planar_simplify)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _uv_sphere(nu=24, nv=14):
    import math
    V = []
    idx = {}
    for i in range(nv + 1):
        th = math.pi * i / nv
        for j in range(nu):
            ph = 2 * math.pi * j / nu
            idx[(i, j)] = len(V)
            V.append((math.sin(th) * math.cos(ph),
                      math.sin(th) * math.sin(ph), math.cos(th)))
    tris = []
    for i in range(nv):
        for j in range(nu):
            a = idx[(i, j)]
            b = idx[(i, (j + 1) % nu)]
            c = idx[(i + 1, (j + 1) % nu)]
            d = idx[(i + 1, j)]
            tris.append((a, b, c))
            tris.append((a, c, d))
    return V, tris


def _self_test():
    ok = True
    V, tris = _uv_sphere()
    for k in (12, 40, 120):
        label, prox = vsa_cluster(V, tris, k, iters=15)
        nlab = len(set(label.tolist()))
        cover = np.all(label >= 0)
        amean, amax = cluster_error(V, tris, label, prox)
        # more pieces -> flatter pieces (lower mean deviation)
        case_ok = cover and nlab >= 1 and amean < 40.0
        ok = ok and case_ok
        print("k=%3d  regions=%3d  cover=%s  normal dev mean=%.1f max=%.1f"
              "  : %s" % (k, nlab, "OK" if cover else "BAD", amean, amax,
                         "OK" if case_ok else "BAD"))
    m12 = cluster_error(V, tris, *vsa_cluster(V, tris, 12, 15))[0]
    m120 = cluster_error(V, tris, *vsa_cluster(V, tris, 120, 15))[0]
    mono = m120 <= m12
    ok = ok and mono
    print("more pieces -> flatter (mean dev %.1f -> %.1f) : %s"
          % (m12, m120, "OK" if mono else "BAD"))
    print("RESULT:", "OK" if ok else "BAD")
    return ok


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        _self_test()
