
# Voronoi Openwork operator for Blender: perforate the active
# mesh with organic Voronoi-cell holes -- inspired by Primoz
# Gabrijelcic's "voronoizer" (github.com/gabr42/voronoizer),
# reworked as an in-Blender operator that runs on arbitrary
# surfaces, including open ones (minimal surfaces, TPMS patches).
#
# Construction (boolean-free):
#   1. triangulate + subdivide the source surface;
#   2. farthest-point sample N seed vertices using GRAPH-GEODESIC
#      distances (Dijkstra over the edge graph) -- crucial for
#      minimal surfaces, where two sheets can be close in space
#      but far apart on the surface;
#   3. label every vertex with its nearest seed (multi-source
#      Dijkstra), giving a geodesic Voronoi tessellation;
#   4. build the distance field to the nearest CELL BOUNDARY and
#      keep only the material within strut-half-width of it,
#      clipping triangles at the interpolated iso-contour (the
#      same field-clipping used by the stereographic shells), so
#      the hole rims are smooth;
#   5. a live Solidify modifier gives the shell its thickness.
#
# The mesh's own boundary (if open) can be kept as a solid frame.
#
# References:
#   - Voronoi diagram: Georgy F. Voronoi (1908); the equivalent
#     Dirichlet tessellation, Lejeune Dirichlet (1850); the dual
#     Delaunay triangulation, Boris N. Delaunay (1934).
#   - E. W. Dijkstra, "A Note on Two Problems in Connexion with
#     Graphs", Numerische Mathematik 1 (1959) -- the shortest-path
#     algorithm used here for the graph-geodesic distances.
#   - Primoz Gabrijelcic, "voronoizer"
#     (https://github.com/gabr42/voronoizer) -- the Voronoi-
#     perforation idea this operator reworks.

bl_info = {
    "name": "Voronoi Openwork",
    "author": "Math Art project (after gabr42/voronoizer)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Styles > Voronoi Openwork",
    "description": "Perforate the active mesh with geodesic "
                   "Voronoi holes",
    "category": "Object",
}

import heapq
import random

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def _dijkstra(adj, sources):
    """Multi-source Dijkstra over adjacency lists
    {v: [(u, w), ...]}; sources is {vertex: start_dist} or an
    iterable (start 0).  Returns (dist, label) with label[v] the
    source that reached v first."""
    if not isinstance(sources, dict):
        sources = {s: 0.0 for s in sources}
    dist = {}
    label = {}
    heap = []
    for s, d0 in sources.items():
        heapq.heappush(heap, (d0, s, s))
    while heap:
        d, v, src = heapq.heappop(heap)
        if v in dist:
            continue
        dist[v] = d
        label[v] = src
        for u, w in adj[v]:
            if u not in dist:
                heapq.heappush(heap, (d + w, u, src))
    return dist, label


def farthest_point_sample(adj, verts, n, rng):
    """n seed vertices spread by farthest-point sampling with
    graph-geodesic distances."""
    seeds = [rng.choice(verts)]
    dist, _ = _dijkstra(adj, seeds)
    for _ in range(n - 1):
        far = max(verts, key=lambda v: dist.get(v, 0.0))
        seeds.append(far)
        d2, _ = _dijkstra(adj, [far])
        for v in verts:
            dv = d2.get(v, 0.0)
            if dv < dist.get(v, 0.0):
                dist[v] = dv
    return seeds


if _IN_BLENDER:

    class OBJECT_OT_voronoi_openwork(bpy.types.Operator):
        """Perforate the active mesh with organic geodesic-Voronoi
        holes (after gabr42's voronoizer); works on open surfaces
        such as minimal surfaces"""
        bl_idname = "object.voronoi_openwork_add"
        bl_label = "Voronoi Openwork"
        bl_options = {'REGISTER', 'UNDO'}

        holes: IntProperty(
            name="Holes", default=40, min=2, max=400,
            description="Number of Voronoi cells (one hole each)")
        strut: FloatProperty(
            name="Strut Width", default=0.35, min=0.02, max=0.95,
            description="Strut width as a fraction of the mean "
                        "cell radius")
        subdiv: IntProperty(
            name="Subdivisions", default=2, min=0, max=4,
            description="Refinement of the surface before "
                        "clipping (smoother hole rims)")
        thickness: FloatProperty(
            name="Thickness", default=0.03, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        keep_frame: BoolProperty(
            name="Keep Boundary Frame", default=True,
            description="Treat the mesh's own open boundary as "
                        "strut material, keeping a solid frame "
                        "around open surfaces")
        rand_seed: IntProperty(name="Random Seed", default=1,
                               min=0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        @classmethod
        def poll(cls, context):
            ob = context.active_object
            return (ob is not None and ob.type == 'MESH'
                    and context.mode == 'OBJECT')

        def execute(self, context):
            src = context.active_object
            deps = context.evaluated_depsgraph_get()
            bm = bmesh.new()
            bm.from_object(src.evaluated_get(deps), deps)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            for _ in range(self.subdiv):
                bmesh.ops.subdivide_edges(
                    bm, edges=bm.edges[:], cuts=1,
                    use_grid_fill=True)
                bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.verts.ensure_lookup_table()
            bm.verts.index_update()

            # edge graph with lengths
            adj = {v.index: [] for v in bm.verts}
            for e in bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                w = (e.verts[0].co - e.verts[1].co).length
                adj[a].append((b, w))
                adj[b].append((a, w))
            verts = [v.index for v in bm.verts if adj[v.index]]
            if len(verts) < 4:
                self.report({'ERROR'}, "mesh has no usable faces")
                bm.free()
                return {'CANCELLED'}

            rng = random.Random(self.rand_seed)
            n = min(self.holes, max(2, len(verts) // 8))
            seeds = farthest_point_sample(adj, verts, n, rng)
            dist_seed, label = _dijkstra(adj, seeds)

            # boundary of the tessellation: edge midpoints whose
            # ends belong to different cells (start half an edge
            # in); optionally the mesh's own open boundary
            sources = {}
            for e in bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                if label.get(a) != label.get(b):
                    w = (e.verts[0].co - e.verts[1].co).length
                    sources[a] = min(sources.get(a, 1e30), w / 2)
                    sources[b] = min(sources.get(b, 1e30), w / 2)
                elif self.keep_frame and len(e.link_faces) == 1:
                    sources[a] = 0.0
                    sources[b] = 0.0
            if not sources:
                self.report({'ERROR'}, "no cell boundaries found")
                bm.free()
                return {'CANCELLED'}
            dist_b, _ = _dijkstra(adj, sources)

            # strut half-width from the mean cell radius
            mean_rad = (sum(dist_seed.values())
                        / max(1, len(dist_seed)))
            hw = self.strut * mean_rad

            # clip triangles keeping the strut side (g < 0); the
            # graph metric wiggles, so relax the field a little
            # first for smooth hole rims
            g = {v: dist_b.get(v, 0.0) - hw for v in adj}
            for _ in range(3):
                g = {v: (0.5 * g[v]
                         + 0.5 * (sum(g[u] for u, _w in adj[v])
                                  / len(adj[v])))
                     if adj[v] else g[v] for v in g}
            for v in g:
                if g[v] == 0.0:
                    g[v] = 1e-12
            out_verts = []
            out_faces = []
            vid = {}

            def corner(i):
                if i not in vid:
                    vid[i] = len(out_verts)
                    out_verts.append(tuple(bm.verts[i].co))
                return vid[i]

            xing = {}

            def crossing(a, b):
                key = (a, b) if a < b else (b, a)
                if key not in xing:
                    ga, gb = g[a], g[b]
                    t = ga / (ga - gb)
                    pa = bm.verts[a].co
                    pb = bm.verts[b].co
                    xing[key] = len(out_verts)
                    out_verts.append(tuple(pa + t * (pb - pa)))
                return xing[key]

            for f in bm.faces:
                cs = [v.index for v in f.verts]
                gs = [g[c] for c in cs]
                if all(x < 0 for x in gs):
                    out_faces.append([corner(c) for c in cs])
                    continue
                if not any(x < 0 for x in gs):
                    continue
                poly = []
                m = len(cs)
                for k in range(m):
                    if gs[k] < 0:
                        poly.append(corner(cs[k]))
                    if (gs[k] < 0) != (gs[(k + 1) % m] < 0):
                        poly.append(crossing(cs[k],
                                             cs[(k + 1) % m]))
                if len(poly) >= 3:
                    out_faces.append(poly)
            bm.free()
            if not out_faces:
                self.report({'ERROR'},
                            "strut width too small: nothing left")
                return {'CANCELLED'}

            me = bpy.data.meshes.new(f"{src.name} Openwork")
            me.from_pydata(out_verts, [], out_faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set(
                'use_smooth', [self.smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(f"{src.name} Openwork", me)
            context.collection.objects.link(obj)
            obj.matrix_world = src.matrix_world.copy()
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{n} cells, V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('holes', 'strut', 'subdiv', 'thickness',
                      'keep_frame', 'rand_seed', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("object.voronoi_openwork_add",
                             icon='MOD_REMESH')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(OBJECT_OT_voronoi_openwork)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.remove(_menu_func)
        bpy.utils.unregister_class(OBJECT_OT_voronoi_openwork)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # pure-python check of the graph machinery on a ring graph
        n = 12
        adj = {i: [((i + 1) % n, 1.0), ((i - 1) % n, 1.0)]
               for i in range(n)}
        rng = random.Random(1)
        seeds = farthest_point_sample(adj, list(range(n)), 3, rng)
        dist, label = _dijkstra(adj, seeds)
        assert len(set(seeds)) == 3
        assert max(dist.values()) <= n / 2
        assert set(label.values()) == set(seeds)
        print("voronoi openwork graph tests passed:",
              sorted(seeds))
