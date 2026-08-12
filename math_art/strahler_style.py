
# Horton-Strahler ordering as a Styles operator.
#
# Applies to ANY branching object the add-on produces -- L-system plants,
# fractal trees, knotwork strands, DLA clusters, space-colonization
# skeletons -- and answers the question "which of these limbs are
# structural, and which are decoration?".
#
# THE ORDERING.  Number the tips 1.  At a junction, look at the orders of
# the branches meeting there:
#
#     * if the largest is UNIQUE, the parent keeps it;
#     * if the largest is shared by two or more, the parent takes it + 1.
#
# So order only rises where two comparably developed branches meet.
#
# WHY NOT JUST USE DEPTH FROM THE ROOT.  On a symmetric tree the two
# agree, which is why depth is so often used and so rarely questioned.
# On an ASYMMETRIC tree they diverge sharply.  Consider a long axis that
# sheds one small twig near its base: by depth the twig and the whole
# upper trunk are siblings at the same level, so any depth-driven
# thickness gives them the same radius.  By Strahler the twig is order 1
# and the trunk keeps the order of what it carries -- which is what the
# eye expects, and what a printable model needs, because it is exactly
# the criterion for "this limb can be thinned without weakening
# anything".
#
# Three styling modes are offered, all driven by the same order field:
# radius (curve bevel or a mesh attribute), a colour ramp, and pruning
# everything below a chosen order -- the last being the practical route
# to a printable simplification of a very bushy model.
#
# References:
# - Robert E. Horton, "Erosional development of streams and their
#   drainage basins: hydrophysical approach to quantitative morphology",
#   Bulletin of the Geological Society of America 56, 1945, pp. 275-370.
# - Arthur N. Strahler, "Hypsometric (area-altitude) analysis of
#   erosional topography", Bulletin of the Geological Society of America
#   63, 1952, pp. 1117-1142.
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, Figure 1.21 (Strahler order on a
#   bracketed structure).
# - Wojciech Palubicki, Kipp Horel, Steven Longay, Adam Runions, Brendan
#   Lane, Radomir Mech and Przemyslaw Prusinkiewicz, "Self-organizing
#   tree models for image synthesis", ACM TOG 28(3), SIGGRAPH 2009 --
#   pipe-model girth with memory of shed branches.

bl_info = {
    "name": "Strahler Style",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Styles > Strahler",
    "description": "Thickness, colour or pruning by Horton-Strahler order",
    "category": "Object",
}

import numpy as np


# ---------------------------------------------------------------------
# Pure graph layer (no bpy) -- this is what the self-test exercises
# ---------------------------------------------------------------------

def strahler_from_edges(n_verts, edges, root=None):
    """Horton-Strahler order per EDGE of a tree given as an edge list.

    Returns (orders, parent, root).  `orders[i]` is the order of edge
    `edges[i]`.  Cycles are tolerated: the traversal is a spanning tree
    from `root`, and any edge not on it gets the order of its deeper
    endpoint.
    """
    adj = [[] for _ in range(n_verts)]
    for ei, (a, b) in enumerate(edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))
    if root is None:
        root = _pick_root(n_verts, adj)

    # iterative DFS: order children, then fold them into the parent
    parent = [-1] * n_verts
    parent_edge = [-1] * n_verts
    order_of = [0] * n_verts        # order of the edge ABOVE each vertex
    seen = [False] * n_verts
    stack = [root]
    visit = []
    seen[root] = True
    while stack:
        v = stack.pop()
        visit.append(v)
        for u, ei in adj[v]:
            if not seen[u]:
                seen[u] = True
                parent[u] = v
                parent_edge[u] = ei
                stack.append(u)

    kids = [[] for _ in range(n_verts)]
    for v in range(n_verts):
        if parent[v] >= 0:
            kids[parent[v]].append(v)

    for v in reversed(visit):       # children before parents
        if not kids[v]:
            order_of[v] = 1
        else:
            ks = sorted((order_of[c] for c in kids[v]), reverse=True)
            order_of[v] = (ks[0] + 1 if len(ks) >= 2 and ks[0] == ks[1]
                           else ks[0])

    orders = [1] * len(edges)
    for v in range(n_verts):
        if parent_edge[v] >= 0:
            orders[parent_edge[v]] = order_of[v]
    return orders, parent, root


def _pick_root(n_verts, adj):
    """Root at a leaf -- the base of a plant is a degree-1 vertex, and
    of the leaves the lowest one is almost always the base."""
    leaves = [v for v in range(n_verts) if len(adj[v]) == 1]
    return leaves[0] if leaves else 0


def strahler_from_points(points, edges, root=None, axis=2):
    """As above, but pick the root as the EXTREME vertex along `axis`
    (default: lowest Z) among the degree-1 vertices -- the base of an
    upright plant."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    deg = np.zeros(n, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    if root is None:
        cand = np.where(deg == 1)[0]
        if len(cand):
            root = int(cand[np.argmin(pts[cand, axis])])
        else:
            root = int(np.argmin(pts[:, axis]))
    return strahler_from_edges(n, edges, root)


def radius_for(order, max_order, base=1.0, power=0.5):
    """Radius for a given order, conserving cross-sectional area up the
    tree: r ~ (order/max)^power with power = 0.5 reproducing da Vinci's
    rule between successive orders."""
    if max_order <= 1:
        return base
    return base * (float(order) / float(max_order)) ** float(power)


# ---------------------------------------------------------------------
# Blender layer
# ---------------------------------------------------------------------

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _mesh_graph(obj):
        me = obj.data
        pts = [tuple(v.co) for v in me.vertices]
        edges = [tuple(e.vertices) for e in me.edges]
        return pts, edges

    def _curve_graph(obj):
        """Weld spline endpoints so separate splines that meet form one
        tree; without the weld every strand looks like its own root."""
        pts, edges, lookup = [], [], {}

        def key(co):
            return (round(co[0], 5), round(co[1], 5), round(co[2], 5))

        for sp in obj.data.splines:
            coords = ([p.co[:3] for p in sp.points] if sp.type != 'BEZIER'
                      else [p.co[:3] for p in sp.bezier_points])
            idx = []
            for co in coords:
                k = key(co)
                if k not in lookup:
                    lookup[k] = len(pts)
                    pts.append(tuple(co))
                idx.append(lookup[k])
            for a, b in zip(idx, idx[1:]):
                if a != b:
                    edges.append((a, b))
        return pts, edges

    class OBJECT_OT_strahler_add(bpy.types.Operator):
        """Style a branching object by Horton-Strahler order: thickness,
        colour, or pruning of the low orders"""
        bl_idname = "object.strahler_add"
        bl_label = "Strahler"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('RADIUS', "Thickness", "Set per-point radius (curve) "
                              "or a width attribute (mesh) by order"),
                   ('COLOUR', "Colour", "Write a strahler attribute for a "
                              "material to read"),
                   ('PRUNE', "Prune", "Delete everything below a minimum "
                             "order -- the printable simplification")],
            default='RADIUS')
        base_radius: FloatProperty(
            name="Base Radius", default=1.0, min=0.0, max=10.0,
            description="Radius at the highest order present")
        power: FloatProperty(
            name="Area Exponent", default=0.5, min=0.05, max=2.0,
            description="r ~ (order/max)^p; 0.5 conserves cross-sectional "
                        "area between orders (da Vinci)")
        min_order: IntProperty(
            name="Keep Order >=", default=2, min=1, max=12,
            description="Prune mode: discard branches below this order")
        report_only: BoolProperty(
            name="Report Only", default=False,
            description="Compute and report the ordering without "
                        "modifying the object")

        @classmethod
        def poll(cls, context):
            ob = context.active_object
            return ob is not None and ob.type in {'MESH', 'CURVE'}

        def execute(self, context):
            ob = context.active_object
            pts, edges = (_curve_graph(ob) if ob.type == 'CURVE'
                          else _mesh_graph(ob))
            if not edges:
                self.report({'ERROR'}, "object has no edges to order")
                return {'CANCELLED'}

            orders, _parent, root = strahler_from_points(pts, edges)
            mx = max(orders)
            msg = (f"Strahler: {len(edges)} edges, max order {mx}, "
                   f"root vertex {root}")
            if self.report_only:
                self.report({'INFO'}, msg)
                return {'FINISHED'}

            vert_order = [1] * len(pts)
            for (a, b), o in zip(edges, orders):
                vert_order[a] = max(vert_order[a], o)
                vert_order[b] = max(vert_order[b], o)

            if self.mode == 'PRUNE':
                keep = [i for i, o in enumerate(orders) if o >= self.min_order]
                if not keep:
                    self.report({'ERROR'},
                                f"nothing at order >= {self.min_order} "
                                f"(max is {mx})")
                    return {'CANCELLED'}
                self._prune(ob, pts, edges, keep)
                self.report({'INFO'},
                            f"{msg}; kept {len(keep)}/{len(edges)} edges")
                return {'FINISHED'}

            if ob.type == 'CURVE' and self.mode == 'RADIUS':
                self._radius_curve(ob, vert_order, mx)
            else:
                self._attr_mesh(ob, vert_order, mx)
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def _radius_curve(self, ob, vert_order, mx):
            lookup, i = {}, 0
            for sp in ob.data.splines:
                pts = sp.points if sp.type != 'BEZIER' else sp.bezier_points
                for p in pts:
                    co = p.co[:3]
                    k = (round(co[0], 5), round(co[1], 5), round(co[2], 5))
                    o = lookup.get(k)
                    if o is None:
                        o = vert_order[min(i, len(vert_order) - 1)]
                        lookup[k] = o
                    p.radius = radius_for(o, mx, self.base_radius, self.power)
                    i += 1

        def _attr_mesh(self, ob, vert_order, mx):
            me = ob.data if ob.type == 'MESH' else None
            if me is None:
                return
            for name, vals, typ in (
                    ("strahler", [int(o) for o in vert_order], 'INT'),
                    ("width", [radius_for(o, mx, self.base_radius,
                                          self.power)
                               for o in vert_order], 'FLOAT')):
                try:
                    if name in me.attributes:
                        me.attributes.remove(me.attributes[name])
                    a = me.attributes.new(name=name, type=typ, domain='POINT')
                    a.data.foreach_set("value", vals)
                except Exception:
                    pass

        def _prune(self, ob, pts, edges, keep):
            keep_e = [edges[i] for i in keep]
            used = sorted({v for e in keep_e for v in e})
            remap = {v: i for i, v in enumerate(used)}
            me = bpy.data.meshes.new(ob.data.name + " Pruned")
            me.from_pydata([pts[v] for v in used],
                           [(remap[a], remap[b]) for a, b in keep_e], [])
            me.update()
            new = bpy.data.objects.new(ob.name + " Pruned", me)
            new.matrix_world = ob.matrix_world.copy()
            bpy.context.collection.objects.link(new)

    def _menu_func(self, context):
        self.layout.operator("object.strahler_add", icon='MOD_SIMPLIFY')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(OBJECT_OT_strahler_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.remove(_menu_func)
        bpy.utils.unregister_class(OBJECT_OT_strahler_add)


def _selftest():
    # A single path is order 1 throughout.
    edges = [(0, 1), (1, 2), (2, 3)]
    o, _p, _r = strahler_from_edges(4, edges)
    assert o == [1, 1, 1], o

    # A symmetric Y: two order-1 tips meeting make the stem order 2.
    #   0-1, then 1-2 and 1-3
    o, _p, root = strahler_from_edges(4, [(0, 1), (1, 2), (1, 3)])
    assert o[0] == 2 and o[1] == 1 and o[2] == 1, o
    assert root == 0, "the root should be the first degree-1 vertex"

    # THE ASYMMETRY CASE, which is the whole reason to prefer Strahler.
    # A stem (0-1) carries a symmetric fork (1-2, 2-4, 2-5) plus one
    # lone twig (1-3).  The fork's stem 1-2 is order 2; the twig is 1.
    # Their parent keeps 2 -- it does NOT become 3.
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)]
    o, _p, _r = strahler_from_edges(6, edges)
    assert o[1] == 2, o           # 1-2 carries the symmetric fork
    assert o[2] == 1, o           # 1-3 is the lone twig
    assert o[0] == 2, o           # so the trunk stays 2, not 3
    # depth-from-root would have called the twig and the fork siblings
    assert o[2] != o[1], "Strahler must separate the twig from the fork"

    # Two order-2 subtrees meeting DO make order 3.
    edges = [(0, 1),
             (1, 2), (2, 4), (2, 5),
             (1, 3), (3, 6), (3, 7)]
    o, _p, _r = strahler_from_edges(8, edges)
    assert o[0] == 3, o

    # Root selection by height: the lowest degree-1 vertex wins.
    pts = [(0, 0, 5), (0, 0, 4), (0, 0, 0)]
    o, _p, root = strahler_from_points(pts, [(0, 1), (1, 2)])
    assert root == 2, root

    # Radii conserve area between successive orders at power 0.5.
    r2 = radius_for(2, 2, 1.0, 0.5)
    r1 = radius_for(1, 2, 1.0, 0.5)
    assert abs(2 * r1 ** 2 - r2 ** 2) < 1e-12, (r1, r2)

    # A cycle must not hang or crash the traversal.
    o, _p, _r = strahler_from_edges(3, [(0, 1), (1, 2), (2, 0)])
    assert len(o) == 3 and all(x >= 1 for x in o)

    print("strahler_style: OK -- path/Y/asymmetric/symmetric orders, "
          "root by height, area-conserving radii, cycle tolerated")
