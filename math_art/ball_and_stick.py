
# Shared "Ball and Stick" style for the polyhedron generators.
#
# Render a polyhedron as a molecular-model-style ball-and-stick frame:
# every edge becomes a solid cylindrical strut and every vertex becomes
# a small sphere (a "node") that caps the struts meeting there.  This is
# the classic way physical polyhedron models (and geodesic domes) are
# built -- rods and connectors -- as opposed to a closed shell or an
# open Leonardo panel frame.
#
# The strut/node primitives here are the shared descendants of the ones
# in geodesic_generator.py, generalised to operate on any (vertices,
# edges) list so every polyhedron generator can offer the same style.
#
# Reference: R. Buckminster Fuller's geodesic "struts and hubs"
# construction (US Patent 2,682,235, 1954); the ball-and-stick idiom
# itself goes back to August Wilhelm von Hofmann's 19th-century
# molecular models.

from math import cos, sin, pi, sqrt


def _unit(v):
    L = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / L, v[1] / L, v[2] / L)


def _frame(d):
    """An orthonormal (d, u, w) frame with the given axis d."""
    d = _unit(d)
    h = (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _unit((d[1] * h[2] - d[2] * h[1],
               d[2] * h[0] - d[0] * h[2],
               d[0] * h[1] - d[1] * h[0]))
    w = (d[1] * u[2] - d[2] * u[1],
         d[2] * u[0] - d[0] * u[2],
         d[0] * u[1] - d[1] * u[0])
    return d, u, w


def _add_strut(verts, faces, p0, p1, radius, nseg=8):
    """A capped cylinder of the given radius from p0 to p1."""
    _d, u, w = _frame((p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]))
    base = len(verts)
    for p in (p0, p1):
        for j in range(nseg):
            t = 2 * pi * j / nseg
            c, s = cos(t), sin(t)
            verts.append((p[0] + radius * (c * u[0] + s * w[0]),
                          p[1] + radius * (c * u[1] + s * w[1]),
                          p[2] + radius * (c * u[2] + s * w[2])))
    for j in range(nseg):
        jn = (j + 1) % nseg
        faces.append([base + j, base + jn,
                      base + nseg + jn, base + nseg + j])
    faces.append([base + j for j in range(nseg - 1, -1, -1)])
    faces.append([base + nseg + j for j in range(nseg)])


def _add_node(verts, faces, center, radius, nu=8, nv=6):
    """A UV sphere of the given radius centred at `center`."""
    cx, cy, cz = center
    rows = []
    for i in range(1, nv):
        phi = pi * i / nv
        sp, cp = sin(phi), cos(phi)
        ring = []
        for j in range(nu):
            t = 2 * pi * j / nu
            ring.append(len(verts))
            verts.append((cx + radius * sp * cos(t),
                          cy + radius * sp * sin(t),
                          cz + radius * cp))
        rows.append(ring)
    top = len(verts)
    verts.append((cx, cy, cz + radius))
    bot = len(verts)
    verts.append((cx, cy, cz - radius))
    for j in range(nu):
        jn = (j + 1) % nu
        faces.append([top, rows[0][j], rows[0][jn]])
        faces.append([bot, rows[-1][jn], rows[-1][j]])
    for i in range(len(rows) - 1):
        for j in range(nu):
            jn = (j + 1) % nu
            faces.append([rows[i][j], rows[i + 1][j],
                          rows[i + 1][jn], rows[i][jn]])


def edges_from_faces(faces):
    """Unique undirected edges (as sorted (a, b) tuples) of a face list."""
    edges = set()
    for f in faces:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    return sorted(edges)


def build_mesh(verts_in, edges, strut_radius=0.02, node_radius=0.035,
               nseg=8):
    """Build a ball-and-stick mesh from vertices and edges.

    `verts_in` is a list of 3-tuples; `edges` is a list of (a, b) index
    pairs.  Returns (verts, faces) with a cylinder per edge and a sphere
    per vertex that is actually used by an edge.  Isolated vertices get
    no node."""
    verts, faces = [], []
    used = set()
    for a, b in edges:
        _add_strut(verts, faces, verts_in[a], verts_in[b],
                   strut_radius, nseg)
        used.add(a)
        used.add(b)
    if node_radius > 0.0:
        for i in sorted(used):
            _add_node(verts, faces, verts_in[i], node_radius)
    return verts, faces


try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _mesh_from(name, verts, faces):
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        # struts and nodes are round -- shade them smooth so the facet
        # segments do not read as flats
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        return me

    def rebuild(obj, strut_radius=0.02, node_radius=0.035, nseg=8):
        """Rebuild `obj`'s mesh in place as a ball-and-stick model from
        its own vertices and edges.  The same mesh datablock is reused
        (its geometry is cleared and refilled), so any reference the
        caller still holds to ``obj.data`` stays valid -- generators
        commonly read ``len(me.vertices)`` for their status report after
        applying the style."""
        me = obj.data
        verts_in = [tuple(v.co) for v in me.vertices]
        edges = [tuple(e.vertices) for e in me.edges]
        if not edges:                        # no edge data -> derive
            edges = edges_from_faces([list(p.vertices)
                                      for p in me.polygons])
        verts, faces = build_mesh(verts_in, edges, strut_radius,
                                  node_radius, nseg)
        me.clear_geometry()
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()

    def emit(context, verts_in, edges, label, strut_radius=0.02,
             node_radius=0.035, nseg=8, location=None):
        """Create and link a new ball-and-stick object from explicit
        vertices and edges.  Returns the object."""
        verts, faces = build_mesh(verts_in, edges, strut_radius,
                                  node_radius, nseg)
        me = _mesh_from(label, verts, faces)
        obj = bpy.data.objects.new(label, me)
        context.collection.objects.link(obj)
        obj.location = (location if location is not None
                        else context.scene.cursor.location)
        return obj

    def register():
        pass

    def unregister():
        pass


def _selftest():
    # A unit cube: 8 verts, 12 edges -> 12 struts + 8 nodes.
    V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    F = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
         [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    edges = edges_from_faces(F)
    if len(edges) != 12:
        raise AssertionError(f"cube should have 12 edges, got {len(edges)}")
    verts, faces = build_mesh(V, edges, 0.05, 0.08, nseg=8)
    # 12 struts (8-gon cylinder = 16 side + 2 caps) + 8 spheres, all
    # closed and non-empty.
    if not verts or not faces:
        raise AssertionError("ball-and-stick produced an empty mesh")
    # every face index must be in range
    n = len(verts)
    for f in faces:
        for i in f:
            if not (0 <= i < n):
                raise AssertionError("face index out of range")
    # node_radius = 0 -> struts only, no spheres
    v2, f2 = build_mesh(V, edges, 0.05, 0.0)
    if len(v2) != 12 * 16:
        raise AssertionError(
            f"12 struts should give {12 * 16} verts, got {len(v2)}")
