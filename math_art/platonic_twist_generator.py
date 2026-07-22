
# Platonic Twist Sculpture Generator for Blender
#
# After George W. Hart's "Platonic Twist" construction: the faces of a
# Platonic solid are shrunk and pushed apart, then reconnected along
# each original edge by a ribbon that makes a half twist (or several)
# on the way -- a family of airy, paper-sculpture-like forms.
# The output is a single welded surface; use the Thickness option
# (Solidify modifier) to make it printable.

bl_info = {
    "name": "Platonic Twist Sculptures",
    "author": "David Krider (Math Art project, after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Platonic Twist",
    "description": "Shrunken Platonic faces joined by twisted ribbons",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi

PHI = (1 + 5 ** 0.5) / 2


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
    return [list(v) for v in V], [list(f) for f in F]


def _slerp_rot(axis_from, axis_to, t):
    """Rotation matrix rotating axis_from toward axis_to by fraction t
    (about their common perpendicular)."""
    d = max(-1.0, min(1.0, sum(axis_from[k] * axis_to[k]
                               for k in range(3))))
    ang = math.acos(d) * t
    ax = (axis_from[1] * axis_to[2] - axis_from[2] * axis_to[1],
          axis_from[2] * axis_to[0] - axis_from[0] * axis_to[2],
          axis_from[0] * axis_to[1] - axis_from[1] * axis_to[0])
    l = math.sqrt(sum(x * x for x in ax))
    if l < 1e-9:
        ax = (1.0, 0.0, 0.0)
        ang = 0.0
    else:
        ax = tuple(x / l for x in ax)
    return ax, ang


def _rot(p, ax, ang):
    c, s = cos(ang), sin(ang)
    d = sum(ax[k] * p[k] for k in range(3))
    cr = (ax[1] * p[2] - ax[2] * p[1], ax[2] * p[0] - ax[0] * p[2],
          ax[0] * p[1] - ax[1] * p[0])
    return tuple(p[k] * c + cr[k] * s + ax[k] * d * (1 - c)
                 for k in range(3))


def build_platonic_twist(kind='CUBE', shrink=0.55, push=0.35,
                         half_twists=1, band_rows=16, edge_cols=6,
                         bulge=0.35, scale=1.0):
    """Shrunken face plates + twisted connecting ribbons, exactly
    welded (band end columns reuse the face edge vertices)."""
    V, F = seed_poly(kind)
    verts = []
    faces = []
    # shrunken faces, with each edge subdivided into edge_cols segments
    face_edge_vids = {}    # (face index, edge position) -> [vert ids]
    for fi, f in enumerate(F):
        m = len(f)
        c = [sum(V[i][k] for i in f) / m for k in range(3)]
        n = _unit(c)
        cl = math.sqrt(sum(x * x for x in c))
        cen = [n[k] * (cl + push) for k in range(3)]
        corner = []
        for i in f:
            corner.append(tuple(cen[k] + (V[i][k] - c[k]) * shrink
                                for k in range(3)))
        ring = []
        for e in range(m):
            a = corner[e]
            b = corner[(e + 1) % m]
            ids = []
            for s2 in range(edge_cols):
                t = s2 / edge_cols
                p = tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
                verts.append(tuple(x * scale for x in p))
                ids.append(len(verts) - 1)
            ring.extend(ids)
            face_edge_vids[(fi, e)] = ids + [None]   # last filled below
        # close the per-edge id lists with the next edge's first vertex
        for e in range(m):
            nxt = face_edge_vids[(fi, (e + 1) % m)][0]
            face_edge_vids[(fi, e)][-1] = nxt
        # fan-fill the plate
        verts.append(tuple(x * scale for x in cen))
        cid = len(verts) - 1
        L = len(ring)
        for i in range(L):
            faces.append([ring[i], ring[(i + 1) % L], cid])

    # directed edge -> (face, edge position)
    e2fe = {}
    for fi, f in enumerate(F):
        m = len(f)
        for e in range(m):
            e2fe[(f[e], f[(e + 1) % m])] = (fi, e)

    # ribbons per undirected edge
    done = set()
    for (a, b), (f1, e1) in e2fe.items():
        if (min(a, b), max(a, b)) in done:
            continue
        done.add((min(a, b), max(a, b)))
        f2, e2 = e2fe[(b, a)]
        ids1 = face_edge_vids[(f1, e1)]            # runs a -> b on face 1
        ids2 = face_edge_vids[(f2, e2)][::-1]      # reversed: a -> b on f2
        m1 = len(F[f1])
        m2 = len(F[f2])
        c1 = [sum(V[i][k] for i in F[f1]) / m1 for k in range(3)]
        c2 = [sum(V[i][k] for i in F[f2]) / m2 for k in range(3)]
        n1 = _unit(c1)
        n2 = _unit(c2)
        P1 = [verts[i] for i in ids1]
        P2 = [verts[i] for i in ids2]
        mid1 = tuple((P1[0][k] + P1[-1][k]) / 2 for k in range(3))
        mid2 = tuple((P2[0][k] + P2[-1][k]) / 2 for k in range(3))
        out = _unit(tuple(mid1[k] + mid2[k] for k in range(3)))
        ax, full_ang = _slerp_rot(n1, n2, 1.0)
        ncols = len(P1)
        rows = [ids1]
        for r in range(1, band_rows):
            t = r / band_rows
            sm = t * t * (3 - 2 * t)
            # center path with outward bulge
            cen = [mid1[k] + (mid2[k] - mid1[k]) * sm for k in range(3)]
            bg = bulge * 4 * sm * (1 - sm)
            cen = [cen[k] + out[k] * bg for k in range(3)]
            # frame: bend n1 toward n2, plus the extra half twists
            row = []
            spine = _unit(tuple(mid2[k] - mid1[k] for k in range(3)))
            for i2 in range(ncols):
                q0 = tuple(P1[i2][k] - mid1[k] for k in range(3))
                q = _rot(q0, ax, full_ang * sm)
                q = _rot(q, spine, half_twists * pi * sm)
                p = tuple((cen[k] + q[k]) * 1.0 for k in range(3))
                verts.append(p)
                row.append(len(verts) - 1)
            rows.append(row)
        # the final row must be exactly the face-2 edge; with a half
        # twist the ribbon arrives reversed
        end = ids2 if half_twists % 2 == 0 else ids2[::-1]
        rows.append(end)
        for r in range(band_rows):
            for i2 in range(ncols - 1):
                # wound opposite to the plate fans so the shared start
                # edge is traversed in the opposite direction (consistent
                # orientation across the junction)
                faces.append([rows[r][i2 + 1], rows[r][i2],
                              rows[r + 1][i2], rows[r + 1][i2 + 1]])
    rim = set()
    for ids in face_edge_vids.values():
        rim.update(ids)
    return verts, faces, len(done), rim


try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_platonic_twist_add(bpy.types.Operator):
        """Shrunken Platonic faces joined by twisted ribbons (after
        Hart's Platonic Twist sculptures)"""
        bl_idname = "mesh.platonic_twist_add"
        bl_label = "Platonic Twist"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Solid",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", "")],
            default='CUBE')
        shrink: FloatProperty(name="Face Shrink", default=0.55,
                              min=0.1, max=0.95)
        push: FloatProperty(
            name="Face Push", default=0.35, min=0.0, max=2.0,
            description="Separation of the faces from the solid")
        half_twists: IntProperty(
            name="Half Twists", default=1, min=0, max=4,
            description="Half turns of each connecting ribbon")
        bulge: FloatProperty(name="Ribbon Bulge", default=0.35,
                             min=0.0, max=1.5)
        band_rows: IntProperty(name="Ribbon Rows", default=16,
                               min=4, max=64)
        edge_cols: IntProperty(name="Ribbon Columns", default=6,
                               min=2, max=24)
        thickness: FloatProperty(
            name="Thickness", default=0.04, min=0.0, max=0.5,
            description="Solidify modifier thickness (0 = pure surface)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            verts, faces, ne, rim = build_platonic_twist(
                self.kind, self.shrink, self.push, self.half_twists,
                self.band_rows, self.edge_cols, self.bulge, self.scale)
            me = bpy.data.meshes.new("PlatonicTwist")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            import bmesh
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            # plate-ribbon junctions are physical creases: mark them
            # sharp so the flat plates are not shaded with ribbon normals
            sharp = [(e.vertices[0] in rim and e.vertices[1] in rim)
                     for e in me.edges]
            me.edges.foreach_set('use_edge_sharp', sharp)
            me.update()
            obj = bpy.data.objects.new("PlatonicTwist", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                # complex mode handles the Moebius-like (non-orientable)
                # surfaces made by odd numbers of half twists
                mod.solidify_mode = 'NON_MANIFOLD'
                mod.nonmanifold_thickness_mode = 'FIXED'
                mod.thickness = self.thickness
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{ne} ribbons")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'shrink', 'push', 'half_twists', 'bulge',
                      'band_rows', 'edge_cols', 'thickness', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.platonic_twist_add", icon='MOD_SCREW')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_platonic_twist_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_platonic_twist_add)


if __name__ == "__main__" and _IN_BLENDER:
    register()
