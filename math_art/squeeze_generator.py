
# Polyhedron Squeeze Surface generator for Blender, inspired by
# Robert Fathauer's "Cubic Squeeze" sculpture: a polyhedron whose
# edges are replaced by curves bending inward along one of their
# two faces, with minimal surfaces spanning the bent frames.
#
# The bend pattern: every edge is claimed by exactly ONE of its
# two adjacent faces and bows toward that face's centre, in that
# face's plane.  Around each face the claimed edges alternate
# with edges claimed by the neighbours -- on a square face the
# claimed pair are opposite edges squeezing toward each other
# (the "squeeze"), and the alternation is only possible when
# every face has an EVEN number of edges.  The generator solves
# the alternation by parity propagation over the face graph and
# reports seeds where no consistent pattern exists.
#
# Each face is then spanned by a membrane over its bent frame
# (concentric rings relaxed by the toolkit's Plateau area
# minimizer, uniform Laplacian as fallback).  The frames' sample
# points are shared verbatim between neighbouring faces, so the
# welded result is watertight with crisp creases along the bent
# edges.

bl_info = {
    "name": "Polyhedron Squeeze Surface",
    "author": "Math Art project (after Robert Fathauer)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Polyhedra with edges squeezed inward and "
                   "minimal surfaces spanning the bent frames",
    "category": "Add Mesh",
}

import math
from collections import deque

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def squeeze_assignment(F, alternate=False):
    """For each face, which of its edge slots it claims: dict
    (face index, slot) -> bool, such that every edge is claimed
    by exactly one of its two faces and claims alternate around
    every face.  Solved by parity propagation; raises ValueError
    when no consistent pattern exists (odd faces, or an
    incompatible cycle)."""
    if any(len(f) % 2 for f in F):
        raise ValueError("every face needs an even number of "
                         "edges")
    edge_slots = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            e = frozenset((f[i], f[(i + 1) % len(f)]))
            edge_slots.setdefault(e, []).append((fi, i))
    if any(len(s) != 2 for s in edge_slots.values()):
        raise ValueError("seed must be a closed manifold mesh")
    # face phase p_f in {0,1}: slot i claimed iff (i + p_f) even.
    # each edge claimed once:  (i + p_f) + (j + p_g) = 1 (mod 2)
    phase = [None] * len(F)
    for start in range(len(F)):
        if phase[start] is not None:
            continue
        phase[start] = 0
        queue = deque([start])
        while queue:
            fi = queue.popleft()
            f = F[fi]
            for i in range(len(f)):
                e = frozenset((f[i], f[(i + 1) % len(f)]))
                (fa, ia), (fb, ib) = edge_slots[e]
                gi, j = (fb, ib) if fa == fi else (fa, ia)
                want = (1 + i + j + phase[fi]) % 2
                if phase[gi] is None:
                    phase[gi] = want
                    queue.append(gi)
                elif phase[gi] != want:
                    raise ValueError(
                        "no consistent squeeze pattern exists "
                        "for this polyhedron")
    if alternate:
        phase = [1 - p for p in phase]
    return {(fi, i): (i + phase[fi]) % 2 == 0
            for fi, f in enumerate(F) for i in range(len(f))}


def build_squeeze(V, F, bend=0.45, m=12, rings=10, iterations=30,
                  alternate=False):
    """(verts, faces, face_ids): the squeeze surface.  Every edge
    bows toward the centre of its claiming face (in that face's
    plane, by `bend` x the midpoint-to-centre distance, with a
    smooth 4t(1-t) profile); each face is spanned by a membrane
    over its bent frame."""
    V = [np.asarray(v, float) for v in V]
    claim = squeeze_assignment(F, alternate)
    centers = [np.mean([V[i] for i in f], axis=0) for f in F]

    # sample every edge once, bowing toward its claiming face
    esamp = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            key = (a, b) if a < b else (b, a)
            if key in esamp or not claim[(fi, i)]:
                continue
            A, B = V[key[0]], V[key[1]]
            mid = (A + B) / 2.0
            d = centers[fi] - mid
            ab = B - A
            ab /= np.linalg.norm(ab) or 1.0
            d = d - np.dot(d, ab) * ab      # in-plane, ortho to AB
            pts = []
            for j in range(m + 1):
                t = j / m
                pts.append(A + (B - A) * t
                           + d * (bend * 4.0 * t * (1.0 - t)))
            esamp[key] = pts

    try:
        try:
            from .minimal_surface_toolkit import minimize_area
        except ImportError:
            from minimal_surface_toolkit import minimize_area
    except Exception:
        minimize_area = None

    vid = {}
    verts = []
    faces = []
    fids = []

    def emit(p, weld):
        if not weld:
            verts.append(tuple(p))
            return len(verts) - 1
        key = tuple(np.round(p, 9))
        if key not in vid:
            vid[key] = len(verts)
            verts.append(tuple(p))
        return vid[key]

    for fi, f in enumerate(F):
        # bent boundary loop (each edge's samples, oriented)
        B = []
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            key = (a, b) if a < b else (b, a)
            pts = esamp[key]
            seq = pts if key[0] == a else pts[::-1]
            B.extend(seq[:-1])
        B = np.array(B)
        nb = len(B)
        cen = B.mean(axis=0)
        R = max(2, rings)
        P = []
        for r in range(R):
            fr = 1.0 - r / R
            P.extend(cen + (B - cen) * fr)
        P.append(cen)
        P = np.array(P)
        tris = []
        for r in range(R - 1):
            for j in range(nb):
                j2 = (j + 1) % nb
                tris.append((r * nb + j, r * nb + j2,
                             (r + 1) * nb + j))
                tris.append((r * nb + j2, (r + 1) * nb + j2,
                             (r + 1) * nb + j))
        last = (R - 1) * nb
        ctr = len(P) - 1
        for j in range(nb):
            tris.append((last + j, last + (j + 1) % nb, ctr))
        fixed = np.zeros(len(P), dtype=bool)
        fixed[:nb] = True
        if minimize_area is not None and iterations > 0:
            minimize_area(P, np.array(tris), fixed,
                          outer_iters=iterations)
        elif iterations > 0:
            nbrs = [set() for _ in range(len(P))]
            for t in tris:
                for q in range(3):
                    nbrs[t[q]].add(t[(q + 1) % 3])
                    nbrs[t[(q + 1) % 3]].add(t[q])
            nbl = [sorted(s) for s in nbrs]
            for _ in range(iterations * 4):
                Q = np.empty_like(P)
                for i, s in enumerate(nbl):
                    Q[i] = P[s].mean(axis=0)
                P[~fixed] += 0.6 * (Q[~fixed] - P[~fixed])
        local = [emit(p, i < nb) for i, p in enumerate(P)]
        for t in tris:
            faces.append([local[i] for i in t])
            fids.append(fi)
    return verts, faces, fids


def _seed(name):
    if name == 'CUBE':
        try:
            from . import spiked_polyhedron_generator as sp
        except ImportError:
            import spiked_polyhedron_generator as sp
        V, F = sp._seed('CUBE')
        return [tuple(v) for v in V], [list(f) for f in F]
    try:
        from . import spacefill_generator as sf
    except ImportError:
        import spacefill_generator as sf
    if name == 'RHOMBIC':
        return ([tuple(v) for v in sf._RD_V],
                [list(f) for f in sf._RD_F])
    if name == 'TRUNCOCT':
        return ([tuple(v) for v in sf._TO_V],
                [list(f) for f in sf._TO_F])
    if name == 'HEXPRISM':
        vs = [(math.cos(2 * math.pi * k / 6),
               math.sin(2 * math.pi * k / 6), z)
              for z in (-0.6, 0.6) for k in range(6)]
        F = [[5, 4, 3, 2, 1, 0], [6, 7, 8, 9, 10, 11]]
        for k in range(6):
            k2 = (k + 1) % 6
            F.append([k, k2, 6 + k2, 6 + k])
        return vs, F
    raise ValueError(name)


if _IN_BLENDER:

    class MESH_OT_squeeze_add(bpy.types.Operator):
        """Squeeze a polyhedron: every edge bows inward along one
        of its faces (alternating around each face, inspired by
        Robert Fathauer's Cubic Squeeze) and minimal surfaces
        span the bent frames; needs faces with an even number of
        edges"""
        bl_idname = "mesh.squeeze_add"
        bl_label = "Polyhedron Squeeze Surface"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Seed",
            items=[('CUBE', "Cube", "the Cubic Squeeze"),
                   ('RHOMBIC', "Rhombic Dodecahedron",
                    "12 rhombic faces"),
                   ('TRUNCOCT', "Truncated Octahedron",
                    "squares and hexagons"),
                   ('HEXPRISM', "Hexagonal Prism", ""),
                   ('ACTIVE', "Active Object",
                    "Squeeze the active mesh object (all faces "
                    "must have an even number of edges)")],
            default='CUBE')
        bend: FloatProperty(
            name="Bend", default=0.45, min=0.0, max=0.95,
            description="Inward bend of each edge, as a fraction "
                        "of the distance from the edge midpoint "
                        "to its face centre")
        samples: IntProperty(
            name="Edge Samples", default=12, min=3, max=48,
            description="Subdivisions of each bent edge")
        rings: IntProperty(
            name="Rings", default=10, min=2, max=48,
            description="Concentric rings toward each face "
                        "centre")
        iterations: IntProperty(
            name="Solver Iterations", default=30, min=0, max=200,
            description="Plateau area-minimization iterations "
                        "per face")
        alternate: BoolProperty(
            name="Alternate Pattern", default=False,
            description="Flip which edge pair each face claims")
        smooth: BoolProperty(name="Smooth Shading", default=False)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
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
                name = f"{src.name} Squeeze"
            else:
                V, F = _seed(self.seed)
                name = ("Cubic Squeeze" if self.seed == 'CUBE'
                        else f"Squeeze ({self.seed.title()})")
            try:
                verts, faces, fids = build_squeeze(
                    V, F, self.bend, self.samples, self.rings,
                    self.iterations, self.alternate)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
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
            attr = me.attributes.new("face_index", 'INT', 'FACE')
            if len(me.polygons) == len(fids):
                attr.data.foreach_set('value', fids)
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
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('seed', 'bend', 'samples', 'rings',
                      'iterations', 'alternate', 'smooth',
                      'thickness', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.squeeze_add",
                             icon='MOD_SIMPLEDEFORM')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_squeeze_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_squeeze_add)


def _selftest():
    CUBE_V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
              for z in (-1, 1)]
    CUBE_F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
              [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    claim = squeeze_assignment(CUBE_F)
    per_face = [sum(claim[(fi, i)] for i in range(4))
                for fi in range(6)]
    edges = {}
    for fi, f in enumerate(CUBE_F):
        for i in range(4):
            e = frozenset((f[i], f[(i + 1) % 4]))
            edges.setdefault(e, 0)
            edges[e] += claim[(fi, i)]
    once = all(c == 1 for c in edges.values())
    # claimed slots alternate around each face
    alt = all(claim[(fi, i)] != claim[(fi, (i + 1) % 4)]
              for fi in range(6) for i in range(4))
    print(f"cube: claims/face={per_face} "
          f"each edge once={once} alternating={alt}")
    assert per_face == [2] * 6 and once and alt
    verts, faces, fids = build_squeeze(
        CUBE_V, CUBE_F, bend=0.45, m=8, rings=6,
        iterations=20)
    cnt = {}
    for f in faces:
        for i in range(len(f)):
            e = frozenset((f[i], f[(i + 1) % len(f)]))
            cnt[e] = cnt.get(e, 0) + 1
    watertight = all(c == 2 for c in cnt.values())
    finite = all(all(math.isfinite(c) for c in v)
                 for v in verts)
    print(f"cube squeeze: V={len(verts)} F={len(faces)} "
          f"watertight={watertight} finite={finite}")
    assert watertight and finite
    # a tetrahedron (odd faces) must be rejected
    TET_F = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    try:
        squeeze_assignment(TET_F)
        raise AssertionError("tetra should be rejected")
    except ValueError:
        print("tetrahedron correctly rejected")
    print("squeeze standalone tests passed")
