
# Celtic Knot generator for Blender: strands weaving over and
# under along the edges of a framework mesh -- a port of Adam
# Newgas's "Celtic Knot" add-on
# (github.com/BorisTheBrave/celtic-knot, MIT license) adapted for
# the Math Art extension.
#
# The framework can be one of the built-in Platonic seeds (so the
# operator works straight from the Add menu) or the active mesh
# object.  Weave types: CELTIC (plain weaving -- every crossing
# alternates) and TWILL (over two, under two; heuristic edge
# coloring after Akleman et al., "Cyclic Twill-Woven Objects",
# 2011).  Optional remeshing (edge subdivision / medial) refines
# the pattern; outputs are bezier strands, beveled pipes, or flat
# ribbon meshes, with optional per-strand / per-braid coloring.
#
# References:
# - Adam Newgas, "Celtic Knot" Blender add-on (original algorithm),
#   github.com/BorisTheBrave/celtic-knot, MIT license.
# - Twill weaving after E. Akleman, J. Chen, Q. Xing & J. L. Gross,
#   "Cyclic Twill-Woven Objects", Computers & Graphics 35(3), 2011,
#   pp. 623-631.
# - Mathematical background on Celtic interlace: Peter R. Cromwell,
#   "Celtic Knotwork: Mathematical Art", Math. Intelligencer 15(1),
#   1993, pp. 36-47.

bl_info = {
    "name": "Celtic Knot (Math Art)",
    "author": "Adam Newgas (original), Math Art project (port)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Celtic Knot",
    "description": "Celtic and twill weavings over a framework "
                   "mesh (after BorisTheBrave/celtic-knot, MIT)",
    "category": "Add Curve",
}

from collections import defaultdict
from math import pi, sin, cos, sqrt
from random import random, seed, choice

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

TWIST_CW = "TWIST_CW"
STRAIGHT = "STRAIGHT"
TWIST_CCW = "TWIST_CCW"
IGNORE = "IGNORE"


def is_boundary(loop):
    return len(loop.link_loops) == 0


def lerp(v1, v2, t):
    return v1 * (1 - t) + v2 * t


def cyclic_zip(items):
    it = iter(items)
    first = prev = next(it)
    for item in it:
        yield prev, item
        prev = item
    yield prev, first


def edge_midpoint(edge):
    return (edge.verts[0].co + edge.verts[1].co) / 2.0


def bmesh_from_pydata(vertices, faces):
    bm = bmesh.new()
    for v in vertices:
        bm.verts.new(v)
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    for f in faces:
        bm.faces.new([bm.verts[v] for v in f])
    bm.edges.index_update()
    bm.edges.ensure_lookup_table()
    i = 0
    for edge in bm.edges:
        for loop in edge.link_loops:
            loop.index = i
            i += 1
    return bm


# ---------------------------------------------------------------- #
#  remeshing (pre-processing the framework)                        #
# ---------------------------------------------------------------- #

def remesh_midedge_subdivision(bm):
    e2n = {}
    v2n = {}
    verts = []
    faces = []
    for vert in bm.verts:
        v2n[vert.index] = len(verts)
        verts.append(vert.co)
    for edge in bm.edges:
        e2n[edge.index] = len(verts)
        verts.append(edge_midpoint(edge))
    for face in bm.faces:
        nf = []
        for loop in face.loops:
            nf.append(v2n[loop.vert.index])
            nf.append(e2n[loop.edge.index])
        faces.append(nf)
    return bmesh_from_pydata(verts, faces)


def remesh_medial(bm):
    e2n = {}
    v2n = {}
    verts = []
    faces = []
    for vert in bm.verts:
        v2n[vert.index] = len(verts)
        verts.append(vert.co)
    for edge in bm.edges:
        e2n[edge.index] = len(verts)
        verts.append(edge_midpoint(edge))
    for face in bm.faces:
        faces.append([e2n[loop.edge.index] for loop in face.loops])
    for face in bm.faces:
        for l1, l2 in cyclic_zip(list(face.loops)):
            faces.append([v2n[l2.vert.index],
                          e2n[l2.edge.index],
                          e2n[l1.edge.index]])
    return bmesh_from_pydata(verts, faces)


def remesh(bm, remesh_type):
    if remesh_type == "EDGE_SUBDIVIDE":
        return remesh_midedge_subdivision(bm)
    if remesh_type == "MEDIAL":
        return remesh_medial(bm)
    return bm


# ---------------------------------------------------------------- #
#  strand walking                                                  #
# ---------------------------------------------------------------- #

class DirectedLoop:
    """An edge loop plus a facing along it."""

    def __init__(self, loop, forward):
        self.loop = loop
        self.forward = forward

    @property
    def reversed(self):
        return DirectedLoop(self.loop, not self.forward)

    @property
    def next_face_loop(self):
        loop = self.loop
        forward = self.forward
        while True:
            loop = (loop.link_loop_next if forward
                    else loop.link_loop_prev)
            if not is_boundary(loop):
                break
        return DirectedLoop(loop, forward)

    @property
    def next_edge_loop(self):
        loop = self.loop
        forward = self.forward
        v = loop.vert.index
        loop = loop.link_loops[0 if forward else -1]
        forward = (loop.vert.index == v) == forward
        return DirectedLoop(loop, forward)


def get_celtic_twists(bm, twist_prob):
    """A twist per edge for plain (celtic) weavings."""
    seed(0)
    twists = []
    for edge in bm.edges:
        if len(edge.link_loops) == 0:
            twists.append(IGNORE)
        elif random() < twist_prob:
            twists.append(TWIST_CW)
        else:
            twists.append(STRAIGHT)
    return twists


def strand_part(prev_loop, loop, forward):
    return forward, frozenset((prev_loop.index, loop.index))


class StrandAnalysisBuilder:
    """Which strand parts belong to which strands / braids."""

    def __init__(self):
        self.crossings = defaultdict(list)
        self.current = 0
        self.strand_indices = {}
        self.strand_size = defaultdict(int)

    def start_strand(self):
        pass

    def add_loop(self, prev_loop, loop, twist, forward):
        if twist != STRAIGHT:
            self.crossings[loop.edge.index].append(self.current)
        key = strand_part(prev_loop, loop, forward)
        self.strand_indices[key] = self.current
        self.strand_size[self.current] += 1

    def end_strand(self):
        self.current += 1

    def all_crossings(self):
        return set(frozenset((x, y))
                   for ll in self.crossings.values()
                   for x in ll for y in ll if x != y)

    def get_strands(self):
        return self.strand_indices

    def get_strand_sizes(self):
        return self.strand_size

    def get_braids(self):
        """Partition strands so crossing strands differ."""
        crossings = self.all_crossings()
        braids = {}
        braid_count = 0
        for s in range(self.current):
            crossed = set(braids[t] for p in crossings if s in p
                          for t in p if t in braids)
            for b in range(braid_count):
                if b not in crossed:
                    break
            else:
                b = braid_count
                braid_count += 1
            braids[s] = b
        return {k: braids[v]
                for k, v in self.strand_indices.items()}


def get_medial_twill_twists(bm, orig_face_len):
    """Twists per edge for a medial-remeshed framework."""
    twists = [TWIST_CW] * len(bm.edges)
    bm.faces.ensure_lookup_table()
    for face in bm.faces[0:orig_face_len]:
        for edge in face.edges:
            twists[edge.index] = TWIST_CCW
    return twists


def get_twill_twists(bm):
    """Over-two-under-two edge coloring by heuristic voting,
    after Akleman et al. 2011."""
    seed(0)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    def move(d):
        return d.next_face_loop.next_edge_loop

    def swap(d):
        return d.next_edge_loop

    class Votes:
        def __init__(self, cw=0, ccw=0):
            self.cw = cw
            self.ccw = ccw

        def __add__(self, other):
            return Votes(self.cw + other.cw, self.ccw + other.ccw)

    def edge_cond_vote(dloop):
        nxt = move(dloop)
        nxt2 = move(nxt)
        t1 = coloring[nxt.loop.edge.index]
        t2 = coloring[nxt2.loop.edge.index]
        if t1 is None or t2 is None:
            return Votes()
        if t1 is TWIST_CW and t2 is TWIST_CW:
            return Votes(0, 1)
        if t1 is TWIST_CCW and t2 is TWIST_CCW:
            return Votes(1, 0)
        return Votes(1, 0) if t1 is TWIST_CW else Votes(0, 1)

    def face_cond_vote(dloop):
        s = move(dloop)
        p = move(swap(dloop.reversed))
        f = move(swap(s))
        ts = coloring[s.loop.edge.index]
        tp = coloring[p.loop.edge.index]
        tf = coloring[f.loop.edge.index]
        if ts is None or tp is None or tf is None:
            return Votes()
        if tp != tf:
            return Votes(1, 0) if ts is TWIST_CW else Votes(0, 1)
        return Votes(1, 1)

    def vert_cond_vote(dloop):
        s = move(dloop)
        p = move(swap(dloop))
        f = move(swap(s.reversed))
        ts = coloring[s.loop.edge.index]
        tp = coloring[p.loop.edge.index]
        tf = coloring[f.loop.edge.index]
        if ts is None or tp is None or tf is None:
            return Votes()
        if tp != tf:
            return Votes(1, 0) if ts is TWIST_CW else Votes(0, 1)
        return Votes(1, 1)

    def count_votes(edge_index):
        edge = bm.edges[edge_index]
        votes = Votes()
        for loop in edge.link_loops:
            votes += edge_cond_vote(DirectedLoop(loop, True))
            votes += edge_cond_vote(DirectedLoop(loop, False))
            votes += face_cond_vote(DirectedLoop(loop, True))
            votes += face_cond_vote(DirectedLoop(loop, False))
            votes += vert_cond_vote(DirectedLoop(loop, True))
            votes += vert_cond_vote(DirectedLoop(loop, False))
        return votes

    frontier = set()
    coloring = [None] * len(bm.edges)
    cached = {}

    def color_edge(edge, twist):
        frontier.discard(edge.index)
        coloring[edge.index] = twist
        for v in edge.verts:
            for other in v.link_edges:
                if coloring[other.index] is None:
                    frontier.add(other.index)
        cached.pop(edge.index, None)
        for v1 in edge.verts:
            for e2 in v1.link_edges:
                for v2 in e2.verts:
                    if v1.index == v2.index:
                        continue
                    for e3 in v2.link_edges:
                        cached.pop(e3.index, None)

    def get_vote(edge_index):
        if edge_index not in cached:
            cached[edge_index] = count_votes(edge_index)
        return cached[edge_index]

    while True:
        uncolored = [i for i, c in enumerate(coloring)
                     if c is None]
        if not uncolored:
            break
        v0 = choice(bm.edges[choice(uncolored)].verts)
        for e in v0.link_edges:
            color_edge(e, TWIST_CW)
            break
        while frontier:
            while True:
                found = False
                for e in list(frontier):
                    edge = bm.edges[e]
                    if is_boundary(edge.link_loops[0]):
                        color_edge(edge, IGNORE)
                        found = True
                if not found:
                    break
            votes = {e: get_vote(e) for e in frontier}
            m = max(max(v.cw, v.ccw) for v in votes.values())
            best_edge, best = choice(
                [(k, v) for k, v in votes.items()
                 if v.cw == m or v.ccw == m])
            color_edge(bm.edges[best_edge],
                       TWIST_CW if best.cw > best.ccw
                       else TWIST_CCW)
    assert all(coloring), "twill coloring failed"
    return coloring


def get_offset(weave_up, weave_down, twist, forward):
    """Signed offset along the surface normal: the strand passing
    over lifts by +weave_up, the one passing under drops by
    -weave_down (the original add-on used one signed pair; here
    both properties are positive distances)."""
    if twist is TWIST_CW:
        return -weave_down if forward else weave_up
    if twist is TWIST_CCW:
        return weave_up if forward else -weave_down
    if twist is STRAIGHT:
        return (weave_up - weave_down) / 2.0
    raise AssertionError(f"unexpected twist {twist}")


def visit_strands(bm, twists, builder):
    """Walks the mesh strand by strand, turning at each edge by
    the given twists, calling the builder per edge crossed."""
    loops_entered = defaultdict(lambda: False)
    loops_exited = defaultdict(lambda: False)

    def make_loop(d):
        builder.start_strand()
        while True:
            if d.forward:
                if loops_exited[d.loop]:
                    break
                loops_exited[d.loop] = True
                d = d.next_face_loop
                loops_entered[d.loop] = True
            else:
                if loops_entered[d.loop]:
                    break
                loops_entered[d.loop] = True
                d = d.next_face_loop
                loops_exited[d.loop] = True
            prev_loop = d.loop
            twist = twists[d.loop.edge.index]
            if twist in (TWIST_CCW, TWIST_CW):
                d = d.next_edge_loop
            builder.add_loop(prev_loop, d.loop, twist, d.forward)
        builder.end_strand()

    for face in bm.faces:
        for loop in face.loops:
            if is_boundary(loop):
                continue
            if not loops_exited[loop]:
                make_loop(DirectedLoop(loop, True))
            if not loops_entered[loop]:
                make_loop(DirectedLoop(loop, False))


# ---------------------------------------------------------------- #
#  builders                                                        #
# ---------------------------------------------------------------- #

if _IN_BLENDER:

    def _make_material(name, rgb):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*rgb, 1.0)
        mat.use_nodes = True
        node = mat.node_tree.nodes.get("Principled BSDF")
        if node is not None:
            node.inputs["Base Color"].default_value = (*rgb, 1.0)
            node.inputs["Roughness"].default_value = 0.45
        return mat

    def _setup_materials(materials_array, materials):
        if materials is None:
            return
        import colorsys
        n = len(set(materials.values()))
        for i in range(n):
            rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.72, 0.85)
            materials_array.append(
                _make_material(f"Celtic Strand {i + 1}", rgb))

    class BezierBuilder:
        """A bezier spline per strand (materials are appended to
        the curve BEFORE the splines exist, because
        spline.material_index is clamped to the material count at
        assignment time)."""

        def __init__(self, bm, crossing_angle, crossing_strength,
                     handle_type, weave_up, weave_down,
                     materials=None):
            self.s = sin(crossing_angle) * crossing_strength
            self.c = cos(crossing_angle) * crossing_strength
            self.handle_type = handle_type
            self.weave_up = weave_up
            self.weave_down = weave_down
            self.curve = bpy.data.curves.new("Celtic Knot",
                                             "CURVE")
            self.curve.dimensions = "3D"
            self.curve.twist_mode = "MINIMUM"
            _setup_materials(self.curve.materials, materials)
            self.midpoints = [edge_midpoint(e) for e in bm.edges]
            self.spline = None
            self.cos = None
            self.lefts = None
            self.rights = None
            self.first = True
            self.materials = materials or defaultdict(int)
            self.material = None

        def start_strand(self):
            self.spline = self.curve.splines.new("BEZIER")
            self.spline.use_cyclic_u = True
            self.cos = []
            self.lefts = []
            self.rights = []
            self.material = None
            self.first = True

        def add_loop(self, prev_loop, loop, twist, forward):
            if not self.first:
                self.spline.bezier_points.add(1)
            self.first = False
            midpoint = self.midpoints[loop.edge.index]
            normal = loop.calc_normal() + prev_loop.calc_normal()
            normal.normalize()
            midpoint = midpoint + get_offset(
                self.weave_up, self.weave_down, twist,
                forward) * normal
            self.cos.extend(midpoint)
            self.material = self.materials[
                strand_part(prev_loop, loop, forward)]
            if self.handle_type != "AUTO":
                tangent = (loop.link_loop_next.vert.co
                           - loop.vert.co)
                tangent.normalize()
                binormal = normal.cross(tangent).normalized()
                if not forward:
                    tangent *= -1
                sb = self.s * binormal
                ct = self.c * tangent
                self.lefts.extend(midpoint - sb - ct)
                self.rights.extend(midpoint + sb + ct)

        def end_strand(self):
            points = self.spline.bezier_points
            points.foreach_set("co", self.cos)
            if self.handle_type != "AUTO":
                points.foreach_set("handle_left", self.lefts)
                points.foreach_set("handle_right", self.rights)
            for bp in points:
                bp.handle_left_type = self.handle_type
                bp.handle_right_type = self.handle_type
            self.spline.material_index = self.material

    class RibbonBuilder:
        """A flat polygonal ribbon per strand."""

        def __init__(self, weave_up, weave_down, length, breadth,
                     materials=None):
            self.weave_up = weave_up
            self.weave_down = weave_down
            self.vertices = []
            self.faces = []
            self.prev_out = None
            self.first_in = None
            self.prev_material = None
            self.c = length
            self.w = breadth
            self.materials = materials or defaultdict(int)
            self.material_values = []

        def get_sub_face(self, v1, v2, v3, v4):
            hc = self.c / 2.0
            hw = self.w / 2.0
            return (
                lerp(lerp(v1, v4, 0.5 - hc),
                     lerp(v2, v3, 0.5 - hc), 0.5 - hw),
                lerp(lerp(v1, v4, 0.5 - hc),
                     lerp(v2, v3, 0.5 - hc), 0.5 + hw),
                lerp(lerp(v1, v4, 0.5 + hc),
                     lerp(v2, v3, 0.5 + hc), 0.5 + hw),
                lerp(lerp(v1, v4, 0.5 + hc),
                     lerp(v2, v3, 0.5 + hc), 0.5 - hw),
            )

        strand_count = 0

        def start_strand(self):
            self.first_in = None
            self.prev_out = None
            self.prev_material = None
            self.strand_count += 1

        def add_loop(self, prev_loop, loop, twist, forward):
            normal = loop.calc_normal() + prev_loop.calc_normal()
            normal.normalize()
            offset = get_offset(self.weave_up, self.weave_down,
                                twist, forward) * normal
            center1 = prev_loop.face.calc_center_median()
            center2 = loop.face.calc_center_median()
            v1 = loop.vert.co
            v2 = loop.link_loop_next.vert.co
            if twist is STRAIGHT:
                if forward:
                    v1, center1, v2, center2 = (center1, v1, v2,
                                                center1)
                else:
                    v1, center1, v2, center2 = (v2, center1,
                                                center1, v1)
            elif not forward:
                v1, center1, v2, center2 = (center1, v2, center2,
                                            v1)
            v1, center1, v2, center2 = self.get_sub_face(
                v1, center1, v2, center2)
            sp = strand_part(prev_loop, loop, forward)
            self.prev_material = material = self.materials[sp]
            i = len(self.vertices)
            self.vertices.extend([v1 + offset, center1 + offset,
                                  v2 + offset, center2 + offset])
            self.faces.append([i, i + 1, i + 2])
            self.faces.append([i, i + 2, i + 3])
            self.material_values.extend([material, material])
            in_verts = [i + 1, i + 0]
            out_verts = [i + 3, i + 2]
            if self.first_in is None:
                self.first_in = in_verts
            if self.prev_out is not None:
                self.faces.append(self.prev_out + in_verts)
                self.material_values.append(material)
            self.prev_out = out_verts

        def end_strand(self):
            self.faces.append(self.prev_out + self.first_in)
            self.material_values.append(self.prev_material)

        def make_mesh(self, materials):
            me = bpy.data.meshes.new("Celtic Knot")
            me.from_pydata(self.vertices, [], self.faces)
            me.validate(clean_customdata=True)
            _setup_materials(me.materials, materials)
            if len(me.polygons) == len(self.material_values):
                me.polygons.foreach_set("material_index",
                                        self.material_values)
            me.update(calc_edges=True)
            return me

    # ------------------------------------------------------------ #
    #  operator                                                    #
    # ------------------------------------------------------------ #

    _PHI = (1.0 + sqrt(5.0)) / 2.0

    def _seed_mesh(name):
        try:
            from . import spiked_polyhedron_generator as sp
        except ImportError:
            import spiked_polyhedron_generator as sp
        V, F = sp._seed(name)
        return [tuple(v) for v in V], [list(f) for f in F]

    class CURVE_OT_celtic_knot_add(bpy.types.Operator):
        """Weave celtic or twill knotwork strands over a framework
        mesh (a built-in Platonic seed or the active mesh object);
        after Adam Newgas's celtic-knot add-on"""
        bl_idname = "curve.celtic_knot_add"
        bl_label = "Celtic Knot"
        bl_options = {'REGISTER', 'UNDO'}

        source: EnumProperty(
            name="Framework",
            items=[('ICOSA', "Icosahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('TETRA', "Tetrahedron", ""),
                   ('ACTIVE', "Active Object",
                    "Weave over the active mesh object")],
            default='ICOSA',
            description="Mesh the strands weave along")
        remesh_type: EnumProperty(
            name="Remesh",
            items=[("NONE", "None", ""),
                   ("EDGE_SUBDIVIDE", "Edge Subdivide",
                    "Subdivide every edge first"),
                   ("MEDIAL", "Medial",
                    "Replace every vertex with a fan of faces")],
            default="NONE",
            description="Pre-process the framework before weaving")
        weave_type: EnumProperty(
            name="Weave",
            items=[("CELTIC", "Celtic",
                    "Plain weaving: alternating crossings"),
                   ("TWILL", "Twill",
                    "Over two then under two (heuristic)")],
            default="CELTIC")
        twist_proportion: FloatProperty(
            name="Twist Proportion", default=1.0, min=0.0, max=1.0,
            description="Fraction of edges that cross (Celtic "
                        "only); the rest pass straight through")
        output: EnumProperty(
            name="Output",
            items=[("BEZIER", "Bezier Strands", "Bare curves"),
                   ("PIPE", "Pipes",
                    "Curves beveled into round tubes"),
                   ("RIBBON", "Ribbons", "Flat ribbon mesh")],
            default="PIPE")
        weave_up: FloatProperty(
            name="Weave Up", default=0.12, min=0.0, max=2.0,
            description="Lift of a strand crossing over")
        weave_down: FloatProperty(
            name="Weave Down", default=0.12, min=0.0, max=2.0,
            description="Drop of a strand crossing under")
        handle_type: EnumProperty(
            name="Handles",
            items=[("AUTO", "Auto", "Automatic control points"),
                   ("ALIGNED", "Aligned",
                    "Fixed crossing angle control points")],
            default="AUTO")
        crossing_angle: FloatProperty(
            name="Crossing Angle", default=pi / 4, min=0.0,
            max=pi / 2, subtype='ANGLE',
            description="Aligned handles: angle between strands "
                        "at a crossing")
        crossing_strength: FloatProperty(
            name="Crossing Strength", default=0.25, min=0.0,
            max=5.0,
            description="Aligned handles: control point length")
        thickness: FloatProperty(
            name="Pipe Radius", default=0.05, min=0.001, max=1.0,
            description="Bevel radius of the pipes")
        resolution: IntProperty(
            name="Bevel Resolution", default=4, min=1, max=12)
        length: FloatProperty(
            name="Ribbon Length", default=0.9, min=0.05, max=1.0,
            description="Fraction along faces the ribbon runs")
        breadth: FloatProperty(
            name="Ribbon Breadth", default=0.5, min=0.05, max=1.0,
            description="Ribbon width as a fraction across faces")
        coloring: EnumProperty(
            name="Coloring",
            items=[("STRAND", "Per Strand",
                    "A distinct material per strand"),
                   ("BRAID", "Per Braid",
                    "As few materials as possible with no two "
                    "crossing strands sharing one"),
                   ("NONE", "None", "")],
            default="STRAND")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Built-in frameworks: "
                                         "half-size of the "
                                         "bounding cube")

        def execute(self, context):
            bm = bmesh.new()
            if self.source == 'ACTIVE':
                obj = context.active_object
                if obj is None or obj.type != 'MESH':
                    self.report({'ERROR'},
                                "no active mesh object to weave "
                                "over (pick a built-in framework "
                                "instead)")
                    bm.free()
                    return {'CANCELLED'}
                bm.from_mesh(obj.data)
            else:
                V, F = _seed_mesh(self.source)
                for v in V:
                    bm.verts.new(tuple(c * self.scale for c in v))
                bm.verts.ensure_lookup_table()
                bm.verts.index_update()
                for f in F:
                    bm.faces.new([bm.verts[i] for i in f])
            bm.edges.index_update()
            bm.edges.ensure_lookup_table()
            i = 0
            for edge in bm.edges:
                for loop in edge.link_loops:
                    loop.index = i
                    i += 1
            orig_faces = len(bm.faces)
            bm = remesh(bm, self.remesh_type)

            if self.weave_type == "CELTIC":
                twists = get_celtic_twists(bm,
                                           self.twist_proportion)
            elif self.remesh_type == "MEDIAL":
                twists = get_medial_twill_twists(bm, orig_faces)
            else:
                twists = get_twill_twists(bm)

            materials = None
            if self.coloring != "NONE":
                analysis = StrandAnalysisBuilder()
                visit_strands(bm, twists, analysis)
                materials = (analysis.get_strands()
                             if self.coloring == "STRAND"
                             else analysis.get_braids())

            if self.output == "RIBBON":
                builder = RibbonBuilder(self.weave_up,
                                        self.weave_down,
                                        self.length, self.breadth,
                                        materials)
                visit_strands(bm, twists, builder)
                data = builder.make_mesh(materials)
                nstr = builder.strand_count
            else:
                builder = BezierBuilder(
                    bm, self.crossing_angle,
                    self.crossing_strength, self.handle_type,
                    self.weave_up, self.weave_down, materials)
                visit_strands(bm, twists, builder)
                data = builder.curve
                if self.output == "PIPE":
                    data.bevel_depth = self.thickness
                    data.bevel_resolution = self.resolution
                nstr = len(data.splines)
            bm.free()
            obj = bpy.data.objects.new("Celtic Knot", data)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{nstr} strand(s)")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'source')
            lay.prop(self, 'remesh_type')
            lay.prop(self, 'weave_type')
            if self.weave_type == 'CELTIC':
                lay.prop(self, 'twist_proportion')
            lay.prop(self, 'output')
            lay.prop(self, 'weave_up')
            lay.prop(self, 'weave_down')
            if self.output in ('BEZIER', 'PIPE'):
                lay.prop(self, 'handle_type')
                if self.handle_type != 'AUTO':
                    lay.prop(self, 'crossing_angle')
                    lay.prop(self, 'crossing_strength')
            if self.output == 'PIPE':
                lay.prop(self, 'thickness')
                lay.prop(self, 'resolution')
            if self.output == 'RIBBON':
                lay.prop(self, 'length')
                lay.prop(self, 'breadth')
            lay.prop(self, 'coloring')
            if self.source != 'ACTIVE':
                lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.celtic_knot_add",
                             icon='MOD_LATTICE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(CURVE_OT_celtic_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_celtic_knot_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        print("celtic knot generator: the strand walk needs "
              "Blender's bmesh; run the extension test suite for "
              "coverage")
