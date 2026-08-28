# A "part" is what actually gets cut: one outer ring plus the rings
# nested inside it.
#
# The distinction matters more than it looks.  A plane section returns
# unordered closed loops, and it is tempting to treat each loop as a
# piece -- but a slice through a torus is an annulus, two loops that are
# ONE piece of material with a hole in it.  Emit them as two parts and
# the layout nests the hole somewhere else on the sheet, the cut order
# is wrong, and the operator reports twice as many pieces as exist.
#
# Nesting is decided by containment parity: a ring inside an odd number
# of other rings is a hole, inside an even number it is an outline.  The
# containment test samples a ring vertex, which is safe here because
# section loops of a closed surface never touch one another -- two rings
# that shared a point would have been one loop.
#
# Winding is normalised on the way out: outlines CCW, holes CW.  Both
# exporters and the area arithmetic rely on that, so it is done once
# here rather than defended everywhere downstream.

from . import polyclip as pc


class Part:
    """One cuttable piece: an outline, its holes, and its paperwork."""

    __slots__ = ('outer', 'holes', 'family', 'slice_index', 'index',
                 'label', 'errors', 'slots', 'offset', 'order')

    def __init__(self, outer, holes=None, family='', slice_index=0,
                 index=0, offset=0.0):
        self.outer = pc.as_ccw(outer)
        self.holes = [pc.as_cw(h) for h in (holes or [])]
        self.family = family
        self.slice_index = slice_index
        self.index = index
        self.offset = offset
        self.label = ''
        self.order = 0
        self.errors = []
        self.slots = []

    # -- geometry ---------------------------------------------------

    def area(self):
        return pc.area(self.outer) - sum(pc.area(h) for h in self.holes)

    def bounds(self):
        return pc.bounds(self.outer)

    def size(self):
        x0, y0, x1, y1 = self.bounds()
        return (x1 - x0, y1 - y0)

    def rings(self):
        """Outline first, then holes."""
        return [self.outer] + list(self.holes)

    def translated(self, dx, dy):
        out = Part([(x + dx, y + dy) for x, y in self.outer],
                   [[(x + dx, y + dy) for x, y in h] for h in self.holes],
                   self.family, self.slice_index, self.index, self.offset)
        out.label = self.label
        out.errors = list(self.errors)
        out.slots = [[(x + dx, y + dy) for x, y in s] for s in self.slots]
        return out

    def fail(self, kind, detail=''):
        self.errors.append((kind, detail))

    def __repr__(self):
        return (f"<Part {self.label or '?'} "
                f"{len(self.outer)}v {len(self.holes)}h "
                f"{len(self.errors)}err>")


def build_parts(loops, family='', slice_index=0, offset=0.0):
    """Group unordered section loops into parts by containment parity."""
    rings = [list(L) for L in loops if len(L) >= 3]
    if not rings:
        return []

    depth = [0] * len(rings)
    for i, ri in enumerate(rings):
        probe = ri[0]
        for j, rj in enumerate(rings):
            if i != j and pc.point_in_polygon(probe, rj):
                depth[i] += 1

    outers = [i for i in range(len(rings)) if depth[i] % 2 == 0]
    holes = [i for i in range(len(rings)) if depth[i] % 2 == 1]

    parts = []
    for k, i in enumerate(outers):
        mine = []
        for j in holes:
            # a hole belongs to the SMALLEST outline containing it
            if not pc.point_in_polygon(rings[j][0], rings[i]):
                continue
            best = min(
                (o for o in outers
                 if pc.point_in_polygon(rings[j][0], rings[o])),
                key=lambda o: pc.area(rings[o]))
            if best == i:
                mine.append(rings[j])
        parts.append(Part(rings[i], mine, family, slice_index, k, offset))
    return parts


def region_intervals(part, origin, direction):
    """Intervals of a line inside the part's MATERIAL (outline minus
    holes).

    Crossings of every ring -- outline and holes alike -- are pooled and
    paired.  That is correct because each crossing of any boundary
    toggles inside/outside, so a line entering through the outline and
    leaving through a hole gives the interval that really is material.
    Treating the outline alone would slot straight across a hole.
    """
    ts = []
    for ring in part.rings():
        for a, b in pc.line_intervals(ring, origin, direction):
            ts.extend((a, b))
    ts.sort()
    if len(ts) % 2:
        raise pc.DegenerateClip("odd crossing count over part rings")
    return [(ts[k], ts[k + 1]) for k in range(0, len(ts), 2)]


# ------------------------------------------------------------------ #

def _selftest():
    outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    hole = [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)]

    parts = build_parts([outer, hole])
    assert len(parts) == 1, \
        f"an annulus is ONE part with a hole, got {len(parts)} parts"
    p = parts[0]
    assert len(p.holes) == 1, "the inner ring is a hole"
    assert abs(p.area() - (16.0 - 4.0)) < 1e-12, f"area {p.area()}"
    assert pc.signed_area(p.outer) > 0, "outline normalised CCW"
    assert pc.signed_area(p.holes[0]) < 0, "hole normalised CW"

    # two disjoint rings are two parts, not one with a hole
    far = [(9.0, 9.0), (10.0, 9.0), (10.0, 10.0), (9.0, 10.0)]
    assert len(build_parts([outer, far])) == 2, "disjoint rings"

    # nesting three deep: ring inside hole is a part again
    inner = [(1.4, 1.4), (2.6, 1.4), (2.6, 2.6), (1.4, 2.6)]
    ps = build_parts([outer, hole, inner])
    assert len(ps) == 2, f"island inside a hole is its own part, got {len(ps)}"

    # three nested rings: even-odd makes the MIDDLE one a hole in the
    # outermost, and the innermost an island in its own right
    big = [(-1.0, -1.0), (5.0, -1.0), (5.0, 5.0), (-1.0, 5.0)]
    ps = build_parts([big, outer, hole])
    assert len(ps) == 2, f"outermost + island, got {len(ps)}"
    owner = [q for q in ps if q.holes]
    assert len(owner) == 1 and abs(pc.area(owner[0].outer) - 36.0) < 1e-9, \
        "the middle ring is a hole in the outermost ring"

    # four deep: a hole must attach to the SMALLEST outline containing
    # it -- the island -- and not jump out to the outermost ring
    pin = [(1.8, 1.8), (2.2, 1.8), (2.2, 2.2), (1.8, 2.2)]
    ps = build_parts([big, outer, hole, pin])
    assert len(ps) == 2, f"outermost + island, got {len(ps)}"
    island = min(ps, key=lambda q: pc.area(q.outer))
    assert len(island.holes) == 1, \
        "the innermost ring is a hole in the island, not in the outer part"
    outermost = max(ps, key=lambda q: pc.area(q.outer))
    assert len(outermost.holes) == 1, "outermost keeps exactly its own hole"

    # region_intervals must respect the hole, not slot across it
    iv = region_intervals(parts[0], (-1.0, 2.0), (1.0, 0.0))
    assert len(iv) == 2, f"a line across an annulus is two spans, got {iv}"
    assert abs(iv[0][0] - 1.0) < 1e-9 and abs(iv[0][1] - 2.0) < 1e-9, iv
    assert abs(iv[1][0] - 4.0) < 1e-9 and abs(iv[1][1] - 5.0) < 1e-9, iv

    # translation keeps everything consistent
    q = parts[0].translated(10.0, 0.0)
    assert abs(q.area() - parts[0].area()) < 1e-12, "translate preserves area"
    assert abs(q.bounds()[0] - 10.0) < 1e-12, "translate moves bounds"

    return True
