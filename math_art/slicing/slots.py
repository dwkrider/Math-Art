# The interlock: two families of slice planes joined by complementary
# half-depth slots.
#
# THE JOINT.  Where a plane of family A meets a plane of family B they
# share a line.  That line enters and leaves the solid at points which,
# because the line lies inside BOTH planes, are on the rim of both
# slices.  Cut A from one rim end inward to the midpoint of the span,
# cut B from the other end inward to the same midpoint, and the two
# pieces pass through one another and stop flush.  Luecking states the
# rule for parallel families -- "the x slices slotted from below and the
# y slices slotted from above ... the point marked at the middle of each
# intersection line marks the extent of the corresponding slot" -- and
# Miller and Akleman give the same rule for planes that are not
# perpendicular.  It is also, exactly, what `slide_together_generator`
# does for Hart's slide-togethers.
#
# ONE SPAN, ONE SOURCE OF TRUTH.  In exact arithmetic the span measured
# in A's section equals the span measured in B's, since both equal
# solid-intersect-line.  In floating point they do not: the two sections
# come from different arithmetic, in different frames, chained
# separately.  So the span is computed ONCE, from A, and handed to B.
# B's own section is used only as a cross-check against a stated
# tolerance, and a disagreement past it is REPORTED as a bad crossing
# rather than split the difference.  Both families are sectioned to
# completion, with their nudged plane offsets committed, before any of
# this runs -- otherwise the two halves of a joint would be discussing
# different planes.
#
# WHY MULTI-INTERVAL CROSSINGS ARE REFUSED.  Where the line meets the
# solid in more than one span, every span yields a perfectly valid pair
# of half-slots and the assembly is still impossible: the parts engage
# by sliding along that line, so B's lower tongue would have to pass
# through A's upper region, which is full-thickness material.  Paper
# flexes past this (which is how Hart's ten-slit pentagrams go
# together); 3 mm plywood does not.  So it is allowed only when the
# clearance setting says the stock is flexible, and reported otherwise.

import math

from . import polyclip as pc
from . import parts as _parts


class SlicePlane:
    """One cutting plane, its frame, and the parts lying in it."""

    __slots__ = ('normal', 'offset', 'u', 'v', 'parts', 'index')

    def __init__(self, normal, offset, u, v, index=0, parts=None):
        self.normal = tuple(float(c) for c in normal)
        self.offset = float(offset)
        self.u = tuple(float(c) for c in u)
        self.v = tuple(float(c) for c in v)
        self.index = index
        self.parts = list(parts or ())


class Family:
    """A named group of cutting planes -- one stack of slices.

    `trimmed` marks a family whose parts were deliberately cut back
    after sectioning, so they no longer cover the whole solid -- radial
    fins, which are half-planes.  It switches off the A-versus-B
    agreement check, which would otherwise fire on every joint: a half
    fin really does have less material than the full ring it meets, and
    that is design, not numerical drift.  The joint span is the overlap
    either way, so nothing else changes.
    """

    def __init__(self, name, planes=None, trimmed=False):
        self.name = name
        self.planes = list(planes or ())
        self.trimmed = bool(trimmed)

    def all_parts(self):
        return [p for pl in self.planes for p in pl.parts]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(a):
    L = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / L, a[1] / L, a[2] / L)


def crossing_line(pa, pb, eps=1e-9):
    """The line where two planes meet, as (point, unit direction).

    None when the planes are parallel -- including the case that makes
    radial fans not self-interlocking, where every plane of the family
    shares the SAME line and there is no pairwise joint to cut.
    """
    na, nb = pa.normal, pb.normal
    w = _cross(na, nb)
    L = math.sqrt(_dot(w, w))
    if L < eps:
        return None
    w = (w[0] / L, w[1] / L, w[2] / L)
    # the point on the line closest to the origin
    ca, cb = pa.offset, pb.offset
    n1 = _cross(w, na)
    n2 = _cross(nb, w)
    det = _dot(na, _cross(nb, w))
    if abs(det) < eps:
        return None
    p = tuple((cb * n1[k] + ca * n2[k]) / det for k in range(3))
    return p, w


def to_frame(plane, point, direction):
    """A 3-D line lying in `plane`, as (origin2, dir2) in its frame.

    The line parameter is preserved: because `direction` lies in the
    plane and (u, v) are orthonormal, dir2 is a unit vector, so the
    same t means the same 3-D point in either family's frame.  That is
    what lets one family's span be handed to the other and compared.
    """
    d, n = plane.offset, plane.normal
    rel = tuple(point[k] - d * n[k] for k in range(3))
    origin2 = (_dot(rel, plane.u), _dot(rel, plane.v))
    dir2 = (_dot(direction, plane.u), _dot(direction, plane.v))
    return origin2, dir2


def plane_intervals(plane, point, direction):
    """Material spans along the line, pooled over the plane's parts.

    Returns [(t0, t1, part)], sorted.  Raises DegenerateClip if any
    part's rings cannot answer cleanly.
    """
    origin2, dir2 = to_frame(plane, point, direction)
    out = []
    for part in plane.parts:
        for t0, t1 in _parts.region_intervals(part, origin2, dir2):
            out.append((t0, t1, part))
    out.sort(key=lambda r: r[0])
    return out


def overlap_intervals(iva, ivb, eps=1e-12):
    """Spans where both families have material.

    Returns (t0, t1, pa, pb, lo_on_a, hi_on_a, lo_on_b, hi_on_b): the
    span, the two parts, and -- crucially -- which end of it lies on
    which part's own RIM.

    That last part is what makes an asymmetric joint work.  A slot has
    to be cut inward from the piece's own edge; a slot starting in the
    middle of a piece is a closed slit that no clipper can open and no
    hand can assemble.  For two untrimmed sections of one solid both
    ends are on both rims and the classic rule applies unchanged.  For
    a radial fin meeting a ring they are not: the span runs from the
    axis to the fin's outer edge, and only the OUTER end is on the
    ring's rim while only the inner end is on the fin's.  Recording
    which is which lets each piece be cut from the end it actually
    owns.
    """
    out = []
    for a0, a1, pa in iva:
        for b0, b1, pb in ivb:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > eps:
                out.append((lo, hi, pa, pb,
                            a0 >= b0 - eps, a1 <= b1 + eps,
                            b0 >= a0 - eps, b1 <= a1 + eps))
    out.sort(key=lambda r: r[0])
    return out


def slot_polygon(origin2, dir2, t_rim, t_end, half, flare=0.0,
                 flare_angle=45.0, overshoot=0.0):
    """The rectangle (or flared hexagon) to subtract from a slice.

    `t_rim` is the end on the piece's rim, `t_end` the midpoint of the
    span.  The mouth is pushed `overshoot` PAST the rim so the cut is
    genuinely open -- a slot that stops exactly on the boundary is a
    tangency, and tangencies are what make a clipper guess.

    Notch Factor widens the mouth by `flare` of the slot width and
    Notch Angle sets how fast it narrows back down: a flared mouth is
    what lets a big assembly that has racked slightly still go
    together.
    """
    sign = 1.0 if t_rim >= t_end else -1.0
    perp = (-dir2[1], dir2[0])

    def at(t, s):
        return (origin2[0] + dir2[0] * t + perp[0] * s,
                origin2[1] + dir2[1] * t + perp[1] * s)

    t_mouth = t_rim + sign * overshoot
    wide = half * (1.0 + max(0.0, flare))
    if flare > 0.0 and 1e-6 < flare_angle < 89.999:
        depth = (wide - half) / math.tan(math.radians(flare_angle))
    else:
        depth = 0.0
        wide = half
    t_taper = t_rim - sign * depth
    # never let the taper run past the far end of the slot
    if (t_taper - t_end) * sign < 0.0:
        t_taper = t_end

    return [at(t_mouth, wide), at(t_taper, half), at(t_end, half),
            at(t_end, -half), at(t_taper, -half), at(t_mouth, -wide)]


class Crossing:
    """One planned joint, kept for reporting."""

    __slots__ = ('ia', 'ib', 'spans', 'errors')

    def __init__(self, ia, ib):
        self.ia = ia
        self.ib = ib
        self.spans = []
        self.errors = []


def plan_interlock(fam_a, fam_b, thickness, clearance=0.0, flare=0.0,
                   flare_angle=45.0, tol_rel=1e-4, flexible=False,
                   scale=1.0):
    """Work out every joint between two families and hang the slot
    polygons on the parts that must carry them.

    Returns (crossings, report) where `report` counts each failure kind.
    Nothing is cut here -- `cut_slots` does that -- so a caller can
    inspect or veto the plan first.
    """
    width = max(1e-9, thickness - clearance)
    half = 0.5 * width
    overshoot = max(width, 1e-9) * 0.75
    tol = tol_rel * max(scale, 1e-9)

    crossings = []
    report = {'joints': 0, 'spans': 0, 'parallel': 0,
              'disagreement': 0, 'unassemblable': 0, 'short': 0,
              'degenerate': 0, 'no_rim': 0}

    for pa in fam_a.planes:
        for pb in fam_b.planes:
            line = crossing_line(pa, pb)
            if line is None:
                report['parallel'] += 1
                continue
            point, w = line
            cr = Crossing(pa.index, pb.index)
            crossings.append(cr)

            # No nudging here, deliberately.  A crossing line running
            # exactly through a section vertex is systematic on a
            # grid-built mesh -- a relief panel meets it at every
            # crossing -- and the obvious fix, displacing the line, is
            # the wrong one: each family drops its own plane's normal
            # component of the displacement, so the two end up querying
            # slightly DIFFERENT lines and start disagreeing about the
            # material by the local slope times the nudge.  The vertex
            # case is settled by convention inside `line_intervals`
            # instead, which costs nothing and keeps both families
            # exact on the same line.
            try:
                iva = plane_intervals(pa, point, w)
                ivb = plane_intervals(pb, point, w)
            except pc.DegenerateClip as exc:
                cr.errors.append(('degenerate', str(exc)))
                report['degenerate'] += 1
                continue

            if not iva and not ivb:
                continue
            report['joints'] += 1

            # The cross-check: for two untrimmed families the line, the
            # solid and therefore the material are the same, so any
            # difference beyond tolerance is numerical drift and the
            # joint is not trustworthy.  Trimmed families are exempt --
            # see Family.trimmed.
            if not (fam_a.trimmed or fam_b.trimmed):
                if len(iva) != len(ivb) or any(
                        abs(a[0] - b[0]) > tol or abs(a[1] - b[1]) > tol
                        for a, b in zip(iva, ivb)):
                    cr.errors.append(
                        ('disagreement',
                         f"A sees "
                         f"{[(round(a,5), round(b,5)) for a, b, _ in iva]}, "
                         f"B sees "
                         f"{[(round(a,5), round(b,5)) for a, b, _ in ivb]}"))
                    report['disagreement'] += 1
                    continue

            # the joint lives where BOTH pieces have material
            shared = overlap_intervals(iva, ivb)
            if not shared:
                continue

            if len(shared) > 1 and not flexible:
                cr.errors.append(
                    ('unassemblable',
                     f"{len(shared)} separate spans on one crossing line: "
                     f"the pieces cannot slide together through solid "
                     f"material"))
                report['unassemblable'] += 1
                continue

            oa, da = to_frame(pa, point, w)
            ob, db = to_frame(pb, point, w)
            # how far each piece's own material reaches along the line;
            # only these extremes can be slid onto
            a_near, a_far = iva[0][0], iva[-1][1]
            b_near, b_far = ivb[0][0], ivb[-1][1]

            for (t0, t1, part_a, part_b,
                 lo_a, hi_a, lo_b, hi_b) in shared:
                if (t1 - t0) < thickness:
                    cr.errors.append(
                        ('short', f"span {t1 - t0:.4g} is thinner than the "
                                  f"material ({thickness:.4g})"))
                    report['short'] += 1
                    continue

                # Each piece must be cut inward from ITS OWN rim, the
                # two must take opposite ends, and -- the part that is
                # easy to miss -- that rim has to be one the piece can
                # actually REACH ALONG THIS LINE.  The pieces assemble
                # by sliding together, so a slot opening onto an
                # interior edge is a slot the other piece can never
                # travel to: it would have to pass through the
                # material lying beyond it.  Being on a rim is
                # necessary, being on the FIRST or LAST rim the line
                # meets is what makes it buildable.
                #
                # For two ordinary sections of a solid the outermost
                # ends are the only ends, so the familiar rule -- A
                # from the +w end, B from the -w end -- is unchanged.
                # Flexible stock is exempt, as it is from the
                # multi-span rule above and for the same reason: card
                # bends around whatever is in the way during assembly,
                # which is how Hart's ten-slit pentagrams go together.
                # Plywood does not.
                def reaches(on_rim, t, extreme):
                    if not on_rim:
                        return False
                    return flexible or abs(t - extreme) <= tol

                reach_a_hi = reaches(hi_a, t1, a_far)
                reach_a_lo = reaches(lo_a, t0, a_near)
                reach_b_hi = reaches(hi_b, t1, b_far)
                reach_b_lo = reaches(lo_b, t0, b_near)
                if reach_a_hi and reach_b_lo:
                    a_rim, b_rim = t1, t0
                elif reach_a_lo and reach_b_hi:
                    a_rim, b_rim = t0, t1
                else:
                    cr.errors.append(
                        ('no_rim',
                         "the slot would open onto an edge the other "
                         "piece cannot slide to: there is material "
                         "beyond it on the same line"))
                    report['no_rim'] += 1
                    continue

                mid = 0.5 * (t0 + t1)
                part_a.slots.append(
                    slot_polygon(oa, da, a_rim, mid, half, flare,
                                 flare_angle, overshoot))
                part_b.slots.append(
                    slot_polygon(ob, db, b_rim, mid, half, flare,
                                 flare_angle, overshoot))
                cr.spans.append((t0, t1))
                report['spans'] += 1

    return crossings, report


def cut_slots(all_parts, dogbone_radius=0.0):
    """Subtract every planned slot from its part.

    Returns a report.  A part whose slot cannot be cut keeps its
    un-slotted outline and carries the error, so the layout still shows
    the piece and the operator can say which joints failed -- an
    exception here would throw away a whole sheet of good parts for one
    bad notch.
    """
    report = {'cut': 0, 'failed': 0, 'split': 0, 'consumed': 0,
              'relieved': 0}
    for part in all_parts:
        ring = part.outer
        for slot in part.slots:
            try:
                rings, _ = pc.difference_robust(ring, slot)
            except pc.DegenerateClip as exc:
                part.fail('slot_clip', str(exc))
                report['failed'] += 1
                continue
            if not rings:
                part.fail('slot_consumed', "the slot removed the whole part")
                report['consumed'] += 1
                continue
            if len(rings) > 1:
                part.fail('slot_split',
                          f"the slot cut the part into {len(rings)} pieces")
                report['split'] += 1
                rings = [max(rings, key=pc.area)]
            ring = rings[0]
            report['cut'] += 1
        if dogbone_radius > 0.0:
            ring, hits = pc.dogbone(ring, dogbone_radius)
            report['relieved'] += hits
        part.outer = pc.as_ccw(ring)
    return report


def check_slot_spacing(family, thickness, clearance=0.0):
    """Slots from planes closer together than the slot is wide would
    overlap on the piece they share.  Cheap to check exactly, and it is
    the most common way a chosen slice count is simply unbuildable."""
    width = max(1e-9, thickness - clearance)
    offs = sorted(p.offset for p in family.planes)
    bad = []
    for a, b in zip(offs, offs[1:]):
        if abs(b - a) < width:
            bad.append((a, b))
    return bad


def label_parts(families):
    """Fusion's Axis-Slice-Part naming, and the assembly order with it.

    The ordinal is not decoration: the families are numbered in the
    order they can actually be inserted -- one family fully, then the
    next -- so the label reads as an instruction as well as a name.
    """
    order = 0
    for fam in families:
        for plane in fam.planes:
            multi = len(plane.parts) > 1
            for part in plane.parts:
                part.label = (f"{fam.name}-{plane.index + 1:02d}"
                              + (f"-{part.index + 1}" if multi else ""))
                part.order = order
                order += 1
    return order


# ------------------------------------------------------------------ #

def _square_plane(normal, offset, u, v, index, half=1.0):
    """A plane carrying one square part, for the checks below."""
    ring = [(-half, -half), (half, -half), (half, half), (-half, half)]
    part = _parts.Part(ring, [], 'T', index, 0, offset)
    return SlicePlane(normal, offset, u, v, index, [part])


def _selftest():
    # --- the joint partitions the span exactly --------------------
    # two crossing planes through a 2x2x2 box: X at x=0.3, Y at y=-0.2
    pa = _square_plane((1, 0, 0), 0.3, (0, 1, 0), (0, 0, 1), 0)
    pb = _square_plane((0, 1, 0), -0.2, (0, 0, 1), (1, 0, 0), 0)
    fa, fb = Family('X', [pa]), Family('Y', [pb])

    line = crossing_line(pa, pb)
    assert line is not None, "perpendicular planes do meet"
    point, w = line
    assert abs(abs(w[2]) - 1.0) < 1e-12, "X meets Y along the Z direction"
    assert abs(point[0] - 0.3) < 1e-12 and abs(point[1] + 0.2) < 1e-12, \
        f"the crossing line passes through both planes, got {point}"

    # both families must measure the same span on that line
    iva = plane_intervals(pa, point, w)
    ivb = plane_intervals(pb, point, w)
    assert len(iva) == 1 and len(ivb) == 1, "one span each"
    assert abs(iva[0][0] - ivb[0][0]) < 1e-12, "span starts agree"
    assert abs(iva[0][1] - ivb[0][1]) < 1e-12, "span ends agree"

    crossings, rep = plan_interlock(fa, fb, thickness=0.1)
    assert rep['spans'] == 1, f"one joint planned, got {rep}"
    assert not crossings[0].errors, crossings[0].errors

    t0, t1 = crossings[0].spans[0]
    mid = 0.5 * (t0 + t1)
    # A carries the +w half, B the -w half: complementary, no overlap,
    # no gap, and together exactly the span
    sa = pa.parts[0].slots[0]
    sb = pb.parts[0].slots[0]

    def span_along(poly, origin2, dir2):
        ts = [(p[0] - origin2[0]) * dir2[0] + (p[1] - origin2[1]) * dir2[1]
              for p in poly]
        return min(ts), max(ts)

    oa, da = to_frame(pa, point, w)
    ob, db = to_frame(pb, point, w)
    a_lo, a_hi = span_along(sa, oa, da)
    b_lo, b_hi = span_along(sb, ob, db)
    assert abs(a_lo - mid) < 1e-9, f"A's slot starts at the midpoint: {a_lo}"
    assert abs(b_hi - mid) < 1e-9, f"B's slot ends at the midpoint: {b_hi}"
    assert a_hi > t1 and b_lo < t0, "each mouth opens past its own rim"

    # --- the slot really is cut, and by the right area ------------
    before = pa.parts[0].area()
    cut = cut_slots([pa.parts[0], pb.parts[0]])
    assert cut['failed'] == 0 and cut['split'] == 0, cut
    removed = before - pa.parts[0].area()
    expect = 0.1 * (t1 - mid)          # width x depth, mouth overshoot
    assert removed > 0.5 * expect, f"slot removed {removed}, expected ~{expect}"

    # --- a span thinner than the stock is refused -----------------
    # both planes at offset 0 so the crossing line really does pass
    # through both little squares -- offset planes would simply miss
    pa2 = _square_plane((1, 0, 0), 0.0, (0, 1, 0), (0, 0, 1), 0, half=0.02)
    pb2 = _square_plane((0, 1, 0), 0.0, (0, 0, 1), (1, 0, 0), 0, half=0.02)
    _, rep2 = plan_interlock(Family('X', [pa2]), Family('Y', [pb2]),
                             thickness=0.5)
    assert rep2['short'] == 1 and rep2['spans'] == 0, \
        f"a span thinner than the material is not a joint: {rep2}"

    # --- co-axial planes share ONE line and cannot pairwise lap ---
    r1 = _square_plane((1, 0, 0), 0.0, (0, 1, 0), (0, 0, 1), 0)
    r2 = _square_plane((1, 0, 0), 0.4, (0, 1, 0), (0, 0, 1), 1)
    assert crossing_line(r1, r2) is None, \
        "parallel planes have no crossing line"

    # --- slots closer together than they are wide would collide ---
    fam = Family('X', [_square_plane((1, 0, 0), 0.0, (0, 1, 0), (0, 0, 1), 0),
                       _square_plane((1, 0, 0), 0.05, (0, 1, 0), (0, 0, 1), 1)])
    assert check_slot_spacing(fam, thickness=0.2), \
        "0.05 apart with a 0.2 slot must be flagged"
    assert not check_slot_spacing(fam, thickness=0.02), \
        "0.05 apart with a 0.02 slot is fine"

    # --- a multi-span crossing is refused unless the stock flexes --
    # A C opening sideways: the crossing line runs up its two arms and
    # through the gap between them, so it meets the material TWICE.
    # The second plane carries the transpose of the same outline, so
    # both families genuinely agree about where the material is -- the
    # point here is the two spans, not a disagreement.
    bar = [(-1.0, -1.0), (1.0, -1.0), (1.0, -0.5), (-0.5, -0.5),
           (-0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (-1.0, 1.0)]
    bar_t = [(b, a) for a, b in bar]
    pu = SlicePlane((1, 0, 0), 0.0, (0, 1, 0), (0, 0, 1), 0,
                    [_parts.Part(bar, [], 'U', 0, 0, 0.0)])
    pw = SlicePlane((0, 1, 0), 0.0, (0, 0, 1), (1, 0, 0), 0,
                    [_parts.Part(bar_t, [], 'W', 0, 0, 0.0)])
    _, rep3 = plan_interlock(Family('U', [pu]), Family('W', [pw]),
                             thickness=0.05, flexible=False)
    assert rep3['unassemblable'] == 1, \
        f"two spans on one line must be refused for rigid stock: {rep3}"
    for p in list(pu.parts) + list(pw.parts):
        p.slots = []
    _, rep4 = plan_interlock(Family('U', [pu]), Family('W', [pw]),
                             thickness=0.05, flexible=True)
    assert rep4['spans'] == 2, \
        f"flexible stock accepts both spans: {rep4}"

    # --- a slot must open onto a rim the other piece can REACH ----
    # A bar with a slab beyond it: the shared span's far end is a rim,
    # but there is more of the same piece past it on the same line, so
    # nothing could ever slide into that slot.
    outer_ring = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    shell = _parts.Part(outer_ring,
                        [[(-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5),
                          (1.5, -1.5)]], 'H', 0, 0, 0.0)
    ph = SlicePlane((1, 0, 0), 0.0, (0, 1, 0), (0, 0, 1), 0, [shell])
    pk = SlicePlane((0, 1, 0), 0.0, (0, 0, 1), (1, 0, 0), 0,
                    [_parts.Part([(b, a) for a, b in outer_ring],
                                 [[(b, a) for a, b in
                                   [(-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5),
                                    (1.5, -1.5)]]], 'K', 0, 0, 0.0)])
    _, rep_h = plan_interlock(Family('H', [ph]), Family('K', [pk]),
                              thickness=0.05, flexible=False)
    assert rep_h['spans'] == 0,         f"a hollow shell cannot be slid together rigidly: {rep_h}"
    assert rep_h['unassemblable'] or rep_h['no_rim'], rep_h

    # --- labels carry family, slice and assembly order ------------
    n = label_parts([fa, fb])
    assert n == 2, f"two parts numbered, got {n}"
    assert pa.parts[0].label == 'X-01', pa.parts[0].label
    assert pb.parts[0].label == 'Y-01', pb.parts[0].label

    return True
