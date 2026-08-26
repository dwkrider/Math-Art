# The orchestrator: mesh in, cuttable sheets out.
#
# SCALE IS DECIDED FIRST, AND ONCE.  Every generator in this add-on
# emits its object centred in a 2 m cube.  Taken literally against 3 mm
# stock that is some six hundred slices and a layout that fits no sheet
# ever made, which is why "target size" is not a nicety but the first
# thing this module does: the mesh is scaled so its longest dimension
# is the requested real size, and from that point on every number in
# the pipeline -- offsets, slot widths, sheet coordinates -- is
# millimetres.  Fusion calls this Object Size and puts it before the
# construction technique for the same reason.
#
# STACKED SPACING IS NOT FREE.  A glued stack only reproduces the shape
# when the planes are one material thickness apart, so for that
# technique the slice count is DERIVED from height / thickness rather
# than offered as a knob that can be set into contradiction with the
# stock.  For interlocked work the two really are independent -- spacing
# is a design choice, thickness is whatever the sheet is -- and both
# stay adjustable.
#
# THE FOUR TECHNIQUES.  Stacked and Interlocked are the two that Sharp,
# Luecking and Fusion all describe.  Radial needs a word: N planes
# through a common axis all meet along that ONE line, so there is no
# pairwise joint to cut and a radial fan cannot interlock with itself.
# The honest construction is the globe -- meridians and parallels -- so
# Radial here means half-fins joined to a stack of rings, and the fins
# are half-planes precisely so they do not have to pass through one
# another at the axis.  Ribs are sections perpendicular to a curve,
# joined to a spine.

import math

import numpy as np

from . import polyclip as pc
from . import sections
from . import parts as _parts
from . import slots as _slots
from . import layout as _layout

AXES = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}


class Settings:
    """Everything the build needs, with buildable defaults."""

    def __init__(self, **kw):
        self.technique = 'STACKED'
        self.target_size = 200.0        # mm, longest dimension
        self.thickness = 3.0            # mm of stock
        self.kerf = 0.15                # mm burnt away by the beam
        self.clearance = 0.0            # + loosens the joint, - tightens
        self.flexible = False           # paper/card rather than ply
        self.sheet_width = 600.0
        self.sheet_height = 400.0
        self.margin = 5.0
        self.axis = 'Z'                 # one axis: unidirectional
        self.axis_a = 'X'               # two axes: bidirectional
        self.axis_b = 'Y'
        self.count_a = 8
        self.count_b = 8
        self.radial_count = 8
        self.ring_count = 4
        self.rib_count = 12
        self.curve = None               # list of 3-D points, for RIBS
        self.slice_gap = 0.0            # space left between slices
        self.use_dowels = True
        self.dowels = 2
        self.dowel_diameter = 4.0
        self.dowel_spacing = 30.0       # least gap between dowels
        self.flare = 0.0                # Fusion's Notch Factor
        self.flare_angle = 45.0         # Fusion's Notch Angle
        self.tool_diameter = 0.0        # > 0 turns on dog-bone relief
        self.label_height = 4.0
        for k, val in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown setting {k!r}")
            setattr(self, k, val)

    def slot_width(self):
        return max(1e-6, self.thickness - self.clearance)


def scale_to_target(verts, target):
    """Scale a vertex array so its longest dimension is `target`, and
    centre it.  Returns (verts_mm, applied_scale)."""
    V = np.asarray(verts, dtype=float)
    lo, hi = V.min(axis=0), V.max(axis=0)
    span = hi - lo
    longest = float(span.max())
    if longest <= 0.0:
        return V.copy(), 1.0
    s = float(target) / longest
    return (V - (lo + hi) * 0.5) * s, s


def _axis_vec(name):
    try:
        return AXES[name.upper()]
    except KeyError:
        raise KeyError(f"axis must be one of X, Y, Z (got {name!r})")


def _family_from_parallel(name, verts, faces, normal, offsets):
    sl, committed, faults, (u, v, n) = sections.section_family(
        verts, faces, normal, offsets)
    planes = []
    for k, loops in enumerate(sl):
        ps = _parts.build_parts(loops, name, k, committed[k])
        planes.append(_slots.SlicePlane(n, committed[k], u, v, k, ps))
    return _slots.Family(name, planes), faults


def _family_from_planes(name, verts, faces, spec):
    sl, committed, faults = sections.section_planes(verts, faces, spec)
    planes = []
    for k, loops in enumerate(sl):
        normal, _off, u, v = spec[k]
        ps = _parts.build_parts(loops, name, k, committed[k])
        planes.append(_slots.SlicePlane(normal, committed[k], u, v, k, ps))
    return _slots.Family(name, planes), faults


def hub_radius(count, thickness, safety=1.15):
    """How far back from the axis radial fins have to start.

    N fins meeting at one axis all want to occupy it, and being
    half-planes is not enough to save them: near the axis the wedge
    between neighbours is narrower than the material.  Two fins of
    thickness t separated by angle 2*pi/N stay clear of one another
    only beyond radius t / (2 sin(pi/N)), so that is where they begin
    -- and the rings, which each fin is slotted to several times, are
    what hold them.
    """
    n = max(2, int(count))
    return safety * thickness / (2.0 * math.sin(math.pi / n))


def _half_plane_clip(part, inner, big):
    """Trim a fin to the wedge beyond `inner` on its own side.

    Cuts away the far half of the plane AND the disc around the axis,
    which is what keeps N fins from colliding where they all meet.
    """
    # a rectangle covering everything at radius below `inner`,
    # including the whole of the far half-plane
    clip = [(-big, -big), (inner, -big), (inner, big), (-big, big)]
    try:
        rings, _ = pc.difference_robust(part.outer, clip)
    except pc.DegenerateClip:
        return [part]
    out = []
    for k, ring in enumerate(rings):
        holes = [h for h in part.holes
                 if pc.point_in_polygon(h[0], ring)]
        q = _parts.Part(ring, holes, part.family, part.slice_index, k,
                        part.offset)
        out.append(q)
    return out


def resample_by_arclength(C, count, closed=None, tol=1e-6):
    """`count` points and unit tangents spaced evenly ALONG the curve.

    Sampling the control points by index instead -- which is what this
    did at first -- spaces the ribs by however densely the curve
    happens to be described, not by distance.  On a knot that bunches
    them through the tight bends and strands them along the straights.

    A closed curve is detected from its endpoints and sampled without
    repeating the seam; an open one is inset slightly from both ends,
    where a perpendicular plane tends to graze the cap rather than cut
    a rib out of it.
    """
    C = np.asarray(C, dtype=float)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0.0:
        raise ValueError("the curve has no length to space ribs along")
    if closed is None:
        extent = float(np.linalg.norm(C.max(axis=0) - C.min(axis=0))) or 1.0
        closed = float(np.linalg.norm(C[0] - C[-1])) < 0.02 * extent

    n = max(2, int(count))
    if closed:
        targets = np.linspace(0.0, total, n, endpoint=False)
    else:
        targets = np.linspace(0.02 * total, 0.98 * total, n)

    pts, tans = [], []
    for t in targets:
        k = int(np.searchsorted(cum, t, side='right') - 1)
        k = max(0, min(len(seg) - 1, k))
        span = seg[k] if seg[k] > tol else 1.0
        f = (t - cum[k]) / span
        pts.append(C[k] + f * (C[k + 1] - C[k]))
        d = C[k + 1] - C[k]
        tans.append(d / (np.linalg.norm(d) or 1.0))
    return np.asarray(pts), np.asarray(tans), closed, total


def place_dowels(family, count, diameter, min_gap=0.0, samples=16):
    """Alignment dowels down a stack.

    Two things this gets right that the obvious version does not.

    A dowel does NOT have to run the whole height of the model.  What
    it has to do is register a slice against its NEIGHBOUR, so dowels
    are placed as runs: walking the stack, a position is kept while it
    still lands in material, and when it runs out that run ends and a
    fresh one starts for the slices that follow.  Insisting on one
    position through every slice meant a shape whose slices wander --
    a knot, where one layer's material is nowhere near the next -- got
    no dowels at all.

    And the unit that needs dowelling is the PIECE, not the slice.  A
    slice through a knot is three separate blobs that will be three
    separate parts on the sheet; two positions chosen for the slice
    can both land in one blob and leave the other two bare, which is
    exactly what happened.  So every PART is topped up to `count`
    independently.

    A part too small to hold even one dowel simply goes without, and
    is counted rather than passed over in silence.

    Returns (runs, drilled, parts_with, parts_without).
    """
    planes = [pl for pl in family.planes if pl.parts]
    parts_total = sum(len(pl.parts) for pl in planes)
    if count <= 0 or diameter <= 0.0 or not planes:
        return 0, 0, 0, parts_total

    r = 0.5 * diameter
    need = r * 1.6
    # Just clear of touching when no spacing is asked for.  A wider
    # floor than that refuses a second dowel on every narrow piece,
    # which is most of them on a tube-shaped model.
    apart = max(2.2 * r, float(min_gap))

    def fits(pt, part):
        if not pc.point_in_polygon(pt, part.outer):
            return False
        d = pc.polygon_distance(pt, part.outer)
        for h in part.holes:
            if pc.point_in_polygon(pt, h):
                return False
            d = min(d, pc.polygon_distance(pt, h))
        return d >= need

    def hosted(pt, plane):
        return plane is not None and any(fits(pt, q) for q in plane.parts)

    def candidates(part, limit=80):
        """Room in this part, roomiest first."""
        x0, y0, x1, y1 = part.bounds()
        if min(x1 - x0, y1 - y0) < 2.0 * need:
            return []
        out = []
        for i in range(samples):
            for j in range(samples):
                pt = (x0 + (x1 - x0) * (i + 0.5) / samples,
                      y0 + (y1 - y0) * (j + 0.5) / samples)
                if not pc.point_in_polygon(pt, part.outer):
                    continue
                d = pc.polygon_distance(pt, part.outer)
                bad = False
                for h in part.holes:
                    if pc.point_in_polygon(pt, h):
                        bad = True
                        break
                    d = min(d, pc.polygon_distance(pt, h))
                if bad or d < need:
                    continue
                out.append((d, pt))
        # Roomiest first, but keep a LONG list: the second dowel has
        # to clear the first by `apart`, and a short list is all
        # clustered around the middle of the piece, so the runner-up
        # positions were being thrown away before anything could pick
        # one far enough out.
        out.sort(key=lambda kv: -kv[0])
        return [pt for _d, pt in out[:limit]]

    active, runs, drilled, with_holes = [], 0, 0, 0
    for i, plane in enumerate(planes):
        nxt = planes[i + 1] if i + 1 < len(planes) else None
        prv = planes[i - 1] if i > 0 else None
        carried = []
        for part in plane.parts:
            mine = [pt for pt in active if fits(pt, part)]
            if len(mine) < count:
                for pt in candidates(part):
                    if len(mine) >= count:
                        break
                    if any(math.dist(pt, q) < apart for q in mine):
                        continue
                    # a lone hole registers nothing: it has to reach a
                    # neighbouring slice to be a joint at all
                    if not (hosted(pt, nxt) or hosted(pt, prv)):
                        continue
                    mine.append(pt)
                    runs += 1
            for pt in mine:
                part.holes.append(pc.as_cw(
                    pc.arc_points(pt[0], pt[1], r, 0.0,
                                  2.0 * math.pi, 16)[:-1]))
                drilled += 1
            if mine:
                with_holes += 1
            carried.extend(mine)
        active = carried
    return runs, drilled, with_holes, parts_total - with_holes


def build(verts, faces, settings, name='slices'):
    """Slice a mesh into cuttable sheets.

    Returns (drawing, families, report).  Nothing raises for bad
    geometry: faults are counted and reported, because one unusable
    layer out of sixty should cost that layer, not the job.
    """
    st = settings
    V, applied = scale_to_target(verts, st.target_size)
    lo, hi = V.min(axis=0), V.max(axis=0)

    report = {'scale': applied, 'technique': st.technique,
              'faults': [], 'interlock': {}, 'cut': {}, 'nest': {},
              'dowels': 0, 'spacing_conflicts': []}
    families = []

    if st.technique == 'STACKED':
        n = _axis_vec(st.axis)
        span = (float(np.dot(hi, n)), float(np.dot(lo, n)))
        a, b = min(span), max(span)
        offs = sections.layer_offsets(a, b, st.thickness,
                                      st.slice_gap)
        fam, faults = _family_from_parallel(st.axis.upper(), V, faces,
                                            n, offs)
        families.append(fam)
        report['faults'] += faults
        if st.use_dowels and st.dowels > 0:
            runs, drilled, with_h, without = place_dowels(
                fam, st.dowels, st.dowel_diameter, st.dowel_spacing)
            report['dowels'] = runs
            report['dowel_holes'] = drilled
            report['dowel_parts'] = with_h
            report['dowels_missing'] = without

    elif st.technique == 'INTERLOCKED':
        if st.axis_a.upper() == st.axis_b.upper():
            raise ValueError("the two slice axes must differ")
        fams = []
        for label, count in ((st.axis_a.upper(), st.count_a),
                             (st.axis_b.upper(), st.count_b)):
            n = _axis_vec(label)
            a = float(np.dot(lo, n))
            b = float(np.dot(hi, n))
            offs = sections.spread_offsets(min(a, b), max(a, b), count)
            fam, faults = _family_from_parallel(label, V, faces, n, offs)
            fams.append(fam)
            report['faults'] += faults
        families += fams

    elif st.technique == 'RADIAL':
        ax = _axis_vec(st.axis)
        e1, e2, _ = sections.plane_frame(ax)
        big = float(np.linalg.norm(hi - lo)) * 2.0
        inner = hub_radius(st.radial_count, st.thickness)
        report['hub_radius'] = inner
        spec = []
        for k in range(max(1, st.radial_count)):
            # The FULL turn, not half of it.  A whole diametral plane
            # at angle th also covers th + pi, so [0, pi) would be
            # right for whole planes -- but these fins are clipped to
            # half-planes (they must be, or every fin would have to
            # pass through every other at the axis), and a half-plane
            # covers ONE direction.  Half a turn of them therefore
            # dressed only half the object.
            th = 2.0 * math.pi * k / max(1, st.radial_count)
            radial = tuple(e1[i] * math.cos(th) + e2[i] * math.sin(th)
                           for i in range(3))
            normal = tuple(ax[(i + 1) % 3] * radial[(i + 2) % 3]
                           - ax[(i + 2) % 3] * radial[(i + 1) % 3]
                           for i in range(3))
            # frame (u, v) = (radial, axis): 2-D coords are (r, z), so
            # the half we keep is simply r >= 0
            spec.append((normal, 0.0, radial, ax))
        fins, faults = _family_from_planes('R', V, faces, spec)
        report['faults'] += faults
        for plane in fins.planes:
            kept = []
            for part in plane.parts:
                kept += _half_plane_clip(part, inner, big)
            for i, q in enumerate(kept):
                q.index = i
            plane.parts = kept
        fins.trimmed = True   # half-planes: see Family.trimmed
        families.append(fins)

        a = float(np.dot(lo, ax))
        b = float(np.dot(hi, ax))
        offs = sections.spread_offsets(min(a, b), max(a, b), st.ring_count)
        rings, faults = _family_from_parallel(st.axis.upper(), V, faces,
                                              ax, offs)
        families.append(rings)
        report['faults'] += faults

    elif st.technique == 'RIBS':
        curve = st.curve
        if not curve or len(curve) < 2:
            raise ValueError(
                "the Ribs technique needs a curve to run along: pick one "
                "in the Curve field, or slice a curve object itself")
        C = np.asarray(curve, dtype=float)
        C = (C - (np.asarray(verts).min(axis=0)
                  + np.asarray(verts).max(axis=0)) * 0.5) * applied
        rib_pts, rib_tans, closed, length = resample_by_arclength(
            C, st.rib_count)
        report['curve_length'] = length
        report['curve_closed'] = closed
        spec, anchors = [], []
        for P, t in zip(rib_pts, rib_tans):
            u, v, nn = sections.plane_frame(t)
            spec.append((tuple(nn), float(np.dot(P, nn)),
                         tuple(u), tuple(v)))
            anchors.append((P, u, v, nn))
        ribs, faults = _family_from_planes('C', V, faces, spec)
        report['faults'] += faults

        # Decide FIRST whether there is a flat spine to slot into,
        # because that decides whether the ribs get threading holes.
        # A flat spine only exists if the curve is flat, so fit a plane
        # to it and measure.  A plane curve gets a real spine; a knot
        # does not, and saying so is better than emitting a "spine"
        # that is a section through the whole tangle and slots to
        # nothing.
        mid = C.mean(axis=0)
        _uu, _ss, vt = np.linalg.svd(C - mid, full_matrices=False)
        normal = vt[2] if vt.shape[0] > 2 else None
        extent = float(np.linalg.norm(C.max(axis=0) - C.min(axis=0))) or 1.0
        flat = (normal is not None
                and float(np.abs((C - mid) @ normal).max()) < 0.02 * extent)

        # A rib is the LOCAL cross section, not everything the plane
        # happens to cut.  A plane perpendicular to a knot's centreline
        # slices every other strand it passes through as well, so
        # keeping the whole section gives each rib a fistful of debris
        # from the far side of the knot.  Keep only the piece the curve
        # actually threads: the part containing the spine point, or the
        # nearest one when the nudge leaves it just outside.
        kept_ribs = 0
        for plane, (P, u, v, nn) in zip(ribs.planes, anchors):
            rel = P - plane.offset * np.asarray(plane.normal)
            here = (float(rel @ np.asarray(u)), float(rel @ np.asarray(v)))
            inside = [q for q in plane.parts
                      if pc.point_in_polygon(here, q.outer)
                      and not any(pc.point_in_polygon(here, h)
                                  for h in q.holes)]
            if not inside and plane.parts:
                inside = [min(plane.parts,
                              key=lambda q: pc.polygon_distance(here,
                                                                q.outer))]
            for q in inside:
                q.index = 0
            plane.parts = inside[:1]
            kept_ribs += len(plane.parts)

            # Thread the rib onto a rod: a small hole where the curve
            # passes through it.  Only when there is NO flat spine --
            # the hole sits exactly where a spine would cross, so
            # drilling it as well would split that joint in two and the
            # rib could not be slotted on at all.  Threading and
            # slotting are alternatives, not companions.
            if (not flat and st.use_dowels and st.dowel_diameter > 0.0
                    and plane.parts):
                q = plane.parts[0]
                r = 0.5 * st.dowel_diameter
                clear = pc.polygon_distance(here, q.outer)
                if pc.point_in_polygon(here, q.outer) and clear > r * 1.5:
                    q.holes.append(pc.as_cw(
                        pc.arc_points(here[0], here[1], r,
                                      0.0, 2.0 * math.pi, 16)[:-1]))
                    report['dowels'] += 1
        report['ribs'] = kept_ribs
        families.append(ribs)

        if flat:
            spine, faults = _family_from_parallel(
                'S', V, faces, tuple(normal), [float(mid @ normal)])
            families.append(spine)
            report['faults'] += faults
            report['spine'] = 'plane'
        else:
            report['spine'] = 'wire'

    else:
        raise ValueError(f"unknown technique {st.technique!r}")

    _slots.label_parts(families)

    if len(families) >= 2:
        crossings, irep = _slots.plan_interlock(
            families[0], families[1], st.thickness, st.clearance,
            st.flare, st.flare_angle, flexible=st.flexible,
            scale=float(np.linalg.norm(hi - lo)))
        report['interlock'] = irep
        for fam in families:
            bad = _slots.check_slot_spacing(fam, st.thickness, st.clearance)
            if bad:
                report['spacing_conflicts'].append((fam.name, len(bad)))

    all_parts = [p for fam in families for p in fam.all_parts()]
    # Count the pieces nothing holds BEFORE the slots are consumed.
    # A part with no joint is not a part of the model: it falls out.
    # This is the failure that made a radial slicing of a knot look
    # like a heap of loose blades, and it deserves to be reported
    # rather than left for the assembler to discover.
    if len(families) >= 2:
        loose = [p for p in all_parts if not p.slots]
        report['unconnected'] = len(loose)
        for p in loose:
            p.fail('unconnected', "no joint holds this piece")
    report['cut'] = _slots.cut_slots(families, 0.5 * st.tool_diameter)
    # a slot may have severed a piece into several, so the part list
    # and the labels are both re-derived rather than reused
    all_parts = [p for fam in families for p in fam.all_parts()]
    _slots.label_parts(families)

    drawing, nrep = _layout.nest(
        all_parts, st.sheet_width, st.sheet_height, st.margin,
        st.kerf, st.label_height, name)
    report['nest'] = nrep
    report['parts'] = len(all_parts)
    report['overhang'] = drawing.bounds_ok()
    return drawing, families, report


def summarise(report):
    """One-line-per-issue summary, for the operator's status line."""
    out = [f"{report.get('parts', 0)} parts on "
           f"{report['nest'].get('sheets', 0)} sheet(s)"]
    il = report.get('interlock') or {}
    if il.get('spans'):
        out.append(f"{il['spans']} joints")
    if il.get('clearance'):
        out.append(f"{il['clearance']} crossings cleared rather than "
                   f"jointed (one piece notched through, no capture)")
    for key, text in (('unassemblable', 'unassemblable crossings'),
                      ('short', 'spans thinner than the stock'),
                      ('disagreement', 'crossings the two families '
                                       'disagree about'),
                      ('degenerate', 'degenerate crossings')):
        if il.get(key):
            out.append(f"{il[key]} {text}")
    cut = report.get('cut') or {}
    for key, text in (('failed', 'slots that could not be cut'),
                      ('split', 'slots that split a part'),
                      ('consumed', 'parts consumed by a slot')):
        if cut.get(key):
            out.append(f"{cut[key]} {text}")
    if cut.get('relieved'):
        out.append(f"{cut['relieved']} dog-bone corners")
    if report['nest'].get('oversize'):
        out.append(f"{report['nest']['oversize']} parts too big for the sheet")
    if report.get('faults'):
        out.append(f"{len(report['faults'])} unusable layers")
    if report.get('spacing_conflicts'):
        out.append("slice spacing is tighter than the slot is wide")
    if report.get('unconnected'):
        out.append(f"{report['unconnected']} pieces nothing holds "
                   f"(no joint reaches them)")
    if report.get('dowel_holes'):
        out.append(f"{report['dowel_holes']} dowel holes in "
                   f"{report['dowel_parts']} pieces "
                   f"({report['dowels']} runs)")
    if report.get('dowels_missing'):
        out.append(f"{report['dowels_missing']} pieces too small for a "
                   f"dowel")
    if report.get('spine') == 'wire':
        out.append("the curve is not flat, so there is no spine to slot "
                   "into: thread the ribs on a rod through their holes")
    return "; ".join(out)


# ------------------------------------------------------------------ #

def _selftest():
    V, F = sections._icosphere(3, 1.0)

    # --- scale: the 2 m cube convention must not reach the sheet --
    Vm, s = scale_to_target(V, 200.0)
    span = Vm.max(axis=0) - Vm.min(axis=0)
    assert abs(float(span.max()) - 200.0) < 1e-9, \
        f"longest dimension must become the target: {span}"
    assert abs(float(Vm.mean(axis=0).max())) < 1e-6, "and be centred"
    assert abs(s - 100.0) < 1e-9, f"a radius-1 sphere scales by 100: {s}"

    # --- stacked: count is DERIVED from thickness, not invented ---
    st = Settings(technique='STACKED', target_size=100.0, thickness=5.0,
                  axis='Z', dowels=2, dowel_diameter=4.0,
                  dowel_spacing=10.0,
                  sheet_width=400.0, sheet_height=300.0)
    d, fams, rep = build(V, F, st, 'sphere')
    assert not rep['faults'], rep['faults']
    assert len(fams) == 1, "stacked slices along ONE axis"
    assert len(fams[0].planes) == 20, \
        f"100 mm of sphere in 5 mm stock is 20 layers, got {len(fams[0].planes)}"
    assert rep['parts'] == 20, rep
    assert not rep['overhang'], rep['overhang']
    assert rep['dowels'] == 2, f"two dowels through the stack: {rep}"
    holed = [p for p in fams[0].all_parts() if p.holes]
    assert len(holed) > 10, "the dowels pass through most of the stack"
    # every slice that can take a dowel gets one -- a dowel registers a
    # slice against its neighbour, so it does NOT have to run the whole
    # height of the model
    n_parts = len(fams[0].all_parts())
    assert rep['dowel_parts'] >= n_parts - 2, (
        f"every piece big enough should be dowelled: "
        f"{rep['dowel_parts']} of {n_parts}")
    # count=2 means TWO dowels in each piece, not two in the model
    assert rep['dowel_holes'] >= 2 * rep['dowel_parts'] - 2, (
        f"a dowel count of 2 should put two holes in each piece: "
        f"{rep['dowel_holes']} holes over {rep['dowel_parts']} pieces")

    # a gap between slices thins the stack out; the pitch is the
    # material plus the gap, never less than the material
    st_sp = Settings(technique='STACKED', target_size=100.0, thickness=5.0,
                     slice_gap=5.0, axis='Z', use_dowels=False,
                     sheet_width=400.0, sheet_height=300.0)
    _, fsp, rsp = build(V, F, st_sp, 'sphere')
    assert len(fsp[0].planes) == 10, (
        "5 mm stock with a 5 mm gap is a 10 mm pitch, so 10 slices; "
        f"got {len(fsp[0].planes)}")
    assert rsp['dowels'] == 0, "dowels switched off means no dowels"
    assert not any(p.holes for p in fsp[0].all_parts()), \
        "and no dowel holes drilled either"

    # asking for the dowels further apart than the piece allows gives
    # fewer dowels, not dowels crowded together where they register
    # nothing
    st_far = Settings(technique='STACKED', target_size=100.0, thickness=5.0,
                      axis='Z', dowels=2, dowel_diameter=4.0,
                      dowel_spacing=90.0,
                      sheet_width=400.0, sheet_height=300.0)
    _, _, rep_far = build(V, F, st_far, 'sphere')
    assert rep_far['dowels'] < 2,         f"a 90 mm gap does not fit twice across this piece: {rep_far}"

    # --- stacked along a different axis really is different -------
    st_x = Settings(technique='STACKED', target_size=100.0, thickness=5.0,
                    axis='X', sheet_width=400.0, sheet_height=300.0)
    _, fx, _ = build(V, F, st_x, 'sphere')
    assert abs(fx[0].planes[0].normal[0]) == 1.0, \
        "the chosen axis is the slicing direction"

    # --- interlocked: two axes, complementary slots ---------------
    st2 = Settings(technique='INTERLOCKED', target_size=100.0,
                   thickness=3.0, axis_a='X', axis_b='Y',
                   count_a=4, count_b=4,
                   sheet_width=400.0, sheet_height=300.0)
    d2, fams2, rep2 = build(V, F, st2, 'sphere')
    assert len(fams2) == 2, "two families"
    assert rep2['interlock']['spans'] == 16, \
        f"4 x 4 crossings all through a sphere: {rep2['interlock']}"
    assert rep2['interlock']['unassemblable'] == 0, rep2['interlock']
    assert rep2['interlock']['disagreement'] == 0, \
        f"the two families must agree about the solid: {rep2['interlock']}"
    assert rep2['cut']['failed'] == 0, rep2['cut']
    assert rep2['cut']['split'] == 0, rep2['cut']
    assert rep2['cut']['cut'] == 32, \
        f"every joint cuts both halves: {rep2['cut']}"

    # same axis twice is not a construction
    try:
        build(V, F, Settings(technique='INTERLOCKED', axis_a='X',
                             axis_b='X'), 'x')
        raise AssertionError("two identical axes should be refused")
    except ValueError:
        pass

    # --- radial: fins are halves, and joined to rings -------------
    st3 = Settings(technique='RADIAL', target_size=100.0, thickness=3.0,
                   axis='Z', radial_count=6, ring_count=3,
                   sheet_width=400.0, sheet_height=300.0)
    d3, fams3, rep3 = build(V, F, st3, 'vase')
    assert len(fams3) == 2 and fams3[0].name == 'R', "fins then rings"
    assert len(fams3[0].planes) == 6 and len(fams3[1].planes) == 3, \
        "six fins, three rings"
    # the fin's straight edge sits on the axis, then kerf compensation
    # grows the whole outline outward by half a kerf -- so half a kerf
    # past the axis is exactly right, and anything more is a fin that
    # was never clipped
    slack = 0.5 * st3.kerf + 1e-6
    # the fins must go all the way round: half-plane fins cover one
    # direction each, so half a turn of them dresses half the object
    dirs = sorted(math.atan2(pl.u[1], pl.u[0]) % (2 * math.pi)
                  for pl in fams3[0].planes)
    widest = max((dirs[(i + 1) % len(dirs)] - dirs[i]) % (2 * math.pi)
                 for i in range(len(dirs)))
    assert widest < 1.2 * (2 * math.pi / len(dirs)), (
        "half-plane fins must spread over the whole turn; widest gap is "
        f"{math.degrees(widest):.0f} deg over {len(dirs)} fins")

    # and they must start clear of the axis, or they collide there
    want_hub = hub_radius(st3.radial_count, st3.thickness)
    assert want_hub > st3.thickness, f"hub radius {want_hub}"
    for plane in fams3[0].planes:
        for part in plane.parts:
            xs = [p[0] for p in part.outer]
            assert min(xs) > want_hub - 0.5 * st3.kerf - 1e-6, (
                f"a fin must start beyond the hub radius {want_hub:.2f}: "
                f"reaches {min(xs):.2f}")
            assert min(xs) > -slack, (
                "a fin is a HALF plane, so it must not cross its axis: "
                f"reaches x={min(xs)}")
    assert rep3['interlock']['spans'] > 0, \
        f"fins must actually join the rings: {rep3['interlock']}"

    # --- ribs are spaced by DISTANCE along the curve, not by index -
    circle = [(math.cos(a), math.sin(a), 0.0)
              for a in np.linspace(0.0, 2 * math.pi, 400, endpoint=False)]
    circle.append(circle[0])
    # deliberately lopsided sampling: dense on one side, sparse on the
    # other, which is exactly what a curve's control points look like
    lop = [circle[i] for i in
           sorted(set(list(range(0, 200, 2)) + list(range(200, 400, 30))))]
    lop.append(lop[0])
    pts, tans, closed, total = resample_by_arclength(lop, 12)
    assert closed, "a curve that returns to its start is closed"
    assert len(pts) == 12, f"12 ribs, got {len(pts)}"
    gaps = [float(np.linalg.norm(pts[(i + 1) % 12] - pts[i]))
            for i in range(12)]
    assert max(gaps) - min(gaps) < 0.05 * max(gaps), (
        "ribs must be evenly spaced along the curve however unevenly "
        f"it is described: gaps {min(gaps):.4f}..{max(gaps):.4f}")
    for t in tans:
        assert abs(float(np.linalg.norm(t)) - 1.0) < 1e-9, "unit tangents"
    # and an open curve is inset from its ends rather than sampling them
    line = [(0.0, 0.0, z) for z in np.linspace(-1.0, 1.0, 50)]
    pts2, _t2, closed2, _L = resample_by_arclength(line, 6)
    assert not closed2, "a straight line is not closed"
    assert pts2[0][2] > -1.0 and pts2[-1][2] < 1.0, "inset from the ends"

    # --- ribs along a curve ---------------------------------------
    curve = [(0.0, 0.0, z) for z in np.linspace(-0.9, 0.9, 12)]
    st4 = Settings(technique='RIBS', target_size=100.0, thickness=3.0,
                   rib_count=6, axis='X', curve=curve,
                   sheet_width=400.0, sheet_height=300.0)
    d4, fams4, rep4 = build(V, F, st4, 'ribs')
    assert len(fams4) == 2 and len(fams4[0].planes) == 6, "six ribs"
    assert rep4['interlock']['spans'] > 0, "ribs join the spine"

    try:
        build(V, F, Settings(technique='RIBS', curve=None), 'x')
        raise AssertionError("ribs without a curve should be refused")
    except ValueError:
        pass

    # --- a grid-built heightfield, which is the real workload -----
    # A relief panel is a regular grid, so its sections have vertices
    # sitting exactly ON the crossing lines at EVERY joint -- the case
    # that a sphere never exercises and that broke this twice: once as
    # degenerate crossings, once (after nudging the line to dodge them)
    # as the two families disagreeing by the local slope times the
    # nudge.  Both must stay at zero.
    nx = 40
    xs = np.linspace(-1.0, 1.0, nx)
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    ZZ = 0.18 * np.sin(4.1 * XX) * np.cos(3.7 * YY) + 0.06 * np.cos(9.0 * XX)
    gv, gf = [], []
    for i in range(nx):
        for j in range(nx):
            gv.append((XX[i, j], YY[i, j], ZZ[i, j] + 0.30))
    for i in range(nx):
        for j in range(nx):
            gv.append((XX[i, j], YY[i, j], -0.30))
    top = lambda i, j: i * nx + j
    bot = lambda i, j: nx * nx + i * nx + j
    for i in range(nx - 1):
        for j in range(nx - 1):
            gf += [(top(i, j), top(i + 1, j), top(i + 1, j + 1)),
                   (top(i, j), top(i + 1, j + 1), top(i, j + 1)),
                   (bot(i, j), bot(i + 1, j + 1), bot(i + 1, j)),
                   (bot(i, j), bot(i, j + 1), bot(i + 1, j + 1))]
    for i in range(nx - 1):                       # side walls
        a, b = 0, nx - 1
        gf += [(top(i, a), top(i + 1, a), bot(i + 1, a)),
               (top(i, a), bot(i + 1, a), bot(i, a)),
               (top(i, b), bot(i + 1, b), top(i + 1, b)),
               (top(i, b), bot(i, b), bot(i + 1, b)),
               (top(a, i), bot(a, i + 1), top(a, i + 1)),
               (top(a, i), bot(a, i), bot(a, i + 1)),
               (top(b, i), top(b, i + 1), bot(b, i + 1)),
               (top(b, i), bot(b, i + 1), bot(b, i))]

    st5 = Settings(technique='INTERLOCKED', target_size=240.0,
                   thickness=3.0, axis_a='X', axis_b='Y',
                   count_a=6, count_b=6,
                   sheet_width=600.0, sheet_height=400.0)
    d5, f5, rep5 = build(gv, gf, st5, 'relief')
    il = rep5['interlock']
    assert not rep5['faults'], f"the slab must section cleanly: {rep5['faults']}"
    assert il['degenerate'] == 0, \
        f"a crossing through a grid vertex is settled by convention: {il}"
    assert il['disagreement'] == 0, \
        f"the two families must agree exactly on a grid mesh: {il}"
    assert il['spans'] == 36, f"every one of 6x6 crossings is a joint: {il}"
    assert rep5['cut']['failed'] == 0 and rep5['cut']['split'] == 0, \
        f"the clipper must survive dense noisy outlines: {rep5['cut']}"

    # --- an unknown technique is refused, not guessed -------------
    try:
        build(V, F, Settings(technique='ORIGAMI'), 'x')
        raise AssertionError("unknown technique should raise")
    except ValueError:
        pass
    try:
        Settings(nonsense=1)
        raise AssertionError("unknown setting should raise")
    except KeyError:
        pass

    # --- the summary says something true --------------------------
    text = summarise(rep2)
    assert 'parts' in text and 'joints' in text, text

    return True
