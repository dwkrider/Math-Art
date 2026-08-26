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
        self.stack_spacing = 0.0        # 0 = one material thickness
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


def _half_plane_clip(part, keep_dir, big):
    """Trim a part to the half of its plane on the `keep_dir` side.

    Radial fins are half-planes: a fin that ran clean through the axis
    would have to pass through every other fin there, and no amount of
    slotting fixes a joint that N pieces all want to occupy.
    """
    dx, dy = keep_dir
    px, py = -dy, dx
    # a rectangle covering everything on the wrong side of the axis
    clip = [(-px * big - dx * big, -py * big - dy * big),
            (px * big - dx * big, py * big - dy * big),
            (px * big, py * big),
            (-px * big, -py * big)]
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


def place_dowels(family, count, diameter, min_gap=0.0,
                 samples=28):
    """Alignment dowel holes running through a whole stack.

    A stacked model with nothing to register it is the one configuration
    Fusion refuses to offer, and rightly: sixty free discs do not glue
    up straight.  A dowel has to pass through EVERY slice, so candidate
    positions are drawn from the most constrained slice -- the one with
    the least material -- and then verified against all the others.
    Starting from the smallest slice is what keeps this affordable;
    grid-searching every slice would be minutes of Python.
    """
    if count <= 0 or diameter <= 0.0:
        return [], 0
    stacks = [pl.parts for pl in family.planes if pl.parts]
    if not stacks:
        return [], 0
    r = 0.5 * diameter
    need = r * 1.6
    # dowels too close together do not register a stack any better than
    # one dowel does, so keep them apart by at least the asked-for gap
    apart = max(4.0 * r, float(min_gap))

    smallest = min((p for ps in stacks for p in ps), key=lambda p: p.area())
    x0, y0, x1, y1 = smallest.bounds()

    def clearance(pt, part):
        if not pc.point_in_polygon(pt, part.outer):
            return -1.0
        d = pc.polygon_distance(pt, part.outer)
        for h in part.holes:
            if pc.point_in_polygon(pt, h):
                return -1.0
            d = min(d, pc.polygon_distance(pt, h))
        return d

    cands = []
    for i in range(samples):
        for j in range(samples):
            pt = (x0 + (x1 - x0) * (i + 0.5) / samples,
                  y0 + (y1 - y0) * (j + 0.5) / samples)
            c = clearance(pt, smallest)
            if c >= need:
                cands.append((c, pt))
    if not cands:
        return [], 0

    cands.sort(reverse=True)
    chosen = []
    for _c, pt in cands:
        if len(chosen) >= count:
            break
        if any(math.dist(pt, q) < apart for q in chosen):
            continue
        worst = min(min(clearance(pt, p) for p in ps) for ps in stacks)
        if worst >= need:
            chosen.append(pt)

    drilled = 0
    for ps in stacks:
        for part in ps:
            for pt in chosen:
                if clearance(pt, part) >= need:
                    part.holes.append(
                        pc.as_cw(pc.arc_points(pt[0], pt[1], r,
                                               0.0, 2.0 * math.pi, 16)[:-1]))
                    drilled += 1
    return chosen, drilled


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
                                      st.stack_spacing)
        fam, faults = _family_from_parallel(st.axis.upper(), V, faces,
                                            n, offs)
        families.append(fam)
        report['faults'] += faults
        if st.use_dowels:
            pts, _drilled = place_dowels(fam, st.dowels, st.dowel_diameter,
                                         st.dowel_spacing)
            report['dowels'] = len(pts)

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
        spec = []
        for k in range(max(1, st.radial_count)):
            th = math.pi * k / max(1, st.radial_count)
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
                kept += _half_plane_clip(part, (1.0, 0.0), big)
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
                "the Ribs technique needs a curve to run along; select a "
                "curve object as well as the mesh")
        C = np.asarray(curve, dtype=float)
        C = (C - (np.asarray(verts).min(axis=0)
                  + np.asarray(verts).max(axis=0)) * 0.5) * applied
        idx = np.linspace(0, len(C) - 1, max(2, st.rib_count))
        spec = []
        for k, f in enumerate(idx):
            i = int(round(f))
            j = min(len(C) - 1, i + 1) if i + 1 < len(C) else i - 1
            t = C[j] - C[i]
            t = t / (np.linalg.norm(t) or 1.0)
            if j < i:
                t = -t
            u, v, nn = sections.plane_frame(t)
            spec.append((tuple(nn), float(np.dot(C[i], nn)),
                         tuple(u), tuple(v)))
        ribs, faults = _family_from_planes('C', V, faces, spec)
        families.append(ribs)
        report['faults'] += faults

        n = _axis_vec(st.axis)
        a = float(np.dot(lo, n))
        b = float(np.dot(hi, n))
        spine, faults = _family_from_parallel('S', V, faces, n,
                                              sections.spread_offsets(
                                                  min(a, b), max(a, b), 1))
        families.append(spine)
        report['faults'] += faults

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
    report['cut'] = _slots.cut_slots(all_parts, 0.5 * st.tool_diameter)

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
    if report.get('dowels'):
        out.append(f"{report['dowels']} dowels")
    if report['nest'].get('oversize'):
        out.append(f"{report['nest']['oversize']} parts too big for the sheet")
    if report.get('faults'):
        out.append(f"{len(report['faults'])} unusable layers")
    if report.get('spacing_conflicts'):
        out.append("slice spacing is tighter than the slot is wide")
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

    # spacing wider than the stock thins the stack out
    st_sp = Settings(technique='STACKED', target_size=100.0, thickness=5.0,
                     stack_spacing=10.0, axis='Z', use_dowels=False,
                     sheet_width=400.0, sheet_height=300.0)
    _, fsp, rsp = build(V, F, st_sp, 'sphere')
    assert len(fsp[0].planes) == 10,         f"100 mm at 10 mm spacing is 10 slices, got {len(fsp[0].planes)}"
    assert rsp['dowels'] == 0, "dowels switched off means no dowels"
    assert not any(p.holes for p in fsp[0].all_parts()),         "and no dowel holes drilled either"

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
    for plane in fams3[0].planes:
        for part in plane.parts:
            xs = [p[0] for p in part.outer]
            assert min(xs) > -slack, (
                "a fin is a HALF plane, so it must not cross its axis: "
                f"reaches x={min(xs)}")
    assert rep3['interlock']['spans'] > 0, \
        f"fins must actually join the rings: {rep3['interlock']}"

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
