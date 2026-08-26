# Nesting the parts onto sheets, and turning them into a Drawing.
#
# KERF, DONE ONCE.  A laser burns away roughly `kerf` of material
# centred on the path it follows, so a part cut on its true outline
# comes out kerf/2 small all round -- and, because a slot is part of
# that same outline, every slot comes out kerf WIDE, which is exactly
# the dimension a press fit cannot afford to lose.  The fix is one
# uniform outward offset of the material boundary by kerf/2, applied
# here, after the slots are already cut in.  The burn then gives back
# precisely what was designed, slots included.  Compensating the
# outline and the slot separately (the tempting version) double-counts
# and leaves the joints loose.
#
# CHIRALITY.  A flat part flipped over is its mirror image, and a
# flared slot mouth is not symmetric, so "which way up" is a real
# question.  The convention, fixed here and nowhere else: every part is
# laid out as seen from the +normal side of its own slice plane -- which
# is what the (u, v) frame already gives, since u x v = n -- and the
# engraved label goes on that same face.  Cut label-side up and the
# assembly matches the preview.
#
# NESTING.  Shelf packing on bounding boxes: sort by height, fill a row
# left to right, start a new row when the row is full, start a new sheet
# when the sheet is full.  Rotation and true no-fit-polygon nesting
# would pack tighter, and are a refinement, not a correctness issue --
# but a part that does not fit the stock AT ALL is a hard error and is
# reported rather than quietly scaled or clipped.

from . import polyclip as pc
from . import glyphs
from .drawing import Drawing


def kerf_compensate(part, kerf):
    """Offset the material boundary outward by half the kerf.

    The outline grows; holes shrink -- a hole is a region of absence, so
    the path that leaves the right hole behind is a smaller one.
    """
    if kerf <= 0.0:
        return part
    r = 0.5 * kerf
    outer = pc.offset_polygon(pc.as_ccw(part.outer), r)
    holes = []
    for h in part.holes:
        # treat the hole as a region, shrink it, put the winding back
        shrunk = pc.offset_polygon(pc.as_ccw(h), -r)
        if pc.area(shrunk) > 1e-12:
            holes.append(pc.as_cw(shrunk))
    part.outer = outer
    part.holes = holes
    return part


def _label_strokes(part, height, inset=0.0):
    """Engraving strokes for a part's label, tucked inside the part.

    Placed at the part's area centroid when that lands in the material,
    and skipped when it does not -- engraving a label into thin air (or
    across a hole) is worse than no label.
    """
    if not part.label:
        return []
    cx, cy = pc.centroid(part.outer)
    if not pc.point_in_polygon((cx, cy), part.outer):
        return []
    if any(pc.point_in_polygon((cx, cy), h) for h in part.holes):
        return []
    w = glyphs.text_width(part.label, height)
    strokes = glyphs.text_strokes(part.label, height,
                                  (cx - 0.5 * w, cy - 0.5 * height))
    # Test the strokes themselves rather than a clearance radius: a
    # radius test compares a wide short label against its longest
    # dimension and refuses labels that plainly fit.  What actually has
    # to be true is that every engraved point lands in material, so
    # check exactly that.
    for stroke in strokes:
        for pt in stroke:
            if not pc.point_in_polygon(pt, part.outer):
                return []
            if any(pc.point_in_polygon(pt, h) for h in part.holes):
                return []
    return strokes


def nest(parts, sheet_width, sheet_height, margin=3.0, kerf=0.0,
         label_height=0.0, name='slices'):
    """Place every part on a sheet and return (Drawing, report).

    Parts arrive in plane coordinates centred wherever the geometry put
    them; each is translated so its bounding box sits at the placement
    point.  Nothing is rotated, so the chirality convention above holds
    all the way to the exporter.
    """
    d = Drawing(name, sheet_width, sheet_height)
    report = {'placed': 0, 'oversize': 0, 'sheets': 0, 'labels': 0,
              'errors': 0}

    usable_w = sheet_width - 2.0 * margin
    usable_h = sheet_height - 2.0 * margin

    prepared = []
    for part in parts:
        kerf_compensate(part, kerf)
        w, h = part.size()
        if w > usable_w or h > usable_h:
            part.fail('oversize',
                      f"{w:.1f} x {h:.1f} mm does not fit "
                      f"{usable_w:.1f} x {usable_h:.1f} mm of usable stock")
            report['oversize'] += 1
            continue
        prepared.append((h, w, part))

    # tallest first: shelf packing wastes least when rows are uniform
    prepared.sort(key=lambda r: -r[0])

    sheet_i = 0
    cx = margin
    cy = margin
    row_h = 0.0
    for h, w, part in prepared:
        if cx + w > margin + usable_w and row_h > 0.0:
            cx = margin
            cy += row_h + margin
            row_h = 0.0
        if cy + h > margin + usable_h and (cy > margin or row_h > 0.0):
            sheet_i += 1
            cx, cy, row_h = margin, margin, 0.0

        x0, y0, _, _ = part.bounds()
        placed = part.translated(cx - x0, cy - y0)
        sheet = d.sheet(sheet_i)

        if label_height > 0.0:
            for stroke in _label_strokes(placed, label_height):
                sheet.add('ENGRAVE', stroke, False, placed.label)
                report['labels'] += 1
        for hole in placed.holes:
            sheet.add('HOLE', hole, True, placed.label)
        layer = 'CUT' if not placed.errors else 'ERROR'
        sheet.add(layer, placed.outer, True, placed.label)
        if placed.errors:
            report['errors'] += 1

        report['placed'] += 1
        cx += w + margin
        row_h = max(row_h, h)

    report['sheets'] = len(d.sheets)
    return d, report


# ------------------------------------------------------------------ #

def _selftest():
    from .parts import Part

    def square(side, label=''):
        p = Part([(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)])
        p.label = label
        return p

    # --- kerf grows the outline and shrinks the hole --------------
    p = Part([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
             [[(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)]])
    before_out = pc.area(p.outer)
    before_hole = pc.area(p.holes[0])
    kerf_compensate(p, 0.2)
    assert pc.area(p.outer) > before_out, "kerf grows the outline"
    assert pc.area(p.holes[0]) < before_hole, "kerf shrinks the hole"
    assert abs(pc.area(p.outer) - 10.2 ** 2) < 1e-9, pc.area(p.outer)
    assert abs(pc.area(p.holes[0]) - 3.8 ** 2) < 1e-9, pc.area(p.holes[0])
    assert pc.signed_area(p.holes[0]) < 0, "hole winding preserved"

    # --- nesting places parts without overlap, inside the sheet ---
    parts = [square(20.0, f"X-{i:02d}") for i in range(12)]
    d, rep = nest(parts, 100.0, 100.0, margin=5.0, label_height=4.0)
    assert rep['placed'] == 12, rep
    assert rep['oversize'] == 0, rep
    assert not d.bounds_ok(), f"parts must stay on the stock: {d.bounds_ok()}"

    boxes = []
    for s in d.sheets:
        for e in s.entities:
            if e.layer == 'CUT':
                boxes.append((s.index, pc.bounds(e.points)))
    assert len(boxes) == 12, f"one outline per part, got {len(boxes)}"
    for i in range(len(boxes)):
        si, (ax0, ay0, ax1, ay1) = boxes[i]
        for j in range(i + 1, len(boxes)):
            sj, (bx0, by0, bx1, by1) = boxes[j]
            if si != sj:
                continue
            overlap = (ax0 < bx1 - 1e-9 and bx0 < ax1 - 1e-9
                       and ay0 < by1 - 1e-9 and by0 < ay1 - 1e-9)
            assert not overlap, f"parts {i} and {j} overlap on sheet {si}"

    # rows cost a margin between parts as well as at the edges, so
    # 90 mm of usable stock takes three 20 mm parts per row, not four
    assert rep['sheets'] == 2, \
        f"twelve 20 mm parts pack 3x3 per 100 mm sheet: {rep}"
    # give them room and they must collapse onto a single sheet -- the
    # check that nesting is not just opening a sheet per part
    _, rep_big = nest([square(20.0, f"X-{i:02d}") for i in range(12)],
                      200.0, 200.0, margin=5.0)
    assert rep_big['sheets'] == 1, f"a big sheet holds all twelve: {rep_big}"

    # --- a part bigger than the stock is an error, not a squeeze --
    d2, rep2 = nest([square(500.0, 'BIG')], 100.0, 100.0, margin=5.0)
    assert rep2['oversize'] == 1 and rep2['placed'] == 0, rep2

    # --- overflow really does open another sheet ------------------
    many = [square(40.0, f"Y-{i:02d}") for i in range(9)]
    d3, rep3 = nest(many, 100.0, 100.0, margin=5.0)
    assert rep3['placed'] == 9, rep3
    assert rep3['sheets'] >= 3, f"nine 40 mm parts need several sheets: {rep3}"
    assert not d3.bounds_ok(), "still inside the stock after wrapping"

    # --- labels are engraved, and land inside the part ------------
    assert rep['labels'] > 0, "labels engraved"
    for s in d.sheets:
        outlines = [e for e in s.entities if e.layer == 'CUT']
        for e in s.entities:
            if e.layer != 'ENGRAVE':
                continue
            owner = [o for o in outlines if o.tag == e.tag]
            assert owner, f"engraving {e.tag} has no part"
            for pt in e.points:
                assert pc.point_in_polygon(pt, owner[0].points), \
                    f"label stroke for {e.tag} fell outside its part"

    # --- a part too small for its label just goes unlabelled ------
    tiny = square(2.0, 'Z-99')
    d4, rep4 = nest([tiny], 100.0, 100.0, margin=5.0, label_height=10.0)
    assert rep4['placed'] == 1 and rep4['labels'] == 0, \
        "an unfittable label is skipped, not scrawled across the part"

    # --- a failed part is drawn on ERROR, never on CUT ------------
    bad = square(20.0, 'B-01')
    bad.fail('slot_clip', 'test')
    d5, rep5 = nest([bad], 100.0, 100.0, margin=5.0)
    assert rep5['errors'] == 1, rep5
    assert d5.counts().get('CUT', 0) == 0, "a broken part must not be cut"
    assert d5.counts().get('ERROR', 0) == 1, d5.counts()

    return True
