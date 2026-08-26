# DXF serializer for a Drawing.
#
# WHY DXF AT ALL, GIVEN SVG.  Because DXF carries NAMED LAYERS as a
# first-class thing: a LAYER table, and every entity naming the layer it
# belongs to.  For output whose whole point is "these lines mean
# different operations -- engrave this, cut that, and never fire on
# this", that is a much better match than SVG, where the distinction has
# to be smuggled through stroke colour and recovered by convention at
# the far end.  Both are written; this is the one that says what it
# means.
#
# VERSION: R12 (AC1009), ASCII.  This is the version laser and CAM
# software accepts most reliably.  Nothing is lost by targeting it: the
# entities that need something later -- LWPOLYLINE (R14+) and SPLINE
# (R14+) -- are ones this tool never emits, because every curve it
# produces is already a polyline.  R12 has no LWPOLYLINE, so a ring is
# POLYLINE / VERTEX... / SEQEND, which is verbose and universally read.
#
# UNITS ARE THE TRAP.  A DXF coordinate is a bare number: the format
# attaches no unit to it.  Modelspace units come from the $INSUNITS
# header variable (4 = millimetres), which not every R12 reader honours.
# So this writer does all three of the things that can be done: it
# writes $INSUNITS = 4 and $MEASUREMENT = 1 for readers that look, it
# emits coordinates such that one drawing unit is exactly one
# millimetre, and the operator's UI says so out loud.  Get this wrong
# and parts come out 25.4 times off, which is the classic way a DXF job
# is scrapped.
#
# Layers are named, not ordered, so the cut-order trick the SVG gets
# from group order does not carry over.  Entities are still emitted
# interior-first as a fallback for CAM that respects file order, and the
# layer NAMES are what an operator sequences by.

from .drawing import LAYER_ORDER, LAYER_STYLE

INSUNITS_MM = 4          # $INSUNITS code for millimetres
MEASUREMENT_METRIC = 1


def _g(code, value):
    """One DXF group: a code line and a value line."""
    if isinstance(value, float):
        return f"{code:>3}\n{value:.6f}\n"
    return f"{code:>3}\n{value}\n"


def _header():
    return (_g(0, 'SECTION') + _g(2, 'HEADER')
            + _g(9, '$ACADVER') + _g(1, 'AC1009')
            + _g(9, '$INSUNITS') + _g(70, INSUNITS_MM)
            + _g(9, '$MEASUREMENT') + _g(70, MEASUREMENT_METRIC)
            + _g(0, 'ENDSEC'))


def _tables(layers):
    out = _g(0, 'SECTION') + _g(2, 'TABLES')
    out += _g(0, 'TABLE') + _g(2, 'LAYER') + _g(70, len(layers))
    for name in layers:
        _rgb, aci, _cuts = LAYER_STYLE[name]
        out += (_g(0, 'LAYER') + _g(2, name) + _g(70, 0)
                + _g(62, aci) + _g(6, 'CONTINUOUS'))
    out += _g(0, 'ENDTAB') + _g(0, 'ENDSEC')
    return out


def _polyline(points, closed, layer):
    out = (_g(0, 'POLYLINE') + _g(8, layer) + _g(66, 1)
           + _g(10, 0.0) + _g(20, 0.0) + _g(30, 0.0)
           + _g(70, 1 if closed else 0))
    for x, y in points:
        out += (_g(0, 'VERTEX') + _g(8, layer)
                + _g(10, float(x)) + _g(20, float(y)) + _g(30, 0.0))
    out += _g(0, 'SEQEND') + _g(8, layer)
    return out


def job_dxf(drawing, include_frame=True, gap=20.0):
    """The whole job as one DXF, sheets tiled side by side.

    The operation layers are shared across the file, which is the
    point: one document whose CUT, HOLE and ENGRAVE layers can each be
    switched off or mapped to a power in a single action, however many
    sheets the job runs to.  Each sheet keeps its own frame on the
    non-cutting SHEET layer, so where one ends and the next begins is
    still legible.
    """
    placed, x = [], 0.0
    for sheet in drawing.sheets:
        for e in sheet.ordered():
            if not include_frame and e.layer == 'SHEET':
                continue
            placed.append((e.layer,
                           [(px + x, py) for px, py in e.points],
                           e.closed))
        x += sheet.width + gap

    used = [n for n in LAYER_ORDER if any(p[0] == n for p in placed)]
    body = _g(0, 'SECTION') + _g(2, 'ENTITIES')
    for layer in LAYER_ORDER:
        for name, pts, closed in placed:
            if name == layer:
                body += _polyline(pts, closed, layer)
    body += _g(0, 'ENDSEC')
    return _header() + _tables(used or ['CUT']) + body + _g(0, 'EOF')


def sheet_dxf(sheet, include_frame=True):
    """One sheet as a complete DXF R12 document."""
    entities = sheet.ordered()
    if not include_frame:
        entities = [e for e in entities if e.layer != 'SHEET']
    used = [n for n in LAYER_ORDER if any(e.layer == n for e in entities)]

    body = _g(0, 'SECTION') + _g(2, 'ENTITIES')
    for layer in LAYER_ORDER:
        for e in entities:
            if e.layer == layer:
                body += _polyline(e.points, e.closed, layer)
    body += _g(0, 'ENDSEC')

    return _header() + _tables(used or ['CUT']) + body + _g(0, 'EOF')


def write(drawing, path, include_frame=True):
    """Write the whole job as ONE DXF file.

    DXF has no page, so a multi-sheet job cannot be paginated -- but it
    can be a single drawing, which is what one file has to mean: the
    sheets tiled side by side, sharing one set of named operation
    layers that can each be switched on or mapped to a power in one go.
    """
    with open(path, 'w', encoding='ascii', newline='\r\n') as fh:
        fh.write(job_dxf(drawing, include_frame))
    return [path]


def parse_groups(text):
    """(code, value) pairs -- a minimal reader, for the checks below."""
    lines = text.splitlines()
    out = []
    for i in range(0, len(lines) - 1, 2):
        out.append((int(lines[i].strip()), lines[i + 1].strip()))
    return out


# ------------------------------------------------------------------ #

def _selftest():
    from .drawing import Drawing

    d = Drawing('t', 100.0, 60.0)
    s = d.sheet(0)
    s.add('CUT', [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],
          True, 'p1')
    s.add('HOLE', [(12.0, 12.0), (14.0, 12.0), (14.0, 14.0)], True, 'h')
    s.add('ENGRAVE', [(11.0, 11.0), (13.0, 11.0)], False, 'lbl')

    text = sheet_dxf(s)
    groups = parse_groups(text)
    assert len(groups) * 2 == len(text.splitlines()), \
        "every group is exactly a code line and a value line"

    # --- structure: sections open and close, file ends with EOF ---
    depth = 0
    for code, val in groups:
        if code == 0 and val == 'SECTION':
            depth += 1
        elif code == 0 and val == 'ENDSEC':
            depth -= 1
            assert depth >= 0, "ENDSEC without SECTION"
    assert depth == 0, "every SECTION is closed"
    assert groups[-1] == (0, 'EOF'), f"file must end with EOF: {groups[-1]}"

    # --- it must declare R12 and millimetres ----------------------
    assert (1, 'AC1009') in groups, "must declare itself as R12"
    ins = [groups[i + 1] for i, g in enumerate(groups)
           if g == (9, '$INSUNITS')]
    assert ins and ins[0] == (70, str(INSUNITS_MM)), \
        f"$INSUNITS must say millimetres, got {ins}"
    meas = [groups[i + 1] for i, g in enumerate(groups)
            if g == (9, '$MEASUREMENT')]
    assert meas and meas[0] == (70, '1'), "must declare itself metric"

    # --- every entity's layer exists in the LAYER table -----------
    declared, in_table = set(), False
    for i, (code, val) in enumerate(groups):
        if code == 2 and val == 'LAYER':
            in_table = True
        if code == 0 and val == 'ENDTAB':
            in_table = False
        if in_table and code == 0 and val == 'LAYER':
            declared.add(groups[i + 1][1])
    used = {val for i, (code, val) in enumerate(groups)
            if code == 8}
    assert used <= declared, \
        f"entities on undeclared layers: {used - declared}"
    assert 'CUT' in declared and 'HOLE' in declared, declared

    # --- one POLYLINE per entity, closed flag honoured ------------
    polys = [i for i, g in enumerate(groups) if g == (0, 'POLYLINE')]
    assert len(polys) == 4, f"frame + three entities, got {len(polys)}"
    seq = sum(1 for g in groups if g == (0, 'SEQEND'))
    assert seq == len(polys), "every POLYLINE is terminated by SEQEND"

    def closed_flag(start):
        for code, val in groups[start:start + 8]:
            if code == 70:
                return val
        return None
    kinds = {}
    for p in polys:
        layer = groups[p + 1][1]
        kinds[layer] = closed_flag(p)
    assert kinds['CUT'] == '1', "a ring is a closed polyline"
    assert kinds['ENGRAVE'] == '0', "an engraved stroke stays open"

    # --- coordinates are millimetres, unflipped -------------------
    xs = [float(v) for c, v in groups if c == 10]
    ys = [float(v) for c, v in groups if c == 20]
    assert max(xs) == 100.0 and max(ys) == 60.0, \
        "the sheet frame carries the true sheet size in mm"
    assert 20.0 in ys, "DXF keeps Y upward -- no flip, unlike SVG"

    # --- SVG and DXF must describe the SAME drawing ---------------
    from . import svg as _svg
    import xml.etree.ElementTree as ET
    ns = '{http://www.w3.org/2000/svg}'
    root = ET.fromstring(_svg.sheet_svg(s))
    svg_counts = {g.get('id'): len(g.findall(f'{ns}path'))
                  for g in root.findall(f'{ns}g')}
    dxf_counts = {}
    for p in polys:
        layer = groups[p + 1][1]
        dxf_counts[layer] = dxf_counts.get(layer, 0) + 1
    assert svg_counts == dxf_counts, (
        f"the two exporters must agree polyline-for-polyline: "
        f"svg={svg_counts} dxf={dxf_counts}")

    # --- the whole job in ONE file, sheets tiled ------------------
    two = Drawing('t', 100.0, 60.0)
    two.sheet(0).add('CUT', [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                     True, 'a')
    two.sheet(1).add('CUT', [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                     True, 'b')
    job = parse_groups(job_dxf(two))
    polys = sum(1 for g in job if g == (0, 'POLYLINE'))
    assert polys == 4, f"two parts and two sheet frames, got {polys}"
    assert sum(1 for c, v in job if c == 0 and v == 'EOF') == 1,         "one file means one EOF, not one per sheet"
    xs = [float(v) for c, v in job if c == 10]
    assert max(xs) > 100.0,         "the second sheet must be tiled clear of the first, not stacked"
    layers = {v for c, v in job if c == 8}
    assert layers == {'CUT', 'SHEET'},         f"operation layers are shared across the file: {layers}"

    # --- dropping the frame drops it from both the table and body -
    plain = parse_groups(sheet_dxf(s, include_frame=False))
    assert not any(c == 8 and v == 'SHEET' for c, v in plain), \
        "frame suppressed on request"

    return True
