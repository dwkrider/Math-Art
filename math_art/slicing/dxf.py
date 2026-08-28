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

# DXF is a CRLF format.  Named, not spelled out inline:
# an escaped literal has been mangled once already.
_CRLF = chr(13) + chr(10)

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


def _tables_named(pairs):
    """LAYER table from (layer name, operation) pairs, so a per-sheet
    name still takes its operation's colour."""
    out = _g(0, 'SECTION') + _g(2, 'TABLES')
    out += _g(0, 'TABLE') + _g(2, 'LAYER') + _g(70, len(pairs))
    for name, op in pairs:
        _rgb, aci, _cuts = LAYER_STYLE[op]
        out += (_g(0, 'LAYER') + _g(2, name) + _g(70, 0)
                + _g(62, aci) + _g(6, 'CONTINUOUS'))
    out += _g(0, 'ENDTAB') + _g(0, 'ENDSEC')
    return out


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


def layer_name(sheet_index, operation, sheets=1):
    """`CUT` for a one-sheet job, `S02_CUT` for a longer one.

    A job of one sheet gains nothing from a prefix, and a job of six
    needs one badly.
    """
    if sheets <= 1:
        return operation
    return f"S{sheet_index + 1:02d}_{operation}"


def job_dxf(drawing, include_frame=True):
    """The whole job as one DXF, ONE LAYER SET PER SHEET.

    Two decisions here, both aimed at the machine rather than at the
    screen.

    Every sheet is drawn at the SAME ORIGIN, stacked rather than tiled.
    A laser operator isolates one sheet's layers, and what they then
    want is that sheet sitting at 0,0 ready to cut -- not sitting two
    metres to the right because it happened to be fifth in the job.

    And every sheet gets its own layers -- S01_CUT, S01_HOLE, S02_CUT
    and so on -- so a sheet can be shown, hidden or sent on its own.
    Sharing one CUT layer across the job puts all six sheets on top of
    one another with no way to separate them, which is the one thing a
    single file must not do.  The operation stays in the name so the
    cut/engrave distinction, and its colour, survive.
    """
    n = len(drawing.sheets)
    placed = []
    for sheet in drawing.sheets:
        for e in sheet.ordered():
            if not include_frame and e.layer == 'SHEET':
                continue
            placed.append((layer_name(sheet.index, e.layer, n),
                           e.layer, list(e.points), e.closed))

    used, seen = [], set()
    for op in LAYER_ORDER:
        for sheet in drawing.sheets:
            name = layer_name(sheet.index, op, n)
            if name not in seen and any(p[0] == name for p in placed):
                seen.add(name)
                used.append((name, op))

    body = _g(0, 'SECTION') + _g(2, 'ENTITIES')
    for name, _op in used:
        for lname, _base, pts, closed in placed:
            if lname == name:
                body += _polyline(pts, closed, lname)
    body += _g(0, 'ENDSEC')
    return _header() + _tables_named(used) + body + _g(0, 'EOF')


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


def write_sheets(drawing, path_for_sheet, include_frame=True):
    """One DXF per sheet, each with PLAIN operation layer names.

    The alternative to the single layered file, and it exists because
    some CAM software copes badly with a drawing carrying many layers.
    Splitting the job into a file per sheet means each one needs only
    CUT, HOLE and ENGRAVE -- no per-sheet prefixes, because the file
    itself is the sheet.
    """
    written = []
    for sheet in drawing.sheets:
        path = path_for_sheet(sheet.index)
        with open(path, 'w', encoding='ascii',
                  newline=_CRLF) as fh:
            fh.write(sheet_dxf(sheet, include_frame))
        written.append(path)
    return written


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

    # --- the whole job in ONE file, a layer set per sheet ---------
    two = Drawing('t', 100.0, 60.0)
    two.sheet(0).add('CUT', [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                     True, 'a')
    two.sheet(1).add('CUT', [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                     True, 'b')
    job = parse_groups(job_dxf(two))
    polys = sum(1 for g in job if g == (0, 'POLYLINE'))
    assert polys == 4, f"two parts and two sheet frames, got {polys}"
    assert sum(1 for c, v in job if c == 0 and v == 'EOF') == 1,         "one file means one EOF, not one per sheet"
    layers = {v for c, v in job if c == 8}
    assert layers == {'S01_CUT', 'S01_SHEET', 'S02_CUT', 'S02_SHEET'}, (
        "each sheet needs its OWN layers or the six of them land on top "
        f"of one another with no way to separate them: {sorted(layers)}")
    xs = [float(v) for c, v in job if c == 10]
    assert max(xs) <= 100.0 + 1e-9, (
        "sheets stack at a common origin -- an isolated sheet has to sit "
        f"at 0,0 ready to cut, not offset by its place in the job: {max(xs)}")

    # ... and the file-per-sheet alternative uses plain names in
    # every file, because there the file IS the sheet
    solo_layers = {v for c, v in parse_groups(sheet_dxf(two.sheets[1]))
                   if c == 8}
    assert solo_layers == {'CUT', 'SHEET'},         f"a per-sheet file needs no per-sheet prefix: {sorted(solo_layers)}"

    # a single-sheet job keeps the plain names
    one = Drawing('t', 100.0, 60.0)
    one.sheet(0).add('CUT', [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)], True, 'a')
    solo = {v for c, v in parse_groups(job_dxf(one)) if c == 8}
    assert solo == {'CUT', 'SHEET'},         f"one sheet needs no prefix: {sorted(solo)}"

    # --- dropping the frame drops it from both the table and body -
    plain = parse_groups(sheet_dxf(s, include_frame=False))
    assert not any(c == 8 and v == 'SHEET' for c, v in plain), \
        "frame suppressed on request"

    return True
