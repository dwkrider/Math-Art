# SVG serializer for a Drawing.
#
# SVG has no concept of a named layer, so the operation a path belongs
# to is carried the way laser software actually reads it: as stroke
# COLOUR, with one <g> per layer so the grouping survives a round trip
# through an editor.  Drivers map colour to power; the colours come from
# drawing.LAYER_STYLE so the DXF's named layers and these colours are
# two views of one decision.
#
# Group order is the cut order, and that is load-bearing rather than
# cosmetic -- see drawing.py.
#
# The document is declared in real millimetres (width/height in mm with
# a matching viewBox), and the Y axis is flipped on the way out: SVG
# counts Y downward, the drawing counts it upward, and a job exported
# without that flip is mirrored -- which for a chirality-sensitive part
# is a scrap sheet, not a cosmetic problem.

from .drawing import LAYER_ORDER, LAYER_STYLE

HAIRLINE = 0.05          # mm; thin enough that any driver reads a cut


def _fmt(x):
    return f"{x:.4f}".rstrip('0').rstrip('.')


def _path(points, closed, height):
    d = []
    for k, (x, y) in enumerate(points):
        d.append(f"{'M' if k == 0 else 'L'}{_fmt(x)},{_fmt(height - y)}")
    if closed:
        d.append('Z')
    return ''.join(d)


def sheet_svg(sheet, include_frame=True):
    """One sheet as a complete SVG document."""
    w, h = sheet.width, sheet.height
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
           f'width="{_fmt(w)}mm" height="{_fmt(h)}mm" '
           f'viewBox="0 0 {_fmt(w)} {_fmt(h)}">']
    entities = sheet.ordered()
    for layer in LAYER_ORDER:
        group = [e for e in entities if e.layer == layer]
        if not group or (layer == 'SHEET' and not include_frame):
            continue
        rgb, _aci, _cuts = LAYER_STYLE[layer]
        colour = 'rgb({},{},{})'.format(*rgb)
        out.append(f'<g id="{layer}" fill="none" stroke="{colour}" '
                   f'stroke-width="{HAIRLINE}">')
        for e in group:
            out.append(f'<path d="{_path(e.points, e.closed, h)}"/>')
        out.append('</g>')
    out.append('</svg>')
    return '\n'.join(out)


def write(drawing, path_for_sheet):
    """Write one SVG per sheet.  `path_for_sheet(index)` gives the
    filename; returns the list written."""
    written = []
    for sheet in drawing.sheets:
        p = path_for_sheet(sheet.index)
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(sheet_svg(sheet))
        written.append(p)
    return written


# ------------------------------------------------------------------ #

def _selftest():
    import xml.etree.ElementTree as ET

    from .drawing import Drawing

    d = Drawing('t', 100.0, 60.0)
    s = d.sheet(0)
    s.add('CUT', [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],
          True, 'p1')
    s.add('HOLE', [(12.0, 12.0), (14.0, 12.0), (14.0, 14.0)], True, 'h')
    s.add('ENGRAVE', [(11.0, 11.0), (13.0, 11.0)], False, 'lbl')

    text = sheet_svg(s)
    root = ET.fromstring(text)
    assert root.tag.endswith('svg'), root.tag

    # the document must claim real millimetres, or the part comes out
    # whatever size the reader guesses
    assert root.get('width') == '100mm', root.get('width')
    assert root.get('height') == '60mm', root.get('height')
    assert root.get('viewBox') == '0 0 100 60', root.get('viewBox')

    ns = '{http://www.w3.org/2000/svg}'
    groups = root.findall(f'{ns}g')
    ids = [g.get('id') for g in groups]
    assert ids == ['SHEET', 'ENGRAVE', 'HOLE', 'CUT'], \
        f"groups must come out in cut order, got {ids}"

    paths = {g.get('id'): g.findall(f'{ns}path') for g in groups}
    assert len(paths['CUT']) == 1 and len(paths['HOLE']) == 1, \
        "one path per entity"

    # closed rings close, open strokes do not
    assert paths['CUT'][0].get('d').endswith('Z'), "a ring must close"
    assert not paths['ENGRAVE'][0].get('d').endswith('Z'), \
        "an engraved stroke is open"

    # Y is flipped: the ring's top edge at y=20 must land at 60-20=40
    dattr = paths['CUT'][0].get('d')
    assert '10,50' in dattr and '10,40' in dattr, \
        f"Y must be flipped for SVG's downward axis: {dattr}"

    # colours come from the shared table, so SVG and DXF agree
    cut_rgb = LAYER_STYLE['CUT'][0]
    assert groups[ids.index('CUT')].get('stroke') == \
        'rgb({},{},{})'.format(*cut_rgb), "cut colour from the shared table"

    # dropping the frame drops only the frame
    plain = ET.fromstring(sheet_svg(s, include_frame=False))
    assert [g.get('id') for g in plain.findall(f'{ns}g')] == \
        ['ENGRAVE', 'HOLE', 'CUT'], "frame suppressed on request"

    return True
