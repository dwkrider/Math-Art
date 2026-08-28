# The device-independent drawing: what gets cut, on which sheet, in
# which layer -- with no idea that SVG or DXF exist.
#
# Two exporters read this and nothing else.  That is the point: the
# layer set, the cut order and the sheet geometry are decided ONCE here,
# so the SVG and the DXF of the same job cannot drift apart.  A check in
# the DXF self-test asserts exactly that -- same drawing in, same
# polyline count per layer out.
#
# LAYER ORDER IS A CORRECTNESS CONTRACT, NOT PRESENTATION.  Interior
# work has to happen before the outline releases the part: engrave the
# label, cut the holes, and only then cut the boundary.  A machine that
# cuts the outline first drops the piece onto the honeycomb and
# everything after that is scrap.  `LAYER_ORDER` is that sequence, and
# both exporters emit in it.
#
# UNITS ARE MILLIMETRES, ALWAYS.  Coordinates in this model are real
# millimetres.  SVG can say so in its own header; DXF cannot be trusted
# to (see dxf.py), which is why the convention is fixed here rather
# than negotiated per format.

# name -> (RGB for SVG, AutoCAD Color Index for DXF, does it cut)
LAYER_STYLE = {
    'ENGRAVE': ((128, 128, 128), 8, True),
    'HOLE':    ((0, 160, 0), 3, True),
    'CUT':     ((0, 0, 255), 5, True),
    'ERROR':   ((255, 0, 0), 1, False),
    'SHEET':   ((210, 170, 0), 2, False),
}

# the sequence a machine should work in
LAYER_ORDER = ('SHEET', 'ENGRAVE', 'HOLE', 'CUT', 'ERROR')

# the subset that actually burns, in cut order
CUT_ORDER = tuple(n for n in LAYER_ORDER if LAYER_STYLE[n][2])


class Entity:
    """One polyline on one layer."""

    __slots__ = ('layer', 'points', 'closed', 'tag')

    def __init__(self, layer, points, closed=True, tag=''):
        if layer not in LAYER_STYLE:
            raise KeyError(f"unknown layer {layer!r}")
        self.layer = layer
        self.points = [(float(x), float(y)) for x, y in points]
        self.closed = bool(closed)
        self.tag = tag


class Sheet:
    """One piece of stock, and everything cut from it."""

    def __init__(self, index, width, height):
        self.index = index
        self.width = float(width)
        self.height = float(height)
        self.entities = []

    def add(self, layer, points, closed=True, tag=''):
        if len(points) < 2:
            return None
        e = Entity(layer, points, closed, tag)
        self.entities.append(e)
        return e

    def frame(self):
        """The non-cutting sheet outline."""
        return [(0.0, 0.0), (self.width, 0.0),
                (self.width, self.height), (0.0, self.height)]

    def ordered(self):
        """Entities in cut order, sheet frame first."""
        out = [Entity('SHEET', self.frame(), True, 'sheet')]
        for layer in LAYER_ORDER:
            if layer == 'SHEET':
                continue
            out.extend(e for e in self.entities if e.layer == layer)
        return out

    def counts(self):
        c = {}
        for e in self.ordered():
            c[e.layer] = c.get(e.layer, 0) + 1
        return c


class Drawing:
    """A whole job: sheets of parts, in millimetres."""

    def __init__(self, name='slices', sheet_width=600.0,
                 sheet_height=400.0):
        self.name = name
        self.sheet_width = float(sheet_width)
        self.sheet_height = float(sheet_height)
        self.sheets = []
        self.notes = []

    def sheet(self, index):
        while len(self.sheets) <= index:
            self.sheets.append(Sheet(len(self.sheets), self.sheet_width,
                                     self.sheet_height))
        return self.sheets[index]

    def counts(self):
        """Polylines per layer across the whole job."""
        c = {}
        for s in self.sheets:
            for layer, k in s.counts().items():
                c[layer] = c.get(layer, 0) + k
        return c

    def bounds_ok(self, eps=1e-6):
        """Every entity inside its own sheet -- a part hanging off the
        stock is not cuttable, and silently clipping it at export time
        would hide that."""
        bad = []
        for s in self.sheets:
            for e in s.entities:
                for x, y in e.points:
                    if (x < -eps or y < -eps or x > s.width + eps
                            or y > s.height + eps):
                        bad.append((s.index, e.tag))
                        break
        return bad


# ------------------------------------------------------------------ #

def _selftest():
    d = Drawing('t', 100.0, 50.0)
    s = d.sheet(0)
    sq = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
    s.add('CUT', sq, True, 'p1')
    s.add('HOLE', [(12.0, 12.0), (14.0, 12.0), (14.0, 14.0)], True, 'h')
    s.add('ENGRAVE', [(11.0, 11.0), (13.0, 11.0)], False, 'lbl')

    # sheets are created on demand, in order
    d.sheet(2)
    assert len(d.sheets) == 3, "asking for sheet 2 creates 0, 1 and 2"
    assert d.sheets[1].width == 100.0, "new sheets take the job's size"

    # cut order: engrave, then holes, then the outline that frees the
    # part -- with the non-cutting frame ahead of all of it
    layers = [e.layer for e in s.ordered()]
    assert layers[0] == 'SHEET', layers
    assert layers.index('ENGRAVE') < layers.index('HOLE') < \
        layers.index('CUT'), f"interior work must precede the outline: {layers}"

    # CUT_ORDER excludes the layers that must never fire the laser
    assert 'SHEET' not in CUT_ORDER and 'ERROR' not in CUT_ORDER, CUT_ORDER
    assert CUT_ORDER == ('ENGRAVE', 'HOLE', 'CUT'), CUT_ORDER

    c = d.counts()
    assert c['CUT'] == 1 and c['HOLE'] == 1 and c['ENGRAVE'] == 1, c
    assert c['SHEET'] == 3, "one frame per sheet"

    # an unknown layer is a mistake, caught where it is made
    try:
        s.add('WHATEVER', sq)
        raise AssertionError("unknown layer should raise")
    except KeyError:
        pass

    # a two-point degenerate ring is dropped rather than exported
    assert s.add('CUT', [(0.0, 0.0)]) is None, "a single point is not a path"

    assert not d.bounds_ok(), "everything so far fits the sheet"
    s.add('CUT', [(90.0, 10.0), (130.0, 10.0), (130.0, 20.0)], True, 'over')
    bad = d.bounds_ok()
    assert bad and bad[0][1] == 'over', f"overhanging part must be flagged: {bad}"

    return True
