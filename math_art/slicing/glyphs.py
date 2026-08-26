# A stroke font for engraved part labels.
#
# Part labels have to survive being cut: a filled outline font would
# need its counters cut out as separate closed loops, and a laser set to
# engrave power wants a PATH to follow, not a shape to fill.  So each
# glyph here is a list of open polylines on a 0..1 by 0..1 box -- single
# strokes, drawn once, exactly what an engraver traces.
#
# The character set is deliberately small: the digits, the hyphen, and
# the capitals the slicer actually emits in family names.  Anything else
# renders as a small box, which reads as "a character was here" instead
# of silently vanishing.  Extending the set is adding one entry.

# Each glyph: list of polylines, each a list of (x, y) in a unit box.
_F = {
    '0': [[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.1, 0.1)],
          [(0.1, 0.1), (0.9, 0.9)]],
    '1': [[(0.3, 0.7), (0.5, 0.9), (0.5, 0.1)], [(0.2, 0.1), (0.8, 0.1)]],
    '2': [[(0.1, 0.75), (0.3, 0.9), (0.7, 0.9), (0.9, 0.75),
           (0.9, 0.6), (0.1, 0.2), (0.1, 0.1), (0.9, 0.1)]],
    '3': [[(0.1, 0.9), (0.9, 0.9), (0.45, 0.55), (0.9, 0.5),
           (0.9, 0.2), (0.7, 0.1), (0.25, 0.1), (0.1, 0.2)]],
    '4': [[(0.7, 0.1), (0.7, 0.9), (0.1, 0.35), (0.9, 0.35)]],
    '5': [[(0.9, 0.9), (0.15, 0.9), (0.15, 0.55), (0.7, 0.55),
           (0.9, 0.42), (0.9, 0.22), (0.72, 0.1), (0.2, 0.1)]],
    '6': [[(0.85, 0.85), (0.55, 0.9), (0.2, 0.75), (0.1, 0.35),
           (0.25, 0.1), (0.65, 0.1), (0.85, 0.25), (0.85, 0.4),
           (0.65, 0.55), (0.25, 0.55), (0.1, 0.42)]],
    '7': [[(0.1, 0.9), (0.9, 0.9), (0.4, 0.1)]],
    '8': [[(0.35, 0.52), (0.15, 0.63), (0.15, 0.8), (0.32, 0.9),
           (0.68, 0.9), (0.85, 0.8), (0.85, 0.63), (0.65, 0.52),
           (0.32, 0.52), (0.1, 0.36), (0.1, 0.2), (0.3, 0.1),
           (0.7, 0.1), (0.9, 0.2), (0.9, 0.36), (0.65, 0.52)]],
    '9': [[(0.15, 0.15), (0.45, 0.1), (0.8, 0.25), (0.9, 0.65),
           (0.75, 0.9), (0.35, 0.9), (0.15, 0.75), (0.15, 0.6),
           (0.35, 0.45), (0.75, 0.45), (0.9, 0.58)]],
    '-': [[(0.15, 0.5), (0.85, 0.5)]],
    '.': [[(0.45, 0.1), (0.55, 0.1), (0.55, 0.2), (0.45, 0.2),
           (0.45, 0.1)]],
    'A': [[(0.1, 0.1), (0.5, 0.9), (0.9, 0.1)], [(0.25, 0.4), (0.75, 0.4)]],
    'B': [[(0.15, 0.1), (0.15, 0.9), (0.7, 0.9), (0.85, 0.78),
           (0.85, 0.62), (0.7, 0.5), (0.15, 0.5)],
          [(0.7, 0.5), (0.88, 0.36), (0.88, 0.22), (0.72, 0.1),
           (0.15, 0.1)]],
    'C': [[(0.9, 0.8), (0.7, 0.9), (0.3, 0.9), (0.1, 0.7),
           (0.1, 0.3), (0.3, 0.1), (0.7, 0.1), (0.9, 0.2)]],
    'D': [[(0.15, 0.1), (0.15, 0.9), (0.6, 0.9), (0.85, 0.68),
           (0.85, 0.32), (0.6, 0.1), (0.15, 0.1)]],
    'E': [[(0.9, 0.9), (0.15, 0.9), (0.15, 0.1), (0.9, 0.1)],
          [(0.15, 0.5), (0.7, 0.5)]],
    'F': [[(0.9, 0.9), (0.15, 0.9), (0.15, 0.1)],
          [(0.15, 0.52), (0.7, 0.52)]],
    'H': [[(0.15, 0.9), (0.15, 0.1)], [(0.85, 0.9), (0.85, 0.1)],
          [(0.15, 0.5), (0.85, 0.5)]],
    'I': [[(0.5, 0.9), (0.5, 0.1)], [(0.25, 0.9), (0.75, 0.9)],
          [(0.25, 0.1), (0.75, 0.1)]],
    'L': [[(0.2, 0.9), (0.2, 0.1), (0.9, 0.1)]],
    'N': [[(0.15, 0.1), (0.15, 0.9), (0.85, 0.1), (0.85, 0.9)]],
    'O': [[(0.3, 0.9), (0.7, 0.9), (0.9, 0.7), (0.9, 0.3),
           (0.7, 0.1), (0.3, 0.1), (0.1, 0.3), (0.1, 0.7),
           (0.3, 0.9)]],
    'P': [[(0.15, 0.1), (0.15, 0.9), (0.7, 0.9), (0.88, 0.75),
           (0.88, 0.6), (0.7, 0.46), (0.15, 0.46)]],
    'R': [[(0.15, 0.1), (0.15, 0.9), (0.7, 0.9), (0.88, 0.75),
           (0.88, 0.6), (0.7, 0.46), (0.15, 0.46)],
          [(0.5, 0.46), (0.88, 0.1)]],
    'S': [[(0.9, 0.8), (0.7, 0.9), (0.3, 0.9), (0.12, 0.75),
           (0.12, 0.62), (0.3, 0.5), (0.7, 0.5), (0.88, 0.38),
           (0.88, 0.24), (0.7, 0.1), (0.3, 0.1), (0.1, 0.2)]],
    'T': [[(0.1, 0.9), (0.9, 0.9)], [(0.5, 0.9), (0.5, 0.1)]],
    'U': [[(0.15, 0.9), (0.15, 0.3), (0.35, 0.1), (0.65, 0.1),
           (0.85, 0.3), (0.85, 0.9)]],
    'V': [[(0.1, 0.9), (0.5, 0.1), (0.9, 0.9)]],
    'W': [[(0.05, 0.9), (0.28, 0.1), (0.5, 0.62), (0.72, 0.1),
           (0.95, 0.9)]],
    'X': [[(0.1, 0.9), (0.9, 0.1)], [(0.9, 0.9), (0.1, 0.1)]],
    'Y': [[(0.1, 0.9), (0.5, 0.5), (0.9, 0.9)], [(0.5, 0.5), (0.5, 0.1)]],
    'Z': [[(0.1, 0.9), (0.9, 0.9), (0.1, 0.1), (0.9, 0.1)]],
    ' ': [],
}

_MISSING = [[(0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9), (0.2, 0.1)]]

ADVANCE = 1.25          # glyph pitch, in glyph widths


def glyph(ch):
    """Strokes for one character, on the unit box."""
    return _F.get(ch.upper(), _MISSING)


def text_strokes(text, height=1.0, origin=(0.0, 0.0)):
    """Polylines spelling `text`, baseline-left at `origin`."""
    out = []
    x, y = origin
    for ch in text:
        for stroke in glyph(ch):
            out.append([(x + px * height, y + py * height)
                        for px, py in stroke])
        x += ADVANCE * height
    return out


def text_width(text, height=1.0):
    return max(0.0, len(text) * ADVANCE * height - 0.25 * height)


# ------------------------------------------------------------------ #

def _selftest():
    # every character the slicer can emit has a real glyph, so a label
    # never silently engraves nothing
    for ch in "0123456789-XYZRCS":
        assert glyph(ch) is not _MISSING, f"no glyph for {ch!r}"
        assert glyph(ch), f"empty glyph for {ch!r}"

    # an unsupported character is visibly a box, not nothing
    assert glyph('@') is _MISSING, "unknown characters fall back to a box"

    # strokes stay inside the box they claim, scaled and placed
    for ch, strokes in _F.items():
        for s in strokes:
            for x, y in s:
                assert -1e-9 <= x <= 1.0 + 1e-9, f"{ch}: x={x} out of box"
                assert -1e-9 <= y <= 1.0 + 1e-9, f"{ch}: y={y} out of box"

    strokes = text_strokes("X-01", height=2.0, origin=(5.0, 7.0))
    assert strokes, "text produces strokes"
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    assert min(xs) >= 5.0 - 1e-9, "text starts at its origin"
    assert min(ys) >= 7.0 - 1e-9, "baseline honoured"
    assert max(ys) <= 7.0 + 2.0 + 1e-9, "height honoured"
    assert max(xs) <= 5.0 + text_width("X-01", 2.0) + 1e-9, \
        "advance width covers the drawn strokes"

    # a space draws nothing but still advances
    assert text_strokes(" ") == [], "space is blank"
    assert text_width("AB", 1.0) > text_width("A", 1.0), "width grows"

    return True
