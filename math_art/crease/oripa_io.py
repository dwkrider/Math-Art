# Reading ORIPA `.cp` and `.opx` crease patterns.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python,
# numpy and the standard library only -- no `bpy`.
#
# WHY BOTH.  ORIPA is the original crease-pattern editor and Oriedita
# is its living successor; between them they are what most origami
# designers actually draw in, and `.cp` is what those drawings get
# saved as.  A FOLD-only importer makes every such user convert first,
# which in practice means finding a converter, which in practice means
# not bothering.  `.cp` in particular is trivial to read -- one line per
# crease -- so there is no excuse for not reading it.
#
# THE TWO FORMATS ARE THE SAME DATA IN DIFFERENT CLOTHES.  A `.cp` is
# plain text, one crease per line, `type x0 y0 x1 y1`.  An `.opx` is
# Java's `XMLEncoder` output -- ORIPA is a Java program and simply
# serialised its object graph -- so it is XML describing `OriLine`
# objects with the same four coordinates and the same type code.  Both
# reduce to tagged segments, which `svg_io.frame_from_segments` already
# knows how to weld into a plane graph.
#
# THE TYPE CODES are ORIPA's, and they are NOT the FOLD ones:
#
#     0  auxiliary / construction line     -> FLAT (marked, not folded)
#     1  contour, i.e. the paper's edge    -> BOUNDARY
#     2  mountain                          -> MOUNTAIN
#     3  valley                            -> VALLEY
#     4  auxiliary (later ORIPA versions)  -> FLAT
#
# Getting this table wrong is quiet and total: every crease imports,
# nothing errors, and the model folds inside out.
#
# References:
#   Jun Mitani, "ORIPA: Origami Pattern Editor" (2005-) -- the `.cp`
#       and `.opx` formats.
#   Oriedita (https://oriedita.github.io/) -- the maintained successor,
#       which reads and writes the same `.cp`.
#   E. D. Demaine, J. S. Ku, R. J. Lang, "A New File Standard to
#       Represent Folded Structures," 2016 -- the FOLD assignments
#       these codes are mapped onto.

import re
import xml.etree.ElementTree as ET

from .fold_io import BOUNDARY, FLAT, MOUNTAIN, VALLEY
from .svg_io import SvgError, frame_from_segments

#: ORIPA line-type code to FOLD assignment.  See the header: this table
#: is the whole format, and a wrong entry is invisible until the model
#: folds the wrong way.
_TYPE_ASSIGN = {
    0: FLAT,
    1: BOUNDARY,
    2: MOUNTAIN,
    3: VALLEY,
    4: FLAT,
}

_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"


class OripaError(SvgError):
    """The file is not usable as an ORIPA crease pattern."""


def _read_text(path_or_text, marker):
    """Return the file's text, whether given a path or the text itself."""
    if isinstance(path_or_text, str) and marker in path_or_text:
        return path_or_text
    try:
        with open(path_or_text, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        raise OripaError(f"cannot read {path_or_text!r}: {exc}") from exc


def read_cp(path_or_text, tol=1e-6, split_crossings=True):
    """Read an ORIPA/Oriedita `.cp` file into a flat `Frame`.

    Returns `(frame, stats)`.  The format is one crease per line,
    `type x0 y0 x1 y1`, with blank lines and anything unparseable
    skipped and counted -- some files carry a header or trailing notes,
    and refusing the whole pattern over a stray line would be silly.
    """
    text = _read_text(path_or_text, "\n")
    segs = []
    unknown = 0
    bad = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        nums = re.findall(_NUM, line)
        if len(nums) < 5:
            bad += 1
            continue
        try:
            code = int(float(nums[0]))
            x0, y0, x1, y1 = (float(v) for v in nums[1:5])
        except ValueError:
            bad += 1
            continue
        kind = _TYPE_ASSIGN.get(code)
        if kind is None:
            kind = FLAT
            unknown += 1
        segs.append((x0, y0, x1, y1, kind))

    if not segs:
        raise OripaError(
            "no crease lines found. A .cp file is one crease per line, "
            "'type x0 y0 x1 y1' -- this file has no line in that form")

    stats = {"curves": 0, "skipped": bad, "uncoloured": unknown,
             "crossings": 0}
    fr = frame_from_segments(segs, tol=tol, split_crossings=split_crossings,
                             stats=stats, title="ORIPA crease pattern")
    return fr, stats


def _floats_in(el):
    """Every number appearing in this element's subtree, in order."""
    out = []
    for node in el.iter():
        for txt in (node.text, node.tail):
            if txt:
                out.extend(float(v) for v in re.findall(_NUM, txt))
    return out


def read_opx(path_or_text, tol=1e-6, split_crossings=True):
    """Read an ORIPA `.opx` file into a flat `Frame`.

    `.opx` is Java `XMLEncoder` output, so its exact shape follows
    whatever ORIPA's classes looked like in the version that wrote it:
    coordinates appear either as four flat properties (`x0`, `y0`, `x1`,
    `y1`) or as two nested `Point2D` objects (`p0`, `p1`).  Rather than
    pin one layout, this finds every object whose class name contains
    `OriLine` and reads the named properties inside it -- which survives
    both, and degrades to a clear error rather than a wrong pattern.
    """
    text = _read_text(path_or_text, "<java")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise OripaError(f"not valid .opx XML: {exc}") from exc

    segs = []
    unknown = 0
    for obj in root.iter("object"):
        cls = obj.get("class") or ""
        if "oriline" not in cls.lower():
            continue
        props = {}
        for void in obj.findall("void"):
            name = void.get("property")
            if not name:
                continue
            vals = _floats_in(void)
            if vals:
                props[name.lower()] = vals
        # flat form: x0 y0 x1 y1 -- or nested: p0 (x, y), p1 (x, y)
        if all(k in props for k in ("x0", "y0", "x1", "y1")):
            x0, y0 = props["x0"][0], props["y0"][0]
            x1, y1 = props["x1"][0], props["y1"][0]
        elif "p0" in props and "p1" in props and \
                len(props["p0"]) >= 2 and len(props["p1"]) >= 2:
            x0, y0 = props["p0"][0], props["p0"][1]
            x1, y1 = props["p1"][0], props["p1"][1]
        else:
            continue
        code = int(props["type"][0]) if props.get("type") else 2
        kind = _TYPE_ASSIGN.get(code)
        if kind is None:
            kind = FLAT
            unknown += 1
        segs.append((x0, y0, x1, y1, kind))

    if not segs:
        raise OripaError(
            "no OriLine objects found. This importer reads ORIPA's .opx, "
            "which is Java XMLEncoder output describing OriLine objects; "
            "the file parsed as XML but contains none")

    stats = {"curves": 0, "skipped": 0, "uncoloured": unknown, "crossings": 0}
    fr = frame_from_segments(segs, tol=tol, split_crossings=split_crossings,
                             stats=stats, title="ORIPA crease pattern")
    return fr, stats


def _selftest():
    from .graph import build_faces

    # --- .cp: a square with both diagonals ---------------------------
    #
    # Type codes, not colours: 1 contour, 2 mountain, 3 valley.  The two
    # diagonals cross in the middle with nothing marking it, which is
    # exactly how these files are written and why the crossing split is
    # not optional.
    cp = "\n".join([
        "1 0.0 0.0 10.0 0.0",
        "1 10.0 0.0 10.0 10.0",
        "1 10.0 10.0 0.0 10.0",
        "1 0.0 10.0 0.0 0.0",
        "2 0.0 0.0 10.0 10.0",
        "3 10.0 0.0 0.0 10.0",
        "",
        "# a comment, and a junk line follows",
        "not a crease",
    ])
    fr, st = read_cp(cp)
    assert st["crossings"] == 1, st
    assert st["skipped"] == 1, f"the junk line should be counted: {st}"
    assert fr.n_verts == 5, fr.n_verts
    assert len(fr.edges) == 8, len(fr.edges)
    kinds = sorted(fr.assignment.tolist())
    assert kinds.count("B") == 4, kinds
    assert kinds.count("M") == 2 and kinds.count("V") == 2, kinds
    fr.faces = build_faces(fr.verts, fr.edges)
    assert fr.n_faces == 4, fr.n_faces

    # the type table is the format; check it end to end rather than
    # trusting the dict literal
    fr2, _s = read_cp("1 0 0 1 0\n2 0 0 0 1\n3 0 0 -1 0\n0 0 0 0 -1")
    got = {str(a) for a in fr2.assignment}
    assert got == {"B", "M", "V", "F"}, got

    # --- .opx, both layouts ORIPA has written ------------------------
    flat = """<java version="1.8" class="java.beans.XMLDecoder">
      <object class="oripa.value.OriLine">
        <void property="x0"><double>0.0</double></void>
        <void property="y0"><double>0.0</double></void>
        <void property="x1"><double>10.0</double></void>
        <void property="y1"><double>0.0</double></void>
        <void property="type"><int>2</int></void>
      </object>
      <object class="oripa.value.OriLine">
        <void property="x0"><double>0.0</double></void>
        <void property="y0"><double>0.0</double></void>
        <void property="x1"><double>0.0</double></void>
        <void property="y1"><double>10.0</double></void>
        <void property="type"><int>3</int></void>
      </object>
    </java>"""
    fr3, _s3 = read_opx(flat)
    assert len(fr3.edges) == 2, len(fr3.edges)
    assert sorted(fr3.assignment.tolist()) == ["M", "V"], fr3.assignment

    nested = """<java version="1.8" class="java.beans.XMLDecoder">
      <object class="oripa.value.OriLine">
        <void property="p0"><object class="java.awt.geom.Point2D$Double">
          <void property="x"><double>0.0</double></void>
          <void property="y"><double>0.0</double></void></object></void>
        <void property="p1"><object class="java.awt.geom.Point2D$Double">
          <void property="x"><double>5.0</double></void>
          <void property="y"><double>5.0</double></void></object></void>
        <void property="type"><int>1</int></void>
      </object>
    </java>"""
    fr4, _s4 = read_opx(nested)
    assert len(fr4.edges) == 1 and str(fr4.assignment[0]) == "B"

    # --- the failures say what to do ---------------------------------
    for fn, text, want in ((read_cp, "hello\nworld", "one crease per line"),
                           (read_opx, "<java></java>", "OriLine")):
        try:
            fn(text)
        except OripaError as exc:
            assert want in str(exc), str(exc)
        else:
            raise AssertionError(f"{fn.__name__} should have raised")

    print("RESULT: OK  crease.oripa_io")
