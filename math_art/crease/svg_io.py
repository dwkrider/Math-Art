# Reading a crease pattern out of an SVG.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python,
# numpy and the standard library only -- no `bpy`.
#
# WHY THIS EXISTS.  FOLD is the interchange format and the one this
# add-on gates on, but it is not what the reference material is actually
# distributed in.  Origami Simulator -- the most widely used origami
# simulator there is -- reads both FOLD and SVG, and ships its own
# pattern library as SVG with no FOLD equivalent: `assets/Origami/
# hypar.svg` is the authoritative pleated hypar and exists in no other
# form.  A FOLD-only importer cannot open the references for the very
# constructions this package implements.
#
# THE COLOUR CONVENTION IS THE FORMAT.  An SVG carries no notion of a
# mountain or a valley; the assignment lives in the stroke colour, and
# the convention is the one every origami tool uses and every crease
# diagram is drawn in -- mountain red, valley blue, boundary black.
# That is a real convention with real files behind it, not a guess, so
# it is read literally and anything unrecognised is imported as
# UNASSIGNED and counted rather than silently dropped.
#
# WHAT BIT, AND WHY `<path>` MATTERS MORE THAN `<line>`.  The obvious
# implementation reads `<line>` elements.  Measured on `hypar.svg`, that
# implementation opens the file and finds NOTHING useful: the entire
# crease pattern is three `<path>` elements of ~100 segments each, one
# per assignment, while the 94 `<line>` elements are the colour legend
# printed beside the drawing.  Illustrator writes paths; hand-written
# SVG writes lines; both occur.
#
# References:
#   E. D. Demaine, J. S. Ku, R. J. Lang, "A New File Standard to
#       Represent Folded Structures," 2016 -- the M/V/F/U/B assignments
#       this maps colours onto.
#   Amanda Ghassaei, "Origami Simulator" (MIT licence), whose SVG
#       colour convention and sample patterns this reads.
#   W3C, "Scalable Vector Graphics (SVG) 1.1," 2011 -- the path grammar
#       and the transform list.

import re
import xml.etree.ElementTree as ET

import numpy as np

from .fold_io import BOUNDARY, FLAT, MOUNTAIN, UNASSIGNED, VALLEY, Frame

#: Stroke colour to crease assignment.  Mountain red and valley blue are
#: the universal diagram convention; black is the sheet boundary.  Cyan
#: and yellow are Origami Simulator's marks for a crease that is drawn
#: but not folded, which is exactly FOLD's "flat".
_COLOUR_ASSIGN = {
    (255, 0, 0): MOUNTAIN,
    (0, 0, 255): VALLEY,
    (0, 0, 0): BOUNDARY,
    (0, 255, 255): FLAT,
    (255, 255, 0): FLAT,
}

#: Colours that mean "this is not part of the pattern".  Magenta is
#: annotation in Origami Simulator's own files -- in `hypar.svg` it is
#: the legend -- so importing it would add 94 stray edges beside the
#: drawing.
_COLOUR_SKIP = {(255, 0, 255)}

_NAMED = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "black": (0, 0, 0),
    "cyan": (0, 255, 255), "aqua": (0, 255, 255), "yellow": (255, 255, 0),
    "magenta": (255, 0, 255), "fuchsia": (255, 0, 255),
    "lime": (0, 255, 0), "green": (0, 128, 0), "white": (255, 255, 255),
}

_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(r"([MmLlHhVvZzCcSsQqTtAa])|(" + _NUM + r")")
_TRANSFORM = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*"
                        r"\(([^)]*)\)")


class SvgError(ValueError):
    """The file is not usable as a crease pattern."""


def _colour(value):
    """Parse an SVG paint value to an (r, g, b) tuple, or None."""
    if not value:
        return None
    v = value.strip().lower()
    if v in ("none", "transparent"):
        return None
    if v in _NAMED:
        return _NAMED[v]
    if v.startswith("#"):
        h = v[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return None
        return None
    m = re.match(r"rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", v)
    if m:
        return tuple(int(round(float(g))) for g in m.groups())
    return None


def _nearest(rgb):
    """Snap a colour to the nearest convention colour, or None.

    Real files are not exactly #FF0000: they are exported from
    Illustrator, run through optimisers, or drawn by hand at #EE1111.
    Snapping to the nearest listed colour within a generous radius reads
    those correctly, while a colour that is near nothing recognisable
    still falls through to UNASSIGNED rather than being forced.
    """
    if rgb is None:
        return None
    best, bestd = None, None
    for ref in list(_COLOUR_ASSIGN) + list(_COLOUR_SKIP):
        d = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if bestd is None or d < bestd:
            best, bestd = ref, d
    # 110 per channel: wide enough for a hand-picked "red", narrow
    # enough that grey does not become black and purple does not become
    # blue.
    return best if bestd is not None and bestd <= 3 * (110 ** 2) else None


def _parse_transform(text):
    """Compose an SVG transform list into a 3x3 matrix."""
    M = np.eye(3)
    if not text:
        return M
    for name, args in _TRANSFORM.findall(text):
        a = [float(x) for x in re.findall(_NUM, args)]
        T = np.eye(3)
        if name == "matrix" and len(a) >= 6:
            T = np.array([[a[0], a[2], a[4]],
                          [a[1], a[3], a[5]],
                          [0.0, 0.0, 1.0]])
        elif name == "translate" and a:
            T[0, 2] = a[0]
            T[1, 2] = a[1] if len(a) > 1 else 0.0
        elif name == "scale" and a:
            sx = a[0]
            sy = a[1] if len(a) > 1 else sx
            T[0, 0], T[1, 1] = sx, sy
        elif name == "rotate" and a:
            th = np.deg2rad(a[0])
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            if len(a) >= 3:                     # rotate about a point
                cx, cy = a[1], a[2]
                A = np.eye(3)
                A[0, 2], A[1, 2] = cx, cy
                B = np.eye(3)
                B[0, 2], B[1, 2] = -cx, -cy
                R = A @ R @ B
            T = R
        elif name == "skewX" and a:
            T[0, 1] = np.tan(np.deg2rad(a[0]))
        elif name == "skewY" and a:
            T[1, 0] = np.tan(np.deg2rad(a[0]))
        M = M @ T
    return M


def _path_segments(d):
    """Line segments of a path's `d`, ignoring curve commands.

    Curves are counted by the caller and skipped, not approximated.  A
    crease is a straight fold; a curved stroke in a crease-pattern SVG
    is either decoration or a curved-crease design, and silently
    polygonising the latter would turn one design crease into forty
    edges and a mesh nobody asked for.
    """
    segs = []
    curves = 0
    cur = start = None
    cmd = None
    buf = []

    def flush():
        nonlocal cur, start, buf, cmd, curves
        rel = cmd.islower()
        c = cmd.upper()
        if c == 'M':
            for j in range(0, len(buf) - 1, 2):
                p = (buf[j], buf[j + 1])
                if rel and cur is not None:
                    p = (cur[0] + p[0], cur[1] + p[1])
                if j == 0:
                    cur = start = p
                else:
                    segs.append((cur, p))
                    cur = p
        elif c == 'L':
            for j in range(0, len(buf) - 1, 2):
                p = (buf[j], buf[j + 1])
                if rel:
                    p = (cur[0] + p[0], cur[1] + p[1])
                segs.append((cur, p))
                cur = p
        elif c == 'H':
            for v in buf:
                p = (cur[0] + v, cur[1]) if rel else (v, cur[1])
                segs.append((cur, p))
                cur = p
        elif c == 'V':
            for v in buf:
                p = (cur[0], cur[1] + v) if rel else (cur[0], v)
                segs.append((cur, p))
                cur = p
        elif c == 'Z':
            if start is not None and cur is not None:
                segs.append((cur, start))
                cur = start
        else:
            curves += 1
        buf = []

    for a, b in _TOKEN.findall(d or ""):
        if a:
            if cmd is not None:
                flush()
            cmd = a
        else:
            buf.append(float(b))
    if cmd is not None:
        flush()
    return segs, curves


def _tag(el):
    return el.tag.split('}')[-1] if '}' in el.tag else el.tag


def _style_colour(el):
    """Stroke colour of an element, from `stroke` or from `style`."""
    c = _colour(el.get("stroke"))
    if c is not None:
        return c
    style = el.get("style") or ""
    m = re.search(r"stroke\s*:\s*([^;]+)", style)
    return _colour(m.group(1)) if m else None


def _collect(el, M, out, stats, inherited=None):
    """Walk the tree, accumulating transformed segments with colours."""
    M = M @ _parse_transform(el.get("transform"))
    colour = _style_colour(el)
    if colour is None:
        colour = inherited
    tag = _tag(el)
    segs = []

    def pts(name):
        return [float(x) for x in re.findall(_NUM, el.get(name) or "")]

    if tag == "line":
        try:
            segs = [((float(el.get("x1", 0)), float(el.get("y1", 0))),
                     (float(el.get("x2", 0)), float(el.get("y2", 0))))]
        except ValueError:
            segs = []
    elif tag in ("polyline", "polygon"):
        p = pts("points")
        q = [(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)]
        segs = [(q[i], q[i + 1]) for i in range(len(q) - 1)]
        if tag == "polygon" and len(q) > 2:
            segs.append((q[-1], q[0]))
    elif tag == "rect":
        try:
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            w, h = float(el.get("width", 0)), float(el.get("height", 0))
        except ValueError:
            w = h = 0.0
        if w > 0 and h > 0:
            c = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            segs = [(c[i], c[(i + 1) % 4]) for i in range(4)]
    elif tag == "path":
        segs, curves = _path_segments(el.get("d"))
        stats["curves"] += curves
    elif tag in ("text", "tspan", "image", "defs"):
        return                                  # not geometry; and do not
        #                                       # descend into <defs>

    if segs:
        rgb = _nearest(colour)
        if rgb in _COLOUR_SKIP:
            stats["skipped"] += len(segs)
        else:
            kind = _COLOUR_ASSIGN.get(rgb, UNASSIGNED)
            if rgb is None:
                stats["uncoloured"] += len(segs)
            for (x1, y1), (x2, y2) in segs:
                a = M @ np.array([x1, y1, 1.0])
                b = M @ np.array([x2, y2, 1.0])
                out.append((float(a[0]), float(a[1]),
                            float(b[0]), float(b[1]), kind))

    for child in el:
        _collect(child, M, out, stats, colour)


def _merge_vertices(segs, tol):
    """Weld endpoints within `tol`, returning (verts, edges, kinds)."""
    keys = {}
    verts = []

    def vid(x, y):
        k = (int(round(x / tol)), int(round(y / tol)))
        # look in the 3x3 neighbourhood, so two points either side of a
        # bucket boundary still weld
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                hit = keys.get((k[0] + dx, k[1] + dy))
                if hit is not None:
                    vx, vy = verts[hit]
                    if (vx - x) ** 2 + (vy - y) ** 2 <= tol * tol:
                        return hit
        keys[k] = len(verts)
        verts.append((x, y))
        return keys[k]

    edges, kinds, seen = [], [], {}
    for x1, y1, x2, y2, kind in segs:
        a, b = vid(x1, y1), vid(x2, y2)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            # A duplicate edge: a boundary claim wins, matching the
            # rule the pattern builders use.
            if kind == BOUNDARY:
                kinds[seen[key]] = BOUNDARY
            continue
        seen[key] = len(edges)
        edges.append(key)
        kinds.append(kind)
    return verts, edges, kinds


def _split_crossings(verts, edges, kinds, tol):
    """Split edges that cross away from a shared endpoint.

    Crease patterns drawn as long strokes routinely cross without a
    vertex -- the diagonals of a square hypar are one stroke each and
    meet in the middle with nothing marking it.  `graph.build_faces`
    rejects such a graph outright, so the crossing has to become a real
    vertex here or the import is unusable.
    """
    V = [list(p) for p in verts]
    E = [list(e) for e in edges]
    K = list(kinds)
    # Distinct split LOCATIONS, not cut parameters.  A proper crossing
    # cuts two edges and so records two parameters, at the same point --
    # reporting that as "2 crossings split" double-counts every one of
    # them.
    marks = set()

    # BATCHED, AND ITERATED TO A FIXED POINT.
    #
    # Batched because splitting one crossing at a time and restarting
    # the O(n^2) scan is hopeless at real sizes: a 483-edge pattern with
    # 99 crossings would rescan a quarter-million pairs a hundred times
    # over.  Each pass instead records every split point found, then
    # rebuilds all the affected edges at once.
    #
    # Iterated because a split makes two shorter edges that may cross
    # something the long one did not reach -- and a single sweep leaves
    # those behind, which `build_faces` rejects, so the import would die
    # at the last step with the whole file already read.
    for _sweep in range(16):
        cuts = {}                     # edge index -> list of parameters

        def cut(idx, w):
            if 1e-9 < w < 1 - 1e-9:
                cuts.setdefault(idx, []).append(w)

        P = np.array(V, dtype=float)
        for i in range(len(E)):
            a, b = E[i]
            p, r = P[a], P[b] - P[a]
            for j in range(i + 1, len(E)):
                c, d = E[j]
                q, s = P[c], P[d] - P[c]
                den = r[0] * s[1] - r[1] * s[0]
                if abs(den) > 1e-12 and len({a, b, c, d}) == 4:
                    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
                    u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / den
                    if 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9:
                        cut(i, t)
                        cut(j, u)
                        continue
                # T-JUNCTION: an endpoint of one edge lying in the
                # INTERIOR of the other.  Not a crossing in the usual
                # sense and easy to leave out, but still not a plane
                # graph -- and it is what any pattern produces wherever
                # a crease stops against another rather than passing
                # through it.
                for node, (idx, e0, e1) in ((c, (i, a, b)), (d, (i, a, b)),
                                            (a, (j, c, d)), (b, (j, c, d))):
                    if node in (e0, e1):
                        continue
                    A, B = P[e0], P[e1]
                    seg = B - A
                    L2 = float(seg @ seg)
                    if L2 <= tol * tol:
                        continue
                    w = float((P[node] - A) @ seg) / L2
                    if 1e-9 < w < 1 - 1e-9 and \
                            float(np.linalg.norm(A + w * seg - P[node])) <= tol:
                        cut(idx, w)
        if not cuts:
            break

        newE, newK = [], []
        for i, (a, b) in enumerate(E):
            ws = cuts.get(i)
            if not ws:
                newE.append([a, b])
                newK.append(K[i])
                continue
            A, B = P[a], P[b]
            chain = [a]
            for w in sorted(set(round(x, 12) for x in ws)):
                x = A + w * (B - A)
                V.append([float(x[0]), float(x[1])])
                chain.append(len(V) - 1)
                marks.add((round(float(x[0]), 9), round(float(x[1]), 9)))
            chain.append(b)
            for u, v in zip(chain, chain[1:]):
                newE.append([u, v])
                newK.append(K[i])
        E, K = newE, newK
    return V, E, K, len(marks)


def frame_from_segments(segs, tol=1e-6, split_crossings=True, stats=None,
                        title="crease pattern"):
    """Turn assignment-tagged line segments into a flat `Frame`.

    Shared by every line-based importer -- SVG, ORIPA `.cp`, Oriedita
    `.opx` -- because they differ only in how a segment gets its
    assignment.  Everything after that is the same problem: weld
    endpoints that were written as separate coordinates, split the
    crossings that a drawing program has no reason to mark, and hand
    back a plane graph.

    `segs` is a sequence of `(x1, y1, x2, y2, assignment)`.
    """
    if stats is None:
        stats = {}
    stats.setdefault("crossings", 0)
    if not segs:
        raise SvgError("no crease lines found in the file")

    # Tolerance is relative to the drawing, since these formats carry
    # no unit: a pattern 1000 units wide and one 1 unit wide need the
    # same weld in *relative* terms.
    xs = [v for s in segs for v in (s[0], s[2])]
    ys = [v for s in segs for v in (s[1], s[3])]
    extent = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    weld = max(tol, extent * 1e-6)

    verts, edges, kinds = _merge_vertices(segs, weld)
    if split_crossings:
        verts, edges, kinds, n = _split_crossings(verts, edges, kinds, weld)
        stats["crossings"] = n
        if n:
            verts, edges, kinds = _merge_vertices(
                [(verts[a][0], verts[a][1], verts[b][0], verts[b][1], k)
                 for (a, b), k in zip(edges, kinds)], weld)

    fr = Frame(
        verts=np.array(verts, dtype=float),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        assignment=np.array(kinds, dtype="<U1"),
        faces=None,
        face_orders=None,
        meta={"frame_title": title},
    )
    stats["vertices"] = int(fr.n_verts)
    stats["edges"] = int(len(fr.edges))
    return fr


def read_svg(path_or_text, tol=1e-6, split_crossings=True):
    """Read an SVG crease pattern into a flat `Frame`.

    Returns `(frame, stats)`.  `stats` reports what was dropped and why
    -- curves skipped, annotation colours skipped, strokes with no
    recognised colour -- because an importer that silently discards half
    a file is worse than one that refuses it.
    """
    text = path_or_text
    looks_like_markup = isinstance(text, str) and text.lstrip().startswith("<")
    if not looks_like_markup:
        try:
            with open(path_or_text, "r", encoding="utf-8",
                      errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            raise SvgError(f"cannot read {path_or_text!r}: {exc}") from exc
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise SvgError(f"not valid SVG/XML: {exc}") from exc

    stats = {"curves": 0, "skipped": 0, "uncoloured": 0, "crossings": 0}
    raw = []
    _collect(root, np.eye(3), raw, stats)
    if not raw:
        raise SvgError(
            "no crease strokes found. Creases are read from stroke "
            "colour -- mountain red, valley blue, boundary black -- so a "
            "drawing with only fills, text or curves imports as nothing")

    # SVG's y axis points DOWN, and this is the only importer that has
    # to care.  Flipping here means an imported pattern is not a mirror
    # image of the same pattern read from FOLD -- which would silently
    # swap every mountain and valley the moment it was folded.  Done on
    # the segments rather than the finished frame so the crossing split
    # runs on the same coordinates the caller will see.
    raw = [(x1, -y1, x2, -y2, k) for x1, y1, x2, y2, k in raw]
    fr = frame_from_segments(raw, tol=tol, split_crossings=split_crossings,
                             stats=stats, title="SVG crease pattern")
    return fr, stats


def stats_summary(stats):
    """One line a user can act on."""
    bits = [f"{stats['vertices']} vertices, {stats['edges']} creases"]
    if stats.get("crossings"):
        bits.append(f"{stats['crossings']} crossing(s) split")
    if stats.get("skipped"):
        bits.append(f"{stats['skipped']} annotation stroke(s) ignored")
    if stats.get("uncoloured"):
        bits.append(f"{stats['uncoloured']} stroke(s) had no recognised "
                    f"crease colour and came in unassigned")
    if stats.get("curves"):
        bits.append(f"{stats['curves']} curve command(s) skipped -- this "
                    f"importer reads straight creases only")
    return "; ".join(bits)


def _selftest():
    from .graph import build_faces

    # --- colours, including the ones real files actually contain -----
    assert _colour("#FF0000") == (255, 0, 0)
    assert _colour("#f00") == (255, 0, 0)
    assert _colour("rgb(0, 0, 255)") == (0, 0, 255)
    assert _colour("none") is None
    assert _nearest((238, 17, 17)) == (255, 0, 0), "a hand-picked red"
    assert _nearest((128, 128, 128)) is None, "grey is not black"

    # --- transforms --------------------------------------------------
    M = _parse_transform("translate(10,5) scale(2)")
    p = M @ np.array([1.0, 1.0, 1.0])
    assert np.allclose(p[:2], [12.0, 7.0]), p
    M = _parse_transform("rotate(90)")
    p = M @ np.array([1.0, 0.0, 1.0])
    assert np.allclose(p[:2], [0.0, 1.0], atol=1e-9), p

    # --- a path is read, and it is the case that matters -------------
    #
    # `hypar.svg` keeps its whole crease pattern in three <path>
    # elements and its legend in 94 <line> elements, so an importer that
    # reads lines and skips paths opens that file and finds only the
    # legend.  Both forms are exercised here for that reason.
    segs, curves = _path_segments("M0,0 H10 V10 H0 Z")
    assert len(segs) == 4 and curves == 0, (segs, curves)
    segs, curves = _path_segments("M0,0 C1,1 2,2 3,3")
    assert curves == 1, "a curve must be counted, not silently dropped"
    segs, _c = _path_segments("m0,0 l5,0 l0,5")
    assert len(segs) == 2 and np.allclose(segs[-1][1], (5.0, 5.0)), segs

    # --- a square with both diagonals, drawn as crossing strokes -----
    svg = """<svg xmlns="http://www.w3.org/2000/svg">
      <path stroke="#000000" d="M0,0 H10 V10 H0 Z"/>
      <path stroke="#FF0000" d="M0,0 L10,10"/>
      <path stroke="#0000FF" d="M10,0 L0,10"/>
      <line stroke="#FF00FF" x1="20" y1="0" x2="30" y2="0"/>
      <text x="0" y="0">mountain</text>
    </svg>"""
    fr, st = read_svg(svg)
    assert st["skipped"] == 1, f"magenta legend not ignored: {st}"
    # the two diagonals cross in the middle: that must become a vertex,
    # or build_faces refuses the graph outright
    assert st["crossings"] == 1, st
    assert fr.n_verts == 5, (fr.n_verts, fr.verts)
    assert len(fr.edges) == 8, len(fr.edges)
    kinds = sorted(fr.assignment.tolist())
    assert kinds.count("B") == 4 and kinds.count("M") == 2 \
        and kinds.count("V") == 2, kinds
    fr.faces = build_faces(fr.verts, fr.edges)
    assert fr.n_faces == 4, fr.n_faces

    # y is flipped: SVG counts downward, and a pattern imported upside
    # down would fold to the mirror image of the same file read as FOLD
    assert fr.verts[:, 1].max() <= 1e-9, fr.verts

    # --- transforms are applied to real geometry ---------------------
    fr2, _s = read_svg(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<g transform="translate(100,0)">'
        '<line stroke="red" x1="0" y1="0" x2="10" y2="0"/></g></svg>')
    assert fr2.verts[:, 0].min() >= 99.0, fr2.verts

    # --- an unrecognised colour is imported, not discarded -----------
    fr3, st3 = read_svg(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<line stroke="#888888" x1="0" y1="0" x2="1" y2="0"/></svg>')
    assert len(fr3.edges) == 1 and str(fr3.assignment[0]) == UNASSIGNED
    assert st3["uncoloured"] == 1, st3

    # --- a drawing with no strokes says so ---------------------------
    try:
        read_svg('<svg xmlns="http://www.w3.org/2000/svg">'
                 '<text x="0" y="0">hello</text></svg>')
    except SvgError as exc:
        assert "no crease strokes" in str(exc)
    else:
        raise AssertionError("an SVG with no creases should raise")

    print("RESULT: OK  crease.svg_io")
