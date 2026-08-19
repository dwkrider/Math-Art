# The GemCad .ASC faceting-design interchange format.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`, no numpy.
#
# GemCad's native `.GEM` format is proprietary; `.ASC` is the plain-text
# one it and Datavue II share, and it is what published faceting designs
# are distributed in.  Reading it is what gives the generator a library
# instead of a handful of hard-coded cuts.
#
# The grammar, from Strickland's manual:
#
#   GemCad 5.0                  magic first line
#   g <teeth> <rotation>        index gear
#   y <fold> <y|n>              radial symmetry, mirror-image flag
#   I <n>                       refractive index the design is cut for
#   H <text>                    heading, up to four lines
#   a <angle> <dist> <idx...>   ONE TIER of facets
#   G <text>                    cutting instruction for the tier above
#   F <text>                    footnote, up to four lines
#
# Within an `a` line the angle is negative for pavilion facets, the
# centre-to-facet distance is positive except on a culet (a 0 degree
# pavilion facet), `n` introduces a facet NAME attached to the index just
# before it, and `G` starts a cutting instruction that runs to the end of
# the line.
#
# Two details make a naive parser wrong on real files:
#
#   * `n` is positional.  In the manual's own Standard Round Brilliant the
#     girdle tier reads `93 n G 87 81 ...` -- the facet is NAMED "G", and
#     that G is not the cutting-instruction keyword.  Whatever token
#     follows `n` is the name, keyword or not.
#   * long index lists are BROKEN ACROSS LINES, so a line that begins with
#     a number continues the tier above rather than starting anything.
#
# Files copy-pasted out of PDFs also arrive with U+2212 MINUS SIGN instead
# of ASCII hyphen-minus, which silently turns every pavilion angle
# positive; `parse_asc` normalises it rather than raising, because the
# failure is otherwise invisible -- the design simply comes out inside
# out.
#
# References:
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises, 2002, pp. 20-26 -- the format definition and
#     the Standard Round Brilliant listing used as the round-trip fixture.
#   Robert H. Long & Norman W. Steele, Datavue II design database --
#     the other publisher of .ASC designs.

try:
    from .design import CutDesign, Tier
except ImportError:                     # flat import outside the package
    from design import CutDesign, Tier

MAGIC = "GemCad 5.0"

# U+2212 MINUS SIGN, plus the dash characters a word processor may
# substitute; all mean ASCII '-' in a number.
_DASHES = "−‐‑‒–—"


def _normalise(text):
    for ch in _DASHES:
        text = text.replace(ch, "-")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_asc(text):
    """Parse a GemCad/Datavue `.ASC` design into a `CutDesign`.

    Tolerant by design: the magic line's version is not enforced, unknown
    line keys are ignored, and index lists may be broken across lines.
    Raises ValueError only when the file carries no facet data at all.
    """
    name = ""
    gear, gear_offset = 96, 0.0
    fold, mirror, ri = 1, False, 1.54
    headings, footnotes, tiers = [], [], []

    for raw in _normalise(text).split("\n"):
        line = raw.strip()
        if not line:
            continue
        key, _, rest = line.partition(" ")
        rest = rest.strip()

        if line.startswith("GemCad"):
            continue
        if key == "g":
            f = rest.split()
            if f:
                gear = int(float(f[0]))
            if len(f) > 1:
                gear_offset = float(f[1])
        elif key == "y":
            f = rest.split()
            if f:
                fold = int(float(f[0]))
            mirror = len(f) > 1 and f[1].lower().startswith("y")
        elif key == "I":
            ri = float(rest.split()[0])
        elif key == "H":
            headings.append(rest)
        elif key == "F":
            footnotes.append(rest)
        elif key == "G":
            if tiers:                       # annotates the tier above
                tiers[-1] = tiers[-1]._replace(note=rest)
        elif key == "a":
            tiers.append(_parse_tier(rest))
        elif key == "n" and tiers:          # continued name
            f = rest.split()
            if f:
                tiers[-1] = tiers[-1]._replace(name=f[0])
        elif tiers and _is_number(key):     # continued index list
            more = _parse_indices(line)
            tiers[-1] = tiers[-1]._replace(
                indices=tiers[-1].indices + more[0],
                name=more[1] or tiers[-1].name,
                note=more[2] or tiers[-1].note)

    if not tiers:
        raise ValueError("no facet tiers ('a' lines) found; "
                         "is this a GemCad .ASC file?")
    if headings:
        name = headings[0]
    return CutDesign(name=name, gear=gear, gear_offset=gear_offset,
                     fold=fold, mirror=mirror, ri=ri, tiers=tuple(tiers),
                     headings=tuple(headings[:4]),
                     footnotes=tuple(footnotes[:4]))


def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _parse_indices(rest):
    """(indices, name, cutting instruction) from a tier's token stream."""
    toks = rest.split()
    idx, name, note = [], "", ""
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "n" and i + 1 < len(toks):
            name = toks[i + 1]              # positional: even if it is "G"
            i += 2
        elif t == "G":
            note = " ".join(toks[i + 1:])
            break
        else:
            idx.append(int(float(t)))
            i += 1
    return tuple(idx), name, note


def _parse_tier(rest):
    toks = rest.split()
    if len(toks) < 2:
        raise ValueError(f"malformed tier line: 'a {rest}'")
    angle = float(toks[0])
    distance = float(toks[1])
    idx, name, note = _parse_indices(" ".join(toks[2:]))
    return Tier(angle_deg=angle, distance=distance, indices=idx,
                name=name, note=note)


def write_asc(design, wrap=0):
    """Serialise a `CutDesign` back to `.ASC` text.

    Formatting matches the manual's own listing -- six decimals on the
    angle, eight on the distance, the facet name after the first index --
    so a published design read and rewritten comes back character for
    character.  `wrap` optionally breaks index lists at that many indices
    per line, which the DOS tools need for long tiers (they cap lines at
    79 characters).
    """
    out = [MAGIC,
           f"g {design.gear} {design.gear_offset:.1f}",
           f"y {design.fold} {'y' if design.mirror else 'n'}",
           f"I {design.ri:g}"]
    out += [f"H {h}" for h in design.headings[:4]]

    for t in design.tiers:
        head = f"a {t.angle_deg:.6f} {t.distance:.8f}"
        idx = list(t.indices)
        if not idx:
            out.append(head)
        else:
            first = f"{head} {idx[0]}"
            if t.name:
                first += f" n {t.name}"
            body = idx[1:]
            if wrap and len(body) > wrap:
                out.append(first)
                for s in range(0, len(body), wrap):
                    out.append(" ".join(str(i) for i in body[s:s + wrap]))
            else:
                out.append(first + ("".join(f" {i}" for i in body)))
        if t.note:
            out.append(f"G {t.note}")

    out += [f"F {f}" for f in design.footnotes[:4]]
    return "\n".join(out) + "\n"


# The manual's listing, verbatim, as the round-trip fixture.
SRB_ASC = """GemCad 5.0
g 96 0.0
y 8 y
I 1.54
H Standard Round Brilliant
H 9/5/02
a -90.000000 1.02653281 93 n G 87 81 75 69 63 57 51 45 39 33 27 21 15 9 3
G 16-sided outline
a -42.500000 0.61819401 93 n 1 87 81 75 69 63 57 51 45 39 33 27 21 15 9 3
a -41.500000 0.61701256 96 n 2 84 72 60 48 36 24 12
a 34.000000 0.68444470 3 n A 9 15 21 27 33 39 45 51 57 63 69 75 81 87 93
a 28.000000 0.60896430 96 n B 12 24 36 48 60 72 84
a 16.000000 0.50613241 6 n C 18 30 42 54 66 78 90
a 0.000000 0.36450932 96 n T
F GemCad for Windows Manual
"""


def _selftest():
    try:
        from . import design as _design
        from . import facets as _facets
    except ImportError:
        import design as _design
        import facets as _facets

    ok = True

    # --- the manual's listing parses to the documented design ------------
    D = parse_asc(SRB_ASC)
    good = (D.gear == 96 and D.fold == 8 and D.mirror
            and abs(D.ri - 1.54) < 1e-12 and len(D.tiers) == 7)
    ok &= good
    print(f"gems.asc: the manual's SRB parses -- 96 index, 8-fold mirror, "
          f"RI 1.54, 7 tiers {'OK' if good else 'misparsed'}")

    counts = tuple(len(t.indices) for t in D.tiers)
    good = counts == (16, 16, 8, 16, 8, 8, 1)
    ok &= good
    print(f"gems.asc: tier index counts {counts} "
          f"{'OK' if good else 'expected (16,16,8,16,8,8,1)'}")

    names = tuple(t.name for t in D.tiers)
    good = names == ("G", "1", "2", "A", "B", "C", "T")
    ok &= good
    print(f"gems.asc: facet names {names} -- the girdle really is named "
          f"'G' {'OK' if good else 'name parsing is positional-blind'}")

    good = D.tiers[0].note == "16-sided outline" and not D.tiers[1].note
    ok &= good
    print(f"gems.asc: the standalone G line annotates the tier above it "
          f"{'OK' if good else 'attached to the wrong tier'}")

    good = D.tiers[0].angle_deg == -90.0 and D.tiers[3].angle_deg == 34.0
    ok &= good
    print(f"gems.asc: pavilion angles stay negative, crown positive "
          f"{'OK' if good else 'sign lost'}")

    # --- it agrees with the hand-entered design in design.py -------------
    R = _design.SRB_GEMCAD
    good = all(abs(a.angle_deg - b.angle_deg) < 1e-12
               and abs(a.distance - b.distance) < 1e-12
               and tuple(sorted(i % 96 for i in a.indices))
               == tuple(sorted(i % 96 for i in b.indices))
               for a, b in zip(D.tiers, R.tiers))
    ok &= good
    print(f"gems.asc: the parsed design matches the hand-entered "
          f"SRB_GEMCAD tier for tier {'OK' if good else 'they disagree'}")

    # --- exact text round-trip -------------------------------------------
    back = write_asc(D)
    good = back == SRB_ASC
    ok &= good
    if not good:
        for i, (x, y) in enumerate(zip(back.split("\n"),
                                       SRB_ASC.split("\n"))):
            if x != y:
                print(f"    first difference on line {i + 1}:\n"
                      f"      wrote {x!r}\n      want  {y!r}")
                break
    print(f"gems.asc: rewriting the manual's listing reproduces it "
          f"character for character {'OK' if good else 'text drifted'}")

    good = parse_asc(write_asc(parse_asc(SRB_ASC))) == D
    ok &= good
    print(f"gems.asc: parse/write/parse is structurally stable "
          f"{'OK' if good else 'unstable'}")

    # --- geometric round-trip --------------------------------------------
    N1, d1, _ = _design.planes(D)
    N2, d2, _ = _design.planes(parse_asc(write_asc(D)))
    P1 = _facets.intersect_halfspaces(N1, d1)
    P2 = _facets.intersect_halfspaces(N2, d2)
    worst = max(abs(a - b)
                for va, vb in zip(sorted(map(tuple, P1.V.tolist())),
                                  sorted(map(tuple, P2.V.tolist())))
                for a, b in zip(va, vb))
    good = worst < 1e-9 and len(P1.faces) == len(P2.faces) == 73
    ok &= good
    print(f"gems.asc: the solid survives a round trip vertex for vertex "
          f"(worst {worst:.2e}) {'OK' if good else 'geometry moved'}")

    # --- the U+2212 trap ---------------------------------------------------
    pdf = SRB_ASC.replace("-", "−")
    Dp = parse_asc(pdf)
    good = Dp.tiers[0].angle_deg == -90.0 and Dp == D
    ok &= good
    print(f"gems.asc: a listing pasted from a PDF (U+2212 minus) parses "
          f"identically {'OK' if good else 'inverted the stone'}")

    # --- wrapped index lists ----------------------------------------------
    wrapped = write_asc(D, wrap=6)
    good = parse_asc(wrapped) == D and len(wrapped.split("\n")) > \
        len(SRB_ASC.split("\n"))
    ok &= good
    print(f"gems.asc: index lists broken across lines reassemble "
          f"{'OK' if good else 'continuation lost'}")

    # --- the culet convention survives a round trip ------------------------
    culet = D._replace(tiers=D.tiers + (Tier(0.0, -0.052, (96,), name="U"),))
    rt = parse_asc(write_asc(culet)).tiers[-1]
    good = rt.angle_deg == 0.0 and rt.distance < 0 and rt.name == "U"
    ok &= good
    print(f"gems.asc: a culet (angle 0, negative distance) round-trips "
          f"{'OK' if good else 'sign lost'}")

    # --- refusal on junk ---------------------------------------------------
    raised = False
    try:
        parse_asc("hello\nthis is not a design\n")
    except ValueError:
        raised = True
    ok &= raised
    print(f"gems.asc: a file with no tiers is refused, not silently empty "
          f"{'OK' if raised else 'accepted junk'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.asc self-test failed")
