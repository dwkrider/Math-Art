# Per-solid metadata for the uniform polyhedra, from Har'El's tables.
#
#   Z. Har'El, "Uniform Solution for Uniform Polyhedra", Geometriae Dedicata 47
#   (1993), 57-110, Appendix II, Tables 4-8.
#
# Those tables are the closest published thing to a record layout for this
# database: per solid they give the Wythoff symbol, the vertex configuration,
# the Euler characteristic, the density, the dual's name, and cross-references
# to Coxeter-Longuet-Higgins-Miller (1954) and to Wenninger's model numbers.
#
# The paper is read from this repository's own converted copy under
# research/papers/. Rows are joined to `uniform_polyhedra_generator.UNIFORMS`
# by Wythoff symbol, with the tokens on each side of the bar sorted, because
# the two sources order them differently.
#
# Two parsing traps, both handled below:
#   * a literal '|' inside $...$ is the Wythoff bar, NOT a column separator;
#   * chi is starred for non-orientable solids, and density is blank where it
#     is undefined (the hemipolyhedra, whose faces pass through the centre).

import os
import re

_MD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "research", "papers", "polyhedra-and-regular-maps", "harel_uniform",
    "harel_uniform.md")


def _split_cells(line):
    cells, buf, math_mode = [], [], False
    for ch in line.strip():
        if ch == "$":
            math_mode = not math_mode
            buf.append(ch)
        elif ch == "|" and not math_mode:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


def _frac(s):
    return re.sub(r"\\[tdc]?frac\{(\d+)\}\{(\d+)\}", r"\1/\2", s)


def _wythoff(s):
    s = _frac(s.replace("$", ""))
    s = s.replace(r"\vert", "|").replace(r"\right.", "").replace(r"\left", "")
    s = s.replace(r"\ ", " ").replace("\\", " ").replace("|", " | ")
    return re.sub(r"\s+", " ", s).strip()


def _vcfg(s):
    s = _frac(s.replace("$", ""))
    s = s.replace(r"\left", "").replace(r"\right", "").replace("\\", "")
    return re.sub(r"\s+", "", s).strip()


def key(w):
    """Order-insensitive Wythoff join key."""
    toks = re.sub(r"\s+", " ", w.replace("|", " | ")).strip().split()
    if "|" in toks:
        i = toks.index("|")
        return "%s|%s" % (",".join(sorted(toks[:i])), ",".join(sorted(toks[i + 1:])))
    return ",".join(sorted(toks))


def rows(path=None):
    """All parsed table rows."""
    path = path or _MD
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = _split_cells(line)
        if cells and cells[0] == "":
            cells = cells[1:-1] if len(cells) > 1 else cells
        if len(cells) < 6 or not re.match(r"^\d+$", cells[1] or ""):
            continue
        namedual = cells[4].replace("*", "")
        if "/" in namedual:
            name, dual = (x.strip() for x in namedual.split("/", 1))
        else:
            name, dual = namedual.strip(), None
        chi_raw = cells[5].replace("$", "").replace("{", "").replace("}", "")
        nonorient = "*" in chi_raw
        try:
            chi = int(chi_raw.replace("^", "").replace("*", "").replace("\\", "").strip())
        except ValueError:
            chi = None
        dens_s = (cells[6] if len(cells) > 6 else "").replace("$", "").strip()
        dens = int(dens_s) if re.match(r"^\d+$", dens_s) else None
        clm = wen = None
        m = re.match(r"^(\d+)\s*,\s*(\d+)$", (cells[7] if len(cells) > 7 else "").strip())
        if m:
            clm, wen = int(m.group(1)), int(m.group(2))
        out.append({"wythoff": _wythoff(cells[2]), "vertex_config": _vcfg(cells[3]),
                    "name": name, "dual": dual, "chi": chi,
                    "orientable": not nonorient, "density": dens,
                    "coxeter_clm": clm, "wenninger": wen,
                    "schwarz": cells[0].replace("$", "")})
    return out


def by_wythoff(path=None):
    out = {}
    for r in rows(path):
        out.setdefault(key(r["wythoff"]), r)
    return out


# The one solid Har'El treats outside the tables: the great
# dirhombicosidodecahedron, the only non-Wythoffian uniform polyhedron, whose
# four-token symbol has no table row. Its values are stated in the paper's
# text (chi = -56) and confirmed by our own V - E + F = 60 - 240 + 124.
U75 = {"wythoff": "| 3/2 5/3 3 5/2", "vertex_config": "(4.5/2.4.3.4.5/3.4.3/2)",
       "name": "great dirhombicosidodecahedron",
       "dual": "great dirhombicosidodecacron", "chi": -56, "orientable": True,
       "density": None, "coxeter_clm": 92, "wenninger": 119, "schwarz": None}
