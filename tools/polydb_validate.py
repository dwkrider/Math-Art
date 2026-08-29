# Validator for the polyhedron database in `data/polyhedra/`.
#
# Every record in the database asserts a `provenance.verified` block.  This
# script is what makes those assertions mean something: it recomputes each
# claim from the stored geometry and fails when the record and the geometry
# disagree.  Run it after any hand-edit or regeneration.
#
#     python tools/polydb_validate.py
#
# Checks, per record:
#   structure    - required keys present, index and record agree
#   euler        - V - E + F equals the stated Euler characteristic, and the
#                  stated counts match the stored arrays
#   manifold     - every edge is used exactly twice, with opposite orientation
#                  (so the winding is coherent and the surface is closed)
#   planar       - every face is flat to tolerance
#   centroid     - the vertex centroid sits at the stated centre
#   metrics      - circumradius / midradius / inradius / edge length recomputed
#                  from the geometry match the stored numbers, and each stored
#                  `exact` string evaluates to its own `value`
#   genus        - chi = 2 - 2g for orientable records
#   operators    - `construction.operator_id` names an operator that is
#                  actually registered somewhere in `math_art/`
#   symmetry     - the Schoenflies symbol is well formed and its axis
#                  order is consistent with the stated group order
#
# Star faces are handled: planarity and the edge pairing are computed on the
# true winding cycle, so a {5/2} pentagram validates as a single face.

import json
import math
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "polyhedra")
TOL = 1e-9


# -- the tiny exact-expression language -------------------------------------
#
# `exact` strings are a deliberately small subset -- integers, + - * / and
# unary minus, parentheses, sqrt(), acos(), and pi -- so they can be checked
# here without a CAS and read by anything else without a parser.

_ALLOWED = re.compile(r"^[0-9+\-*/(). a-z_]+$")
# vertices_exact may also reference constant names (C0, C1, ...)
_ALLOWED_C = re.compile(r"^[0-9+\-*/(). A-Za-z_]+$")
_ENV = {"sqrt": math.sqrt, "acos": math.acos, "asin": math.asin,
        "atan": math.atan, "cos": math.cos, "sin": math.sin, "tan": math.tan,
        "pi": math.pi, "phi": (1 + math.sqrt(5)) / 2}


def eval_exact(expr, consts=None):
    """Evaluate an `exact` string, optionally resolving names defined in the
    record's `constants` block. Returns None if it is not in the subset."""
    if not isinstance(expr, str):
        return None
    if not _ALLOWED.match(expr) and not _ALLOWED_C.match(expr):
        return None
    env = dict(_ENV)
    if consts:
        env.update(consts)
    try:
        return eval(expr, {"__builtins__": {}}, env)       # noqa: S307
    except Exception:
        return None


# -- geometry helpers -------------------------------------------------------

def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _det3(m):
    a, b, c = m
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _line_distance(a, b):
    """Perpendicular distance from the origin to the line through a and b."""
    d = _sub(b, a)
    L = _norm(d)
    if L < 1e-15:
        return _norm(a)
    ap = [-a[i] for i in range(3)]
    cr = _cross(ap, d)
    return _norm(cr) / L


def _newell(vs):
    """Newell normal -- correct for non-convex and star polygons, where a
    single cross product of three consecutive vertices can pick a reflex
    corner and point the wrong way."""
    n = [0.0, 0.0, 0.0]
    for p, q in zip(vs, vs[1:] + vs[:1]):
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return n


def _plane_normal(vs):
    """Unit normal, falling back to a least-squares plane when the Newell
    normal cancels. It cancels exactly on a CROSSED face -- an
    antiparallelogram's halves wind opposite ways -- and those faces are real:
    they are the dual faces of the star uniforms whose vertex figure is a
    crossed rectangle. Rejecting them as degenerate would be wrong."""
    n = _newell(vs)
    L = _norm(n)
    scale = max([_norm(_sub(v, vs[0])) for v in vs[1:]] or [1.0]) or 1.0
    if L >= 1e-12 * scale * scale:
        return [c / L for c in n]
    c = [sum(v[i] for v in vs) / len(vs) for i in range(3)]
    s = [[0.0] * 3 for _ in range(3)]
    for v in vs:
        d = [v[i] - c[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                s[i][j] += d[i] * d[j]
    try:
        import numpy as np
        _w, vec = np.linalg.eigh(np.array(s))
        nn = [float(x) for x in vec[:, 0]]
    except Exception:
        return None
    L2 = _norm(nn)
    return [x / L2 for x in nn] if L2 > 1e-14 else None


# -- the checks -------------------------------------------------------------

def check(rec, path, errors, warnings):
    def err(msg):
        errors.append("%s: %s" % (rec.get("slug", path), msg))

    def warn(msg):
        warnings.append("%s: %s" % (rec.get("slug", path), msg))

    for key in ("schema_version", "slug", "name", "families", "ids", "notation",
                "symmetry", "combinatorics", "metrics", "geometry", "provenance"):
        if key not in rec:
            err("missing top-level key %r" % key)
            return

    geo, comb, met = rec["geometry"], rec["combinatorics"], rec["metrics"]
    V, F = geo["vertices"], geo["faces"]

    # counts ---------------------------------------------------------------
    # A one-sided surface CANNOT be coherently wound -- that is what
    # non-orientable means -- so the coherence check applies only to records
    # that claim to be orientable.
    orientable = comb.get("orientable", True)
    edges = defaultdict(int)
    directed = set()
    incoherent = 0
    for f in F:
        if len(set(f)) != len(f):
            err("face %r repeats a vertex" % (f,))
        for a, b in zip(f, f[1:] + f[:1]):
            edges[frozenset((a, b))] += 1
            if (a, b) in directed:
                incoherent += 1
                if orientable and incoherent <= 4:
                    err("edge (%d,%d) traversed twice in the same direction "
                        "(incoherent winding)" % (a, b))
            directed.add((a, b))
    if orientable and incoherent > 4:
        err("%d edges in all have incoherent winding" % incoherent)
    if not orientable and incoherent == 0:
        warn("declared non-orientable but the winding is coherent")

    counts = comb["counts"]
    if len(V) != counts["vertices"]:
        err("counts.vertices=%d but %d vertices stored"
            % (counts["vertices"], len(V)))
    if len(F) != counts["faces"]:
        err("counts.faces=%d but %d faces stored" % (counts["faces"], len(F)))
    if len(edges) != counts["edges"]:
        err("counts.edges=%d but %d distinct edges in the face cycles"
            % (counts["edges"], len(edges)))

    bad = [tuple(e) for e, n in edges.items() if n != 2]
    if bad:
        err("%d edge(s) not shared by exactly two faces: %r"
            % (len(bad), bad[:5]))

    # Euler / genus --------------------------------------------------------
    chi = len(V) - len(edges) + len(F)
    if chi != comb["euler_characteristic"]:
        err("Euler characteristic is %d, record says %d"
            % (chi, comb["euler_characteristic"]))
    comp = rec.get("compound")
    if comp:
        # A compound is several closed surfaces, so chi is the SUM over
        # components and genus is not defined for the figure as a whole.
        ncomp = comp.get("component_count") or 0
        if comb.get("genus") is not None:
            err("compound records must not claim a genus: the figure is not a "
                "single surface")
        parts = comp.get("components") or []
        if parts:
            flat = sorted(i for c in parts for i in (c.get("faces") or []))
            if flat != list(range(len(F))):
                err("compound.components[].faces is not a partition of the "
                    "%d faces" % len(F))
            if len(parts) != ncomp:
                err("component_count=%d but %d components listed"
                    % (ncomp, len(parts)))
        if ncomp and chi % ncomp == 0:
            pass                      # a plausible sum; nothing stronger to assert
    elif comb.get("orientable") and comb.get("genus") is not None:
        if chi != 2 - 2 * comb["genus"]:
            err("genus %d is inconsistent with chi=%d"
                % (comb["genus"], chi))

    # planarity ------------------------------------------------------------
    worst = 0.0
    for i, f in enumerate(F):
        vs = [V[j] for j in f]
        n = _plane_normal(vs)
        if n is None:
            err("face %d has a degenerate normal" % i)
            continue
        d = _dot(n, vs[0])
        dev = max(abs(_dot(n, p) - d) for p in vs)
        worst = max(worst, dev)
    if worst > TOL:
        err("worst face out of plane by %.3e" % worst)

    # centroid -------------------------------------------------------------
    cen = [sum(v[i] for v in V) / len(V) for i in range(3)]
    want = geo.get("centroid", [0, 0, 0])
    if max(abs(cen[i] - want[i]) for i in range(3)) > 1e-9:
        err("vertex centroid %r != stated %r" % ([round(c, 12) for c in cen], want))

    # metrics --------------------------------------------------------------
    radii = [_norm(v) for v in V]
    if max(radii) - min(radii) < TOL:
        _cmp(met.get("circumradius"), max(radii), "circumradius", err)
    elif met.get("circumradius") is not None:
        warn("vertices are not cospherical but a circumradius is recorded")

    elens = [_norm(_sub(V[a], V[b])) for a, b in (tuple(e) for e in edges)]
    if max(elens) - min(elens) < TOL:
        if not met.get("edge_lengths_uniform", False):
            warn("all edges are equal but edge_lengths_uniform is not set")
        _cmp(met.get("edge_length"), max(elens), "edge_length", err)
    elif met.get("edge_lengths_uniform"):
        err("edge_lengths_uniform is true but edge lengths vary by %.3e"
            % (max(elens) - min(elens)))

    # Midradius is the radius of the sphere the EDGES are tangent to, i.e. the
    # perpendicular distance to the edge's LINE. For a uniform solid that
    # equals the distance to the midpoint, because an edge's ends are
    # equidistant from the centre; for a dual it does not, and using midpoints
    # would report a quantity that is neither the midradius nor even constant.
    mids = [_line_distance(V[a], V[b]) for a, b in (tuple(e) for e in edges)]
    if max(mids) - min(mids) < TOL:
        _cmp(met.get("midradius"), max(mids), "midradius", err)
    elif met.get("midradius") is not None:
        warn("edges are not tangent to a common sphere but a midradius is "
             "recorded (spread %.3e)" % (max(mids) - min(mids)))

    # inradius per face type
    per_type = defaultdict(list)
    for f in F:
        vs = [V[j] for j in f]
        n = _plane_normal(vs)
        if n is None:
            continue
        per_type[len(f)].append(abs(_dot(n, vs[0])))
    for entry in met.get("inradius") or []:
        m = re.match(r"\{(\d+)", str(entry.get("face", "")))
        if not m:
            continue
        got = per_type.get(int(m.group(1)))
        if got and max(got) - min(got) < TOL:
            _cmp(entry, got[0], "inradius %s" % entry["face"], err)

    # dihedral angles ------------------------------------------------------
    # Recompute every dihedral from the face normals and group by the pair of
    # face types, so a record cannot carry an angle its geometry disproves.
    normals, sides = [], []
    for f in F:
        vs = [V[j] for j in f]
        normals.append(_plane_normal(vs))
        sides.append(len(f))
    incident = defaultdict(list)
    for fi, f in enumerate(F):
        for a, b in zip(f, f[1:] + f[:1]):
            incident[frozenset((a, b))].append(fi)
    seen_pairs = defaultdict(set)
    for e, fis in incident.items():
        if len(fis) != 2 or normals[fis[0]] is None or normals[fis[1]] is None:
            continue
        c = max(-1.0, min(1.0, _dot(normals[fis[0]], normals[fis[1]])))
        deg = math.degrees(math.pi - math.acos(c))
        key = tuple(sorted((sides[fis[0]], sides[fis[1]])))
        seen_pairs[key].add(round(deg, 7))
    for entry in met.get("dihedral_angles") or []:
        if entry.get("degrees") is None:
            continue
        pair = tuple(sorted(int(re.match(r"\{(\d+)", s).group(1))
                            for s in entry["between"]))
        got = seen_pairs.get(pair)
        if not got:
            warn("dihedral %r: no such face-type pair in the geometry"
                 % (entry["between"],))
        elif not any(abs(g - entry["degrees"]) < 1e-6 for g in got):
            err("dihedral %r: stored %.9f deg, geometry gives %r"
                % (entry["between"], entry["degrees"], sorted(got)))
        # A dihedral's exact form is in radians. Compare COSINES, not angles:
        # acos is stationary at 0 and pi, so near a flat or fully folded
        # dihedral an angle difference of 1e-8 corresponds to a cosine
        # difference of 1e-16, and an angle-space test would reject a value
        # that is right to every digit the geometry actually carries.
        if entry.get("exact") and entry.get("radians") is not None:
            ev = eval_exact(entry["exact"])
            if ev is None:
                err("dihedral %r: exact %r is outside the allowed subset"
                    % (entry["between"], entry["exact"]))
            elif abs(math.cos(ev) - math.cos(entry["radians"])) > 1e-9:
                err("dihedral %r: exact %r = %.12f rad but radians is %.12f"
                    % (entry["between"], entry["exact"], ev, entry["radians"]))
        # A dihedral of 180 degrees means two adjacent faces are COPLANAR --
        # worth surfacing, since it affects what counts as a face.
        if entry.get("degrees") is not None and abs(entry["degrees"] - 180.0) < 1e-6:
            warn("dihedral %r is 180 degrees: these adjacent faces are coplanar"
                 % (entry["between"],))

    # exact vertices -------------------------------------------------------
    # Every exact coordinate must reproduce its own float. This is the check
    # that keeps the symbolic and numeric tables from ever drifting apart.
    consts = {}
    for name, m in (geo.get("constants") or {}).items():
        if isinstance(m, dict) and m.get("value") is not None:
            consts[name] = m["value"]
        if isinstance(m, dict) and m.get("exact"):
            got = eval_exact(m["exact"], consts)
            if got is None:
                err("constant %s: exact %r is outside the allowed subset"
                    % (name, m["exact"]))
            elif m.get("value") is not None and abs(got - m["value"]) > 1e-9:
                err("constant %s: exact %r = %.15g but value is %.15g"
                    % (name, m["exact"], got, m["value"]))

    vex = geo.get("vertices_exact")
    if vex is not None:
        if len(vex) != len(V):
            err("vertices_exact has %d entries but %d vertices"
                % (len(vex), len(V)))
        else:
            worst_v, badexpr = 0.0, 0
            for i, (row, want) in enumerate(zip(vex, V)):
                for k, s in enumerate(row):
                    got = eval_exact(s, consts)
                    if got is None:
                        badexpr += 1
                        if badexpr <= 3:
                            err("vertices_exact[%d][%d]: %r is not evaluable"
                                % (i, k, s))
                        continue
                    worst_v = max(worst_v, abs(got - want[k]))
            if worst_v > 1e-9:
                err("vertices_exact disagrees with vertices by up to %.3e"
                    % worst_v)

    # grouping blocks ------------------------------------------------------
    for blockname, nelem in (("face_groups", len(F)),
                             ("vertex_groups", len(V)),
                             ("edge_groups", len(edges))):
        block = rec.get(blockname)
        if not block:
            continue
        orb = block.get("symmetry_orbit")
        if orb is not None:
            flat = [i for g in orb for i in g]
            if sorted(flat) != list(range(nelem)):
                err("%s.symmetry_orbit is not a partition of the %d elements"
                    % (blockname, nelem))
        if blockname == "face_groups":
            pt = block.get("polygon_type") or {}
            flat = sorted(i for g in pt.values() for i in g)
            if pt and flat != list(range(len(F))):
                err("face_groups.polygon_type is not a partition of the faces")
            for sides, idxs in pt.items():
                for i in idxs:
                    if len(F[i]) != int(sides):
                        err("face_groups.polygon_type[%s] contains face %d "
                            "which has %d sides" % (sides, i, len(F[i])))

    # symmetry orbit counts must match the stated orbit numbers
    sym = rec.get("symmetry") or {}
    stated = sym.get("orbits") or {}
    for key, blockname in (("faces", "face_groups"), ("vertices", "vertex_groups"),
                           ("edges", "edge_groups")):
        block = rec.get(blockname) or {}
        orb = block.get("symmetry_orbit")
        if orb is not None and stated.get(key) is not None:
            if len(orb) != stated[key]:
                err("symmetry.orbits.%s = %d but %s lists %d orbits"
                    % (key, stated[key], blockname, len(orb)))

    # space filling ---------------------------------------------------------
    # The lattice claim is checkable: a cell of the translation lattice must
    # have exactly the volume the solid occupies in the tiling. This is what
    # keeps a curated field from being an unfalsifiable assertion.
    sf = rec.get("space_filling")
    if sf:
        basis = (sf.get("lattice") or {}).get("basis")
        vol = (met.get("volume") or {}).get("value")
        if basis and vol:
            n = sf.get("cells_per_lattice_point") or 1
            det = abs(_det3(basis))
            want = vol * n
            if abs(det - want) > 1e-6 * max(1.0, want):
                err("space_filling: |det(lattice basis)| = %.9g but volume x "
                    "cells = %.9g -- the lattice does not tile with this cell"
                    % (det, want))
        if sf.get("parallelohedron") and not sf.get("fills_space_alone"):
            err("space_filling: parallelohedron implies fills_space_alone")
        if sf.get("fills_space_alone") is False and not sf.get("combination_fillings"):
            warn("space_filling records that it does not tile alone but lists "
                 "no combination_fillings")

    # every `exact` string must agree with its own `value`
    _walk_exact(met, rec.get("slug", path), errors)


def _cmp(entry, computed, label, err):
    if not isinstance(entry, dict) or entry.get("value") is None:
        return
    if abs(entry["value"] - computed) > 1e-9:
        err("%s: stored %.15g, geometry gives %.15g"
            % (label, entry["value"], computed))


def _walk_exact(node, slug, errors):
    if isinstance(node, dict):
        if (node.get("exact") is not None and "value" in node
                and node["value"] is not None):
            got = eval_exact(node["exact"])
            if got is None:
                errors.append("%s: exact %r is outside the allowed subset"
                              % (slug, node["exact"]))
            elif abs(got - node["value"]) > 1e-9:
                errors.append("%s: exact %r = %.15g but value is %.15g"
                              % (slug, node["exact"], got, node["value"]))
        for v in node.values():
            _walk_exact(v, slug, errors)
    elif isinstance(node, list):
        for v in node:
            _walk_exact(v, slug, errors)


def check_duals(records, errors):
    """A dual pointer must exist and be mutual."""
    by_slug = {r["slug"]: r for r in records}
    for r in records:
        d = r["combinatorics"].get("dual")
        if d is None or d not in by_slug:
            continue                       # not yet in the database
        back = by_slug[d]["combinatorics"].get("dual")
        if back != r["slug"]:
            errors.append("%s: dual %r does not point back (says %r)"
                          % (r["slug"], d, back))
        rc, dc = r["combinatorics"]["counts"], by_slug[d]["combinatorics"]["counts"]
        # Reciprocity is COMBINATORIAL. Where a solid has coplanar faces, the
        # dual's vertices coincide in pairs, so its stored table holds fewer
        # distinct POINTS than the parent has faces; adding the merged count
        # back recovers the combinatorial figure. McCooey's tables show the
        # same coincidence, so this is a property of the solids, not of our
        # construction.
        rv = rc["vertices"] + (r["combinatorics"].get("coincident_vertices") or 0)
        dv = dc["vertices"] + (by_slug[d]["combinatorics"].get("coincident_vertices") or 0)
        if (rv, rc["edges"], rc["faces"]) != (dc["faces"], dc["edges"], dv):
            errors.append("%s: dual %r has non-reciprocal V/E/F counts "
                          "(%d/%d/%d vs %d/%d/%d, coincident-adjusted)"
                          % (r["slug"], d, rv, rc["edges"], rc["faces"],
                             dv, dc["edges"], dc["faces"]))


_SCHOENFLIES = re.compile(
    r"^(?:C1|Ci|Cs|[TOI]|Td|Th|Oh|Ih|[CDS](?:\d{1,3})[hvd]?)$")


def check_symmetry_symbols(records, errors):
    """Point-group symbols must be well formed, and agree with `order`.

    Nothing else looks at these strings, so a classifier that emits a
    malformed one is invisible: the record still validates geometrically
    and the bad symbol simply propagates.  That is how 30 records came to
    claim groups like "C298156827v" -- a near-identity matrix surviving
    the classifier's identity test gave n = round(2*pi/ang) in the
    hundreds of millions, which then flowed into the orbifold, Coxeter and
    Hermann-Mauguin notations too.

    The axis order is bounded here because `order` is computed
    independently (it is just the size of the group) and so cannot come
    from the same mistake: for a cyclic group |G| = n, and for the
    dihedral and improper families |G| = 2n, so n can never exceed the
    stated order.
    """
    for r in records:
        y = r.get("symmetry") or {}
        s = y.get("schoenflies")
        order = y.get("order")
        if not s:
            continue
        if not _SCHOENFLIES.match(s):
            errors.append("%s: malformed Schoenflies symbol %r"
                          % (r["slug"], s))
            continue
        digits = "".join(c for c in s if c.isdigit())
        if digits and order and int(digits) > order:
            errors.append("%s: Schoenflies %r has axis order %s, above the "
                          "group order %s" % (r["slug"], s, digits, order))
        # The derived notations concatenate axis orders without separators --
        # D10h is orbifold "*2210", meaning *2.2.10 -- so their digit runs
        # cannot be read as a single number and cannot be bounded by `order`.
        # A run of six or more digits is still unambiguous corruption: the
        # widest legitimate case in this database is a 32-gonal prism at
        # "*2232", and even a 999-gon would reach only five.
        for name in ("orbifold", "coxeter", "hermann_mauguin"):
            v = y.get(name)
            if v and re.search(r"\d{6,}", str(v)):
                errors.append("%s: %s %r carries an implausible axis order"
                              % (r["slug"], name, v))


def check_operators(records, errors, warnings):
    """`construction.operator_id` must name an operator that exists.

    The field is the record's only pointer back to the add-on: it is what
    tells a reader -- or the companion website -- which generator builds
    this solid.  Nothing else validates it, so a rename in `math_art/`
    leaves the whole database quietly pointing at an operator that is no
    longer registered.  That is exactly what happened to the compound and
    geodesic records, which named `mesh.compound_add` and
    `mesh.geodesic_sphere_add` long after the operators had become
    `mesh.polyhedron_compound_add` and `mesh.geodesic_add`.

    Scanning for `bl_idname` textually keeps this runnable without Blender,
    which is the whole point of the validator.
    """
    pkg = os.path.join(os.path.dirname(ROOT), "..", "math_art")
    pkg = os.path.normpath(pkg)
    if not os.path.isdir(pkg):
        warnings.append("math_art/ not found -- operator ids not checked")
        return
    known = set()
    for dirpath, _dirs, files in os.walk(pkg):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8",
                      errors="replace") as fh:
                known.update(re.findall(r'bl_idname\s*=\s*"([\w.]+)"',
                                        fh.read()))
    for r in records:
        op = (r.get("construction") or {}).get("operator_id")
        if op and op not in known:
            errors.append("%s: construction.operator_id %r is not a "
                          "registered operator" % (r["slug"], op))


def main():
    index_path = os.path.join(ROOT, "index.json")
    if not os.path.exists(index_path):
        print("no index at %s" % index_path)
        return 1
    with open(index_path) as fh:
        index = json.load(fh)

    errors, warnings, records = [], [], []

    # JSON Schema, when the library is available. The schema checks shape;
    # everything below checks that the shape tells the truth.
    rec_schema = None
    try:
        import jsonschema
        with open(os.path.join(ROOT, "schema", "index.schema.json")) as fh:
            jsonschema.validate(index, json.load(fh))
        with open(os.path.join(ROOT, "schema", "polyhedron.schema.json")) as fh:
            rec_schema = json.load(fh)
    except ImportError:
        warnings.append("jsonschema not installed -- shape not checked, "
                        "only geometry (pip install jsonschema)")
    except Exception as exc:                                  # noqa: BLE001
        errors.append("index.json fails its schema: %s" % exc)

    for entry in index["entries"]:
        p = os.path.join(ROOT, entry["path"])
        if not os.path.exists(p):
            errors.append("index points at missing file %s" % entry["path"])
            continue
        with open(p) as fh:
            rec = json.load(fh)
        records.append(rec)
        if rec_schema is not None:
            import jsonschema
            try:
                jsonschema.validate(rec, rec_schema)
            except jsonschema.ValidationError as exc:
                errors.append("%s fails its schema at %s: %s"
                              % (entry["path"],
                                 "/".join(str(p) for p in exc.absolute_path),
                                 exc.message))
        if rec["slug"] != entry["slug"]:
            errors.append("%s: index slug %r != record slug %r"
                          % (entry["path"], entry["slug"], rec["slug"]))
        if rec["combinatorics"]["counts"] != entry["counts"]:
            errors.append("%s: index counts disagree with the record"
                          % rec["slug"])
        check(rec, entry["path"], errors, warnings)

    if index.get("count") != len(index["entries"]):
        errors.append("index count=%r but %d entries listed"
                      % (index.get("count"), len(index["entries"])))

    # Slugs are the database's primary key: a duplicate means two files claim
    # the same solid, and every dual pointer to it is ambiguous.
    seen_slug = {}
    for entry in index["entries"]:
        if entry["slug"] in seen_slug:
            errors.append("duplicate slug %r: %s and %s"
                          % (entry["slug"], seen_slug[entry["slug"]],
                             entry["path"]))
        seen_slug[entry["slug"]] = entry["path"]
    check_duals(records, errors)
    check_operators(records, errors, warnings)
    check_symmetry_symbols(records, errors)

    for w in warnings:
        print("WARN  %s" % w)
    for e in errors:
        print("FAIL  %s" % e)
    print("\n%d record(s) checked, %d error(s), %d warning(s)"
          % (len(records), len(errors), len(warnings)))
    print("RESULT: %s" % ("OK" if not errors else "FAILED"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
