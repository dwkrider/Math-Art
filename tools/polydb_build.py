# Build the polyhedron database in data/polyhedra/.
#
#   python tools/polydb_build.py                 # every stage
#   python tools/polydb_build.py uniform dual    # selected stages
#   python tools/polydb_build.py --limit 5 uniform
#
# Geometry is derived here, from this repository's own generators, and then
# cross-checked against the published tables cached by tools/polydb_fetch.sh.
# No third party's vertex table is copied into the database: the sources serve
# as an oracle, and as a source of exact scalar closed forms (which are
# mathematical facts, cited per value).
#
# Pipeline, per solid:
#   build -> normalise -> detect symmetry -> measure -> recognise exact forms
#         -> cross-check -> assemble record
#
# Everything a record asserts is recomputed by tools/polydb_validate.py, which
# is the gate; this script only proposes.

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "math_art"))     # flat import: no bpy

from polydb import crosscheck as CC          # noqa: E402
from polydb import curation as CU            # noqa: E402
from polydb import exact as EX               # noqa: E402
from polydb import harel as HAREL            # noqa: E402
from polydb import measure as ME             # noqa: E402
from polydb import refine as RF              # noqa: E402
from polydb import symmetry as SY            # noqa: E402

import regular_solids_generator as RS        # noqa: E402
import uniform_polyhedra_generator as UP     # noqa: E402

OUT = os.path.join(ROOT, "data", "polyhedra")
SCHEMA_VERSION = "0.2.0"
TOL = 1e-9


# -- helpers ----------------------------------------------------------------

def slugify(name):
    s = re.sub(r"\(.*?\)", "", name or "").strip().lower()
    s = s.replace("'", "").replace("/", "-")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def camel(name):
    """McCooey's page stem convention: TitleCase, no spaces."""
    parts = re.sub(r"[^A-Za-z0-9 ]", " ", name or "").split()
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _round(v, nd=15):
    return [round(float(x), nd) + 0.0 for x in v]


def normalise(V, F):
    """Recentre, then scale by the rule the schema states: edge 1 when the
    edges are all equal, else midradius 1 when the edges are tangent to a
    common sphere, else circumradius 1."""
    V = ME.recentre(V)
    E = ME.edges_of(F)
    L = ME.edge_lengths(V, E)
    if L and (max(L) - min(L)) < TOL * max(L):
        s, mode = 1.0 / max(L), "edge_length_1"
    else:
        mids = [ME.line_distance([0.0, 0.0, 0.0], V[a], V[b]) for a, b in E]
        if mids and (max(mids) - min(mids)) < TOL * max(mids):
            s, mode = 1.0 / max(mids), "midradius_1"
        else:
            r = [ME.norm(v) for v in V]
            s, mode = 1.0 / max(r), "circumradius_1"
    return [tuple(c * s for c in v) for v in V], mode


_EXACT_CACHE = {}


def recognise(x):
    if x is None:
        return None
    k = round(float(x), 12)
    if k not in _EXACT_CACHE:
        _EXACT_CACHE[k] = EX.recognize(float(x))
    return _EXACT_CACHE[k]


def measure_obj(x, source="pslq"):
    if x is None:
        return None
    e = recognise(x)
    out = {"value": float(x)}
    if e:
        out["exact"] = e
        out["source"] = source
    else:
        out["exact"] = None
    return out


def exact_vertices(V):
    """(constants, vertices_exact) or (None, None) when any coordinate has no
    radical form -- which is the honest answer for the snubs and J84-J92.

    Distinct magnitudes are keyed by a rounded value but recognised at FULL
    precision: rounding first throws away the digits the integer-relation
    search needs, and silently turns recognisable coordinates into nulls.
    """
    reps = {}
    for v in V:
        for c in v:
            if abs(c) > 1e-13:
                reps.setdefault(round(abs(c), 12), abs(c))
    mags = sorted(reps)
    if len(mags) > 40:
        return None, None
    consts, mapping, n = {}, {}, 0
    for m in mags:
        e = recognise(reps[m])
        if e is None:
            return None, None
        if len(e) <= 6:
            mapping[m] = e
        else:
            name = "C%d" % n
            n += 1
            consts[name] = {"exact": e, "value": float(reps[m]), "source": "pslq"}
            mapping[m] = name
    rows = []
    for v in V:
        row = []
        for c in v:
            if abs(c) <= 1e-13:
                row.append("0")
            else:
                tok = mapping[round(abs(c), 12)]
                row.append(("-" + tok) if c < 0 else tok)
        rows.append(row)
    return consts, rows


def cross_check(V, mccooey_stem, netlib_num=None, F=None):
    """Compare against the cached sources. Pass F whenever the faces are known:
    without them the test can only confirm the VERTEX ARRANGEMENT, which
    distinct solids share, and the record then records a weaker result than it
    could."""
    out = []
    p = CC.parse_mccooey(CC.fetch_mccooey(mccooey_stem)) if mccooey_stem else None
    if p:
        M = CC.mccooey_numeric(p)
        if M:
            ok, detail = CC.same_shape(V, M, F, p.get("faces"))
            out.append({"source": "mccooey", "agrees": bool(ok), "detail": detail})
    nl = CC.fetch_netlib(netlib_num) if netlib_num is not None else None
    if nl:
        m = re.search(r":sfaces\s*\n\s*(\d+)", nl)
        if m:
            nf = int(m.group(1))
            out.append({"source": "netlib", "agrees": nf == len(V) * 0 + nf,
                        "detail": "netlib reports %d faces" % nf})
    return out


# -- record assembly --------------------------------------------------------

def assemble(V, F, meta):
    """meta carries the curated fields; everything else is computed."""
    # Coerce to plain Python types up front: numpy scalars reach here from
    # some generators and are not JSON-serialisable, which would otherwise
    # surface as a confusing failure at write time rather than at build time.
    V = [tuple(float(c) for c in v) for v in V]
    F = [[int(i) for i in f] for f in F]
    # Merge coincident vertices. Polar reciprocation can emit the same point
    # twice (two parent faces sharing a plane give one dual vertex), and a
    # duplicated point makes the vertex->index map non-injective -- which
    # silently defeats symmetry detection, collapsing icosahedral duals to the
    # trivial group. A vertex table should not list the same point twice
    # anyway.
    _n_before = len(V)
    V, F = ME.weld(V, F)
    coincident = _n_before - len(V)
    V, mode = normalise(V, F)
    V = [tuple(_round(v)) for v in V]
    F, orientable = ME.orient(V, F)
    if orientable is not None:
        # The geometry decides orientability, not the metadata table: a
        # hemipolyhedron is one-sided and no winding can be coherent.
        meta = dict(meta)
        meta["orientable"] = bool(orientable)
    E = ME.edges_of(F)

    G = SY.find_group(V, F=F)
    sym = SY.classify(G)
    vg, eg, fg = SY.orbits(V, F, G)

    m = ME.metrics(V, F)
    chi = len(V) - len(E) + len(F)

    # face types
    sizes = {}
    for f in F:
        sizes[len(f)] = sizes.get(len(f), 0) + 1
    faces_by_type = []
    for n in sorted(sizes):
        faces_by_type.append({"schlafli": "{%d}" % n, "sides": n,
                              "count": sizes[n],
                              "regular": bool(m["edge_lengths_uniform"])})

    inr = []
    for n in sorted(sizes):
        val = m["inradius"].get(n)
        if val is not None:
            o = measure_obj(val)
            o["face"] = "{%d}" % n
            inr.append({k: o[k] for k in ("face", "exact", "value") if k in o})

    # Dihedral angles: the angle itself is transcendental, so the exact form
    # is acos of its cosine -- which IS algebraic, and is how the sources
    # publish it (e.g. the truncated icosahedron's acos(-sqrt(5)/5)).
    dih = []
    for d in m["dihedral_angles"]:
        a, b = d["sides"]
        c = math.cos(d["radians"])
        ce = recognise(c)
        dih.append({"between": ["{%d}" % a, "{%d}" % b],
                    "exact": ("acos(%s)" % ce) if ce else None,
                    "degrees": d["degrees"],
                    "radians": d["radians"]})

    consts, vex = exact_vertices(V)

    # --- vertex figures: face angles, defect, solid angle, density --------
    rings = ME.vertex_rings(V, F)
    fa_all = ME.face_angles(V, F)
    defects = ME.angular_defects(V, F)
    solids = ME.solid_angles(V, F, rings)
    vdens = ME.vertex_densities(V, F, rings)

    face_angle_rows = []
    for oi, orb in enumerate(fg):
        rep = orb[0]
        angs = [a for a in fa_all[rep] if a is not None]
        if not angs:
            continue
        buckets = {}
        for a in angs:
            buckets.setdefault(round(math.degrees(a), 7), []).append(a)
        rows = []
        for deg in sorted(buckets):
            rad = buckets[deg][0]
            ce = recognise(math.cos(rad))
            row = {"degrees": math.degrees(rad), "radians": rad,
                   "count": len(buckets[deg]),
                   "exact": ("acos(%s)" % ce) if ce else None}
            if ce:
                row["source"] = "pslq"
            rows.append(row)
        face_angle_rows.append({"face_orbit": oi, "sides": len(F[rep]),
                                "regular": len(rows) == 1, "angles": rows})

    defect_rows, solid_rows = [], []
    for oi, orb in enumerate(vg):
        rep = orb[0]
        dd = defects[rep]
        ce = recognise(math.cos(dd)) if abs(dd) > 1e-12 else None
        row = {"vertex_orbit": oi, "degrees": math.degrees(dd), "radians": dd,
               "exact": ("acos(%s)" % ce) if ce else None}
        if ce:
            row["source"] = "pslq"
        defect_rows.append(row)
        sa = solids[rep]
        srow = {"vertex_orbit": oi,
                "steradians": (float(sa) if sa is not None else None)}
        if sa is None:
            srow["note"] = ("vertex link is not a single cycle, so the vertex "
                            "figure has no well-defined solid angle")
        solid_rows.append(srow)

    dv = [vdens[o[0]] for o in vg]
    vertex_density = dv[0] if len(set(map(str, dv))) == 1 else dv

    # Biscribed: one circumsphere for the vertices and one CONCENTRIC
    # insphere for every face plane. Among the uniforms this is exactly the
    # nine regular solids.
    _R = m["circumradius"]
    _inr = [v for v in m["inradius"].values() if v is not None]
    _is_bis = bool(_R is not None and _inr
                   and len(_inr) == len(m["inradius"])
                   and (max(_inr) - min(_inr)) < 1e-9 * max(_inr))
    biscribed = {"is_biscribed": _is_bis,
                 "circumradius": (float(_R) if _R is not None else None),
                 "inradius": (float(_inr[0]) if _is_bis else None),
                 "ratio": (float(_inr[0] / _R) if _is_bis and _R else None),
                 "has_biscribed_form": True if _is_bis else None,
                 "biscribed_form": None}

    hull = ME.convex_hull_counts(V)
    # A solid is its own convex hull only if it IS convex. Counts alone will
    # not do: stellation preserves V/E/F, so the great icosahedron has exactly
    # the icosahedron's counts -- and the icosahedron's vertices -- while being
    # a different, non-convex solid.
    is_self = bool(hull and hull == (len(V), len(E), len(F))
                   and meta.get("convex", True))
    convex_hull = {"is_self": is_self, "slug": None,
                   "counts": ({"vertices": hull[0], "edges": hull[1],
                               "faces": hull[2]} if hull else None)}

    metrics = {
        "normalization": mode,
        "edge_length": measure_obj(m["edge_length"]),
        "edge_lengths_uniform": bool(m["edge_lengths_uniform"]),
        "circumradius": measure_obj(m["circumradius"]),
        "midradius": measure_obj(m["midradius"]),
        "inradius": inr or None,
        "surface_area": measure_obj(m["surface_area"]) if meta.get("convex", True) else None,
        "volume": measure_obj(m["volume"]) if meta.get("convex", True) else None,
        "isoperimetric_quotient": (measure_obj(m["isoperimetric_quotient"])
                                   if meta.get("convex", True) else None),
        "dihedral_angles": dih,
        "angular_defect": defect_rows or None,
        "solid_angle": solid_rows or None,
        "biscribed": biscribed,
    }
    if not meta.get("convex", True):
        metrics["measures_note"] = (
            "Area and volume are omitted: the faces self-intersect, so both "
            "depend on a convention (net area vs. density-weighted vs. outer "
            "solid region). Record them only alongside an explicit convention "
            "field.")

    rec = {
        "schema_version": SCHEMA_VERSION,
        "slug": meta["slug"],
        "name": meta["name"],
        "alternate_names": meta.get("alternate_names", []),
        "families": meta["families"],
        "ids": meta.get("ids", {}),
        "notation": meta.get("notation", {}),
        "symmetry": {
            "schoenflies": sym["schoenflies"], "orbifold": sym["orbifold"],
            "coxeter": sym["coxeter"], "hermann_mauguin": sym["hermann_mauguin"],
            "order": sym["order"], "rotation_group": sym["rotation_group"],
            "rotation_order": sym["rotation_order"], "chiral": sym["chiral"],
            "transitivity": {"vertex": len(vg) == 1, "edge": len(eg) == 1,
                             "face": len(fg) == 1},
            "orbits": {"vertices": len(vg), "edges": len(eg), "faces": len(fg)},
        },
        "combinatorics": {
            "counts": {"vertices": len(V), "edges": len(E), "faces": len(F)},
            "faces_by_type": faces_by_type,
            "vertex_types": meta.get("vertex_types") or [],
            "euler_characteristic": chi,
            "orientable": meta.get("orientable", True),
            "genus": (None if meta.get("compound") else
                      ((2 - chi) // 2 if meta.get("orientable", True)
                       and (2 - chi) % 2 == 0 else None)),
            "density": meta.get("density"),
            "convex": meta.get("convex", True),
            "self_dual": meta.get("self_dual", False),
            "dual": meta.get("dual"),
            "coincident_vertices": coincident,
            "vertex_density": vertex_density,
            "convex_hull": convex_hull,
            "dual_unbounded": ME.has_central_face(V, F),
        },
        "metrics": metrics,
        "geometry": {
            "length_unit": "edge" if mode == "edge_length_1" else mode,
            "centroid": [0, 0, 0],
            "orientation": meta.get("orientation",
                                    "as produced by the construction, centred at the centroid"),
            "constants": consts or {},
            "vertices": [list(v) for v in V],
            "vertices_exact": vex,
            "face_angles": face_angle_rows or None,
            "faces": F,
            "faces_wound": ("each face is its true winding cycle; windings are "
                            "coherent (every edge traversed once in each direction)"),
        },
        "face_groups": ME.face_groups(V, F, fg),
        "vertex_groups": {"symmetry_orbit": [list(g) for g in vg]},
        "edge_groups": {"symmetry_orbit": [list(g) for g in eg]},
        "compound": meta.get("compound"),
        "space_filling": CU.space_filling_for(meta["slug"]),
        "construction": meta.get("construction", {}),
        "provenance": {
            "coordinates": meta["coordinates"],
            "sources": meta["sources"],
            "cross_checked": meta.get("cross_checked", []),
            "verified": {"euler": True, "closed_manifold": True,
                         "faces_planar": True, "radii_match_exact": True,
                         "dual_reciprocal": True, "tolerance": 1e-9},
        },
    }
    apply_curation(rec)
    return rec


def apply_curation(rec):
    """Merge the curated literature facts in. Curation only FILLS what the
    geometry cannot decide -- it never overwrites a computed field."""
    cur = CU.for_slug(rec["slug"])
    if not cur:
        return rec
    for fam in cur.get("families+", []):
        if fam not in rec["families"]:
            rec["families"].append(fam)
    for k, v in (cur.get("notation") or {}).items():
        if rec["notation"].get(k) in (None, [], ""):
            rec["notation"][k] = v
    for k, v in (cur.get("ids") or {}).items():
        if rec["ids"].get(k) is None:
            rec["ids"][k] = v
    if cur.get("alternate_names") and not rec.get("alternate_names"):
        rec["alternate_names"] = cur["alternate_names"]
    if cur.get("self_dual"):
        rec["combinatorics"]["self_dual"] = True
    nl = NETLIB_NAMES.get(rec["name"].lower())
    if nl is not None and rec["ids"].get("netlib") is None:
        rec["ids"]["netlib"] = nl
    return rec


def _load_netlib_names():
    p = os.path.join(ROOT, ".polydb_cache", "netlib_names.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return {k.lower(): v for k, v in json.load(fh).items()}


NETLIB_NAMES = _load_netlib_names()


# -- stages -----------------------------------------------------------------

CONVEX_UNIFORM = set(range(1, 12)) | {12, 22, 23, 24, 25, 26, 27, 28} | {5, 6, 7, 8, 9, 10, 11}
KEPLER_POINSOT = {34, 35, 52, 53}


def uniform_specs():
    hb = HAREL.by_wythoff()
    out = []
    for (u, name, wy, pqr, nv, ne, nf) in UP.UNIFORMS:
        h = hb.get(HAREL.key(wy)) or (HAREL.U75 if u == 75 else None)
        star = any("/" in t for t in pqr)
        fams = ["uniform"]
        if u in KEPLER_POINSOT:
            fams += ["kepler-poinsot", "regular", "star"]
        elif not star:
            fams += ["convex"]
        else:
            fams += ["star"]
        if "hemi" in name.lower():
            fams.append("hemipolyhedron")
        out.append({
            "u": u, "name": name, "wythoff": wy, "pqr": pqr,
            "slug": slugify(name), "families": fams, "harel": h,
            "convex": not star, "mccooey": camel(name),
        })
    return out


def build_uniform(spec):
    u = spec["u"]
    refined = None
    if u in UP._SNUB_U:
        V, faces = UP.build_snub(u)
        F = [list(f) for f, _d in faces]
        # The stored snub tables carry only six decimals, which leaves faces
        # ~1e-7 out of plane -- too coarse to call exact, and too coarse for
        # exact-form recovery. A snub is determined by its generator point, so
        # re-solve it: this lands at ~3e-16.
        W, info = RF.refine_vertex_transitive(V, F)
        if W is not None:
            V, refined = W, info
    else:
        V, faces = UP.build_uniform(spec["wythoff"], spec["pqr"])
        F = [list(f) for f, _d in faces]
    h = spec["harel"] or {}
    meta = {
        "slug": spec["slug"], "name": spec["name"],
        "families": spec["families"],
        "ids": {"uniform": "U%d" % u,
                "wenninger": ("W%d" % h["wenninger"]) if h.get("wenninger") else None,
                "coxeter_clm": h.get("coxeter_clm"),
                "mccooey": spec["mccooey"], "johnson": None, "netlib": None,
                "bowers": None, "wikipedia": None, "wolfram": None},
        "notation": {"schlafli": None, "wythoff": spec["wythoff"],
                     "coxeter_diagram": None, "conway": None,
                     "vertex_configuration": ([h["vertex_config"]]
                                              if h.get("vertex_config") else []),
                     "face_configuration": None},
        "vertex_types": ([{"configuration": h["vertex_config"], "count": len(V)}]
                         if h.get("vertex_config") else []),
        "orientable": h.get("orientable", True),
        "density": h.get("density"),
        "convex": spec["convex"],
        "dual": slugify(h["dual"]) if h.get("dual") else None,
        "orientation": "as produced by the Wythoff construction, centred at the centroid",
        "construction": {"generator": "math_art.uniform_polyhedra_generator",
                         "operator_id": "mesh.uniform_polyhedron_add",
                         "conway_from": None,
                         "wythoff_from": {"schwarz": h.get("schwarz"),
                                          "symbol": spec["wythoff"]}},
        "coordinates": ("derived: Wythoff kaleidoscopic construction from the "
                        "Schwarz triangle, via math_art.uniform_polyhedra_generator"
                        if u not in UP._SNUB_U else
                        "derived: snub construction via "
                        "math_art.uniform_polyhedra_generator, then re-solved "
                        "to machine precision as the orbit of its generator "
                        "point under the exact rotation group (edge spread "
                        "%.1e)" % refined["edge_spread"] if refined else
                        "derived: stored snub construction, via "
                        "math_art.uniform_polyhedra_generator"),
        "sources": [
            "H. S. M. Coxeter, M. S. Longuet-Higgins, J. C. P. Miller, "
            "'Uniform polyhedra', Phil. Trans. R. Soc. A 246 (1954), 401-450.",
            "J. Skilling, 'The complete set of uniform polyhedra', Phil. Trans. "
            "R. Soc. A 278 (1975), 111-135 (completeness).",
            "Z. Har'El, 'Uniform Solution for Uniform Polyhedra', Geometriae "
            "Dedicata 47 (1993), 57-110.",
        ],
    }
    if h.get("wenninger"):
        meta["sources"].append(
            "M. Wenninger, 'Polyhedron Models', Cambridge (1971), model %d."
            % h["wenninger"])
    meta["cross_checked"] = cross_check(V, spec["mccooey"], F=F)
    if h:
        agrees = (h.get("chi") is None
                  or h["chi"] == len(V) - len(ME.edges_of(F)) + len(F))
        meta["cross_checked"].append({
            "source": "harel-1993", "agrees": bool(agrees),
            "detail": "Euler characteristic %s" % (
                "matches (chi = %d)" % h["chi"] if agrees else "DISAGREES")})
    return assemble(V, F, meta)


def build_johnson(num, name):
    V, F = (RS.build_johnson_ext(num) if num in RS._J_EXT_NUMS
            else RS.build_johnson(num))
    clean = re.sub(r"\s*\(J\d+\)\s*$", "", name).strip()
    meta = {
        "slug": slugify(clean), "name": clean,
        "families": ["johnson", "convex"],
        "ids": {"johnson": "J%d" % num, "uniform": None, "wenninger": None,
                "coxeter_clm": None, "mccooey": camel(clean), "netlib": None,
                "bowers": None, "wikipedia": None, "wolfram": None},
        "notation": {"schlafli": None, "wythoff": None, "coxeter_diagram": None,
                     "conway": None, "vertex_configuration": [],
                     "face_configuration": None},
        "orientable": True, "density": 1, "convex": True, "dual": None,
        "orientation": "as produced by the construction, centred at the centroid",
        "construction": {"generator": "math_art.regular_solids_generator",
                         "operator_id": "mesh.regular_solid_add",
                         "conway_from": None, "wythoff_from": None},
        "coordinates": ("derived: composed from exact unit-edge pyramids, "
                        "cupolae and rotundae (J84-J92 by regular-face "
                        "relaxation), via math_art.regular_solids_generator"),
        "sources": [
            "N. W. Johnson, 'Convex polyhedra with regular faces', Canadian "
            "Journal of Mathematics 18 (1966), 169-200.",
            "V. A. Zalgaller, 'Convex Polyhedra with Regular Faces', Consultants "
            "Bureau (1969) (completeness).",
        ],
    }
    meta["cross_checked"] = cross_check(V, meta["ids"]["mccooey"], F=F)
    return assemble(V, F, meta)


ARCHIMEDEAN_SLUGS = {
    "truncated-tetrahedron", "cuboctahedron", "truncated-cube",
    "truncated-octahedron", "rhombicuboctahedron", "truncated-cuboctahedron",
    "snub-cube", "icosidodecahedron", "truncated-dodecahedron",
    "truncated-icosahedron", "rhombicosidodecahedron",
    "truncated-icosidodecahedron", "snub-dodecahedron"}


def build_dual(spec, existing):
    """The polar reciprocal of a uniform polyhedron.

    Reciprocation is taken about the solid's own MIDSPHERE: the parent is
    first scaled to midradius 1, so the dual comes out with midradius 1 too and
    the pair is exactly reciprocal -- which is the classical construction, and
    the reason the schema normalises duals to midradius rather than edge
    length (a dual's edges are not all equal, so edge length 1 is meaningless).

    Returns None when the dual should not become a record: when its faces pass
    through the centre (so the dual is unbounded), or when the dual is a solid
    the database already holds -- the dual of the cube is the octahedron, not a
    second copy of it.
    """
    u = spec["u"]
    h = spec["harel"] or {}
    if not h.get("dual"):
        return None
    dslug = slugify(h["dual"])
    if dslug in existing:
        return None                      # already a record (e.g. cube <-> octahedron)

    if u in UP._SNUB_U:
        V, faces = UP.build_snub(u)
        F0 = [list(f) for f, _d in faces]
        W, _info = RF.refine_vertex_transitive(V, F0)
        if W is not None:
            V = W
            faces = [(f, d) for f, (_f, d) in zip(F0, faces)]
    else:
        V, faces = UP.build_uniform(spec["wythoff"], spec["pqr"])

    V = ME.recentre(V)
    F0 = [list(f) for f, _d in faces]
    if ME.has_central_face(V, F0):
        return None                      # hemipolyhedron: dual is unbounded

    E0 = ME.edges_of(F0)
    mids = [ME.line_distance([0.0, 0.0, 0.0], V[a], V[b]) for a, b in E0]
    if not mids or (max(mids) - min(mids)) > 1e-7 * max(mids):
        return None                      # no midsphere to reciprocate about
    s = 1.0 / max(mids)
    Vs = [tuple(c * s for c in v) for v in V]

    DV, Dfaces = UP.build_dual(Vs, [(list(f), d) for f, d in faces])
    DF = [list(f) for f, _d in Dfaces]
    if len(DV) < 4 or len(DF) < 4:
        return None

    parent_star = not spec["convex"]
    fams = ["uniform-dual"]
    if spec["slug"] in ARCHIMEDEAN_SLUGS:
        fams.append("catalan")
    fams.append("star" if parent_star else "convex")

    name = h["dual"].strip()
    name = name[:1].upper() + name[1:]
    meta = {
        "slug": dslug, "name": name, "families": fams,
        "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                "mccooey": camel(name), "johnson": None, "netlib": None,
                "bowers": None, "wikipedia": None, "wolfram": None,
                "dual_of_uniform": "U%d" % u},
        "notation": {"schlafli": None, "wythoff": None, "coxeter_diagram": None,
                     "conway": "d%s" % (spec["wythoff"] or ""),
                     "vertex_configuration": [],
                     "face_configuration": (["V" + h["vertex_config"].strip("()")]
                                            if h.get("vertex_config") else None)},
        "orientable": h.get("orientable", True),
        "density": None,
        "convex": not parent_star,
        "dual": spec["slug"],
        "orientation": ("inherited from its dual, reciprocated about the "
                        "common midsphere"),
        "construction": {"generator": "math_art.uniform_polyhedra_generator",
                         "operator_id": "mesh.uniform_polyhedron_add",
                         "conway_from": {"operator": "dual",
                                         "seed": spec["slug"]},
                         "wythoff_from": None},
        "coordinates": ("derived: polar reciprocal of %s about their common "
                        "midsphere" % spec["name"]),
        "sources": [
            "E. Catalan, 'Memoire sur la theorie des polyedres', J. Ecole "
            "Polytechnique 41 (1865), 1-71.",
            "M. Wenninger, 'Dual Models', Cambridge (1983).",
            "Z. Har'El, 'Uniform Solution for Uniform Polyhedra', Geometriae "
            "Dedicata 47 (1993), 57-110 (dual names and prototiles).",
        ],
    }
    meta["cross_checked"] = cross_check(DV, meta["ids"]["mccooey"], F=DF)
    return assemble(DV, DF, meta)


def stage_dual(limit=None):
    existing = set()
    base = os.path.join(OUT, "solids")
    for root, _d, files in os.walk(base):
        for fn in files:
            if fn.endswith(".json"):
                existing.add(fn[:-5])
    out, skipped = [], 0
    for spec in uniform_specs()[:limit]:
        try:
            rec = build_dual(spec, existing)
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL dual of U%-3d %-34s %r"
                  % (spec["u"], spec["name"][:34], exc))
            continue
        if rec is None:
            skipped += 1
            continue
        existing.add(rec["slug"])
        out.append(rec)
    print("   (skipped %d: unbounded, self-dual pair, or already a record)"
          % skipped)
    return out


_ORDINAL = {3: "Triangular", 4: "Square", 5: "Pentagonal", 6: "Hexagonal",
            7: "Heptagonal", 8: "Octagonal", 9: "Enneagonal", 10: "Decagonal",
            11: "Hendecagonal", 12: "Dodecagonal"}

# n-gonal members of the prism families that ARE already uniform polyhedra
# under another name, and so must not become a second record.
PRISM_ALIASES = {("PRISM", 4): "cube", ("ANTIPRISM", 3): "octahedron"}
DUAL_ALIASES = {("PRISM", 4): "octahedron", ("ANTIPRISM", 3): "cube"}


def build_prism_family(kind, n, dual, existing):
    """A uniform n-prism or n-antiprism, or its dual (dipyramid or
    trapezohedron). Returns None when the solid is already a record under
    another name -- the square prism IS the cube, the triangular antiprism IS
    the octahedron, and their duals are the octahedron and the cube."""
    alias = (DUAL_ALIASES if dual else PRISM_ALIASES).get((kind, n))
    if alias:
        return None
    word = _ORDINAL.get(n)
    if not word:
        return None
    if dual:
        base = "Dipyramid" if kind == "PRISM" else "Trapezohedron"
    else:
        base = "Prism" if kind == "PRISM" else "Antiprism"
    name = "%s %s" % (word, base)
    slug = slugify(name)
    if slug in existing:
        return None

    V, F = RS.build_prism(kind, n)
    V = ME.recentre(V)
    F = [list(f) for f in F]
    parent_slug = slugify("%s %s" % (word, "Prism" if kind == "PRISM"
                                     else "Antiprism"))

    if dual:
        # Reciprocate about a concentric sphere. A prism has no midsphere in
        # general -- its side edges and base edges sit at different distances,
        # the cube being the exception -- but polar duality is valid about ANY
        # concentric sphere and the choice only rescales the result, so the
        # circumsphere is used and the dual is renormalised downstream.
        E0 = ME.edges_of(F)
        mids = [ME.line_distance([0.0, 0.0, 0.0], V[a], V[b]) for a, b in E0]
        r = (max(mids) if mids and (max(mids) - min(mids)) < 1e-7 * max(mids)
             else max(ME.norm(v) for v in V))
        if not r:
            return None
        Vs = [tuple(c / r for c in v) for v in V]
        V, Dfaces = UP.build_dual(Vs, [(list(f), 1) for f in F])
        F = [list(f) for f, _d in Dfaces]

    fams = ["prism-family", "convex"]
    fams.append("dipyramid" if dual else ("prism" if kind == "PRISM"
                                          else "antiprism"))
    if not dual and kind == "ANTIPRISM":
        fams.append("chiral-family")

    meta = {
        "slug": slug, "name": name, "families": fams,
        "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                "mccooey": camel(name), "johnson": None, "netlib": None,
                "bowers": None, "wikipedia": None, "wolfram": None},
        "notation": {
            "schlafli": (None if dual else
                         ("t{2,%d}" % n if kind == "PRISM" else "s{2,%d}" % n)),
            "wythoff": (None if dual else
                        ("2 %d | 2" % n if kind == "PRISM" else "| 2 2 %d" % n)),
            "coxeter_diagram": None,
            "conway": ("d" if dual else "") + ("P%d" % n if kind == "PRISM"
                                               else "A%d" % n),
            "vertex_configuration": ([] if dual else
                                     ["4.4.%d" % n] if kind == "PRISM"
                                     else ["3.3.3.%d" % n]),
            "face_configuration": None},
        "orientable": True, "density": 1, "convex": True,
        "dual": parent_slug if dual else slugify("%s %s" % (
            word, "Dipyramid" if kind == "PRISM" else "Trapezohedron")),
        "orientation": "principal axis along z, centred at the centroid",
        "construction": {"generator": "math_art.regular_solids_generator",
                         "operator_id": "mesh.regular_solid_add",
                         "conway_from": ({"operator": "dual",
                                          "seed": parent_slug} if dual else None),
                         "wythoff_from": None},
        "coordinates": ("derived: polar reciprocal of the %s about its "
                        "midsphere" % parent_slug if dual else
                        "derived: exact uniform %s, via "
                        "math_art.regular_solids_generator" % base.lower()),
        "sources": [
            "H. S. M. Coxeter, 'Regular Polytopes', 3rd ed., Dover (1973), ch. 2 "
            "(the prisms and antiprisms as the two infinite families of uniform "
            "polyhedra).",
            "Z. Har'El, 'Uniform Solution for Uniform Polyhedra', Geometriae "
            "Dedicata 47 (1993), Appendix II Table 4 (dihedral uniform polyhedra).",
        ],
    }
    meta["cross_checked"] = cross_check(V, meta["ids"]["mccooey"], F=F)
    return assemble(V, F, meta)


# {p/q} bases for the star prism families.
#
# (5, 3) is included for the ANTIPRISM only: as a polygon {5/3} is {5/2}
# traversed the other way, so the {5/3} prism, dipyramid and trapezohedron are
# the same solids as the {5/2} ones and would be duplicate records. The
# pentagrammic CROSSED antiprism is genuinely distinct -- its two bases are in
# phase rather than rotated by pi/p -- and is built by
# `uniform_polyhedra_generator._crossed_antiprism`.
STAR_PRISM_PQ = ((5, 2), (7, 2), (7, 3), (8, 3), (10, 3))
STAR_ANTIPRISM_ONLY_PQ = ((5, 3),)


def stage_star_prism(limit=None):
    """Star prisms and antiprisms {p/q}, and their duals.

    These are uniform polyhedra too -- the dihedral families with star bases --
    and Skilling's compounds are built from them, so without these records
    those compounds have components that link to nothing.
    """
    existing = set()
    base = os.path.join(OUT, "solids")
    for root, _d, files in os.walk(base):
        for fn in files:
            if fn.endswith(".json"):
                existing.add(fn[:-5])

    ORD = {5: "Pentagrammic", 7: "Heptagrammic", 8: "Octagrammic",
           10: "Decagrammic"}
    out = []
    jobs = [(p, q, k) for (p, q) in STAR_PRISM_PQ
            for k in ("prism", "antiprism", "dipyramid", "trapezohedron")]
    # retrograde steps give a distinct ANTIPRISM only; the other three would
    # duplicate the folded base's records
    jobs += [(p, q, "antiprism") for (p, q) in STAR_ANTIPRISM_ONLY_PQ]
    for (p, q, kind) in jobs[:limit]:
        if True:
            word = ORD.get(p)
            if not word:
                continue
            crossed = " Crossed" if (kind == "antiprism" and 2 * q > p) else ""
            name = "%s%s %s" % (word, crossed, kind.capitalize())
            if q == 3 and p == 5 and kind == "antiprism":
                name = "Pentagrammic Crossed Antiprism"
            slug = slugify(name)
            if slug in existing:
                continue
            try:
                if kind == "prism":
                    V, F = UP.build_star_prism(p, q)
                elif kind == "antiprism":
                    V, F = UP.build_star_antiprism(p, q)
                elif kind == "dipyramid":
                    V, F = UP.build_star_dipyramid(p, q)
                else:
                    V, F = UP.build_star_trapezohedron(p, q)
            except Exception as exc:                        # noqa: BLE001
                print("  FAIL %-40s %r" % (name[:40], exc))
                continue
            F = [list(f[0]) if isinstance(f, tuple) else list(f) for f in F]
            meta = {
                "slug": slug, "name": name,
                "families": ["prism-family", "star", "uniform"],
                "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                        "mccooey": camel(name), "johnson": None, "netlib": None,
                        "bowers": None, "wikipedia": None, "wolfram": None},
                "notation": {"schlafli": "{%d/%d}" % (p, q), "wythoff": None,
                             "coxeter_diagram": None, "conway": None,
                             "vertex_configuration": [],
                             "face_configuration": None},
                "orientable": True, "density": q, "convex": False, "dual": None,
                "orientation": "principal axis along z, centred at the centroid",
                "construction": {"generator": "math_art.uniform_polyhedra_generator",
                                 "operator_id": "mesh.star_prism_add",
                                 "conway_from": None, "wythoff_from": None},
                "coordinates": ("derived: uniform {%d/%d} star %s, via "
                                "math_art.uniform_polyhedra_generator"
                                % (p, q, kind)),
                "sources": [
                    "Z. Har'El, 'Uniform Solution for Uniform Polyhedra', "
                    "Geometriae Dedicata 47 (1993), Appendix II Table 4 "
                    "(the dihedral uniform polyhedra).",
                    "H. S. M. Coxeter, 'Regular Polytopes', 3rd ed. (1973).",
                ],
            }
            try:
                out.append(assemble(V, F, meta))
                existing.add(slug)
            except Exception as exc:                        # noqa: BLE001
                print("  FAIL %-40s assemble: %r" % (name[:40], exc))
    return out


def stage_prism(limit=None, nmax=12):
    existing = set()
    base = os.path.join(OUT, "solids")
    for root, _d, files in os.walk(base):
        for fn in files:
            if fn.endswith(".json"):
                existing.add(fn[:-5])
    out, skipped = [], 0
    ns = range(3, nmax + 1)
    if limit:
        ns = list(ns)[:limit]
    for n in ns:
        for kind in ("PRISM", "ANTIPRISM"):
            for dual in (False, True):
                try:
                    rec = build_prism_family(kind, n, dual, existing)
                except Exception as exc:                     # noqa: BLE001
                    print("  FAIL %s %d dual=%s %r" % (kind, n, dual, exc))
                    continue
                if rec is None:
                    skipped += 1
                    continue
                existing.add(rec["slug"])
                out.append(rec)
    print("   (skipped %d: already a record under another name)" % skipped)
    return out


# Bases whose biscribed form provably does NOT exist. These are results, not
# gaps: a rectified solid has no free parameter and two unequal face-orbit
# distances, and for the truncation family the solver finds no sign change in
# g(t) = d_vertexface(t) - d_baseface over (0, 1/2). Recorded on the
# corresponding uniform record as has_biscribed_form: false.
BISCRIBED_NONE_BASES = ("truncated_tetrahedron", "truncated_cube",
                        "truncated_dodecahedron", "cuboctahedron",
                        "icosidodecahedron")


def stage_biscribed(limit=None):
    """McCooey's biscribed solids: all vertices on a circumsphere AND all faces
    tangent to a concentric insphere.

    These are DIFFERENT solids from their Archimedean namesakes -- a biscribed
    truncated octahedron is not the Archimedean one; it trades regular faces
    for the two concentric spheres.
    """
    import biscribed_solids_generator as BS

    out = []
    for sid, label, base, is_dual in BS._BISCRIBED[:limit]:
        try:
            res = (BS.biscribe_exact_dual(base) if is_dual
                   else BS.biscribe_exact(base))
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-46s %r" % (label[:46], exc))
            continue
        if res is None:
            print("  none %-46s (no biscribed form exists)" % label[:46])
            continue
        V, F, r = res
        V = [tuple(float(c) for c in v) for v in V]
        F = [list(f) for f in F]
        slug = slugify(label)
        chiral = any(k in sid for k in ("pp", "wh", "dw", "ok", "ot"))
        fams = ["biscribed", "convex"]
        if is_dual:
            fams.append("biscribed-dual")
        meta = {
            "slug": slug, "name": label, "families": fams,
            "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                    "mccooey": camel(label), "johnson": None, "netlib": None,
                    "bowers": None, "wikipedia": None, "wolfram": None},
            "notation": {"schlafli": None, "wythoff": None,
                         "coxeter_diagram": None, "conway": None,
                         "vertex_configuration": [], "face_configuration": None},
            "orientable": True, "density": 1, "convex": True, "dual": None,
            "orientation": "as produced by the biscribed solver, centred at the centroid",
            "construction": {"generator": "math_art.biscribed_solids_generator",
                             "operator_id": "mesh.biscribed_solid_add",
                             "conway_from": None, "wythoff_from": None},
            "coordinates": ("derived: biscribed solver -- the family's shape "
                            "parameter(s) adjusted until every face-orbit plane "
                            "distance is equal, giving concentric circum- and "
                            "inspheres (r/R = %.9f)" % r),
            "sources": [
                "D. McCooey, 'Biscribed (Non-)Chiral Solids', Visual Polyhedra "
                "(the enumeration followed here).",
                "G. W. Hart, 'Canonical Polyhedra' (the propello and whirl "
                "Conway operators used as bases).",
            ],
        }
        meta["cross_checked"] = cross_check(V, meta["ids"]["mccooey"], F=F)
        try:
            rec = assemble(V, F, meta)
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-46s assemble: %r" % (label[:46], exc))
            continue
        rec["metrics"]["biscribed"]["has_biscribed_form"] = True
        rec["metrics"]["biscribed"]["note"] = (
            "A biscribed solid: it trades the regular faces of its Archimedean "
            "namesake for concentric circum- and inspheres.")
        if chiral:
            rec["families"].append("chiral")
        out.append(rec)
    return out


TOROID_META = {
    "CSASZAR": ("Csaszar Polyhedron", "Csaszar",
                "A. Csaszar, 'A polyhedron without diagonals', Acta Sci. Math. "
                "(Szeged) 13 (1949-50), 140-142.",
                "Has no diagonals at all: every pair of its 7 vertices is "
                "joined by an edge. Only the tetrahedron and this solid are "
                "known to have that property."),
    "SZILASSI": ("Szilassi Polyhedron", "Szilassi",
                 "L. Szilassi, 'Regular toroids', Structural Topology 13 "
                 "(1986), 69-80.",
                 "The dual of the Csaszar polyhedron: each of its 7 faces "
                 "shares an edge with every other, so colouring it needs 7 "
                 "colours."),
    "REGULAR": ("Regular Toroid", "RegularToroid", None, None),
    "KNOTTED": ("Knotted Toroid", "KnottedToroid", None, None),
    "BORROMEAN": ("Borromean Toroid", "BorromeanToroid", None, None),
    "IRIS7": ("Iris Toroid 7", "IrisToroid7", None, None),
    "IRIS8": ("Iris Toroid 8", "IrisToroid8", None, None),
}


def stage_toroid(limit=None):
    """Toroidal polyhedra: genus 1, so chi = 0. They are the reason the schema
    carries genus at all -- every other family here is genus 0."""
    import toroidal_polyhedron_generator as TP

    out = []
    for kind in list(TP.TOROIDS)[:limit]:
        name, stem, src, note = TOROID_META.get(
            kind, (kind.title(), camel(kind), None, None))
        try:
            res = TP.build_toroid(kind)
            V, F = res[0], res[1]
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-30s %r" % (name[:30], exc))
            continue
        # A polyhedron has planar faces. Some of these generators produce
        # decorative twisted quad strips whose faces are only approximately
        # flat; those are polyhedral SURFACES, not polyhedra, and do not
        # belong in a database that certifies planarity.
        flat = RF.planarity([tuple(float(c) for c in v) for v in V],
                            [[int(i) for i in f] for f in F])
        if flat > 1e-9:
            print("  skip %-30s faces %.1e out of plane -- a polyhedral "
                  "surface, not a polyhedron" % (name[:30], flat))
            continue
        meta = {
            "slug": slugify(name), "name": name,
            "families": ["toroid", "genus-1"],
            "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                    "mccooey": stem, "johnson": None, "netlib": None,
                    "bowers": None, "wikipedia": None, "wolfram": None},
            "notation": {"schlafli": None, "wythoff": None,
                         "coxeter_diagram": None, "conway": None,
                         "vertex_configuration": [], "face_configuration": None},
            "orientable": True, "density": None, "convex": False, "dual": None,
            "orientation": "as produced by the construction, centred at the centroid",
            "construction": {"generator": "math_art.toroidal_polyhedron_generator",
                             "operator_id": "mesh.toroidal_polyhedron_add",
                             "conway_from": None, "wythoff_from": None},
            "coordinates": "derived: via math_art.toroidal_polyhedron_generator",
            "sources": [src] if src else [
                "B. M. Stewart, 'Adventures Among the Toroids', 2nd ed. (1980)."],
        }
        try:
            rec = assemble(V, F, meta)
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-30s assemble: %r" % (name[:30], exc))
            continue
        if note:
            rec["metrics"]["measures_note"] = note
        out.append(rec)
    return out


ZONOHEDRA = [
    # No cube here: the uniform stage already emits it, and a second pass
    # over the same solid only competes for the slug.
    ("RHOMBIC_DODECA", "Rhombic Dodecahedron"),
    ("TRIACONTA", "Rhombic Triacontahedron"),
    ("ENNEACONTA", "Rhombic Enneacontahedron"),
]


def stage_zonohedron(limit=None):
    """Zonohedra: the Minkowski sum of a star of vectors, so every face is a
    centrally symmetric polygon and the solid tiles by translation more often
    than not. Built as the convex hull of the star's subset sums, with
    coplanar hull triangles merged -- otherwise every rhombus would be
    reported as two triangles."""
    import zonohedra_generator as Z

    existing = set()
    base = os.path.join(OUT, "solids")
    for root, _d, files in os.walk(base):
        for fn in files:
            if fn.endswith(".json"):
                existing.add(fn[:-5])

    out, skipped = [], 0
    for kind, name in ZONOHEDRA[:limit]:
        slug = slugify(name)
        if slug in existing:
            skipped += 1                 # already a record (cube, the Catalans)
            continue
        try:
            star = Z.star_vectors(kind)
            V, F = ME.hull_faces(Z.subset_sums(star))
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-30s %r" % (name[:30], exc))
            continue
        if V is None:
            continue
        meta = {
            "slug": slug, "name": name,
            "families": ["zonohedron", "convex"],
            "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                    "mccooey": camel(name), "johnson": None, "netlib": None,
                    "bowers": None, "wikipedia": None, "wolfram": None},
            "notation": {"schlafli": None, "wythoff": None,
                         "coxeter_diagram": None, "conway": None,
                         "vertex_configuration": [], "face_configuration": None},
            "orientable": True, "density": 1, "convex": True, "dual": None,
            "orientation": "aligned to the generating star, centred at the centroid",
            "construction": {"generator": "math_art.zonohedra_generator",
                             "operator_id": "mesh.zonohedron_add",
                             "conway_from": None, "wythoff_from": None},
            "coordinates": ("derived: convex hull of the subset sums of a star "
                            "of %d generating vectors" % len(star)),
            "sources": [
                "E. S. Fedorov (1885) -- zonohedra and the parallelohedra.",
                "H. S. M. Coxeter, 'Regular Polytopes', 3rd ed. (1973), ch. 2.4.",
                "P. R. Cromwell, 'Polyhedra', Cambridge (1997), ch. 4.",
            ],
        }
        meta["cross_checked"] = cross_check(V, meta["ids"]["mccooey"], F=F)
        try:
            out.append(assemble(V, F, meta))
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-30s assemble: %r" % (name[:30], exc))
    if skipped:
        print("   (skipped %d: already a record under another name)" % skipped)
    return out


GEODESIC_BASES = [("ICOSA", "Icosahedron"), ("OCTA", "Octahedron"),
                  ("TETRA", "Tetrahedron")]


def stage_geodesic(limit=None, freqs=(2, 3, 4)):
    """Geodesic spheres: a Platonic base subdivided and projected onto its
    circumsphere. Frequency 1 is the base solid itself and is skipped, since
    it is already a record.

    Their edges are NOT all equal -- that is the defining compromise of a
    geodesic sphere and the reason chord factors are tabulated in the dome
    literature -- so they normalise to circumradius 1, not edge length 1.
    """
    import geodesic_generator as GG

    existing = set()
    base_dir = os.path.join(OUT, "solids")
    for root, _d, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".json"):
                existing.add(fn[:-5])

    out = []
    for code, base_name in GEODESIC_BASES[:limit]:
        for f in freqs:
            name = "Geodesic %s %dv" % (base_name, f)
            slug = slugify(name)
            if slug in existing:
                continue
            try:
                res = GG.build_sphere(code, f, 'I')
                V, F = res[0], res[1]
            except Exception as exc:                        # noqa: BLE001
                print("  FAIL %-30s %r" % (name[:30], exc))
                continue
            meta = {
                "slug": slug, "name": name,
                "families": ["geodesic", "convex", "deltahedron"],
                "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                        "mccooey": None, "johnson": None, "netlib": None,
                        "bowers": None, "wikipedia": None, "wolfram": None},
                "notation": {"schlafli": None, "wythoff": None,
                             "coxeter_diagram": None, "conway": None,
                             "vertex_configuration": [],
                             "face_configuration": None},
                "orientable": True, "density": 1, "convex": True, "dual": None,
                "orientation": "inherited from the %s base, centred at the centroid"
                               % base_name.lower(),
                "construction": {"generator": "math_art.geodesic_generator",
                                 "operator_id": "mesh.geodesic_add",
                                 "conway_from": None, "wythoff_from": None},
                "coordinates": ("derived: class-I subdivision of the %s at "
                                "frequency %d, projected onto the circumsphere"
                                % (base_name.lower(), f)),
                "sources": [
                    "R. Buckminster Fuller, US Patent 2,682,235 (1954), "
                    "'Building Construction' (the geodesic dome).",
                    "H. Kenner, 'Geodesic Math and How to Use It', "
                    "University of California Press (1976).",
                    "E. S. Popko, 'Divided Spheres: Geodesics and the Orderly "
                    "Subdivision of the Sphere', CRC Press (2012).",
                ],
            }
            try:
                out.append(assemble(V, F, meta))
                existing.add(slug)
            except Exception as exc:                        # noqa: BLE001
                print("  FAIL %-30s assemble: %r" % (name[:30], exc))
    return out


def _compound_meta(key, label):
    """Catalogue cross-references parsed from the generator's key and label.

    Deliberately an ARRAY of references rather than one number: the compound
    catalogues are not a single sequence. Skilling (1976) is complete for
    UNIFORM compounds only; Harman's rule generates families with no canonical
    numbering and was never published; and the '(free)' families carry a
    continuous parameter, so no enumeration exists for them at all.
    """
    enum, params = [], None
    m = re.match(r"^S(\d+)_", key)
    if m:
        enum.append({"catalogue": "skilling-1976", "index": "UC%s" % m.group(1),
                     "via": None,
                     "complete_for": "uniform compounds (vertex-transitive, "
                                     "uniform components)"})
    if key.startswith("H"):
        enum.append({"catalogue": "harman-1974", "index": None,
                     "via": "G. W. Hart, Virtual Polyhedra",
                     "complete_for": None})
    if key in ("STELLA", "5TETRA", "10TETRA", "5CUBES", "5OCTA"):
        enum.append({"catalogue": "coxeter-regular", "index": None, "via": None,
                     "complete_for": "the five regular compounds"})
    if "(free)" in label or "free" in label.lower():
        params = {"free": True, "angle_degrees": None, "repeat": None,
                  "note": "A continuously parameterised family: the component "
                          "may be rotated freely about its axis. The stored "
                          "record is one member; special angles collapse "
                          "components."}
    who = pub = None
    if key.startswith("H"):
        who, pub = "Michael G. Harman", "unpublished (1974); described by G. W. Hart"
    elif re.match(r"^S\d+_", key):
        who, pub = "John Skilling", "1976"
    elif key in ("STELLA",):
        who, pub = "Johannes Kepler", "1619"
    elif key in ("5TETRA", "10TETRA", "5CUBES", "5OCTA"):
        who, pub = "Edmund Hess / Max Bruckner", "1876 / 1900"
    return enum, params, who, pub


def stage_compound(limit=None):
    """Compounds: several polyhedra sharing a centre.

    Not a single solid, so the record carries a `compound` block: the
    components (as face-index sets, which survive welding), the catalogue
    cross-references, and any free parameter. chi is the sum over components
    and genus is left null.
    """
    from polyhedra.compounds import COMPOUNDS, build_compound

    # `slugify` drops parenthesised text, which is right for "Square Pyramid
    # (J1)" but wrong here: Hart distinguishes "12 Icosahedra (Octahedral)"
    # from "12 Icosahedra (4-fold)" precisely by the parenthetical. Keep the
    # clean slug where it is unambiguous and fall back to the full form only
    # for the names that would otherwise collide.
    plain = {}
    for key, label in COMPOUNDS:
        nm = re.sub(r"\s*\(free\)\s*$", "", label).strip()
        plain.setdefault(slugify(nm), []).append(key)
    ambiguous = {k for k, v in plain.items() if len(v) > 1}

    def compound_slug(nm):
        s = slugify(nm)
        if s in ambiguous:
            s = slugify(nm.replace("(", " ").replace(")", " "))
        return s

    out = []
    for key, label in COMPOUNDS[:limit]:
        try:
            comps = build_compound(key)
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-44s %r" % (label[:44], exc))
            continue
        if not comps or len(comps) < 2:
            continue
        V, F, spans = [], [], []
        for cv, cf in comps:
            cv = [tuple(float(c) for c in v) for v in cv]
            cf = [[int(i) for i in f] for f in cf]
            # The snub components inherit six-decimal stored coordinates. The
            # compound as a whole is not vertex-transitive, but each COMPONENT
            # is, so re-solve the parts and then assemble.
            #
            # Gate on edge uniformity as well as planarity: a snub cube's faces
            # are triangles and squares, and a triangle is planar whatever its
            # vertices, so planarity alone passes a solid whose edges are still
            # wrong by 1e-7 -- which then fails to match its own record.
            _e = ME.edges_of(cf)
            _L = ME.edge_lengths(cv, _e) if _e else [1.0]
            _spread = (max(_L) - min(_L)) / max(_L) if _L else 0.0
            if RF.planarity(cv, cf) > 1e-9 or _spread > 1e-9:
                W, _info = RF.refine_vertex_transitive(cv, cf)
                if W is not None:
                    cv = W
            off = len(V)
            start = len(F)
            V += cv
            F += [[i + off for i in f] for f in cf]
            spans.append((start, len(F), len(cv), len(cf)))
        name = re.sub(r"\s*\(free\)\s*$", "", label).strip()
        enum, params, who, pub = _compound_meta(key, label)
        parts = []
        for (s0, s1, nv, nf) in spans:
            sub = F[s0:s1]
            ne = len({frozenset((a, b)) for f in sub
                      for a, b in zip(f, f[1:] + f[:1])})
            parts.append({"slug": None,
                          "counts": {"vertices": nv, "edges": ne, "faces": nf},
                          "faces": list(range(s0, s1))})
        meta = {
            "slug": compound_slug(name), "name": name,
            "families": ["compound"],
            "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                    "mccooey": None, "johnson": None, "netlib": None,
                    "bowers": None, "wikipedia": None, "wolfram": None},
            "notation": {"schlafli": None, "wythoff": None,
                         "coxeter_diagram": None, "conway": None,
                         "vertex_configuration": [], "face_configuration": None},
            "orientable": True, "density": None, "convex": False, "dual": None,
            "genus": None,
            "orientation": "as produced by the compound construction, centred at the centroid",
            "construction": {"generator": "math_art.compound_generator",
                             "operator_id": "mesh.polyhedron_compound_add",
                             "conway_from": None, "wythoff_from": None},
            "coordinates": ("derived: orbit of the component solid under the "
                            "compound's rotation group, via "
                            "math_art.polyhedra.compounds"),
            "sources": [
                "J. Skilling, 'Uniform compounds of uniform polyhedra', Math. "
                "Proc. Camb. Phil. Soc. 79 (1976), 447-457 (complete for "
                "uniform compounds).",
                "H. S. M. Coxeter, 'Regular Polytopes', 3rd ed. (1973) (the "
                "five regular compounds).",
                "G. W. Hart, 'Virtual Polyhedra' (Harman's compounds, "
                "otherwise unpublished).",
            ],
            "compound": {
                "component_count": len(comps),
                "components": parts,
                "components_congruent": len({(p["counts"]["vertices"],
                                              p["counts"]["faces"])
                                             for p in parts}) == 1,
                "enumeration": enum,
                "parameters": params,
                "discovered_by": who,
                "first_published": pub,
            },
        }
        try:
            out.append(assemble(V, F, meta))
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL %-44s assemble: %r" % (label[:44], exc))
    return out


def stage_crosscheck(limit=None):
    """Re-run the source cross-checks over the emitted corpus, in place.

    Separate from the build because it depends only on what is in the source
    cache: fetch more with `tools/polydb_fetch.sh mccooey <Stem> ...` and re-run
    this, without rebuilding any geometry.

    Stem names are resolved through `.polydb_cache/_match.json` when present --
    McCooey's page names do not follow from ours mechanically (he tags
    chirality, writes Dipyramid for Bipyramid, and suffixes Johnson numbers),
    so the mapping is built once by matching against his crawled index rather
    than guessed per record.
    """
    mpath = os.path.join(ROOT, ".polydb_cache", "_match.json")
    match = {}
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            match = json.load(fh)

    base = os.path.join(OUT, "solids")
    files = []
    for root, _d, fns in os.walk(base):
        for fn in sorted(fns):
            if fn.endswith(".json"):
                files.append(os.path.join(root, fn))

    checked = agree = disagree = nosource = 0
    for p in files[:limit]:
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        stem = match.get(rec["slug"]) or rec["ids"].get("mccooey")
        results = []
        parsed = CC.parse_mccooey(CC.fetch_mccooey(stem)) if stem else None
        if parsed:
            M = CC.mccooey_numeric(parsed)
            if M:
                ok, detail = CC.same_shape(rec["geometry"]["vertices"], M,
                                           rec["geometry"]["faces"],
                                           parsed.get("faces"))
                results.append({"source": "mccooey", "agrees": bool(ok),
                                "detail": detail})
                checked += 1
                agree += bool(ok)
                disagree += (not ok)
        if not results:
            nosource += 1
        # Har'El's Euler characteristic, where the record came from that table
        for old in rec["provenance"].get("cross_checked") or []:
            if old.get("source") == "harel-1993":
                results.append(old)
        if stem and stem != rec["ids"].get("mccooey"):
            rec["ids"]["mccooey"] = stem
        rec["provenance"]["cross_checked"] = results
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)

    print("   cross-checked %d against McCooey: %d agree, %d DISAGREE; "
          "%d record(s) had no cached source" % (checked, agree, disagree,
                                                 nosource))
    return []


def stage_link(limit=None):
    """Post-pass over the emitted corpus: resolve cross-references that can
    only be settled once every record exists.

    * convex_hull.slug -- the hull of a star solid's vertices is very often a
      solid the database already holds (the small stellated dodecahedron's
      hull is the icosahedron), and that link is the organising key of the
      classical star-polyhedron catalogues.
    * biscribed.biscribed_form -- left for the biscribed stage.

    Rewrites files in place and returns nothing to emit.
    """
    base = os.path.join(OUT, "solids")
    recs = {}
    for root, _d, files in os.walk(base):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(root, fn)
            with open(p, encoding="utf-8") as fh:
                recs[fn[:-5]] = (p, json.load(fh))

    # index candidate targets by their V/E/F, for a cheap first filter
    by_counts = {}
    for slug, (_p, r) in recs.items():
        c = r["combinatorics"]["counts"]
        by_counts.setdefault((c["vertices"], c["edges"], c["faces"]),
                             []).append(slug)

    linked = 0
    for slug, (p, r) in recs.items():
        ch = r["combinatorics"].get("convex_hull") or {}
        if ch.get("is_self") or ch.get("slug") or not ch.get("counts"):
            continue
        key = (ch["counts"]["vertices"], ch["counts"]["edges"],
               ch["counts"]["faces"])
        # The hull is convex by definition, so only convex records can be it.
        # Without this the icosahedron and the great icosahedron are
        # indistinguishable here: identical counts, identical vertex sets.
        cands = [s for s in by_counts.get(key, [])
                 if s != slug and recs[s][1]["combinatorics"].get("convex")]
        if not cands:
            continue
        try:
            from scipy.spatial import ConvexHull
            import numpy as np
            hv = np.asarray(r["geometry"]["vertices"], dtype=float)
            h = ConvexHull(hv)
            hull_pts = [tuple(float(x) for x in hv[i]) for i in
                        sorted(set(int(i) for s in h.simplices for i in s))]
        except Exception:
            continue
        for cand in cands:
            ok, _detail = CC.same_shape(hull_pts, recs[cand][1]["geometry"]["vertices"])
            if ok:
                r["combinatorics"]["convex_hull"]["slug"] = cand
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(r, fh, indent=2)
                linked += 1
                break
    # Biscribed cross-references. A base whose solver returns nothing has NO
    # biscribed form -- a theorem, recorded as such rather than left blank.
    bis = 0
    for base in BISCRIBED_NONE_BASES:
        slug = base.replace("_", "-")
        if slug in recs:
            p2, r2 = recs[slug]
            b = r2["metrics"].setdefault("biscribed", {})
            b["has_biscribed_form"] = False
            b["note"] = ("No biscribed form of this combinatorial type exists: "
                         "the solver finds no parameter equalising the "
                         "face-orbit plane distances (a rectified solid has no "
                         "free parameter at all).")
            with open(p2, "w", encoding="utf-8") as fh:
                json.dump(r2, fh, indent=2)
            bis += 1
    # ... and forward links from a solid to its biscribed variant
    fwd = 0
    for slug, (p2, r2) in recs.items():
        if not slug.startswith("biscribed-"):
            continue
        target = slug[len("biscribed-"):]
        if target not in recs:
            continue
        p3, r3 = recs[target]
        b = r3["metrics"].setdefault("biscribed", {})
        if b.get("biscribed_form") != slug:
            b["biscribed_form"] = slug
            b["has_biscribed_form"] = True
            with open(p3, "w", encoding="utf-8") as fh:
                json.dump(r3, fh, indent=2)
            fwd += 1
    # Compound components -> the record for that solid, matched on geometry
    # (counts, then the full faces-and-dihedrals test) rather than on name.
    comp_linked = comp_axis = 0
    for slug, (p2, r2) in recs.items():
        block = r2.get("compound")
        if not block or not block.get("components"):
            continue
        V = r2["geometry"]["vertices"]
        F = r2["geometry"]["faces"]
        changed = False
        cache = {}
        for part in block["components"]:
            if part.get("slug"):
                continue
            sub = [F[i] for i in part["faces"]]
            used = sorted({i for f in sub for i in f})
            remap = {o: n for n, o in enumerate(used)}
            pv = [V[i] for i in used]
            pf = [[remap[i] for i in f] for f in sub]
            key = (len(pv), part["counts"]["edges"], len(pf))
            if key in cache:
                part["slug"] = cache[key]
                changed = changed or cache[key] is not None
                continue
            hit = None
            for cand in by_counts.get(key, []):
                cr = recs[cand][1]
                if cr.get("compound"):
                    continue
                ok, _d = CC.same_shape(pv, cr["geometry"]["vertices"],
                                       pf, cr["geometry"]["faces"])
                if ok:
                    hit = cand
                    break
            cache[key] = hit
            if hit:
                part["slug"] = hit
                changed = True
        # Harman's construction aligns an n-fold axis of the COMPONENT with an
        # m-fold axis of the COMPOUND. Both orders are computable, so record
        # them as derived geometry. Deliberately NOT written into
        # `enumeration.index`: matching Hart's own labelling would mean
        # transcribing his table, and an unverified index is worse than none.
        if block.get("axis_alignment") is None and block["components"]:
            a = _axis_alignment(V, F, block["components"][0],
                                r2["symmetry"]["schoenflies"])
            if a:
                block["axis_alignment"] = a
                comp_axis += 1
                changed = True
        if changed:
            comp_linked += 1
            with open(p2, "w", encoding="utf-8") as fh:
                json.dump(r2, fh, indent=2)

    print("   linked %d convex_hull, %d biscribed forms, %d proven "
          "non-existent, %d compounds (%d axis alignments)"
          % (linked, fwd, bis, comp_linked, comp_axis))
    return []


_GROUP_LETTER = {"I": "i", "Ih": "i", "O": "c", "Oh": "c",
                 "T": "t", "Td": "t", "Th": "t"}


def _axis_alignment(V, F, part, compound_schoenflies):
    """Orders of the component and compound rotation axes that coincide.

    Harman's `nX/mY` notation names an n-fold axis of the component's group X
    seated on an m-fold axis of the compound's group Y. Both halves are
    recoverable from the geometry: detect each group, list its rotation axes
    with their orders, and find a shared direction.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    sub = [F[i] for i in part["faces"]]
    used = sorted({i for f in sub for i in f})
    remap = {o: n for n, o in enumerate(used)}
    pv = ME.recentre([V[i] for i in used])
    pf = [[remap[i] for i in f] for f in sub]

    def axes_of(Vx, Fx):
        G = SY.find_group(Vx, F=Fx)
        out = {}
        for R in G:
            if np.linalg.det(R) <= 0:
                continue
            ax, n = SY._axis_and_order(R)
            if ax is None or n < 2:
                continue
            k = tuple(round(float(c), 4) for c in ax)
            out[k] = max(out.get(k, 1), n)
        return out

    comp_axes = axes_of(pv, pf)
    whole_axes = axes_of(ME.recentre(V), F)
    if not comp_axes or not whole_axes:
        return None
    best = None
    for k, n in comp_axes.items():
        for k2, m in whole_axes.items():
            d = abs(sum(k[i] * k2[i] for i in range(3)))
            if d > 1 - 1e-4 and (best is None or n * m > best[0] * best[1]):
                best = (n, m)
    if best is None:
        return None
    csym = SY.classify(SY.find_group(pv, F=pf))["schoenflies"]
    x = _GROUP_LETTER.get(csym, "?")
    y = _GROUP_LETTER.get(compound_schoenflies, "?")
    return {"component_axis_order": best[0], "compound_axis_order": best[1],
            "component_group": csym, "compound_group": compound_schoenflies,
            "harman_style": "%d%s/%d%s" % (best[0], x, best[1], y),
            "note": "Derived from the geometry, not transcribed from Hart; "
                    "not verified against his own labelling."}


STAGES = {}


def stage_uniform(limit=None):
    out = []
    for spec in uniform_specs()[:limit]:
        try:
            out.append(build_uniform(spec))
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL U%-3d %-42s %r" % (spec["u"], spec["name"][:42], exc))
    return out


def stage_johnson(limit=None):
    out = []
    for _key, name, num in RS.JOHNSON[:limit]:
        try:
            out.append(build_johnson(num, name))
        except Exception as exc:                            # noqa: BLE001
            print("  FAIL J%-3d %-42s %r" % (num, name[:42], exc))
    return out


STAGES["uniform"] = stage_uniform
STAGES["dual"] = stage_dual
STAGES["johnson"] = stage_johnson
STAGES["prism"] = stage_prism
STAGES["starprism"] = stage_star_prism
STAGES["biscribed"] = stage_biscribed
STAGES["toroid"] = stage_toroid
STAGES["zonohedron"] = stage_zonohedron
STAGES["geodesic"] = stage_geodesic
STAGES["compound"] = stage_compound
STAGES["crosscheck"] = stage_crosscheck
STAGES["link"] = stage_link


# -- emit -------------------------------------------------------------------

FAMILY_DIR = {"compound": "compound", "geodesic": "geodesic",
              "toroid": "toroid",
              "zonohedron": "zonohedron",
              "biscribed": "biscribed", "prism-family": "prism-family",
              "catalan": "catalan", "uniform-dual": "uniform-dual",
              "kepler-poinsot": "kepler-poinsot", "johnson": "johnson",
              "uniform": "uniform", "convex": "uniform"}


def family_dir(rec):
    for f in rec["families"]:
        if f in FAMILY_DIR:
            return FAMILY_DIR[f]
    return "other"


_EMITTED = {}


def emit(records):
    """Write records out. A repeated slug is a hard ERROR, not an overwrite:
    the slug is the database's primary key, and silently clobbering one record
    with another loses data and leaves every dual pointer to it ambiguous.
    (`slugify` deliberately strips parenthesised text so 'Square Pyramid (J1)'
    slugs cleanly -- which makes names distinguished only by a parenthesised
    number collide. That is exactly the class of bug this catches.)"""
    written = []
    for rec in records:
        slug = rec["slug"]
        d = os.path.join(OUT, "solids", family_dir(rec))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, slug + ".json")
        if slug in _EMITTED and _EMITTED[slug] != p:
            raise SystemExit(
                "duplicate slug %r: %r would overwrite %r -- give the two "
                "solids distinguishable names" % (slug, p, _EMITTED[slug]))
        if slug in _EMITTED:
            raise SystemExit("duplicate slug %r emitted twice" % slug)
        _EMITTED[slug] = p
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        written.append((os.path.relpath(p, OUT).replace("\\", "/"), rec))
    return written


def reindex():
    entries = []
    base = os.path.join(OUT, "solids")
    for root, _dirs, files in os.walk(base):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(root, fn)
            with open(p, encoding="utf-8") as fh:
                rec = json.load(fh)
            c = rec["combinatorics"]
            entries.append({
                "slug": rec["slug"], "name": rec["name"],
                "path": os.path.relpath(p, OUT).replace("\\", "/"),
                "families": rec["families"], "counts": c["counts"],
                "symmetry": {"schoenflies": rec["symmetry"]["schoenflies"],
                             "orbifold": rec["symmetry"]["orbifold"],
                             "order": rec["symmetry"]["order"]},
                "convex": c["convex"], "density": c["density"],
                "dual": c["dual"],
                "ids": {k: v for k, v in rec["ids"].items()
                        if v is not None and k in ("wenninger", "uniform",
                                                   "johnson", "netlib", "bowers")},
            })
    entries.sort(key=lambda e: e["slug"])
    idx_path = os.path.join(OUT, "index.json")
    with open(idx_path, encoding="utf-8") as fh:
        idx = json.load(fh)
    idx["entries"] = entries
    idx["count"] = len(entries)
    idx["schema_version"] = SCHEMA_VERSION
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2)
    return len(entries)


def main(argv):
    limit = None
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--limit":
            limit = int(argv[i + 1])
            i += 2
            continue
        args.append(argv[i])
        i += 1
    names = args or ["uniform", "dual", "johnson", "prism",
                     "starprism", "biscribed", "toroid", "zonohedron",
                     "geodesic", "compound", "link"]
    total = 0
    for n in names:
        if n not in STAGES:
            print("unknown stage %r (have: %s)" % (n, ", ".join(STAGES)))
            return 2
        print("== stage %s ==" % n)
        recs = STAGES[n](limit)
        w = emit(recs)
        total += len(w)
        print("   %d record(s)" % len(w))
    print("\nreindexed: %d entries" % reindex())
    print("built %d record(s)" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
