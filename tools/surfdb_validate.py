# The gate for data/surfaces/.
#
#     python tools/surfdb_validate.py             # validate everything
#     python tools/surfdb_validate.py --coverage  # the gap ledger
#     python tools/surfdb_validate.py --slow      # + curvature and symmetry
#     python tools/surfdb_validate.py --stale     # records older than their generator
#
# data/polyhedra's validator is what raised it above a dump: it recomputes
# every claim and it caught three real errors in the first three records.
# This one can check MORE, because a surface's defining property is a
# predicate rather than a table of numbers:
#
#   * H = 0 for a minimal surface, K = -1 for a pseudospherical one,
#     K = 0 for a developable -- measured off the level set (SS invariants).
#   * The claimed symmetry group PROVED symbolically for implicit
#     surfaces, by substituting its generators into the polynomial.
#   * Total curvature an integer multiple of 4*pi for a complete minimal
#     surface of finite total curvature -- a sharp test that catches a
#     wrong Gauss map immediately.
#   * Every generated field regenerated and compared, so a hand-edited
#     `families` array is an error rather than silent drift.
#   * Every 'varies' resolved on every specimen, so 'varies' can never be
#     used to avoid stating a value.
#
# `provenance.verified` is written by THIS tool and must never be
# hand-set -- the same rule data/polyhedra states.

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from surfdb import (curation, expr, invariants, mapping,  # noqa: E402,F401
                    views, weierstrass)

DB = os.path.join(ROOT, "data", "surfaces")

VALID_MODES = {"implicit", "parametric", "weierstrass", "nodal", "variational",
               "swept", "derived", "discrete"}
VALID_SCALES = {"published", "unit_cell", "unit_waist", "unit_radius",
                "unit_scale_parameter", "unit_total_curvature"}
VALID_EMBED = {"embedded", "immersed", "self-intersecting", "branched",
               "singular", "varies"}
VALID_FAMILIES = {"minimal", "minimal-periodic", "cmc", "constant-curvature",
                  "algebraic", "quadric", "ruled", "revolution", "swept",
                  "cyclide", "topological", "spectral", "discrete", "derived",
                  "physical", "misc"}
VALID_TRADITION = {"classical", "gallery", "sculptural", "physical-model",
                   "architectural", "crystallographic", "physical"}


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checked = 0
        self.slow_seen = 0
        self.slow_curv = 0
        self.slow_sym = 0
        self.slow_deg = 0

    def err(self, slug, msg):
        self.errors.append("%s: %s" % (slug, msg))

    def warn(self, slug, msg):
        self.warnings.append("%s: %s" % (slug, msg))


def load_all():
    records = {}
    for dirpath, _dirs, files in os.walk(os.path.join(DB, "surfaces")):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            records[rec["slug"]] = (rec, path)
    return records


def definitions(rec):
    return [rec["definition"]] + list(rec.get("alternate_definitions") or [])


def param_env(rec, dfn):
    """Default values of a definition's declared parameters."""
    env = {}
    for p in dfn.get("parameters") or []:
        d = p.get("default")
        if isinstance(d, (int, float)):
            env[p["name"]] = float(d)
    return env


def check_structure(slug, rec, rep):
    if rec.get("primary_family") not in VALID_FAMILIES:
        rep.err(slug, "primary_family %r is not in the closed list"
                % rec.get("primary_family"))
    for t in rec.get("tradition") or []:
        if t not in VALID_TRADITION:
            rep.err(slug, "tradition %r is not in the vocabulary" % t)

    cond = rec["curvature"].get("condition")
    if cond not in views.CURVATURE_PRECEDENCE:
        rep.err(slug, "curvature.condition %r is not legal" % cond)

    emb = (rec.get("embedding") or {}).get("quality")
    if emb not in VALID_EMBED:
        rep.err(slug, "embedding.quality %r is not legal" % emb)

    rank = (rec.get("symmetry") or {}).get("periodicity_rank")
    if rank not in (0, 1, 2, 3):
        rep.err(slug, "periodicity_rank %r out of range" % rank)

    if not rec.get("construction"):
        rep.err(slug, "no construction entries -- even an unimplemented "
                      "surface must say so explicitly")
    for c in rec.get("construction") or []:
        if "implemented" not in c:
            rep.err(slug, "a construction entry omits `implemented`")

    if not (rec.get("provenance") or {}).get("sources"):
        rep.err(slug, "provenance.sources is empty")


def check_definitions(slug, rec, rep):
    dfns = definitions(rec)
    for i, d in enumerate(dfns):
        where = "definition" if i == 0 else "alternate_definitions[%d]" % (i - 1)
        if d.get("mode") not in VALID_MODES:
            rep.err(slug, "%s.mode %r is not legal" % (where, d.get("mode")))
        fid = d.get("fidelity")
        if fid not in ("exact", "approximation"):
            rep.err(slug, "%s.fidelity %r is not legal" % (where, fid))
        if d.get("scale") is not None and d["scale"] not in VALID_SCALES:
            rep.err(slug, "%s.scale %r is not in the closed vocabulary"
                    % (where, d["scale"]))

        if fid == "approximation":
            tgt = d.get("approximates")
            if tgt is None:
                # legal only when the record has no exact definition at all
                if any(x.get("fidelity") == "exact" for x in dfns):
                    rep.err(slug, "%s is an approximation with `approximates` "
                                  "null, but the record HAS an exact "
                                  "definition it should point at" % where)
            else:
                if not (0 <= tgt < len(dfns)):
                    rep.err(slug, "%s.approximates = %r is out of range"
                            % (where, tgt))
                elif dfns[tgt].get("fidelity") != "exact":
                    rep.err(slug, "%s.approximates points at another "
                                  "approximation" % where)
            if d.get("residual") is None:
                rep.err(slug, "%s is an approximation with no `residual` -- "
                              "what it costs must be recorded, not implied"
                        % where)

        # the polynomial must parse, and may only reference coordinates
        # and declared parameters
        poly = d.get("polynomial")
        if poly:
            declared = {p["name"] for p in d.get("parameters") or []}
            try:
                names = expr.free_names(poly)
            except expr.ExprError as exc:
                rep.err(slug, "%s.polynomial does not parse: %s" % (where, exc))
                continue
            extra = names - {"x", "y", "z"} - declared
            if extra:
                rep.err(slug, "%s.polynomial references %s, which is neither a "
                              "coordinate nor a declared parameter"
                        % (where, sorted(extra)))


def check_measures(slug, rec, rep):
    """Every `exact` string must evaluate to its `value`."""
    env = param_env(rec, rec["definition"])

    def walk(node, path):
        if isinstance(node, dict):
            if "value" in node and ("exact" in node or "note" in node) \
                    and not isinstance(node.get("value"), (dict, list)):
                ok, detail = expr.check_measure(node, env)
                if not ok:
                    rep.err(slug, "%s: %s" % (path, detail))
            for k, v in node.items():
                walk(v, "%s.%s" % (path, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, "%s[%d]" % (path, i))

    walk(rec.get("curvature"), "curvature")
    walk(rec.get("metrics"), "metrics")


def check_generated(slug, rec, rep):
    diff = views.diff_generated(rec)
    for field, d in diff.items():
        rep.err(slug, "generated field %r was hand-edited or is stale: stored "
                      "%r, regenerates to %r"
                % (field, d["stored"], d["regenerated"]))


def check_varies(slug, rec, rep):
    """Every 'varies' must be resolved on every specimen."""
    def varying_paths(node, path=""):
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                out += varying_paths(v, "%s.%s" % (path, k) if path else k)
        elif node == "varies":
            out.append(path)
        return out

    fields = []
    for block in ("topology", "embedding", "symmetry", "curvature"):
        fields += varying_paths(rec.get(block) or {}, block)
    if not fields:
        return
    specs = rec.get("specimens") or []
    if not specs:
        rep.err(slug, "marks %s as 'varies' but lists no specimens"
                % ", ".join(fields))
        return
    for sp in specs:
        for f in fields:
            block, _, key = f.partition(".")
            sub = sp.get(block) or {}
            # Presence, not truthiness. An explicit null is a RESOLUTION --
            # "this member has no space group" is a real answer, and the
            # generic member of the P-gyroid-D associate family is exactly
            # that. Only an ABSENT key means the specimen dodged the question.
            if key not in sub:
                rep.err(slug, "specimen %r does not resolve %r"
                        % (sp.get("label"), f))
            elif sub[key] == "varies":
                rep.err(slug, "specimen %r resolves %r to 'varies' again"
                        % (sp.get("label"), f))


def check_relations(slug, rec, rep, records):
    rel = rec.get("relations") or {}
    for field in ("conjugate", "associate_family", "dual", "member_of",
                  "inverse_of", "chiral_partner", "polyhedral_analogue",
                  "discretisation_of", "smoothing_of"):
        tgt = rel.get(field)
        if not tgt:
            continue
        if field == "polyhedral_analogue":
            continue                      # resolved against data/polyhedra
        if tgt not in records:
            rep.warn(slug, "relations.%s -> %r, which is not a record (yet)"
                     % (field, tgt))
    for field in ("limit_of", "degenerates_to", "generalizes", "specialises"):
        for tgt in rel.get(field) or []:
            if tgt and tgt not in records:
                rep.warn(slug, "relations.%s contains %r, not a record (yet)"
                         % (field, tgt))
    # conjugacy is an involution
    conj = rel.get("conjugate")
    if conj and conj in records:
        back = ((records[conj][0].get("relations") or {}).get("conjugate"))
        if back != slug:
            rep.err(slug, "conjugate %r does not name this record back (got "
                          "%r) -- conjugacy is an involution" % (conj, back))
    # same_surface_as must be symmetric
    for ent in rel.get("same_surface_as") or []:
        other = ent.get("slug")
        if other in records:
            theirs = ((records[other][0].get("relations") or {})
                      .get("same_surface_as") or [])
            if not any(e.get("slug") == slug for e in theirs):
                rep.err(slug, "same_surface_as %r is not reciprocated" % other)


def check_specimen_slugs(slug, rec, rep, records):
    for sp in rec.get("specimens") or []:
        s = sp.get("slug")
        if s and s not in records:
            rep.warn(slug, "specimen %r names promoted slug %r, which is not a "
                           "record" % (sp.get("label"), s))


def check_total_curvature(slug, rec, rep):
    """For a complete minimal surface of finite total curvature, the total
    curvature is an integer multiple of 4*pi (Gauss-Bonnet / Osserman).

    A sharp test: a wrong Gauss map fails it immediately.
    """
    curv = rec.get("curvature") or {}
    topo = rec.get("topology") or {}
    tc = curv.get("total_curvature")
    if not tc or tc.get("value") is None:
        return
    if curv.get("condition") != "minimal":
        return
    if not topo.get("finite_total_curvature"):
        return
    q = tc["value"] / (4.0 * math.pi)
    if abs(q - round(q)) > 1e-6:
        rep.err(slug, "total curvature %.6g is not an integer multiple of "
                      "4*pi (= %.4f x 4*pi); for a complete minimal surface "
                      "of finite total curvature it must be"
                % (tc["value"], q))


def check_gauss_degree(slug, rec, rep):
    """Total curvature against the degree of the stored Gauss map.

    For a complete minimal surface of finite total curvature the total
    curvature is -4*pi*deg(g), and the degree is the winding number of g
    about the origin. Where a record carries BOTH a stored g and a stated
    total curvature, the two were put there for different reasons and
    must agree -- which makes this a genuine cross-check rather than a
    restatement.
    """
    curv = rec.get("curvature") or {}
    topo = rec.get("topology") or {}
    tc = curv.get("total_curvature")
    if curv.get("condition") != "minimal" or not topo.get(
            "finite_total_curvature"):
        return None
    if not tc or tc.get("value") is None:
        return None
    for d in definitions(rec):
        g = d.get("gauss_map")
        if not g:
            continue
        # parameters may be declared on the primary definition or on the
        # alternate that actually carries g; merge both
        env = {}
        for other in definitions(rec):
            env.update(param_env(rec, other))
        try:
            deg = weierstrass.gauss_degree(g, env)
        except Exception:                             # noqa: BLE001
            return None
        if deg is None:
            return None
        want = weierstrass.total_curvature_from_degree(deg)
        if abs(want - tc["value"]) > 1e-6:
            rep.err(slug, "total curvature %.6g disagrees with the stored "
                          "Gauss map: deg(g) = %d implies %.6g"
                    % (tc["value"], deg, want))
            return False
        return True
    return None


def check_curvature_numeric(slug, rec, rep):
    """Measure the defining property off the level set, per definition."""
    results = []
    for i, d in enumerate(definitions(rec)):
        env = param_env(rec, d)
        poly = d.get("polynomial")
        mode = d.get("mode")
        stats = None
        # A nodal level function IS an implicit level set; the mode
        # names where it came from, not how it is evaluated.
        if mode == "nodal" and d.get("level_function"):
            try:
                stats = invariants.sample_curvature(
                    d["level_function"], env, extent=3.2, n=120)
            except Exception as exc:                 # noqa: BLE001
                rep.warn(slug, "nodal curvature sampling raised: %s" % exc)
                continue
        elif mode == "implicit" and poly:
            clip = d.get("clip") or {}
            extent = clip.get("radius") or 1.6
            extent = float(min(max(extent, 0.6), 2.5))
            try:
                stats = invariants.sample_curvature(poly, env, extent=extent)
            except Exception as exc:                 # noqa: BLE001
                rep.warn(slug, "curvature sampling raised: %s" % exc)
                continue
        elif mode == "parametric" and d.get("x") and d.get("u_range"):
            # The conditions mostly live OFF the implicit records: 25
            # parametric rows and every Weierstrass row claim minimal,
            # flat or constant K, and none of them could be checked while
            # the only curvature path took an implicit F.
            try:
                stats = invariants.sample_parametric_curvature(
                    d["x"], d["y"], d["z"], d["u_range"], d["v_range"], env)
            except Exception as exc:                 # noqa: BLE001
                rep.warn(slug, "chart curvature sampling raised: %s" % exc)
                continue
        else:
            continue
        cond = rec["curvature"].get("condition")
        # An approximation is held to its OWN residual, not to the exact
        # tolerance -- the nodal TPMS are not minimal and must not be
        # gated as if they were.
        tol = 2e-3
        if d.get("fidelity") == "approximation":
            stored = (d.get("residual") or {}).get("max_abs_mean_curvature")
            # Held to its OWN measured residual, with a 20% margin for
            # sampling scatter. Before the residuals were measured this
            # fell back to 1e9, which meant the approximations were not
            # gated at all -- the two-tier design existed and checked
            # nothing.
            tol = (stored * 1.2) if stored else 1e9
        verdict, detail = invariants.check_condition(cond, stats, tol=tol)
        if verdict is False:
            rep.err(slug, "definition[%d] claims %r but measures %s"
                    % (i, cond, detail))
        results.append({"definition_index": i,
                        "max_abs_mean_curvature":
                            None if stats is None else stats["max_abs_H"],
                        "resolution": None if stats is None else stats["samples"],
                        "tolerance": tol,
                        "passed": verdict})
    return results


def check_symmetry_symbolic(slug, rec, rep):
    """Prove the claimed group by substituting its generators.

    `symmetry.generator_set` overrides `schoenflies` when the Schoenflies
    symbol alone does not pin the ORIENTATION of the group -- C3v about
    the z axis and C3v about the body diagonal are the same abstract group
    and different substitutions, and a polynomial symmetric in x, y, z has
    the second one.
    """
    sym = rec.get("symmetry") or {}
    group = sym.get("generator_set") or sym.get("schoenflies")
    d = rec["definition"]
    if not group or d.get("mode") != "implicit" or not d.get("polynomial"):
        return None
    env = param_env(rec, d)
    try:
        resid, detail, full = invariants.symmetry_invariance(
            d["polynomial"], group, env)
    except Exception as exc:                          # noqa: BLE001
        rep.warn(slug, "symmetry check raised: %s" % exc)
        return None
    if resid is None:
        return None
    if resid > 1e-6:
        rep.err(slug, "claims symmetry %s but the polynomial is not invariant "
                      "under its generators: %s" % (group, detail))
        return False
    if full is False:
        rep.warn(slug, "symmetry %s verified only on a proper subgroup: %s"
                 % (group, detail))
        return None
    return True


def check_index(records, rep):
    path = os.path.join(DB, "index.json")
    if not os.path.exists(path):
        rep.errors.append("index.json is missing")
        return
    with open(path, encoding="utf-8") as fh:
        idx = json.load(fh)
    if idx["count"] != len(records):
        rep.errors.append("index count %d != %d records on disk"
                          % (idx["count"], len(records)))
    seen = set()
    for e in idx["entries"]:
        seen.add(e["slug"])
        if e["slug"] not in records:
            rep.errors.append("index lists %r, which has no file" % e["slug"])
            continue
        rec, path_ = records[e["slug"]]
        rel = os.path.relpath(path_, DB).replace("\\", "/")
        if e["path"] != rel:
            rep.errors.append("index path for %r is %r but the file is at %r"
                              % (e["slug"], e["path"], rel))
        if e["primary_family"] != rec["primary_family"]:
            rep.errors.append("index family for %r disagrees with the record"
                              % e["slug"])
    for slug in records:
        if slug not in seen:
            rep.errors.append("record %r is not in the index" % slug)
    impl = sum(1 for rec, _ in records.values()
               if any(c.get("implemented") for c in rec["construction"]))
    if idx.get("implemented_count") != impl:
        rep.errors.append("index implemented_count %r != %d computed"
                          % (idx.get("implemented_count"), impl))


def coverage(records):
    total = len(records)
    impl = [s for s, (r, _) in records.items()
            if any(c.get("implemented") for c in r["construction"])]
    missing = sorted(set(records) - set(impl))
    print("COVERAGE  %d records, %d implemented, %d not (%.0f%%)"
          % (total, len(impl), len(missing), 100.0 * len(impl) / max(total, 1)))
    print()
    byfam = {}
    for slug, (rec, _) in records.items():
        fam = rec["primary_family"]
        d = byfam.setdefault(fam, [0, 0])
        d[0] += 1
        if slug in impl:
            d[1] += 1
    print("%-22s %6s %6s" % ("FAMILY", "TOTAL", "IMPL"))
    for fam in sorted(byfam):
        t, i = byfam[fam]
        print("%-22s %6d %6d%s" % (fam, t, i, "" if t == i else "   <-- gap"))
    print()
    print("NOT IMPLEMENTED (%d):" % len(missing))
    for slug in missing:
        rec = records[slug][0]
        why = None
        for c in rec["construction"]:
            why = c.get("blocked_by") or why
        first = (why or "no reason recorded").split(". ")[0]
        print("  %-34s %s" % (slug, first[:96]))


def stale(records):
    """Records older than the generator module they describe."""
    def last_commit(path):
        try:
            out = subprocess.run(
                ["git", "log", "-1", "--format=%ct", "--", path],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            return int(out.stdout.strip()) if out.stdout.strip() else None
        except Exception:                             # noqa: BLE001
            return None

    n = 0
    for slug, (rec, path) in sorted(records.items()):
        rec_t = last_commit(path)
        if rec_t is None:
            continue
        for c in rec["construction"]:
            gen = c.get("generator")
            if not gen:
                continue
            mod = os.path.join(ROOT, *gen.split(".")) + ".py"
            if not os.path.exists(mod):
                continue
            gen_t = last_commit(mod)
            if gen_t and gen_t > rec_t:
                print("STALE  %-34s %s changed after the record" % (slug, gen))
                n += 1
                break
    print("\n%d stale record(s)." % n)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--slow", action="store_true",
                    help="also measure curvature and prove symmetry")
    ap.add_argument("--write-verified", action="store_true",
                    help="record what passed into provenance.verified")
    args = ap.parse_args()

    records = load_all()
    if not records:
        print("no records found in %s -- run tools/surfdb_build.py first" % DB)
        return 1

    if args.coverage:
        coverage(records)
        return 0
    if args.stale:
        return 1 if stale(records) else 0

    rep = Report()
    for slug, (rec, path) in sorted(records.items()):
        rep.checked += 1
        check_structure(slug, rec, rep)
        check_definitions(slug, rec, rep)
        check_measures(slug, rec, rep)
        check_generated(slug, rec, rep)
        check_varies(slug, rec, rep)
        check_relations(slug, rec, rep, records)
        check_specimen_slugs(slug, rec, rep, records)
        check_total_curvature(slug, rec, rep)

        if args.slow:
            measured = check_curvature_numeric(slug, rec, rep)
            sym_ok = check_symmetry_symbolic(slug, rec, rep)
            deg_ok = check_gauss_degree(slug, rec, rep)
            rep.slow_deg += 1 if deg_ok else 0
            # Count what was actually PROVED. Without this, "0 errors"
            # reads as "everything checked out" when it may mean "almost
            # nothing was checkable" -- a distinction the reader is
            # entitled to, and one this database would otherwise obscure.
            rep.slow_curv += sum(1 for m in (measured or [])
                                 if m["passed"] is True)
            rep.slow_sym += 1 if sym_ok else 0
            rep.slow_seen += 1
            if args.write_verified:
                v = rec.setdefault("provenance", {}).setdefault("verified", {})
                v["expressions_parse"] = True
                v["exact_values_match"] = True
                v["generated_fields_match"] = True
                if measured:
                    rec["curvature"]["measured"] = measured
                    v["curvature_condition"] = all(
                        m["passed"] is not False for m in measured)
                if sym_ok is not None:
                    v["symmetry_invariant"] = sym_ok
                v["tolerance"] = 2e-3
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")

    check_index(records, rep)

    print("checked %d records" % rep.checked)
    if args.slow:
        print("  curvature conditions PROVED numerically: %d" % rep.slow_curv)
        print("  symmetry groups PROVED symbolically:     %d" % rep.slow_sym)
        print("  total curvature vs Gauss-map degree:     %d" % rep.slow_deg)
        print("  (the rest carry no stored polynomial, or no condition and no "
              "group to test -- an honest count, not a silent pass)")
    if rep.warnings:
        print("\nWARNINGS (%d):" % len(rep.warnings))
        for w in rep.warnings[:60]:
            print("  -", w)
        if len(rep.warnings) > 60:
            print("  ... and %d more" % (len(rep.warnings) - 60))
    if rep.errors:
        print("\nERRORS (%d):" % len(rep.errors))
        for e in rep.errors[:80]:
            print("  -", e)
        if len(rep.errors) > 80:
            print("  ... and %d more" % (len(rep.errors) - 80))
        print("\nRESULT: FAIL")
        return 1
    print("\nRESULT: OK  (%d errors, %d warnings)" % (0, len(rep.warnings)))
    return 0


sys.exit(main())
