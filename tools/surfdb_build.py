# Build data/surfaces/ from the generator registries plus curated facts.
#
#     python tools/surfdb_build.py                 # every stage
#     python tools/surfdb_build.py algebraic tpms  # selected stages
#     python tools/surfdb_build.py --mapping-report # what happened to each row
#
# THE STRATEGY, in one line: derive what the code knows, curate what the
# literature knows.
#
# The registries in math_art/ carry labels, family membership, enum keys,
# clip regions and which operator reaches a surface.  They do NOT carry
# who found it, when, its symmetry group, its genus or its ends -- that is
# tools/surfdb/curation.py.  The build joins the two and emits records.
#
# What it deliberately does NOT do is emit one record per registry row.
# Rows are per-OPERATOR; records are per-SURFACE, and the difference is
# large enough to need its own declared table (tools/surfdb/mapping.py):
# Catalan's surface and the Bjorling cycloid row are one surface; the
# Whitney umbrella ships from two operators in two definition modes; the
# three Dupin cyclide rows are three REGIMES of one formula.  Every row's
# disposition is auditable with --mapping-report.
#
# Records are DERIVED, never transcribed.  Mathcurve, the Virtual Math
# Museum and Hauser's gallery are oracles: cited, not copied.  And every
# candidate polynomial in curation.py is checked numerically against the
# shipped implementation before it reaches a record -- a mistyped
# coefficient does not error, it silently produces a different surface, so
# nothing is trusted that has not been reproduced independently.

import argparse
import copy
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# The engine packages are bpy-free and import flat, as data/polyhedra's
# HANDOFF warns: the math_art package __init__ imports bpy and fails
# outside Blender.
sys.path.insert(0, os.path.join(ROOT, "math_art"))

from surfdb import (algsurf, charts, curation, ferreol,  # noqa: E402
                    invariants, mapping, nodal, papers, polynomial,
                    published, references, registry, sources, tail,
                    views, vmm, wedata, weextract)

try:
    from surfdb import parcharts                          # noqa: E402
except ImportError:                                       # module not written yet
    parcharts = None

OUT = os.path.join(ROOT, "data", "surfaces")
SCHEMA_VERSION = "0.1.0"
DATABASE_VERSION = "0.1.0"

# Rows that are UI modes, colouring options or output styles rather than
# surfaces.  Excluded explicitly so the count means something.
NOT_A_SURFACE = {
    "delaunay": {"SURFACE", "ROULETTE", "CUSTOM"},
    "supershape": {"NONE", "LATITUDE", "DISTANCE", "CUSTOM"},
    "ruled": {"KNOT_SPAN"},
    "constwidth": set(),
}

# Flat generator modules, read as source (they import bpy).
FLAT_SOURCES = [
    # (source name, module, top-level tables, scan inline enums, stage,
    #  operator id, default primary family)
    ("topological", "topological_surface_generator", ("PRESET_ITEMS",), False,
     "topological", "mesh.topological_surface_add", "topological"),
    ("ruled", "ruled_surface_generator",
     ("_NAMED_LABELS", "_CONOID_KINDS", "_MODES"), False,
     "ruled", "mesh.ruled_surface_add", "ruled"),
    ("hyperbolic", "hyperbolic_surface_generator", ("PRESETS",), False,
     "constant-curvature", "mesh.hyperbolic_surface_add", "constant-curvature"),
    ("spherical", "spherical_surface_generator", ("PRESETS",), False,
     "constant-curvature", "mesh.spherical_surface_add", "constant-curvature"),
    ("helical", "helical_surface_generator", ("_SURFACES",), False,
     "swept", "mesh.helical_surface_add", "swept"),
    ("curiosity", "curiosity_surface_generator", (), True,
     "misc", "mesh.curiosity_surface_add", "misc"),
    ("delaunay", "delaunay_generator", (), True,
     "cmc", "mesh.delaunay_surface_add", "cmc"),
    ("steinmetz", "steinmetz_generator", (), True,
     "derived", "mesh.steinmetz_add", "derived"),
    ("constwidth", "constant_width_generator", (), True,
     "misc", "mesh.constant_width_add", "misc"),
]

# Where a derived row's primary family differs from its source default.
FAMILY_OVERRIDE = {
    "curiosity:CYCLIDE_RING": "cyclide",
    "curiosity:CYCLIDE_HORN": "cyclide",
    "curiosity:CYCLIDE_SPINDLE": "cyclide",
    "curiosity:GABRIEL": "revolution",
    "curiosity:ZOLL": "revolution",
    "curiosity:TANNERY_PEAR": "revolution",
    "curiosity:TANNERY_HOURGLASS": "revolution",
    "curiosity:NEILOID": "revolution",
    "curiosity:BOUGUER": "revolution",
    "curiosity:PENDANT_DROP": "revolution",
    "curiosity:SCHWARZ_LANTERN": "discrete",
    "ruled:HYPERBOLOID": "quadric",
    "ruled:HYPAR": "quadric",
    # The Darboux cyclide ships as a quartic level set, so it arrives
    # from the algebraic registry -- but "cyclide" is the classification
    # that says something, and it is where the curated record and the
    # Dupin cyclide it generalises already live. Without this the record
    # changes family directory when it gets built, which is also how it
    # acquires a ghost: the writer creates the new path and nothing
    # deletes the old one, so `cyclide/darboux-cyclide.json` would sit
    # beside `algebraic/darboux-cyclide.json` still claiming the surface
    # is unbuilt.
    "algebraic:DARBOUX_CYCLIDE": "cyclide",
}

_slug_cache = {}


def slugify(text):
    """Kebab-case slug, KEEPING parenthesised text.

    data/polyhedra's slugify strips parentheses, and it is right to: its
    parentheticals are catalogue numbers ("Square Pyramid (J1)"), so
    dropping them gives the clean name and the J-number lives in `ids`.

    For surfaces the same rule is actively wrong. Here the parenthetical
    IS the mathematics -- "Chen-Gackstatter" and "Chen-Gackstatter (higher
    genus)" are different surfaces, as are "Six-Ended Scherk Tower" and
    "... (genus 1)", and "Antiprismatic k-noid (nn=5)" and "... (full
    family)". Stripping collapsed nine distinct records onto five slugs,
    and because the later row silently overwrote the earlier one, Wolf
    Barth's 65-nodal sextic ended up carrying a Goursat family member's
    coefficients while still looking entirely plausible.
    """
    import re
    s = text.replace("'", "").replace("’", "")
    # Comparison operators carry meaning and must not collapse: the Goursat
    # tetrahedral family is indexed by them, and "(k < 0)", "(k = 0)" and
    # "(k > 4)" are three different surfaces that all reduce to "k-0" if the
    # operator is treated as punctuation.
    s = s.replace("<=", " le ").replace(">=", " ge ")
    s = s.replace("<", " lt ").replace(">", " gt ").replace("=", " eq ")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "unnamed"


def deep_merge(base, extra):
    """Merge `extra` into `base`, recursively. `extra` wins on scalars."""
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def blank_record(slug, name, primary_family):
    return {
        "schema_version": SCHEMA_VERSION,
        "slug": slug,
        "name": name,
        "primary_family": primary_family,
        "families": [],
        "definition": {"mode": "parametric", "fidelity": "exact"},
        "curvature": {"condition": "none"},
        "topology": {},
        "symmetry": {"kind": "none", "periodicity_rank": 0},
        "embedding": {"quality": "immersed"},
        "construction": [],
        "provenance": {
            "definition": "Derived from the Math Art generator registries and "
                          "cross-referenced to the literature; not transcribed "
                          "from any gallery's compilation.",
            "sources": [],
        },
    }


class Builder:
    def __init__(self):
        self.records = {}        # slug -> record
        self.families = {}       # family slug -> record (for specimen attach)
        self.report = []         # (source, key, disposition, target)
        self.problems = []
        self.claimed = {}        # slug -> the "source:key" that owns it

    # -- record plumbing -------------------------------------------------

    def get(self, slug, name=None, family="misc"):
        if slug not in self.records:
            self.records[slug] = blank_record(slug, name or slug, family)
        return self.records[slug]

    def emit(self, slug, name, family, **kw):
        if slug in self.records and kw.get("strict", True):
            # polydb's lesson: raise rather than overwrite. Two surfaces
            # sharing a slug is a mapping bug, not something to paper over.
            self.problems.append("slug collision: %r emitted twice" % slug)
        rec = self.get(slug, name, family)
        rec["name"] = name
        rec["primary_family"] = family
        return rec

    def add_construction(self, rec, entry):
        for existing in rec["construction"]:
            if (existing.get("operator_id") == entry.get("operator_id")
                    and existing.get("key") == entry.get("key")):
                return
        rec["construction"].append(entry)

    def add_specimen(self, family_slug, label, name=None, family="misc"):
        rec = self.get(family_slug, name or family_slug.replace("-", " ").title(),
                       family)
        specs = rec.setdefault("specimens", [])
        if not any(s.get("label") == label for s in specs):
            specs.append({"label": label})
        return rec

    # -- citations -------------------------------------------------------

    def cite(self, rec, module_rel, line_no, label):
        """Attach the vetted References block nearest to a row.

        The contributor notes already require every generator module to credit
        the
        mathematics it implements, and those blocks are kept current by
        whoever writes the generator.  Reading them beats retyping them:
        a retyped citation goes subtly wrong and then looks authoritative.
        The NEAREST block is used, because weierstrass.py carries 15 and
        zoo.py 8 -- one per family section.
        """
        path = os.path.join(ROOT, "math_art", *module_rel.split(".")) + ".py"
        got = references.for_row(path, line_no or 1, label)
        if not got:
            return
        srcs = rec["provenance"].setdefault("sources", [])
        for r in got:
            if r not in srcs:
                srcs.append(r)

    # -- dispositions ----------------------------------------------------

    def place(self, source, key, label, family, construction):
        """Route one registry row through the mapping table."""
        kind, payload = mapping.disposition(source, key)
        ref = "%s:%s" % (source, key)
        fam = FAMILY_OVERRIDE.get(ref, family)

        if kind == "merge":
            rec = self.get(payload, label, fam)
            self.add_construction(rec, construction)
            self.report.append((source, key, "merge", payload))
            return payload

        if kind == "specimen":
            fam_slug, spec_label = payload
            rec = self.add_specimen(fam_slug, spec_label, family=fam)
            self.add_construction(rec, dict(construction, preset=key))
            self.report.append((source, key, "specimen", fam_slug))
            return fam_slug

        if kind == "promote":
            slug, family_slug = payload
            rec = self.get(slug, label, fam)
            rec["name"] = label
            rec.setdefault("relations", {})["member_of"] = family_slug
            self.add_construction(rec, construction)
            self.report.append((source, key, "promote", slug))
            return slug

        slug = payload or slugify(label)

        # SLUG COLLISION. Two different rows must never land on one slug.
        # This is not hypothetical: the Goursat dodecahedral row
        # DODEC_BARTH is labelled after Barth and slugifies to
        # "barth-sextic", the same slug the CLASSICAL row BARTH is aliased
        # to -- and because it sorts later it silently OVERWROTE Wolf
        # Barth's 65-nodal sextic with a Goursat family member carrying
        # Goursat's coefficients. The record still looked plausible.
        #
        # data/polyhedra hit the same class of bug ("names differing only
        # by a parenthesised number collide; `emit` now raises rather than
        # overwriting"). Here the row is disambiguated by its registry
        # family and the collision is REPORTED, so it gets a deliberate
        # alias rather than a silent merge.
        if slug in self.claimed and self.claimed[slug] != ref:
            other = self.claimed[slug]
            fam_key = construction.get("family") or source
            slug = "%s-%s" % (slug, slugify(fam_key))
            self.problems.append(
                "slug collision: %s and %s both slugify to the same name; "
                "%s was disambiguated to %r. Add an explicit ALIAS in "
                "tools/surfdb/mapping.py to name it properly."
                % (other, ref, ref, slug))
        self.claimed[slug] = ref

        rec = self.get(slug, label, fam)
        rec["name"] = label
        rec["primary_family"] = fam
        self.add_construction(rec, construction)
        self.report.append((source, key, "emit", slug))
        return slug

    # -- stages ----------------------------------------------------------

    def stage_algebraic(self):
        from surfaces import algebraic as A
        alg_lines = registry.dict_key_lines(
            open(os.path.join(ROOT, "math_art", "surfaces", "algebraic.py"),
                 encoding="utf-8").read(),
            r"^\s*\('([A-Z0-9_]+)',")
        for key, fam_key in sorted(A.SURFACE_FAMILY.items()):
            entry = A.PRESETS.get(key)
            if entry is None:
                continue
            label, fn, clip_shape, clip_radius = entry[0], entry[1], entry[2], entry[3]
            slug = self.place(
                "algebraic", key, label, "algebraic",
                {"generator": "math_art.surfaces.algebraic",
                 "operator_id": "mesh.algebraic_surface_add",
                 "family": fam_key, "key": key,
                 "definition_index": 0, "implemented": True})
            rec = self.records[slug]
            self.cite(rec, "surfaces.algebraic", alg_lines.get(key), label)
            dfn = rec["definition"]
            dfn["mode"] = "implicit"
            dfn["fidelity"] = "exact"
            dfn["exactness"] = "elementary"
            dfn["scale"] = "published"
            if clip_shape:
                dfn["clip"] = {"shape": clip_shape, "radius": clip_radius}
            rec["metrics"] = dict(rec.get("metrics") or {},
                                  normalization="published")

            # the polynomial, VERIFIED against the shipped implementation
            poly, detail = self._polynomial_for(key, slug, fn, clip_radius, A)
            dfn["polynomial"] = poly
            if poly is None:
                dfn["note"] = (
                    "Polynomial not stored: %s. The shipped implementation in "
                    "math_art/surfaces/algebraic.py is authoritative; an "
                    "unverified transcription would silently define a "
                    "different surface." % detail)
            else:
                rec["provenance"]["definition"] = (
                    "Closed form reproduced numerically against the shipped "
                    "implementation (math_art/surfaces/algebraic.py) over "
                    "240 sample points: %s." % detail)

            # parameters the operator exposes
            params = []
            for p in A.PRESET_PARAMS.get(key, ()):  # (attr, label, type, default, lo, hi, desc)
                params.append({
                    "name": p[0], "symbol": p[1],
                    "domain": "[%g, %g]" % (p[4], p[5]),
                    "default": p[3],
                    "integer": p[2] == "INT",
                    "note": p[6] if len(p) > 6 else "",
                })
            if params:
                dfn["parameters"] = params

            # The Goursat families are DEFINED as the polynomials invariant
            # under a regular solid's symmetry group, so the group is a
            # derivable fact, not a curated guess -- and the validator then
            # PROVES it by substituting the generators. If one of these
            # fails it is a real finding about the shipped implementation,
            # which is the point of stating it.
            goursat_group = {"GOURSAT_TET": ("Td", 24, "*332"),
                             "GOURSAT_OCT": ("Oh", 48, "*432"),
                             "GOURSAT_DODEC": ("Ih", 120, "*532")}.get(fam_key)
            if goursat_group:
                sch, order, orb = goursat_group
                rec["symmetry"].update(kind="point", periodicity_rank=0,
                                       schoenflies=sch, order=order,
                                       orbifold=orb, chiral=False)
                # The dodecahedral sextic is built from the six planes whose
                # normals are the five-fold axes, which puts a C5 axis on z
                # rather than 2-fold axes on the coordinate axes. Same group,
                # different frame -- and the symbolic check needs the frame.
                if fam_key == "GOURSAT_DODEC":
                    rec["symmetry"]["generator_set"] = "Ih_z5"

            if fam_key == "HAUSER":
                rec.setdefault("tradition", []).append("gallery")
                rec.setdefault("ids", {})["hauser"] = label
                rec["provenance"].setdefault("sources", []).append(
                    "H. Hauser, 'Bildergalerie algebraischer Flaechen', "
                    "Universitaet Wien.")
            elif fam_key == "RECORD":
                rec.setdefault("tradition", []).append("classical")
            elif fam_key.startswith("GOURSAT"):
                rec.setdefault("tradition", []).append("classical")
                rec["provenance"].setdefault("sources", []).append(
                    "E. Goursat, 'Etude des surfaces qui admettent tous les "
                    "plans de symetrie d'un polyedre regulier', Ann. Sci. ENS "
                    "3e serie 4 (1887).")
            elif fam_key == "NAMED":
                rec.setdefault("tradition", []).append("gallery")
                rec["provenance"].setdefault("sources", []).append(
                    "3DXM Virtual Math Museum, implicit-surfaces gallery.")
            elif fam_key == "MATHCURVE":
                rec.setdefault("tradition", []).append("gallery")
                rec["provenance"].setdefault("sources", []).append(
                    "R. Ferreol, Encyclopedie des formes mathematiques "
                    "remarquables (mathcurve).")
            else:
                rec.setdefault("tradition", []).append("classical")

    @staticmethod
    def _goursat_polynomial(fam, co):
        """The Goursat family form with this row's coefficients baked in.

        Transcribed from math_art/surfaces/goursat.py's four field
        functions, at a = 1 (the size parameter's default).  Verified
        against that same implementation before it reaches a record, so a
        slip here is caught rather than shipped -- which matters more than
        usual for the dodecahedral base, whose sextic core is a
        ten-term expression that is easy to get subtly wrong.
        """
        def num(v):
            return repr(float(v))
        r2 = "(x**2 + y**2 + z**2)"
        if fam == "OCT4":
            k, k1, k2 = co
            return ("x**4 + y**4 + z**4 + %s*%s**2 + %s*%s + %s"
                    % (num(k), r2, num(k1), r2, num(k2)))
        if fam == "TET3":
            k, k1 = co
            return "x*y*z + %s*%s - %s" % (num(k1), r2, num(k))
        if fam == "TET4":
            return ("(x + y + z - 1)*(-x - y + z - 1)*(x - y - z - 1)"
                    "*(-x + y - z - 1) - (x**2 + y**2 + z**2 - 3)**2")
        if fam == "DODEC6":
            k, k1, k2, k3 = co
            base = ("(z**6 - 5*(x**2 + y**2)*z**4 + 5*(x**2 + y**2)**2*z**2"
                    " - 2*(x**4 - 10*x**2*y**2 + 5*y**4)*x*z)")
            return ("%s + %s*%s**3 + %s*%s**2 + %s*%s + %s"
                    % (base, num(k), r2, num(k1), r2, num(k2), r2, num(k3)))
        return None

    def _polynomial_for(self, key, slug, fn, clip_radius, A):
        cand = curation.polynomial_for(slug)
        if cand is None:
            try:
                g = A._goursat()
                if key in g.PRESETS:
                    _label, fam, co, _clip, _desc = g.PRESETS[key]
                    cand = self._goursat_polynomial(fam, co)
            except Exception:                            # noqa: BLE001
                cand = None
        if cand is None and key in A.HAUSER_EQUATION:
            cand = None  # converted below from gallery notation
            raw = A.HAUSER_EQUATION[key]
            sig = inspect.signature(fn)
            extra = {p: sig.parameters[p].default
                     for p in list(sig.parameters)[3:]}
            try:
                oracle = (lambda x, y, z, _f=fn, _e=extra: _f(x, y, z, **_e))
                poly, detail = polynomial.convert_verified(
                    raw, oracle, extent=min(clip_radius or 1.5, 1.5))
                return poly, detail
            except Exception as exc:                    # noqa: BLE001
                return None, "conversion raised: %s" % exc
        if cand is None:
            return None, ("no closed form is stored for this surface; it is "
                          "defined by its shipped implementation")
        sig = inspect.signature(fn)
        extra = {p: sig.parameters[p].default for p in list(sig.parameters)[3:]}
        try:
            oracle = (lambda x, y, z, _f=fn, _e=extra: _f(x, y, z, **_e))
            ok, detail = polynomial.verify_against(
                cand, oracle, extent=min(clip_radius or 1.5, 1.5))
        except Exception as exc:                        # noqa: BLE001
            return None, "verification raised: %s" % exc
        return (cand if ok else None), detail

    def stage_minimal(self):
        import minsurf
        zoo_src = open(os.path.join(ROOT, "math_art", "minsurf", "zoo.py"),
                       encoding="utf-8").read()
        zoo_lines = registry.dict_key_lines(
            zoo_src, r"^(?:    '|WE_SURFACES\[')([A-Z0-9_]+)'")
        par_lines = registry.dict_key_lines(
            open(os.path.join(ROOT, "math_art", "minsurf", "parametric.py"),
                 encoding="utf-8").read(), r"^\s*'([A-Z0-9_]+)':")
        fam_family = {
            "CLASSICAL": "minimal", "BJORLING": "minimal", "SPHERES": "minimal",
            "TORI": "minimal", "HIGHER": "minimal", "NONORIENT": "minimal",
            "SINGLY": "minimal-periodic", "DOUBLY": "minimal-periodic",
        }
        rank = {"SINGLY": 1, "DOUBLY": 2}
        for key, spec in sorted(minsurf.PARAMETRIC.items()):
            label = spec[0] if isinstance(spec, tuple) else str(key)
            fam_key = minsurf.SURFACE_FAMILY.get(key, "CLASSICAL")
            family = fam_family.get(fam_key, "minimal")
            slug = self.place(
                "minsurf", key, label, family,
                {"generator": "math_art.minsurf.parametric",
                 "operator_id": ("mesh.periodic_minimal_add"
                                 if fam_key in ("SINGLY", "DOUBLY")
                                 else "mesh.parametric_minimal_add"),
                 "family": fam_key, "key": key,
                 "definition_index": 0, "implemented": True})
            rec = self.records[slug]
            if key in zoo_lines:
                self.cite(rec, "minsurf.zoo", zoo_lines[key], label)
            else:
                self.cite(rec, "minsurf.parametric", par_lines.get(key), label)
            rec["curvature"]["condition"] = "minimal"
            rec["curvature"].setdefault(
                "mean", {"exact": "0", "value": 0.0, "source": "classical"})
            dfn = rec["definition"]
            if fam_key == "BJORLING":
                dfn["mode"] = "weierstrass"
                dfn["domain"] = "disk"
                dfn["note"] = ("Bjorling's problem: the minimal surface "
                               "through a given curve with a given normal "
                               "along it.")
            elif fam_key == "CLASSICAL":
                dfn["mode"] = "parametric"
            else:
                dfn["mode"] = "weierstrass"
            dfn["fidelity"] = "exact"
            dfn.setdefault("exactness",
                           "elementary" if fam_key == "CLASSICAL"
                           else "numerical-integral")
            sym = rec["symmetry"]
            if fam_key in rank:
                sym["kind"] = "rod" if rank[fam_key] == 1 else "layer"
                sym["periodicity_rank"] = rank[fam_key]
            rec["topology"].setdefault("complete", True)
            rec["topology"].setdefault("compact", False)
            if fam_key == "NONORIENT":
                rec["topology"]["orientable"] = False
                rec["topology"]["one_sided"] = True
            rec.setdefault("tradition", []).append("classical")

    EXACT_TOL = 1e-6

    def _nodal_block(self, key, level_fns, oracle):
        """The nodal definition for a TPMS row, with its fidelity MEASURED.

        Fidelity is not asserted here, it is observed. Every TPMS row was
        previously stamped `approximation` on the assumption that a nodal
        level set approximates its minimal surface -- true for most of
        them, and false for SCHERKT, whose level function
        sin(z) - sinh(x)*sinh(y) is Scherk's second surface EXACTLY and
        measures max|H| = 0. Measuring instead of assuming is also what
        finally gives the two-tier tolerance real numbers: before this,
        every residual was null and the approximations were ungated.
        """
        text = level_fns.get(key)
        block = {
            "mode": "nodal", "fidelity": "approximation",
            "approximates": 0, "scale": "unit_cell", "lattice": "cubic",
            "level": 0.0,
            "residual": {"max_abs_mean_curvature": None,
                         "measured_at_resolution": None,
                         "note": "No level function is stored for this row, "
                                 "so the cost of the approximation has not "
                                 "been measured."},
            "note": "Standard nodal approximation as shipped in "
                    "math_art/minsurf/tpms.py.",
        }
        if not text:
            return block
        ok, detail = nodal.verify(text, oracle)
        if not ok:
            self.problems.append(
                "nodal level function for %s disagrees with the shipped "
                "implementation: %s" % (key, detail))
            return block
        block["level_function"] = text
        st = invariants.sample_curvature(text, extent=3.2, n=120)
        if st is None:
            return block
        h = st["max_abs_H"]
        block["residual"] = {
            "max_abs_mean_curvature": round(h, 6),
            "measured_at_resolution": st["samples"],
            "note": ("Measured: the level set's mean curvature over %d "
                     "on-surface samples. A nodal fit is a truncated Fourier "
                     "series, not the minimal surface." % st["samples"]),
        }
        if h <= self.EXACT_TOL:
            # not an approximation at all
            block["fidelity"] = "exact"
            block["exactness"] = "elementary"
            block.pop("approximates", None)
            block["residual"]["note"] = (
                "Measured max|H| = %.2g, i.e. minimal to numerical precision: "
                "this level function defines the surface EXACTLY and is not "
                "an approximation, despite sitting in the nodal table." % h)
        return block

    def stage_tpms(self):
        import minsurf
        tpms_lines = registry.dict_key_lines(
            open(os.path.join(ROOT, "math_art", "minsurf", "tpms.py"),
                 encoding="utf-8").read(), r"^\s*'([A-Z0-9_]+)':")
        level_fns = nodal.extract(minsurf.TPMS)
        for key in sorted(minsurf.TPMS):
            label = "%s surface" % key.replace("_", " ").title()
            slug = self.place(
                "tpms", key, label, "minimal-periodic",
                {"generator": "math_art.minsurf.tpms",
                 "operator_id": "mesh.periodic_minimal_add",
                 "family": "TPMS", "key": key,
                 "definition_index": None, "implemented": True})
            rec = self.records[slug]
            self.cite(rec, "minsurf.tpms", tpms_lines.get(key), label)
            rec["curvature"]["condition"] = "minimal"
            rec["symmetry"].update(kind="space", periodicity_rank=3)
            rec["topology"].update(complete=True, compact=False,
                                   orientable=True)
            rec["metrics"] = dict(rec.get("metrics") or {},
                                  normalization="unit_cell")
            rec.setdefault("tradition", []).append("crystallographic")
            # The nodal block, with its fidelity MEASURED rather than
            # assumed -- see _nodal_block.
            self._attach_nodal(rec, self._nodal_block(
                key, level_fns, minsurf.TPMS[key][1]))

        for key in sorted(minsurf.TPMS_EXACT):
            label = "%s (exact)" % key
            slug = self.place(
                "tpms_exact", key, label, "minimal-periodic",
                {"generator": "math_art.minsurf.tpms",
                 "operator_id": "mesh.periodic_minimal_add",
                 "family": "TPMS_EXACT", "key": key,
                 "definition_index": 0, "implemented": True})
            rec = self.records[slug]
            self.cite(rec, "minsurf.tpms", tpms_lines.get(key), label)
            rec["curvature"]["condition"] = "minimal"
            rec["symmetry"].update(kind="space", periodicity_rank=3)
            rec["topology"].update(complete=True, compact=False)
            rec["metrics"] = dict(rec.get("metrics") or {},
                                  normalization="unit_cell")
            d = rec["definition"]
            d.update(mode="weierstrass", fidelity="exact",
                     exactness="numerical-integral", scale="unit_cell")
            rec.setdefault("tradition", []).append("crystallographic")

    def _attach_nodal(self, rec, nodal):
        """Put the nodal block in the right slot.

        If the record already has an exact definition (the promoted P /
        gyroid / D do, via TPMS_EXACT), the nodal one becomes an
        ALTERNATE pointing at index 0.  Otherwise it is the only
        definition there is, and `approximates` has nothing to point at --
        which the schema must not be lied to about.
        """
        primary = rec["definition"]
        if primary.get("fidelity") == "exact" and primary.get("mode") != "parametric":
            alts = rec.setdefault("alternate_definitions", [])
            if not any(a.get("mode") == "nodal" for a in alts):
                alts.append(nodal)
        else:
            nodal = dict(nodal)
            if nodal.get("fidelity") == "approximation":
                nodal["approximates"] = None
                nodal["note"] = (
                    nodal["note"] + " No exact definition is stored for this "
                    "surface, so this approximation is all the record has; "
                    "`approximates` is null rather than pointing at itself.")
            rec["definition"] = nodal

    def stage_flat(self, only_stage=None):
        for (src, module, tables, inline, stage, op, family) in FLAT_SOURCES:
            if only_stage and stage != only_stage:
                continue
            rows, path = registry.read_rows(
                ROOT + "/math_art" if False else os.path.join(ROOT, "math_art"),
                module, tables=tables, inline=inline,
                exclude=NOT_A_SURFACE.get(src, set()))
            if not rows:
                self.problems.append(
                    "no rows extracted from %s -- its table shape may have "
                    "changed; the build will not silently emit nothing" % path)
                continue
            for key, label, line in rows:
                slug = self.place(
                    src, key, label, family,
                    {"generator": "math_art.%s" % module,
                     "operator_id": op, "key": key,
                     "definition_index": 0, "implemented": True})
                rec = self.records[slug]
                self.cite(rec, module, line, label)
                if src in ("hyperbolic",):
                    rec["curvature"]["condition"] = "k-const-negative"
                elif src == "spherical":
                    rec["curvature"]["condition"] = "k-const-positive"
                elif src == "delaunay":
                    rec["curvature"].setdefault("condition", "cmc")

    def stage_quadric(self):
        """The degree-2 surfaces.

        Filed apart from `algebraic` because degree 2 is exactly solvable
        and anchors the whole degree axis, and because the literature
        always treats the quadrics as their own chapter.

        Almost none of them are IMPLEMENTED, and saying so is the point.
        An earlier draft of the design marked this stage as shipped; it
        is not.  Only the hyperboloid of one sheet and the hyperbolic
        paraboloid have math_art operators (both as RULED surfaces), and
        the circular cylinder arrives as the Delaunay family's H = 1/2r
        member.  An "ellipsoid" exists only as a superellipsoid special
        case, which is a different family, not an exact quadric.  Blender
        primitives are not math_art constructions and do not count.
        """
        for slug, spec in QUADRICS.items():
            rec = self.get(slug, spec["name"], "quadric")
            rec["name"] = spec["name"]
            rec["primary_family"] = "quadric"
            d = rec["definition"]
            d.update(mode="implicit", fidelity="exact", exactness="elementary",
                     scale="unit_radius", degree=2,
                     polynomial=spec["polynomial"])
            rec["metrics"] = dict(rec.get("metrics") or {},
                                  normalization="unit_radius")
            rec.setdefault("tradition", []).append("classical")
            rec["provenance"] = {
                "definition": "Classical closed form, written directly in the "
                              "exact language and checked by the validator.",
                "sources": spec.get("sources", ["Classical."]),
            }
            deep_merge(rec, spec.get("extra", {}))
            # P1 of the implementation plan: math_art/quadric_generator.py
            # builds all thirteen from their exact charts. Three of them
            # ALSO ship as ruled surfaces or as a Delaunay member, so those
            # records carry two construction entries -- being reachable two
            # ways is a fact about the surface, not a duplication.
            kind = QUADRIC_KIND.get(slug)
            if kind:
                self.add_construction(rec, {
                    "generator": "math_art.quadric_generator",
                    "operator_id": "mesh.quadric_add",
                    "key": kind, "definition_index": 0,
                    "implemented": True,
                })
            elif not rec["construction"]:
                rec["construction"] = [{
                    "implemented": False,
                    "blocked_by": "No math_art operator builds this quadric.",
                    "resume": "Add it to math_art/quadric_generator.py.",
                }]
            self.report.append(("quadric", slug, "emit", slug))

    def stage_curated_only(self):
        """Emit curated records that no registry row produces.

        The torus and the Delaunay family record are real surfaces with
        real literature that simply have no enum row of their own -- the
        torus because nothing in math_art adds a bare torus, the Delaunay
        family because its MEMBERS ship rather than the family.  Without
        this the curation for them would sit unused, which the build
        reports as an orphan.
        """
        for slug in CURATED_ONLY:
            facts = curation.facts_for(slug)
            if not facts:
                self.problems.append(
                    "CURATED_ONLY names %r but curation has no facts for it"
                    % slug)
                continue
            rec = self.get(slug, facts.get("name", slug.replace("-", " ").title()),
                           facts.get("primary_family", "misc"))
            if not rec["construction"]:
                rec["construction"] = [{
                    "implemented": False,
                    "blocked_by": CURATED_ONLY[slug],
                }]
            self.report.append(("curated", slug, "emit", slug))

    def stage_presets(self):
        """Presets added to operators that are not FLAT_SOURCES.

        `mesh.hopf_torus_add` lives in a module with two operators, so
        reading its enums wholesale would emit rows for both; naming the
        one preset is safer than teaching the reader to disambiguate.
        """
        for slug, (module, op, key, name) in IMPLEMENTED_PRESET.items():
            path = os.path.join(ROOT, "math_art", *module.split(".")) + ".py"
            src = ""
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            if key not in src:
                self.problems.append(
                    "IMPLEMENTED_PRESET claims %s has key %r, which is not in "
                    "the module source" % (op, key))
                continue
            rec = self.get(slug, name, "misc")
            self.add_construction(rec, {
                "generator": "math_art." + module, "operator_id": op,
                "key": key, "definition_index": 0, "implemented": True,
            })
            self.report.append(("preset", slug, "emit", slug))

    def stage_singletons(self):
        """Operators that build one surface and so expose no enum rows.

        Verified against math_art/menu_defs.py: an operator listed here
        that no longer exists is a build problem, not a silent skip.
        """
        for slug, name, family, module, op, cond in SINGLETONS:
            path = os.path.join(ROOT, "math_art", *module.split(".")) + ".py"
            exists = os.path.exists(path)
            rec = self.get(slug, name, family)
            rec["name"] = name
            rec["primary_family"] = family
            if cond != "none":
                rec["curvature"]["condition"] = cond
            self.add_construction(rec, {
                "generator": "math_art." + module,
                "operator_id": op, "definition_index": 0,
                "implemented": bool(exists),
                "blocked_by": None if exists else
                "module %s not found in this tree" % module,
            })
            if exists:
                self.cite(rec, module, 1, name)
            else:
                self.problems.append(
                    "SINGLETONS names %s but %s does not exist" % (op, path))
            self.report.append(("singleton", slug, "emit", slug))

    def stage_missing(self):
        """Records for surfaces that are NOT implemented.

        The point of the database as a coverage ledger: an absent surface
        with a stated reason is strictly more useful than no record, and
        far more useful than a markdown list that goes stale.
        """
        for slug, spec in MISSING.items():
            # A MISSING entry is a CLAIM that nothing builds this surface.
            # If an earlier stage found an operator that does, the operator
            # wins and the claim is stale -- silently overwriting a working
            # construction with `implemented: false` would make the ledger
            # lie in the one direction it must never lie.
            existing = self.records.get(slug)
            if existing and any(c.get("implemented")
                                for c in existing.get("construction") or []):
                # The CLAIM is stale. The facts attached to it are not,
                # and dropping them is the more damaging mistake: the
                # curated entry is where the PUBLISHED invariant lives --
                # "165 ordinary double points", "35 cusps" -- and that
                # number is the only independent oracle these surfaces
                # have. Discard it when the surface gets built and the
                # implementation can never be checked against the
                # literature again; the record would carry a verified
                # polynomial and nothing to verify it against, which is
                # backwards, because the surface got built precisely
                # because someone found the source.
                #
                # So: merge the facts, drop the claim. Never the
                # `implemented: false` construction, never the "not
                # transcribed" provenance, and never `polynomial: None`
                # over a stored equation.
                extra = copy.deepcopy(spec.get("extra", {}))
                edfn = extra.get("definition")
                if isinstance(edfn, dict):
                    edfn.pop("polynomial", None)
                    # keep BOTH notes: the built one says what this record
                    # stores, the curated one describes the construction
                    curated_note = edfn.pop("note", None)
                    if curated_note:
                        have = (existing.get("definition") or {}).get("note")
                        edfn["note"] = ("%s\n\n%s" % (have, curated_note)
                                        if have else curated_note)
                deep_merge(existing, extra)
                prov = existing.setdefault("provenance", {})
                have_src = prov.setdefault("sources", [])
                for s in spec.get("sources", []):
                    if s not in have_src:
                        have_src.append(s)
                self.report.append(("missing", slug, "superseded", slug))
                continue
            rec = self.get(slug, spec["name"], spec.get("family", "algebraic"))
            rec["name"] = spec["name"]
            rec["primary_family"] = spec.get("family", "algebraic")
            rec["definition"] = dict(
                {"mode": spec.get("mode", "implicit"), "fidelity": "exact",
                 "polynomial": None},
                **spec.get("definition", {}))
            rec["construction"] = [{
                "implemented": False,
                "blocked_by": spec.get("blocked_by"),
                "resume": spec.get("resume"),
            }]
            rec["provenance"] = {
                "definition": "Not transcribed. The record exists to state "
                              "that the surface is absent and why; a "
                              "fabricated equation would be worse than a null.",
                "sources": spec.get("sources", ["see resume pointer"]),
            }
            deep_merge(rec, spec.get("extra", {}))
            self.report.append(("missing", slug, "emit", slug))

    # -- finish ----------------------------------------------------------

    def curate(self):
        for slug, rec in self.records.items():
            deep_merge(rec, curation.facts_for(slug))
            # independently published invariants, for the drive stage to
            # compare the generator against
            deep_merge(rec, published.invariants_for(slug))
            deep_merge(rec, papers.invariants_for(slug))
            for table in (ferreol.ids(), vmm.ids(), algsurf.ids()):
                extra = table.get(slug)
                if extra:
                    got = rec.setdefault("ids", {})
                    for k, v in extra.items():
                        got.setdefault(k, v)
            poly = curation.polyhedral_analogue(slug)
            if poly:
                rec.setdefault("relations", {})["polyhedral_analogue"] = poly

        # suspected-identity edges, both directions
        for ent in mapping.SUSPECTED_SAME:
            for a, b in ((ent["a"], ent["b"]), (ent["b"], ent["a"])):
                if a in self.records:
                    rel = self.records[a].setdefault("relations", {})
                    lst = rel.setdefault("same_surface_as", [])
                    if not any(e.get("slug") == b for e in lst):
                        lst.append({"slug": b,
                                    "confidence": ent["confidence"],
                                    "check": ent["check"],
                                    "source": ent["source"]})

    def finish(self):
        for slug, rec in self.records.items():
            self._named_after(rec)
            self._attach_chart(slug, rec)
            self._attach_we(slug, rec)
            self._resolve_ids(slug, rec)
            self._definition_note(rec)
        for rec in self.records.values():
            if not rec["provenance"].get("sources"):
                rec["provenance"]["sources"] = [
                    "Derived from the Math Art generator registries; no "
                    "literature citation has been curated for this record yet."]
            # dedupe tradition
            if rec.get("tradition"):
                rec["tradition"] = sorted(set(rec["tradition"]))
            views.apply_all(rec)

    def _named_after(self, rec):
        """Whose name the surface carries.

        Set only when a surname in the LABEL also appears in one of the
        record's citations -- two independent confirmations. `named_after`
        rather than `discovered_by` deliberately: "Wei Doubly Periodic" is
        certainly named for Wei, but asserting Wei DISCOVERED it is a
        stronger claim than the label supports, and the database does not
        make claims it cannot back.
        """
        if rec.get("named_after") or rec.get("discovered_by"):
            return
        names = references.surnames(rec.get("name", ""))
        if not names:
            return
        cited = " ".join((rec.get("provenance") or {}).get("sources") or [])
        hit = sorted(n for n in names if n in cited)
        if hit:
            rec["named_after"] = ", ".join(hit)

    def _resolve_ids(self, slug, rec):
        """Cross-reference IDs that RESOLVE against the local mirrors.

        Where a mirror files a surface under a DIFFERENT name, that name
        is recorded as an alternate rather than discarded: Ferreol calls
        the canal surface "Tube", Riemann's minimal example "Skew
        catenoid", the Jorge-Meeks k-noid "Trinoid" and the genus-g
        surface "n-holed torus". Those are real alternate names -- and
        capturing them is better than suppressing the cross-check's
        identity test, which flagged all seven as possible wrong ids.
        """
        ids = rec.setdefault("ids", {})
        if not ids.get("mathcurve"):
            got = sources.mathcurve_id(slug, rec.get("name", ""))
            if got:
                ids["mathcurve"] = got
        stem = ids.get("mathcurve")
        if stem:
            try:
                from surfdb import crosscheck as _cc
                text = _cc.read_page(stem, 3000)
                import re as _re
                m = _re.search(r"^#\s+(.+)$", text or "", _re.M)
                title = (m.group(1).strip() if m else "")
            except Exception:                         # noqa: BLE001
                title = ""
            if title and title.lower() != (rec.get("name") or "").lower():
                alts = rec.setdefault("alternate_names", [])
                if title not in alts:
                    alts.append(title)
        if not ids:
            rec.pop("ids", None)

    @staticmethod
    def _chart_block(ch, note=None):
        block = {"x": ch["x"], "y": ch["y"], "z": ch["z"],
                 "u_range": list(ch["u_range"]),
                 "v_range": list(ch["v_range"])}
        if ch.get("periodic_u"):
            block["periodic_u"] = True
        if ch.get("periodic_v"):
            block["periodic_v"] = True
        if note or ch.get("note"):
            block["note"] = note or ch.get("note")
        return block

    def _attach_chart(self, slug, rec):
        """Store the parametric chart, where one is verified for this slug.

        Two sources, two different gates:

        * tools/surfdb/charts.py -- condition-gated: the chart must
          measure the curvature condition its record claims (H = 0,
          K = +-1, K = 0), so only gated records can carry one.
        * tools/surfdb/parcharts.py -- oracle-gated: the chart is
          reproduced pointwise against the shipped builder (which
          imports headlessly), verified HERE at build time, so it works
          for records with no curvature condition too.  A chart that
          fails its oracle is reported and dropped, never stored.

        A chart for a record whose primary definition is Weierstrass
        (the merged Catalan/Bjorling row) becomes an ALTERNATE
        definition rather than being silently discarded.
        """
        d = rec["definition"]
        ch = charts.chart_for(slug)
        provenance = None
        if ch:
            provenance = (
                "Chart as curated in tools/surfdb/charts.py, verified "
                "numerically against the curvature condition the record "
                "claims (measured over the chart by the validator and the "
                "charts self-test).")
        elif parcharts is not None:
            ch = parcharts.chart_for(slug)
            if ch:
                try:
                    ok, detail = parcharts.verify(slug)
                except Exception as exc:                  # noqa: BLE001
                    ok, detail = False, "verify raised: %s" % exc
                if not ok:
                    self.problems.append(
                        "parametric chart for %s fails its oracle: %s"
                        % (slug, detail))
                    return
                provenance = (
                    "Chart reproduced numerically against the shipped "
                    "implementation: %s." % detail)
            elif (d.get("mode") == "parametric" and not d.get("x")
                    and not d.get("polynomial") and not d.get("note")):
                why = parcharts.reason_for(slug)
                if why:
                    d["note"] = (
                        "No chart is stored: %s. The shipped implementation "
                        "is authoritative; an unverified transcription would "
                        "silently define a different surface." % why)
        if not ch:
            return
        if d.get("mode") == "parametric" and not d.get("polynomial"):
            block = self._chart_block(ch)
            d.update(block)
            if "note" not in block:
                d.pop("note", None)
            d.setdefault("exactness", "elementary")
            if provenance:
                rec["provenance"]["definition"] = provenance
        elif not d.get("x"):
            # the same surface by another construction: an alternate
            alts = rec.setdefault("alternate_definitions", [])
            if not any(a.get("x") for a in alts):
                block = self._chart_block(ch)
                block.update(mode="parametric", fidelity="exact",
                             exactness="elementary")
                alts.append(block)

    def _zoo_key_for(self, rec):
        """The engine key a record was built from, if any.

        Covers the zoo/parametric registries AND the exact-TPMS rows
        (tpms.py), whose keys the extractor also dispositions.
        """
        for c in rec.get("construction") or []:
            if (c.get("generator") == "math_art.minsurf.parametric"
                    and c.get("key")):
                return c["key"]
            if (c.get("generator") == "math_art.minsurf.tpms"
                    and c.get("family") == "TPMS_EXACT" and c.get("key")):
                return c["key"]
        return None

    def _attach_we(self, slug, rec):
        """Store the Weierstrass pair, verified against the zoo row.

        Two oracle-gated sources, in order:

        1. The AST extractor (tools/surfdb/weextract.py), which re-reads
           zoo.py FRESH on every build and verifies each emitted pair
           against the shipped callables -- so the stored data follows
           the engine as it changes instead of rotting behind it (the
           hand table lost three rows to exactly that drift).
        2. The curated table (tools/surfdb/wedata.py) as fallback, which
           still self-verifies against the zoo before it is trusted.

        Rows the extractor cannot express (dedicated meshers, elliptic
        immersions) contribute their honest REASON to the definition
        note, plus the domain/puncture metadata that IS extractable.

        Where the record's primary definition is already a parametric
        chart (the classical rows), the pair becomes an ALTERNATE
        definition -- the same surface by another construction, which is
        exactly what that list is for.
        """
        d = rec["definition"]
        key = self._zoo_key_for(rec) or wedata.ZOO_KEY.get(slug)
        fams = {c.get("family") for c in rec.get("construction") or []}
        if not key:
            if "TPMS_EXACT" in fams and d.get("mode") == "weierstrass" \
                    and not d.get("gauss_map") and not d.get("note"):
                d["note"] = (
                    "No (g, dh) pair is stored: the exact data live on a "
                    "genus-3 branched cover (the hyperelliptic square/"
                    "hexagonal curve) integrated by math_art/minsurf/"
                    "weierstrass.pgd_build and hexagonal.py; g and dh are "
                    "algebraic on that cover, not single-valued elementary "
                    "expressions on a plane domain, so there is nothing the "
                    "exact language can store without lying about the "
                    "Riemann surface underneath.")
            return
        block = None
        reason = None
        got = {}
        try:
            got = weextract.extract_all().get(key) or {}
        except Exception as exc:                      # noqa: BLE001
            self.problems.append("weextract failed on %s: %s" % (key, exc))
        if got.get("bj"):
            # A Bjorling row's defining datum is its SEED, not a (g, dh)
            # pair: store the verified curve/normal in the definition,
            # which is what actually specifies the surface.
            bj = got["bj"]
            if d.get("mode") == "weierstrass" and not d.get("gauss_map"):
                nrm = ("the Frenet principal normal of the curve"
                       if bj["normal"] == "frenet"
                       else "n(t) = (%s)" % ", ".join(bj["normal"]))
                d["note"] = (
                    "Bjorling's problem: the unique minimal surface through "
                    "a given real-analytic strip. The defining seed, "
                    "reproduced numerically against the shipped row "
                    "(math_art/minsurf/zoo.py, %s): c(t) = (%s), with "
                    "normal %s, t in [%s, %s]. The Weierstrass data is the "
                    "holomorphic extension of the seed, computed "
                    "numerically by the engine -- the seed, not a (g, dh) "
                    "pair, is what specifies this surface."
                    % (key, ", ".join(bj["curve"]), nrm,
                       bj["t_range"][0], bj["t_range"][1]))
                if bj.get("params"):
                    d.setdefault("parameters", [
                        {"name": n, "domain": "see the generator's p_from",
                         "default": v, "integer": isinstance(v, int)}
                        for n, v in sorted(bj["params"].items())])
                rec["provenance"]["definition"] = (
                    "Bjorling seed reproduced numerically against the "
                    "shipped implementation (math_art/minsurf/zoo.py, row "
                    "%s): %s." % (key, got.get("detail", "")))
            return
        if got.get("g"):
            params = dict(got["params"])
            # curated member overrides (e.g. the classical k = 1 Enneper,
            # which is the member the record's other facts describe)
            params.update(wedata.PARAM_DEFAULT.get(slug, {}))
            block = {
                "mode": "weierstrass", "fidelity": "exact",
                "exactness": "numerical-integral",
                "gauss_map": got["g"], "height_differential": got["dh"],
                "parameters": [
                    {"name": n, "domain": "see the generator's p_from",
                     "default": v, "integer": isinstance(v, int)}
                    for n, v in sorted(params.items())],
                "note": "Extracted from the shipped row's source by AST "
                        "(complete multi-line lambdas, %s) and reproduced "
                        "numerically against the shipped callables over "
                        "the complex plane, exactly -- no scalar slack."
                        % got.get("derived", "g/dh"),
            }
            if got.get("note_extra"):
                block["note"] += " " + got["note_extra"]
            if key == "PGD":
                block["associate_angle_degrees"] = 0.0
            rec["provenance"]["definition"] = (
                "Weierstrass data reproduced numerically against the shipped "
                "implementation (math_art/minsurf/zoo.py, row %s) by sampling "
                "g and dh over rings in the complex plane at the row's "
                "default parameters: %s." % (key, got.get("detail", "")))
        else:
            reason = got.get("reason")

        if block is None:
            ent = wedata.data_for(slug)
            if ent:
                try:
                    from minsurf import zoo as _zoo
                    spec = _zoo.WE_SURFACES[wedata.ZOO_KEY[slug]]
                    p = spec["p_from"](3, 1.0)
                    ok, details = wedata.verify(slug, spec, p)
                    if ok:
                        defaults = wedata.defaults_for(slug, p)
                        block = {
                            "mode": "weierstrass", "fidelity": "exact",
                            "exactness": "numerical-integral",
                            "gauss_map": ent["g"],
                            "height_differential": ent["dh"],
                            "parameters": [
                                {"name": n, "domain": "see the generator",
                                 "default": v,
                                 "integer": isinstance(v, int)}
                                for n, v in sorted(defaults.items())],
                            "note": "Verified against math_art/minsurf/"
                                    "zoo.py by sampling both over the "
                                    "complex plane; g exactly, dh up to "
                                    "the constant multiple that merely "
                                    "scales the surface.",
                        }
                        rec["provenance"]["definition"] = (
                            "Weierstrass data reproduced numerically against "
                            "the shipped implementation (math_art/minsurf/"
                            "zoo.py, row %s): %s."
                            % (wedata.ZOO_KEY[slug], "; ".join(details)))
                    else:
                        self.problems.append(
                            "Weierstrass data for %s disagrees with the "
                            "shipped row: %s" % (slug, "; ".join(details)))
                except Exception as exc:              # noqa: BLE001
                    self.problems.append("cannot verify WE data for %s: %s"
                                         % (slug, exc))

        # domain / punctures are extractable even where the pair is not --
        # for a Weierstrass surface they are part of what renders it
        if d.get("mode") == "weierstrass":
            if got.get("domain") and not d.get("domain"):
                d["domain"] = got["domain"]
            if got.get("punctures") and not d.get("punctures"):
                d["punctures"] = got["punctures"]

        if block is None:
            if reason and d.get("mode") == "weierstrass" \
                    and not d.get("gauss_map") and not d.get("note"):
                d["note"] = (
                    "No (g, dh) pair is stored: %s. The shipped "
                    "implementation is authoritative; an unverified "
                    "transcription would silently define a different "
                    "surface." % reason)
            return

        if got.get("domain"):
            block.setdefault("domain", got["domain"])
        if got.get("punctures"):
            block.setdefault("punctures", got["punctures"])
        if d.get("mode") == "weierstrass" and not d.get("gauss_map"):
            d.update(block)
            d.pop("note", None)
            d["note"] = block["note"]
        elif d.get("mode") != "weierstrass":
            alts = rec.setdefault("alternate_definitions", [])
            if not any(a.get("gauss_map") for a in alts):
                alts.append(block)

    def _definition_note(self, rec):
        """No silent blanks: say when the defining datum is not stored.

        A record whose definition carries neither a polynomial, a chart,
        Weierstrass data nor a level function is not thereby WRONG -- most
        of the Weierstrass zoo is built by a dedicated function with no
        extractable closed form. But an empty definition block reads like
        an oversight, so it states the situation and names what IS
        authoritative.
        """
        for d in [rec["definition"]] + list(rec.get("alternate_definitions") or []):
            has = any(d.get(k) for k in
                      ("polynomial", "x", "gauss_map", "level_function",
                       "profile_curve", "energy", "operation",
                       "construction_rule"))
            if has or d.get("note"):
                continue
            gen = next((c.get("generator") for c in rec.get("construction") or []
                        if c.get("generator")), None)
            impl = any(c.get("implemented")
                       for c in rec.get("construction") or [])
            if not impl:
                # There is no shipped implementation to defer to, and
                # claiming one would make the ledger lie: the record's
                # whole point is that the surface is ABSENT.
                d["note"] = (
                    "No closed form is stored: the surface is not "
                    "implemented in math_art, so there is no shipped "
                    "code to verify a transcription against, and an "
                    "unverified formula from the literature would be "
                    "worse than a null. See construction.blocked_by.")
            else:
                d["note"] = (
                    "No closed form is stored for this surface. It is "
                    "defined by its shipped implementation"
                    + (" in %s" % gen if gen else "")
                    + ", which is authoritative; an unverified transcription "
                    "would silently define a different surface.")

    def write(self):
        written = 0
        for slug, rec in sorted(self.records.items()):
            folder = os.path.join(OUT, "surfaces", rec["primary_family"])
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, slug + ".json"), "w",
                      encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            written += 1
        self.write_index()
        return written

    def write_index(self):
        entries = []
        fams = set()
        impl = 0
        for slug, rec in sorted(self.records.items()):
            fams.update(rec.get("families") or [])
            is_impl = any(c.get("implemented")
                          for c in rec.get("construction") or [])
            impl += 1 if is_impl else 0
            dfns = [rec["definition"]] + list(rec.get("alternate_definitions") or [])
            topo = rec.get("topology") or {}
            sym = rec.get("symmetry") or {}
            ends = topo.get("ends") or []
            end_total = 0
            for e in ends:
                if isinstance(e.get("count"), int):
                    end_total += e["count"]
                else:
                    end_total = None
                    break
            entries.append({
                "slug": slug, "name": rec["name"],
                "path": "surfaces/%s/%s.json" % (rec["primary_family"], slug),
                "primary_family": rec["primary_family"],
                "families": rec.get("families") or [],
                "definition_mode": rec["definition"].get("mode"),
                "has_approximation": any(d.get("fidelity") == "approximation"
                                         for d in dfns),
                "curvature_condition": rec["curvature"].get("condition"),
                "periodicity_rank": sym.get("periodicity_rank", 0),
                "genus": topo.get("genus"),
                "genus_per_cell": topo.get("genus_per_cell"),
                "orientable": topo.get("orientable"),
                "ends": end_total,
                "embedding": (rec.get("embedding") or {}).get("quality"),
                "symmetry": {
                    "kind": sym.get("kind"),
                    "symbol": (sym.get("hermann_mauguin")
                               or sym.get("schoenflies")),
                    "ita_number": sym.get("ita_number"),
                    "order": sym.get("order"),
                },
                "implemented": is_impl,
                "operator_ids": sorted({c["operator_id"]
                                        for c in rec.get("construction") or []
                                        if c.get("operator_id")}),
                "specimen_count": len(rec.get("specimens") or []),
            })
        index = {
            "schema_version": SCHEMA_VERSION,
            "database_version": DATABASE_VERSION,
            "updated": BUILD_DATE,
            "normalization_default": "published",
            "families": sorted(fams),
            "primary_families": sorted({e["primary_family"] for e in entries}),
            "count": len(entries),
            "implemented_count": impl,
            "entries": entries,
        }
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


BUILD_DATE = "2026-08-28"

# Record slug -> the enum key on mesh.quadric_add.
# Surfaces implemented as a PRESET on an operator whose module is not a
# FLAT_SOURCE.  slug -> (module, operator id, enum key, name)
IMPLEMENTED_PRESET = {
    "bianchi-pinkall-flat-torus": (
        "hopf_fibration_generator", "mesh.hopf_torus_add",
        "BIANCHI_PINKALL", "Bianchi-Pinkall Flat Torus"),
}


QUADRIC_KIND = {
    "sphere": "SPHERE",
    "spheroid": "SPHEROID",
    "ellipsoid": "ELLIPSOID",
    "elliptic-paraboloid": "ELLIPTIC_PARABOLOID",
    "hyperbolic-paraboloid": "HYPERBOLIC_PARABOLOID",
    "hyperboloid-one-sheet": "HYPERBOLOID_ONE",
    "hyperboloid-two-sheets": "HYPERBOLOID_TWO",
    "elliptic-cone": "ELLIPTIC_CONE",
    "circular-cylinder": "CIRCULAR_CYLINDER",
    "elliptic-cylinder": "ELLIPTIC_CYLINDER",
    "hyperbolic-cylinder": "HYPERBOLIC_CYLINDER",
    "parabolic-cylinder": "PARABOLIC_CYLINDER",
    "plane": "PLANE",
}


# The thirteen classical quadrics, at unit radius.  Nine non-degenerate
# real quadrics, plus the sphere and spheroid as the distinguished
# ellipsoids and the plane as the degenerate limit that every ruled and
# developable discussion needs to refer to.
QUADRICS = {
    "sphere": {
        "name": "Sphere", "polynomial": "x**2 + y**2 + z**2 - 1",
        "extra": {"alternate_names": ["Unit sphere"]},
    },
    "spheroid": {
        "name": "Spheroid", "polynomial": "x**2 + y**2 + z**2/c**2 - 1",
        "extra": {
            "alternate_names": ["Ellipsoid of revolution"],
            "definition": {"parameters": [
                {"name": "c", "symbol": "c", "domain": "(0, inf)",
                 "default": 2.0,
                 "note": "polar semi-axis; c > 1 prolate, c < 1 oblate, "
                         "c = 1 the sphere"}]},
            "topology": {"genus": 0, "euler_characteristic": 2,
                         "orientable": True, "compact": True},
            "embedding": {"quality": "embedded"},
            "specimens": [
                {"label": "prolate spheroid", "parameters": {"c": 2.0}},
                {"label": "oblate spheroid", "parameters": {"c": 0.5}},
                {"label": "sphere", "parameters": {"c": 1.0}, "slug": "sphere"},
            ],
        },
    },
    "ellipsoid": {
        "name": "Ellipsoid",
        "polynomial": "x**2/a**2 + y**2/b**2 + z**2/c**2 - 1",
        "extra": {
            "ids": {"mathworld": "Ellipsoid", "mathcurve": "ch1207_ellipsoid_2"},
            "definition": {"parameters": [
                {"name": "a", "symbol": "a", "domain": "(0, inf)", "default": 1.0},
                {"name": "b", "symbol": "b", "domain": "(0, inf)", "default": 0.7},
                {"name": "c", "symbol": "c", "domain": "(0, inf)", "default": 0.4}]},
            "topology": {"genus": 0, "euler_characteristic": 2,
                         "orientable": True, "compact": True,
                         "boundary_components": 0},
            "embedding": {"quality": "embedded"},
        },
    },
    "elliptic-paraboloid": {
        "name": "Elliptic Paraboloid",
        "polynomial": "x**2/a**2 + y**2/b**2 - z",
        "extra": {
            "ids": {"mathworld": "EllipticParaboloid",
                    "mathcurve": "ch1263_paraboloidelliptic_2"},
            "definition": {"parameters": [
                {"name": "a", "symbol": "a", "domain": "(0, inf)", "default": 1.0},
                {"name": "b", "symbol": "b", "domain": "(0, inf)", "default": 1.0}]},
            "topology": {"compact": False, "orientable": True},
            "embedding": {"quality": "embedded"},
        },
    },
    "hyperboloid-two-sheets": {
        "name": "Hyperboloid of Two Sheets",
        "polynomial": "x**2/a**2 + y**2/b**2 - z**2/c**2 + 1",
        "extra": {
            "ids": {"mathworld": "Two-SheetedHyperboloid",
                    "mathcurve": "ch1287_hyperboloid2_2"},
            "definition": {"parameters": [
                {"name": "a", "symbol": "a", "domain": "(0, inf)", "default": 1.0},
                {"name": "b", "symbol": "b", "domain": "(0, inf)", "default": 1.0},
                {"name": "c", "symbol": "c", "domain": "(0, inf)", "default": 1.0}]},
            "topology": {"compact": False, "orientable": True},
            "embedding": {"quality": "embedded"},
            "notes": {"caveats": [
                "Disconnected: two sheets. The only quadric in this table "
                "whose real locus has two components, which is why a "
                "component count is worth measuring rather than assuming."]},
        },
    },
    "elliptic-cone": {
        "name": "Elliptic Cone",
        "polynomial": "x**2/a**2 + y**2/b**2 - z**2",
        "extra": {
            "ids": {"mathworld": "EllipticCone", "mathcurve": "ch1235_coneelliptique_2"},
            "definition": {"parameters": [
                {"name": "a", "symbol": "a", "domain": "(0, inf)", "default": 1.0},
                {"name": "b", "symbol": "b", "domain": "(0, inf)", "default": 1.0}]},
            "curvature": {"condition": "flat"},
            "topology": {"compact": False, "orientable": True},
            "embedding": {"quality": "singular",
                          "singularities": [{"type": "conical point", "count": 1}]},
            "notes": {"caveats": [
                "Developable away from the apex, and ruled; the apex is the "
                "one point where it is not a smooth surface."]},
        },
    },
    "circular-cylinder": {
        "name": "Circular Cylinder", "polynomial": "x**2 + y**2 - 1",
        "extra": {
            "ids": {"mathworld": "Cylinder", "mathcurve": "ch1230_cylindrederevolution_2"},
            "curvature": {"condition": "flat",
                          "gaussian": {"exact": "0", "value": 0.0,
                                       "source": "classical"}},
            "topology": {"compact": False, "orientable": True, "genus": 0},
            "embedding": {"quality": "embedded"},
            "relations": {"member_of": "delaunay-surface"},
            "notes": {"caveats": [
                "Both flat (K = 0, so developable) and CMC (H = 1/2r), which "
                "is why it sits in the Delaunay family. The curvature facet "
                "records `flat` as the stronger condition."]},
        },
    },
    "elliptic-cylinder": {
        "name": "Elliptic Cylinder", "polynomial": "x**2/a**2 + y**2/b**2 - 1",
        "extra": {
            "ids": {"mathworld": "EllipticCylinder", "mathcurve": "ch1248_cylindreelliptique_2"},
            "definition": {"parameters": [
                {"name": "a", "symbol": "a", "domain": "(0, inf)", "default": 1.0},
                {"name": "b", "symbol": "b", "domain": "(0, inf)", "default": 0.6}]},
            "curvature": {"condition": "flat"},
            "topology": {"compact": False, "orientable": True},
            "embedding": {"quality": "embedded"},
        },
    },
    "hyperbolic-cylinder": {
        "name": "Hyperbolic Cylinder", "polynomial": "x**2/a**2 - y**2/b**2 - 1",
        "extra": {
            "ids": {"mathworld": "HyperbolicCylinder", "mathcurve": "ch1249_cylindrehyperbolic_2"},
            "definition": {"parameters": [
                {"name": "a", "symbol": "a", "domain": "(0, inf)", "default": 1.0},
                {"name": "b", "symbol": "b", "domain": "(0, inf)", "default": 1.0}]},
            "curvature": {"condition": "flat"},
            "topology": {"compact": False, "orientable": True},
            "embedding": {"quality": "embedded"},
        },
    },
    "parabolic-cylinder": {
        "name": "Parabolic Cylinder", "polynomial": "x**2 - z",
        "extra": {
            "ids": {"mathworld": "ParabolicCylinder", "mathcurve": "ch1250_cylindreparabolic_2"},
            "curvature": {"condition": "flat"},
            "topology": {"compact": False, "orientable": True},
            "embedding": {"quality": "embedded"},
        },
    },
    "plane": {
        "name": "Plane", "polynomial": "z",
        "extra": {
            "ids": {"mathworld": "Plane", "mathcurve": "ch1320_plan_2"},
            "curvature": {"condition": "minimal",
                          "mean": {"exact": "0", "value": 0.0,
                                   "source": "classical"},
                          "gaussian": {"exact": "0", "value": 0.0,
                                       "source": "classical"}},
            "topology": {"compact": False, "orientable": True, "complete": True,
                         "genus": 0, "finite_total_curvature": True},
            "embedding": {"quality": "embedded"},
            "notes": {"caveats": [
                "Both minimal and flat -- the only surface that is both, and "
                "the trivial member of every minimal-surface classification. "
                "Filed as a quadric because degree is the axis this folder "
                "sorts on; the curvature facet records `minimal` as the "
                "stronger condition."]},
        },
    },
}

# Curated surfaces with no registry row of their own.  `slug -> why it is
# not implemented`.
CURATED_ONLY = {
    "torus": "No math_art operator adds a bare torus; it appears only as the "
             "base of derived generators. Blender's own primitive is not a "
             "math_art construction.",
    "delaunay-surface": "The family record. Its MEMBERS ship "
                        "(mesh.delaunay_surface_add), so the family itself "
                        "has no row of its own.",
}

# Operators that build exactly ONE surface, so they have no enum to read.
# (slug, name, primary family, module, operator id, curvature condition)
SINGLETONS = [
    ("oloid", "Oloid", "ruled", "oloid_generator", "mesh.oloid_add", "flat"),
    ("sphericon", "Sphericon", "ruled", "sphericon_generator",
     "mesh.sphericon_add", "flat"),
    ("spherical-harmonic-surface", "Spherical Harmonic Surface", "spectral",
     "spherical_harmonic_generator", "mesh.spherical_harmonic_add", "none"),
    ("supershape", "Supershape", "spectral", "supershape_generator",
     "mesh.supershape_add", "none"),
    ("atomic-orbital-surface", "Atomic Orbital Surface", "spectral",
     "minsurf.orbital", "mesh.orbital_add", "none"),
    ("scherk-collins-sculpture", "Scherk-Collins Sculpture", "minimal",
     "scherk_collins_generator", "mesh.scherk_collins_add", "none"),
    ("seifert-surface", "Seifert Surface", "topological",
     "seifert_surface_generator", "mesh.seifert_surface_add", "none"),
    ("plateau-span", "Plateau Soap-Film Span", "physical",
     "minimal_surface_toolkit", "object.minimal_span", "minimal"),
    ("bubble-cluster", "Bubble Cluster", "physical", "bubble_generator",
     "mesh.bubble_cluster_add", "cmc"),
    ("relaxed-bubble", "Relaxed Bubble", "physical", "bubble_generator",
     "mesh.relaxed_bubble_add", "cmc"),
    ("capillary-surface", "Capillary Surface", "physical", "cmc_generator",
     "mesh.cmc_capillary_add", "cmc"),
    ("willmore-surface", "Willmore Surface", "topological",
     "willmore_generator", "mesh.willmore_add", "willmore"),
    ("bryant-surface", "Bryant Surface", "cmc", "bryant_generator",
     "mesh.bryant_surface_add", "cmc1-bryant"),
    ("d-form", "D-Form", "misc", "dform_generator", "mesh.dform_add", "flat"),
    ("hopf-torus", "Hopf Torus", "topological", "hopf_fibration_generator",
     "mesh.hopf_torus_add", "none"),
    ("crochet-hyperbolic-surface", "Crochet Hyperbolic Surface",
     "constant-curvature", "crochet_generator", "mesh.crochet_add",
     "k-const-negative"),
    ("saddle-polyhedron", "Saddle Polyhedron", "discrete",
     "saddle_polyhedron_generator", "mesh.saddle_polyhedron_add", "none"),
    ("minimal-polyhedron", "Minimal Polyhedron", "discrete",
     "minimal_polyhedron_generator", "mesh.minimal_surface_polyhedron_add",
     "minimal"),
    # NOTE: math_art/pearce_surface.py is a LIBRARY -- it defines no
    # operator. An earlier revision listed "mesh.pearce_surface_add",
    # which does not exist; the drive stage caught it by trying to call
    # it. Pearce's saddle surfaces are reached through the saddle
    # polyhedron generator, which is where the record points.
    ("pearce-saddle-surface", "Pearce Saddle Surface", "discrete",
     "pearce_surface", "mesh.saddle_polyhedron_add", "none"),
]


# ---------------------------------------------------------------------------
# Surfaces that are NOT implemented.  Each states WHY, and where to resume.
# This is the half of the database that a hand-written gap list cannot keep
# honest -- research/missing-surfaces-catalog.md listed 19 headline gaps and
# 13 of them had silently closed.
# ---------------------------------------------------------------------------

MISSING = {
    "sarti-dodecic": {
        "name": "Sarti Dodecic", "family": "algebraic",
        "blocked_by": "The pencil's singular members are fixed by a condition "
                      "not reconstructible from the shipped symmetric-function "
                      "machinery; see math_art/surfaces/algebraic.py:680-693, "
                      "where the omission is documented in code.",
        "resume": "A. Sarti, 'Pencils of symmetric surfaces in P_3', J. Algebra "
                  "246 (2001); research/papers/algebraic-surfaces/"
                  "sarti-2001-pencils-symmetric-surfaces/",
        "sources": ["A. Sarti, J. Algebra 246 (2001)."],
        "extra": {
            "discovered_by": "Alessandra Sarti", "year": 2001,
            "definition": {"degree": 12, "exactness": "elementary",
                           "note": "Invariant under the reflection group of "
                                   "the 600-cell."},
            "embedding": {"quality": "singular", "is_record": True,
                          "record_for": "maximum known nodes, degree 12",
                          "singularities": [{"type": "node (A1)", "count": 600}]},
            "tradition": ["classical"],
        },
    },
    "bianchi-pinkall-flat-torus": {
        "name": "Bianchi-Pinkall Flat Torus", "family": "topological",
        "mode": "derived",
        "blocked_by": "Not built. The Hopf lift of a spherical curve already "
                      "ships (mesh.hopf_torus_add), so this is a curve-family "
                      "change rather than new machinery.",
        "resume": "U. Pinkall, 'Hopf tori in S^3', Invent. Math. 81 (1985). "
                  "Heed the willmore-tori lesson in BACKLOG.md: the Hopf lift "
                  "degenerates at BOTH beta = 0 and beta = pi, and an untreated "
                  "case collapsed to aspect 0.006.",
        "sources": ["U. Pinkall, 'Hopf tori in S^3', Invent. Math. 81 (1985)."],
        "extra": {"discovered_by": "Luigi Bianchi; Ulrich Pinkall",
                  "year": 1985,
                  "curvature": {"condition": "flat"},
                  "topology": {"genus": 1, "compact": True, "orientable": True,
                               "euler_characteristic": 0},
                  "embedding": {"quality": "embedded"},
                  "tradition": ["classical"]},
    },
    "spherical-helicoid": {
        "name": "Spherical Helicoid", "family": "swept", "mode": "swept",
        "blocked_by": "Not built.",
        "resume": "3DXM Virtual Math Museum. Slots into "
                  "math_art/helical_surface_generator.py beside Corkscrew, "
                  "Hyperbolic Helicoid and Seashell.",
        "sources": ["3DXM Virtual Math Museum, Surfaces gallery."],
        "extra": {"tradition": ["gallery"]},
    },
    "multi-soliton-pseudospherical": {
        "name": "Multi-Soliton Pseudospherical Surface",
        "family": "constant-curvature", "mode": "parametric",
        "blocked_by": "Only the single BREATHER preset ships. The 2-, 3- and "
                      "4-soliton surfaces need Sym's formula.",
        "resume": "A. Bobenko, 'Surfaces in terms of 2 by 2 matrices' (1994) "
                  "for Sym's formula -- converted in research/papers/. NOTE: "
                  "Melko-Sterling 1993 does NOT contain a closed-form breather "
                  "parametrisation despite being cited for one. The Bianchi "
                  "permutability the bubbleton module already implements is the "
                  "same machinery.",
        "sources": ["A. I. Bobenko, in Harmonic Maps and Integrable Systems "
                    "(1994)."],
        "extra": {"curvature": {"condition": "k-const-negative",
                                "gaussian": {"exact": "-1", "value": -1.0}},
                  "embedding": {"quality": "self-intersecting"},
                  "tradition": ["classical"]},
    },
    "schoen-batwing": {
        "name": "Schoen Batwing Surface", "family": "minimal-periodic",
        "mode": "weierstrass",
        "blocked_by": "No published nodal formula, and no exact Weierstrass "
                      "tiling has been built. One of ~18 second-tier TPMS in "
                      "the same position.",
        "resume": "K. Brakke's Surface Evolver periodic-surface collection; "
                  "A. H. Schoen's own site. See "
                  "research/missing-surfaces-catalog.md Part D2.",
        "sources": ["A. H. Schoen, NASA TN D-5541 (1970) and later web "
                    "listings; K. Brakke's Surface Evolver collection."],
        "extra": {"curvature": {"condition": "minimal"},
                  "symmetry": {"kind": "space", "periodicity_rank": 3},
                  "topology": {"complete": True, "compact": False},
                  "tradition": ["crystallographic"]},
    },
    "canal-surface": {
        "name": "Canal Surface", "family": "derived", "mode": "derived",
        "blocked_by": "Not built as a general transform. The constant-radius "
                      "case (a pipe surface) already exists implicitly wherever "
                      "the repo tubes a curve (math_art/knots/tube.py), so the "
                      "new capability is the VARYING-radius envelope.",
        "resume": "R. Ferreol, mathcurve, 'surface canal'. Part B4 of "
                  "research/missing-surfaces-catalog.md recommends treating "
                  "the derived-surface transforms as one design pass rather "
                  "than seven entries.",
        "sources": ["G. Monge; see R. Ferreol, mathcurve, 'surface canal'."],
        "extra": {"definition": {"operation": "canal"},
                  "tradition": ["classical"]},
    },
    "focal-surface": {
        "name": "Focal Surface", "family": "derived", "mode": "derived",
        "blocked_by": "Not built.",
        "resume": "The locus of the principal centres of curvature. Flagged in "
                  "research/missing-surfaces-catalog.md B4 as the "
                  "highest-value member of the derived family, because it "
                  "produces something visually unlike anything shipped, and it "
                  "is a natural companion to math_art/curvature_color.py.",
        "sources": ["mathcurve, 'surface focale'."],
        "extra": {"definition": {"operation": "focal"},
                  "tradition": ["classical"]},
    },
    "dyck-surface": {
        "name": "Dyck's Surface", "family": "topological", "mode": "parametric",
        "blocked_by": "Not a separate generator BY DESIGN: it is the k = 3 "
                      "case of the shipped non-orientable genus-k row, which "
                      "subsumes it. This record exists so the name resolves.",
        "resume": "Build it from mesh.topological_surface_add with "
                  "preset NONORIENT and k = 3.",
        "sources": ["W. von Dyck, 'Beitrage zur Analysis situs', Math. Ann. 32 "
                    "(1888)."],
        "extra": {"discovered_by": "Walther von Dyck", "year": 1888,
                  "topology": {"orientable": False, "one_sided": True,
                               "compact": True, "non_orientable_genus": 3,
                               "euler_characteristic": -1},
                  "embedding": {"quality": "immersed"},
                  "relations": {"member_of": "non-orientable-genus-k"},
                  "tradition": ["classical", "physical-model"]},
    },
    "klein-quartic": {
        "name": "Klein Quartic", "family": "topological", "mode": "implicit",
        "blocked_by": "Not built. It is a Riemann surface with a named "
                      "sculptural realisation, which is what brings it in "
                      "scope; a generic hyperelliptic curve would not be.",
        "resume": "F. Klein (1878). Helaman Ferguson's 'The Eightfold Way' is "
                  "the sculpture; see research/books/ for "
                  "helaman_ferguson_mathematics_in_stone_and_bronze_1994.",
        "sources": ["F. Klein, 'Ueber die Transformation siebenter Ordnung der "
                    "elliptischen Functionen', Math. Ann. 14 (1878)."],
        "extra": {"discovered_by": "Felix Klein", "year": 1878,
                  "definition": {"degree": 4,
                                 "note": "x^3 y + y^3 z + z^3 x = 0 in the "
                                         "complex projective plane -- not a "
                                         "real surface in R^3, so the "
                                         "realisation is a genus-3 handlebody "
                                         "carrying its 336-fold symmetry."},
                  "topology": {"genus": 3, "compact": True, "orientable": True,
                               "euler_characteristic": -4},
                  "tradition": ["classical", "sculptural"]},
    },
    "darboux-cyclide": {
        "name": "Darboux Cyclide", "family": "cyclide", "mode": "implicit",
        "blocked_by": "Not built. Best done as a quartic level set, which puts "
                      "it in the algebraic machinery rather than the "
                      "parametric cyclide module.",
        "resume": "G. Darboux. Prominent in modern architectural geometry as a "
                  "surface carrying several families of circles.",
        "sources": ["G. Darboux, Principes de geometrie analytique (1917)."],
        "extra": {"discovered_by": "Gaston Darboux",
                  "definition": {"degree": 4},
                  "relations": {"generalizes": ["dupin-cyclide"]},
                  "tradition": ["classical", "architectural"]},
    },
}


MISSING.update(tail.records())
MISSING.update(ferreol.records())
MISSING.update(vmm.records())
MISSING.update(algsurf.records())
MISSING.update(papers.records())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="*", help="stages to build (default: all)")
    ap.add_argument("--mapping-report", action="store_true",
                    help="print every registry row and its disposition")
    args = ap.parse_args()

    b = Builder()
    want = set(args.stages)
    def run(name):
        return not want or name in want

    if run("algebraic"):
        b.stage_algebraic()
    if run("minimal"):
        b.stage_minimal()
    if run("tpms"):
        b.stage_tpms()
    # dedupe: two sources share the 'constant-curvature' stage, and
    # running it once per source would process both twice and double the
    # mapping report
    for stage in dict.fromkeys(s[4] for s in FLAT_SOURCES):
        if run(stage):
            b.stage_flat(only_stage=stage)
    if run("quadric"):
        b.stage_quadric()
    if run("presets"):
        b.stage_presets()
    if run("singletons"):
        b.stage_singletons()
    if run("curated"):
        b.stage_curated_only()
    if run("missing"):
        b.stage_missing()

    b.curate()
    b.finish()

    # Curated facts that never reached a record are a mapping bug: either
    # the slug drifted or the row is not being read. Report, do not ignore.
    if not want:
        orphans = sorted(s for s in curation.FACTS if s not in b.records)
        for slug in orphans:
            b.problems.append(
                "curated facts for %r were never attached to a record -- the "
                "slug has drifted from what the build emits, or the row is "
                "not being read" % slug)

    if args.mapping_report:
        print("%-12s %-24s %-9s %s" % ("SOURCE", "ROW", "ACTION", "TARGET"))
        for src, key, kind, target in b.report:
            print("%-12s %-24s %-9s %s" % (src, key, kind, target or ""))
        print()

    n = b.write()
    impl = sum(1 for r in b.records.values()
               if any(c.get("implemented") for c in r["construction"]))
    counts = {}
    for r in b.records.values():
        counts[r["primary_family"]] = counts.get(r["primary_family"], 0) + 1

    print("wrote %d records (%d implemented, %d not) to %s"
          % (n, impl, n - impl, os.path.relpath(OUT, ROOT)))
    for fam in sorted(counts):
        print("   %-20s %d" % (fam, counts[fam]))
    if b.problems:
        print("\nPROBLEMS (%d):" % len(b.problems))
        for p in b.problems:
            print("  -", p)
        return 1
    return 0


sys.exit(main())
