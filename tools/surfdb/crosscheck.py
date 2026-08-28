# Cross-checking records against the local Ferreol mirror.
#
# A record derived from this repo's own generators and cited to the
# literature is still a SINGLE derivation.  What raises it above that is
# independent confirmation, which is why `data/polyhedra` carries a
# `cross_checked` block and reports 270 of 272 checkable records
# confirmed against McCooey -- including, importantly, the two that
# DISAGREE.  Those two are recorded honestly rather than reconciled, and
# that is the standard this module works to.
#
# The mirror is local: S:/data/math_art/references/websites/mathcurve/,
# 181 surface chapters, already resolved by `ids.mathcurve`.  Nothing is
# fetched; nothing is republished.  Ferreol's text is an ORACLE -- read to
# confirm, cited, never copied into a record.
#
# What is compared, and only what can be compared honestly:
#
#   identity   the page's title corroborates the record's name
#   year       "studied by X in YYYY" against `year`
#   curvature  the page's own words ("minimal surface", "constant mean
#              curvature", "zero mean curvature") against `condition`
#
# Anything the page does not state is reported as "not stated", never as
# agreement.  Silence is not confirmation.

import os
import re

from . import sources

_YEAR = re.compile(r"\b(?:studied|discovered|described|introduced|considered)"
                   r"\b[^.]{0,80}?\bin\s+(\d{4})", re.I)
_YEAR2 = re.compile(r"\bby\s+[A-Z][A-Za-z.\- ]{2,30}?\s+in\s+(\d{4})")

_CURVATURE_WORDS = {
    "minimal": [r"minimal surface", r"zero mean curvature",
                r"mean curvature\s+(?:is\s+)?(?:zero|null)"],
    "cmc": [r"constant mean curvature"],
    "k-const-negative": [r"constant negative (?:total |Gaussian )?curvature",
                         r"pseudospherical",
                         r"curvature\s*=?\s*-1"],
    "k-const-positive": [r"constant positive (?:total |Gaussian )?curvature"],
    "flat": [r"developable", r"zero Gaussian curvature",
             r"Gaussian curvature\s+(?:is\s+)?(?:zero|null)"],
}


def page_path(stem):
    return os.path.join(sources.SURFACES, stem + ".md")


def read_page(stem, limit=60000):
    p = page_path(stem)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read(limit)


def _title(text):
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return (m.group(1).strip() if m else "")


def _norm(t):
    """Lowercase words, with hyphenation collapsed.

    Ferreol writes "Sea-shell" where the record says "Seashell"; without
    collapsing the hyphen the titles share no word and a correct id is
    reported as pointing at the wrong chapter.
    """
    t = re.sub(r"[-‐-―]", "", (t or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def check(rec, text):
    """Compare one record against its mirror page.

    Returns a list of `cross_checked` entries -- one per property
    actually comparable.  A property the page does not mention produces
    no entry rather than a false confirmation.
    """
    out = []
    stem = (rec.get("ids") or {}).get("mathcurve")

    # identity
    title = _title(text)
    name_words = set(_norm(rec.get("name", "")).split())
    title_words = set(_norm(title).split())
    common = name_words & title_words
    strong = bool(common - {"surface", "the", "of", "a"})
    out.append({
        "source": "mathcurve:%s" % stem,
        "agrees": bool(strong),
        "detail": "page title %r vs record name %r%s"
                  % (title, rec.get("name"),
                     "" if strong else
                     " -- no distinctive word in common; the ID may be "
                     "pointing at the wrong chapter"),
    })

    # discovery year
    year = rec.get("year")
    if isinstance(year, int):
        found = None
        for pat in (_YEAR, _YEAR2):
            m = pat.search(text)
            if m:
                found = int(m.group(1))
                break
        if found:
            agree = abs(found - year) <= 1
            out.append({
                "source": "mathcurve:%s" % stem,
                "agrees": agree,
                "detail": "record year %d, page says %d%s"
                          % (year, found,
                             "" if agree else
                             " -- recorded as a disagreement rather than "
                             "reconciled; sources differ on when the surface "
                             "was first studied"),
            })

    # curvature condition, in the page's own words
    cond = (rec.get("curvature") or {}).get("condition")
    if cond in _CURVATURE_WORDS:
        hit = any(re.search(p, text, re.I) for p in _CURVATURE_WORDS[cond])
        if hit:
            out.append({
                "source": "mathcurve:%s" % stem,
                "agrees": True,
                "detail": "page independently describes this as %r" % cond,
            })
        else:
            # the page may simply not discuss curvature; only flag when it
            # asserts a DIFFERENT condition
            # A single passing mention is not a contradiction. These
            # pages cross-reference constantly -- the helicoid chapter
            # says "developable" once, about the DEVELOPABLE helicoid,
            # which is a different surface -- so a contradiction has to
            # be repeated before it counts. Silence stays "not stated".
            def _count(pats):
                return sum(len(re.findall(p, text, re.I)) for p in pats)
            others = [c for c, pats in _CURVATURE_WORDS.items()
                      if c != cond and _count(pats) >= 2]
            if others:
                out.append({
                    "source": "mathcurve:%s" % stem,
                    "agrees": False,
                    "detail": "record says %r but the page describes %s"
                              % (cond, " and ".join(repr(o) for o in others)),
                })
    return out


def run(records, verbose=False):
    """Cross-check every record carrying a resolvable mathcurve ID.

    `records` is slug -> record; records are updated in place. Returns
    (checked, agreements, disagreements, skipped).
    """
    checked = agree = disagree = skipped = 0
    for slug, rec in sorted(records.items()):
        stem = (rec.get("ids") or {}).get("mathcurve")
        if not stem:
            skipped += 1
            continue
        text = read_page(stem)
        if text is None:
            skipped += 1
            continue
        entries = check(rec, text)
        if not entries:
            skipped += 1
            continue
        rec.setdefault("provenance", {})["cross_checked"] = entries
        checked += 1
        for e in entries:
            if e["agrees"]:
                agree += 1
            else:
                disagree += 1
                if verbose:
                    print("  DISAGREE %-30s %s" % (slug, e["detail"]))
    return checked, agree, disagree, skipped


def _selftest():
    """Checks the comparison logic on synthetic pages; raises on failure."""
    page = ("# Catenoid\n\n"
            "CATENOID\n\nSurface studied by Euler in 1740.\n"
            "The catenoid is a minimal surface of revolution.\n")

    rec = {"name": "Catenoid", "year": 1744,
           "ids": {"mathcurve": "ch1198_catenoid_2"},
           "curvature": {"condition": "minimal"}}
    ents = check(rec, page)
    kinds = {e["detail"].split()[0] for e in ents}
    assert any("title" in e["detail"] for e in ents)

    # identity must agree
    assert ents[0]["agrees"] is True, ents[0]

    # the year DISAGREES (1744 vs 1740) and must be reported, not hidden
    yr = [e for e in ents if "page says" in e["detail"]]
    assert yr and yr[0]["agrees"] is False, yr
    assert "disagreement" in yr[0]["detail"]

    # the curvature condition is independently confirmed
    cv = [e for e in ents if "independently describes" in e["detail"]]
    assert cv and cv[0]["agrees"] is True, cv

    # a page that says nothing about curvature must NOT count as agreement
    quiet = "# Something\n\nA surface with no curvature discussion.\n"
    rec2 = {"name": "Something", "ids": {"mathcurve": "x"},
            "curvature": {"condition": "minimal"}}
    ents2 = check(rec2, quiet)
    assert not any("independently describes" in e["detail"] for e in ents2), \
        "silence must never be recorded as confirmation"

    # a page that REPEATEDLY asserts a different condition must be flagged
    wrong = ("# Thing\n\nThis is a surface of constant mean curvature.\n"
             "Every constant mean curvature surface of this kind is closed.\n")
    rec3 = {"name": "Thing", "ids": {"mathcurve": "x"},
            "curvature": {"condition": "minimal"}}
    ents3 = check(rec3, wrong)
    assert any(e["agrees"] is False and "but the page describes" in e["detail"]
               for e in ents3), ents3

    # ...but ONE passing mention must NOT be. This is the real case that
    # produced a false positive: Ferreol's helicoid chapter says
    # "developable" once, about the DEVELOPABLE helicoid, which is a
    # different surface. Cross-references are normal on these pages, and
    # treating one as a contradiction contradicts a correct record.
    passing = ("# Helicoid\n\nThe helicoid is ruled.\n"
               "See also the developable helicoid.\n")
    rec3b = {"name": "Helicoid", "ids": {"mathcurve": "x"},
             "curvature": {"condition": "minimal"}}
    ents3b = check(rec3b, passing)
    assert not any("but the page describes" in e["detail"] for e in ents3b), \
        "a single cross-reference must not contradict the record: %r" % ents3b

    # hyphenation must not break identity: Ferreol writes "Sea-shell"
    shell = "# Sea-shell\n\nA shell surface.\n"
    rec3c = {"name": "Seashell", "ids": {"mathcurve": "x"}, "curvature": {}}
    assert check(rec3c, shell)[0]["agrees"] is True, check(rec3c, shell)

    # a mismatched title must be flagged, not silently accepted
    rec4 = {"name": "Gyroid", "ids": {"mathcurve": "ch1198_catenoid_2"},
            "curvature": {"condition": "minimal"}}
    ents4 = check(rec4, page)
    assert ents4[0]["agrees"] is False, ents4[0]
    del kinds

    print("RESULT: OK  (surfdb.crosscheck)")
