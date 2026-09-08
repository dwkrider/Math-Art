"""Build-time metadata and per-object pages for the companion site.

    python tools/site_seo.py

Writes, under web/:

    catalog/surfaces/<slug>.html    one page per surface   (473)
    catalog/polyhedra/<slug>.html   one page per solid     (471)
    catalog/surfaces/index.html     A-Z index of each
    catalog/polyhedra/index.html
    sitemap.xml
    robots.txt

and refreshes a marked block in the <head> of index.html and the two
module pages with Open Graph, Twitter Card, canonical and JSON-LD.

WHY THE PER-OBJECT PAGES EXIST.  The interactive pages address a
selection with a fragment -- modules/surfaces.html#gyroid -- and a
fragment is not a URL to a crawler: all 473 surfaces are the same
document as far as a search engine is concerned, and the one description
that document carries has to cover every one of them. Titles written by
the viewer at selection time do not help either, since they exist only
after the module graph has run.

So each object also gets a real page at a real URL, with its metadata in
the served HTML. These are not doorway pages: each carries the record's
own properties, its definition and its sources -- the same material the
detail panel shows -- and links into the viewer for the interactive
version. The A-Z indexes give crawlers, and readers, a path to them that
the JS-rendered catalogue does not provide.

Everything here is derived from the two databases and regenerated in
place, so it is safe to re-run and pointless to hand-edit.

DESCRIPTIONS ARE TEMPLATED, NEVER PARAPHRASED.  A surface's stored
definition note is written for a reader who already knows the subject,
and is often carefully hedged about what is and is not being claimed.
Summarising it automatically would produce confident sentences the
database does not support. The summaries below are assembled only from
the categorical fields -- family, curvature, definition mode, topology,
symmetry -- which are controlled vocabularies whose values are known in
advance. The note itself is still shown, quoted verbatim, in the body.

SOCIAL IMAGES are the existing 320x320 catalogue thumbnails, so the card
type is `summary` rather than `summary_large_image`: a square tile
declared as a wide card gets letterboxed by every consumer that honours
the declaration. Generating 1200x630 cards would put ~950 more images in
LFS for nothing the square tiles do not already give.
"""
import html
import json
import os
import re
import shutil
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(PROJ, "web")
BASE = "https://dwkrider.github.io/Math-Art/"
SITE = "Math Art"
REPO = "https://github.com/dwkrider/Math-Art"


def esc(s):
    return html.escape(str(s), quote=True)


def url(rel):
    return BASE + str(rel).lstrip("/")


# --------------------------------------------------------------------
# Controlled vocabularies -> English.
#
# Every key here is a value that actually occurs in the databases. An
# unknown value falls back to the raw token rather than being dropped,
# so a newly added one reads as slightly clumsy prose instead of
# silently vanishing from the description.
# --------------------------------------------------------------------

FAMILY = {
    "minimal-periodic": "periodic minimal surface",
    "algebraic": "algebraic surface",
    "minimal": "minimal surface",
    "topological": "topological surface",
    "quadric": "quadric surface",
    "ruled": "ruled surface",
    "constant-curvature": "surface of constant curvature",
    "revolution": "surface of revolution",
    "misc": "surface",
    "swept": "swept surface",
    "cmc": "constant-mean-curvature surface",
    "physical": "physical surface model",
    "discrete": "discrete surface",
    "spectral": "spectral surface",
    "derived": "derived surface",
    "cyclide": "cyclide",
}

PERIODIC = {1: "singly periodic", 2: "doubly periodic", 3: "triply periodic"}

MODE = {
    "weierstrass": "given by a Weierstrass representation",
    "implicit": "defined by an implicit equation",
    "parametric": "given by a parametrisation",
    "nodal": "defined as a nodal algebraic surface",
    "derived": "derived from another surface in the catalogue",
}

# Phrased to follow a comma after the family, so each is an adjectival
# phrase and not a bare adjective: "a discrete surface, minimal, given
# by..." reads as a list of unrelated words.
CURVATURE = {
    "minimal": "of zero mean curvature",
    "cmc": "of constant mean curvature",
    "cmc1-bryant": "of constant mean curvature one in the Bryant sense",
    "flat": "of zero Gaussian curvature",
    "k-const-negative": "of constant negative Gaussian curvature",
    "k-const-positive": "of constant positive Gaussian curvature",
    "weingarten": "a Weingarten surface",
    "willmore": "a critical point of the Willmore energy",
}

EMBEDDING = {
    "embedded": "embedded",
    "immersed": "immersed",
    "singular": "immersed with singularities",
    "self-intersecting": "self-intersecting",
    "varies": "embedded or immersed depending on parameters",
}

SYMMETRY = {
    "space": "a crystallographic space group",
    "crystallographic": "a crystallographic group",
    "layer": "a layer group",
    "rod": "a rod group",
    "point": "a point group",
    "continuous": "a continuous symmetry group",
}

def an(phrase):
    """"a" or "an", by the sound the phrase actually starts with.

    Spelling is not the test: "an Ih-symmetric" but "a uniform", and the
    catalogue has both. Only the handful of vowel-letter words that begin
    with a consonant sound need listing.
    """
    w = phrase.split()[0].lower().strip("(")
    if w[:3] in ("uni", "eul") or w[:2] == "u-":
        return "a " + phrase
    return ("an " if w[:1] in "aeiou" else "a ") + phrase


SOLID_KIND = (
    ("platonic", "Platonic solid"),
    ("archimedean", "Archimedean solid"),
    ("catalan", "Catalan solid"),
    ("johnson", "Johnson solid"),
    ("kepler-poinsot", "Kepler-Poinsot star polyhedron"),
    ("compound", "polyhedral compound"),
    ("uniform-dual", "dual of a uniform polyhedron"),
    ("uniform", "uniform polyhedron"),
    ("zonohedron", "zonohedron"),
    ("deltahedron", "deltahedron"),
    ("toroid", "toroidal polyhedron"),
    ("geodesic", "geodesic polyhedron"),
    ("antiprism", "antiprism"),
    ("prism", "prism"),
)


def surface_summary(e):
    """One sentence about a surface, from its categorical fields only."""
    fam = FAMILY.get(e["primary_family"], e["primary_family"])
    rank = e.get("periodicity_rank") or 0
    # "triply periodic minimal surface" reads better than "periodic
    # minimal surface, triply periodic", so the rank is folded into the
    # family phrase when the family is already the periodic one.
    cur = e.get("curvature_condition")

    # The constant-curvature family names a condition the record then
    # states precisely, so the general phrase is dropped in favour of the
    # specific one rather than printing both.
    if fam == "surface of constant curvature" and cur in CURVATURE:
        fam = "surface " + CURVATURE[cur]
        cur = None

    if rank in PERIODIC and fam == "periodic minimal surface":
        lead = "%s minimal surface" % PERIODIC[rank]
    elif rank in PERIODIC:
        lead = "%s %s" % (PERIODIC[rank], fam)
    else:
        lead = fam

    bits = ["%s is %s" % (e["name"], an(lead))]

    # Minimality is already carried by the family phrase for the minimal
    # families, so repeating it there would read as a stutter.
    dup = ((cur == "minimal" and "minimal" in lead)
           or (cur == "cmc" and lead == "constant-mean-curvature surface"))
    if cur and cur != "none" and not dup:
        bits.append(CURVATURE.get(cur, cur))

    mode = MODE.get(e.get("definition_mode"))
    if mode:
        bits.append(mode)

    facts = []
    gpc = e.get("genus_per_cell")
    g = e.get("genus")
    if gpc is not None:
        facts.append("genus %s per unit cell" % gpc)
    elif g is not None:
        facts.append("genus %s" % g)
    ends = e.get("ends")
    if ends:
        facts.append("%d end%s" % (ends, "" if ends == 1 else "s"))
    emb = EMBEDDING.get(e.get("embedding"))
    if emb:
        facts.append(emb)
    if facts:
        bits.append(", ".join(facts))

    sy = e.get("symmetry") or {}
    kind = SYMMETRY.get(sy.get("kind"))
    if kind and sy.get("symbol"):
        bits.append("with %s symmetry (%s)" % (sy["symbol"], kind))
    elif kind:
        bits.append("with %s" % kind)

    return ", ".join(bits) + "."


def solid_summary(e):
    """One sentence about a polyhedron, from its counts and symmetry."""
    fams = e.get("families") or []
    kind = "convex polyhedron" if e.get("convex") else "polyhedron"
    for tag, phrase in SOLID_KIND:
        if tag in fams:
            kind = phrase
            break

    bits = ["%s is %s" % (e["name"], an(kind))]
    c = e.get("counts") or {}
    if c.get("vertices") and c.get("edges") and c.get("faces"):
        bits.append("with %d vertices, %d edges and %d faces"
                    % (c["vertices"], c["edges"], c["faces"]))
    s = e.get("symmetry") or {}
    if s.get("schoenflies"):
        sym = "symmetry group %s" % s["schoenflies"]
        if s.get("order"):
            sym += " of order %d" % s["order"]
        bits.append(sym)
    return ", ".join(bits) + "."


def noun(phrase):
    """Strip the leading article or preposition from a vocabulary phrase.

    The same strings do duty in two places: in a sentence, where they
    follow a comma and read as clauses ("..., of zero mean curvature,
    given by a parametrisation"), and in the properties table, where
    they are values and must be bare noun phrases.
    """
    if not phrase:
        return phrase
    for lead in ("of ", "given by ", "defined by ", "defined as ",
                 "derived from ", "a ", "an "):
        if phrase.startswith(lead):
            return phrase[len(lead):]
    return phrase


def clip(text, limit=300):
    """Trim to a whole word.

    Search results truncate somewhere around 155-160 characters, but the
    tail is still read by some consumers and costs nothing to keep, so
    this only guards against the genuinely long.
    """
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;") + "..."


# --------------------------------------------------------------------
# The <head> block.
#
# Written between markers so the generator owns exactly that region and
# the hand-written parts of each page are left alone. Re-running
# replaces the block rather than appending a second copy.
# --------------------------------------------------------------------

BEGIN = "<!-- seo:begin  generated by tools/site_seo.py -- do not edit -->"
END = "<!-- seo:end -->"


def head_block(*, canonical, title, desc, image, image_alt, ld, depth,
               og_type="website"):
    """Open Graph + Twitter + canonical + JSON-LD, as one indented block.

    `depth` is how far the page is below web/, so the same block can
    carry a relative stylesheet-style path if one is ever needed; the
    social tags themselves must be absolute, since a crawler fetching
    them has no page context to resolve against.
    """
    lines = [
        BEGIN,
        '<link rel="canonical" href="%s">' % esc(canonical),
        '<meta property="og:type" content="%s">' % esc(og_type),
        '<meta property="og:site_name" content="%s">' % esc(SITE),
        '<meta property="og:title" content="%s">' % esc(title),
        '<meta property="og:description" content="%s">' % esc(desc),
        '<meta property="og:url" content="%s">' % esc(canonical),
        '<meta property="og:image" content="%s">' % esc(image),
        '<meta property="og:image:width" content="320">',
        '<meta property="og:image:height" content="320">',
        '<meta property="og:image:alt" content="%s">' % esc(image_alt),
        '<meta property="og:locale" content="en_GB">',
        '<meta name="twitter:card" content="summary">',
        '<meta name="twitter:title" content="%s">' % esc(title),
        '<meta name="twitter:description" content="%s">' % esc(desc),
        '<meta name="twitter:image" content="%s">' % esc(image),
        '<meta name="twitter:image:alt" content="%s">' % esc(image_alt),
        '<script type="application/ld+json">',
        json.dumps(ld, indent=2, ensure_ascii=False),
        '</script>',
        END,
    ]
    return "\n".join(lines)


def patch_head(path, block):
    """Put the block in a page's <head>, replacing any earlier one."""
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    if BEGIN in s and END in s:
        a = s.index(BEGIN)
        b = s.index(END) + len(END)
        s = s[:a] + block + s[b:]
    else:
        s = s.replace("</head>", block + "\n</head>", 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(s)


# --------------------------------------------------------------------
# Object pages.
# --------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="../../css/site.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14' font-size='14'>&#9672;</text></svg>">
{head}
</head>
<body class="module-page">

<header class="site-head">
  <a class="brand" href="../../index.html">Math&nbsp;Art</a>
  <nav>
    <a href="../../index.html">Modules</a>
    <a href="index.html">All {plural}</a>
  </nav>
</header>

<main class="object-page">

  <div class="object-head">
{figure}
    <div class="object-lead">
      <h1>{name}</h1>
      <p class="object-summary">{summary}</p>
      <p><a class="object-open" href="{viewer}">Open {name} in the interactive viewer &rarr;</a></p>
{chips}
    </div>
  </div>

{sections}
</main>

<footer class="site-foot">
  <p>
    {foot}
    <a href="index.html">Full index of {plural}</a>.
  </p>
</footer>

<script>
// Reveal and wire the copy buttons. They ship hidden so that a reader
// without scripting sees no control that cannot work; the LaTeX itself
// is already in the page, inside each formula's MathML <annotation>,
// so nothing here is the only copy of anything.
for (const b of document.querySelectorAll('.eqn-copy')) {{
  b.hidden = false;
  b.addEventListener('click', () => {{
    navigator.clipboard?.writeText(b.dataset.latex).then(
      () => {{ b.textContent = 'copied';
              setTimeout(() => {{ b.textContent = 'copy LaTeX'; }}, 1200); }},
      () => {{ b.textContent = 'copy failed'; }});
  }});
}}
</script>
</body>
</html>
"""


def figure_html(thumb_rel, name, indent="    "):
    if not thumb_rel:
        return ('%s<div class="object-figure object-figure-blank">'
                '<span>not yet built</span></div>' % indent)
    return ('%s<img class="object-figure" src="%s" alt="%s" '
            'width="320" height="320">' % (indent, esc(thumb_rel), esc(name)))


def formulas_html(items):
    """The stored formulae, as MathML.

    MathML goes into the page as markup because it IS markup, and it is
    generated by describe.py rather than supplied by anyone. No script and
    no library: browsers lay MathML Core out natively, and vendoring a
    LaTeX renderer for this would be a large dependency on a site that
    has none. The LaTeX travels inside each formula as an <annotation>,
    which is where a copy-paste-aware reader finds it.
    """
    out = []
    many = len(items) > 1
    for f in items:
        out.append('    <figure class="eqn">')
        if many:
            out.append('      <figcaption class="eqn-label">%s</figcaption>'
                       % esc(f["label"]))
        # The relation is typeset inside the MathML, so there is nothing
        # to place beside it here.
        out.append('      <div class="eqn-math">%s</div>' % f["mathml"])
        # The LaTeX is already in the page, inside the MathML annotation;
        # this only lifts it to the clipboard. Hidden without scripting,
        # since a button that cannot do anything is worse than no button.
        out.append('      <button type="button" class="eqn-copy" hidden '
                   'data-latex="%s">copy LaTeX</button>' % esc(f["latex"]))
        out.append("    </figure>")
    return chr(10).join(out)


def section(title, body):
    if not body:
        return ""
    return ('  <section class="panel-section">\n    <h2>%s</h2>\n%s\n'
            '  </section>\n' % (esc(title), body))


def deflist(rows):
    rows = [(k, v) for k, v in rows if v not in (None, "", [])]
    if not rows:
        return ""
    out = ['    <dl class="deflist">']
    for k, v in rows:
        out.append("      <dt>%s</dt><dd>%s</dd>" % (esc(k), esc(v)))
    out.append("    </dl>")
    return "\n".join(out)


def chips(tags, indent="      "):
    if not tags:
        return ""
    inner = " ".join('<span class="tag">%s</span>' % esc(t) for t in sorted(tags))
    return '%s<p class="chips">%s</p>' % (indent, inner)


def sources_html(items):
    if not items:
        return ""
    out = ['    <ul class="sources">']
    for s in items:
        out.append("      <li>%s</li>" % esc(s))
    out.append("    </ul>")
    return "\n".join(out)


# --------------------------------------------------------------------
# Surfaces.
# --------------------------------------------------------------------

def load_surfaces():
    root = os.path.join(PROJ, "data", "surfaces")
    with open(os.path.join(root, "index.json"), encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]
    out = []
    for e in entries:
        with open(os.path.join(root, e["path"]), encoding="utf-8") as fh:
            out.append((e, json.load(fh)))
    return out


def surface_page(e, rec, meshes):
    slug = e["slug"]
    name = e["name"]
    # The record carries its own description now, built by
    # tools/surfdb/describe.py from the same vocabularies this file used
    # to duplicate. Prefer it, so the page and the detail panel cannot
    # drift apart; fall back only for a record built before that landed.
    desc = rec.get("description") or {}
    summary = desc.get("summary") or surface_summary(e)
    built = slug in meshes
    thumb = ("../../thumbs/surfaces/%s.png" % slug
             if os.path.exists(os.path.join(WEB, "thumbs", "surfaces",
                                            slug + ".png")) else None)
    image = url("thumbs/surfaces/%s.png" % slug) if thumb else url(
        "thumbs/surfaces/gyroid.png")
    canonical = url("catalog/surfaces/%s.html" % slug)

    sy = e.get("symmetry") or {}
    topo = rec.get("topology") or {}
    d = rec.get("definition") or {}

    rows = [
        ("Family", e.get("primary_family")),
        ("Given by", noun(MODE.get(e.get("definition_mode"),
                                   e.get("definition_mode")))),
        ("Curvature", noun(CURVATURE.get(e.get("curvature_condition")))
            if e.get("curvature_condition") != "none" else "no condition imposed"),
        ("Periodicity", PERIODIC.get(e.get("periodicity_rank"))
            or ("not periodic" if e.get("periodicity_rank") == 0 else None)),
        ("Genus", e.get("genus")),
        ("Genus per cell", e.get("genus_per_cell")),
        ("Ends", e.get("ends")),
        ("Embedding", EMBEDDING.get(e.get("embedding"), e.get("embedding"))),
        ("Orientable", None if e.get("orientable") is None
            else ("yes" if e["orientable"] else "no")),
        ("Symmetry", sy.get("symbol")),
        ("Symmetry kind", noun(SYMMETRY.get(sy.get("kind")))),
        ("Fidelity", d.get("fidelity")),
        ("Exactness", d.get("exactness")),
    ]

    secs = ""
    if desc.get("curated"):
        secs += section("About", '    <p class="object-prose">%s</p>'
                        % esc(desc["curated"]))
    if desc.get("formulas"):
        secs += section("Formula", formulas_html(desc["formulas"]))
    secs += section("Properties", deflist(rows))

    # The stored note, verbatim. It is the one piece of real prose the
    # database holds about a surface, and paraphrasing it is exactly
    # what this generator must not do.
    if d.get("note"):
        secs += section("Definition", '    <p class="detail-notes">%s</p>'
                        % esc(d["note"]))

    if not built:
        why = rec.get("blocked_by") or rec.get("resume")
        msg = ("This surface is catalogued but not yet built by the "
               "extension, so it has no mesh or thumbnail here.")
        if why:
            msg += " " + str(why)
        secs += section("Not yet built", '    <p class="notice">%s</p>' % esc(msg))

    prov = rec.get("provenance") or {}
    body = ""
    if prov.get("definition"):
        body += '    <p class="provenance">%s</p>\n' % esc(prov["definition"])
    body += sources_html(prov.get("sources"))
    secs += section("Sources", body)

    ld = {
        "@context": "https://schema.org",
        "@type": "CreativeWork",
        "name": name,
        "description": clip(summary),
        "url": canonical,
        "image": image,
        "isPartOf": {"@type": "Collection", "name": "Math Art surface catalogue",
                     "url": url("modules/surfaces.html")},
        "about": {"@type": "Thing", "name": "Mathematical surface"},
        "keywords": ", ".join(sorted(set(e.get("families") or []))),
        "license": REPO,
    }
    if prov.get("sources"):
        ld["citation"] = list(prov["sources"])

    head = head_block(canonical=canonical,
                      title="%s - %s" % (name, SITE),
                      desc=clip(summary, 200), image=image, image_alt=name,
                      ld=ld, depth=2, og_type="article")

    return PAGE.format(
        title=esc("%s - Surfaces - %s" % (name, SITE)),
        desc=esc(clip(summary, 200)), head=head, name=esc(name),
        summary=esc(summary), plural="surfaces",
        viewer=esc("../../modules/surfaces.html#" + slug),
        figure=figure_html(thumb, name), chips=chips(e.get("families")),
        sections=secs,
        foot=("Properties, definition and sources come from the project's "
              "surface database."))


# --------------------------------------------------------------------
# Polyhedra.
# --------------------------------------------------------------------

def load_solids():
    root = os.path.join(PROJ, "data", "polyhedra")
    with open(os.path.join(root, "index.json"), encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]
    out = []
    for e in entries:
        with open(os.path.join(root, e["path"]), encoding="utf-8") as fh:
            out.append((e, json.load(fh)))
    return out


def solid_page(e, rec, known):
    slug = e["slug"]
    name = e["name"]
    summary = solid_summary(e)
    thumb = ("../../thumbs/polyhedra/%s.png" % slug
             if os.path.exists(os.path.join(WEB, "thumbs", "polyhedra",
                                            slug + ".png")) else None)
    image = url("thumbs/polyhedra/%s.png" % slug) if thumb else url(
        "thumbs/polyhedra/truncated-icosahedron.png")
    canonical = url("catalog/polyhedra/%s.html" % slug)

    c = e.get("counts") or {}
    s = e.get("symmetry") or {}
    m = rec.get("metrics") or {}
    ids = e.get("ids") or {}

    rows = [
        ("Vertices", c.get("vertices")),
        ("Edges", c.get("edges")),
        ("Faces", c.get("faces")),
        ("Convex", "yes" if e.get("convex") else "no"),
        ("Density", e.get("density")),
        ("Schlafli symbol", rec.get("schlafli") or ids.get("schlafli")),
        ("Wythoff symbol", rec.get("wythoff") or ids.get("wythoff")),
        ("Symmetry", s.get("schoenflies")),
        ("Orbifold", s.get("orbifold")),
        ("Symmetry order", s.get("order")),
        ("Dual", e.get("dual")),
        ("Volume", m.get("volume")),
        ("Surface area", m.get("area") or m.get("surface_area")),
        ("Circumradius", m.get("circumradius")),
        ("Inradius", m.get("inradius")),
        ("Midradius", m.get("midradius")),
    ]
    secs = section("Properties", deflist(rows))

    # Ten of the hemipolyhedra name a dual that the database has no
    # record for -- the hemi duals have vertices at infinity and are not
    # catalogued as solids. The property row still states the name; only
    # the link is withheld, because a link to a page that was never
    # generated is a 404 rather than a missing feature.
    if e.get("dual") in known:
        secs += section("Dual", '    <p><a href="%s.html">%s</a></p>'
                        % (esc(e["dual"]), esc(e["dual"])))

    ext = []
    if ids.get("wikipedia"):
        ext.append('<a rel="nofollow" href="https://en.wikipedia.org/wiki/%s">'
                   'Wikipedia</a>' % esc(ids["wikipedia"]))
    if ids.get("mathworld"):
        ext.append('<a rel="nofollow" href="https://mathworld.wolfram.com/'
                   '%s.html">MathWorld</a>' % esc(ids["mathworld"]))
    if ext:
        secs += section("Elsewhere", '    <p class="ext-links">%s</p>'
                        % " &middot; ".join(ext))

    ld = {
        "@context": "https://schema.org",
        "@type": "CreativeWork",
        "name": name,
        "description": clip(summary),
        "url": canonical,
        "image": image,
        "isPartOf": {"@type": "Collection", "name": "Math Art polyhedron catalogue",
                     "url": url("modules/polyhedra.html")},
        "about": {"@type": "Thing", "name": "Polyhedron"},
        "keywords": ", ".join(sorted(set(e.get("families") or []))),
        "license": REPO,
    }

    head = head_block(canonical=canonical,
                      title="%s - %s" % (name, SITE),
                      desc=clip(summary, 200), image=image, image_alt=name,
                      ld=ld, depth=2, og_type="article")

    return PAGE.format(
        title=esc("%s - Polyhedra - %s" % (name, SITE)),
        desc=esc(clip(summary, 200)), head=head, name=esc(name),
        summary=esc(summary), plural="polyhedra",
        viewer=esc("../../modules/polyhedra.html#" + slug),
        figure=figure_html(thumb, name), chips=chips(e.get("families")),
        sections=secs,
        foot=("Counts, symmetry and metrics come from the project's "
              "polyhedron database."))


# --------------------------------------------------------------------
# The A-Z indexes.
#
# These are the crawl path. Without them the object pages are reachable
# only from sitemap.xml, and a page with no inbound link from the site
# itself reads as orphaned however correct its metadata is.
# --------------------------------------------------------------------

INDEX_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="../../css/site.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14' font-size='14'>&#9672;</text></svg>">
{head}
</head>
<body class="module-page">

<header class="site-head">
  <a class="brand" href="../../index.html">Math&nbsp;Art</a>
  <nav>
    <a href="../../index.html">Modules</a>
    <a href="{viewer}">Interactive view</a>
  </nav>
</header>

<div class="module-intro">
  <h1>{heading}</h1>
  <p>{blurb}</p>
</div>

<main class="object-index">
{groups}
</main>

<footer class="site-foot">
  <p><a href="{viewer}">Browse the same {plural} interactively</a>.</p>
</footer>

</body>
</html>
"""


def index_page(kind, plural, heading, blurb, entries, viewer, summary_of):
    """One A-Z page listing every object, grouped by initial."""
    groups = {}
    for e in entries:
        ch = e["name"][0].upper()
        groups.setdefault(ch if ch.isalpha() else "0-9", []).append(e)

    out = []
    for ch in sorted(groups):
        out.append('  <section class="index-group">')
        out.append("    <h2>%s</h2>" % esc(ch))
        out.append('    <ul class="index-list">')
        for e in sorted(groups[ch], key=lambda r: r["name"].lower()):
            out.append('      <li><a href="%s.html">%s</a></li>'
                       % (esc(e["slug"]), esc(e["name"])))
        out.append("    </ul>")
        out.append("  </section>")

    canonical = url("catalog/%s/index.html" % kind)
    desc = ("An A to Z index of all %d %s in the Math Art catalogue, each "
            "with its properties, definition and sources."
            % (len(entries), plural))
    ld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": heading,
        "description": desc,
        "url": canonical,
        "isPartOf": {"@type": "WebSite", "name": SITE, "url": BASE},
        "hasPart": [{"@type": "CreativeWork", "name": e["name"],
                     "url": url("catalog/%s/%s.html" % (kind, e["slug"]))}
                    for e in entries],
    }
    head = head_block(canonical=canonical, title="%s - %s" % (heading, SITE),
                      desc=desc,
                      image=url("thumbs/surfaces/gyroid.png" if kind == "surfaces"
                                else "thumbs/polyhedra/truncated-icosahedron.png"),
                      image_alt=heading, ld=ld, depth=2)

    return INDEX_PAGE.format(
        title=esc("%s - %s" % (heading, SITE)), desc=esc(desc), head=head,
        heading=esc(heading), blurb=esc(blurb), plural=esc(plural),
        viewer=esc(viewer), groups="\n".join(out))


# --------------------------------------------------------------------
# sitemap.xml and robots.txt
# --------------------------------------------------------------------

def sitemap(urls):
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for loc, prio in urls:
        out.append("  <url>")
        out.append("    <loc>%s</loc>" % esc(loc))
        out.append("    <priority>%.1f</priority>" % prio)
        out.append("  </url>")
    out.append("</urlset>")
    return "\n".join(out) + "\n"


ROBOTS = """# %s -- companion site
#
# NOTE ON WHERE THIS FILE LIVES.  Crawlers read robots.txt only from the
# root of a host: https://dwkrider.github.io/robots.txt, which belongs to
# a different repository. This site is a project site under /Math-Art/,
# so THIS file is not consulted, and the Sitemap line below will not be
# discovered from it. Submit the sitemap directly instead (Google Search
# Console, Bing Webmaster Tools), or copy the Sitemap line into the
# root site's robots.txt.
#
# It is still written, because it costs nothing, documents the intent in
# one obvious place, and becomes live the moment the site is served from
# a domain root or a custom domain.
#
# Everything here is public and static. The only exclusions are payload
# directories: mesh JSON, the vendored three.js build and the sprite
# sheet are fetched by the pages themselves and are not documents a
# search result should ever land on.

User-agent: *
Allow: /
Disallow: /Math-Art/surfaces/
Disallow: /Math-Art/vendor/
Disallow: /Math-Art/data/

Sitemap: %ssitemap.xml
""" % (SITE, BASE)


# --------------------------------------------------------------------
# The three hand-written pages.
# --------------------------------------------------------------------

def patch_site_pages(n_surf, n_solid):
    home_ld = [
        {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": SITE,
            "url": BASE,
            "description": ("An interactive companion to the Math Art "
                            "project: polyhedra, surfaces, patterns, knots "
                            "and fractals, computed live in the browser."),
            "inLanguage": "en",
        },
        {
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": SITE,
            "applicationCategory": "https://schema.org/MultimediaApplication",
            "operatingSystem": "Any modern web browser",
            "url": BASE,
            "codeRepository": REPO,
            "description": ("A Blender extension and companion website for "
                            "generating mathematical art: %d polyhedra, %d "
                            "surfaces, patterns, knots and fractals."
                            % (n_solid, n_surf)),
            "offers": {"@type": "Offer", "price": "0",
                       "priceCurrency": "USD"},
        },
    ]
    patch_head(os.path.join(WEB, "index.html"), head_block(
        canonical=BASE, title="%s - interactive mathematical geometry" % SITE,
        desc=("An interactive companion to the Math Art project: %d polyhedra "
              "and %d surfaces with exact geometry, drawn live in your "
              "browser." % (n_solid, n_surf)),
        image=url("thumbs/surfaces/gyroid.png"),
        image_alt="The gyroid, a triply periodic minimal surface",
        ld=home_ld, depth=0))

    patch_head(os.path.join(WEB, "modules", "surfaces.html"), head_block(
        canonical=url("modules/surfaces.html"),
        title="Surfaces - %s" % SITE,
        desc=("Browse %d surfaces -- minimal, periodic minimal, algebraic, "
              "quadric, constant-curvature, ruled and topological -- with "
              "their definitions, curvature, topology and symmetry." % n_surf),
        image=url("thumbs/surfaces/gyroid.png"),
        image_alt="The gyroid, a triply periodic minimal surface",
        ld={
            "@context": "https://schema.org",
            "@type": "CollectionPage",
            "name": "Surfaces",
            "url": url("modules/surfaces.html"),
            "description": "An interactive catalogue of %d mathematical "
                           "surfaces." % n_surf,
            "isPartOf": {"@type": "WebSite", "name": SITE, "url": BASE},
        }, depth=1))

    patch_head(os.path.join(WEB, "modules", "polyhedra.html"), head_block(
        canonical=url("modules/polyhedra.html"),
        title="Polyhedra - %s" % SITE,
        desc=("Browse %d polyhedra -- Platonic, Archimedean, Catalan, the "
              "Johnson solids, Kepler-Poinsot stars, uniform polyhedra, "
              "compounds and zonohedra -- with their symmetry groups and "
              "exact metrics." % n_solid),
        image=url("thumbs/polyhedra/truncated-icosahedron.png"),
        image_alt="The truncated icosahedron",
        ld={
            "@context": "https://schema.org",
            "@type": "CollectionPage",
            "name": "Polyhedra",
            "url": url("modules/polyhedra.html"),
            "description": "An interactive catalogue of %d polyhedra."
                           % n_solid,
            "isPartOf": {"@type": "WebSite", "name": SITE, "url": BASE},
        }, depth=1))


def write(path, text):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main():
    surfaces = load_surfaces()
    solids = load_solids()

    with open(os.path.join(WEB, "surface-meshes.json"), encoding="utf-8") as fh:
        meshes = set(json.load(fh).get("meshes") or [])

    # Wiped, not merged: a record renamed or dropped upstream would
    # otherwise leave an orphan page that stays in the index for months.
    for d in ("catalog/surfaces", "catalog/polyhedra"):
        p = os.path.join(WEB, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p)

    for e, rec in surfaces:
        write(os.path.join(WEB, "catalog", "surfaces", e["slug"] + ".html"),
              surface_page(e, rec, meshes))
    known = {e["slug"] for e, _ in solids}
    for e, rec in solids:
        write(os.path.join(WEB, "catalog", "polyhedra", e["slug"] + ".html"),
              solid_page(e, rec, known))

    write(os.path.join(WEB, "catalog", "surfaces", "index.html"),
          index_page("surfaces", "surfaces", "All surfaces",
                     "Every surface in the catalogue, A to Z. Each links to "
                     "its properties, definition and sources, and from there "
                     "into the interactive viewer.",
                     [e for e, _ in surfaces], "../../modules/surfaces.html",
                     surface_summary))
    write(os.path.join(WEB, "catalog", "polyhedra", "index.html"),
          index_page("polyhedra", "polyhedra", "All polyhedra",
                     "Every solid in the catalogue, A to Z. Each links to its "
                     "counts, symmetry and metrics, and from there into the "
                     "interactive viewer.",
                     [e for e, _ in solids], "../../modules/polyhedra.html",
                     solid_summary))

    urls = [(BASE, 1.0),
            (url("modules/polyhedra.html"), 0.9),
            (url("modules/surfaces.html"), 0.9),
            (url("catalog/polyhedra/index.html"), 0.7),
            (url("catalog/surfaces/index.html"), 0.7)]
    urls += [(url("catalog/surfaces/%s.html" % e["slug"]), 0.5)
             for e, _ in surfaces]
    urls += [(url("catalog/polyhedra/%s.html" % e["slug"]), 0.5)
             for e, _ in solids]
    write(os.path.join(WEB, "sitemap.xml"), sitemap(urls))
    write(os.path.join(WEB, "robots.txt"), ROBOTS)

    patch_site_pages(len(surfaces), len(solids))

    print("catalog/surfaces  : %d pages + index" % len(surfaces))
    print("catalog/polyhedra : %d pages + index" % len(solids))
    print("sitemap.xml       : %d URLs" % len(urls))
    print("robots.txt        : written")
    print("head blocks       : index.html, modules/surfaces.html, "
          "modules/polyhedra.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
