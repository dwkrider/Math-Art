# Companion-site gate -- pure Python, no Blender and no browser needed.
#
# The site under web/ has no build step, which is what makes it cheap to
# work on and also what removes the safety net a bundler would give: a
# renamed file, a mistyped import or a solid with no thumbnail all fail
# silently in the browser, and only for the reader who happens to open
# that page.  This is the substitute.
#
#     python tests/test_web.py
#     python tests/test_web.py --quiet      only the failures
#
# What it enforces:
#
#   coverage    every solid has a thumbnail and every thumbnail a solid;
#               every IMPLEMENTED surface has a baked mesh and a tile.
#               An orphan either way means a slug was renamed and the
#               artifact was left behind
#   links       every local href/src in the HTML resolves on disk, and
#               every relative import in the JS resolves too
#   importmap   the bare specifiers the modules import are declared, and
#               point at files that exist
#   seo         every database record has a generated catalog page, every
#               page has a record, the JSON-LD parses, and sitemap.xml
#               agrees with what is on disk.  This catches the tool not
#               having been re-run after a database change
#   offline     nothing fetches from a third-party host at runtime.  The
#               site is meant to compute everything locally, so a CDN
#               reference is a correctness bug, not a style choice
#
# `node --check` is used to syntax-check the modules when node is
# available and skipped with a note when it is not, so the gate still
# runs on a machine with no toolchain at all.
import json
import os
import re
import shutil
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(PROJ, "web")
THUMBS = os.path.join(WEB, "thumbs", "polyhedra")
DB = os.path.join(PROJ, "data", "polyhedra")
SURF_DB = os.path.join(PROJ, "data", "surfaces")
SURF_THUMBS = os.path.join(WEB, "thumbs", "surfaces")
SURF_MESHES = os.path.join(WEB, "surfaces")

# Hosts are never contacted at runtime; the only absolute URLs allowed
# are ones a reader clicks.
# dwkrider.github.io is the site's own origin. It appears in the
# canonical link and the Open Graph tags, which the specifications
# require to be absolute; nothing fetches them, so they are not the
# remote dependency this check is looking for.
ALLOWED_LINK_HOSTS = ("github.com", "en.wikipedia.org",
                      "mathworld.wolfram.com", "dwkrider.github.io")


def _index():
    with open(os.path.join(DB, "index.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _web_files(ext):
    out = []
    for dirpath, dirs, files in os.walk(WEB):
        dirs[:] = [d for d in dirs if d not in ("vendor", "thumbs", "data")]
        for fn in files:
            if fn.endswith(ext):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def check_thumbnails(fail):
    entries = _index()["entries"]
    slugs = {e["slug"] for e in entries}
    have = {fn[:-4] for fn in os.listdir(THUMBS) if fn.endswith(".png")} \
        if os.path.isdir(THUMBS) else set()

    missing = sorted(slugs - have)
    for s in missing:
        fail("no thumbnail for %s (run tools/render_polyhedra_thumbs.py)" % s)
    for s in sorted(have - slugs):
        fail("orphan thumbnail web/thumbs/polyhedra/%s.png names no solid" % s)
    return len(slugs), len(have), len(missing)


# Surfaces that are implemented but cannot be baked, with the reason.
# Same idea as PANEL_ONLY in tests/test_extension.py: an exception has to
# be written down and justified, not silently tolerated. Anything here
# still appears in the catalogue with its mathematics; it just has no
# mesh, exactly like a record no generator implements.
UNDRIVEABLE = {
    "plateau-span":
        "object.minimal_span spans the curves that are SELECTED, so it "
        "cannot be driven from an empty scene -- its poll() fails. It is "
        "a tool applied to a user's own boundary, not a surface with a "
        "canonical shape to bake.",
}


def check_surfaces(fail):
    """Every surface a generator implements must have a mesh and a tile.

    Scoped to `implemented`, because 61 records describe surfaces no
    generator builds: those legitimately have neither, and the module
    says so rather than showing an empty stage as a failure.
    """
    path = os.path.join(SURF_DB, "index.json")
    if not os.path.exists(path):
        fail("no surface database at %s" % SURF_DB)
        return 0, 0
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]
    want = {e["slug"] for e in entries
            if e.get("implemented") and e["slug"] not in UNDRIVEABLE}
    all_slugs = {e["slug"] for e in entries}

    meshes = {fn[:-5] for fn in os.listdir(SURF_MESHES)
              if fn.endswith(".json")} if os.path.isdir(SURF_MESHES) else set()
    thumbs = {fn[:-4] for fn in os.listdir(SURF_THUMBS)
              if fn.endswith(".png")} if os.path.isdir(SURF_THUMBS) else set()

    for s in sorted(want - meshes):
        fail("no mesh for %s (run tools/surfdb_export.py)" % s)
    for s in sorted(want - thumbs):
        fail("no thumbnail for %s (run tools/surfdb_export.py)" % s)
    # An artifact for a slug the database does not know is a rename that
    # left its files behind.
    for s in sorted(meshes - all_slugs):
        fail("orphan mesh web/surfaces/%s.json names no surface" % s)
    for s in sorted(thumbs - all_slugs):
        fail("orphan thumbnail web/thumbs/surfaces/%s.png names no surface" % s)

    # The clustered view draws from a sprite sheet, so a stale atlas shows
    # the wrong surface under the right name -- a silent, plausible-looking
    # error. It must list exactly the tiles on disk.
    apath = os.path.join(WEB, "thumbs", "surfaces-atlas.json")
    if not os.path.exists(apath):
        fail("no web/thumbs/surfaces-atlas.json "
             "(run tools/build_thumb_atlas.py)")
    else:
        with open(apath, encoding="utf-8") as fh:
            atlas = json.load(fh)
        listed = set(atlas.get("tiles") or {})
        if not os.path.exists(os.path.join(WEB, "thumbs",
                                           "surfaces-atlas.png")):
            fail("surfaces-atlas.json exists but the .png does not")
        for s in sorted(listed - thumbs):
            fail("atlas lists %s, which has no tile" % s)
        for s in sorted(thumbs - listed):
            fail("%s has a tile but the atlas omits it "
                 "(re-run tools/build_thumb_atlas.py)" % s)

    # The page reads this manifest instead of guessing from `implemented`,
    # so a stale one would mislabel tiles. It has to agree with the disk.
    mpath = os.path.join(WEB, "surface-meshes.json")
    if not os.path.exists(mpath):
        fail("no web/surface-meshes.json (run tools/surfdb_export.py)")
    else:
        with open(mpath, encoding="utf-8") as fh:
            listed = set(json.load(fh).get("meshes") or [])
        for s in sorted(listed - meshes):
            fail("surface-meshes.json lists %s, which has no mesh file" % s)
        for s in sorted(meshes - listed):
            fail("%s has a mesh but surface-meshes.json omits it" % s)
    return len(want), len(meshes)


def check_html_links(fail):
    checked = 0
    for path in _web_files(".html"):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        base = os.path.dirname(path)
        for m in re.finditer(r'(?:href|src)="([^"]+)"', src):
            url = m.group(1)
            if url.startswith("#") or url.startswith("data:"):
                continue
            if url.startswith(("http://", "https://")):
                if not any(h in url for h in ALLOWED_LINK_HOSTS):
                    fail("%s references third-party host: %s"
                         % (os.path.relpath(path, PROJ), url))
                continue
            # A fragment addresses a place inside the target, not a
            # different file: catalog pages link to modules/surfaces.html
            # #gyroid, and the file to test for is the part before the #.
            rel = url.split("#")[0].split("?")[0]
            if not rel:
                continue
            target = os.path.normpath(os.path.join(base, rel))
            checked += 1
            if not os.path.exists(target):
                fail("%s -> %s does not exist"
                     % (os.path.relpath(path, PROJ), url))
    return checked


def check_import_map(fail):
    """Bare specifiers in the modules must be declared and must resolve."""
    page = os.path.join(WEB, "modules", "polyhedra.html")
    with open(page, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'<script type="importmap">\s*(\{.*?\})\s*</script>',
                  src, re.S)
    if not m:
        fail("no import map in web/modules/polyhedra.html")
        return 0
    imports = json.loads(m.group(1))["imports"]
    base = os.path.dirname(page)
    for spec, target in imports.items():
        # A trailing-slash mapping is a prefix; check the directory.
        probe = target if not target.endswith("/") else target
        full = os.path.normpath(os.path.join(base, probe))
        if not os.path.exists(full):
            fail("import map %r -> %s does not exist" % (spec, target))

    # Every bare specifier the JS actually imports must be covered.
    prefixes = [s for s in imports if s.endswith("/")]
    for path in _web_files(".js"):
        with open(path, encoding="utf-8") as fh:
            js = fh.read()
        for spec in re.findall(r'^\s*import[^\'"]*[\'"]([^\'"]+)[\'"]', js,
                               re.M):
            if spec.startswith((".", "/")):
                continue
            if spec in imports:
                continue
            if any(spec.startswith(p) for p in prefixes):
                continue
            fail("%s imports %r, which the import map does not declare"
                 % (os.path.relpath(path, PROJ), spec))
    return len(imports)


def check_js_relative_imports(fail):
    checked = 0
    for path in _web_files(".js"):
        base = os.path.dirname(path)
        with open(path, encoding="utf-8") as fh:
            js = fh.read()
        for spec in re.findall(r'(?:^\s*import[^\'"]*|import\()[\'"]'
                               r'(\.[^\'"]+)[\'"]', js, re.M):
            target = os.path.normpath(os.path.join(base, spec))
            checked += 1
            if not os.path.exists(target):
                fail("%s imports %r, which does not exist"
                     % (os.path.relpath(path, PROJ), spec))
    return checked


def check_no_remote_fetch(fail):
    """The site computes locally; nothing may be fetched from a host."""
    pat = re.compile(r'(?:fetch|importScripts|new\s+Worker)\s*\(\s*[\'"`]'
                     r'(https?://[^\'"`]+)')
    for path in _web_files(".js") + _web_files(".css"):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in pat.finditer(text):
            fail("%s fetches from %s at runtime"
                 % (os.path.relpath(path, PROJ), m.group(1)))
        for m in re.finditer(r'@import\s+url\(\s*[\'"]?(https?://[^\)\'"]+)',
                             text):
            fail("%s imports remote CSS: %s"
                 % (os.path.relpath(path, PROJ), m.group(1)))


def check_js_syntax(fail, quiet):
    node = shutil.which("node")
    if not node:
        if not quiet:
            print("  node not found -- JS syntax not checked")
        return 0
    n = 0
    for path in _web_files(".js"):
        r = subprocess.run([node, "--check", path],
                           capture_output=True, text=True)
        n += 1
        if r.returncode != 0:
            fail("%s fails node --check: %s"
                 % (os.path.relpath(path, PROJ),
                    (r.stderr or "").strip().splitlines()[:1]))
    return n


def check_layout_tracks(fail):
    """The viewer's column may not be sized by what is inside it.

    A bare `1fr` is `minmax(auto, 1fr)`, and that auto minimum lets a
    wide child stretch the track. It happened: a long implicit equation
    in the detail panel widened the workspace column, the stage stretched
    to match, and the surface -- drawn centred in the canvas -- ended up
    off the right-hand edge and out of view.

    Only the containers on that path are checked. Elsewhere a bare 1fr is
    ordinary and correct.
    """
    css = os.path.join(WEB, "css", "site.css")
    if not os.path.exists(css):
        return 0
    with open(css, encoding="utf-8") as fh:
        text = fh.read()
    checked = 0
    for sel in (".module-layout", ".workspace"):
        for m in re.finditer(re.escape(sel) + r"[^{]*\{([^}]*)\}", text):
            body = m.group(1)
            gtc = re.search(r"grid-template-columns:([^;]*);", body)
            if not gtc:
                continue
            checked += 1
            tracks = gtc.group(1)
            if "1fr" in tracks and "minmax(0" not in tracks:
                fail("%s uses a bare 1fr (%s) -- it must be minmax(0, 1fr), "
                     "or a wide equation stretches the viewer"
                     % (sel, tracks.strip()))
    return checked


def check_lfs_deploy(fail):
    """The deploy pulls every LFS path the site serves.

    This is the one failure the rest of this file cannot see. Locally
    every LFS file is smudged, so the meshes and tiles are real and each
    check here passes; on the deployed site they are 130-byte pointer
    files unless the workflow asked for them by name. It served 466
    pointer meshes that way -- the tiles were pulled and looked fine, so
    the catalogue seemed healthy and only the 3-D view was empty, on a
    green build.

    So the check is static: whatever .gitattributes marks as LFS under
    web/ has to appear in the workflow's --include list.
    """
    ga = os.path.join(PROJ, ".gitattributes")
    wf = os.path.join(PROJ, ".github", "workflows", "pages.yml")
    if not (os.path.exists(ga) and os.path.exists(wf)):
        return 0
    with open(ga, encoding="utf-8") as fh:
        tracked = [ln.split()[0] for ln in fh
                   if "filter=lfs" in ln and ln.strip()
                   and ln.split()[0].startswith("web/")]
    with open(wf, encoding="utf-8") as fh:
        flow = fh.read()
    m = re.search(r'lfs pull --include="([^"]+)"', flow)
    if not m:
        fail("pages.yml has no `git lfs pull --include=` -- the site would "
             "deploy every LFS file as a pointer")
        return 0
    inc = [x.strip() for x in m.group(1).split(",")]
    for pat in tracked:
        top = pat.split("/")[0] + "/" + pat.split("/")[1]      # web/<dir>
        if not any(i.startswith(top) for i in inc):
            fail("%s is LFS-tracked but pages.yml does not pull it; it "
                 "would deploy as pointer files" % pat)
    return len(tracked)


def check_seo(fail):
    """The generated metadata, pages and sitemap are present and agree.

    tools/site_seo.py derives all of this from the two databases, so the
    failure this guards against is the tool not having been re-run: a
    record added upstream then has no page, and sitemap.xml advertises a
    set of URLs that no longer matches what is on disk. Both are silent
    -- the site looks fine and the missing object is simply absent from
    search -- which is exactly the kind of drift a gate is for.
    """
    n_pages = 0

    # Every record in each database has a page, and every page has a record.
    for kind, db, key in (("surfaces", os.path.join(PROJ, "data", "surfaces"),
                           "surfaces"),
                          ("polyhedra", DB, "polyhedra")):
        with open(os.path.join(db, "index.json"), encoding="utf-8") as fh:
            want = {e["slug"] for e in json.load(fh)["entries"]}
        d = os.path.join(WEB, "catalog", kind)
        if not os.path.isdir(d):
            fail("web/catalog/%s/ is missing -- run tools/site_seo.py" % kind)
            continue
        have = {f[:-5] for f in os.listdir(d)
                if f.endswith(".html") and f != "index.html"}
        n_pages += len(have)
        for s in sorted(want - have):
            fail("%s has no catalog page (run tools/site_seo.py): %s"
                 % (kind, s))
        for s in sorted(have - want):
            fail("%s catalog page has no database record: %s" % (kind, s))
        if not os.path.exists(os.path.join(d, "index.html")):
            fail("web/catalog/%s/index.html is missing" % kind)

    # The three hand-written pages carry a generated head block.
    for rel in ("index.html", "modules/surfaces.html", "modules/polyhedra.html"):
        p = os.path.join(WEB, *rel.split("/"))
        with open(p, encoding="utf-8") as fh:
            src = fh.read()
        for tag in ('rel="canonical"', 'property="og:title"',
                    'name="twitter:card"', 'application/ld+json'):
            if tag not in src:
                fail("web/%s carries no %s" % (rel, tag))

    # Every JSON-LD block parses. A malformed one is ignored silently by
    # every consumer, so nothing would ever report it.
    for path in _web_files(".html"):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        for m in re.finditer(r'<script type="application/ld\+json">(.*?)</script>',
                             src, re.S):
            try:
                json.loads(m.group(1))
            except ValueError as exc:
                fail("%s has unparseable JSON-LD: %s"
                     % (os.path.relpath(path, PROJ), exc))
                break

    # Every surface record carries a description, and every formula on a
    # page is well-formed MathML.
    #
    # Malformed MathML is the failure worth guarding: a browser given
    # broken markup does not report anything, it builds unknown elements
    # that lay out as inline text, so a wrong equation appears as a run
    # of loose letters and numbers and looks like a styling problem.
    from xml.etree import ElementTree as ET
    sdb = os.path.join(PROJ, "data", "surfaces")
    with open(os.path.join(sdb, "index.json"), encoding="utf-8") as fh:
        sidx = json.load(fh)["entries"]
    missing = 0
    for e in sidx:
        with open(os.path.join(sdb, e["path"]), encoding="utf-8") as fh:
            d = (json.load(fh) or {}).get("description") or {}
        if not d.get("summary"):
            missing += 1
    if missing:
        fail("%d surface record(s) carry no description -- run "
             "tools/surfdb_build.py" % missing)

    n_math = bad_math = 0
    for path in _web_files(".html"):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        for m in re.finditer("<math" + chr(92) + "b.*?</math>", src, re.S):
            n_math += 1
            try:
                ET.fromstring(m.group(0))
            except ET.ParseError as exc:
                bad_math += 1
                if bad_math <= 3:
                    fail("%s has malformed MathML: %s"
                         % (os.path.relpath(path, PROJ), exc))

    # sitemap.xml lists only URLs that exist on disk, and lists them all.
    sm = os.path.join(WEB, "sitemap.xml")
    n_urls = 0
    if not os.path.exists(sm):
        fail("web/sitemap.xml is missing -- run tools/site_seo.py")
    else:
        with open(sm, encoding="utf-8") as fh:
            body = fh.read()
        locs = re.findall(r"<loc>([^<]+)</loc>", body)
        n_urls = len(locs)
        listed = set()
        for loc in locs:
            rel = loc.split("github.io/Math-Art/", 1)[-1] or "index.html"
            listed.add(rel)
            p = os.path.join(WEB, *rel.split("/"))
            if not os.path.exists(p):
                fail("sitemap.xml lists %s, which is not on disk" % rel)
        for path in _web_files(".html"):
            rel = os.path.relpath(path, WEB).replace(os.sep, "/")
            if rel == "index.html":
                continue
            if rel not in listed:
                fail("%s is not listed in sitemap.xml" % rel)

    # robots.txt has to point at the sitemap, or it is decoration.
    rb = os.path.join(WEB, "robots.txt")
    if not os.path.exists(rb):
        fail("web/robots.txt is missing -- run tools/site_seo.py")
    else:
        with open(rb, encoding="utf-8") as fh:
            body = fh.read()
        if "Sitemap:" not in body:
            fail("web/robots.txt names no Sitemap")
        if "sitemap.xml" not in body:
            fail("web/robots.txt does not point at sitemap.xml")

    return n_pages, n_urls, n_math


def check_cluster_cache(fail, quiet):
    """Run the browser modules' own headless checks, if node is here.

    The layout cache and the tile gate are behaviours no other check in
    this file can see: a broken cache still draws the right picture, just
    slowly, and a broken gate draws it in the wrong order. Both need a
    stopwatch and a stubbed DOM, which is what tests/web/ provides.
    """
    node = shutil.which("node")
    tests = [os.path.join(PROJ, "tests", "web", n) for n in
             ("test_cluster_cache.mjs", "test_stl_export.mjs")]
    for t in tests:
        if not os.path.exists(t):
            fail("%s is missing" % os.path.relpath(t, PROJ))
            return False
    test = tests[0]
    if not node:
        if not quiet:
            print("cluster    : skipped (node not on PATH)")
        return True
    good = True
    for t in tests:
        r = subprocess.run([node, t], capture_output=True, text=True)
        if r.returncode != 0:
            good = False
            name = os.path.basename(t)
            for line in (r.stdout + r.stderr).splitlines():
                if "FAIL" in line or "Error" in line:
                    fail("%s: %s" % (name, line.strip()))
            if "FAIL" not in r.stdout:
                fail("%s exited %d" % (name, r.returncode))
    return good


def main(argv):
    quiet = "--quiet" in argv
    failures = []

    def fail(msg):
        failures.append(msg)

    if not os.path.isdir(WEB):
        print("no web/ directory")
        return 1

    n_slugs, n_thumbs, n_missing = check_thumbnails(fail)
    n_surf, n_mesh = check_surfaces(fail)
    n_links = check_html_links(fail)
    n_map = check_import_map(fail)
    n_imports = check_js_relative_imports(fail)
    check_no_remote_fetch(fail)
    n_js = check_js_syntax(fail, quiet)
    n_pages, n_urls, n_math = check_seo(fail)
    cache_ok = check_cluster_cache(fail, quiet)
    n_lfs = check_lfs_deploy(fail)
    n_tracks = check_layout_tracks(fail)

    if not quiet:
        print("thumbnails : %d of %d solids (%d missing)"
              % (n_thumbs - max(0, n_thumbs - n_slugs), n_slugs, n_missing))
        print("surfaces   : %d meshes for %d implemented surfaces"
              % (n_mesh, n_surf))
        print("html links : %d local references resolved" % n_links)
        print("import map : %d specifier(s)" % n_map)
        print("js imports : %d relative import(s) resolved" % n_imports)
        print("js syntax  : %d module(s) checked" % n_js)
        print("seo pages  : %d object pages, %d sitemap URLs"
              % (n_pages, n_urls))
        print("formulae   : %d MathML blocks, all well-formed" % n_math)
        print("lfs deploy : %d tracked path(s), all pulled by the workflow"
              % n_lfs)
        print("layout     : %d viewer track(s) bounded" % n_tracks)
        if cache_ok:
            print("browser js : cluster cache, tile gate and STL export OK")

    if failures:
        print("\n%d FAILURE(S):" % len(failures))
        for f in failures:
            print("   ", f)
        print("RESULT: FAIL")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
