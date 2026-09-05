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
#   coverage    every solid in the database has a thumbnail, and every
#               thumbnail belongs to a solid (an orphan means a slug was
#               renamed and the tile was left behind)
#   links       every local href/src in the HTML resolves on disk, and
#               every relative import in the JS resolves too
#   importmap   the bare specifiers the modules import are declared, and
#               point at files that exist
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

# Hosts are never contacted at runtime; the only absolute URLs allowed
# are ones a reader clicks.
ALLOWED_LINK_HOSTS = ("github.com", "en.wikipedia.org", "mathworld.wolfram.com")


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
            target = os.path.normpath(os.path.join(base, url.split("?")[0]))
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


def main(argv):
    quiet = "--quiet" in argv
    failures = []

    def fail(msg):
        failures.append(msg)

    if not os.path.isdir(WEB):
        print("no web/ directory")
        return 1

    n_slugs, n_thumbs, n_missing = check_thumbnails(fail)
    n_links = check_html_links(fail)
    n_map = check_import_map(fail)
    n_imports = check_js_relative_imports(fail)
    check_no_remote_fetch(fail)
    n_js = check_js_syntax(fail, quiet)

    if not quiet:
        print("thumbnails : %d of %d solids (%d missing)"
              % (n_thumbs - max(0, n_thumbs - n_slugs), n_slugs, n_missing))
        print("html links : %d local references resolved" % n_links)
        print("import map : %d specifier(s)" % n_map)
        print("js imports : %d relative import(s) resolved" % n_imports)
        print("js syntax  : %d module(s) checked" % n_js)

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
