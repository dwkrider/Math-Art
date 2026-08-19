# Documentation coverage gate -- pure Python, no Blender needed.
#
# The generator docs are owned by the integrator and written per merge,
# not by feature branches (see CLAUDE.md).  That only works if the gap
# is visible, so this is the ledger: it fails when a generator has no
# page, when the gallery is stale, or when a page points at a picture
# that is not there -- all three of which had happened by the time this
# was written (34 generators unreachable from the gallery, 50 broken
# image references).
#
#     python tests/test_docs.py
#     python tests/test_docs.py --quiet     only the failures
#
# Option drift -- a page's Options table disagreeing with the operator
# -- is reported as a WARNING rather than a failure.  It is advisory by
# design: an operator gaining a property does not make the prose wrong,
# and a hard failure here would block merges on prose edits.  The count
# is printed every run so the debt stays in view.
import json
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(PROJ, "docs")
GEN = os.path.join(DOCS, "generators")
sys.path.insert(0, os.path.join(PROJ, "tools"))

import subjects                                            # noqa: E402

menu_defs = subjects.load_menu_defs()

# Operators that deliberately have no page, with the reason.  Keep this
# honest: it is the list of things a reader cannot look up.
UNDOCUMENTED = {
    # Nothing to photograph and little to say: it emits the phyllotaxis
    # seed points as a points-only mesh, consumed by other generators.
    # It is covered inside the phyllotaxis page instead.
    "mesh.receptacle_add": "points-only helper, covered by phyllotaxis",
}


def _pages():
    return sorted(f[:-3] for f in os.listdir(GEN) if f.endswith(".md"))


def check_every_operator_has_a_page(fail):
    pages = set(_pages())
    for op in menu_defs.unique_ops():
        if op in UNDOCUMENTED:
            continue
        slug = subjects.slug_for(op)
        if slug not in pages:
            fail(f"{op}: no page at docs/generators/{slug}.md "
                 f"(scaffold one, or add it to UNDOCUMENTED with a reason)")
    for op, reason in UNDOCUMENTED.items():
        if op not in menu_defs.unique_ops():
            fail(f"UNDOCUMENTED lists {op}, which is in no menu")
        elif not reason.strip():
            fail(f"UNDOCUMENTED[{op}] has no reason")


def check_no_orphan_pages(fail):
    slugs = {subjects.slug_for(op) for op in menu_defs.unique_ops()}
    for slug in _pages():
        if slug not in slugs:
            fail(f"docs/generators/{slug}.md documents no menu operator "
                 f"(operator renamed? add a SLUG_OVERRIDE in subjects.py)")


def check_gallery_is_current(fail):
    sys.path.insert(0, DOCS)
    try:
        import build_index
    except Exception as e:                                  # pragma: no cover
        fail(f"cannot import docs/build_index.py: {e!r}")
        return
    text, _ = build_index.build()
    out = os.path.join(DOCS, "README.md")
    current = (open(out, encoding="utf-8").read()
               if os.path.exists(out) else None)
    if current != text:
        fail("docs/README.md is stale -- run `python docs/build_index.py`")


def check_images_exist(fail):
    md = [os.path.join(DOCS, "README.md")]
    md += [os.path.join(GEN, s + ".md") for s in _pages()]
    for path in md:
        base = os.path.dirname(path)
        text = open(path, encoding="utf-8").read()
        refs = re.findall(r'(?:src="|\]\()([^")\s]+\.png)', text)
        for ref in refs:
            if not os.path.exists(os.path.normpath(
                    os.path.join(base, ref))):
                fail(f"{os.path.relpath(path, PROJ)}: missing image {ref}")


def check_page_shape(fail):
    """Every page needs the sections a reader expects to find."""
    for slug in _pages():
        text = open(os.path.join(GEN, slug + ".md"),
                    encoding="utf-8").read()
        if not text.startswith("# "):
            fail(f"{slug}.md: no H1 title (the gallery uses it as the label)")
        for section in ("## Overview", "## Options", "## References"):
            if section not in text:
                fail(f"{slug}.md: no '{section}' section")


def report_option_drift():
    """Advisory: pages whose Options table has fallen behind the operator.

    Compares each page's table against docs/options_snapshot.json, which
    the scaffolder writes from the live RNA.  Returns a warning count.
    """
    snap_path = os.path.join(DOCS, "options_snapshot.json")
    if not os.path.exists(snap_path):
        print("  (no options_snapshot.json -- run docs/scaffold_pages.py "
              "under Blender to create it)")
        return 0
    snap = json.load(open(snap_path, encoding="utf-8"))
    warned = 0
    for op in sorted(menu_defs.unique_ops()):
        if op in UNDOCUMENTED or op not in snap:
            continue
        path = os.path.join(GEN, subjects.slug_for(op) + ".md")
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        m = re.search(r"## Options\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
        if not m:
            continue
        listed = {row.split("|")[1].strip()
                  for row in m.group(1).splitlines()
                  if row.startswith("|") and row.count("|") >= 3}
        listed -= {"Option", "---"}
        if not listed:
            continue
        expected = set(snap[op])
        missing = expected - listed
        if missing:
            warned += 1
            print(f"  WARN {subjects.slug_for(op)}.md: "
                  f"{len(missing)} option(s) not documented: "
                  f"{', '.join(sorted(missing)[:6])}"
                  + (" ..." if len(missing) > 6 else ""))
    return warned


def _git_time(path):
    """Commit time of `path`'s last change, or None if untracked."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", path],
            cwd=PROJ, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    out = out.stdout.strip()
    return int(out) if out.isdigit() else None


def _dirty_paths():
    """Repo-relative paths with uncommitted changes, forward-slashed.

    A figure re-rendered but not yet committed still carries its *old*
    commit time, so comparing timestamps alone would flag the very
    images a docs pass just refreshed.  Anything dirty is by definition
    fresher than the last commit of the module.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             cwd=PROJ, capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return set()
    paths = set()
    for line in out.splitlines():
        p = line[3:].strip().strip('"')
        if " -> " in p:                       # a rename
            p = p.split(" -> ", 1)[1]
        paths.add(p.replace("\\", "/"))
    return paths


def report_stale_figures():
    """Advisory: figures committed before the generator last changed.

    This is the "which generators did the merge affect" detector.  A new
    generator is easy to spot -- it has no page.  The case that rots
    silently is an *existing* generator whose defaults, enum options or
    geometry changed, leaving a figure that no longer shows what the
    operator produces.  Comparing commit times catches exactly that,
    and needs nothing but git.
    """
    dirty = _dirty_paths()
    stale = []
    for op in sorted(menu_defs.unique_ops()):
        if op in UNDOCUMENTED:
            continue
        img = "docs/images/%s.png" % subjects.slug_for(op)
        if not os.path.exists(os.path.join(PROJ, img)):
            continue
        # Just re-rendered (uncommitted) -- that is a docs pass in
        # progress, not drift.
        if img in dirty:
            continue
        mod = subjects.module_for(op)
        if mod is None:
            continue
        t_img = _git_time(img)
        t_mod = _git_time(os.path.relpath(mod, PROJ).replace("\\", "/"))
        # An untracked figure has never been committed, so there is no
        # "before" to compare against.
        if t_img is None or t_mod is None:
            continue
        if t_mod > t_img:
            stale.append((subjects.slug_for(op), os.path.basename(mod)))
    for slug, mod in stale:
        print(f"  WARN {slug}.png predates its generator ({mod}) -- "
              f"re-render it")
    return len(stale)


def main(argv):
    quiet = "--quiet" in argv
    failures = []

    def fail(msg):
        failures.append(msg)

    checks = (check_every_operator_has_a_page,
              check_no_orphan_pages,
              check_gallery_is_current,
              check_images_exist,
              check_page_shape)
    for check in checks:
        before = len(failures)
        check(fail)
        if not quiet:
            n = len(failures) - before
            print(f"  {'FAIL' if n else 'ok  '} "
                  f"{check.__name__} ({n} problem(s))")

    print("\noption drift (advisory):")
    warned = report_option_drift()
    print(f"  {warned} page(s) with options the operator has and the "
          f"page does not")

    print("\nstale figures (advisory):")
    stale = report_stale_figures()
    print(f"  {stale} figure(s) older than the generator they show")

    n_ops = len(menu_defs.unique_ops())
    print(f"\n{len(_pages())} pages for {n_ops} operators "
          f"({len(UNDOCUMENTED)} deliberately undocumented)")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("   ", f)
        print("RESULT: FAIL")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
