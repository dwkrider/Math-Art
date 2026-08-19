"""Generate docs/README.md -- the generator gallery -- from the menus.

The gallery used to be written by hand, and had drifted badly: three of
the ten Add-menu submenus (Plants & Growth, Patterns, Rollers) had no
section at all, so 34 generators were unreachable from the front page,
and four thumbnails pointed at renders that were never committed.

Both failures are structural, so the fix is structural: the section
order and membership come from `math_art/menu_defs.py`, which is the
same table that builds the Add menu, and a row is only emitted for a
page that exists, with a thumbnail only if the render exists.  A
generated index cannot contain a broken link.

Each row's label is the page's own `# H1`, so the gallery cannot
disagree with the page it links to, and no Blender is needed here.

Run:  python docs/build_index.py
      python docs/build_index.py --check      (exit 1 if out of date)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "tools"))

import subjects                                            # noqa: E402
menu_defs = subjects.load_menu_defs()

GEN = os.path.join(HERE, "generators")
IMG = os.path.join(HERE, "images")
OUT = os.path.join(HERE, "README.md")
PER_ROW = 3
THUMB = 240

HEADER = """# Math Art — Generator Gallery

Every documented generator in the Math Art extension, in the order they
appear under **Add ▸ Math Art**. Click any shape for its page — an
overview, the configuration options, the underlying mathematics, and
references.

Each shape is generated centered on the origin fitting a 2 m cube, and
shown with a consistent studio render (regenerate them with
[`render_docs.py`](render_docs.py); regenerate this page with
[`build_index.py`](build_index.py) — do not edit it by hand).
"""

FOOTER = """
---

← Back to the [project README](../README.md)
"""


def page_title(slug):
    """The page's own H1, which is the label the gallery shows."""
    path = os.path.join(GEN, slug + ".md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# "):
                return line[2:].strip()
    return slug.replace("_", " ").title()


def cell(slug, title):
    """One gallery cell: thumbnail over a bold link, or just the link.

    A generator whose figure has not been rendered yet still gets a
    row -- the page is the point, the picture is the decoration -- but
    it gets no <img> tag, because an <img> at a path that does not
    exist is the broken-thumbnail bug this script exists to prevent.
    """
    link = f"generators/{slug}.md"
    if os.path.exists(os.path.join(IMG, slug + ".png")):
        return (f'[<img src="images/{slug}.png" width="{THUMB}">]({link})'
                f'<br>**[{title}]({link})**')
    return f'**[{title}]({link})**<br><sub>(figure pending)</sub>'


def table(cells):
    rows = ["| | | |", "|:--:|:--:|:--:|"]
    for i in range(0, len(cells), PER_ROW):
        chunk = list(cells[i:i + PER_ROW])
        chunk += [""] * (PER_ROW - len(chunk))
        rows.append("| " + " | ".join(chunk) + " |")
    return rows


def sections():
    """(label, [slug...]) per menu, in Add-menu order, deduplicated.

    An operator can legitimately sit in two menus (an L-system is both
    a fractal and a plant).  In the menus that is a convenience; in a
    gallery it would be the same picture twice, so each generator is
    shown under the first menu that claims it.
    """
    seen = set()
    out = []
    for menu in menu_defs.MENU_ORDER + (menu_defs.STYLES,):
        slugs = []
        for entry in menu.entries:
            if entry.op is None or entry.op in seen:
                continue
            seen.add(entry.op)
            slugs.append(subjects.slug_for(entry.op))
        if slugs:
            out.append((menu.label, slugs))
    root = [subjects.slug_for(e.op) for e in menu_defs.ROOT_ENTRIES
            if e.op not in seen]
    if root:
        out.append(("Sculpture", root))
    return out


def build():
    lines = [HEADER]
    undocumented = []
    for label, slugs in sections():
        cells = []
        for slug in slugs:
            title = page_title(slug)
            if title is None:
                undocumented.append(slug)
                continue
            cells.append(cell(slug, title))
        if not cells:
            continue
        lines.append(f"\n## {label}\n")
        lines.extend(table(cells))
    lines.append(FOOTER)
    return "\n".join(lines), undocumented


def main(argv):
    text, undocumented = build()
    check = "--check" in argv
    current = (open(OUT, encoding="utf-8").read()
               if os.path.exists(OUT) else None)
    if check:
        if current != text:
            print("docs/README.md is out of date -- "
                  "run `python docs/build_index.py`")
            return 1
        print("docs/README.md is up to date")
        return 0
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    n = sum(len(s) for _, s in sections())
    print(f"wrote {OUT}: {n - len(undocumented)}/{n} generators listed")
    if undocumented:
        print(f"{len(undocumented)} with no page yet "
              f"(they are simply left out until one exists):")
        for slug in sorted(undocumented):
            print("   ", slug)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
