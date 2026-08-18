"""Build a local HTML contact sheet of the Add-menu icons.

Reads the menu tree from `math_art/menu_defs.py`, the baked PNGs from
`math_art/icons/`, and each operator's label out of the generator
sources, then writes one plain HTML file:

    python tools/icon_contact_sheet.py            # -> icon_contact_sheet.html
    python tools/icon_contact_sheet.py -o /tmp/sheet.html

Images are referenced relatively, not embedded, so the page stays a few
kilobytes -- which means it has to sit at the repository root for
`math_art/icons/...` to resolve.  Pass -o elsewhere only if you do not
mind the images going missing.

Every row shows the icon three ways at once -- menu scale on a dark
menu, menu scale on a light one, and full size -- because those are the
questions a bake leaves open: does the render survive the ~20 px a menu
row gives it, and does it hold against both menu themes.  Showing all
three side by side keeps the page free of scripting.
"""
import argparse
import html
import importlib.util
import os
import re

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(PROJ, "math_art")


def _load(name):
    """Import a bpy-free module from math_art/ without the package."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PKG, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


menu_defs = _load("menu_defs")

_IDNAME = re.compile(r'bl_idname\s*=\s*"((?:mesh|curve|object)\.[a-z0-9_]+)"')
_LABEL = re.compile(r'bl_label\s*=\s*"([^"]+)"')


def operator_sources():
    """Map operator id -> (module path, bl_label).

    The label is the `bl_label` following the `bl_idname` in the same
    class body, which in this codebase is always a few lines away.
    """
    found = {}
    for root, _dirs, files in os.walk(PKG):
        if "__pycache__" in root:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            rel = os.path.relpath(path, PKG).replace("\\", "/")
            for m in _IDNAME.finditer(src):
                lab = _LABEL.search(src[m.end():m.end() + 400])
                found.setdefault(m.group(1),
                                 (rel, lab.group(1) if lab else ""))
    return found


CSS = """\
body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
       padding: 0 1rem; color: #1a1a1a; background: #fff; }
h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
h2 { font-size: 1rem; margin: 2rem 0 .4rem; }
p  { color: #555; margin: .2rem 0 1rem; }
code { font-family: ui-monospace, Consolas, monospace; font-size: .85em; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #e5e5e5;
         vertical-align: middle; }
th { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
     color: #666; font-weight: 600; }
td.i { width: 1%; white-space: nowrap; }
td.dark  { background: #1d1d1f; }
td.light { background: #e8e8ea; }
img.s { width: 20px; height: 20px; display: block; }
img.l { width: 64px; height: 64px; display: block; }
.none { color: #999; font-size: .75rem; }
.st { font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; }
.baked { color: #1a6b52; } .fallback { color: #8a6d1f; } .pinned { color: #777; }
"""


def build(out_path):
    icons = os.path.join(PKG, "icons")
    srcs = operator_sources()
    rel_root = os.path.dirname(os.path.abspath(out_path))

    n = {"baked": 0, "fallback": 0, "pinned": 0}
    body = []
    for menu in menu_defs.ALL_MENUS:
        rows = []
        count = 0
        for e in menu.entries:
            if e.op is None:
                continue
            count += 1
            src, blabel = srcs.get(e.op, ("", ""))
            label = e.text or blabel or e.op
            png = os.path.join(icons, e.op.replace(".", "_") + ".png")
            if e.builtin:
                state, word = "pinned", "pinned"
            elif os.path.isfile(png):
                state, word = "baked", "baked"
            else:
                state, word = "fallback", "fallback"
            n[state] += 1

            if state == "baked":
                url = html.escape(
                    os.path.relpath(png, rel_root).replace("\\", "/"))
                alt = html.escape(label)
                cells = (f'<td class="i dark"><img class="s" src="{url}" alt="{alt}"></td>'
                         f'<td class="i light"><img class="s" src="{url}" alt=""></td>'
                         f'<td class="i dark"><img class="l" src="{url}" alt=""></td>')
            else:
                glyph = f'<span class="none">{html.escape(e.icon)}</span>'
                cells = (f'<td class="i">{glyph}</td>'
                         f'<td class="i"></td><td class="i"></td>')
            rows.append(
                f"<tr>{cells}"
                f"<td>{html.escape(label)}</td>"
                f"<td><code>{html.escape(e.op)}</code></td>"
                f"<td><code>{html.escape(src)}</code></td>"
                f'<td class="st {state}">{word}</td></tr>')
        body.append(
            f"<h2>{html.escape(menu.label)} "
            f"<code>({count})</code></h2>\n"
            "<table><tr><th>20px</th><th>20px</th><th>64px</th><th>Entry</th>"
            "<th>Operator</th><th>Module</th><th></th></tr>\n"
            + "\n".join(rows) + "</table>")

    total = sum(n.values())
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Math Art menu icons</title>
<style>
{CSS}</style>
<h1>Add &rsaquo; Math Art menu icons</h1>
<p>{total} entries in {len(menu_defs.ALL_MENUS)} submenus &mdash;
{n['baked']} baked, {n['fallback']} still drawing a built-in glyph,
{n['pinned']} pinned to one deliberately.
The first three columns are the same icon at menu scale on a dark menu,
at menu scale on a light one, and at full size.
Rows with no render name the Blender glyph they fall back to.</p>
{chr(10).join(body)}
<p><small>Generated by <code>tools/icon_contact_sheet.py</code>.
Images are relative, so keep this file at the repository root.</small></p>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"{out_path}: {total} entries "
          f"({n['baked']} baked, {n['fallback']} fallback, "
          f"{n['pinned']} pinned), {os.path.getsize(out_path)/1024:.0f} KB")


def main():
    ap = argparse.ArgumentParser(prog="icon_contact_sheet")
    ap.add_argument("-o", "--out",
                    default=os.path.join(PROJ, "icon_contact_sheet.html"))
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
