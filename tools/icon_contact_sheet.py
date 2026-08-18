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
body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 78rem;
       padding: 0 1rem; color: #1a1a1a; background: #fff; }
h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .6rem; padding-bottom: .25rem;
     border-bottom: 2px solid #1a1a1a; }
p  { color: #555; margin: .2rem 0 1rem; }
code { font-family: ui-monospace, Consolas, monospace; font-size: .85em; }

/* One card per entry.  The render is the subject of review, so it gets
   the space; the identifiers under it are what a comment cites. */
.grid { display: grid; gap: .75rem;
        grid-template-columns: repeat(auto-fill, minmax(158px, 1fr)); }
.card { border: 1px solid #ddd; border-radius: 3px; overflow: hidden;
        display: flex; flex-direction: column; }
.shot { display: flex; align-items: stretch; }
.shot .big   { background: #1d1d1f; padding: 10px; flex: 1;
               display: grid; place-items: center; }
.shot .side  { display: flex; flex-direction: column; }
.shot .side div { display: grid; place-items: center; width: 34px; flex: 1; }
.shot .on-dark  { background: #1d1d1f; }
.shot .on-light { background: #e8e8ea; }
img.l { width: 128px; height: 128px; display: block;
        image-rendering: pixelated; }
img.s { width: 20px; height: 20px; display: block; }
.id { padding: .4rem .5rem; border-top: 1px solid #eee; }
.id b { display: block; font-size: .8rem; font-weight: 600; }
.id code { display: block; font-size: .68rem; color: #666;
           overflow-wrap: anywhere; }
.id .mod { display: block; font-size: .64rem; color: #999;
           overflow-wrap: anywhere; }
.miss { background: #fff8e5; color: #8a6d1f; font-size: .72rem;
        padding: 1.2rem .5rem; text-align: center; }
.pin  { background: #f2f2f4; color: #666; }
.warn { background: #fff8e5; border: 1px solid #e8d9a8; padding: .6rem .8rem;
        border-radius: 3px; margin: 1rem 0; font-size: .85rem; color: #6b5514; }
</style>
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
                shot = (f'<div class="shot">'
                        f'<div class="big">'
                        f'<img class="l" src="{url}" alt="{alt}"></div>'
                        f'<div class="side">'
                        f'<div class="on-dark">'
                        f'<img class="s" src="{url}" alt=""></div>'
                        f'<div class="on-light">'
                        f'<img class="s" src="{url}" alt=""></div>'
                        f'</div></div>')
            else:
                shot = (f'<div class="miss{" pin" if e.builtin else ""}">'
                        f'{"pinned to" if e.builtin else "no render &mdash;"}'
                        f' <code>{html.escape(e.icon)}</code></div>')
            rows.append(
                f'<div class="card">{shot}'
                f'<div class="id"><b>{html.escape(label)}</b>'
                f'<code>{html.escape(e.op)}</code>'
                f'<span class="mod">{html.escape(src)}</span>'
                f'</div></div>')
        body.append(
            f"<h2>{html.escape(menu.label)} "
            f"<code>({count})</code></h2>\n"
            f'<div class="grid">{"".join(rows)}</div>')

    total = sum(n.values())
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Math Art menu icons</title>
<style>
{CSS}
<h1>Add &rsaquo; Math Art menu icons</h1>
<p>{total} entries in {len(menu_defs.ALL_MENUS)} submenus &mdash;
{n['baked']} rendered, {n['fallback']} with no render,
{n['pinned']} pinned to a built-in glyph on purpose.
Each card shows the render enlarged on a dark menu, with the two small
squares beside it giving actual menu scale on a dark and a light menu.
Cite the operator id under a card when you want a render changed.</p>
<p class="warn"><b>Renders are 64&times;64.</b> The large image is that
file scaled up with no smoothing, so blockiness here is the icon's real
resolution and not a fault in the render. Judge framing, pose, contrast
and whether the form is recognisable &mdash; not sharpness.</p>
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
