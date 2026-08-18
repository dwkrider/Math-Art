"""Build a self-contained HTML contact sheet of the Add-menu icons.

Reads the menu tree from `math_art/menu_defs.py`, the baked PNGs from
`math_art/icons/`, and the operator labels straight out of the
generator sources, then writes one HTML file with every icon embedded
as a data URI -- no Blender, no network, no image folder to ship
alongside it.

    python tools/icon_contact_sheet.py [-o out.html]

The point of the sheet is to answer the two questions a bake leaves
open: does each render still read at the ~20 px a menu row actually
gives it, and does it hold up against both a light and a dark menu?
So the page carries a ground switch and a size switch rather than
showing every icon once at a flattering size.
"""
import argparse
import base64
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
    """Map operator id -> (module basename, bl_label).

    The label is the `bl_label` that follows the `bl_idname` inside the
    same class body, which in this codebase is always within a few
    lines of it.
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
                tail = src[m.end():m.end() + 400]
                lab = _LABEL.search(tail)
                found.setdefault(m.group(1),
                                 (rel, lab.group(1) if lab else ""))
    return found


def _data_uri(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


# ------------------------------------------------------------------ page
CSS = """
*{box-sizing:border-box}
:root{
  --bg:#e9ebee; --surface:#fff; --edge:#d3d7dd;
  --ink:#191c21; --muted:#5d6572; --faint:#8a919d;
  --accent:#2f6f6a; --accent-soft:#dceceb;
  --mark:#a97b12; --mark-soft:#f6ecd4;
  --film:#16181b; --sheet:#f3f4f6;
  --shadow:0 1px 2px rgba(20,24,30,.08),0 8px 24px rgba(20,24,30,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#131519; --surface:#1c1f24; --edge:#2c313a;
    --ink:#e7eaee; --muted:#99a1ae; --faint:#6d7583;
    --accent:#6fb8b0; --accent-soft:#1d3330;
    --mark:#e0b03a; --mark-soft:#33290f;
    --film:#0d0e10; --sheet:#22262c;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.28);
  }
}
:root[data-theme="dark"]{
  --bg:#131519; --surface:#1c1f24; --edge:#2c313a;
  --ink:#e7eaee; --muted:#99a1ae; --faint:#6d7583;
  --accent:#6fb8b0; --accent-soft:#1d3330;
  --mark:#e0b03a; --mark-soft:#33290f;
  --film:#0d0e10; --sheet:#22262c;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.28);
}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px; line-height:1.55;
}
.mono{font-family:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 80px}

header{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-end;
  justify-content:space-between;margin-bottom:8px}
h1{font-size:29px;line-height:1.15;margin:0;letter-spacing:-.02em;
  text-wrap:balance;font-weight:640}
.sub{margin:6px 0 0;color:var(--muted);max-width:62ch}

.tally{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0 8px;padding:0;list-style:none}
.tally li{display:flex;align-items:baseline;gap:8px;background:var(--surface);
  border:1px solid var(--edge);border-radius:3px;padding:9px 13px;box-shadow:var(--shadow)}
.tally b{font-size:19px;font-variant-numeric:tabular-nums;font-weight:640}
.tally span{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}

.controls{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:20px;
  align-items:center;padding:13px 0;margin:20px 0 28px;
  background:var(--bg);border-bottom:1px solid var(--edge)}
.ctl{display:flex;align-items:center;gap:9px}
.ctl>span{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.seg{display:flex;border:1px solid var(--edge);border-radius:3px;overflow:hidden;background:var(--surface)}
.seg button{appearance:none;border:0;background:transparent;color:var(--muted);
  font:inherit;font-size:13px;padding:6px 13px;cursor:pointer}
.seg button+button{border-left:1px solid var(--edge)}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--surface)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}

section{margin:0 0 34px}
.mhead{display:flex;align-items:baseline;gap:12px;margin:0 0 12px;
  padding-bottom:7px;border-bottom:1px solid var(--edge)}
.mhead h2{margin:0;font-size:16px;font-weight:640;letter-spacing:-.01em}
.mhead .n{font-size:12px;color:var(--faint);font-variant-numeric:tabular-nums}

.grid{display:grid;gap:10px;
  grid-template-columns:repeat(auto-fill,minmax(196px,1fr))}
.cell{background:var(--surface);border:1px solid var(--edge);border-radius:3px;
  padding:11px;display:flex;gap:11px;align-items:flex-start;box-shadow:var(--shadow)}
.cell.sep{grid-column:1/-1;background:transparent;border:0;box-shadow:none;
  padding:0;height:1px;background:var(--edge);margin:5px 0}

.frame{flex:0 0 auto;display:grid;place-items:center;border-radius:2px;
  background:var(--film);width:76px;height:76px;overflow:hidden}
body[data-ground="light"] .frame{background:#e4e5e7}
body[data-ground="check"] .frame{
  background-color:#9aa0a8;
  background-image:linear-gradient(45deg,#7f858d 25%,transparent 25%,transparent 75%,#7f858d 75%),
                   linear-gradient(45deg,#7f858d 25%,transparent 25%,transparent 75%,#7f858d 75%);
  background-size:12px 12px;background-position:0 0,6px 6px}
/* The page is wrapped in a <body> it does not own, so every default
   has to hold with no data-* attribute set; the switches only override. */
.frame img{display:block;width:20px;height:20px}
body[data-size="menu"] .frame img{width:20px;height:20px}
body[data-size="real"] .frame img{width:64px;height:64px}
.frame.none{border:1px dashed var(--edge);background:transparent}
.frame.none em{font-style:normal;font-size:9.5px;line-height:1.3;text-align:center;
  color:var(--faint);padding:4px;word-break:break-word}

.meta{min-width:0;flex:1}
.meta .lab{font-weight:600;font-size:13.5px;line-height:1.3;
  overflow-wrap:anywhere}
.meta .op{font-size:11.5px;color:var(--muted);margin-top:3px;overflow-wrap:anywhere}
.meta .src{font-size:11px;color:var(--faint);margin-top:2px;overflow-wrap:anywhere}
.chip{display:inline-block;margin-top:7px;font-size:10px;letter-spacing:.08em;
  text-transform:uppercase;padding:2px 7px;border-radius:2px;font-weight:600}
.chip.baked{background:var(--accent-soft);color:var(--accent)}
.chip.fallback{background:var(--mark-soft);color:var(--mark)}
.chip.pinned{background:var(--edge);color:var(--muted)}

footer{margin-top:46px;padding-top:18px;border-top:1px solid var(--edge);
  color:var(--faint);font-size:12.5px}
@media (max-width:560px){.grid{grid-template-columns:1fr}.wrap{padding:28px 16px 60px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = """
for (const seg of document.querySelectorAll('.seg')) {
  seg.addEventListener('click', (ev) => {
    const b = ev.target.closest('button'); if (!b) return;
    const key = seg.dataset.key;
    document.body.dataset[key] = b.dataset.val;
    for (const o of seg.querySelectorAll('button'))
      o.setAttribute('aria-pressed', String(o === b));
  });
}
"""


def build(out_path):
    icons_dir = os.path.join(PKG, "icons")
    srcs = operator_sources()
    bakeable = set(menu_defs.bakeable_ops())

    n_baked = n_fallback = n_pinned = 0
    sections = []
    for menu in menu_defs.ALL_MENUS:
        cells = []
        shown = 0
        for entry in menu.entries:
            if entry.op is None:
                cells.append('<div class="cell sep"></div>')
                continue
            shown += 1
            src, blabel = srcs.get(entry.op, ("", ""))
            label = entry.text or blabel or entry.op
            png = os.path.join(icons_dir, entry.op.replace(".", "_") + ".png")
            if entry.builtin:
                state, chip = "pinned", "pinned to built-in"
                n_pinned += 1
            elif os.path.isfile(png):
                state, chip = "baked", "baked"
                n_baked += 1
            else:
                state, chip = "fallback", "built-in fallback"
                n_fallback += 1

            if state == "baked":
                frame = (f'<div class="frame">'
                         f'<img src="{_data_uri(png)}" alt="{html.escape(label)} icon"></div>')
            else:
                frame = (f'<div class="frame none"><em class="mono">'
                         f'{html.escape(entry.icon)}</em></div>')
            cells.append(
                f'<div class="cell">{frame}<div class="meta">'
                f'<div class="lab">{html.escape(label)}</div>'
                f'<div class="op mono">{html.escape(entry.op)}</div>'
                f'<div class="src mono">{html.escape(src)}</div>'
                f'<span class="chip {state}">{chip}</span>'
                f'</div></div>')
        sections.append(
            f'<section><div class="mhead"><h2>{html.escape(menu.label)}</h2>'
            f'<span class="n mono">{shown} entries &middot; '
            f'{html.escape(menu.idname)}</span></div>'
            f'<div class="grid">{"".join(cells)}</div></section>')

    total = n_baked + n_fallback + n_pinned
    page = f"""<title>Math Art Icon Sheet</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <div>
    <h1>Add &rsaquo; Math Art icon sheet</h1>
    <p class="sub">Every entry in the Add menu, with the baked render it
    draws today. Frames marked <em>built-in fallback</em> have no PNG yet and
    still draw the Blender glyph named in the frame.</p>
  </div>
</header>

<ul class="tally">
  <li><b>{total}</b><span>entries</span></li>
  <li><b>{n_baked}</b><span>baked</span></li>
  <li><b>{n_fallback}</b><span>built-in fallback</span></li>
  <li><b>{n_pinned}</b><span>pinned</span></li>
  <li><b>{len(menu_defs.ALL_MENUS)}</b><span>menus</span></li>
</ul>

<div class="controls">
  <div class="ctl"><span>Icon size</span>
    <div class="seg" data-key="size">
      <button data-val="menu" aria-pressed="true">Menu (20px)</button>
      <button data-val="real" aria-pressed="false">Full (64px)</button>
    </div>
  </div>
  <div class="ctl"><span>Menu ground</span>
    <div class="seg" data-key="ground">
      <button data-val="dark" aria-pressed="true">Dark</button>
      <button data-val="light" aria-pressed="false">Light</button>
      <button data-val="check" aria-pressed="false">Checker</button>
    </div>
  </div>
</div>

{"".join(sections)}

<footer>Generated by <span class="mono">tools/icon_contact_sheet.py</span>
from <span class="mono">math_art/menu_defs.py</span> and
<span class="mono">math_art/icons/</span>. Re-run it after a bake to refresh.</footer>
</div>
<script>{JS}</script>
"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"{out_path}: {total} entries "
          f"({n_baked} baked, {n_fallback} fallback, {n_pinned} pinned), "
          f"{os.path.getsize(out_path)/1024:.0f} KB")


def main():
    ap = argparse.ArgumentParser(prog="icon_contact_sheet")
    ap.add_argument("-o", "--out",
                    default=os.path.join(PROJ, "icon_contact_sheet.html"))
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
