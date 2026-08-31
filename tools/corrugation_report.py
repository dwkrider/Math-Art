"""Render an HTML validation report for the surface corrugation.

    blender --background --factory-startup --python tools/corrugation_report.py

Writes `corrugation_report.html` at the repo root -- gitignored, the
same convention as `tools/icon_contact_sheet.py`.

WHY A PICTURE REPORT AND NOT JUST NUMBERS.  The corrugation's own
metrics -- fit error, flattening residual, area ratio -- all measure it
against ITSELF, and they were at their best on the catenoid at exactly
the moment its fold was at its worst.  Three columns side by side make
the question unavoidable: here is the shape it aimed at, here is the
pattern it handed you, here is what that pattern actually folds into.
The first and third columns matching is the entire claim of the
operator, and for curved targets they do not match.

The images are embedded as data URIs so the file is one self-contained
thing that can be opened or sent anywhere.
"""

import base64
import io as _io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "docs"), os.path.join(ROOT, "math_art"),
           ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                        # noqa: E402

import bpy                                                # noqa: E402

import bake_menu_icons as bmi                             # noqa: E402

OUT = os.path.join(ROOT, "corrugation_report.html")
TMP = os.path.join(os.environ.get("TEMP", "/tmp"), "corrugation_report")
RES = 460
NU = NV = 12

TARGETS = ("PLANE", "HYPAR", "SCHERK", "SPHERE", "CATENOID")

#: Three-quarter for the 3-D forms; straight down for the flat pattern,
#: where the crease layout IS the information and a tilt only hides it.
ORIENT_3D = (np.deg2rad(62.0), 0.0, np.deg2rad(28.0))
ORIENT_FLAT = (0.0, 0.0, 0.0)


def _render(obj, path, orient):
    """Render one object alone, without disturbing the scene.

    ISOLATE BY VISIBILITY, NOT BY DELETION.  The first version of this
    deleted every object but the subject -- including the camera and
    lights the studio rig had just built, so the very first render died
    with "Cannot render, no camera".  `render_docs.subjects()` already
    excludes anything hidden from rendering when it frames the shot, so
    hiding is both correct and reversible, which matters here because
    each object has to be rendered again after it is folded.
    """
    was = [(o, o.hide_render) for o in bpy.data.objects if o.type == 'MESH']
    keep_rot = tuple(obj.rotation_euler)
    keep_loc = tuple(obj.location)
    try:
        for o, _h in was:
            o.hide_render = (o is not obj)
        obj.rotation_euler = orient
        obj.location = (0.0, 0.0, 0.0)
        bmi._render_to(path)
    finally:
        for o, h in was:
            o.hide_render = h
        obj.rotation_euler = keep_rot
        obj.location = keep_loc
    return path


#: Crease colours, matching the viewport overlay so the report and the
#: screen agree.
_SVG_COLOUR = {"M": "#d92b2b", "V": "#2b5bd9", "B": "#222222",
               "U": "#c08a26", "F": "#8a8a8a"}


def crease_svg(frame, size=460):
    """Draw a crease pattern as inline SVG.

    NOT a render.  A crease pattern's content is which line is a
    mountain and which a valley, and that lives in an edge ATTRIBUTE
    drawn by a viewport overlay -- so rendering the mesh gives a
    featureless white polygon, which is exactly what the first version
    of this report showed.  Drawing the edges directly is both truthful
    and sharper than any raster of them.
    """
    V = np.asarray(frame.verts, dtype=float)[:, :2]
    E = np.asarray(frame.edges, dtype=np.int64).reshape(-1, 2)
    A = frame.assignment
    lo, hi = V.min(0), V.max(0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) or 1.0
    pad = 0.04 * span
    sc = (size - 2 * pad) / span

    def px(p):
        # SVG y runs down; the pattern is stored y-up
        return ((p[0] - lo[0]) * sc + pad,
                (hi[1] - p[1]) * sc + pad)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" '
             f'viewBox="0 0 {size} {size}">']
    # boundary last so it reads on top of the creases
    order = sorted(range(len(E)),
                   key=lambda k: str(A[k]) == "B")
    for k in order:
        a, b = E[k]
        x1, y1 = px(V[a])
        x2, y2 = px(V[b])
        col = _SVG_COLOUR.get(str(A[k]), "#999")
        w = 1.9 if str(A[k]) == "B" else 1.1
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                     f'y2="{y2:.1f}" stroke="{col}" stroke-width="{w}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _kabsch_to(obj, want):
    """Rotate a folded object onto the target, in place.

    The compliant solver fixes no orientation -- the sheet settles
    wherever it likes -- so photographing it from the same camera angle
    as the target compares two different viewpoints and makes a match
    look like a mismatch.  The PLANE row proved it: 0.003 rms error and
    two pictures that shared no resemblance.
    """
    got = _co(obj)
    A = got - got.mean(0)
    B = want - want.mean(0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    R = Vt.T @ np.diag(
        [1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))]) @ U.T
    new = (R @ A.T).T + want.mean(0)
    for v, p in zip(obj.data.vertices, new):
        v.co = p
    obj.data.update()


def _b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _align(got, want):
    A = got - got.mean(0)
    B = want - want.mean(0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    det = float(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(det)]) @ U.T
    dev = np.linalg.norm((R @ A.T).T - B, axis=1)
    scale = float(np.ptp(want, axis=0).max()) or 1.0
    return float(dev.mean()) / scale, float(dev.max()) / scale


def _co(obj):
    return np.array([tuple(v.co) for v in obj.data.vertices], dtype=float)


def main():
    import math_art
    try:
        math_art.register()
    except Exception as exc:
        print("[note] register:", exc)
    from math_art import crease

    os.makedirs(TMP, exist_ok=True)
    bmi.RES = RES
    bmi._setup()

    import render_docs as rd

    rows = []
    for kind in TARGETS:
        t0 = time.time()
        rd.clear_sculpts()
        bpy.ops.mesh.corrugation_add(target=kind, nu=NU, nv=NV, relax=1200)
        corr = bpy.data.objects[f"{kind.title()} Corrugation"]
        pat = bpy.data.objects[f"{kind.title()} Crease Pattern"]
        want = _co(corr)
        rep = dict(crease.corrugate.fit(kind, nu=NU, nv=NV, amplitude=0.12,
                                        iters=1200, axis=None)[2])

        imgs = {}
        imgs["target"] = _render(
            corr, os.path.join(TMP, f"{kind}_target.png"), ORIENT_3D)
        svg = crease_svg(crease.corrugate.fit(
            kind, nu=NU, nv=NV, amplitude=0.12, iters=1200, axis=None)[0])

        # fold the pattern the operator handed out, then render THAT
        bpy.context.view_layer.objects.active = pat
        bpy.ops.object.fold_solve(solver='COMPLIANT', drive=1.0, steps=12,
                                  animate=False, colour_strain=False)
        got = _co(pat)
        rms, mx = _align(got, want)
        depth = float(np.ptp(got[:, 2])) / (float(np.ptp(want[:, 2])) or 1.0)
        _kabsch_to(pat, want)          # same viewpoint as the target
        imgs["folded_result"] = _render(
            pat, os.path.join(TMP, f"{kind}_result.png"), ORIENT_3D)

        rows.append(dict(kind=kind, imgs=imgs, svg=svg, depth=depth,
                         rms=rms, mx=mx, rep=rep, secs=time.time() - t0))
        print(f"[{kind}] depth={depth:.2f} rms={rms:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    _write(rows)
    print("wrote", OUT)
    return 0


def _write(rows):
    def img(p):
        return (f'<img src="data:image/png;base64,{_b64(p)}" '
                f'alt="" loading="lazy">')

    body = []
    for r in rows:
        k = r["kind"]
        rep = r["rep"]
        control = k == "PLANE"
        verdict = ("exact — this one is developable, so it is the control"
                   if control else
                   f"reaches {r['depth'] * 100:.0f}% of the intended depth")
        cls = "ok" if control else ("warn" if r["depth"] > 0.75 else "bad")
        body.append(f"""
<section>
  <h2>{k.title()} <span class="badge {cls}">{verdict}</span></h2>
  <div class="grid">
    <figure>{img(r['imgs']['target'])}
      <figcaption><b>1. Target</b><br>The shape the operator displays.
      Built directly from the surface by offsetting along its normals —
      <em>not</em> folded from anything.</figcaption></figure>
    <figure>{r['svg']}
      <figcaption><b>2. Crease pattern</b><br>Flattened from (1).
      Mountain red, valley blue. Unfolding residual
      {rep['max_edge_error']:.2f}.</figcaption></figure>
    <figure>{img(r['imgs']['folded_result'])}
      <figcaption><b>3. What (2) actually folds to</b><br>Bending Paper
      solver, full drive. This is the column that must match (1), and
      for curved targets it does not.</figcaption></figure>
  </div>
  <table>
    <tr><td>depth reached</td><td>{r['depth'] * 100:.0f}%</td>
        <td>folded z-extent over the target's</td></tr>
    <tr><td>shape error</td><td>{r['rms']:.3f} rms, {r['mx']:.3f} max</td>
        <td>Kabsch-aligned, as a fraction of model size</td></tr>
    <tr><td>pattern drift</td><td>{rep.get('drift', float('nan')):.3f}</td>
        <td>seeded at the target, how far it walks away</td></tr>
    <tr><td>unfolding residual</td><td>{rep['max_edge_error']:.3f}</td>
        <td>how far the pattern is from developable</td></tr>
    <tr><td>flat sheet</td>
        <td>{rep['sheet_w']:.2f} &times; {rep['sheet_h']:.2f}</td>
        <td>{rep['area_ratio']:.2f}&times; the target's area</td></tr>
    <tr><td>pleats along</td><td>{'UV'[rep.get('axis', 0)]}</td>
        <td>chosen automatically, by drift</td></tr>
    <tr><td>inverted triangles</td><td>{rep.get('flipped', 0)}</td>
        <td>must be zero, or the pattern cannot be cut out</td></tr>
  </table>
</section>""")

    html = f"""<title>Surface Corrugation Validation</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --mut:#666; --line:#ddd; --card:#fafafa; }}
:root:not([data-theme="light"]) {{}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#16181d; --fg:#e8e8ea; --mut:#9aa0a6; --line:#33363d; --card:#1d2026; }}
}}
:root[data-theme="dark"] {{
  --bg:#16181d; --fg:#e8e8ea; --mut:#9aa0a6; --line:#33363d; --card:#1d2026; }}
body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem;
  max-width:1180px; font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }}
h1 {{ font-size:1.6rem; margin:0 0 .25rem; }}
h2 {{ font-size:1.15rem; margin:2.5rem 0 .75rem; border-bottom:1px solid var(--line);
  padding-bottom:.4rem; }}
.lede {{ color:var(--mut); max-width:70ch; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:1rem; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--line);
  border-radius:8px; padding:.6rem; }}
img {{ width:100%; height:auto; display:block; border-radius:4px; }}
figcaption {{ font-size:.82rem; color:var(--mut); margin-top:.5rem; }}
table {{ width:100%; border-collapse:collapse; margin-top:.9rem; font-size:.88rem; }}
td {{ padding:.3rem .5rem; border-bottom:1px solid var(--line); }}
td:first-child {{ color:var(--mut); width:12rem; }}
td:nth-child(2) {{ font-variant-numeric:tabular-nums; font-weight:600; }}
td:last-child {{ color:var(--mut); }}
.badge {{ font-size:.72rem; font-weight:600; padding:.15rem .5rem; border-radius:99px;
  vertical-align:middle; margin-left:.4rem; }}
.ok {{ background:#1f7a3d22; color:#2e9e57; }}
.warn {{ background:#a1740022; color:#c79000; }}
.bad {{ background:#a12d2d22; color:#d05353; }}
.note {{ background:var(--card); border-left:3px solid var(--line);
  padding:.75rem 1rem; border-radius:0 6px 6px 0; margin:1.25rem 0; }}
</style>
<h1>Surface Corrugation — validation</h1>
<p class="lede">Each row generates a corrugation, takes the crease pattern the
operator hands out, folds it with the Bending Paper solver, and compares the
result with the shape that was displayed. Grid {NU}&times;{NV}, pleat depth 0.12,
full drive.</p>
<div class="note"><b>Read column 1 against column 3.</b> Column 1 is not a folded
state — it is built directly from the target surface by offsetting along its
normals, and the crease pattern is flattened <em>from</em> it. The operator
displays the two together, which implies the pattern folds into that shape.
<b>Only the plane satisfies that</b>, because only the plane's flattening is an
isometry. For curved targets the pattern is not developable, so its lowest-energy
folded state is a different, shallower shape — visible in column 3.</div>
<div class="note">This is the boundary of P6a, not a bug to be tuned out. Getting
columns 1 and 3 to agree needs the pattern to be developable <em>by
construction</em> — optimising the panel geometry of a generalised Miura until
its folded state matches the target (Dudte et al. 2016), rather than building a
shape and hunting for a pattern afterwards.</div>
{''.join(body)}
"""
    with _io.open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)


sys.exit(main())
