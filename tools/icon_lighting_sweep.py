"""Sweep icon lighting and write an HTML contact sheet to judge it by.

The icons fail in two DIFFERENT ways depending on the subject, which is
why fixing one made the other worse:

* **Coloured subjects wash out.**  AgX at -0.5 drives every channel to
  clipping, so the saddle-polyhedron palette -- (0.85, 0.30, 0.24) and
  friends -- renders at a measured saturation of 0.095.  That is the
  problem the `Standard` / -3.5 change was aimed at.
* **White subjects go flat.**  The rig runs TWO rim lights at 750 W each
  against a key of 320 and a fill of 70.  Standard three-point practice
  is a dominant key, a fill at roughly half to a quarter of it, and a
  rim used for edge separation only; here the rims are more than twice
  the key, so their wrap light fills the shadow side and a white subject
  loses every gradient.  Exposure cannot fix that -- it scales all of it
  together, and pulling exposure down just makes a dark flat icon.

So a variant has to be judged on BOTH at once, and every variant here
renders the same solid twice: once in its face palette, once in white
plastic.  The sweep covers the things that actually move those two
numbers -- view transform (the tone curve is what clips the colour),
exposure, rim strength relative to key, fill ratio and ambient -- and
measures each render so the contact sheet carries numbers as well as
pictures:

* **saturation** -- mean HSV saturation over the opaque pixels.  Higher
  is more colour surviving the tone curve.
* **shading contrast** -- standard deviation of luminance over the
  opaque pixels.  Higher is more form-describing gradient, and this is
  the one the rim lights destroy.

    blender --background --factory-startup --python tools/icon_lighting_sweep.py

Writes an `index.html` plus its PNGs to a scratch directory (override
with the `ICON_SWEEP_OUT` environment variable).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in ("tools", "docs", "math_art"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import bpy                                                # noqa: E402
import numpy as np                                        # noqa: E402
import bake_menu_icons as bmi                             # noqa: E402
import render_docs as rd                                  # noqa: E402

#: Scratch renders go to a temp directory, NOT to `dev/`.  `dev/` is
#: where the build zips live and nothing else belongs there.
OUT = os.environ.get("ICON_SWEEP_OUT") or os.path.join(
    os.path.expandvars(r"%TEMP%"), "icon_lighting_sweep")
OUT = os.path.normpath(OUT)

RES = 160

#: The subject.  A spiky solid with real curvature, and one whose face
#: palette exercises the whole colour range -- so the same render tells
#: us about saturation, and its white twin tells us about shading.
SUBJECT = 'TETRAHEDRAL_DECAHEDRON'
ORIENT = (math.radians(75), math.radians(10), math.radians(20))

#: Rig defaults, as render_docs sets them, so a variant can scale them.
BASE = {"Key Light": 320.0, "Fill Light": 70.0, "Rim Light L": 750.0,
        "Rim Light R": 750.0, "Top Light": 150.0}

#: Tone curves worth trying.  AgX is the docs default and the one that
#: clips the palette; Standard is what the icons ship with now; Khronos
#: PBR Neutral exists precisely to roll highlights off WITHOUT the hue
#: shift and desaturation a filmic curve introduces, which is the exact
#: complaint here.  Any name Blender does not offer is skipped.
TONE = [('AgX', -0.5), ('AgX', -1.5), ('Standard', -2.0),
        ('Standard', -3.0), ('Standard', -3.5),
        ('Khronos PBR Neutral', -1.0), ('Khronos PBR Neutral', -2.0),
        ('Filmic', -0.5)]

RIMS = (1.00, 0.35, 0.10)


_HAVE_VT = {}


def _have_vt(name):
    """Is this view transform in the loaded OCIO config?

    `view_transform` is a dynamic enum -- its items come from the colour
    config at runtime, so `bl_rna.properties[...].enum_items` is EMPTY
    and introspecting it rejects every name including the ones in use.
    The only reliable probe is to assign it and see whether it sticks.
    """
    if name not in _HAVE_VT:
        vs = bpy.context.scene.view_settings
        keep = vs.view_transform
        try:
            vs.view_transform = name
            _HAVE_VT[name] = (vs.view_transform == name)
        except TypeError:
            _HAVE_VT[name] = False
        vs.view_transform = keep
    return _HAVE_VT[name]


def _available(names):
    """Keep only the view transforms this Blender actually has."""
    keep, dropped = [], []
    for vt, ex in names:
        (keep if _have_vt(vt) else dropped).append((vt, ex))
    if dropped:
        print("skipping unavailable view transforms: %s"
              % sorted({vt for vt, _ in dropped}), flush=True)
    return keep


def _variants():
    """The grid, built once the available tone curves are known."""
    out = []
    for vt, ex in _available(TONE):
        for rim in RIMS:
            out.append(dict(vt=vt, ex=ex, rim=rim, fill=1.0, world=1.0,
                            study='tone'))
    # A second, smaller study: rim already tamed, push fill and ambient
    # instead, to see whether soft light can carry the shadow side.
    for vt, ex in _available([('AgX', -0.5), ('Standard', -2.5)]):
        for fill in (0.4, 2.0):
            for world in (1.0, 12.0):
                out.append(dict(vt=vt, ex=ex, rim=0.35, fill=fill,
                                world=world, study='fill'))
    return out


def _apply(v):
    scene = bpy.context.scene
    scene.view_settings.view_transform = v['vt']
    scene.view_settings.exposure = v['ex']
    for name, energy in BASE.items():
        o = bpy.data.objects.get(name)
        if o is None:
            continue
        k = v['rim'] if name.startswith("Rim") else (
            v['fill'] if name.startswith("Fill") else 1.0)
        o.data.energy = energy * k
    w = scene.world
    if w is not None and w.use_nodes:
        bg = w.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs["Color"].default_value = (
                0.004 * v['world'], 0.004 * v['world'],
                0.005 * v['world'], 1.0)


def measure(path):
    """Mean saturation and luminance spread over the opaque pixels.

    Both are computed on the WRITTEN icon rather than on the render
    buffer, so what is measured is exactly what ships.
    """
    img = bpy.data.images.load(path)
    try:
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        px = buf.reshape(h * w, 4)
    finally:
        bpy.data.images.remove(img)
    rgb = px[px[:, 3] > 0.5, :3]
    if not len(rgb):
        return dict(sat=0.0, contrast=0.0, mean=0.0)
    hi = rgb.max(axis=1)
    lo = rgb.min(axis=1)
    sat = np.where(hi > 1e-6, (hi - lo) / np.maximum(hi, 1e-6), 0.0)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return dict(sat=float(sat.mean()), contrast=float(lum.std()),
                mean=float(lum.mean()))


def _uniform_white(obj):
    """Replace the face palette with one neutral plastic."""
    m = bpy.data.materials.new("sweep_white")
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.84, 0.84, 0.86, 1.0)
    b.inputs["Roughness"].default_value = 0.38
    obj.data.materials.clear()
    obj.data.materials.append(m)


def _shoot(v, index, white):
    """Build the subject, apply the variant, render, measure."""
    rd.clear_sculpts()
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.saddle_polyhedron_add(
        solid=SUBJECT, face_style='MINIMAL', density=3, smoothness=25,
        layout_kind='SINGLE', smooth=True, colour_by='FACE')
    obj = bpy.context.view_layer.objects.active
    obj.rotation_euler = ORIENT
    if white:
        _uniform_white(obj)
    _apply(v)
    name = "v%02d%s.png" % (index, "w" if white else "c")
    bmi._render_to(os.path.join(OUT, name))
    return name, measure(os.path.join(OUT, name))


def main():
    os.makedirs(OUT, exist_ok=True)
    bmi.RES = RES
    bmi._setup()
    records = []
    for i, v in enumerate(_variants()):
        try:
            colour, cstat = _shoot(v, i, white=False)
            white, wstat = _shoot(v, i, white=True)
        except Exception as exc:
            print("[%02d] FAIL %s" % (i, exc), flush=True)
            continue
        rec = dict(v)
        rec.update(id="v%02d" % i, colour=colour, white=white,
                   sat=cstat['sat'], colour_contrast=cstat['contrast'],
                   white_contrast=wstat['contrast'],
                   white_mean=wstat['mean'])
        records.append(rec)
        print("[%02d] %-20s ex%+.1f rim x%.2f fill x%.1f amb x%-2g | "
              "sat %.3f  colour-contrast %.3f  white-contrast %.3f"
              % (i, v['vt'], v['ex'], v['rim'], v['fill'], v['world'],
                 rec['sat'], rec['colour_contrast'], rec['white_contrast']),
              flush=True)

    with open(os.path.join(OUT, "sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)
    _write_html(records)
    print("\n%d variants (%d renders) -> %s"
          % (len(records), 2 * len(records),
             os.path.join(OUT, "index.html")))


def _write_html(records):
    html = ["<title>Icon lighting sweep</title>",
            "<style>body{background:#2b2b2b;color:#ddd;"
            "font:13px system-ui;margin:24px}"
            ".g{display:grid;grid-template-columns:repeat(auto-fill,"
            "minmax(230px,1fr));gap:18px}"
            ".c{background:#3a3a3a;border-radius:8px;padding:10px}"
            ".c img{width:104px;height:104px;background:#4a4a4a;"
            "border-radius:4px}"
            ".c div{margin-top:8px;font-size:11px;color:#bbb}</style>",
            "<h1>Icon lighting sweep &mdash; %s</h1>" % SUBJECT,
            "<p>Left: face palette. Right: white plastic. Rig defaults "
            "are rim x1.00 (two rims at 750&nbsp;W against a 320&nbsp;W "
            "key).</p>", "<div class=g>"]
    for r in records:
        html.append(
            '<div class=c><img src="%s"><img src="%s">'
            '<div>%s ex%+.1f | rim x%.2f | fill x%.1f | amb x%g<br>'
            'sat %.3f &middot; white-contrast %.3f</div></div>'
            % (r['colour'], r['white'], r['vt'], r['ex'], r['rim'],
               r['fill'], r['world'], r['sat'], r['white_contrast']))
    html.append("</div>")
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))


main()
