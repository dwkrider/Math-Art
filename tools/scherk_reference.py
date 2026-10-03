"""Dump the Scherk-Collins engine's mid-surface as JSON, for the
browser port's parity test.

    python tools/scherk_reference.py [out.json]

With no argument it writes to stdout, which is how
`tests/web/test_scherk.mjs` runs it: the reference is then always the
engine as it stands on this checkout, so a change to the mathematics
in `math_art/minsurf/scherk.py` fails the site gate until
`web/js/scherk-math.js` follows.

Only the mid-surface is dumped -- `generate_sculpture(return_grids=True)`
stops exactly there. The thickness, rims and welds that the generator
adds afterwards are a Blender-side construction the page does not
reproduce; it thickens with the site's own solidifier instead.
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "math_art"))

from minsurf.scherk import (PRESETS, Params, XY_SCALE, fit_transform,   # noqa: E402
                            generate_sculpture, ring_closes)

# Every preset, plus a few combinations that exercise the awkward
# corners: an open tower with a fractional phase, a single branch (the
# degenerate order-one saddle), a flange wide enough to cut every hole
# open, and a part-warp that does not close.
EXTRA = [
    ("phase-cut", dict(branches=2, storeys=3, height=1.5, flange=1.2,
                       warp=0, twist=0, azimuth=0, phase=0.37, detail=4)),
    ("one-branch", dict(branches=1, storeys=2, height=1.5, flange=1.5,
                        warp=0, twist=45, azimuth=10, detail=3)),
    ("wide-flange", dict(branches=3, storeys=2, height=1.0, flange=2.4,
                         warp=0, twist=0, azimuth=0, detail=4)),
    # A NARROW flange is what opens the holes: levels with
    # sin(pi f) > sinh(W)^2 have no curve at all, and W = flange + 0.1
    # drops below asinh(1) once the flange goes under 0.781. These two
    # exercise that path, which changes the surface's topology rather
    # than its shape.
    ("narrow-flange", dict(branches=2, storeys=2, height=1.5, flange=0.7,
                           warp=0, twist=0, azimuth=0, detail=4)),
    ("narrow-warped", dict(branches=3, storeys=4, height=1.2, flange=0.72,
                           warp=360, twist=180, azimuth=45, detail=3)),
    ("part-warp", dict(branches=2, storeys=4, height=1.2, flange=1.1,
                       warp=200.0, twist=90, azimuth=30, detail=4)),
    ("stretched", dict(branches=2, storeys=2, height=1.5, flange=1.5,
                       warp=0, twist=0, azimuth=0, detail=3,
                       scale_x=1.4, scale_y=0.7, scale_z=1.2,
                       global_scale=1.3)),
]


def fl(pt):
    return [float(pt[0]), float(pt[1]), float(pt[2])]


def dump_case(label, kw):
    p = Params(**kw)
    g, R, m = generate_sculpture(p, return_grids=True)
    # keys are (cell, branch); sort them so the two sides agree on order
    rows_out = {}
    for (cidx, j), rows in sorted(g.items()):
        rows_out["%d,%d" % (cidx, j)] = [
            None if row is None else [fl(pt) for pt in row] for row in rows]
    allpts = [pt for rows in g.values() for row in rows
              if row is not None for pt in row]
    centre, factor = fit_transform(allpts, p.global_scale)
    return {
        "label": label,
        "kwargs": kw,
        "R": R,
        "m": m,
        "closes": bool(ring_closes(p)),
        "grids": rows_out,
        "fit": {"centre": fl(centre), "factor": float(factor)},
    }


def main():
    out = {
        "xy_scale": float(XY_SCALE),
        "presets": {k: [v[0], v[1]] for k, v in PRESETS.items()},
        "cases": [],
        "closes": [],
    }

    for name, (_label, kw) in sorted(PRESETS.items()):
        # the presets are drawn at a low detail: the point is the
        # mathematics, and a depth-11 demo would be megabytes
        kw2 = dict(kw)
        kw2["detail"] = min(int(kw2.get("detail", 5)), 3)
        out["cases"].append(dump_case("preset:" + name, kw2))

    for label, kw in EXTRA:
        out["cases"].append(dump_case(label, kw))

    # the closure rule on its own, over a grid of twists and warps
    for b in (1, 2, 3, 4):
        for storeys in (2, 3, 7):
            for twist in (0, 90, 135, 180, 270, 360):
                for warp in (0, 180, 360):
                    p = Params(branches=b, storeys=storeys, twist=twist,
                               warp=warp)
                    out["closes"].append(
                        {"branches": b, "storeys": storeys, "twist": twist,
                         "warp": warp, "closes": bool(ring_closes(p))})

    text = json.dumps(out)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s (%.1f MB)" % (sys.argv[1], len(text) / 1e6))
    else:
        sys.stdout.write(text)


main()
