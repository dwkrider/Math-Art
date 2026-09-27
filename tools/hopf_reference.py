"""Dump the Hopf fibration engine's output as JSON, for the browser
port's parity test.

    python tools/hopf_reference.py [out.json]

With no argument it writes to stdout, which is how
`tests/web/test_hopf.mjs` runs it: the reference is then always the
generator as it stands on this checkout, so a change to the
mathematics in `math_art/hopf_fibration_generator.py` fails the site
gate until `web/js/hopf-math.js` follows.

The RANDOM preset is left out on purpose: it draws from numpy's
Generator, whose stream a browser cannot reproduce, so the web module
does not offer it either.
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "math_art"))

import hopf_fibration_generator as H                          # noqa: E402

PRESETS = ["LATITUDES", "FLOWER", "GREATCIRCLE", "CAP", "LOXODROME",
           "CURL", "FIBONACCI", "TETRA", "OCTA", "CUBE", "ICOSA", "DODECA"]

# (label, kwargs for build_fibers) -- the combinations the page can ask
# for, including the awkward ones: a near-axis fibre kept as an arc, a
# (P, Q) torus curve, both chiralities at once, and a mid-flow rotation.
CASES = [
    ("latitudes", dict(preset="LATITUDES", n_lat=5, n_fiber=12, samples=64)),
    ("latitudes-rot", dict(preset="LATITUDES", n_lat=4, n_fiber=8, samples=64,
                           s3_rot=37.5)),
    ("latitudes-axis", dict(preset="LATITUDES", n_lat=5, n_fiber=8, samples=96,
                            lat_min=6.0, lat_max=170.0, include_axis=True)),
    ("flower", dict(preset="FLOWER", n_fiber=16, samples=64)),
    ("greatcircle-both", dict(preset="GREATCIRCLE", n_fiber=10, samples=64,
                              chirality="BOTH")),
    ("cap", dict(preset="CAP", n_fiber=24, samples=48, lat_min=25.0,
                 lat_max=75.0)),
    ("loxodrome", dict(preset="LOXODROME", n_fiber=40, samples=48,
                       extra={"turns": 4.0})),
    ("curl", dict(preset="CURL", n_fiber=30, samples=48,
                  extra={"curl_lobes": 6, "curl_amp": 22.0})),
    ("fibonacci", dict(preset="FIBONACCI", n_fiber=24, samples=48)),
    ("icosa-pq", dict(preset="ICOSA", samples=96, P=2, Q=3)),
    ("dodeca-left", dict(preset="DODECA", samples=48, chirality="LEFT")),
    ("tetra", dict(preset="TETRA", samples=48)),
    ("octa", dict(preset="OCTA", samples=48)),
    ("cube", dict(preset="CUBE", samples=48)),
]


def fl(a):
    return [[float(x) for x in row] for row in a]


def main():
    out = {
        "tilt": list(H._TILT),
        "presets": PRESETS,
        "base_points": {},
        "fiber_s3": [],
        "cases": [],
        "colors": [],
    }

    # the providers, before any tilt
    for p in PRESETS:
        pts = H.base_points(p, 5, 12, 20.0, 160.0,
                            {"turns": 4.0, "curl_lobes": 6, "curl_amp": 22.0})
        out["base_points"][p] = [[float(x) for x in q] for q in pts]

    # the kernel on its own: a few base points, both chiralities, some
    # (P, Q), with and without the S^3 rotation
    for base in [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, -0.6, 0.74162),
                 (-0.5, 0.5, -0.70711)]:
        b = tuple(float(x) for x in H._normalize(base))
        for (P, Q, chi, rot) in [(1, 1, "RIGHT", 0.0), (1, 1, "LEFT", 0.0),
                                 (2, 3, "RIGHT", 0.0), (1, 1, "RIGHT", 40.0)]:
            X = H.fiber_s3(b, 32, P, Q, chi)
            if rot:
                X = H._s3_rotate(X, H._quat_left(rot * 3.141592653589793 / 180.0))
            out["fiber_s3"].append({
                "base": list(b), "P": P, "Q": Q, "chirality": chi, "rot": rot,
                "s3": fl(X), "projected": fl(H.stereographic(X)),
            })

    # A general unit quaternion, which the generator's own flow never
    # uses -- `_quat_left` always has j = k = 0, so the j and k terms of
    # the product are dead in every picture. They are live code in the
    # port, so they are exercised here: without this a conjugated
    # quaternion product passes every other check.
    out["quat_rotations"] = []
    for q in [(0.5, 0.5, 0.5, 0.5),
              (0.182574, -0.365148, 0.547723, 0.730297),
              (0.0, 0.707107, 0.0, 0.707107)]:
        b = tuple(float(x) for x in H._normalize((0.3, -0.6, 0.74162)))
        X = H.fiber_s3(b, 24, 1, 1, "RIGHT")
        out["quat_rotations"].append({
            "q": list(q), "base": list(b),
            "s3": fl(H._s3_rotate(X, q)),
        })

    for label, kw in CASES:
        fibers, bases, closed, stats = H.build_fibers(
            return_stats=True, **kw)
        out["cases"].append({
            "label": label, "kwargs": kw,
            "fibers": [fl(f) for f in fibers],
            "bases": [[float(x) for x in b] for b in bases],
            "closed": [bool(c) for c in closed],
            "dropped": int(stats["dropped"]),
        })

    for style in ("RAINBOW", "PASTEL", "MONO"):
        for base in [(0.0, 0.0, 1.0), (0.6, 0.8, 0.0), (-0.3, 0.2, -0.93),
                     (0.0, -1.0, 0.0)]:
            b = tuple(float(x) for x in H._normalize(base))
            out["colors"].append({
                "base": list(b), "style": style,
                "rgb": [float(x) for x in H._palette_rgb(b, style)],
                "param": [float(x) for x in H._param_rgb(0.3, style)],
            })

    text = json.dumps(out)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s (%.1f MB)" % (sys.argv[1], len(text) / 1e6))
    else:
        sys.stdout.write(text)


main()
