"""Dump the Chair44 engine's output as JSON, for the browser port's
parity test.

    python tools/chair44_reference.py [out.json]

With no argument it writes to stdout, which is how
`tests/web/test_chair44.mjs` runs it: the reference is then always the
generator as it stands on this checkout, so a change to the mathematics
in `math_art/ifs/chair44.py` fails the site gate until
`web/js/chair44-math.js` follows.

Everything is emitted as floats. The engine works in exact rationals
and the port in doubles, so the test compares numerically, at a
tolerance far below the smallest distance the construction contains.
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "math_art"))

from ifs import chair44 as C                                  # noqa: E402

# The exaggerated mode the site offers, as the Blender layer builds it:
# bases five times over, heights scaled by Relief.
RELIEF = 30.0
DEPTHS = (0, 1, 2, 3)


def fl(x):
    return float(x)


def mesh(verts, faces):
    return {"verts": [[fl(c) for c in p] for p in verts],
            "faces": [list(f) for f in faces]}


def main():
    from fractions import Fraction

    bare_v, bare_f = C.bare_tile_mesh()
    arrow_v, arrow_f, arrow_c = C.arrow_tile_mesh()
    true_v, true_f = C.tile_mesh()
    big_h = (Fraction(RELIEF).limit_denominator(10 ** 6) * C.HEIGHT_TRUE)
    big_v, big_f = C.tile_mesh(C.ETA_SHOWN, big_h)

    out = {
        "relief": RELIEF,
        "centroid": [fl(x) for x in C.CENTROID],
        "eta_true": fl(C.ETA_TRUE),
        "eta_shown": fl(C.ETA_SHOWN),
        "height_true": fl(C.HEIGHT_TRUE),
        "min_contact_span": fl(C.MIN_CONTACT_SPAN),
        # null where the bound is infinite: JSON has no Infinity, and
        # json.dumps' non-standard spelling of it is not parseable by
        # the browser or by node.
        "max_relief": {str(g): (None if C.max_relief(g) == float('inf')
                                else C.max_relief(g))
                       for g in (0.80, 0.90, 0.92, 0.97, 1.0)},
        "arrow_area": {"full": C.arrow_area(0), "half": C.arrow_area(1)},
        "panel_home": [[list(C.panel_home(k)[0]),
                        [fl(x) for x in C.panel_home(k)[1]],
                        C.arrow_half(C.PANELS[k][0],
                                     C._panel_axes(k[0])[2] != k[1])]
                       for k in sorted(C.PANELS, key=lambda k: C.PANELS[k][0])],
        "proper_frames": [[list(r) for r in M] for M in C.PROPER_FRAMES],
        "features": [[[fl(x) for x in c], list(n), a]
                     for c, n, a in C.features()],
        "meshes": {
            "bare": mesh(bare_v, bare_f),
            "arrow": dict(mesh(arrow_v, arrow_f), colors=list(arrow_c)),
            "true": mesh(true_v, true_f),
            "enlarged": mesh(big_v, big_f),
        },
        "volumes": {
            "bare": fl(C.mesh_volume(bare_v, bare_f)),
            "true": fl(C.mesh_volume(true_v, true_f)),
            "enlarged": fl(C.mesh_volume(big_v, big_f)),
        },
        "patches": {},
        "contacts": {},
    }

    for d in DEPTHS:
        poses = C.patch(d)
        out["patches"][str(d)] = [
            {"G": [list(r) for r in G], "t": list(t), "group": g,
             "frame": C.frame_index(G)}
            for G, t, g in poses]
        if d <= 2:
            out["contacts"][str(d)] = [
                {"i": i, "j": j, "G": [list(r) for r in G], "t": list(t)}
                for i, j, G, t in C.contacts(poses)]
    # the atlas membership the engine's own self-test checks
    atlas = {(tuple(C.frame(p, s)), tuple(off)) for p, s, off in C.ATLAS44}
    poses2 = C.patch(2)
    out["depth2_all_in_atlas"] = all(
        (tuple(G), tuple(t)) in atlas for _i, _j, G, t in C.contacts(poses2))

    text = json.dumps(out)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s (%.1f MB)" % (sys.argv[1], len(text) / 1e6))
    else:
        sys.stdout.write(text)


main()
