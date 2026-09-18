"""Reference output from the belt trick generator, for the web port.

    python tools/belt_reference.py > reference.json

Imports math_art/belt_trick_generator.py directly -- it guards its bpy
import, so no Blender is needed -- and runs it over a fixed set of cases,
printing what tests/web/test_belt_math.mjs compares web/js/belt-math.js
against.

WHY THIS RUNS AT TEST TIME AND IS NOT A COMMITTED FILE.  The full output
of every solid, axis and turn is 26.9 million numbers, about 512 MB.
And a committed subset goes stale the day the generator changes, with
nothing to say so. Run live, the reference is always the generator as it
stands: change the mathematics on master and the web port's gate fails
until the port follows.

WHAT IS COMPARED, per case:
  - prepare(): the renormalised face distance, the packing limit, the
    width actually built, the end of the straight run, the fit, the
    strip thickness, and every belt's normal, width direction, face
    inradius and face distance, IN ORDER (the order is the colours);
  - each belt's drawn centre line and width direction at several turns.
    The cross-section rows are those two combined by identical arithmetic
    in both languages, so comparing them adds nothing but bulk -- except
    once, for one case, to check that combination itself;
  - diagnose(): the level, the measurements and the warning, for a case
    that reports cleanly and one that warns.

THE TURNS.  Chosen where the construction is hardest -- the S-bends near
120 and 525 degrees, the coil at 300, and around 360, the one place the
width pass sits on a branch threshold (at exactly 360 degrees 44 samples
of the cube have a tangential speed within 100x of the 1e-13 cut-off).
"""
import importlib.util
import json
import math
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "belt", os.path.join(PROJ, "math_art", "belt_trick_generator.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

DEG = math.pi / 180.0

# (label, prepare kwargs, turns in degrees)
FULL = (60.0, 300.0, 359.5, 360.0, 360.5, 525.0, 719.5, 720.0)
SHORT = (120.0, 300.0)
CASES = []
for kind in ("TWO", "CUBE", "ICOSA"):
    for axis in "XYZ":
        CASES.append(("%s/%s" % (kind, axis),
                      dict(kind=kind, spin_axis=axis), FULL))
CASES += [
    ("TETRA/Z", dict(kind="TETRA"), SHORT),
    ("OCTA/Z", dict(kind="OCTA"), SHORT),
    ("DODECA/Z", dict(kind="DODECA"), SHORT),
    ("SNUB/Z", dict(kind="SNUB"), SHORT),
    ("SNUB/Z/mirror", dict(kind="SNUB", mirror=True), SHORT),
    ("GEO/Z/f1", dict(kind="GEO", freq=1), SHORT),
    ("GEO/Z/f2", dict(kind="GEO", freq=2), SHORT),
]

NS = 160


def line(prep, belt, psi):
    """The drawn centre line and the unit width direction at each sample,
    exactly as belt_rows builds the rows from them."""
    u, w, _, hf = belt
    rows, info = B.belt_rows(psi, u, w, 1.0, hf, prep["width"],
                             prep["reach"], NS, 3, r_min=prep["r_min"],
                             n=prep["n"], m=prep["m"],
                             smoothing=prep["smoothing"])
    P = info["P"]
    # recover the width direction from the 3-sample row: it is
    # (row[2] - row[1]) / (width / 2), the same unit vector
    half_w = 0.5 * prep["width"]
    d = [tuple((row[2][k] - row[1][k]) / half_w for k in range(3))
         for row in rows]
    return [list(p) for p in P], [list(x) for x in d]


def main():
    out = {"ns": NS, "cases": []}
    for label, kw, turns in CASES:
        prep = B.prepare(**kw)
        case = {
            "label": label,
            "kwargs": kw,
            "half": prep["half"], "limit": prep["limit"],
            "width": prep["width"], "fit": prep["fit"],
            "r_min": prep["r_min"], "solid_thick": prep["solid_thick"],
            "nverts": len(prep["verts"]), "nfaces": len(prep["faces"]),
            "belts": [[list(u), list(w), rf, hf]
                      for u, w, rf, hf in prep["belts"]],
            "turns": [],
        }
        for t in turns:
            psi = 0.5 * t * DEG
            lines = [line(prep, b, psi) for b in prep["belts"]]
            case["turns"].append({"turn": t,
                                  "P": [l[0] for l in lines],
                                  "dir": [l[1] for l in lines]})
        out["cases"].append(case)

    # The rows themselves, once, to check the combination step.
    prep = B.prepare(kind="CUBE")
    u, w, _, hf = prep["belts"][0]
    rows, _ = B.belt_rows(300.0 * DEG * 0.5, u, w, 1.0, hf, prep["width"],
                          prep["reach"], NS, 11, r_min=prep["r_min"],
                          n=prep["n"], m=prep["m"],
                          smoothing=prep["smoothing"])
    out["rows_check"] = {"kind": "CUBE", "turn": 300.0, "belt": 0,
                         "rows": [[list(p) for p in row] for row in rows]}

    # diagnose(): one clean report and one that warns.
    out["diagnose"] = []
    for label, kw, t in (("CUBE", dict(kind="CUBE"), 300.0),
                         ("GEO", dict(kind="GEO"), 300.0)):
        prep = B.prepare(**kw)
        all_rows = B.build_belts(0.5 * t * DEG, prep["belts"], 1.0,
                                 prep["half"], prep["width"], prep["reach"],
                                 NS, 11, r_min=prep["r_min"], n=prep["n"],
                                 m=prep["m"], smoothing=prep["smoothing"])[3]
        level, meas, warning = B.diagnose(prep, all_rows)
        out["diagnose"].append({"label": label, "kwargs": kw, "turn": t,
                                "level": level, "meas": meas,
                                "warning": warning})

    json.dump(out, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
