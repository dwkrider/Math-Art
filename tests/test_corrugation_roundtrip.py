"""Does a corrugation's crease pattern fold back into its corrugation?

Run under Blender, against the installed or worktree add-on:

    blender --background --factory-startup --python \
        tests/test_corrugation_roundtrip.py

THE CLAIM BEING TESTED.  `mesh.corrugation_add` emits two objects: a
pleated form fitted to a target surface, and the flat crease pattern it
says produces that form.  Folding the pattern should give the form back.
Everything else the operator reports -- fit error, flattening residual,
area ratio -- measures the corrugation against ITSELF and can look fine
while this fails, which is exactly what happened: the reported numbers
were at their best on the catenoid while its fold was at its worst.

WHAT IS MEASURED, per target:

    depth      folded z-extent as a fraction of the intended one.  The
               single most legible number: a fold that reaches 0.5 has
               got the shape half right whatever else is true.
    shape      Kabsch-aligned per-vertex deviation, as a fraction of
               model size.  Alignment is necessary because the fold is
               free to sit anywhere in space; reflection is allowed and
               reported, since a mirrored fold is a real result and not
               a scoring artefact.
    strain     worst axial strain.  Paper does not stretch, so a large
               value means the sheet is being torn to reach the target
               rather than folded to it.
    drift      the reviewer's test: seed the solver AT the intended form
               and relax.  If it walks away, the pattern does not encode
               that shape at all, and no solver setting will fix it.
               This separates "the solve is short" from "the pattern is
               wrong", which the other numbers cannot.

PLANE is the control and is the row to read first: a pleated plane is
developable, so its pattern is exact and every column must be near
perfect.  A failure there is a bug; a poor row for a curved target is
the Theorema Egregium, which no amount of solver work will repeal.
"""

import os
import sys
import time

import numpy as np

import bpy

# Run against the checkout this script lives in, not whatever build
# happens to be installed -- a validation harness that silently tested
# the wrong code would be worse than none.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

TARGETS = ("PLANE", "HYPAR", "SCHERK", "SPHERE", "CATENOID")
NU = NV = 12


def _align(got, want):
    """Kabsch fit of `got` onto `want`; returns (rms, max, reflected)."""
    A = got - got.mean(0)
    B = want - want.mean(0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    det = float(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(det)]) @ U.T
    dev = np.linalg.norm((R @ A.T).T - B, axis=1)
    scale = float(np.ptp(want, axis=0).max()) or 1.0
    return float(dev.mean()) / scale, float(dev.max()) / scale, det < 0


def _co(obj):
    return np.array([tuple(v.co) for v in obj.data.vertices], dtype=float)


def main():
    try:
        import math_art
    except ImportError:
        print(f"math_art is not importable from {_ROOT}")
        return 1
    try:
        math_art.register()
    except Exception as exc:                          # already enabled
        print(f"[note] register: {exc}")
    from math_art import crease

    print(f"{'target':9s} {'depth':>7} {'shape rms':>10} {'max':>7} "
          f"{'strain':>7} {'drift':>7} {'refl':>5} {'secs':>6}")
    worst_plane = 0.0
    rows = []
    for kind in TARGETS:
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        t0 = time.time()
        bpy.ops.mesh.corrugation_add(target=kind, nu=NU, nv=NV, relax=1200)
        corr = bpy.data.objects[f"{kind.title()} Corrugation"]
        pat = bpy.data.objects[f"{kind.title()} Crease Pattern"]
        want = _co(corr)

        bpy.context.view_layer.objects.active = pat
        bpy.ops.object.fold_solve(solver='COMPLIANT', drive=1.0, steps=12,
                                  animate=False, colour_strain=False)
        got = _co(pat)
        rms, mx, refl = _align(got, want)
        depth = float(np.ptp(got[:, 2])) / (float(np.ptp(want[:, 2])) or 1.0)

        # strain, and the drift test, from the engine directly.
        #
        # `axis=None` MATTERS: the operator defaults to Automatic pleat
        # direction, and on several targets it chooses V.  Re-fitting
        # here with the default axis=0 builds a DIFFERENT pattern, and
        # comparing the folded positions against its rest lengths gave
        # nonsense -- strain of 0.87 to 1.39, against the operator's own
        # 5.7e-03 for the same fold.  A harness that quietly measures a
        # different object than the one under test is worse than no
        # harness.
        fr, folded, _rep = crease.corrugate.fit(kind, nu=NU, nv=NV,
                                                amplitude=0.12, iters=1200,
                                                axis=None)
        cf = crease.compliant.CompliantFolder(fr)
        cf.pos = np.asarray(got, dtype=float).copy()
        strain = float(np.abs(cf.edge_strain()).max())
        drift = crease.corrugate.equilibrium_drift(fr, folded, steps=4000)

        el = time.time() - t0
        print(f"{kind:9s} {depth:7.2f} {rms:10.3f} {mx:7.3f} {strain:7.3f} "
              f"{drift:7.3f} {'yes' if refl else 'no':>5} {el:6.1f}")
        rows.append((kind, depth, rms, strain, drift))
        if kind == "PLANE":
            worst_plane = max(rms, abs(1.0 - depth))

    print()
    # The control is the only row that can FAIL rather than merely
    # disappoint: a pleated plane is developable, so its pattern is
    # exact and the fold has no excuse.
    if worst_plane < 0.10:
        print(f"CONTROL OK: the developable case round-trips "
              f"(worst {worst_plane:.3f})")
        ok = True
    else:
        print(f"CONTROL FAILED: a pleated plane is developable and must "
              f"fold back exactly, but is off by {worst_plane:.3f}")
        ok = False

    curved = [r for r in rows if r[0] != "PLANE"]
    print("Curved targets, for the record (these are approximations, and "
          "the plan says so):")
    for kind, depth, rms, strain, drift in curved:
        print(f"  {kind:9s} reaches {depth * 100:3.0f}% of the intended "
              f"depth, shape error {rms:.3f}, pattern drift {drift:.3f}")
    print("\nRESULT: " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


sys.exit(main())
