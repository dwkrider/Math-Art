# The knot build pipeline.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# Braid word -> closure embedding -> resample -> relax -> resample -> fit.

import numpy as np

from .braid import braid_closure_points, closure_components, parse_letters
from .relax import relax_knot
from .resample import resample_closed


def build_knot(braid_text, samples=240, iters=150, repel=0.35,
               scale=1.0, mirror=False):
    word = parse_letters(braid_text)
    if closure_components(word) != 1:
        raise ValueError("braid closure is a link, not a knot")
    pts = braid_closure_points(word)
    P = resample_closed(pts, max(samples, 4 * len(pts) // 3))
    P = relax_knot(P, iters=iters, repel=repel)
    P = resample_closed(P, samples)
    if mirror:
        P[:, 2] = -P[:, 2]
    # Center on the origin and fit within a 2 m cube (max extent
    # 2.0 * scale) so the knot lands framed by default.
    if len(P):
        lo = P.min(axis=0)
        hi = P.max(axis=0)
        cen = 0.5 * (lo + hi)
        ext = float((hi - lo).max())
        s = (2.0 * scale / ext) if ext > 1e-9 else scale
        P = (P - cen) * s
    return P


def _selftest():
    ok = True

    for name, word in (('3_1', 'AAA'), ('4_1', 'AbAb'), ('6_2', 'AAAbAb')):
        P = build_knot(word, samples=120, iters=40)
        lo, hi = P.min(axis=0), P.max(axis=0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        g = (len(P) == 120 and bool(np.all(np.isfinite(P)))
             and cen < 1e-9 and abs(ext - 2.0) < 1e-9
             and float(seg.max()) < 5.0 * float(np.median(seg)))
        ok &= g
        print(f"build: {name} {len(P)} pts, centred |c|={cen:.1e}, "
              f"ext={ext:.6f} {'OK' if g else 'FAIL'}")

    # `scale` multiplies the fitted size, and `mirror` flips z only.
    P = build_knot('AAA', samples=64, iters=20, scale=2.5)
    ext = float(np.max(P.max(axis=0) - P.min(axis=0)))
    Pm = build_knot('AAA', samples=64, iters=20, mirror=True)
    P0 = build_knot('AAA', samples=64, iters=20)
    good = (abs(ext - 5.0) < 1e-9
            and np.allclose(Pm[:, :2], P0[:, :2])
            and np.allclose(Pm[:, 2], -P0[:, 2]))
    ok &= good
    print(f"build: scale ext={ext:.4f} (exp 5), mirror flips z only "
          f"{'OK' if good else 'FAIL'}")

    # A word whose closure is a LINK must be refused by the knot pipeline.
    try:
        build_knot('aa')
        good = False
    except ValueError:
        good = True
    ok &= good
    print(f"build: link word rejected {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("build self-test failed")
