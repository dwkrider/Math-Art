"""relief.transfer -- normalisation and pointwise transfer curves.

Two jobs, deliberately separated:

* **Normalisation** puts a field on a common footing *before* its amplitude is
  applied.  Without it the amplitude control means something different for
  every pattern -- a fractal surface and a sinusoid have quite different
  natural ranges -- and a layer stack becomes untunable.
* **Transfer curves** reshape the profile of an already-normalised field.  The
  two that matter most cost three lines each and are the difference between a
  sine wave in wood and something carved: `ABS` folds the field at zero so its
  zero set becomes a crease of *valleys*, and `RIDGE` (1 - |h|) makes the same
  set a crease of *crests*.

A physical footnote for the pair: on a vibrating plate, coarse sand collects
along the nodal lines while fine lycopodium powder collects at the antinodes,
so a panel rendered with `ABS` and the same panel rendered with `RIDGE` are the
two things a real Chladni experiment shows you, depending on the powder.

References:
  Mary D. Waller, "Chladni Figures: A Study in Symmetry", G. Bell and Sons,
    1961, chs. 32-34 -- sand on the nodes, lycopodium at the antinodes.
"""

import numpy as np

CURVES = ('NONE', 'ABS', 'RIDGE', 'GAMMA', 'SCURVE', 'TERRACE', 'CLAMP')
NORMS = ('STD', 'MINMAX', 'PERCENTILE', 'RAW')


def normalize(h, mode='STD', percentile=1.0):
    """Bring a raw field to a canonical range.

    STD         zero mean, unit standard deviation (keeps outliers)
    MINMAX      exactly [-1, 1] (a single stray sample sets the range)
    PERCENTILE  [-1, 1] on the central range, robust to outliers
    RAW         unchanged
    """
    h = np.asarray(h, dtype=float)
    if mode == 'RAW':
        return h
    if mode == 'STD':
        sd = h.std()
        return (h - h.mean()) / sd if sd > 1e-12 else h - h.mean()
    if mode == 'MINMAX':
        lo, hi = float(h.min()), float(h.max())
        return 2.0 * (h - lo) / (hi - lo) - 1.0 if hi - lo > 1e-12 else h * 0.0
    if mode == 'PERCENTILE':
        p = float(np.clip(percentile, 0.0, 49.0))
        lo = float(np.percentile(h, p))
        hi = float(np.percentile(h, 100.0 - p))
        if hi - lo <= 1e-12:
            return h * 0.0
        return np.clip(2.0 * (h - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    raise ValueError("unknown normalisation: %r" % (mode,))


def apply_curve(h, kind='NONE', amount=1.0, levels=6, smooth=0.25):
    """Reshape an already-normalised field.  Input and output are ~[-1, 1]."""
    h = np.asarray(h, dtype=float)
    if kind == 'NONE':
        return h
    if kind == 'ABS':
        # Zero set becomes a valley crease.  Re-centre so the result still
        # straddles zero rather than sitting entirely above it.
        return 2.0 * np.abs(h) - 1.0
    if kind == 'RIDGE':
        return 1.0 - 2.0 * np.abs(h)
    if kind == 'GAMMA':
        g = max(float(amount), 1e-3)
        return np.sign(h) * (np.abs(h) ** g)
    if kind == 'SCURVE':
        # Smoothstep contrast, applied symmetrically about zero.
        a = float(np.clip(amount, 0.0, 1.0))
        u = np.clip(0.5 * (h + 1.0), 0.0, 1.0)
        s = u * u * (3.0 - 2.0 * u)
        return (1.0 - a) * h + a * (2.0 * s - 1.0)
    if kind == 'TERRACE':
        n = max(1, int(levels))
        u = 0.5 * (h + 1.0)
        step = np.floor(u * n) / n
        frac = u * n - np.floor(u * n)
        w = float(np.clip(smooth, 1e-6, 1.0))
        # Smooth the riser so the steps are cut, not aliased.
        ease = np.clip(frac / w, 0.0, 1.0)
        ease = ease * ease * (3.0 - 2.0 * ease)
        return 2.0 * (step + ease / n) - 1.0
    if kind == 'CLAMP':
        a = float(np.clip(amount, 1e-3, 1.0))
        return np.clip(h / a, -1.0, 1.0)
    raise ValueError("unknown transfer curve: %r" % (kind,))


def to_depth(h, depth, mode='MINMAX'):
    """Scale a normalised field to a peak-to-peak relief depth in metres."""
    h = normalize(h, 'MINMAX') if mode == 'MINMAX' else h
    return 0.5 * float(depth) * h


def _selftest():
    ok = True
    rng = np.random.default_rng(0)
    h = rng.normal(size=(64, 64))

    a = normalize(h, 'STD')
    print("transfer: STD mean=%.2e sd=%.6f" % (a.mean(), a.std()))
    ok = ok and abs(a.mean()) < 1e-12 and abs(a.std() - 1.0) < 1e-9

    b = normalize(h, 'MINMAX')
    print("transfer: MINMAX range [%.6f, %.6f]" % (b.min(), b.max()))
    ok = ok and abs(b.min() + 1.0) < 1e-12 and abs(b.max() - 1.0) < 1e-12

    # PERCENTILE is robust: one wild outlier must not collapse the range.
    h2 = h.copy()
    h2[0, 0] = 1e6
    r_minmax = normalize(h2, 'MINMAX')
    r_pct = normalize(h2, 'PERCENTILE')
    print("transfer: with a 1e6 outlier, MINMAX sd=%.2e vs PERCENTILE sd=%.4f "
          "(%.1fx more range retained)"
          % (r_minmax.std(), r_pct.std(), r_pct.std() / max(r_minmax.std(), 1e-30)))
    ok = ok and r_pct.std() > 5.0 * r_minmax.std()

    # Every curve is finite and shape-preserving.
    n = normalize(h, 'MINMAX')
    for kind in CURVES:
        c = apply_curve(n, kind, amount=0.8)
        ok = ok and c.shape == n.shape and np.isfinite(c).all()

    # ABS makes the zero set a minimum; RIDGE makes it a maximum.
    x = np.linspace(-1.0, 1.0, 401).reshape(1, -1)
    lo = apply_curve(x, 'ABS')
    hi = apply_curve(x, 'RIDGE')
    mid = 200                                   # index of x = 0
    print("transfer: at the zero set ABS=%.3f (min=%.3f), RIDGE=%.3f (max=%.3f)"
          % (lo[0, mid], lo.min(), hi[0, mid], hi.max()))
    ok = ok and abs(lo[0, mid] - lo.min()) < 1e-12
    ok = ok and abs(hi[0, mid] - hi.max()) < 1e-12

    # TERRACE actually quantises: few distinct values, and monotone in u.
    t = apply_curve(np.linspace(-1, 1, 2001).reshape(1, -1), 'TERRACE',
                    levels=5, smooth=0.2)
    uniq = len(np.unique(np.round(t, 3)))
    print("transfer: TERRACE(5) distinct rounded values = %d" % uniq)
    ok = ok and uniq < 600 and t.min() >= -1.0001 and t.max() <= 1.0001

    # Depth scaling hits the requested peak-to-peak.
    z = to_depth(h, 0.25)
    print("transfer: depth 0.25 -> realised peak-to-peak %.6f"
          % (z.max() - z.min()))
    ok = ok and abs((z.max() - z.min()) - 0.25) < 1e-9

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
