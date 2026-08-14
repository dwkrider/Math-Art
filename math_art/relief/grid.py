"""relief.grid -- the sample domain of a relief panel.

A relief panel is a scalar field h(x, y) sampled on a rectangle (or a masked
sub-region of one) and lifted into a surface.  This module owns the sampling
geometry: how many samples, where they sit, which of them are inside the panel
outline, and the border window that fades the relief to a flat rim.

Two rules are enforced here because everything downstream depends on them:

* **Sample cells stay square.**  The vertical resolution is *derived* from the
  aspect ratio, never entered independently.  A non-square cell silently
  distorts every isotropic construction in the engine -- a radially symmetric
  kernel becomes elliptical, a circular drumhead mode becomes an oval, and an
  N-fold symmetric wave sum loses its symmetry.
* **The footprint, not the bounding box, is what gets fitted.**  Fitting the
  whole bounding box into a fixed cube (the convention used elsewhere in this
  add-on for closed surfaces) would make the relief-depth control shrink the
  panel, because Z would compete with X for the fit budget.

References:
  The raised-cosine (Hann) border window is the standard apodisation of
  Julius von Hann, described in Ralph B. Blackman and John W. Tukey, "The
  Measurement of Power Spectra", Dover, 1958 -- used here so a bordered panel
  meets its neighbour with vanishing value *and* vanishing slope.
"""

import math

import numpy as np

SHAPES = ('RECT', 'DISC', 'ELLIPSE', 'ROUNDED_RECT')

# Cap on total samples, so an accidental 4096 does not hang Blender.
MAX_SAMPLES = 2_200_000


def derive_ny(nx, aspect):
    """Vertical sample count that keeps cells square for the given aspect.

    With `nx` samples spanning a width W, the cell pitch is W/(nx-1); the panel
    height is W*aspect, so ny-1 must be the nearest integer number of pitches
    that spans it.  Returns at least 2.
    """
    if nx < 2:
        raise ValueError("nx must be >= 2")
    if aspect <= 0.0:
        raise ValueError("aspect must be positive")
    return max(2, int(round((nx - 1) * aspect)) + 1)


def clamp_resolution(nx, aspect, max_samples=MAX_SAMPLES):
    """Reduce `nx` until nx*ny fits the sample budget.  Returns (nx, ny)."""
    nx = max(2, int(nx))
    ny = derive_ny(nx, aspect)
    while nx > 2 and nx * ny > max_samples:
        nx = max(2, int(nx * math.sqrt(max_samples / float(nx * ny))))
        ny = derive_ny(nx, aspect)
    return nx, ny


def make_grid(width=2.0, aspect=1.0, resolution=256, max_samples=MAX_SAMPLES):
    """Sample the panel rectangle centred on the origin.

    Returns `(X, Y, info)` where X and Y are (ny, nx) coordinate arrays and
    `info` carries the realised resolution and pitch.  The realised values are
    reported because `resolution` and `aspect` are both subject to adjustment
    (the sample cap, and the square-cell rule).
    """
    nx, ny = clamp_resolution(resolution, aspect, max_samples)
    half_w = 0.5 * float(width)
    half_h = 0.5 * float(width) * float(aspect)
    xs = np.linspace(-half_w, half_w, nx)
    ys = np.linspace(-half_h, half_h, ny)
    X, Y = np.meshgrid(xs, ys)
    dx = (2.0 * half_w) / (nx - 1)
    dy = (2.0 * half_h) / (ny - 1)
    info = {
        'nx': nx, 'ny': ny,
        'dx': dx, 'dy': dy,
        'width': 2.0 * half_w,
        'height': 2.0 * half_h,
        'aspect': (2.0 * half_h) / (2.0 * half_w) if half_w > 0 else 1.0,
        'nyquist': math.pi / dx,       # highest representable angular freq
    }
    return X, Y, info


def mask_for(shape, X, Y, corner=0.15):
    """Boolean (ny, nx) mask of the samples inside the panel outline."""
    if shape == 'RECT':
        return np.ones(X.shape, dtype=bool)
    hx = np.abs(X).max()
    hy = np.abs(Y).max()
    if shape == 'DISC':
        r = min(hx, hy)
        return (X * X + Y * Y) <= r * r
    if shape == 'ELLIPSE':
        return ((X / hx) ** 2 + (Y / hy) ** 2) <= 1.0
    if shape == 'ROUNDED_RECT':
        # Distance to the inner rectangle inset by the corner radius.
        rad = float(corner) * min(hx, hy)
        qx = np.maximum(np.abs(X) - (hx - rad), 0.0)
        qy = np.maximum(np.abs(Y) - (hy - rad), 0.0)
        return (qx * qx + qy * qy) <= rad * rad
    raise ValueError("unknown panel shape: %r" % (shape,))


def border_window(X, Y, margin, shape='RECT'):
    """Raised-cosine falloff to zero over `margin` (a fraction of the short
    half-extent).  Returns an array of ones when `margin` is 0.

    The window and its first derivative both vanish at the rim, so a bordered
    panel abuts its neighbour with no step and no crease.
    """
    if margin <= 0.0:
        return np.ones(X.shape)
    hx = np.abs(X).max()
    hy = np.abs(Y).max()
    m = float(margin) * min(hx, hy)
    if m <= 0.0:
        return np.ones(X.shape)
    if shape in ('DISC', 'ELLIPSE'):
        r = np.sqrt((X / hx) ** 2 + (Y / hy) ** 2) * min(hx, hy)
        d = np.clip((min(hx, hy) - r) / m, 0.0, 1.0)
    else:
        dxe = np.clip((hx - np.abs(X)) / m, 0.0, 1.0)
        dye = np.clip((hy - np.abs(Y)) / m, 0.0, 1.0)
        d = np.minimum(dxe, dye)
    return 0.5 - 0.5 * np.cos(math.pi * d)


def wavelength_in_cells(wavelength, info):
    """How many samples span one wavelength.  Below ~4 the layer aliases."""
    return float(wavelength) / max(info['dx'], 1e-12)


def _selftest():
    ok = True

    # Square cells across a sweep of aspect ratios.
    worst = 0.0
    for aspect in (0.25, 0.5, 0.75, 1.0, 1.3333, 1.4142, 2.0, 3.0):
        _, _, info = make_grid(width=2.0, aspect=aspect, resolution=257)
        rel = abs(info['dx'] - info['dy']) / info['dx']
        worst = max(worst, rel)
    # The derived ny rounds to a whole number of cells, so the pitch match is
    # limited by that rounding, not by float error.
    print("grid: worst cell anisotropy over aspect sweep = %.3e" % worst)
    ok = ok and worst < 5e-3

    # Aspect is honoured in the realised extents.
    _, _, info = make_grid(width=2.0, aspect=1.5, resolution=129)
    ok = ok and abs(info['height'] / info['width'] - 1.5) < 1e-12
    print("grid: realised aspect = %.6f (want 1.5)" %
          (info['height'] / info['width']))

    # Sample cap.
    nx, ny = clamp_resolution(4096, 1.0, max_samples=1_000_000)
    print("grid: 4096 clamped to %dx%d = %d samples" % (nx, ny, nx * ny))
    ok = ok and nx * ny <= 1_000_000

    # Masks have the expected areas.
    X, Y, info = make_grid(width=2.0, aspect=1.0, resolution=401)
    frac_disc = mask_for('DISC', X, Y).mean()
    print("grid: disc area fraction = %.4f (want pi/4 = %.4f)" %
          (frac_disc, math.pi / 4.0))
    ok = ok and abs(frac_disc - math.pi / 4.0) < 5e-3
    ok = ok and mask_for('RECT', X, Y).all()

    # Border window: 0 at the rim, 1 in the middle, and FLAT at the rim.
    w = border_window(X, Y, 0.25)
    ok = ok and abs(w[0, :].max()) < 1e-12 and abs(w[:, 0].max()) < 1e-12
    ok = ok and abs(w[info['ny'] // 2, info['nx'] // 2] - 1.0) < 1e-12

    # The rim slope is analytically zero (d/dd of 0.5-0.5cos(pi d) is
    # 0.5 pi sin(pi d), which vanishes at d=0).  A one-sided difference at the
    # boundary cannot show that directly -- it reports its own O(dx w'')
    # truncation -- so test that the estimate CONVERGES to zero: halving the
    # pitch must roughly halve it.  A genuine slope discontinuity would not
    # shrink at all.
    slopes = []
    for res in (201, 401, 801):
        Xr, Yr, ir = make_grid(width=2.0, aspect=1.0, resolution=res)
        wr = border_window(Xr, Yr, 0.25)
        row = wr[ir['ny'] // 2, :]
        slopes.append(abs((row[1] - row[0]) / ir['dx']))
    ratio = slopes[0] / max(slopes[-1], 1e-30)
    print("grid: border rim slope %.3e -> %.3e -> %.3e (4x refine, ratio %.2f)"
          % (slopes[0], slopes[1], slopes[2], ratio))
    ok = ok and ratio > 3.0 and slopes[-1] < slopes[0]

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
