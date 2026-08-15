"""relief.imprint -- pressing a 3-D object into a shallow panel.

A raw depth map is not a relief.  Ray-cast an object, scale the result into
10 mm of panel, and you get a stepped blob: the object sits on a pedestal of
full depth, its silhouette is a cliff, and its interior modelling is crushed
into the few levels left over.  Making it read requires compressing the *range*
while preserving the *detail*, which is the same problem as tone-mapping a
high-dynamic-range photograph and is solved the same ways.

Three stages, in order:

1. **Silhouette handling.**  The object/background discontinuity must be
   removed before anything else, or every later stage spends its entire budget
   on that one cliff.
2. **Compression.**  Two methods.  Adaptive histogram equalisation works
   directly on the height field, needs no solver, and is the default.
   Gradient-domain compression attenuates large gradients and reintegrates by
   solving a Poisson equation -- better on organic shapes, and it reuses the
   spectral machinery already here.
3. **Normalisation.**  After compression, never before: one stray ray through
   a gap otherwise sets the whole range.

References:
  Tim Weyrich, Jia Deng, Connelly Barnes, Szymon Rusinkiewicz and Adam
    Finkelstein, "Digital Bas-Relief from 3D Scenes", ACM TOG 26(3)
    (SIGGRAPH 2007), art. 32 -- silhouette-gradient removal, gradient
    attenuation, Poisson reintegration.
  Raanan Fattal, Dani Lischinski and Michael Werman, "Gradient Domain High
    Dynamic Range Compression", ACM TOG 21(3) (SIGGRAPH 2002), 249-256 -- the
    attenuation function adapted above.
  Xianfang Sun, Paul L. Rosin, Ralph R. Martin and Frank C. Langbein,
    "Bas-Relief Generation Using Adaptive Histogram Equalization", IEEE TVCG
    15(4), 2009, 642-653 -- the solver-free method used as the default, with
    gradient weights to protect detailed regions.
  Jiri Kerber, Meili Wang, Jian Chang, Jian J. Zhang, Alexander Belyaev and
    Hans-Peter Seidel, "Computer Assisted Relief Generation -- A Survey",
    Computer Graphics Forum 31(8), 2012, 2363-2377.
  Peter N. Belhumeur, David J. Kriegman and Alan L. Yuille, "The Bas-Relief
    Ambiguity", International Journal of Computer Vision 35(1), 1999, 33-44 --
    why a shallow relief can read as deep at all: under unknown distant
    lighting a Lambertian surface is image-indistinguishable from its
    transform z -> lambda z + mu x + nu y.
"""

import math

import numpy as np

METHODS = ('AHE', 'GRADIENT', 'LINEAR')


def poisson_neumann(div, dx=1.0, dy=1.0):
    """Solve laplacian(u) = div with zero normal derivative at the panel edge.

    NumPy has no DCT, so the Neumann problem is handled by mirroring the
    right-hand side to twice the size in each axis -- the even extension whose
    Fourier series *is* the cosine series -- solving there with `rfft2`, and
    cropping.  The solution is fixed only up to a constant, so the mean is
    removed.
    """
    d = np.asarray(div, dtype=float)
    ny, nx = d.shape
    big = np.zeros((2 * ny, 2 * nx))
    big[:ny, :nx] = d
    big[ny:, :nx] = d[::-1, :]
    big[:ny, nx:] = d[:, ::-1]
    big[ny:, nx:] = d[::-1, ::-1]

    NY, NX = big.shape
    ky = np.arange(NY).reshape(-1, 1)
    kx = np.arange(NX).reshape(1, -1)
    # Eigenvalues of the 5-point discrete Laplacian on a periodic grid.
    lam = ((2.0 * np.cos(2.0 * math.pi * kx / NX) - 2.0) / (dx * dx)
           + (2.0 * np.cos(2.0 * math.pi * ky / NY) - 2.0) / (dy * dy))
    lam[0, 0] = 1.0
    u_hat = np.fft.fft2(big) / lam
    u_hat[0, 0] = 0.0
    u = np.real(np.fft.ifft2(u_hat))[:ny, :nx]
    return u - u.mean()


def _divergence(gx, gy, dx=1.0, dy=1.0):
    ddx = np.zeros_like(gx)
    ddy = np.zeros_like(gy)
    ddx[:, 1:-1] = (gx[:, 2:] - gx[:, :-2]) / (2.0 * dx)
    ddx[:, 0] = (gx[:, 1] - gx[:, 0]) / dx
    ddx[:, -1] = (gx[:, -1] - gx[:, -2]) / dx
    ddy[1:-1, :] = (gy[2:, :] - gy[:-2, :]) / (2.0 * dy)
    ddy[0, :] = (gy[1, :] - gy[0, :]) / dy
    ddy[-1, :] = (gy[-1, :] - gy[-2, :]) / dy
    return ddx + ddy


def kill_silhouette(h, mask, dx=1.0, dy=1.0, threshold=None):
    """Gradients of `h`, with the object/background cliff removed.

    Returns `(gx, gy)`.  Any gradient that straddles the silhouette -- or that
    exceeds `threshold` -- is zeroed, so the reintegrated surface no longer
    spends its range on a step that carries no shape information.
    """
    h = np.asarray(h, dtype=float)
    m = np.asarray(mask, dtype=bool)
    gy, gx = np.gradient(h, dy, dx)
    inside_x = np.ones_like(m)
    inside_y = np.ones_like(m)
    inside_x[:, 1:-1] = m[:, 2:] & m[:, :-2]
    inside_y[1:-1, :] = m[2:, :] & m[:-2, :]
    gx = np.where(m & inside_x, gx, 0.0)
    gy = np.where(m & inside_y, gy, 0.0)
    if threshold:
        mag = np.hypot(gx, gy)
        big = mag > float(threshold)
        gx = np.where(big, 0.0, gx)
        gy = np.where(big, 0.0, gy)
    return gx, gy


def compress_gradient(h, mask=None, dx=1.0, dy=1.0, alpha=0.1, beta=0.85,
                      threshold=None):
    """Gradient-domain compression: attenuate, then reintegrate.

    Fattal's attenuation `s = (alpha/m) (m/alpha)^beta` leaves small gradients
    almost untouched and pulls large ones down, so fine modelling survives
    while the overall range collapses.  `alpha` is expressed as a multiple of
    the mean gradient magnitude, which makes it scale-free.
    """
    h = np.asarray(h, dtype=float)
    m = (np.ones(h.shape, dtype=bool) if mask is None
         else np.asarray(mask, dtype=bool))
    gx, gy = kill_silhouette(h, m, dx, dy, threshold)
    mag = np.hypot(gx, gy)
    mean_mag = float(mag[m].mean()) if m.any() else 0.0
    a = max(float(alpha) * max(mean_mag, 1e-12), 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (a / np.maximum(mag, 1e-12)) * (np.maximum(mag, 1e-12)
                                            / a) ** float(beta)
    s = np.where(mag > 1e-12, s, 1.0)
    out = poisson_neumann(_divergence(gx * s, gy * s, dx, dy), dx, dy)
    return np.where(m, out, out[m].min() if m.any() else 0.0)


def compress_ahe(h, mask=None, bins=256, clip=4.0, gradient_weight=True,
                 dx=1.0, dy=1.0, percentile=0.2):
    """Adaptive histogram equalisation of the height field.

    A histogram of heights is built (optionally weighted by gradient
    magnitude, so detailed regions claim more of the output range), its counts
    are clipped and the excess redistributed, and the normalised cumulative
    sum becomes the height mapping.  No solver, no linear algebra.

    The histogram range is taken from percentiles rather than from the
    extremes.  A ray-cast depth map routinely contains a few stray samples --
    one ray through a gap in the mesh, or a distant background hit -- and
    binning over the full min/max would drop every real sample into the first
    bin, degenerating the mapping to a step.  Clipping the range instead costs
    nothing and removes the failure mode.
    """
    h = np.asarray(h, dtype=float)
    m = (np.ones(h.shape, dtype=bool) if mask is None
         else np.asarray(mask, dtype=bool))
    if not m.any():
        return np.zeros_like(h)
    vals = h[m]
    p = float(np.clip(percentile, 0.0, 20.0))
    lo = float(np.percentile(vals, p))
    hi = float(np.percentile(vals, 100.0 - p))
    if hi - lo < 1e-12:
        lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return np.zeros_like(h)
    vals = np.clip(vals, lo, hi)
    h = np.clip(h, lo, hi)

    if gradient_weight:
        gy, gx = np.gradient(h, dy, dx)
        w = np.hypot(gx, gy)[m]
        w = w / (w.mean() + 1e-12) + 0.05      # never fully ignore flat areas
    else:
        w = np.ones_like(vals)

    hist, edges = np.histogram(vals, bins=int(bins), range=(lo, hi),
                               weights=w)
    # Contrast limiting: clip tall bins, spread the excess evenly.
    if clip and clip > 0:
        limit = float(clip) * hist.mean()
        excess = float(np.clip(hist - limit, 0.0, None).sum())
        hist = np.minimum(hist, limit) + excess / len(hist)
    cdf = np.cumsum(hist)
    cdf = cdf / max(cdf[-1], 1e-12)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = np.interp(h, centres, cdf)
    return np.where(m, 2.0 * out - 1.0, -1.0)


def imprint(depth, mask=None, method='AHE', dx=1.0, dy=1.0, **kw):
    """Compress a raw depth map into something that reads as relief."""
    if method == 'LINEAR':
        h = np.asarray(depth, dtype=float)
        m = (np.ones(h.shape, dtype=bool) if mask is None
             else np.asarray(mask, dtype=bool))
        return np.where(m, h, h[m].min() if m.any() else 0.0)
    if method == 'AHE':
        return compress_ahe(depth, mask, dx=dx, dy=dy,
                            bins=int(kw.get('bins', 256)),
                            clip=float(kw.get('clip', 4.0)),
                            gradient_weight=bool(kw.get('gradient_weight',
                                                        True)))
    if method == 'GRADIENT':
        return compress_gradient(depth, mask, dx=dx, dy=dy,
                                 alpha=float(kw.get('alpha', 0.1)),
                                 beta=float(kw.get('beta', 0.85)),
                                 threshold=kw.get('threshold'))
    raise ValueError("unknown imprint method: %r" % (method,))


def gbr_shear(h, X, Y, mu=0.0, nu=0.0, lam=1.0):
    """The generalized bas-relief transform z -> lambda z + mu x + nu y.

    Under unknown distant lighting a Lambertian surface and its GBR transform
    produce identical images, so this whole family is free -- lambda is the
    honest meaning of a depth control, and (mu, nu) tilts the panel without
    changing how it reads.
    """
    return float(lam) * np.asarray(h, dtype=float) + float(mu) * X \
        + float(nu) * Y


def _selftest():
    ok = True
    from . import grid as _grid

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=97)
    dx, dy = info['dx'], info['dy']

    # --- the Poisson solver recovers a field from its own Laplacian ---
    # The test field must SATISFY the boundary condition the solver imposes.
    # A Neumann solve cannot reproduce a field with non-zero normal
    # derivative at the rim, so use a cosine mode of the panel, which has
    # du/dn = 0 on all four edges by construction.
    # Test the solver against its OWN operator rather than against the
    # continuum: apply it to a right-hand side, then take the discrete
    # Laplacian of the result and check it comes back.  Comparing a discrete
    # solve to an analytic Laplacian measures the discretisation, not the
    # solver.
    Lx = info['width'] + dx
    Ly = info['height'] + dy
    kx = 2.0 * math.pi / Lx
    ky = 3.0 * math.pi / Ly
    div = np.cos(kx * (X - X.min())) * np.cos(ky * (Y - Y.min()))
    div = div - div.mean()                  # solvability: Neumann needs this
    u = poisson_neumann(div, dx, dy)
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx ** 2
        + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy ** 2)
    s = (slice(3, -3), slice(3, -3))
    rel = float(np.abs(lap[s] - div[s]).max() / np.abs(div[s]).max())
    print("imprint: Neumann Poisson operator round-trip rel err = %.3e" % rel)
    ok = ok and rel < 1e-6

    # --- a synthetic "object": a dome on a flat ground plane ----------
    r = np.hypot(X, Y)
    R = 0.55
    dome = np.where(r < R, np.sqrt(np.maximum(R * R - r * r, 0.0)), 0.0)
    # Interior modelling: fine ripples on the dome, small next to its height.
    dome = dome + 0.02 * np.sin(30.0 * X) * np.sin(28.0 * Y) * (r < R)
    obj = r < R

    def legibility(h):
        """RMS of the fine detail, relative to the field's total range.

        A raw depth map spends its range on the silhouette cliff, so the
        modelling that survives is a tiny fraction of it; compression is
        supposed to raise exactly this number.
        """
        hh = np.asarray(h, float)
        rng = float(hh[obj].max() - hh[obj].min())
        if rng < 1e-12:
            return 0.0
        gy, gx = np.gradient(hh, dy, dx)
        inner = obj & (r < R * 0.8)
        return float(np.hypot(gx, gy)[inner].std() / rng)

    raw = legibility(dome)
    ahe = legibility(imprint(dome, obj, 'AHE', dx=dx, dy=dy))
    grd = legibility(imprint(dome, obj, 'GRADIENT', dx=dx, dy=dy))
    print("imprint: detail-to-range  raw=%.3f  AHE=%.3f  GRADIENT=%.3f"
          % (raw, ahe, grd))
    ok = ok and ahe > raw and grd > raw

    # Both compressors keep the output finite and bounded.
    for method in METHODS:
        out = imprint(dome, obj, method, dx=dx, dy=dy)
        ok = ok and np.isfinite(out).all()
        ok = ok and out.shape == dome.shape

    # Silhouette killing really does remove the cliff.
    gx0, gy0 = np.gradient(dome, dy, dx)
    gx1, gy1 = kill_silhouette(dome, obj, dx, dy)
    edge = obj & ~np.roll(obj, 1, axis=1)
    print("imprint: max |grad| on the silhouette  before=%.3f  after=%.3f"
          % (float(np.hypot(gx0, gy0)[edge].max()),
             float(np.hypot(gx1, gy1)[edge].max())))
    ok = ok and (float(np.hypot(gx1, gy1)[edge].max())
                 < float(np.hypot(gx0, gy0)[edge].max()))

    # AHE is robust to a stray outlier ray; a linear map is not.  Compare on
    # equal terms -- both normalised to [-1,1] -- and measure how much of the
    # output range the BULK of the data actually occupies.  A single wild
    # sample should not be able to crush everything else into a sliver.
    # The spike must land INSIDE the object, or masking removes it before
    # either method sees it and the comparison measures nothing.
    spiked = dome.copy()
    spiked[dome.shape[0] // 2, dome.shape[1] // 2] = 50.0

    def spread(a):
        """Interquartile share of the range, measured ON THE OBJECT.

        Measuring over the whole panel would be dominated by the flat
        background, which is three quarters of the samples here and tells us
        nothing about how well the object's modelling survived.
        """
        v = np.asarray(a)[obj]
        q1, q3 = np.percentile(v, [25.0, 75.0])
        rng = float(np.asarray(a).max() - np.asarray(a).min())
        return float((q3 - q1) / rng) if rng > 1e-12 else 0.0

    lin = imprint(spiked, obj, 'LINEAR', dx=dx, dy=dy)
    a_ = imprint(spiked, obj, 'AHE', dx=dx, dy=dy)
    print("imprint: with a 50x outlier, interquartile share of the range on "
          "the object: LINEAR=%.4f  AHE=%.4f" % (spread(lin), spread(a_)))
    ok = ok and spread(a_) > 10.0 * spread(lin)

    # The GBR shear does not change the field's shape, only its tilt.
    g = gbr_shear(dome, X, Y, mu=0.1, nu=-0.05, lam=1.0)
    ok = ok and np.isfinite(g).all()
    resid = g - dome - (0.1 * X - 0.05 * Y)
    ok = ok and float(np.abs(resid).max()) < 1e-12

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
