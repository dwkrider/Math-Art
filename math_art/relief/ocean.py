"""relief.ocean -- a wind-driven sea surface by spectral synthesis.

Build the surface in the frequency domain instead of the spatial one: give
every wavevector an amplitude drawn from an oceanographic spectrum and a random
phase, then take one inverse FFT.  The whole sea appears at once, at the cost
of a single transform, and every statistical property of it is inherited from
the spectrum rather than tuned by hand.

The Phillips spectrum supplies the amplitudes::

    P(k) = A exp(-1 / (k L)^2) |k_hat . w_hat|^2 / k^4,     L = V^2 / g

The k^-4 tail is what makes a sea look like a sea -- large swells carrying
progressively finer chop, with a fixed relationship between the two.  The
directional factor |k_hat . w_hat|^2 suppresses waves running across the wind,
which is what gives a fetch its grain.  The exponential kills waves too long
for the wind speed to have raised.

**It tiles exactly, and for the same reason the discrete Fourier transform
exists at all**: a field synthesised from a finite set of integer-frequency
waves on a grid is periodic in that grid, identically.  No snapping and no
residual -- the seam is a consequence of the method rather than a repair to it.

**Choppiness.**  A real wave crest is sharp and its trough broad, which no
height field can do on its own: Tessendorf gets it by displacing the surface
*horizontally* toward the crests, and the result is a parametric surface that
can fold over itself.  A relief panel must stay single-valued, so the same
displacement field is applied here as an inverse warp -- the height is read
from where the water came from rather than moved to where it goes.  Crests
sharpen, troughs broaden, and nothing folds.  The Jacobian is reported, since
where it would have gone negative is exactly where the true surface would have
broken.

References:
  Jerry Tessendorf, "Simulating Ocean Water", SIGGRAPH course notes,
    1999-2004 -- the FFT synthesis, the Phillips spectrum as used here, and
    the choppy-wave displacement.
  O. M. Phillips, "The Dynamics of the Upper Ocean", 2nd ed., Cambridge
    University Press, 1977 -- the equilibrium range and the k^-4 law.
  Gary A. Mastin, Peter A. Watterberg and John F. Mareda, "Fourier Synthesis
    of Ocean Scenes", IEEE Computer Graphics and Applications 7(3), 1987,
    16-23 -- spectral synthesis of a sea surface.
"""

import math

import numpy as np

G = 9.81


def phillips(n, patch=100.0, wind=8.0, direction=0.0, amplitude=1.0,
             cutoff=0.5, align=2.0):
    """Phillips amplitudes on an n x n wavevector grid.

    `patch` is the physical size of the tile in metres, which together with
    the wind speed decides how many waves cross it -- the spectrum is written
    in real units because that is what makes the k^-4 law mean anything.
    """
    n = int(n)
    k1 = 2.0 * math.pi * np.fft.fftfreq(n, d=patch / n)
    KX, KY = np.meshgrid(k1, k1)
    k2 = KX * KX + KY * KY
    k = np.sqrt(k2)

    L = float(wind) ** 2 / G                 # largest wave this wind raises
    wx, wy = math.cos(direction), math.sin(direction)
    with np.errstate(divide='ignore', invalid='ignore'):
        khx = np.where(k > 0, KX / np.maximum(k, 1e-12), 0.0)
        khy = np.where(k > 0, KY / np.maximum(k, 1e-12), 0.0)
        cosw = khx * wx + khy * wy
        p = (float(amplitude) * np.exp(-1.0 / np.maximum(k2 * L * L, 1e-30))
             / np.maximum(k2 * k2, 1e-30)
             * np.abs(cosw) ** float(align))
        # Waves shorter than `cutoff` metres are below what the surface can
        # carry; without this the k^-4 tail keeps adding unresolvable ripple.
        p *= np.exp(-k2 * float(cutoff) ** 2)
    p[k == 0] = 0.0                          # no zero-frequency swell
    return np.where(np.isfinite(p), p, 0.0), KX, KY, k


def spectrum_field(n=256, patch=100.0, wind=8.0, direction=0.0, seed=1,
                   amplitude=1.0, cutoff=0.5, align=2.0):
    """One realisation of the sea surface, and its choppy displacement.

    Returns `(h, dx, dy)`, all real n x n arrays on a torus.
    """
    p, KX, KY, k = phillips(n, patch, wind, direction, amplitude, cutoff,
                            align)
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    xr = rng.normal(size=(n, n))
    xi = rng.normal(size=(n, n))
    h0 = (xr + 1j * xi) * np.sqrt(np.maximum(p, 0.0) / 2.0)

    # A real height field needs a Hermitian spectrum.  Tessendorf builds it as
    # h0(k) + conj(h0(-k)), which is Hermitian for ANY h0 -- so the surface is
    # real by construction rather than by discarding an imaginary part that
    # should not have been there.
    flip = np.conj(h0[(-np.arange(n)) % n][:, (-np.arange(n)) % n])
    ht = h0 + flip

    h = np.real(np.fft.ifft2(ht))
    # Horizontal displacement toward the crests.
    #
    # **The sign is +i k_hat here, not the -i the notes are usually written
    # with.**  That is a convention difference, not a disagreement: the sign
    # of i in the displacement is tied to the sign of i in the inverse
    # transform, and numpy's `ifft` uses exp(+ikx).  Checked on a single
    # cosine, where the answer is known -- a Gerstner wave moves points
    # *toward* the crest, and -i moves them away, broadening the crest and
    # flattening exactly the feature choppiness exists to sharpen.
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(k > 0, 1.0 / np.maximum(k, 1e-12), 0.0)
    dx = np.real(np.fft.ifft2(1j * KX * inv * ht))
    dy = np.real(np.fft.ifft2(1j * KY * inv * ht))

    # **Calibrate the amplitude to a real sea state.**  The constant A in the
    # Phillips spectrum is free, and leaving it at 1 gives a 100 m patch of
    # 4 mm waves -- harmless for a height field, which gets normalised anyway,
    # but fatal for choppiness: the displacement is then a hundredth of a cell
    # and the crests never sharpen.  Pierson and Moskowitz's fully developed
    # sea fixes it, with significant wave height H_s = 0.21 V^2 / g and
    # H_s = 4 rms(h).  Scaling h and D together preserves the ratio between
    # them, which is the only quantity choppiness depends on -- and it makes
    # wind speed do something visible, since a stronger wind now raises taller
    # and correspondingly choppier waves.
    #
    #   Willard J. Pierson and Lionel Moskowitz, "A proposed spectral form for
    #   fully developed wind seas...", J. Geophys. Res. 69(24), 1964,
    #   5181-5190.
    rms = float(h.std())
    if rms > 1e-30:
        target = 0.21 * float(wind) ** 2 / G / 4.0
        scale = target / rms
        h = h * scale
        dx = dx * scale
        dy = dy * scale
    return h, dx, dy


def _sample_wrapped(a, gx, gy):
    """Bilinear sample of a periodic array at fractional indices."""
    n = a.shape[0]
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = gx - x0
    fy = gy - y0
    x0 %= n
    y0 %= n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x1] * fx * (1 - fy)
            + a[y1, x0] * (1 - fx) * fy + a[y1, x1] * fx * fy)


def cusp_lambda(dx, dy, cell):
    """The displacement scale at which the steepest crest first cusps.

    A sea has no single wavelength, so "the steepness of the dominant wave"
    is not well defined -- weighting the spectrum one way puts the peak at
    8 m and another way at the patch size, and the choppiness that results
    differs by a factor of forty.  The folding limit is unambiguous instead:
    the map x -> x + lam D stops being injective where its Jacobian

        J(lam) = 1 + lam (a + b) + lam^2 (a b - c),
        a = dDx/dx,  b = dDy/dy,  c = (dDx/dy)(dDy/dx)

    first reaches zero.  That is precisely the Stokes cusp -- a Gerstner wave
    cusps exactly when it starts to fold -- so it is both the physically
    meaningful scale and a computable one.  Per cell this is a quadratic;
    take the smallest positive root anywhere in the field.
    """
    a = np.gradient(dx, cell, axis=1)
    b = np.gradient(dy, cell, axis=0)
    c = np.gradient(dx, cell, axis=0) * np.gradient(dy, cell, axis=1)
    q = a * b - c
    lin = a + b
    best = np.inf
    # Linear case, |q| ~ 0: root at -1/lin where lin < 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        lin_root = np.where(lin < 0, -1.0 / lin, np.inf)
        small = np.abs(q) < 1e-12
        if small.any():
            best = min(best, float(np.min(lin_root[small])))
        disc = lin * lin - 4.0 * q
        ok = (~small) & (disc >= 0.0)
        if ok.any():
            r = np.sqrt(np.maximum(disc, 0.0))
            r1 = (-lin - r) / (2.0 * q)
            r2 = (-lin + r) / (2.0 * q)
            pos = np.where(r1 > 1e-12, r1, np.inf)
            pos = np.minimum(pos, np.where(r2 > 1e-12, r2, np.inf))
            best = min(best, float(np.min(pos[ok])))
    return best if np.isfinite(best) and best > 0 else 1.0


def _forward_choppy(h, dx, dy, shift, super_=2):
    """Displace the surface and keep the part you would actually see.

    Choppiness is a *forward* map: the water at x ends up at x + lam D(x).
    Inverting that to first order -- reading the height from x - lam D(x) --
    is only right while the displacement is small, and it fails exactly where
    the effect matters: measured against a known sea, it left the crests
    almost as round as it found them until lam was large enough that the true
    surface had already overturned.

    So the map is applied forwards, and where it folds -- where two parcels of
    water land on the same place -- the HIGHEST is kept, because that is the
    one an observer above the panel sees.  Crests sharpen properly, a breaking
    wave resolves into an overhang read from above, and the field stays
    single-valued, which a relief panel requires.

    The source is supersampled first so that stretched regions, where the map
    spreads parcels apart, still receive a value rather than a hole.
    """
    n = h.shape[0]
    k = max(1, int(super_))
    # Source positions on a k-times finer lattice, in coarse-grid units.
    g = (np.arange(n * k) + 0.5) / k - 0.5
    SX, SY = np.meshgrid(g, g)
    hs = _sample_wrapped(h, SX, SY)
    ds_x = _sample_wrapped(dx, SX, SY)
    ds_y = _sample_wrapped(dy, SX, SY)

    tx = np.mod(np.round(SX + ds_x * shift).astype(np.int64), n)
    ty = np.mod(np.round(SY + ds_y * shift).astype(np.int64), n)

    out = np.full((n, n), -np.inf)
    np.maximum.at(out, (ty.ravel(), tx.ravel()), hs.ravel())
    # Anywhere the map reached nothing at all, fall back to the undisplaced
    # surface rather than leaving a pit.
    empty = ~np.isfinite(out)
    if empty.any():
        out[empty] = h[empty]
    return out


def ocean(X, Y, info, sim=256, patch=100.0, wind=8.0, direction=0.0, seed=1,
          choppy=0.0, cutoff=0.5, align=2.0, report=None):
    """Sample a synthesised sea onto the panel.

    `choppy` applies Tessendorf's displacement as an inverse warp, sharpening
    crests without letting the surface fold.  If `report` is a dict it
    receives the minimum Jacobian -- below zero is where a true displaced
    surface would have overturned.
    """
    n = int(max(32, sim))
    h, dxf, dyf = spectrum_field(n=n, patch=patch, wind=wind,
                                 direction=direction, seed=seed,
                                 cutoff=cutoff, align=align)
    ny, nx = X.shape
    gx = np.arange(nx) * (n / float(max(nx - 1, 1)))
    gy = np.arange(ny) * (n / float(max(ny - 1, 1)))
    GX, GY = np.meshgrid(gx, gy)

    lam = float(choppy)
    if lam > 0.0:
        cell = patch / n
        # **`choppy` is measured in cusp limits, not in Tessendorf's raw
        # lambda.**  A fully developed Pierson-Moskowitz sea is gentle -- its
        # energy sits at tens of metres carrying about 0.34 m rms -- so the
        # true lambda = 1 displaces a crest by a few percent of its own
        # wavelength and sharpens nothing visible.  The choppy seas in
        # graphics come from lambda well above 1, which Tessendorf treats as a
        # free artistic parameter; but a free parameter whose useful range
        # shifts with patch size, wind and grid is not a control anyone can
        # use.  Scaling by the folding limit fixes the meaning: 0 is a smooth
        # swell, 1 puts the steepest crest exactly at the cusp, and above 1
        # the sharpest crests break.
        lam = lam * cusp_lambda(dxf, dyf, cell)
        if report is not None:
            # Jacobian of x + lam D, the quantity Tessendorf watches: where
            # this goes negative the real surface has folded over.
            ddx = np.gradient(dxf, cell, axis=1)
            ddy = np.gradient(dyf, cell, axis=0)
            cross = np.gradient(dxf, cell, axis=0) * np.gradient(dyf, cell,
                                                                 axis=1)
            jac = (1.0 + lam * ddx) * (1.0 + lam * ddy) - lam * lam * cross
            report['jacobian_min'] = float(jac.min())
            report['folds'] = bool(jac.min() < 0.0)
        h = _forward_choppy(h, dxf, dyf, lam / cell)
    out = _sample_wrapped(h, GX, GY)
    s = out.std()
    return (out - out.mean()) / s if s > 1e-12 else out - out.mean()


def _selftest():
    ok = True
    from . import grid as _grid
    from . import tiling as _tiling

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=192)

    h = ocean(X, Y, info, sim=256, seed=3)
    print("ocean: range [%.2f, %.2f] finite=%s sd=%.4f"
          % (h.min(), h.max(), np.isfinite(h).all(), h.std()))
    ok = ok and np.isfinite(h).all() and h.std() > 1e-6

    # **The spectrum is the claim, so it is what gets measured.**  Phillips'
    # equilibrium range goes as k^-4 in the power spectrum of the surface;
    # fitting the radial average of the realisation's own spectrum must
    # recover that exponent, well away from the ends where the wind cutoff and
    # the short-wave damping bend it.
    n = 256
    hs, _, _ = spectrum_field(n=n, patch=100.0, wind=10.0, seed=5,
                              align=0.0, cutoff=0.05)
    P = np.abs(np.fft.fft2(hs)) ** 2
    kf = 2.0 * math.pi * np.fft.fftfreq(n, d=100.0 / n)
    KX, KY = np.meshgrid(kf, kf)
    kmag = np.hypot(KX, KY).ravel()
    pv = P.ravel()
    lo, hi = 0.8, 6.0                      # inside the equilibrium range
    sel = (kmag > lo) & (kmag < hi) & (pv > 0)
    bins = np.logspace(math.log10(lo), math.log10(hi), 14)
    idx = np.digitize(kmag[sel], bins)
    xs, ys = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 30:
            xs.append(math.log(0.5 * (bins[b - 1] + bins[b])))
            ys.append(math.log(pv[sel][m].mean()))
    slope = float(np.polyfit(xs, ys, 1)[0])
    print("ocean: radial power spectrum slope %.2f (Phillips k^-4)" % slope)
    ok = ok and -4.6 < slope < -3.4

    # Synthesised on a grid from integer frequencies, so it is periodic
    # identically -- no snapping was applied and none is needed.
    chk = _tiling.check(h, 'TORUS', info, 'OCEAN')
    print("ocean: torus joint step x%.3f curvature x%.3f ok=%s"
          % (chk['step'], chk['curvature'], chk['ok']))
    ok = ok and chk['ok']

    # Wind direction gives the sea its grain: with the directional factor on,
    # gradients across the wind must exceed those along it.
    calm = ocean(X, Y, info, sim=192, seed=7, direction=0.0, align=6.0)
    gy, gx = np.gradient(calm)
    print("ocean: gradient across wind / along wind = %.2f (wind along x)"
          % float(gx.std() / gy.std()))
    ok = ok and gx.std() > gy.std()

    # Choppiness must narrow the crests.  Measuring that took three attempts,
    # and the two that failed are worth recording:
    #
    #   * **Skew** barely moves.  The cusp limit is reached at the single
    #     steepest crest in the field while the rest of a broad-spectrum sea
    #     stays gentle, so the global histogram hardly notices -- and a
    #     measure that hardly notices cannot tell a correct implementation
    #     from a sign error, which is exactly what it failed to do here.
    #   * **Area above a fixed threshold** is undone by the normalisation:
    #     narrowing the crests lowers the field's own standard deviation, so
    #     rescaling to unit sd puts the area back almost exactly.
    #
    # What does discriminate is scale-invariant crest geometry: the area
    # within a fraction of the maximum, and the height of the maximum against
    # the bulk.
    rep = {}
    flat = ocean(X, Y, info, sim=192, seed=11, choppy=0.0)
    chop = ocean(X, Y, info, sim=192, seed=11, choppy=2.0, report=rep)

    def crest(a):
        z = (a - a.mean()) / a.std()
        return float((z > 0.6 * z.max()).mean()), float(z.max())

    a0, m0 = crest(flat)
    a1, m1 = crest(chop)
    print("ocean: crest area %.4f -> %.4f (%.1fx narrower), peak/rms "
          "%.2f -> %.2f, jacobian min %.3f (folds=%s)"
          % (a0, a1, a0 / max(a1, 1e-9), m0, m1,
             rep.get('jacobian_min', 0.0), rep.get('folds')))
    ok = ok and a1 < 0.75 * a0 and m1 > m0

    # choppy is measured in cusp limits, so at exactly 1 the steepest crest
    # must sit exactly at the folding threshold.  This is the calibration
    # itself, and it is what makes the control mean the same thing at any
    # patch size, wind and grid.
    rep1 = {}
    ocean(X, Y, info, sim=192, seed=11, choppy=1.0, report=rep1)
    print("ocean: at choppy=1 the jacobian minimum is %+.4f (0 by definition)"
          % rep1['jacobian_min'])
    ok = ok and abs(rep1['jacobian_min']) < 0.02

    # The displacement must not break the seam: it is periodic, so the
    # displaced surface is too.
    chk2 = _tiling.check(chop, 'TORUS', info, 'OCEAN')
    print("ocean: choppy torus joint step x%.3f ok=%s"
          % (chk2['step'], chk2['ok']))
    ok = ok and chk2['ok']

    # Wind speed sets the longest wave: a stronger wind must put more energy
    # into the low frequencies.
    def low_energy(v):
        hs2, _, _ = spectrum_field(n=128, patch=100.0, wind=v, seed=2)
        P2 = np.abs(np.fft.fft2(hs2)) ** 2
        kf2 = 2.0 * math.pi * np.fft.fftfreq(128, d=100.0 / 128)
        K2 = np.hypot(*np.meshgrid(kf2, kf2))
        return float(P2[(K2 > 0) & (K2 < 0.5)].sum() / P2[K2 > 0].sum())
    lo_w, hi_w = low_energy(5.0), low_energy(14.0)
    print("ocean: low-frequency share, wind 5 m/s %.3f -> 14 m/s %.3f"
          % (lo_w, hi_w))
    ok = ok and hi_w > lo_w

    # Deterministic, and the seed matters.
    a = ocean(X, Y, info, sim=128, seed=4)
    b = ocean(X, Y, info, sim=128, seed=4)
    c = ocean(X, Y, info, sim=128, seed=5)
    ok = ok and np.array_equal(a, b) and not np.allclose(a, c)
    print("ocean: repeatable=%s seed-varies=%s"
          % (np.array_equal(a, b), not np.allclose(a, c)))

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
