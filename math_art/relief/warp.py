"""relief.warp -- orientation (guide) fields, domain warping, phase solving.

Quasi-parallel ridges that fan, split and sweep across a panel are an
*orientation* phenomenon, not an amplitude one.  A wave with a single global
direction cannot produce them however its amplitude is modulated; what is
needed is a direction field, and a phase that follows it.

Three tools live here:

* **Orientation fields** -- a per-sample angle theta(x, y) that later layers
  consume as their local wave direction.
* **Domain warping** -- displacing the coordinates a field is evaluated at,
  p' = p + W g(p).  Periodicity composes through this: if the field and the
  warp are both L-periodic then so is the composition, which is what lets the
  seamless mode (a later phase) survive the warp stage.
* **Phase solving** -- given a direction field d and a target spacing, find the
  scalar phase phi whose gradient best matches omega*d, so that sin(phi) is a
  banded field flowing along d.  omega*d is generally not a gradient, so this
  is a least-squares problem, solved spectrally.

References:
  Ken Perlin, "An Image Synthesizer", Computer Graphics 19(3) (SIGGRAPH 1985),
    287-296 -- gradient noise, and the idea of perturbing a function's phase
    with noise (his marble) that the warp stage generalises.
  Ken Perlin and Fabrice Neyret, "Flow Noise", SIGGRAPH 2001 Technical
    Sketches -- per-octave rotation and advection of fine octaves by coarse.
  Robert T. Frankot and Rama Chellappa, "A Method for Enforcing Integrability
    in Shape from Shading Algorithms", IEEE Trans. PAMI 10(4), 1988, 439-451 --
    the spectral least-squares integration of a non-integrable gradient field
    used by `phase_from_direction`.
  Felix Knoeppel, Keenan Crane, Ulrich Pinkall and Peter Schroeder, "Stripe
    Patterns on Surfaces", ACM TOG 34(4) (SIGGRAPH 2015), art. 39 -- the
    complex-phase formulation whose branch points are the bifurcations this
    module approximates; the Poisson route here is its smooth relative.

The two-level warp idiom (a field evaluated at coordinates displaced by a
second field, itself displaced by a third) is a practitioner formulation
popularised by Inigo Quilez rather than a peer-reviewed construction, and is
cited as such.
"""

import math

import numpy as np

ORIENTATIONS = ('CONSTANT', 'RADIAL', 'TANGENT', 'SPIRAL', 'CURL', 'GRADIENT')


def smooth_field(X, Y, seed=1, freq=1.0, octaves=4, lacunarity=2.0,
                 gain=0.5):
    """A band-limited, deterministic scalar field, zero mean and unit sd.

    Built as a sum of cosine modes with random directions and phases -- the
    spectral-synthesis construction, which is band-limited by construction so
    it cannot alias below the octave it stops at.
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    out = np.zeros(X.shape)
    amp = 1.0
    f = float(freq)
    for _ in range(max(1, int(octaves))):
        # A handful of directions per octave keeps it close to isotropic.
        for _k in range(3):
            a = rng.uniform(0.0, 2.0 * math.pi)
            ph = rng.uniform(0.0, 2.0 * math.pi)
            kx = 2.0 * math.pi * f * math.cos(a)
            ky = 2.0 * math.pi * f * math.sin(a)
            out += amp * np.cos(kx * X + ky * Y + ph)
        amp *= gain
        f *= lacunarity
    out -= out.mean()
    sd = out.std()
    return out / sd if sd > 1e-12 else out


def domain_warp(X, Y, strength=0.0, seed=1, iterations=2, freq=0.6):
    """Displace the sample coordinates by a smooth vector field.

    `strength` is in domain units, so it keeps its meaning as resolution
    changes.  Two iterations is where the fanning, bifurcating character
    appears; one gives a gentle bend.
    """
    if strength <= 0.0:
        return X, Y
    Xw, Yw = X, Y
    for it in range(max(1, int(iterations))):
        # Decorrelated components: distinct seeds, not offsets of one field.
        gx = smooth_field(Xw, Yw, seed=seed * 7919 + 13 * it, freq=freq)
        gy = smooth_field(Xw, Yw, seed=seed * 6271 + 29 * it, freq=freq)
        s = float(strength) / (it + 1.0)      # coarse first, then finer
        Xw = Xw + s * gx
        Yw = Yw + s * gy
    return Xw, Yw


def orientation_field(X, Y, kind='CONSTANT', angle=0.0, seed=1, freq=0.5,
                      swirl=1.0):
    """Return a per-sample direction angle theta(x, y), in radians."""
    if kind == 'CONSTANT':
        return np.full(X.shape, float(angle))
    if kind == 'RADIAL':
        return np.arctan2(Y, X)
    if kind == 'TANGENT':
        return np.arctan2(Y, X) + 0.5 * math.pi
    if kind == 'SPIRAL':
        r = np.hypot(X, Y)
        return np.arctan2(Y, X) + float(swirl) * r + float(angle)
    if kind == 'CURL':
        # Direction of the divergence-free field grad-perp of a smooth scalar.
        psi = smooth_field(X, Y, seed=seed, freq=freq)
        gy, gx = np.gradient(psi, Y[:, 0], X[0, :])
        return np.arctan2(gx, -gy) + float(angle)
    if kind == 'GRADIENT':
        phi = smooth_field(X, Y, seed=seed, freq=freq)
        gy, gx = np.gradient(phi, Y[:, 0], X[0, :])
        return np.arctan2(gy, gx) + float(angle)
    raise ValueError("unknown orientation field: %r" % (kind,))


def phase_from_direction(theta, wavelength, X, Y):
    """Least-squares phase whose gradient best matches omega * d.

    With d = (cos theta, sin theta) and omega = 2 pi / wavelength, the target
    gradient g = omega d is generally not integrable, so solve

        laplacian(phi) = div(g)

    spectrally: phi_hat(k) = -i k . g_hat(k) / |k|^2.

    **The mean gradient must be split off first.**  The periodic Laplacian
    annihilates a linear ramp -- a constant g has zero divergence everywhere,
    so a naive spectral solve returns phi = 0 and loses the carrier entirely
    (the bands vanish).  So decompose g into its mean and a fluctuation:

        phi = (mean g) . p   +   phi_periodic

    integrating the mean part exactly as a ramp and leaving only the
    zero-mean remainder, which the periodic solve represents faithfully.

    Where the direction field is singular the resulting bands fan and
    bifurcate, which is the effect wanted.  The transform is periodic, so the
    fluctuating part wraps; a border window hides that at the rim.

    *Frankot and Chellappa (1988).*
    """
    ny, nx = theta.shape
    dx = float(X[0, 1] - X[0, 0]) if nx > 1 else 1.0
    dy = float(Y[1, 0] - Y[0, 0]) if ny > 1 else 1.0
    omega = 2.0 * math.pi / max(float(wavelength), 1e-9)
    gx = omega * np.cos(theta)
    gy = omega * np.sin(theta)

    # Exactly integrable mean part -> a linear ramp.
    gxm = float(gx.mean())
    gym = float(gy.mean())
    ramp = gxm * X + gym * Y

    # Zero-mean remainder -> periodic Poisson solve.
    gxf = gx - gxm
    gyf = gy - gym
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX * KX + KY * KY
    k2[0, 0] = 1.0                      # avoid 0/0; DC handled below

    num = -1j * (KX * np.fft.fft2(gxf) + KY * np.fft.fft2(gyf))
    phi_hat = num / k2
    phi_hat[0, 0] = 0.0
    return ramp + np.real(np.fft.ifft2(phi_hat))


def _selftest():
    ok = True
    from . import grid as _grid

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=129)

    # smooth_field is deterministic, zero-mean, unit-sd.
    a = smooth_field(X, Y, seed=4)
    b = smooth_field(X, Y, seed=4)
    c = smooth_field(X, Y, seed=5)
    print("warp: smooth_field mean=%.2e sd=%.6f  repeatable=%s  seed-varies=%s"
          % (a.mean(), a.std(), np.array_equal(a, b), not np.allclose(a, c)))
    ok = ok and abs(a.mean()) < 1e-12 and abs(a.std() - 1.0) < 1e-9
    ok = ok and np.array_equal(a, b) and not np.allclose(a, c)

    # Zero warp strength is exactly the identity.
    Xw, Yw = domain_warp(X, Y, strength=0.0)
    ok = ok and Xw is X and Yw is Y
    # Non-zero warp displaces by roughly the requested magnitude.
    Xw, Yw = domain_warp(X, Y, strength=0.1, seed=3, iterations=1)
    disp = np.hypot(Xw - X, Yw - Y)
    print("warp: |displacement| mean=%.4f max=%.4f (strength 0.1)"
          % (disp.mean(), disp.max()))
    ok = ok and 0.01 < disp.mean() < 0.5

    # Orientation fields all produce finite angles of the right shape.
    for kind in ORIENTATIONS:
        th = orientation_field(X, Y, kind=kind, seed=2)
        ok = ok and th.shape == X.shape and np.isfinite(th).all()

    # The phase solve recovers an exactly integrable target: for a CONSTANT
    # direction, omega*d IS a gradient, so grad(phi) must match it closely.
    lam = 0.5
    th = orientation_field(X, Y, kind='CONSTANT', angle=0.7)
    phi = phase_from_direction(th, lam, X, Y)
    gy, gx = np.gradient(phi, Y[:, 0], X[0, :])
    omega = 2.0 * math.pi / lam
    # Compare on the interior, away from the periodic wrap at the rim.
    s = (slice(8, -8), slice(8, -8))
    err = max(np.abs(gx[s] - omega * math.cos(0.7)).max(),
              np.abs(gy[s] - omega * math.sin(0.7)).max()) / omega
    print("warp: constant-direction phase gradient rel err = %.3e" % err)
    ok = ok and err < 0.02

    # ...and the resulting band field really is a wave of that wavelength.
    band = np.sin(phi)
    zero_crossings = np.sum(np.diff(np.signbit(band[info['ny'] // 2, :])))
    print("warp: constant-direction band has %d sign changes across 2 m "
          "(expect ~%d at lambda=%.2f)"
          % (abs(zero_crossings), int(2.0 * abs(math.cos(0.7)) / (lam / 2.0)),
             lam))
    ok = ok and abs(zero_crossings) >= 4

    # A varying direction field still yields a finite, banded field.
    th = orientation_field(X, Y, kind='CURL', seed=11, freq=0.7)
    phi = phase_from_direction(th, 0.4, X, Y)
    band = np.sin(phi)
    print("warp: curl-guided band field range [%.3f, %.3f]"
          % (band.min(), band.max()))
    ok = ok and np.isfinite(phi).all() and band.min() < -0.5 < 0.5 < band.max()

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
