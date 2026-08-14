"""relief.fields -- the Phase-0 pattern layers of a relief panel.

Each field takes the sample grid and a parameter dict and returns a scalar
array of the same shape.  Fields are registered in `FIELDS` so the operator can
enumerate them without knowing any of them individually.

Normalisation is deliberately *not* done here.  A field returns its natural
range; `relief.transfer.normalize` puts it on a common footing before the
layer's amplitude is applied, so that one amplitude slider means the same thing
whatever the pattern.

References:
  Franz Josef von Gerstner (1802) -- the trochoidal wave, whose sharp crests
    and broad troughs the `steepness` parameter of `wave` interpolates toward;
    see also Alain Fournier and William T. Reeves, "A Simple Model of Ocean
    Waves", Computer Graphics 20(4) (SIGGRAPH 1986), 75-84.
  Jerry Tessendorf, "Simulating Ocean Water", SIGGRAPH course notes,
    1999-2004 -- the observation that a horizontal displacement proportional to
    the wave gradient "sharpens peaks in the height field and broadens
    valleys", which is what `steepness` emulates here without displacing the
    grid.
  Benoit B. Mandelbrot, "The Fractal Geometry of Nature", Freeman, 1982;
    Dietmar Saupe, "Algorithms for random fractals", in Heinz-Otto Peitgen and
    Dietmar Saupe (eds.), "The Science of Fractal Images", Springer, 1988 --
    the spectral-synthesis fractional Brownian surface reused by `fbm`.
  Christiaan Huygens, "Traite de la Lumiere" (1690) -- the superposition of
    circular wavefronts that `ripple` draws.
"""

import math

import numpy as np

from . import warp as _warp


def _guided_phase(X, Y, p):
    """Phase of a wave whose direction may vary across the panel."""
    kind = p.get('orient', 'CONSTANT')
    lam = max(float(p.get('wavelength', 0.5)), 1e-6)
    if kind == 'CONSTANT':
        # Closed form: no solve needed, and no periodic wrap at the rim.
        a = float(p.get('angle', 0.0))
        k = 2.0 * math.pi / lam
        return k * (X * math.cos(a) + Y * math.sin(a)) + float(p.get('phase', 0.0))
    theta = _warp.orientation_field(
        X, Y, kind=kind, angle=float(p.get('angle', 0.0)),
        seed=int(p.get('seed', 1)), freq=float(p.get('orient_freq', 0.5)),
        swirl=float(p.get('swirl', 1.0)))
    return _warp.phase_from_direction(theta, lam, X, Y) + \
        float(p.get('phase', 0.0))


def _sharpen(s, steepness):
    """Trochoidal-ish crest sharpening of a unit sinusoid.

    At steepness 0 this is the sinusoid unchanged.  As it rises the crests
    narrow and the troughs broaden, approaching the cusped profile of a
    Gerstner wave -- obtained here by reshaping the profile rather than by
    displacing the grid, so the result stays a single-valued height field and
    cannot fold over itself.
    """
    q = float(np.clip(steepness, 0.0, 1.0))
    if q <= 0.0:
        return s
    u = 0.5 * (s + 1.0)                       # to [0,1]
    u = u ** (1.0 + 3.0 * q)                  # bias toward the trough
    return 2.0 * u - 1.0


def wave(X, Y, info, p):
    """A single directional wave, optionally steered by an orientation field."""
    return _sharpen(np.sin(_guided_phase(X, Y, p)),
                    p.get('steepness', 0.0))


def wave_train(X, Y, info, p):
    """Several waves of jittered direction and wavelength, summed.

    With an orientation field other than CONSTANT each component is steered by
    the same field, so they stay coherent rather than cancelling.
    """
    n = max(1, int(p.get('count', 3)))
    rng = np.random.default_rng(int(p.get('seed', 1)) & 0x7FFFFFFF)
    spread = float(p.get('spread', 0.4))
    lam0 = max(float(p.get('wavelength', 0.5)), 1e-6)
    out = np.zeros(X.shape)
    for i in range(n):
        q = dict(p)
        q['angle'] = float(p.get('angle', 0.0)) + rng.uniform(-spread, spread)
        q['wavelength'] = lam0 * float(rng.uniform(0.75, 1.35))
        q['phase'] = float(rng.uniform(0.0, 2.0 * math.pi))
        q['seed'] = int(p.get('seed', 1)) + 101 * i
        out += _sharpen(np.sin(_guided_phase(X, Y, q)),
                        p.get('steepness', 0.0)) / math.sqrt(n)
    return out


def ripple(X, Y, info, p):
    """Interfering circular wavefronts from a few point sources."""
    n = max(1, int(p.get('sources', 3)))
    rng = np.random.default_rng(int(p.get('seed', 1)) & 0x7FFFFFFF)
    lam = max(float(p.get('wavelength', 0.35)), 1e-6)
    k = 2.0 * math.pi / lam
    hx = float(np.abs(X).max())
    hy = float(np.abs(Y).max())
    out = np.zeros(X.shape)
    for i in range(n):
        cx = rng.uniform(-hx, hx)
        cy = rng.uniform(-hy, hy)
        r = np.hypot(X - cx, Y - cy)
        out += np.sin(k * r + rng.uniform(0.0, 2.0 * math.pi))
    return out / math.sqrt(n)


def fbm(X, Y, info, p):
    """Fractional Brownian surface by spectral synthesis.

    Reuses the mode generators already shipped for the Fractal Relief
    generator rather than duplicating them.
    """
    # The mode generators live in the `ifs` engine package (extracted there
    # from fractal_surface_generator in 1.14.21); reuse, do not duplicate.
    try:
        from ..ifs.spectral import (fbm_modes, weierstrass_modes, eval_field)
    except ImportError:                       # flat import outside the package
        from ifs.spectral import (fbm_modes, weierstrass_modes, eval_field)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=-1)
    if p.get('method', 'FBM') == 'WEIERSTRASS':
        modes = weierstrass_modes(int(p.get('octaves', 8)),
                                  float(p.get('lacunarity', 2.0)),
                                  float(p.get('dim', 2.3)))
    else:
        modes = fbm_modes(int(p.get('modes', 240)), int(p.get('octaves', 8)),
                          float(p.get('lacunarity', 2.0)),
                          float(p.get('hurst', 0.7)), int(p.get('seed', 1)))
    return eval_field(pts, modes).reshape(X.shape)


def drumhead(X, Y, info, p):
    """Circular membrane eigenmode -- pair it with the DISC panel shape."""
    from . import plates as _plates
    return _plates.circular_membrane(X, Y, m=int(p.get('mode_m', 1)),
                                     n=int(p.get('mode_n', 1)),
                                     phase=float(p.get('phase', 0.0)))


def membrane(X, Y, info, p):
    """Clamped rectangular membrane eigenmode."""
    from . import plates as _plates
    return _plates.rect_membrane(X, Y, m=int(p.get('mode_m', 2)),
                                 n=int(p.get('mode_n', 3)))


def chladni(X, Y, info, p):
    """Free-plate Chladni figure.

    `exact=True` runs the Rayleigh-Ritz solver and picks a real mode; the
    fast path is Rayleigh's cosine combination with a freely mixable chi.
    """
    from . import plates as _plates
    if p.get('exact', True):
        lam, vec, s = _plates.free_plate_cached(
            s=int(p.get('ritz', 10)), poisson=float(p.get('poisson', 0.225)),
            aspect=1.0, count=int(p.get('mode_index', 1)) + 6)
        # Skip the three rigid-body modes; index 1 is the first real figure.
        real = [i for i in range(lam.size) if lam[i] > 1.0]
        if not real:
            return np.zeros(X.shape)
        k = real[min(max(int(p.get('mode_index', 1)) - 1, 0), len(real) - 1)]
        return _plates.plate_mode_field(X, Y, vec[:, k], s)
    return _plates.chladni_rayleigh(X, Y, m=int(p.get('mode_m', 2)),
                                    n=int(p.get('mode_n', 3)),
                                    chi=float(p.get('chi', 1.0)))


def zernike_field(X, Y, info, p):
    """A Zernike aberration mode on the unit disc."""
    from . import special as _sp
    R = min(float(np.abs(X).max()), float(np.abs(Y).max())) or 1.0
    rho = np.clip(np.hypot(X, Y) / R, 0.0, 1.0)
    th = np.arctan2(Y, X)
    n = int(p.get('zern_n', 4))
    m = int(p.get('zern_m', 2))
    if (n - abs(m)) % 2 or abs(m) > n:      # nearest legal m
        m = abs(m) - 1 if (n - abs(m)) % 2 else abs(m)
        m = max(-n, min(n, m))
    out = _sp.zernike(n, m, rho, th)
    return np.where(np.hypot(X, Y) <= R, out, 0.0)


def hermite_gauss(X, Y, info, p):
    """Hermite-Gauss (TEM_mn) transverse mode."""
    from . import special as _sp
    w = max(float(p.get('waist', 0.5)), 1e-6)
    hx = float(np.abs(X).max()) or 1.0
    sx = X / (w * hx)
    sy = Y / (w * hx)
    return (_sp.hermite_function(int(p.get('mode_m', 2)), math.sqrt(2.0) * sx)
            * _sp.hermite_function(int(p.get('mode_n', 1)),
                                   math.sqrt(2.0) * sy))


# id -> (menu label, description, function)
FIELDS = {
    'WAVE':       ("Directional Wave",
                   "One wave; steer it with an orientation field", wave),
    'WAVE_TRAIN': ("Wave Train",
                   "Several jittered waves summed -- the drapery base",
                   wave_train),
    'RIPPLE':     ("Radial Ripple",
                   "Interfering circular wavefronts", ripple),
    'FBM':        ("Fractal (fBm)",
                   "Spectral-synthesis fractional Brownian surface", fbm),
    'CHLADNI':    ("Chladni Plate",
                   "Free-plate vibration figure, solved by Rayleigh-Ritz",
                   chladni),
    'DRUMHEAD':   ("Drumhead",
                   "Circular membrane eigenmode; use the Disc shape",
                   drumhead),
    'MEMBRANE':   ("Rectangular Membrane",
                   "Clamped membrane eigenmode sin(m)sin(n)", membrane),
    'ZERNIKE':    ("Zernike",
                   "Optical aberration mode on the disc", zernike_field),
    'HERMITE':    ("Hermite-Gauss",
                   "TEM_mn laser transverse mode", hermite_gauss),
}


def evaluate(kind, X, Y, info, params):
    """Evaluate a registered field by id."""
    try:
        fn = FIELDS[kind][2]
    except KeyError:
        raise ValueError("unknown field: %r" % (kind,))
    return fn(X, Y, info, params or {})


def _selftest():
    ok = True
    from . import grid as _grid

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=97)

    for kind in FIELDS:
        h = evaluate(kind, X, Y, info, {'seed': 3})
        finite = np.isfinite(h).all()
        varies = h.std() > 1e-6
        print("fields: %-11s shape=%s finite=%s sd=%.4f"
              % (kind, h.shape, finite, h.std()))
        ok = ok and h.shape == X.shape and finite and varies

    # A constant-direction wave has the wavelength it was asked for: count
    # sign changes along the direction of travel.
    lam = 0.4
    h = evaluate('WAVE', X, Y, info,
                 {'wavelength': lam, 'angle': 0.0, 'orient': 'CONSTANT'})
    row = h[info['ny'] // 2, :]
    crossings = int(np.sum(np.diff(np.signbit(row))))
    expect = int(round(info['width'] / (lam / 2.0))) - 1
    print("fields: WAVE lambda=%.2f -> %d sign changes (expect ~%d)"
          % (lam, crossings, expect))
    ok = ok and abs(crossings - expect) <= 2

    # Steepness sharpens crests.  Narrow peaks standing over a broad trough
    # put most of the surface low with a long tail upward, so the skew goes
    # POSITIVE -- the "sharpens peaks, broadens valleys" profile.  A sinusoid
    # is symmetric, so it starts at zero.
    flat = evaluate('WAVE', X, Y, info, {'wavelength': 0.5, 'steepness': 0.0})
    sharp = evaluate('WAVE', X, Y, info, {'wavelength': 0.5, 'steepness': 0.9})

    def sk(a):
        return float(((a - a.mean()) ** 3).mean() / (a.std() ** 3 + 1e-30))

    # Fraction of the panel below the midpoint should also rise.
    below = lambda a: float((a < a.mean()).mean())
    print("fields: steepness skew %.3f -> %.3f, fraction below mean %.2f -> %.2f"
          % (sk(flat), sk(sharp), below(flat), below(sharp)))
    ok = ok and abs(sk(flat)) < 0.1                  # sinusoid is symmetric
    ok = ok and sk(sharp) > sk(flat) + 0.2           # crests narrowed
    ok = ok and below(sharp) > below(flat)           # troughs broadened

    # Determinism.
    a = evaluate('WAVE_TRAIN', X, Y, info, {'seed': 7})
    b = evaluate('WAVE_TRAIN', X, Y, info, {'seed': 7})
    c = evaluate('WAVE_TRAIN', X, Y, info, {'seed': 8})
    ok = ok and np.array_equal(a, b) and not np.allclose(a, c)
    print("fields: wave_train repeatable=%s seed-varies=%s"
          % (np.array_equal(a, b), not np.allclose(a, c)))

    # An orientation field really does steer the wave: a curl-guided wave
    # differs from the constant-direction one.
    g = evaluate('WAVE', X, Y, info,
                 {'wavelength': 0.4, 'orient': 'CURL', 'seed': 5})
    d = evaluate('WAVE', X, Y, info,
                 {'wavelength': 0.4, 'orient': 'CONSTANT'})
    print("fields: guided vs constant rms difference = %.4f"
          % float(np.sqrt(((g - d) ** 2).mean())))
    ok = ok and np.sqrt(((g - d) ** 2).mean()) > 0.2

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
