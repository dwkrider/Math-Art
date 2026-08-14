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


def _tiled_wave(X, Y, info, p, mode):
    """A wave built in the basis whose boundary condition gives the seam.

    Torus uses a wavevector snapped to the dual lattice; mirror and antimirror
    use separable cosine or sine products, which are the only forms satisfying
    Neumann or Dirichlet on all four edges at once.  Being separable, they
    cannot follow an orientation field -- so a tiled wave is a straight wave,
    and the caller is told rather than being given a seam.
    """
    from . import tiling as _t
    Lx, Ly = _t.periods(info)
    lam = float(p.get('wavelength', 0.5))
    ang = float(p.get('angle', 0.0))
    if mode == 'TORUS':
        kx, ky, _rep = _t.snap_torus(lam, ang, Lx, Ly)
        return np.sin(kx * (X - X.min()) + ky * (Y - Y.min())
                      + float(p.get('phase', 0.0)))
    m, n = _t.snap_symmetric(lam, ang, Lx, Ly, mode)
    return _t.symmetric_wave(X, Y, info, m, n, mode)


def wave(X, Y, info, p):
    """A single directional wave, optionally steered by an orientation field."""
    mode = p.get('tiling', 'NONE')
    if mode in ('TORUS', 'MIRROR', 'ANTIMIRROR'):
        return _sharpen(_tiled_wave(X, Y, info, p, mode),
                        p.get('steepness', 0.0))
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
    mode = p.get('tiling', 'NONE')
    out = np.zeros(X.shape)
    for i in range(n):
        q = dict(p)
        q['angle'] = float(p.get('angle', 0.0)) + rng.uniform(-spread, spread)
        q['wavelength'] = lam0 * float(rng.uniform(0.75, 1.35))
        q['phase'] = float(rng.uniform(0.0, 2.0 * math.pi))
        q['seed'] = int(p.get('seed', 1)) + 101 * i
        if mode in ('TORUS', 'MIRROR', 'ANTIMIRROR'):
            comp = _tiled_wave(X, Y, info, q, mode)
        else:
            comp = np.sin(_guided_phase(X, Y, q))
        out += _sharpen(comp, p.get('steepness', 0.0)) / math.sqrt(n)
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
    mode = p.get('tiling', 'NONE')
    if mode in ('TORUS', 'MIRROR', 'ANTIMIRROR'):
        from . import tiling as _t
        return _t.lattice_fbm(X, Y, info, mode,
                              hurst=float(p.get('hurst', 0.7)),
                              octaves=int(p.get('octaves', 8)),
                              count=int(p.get('modes', 240)),
                              seed=int(p.get('seed', 1)),
                              lacunarity=float(p.get('lacunarity', 2.0)),
                              method=p.get('method', 'FBM'),
                              dim=float(p.get('dim', 2.3)))
    aniso = float(p.get('anisotropy', 0.0))
    if aniso > 0.0:
        from . import tiling as _t
        return _t.anisotropic_fbm(X, Y, info, hurst=float(p.get('hurst', 0.7)),
                                  octaves=int(p.get('octaves', 8)),
                                  count=int(p.get('modes', 240)),
                                  seed=int(p.get('seed', 1)),
                                  wind=float(p.get('wind', 0.0)),
                                  spread=aniso,
                                  method=p.get('method', 'FBM'),
                                  dim=float(p.get('dim', 2.3)),
                                  lacunarity=float(p.get('lacunarity', 2.0)))
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


def elliptic(X, Y, info, p):
    """Doubly periodic relief from the Weierstrass/Jacobi functions."""
    from . import elliptic as _e
    return _e.elliptic_field(X, Y, info, kind=p.get('ell_kind', 'WP'),
                             tau_re=float(p.get('tau_re', 0.0)),
                             tau_im=float(p.get('tau_im', 1.0)),
                             cells=float(p.get('ell_cells', 1.0)),
                             part=p.get('ell_part', 'SPHERE'))


def truchet(X, Y, info, p):
    """Truchet arc tiling -- smooth meandering lanes from a random assembly."""
    from . import tiles as _t
    return _t.truchet(X, Y, info, cells=int(p.get('tile_cells', 6)),
                      seed=int(p.get('seed', 1)),
                      lane=float(p.get('lane', 0.25)),
                      multiscale=float(p.get('multiscale', 0.0)))


def seigaiha(X, Y, info, p):
    """The Japanese overlapping-wave motif."""
    from . import tiles as _t
    return _t.seigaiha(X, Y, info, cells=int(p.get('tile_cells', 6)),
                       seed=int(p.get('seed', 1)),
                       rings=int(p.get('rings', 3)),
                       crown=float(p.get('crown', 0.55)))


def _splat_common(X, Y, info, p, pts, wts):
    """Shared tail of the object and scatter layers."""
    from . import kernels as _k
    mode = p.get('obj_mode', 'SPLAT')
    sigma = float(p.get('sigma', 0.0))
    if sigma <= 0.0:                       # auto bandwidth
        sigma = _k.auto_sigma(pts, area=info['width'] * info['height'])
    sigma = max(sigma, 0.5 * info['dx'])
    kernel = p.get('kernel', 'GAUSSIAN')
    wrap = bool(p.get('wrap', False))

    if mode == 'MAX':
        return _k.splat_max(pts, wts, X, Y, kernel, sigma,
                            smooth=float(p.get('merge', 0.0)))
    if mode == 'INTERPOLATE':
        z = (pts[:, 2] if pts.shape[1] > 2 else
             (wts if wts is not None else np.ones(len(pts))))
        return _k.shepard(pts, z, X, Y, power=float(p.get('power', 2.0)),
                          radius=p.get('shepard_radius'))
    if mode == 'ENGRAVE':
        mask = _k.rasterise(pts, None, X, Y) > 0.0
        if not mask.any():
            return np.zeros(X.shape)
        d = _k.distance_transform(mask, info['dx'], info['dy'], wrap=wrap)
        w = max(float(p.get('groove', 0.08)), 1e-6)
        return -np.clip(1.0 - d / w, 0.0, 1.0)
    # SPLAT (density)
    exact = sigma < 1.5 * info['dx']
    return _k.splat(pts, wts, X, Y, kernel, sigma, wrap=wrap, exact=exact)


def object_field(X, Y, info, p):
    """Relief driven by an arbitrary scene object.

    The operator does the Blender-side work (depsgraph evaluation, point
    sampling, ray casting) and passes plain arrays in, so this stays free of
    `bpy`.  Modes: SPLAT / MAX / INTERPOLATE / ENGRAVE use the points;
    IMPRINT uses a ray-cast depth map, compressed so it reads at panel depth.
    """
    from . import imprint as _imp
    mode = p.get('obj_mode', 'SPLAT')

    if mode == 'IMPRINT':
        depth = p.get('depth_map')
        if depth is None:
            return np.zeros(X.shape)
        m = p.get('obj_mask')
        m = None if m is None else np.asarray(m, dtype=bool)
        return _imp.imprint(np.asarray(depth, dtype=float), m,
                            p.get('compress', 'AHE'),
                            dx=info['dx'], dy=info['dy'],
                            alpha=float(p.get('alpha', 0.1)),
                            beta=float(p.get('beta', 0.85)))

    pts = p.get('points')
    if pts is None or len(pts) == 0:
        return np.zeros(X.shape)
    pts = np.asarray(pts, dtype=float)
    wts = p.get('weights')
    wts = None if wts is None else np.asarray(wts, dtype=float)
    return _splat_common(X, Y, info, p, pts, wts)


def scatter(X, Y, info, p):
    """The same splat engine over a generated point process.

    This is sparse convolution noise: swap the object's points for a Poisson
    (or blue-noise, or low-discrepancy) process and the imprint layer becomes
    a noise generator.  Blue noise is the default because the low-frequency
    energy of a uniform random set survives the kernel's low-pass and reads as
    clumping.
    """
    from . import kernels as _k
    hx = float(np.abs(X).max())
    hy = float(np.abs(Y).max())
    seed = int(p.get('seed', 1))
    n = max(1, int(p.get('points_n', 120)))
    proc = p.get('process', 'BLUE')
    if proc == 'BLUE':
        r = math.sqrt(4.0 * hx * hy / max(n, 1)) * 0.7
        q = _k.poisson_disk(2.0 * hx, 2.0 * hy, r, seed=seed)
    elif proc == 'HALTON':
        q = _k.halton(n, seed=seed) * np.array([2.0 * hx, 2.0 * hy])
    else:                                   # UNIFORM
        rng = np.random.default_rng(seed & 0x7FFFFFFF)
        q = rng.random((n, 2)) * np.array([2.0 * hx, 2.0 * hy])
    pts = q - np.array([hx, hy])
    rng = np.random.default_rng((seed + 7717) & 0x7FFFFFFF)
    wts = rng.uniform(-1.0, 1.0, size=len(pts))
    return _splat_common(X, Y, info, p, pts, wts)


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
    'ELLIPTIC':   ("Elliptic (doubly periodic)",
                   "Weierstrass/Jacobi functions -- tiles exactly, by theorem",
                   elliptic),
    'TRUCHET':    ("Truchet",
                   "Arc tiles that always join tangentially: flowing lanes",
                   truchet),
    'SEIGAIHA':   ("Seigaiha",
                   "The Japanese overlapping wave-fan ornament", seigaiha),
    'OBJECT':     ("Object",
                   "Imprint a scene object: splat, drape, engrave or press",
                   object_field),
    'SCATTER':    ("Scatter (sparse noise)",
                   "The same kernel splat over a generated point process",
                   scatter),
}


# Display order for the UI.  Deliberately NOT alphabetical: the waves come
# first because they are the organic base most panels start from, the
# closed-form vibration and optical modes next, and the point-driven layers
# last -- Object is the one that needs a scene object selected, so it belongs
# at the end of the list rather than in the middle of it.
FIELD_ORDER = [
    'WAVE', 'WAVE_TRAIN', 'RIPPLE', 'FBM',
    'CHLADNI', 'DRUMHEAD', 'MEMBRANE', 'ZERNIKE', 'HERMITE',
    'ELLIPTIC', 'TRUCHET', 'SEIGAIHA',
    'SCATTER', 'OBJECT',
]


def ordered_fields():
    """(id, label, description) in display order, with any stragglers last."""
    seen = [k for k in FIELD_ORDER if k in FIELDS]
    rest = [k for k in sorted(FIELDS) if k not in seen]
    return [(k, FIELDS[k][0], FIELDS[k][1]) for k in seen + rest]


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

    rng = np.random.default_rng(5)
    demo_pts = rng.uniform(-0.8, 0.8, size=(40, 2))
    base = {'seed': 3, 'points': demo_pts,
            'weights': rng.uniform(0.5, 1.5, 40)}
    for kind in FIELDS:
        h = evaluate(kind, X, Y, info, dict(base))
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

    # The two fractal styles must actually be two styles.  This regressed
    # once: the tiling and anisotropy branches built fBm modes unconditionally
    # and dropped `method` on the floor, so every preset that set either one
    # gave the same surface from both menu entries.  Sweep the combinations,
    # because it is the SPECIAL paths that forget.
    Xf, Yf, inf_f = _grid.make_grid(width=2.0, aspect=1.0, resolution=128)
    for tiling_mode in ('NONE', 'TORUS', 'MIRROR', 'ANTIMIRROR'):
        for aniso in (0.0, 0.6):
            base = dict(tiling=tiling_mode, anisotropy=aniso, octaves=7,
                        seed=3, hurst=0.7, dim=2.3, wind=0.4)
            a = fbm(Xf, Yf, inf_f, dict(base, method='FBM'))
            b = fbm(Xf, Yf, inf_f, dict(base, method='WEIERSTRASS'))
            # Normalise before comparing: a difference in overall gain is not
            # a difference in style, and the two spectra scale differently.
            na = a / (a.std() or 1.0)
            nb = b / (b.std() or 1.0)
            rms = float(np.sqrt(((na - nb) ** 2).mean()))
            print("fields: fractal styles differ, tiling=%-10s anisotropy=%.1f "
                  "-> normalised rms %.3f" % (tiling_mode, aniso, rms))
            ok = ok and rms > 0.2

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
