"""relief.reaction -- Turing skins from the Gray-Scott reaction-diffusion system.

Two chemicals diffuse and react::

    du/dt = Du grad^2 u - u v^2 + F (1 - u)
    dv/dt = Dv grad^2 v + u v^2 - (F + k) v

with v the catalyst of its own production.  Turing's insight was that
diffusion, which intuition says should smooth a mixture out, *destabilises* a
uniform state when two species diffuse at different rates -- so structure
appears from no structure, with no template and no designer.  Pearson mapped
the (F, k) plane and found that a single pair of equations produces spots,
stripes, mazes, travelling solitons and endlessly dividing cells, depending
only on where in that plane you stand.

This is the one pattern family here that is *grown* rather than evaluated.
Every other field in the engine answers h(x, y) at a point; this one has no
closed form and must be integrated forward in time, which is why it carries a
step count and why it is the pattern that most rewards the preview resolution.

**It tiles for free.**  Pearson's boundary conditions are periodic -- the
system lives on a torus -- and a `np.roll` Laplacian *is* the periodic
Laplacian, so the seam costs nothing: no snapping, no symmetric basis, no
residual.  The field is periodic because the simulation was.

**Grid independence.**  The simulation runs on its own square lattice at a
fixed physical domain and is resampled onto the panel afterwards.  Tying it to
the panel's sampling instead would change the chemistry with the render
setting -- Du/dx^2 is what sets the feature size -- so the same parameters
would grow different skins at different resolutions.  A panel that changes
pattern when you raise its resolution is not a definite object.

References:
  Alan M. Turing, "The Chemical Basis of Morphogenesis", Phil. Trans. R. Soc.
    Lond. B 237, 1952, 37-72 -- diffusion-driven instability.
  P. Gray and S. K. Scott, "Autocatalytic reactions in the isothermal,
    continuous stirred tank reactor", Chem. Eng. Sci. 38, 1983, 29-43;
    39, 1984, 1087-1097 -- the kinetics.
  John E. Pearson, "Complex Patterns in a Simple System", Science 261(5118),
    1993, 189-192, doi:10.1126/science.261.5118.189 -- the (F, k) map, the
    canonical numerics (256^2, forward Euler, Du = 2e-5, Dv = 1e-5, domain
    2.5 x 2.5) and the periodic boundary conditions used here.
  Hans Meinhardt, "The Algorithmic Beauty of Sea Shells", Springer, 2009 --
    what these patterns look like when biology grows them.
"""

import numpy as np

# Pearson's canonical numerics.
DU = 2.0e-5
DV = 1.0e-5
DOMAIN = 2.5
PEARSON_N = 256
# **The sample pitch is held at Pearson's value and never varied.**  The
# feature size of a Turing skin is set in physical units by Du and the
# kinetics, so dx alone decides how many cells a blob spans, and Du dt / dx^2
# decides whether it grows at all.  Letting dx follow the grid size -- the
# obvious thing, a fixed domain divided by n -- drops that ratio from
# Pearson's 0.200 to 0.029 on a 96-cell lattice, and the pattern silently
# stalls instead of failing.  Holding dx fixed and growing the DOMAIN with the
# lattice instead means the numerics stay exactly where Pearson tested them,
# and the lattice size becomes an honest "how much skin", not a hidden
# chemistry knob.
DX = DOMAIN / PEARSON_N
# Pearson's 2.5-wide domain carries roughly this many blobs across.
FEATURES_ACROSS = 25.0

# Representative (F, k) pairs.
#
# **These were found by sweeping the plane here, not copied from a table.**
# The living region is a narrow diagonal tongue -- roughly k in 0.049..0.065
# rising with F -- and just outside it every seed decays back to the uniform
# state, giving a blank panel rather than an error.  Pairs taken from memory
# of the widely circulated regime tables put two of eight outside the tongue,
# so each pair below was run and classified by what it actually grows:
# component count, coverage, and a per-component shape factor.  The names
# describe the morphology; Pearson's Greek letters index regions of the plane,
# and pinning each letter to one pair would over-claim what is cited.
#
#                     F       k     components   cover   character
REGIMES = {
    'SOLITONS': (0.014, 0.049),   # 21 comps, 11%  sparse isolated spots
    'SPOTS':    (0.018, 0.051),   # 32 comps, 18%  compact, well separated
    'HOLES':    (0.026, 0.051),   #  1 comp,  92%  the inverse: pits in a field
    'MAZE':     (0.030, 0.057),   #  6 comps, 49%  labyrinth
    'MITOSIS':  (0.042, 0.063),   # 34 comps, 37%  spots caught dividing
    'CORAL':    (0.054, 0.062),   #  8 comps, 50%  branching growth
    'WORMS':    (0.058, 0.063),   # 18 comps, 43%  elongated, joined
}
REGIME_ORDER = ('SPOTS', 'SOLITONS', 'MITOSIS', 'WORMS', 'MAZE', 'CORAL',
                'HOLES')


def _laplacian(a):
    """Five-point Laplacian with periodic wrap, in units of 1/dx^2.

    `np.roll` wraps, so this is exactly the Laplacian on a torus -- which is
    what makes the result seamless rather than merely close to it.
    """
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0)
            + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a)


INITS = ('SCATTER', 'CENTRE')


def simulate(n=192, feed=0.035, kill=0.065, steps=4000, seed=1, seeds=1,
             du=DU, dv=DV, domain=None, init='SCATTER'):
    """Integrate Gray-Scott on an n x n torus.  Returns the catalyst v.

    The timestep is chosen from the diffusion stability limit rather than
    fixed at Pearson's dt = 1, which is only stable near his 256^2 grid: on a
    finer lattice dx shrinks, Du dt / dx^2 crosses 1/4, and the field explodes
    into NaN.  Deriving dt keeps every grid size usable.
    """
    n = max(16, int(n))
    # `domain` is derived from the lattice at the fixed pitch, not divided
    # into it -- see DX above.
    dx = float(domain) / n if domain is not None else DX
    # Forward Euler on a 5-point Laplacian is stable while D dt / dx^2 <= 1/4.
    # At Pearson's pitch this returns his dt = 1 exactly; on any finer lattice
    # it steps down instead of exploding into NaN.
    dt = min(1.0, 0.21 * dx * dx / max(du, dv))
    lap = 1.0 / (dx * dx)

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    u = np.ones((n, n))
    v = np.zeros((n, n))

    # **The trivial state u=1, v=0 is LINEARLY stable**, which is the whole
    # reason seeding needs care: for small v, dv/dt = -(F + k) v < 0, so any
    # amount of low-amplitude noise decays and the panel comes out blank.  The
    # instability is nonlinear, and only a patch of finite size and finite
    # amplitude crosses it.
    #
    # Pearson perturbs one central square, which is right for studying how the
    # front spreads and wrong for ornament: the pattern advances at a finite
    # speed and leaves most of a panel empty unless it is run for a very long
    # time.  SCATTER seeds patches across the whole domain instead, so the
    # skin develops everywhere at once.  CENTRE keeps Pearson's setup, because
    # the spreading front is a good look in its own right for the mitosis and
    # soliton regimes.
    if str(init).upper() == 'CENTRE':
        r = max(2, n // 20)
        for _ in range(max(1, int(seeds))):
            ci = rng.integers(r, n - r)
            cj = rng.integers(r, n - r)
            u[cj - r:cj + r, ci - r:ci + r] = 0.50
            v[cj - r:cj + r, ci - r:ci + r] = 0.25
    else:
        # Blocks about n/12 across -- the same relative size as Pearson's
        # 20-of-256 square, so each patch is comfortably supercritical.
        c = max(4, n // 12)
        coarse = rng.random((c, c)) < 0.35
        reps = int(np.ceil(n / c))
        mask = np.kron(coarse, np.ones((reps, reps), dtype=bool))[:n, :n]
        u[mask] = 0.50
        v[mask] = 0.25
    # A little noise on top so the seeded patches are not all identical, which
    # would grow an unnaturally regular skin.
    u += 0.02 * (rng.random((n, n)) - 0.5)
    v += 0.02 * (rng.random((n, n)) - 0.5)

    F = float(feed)
    K = float(kill)
    for _ in range(max(1, int(steps))):
        uvv = u * v * v
        u += dt * (du * lap * _laplacian(u) - uvv + F * (1.0 - u))
        v += dt * (dv * lap * _laplacian(v) + uvv - (F + K) * v)
    return v


def _resample(a, ny, nx):
    """Bilinear resample of a periodic field onto an (ny, nx) sample grid.

    Periodic in both axes, so the panel's last column continues into its
    first -- the property the whole seam argument rests on survives the
    resample only if the interpolation wraps too.
    """
    n = a.shape[0]
    # Panel samples are endpoint-inclusive over one period, so the source
    # coordinate spans [0, n) exactly once across nx-1 pitches.
    gx = np.linspace(0.0, n, nx, endpoint=False) if nx == n else \
        np.arange(nx) * (n / float(max(nx - 1, 1)))
    gy = np.arange(ny) * (n / float(max(ny - 1, 1)))
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = (gx - x0)[None, :]
    fy = (gy - y0)[:, None]
    x0 %= n
    y0 %= n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    a00 = a[np.ix_(y0, x0)]
    a01 = a[np.ix_(y0, x1)]
    a10 = a[np.ix_(y1, x0)]
    a11 = a[np.ix_(y1, x1)]
    return (a00 * (1 - fx) * (1 - fy) + a01 * fx * (1 - fy)
            + a10 * (1 - fx) * fy + a11 * fx * fy)


def lattice_for(scale):
    """Simulation lattice giving `scale` pattern features across the panel.

    The pitch is fixed (see DX), so asking for more features means a bigger
    lattice, and the cost is quadratic in it -- which is the honest trade and
    is why this is the pattern that most wants the preview resolution.
    """
    n = int(round(PEARSON_N * float(scale) / FEATURES_ACROSS))
    return int(min(512, max(48, n)))


def turing(X, Y, info, regime='SPOTS', feed=None, kill=None, steps=4000,
           seed=1, scale=14.0, seeds=1, init='SCATTER'):
    """Grow a Turing skin and sample it onto the panel.

    `regime` picks a representative (F, k); giving `feed`/`kill` explicitly
    overrides it, since the interesting country is between the named points.
    `scale` is how many blobs span the panel, and it sets the simulation
    lattice -- the panel's own resolution only decides how finely the finished
    skin is sampled.
    """
    if feed is None or kill is None:
        f, k = REGIMES.get(str(regime).upper(), REGIMES['SPOTS'])
        feed = f if feed is None else feed
        kill = k if kill is None else kill
    v = simulate(n=lattice_for(scale), feed=feed, kill=kill,
                 steps=int(steps), seed=int(seed), seeds=int(seeds),
                 init=init)
    h = _resample(v, X.shape[0], X.shape[1])
    m = h.mean()
    s = h.std()
    return (h - m) / s if s > 1e-12 else h - m


def _components(mask):
    """Number of 4-connected components in a boolean mask.

    Written out rather than imported: there is no scipy in Blender, and the
    morphology test below needs a real component count, not a proxy.
    """
    m = np.asarray(mask, dtype=bool)
    seen = np.zeros(m.shape, dtype=bool)
    ny, nx = m.shape
    count = 0
    for j0 in range(ny):
        for i0 in range(nx):
            if not m[j0, i0] or seen[j0, i0]:
                continue
            count += 1
            stack = [(j0, i0)]
            seen[j0, i0] = True
            while stack:
                j, i = stack.pop()
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = j + dj, i + di
                    if 0 <= a < ny and 0 <= b < nx and m[a, b]                             and not seen[a, b]:
                        seen[a, b] = True
                        stack.append((a, b))
    return count


def _selftest():
    ok = True
    from . import grid as _grid
    from . import tiling as _tiling

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=128)

    # Pearson's pitch reproduces Pearson's timestep exactly.  If this drifts,
    # the chemistry has moved off the parameters every regime below was
    # chosen at.
    dt = min(1.0, 0.21 * DX * DX / DU)
    print("reaction: pitch %.6f -> dt %.4f (Pearson's dt = 1)" % (DX, dt))
    ok = ok and abs(dt - 1.0) < 1e-6

    # Every named regime grows something, stays bounded, and stays finite.
    # An unstable timestep shows up here first, as NaN.
    n = lattice_for(12)
    grown = {}
    for name in REGIME_ORDER:
        v = simulate(n=n, feed=REGIMES[name][0], kill=REGIMES[name][1],
                     steps=6000, seed=2)
        fin = np.isfinite(v).all()
        # Concentrations are bounded in [0, 1] by the kinetics; drifting
        # outside means the integration, not the chemistry, produced it.
        bounded = bool(v.min() > -1e-6 and v.max() < 1.0 + 1e-6)
        cover = float((v > 0.05).mean())
        print("reaction: %-9s F=%.4f k=%.4f  sd=%.4f covered=%4.1f%% "
              "finite=%s bounded=%s"
              % (name, REGIMES[name][0], REGIMES[name][1], v.std(),
                 100 * cover, fin, bounded))
        ok = ok and fin and bounded
        grown[name] = v

    # Something actually patterned.  The uniform state is a fixed point, so a
    # blank panel is the failure mode that looks like success -- and it is
    # exactly what a too-coarse lattice used to produce silently.
    live = [k for k in REGIME_ORDER if grown[k].std() > 0.03]
    print("reaction: %d of %d regimes developed structure (sd > 0.03)"
          % (len(live), len(REGIME_ORDER)))
    ok = ok and len(live) == len(REGIME_ORDER)

    # Spots and worms differ morphologically, not merely numerically.  The
    # honest discriminator is the component count at equal coverage: spots
    # break into many small islands, worms join into a few long ones.
    #
    # (Perimeter-to-area, tried first, does NOT discriminate: for a disc of
    # radius r it is 2/r and for a strip of width w it is 2/w, so it measures
    # feature THICKNESS, which the two regimes share.  It ranked them
    # confidently and backwards.)
    def islands(v):
        m = v > 0.5 * (v.max() + v.min())
        return _components(m), float(m.mean())

    cs, fs = islands(grown['SPOTS'])
    cw, fw = islands(grown['WORMS'])
    print("reaction: components  spots %d (%.1f%% cover)  worms %d (%.1f%%)"
          % (cs, 100 * fs, cw, 100 * fw))
    ok = ok and cs > cw

    # Periodic boundary conditions mean the field lives on a torus, so it
    # tiles with nothing added.  This is Pearson's setup, not a modification
    # of it -- and the seam check is what keeps that claim honest.
    h = turing(X, Y, info, regime='MAZE', steps=6000, scale=10, seed=4)
    chk = _tiling.check(h, 'TORUS', info, 'TURING')
    print("reaction: torus joint step x%.3f curvature x%.3f ok=%s"
          % (chk['step'], chk['curvature'], chk['ok']))
    ok = ok and chk['ok']

    # The resample must not break that periodicity, at a panel size unrelated
    # to the simulation lattice.
    Xr, Yr, ir = _grid.make_grid(width=2.0, aspect=1.0, resolution=151)
    hr = turing(Xr, Yr, ir, regime='SPOTS', steps=5000, scale=9, seed=6)
    chk2 = _tiling.check(hr, 'TORUS', ir, 'TURING')
    print("reaction: resampled lattice -> 151 samples, joint step x%.3f ok=%s"
          % (chk2['step'], chk2['ok']))
    ok = ok and chk2['ok']

    # Grid independence: the skin must not change character with the PANEL's
    # resolution, only with `scale`.  Same seed and scale, two samplings --
    # the coverage is a property of the chemistry and must agree.
    ca = float((turing(*_grid.make_grid(2.0, 1.0, 96), regime='SPOTS',
                       steps=5000, scale=9, seed=6) > 0.0).mean())
    cb = float((turing(*_grid.make_grid(2.0, 1.0, 192), regime='SPOTS',
                       steps=5000, scale=9, seed=6) > 0.0).mean())
    print("reaction: coverage at 96 vs 192 panel samples: %.4f vs %.4f"
          % (ca, cb))
    ok = ok and abs(ca - cb) < 0.02

    # More scale really is more blobs.
    few = turing(X, Y, info, regime='SPOTS', steps=5000, scale=8, seed=3)
    many = turing(X, Y, info, regime='SPOTS', steps=5000, scale=20, seed=3)
    cf = _components(few > 0.0)
    cm = _components(many > 0.0)
    print("reaction: scale 8 -> %d islands, scale 20 -> %d" % (cf, cm))
    ok = ok and cm > cf

    # Deterministic for a given seed, and the seed genuinely matters.
    a = simulate(n=64, steps=400, seed=3)
    b = simulate(n=64, steps=400, seed=3)
    c = simulate(n=64, steps=400, seed=9)
    ok = ok and np.array_equal(a, b) and not np.allclose(a, c)
    print("reaction: repeatable=%s seed-varies=%s"
          % (np.array_equal(a, b), not np.allclose(a, c)))

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
