"""relief.symmetry -- wallpaper functions and quasicrystalline wave sums.

Two constructions that both make a pattern out of plane waves, and differ in
whether the wavevectors close under a crystallographic group.

**Wallpaper.**  Sum lattice waves, then average the sum over a point group.
Averaging a function over a group produces something the group cannot change,
so the result is invariant under every symmetry of that group *by construction*
rather than by arrangement -- and because each wave already has integer
frequency on the lattice, it is invariant under the lattice translations too.
Group plus lattice is exactly a wallpaper group, so all seventeen come out of
one mechanism: pick the lattice, list the generators, average.  This is Farris's
recipe method with the recipes derived rather than tabulated.

**Quasicrystal.**  Take N plane waves with wavevectors at equal angles around
the circle.  For N = 4 or 6 the wavevectors span a lattice and the result is an
ordinary periodic pattern; for N = 5, 7 or higher the angles are incommensurate
and no lattice contains them, so the field has perfect N-fold rotational
symmetry and no translational period at all.  It is ordered everywhere and
repeats nowhere -- the two-dimensional shadow of de Bruijn's projection of a
higher-dimensional lattice, and the same arithmetic that makes a Penrose tiling
aperiodic.

That is beautiful and it does not tile, so an **approximant** is offered:
snap each wavevector to the nearest one the panel's dual lattice actually
carries.  The pattern then repeats exactly at the panel edge while remaining
locally indistinguishable from the quasicrystal over a few cells -- which is
precisely what a crystallographic approximant is in the physics.

References:
  Frank A. Farris, "Creating Symmetry: The Artful Mathematics of Wallpaper
    Patterns", Princeton University Press, 2015 -- the wave-superposition
    construction of wallpaper functions, and the recipes for the 17 groups.
  N. G. de Bruijn, "Algebraic theory of Penrose's non-periodic tilings of the
    plane", Indagationes Mathematicae 43, 1981, 39-66 -- the projection view
    of aperiodic order.
  Dov Levine and Paul J. Steinhardt, "Quasicrystals: A New Class of Ordered
    Structures", Phys. Rev. Lett. 53, 1984, 2477-2480 -- quasicrystals and
    their periodic approximants.
  Theo Hahn (ed.), "International Tables for Crystallography, Volume A",
    Springer -- the plane groups and their generators.
"""

import math

import numpy as np

# Each group: the lattice it lives on, and generators as (matrix, translation)
# in FRACTIONAL lattice coordinates.  Translations are fractions of a lattice
# vector, which is how a glide (a reflection combined with a half-step) is
# expressed.  The full group is generated from these by closure, so only the
# generators need to be right.
SQUARE, HEX, RECT = 'SQUARE', 'HEX', 'RECT'

GROUPS = {
    'P1':   (SQUARE, []),
    'P2':   (SQUARE, [((-1, 0, 0, -1), (0.0, 0.0))]),
    'PM':   (RECT,   [((-1, 0, 0, 1), (0.0, 0.0))]),
    'PG':   (RECT,   [((-1, 0, 0, 1), (0.0, 0.5))]),
    'CM':   (RECT,   [((-1, 0, 0, 1), (0.0, 0.0)),
                      ((1, 0, 0, 1), (0.5, 0.5))]),
    'PMM':  (RECT,   [((-1, 0, 0, 1), (0.0, 0.0)),
                      ((1, 0, 0, -1), (0.0, 0.0))]),
    'PMG':  (RECT,   [((-1, 0, 0, -1), (0.0, 0.0)),
                      ((-1, 0, 0, 1), (0.5, 0.0))]),
    'PGG':  (RECT,   [((-1, 0, 0, -1), (0.0, 0.0)),
                      ((-1, 0, 0, 1), (0.5, 0.5))]),
    'CMM':  (RECT,   [((-1, 0, 0, 1), (0.0, 0.0)),
                      ((1, 0, 0, -1), (0.0, 0.0)),
                      ((1, 0, 0, 1), (0.5, 0.5))]),
    'P4':   (SQUARE, [((0, -1, 1, 0), (0.0, 0.0))]),
    'P4M':  (SQUARE, [((0, -1, 1, 0), (0.0, 0.0)),
                      ((0, 1, 1, 0), (0.0, 0.0))]),
    'P4G':  (SQUARE, [((0, -1, 1, 0), (0.0, 0.0)),
                      ((0, 1, 1, 0), (0.5, 0.5))]),
    # On the hexagonal lattice the 3-fold rotation is the integer matrix
    # [[0,-1],[1,-1]] -- rotations look like this only in the lattice basis,
    # which is the reason to work in fractional coordinates throughout.
    'P3':   (HEX,    [((0, -1, 1, -1), (0.0, 0.0))]),
    'P3M1': (HEX,    [((0, -1, 1, -1), (0.0, 0.0)),
                      ((0, -1, -1, 0), (0.0, 0.0))]),
    'P31M': (HEX,    [((0, -1, 1, -1), (0.0, 0.0)),
                      ((0, 1, 1, 0), (0.0, 0.0))]),
    'P6':   (HEX,    [((1, -1, 1, 0), (0.0, 0.0))]),
    'P6M':  (HEX,    [((1, -1, 1, 0), (0.0, 0.0)),
                      ((0, 1, 1, 0), (0.0, 0.0))]),
}
GROUP_ORDER = ('P1', 'P2', 'PM', 'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',
               'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M')


def _mul(g, h):
    """Compose two (matrix, translation) pairs; translations reduce mod 1."""
    (a, b, c, d), (tx, ty) = g
    (e, f, i, j), (ux, uy) = h
    m = (a * e + b * i, a * f + b * j, c * e + d * i, c * f + d * j)
    t = ((a * ux + b * uy + tx) % 1.0, (c * ux + d * uy + ty) % 1.0)
    return m, (round(t[0], 9) % 1.0, round(t[1], 9) % 1.0)


def close_group(gens, limit=64):
    """Every element generated by `gens`, including the identity.

    Small groups -- the largest plane point group here has 12 elements -- so a
    breadth-first closure is ample, and it means only the generators have to be
    stated correctly rather than a full multiplication table.
    """
    ident = ((1, 0, 0, 1), (0.0, 0.0))
    seen = {ident}
    frontier = [ident]
    gs = [(tuple(m), (float(t[0]) % 1.0, float(t[1]) % 1.0)) for m, t in gens]
    while frontier and len(seen) <= limit:
        cur = frontier.pop()
        for g in gs:
            nxt = _mul(g, cur)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return sorted(seen)


def lattice_basis(kind, aspect=1.0):
    """Columns are the lattice vectors, in panel units of one period."""
    if kind == HEX:
        # 60-degree lattice.  A rectangle holds a whole number of cells only
        # when its aspect is sqrt(3), which `hex_aspect` reports.
        return np.array([[1.0, 0.5], [0.0, math.sqrt(3.0) / 2.0]])
    if kind == RECT:
        return np.array([[1.0, 0.0], [0.0, float(aspect)]])
    return np.array([[1.0, 0.0], [0.0, 1.0]])


def wallpaper(X, Y, info, group='P4M', waves=6, seed=1, cells=2.0,
              freq_max=3, aspect=1.0):
    """A wallpaper function of `group`, from a symmetrised sum of lattice waves.

    `cells` is how many lattice cells span the panel width; `waves` how many
    distinct lattice frequencies go into the sum before symmetrising.
    """
    name = str(group).upper()
    kind, gens = GROUPS.get(name, GROUPS['P4M'])
    elements = close_group(gens)

    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    B = lattice_basis(kind, aspect)
    Binv = np.linalg.inv(B)

    # Panel -> fractional lattice coordinates, `cells` cells across the width.
    sx = float(cells) / (2.0 * hx)
    sy = float(cells) / (2.0 * hx)          # isotropic: square cells stay square
    px = (X + hx) * sx
    py = (Y + hy) * sy
    U = Binv[0, 0] * px + Binv[0, 1] * py
    V = Binv[1, 0] * px + Binv[1, 1] * py

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fm = max(1, int(freq_max))
    modes = []
    for _ in range(max(1, int(waves))):
        n = int(rng.integers(-fm, fm + 1))
        m = int(rng.integers(-fm, fm + 1))
        if n == 0 and m == 0:
            n = 1
        modes.append((n, m, float(rng.uniform(0.0, 2.0 * math.pi)),
                      1.0 / math.hypot(n, m)))

    out = np.zeros(X.shape)
    for A, t in elements:
        a, b, c, d = A
        # Act on the fractional coordinates, then evaluate the lattice waves.
        Ug = a * U + b * V + t[0]
        Vg = c * U + d * V + t[1]
        for n, m, ph, amp in modes:
            out += amp * np.cos(2.0 * math.pi * (n * Ug + m * Vg) + ph)
    out /= float(len(elements))
    s = out.std()
    return out / s if s > 1e-12 else out


def hex_aspect():
    """Panel aspect at which a hexagonal-lattice group tiles the rectangle."""
    return math.sqrt(3.0)


def quasicrystal(X, Y, info, fold=5, cells=6.0, phase=0.0, approximant=False,
                 sharp=0.0, extent=None):
    """Sum of `fold` plane waves at equal angles: aperiodic order.

    The directions are spread over a half-turn, since a cosine is even and a
    wave at angle t is the same wave as one at t + pi.  With zero phase the
    field is therefore invariant under rotation by **pi / fold** -- so five
    waves give a ten-fold pattern, which is the familiar one.  Five-fold
    symmetry is impossible for any lattice, which is exactly why this field
    cannot have a translational period.

    With `approximant`, each wavevector is snapped to the panel's dual
    lattice, which makes the field periodic at the cost of exact rotational
    symmetry -- the standard trade, and the reason approximants exist.

    `extent` overrides the panel half-width taken from the samples.  Rotating
    the sample points changes their bounding box, which would otherwise change
    the frequency and make a symmetry test compare two different fields.
    """
    n = max(2, int(fold))
    hx = float(extent) if extent else (float(np.abs(X).max()) or 1.0)
    hy = float(extent) if extent else (float(np.abs(Y).max()) or 1.0)
    k0 = math.pi * float(cells) / hx
    Lx = 2.0 * hx
    Ly = 2.0 * hy

    out = np.zeros(X.shape)
    for j in range(n):
        th = math.pi * j / n
        kx = k0 * math.cos(th)
        ky = k0 * math.sin(th)
        if approximant:
            # Nearest wavevector the panel's own lattice carries.
            kx = 2.0 * math.pi * round(kx * Lx / (2.0 * math.pi)) / Lx
            ky = 2.0 * math.pi * round(ky * Ly / (2.0 * math.pi)) / Ly
            out += np.cos(kx * (X + hx) + ky * (Y + hy) + float(phase))
        else:
            out += np.cos(kx * X + ky * Y + float(phase))
    out /= math.sqrt(n)
    if sharp > 0.0:
        # Terracing the sum is what turns a wave interference pattern into
        # something that reads as tiles rather than as ripples.
        out = np.tanh(float(sharp) * out)
    s = out.std()
    return out / s if s > 1e-12 else out


def _selftest():
    ok = True
    from . import grid as _grid
    from . import tiling as _tiling

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=160)

    # Group closure: the orders are known from crystallography, so this is a
    # real check on the generators rather than a smoke test.
    want = {'P1': 1, 'P2': 2, 'PM': 2, 'PG': 2, 'CM': 4, 'PMM': 4, 'PMG': 4,
            'PGG': 4, 'CMM': 8, 'P4': 4, 'P4M': 8, 'P4G': 8, 'P3': 3,
            'P3M1': 6, 'P31M': 6, 'P6': 6, 'P6M': 12}
    bad = []
    for name in GROUP_ORDER:
        kind, gens = GROUPS[name]
        got = len(close_group(gens))
        if got != want[name]:
            bad.append("%s got %d want %d" % (name, got, want[name]))
    print("symmetry: group orders %s"
          % ("all correct" if not bad else "WRONG: " + "; ".join(bad)))
    ok = ok and not bad

    # Every group produces a finite, varying field.
    for name in GROUP_ORDER:
        h = wallpaper(X, Y, info, group=name, waves=5, seed=3, cells=2.0)
        fin = np.isfinite(h).all()
        ok = ok and fin and h.std() > 1e-6
        if not fin:
            print("symmetry: %s NOT FINITE" % name)

    # **The point of the construction**: the field is invariant under every
    # generator, to machine precision.  Averaging over a group cannot produce
    # anything else, so a failure here means the generator matrices are wrong
    # -- which is exactly what this is for.
    worst = 0.0
    for name in GROUP_ORDER:
        kind, gens = GROUPS[name]
        if not gens:
            continue
        B = lattice_basis(kind)
        Binv = np.linalg.inv(B)
        rng = np.random.default_rng(11)
        P = rng.random((400, 2)) * 2.0 - 1.0          # panel points
        hxx = 1.0
        px = (P[:, 0] + hxx) * (2.0 / 2.0)
        py = (P[:, 1] + hxx) * (2.0 / 2.0)
        U = Binv[0, 0] * px + Binv[0, 1] * py
        V = Binv[1, 0] * px + Binv[1, 1] * py

        def eval_at(u, v, name=name, kind=kind):
            gens_ = GROUPS[name][1]
            elements = close_group(gens_)
            rng2 = np.random.default_rng(3)
            modes = []
            for _ in range(5):
                n_ = int(rng2.integers(-3, 4))
                m_ = int(rng2.integers(-3, 4))
                if n_ == 0 and m_ == 0:
                    n_ = 1
                modes.append((n_, m_, float(rng2.uniform(0, 2 * math.pi)),
                              1.0 / math.hypot(n_, m_)))
            acc = np.zeros(np.shape(u))
            for A, t in elements:
                a, b, c, d = A
                ug = a * u + b * v + t[0]
                vg = c * u + d * v + t[1]
                for n_, m_, ph, amp in modes:
                    acc += amp * np.cos(2 * math.pi * (n_ * ug + m_ * vg) + ph)
            return acc / len(elements)

        base = eval_at(U, V)
        for A, t in gens:
            a, b, c, d = A
            Ug = a * U + b * V + t[0]
            Vg = c * U + d * V + t[1]
            err = float(np.abs(eval_at(Ug, Vg) - base).max())
            worst = max(worst, err)
    print("symmetry: worst wallpaper generator violation %.2e "
          "(invariance is by construction)" % worst)
    ok = ok and worst < 1e-10

    # Lattice periodicity: shifting by one cell changes nothing.
    for name in ('P1', 'P4M', 'CMM'):
        kind, gens = GROUPS[name]
        B = lattice_basis(kind)
        Binv = np.linalg.inv(B)
        rng = np.random.default_rng(5)
        P = rng.random((300, 2))
        U, V = P[:, 0] * 4.0, P[:, 1] * 4.0
        elements = close_group(gens)

        def f(u, v, elements=elements):
            acc = np.zeros(np.shape(u))
            for A, t in elements:
                a, b, c, d = A
                acc += np.cos(2 * math.pi * (2 * (a * u + b * v + t[0])
                                             + (c * u + d * v + t[1])))
            return acc / len(elements)
        shift = float(np.abs(f(U + 1.0, V) - f(U, V)).max())
        ok = ok and shift < 1e-10
    print("symmetry: lattice translation invariance %.2e" % shift)

    # Quasicrystal: exact N-fold rotational symmetry about the centre.  This
    # is what distinguishes it from a wave train -- and 5-fold is impossible
    # for any lattice, which is the whole point.
    for fold in (5, 7, 8, 11):
        h = quasicrystal(X, Y, info, fold=fold, cells=5.0, extent=1.0)
        # Directions span a half-turn, so the symmetry angle is pi / fold.
        ang = math.pi / fold
        ca, sa = math.cos(ang), math.sin(ang)
        Xr = ca * X - sa * Y
        Yr = sa * X + ca * Y
        # `extent` pinned, or the rotated bounding box would rescale the
        # wavevectors and this would compare two different patterns.
        hr = quasicrystal(Xr, Yr, info, fold=fold, cells=5.0, extent=1.0)
        # Compare only where the rotated sample still lies inside the panel.
        m = (np.abs(Xr) < 0.9) & (np.abs(Yr) < 0.9)
        err = float(np.abs(hr[m] - h[m]).max())
        print("symmetry: quasicrystal %2d-fold rotation error %.2e"
              % (fold, err))
        ok = ok and err < 1e-9

    # ... and it genuinely does NOT tile, while its approximant does.
    q = quasicrystal(X, Y, info, fold=5, cells=5.0, approximant=False)
    qa = quasicrystal(X, Y, info, fold=5, cells=5.0, approximant=True)
    c1 = _tiling.check(q, 'TORUS', info, 'QUASI')
    c2 = _tiling.check(qa, 'TORUS', info, 'QUASI')
    print("symmetry: quasicrystal joint x%.3f ok=%s | approximant x%.3f ok=%s"
          % (c1['step'], c1['ok'], c2['step'], c2['ok']))
    ok = ok and c2['ok'] and not c1['ok']

    # An approximant is still locally quasicrystal-like: it must not collapse
    # into a plain grating, which is the failure mode of snapping too hard.
    print("symmetry: approximant vs quasicrystal correlation %.3f"
          % float(np.corrcoef(q.ravel(), qa.ravel())[0, 1]))

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
