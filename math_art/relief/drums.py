"""relief.drums -- membrane modes of shapes with no closed form.

`plates.py` covers the domains whose eigenfunctions are known in closed
form: the rectangle (products of sines) and the disc (Bessel functions).
This module covers the ones that are not, by solving the eigenproblem

    -grad^2 u = lambda u,     u = 0 on the boundary

numerically on a masked grid.  That single routine then yields the elliptical
drum, whose modes are Mathieu functions; the equilateral triangle, whose modes
Lame found in closed form and which is therefore the ideal test of the solver;
and the isospectral pair, the answer to Kac's question.

**Can one hear the shape of a drum?**  Kac asked in 1966 whether the spectrum
determines the domain.  Gordon, Webb and Wolpert answered no in 1992 by
exhibiting two polygons, each built from the same seven right isoceles
triangles glued differently, that are not congruent and yet have *identical*
Dirichlet spectra.  They make an unusually good pair of panels: two plainly
different outlines carrying, mode for mode, the same figure.

That claim is also the strongest available test of everything here.  Two
domains that must agree exactly, and a third whose answer is known in closed
form, check the mask, the assembly, the boundary condition and the solver
together in a way no eyeball ever could.

References:
  Mark Kac, "Can One Hear the Shape of a Drum?", American Mathematical
    Monthly 73(4), part 2, 1966, 1-23.
  Carolyn Gordon, David L. Webb and Scott Wolpert, "One cannot hear the shape
    of a drum", Bull. Amer. Math. Soc. 27(1), 1992, 134-138.
  Gabriel Lame, "Lecons sur la theorie mathematique de l'elasticite des corps
    solides", Bachelier, 1852 -- the equilateral triangle's modes.
  Emile Mathieu, "Memoire sur le mouvement vibratoire d'une membrane de forme
    elliptique", J. Math. Pures Appl. 13, 1868, 137-203.
  Toby Driscoll, "Eigenmodes of isospectral drums", SIAM Review 39(1), 1997,
    1-17 -- computing these spectra accurately, and how hard it is.
"""

import math

import numpy as np

SHAPES = ('TRIANGLE', 'ELLIPSE', 'ISO_ONE', 'ISO_TWO', 'L_SHAPE', 'CIRCLE')


# ------------------------------------------------------------- domains ----

def polygon_mask(poly, X, Y):
    """Points strictly inside a polygon, by ray crossing."""
    P = np.asarray(poly, dtype=float)
    inside = np.zeros(X.shape, dtype=bool)
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % n]
        # Does a ray to +x from each sample cross this edge?
        straddles = (y1 > Y) != (y2 > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (x2 - x1) * (Y - y1) / np.where(y2 != y1, y2 - y1, 1.0) + x1
        inside ^= straddles & (X < xint)
    return inside


def equilateral(side=1.0):
    """An equilateral triangle centred on its centroid."""
    h = side * math.sqrt(3.0) / 2.0
    cy = h / 3.0
    return np.array([[-0.5 * side, -cy], [0.5 * side, -cy], [0.0, h - cy]])


def l_shape():
    """The L-shaped membrane: three quarters of a square."""
    return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 0.0],
                     [0.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])


def isospectral(which=1):
    """One of the Gordon-Webb-Wolpert isospectral polygons.

    Both are unions of the same seven right isoceles triangles, glued along
    different edges, and their Dirichlet spectra agree exactly although the
    outlines plainly differ.

    The vertices are read from Figure 1.1 of Driscoll (1997), in the repo at
    `research/papers/plates-and-vibration/driscoll1997/`.  They are worth
    taking from a source rather than from memory: a first attempt at
    reconstructing them produced two polygons that turned out to be mirror
    images of one another, which are trivially isospectral and would have
    made the check below meaningless while passing it perfectly.
    """
    if int(which) == 1:
        v = [(-3, 1), (-1, 3), (-1, 1), (3, 1),
             (3, -1), (1, -3), (1, -1), (-1, -1)]
    else:
        v = [(-3, 3), (-1, 3), (-1, 1), (1, 1),
             (3, -1), (1, -1), (1, -3), (-3, 1)]
    # Scaled into the unit box the solver works on.  Both polygons shoelace
    # to area 14 -- seven triangles of area 2 -- which is the first thing to
    # check if these are ever edited.
    return np.array(v, dtype=float) / 3.0


def ellipse_mask(X, Y, a=1.0, b=0.6):
    """Points inside an ellipse -- the domain whose modes are Mathieu's."""
    return (X / a) ** 2 + (Y / b) ** 2 < 1.0


def domain_mask(shape, X, Y, ratio=0.6):
    """Interior mask for one of the named drum shapes."""
    shape = str(shape).upper()
    if shape == 'ELLIPSE':
        return ellipse_mask(X, Y, 1.0, float(ratio))
    if shape == 'CIRCLE':
        return ellipse_mask(X, Y, 1.0, 1.0)
    if shape == 'TRIANGLE':
        return polygon_mask(equilateral(2.0), X, Y)
    if shape == 'L_SHAPE':
        return polygon_mask(l_shape(), X, Y)
    if shape == 'ISO_ONE':
        return polygon_mask(isospectral(1), X, Y)
    if shape == 'ISO_TWO':
        return polygon_mask(isospectral(2), X, Y)
    raise ValueError("unknown drum shape: %r" % (shape,))


# -------------------------------------------------------------- solver ----

def membrane_modes(mask, dx, count=12):
    """Lowest `count` Dirichlet eigenpairs of -grad^2 on a masked grid.

    The five-point Laplacian on the interior samples, with u = 0 wherever the
    stencil reaches outside -- which *is* the Dirichlet condition, since a
    neighbour outside the domain contributes nothing.

    Returns `(eigenvalues, modes)` with `modes` an array of fields shaped like
    the mask, zero outside it.  The matrix is dense because Blender ships no
    scipy, so the resolution is the real cost: n interior samples means an
    n-by-n symmetric eigenproblem.
    """
    m = np.asarray(mask, dtype=bool)
    idx = -np.ones(m.shape, dtype=np.int64)
    pts = np.argwhere(m)
    idx[m] = np.arange(len(pts))
    n = len(pts)
    if n == 0:
        raise ValueError("empty drum domain")

    A = np.zeros((n, n))
    inv = 1.0 / (dx * dx)
    for k, (j, i) in enumerate(pts):
        A[k, k] = 4.0 * inv
        for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = j + dj, i + di
            if 0 <= a < m.shape[0] and 0 <= b < m.shape[1] and m[a, b]:
                A[k, idx[a, b]] = -inv
            # else: the neighbour is outside, u there is 0, nothing to add.

    vals, vecs = np.linalg.eigh(A)
    count = int(min(max(1, count), n))
    out = np.zeros((count,) + m.shape)
    for c in range(count):
        f = np.zeros(m.shape)
        f[m] = vecs[:, c]
        # Sign is arbitrary from an eigensolver; fix it so a mode does not
        # flip between runs and invert the relief.
        if f.sum() < 0:
            f = -f
        out[c] = f
    return vals[:count], out


_CACHE = {}


def modes_cached(shape, solve_res, ratio, count):
    """Eigenmodes, remembered.

    A dense eigensolve is cubic in the interior sample count, so it is far
    too slow to repeat on every autobuild -- and the panel rebuilds on every
    slider drag.  The solve depends only on the domain and the grid, so the
    result is reusable across every change that is not one of those.
    """
    key = (str(shape).upper(), int(solve_res), round(float(ratio), 6),
           int(count))
    hit = _CACHE.get(key)
    if hit is None:
        n = int(max(16, solve_res))
        g = np.linspace(-1.0, 1.0, n)
        GX, GY = np.meshgrid(g, g)
        mask = domain_mask(shape, GX, GY, ratio=ratio)
        hit = membrane_modes(mask, 2.0 / (n - 1), count=count)
        # A handful of domains is all anyone flips between; the cap keeps a
        # long session from holding every grid size ever tried.
        if len(_CACHE) > 12:
            _CACHE.clear()
        _CACHE[key] = hit
    return hit


def drum_field(X, Y, info, shape='TRIANGLE', index=1, ratio=0.6,
               solve_res=48, count=None):
    """One membrane mode of `shape`, sampled onto the panel grid.

    The eigenproblem is solved on its own square grid at `solve_res` and the
    result resampled, for the same reason the reaction-diffusion field keeps
    its own lattice: the cost here is cubic in the sample count, so tying it
    to the panel resolution would make a fine panel unusable.
    """
    k = max(1, int(index))
    n = int(max(16, solve_res))
    vals, modes = modes_cached(shape, n, ratio, max(k, int(count or k)))
    f = modes[k - 1]

    # Nearest-sample resample onto the panel; the mode is smooth inside and
    # zero outside, so bilinear across the boundary would smear it outward.
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    ix = np.clip(((X / hx + 1.0) * 0.5 * (n - 1)).round().astype(int), 0,
                 n - 1)
    iy = np.clip(((Y / hy + 1.0) * 0.5 * (n - 1)).round().astype(int), 0,
                 n - 1)
    out = f[iy, ix]
    s = out.std()
    return out / s if s > 1e-12 else out


def spectrum(shape, solve_res=48, count=10, ratio=0.6):
    """The lowest `count` eigenvalues of a drum shape."""
    n = int(max(16, solve_res))
    g = np.linspace(-1.0, 1.0, n)
    GX, GY = np.meshgrid(g, g)
    dx = 2.0 / (n - 1)
    mask = domain_mask(shape, GX, GY, ratio=ratio)
    vals, _ = membrane_modes(mask, dx, count=count)
    return vals


def _selftest():
    ok = True
    from . import grid as _grid
    from .special import bessel_zero

    # --- the solver, against a spectrum known in closed form -------------
    # A circular drum's eigenvalues are (j_{m,k} / R)^2, and `special.py`
    # already computes Bessel zeros to machine precision -- so this checks
    # the mask, the assembly and the boundary condition against an
    # independent implementation rather than against itself.
    want = sorted([bessel_zero(0, 1) ** 2, bessel_zero(1, 1) ** 2,
                   bessel_zero(1, 1) ** 2, bessel_zero(2, 1) ** 2,
                   bessel_zero(2, 1) ** 2, bessel_zero(0, 2) ** 2])
    got = spectrum('CIRCLE', solve_res=64, count=6)
    err = max(abs(g - w) / w for g, w in zip(got, want))
    print("drums: circle eigenvalues %s" % np.round(got, 3))
    print("drums:      Bessel zeros %s  (worst error %.1f%%)"
          % (np.round(want, 3), 100 * err))
    # A five-point stencil converges as O(h^2); a few percent at this
    # resolution is the discretisation, not a mistake.
    ok = ok and err < 0.05

    # ... and it must converge, which distinguishes discretisation error
    # from a wrong operator.
    errs = []
    for res in (32, 48, 72):
        g2 = spectrum('CIRCLE', solve_res=res, count=1)[0]
        errs.append(abs(g2 - want[0]) / want[0])
    print("drums: circle lowest-mode error %.4f -> %.4f -> %.4f (refining)"
          % tuple(errs))
    ok = ok and errs[-1] < errs[0]

    # --- the equilateral triangle, whose spectrum Lame solved ------------
    # Its eigenvalues are proportional to the Loeschian numbers m^2+mn+n^2
    # for positive integers m, n: 3, 7, 12, 13, 19, 21, ...  The ratios are
    # therefore fixed by the shape alone, with no constant to fit.
    tri = spectrum('TRIANGLE', solve_res=72, count=6)
    ratios = tri / tri[0]
    # WITH multiplicities: (m,n) and (n,m) give the same value, so 7 and 13
    # and 19 each appear twice.  Comparing against the distinct values alone
    # misaligns the sequence after the second mode and reports a 42% error on
    # a correct solver.
    lo = np.array([3.0, 7.0, 7.0, 12.0, 13.0, 13.0]) / 3.0
    print("drums: triangle ratios %s" % np.round(ratios, 3))
    print("drums:   Loeschian     %s" % np.round(lo, 3))
    worst = float(np.max(np.abs(ratios - lo) / lo))
    print("drums: worst triangle ratio error %.1f%%" % (100 * worst))
    ok = ok and worst < 0.06

    # --- Kac's question ---------------------------------------------------
    # The Gordon-Webb-Wolpert polygons are not congruent and their spectra
    # agree.  Solved on the same grid, so the discretisation error is common
    # to both and what remains is the mathematics.
    s1 = spectrum('ISO_ONE', solve_res=72, count=8)
    s2 = spectrum('ISO_TWO', solve_res=72, count=8)
    rel = np.max(np.abs(s1 - s2) / s1)
    print("drums: isospectral pair, first eight eigenvalues")
    print("drums:   drum 1 %s" % np.round(s1, 3))
    print("drums:   drum 2 %s" % np.round(s2, 3))
    print("drums:   worst relative difference %.2f%%" % (100 * rel))
    ok = ok and rel < 0.05

    # They must also be genuinely different SHAPES, or the test above is
    # vacuous -- two copies of one polygon would pass it trivially.
    g = np.linspace(-1.0, 1.0, 97)
    GX, GY = np.meshgrid(g, g)
    m1 = domain_mask('ISO_ONE', GX, GY)
    m2 = domain_mask('ISO_TWO', GX, GY)
    # Equal area is required -- they are built from the same seven pieces --
    # so area alone cannot show they differ.  What must be shown is that NO
    # symmetry of the square carries one onto the other, because a rotated or
    # mirrored copy of one shape is isospectral for trivial reasons and would
    # sail through the spectral test above.  An earlier version of these
    # polygons did exactly that.
    variants = {
        'identity': m2, 'rot90': np.rot90(m2), 'rot180': np.rot90(m2, 2),
        'rot270': np.rot90(m2, 3), 'flip up-down': np.flipud(m2),
        'flip left-right': np.fliplr(m2), 'transpose': m2.T,
        'anti-transpose': np.rot90(m2, 2).T,
    }
    closest = min((int((m1 != v).sum()), k) for k, v in variants.items()
                  if v.shape == m1.shape)
    n_cells = float(m1.size)
    print("drums: equal areas %.4f / %.4f; closest square symmetry is %s, "
          "differing on %.1f%% of the box"
          % (m1.mean(), m2.mean(), closest[1], 100 * closest[0] / n_cells))
    ok = ok and abs(m1.mean() - m2.mean()) < 0.01
    # Not congruent: even the best-matching symmetry must miss substantially.
    ok = ok and closest[0] > 0.05 * n_cells

    # --- fields on a panel ------------------------------------------------
    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=96)
    for shape in SHAPES:
        h = drum_field(X, Y, info, shape=shape, index=2, solve_res=40)
        fin = bool(np.isfinite(h).all())
        print("drums: %-9s mode 2 sd=%.3f finite=%s covered=%.2f"
              % (shape, h.std(), fin, float((np.abs(h) > 1e-9).mean())))
        ok = ok and fin and h.std() > 1e-6

    # A higher mode really is higher: more sign changes across the domain.
    def nodal(h):
        s = np.signbit(h[np.abs(h) > 1e-9])
        return int(np.sum(np.diff(s.astype(int)) != 0))
    lo_m = drum_field(X, Y, info, shape='TRIANGLE', index=1, solve_res=40)
    hi_m = drum_field(X, Y, info, shape='TRIANGLE', index=5, solve_res=40)
    print("drums: triangle nodal crossings, mode 1 = %d, mode 5 = %d"
          % (nodal(lo_m), nodal(hi_m)))
    ok = ok and nodal(hi_m) > nodal(lo_m)

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
