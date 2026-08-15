"""relief.elliptic -- doubly periodic relief from elliptic functions.

An elliptic function is, by definition, meromorphic and doubly periodic.  Map
the panel onto one period parallelogram and the resulting field tiles
*exactly*, in closed form -- no wavevector snapping, no basis projection, no
measured residual.  This is the one pattern family whose seamlessness is a
theorem rather than a construction.

That applies to the Weierstrass P-function and its derivative.  It does NOT
apply to the Weierstrass zeta function or to the Jacobi thetas, which are only
*quasi*-periodic -- zeta gains a constant per period and theta gains a factor
-- so they drift across a cell and do not tile.  They are offered here as
shapes, and `DOUBLY_PERIODIC` names the two that tile.

The Weierstrass P-function, its derivative, and the Jacobi theta functions are
already implemented in this add-on for the minimal-surface work, so this module
is a thin wrapper: `math_art/minsurf/elliptic.py` supplies `_Lattice` with
`wp`, `wp_prime` and `zeta`, all vectorised, with a theta q-series whose terms
decay like q^((n+1/2)^2) -- super-geometrically, so a handful of terms reach
machine precision.

**Poles are the whole difficulty.**  P has a double pole at every lattice
point, so its real part runs to infinity exactly where a relief needs a finite
height.  Clamping produces a flat mesa with a visible rim; taking a logarithm
moves the problem without solving it.  The clean answer is the Riemann-sphere
height

    h = (|f|^2 - 1) / (|f|^2 + 1)

which is the stereographic z-coordinate of f on the sphere: it is smooth
*through* the pole, mapping f -> infinity to +1 and f -> 0 to -1.  A pole
becomes a rounded summit and a zero a rounded pit, and the field is bounded
everywhere without a single special case.

References:
  Carl Gustav Jacob Jacobi, "Fundamenta Nova Theoriae Functionum
    Ellipticarum", Koenigsberg, 1829 -- the theta functions.
  Karl Weierstrass, Berlin lectures (1860s) -- the P-function and its
    differential equation.
  NIST Digital Library of Mathematical Functions, https://dlmf.nist.gov/ --
    ch. 20 (theta functions, the q-series used here), ch. 23 (Weierstrass
    elliptic functions, 23.6 for P and zeta in terms of theta).
"""

import math

import numpy as np

KINDS = ('WP', 'WP_PRIME', 'ZETA', 'THETA')

# Only the first two are genuinely ELLIPTIC, i.e. doubly periodic, and only
# they tile.  The other two are quasi-periodic and are offered as shapes, not
# as seamless ones:
#
#   zeta(z + 2 w) = zeta(z) + 2 eta      (a constant is added per period)
#   theta1(z + pi) = -theta1(z),  theta1(z + pi tau) = -e^(-i(2z+pi tau)) theta1(z)
#
# Both drift across a period, so a panel built from them does not abut itself
# however the cell is cut.  Calling the whole family "doubly periodic" would
# be wrong, and the seam check catches it -- zeta scores x7.7 on the joint.
DOUBLY_PERIODIC = ('WP', 'WP_PRIME')


def _lattice(tau_re=0.0, tau_im=1.0):
    """A lattice with real half-period 1/2 and ratio tau."""
    try:
        from ..minsurf.elliptic import _Lattice
    except ImportError:                     # flat import outside the package
        from minsurf.elliptic import _Lattice
    tau = complex(float(tau_re), max(float(tau_im), 0.05))
    return _Lattice(0.5, tau)


def sphere_height(f):
    """Stereographic height of a complex value: bounded, smooth through poles.

    Maps |f| -> infinity to +1 and f -> 0 to -1, with the unit circle at 0.
    """
    a = np.abs(np.asarray(f))
    a2 = a * a
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (a2 - 1.0) / (a2 + 1.0)
    return np.where(np.isfinite(out), out, 1.0)


def elliptic_field(X, Y, info, kind='WP', tau_re=0.0, tau_im=1.0,
                   cells=1.0, part='SPHERE', phase=0.0):
    """Evaluate an elliptic function over `cells` periods of the panel.

    `part` selects how the complex value becomes a height: SPHERE is the
    stereographic map above (the only one that survives a pole), RE and IM
    take components, ABS the modulus.
    """
    lat = _lattice(tau_re, tau_im)
    # Map the panel onto `cells` copies of the period parallelogram.  The
    # lattice has half-period w1 = 1/2, so a full period is 1 in x and tau
    # in y; sweeping exactly that range is what makes the result tile.
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    u = (X / (2.0 * hx) + 0.5) * float(cells)          # [0, cells)
    v = (Y / (2.0 * hy) + 0.5) * float(cells)
    tau = complex(float(tau_re), max(float(tau_im), 0.05))
    z = (u + v * tau.real) + 1j * (v * tau.imag)
    # Nudge off the lattice points themselves, where P is infinite.
    z = z + (1e-6 + 1e-6j)

    if kind == 'WP':
        f = lat.wp(z)
    elif kind == 'WP_PRIME':
        f = lat.wp_prime(z)
    elif kind == 'ZETA':
        f = lat.zeta(z)
    elif kind == 'THETA':
        try:
            from ..minsurf.elliptic import _theta1_series
        except ImportError:
            from minsurf.elliptic import _theta1_series
        f = _theta1_series(math.pi * z, lat.q)[0]
    else:
        raise ValueError("unknown elliptic kind: %r" % (kind,))

    f = f * np.exp(1j * float(phase))
    if part == 'SPHERE':
        return sphere_height(f)
    if part == 'RE':
        return np.real(f)
    if part == 'IM':
        return np.imag(f)
    if part == 'ABS':
        return np.abs(f)
    raise ValueError("unknown part: %r" % (part,))


def _selftest():
    ok = True
    from . import grid as _grid
    from . import tiling as _tiling

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=128)

    # Every kind evaluates to a finite, bounded, varying field.
    for kind in KINDS:
        h = elliptic_field(X, Y, info, kind=kind)
        finite = np.isfinite(h).all()
        print("elliptic: %-9s range [%.3f, %.3f] finite=%s sd=%.4f"
              % (kind, h.min(), h.max(), finite, h.std()))
        ok = ok and finite and h.std() > 1e-6
        ok = ok and h.min() >= -1.0001 and h.max() <= 1.0001   # SPHERE bounds

    # The point of the family: the doubly periodic ones tile by construction,
    # with no snapping -- and the quasi-periodic ones demonstrably do not.
    for kind in KINDS:
        h = elliptic_field(X, Y, info, kind=kind, cells=1.0)
        chk = _tiling.check(h, 'TORUS', info, 'ELLIPTIC')
        expect = kind in DOUBLY_PERIODIC
        print("elliptic: %-9s torus joint step x%.3f curvature x%.3f ok=%-5s "
              "(doubly periodic: %s)"
              % (kind, chk['step'], chk['curvature'], chk['ok'], expect))
        ok = ok and (chk['ok'] == expect)

    # More cells across the panel is still seamless, and really is more cells.
    # Counting samples above a threshold would NOT show this: the fraction of
    # a period cell lying near a pole is scale-invariant, so that count stays
    # put. Count the field's oscillations along a row instead.
    def crossings(h):
        row = h[h.shape[0] // 3, :]
        return int(np.sum(np.diff(np.signbit(row - row.mean()))))

    h1 = elliptic_field(X, Y, info, kind='WP', cells=1.0)
    h2 = elliptic_field(X, Y, info, kind='WP', cells=2.0)
    chk = _tiling.check(h2, 'TORUS', info, 'ELLIPTIC')
    print("elliptic: 1 cell -> %d oscillations, 2 cells -> %d (seamless: %s)"
          % (crossings(h1), crossings(h2), chk['ok']))
    ok = ok and chk['ok'] and crossings(h2) > crossings(h1)

    # The sphere map really does tame the pole: P itself is unbounded.
    lat = _lattice()
    z = np.array([1e-4 + 1e-4j])
    raw = float(np.abs(lat.wp(z))[0])
    tamed = float(sphere_height(lat.wp(z))[0])
    print("elliptic: near a pole |P| = %.3e, sphere height = %.6f" % (raw, tamed))
    ok = ok and raw > 1e6 and abs(tamed - 1.0) < 1e-6

    # tau controls the cell shape, so the field genuinely changes with it.
    a = elliptic_field(X, Y, info, kind='WP', tau_im=1.0)
    b = elliptic_field(X, Y, info, kind='WP', tau_im=1.6)
    print("elliptic: tau_im 1.0 vs 1.6 rms difference %.4f"
          % float(np.sqrt(((a - b) ** 2).mean())))
    ok = ok and float(np.sqrt(((a - b) ** 2).mean())) > 0.01

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
