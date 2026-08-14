"""relief.tiling -- making a panel abut its own copies invisibly.

Seamless tiling is an *option* here, never a constraint on the default path.
When it is switched on, the mechanism is worth stating plainly, because the
naive route is to bolt "make it tile" onto each pattern separately -- N special
cases and N bugs.

**Each seam behaviour is the boundary condition of a standard basis.**

    basis                     boundary condition        seamless under
    ------------------------  ------------------------  -------------------
    exp(2 pi i m x / L)       periodic                  translation
    cos(m pi x / L)           Neumann, dh/dn = 0        mirror
    sin(m pi x / L)           Dirichlet, h = 0          odd extension

So the way to make a layer tile is not to post-process it, but to *build it in
the matching basis*:

* **Torus** -- snap every wavevector onto the dual lattice
  `2 pi (m/Lx, n/Ly)`.  The relative change in wavelength and direction is
  O(1/(|k| L)), invisible beyond a few periods across the panel.
* **Mirror** -- build from separable cosine products.  Every odd derivative of
  a cosine series vanishes at both edges, and odd derivatives are exactly what
  a reflection negates, so the mirror extension *is* the same analytic
  function: the seam is C-infinity, not merely C1.
* **Antimirror** -- sine products, which vanish at the rim.  These tile under
  *odd* extension, meaning the neighbouring panel is the negative relief:
  alternating relief and intaglio meeting at the base plane.  Mirroring a sine
  series instead leaves a visible kink, which is the trap this table exists to
  prevent.

Two families are genuinely excluded rather than merely awkward:

* A true 5- or 7-fold quasiperiodic field **cannot** be periodic -- that is the
  crystallographic restriction, not a missing feature.  The honest substitute
  is a periodic approximant, obtained by snapping the star of wavevectors to a
  fine lattice.
* Radial ripples do not periodise: summing over lattice images converges
  miserably, and wrapped distance leaves C1 kinks along the midlines between
  images.

References:
  Historical validation of the two non-torus columns: Erwin Hauer's cast screen
    modules are cell-periodic and even about their cell boundaries -- Neumann
    edges, so units join without a crease -- while Norman Carlberg's hypar
    modules carry a checkerboard sign, the antisymmetric extension, joining C0
    with a deliberate crease as the aesthetic.
  John Makhoul, "A fast cosine transform in one and two dimensions", IEEE
    Trans. ASSP 28(1), 1980, 27-34.
  Stephen A. Martucci, "Symmetric convolution and the discrete sine and cosine
    transforms", IEEE Trans. Signal Processing 42(5), 1994, 1038-1051 -- which
    transform corresponds to which boundary symmetry.
  Frank A. Farris, "Creating Symmetry", Princeton University Press, 2015 --
    lattice waves, and why a plane group fixes the panel's lattice.
"""

import math

import numpy as np

MODES = ('NONE', 'TORUS', 'MIRROR', 'ANTIMIRROR')

# Patterns that cannot be made to tile, and why.  Reported rather than
# silently producing a seam.
UNTILEABLE = {
    'RIPPLE': "radial ripples do not periodise (the 2-D Helmholtz Green's "
              "function decays too slowly for an image sum)",
    'OBJECT': "an imprint is bounded by its object; wrap the kernel instead",
}


def periods(info):
    """The tiling periods of a panel.

    The sample grid includes both endpoints, so the panel's *extent* and its
    *period* are not the same thing and confusing them puts every seam test
    one sample pitch out.  Two conventions are possible:

      half-open  period = extent + pitch; the last column is one pitch before
                 the next tile's first, as an FFT assumes.
      shared     period = extent; the last column IS the next tile's first.

    A mesh panel wants the shared convention: neighbouring tiles abut on a
    duplicated row or column, which simply welds.  Everything here therefore
    uses `period = extent`, and a wave snapped to it satisfies h[0] == h[-1]
    and h'[0] == h'[-1] exactly rather than approximately.

    (The FFT paths in `kernels` keep the half-open convention internally --
    that is what a discrete transform means -- which is why kernel wrapping
    is a separate flag from panel tiling.)
    """
    return (info['width'], info['height'])


def snap_torus(wavelength, angle, Lx, Ly):
    """Nearest wavevector on the dual lattice.  Returns `(kx, ky, report)`."""
    lam = max(float(wavelength), 1e-9)
    m = round(Lx * math.cos(angle) / lam)
    n = round(Ly * math.sin(angle) / lam)
    if m == 0 and n == 0:                  # never snap a wave out of existence
        if abs(math.cos(angle)) >= abs(math.sin(angle)):
            m = 1 if math.cos(angle) >= 0 else -1
        else:
            n = 1 if math.sin(angle) >= 0 else -1
    kx = 2.0 * math.pi * m / Lx
    ky = 2.0 * math.pi * n / Ly
    k = math.hypot(kx, ky)
    report = {
        'm': m, 'n': n,
        'wavelength': (2.0 * math.pi / k) if k > 0 else float('inf'),
        'angle': math.atan2(ky, kx),
    }
    return kx, ky, report


def snap_symmetric(wavelength, angle, Lx, Ly, mode):
    """Nearest separable half-wave numbers for a mirror/antimirror basis.

    The basis is cos(m pi x / Lx) cos(n pi y / Ly) (mirror) or the sine
    equivalent (antimirror), so the admissible wavenumbers are pi m / L.
    """
    lam = max(float(wavelength), 1e-9)
    m = max(0, round(2.0 * Lx * abs(math.cos(angle)) / lam))
    n = max(0, round(2.0 * Ly * abs(math.sin(angle)) / lam))
    if mode == 'ANTIMIRROR':               # sin(0 * x) is identically zero
        m = max(1, m)
        n = max(1, n)
    if m == 0 and n == 0:
        m = 1
    return m, n


def symmetric_wave(X, Y, info, m, n, mode):
    """A single separable basis function of the mirror/antimirror basis."""
    Lx, Ly = periods(info)
    u = (X - X.min()) * math.pi / Lx
    v = (Y - Y.min()) * math.pi / Ly
    if mode == 'ANTIMIRROR':
        return np.sin(m * u) * np.sin(n * v)
    return np.cos(m * u) * np.cos(n * v)


def tile(h, nx=2, ny=2, mode='TORUS'):
    """Lay out an nx-by-ny run of the panel.

    Edges are SHARED, so each piece drops its trailing row/column except the
    last -- otherwise every joint would carry a duplicated line of samples and
    look like a defect in the very thing being tested.
    """
    h = np.asarray(h, dtype=float)

    def row(j):
        cols = []
        for i in range(nx):
            t = h
            if mode == 'MIRROR' and i % 2:
                t = t[:, ::-1]
            elif mode == 'ANTIMIRROR' and i % 2:
                t = -t[:, ::-1]
            if mode in ('MIRROR', 'ANTIMIRROR') and j % 2:
                t = (t[::-1, :] if mode == 'MIRROR' else -t[::-1, :])
            cols.append(t if i == nx - 1 else t[:, :-1])
        return np.concatenate(cols, axis=1)

    rows = [row(j) if j == ny - 1 else row(j)[:-1, :] for j in range(ny)]
    return np.concatenate(rows, axis=0)


def seam_error(h, mode, info):
    """How visible the joint is.  Returns `(c0, c1)`, both scale-free.

    Measured on the assembled two-tile run rather than on the panel alone,
    because that is what the eye sees.  Both numbers are ratios against the
    panel's own interior, so ~1 means the joint is indistinguishable from
    ordinary detail and >>1 means a visible line:

      c0  largest step across the joint, over the largest interior step
      c1  largest CURVATURE (second difference) at the joint, over the largest
          interior curvature -- a slope discontinuity is a spike in curvature,
          and it is the slope break, not the value break, that catches the eye
          under raking light

    A one-sided difference at the rim would instead report its own O(dx h'')
    truncation, which is why this measures the assembled joint.
    """
    h = np.asarray(h, dtype=float)
    if mode == 'NONE':
        return 0.0, 0.0
    big = tile(h, 2, 1, mode)
    j = h.shape[1] - 1                       # column index of the joint

    d1 = np.abs(np.diff(big, axis=1))
    step_joint = float(d1[:, j - 1:j + 1].max())
    step_inner = float(d1[:, 2:j - 2].max()) if j > 6 else float(d1.max())

    d2 = np.abs(np.diff(big, n=2, axis=1))
    curv_joint = float(d2[:, max(0, j - 2):j + 2].max())
    curv_inner = float(d2[:, 3:j - 3].max()) if j > 8 else float(d2.max())

    c0 = step_joint / (step_inner + 1e-30)
    c1 = curv_joint / (curv_inner + 1e-30)
    return c0, c1


def corner_tiles(fields, corners, X, Y, quintic=True):
    """Blend four seamless fields by corner colour.

    One seamless panel arrayed 8x8 shows its period at once; the eye finds it
    immediately.  A *set* of panels that share corner colours does not repeat.
    Each tile interpolates four fields with corner weights built from a
    smoothstep, whose derivative vanishes at 0 and 1 -- so two tiles sharing an
    edge's two corner colours join C1, because the cross-seam weight-derivative
    terms drop out.  With C colours there are C**4 distinct tiles.
    """
    a, b, c, d = corners
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    u = (X + hx) / (2.0 * hx)
    v = (Y + hy) / (2.0 * hy)
    if quintic:
        s = lambda t: t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    else:
        s = lambda t: t * t * (3.0 - 2.0 * t)
    su, sv = s(np.clip(u, 0, 1)), s(np.clip(v, 0, 1))
    return ((1 - su) * (1 - sv) * fields[a] + su * (1 - sv) * fields[b]
            + (1 - su) * sv * fields[c] + su * sv * fields[d])


def _selftest():
    ok = True
    from . import grid as _grid
    from . import fields as _fields

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=128)
    Lx, Ly = periods(info)

    # --- TORUS: snapped waves are periodic to machine precision ---------
    worst_c0 = worst_c1 = 0.0
    for lam, ang in ((0.5, 0.0), (0.37, 0.7), (0.8, -1.1), (0.25, 2.3)):
        kx, ky, rep = snap_torus(lam, ang, Lx, Ly)
        h = np.sin(kx * (X - X.min()) + ky * (Y - Y.min()))
        c0, c1 = seam_error(h, 'TORUS', info)
        worst_c0 = max(worst_c0, c0)
        worst_c1 = max(worst_c1, c1)
    print("tiling: TORUS snapped waves  joint step x%.3f  curvature x%.3f "
          "(1.0 = indistinguishable)" % (worst_c0, worst_c1))
    ok = ok and worst_c0 < 1.05 and worst_c1 < 1.05

    # An unsnapped wave is NOT seamless -- the test can tell the difference.
    k = 2.0 * math.pi / 0.37
    bad = np.sin(k * (X - X.min()))
    c0b, c1b = seam_error(bad, 'TORUS', info)
    print("tiling: unsnapped wave       joint step x%.3f  curvature x%.3f "
          "(should be >> 1)" % (c0b, c1b))
    ok = ok and max(c0b, c1b) > 2.0

    # Snapping barely moves the wave.
    _kx, _ky, rep = snap_torus(0.37, 0.7, Lx, Ly)
    print("tiling: snap 0.370 -> %.4f m, angle 0.700 -> %.4f rad"
          % (rep['wavelength'], rep['angle']))
    ok = ok and abs(rep['wavelength'] - 0.37) < 0.05

    # --- MIRROR: cosine products have zero rim slope --------------------
    worst = 0.0
    for lam, ang in ((0.5, 0.3), (0.29, 1.0), (0.65, -0.4)):
        m, n = snap_symmetric(lam, ang, Lx, Ly, 'MIRROR')
        h = symmetric_wave(X, Y, info, m, n, 'MIRROR')
        c0, c1 = seam_error(h, 'MIRROR', info)
        worst = max(worst, c1)
    print("tiling: MIRROR cosine modes  joint curvature x%.3f" % worst)
    ok = ok and worst < 1.05

    # --- ANTIMIRROR: sine products vanish at the rim --------------------
    worst = 0.0
    for lam, ang in ((0.5, 0.3), (0.29, 1.0)):
        m, n = snap_symmetric(lam, ang, Lx, Ly, 'ANTIMIRROR')
        h = symmetric_wave(X, Y, info, m, n, 'ANTIMIRROR')
        c0, c1 = seam_error(h, 'ANTIMIRROR', info)
        worst = max(worst, max(c0, c1))
    print("tiling: ANTIMIRROR sine modes joint x%.3f" % worst)
    ok = ok and worst < 1.05

    # A sine series MIRRORED has a kink -- the trap the table warns about.
    m, n = snap_symmetric(0.5, 0.3, Lx, Ly, 'ANTIMIRROR')
    hs = symmetric_wave(X, Y, info, m, n, 'ANTIMIRROR')
    _c0, c1_wrong = seam_error(hs, 'MIRROR', info)
    print("tiling: sine field mirrored  joint curvature x%.1f (a kink -- this "
          "is the trap)" % c1_wrong)
    ok = ok and c1_wrong > 2.0

    # --- the assembled tiling is actually continuous --------------------
    kx, ky, _ = snap_torus(0.4, 0.6, Lx, Ly)
    h = np.sin(kx * (X - X.min()) + ky * (Y - Y.min()))
    big = tile(h, 2, 2, 'TORUS')
    col = big.shape[1] // 2
    jump = float(np.abs(big[:, col] - big[:, col - 1]).max())
    inner = float(np.abs(np.diff(big, axis=1)).max())
    print("tiling: 2x2 torus layout, joint step %.2e vs interior step %.2e"
          % (jump, inner))
    ok = ok and jump <= inner * 1.5

    m, n = snap_symmetric(0.4, 0.6, Lx, Ly, 'MIRROR')
    hm = symmetric_wave(X, Y, info, m, n, 'MIRROR')
    bigm = tile(hm, 2, 2, 'MIRROR')
    col = bigm.shape[1] // 2
    jump = float(np.abs(bigm[:, col] - bigm[:, col - 1]).max())
    inner = float(np.abs(np.diff(bigm, axis=1)).max())
    print("tiling: 2x2 mirror layout, joint step %.2e vs interior step %.2e"
          % (jump, inner))
    ok = ok and jump <= inner * 1.5

    # --- corner tiles join C1 -------------------------------------------
    flds = []
    for s in range(4):
        kx, ky, _ = snap_torus(0.3 + 0.05 * s, 0.4 * s, Lx, Ly)
        flds.append(np.sin(kx * (X - X.min()) + ky * (Y - Y.min()) + s))
    t1 = corner_tiles(flds, (0, 1, 2, 3), X, Y)
    t2 = corner_tiles(flds, (1, 0, 3, 2), X, Y)      # shares the seam colours
    joined = np.concatenate([t1[:, :-1], t2], axis=1)
    j = t1.shape[1] - 1
    d2 = np.abs(np.diff(joined, n=2, axis=1))
    ratio = float(d2[:, j - 2:j + 2].max()) / float(d2[:, 3:j - 3].max())
    c0 = float(np.abs(t1[:, -1] - t2[:, 0]).max())
    print("tiling: corner tiles  shared edge %.2e, joint curvature x%.3f"
          % (c0, ratio))
    ok = ok and c0 < 1e-12 and ratio < 1.5

    # Untileable patterns are named, not silently seamed.
    print("tiling: reported untileable: %s" % sorted(UNTILEABLE))
    ok = ok and 'RIPPLE' in UNTILEABLE

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
