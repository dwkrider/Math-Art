"""relief.kernels -- splatting kernels, point processes, distance transforms.

The engine behind "take an object and apply a kernel at each of its points".

The construction generalises further than it first appears.  Sparse convolution
noise is *defined* as a kernel convolved with a Poisson impulse process --

    N(x) = [ h * sum_i w_i delta_{p_i} ](x)   =   sum_i w_i h(x - p_i)

-- which is the same expression as splatting an object's points.  So one
`splat()` serves both: swap the point source and an object imprint becomes a
noise field.  With a Gabor kernel the same code emits oriented, band-limited
noise, because the noise spectrum is the kernel's spectrum scaled by a
constant.

Two engineering points decide the implementation:

* **Cost.** The direct sum is O(grid * points) -- a 512^2 panel against a
  10k-vertex object is 2.6e9 kernel evaluations, hopeless in Python.  So the
  default path rasterises the points and convolves by FFT, which costs
  O(N^2 log N) *independent of both point count and kernel radius*.  Direct
  summation is kept only for small sigma, where the sampled kernel would alias.
* **Edge continuity.** At its support boundary a C0 kernel puts a slope
  discontinuity around every splat, which reads as a crease ring under raking
  light.  That is a defect for drapery and arguably a feature for coin-like
  relief, so the table records the continuity and lets it be chosen.

References:
  J. P. Lewis, "Algorithms for Solid Noise Synthesis", Computer Graphics 23(3)
    (SIGGRAPH 1989) -- sparse convolution noise; the identity above.
  Jarke J. van Wijk, "Spot Noise -- Texture Synthesis for Data Visualization",
    Computer Graphics 25(4) (SIGGRAPH 1991), 309-318 -- data-shaped spots.
  Ares Lagae, Sylvain Lefebvre, George Drettakis and Philip Dutre, "Procedural
    Noise using Sparse Gabor Convolution", ACM TOG 28(3) (SIGGRAPH 2009),
    art. 54 -- the Gabor kernel, and the analytic variance used to normalise
    a splat layer without measuring it.
  James F. Blinn, "A Generalization of Algebraic Surface Drawing", ACM TOG
    1(3), 1982, 235-256 -- the exponential blobby model.
  Geoff Wyvill, Craig McPheeters and Brian Wyvill, "Data structure for soft
    objects", The Visual Computer 2(4), 1986, 227-234 -- the degree-6
    polynomial kernel below, designed for C(0)=1, C(R/2)=1/2, C(R)=C'(R)=0
    and no square root.
  Holger Wendland, "Piecewise polynomial, positive definite and compactly
    supported radial functions of minimal degree", Advances in Computational
    Mathematics 4, 1995, 389-396.
  Donald Shepard, "A two-dimensional interpolation function for
    irregularly-spaced data", Proc. 23rd ACM National Conference, 1968,
    517-524; Richard Franke and Gregory Nielson, "Smooth interpolation of
    large sets of scattered data", Int. J. Numer. Methods Eng. 15, 1980,
    1691-1704 -- the local weight that cures Shepard's flat spots.
  Pedro F. Felzenszwalb and Daniel P. Huttenlocher, "Distance Transforms of
    Sampled Functions", Theory of Computing 8, 2012, 415-428 -- the exact
    separable transform used for engraving (and, with a paraboloid, for
    tool-reachability later).
  Robert Bridson, "Fast Poisson Disk Sampling in Arbitrary Dimensions",
    SIGGRAPH 2007 sketches -- the blue-noise point process.
  Murray Rosenblatt (1956); Emanuel Parzen (1962); Bernard W. Silverman,
    "Density Estimation for Statistics and Data Analysis", Chapman and Hall,
    1986 -- kernel density estimation and the bandwidth rule of thumb.
"""

import math

import numpy as np

# id -> (label, support ('compact'/'inf'), continuity at the support edge, fn)
# Every kernel is normalised to K(0) = 1 and takes r already divided by sigma.


def _k_gaussian(r):
    return np.exp(-0.5 * r * r)


def _k_blinn(r, b=4.0):
    return np.exp(-b * r * r)


def _k_wyvill(r):
    """Wyvill's degree-6 soft-object polynomial (a cubic in r^2).

    -(4/9) r^6 + (17/9) r^4 - (22/9) r^2 + 1, equivalently
    (r^2 - 1)^2 (9 - 4 r^2) / 9.
    """
    r2 = np.clip(r * r, 0.0, 1.0)
    return (-(4.0 / 9.0) * r2 ** 3 + (17.0 / 9.0) * r2 * r2
            - (22.0 / 9.0) * r2 + 1.0) * (r <= 1.0)


def _k_triweight(r):
    return np.clip(1.0 - r * r, 0.0, None) ** 3 * (r <= 1.0)


def _k_wendland(r):
    """Wendland phi_{3,1}: (1-r)_+^4 (4r + 1).  C2, strictly positive definite."""
    t = np.clip(1.0 - r, 0.0, None)
    return t ** 4 * (4.0 * r + 1.0) * (r <= 1.0)


def _k_cosine(r):
    return 0.5 * (1.0 + np.cos(math.pi * np.clip(r, 0.0, 1.0))) * (r <= 1.0)


def _k_epanechnikov(r):
    return np.clip(1.0 - r * r, 0.0, None) * (r <= 1.0)


def _k_cone(r):
    return np.clip(1.0 - r, 0.0, None)


def _k_cauchy(r):
    return 1.0 / (1.0 + r * r)


KERNELS = {
    'GAUSSIAN':     ("Gaussian", 'inf', 'C-inf', _k_gaussian),
    'BLINN':        ("Blinn blobby", 'inf', 'C-inf', _k_blinn),
    'WYVILL':       ("Wyvill soft object", 'compact', 'C1', _k_wyvill),
    'TRIWEIGHT':    ("Triweight", 'compact', 'C2', _k_triweight),
    'WENDLAND':     ("Wendland", 'compact', 'C2', _k_wendland),
    'COSINE':       ("Cosine bump", 'compact', 'C1', _k_cosine),
    'EPANECHNIKOV': ("Epanechnikov", 'compact', 'C0', _k_epanechnikov),
    'CONE':         ("Cone", 'compact', 'C0', _k_cone),
    'CAUCHY':       ("Cauchy", 'inf', 'C-inf', _k_cauchy),
}

# Radius (in units of sigma) beyond which a kernel is treated as zero.
_TRUNCATION = {'compact': 1.0, 'inf': 4.0}


def kernel_values(kind, r):
    """Evaluate a registered kernel at radius r (already divided by sigma)."""
    try:
        return KERNELS[kind][3](np.asarray(r, dtype=float))
    except KeyError:
        raise ValueError("unknown kernel: %r" % (kind,))


def auto_sigma(points, area, n=None):
    """A defensible default bandwidth.

    In two dimensions Scott's and Silverman's rules of thumb coincide exactly
    (the factor (d+2)/4 is 1 at d = 2), leaving

        sigma = sigma_hat * N^(-1/6)

    with sigma_hat the per-axis spread of the projected points.  Falls back to
    the mean point spacing sqrt(area/N) when the points are degenerate.
    """
    p = np.asarray(points, dtype=float)
    n = int(n or len(p))
    if n < 2:
        return math.sqrt(max(area, 1e-12)) * 0.1
    sd = float(np.sqrt(0.5 * (p[:, 0].std() ** 2 + p[:, 1].std() ** 2)))
    if sd < 1e-9:
        return math.sqrt(max(area, 1e-12) / n)
    return sd * n ** (-1.0 / 6.0)


# ------------------------------------------------------------------
# Rasterise + convolve
# ------------------------------------------------------------------

def rasterise(points, weights, X, Y):
    """Bilinearly accumulate weighted points onto the sample grid."""
    ny, nx = X.shape
    x0, y0 = float(X[0, 0]), float(Y[0, 0])
    dx = float(X[0, 1] - X[0, 0]) if nx > 1 else 1.0
    dy = float(Y[1, 0] - Y[0, 0]) if ny > 1 else 1.0
    grid = np.zeros((ny, nx))
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return grid
    w = (np.ones(len(p)) if weights is None
         else np.asarray(weights, dtype=float))
    gx = (p[:, 0] - x0) / dx
    gy = (p[:, 1] - y0) / dy
    i0 = np.floor(gx).astype(int)
    j0 = np.floor(gy).astype(int)
    fx = gx - i0
    fy = gy - j0
    for di in (0, 1):
        for dj in (0, 1):
            ii = i0 + di
            jj = j0 + dj
            ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            wt = ((fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy) * w)
            np.add.at(grid, (jj[ok], ii[ok]), wt[ok])
    return grid


def _kernel_image(kind, sigma, X, Y, wrap):
    """The kernel sampled on the grid, centred for circular convolution."""
    ny, nx = X.shape
    dx = float(X[0, 1] - X[0, 0]) if nx > 1 else 1.0
    dy = float(Y[1, 0] - Y[0, 0]) if ny > 1 else 1.0
    ix = np.fft.fftfreq(nx, d=1.0 / nx) * dx
    iy = np.fft.fftfreq(ny, d=1.0 / ny) * dy
    KXm, KYm = np.meshgrid(ix, iy)
    r = np.hypot(KXm, KYm) / max(float(sigma), 1e-12)
    return kernel_values(kind, r)


def splat(points, weights, X, Y, kernel='GAUSSIAN', sigma=0.1, wrap=False,
          exact=False):
    """Sum a kernel placed at every point.

    The default path rasterises and convolves by FFT: cost independent of the
    number of points and of sigma.  `exact=True` evaluates the direct sum,
    which is only worth it for sigma below about one cell, where the sampled
    kernel aliases.

    `wrap=True` keeps the circular convolution (points near an edge splat
    across it, which is the seamless-tiling semantics); `wrap=False` pads so
    that content cannot bleed around the panel.
    """
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return np.zeros(X.shape)
    w = (np.ones(len(p)) if weights is None
         else np.asarray(weights, dtype=float))

    if exact:
        out = np.zeros(X.shape)
        cut = _TRUNCATION[KERNELS[kernel][1]] * sigma
        for (px, py), wi in zip(p[:, :2], w):
            dxg = X - px
            dyg = Y - py
            r2 = dxg * dxg + dyg * dyg
            near = r2 <= cut * cut
            if not near.any():
                continue
            out[near] += wi * kernel_values(kernel,
                                            np.sqrt(r2[near]) / sigma)
        return out

    if wrap:
        src = rasterise(p, w, X, Y)
        ker = _kernel_image(kernel, sigma, X, Y, wrap)
        return np.real(np.fft.ifft2(np.fft.fft2(src) * np.fft.fft2(ker)))

    # Zero-pad by the kernel support so the convolution cannot wrap.
    ny, nx = X.shape
    dx = float(X[0, 1] - X[0, 0]) if nx > 1 else 1.0
    pad = int(math.ceil(_TRUNCATION[KERNELS[kernel][1]] * sigma / dx)) + 1
    pad = min(pad, max(nx, ny))
    src = rasterise(p, w, X, Y)
    big = np.zeros((ny + 2 * pad, nx + 2 * pad))
    big[pad:pad + ny, pad:pad + nx] = src
    iy = np.fft.fftfreq(big.shape[0], d=1.0 / big.shape[0]) * dx
    ix = np.fft.fftfreq(big.shape[1], d=1.0 / big.shape[1]) * dx
    KXm, KYm = np.meshgrid(ix, iy)
    ker = kernel_values(kernel, np.hypot(KXm, KYm) / max(float(sigma), 1e-12))
    out = np.real(np.fft.ifft2(np.fft.fft2(big) * np.fft.fft2(ker)))
    return out[pad:pad + ny, pad:pad + nx]


def splat_max(points, weights, X, Y, kernel='WYVILL', sigma=0.1, smooth=0.0):
    """Union of bumps rather than their sum.

    SUM bulges wherever supports overlap (that is density semantics); MAX
    gives a clean union but leaves a C1 crease along the equidistance seam.
    `smooth` > 0 rounds only that seam, which is what "merged like metaballs"
    actually means.

    This must happen INSIDE the layer, on the raw non-negative splat, not as a
    stack blend mode -- once a layer has been normalised to zero mean, a
    maximum against another layer is meaningless.
    """
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return np.zeros(X.shape)
    w = (np.ones(len(p)) if weights is None
         else np.asarray(weights, dtype=float))
    out = np.full(X.shape, -np.inf)
    cut = _TRUNCATION[KERNELS[kernel][1]] * sigma
    for (px, py), wi in zip(p[:, :2], w):
        r = np.hypot(X - px, Y - py)
        v = np.where(r <= cut, wi * kernel_values(kernel, r / sigma), 0.0)
        if smooth <= 0.0:
            out = np.maximum(out, v)
        else:
            k = float(smooth)
            a, b = out, v
            a = np.where(np.isneginf(a), b - 1e9, a)
            h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
            out = b * h + a * (1.0 - h) + k * h * (1.0 - h)
    return np.where(np.isneginf(out), 0.0, out)


def shepard(points, values, X, Y, power=2.0, radius=None):
    """Inverse-distance-weighted interpolation, with local support.

    Plain Shepard has two well-known faults: with power > 1 the gradient
    vanishes at every data point (flat spots, the bull's-eye look), and far
    from the data the surface tends to the global mean.  A finite `radius`
    applies the Franke-Nielson local weight, which cures the second and
    softens the first.  It drapes a surface THROUGH the points rather than
    bumping outward from them.
    """
    p = np.asarray(points, dtype=float)
    v = np.asarray(values, dtype=float)
    if p.size == 0:
        return np.zeros(X.shape)
    num = np.zeros(X.shape)
    den = np.zeros(X.shape)
    for (px, py), vi in zip(p[:, :2], v):
        d = np.hypot(X - px, Y - py)
        np.maximum(d, 1e-9, out=d)
        if radius:
            R = float(radius)
            w = np.clip((R - d) / (R * d), 0.0, None) ** 2
        else:
            w = d ** (-float(power))
        num += w * vi
        den += w
    return np.where(den > 0.0, num / np.maximum(den, 1e-300), 0.0)


# ------------------------------------------------------------------
# Exact Euclidean distance transform (Felzenszwalb-Huttenlocher)
# ------------------------------------------------------------------

def _edt_1d(f):
    """Lower envelope of parabolas: min_q ( f[q] + (p-q)^2 ), per row."""
    n = f.shape[-1]
    out = np.empty_like(f)
    for row in range(f.shape[0]):
        fr = f[row]
        v = np.zeros(n, dtype=int)
        z = np.empty(n + 1)
        z[0], z[1] = -np.inf, np.inf
        k = 0
        for q in range(1, n):
            s = ((fr[q] + q * q) - (fr[v[k]] + v[k] * v[k])) / (2.0 * q
                                                                - 2.0 * v[k])
            while s <= z[k]:
                k -= 1
                s = ((fr[q] + q * q) - (fr[v[k]] + v[k] * v[k])) \
                    / (2.0 * q - 2.0 * v[k])
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = np.inf
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            out[row, q] = (q - v[k]) ** 2 + fr[v[k]]
    return out


def distance_transform(mask, dx=1.0, dy=1.0, wrap=False):
    """Exact Euclidean distance from every sample to the nearest True cell.

    Separable and exact.  With `wrap` the domain is a torus: the transform is
    run on a 3x tiling and the middle third kept, because the wrapped distance
    never needs a shift of more than one period.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.full(m.shape, np.inf)
    ny, nx = m.shape
    if wrap:
        m = np.tile(m, (3, 3))
    big = np.where(m, 0.0, 1e12)
    d = _edt_1d(big)                       # along rows (x)
    d = _edt_1d(d.T).T                     # along columns (y)
    if wrap:
        d = d[ny:2 * ny, nx:2 * nx]
    # Cells are square by construction, so one pitch suffices.
    return np.sqrt(d) * float(dx)


# ------------------------------------------------------------------
# Point processes (the same splat engine, different point source)
# ------------------------------------------------------------------

def poisson_disk(width, height, radius, seed=1, k=30):
    """Bridson's O(N) blue-noise sampler.

    Background grid of cell r/sqrt(2), an active list, and k candidate points
    drawn from the annulus [r, 2r).  Blue noise matters here because the
    low-frequency energy of a uniform random point set survives the kernel's
    low-pass and reads as clumping.
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    r = float(radius)
    cell = r / math.sqrt(2.0)
    gw = max(1, int(math.ceil(width / cell)))
    gh = max(1, int(math.ceil(height / cell)))
    grid = -np.ones((gh, gw), dtype=int)
    pts = []
    active = []

    def emit(pt):
        pts.append(pt)
        i = min(gw - 1, int(pt[0] / cell))
        j = min(gh - 1, int(pt[1] / cell))
        grid[j, i] = len(pts) - 1
        active.append(len(pts) - 1)

    emit((rng.uniform(0, width), rng.uniform(0, height)))
    while active:
        idx = active[rng.integers(0, len(active))]
        base = pts[idx]
        placed = False
        for _ in range(k):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            rad = r * math.sqrt(rng.uniform(1.0, 4.0))
            cand = (base[0] + rad * math.cos(ang),
                    base[1] + rad * math.sin(ang))
            if not (0.0 <= cand[0] < width and 0.0 <= cand[1] < height):
                continue
            ci = min(gw - 1, int(cand[0] / cell))
            cj = min(gh - 1, int(cand[1] / cell))
            good = True
            for jj in range(max(0, cj - 2), min(gh, cj + 3)):
                for ii in range(max(0, ci - 2), min(gw, ci + 3)):
                    n = grid[jj, ii]
                    if n >= 0 and math.hypot(pts[n][0] - cand[0],
                                             pts[n][1] - cand[1]) < r:
                        good = False
                        break
                if not good:
                    break
            if good:
                emit(cand)
                placed = True
                break
        if not placed:
            active.remove(idx)
    return np.array(pts)


def halton(n, seed=0):
    """`n` points of the 2-D Halton sequence (bases 2 and 3), in [0,1)^2."""
    def radical(i, base):
        f, out = 1.0, 0.0
        while i > 0:
            f /= base
            out += f * (i % base)
            i //= base
        return out
    return np.array([[radical(i + 1 + seed, 2), radical(i + 1 + seed, 3)]
                     for i in range(int(n))])


def gabor_noise(X, Y, info, count=400, freq=8.0, bandwidth=0.3, angle=0.0,
                spread=0.0, seed=1, wrap=True):
    """Sparse convolution noise with a Gabor kernel.

    A Gabor kernel is a Gaussian envelope times a cosine carrier, so its
    spectrum is a Gaussian *centred on the carrier frequency* rather than at
    zero.  Scattering such kernels gives noise whose energy sits in a chosen
    band at a chosen orientation -- the one noise here that can be told what
    frequency to occupy, instead of being filtered afterwards and hoped for.

    This is the same machinery as the object layer: sparse convolution noise
    is object splatting with the object swapped for a kernel (Lewis 1989), so
    the drapery-like fibrous look and the imprint path share one
    implementation.

    `spread` 0 makes every kernel share `angle`, giving a strongly directional
    weave; increasing it randomises the orientations toward isotropy.

    `bandwidth` is the envelope width as a fraction of the carrier frequency,
    and it has to stay well under 1 for the noise to be band-limited in any
    useful sense: at 1.4 the Gaussian is as wide as the frequency it is
    centred on, and the energy peak lands at 8 cycles when 12 were asked for.
    At 0.3 the peak sits exactly where requested, at every frequency tested.

    References:
      J. P. Lewis, "Algorithms for Solid Noise Synthesis", SIGGRAPH 1989,
        263-270 -- sparse convolution noise.
      Ares Lagae, Sylvain Lefebvre, George Drettakis and Philip Dutre,
        "Procedural Noise using Sparse Gabor Convolution", ACM TOG 28(3)
        (SIGGRAPH 2009), art. 54 -- the Gabor kernel, its band-limited
        spectrum, and the setting of its bandwidth.
    """
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n = max(1, int(count))

    px = rng.uniform(-hx, hx, n)
    py = rng.uniform(-hy, hy, n)
    ph = rng.uniform(0.0, 2.0 * math.pi, n)
    w = rng.normal(size=n)
    th = float(angle) + rng.uniform(-float(spread), float(spread), n)

    F = float(freq) / (2.0 * hx)             # cycles per unit length
    a = float(bandwidth) * F                 # envelope width, in Lagae's terms
    # Truncate at three envelope widths: beyond that the Gaussian contributes
    # less than 1e-4 of its peak, and evaluating the whole panel per kernel
    # would cost far more than the result differs by.
    rad = 3.0 / max(a, 1e-9)

    out = np.zeros(X.shape)
    Lx, Ly = 2.0 * hx, 2.0 * hy
    for i in range(n):
        dx = X - px[i]
        dy = Y - py[i]
        if wrap:
            dx = dx - Lx * np.round(dx / Lx)
            dy = dy - Ly * np.round(dy / Ly)
        r2 = dx * dx + dy * dy
        near = r2 < rad * rad
        if not near.any():
            continue
        env = np.exp(-math.pi * a * a * r2)
        carrier = np.cos(2.0 * math.pi * F
                         * (dx * math.cos(th[i]) + dy * math.sin(th[i]))
                         + ph[i])
        out += w[i] * np.where(near, env * carrier, 0.0)
    sd = out.std()
    return out / sd if sd > 1e-12 else out


def _selftest():
    ok = True
    from . import grid as _grid

    # Every kernel: K(0) = 1, compact ones exactly zero past their support.
    for kind, (label, support, cont, _fn) in sorted(KERNELS.items()):
        k0 = float(kernel_values(kind, np.array([0.0]))[0])
        beyond = float(np.abs(kernel_values(
            kind, np.linspace(1.0001, 3.0, 50))).max())
        ok = ok and abs(k0 - 1.0) < 1e-12
        if support == 'compact':
            ok = ok and beyond == 0.0
    print("kernels: %d kernels, K(0)=1, compact ones vanish past support"
          % len(KERNELS))

    # The claimed edge continuity is real: the jump in K' at r=1 is zero for
    # C1+ kernels and non-zero for the C0 ones.
    h = 1e-6
    for kind in ('WYVILL', 'WENDLAND', 'COSINE', 'TRIWEIGHT',
                 'EPANECHNIKOV', 'CONE'):
        d_in = float((kernel_values(kind, np.array([1.0]))[0]
                      - kernel_values(kind, np.array([1.0 - h]))[0]) / h)
        cont = KERNELS[kind][2]
        if cont == 'C0':
            ok = ok and abs(d_in) > 0.5
        else:
            ok = ok and abs(d_in) < 1e-3
    print("kernels: edge-continuity claims verified (C0 kernels really do "
          "kink; C1/C2 do not)")

    # Wyvill's design constraints, from the 1986 paper.
    w = kernel_values('WYVILL', np.array([0.0, 0.5, 1.0]))
    print("kernels: Wyvill C(0)=%.6f C(1/2)=%.6f C(1)=%.6f" % tuple(w))
    ok = ok and abs(w[0] - 1.0) < 1e-12 and abs(w[1] - 0.5) < 1e-12
    ok = ok and abs(w[2]) < 1e-12

    # --- the gate: FFT convolution == direct sum -----------------------
    # This only holds with points ON grid nodes (bilinear splatting of an
    # off-node point convolves the point set with a tent) and with the direct
    # sum computed periodically, matching the circular convolution.
    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=64)
    rng = np.random.default_rng(3)
    ij = rng.integers(0, 64, size=(8, 2))
    pts = np.stack([X[ij[:, 0], ij[:, 1]], Y[ij[:, 0], ij[:, 1]]], axis=1)
    wts = rng.uniform(0.5, 1.5, size=8)
    sig = 0.18

    fftv = splat(pts, wts, X, Y, 'GAUSSIAN', sig, wrap=True)
    # Periodic direct sum: nearest-image offsets in both axes.
    Lx = info['width'] + info['dx']
    Ly = info['height'] + info['dy']
    direct = np.zeros(X.shape)
    for (px, py), wi in zip(pts, wts):
        ddx = X - px
        ddy = Y - py
        ddx -= Lx * np.round(ddx / Lx)
        ddy -= Ly * np.round(ddy / Ly)
        direct += wi * _k_gaussian(np.hypot(ddx, ddy) / sig)
    rel = float(np.abs(fftv - direct).max() / np.abs(direct).max())
    print("kernels: FFT vs periodic direct sum, on-node points: rel err %.2e"
          % rel)
    ok = ok and rel < 1e-10

    # Off-node points differ from the exact sum only by the bilinear tent.
    # Both sides must be NON-periodic here: the exact path computes plain
    # distances and truncates at 4 sigma, so comparing it against a wrapped
    # FFT measures the wrap, not the tent.
    pts_off = pts + 0.5 * info['dx']
    a = splat(pts_off, wts, X, Y, 'GAUSSIAN', sig, wrap=False)
    b = splat(pts_off, wts, X, Y, 'GAUSSIAN', sig, wrap=False, exact=True)
    rel_off = float(np.abs(a - b).max() / np.abs(b).max())
    print("kernels: off-node FFT vs exact (both unwrapped): rel err %.2e "
          "-- the bilinear tent, at sigma = %.1f cells"
          % (rel_off, sig / info['dx']))
    ok = ok and rel_off < 0.02

    # The tent's effect shrinks as sigma grows past the cell pitch, which is
    # why the exact path is only needed below about one cell.
    errs = []
    for s_cells in (1.0, 2.0, 4.0, 8.0):
        ss = s_cells * info['dx']
        aa = splat(pts_off, wts, X, Y, 'GAUSSIAN', ss, wrap=False)
        bb = splat(pts_off, wts, X, Y, 'GAUSSIAN', ss, wrap=False, exact=True)
        errs.append(float(np.abs(aa - bb).max() / np.abs(bb).max()))
    print("kernels: tent error vs sigma in cells [1,2,4,8] = %s"
          % ["%.3f" % e for e in errs])
    ok = ok and errs[-1] < errs[0]

    # Auto bandwidth is sane.
    s = auto_sigma(pts, area=4.0)
    print("kernels: auto sigma for 8 points over 4 m^2 = %.4f" % s)
    ok = ok and 0.01 < s < 2.0

    # MAX blending really is a union (never exceeds the largest single bump).
    one = splat_max(pts[:1], wts[:1], X, Y, 'WYVILL', 0.3)
    allp = splat_max(pts, wts, X, Y, 'WYVILL', 0.3)
    summed = splat(pts, wts, X, Y, 'WYVILL', 0.3)
    print("kernels: peak  single=%.3f  max=%.3f  sum=%.3f (sum bulges)"
          % (one.max(), allp.max(), summed.max()))
    ok = ok and allp.max() <= max(wts) + 1e-9 and summed.max() > allp.max()

    # Distance transform against brute force.
    m = np.zeros((24, 24), dtype=bool)
    m[5, 7] = m[18, 3] = m[11, 20] = True
    d = distance_transform(m, dx=1.0, dy=1.0)
    jj, ii = np.nonzero(m)
    gy, gx = np.mgrid[0:24, 0:24]
    brute = np.min(np.stack([np.hypot(gx - x, gy - y)
                             for y, x in zip(jj, ii)]), axis=0)
    err = float(np.abs(d - brute).max())
    print("kernels: EDT vs brute force max err = %.2e" % err)
    ok = ok and err < 1e-9

    # ...and the wrapped variant is never longer than the unwrapped one.
    dw = distance_transform(m, dx=1.0, dy=1.0, wrap=True)
    print("kernels: wrapped EDT max %.3f <= unwrapped max %.3f"
          % (dw.max(), d.max()))
    ok = ok and dw.max() <= d.max() + 1e-9

    # Shepard reproduces its data values at the sample points.
    sp = np.array([[-0.5, -0.5], [0.4, 0.2], [0.0, 0.7]])
    sv = np.array([1.0, -0.5, 0.25])
    f = shepard(sp, sv, X, Y, power=2.0)
    got = []
    for (px, py), vi in zip(sp, sv):
        k = np.argmin((X - px) ** 2 + (Y - py) ** 2)
        got.append(abs(f.ravel()[k] - vi))
    print("kernels: Shepard reproduces data values to %.3f" % max(got))
    ok = ok and max(got) < 0.15

    # Point processes.
    pd = poisson_disk(2.0, 2.0, 0.16, seed=2)
    dmin = min(math.hypot(*(pd[i] - pd[j]))
               for i in range(len(pd)) for j in range(i + 1, len(pd)))
    print("kernels: Poisson-disk %d points, min spacing %.4f (>= r = 0.16)"
          % (len(pd), dmin))
    ok = ok and len(pd) > 20 and dmin >= 0.16 - 1e-9

    hl = halton(64)
    ok = ok and hl.shape == (64, 2) and hl.min() >= 0.0 and hl.max() < 1.0

    # Gabor noise puts its energy where it is told to.  That is the whole
    # claim of the kernel -- a Gaussian envelope times a carrier has a
    # spectrum centred ON the carrier -- so it is what gets measured.
    Xg, Yg, ig = _grid.make_grid(width=2.0, aspect=1.0, resolution=192)

    def spectral_peak(h):
        P = np.abs(np.fft.fftshift(np.fft.fft2(h))) ** 2
        n = h.shape[0]
        c = n // 2
        ky, kx = np.mgrid[-c:n - c, -c:n - c]
        kr = np.hypot(kx, ky)
        prof = [P[(kr >= r) & (kr < r + 1)].mean() for r in range(1, c)]
        return int(np.argmax(prof)) + 1

    for want in (6, 12, 20):
        g = gabor_noise(Xg, Yg, ig, count=250, freq=float(want), seed=3,
                        spread=3.1416)
        got = spectral_peak(g)
        print("kernels: gabor freq %2d -> spectral peak %2d cycles"
              % (want, got))
        ok = ok and abs(got - want) <= max(1, 0.15 * want)

    # Orientation: with no spread every kernel shares one direction, so the
    # field must be strongly anisotropic; with full spread it must not be.
    aniso = gabor_noise(Xg, Yg, ig, count=250, freq=10.0, seed=5, spread=0.0)
    iso = gabor_noise(Xg, Yg, ig, count=250, freq=10.0, seed=5, spread=3.1416)
    r_a = float(np.gradient(aniso)[1].std() / np.gradient(aniso)[0].std())
    r_i = float(np.gradient(iso)[1].std() / np.gradient(iso)[0].std())
    print("kernels: gabor gradient anisotropy  aligned %.2f  isotropic %.2f"
          % (r_a, r_i))
    ok = ok and r_a > 2.0 * r_i

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
