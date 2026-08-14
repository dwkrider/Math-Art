# Strange attractors of continuous dynamical systems.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Each preset is an autonomous ODE dx/dt = f(x) whose trajectory is drawn
# to a bounded set of zero volume: the flow contracts phase-space volume
# (div f < 0) while neighbouring trajectories separate exponentially, so
# the orbit never closes and never escapes.  The curve is integrated,
# resampled to even arclength and normalised into the 2 m cube; the
# generator sweeps a tube along it.
#
# References:
# - E. N. Lorenz, "Deterministic Nonperiodic Flow", Journal of the
#   Atmospheric Sciences 20, 1963, pp. 130-141.
# - O. E. Rossler, "An equation for continuous chaos", Physics Letters A
#   57, 1976, pp. 397-398.
# - J. C. Sprott, "Chaos and Time-Series Analysis", Oxford University
#   Press, 2003 -- the source of most of the named systems here.

import math

import numpy as np


def _presets():
    S = math.sin
    C = math.cos
    T = math.tanh
    sgn = lambda v: (v > 0) - (v < 0)  # noqa: E731

    P = {}

    def A(key, label, rhs, params, x0, dt, steps=10000,
          transient=500, method='RK4'):
        P[key] = (label, rhs, params, x0, dt, steps, transient,
                  method)

    A('LORENZ', "Lorenz",
      lambda x, y, z, p: (p['a'] * (y - x),
                          x * (p['b'] - z) - y,
                          x * y - p['c'] * z),
      {'a': 10.0, 'b': 28.0, 'c': 8.0 / 3.0},
      (0.1, 0.0, -0.1), 0.006)

    A('AIZAWA', "Aizawa",
      lambda x, y, z, p: ((z - p['b']) * x - p['d'] * y,
                          p['d'] * x + (z - p['b']) * y,
                          p['c'] + p['a'] * z - z ** 3 / 3.0
                          - (x * x + y * y) * (1 + p['e'] * z)
                          + p['f'] * z * x ** 3),
      {'a': 0.95, 'b': 0.7, 'c': 0.6, 'd': 3.5, 'e': 0.25, 'f': 0.1},
      (0.1, 0.0, 0.0), 0.01)

    A('ANISHCHENKO', "Anishchenko-Astakhov",
      lambda x, y, z, p: (p['m'] * x + y - x * z,
                          -x,
                          -p['g'] * z + p['g'] * (x > 0) * x * x),
      {'m': 1.2, 'g': 0.5},
      (1.0, 1.0, 1.0), 0.01, 20000)

    A('ARNEODO', "Arneodo",
      lambda x, y, z, p: (y, z,
                          -p['a'] * x - p['b'] * y - z
                          + p['c'] * x ** 3),
      {'a': -5.5, 'b': 3.5, 'c': -1.0},
      (0.1, 0.0, 0.0), 0.01, 15000)

    A('BOUALI', "Bouali",
      lambda x, y, z, p: (x * (4 - y) + p['a'] * z,
                          -y * (1 - x * x),
                          -x * (1.5 - p['s'] * z) - 0.05 * z),
      {'a': 0.3, 's': 1.0},
      (1.0, 0.1, 0.1), 0.006, 20000)

    A('BURKESHAW', "Burke-Shaw",
      lambda x, y, z, p: (-p['s'] * (x + y),
                          -y - p['s'] * x * z,
                          p['s'] * x * y + p['v']),
      {'s': 10.0, 'v': 4.272},
      (1.0, 0.0, 0.0), 0.003)

    A('CHENCELIKOVSKY', "Chen-Celikovsky",
      lambda x, y, z, p: (p['a'] * (y - x),
                          -x * z + p['c'] * y,
                          x * y - p['b'] * z),
      {'a': 36.0, 'b': 3.0, 'c': 20.0},
      (1.0, 1.0, 1.0), 0.002)

    A('CHENLEE', "Chen-Lee",
      lambda x, y, z, p: (p['a'] * x - y * z,
                          p['b'] * y + x * z,
                          p['c'] * z + x * y / 3.0),
      {'a': 5.0, 'b': -10.0, 'c': -0.38},
      (1.0, 0.0, 4.5), 0.003, 15000)

    # Chua diode in the poster's form G = eps*x + (delta+eps)
    # * (|x+1| - |x-1|). The poster prints delta=-1, eps=0, which
    # provably decays to a fixed point; these values are the
    # canonical diode slopes (inner -8/7, outer -5/7) that give the
    # double scroll the render shows.
    A('CHUA', "Chua",
      lambda x, y, z, p: (p['alpha'] * (y - x - (
          p['eps'] * x + (p['delta'] + p['eps'])
          * (abs(x + 1) - abs(x - 1)))),
                          p['beta'] * (x - y + z),
                          -p['gamma'] * y),
      {'alpha': 15.6, 'beta': 1.0, 'gamma': 25.58, 'delta': 0.5,
       'eps': -5.0 / 7.0},
      (0.7, 0.0, 0.0), 0.01, 20000)

    A('COULLET', "Coullet",
      lambda x, y, z, p: (y, z,
                          p['a'] * x + p['b'] * y + p['c'] * z
                          + p['d'] * x ** 3),
      {'a': 0.8, 'b': -1.1, 'c': -0.45, 'd': -1.0},
      (0.1, 0.41, 0.31), 0.02, 15000)

    A('DADRAS', "Dadras",
      lambda x, y, z, p: (y - p['p'] * x + p['q'] * y * z,
                          p['r'] * y - x * z + z,
                          p['s'] * x * y - p['e'] * z),
      {'p': 3.0, 'q': 2.7, 'r': 1.7, 's': 2.0, 'e': 9.0},
      (0.1, 0.1, 0.1), 0.005, 15000)

    A('DEQUANLI', "Dequan Li",
      lambda x, y, z, p: (p['a'] * (y - x) + p['d'] * x * z,
                          p['k'] * x + p['f'] * y - x * z,
                          p['c'] * z + x * y - p['e'] * x * x),
      {'a': 40.0, 'c': 1.833, 'd': 0.16, 'e': 0.65, 'k': 55.0,
       'f': 20.0},
      (0.349, 0.0, -0.16), 0.0004, 20000)

    A('FINANCE', "Finance",
      lambda x, y, z, p: ((1.0 / p['b'] - p['a']) * x + z + x * y,
                          -p['b'] * y - x * x,
                          -x - p['c'] * z),
      {'a': 0.001, 'b': 0.2, 'c': 1.1},
      (0.1, 0.0, 0.0), 0.03, 15000)

    A('FOURWING', "Four-Wing",
      lambda x, y, z, p: (p['a'] * x - p['b'] * y * z,
                          -p['c'] * y + x * z,
                          p['k'] * x - p['d'] * z + x * y),
      {'a': 4.0, 'b': 6.0, 'c': 10.0, 'd': 5.0, 'k': 1.0},
      (0.1, 0.1, 0.1), 0.005, 20000)

    A('GENESIOTESI', "Genesio-Tesi",
      lambda x, y, z, p: (y, z,
                          -p['c'] * x - p['b'] * y - p['a'] * z
                          + x * x),
      {'a': 0.44, 'b': 1.1, 'c': 1.0},
      (0.1, 0.1, 0.1), 0.02, 15000)

    A('HADLEY', "Hadley",
      lambda x, y, z, p: (-y * y - z * z - p['a'] * x
                          + p['a'] * p['f'],
                          x * y - p['b'] * x * z - y + p['g'],
                          p['b'] * x * y + x * z - z),
      {'a': 0.2, 'b': 4.0, 'f': 8.0, 'g': 1.0},
      (0.1, 0.0, 0.0), 0.005, 15000)

    A('HALVORSEN', "Halvorsen",
      lambda x, y, z, p: (-p['a'] * x - 4 * y - 4 * z - y * y,
                          -p['a'] * y - 4 * z - 4 * x - z * z,
                          -p['a'] * z - 4 * x - 4 * y - x * x),
      {'a': 1.4},
      (1.0, 0.0, 0.0), 0.005)

    A('LIUCHEN', "Liu-Chen",
      lambda x, y, z, p: (p['a'] * y + p['b'] * x + p['c'] * y * z,
                          p['d'] * y - z + p['e'] * x * z,
                          p['f'] * z + p['g'] * x * y),
      {'a': 2.4, 'b': -3.78, 'c': 14.0, 'd': -11.0, 'e': 4.0,
       'f': 5.58, 'g': -1.0},
      (1.0, 3.0, 5.0), 0.002, 20000)

    # dz really is +z here (verified against both Meier's formula
    # sheet and the artist's poster); Mod 2 is the -z variant. The
    # exact +z system escapes to infinity at t ~ 16-22 -- the
    # reference render exists only because the Cinema 4D plugin
    # integrated with Euler, whose numerical dissipation keeps the
    # trajectory bounded. Reproduced faithfully: Euler method.
    A('LORENZMOD1', "Lorenz Mod 1",
      lambda x, y, z, p: (-p['a'] * x + y * y - z * z
                          + p['a'] * p['c'],
                          x * (y - p['b'] * z) + p['d'],
                          z + x * (p['b'] * y + z)),
      {'a': 0.1, 'b': 4.0, 'c': 14.0, 'd': 0.08},
      (0.1, 0.1, 0.1), 0.005, 20000, 500, 'EULER')

    A('LORENZMOD2', "Lorenz Mod 2",
      lambda x, y, z, p: (-p['a'] * x + y * y - z * z
                          + p['a'] * p['c'],
                          x * (y - p['b'] * z) + p['d'],
                          -z + x * (p['b'] * y + z)),
      {'a': 0.9, 'b': 5.0, 'c': 9.9, 'd': 1.0},
      (1.0, 1.0, 1.0), 0.003, 15000)

    # 4D system; the curve is the (x, y, z) projection
    A('LORENZSTENFLO', "Lorenz-Stenflo",
      lambda x, y, z, w, p: (p['a'] * (y - x) + p['d'] * w,
                             x * (p['c'] - z) - y,
                             x * y - p['b'] * z,
                             -x - p['a'] * w),
      {'a': 2.0, 'b': 0.7, 'c': 26.0, 'd': 1.5},
      (-1.0, 1.0, -1.0, 0.0), 0.01, 20000)

    # two coupled Lorenz systems (6D); the plotted projection is the
    # driven subsystem (x2, y2, z2), which carries the coupling
    A('COUPLEDLORENZ', "Coupled Lorenz",
      lambda x2, y2, z2, x1, y1, z1, p: (
          p['o'] * (y2 - x2) + p['e'] * (x1 - x2),
          p['r2'] * x2 - y2 - x2 * z2,
          -p['b'] * z2 + x2 * y2,
          p['o'] * (y1 - x1),
          p['r1'] * x1 - y1 - x1 * z1,
          -p['b'] * z1 + x1 * y1),
      {'b': 8.0 / 3.0, 'o': 10.0, 'r1': 35.0, 'r2': 1.15,
       'e': 2.85},
      (0.1, 0.1, 0.1, 0.1, 0.1, 0.1), 0.004, 20000)

    A('LUCHEN', "Lu-Chen",
      lambda x, y, z, p: (-p['a'] * p['b'] * x / (p['a'] + p['b'])
                          - y * z + p['c'],
                          p['a'] * y + x * z,
                          p['b'] * z + x * y),
      {'a': -10.0, 'b': -4.0, 'c': 18.1},
      (0.0, 0.0, 2.0), 0.005, 15000)

    A('NEWTONLEIPNIK', "Newton-Leipnik",
      lambda x, y, z, p: (-p['a'] * x + y + 10 * y * z,
                          -x - 0.4 * y + 5 * x * z,
                          p['b'] * z - 5 * x * y),
      {'a': 0.4, 'b': 0.175},
      (0.349, 0.0, -0.16), 0.02, 15000)

    A('NOSEHOOVER', "Nose-Hoover",
      lambda x, y, z, p: (y,
                          -x + y * z,
                          p['a'] - y * y),
      {'a': 1.5},
      (1.0, 0.0, 0.0), 0.01, 20000)

    # 4D system; the curve is the (x, y, z) projection
    A('QI', "Qi",
      lambda x, y, z, w, p: (p['a'] * (y - x) + y * z * w,
                             p['b'] * (x + y) - x * z * w,
                             -p['c'] * z + x * y * w,
                             -p['d'] * w + x * y * z),
      {'a': 30.0, 'b': 10.0, 'c': 1.0, 'd': 10.0},
      (1.0, 0.5, 7.0, 5.0), 0.0004, 20000)

    A('QICHEN', "Qi-Chen",
      lambda x, y, z, p: (p['a'] * (y - x) + y * z,
                          p['c'] * x + y - x * z,
                          x * y - p['b'] * z),
      {'a': 38.0, 'b': 8.0 / 3.0, 'c': 80.0},
      (3.0, -4.0, 3.0), 0.0015)

    # r=12 is below the chaotic regime -- the reference render shows
    # the decaying double spiral, so keep the transient (transient=0)
    A('RAYLEIGHBENARD', "Rayleigh-Benard",
      lambda x, y, z, p: (-p['a'] * (x - y),
                          p['r'] * x - y - x * z,
                          x * y - p['b'] * z),
      {'a': 9.0, 'r': 12.0, 'b': 5.0},
      (0.1, 0.0, 0.0), 0.01, 8000, 0)

    A('ROSSLER', "Roessler",
      lambda x, y, z, p: (-y - z,
                          x + p['a'] * y,
                          p['b'] + z * (x - p['c'])),
      {'a': 0.2, 'b': 0.2, 'c': 5.7},
      (1.0, 1.0, 0.0), 0.02)

    A('RUCKLIDGE', "Rucklidge",
      lambda x, y, z, p: (-p['k'] * x + p['a'] * y - y * z,
                          x,
                          -z + y * y),
      {'k': 2.0, 'a': 6.7},
      (1.0, 0.0, 0.0), 0.01, 15000)

    A('SAKARYA', "Sakarya",
      lambda x, y, z, p: (-x + y + y * z,
                          -x - y + p['a'] * x * z,
                          z - p['b'] * x * y),
      {'a': 0.4, 'b': 0.3},
      (1.0, -1.0, 1.0), 0.01, 15000)

    A('SHIMIZUMORIOKA', "Shimizu-Morioka",
      lambda x, y, z, p: (y,
                          x - p['a'] * y - x * z,
                          -p['b'] * z + x * x),
      {'a': 0.75, 'b': 0.45},
      (0.1, 0.0, 0.0), 0.01, 20000)

    A('THOMAS', "Thomas",
      lambda x, y, z, p: (-p['b'] * x + S(y),
                          -p['b'] * y + S(z),
                          -p['b'] * z + S(x)),
      {'b': 0.19},
      (0.1, 0.0, 0.0), 0.05, 20000)

    A('TSUCS1', "Three-Scroll Unified 1",
      lambda x, y, z, p: (p['a'] * (y - x) + p['d'] * x * z,
                          p['f'] * y - x * z,
                          p['c'] * z + x * y - p['e'] * x * x),
      {'a': 40.0, 'c': 0.833, 'd': 0.5, 'e': 0.65, 'f': 20.0},
      (0.1, 1.0, -0.1), 0.001, 20000)

    A('TSUCS2', "Three-Scroll Unified 2",
      lambda x, y, z, p: (p['a'] * (y - x) + p['d'] * x * z,
                          p['b'] * x - x * z + p['f'] * y,
                          p['c'] * z + x * y - p['e'] * x * x),
      {'a': 40.0, 'b': 55.0, 'c': 1.833, 'd': 0.16, 'e': 0.65,
       'f': 20.0},
      (0.1, 1.0, -0.1), 0.0004, 20000)

    A('WANGSUN', "Wang-Sun",
      lambda x, y, z, p: (x * p['a'] + p['c'] * y * z,
                          p['b'] * x + p['d'] * y - x * z,
                          p['e'] * z + p['f'] * x * y),
      {'a': 0.2, 'b': -0.01, 'c': 1.0, 'd': -0.4, 'e': -1.0,
       'f': -1.0},
      (0.3, 0.1, 1.0), 0.02, 30000)

    A('WIMOLBANLUE', "Wimol-Banlue",
      lambda x, y, z, p: (y - x,
                          -z * T(x),
                          -p['a'] + x * y + abs(y)),
      {'a': 2.0},
      (1.0, 1.0, 1.0), 0.02, 20000)

    A('YUWANG', "Yu-Wang",
      lambda x, y, z, p: (p['a'] * (y - x),
                          p['b'] * x - p['c'] * x * z,
                          math.e ** (x * y) - p['d'] * z),
      {'a': 10.0, 'b': 40.0, 'c': 2.0, 'd': 2.5},
      (2.2, 2.4, 28.0), 0.002, 15000)

    A('ZHOUCHEN', "Zhou-Chen",
      lambda x, y, z, p: (p['a'] * x + p['b'] * y + y * z,
                          p['c'] * y - x * z + p['d'] * y * z,
                          p['e'] * z - x * y),
      {'a': 2.97, 'b': 0.15, 'c': -3.0, 'd': 1.0, 'e': -8.78},
      (1.0, 1.0, 1.0), 0.004, 20000)

    return P


PRESETS = _presets()


_ORDER = sorted(PRESETS, key=lambda k: PRESETS[k][0])


def integrate(rhs, params, x0, dt, steps, transient,
              method='RK4'):
    """Fixed-step RK4 (or Euler) in any dimension; returns the
    trajectory (first three state components) after the transient.
    Raises OverflowError/ValueError if the system escapes."""
    s = list(x0)
    n = len(s)
    rng = range(n)
    euler = method == 'EULER'
    pts = []
    for i in range(steps + transient):
        k1 = rhs(*s, params)
        if euler:
            for j in rng:
                s[j] += dt * k1[j]
        else:
            k2 = rhs(*(s[j] + 0.5 * dt * k1[j] for j in rng),
                     params)
            k3 = rhs(*(s[j] + 0.5 * dt * k2[j] for j in rng),
                     params)
            k4 = rhs(*(s[j] + dt * k3[j] for j in rng), params)
            for j in rng:
                s[j] += dt / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j]
                                    + k4[j])
        if any(abs(v) > 1e6 for v in s):
            raise OverflowError("trajectory escaped")
        if i >= transient:
            pts.append((s[0], s[1], s[2]))
    return pts


def normalize(pts, size):
    """Center on the bounding-box center and scale the largest
    extent to `size`."""
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    c = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
    ext = max(hi[k] - lo[k] for k in range(3)) or 1.0
    s = size / ext
    return [((p[0] - c[0]) * s, (p[1] - c[1]) * s,
             (p[2] - c[2]) * s) for p in pts]


def speeds(pts):
    """Per-point segment speed (length to the next point), last value
    repeated; used for speed-based tapering."""
    v = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        v.append(math.dist(a, b))
    v.append(v[-1] if v else 0.0)
    return v


def resample(pts, n):
    """Resample an open polyline to n points, evenly spaced by arc
    length."""
    seg = [math.dist(pts[i], pts[i + 1])
           for i in range(len(pts) - 1)]
    total = sum(seg)
    if total <= 0.0:
        return [pts[0]] * n
    out = [pts[0]]
    want = total / (n - 1)
    j, acc = 0, 0.0
    for i in range(1, n - 1):
        s = i * want
        while j < len(seg) - 1 and acc + seg[j] < s:
            acc += seg[j]
            j += 1
        t = (s - acc) / (seg[j] or 1.0)
        a, b = pts[j], pts[j + 1]
        out.append((a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2])))
    out.append(pts[-1])
    return out


def build_attractor(key, size=10.0, dt_scale=1.0, steps=0,
                    params=None, samples=0):
    """Integrate preset `key`; returns (points, speed01) where
    speed01 is the per-point speed normalized to 0..1 (empty when
    resampling, which evens out the spacing)."""
    (label, rhs, dparams, x0, dt, dsteps, transient,
     method) = PRESETS[key]
    p = dict(dparams)
    if params:
        p.update(params)
    n = steps or dsteps
    pts = integrate(rhs, p, x0, dt * dt_scale, n, transient,
                    method)
    spd = speeds(pts)
    pts = normalize(pts, size)
    if samples:
        # arc-length resample: recompute speeds by sampling nearest
        # original speed is overkill; even spacing means no taper info
        pts = resample(pts, samples)
        spd = []
    if spd:
        lo, hi = min(spd), max(spd)
        rng = (hi - lo) or 1.0
        spd = [(s - lo) / rng for s in spd]
    return pts, spd
