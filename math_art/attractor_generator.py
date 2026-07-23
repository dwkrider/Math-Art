
# Strange Attractor Generator for Blender
#
# All 38 strange attractors of Chaotic Atmospheres' "MATHRULES" art
# series (chaoticatmospheres.com/mathrules-strange-attractors) as
# presets (plus Zhou-Chen from the same collection): the ODE systems
# and parameter values follow Juergen Meier's compilation
# (3d-meier.de, Tutorial 19), the artist's stated source, with
# Lorenz / Roessler / Chua transcribed from the posters themselves.
# The 4D systems (Qi, Lorenz-Stenflo) and the 6D coupled Lorenz pair
# are drawn as their (x, y, z) projection. The system is integrated with fixed-step RK4, the transient
# discarded, and the trajectory emitted as a curve -- optionally
# resampled to even arc length, with the tube radius modulated by the
# local speed (slow regions thicken, the look of the original
# renders).
#
# Run this file with plain python for a self-test: every preset must
# integrate to a bounded, non-collapsing, non-escaping trajectory.

bl_info = {
    "name": "Strange Attractors",
    "author": "Math Art project (systems after Juergen Meier "
              "and Chaotic Atmospheres)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Strange Attractor",
    "description": "Chaotic attractor curves: the 38 systems of the "
                   "Chaotic Atmospheres MATHRULES series",
    "category": "Add Curve",
}

import math


# ---- the systems -------------------------------------------------------
# Each entry: key -> (label, rhs(*state, p) -> dstate,
#                     params dict, x0, dt, steps, transient, method)
# The state may have more than 3 components (Qi and Lorenz-Stenflo
# are 4D, the coupled Lorenz pair is 6D); the first three are the
# plotted projection. dt/steps are tuned so each preset draws the
# attractor's familiar shape; all values verified by the __main__
# self-test. method is 'RK4' except where the reference renders
# depend on Euler's numerical dissipation (see Lorenz Mod 1).


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


# ---- integration -------------------------------------------------------

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


# ---- Blender layer -----------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _preset_items():
        items = []
        for k in _ORDER:
            label, rhs, params = PRESETS[k][:3]
            desc = ", ".join(f"{n}={v:g}"
                             for n, v in sorted(params.items()))
            items.append((k, label, f"parameters: {desc}"))
        return items

    class CURVE_OT_attractor_add(bpy.types.Operator):
        """Add a strange attractor trajectory curve (presets after
        the Chaotic Atmospheres / Juergen Meier collection)"""
        bl_idname = "curve.attractor_add"
        bl_label = "Strange Attractor"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(name="Attractor",
                             items=_preset_items(),
                             default='LORENZ')
        steps: IntProperty(
            name="Steps", default=0, min=0, max=200000, step=1000,
            description="Integration steps (0 = preset default)")
        dt_scale: FloatProperty(
            name="Time Step Scale", default=1.0, min=0.05, max=10.0,
            description="Multiplier on the preset's integration step")
        size: FloatProperty(name="Size", default=10.0, min=0.1,
                            max=1000.0,
                            description="Largest bounding-box extent")
        spline: EnumProperty(
            name="Spline",
            items=[('POLY', "Poly", "one point per sample"),
                   ('BEZIER', "Bezier",
                    "auto-handles, smoother at low sample counts")],
            default='POLY')
        samples: IntProperty(
            name="Resample", default=0, min=0, max=20000,
            description="Resample to N evenly spaced points "
                        "(0 = raw integration points; even spacing "
                        "disables speed taper)")
        radius: FloatProperty(
            name="Tube Radius", default=0.05, min=0.0, max=10.0,
            step=1, precision=3,
            description="Curve bevel depth (0 = wire only)")
        resolution: IntProperty(name="Bevel Resolution", default=4,
                                min=0, max=16)
        taper: FloatProperty(
            name="Speed Taper", default=0.0, min=0.0, max=1.0,
            description="Thicken the tube where the flow is slow "
                        "(0 = uniform radius)")
        profile_sides: IntProperty(
            name="Profile Sides", default=0, min=0, max=12,
            description="Polygonal tube cross-section with N flat "
                        "sides (0 = round; e.g. 5 gives the pentagon "
                        "profile used in the reference lorentz.blend)")

        def execute(self, context):
            try:
                pts, spd = build_attractor(
                    self.preset, self.size, self.dt_scale,
                    self.steps, None, self.samples)
            except (OverflowError, ValueError) as e:
                self.report({'ERROR'},
                            f"integration failed: {e} -- try a "
                            f"smaller Time Step Scale")
                return {'CANCELLED'}
            label = PRESETS[self.preset][0]
            name = f"{label} Attractor"
            cu = bpy.data.curves.new(name, 'CURVE')
            cu.dimensions = '3D'
            if self.spline == 'BEZIER':
                sp = cu.splines.new('BEZIER')
                sp.bezier_points.add(len(pts) - 1)
                for i, pnt in enumerate(pts):
                    bp = sp.bezier_points[i]
                    bp.co = pnt
                    bp.handle_left_type = 'AUTO'
                    bp.handle_right_type = 'AUTO'
                bpts = sp.bezier_points
            else:
                sp = cu.splines.new('POLY')
                sp.points.add(len(pts) - 1)
                for i, pnt in enumerate(pts):
                    sp.points[i].co = (pnt[0], pnt[1], pnt[2], 1.0)
                bpts = sp.points
            if self.taper > 0.0 and spd:
                # slow flow -> fat tube, like the original renders
                for i, s in enumerate(spd):
                    bpts[i].radius = 1.0 + self.taper * 3.0 * (1 - s)
            cu.bevel_depth = self.radius
            cu.bevel_resolution = self.resolution
            if self.radius > 0:
                cu.use_fill_caps = True
            obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            if self.radius > 0 and self.profile_sides >= 3:
                cu.bevel_mode = 'OBJECT'
                cu.bevel_object = self._profile(
                    context, name, self.profile_sides, self.radius,
                    obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{label}: {len(pts)} points")
            return {'FINISHED'}

        @staticmethod
        def _profile(context, name, sides, radius, parent):
            """Closed N-gon poly curve used as the bevel object (the
            pentagonal-cross-section trick from lorentz.blend)."""
            cu = bpy.data.curves.new(f"{name} Profile", 'CURVE')
            cu.dimensions = '2D'
            sp = cu.splines.new('POLY')
            sp.points.add(sides - 1)
            for i in range(sides):
                a = 2 * math.pi * i / sides + math.pi / 2
                sp.points[i].co = (radius * math.cos(a),
                                   radius * math.sin(a), 0.0, 1.0)
            sp.use_cyclic_u = True
            prof = bpy.data.objects.new(f"{name} Profile", cu)
            context.collection.objects.link(prof)
            prof.parent = parent
            prof.hide_set(True)
            prof.hide_render = True
            return prof

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            for k in ('steps', 'dt_scale', 'size', 'spline',
                      'samples', 'radius'):
                lay.prop(self, k)
            if self.radius > 0:
                lay.prop(self, 'resolution')
                lay.prop(self, 'profile_sides')
            if not self.samples:
                lay.prop(self, 'taper')

    def _menu_func(self, context):
        self.layout.operator_menu_enum("curve.attractor_add",
                                       "preset",
                                       text="Strange Attractor",
                                       icon='RNDCURVE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_attractor_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_attractor_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        fails = []
        for key in _ORDER:
            label = PRESETS[key][0]
            try:
                pts, spd = build_attractor(key)
                lo = [min(p[k] for p in pts) for k in range(3)]
                hi = [max(p[k] for p in pts) for k in range(3)]
                ext = sorted(hi[k] - lo[k] for k in range(3))
                # non-degenerate: real extent in at least 2 axes
                flat = ext[1] < 0.5
                # not settled onto a fixed point: the last fifth of
                # the trajectory still covers a decent volume
                tail = pts[-len(pts) // 5:]
                tlo = [min(p[k] for p in tail) for k in range(3)]
                thi = [max(p[k] for p in tail) for k in range(3)]
                text = max(thi[k] - tlo[k] for k in range(3))
                # Rayleigh-Benard is a deliberate transient spiral
                stuck = text < 0.5 and key != 'RAYLEIGHBENARD'
                bad = ([] + (["flat"] if flat else [])
                       + (["stuck"] if stuck else []))
                print(f"{label:28s} {len(pts):6d} pts  "
                      f"ext=({ext[0]:5.2f},{ext[1]:5.2f},"
                      f"{ext[2]:5.2f})  tail={text:5.2f}  "
                      f"{'OK' if not bad else ' '.join(bad)}")
                if bad:
                    fails.append(key)
            except (OverflowError, ValueError, ZeroDivisionError) \
                    as e:
                print(f"{label:28s} FAILED: {e}")
                fails.append(key)
        print(f"\n{len(PRESETS)} presets:",
              "ALL OK" if not fails else f"FAILURES: {fails}")
