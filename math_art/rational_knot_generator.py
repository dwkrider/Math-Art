
# Rational & Pretzel Knots generator for Blender: the classical
# small-knot families of Conway's tangle calculus, drawn as explicit
# planar diagrams embedded in 3D with the strands woven strictly
# over/under -- a small piece of knotwork -- and swept as rope tubes
# or curves.
#
# THREE FAMILIES.
#
# RATIONAL (2-bridge) knots from a continued fraction [a1, ..., an]
# (Conway notation).  The standard 4-plat diagram is built by growing
# a rational tangle from the trivial 0-tangle: entry a_i applies a_i
# signed half-twists as a HORIZONTAL twist region appended at the
# right (i odd) or a VERTICAL twist region appended at the bottom
# (i even), alternating, exactly as in Conway's tangle construction.
# The tangle is closed with the NUMERATOR closure (NW joined to NE
# above, SW to SE below) when the sequence ends on a horizontal
# region, and with the equivalent DENOMINATOR closure (NW-SW, NE-SE)
# when it ends on a vertical one -- the standard plat closure that
# makes both parities realize K(p/q).  CONVENTION: the fraction of
# [a1, ..., an] is the continued fraction
#     p/q = a_n + 1/(a_{n-1} + 1/( ... + 1/a_1)),
# so [3] -> 3/1 (trefoil), [2,2] -> 5/2 (figure-eight 4_1),
# [3,2] -> 7/3 (5_2).  By Schubert's theorem the result is the
# 2-bridge knot/link K(p/q): a knot when p is odd, a 2-component
# link when p is even.
#
# PRETZEL links P(p1, ..., pn): n vertical 2-strand twist regions
# side by side, region i carrying p_i signed half-twists; the tops
# are joined in a row and closed by an outer return arc, likewise the
# bottoms (the standard pretzel closure).
#
# TWIST knots C(n, 2): one twist region of n half-twists closed by a
# clasp of two crossings -- realized here as the rational tangle
# [n, 2], fraction (2n+1)/n: n=1 trefoil, n=2 figure-eight, n=3 5_2,
# n=4 6_1 (stevedore), n=5 7_2, ...
#
# SIGN CONVENTION for half-twists: within every twist region the two
# strands swap once per half-twist; for a positive entry the strand
# running down-and-right (horizontal regions) / down-and-right
# (vertical regions) passes OVER, for a negative entry under.  A
# same-sign twist region is automatically alternating along each
# strand, so an all-positive continued fraction (and an all-positive
# pretzel) yields the classical alternating diagram; the self-test
# checks this.  The over/under bit at each crossing is deterministic
# from the twist signs -- no global solve is needed.  It is rendered
# as a z offset smoothstepped along each strand between crossings
# (the knot-carpet weave), and the tube mesh reuses the knot carpet's
# proximity-welded rotation-minimizing sweep.
#
# References:
# - J. H. Conway, "An enumeration of knots and links, and some of
#   their algebraic properties", in Computational Problems in
#   Abstract Algebra, Pergamon, 1970 -- rational tangles, the
#   continued-fraction notation.
# - H. Schubert, "Knoten mit zwei Bruecken", Math. Z. 65:133-170,
#   1956 -- the classification of 2-bridge knots by p/q.
# - L. H. Kauffman & S. Lambropoulou, "On the classification of
#   rational knots", Enseign. Math. 49:357-410, 2003 -- an elementary
#   proof of Schubert's theorem via tangle calculus.
# - H. F. Trotter, "Non-invertible knots exist", Topology 2:275-280,
#   1963 -- the pretzel knots P(p, q, r).
# - C. C. Adams, "The Knot Book", W. H. Freeman, 1994 -- twist
#   knots, pretzel links, rational tangles (ch. 2).
# - D. Rolfsen, "Knots and Links", Publish or Perish, 1976 -- the
#   knot table names (4_1, 5_2, 6_1, ...) used for the presets.

bl_info = {
    "name": "Rational & Pretzel Knots",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Rational / Pretzel Knot",
    "description": "2-bridge (rational), pretzel and twist knots as "
                   "woven over/under diagrams -- rope tube or curve",
    "category": "Add Curve",
}

import re
from math import pi

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from . import knot_carpet_generator as kcg
except ImportError:                     # CLI / standalone use
    import knot_carpet_generator as kcg


# --------------------------------------------------------------------
# Continued fraction
# --------------------------------------------------------------------

def cf_fraction(a):
    """(p, q) of the continued fraction [a1, ..., an] under the
    convention  p/q = a_n + 1/(a_{n-1} + 1/(... + 1/a_1)):
    [3] -> (3, 1), [2, 2] -> (5, 2), [3, 2] -> (7, 3).  q is
    normalized non-negative; p carries the sign (mirror image)."""
    p, q = 1, 0
    for ai in a:
        p, q = int(ai) * p + q, p
    if q < 0:
        p, q = -p, -q
    return p, q


# --------------------------------------------------------------------
# Diagram pieces
# --------------------------------------------------------------------
#
# A diagram is a bag of open 2D polyline PIECES whose endpoints meet
# exactly (twist-region strands, connectors, corner fillets, closure
# arcs) plus a list of CROSSING records (center, over-branch
# direction, catch radius).  Pieces are assembled into closed loops
# by endpoint matching; the crossings are then located on the
# assembled loops and each of the two passing branches is classified
# over or under by its tangent -- deterministic from the twist signs.

# All lengths are in units of the base strand gap g = 1.

_LEAD = 0.6          # straight lead-in before/after a twist region
_FILLET = 0.6        # corner fillet radius of the 4-plat routing
_V_OVER = 1.0        # relative handedness of vertical vs horizontal
                     # regions (+1 makes all-positive inputs give the
                     # classical alternating diagrams; see self-test)


class _Diagram:
    def __init__(self):
        self.pieces = []          # list of (N, 2) arrays
        self.crossings = []       # (cx, cy, ux, uy, rcatch)

    def add(self, pts):
        P = np.asarray(pts, float)
        if len(P) >= 2 and np.linalg.norm(P[-1] - P[0]) > 1e-9:
            self.pieces.append(P)


def _h_region(dia, x0, yT, yB, m, s, tw):
    """Horizontal twist region: two strands entering at (x0, yT) and
    (x0, yB) swap m times while running right.  Advance per half-twist
    is tw * (row gap), so the crossing angle is gap-independent.
    Returns the exit x."""
    c, h = 0.5 * (yT + yB), 0.5 * (yT - yB)
    w = 2.0 * h * tw
    if m == 0:
        return x0
    npr = max(16, int(24 * h))            # samples per half-twist
    t = np.linspace(0.0, m, m * npr + 1)
    x = x0 + w * t
    yA = c + h * np.cos(pi * t)
    dia.add(np.column_stack([x, yA]))
    dia.add(np.column_stack([x, 2.0 * c - yA]))
    # over branch: the DESCENDING strand for s > 0 (tangent (w, -pi h))
    u = np.array([w, -pi * h if s > 0 else pi * h])
    u /= np.linalg.norm(u)
    rc = min(0.5 * h, 0.4)
    for j in range(m):
        dia.crossings.append((x0 + w * (j + 0.5), c, u[0], u[1], rc))
    return x0 + w * m


def _v_region(dia, y0, xL, xR, m, s, tw):
    """Vertical twist region: two strands entering at (xL, y0) and
    (xR, y0) swap m times while running down.  Returns the exit y."""
    c, h = 0.5 * (xL + xR), 0.5 * (xR - xL)
    w = 2.0 * h * tw
    if m == 0:
        return y0
    npr = max(16, int(24 * h))
    t = np.linspace(0.0, m, m * npr + 1)
    y = y0 - w * t
    xA = c - h * np.cos(pi * t)
    dia.add(np.column_stack([xA, y]))
    dia.add(np.column_stack([2.0 * c - xA, y]))
    # over branch: the strand moving right-and-down for s > 0
    u = np.array([pi * h if s * _V_OVER > 0 else -pi * h, -w])
    u /= np.linalg.norm(u)
    rc = min(0.5 * h, 0.4)
    for j in range(m):
        dia.crossings.append((c, y0 - w * (j + 0.5), u[0], u[1], rc))
    return y0 - w * m


def _fillet(p, mode, r, n=14):
    """Quarter-circle corner: 'h_down' turns +x travel to -y,
    'hL_down' turns -x travel to -y, 'v_right' turns -y travel to +x.
    Starts exactly at p; the far endpoint is (p.x +/- r, p.y - r)."""
    th = np.linspace(0.0, 0.5 * pi, n)
    if mode == 'h_down':
        pts = np.column_stack([p[0] + r * np.sin(th),
                               p[1] - r * (1.0 - np.cos(th))])
        end = (p[0] + r, p[1] - r)
    elif mode == 'hL_down':
        pts = np.column_stack([p[0] - r * np.sin(th),
                               p[1] - r * (1.0 - np.cos(th))])
        end = (p[0] - r, p[1] - r)
    else:                                             # 'v_right'
        pts = np.column_stack([p[0] + r * (1.0 - np.cos(th)),
                               p[1] - r * np.sin(th)])
        end = (p[0] + r, p[1] - r)
    pts[0] = p
    pts[-1] = end
    return pts, end


def _sine_arc(p, q, bulge, expo=1.0, n=96):
    """Closure arc from p to q bulging by `bulge` (+up / -down): a
    half-sine profile, sharpened toward vertical end tangents by
    expo < 1.  Endpoints are forced exact for the port matcher."""
    t = np.linspace(0.0, 1.0, n)
    prof = np.sin(pi * t)
    if expo != 1.0:
        prof = prof ** expo
    pts = np.column_stack([p[0] + (q[0] - p[0]) * t,
                           p[1] + (q[1] - p[1]) * t + bulge * prof])
    pts[0] = p
    pts[-1] = q
    return pts


def _side_arc(p, q, bulge, expo=0.35, n=120):
    """Sideways closure arc from p to q bulging in x by `bulge`
    (+right / -left), sharpened toward vertical near-end tangents so
    it clears the outermost strand columns it closes around."""
    t = np.linspace(0.0, 1.0, n)
    prof = np.sin(pi * t) ** expo
    pts = np.column_stack([p[0] + (q[0] - p[0]) * t + bulge * prof,
                           p[1] + (q[1] - p[1]) * t])
    pts[0] = p
    pts[-1] = q
    return pts


# --------------------------------------------------------------------
# The three families as piece bags
# --------------------------------------------------------------------

def build_rational(a, tw=0.8):
    """The 4-plat diagram of the rational tangle [a1, ..., an] with
    numerator closure.  Odd entries twist at the RIGHT (horizontal
    regions between the NE/SE ends), even entries at the BOTTOM
    (vertical regions between the SW/SE ends); quarter-circle fillets
    turn the strand ends between the two directions."""
    dia = _Diagram()
    s0, r = _LEAD, _FILLET
    yT, yB0 = 0.0, -1.0
    dia.add([(0.0, yT), (s0, yT)])              # top stub, NW..NE
    dia.add([(0.0, yB0), (s0, yB0)])            # bottom stub, SW..SE
    nw = (0.0, yT)
    ne = (s0, yT)
    sw, sws = (0.0, yB0), 'hL'                  # free end faces -x
    se, ses = (s0, yB0), 'h'                    # free end faces +x
    for i, ai in enumerate(a):
        m, sg = abs(int(ai)), (1 if ai >= 0 else -1)
        if m == 0:
            continue
        if i % 2 == 0:                          # horizontal, at right
            if ses == 'v':
                pts, se = _fillet(se, 'v_right', r)
                dia.add(pts)
                ses = 'h'
            x0 = max(ne[0], se[0]) + _LEAD
            if x0 > ne[0] + 1e-9:
                dia.add([ne, (x0, ne[1])])
            if x0 > se[0] + 1e-9:
                dia.add([se, (x0, se[1])])
            x1 = _h_region(dia, x0, ne[1], se[1], m, sg, tw)
            ne = (x1, ne[1])
            se = (x1, se[1])
        else:                                   # vertical, at bottom
            if ses == 'h':
                pts, se = _fillet(se, 'h_down', r)
                dia.add(pts)
                ses = 'v'
            if sws == 'hL':
                pts, sw = _fillet(sw, 'hL_down', r)
                dia.add(pts)
                sws = 'v'
            y0 = min(sw[1], se[1]) - _LEAD
            if y0 < sw[1] - 1e-9:
                dia.add([sw, (sw[0], y0)])
            if y0 < se[1] - 1e-9:
                dia.add([se, (se[0], y0)])
            y1 = _v_region(dia, y0, sw[0], se[0], m, sg, tw)
            sw = (sw[0], y1)
            se = (se[0], y1)
    # Plat closure.  Numerator closure (NW-NE joined above, SW-SE
    # below) is the correct 2-bridge closure when the LAST twist
    # region is horizontal (odd-length sequence): the tangle fraction
    # is then F(T) = p/q and N(T) = K(p/q).  For an even-length
    # sequence the last region is vertical and F(T) = q/p, so the
    # DENOMINATOR closure (NW-SW joined at the left, NE-SE at the
    # right) is used instead: D(T) = K(p/q) again.  Either way the
    # diagram realizes K(p/q) with p/q = a_n + 1/(... + 1/a_1).
    if len(a) % 2 == 1:
        width = max(ne[0] - nw[0], se[0] - sw[0], 1.0)
        dia.add(_sine_arc(nw, ne, max(0.9, 0.18 * width)))
        dia.add(_sine_arc(sw, se, -max(0.9, 0.18 * width)))
    else:
        height = max(nw[1] - sw[1], ne[1] - se[1], 1.0)
        bulge = max(0.9, 0.18 * height)
        dia.add(_side_arc(nw, sw, -bulge))
        dia.add(_side_arc(ne, se, bulge))
    return dia


def build_pretzel(p_list, tw=0.8, spread=2.4):
    """The standard pretzel diagram P(p1, ..., pn): one vertical
    twist region per entry, tops joined in a row (short arcs between
    neighbours, an outer return arc from the first to the last), the
    bottoms likewise."""
    dia = _Diagram()
    n = len(p_list)
    w = tw                                       # advance, gap = 1
    depth = max([abs(int(p)) * w for p in p_list] + [w])
    yB = -(depth + _LEAD)
    xs = [i * spread for i in range(n)]
    for i, pv in enumerate(p_list):
        m, sg = abs(int(pv)), (1 if pv >= 0 else -1)
        xL, xR = xs[i] - 0.5, xs[i] + 0.5
        y1 = _v_region(dia, 0.0, xL, xR, m, sg, tw)
        dia.add([(xL, y1), (xL, yB)])
        dia.add([(xR, y1), (xR, yB)])
    for i in range(n - 1):                       # neighbour arcs
        dia.add(_sine_arc((xs[i] + 0.5, 0.0), (xs[i + 1] - 0.5, 0.0),
                          0.55, expo=0.5, n=48))
        dia.add(_sine_arc((xs[i] + 0.5, yB), (xs[i + 1] - 0.5, yB),
                          -0.55, expo=0.5, n=48))
    dia.add(_sine_arc((xs[0] - 0.5, 0.0), (xs[-1] + 0.5, 0.0),
                      1.7, expo=0.35, n=160))    # outer return arcs
    dia.add(_sine_arc((xs[0] - 0.5, yB), (xs[-1] + 0.5, yB),
                      -1.7, expo=0.35, n=160))
    return dia


# --------------------------------------------------------------------
# Assembly: pieces -> closed component loops
# --------------------------------------------------------------------

def _assemble(pieces):
    """Join the open pieces end-to-end into closed loops (endpoint
    coordinates matched to 1e-5).  Every junction must be exactly
    2-valent; the walk then decomposes the diagram into its link
    components.  Returns a list of (N, 2) closed polylines."""
    def key(p):
        return (round(float(p[0]), 5), round(float(p[1]), 5))

    endmap = {}
    for i, P in enumerate(pieces):
        endmap.setdefault(key(P[0]), []).append((i, 0))
        endmap.setdefault(key(P[-1]), []).append((i, 1))
    for k, v in endmap.items():
        if len(v) != 2:
            raise ValueError("diagram not closed at %s (%d ends)"
                             % (k, len(v)))
    used = [False] * len(pieces)
    loops = []
    for start in range(len(pieces)):
        if used[start]:
            continue
        pts = []
        i, e = start, 0
        while True:
            used[i] = True
            P = pieces[i] if e == 0 else pieces[i][::-1]
            seg = [tuple(q) for q in P]
            pts.extend(seg if not pts else seg[1:])
            ends = endmap[key(P[-1])]
            nxt = ends[1] if ends[0] == (i, 1 - e) else ends[0]
            i, e = nxt
            if used[i]:
                break
        if pts and key(pts[0]) == key(pts[-1]):
            pts.pop()
        loops.append(np.asarray(pts, float))
    return loops


def _resample(P, step):
    """Uniform arclength resampling of a closed polyline (linear
    interpolation, so the path itself is unchanged)."""
    Q = np.vstack([P, P[:1]])
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    n = max(48, int(round(total / step)))
    si = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack([np.interp(si, s, Q[:, 0]),
                            np.interp(si, s, Q[:, 1])])


def _cyclic_runs(idx, n):
    """Split sorted index array `idx` into cyclically contiguous
    runs of a length-n ring."""
    idx = np.asarray(idx)
    if len(idx) == n:
        return [idx]
    br = np.nonzero(np.diff(idx) > 1)[0]
    runs = list(np.split(idx, br + 1))
    if len(runs) > 1 and idx[0] == 0 and idx[-1] == n - 1:
        runs[0] = np.concatenate([runs[-1], runs[0]])
        runs.pop()
    return runs


def _assign_crossings(loops, crossings):
    """Locate every crossing on the assembled loops and classify its
    two passing branches over (+1) / under (-1) by tangent direction
    against the recorded over-branch direction.  Returns one sorted
    [(vertex_index, sign)] list per loop, and asserts each crossing
    is exactly one-over-one-under."""
    signed = [[] for _ in loops]
    for (cx, cy, ux, uy, rc) in crossings:
        cands = []
        for li, P in enumerate(loops):
            d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
            idx = np.nonzero(d < rc)[0]
            if len(idx) == 0:
                continue
            for run in _cyclic_runs(idx, len(P)):
                j = int(run[int(np.argmin(d[run]))])
                t = P[(j + 1) % len(P)] - P[(j - 1) % len(P)]
                nt = float(np.linalg.norm(t))
                if nt < 1e-12:
                    continue
                dot = abs(float(t[0]) * ux + float(t[1]) * uy) / nt
                cands.append((dot, li, j))
        if len(cands) != 2:
            raise ValueError("crossing at (%.2f, %.2f): found %d "
                             "branches, expected 2"
                             % (cx, cy, len(cands)))
        cands.sort(reverse=True)
        if cands[0][0] - cands[1][0] < 0.05:
            raise ValueError("ambiguous over/under at (%.2f, %.2f)"
                             % (cx, cy))
        signed[cands[0][1]].append((cands[0][2], 1))
        signed[cands[1][1]].append((cands[1][2], -1))
    return [sorted(sg) for sg in signed]


def _weave_z(P, signed, h):
    """Per-vertex z offset of one closed loop: +/-h at each crossing
    (over/under), smoothstepped along arclength in between -- the
    knot-carpet weave profile."""
    n = len(P)
    if not signed or h <= 0.0:
        return np.zeros(n)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1]) + float(np.linalg.norm(P[0] - P[-1]))
    anch = sorted((float(s[i % n]), sg * h) for i, sg in signed)
    sa = np.array([anch[-1][0] - total] + [a[0] for a in anch]
                  + [anch[0][0] + total])
    za = np.array([anch[-1][1]] + [a[1] for a in anch]
                  + [anch[0][1]])
    j = np.clip(np.searchsorted(sa, s, side='right') - 1,
                0, len(sa) - 2)
    seg = sa[j + 1] - sa[j]
    u = np.clip((s - sa[j]) / np.where(seg < 1e-12, 1.0, seg),
                0.0, 1.0)
    f = u * u * (3.0 - 2.0 * u)
    return za[j] + (za[j + 1] - za[j]) * f


# --------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------

def build_knot(family, ints, twist_len=0.8, spread=2.4, samples=10,
               weave=0.4):
    """Diagram -> woven 3D centerlines, centered and fit to the 2 m
    cube.  Returns a dict:
      loops    -- one (N, 3) closed centerline per link component
      loops2d  -- the flat diagram loops (diagram units)
      signed   -- per-loop [(vertex, +1 over / -1 under)]
      ncross   -- crossing count of the diagram
      ncomp    -- number of link components
      fraction -- (p, q) for RATIONAL / TWIST, else None
    """
    ints = [int(v) for v in ints]
    if not ints:
        raise ValueError("empty notation")
    if family == 'TWIST':
        a = [ints[0], 2]
        dia = build_rational(a, twist_len)
        frac = cf_fraction(a)
    elif family == 'RATIONAL':
        dia = build_rational(ints, twist_len)
        frac = cf_fraction(ints)
    elif family == 'PRETZEL':
        dia = build_pretzel(ints, twist_len, spread)
        frac = None
    else:
        raise ValueError("unknown family %r" % (family,))
    step = 1.0 / max(4, int(samples))
    loops2 = [_resample(P, step) for P in _assemble(dia.pieces)]
    signed = _assign_crossings(loops2, dia.crossings)
    loops3 = [np.column_stack([P, _weave_z(P, sg, weave)])
              for P, sg in zip(loops2, signed)]
    allp = np.vstack(loops3)
    ctr = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    ext = float((allp.max(axis=0) - allp.min(axis=0)).max())
    sc = 2.0 / ext if ext > 1e-12 else 1.0
    loops3 = [(P - ctr) * sc for P in loops3]
    return dict(loops=loops3, loops2d=loops2, signed=signed,
                ncross=len(dia.crossings), ncomp=len(loops3),
                fraction=frac)


def _parse_ints(text):
    toks = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    if not toks:
        raise ValueError("empty notation")
    return [int(t) for t in toks]


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

if _IN_BLENDER:

    _PRESETS = {
        'TREFOIL': ('RATIONAL', "3"),
        'FIG8': ('RATIONAL', "2 2"),
        'K5_2': ('RATIONAL', "3 2"),
        'K6_1': ('TWIST', "4"),
        'K7_2': ('TWIST', "5"),
        'P357': ('PRETZEL', "3 5 7"),
        'PM237': ('PRETZEL', "-2 3 7"),
        'P222': ('PRETZEL', "2 2 2"),
    }

    class CURVE_OT_rational_knot_add(bpy.types.Operator):
        """Add a rational (2-bridge), pretzel or twist knot as a
        woven over/under diagram -- rope tube mesh or curve"""
        bl_idname = "curve.rational_knot_add"
        bl_label = "Rational / Pretzel Knot"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            items=[('FIG8', "Figure-Eight 4_1 -- rational [2 2]", ""),
                   ('TREFOIL', "Trefoil 3_1 -- rational [3]", ""),
                   ('K5_2', "5_2 -- rational [3 2]", ""),
                   ('K6_1', "6_1 Stevedore -- twist n=4", ""),
                   ('K7_2', "7_2 -- twist n=5", ""),
                   ('P357', "Pretzel (3,5,7)", ""),
                   ('PM237', "Pretzel (-2,3,7)", ""),
                   ('P222', "Pretzel (2,2,2) -- 3-component link",
                    ""),
                   ('CUSTOM', "Custom", "Use Family + Notation")],
            default='FIG8')
        family: EnumProperty(
            name="Family",
            items=[('RATIONAL', "Rational (2-bridge)",
                    "Conway continued fraction [a1 a2 ...]"),
                   ('PRETZEL', "Pretzel",
                    "P(p1, p2, ...): one vertical twist region "
                    "per entry"),
                   ('TWIST', "Twist Knot",
                    "n half-twists closed by a clasp: C(n, 2)")],
            default='RATIONAL')
        notation: StringProperty(
            name="Notation", default="2 2",
            description="Signed integers, space or comma separated: "
                        "continued fraction for RATIONAL, twist "
                        "counts for PRETZEL, a single n for TWIST")
        twist_len: FloatProperty(
            name="Twist Length", default=0.8, min=0.5, max=2.5,
            description="Advance per half-twist relative to the "
                        "strand gap (crossing size)")
        spread: FloatProperty(
            name="Region Spacing", default=2.4, min=1.4, max=6.0,
            description="Distance between pretzel twist regions "
                        "(strand-gap units)")
        samples: IntProperty(
            name="Samples", default=10, min=6, max=40,
            description="Sample density: points per strand-gap of "
                        "arclength")
        weave: FloatProperty(
            name="Weave Depth", default=0.4, min=0.0, max=2.0,
            description="Over/under lift at each crossing "
                        "(strand-gap units, before fitting)")
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh Tube", "swept woven rope"),
                   ('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", "")],
            default='MESH')
        radius: FloatProperty(
            name="Tube Radius", default=0.05, min=0.0, max=0.5,
            step=1, precision=3,
            description="Curve bevel depth / tube radius (m)")
        resolution: IntProperty(name="Bevel Resolution", default=6,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32)
        color_components: BoolProperty(
            name="Color Components", default=True,
            description="One material with a distinct color per "
                        "link component (links only)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            fam, nota = ((self.family, self.notation)
                         if self.preset == 'CUSTOM'
                         else _PRESETS[self.preset])
            try:
                ints = _parse_ints(nota)
                k = build_knot(fam, ints, self.twist_len,
                               self.spread, self.samples, self.weave)
            except (ValueError, KeyError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            comps = [np.asarray(P) * self.scale for P in k['loops']]
            d = len(comps)
            tag = " ".join(str(v) for v in ints)
            if fam == 'PRETZEL':
                name = "Pretzel (%s)" % ",".join(str(v)
                                                 for v in ints)
            elif fam == 'TWIST':
                name = "Twist Knot %d" % ints[0]
            else:
                name = "Rational Knot [%s]" % tag
            color = self.color_components and d > 1
            if self.output == 'MESH':
                verts, faces, midx = [], [], []
                for ci, P in enumerate(comps):
                    v, f = kcg._tube_welded([tuple(p) for p in P],
                                            self.radius,
                                            self.tube_sides)
                    base = len(verts)
                    verts.extend(v)
                    faces.extend([[base + i for i in face]
                                  for face in f])
                    midx.extend([ci] * len(f))
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
                if color and len(me.polygons) == len(midx):
                    me.polygons.foreach_set('material_index', midx)
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                for P in comps:
                    if self.output == 'BEZIER':
                        sp = cu.splines.new('BEZIER')
                        sp.bezier_points.add(len(P) - 1)
                        for i, pnt in enumerate(P):
                            bp = sp.bezier_points[i]
                            bp.co = pnt
                            bp.handle_left_type = 'AUTO'
                            bp.handle_right_type = 'AUTO'
                    else:
                        sp = cu.splines.new('POLY')
                        sp.points.add(len(P) - 1)
                        for i, pnt in enumerate(P):
                            sp.points[i].co = (pnt[0], pnt[1],
                                               pnt[2], 1.0)
                    sp.use_cyclic_u = True
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            if color:
                import colorsys
                for ci in range(d):
                    rgb = colorsys.hsv_to_rgb(ci / d, 0.72, 0.9)
                    mat = bpy.data.materials.new(
                        "%s C%d" % (name, ci + 1))
                    mat.diffuse_color = (*rgb, 1.0)
                    mat.use_nodes = True
                    node = mat.node_tree.nodes.get(
                        "Principled BSDF")
                    if node:
                        node.inputs["Base Color"].default_value \
                            = (*rgb, 1.0)
                    obj.data.materials.append(mat)
                if self.output != 'MESH':
                    for ci, sp in enumerate(obj.data.splines):
                        sp.material_index = ci
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            frac = k['fraction']
            ftxt = (", p/q = %d/%d" % frac) if frac else ""
            self.report({'INFO'},
                        "%s: %d component(s), %d crossings%s"
                        % (name, d, k['ncross'], ftxt))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                lay.prop(self, 'family')
                lay.prop(self, 'notation')
            fam = (self.family if self.preset == 'CUSTOM'
                   else _PRESETS[self.preset][0])
            lay.prop(self, 'twist_len')
            if fam == 'PRETZEL':
                lay.prop(self, 'spread')
            for kname in ('samples', 'weave', 'output', 'radius'):
                lay.prop(self, kname)
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'color_components')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.rational_knot_add",
                             icon='MOD_CURVE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(CURVE_OT_rational_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_rational_knot_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _count_intersections(loops):
    """Total 2D crossings of the assembled diagram, found purely
    geometrically (pairwise + self segment intersections)."""
    tot = 0
    for i in range(len(loops)):
        tot += len(kcg._seg_cross(loops[i], loops[i])) // 2
        for j in range(i + 1, len(loops)):
            tot += len(kcg._seg_cross(loops[i], loops[j]))
    return tot


def _is_alternating(signed):
    for sg in signed:
        seq = [s for _i, s in sg]
        if len(seq) >= 2 and any(
                seq[i] == seq[(i + 1) % len(seq)]
                for i in range(len(seq))):
            return False
    return True


def _selftest():
    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-46s %s" % (label, "OK" if cond else "BAD"))
        return cond

    # -- rational / 2-bridge: fraction, crossings, components,
    #    alternation, geometric crossing count
    print("RATIONAL (2-bridge), convention "
          "p/q = a_n + 1/(a_(n-1) + ... + 1/a_1):")
    cases = [([3], (3, 1)), ([2], (2, 1)), ([2, 2], (5, 2)),
             ([3, 2], (7, 3)), ([2, 2, 2], (12, 5)),
             ([4, 2], (9, 4)), ([3, 2, 1], (10, 7))]
    for a, frac in cases:
        k = build_knot('RATIONAL', a, samples=8)
        p, q = k['fraction']
        want_comp = 2 if p % 2 == 0 else 1
        nplus = sum(1 for sg in k['signed'] for _i, s in sg if s > 0)
        nminus = sum(1 for sg in k['signed'] for _i, s in sg
                     if s < 0)
        geo = _count_intersections(k['loops2d'])
        check("[%s] -> %d/%d, %dx, %d comp" %
              (" ".join(map(str, a)), p, q, k['ncross'], k['ncomp']),
              (p, q) == frac
              and k['ncross'] == sum(abs(v) for v in a)
              and k['ncomp'] == want_comp
              and nplus == nminus == k['ncross']
              and geo == k['ncross']
              and (_is_alternating(k['signed'])
                   if all(v > 0 for v in a) else True))

    # -- pretzel: crossings, component counts, alternation
    print("PRETZEL:")
    pcases = [((3, 5, 7), 1), ((2, 2, 2), 3), ((-2, 3, 7), 1),
              ((1, 1), 2), ((3, 3), 2), ((2, 3, 7), 1)]
    for pl, want_comp in pcases:
        k = build_knot('PRETZEL', pl, samples=8)
        nplus = sum(1 for sg in k['signed'] for _i, s in sg if s > 0)
        nminus = sum(1 for sg in k['signed'] for _i, s in sg
                     if s < 0)
        geo = _count_intersections(k['loops2d'])
        check("P(%s): %dx, %d comp" %
              (",".join(map(str, pl)), k['ncross'], k['ncomp']),
              k['ncross'] == sum(abs(v) for v in pl)
              and k['ncomp'] == want_comp
              and nplus == nminus == k['ncross']
              and geo == k['ncross']
              and (_is_alternating(k['signed'])
                   if all(v > 0 for v in pl) else True))

    # -- twist knots C(n, 2) = rational [n, 2]
    print("TWIST:")
    for n, frac in ((1, (3, 1)), (2, (5, 2)), (3, (7, 3)),
                    (4, (9, 4)), (5, (11, 5))):
        k = build_knot('TWIST', [n], samples=8)
        check("n=%d -> %d/%d, %dx, %d comp" %
              (n, *k['fraction'], k['ncross'], k['ncomp']),
              k['fraction'] == frac and k['ncross'] == n + 2
              and k['ncomp'] == 1
              and _is_alternating(k['signed']))

    # -- mesh tube (figure-eight), centering, fit to the 2 m cube
    print("GEOMETRY:")
    k = build_knot('RATIONAL', [2, 2], samples=10)
    allp = np.vstack(k['loops'])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    check("centered, fits 2 m cube",
          np.all(lo >= -1.0 - 1e-9) and np.all(hi <= 1.0 + 1e-9)
          and abs(float((hi - lo).max()) - 2.0) < 1e-9
          and np.all(np.abs(lo + hi) < 1e-9))
    for P in k['loops']:
        v, f = kcg._tube_welded([tuple(p) for p in P], 0.05, 10)
        check("tube: %d verts, %d faces, all quads, finite"
              % (len(v), len(f)),
              len(v) > 0 and len(f) > 0
              and all(len(face) == 4 for face in f)
              and np.all(np.isfinite(np.asarray(v))))
    kp = build_knot('PRETZEL', [3, 5, 7], samples=8)
    allp = np.vstack(kp['loops'])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    check("pretzel centered, fits 2 m cube",
          np.all(lo >= -1.0 - 1e-9) and np.all(hi <= 1.0 + 1e-9)
          and abs(float((hi - lo).max()) - 2.0) < 1e-9)

    assert ok, "SELF-TEST FAILED"
    print("RESULT: OK")


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        _selftest()
