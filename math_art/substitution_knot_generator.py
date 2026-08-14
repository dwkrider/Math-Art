
# Substitution Fractal Knot Generator for Blender
#
# Fractal knots built by ITERATIVE SUBSTITUTION -- Robert Fathauer's
# method (Bridges 2007): start from a particular projection of a knot,
# redrawn so that some of the gaps in the diagram are similar in shape
# to the whole diagram; treat those gaps as SUBSTITUTION SLOTS and
# replace the strand's straight passage through each slot with a
# scaled-down (rotated / reflected as needed) copy of the ENTIRE
# diagram, splicing the copy's two free ends into the parent strand;
# iterate.  Every copy carries its own slots, so the limit of the
# process is a self-similar "knot within a knot within a knot" -- with
# infinitely many crossings a WILD knot.  The diagram is kept valence-4
# and ALTERNATING (over/under strictly alternates along the strand)
# throughout, exactly as Fathauer prescribes: crossings never collide
# when a copy is scaled in, and the splice joins the strands so the
# alternating weave is maintained.
#
# THE BASE MOTIFS here are the standard star projections of the (2,p)
# torus knots (p odd: trefoil 3_1, cinquefoil 5_1), drawn as the polar
# curve r = R + a cos(p theta / 2), theta in [0, 4 pi) -- a unicursal
# closed strand with p lobes and p alternating crossings.  The tips of
# the lobes are flattened into straight SLOT chords.  A spliced COPY
# opens one lobe into a GATE (its pair of leader ends): the tip arc of
# that lobe is removed and the copy's whole body then lies strictly on
# one side of the gate chord, so gluing the gate onto a parent slot
# chord -- by the direct or reflected similarity chosen automatically
# to hang the body on the empty outward side -- can never run the copy
# back through the parent strand.  The ROOT instance is closed and
# needs no gate, so it keeps slots on ALL p lobes ('bloom' presets:
# a trefoil whose three lobes each grow a smaller trefoil), while
# every copy branches through its p - 1 free lobes; 'clasp' presets
# gate the root too (m = p - 1 throughout).  The scale factor per
# level is the slot-chord : gate-chord ratio = `slot_scale`, and with
# m >= 2 slots per motif the substitution genuinely branches (m = 1
# would be Fathauer's "rather trivial" linear chain).
#
# OVER / UNDER.  Every crossing of the base motif lies at a control
# vertex the strand passes through twice, and similarities preserve
# that coincidence, so the substituted diagram's crossings are known
# exactly without any geometric intersection search.  A closed strand
# whose crossings are all self-crossings always admits exactly two
# alternating assignments (the checkerboard coloring of the diagram's
# faces); the assignment is resolved here as parity (XOR) constraints
# between consecutive crossings along the strand with the Islamic
# strapwork engine's parity union-find, and the self-test verifies
# both the consistency and the strict over/under alternation.
#
# STRAND WIDTH TELESCOPES: each copy's strand width (and its weave
# amplitude) shrinks by the copy's geometric scale factor -- or by an
# explicit per-level factor -- so deeper sub-knots are drawn with
# proportionally thinner strands, matching Fathauer's drawings.  The
# strand renders as a woven flat ribbon, an interlaced (cut) flat
# diagram, a round swept tube, or a bare centerline curve, reusing the
# strapwork ribbon / weave machinery and the welded-tube sweep.
#
# NOTE: this is the iterative-SUBSTITUTION construction, which is
# entirely distinct from the recursive CABLING construction of
# `fractal_knot_generator` (a satellite/cable operation on a space
# curve).  Substitution nests visible smaller copies of the whole
# diagram inside itself; cabling replaces the rope of one knot with a
# braided bundle.
#
# References:
#   Robert W. Fathauer, "Fractal Knots Created by Iterative
#     Substitution", Bridges 2007 Conference Proceedings, pp. 335-342.
#     https://archive.bridgesmathart.org/2007/bridges2007-335.html
#   Robert W. Fathauer, "A New Method for Designing Iterated Knots",
#     Bridges 2009 Conference Proceedings.
#   Robert W. Fathauer, "Compendium of Fractal Knots",
#     https://www.mathartfun.com/FractalKnots/index.html
#   (Earlier iterated knots that Fathauer credits: Carlo Sequin's 2005
#   examples and Paul Gailiunas' Sierpinski-gasket knot, Bridges 2006.)

bl_info = {
    "name": "Substitution Fractal Knot",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Knots & Curves",
    "description": "Fathauer iterative-substitution fractal knots -- "
                   "self-similar alternating knots within knots",
    "category": "Add Curve",
}

from math import pi, cos, sin, hypot

import numpy as np

try:                                  # inside the math_art package
    from .patterns.polygon2d import (arclen, unit)
    from .patterns.ribbon import (band_ribbon_faces, band_ribbon_faces_z, catmull_rom, miter)
    from .patterns.weave import (ParityDSU, weave_zoff)
except ImportError:                   # flat import (test runner)
    from patterns.polygon2d import (arclen, unit)
    from patterns.ribbon import (band_ribbon_faces, band_ribbon_faces_z, catmull_rom, miter)
    from patterns.weave import (ParityDSU, weave_zoff)

try:                                  # inside the math_art package
    from .curve_frames import welded_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import welded_tube

try:
    from .patterns import common as pc
    from . import islamic_pattern_generator as isl
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import islamic_pattern_generator as isl


# --------------------------------------------------------------------
# Planar similarities (complex-number form)
# --------------------------------------------------------------------
#
# A slot splice is the unique direct (or reflected) similarity mapping
# the motif's gate ends G0, G1 onto the slot chord ends E0, E1.  With
# points as complex numbers a direct similarity is z -> a z + b and a
# reflected one is z -> a conj(z) + b.

def _sim(g0, g1, e0, e1, reflect):
    """The similarity taking gate (g0, g1) to slot edge (e0, e1)."""
    if reflect:
        a = (e1 - e0) / (g1 - g0).conjugate()
        b = e0 - a * g0.conjugate()
    else:
        a = (e1 - e0) / (g1 - g0)
        b = e0 - a * g0
    return (a, b, reflect)


def _apply(T, z):
    a, b, reflect = T
    return a * (z.conjugate() if reflect else z) + b


def _side(e0, e1, z):
    """+1 / -1: which side of the directed line e0 -> e1 the point z
    lies on (sign of the 2D cross product)."""
    c = ((e1 - e0).conjugate() * (z - e0)).imag
    return 1 if c >= 0.0 else -1


# --------------------------------------------------------------------
# Base motifs: gated, slotted (2,p) torus-knot star diagrams
# --------------------------------------------------------------------

class _Motif:
    """A base diagram ready for substitution.

    pts      -- complex ndarray: the OPEN control strand G0 .. G1 (the
                closed standalone knot adds the straight gate chord
                pts[-1] -> pts[0]).
    cross    -- [(ia, ib)]: control-index pairs where the strand
                passes twice through the same point (the crossings).
    slot_of  -- {ia: True}: control indices starting a slot edge
                (ia -> ia+1 is the straight chord a copy replaces).
    g0, g1   -- the gate anchor points (pts[0], pts[-1]).
    centroid -- mean of pts (used for the body-side test).
    m        -- number of slots; scale -- achieved child/parent scale.
    """

    def __init__(self, pts, cross, slots, scale, cgate, name):
        self.pts = np.asarray(pts, complex)
        self.cross = list(cross)
        self.slot_of = {a: True for a in slots}
        self.g0 = complex(self.pts[0])
        self.g1 = complex(self.pts[-1])
        self.centroid = complex(self.pts.mean())
        self.m = len(slots)
        self.scale = float(scale)
        self.cgate = float(cgate)
        self.name = name


def _torus_motif(p, gate, slot_scale, spl=6, name='',
                 target_cgate=None, slotted=True):
    """Build the slotted star diagram of the (2,p) torus knot (p odd).
    The polar curve r = R + a cos(p t / 2), t in [0, 4 pi), has p
    lobes (tips at t = 4 pi k / p) and p crossings (the two passes t
    and t + 2 pi coincide where cos vanishes, i.e. at odd multiples of
    pi / p).  `gate` is 'TIP' (open lobe 0 into the gate; the p - 1
    other lobes carry slots -- the form every spliced copy takes) or
    'NONE' (the closed root form: no gate, slots on all p lobes;
    `target_cgate` must then supply the child's gate chord length).
    The slot half-width is solved so the slot chord is `slot_scale`
    times the gate chord, making slot_scale the exact child/parent
    scale factor per substitution level.  `slotted=False` builds the
    LEAF form -- same curve and gate but round, unflattened lobes --
    used wherever the recursion bottoms out and the slots would stay
    empty."""
    if p < 3 or p % 2 == 0:
        raise ValueError("p must be odd and >= 3 (unicursal)")
    q = 0.5 * p
    R, A = 1.0, 0.7
    spl = max(3, int(spl))
    step = pi / (p * spl)               # theta grid step
    N = 4 * p * spl                     # grid points per 4 pi period

    def P(th):
        r = R + A * cos(q * th)
        return complex(r * cos(th), r * sin(th))

    if gate == 'NONE':
        if target_cgate is None:
            raise ValueError("gate NONE needs the child gate chord")
        jg = 2 * p * spl                # start at an inner point
        gh = 0                          # no gate cut: a closed loop
        tips = list(range(p))
        c_gate = float(target_cgate)
    else:                               # 'TIP'
        jg = 0                          # theta = 0: lobe-0 tip
        gh = max(2, int(round(0.65 * spl)))
        tips = list(range(1, p))
        c_gate = abs(P((jg + gh) * step) - P((jg - gh) * step))

    # Solve the slot half-width ds so the slot chord (the flattened
    # lobe tip) is slot_scale * gate chord.  The chord grows
    # monotonically with ds, so bisect; ds stays well clear of the
    # crossings (at parameter distance pi/p from each tip).
    def chord(d):
        return abs(P(d) - P(-d))        # any tip, by symmetry

    if not slotted:
        tips = []                       # leaf form: round lobes
        ds, scale = 0.0, 0.0
    else:
        lo, hi = 0.3 * step, 0.80 * spl * step
        target = max(chord(lo), min(slot_scale * c_gate, chord(hi)))
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if chord(mid) < target:
                lo = mid
            else:
                hi = mid
        ds = 0.5 * (lo + hi)
        scale = chord(ds) / c_gate

    # Traversal: grid samples from jg+gh to jg+N-gh (the whole period
    # when there is no gate); each slot tip has exactly one occurrence
    # in that range, where nearby samples are dropped and the two
    # exact chord ends are inserted instead.
    j0 = jg + gh
    j1 = jg + N - gh - (1 if gate == 'NONE' else 0)
    slot_marks = []                     # (theta_tip,) in traversal
    for k in tips:
        jt = 4 * k * spl
        while jt < j0:
            jt += N
        if not (j0 < jt < j1):
            raise ValueError("slot tip falls inside the gate")
        slot_marks.append(jt * step)
    drop = ds + 0.4 * step
    events = []                         # (theta, kind, payload)
    for j in range(j0, j1 + 1):
        th = j * step
        if any(abs(th - tt) < drop for tt in slot_marks):
            continue
        events.append((th, 'v', j % N))
    for tt in slot_marks:
        events.append((tt - ds, 'a', None))
        events.append((tt + ds, 'b', None))
    events.sort(key=lambda e: e[0])

    pts, slots, gridmap = [], [], {}
    for th, kind, payload in events:
        idx = len(pts)
        pts.append(P(th))
        if kind == 'v':
            gridmap[payload] = idx
        elif kind == 'a':
            slots.append(idx)
    cross = []
    for k in range(p):
        ja = ((2 * k + 1) * spl) % N
        jb = (ja + 2 * p * spl) % N
        if ja not in gridmap or jb not in gridmap:
            raise ValueError("a crossing sample was dropped")
        cross.append((gridmap[ja], gridmap[jb]))
    return _Motif(pts, cross, slots, scale, c_gate, name)


# Preset table: name -> (p, style).  BLOOM presets keep the root
# closed with a slot on every one of its p lobes (copies still branch
# through p - 1); CLASP presets gate the root too (m = p - 1
# throughout).
_PRESETS = {
    'TREFOIL': (3, 'BLOOM'),        # every trefoil lobe sprouts
    'TREFOIL_TWIN': (3, 'CLASP'),   # binary-branching trefoil
    'CINQUEFOIL': (5, 'BLOOM'),     # every cinquefoil lobe sprouts
    'CINQUEFOIL_QUAD': (5, 'CLASP'),
}


def make_motif(base='TREFOIL', slot_scale=0.42, spl=6):
    """Motif kit for a preset: {'root', 'child', 'root0', 'child0'}.
    `child` is the TIP-gated form every spliced copy uses; `root` is
    the closed all-slots form (BLOOM) or the gated form closed by its
    gate chord (CLASP); the '0' variants are the round-lobed LEAF
    forms drawn where the recursion bottoms out."""
    p, style = _PRESETS[base]
    child = _torus_motif(p, 'TIP', slot_scale, spl, base)
    child0 = _torus_motif(p, 'TIP', slot_scale, spl, base,
                          slotted=False)
    if style == 'BLOOM':
        root = _torus_motif(p, 'NONE', slot_scale, spl, base,
                            target_cgate=child.cgate)
        root0 = _torus_motif(p, 'NONE', slot_scale, spl, base,
                             target_cgate=child.cgate, slotted=False)
    else:
        root, root0 = child, child0
    return {'root': root, 'child': child,
            'root0': root0, 'child0': child0}


# --------------------------------------------------------------------
# Iterative substitution (the knot IFS)
# --------------------------------------------------------------------

def _expand(motif, kit, depth, T, level, out):
    """Append one instance of `motif` under similarity T at recursion
    `level`, substituting a copy into every slot while depth > 0 (the
    slotted `child` form while it will branch again, the round-lobed
    leaf form at the bottom).  The splice similarity maps the child's
    gate onto the slot chord; direct vs reflected is chosen so the
    child's body lands on the side of the chord AWAY from this
    instance's own body (the empty outward side), keeping the copies
    clear of the parent strand."""
    g = abs(T[0])
    K = len(motif.pts)
    imap = np.empty(K, int)
    c_par = _apply(T, motif.centroid)
    for i in range(K):
        w = _apply(T, complex(motif.pts[i]))
        imap[i] = len(out['pts'])
        out['pts'].append(w)
        out['g'].append(g)
        out['lvl'].append(level)
        if i in motif.slot_of and depth > 0 and i + 1 < K:
            sub = kit['child'] if depth > 1 else kit['child0']
            e0 = w
            e1 = _apply(T, complex(motif.pts[i + 1]))
            S = _sim(sub.g0, sub.g1, e0, e1, False)
            if _side(e0, e1, _apply(S, sub.centroid)) == \
                    _side(e0, e1, c_par):
                S = _sim(sub.g0, sub.g1, e0, e1, True)
            _expand(sub, kit, depth - 1, S, level + 1, out)
    for (ia, ib) in motif.cross:
        out['cross'].append((int(imap[ia]), int(imap[ib]), g, level))


def _dedupe(out):
    """Drop consecutive duplicate points (the shared splice ends where
    a copy's leaders meet the parent strand), remapping crossings."""
    pts = out['pts']
    n = len(pts)
    keep_pts, keep_g, keep_lvl = [], [], []
    remap = np.empty(n, int)
    for i in range(n):
        if keep_pts and abs(pts[i] - keep_pts[-1]) < 1e-9:
            remap[i] = len(keep_pts) - 1
            continue
        remap[i] = len(keep_pts)
        keep_pts.append(pts[i])
        keep_g.append(out['g'][i])
        keep_lvl.append(out['lvl'][i])
    cross = [(int(remap[ia]), int(remap[ib]), g, lv)
             for (ia, ib, g, lv) in out['cross']]
    return keep_pts, keep_g, keep_lvl, cross


def instance_count(base, depth):
    """Motif instances after `depth` substitutions (root included):
    the root, its m_root children, and m_child-fold branching below --
    so crossings grow by the substitution factor per level (the
    self-similarity the self-test checks)."""
    p, style = _PRESETS[base]
    m_root = p if style == 'BLOOM' else p - 1
    m_child = p - 1
    total, n = 1, m_root
    for _k in range(1, max(0, int(depth)) + 1):
        total += n
        n *= m_child
    return total


def build_diagram(base='TREFOIL', depth=3, slot_scale=0.42, spl=6):
    """The substituted diagram: a dict with
    pts   -- complex list, the single CLOSED control strand (the
             closing edge is pts[-1] -> pts[0]),
    g     -- per-point instance scale factor (1.0 at the root),
    lvl   -- per-point substitution level (0 at the root),
    cross -- [(ia, ib, g, lvl)] crossing control-index pairs with the
             owning instance's scale and level,
    motif -- the root motif;  components is always 1 (unicursal)."""
    kit = make_motif(base, slot_scale, spl)
    top = kit['root'] if depth > 0 else kit['root0']
    out = {'pts': [], 'g': [], 'lvl': [], 'cross': []}
    _expand(top, kit, depth, (1.0 + 0.0j, 0.0 + 0.0j, False), 0, out)
    pts, g, lvl, cross = _dedupe(out)
    return {'pts': pts, 'g': g, 'lvl': lvl, 'cross': cross,
            'motif': kit['root'], 'depth': depth}


# --------------------------------------------------------------------
# Alternating over/under (parity union-find over the strand)
# --------------------------------------------------------------------

def solve_alternation(diagram):
    """Assign over/under so the weave alternates along the strand.
    For crossing c with passages at control indices (ia, ib), let
    x_c = 1 mean "the ia passage rides over".  Walking the strand,
    consecutive passages must alternate, giving the parity constraint
    x_c XOR x_c' = 1 XOR k XOR k' between neighboring passages (k = 0
    for an ia passage, 1 for an ib passage), solved with the strapwork
    engine's parity union-find.  Returns (signed, ok): `signed` is
    [(control_index, +1 over / -1 under, amp_scale, level)] sorted
    along the strand; ok is False on a parity clash (never happens for
    a closed strand of self-crossings -- the checkerboard property --
    but verified anyway)."""
    cross = diagram['cross']
    dsu = ParityDSU()
    passages = []
    for cid, (ia, ib, g, lv) in enumerate(cross):
        dsu.add(cid)
        passages.append((ia, cid, 0, g, lv))
        passages.append((ib, cid, 1, g, lv))
    passages.sort()
    ok = True
    P = len(passages)
    for t in range(P):
        _i0, c0, k0, _g0, _l0 = passages[t]
        _i1, c1, k1, _g1, _l1 = passages[(t + 1) % P]
        ok = dsu.union(c0, c1, 1 ^ k0 ^ k1) and ok
    signed = []
    for (idx, cid, k, g, lv) in passages:
        _root, x = dsu.find(cid)
        sense = x ^ k                   # 1 = over here
        signed.append((idx, 1 if sense else -1, g, lv))
    return signed, ok


# --------------------------------------------------------------------
# Rendering: telescoped ribbon / tube / curve
# --------------------------------------------------------------------

def _miter_ribbon_var(points, widths, closed, limit=4.0):
    """Per-vertex-width variant of the strapwork miter ribbon: offset
    the polyline left/right by widths[i] / 2 with mitered joints, so
    the strand can telescope thinner into the deeper copies."""
    V = list(points)
    k = len(V)
    left, right = [], []
    if closed:
        d = [unit(V[(i + 1) % k][0] - V[i][0],
                       V[(i + 1) % k][1] - V[i][1]) for i in range(k)]
        for i in range(k):
            dp, dn = d[(i - 1) % k], d[i]
            mx, my, s = miter((-dp[1], dp[0]), (-dn[1], dn[0]),
                                   limit)
            w = 0.5 * widths[i] * s
            left.append((V[i][0] + mx * w, V[i][1] + my * w))
            right.append((V[i][0] - mx * w, V[i][1] - my * w))
    else:
        d = [unit(V[i + 1][0] - V[i][0], V[i + 1][1] - V[i][1])
             for i in range(k - 1)]
        for i in range(k):
            if i == 0:
                dn = d[0]
                mx, my, s = -dn[1], dn[0], 1.0
            elif i == k - 1:
                dp = d[-1]
                mx, my, s = -dp[1], dp[0], 1.0
            else:
                dp, dn = d[i - 1], d[i]
                mx, my, s = miter((-dp[1], dp[0]),
                                       (-dn[1], dn[0]), limit)
            w = 0.5 * widths[i] * s
            left.append((V[i][0] + mx * w, V[i][1] + my * w))
            right.append((V[i][0] - mx * w, V[i][1] - my * w))
    return left, right


def _smooth_width(wid, step, blend):
    """Blend the per-vertex strand width so it TAPERS gradually from one
    substitution level to the next instead of stepping abruptly at each
    splice.  The window is measured in control-segments (blend * a few) --
    and because catmull_rom lays down a fixed number of points per
    segment, the taper length scales with the local copy: a big-to-medium
    join blends over a long arc, a medium-to-small join over a
    proportionally shorter one.  blend = 0 restores the hard steps."""
    if blend <= 0.0:
        return wid
    n = len(wid)
    half = max(1, min(int(round(blend * 2.0 * max(1, step))), n // 2 - 1))
    if half < 1:
        return wid
    kern = np.ones(2 * half + 1) / (2 * half + 1)
    w = np.asarray(wid, float)
    for _ in range(2):                      # twice -> smooth (triangular)
        pad = np.concatenate([w[-half:], w, w[:half]])
        w = np.convolve(pad, kern, mode='valid')
    return w


def _prepare(base, depth, slot_scale, strand_width, telescope, samples,
             weave_height, spl=6, width_blend=0.0):
    """Shared geometry prep: the densified path, per-path-vertex width
    and level, the signed weave anchors, and the z offsets.  Widths
    and weave amplitudes telescope by each instance's true geometric
    scale (telescope = 0, AUTO) or by telescope**level; `width_blend`
    tapers the width smoothly across the level joins."""
    diagram = build_diagram(base, depth, slot_scale, spl)
    signed, ok = solve_alternation(diagram)
    pts = diagram['pts']
    n = len(pts)
    ctrl = [(z.real, z.imag) for z in pts]

    def fac(g, lv):
        return g if telescope <= 0.0 else telescope ** lv

    wctrl = np.array([strand_width * fac(g, lv)
                      for g, lv in zip(diagram['g'], diagram['lvl'])])
    subdiv = max(1, int(samples))
    step = subdiv if (subdiv >= 2 and n >= 3) else 1
    path = catmull_rom(ctrl, True, subdiv) if step > 1 else ctrl
    m = len(path)
    if step > 1:                        # lerp widths along each segment
        t = np.arange(step) / float(step)
        wnext = np.roll(wctrl, -1)
        wid = (wctrl[:, None] * (1.0 - t)[None, :]
               + wnext[:, None] * t[None, :]).reshape(-1)
    else:
        wid = wctrl.copy()
    wid = _smooth_width(wid, step, width_blend)
    lvlp = np.repeat(np.asarray(diagram['lvl'], int), step)
    anchors = [(idx * step, sg * weave_height * fac(g, lv))
               for (idx, sg, g, lv) in signed]
    zoff = weave_zoff(path, True, anchors, 1.0) if anchors \
        else [0.0] * m
    return {'diagram': diagram, 'signed': signed, 'alt_ok': ok,
            'path': path, 'wid': wid, 'lvl': lvlp, 'zoff': zoff,
            'step': step}


def _ribbon_mats(span, relief, closed, lvl_of):
    """Face materials parallel to band_ribbon_faces(_z) output: `span`
    top quads, then (with relief) span (bottom, left, right) triples
    and two end caps when open."""
    mats = [lvl_of(i) for i in range(span)]
    if relief:
        for i in range(span):
            mats.extend([lvl_of(i)] * 3)
        if not closed:
            mats.extend([lvl_of(0), lvl_of(span - 1)])
    return mats


def _cut_runs(path, wid, lvl, zoff, signed_paths):
    """FLAT interlace: remove a gap from the strand around every UNDER
    passage (gap radius ~ the local strand width, which telescopes),
    returning the kept runs as open pieces of (pts, wid, lvl)."""
    m = len(path)
    s, total = arclen(path, True)
    keep = np.ones(m, bool)
    sa = np.asarray(s)
    for (pi, sg) in signed_paths:
        if sg >= 0:
            continue
        h = 1.05 * wid[pi]
        d = np.abs(sa - s[pi])
        d = np.minimum(d, total - d)    # cyclic arclength distance
        keep &= d > h
    if keep.all():
        return [(list(path), wid, lvl, zoff, True)]
    # rotate so index 0 is removed, then take maximal kept runs
    start = int(np.argmin(keep))
    order = [(start + i) % m for i in range(m)]
    runs, cur = [], []
    for i in order:
        if keep[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        if len(run) < 2:
            continue
        out.append(([path[i] for i in run],
                    np.array([wid[i] for i in run]),
                    np.array([lvl[i] for i in run]),
                    [zoff[i] for i in run], False))
    return out


# Path-point budgets (before the tube/ribbon multiplies them) -- the
# depth is clamped so a deep, high-branching preset cannot explode.
_BUDGET = {'TUBE': 60000, 'RIBBON': 240000, 'CURVE': 240000}


def clamp_depth(base, depth, samples, output, spl=6):
    """Largest depth <= `depth` whose densified path stays within the
    output's point budget.  Returns (depth, clamped?)."""
    p, _style = _PRESETS[base]
    K = 4 * p * spl                     # ~ control points per instance
    budget = _BUDGET.get(output, 240000)
    d = max(0, int(depth))
    while d > 0 and \
            K * instance_count(base, d) * max(1, samples) > budget:
        d -= 1
    return d, d != int(depth)


def build_cells(base='TREFOIL', depth=3, slot_scale=0.42,
                strand_width=0.11, telescope=0.0, samples=6,
                output='TUBE', mode='WOVEN', weave_height=0.09,
                height=0.0, tube_sides=12, color_by='LEVEL', spl=6,
                width_blend=0.0):
    """Build the fractal knot as (verts, faces, mats) cells plus an
    info dict.  TUBE sweeps a round welded tube along the woven 3D
    centerline with per-ring telescoped radius; RIBBON is the flat
    strapwork ribbon, WOVEN (z-undulating) or FLAT (under-strand cut
    at each crossing); colors follow the substitution level so each
    scale of the fractal reads in its own hue."""
    depth, clamped = clamp_depth(base, depth, samples, output, spl)
    prep = _prepare(base, depth, slot_scale, strand_width, telescope,
                    samples, weave_height, spl, width_blend)
    path, wid, lvl, zoff = (prep['path'], prep['wid'], prep['lvl'],
                            prep['zoff'])
    info = {'crossings': len(prep['diagram']['cross']),
            'points': len(path), 'depth': depth, 'clamped': clamped,
            'components': 1, 'alt_ok': prep['alt_ok'],
            'm': prep['diagram']['motif'].m,
            'scale': prep['diagram']['motif'].scale}

    def mat_of(level):
        if color_by == 'LEVEL':
            return int(level) % len(pc.PALETTE_RGBA)
        return 0

    cells = []
    if output == 'TUBE':
        # 3D woven centerline; per-ring radius telescopes with the
        # local strand width (unit-radius sweep, rings rescaled).
        pts3 = np.array([(x, y, z)
                         for (x, y), z in zip(path, zoff)])
        verts, faces = welded_tube(pts3, 1.0, int(tube_sides))
        if not faces:
            return [], info
        n = len(pts3)
        sides = int(tube_sides)
        V = np.asarray(verts).reshape(n, sides, 3)
        rad = 0.5 * wid
        V = pts3[:, None, :] + rad[:, None, None] * (V - pts3[:, None, :])
        verts = [tuple(v) for v in V.reshape(-1, 3)]
        mats = []
        for i in range(n):
            mats.extend([mat_of(lvl[i])] * sides)
        cells.append((verts, [tuple(f) for f in faces], mats))
    elif mode == 'FLAT':
        signed_paths = [(idx * prep['step'], sg)
                        for (idx, sg, _g, _lv) in prep['signed']]
        for (sp, sw, sl, _sz, closed) in _cut_runs(path, wid, lvl,
                                                   zoff, signed_paths):
            left, right = _miter_ribbon_var(sp, sw, closed)
            cv, cf = band_ribbon_faces(left, right, closed, height)
            if not cf:
                continue
            span = len(sp) if closed else len(sp) - 1
            mats = _ribbon_mats(span, height > 0.0, closed,
                                lambda i, sl=sl: mat_of(sl[i]))
            cells.append((cv, cf, mats))
    else:                               # WOVEN ribbon
        left, right = _miter_ribbon_var(path, wid, True)
        cv, cf = band_ribbon_faces_z(left, right, True, height,
                                         zoff)
        if cf:
            span = len(path)
            mats = _ribbon_mats(span, height > 0.0, True,
                                lambda i: mat_of(lvl[i]))
            cells.append((cv, cf, mats))
    return cells, info


def curve_path(base='TREFOIL', depth=3, slot_scale=0.42,
               strand_width=0.11, telescope=0.0, samples=6,
               weave_height=0.09, spl=6, width_blend=0.0):
    """The woven 3D centerline as one closed polyline (CURVE output),
    plus the info dict."""
    depth, clamped = clamp_depth(base, depth, samples, 'CURVE', spl)
    prep = _prepare(base, depth, slot_scale, strand_width, telescope,
                    samples, weave_height, spl, width_blend)
    pts3 = [(x, y, z) for (x, y), z in zip(prep['path'], prep['zoff'])]
    info = {'crossings': len(prep['diagram']['cross']),
            'points': len(pts3), 'depth': depth, 'clamped': clamped,
            'components': 1, 'alt_ok': prep['alt_ok'],
            'm': prep['diagram']['motif'].m,
            'scale': prep['diagram']['motif'].scale}
    return pts3, info


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _shade_smooth(obj):
        if obj is not None and obj.type == 'MESH':
            for poly in obj.data.polygons:
                poly.use_smooth = True
            obj.data.update()

    def _emit_curve(context, name, pts3, span=2.0, operator=None):
        """One cyclic POLY spline along the woven centerline, centered
        and scaled into the 2 m cube."""
        a = np.asarray(pts3, float)
        lo, hi = a.min(axis=0), a.max(axis=0)
        c = 0.5 * (lo + hi)
        s = span / max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        sp = cu.splines.new('POLY')
        sp.points.add(len(a) - 1)
        for i, v in enumerate(a):
            sp.points[i].co = ((v[0] - c[0]) * s, (v[1] - c[1]) * s,
                               (v[2] - c[2]) * s, 1.0)
        sp.use_cyclic_u = True
        obj = bpy.data.objects.new(name, cu)
        context.collection.objects.link(obj)
        if operator is not None:
            from bpy_extras import object_utils
            obj.matrix_world = object_utils.add_object_align_init(
                context, operator)
        else:
            obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class CURVE_OT_substitution_knot_add(bpy.types.Operator,
                                         AddObjectHelper):
        """Add a Fathauer iterative-substitution fractal knot"""
        bl_idname = "curve.substitution_knot_add"
        bl_label = "Substitution Fractal Knot"
        bl_options = {'REGISTER', 'UNDO'}

        base: EnumProperty(
            name="Base Knot",
            items=[('TREFOIL', "Trefoil (3 slots)",
                    "Trefoil 3_1; every lobe grows a smaller trefoil"),
                   ('TREFOIL_TWIN', "Trefoil Clasp (2 slots)",
                    "Trefoil 3_1 opened at one lobe; the two free "
                    "lobes grow smaller trefoils"),
                   ('CINQUEFOIL', "Cinquefoil (5 slots)",
                    "Cinquefoil 5_1; every lobe grows a smaller "
                    "cinquefoil"),
                   ('CINQUEFOIL_QUAD', "Cinquefoil Clasp (4 slots)",
                    "Cinquefoil 5_1 opened at one lobe; four lobes "
                    "grow smaller cinquefoils")],
            default='TREFOIL',
            description="The base alternating diagram that is "
                        "substituted into its own slots")
        depth: IntProperty(
            name="Depth", default=3, min=0, max=4,
            description="Substitution iterations (0 = the plain base "
                        "knot); clamped if the mesh would explode")
        slot_scale: FloatProperty(
            name="Slot Scale", default=0.42, min=0.20, max=0.60,
            description="Child/parent scale factor per substitution "
                        "level (the slot chord relative to the gate)")
        strand_width: FloatProperty(
            name="Strand Width", default=0.11, min=0.02, max=0.40,
            description="Root strand width (ribbon width / tube "
                        "diameter); telescopes into deeper copies")
        telescope: FloatProperty(
            name="Width Telescope", default=0.0, min=0.0, max=1.0,
            description="Per-level width factor; 0 = automatic "
                        "(follow each copy's true geometric scale)")
        width_blend: FloatProperty(
            name="Width Blend", default=0.5, min=0.0, max=1.0,
            description="Taper the strand width gradually across the "
                        "joins between nesting levels; 0 = abrupt steps")
        samples: IntProperty(
            name="Samples", default=6, min=2, max=16,
            description="Catmull-Rom subdivisions per control segment")
        output: EnumProperty(
            name="Output",
            items=[('TUBE', "Round Tube",
                    "Welded round tube along the woven centerline"),
                   ('RIBBON', "Ribbon",
                    "Flat strapwork ribbon (woven or interlace-cut)"),
                   ('CURVE', "Centerline Curve",
                    "The woven centerline as a curve object")],
            default='TUBE')
        interlace_mode: EnumProperty(
            name="Interlace",
            items=[('WOVEN', "Woven (3D)",
                    "Lift/dip the strand at each crossing"),
                   ('FLAT', "Flat Knotwork",
                    "Cut the under-strand at each crossing (ribbon "
                    "output only; tubes always weave)")],
            default='WOVEN')
        weave_height: FloatProperty(
            name="Weave Height", default=0.09, min=0.0, max=0.5,
            description="Z amplitude of the weave at a root-level "
                        "crossing; telescopes with depth")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=0.5,
            description="0 = flat ribbon; > 0 extrudes the ribbon "
                        "(ribbon output)")
        tube_sides: IntProperty(
            name="Tube Sides", default=12, min=3, max=32,
            description="Facets around the tube cross-section")
        color_by: EnumProperty(
            name="Color By",
            items=[('LEVEL', "Substitution Level",
                    "A palette color per nesting depth"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='LEVEL')

        def execute(self, context):
            if self.output == 'CURVE':
                pts3, info = curve_path(
                    self.base, self.depth, self.slot_scale,
                    self.strand_width, self.telescope, self.samples,
                    self.weave_height, width_blend=self.width_blend)
                obj = _emit_curve(context, "Substitution Knot", pts3,
                                  operator=self)
            else:
                cells, info = build_cells(
                    self.base, self.depth, self.slot_scale,
                    self.strand_width, self.telescope, self.samples,
                    self.output, self.interlace_mode,
                    self.weave_height, self.height, self.tube_sides,
                    self.color_by, width_blend=self.width_blend)
                obj = pc.emit(context, "Substitution Knot", cells,
                              separate=False, fit=True, operator=self)
                if obj is not None and self.output == 'TUBE':
                    _shade_smooth(obj)
            if obj is None:
                self.report({'ERROR'}, "no knot generated")
                return {'CANCELLED'}
            if info['clamped']:
                self.report({'WARNING'},
                            "depth clamped to %d (mesh size)"
                            % info['depth'])
            if obj.type == 'MESH':
                self.report({'INFO'},
                            "%s depth %d  %d crossings  V=%d F=%d"
                            % (self.base, info['depth'],
                               info['crossings'],
                               len(obj.data.vertices),
                               len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s depth %d  %d crossings"
                            % (self.base, info['depth'],
                               info['crossings']))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'base')
            lay.prop(self, 'depth')
            lay.prop(self, 'slot_scale')
            lay.prop(self, 'strand_width')
            lay.prop(self, 'telescope')
            lay.prop(self, 'width_blend')
            lay.prop(self, 'samples')
            lay.prop(self, 'output')
            if self.output == 'RIBBON':
                lay.prop(self, 'interlace_mode')
                lay.prop(self, 'height')
            if self.output == 'TUBE':
                lay.prop(self, 'tube_sides')
            if self.output != 'CURVE' and \
                    (self.output == 'TUBE'
                     or self.interlace_mode == 'WOVEN'):
                lay.prop(self, 'weave_height')
            if self.output != 'CURVE':
                lay.prop(self, 'color_by')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("curve.substitution_knot_add",
                             icon='CURVE_NCURVE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_substitution_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_substitution_knot_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _proper_pairs(pts):
    """Count segment pairs of the closed control polyline that cross
    PROPERLY (in their open interiors).  Crossings meet at a shared
    vertex, giving zero cross products there, so a correct embedded
    diagram scores exactly zero."""
    P = np.asarray([(z.real, z.imag) for z in pts])
    n = len(P)
    Q = np.roll(P, -1, axis=0)
    eps = 1e-9
    bad = 0
    for i in range(n - 2):
        j1 = n if i > 0 else n - 1      # skip the two adjacent pairs
        C = P[i + 2:j1]
        D = Q[i + 2:j1]
        if len(C) == 0:
            continue
        A, B = P[i], Q[i]
        ab = B - A
        d1 = ab[0] * (C[:, 1] - A[1]) - ab[1] * (C[:, 0] - A[0])
        d2 = ab[0] * (D[:, 1] - A[1]) - ab[1] * (D[:, 0] - A[0])
        cd = D - C
        d3 = (cd[:, 0] * (A[1] - C[:, 1])
              - cd[:, 1] * (A[0] - C[:, 0]))
        d4 = (cd[:, 0] * (B[1] - C[:, 1])
              - cd[:, 1] * (B[0] - C[:, 0]))
        bad += int(np.count_nonzero((d1 * d2 < -eps)
                                    & (d3 * d4 < -eps)))
    return bad


def _selftest():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL:", msg)

    for base, (p, style) in _PRESETS.items():
        m_root = p if style == 'BLOOM' else p - 1
        print("--- %s (p=%d, %s, %d root slots) ---"
              % (base, p, style, m_root))
        max_d = 3 if p <= 3 else 2
        for depth in range(0, max_d + 1):
            dia = build_diagram(base, depth, slot_scale=0.42)
            pts = dia['pts']
            n = len(pts)
            arr = np.asarray([(z.real, z.imag) for z in pts])
            # finite, bounded, closed strand of distinct points
            check(np.all(np.isfinite(arr)), "non-finite points")
            check(float(np.abs(arr).max()) < 10.0, "diagram blew up")
            seg = np.roll(arr, -1, 0) - arr
            check(float(np.hypot(seg[:, 0], seg[:, 1]).min()) > 1e-6,
                  "consecutive duplicate points")
            # self-similar crossing growth: C0 * instances
            expect = p * instance_count(base, depth)
            check(len(dia['cross']) == expect,
                  "crossing count %d != %d" % (len(dia['cross']),
                                               expect))
            # every crossing is a genuine coincident pair
            for (ia, ib, _g, _lv) in dia['cross']:
                check(abs(pts[ia] - pts[ib]) < 1e-9,
                      "crossing passes do not coincide")
                check(ia != ib, "degenerate crossing")
            # alternating weave: parity solve consistent, one over +
            # one under per crossing, strict alternation on the strand
            signed, alt_ok = solve_alternation(dia)
            check(alt_ok, "parity clash in alternation solve")
            sense = {}
            for (idx, sg, _g, _lv) in signed:
                sense[idx] = sg
            for (ia, ib, _g, _lv) in dia['cross']:
                check(sense[ia] == -sense[ib],
                      "crossing not one-over-one-under")
            order = [sg for (_idx, sg, _g, _lv) in signed]
            for t in range(len(order)):
                check(order[t] == -order[(t + 1) % len(order)],
                      "weave does not alternate along the strand")
            # embedding: no unrecorded (proper) strand intersections
            if n <= 4000:
                check(_proper_pairs(pts) == 0,
                      "unexpected strand self-intersection")
            print("depth %d: %5d ctrl pts, %4d crossings, "
                  "1 component, scale %.3f"
                  % (depth, n, len(dia['cross']),
                     dia['motif'].scale))

    # mesh outputs at the default (depth-3 trefoil-in-trefoil)
    for output, mode in (('TUBE', 'WOVEN'), ('RIBBON', 'WOVEN'),
                         ('RIBBON', 'FLAT')):
        cells, info = build_cells('TREFOIL', 3, output=output,
                                  mode=mode, height=0.03)
        check(len(cells) > 0, "%s/%s produced no cells"
              % (output, mode))
        verts = [v for (cv, cf, cm) in cells for v in cv]
        faces = [f for (cv, cf, cm) in cells for f in cf]
        check(len(faces) > 0, "no faces")
        va = np.asarray(verts)
        check(np.all(np.isfinite(va)), "non-finite mesh")
        check(all(len(f) == 4 for f in faces), "non-quad faces")
        for (cv, cf, cm) in cells:
            check(len(cm) == len(cf), "mats/faces length mismatch")
            check(max(max(f) for f in cf) < len(cv),
                  "face index out of range")
        # fits the 2 m cube once centered/scaled (the emit convention)
        fit = np.asarray(pc.center_scale([tuple(v) for v in verts]))
        check(float(np.abs(fit).max()) <= 1.0 + 1e-6,
              "does not fit the 2 m cube")
        print("%s/%s: V=%d F=%d  crossings=%d  fits 2m cube"
              % (output, mode, len(verts), len(faces),
                 info['crossings']))

    pts3, info = curve_path('TREFOIL', 3)
    check(len(pts3) > 100, "curve output too short")
    check(all(np.isfinite(v).all() for v in np.asarray(pts3)),
          "non-finite curve")
    print("CURVE: %d points, %d crossings" % (len(pts3),
                                              info['crossings']))
    print("RESULT: %s" % ("OK" if ok else "FAILED"))
    return ok
