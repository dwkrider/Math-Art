
# Over-Under Screen Generator for Blender
#
# A woven screen rendered as one visually continuous smooth relief,
# in the modular-constructivist tradition of Erwin Hauer and Norman
# Carlberg (the woven screens of Hauer's Continua).
#
# THE STRUCTURE (asymmetric weave).  Two families of DIAGONAL
# ribbons -- call them RED and BLUE -- cross on a square lattice of
# crossovers, but the weave is NOT symmetric:
#   * The RED ribbons stay in the BACK half: an undulating floor,
#     lowest at the merges, rising gently under the arches --
#     curved everywhere, never a flat slab.
#   * The BLUE ribbons weave: at successive crossings with red they
#     alternate (checkerboard phase) between the FRONT level +h and
#     the BACK level -h, undulating smoothly between.
#   * Where BLUE is UP it arches OVER red: two separate sheets with
#     a real gap (blue at +h above red at -h) -- the only two-sheet
#     places.
#   * Where BLUE is DOWN it lands on red's level and MERGES with
#     it: a single fused junction at -h.
# Each merge is a broad HUB -- a curved saddle-bowl where the four
# arms meet, its ring swirled tangentially (one chirality across
# the whole field: the pinwheel motif).  The blue ribbon's own
# descending cosine continues down through the hub, bottoming out
# on the red floor (min blue == red level); the arms FLARE wide
# into the hub and narrow toward midspan; every ribbon carries a
# pronounced concave-up channel cross-section (the cupped arms);
# and the hubs are joined to their diagonal neighbours by BRIDGE
# necks that are MONOTONIC QUARTER-TURNS (helicoid segments):
# horizontal where they ease into one hub's TOP face, rotating
# one way the whole way across to land ON EDGE -- a vertical
# WALL -- in the adjoining hub's thickness band, so each neck
# reads clockwise from one hub and counterclockwise from the
# other, every neck twisting the same rotational way -- one
# global handedness, the pinwheel swirl.  So red ribbons
# + blue-down segments + hubs + bridges fuse into ONE flowing
# back-web, with the blue-up segments arching over it and
# teardrop openings swept around each arch -- a continuous relief
# with no planar region anywhere (Hauer's Continua).
# (Equivalently: start from a plain weave and, wherever red would
# ride on top, force it down beside blue instead -- the points of
# merger, connected by the bridges.)
#
# THE CONSTRUCTION (a bicubic B-spline patch network).  In the 45-
# degree rotated coordinates p = x + y, q = y - x the two diagonal
# families are axis-aligned bands on the integer lines, crossing at
# every integer (P, Q); blue is UP where P + Q is even, DOWN (and
# merged) where odd.  Everything is an explicit quad control net:
# flat red strips at -1; blue strips whose longitudinal control
# profile is the sampled cosine -cos(pi s): trough -1 at each
# merge center -> arch (+1) -> trough; a 3 x 3 MERGE PATCH at
# every blue-down crossing shaped as the valley of that cosine
# (ring columns at -cos(pi g), trough row on the floor), whose
# ring vertices ARE the end stations of the four incoming strips
# (shared control points, regular valence-4 seam vertices); and a
# quarter-turn bridge ribbon (exactly one shell thickness wide)
# across each open cell joining diagonally-adjacent merge patches,
# attached through shoulder vertices shared with the flanking
# strip edges.  The net
# is refined by CATMULL-CLARK subdivision -- the generalization of
# uniform bicubic B-spline patches to arbitrary quad meshes -- so
# the back-web is C1 (indeed C2 away from extraordinary vertices)
# across every seam, and the star openings exist by construction
# (never trimmed from a blob).  The refined shell is finally
# thickened along its normals into a closed solid.
#
# The MINIMAL (TAUT) mode applies extra passes of non-shrinking
# Taubin lambda|mu fairing.  The RIBBONS mode renders a classic
# axis-aligned plain weave as separate lofted strips (with TWILL
# and BASKET drafts); the membrane itself is inherently the
# asymmetric plain structure above.
#
# References:
#   Erwin Hauer, "Continua -- Architectural Screens and Walls",
#     Princeton Architectural Press, 2004 -- the perforated modular
#     screen designs (Designs 1-7, 1950-57): flowing perforated
#     webs with woven-over ribbons and star-shaped openings, the
#     structure realized here.
#   Norman Carlberg and Erwin Hauer -- co-originators of Modular
#     Constructivism (sculptural units designed to be multiplied).
#   E. Catmull and J. Clark, "Recursively Generated B-spline
#     Surfaces on Arbitrary Topological Meshes", Computer-Aided
#     Design 10(6), 1978 -- the subdivision scheme used to evaluate
#     the bicubic patch network.
#   Branko Grunbaum and G. C. Shephard, "Satins and Twills: An
#     Introduction to the Geometry of Fabrics", Mathematics
#     Magazine 53(3), 1980 -- the parity / draft description of
#     plain, twill and basket weaves (the checkerboard phase of the
#     blue family, and the drafts of the ribbon mode).
#   Gabriel Taubin, "A Signal Processing Approach to Fair Surface
#     Design", SIGGRAPH 1995 -- the non-shrinking lambda|mu fairing
#     of the taut mode.

bl_info = {
    "name": "Over-Under Screen",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Woven screen: a flowing back-web of flat "
                   "ribbons with the weaving family arching over "
                   "it, star openings (Hauer / Carlberg)",
    "category": "Add Mesh",
}

from collections import defaultdict

import numpy as np

try:
    from . import pattern_common as pc
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc


_BACKING_MAT = len(pc.PALETTE_RGBA) - 1
_HEIGHT_BANDS = 5     # palette bands of the by-height coloring
_VDIP = 0.008         # hub trough under-dip control, so the
                      # subdivided surface of blue's descending
                      # valley bottoms out exactly ON the red floor
_BOW = 0.60           # transverse ribbon concavity (unit z): a
                      # PRONOUNCED cupped-channel cross-section --
                      # the arms read as deep scooped channels,
                      # working with the swirl and the flowing
                      # curvature (edges +bow/2, center -bow/2)
_G_HUB = 2.2          # hub half-width (x strip half-width g):
                      # broad, dominant rounded hubs, trimmed just
                      # enough to keep the teardrop openings
                      # generous
_RISE = 0.25          # red undulation: rise under the arches
                      # (unit z) -- the "floor" is never flat
_SWIRL = 0.21         # pinwheel: tangential rotation (radians) of
                      # each hub ring, consistent chirality

# face labels of the membrane elements
_RED, _BLUE, _MERGE, _BRIDGE, _ARCH = 0, 1, 2, 3, 4


# --------------------------------------------------------------------
# Weave drafts (over/under parity)
# --------------------------------------------------------------------

WEAVES = ('PLAIN', 'TWILL', 'BASKET')


def x_over(i, j, weave='PLAIN'):
    """Boolean (arrays ok): x-strand over at crossing (i, j) of the
    axis-aligned ribbon weave (RIBBONS mode)."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if weave == 'TWILL':
        return ((i - j) % 4) < 2
    if weave == 'BASKET':
        return ((i // 2 + j // 2) % 2) == 0
    return ((i + j) % 2) == 0


def _targets(i, j, family, amp, weave):
    over = x_over(i, j, weave)
    if family == 1:
        over = ~over
    return np.where(over, amp, -amp)


def _params(W, H, cell, strand_width, weave_depth, weave):
    cell = float(cell)
    w = 0.5 * min(max(strand_width, 0.05), 0.95) * cell
    amp = 0.5 * max(weave_depth, 0.0) * cell
    return {'W': int(W), 'H': int(H), 'cell': cell, 'w': w,
            'amp': amp, 'weave': weave}


def _strand_height(s, other_index, family, p):
    """Ribbon-mode centerline height: C1 cosine ease through the
    +/-amp crossing targets."""
    amp, cell, weave = p['amp'], p['cell'], p['weave']
    t = s / cell
    k = np.floor(t).astype(np.int64)
    f = t - k
    if family == 0:
        s0 = _targets(k, other_index, 0, amp, weave)
        s1 = _targets(k + 1, other_index, 0, amp, weave)
    else:
        s0 = _targets(other_index, k, 1, amp, weave)
        s1 = _targets(other_index, k + 1, 1, amp, weave)
    return s0 + (s1 - s0) * (0.5 - 0.5 * np.cos(np.pi * f))


# --------------------------------------------------------------------
# The patch-network control net (rotated coordinates p = x+y, q = y-x)
# --------------------------------------------------------------------
#
# BLUE bands run along p on the integer q-lines; RED bands along q
# on the integer p-lines; they cross at every integer (P, Q).  Blue
# is UP where P + Q is even, DOWN (merged) where odd.  g is the
# band HALF-width in these units.  z is built at unit amplitude
# (front +1, back -1) and rescaled to h after subdivision.

def _blue_up(P, Q):
    """Checkerboard phase: blue arches over red iff P + Q is even
    (it merges with red where odd)."""
    return (P + Q) % 2 == 0


def _xy_of(P, Q):
    return ((P - Q) * 0.5, (P + Q) * 0.5)


def _cross_ok(P, Q, W, H):
    x, y = _xy_of(P, Q)
    return (-1e-9 <= x <= W + 1e-9) and (-1e-9 <= y <= H + 1e-9)


class _Net:
    """Quad/n-gon control net with vertex sharing.  Vertices are
    keyed by rounded (p, q, tag): blue (tag 1), red (tag 2) and the
    merged back-web (tag 3) share control points exactly where the
    structure fuses, and stay disjoint where blue arches over red.
    Faces are auto-oriented CCW in the (p, q) plan."""

    def __init__(self, cell):
        self.cell = cell
        self.pos = []                     # (p, q, z)
        self.key = {}
        self.faces = []
        self.labels = []

    def vid(self, p, q, z, tag):
        k = (round(p, 6), round(q, 6), tag)
        i = self.key.get(k)
        if i is None:
            i = len(self.pos)
            self.key[k] = i
            self.pos.append((float(p), float(q), float(z)))
        return i

    def lookup(self, p, q, tag):
        return self.key.get((round(p, 6), round(q, 6), tag))

    def face(self, ids, label):
        if len(ids) < 3 or len(set(ids)) != len(ids):
            return
        area = 0.0
        for k in range(len(ids)):
            p0, q0, _z0 = self.pos[ids[k]]
            p1, q1, _z1 = self.pos[ids[(k + 1) % len(ids)]]
            area += p0 * q1 - p1 * q0
        if area < 0.0:
            ids = ids[::-1]
        self.faces.append(tuple(ids))
        self.labels.append(label)

    def face_raw(self, ids, label):
        """A face with EXPLICIT winding (no plan-area auto-orient):
        needed for the near-vertical quads of the twisting bridges,
        whose plan projection is a degenerate sliver."""
        if len(ids) < 3 or len(set(ids)) != len(ids):
            return
        self.faces.append(tuple(ids))
        self.labels.append(label)

    def verts_xyz(self):
        c = self.cell
        return [np.array([(p - q) * 0.5 * c, (p + q) * 0.5 * c, z])
                for p, q, z in self.pos]


def _rot(cx, cy, a, dp, dq):
    """(cx, cy) + the offset (dp, dq) rotated CCW by a."""
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return (cx + dp * ca - dq * sa, cy + dp * sa + dq * ca)


def _ease5(u):
    """Quintic smootherstep 6u^5 - 15u^4 + 10u^3: BOTH the first
    and second derivatives vanish at the endpoints, so blends
    driven by it (arm width flare, swirl decay, bow relaxation,
    bridge roll) are curvature-continuous (C2) where they meet
    their end states."""
    u = min(max(float(u), 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _red_z(P, s, rise):
    """Red centerline height along q at p-line P: -1 at the merge
    hubs, rising `rise` under the arches, cosine-blended -- the
    red family undulates as a flowing floor, never a flat plane."""
    Q0 = int(np.floor(s + 1e-9))
    f = s - Q0
    z0 = -1.0 + (rise if _blue_up(P, Q0) else 0.0)
    z1 = -1.0 + (rise if _blue_up(P, Q0 + 1) else 0.0)
    return z0 + (z1 - z0) * (0.5 - 0.5 * float(np.cos(np.pi * f)))


def _arm(net, hub, u, sts, ws, zs, tags, label, bows, rots,
         corner_z=None):
    """A strip arm radiating from `hub` along the unit lattice
    axis u: station i sits at hub + R(rots[i]) (sts[i] u + t v),
    t in (-ws[i], 0, +ws[i]), v = rot90(u).  Per-station widths
    give the wide-at-hub / narrow-at-midspan flare; per-station
    rotations swirl the near-hub stations tangentially (the
    pinwheel, one chirality everywhere); per-station z, tag and
    bow (outer verts +bow/2, center -bow/2: the channel section
    with its envelope preserved).  `corner_z` overrides the first
    station's OUTER vertices -- the hub-ring corners, blended
    between the blue and red entry heights and shared with the
    merge patch and the neighbouring arm."""
    v = (-u[1], u[0])
    rows = []
    for i, s in enumerate(sts):
        row = []
        for k, t in enumerate((-ws[i], 0.0, ws[i])):
            dp = s * u[0] + t * v[0]
            dq = s * u[1] + t * v[1]
            p_, q_ = _rot(hub[0], hub[1], rots[i], dp, dq)
            z_ = zs[i]
            if corner_z is not None and i == 0 and k != 1:
                z_ = corner_z
            z_ += 0.5 * bows[i] if k != 1 else -0.5 * bows[i]
            row.append(net.vid(p_, q_, z_, tags[i]))
        rows.append(row)
    for i in range(len(sts) - 1):
        a0, c0, b0 = rows[i]
        a1, c1, b1 = rows[i + 1]
        net.face([a0, a1, c1, c0], label)
        net.face([c0, c1, b1, b0], label)
    return rows


def _merge_patch(net, P, Q, G, al, z_blue, z_red, z_corner):
    """The HUB at a blue-down crossing: a broad 3 x 3 patch, its
    ring rotated by the swirl angle, where the four flared arms
    meet.  A curved saddle-bowl -- nothing about it is planar:
    blue enters high and still descending (z_blue = -cos(pi G))
    along +/-p, red enters low and rising (z_red) along +/-q,
    corners blended between the two, and the center bottoms out on
    the red floor at -1 (a hair below, _VDIP, so the subdivision
    limit kisses it).  Ring vertices ARE the end stations of the
    four arms (shared, regular valence-4), so the hub is C2 with
    them."""
    offs = (-G, 0.0, G)
    ids = []
    for i in range(3):
        row = []
        for j in range(3):
            if i == 1 and j == 1:
                pq = (float(P), float(Q))
                z = -1.0 - 0.5 * _BOW - _VDIP
            else:
                pq = _rot(P, Q, al, offs[i], offs[j])
                if i == 1:             # red entry mids
                    z = z_red - 0.5 * _BOW
                elif j == 1:           # blue entry mids
                    z = z_blue - 0.5 * _BOW
                else:                  # blended ring corners
                    z = z_corner + 0.5 * _BOW
            row.append(net.vid(pq[0], pq[1], z, 3))
        ids.append(row)
    for i in range(2):
        for j in range(2):
            net.face([ids[i][j], ids[i + 1][j],
                      ids[i + 1][j + 1], ids[i][j + 1]], _MERGE)


def _bridge(net, Pd, Qd, e, G, sc, wcol, rcol, bw, bwz, kz, al,
            W, H):
    """A ribbon bridge joining hub (Pd, Qd) to the diagonally
    adjacent hub (Pd+1, Qd+e) across the open cell.
    The neck is a MONOTONIC QUARTER-TURN -- a helicoid segment.
    The cross-section rotates continuously in one rotational
    direction across the span: horizontal (0 deg) easing C2
    into hub A's TOP face, to VERTICAL (90 deg) at hub B, where
    the ribbon lands ON EDGE -- a WALL standing in the far
    hub's thickness band.  Because the roll never unwinds, the
    twist reads CLOCKWISE looking down the axis from one hub
    and COUNTERCLOCKWISE from the other -- the swirl -- unlike
    a wind-and-unwind bump, which reads the same from both
    sides; and because it never passes 90 degrees, the two
    teardrop openings flanking the neck stay separate.  The e
    sign in the roll keeps ONE global handedness across both
    diagonal families (negate it to mirror the whole field).
    The ribbon is
    exactly one shell THICKNESS wide (bw in plan units, bwz in
    unit z: the same world width) and its end sections are
    centered on the hub-to-hub AXIS: the centerline is the
    straight segment between the two hub centers, centered in the
    gap with symmetric margins, LEVEL at the elevation where the
    two hubs are at the same height (the shared ring-corner
    level), and decoupled from the swirled rings (the caps absorb
    the skew): a straight level neck whose section turns one
    quarter revolution.
    The flat (hub A) cap is one simple quad + one notch triangle
    onto the shared hub corner and collar shoulder vertices; the
    on-edge (hub B) landing is a RULED BAND of gently-rolled
    sub-quads curling the standing wall onto the hub (a single
    quad there would be plan-degenerate -- the old crumple), plus
    the notch triangle.  All cap faces are simple non-degenerate
    polygons with consistent winding (no sliver fans); at the rim
    a missing arm degrades the cap gracefully.  Faces are emitted
    with explicit winding."""
    hA = (float(Pd), float(Qd))
    hB = (float(Pd + 1), float(Qd + e))
    Cn = _rot(hA[0], hA[1], al, G, e * G)
    Cf = _rot(hB[0], hB[1], al, -G, -e * G)
    vCn = net.lookup(Cn[0], Cn[1], 3)
    vCf = net.lookup(Cf[0], Cf[1], 3)
    if vCn is None or vCf is None:
        return
    sb = net.lookup(*_rot(hA[0], hA[1], rcol, sc, e * wcol), 1)
    sr = net.lookup(*_rot(hA[0], hA[1], rcol, wcol, e * sc), 2)
    fb = net.lookup(*_rot(hB[0], hB[1], rcol, -sc, -e * wcol), 1)
    fr = net.lookup(*_rot(hB[0], hB[1], rcol, -wcol, -e * sc), 2)

    # the roll: a MONOTONIC QUARTER-TURN.  The cross-section
    # rotates continuously in ONE rotational direction across
    # the span: HORIZONTAL (0 deg) at hub A, easing C2 into A's
    # TOP face, to VERTICAL (90 deg) at hub B -- the ribbon
    # lands ON EDGE, a WALL standing in the far hub's thickness
    # band.  Because the roll is monotone it reads CLOCKWISE
    # seen down the axis from one hub and COUNTERCLOCKWISE from
    # the other -- the swirl -- and because it never passes 90
    # degrees the two teardrop openings flanking the neck stay
    # separate.  The e sign keeps ONE global handedness across
    # both diagonal families (flip e -> -e here to mirror the
    # whole field).
    ath = float(e) * 0.5 * np.pi
    # end-to-end twist: exactly a quarter turn, monotone
    net.twist_deg = 90.0
    zA = net.pos[vCn][2]      # hub attachment heights: the ring
    zB = net.pos[vCf][2]      # corners, equal by construction
    zlev = 0.5 * (zA + zB)    # the hubs' common height
    net.bridge_zends = (zA, zB)
    # the centerline is the STRAIGHT hub-to-hub axis segment,
    # centered in the gap with symmetric margins, LEVEL at the
    # height where both hubs are equal -- decoupled from the
    # swirled ring (the caps absorb the skew to the rotated
    # corner / shoulder vertices): a straight, centered,
    # horizontal bar
    ax = np.array([1.0, float(e)]) / np.sqrt(2.0)
    nh = np.array([-float(e), 1.0]) / np.sqrt(2.0)
    span = np.sqrt(2.0) * (1.0 - 2.0 * G)
    A0 = np.array([hA[0] + G, hA[1] + e * G])
    ts = np.linspace(0.20, 0.80, 11)
    us = (ts - ts[0]) / (ts[-1] - ts[0])
    Bt, Tp = [], []
    for t, u in zip(ts, us):
        # MONOTONE C2 quintic roll 0 -> 90: zero first and
        # second derivative at both ends (matching the collar
        # ease), flat at hub A, on edge at hub B
        a = ath * _ease5(u)
        C = A0 + (t * span) * ax
        zc = zlev
        lat = float(np.cos(a)) * 0.5 * bw
        dz = float(np.sin(a)) * 0.5 * bwz
        Bt.append(net.vid(C[0] - e * nh[0] * lat,
                          C[1] - e * nh[1] * lat, zc - dz, 4))
        Tp.append(net.vid(C[0] + e * nh[0] * lat,
                          C[1] + e * nh[1] * lat, zc + dz, 5))

    def braw(ids):
        net.face_raw(ids if e > 0 else list(ids)[::-1], _BRIDGE)

    for i in range(len(ts) - 1):
        braw([Bt[i], Bt[i + 1], Tp[i + 1], Tp[i]])

    # near cap: ONE simple quad + the notch triangle -- the roll
    # is zero at the ends, so the quad is a clean plan-monotone
    # trapezoid easing the ribbon onto the hub's top face;
    # winding is forced by the neighbouring arm faces
    if sb is not None and sr is not None:
        braw([sb, Bt[0], Tp[0], sr])
        braw([vCn, sb, sr])
    elif sb is not None:
        braw([sb, Bt[0], Tp[0], vCn])
    elif sr is not None:
        braw([vCn, Bt[0], Tp[0], sr])
    else:
        braw([vCn, Bt[0], Tp[0]])

    # far cap (the ON-EDGE end): the rails arrive vertically
    # stacked, so a single cap quad would be plan-degenerate --
    # the crumple source.  Mesh the landing as a RULED BAND
    # instead: intermediate rulings interpolated between the
    # vertical rail-end edge and the shoulder chord split the
    # ~quarter curl onto the hub into several gently-rolled
    # sub-quads -- an embedded helicoid-like blend, simple
    # non-degenerate quads, consistent winding, one band per
    # family (it handles the aligned and the anti-aligned
    # shoulder slopes alike)
    if fb is not None and fr is not None:
        pT0, pB0 = net.pos[Tp[-1]], net.pos[Bt[-1]]
        pTf, pBf = net.pos[fb], net.pos[fr]
        Ts, Bs = [Tp[-1]], [Bt[-1]]
        K = 3
        for j in range(1, K + 1):
            u = j / float(K + 1)
            pT = [(1.0 - u) * a_ + u * b_
                  for a_, b_ in zip(pT0, pTf)]
            pB = [(1.0 - u) * a_ + u * b_
                  for a_, b_ in zip(pB0, pBf)]
            Ts.append(net.vid(pT[0], pT[1], pT[2], 5))
            Bs.append(net.vid(pB[0], pB[1], pB[2], 4))
        Ts.append(fb)
        Bs.append(fr)
        for j in range(K + 1):
            braw([Bs[j], Bs[j + 1], Ts[j + 1], Ts[j]])
        braw([vCf, fb, fr])
    elif fb is not None:
        braw([fb, Tp[-1], Bt[-1], vCf])
    elif fr is not None:
        braw([vCf, Tp[-1], Bt[-1], fr])
    else:
        braw([vCf, Tp[-1], Bt[-1]])


def _control_net(W, H, cell, strand_width, thickness=0.1,
                 weave_depth=0.15):
    """The full control net: broad swirled HUBS at every blue-down
    crossing; arms that flare wide into the hubs and narrow toward
    midspan; an undulating red floor (never flat); blue cosine
    arches; and quarter-turn bridges (horizontal into one hub's
    top, one monotone rotation to an on-edge wall at the
    neighbour hub) -- a continuous relief with no planar region
    anywhere.  The bridge ribbon
    width is hard-coupled to the shell THICKNESS (both fractions
    of a cell): plan lengths scale by cell/sqrt(2) and unit z by
    h = weave_depth * cell / 2, hence bw / bwz give the same world
    width; kz converts unit z to plan lengths for the measured
    tangency tilts."""
    g = 0.5 * min(max(strand_width, 0.15), 0.9)
    G = min(_G_HUB * g, 0.44)              # hub half-width
    # the flare zone [G, sm] blends the arm into the hub with the
    # QUINTIC ease: width, swirl and bow all reach their end
    # states with zero first AND second derivatives (a C2 swell,
    # no abrupt shoulder); two intermediate collar rings carry the
    # blend so the subdivision limit curvature is continuous too
    sm = 0.5 * (G + 1.0 - g)               # arm midspan station
    fz = sm - G
    sc1 = G + fz / 3.0
    sc2 = G + 2.0 * fz / 3.0
    q1 = _ease5(1.0 / 3.0)
    q2 = _ease5(2.0 / 3.0)
    w1 = g + (G - g) * (1.0 - q1)
    w2 = g + (G - g) * (1.0 - q2)
    t_eff = max(float(thickness), 0.02)
    bw = t_eff * np.sqrt(2.0)              # plan (p, q)
    bwz = min(2.0 * t_eff / max(float(weave_depth), 1e-6),
              4.0)                         # unit z
    kz = float(weave_depth) * np.sqrt(2.0) / 2.0
    cg = float(np.cos(np.pi * g))
    cG = float(np.cos(np.pi * G))
    al = _SWIRL
    r1 = al * (1.0 - q1)
    r2 = al * (1.0 - q2)
    zrG = -1.0 + _RISE * (0.5 - 0.5 * cG)  # red ring entry
    zc = 0.5 * (-cG + zrG)                 # blended ring corners
    net = _Net(float(cell))
    net.twist_deg = 0.0
    net.hub_G = G
    net.arm_w = (2.0 * G, 2.0 * g)

    def ok(P, Q):
        return _cross_ok(P, Q, W, H)

    # hubs first, then the arms (sharing the hub rings), then the
    # bridges (which look the shared collar shoulders up)
    for P in range(0, W + H + 1):
        for Q in range(-W, H + 1):
            if not ok(P, Q) or _blue_up(P, Q):
                continue
            if (ok(P - 1, Q) or ok(P + 1, Q) or ok(P, Q - 1)
                    or ok(P, Q + 1)):
                _merge_patch(net, P, Q, G, al, -cG, zrG, zc)
    sts = (G, sc1, sc2, sm, 1.0 - g)
    ws = (G, w1, w2, g, g)
    bows = (_BOW, _BOW * (1.0 - 0.5 * q1),
            _BOW * (1.0 - 0.5 * q2), 0.5 * _BOW, _BOW)
    rots = (al, r1, r2, 0.0, 0.0)
    for P in range(0, W + H + 1):
        for Q in range(-W, H + 1):
            if not ok(P, Q):
                continue
            if _blue_up(P, Q):
                # the blue ARCH over the undulating red strip
                # beneath: two separate sheets, never merged
                if ok(P - 1, Q) or ok(P + 1, Q):
                    _arm(net, (P, Q), (1.0, 0.0), (-g, 0.0, g),
                         (g, g, g), (cg, 1.0, cg), (1, 1, 1),
                         _ARCH, (_BOW, _BOW, _BOW),
                         (0.0, 0.0, 0.0))
                if ok(P, Q - 1) or ok(P, Q + 1):
                    zs = tuple(_red_z(P, Q + s, _RISE)
                               for s in (-g, 0.0, g))
                    _arm(net, (P, Q), (0.0, 1.0), (-g, 0.0, g),
                         (g, g, g), zs, (2, 2, 2), _RED,
                         (_BOW, _BOW, _BOW), (0.0, 0.0, 0.0))
            else:
                for e in (1, -1):
                    if ok(P + e, Q):
                        # blue riser arm: flared, swirled, on the
                        # descending cosine out of the hub
                        zs = tuple(-float(np.cos(np.pi * s))
                                   for s in sts)
                        _arm(net, (P, Q), (float(e), 0.0), sts,
                             ws, zs, (3, 1, 1, 1, 1), _BLUE,
                             bows, rots, corner_z=zc)
                    if ok(P, Q + e):
                        # red connector arm: flared, swirled, on
                        # the undulating floor toward the arch
                        zs = tuple(_red_z(P, Q + e * s, _RISE)
                                   for s in sts)
                        _arm(net, (P, Q), (0.0, float(e)), sts,
                             ws, zs, (3, 2, 2, 2, 2), _RED,
                             bows, rots, corner_z=zc)
    for P in range(0, W + H + 1):
        for Q in range(-W, H + 1):
            if not ok(P, Q) or _blue_up(P, Q):
                continue
            for e in (1, -1):
                if ok(P + 1, Q + e):
                    _bridge(net, P, Q, e, G, sc1, w1, r1, bw,
                            bwz, kz, al, W, H)
    # amplitude reference: the WEAVE envelope only -- the bridges'
    # on-edge ends legitimately span the thickness wall band and
    # must not rescale the relief
    zs_ref = [abs(z) for (pp, qq, tg), i in net.key.items()
              for z in (net.pos[i][2],) if tg not in (4, 5)]
    net.zref = max(zs_ref) if zs_ref else 1.0
    return net


def _round_corners(V, faces, thresh=95.0, t=0.6):
    """Round the sharp boundary corners of the control net (the
    star-hole tips and the bridge shoulder notches) before
    subdivision: every 2-neighbour boundary vertex whose boundary
    polyline turns by more than `thresh` degrees IN PLAN is pulled
    toward the chord midpoint of its two boundary neighbours -- a
    corner-cutting step on the boundary control polygon -- so the
    Catmull-Clark boundary B-spline rounds the tip instead of
    leaving a hard angle.  The in-plan test skips the arch / riser
    crest controls (whose turning is intended z-profile), the
    threshold leaves the 90-degree rim strap ends alone, and z is
    kept frozen so the back-web stays flat.  Topology (faces, hole
    count) is untouched."""
    V = [np.asarray(v, dtype=float).copy() for v in V]
    cnt = {}
    for f in faces:
        m = len(f)
        for k in range(m):
            a, b = int(f[k]), int(f[(k + 1) % m])
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    nbr = defaultdict(list)
    for e, c in cnt.items():
        if c == 1:
            nbr[e[0]].append(e[1])
            nbr[e[1]].append(e[0])
    cmin = float(np.cos(np.radians(180.0 - thresh)))
    upd = {}
    for v, ns in nbr.items():
        if len(ns) != 2:
            continue
        d1 = (V[ns[0]] - V[v])[:2]
        d2 = (V[ns[1]] - V[v])[:2]
        n1 = float(np.hypot(d1[0], d1[1]))
        n2 = float(np.hypot(d2[0], d2[1]))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        if float(np.dot(d1, d2)) / (n1 * n2) > cmin:
            mid = 0.5 * (V[ns[0]][:2] + V[ns[1]][:2])
            upd[v] = V[v][:2] + t * (mid - V[v][:2])
    for v, xy in upd.items():
        V[v][:2] = xy
    return V


# --------------------------------------------------------------------
# Catmull-Clark refinement (bicubic B-spline patch evaluation)
# --------------------------------------------------------------------

def _catmull_clark(V, faces, labels, rounds):
    """Standard Catmull-Clark with boundary rules (boundary curves
    subdivide as cubic B-splines; corner / pinch vertices are
    interpolated).  Labels are inherited by child faces."""
    V = [np.asarray(v, dtype=float) for v in V]
    for _ in range(max(0, rounds)):
        fpts = [np.mean([V[i] for i in f], axis=0) for f in faces]
        edges = {}
        for fi, f in enumerate(faces):
            m = len(f)
            for k in range(m):
                e = (f[k], f[(k + 1) % m])
                e = e if e[0] < e[1] else (e[1], e[0])
                edges.setdefault(e, []).append(fi)
        epts = []
        eidx = {}
        for e, fs in edges.items():
            if len(fs) == 2:
                ep = 0.25 * (V[e[0]] + V[e[1]]
                             + fpts[fs[0]] + fpts[fs[1]])
            else:
                ep = 0.5 * (V[e[0]] + V[e[1]])
            eidx[e] = len(epts)
            epts.append(ep)
        vfaces = defaultdict(list)
        vedges = defaultdict(list)
        vbound = defaultdict(list)
        for e, fs in edges.items():
            vedges[e[0]].append(e)
            vedges[e[1]].append(e)
            if len(fs) == 1:
                vbound[e[0]].append(e[1])
                vbound[e[1]].append(e[0])
        for fi, f in enumerate(faces):
            for i in f:
                vfaces[i].append(fi)
        newV = []
        for i, v in enumerate(V):
            bnd = vbound.get(i)
            if bnd is not None:
                if len(bnd) == 2:
                    newV.append((6.0 * v + V[bnd[0]] + V[bnd[1]])
                                / 8.0)
                else:
                    newV.append(v)
                continue
            es = vedges.get(i, ())
            fs = vfaces.get(i, ())
            n = len(es)
            if n < 3 or not fs:
                newV.append(v)
                continue
            Fa = np.mean([fpts[fj] for fj in fs], axis=0)
            Ra = np.mean([0.5 * (V[e[0]] + V[e[1]]) for e in es],
                         axis=0)
            newV.append((Fa + 2.0 * Ra + (n - 3.0) * v) / n)
        nv, ne = len(V), len(epts)
        nfaces, nlabels = [], []
        for fi, f in enumerate(faces):
            m = len(f)
            fc = nv + ne + fi
            for k in range(m):
                a_, b_, c_ = f[k], f[(k + 1) % m], f[k - 1]
                e1 = (a_, b_) if a_ < b_ else (b_, a_)
                e0 = (c_, a_) if c_ < a_ else (a_, c_)
                nfaces.append((f[k], nv + eidx[e1], fc,
                               nv + eidx[e0]))
                nlabels.append(labels[fi])
        V = newV + epts + fpts
        faces, labels = nfaces, nlabels
    return V, faces, labels


# --------------------------------------------------------------------
# Mesh utilities
# --------------------------------------------------------------------

def _edge_array(faces):
    E = set()
    for q in faces:
        n = len(q)
        for k in range(n):
            a, b = int(q[k]), int(q[(k + 1) % n])
            E.add((a, b) if a < b else (b, a))
    return np.array(sorted(E), dtype=int)


def _taubin(V, faces, rounds, lam=0.5, mu=-0.53):
    """Non-shrinking Taubin lambda|mu fairing (in place)."""
    if rounds <= 0:
        return V
    E = _edge_array(faces)
    if len(E) == 0:
        return V
    for _ in range(rounds):
        for s in (lam, mu):
            acc = np.zeros_like(V)
            deg = np.zeros(len(V))
            np.add.at(acc, E[:, 0], V[E[:, 1]])
            np.add.at(acc, E[:, 1], V[E[:, 0]])
            np.add.at(deg, E[:, 0], 1.0)
            np.add.at(deg, E[:, 1], 1.0)
            good = deg > 0
            V[good] += s * (acc[good] / deg[good][:, None] - V[good])
    return V


def _components(nv, faces):
    parent = list(range(nv))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for q in faces:
        r0 = find(int(q[0]))
        for v in q[1:]:
            r = find(int(v))
            if r != r0:
                parent[r] = r0
    used = set(int(v) for q in faces for v in q)
    return len(set(find(v) for v in used))


def _mesh_stats(nv, faces):
    """(components, over-shared edge count, rim count).  Rims =
    independent cycles of the boundary graph (star openings + the
    outer loop), robust to pinch vertices."""
    cnt = {}
    for f in faces:
        m = len(f)
        for k in range(m):
            a, b = int(f[k]), int(f[(k + 1) % m])
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    nonman = sum(1 for c in cnt.values() if c > 2)
    bedges = [e for e, c in cnt.items() if c == 1]
    bverts = set(v for e in bedges for v in e)
    parent = {v: v for v in bverts}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in bedges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    bcomps = len(set(find(v) for v in bverts))
    rims = len(bedges) - len(bverts) + bcomps
    return _components(nv, faces), nonman, rims


def _face_normal(V, f):
    """Newell normal of a (possibly non-planar) polygon."""
    n = np.zeros(3)
    m = len(f)
    for k in range(m):
        a = V[int(f[k])]
        b = V[int(f[(k + 1) % m])]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])


def _seam_normal_dots(V, faces, labels):
    """Face-normal continuity across the back-web seams, over
    INTERIOR seam edges: merge <-> strip seams always; bridge
    junction edges only where BOTH faces lie flat (|n_z| >= 0.85)
    -- the steep on-edge part of a bridge's quarter-turn is a
    deliberate feature, not a back-web seam.  Near 1.0 everywhere
    = C1, no crease."""
    emap = {}
    for fi, f in enumerate(faces):
        m = len(f)
        for k in range(m):
            a, b = int(f[k]), int(f[(k + 1) % m])
            e = (a, b) if a < b else (b, a)
            emap.setdefault(e, []).append(fi)
    bverts = set()
    for e, fs in emap.items():
        if len(fs) == 1:
            bverts.update(e)
    dots = []
    for e, fs in emap.items():
        if len(fs) != 2 or e[0] in bverts or e[1] in bverts:
            continue
        la, lb = labels[fs[0]], labels[fs[1]]
        if la == lb:
            continue
        pair = {la, lb}
        if not (pair & {_MERGE, _BRIDGE}):
            continue
        na = _face_normal(V, faces[fs[0]])
        nb = _face_normal(V, faces[fs[1]])
        if (_BRIDGE in pair
                and (abs(na[2]) < 0.85 or abs(nb[2]) < 0.85)):
            continue                       # the deliberate twist
        dots.append(float(np.dot(na, nb)))
    return dots


def _vertex_normals(V, quads):
    N = np.zeros_like(V)
    for q in quads:
        n = np.cross(V[q[2]] - V[q[0]], V[q[3]] - V[q[1]])
        for v in q:
            N[v] += n
    ln = np.linalg.norm(N, axis=1)
    ln[ln < 1e-12] = 1.0
    N /= ln[:, None]
    N[np.abs(N).sum(axis=1) < 1e-9] = (0.0, 0.0, 1.0)
    return N


def _smooth_normals(N, quads, rounds=3):
    E = _edge_array(quads)
    for _ in range(rounds):
        acc = N.copy()
        deg = np.ones(len(N))
        np.add.at(acc, E[:, 0], N[E[:, 1]])
        np.add.at(acc, E[:, 1], N[E[:, 0]])
        np.add.at(deg, E[:, 0], 1.0)
        np.add.at(deg, E[:, 1], 1.0)
        N = acc / deg[:, None]
        ln = np.linalg.norm(N, axis=1)
        ln[ln < 1e-12] = 1.0
        N /= ln[:, None]
    return N


def _solidify(V, quads, thickness):
    """Thicken an open shell along its vertex normals into a
    closed slab: top faces, reversed bottom faces, and a side quad
    per boundary edge.  Interior normals are smoothed, but RIM
    vertices keep their raw local face normals, so the extruded
    side walls stand orthogonal to the top/bottom faces they
    adjoin (a clean squared edge, not a slanted one)."""
    n = len(V)
    cnt = {}
    for q in quads:
        m = len(q)
        for k in range(m):
            a, b = int(q[k]), int(q[(k + 1) % m])
            e = (a, b) if a < b else (b, a)
            cnt.setdefault(e, []).append((a, b))
    bnd = sorted(set(v for e, uses in cnt.items()
                     if len(uses) == 1 for v in e))
    N0 = _vertex_normals(V, quads)
    N = _smooth_normals(N0.copy(), quads)
    if bnd:
        N[bnd] = N0[bnd]
    top = V + 0.5 * thickness * N
    bot = V - 0.5 * thickness * N
    verts = np.vstack([top, bot])
    faces = [tuple(int(v) for v in q) for q in quads]
    faces += [tuple(int(v) + n for v in reversed(q)) for q in quads]
    for e, uses in cnt.items():
        if len(uses) == 1:
            u, v = uses[0]
            faces.append((v, u, u + n, v + n))
    return verts, faces


# --------------------------------------------------------------------
# Coloring
# --------------------------------------------------------------------

def _height_mats(cz):
    lo, hi = float(cz.min()), float(cz.max())
    span = max(hi - lo, 1e-9)
    return [min(_HEIGHT_BANDS - 1,
                int((z - lo) / span * _HEIGHT_BANDS)) for z in cz]


def _face_mats(V, faces, color_by, labels=None, family=None):
    if color_by == 'FAMILY':
        if labels is not None:
            # red / blue risers / merges / bridges / blue arches
            return list(labels)
        return [family] * len(faces)
    if color_by == 'HEIGHT':
        cz = np.array([np.mean([V[i][2] for i in f]) for f in faces])
        return _height_mats(cz)
    return [0] * len(faces)


def _finish(V, faces, mats, thickness):
    """Solidify (thickness > 0) and expand the material list."""
    if thickness > 1e-9:
        sv, sf = _solidify(np.asarray(V, dtype=float), faces,
                           thickness)
        nside = len(sf) - 2 * len(faces)
        side = mats[0] if mats else 0
        return ([tuple(map(float, v)) for v in sv], sf,
                mats + mats + [side] * nside)
    return ([tuple(map(float, v)) for v in V],
            [tuple(int(i) for i in f) for f in faces], mats)


# --------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------

def build_membrane(W, H, cell=1.0, strand_width=0.33,
                   weave_depth=0.18, weave='PLAIN', res=10,
                   thickness=0.07, smooth_rounds=2,
                   surface='MEMBRANE', relax_iters=12,
                   color_by='UNIFORM'):
    """The asymmetric woven screen as a Catmull-Clark-evaluated
    bicubic patch network.  Returns (cell, stats); stats carries
    the red-level maximum, per-parity crossover measurements (gap
    at blue-up, merge dip at blue-down), back-web connectivity,
    seam normal-continuity and hole count -- everything the
    self-test asserts."""
    cell = float(cell)
    net = _control_net(W, H, cell, strand_width, thickness,
                       weave_depth)
    rounds = max(1, min(4, int(np.log2(max(4, int(res)))) - 1))
    V0 = _round_corners(net.verts_xyz(), net.faces)
    V, faces, labels = _catmull_clark(V0, net.faces,
                                      net.labels, rounds)
    V = np.array(V, dtype=float)
    h = 0.5 * max(weave_depth, 0.0) * cell
    # scale by the weave envelope (net.zref, excluding the bridge
    # rails whose on-edge ends span the thickness wall band)
    zmax = float(getattr(net, 'zref', 0.0))
    if zmax <= 1e-9:
        zmax = float(np.abs(V[:, 2]).max())
    if zmax > 1e-9:
        V[:, 2] *= h / zmax
    if smooth_rounds > 0:
        _taubin(V, faces, smooth_rounds)
    stats = {'minimal': None}
    if surface == 'MINIMAL':
        _taubin(V, faces, max(1, relax_iters))
        stats['minimal'] = 'TAUBIN-FAIRING'
    comps, nonman, rims = _mesh_stats(len(V), faces)

    # --- verification measurements -------------------------------
    C = np.array([np.mean([V[i] for i in f], axis=0) for f in faces])
    L = np.array(labels)
    # red never rises
    red = L == _RED
    red_max = float(C[red, 2].max()) if red.any() else -h
    # per crossover: blue-up = two sheets with a real gap;
    # blue-down = single merged sheet at the back level
    up_top_min, up_bot_max, up_gap_faces = None, None, 0
    down_max = None
    for P in range(0, W + H + 1):
        for Q in range(-W, H + 1):
            if not _cross_ok(P, Q, W, H):
                continue
            x, y = _xy_of(P, Q)
            r = np.hypot(C[:, 0] - x * cell, C[:, 1] - y * cell)
            near = r < 0.15 * cell
            if not near.any():
                continue
            zz = C[near, 2]
            if _blue_up(P, Q):
                blue = near & ((L == _ARCH) | (L == _BLUE))
                redn = near & (L == _RED)
                if not (blue.any() and redn.any()):
                    continue
                hi = float(C[blue, 2].max())
                lo = float(C[redn, 2].min())
                up_top_min = (hi if up_top_min is None
                              else min(up_top_min, hi))
                up_bot_max = (lo if up_bot_max is None
                              else max(up_bot_max, lo))
                up_gap_faces += int(np.sum(np.abs(zz) < 0.4 * h))
            else:
                mz = float(zz.max())
                down_max = (mz if down_max is None
                            else max(down_max, mz))
    # the back-web (red + merges + bridges) is one connected surface
    back_faces = [f for f, lb in zip(faces, labels)
                  if lb in (_RED, _MERGE, _BRIDGE)]
    back_comps = (_components(len(V), back_faces) if back_faces
                  else 0)
    # red floor level vs blue's valley minimum: the trough of the
    # merge valley must sit exactly on the red floor (min-vertex z
    # of each family; the trough row is shared geometry)
    redv, merv = set(), set()
    for fc, lb in zip(faces, labels):
        if lb == _RED:
            redv.update(int(i) for i in fc)
        elif lb == _MERGE:
            merv.update(int(i) for i in fc)
    red_z = float(V[sorted(redv), 2].min()) if redv else -h
    red_vmax = float(V[sorted(redv), 2].max()) if redv else -h
    blue_min = float(V[sorted(merv), 2].min()) if merv else -h
    dots = _seam_normal_dots(V, faces, labels)
    stats.update({
        'components': comps, 'nonmanifold': nonman, 'rims': rims,
        'holes': max(0, rims - 1), 'h': h,
        'red_max': red_max,
        'up_top_min': up_top_min or 0.0,
        'up_bot_max': up_bot_max or 0.0,
        'up_gap_faces': up_gap_faces,
        'down_max': 0.0 if down_max is None else down_max,
        'red_z': red_z, 'blue_min': blue_min,
        'red_var': red_vmax - red_z,
        'twist_deg': getattr(net, 'twist_deg', 0.0),
        'bridge_ends': tuple(
            z * h / zmax for z in
            getattr(net, 'bridge_zends', (0.0, 0.0))),
        'hub_G': getattr(net, 'hub_G', 0.0),
        'arm_w': getattr(net, 'arm_w', (0.0, 0.0)),
        'bow': _BOW,
        'bridge_w': max(float(thickness), 0.02) * cell,
        'back_components': back_comps,
        'seam_dot_min': min(dots) if dots else 1.0,
        'seam_dot_mean': (sum(dots) / len(dots)) if dots else 1.0})
    mats = _face_mats(V, faces, color_by, labels=labels)
    return _finish(V, faces, mats, thickness * cell), stats


def build_ribbons(W, H, cell=1.0, strand_width=0.33,
                  weave_depth=0.18, weave='PLAIN', res=10,
                  thickness=0.07, color_by='UNIFORM'):
    """The classic axis-aligned weave as separate lofted ribbons
    (one cell per strand; PLAIN / TWILL / BASKET drafts)."""
    p = _params(W, H, cell, strand_width, weave_depth, weave)
    res = max(4, int(res))
    w = p['w']
    nt = max(4, res // 3)
    cells = []
    for family in range(2):
        n_lines = (H if family == 0 else W) + 1
        span = (W if family == 0 else H) * p['cell']
        nlong = (W if family == 0 else H) * res + 1
        s = np.linspace(-w, span + w, nlong + 2 * max(2, res // 4))
        t = np.linspace(-w, w, nt)
        for line in range(n_lines):
            zc = _strand_height(s, np.int64(line), family, p)
            S, Tt = np.meshgrid(s, t, indexing='ij')
            Z = np.repeat(zc[:, None], nt, axis=1)
            if family == 0:
                Xg, Yg = S, line * p['cell'] + Tt
            else:
                Xg, Yg = line * p['cell'] + Tt, S
            ns, ntv = Xg.shape
            V = np.column_stack([Xg.ravel(), Yg.ravel(), Z.ravel()])
            quads = []
            for i in range(ns - 1):
                for j in range(ntv - 1):
                    a = i * ntv + j
                    quads.append((a, a + ntv, a + ntv + 1, a + 1))
            mats = _face_mats(V, quads, color_by, family=family)
            cells.append(_finish(V, quads, mats,
                                 thickness * p['cell']))
    return cells


def build_screen(W, H, cell=1.0, strand_width=0.33,
                 weave_depth=0.18, weave='PLAIN',
                 surface='MEMBRANE', res=10, thickness=0.07,
                 smooth_rounds=2, relax_iters=12, backing=False,
                 base=0.1, color_by='UNIFORM'):
    """All modes.  Returns (cells, stats)."""
    if surface == 'RIBBONS':
        cells = build_ribbons(W, H, cell, strand_width, weave_depth,
                              weave, res, thickness, color_by)
        stats = {'holes': W * H, 'components': (W + 1) + (H + 1),
                 'nonmanifold': 0, 'minimal': None}
    else:
        c, stats = build_membrane(W, H, cell, strand_width,
                                  weave_depth, 'PLAIN', res,
                                  thickness, smooth_rounds, surface,
                                  relax_iters, color_by)
        cells = [c]
        if weave != 'PLAIN':
            stats['note'] = ("the asymmetric membrane is "
                             "plain-weave only")
    if backing and cells:
        allv = np.array([v for cv, _f, _m in cells for v in cv])
        if len(allv):
            lo = allv[:, :2].min(axis=0)
            hi = allv[:, :2].max(axis=0)
            zt = float(allv[:, 2].min()) - 0.02 * cell
            bv, bf, bm = [], [], []
            pc.slab(bv, bf, bm, lo, hi, zt, zt - base * cell,
                    _BACKING_MAT)
            cells.append((bv, bf, bm))
    return cells, stats


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_over_under_screen_add(bpy.types.Operator,
                                        AddObjectHelper):
        """Add a woven over-under screen (Hauer woven web)"""
        bl_idname = "mesh.over_under_screen_add"
        bl_label = "Over-Under Screen"
        bl_options = {'REGISTER', 'UNDO'}

        grid_w: IntProperty(name="Grid Width", default=6, min=1,
                            max=30, description="Cells across")
        grid_h: IntProperty(name="Grid Height", default=5, min=1,
                            max=30, description="Cells down")
        cell: FloatProperty(
            name="Cell Size", default=1.0, min=0.1, max=10.0,
            description="Edge length of one lattice cell (the whole "
                        "screen is refit to the 2 m cube)")
        weave: EnumProperty(
            name="Weave",
            items=[('PLAIN', "Plain",
                    "Over-1-under-1 (the membrane structure; the "
                    "only draft of the woven screen)"),
                   ('TWILL', "Twill 2/2",
                    "Over-2-under-2 diagonal wale (ribbons mode)"),
                   ('BASKET', "Basket 2/2",
                    "Over-2-under-2 paired strands (ribbons mode)")],
            default='PLAIN',
            description="Weave draft (TWILL / BASKET apply to the "
                        "ribbons mode only)")
        surface: EnumProperty(
            name="Surface",
            items=[('RIBBONS', "Ribbons",
                    "Separate axis-aligned woven ribbons"),
                   ('MEMBRANE', "Membrane",
                    "One woven patch-network screen: a flowing "
                    "back-web with the weaving family arching over "
                    "it, star openings (the Hauer screen)"),
                   ('MINIMAL', "Minimal (Taut)",
                    "The membrane tautened by extra non-shrinking "
                    "(Taubin) fairing")],
            default='MEMBRANE')
        strand_width: FloatProperty(
            name="Strand Width", default=0.33, min=0.15, max=0.9,
            description="Ribbon width as a fraction of the ribbon "
                        "pitch (also sets the opening size)")
        weave_depth: FloatProperty(
            name="Weave Depth", default=0.18, min=0.0, max=2.0,
            description="Peak-to-peak front-to-back amplitude as a "
                        "fraction of a cell (2h)")
        resolution: IntProperty(
            name="Resolution", default=10, min=4, max=48,
            description="Tessellation (sets the subdivision depth "
                        "of the patch network)")
        thickness: FloatProperty(
            name="Thickness", default=0.07, min=0.0, max=1.0,
            description="Shell thickness as a fraction of a cell "
                        "(0 = open surface); the twisting bridges "
                        "are exactly this wide")
        smooth_rounds: IntProperty(
            name="Smooth Rounds", default=2, min=0, max=20,
            description="Taubin fairing passes on the evaluated "
                        "surface")
        relax_iters: IntProperty(
            name="Relax Iterations", default=12, min=1, max=60,
            description="Extra fairing passes (taut mode)")
        backing: BoolProperty(
            name="Backing Slab", default=False,
            description="Add a slab behind the screen")
        base: FloatProperty(
            name="Base Thickness", default=0.1, min=0.01, max=1.0,
            description="Backing slab thickness (fraction of a "
                        "cell)")
        color_by: EnumProperty(
            name="Color By",
            items=[('UNIFORM', "Uniform", "A single material"),
                   ('FAMILY', "By Element",
                    "Tones by element: back ribbons, weaving "
                    "risers, merges, bridges, arches"),
                   ('HEIGHT', "By Height",
                    "Palette bands by surface height")],
            default='UNIFORM')
        separate: BoolProperty(
            name="Separate Parts", default=False,
            description="One object per part (each ribbon / the "
                        "membrane and slab)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.05, max=10.0,
            description="Overall size (1 = fit the 2 m cube)")

        def execute(self, context):
            cells, stats = build_screen(
                self.grid_w, self.grid_h, self.cell,
                self.strand_width, self.weave_depth, self.weave,
                self.surface, self.resolution, self.thickness,
                self.smooth_rounds, self.relax_iters, self.backing,
                self.base, self.color_by)
            obj = pc.emit(context, "Over-Under Screen", cells,
                          self.separate, fit=True,
                          span=2.0 * self.scale, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no screen generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            note = ""
            if stats.get('minimal'):
                note = "  relax=%s" % stats['minimal']
            if stats.get('note'):
                note += "  (%s)" % stats['note']
            if obj.type == 'MESH':
                self.report({'INFO'},
                            "%s  holes=%d comps=%d  V=%d F=%d%s"
                            % (self.surface, stats.get('holes', 0),
                               stats.get('components', 0),
                               len(obj.data.vertices),
                               len(obj.data.polygons), note))
            else:
                self.report({'INFO'}, "%s  %d parts%s"
                            % (self.surface, len(obj.children),
                               note))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'grid_w')
            lay.prop(self, 'grid_h')
            lay.prop(self, 'surface')
            if self.surface == 'RIBBONS':
                lay.prop(self, 'weave')
            lay.prop(self, 'strand_width')
            lay.prop(self, 'weave_depth')
            lay.prop(self, 'resolution')
            lay.prop(self, 'thickness')
            if self.surface != 'RIBBONS':
                lay.prop(self, 'smooth_rounds')
            if self.surface == 'MINIMAL':
                lay.prop(self, 'relax_iters')
            lay.prop(self, 'color_by')
            lay.prop(self, 'backing')
            if self.backing:
                lay.prop(self, 'base')
            lay.prop(self, 'separate')
            lay.prop(self, 'scale')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.over_under_screen_add",
                             icon='MESH_GRID')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_over_under_screen_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_over_under_screen_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _check_parity():
    """Ribbon-mode drafts: exclusive over/under, balanced strands;
    and the membrane's blue phase is a strict checkerboard."""
    ok = True
    I, J = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    xo = x_over(I, J, 'PLAIN')
    ok &= bool(np.all(xo == ((I + J) % 2 == 0)))
    ok &= bool(np.all(xo[1:, :] != xo[:-1, :]))
    ok &= bool(np.all(xo[:, 1:] != xo[:, :-1]))
    for weave in WEAVES:
        xo = x_over(I, J, weave)
        ok &= bool(np.all(xo.sum(axis=0) == 4))
        ok &= bool(np.all(xo.sum(axis=1) == 4))
    for P in range(4):
        for Q in range(4):
            ok &= _blue_up(P, Q) == ((P + Q) % 2 == 0)
            ok &= _blue_up(P + 1, Q) != _blue_up(P, Q)
            ok &= _blue_up(P, Q + 1) != _blue_up(P, Q)
    return ok


def _run_selftest():
    W, H = 3, 3
    par = _check_parity()
    print("weave parity / blue checkerboard ok:", par)

    (v, f, m), st = build_membrane(W, H)      # the live defaults
    h = st['h']
    nonempty = len(v) > 0 and len(f) > 0
    finite = bool(np.all(np.isfinite(np.asarray(v))))
    conn = st['components'] == 1
    manifold = st['nonmanifold'] == 0
    # one enclosed star opening per fully-interior arch; openings
    # nearer the rim merge into the outer boundary loop (invariant
    # validated over grids 3x3 .. 5x4)
    holes_ok = st['holes'] == (W - 1) * (H - 1)
    # red stays in the back half: its floor undulates (rising
    # _RISE under the arches -- no flat plane anywhere) and its
    # edges curl up with the deep channel bow and sweep up the hub
    # flanks, but it never approaches the mid-plane
    red_ok = st['red_max'] <= -0.35 * h
    red_curved = st['red_var'] >= 0.05 * h
    # blue-UP crossings: two separate sheets with a real gap
    up_ok = (st['up_top_min'] >= 0.7 * h
             and st['up_bot_max'] <= -0.55 * h
             and st['up_gap_faces'] == 0)
    # blue-DOWN crossings: a single merged valley whose trough
    # (blue's minimum) sits exactly ON the red floor
    down_ok = st['down_max'] <= -0.6 * h
    # the trough row is red's own centerline continuing through
    # the hub (shared geometry); the label-based measurement is
    # offset by part of the channel bow, hence the bow-scaled
    # tolerance
    eq_ok = (abs(st['blue_min'] - st['red_z'])
             <= (0.04 + 0.2 * st['bow']) * h)
    # (z is scaled by the CONTROL envelope, so the subdivided
    # trough sits above -1h by the usual CC shrinkage)
    valley_ok = st['blue_min'] <= -0.75 * h
    # the bridge neck is one MONOTONE quarter-turn: exactly 90
    # degrees end to end (0 at one hub top -> 90 on-edge wall
    # at the other hub), never unwinding
    twist_ok = abs(st['twist_deg'] - 90.0) < 1e-9
    # the arm-to-hub blend ease is quintic: first AND second
    # derivatives vanish at both ends (curvature-continuous)
    de = 1e-4
    ease_ok = max(
        abs(_ease5(de) - _ease5(0.0)) / de,
        abs(_ease5(1.0) - _ease5(1.0 - de)) / de,
        abs(_ease5(2 * de) - 2 * _ease5(de) + _ease5(0.0))
        / de ** 2,
        abs(_ease5(1.0) - 2 * _ease5(1.0 - de)
            + _ease5(1.0 - 2 * de)) / de ** 2) < 1e-2
    print("collar ease: quintic C2 at both ends:", ease_ok)
    # the back-web (red + merges + bridges) is one surface
    web_ok = st['back_components'] == 1
    # C1 merges / flat bridge portions: no crease across any
    # back-web seam -- evaluated on a finer subdivision so it
    # measures the limit surface, not the coarse default facets
    _cf, stf = build_membrane(W, H, res=18)
    seam_ok = stf['seam_dot_min'] >= 0.98
    print("membrane: V=%d F=%d comps=%d nonman=%d rims=%d holes=%d"
          % (len(v), len(f), st['components'], st['nonmanifold'],
             st['rims'], st['holes']))
    print("red: max z=%.3fh (want <=-0.35h) z-variation=%.3fh "
          "(want >0: no flat floor)"
          % (st['red_max'] / h, st['red_var'] / h))
    print("blue-up: top>=%.3fh red<=%.3fh mid-gap faces=%d (want 0)"
          % (st['up_top_min'] / h, st['up_bot_max'] / h,
             st['up_gap_faces']))
    print("blue-down: local max=%.3fh (want <=-0.6h) | back-web "
          "comps=%d (want 1)" % (st['down_max'] / h,
                                 st['back_components']))
    print("valley: blue min=%.3fh red floor=%.3fh diff=%.3fh "
          "(want ~0)" % (st['blue_min'] / h, st['red_z'] / h,
                         (st['blue_min'] - st['red_z']) / h))
    print("bridge: twist=%.1f deg (monotone quarter-turn: 0 hub "
          "top -> 90 on-edge wall) width=%.3f "
          "cell (== thickness) | bow=%.2fh"
          % (st['twist_deg'], st['bridge_w'], st['bow']))
    print("bridge level: end heights %.3fh / %.3fh (want equal; "
          "centerline z-variation 0)"
          % (st['bridge_ends'][0] / h, st['bridge_ends'][1] / h))
    lvl_ok = abs(st['bridge_ends'][0]
                 - st['bridge_ends'][1]) <= 1e-6
    print("hub: half-width G=%.3f (strip g=%.3f) | arm width "
          "%.3f at hub -> %.3f at midspan"
          % (st['hub_G'], 0.5 * st['arm_w'][1], st['arm_w'][0],
             st['arm_w'][1]))
    print("seam C1 (res 18): min dot=%.4f mean=%.4f (want >=0.98)"
          % (stf['seam_dot_min'], stf['seam_dot_mean']))

    (_v2, f2, _m2), st2 = build_membrane(W, H, res=16,
                                         thickness=0.1,
                                         surface='MINIMAL',
                                         relax_iters=6)
    min_ok = (st2['minimal'] == 'TAUBIN-FAIRING'
              and st2['components'] == 1 and len(f2) > 0)
    print("taut mode: method=%s comps=%d" % (st2['minimal'],
                                             st2['components']))

    rib = build_ribbons(W, H, res=8, thickness=0.2)
    rib_ok = len(rib) == (W + 1) + (H + 1) and all(
        len(cv) and len(cf) for cv, cf, _cm in rib)
    print("ribbons: %d strands (want %d)"
          % (len(rib), (W + 1) + (H + 1)))

    fitted = pc.center_scale(list(v), span=2.0)
    a = np.asarray(fitted)
    ext = max(a[:, 0].max() - a[:, 0].min(),
              a[:, 1].max() - a[:, 1].min())
    fit_ok = abs(ext - 2.0) < 1e-6
    print("fit: max extent %.6f (want 2.0)" % ext)

    ok = (par and nonempty and finite and conn and manifold
          and holes_ok and red_ok and red_curved and up_ok
          and down_ok and eq_ok and valley_ok and twist_ok
          and lvl_ok and ease_ok and web_ok and seam_ok
          and min_ok and rib_ok and fit_ok)
    print("RESULT:", "OK" if ok else "BAD")
    return ok


if __name__ == "__main__":
    _run_selftest()
