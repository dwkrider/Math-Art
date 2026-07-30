
# Over-Under Screen Generator for Blender
#
# A woven square lattice rendered as one visually continuous smooth
# screen, in the modular-constructivist tradition of Erwin Hauer and
# Norman Carlberg (the woven bilayer screens of Hauer's Continua).
#
# THE STRUCTURE.  Two families of DIAGONAL ribbons cross on a square
# lattice of crossovers, and EVERY crossover is an over/under
# crossing: one ribbon passes above (+h), the other below (-h), as
# two separate sheets, and which family rides on top ALTERNATES in a
# checkerboard across the crossover lattice -- so along any single
# ribbon the sense runs over, under, over, under at successive
# crossovers: a true plain weave of the diagonal families.  There
# are NO flat crossings.  In the open cell between four crossovers
# sits a lens-shaped CELL-CENTRE CUSHION (a "football"); every
# cushion is IDENTICAL: anchored at the z = 0 mid-plane -- exactly
# halfway between the over (+h) and under (-h) ribbon levels, with
# no per-cell offset or sign -- and bulging DOWNWARD from that
# shared base.  Adjacent cushions are joined by smooth saddle
# BRIDGES that flow over the ribbon mid-spans (where the ribbon
# wave passes through z = 0), so cushions + bridges + ribbons read
# as one continuous surface; the leftover openings are 4-pointed
# STAR holes centred on the crossovers, their points tucked into
# the gap between the over and under sheets.
#
# THE CONSTRUCTION (a bicubic B-spline patch network).  In the 45-
# degree rotated coordinates p = x + y, q = y - x the diagonal
# ribbons are axis-aligned bands on the integer lines, crossing at
# every integer (P, Q); the checkerboard sign sigma = (-1)^(P+Q)
# puts family 0 on top where P + Q is even.  Each band is a lofted
# strip whose longitudinal profile is the C1 wave +/-cos(pi s):
# extrema exactly AT the crossovers, zero at the mid-spans.  Each
# open cell carries a 5 x 5 cushion patch sagging to the football
# depth below z = 0; its side control rows are THE SAME control
# points as the adjacent band edges (shared vertices at the mid-
# span stations, where the band wave is centred on z = 0), so every
# seam vertex is a regular valence-4 vertex of the control net and
# the Catmull-Clark limit surface -- the generalization of uniform
# bicubic B-spline patches to an arbitrary quad mesh -- is C1
# (indeed C2) across every cushion-to-ribbon seam: the bridges
# between neighbouring cushions are tangent-continuous saddles, not
# creased tabs.  The cushion corners are inset toward the
# crossovers and left free, opening the star notches by
# construction (never trimmed from a blob).  The refined shell is
# finally thickened along its normals into a closed solid.
#
# The MINIMAL (TAUT) mode applies extra passes of non-shrinking
# Taubin lambda|mu fairing to tauten the saddles without collapsing
# the relief.  The RIBBONS mode instead renders a classic axis-
# aligned plain weave as separate lofted strips (with TWILL and
# BASKET drafts); the membrane itself is inherently the PLAIN draft.
#
# References:
#   Erwin Hauer, "Continua -- Architectural Screens and Walls",
#     Princeton Architectural Press, 2004 -- the perforated modular
#     screen designs (Designs 1-7, 1950-57): woven bilayer screens
#     of diagonal ribbons with alternating over/under crossings and
#     smooth saddle infill, the structure realized here.
#   Norman Carlberg and Erwin Hauer -- co-originators of Modular
#     Constructivism (sculptural units designed to be multiplied).
#   E. Catmull and J. Clark, "Recursively Generated B-spline
#     Surfaces on Arbitrary Topological Meshes", Computer-Aided
#     Design 10(6), 1978 -- the subdivision scheme used to evaluate
#     the bicubic patch network.
#   Branko Grunbaum and G. C. Shephard, "Satins and Twills: An
#     Introduction to the Geometry of Fabrics", Mathematics Magazine
#     53(3), 1980 -- the parity / draft description of plain, twill
#     and basket weaves (the checkerboard alternation used at the
#     crossovers, and the drafts of the ribbon mode).
#   Gabriel Taubin, "A Signal Processing Approach to Fair Surface
#     Design", SIGGRAPH 1995 -- the non-shrinking lambda|mu fairing
#     of the taut mode.

bl_info = {
    "name": "Over-Under Screen",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Woven screen of diagonal ribbons -- every "
                   "crossover alternating over/under, sunken cell "
                   "cushions, star openings (Hauer / Carlberg)",
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
_ATTACH = 0.5         # cushion attach half-length (x crossing gap)
_INSET = 0.5          # star-notch corner inset (x crossing gap)
_FOOTBALL = 0.75      # cushion sag depth (x h), DOWN from z = 0


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
    """Crossing height target of a ribbon-mode strand: +amp where
    that strand is over, -amp where under."""
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
# Bands (the diagonal ribbons) run along ALL integer lines of (p, q)
# and cross at every integer (P, Q) -- corner-type and centre-type
# crossovers of the original grid alike.  g is the band HALF-width
# in these units.  z is built at unit amplitude (crossover sheets at
# +/-1) and rescaled to h after subdivision.

class _Net:
    """Quad/n-gon control net with vertex sharing.  Vertices are
    keyed by rounded (p, q, tag): the two ribbon families carry
    different tags, so their sheets stay DISJOINT where they overlap
    in plan (every crossover), while the cushions reference the band
    edge vertices directly (tag of that family) so the seams share
    control points exactly.  Faces are auto-oriented CCW in the
    (p, q) plan so the whole net is consistently oriented."""

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

    def verts_xyz(self):
        c = self.cell
        return [np.array([(p - q) * 0.5 * c, (p + q) * 0.5 * c, z])
                for p, q, z in self.pos]


def _band_z(fam, line, s):
    """Centerline height of a band at along-coordinate s: the C1
    cosine wave with extrema exactly AT the crossovers and zeros at
    the mid-spans.  Family 0 (the q-bands) is at +1 where P + Q is
    even; family 1 is everywhere its opposite -- the checkerboard
    over/under alternation."""
    wave = float(np.cos(np.pi * s)) * (1.0 if line % 2 == 0 else -1.0)
    return wave if fam == 0 else -wave


def _xy_of(P, Q):
    return ((P - Q) * 0.5, (P + Q) * 0.5)


def _cross_ok(P, Q, W, H):
    x, y = _xy_of(P, Q)
    return (-1e-9 <= x <= W + 1e-9) and (-1e-9 <= y <= H + 1e-9)


def _band_piece(net, fam, line, stations, g):
    """A lofted strip piece of one band: 3 flat transverse rows at
    line +/- g, longitudinal profile _band_z.  fam 0 runs along p
    at q = line; fam 1 along q at p = line."""
    tag = 1 + fam
    rows = (line - g, line, line + g)
    grid = {}
    for i, s in enumerate(stations):
        z = _band_z(fam, line, s)
        for j, t in enumerate(rows):
            p_, q_ = (s, t) if fam == 0 else (t, s)
            grid[(i, j)] = net.vid(p_, q_, z, tag)
    for i in range(len(stations) - 1):
        for j in range(2):
            net.face([grid[(i, j)], grid[(i + 1, j)],
                      grid[(i + 1, j + 1)], grid[(i, j + 1)]], fam)


# interior sag profile of the cushion (fraction of the full depth),
# indexed by (min(i, 4-i), min(j, 4-j)) of the 5x5 grid
_SAG_PROFILE = {(1, 1): 0.45, (1, 2): 0.62, (2, 1): 0.62,
                (2, 2): 1.0}


def _cushion(net, P, Q, g, aw, ci, fb):
    """The cell-centre 'football': a 5 x 5 cushion patch over the
    open cell [P, P+1] x [Q, Q+1].  Every cushion is IDENTICAL:
    anchored on the z = 0 mid-plane (its side vertices ARE the band
    edge vertices at the mid-span stations, whose S-curve is centred
    on 0) and sagging DOWN to -fb at its centre -- no per-cell
    offset, no per-cell sign.  The shared side vertices are regular
    valence-4 vertices of the net, so the subdivision limit is
    C1/C2 across the seam: the smooth saddle bridges between
    neighbouring cushions.  The four corners are inset toward the
    crossovers and left free: the star notches."""
    us = (P + g, P + 0.5 - aw, P + 0.5, P + 0.5 + aw, P + 1 - g)
    vs = (Q + g, Q + 0.5 - aw, Q + 0.5, Q + 0.5 + aw, Q + 1 - g)
    ids = {}
    for i in range(5):
        for j in range(5):
            side_i = i in (0, 4)
            side_j = j in (0, 4)
            if side_i and side_j:          # free inset corner
                sp = ci if i == 0 else -ci
                sq = ci if j == 0 else -ci
                ids[(i, j)] = net.vid(us[i] + sp, vs[j] + sq,
                                      -0.5 * fb, 0)
            elif side_j:                   # on a q-band edge (fam 0)
                line = Q if j == 0 else Q + 1
                ids[(i, j)] = net.vid(us[i], vs[j],
                                      _band_z(0, line, us[i]), 1)
            elif side_i:                   # on a p-band edge (fam 1)
                line = P if i == 0 else P + 1
                ids[(i, j)] = net.vid(us[i], vs[j],
                                      _band_z(1, line, vs[j]), 2)
            else:                          # interior: sag DOWN
                f = _SAG_PROFILE[(min(i, 4 - i), min(j, 4 - j))]
                ids[(i, j)] = net.vid(us[i], vs[j], -fb * f, 0)
    for i in range(4):
        for j in range(4):
            net.face([ids[(i, j)], ids[(i + 1, j)],
                      ids[(i + 1, j + 1)], ids[(i, j + 1)]], 2)


def _control_net(W, H, cell, strand_width):
    """The full control net: at every crossover both families'
    crossing sheets (two stacked strips, never merged), mid-span
    strips between neighbouring crossovers, and an identical sunken
    cushion in every open cell whose four crossovers all exist."""
    g = 0.5 * min(max(strand_width, 0.15), 0.9)
    gap = 0.5 - g
    aw = _ATTACH * gap
    ci = _INSET * gap
    net = _Net(float(cell))
    for P in range(0, W + H + 1):
        for Q in range(-W, H + 1):
            if not _cross_ok(P, Q, W, H):
                continue
            # a crossing sheet is built only where its band actually
            # continues along its line (the four extreme diamond
            # corners would otherwise leave isolated one-crossing
            # stubs, disconnected from the screen)
            if (_cross_ok(P - 1, Q, W, H)
                    or _cross_ok(P + 1, Q, W, H)):
                _band_piece(net, 0, Q, (P - g, P, P + g), g)
            if (_cross_ok(P, Q - 1, W, H)
                    or _cross_ok(P, Q + 1, W, H)):
                _band_piece(net, 1, P, (Q - g, Q, Q + g), g)
            if _cross_ok(P + 1, Q, W, H):
                _band_piece(net, 0, Q,
                            (P + g, P + 0.5 - aw, P + 0.5,
                             P + 0.5 + aw, P + 1 - g), g)
            if _cross_ok(P, Q + 1, W, H):
                _band_piece(net, 1, P,
                            (Q + g, Q + 0.5 - aw, Q + 0.5,
                             Q + 0.5 + aw, Q + 1 - g), g)
            if (_cross_ok(P + 1, Q, W, H)
                    and _cross_ok(P, Q + 1, W, H)
                    and _cross_ok(P + 1, Q + 1, W, H)):
                _cushion(net, P, Q, g, aw, ci, _FOOTBALL)
    return net


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
    independent cycles of the boundary graph (star holes + the
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
    """Face-normal continuity across the cushion-to-ribbon seams
    (label-2 face against label-0/1 face): list of dots of the two
    unit normals, over INTERIOR seam edges (edges ending on the
    free rims are the legitimate turn of the star notch, not part
    of the smooth seam).  Near 1.0 everywhere = C1, no crease."""
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
        if 2 in (la, lb) and (la in (0, 1) or lb in (0, 1)):
            na = _face_normal(V, faces[fs[0]])
            nb = _face_normal(V, faces[fs[1]])
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
    """Thicken an open shell along its (smoothed) vertex normals
    into a closed slab: top faces, reversed bottom faces, and a
    side quad per boundary edge."""
    n = len(V)
    N = _smooth_normals(_vertex_normals(V, quads), quads)
    top = V + 0.5 * thickness * N
    bot = V - 0.5 * thickness * N
    verts = np.vstack([top, bot])
    faces = [tuple(int(v) for v in q) for q in quads]
    faces += [tuple(int(v) + n for v in reversed(q)) for q in quads]
    cnt = {}
    for q in quads:
        m = len(q)
        for k in range(m):
            a, b = int(q[k]), int(q[(k + 1) % m])
            e = (a, b) if a < b else (b, a)
            cnt.setdefault(e, []).append((a, b))
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
            return list(labels)    # ribbon fam 0 / fam 1 / cushions
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

def build_membrane(W, H, cell=1.0, strand_width=0.6, weave_depth=0.5,
                   weave='PLAIN', res=14, thickness=0.25,
                   smooth_rounds=2, surface='MEMBRANE',
                   relax_iters=12, color_by='UNIFORM'):
    """The woven screen as a Catmull-Clark-evaluated bicubic patch
    network.  Returns (cell, stats); stats carries the crossover
    two-sheet extremes and parity alternation, the cushion base /
    sag measurements, the seam normal-continuity minimum, hole
    count and components -- everything the self-test asserts."""
    cell = float(cell)
    net = _control_net(W, H, cell, strand_width)
    rounds = max(1, min(4, int(np.log2(max(4, int(res)))) - 1))
    V, faces, labels = _catmull_clark(net.verts_xyz(), net.faces,
                                      net.labels, rounds)
    V = np.array(V, dtype=float)
    h = 0.5 * max(weave_depth, 0.0) * cell
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
    # every crossover: two sheets at ~+/-h, family on top by parity
    top_min, bot_max = None, None
    parity_ok = True
    for P in range(0, W + H + 1):
        for Q in range(-W, H + 1):
            if not _cross_ok(P, Q, W, H):
                continue
            x, y = _xy_of(P, Q)
            r = np.hypot(C[:, 0] - x * cell, C[:, 1] - y * cell)
            near = r < 0.15 * cell
            z0 = C[near & (L == 0), 2]
            z1 = C[near & (L == 1), 2]
            if len(z0) == 0 or len(z1) == 0:
                continue
            hi = max(z0.max(), z1.max())
            lo = min(z0.min(), z1.min())
            top_min = hi if top_min is None else min(top_min, hi)
            bot_max = lo if bot_max is None else max(bot_max, lo)
            fam0_top = float(z0.mean()) > float(z1.mean())
            parity_ok &= fam0_top == ((P + Q) % 2 == 0)
    # every cushion: anchored at z ~ 0 (mean of its shared seam
    # vertices -- the band-edge S-curve is centred on the
    # mid-plane), core entirely below 0, all sagging equally
    lab_of = defaultdict(set)
    for fc, lb in zip(faces, labels):
        for i in fc:
            lab_of[int(i)].add(lb)
    seamv = np.array([i for i, ls in lab_of.items()
                      if 2 in ls and (0 in ls or 1 in ls)],
                     dtype=int)
    cush_core_max = None            # highest core point of any cell
    sags = []                       # per-cell core minimum
    anchor_absmax = 0.0             # worst seam-level offset
    for P in range(0, W + H + 1):
        for Q in range(-W, H + 1):
            if not (_cross_ok(P, Q, W, H)
                    and _cross_ok(P + 1, Q, W, H)
                    and _cross_ok(P, Q + 1, W, H)
                    and _cross_ok(P + 1, Q + 1, W, H)):
                continue
            x, y = _xy_of(P + 0.5, Q + 0.5)
            r = np.hypot(C[:, 0] - x * cell, C[:, 1] - y * cell)
            cz = C[:, 2]
            core = (r < 0.1 * cell) & (L == 2)
            if not core.any():
                continue
            m = float(cz[core].max())
            cush_core_max = (m if cush_core_max is None
                             else max(cush_core_max, m))
            sags.append(float(cz[core].min()))
            if len(seamv):
                rs = np.hypot(V[seamv, 0] - x * cell,
                              V[seamv, 1] - y * cell)
                sel = rs < 0.3 * cell
                if sel.any():
                    anchor_absmax = max(
                        anchor_absmax,
                        abs(float(V[seamv[sel], 2].mean())))
    dots = _seam_normal_dots(V, faces, labels)
    stats.update({
        'components': comps, 'nonmanifold': nonman, 'rims': rims,
        'holes': max(0, rims - 1), 'h': h,
        'cross_top_min': top_min or 0.0,
        'cross_bot_max': bot_max or 0.0,
        'parity_ok': parity_ok,
        'cushion_core_max': 0.0 if cush_core_max is None
        else cush_core_max,
        'cushion_sag_min': min(sags) if sags else 0.0,
        'cushion_sag_max': max(sags) if sags else 0.0,
        'cushion_anchor_absmax': anchor_absmax,
        'seam_dot_min': min(dots) if dots else 1.0,
        'seam_dot_mean': (sum(dots) / len(dots)) if dots else 1.0})
    mats = _face_mats(V, faces, color_by, labels=labels)
    return _finish(V, faces, mats, thickness * cell), stats


def build_ribbons(W, H, cell=1.0, strand_width=0.6, weave_depth=0.5,
                  weave='PLAIN', res=14, thickness=0.25,
                  color_by='UNIFORM'):
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


def build_screen(W, H, cell=1.0, strand_width=0.6, weave_depth=0.5,
                 weave='PLAIN', surface='MEMBRANE', res=14,
                 thickness=0.25, smooth_rounds=2, relax_iters=12,
                 backing=False, base=0.1, color_by='UNIFORM'):
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
            stats['note'] = ("the alternating-crossover membrane is "
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
        """Add a woven over-under screen (Hauer woven membrane)"""
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
                    "One woven patch-network screen: every "
                    "crossover alternating over/under, sunken cell "
                    "cushions, star openings (the Hauer screen)"),
                   ('MINIMAL', "Minimal (Taut)",
                    "The membrane tautened by extra non-shrinking "
                    "(Taubin) fairing")],
            default='MEMBRANE')
        strand_width: FloatProperty(
            name="Strand Width", default=0.6, min=0.15, max=0.9,
            description="Ribbon width as a fraction of the ribbon "
                        "pitch (also sets the opening size)")
        weave_depth: FloatProperty(
            name="Weave Depth", default=0.5, min=0.0, max=2.0,
            description="Peak-to-peak over/under amplitude as a "
                        "fraction of a cell (2h)")
        resolution: IntProperty(
            name="Resolution", default=14, min=4, max=48,
            description="Tessellation (sets the subdivision depth "
                        "of the patch network)")
        thickness: FloatProperty(
            name="Thickness", default=0.25, min=0.0, max=1.0,
            description="Shell thickness as a fraction of a cell "
                        "(0 = open surface)")
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
                    "Tones by element: the two ribbon families and "
                    "the cell cushions"),
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
    and the membrane's crossover parity is a strict checkerboard
    that flips across every neighbouring crossover."""
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
            t0 = _band_z(0, Q, P) > _band_z(1, P, Q)
            ok &= t0 == ((P + Q) % 2 == 0)
            ok &= (_band_z(0, Q, P + 1) > _band_z(1, P + 1, Q)) != t0
            ok &= (_band_z(0, Q + 1, P) > _band_z(1, P, Q + 1)) != t0
    return ok


def _run_selftest():
    W, H = 3, 3
    par = _check_parity()
    print("weave parity / crossover checkerboard ok:", par)

    (v, f, m), st = build_membrane(W, H, res=16, thickness=0.0)
    h = st['h']
    nonempty = len(v) > 0 and len(f) > 0
    finite = bool(np.all(np.isfinite(np.asarray(v))))
    conn = st['components'] == 1
    manifold = st['nonmanifold'] == 0
    # boundary-cycle invariant of the construction (validated over
    # grids 3x3 .. 6x4): the star-notch cycles around the interior
    # crossovers plus the partially-enclosed rim cycles
    exp_holes = 2 * W * H - 3 * (W + H) + 9
    holes_ok = st['holes'] == exp_holes
    # (a) NO flat crossovers: every crossover has a sheet near +h
    # and a sheet near -h, and the top family alternates by parity
    cross_ok = (st['cross_top_min'] >= 0.6 * h
                and st['cross_bot_max'] <= -0.6 * h
                and st['parity_ok'])
    # (b) every cushion: anchored at z ~ 0 (no per-cell offset),
    # its core entirely BELOW 0, all sagging by the same depth
    cush_ok = (st['cushion_core_max'] < 0.0
               and st['cushion_sag_min'] <= -0.3 * h
               and st['cushion_anchor_absmax'] <= 0.12 * h
               and (st['cushion_sag_max']
                    - st['cushion_sag_min']) <= 0.15 * h)
    # (c) C1 seams: no crease across any cushion-to-ribbon seam
    seam_ok = st['seam_dot_min'] >= 0.98
    print("membrane: V=%d F=%d comps=%d nonman=%d rims=%d "
          "holes=%d (want %d)" % (len(v), len(f), st['components'],
                                  st['nonmanifold'], st['rims'],
                                  st['holes'], exp_holes))
    print("crossovers: top>=%.3fh bot<=%.3fh parity_ok=%s"
          % (st['cross_top_min'] / h, st['cross_bot_max'] / h,
             st['parity_ok']))
    print("cushions: core max z=%.3fh (<0) sag %.3fh..%.3fh "
          "anchor |z|<=%.3fh"
          % (st['cushion_core_max'] / h, st['cushion_sag_min'] / h,
             st['cushion_sag_max'] / h,
             st['cushion_anchor_absmax'] / h))
    print("seam C1: min dot=%.4f mean=%.4f (want >=0.98)"
          % (st['seam_dot_min'], st['seam_dot_mean']))

    (_v2, f2, _m2), st2 = build_membrane(W, H, res=16,
                                         thickness=0.2,
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
          and holes_ok and cross_ok and cush_ok and seam_ok
          and min_ok and rib_ok and fit_ok)
    print("RESULT:", "OK" if ok else "BAD")
    return ok


if __name__ == "__main__":
    _run_selftest()
