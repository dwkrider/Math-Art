
# Over-Under Screen Generator for Blender
#
# A woven square lattice rendered as one visually continuous smooth
# screen, in the modular-constructivist tradition of Erwin Hauer and
# Norman Carlberg (Hauer's Design 1, the woven bilayer screen).
#
# THE STRUCTURE.  Label a repeating unit by a 3 x 3 node grid
#
#         A B C
#         D E F
#         G H I
#
# The crossings ALTERNATE -- this is NOT an axis-aligned plain weave:
#   * A, C, G, I (the unit corners) are FLAT crossings at the middle
#     level: two ribbons meet there at z ~ 0, tangent-plane flat.
#   * E (the unit centre) is the OVER/UNDER crossing: one ribbon
#     passes above (+h), the other below (-h), as TWO SEPARATE
#     sheets -- the strips overlap in plan but are never merged.
#   * B, D, F, H (the edge midpoints) are not crossings but SADDLE
#     BRIDGES joining the flat crossings (B joins A-C, D joins A-G,
#     F joins C-I, H joins G-I).
# The ribbons run on the DIAGONALS: one family A -> E -> I, the
# other C -> E -> G.  Every lattice corner (a, b) is a flat
# crossing; the two diagonal ribbons through it share the layer
# parity s = (-1)^(a+b), so both arrive at their common extremum
# z = 0 and meet flush there.  The upper layer (s = +1) is a
# diagonal lattice of ribbons bulging up to +h at its centres; the
# lower layer (s = -1) mirrors it below; at each centre E an upper
# and a lower ribbon cross at +h / -h WITHOUT touching.  The axis-
# aligned bridges each join an upper corner to a lower corner
# (adjacent corners always have opposite parity), sweeping through
# z = 0 -- the saddles that tie the two layers into one screen.
# The leftover openings are the 4-pointed STAR holes centred on the
# over/under crossings, their points reaching toward the bridges.
#
# THE CONSTRUCTION (a bicubic B-spline patch network).  In the 45-
# degree rotated coordinates p = x + y, q = y - x the diagonal
# ribbons become an axis-aligned lattice, and the surface is built
# as an explicit quad CONTROL NET: 3 x 3 junction patches at the
# flat crossings, lofted ribbon strips through the over/under
# crossings (their longitudinal profile the C1 wave sin^2(pi u / 2),
# 0 at flat crossings and +/-1 at woven ones), and bridge strips
# across each rotated cell, attached through shoulder vertices
# shared with the ribbon edges -- every connection is a deliberately
# shared control edge, and the two ribbon families keep disjoint
# control points wherever they cross over/under.  The net is
# refined by CATMULL-CLARK subdivision -- the generalization of
# uniform bicubic B-spline patches to an arbitrary quad control
# mesh -- so the limit surface is a smooth spline patch network,
# watertight by construction, with the star holes left open (never
# trimmed from a blob).  The refined shell is finally thickened
# along its normals into a closed solid.
#
# The MINIMAL (TAUT) mode applies extra passes of non-shrinking
# Taubin lambda|mu fairing to tauten the saddles without collapsing
# the relief.  The RIBBONS mode instead renders a classic axis-
# aligned plain weave as separate lofted strips (with TWILL and
# BASKET drafts); the alternating-crossing membrane itself is the
# PLAIN structure and has no twill analogue.
#
# References:
#   Erwin Hauer, "Continua -- Architectural Screens and Walls",
#     Princeton Architectural Press, 2004 -- the perforated modular
#     screen designs (Designs 1-7, 1950-57); Design 1 is the woven
#     bilayer screen with flat and over/under crossings alternating,
#     the structure realized here.
#   Norman Carlberg and Erwin Hauer -- co-originators of Modular
#     Constructivism (sculptural units designed to be multiplied).
#   E. Catmull and J. Clark, "Recursively Generated B-spline
#     Surfaces on Arbitrary Topological Meshes", Computer-Aided
#     Design 10(6), 1978 -- the subdivision scheme used to evaluate
#     the bicubic patch network.
#   Branko Grunbaum and G. C. Shephard, "Satins and Twills: An
#     Introduction to the Geometry of Fabrics", Mathematics Magazine
#     53(3), 1980 -- the parity / draft description of plain, twill
#     and basket weaves (used by the ribbon mode).
#   Gabriel Taubin, "A Signal Processing Approach to Fair Surface
#     Design", SIGGRAPH 1995 -- the non-shrinking lambda|mu fairing
#     of the taut mode.

bl_info = {
    "name": "Over-Under Screen",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Woven screen of diagonal ribbons -- flat and "
                   "over/under crossings alternating, saddle "
                   "bridges, star openings (Hauer / Carlberg)",
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
_SAG = -0.4           # junction centre control (x edge value)
_SHOULDER = 0.45      # bridge shoulder inset (x band-edge gap)
_BRIDGE_W = 2.0       # bridge width (x shoulder inset)


# --------------------------------------------------------------------
# Weave drafts (over/under parity; used by the RIBBONS mode)
# --------------------------------------------------------------------

WEAVES = ('PLAIN', 'TWILL', 'BASKET')


def x_over(i, j, weave='PLAIN'):
    """Boolean (arrays ok): x-strand over at crossing (i, j) of the
    axis-aligned ribbon weave."""
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
# All layout happens in (p, q), where the diagonal ribbons are axis-
# aligned: lattice corners (a, b) sit at (a+b, b-a), flat crossings
# every 2 units along a band, woven crossings between.  g is the
# band HALF-width in these units (band width = strand_width x the
# ribbon pitch).  z is built at unit amplitude and rescaled to h
# after subdivision.

def _prof(u):
    """Longitudinal band profile: 0 (flat crossing, zero slope) to
    1 (woven crossing, zero slope), C1."""
    return float(np.sin(0.5 * np.pi * u) ** 2)


class _Net:
    """Quad/n-gon control net with vertex sharing.  Vertices are
    keyed by rounded (p, q, layer-tag) so patches that must join
    share control points exactly -- and the two ribbon layers, with
    opposite tags, stay DISJOINT where they overlap in plan (the
    over/under crossings).  Faces are auto-oriented CCW in the
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

    def has(self, p, q, tag):
        return (round(p, 6), round(q, 6), tag) in self.key

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


def _layer(a, b):
    """Flat-crossing layer parity: +1 (upper) iff a + b is even."""
    return 1 if (a + b) % 2 == 0 else -1


def _junction(net, a, b, g):
    """3 x 3 patch at flat crossing (a, b): ring controls continue
    the four band arms (z = prof(g) x layer), the centre sags past
    zero so the limit surface meets flat at z ~ 0."""
    s = _layer(a, b)
    P, Q = a + b, b - a
    ct = _prof(g) * s
    ps = (P - g, P, P + g)
    qs = (Q - g, Q, Q + g)
    ids = [[net.vid(ps[i], qs[j],
                    _SAG * ct if (i == 1 and j == 1) else ct, s)
            for j in range(3)] for i in range(3)]
    for i in range(2):
        for j in range(2):
            net.face([ids[i][j], ids[i + 1][j],
                      ids[i + 1][j + 1], ids[i][j + 1]], 2)


def _segment(net, a, b, axis, g, d):
    """One diagonal ribbon span between adjacent flat crossings,
    passing its woven crossing at +/-1: axis 0 is the A->E->I
    family (corner (a,b) -> (a+1,b+1)), axis 1 the C->E->G family
    (corner (a+1,b) -> (a,b+1)).  A lofted strip of 5 stations x 3
    rows; where the two families cross over/under their vertices
    carry opposite layer tags, so the strips remain separate
    sheets.  The outer-row faces adjacent to a junction pick up the
    bridge SHOULDER vertices (if built) so the bridges share edges
    with the band -- those faces become pentagons."""
    if axis == 0:
        s = _layer(a, b)
        P0, Q0 = a + b, b - a

        def co(u, v):
            return (P0 + u, Q0 + v)
    else:
        s = _layer(a + 1, b)
        P0, Q0 = a + b + 1, b - a - 1

        def co(u, v):
            return (P0 + v, Q0 + u)
    off = (g, 1.0 - g, 1.0, 1.0 + g, 2.0 - g)
    zs = [s * _prof(u) for u in off]
    grid = {}
    for i, u in enumerate(off):
        for j, v in enumerate((-g, 0.0, g)):
            p_, q_ = co(u, v)
            grid[(i, j)] = net.vid(p_, q_, zs[i], s)
    zsh = s * _prof(g + d)
    for jm, jo, vo in ((1, 0, -g), (1, 2, g)):
        for i in range(4):
            poly = [grid[(i, jm)], grid[(i + 1, jm)],
                    grid[(i + 1, jo)]]
            if i == 3:
                # the far junction has the SAME layer parity (a+b
                # and a+b+2), and prof is symmetric about the woven
                # crossing, so the far shoulder is tag s at +zsh
                p_, q_ = co(2.0 - g - d, vo)
                if net.has(p_, q_, s):
                    poly.append(net.vid(p_, q_, zsh, s))
            if i == 0:
                p_, q_ = co(g + d, vo)
                if net.has(p_, q_, s):
                    poly.append(net.vid(p_, q_, zsh, s))
            poly.append(grid[(i, jo)])
            net.face(poly, axis)


def _bridge(net, a, b, horizontal, g, d, bw, W, H):
    """Saddle bridge along the lattice edge (a,b)->(a+1,b) or
    (a,b)->(a,b+1): a strip across the rotated cell joining the
    upper junction corner (+prof(g)) to the lower one (-prof(g))
    through z = 0.  Its end caps attach through SHOULDER vertices
    placed on the flanking band edges (shared with the band
    pentagons); a shoulder triangle closes the notch at the junction
    corner.  At the outer rim a missing arm degrades the cap to a
    quad / triangle on the corner vertex."""
    s1 = _layer(a, b)
    P1, Q1 = a + b, b - a
    e = -1 if horizontal else 1
    ct = _prof(g)
    zsh = _prof(g + d)
    Cn = (P1 + g, Q1 + e * g)
    Cf = (P1 + 1.0 - g, Q1 + e * (1.0 - g))
    if horizontal:
        near = (((Cn[0] + d, Cn[1]), b < H),
                ((Cn[0], Cn[1] - d), b > 0))
        far = (((Cf[0] - d, Cf[1]), b > 0),
               ((Cf[0], Cf[1] + d), b < H))
    else:
        near = (((Cn[0] + d, Cn[1]), a < W),
                ((Cn[0], Cn[1] + d), a > 0))
        far = (((Cf[0] - d, Cf[1]), a > 0),
               ((Cf[0], Cf[1] - d), a < W))
    vCn = net.vid(Cn[0], Cn[1], ct * s1, s1)
    vCf = net.vid(Cf[0], Cf[1], -ct * s1, -s1)
    dp = np.array([Cf[0] - Cn[0], Cf[1] - Cn[1]])
    dh = dp / np.hypot(dp[0], dp[1])
    nh = np.array([-dh[1], dh[0]])
    st = []
    for t in (0.38, 0.62):
        Pt = np.array(Cn) + t * dp
        z = ct * s1 * (1.0 - 2.0 * t)
        A = net.vid(Pt[0] - nh[0] * bw * 0.5,
                    Pt[1] - nh[1] * bw * 0.5, z, 0)
        B = net.vid(Pt[0] + nh[0] * bw * 0.5,
                    Pt[1] + nh[1] * bw * 0.5, z, 0)
        st.append((A, B))

    def cap(vC, Cxy, specs, ssh, A, B):
        sh = []
        for (pq, ok) in specs:
            if not ok:
                continue
            side = ((pq[0] - Cxy[0]) * nh[0]
                    + (pq[1] - Cxy[1]) * nh[1])
            sh.append((side, net.vid(pq[0], pq[1], zsh * ssh, ssh)))
        if len(sh) == 2:
            sh.sort()                      # A (-n) side first
            net.face([sh[0][1], A, B, sh[1][1]], 3)
            net.face([vC, sh[0][1], sh[1][1]], 3)
        elif len(sh) == 1:
            if sh[0][0] < 0.0:
                net.face([sh[0][1], A, B, vC], 3)
            else:
                net.face([vC, A, B, sh[0][1]], 3)
        else:
            net.face([vC, A, B], 3)

    cap(vCn, Cn, near, s1, st[0][0], st[0][1])
    net.face([st[0][0], st[0][1], st[1][1], st[1][0]], 3)
    cap(vCf, Cf, far, -s1, st[1][0], st[1][1])


def _control_net(W, H, cell, strand_width):
    """The full control net: junctions at every corner, a bridge on
    every lattice edge (rim included -- the finished border), and a
    diagonal ribbon span of each family per cell."""
    g = 0.5 * min(max(strand_width, 0.15), 0.9)
    d = _SHOULDER * (1.0 - 2.0 * g)
    bw = _BRIDGE_W * d
    net = _Net(float(cell))
    for a in range(W + 1):
        for b in range(H + 1):
            _junction(net, a, b, g)
    for a in range(W + 1):
        for b in range(H + 1):
            if a < W:
                _bridge(net, a, b, True, g, d, bw, W, H)
            if b < H:
                _bridge(net, a, b, False, g, d, bw, W, H)
    for a in range(W):
        for b in range(H):
            _segment(net, a, b, 0, g, d)
            _segment(net, a, b, 1, g, d)
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
            return list(labels)        # NE / NW / junction / bridge
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
    """The fused screen as a Catmull-Clark-evaluated bicubic patch
    network.  Returns (cell, stats); stats includes the hole count
    (rims minus the outer loop), components, the flat-crossing max
    |z|, the over/under z-span and the count of stray mid-level
    vertices at the woven crossings (two-sheet verification)."""
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
    # verification: flat crossings ~ 0; at every woven crossing two
    # SEPARATE sheets ~2h apart with an empty mid-band between them
    flat_max = 0.0
    for a in range(W + 1):
        for b in range(H + 1):
            r = np.hypot(V[:, 0] - a * cell, V[:, 1] - b * cell)
            zz = V[r < 0.12 * cell, 2]
            if len(zz):
                flat_max = max(flat_max, float(np.abs(zz).max()))
    span_min, stray = None, 0
    for a in range(W):
        for b in range(H):
            r = np.hypot(V[:, 0] - (a + 0.5) * cell,
                         V[:, 1] - (b + 0.5) * cell)
            zz = V[r < 0.12 * cell, 2]
            if len(zz):
                sp = float(zz.max() - zz.min())
                span_min = sp if span_min is None else min(span_min,
                                                           sp)
                stray += int(np.sum(np.abs(zz) < 0.4 * h))
    stats.update({'components': comps, 'nonmanifold': nonman,
                  'rims': rims, 'holes': max(0, rims - 1),
                  'flat_max': flat_max, 'span_min': span_min or 0.0,
                  'stray_mid': stray, 'h': h})
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
            stats['note'] = ("the alternating-crossing membrane is "
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
        """Add a woven over-under screen (Hauer bilayer membrane)"""
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
                    "only draft of the alternating-crossing "
                    "screen)"),
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
                    "One fused patch-network screen: flat and "
                    "over/under crossings alternating, saddle "
                    "bridges, star openings (the Hauer screen)"),
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
                    "Tones by element: the two ribbon families, "
                    "flat junctions, bridges"),
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
    """Ribbon-mode drafts: exclusive over/under, balanced strands."""
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
    return ok


def _check_structure():
    """The A-I structure: adjacent flat crossings alternate layers
    (every bridge joins an upper to a lower junction) and at every
    centre the two diagonal families take opposite layers -- one
    ribbon over, one under."""
    ok = True
    for a in range(4):
        for b in range(4):
            ok &= _layer(a, b) == (1 if (a + b) % 2 == 0 else -1)
            if a < 3:
                ok &= _layer(a, b) == -_layer(a + 1, b)
            if b < 3:
                ok &= _layer(a, b) == -_layer(a, b + 1)
            if a < 3 and b < 3:
                # NE family enters centre (a+.5, b+.5) from (a, b);
                # NW family from (a+1, b): opposite layers
                ok &= _layer(a, b) == -_layer(a + 1, b)
    return ok


def _run_selftest():
    W, H = 3, 2
    par = _check_parity()
    struct = _check_structure()
    print("ribbon parity ok:", par, "| A-I structure ok:", struct)

    (v, f, m), st = build_membrane(W, H, res=16, thickness=0.0)
    nonempty = len(v) > 0 and len(f) > 0
    finite = bool(np.all(np.isfinite(np.asarray(v))))
    conn = st['components'] == 1
    manifold = st['nonmanifold'] == 0
    holes_ok = st['holes'] == W * H
    h = st['h']
    flat_ok = st['flat_max'] <= 0.3 * h
    span_ok = st['span_min'] >= 1.5 * h
    sheets_ok = st['stray_mid'] == 0        # nothing merged at E
    print("membrane: V=%d F=%d comps=%d nonman=%d rims=%d "
          "holes=%d (want %d)" % (len(v), len(f), st['components'],
                                  st['nonmanifold'], st['rims'],
                                  st['holes'], W * H))
    print("flat max|z|=%.3fh (<=0.3h) | E span=%.3fh (>=1.5h) | "
          "stray mid-level verts at E: %d (want 0)"
          % (st['flat_max'] / h, st['span_min'] / h,
             st['stray_mid']))

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

    ok = (par and struct and nonempty and finite and conn
          and manifold and holes_ok and flat_ok and span_ok
          and sheets_ok and min_ok and rib_ok and fit_ok)
    print("RESULT:", "OK" if ok else "BAD")
    return ok


if __name__ == "__main__":
    _run_selftest()
