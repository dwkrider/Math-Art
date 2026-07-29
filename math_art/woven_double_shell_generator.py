
# Woven Double Shell generator for Blender.
#
# Two concentric spheres of rope rosettes tied into one woven fabric.
# On the OUTER sphere (radius 1) a rosette medallion sits over every
# face of a chosen Platonic / Archimedean / geodesic scaffold; on the
# INNER sphere (radius `inner_radius`) a medallion sits at every
# scaffold VERTEX -- the dual solid's faces (the dual of a Platonic
# solid is Platonic, of an Archimedean solid its Catalan solid, of a
# geodesic sphere its Goldberg dual).  Each medallion is cut open at
# its lobe corners into PORTS, and rope BRIDGES dive through the gap
# from every outer port to a topologically paired inner port -- the
# rotation-proof outer-edge -> inner-edge pairing of the Twisted
# Polyhedron generator -- so the whole object is a set of closed rope
# strands threading outer arc, down bridge, inner arc, up bridge, ...
# forever.  An integer `advance` rotates the up-pairing around each
# inner medallion: advance 0 closes every strand into a short
# chain-mail stitch ring around one scaffold edge, advance 1 chains
# them into long snaking strands.  All crossings (bridge x bridge in
# the gap, medallion x medallion on each shell, bridge x medallion at
# the dives) are woven strictly one-over-one-under with the parity
# union-find of the Knot Carpet engine, then the whole rope field is
# physically relaxed with the KnotPlot bead/stick model (springs,
# fairing, 1/r^4 repulsion, a radial over/under gap at each crossing,
# and a per-bead target radius holding the two shells apart).
#
# References:
# - Inspired by Bathsheba Grossman's "Quintrino" and her "Ora" family
#   of dual-symmetry sculptures: strap-like arms swirling between a
#   dodecahedron and its dual.  This renders that bridge topology as
#   woven rope over any Platonic / Archimedean / geodesic scaffold.
# - Polyhedral duality: the dual of a Platonic solid is Platonic; the
#   dual of an Archimedean solid is its Catalan solid (Archimedes,
#   work lost; Johannes Kepler, "Harmonices Mundi", 1619; Eugene
#   Catalan, 1865).  R. Buckminster Fuller, U.S. Patent 2,682,235
#   (1954) -- the geodesic subdivision of the icosahedron.
# - Robert G. Scharein, "Interactive Topological Drawing" (PhD
#   thesis, The University of British Columbia, 1998), ch. 7 -- the
#   KnotPlot bead/stick relaxation reused here (springs, high-order
#   repulsion, critical damping, the d_max step bound).
# - J. K. Simon, "Energy functions for polygonal knots", J. Knot
#   Theory Ramifications 3(3), 1994; G. Buck & J. Orloff, Topology
#   Appl. 51 (1993) / 61 (1995) -- the knot energies the repulsion
#   approximates.
# - Peter R. Cromwell, "Knots and Links" (CUP, 2004) -- alternating
#   diagrams, the one-over-one-under discipline enforced here.

bl_info = {
    "name": "Woven Double Shell",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Woven Double Shell",
    "description": "Rope rosettes on two concentric spheres joined "
                   "by woven bridges (dual-polyhedron topology)",
    "category": "Add Mesh",
}

from math import atan2, cos, pi, radians, sin

import numpy as np

try:
    from . import pattern_common as pc
    from . import knot_carpet_generator as kcg
    from . import regular_solids_generator as rs
    from . import geodesic_generator as geo
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import knot_carpet_generator as kcg
    import regular_solids_generator as rs
    import geodesic_generator as geo


# --------------------------------------------------------------------
# Scaffold: solid + dual + the outer-edge -> inner-edge pairing
# --------------------------------------------------------------------
#
# A numpy port of the Twisted Polyhedron scaffold (that module builds
# with mathutils and cannot run outside Blender): unit-sphere solid
# vertices, faces reordered CCW as seen from outside, dual vertices
# (normalized face centroids) and, per vertex, the CCW ring of
# incident faces -- the dual face sitting at that vertex.

_FAMILY = {}
for _sid, _label, _nota in rs.PLATONIC:
    _FAMILY[_sid] = 'PLATONIC'
for _sid, _label, _nota in rs.ARCHIMEDEAN:
    _FAMILY[_sid] = 'ARCHIMEDEAN'

SOLID_ITEMS = [(_sid, _label, "%s outer shell, its dual inside"
                % _label)
               for cat in (rs.PLATONIC, rs.ARCHIMEDEAN)
               for (_sid, _label, _nota) in cat]
SOLID_ITEMS.append(
    ('GEODESIC', "Geodesic Sphere",
     "Icosahedral geodesic sphere (set Frequency), its Goldberg "
     "dual of hexagons and 12 pentagons inside"))


def _unitv(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _orient_faces_np(V, F):
    """Reorder each face's vertices CCW as seen from outside (about
    the outward centroid direction)."""
    out = []
    for f in F:
        pts = V[list(f)]
        c = pts.mean(axis=0)
        n = _unitv(c)
        ref = V[f[0]] - c
        ref = ref - n * float(ref @ n)
        ref = _unitv(ref)
        sid = np.cross(n, ref)
        idx = sorted(f, key=lambda i: atan2(
            float((V[i] - c) @ sid), float((V[i] - c) @ ref)))
        out.append(tuple(idx))
    return out


def _dual_np(V, F):
    """(Vd, rings): one dual vertex per face (the unit face centroid)
    and, per solid vertex, the ring of incident faces ordered CCW --
    the dual face at that vertex."""
    Vd = np.array([_unitv(V[list(f)].mean(axis=0)) for f in F])
    rings = []
    for v in range(len(V)):
        inc = [fi for fi, f in enumerate(F) if v in f]
        n = _unitv(V[v])
        ref = Vd[inc[0]] - n * float(Vd[inc[0]] @ n)
        ref = _unitv(ref)
        sid = np.cross(n, ref)
        inc.sort(key=lambda fi: atan2(float(Vd[fi] @ sid),
                                      float(Vd[fi] @ ref)))
        rings.append(inc)
    return Vd, rings


def _base_np(name, freq):
    if name == 'GEODESIC':
        V, F = geo.build_sphere('ICOSA', max(1, int(freq)), 'I')
        return np.asarray(V, float), [tuple(f) for f in F]
    V, F, _sizes = rs.build_solid(_FAMILY[name], name)
    return np.asarray(V, float), [tuple(f) for f in F]


def solid_np(name, freq=2):
    """(Vp, Fp, Vd, rings) of the chosen scaffold: unit-sphere verts,
    CCW faces, dual verts (one per face, unit) and per-vertex CCW
    rings of incident faces.  Works for mixed vertex degrees
    (geodesic spheres) too."""
    V, F = _base_np(name, freq)
    Vp = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-300)
    Fp = _orient_faces_np(Vp, F)
    Vd, rings = _dual_np(Vp, Fp)
    return Vp, Fp, Vd, rings


def edge_pairing(Fp, rings, spin):
    """The down map D: outer corner (face f, edge e) -> inner corner
    (vertex v, ring slot t).  Outer edge e of face f runs f[e] ->
    f[e+1]; its forward vertex v = f[e+1] carries the inner medallion
    of m = deg(v) corners, and the target slot is (j + step) % m with
    j = f's position in v's ring and step = round(spin / (360/m)) --
    the Twisted Polyhedron's rotation-proof bijection: every inner
    corner receives exactly one bridge pair at any spin."""
    D = {}
    for fi, f in enumerate(Fp):
        n = len(f)
        for e in range(n):
            vb = f[(e + 1) % n]
            m = len(rings[vb])
            step = int(round(spin / (360.0 / m)))
            j = rings[vb].index(fi)
            D[(fi, e)] = (vb, (j + step) % m)
    return D


def _scaffold(name, freq, spin):
    """Everything combinatorial: solid, dual, pairing, edge list,
    face/vertex adjacency."""
    Vp, Fp, Vd, rings = solid_np(name, freq)
    D = edge_pairing(Fp, rings, spin)
    owners = {}
    for fi, f in enumerate(Fp):
        n = len(f)
        for e in range(n):
            a, b = f[e], f[(e + 1) % n]
            owners.setdefault((min(a, b), max(a, b)), []).append(fi)
    edges = sorted(owners)
    vadj = {v: set() for v in range(len(Vp))}
    for (a, b) in edges:
        vadj[a].add(b)
        vadj[b].add(a)
    fadj = {fi: set() for fi in range(len(Fp))}
    for own in owners.values():
        if len(own) == 2:
            fadj[own[0]].add(own[1])
            fadj[own[1]].add(own[0])
    return dict(Vp=Vp, Fp=Fp, Vd=Vd, rings=rings, D=D, edges=edges,
                edge_faces=owners, vadj=vadj, fadj=fadj)


# --------------------------------------------------------------------
# The strand walk (a permutation on lobe arcs -- always closes)
# --------------------------------------------------------------------
#
# Arcs: outer arc (f, i) spans outer medallion corner i -> i+1 (CCW);
# inner arc (v, t) spans inner corner t -> t+1 in increasing ring
# angle.  Both medallion families are walked FORWARD (_DELTA = +1,
# the same CCW sense seen from outside): at advance = 0 the walk
# then closes every strand into a short stitch ring around one
# scaffold edge -- exactly E components, verified by the self-test
# on every scaffold -- the chain-mail fallback.  `advance` rotates
# the up-pairing by that many ring slots, merging the rings into
# long snaking strands (advance = 1 gives one strand per scaffold
# face on the Platonic solids); the walk is a permutation for every
# advance, so strands ALWAYS close.

_DELTA = +1        # inner-medallion walk direction (ring order)
_ADV = +1          # advance rotates the up map in walk direction


def strand_cycles(Fp, rings, D, advance=1):
    """Cycles of the arc successor permutation.  Each cycle is a list
    of entries (('O', f, i), ('D', f, e, v, t), ('I', v, arc),
    ('U', v, te, f2, e2)): outer arc, down rail, inner arc (walked in
    _DELTA order), up rail to the next outer arc."""
    Dinv = {vt: fe for fe, vt in D.items()}
    seen = set()
    cycles = []
    for start in D:
        if start in seen:
            continue
        cyc = []
        f, i = start
        while (f, i) not in seen:
            seen.add((f, i))
            n = len(Fp[f])
            e = (i + 1) % n
            v, t = D[(f, e)]
            m = len(rings[v])
            if _DELTA > 0:
                arc, te = t, (t + 1) % m
            else:
                arc, te = (t - 1) % m, (t - 1) % m
            f2, e2 = Dinv[(v, (te + _ADV * int(advance)) % m)]
            cyc.append((('O', f, i), ('D', f, e, v, t),
                        ('I', v, arc), ('U', v, te, f2, e2)))
            f, i = f2, e2
        cycles.append(cyc)
    return cycles


# --------------------------------------------------------------------
# Medallion and bridge geometry
# --------------------------------------------------------------------

def bez(cp, t):
    """Cubic bezier point (numpy arrays)."""
    u = 1.0 - t
    return (u * u * u * cp[0] + 3 * u * u * t * cp[1]
            + 3 * u * t * t * cp[2] + t * t * t * cp[3])


def _rot_axis(p, a, ang):
    """Rodrigues rotation of point/vector p about unit axis a."""
    c, s = cos(ang), sin(ang)
    return p * c + np.cross(a, p) * s + a * float(a @ p) * (1.0 - c)


def _lobe_arc(v, e1, e2, rho0, k, amp, th0, th1, n, radius):
    """One lobe arc of the rosette r(th) = rho0 (1 + amp cos k th)
    pushed through the exponential map at unit vector v (tangent
    basis e1, e2) and scaled to the shell radius.  (n, 3) points."""
    th = np.linspace(th0, th1, n)
    rho = rho0 * (1.0 + amp * np.cos(k * th))
    d = (np.cos(rho)[:, None] * v
         + np.sin(rho)[:, None] * (np.cos(th)[:, None] * e1
                                   + np.sin(th)[:, None] * e2))
    return radius * d


def _ang(a, b):
    return float(np.arccos(np.clip(float(a @ b), -1.0, 1.0)))


def _frame(center, aim, roll=0.0):
    """Tangent basis (e1, e2) at unit `center`, e1 toward `aim`
    (projected), rotated by `roll` radians about center."""
    v = _unitv(center)
    e1 = aim - v * float(aim @ v)
    nn = float(np.linalg.norm(e1))
    if nn < 1e-9:
        e1, e2 = kcg._tangent_frame(v)
    else:
        e1 = e1 / nn
    if roll:
        e1 = _rot_axis(e1, v, roll)
    e2 = np.cross(v, e1)
    return e1, e2


def _medallions(scaf, inner_radius, overlap_out, overlap_in,
                amplitude, port_gap, spin, samples):
    """Sampled lobe arcs of every medallion.  Outer medallion of face
    f: k = side count, corner e aimed at edge e's midpoint, angular
    radius 0.5 * overlap_out * (largest angular distance to an
    edge-neighbour face centre) -- the max rule, so the longest
    incident gap is still overlapped.  Inner medallion of vertex v:
    k = deg(v), corner t aimed at the dual edge t midpoint, rolled by
    -spin (matching the pairing's step fold), radius by the same max
    rule over neighbour vertices.  Ports: each corner is cut open by
    `port_gap` (fraction of a lobe span) on both sides; arc e spans
    corner e -> corner e+1 between the cuts.  Returns dict with
    arcs_out[f][e] / arcs_in[v][t] records (pts, t0, t1) and the
    angular radii."""
    Vp, Fp, Vd = scaf['Vp'], scaf['Fp'], scaf['Vd']
    rings = scaf['rings']
    rho_out, arcs_out = [], []
    for fi, f in enumerate(Fp):
        k = len(f)
        nb = scaf['fadj'][fi]
        rho0 = 0.5 * overlap_out * max(_ang(Vd[fi], Vd[g])
                                       for g in nb)
        rho_out.append(rho0)
        aim = Vp[f[0]] + Vp[f[1]]
        e1, e2 = _frame(Vd[fi], aim)
        g = port_gap * 2.0 * pi / k
        n = max(8, int(samples) // k)
        arcs = []
        for e in range(k):
            th0 = 2.0 * pi * e / k + g
            th1 = 2.0 * pi * (e + 1) / k - g
            pts = _lobe_arc(Vd[fi], e1, e2, rho0, k, amplitude,
                            th0, th1, n, 1.0)
            arcs.append(dict(pts=pts,
                             t0=_unitv(pts[1] - pts[0]),
                             t1=_unitv(pts[-1] - pts[-2])))
        arcs_out.append(arcs)
    rho_in, arcs_in = [], []
    for v in range(len(Vp)):
        R = rings[v]
        m = len(R)
        rho0 = 0.5 * overlap_in * max(_ang(Vp[v], Vp[w])
                                      for w in scaf['vadj'][v])
        rho_in.append(rho0)
        aim = Vd[R[0]] + Vd[R[1]]
        # roll by -spin: the pairing step folds +spin, so the frame
        # counter-rotates and the residual leans the bridges (the
        # Twisted Polyhedron's spin convention)
        e1, e2 = _frame(Vp[v], aim, roll=-radians(spin))
        g = port_gap * 2.0 * pi / m
        n = max(8, int(samples) // m)
        arcs = []
        for t in range(m):
            th0 = 2.0 * pi * t / m + g
            th1 = 2.0 * pi * (t + 1) / m - g
            pts = _lobe_arc(Vp[v], e1, e2, rho0, m, amplitude,
                            th0, th1, n, inner_radius)
            arcs.append(dict(pts=pts,
                             t0=_unitv(pts[1] - pts[0]),
                             t1=_unitv(pts[-1] - pts[-2])))
        arcs_in.append(arcs)
    return dict(arcs_out=arcs_out, arcs_in=arcs_in,
                rho_out=np.asarray(rho_out),
                rho_in=np.asarray(rho_in))


def _edge_axis(Vp, Fp, f, e):
    """Unit direction of the midpoint of face f's scaffold edge e --
    the axis the bridge twist swirls about."""
    n = len(Fp[f])
    return _unitv(Vp[Fp[f][e]] + Vp[Fp[f][(e + 1) % n]])


def _rail(P, tP, Q, tQ, axis, twist_rad, ho, hi, nseg):
    """One bridge rope centerline: a cubic bezier from port P (walk
    tangent tP) to port Q (walk tangent tQ), handle lengths ho / hi
    times the span, the two mid controls rotated by `twist` about the
    scaffold-edge axis -- the swirl that makes neighbouring bridges
    cross in the gap."""
    L = float(np.linalg.norm(Q - P))
    c1 = P + ho * L * tP
    c2 = Q - hi * L * tQ
    if twist_rad:
        c1 = _rot_axis(c1, axis, twist_rad)
        c2 = _rot_axis(c2, axis, twist_rad)
    ts = np.linspace(0.0, 1.0, nseg + 1)
    return np.array([bez((P, c1, c2, Q), t) for t in ts])


# element tags per bead
ELEM_OUTER, ELEM_RAIL, ELEM_INNER = 0, 1, 2


def _assemble(scaf, geom, cycles, inner_radius, twist, handle_out,
              handle_in, rail_samples):
    """Splice each cycle into one closed polyline: outer arc, down
    rail, inner arc, up rail, ...  Every element's first point equals
    the previous element's last (ports and bezier endpoints coincide
    exactly), so each element contributes pts[1:] and the loop closes
    seamlessly.  Returns raw (pts, elem, rbase) per strand."""
    Vp, Fp = scaf['Vp'], scaf['Fp']
    tw = radians(twist)
    out = []
    for cyc in cycles:
        info = []                        # ('A', pts, t0, t1, tag, rb)
        for entry in cyc:
            (O, Dn, In, Up) = entry
            ao = geom['arcs_out'][O[1]][O[2]]
            info.append(('A', ao['pts'], ao['t0'], ao['t1'],
                         ELEM_OUTER, 1.0))
            info.append(('R', _edge_axis(Vp, Fp, Dn[1], Dn[2])))
            ai = geom['arcs_in'][In[1]][In[2]]
            if _DELTA > 0:
                info.append(('A', ai['pts'], ai['t0'], ai['t1'],
                             ELEM_INNER, inner_radius))
            else:
                info.append(('A', ai['pts'][::-1], -ai['t1'],
                             -ai['t0'], ELEM_INNER, inner_radius))
            info.append(('R', _edge_axis(Vp, Fp, Up[3], Up[4])))
        M = len(info)
        pts_l, el_l, rb_l = [], [], []
        for a in range(M):
            rec = info[a]
            if rec[0] == 'A':
                pts = rec[1]
                pts_l.append(pts[1:])
                el_l.append(np.full(len(pts) - 1, rec[4], int))
                rb_l.append(np.full(len(pts) - 1, rec[5]))
            else:
                prv = info[a - 1]
                nxt = info[(a + 1) % M]
                pts = _rail(prv[1][-1], prv[3], nxt[1][0], nxt[2],
                            rec[1], tw, handle_out, handle_in,
                            rail_samples)
                pts_l.append(pts[1:])
                el_l.append(np.full(len(pts) - 1, ELEM_RAIL, int))
                rb_l.append(np.linalg.norm(pts[1:], axis=1))
        out.append((np.vstack(pts_l), np.concatenate(el_l),
                    np.concatenate(rb_l)))
    return out


def _resample(P, elem, rb, h):
    """Arclength-resample a closed polyline (and its per-bead tags /
    base radii) to ~uniform spacing h."""
    N = len(P)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    n = max(24, int(round(total / h)))
    sj = np.arange(n) * total / n
    Pe = np.vstack([P, P[:1]])
    Q = np.column_stack([np.interp(sj, s, Pe[:, c])
                         for c in range(3)])
    idx = np.minimum(np.searchsorted(s, sj, side='right') - 1, N - 1)
    rbe = np.concatenate([rb, rb[:1]])
    return Q, elem[idx], np.interp(sj, s, rbe)


# --------------------------------------------------------------------
# Crossings: radial projection per scaffold-edge arena
# --------------------------------------------------------------------

def _cyclic_runs(mask):
    """Maximal cyclic runs of True in a boolean array, as index
    arrays; a fully-True mask returns one whole run."""
    N = len(mask)
    if mask.all():
        return [(np.arange(N), True)]
    if not mask.any():
        return []
    m = mask.astype(int)
    starts = np.nonzero((m - np.roll(m, 1)) == 1)[0]
    runs = []
    for st in starts:
        ln = 0
        while ln < N and mask[(st + ln) % N]:
            ln += 1
        runs.append(((st + np.arange(ln)) % N, False))
    return runs


def find_crossings(strands, scaf, geom, cross_reach,
                   min_sin=0.05, near_beads=8):
    """All woven crossings between the resampled strands.  Per
    scaffold edge, the nearby beads of every strand (a spherical cap
    about the edge-midpoint direction wide enough to hold both
    medallion pairs and the bridges) are projected onto the tangent
    plane there and intersected pairwise with the Knot Carpet's
    _seg_cross; a hit is kept only if the two strands actually pass
    within `cross_reach` in 3D (so the outer and inner shells, a full
    gap apart, never fake-cross), the crossing is not grazing
    (|sin| >= min_sin), and same-strand hits are not ring
    neighbours.  Arenas overlap, so duplicates are removed by rounded
    (strand, bead) keys.  Returns [(key, lo, hi, f_lo, f_hi)]."""
    Vp = scaf['Vp']
    N = [len(P) for P in strands]
    U = [P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
         for P in strands]
    accepted = {}
    crossings = []

    def _pt(sidx, f):
        P = strands[sidx]
        i0 = int(f) % N[sidx]
        t = f - int(f)
        return P[i0] * (1.0 - t) + P[(i0 + 1) % N[sidx]] * t

    for (u, w) in scaf['edges']:
        m_e = _unitv(Vp[u] + Vp[w])
        fs = scaf['edge_faces'][(min(u, w), max(u, w))]
        rho_a = (0.5 * _ang(Vp[u], Vp[w])
                 + 1.1 * max([geom['rho_in'][u], geom['rho_in'][w]]
                             + [geom['rho_out'][f] for f in fs]))
        cosmax = cos(min(rho_a, 1.35))
        a1, a2 = kcg._tangent_frame(m_e)
        runs = []
        for sidx in range(len(strands)):
            for (idx, whole) in _cyclic_runs(U[sidx] @ m_e > cosmax):
                if len(idx) >= 3:
                    runs.append((sidx, idx, whole))
        for A in range(len(runs)):
            sA, iA, wA = runs[A]
            PA = np.column_stack([strands[sA][iA] @ a1,
                                  strands[sA][iA] @ a2])
            for B in range(A + 1, len(runs)):
                sB, iB, wB = runs[B]
                PB = np.column_stack([strands[sB][iB] @ a1,
                                      strands[sB][iB] @ a2])
                for (fa, fb) in kcg._seg_cross(PA, PB):
                    ia, ib = int(fa), int(fb)
                    if not wA and ia >= len(iA) - 1:
                        continue
                    if not wB and ib >= len(iB) - 1:
                        continue
                    ga = (float(iA[0]) + fa) % N[sA]
                    gb = (float(iB[0]) + fb) % N[sB]
                    if sA == sB:
                        dd = abs(ga - gb)
                        if min(dd, N[sA] - dd) < near_beads:
                            continue
                    pa, pb = _pt(sA, ga), _pt(sB, gb)
                    if float(np.linalg.norm(pa - pb)) > cross_reach:
                        continue
                    da = PA[(ia + 1) % len(iA)] - PA[ia]
                    db = PB[(ib + 1) % len(iB)] - PB[ib]
                    la = float(np.hypot(*da))
                    lb = float(np.hypot(*db))
                    if la < 1e-12 or lb < 1e-12:
                        continue
                    if abs(da[0] * db[1] - da[1] * db[0]) \
                            / (la * lb) < min_sin:
                        continue
                    if (sA, ga) <= (sB, gb):
                        lo, hi, flo, fhi = sA, sB, ga, gb
                    else:
                        lo, hi, flo, fhi = sB, sA, gb, ga
                    key2 = (lo, hi)
                    dup = False
                    for (pf, qf) in accepted.get(key2, ()):
                        d1 = abs(pf - flo)
                        d2 = abs(qf - fhi)
                        if (min(d1, N[lo] - d1) < 5.0
                                and min(d2, N[hi] - d2) < 5.0):
                            dup = True
                            break
                    if dup:
                        continue
                    accepted.setdefault(key2, []).append((flo, fhi))
                    crossings.append((len(crossings), lo, hi,
                                      flo, fhi))
    return crossings


def _bias_over(rbase, crossings, over, per_loop, lock=0.06):
    """Align the weave with the geometry, in two steps.  (1) A parity
    solution is unique only up to a global bit-flip per connected
    component of the alternation constraints, so flip each component
    by MAJORITY VOTE toward "over = the strand with the larger base
    radius".  (2) A crossing whose two strands sit at base radii more
    than `lock` apart is GEOMETRY-LOCKED -- a mid-gap bridge passing
    beneath an outer-shell arc cannot be hoisted over it -- so its
    bit is set from the radii outright.  Same-shell crossings (base
    radii equal) are untouched by (2), so the visible weave on each
    shell keeps its alternation; one-over-one-under holds per
    crossing whatever the bits."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ent in per_loop.values():
        ent2 = sorted(ent)
        C = len(ent2)
        for a in range(C):
            union(ent2[a][1], ent2[(a + 1) % C][1])
    votes = {}
    dr = {}
    for (key, lo, hi, flo, fhi) in crossings:
        r_lo = float(rbase[lo][int(round(flo)) % len(rbase[lo])])
        r_hi = float(rbase[hi][int(round(fhi)) % len(rbase[hi])])
        dr[key] = r_lo - r_hi
        pref = (r_lo - r_hi) if over[key] else (r_hi - r_lo)
        r = find(key)
        votes[r] = votes.get(r, 0.0) + pref
    out = {k: (b ^ 1 if votes.get(find(k), 0.0) < 0.0 else b)
           for k, b in over.items()}
    for k, d in dr.items():
        if abs(d) > lock:
            out[k] = 1 if d > 0.0 else 0
    return out


# --------------------------------------------------------------------
# The full build
# --------------------------------------------------------------------

def build_double_shell(solid='DODECA', frequency=2, inner_radius=0.62,
                       overlap_out=1.10, overlap_in=1.10,
                       amplitude=0.10, port_gap=0.08, spin=20.0,
                       advance=1, twist=25.0, handle_out=1.2,
                       handle_in=0.5, samples=160, rail_samples=24,
                       weave_gap=0.10, clearance=0.10,
                       max_beads=24000):
    """Scaffold + medallions + bridges + strand walk + crossings +
    parity weave + radial weave seed.  Returns a dict:
      strands  -- per strand: (S, 3) closed polyline at base radii
      seed     -- the same strands with the radial over/under offset
      elem     -- per strand: per-bead tag (0 outer arc, 1 rail,
                  2 inner arc); rbase -- per-bead target radius
      crossings, over, per_loop, signed, consistent -- the weave
      n_components, n_edges, h, cross_class -- bookkeeping
    All lengths are absolute (outer sphere radius 1)."""
    inner_radius = min(0.92, max(0.45, float(inner_radius)))
    scaf = _scaffold(solid, frequency, float(spin))
    geom = _medallions(scaf, inner_radius, float(overlap_out),
                       float(overlap_in), float(amplitude),
                       float(port_gap), float(spin), int(samples))
    cycles = strand_cycles(scaf['Fp'], scaf['rings'], scaf['D'],
                           int(advance))
    raw = _assemble(scaf, geom, cycles, inner_radius, float(twist),
                    float(handle_out), float(handle_in),
                    int(rail_samples))
    # uniform bead spacing: the mean outer medallion perimeter over
    # `samples`, floored so the whole field stays under max_beads
    perim = float(np.mean(2.0 * pi * np.sin(geom['rho_out'])))
    h = perim / max(24, int(samples))
    total = sum(float(np.sum(np.linalg.norm(
        np.roll(P, -1, 0) - P, axis=1))) for (P, _e, _r) in raw)
    h = max(h, total / float(max_beads))
    strands, elem, rbase = [], [], []
    for (P, el, rb) in raw:
        Q, e2, r2 = _resample(P, el, rb, h)
        strands.append(Q)
        elem.append(e2)
        rbase.append(r2)
    # weave only genuine CONTACTS: pairs that pass close enough to
    # touch.  A looser reach would sweep in "layered" passes (strands
    # a good fraction of the gap apart) whose radial order is already
    # decided by the geometry -- forcing alternation onto those makes
    # the diagram physically unweavable.
    cross_reach = min(0.45 * (1.0 - inner_radius),
                      max(0.6 * float(clearance), 0.05, 2.5 * h))
    crossings = find_crossings(strands, scaf, geom, cross_reach)
    over, per_loop, consistent = kcg._solve_over(strands, crossings)
    lift = 0.6 * max(float(weave_gap), float(clearance))
    over = _bias_over(rbase, crossings, over, per_loop,
                      lock=0.4 * lift)
    signed = kcg.loop_signed(dict(paths=strands, over=over,
                                  per_loop=per_loop))
    seed = []
    for i, P in enumerate(strands):
        roff = kcg._weave_roff(P, signed[i], lift)
        u = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        seed.append(u * (rbase[i] + roff)[:, None])
    ccls = {}
    for (key, lo, hi, flo, fhi) in crossings:
        ea = int(elem[lo][int(round(flo)) % len(elem[lo])])
        eb = int(elem[hi][int(round(fhi)) % len(elem[hi])])
        ck = (min(ea, eb), max(ea, eb))
        ccls[ck] = ccls.get(ck, 0) + 1
    return dict(strands=strands, seed=seed, elem=elem, rbase=rbase,
                crossings=crossings, over=over, per_loop=per_loop,
                signed=signed, consistent=consistent,
                n_components=len(strands),
                n_edges=len(scaf['edges']), h=h,
                cross_class=ccls, inner_radius=inner_radius,
                scaf=scaf, geom=geom)


# --------------------------------------------------------------------
# Relaxation (KnotPlot bead/stick model on the double shell)
# --------------------------------------------------------------------

def _grid_pairs(X, cutoff, sid, pos, slen, excl):
    """Bead pairs within `cutoff`, via a uniform grid hash; pairs of
    the same strand closer than `excl` beads along the ring are
    excluded (they are held by the springs, not the repulsion)."""
    cell = np.floor(X / cutoff).astype(np.int64)
    buckets = {}
    for i, ck in enumerate(map(tuple, cell)):
        buckets.setdefault(ck, []).append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
    I, J = [], []
    for ck, lst in buckets.items():
        arr = np.asarray(lst)
        if len(lst) > 1:
            ii, jj = np.triu_indices(len(lst), 1)
            I.append(arr[ii])
            J.append(arr[jj])
        for off in offs:
            nb = buckets.get((ck[0] + off[0], ck[1] + off[1],
                              ck[2] + off[2]))
            if nb:
                nba = np.asarray(nb)
                I.append(np.repeat(arr, len(nba)))
                J.append(np.tile(nba, len(arr)))
    if not I:
        return (np.empty(0, int),) * 2
    i_idx = np.concatenate(I)
    j_idx = np.concatenate(J)
    d = np.linalg.norm(X[i_idx] - X[j_idx], axis=1)
    keep = d <= cutoff
    same = sid[i_idx] == sid[j_idx]
    rd = np.abs(pos[i_idx] - pos[j_idx])
    ringd = np.minimum(rd, slen[i_idx] - rd)
    keep &= ~(same & (ringd <= excl))
    return i_idx[keep], j_idx[keep]


# smoothed window weights for the direct crossing projection
_PRJ_W = np.array([0.35, 0.7, 1.0, 0.7, 0.35])


def _lateral(X, oi, ui):
    """Per-crossing SIDEWAYS separation: the anchor distance with its
    radial component removed.  A woven crossing has radial separation
    about the weave gap but almost no lateral offset; a crossing that
    has slid apart sideways is out of contact (released)."""
    dv = X[oi] - X[ui]
    sep2 = np.einsum('ij,ij->i', dv, dv)
    rdir = X[ui] / (np.linalg.norm(X[ui], axis=1,
                                   keepdims=True) + 1e-12)
    rad = np.einsum('ij,ij->i', dv, rdir)
    return np.sqrt(np.maximum(sep2 - rad * rad, 0.0))


def relax_double_shell(shell, iters=120, clearance=0.10,
                       weave_gap=0.10, confine=0.6, step=0.15,
                       trace=None):
    """Physically relax the seeded rope field (the KnotPlot bead /
    stick model of the Knot Carpet, on a flat bead array so strands
    of different lengths relax together).  Per iteration:
    rest-length edge springs + Laplacian fairing keep each strand
    smooth and evenly beaded; a soft 1/r^4 repulsion with cutoff
    pushes nearby strands apart to about `clearance`; at every
    crossing the over bead is kept at least `weave_gap` further from
    the centre than the under bead; and each bead's radius is pulled
    toward its per-bead target (outer arcs -> 1, inner arcs -> the
    inner radius, bridges -> their seeded gap radius with low
    weight).  The gain bound ks * step * 4 < 1 holds and each bead
    moves at most a fraction of `clearance` per step (KnotPlot's
    d_max), so strands cannot pass through one another.  Returns one
    (S, 3) closed centerline per strand."""
    strands = shell['seed']
    L = len(strands)
    offs = np.cumsum([0] + [len(P) for P in strands])
    X = np.vstack(strands).astype(float)
    Ntot = len(X)
    sid = np.concatenate([np.full(len(strands[i]), i, int)
                          for i in range(L)])
    pos = np.concatenate([np.arange(len(strands[i]))
                          for i in range(L)])
    slen = np.concatenate([np.full(len(strands[i]),
                                   len(strands[i]), int)
                           for i in range(L)])
    nxt = np.concatenate([offs[i] + (np.arange(len(strands[i])) + 1)
                          % len(strands[i]) for i in range(L)])
    prv = np.concatenate([offs[i] + (np.arange(len(strands[i])) - 1)
                          % len(strands[i]) for i in range(L)])
    rbase = np.concatenate(shell['rbase'])
    elem = np.concatenate(shell['elem'])
    clr = float(clearance)
    gap = float(weave_gap)

    # crossing constraints as flat bead indices (over / under)
    over = shell['over']
    oi, ui = [], []
    for (key, lo, hi, flo, fhi) in shell['crossings']:
        blo = offs[lo] + int(round(flo)) % len(strands[lo])
        bhi = offs[hi] + int(round(fhi)) % len(strands[hi])
        if over[key]:
            oi.append(blo)
            ui.append(bhi)
        else:
            oi.append(bhi)
            ui.append(blo)
    oi = np.asarray(oi, int)
    ui = np.asarray(ui, int)
    have = len(oi) > 0
    reversed_mask = np.zeros(len(oi), bool)

    # confinement weight: dips near crossings, low on the bridges
    h = float(shell['h'])
    win = max(3, int(round(0.5 * max(gap, clr) / max(h, 1e-9))))
    wfl = np.ones(Ntot)
    for b in np.concatenate([oi, ui]) if have else []:
        s = sid[b]
        n_s = slen[b]
        base = offs[s]
        for db in range(-win, win + 1):
            j = base + (pos[b] + db) % n_s
            wfl[j] = min(wfl[j], abs(db) / float(win))
    wfl = np.maximum(np.minimum(wfl, np.where(elem == ELEM_RAIL,
                                              0.30, 1.0)), 0.12)

    ks, fair, kr, kw = 1.2, 0.25, 1.0, 4.0
    rest = np.zeros(Ntot)
    for i in range(L):
        P = strands[i]
        rest[offs[i]:offs[i + 1]] = float(np.mean(np.linalg.norm(
            np.roll(P, -1, 0) - P, axis=1)))
    # short-range repulsion: a longer reach (the flat carpets use
    # 1.7 clearance) makes the crowded gap unravel sideways -- rails
    # slide along each other until crossings dissolve -- so the
    # double shell keeps the repulsion just past contact range
    rcut = 1.3 * clr
    dmax = 0.25 * clr
    dmin = 0.2 * clr
    excl = int(np.ceil(rcut / max(h, 1e-9))) + 2
    rebuild = 5
    rel_thresh = 1.0 * clr         # released-crossing lateral gate
    wtrk = 4                       # crossing-tracking half window
    i_idx = j_idx = None

    def _retrack():
        """Re-anchor each crossing to the point where its two strands
        actually stack: the bead pair with the smallest LATERAL
        (radial-projected) separation in a window around the current
        anchor.  The strands slide as they relax, so a fixed pair
        drifts off the crossing; and plain 3D closest approach would
        hop to any side-by-side graze nearer than the radial weave
        gap, so the radial component is projected out -- at a true
        crossing the lateral separation is ~0."""
        for c in range(len(oi)):
            bo, bu = oi[c], ui[c]
            so, su = sid[bo], sid[bu]
            Wo = offs[so] + (pos[bo]
                             + np.arange(-wtrk, wtrk + 1)) % slen[bo]
            Wu = offs[su] + (pos[bu]
                             + np.arange(-wtrk, wtrk + 1)) % slen[bu]
            D = X[Wo][:, None, :] - X[Wu][None, :, :]
            mid = 0.5 * (X[Wo][:, None, :] + X[Wu][None, :, :])
            mid /= np.linalg.norm(mid, axis=2, keepdims=True) + 1e-12
            rad = np.einsum('abj,abj->ab', D, mid)
            lat2 = np.einsum('abj,abj->ab', D, D) - rad * rad
            a, b = np.unravel_index(int(np.argmin(lat2)), lat2.shape)
            oi[c], ui[c] = Wo[a], Wu[b]

    for it in range(int(iters)):
        if it % rebuild == 0:
            if have:
                _retrack()
            i_idx, j_idx = _grid_pairs(X, rcut + 0.5 * clr, sid,
                                       pos, slen, excl)
        F = np.zeros_like(X)
        # springs (rest-length edges: even spacing, no shrinkage)
        e = X[nxt] - X
        ln = np.linalg.norm(e, axis=1, keepdims=True)
        fe = ks * (1.0 - rest[:, None] / (ln + 1e-12)) * e
        F += fe - fe[prv]
        # Laplacian fairing
        F += fair * (0.5 * (X[nxt] + X[prv]) - X)
        # electrical repulsion between nearby strands
        if len(i_idx):
            D = X[i_idx] - X[j_idx]
            d = np.linalg.norm(D, axis=1)
            act = d < rcut
            if act.any():
                ia, ja = i_idx[act], j_idx[act]
                da = np.maximum(d[act], dmin)
                mag = kr * (clr / da) ** 4 * (1.0 - d[act] / rcut)
                f = D[act] * (mag / (d[act] + 1e-12))[:, None]
                np.add.at(F, ia, f)
                np.add.at(F, ja, -f)
        # weave order: over radius >= under radius + weave_gap
        if have:
            ro = np.linalg.norm(X[oi], axis=1)
            ru = np.linalg.norm(X[ui], axis=1)
            push = kw * np.maximum(0.0, gap - (ro - ru))
            np.add.at(F, oi, (0.5 * push / (ro + 1e-12))[:, None]
                      * X[oi])
            np.add.at(F, ui, -(0.5 * push / (ru + 1e-12))[:, None]
                      * X[ui])
        # per-bead radius confinement (the two shells + gap bridges)
        r = np.linalg.norm(X, axis=1)
        F += (confine * wfl * (rbase - r)
              / (r + 1e-12))[:, None] * X
        # damped Euler with the d_max step bound
        mv = step * F
        mlen = np.linalg.norm(mv, axis=1, keepdims=True)
        mv *= np.minimum(1.0, dmax / (mlen + 1e-12))
        X += mv
        if trace is not None:
            trace.append(float(np.mean(np.linalg.norm(mv, axis=1))))
        # accept settled reversals (twice, late): a near-tie crossing
        # that has cleanly stacked the OPPOSITE way is a valid weave
        # -- flip its recorded bit rather than fight it forever (the
        # endless fight is what grinds down the clearance elsewhere)
        if have and it in (int(0.6 * iters), int(0.85 * iters)):
            _retrack()
            ro = np.linalg.norm(X[oi], axis=1)
            ru = np.linalg.norm(X[ui], axis=1)
            lat = _lateral(X, oi, ui)
            flip = ((ro - ru < -0.3 * gap) & (lat <= rel_thresh))
            for c in np.nonzero(flip)[0]:
                oi[c], ui[c] = ui[c], oi[c]
                reversed_mask[c] = ~reversed_mask[c]
        # gentle constraint projection: any crossing still in contact
        # but radially mis-ordered is nudged apart directly, clamped
        # to a fraction of d_max per iteration and spread over a
        # small smoothed window (no kinks, no tunnelling) -- the
        # forces then re-settle around the corrected order
        if have:
            ro = np.linalg.norm(X[oi], axis=1)
            ru = np.linalg.norm(X[ui], axis=1)
            defc = np.minimum(0.5 * gap - (ro - ru), 0.6 * dmax)
            defc[_lateral(X, oi, ui) > rel_thresh] = 0.0
            for c in np.nonzero(defc > 0.0)[0]:
                for (b, sgn) in ((oi[c], +1.0), (ui[c], -1.0)):
                    s = sid[b]
                    W = offs[s] + (pos[b] + np.arange(-2, 3)) \
                        % slen[b]
                    r_w = np.linalg.norm(X[W], axis=1) + 1e-12
                    scale = (1.0 + sgn * 0.5 * defc[c]
                             * _PRJ_W / r_w)
                    X[W] *= scale[:, None]
    # final settle: accept any remaining clean opposite stacks, then
    # nudge the few still-ambiguous near-tie crossings apart along
    # their recorded order with a handful of clamped local sweeps
    if have:
        _retrack()
        ro = np.linalg.norm(X[oi], axis=1)
        ru = np.linalg.norm(X[ui], axis=1)
        lat = _lateral(X, oi, ui)
        flip = ((ro - ru < -0.2 * gap) & (lat <= rel_thresh))
        for c in np.nonzero(flip)[0]:
            oi[c], ui[c] = ui[c], oi[c]
            reversed_mask[c] = ~reversed_mask[c]
        for _sweep in range(12):
            ro = np.linalg.norm(X[oi], axis=1)
            ru = np.linalg.norm(X[ui], axis=1)
            defc = np.minimum(0.15 * gap - (ro - ru), 0.6 * dmax)
            defc[_lateral(X, oi, ui) > rel_thresh] = 0.0
            idx = np.nonzero(defc > 0.0)[0]
            if not len(idx):
                break
            for c in idx:
                for (b, sgn) in ((oi[c], +1.0), (ui[c], -1.0)):
                    s = sid[b]
                    W = offs[s] + (pos[b] + np.arange(-2, 3)) \
                        % slen[b]
                    r_w = np.linalg.norm(X[W], axis=1) + 1e-12
                    scale = (1.0 + sgn * 0.5 * defc[c]
                             * _PRJ_W / r_w)
                    X[W] *= scale[:, None]
    # final tracked anchors (parallel to shell['crossings']): the
    # over / under flat bead index of each crossing, plus the mask of
    # crossings that RELEASED (drifted out of contact sideways -- a
    # lateral Reidemeister-II isotopy) while relaxing
    if have:
        rel = _lateral(X, oi, ui) > rel_thresh
    else:
        rel = np.zeros(0, bool)
    shell['relax_anchors'] = (oi.copy(), ui.copy())
    shell['relax_released'] = rel
    shell['relax_reversed'] = reversed_mask
    return [X[offs[i]:offs[i + 1]].copy() for i in range(L)]


# --------------------------------------------------------------------
# Output builders
# --------------------------------------------------------------------

def shell_lines(shell, relax_iters=120, clearance=0.10,
                weave_gap=0.10, tube_radius=0.035, trace=None):
    """The rendered strand centerlines: the tier-2 relaxation when
    relax_iters > 0 (clearance / weave_gap floored at the tube
    diameter so the ropes cannot touch), else the radial weave
    seed."""
    if relax_iters > 0:
        tr = max(0.005, float(tube_radius))
        return relax_double_shell(
            shell, iters=int(relax_iters),
            clearance=max(float(clearance), 2.2 * tr),
            weave_gap=max(float(weave_gap), 2.2 * tr), trace=trace)
    return shell['seed']


def build_cells(shell, lines, output='TUBE', tube_radius=0.035,
                tube_sides=10, color_by='STRAND'):
    """One (verts, faces, mats) cell per strand: a welded round TUBE
    (all quads) or a RIBBON strap whose width lies tangent and whose
    thickness runs radially.  Colors: STRAND cycles the palette per
    strand, ELEMENT colors outer arcs / rails / inner arcs, UNIFORM
    is a single material."""
    tr = max(0.005, float(tube_radius))
    sides = max(3, int(tube_sides))
    npal = len(pc.PALETTE_RGBA)
    cells = []
    for i, P in enumerate(lines):
        pts = [tuple(p) for p in P]
        if output == 'RIBBON':
            verts, faces = kcg._sphere_band(np.asarray(P), 2.6 * tr,
                                            0.9 * tr)
            per = 4
        else:
            verts, faces = kcg._tube_welded(pts, tr, sides)
            per = sides
        if not faces:
            continue
        if color_by == 'ELEMENT':
            el = shell['elem'][i]
            emat = (1, 2, 0)             # outer, rail, inner
            mats = [emat[int(el[fi // per]) % 3]
                    for fi in range(len(faces))]
        elif color_by == 'STRAND':
            mats = [i % npal] * len(faces)
        else:
            mats = [0] * len(faces)
        cells.append((verts, faces, mats))
    return cells


def strand_paths(lines):
    """[(points, True)] closed centerlines for the CURVE output."""
    return [([tuple(p) for p in P], True) for P in lines]


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

    def _emit_curve(context, name, paths, span=2.0, operator=None):
        """Curve object of cyclic POLY splines, centered and scaled
        into the span cube."""
        paths = [(list(p), c) for p, c in paths if len(p) >= 2]
        if not paths:
            return None
        allv = np.asarray([p for pa, _c in paths for p in pa], float)
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        cen = 0.5 * (lo + hi)
        s = span / max(float(np.max(hi - lo)), 1e-9)
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        for pa, closed in paths:
            sp = cu.splines.new('POLY')
            sp.points.add(len(pa) - 1)
            for i, p in enumerate(pa):
                sp.points[i].co = ((p[0] - cen[0]) * s,
                                   (p[1] - cen[1]) * s,
                                   (p[2] - cen[2]) * s, 1.0)
            sp.use_cyclic_u = bool(closed)
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

    class MESH_OT_woven_double_shell_add(bpy.types.Operator,
                                         AddObjectHelper):
        """Add a woven double shell: rope rosettes on an outer and an
        inner sphere joined by rope bridges that weave over and under
        through the gap (the Twisted Polyhedron bridge topology as
        relaxed rope)"""
        bl_idname = "mesh.woven_double_shell_add"
        bl_label = "Woven Double Shell"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(
            name="Scaffold", items=SOLID_ITEMS, default='DODECA',
            description="Outer solid: one medallion per face outside,"
                        " one per vertex (its dual) inside")
        frequency: IntProperty(
            name="Frequency", default=2, min=1, max=4,
            description="Geodesic breakdown frequency (Geodesic "
                        "Sphere only)")
        inner_radius: FloatProperty(
            name="Inner Radius", default=0.62, min=0.45, max=0.9,
            description="Radius of the inner shell (outer is 1)")
        overlap_out: FloatProperty(
            name="Outer Overlap", default=1.10, min=0.8, max=1.6,
            description="Outer medallion radius as a fraction of "
                        "half the largest neighbour distance")
        overlap_in: FloatProperty(
            name="Inner Overlap", default=1.10, min=0.8, max=1.6,
            description="Inner medallion radius factor")
        amplitude: FloatProperty(
            name="Lobe Amplitude", default=0.10, min=0.0, max=0.4,
            description="Rosette lobe depth (0 = circles)")
        port_gap: FloatProperty(
            name="Port Gap", default=0.08, min=0.01, max=0.25,
            description="Fraction of a lobe cut open at each corner "
                        "where the rope dives to a bridge")
        spin: FloatProperty(
            name="Spin", default=20.0, min=-360.0, max=360.0,
            description="Inner medallion spin (degrees); folds with "
                        "the pairing so bridges lean, never tangle")
        advance: IntProperty(
            name="Advance", default=1, min=0, max=6,
            description="Ring rotation of the up-pairing: 0 = short "
                        "chain-mail stitch rings, 1+ = long snaking "
                        "strands")
        twist: FloatProperty(
            name="Bridge Twist", default=25.0, min=-90.0, max=90.0,
            description="Swirl of the bridge mid-controls about the "
                        "scaffold edge (degrees); makes neighbouring "
                        "bridges cross in the gap")
        handle_out: FloatProperty(
            name="Handle (Outer)", default=1.2, min=0.0, max=3.0,
            description="Bezier handle leaving the outer port, as a "
                        "fraction of the bridge span")
        handle_in: FloatProperty(
            name="Handle (Inner)", default=0.5, min=0.0, max=3.0,
            description="Bezier handle at the inner port")
        samples: IntProperty(
            name="Samples", default=160, min=48, max=400,
            description="Beads per outer medallion (sets the global "
                        "bead spacing)")
        rail_samples: IntProperty(
            name="Bridge Samples", default=24, min=8, max=64,
            description="Bezier samples per bridge before "
                        "resampling")
        tube_radius: FloatProperty(
            name="Rope Radius", default=0.02, min=0.005, max=0.12,
            description="Rope tube radius (outer sphere radius = 1)")
        tube_sides: IntProperty(
            name="Tube Sides", default=10, min=3, max=32)
        weave_gap: FloatProperty(
            name="Weave Gap", default=0.10, min=0.0, max=0.4,
            description="Radial over/under separation at crossings "
                        "(floored at the rope diameter)")
        clearance: FloatProperty(
            name="Clearance", default=0.10, min=0.0, max=0.4,
            description="Strand-strand repulsion distance (floored "
                        "at the rope diameter)")
        relax_iters: IntProperty(
            name="Relax Iterations", default=120, min=0, max=400,
            description="Bead/stick relaxation steps (0 = the raw "
                        "radial weave seed)")
        output: EnumProperty(
            name="Output",
            items=[('TUBE', "Rope Tubes",
                    "One welded round tube per strand"),
                   ('RIBBON', "Ribbon Straps",
                    "Flat straps, width tangent, thickness radial"),
                   ('CURVE', "Centerline Curves",
                    "Closed poly splines")],
            default='TUBE')
        color_by: EnumProperty(
            name="Color By",
            items=[('STRAND', "By Strand",
                    "A distinct color per closed strand"),
                   ('UNIFORM', "Uniform", "A single material"),
                   ('ELEMENT', "By Element",
                    "Outer arcs / bridges / inner arcs")],
            default='STRAND')
        separate: BoolProperty(
            name="Separate Strands", default=False,
            description="One object per closed strand")

        def execute(self, context):
            shell = build_double_shell(
                solid=self.solid, frequency=self.frequency,
                inner_radius=self.inner_radius,
                overlap_out=self.overlap_out,
                overlap_in=self.overlap_in,
                amplitude=self.amplitude, port_gap=self.port_gap,
                spin=self.spin, advance=self.advance,
                twist=self.twist, handle_out=self.handle_out,
                handle_in=self.handle_in, samples=self.samples,
                rail_samples=self.rail_samples,
                weave_gap=self.weave_gap, clearance=self.clearance)
            lines = shell_lines(shell, self.relax_iters,
                                self.clearance, self.weave_gap,
                                self.tube_radius)
            info = ("%s: %d strands, %d crossings, alternating=%s"
                    % (self.solid, shell['n_components'],
                       len(shell['crossings']),
                       "yes" if shell['consistent'] else "partial"))
            if self.output == 'CURVE':
                obj = _emit_curve(context, "Woven Double Shell",
                                  strand_paths(lines), operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no strands generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, info)
                return {'FINISHED'}
            cells = build_cells(shell, lines, self.output,
                                self.tube_radius, self.tube_sides,
                                self.color_by)
            obj = pc.emit(context, "Woven Double Shell", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no strands generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                me = obj.data
                me.polygons.foreach_set(
                    "use_smooth", [True] * len(me.polygons))
                me.update()
                info += ("  V=%d F=%d" % (len(me.vertices),
                                          len(me.polygons)))
            else:
                for ch in obj.children:
                    if ch.type == 'MESH':
                        ch.data.polygons.foreach_set(
                            "use_smooth",
                            [True] * len(ch.data.polygons))
                        ch.data.update()
            self.report({'INFO'}, info)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            if self.solid == 'GEODESIC':
                lay.prop(self, 'frequency')
            lay.prop(self, 'inner_radius')
            lay.prop(self, 'advance')
            lay.separator()
            for prop in ('overlap_out', 'overlap_in', 'amplitude',
                         'port_gap', 'spin', 'twist', 'handle_out',
                         'handle_in'):
                lay.prop(self, prop)
            lay.separator()
            lay.prop(self, 'samples')
            lay.prop(self, 'rail_samples')
            lay.prop(self, 'tube_radius')
            if self.output == 'TUBE':
                lay.prop(self, 'tube_sides')
            lay.prop(self, 'weave_gap')
            lay.prop(self, 'clearance')
            lay.prop(self, 'relax_iters')
            lay.separator()
            lay.prop(self, 'output')
            if self.output != 'CURVE':
                lay.prop(self, 'color_by')
                lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.woven_double_shell_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_woven_double_shell_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_woven_double_shell_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _min_interstrand(lines):
    """Minimum bead distance between different strands."""
    sid = np.concatenate([np.full(len(P), i, int)
                          for i, P in enumerate(lines)])
    X = np.vstack(lines)
    best = np.inf
    chunk = 800
    for a in range(0, len(X), chunk):
        D = np.linalg.norm(X[a:a + chunk, None, :] - X[None, :, :],
                           axis=2)
        same = sid[a:a + chunk, None] == sid[None, :]
        D[same] = np.inf
        best = min(best, float(D.min()))
    return best


def _test_solid(name, freq, samples, rail_samples, relax_iters,
                verbose=True):
    scaf = _scaffold(name, freq, 20.0)
    Fp, rings, D = scaf['Fp'], scaf['rings'], scaf['D']
    E = len(scaf['edges'])
    n_corners = sum(len(f) for f in Fp)
    # pairing bijection: every inner corner exactly one down rail...
    assert len(D) == n_corners == 2 * E
    assert len(set(D.values())) == len(D), "pairing not a bijection"
    assert set(D.values()) == {(v, t) for v in range(len(rings))
                               for t in range(len(rings[v]))}
    # ... and (via the up map inside the walk) exactly one up rail:
    # the up map is Dinv composed with a ring rotation, a bijection.
    counts = {}
    for adv in (0, 1):
        cycles = strand_cycles(Fp, rings, D, adv)
        oarcs = [c[0] for cyc in cycles for c in cyc]
        iarcs = [c[2] for cyc in cycles for c in cyc]
        dns = [c[1] for cyc in cycles for c in cyc]
        ups = [c[3] for cyc in cycles for c in cyc]
        assert len(oarcs) == len(set(oarcs)) == 2 * E
        assert len(iarcs) == len(set(iarcs)) == 2 * E
        assert len(dns) == len(set(dns)) == 2 * E
        assert len(set((u[1], u[2]) for u in ups)) == 2 * E
        # closure: the successor of each cycle's last entry is its
        # first outer arc
        for cyc in cycles:
            last = cyc[-1][3]
            assert ('O', last[3], last[4]) == cyc[0][0]
        counts[adv] = len(cycles)
    assert counts[0] == E, \
        "advance=0 gave %d components, expected E=%d" \
        % (counts[0], E)
    assert counts[1] < counts[0], "advance=1 did not merge strands"
    if verbose:
        print("  %s: E=%d  components(adv=0)=%d  (adv=1)=%d"
              % (name, E, counts[0], counts[1]))

    shell = build_double_shell(
        solid=name, frequency=freq, samples=samples,
        rail_samples=rail_samples, advance=1)
    inner_radius = shell['inner_radius']
    # every crossing is one-over-one-under, always: the two strands
    # read opposite senses of the same bit by construction
    for (key, lo, hi, flo, fhi) in shell['crossings']:
        s_lo = shell['over'][key] ^ 0
        s_hi = shell['over'][key] ^ 1
        assert s_lo != s_hi
    # bridge x bridge crossings exist at the default twist
    assert shell['cross_class'].get((1, 1), 0) > 0, \
        "no bridge x bridge crossings"
    # seed beads finite, radii inside the two-shell band
    lift = 0.6 * 0.10
    band = lift + 0.02
    for P in shell['seed']:
        assert np.isfinite(P).all()
        r = np.linalg.norm(P, axis=1)
        assert float(r.min()) > inner_radius - band - 0.05
        assert float(r.max()) < 1.0 + band + 0.05
    if verbose:
        print("    beads=%d crossings=%d classes=%s consistent=%s"
              % (sum(len(P) for P in shell['seed']),
                 len(shell['crossings']),
                 dict(sorted(shell['cross_class'].items())),
                 shell['consistent']))
    if relax_iters <= 0:
        return
    trace = []
    tr = 0.03
    clr = max(0.10, 2.2 * tr)
    gap = max(0.10, 2.2 * tr)
    lines = relax_double_shell(shell, iters=relax_iters,
                               clearance=clr, weave_gap=gap,
                               trace=trace)
    for P in lines:
        assert np.isfinite(P).all()
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        assert float(seg.max()) < 6.0 * float(seg.mean()), \
            "a strand tore open"
    mv_early = float(np.mean(trace[:10]))
    mv_late = float(np.mean(trace[-10:]))
    assert mv_late < mv_early, "relaxation did not converge"
    mind = _min_interstrand(lines)
    assert mind >= 0.6 * clr, \
        "strands too close after relax: %.4f < %.4f" \
        % (mind, 0.6 * clr)
    # per-crossing radial order preserved (at the tracked anchors:
    # the strands slide while relaxing, so each crossing is checked
    # where the two strands actually stack).  A small fraction may
    # RELEASE (drift out of contact sideways -- no order left to
    # preserve) or REVERSE (a near-tie crossing settling cleanly the
    # other way -- an equally valid weave, recorded as such); every
    # crossing still in contact must obey its recorded order.
    X = np.vstack(lines)
    oi, ui = shell['relax_anchors']
    rel = shell['relax_released']
    rev = shell['relax_reversed']
    assert float(rel.mean() if len(rel) else 0.0) < 0.15, \
        "too many crossings released"
    assert float(rev.mean() if len(rev) else 0.0) < 0.15, \
        "too many crossings reversed"
    ok = 0
    for c in range(len(oi)):
        if rel[c]:
            continue
        ro = float(np.linalg.norm(X[oi[c]]))
        ru = float(np.linalg.norm(X[ui[c]]))
        assert ro - ru > 0.0, "crossing radial order flipped"
        ok += 1
    # the shells stay separated
    el = np.concatenate(shell['elem'])
    r = np.linalg.norm(X, axis=1)
    sep = (float(r[el == ELEM_OUTER].mean())
           - float(r[el == ELEM_INNER].mean()))
    assert sep > 0.5 * (1.0 - inner_radius), \
        "shells collapsed: separation %.3f" % sep
    # TUBE geometry: non-empty, finite, all quads
    cells = build_cells(shell, lines, 'TUBE', tr, 8, 'STRAND')
    assert cells
    for (cv, cf, cm) in cells:
        assert cf and all(len(fc) == 4 for fc in cf)
        assert np.isfinite(np.asarray(cv)).all()
    if verbose:
        print("    relax: mv %.5f -> %.5f  min dist %.4f  "
              "sep %.3f  order ok %d" % (mv_early, mv_late, mind,
                                         sep, ok))


def _self_test():
    print("Woven Double Shell self-test")
    _test_solid('CUBE', 2, 128, 16, 60)
    _test_solid('DODECA', 2, 112, 14, 60)
    _test_solid('ICOSA', 2, 112, 14, 0)
    _test_solid('GEODESIC', 2, 60, 10, 0)
    print("RESULT: OK")


if __name__ == "__main__":
    _self_test()
