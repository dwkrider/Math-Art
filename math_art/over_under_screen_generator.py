
# Over-Under Screen Generator for Blender
#
# A plain-weave square lattice rendered as ONE continuous smooth
# surface: two strand families cross on a grid, weave strictly over
# and under -- the UNDER strand passing genuinely BENEATH the over
# strand -- and fuse into a single two-sheet membrane by smooth
# saddles, with a star-shaped opening in each cell: the woven
# perforated screen of the modular-constructivist relief tradition
# of Erwin Hauer and Norman Carlberg.
#
# THE WEAVE.  On a W x H grid of square cells the strands run along
# the grid lines: H+1 x-strands (rows) and W+1 y-strands (columns),
# crossing at the (W+1)(H+1) lattice points (i, j).  The over/under
# is the checkerboard parity of PLAIN weave: at crossing (i, j) the
# x-strand rides OVER iff (i + j) is even, else the y-strand is over
# -- so the sense flips across every adjacent crossing along either
# strand, the alternating condition of a true tabby.  Each strand's
# centerline height is a smooth undulation through its crossing
# targets: the x-strand of row j follows
#     z(x) = amp * cos(pi x / cell + j pi),
# the y-strand of column i follows
#     z(y) = -amp * cos(pi y / cell + i pi),
# which meet at every crossing at +amp (the over strand) and -amp
# (the under strand).  The same targets, interpolated with a C1
# cosine ease between crossings, generalize to the TWILL (2/2,
# diagonal wale) and BASKET (over-2-under-2, paired strands) drafts
# of fabric geometry.
#
# THE MEMBRANE (an implicit surface).  A height field cannot be
# two-sheet, so the fused membrane is built as the LEVEL SET of a
# smooth scalar field.  Each family is a wide flat undulating band:
# an elliptical-cross-section sweep (half-width w, half-thickness t)
# along its undulating centerline, written as a normalized inside/
# outside function.  The two families are joined by an exponential
# soft-min (the classic blended-implicit / "blobby" union), which
# webs adjacent flanks into smooth saddle membranes wherever the two
# bands come close, and a small super-Gaussian WELD term at every
# crossing fuses the over sheet to the under sheet through the weave
# gap -- one connected solid, cast like Hauer's screens.  The cell
# centres, far from every band, stay open: viewed from the front the
# openings read as 4-pointed stars pinched by the crossing saddles,
# and beside each crossing the under strand shows genuinely BELOW
# the over strand.  The zero level set is extracted by the shared
# marching-tetrahedra sampler and lightly faired; the result is a
# closed watertight shell (no separate solidify step -- the field's
# thickness IS the shell).
#
# MINIMAL (TAUT) MODE.  Pinning the free ridge lines of a marching
# mesh is ill-posed, so instead of a pinned Plateau solve the taut
# mode applies extra passes of NON-SHRINKING Taubin fairing: the
# lambda|mu two-step low-pass filter smooths and tautens the saddle
# webs without collapsing the weave relief the way an unpinned area
# minimization would.
#
# References:
#   Erwin Hauer, "Continua -- Architectural Screens and Walls",
#     Princeton Architectural Press, 2004 -- the perforated modular
#     screen designs (Designs 1-7, 1950-57) and Hauer's smooth
#     saddle-surface method; Design 1 is the woven bilayer screen
#     this generator's fused membrane follows.
#   Norman Carlberg and Erwin Hauer -- co-originators of Modular
#     Constructivism (sculptural units designed to be multiplied).
#   Branko Grunbaum and G. C. Shephard, "Satins and Twills: An
#     Introduction to the Geometry of Fabrics", Mathematics Magazine
#     53(3), 1980 -- the parity / draft description of plain, twill
#     and basket weaves used for the over/under assignment.
#   James F. Blinn, "A Generalization of Algebraic Surface Drawing",
#     ACM Transactions on Graphics 1(3), 1982 -- the blended
#     implicit-field ("blobby") technique behind the smooth union
#     that fuses the two strand families.
#   Gabriel Taubin, "A Signal Processing Approach to Fair Surface
#     Design", SIGGRAPH 1995 -- the non-shrinking lambda|mu fairing
#     used to smooth and tauten the extracted shell.

bl_info = {
    "name": "Over-Under Screen",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Plain-weave square lattice fused into one smooth "
                   "two-sheet saddle membrane with a star opening "
                   "per cell (Hauer / Carlberg woven screen)",
    "category": "Add Mesh",
}

from math import pi, cos, hypot, ceil                 # noqa: F401

import numpy as np

try:
    from . import pattern_common as pc
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc

# The marching-tetrahedra level-set sampler lives in the minimal-
# surface toolkit; imported softly so the weave math stays testable
# on its own (MEMBRANE falls back to RIBBONS without it).
try:
    from . import minimal_surface_toolkit as _mst
except Exception:
    try:
        import minimal_surface_toolkit as _mst
    except Exception:
        _mst = None


_BACKING_MAT = len(pc.PALETTE_RGBA) - 1
_HEIGHT_BANDS = 5     # palette bands of the by-height coloring
_SMIN_K = 2.2         # soft-min sharpness (normalized field units)
_WELD_SIG = 0.55      # crossing weld radius, as a fraction of w


# --------------------------------------------------------------------
# Weave drafts (over/under parity)
# --------------------------------------------------------------------
#
# x_over(i, j) is True where the x-strand (row family) rides over at
# crossing (i, j); the y-strand is over exactly where it does not --
# every crossing has exactly one family on top.  PLAIN is the
# checkerboard parity; TWILL 2/2 shifts an over-2-under-2 run one
# crossing per row (the diagonal wale); BASKET 2/2 pairs the strands.

WEAVES = ('PLAIN', 'TWILL', 'BASKET')


def x_over(i, j, weave='PLAIN'):
    """Boolean (arrays ok): x-strand over at crossing (i, j)."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if weave == 'TWILL':
        return ((i - j) % 4) < 2
    if weave == 'BASKET':
        return ((i // 2 + j // 2) % 2) == 0
    return ((i + j) % 2) == 0


def _targets(i, j, family, amp, weave):
    """Crossing height target of a strand: +amp where that strand is
    over, -amp where under.  family 0 = x-strands, 1 = y-strands."""
    over = x_over(i, j, weave)
    if family == 1:
        over = ~over
    return np.where(over, amp, -amp)


# --------------------------------------------------------------------
# Strand geometry
# --------------------------------------------------------------------

def _params(W, H, cell, strand_width, weave_depth, weave):
    """Bundle of derived geometry constants.
      w   -- strand half width
      amp -- undulation amplitude (weave_depth is peak-to-peak, as a
             fraction of the cell)"""
    cell = float(cell)
    w = 0.5 * min(max(strand_width, 0.05), 0.95) * cell
    amp = 0.5 * max(weave_depth, 0.0) * cell
    return {'W': int(W), 'H': int(H), 'cell': cell, 'w': w,
            'amp': amp, 'weave': weave}


def _strand_height(s, other_index, family, p):
    """Centerline height of one strand family at arc position `s`
    (the coordinate ALONG the strand), for the strand on grid line
    `other_index`.  The height is the C1 cosine ease through the
    +/-amp crossing targets, which for PLAIN weave reproduces
    amp * cos(pi s / cell + line * pi) exactly."""
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
# The implicit weave field
# --------------------------------------------------------------------
#
# Each family is a normalized inside/outside function of its nearest
# band: an elliptical cross-section (half-width w laterally, half-
# thickness t vertically) about the undulating centerline -- a wide
# flat lens-edged band.  f < 0 inside, f = 0 on the band surface.

def _band_fields(X, Y, Z, p, th):
    """(fx, fy, dx, dy): the two families' band functions and the
    in-plane offsets from the nearest column / row line."""
    cell, w = p['cell'], p['w']
    jn = np.clip(np.round(Y / cell), 0, p['H']).astype(np.int64)
    im = np.clip(np.round(X / cell), 0, p['W']).astype(np.int64)
    dy = Y - jn * cell
    dx = X - im * cell
    zx = _strand_height(X, jn, 0, p)
    zy = _strand_height(Y, im, 1, p)
    fx = np.sqrt((dy / w) ** 2 + ((Z - zx) / th) ** 2) - 1.0
    fy = np.sqrt((dx / w) ** 2 + ((Z - zy) / th) ** 2) - 1.0
    return fx, fy, dx, dy


def _weave_field(p, th):
    """The fused-membrane scalar field (negative inside) and the
    flush clip rectangle (x0, x1, y0, y1).

    F = softmin(fx, fy) - weld, clipped flush at the outer rim:
      * the exponential soft-min (Blinn-style blended union) webs the
        two families' flanks into smooth saddles where they approach;
      * the weld is a super-Gaussian well down each crossing axis
        that fuses the over sheet to the under sheet through the
        weave gap (radius ~_WELD_SIG * w, vertical reach ~amp), so
        the shell is one connected solid;
      * the hard max with the box planes cuts the strand ends flush
        (the crisp rectangular outer rim)."""
    cell, w, amp = p['cell'], p['w'], p['amp']
    x0, x1 = -w, p['W'] * cell + w
    y0, y1 = -w, p['H'] * cell + w
    sig = _WELD_SIG * w
    zeta = max(amp, th)
    # weld strength: enough to bridge the mid-gap field value
    # (amp/th - 1) with margin, but never a giant blob
    A = float(np.clip(amp / th - 1.0 + 0.7, 0.4, 4.0))
    k = _SMIN_K

    def field(X, Y, Z):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=float)
        fx, fy, dx, dy = _band_fields(X, Y, Z, p, th)
        F = -np.logaddexp(-k * fx, -k * fy) / k
        rho4 = ((dx * dx + dy * dy) / (sig * sig)) ** 2
        F = F - A * np.exp(-np.minimum(rho4 + (Z / zeta) ** 4, 60.0))
        clipf = np.maximum(np.maximum(x0 - X, X - x1),
                           np.maximum(y0 - Y, Y - y1)) / th
        return np.maximum(F, clipf)

    return field, (x0, x1, y0, y1)


# --------------------------------------------------------------------
# Mesh utilities (fairing, topology checks, ribbon solidify)
# --------------------------------------------------------------------

def _edge_array(faces):
    """Unique undirected edges of a face list as an (ne, 2) array."""
    E = set()
    for q in faces:
        n = len(q)
        for k in range(n):
            a, b = int(q[k]), int(q[(k + 1) % n])
            E.add((a, b) if a < b else (b, a))
    return np.array(sorted(E), dtype=int)


def _taubin(V, faces, rounds, lam=0.5, mu=-0.53):
    """Taubin lambda|mu fairing: alternating positive / negative
    umbrella steps -- smooths without the shrinkage of plain
    Laplacian smoothing, so the weave relief survives.  V is
    modified in place."""
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


def _project_level(V, field, step, iters=2):
    """Newton-project vertices back onto the zero level set of the
    field (numeric central-difference gradient).  Fairing drifts the
    mesh off the surface and marching leaves staircase ribbing on
    steep flanks; snapping each vertex to the analytic level set
    removes both.  Displacements are clamped to under a grid step so
    a degenerate gradient can never fling a vertex."""
    eps = 0.25 * step
    lim = 0.75 * step
    for _ in range(iters):
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        f0 = field(x, y, z)
        gx = (field(x + eps, y, z) - field(x - eps, y, z))
        gy = (field(x, y + eps, z) - field(x, y - eps, z))
        gz = (field(x, y, z + eps) - field(x, y, z - eps))
        g2 = gx * gx + gy * gy + gz * gz
        ok = g2 > 1e-12
        t = np.where(ok, 2.0 * eps * f0 / np.maximum(g2, 1e-12), 0.0)
        dx, dy, dz = -t * gx, -t * gy, -t * gz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        f = np.where(d > lim, lim / np.maximum(d, 1e-12), 1.0)
        V[:, 0] += dx * f
        V[:, 1] += dy * f
        V[:, 2] += dz * f
    return V


def _components(nv, faces):
    """Connected components among the vertices used by `faces`."""
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


def _shell_stats(V, T):
    """(components, genus, nonmanifold-edge count) of a closed tri
    shell.  Genus from Euler: chi = V - E + F = 2c - 2g."""
    cnt = {}
    for t in T:
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    ne = len(cnt)
    nonman = sum(1 for c in cnt.values() if c != 2)
    comps = _components(len(V), T)
    chi = len(V) - ne + len(T)
    genus = max(0, (2 * comps - chi) // 2)
    return comps, genus, nonman


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
    """Laplacian-average offset normals so thickened ribbon side
    walls cannot tear where the undulation turns steep."""
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
    """Thicken an open quad shell along its (smoothed) vertex normals
    into a closed slab: top faces, reversed bottom faces, and a side
    quad per boundary edge (oriented with the top winding so normals
    face outward).  Used by the RIBBONS mode."""
    n = len(V)
    N = _smooth_normals(_vertex_normals(V, quads), quads)
    top = V + 0.5 * thickness * N
    bot = V - 0.5 * thickness * N
    verts = np.vstack([top, bot])
    faces = [tuple(int(v) for v in q) for q in quads]
    faces += [tuple(int(q[k]) + n for k in (3, 2, 1, 0)) for q in quads]
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


def _membrane_mats(V, T, p, th, color_by):
    if color_by == 'FAMILY':
        C = (V[T[:, 0]] + V[T[:, 1]] + V[T[:, 2]]) / 3.0
        fx, fy, _dx, _dy = _band_fields(C[:, 0], C[:, 1], C[:, 2],
                                        p, th)
        return [0 if a <= b else 1 for a, b in zip(fx, fy)]
    if color_by == 'HEIGHT':
        cz = (V[T[:, 0], 2] + V[T[:, 1], 2] + V[T[:, 2], 2]) / 3.0
        return _height_mats(cz)
    return [0] * len(T)


def _ribbon_mats(V, quads, color_by, family):
    if color_by == 'FAMILY':
        return [family] * len(quads)
    if color_by == 'HEIGHT':
        cz = np.array([V[list(q), 2].mean() for q in quads])
        return _height_mats(cz)
    return [0] * len(quads)


# --------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------

def _open_cells(field, p, th):
    """How many cell centres are open sight-lines: the field stays
    positive (outside) along the whole vertical through the centre.
    A correct screen has one open star hole per cell."""
    zr = p['amp'] + th + 0.1 * p['cell']
    zs = np.linspace(-zr, zr, 61)
    n = 0
    for i in range(p['W']):
        for j in range(p['H']):
            cx = (i + 0.5) * p['cell']
            cy = (j + 0.5) * p['cell']
            v = field(np.full_like(zs, cx), np.full_like(zs, cy), zs)
            if v.min() > 0.0:
                n += 1
    return n


def _z_crossings(field, x, y, zlim, n=241):
    """Sign changes of the field along the vertical at (x, y) -- two
    per sheet pierced -- and the z-span between first and last."""
    zs = np.linspace(-zlim, zlim, n)
    v = field(np.full_like(zs, x), np.full_like(zs, y), zs)
    sgn = np.sign(v)
    idx = np.nonzero(sgn[1:] * sgn[:-1] < 0)[0]
    if len(idx) == 0:
        return 0, 0.0
    return len(idx), float(zs[idx[-1]] - zs[idx[0]])


def build_membrane(W, H, cell=1.0, strand_width=0.6, weave_depth=0.5,
                   weave='PLAIN', res=14, thickness=0.25,
                   smooth_rounds=2, surface='MEMBRANE',
                   relax_iters=12, color_by='UNIFORM'):
    """The fused two-sheet woven membrane as one closed shell (a
    marching-tetrahedra level set of the weave field).  Returns
    (cell, stats): cell = (verts, faces, mats); stats holds the open
    cell count, genus (the weave's handles), components, non-
    manifold edge count and the taut-mode method."""
    if _mst is None:
        raise RuntimeError("minimal_surface_toolkit unavailable")
    p = _params(W, H, cell, strand_width, weave_depth, weave)
    th = max(0.5 * thickness * p['cell'], 0.02 * p['cell'])
    res = max(4, int(res))
    step = p['cell'] / res
    field, (x0, x1, y0, y1) = _weave_field(p, th)
    mrg = 3.0 * step
    zr = p['amp'] + th + 0.12 * p['cell']
    box_min = (x0 - mrg, y0 - mrg, -zr - mrg)
    box_max = (x1 + mrg, y1 + mrg, zr + mrg)
    res3 = tuple(max(4, int(ceil((b - a) / step)))
                 for a, b in zip(box_min, box_max))
    V, T = _mst.marching_tets(field, box_min, box_max, res3)
    if len(T) == 0:
        return ([], [], []), {'holes': 0, 'genus': 0,
                              'components': 0, 'nonmanifold': 0,
                              'minimal': None}
    V = np.array(V, dtype=float)
    _taubin(V, T, smooth_rounds)
    _project_level(V, field, step)      # kill marching staircase
    stats = {'minimal': None}
    if surface == 'MINIMAL':
        # non-shrinking Taubin fairing in place of a pinned Plateau
        # solve: tautens the saddle webs, keeps the weave relief
        _taubin(V, T, max(1, relax_iters))
        _project_level(V, field, step, iters=1)
        stats['minimal'] = 'TAUBIN-FAIRING'
    comps, genus, nonman = _shell_stats(V, T)
    stats.update({'genus': genus, 'components': comps,
                  'nonmanifold': nonman,
                  'holes': _open_cells(field, p, th)})
    mats = _membrane_mats(V, T, p, th, color_by)
    faces = [tuple(int(v) for v in t) for t in T]
    verts = [tuple(map(float, v)) for v in V]
    return (verts, faces, mats), stats


def build_ribbons(W, H, cell=1.0, strand_width=0.6, weave_depth=0.5,
                  weave='PLAIN', res=14, thickness=0.25,
                  color_by='UNIFORM'):
    """The two families as separate woven ribbons (not fused): one
    cell per strand, so `separate` can emit them individually."""
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
            mats = _ribbon_mats(V, quads, color_by, family)
            tk = thickness * p['cell']
            if tk > 1e-9:
                sv, sf = _solidify(V, quads, tk)
                nside = len(sf) - 2 * len(quads)
                side = mats[0] if mats else 0
                mats = mats + mats + [side] * nside
                cells.append(([tuple(map(float, v)) for v in sv],
                              sf, mats))
            else:
                faces = [tuple(int(v) for v in q) for q in quads]
                cells.append(([tuple(map(float, v)) for v in V],
                              faces, mats))
    return cells


def build_screen(W, H, cell=1.0, strand_width=0.6, weave_depth=0.5,
                 weave='PLAIN', surface='MEMBRANE', res=14,
                 thickness=0.25, smooth_rounds=2, relax_iters=12,
                 backing=False, base=0.1, color_by='UNIFORM'):
    """All modes.  Returns (cells, stats)."""
    if surface == 'RIBBONS' or _mst is None:
        cells = build_ribbons(W, H, cell, strand_width, weave_depth,
                              weave, res, thickness, color_by)
        stats = {'holes': W * H, 'genus': 0,
                 'components': (W + 1) + (H + 1), 'nonmanifold': 0,
                 'minimal': None}
        if surface != 'RIBBONS':
            stats['note'] = "toolkit unavailable: ribbons fallback"
    else:
        c, stats = build_membrane(W, H, cell, strand_width,
                                  weave_depth, weave, res, thickness,
                                  smooth_rounds, surface,
                                  relax_iters, color_by)
        cells = [c]
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
        """Add a woven over-under screen (fused saddle membrane)"""
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
                    "Over-1-under-1 checkerboard (tabby)"),
                   ('TWILL', "Twill 2/2",
                    "Over-2-under-2, shifted one crossing per row "
                    "(diagonal wale)"),
                   ('BASKET', "Basket 2/2",
                    "Over-2-under-2 with strands acting in pairs")],
            default='PLAIN',
            description="The weave draft (over/under pattern)")
        surface: EnumProperty(
            name="Surface",
            items=[('RIBBONS', "Ribbons",
                    "The two strand families as separate woven "
                    "ribbons"),
                   ('MEMBRANE', "Membrane",
                    "One fused two-sheet saddle membrane with a "
                    "star opening per cell (the Hauer screen)"),
                   ('MINIMAL', "Minimal (Taut)",
                    "The membrane tautened by extra non-shrinking "
                    "(Taubin) fairing of the saddle webs")],
            default='MEMBRANE')
        strand_width: FloatProperty(
            name="Strand Width", default=0.6, min=0.15, max=0.9,
            description="Strand width as a fraction of a cell (also "
                        "sets the opening size)")
        weave_depth: FloatProperty(
            name="Weave Depth", default=0.5, min=0.0, max=2.0,
            description="Peak-to-peak over/under amplitude as a "
                        "fraction of a cell (the two-sheet "
                        "separation)")
        resolution: IntProperty(
            name="Resolution", default=14, min=4, max=48,
            description="Field samples per cell (raise for finer "
                        "shells)")
        thickness: FloatProperty(
            name="Thickness", default=0.25, min=0.0, max=1.0,
            description="Band thickness as a fraction of a cell "
                        "(the shell is the thickened field)")
        smooth_rounds: IntProperty(
            name="Smooth Rounds", default=2, min=0, max=20,
            description="Taubin fairing passes on the extracted "
                        "shell")
        relax_iters: IntProperty(
            name="Relax Iterations", default=12, min=1, max=60,
            description="Extra fairing passes (taut mode)")
        backing: BoolProperty(
            name="Backing Slab", default=False,
            description="Add a slab behind the screen")
        base: FloatProperty(
            name="Base Thickness", default=0.1, min=0.01, max=1.0,
            description="Backing slab thickness (fraction of a cell)")
        color_by: EnumProperty(
            name="Color By",
            items=[('UNIFORM', "Uniform", "A single material"),
                   ('FAMILY', "By Family",
                    "Two tones, one per strand family"),
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
                            "%s %s  holes=%d genus=%d comps=%d  "
                            "V=%d F=%d%s"
                            % (self.surface, self.weave,
                               stats['holes'], stats['genus'],
                               stats['components'],
                               len(obj.data.vertices),
                               len(obj.data.polygons), note))
            else:
                self.report({'INFO'}, "%s %s  %d parts%s"
                            % (self.surface, self.weave,
                               len(obj.children), note))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'grid_w')
            lay.prop(self, 'grid_h')
            lay.prop(self, 'weave')
            lay.prop(self, 'surface')
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
    """PLAIN is a strict alternating plain weave; every draft gives
    exactly one family over per crossing and balanced strands."""
    ok = True
    I, J = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    xo = x_over(I, J, 'PLAIN')
    ok &= bool(np.all(xo == ((I + J) % 2 == 0)))
    ok &= bool(np.all(xo[1:, :] != xo[:-1, :]))      # flips along x
    ok &= bool(np.all(xo[:, 1:] != xo[:, :-1]))      # flips along y
    for weave in WEAVES:
        xo = x_over(I, J, weave)
        # exclusivity is structural (y_over = ~x_over); check balance:
        # every strand is over at exactly half its crossings
        ok &= bool(np.all(xo.sum(axis=0) == 4))
        ok &= bool(np.all(xo.sum(axis=1) == 4))
    # TWILL wale runs diagonally
    xo = x_over(I, J, 'TWILL')
    ok &= bool(np.all(xo[1:, 1:] == xo[:-1, :-1]))
    return ok


def _check_crossing_heights():
    """At each crossing the two strands sit at +/-amp and the over
    strand (by parity) is the higher one."""
    p = _params(4, 3, 1.0, 0.5, 0.5, 'PLAIN')
    amp = p['amp']
    ok = True
    for i in range(5):
        for j in range(4):
            zx = float(_strand_height(np.array([i * 1.0]),
                                      np.int64(j), 0, p)[0])
            zy = float(_strand_height(np.array([j * 1.0]),
                                      np.int64(i), 1, p)[0])
            hi, lo = max(zx, zy), min(zx, zy)
            ok &= abs(hi - amp) < 1e-9 and abs(lo + amp) < 1e-9
            ok &= (zx > zy) == bool(x_over(i, j, 'PLAIN'))
    return ok


def _run_selftest():
    W, H = 4, 3
    par = _check_parity()
    xh = _check_crossing_heights()
    print("weave parity ok:", par, "| crossing heights ok:", xh)

    if _mst is None:
        print("minimal_surface_toolkit missing -- membrane untested")
        print("RESULT:", "OK" if (par and xh) else "BAD")
        return par and xh

    # res >= 12: below that the thin saddle webs undersample and the
    # shell picks up spurious pinhole handles (genus artifacts)
    cell, stats = build_membrane(W, H, res=12, surface='MEMBRANE')
    v, f, m = cell
    nonempty = len(v) > 0 and len(f) > 0
    finite = bool(np.all(np.isfinite(np.asarray(v))))
    holes_ok = stats['holes'] == W * H
    genus_ok = stats['genus'] == W * H       # a handle per cell
    conn_ok = stats['components'] == 1
    ne = len(_edge_array(f))
    closed_ok = stats['nonmanifold'] <= max(2, int(0.005 * ne))
    print("membrane: V=%d F=%d openings=%d (want %d) comps=%d "
          "genus=%d nonmanifold=%d/%d"
          % (len(v), len(f), stats['holes'], W * H,
             stats['components'], stats['genus'],
             stats['nonmanifold'], ne))

    # genuinely two-sheet: beside a crossing the vertical pierces
    # both the over and the under sheet (>= 4 surface crossings) and
    # the piercings span about the full weave separation; at the
    # crossing core the weld has fused the sheets (exactly 2)
    p = _params(W, H, 1.0, 0.55, 0.5, 'PLAIN')
    th = max(0.5 * 0.25, 0.02)
    fld, _box = _weave_field(p, th)
    zlim = p['amp'] + th + 0.1
    ncr, span = _z_crossings(fld, 1.0, 1.0 + 0.75 * p['w'], zlim)
    two_sheet = ncr >= 4 and span >= 1.2 * p['amp']
    ncr_c, _sp = _z_crossings(fld, 1.0, 1.0, zlim)
    welded = ncr_c == 2
    print("two-sheet: crossings=%d span=%.3f (amp=%.3f) | "
          "weld crossings=%d (want 2)" % (ncr, span, p['amp'], ncr_c))

    _c2, st2 = build_membrane(W, H, res=10, surface='MINIMAL',
                              relax_iters=6)
    min_ok = (st2['minimal'] == 'TAUBIN-FAIRING'
              and st2['components'] == 1
              and st2['holes'] == W * H)
    print("taut mode: method=%s comps=%d openings=%d"
          % (st2['minimal'], st2['components'], st2['holes']))

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

    ok = (par and xh and nonempty and finite and holes_ok
          and genus_ok and conn_ok and closed_ok and two_sheet
          and welded and min_ok and rib_ok and fit_ok)
    print("RESULT:", "OK" if ok else "BAD")
    return ok


if __name__ == "__main__":
    _run_selftest()
