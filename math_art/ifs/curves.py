# Space-filling CURVES -- Hilbert, Peano, Gosper, Moore.
#
# Part of the Math Art ifs engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Distinct from `spacefill.py`, which tiles space with CELLS.  These are
# curves whose limit is surjective onto a region: Hilbert's and Peano's
# in the square, Gosper's on the hexagonal lattice.
#
# References:
# - G. Peano, "Sur une courbe, qui remplit toute une aire plane",
#   Mathematische Annalen 36, 1890, pp. 157-160.
# - D. Hilbert, "Ueber die stetige Abbildung einer Linie auf ein
#   Flachenstuck", Mathematische Annalen 38, 1891, pp. 459-460.




def hilbert_points(order, dim):
    """Points of the Hilbert curve on the 2^order grid (Skilling's
    transpose algorithm), as integer tuples starting at the origin."""
    n = dim
    npts = 2 ** (order * n)
    pts = []
    for d in range(npts):
        # index -> transposed coordinates
        X = [0] * n
        for i in range(order * n):
            if d >> i & 1:
                X[n - 1 - i % n] |= 1 << (i // n)
        # Gray decode
        t = X[n - 1] >> 1
        for i in range(n - 1, 0, -1):
            X[i] ^= X[i - 1]
        X[0] ^= t
        # undo excess work
        Q = 2
        while Q != 1 << order:
            P = Q - 1
            for i in range(n - 1, -1, -1):
                if X[i] & Q:
                    X[0] ^= P
                else:
                    t = (X[0] ^ X[i]) & P
                    X[0] ^= t
                    X[i] ^= t
            Q <<= 1
        pts.append(tuple(X))
    return pts


def _axis_diff(a, b):
    """(axis, sign) of the single-coordinate difference b - a."""
    for k in range(len(a)):
        if a[k] != b[k]:
            return k, (1 if b[k] > a[k] else -1)
    raise ValueError("identical corners")


# per dimension: ring of orthants, entry sides of block 0, and each
# block's traversal axis. Solved once from the side-propagation
# constraints (each block's Hilbert may traverse any axis; its exit
# corner must sit on the face toward the next orthant).
_MOORE = {
    2: ([(0, 0), (1, 0), (1, 1), (0, 1)],
        (0, 1), (0, 0, 0, 0)),
    3: ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
        (0, 0, 1), (0, 1, 1, 0, 0, 1, 1, 0)),
}


def moore_points(order, dim):
    """Closed Moore curve: 2^dim order-(order-1) Hilbert blocks in
    the orthants of a Gray-code ring, each mapped by a signed axis
    permutation so every exit meets the next entry and the loop
    closes."""
    ring, entry0, taus = _MOORE[dim]
    if order < 2:
        return list(ring)
    s = 2 ** (order - 1)                 # sub-block grid size
    base = hilbert_points(order - 1, dim)
    alpha, _ = _axis_diff(base[0], base[-1])   # base runs along alpha
    pts = []
    sigma = list(entry0)                 # entry corner sides (0/1)
    for k in range(len(ring)):
        d_axis, d_sign = _axis_diff(ring[k],
                                    ring[(k + 1) % len(ring)])
        tau = taus[k]                    # this block's traversal axis
        origin = tuple(c * s for c in ring[k])
        # signed permutation: base axis alpha -> block axis tau
        perm = [None] * dim              # block axis -> base axis
        perm[tau] = alpha
        rest = iter(a for a in range(dim) if a != alpha)
        for a in range(dim):
            if perm[a] is None:
                perm[a] = next(rest)
        for p in base:
            pts.append(tuple(
                origin[a] + (s - 1 - p[perm[a]] if sigma[a]
                             else p[perm[a]])
                for a in range(dim)))
        # next entry sides: flip along tau (exit), cross along d_axis
        sigma[tau] = 1 - sigma[tau]
        sigma[d_axis] = 1 - (1 if d_sign > 0 else 0)
    return pts


def _sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def _gilbert2d(x, y, ax, ay, bx, by):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = _sgn(ax), _sgn(ay)
    dbx, dby = _sgn(bx), _sgn(by)
    if h == 1:
        for _ in range(w):
            yield (x, y)
            x, y = x + dax, y + day
        return
    if w == 1:
        for _ in range(h):
            yield (x, y)
            x, y = x + dbx, y + dby
        return
    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)
    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            ax2, ay2 = ax2 + dax, ay2 + day
        yield from _gilbert2d(x, y, ax2, ay2, bx, by)
        yield from _gilbert2d(x + ax2, y + ay2,
                              ax - ax2, ay - ay2, bx, by)
    else:
        if (h2 % 2) and (h > 2):
            bx2, by2 = bx2 + dbx, by2 + dby
        yield from _gilbert2d(x, y, bx2, by2, ax2, ay2)
        yield from _gilbert2d(x + bx2, y + by2, ax, ay,
                              bx - bx2, by - by2)
        yield from _gilbert2d(x + (ax - dax) + (bx2 - dbx),
                              y + (ay - day) + (by2 - dby),
                              -bx2, -by2,
                              -(ax - ax2), -(ay - ay2))


def gilbert2d(width, height):
    """Every cell of a width x height grid exactly once, with unit
    steps, for arbitrary (not just power-of-two) sizes."""
    if width >= height:
        return list(_gilbert2d(0, 0, width, 0, 0, height))
    return list(_gilbert2d(0, 0, 0, height, width, 0))


def _gilbert3d(x, y, z, ax, ay, az, bx, by, bz, cx, cy, cz):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)
    dax, day, daz = _sgn(ax), _sgn(ay), _sgn(az)
    dbx, dby, dbz = _sgn(bx), _sgn(by), _sgn(bz)
    dcx, dcy, dcz = _sgn(cx), _sgn(cy), _sgn(cz)
    if h == 1 and d == 1:
        for _ in range(w):
            yield (x, y, z)
            x, y, z = x + dax, y + day, z + daz
        return
    if w == 1 and d == 1:
        for _ in range(h):
            yield (x, y, z)
            x, y, z = x + dbx, y + dby, z + dbz
        return
    if w == 1 and h == 1:
        for _ in range(d):
            yield (x, y, z)
            x, y, z = x + dcx, y + dcy, z + dcz
        return
    ax2, ay2, az2 = ax // 2, ay // 2, az // 2
    bx2, by2, bz2 = bx // 2, by // 2, bz // 2
    cx2, cy2, cz2 = cx // 2, cy // 2, cz // 2
    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)
    if (w2 % 2) and (w > 2):
        ax2, ay2, az2 = ax2 + dax, ay2 + day, az2 + daz
    if (h2 % 2) and (h > 2):
        bx2, by2, bz2 = bx2 + dbx, by2 + dby, bz2 + dbz
    if (d2 % 2) and (d > 2):
        cx2, cy2, cz2 = cx2 + dcx, cy2 + dcy, cz2 + dcz
    if (2 * w > 3 * h) and (2 * w > 3 * d):
        # wide case: split in w only
        yield from _gilbert3d(x, y, z, ax2, ay2, az2,
                              bx, by, bz, cx, cy, cz)
        yield from _gilbert3d(x + ax2, y + ay2, z + az2,
                              ax - ax2, ay - ay2, az - az2,
                              bx, by, bz, cx, cy, cz)
    elif 3 * h > 4 * d:
        # do not split in d
        yield from _gilbert3d(x, y, z, bx2, by2, bz2,
                              cx, cy, cz, ax2, ay2, az2)
        yield from _gilbert3d(x + bx2, y + by2, z + bz2,
                              ax, ay, az,
                              bx - bx2, by - by2, bz - bz2,
                              cx, cy, cz)
        yield from _gilbert3d(x + (ax - dax) + (bx2 - dbx),
                              y + (ay - day) + (by2 - dby),
                              z + (az - daz) + (bz2 - dbz),
                              -bx2, -by2, -bz2,
                              cx, cy, cz,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2))
    elif 3 * d > 4 * h:
        # do not split in h
        yield from _gilbert3d(x, y, z, cx2, cy2, cz2,
                              ax2, ay2, az2, bx, by, bz)
        yield from _gilbert3d(x + cx2, y + cy2, z + cz2,
                              ax, ay, az, bx, by, bz,
                              cx - cx2, cy - cy2, cz - cz2)
        yield from _gilbert3d(x + (ax - dax) + (cx2 - dcx),
                              y + (ay - day) + (cy2 - dcy),
                              z + (az - daz) + (cz2 - dcz),
                              -cx2, -cy2, -cz2,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2),
                              bx, by, bz)
    else:
        # regular case: split in all of w/h/d
        yield from _gilbert3d(x, y, z, bx2, by2, bz2,
                              cx2, cy2, cz2, ax2, ay2, az2)
        yield from _gilbert3d(x + bx2, y + by2, z + bz2,
                              cx, cy, cz, ax2, ay2, az2,
                              bx - bx2, by - by2, bz - bz2)
        yield from _gilbert3d(x + (bx2 - dbx) + (cx - dcx),
                              y + (by2 - dby) + (cy - dcy),
                              z + (bz2 - dbz) + (cz - dcz),
                              ax, ay, az, -bx2, -by2, -bz2,
                              -(cx - cx2), -(cy - cy2),
                              -(cz - cz2))
        yield from _gilbert3d(x + (ax - dax) + bx2 + (cx - dcx),
                              y + (ay - day) + by2 + (cy - dcy),
                              z + (az - daz) + bz2 + (cz - dcz),
                              -cx, -cy, -cz,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2),
                              bx - bx2, by - by2, bz - bz2)
        yield from _gilbert3d(x + (ax - dax) + (bx2 - dbx),
                              y + (ay - day) + (by2 - dby),
                              z + (az - daz) + (bz2 - dbz),
                              -bx2, -by2, -bz2,
                              cx2, cy2, cz2,
                              -(ax - ax2), -(ay - ay2),
                              -(az - az2))


def gilbert3d(width, height, depth):
    """Every cell of a width x height x depth cuboid exactly once,
    with unit steps, for arbitrary sizes (even sizes recommended)."""
    if width >= height and width >= depth:
        return list(_gilbert3d(0, 0, 0, width, 0, 0,
                               0, height, 0, 0, 0, depth))
    if height >= width and height >= depth:
        return list(_gilbert3d(0, 0, 0, 0, height, 0,
                               width, 0, 0, 0, 0, depth))
    return list(_gilbert3d(0, 0, 0, 0, 0, depth,
                           width, 0, 0, 0, height, 0))


def chaikin(pts, rounds, closed):
    """Corner-cutting smoothing (keeps endpoints of open curves)."""
    for _ in range(rounds):
        new = []
        n = len(pts)
        rng = range(n) if closed else range(n - 1)
        if not closed:
            new.append(pts[0])
        for i in rng:
            a, b = pts[i], pts[(i + 1) % n]
            new.append(tuple(0.75 * a[k] + 0.25 * b[k]
                             for k in range(3)))
            new.append(tuple(0.25 * a[k] + 0.75 * b[k]
                             for k in range(3)))
        if not closed:
            new.append(pts[-1])
        pts = new
    return pts


def build_curve(kind='HILBERT3D', order=3, size=2.0, rounds=1,
                gw=12, gh=8, gd=4):
    """Returns (points3d, closed)."""
    dim = 3 if kind.endswith('3D') else 2
    if kind.startswith('GILBERT'):
        if dim == 3:
            ipts = gilbert3d(gw, gh, gd)
            dims = (gw, gh, gd)
        else:
            ipts = gilbert2d(gw, gh)
            dims = (gw, gh, 1)
        s = size / max(dims)
        pts = [tuple((p[k] - (dims[k] - 1) / 2.0) * s
                     if k < dim else 0.0
                     for k in range(3))
               for p in (q + (0,) * (3 - len(q)) for q in ipts)]
        return chaikin(pts, rounds, False), False
    if kind.startswith('HILBERT'):
        ipts = hilbert_points(order, dim)
        n = 2 ** order
        closed = False
    else:
        ipts = moore_points(order, dim)
        n = 2 ** order
        closed = True
    s = size / n
    off = size / 2.0 - s / 2.0
    pts = [(p[0] * s - off, p[1] * s - off,
            (p[2] * s - off) if dim == 3 else 0.0) for p in ipts]
    return chaikin(pts, rounds, closed), closed
