# Voxel grids to watertight meshes -- the shared surfacing layer.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Both fractal families in this package land in the same place: a set of
# occupied cells on an integer lattice.  Turning that into a printable
# object is one problem, solved once here -- walk the exterior faces
# (a face is on the boundary exactly when one of the two cells sharing
# it is occupied and the other is not), then repair and normalise.
#
# The walker is vectorised over the six face directions; the pure-Python
# fallback `_voxel_surface_slow` is kept because it is obviously correct
# and the self-tests check the fast path against it.
#
# References:
# - W. E. Lorensen and H. E. Cline, "Marching cubes: A high resolution
#   3D surface construction algorithm", SIGGRAPH 1987 -- the contouring
#   ancestor; the smooth output here uses marching tetrahedra from the
#   sibling `minsurf` package instead, which cannot produce ambiguous
#   cases.

import numpy as np


def _toolkit():
    """The sibling `minsurf` engine package supplies marching_tets.

    Note the TWO dots: `minsurf` is a sibling of this package, not a
    child of it.  A single dot resolves to `math_art.ifs.minsurf`, which
    does not exist -- and the flat fallback then hides the mistake
    anywhere `math_art/` happens to be on sys.path, which is the case in
    the headless test runner but NOT inside Blender.
    """
    try:
        from .. import minsurf as mst
    except ImportError:
        import minsurf as mst
    return mst


_FACE_DIRS = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)


def edge_stats(faces):
    """(boundary, non_manifold) edge counts.

    A boundary edge (one incident face) means the surface has a hole --
    always a bug here.  An edge with FOUR incident faces means two cubes
    of the set meet only along that edge; the surface still encloses its
    solid, but it is not a manifold there.  That happens for real in
    these families -- Sierpinski-like sets touch at edges and corners by
    construction -- so it is reported, not treated as an error."""
    if not len(faces):
        return 0, 0
    e = []
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e.append((a, b) if a < b else (b, a))
    arr = np.asarray(e, dtype=np.int64)
    _, counts = np.unique(arr, axis=0, return_counts=True)
    return (int(np.sum(counts == 1)), int(np.sum(counts > 2)))


def blur_density(dens):
    """Separable 3-tap blur with ZERO padding, and an empty shell left
    around the grid.

    np.roll would wrap, bleeding density from one face of the box to
    the opposite one; and unless the outermost layer is empty the
    contour runs into the sample box and comes out as an open surface
    with a boundary, whose normals then mean nothing."""
    out = dens.astype(float)
    for axis in (0, 1, 2):
        lo = np.zeros_like(out)
        hi = np.zeros_like(out)
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(1, None)
        sl_hi[axis] = slice(None, -1)
        lo[tuple(sl_hi)] = out[tuple(sl_lo)]
        hi[tuple(sl_lo)] = out[tuple(sl_hi)]
        out = (out + lo + hi) / 3.0
    out[0, :, :] = out[-1, :, :] = 0.0
    out[:, 0, :] = out[:, -1, :] = 0.0
    out[:, :, 0] = out[:, :, -1] = 0.0
    return out


def orient_outward(verts, faces):
    """Reverse every face when the mesh is inside-out.

    The divergence theorem gives the enclosed volume as
    sum over faces of (1/6) v0 . ((v1-v0) x (v2-v0)); a closed surface
    whose normals point outward has it positive.  An orientation-
    REVERSING transform -- and det M is negative for every ABC tile, so
    M^-k reverses at odd levels -- silently turns a solid inside out
    without changing a single vertex."""
    V = np.asarray(verts, dtype=float)
    tot = 0.0
    for f in faces:
        f = list(f)
        for i in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[i]], V[f[i + 1]]
            tot += float(np.dot(a, np.cross(b - a, c - a)))
    if tot < 0.0:
        return [tuple(reversed(tuple(f))) for f in faces]
    return [tuple(f) for f in faces]


def fill_pinholes(cells, res, need=5):
    """Fill empty cells that have `need` or more of their six face
    neighbours occupied.

    A finite sample leaves isolated gaps inside a solid region; they
    are sampling noise, and each one is a void a slicer would try to
    print around.  The threshold is deliberately conservative -- five
    of six means the cell is all but enclosed -- so this closes
    pinholes without dilating the set or bridging a genuine thin gap
    (which would inflate the volume, the trap with rasterising these
    tiles)."""
    g = np.zeros((res, res, res), dtype=bool)
    g[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    n = np.zeros((res, res, res), dtype=np.int8)
    for axis in (0, 1, 2):
        for shift in (1, -1):
            n += np.roll(g, shift, axis=axis)
    g |= (~g) & (n >= int(need))
    return np.argwhere(g).astype(np.int64)


def _packer(pts, pad=2):
    """A collision-free integer key for lattice points, or None when
    the bounding box is too large to pack into an int64."""
    lo = pts.min(axis=0) - pad
    span = (pts.max(axis=0) + pad) - lo + 1
    if int(span[0]) * int(span[1]) * int(span[2]) > (1 << 62):
        return None
    return lo, span.astype(np.int64)


def _pack(pts, lo, span):
    q = pts - lo
    return (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]


def voxel_surface(cells):
    """Watertight exterior surface of a set of unit cubes on the
    integer lattice: only faces between an occupied cell and an empty
    neighbour are emitted, with shared vertices.  Returns integer
    vertices and an (n, 4) array of quad faces.

    Same walker the Fractal Sponge generator uses, but vectorised --
    the occupancy and vertex-welding lookups are sorted-key searches
    rather than Python dict hits, because these sets run to hundreds of
    thousands of cells."""
    cells = np.unique(np.asarray(cells, dtype=np.int64), axis=0)
    if not len(cells):
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 4),
                                                          dtype=np.int64)
    pk = _packer(cells)
    if pk is None:                     # pathological span: sparse path
        return _voxel_surface_slow(cells)
    lo, span = pk
    keys = np.sort(_pack(cells, lo, span))

    quads = []
    for (d, corners) in _FACE_DIRS:
        nb = cells + np.asarray(d, dtype=np.int64)
        nk = _pack(nb, lo, span)
        pos = np.searchsorted(keys, nk)
        pos_c = np.clip(pos, 0, len(keys) - 1)
        occupied = keys[pos_c] == nk
        free = cells[~occupied]
        if not len(free):
            continue
        quads.append(np.stack(
            [free + np.asarray(c, dtype=np.int64) for c in corners],
            axis=1))
    if not quads:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 4),
                                                          dtype=np.int64)
    Q = np.concatenate(quads, axis=0)              # (nf, 4, 3)
    flat = Q.reshape(-1, 3)
    verts, inv = np.unique(flat, axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 4)


def _voxel_surface_slow(cells):
    """Dict-based fallback for point sets whose bounding box will not
    pack into an int64 key."""
    occ = set(map(tuple, cells.tolist()))
    verts, vid, faces = [], {}, []

    def vertex(key):
        i = vid.get(key)
        if i is None:
            i = len(verts)
            vid[key] = i
            verts.append(key)
        return i

    for (cx, cy, cz) in sorted(occ):
        for (d, corners) in _FACE_DIRS:
            if (cx + d[0], cy + d[1], cz + d[2]) in occ:
                continue
            faces.append([vertex((cx + a, cy + b, cz + c))
                          for (a, b, c) in corners])
    return (np.asarray(verts, dtype=np.int64),
            np.asarray(faces, dtype=np.int64))


MAX_CELLS = 300000


def _as_quads(faces):
    """The vectorised walker returns an (n, 4) array; Blender wants a
    list of tuples."""
    return [tuple(int(i) for i in f) for f in np.asarray(faces)]


def _occupied_cells(P, res, cover=0.98):
    """Bin points into a res^3 grid over their own bounding box and
    return (cells, counts, lo, cell_size).  A robust box is used: the
    outermost `1 - cover` of the points, which for a chaos game are
    stragglers still converging, would otherwise stretch the grid."""
    lo = np.quantile(P, (1.0 - cover) / 2.0, axis=0)
    hi = np.quantile(P, 1.0 - (1.0 - cover) / 2.0, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    s = float(np.max(span)) / int(res)
    lo = 0.5 * (lo + hi) - 0.5 * s * int(res)
    idx = np.floor((P - lo) / s).astype(np.int64)
    idx = idx[np.all((idx >= 0) & (idx < int(res)), axis=1)]
    cells, counts = np.unique(idx, axis=0, return_counts=True)
    return cells, counts, lo, s


def keep_largest(verts, tris):
    """Biggest connected piece only -- a chaos game on a coarse grid
    leaves speckle."""
    parent = list(range(len(verts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in tris:
        a = find(int(t[0]))
        for i in (1, 2):
            b = find(int(t[i]))
            if a != b:
                parent[b] = a
    lab = np.array([find(int(t[0])) for t in tris])
    vals, counts = np.unique(lab, return_counts=True)
    keep = tris[lab == vals[int(np.argmax(counts))]]
    used = np.unique(keep)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[keep]


def center_fit(verts, scale=1.0):
    """Centre on the bounding box and fit the largest extent to a 2 m
    cube (the project-wide convention), then apply `scale`."""
    verts = np.asarray(verts, dtype=float)
    if not len(verts):
        return verts
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = float((hi - lo).max())
    return (verts - 0.5 * (lo + hi)) * (2.0 / ext
                                        if ext > 1e-9 else 1.0) * scale


def _signed_volume(verts, faces):
    """Enclosed volume by the divergence theorem; positive when the
    normals point outward."""
    V = np.asarray(verts, dtype=float)
    tot = 0.0
    for f in faces:
        f = list(f)
        for i in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[i]], V[f[i + 1]]
            tot += float(np.dot(a, np.cross(b - a, c - a))) / 6.0
    return tot


def _selftest():
    """The surfacing invariants, checked against the slow reference.

    The fast walker is vectorised over the six face directions and is
    the one thing here that could silently produce a plausible-but-wrong
    mesh, so it is checked against `_voxel_surface_slow` -- which is
    obviously correct and therefore worth keeping.
    """
    ok = True

    # A single cell is a cube: 8 verts, 6 quads, closed, volume 1.
    cells = np.array([[0, 0, 0]], dtype=np.int64)
    V, F = voxel_surface(cells)
    good = len(F) == 6 and abs(abs(_signed_volume(V, F)) - 1.0) < 1e-9
    ok &= good
    print(f"voxel: one cell -> {len(F)} faces, volume "
          f"{abs(_signed_volume(V, F)):.3f} {'OK' if good else 'FAIL'}")

    # A solid n x n x n block has exactly 6 n^2 exterior faces and volume
    # n^3: interior faces must all cancel.
    bad = []
    for n in (2, 3, 4):
        g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), -1)
        c = g.reshape(-1, 3).astype(np.int64)
        V, F = voxel_surface(c)
        if len(F) != 6 * n * n or abs(abs(_signed_volume(V, F)) - n ** 3) > 1e-6:
            bad.append(f"{n}:{len(F)}!={6*n*n}")
    good = not bad
    ok &= good
    print(f"voxel: n^3 blocks have 6n^2 exterior faces and volume n^3 "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # The fast walker must agree with the slow reference face for face,
    # on a shape with concavities and a through-hole.
    rng = np.random.default_rng(12345)
    g = np.stack(np.meshgrid(*[np.arange(5)] * 3, indexing='ij'), -1)
    c = g.reshape(-1, 3).astype(np.int64)
    c = c[rng.random(len(c)) > 0.35]
    Vf, Ff = voxel_surface(c)
    Vs, Fs = _voxel_surface_slow(c)
    key = lambda V, F: sorted(tuple(sorted(tuple(np.round(V[i], 9))
                                           for i in f)) for f in F)
    good = key(Vf, Ff) == key(Vs, Fs)
    ok &= good
    print(f"voxel: fast walker matches the slow reference on {len(c)} "
          f"scattered cells ({len(Ff)} faces) {'OK' if good else 'FAIL'}")

    # Every exterior-face mesh must be closed: each edge used exactly
    # twice, so the boundary-edge count is zero.
    nb, _nm = edge_stats(Ff)
    good = nb == 0
    ok &= good
    print(f"voxel: the scattered mesh is closed ({nb} boundary edges) "
          f"{'OK' if good else 'FAIL'}")

    # center_fit puts the result in the 2 m cube, centred at the origin --
    # the project-wide convention.
    Vc = center_fit(Vf, 1.0)
    ext = Vc.max(axis=0) - Vc.min(axis=0)
    ctr = 0.5 * (Vc.max(axis=0) + Vc.min(axis=0))
    good = (abs(float(ext.max()) - 2.0) < 1e-9
            and float(np.abs(ctr).max()) < 1e-9)
    ok &= good
    print(f"voxel: center_fit gives extent {float(ext.max()):.6f} centred "
          f"at {float(np.abs(ctr).max()):.1e} {'OK' if good else 'FAIL'}")

    # orient_outward must make the signed volume positive, whatever the
    # incoming winding.
    Fo = orient_outward(Vf, [tuple(int(i) for i in f) for f in Ff])
    good = _signed_volume(Vf, Fo) > 0
    ok &= good
    print(f"voxel: orient_outward gives positive volume "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("voxel self-test failed")
