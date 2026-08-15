"""relief.mesh -- turning a sampled height field into geometry.

Two output forms in this phase:

* **SHEET** -- the displaced grid alone, an open surface.
* **SLAB** -- the panel form: the relief on top, a flat back, and skirt walls
  joining them.  This is watertight, which is what printing, milling and
  boolean work all require.

The slab is built so that watertightness is structural rather than hopeful.
Top and bottom carry the same quad topology over the masked region; the skirt
is generated from the *boundary edges* of the top surface -- the edges that
belong to exactly one top face.  Every edge then lies in exactly two faces: an
interior edge in two top (or two bottom) faces, a rim edge in one top face and
one skirt quad, and each vertical skirt edge in two adjacent skirt quads.  That
holds for any mask whose boundary forms closed loops, so a disc panel is as
watertight as a rectangular one with no special-casing.

The fit deliberately defaults to the *footprint*, not the bounding box.  Fitting
the whole bounding box into a fixed cube -- the convention this add-on uses for
closed surfaces -- would make the relief-depth control shrink the panel, since
Z would compete with X for the fit budget.
"""

import numpy as np

FORMS = ('SHEET', 'SLAB')
FITS = ('FOOTPRINT', 'CUBE', 'NONE')


def _index_map(mask):
    """Vertex index per masked sample, -1 elsewhere.  Returns (idx, count)."""
    idx = np.full(mask.shape, -1, dtype=np.int64)
    n = int(mask.sum())
    idx[mask] = np.arange(n, dtype=np.int64)
    return idx, n


def _quads(idx, mask):
    """Quad faces over cells whose four corners are all inside the mask.

    Winding is counter-clockwise seen from +Z.
    """
    ny, nx = mask.shape
    a = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1]
    rr, cc = np.nonzero(a)
    return np.stack([idx[rr, cc], idx[rr, cc + 1],
                     idx[rr + 1, cc + 1], idx[rr + 1, cc]], axis=1)


def _boundary_directed_edges(faces):
    """Directed edges of `faces` whose reverse is absent -- the rim.

    Returned in the direction they appear in their own face, so a skirt quad
    built with the edge reversed keeps the whole mesh consistently oriented.
    """
    e = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            e[(int(f[i]), int(f[(i + 1) % k]))] = True
    return [ab for ab in e if (ab[1], ab[0]) not in e]


def sheet(X, Y, Z, mask=None):
    """Open surface: the displaced grid over the mask."""
    if mask is None:
        mask = np.ones(X.shape, dtype=bool)
    idx, n = _index_map(mask)
    verts = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
    faces = [[int(v) for v in q] for q in _quads(idx, mask)]
    return verts, faces


def slab(X, Y, Z, mask=None, base_thickness=0.1):
    """Watertight panel: relief on top, flat back, skirt walls between."""
    if mask is None:
        mask = np.ones(X.shape, dtype=bool)
    idx, n = _index_map(mask)
    top = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
    z_base = float(Z[mask].min()) - float(base_thickness)
    bot = top.copy()
    bot[:, 2] = z_base
    verts = np.concatenate([top, bot], axis=0)

    quads = _quads(idx, mask)
    faces = [[int(v) for v in q] for q in quads]              # top, CCW up
    # Bottom: same cells, reversed so the outward normal points down.
    faces += [[int(q[3]) + n, int(q[2]) + n, int(q[1]) + n, int(q[0]) + n]
              for q in quads]
    # Skirt: one quad per rim edge, edge reversed so orientation stays
    # consistent with the top face it came from.
    for (a, b) in _boundary_directed_edges(quads):
        faces.append([b, a, a + n, b + n])
    return verts, faces


def apply_fit(verts, mode='FOOTPRINT', scale=1.0, span=2.0):
    """Centre and scale the result.

    FOOTPRINT  uniform scale so the longer XY axis spans `span`*scale; Z rides
               along, so relief depth stays proportional and the depth control
               does not resize the panel.
    CUBE       whole bounding box into the `span` cube (this add-on's
               convention for closed surfaces).
    NONE       centred only, literal metres.
    """
    v = np.asarray(verts, dtype=float).copy()
    if v.size == 0:
        return v
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    v -= 0.5 * (lo + hi)
    ext = hi - lo
    if mode == 'FOOTPRINT':
        e = float(max(ext[0], ext[1]))
        s = (span / e) if e > 1e-12 else 1.0
    elif mode == 'CUBE':
        e = float(ext.max())
        s = (span / e) if e > 1e-12 else 1.0
    elif mode == 'NONE':
        s = 1.0
    else:
        raise ValueError("unknown fit: %r" % (mode,))
    return v * (s * float(scale))


def pierce(mask, h, threshold=0.0, invert=False, min_island=8):
    """Open holes through the panel wherever the field falls below a level.

    Piercing needs no new meshing: `slab` walls whatever boundary the mask
    has, and the boundary of a mask with holes in it includes those holes, so
    removing samples is the whole operation.  The result is watertight for the
    same reason the unpierced slab is.

    `min_island` discards holes smaller than that many samples.  A threshold
    cutting near a noisy field's mean opens a scatter of one- and two-sample
    pinholes that no process can make -- too fine to mill, too fine to print,
    and on screen just dirt.
    """
    m = np.asarray(mask, dtype=bool)
    z = np.asarray(h, dtype=float)
    hole = (z > float(threshold)) if invert else (z < float(threshold))
    hole &= m
    if int(min_island) > 1 and hole.any():
        hole = _drop_small(hole, int(min_island))
    return _break_pinches(m & ~hole)


def quad_mask(keep):
    """Which cells become faces: those whose four corner samples are kept."""
    k = np.asarray(keep, dtype=bool)
    return k[:-1, :-1] & k[:-1, 1:] & k[1:, :-1] & k[1:, 1:]


def _count_pinches(keep):
    """Diagonal pinches, counted between FACES rather than between samples.

    This distinction is the whole of it.  A face exists where all four corner
    samples are kept, so it is faces that can touch at a corner and faces
    whose walls then share an edge four ways.  Counting pinches in the sample
    mask instead reports zero on a mask riddled with them.
    """
    q = quad_mask(keep)
    a = q[:-1, :-1]
    b = q[1:, 1:]
    c = q[:-1, 1:]
    d = q[1:, :-1]
    return int(((a & b & ~c & ~d) | (c & d & ~a & ~b)).sum())


def _break_pinches(keep):
    """Fill the diagonal pinches a threshold leaves behind.

    Where two kept cells meet only at a corner -- a 2x2 block holding one
    diagonal pair -- the walls built around the two holes meet along a single
    edge, and that edge ends up shared by four faces instead of two.  The
    result passes a watertightness check (no boundary edge is unpaired) while
    being **non-manifold**, so it is not a solid: no printer or mill can make
    material that touches itself along a line of zero thickness.

    Filling one of the two missing cells removes the pinch and costs a single
    sample of opening.  Keeping material rather than removing it is the safer
    direction: the alternative disconnects the two kept cells, which can
    orphan a whole region of the panel.
    """
    k = np.array(keep, dtype=bool)
    # Filling one pinch can create another, so iterate to a fixed point.  Each
    # pass strictly adds samples and the grid is finite, so it terminates; the
    # cap is a backstop, not the mechanism.
    for _ in range(64):
        q = quad_mask(k)
        a = q[:-1, :-1]          # face at (j, i)
        b = q[1:, 1:]            # face at (j+1, i+1)
        c = q[:-1, 1:]           # face at (j, i+1)
        d = q[1:, :-1]           # face at (j+1, i)
        # The two diagonal orientations need different faces filled: filling
        # the same one for both silently fixes only half of them.
        fall = a & b & ~c & ~d   # faces on the main diagonal
        rise = c & d & ~a & ~b   # faces on the anti-diagonal
        if not (fall.any() or rise.any()):
            break
        # Realising a face means keeping all four of its corner samples.
        for sel, dj, di in ((fall, 0, 1), (rise, 0, 0)):
            jj, ii = np.nonzero(sel)
            for oj in (0, 1):
                for oi in (0, 1):
                    k[jj + dj + oj, ii + di + oi] = True
    return k


def _drop_small(flag, least):
    """Clear connected runs of `flag` smaller than `least` samples."""
    out = np.array(flag, dtype=bool)
    seen = np.zeros(out.shape, dtype=bool)
    ny, nx = out.shape
    for j0 in range(ny):
        for i0 in range(nx):
            if not out[j0, i0] or seen[j0, i0]:
                continue
            stack = [(j0, i0)]
            seen[j0, i0] = True
            cells = []
            while stack:
                j, i = stack.pop()
                cells.append((j, i))
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = j + dj, i + di
                    if 0 <= a < ny and 0 <= b < nx and out[a, b]                             and not seen[a, b]:
                        seen[a, b] = True
                        stack.append((a, b))
            if len(cells) < least:
                for j, i in cells:
                    out[j, i] = False
    return out


def edge_report(faces):
    """(open_edges, nonmanifold_edges, oriented) for a face list."""
    from collections import Counter
    und = Counter()
    dire = Counter()
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = int(f[i]), int(f[(i + 1) % k])
            und[(min(a, b), max(a, b))] += 1
            dire[(a, b)] += 1
    open_e = sum(1 for c in und.values() if c == 1)
    nonman = sum(1 for c in und.values() if c > 2)
    oriented = all(c == 1 for c in dire.values())
    return open_e, nonman, oriented


def _selftest():
    ok = True
    from . import grid as _grid

    for shape in ('RECT', 'DISC', 'ELLIPSE', 'ROUNDED_RECT'):
        X, Y, info = _grid.make_grid(width=2.0, aspect=1.3, resolution=61)
        mask = _grid.mask_for(shape, X, Y)
        Z = 0.1 * np.sin(6.0 * X) * np.cos(5.0 * Y)

        v, f = slab(X, Y, Z, mask, base_thickness=0.08)
        open_e, nonman, oriented = edge_report(f)
        print("mesh: SLAB %-13s V=%-6d F=%-6d open=%d nonmanifold=%d "
              "oriented=%s" % (shape, len(v), len(f), open_e, nonman, oriented))
        ok = ok and open_e == 0 and nonman == 0 and oriented
        ok = ok and np.isfinite(v).all()

        # The flat back really is flat, and sits below the relief.
        zb = v[len(v) // 2:, 2]
        ok = ok and float(zb.max() - zb.min()) < 1e-12
        ok = ok and float(zb.max()) < float(Z[mask].min()) + 1e-12

        # A sheet over the same mask is open by construction.
        vs, fs = sheet(X, Y, Z, mask)
        oe, nm, _ = edge_report(fs)
        ok = ok and oe > 0 and nm == 0

    # Fit modes.
    X, Y, info = _grid.make_grid(width=2.0, aspect=0.5, resolution=41)
    Z = 0.4 * np.sin(4.0 * X)
    v, f = slab(X, Y, Z, None, base_thickness=0.1)

    fp = apply_fit(v, 'FOOTPRINT', scale=1.0, span=2.0)
    ext = fp.max(axis=0) - fp.min(axis=0)
    print("mesh: FOOTPRINT extents = %s (long XY axis should be 2.0)"
          % np.round(ext, 4))
    ok = ok and abs(max(ext[0], ext[1]) - 2.0) < 1e-9

    # The decisive property: changing relief depth must NOT resize the panel
    # under FOOTPRINT, but does under CUBE.
    v2, _ = slab(X, Y, 4.0 * Z, None, base_thickness=0.1)
    fp2 = apply_fit(v2, 'FOOTPRINT', 1.0, 2.0)
    cu = apply_fit(v, 'CUBE', 1.0, 2.0)
    cu2 = apply_fit(v2, 'CUBE', 1.0, 2.0)
    fw1 = (fp.max(axis=0) - fp.min(axis=0))[0]
    fw2 = (fp2.max(axis=0) - fp2.min(axis=0))[0]
    cw1 = (cu.max(axis=0) - cu.min(axis=0))[0]
    cw2 = (cu2.max(axis=0) - cu2.min(axis=0))[0]
    print("mesh: 4x deeper relief -> FOOTPRINT width %.4f->%.4f, "
          "CUBE width %.4f->%.4f" % (fw1, fw2, cw1, cw2))
    ok = ok and abs(fw1 - fw2) < 1e-9        # footprint is depth-independent
    ok = ok and cw2 < cw1 - 1e-3             # cube shrinks the panel

    # Piercing opens holes right through, and the result must still be a
    # solid: `slab` walls whatever boundary the mask has, and a hole's rim is
    # part of that boundary, so a pierced panel is watertight for exactly the
    # same reason an unpierced one is.
    Xp, Yp, ip = _grid.make_grid(width=2.0, aspect=1.0, resolution=97)
    field = np.sin(4.0 * Xp) * np.cos(4.0 * Yp)
    full = np.ones(Xp.shape, dtype=bool)
    for thr in (-0.3, 0.0, 0.3):
        pm = pierce(full, field, threshold=thr, min_island=6)
        vv, ff = slab(Xp, Yp, 0.1 * field, pm, 0.1)
        op, nm, ori = edge_report(ff)
        print("mesh: pierced at %+.1f -> %.0f%% open, %d verts, open edges %d,"
              " non-manifold %d, oriented %s"
              % (thr, 100 * (1 - pm.mean()), len(vv), op, nm, ori))
        ok = ok and op == 0 and nm == 0 and ori

    # A field with fine structure is where diagonal pinches appear, and they
    # are the failure that LOOKS closed: no unpaired edge, but an edge shared
    # by four faces, which is not a solid.  Noise makes plenty of them.
    rngp = np.random.default_rng(9)
    speckle = np.asarray(
        [[float(v) for v in row] for row in rngp.normal(size=Xp.shape)])
    raw = np.ones(Xp.shape, dtype=bool) & (speckle > 0.0)
    pinched = int(_count_pinches(raw))
    fixed = pierce(full, speckle, threshold=0.0, min_island=1)
    _, nm2, _ = edge_report(slab(Xp, Yp, 0.1 * speckle, fixed, 0.1)[1])
    print("mesh: speckle mask had %d diagonal pinches; after pierce %d "
          "non-manifold edges" % (pinched, nm2))
    ok = ok and pinched > 0 and nm2 == 0

    # Pinholes are discarded: a threshold near a noisy field's mean opens a
    # scatter of one-sample holes that nothing can manufacture.
    rng = np.random.default_rng(4)
    noise = rng.normal(size=Xp.shape)
    loose = pierce(full, noise, threshold=0.0, min_island=1)
    tidy = pierce(full, noise, threshold=0.0, min_island=12)
    print("mesh: min_island 1 -> %.1f%% open, 12 -> %.1f%% open"
          % (100 * (1 - loose.mean()), 100 * (1 - tidy.mean())))
    ok = ok and (1 - tidy.mean()) < (1 - loose.mean())

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
