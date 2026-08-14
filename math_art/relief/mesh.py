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

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
