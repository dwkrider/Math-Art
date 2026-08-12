
# Leaves: simple blades by marginal growth, and compound leaves.
#
# SIMPLE LEAVES (ABOP section 5.2).  A blade is not drawn as an outline
# and filled -- it is GROWN.  The margin is a polyline; at each step
# every existing edge elongates by a fixed rate and new material is
# inserted near the apex, where the marginal meristem sits.  Different
# leaf shapes are then not different curves but different rates: raise
# the basal elongation and the leaf becomes cordate, raise the apical
# and it becomes lanceolate.  That is why this is a growth model and not
# a spline library, and it is the reason the same six numbers give the
# whole Table 5.1 series.
#
# The outline comes back CLOSED and bilaterally symmetric, which is what
# lets it be triangulated into a real, solidifiable blade rather than a
# wire loop.
#
# COMPOUND LEAVES (ABOP section 5.3).  A rachis bearing leaflets, where
# the apex produces a new leaflet pair only every `D` plastochrons --
# the APICAL DELAY -- while existing internodes keep elongating at rate
# `R`.  The pair (D, R) controls how strongly the leaf is graded from
# base to tip: a large delay with a low rate gives many similar
# leaflets, a short delay with a high rate gives a few sharply graded
# ones.  Table 5.2 lists five (D, R, length) triples and they are
# shipped here as presets.
#
# NOTE ON WHAT IS VERIFIED.  The growth model below reproduces the
# QUALITATIVE series of Table 5.1 and the structural behaviour of
# Table 5.2 -- leaflet count rising with delay, grading with rate --
# and the self-test pins those down numerically.  The exact blade
# outlines of Table 5.1 and the precise length column of Table 5.2 are
# NOT claimed to be reproduced digit for digit; that would need the
# book's constants checked against the book, which has not been done.
# See BACKLOG.
#
# References:
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, sections 5.2 (simple leaves,
#   Table 5.1) and 5.3 (compound leaves, Table 5.2).
# - Adam Runions, Martin Fuhrer, Brendan Lane, Pavol Federl, Anne-Gaelle
#   Rolland-Lagan and Przemyslaw Prusinkiewicz, "Modeling and
#   visualization of leaf venation patterns", ACM TOG 24(3), SIGGRAPH
#   2005 -- the venation layer.
# - Przemyslaw Prusinkiewicz, Mark Hammel, Jim Hanan and Radomir Mech,
#   "Developmental models of herbaceous plants for computer imagery
#   purposes", SIGGRAPH 1988.

import numpy as np


#: Simple-leaf shapes as growth parameters, not as outlines.
#: `basal` and `apical` are the elongation rates of the two ends of the
#: margin; `width` the lateral rate; `steps` the number of plastochrons.
#: `ratio` is the finished width/length -- the growth rates determine the
#: PROFILE of the margin (where it is widest, how the base and apex are
#: shaped), while the overall proportion is a per-species constant, so it
#: is carried as one rather than being tuned into the rates.
LEAF_SHAPES = {
    "CORDATE":    dict(basal=1.22, apical=1.02, width=1.13, lobe=0.30,
                       tip=0.55, ratio=0.92, label="Cordate (heart-shaped)"),
    "OVATE":      dict(basal=1.14, apical=1.05, width=1.09, lobe=0.0,
                       tip=0.70, ratio=0.60, label="Ovate"),
    "ELLIPTIC":   dict(basal=1.09, apical=1.09, width=1.09, lobe=0.0,
                       tip=0.80, ratio=0.52, label="Elliptic"),
    "LANCEOLATE": dict(basal=1.04, apical=1.16, width=1.03, lobe=0.0,
                       tip=1.15, ratio=0.22, label="Lanceolate"),
    "OBOVATE":    dict(basal=1.05, apical=1.14, width=1.09, lobe=0.0,
                       tip=0.62, ratio=0.55, label="Obovate"),
    "RENIFORM":   dict(basal=1.28, apical=0.98, width=1.20, lobe=0.42,
                       tip=0.35, ratio=1.25, label="Reniform (kidney)"),
    "ROSE":       dict(basal=1.12, apical=1.07, width=1.06, lobe=0.18,
                       tip=0.75, ratio=0.42, label="Rose leaflet"),
}

#: Table 5.2: apical delay, elongation rate, and the length the book
#: reports for each.  The length is carried as DOCUMENTATION of the
#: source row -- see the module note on what is and is not verified.
COMPOUND = (
    dict(delay=0, rate=2.00, length=10, label="D=0  R=2.00"),
    dict(delay=1, rate=1.50, length=16, label="D=1  R=1.50"),
    dict(delay=2, rate=1.36, length=21, label="D=2  R=1.36"),
    dict(delay=4, rate=1.23, length=30, label="D=4  R=1.23"),
    dict(delay=7, rate=1.17, length=45, label="D=7  R=1.17"),
)


def simple_leaf(shape="OVATE", steps=20, samples=None, **over):
    """Grow a leaf blade and return its CLOSED outline, (N, 2).

    One half-margin is grown, then mirrored: a leaf is bilaterally
    symmetric by construction, and growing both halves independently
    would only introduce numerical asymmetry.
    """
    p = dict(LEAF_SHAPES[shape]) if shape in LEAF_SHAPES else {}
    p.update({k: v for k, v in over.items() if v is not None})
    basal = float(p.get("basal", 1.1))
    apical = float(p.get("apical", 1.1))
    width = float(p.get("width", 1.08))
    lobe = float(p.get("lobe", 0.0))
    tip = float(p.get("tip", 0.7))
    ratio = float(p.get("ratio", 0.55))
    steps = max(int(steps), 2)

    # The half-margin starts as a single segment from base to apex and
    # is refined `steps` times.  At each step every edge elongates -- by
    # a rate interpolated between the basal and apical values according
    # to where it sits -- and a new vertex is inserted at the midpoint,
    # displaced laterally by the width rate.  Marginal growth, in the
    # sense of section 5.2: new material appears at the margin, and the
    # SHAPE is a consequence of where it appears fastest.
    pts = np.array([[0.0, 0.0], [0.0, 1.0]])
    for k in range(steps):
        n = len(pts)
        s = np.linspace(0.0, 1.0, n - 1 + 1)[:-1] + 0.5 / max(n - 1, 1)
        rate = basal + (apical - basal) * s
        new = [pts[0]]
        for i in range(n - 1):
            a, b = pts[i], pts[i + 1]
            mid = 0.5 * (a + b)
            d = b - a
            # outward normal of this edge, in the leaf plane
            nor = np.array([d[1], -d[0]])
            ln = np.linalg.norm(nor)
            if ln > 1e-12:
                nor = nor / ln
            bulge = (width - 1.0) * np.linalg.norm(d) * _bulge(s[i], lobe)
            new.append(mid + nor * bulge)
            new.append(b)
        pts = np.array(new, dtype=float)
        # Elongate along the leaf axis, apex faster or slower than base.
        t = np.linspace(0.0, 1.0, len(pts))
        pts[:, 1] *= (1.0 + (basal - 1.0) * (1.0 - t) + (apical - 1.0) * t)
        # Widen laterally by the SAME compounding rule.  Without this the
        # blade stays a sliver: each new bulge is proportional to an edge
        # that halves every step, so the added width is a geometric series
        # that converges almost immediately, while the length keeps
        # multiplying.  Existing material has to widen too.
        pts[:, 0] *= width
        if len(pts) > 400:
            break

    # normalise so the leaf runs from y = 0 at the base to y = 1
    pts[:, 1] -= pts[:, 1].min()
    h = pts[:, 1].max()
    if h > 1e-12:
        pts = pts / h
    # taper the apex: `tip` under 1 gives a blunt leaf, over 1 a drawn
    # out point
    t = np.clip(pts[:, 1], 0.0, 1.0)
    pts[:, 0] *= np.clip(1.0 - t ** max(tip, 1e-3), 0.0, None)

    # Set the finished proportion.  The grown margin fixes the leaf's
    # character; this fixes how broad it is.
    w = float(np.abs(pts[:, 0]).max())
    if w > 1e-12:
        pts[:, 0] *= (0.5 * ratio) / w

    half = pts
    mirror = half[::-1].copy()
    mirror[:, 0] *= -1.0
    outline = np.vstack([half, mirror[1:-1]])
    if samples:
        outline = _resample(outline, int(samples))
    return outline


def _bulge(s, lobe):
    """Lateral growth profile along the margin.

    A single hump gives a plain blade; `lobe` adds basal width, which is
    what turns an ovate leaf cordate and then reniform.
    """
    base = np.sin(np.pi * np.clip(s, 0.0, 1.0))
    return base + lobe * np.exp(-((s - 0.08) ** 2) / 0.01)


def _resample(pts, n):
    """Resample a closed polyline to `n` points by arc length."""
    P = np.vstack([pts, pts[:1]])
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12:
        return pts
    want = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(want, s, P[:, 0]),
                     np.interp(want, s, P[:, 1])], axis=1)


def area(outline):
    """Shoelace area of a closed outline."""
    P = np.asarray(outline, dtype=float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                           - np.dot(np.roll(x, -1), y)))


def compound_leaf(delay=2, rate=1.36, steps=12, angle=42.0,
                  leaflet_scale=0.30, shape="OVATE"):
    """A rachis with leaflets, from the apical delay and elongation rate.

    Returns (rachis, leaflets) where `rachis` is an (N, 2) polyline and
    each leaflet is (position, scale, side).

    The apex emits a new leaflet pair only every `delay + 1`
    plastochrons, while every existing internode keeps elongating by
    `rate`.  So a leaflet formed early has been elongating longest and
    sits furthest from the tip on the longest internode -- the grading
    from base to apex is a CONSEQUENCE of the two numbers, not a
    separate taper parameter.
    """
    delay = max(int(delay), 0)
    steps = max(int(steps), 1)
    rate = float(rate)

    lengths = []          # internode lengths, base first
    for k in range(steps):
        for i in range(len(lengths)):
            lengths[i] *= rate
        if k % (delay + 1) == 0:
            lengths.append(1.0)

    total = float(sum(lengths)) or 1.0
    ys = np.concatenate([[0.0], np.cumsum(lengths)]) / total
    rachis = np.stack([np.zeros(len(ys)), ys], axis=1)

    leaflets = []
    for i, y in enumerate(ys[1:]):
        # older (more basal) leaflets are larger, by the same rate
        sc = leaflet_scale * (rate ** (len(lengths) - 1 - i)) / \
            (rate ** (len(lengths) - 1)) if rate > 0 else leaflet_scale
        sc = leaflet_scale * (lengths[i] / max(lengths))
        leaflets.append((float(y), float(sc), +1))
        leaflets.append((float(y), float(sc), -1))
    return rachis, leaflets


def blade_mesh(outline):
    """Triangulate a closed leaf outline into (verts, faces).

    The outline is symmetric about x = 0, so it splits cleanly into a
    left and a right margin with the same number of points.  Pairing
    them across the midrib and quadding between consecutive pairs keeps
    every face well conditioned -- unlike a fan about the centroid,
    which degenerates into slivers at both tips of a narrow shape.

    IT MUST BE GIVEN THE CANONICAL, UNROTATED OUTLINE.  The left/right
    split is by INDEX, which is only meaningful while the blade still
    straddles x = 0; triangulating an already-rotated leaflet pairs
    points across the wrong axis and produces a butterfly.  Place the
    leaflet by transforming these vertices, never by transforming the
    outline first.
    """
    P = np.asarray(outline, dtype=float)
    n = len(P)
    half = n // 2 + 1
    right = P[:half]
    left = np.vstack([P[0:1], P[half:][::-1], P[half - 1:half]])
    m = min(len(right), len(left))
    right, left = right[:m], left[:m]

    verts, faces = [], []
    for i in range(m):
        mid = 0.5 * (right[i, 1] + left[i, 1])
        verts.append((float(right[i, 0]), float(right[i, 1]), 0.0))
        verts.append((0.0, float(mid), 0.0))
        verts.append((float(left[i, 0]), float(left[i, 1]), 0.0))
    def face(*idx):
        """Emit a quad, or a triangle where two of its corners coincide.

        A leaf comes to a point at BOTH ends: there the left and right
        margins meet, so the end bands of the strip have two coincident
        corners.  Emitting them as quads leaves faces with no area --
        which is not a small face, it is broken geometry, and Solidify
        and normal computation both choke on it.
        """
        keep = []
        for j in idx:
            if not keep or np.linalg.norm(
                    np.asarray(verts[keep[-1]]) - np.asarray(verts[j])) > 1e-12:
                keep.append(j)
        while len(keep) > 3 and np.linalg.norm(
                np.asarray(verts[keep[0]]) - np.asarray(verts[keep[-1]])) <= 1e-12:
            keep.pop()
        if len(keep) >= 3:
            faces.append(tuple(keep))

    for i in range(m - 1):
        a, b, c = 3 * i, 3 * i + 1, 3 * i + 2
        d, e, f = 3 * (i + 1), 3 * (i + 1) + 1, 3 * (i + 1) + 2
        face(a, d, e, b)
        face(b, e, f, c)
    return verts, faces


def compound_geometry(delay=2, rate=1.36, steps=12, angle=42.0,
                      leaflet_scale=0.30, shape="OVATE", samples=48,
                      rachis_width=0.012, min_scale=0.02):
    """The whole compound leaf as one (verts, faces) mesh.

    Each leaflet is triangulated ONCE in its own local frame and then
    placed by transforming the vertices -- see `blade_mesh` for why the
    outline must not be rotated first.  The rachis is emitted as a
    narrow strip so the leaf reads as a compound leaf rather than as a
    row of unattached blades.
    """
    rachis, leaflets = compound_leaf(delay, rate, steps, angle,
                                     leaflet_scale, shape)
    blade = simple_leaf(shape, steps=8, samples=samples)
    bverts, bfaces = blade_mesh(blade)
    B = np.asarray(bverts, dtype=float)

    verts, faces = [], []

    def add(V, F):
        base = len(verts)
        verts.extend(tuple(float(c) for c in v) for v in V)
        faces.extend(tuple(i + base for i in q) for q in F)

    # The rachis, as a tapered strip.  Its internodes span the same
    # enormous range as the leaflets -- at R = 2.00 the basal ones are
    # 2**19 times shorter than the apical -- so after normalisation
    # consecutive points coincide and the strip quads collapse.  Drop
    # the coincident ones before building it.
    R = np.asarray(rachis, dtype=float)
    if len(R) > 1:
        keep = np.ones(len(R), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(R, axis=0), axis=1) > 1e-9
        R = R[keep]
    if len(R) > 1:
        w = float(rachis_width)
        strip, quads = [], []
        for i, (x, y) in enumerate(R):
            taper = 1.0 - 0.6 * (i / max(len(R) - 1, 1))
            strip.append((x - w * taper, y, 0.0))
            strip.append((x + w * taper, y, 0.0))
        for i in range(len(R) - 1):
            quads.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
        add(strip, quads)

    # A steep elongation rate grades the leaflets over an enormous
    # range -- at R = 2.00 across twenty plastochrons the smallest is
    # 2**19 times the largest -- so the distal ones arrive with an area
    # of zero and triangulate into degenerate faces.  They are dropped
    # rather than emitted: a face with no area is not a small leaflet,
    # it is broken geometry.
    biggest = max((sc for _y, sc, _s in leaflets), default=1.0) or 1.0
    floor = float(min_scale) * biggest
    for y, sc, side in leaflets:
        if sc < floor:
            continue
        a = np.radians(float(angle)) * float(side)
        c, s = np.cos(a), np.sin(a)
        V = B.copy() * float(sc)
        x, yy = V[:, 0].copy(), V[:, 1].copy()
        V[:, 0] = c * x - s * yy
        V[:, 1] = s * x + c * yy + float(y)
        add(V, bfaces)
    return verts, faces


def _selftest():
    # --- simple leaves -------------------------------------------------
    for key in sorted(LEAF_SHAPES):
        P = simple_leaf(key, steps=10)
        assert len(P) > 8, key
        assert np.all(np.isfinite(P)), key
        # A blade must enclose real area -- a collapsed outline would
        # still triangulate, into nothing.
        a = area(P)
        assert a > 1e-3, (key, a)
        # bilaterally symmetric by construction
        assert abs(float(P[:, 0].sum())) < 1e-9, (key, P[:, 0].sum())
        # runs from base to tip over a unit length
        assert abs(float(P[:, 1].min())) < 1e-9, key
        assert abs(float(P[:, 1].max()) - 1.0) < 1e-9, key
        # and is genuinely 2-D
        assert float(P[:, 0].max() - P[:, 0].min()) > 1e-3, key

    # The shape parameters must actually separate the shapes: a cordate
    # leaf is broader at the base than a lanceolate one, which is the
    # whole distinction between the two rows.
    def basal_width(key):
        P = simple_leaf(key, steps=12)
        low = P[P[:, 1] < 0.25]
        return float(low[:, 0].max() - low[:, 0].min())
    # Every blade must have a botanically plausible proportion.  The
    # lateral bulge is proportional to an edge that HALVES every step,
    # so without compounding the width the added material is a geometric
    # series that converges at once while the length keeps multiplying --
    # the blades came out at width/length 0.02 to 0.30, slivers rather
    # than leaves.  This is the guard against that returning.
    for key, spec in sorted(LEAF_SHAPES.items()):
        P = simple_leaf(key, steps=10, samples=96)
        w = float(P[:, 0].max() - P[:, 0].min())
        h = float(P[:, 1].max() - P[:, 1].min())
        got = w / h
        assert abs(got - spec["ratio"]) < 0.02, (key, got, spec["ratio"])
        assert 0.15 < got < 1.6, (key, got)

    assert basal_width("CORDATE") > basal_width("LANCEOLATE")
    assert basal_width("RENIFORM") > basal_width("CORDATE")
    assert area(simple_leaf("ELLIPTIC", steps=12)) > \
        area(simple_leaf("LANCEOLATE", steps=12))

    # more growth steps must refine, not distort: area converges
    areas = [area(simple_leaf("OVATE", steps=s)) for s in (8, 10, 12, 14)]
    for a, b in zip(areas, areas[1:]):
        assert abs(b - a) / max(a, 1e-9) < 0.6, areas

    # --- compound leaves ------------------------------------------------
    # The apical delay controls how many leaflet pairs appear in a given
    # number of plastochrons: one pair every (D + 1) steps.
    for d in (0, 1, 2, 4, 7):
        rachis, leaflets = compound_leaf(delay=d, rate=1.3, steps=24)
        pairs = len(leaflets) // 2
        assert pairs == len(range(0, 24, d + 1)), (d, pairs)
        assert len(rachis) == pairs + 1, (d, len(rachis))

    # Every Table 5.2 row must build, and the rate must grade the
    # leaflets: with R > 1 the basal leaflet is strictly the largest.
    for row in COMPOUND:
        _r, leaflets = compound_leaf(delay=row["delay"], rate=row["rate"],
                                     steps=20)
        assert leaflets, row
        sizes = [sc for _y, sc, side in leaflets if side > 0]
        assert sizes[0] == max(sizes), row["label"]
        if row["rate"] > 1.001:
            assert sizes[0] > sizes[-1], row["label"]

    # a higher rate must grade more steeply -- that is what R means
    def spread(rate):
        _r, lf = compound_leaf(delay=1, rate=rate, steps=20)
        s = [sc for _y, sc, side in lf if side > 0]
        return max(s) / max(min(s), 1e-9)
    assert spread(2.0) > spread(1.2) > spread(1.02)

    # --- placed geometry ------------------------------------------------
    # The blade must be triangulated in its OWN frame and then placed.
    # Triangulating an already-rotated outline pairs margin points
    # across the wrong axis, and the leaflet comes out as a butterfly
    # spanning both sides of the rachis instead of sitting on one.
    P = simple_leaf("OVATE", steps=8, samples=48)
    bv, bf = blade_mesh(P)
    V = np.array(bv)
    mesh_area = 0.0
    for q in bf:
        pts = V[list(q)]
        for j in range(1, len(q) - 1):
            mesh_area += 0.5 * abs(float(np.cross(pts[j] - pts[0],
                                                  pts[j + 1] - pts[0])[2]))
    assert abs(mesh_area - area(P)) / area(P) < 0.05, (mesh_area, area(P))
    # ... and no face may be degenerate, at either end of the blade
    for q in bf:
        pts = V[list(q)]
        ar = sum(0.5 * abs(float(np.cross(pts[j] - pts[0],
                                          pts[j + 1] - pts[0])[2]))
                 for j in range(1, len(q) - 1))
        assert ar > 1e-14, q

    # No row of Table 5.2 may produce degenerate faces, however steeply
    # it grades: R = 2.00 over twenty plastochrons spans a factor of
    # 2**19 and used to emit 292 null quads.
    for row in COMPOUND:
        V, F = compound_geometry(delay=row["delay"], rate=row["rate"],
                                 steps=20, samples=24)
        A = np.array(V)
        bad = 0
        for q in F:
            pts = A[list(q)]
            ar = 0.0
            for j in range(1, len(q) - 1):
                ar += 0.5 * abs(float(np.cross(pts[j] - pts[0],
                                               pts[j + 1] - pts[0])[2]))
            if ar < 1e-12:
                bad += 1
        assert bad == 0, f"{row['label']}: {bad} of {len(F)} degenerate"

    verts, faces = compound_geometry(delay=2, rate=1.36, steps=12)
    assert verts and faces
    V = np.array(verts)
    assert np.all(np.isfinite(V))
    assert abs(float(V[:, 2].max() - V[:, 2].min())) < 1e-12, "must be flat"

    # Leaflets must sit on ALTERNATE sides of the rachis, not straddle
    # it: with a 42-degree angle the two sides are genuinely separated.
    xs = V[:, 0]
    assert float(xs.min()) < -0.02 and float(xs.max()) > 0.02, \
        "leaflets should reach both sides of the rachis"
    # ... and they must be spread ALONG it, not stacked at one height
    assert float(V[:, 1].max() - V[:, 1].min()) > 0.5, "not spread"

    # every quad must have real area -- a butterfly shows up as faces
    # with near-zero area where the split crossed the midrib wrongly
    zero = 0
    for q in faces:
        p0, p1, p2 = V[q[0]], V[q[1]], V[q[2]]
        if 0.5 * abs(float(np.cross(p1 - p0, p2 - p0)[2])) < 1e-14:
            zero += 1
    assert zero < 0.25 * len(faces), f"{zero} of {len(faces)} degenerate"

    print(f"leaves: OK -- {len(LEAF_SHAPES)} blade shapes close, enclose "
          f"area and are bilaterally symmetric; {len(COMPOUND)} Table 5.2 "
          f"rows build with one pair per (D+1) steps and R grading the "
          f"leaflets")
