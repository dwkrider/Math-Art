# Marking fold lines sharp, for generators whose surfaces are creased.
#
# A shared helper, not a generator: it registers nothing and is imported
# directly by the modules that need it (`dform_generator`,
# `squeeze_generator`, `vortex_generator`, `bubble_generator`,
# `cmc_generator`, ...), the same way `pattern_common` is used.
#
# WHY IT EXISTS.  Several generators here build surfaces that are smooth
# in patches but genuinely folded where the patches meet -- a D-form's
# seam, the bent frames of a squeeze, the spokes of a vertex vortex.
# Those creases are the shape: they are where the Gaussian curvature is
# concentrated, and the patches between them are developable or minimal
# and want smooth shading.  Shading the fold smooth as well rounds it
# into a soft bulge and loses exactly the feature the sculpture is about.
#
# Blender has done this the same way since 4.1: the `sharp_edge` boolean
# attribute drives split normals directly, so a mesh can be shaded
# smooth everywhere and still show a crisp fold, with no auto-smooth
# setting and no Edge Split modifier.  The matching `crease_edge` float
# is what a Subdivision Surface modifier reads, so both are set -- a
# creased edge stays a fold when the user subdivides, which is the
# common next step for these meshes.

def mark_sharp(me, edges, crease=True):
    """Mark `edges` (pairs of vertex indices) sharp on mesh `me`.

    Pairs may be given in either order and need not exist; only the
    edges the mesh actually has are touched.  Returns how many were
    marked, which is worth checking in a self-test -- silently marking
    nothing is the failure mode here, and it looks identical to success
    in a render until you go hunting for the crease.
    """
    want = set()
    for a, b in edges:
        a, b = int(a), int(b)
        if a != b:
            want.add((a, b) if a < b else (b, a))
    if not want:
        return 0

    hit = []
    for i, e in enumerate(me.edges):
        a, b = e.vertices
        key = (a, b) if a < b else (b, a)
        if key in want:
            hit.append(i)
    for i in hit:
        me.edges[i].use_edge_sharp = True

    if crease and hit:
        # optional: only Subdivision Surface reads it, and the attribute
        # API has moved around between versions, so never let it break
        # the operator that called us
        try:
            att = me.attributes.get("crease_edge")
            if att is None:
                att = me.attributes.new("crease_edge", 'FLOAT', 'EDGE')
            for i in hit:
                att.data[i].value = 1.0
        except (RuntimeError, TypeError, AttributeError, KeyError):
            pass
    return len(hit)


def mark_sharp_by_angle(me, degrees=35.0, crease=True):
    """Mark every edge where the surface genuinely folds.

    For generators whose creases are known at build time -- a D-form's
    seam, a squeeze's bent frames -- pass the edges to `mark_sharp`
    instead; that is exact and cannot be fooled.  This is for the solids
    whose folds are a consequence of the construction rather than an
    input: the bicylinder's two intersection ellipses, a sphericon's
    turned rims, a Meissner body's ground edges, the Gomboc's ridge.
    Those are all strong dihedral discontinuities on an otherwise smooth
    surface, so an angle test finds exactly them and nothing else, and
    it does not care how the mesh was generated.

    Reads polygon normals directly rather than going through bmesh: the
    crease attribute has to be written by edge index afterwards, and a
    bmesh round trip is free to renumber.
    """
    import math

    if not me.polygons or not me.edges:
        return 0
    ef = {}
    for p in me.polygons:
        for k in p.edge_keys:
            ef.setdefault(k, []).append(p.index)

    thr = math.cos(math.radians(max(0.0, min(179.0, float(degrees)))))
    hit = []
    for i, e in enumerate(me.edges):
        fs = ef.get(e.key)
        if not fs or len(fs) != 2:
            continue
        n1 = me.polygons[fs[0]].normal
        n2 = me.polygons[fs[1]].normal
        if n1.dot(n2) < thr:
            hit.append(i)
    for i in hit:
        me.edges[i].use_edge_sharp = True

    if crease and hit:
        try:
            att = me.attributes.get("crease_edge")
            if att is None:
                att = me.attributes.new("crease_edge", 'FLOAT', 'EDGE')
            for i in hit:
                att.data[i].value = 1.0
        except (RuntimeError, TypeError, AttributeError, KeyError):
            pass
    return len(hit)


def label_junction_edges(tris, labels):
    """Edges where faces carrying DIFFERENT region labels meet.

    `tris` are triangles (or any polygons) as vertex-index tuples and
    `labels` is one `(front, back)` region pair per face -- the
    labeling the volume-constrained evolver uses.  An edge whose
    incident faces do not all carry the same pair is exactly a film
    junction: a Plateau border, a triple line, or a cap-film seam.
    This is derived from the topology the solver itself maintains, so
    unlike a dihedral-angle test it cannot be fooled by a junction
    that happens to be flat (an equal double bubble's planar film
    meets the caps at 120 degrees, but a film-film junction stays a
    junction at ANY angle).

    Pairs are compared unordered, so a film is one film regardless of
    which side the face normal calls "front".  Returns the junction
    edges as (a, b) vertex-index pairs, ready for `mark_sharp`.
    Non-manifold edges (three films on one edge -- the generic
    Plateau border) are handled the same way: any edge with two or
    more distinct incident labels is a junction.
    """
    ed = {}
    for t, lab in zip(tris, labels):
        f, b = int(lab[0]), int(lab[1])
        lk = (f, b) if f <= b else (b, f)
        n = len(t)
        for k in range(n):
            a, c = int(t[k]), int(t[(k + 1) % n])
            e = (a, c) if a < c else (c, a)
            s = ed.get(e)
            if s is None:
                ed[e] = {lk}
            else:
                s.add(lk)
    return [e for e, ls in ed.items() if len(ls) > 1]


def boundary_edges(tris):
    """Edges used by exactly one face: the open boundary of the mesh.

    For an open surface (a liquid bridge between pinned rims, a
    sessile drop whose wetted disk is not meshed, a free-boundary
    film) the physical crease lines -- pinned rims, contact lines --
    ARE the mesh boundary.  A boundary edge has only one face, so
    `sharp_edge` cannot split any normals there; what matters is the
    `crease_edge` weight `mark_sharp` sets alongside it, which stops
    a Subdivision Surface modifier from shrinking the open boundary
    away from the ring / floor it is pinned to.  Returns (a, b)
    vertex-index pairs, ready for `mark_sharp`.
    """
    cnt = {}
    for t in tris:
        n = len(t)
        for k in range(n):
            a, c = int(t[k]), int(t[(k + 1) % n])
            e = (a, c) if a < c else (c, a)
            cnt[e] = cnt.get(e, 0) + 1
    return [e for e, c in cnt.items() if c == 1]


def ring_edges(loop, closed=True):
    """Consecutive index pairs along a polyline (or closed ring)."""
    idx = [int(i) for i in loop]
    n = len(idx)
    if n < 2:
        return []
    out = [(idx[i], idx[i + 1]) for i in range(n - 1)]
    if closed:
        out.append((idx[-1], idx[0]))
    return out


def _selftest():
    ok = True

    good = ring_edges([0, 1, 2, 3]) == [(0, 1), (1, 2), (2, 3), (3, 0)]
    ok &= good
    print(f"sharp_creases: closed ring edges {'OK' if good else 'FAIL'}")

    good = ring_edges([0, 1, 2], closed=False) == [(0, 1), (1, 2)]
    ok &= good
    print(f"sharp_creases: open polyline edges {'OK' if good else 'FAIL'}")

    good = ring_edges([5]) == [] and ring_edges([]) == []
    ok &= good
    print(f"sharp_creases: degenerate loops give no edges "
          f"{'OK' if good else 'FAIL'}")

    # `mark_sharp` needs a real mesh, so it is exercised by the Blender
    # suites; here just confirm the no-op path cannot raise
    good = mark_sharp(None, []) == 0
    ok &= good
    print(f"sharp_creases: empty edge list is a no-op "
          f"{'OK' if good else 'FAIL'}")

    # label junctions: two films sharing edge (1, 2) -> that edge and
    # only that edge; same labels (in either orientation) -> none
    T2 = [(0, 1, 2), (2, 1, 3)]
    good = label_junction_edges(T2, [(0, 1), (0, 2)]) == [(1, 2)]
    good &= label_junction_edges(T2, [(0, 1), (1, 0)]) == []
    good &= label_junction_edges(T2, [(0, 1), (0, 1)]) == []
    ok &= good
    print(f"sharp_creases: label junction edges "
          f"{'OK' if good else 'FAIL'}")

    # non-manifold Plateau border: three films on one edge
    T3 = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
    good = label_junction_edges(T3, [(0, 1), (0, 2), (1, 2)]) \
        == [(0, 1)]
    ok &= good
    print(f"sharp_creases: non-manifold junction edge "
          f"{'OK' if good else 'FAIL'}")

    # boundary edges: lone triangle has 3; a shared edge is interior
    good = sorted(boundary_edges([(0, 1, 2)])) \
        == [(0, 1), (0, 2), (1, 2)]
    good &= sorted(boundary_edges(T2)) \
        == [(0, 1), (0, 2), (1, 3), (2, 3)]
    ok &= good
    print(f"sharp_creases: boundary edges "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("sharp_creases self-test failed")
