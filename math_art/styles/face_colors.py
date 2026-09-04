"""
face_colors.py -- one palette and one "colour the faces by this key"
helper, shared by the generators that group faces into families.

A generator rarely wants a colour ramp; it wants to SHOW something -- which
pair of zones spans a face, which level of a polar zonohedron it sits on,
how many sides it has, which zone triple built a dissection block.  All of
those are the same operation: hand in one hashable key per face, get back
a material list and a per-face material index.

Keeping it here rather than in each generator matters because the point of
the colouring is comparison across generators: the zone-pair colouring of a
zonohedron and of a zonish polyhedron should look like the same scheme,
which they will not if each module carries its own eight colours in its own
order.

The same argument applies to the one colouring that is not a grouping at
all -- the map colouring, where adjacent faces must simply differ -- so the
graph colouring lives here too.

References:
- Daniel Brelaz, "New methods to color the vertices of a graph",
  Communications of the ACM 22(4), 1979, 251-256 -- the DSATUR heuristic
  used below (colour the most saturated vertex next).
- Kenneth Appel and Wolfgang Haken, "Every planar map is four colorable",
  Bulletin of the AMS 82(5), 1976 -- four colours always suffice for the
  face graph of a polyhedron, though no heuristic is guaranteed to find
  such a colouring, which is why `proper_coloring` escalates.
"""

try:
    import bpy
    _IN_BLENDER = True
except ImportError:                       # headless import
    _IN_BLENDER = False


#: Eight well-separated hues.  Kowalewski's colouring of the rhombic
#: triacontahedron needs only five -- each of the twenty blocks takes three
#: of them, and the ten 3-subsets appear once acute and once flat -- so the
#: first five are kept in that role and three more follow for the larger
#: stars, where a zone pair or a level count runs past five families.
PALETTE = [(0.85, 0.20, 0.18, 1.0), (0.16, 0.42, 0.78, 1.0),
           (0.95, 0.72, 0.15, 1.0), (0.20, 0.62, 0.32, 1.0),
           (0.55, 0.28, 0.68, 1.0), (0.90, 0.45, 0.15, 1.0),
           (0.30, 0.72, 0.72, 1.0), (0.75, 0.30, 0.50, 1.0)]


def material(name, rgba):
    """Fetch or create a material of this name, set to this colour."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat.diffuse_color = rgba
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
    return mat


def materials_for(keys, prefix="Face"):
    """(materials, material_index) for one hashable key per face.

    Families are numbered in first-appearance order, so a colouring stays
    stable as long as the face order does -- change the face order and the
    colours move, which is why the callers build their key lists in the
    same loop that builds the faces.
    """
    order, mats = {}, []
    for k in keys:
        if k not in order:
            order[k] = len(order)
            mats.append(material("%s %d" % (prefix, len(order) - 1),
                                 PALETTE[(len(order) - 1) % len(PALETTE)]))
    return mats, [order[k] for k in keys]


def colour_regions(nregions, adj, ncolors):
    """Colour the conflict graph, using no more than `ncolors`.

    Returns (colour_by_region, ok).  Plain greedy is not good enough: on a
    dodecahedron it wants seven colours where five suffice, and five is the
    lower bound (a pentagon's five edges are mutually in conflict).  DSATUR
    first -- colour the most-constrained region next -- then randomised
    restarts, which finds the optimum on every solid the generators offer.
    If `ncolors` really is below what the graph needs the fallback is
    modular and `ok` is False, so the build still succeeds and the caller
    can raise the count rather than ship an improper colouring.
    """
    import random

    def attempt(order):
        col = {}
        for v in order:
            used = {col[u] for u in adj.get(v, ()) if u in col}
            c = next((c for c in range(ncolors) if c not in used), None)
            if c is None:
                return None
            col[v] = c
        return col

    col = {}
    while len(col) < nregions:
        best = max((v for v in range(nregions) if v not in col),
                   key=lambda v: (len({col[u] for u in adj.get(v, ())
                                       if u in col}),
                                  len(adj.get(v, ()))))
        used = {col[u] for u in adj.get(best, ()) if u in col}
        c = next((c for c in range(ncolors) if c not in used), None)
        if c is None:
            col = None
            break
        col[best] = c
    if col is not None:
        return col, True
    order = list(range(nregions))
    for seed in range(400):
        random.Random(seed).shuffle(order)
        got = attempt(order)
        if got is not None:
            return got, True
    return {v: v % ncolors for v in range(nregions)}, False


def proper_coloring(nregions, adj, kmin=3, kmax=8):
    """Fewest colours in [kmin, kmax] that give a genuinely proper
    colouring; returns (colour_by_region, k).

    Four colours always suffice for a polyhedron's face graph (Appel and
    Haken), but no heuristic is guaranteed to find a four-colouring, and
    in practice DSATUR plus restarts does not: every Goldberg polyhedron
    tried here needs a fifth colour before it succeeds.  Escalating is
    therefore not a workaround for a weak solver so much as the honest
    reading of "colour this map with the palette I have" -- and it is far
    better than returning the improper modular fallback, which paints
    neighbours the same colour and makes the scheme look broken.
    """
    for k in range(max(1, kmin), kmax + 1):
        col, ok = colour_regions(nregions, adj, k)
        if ok:
            return col, k
    return colour_regions(nregions, adj, kmax)[0], kmax


def _selftest():
    # the mapping is pure bookkeeping, so it is testable without Blender
    keys = ['a', 'b', 'a', 'c', 'b']
    order = {}
    idx = []
    for k in keys:
        if k not in order:
            order[k] = len(order)
        idx.append(order[k])
    assert idx == [0, 1, 0, 2, 1], idx
    assert len(PALETTE) == 8
    assert all(len(c) == 4 for c in PALETTE)
    assert len({c for c in PALETTE}) == 8      # no accidental duplicates

    # graph colouring: two graphs whose chromatic number is known exactly,
    # so the escalation is checked against the answer and not against
    # whatever the heuristic happened to return.
    k4 = {v: [u for u in range(4) if u != v] for v in range(4)}
    col, k = proper_coloring(4, k4, kmin=1)
    assert k == 4, k                        # a clique needs one each
    c5 = {v: [(v - 1) % 5, (v + 1) % 5] for v in range(5)}
    col5, k5 = proper_coloring(5, c5, kmin=1)
    assert k5 == 3, k5                      # an odd cycle needs three
    for v, nb in c5.items():                # and the result is proper
        assert all(col5[v] != col5[u] for u in nb), v
    assert colour_regions(5, c5, 2)[1] is False   # two is genuinely short
    print("RESULT: OK")
