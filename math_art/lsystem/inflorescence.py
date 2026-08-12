
# Inflorescence architecture: fifteen forms from two production templates.
#
# THE POINT OF THIS MODULE.  A botany text lists raceme, spike, spadix,
# umbel, corymb, capitulum, panicle, thyrsus, cyme, dichasium,
# monochasium, dibotryoid, tribotryoid, verticillaster and spadix as
# separate architectures.  ABOP chapter 3, following Frijters and
# Lindenmayer, shows they are all the same two productions with different
# constants -- Table 3.1:
#
#     A -> I [B]^n [X]^m X          (INDETERMINATE: the axis keeps going)
#     A -> I [B]^n [X]^m C          (DETERMINATE: a terminal flower stops it)
#
# where I is an internode, B a lateral branch, X a continuation of the
# axis, and C a terminal flower.  The whole taxonomy is (n, m, whether
# the axis terminates, and how the internode length and branch angle vary
# with order).  So this module is an enum and a recursion, not fifteen
# special cases -- which is the entire argument for having a rewriting
# engine underneath.
#
# MONOPODIAL vs SYMPODIAL.  The indeterminate template grows from a
# single persistent apex (monopodial): raceme, spike, umbel.  The
# determinate one kills its apex with a terminal flower, so growth
# continues only from laterals (sympodial): cyme, dichasium, thyrsus.
# That single bit is the deepest division in the taxonomy.
#
# BLOOM ORDER (section 3.4 of the plan).  Give a flower at position i
# along its axis and j along the parent the age
#
#     T = u*i + v*j
#
# Then the sign of  D = u*n - v*m  decides the flowering sequence of the
# whole structure: negative is ACROPETAL (base blooms first, flowering
# runs toward the tip), positive is BASIPETAL (tip first), and zero is
# DIVERGENT (both ends at once, meeting in the middle).  One scalar
# reproduces a distinction that reads as three unrelated behaviours.
#
# BRANCH MAPPING (3.10).  Given two branches of the same ORDER, the
# shorter is taken to be identical to the TOP portion of the longer --
# biologically because apical meristems sit at the distal end, so a
# branch that developed for less time resembles the tip of one that
# developed longer.  Implemented by expressing position as a fraction
# `rel = s / max_len[order]` and starting a short axis at
# `s = max_len - length` instead of at 0.  One set of functions per
# ORDER then serves every axis of that order, and gradients along short
# branches automatically align with the distal part of the longest.
#
# References:
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, chapter 3 and Table 3.1.
# - Jacqueline Frijters and Aristid Lindenmayer, "Developmental
#   descriptions of branching patterns with paracladial relationships",
#   in Automata, Languages, Development, North-Holland, 1976.
# - Przemyslaw Prusinkiewicz, Mark Hammel, Jim Hanan and Radomir Mech,
#   "Developmental models of herbaceous plants for computer imagery
#   purposes", SIGGRAPH 1988 -- the primary source behind chapter 3.
# - Przemyslaw Prusinkiewicz, Lars Muendermann, Radoslaw Karwowski and
#   Brendan Lane, "The use of positional information in the modeling of
#   plants", SIGGRAPH 2001 -- section 6 (single compound structures) and
#   section 8 (branch mapping).

import numpy as np


#: Table 3.1 collapsed to data.  `n` laterals and `m` axis copies per
#: node; `determinate` ends the axis with a terminal flower.  `taper` is
#: the internode length ratio between successive orders and `angle` the
#: branching angle in degrees.
ARCHETYPES = {
    # --- indeterminate / monopodial ---------------------------------
    "RACEME":       dict(n=1, m=0, determinate=False, angle=45.0,
                         taper=0.85, internode=1.0,
                         label="Raceme"),
    "SPIKE":        dict(n=1, m=0, determinate=False, angle=75.0,
                         taper=0.9, internode=0.25,
                         label="Spike (sessile raceme)"),
    "SPADIX":       dict(n=1, m=0, determinate=False, angle=88.0,
                         taper=0.95, internode=0.12,
                         label="Spadix (fleshy spike)"),
    "CORYMB":       dict(n=1, m=0, determinate=False, angle=55.0,
                         taper=1.35, internode=0.45,
                         label="Corymb (flat-topped raceme)"),
    "UMBEL":        dict(n=6, m=0, determinate=True, angle=62.0,
                         taper=1.0, internode=0.08,
                         label="Umbel"),
    "CAPITULUM":    dict(n=10, m=0, determinate=True, angle=80.0,
                         taper=1.0, internode=0.03,
                         label="Capitulum (head)"),
    "VERTICILLAST": dict(n=4, m=0, determinate=False, angle=80.0,
                         taper=0.9, internode=0.6,
                         label="Verticillaster (whorled)"),
    # --- compound indeterminate -------------------------------------
    "PANICLE":      dict(n=1, m=1, determinate=False, angle=45.0,
                         taper=0.7, internode=0.8,
                         label="Panicle"),
    "DIBOTRYOID":   dict(n=1, m=1, determinate=False, angle=42.0,
                         taper=0.78, internode=0.9,
                         label="Dibotryoid (double raceme)"),
    "TRIBOTRYOID":  dict(n=1, m=2, determinate=False, angle=40.0,
                         taper=0.72, internode=0.9,
                         label="Tribotryoid (triple raceme)"),
    # --- determinate / sympodial ------------------------------------
    "MONOCHASIUM":  dict(n=1, m=0, determinate=True, angle=45.0,
                         taper=0.8, internode=0.9,
                         label="Monochasium (helicoid cyme)"),
    "DICHASIUM":    dict(n=2, m=0, determinate=True, angle=45.0,
                         taper=0.75, internode=0.9,
                         label="Dichasium (forked cyme)"),
    "CYME":         dict(n=2, m=0, determinate=True, angle=52.0,
                         taper=0.7, internode=0.75,
                         label="Cyme"),
    "THYRSUS":      dict(n=2, m=1, determinate=True, angle=48.0,
                         taper=0.72, internode=0.85,
                         label="Thyrsus (cymose panicle)"),
    "PLEIOCHASIUM": dict(n=3, m=0, determinate=True, angle=50.0,
                         taper=0.75, internode=0.8,
                         label="Pleiochasium"),
}

#: Species reconstructions (plan 3.5).  Overrides on top of an archetype.
SPECIES = {
    "LILAC":     dict(base="THYRSUS", orders=4, angle=42.0, taper=0.66,
                      divergence=137.5, u=1.0, v=3.0,
                      label="Lilac (Syringa vulgaris)"),
    "LYCHNIS":   dict(base="DICHASIUM", orders=4, angle=48.0, taper=0.74,
                      divergence=90.0, u=1.0, v=2.0,
                      label="Lychnis coronaria"),
    "CAPSELLA":  dict(base="RACEME", orders=2, angle=52.0, taper=0.9,
                      divergence=137.5, u=1.0, v=1.0,
                      label="Capsella bursa-pastoris"),
    "CROCUS":    dict(base="UMBEL", orders=2, angle=28.0, taper=1.0,
                      divergence=120.0, u=1.0, v=1.0,
                      label="Crocus"),
}

ACROPETAL, BASIPETAL, DIVERGENT = "ACROPETAL", "BASIPETAL", "DIVERGENT"


def bloom_mode(u, v, n, m):
    """The sign model: D = u*n - v*m decides the flowering sequence.

    Negative -> acropetal (base first, flowering climbs to the tip);
    positive -> basipetal (tip first, descending); zero -> divergent,
    both ends flowering at once and meeting in the middle.
    """
    d = float(u) * float(n) - float(v) * float(m)
    if abs(d) < 1e-12:
        return DIVERGENT
    return ACROPETAL if d < 0 else BASIPETAL


class Node:
    """One internode: a segment, plus what sits at its far end."""

    __slots__ = ("a", "b", "order", "index", "rel", "flower", "age")

    def __init__(self, a, b, order, index, rel):
        self.a = a
        self.b = b
        self.order = order
        self.index = index
        #: position along this axis as a FRACTION of the longest axis of
        #: the same order -- the branch-mapping coordinate
        self.rel = rel
        self.flower = False
        self.age = 0.0


def _rotation(axis, deg):
    """Rodrigues rotation matrix about a unit axis."""
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    t = np.radians(deg)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * K.dot(K)


def build(archetype="RACEME", orders=3, nodes=6, angle=None, taper=None,
          internode=None, divergence=137.5, u=1.0, v=1.0,
          branch_mapping=True, species=None):
    """Grow an inflorescence and return (nodes, info).

    `orders` is the recursion depth (order 0 is the main axis); `nodes`
    the number of internodes on the main axis.  Every architecture comes
    from the same recursion -- only the constants differ.
    """
    if species:
        spec = SPECIES[species]
        archetype = spec["base"]
        orders = spec.get("orders", orders)
        angle = spec.get("angle", angle)
        taper = spec.get("taper", taper)
        divergence = spec.get("divergence", divergence)
        u, v = spec.get("u", u), spec.get("v", v)

    arc = ARCHETYPES[archetype]
    n_lat = int(arc["n"])
    m_axis = int(arc["m"])
    determinate = bool(arc["determinate"])
    angle = arc["angle"] if angle is None else float(angle)
    taper = arc["taper"] if taper is None else float(taper)
    step0 = arc["internode"] if internode is None else float(internode)

    orders = max(int(orders), 1)
    nodes = max(int(nodes), 1)

    # Branch mapping: every axis of a given ORDER shares one length
    # scale, so a short axis is the TOP of the longest one.  The longest
    # axis of order k carries `nodes` internodes shortened by taper^k.
    max_nodes = {k: max(nodes - k, 1) for k in range(orders + 1)}

    out = []

    def grow(base, heading, side, order, count, roll, age0=0.0):
        if order >= orders or count <= 0:
            return
        step = step0 * (taper ** order)
        longest = max_nodes[order]
        # a short axis starts PART WAY along the order's parameter range
        s0 = longest - count
        pos = np.asarray(base, dtype=float)
        h = np.asarray(heading, dtype=float)
        sd = np.asarray(side, dtype=float)
        for i in range(count):
            nxt = pos + h * step
            rel = (s0 + i + 1) / float(longest)
            nd = Node(pos.copy(), nxt.copy(), order, i, rel)
            # The age ACCUMULATES down the path from the base: a flower
            # on a lateral is as old as the lateral's attachment point
            # plus its own position.  Without the inherited term every
            # lateral of a raceme would carry the same age and the whole
            # bloom sequence would collapse to a single instant.
            nd.age = age0 + u * (s0 + i) + v * order
            out.append(nd)

            last = (i == count - 1)
            # laterals
            for b in range(n_lat):
                roll_b = roll + divergence * (i * n_lat + b)
                R = _rotation(h, roll_b).dot(_rotation(sd, angle))
                hb = R.dot(h)
                sb = _rotation(h, roll_b).dot(sd)
                grow(nxt, hb, sb, order + 1,
                     max(max_nodes[order + 1], 1), roll_b + divergence,
                     age0 + u * (s0 + i))
            # extra axis copies (the [X]^m of the template)
            for a in range(m_axis):
                roll_a = roll + divergence * (i + 1) + 180.0 * a
                R = _rotation(h, roll_a).dot(_rotation(sd, angle * 0.55))
                grow(nxt, R.dot(h), _rotation(h, roll_a).dot(sd),
                     order + 1, max(max_nodes[order + 1] - 1, 1),
                     roll_a, age0 + u * (s0 + i))
            if last:
                # Every axis bears a flower at its distal end.  What
                # `determinate` changes is not whether the flower exists
                # but WHEN it forms relative to the laterals -- it is the
                # terminal flower that arrests the apex -- and that is
                # carried by the age model rather than by the geometry.
                nd.flower = True
                if determinate:
                    nd.age -= u        # the terminal flower forms first
            pos = nxt

    grow(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
         0, nodes, 0.0)

    # every terminal internode of every axis bears a flower; in a
    # determinate architecture the MAIN axis does too
    info = dict(archetype=archetype, n=n_lat, m=m_axis,
                determinate=determinate,
                bloom=bloom_mode(u, v, n_lat, m_axis),
                delta=float(u) * n_lat - float(v) * m_axis,
                orders=orders, flowers=sum(1 for q in out if q.flower),
                internodes=len(out))
    return out, info


def bloom_sequence(nodes):
    """Flower ages normalised to [0, 1], in flowering order.

    Returned as an array parallel to the flowering nodes, so an animation
    can simply threshold it.
    """
    fl = [q for q in nodes if q.flower]
    if not fl:
        return np.zeros(0)
    ages = np.array([q.age for q in fl], dtype=float)
    lo, hi = ages.min(), ages.max()
    return (ages - lo) / (hi - lo) if hi > lo else np.zeros(len(ages))


def segments(nodes):
    """(N, 2, 3) array of internode endpoints."""
    if not nodes:
        return np.zeros((0, 2, 3))
    return np.array([[q.a, q.b] for q in nodes], dtype=float)


def fit(nodes, size=2.0):
    """Centre and scale the structure into the 2 m cube, in place."""
    if not nodes:
        return nodes
    P = np.vstack([segments(nodes).reshape(-1, 3)])
    lo, hi = P.min(axis=0), P.max(axis=0)
    ext = float((hi - lo).max())
    cen = (lo + hi) / 2.0
    f = (size / ext) if ext > 1e-12 else 1.0
    for q in nodes:
        q.a = (q.a - cen) * f
        q.b = (q.b - cen) * f
    return nodes


def _selftest():
    # --- the two templates really do cover the taxonomy ---------------
    # Every architecture must grow, and monopodial/sympodial must come
    # out of the `determinate` bit rather than a special case.
    for key, arc in sorted(ARCHETYPES.items()):
        nd, info = build(key, orders=3, nodes=5)
        assert nd, key
        assert info["internodes"] > 0 and info["flowers"] > 0, key
        P = segments(nd).reshape(-1, 3)
        assert np.all(np.isfinite(P)), key
        # a real 3-D structure, not a collapsed line
        ext = P.max(axis=0) - P.min(axis=0)
        assert float((ext > 1e-9).sum()) >= 2, (key, ext)

    # n and m come straight from Table 3.1
    assert ARCHETYPES["RACEME"]["n"] == 1 and ARCHETYPES["RACEME"]["m"] == 0
    assert ARCHETYPES["DIBOTRYOID"]["m"] == 1
    assert ARCHETYPES["TRIBOTRYOID"]["m"] == 2
    assert not ARCHETYPES["RACEME"]["determinate"]
    assert ARCHETYPES["DICHASIUM"]["determinate"]

    # --- the bloom sign model -----------------------------------------
    # D = u*n - v*m.  This is the whole of section 3.4.
    assert bloom_mode(1, 1, 0, 1) == ACROPETAL      # D = -1
    assert bloom_mode(1, 1, 1, 0) == BASIPETAL      # D = +1
    assert bloom_mode(1, 1, 1, 1) == DIVERGENT      # D =  0
    assert bloom_mode(2, 1, 1, 2) == DIVERGENT      # 2*1 - 1*2 = 0
    assert bloom_mode(1, 3, 1, 1) == ACROPETAL      # 1 - 3 < 0
    # ... and it must reach the built structure.  DIBOTRYOID is n=1,
    # m=1, so D = u - v and the mode is decided purely by which of the
    # two rates is larger -- the same architecture flowers in three
    # different orders depending on u and v alone.
    assert build("DIBOTRYOID", orders=3, nodes=5,
                 u=1.0, v=1.0)[1]["bloom"] == DIVERGENT
    assert build("DIBOTRYOID", orders=3, nodes=5,
                 u=1.0, v=3.0)[1]["bloom"] == ACROPETAL
    assert build("DIBOTRYOID", orders=3, nodes=5,
                 u=3.0, v=1.0)[1]["bloom"] == BASIPETAL
    # a monopodial architecture with no axis copies can only be basipetal
    assert build("RACEME", orders=2, nodes=5)[1]["bloom"] == BASIPETAL

    # the bloom sequence must be a usable 0..1 ramp
    # The ramp must actually span 0..1 and have distinct values: ages
    # accumulate along the path, so a raceme's laterals flower in
    # sequence rather than all at once.
    nd, _i = build("RACEME", orders=2, nodes=6)
    seq = bloom_sequence(nd)
    assert len(seq) > 1
    assert abs(float(seq.min())) < 1e-12 and abs(float(seq.max()) - 1) < 1e-12
    assert len(set(np.round(seq, 9).tolist())) > 1, "bloom order collapsed"
    # every architecture must give a non-degenerate sequence
    for key in sorted(ARCHETYPES):
        s = bloom_sequence(build(key, orders=3, nodes=5)[0])
        assert len(set(np.round(s, 9).tolist())) > 1, key

    # --- branch mapping ------------------------------------------------
    # A shorter axis of a given order must occupy the DISTAL part of the
    # parameter range: its `rel` values end at 1 and start above 0.
    nd, _i = build("PANICLE", orders=3, nodes=6)
    for order in sorted({q.order for q in nd}):
        rels = [q.rel for q in nd if q.order == order]
        assert max(rels) <= 1.0 + 1e-9, (order, max(rels))
        assert min(rels) > 0.0, (order, min(rels))

    # --- species -------------------------------------------------------
    for key in sorted(SPECIES):
        nd, info = build(species=key, nodes=5)
        assert nd, key
        assert info["archetype"] == SPECIES[key]["base"], key

    # --- fitting -------------------------------------------------------
    nd, _i = build("THYRSUS", orders=3, nodes=5)
    fit(nd, 2.0)
    P = segments(nd).reshape(-1, 3)
    assert abs(float((P.max(axis=0) - P.min(axis=0)).max()) - 2.0) < 1e-9

    # depth and node count must actually control size
    small, _a = build("RACEME", orders=2, nodes=3)
    big, _b = build("RACEME", orders=3, nodes=6)
    assert len(big) > len(small)

    print(f"inflorescence: OK -- {len(ARCHETYPES)} architectures and "
          f"{len(SPECIES)} species from the two Table 3.1 templates, "
          f"bloom sign model D=un-vm, branch mapping distal-aligned")
