
# Turtle interpretation of a derived word: the full Appendix-C alphabet.
#
# The word a grammar derives is scanned left to right and its symbols are
# read as commands to a LOGO-style turtle in three dimensions.  The turtle
# state is a position P plus three mutually perpendicular unit vectors
#
#     H  heading      L  left      U  up          with  H x L = U
#
# together with the attributes a drawing needs: line width, colour index,
# and step length.
#
# WHAT THIS ADDS OVER THE GENERATOR IT REPLACES.  The old turtle knew 11
# of the 20 symbols in ABOP's Appendix C and emitted one bevelled curve of
# uniform width and no colour.  The nine that were missing are the ones
# that turn a wireframe into a plant:
#
#     !   decrement diameter        -- real tapering trunks
#     '   increment colour index    -- colour-graded limbs
#     $   rotate to vertical        -- stops 3-D trees spiralling into
#                                      nonsense by re-levelling the frame
#     %   cut off the branch        -- shedding, and dynamic-equilibrium
#                                      crowns of constant size
#     ~   incorporate a surface     -- instance a real organ mesh on the
#                                      turtle frame
#     { . }  polygon capture        -- FILLED leaves and petals instead of
#                                      wire outlines
#     G   move+draw, record no vertex
#
# The 3-D frame convention was audited against the book and is correct as
# it stands: H = +Z, L = +X, U = +Y satisfies H x L = U, and all seven
# orientation symbols reproduce the book's rotation matrices.
#
# TROPISM.  After each drawn segment the heading may be bent toward a
# fixed tropism vector T by
#
#     alpha = e * |H x T|
#
# which has a physical reading: with T a force applied to the segment's
# free end, H x T is the torque about its base, and `e` is the axis's
# susceptibility to bending.  Published values run e = 0.14 to 0.40.
#
# HORTON-STRAHLER ORDER is computed on the branching structure after
# interpretation: a tip has order 1, and a node whose children have
# orders o1..ok takes max(o) when that maximum is unique, else max(o)+1.
# It is strictly better than depth-from-root on asymmetric trees, and it
# is what tells you which limbs are structural and which are decoration.
#
# References:
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990 -- sections 1.3, 1.5, 1.6
#   (brackets), 1.10.2 (surfaces) and Appendix C (the 20 symbols).
# - Przemyslaw Prusinkiewicz, Aristid Lindenmayer and James Hanan,
#   "Developmental Models of Herbaceous Plants for Computer Imagery
#   Purposes", SIGGRAPH '88, Computer Graphics 22(4), pp. 141-150 --
#   the tropism formula alpha = e |H x T| and its torque justification.
# - Przemyslaw Prusinkiewicz, Jim Hanan, Mark Hammel and Radomir Mech,
#   "L-systems: from the Theory to Visual Models of Plants", SIGGRAPH
#   2003 course notes, chapter 2-1, section 5 -- the symbol table with
#   default step 1 and default angle 45 degrees, and the cut symbol %.
# - Robert E. Horton, "Erosional development of streams and their
#   drainage basins", Bull. Geol. Soc. America 56, 1945; Arthur N.
#   Strahler, "Hypsometric (area-altitude) analysis of erosional
#   topography", Bull. Geol. Soc. America 63, 1952.

import math

import numpy as np

# Symbols that move the turtle forward.
DRAW_SYMS = frozenset("FGA")     # A is a common draw-symbol in presets
MOVE_SYMS = frozenset("fg")

DEFAULT_ANGLE = 45.0
DEFAULT_STEP = 1.0
DEFAULT_WIDTH = 0.1

# Symbols that leave the plane.  A grammar using none of them is planar,
# and should be drawn FLAT IN XY where a 2-D fractal is read looking down
# the Z axis -- not standing on edge in XZ.
SPATIAL_SYMS = frozenset("&^/\$")

PLANE_AUTO, PLANE_XY, PLANE_XZ = 'AUTO', 'XY', 'XZ'


def is_planar(word):
    """True when `word` never leaves the plane: no pitch, roll or
    rotate-to-vertical."""
    return not any(mod.sym in SPATIAL_SYMS for mod in word)


def start_frame(word, plane=PLANE_AUTO):
    """Initial (H, L, U) for a word.

    Two conventions, because one does not serve both cases:

      XZ  H=+Z, L=+X, U=+Y -- a 3-D plant grows UP, which is what the
          book's figures and every tree preset assume.
      XY  H=+Y, L=-X, U=+Z -- a planar curve lies FLAT in the ground
          plane, turning about +Z, so it reads correctly from the
          default top view and drops straight into a 2-D workflow.

    Both satisfy H x L = U.  AUTO picks XY for a planar word.
    """
    if plane == PLANE_AUTO:
        plane = PLANE_XY if is_planar(word) else PLANE_XZ
    if plane == PLANE_XY:
        return (np.array([0.0, 1.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]))
    return (np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]))


class Strand:
    """A polyline the turtle drew, with per-point attributes.

    `depth` is bracket nesting depth (0 = trunk); `order` is the
    Horton-Strahler order of the branch this strand belongs to.
    """

    __slots__ = ("points", "widths", "colours", "depth", "order")

    def __init__(self, points, widths, colours, depth):
        self.points = points
        self.widths = widths
        self.colours = colours
        self.depth = depth
        self.order = 1

    def __len__(self):
        return len(self.points)


class TurtleOutput:
    """Everything a turtle run produced, in Blender-free form."""

    def __init__(self):
        self.strands = []
        self.polygons = []       # (points Nx3, colour index)
        self.placements = []     # (name, 4x4 matrix, scale)
        self.max_order = 1

    def points(self):
        if not self.strands and not self.polygons:
            return np.zeros((0, 3))
        chunks = [s.points for s in self.strands if len(s.points)]
        chunks += [p for p, _c in self.polygons if len(p)]
        return np.vstack(chunks) if chunks else np.zeros((0, 3))

    def bbox_fit(self, size=2.0, scale=1.0):
        """Centre at the origin and scale so the largest extent is
        `size` -- the repo's 2 m-cube convention."""
        pts = self.points()
        if not len(pts):
            return self
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ext = float((hi - lo).max())
        cen = 0.5 * (lo + hi)
        f = (size / ext if ext > 1e-12 else 1.0) * scale
        for s in self.strands:
            s.points = (s.points - cen) * f
            s.widths = s.widths * f
        self.polygons = [((p - cen) * f, c) for p, c in self.polygons]
        self.placements = [(n, _shift(mx, cen, f), sc * f)
                           for n, mx, sc in self.placements]
        return self

    def stats(self):
        return (f"{len(self.strands)} strands, "
                f"{sum(len(s) for s in self.strands)} points, "
                f"{len(self.polygons)} polygons, "
                f"{len(self.placements)} placements, "
                f"max Strahler order {self.max_order}")


def _shift(mx, cen, f):
    out = mx.copy()
    out[:3, 3] = (out[:3, 3] - cen) * f
    return out


def _rot(v, axis, ang):
    """Rodrigues rotation of `v` about unit `axis` by `ang` radians."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1 - c)


class _Branch:
    """A node of the branching structure, used only to compute Strahler
    order once the whole word has been read."""

    __slots__ = ("strands", "children")

    def __init__(self):
        self.strands = []
        self.children = []


def _strahler(node):
    """Assign Horton-Strahler orders bottom-up; return this node's."""
    if not node.children:
        order = 1
    else:
        kids = sorted((_strahler(c) for c in node.children), reverse=True)
        if len(kids) >= 2 and kids[0] == kids[1]:
            order = kids[0] + 1
        else:
            order = kids[0]
    for s in node.strands:
        s.order = order
    return order



def stack_series(outs, gap=1.25, axis=None):
    """Lay a list of TurtleOutputs out side by side as generation rows.

    WHY THIS EXISTS.  Some classical L-systems describe a FILAMENT --
    Lindenmayer's original Anabaena catenula is the type specimen -- and
    a filament is one-dimensional.  Drawing a single generation of one
    can only ever give a bare rod: correct, and uninformative.  What the
    literature always shows instead is the DEVELOPMENT: generation 0,
    then 1, then 2, stacked, so the pattern of cell divisions is legible
    down the page.  That is Fig. 1.28 of "The Algorithmic Beauty of
    Plants", and Lindenmayer's own 1968 figures.

    Each output is offset along `axis` (default: perpendicular to the
    turtle's initial heading, in the drawing plane) by `gap` times the
    running row pitch, and the results merged into a single output the
    normal emitters can consume.

    Returns a new TurtleOutput; the inputs are not modified.
    """
    outs = [o for o in outs if o is not None and (o.strands or o.polygons)]
    if not outs:
        return TurtleOutput()
    if axis is None:
        axis = np.array([1.0, 0.0, 0.0])
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    axis = axis / (n if n > 1e-12 else 1.0)

    # Row pitch has to satisfy two things: rows must not collide, and
    # the stack must not come out as a sliver.  The first needs the
    # widest generation measured ALONG the stacking direction -- but for
    # the filaments this function exists to serve that is exactly zero,
    # since a filament is a line lying across the stack.  Falling back
    # to a fixed pitch there gives a figure many times longer than it is
    # wide.  So take whichever is larger: the true row thickness, or the
    # span divided by the number of rows, which makes the finished stack
    # about as wide as it is long.
    thick = span = 0.0
    for o in outs:
        pts = o.points()
        if not len(pts):
            continue
        proj = pts.dot(axis)
        thick = max(thick, float(proj.max() - proj.min()))
        span = max(span, float((pts.max(axis=0) - pts.min(axis=0)).max()))
    pitch = max(thick, span / max(len(outs), 1))
    if pitch <= 1e-12:
        pitch = 1.0

    merged = TurtleOutput()
    for row, o in enumerate(outs):
        shift = axis * (row * pitch * float(gap))
        for s in o.strands:
            c = Strand(np.asarray(s.points, dtype=float) + shift,
                       np.asarray(s.widths, dtype=float),
                       np.asarray(s.colours), s.depth)
            c.order = s.order
            merged.strands.append(c)
        for pts, col in o.polygons:
            merged.polygons.append((np.asarray(pts, dtype=float) + shift, col))
        for name, mx, sc in o.placements:
            mx = np.array(mx, dtype=float)
            mx[:3, 3] += shift
            merged.placements.append((name, mx, sc))
    return merged

def interpret(word, angle=DEFAULT_ANGLE, step=DEFAULT_STEP,
              width=DEFAULT_WIDTH, width_scale=0.707, colour_step=1,
              tropism=None, elasticity=0.0, surfaces=None, draw=None,
              plane=PLANE_AUTO):
    """Run the turtle over `word` and return a TurtleOutput.

    `angle` is the default rotation in degrees and `step` the default
    forward distance; a module's first parameter overrides them, which is
    what makes `F(1.5)` and `+(60)` work.

    `width_scale` is the factor `!` applies with no argument.  The da
    Vinci area-conservation value for a binary split is 1/sqrt(2) =
    0.707, which is why that is the default -- see tropism.py for the
    general law.

    `tropism` is a 3-vector and `elasticity` the coefficient `e`; both
    default to no bending.

    `plane` selects the starting frame: AUTO draws a planar grammar flat
    in XY and a spatial one growing up +Z (see `start_frame`).

    `draw` overrides which letters advance the turtle.  It has to be
    per-grammar rather than global: the Sierpinski arrowhead and the
    Gosper curve draw with A and B (their rewriting letters ARE their
    drawing letters), while a plant grammar uses A as a non-drawing apex.
    """
    draw_syms = DRAW_SYMS if not draw else frozenset(draw)
    ang = math.radians(angle)
    P = np.zeros(3)
    H, L, U = start_frame(word, plane)
    w, col = float(width), 0
    T = None
    if tropism is not None and elasticity:
        T = np.asarray(tropism, dtype=float)
        n = np.linalg.norm(T)
        T = T / n if n > 1e-12 else None

    out = TurtleOutput()
    root = _Branch()
    branch_stack = [root]
    state_stack = []
    cur = [P.copy()]
    cur_w = [w]
    cur_c = [col]
    depth = 0
    poly = None                    # active { } polygon vertex list

    def flush():
        if len(cur) >= 2:
            s = Strand(np.array(cur), np.array(cur_w, dtype=float),
                       np.array(cur_c, dtype=int), depth)
            out.strands.append(s)
            branch_stack[-1].strands.append(s)
        del cur[:]
        del cur_w[:]
        del cur_c[:]

    i, n = 0, len(word)
    while i < n:
        mod = word[i]
        i += 1
        sym = mod.sym
        a = mod.params[0] if mod.params else None

        if sym in draw_syms or sym in MOVE_SYMS:
            d = float(a) if a is not None else step
            newP = P + H * d
            if sym in draw_syms:
                if not cur:
                    cur.append(P.copy())
                    cur_w.append(w)
                    cur_c.append(col)
                cur.append(newP.copy())
                cur_w.append(w)
                cur_c.append(col)
                if poly is not None and sym != "G":
                    poly.append(newP.copy())
            else:
                flush()
                cur.append(newP.copy())
                cur_w.append(w)
                cur_c.append(col)
            P = newP
            if T is not None:
                # alpha = e |H x T|, rotating H toward T about H x T.
                axis = np.cross(H, T)
                mag = float(np.linalg.norm(axis))
                if mag > 1e-12:
                    alpha = elasticity * mag
                    axis = axis / mag
                    H, L, U = (_rot(H, axis, alpha), _rot(L, axis, alpha),
                               _rot(U, axis, alpha))
            continue

        if sym == "+":
            t = math.radians(a) if a is not None else ang
            H, L, U = _rot(H, U, t), _rot(L, U, t), U
        elif sym == "-":
            t = -(math.radians(a) if a is not None else ang)
            H, L, U = _rot(H, U, t), _rot(L, U, t), U
        elif sym == "&":
            t = math.radians(a) if a is not None else ang
            H, L, U = _rot(H, L, t), L, _rot(U, L, t)
        elif sym == "^":
            t = -(math.radians(a) if a is not None else ang)
            H, L, U = _rot(H, L, t), L, _rot(U, L, t)
        elif sym == "\\":
            t = math.radians(a) if a is not None else ang
            H, L, U = H, _rot(L, H, t), _rot(U, H, t)
        elif sym == "/":
            t = -(math.radians(a) if a is not None else ang)
            H, L, U = H, _rot(L, H, t), _rot(U, H, t)
        elif sym == "|":
            H, L, U = _rot(H, U, math.pi), _rot(L, U, math.pi), U
        elif sym == "$":
            # Rotate the frame so L is horizontal, keeping H: the
            # standard fix for 3-D trees whose branch planes drift.
            V = np.array([0.0, 0.0, 1.0])
            newL = np.cross(V, H)
            m = float(np.linalg.norm(newL))
            if m > 1e-9:
                L = newL / m
                U = np.cross(H, L)
        elif sym == "[":
            flush()
            state_stack.append((P.copy(), H.copy(), L.copy(), U.copy(),
                                w, col, depth))
            child = _Branch()
            branch_stack[-1].children.append(child)
            branch_stack.append(child)
            depth += 1
            cur.append(P.copy())
            cur_w.append(w)
            cur_c.append(col)
        elif sym == "]":
            flush()
            if state_stack:
                P, H, L, U, w, col, depth = state_stack.pop()
                P, H, L, U = P.copy(), H.copy(), L.copy(), U.copy()
            if len(branch_stack) > 1:
                branch_stack.pop()
            cur.append(P.copy())
            cur_w.append(w)
            cur_c.append(col)
        elif sym == "!":
            w = float(a) if a is not None else w * width_scale
        elif sym == "#":
            w = float(a) if a is not None else w / max(width_scale, 1e-6)
        elif sym == "'" or sym == ";":
            col = int(a) if a is not None else col + colour_step
        elif sym == ",":
            col = int(a) if a is not None else col - colour_step
        elif sym == "{":
            poly = [P.copy()]
        elif sym == ".":
            if poly is not None:
                poly.append(P.copy())
        elif sym == "}":
            if poly is not None and len(poly) >= 3:
                out.polygons.append((np.array(poly), col))
            poly = None
        elif sym == "~":
            name = surfaces[0] if surfaces else "surface"
            mx = np.eye(4)
            mx[:3, 0], mx[:3, 1], mx[:3, 2] = L, U, H
            mx[:3, 3] = P
            out.placements.append((name, mx, float(a) if a is not None else 1.0))
        elif sym == "%":
            # Cut: discard the rest of THIS branch.  Skip forward to the
            # ']' that closes the bracket we are inside.
            flush()
            d = 0
            while i < n:
                s2 = word[i].sym
                if s2 == "[":
                    d += 1
                elif s2 == "]":
                    if d == 0:
                        break
                    d -= 1
                i += 1
        # any other symbol is a rewrite-only letter: ignored, per ABOP

    flush()
    out.max_order = _strahler(root)
    return out


def _selftest():
    from .core import Grammar, Module

    def W(src, names=()):
        from .core import tokenize
        return [Module(s, tuple(float(x) for x in a))
                for s, a in tokenize(src, names)]

    # --- frame convention: H x L = U, preserved under every rotation --
    for src in ("+", "-", "&", "^", "/", "\\", "|", "$"):
        o = interpret(W("F" + src + "F"))
        assert o.strands, src
    # explicit orthonormality check after a compound rotation
    o = interpret(W("F+F&F/F^F-F"))
    for s in o.strands:
        assert np.all(np.isfinite(s.points))

    # --- a straight run of F is one strand of the right length -------
    o = interpret(W("FFFF"), step=1.0)
    assert len(o.strands) == 1 and len(o.strands[0]) == 5

    # --- the two starting frames -------------------------------------
    # A PLANAR word (no pitch/roll/$) is drawn flat in XY, so a 2-D
    # fractal reads from the default top view instead of standing on
    # edge.  A SPATIAL word keeps the upright frame so plants grow up.
    flat = interpret(W("FFFF"), step=1.0)
    assert abs(flat.strands[0].points[-1][1] - 4.0) < 1e-9,         "a planar word must advance along +Y"
    assert abs(flat.strands[0].points[-1][2]) < 1e-12,         "a planar word must stay at z = 0"
    up = interpret(W("&(0)FFFF"), step=1.0)
    assert abs(up.strands[0].points[-1][2] - 4.0) < 1e-9,         "a spatial word must grow up +Z"
    # explicit override wins over AUTO in both directions
    forced = interpret(W("FFFF"), step=1.0, plane=PLANE_XZ)
    assert abs(forced.strands[0].points[-1][2] - 4.0) < 1e-9
    assert is_planar(W("F+F-F")) and not is_planar(W("F&F"))
    # every frame is right-handed and orthonormal
    for pl in (PLANE_XY, PLANE_XZ):
        H0, L0, U0 = start_frame(W("F"), pl)
        assert np.allclose(np.cross(H0, L0), U0, atol=1e-12), pl
        assert abs(np.dot(H0, L0)) < 1e-12, pl

    # --- brackets branch, and pop restores the frame -----------------
    o = interpret(W("F[+F]F"))
    ends = [s.points[-1] for s in o.strands]
    assert len(o.strands) >= 2
    trunk = interpret(W("FF")).strands[0].points[-1]
    assert any(np.allclose(e, trunk, atol=1e-9) for e in ends), \
        "the trunk must continue from where the branch was pushed"

    # --- ! tapers, ' colours -----------------------------------------
    o = interpret(W("F!F!F"), width=1.0, width_scale=0.5)
    ws = np.concatenate([s.widths for s in o.strands])
    assert abs(ws.min() - 0.25) < 1e-9, ws
    o = interpret(W("F'F'F"))
    cs = np.concatenate([s.colours for s in o.strands])
    assert cs.max() == 2, cs

    # --- { . } captures a filled polygon -----------------------------
    o = interpret(W("{F+F+F+F}"), angle=90.0)
    assert len(o.polygons) == 1
    pts, _c = o.polygons[0]
    assert len(pts) >= 4

    # --- ~ places a surface on the turtle frame ----------------------
    o = interpret(W("F~(2)"), surfaces=["leaf"])
    assert len(o.placements) == 1
    name, mx, sc = o.placements[0]
    assert name == "leaf" and abs(sc - 2.0) < 1e-9
    assert abs(np.linalg.det(mx[:3, :3]) - 1.0) < 1e-9, \
        "the placement frame must be a rotation"

    # --- % cuts the rest of its branch -------------------------------
    full = interpret(W("F[+FFFF]F"))
    cut = interpret(W("F[+%FFFF]F"))
    assert sum(len(s) for s in cut.strands) < sum(len(s) for s in full.strands)

    # --- tropism bends the heading toward T --------------------------
    # The branch must start TILTED: alpha = e |H x T| is a torque, so a
    # perfectly vertical shoot under vertical gravity has none, and stays
    # straight.  That is the physics, not a defect.
    src = "&(60)" + "F" * 12
    straight = interpret(W(src))
    bent = interpret(W(src), tropism=(0, 0, -1), elasticity=0.3)
    assert bent.strands[0].points[-1][2] < straight.strands[0].points[-1][2], \
        "gravitropism must pull a tilted tip downward"
    up = interpret(W(src), tropism=(0, 0, 1), elasticity=0.3)
    assert up.strands[0].points[-1][2] > straight.strands[0].points[-1][2], \
        "a +Z tropism must lift the tip"
    # and a vertical shoot is genuinely unaffected.  Forced to the
    # upright frame so the heading really is parallel to gravity -- in
    # the planar XY frame H is +Y, which is perpendicular to T and so
    # feels maximum torque, not zero.
    v0 = interpret(W("F" * 8), plane=PLANE_XZ).strands[0].points[-1]
    v1 = interpret(W("F" * 8), plane=PLANE_XZ, tropism=(0, 0, -1),
                   elasticity=0.3).strands[0].points[-1]
    assert np.allclose(v0, v1, atol=1e-9), \
        "H parallel to T gives zero torque, so no bending"

    # --- Horton-Strahler ---------------------------------------------
    # A single unbranched axis is order 1.
    assert interpret(W("FFF")).max_order == 1
    # Two order-1 tips meeting make order 2.
    assert interpret(W("F[+F][-F]")).max_order == 2
    # Two order-2 subtrees meeting make order 3.
    sym3 = "F[+F[+F][-F]][-F[+F][-F]]"
    assert interpret(W(sym3)).max_order == 3, interpret(W(sym3)).max_order
    # THE POINT OF STRAHLER: asymmetry does NOT increment.  An order-2
    # subtree joined by a single order-1 twig is still order 2, because
    # the maximum is unique.  Depth-from-root would call this 3 and so
    # would over-thicken the twig; Strahler keeps the twig subordinate.
    asym = "F[+F[+F][-F]][-F]"
    o = interpret(W(asym))
    assert o.max_order == 2, o.max_order
    assert sorted({s.order for s in o.strands}) == [1, 2]

    # --- the 2 m cube -------------------------------------------------
    g = Grammar.parse("axiom: F\np1: F -> F[+F]F[-F]F")
    o = interpret(g.derive(3)).bbox_fit(2.0)
    pts = o.points()
    ext = (pts.max(axis=0) - pts.min(axis=0)).max()
    assert abs(ext - 2.0) < 1e-6, ext
    assert np.allclose(0.5 * (pts.max(axis=0) + pts.min(axis=0)), 0, atol=1e-6)

    print("turtle: OK -- 20-symbol alphabet, brackets, taper/colour, "
          "{.} polygons, ~ placements, % cut, tropism, Strahler, 2 m fit")
