"""Construct entry 19, the Wurtzite nodal tetrahedron, and add it to
hex_solids.pkl for tools/emit_pearce_data.py to offer to the gate.

THE SOLID.  Pearce's nodal polyhedron of the lonsdaleite (wurtzite)
net: a tetrahedron of hexagons around a node, "bounded by one plane
regular hexagon and three saddle hexagons with 2-fold symmetry"
(ch. 8, p. 107).  Combinatorially its structure is forced by the row's
valence column (4 primary z3, 6 secondary z2, 4 hexagon faces): a
planar regular hexagon of six full-[110] in-plane branches, with three
of its alternate corners tied by half-[111] bonds to a cap of three
2-connected vertices that meet a 3-connected apex on the axis through
full-[110] branches.

TWO REPRESENTATIVES, ONE ROW.  The EXACT space-filling form of this
polyhedron -- the dual-net cell whose vertices are the centres of the
2 trihedra + 2 pentahedra (entries 6 and 25) meeting at each node, one
cell per node, volume sqrt(2)/4 -- has edge lengths 1/sqrt(3) a and
sqrt(2/3) a, which are NOT Universal Node branch moduli at any common
scale (checked below).  Table 8.1 tabulates the Universal-Node-
proportioned representative, which this script builds: every edge a
legal hexagonal-lattice branch (9 full-[110] + 3 half-[111], exactly
the row's branch columns), every corner 90d or 120d as printed.  That
representative does not itself close-pack (its volume is 9/16 sqrt(2),
16 copies per 9 hexagonal cells), just as Pearce's plastic Universal
Node models approximate rather than reproduce several of his fillings.

References:
- Peter Pearce, "Structure in Nature is a Strategy for Design", The
  MIT Press, 1978, ch. 8 -- Table 8.1 entry 19, Table 8.2 entry 14,
  and the "Wurtzite and Carborundum" nodal-polyhedron discussion.
"""
import pickle
import sys

MA = (r"C:\Users\dkrid\Projects\2026_07_21_Math_Art"
      r"\.claude\worktrees\spidrons\math_art")
sys.path.insert(0, MA)
import pearce_net as pn                                     # noqa: E402


def build19():
    """The Universal-Node representative, integer 24ths of the cell."""
    b = [(32, 16, 0), (32, 40, 0), (8, 40, 0), (-16, 16, 0),
         (-16, -8, 0), (8, -8, 0)]                # base hexagon
    T = [(32, 16, 9), (8, 40, 9), (-16, -8, 9)]   # caps over b0, b2, b4
    apex = (8, 16, 9)
    V = tuple(b) + tuple(T) + (apex,)
    F = ((0, 1, 2, 3, 4, 5),
         (0, 1, 2, 7, 9, 6),
         (2, 3, 4, 8, 9, 7),
         (4, 5, 0, 6, 9, 8))
    return V, F


def check(V, F):
    """Verify the solid against its Table 8.1 row, column by column."""
    import pearce_table as pt
    r = [x for x in pt.TABLE if x['number'] == 19][0]
    X = pn.cartesian(V, pn.HEX_BASIS, pn.HEX_DIVISOR)
    assert all(pn.closes([V[i] for i in f]) for f in F)
    assert pn.euler(V, F) == (10, 12, 4, 2), pn.euler(V, F)
    hist, _ = pn.valence_histogram(F)
    assert hist == {3: 4, 2: 6}, hist
    bt = pn.branch_totals(V, F, lattice='HEX')
    assert bt == dict(r['branches']), bt
    got = {}
    for cyc in F:
        loop = [X[i] for i in cyc]
        k = (len(cyc), pn.face_symmetry_label(loop),
             pn.face_plane_class(loop, lattice='HEX'))
        got[k] = got.get(k, 0) + 1
    want = {}
    for fd in r['faces']:
        k = (fd['n'], fd['symmetry'], fd['plane'])
        want[k] = want.get(k, 0) + fd['count']
    assert got == want, (got, want)
    ax = pn.axis_counts(X, lattice='HEX')
    assert ax == tuple(r['axes']), ax
    assert pn.is_closed_surface(F)
    assert pn.orientation_consistent(pn.orient_faces(X, F))
    print("entry 19 passes every row column (branches %r, axes %r)"
          % (bt, ax))


def check_ops_against_spacegroups():
    """Cross-check pearce_net's hexagonal point group against the
    Hall-symbol engine: the rotation parts of P6/mmm (the hexagonal
    holohedry) must be exactly our 24 integer matrices, transposed for
    the row-vector convention."""
    sys.path.insert(0, r"C:\Users\dkrid\Projects\2026_07_21_Math_Art"
                       r"\.claude\worktrees\spidrons\tools")
    import spacegroups as sg
    rots = {tuple(tuple(row) for row in R) for R, _t in sg.ops('P6/mmm')}
    ours = {tuple(tuple(int(x) for x in row) for row in M.T)
            for M, _R in pn.hex_point_ops()}
    assert rots == ours, "hex point ops disagree with P6/mmm"
    print("hex point ops == rotation parts of P6/mmm (%d ops)" % len(ours))


def check_dual_form_is_not_UN():
    """The exact space-filler's edges are Universal-Node-illegal at
    every scale -- the measurement the module docstring cites.

    Its in-plane edges run along the mid-directions (2,1,0) (between
    the a-axes), which is a PLANE-class direction of the hexagonal
    system but not a branch direction at all; its axial edges run
    along c, a legal [111] direction, but at c/2 -- between the half
    and full moduli.  Scaling to fix the axial modulus (3/4 or 3/2)
    leaves the in-plane direction just as illegal, so no common scale
    exists.  (A length-only comparison DOES find coincidences -- e.g.
    at scale sqrt(3)/2 the lengths match half-[110] and half-[100] --
    which is why this check tests vectors, not lengths.)"""
    assert pn.hex_branch_class((16, 8, 0)) is None          # mid dir
    assert pn.hex_branch_class((0, 0, 12)) is None          # c at c/2
    for s_num, s_den in ((3, 4), (3, 2)):                   # fix axial
        ax = (0, 0, 12 * s_num // s_den)
        assert pn.hex_branch_class(ax) is not None
        inpl = (16 * s_num, 8 * s_num, 0)
        if all(x % s_den == 0 for x in inpl):
            inpl = tuple(x // s_den for x in inpl)
            assert pn.hex_branch_class(inpl) is None, inpl
    print("dual (space-filling) form confirmed non-UN at every scale")


def main():
    V, F = build19()
    check(V, F)
    check_ops_against_spacegroups()
    check_dual_form_is_not_UN()
    path = "hex_solids.pkl"
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except FileNotFoundError:
        data = {}
    data[19] = (V, F)
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    print("hex_solids.pkl now offers: %s" % sorted(data))


if __name__ == "__main__":
    main()
