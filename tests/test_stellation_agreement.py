# Cross-validation of the two stellation engines against each other.
#
# `math_art/stellation_engine.py` enumerates the cells of the icosahedron's
# twenty face planes with a hand-rolled, icosahedron-specific algorithm
# (stdlib only).  `math_art/general_stellation.py` does the same job for an
# arbitrary convex seed, by a different route: per-plane distances, a rank-3
# degeneracy check, convex-hull facet cycling, and the seed's symmetry group
# acting on cells by exact plane permutation.
#
# Two independent implementations of the same hard enumeration are worth
# pointing at each other while both exist.  This runner does that: it builds
# every one of the fifty-nine icosahedra both ways and compares the results.
#
#     python tests/test_stellation_agreement.py
#
# The shells are matched on `(power, size)` -- the number of face planes
# crossed on the way out from the centre, and the number of cells in the
# orbit -- NOT on label order, which differs between the two engines:
#
#     Du Val   power  size        general
#     e1       4      20     ->   s05     (not s04)
#     e2       4      60     ->   s04
#     f1       5      120    ->   s07     (not s06)
#     f2       5      12     ->   s06
#
# so a mapping built from label order would silently swap two pairs and
# produce plausible-looking wrong solids.  The key's uniqueness is asserted
# rather than assumed.
#
# Comparison is on rotation-invariant signatures (vertex and face counts, the
# multiset of vertex radii, the face-size histogram) because the two engines
# construct their icosahedron independently and need not agree on
# orientation.
#
# References:
# - H. S. M. Coxeter, P. Du Val, H. T. Flather and J. F. Petrie, "The
#   Fifty-Nine Icosahedra", University of Toronto Studies (1938); 3rd ed.,
#   Tarquin (1999) -- the shell notation and Miller's rules.
# - S. M. Crennell (1974/1979), the standard 1..59 cross-reference index.
import os
import sys
import math

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))

import stellation_engine as se          # noqa: E402
import general_stellation as gs         # noqa: E402


def shell_map(engine):
    """Du Val label -> general-engine label, keyed on (power, size).

    Raises if the key is not unique on either side: a seed whose shells
    collide on (power, size) cannot be relabelled this way, and must fail
    loudly rather than be matched arbitrarily.
    """
    se._ensure()
    gkeys = {}
    for sh in engine.shells:
        k = (sh['power'], sh['size'])
        if k in gkeys:
            raise AssertionError(
                '(power, size) = %r is not unique in the general engine: '
                'shells %r and %r collide' % (k, gkeys[k], sh['label']))
        gkeys[k] = sh['label']

    skeys = {}
    for L in se._LABEL_ORDER:
        size, _radius, power = se.SHELL_INFO[L]
        k = (power, size)
        if k in skeys:
            raise AssertionError(
                '(power, size) = %r is not unique in stellation_engine: '
                'shells %r and %r collide' % (k, skeys[k], L))
        skeys[k] = L

    if set(skeys) != set(gkeys):
        raise AssertionError(
            'the engines disagree on the shell decomposition itself.\n'
            '  stellation_engine: %s\n  general_stellation: %s'
            % (sorted(skeys), sorted(gkeys)))

    return {L: gkeys[k] for k, L in skeys.items()}


def signature(V, F):
    """A rotation-invariant fingerprint of a built surface."""
    radii = sorted(round(math.sqrt(sum(c * c for c in v)), 6) for v in V)
    return (len(V), len(F),
            tuple(radii),
            tuple(sorted(len(f) for f in F)))


def mapped_code(engine, dumap, k):
    """Crennell k translated through the (power, size) map -- the
    independent route, built from stellation_engine's own tables."""
    duval = se.CRENNELL[k]
    code = [dumap[L] for L in duval]
    if k in se.CRENNELL_CHIRAL:
        for L in duval:
            sh = engine.shell_by_label[dumap[L]]
            if sh['chiral']:
                code = [c for c in code if c != sh['label']]
                code.append(sh['hands'][0])
                break
    return code


def main():
    engine = gs.stellations_of('icosahedron')
    dumap = shell_map(engine)

    print('shell map (Du Val -> general), keyed on (power, size):')
    for L in se._LABEL_ORDER:
        size, _r, power = se.SHELL_INFO[L]
        print('  %-3s power=%d size=%3d  ->  %s' % (L, power, size, dumap[L]))

    # The one chiral shell must be chiral on BOTH sides, or the 27 chiral
    # Crennell figures are not reachable from the general engine.
    chiral = [sh['label'] for sh in engine.shells if sh['chiral']]
    assert chiral == [dumap['f1']], (
        'expected exactly one chiral shell (%s); general engine says %r'
        % (dumap['f1'], chiral))
    hands = engine.shell_by_label[dumap['f1']]['hands']
    assert len(hands) == 2 and len(hands[0]) == len(hands[1]) == 60, (
        'f1 must split into two 60-cell hands; got %r'
        % ([len(h) for h in hands],))
    print('chiral shell %s splits into hands of %s -- OK'
          % (dumap['f1'], [len(h) for h in hands]))

    # The Du Val strings must agree between the modules before the codes
    # built from them can mean anything.
    assert se._CRENNELL_STR == gs._CRENNELL_STR, \
        'the two modules carry different Crennell tables'

    bad = []
    for k in range(1, 60):
        v1, f1 = se.build(se.crennell_cells(k))
        s1 = signature(v1, f1)

        # Two independent routes into the general engine: the (power, size)
        # map derived here, and the module's own classical vocabulary.
        # Both must land on the same solid as stellation_engine.
        for route, code in (('mapped', mapped_code(engine, dumap, k)),
                            ('crennell_code', gs.crennell_code(k))):
            v2, f2 = engine.build(code)
            s2 = signature(v2, f2)
            if s1 != s2:
                bad.append((k, route, s1, s2))

    print()
    for k, route, s1, s2 in bad:
        kind = 'chiral' if k in se.CRENNELL_CHIRAL else 'reflexible'
        print('BAD  Crennell %2d (%s, %r) via %s'
              % (k, kind, se._CRENNELL_STR[k], route))
        print('       stellation_engine: V=%d F=%d' % (s1[0], s1[1]))
        print('       general_stellation: V=%d F=%d' % (s2[0], s2[1]))
        print('       radii agree: %s   face sizes agree: %s'
              % (s1[2] == s2[2], s1[3] == s2[3]))

    n_ref = len(se.CRENNELL_REFLEXIBLE)
    n_chi = len(se.CRENNELL_CHIRAL)
    print('checked %d stellations (%d reflexible, %d chiral): %d mismatches'
          % (59, n_ref, n_chi, len(bad)))
    print('RESULT: %s' % ('OK' if not bad else 'FAIL'))
    return 1 if bad else 0


def _selftest():
    """Raise on any disagreement (for programmatic use)."""
    if main() != 0:
        raise AssertionError('the two stellation engines disagree')


if __name__ == '__main__':
    sys.exit(main())
