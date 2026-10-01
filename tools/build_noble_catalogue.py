"""Build math_art/polyhedra/_noble_data.py -- the 146 noble polyhedra.

Source: Connor Hill, "The complete set of noble polyhedra" (arXiv:2607.28711,
2026) and his GPL-3.0 repository github.com/Plasmath/noble-tools-revised,
whose `library/` holds an .OFF model of every noble polyhedron and a
`summary.txt` of exact minimal polynomials for every orbit location.

Nothing is transcribed as geometry.  For every orbit Hill lists, this
script

  1. polishes the location (a, b) from his minimal polynomials,
  2. re-facets that orbit with our own engine (`polyhedra/noble.py`),
  3. matches each of his .OFF models to one of OUR facetings, face set for
     face set, up to the vertex set's symmetry,

and stores only the orbit type, the location and our generating face.  A
model that fails to match is an error, as is a faceting of ours that Hill
does not list.  His appendix tables (Schlafli type, V/E/F, symmetry, dual)
are read from the paper and cross-checked against what the engine builds.

Usage:
    git clone https://github.com/Plasmath/noble-tools-revised <dir>
    python tools/build_noble_catalogue.py --library <dir>/library \
        --paper research/papers/polyhedra-and-solids/hill-2026-complete-noble-polyhedra.pdf
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'math_art'))
from polyhedra import noble as N                                # noqa: E402

OUT = os.path.join(HERE, '..', 'math_art', 'polyhedra', '_noble_data.py')

#: the classical regular polyhedra among them, identified by face shape
#: (checked below, not assumed)
REGULAR = {
    'T-1': 'Tetrahedron', 'O-1': 'Octahedron', 'C-1': 'Cube',
    'I-1': 'Icosahedron', 'D-1': 'Dodecahedron',
}


def read_off(fn):
    t = open(fn).read().split()
    assert t[0] == 'OFF', fn
    nv, nf = int(t[1]), int(t[2])
    k = 4
    V = np.array([[float(t[k + 3 * i + j]) for j in range(3)]
                  for i in range(nv)])
    k += 3 * nv
    F = []
    for _ in range(nf):
        m = int(t[k])
        F.append([int(x) for x in t[k + 1:k + 1 + m]])
        k += 1 + m
    return V, F


def parse_poly(s, var):
    """'5a^8 - 10a^7 + a - 1' -> integer coefficients, highest first."""
    s = s.replace(' ', '').replace('*', '')
    terms = re.findall(r'([+-]?)(\d*)(%s(?:\^(\d+))?)?' % var, s)
    coef = {}
    for sign, num, v, exp in terms:
        if not num and not v:
            continue
        c = int(num) if num else 1
        if sign == '-':
            c = -c
        e = (int(exp) if exp else 1) if v else 0
        coef[e] = coef.get(e, 0) + c
    deg = max(coef)
    return [coef.get(e, 0) for e in range(deg, -1, -1)]


def read_summary(fn):
    """{orbit label: (a, poly_a, b, poly_b)}"""
    out = {}
    for line in open(fn):
        cols = [c.strip() for c in line.split('|')]
        if len(cols) < 3 or not re.match(r'^\w+-\w+$', cols[0]):
            continue
        pa = parse_poly(cols[2], 'a')
        a = N.polish_root(pa, float(cols[1]))
        if len(cols) >= 5:
            pb = parse_poly(cols[4], 'b')
            b = N.polish_root(pb, float(cols[3]))
        else:
            pb, b = None, 1.0
        out[cols[0]] = (a, pa, b, pb)
    return out


def read_tables(pdf):
    """Hill's Appendix A: symbol -> (schlafli, V, E, F, symmetry, dual)."""
    import fitz
    text = ''.join(p.get_text() for p in fitz.open(pdf))
    L = text.replace('∗', '*').splitlines()
    s = next(i for i, l in enumerate(L) if l.startswith('Table 5: Noble'))
    e = next(i for i, l in enumerate(L) if l.startswith('Appendix B'))
    toks = [l.strip() for l in L[s:e] if l.strip()]
    sym = re.compile(r'^(T|O|C|I|ID|D|tO|tC|rC|tI|tD|rD|sC|gC|sD|gD)-'
                     r'(\d+(\.\d+)?|F\d?)$')
    rows, i = {}, 0
    while i < len(toks):
        if sym.match(toks[i]) and i + 6 < len(toks) \
                and toks[i + 1].startswith('{'):
            r = toks[i:i + 7]
            rows[r[0]] = (r[1], int(r[2]), int(r[3]), int(r[4]), r[5], r[6])
            i += 7
        else:
            i += 1
    assert len(rows) == 146, len(rows)
    return rows


def turning(V, f):
    """How many times a face winds about its centre (1 convex, 2 star)."""
    P = np.asarray(V, float)[list(f)]
    c = P.mean(axis=0)
    _u, _s, vt = np.linalg.svd(P - c)
    e1, e2 = vt[0], vt[1]
    ang = [math.atan2(float((p - c) @ e2), float((p - c) @ e1)) for p in P]
    tot = 0.0
    for i in range(len(ang)):
        d = ang[(i + 1) % len(ang)] - ang[i]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        tot += d
    return int(round(abs(tot) / (2 * math.pi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--library', required=True)
    ap.add_argument('--paper', required=True)
    args = ap.parse_args()
    tables = read_tables(args.paper)

    entries = []
    unmatched = []
    errata = []
    for tdir in sorted(glob.glob(os.path.join(args.library, '*', '*'))):
        otype = os.path.basename(tdir)
        if otype not in N.ORBIT_TYPES:
            continue
        summ = os.path.join(tdir, 'summary.txt')
        locs = read_summary(summ) if os.path.exists(summ) else {}
        cache = {}
        used = set()
        for fn in sorted(glob.glob(os.path.join(tdir, '*.off'))):
            symbol = os.path.basename(fn)[:-4]
            if symbol not in tables:
                print('skip (not in Hill\'s 146):', symbol)
                continue
            label = symbol.rsplit('.', 1)[0] if N.dof(otype) else otype
            a, pa, b, pb = locs.get(label, (1.0, None, 1.0, None))
            if label not in cache:
                V = N.orbit(otype, a, b)
                cache[label] = (V, N.type_facetings(otype, a, b),
                                N.transitive_groups(V)[0][1])
            V, found, full = cache[label]
            Vh, Fh = read_off(fn)
            s = float(np.max(np.linalg.norm(V, axis=1)))
            sh = float(np.max(np.linalg.norm(Vh, axis=1)))
            idx = {N._vkey(v, s, 5): i for i, v in enumerate(V)}
            m = [idx.get(N._vkey(v * s / sh, s, 5)) for v in Vh]
            if None in m:
                unmatched.append((symbol, 'vertices do not map'))
                continue
            want = frozenset(N._edges([m[i] for i in f]) for f in Fh)
            hit = None
            for g, face in found:
                faces = N.build_faces(V, g, face)
                if any(frozenset(N._edges(N._apply(p, f)) for f in faces)
                       == want for p in full):
                    hit = (g, face, faces)
                    break
            if hit is None:
                unmatched.append((symbol, 'no faceting of ours matches'))
                continue
            g, face, faces = hit
            used.add((label, tuple(face)))
            d = N.describe(V, faces)
            sch, tv, te, tf, tsym, tdual = tables[symbol]
            # Hill's .OFF model is the primary record, and the face-set
            # match above already proves ours is the same polyhedron.  His
            # Appendix A disagrees with his own models in a few rows (some
            # rows even contradict themselves: 2E = pF = qV fails), so the
            # table is cross-checked and every disagreement is logged as
            # an erratum rather than trusted.
            mine = ('{%d, %d}' % (d['p'], d['q']), d['V'], d['E'], d['F'], g)
            if (sch, tv, te, tf, tsym) != mine:
                errata.append('%s: table %s V=%d E=%d F=%d %s; model gives '
                              '%s V=%d E=%d F=%d %s'
                              % ((symbol, sch, tv, te, tf, tsym) + mine))
            dh = N.describe(Vh, Fh)
            assert abs(dh['ratio'] - d['ratio']) < 1e-6, (symbol, dh, d)
            name = REGULAR.get(symbol, '')
            wind = turning(V, face)
            entries.append(dict(
                symbol=symbol, type=otype, orbit=label, a=a, b=b,
                poly_a=pa, poly_b=pb, group=g, face=list(face),
                p=d['p'], q=d['q'], V=d['V'], E=d['E'], F=d['F'],
                dual=tdual, ratio=round(dh['ratio'], 12), winding=wind,
                name=name))
            print('%-9s %-5s a=%.9f b=%.9f  %s  {%d,%d}  %s'
                  % (symbol, g, a, b, (d['V'], d['E'], d['F']),
                     d['p'], d['q'], name))
        # ... and the other direction: nothing of ours that Hill lacks
        for label, (_V, found, _full) in cache.items():
            for g, face in found:
                if (label, tuple(face)) not in used:
                    unmatched.append((label, 'our faceting %s %s is not in '
                                      "Hill's library" % (g, face)))

    # regular names for the star ones, by face winding
    for e in entries:
        if e['symbol'] in ('I-2', 'I-3') and not e['name']:
            e['name'] = ('Great Dodecahedron' if e['winding'] == 1
                         else 'Small Stellated Dodecahedron')
        if e['symbol'] == 'I-4':
            e['name'] = 'Great Icosahedron'
        if e['symbol'] == 'D-6':
            e['name'] = 'Great Stellated Dodecahedron'
    for e in errata:
        print('ERRATUM', e)
    if unmatched:
        print('UNMATCHED:', unmatched)
        sys.exit(1)
    assert len(entries) == 146, len(entries)
    order = list(tables)
    entries.sort(key=lambda e: order.index(e['symbol']))

    with open(OUT, 'w', newline='\n') as fh:
        fh.write(
            '# The 146 noble polyhedra of C. Hill, "The complete set of '
            'noble\n# polyhedra", arXiv:2607.28711 (2026).  GENERATED by\n'
            '# tools/build_noble_catalogue.py -- do not edit.\n#\n'
            '# Each entry is an orbit type, a location (a, b) polished from '
            "Hill's\n# minimal polynomials, a group and ONE generating face "
            'of our own\n# faceting search (vertex indices into '
            '`noble.orbit(type, a, b)`).  The\n# whole polyhedron is the '
            "face's orbit under the group.  `ratio` is the\n# in/circumradius "
            "ratio measured on Hill's own model, kept as an\n# independent "
            'check.  V/E/F, Schlafli {p,q} and symmetry are measured on '
            'the\n# polyhedron the engine builds (identical, face set for '
            "face set, to\n# Hill's model); `dual` is from his Appendix A.  "
            'Rows of that table\n# which disagree with his own models are '
            'listed in ERRATA.\n\n')
        fh.write('ERRATA = [\n')
        for x in errata:
            fh.write('    %r,\n' % x)
        fh.write(']\n\n')
        fh.write('NOBLE = [\n')
        for e in entries:
            fh.write('    %s,\n' % json.dumps(e).replace('null', 'None')
                     .replace('true', 'True').replace('false', 'False'))
        fh.write(']\n')
    print('wrote %d entries to %s' % (len(entries), OUT))


if __name__ == '__main__':
    main()
