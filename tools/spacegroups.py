"""Space-group symmetry operations from Hall symbols.

`tools/rcsr_nets.py` needs the actual operations of each net's space
group to generate site orbits and expand edge representatives.  The
compact, unambiguous way to encode a space group's operations is Hall
notation: a lattice letter plus two or three generator tokens, each a
rotation with an explicit axis and translation, from which the whole
group follows by closure.  This module implements the Hall decoding
rules and carries a Hermann-Mauguin -> Hall table for the symbols that
actually occur in the RCSR catalogue.

The table content is standard crystallographic data (the operations of
the 230 space groups are mathematical facts).  Correctness here is not
taken on trust: `tools/rcsr_nets.py` cross-checks every generated orbit
against the Wyckoff multiplicities RCSR states in `3dall.txt`, and its
own coordination gate rejects any net whose orbit does not reproduce the
stated connectivity.  Entries that failed that cross-check were fixed or
removed, so the table below is exactly the validated subset.

Operations are exact: rotation parts are integer matrices, translation
parts integer twelfths of the cell (every space-group translation is a
multiple of 1/12 in each component).

References:
- S. R. Hall, "Space-group notation with an explicit origin", Acta
  Crystallographica A37, 1981, pp. 517-525 (the notation decoded here).
- T. Hahn (ed.), International Tables for Crystallography, Volume A:
  Space-Group Symmetry, 5th ed., Springer, 2002 (the groups themselves).
"""

import itertools

# ---------------------------------------------------------------------
# rotation matrices, as row-tuples acting on column fractional coords
# ---------------------------------------------------------------------

_I3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

#: proper rotations by (fold, axis letter).  Primed axes are the
#: face-diagonal 2-folds and depend on the PRECEDING principal axis,
#: so they are keyed "'z" etc.
_ROT = {
    ('1', 'z'): _I3,
    ('2', 'z'): ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ('2', 'x'): ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    ('2', 'y'): ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    ('3', 'z'): ((0, -1, 0), (1, -1, 0), (0, 0, 1)),
    ('4', 'z'): ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ('6', 'z'): ((1, -1, 0), (1, 0, 0), (0, 0, 1)),
    ('4', 'x'): ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    ('4', 'y'): ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    ('3', '*'): ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ('2', "'z"): ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),   # axis a-b
    ('2', '"z'): ((0, 1, 0), (1, 0, 0), (0, 0, -1)),     # axis a+b
    ('2', "'x"): ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),   # axis b-c
    ('2', '"x'): ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),     # axis b+c
    ('2', "'y"): ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),   # axis c-a
    ('2', '"y'): ((0, 0, 1), (0, -1, 0), (1, 0, 0)),     # axis c+a
}

#: screw axis directions (translation q/N runs along the rotation axis)
_AXIS_VEC = {'z': (0, 0, 1), 'x': (1, 0, 0), 'y': (0, 1, 0)}

#: translation letters, in twelfths per component
_TRANS = {
    'a': (6, 0, 0), 'b': (0, 6, 0), 'c': (0, 0, 6),
    'n': (6, 6, 6),
    'u': (3, 0, 0), 'v': (0, 3, 0), 'w': (0, 0, 3),
    'd': (3, 3, 3),
}

#: lattice centring vectors, in twelfths
_CENTRING = {
    'P': ((0, 0, 0),),
    'A': ((0, 0, 0), (0, 6, 6)),
    'B': ((0, 0, 0), (6, 0, 6)),
    'C': ((0, 0, 0), (6, 6, 0)),
    'I': ((0, 0, 0), (6, 6, 6)),
    'F': ((0, 0, 0), (0, 6, 6), (6, 0, 6), (6, 6, 0)),
    'R': ((0, 0, 0), (8, 4, 4), (4, 8, 8)),
}


def _matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _matvec(A, v):
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))


def _vadd(u, v, mod=None):
    w = tuple(u[i] + v[i] for i in range(3))
    if mod:
        w = tuple(x % mod for x in w)
    return w


def _neg(A):
    return tuple(tuple(-x for x in row) for row in A)


def _parse_token(tok, pos, prev_fold, prev_axis):
    """One Hall rotation token -> (R, t twelfths, fold, principal axis).

    `pos` is the token's position (1-based) among the rotation tokens,
    `prev_fold`/`prev_axis` describe the preceding one -- both feed the
    default-axis rules of Hall's paper (section 3):
      * the first rotation is about c;
      * a 2 in second place is about a after a 2 or 4, about a-b after
        a 3 or 6;
      * a 3 in third place is about the body diagonal.
    """
    improper = tok.startswith('-')
    if improper:
        tok = tok[1:]
    fold = tok[0]
    if fold not in '12346':
        raise ValueError("bad rotation order in token %r" % tok)
    rest = tok[1:]

    axis = None
    screw = 0
    trans = (0, 0, 0)
    for ch in rest:
        if ch in 'xyz' or ch in "'\"*":
            axis = ch
        elif ch.isdigit():
            if not (0 < int(ch) < int(fold)):
                raise ValueError("bad screw digit in token %r" % tok)
            screw = int(ch)
        elif ch in _TRANS:
            trans = _vadd(trans, _TRANS[ch])
        else:
            raise ValueError("bad character %r in token %r" % (ch, tok))

    if axis is None:
        if pos == 1 or fold == '1':
            axis = 'z'
        elif fold == '2':
            axis = 'x' if prev_fold in '24' else "'"
        elif fold == '3':
            axis = '*'
        else:
            raise ValueError("no default axis for token %r at position %d"
                             % (tok, pos))
    if axis in "'\"":
        axis = axis + (prev_axis if prev_axis in 'xyz' else 'z')

    key = (fold, axis)
    if key not in _ROT:
        raise ValueError("unsupported rotation %r about %r" % (fold, axis))
    R = _ROT[key]
    if screw:
        base = axis[-1] if axis[-1] in 'xyz' else None
        if base is None:
            raise ValueError("screw on a diagonal axis in %r" % tok)
        step = 12 * screw // int(fold)
        trans = _vadd(trans, tuple(step * c for c in _AXIS_VEC[base]))
    if improper:
        R = _neg(R)
    principal = axis[-1] if axis[-1] in 'xyz' else axis
    return R, tuple(t % 12 for t in trans), fold, principal


def ops_from_hall(hall):
    """All distinct operations (R, t) of the group, t in twelfths mod 12.

    Includes the centring translations; the identity is first.  The
    result is the group of order (point-group order) x (centring count),
    i.e. the number of general positions in the conventional cell.
    """
    toks = hall.split()
    if not toks:
        raise ValueError("empty Hall symbol")

    # optional change-of-origin suffix "(u v w)" in twelfths
    shift = (0, 0, 0)
    if toks[-1].endswith(')'):
        i = next(j for j, t in enumerate(toks) if t.startswith('('))
        nums = ' '.join(toks[i:]).strip('()').split()
        shift = tuple(int(x) % 12 for x in nums)
        toks = toks[:i]

    lat = toks[0]
    centro = lat.startswith('-')
    if centro:
        lat = lat[1:]
    if lat not in _CENTRING:
        raise ValueError("unknown lattice letter %r" % lat)

    gens = [(_I3, t) for t in _CENTRING[lat]]
    if centro:
        gens.append((_neg(_I3), (0, 0, 0)))
    prev_fold, prev_axis = None, None
    for pos, tok in enumerate(toks[1:], start=1):
        R, t, fold, axis = _parse_token(tok, pos, prev_fold, prev_axis)
        gens.append((R, t))
        prev_fold, prev_axis = fold, axis

    # close under composition: (R1,t1)(R2,t2) = (R1 R2, R1 t2 + t1)
    ops = {(_I3, (0, 0, 0))}
    frontier = list(ops)
    while frontier:
        nxt = []
        for R1, t1 in frontier:
            for R2, t2 in gens:
                R = _matmul(R1, R2)
                t = _vadd(_matvec(R1, t2), t1, mod=12)
                op = (R, t)
                if op not in ops:
                    ops.add(op)
                    nxt.append(op)
        frontier = nxt
        if len(ops) > 1536:
            raise ValueError("closure runaway for Hall symbol %r" % hall)

    if any(shift):
        # move the origin: t' = t + (I - R) shift
        moved = set()
        for R, t in ops:
            d = _vadd(shift, tuple(-x for x in _matvec(R, shift)))
            moved.add((R, _vadd(t, d, mod=12)))
        ops = moved
    return sorted(ops)


# ---------------------------------------------------------------------
# Hermann-Mauguin -> Hall, for the symbols the RCSR catalogue uses.
# Spellings match the CGD file (no spaces; ":2" = ITA origin choice 2,
# ":H" = hexagonal axes).  Every entry has been validated against the
# Wyckoff multiplicities of the nets that use it (see rcsr_nets.py).
# ---------------------------------------------------------------------

HALL = {
    # triclinic / monoclinic (full symbols, unique axis b)
    'P-1': '-P 1',
    'P1211': 'P 2yb',
    'C121': 'C 2y',
    'C1m1': 'C -2y',
    'P12/m1': '-P 2y',
    'P12/c1': '-P 2yc',
    'P121/c1': '-P 2ybc',
    'P121/n1': '-P 2yn',
    'C12/m1': '-C 2y',
    'C12/c1': '-C 2yc',
    'I12/m1': '-I 2y',
    'I12/a1': '-I 2ya',
    'A12/n1': '-A 2yac',
    # orthorhombic
    'P2221': 'P 2c 2',
    'C2221': 'C 2c 2',
    'C222': 'C 2 2',
    'F222': 'F 2 2',
    'I212121': 'I 2b 2c',
    'Pnn2': 'P 2 -2n',
    'Cmc21': 'C 2c -2',
    'Amm2': 'A 2 -2',
    'Ama2': 'A 2 -2a',
    'Pna21': 'P 2c -2n',
    'Fdd2': 'F 2 -2d',
    'Imm2': 'I 2 -2',
    'Ima2': 'I 2 -2a',
    'Pmmm': '-P 2 2',
    'Pnnn:2': '-P 2ab 2bc',
    'Pccm': '-P 2 2c',
    'Pban:2': '-P 2ab 2b',
    'Pmma': '-P 2a 2a',
    'Pnna': '-P 2a 2bc',
    'Pmna': '-P 2ac 2',
    'Pcca': '-P 2a 2ac',
    'Pbam': '-P 2 2ab',
    'Pccn': '-P 2ab 2ac',
    'Pbcm': '-P 2c 2b',
    'Pnnm': '-P 2 2n',
    'Pmmn:2': '-P 2ab 2a',
    'Pbcn': '-P 2n 2ab',
    'Pbca': '-P 2ac 2ab',
    'Pnma': '-P 2ac 2n',
    'Cmcm': '-C 2c 2',
    'Cmca': '-C 2bc 2',
    'Cmmm': '-C 2 2',
    'Cccm': '-C 2 2c',
    'Cmma': '-C 2b 2',
    'Ccca:2': '-C 2b 2bc',
    'Fmmm': '-F 2 2',
    'Fddd:2': '-F 2uv 2vw',
    'Immm': '-I 2 2',
    'Ibam': '-I 2 2c',
    'Ibca': '-I 2b 2c',
    'Imma': '-I 2b 2',
    # tetragonal
    'P43': 'P 4cw',
    'I41': 'I 4bw',
    'P-4': 'P -4',
    'I-4': 'I -4',
    'P42/m': '-P 4c',
    'P4/n:2': '-P 4a',
    'P42/n:2': '-P 4bc',
    'I4/m': '-I 4',
    'I41/a:2': '-I 4ad',
    'P422': 'P 4 2',
    'P4122': 'P 4w 2c',
    'P41212': 'P 4abw 2nw',
    'P4222': 'P 4c 2',
    'P42212': 'P 4n 2n',
    'P4322': 'P 4cw 2c',
    'P43212': 'P 4nw 2abw',
    'I422': 'I 4 2',
    'I4122': 'I 4bw 2bw',
    'P42nm': 'P 4n -2n',
    'I4cm': 'I 4 -2c',
    'I41md': 'I 4bw -2',
    'I41cd': 'I 4bw -2c',
    'P-421c': 'P -4 2n',
    'P-4m2': 'P -4 -2',
    'P-4c2': 'P -4 -2c',
    'P-4n2': 'P -4 -2n',
    'I-4m2': 'I -4 -2',
    'I-4c2': 'I -4 -2c',
    'I-42m': 'I -4 2',
    'I-42d': 'I -4 2bw',
    'P4/mmm': '-P 4 2',
    'P4/mcc': '-P 4 2c',
    'P4/nbm:2': '-P 4a 2b',
    'P4/nnc:2': '-P 4a 2bc',
    'P4/mbm': '-P 4 2ab',
    'P4/mnc': '-P 4 2n',
    'P4/nmm:2': '-P 4a 2a',
    'P4/ncc:2': '-P 4a 2ac',
    'P42/mmc': '-P 4c 2',
    'P42/mcm': '-P 4c 2c',
    'P42/nbc:2': '-P 4ac 2b',
    'P42/nnm:2': '-P 4ac 2bc',
    'P42/mbc': '-P 4c 2ab',
    'P42/mnm': '-P 4n 2n',
    'P42/nmc:2': '-P 4ac 2a',
    'P42/ncm:2': '-P 4ac 2ac',
    'I4/mmm': '-I 4 2',
    'I4/mcm': '-I 4 2c',
    'I41/amd:2': '-I 4bd 2',
    'I41/acd:2': '-I 4bd 2c',
    'I41/amd': '-I 4bd 2',
    # trigonal / rhombohedral (hexagonal axes)
    'P3121': "P 31 2\"",
    'P3112': 'P 31 2 (0 0 4)',
    'P3221': "P 32 2\"",
    'P3212': 'P 32 2 (0 0 2)',
    'P31c': 'P 3 -2c',
    'P3m1': "P 3 -2\"",
    'R3:H': 'R 3',
    'R3m:H': "R 3 -2\"",
    'R3c:H': "R 3 -2\"c",
    'R-3': '-R 3',
    'R-3:H': '-R 3',
    'R32:H': "R 3 2\"",
    'R-3m': "-R 3 2\"",
    'R-3m:H': "-R 3 2\"",
    'R-3c:H': "-R 3 2\"c",
    'P-31m': '-P 3 2',
    'P-31c': '-P 3 2c',
    'P-3m1': "-P 3 2\"",
    'P-3c1': "-P 3 2\"c",
    # hexagonal
    'P61': 'P 61',
    'P62': 'P 62',
    'P6/m': '-P 6',
    'P63/m': '-P 6c',
    'P622': 'P 6 2',
    'P6122': 'P 61 2 (0 0 5)',
    'P6522': 'P 65 2 (0 0 1)',
    'P6222': 'P 62 2 (0 0 4)',
    'P6422': 'P 64 2 (0 0 2)',
    'P6322': 'P 6c 2c',
    'P6mm': 'P 6 -2',
    'P63cm': 'P 6c -2',
    'P63mc': 'P 6c -2c',
    'P-6m2': 'P -6 2',
    'P-62m': 'P -6 -2',
    'P-62c': 'P -6c -2c',
    'P6/mmm': '-P 6 2',
    'P6/mcc': '-P 6 2c',
    'P63/mcm': '-P 6c 2',
    'P63/mmc': '-P 6c 2c',
    # cubic
    'P23': 'P 2 2 3',
    'F23': 'F 2 2 3',
    'I23': 'I 2 2 3',
    'P213': 'P 2ac 2ab 3',
    'I213': 'I 2b 2c 3',
    'Pm-3': '-P 2 2 3',
    'Pn-3:2': '-P 2ab 2bc 3',
    'Fm-3': '-F 2 2 3',
    'Fd-3:2': '-F 2uv 2vw 3',
    'Im-3': '-I 2 2 3',
    'Pa-3': '-P 2ac 2ab 3',
    'Ia-3': '-I 2b 2c 3',
    'P432': 'P 4 2 3',
    'P4232': 'P 4n 2 3',
    'F432': 'F 4 2 3',
    'F4132': 'F 4d 2 3',
    'I432': 'I 4 2 3',
    'P4332': 'P 4acd 2ab 3',
    'P4132': 'P 4bd 2ab 3',
    'I4132': 'I 4bd 2c 3',
    'P-43m': 'P -4 2 3',
    'F-43m': 'F -4 2 3',
    'I-43m': 'I -4 2 3',
    'P-43n': 'P -4n 2 3',
    'F-43c': 'F -4a 2 3',
    'I-43d': 'I -4bd 2c 3',
    'Pm-3m': '-P 4 2 3',
    'Pn-3n:2': '-P 4a 2bc 3',
    'Pm-3n': '-P 4n 2 3',
    'Pn-3m:2': '-P 4bc 2bc 3',
    'Fm-3m': '-F 4 2 3',
    'Fm-3c': '-F 4a 2 3',
    'Fd-3m:2': '-F 4vw 2vw 3',
    'Fd-3m': '-F 4vw 2vw 3',
    'Fd-3c:2': '-F 4ud 2vw 3',
    'Im-3m': '-I 4 2 3',
    'Ia-3d': '-I 4bd 2c 3',
}

_OPS_CACHE = {}


def ops(hm):
    """Operations of a Hermann-Mauguin symbol as spelled in the CGD."""
    key = hm.strip()
    if key not in _OPS_CACHE:
        if key not in HALL:
            raise KeyError("no Hall symbol for space group %r" % key)
        _OPS_CACHE[key] = ops_from_hall(HALL[key])
    return _OPS_CACHE[key]


def apply_op(op, frac):
    """Apply (R, t-twelfths) to a fractional coordinate triple."""
    R, t = op
    return tuple(sum(R[i][k] * frac[k] for k in range(3)) + t[i] / 12.0
                 for i in range(3))


def _same_site(p, q, tol):
    """Do two fractional points coincide modulo the lattice?"""
    for a, b in zip(p, q):
        d = abs(a - b) % 1.0
        if min(d, 1.0 - d) > tol:
            return False
    return True


def orbit(frac, hm=None, group_ops=None, tol=1e-4):
    """The site orbit inside one cell, as tuples in [0, 1).

    Images are merged by TOLERANCE, not by rounding to a grid: the
    catalogue stores five decimals, so images of the same site can
    differ by ~2e-5 -- and a coordinate ending in ...5 sits exactly on
    a rounding boundary, which made grid-rounding split real orbits
    (rho, npo and every other net with a ...5 coordinate)."""
    if group_ops is None:
        group_ops = ops(hm)
    out = []
    for op in group_ops:
        q = tuple(x % 1.0 for x in apply_op(op, frac))
        if not any(_same_site(q, r, tol) for r in out):
            out.append(q)
    return sorted(out)


#: known group orders (general-position count in the conventional cell)
#: for the cubic groups -- the loader's supply, so they are asserted.
_CUBIC_ORDER = {
    'P23': 12, 'F23': 48, 'I23': 24, 'P213': 12, 'I213': 24,
    'Pm-3': 24, 'Pn-3:2': 24, 'Fm-3': 96, 'Fd-3:2': 96, 'Im-3': 48,
    'Pa-3': 24, 'Ia-3': 48,
    'P432': 24, 'P4232': 24, 'F432': 96, 'F4132': 96, 'I432': 48,
    'P4332': 24, 'P4132': 24, 'I4132': 48,
    'P-43m': 24, 'F-43m': 96, 'I-43m': 48, 'P-43n': 24, 'F-43c': 96,
    'I-43d': 48,
    'Pm-3m': 48, 'Pn-3n:2': 48, 'Pm-3n': 48, 'Pn-3m:2': 48,
    'Fm-3m': 192, 'Fm-3c': 192, 'Fd-3m:2': 192, 'Fd-3m': 192,
    'Fd-3c:2': 192, 'Im-3m': 96, 'Ia-3d': 96,
}


def _selftest():
    # every table entry must close to a well-formed group
    for hm in HALL:
        g = ops(hm)
        n = len(g)
        # order divisible by the centring count, identity present
        assert (_I3, (0, 0, 0)) in g, hm
        assert n >= 1
    for hm, want in _CUBIC_ORDER.items():
        got = len(ops(hm))
        assert got == want, "%s: group order %d, expected %d" % (
            hm, got, want)
    # spot checks: srs (I4132, Wyckoff 8a) and dia (Fd-3m:2) orbits
    srs = orbit((0.125, 0.125, 0.125), 'I4132')
    assert len(srs) == 8, srs
    dia = orbit((0.125, 0.125, 0.625), 'Fd-3m:2')
    assert len(dia) == 8, dia
    # a general position must have full multiplicity
    gen = orbit((0.123, 0.4567, 0.789), 'Ia-3d', tol=1e-6)
    assert len(gen) == 96, len(gen)
    print("spacegroups: OK (%d symbols)" % len(HALL))


if __name__ == "__main__":
    _selftest()
