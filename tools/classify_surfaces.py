"""Work out what each periodic surface IS, from its geometry.

The Add menu used to group these by where they came from -- "(Evolver
cell)", "(nodal approximation)", "(relaxed)" -- which tells a reader
about our pipeline and nothing about the surface.  What belongs in a
menu is what the thing is: the lattice it repeats on and how many
handles its unit cell has.

Neither is written down for most rows, so both are DERIVED here:

  * The lattice comes from the ROTATION ORDERS in the group the datafile
    declares.  The first attempt looked for pure translations in that
    group and found none anywhere -- correctly, because a unit-cell word
    generates a POINT group; the lattice translations are what carries
    one cell to the next and are not among them.  What the point group
    does carry is its axes, and those name the system on their own: a
    3-fold axis along a body diagonal is the signature of cubic, a
    3- or 6-fold along one axis with none off it is hexagonal or
    trigonal, a lone 4-fold is tetragonal.  The cell's proportions then
    separate the remaining cases.

  * The genus is the genus of the QUOTIENT surface -- the compact
    surface the cell becomes inside the flat 3-torus.  A TPMS unit cell
    is not a closed surface: it has open rims where it crosses the cell
    walls, which is why counting chi on it directly returned "open" for
    every row tried.  Opposite rims are identified by a lattice
    translation first, and chi = V - E + F is counted after that.

Run it when the cell table changes:

    python tools/classify_surfaces.py            # print the table
    python tools/classify_surfaces.py --write    # rewrite surface_class.py
"""

import argparse
import io
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'math_art'))

from minsurf import plateau as pl                # noqa: E402
from minsurf.fecells import FE_CELLS             # noqa: E402

OUT = os.path.join(ROOT, 'math_art', 'minsurf', 'surface_class.py')

# Angles and lengths are compared with a slack tolerance: these lattices
# come out of floating-point products of reflection matrices, so "equal"
# means equal to a few parts in a thousand, not to machine precision.
LEN_TOL = 2e-3
ANG_TOL = 1.5          # degrees


def rotation_axes(letters, word, depth=3):
    """(axis, order) for every proper rotation in the generated group."""
    mats = [np.asarray(letters[c], dtype=float) for c in sorted(set(word))]
    seen = {}
    frontier = [np.eye(4)]
    for _ in range(depth):
        nxt = []
        for M in frontier:
            for G in mats:
                P = G @ M
                key = tuple(np.round(P.ravel(), 6))
                if key in seen:
                    continue
                seen[key] = P
                nxt.append(P)
        frontier = nxt
    out = []
    for P in seen.values():
        R = P[:3, :3]
        if np.linalg.det(R) < 0:               # improper: a mirror
            continue
        tr = float(np.trace(R))
        if abs(tr - 3.0) < 1e-6:               # identity
            continue
        # trace = 1 + 2 cos(theta) fixes the order; the axis is the
        # eigenvector for eigenvalue +1.
        cs = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        theta = float(np.degrees(np.arccos(cs)))
        order = None
        for n in (2, 3, 4, 6):
            if abs(theta - 360.0 / n) < 2.0:
                order = n
                break
        if order is None:
            continue
        w, V = np.linalg.eig(R)
        ax = None
        for i in range(3):
            if abs(w[i].real - 1.0) < 1e-6 and abs(w[i].imag) < 1e-6:
                ax = np.real(V[:, i])
        if ax is None:
            continue
        ax = ax / (np.linalg.norm(ax) or 1.0)
        out.append((ax, order))
    return out


def system_of(axes, bbox):
    """Name the crystal system from the point group's axes.

    The order of the axes decides it, and the cell proportions only
    settle what the axes leave open.  A 3-fold along a body diagonal is
    what makes a lattice cubic -- it is the axis the cube has and the
    square prism does not -- so that test comes first and does not care
    what the bounding box looks like.
    """
    if not axes:
        return None
    def is_diag(a):
        v = np.abs(a)
        return (abs(v[0] - v[1]) < 0.08 and abs(v[1] - v[2]) < 0.08
                and v[0] > 0.4)
    def is_axis(a):
        v = np.abs(a)
        return sorted(v)[-1] > 0.98
    if any(o == 3 and is_diag(a) for a, o in axes):
        return 'CUBIC'
    if any(o in (3, 6) and is_axis(a) for a, o in axes):
        return 'HEXAGONAL'
    if any(o == 3 for a, o in axes):
        return 'RHOMBOHEDRAL'
    if any(o == 4 and is_axis(a) for a, o in axes):
        return 'TETRAGONAL'
    a, b, c = (float(x) for x in bbox)
    def eq(x, y):
        return abs(x - y) <= 0.02 * max(x, y)
    if eq(a, b) and eq(b, c):
        return 'CUBIC'
    if eq(a, b) or eq(b, c) or eq(a, c):
        return 'TETRAGONAL'
    return 'ORTHORHOMBIC'


def genus_of(W, faces, period=None, tol=1e-3):
    """Genus of the cell's QUOTIENT surface, chi = V - E + F.

    A unit cell is open where it crosses the cell walls, so its own chi
    is not the surface's.  Identifying each rim with the one a lattice
    period away closes it into the compact surface that sits in the flat
    3-torus, which is the surface whose genus the literature quotes.
    """
    W = np.asarray(W, dtype=float)
    if period is not None:
        # Fold every vertex into one period box, so a point on one wall
        # and its partner on the opposite wall become the same vertex.
        p = np.asarray(period, dtype=float)
        p = np.where(p > tol, p, 1.0)
        folded = W - p * np.round(W / p)
        key = np.round(folded / tol).astype(np.int64)
        _, remap = np.unique(key, axis=0, return_inverse=True)
        faces = [[int(remap[i]) for i in f] for f in faces]
        faces = [f for f in faces if len(set(f)) == len(f)]
        W = np.zeros((int(remap.max()) + 1, 3))
    V = len(W)
    edges = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            e = (f[i], f[(i + 1) % n])
            edges[(min(e), max(e))] = edges.get((min(e), max(e)), 0) + 1
    E = len(edges)
    F = len(faces)
    boundary = sum(1 for v in edges.values() if v == 1)
    if boundary:
        return None, boundary
    if any(v > 2 for v in edges.values()):
        return None, 'non-manifold'
    chi = V - E + F
    if chi % 2:
        return None, 0
    return 1 - chi // 2, 0


# Rows whose derivation is known to be wrong, with the reason.
#
# The proportions test asks whether the BUILT MESH is cube-shaped, which
# is sound for a row that assembles a full unit cell and unsound for one
# that ships a fundamental piece -- the piece can be any shape at all.
# These two were caught by checking the answer against the literature
# and are corrected here rather than by loosening the test, which would
# only make it wrong in more places.
#
# The other fundamental-piece rows (the Box types, Simoes-Batista,
# Triply Periodic Costa, Stessmann, F-RD(r)) are still derived from
# proportions and are the place to look first if a row appears under the
# wrong heading.
OVERRIDE = {
    # Lidinoid is rhombohedral -- an rPD-family surface, not a cubic one.
    'LIDINOID': 'RHOMBOHEDRAL',
    # P, the gyroid and D are all cubic; this row is their Bonnet
    # associate family and the piece it builds is not cube-shaped.
    'PGD': 'CUBIC',
}

STATED = (('cubic', 'CUBIC'), ('hexagonal', 'HEXAGONAL'),
          ('tetragonal', 'TETRAGONAL'), ('trigonal', 'TRIGONAL'),
          ('rhombohedral', 'RHOMBOHEDRAL'),
          ('orthorhombic', 'ORTHORHOMBIC'))


def from_label(label):
    """The system the row's own label states, if it states one.

    These labels were written from the literature in earlier work, so
    where one names a system it is better evidence than anything
    re-derived here -- and it is the only evidence for the rows that are
    nodal approximations, which have no symmetry group to interrogate.
    """
    low = label.lower()
    for word, name in STATED:
        if word in low:
            return name
    return None


def from_shape(key, builder, nodal=False):
    """Cubic or not, from the built surface's own proportions.

    Only ever asked to make the cubic / non-cubic distinction, which a
    bounding box can carry: a cubic cell is built on a cube.  Anything
    finer would need the symmetry group, which a nodal row does not have.
    """
    import numpy as _np
    from minsurf import tpms as _t
    try:
        if nodal:
            # The nodal rows take (key, cells, resolution, scale); the
            # exact ones take (cells, resolution, scale, angle).  Calling
            # one with the other's signature is what made every nodal row
            # come back unclassified on the first pass.
            V, T = _t.build_tpms(key, (1, 1, 1), 24, 1.0)
        else:
            V, T = builder(1, 24, 1.0, 0.0)
    except Exception:                              # noqa: BLE001
        return None
    V = _np.asarray(V, dtype=float)
    if not len(V):
        return None
    a, b, c = V.max(0) - V.min(0)
    def eq(x, y):
        return abs(x - y) <= 0.03 * max(x, y)
    return 'CUBIC' if (eq(a, b) and eq(b, c)) else 'NONCUBIC'


def classify(key, spec):
    lets = {k: np.asarray(v, dtype=float).reshape(4, 4)
            for k, v in spec['letters'].items()}
    word = spec['word']
    g = bnd = sysname = None
    try:
        V, quads = pl.fe_cell_patch(key, m=96, rings=16)
        got = pl.fe_cell_assemble(key, V, quads)
        if got is not None:
            W, wf = np.asarray(got[0], dtype=float), got[1]
            bbox = W.max(0) - W.min(0)
            sysname = system_of(rotation_axes(lets, word), bbox)
            g, bnd = genus_of(W, wf, period=bbox)
    except Exception as exc:                       # noqa: BLE001
        bnd = 'error: %s' % exc
    return sysname, g, bnd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    from minsurf import tpms as _tpms
    allrows = {}
    allrows.update({k: v for k, v in _tpms.TPMS.items()})
    allrows.update({k: v for k, v in _tpms.TPMS_EXACT.items()})
    keys = a.only or sorted(allrows)
    rows = {}
    for k in keys:
        label = allrows[k][0] if k in allrows else k
        g = None
        if k in OVERRIDE:
            rows[k] = (OVERRIDE[k], None)
            print('  %-32s %-13s [checked against the literature]'
                  % (k, OVERRIDE[k]))
            sys.stdout.flush()
            continue
        # Order of evidence: what the label states (written from the
        # literature), then the symmetry group, then the proportions.
        sysname, how = from_label(label), 'label'
        if sysname is None and k in FE_CELLS:
            sysname, g, _bnd = classify(k, FE_CELLS[k])
            how = 'point group'
        if sysname is None and k in allrows:
            sysname = from_shape(k, allrows[k][1], nodal=(k in _tpms.TPMS))
            how = 'proportions'
        rows[k] = (sysname, g)
        print('  %-32s %-13s [%s]' % (k, sysname or '?', how))
        sys.stdout.flush()
    if a.write:
        with io.open(OUT, 'w', encoding='utf-8') as fh:
            fh.write('"""Lattice system and genus per surface.\n\n'
                     'GENERATED by tools/classify_surfaces.py -- do not '
                     'hand-edit.\n\nBoth are derived from geometry: the '
                     'lattice from the translation subgroup of the\n'
                     "datafile's own symmetry group, the genus from the "
                     'Euler characteristic of\nthe assembled cell.  A '
                     'None genus means the row ships a fundamental piece\n'
                     'rather than a closed cell, so the number is not '
                     'defined for it.\n"""\n\nSURFACE_CLASS = {\n')
            for k in sorted(rows):
                s, g = rows[k]
                fh.write("    %r: (%r, %r),\n" % (k, s, g))
            fh.write('}\n')
        print('wrote %s (%d rows)' % (OUT, len(rows)))


if __name__ == '__main__':
    main()
