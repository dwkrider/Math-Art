"""Read Ken Brakke's Evolver datafiles and write what they define into
the surface database.

THE DATAFILES ARE NOT IN THIS REPO and must not be.  They are a local
mirror of kenbrakke.com; this script reads them where they sit and
records the DEFINING DATA -- the boundary contour, the symmetry
generator matrices, and the word that assembles the cell -- into the
matching record under `data/surfaces`, which is where a surface's
definition belongs.

The chain is therefore:

    .fe (external mirror)  ->  tools/surfdb/fecells.py  ->  records
                           ->  math_art/minsurf/fecells.py (shipped)

Run this only when the datafiles change or a new surface is added.  The
generated Python is not to be hand-edited: hand transcription of exactly
these numbers is what produced several wrong surfaces before this
pipeline existed.

    python tools/bake_fe_cells.py            # rewrite tools/surfdb/fecells.py
    python tools/bake_fe_cells.py --module   # rewrite the shipped module
                                             # FROM the database
"""

import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'math_art'))

from minsurf import fedata, plateau as pl      # noqa: E402

CURATION_OUT = os.path.join(HERE, 'surfdb', 'fecells.py')
MODULE_OUT = os.path.join(ROOT, 'math_art', 'minsurf', 'fecells.py')
RECORDS = os.path.join(ROOT, 'data', 'surfaces', 'surfaces')

# datafile -> record slug.  Only surfaces whose record is unambiguous;
# `cd.fe` and `disphenoid19.fe` have no matching record yet and are
# deliberately left out rather than filed under a guess.
SLUGS = {
    'CLP.fe': 'clp-exact',
    'I-8.fe': 'schoen-i8',
    'I-9.fe': 'schoen-i9',
    'IWP.fe': 'iwp-surface',
    'RII.fe': 'schoen-rii',
    'RIII.fe': 'schoen-riii',
    'dcell.fe': 'schwarz-d',
    'hcell.fe': 'h-exact',
    'neovius.fe': 'neovius-surface',
    'pcell.fe': 'schwarz-p',
}

PREFER = ['showcube', 'cube', 'full', 'layers', 'showcubelet', 'showsix',
          'showfour', 'showrhombic', 'stack8', 'stack6', 'stack4',
          'showcube_alt', 'seven', 'stack12']


def relax_from(fe, m=72, rings=12, iters=250):
    loops = fe.boundary_loops()
    if len(loops) == 1:
        poly = loops[0]
        lp = pl.resample_loop(np.vstack([poly, poly[:1]]), m)
        V, quads, fixed = pl.build_disk_grid(lp, rings)
    elif len(loops) == 2:
        a, b = loops
        la = pl.resample_loop(np.vstack([a, a[:1]]), m)
        lb = pl.resample_loop(np.vstack([b, b[:1]]), m)
        if float(np.mean(la[:, 2])) > float(np.mean(lb[:, 2])):
            la, lb = lb, la
        j = int(np.argmin(np.linalg.norm(lb - la[0], axis=1)))
        lb = np.roll(lb, -j, axis=0)
        V, quads, fixed = pl.build_annulus_grid(la, lb, rings)
    else:
        return None
    V = np.asarray(V, dtype=float)
    fixed = np.asarray(fixed, dtype=bool)
    T = np.asarray(pl._quads_to_tris(quads))
    V = np.asarray(pl.minimize_area(V.copy(), T, fixed, outer_iters=iters),
                   dtype=float)
    return V, [tuple(f) for f in quads], loops


def verify(V, quads, lets, word):
    mats = pl.eval_transform_expr(lets, word)
    if not (1 < len(mats) <= 256):
        return None
    V = np.asarray(V, dtype=float)
    pts = np.concatenate([V @ M[:3, :3].T + M[:3, 3] for M in mats])
    nV = len(V)
    faces = []
    for j, M in enumerate(mats):
        flip = float(np.linalg.det(M[:3, :3])) < 0.0
        for f in quads:
            g = tuple(i + j * nV for i in f)
            faces.append(g[::-1] if flip else g)
    span = float(np.max(pts.max(0) - pts.min(0))) or 1.0
    W, wf = pl._weld_points(pts, faces, 1e-4 * span)
    dup, over, comps = pl._orbit_defects(W, wf)
    bb = np.asarray(W).max(0) - np.asarray(W).min(0)
    if dup or over or comps != 1 or float(np.min(bb)) <= 1e-6:
        return None
    return len(mats)


def harvest():
    out = {}
    for fn, slug in sorted(SLUGS.items()):
        path = os.path.join(fedata.MIRROR_DOWNLOADS, fn)
        if not os.path.exists(path):
            print("  %-14s datafile absent, skipped" % fn)
            continue
        fe = fedata.read(path)
        got = relax_from(fe)
        if got is None:
            print("  %-14s boundary is %d loops, skipped"
                  % (fn, len(fe.boundary_loops())))
            continue
        V, quads, loops = got
        lets = fe.letters()
        order = ([k for k in PREFER if k in fe.words]
                 + sorted(k for k in fe.words if k not in PREFER))
        pick = None
        for name in order:
            word = fe.words[name]
            if any(ch not in lets for ch in word):
                continue
            n = verify(V, quads, lets, word)
            if n:
                pick = (name, word, n)
                break
        if pick is None:
            print("  %-14s no word assembles, skipped" % fn)
            continue
        name, word, n = pick
        out[slug] = {
            'source': fn,
            'command': name,
            'word': word,
            'copies': n,
            'loops': [[[float(c) for c in p] for p in lp] for lp in loops],
            'generators': {c: [float(x) for x in np.ravel(lets[c])]
                           for c in sorted(set(word))},
        }
        print("  %-14s -> %-16s %s=%r  %d copies"
              % (fn, slug, name, word, n))
    return out


def write_curation(data):
    with io.open(CURATION_OUT, 'w', encoding='utf-8') as fh:
        fh.write('"""Cell definitions read out of Surface Evolver datafiles.\n\n'
                 'GENERATED by tools/bake_fe_cells.py -- do not hand-edit.\n\n'
                 "The datafiles themselves are NOT in this repo; they are a\n"
                 'local mirror of kenbrakke.com.  What a surface IS -- its\n'
                 'boundary contour, its symmetry generators and the word that\n'
                 'assembles its cell -- belongs in the surface record, so the\n'
                 'build merges this into `definition.evolver_cell`.\n"""\n\n')
        fh.write("FE_CELLS = ")
        fh.write(json.dumps(data, indent=1, sort_keys=True))
        fh.write("\n\n\ndef facts_for(slug):\n"
                 '    """`definition` fragment for a slug, or {}."""\n'
                 "    cell = FE_CELLS.get(slug)\n"
                 "    if not cell:\n"
                 "        return {}\n"
                 "    return {'definition': {'evolver_cell': cell}}\n")
    print("wrote %s (%d surfaces)" % (CURATION_OUT, len(data)))


def write_module():
    """Regenerate the shipped module FROM the database records."""
    cells = {}
    for root, _d, files in os.walk(RECORDS):
        for fn in files:
            if not fn.endswith('.json'):
                continue
            rec = json.load(io.open(os.path.join(root, fn), encoding='utf-8'))
            cell = (rec.get('definition') or {}).get('evolver_cell')
            if cell:
                cells[rec['slug']] = (rec['name'], cell)
    with io.open(MODULE_OUT, 'w', encoding='utf-8') as fh:
        fh.write('"""Evolver cells, generated FROM the surface database.\n\n'
                 'GENERATED by `python tools/bake_fe_cells.py --module` -- do\n'
                 'not hand-edit.  The defining data lives in the records under\n'
                 '`data/surfaces`; this module exists only so the shipped\n'
                 'extension carries the numbers without needing the database\n'
                 'or the original datafiles.\n"""\n\n'
                 'import numpy as np\n\nFE_CELLS = {\n')
        for slug in sorted(cells):
            name, c = cells[slug]
            key = slug.upper().replace('-', '_')
            fh.write("    %r: {\n        'title': %r,\n        'slug': %r,\n"
                     % (key, name, slug))
            fh.write("        'source': %r,\n        'word': %r,\n"
                     "        'copies': %d,\n"
                     % (c['source'], c['word'], c['copies']))
            fh.write("        'loops': (\n")
            for lp in c['loops']:
                flat = [x for p in lp for x in p]
                fh.write("            np.array([%s]).reshape(-1, 3),\n"
                         % ", ".join("%.12g" % v for v in flat))
            fh.write("        ),\n        'letters': {\n")
            for ch in sorted(c['generators']):
                fh.write("            %r: np.array([%s]).reshape(4, 4),\n"
                         % (ch, ", ".join("%.12g" % v
                                          for v in c['generators'][ch])))
            fh.write("        },\n    },\n")
        fh.write("}\n")
    print("wrote %s (%d cells from the database)" % (MODULE_OUT, len(cells)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--module', action='store_true',
                    help="regenerate the shipped module from the database")
    args = ap.parse_args()
    if args.module:
        write_module()
        return
    print("reading datafiles from %s" % fedata.MIRROR_DOWNLOADS)
    write_curation(harvest())


main()
