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

# ADJOINT DATAFILES
#
# Half of Brakke's collection defines its surface as the CONJUGATE of the
# disk spanning a pinned polygon, and the conjugate is only the first
# step: each file then carries a short script that moves the result into
# place before declaring which boundary arc lies on which mirror plane.
# `minsurf.plateau.fe_adjoint_patch` runs that script; what is harvested
# here is the conjugate's own boundary, because a contour is all the
# shipped code needs to span the patch again.
#
# WHAT IS GATED, and why it is not the topology.  A wrong subgroup, a
# wrong family member and a wrong Gauss map have each, in this project,
# produced a cell that welded into one clean sheet and was still wrong.
# So a cell ships only if it also agrees with SURFACE EVOLVER'S OWN
# framed adjoint -- area within `AREA_TOL` of the number Evolver reports
# for the same datafile -- or, for the nine datafiles Evolver did not
# finish inside the time limit, if its boundary lands on the planes the
# datafile declares to within `RESID_TOL` of the patch's own size.  That
# residual is the intrinsic form of the same question: the conjugate of a
# minimal surface is minimal, so if the framing put it in the right place
# its boundary is ALREADY on those planes, and no projection is needed to
# get it there.
#
# The measured numbers are recorded beside each cell rather than kept in
# a scratch file, so the claim is auditable from the database alone.

import argparse
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'math_art'))
sys.path.insert(0, HERE)

from minsurf import fedata, plateau as pl      # noqa: E402
from surfdb.evolver_truth import EVOLVER_ADJOINT      # noqa: E402

CURATION_OUT = os.path.join(HERE, 'surfdb', 'fecells.py')
MODULE_OUT = os.path.join(ROOT, 'math_art', 'minsurf', 'fecells.py')
RECORDS = os.path.join(ROOT, 'data', 'surfaces', 'surfaces')

# datafile -> record slug, for the files that pin their contour outright
# (no conjugation step).  As with ADJOINT below, every pairing is read
# off the datafile's own header comment: `N14.fe` calls itself
# `CPDadj.fe` in its first line, so the filename alone would have filed
# it under the wrong surface entirely.
SLUGS = {
    'CLP.fe': 'clp-exact',
    'CScell.fe': 'fischer-koch-cs',       # "minimal surface C(S) from
    'CYcell.fe': 'fischer-koch-cy',       #  Fischer and Koch"
    'I-6.fe': 'schoen-i6',
    'I-8.fe': 'schoen-i8',
    'I-9.fe': 'schoen-i9',
    'IWP.fe': 'iwp-surface',
    'N14.fe': 'neovius-n14',
    'N26.fe': 'neovius-n26',
    'N38.fe': 'neovius-n38',
    'RII.fe': 'schoen-rii',
    'RIII.fe': 'schoen-riii',
    'Scell.fe': 'fk-s-surface',
    'cd.fe': 'cd-surface',
    'dcell.fe': 'schwarz-d',
    'hcell.fe': 'h-exact',
    'neovius.fe': 'neovius-surface',
    'pcell.fe': 'schwarz-p',
    'ycell.fe': 'fk-y-surface',
}

# `disphenoid19.fe` -- "fundamental cell for Schoen's complementary D
# minimal surface, genus 19" -- assembles, but the database has no
# genus-19 disphenoid record and the C(D) record is `cd.fe`'s.  Left out
# rather than filed under a guess.

PREFER = ['showcube', 'cube', 'full', 'layers', 'showcubelet', 'showsix',
          'showfour', 'showrhombic', 'stack8', 'stack6', 'stack4',
          'showcube_alt', 'seven', 'stack12']

RESID_TOL = 0.15        # boundary-to-declared-plane, as a fraction of span
AREA_TOL = 0.10         # patch area against Evolver's own framed adjoint

# Adjoint datafile -> record slug, and the title to give a surface that
# has no record yet.  Every pairing below is read off the datafile's own
# header comment ("Adjoint of Schoen's C41-4(P) surface."), never guessed
# from the filename: `manta51adj.fe` calls itself the genus 57 manta in
# its first line, and `hybrid-1adj.fe` announces itself as `hybrid-1.fe`.
ADJOINT = {
    'FRDadj.fe': ('frd-surface', "Schoen F-RD", None),
    'GW5adj.fe': ('schoen-gw', "Schoen GW (graphite-wurtzite)", None),
    'HRHTadj.fe': ('schoen-hybrid-ht-h2r', "Schoen H''-R | H'-T", None),
    'HRadj.fe': ('weber-h2r', "Schoen H''-R", None),
    'HTTRadj.fe': (None, "Schoen H'-T | T'-R'", 'HT_TR'),
    'HTadj.fe': ('schoen-h-t', "Schoen H'-T", None),
    'SSadj.fe': ('schoen-s-s', "Schoen S'-S''", None),
    'TRHTadj.fe': ('schoen-hybrid-tr-ht', "Schoen T'-R' | H'-T", None),
    'TRadj.fe': ('weber-trr', "Schoen T'-R'", None),
    'batwing41adj.fe': ('schoen-batwing-41', "Schoen C41-4(P)", None),
    'batwing57adj.fe': ('schoen-batwing-57', "Schoen C57-4(P)", None),
    'batwingadj.fe': ('schoen-batwing', "Schoen Batwing", None),
    'c21padj.fe': (None, "Schoen C21(P)", 'C21P'),
    'c27padj.fe': ('schoen-c27-p', "Schoen C27(P)", None),
    'c33padj.fe': ('schoen-c33-p', "Schoen C33(P)", None),
    'c39padj.fe': ('schoen-c39-p', "Schoen C39(P)", None),
    'c45padj.fe': ('schoen-c45-p', "Schoen C45(P)", None),
    'disphenoid31adj.fe': ('disphenoid-family-a-genus-31', "Disphenoid 31", None),
    'disphenoid35adj.fe': ('disphenoid-family-b-genus-35', "Disphenoid 35", None),
    'disphenoid43adj.fe': ('disphenoid-family-a-genus-43', "Disphenoid 43", None),
    'disphenoid51adj.fe': ('disphenoid-family-b-genus-51', "Disphenoid 51", None),
    'disphenoid55adj.fe': ('disphenoid-family-a-genus-55', "Disphenoid 55", None),
    'disphenoid67adj.fe': ('disphenoid-family-b-genus-67', "Disphenoid 67", None),
    'hexplane1adj.fe': (None, "Hexplane 1", 'HEXPLANE1'),
    'hexplane2adj.fe': (None, "Hexplane 2", 'HEXPLANE2'),
    'hexplane3adj.fe': (None, "Hexplane 3", 'HEXPLANE3'),
    'hexplane4adj.fe': (None, "Hexplane 4", 'HEXPLANE4'),
    'hexplane5adj.fe': (None, "Hexplane 5", 'HEXPLANE5'),
    'hybrid-1adj.fe': ('schoen-hybrid-1', "Schoen Hybrid P | F-RD", None),
    'manta35adj.fe': ('schoen-manta-genus-35', "Schoen Manta (genus 35)", None),
    'manta51adj.fe': ('schoen-manta-genus-51', "Schoen Manta (genus 57)", None),
    'mantaadj.fe': ('schoen-manta-genus-19', "Schoen Manta", None),
    'octoadj.fe': ('octo-surface', "Schoen O,C-TO", None),
    'pbatadj.fe': ('brakke-pseudo-batwing', "Brakke Pseudo-Batwing", None),
    'pssadj.fe': ('schoen-hybrid-ss-p', "Schoen P | S'-S''", None),
    's12adj.fe': (None, "Schoen p.12 Surface", 'SCHOEN12'),
    's14adj.fe': (None, "Schoen p.14 Surface", 'SCHOEN14'),
    'triplane0adj.fe': (None, "Triplane 0", 'TRIPLANE0'),
    'triplane1adj.fe': (None, "Triplane 1", 'TRIPLANE1'),
    'triplane2adj.fe': (None, "Triplane 2", 'TRIPLANE2'),
    'triplane3adj.fe': (None, "Triplane 3", 'TRIPLANE3'),
    'triplane4adj.fe': (None, "Triplane 4", 'TRIPLANE4'),
    'triplane5adj.fe': (None, "Triplane 5", 'TRIPLANE5'),
}


FLAT_TOL = 0.02      # a patch this planar has not been solved, only spanned


def is_flat(V):
    """Is this patch essentially the flat disk that spans its boundary?

    The detector for a whole class of silent failure.  Four datafiles --
    Schwarz P, Schwarz D, Neovius and Schoen's I-WP -- pin no boundary
    edge at all: they hand you a FLAT quadrilateral (pcell's four corners
    all have x = 0.5) and expect the solve to curve it while every edge
    slides in its own mirror plane, holding the volume the `bodies`
    section fixes.  Spanning it as a contour hands the flat square
    straight back, and 48 copies of a flat square still weld into one
    clean closed sheet and pass every topological check.  It is a
    faceted star, and it shipped.
    """
    V = np.asarray(V, dtype=float)
    if len(V) < 4:
        return True
    _u, s, _vt = np.linalg.svd(V - V.mean(0), full_matrices=False)
    return float(s[2] / max(s[0], 1e-30)) < FLAT_TOL


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
    # Honour the datafile's own boundary conditions rather than pinning
    # the polygon it happens to start from -- see `plateau.fe_slide_planes`.
    got = pl.fe_pinned_patch(fe, m=m, rings=rings, iters=iters)
    if got is None:
        return None
    return got[0], got[1], loops


def verify(V, quads, lets, word):
    """(copies, weld tolerance) if this word closes the cell, else None.

    The tolerance is searched rather than fixed at 1e-4: that single
    value fused sheets that merely pass close together, which is what
    made I-6's cell -- perfectly sound at 1e-8 -- look broken.
    """
    mats = pl.eval_transform_expr(lets, word)
    if len(mats) > 2048:
        return None
    mats = pl.dedupe_placements(V, mats)
    if not (1 < len(mats) <= 256):
        return None
    for tol in pl.FE_WELD_LADDER:
        W, wf = pl.assemble_orbit(V, quads, mats, tol)
        if pl.fe_orbit_ok(W, wf):
            return len(mats), tol
    return None


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
        if is_flat(V):
            print("  %-14s HELD: patch stayed flat -- its boundary is free "
                  "on planes and the volume-constrained solve is not "
                  "converging yet" % fn)
            continue
        lets = fe.letters()
        words = pl.fe_words_with_fallback(fe)
        order = ([k for k in PREFER if k in words]
                 + sorted(k for k in words if k not in PREFER))
        pick = None
        for name in order:
            word = words[name]
            if any(ch not in lets for ch in word):
                continue
            got_v = verify(V, quads, lets, word)
            if got_v:
                pick = (name, word) + got_v
                break
        if pick is None:
            print("  %-14s no word assembles, skipped" % fn)
            continue
        name, word, n, tol = pick
        out[slug] = {
            'source': fn,
            'record_slug': slug,
            'command': name,
            'word': word,
            'copies': n,
            'weld_tol': tol,
            'loops': [[[float(c) for c in p] for p in lp] for lp in loops],
            'generators': {c: [float(x) for x in np.ravel(lets[c])]
                           for c in sorted(set(word))},
        }
        print("  %-14s -> %-20s %-14s %3d copies (weld %.0e)"
              % (fn, slug, "%s=%s" % (name, word), n, tol))
    return out


def slug_for(title):
    """The slug `surfdb_build` will derive from a row title."""
    s = title.lower()
    # Titles here are ASCII (H''-R, not a prime glyph); the split
    # below drops every non-alphanumeric anyway, so this only has to
    # agree with what `surfdb_build` does to a row label.
    return '-'.join(p for p in re.split(r'[^a-z0-9]+', s) if p)


def harvest_adjoint(m=96, rings=16, iters=400):
    """Run every adjoint datafile and keep the cells that check out."""
    out, held = {}, []
    for fn in sorted(ADJOINT):
        slug, title, row_key = ADJOINT[fn]
        path = os.path.join(fedata.MIRROR_DOWNLOADS, fn)
        if not os.path.exists(path):
            print("  %-19s datafile absent, skipped" % fn)
            continue
        fe = fedata.read(path)
        got = pl.fe_adjoint_patch(fe, m=m, rings=rings, iters=iters)
        if got is None:
            held.append((fn, "no patch"))
            continue
        V, quads, lets, env, loops = got
        resid = float(env.get('_plane_residual', 1.0))
        T = np.asarray(pl._quads_to_tris(quads))
        area = float(pl.mesh_area(np.asarray(V), T))
        truth = EVOLVER_ADJOINT.get(fn)
        if truth is not None:
            ratio = area / truth['area']
            why = "area %.3f x Evolver" % ratio
            ok = abs(ratio - 1.0) <= AREA_TOL
        else:
            ratio, why = None, "residual %.1e (Evolver did not finish)" % resid
            ok = resid <= RESID_TOL
        if not ok:
            held.append((fn, why))
            continue
        if resid > RESID_TOL:
            held.append((fn, "residual %.1e" % resid))
            continue
        pick = None
        for name in pl.fe_word_order(fe.words):
            word = fe.words[name]
            if any(ch not in lets for ch in word):
                continue
            mats = pl.eval_transform_expr(lets, word)
            if not (1 < len(mats) <= 256):
                continue
            for tol in pl.FE_WELD_LADDER:
                W, wf = pl.assemble_orbit(V, quads, mats, tol)
                if pl.fe_orbit_ok(W, wf):
                    pick = (name, word, len(mats), tol)
                    break
            if pick:
                break
        if pick is None:
            held.append((fn, "no word assembles"))
            continue
        name, word, ncopy, tol = pick
        # Two names, because they answer two questions.  `key` is the
        # generator row this cell becomes -- in the house style of the
        # rows beside it (`SS`, `H2R`, `TR`), not a slug.  `record_slug`
        # is the database record it belongs to, which for a surface the
        # database already knows is that record, and for a new one is
        # whatever `surfdb_build` will derive from the row.
        key = slug or row_key
        record = slug or ('%s-exact' % row_key.lower().replace('_', '-'))
        out[key] = {
            'source': fn,
            'title': title,
            'record_slug': record,
            'route': 'adjoint',
            # Whether this cell should also become a GENERATOR row.  It
            # should only when the surface has no generator already:
            # Schoen's GW ships from an exact Weierstrass row, and for
            # that one the cell is definition data for the record, not a
            # second way to draw the same surface in the Add menu.
            'new_row': slug is None,
            'command': name,
            'word': word,
            'copies': ncopy,
            'weld_tol': tol,
            'plane_residual': round(resid, 6),
            'patch_area': round(area, 6),
            'evolver_area': (round(truth['area'], 6) if truth else None),
            'loops': [[[float(c) for c in p] for p in lp] for lp in loops],
            'generators': {c: [float(x) for x in np.ravel(lets[c])]
                           for c in sorted(set(word))},
        }
        print("  %-19s -> %-32s %s=%r %3d copies  %s"
              % (fn, key, name, word, ncopy, why))
    for fn, why in held:
        print("  %-19s HELD: %s" % (fn, why))
    print("  %d cells, %d held" % (len(out), len(held)))
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
        # Emitted as JSON and parsed at import rather than written as a
        # Python literal: the table carries booleans and nulls, and
        # `json.dumps` spells those `true` and `null`, which are a
        # NameError the moment the module is imported.
        fh.write("import json\n\nFE_CELLS = json.loads(r'''")
        fh.write(json.dumps(data, indent=1, sort_keys=True))
        fh.write("''')\n\n"
                 "# The table is keyed by GENERATOR ROW; records are looked\n"
                 "# up by the slug each row's record carries, which for the\n"
                 "# surfaces the database already knew is its existing slug.\n"
                 "_BY_SLUG = {c.get('record_slug', k): c\n"
                 "            for k, c in FE_CELLS.items()}\n"
                 "\n\ndef facts_for(slug):\n"
                 '    """`definition` fragment for a slug, or {}."""\n'
                 "    cell = _BY_SLUG.get(slug)\n"
                 "    if not cell:\n"
                 "        return {}\n"
                 "    return {'definition': {'evolver_cell': cell}}\n")
    print("wrote %s (%d surfaces)" % (CURATION_OUT, len(data)))


def write_module():
    """Regenerate the shipped module FROM the database records.

    With one deliberate exception, because the two feed each other: a
    surface the database has no record for gets that record from the
    generator row, and the generator row comes from this module.  So a
    freshly harvested cell is taken from the bake the first time round,
    and from its record on every pass after that -- the record stays
    authoritative wherever one exists.
    """
    cells, by_slug = {}, {}
    from surfdb.fecells import FE_CELLS as _BAKED
    for key, cell in sorted(_BAKED.items()):
        cells[key] = (cell.get('title') or key, cell)
        by_slug[cell.get('record_slug', key)] = key
    for root, _d, files in os.walk(RECORDS):
        for fn in files:
            if not fn.endswith('.json'):
                continue
            rec = json.load(io.open(os.path.join(root, fn), encoding='utf-8'))
            cell = (rec.get('definition') or {}).get('evolver_cell')
            if cell:
                key = by_slug.get(rec['slug'], rec['slug'])
                cells[key] = (cell.get('title') or rec['name'], cell)
    with io.open(MODULE_OUT, 'w', encoding='utf-8') as fh:
        fh.write('"""Evolver cells, generated FROM the surface database.\n\n'
                 'GENERATED by `python tools/bake_fe_cells.py --module` -- do\n'
                 'not hand-edit.  The defining data lives in the records under\n'
                 '`data/surfaces`; this module exists only so the shipped\n'
                 'extension carries the numbers without needing the database\n'
                 'or the original datafiles.\n"""\n\n'
                 'import numpy as np\n\nFE_CELLS = {\n')
        for src in sorted(cells):
            name, c = cells[src]
            key = src.upper().replace('-', '_')
            fh.write("    %r: {\n        'title': %r,\n        'slug': %r,\n"
                     % (key, name, c.get('record_slug', src)))
            fh.write("        'source': %r,\n        'word': %r,\n"
                     "        'copies': %d,\n        'tol': %g,\n"
                     "        'new_row': %r,\n"
                     % (c['source'], c['word'], c['copies'],
                        c.get('weld_tol', 1e-4), bool(c.get('new_row'))))
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
    data = harvest()
    print("adjoint datafiles:")
    data.update(harvest_adjoint())
    write_curation(data)


main()
