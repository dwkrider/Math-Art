"""Lift each minimal surface's DEFINING DATA out of the shipped code.

An algebraic record carries the polynomial that defines its surface.  A
minimal-periodic record carried none of the equivalent -- `gyroid.json`
said "mode: nodal, level 0, cubic lattice" and never said WHICH nodal
surface, though the equation sits in `minsurf/tpms.py` a few lines from
where the record is built.  Two of the three kinds of row here have such
data and neither reached the database:

  * A NODAL row is a trigonometric polynomial, exactly analogous to the
    algebraic polynomial: the gyroid is
    `sin x cos y + sin y cos z + sin z cos x = 0`.
  * An EXACT WEIERSTRASS row is defined by its Weierstrass data on a
    domain: branch points with exponents, an integration constant, the
    torus modulus, and the rectangle in the parameter plane that is
    integrated.  `hexagonal._SPECS` already holds all of it.

  (The third kind, an Evolver cell, already stores its contour,
  generator matrices and assembly word in `definition.evolver_cell`.)

EXTRACTED, NOT TRANSCRIBED.  The database has a standing rule about
this, and `barth-decic` is where it is written down: its `polynomial` is
null with the note that "an unverified transcription would silently
define a different surface".  Hand-copying formulas is also what
produced several wrong surfaces here before the datafile pipeline
existed.  So the nodal expression is read from the function's own source
and normalised, and the Weierstrass data is obtained by CALLING each
spec's `terms(a, tau)` at its own moduli -- the same call the builder
makes.  Where a row resists that, the field is left null with a note
saying so, rather than filled with something plausible.

    python tools/extract_definitions.py           # print what it finds
    python tools/extract_definitions.py --write   # write surface_defs.py
"""

import argparse
import ast
import inspect
import io
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'math_art'))

import numpy as np                                # noqa: E402

from minsurf import tpms as _tpms                 # noqa: E402
from minsurf import hexagonal as _hex             # noqa: E402

OUT = os.path.join(HERE, 'surfdb', 'surface_defs.py')

# `np.sin(x)` reads as code; `sin(x)` reads as mathematics.  Only the
# call prefix is dropped -- the structure of the expression is left
# exactly as the shipped function has it, so the result can still be
# compared against the source by eye.
_NP = re.compile(r'\bnp\.')


def nodal_expression(fn):
    """The trig polynomial a nodal builder evaluates, as text.

    Returns (expression, None) or (None, why-not).
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        return None, 'source unavailable: %s' % exc
    try:
        # `textwrap.dedent`, never `inspect.cleandoc`: cleandoc is for
        # DOCSTRINGS and strips the leading indentation off every line
        # after the first, which turns a function body into a syntax
        # error.  It silently cost every nodal row on the first run.
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError as exc:
        return None, 'unparsable: %s' % exc
    body = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    if len(body) != 1:
        return None, ('%d return statements -- not a single expression'
                      % len(body))
    try:
        text = ast.unparse(body[0].value)
    except Exception as exc:                       # noqa: BLE001
        return None, 'could not unparse: %s' % exc
    text = _NP.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, None


def weierstrass_data(key, spec):
    """Branch points, constant, modulus and domain for an exact row.

    `terms` is a function of the spec's own moduli, so it is CALLED
    rather than read: that is what the builder does, and it is the only
    way to get the numbers the surface is actually integrated from.
    """
    out = {}
    tau = spec.get('tau')
    a = spec.get('a')
    if tau is not None:
        out['modulus_tau'] = [float(np.real(tau)), float(np.imag(tau))]
    if a is not None:
        out['moduli_a'] = float(np.real(a))
    terms = spec.get('terms')
    if callable(terms) and tau is not None and a is not None:
        try:
            got = terms(a, tau)
            out['branch_points'] = [
                {'point': [float(np.real(p)), float(np.imag(p))],
                 'exponent': float(np.real(e))} for p, e in got]
        except Exception as exc:                   # noqa: BLE001
            out['branch_points'] = None
            out['branch_points_note'] = (
                'not extracted: calling the spec\'s own `terms(a, tau)` '
                'raised %s' % exc)
    const = spec.get('const')
    if const is not None:
        out['constant'] = [float(np.real(const)), float(np.imag(const))]
    # The domain: the rectangle in the parameter plane that is
    # integrated.  `ylim` is often a function of tau, so it is called
    # the same way `terms` is.
    xlim, ylim = spec.get('xlim'), spec.get('ylim')
    dom = {}
    if xlim is not None:
        dom['x'] = [float(xlim[0]), float(xlim[1])]
    if callable(ylim) and tau is not None:
        try:
            lo, hi = ylim(tau)
            dom['y'] = [float(np.real(lo)), float(np.real(hi))]
        except Exception as exc:                   # noqa: BLE001
            dom['y_note'] = 'not extracted: %s' % exc
    elif ylim is not None:
        dom['y'] = [float(ylim[0]), float(ylim[1])]
    if dom:
        out['domain'] = dom
    if spec.get('trigroup'):
        out['triangle_group'] = list(spec['trigroup'])
    return out or None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    nodal, weier, skipped = {}, {}, []
    for key, row in sorted(_tpms.TPMS.items()):
        expr, why = nodal_expression(row[1])
        if expr:
            nodal[key] = expr
        else:
            skipped.append(('nodal', key, why))
    for key, spec in sorted(_hex._SPECS.items()):
        got = weierstrass_data(key, spec)
        if got:
            weier[key] = got
        else:
            skipped.append(('weierstrass', key, 'spec carries no data'))

    print('nodal polynomials extracted: %d' % len(nodal))
    for k in sorted(nodal)[:6]:
        print('   %-10s %s' % (k, nodal[k][:88]))
    print('\nWeierstrass data extracted: %d' % len(weier))
    for k in sorted(weier)[:6]:
        w = weier[k]
        print('   %-12s tau=%s branch=%s domain=%s'
              % (k, w.get('modulus_tau'),
                 len(w['branch_points']) if w.get('branch_points') else None,
                 w.get('domain')))
    if skipped:
        print('\nnot extracted (%d):' % len(skipped))
        for kind, k, why in skipped[:10]:
            print('   %-12s %-14s %s' % (kind, k, why))

    if a.write:
        import json
        with io.open(OUT, 'w', encoding='utf-8') as fh:
            fh.write('"""Defining data for the minimal surfaces.\n\n'
                     'GENERATED by tools/extract_definitions.py -- do not '
                     'hand-edit.\n\nRead OUT OF the shipped builders rather '
                     'than transcribed into them:\nthe nodal expression is '
                     "the source of the function that is\nevaluated, and the "
                     "Weierstrass data comes from calling each spec's\n"
                     '`terms(a, tau)` at its own moduli.  A surface whose '
                     'data could not be\nobtained that way is absent here '
                     'rather than approximated.\n"""\n\nimport json\n\n')
            fh.write('NODAL_POLYNOMIAL = json.loads(r\'\'\'')
            fh.write(json.dumps(nodal, indent=1, sort_keys=True))
            fh.write("''')\n\nWEIERSTRASS = json.loads(r'''")
            fh.write(json.dumps(weier, indent=1, sort_keys=True))
            fh.write("''')\n")
        print('\nwrote %s (%d nodal, %d Weierstrass)'
              % (OUT, len(nodal), len(weier)))


if __name__ == '__main__':
    main()
