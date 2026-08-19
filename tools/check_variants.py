"""Check every declared documentation gallery against the operators.

`tools/subjects.py` names, per generator, the property whose options
make the "Variants" grid on that generator's doc page.  A name there is
a claim about the operator, and a wrong one fails quietly -- an empty
grid, or a grid missing the options added since.  This script resolves
every declared gallery for real and reports what it found, so the claim
is checkable in seconds rather than after an hour of rendering.

Run:

    blender --background --factory-startup --python tools/check_variants.py

Report only the problems:

    ... --python tools/check_variants.py -- --quiet

Exits non-zero if any declared gallery fails to resolve, which is what
makes it usable as a pre-render gate.
"""
import os
import sys

import bpy

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "tools"))

import math_art                                            # noqa: E402
math_art.register()

from math_art import menu_defs                             # noqa: E402
import subjects                                            # noqa: E402


def main(argv):
    quiet = "--quiet" in argv
    ops = menu_defs.unique_ops()
    declared = (set(subjects.VARIANT_SELECTOR) | set(subjects.VARIANT_GROUP)
                | set(subjects.VARIANT_EXTRA))

    # A gallery declared for an operator that is in no menu is dead
    # weight: nothing will ever render or link it.
    orphan = sorted(declared - set(ops))
    bad, total = [], 0
    rows = []
    for op in ops:
        if op not in declared:
            continue
        try:
            vs = subjects.variants_for(op)
        except Exception as e:
            bad.append((op, repr(e)))
            continue
        total += len(vs)
        groups = sorted({v[3] for v in vs if v[3] is not None})
        rows.append((op, len(vs), groups))
        if len(vs) < 2:
            bad.append((op, f"only {len(vs)} variant(s)"))

    if not quiet:
        for op, n, groups in rows:
            g = f"  [{', '.join(groups)}]" if groups else ""
            print(f"  {op:<42} {n:>3}{g}")

    print(f"\n{len(rows)} galleries, {total} variant renders in all")

    no_gallery = sorted(set(ops) - declared - set(subjects.SKIP))
    if no_gallery:
        print(f"\n{len(no_gallery)} operators with no gallery declared "
              f"(a single-form generator needs none):")
        for op in no_gallery:
            print("   ", op)
    if orphan:
        print(f"\n{len(orphan)} galleries for non-menu operators:")
        for op in orphan:
            print("   ", op)
    if bad:
        print(f"\nFAILED {len(bad)}:")
        for op, why in bad:
            print(f"    {op}: {why}")
        return 1
    print("\nRESULT: OK")
    return 0


sys.exit(main(sys.argv[sys.argv.index("--") + 1:]
              if "--" in sys.argv else []))
