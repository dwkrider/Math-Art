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


# Enums that say how a shape is *presented* rather than which shape it
# is: a gallery of these would be the same object six times over.
PRESENTATION = {
    "style", "coloring", "colour_mode", "color_by", "color_mode",
    "output", "align", "material", "display", "handedness",
    "spline_type", "face_color", "interlace_mode", "rig", "bands",
}


def _candidates(op):
    """Enums on `op` that could plausibly be an undeclared selector."""
    mod, _, fn = op.partition('.')
    try:
        rna = getattr(getattr(bpy.ops, mod), fn).get_rna_type()
    except Exception:
        return ""
    out = []
    for p in rna.properties:
        if p.type != 'ENUM' or p.identifier in PRESENTATION:
            continue
        try:
            n = len(p.enum_items)
        except Exception:
            continue
        # 0 means a callback enum, which RNA will not resolve here --
        # those are two-level selectors and need a VARIANT_GROUP entry.
        if n >= 3:
            out.append(f"{p.identifier}({n})")
        elif n == 0:
            out.append(f"{p.identifier}(dynamic)")
    return ("   <-- possible selector: " + ", ".join(out)) if out else ""


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

    # An operator with no gallery is usually a single-form generator,
    # which needs none.  But a *missed* gallery looks identical in a
    # bare list, so say which of them still carry a multi-option enum
    # that might be a selector -- that is how the tight link's six
    # links and the Willmore modes were caught after a merge.
    no_gallery = sorted(set(ops) - declared - set(subjects.SKIP))
    if no_gallery:
        print(f"\n{len(no_gallery)} operators with no gallery declared:")
        for op in no_gallery:
            print(f"    {op}{_candidates(op)}")
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
