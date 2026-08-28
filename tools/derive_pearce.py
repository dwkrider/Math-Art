"""Derive Pearce solids from other Pearce solids and gate the results.

Runs `math_art/pearce_derive.py`'s plan (truncation / fission /
blunting of already-verified solids), validates every candidate with
the emitter's own `validate()` gate, and writes the survivors to
`tools/derived.pkl` in the resolver's record shape:

    {entry_number: (match_kind, source_tag, verts, faces)}

The source tag is the (class, modulus) kind tuple of the solid's own
edges -- the same shape entry #35 and #47 already use -- so the
emitter's net-shippability pass and the packing probe both keep
working (`pearce_net.mixed_net` accepts it directly).  Provenance (the
parent entry and operation) is printed here and recorded in
`pearce_derive.PLAN`.

The gate is IMPORTED FROM THE EMITTER, not copied: `validate()` is
extracted from tools/emit_pearce_data.py by AST so that running this
driver never executes the emitter script itself (which loads the
resolver's pickles and rewrites math_art/pearce_data.py).
"""
import ast
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MA = os.path.join(os.path.dirname(HERE), "math_art")
sys.path.insert(0, MA)

import pearce_net as pn                      # noqa: E402
import pearce_table as pt                    # noqa: E402
import pearce_derive as pdv                  # noqa: E402

BY_NUM = {r['number']: r for r in pt.TABLE}


def emitter_validate():
    """The emitter's `validate` (and its `_cart` helper), compiled from
    tools/emit_pearce_data.py without executing the script."""
    path = os.path.join(HERE, "emit_pearce_data.py")
    with open(path, "r") as fh:
        tree = ast.parse(fh.read(), filename=path)
    wanted = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "validate", "_cart"):
            wanted[node.name] = node
    if set(wanted) != {"validate", "_cart"}:
        raise RuntimeError("emit_pearce_data.py no longer defines "
                           "validate/_cart as top-level functions")
    mod = ast.Module(body=[wanted["_cart"], wanted["validate"]],
                     type_ignores=[])
    ns = {"pn": pn, "BY_NUM": BY_NUM}
    exec(compile(mod, path, "exec"), ns)
    return ns["validate"]


def main():
    validate = emitter_validate()
    parents = pdv.parents_from_data()
    got, why = pdv.derive_all(parents)

    accepted = {}
    print("== derivation report ==")
    for num in sorted(set(pdv.PLAN) | set(pdv.BLOCKED)):
        r = BY_NUM[num]
        tag = "#%-2d %s" % (num, r['name'])
        if num not in got:
            print("FAIL   %s\n       %s" % (tag, why.get(num, "?")))
            continue
        V, F, note = got[num]
        gate = validate(num, V, F, 'FULL', None)
        if gate is not None:
            print("REJECT %s\n       emitter gate: %s" % (tag, gate))
            continue
        op, src, _spec = pdv.PLAN[num]
        kinds = tuple(sorted(set(pdv.edge_kinds(V, F))))
        accepted[num] = ('FULL', kinds, list(V), list(F))
        print("OK     %s\n       %s of #%d; %s; edges %s"
              % (tag, op, src, note,
                 " + ".join("%d %s-%s" % (c, k[0], k[1])
                            for k, c in sorted(
                                pdv.edge_kinds(V, F).items()))))

    out = os.path.join(HERE, "derived.pkl")
    with open(out, "wb") as fh:
        pickle.dump(accepted, fh)
    print("wrote %s: %d solids (%s)" % (out, len(accepted),
                                        sorted(accepted)))


if __name__ == "__main__":
    main()
