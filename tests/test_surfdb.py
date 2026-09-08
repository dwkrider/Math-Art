"""Self-tests for the surface-database tooling (tools/surfdb/).

Headless and Blender-free, like the rest of tools/.  Run directly:

    python tests/test_surfdb.py

Each module under tools/surfdb/ owns a `_selftest()` that raises on
failure, matching the convention `tests/test_selftests.py` uses for
math_art/ modules.  The tooling lives under tools/ rather than math_art/,
so that runner does not discover it -- hence this file.

The interesting tests are the mathematical ones, and they are written as
REGRESSIONS for bugs this database actually shipped into and then caught:

  * surfdb.polynomial  -- a mistyped exponent must be rejected by the
    numerical oracle, and a correct conversion accepted up to a scalar.
  * surfdb.invariants  -- Scherk's surface must measure as minimal and a
    15%-perturbed version must not; Cayley's cubic must verify as Td and
    NOT as Oh; Barth's sextic must verify as icosahedral, which pins the
    5-fold axis to (phi, 1, 0) rather than (1, phi, 0).
  * surfdb.expr        -- `__import__('os')` must be rejected at PARSE
    time, not merely fail to evaluate.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

MODULES = ("expr", "views", "mapping", "polynomial", "registry", "curation",
           "invariants", "references", "sources", "tail", "crosscheck",
           "charts", "weierstrass", "wedata", "weextract", "parcharts",
           "algextract", "symderive", "describe",
           "nodal", "ferreol", "vmm", "published", "algsurf", "papers")

# `parcharts` was once listed here BEFORE it was written, so the suite
# failed on a module that did not exist -- a gate reporting a failure
# nobody can act on trains people to ignore it. It was removed with a
# note to re-add it in the same commit that added the module. The module
# now exists, so it is listed again, alongside `algextract`.


def main():
    import importlib
    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module("surfdb." + name)
        except Exception as exc:                      # noqa: BLE001
            # a listed module that cannot even import is a FAILURE of that
            # module, not a reason to abort the whole run unreported
            failed.append((name, exc))
            print("FAIL surfdb.%s (import): %s" % (name, exc))
            continue
        fn = getattr(mod, "_selftest", None)
        if fn is None:
            print("SKIP surfdb.%s (no _selftest)" % name)
            continue
        try:
            fn()
            extra = getattr(mod, "_selftest_parametric", None) or                 getattr(mod, "_selftest_complex", None)
            if extra is not None:
                extra()
        except Exception as exc:                      # noqa: BLE001
            failed.append((name, exc))
            print("FAIL surfdb.%s: %s" % (name, exc))
    if failed:
        print("\nRESULT: FAIL (%d of %d)" % (len(failed), len(MODULES)))
        return 1
    print("\nRESULT: OK (%d modules)" % len(MODULES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
