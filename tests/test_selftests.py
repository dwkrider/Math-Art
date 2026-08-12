# Headless self-test runner for the math_art generator modules.
#
# The generator modules each expose a `_selftest()` (a few use the older
# name `_self_test()`) that exercises their pure-Python numeric core
# WITHOUT Blender -- the check that used to live in each module's
# `if __name__ == "__main__":` block before those were removed for the
# Blender extension submission.  This runner imports every module outside
# Blender and calls that function, so the standalone coverage is preserved
# and runnable in one shot:
#
#     python tests/test_selftests.py            # run all
#     python tests/test_selftests.py frieze isohedral   # run a subset
#
# A module is a FAILURE if its self-test raises OR prints a failure marker
# (BAD / FAIL). Modules that cannot be imported outside Blender, or that
# expose no self-test, are reported as SKIP -- they are covered by the
# Blender-driven suites in this directory instead. Exit status is non-zero
# if any self-test fails.
import io
import os
import re
import sys
import time
import types
import importlib
import contextlib
import traceback

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_ART = os.path.join(PROJ, 'math_art')

# Import the generators as submodules of a synthetic `math_art` package so
# their intra-package `from . import sibling` statements resolve, WITHOUT
# executing the real math_art/__init__.py (which imports bpy). Keeping the
# folder on sys.path as well lets the modules' `except ImportError: import
# sibling` fallbacks resolve too.
sys.path.insert(0, MATH_ART)
_PKG = 'math_art'
if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [MATH_ART]
    sys.modules[_PKG] = _pkg

# printed tokens that mean a self-test detected a problem
_FAIL_MARKER = re.compile(r'\b(BAD|FAIL(?:URE|ED|URES)?)\b')


def _discover():
    """Top-level modules, PLUS one level of subpackages.

    The subpackage walk is not optional.  Discovery used to be a flat
    listdir over `math_art/*.py`, which silently skipped anything inside
    a package -- so moving an engine into `math_art/lsystem/` would have
    made every one of its self-tests invisible while the suite still
    reported success.  A green bar that means nothing is the worst
    failure mode available, so subpackages are walked explicitly and
    their names reported individually.

    A subpackage is included when it has an `__init__.py`; its own
    `_selftest` runs first (as `<pkg>`), then each submodule
    (`<pkg>.<mod>`).  `math_art/seifert/` has no self-tests today and is
    simply reported as SKIP, which is accurate rather than silent.
    """
    names = [f[:-3] for f in sorted(os.listdir(MATH_ART))
             if f.endswith('.py') and f != '__init__.py']
    for entry in sorted(os.listdir(MATH_ART)):
        sub = os.path.join(MATH_ART, entry)
        if not os.path.isdir(sub) or entry.startswith(('.', '_', 'test')):
            continue
        if not os.path.exists(os.path.join(sub, '__init__.py')):
            continue
        names.append(entry)
        names.extend(f"{entry}.{f[:-3]}" for f in sorted(os.listdir(sub))
                     if f.endswith('.py') and f != '__init__.py')
    return names


def _selftest_fn(mod):
    for attr in ('_selftest', '_self_test', '_run_selftest'):
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    return None


def main(argv):
    wanted = set(argv)
    results = {'OK': [], 'FAIL': [], 'SKIP': []}
    for name in _discover():
        if wanted and name not in wanted:
            continue
        try:
            mod = importlib.import_module(f'{_PKG}.{name}')
        except Exception as exc:  # noqa: BLE001 - a bpy-only module, or a
            # missing optional dependency: not this runner's concern
            results['SKIP'].append(name)
            print(f"SKIP  {name:34s} (import: {type(exc).__name__}: {exc})")
            continue
        fn = _selftest_fn(mod)
        if fn is None:
            results['SKIP'].append(name)
            print(f"SKIP  {name:34s} (no self-test)")
            continue
        buf = io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
            dt = time.time() - t0
            out = buf.getvalue()
            if _FAIL_MARKER.search(out):
                results['FAIL'].append(name)
                print(f"FAIL  {name:34s} ({dt:5.1f}s) printed failure marker")
            else:
                results['OK'].append(name)
                print(f"OK    {name:34s} ({dt:5.1f}s)")
        except Exception:  # noqa: BLE001 - report any self-test error
            dt = time.time() - t0
            results['FAIL'].append(name)
            print(f"FAIL  {name:34s} ({dt:5.1f}s) raised")
            traceback.print_exc()

    n = sum(len(v) for v in results.values())
    print(f"\n{'=' * 60}")
    print(f"ran {n} modules: {len(results['OK'])} OK, "
          f"{len(results['FAIL'])} FAIL, {len(results['SKIP'])} SKIP")
    if results['FAIL']:
        print("FAILURES:", ", ".join(results['FAIL']))
        print("RESULT: FAIL")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
