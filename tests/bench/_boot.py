# Headless import bootstrap for the bench harness.
#
# Mirrors tests/test_selftests.py: the real math_art/__init__.py imports
# bpy, so the engine subpackages are imported under a synthetic
# `math_art` package whose __path__ points at the source tree.  Import
# this module before importing anything from math_art.
import os
import sys
import types

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATH_ART = os.path.join(PROJ, 'math_art')

if MATH_ART not in sys.path:
    sys.path.insert(0, MATH_ART)

_PKG = 'math_art'
if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [MATH_ART]
    sys.modules[_PKG] = _pkg
