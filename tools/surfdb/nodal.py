"""Nodal level functions for the TPMS, extracted and verified.

The 23 nodal triply-periodic surfaces are the database's only
`fidelity: approximation` definitions, and the two-tier tolerance built
around them -- exact rows gated tight, approximations gated at their own
published residual -- was, until this module, a design carrying no
numbers: every `residual.max_abs_mean_curvature` was null, so the
approximations were effectively ungated and the mechanism untested.

The cause was simply that no level function was stored, so there was
nothing to measure.  math_art/minsurf/tpms.py holds each one as a plain
`def` of numpy calls:

    def _f_p(x, y, z):
        return np.cos(x) + np.cos(y) + np.cos(z)

which converts to the exact language by dropping the `np.` prefixes.
The conversion is mechanical, so it is DONE mechanically -- and then
checked against the shipped callable over 240 sample points by the same
oracle that guards the implicit block, because "mechanical" is not
"correct" and the Weierstrass block already proved how a plausible
transcription can be silently wrong.

With the function stored, the mean curvature of each nodal surface can
be measured, which is what finally gives the fidelity gate real numbers:
these surfaces are NOT minimal, and the residual says by how much.
"""

import re

_STRIP = re.compile(r"\bnp\.")
_DEF = re.compile(r"^\s*def\s+\w+\([^)]*\)\s*:\s*$")


def convert(source):
    """A `def f(x, y, z): return <expr>` body, in the exact language.

    Comments and the signature are dropped, `np.` prefixes removed, and
    continuation lines joined. Returns None when the function is not of
    that simple shape -- several TPMS rows compute intermediates, and
    guessing at those would be exactly the transcription risk this
    database refuses to take.
    """
    lines = []
    seen_def = False
    for raw in source.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not seen_def:
            if _DEF.match(line):
                seen_def = True
            continue
        lines.append(line.strip())
    if not lines:
        return None
    body = " ".join(lines)
    if not body.startswith("return "):
        return None                      # has intermediates; not convertible
    body = body[len("return "):].strip()
    if "return" in body or "=" in body.replace("==", ""):
        return None
    body = _STRIP.sub("", body)
    if re.search(r"\b(?:where|sum|abs_|maximum|minimum|array)\b", body):
        return None
    return body


def extract(tpms_table, keys=None):
    """key -> level-function string, for rows convertible to the language.

    `tpms_table` is math_art.minsurf.tpms.TPMS: key -> (label, fn, ...).
    """
    import inspect
    out = {}
    for key, row in tpms_table.items():
        if keys is not None and key not in keys:
            continue
        fn = row[1] if isinstance(row, (tuple, list)) and len(row) > 1 else None
        if not callable(fn):
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        got = convert(src)
        if got:
            out[key] = got
    return out


def verify(text, fn, extent=3.2):
    """Check a converted level function against the shipped callable."""
    from . import polynomial
    return polynomial.verify_against(text, fn, extent=extent)


def _selftest():
    """Conversion and the oracle; raises on failure."""
    src = ("def _f_p(x, y, z):\n"
           "    return np.cos(x) + np.cos(y) + np.cos(z)\n")
    assert convert(src) == "cos(x) + cos(y) + cos(z)", convert(src)

    multi = ("def _f_d(x, y, z):\n"
             "    # a comment\n"
             "    return (np.sin(x) * np.sin(y) * np.sin(z)\n"
             "            + np.cos(x) * np.cos(y) * np.cos(z))\n")
    got = convert(multi)
    assert got == "(sin(x) * sin(y) * sin(z) + cos(x) * cos(y) * cos(z))", got

    # a function with intermediates must be REFUSED, not guessed at
    hard = ("def _f_q(x, y, z):\n"
            "    r = np.cos(x)\n"
            "    return r + np.cos(y)\n")
    assert convert(hard) is None, "a body with intermediates must not convert"

    # and the oracle must still reject a wrong conversion
    import numpy as np
    ok, _ = verify("cos(x) + cos(y) + cos(z)",
                   lambda x, y, z: np.cos(x) + np.cos(y) + np.cos(z))
    assert ok
    ok, _ = verify("cos(x) + cos(y) + cos(z)",
                   lambda x, y, z: np.cos(x) + np.cos(y) + np.sin(z))
    assert not ok, "a wrong term must not pass the oracle"

    # against the real tree, if present
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ma = os.path.join(root, "math_art")
    if os.path.isdir(ma):
        if ma not in sys.path:
            sys.path.insert(0, ma)
        from minsurf import tpms as _t                # noqa: E402
        got = extract(_t.TPMS)
        assert len(got) >= 15, \
            "expected most nodal rows to convert, got %d" % len(got)
        bad = []
        for key, text in got.items():
            ok, detail = verify(text, _t.TPMS[key][1])
            if not ok:
                bad.append("%s: %s" % (key, detail))
        assert not bad, "converted level functions disagreeing:\n  " + \
            "\n  ".join(bad)
        print("RESULT: OK  (surfdb.nodal, %d of %d TPMS level functions "
              "converted and verified)" % (len(got), len(_t.TPMS)))
        return
    print("RESULT: OK  (surfdb.nodal, math_art absent)")
