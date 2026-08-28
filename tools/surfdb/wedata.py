"""Weierstrass data by slug, transcribed from math_art/minsurf/zoo.py.

Each entry is (g, dh) in the exact expression language, with `z` the
complex variable, `i` the imaginary unit, and the row's own parameters
free. Every one is checked against the shipped callable
(`tools/surfdb/weierstrass.compare`), so a mistranscription is rejected
rather than stored -- the same discipline the implicit polynomials get.

`g` is compared EXACTLY; `dh` is allowed one constant multiple, because
scaling dh scales the surface while rescaling g would be a different
surface altogether.

WHY THE ORACLE MATTERS HERE MORE THAN ANYWHERE ELSE.  The first version
of this table was transcribed from `inspect.getsource` on each lambda --
and `inspect.getsource` returns only the FIRST LINE of a multi-line
lambda.  Eight of the sixteen entries were therefore truncated
fragments: `M3_PYR`'s Gauss map really is

    rho * z^(n-1) / (z^n - a^n)

and the truncated form kept only the numerator.  Every one of those eight
was a plausible-looking rational function of the right shape and the
wrong value, and all eight were caught.  Reading these out of the file by
line range, not by `getsource`, is the only safe way to transcribe them.

Only the 16 zoo rows exposing `g` and `dh` as callables can be checked
this way; the rest of the Weierstrass zoo is built by dedicated functions
with no extractable pair, and those records keep their explicit "no
closed form is stored" note.
"""

# slug -> {g, dh, params: the names taken from the row's p_from(...)}
WE = {
    "enneper-surface": {
        "g": "z**k", "dh": "2.0*z**k", "params": ("k",),
    },
    "richmond-generalized-g-eq-z-k": {
        "g": "z**k", "dh": "1", "params": ("k",),
    },
    "double-enneper": {
        "g": "z", "dh": "z*z - 1.0 + 1.0/(z*z)", "params": (),
    },
    "jeeners-flower": {
        "g": "z", "dh": "2.0*z**(n + 1)", "params": ("n",),
    },
    "henneberg-one-sided": {
        "g": "z", "dh": "2.0*z*(1.0 - z**(-4))", "params": (),
    },
    "sphere-catenoid-enneper-end": {
        "g": "rho*(z - 1.0/z)", "dh": "z - 1.0/z", "params": ("rho",),
    },
    "finite-riemann-plane-2-catenoids": {
        "g": "t*(z*z + 3.0)/(z*z - 1.0)",
        "dh": "(z*z + 3.0)/(z*z - 1.0)", "params": ("t",),
    },
    "lopez-sphere-2-ends-of-index-2": {
        "g": "B*(z*z + i*c*z + 1.0)/z",
        "dh": "i*B*(z*z + i*c*z + 1.0)/(z*z)", "params": ("c", "B"),
    },
    "pyramidal-k-noid": {
        "g": "rho*z**(n - 1)/(z**n - a**n)",
        "dh": "z**(n - 1)*(z**n - a**n)/(z**n - 1.0)**2",
        "params": ("n", "a", "rho"),
    },
    "prismatic-k-noid": {
        "g": "a**nn*z**(nn - 1)*(z**nn - 1.0/a**nn)/(z**nn - a**nn)",
        "dh": "z**(nn - 1)*(z**nn - a**nn)*(z**nn - 1.0/a**nn)"
              "/((z**nn - b**nn)**2*(z**nn - 1.0/b**nn)**2)",
        "params": ("nn", "a", "b"),
    },
    "bipyramidal-k-noid": {
        "g": "(z**m - s**m)/(z*(s**m*z**m - 1.0))",
        "dh": "(z**m - s**m)*(s**m*z**m - 1.0)/((z**m - 1.0)**2*z)",
        "params": ("m", "s"),
    },
    "antiprismatic-k-noid-nn-eq-5": {
        "g": "rho*z**(nn - 1)*(z**nn + 1.0/a**nn)/(z**nn - a**nn)",
        "dh": "z**(nn - 1)*(z**nn - a**nn)*(z**nn + 1.0/a**nn)"
              "/((z**nn - b**nn)**2*(z**nn + 1.0/b**nn)**2)",
        "params": ("nn", "a", "b", "rho"),
    },
    "symmetrized-finite-riemann-2m-catenoids": {
        "g": "rho*z**(m - 1)/(z**(2*m) - a**(2*m))",
        "dh": "z**(m - 1)*(z**(2*m) - a**(2*m))/(z**(2*m) - 1.0)**2",
        "params": ("m", "a", "rho"),
    },
    "k-noid-with-enneper-ends": {
        "g": "z**(k - 1)*(z**k - R**k)/(1.0 - R**k*z**k)",
        "dh": "(1.0 - (z**k + z**(-k))/(R**k + R**(-k)))"
              "/(z*(z**k + z**(-k) - 2.0)**2)",
        "params": ("k", "R"),
    },
    "kusner-projective-plane-p-planar-ends": {
        "g": "z**(p - 1)*(z**p - s)/(s*z**p + 1.0)",
        "dh": "i*z**(p - 1)*(z**p - s)*(1.0 + s*z**p)"
              "/(z**(2*p) + 2.0*s*z**p/(p - 1) - 1.0)**2",
        "params": ("p", "s"),
    },
}

# slug -> the zoo row it is transcribed from, for the oracle check.
ZOO_KEY = {
    "enneper-surface": "ENNEPER",
    "richmond-generalized-g-eq-z-k": "RICHMOND_K",
    "double-enneper": "DOUBLE_ENNEPER",
    "jeeners-flower": "JEENER",
    "henneberg-one-sided": "HENNEBERG_RP2",
    "sphere-catenoid-enneper-end": "M3_CATENN",
    "finite-riemann-plane-2-catenoids": "M3_FRIEM",
    "lopez-sphere-2-ends-of-index-2": "M3_LOPEZ",
    "pyramidal-k-noid": "M3_PYR",
    "prismatic-k-noid": "M3_PRISM",
    "bipyramidal-k-noid": "M3_BIPYR",
    "antiprismatic-k-noid-nn-eq-5": "M3_ANTI5",
    "symmetrized-finite-riemann-2m-catenoids": "SYMM_FRIEM",
    "k-noid-with-enneper-ends": "KNOID_ENN_ENDS",
    "kusner-projective-plane-p-planar-ends": "KUSNER_RP2",
}


# Where the record's OTHER curated facts describe a different member of
# the family from the one the operator defaults to.
#
# `enneper-surface` is the case that exposed this: the record's total
# curvature (-4*pi) and degree-1 Gauss map are the CLASSICAL Enneper,
# k = 1, while the shipped row's p_from returns k = 3 -- the three-fold
# member. Declaring k = 1 makes the record internally consistent, and
# the Gauss-degree cross-check then actually agrees with the total
# curvature instead of contradicting it.
PARAM_DEFAULT = {
    "enneper-surface": {"k": 1},
}


def data_for(slug):
    return WE.get(slug)


def defaults_for(slug, p):
    """Parameter defaults for a record: the zoo row's, with overrides."""
    ent = WE.get(slug)
    if not ent:
        return {}
    out = {n: p[n] for n in ent["params"] if n in p}
    out.update(PARAM_DEFAULT.get(slug, {}))
    return out


def verify(slug, spec, p):
    """Check one stored pair against a zoo row's callables and params.

    Returns (ok, [details]).
    """
    from . import weierstrass
    ent = WE[slug]
    missing = [n for n in ent["params"] if n not in p]
    if missing:
        return False, ["row does not supply %s" % missing]
    env = {n: p[n] for n in ent["params"]}
    out, ok_all = [], True
    for field, allow in (("g", False), ("dh", True)):
        ok, detail = weierstrass.compare(
            ent[field], lambda z, _f=spec[field], _p=p: _f(z, _p),
            env, allow_scalar=allow)
        out.append("%s: %s" % (field, detail))
        ok_all = ok_all and ok
    return ok_all, out


def _selftest():
    """Every stored pair must reproduce the shipped callable."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ma = os.path.join(root, "math_art")
    if not os.path.isdir(ma):
        print("RESULT: OK  (surfdb.wedata, %d entries; math_art absent, "
              "not verified)" % len(WE))
        return
    if ma not in sys.path:
        sys.path.insert(0, ma)
    from minsurf import zoo                          # noqa: E402

    assert set(WE) == set(ZOO_KEY), "WE and ZOO_KEY must cover the same slugs"

    bad, checked = [], 0
    for slug in WE:
        key = ZOO_KEY[slug]
        if key not in zoo.WE_SURFACES:
            bad.append("%s: no zoo row %r" % (slug, key))
            continue
        spec = zoo.WE_SURFACES[key]
        try:
            p = spec["p_from"](3, 1.0)
        except Exception as exc:                      # noqa: BLE001
            bad.append("%s: p_from raised: %s" % (slug, exc))
            continue
        ok, details = verify(slug, spec, p)
        if ok:
            checked += 1
        else:
            bad.append("%s -> %s" % (slug, "; ".join(details)))
    if bad:
        raise AssertionError(
            "Weierstrass data disagreeing with the shipped implementation:\n  "
            + "\n  ".join(bad))
    print("RESULT: OK  (surfdb.wedata, %d g/dh pairs verified against "
          "math_art/minsurf/zoo.py)" % checked)
