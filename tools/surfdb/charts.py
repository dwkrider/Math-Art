"""Parametric charts, by slug -- and why they are safe to write down.

A chart here is (x, y, z, u_range, v_range) in the exact expression
language, with `u` and `v` free.  Unlike the implicit block, there is no
shipped implementation to sample as an oracle for most of these -- the
flat generator modules import bpy and cannot be called headlessly.

What makes writing them down safe anyway is that a chart claiming to be a
pseudosphere is CHECKED AGAINST THE PROPERTY THAT DEFINES ONE: the
validator measures K over the chart and requires K = -1 to 2e-3.  A
mistranscribed chart does not quietly become a slightly different
surface; it stops satisfying the condition its record claims, and fails.

That is weaker than the implicit oracle -- it pins the surface's
curvature class, not its identity -- so a chart is added ONLY for a
record whose curvature condition is gated (`minimal`, `flat`,
`k-const-positive`, `k-const-negative`).  For a record with no condition
there would be nothing to catch an error, and the record keeps its
explicit "no closed form is stored" note instead.

Ranges keep clear of chart singularities; the sampler additionally trims
a 4% margin off each edge.
"""

CHARTS = {
    # -- classical minimal surfaces (H = 0) --------------------------------
    "catenoid": {
        "x": "cosh(v)*cos(u)", "y": "cosh(v)*sin(u)", "z": "v",
        "u_range": ("0", "2*pi"), "v_range": ("-1.5", "1.5"),
        "periodic_u": True,
    },
    "helicoid": {
        "x": "v*cos(u)", "y": "v*sin(u)", "z": "u",
        "u_range": ("-pi", "pi"), "v_range": ("-1.5", "1.5"),
    },
    "enneper-surface": {
        "x": "u - u**3/3 + u*v**2",
        "y": "v - v**3/3 + v*u**2",
        "z": "u**2 - v**2",
        "u_range": ("-1.4", "1.4"), "v_range": ("-1.4", "1.4"),
    },
    "henneberg-surface": {
        "x": "2*sinh(u)*cos(v) - (2.0/3.0)*sinh(3*u)*cos(3*v)",
        "y": "2*sinh(u)*sin(v) + (2.0/3.0)*sinh(3*u)*sin(3*v)",
        "z": "2*cosh(2*u)*cos(2*v)",
        "u_range": ("0.05", "1.0"), "v_range": ("0", "pi"),
    },
    "catalan-surface": {
        "x": "u - sin(u)*cosh(v)",
        "y": "1 - cos(u)*cosh(v)",
        "z": "4*sin(u/2)*sinh(v/2)",
        "u_range": ("0", "4*pi"), "v_range": ("-1.2", "1.2"),
    },
    # `catalan` is a second record for the same classical surface (from the
    # math_art.minsurf.parametric catalog, whose _catalan uses this exact
    # formula on U in [-pi, 3*pi]); same chart, same verification.
    "catalan": {
        "x": "u - sin(u)*cosh(v)",
        "y": "1 - cos(u)*cosh(v)",
        "z": "4*sin(u/2)*sinh(v/2)",
        "u_range": ("0", "4*pi"), "v_range": ("-1.2", "1.2"),
    },
    "bour-surface": {
        "x": "u*cos(v) - u**2*cos(2*v)/2",
        "y": "-u*sin(v) - u**2*sin(2*v)/2",
        "z": "4/3*u**(3.0/2.0)*cos(3*v/2)",
        "u_range": ("0.05", "1.4"), "v_range": ("0", "2*pi"),
    },
    # Richmond. An earlier candidate WRITTEN FROM MEMORY measured
    # max |H| = 251 against a tolerance of 2e-3 -- not a minimal surface at
    # all -- and was deleted rather than patched. This one is instead
    # TRANSCRIBED from the shipped implementation
    # (math_art/minsurf/parametric.py, _richmond):
    #     z = u e^{iv};  x = Re(-1/(2z) - z^3/6);  y = Im(-1/(2z) + z^3/6);
    #     third coordinate = Re(z)
    # expanded into real u, v below, and it measures max |H| ~ 4e-8. The
    # gate did exactly its job both times: it killed the guess and it
    # passes the vetted source. u stays off the puncture at u = 0 (the
    # planar end, where -1/(2z) blows up).
    "richmond-surface": {
        "x": "-cos(v)/(2*u) - u**3*cos(3*v)/6",
        "y": "sin(v)/(2*u) + u**3*sin(3*v)/6",
        "z": "u*cos(v)",
        "u_range": ("0.3", "1.25"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
    },
    # Generalized Richmond, g = z^k with dh = dz (the Weierstrass data the
    # shipped catalog row carries: math_art/minsurf/zoo.py, 'RICHMOND_K').
    # Integrating (g, dh) gives
    #     x + iy ~ z^(1-k)/(2(1-k)) - z^(k+1)/(2(k+1)),  x3 = Re(z) ,
    # here at k = 3 (k = 2 is the classical Richmond above), with the same
    # sign convention as the shipped classical Richmond.
    "richmond-generalized-g-eq-z-k": {
        "x": "-cos(2*v)/(4*u**2) - u**4*cos(4*v)/8",
        "y": "sin(2*v)/(4*u**2) + u**4*sin(4*v)/8",
        "z": "u*cos(v)",
        "u_range": ("0.35", "1.25"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
    },
    # Jeener's flower: Weierstrass data f = z^2, g = z, which integrates in
    # closed form to x = Re(w^3/3 - w^5/5), y = Re(i(w^3/3 + w^5/5)),
    # x3 = Re(w^4/2) -- the closed form stated in the shipped catalog row
    # (math_art/minsurf/zoo.py, 'JEENER', n = 2, Jeener's own case). The
    # immersion is BRANCHED at w = 0 (the metric vanishes like |w|^4
    # there), so u stays off the flower's centre.
    "jeeners-flower": {
        "x": "u**3*cos(3*v)/3 - u**5*cos(5*v)/5",
        "y": "-(u**3*sin(3*v)/3 + u**5*sin(5*v)/5)",
        "z": "u**4*cos(4*v)/2",
        "u_range": ("0.2", "1.1"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
    },
    # Scherk's first (doubly periodic) surface, the graph
    #     z = ln(cos u / cos v)
    # over one checkerboard square |u|, |v| < pi/2 -- exactly the shipped
    # builder (math_art/minsurf/parametric.py, _scherk_graph, which uses
    # the window 0.47*pi).  The record's primary definition stays the
    # Weierstrass block; this chart attaches as the alternate parametric
    # definition, and the H = 0 gate verifies the transcription.
    "scherk-doubly-periodic": {
        "x": "u", "y": "v", "z": "log(cos(u)/cos(v))",
        "u_range": ("-0.47*pi", "0.47*pi"), "v_range": ("-0.47*pi", "0.47*pi"),
    },
    # Catenoid-helicoid associate (Bonnet) family, from the shipped _cathel
    # (math_art/minsurf/parametric.py):
    #     x = cos(t) cosh u cos v + sin(t) sinh u sin v
    #     y = cos(t) cosh u sin v - sin(t) sinh u cos v
    #     z = cos(t) u + sin(t) v
    # Every member is minimal; the chart FIXES t = pi/4 (cos t = sin t =
    # sqrt(2)/2), the midpoint of the deformation, because t = 0 and
    # t = pi/2 are just the catenoid and helicoid records again. Not
    # periodic in v: z carries a bare +v, so v + 2*pi is a different point
    # (the intermediate members are the non-embedded, helicoid-like ones).
    "catenoid-helicoid-associate-family": {
        "x": "(sqrt(2)/2)*(cosh(u)*cos(v) + sinh(u)*sin(v))",
        "y": "(sqrt(2)/2)*(cosh(u)*sin(v) - sinh(u)*cos(v))",
        "z": "(sqrt(2)/2)*(u + v)",
        "u_range": ("-1.2", "1.2"), "v_range": ("0", "2*pi"),
    },
    # NO minimal-polyhedron chart: the record is a piecewise-LINEAR
    # (discrete-minimal) surface -- there is no smooth parametric patch to
    # measure H on.
    # NO plateau-span chart: the soap film spanning a frame is computed by
    # numerical relaxation (math_art.minimal_surface_toolkit); no closed
    # form exists to transcribe.

    # -- constant negative curvature (K = -1) ------------------------------
    "pseudosphere": {
        "x": "cos(u)/cosh(v)", "y": "sin(u)/cosh(v)", "z": "v - tanh(v)",
        "u_range": ("0", "2*pi"), "v_range": ("0.1", "3.0"),
        "periodic_u": True,
    },
    # Dini's surface has K = -1/(a^2 + b^2), NOT -1, where a is the
    # radius and b the twist. With a = 1 and b = 0.2 the measured K is
    # -0.962 -- which is what the verifier reported, and the chart was
    # right; it was the NORMALISATION that was wrong. Taking
    # a = sqrt(1 - b^2) puts a^2 + b^2 = 1 and K = -1 exactly, which is
    # the convention every other constant-curvature record here uses.
    "dini-surface": {
        "x": "sqrt(1 - 0.2**2)*cos(u)*sin(v)",
        "y": "sqrt(1 - 0.2**2)*sin(u)*sin(v)",
        "z": "sqrt(1 - 0.2**2)*(cos(v) + log(tan(v/2))) + 0.2*u",
        "u_range": ("0", "4*pi"), "v_range": ("0.15", "2.0"),
    },
    "kuen-surface": {
        "x": "2*(cos(u) + u*sin(u))*sin(v)/(1 + u**2*sin(v)**2)",
        "y": "2*(sin(u) - u*cos(u))*sin(v)/(1 + u**2*sin(v)**2)",
        "z": "log(tan(v/2)) + 2*cos(v)/(1 + u**2*sin(v)**2)",
        "u_range": ("-4.0", "4.0"), "v_range": ("0.3", "2.5"),
    },
    # Breather surface, transcribed from the shipped _breather
    # (math_art/hyperbolic_surface_generator.py):
    #     w = sqrt(1 - b^2)
    #     D = b (w^2 cosh^2(b u) + b^2 sin^2(w v))
    #     x = -u + 2 w^2 cosh(b u) sinh(b u) / D
    #     y = 2 w cosh(b u) (w cos(w v) cos v + sin(w v) sin v) / D
    #     z = 2 w cosh(b u) (w cos(w v) sin v - sin(w v) cos v) / D
    # K = -1 for EVERY breather parameter 0 < b < 1 (the surface comes
    # from a sine-Gordon solution through Chebyshev coordinates, where
    # K = -1 is built into the fundamental forms), so unlike Dini there is
    # no normalisation to get wrong; the chart fixes b = 0.4, the shipped
    # default. D >= b w^2 > 0, so the formula has no poles; the u-extent
    # follows the shipped window |u| <= 1.6/b = 4.
    "breather-surface": {
        "x": "-u + 2*(1 - 0.4**2)*cosh(0.4*u)*sinh(0.4*u)"
             "/(0.4*((1 - 0.4**2)*cosh(0.4*u)**2"
             " + 0.4**2*sin(sqrt(1 - 0.4**2)*v)**2))",
        "y": "2*sqrt(1 - 0.4**2)*cosh(0.4*u)"
             "*(sqrt(1 - 0.4**2)*cos(sqrt(1 - 0.4**2)*v)*cos(v)"
             " + sin(sqrt(1 - 0.4**2)*v)*sin(v))"
             "/(0.4*((1 - 0.4**2)*cosh(0.4*u)**2"
             " + 0.4**2*sin(sqrt(1 - 0.4**2)*v)**2))",
        "z": "2*sqrt(1 - 0.4**2)*cosh(0.4*u)"
             "*(sqrt(1 - 0.4**2)*cos(sqrt(1 - 0.4**2)*v)*sin(v)"
             " - sin(sqrt(1 - 0.4**2)*v)*cos(v))"
             "/(0.4*((1 - 0.4**2)*cosh(0.4*u)**2"
             " + 0.4**2*sin(sqrt(1 - 0.4**2)*v)**2))",
        "u_range": ("-4", "4"), "v_range": ("-6", "6"),
    },
    # NO Minding-surface chart: the bulge/spindle meridians have
    # r = a cosh u (resp. a sinh u) but their height is the QUADRATURE
    # z = integral sqrt(1 - a^2 sinh^2 t) dt -- an incomplete elliptic
    # integral, outside the expression language (the shipped generator
    # integrates it numerically with Simpson's rule). The only elementary
    # member of Minding's K = -1 rotational family is the pseudosphere,
    # which has its own record and chart above.
    # NO Amsler-surface chart: the surface is defined through a Painleve
    # III ODE that the shipped generator integrates with RK4; it has no
    # closed form to transcribe.
    # NO multi-soliton chart: the record's own construction block says the
    # 2+ soliton surfaces are NOT yet implemented (they need Sym's
    # formula); there is neither a closed form nor a shipped
    # implementation to transcribe.
    # NO crochet-hyperbolic chart: the crochet surface is grown by a
    # discrete stitch-increase rule; K = -1 holds in the crochet limit,
    # not as a closed-form immersion.

    # -- constant positive curvature (K = +1) ------------------------------
    "sphere": {
        "x": "sin(u)*cos(v)", "y": "sin(u)*sin(v)", "z": "cos(u)",
        "u_range": ("0", "pi"), "v_range": ("0", "2*pi"),
        "periodic_v": True,
    },
    # Sievert's surface, transcribed from the shipped sievert_point
    # (math_art/spherical_surface_generator.py), at its default scale
    # a = 1, where K = +1 exactly:
    #     den = 2 - sin^2 v cos^2 u
    #     rho = 2 sqrt(2 + 2 sin^2 u) sin v / den
    #     theta = -u/sqrt(2) + atan(sqrt(2) tan u)
    #     z = ln tan(v/2) + 4 cos v / den
    # u stays inside (-pi/2, pi/2) -- atan(sqrt(2) tan u) jumps branch at
    # the ends -- and v inside (0, pi), whose ends are the surface's two
    # logarithmic ends, not a defect of the chart.
    "sieverts-surface": {
        "x": "(2*sqrt(2 + 2*sin(u)**2)*sin(v)/(2 - sin(v)**2*cos(u)**2))"
             "*cos(-u/sqrt(2) + atan(sqrt(2)*tan(u)))",
        "y": "(2*sqrt(2 + 2*sin(u)**2)*sin(v)/(2 - sin(v)**2*cos(u)**2))"
             "*sin(-u/sqrt(2) + atan(sqrt(2)*tan(u)))",
        "z": "log(tan(v/2)) + 4*cos(v)/(2 - sin(v)**2*cos(u)**2)",
        "u_range": ("-1.4", "1.4"), "v_range": ("0.3", "pi - 0.3"),
    },
    # NO k-positive-revolution chart: the spindle/bulge meridians have
    # r = a sin u but height z = integral sqrt(1 - a^2 cos^2 t) dt, an
    # incomplete elliptic integral of the second kind, outside the
    # expression language (the shipped generator does the quadrature
    # numerically). The one elementary member, a = 1, IS the sphere --
    # which is the record above.

    # -- flat (K = 0) ------------------------------------------------------
    "circular-cylinder": {
        "x": "cos(u)", "y": "sin(u)", "z": "v",
        "u_range": ("0", "2*pi"), "v_range": ("-1", "1"),
        "periodic_u": True,
    },
    # Oloid: the developable strip ruled between the two perpendicular
    # unit circles, using the exact ruling correspondence of Dirnboeck &
    # Stachel ("The Development of the Oloid", J. Geometry and Graphics 1,
    # 1997), as carried in the shipped module header
    # (math_art/oloid_generator.py):
    #     A(t) = (sin t, -1/2 - cos t, 0),  t in [-2pi/3, 2pi/3]
    #     B(t) = (0, 1/2 - cos t/(1 + cos t), sqrt(1 + 2 cos t)/(1 + cos t))
    # chart r(u, v) = (1 - v) A(u) + v B(u), the upper (+z) half of the
    # hull surface. Developability (K = 0) is a property of THIS pairing
    # of circle points -- a wrong correspondence would still be ruled but
    # not flat, so the K-gate genuinely checks the transcription. u stays
    # short of +-2pi/3, where 1 + 2 cos t vanishes and the ruling family
    # ends (the sqrt becomes singular).
    "oloid": {
        "x": "(1 - v)*sin(u)",
        "y": "(1 - v)*(-1/2 - cos(u)) + v*(1/2 - cos(u)/(1 + cos(u)))",
        "z": "v*sqrt(1 + 2*cos(u))/(1 + cos(u))",
        "u_range": ("-1.8", "1.8"), "v_range": ("0", "1"),
    },
    # Sphericon: the classic (4,1) sphericon's surface is four congruent
    # conical bands (the bicone swept by a square about its diagonal,
    # halved and rejoined with a quarter turn); it is only PIECEWISE
    # smooth, so no single chart can cross the seams. This chart is one
    # band exactly: a half-turn of the 45-degree cone (apex (0, 0, 1),
    # base the unit circle), which is isometrically the piece the shipped
    # generator builds (math_art/sphericon_generator.py). v stays off the
    # apex at v = 0, where the chart degenerates.
    "sphericon": {
        "x": "v*cos(u)", "y": "v*sin(u)", "z": "1 - v",
        "u_range": ("0", "pi"), "v_range": ("0.1", "1"),
    },
    # NO d-form chart: a D-form's developable is determined by numerically
    # rolling/joining the two ellipse boundaries (math_art.dform_generator);
    # no closed-form parametrisation is known to transcribe.
}


def chart_for(slug):
    return CHARTS.get(slug)


def _selftest():
    """Every stored chart must satisfy the condition it is stored for."""
    from . import invariants

    # slug -> the condition the chart is supposed to exhibit
    EXPECT = {
        "catenoid": "minimal", "helicoid": "minimal",
        "enneper-surface": "minimal", "henneberg-surface": "minimal",
        "catalan-surface": "minimal", "catalan": "minimal",
        "bour-surface": "minimal",
        "richmond-surface": "minimal",
        "richmond-generalized-g-eq-z-k": "minimal",
        "jeeners-flower": "minimal",
        "scherk-doubly-periodic": "minimal",
        "catenoid-helicoid-associate-family": "minimal",
        "pseudosphere": "k-const-negative", "dini-surface": "k-const-negative",
        "kuen-surface": "k-const-negative",
        "breather-surface": "k-const-negative",
        "sphere": "k-const-positive",
        "sieverts-surface": "k-const-positive",
        "circular-cylinder": "flat",
        "oloid": "flat", "sphericon": "flat",
    }
    bad = []
    for slug, ch in CHARTS.items():
        cond = EXPECT.get(slug)
        assert cond, "chart %r has no expected condition in the self-test" % slug
        st = invariants.sample_parametric_curvature(
            ch["x"], ch["y"], ch["z"], ch["u_range"], ch["v_range"])
        ok, detail = invariants.check_condition(cond, st)
        if not ok:
            bad.append("%s (%s): %s" % (slug, cond, detail))
    if bad:
        raise AssertionError("charts failing their own condition:\n  "
                             + "\n  ".join(bad))
    print("RESULT: OK  (surfdb.charts, %d charts, each verified against the "
          "condition it claims)" % len(CHARTS))
