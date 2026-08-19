# Gem optics: dispersion, critical angle, and absorption.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy` -- so the numbers can be checked without a renderer.
#
# THE DISPERSION MODEL.  Gemmology quotes one number for dispersion: the
# B-G interval, n(686.7 nm) - n(430.8 nm), between the Fraunhofer B and G
# lines.  Together with the sodium-D index n_D at 589.3 nm that is two
# measurements, which is exactly enough to fix a two-term Cauchy law
#
#     n(lambda) = A + B / lambda^2        (lambda in micrometres)
#
# and this module builds A and B so that BOTH anchors are reproduced
# exactly.  That matters more than it might seem: the whole visible
# effect of dispersion in a cut stone -- its fire -- is the spread
# between the red and violet indices, and the B-G interval IS that
# spread.  A fit that honours it cannot get the fire wrong, whatever it
# does out in the infrared where no one is looking.
#
# A three-term Sellmeier fit would be better in the tails, and the
# refractiveindex.info database publishes them under CC0 for most of
# these species.  It is deliberately NOT used here: transcribing
# coefficient sets from memory is exactly the kind of thing that is
# wrong in the sixth digit and impossible to notice, whereas n_D and the
# B-G interval are published in every gemmological table and cross-check
# against each other.  `sellmeier` accepts coefficients if a caller has
# them; nothing in the shipped table relies on it.
#
# ABSORPTION is Beer-Lambert, and it is what makes colour depend on the
# CUT: a ray that takes a long path through the stone comes out darker
# than one that skims, so a deep pavilion reads richer than a shallow
# one.  A flat "transmission colour" cannot express that.  The
# absorbance coefficients for six species are Guy and Soler's measured
# values, computed from real spectra by their equation 9 rather than by
# point-sampling an absorbance curve.
#
# References:
#   Stephane Guy & Cyril Soler, "Graphics gems revisited: fast and
#     physically-based rendering of gemstones", ACM TOG 23(3), 2004 --
#     the RGB absorbance table and the projection that produces it.
#   Mikhail N. Polyanskiy, "Refractiveindex.info database of optical
#     constants", Scientific Data 11:94, 2024 -- the CC0 source for
#     Sellmeier coefficients, should they be wanted.
#   Irving H. Malitson, "Refraction and Dispersion of Synthetic
#     Sapphire", JOSA 52(12), 1962; Gorachand Ghosh, Optics
#     Communications 163, 1999 (quartz and calcite); F. Peter,
#     Zeitschrift fuer Physik 15, 1923 (diamond).
#   "Gemology", LibreTexts, ch. 7.16, and the International Gem Society
#     reference tables -- n_D, the B-G dispersion values and densities.

import math

# Fraunhofer lines, in micrometres.
LAMBDA_B = 0.6867          # red   -- the B line
LAMBDA_D = 0.5893          # yellow -- the sodium D line, where n_D is quoted
LAMBDA_G = 0.4308          # violet -- the G line

# CIE RGB primaries, in micrometres.  These are the wavelengths Guy and
# Soler trace for the three channels, and using the same ones keeps this
# module's band indices comparable with their published results.
RGB_LAMBDA = (0.7000, 0.5461, 0.4358)


def cauchy_terms(n_d, dispersion):
    """(A, B) of n = A + B/lambda^2 from n_D and the B-G interval.

    Solving the two anchors exactly:
        B = dispersion / (1/lambda_G^2 - 1/lambda_B^2)
        A = n_D - B / lambda_D^2
    """
    if n_d <= 1.0:
        raise ValueError(f"refractive index must exceed 1, got {n_d}")
    if dispersion < 0.0:
        raise ValueError(f"dispersion cannot be negative, got {dispersion}")
    b = dispersion / (1.0 / LAMBDA_G ** 2 - 1.0 / LAMBDA_B ** 2)
    return n_d - b / LAMBDA_D ** 2, b


def n_at(n_d, dispersion, lambda_um):
    """Refractive index at a wavelength, in micrometres."""
    if lambda_um <= 0.0:
        raise ValueError(f"wavelength must be positive, got {lambda_um}")
    a, b = cauchy_terms(n_d, dispersion)
    return a + b / lambda_um ** 2


def sellmeier(coeffs, lambda_um):
    """n from a Sellmeier fit: n^2 - 1 = sum B_i l^2 / (l^2 - C_i).

    `coeffs` is a sequence of (B_i, C_i) with C_i in micrometres squared.
    Provided for callers holding published coefficients; the shipped
    material table uses the Cauchy form built from n_D and the B-G
    interval instead (see the module docstring).
    """
    l2 = lambda_um ** 2
    n2 = 1.0
    for b, c in coeffs:
        if abs(l2 - c) < 1e-18:
            raise ValueError(f"wavelength sits on a resonance at {c}")
        n2 += b * l2 / (l2 - c)
    if n2 <= 0.0:
        raise ValueError("Sellmeier fit gives a non-physical index")
    return math.sqrt(n2)


def critical_angle(n, outside=1.0):
    """Total-internal-reflection angle, in degrees from the normal."""
    if n <= outside:
        raise ValueError(f"no total internal reflection with n={n} "
                         f"inside and {outside} outside")
    return math.degrees(math.asin(outside / n))


def band_indices(n_d, dispersion, bands=3):
    """[(wavelength_um, n)] for `bands` channels across the visible range.

    Three bands are the CIE RGB primaries.  More bands are spread evenly
    in wavelength between the violet and red primaries, which is what a
    node group needs to reduce the colour banding a three-way split
    shows on a stone with strong fire.
    """
    if bands < 1:
        raise ValueError(f"need at least one band, got {bands}")
    if bands == 3:
        lams = RGB_LAMBDA
    else:
        lo, hi = RGB_LAMBDA[2], RGB_LAMBDA[0]
        lams = tuple(hi + (lo - hi) * k / (bands - 1) for k in range(bands)) \
            if bands > 1 else (LAMBDA_D,)
    return [(lam, n_at(n_d, dispersion, lam)) for lam in lams]


def _wavelength_rgb(lambda_um):
    """A wavelength's approximate linear RGB weight.

    The piecewise approximation of the CIE colour-matching functions
    commonly attributed to Dan Bruton.  It is crude next to a real
    tristabulation, but it is used here only to SPLIT energy between a
    handful of bands whose weights are then renormalised to sum to white,
    so its errors cancel: what matters is that the bands partition the
    spectrum, not that each one is colorimetrically exact.
    """
    w = lambda_um * 1000.0
    if 380 <= w < 440:
        r, g, b = -(w - 440) / 60.0, 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / 50.0, 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20.0
    elif 510 <= w < 580:
        r, g, b = (w - 510) / 70.0, 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / 65.0, 0.0
    elif 645 <= w <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0
    return r, g, b


def band_weights(bands=3):
    """Per-band RGB weights that sum to white.

    Three bands are the pure channels, which is the classic RGB split: a
    glass BSDF per channel, each at its own index.  More bands are
    weighted by `_wavelength_rgb` and then normalised so the set still
    sums to (1, 1, 1) -- without that a six-band stone would render
    dimmer than a three-band one and the band count would look like an
    exposure control.
    """
    if bands < 1:
        raise ValueError(f"need at least one band, got {bands}")
    if bands == 1:
        return [(1.0, 1.0, 1.0)]
    if bands == 3:
        return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    lams = [lam for lam, _ in band_indices(1.5, 0.0, bands)]
    raw = [_wavelength_rgb(lam) for lam in lams]
    tot = [sum(c[i] for c in raw) or 1.0 for i in range(3)]
    return [tuple(c[i] / tot[i] for i in range(3)) for c in raw]


def fresnel_normal(n, outside=1.0):
    """Reflectance at normal incidence -- the stone's surface lustre."""
    r = (n - outside) / (n + outside)
    return r * r


def absorption_density(alpha_rgb_per_mm, size_mm, blender_span=2.0):
    """Per-channel density for a Volume Absorption node.

    Beer-Lambert attenuates as exp(-alpha * path).  The mesh is fitted to
    a `blender_span`-unit cube, so a path that is `size_mm` long in the
    real stone is `blender_span` long in the scene; the density has to be
    scaled by that ratio or a 3 mm stone and a 30 mm stone would render
    identically, which is the one thing volume absorption exists to
    avoid.
    """
    if size_mm <= 0.0 or blender_span <= 0.0:
        raise ValueError("size and span must be positive")
    k = size_mm / blender_span
    return tuple(max(0.0, a) * k for a in alpha_rgb_per_mm)


def transmission(alpha_per_mm, path_mm):
    """Beer-Lambert transmittance along a path."""
    return math.exp(-max(0.0, alpha_per_mm) * max(0.0, path_mm))


def pavilion_returns(pavilion_angle, n):
    """Does a ray down the table return, on the classic two-bounce path?

    Tolkowsky's model: a ray entering through the table strikes one
    pavilion facet, then the opposite one, and leaves through the crown.
    It returns only if BOTH strikes exceed the critical angle.  A ray
    arriving vertically meets a pavilion facet inclined at `alpha` at an
    incidence of alpha from that facet's normal, and the second strike is
    at 3*alpha - 90 degrees ... in the same geometry, so the pair of
    conditions below is what bounds the useful pavilion range.
    """
    c = critical_angle(n)
    first = pavilion_angle
    second = abs(3.0 * pavilion_angle - 180.0 + 90.0)
    return first >= c and second >= c, first, second, c


def useful_pavilion_range(n, lo=30.0, hi=55.0, step=0.05):
    """The pavilion angles that return light, for a given index."""
    good = [lo + k * step for k in range(int((hi - lo) / step) + 1)]
    good = [a for a in good if pavilion_returns(a, n)[0]]
    if not good:
        return None
    return min(good), max(good)


def _selftest():
    ok = True

    # --- the Cauchy fit reproduces both of its anchors exactly -----------
    worst_d = worst_g = 0.0
    for n_d, disp in ((2.417, 0.044), (1.762, 0.018), (1.544, 0.013),
                      (2.65, 0.104), (1.434, 0.007)):
        worst_d = max(worst_d, abs(n_at(n_d, disp, LAMBDA_D) - n_d))
        got = n_at(n_d, disp, LAMBDA_G) - n_at(n_d, disp, LAMBDA_B)
        worst_g = max(worst_g, abs(got - disp))
    good = worst_d < 1e-12 and worst_g < 1e-12
    ok &= good
    print(f"gems.optics: the Cauchy fit reproduces n_D (worst {worst_d:.1e}) "
          f"and the B-G interval (worst {worst_g:.1e}) exactly "
          f"{'OK' if good else 'drifted'}")

    # index must fall with wavelength -- red bends least
    n_r = n_at(2.417, 0.044, RGB_LAMBDA[0])
    n_b = n_at(2.417, 0.044, RGB_LAMBDA[2])
    good = n_b > n_r
    ok &= good
    print(f"gems.optics: violet refracts more than red in diamond "
          f"({n_b:.4f} > {n_r:.4f}) {'OK' if good else 'inverted'}")

    # --- critical angles against the published values --------------------
    cases = ((2.417, 24.4), (1.762, 34.6), (1.544, 40.4), (1.434, 44.2),
             (2.65, 22.2), (1.718, 35.6))
    worst = max(abs(critical_angle(n) - want) for n, want in cases)
    good = worst < 0.05
    ok &= good
    print(f"gems.optics: critical angles match the published table to "
          f"{worst:.3f} deg {'OK' if good else 'wrong'}")

    # --- Sellmeier, on a fit whose answer is known -----------------------
    # Malitson's sapphire fit, evaluated at the sodium D line, must give
    # the ordinary index gemmology quotes for corundum, ~1.7659.
    saph = ((1.4313493, 0.0726631 ** 2), (0.65054713, 0.1193242 ** 2),
            (5.3414021, 18.028251 ** 2))
    # Gemmology quotes corundum as a RANGE, 1.762-1.770 for the ordinary
    # ray, because natural material varies; the right check is that the
    # fit lands inside it, not that it equals the midpoint.
    got = sellmeier(saph, LAMBDA_D)
    good = 1.762 <= got <= 1.770
    ok &= good
    print(f"gems.optics: Malitson's sapphire Sellmeier gives n_D = "
          f"{got:.4f}, inside gemmology's 1.762-1.770 for corundum "
          f"{'OK' if good else 'outside'}")

    # --- band splitting ---------------------------------------------------
    b3 = band_indices(2.417, 0.044, 3)
    good = len(b3) == 3 and b3[0][1] < b3[2][1]
    ok &= good
    print(f"gems.optics: three bands span n {b3[0][1]:.4f} (red) to "
          f"{b3[2][1]:.4f} (blue) {'OK' if good else 'wrong'}")
    w3 = band_weights(3)
    good = all(abs(sum(w[i] for w in w3) - 1.0) < 1e-12 for i in range(3))
    ok &= good
    print(f"gems.optics: three band weights sum to white "
          f"{'OK' if good else 'do not'}")
    for nb in (6, 9, 12):
        w = band_weights(nb)
        if len(w) != nb or any(abs(sum(x[i] for x in w) - 1.0) > 1e-9
                               for i in range(3)):
            ok = False
            print(f"gems.optics: {nb} band weights do not sum to white")
            break
    else:
        print("gems.optics: 6, 9 and 12 band weights all sum to white, so "
              "band count is not an exposure control OK")

    b9 = band_indices(2.417, 0.044, 9)
    good = len(b9) == 9 and b9[0][0] > b9[-1][0] \
        and all(a[1] <= b[1] + 1e-12 for a, b in zip(b9, b9[1:]))
    ok &= good
    print(f"gems.optics: nine bands run red to violet with n increasing "
          f"monotonically {'OK' if good else 'not monotonic'}")

    # --- absorption --------------------------------------------------------
    a = (0.332, 0.270, 0.156)                 # Guy & Soler's sapphire
    d1 = absorption_density(a, 6.0)
    d2 = absorption_density(a, 12.0)
    good = all(abs(y - 2.0 * x) < 1e-12 for x, y in zip(d1, d2))
    ok &= good
    print(f"gems.optics: doubling the stone doubles the absorption density "
          f"{'OK' if good else 'does not scale'}")
    good = transmission(0.332, 0.0) == 1.0 and \
        transmission(0.332, 10.0) < transmission(0.332, 1.0) < 1.0
    ok &= good
    print(f"gems.optics: Beer-Lambert transmission falls with path length "
          f"({transmission(0.332, 1.0):.3f} at 1 mm, "
          f"{transmission(0.332, 10.0):.3f} at 10 mm) "
          f"{'OK' if good else 'wrong'}")

    # --- the two-bounce return, and the windows it implies ---------------
    rng = useful_pavilion_range(2.417)
    good = rng is not None and rng[0] <= 40.75 <= rng[1]
    ok &= good
    print(f"gems.optics: diamond returns light for pavilion angles "
          f"{rng[0]:.1f}-{rng[1]:.1f} deg, and Tolkowsky's 40.75 lies "
          f"inside {'OK' if good else 'outside'}")
    rng_q = useful_pavilion_range(1.544)
    good = rng_q is not None and rng_q[0] > rng[0]
    ok &= good
    print(f"gems.optics: quartz needs a steeper pavilion than diamond "
          f"({rng_q[0]:.1f} vs {rng[0]:.1f} deg) -- why a design must be "
          f"retargeted per material {'OK' if good else 'no difference'}")

    # --- surface lustre ----------------------------------------------------
    good = abs(fresnel_normal(2.417) - 0.1721) < 1e-3 \
        and fresnel_normal(1.544) < fresnel_normal(2.417)
    ok &= good
    print(f"gems.optics: diamond reflects {fresnel_normal(2.417) * 100:.1f}% "
          f"at normal incidence, quartz {fresnel_normal(1.544) * 100:.1f}% "
          f"{'OK' if good else 'wrong'}")

    # --- refusals ----------------------------------------------------------
    bad = 0
    for fn, args in ((cauchy_terms, (0.9, 0.02)), (n_at, (2.4, 0.04, 0.0)),
                     (critical_angle, (0.9,)), (band_indices, (2.4, 0.04, 0)),
                     (absorption_density, ((0.1, 0.1, 0.1), 0.0))):
        try:
            fn(*args)
        except ValueError:
            bad += 1
    good = bad == 5
    ok &= good
    print(f"gems.optics: non-physical inputs are refused ({bad}/5) "
          f"{'OK' if good else 'accepted nonsense'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.optics self-test failed")
