# The gem material table.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`.
#
# Each species carries the four numbers that actually change a render --
# the sodium-D refractive index, the B-G dispersion, the specific gravity
# and, where it has been measured, the RGB absorbance -- plus the optic
# character and birefringence, which are recorded for correctness even
# though the shader cannot yet act on them.
#
# WHERE THE NUMBERS COME FROM, and where they do not.  n_D, dispersion and
# density are published gemmological constants, taken from the standard
# reference tables (LibreTexts "Gemology" ch. 7.16 for the B-G dispersion
# values, the International Gem Society tables for indices and densities).
# They are facts about minerals, and each record names its source.
#
# ABSORBANCE is different, and the distinction is kept explicit in the
# data rather than blurred.  Six species carry MEASURED coefficients,
# Guy and Soler's, computed from real absorbance spectra by the colour-
# matching projection in their equation 9 rather than by point-sampling a
# curve at three wavelengths.  Those records are marked `measured=True`.
# Everything else carries a plausible TINT chosen to look like the stone,
# marked `measured=False`, because inventing a number and presenting it
# beside a measured one would make the table untrustworthy as a whole.
# Anything reading this table for anything but rendering should filter on
# that flag.
#
# OctoNus publish measured visible-absorption spectra, in mm^-1, for
# nine more gem materials (amethyst, corundum in three colours, cubic
# zirconia in four, several diamond colours, sapphire, emerald, topaz
# along and across the c-axis, and yellow quartz).  Fitting those to RGB
# through the same projection would move most of this table from tinted
# to measured; it is the obvious next improvement and needs no new
# machinery, only the spectra.
#
# References:
#   Stephane Guy & Cyril Soler, "Graphics gems revisited: fast and
#     physically-based rendering of gemstones", ACM TOG 23(3):231-238,
#     2004 -- the measured (K_r, K_g, K_b) values and the n_o/n_e pairs
#     for garnet, tourmaline, peridot, diamond, sapphire and andalusite.
#   "Gemology", LibreTexts, ch. 7.16 "Dispersion" -- the B-G interval
#     values, measured between the Fraunhofer B (686.7 nm) and G
#     (430.8 nm) lines.
#   International Gem Society, "Refractive Indices and Double Refraction
#     of Selected Gems" and "Specific Gravity Values of Selected Gems".
#   OctoNus, "Visible absorption spectra and DiamCalc files of colored
#     gem materials" -- the measured spectra noted above.

from typing import NamedTuple


class Material(NamedTuple):
    """One gem material, as the renderer and the cutter both need it."""

    label: str
    n_d: float                  # refractive index at the sodium D line
    dispersion: float           # B-G interval, n(430.8) - n(686.7)
    sg: float                   # specific gravity, g/cm^3
    optic: str                  # 'I' isotropic, 'U+/-' uniaxial, 'B+/-' biaxial
    birefringence: float = 0.0
    alpha_rgb: tuple = (0.0, 0.0, 0.0)      # absorbance, mm^-1
    measured: bool = False      # is alpha_rgb measured, or an artistic tint?
    source: str = ""
    note: str = ""


# Measured absorbances are Guy & Soler's; their n values are quoted as
# n +- half the dispersion, which is where those dispersion figures come
# from for the six species they cover.
MATERIALS = {
    "DIAMOND": Material(
        "Diamond", 2.417, 0.044, 3.52, 'I',
        alpha_rgb=(0.001, 0.001, 0.001), measured=True,
        source="Guy & Soler 2004 (absorbance); LibreTexts 7.16 (dispersion)",
        note="the reference stone: high index, high dispersion, nearly "
             "transparent"),
    "RUBY": Material(
        "Ruby (corundum)", 1.762, 0.018, 3.99, 'U-', birefringence=0.008,
        alpha_rgb=(0.062, 0.360, 0.320), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint: red passes, green and blue absorbed"),
    "SAPPHIRE": Material(
        "Blue Sapphire (corundum)", 1.762, 0.018, 3.99, 'U-',
        birefringence=0.008,
        alpha_rgb=(0.332, 0.270, 0.156), measured=True,
        source="Guy & Soler 2004 (absorbance, extraordinary ray)",
        note="their light-blue sapphire; the ordinary ray is "
             "(0.165, 0.147, 0.185)"),
    "EMERALD": Material(
        "Emerald (beryl)", 1.577, 0.014, 2.72, 'U-', birefringence=0.006,
        alpha_rgb=(0.230, 0.045, 0.140), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint"),
    "AQUAMARINE": Material(
        "Aquamarine (beryl)", 1.577, 0.014, 2.71, 'U-', birefringence=0.005,
        alpha_rgb=(0.115, 0.040, 0.020), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint"),
    "TOURMALINE": Material(
        "Tourmaline (dravite)", 1.642, 0.017, 3.06, 'U-', birefringence=0.018,
        alpha_rgb=(0.033, 0.034, 0.082), measured=True,
        source="Guy & Soler 2004 (absorbance, ordinary ray)",
        note="yellow-green ordinary ray; the extraordinary ray is "
             "(0.010, 0.076, 0.015), blue-green -- this is the "
             "pleochroism their photograph shows"),
    "PERIDOT": Material(
        "Peridot", 1.680, 0.020, 3.34, 'B+', birefringence=0.036,
        alpha_rgb=(0.023, 0.015, 0.051), measured=True,
        source="Guy & Soler 2004 (absorbance, ordinary ray)",
        note="strong birefringence: back facets visibly double"),
    "ANDALUSITE": Material(
        "Andalusite", 1.635, 0.016, 3.15, 'B-', birefringence=0.010,
        alpha_rgb=(0.0056, 0.006, 0.0183), measured=True,
        source="Guy & Soler 2004 (absorbance, ordinary ray)",
        note="biaxial and strongly pleochroic; their extraordinary ray is "
             "(0.170, 0.175, 0.257)"),
    "PYROPE": Material(
        "Pyrope Garnet", 1.746, 0.022, 3.75, 'I',
        alpha_rgb=(0.136, 0.153, 0.175), measured=True,
        source="Guy & Soler 2004 (absorbance); IGS (n, SG)",
        note="their orange-red garnet"),
    "ALMANDINE": Material(
        "Almandine Garnet", 1.790, 0.027, 4.05, 'I',
        alpha_rgb=(0.180, 0.320, 0.380), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint"),
    "SPINEL": Material(
        "Spinel", 1.718, 0.020, 3.60, 'I',
        alpha_rgb=(0.070, 0.230, 0.240), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint"),
    "TOPAZ": Material(
        "Topaz", 1.620, 0.014, 3.53, 'B+', birefringence=0.010,
        alpha_rgb=(0.020, 0.035, 0.075), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint"),
    "AMETHYST": Material(
        "Amethyst (quartz)", 1.544, 0.013, 2.65, 'U+', birefringence=0.009,
        alpha_rgb=(0.070, 0.150, 0.060), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint; OctoNus publish a measured spectrum"),
    "ROCK_CRYSTAL": Material(
        "Rock Crystal (quartz)", 1.544, 0.013, 2.65, 'U+', birefringence=0.009,
        alpha_rgb=(0.0, 0.0, 0.0), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="colourless: no absorbance to measure"),
    "ZIRCON": Material(
        "Zircon (high)", 1.960, 0.039, 4.68, 'U+', birefringence=0.059,
        alpha_rgb=(0.010, 0.012, 0.030), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="the strongest doubling of any common gem: Guy & Soler "
             "measure 2.33 degrees between its ordinary and "
             "extraordinary rays"),
    "SPHENE": Material(
        "Sphene (titanite)", 1.900, 0.051, 3.53, 'B+', birefringence=0.120,
        alpha_rgb=(0.030, 0.060, 0.180), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="more dispersion than diamond, and extreme birefringence"),
    "DEMANTOID": Material(
        "Demantoid Garnet (andradite)", 1.888, 0.057, 3.84, 'I',
        alpha_rgb=(0.150, 0.030, 0.180), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="the most dispersive natural gem in common use"),
    "BENITOITE": Material(
        "Benitoite", 1.757, 0.046, 3.66, 'U+', birefringence=0.047,
        alpha_rgb=(0.220, 0.140, 0.030), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="absorbance is a tint"),
    "FLUORITE": Material(
        "Fluorite", 1.434, 0.007, 3.18, 'I',
        alpha_rgb=(0.040, 0.020, 0.050), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="the lowest index and dispersion here: little fire, and a "
             "critical angle of 44 degrees that a cut must respect"),
    "CUBIC_ZIRCONIA": Material(
        "Cubic Zirconia", 2.170, 0.060, 5.75, 'I',
        alpha_rgb=(0.002, 0.002, 0.002), measured=False,
        source="IGS (n, SG); dispersion is the commonly quoted 0.058-0.066 "
               "range, not the LibreTexts table, which omits CZ",
        note="the diamond simulant: more dispersion than diamond, less "
             "index"),
    "MOISSANITE": Material(
        "Moissanite (synthetic)", 2.670, 0.104, 3.22, 'U+',
        birefringence=0.043,
        alpha_rgb=(0.004, 0.004, 0.006), measured=False,
        source="IGS (n, SG); LibreTexts 7.16 (dispersion)",
        note="dispersion more than twice diamond's -- the excess fire "
             "that identifies it"),
    "YAG": Material(
        "YAG (synthetic)", 1.833, 0.028, 4.60, 'I',
        alpha_rgb=(0.002, 0.002, 0.002), measured=False,
        source="IGS (n, SG)",
        note="an older diamond simulant"),
}

DEFAULT = "DIAMOND"


def get(key):
    try:
        return MATERIALS[key]
    except KeyError:
        raise KeyError(f"no material named {key!r}; known: "
                       f"{', '.join(sorted(MATERIALS))}") from None


def material_items():
    """`(key, label, description)` triples for a UI, ordered by index."""
    out = []
    for key in sorted(MATERIALS, key=lambda k: -MATERIALS[k].n_d):
        m = MATERIALS[key]
        out.append((key, m.label,
                    f"n {m.n_d:.3f}, dispersion {m.dispersion:.3f}, "
                    f"SG {m.sg:.2f}" + (f" -- {m.note}" if m.note else "")))
    return out


def _selftest():
    try:
        from . import optics as _o
    except ImportError:
        import optics as _o

    ok = True

    good = len(MATERIALS) >= 15 and DEFAULT in MATERIALS
    ok &= good
    print(f"gems.materials: {len(MATERIALS)} materials registered "
          f"{'OK' if good else 'too few'}")

    # every record must be physically possible
    bad = [k for k, m in MATERIALS.items()
           if not (1.0 < m.n_d < 3.5) or not (0.0 <= m.dispersion < 0.4)
           or not (1.0 < m.sg < 8.0) or m.birefringence < 0.0
           or len(m.alpha_rgb) != 3 or any(a < 0 for a in m.alpha_rgb)
           or not m.source]
    good = not bad
    ok &= good
    print(f"gems.materials: every record is physically possible and cites a "
          f"source {'OK' if good else 'bad: ' + ', '.join(bad)}")

    # isotropic species must have no birefringence, and only they
    bad = [k for k, m in MATERIALS.items()
           if (m.optic == 'I') != (m.birefringence == 0.0)]
    good = not bad
    ok &= good
    print(f"gems.materials: isotropic species have zero birefringence and "
          f"anisotropic ones do not {'OK' if good else ', '.join(bad)}")

    # the measured/tinted distinction must be real, not decorative
    meas = [k for k, m in MATERIALS.items() if m.measured]
    good = sorted(meas) == sorted(["DIAMOND", "SAPPHIRE", "TOURMALINE",
                                   "PERIDOT", "ANDALUSITE", "PYROPE"])
    ok &= good
    print(f"gems.materials: exactly the six Guy & Soler species are marked "
          f"measured ({len(meas)}) {'OK' if good else 'mislabelled'}")
    bad = [k for k in meas if "Guy & Soler" not in MATERIALS[k].source]
    good = not bad
    ok &= good
    print(f"gems.materials: every measured record names its measurement "
          f"{'OK' if good else ', '.join(bad)}")

    # --- the numbers agree with what the literature says about them -------
    d = get("DIAMOND")
    good = abs(_o.critical_angle(d.n_d) - 24.4) < 0.1
    ok &= good
    print(f"gems.materials: diamond's critical angle comes out "
          f"{_o.critical_angle(d.n_d):.2f} deg, the published 24.4 "
          f"{'OK' if good else 'wrong'}")

    # moissanite must out-disperse diamond, and be identified by it
    good = get("MOISSANITE").dispersion > 2.0 * d.dispersion
    ok &= good
    print(f"gems.materials: moissanite disperses "
          f"{get('MOISSANITE').dispersion / d.dispersion:.1f}x diamond, "
          f"which is how it is told apart {'OK' if good else 'wrong'}")

    # demantoid is the most dispersive NATURAL gem here
    nat = {k: m for k, m in MATERIALS.items()
           if "synthetic" not in m.label.lower() and k != "CUBIC_ZIRCONIA"}
    top = max(nat, key=lambda k: nat[k].dispersion)
    good = top in ("DEMANTOID", "SPHENE")
    ok &= good
    print(f"gems.materials: the most dispersive natural gem in the table is "
          f"{nat[top].label} at {nat[top].dispersion:.3f} "
          f"{'OK' if good else 'unexpected'}")

    # carat weight: a 6.5 mm round brilliant in diamond is about 1 ct, and
    # the same stone in quartz is much lighter
    good = get("ROCK_CRYSTAL").sg < d.sg < get("CUBIC_ZIRCONIA").sg
    ok &= good
    print(f"gems.materials: densities order quartz < diamond < CZ "
          f"({get('ROCK_CRYSTAL').sg} < {d.sg} < "
          f"{get('CUBIC_ZIRCONIA').sg}) {'OK' if good else 'wrong'}")

    # every material must yield a usable pavilion range
    bad = [k for k, m in MATERIALS.items()
           if _o.useful_pavilion_range(m.n_d) is None]
    good = not bad
    ok &= good
    print(f"gems.materials: every material has a pavilion window that "
          f"returns light {'OK' if good else ', '.join(bad)}")

    # and the windows must differ -- that is why designs are retargeted
    lo_d = _o.useful_pavilion_range(d.n_d)[0]
    lo_f = _o.useful_pavilion_range(get("FLUORITE").n_d)[0]
    good = lo_f > lo_d + 5.0
    ok &= good
    print(f"gems.materials: fluorite needs a pavilion {lo_f - lo_d:.1f} deg "
          f"steeper than diamond {'OK' if good else 'no difference'}")

    good = len(material_items()) == len(MATERIALS) \
        and material_items()[0][0] == "MOISSANITE"
    ok &= good
    print(f"gems.materials: the UI list covers every material, highest "
          f"index first {'OK' if good else 'wrong order'}")

    raised = False
    try:
        get("UNOBTAINIUM")
    except KeyError:
        raised = True
    ok &= raised
    print(f"gems.materials: an unknown material raises "
          f"{'OK' if raised else 'returned something'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.materials self-test failed")
