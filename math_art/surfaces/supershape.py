# The superformula, and the surfaces it sweeps out.
#
# Part of the Math Art surfaces engine (`math_art/surfaces/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Gielis's superformula generalises the superellipse to m-fold symmetry
#
#     r(phi) = ( |cos(m phi/4)/a|^n2 + |sin(m phi/4)/b|^n3 )^(-1/n1)
#
# and covers circles, polygons, stars and a great many natural outlines
# as one family.  The 3-D surface is the spherical product of two such
# curves, one in longitude and one in latitude.
#
# References:
# - J. Gielis, "A generic geometric transformation that unifies a wide
#   range of natural and abstract shapes", American Journal of Botany
#   90, 2003, pp. 333-338.

import math
from math import sin, cos, pi


def _superformula(angle, m, n1, n2, n3, a=1.0, b=1.0):
    """Gielis superformula radius r(angle).

    Robust against the parameter extremes users reach for: a, b and n1
    are clamped away from zero, the cos/sin bases are made positive
    before the fractional powers (a negative base under a real exponent
    is undefined), a vanishing inner sum returns 0 (a finite fallback
    that collapses the point to the origin rather than dividing by
    zero) and huge exponents that overflow also fall back to 0.
    """
    a = a if abs(a) > 1e-9 else 1e-9
    b = b if abs(b) > 1e-9 else 1e-9
    n1 = n1 if abs(n1) > 1e-9 else 1e-9
    t = m * angle / 4.0
    t1 = abs(cos(t) / a)
    t2 = abs(sin(t) / b)
    try:
        s = t1 ** n2 + t2 ** n3
    except (OverflowError, ValueError):
        return 0.0
    if s < 1e-12:
        return 0.0
    try:
        return s ** (-1.0 / n1)
    except (OverflowError, ValueError, ZeroDivisionError):
        return 0.0


def _spow(w_fn, w, e):
    """Signed power sign(f)*|f|^e used by the superellipsoid; keeps the
    correct octant while allowing fractional exponents."""
    v = w_fn(w)
    if v == 0.0:
        return 0.0
    return math.copysign(abs(v) ** e, v)


def _signed_volume(verts, faces):
    """Six times the signed volume of the closed mesh (fan-triangulated
    per face); its sign tells us whether the faces wind outward."""
    acc = 0.0
    for f in faces:
        ax, ay, az = verts[f[0]]
        for i in range(1, len(f) - 1):
            bx, by, bz = verts[f[i]]
            cx, cy, cz = verts[f[i + 1]]
            acc += (ax * (by * cz - bz * cy)
                    - ay * (bx * cz - bz * cx)
                    + az * (bx * cy - by * cx))
    return acc


def _ensure_outward(verts, faces):
    """Reverse every face if the mesh is inside-out, so normals point
    out.  Supershape lobes make the surface non-star-shaped, so we test
    the actual signed volume rather than trusting a fixed winding."""
    if _signed_volume(verts, faces) < 0.0:
        faces = [list(reversed(f)) for f in faces]
    return verts, faces


def build_supershape_3d(p_theta, p_phi, segments=128, rings=64):
    """Spherical product of two superformulas.

    p_theta / p_phi are (m, n1, n2, n3, a, b) for the longitude and
    latitude superformulas.  Genus-0: the two poles at phi = +/-pi/2
    are single welded vertices joined by triangle fans, and the theta
    seam is welded by index wrap -- so the result is watertight.
    """
    segments = max(3, int(segments))
    rings = max(2, int(rings))
    m1, n11, n12, n13, a1, b1 = p_theta
    m2, n21, n22, n23, a2, b2 = p_phi

    verts = []
    # north pole: phi = +pi/2 -> cos(phi)=0 collapses the ring to a point
    rN = _superformula(pi / 2.0, m2, n21, n22, n23, a2, b2)
    verts.append((0.0, 0.0, rN))
    for i in range(1, rings):
        phi = pi / 2.0 - pi * i / rings          # +pi/2 (N) down to -pi/2 (S)
        r2 = _superformula(phi, m2, n21, n22, n23, a2, b2)
        cphi, sphi = cos(phi), sin(phi)
        for j in range(segments):
            theta = -pi + 2.0 * pi * j / segments
            r1 = _superformula(theta, m1, n11, n12, n13, a1, b1)
            verts.append((r1 * cos(theta) * r2 * cphi,
                          r1 * sin(theta) * r2 * cphi,
                          r2 * sphi))
    rS = _superformula(-pi / 2.0, m2, n21, n22, n23, a2, b2)
    verts.append((0.0, 0.0, -rS))
    south = len(verts) - 1

    def vid(i, j):
        return 1 + (i - 1) * segments + j % segments

    faces = []
    for j in range(segments):                    # north fan
        faces.append([0, vid(1, j), vid(1, j + 1)])
    for i in range(1, rings - 1):                 # interior quads
        for j in range(segments):
            faces.append([vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)])
    for j in range(segments):                    # south fan
        faces.append([vid(rings - 1, j), south, vid(rings - 1, j + 1)])
    return _ensure_outward(verts, faces)


def build_superellipsoid(e1=1.0, e2=1.0, rx=1.0, ry=1.0, rz=1.0,
                         segments=128, rings=64):
    """Barr superquadric ellipsoid.  e1 (vertical) and e2 (horizontal)
    are the shape exponents: <1 boxy, 1 an ellipsoid, 1..2 a cushion,
    >2 pinched/octahedral.  Same pole/seam welding as the supershape."""
    segments = max(3, int(segments))
    rings = max(2, int(rings))
    verts = [(0.0, 0.0, rz)]                      # north pole (phi=+pi/2)
    for i in range(1, rings):
        phi = pi / 2.0 - pi * i / rings
        ce = _spow(cos, phi, e1)
        se = _spow(sin, phi, e1)
        for j in range(segments):
            theta = -pi + 2.0 * pi * j / segments
            verts.append((rx * ce * _spow(cos, theta, e2),
                          ry * ce * _spow(sin, theta, e2),
                          rz * se))
    verts.append((0.0, 0.0, -rz))                # south pole
    south = len(verts) - 1

    def vid(i, j):
        return 1 + (i - 1) * segments + j % segments

    faces = []
    for j in range(segments):
        faces.append([0, vid(1, j), vid(1, j + 1)])
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append([vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)])
    for j in range(segments):
        faces.append([vid(rings - 1, j), south, vid(rings - 1, j + 1)])
    return _ensure_outward(verts, faces)


def build_supertoroid(p_ring, p_tube, torus_radius=2.0,
                      segments=128, rings=48):
    """Two superformulas on a torus: p_ring modulates the big ring
    outline (a lobed / gear-toothed path) and p_tube shapes the tube
    cross-section (a faceted / star profile).  Genus-1: both directions
    wrap, so there are no poles -- a pure double-wrapped quad grid."""
    segments = max(3, int(segments))             # around the ring
    rings = max(3, int(rings))                   # around the tube
    verts = []
    for j in range(segments):
        theta = -pi + 2.0 * pi * j / segments
        r_ring = _superformula(theta, *p_ring)
        ct, st = cos(theta), sin(theta)
        for i in range(rings):
            phi = -pi + 2.0 * pi * i / rings
            r_tube = _superformula(phi, *p_tube)
            tube = torus_radius + r_tube * cos(phi)
            verts.append((tube * r_ring * ct,
                          tube * r_ring * st,
                          r_tube * sin(phi)))

    def vid(j, i):
        return (j % segments) * rings + (i % rings)

    faces = []
    for j in range(segments):
        for i in range(rings):
            faces.append([vid(j, i), vid(j + 1, i),
                          vid(j + 1, i + 1), vid(j, i + 1)])
    return _ensure_outward(verts, faces)


def build_shell(p_section, whorls=4.0, growth=1.7, tube=0.35,
                z_rate=1.4, segments=256, rings=48):
    """A superformula cross-section swept along a logarithmic spiral.

    The coil parameter t runs over [0, 2pi*whorls]; each turn the whole
    section scales by `growth`, giving the self-similar expansion of a
    seashell.  z climbs with the coil (z_rate) so the whorls stack into
    a spire.  Open surface (two boundary rims), so it is not welded shut
    along the coil; the tube cross-section (phi) does wrap.
    """
    segments = max(3, int(segments))             # samples along the coil
    rings = max(3, int(rings))                   # around the cross-section
    T = 2.0 * pi * max(0.1, whorls)
    growth = max(1.001, growth)
    ln_g = math.log(growth) / (2.0 * pi)         # grow = growth^(t/2pi)
    verts = []
    for s in range(segments + 1):                # inclusive: open coil ends
        t = T * s / segments
        grow = math.exp(ln_g * t)
        ct, st = cos(t), sin(t)
        for k in range(rings):
            phi = -pi + 2.0 * pi * k / rings
            r = _superformula(phi, *p_section)
            rt = tube * grow * r                 # tube radius grows with coil
            radial = grow + rt * cos(phi)        # distance from the coil axis
            verts.append((radial * ct,
                          radial * st,
                          z_rate * grow + rt * sin(phi)))

    def vid(s, k):
        return s * rings + k % rings

    faces = []
    for s in range(segments):
        for k in range(rings):
            faces.append([vid(s, k), vid(s + 1, k),
                          vid(s + 1, k + 1), vid(s, k + 1)])
    return verts, faces


_R = (1.0, 1.0)                                  # (a, b) shorthand


PRESETS = {
    # --- SUPERSHAPE_3D ---------------------------------------------------
    'SPHERE': dict(mode='SUPERSHAPE_3D',
                   theta=(0, 1, 1, 1, 1, 1), phi=(0, 1, 1, 1, 1, 1)),
    'STARFISH': dict(mode='SUPERSHAPE_3D',
                     theta=(5, 2, 7, 7, 1, 1), phi=(5, 2, 7, 7, 1, 1)),
    'FLOWER': dict(mode='SUPERSHAPE_3D',
                   theta=(8, 0.5, 0.5, 8, 1, 1), phi=(8, 0.5, 0.5, 8, 1, 1)),
    'GEAR': dict(mode='SUPERSHAPE_3D',
                 theta=(16, 0.5, 0.5, 16, 1, 1), phi=(2, 2, 2, 2, 1, 1)),
    'SPIKY': dict(mode='SUPERSHAPE_3D',
                  theta=(7, 0.2, 1.7, 1.7, 1, 1),
                  phi=(7, 0.2, 1.7, 1.7, 1, 1)),
    'ROUNDED_CUBE': dict(mode='SUPERSHAPE_3D',
                         theta=(4, 12, 15, 15, 1, 1),
                         phi=(4, 12, 15, 15, 1, 1)),
    'SEED_POD': dict(mode='SUPERSHAPE_3D',
                     theta=(3, 5, 18, 18, 1, 1), phi=(2, 2, 2, 2, 1, 1)),
    'TWISTED': dict(mode='SUPERSHAPE_3D',
                    theta=(6, 60, 55, 30, 1, 1), phi=(3, 8, 4, 12, 1, 1)),
    'BULB': dict(mode='SUPERSHAPE_3D',
                 theta=(2, 2, 2, 2, 1, 1), phi=(6, 1, 7, 8, 1, 1)),
    # --- SUPERTOROID -----------------------------------------------------
    'TORUS': dict(mode='SUPERTOROID', torus_radius=2.0,
                  theta=(0, 1, 1, 1, 1, 1), phi=(0, 1, 1, 1, 1, 1)),
    'GEAR_RING': dict(mode='SUPERTOROID', torus_radius=2.2,
                      theta=(12, 8, 8, 8, 1, 1), phi=(0, 1, 1, 1, 1, 1)),
    'STAR_TUBE': dict(mode='SUPERTOROID', torus_radius=2.5,
                      theta=(0, 1, 1, 1, 1, 1), phi=(5, 0.3, 1.7, 1.7, 1, 1)),
    'FACETED_RING': dict(mode='SUPERTOROID', torus_radius=2.5,
                         theta=(6, 20, 20, 20, 1, 1),
                         phi=(4, 12, 12, 12, 1, 1)),
    # --- SHELL -----------------------------------------------------------
    'CONCH': dict(mode='SHELL', whorls=4.0, growth=1.7, tube=0.42,
                  z_rate=1.6, phi=(3, 40, 10, 10, 1, 1)),
    'SNAIL': dict(mode='SHELL', whorls=3.5, growth=2.0, tube=0.55,
                  z_rate=0.4, phi=(0, 1, 1, 1, 1, 1)),
    'TUSK': dict(mode='SHELL', whorls=2.2, growth=2.4, tube=0.35,
                 z_rate=2.4, phi=(5, 4, 12, 12, 1, 1)),
    # --- SUPERELLIPSOID --------------------------------------------------
    'SE_SPHERE': dict(mode='SUPERELLIPSOID', e1=1.0, e2=1.0),
    'SE_CUBE': dict(mode='SUPERELLIPSOID', e1=0.2, e2=0.2),
    'SE_CYLINDER': dict(mode='SUPERELLIPSOID', e1=0.2, e2=1.0),
    'SE_OCTAHEDRON': dict(mode='SUPERELLIPSOID', e1=2.0, e2=2.0),
    'SE_STAR': dict(mode='SUPERELLIPSOID', e1=3.5, e2=3.5),
    'SE_PILLOW': dict(mode='SUPERELLIPSOID', e1=1.0, e2=1.6),
}


# Menu ordering for the preset enum (mode, label) alongside CUSTOM.
_PRESET_ORDER = [
    ('SPHERE', "Sphere"), ('STARFISH', "Starfish"), ('FLOWER', "Flower"),
    ('GEAR', "Gear Blob"), ('SPIKY', "Spiky"), ('ROUNDED_CUBE', "Rounded Cube"),
    ('SEED_POD', "Seed Pod"), ('TWISTED', "Twisted"), ('BULB', "Bulb"),
    ('TORUS', "Torus"), ('GEAR_RING', "Gear Ring"),
    ('STAR_TUBE', "Star Tube"), ('FACETED_RING', "Faceted Ring"),
    ('CONCH', "Conch Shell"), ('SNAIL', "Snail Shell"), ('TUSK', "Tusk Shell"),
    ('SE_SPHERE', "Ellipsoid"), ('SE_CUBE', "Cube"),
    ('SE_CYLINDER', "Cylinder"), ('SE_OCTAHEDRON', "Octahedron"),
    ('SE_STAR', "Star (Superquad)"), ('SE_PILLOW', "Pillow"),
]


def build_from_preset(name, segments=128, rings=64):
    """Convenience for headless use: build the named preset's mesh."""
    p = PRESETS[name]
    mode = p['mode']
    if mode == 'SUPERSHAPE_3D':
        return build_supershape_3d(p['theta'], p['phi'], segments, rings)
    if mode == 'SUPERTOROID':
        return build_supertoroid(p['theta'], p['phi'],
                                 p.get('torus_radius', 2.0), segments, rings)
    if mode == 'SUPERELLIPSOID':
        return build_superellipsoid(p['e1'], p['e2'], 1.0, 1.0, 1.0,
                                    segments, rings)
    if mode == 'SHELL':
        return build_shell(p['phi'], p.get('whorls', 4.0),
                           p.get('growth', 1.7), p.get('tube', 0.35),
                           p.get('z_rate', 1.4), max(segments, 128), rings)
    raise ValueError(mode)
