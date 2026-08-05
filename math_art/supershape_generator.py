
# Supershape / Superformula Generator for Blender
#
# Johan Gielis's superformula -- a single polar equation that, by
# varying a handful of parameters, unifies an astonishing range of
# natural and abstract outlines (circles, polygons, stars, flowers,
# starfish, seed pods, gears). This module builds the 2D superformula
# up into solid meshes several ways:
#
#   SUPERSHAPE_3D   spherical product of two superformulas -- the
#                   classic 3D supershape (genus-0 blob with two poles)
#   SUPERTOROID     the two superformulas mapped onto torus topology
#                   (genus-1 ring; lobed outline x faceted tube)
#   SHELL           a superformula cross-section swept along a
#                   logarithmic spiral -- seashell / conch forms
#   SUPERELLIPSOID  Barr's superquadric ellipsoid (cube <-> sphere <->
#                   octahedron <-> pinched star)
#
# The 2D superformula is
#
#   r(phi) = ( |cos(m phi/4)/a|^n2 + |sin(m phi/4)/b|^n3 )^(-1/n1)
#
# and the 3D supershape is the spherical product of two such radii
# r1(theta), r2(phi) with theta in [-pi,pi] (longitude) and phi in
# [-pi/2,pi/2] (latitude):
#
#   x = r1(theta) cos(theta) . r2(phi) cos(phi)
#   y = r1(theta) sin(theta) . r2(phi) cos(phi)
#   z =                        r2(phi) sin(phi)
#
# Geometry only; materials for optional coloring, rendering left to
# Blender.
#
# References:
#   Gielis, J. (2003). "A generic geometric transformation that unifies
#     a wide range of natural and abstract shapes." American Journal of
#     Botany 90(3), 333-338. doi:10.3732/ajb.90.3.333 -- the paper that
#     introduced the superformula.
#   Bourke, P. "Supershapes / SuperFormula."
#     https://paulbourke.net/geometry/supershape/ -- the 3D spherical
#     product, shell and toroidal mappings this module follows.
#   Barr, A. H. (1981). "Superquadrics and Angle-Preserving
#     Transformations." IEEE Computer Graphics & Applications 1(1),
#     11-23 -- the superellipsoid (SUPERELLIPSOID mode).

bl_info = {
    "name": "Supershape",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces > Supershape",
    "description": "Gielis superformula: 3D supershapes, supertoroids, "
                   "seashells and superellipsoids",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi

# ==========================================================================
# Core superformula (pure Python -- no bpy)
# ==========================================================================

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


# ==========================================================================
# Mesh topology helpers
# ==========================================================================

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


# ==========================================================================
# Builders -- each returns (verts, faces); no bpy
# ==========================================================================

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


# ==========================================================================
# Preset gallery
# ==========================================================================
# Each preset carries its own mode plus every parameter that mode needs,
# so choosing a preset fully determines the shape.  Parameter sets are
# (m, n1, n2, n3, a, b).  Values are chosen to give recognisable,
# non-degenerate forms; verified for a genuine 3D bounding box.

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


# ==========================================================================
# Blender operator
# ==========================================================================

try:
    import bpy
    from bpy.props import (FloatProperty, IntProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_supershape_add(bpy.types.Operator):
        """Gielis superformula: 3D supershapes, supertoroids, seashells
        and superellipsoids from a compact set of parameters"""
        bl_idname = "mesh.supershape_add"
        bl_label = "Supershape"
        bl_options = {'REGISTER', 'UNDO'}

        _GRADIENT = [(0.16, 0.32, 0.63), (0.20, 0.55, 0.72),
                     (0.27, 0.71, 0.60), (0.55, 0.78, 0.36),
                     (0.93, 0.80, 0.24), (0.95, 0.60, 0.22),
                     (0.90, 0.36, 0.23), (0.74, 0.27, 0.44)]

        preset: EnumProperty(
            name="Preset",
            description="A curated starting shape; choose Custom to dial "
                        "in the parameters by hand",
            items=([('CUSTOM', "Custom", "Hand-tuned parameters")]
                   + [(k, lbl, PRESETS[k]['mode']) for k, lbl in
                      _PRESET_ORDER]),
            default='STARFISH')

        mode: EnumProperty(
            name="Mode",
            description="Which superformula construction to build "
                        "(used when Preset is Custom)",
            items=[('SUPERSHAPE_3D', "3D Supershape",
                    "Spherical product of two superformulas"),
                   ('SUPERTOROID', "Supertoroid",
                    "Two superformulas on a torus (lobed ring x tube)"),
                   ('SHELL', "Shell",
                    "Superformula section swept along a log spiral"),
                   ('SUPERELLIPSOID', "Superellipsoid",
                    "Barr superquadric: cube <-> sphere <-> star")],
            default='SUPERSHAPE_3D')

        # superformula set 1 (longitude / ring / shell section)
        m1: FloatProperty(name="m (set 1)", default=5.0, min=0.0, max=50.0)
        n1_1: FloatProperty(name="n1 (set 1)", default=2.0, min=0.01, max=100.0)
        n2_1: FloatProperty(name="n2 (set 1)", default=7.0, min=-100.0, max=100.0)
        n3_1: FloatProperty(name="n3 (set 1)", default=7.0, min=-100.0, max=100.0)
        a1: FloatProperty(name="a (set 1)", default=1.0, min=0.01, max=10.0)
        b1: FloatProperty(name="b (set 1)", default=1.0, min=0.01, max=10.0)

        # superformula set 2 (latitude / tube)
        m2: FloatProperty(name="m (set 2)", default=5.0, min=0.0, max=50.0)
        n1_2: FloatProperty(name="n1 (set 2)", default=2.0, min=0.01, max=100.0)
        n2_2: FloatProperty(name="n2 (set 2)", default=7.0, min=-100.0, max=100.0)
        n3_2: FloatProperty(name="n3 (set 2)", default=7.0, min=-100.0, max=100.0)
        a2: FloatProperty(name="a (set 2)", default=1.0, min=0.01, max=10.0)
        b2: FloatProperty(name="b (set 2)", default=1.0, min=0.01, max=10.0)

        torus_radius: FloatProperty(
            name="Torus Radius", default=2.0, min=0.0, max=20.0,
            description="Ring radius; keep above the tube size to stay "
                        "embedded (Supertoroid)")

        # superellipsoid exponents
        e1: FloatProperty(name="Exponent (vertical)", default=1.0,
                          min=0.05, max=8.0)
        e2: FloatProperty(name="Exponent (horizontal)", default=1.0,
                          min=0.05, max=8.0)

        # shell controls
        whorls: FloatProperty(name="Whorls", default=4.0, min=0.5, max=12.0)
        growth: FloatProperty(name="Growth / Whorl", default=1.7,
                              min=1.01, max=4.0)
        tube: FloatProperty(name="Tube Size", default=0.35, min=0.02, max=2.0)
        z_rate: FloatProperty(name="Spire Rise", default=1.4,
                              min=0.0, max=6.0)

        segments: IntProperty(
            name="Segments", default=128, min=3, max=1024,
            description="Longitude / ring resolution")
        rings: IntProperty(
            name="Rings", default=64, min=2, max=512,
            description="Latitude / tube resolution")

        coloring: EnumProperty(
            name="Coloring",
            items=[('NONE', "None", "No materials"),
                   ('LATITUDE', "Latitude Bands",
                    "Gradient of materials from pole to pole (or along "
                    "the tube / coil)"),
                   ('DISTANCE', "By Distance",
                    "Gradient by distance from the centre")],
            default='LATITUDE')

        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        # ------------------------------------------------------------------
        def _params(self):
            """Resolve (mode, and everything the builder needs) from the
            preset or the custom fields."""
            if self.preset != 'CUSTOM':
                p = PRESETS[self.preset]
                return p['mode'], p
            p = dict(
                mode=self.mode,
                theta=(self.m1, self.n1_1, self.n2_1, self.n3_1,
                       self.a1, self.b1),
                phi=(self.m2, self.n1_2, self.n2_2, self.n3_2,
                     self.a2, self.b2),
                torus_radius=self.torus_radius,
                e1=self.e1, e2=self.e2,
                whorls=self.whorls, growth=self.growth,
                tube=self.tube, z_rate=self.z_rate)
            return self.mode, p

        def _build(self):
            mode, p = self._params()
            seg, rng = self.segments, self.rings
            if mode == 'SUPERSHAPE_3D':
                return build_supershape_3d(p['theta'], p['phi'], seg, rng)
            if mode == 'SUPERTOROID':
                return build_supertoroid(p['theta'], p['phi'],
                                         p.get('torus_radius', 2.0), seg, rng)
            if mode == 'SUPERELLIPSOID':
                return build_superellipsoid(p['e1'], p['e2'], 1.0, 1.0, 1.0,
                                            seg, rng)
            # SHELL: needs a decent number of samples along the coil
            return build_shell(p['phi'], p.get('whorls', 4.0),
                               p.get('growth', 1.7), p.get('tube', 0.35),
                               p.get('z_rate', 1.4), max(seg, 96), rng)

        @classmethod
        def _material(cls, i, n):
            name = f"Supershape {i + 1}/{n}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._GRADIENT[
                    int(round(i / max(1, n - 1) * (len(cls._GRADIENT) - 1)))]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.4
            return mat

        def _apply_color(self, me, verts, faces):
            n = min(8, max(2, self.rings // 4))
            if self.coloring == 'LATITUDE':
                # band by the mean z of each face (pole-to-pole / along coil)
                zs = [v[2] for v in verts]
                zlo, zhi = min(zs), max(zs)
                span = (zhi - zlo) or 1.0
                bands = [min(n - 1, int((sum(verts[k][2] for k in f)
                                         / len(f) - zlo) / span * n))
                         for f in faces]
            else:                                # DISTANCE
                ds = [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                      for v in verts]
                dlo, dhi = min(ds), max(ds)
                span = (dhi - dlo) or 1.0
                bands = [min(n - 1, int((sum(ds[k] for k in f) / len(f)
                                         - dlo) / span * n)) for f in faces]
            for i in range(n):
                me.materials.append(self._material(i, n))
            me.polygons.foreach_set('material_index', bands)

        def execute(self, context):
            for o in context.selected_objects:
                o.select_set(False)
            verts, faces = self._build()
            if not verts or not faces:
                self.report({'ERROR'}, "Degenerate parameters produced "
                                       "an empty mesh")
                return {'CANCELLED'}
            # center on the bbox and fit into a 2 m cube (scale 1 -> 2 m)
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            cen = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)),
                   0.5 * (min(zs) + max(zs)))
            ext = max(max(xs) - min(xs), max(ys) - min(ys),
                      max(zs) - min(zs))
            s = (2.0 * self.scale / ext) if ext > 1e-9 else 1.0
            verts = [((v[0] - cen[0]) * s, (v[1] - cen[1]) * s,
                      (v[2] - cen[2]) * s) for v in verts]

            me = bpy.data.meshes.new("Supershape")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
            if (self.coloring != 'NONE'
                    and len(me.polygons) == len(faces)):
                self._apply_color(me, verts, faces)
            me.update()

            obj = bpy.data.objects.new("Supershape", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            obj.select_set(True)
            context.view_layer.objects.active = obj
            mode, _ = self._params()
            self.report({'INFO'},
                        f"{mode}: {len(verts)} verts, {len(faces)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            custom = self.preset == 'CUSTOM'
            mode = self.mode if custom else PRESETS[self.preset]['mode']
            if custom:
                lay.prop(self, 'mode')
                if mode in ('SUPERSHAPE_3D', 'SUPERTOROID'):
                    box = lay.box()
                    box.label(text="Longitude / Ring"
                              if mode == 'SUPERTOROID' else "Longitude")
                    for k in ('m1', 'n1_1', 'n2_1', 'n3_1', 'a1', 'b1'):
                        box.prop(self, k)
                    box = lay.box()
                    box.label(text="Tube" if mode == 'SUPERTOROID'
                              else "Latitude")
                    for k in ('m2', 'n1_2', 'n2_2', 'n3_2', 'a2', 'b2'):
                        box.prop(self, k)
                    if mode == 'SUPERTOROID':
                        lay.prop(self, 'torus_radius')
                elif mode == 'SUPERELLIPSOID':
                    lay.prop(self, 'e1')
                    lay.prop(self, 'e2')
                elif mode == 'SHELL':
                    box = lay.box()
                    box.label(text="Cross-Section (superformula)")
                    for k in ('m1', 'n1_1', 'n2_1', 'n3_1'):
                        box.prop(self, k)
                    for k in ('whorls', 'growth', 'tube', 'z_rate'):
                        lay.prop(self, k)
            lay.prop(self, 'segments')
            lay.prop(self, 'rings')
            lay.prop(self, 'coloring')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.supershape_add", icon='SURFACE_NSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_supershape_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_supershape_add)


# ==========================================================================
# Standalone self-check (run outside Blender)
# ==========================================================================

def _chi(verts, faces):
    edges = set()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edges.add((min(a, b), max(a, b)))
    return len(verts) - len(edges) + len(faces)


def _aspect(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    dims = sorted([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)])
    return dims[0] / dims[2] if dims[2] > 1e-9 else 0.0


if __name__ == "__main__" and not _IN_BLENDER:
    print("== builders ==")
    v, f = build_supershape_3d((5, 2, 7, 7, 1, 1), (5, 2, 7, 7, 1, 1), 96, 48)
    print(f"supershape_3d: V={len(v)} F={len(f)} chi={_chi(v, f)} "
          f"aspect={_aspect(v):.3f} {'OK' if _chi(v, f) == 2 else 'BAD'}")
    v, f = build_superellipsoid(0.3, 0.3, 1, 1, 1, 96, 48)
    print(f"superellipsoid: V={len(v)} F={len(f)} chi={_chi(v, f)} "
          f"aspect={_aspect(v):.3f} {'OK' if _chi(v, f) == 2 else 'BAD'}")
    v, f = build_supertoroid((6, 8, 8, 8, 1, 1), (4, 12, 12, 12, 1, 1),
                             2.5, 96, 48)
    print(f"supertoroid: V={len(v)} F={len(f)} chi={_chi(v, f)} "
          f"aspect={_aspect(v):.3f} {'OK' if _chi(v, f) == 0 else 'BAD'}")
    v, f = build_shell((3, 40, 10, 10, 1, 1), 4, 1.7, 0.42, 1.6, 200, 40)
    print(f"shell: V={len(v)} F={len(f)} chi={_chi(v, f)} "
          f"aspect={_aspect(v):.3f}")
    print("== presets ==")
    bad = []
    for name in PRESETS:
        v, f = build_from_preset(name, 80, 40)
        asp = _aspect(v)
        ok = bool(v) and bool(f) and asp > 0.02
        if not ok:
            bad.append(name)
        print(f"  {name:14s} V={len(v):5d} F={len(f):5d} "
              f"aspect={asp:.3f} {'OK' if ok else 'FLAT!'}")
    print("\nRESULT:", "OK" if not bad else f"FLAT PRESETS: {bad}")
