
# Polystix -- non-intersecting cubic rod packings for Blender
#
# Symmetric bundles of straight rods running in 3 or 4 directions that
# interlock through space WITHOUT touching -- the "polystix" family of
# invariant cubic cylinder/rod packings. Distinct from linked-loop
# tangles (see polylinks_generator): here the elements are infinite
# straight rods on a crystallographic lattice, clipped to a finite
# sculptural cell.
#
# Nomenclature (Conway): the prefix counts the SIDES of the rod's
# cross-section prism, not the number of directions.
#   * tetrastix -- square rods in 3 directions (the cube axes, <100>),
#     primitive-cubic packing (O'Keeffe's Pi*); square prisms fill 3/4.
#   * hemistix  -- square rods in 3 directions, the alternative
#     body-centred-cubic packing (O'Keeffe's +Pi).
#   * hexastix  -- hexagonal rods in 4 directions (the body diagonals,
#     <111>), the garnet packing (O'Keeffe's Gamma, space group Ia-3d);
#     hexagonal prisms fill exactly 3/4 -- the "bundle of pencils" form.
#   * tristix   -- triangular rods in the same 4 <111> directions, a
#     chiral packing (O'Keeffe's +Omega, space group I4_1 32).
#   * +Sigma    -- a second chiral 4-direction <111> packing.
# Every <111> packing shares the same four direction vectors and
# differs only in the registration (offset) of the four families.
#
# The four <111> rods meet pairwise at the tetrahedral angle
# arccos(1/3) = 70.5288 deg. In the plane perpendicular to each
# direction the rod centres form a triangular (hexagonal) lattice; the
# other families thread through the gaps. For the garnet (hexastix)
# packing with cube side a, nearest inter-family axis distance is
# a*sqrt(2)/4, so round rods just touch at radius r = a*sqrt(2)/8 and
# the round-cylinder packing fraction is pi*sqrt(3)/8 ~ 0.6802.
#
# References:
#   - M. O'Keeffe & S. Andersson, "Rod Packings and Crystal Chemistry",
#     Acta Cryst. A33 (1977) 914-923.
#   - M. O'Keeffe, J. Plevert, Y. Teshima, Y. Watanabe, T. Ogama,
#     "The Invariant Cubic Rod (Cylinder) Packings: Symmetries and
#     Coordinates", Acta Cryst. A57 (2001) 110-111.
#   - J. H. Conway, H. Burgiel, C. Goodman-Strauss, "The Symmetries of
#     Things" (A K Peters, 2008) -- coins polystix/tetrastix/hexastix.
#   - A. Holden, "Shapes, Space and Symmetry" (Columbia U.P., 1971) --
#     earliest rod-packing sculptures.
#   - A. Widmark, "Sculpture Design with Hexastix and Related
#     Non-Intersecting Cylinder Packings", Bridges 2021, pp. 293-296.
#     https://archive.bridgesmathart.org/2021/bridges2021-293.html
#   - A. Widmark, "Polystix Sculpture Design Revisited", Bridges 2022,
#     pp. 379-382.
#     https://archive.bridgesmathart.org/2022/bridges2022-379.html
#   - K. Hui & J. S. Purcell, "On the geometry of rod packings in the
#     3-torus", arXiv:2212.04662 (2023) -- explicit O'Keeffe coordinates.

bl_info = {
    "name": "Polystix",
    "author": "Math Art project (after O'Keeffe, Conway & Widmark)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polystix",
    "description": "Non-intersecting cubic rod packings "
                   "(hexastix, tetrastix, tristix)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, sqrt, pi

# The four <111> directions used by every hexastix/tristix/Sigma
# packing (one representative from each antipodal pair of body
# diagonals / the four 3-fold axes of the cube).
_D111 = [(1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1)]

# Each packing: direction vectors (integer), the offset point of each
# family (in units of the cube side a, verbatim O'Keeffe / Hui-Purcell
# coordinates), the translation lattice ('PRIM' simple cubic or 'BCC'
# body-centred), and the canonical native prism cross-section (# sides).
PACKINGS = {
    'TETRASTIX': dict(
        label="Tetrastix - square rods, 3 directions",
        dirs=[(0, 0, 1), (1, 0, 0), (0, 1, 0)],
        offsets=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.0, 0.5)],
        lattice='PRIM', native=4),
    'HEMISTIX': dict(
        label="Hemistix - square rods, 3 directions (BCC)",
        dirs=[(0, 0, 1), (0, 0, 1), (1, 0, 0),
              (1, 0, 0), (0, 1, 0), (0, 1, 0)],
        offsets=[(0.0, 0.0, 0.0), (5 / 8, 5 / 8, 0.0),
                 (0.0, 0.5, 0.0), (0.0, 7 / 8, 0.25),
                 (0.5, 0.0, 0.5), (0.75, 0.0, 1 / 8)],
        # the six representative rods already realise the body-centred
        # symmetry, so they tile on the PRIMITIVE lattice; adding the
        # (1/2,1/2,1/2) centring on top makes perpendicular families
        # cross.
        lattice='PRIM', native=4),
    'HEXASTIX': dict(
        label="Hexastix - hexagonal rods, 4 directions (pencils)",
        dirs=list(_D111),
        offsets=[(1 / 8, 0.0, 1 / 4), (3 / 8, 3 / 4, 0.0),
                 (7 / 8, 1 / 4, 0.0), (3 / 8, 1 / 4, 0.0)],
        lattice='BCC', native=6),
    'TRISTIX': dict(
        label="Tristix / +Omega - triangular rods, 4 dir (chiral)",
        dirs=list(_D111),
        offsets=[(1 / 3, 2 / 3, 0.0), (2 / 3, 2 / 3, 0.0),
                 (2 / 3, 1 / 3, 0.0), (1 / 3, 1 / 3, 0.0)],
        lattice='BCC', native=3),
    'SIGMA': dict(
        label="+Sigma - hexagonal rods, 4 directions (chiral)",
        dirs=list(_D111),
        offsets=[(1 / 3, 2 / 3, 0.0), (1 / 6, 2 / 3, 0.0),
                 (2 / 3, 5 / 6, 0.0), (5 / 6, 5 / 6, 0.0)],
        lattice='BCC', native=6),
}


# ----------------------------------------------------------------------
# small vector helpers (plain tuples, no numpy dependency)

def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _add(p, q):
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def _scale(p, s):
    return (p[0] * s, p[1] * s, p[2] * s)


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _norm(u):
    return sqrt(_dot(u, u))


def _unit(u):
    n = _norm(u) or 1.0
    return (u[0] / n, u[1] / n, u[2] / n)


# ----------------------------------------------------------------------
# packing geometry

def _lattice_translations(lattice, rng):
    """Integer (and, for BCC, half-integer-centred) lattice points with
    each coordinate index in range(-rng, rng + 1)."""
    out = []
    R = range(-rng, rng + 1)
    for i in R:
        for j in R:
            for k in R:
                out.append((float(i), float(j), float(k)))
                if lattice == 'BCC':
                    out.append((i + 0.5, j + 0.5, k + 0.5))
    return out


def _skew_line_distance(p, u, q, v):
    """Distance between the infinite lines p + t u and q + s v."""
    n = _cross(u, v)
    nn = _norm(n)
    if nn < 1e-9:                      # parallel
        w = _sub(p, q)
        t = _dot(w, u) / (_dot(u, u) or 1.0)
        perp = _sub(w, _scale(u, t))
        return _norm(perp)
    return abs(_dot(_sub(p, q), n)) / nn


def _max_radius(dirs, offsets, lattice):
    """Largest rod radius before any two rods touch: half the minimum
    distance between the axes of distinct rods, over a neighbourhood of
    lattice translates (the minimum is attained locally). Only the same
    physical line (parallel AND coincident) is skipped, so a genuine
    crossing between two families -- an invalid packing -- correctly
    drives the result toward zero instead of being masked."""
    lat = _lattice_translations(lattice, 3)
    best = 1e18
    for i, di in enumerate(dirs):
        for j, dj in enumerate(dirs):
            for L in lat:
                q = _add(offsets[j], L)
                d = _skew_line_distance(offsets[i], di, q, dj)
                parallel = _norm(_cross(di, dj)) < 1e-9
                if parallel and d < 1e-7:          # same line: skip
                    continue
                if d < best:
                    best = d
    return best / 2.0


def _family_frame(i, dirs, offsets, lattice):
    """In-plane frame (u, e1, e2) for rod family i, with e1 aimed along
    the signed common perpendicular toward the nearest rod of another
    family so a prism flat squarely faces that neighbour."""
    u = _unit(dirs[i])
    lat = _lattice_translations('BCC', 2)
    best_d, best_g = 1e18, None
    for j, dj in enumerate(dirs):
        vj = _unit(dj)
        n = _cross(u, vj)
        nn = _norm(n)
        if nn < 1e-9:                       # parallel family: no flat
            continue
        for L in lat:
            q = _add(offsets[j], L)
            d = _skew_line_distance(offsets[i], u, q, dj)
            if d < 1e-6:
                continue
            if d < best_d - 1e-9:
                s = _dot(_sub(q, offsets[i]), n)
                g = _scale(n, (1.0 if s >= 0 else -1.0) / nn)
                best_d, best_g = d, g
    if best_g is None:                      # fallback basis
        a = (0, 0, 1) if abs(u[2]) < 0.9 else (1, 0, 0)
        best_g = _unit(_cross(u, a))
    e1 = _unit(best_g)
    e2 = _unit(_cross(u, e1))
    return u, e1, e2


def _clip_planes(clip, half):
    """Convex clip volume as a list of (outward-normal, offset) planes
    n.p <= c, plus a sphere radius (or 0). Sized so the volume spans
    roughly +/- half about the origin."""
    planes = []
    sphere = 0.0
    ax = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
          (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    if clip == 'CUBE':
        for n in ax:
            planes.append((n, half))
    elif clip == 'SPHERE':
        sphere = half
    elif clip == 'RHOMBIC_DODECA':
        for a in (1, -1):
            for b in (1, -1):
                planes.append(((a, b, 0), half * sqrt(2)))
                planes.append(((a, 0, b), half * sqrt(2)))
                planes.append(((0, a, b), half * sqrt(2)))
    elif clip == 'TRUNC_OCTA':
        for n in ax:                                  # square faces
            planes.append((n, half))
        for a in (1, -1):                             # hexagon faces
            for b in (1, -1):
                for c in (1, -1):
                    planes.append(((a, b, c), 1.5 * half))
    return planes, sphere


def _clip_segment(Q, uh, planes, sphere):
    """Interval [t0, t1] of the line Q + t*uh inside the convex volume,
    or None if it misses."""
    t0, t1 = -1e18, 1e18
    for n, c in planes:
        nu = _dot(n, uh)
        nq = _dot(n, Q)
        if abs(nu) < 1e-12:
            if nq - c > 1e-9:
                return None
        else:
            t = (c - nq) / nu
            if nu > 0:
                t1 = min(t1, t)
            else:
                t0 = max(t0, t)
    if sphere > 0.0:
        b = _dot(Q, uh)
        cc = _dot(Q, Q) - sphere * sphere
        disc = b * b - cc
        if disc <= 0.0:
            return None
        sd = sqrt(disc)
        t0 = max(t0, -b - sd)
        t1 = min(t1, -b + sd)
    if t1 - t0 <= 1e-6:
        return None
    return t0, t1


def _cross_section(sides, radius, prism):
    """Local (x, y) polygon of a rod cross-section in its own frame.
    For a native prism the given radius is the apothem (centre-to-flat),
    with one flat facing +x so it registers against a neighbouring
    family; for a cylinder it is the tube radius."""
    if prism:
        R = radius / cos(pi / sides)          # apothem -> circumradius
        phase = pi / sides                    # vertex between two flats
    else:
        R = radius
        phase = 0.0
    return [(R * cos(phase + 2 * pi * s / sides),
             R * sin(phase + 2 * pi * s / sides)) for s in range(sides)]


def _add_rod(P0, P1, e1, e2, section, cap, verts, faces, tag_faces, tag):
    n = len(section)
    base = len(verts)
    for P in (P0, P1):
        for (x, y) in section:
            verts.append((P[0] + x * e1[0] + y * e2[0],
                          P[1] + x * e1[1] + y * e2[1],
                          P[2] + x * e1[2] + y * e2[2]))
    r0, r1 = base, base + n
    for s in range(n):
        s2 = (s + 1) % n
        faces.append([r0 + s, r0 + s2, r1 + s2, r1 + s])
        tag_faces.append(tag)
    if cap:
        faces.append([r0 + s for s in range(n - 1, -1, -1)])
        tag_faces.append(tag)
        faces.append([r1 + s for s in range(n)])
        tag_faces.append(tag)


def build_polystix(packing='HEXASTIX', cross_section='PRISM', fill=0.98,
                   extent=4, clip='RHOMBIC_DODECA', handedness='RIGHT',
                   tube_sides=16, cap_ends=True, overhang=0.0,
                   max_rods=6000):
    """Return (verts, faces, face_dir, dir_index) for a polystix
    packing. face_dir[k] is the direction-family colour index of mesh
    face k; dir_index maps each distinct direction to a colour slot.

    ``overhang`` lengthens every rod past the interleaved core by that
    many lattice cells at each end (negative retracts the rods to expose
    the weave)."""
    spec = PACKINGS[packing]
    dirs = spec['dirs']
    offsets = spec['offsets']
    lattice = spec['lattice']
    prism = (cross_section == 'PRISM')
    sides = spec['native'] if prism else max(3, tube_sides)

    r_max = _max_radius(dirs, offsets, lattice)
    radius = max(1e-4, fill) * r_max

    half = extent / 2.0
    planes, sphere = _clip_planes(clip, half)

    # colour slot per distinct direction (up to sign): hemistix has six
    # families but three directions -> three colours; hexastix four.
    def _dkey(d):
        u = d if d >= tuple(-x for x in d) else tuple(-x for x in d)
        return tuple(round(x, 6) for x in _unit(u))

    dir_index = {}
    for d in dirs:
        dir_index.setdefault(_dkey(d), len(dir_index))

    # a stable in-plane frame per family: e1 points along the SIGNED
    # common perpendicular toward an actual nearest contacting rod of
    # another family, so a prism flat faces that neighbour (the
    # space-filling registration). Using a real neighbour with the
    # correct sign -- not just an unsigned cross(u_i, u_j) axis -- is
    # what fixes the chiral triangle (tristix): a triangle has only
    # three flats, so a wrong sign would aim a corner at a neighbour.
    # e2 completes a frame in the plane perpendicular to u.
    frames = [_family_frame(i, dirs, offsets, lattice)
              for i in range(len(dirs))]

    lat = _lattice_translations(lattice, extent + 1)

    verts, faces, tag_faces = [], [], []
    seen = set()
    n_rods = 0
    truncated = False
    for i, d in enumerate(dirs):
        u, e1, e2 = frames[i]
        color = dir_index[_dkey(d)]
        section = _cross_section(sides, radius, prism)
        for L in lat:
            Q = _add(offsets[i], L)
            # canonicalise the line: drop the component along u, so
            # translates that only slide a rod along its own axis (incl.
            # the BCC centring parallel to a <111> rod) collapse to one.
            proj = _dot(Q, u)
            perp = _sub(Q, _scale(u, proj))
            key = (i, tuple(round(x, 5) for x in perp))
            if key in seen:
                continue
            seen.add(key)
            seg = _clip_segment(Q, u, planes, sphere)
            if seg is None:
                continue
            t0, t1 = seg
            if overhang:              # push rod ends past the core
                t0 -= overhang
                t1 += overhang
                if t1 - t0 <= 1e-6:   # retracted to nothing: drop it
                    continue
            P0 = _add(Q, _scale(u, t0))
            P1 = _add(Q, _scale(u, t1))
            _add_rod(P0, P1, e1, e2, section, cap_ends,
                     verts, faces, tag_faces, color)
            n_rods += 1
            if n_rods >= max_rods:
                truncated = True
                break
        if truncated:
            break

    if handedness == 'LEFT':          # reflect through x = 0 (mirror)
        verts = [(-x, y, z) for (x, y, z) in verts]
        faces = [list(reversed(f)) for f in faces]

    return verts, faces, tag_faces, len(dir_index), n_rods, truncated


try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, BoolProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PRESETS = {
        'PENCILS': ("Hexagonal Pencils (hexastix)",
                    dict(packing='HEXASTIX', cross_section='PRISM',
                         fill=1.0, clip='RHOMBIC_DODECA')),
        'HEXCYL': ("Cylinder Bundle (hexastix)",
                   dict(packing='HEXASTIX', cross_section='CYLINDER',
                        fill=0.95, clip='SPHERE')),
        'TETRA': ("Tetrastix Cubes",
                  dict(packing='TETRASTIX', cross_section='PRISM',
                       fill=0.9, clip='CUBE')),
        'TRI': ("Tristix (chiral triangles)",
                dict(packing='TRISTIX', cross_section='PRISM',
                     fill=1.0, clip='TRUNC_OCTA')),
        'SIGMA': ("+Sigma bundle (chiral)",
                  dict(packing='SIGMA', cross_section='CYLINDER',
                       fill=0.95, clip='RHOMBIC_DODECA')),
    }

    _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                (0.62, 0.40, 0.75), (0.25, 0.72, 0.72)]

    class MESH_OT_polystix_add(bpy.types.Operator):
        """Non-intersecting cubic rod packing (hexastix / tetrastix /
        tristix), after O'Keeffe, Conway and Widmark"""
        bl_idname = "mesh.polystix_add"
        bl_label = "Polystix"
        bl_options = {'REGISTER', 'UNDO'}

        def _preset_chosen(self, context):
            if self.preset != 'CUSTOM':
                for k, v in PRESETS[self.preset][1].items():
                    setattr(self, k, v)

        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Custom", "")] +
                  [(k, v[0], "") for k, v in PRESETS.items()],
            default='PENCILS', update=_preset_chosen)
        packing: EnumProperty(
            name="Packing",
            items=[(k, PACKINGS[k]['label'], "")
                   for k in ('TETRASTIX', 'HEMISTIX', 'HEXASTIX',
                             'TRISTIX', 'SIGMA')],
            default='HEXASTIX')
        cross_section: EnumProperty(
            name="Cross Section",
            items=[('PRISM', "Native Prism",
                    "The canonical space-filling polygon for this "
                    "packing (square / hexagon / triangle), oriented "
                    "so flats face the interpenetrating rods"),
                   ('CYLINDER', "Round Cylinder",
                    "Circular rods (n-gon tube)")],
            default='PRISM')
        fill: FloatProperty(
            name="Fill", default=0.98, min=0.02, max=1.2,
            description="Rod radius as a fraction of the just-touching "
                        "radius for this packing (1.0 = rods touch)")
        extent: IntProperty(
            name="Extent", default=4, min=1, max=10,
            description="Number of lattice cells the arrangement spans "
                        "(more = more rods)")
        clip: EnumProperty(
            name="Clip Volume",
            items=[('CUBE', "Cube", ""),
                   ('RHOMBIC_DODECA', "Rhombic Dodecahedron", ""),
                   ('TRUNC_OCTA', "Truncated Octahedron", ""),
                   ('SPHERE', "Sphere", "")],
            default='RHOMBIC_DODECA')
        handedness: EnumProperty(
            name="Handedness",
            items=[('RIGHT', "Right", ""), ('LEFT', "Left (mirror)", "")],
            default='RIGHT',
            description="Mirror the packing (flips the chiral "
                        "enantiomorph of tristix / +Sigma)")
        tube_sides: IntProperty(name="Tube Sides", default=16,
                                min=3, max=48)
        overhang: FloatProperty(
            name="Overhang", default=0.0, min=-4.0, max=8.0,
            description="Extend each rod past the interleaved core by "
                        "this many cells at both ends (negative "
                        "retracts the rods to expose the weave)")
        cap_ends: BoolProperty(name="Cap Ends", default=True)
        coloring: EnumProperty(
            name="Coloring",
            items=[('DIRECTION', "Per Direction",
                    "One material per rod direction (3 or 4 colours)"),
                   ('NONE', "None", "")],
            default='DIRECTION')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        @classmethod
        def _material_for(cls, i):
            name = f"Polystix {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[i] if i < len(cls._PALETTE) else \
                    __import__('colorsys').hsv_to_rgb(
                        (i * 0.618034) % 1.0, 0.6, 0.8)
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.4
            return mat

        _PALETTE = _PALETTE

        def execute(self, context):
            if self.preset != 'CUSTOM':
                self._preset_chosen(context)
            (verts, faces, face_dir, ncols,
             n_rods, truncated) = build_polystix(
                self.packing, self.cross_section, self.fill,
                self.extent, self.clip, self.handedness,
                self.tube_sides, self.cap_ends, self.overhang)
            if not verts:
                self.report({'WARNING'},
                            "No rods in the clip volume; raise Extent")
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Polystix")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # centre and fit within a 2 x scale cube at the origin
            lo = [min(v.co[k] for v in me.vertices) for k in range(3)]
            hi = [max(v.co[k] for v in me.vertices) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) or 1.0
            f = self.scale / half
            for v in me.vertices:
                v.co = [(v.co[k] - (lo[k] + hi[k]) / 2.0) * f
                        for k in range(3)]
            if (self.coloring == 'DIRECTION'
                    and len(me.polygons) == len(faces)):
                for i in range(ncols):
                    me.materials.append(self._material_for(i))
                me.polygons.foreach_set('material_index', face_dir)
                attr = me.attributes.new("dir_index", 'INT', 'FACE')
                attr.data.foreach_set('value', face_dir)
            me.update()
            obj = bpy.data.objects.new("Polystix", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            msg = f"{n_rods} rods"
            if truncated:
                msg += " (capped; lower Extent)"
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        # properties a non-Custom preset overwrites on execute -- these
        # are greyed out (locked) while a preset is active.
        _PRESET_DRIVEN = {'packing', 'cross_section', 'fill', 'clip'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            locked = self.preset != 'CUSTOM'
            keys = ['packing', 'cross_section', 'fill', 'extent', 'clip',
                    'handedness']
            if self.cross_section == 'CYLINDER':
                keys.append('tube_sides')
            keys += ['overhang', 'cap_ends', 'coloring', 'scale']
            for k in keys:
                row = lay.row()
                row.enabled = not (locked and k in self._PRESET_DRIVEN)
                row.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.polystix_add", icon='MESH_CYLINDER')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polystix_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polystix_add)


if __name__ == "__main__" and _IN_BLENDER:
    register()
