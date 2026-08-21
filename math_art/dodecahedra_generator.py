
# Twelve-faced solids
#
# The regular dodecahedron is one of many, and Hart's survey of the
# twelve-faced solids sorts them by what their faces actually are.  This
# generator builds the ones with a shape parameter, which is what makes
# them worth a slider rather than a stored vertex list:
#
#   PYRITOHEDRON   twelve irregular pentagons of pyritohedral symmetry,
#                  the form iron pyrite grows in.  One dial h runs the
#                  whole family: h = 0 is the cube (opposite pentagon
#                  pairs flatten into one square), h = 1/phi is the
#                  REGULAR dodecahedron, and h = 1 is the rhombic
#                  dodecahedron (each pentagon degenerates to a rhombus).
#   TETARTOID      twelve irregular pentagons of CHIRAL tetrahedral
#                  symmetry -- the pentagonal-tritetrahedral form, which
#                  cobaltite grows in.  Two dials, since the face plane
#                  is a general direction rather than one on a mirror.
#   SCALENOHEDRON  twelve scalene triangles: a hexagonal dipyramid whose
#                  equator zigzags, so alternate ring vertices sit above
#                  and below the mid-plane.  Calcite grows in this.
#   DITRIGONAL     twelve isosceles triangles over a ditrigonal hexagon
#                  -- the ring alternates in RADIUS instead of height.
#   HEX_DIPYRAMID  twelve isosceles triangles, the plain hexagonal
#                  dipyramid, for comparison with the two above.
#   ELONGATED      the elongated dodecahedron: 8 rhombi + 4 hexagons,
#                  one of Fedorov's five parallelohedra, made by slicing
#                  a rhombic dodecahedron across a 4-fold axis and
#                  inserting a prism.
#   TRAPEZO_RHOMBIC  6 rhombi + 6 trapezoids: the twisted rhombic
#                  dodecahedron, which is to hexagonal close packing
#                  what the rhombic dodecahedron is to cubic.
#
# The two pentagonal families are built as PLANE ORBITS: take one plane,
# sweep it around the twelve rotations of the tetrahedral group, and
# intersect the half-spaces.  Every face is then planar by construction
# and exactly twelve of them exist, which is what a vertex-list
# construction has to be checked for afterwards.
#
# References:
# - George W. Hart, "Dodecahedra", Virtual Polyhedra
#   (georgehart.com/virtual-polyhedra/dodecahedra.html), the survey this
#   follows.
# - Pyritohedron and tetartoid as crystal forms: J. D. Dana, "A System of
#   Mineralogy"; and P. Cromwell, "Polyhedra", Cambridge (1997), ch. 9.
# - The five parallelohedra: E. S. Fedorov (1885); H. S. M. Coxeter,
#   "Regular Polytopes", 3rd ed., Dover (1973).
# - The trapezo-rhombic dodecahedron and close packing: L. Pauling,
#   "The Nature of the Chemical Bond", Cornell (1960).

bl_info = {
    "name": "Twelve-Faced Solids",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Pyritohedron, tetartoid, scalenohedron and the other "
                   "twelve-faced solids",
    "category": "Add Mesh",
}

import math

try:
    from .polyhedra import hull as _hull
    from .polyhedra import fit as _fit
    from .styles import shell as _shell
except ImportError:                       # flat-file / headless import
    from polyhedra import hull as _hull
    from polyhedra import fit as _fit
    from styles import shell as _shell

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------
# the chiral tetrahedral rotation group, as coordinate maps
# --------------------------------------------------------------------------
# Twelve rotations: the identity and three half-turns about the
# coordinate axes, then the eight third-turns about the cube diagonals,
# which cycle the coordinates.

def _tetra_rotations():
    out = []
    for sx, sy, sz in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        out.append(lambda p, a=sx, b=sy, c=sz: (a * p[0], b * p[1], c * p[2]))
        out.append(lambda p, a=sx, b=sy, c=sz: (c * p[2], a * p[0], b * p[1]))
        out.append(lambda p, a=sx, b=sy, c=sz: (b * p[1], c * p[2], a * p[0]))
    return out


_T = _tetra_rotations()


def _orbit(v):
    """The twelve images of a direction, duplicates removed."""
    out = []
    for r in _T:
        w = r(v)
        if not any(all(abs(w[k] - u[k]) < 1e-9 for k in range(3))
                   for u in out):
            out.append(w)
    return out


def _from_planes(normals):
    """Intersect { n . x <= |n| } over a set of normals."""
    planes = []
    for n in normals:
        ln = math.sqrt(sum(c * c for c in n))
        if ln < 1e-12:
            continue
        planes.append(([c / ln for c in n], ln))
    V = _hull.halfspace_vertices(planes)
    return V, _hull.hull_faces(V)


# --------------------------------------------------------------------------
# the solids
# --------------------------------------------------------------------------

def pyritohedron(h=None):
    """Twelve pentagons of pyritohedral symmetry.

    h = 0 cube, h = 1/phi regular dodecahedron, h = 1 rhombic
    dodecahedron.  Built from the orbit of the plane through (0, 1, h).
    """
    if h is None:
        h = 1.0 / PHI
    h = max(0.0, min(1.0, float(h)))
    if h < 1e-9:                          # the cube: pentagons collapse
        return _from_planes(_orbit((0.0, 1.0, 0.0)))
    return _from_planes(_orbit((0.0, 1.0, h)))


def tetartoid(a=1.0, b=0.55, c=0.15):
    """Twelve pentagons of chiral tetrahedral symmetry.

    A general direction has a twelve-element orbit under the rotations
    alone, with no mirror to force the face symmetric -- which is what
    makes this the chiral relative of the pyritohedron.
    """
    return _from_planes(_orbit((float(a), float(b), float(c))))


def _dipyramid(ring, top, bottom=None):
    """Two apexes over a closed ring of equatorial vertices."""
    n = len(ring)
    bottom = [-top[0], -top[1], -top[2]] if bottom is None else bottom
    V = list(ring) + [tuple(top), tuple(bottom)]
    ti, bi = n, n + 1
    F = []
    for k in range(n):
        k2 = (k + 1) % n
        F.append([ti, k, k2])
        F.append([bi, k2, k])
    return V, F


def scalenohedron(sides=3, zig=0.35, height=1.4):
    """A hexagonal (or 2n-gonal) scalenohedron: the equator zigzags, so
    every face is a scalene triangle."""
    n = 2 * max(2, int(sides))
    ring = []
    for k in range(n):
        a = 2 * math.pi * k / n
        z = zig if k % 2 == 0 else -zig
        ring.append((math.cos(a), math.sin(a), z))
    return _dipyramid(ring, (0.0, 0.0, height))


def ditrigonal_dipyramid(inner=0.62, height=1.4):
    """Twelve isosceles triangles over a ditrigonal hexagon -- the ring
    alternates in radius rather than in height."""
    ring = []
    for k in range(6):
        a = 2 * math.pi * k / 6
        r = 1.0 if k % 2 == 0 else float(inner)
        ring.append((r * math.cos(a), r * math.sin(a), 0.0))
    return _dipyramid(ring, (0.0, 0.0, height))


def hexagonal_dipyramid(height=1.4):
    ring = [(math.cos(2 * math.pi * k / 6), math.sin(2 * math.pi * k / 6),
             0.0) for k in range(6)]
    return _dipyramid(ring, (0.0, 0.0, height))


def elongated_dodecahedron(stretch=0.5):
    """8 rhombi + 4 hexagons.

    The rhombic dodecahedron cut across a 4-fold axis with a prism of
    height 2*stretch inserted: the four equatorial vertices in that plane
    each split into a pair, and the eight cube-corner vertices move out
    with the halves they belong to.
    """
    e = float(stretch)
    V = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                V.append((sx * 1.0, sy * 1.0, sz * (1.0 + e)))
    V += [(0.0, 0.0, 2.0 + e), (0.0, 0.0, -(2.0 + e))]
    for sx, sy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        V.append((float(sx), float(sy), e))
        V.append((float(sx), float(sy), -e))
    # At stretch 0 each equatorial pair lands on the same point and the
    # solid is the rhombic dodecahedron again, so weld before hulling
    # rather than handing it coincident vertices.
    out = []
    for p in V:
        if not any(all(abs(p[k] - q[k]) < 1e-9 for k in range(3))
                   for q in out):
            out.append(p)
    return out, _hull.hull_faces(out)


def _close_packed_cell(twist):
    """Voronoi cell of a close packing: the half-space intersection
    against the twelve touching neighbours.

    Both close packings put six neighbours in the plane and three above;
    the only difference is where the three BELOW sit.  Stacking them
    under the upper three (twist=False, ABAB) gives hexagonal close
    packing and the trapezo-rhombic dodecahedron; rotating them sixty
    degrees (twist=True, ABCABC) gives cubic close packing and the
    rhombic dodecahedron.  That single choice is the whole difference
    between the two solids, so it is one flag rather than two builders.
    """
    r = 1.0 / math.sqrt(3.0)
    h = math.sqrt(2.0 / 3.0)
    nb = [(math.cos(math.pi * k / 3), math.sin(math.pi * k / 3), 0.0)
          for k in range(6)]
    for k in range(3):
        a = math.radians(30 + 120 * k)
        nb.append((r * math.cos(a), r * math.sin(a), h))
        b = a + (math.radians(60) if twist else 0.0)
        nb.append((r * math.cos(b), r * math.sin(b), -h))
    planes = []
    for n in nb:
        ln = math.sqrt(sum(c * c for c in n))
        planes.append(([c / ln for c in n], ln / 2.0))
    V = _hull.halfspace_vertices(planes)
    return V, _hull.hull_faces(V)


def trapezo_rhombic_dodecahedron():
    """6 rhombi + 6 trapezoids -- the twisted rhombic dodecahedron, which
    is to hexagonal close packing what the rhombic dodecahedron is to
    cubic."""
    return _close_packed_cell(twist=False)


SOLIDS = [
    ('PYRITOHEDRON', "Pyritohedron", "12 irregular pentagons; the dial "
     "runs cube -> regular dodecahedron -> rhombic dodecahedron"),
    ('TETARTOID', "Tetartoid", "12 irregular pentagons, chiral "
     "tetrahedral symmetry"),
    ('SCALENOHEDRON', "Hexagonal Scalenohedron", "12 scalene triangles "
     "over a zigzag equator"),
    ('DITRIGONAL', "Ditrigonal Dipyramid", "12 isosceles triangles over "
     "a ditrigonal hexagon"),
    ('HEX_DIPYRAMID', "Hexagonal Dipyramid", "12 isosceles triangles"),
    ('ELONGATED', "Elongated Dodecahedron", "8 rhombi + 4 hexagons; one "
     "of the five parallelohedra"),
    ('TRAPEZO_RHOMBIC', "Trapezo-Rhombic Dodecahedron", "6 rhombi + 6 "
     "trapezoids; the cell of hexagonal close packing"),
]


def build(kind, h=None, a=1.0, b=0.55, c=0.15, zig=0.35, inner=0.62,
          height=1.4, stretch=0.5):
    if kind == 'PYRITOHEDRON':
        return pyritohedron(h)
    if kind == 'TETARTOID':
        return tetartoid(a, b, c)
    if kind == 'SCALENOHEDRON':
        return scalenohedron(3, zig, height)
    if kind == 'DITRIGONAL':
        return ditrigonal_dipyramid(inner, height)
    if kind == 'HEX_DIPYRAMID':
        return hexagonal_dipyramid(height)
    if kind == 'ELONGATED':
        return elongated_dodecahedron(stretch)
    if kind == 'TRAPEZO_RHOMBIC':
        return trapezo_rhombic_dodecahedron()
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_twelve_faced_add(bpy.types.Operator):
        """Add a twelve-faced solid: pyritohedron, tetartoid,
        scalenohedron and their relatives"""
        bl_idname = "mesh.twelve_faced_add"
        bl_label = "Twelve-Faced Solid"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(name="Solid", items=SOLIDS,
                            default='PYRITOHEDRON')
        shape: FloatProperty(
            name="Shape", default=1.0 / PHI, min=0.0, max=1.0,
            description="Pyritohedron dial: 0 is the cube, 1/phi (0.618) "
                        "the regular dodecahedron, 1 the rhombic "
                        "dodecahedron")
        tilt: FloatProperty(
            name="Tilt", default=0.55, min=0.05, max=1.5,
            description="Tetartoid face direction, second component")
        skew: FloatProperty(
            name="Skew", default=0.15, min=0.0, max=1.0,
            description="Tetartoid face direction, third component. Zero "
                        "puts the face plane back on a mirror and the "
                        "solid stops being chiral")
        zigzag: FloatProperty(
            name="Zigzag", default=0.35, min=0.0, max=1.0,
            description="How far alternate equator vertices rise and fall")
        waist: FloatProperty(
            name="Waist", default=0.62, min=0.1, max=1.0,
            description="Inner radius of the ditrigonal hexagon's short "
                        "alternate vertices")
        height: FloatProperty(
            name="Height", default=1.4, min=0.1, max=6.0,
            description="Apex height above the equator")
        stretch: FloatProperty(
            name="Stretch", default=0.5, min=0.0, max=3.0,
            description="Depth of the prism inserted into the rhombic "
                        "dodecahedron; zero gives it back unchanged")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        __annotations__.update(_shell.style_properties())

        def execute(self, context):
            try:
                V, F = build(self.solid, h=self.shape, a=1.0, b=self.tilt,
                             c=self.skew, zig=self.zigzag, inner=self.waist,
                             height=self.height, stretch=self.stretch)
            except (ValueError, RuntimeError) as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            V = _fit.fit_cube(V, 2.0 * self.scale)
            obj = _shell.apply(self, context, V, F, "Twelve-Faced Solid")
            if obj is None:
                self.report({'INFO'}, f"{len(F)} face segments")
            else:
                self.report({'INFO'}, f"V={len(V)} F={len(F)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            if self.solid == 'PYRITOHEDRON':
                lay.prop(self, 'shape')
            if self.solid == 'TETARTOID':
                lay.prop(self, 'tilt')
                lay.prop(self, 'skew')
            if self.solid == 'SCALENOHEDRON':
                lay.prop(self, 'zigzag')
            if self.solid == 'DITRIGONAL':
                lay.prop(self, 'waist')
            if self.solid in ('SCALENOHEDRON', 'DITRIGONAL',
                              'HEX_DIPYRAMID'):
                lay.prop(self, 'height')
            if self.solid == 'ELONGATED':
                lay.prop(self, 'stretch')
            _shell.draw_style(self, lay)
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.twelve_faced_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_twelve_faced_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_twelve_faced_add)


def _selftest():
    def euler(V, F):
        E = set()
        for f in F:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                E.add((min(a, b), max(a, b)))
        return len(V) - len(E) + len(F)

    # every one of them really has twelve faces, and closes
    for kind, lbl, _d in SOLIDS:
        V, F = build(kind)
        assert len(F) == 12, (kind, len(F), sorted(len(f) for f in F))
        assert euler(V, F) == 2, (kind, euler(V, F))

    # the pyritohedron's three landmarks
    V, F = pyritohedron(0.0)
    assert len(F) == 6, ("h=0 should be the cube", len(F))
    V, F = pyritohedron(1.0 / PHI)
    assert len(F) == 12 and all(len(f) == 5 for f in F), \
        ("h=1/phi should be the regular dodecahedron",
         sorted(len(f) for f in F))
    # ... and at 1/phi the pentagons really are regular
    for f in F:
        L = [math.dist(V[f[k]], V[f[(k + 1) % 5]]) for k in range(5)]
        assert max(L) - min(L) < 1e-9, ("irregular pentagon", L)
    V, F = pyritohedron(1.0)
    assert len(F) == 12 and all(len(f) == 4 for f in F), \
        ("h=1 should be the rhombic dodecahedron",
         sorted(len(f) for f in F))

    # the tetartoid is chiral: no mirror image of the vertex set can be
    # matched to itself by any coordinate reflection
    V, F = tetartoid()
    assert all(len(f) == 5 for f in F), sorted(len(f) for f in F)
    mirrored = [(-v[0], v[1], v[2]) for v in V]
    same = all(any(math.dist(m, v) < 1e-6 for v in V) for m in mirrored)
    assert not same, "the tetartoid came out mirror-symmetric"

    # face-shape checks: scalene means all three edges differ
    V, F = scalenohedron()
    for f in F:
        L = sorted(round(math.dist(V[f[k]], V[f[(k + 1) % 3]]), 6)
                   for k in range(3))
        assert L[0] != L[1] and L[1] != L[2], ("not scalene", L)

    # the elongated dodecahedron: 8 rhombi + 4 hexagons, 18 vertices
    V, F = elongated_dodecahedron(0.5)
    assert len(V) == 18, len(V)
    assert sorted(len(f) for f in F) == [4] * 8 + [6] * 4, \
        sorted(len(f) for f in F)
    # and zero stretch gives the rhombic dodecahedron back
    V, F = elongated_dodecahedron(0.0)
    assert len(F) == 12 and all(len(f) == 4 for f in F), \
        sorted(len(f) for f in F)

    # the trapezo-rhombic dodecahedron: 6 rhombi + 6 trapezoids
    V, F = trapezo_rhombic_dodecahedron()
    assert len(F) == 12, len(F)
    assert all(len(f) == 4 for f in F), sorted(len(f) for f in F)

    print("RESULT: OK")
