
# 4D Regular Polytope Generator for Blender
#
# The six regular convex 4-polytopes -- 5-cell, tesseract (8-cell),
# 16-cell, 24-cell, 600-cell and 120-cell -- rendered as edge
# frameworks in 3D. Edges are either
#
#   STRAIGHT: a perspective projection from 4-space (large distance
#             approaches orthographic, small distance approaches a
#             Schlegel diagram), or
#   CURVED:   vertices are placed on the unit 3-sphere, edges follow
#             great-circle arcs, and the whole picture is mapped by
#             stereographic projection -- every edge becomes a circular
#             arc, the classic "hyperbolic-looking" rendering.
#
# Struts can taper with the local projection scale (near features fat,
# far features thin), and vertices can be capped with spheres.
#
# Beyond the six regular ones the module builds the UNIFORM (semi-regular)
# polychora -- the 4-dimensional analogues of the Archimedean solids -- by
# Wythoff's kaleidoscope, in `polytopes/wythoff.py`.  Take the four mirrors
# of a rank-4 reflection group, mark which of them the generating point is
# held off ("ringing" nodes of the Coxeter diagram), and the orbit of that
# point is the vertex set.  All 15 ringings of each of the four groups
# [3,3,3], [4,3,3], [3,4,3] and [5,3,3] are available, from the 5-cell up
# to the omnitruncated 120-cell's 14400 vertices and 28800 edges.
#
# References:
# - The six regular convex 4-polytopes: Ludwig Schlafli (c. 1852).
# - H. S. M. Coxeter, "Regular Polytopes".
# - Schlegel diagrams: Victor Schlegel (1883).
# - W. A. Wythoff, "A relation between the polytopes of the C600-family",
#   Proc. Section of Sciences, K. Akad. van Wetenschappen te Amsterdam 20
#   (1918), 966-970.
# - H. S. M. Coxeter, "Wythoff's construction for uniform polytopes",
#   Proc. London Math. Soc. (2) 38 (1935), 327-339.
# - Alicia Boole Stott, "Geometrical deduction of semiregular from regular
#   polytopes and space fillings" (1910).

bl_info = {
    "name": "4D Polytopes",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > 4D Polytope",
    "description": "Tesseract, 120-cell and friends, straight or "
                   "stereographically curved edges",
    "category": "Add Mesh",
}




# --------------------------------------------------------------------------
# Vertex constructions (standard coordinates)
# --------------------------------------------------------------------------




















# --------------------------------------------------------------------------
# Duals, equatorial cutaway, Hopf rings (after Segerman, Visualizing
# Mathematics with 3D Printing, figs 3-20/22/23, 3-25, 3-29)
# --------------------------------------------------------------------------





















# --------------------------------------------------------------------------
# 4D rotation and projection
# --------------------------------------------------------------------------











# --------------------------------------------------------------------------
# Strut / sphere mesh helpers
# --------------------------------------------------------------------------











# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------









# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

# The mathematics lives in the sibling `polytopes` engine package;
# this module is the Blender layer over it.
try:
    from .polytopes.regular import (COUNTS, _cell_inradius,
                                        _half_filter, _leonardo_panels,
                                        _pole_angles, _slerp4, add_sphere,
                                        add_strut, dual_vertices,
                                        polytope_edges, polytope_faces,
                                        polytope_vertices, project_point,
                                        ring_cell_points, rotate4)
    from .polytopes import wythoff as _wy
except ImportError:  # flat import outside the package
    from polytopes.regular import (COUNTS, _cell_inradius,
                                       _half_filter, _leonardo_panels,
                                       _pole_angles, _slerp4, add_sphere,
                                       add_strut, dual_vertices,
                                       polytope_edges, polytope_faces,
                                       polytope_vertices, project_point,
                                       ring_cell_points, rotate4)
    from polytopes import wythoff as _wy


# the 15 non-empty ringings, in a stable menu order
_RING_MASKS = [tuple(int(c) for c in format(m, '04b'))
               for m in range(1, 16)]


def wythoff_kind(family, rings):
    """The `kind` string naming a uniform polychoron."""
    return "W:%s:%s" % (family, ''.join(str(b) for b in rings))


try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, IntProperty,
                           BoolProperty)
    import bmesh
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def build_polytope(kind='CELL8', style='CURVED', proj_dist=1.05,
                   rot_xw=0.0, rot_yw=0.0, rot_zw=0.0, rot_xy=0.0,
                   arc_segments=12, radius=0.03, sides=6, taper=True,
                   vertex_spheres=True, sphere_factor=1.6, scale=1.0,
                   render='EDGES', border=0.35, panel_thickness=0.03):
    """Original interface, kept for compatibility: the plain single
    framework with no cutaway, compound or rings."""
    verts, faces, _mat, st = build_polytope_ex(
        kind, style, proj_dist, rot_xw, rot_yw, rot_zw, rot_xy,
        arc_segments, radius, sides, taper, vertex_spheres,
        sphere_factor, scale, render, border, panel_thickness)
    return verts, faces, st['nv'], st['ne']


# Stays in the Blender layer, not the engine: the ring form needs
# bmesh, so this function branches on `_IN_BLENDER` and calls the
# bmesh-backed `_hull_geometry`.  A function that asks whether
# Blender is present is presentation, not mathematics.
def build_polytope_ex(kind='CELL8', style='CURVED', proj_dist=1.05,
                      rot_xw=0.0, rot_yw=0.0, rot_zw=0.0, rot_xy=0.0,
                      arc_segments=12, radius=0.03, sides=6,
                      taper=True, vertex_spheres=True,
                      sphere_factor=1.6, scale=1.0, render='EDGES',
                      border=0.35, panel_thickness=0.03, half=False,
                      dual_compound=False, rings=0,
                      ring_cell_scale=0.9, rings_only=False):
    """Superset of build_polytope: equatorial cutaway ('half'), the
    primal + dual compound, and solid Hopf rings of the 120-cell
    (rings need bmesh, so they only build inside Blender).
    Returns (verts, faces, face_mat, stats): face_mat gives one
    material-slot index per face (0 primal, 1 dual if present, then
    one slot per ring); stats is a dict of element counts."""
    systems = []                        # (V4, E, F2) per framework
    if kind.startswith('P:'):
        _, which, seed, hgt = kind.split(':')
        V4, E = (_wy.hyperprism(seed, float(hgt)) if which == 'PRISM'
                 else _wy.hyperpyramid(seed, float(hgt)))
        F2 = None
        dual_compound = False
        rings = 0
    elif kind.startswith('W:'):
        # A uniform (semi-regular) polychoron, built from its Coxeter
        # diagram by the kaleidoscope in polytopes/wythoff.py.  Only
        # vertices and edges exist for these -- extracting the 2-faces of
        # an omnitruncated 120-cell is a different order of computation --
        # so the panel renderer is not offered and dual / ring forms,
        # which are defined per regular polytope, are skipped.
        _, fam, bits = kind.split(':')
        V4, E = _wy.build(fam, tuple(int(c) for c in bits))
        F2 = None
        dual_compound = False
        rings = 0
    else:
        V4 = polytope_vertices(kind)
        E = polytope_edges(V4)
        F2 = (polytope_faces(kind, V4, E) if render == 'LEONARDO'
              else None)
    if half:
        V4, E, F2 = _half_filter(V4, E, F2)
    systems.append((V4, E, F2))
    if dual_compound:
        D4, dkind, dkey = dual_vertices(kind)
        ED = polytope_edges(D4)
        FD = (polytope_faces(dkind, D4, ED, cache_key=dkey)
              if render == 'LEONARDO' else None)
        if style != 'CURVED':
            # dual vertices sit at the primal's cell centers; in
            # curved mode both frameworks live on the unit 3-sphere
            r = _cell_inradius(kind)
            D4 = [tuple(x * r for x in v) for v in D4]
        if half:
            D4, ED, FD = _half_filter(D4, ED, FD)
        systems.append((D4, ED, FD))
    cells = []
    if rings > 0 and kind == 'CELL120' and _IN_BLENDER:
        cells = ring_cell_points(rings, ring_cell_scale, half)
    if rings_only and cells:
        # the book's fig 3-29 look: rings of dodecahedra standing
        # alone, without the 1200-strut edge framework around them
        systems = []
    # the same 4D rotation for every point set, so compounds and
    # rings stay aligned with the primal framework
    systems = [(rotate4(V, rot_xw, rot_yw, rot_zw, rot_xy), Es, Fs)
               for (V, Es, Fs) in systems]
    cells = [(ri, rotate4(P, rot_xw, rot_yw, rot_zw, rot_xy))
             for (ri, P) in cells]
    if style == 'CURVED':
        # exact stereographic projection: pole ON the unit 3-sphere
        dist = 1.0
        allv = ([v for (V, _e, _f) in systems for v in V]
                or [p for (_ri, P) in cells for p in P])
        pa, pb = _pole_angles(allv)
        if pa != 0.0 or pb != 0.0:
            systems = [(rotate4(V, pa, pb, 0.0, 0.0), Es, Fs)
                       for (V, Es, Fs) in systems]
            cells = [(ri, rotate4(P, pa, pb, 0.0, 0.0))
                     for (ri, P) in cells]
    else:
        dist = max(proj_dist, 1.001)
    verts = []
    faces = []
    edges = []                              # only the Wireframe render
    face_mat = []
    for mi, (V, Es, Fs) in enumerate(systems):
        proj = {}
        for i, v in enumerate(V):
            p, s = project_point(v, dist)
            proj[i] = (tuple(c * scale for c in p), s)
        nf0 = len(faces)
        if render == 'LEONARDO':
            # flat panels per 2D face: stereographic projection maps
            # the circle through a face's vertices to a circle in
            # R^3, so every projected face is planar in both styles
            nvp = len(V)
            O = tuple(sum(proj[i][0][k] for i in range(nvp)) / nvp
                      for k in range(3))
            pv, pf = _leonardo_panels(Fs, proj, O, border,
                                      panel_thickness, taper,
                                      scale)
            base = len(verts)
            verts.extend(pv)
            faces.extend([[base + i for i in f] for f in pf])
        else:
            for (i, j) in Es:
                if style == 'CURVED':
                    pts = []
                    scls = []
                    for k in range(arc_segments + 1):
                        t = k / arc_segments
                        q = _slerp4(V[i], V[j], t)
                        p, s = project_point(q, dist)
                        pts.append(tuple(c * scale for c in p))
                        scls.append(s)
                else:
                    pts = [proj[i][0], proj[j][0]]
                    scls = [proj[i][1], proj[j][1]]
                    if arc_segments > 1:
                        # subdivide straight edges too (for tapering)
                        a, b = pts
                        sa, sb = scls
                        pts = [tuple(a[k]
                                     + (b[k] - a[k]) * t / arc_segments
                                     for k in range(3))
                               for t in range(arc_segments + 1)]
                        scls = [sa + (sb - sa) * t / arc_segments
                                for t in range(arc_segments + 1)]
                if render == 'WIREFRAME':
                    # bare polyline along the (possibly curved) edge
                    base = len(verts)
                    verts.extend(pts)
                    for a in range(len(pts) - 1):
                        edges.append((base + a, base + a + 1))
                    continue
                if taper:
                    radii = [radius * s * scale for s in scls]
                else:
                    radii = [radius * scale] * len(pts)
                # "Struts" matches the square-section beams the other
                # generators get from the Wireframe modifier; Ball and
                # Stick keeps its round, user-adjustable cross-section
                strut_sides = 4 if render == 'EDGES' else sides
                add_strut(verts, faces, pts, radii, strut_sides)
            if render == 'BALLSTICK':
                for i in range(len(V)):
                    p, s = proj[i]
                    r = (radius * sphere_factor
                         * (s if taper else 1.0) * scale)
                    add_sphere(verts, faces, p, r)
        face_mat.extend([mi] * (len(faces) - nf0))
    ring_base = len(systems)
    n_cells = 0
    for (ri, P) in cells:
        pts = [tuple(c * scale for c in project_point(p, dist)[0])
               for p in P]
        hv, hf = _hull_geometry(pts)
        base = len(verts)
        verts.extend(hv)
        faces.extend([[base + i for i in f] for f in hf])
        face_mat.extend([ring_base + ri] * len(hf))
        n_cells += 1
    stats = {'nv': (len(systems[0][0]) if systems else 0),
             'ne': (len(systems[0][1]) if systems else 0),
             'dual_nv': (len(systems[1][0])
                         if dual_compound and len(systems) > 1 else 0),
             'dual_ne': (len(systems[1][1])
                         if dual_compound and len(systems) > 1 else 0),
             'n_systems': len(systems), 'n_cells': n_cells,
             'n_rings': (min(rings, 12) if cells else 0),
             'wire_edges': edges}
    return verts, faces, face_mat, stats


if _IN_BLENDER:

    def _hull_geometry(pts):
        """Convex hull (verts, faces) of a 3D point cloud via bmesh.
        Each projected dodecahedral ring cell is convex (central
        projection of a convex cell from a point beyond it), so its
        hull is exactly the solid cell."""
        bm = bmesh.new()
        for p in pts:
            bm.verts.new(p)
        bmesh.ops.convex_hull(bm, input=bm.verts[:])
        bm.verts.index_update()
        used = sorted({v.index for f in bm.faces for v in f.verts})
        remap = {vi: n for n, vi in enumerate(used)}
        bm.verts.ensure_lookup_table()
        hv = [tuple(bm.verts[vi].co) for vi in used]
        hf = [[remap[v.index] for v in f.verts] for f in bm.faces]
        bm.free()
        return hv, hf

    def _make_material(name, rgba):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = rgba
        mat.use_nodes = True
        node = mat.node_tree.nodes.get('Principled BSDF')
        if node is not None:
            node.inputs['Base Color'].default_value = rgba
        return mat

    def _ring_color(ri):
        """Distinct hue per Hopf ring (12 around the color wheel)."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb((ri / 12.0) % 1.0, 0.75, 0.9)
        return (r, g, b, 1.0)

    class MESH_OT_polytope4d_add(bpy.types.Operator):
        """Regular 4-polytope edge framework, projected to 3D with
        straight or stereographically curved edges"""
        bl_idname = "mesh.polytope4d_add"
        bl_label = "4D Polytope"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Polytope",
            items=[('CELL5', "5-cell", "4-simplex: 5 vertices, 10 edges"),
                   ('CELL8', "Tesseract (8-cell)", "16 vertices, 32 edges"),
                   ('CELL16', "16-cell", "8 vertices, 24 edges"),
                   ('CELL24', "24-cell", "24 vertices, 96 edges"),
                   ('CELL120', "120-cell", "600 vertices, 1200 edges"),
                   ('CELL600', "600-cell", "120 vertices, 720 edges")],
            default='CELL8',
            description="Which of the six regular convex 4-polytopes to "
                        "project into 3D")
        form: EnumProperty(
            name="Form",
            items=[('REGULAR', "Regular",
                    "One of the six regular convex 4-polytopes"),
                   ('UNIFORM', "Uniform (semi-regular)",
                    "A vertex-transitive polytope from Wythoff's "
                    "kaleidoscope: the truncations, rectifications and "
                    "expansions of the regular ones, and the "
                    "4-dimensional analogues of the Archimedean solids"),
                   ('PRISM', "Hyperprism",
                    "A polyhedron translated along the fourth axis; the "
                    "cube's hyperprism is the tesseract"),
                   ('PYRAMID', "Hyperpyramid",
                    "A polyhedron joined to a single apex off its "
                    "hyperplane; the tetrahedron's is the 5-cell")],
            default='REGULAR',
            description="Build a regular polytope, a uniform one, or one "
                        "of the two prismatic families")
        seed: EnumProperty(
            name="Base",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", "")],
            default='CUBE',
            description="The 3-dimensional polyhedron the hyperprism or "
                        "hyperpyramid is raised on")
        seed_height: FloatProperty(
            name="Fourth-Axis Extent", default=1.0, min=0.05, max=10.0,
            description="How far the second copy is translated, or how "
                        "far the apex stands off, along the fourth axis")
        family: EnumProperty(
            name="Symmetry",
            items=[(k, _wy.FAMILIES[k][0], "order %d"
                    % _wy.GROUP_ORDER[k]) for k in ('A4', 'B4', 'F4', 'H4')],
            default='H4',
            description="Which rank-4 reflection group provides the "
                        "mirrors (uniform only)")
        wythoff_rings: EnumProperty(
            name="Ringed Nodes",
            items=[(''.join(str(b) for b in bits),
                    ''.join(str(b) for b in bits),
                    "Hold the generating point off mirror(s) %s"
                    % ', '.join(str(i + 1) for i in range(4) if bits[i]))
                   for bits in _RING_MASKS],
            default='1000',
            description="Which nodes of the Coxeter diagram are ringed. "
                        "The generating point stands off a ringed mirror "
                        "and lies on an unringed one, so the ringing "
                        "picks out the polytope (uniform only)")
        style: EnumProperty(
            name="Edges",
            items=[('CURVED', "Curved (stereographic)",
                    "Vertices on the 3-sphere, edges as great-circle "
                    "arcs, stereographically projected: circular arcs"),
                   ('STRAIGHT', "Straight (perspective)",
                    "Direct 4D perspective projection; small distance "
                    "approaches a Schlegel diagram")],
            default='CURVED',
            description="Straight 4D perspective edges, or great-circle "
                        "arcs from stereographic projection")
        proj_dist: FloatProperty(
            name="Projection Distance", default=1.05, min=1.001, max=10.0,
            description="Eye distance along w for STRAIGHT edges (near 1 "
                        "= Schlegel diagram; for the 5/16/600-cell a "
                        "vertex sits at w=1, so rotate a little or back "
                        "off the distance). Curved mode always uses the "
                        "exact stereographic projection")
        rot_xw: FloatProperty(name="Rotate XW", default=0.0,
                              min=-180.0, max=180.0,
                              description="Rotation in the XW plane before "
                              "projecting from 4D, in degrees")
        rot_yw: FloatProperty(name="Rotate YW", default=0.0,
                              min=-180.0, max=180.0,
                              description="Rotation in the YW plane before "
                              "projecting from 4D, in degrees")
        rot_zw: FloatProperty(name="Rotate ZW", default=0.0,
                              min=-180.0, max=180.0,
                              description="Rotation in the ZW plane before "
                              "projecting from 4D, in degrees")
        rot_xy: FloatProperty(name="Rotate XY", default=0.0,
                              min=-180.0, max=180.0,
                              description="Rotation in the XY plane before "
                              "projecting from 4D, in degrees")
        arc_segments: IntProperty(
            name="Arc Segments", default=12, min=1, max=48,
            description="Samples per edge (curved edges and tapering)")
        radius: FloatProperty(name="Strut Radius", default=0.03,
                              min=0.002, max=0.5, step=1, precision=3,
                              description="Radius of the edge struts")
        sides: IntProperty(name="Strut Sides", default=6, min=3, max=16,
                           description="Cross-section sides of each round "
                           "strut (Ball and Stick style)")
        taper: BoolProperty(
            name="Taper With Projection", default=True,
            description="Scale strut thickness by the local projection "
                        "factor (near-the-pole features fatter)")
        vertex_spheres: BoolProperty(name="Vertex Spheres", default=True,
                                     description="Place a sphere at each "
                                     "vertex")
        sphere_factor: FloatProperty(name="Sphere Size", default=1.6,
                                     min=1.0, max=4.0,
                                     description="Vertex sphere size "
                                     "relative to the strut radius (Ball "
                                     "and Stick style)")
        render: EnumProperty(
            name="Style",
            items=[('EDGES', "Struts",
                    "Solid tubes along the projected edges (no vertex "
                    "spheres) -- the same edge-strut style as the "
                    "other polyhedron generators, following the curved "
                    "stereographic arcs"),
                   ('BALLSTICK', "Ball and Stick",
                    "Struts along the projected edges with a sphere "
                    "at every vertex (ball-and-stick model); the "
                    "struts still follow the curved stereographic "
                    "arcs"),
                   ('WIREFRAME', "Wireframe",
                    "Projected edges as a bare wireframe of polylines "
                    "(no solid struts or spheres)"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "A flat open panel per 2D face of the polytope "
                    "(projected faces are planar in both edge "
                    "styles, since stereographic projection maps "
                    "the circle through a face's vertices to a "
                    "circle)")],
            default='EDGES',
            description="How the projected framework is built: struts, "
                        "ball-and-stick, wireframe, or open panels")
        border: FloatProperty(
            name="Border", default=0.06, min=0.005, max=1.0,
            description="Leonardo panel frame width (fraction of "
                        "face whatever its size")
        panel_thickness: FloatProperty(
            name="Panel Thickness", default=0.03, min=0.002, max=0.5,
            step=1, precision=3,
            description="Thickness of the Leonardo face panels")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the framework")
        half: BoolProperty(
            name="Half (Cutaway)", default=False,
            description="Keep only the elements on one side of the "
                        "equatorial hyperplane (w <= 0, equator "
                        "included) before projection -- Segerman's "
                        "legible 'half of a 24/120/600-cell' models")
        dual_compound: BoolProperty(
            name="Dual Compound", default=False,
            description="Also build the dual polytope (5<->5, "
                        "8<->16, 24<->24, 120<->600), its vertices "
                        "at this polytope's cell centers, in a "
                        "second material slot")
        rings: IntProperty(
            name="Hopf Rings", default=0, min=0, max=12,
            description="120-cell only: render N of the 12 rings of "
                        "10 dodecahedral cells that partition the "
                        "120-cell along Hopf fibers, as solid "
                        "shrunken cells with one material per ring")
        ring_cell_scale: FloatProperty(
            name="Ring Cell Scale", default=0.9, min=0.1, max=1.0,
            description="Shrink factor of each ring cell toward its "
                        "4D centroid (gaps make the rings legible)")
        rings_only: BoolProperty(
            name="Rings Only", default=False,
            description="Drop the edge framework and show just the "
                        "rings of solid cells (fig 3-29 style; also "
                        "the printable version)")

        def execute(self, context):
            if self.form == 'UNIFORM':
                kind = wythoff_kind(self.family, self.wythoff_rings)
            elif self.form in ('PRISM', 'PYRAMID'):
                kind = "P:%s:%s:%.6f" % (self.form, self.seed,
                                         self.seed_height)
            else:
                kind = self.kind
            verts, faces, face_mat, st = build_polytope_ex(
                kind, self.style, self.proj_dist, self.rot_xw,
                self.rot_yw, self.rot_zw, self.rot_xy,
                self.arc_segments, self.radius, self.sides, self.taper,
                self.vertex_spheres, self.sphere_factor, self.scale,
                self.render, self.border, self.panel_thickness,
                self.half, self.dual_compound, self.rings,
                self.ring_cell_scale, self.rings_only)
            # center on the origin and fit within a 2 m cube (times the
            # Scale property), so the framework fills the cube by default
            if verts:
                xs = [v[0] for v in verts]
                ys = [v[1] for v in verts]
                zs = [v[2] for v in verts]
                cen = (0.5 * (min(xs) + max(xs)),
                       0.5 * (min(ys) + max(ys)),
                       0.5 * (min(zs) + max(zs)))
                ext = max(max(xs) - min(xs), max(ys) - min(ys),
                          max(zs) - min(zs))
                s = (2.0 * self.scale / ext) if ext > 1e-9 else 1.0
                verts = [((v[0] - cen[0]) * s, (v[1] - cen[1]) * s,
                          (v[2] - cen[2]) * s) for v in verts]
            me = bpy.data.meshes.new("Polytope4D")
            me.from_pydata(verts, st.get('wire_edges', []), faces)
            n_rings = st['n_rings']
            if st['n_systems'] > 1 or n_rings > 0:
                mats = []
                if st['n_systems'] >= 1:
                    mats.append(_make_material(
                        "Polytope4D Primal", (0.75, 0.78, 0.85, 1.0)))
                if st['n_systems'] > 1:
                    mats.append(_make_material(
                        "Polytope4D Dual", (0.9, 0.45, 0.15, 1.0)))
                for ri in range(n_rings):
                    mats.append(_make_material(
                        "Polytope4D Ring %d" % (ri + 1),
                        _ring_color(ri)))
                for mat in mats:
                    me.materials.append(mat)
                me.polygons.foreach_set('material_index', face_mat)
            me.validate(clean_customdata=True)
            # Only Ball and Stick's round cylinders (and spheres) shade
            # smooth; the square-section "Struts" beams, flat Leonardo
            # panels and solid ring cells must stay flat, or the square
            # profile reads as a round tube
            smooth = self.render == 'BALLSTICK'
            if n_rings > 0 and smooth:
                rb = st['n_systems']
                flags = [m < rb for m in face_mat]
                me.polygons.foreach_set('use_smooth',
                                        flags[:len(me.polygons)])
            else:
                me.polygons.foreach_set(
                    'use_smooth', [smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Polytope4D", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if st['n_systems']:
                msg = f"{st['nv']} vertices, {st['ne']} edges"
                if self.half:
                    msg += " (half)"
            else:
                msg = "rings only"
            if st['n_systems'] > 1:
                msg += (f" + dual {st['dual_nv']} vertices, "
                        f"{st['dual_ne']} edges")
            if n_rings > 0:
                msg += (f" + {n_rings} Hopf rings, "
                        f"{st['n_cells']} cells")
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'form')
            if self.form == 'UNIFORM':
                lay.prop(self, 'family')
                lay.prop(self, 'wythoff_rings')
                nv = _wy.expected_vertices(
                    self.family,
                    tuple(int(c) for c in self.wythoff_rings))
                lay.label(text="%d vertices" % nv)
            elif self.form in ('PRISM', 'PYRAMID'):
                lay.prop(self, 'seed')
                lay.prop(self, 'seed_height')
            else:
                lay.prop(self, 'kind')
            lay.prop(self, 'style')
            if self.style == 'STRAIGHT':
                lay.prop(self, 'proj_dist')
            col = lay.column(align=True)
            for k in ('rot_xw', 'rot_yw', 'rot_zw', 'rot_xy'):
                col.prop(self, k)
            lay.prop(self, 'render')
            if self.render == 'LEONARDO':
                for k in ('border', 'panel_thickness', 'taper',
                          'scale'):
                    lay.prop(self, k)
            elif self.render == 'WIREFRAME':
                # bare polylines: only arc smoothness and overall scale
                lay.prop(self, 'arc_segments')
                lay.prop(self, 'scale')
            else:
                # EDGES ("Struts") = square-section beams, no spheres;
                # BALLSTICK = round struts (adjustable sides) + spheres.
                # The vertex_spheres toggle is retired -- the style now
                # decides -- so it is never drawn.
                keys = ['arc_segments', 'radius']
                if self.render == 'BALLSTICK':
                    keys.append('sides')
                keys.append('taper')
                if self.render == 'BALLSTICK':
                    keys.append('sphere_factor')
                keys.append('scale')
                for k in keys:
                    lay.prop(self, k)
            lay.separator()
            col = lay.column(align=True)
            col.label(text="Cutaway, Compound & Rings")
            col.prop(self, 'half')
            # the dual and the Hopf rings are defined per regular
            # polytope, so they are not offered for the uniform ones
            sub = col.column(align=True)
            sub.enabled = (self.form == 'REGULAR')
            sub.prop(self, 'dual_compound')
            row = sub.row(align=True)
            row.enabled = (self.form == 'REGULAR'
                           and self.kind == 'CELL120')
            row.prop(self, 'rings')
            if (self.form == 'REGULAR' and self.kind == 'CELL120'
                    and self.rings > 0):
                col.prop(self, 'ring_cell_scale')
                col.prop(self, 'rings_only')

    def _menu_func(self, context):
        self.layout.operator("mesh.polytope4d_add", icon='MESH_CUBE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polytope4d_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polytope4d_add)


def _selftest():
    bad = []
    for kind, (env, ene) in COUNTS.items():
        V = polytope_vertices(kind)
        E = polytope_edges(V)
        ok = (len(V), len(E)) == (env, ene)
        print(f"{kind:8s}: V={len(V):4d} E={len(E):5d}  "
              f"expect {env},{ene}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(kind)
    assert not bad
