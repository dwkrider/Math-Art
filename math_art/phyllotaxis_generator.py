
# Phyllotaxis / Fibonacci Spiral Form generator for Blender
#
# Places N florets by the golden-angle (Vogel) model of spiral
# phyllotaxis: the i-th primordium sits at polar angle
#
#       theta_i = i * 137.50776...  degrees                       (1)
#
# where 137.50776 deg = 360 deg * (2 - phi) = 360 deg / phi^2 is the
# golden angle (phi = (1 + sqrt5)/2 the golden ratio), and at radius
#
#       r_i = c * sqrt(i)                                          (2)
#
# so that equal numbers of florets fall in equal areas and the packing
# is uniform.  Because the divergence (1) is the "most irrational"
# fraction of a turn, no two florets ever line up radially; instead the
# eye reads two families of counter-winding spirals (parastichies)
# whose counts are consecutive Fibonacci numbers (8/13, 13/21, 21/34,
# ...), exactly as in a sunflower head, pine cone or cactus areole.
#
# The flat Vogel disk is optionally draped over a surface -- a
# paraboloid dome or cone (a cactus body), or replaced by the
# equal-area spherical Fibonacci lattice
#
#       z_i = 1 - 2*(i + 1/2)/N,  rho_i = sqrt(1 - z_i^2),         (3)
#       (x, y) = rho_i * (cos theta_i, sin theta_i)
#
# (a pineapple / spherical seed head).  A CREST form corrugates the
# disk into parallel ridges, the fasciated ("cristate") growth Fathauer
# builds as brain-like crested-cactus ceramics: a line of growth rather
# than a point.  A small oriented floret -- a flattened seed bump, an
# areole spike, or a disc floret -- is instanced at every point,
# tangent to the surface, its size following the local packing spacing
# (optionally graded so the central florets are smaller).  Reversing the
# sign of (1) flips the handedness (chirality) of the whole head.
#
# Coloring by parastichy number k tints each residue class i mod k a
# different color, so choosing a Fibonacci k lights up that family of
# spiral arms.
#
# References:
#   - H. Vogel, "A better way to construct the sunflower head",
#     Mathematical Biosciences 44 (1979), pp. 179-189 -- the
#     r = c*sqrt(i), theta = i*137.5 deg model (equations (1)-(2)).
#   - S. Douady & Y. Couder, "Phyllotaxis as a Physical Self-Organized
#     Growth Process", Phys. Rev. Lett. 68 (1992), pp. 2098-2101 --
#     the golden angle as the dynamical attractor of primordium
#     packing.
#   - R. V. Jean, "Phyllotaxis: A Systemic Study in Plant Morphogenesis",
#     Cambridge University Press, 1994 (parastichy / Fibonacci counts).
#   - H. S. M. Coxeter, "Introduction to Geometry", 2nd ed., Wiley,
#     1969, Sec. 11.5 -- the golden angle 360/phi^2 and continued
#     fractions.
#   - R. W. Fathauer, "Crested Cactuses and Mathematical Sculpture",
#     Bridges 2022 Conference Proceedings, pp. 111-118 -- phyllotactic
#     and fasciated ("cristate", line-of-growth) ceramic forms.
#   - A. Gonzalez, "Measurement of areas on a sphere using Fibonacci
#     and latitude-longitude lattices", Math. Geosci. 42 (2010),
#     pp. 49-64 -- the equal-area spherical Fibonacci lattice (3).

bl_info = {
    "name": "Phyllotaxis",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Odds & Ends",
    "description": "Golden-angle (Vogel) phyllotaxis florets on a disk, "
                   "dome, cone, sphere or crest, with parastichy "
                   "coloring and chirality",
    "category": "Add Mesh",
}





# --------------------------------------------------------------------
# Pure-math core (no bpy)
# --------------------------------------------------------------------





# ---- floret templates (unit shapes in a local t1/t2/normal frame) ----

try:
    from .surfaces.primitives import icosphere as _icosphere_shared
except ImportError:  # flat import outside the package
    from surfaces.primitives import icosphere as _icosphere_shared
# The mathematics lives in the sibling `lsystem` engine package;
# this module is the Blender layer over it.
try:
    from .lsystem.phyllotaxis import (GOLDEN_ANGLE_DEG,
                                          build_phyllotaxis, build_points,
                                          nearest_neighbor_dist,
                                          phyllotaxis_points)
except ImportError:  # flat import outside the package
    from lsystem.phyllotaxis import (GOLDEN_ANGLE_DEG,
                                         build_phyllotaxis, build_points,
                                         nearest_neighbor_dist,
                                         phyllotaxis_points)




















# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PALETTE = [
        (0.90, 0.75, 0.30), (0.82, 0.45, 0.28), (0.55, 0.68, 0.35),
        (0.35, 0.55, 0.62), (0.72, 0.38, 0.45), (0.48, 0.42, 0.66),
        (0.86, 0.62, 0.42), (0.40, 0.62, 0.50),
    ]

    def _mat(name, col):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        b = next(nd for nd in mat.node_tree.nodes
                 if nd.type == 'BSDF_PRINCIPLED')
        b.inputs["Base Color"].default_value = (*col, 1)
        b.inputs["Roughness"].default_value = 0.5
        # also set the solid-viewport color so the coloring is visible
        # in Solid / Workbench shading, not only in Material Preview
        mat.diffuse_color = (*col, 1)
        return mat

    def _materials(me, tags, color_by):
        if color_by == 'UNIFORM' or not tags:
            me.materials.append(_mat("Phyllotaxis", (0.86, 0.72, 0.42)))
            return
        nb = max(tags) + 1
        for i in range(nb):
            me.materials.append(
                _mat("Phyllotaxis %d" % i, _PALETTE[i % len(_PALETTE)]))
        if len(tags) == len(me.polygons):
            me.polygons.foreach_set("material_index", tags)

    class MESH_OT_phyllotaxis_add(bpy.types.Operator):
        """Add a golden-angle phyllotaxis form: N florets placed by the
        Vogel spiral (theta = i*137.5 deg, r = sqrt(i)) on a disk, dome,
        cone, sphere or crest, with emergent Fibonacci spiral arms"""
        bl_idname = "mesh.phyllotaxis_add"
        bl_label = "Phyllotaxis"
        bl_options = {'REGISTER', 'UNDO'}

        count: IntProperty(
            name="Florets", default=500, min=1, max=4000,
            description="Number of florets placed by the golden-angle "
                        "spiral")
        form: EnumProperty(
            name="Form",
            description="Surface the florets are placed on",
            items=[('DISK', "Disk (sunflower)",
                    "Flat Vogel disk, the classic sunflower head"),
                   ('DOME', "Dome (cactus)",
                    "Florets draped over a paraboloid cap"),
                   ('CONE', "Cone",
                    "Florets draped over a cone"),
                   ('SPHERE', "Sphere (pineapple)",
                    "Equal-area spherical Fibonacci lattice"),
                   ('CREST', "Crest (cristate)",
                    "Corrugated ridges: the fasciated crested-cactus "
                    "growth")],
            default='DISK')
        floret: EnumProperty(
            name="Floret",
            description="Shape placed at each floret point",
            items=[('BUMP', "Seed Bump",
                    "A flattened spheroid seed at each point"),
                   ('SPIKE', "Areole Spike",
                    "A cone spike pointing out along the surface"),
                   ('DISC', "Disc Floret",
                    "A flat disc tangent to the surface")],
            default='BUMP')
        output: EnumProperty(
            name="Output",
            description="Floret solids, or just the center points for "
                        "instancing",
            items=[('SOLID', "Floret Solids",
                    "A mesh of floret solids (bumps / spikes / discs)"),
                   ('POINTS', "Points Only",
                    "Just a vertex at each floret center -- for "
                    "instancing your own object. Carries point "
                    "attributes: parastichy (int), surface_normal "
                    "(vector), floret_size (float)")],
            default='SOLID')
        chirality: EnumProperty(
            name="Handedness",
            description="Winding direction of the spiral",
            items=[('RIGHT', "Right", "Counterclockwise spiral"),
                   ('LEFT', "Left", "Clockwise (mirror) spiral")],
            default='RIGHT')
        divergence: FloatProperty(
            name="Divergence", default=GOLDEN_ANGLE_DEG,
            min=1.0, max=179.0,
            description="Angle between successive florets in degrees "
                        "(137.50776 = the golden angle; small "
                        "detunings change the parastichy counts)")
        dome_height: FloatProperty(
            name="Height", default=0.8, min=0.0, max=4.0,
            description="Height of the dome / cone / crest (ignored for "
                        "Disk and Sphere)")
        crest_waves: FloatProperty(
            name="Crest Waves", default=3.0, min=0.5, max=12.0,
            description="Number of parallel ridges across the Crest form")
        crest_amp: FloatProperty(
            name="Crest Depth", default=0.35, min=0.0, max=1.5,
            description="Ridge amplitude of the Crest form")
        fill: FloatProperty(
            name="Floret Fill", default=0.9, min=0.05, max=2.5,
            description="Floret size as a fraction of the local packing "
                        "spacing (>1 makes florets overlap)")
        size_grade: FloatProperty(
            name="Size Grading", default=0.0, min=0.0, max=0.95,
            description="Shrink the central florets relative to the rim "
                        "(0 = uniform size)")
        spike_ratio: FloatProperty(
            name="Spike Height", default=2.2, min=0.2, max=8.0,
            description="Areole-spike height as a multiple of its base "
                        "radius")
        flatten: FloatProperty(
            name="Bump Flatten", default=0.55, min=0.05, max=1.0,
            description="Seed-bump thickness along the normal (1 = a "
                        "full sphere)")
        subdiv: IntProperty(
            name="Bump Detail", default=1, min=0, max=3,
            description="Icosphere subdivisions for the seed bumps")
        seg: IntProperty(
            name="Segments", default=12, min=3, max=48,
            description="Sides of the spike / disc florets")
        color_by: EnumProperty(
            name="Color By",
            description="How the florets are colored",
            items=[('PARASTICHY', "Parastichy Arms",
                    "Color each residue class i mod k, lighting up the "
                    "k-arm family of spirals (pick a Fibonacci k)"),
                   ('RING', "Radial Rings",
                    "Color by distance from the center"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='PARASTICHY')
        parastichy: IntProperty(
            name="Parastichy k", default=13, min=1, max=144,
            description="Which spiral-arm family to color; a Fibonacci "
                        "number (8, 13, 21, 34...) matches the natural "
                        "parastichies")
        smooth_shading: BoolProperty(
            name="Smooth Shading", default=True,
            description="Shade the floret solids smooth rather than "
                        "faceted")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 fits a "
                        "2 m cube centered on the origin)")

        def _place(self, context, me, name):
            obj = bpy.data.objects.new(name, me)
            context.scene.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            obj["math_art_pattern"] = True
            return obj

        def _execute_points(self, context):
            import numpy as np
            P, Nrm, size, cls = build_points(
                self.count, self.form, self.divergence, self.dome_height,
                self.chirality, self.crest_waves, self.crest_amp,
                self.fill, self.size_grade, self.color_by,
                self.parastichy)
            lo, hi = P.min(axis=0), P.max(axis=0)
            c = 0.5 * (lo + hi)
            ext = float(max(hi[0] - lo[0], hi[1] - lo[1],
                            hi[2] - lo[2])) or 1.0
            s = 2.0 * self.scale / ext
            verts = [tuple((P[i] - c) * s) for i in range(len(P))]
            me = bpy.data.meshes.new("Phyllotaxis Points")
            me.from_pydata(verts, [], [])
            a = me.attributes.new("parastichy", 'INT', 'POINT')
            a.data.foreach_set("value", [int(v) for v in cls])
            a = me.attributes.new("surface_normal", 'FLOAT_VECTOR',
                                  'POINT')
            a.data.foreach_set("vector",
                               np.asarray(Nrm, float).ravel())
            a = me.attributes.new("floret_size", 'FLOAT', 'POINT')
            a.data.foreach_set("value",
                               [float(v) * s for v in size])
            me.update()
            obj = self._place(context, me, "Phyllotaxis Points")
            self.report({'INFO'}, "Phyllotaxis %s: %d points" %
                        (self.form, len(me.vertices)))
            return {'FINISHED'}

        def execute(self, context):
            if self.output == 'POINTS':
                return self._execute_points(context)
            verts, faces, tags, fl_smooth = build_phyllotaxis(
                self.count, self.form, self.floret, self.divergence,
                self.dome_height, self.chirality, self.crest_waves,
                self.crest_amp, self.fill, self.size_grade,
                self.spike_ratio, self.flatten, self.subdiv, self.seg,
                self.color_by, self.parastichy)
            if not verts:
                self.report({'ERROR'}, "no florets generated")
                return {'CANCELLED'}

            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            cz = 0.5 * (min(zs) + max(zs))
            ext = max(max(xs) - min(xs), max(ys) - min(ys),
                      max(zs) - min(zs)) or 1.0
            s = 2.0 * self.scale / ext
            verts = [((v[0] - cx) * s, (v[1] - cy) * s,
                      (v[2] - cz) * s) for v in verts]

            me = bpy.data.meshes.new("Phyllotaxis")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=False)
            smooth = self.smooth_shading and fl_smooth
            me.polygons.foreach_set('use_smooth',
                                    [smooth] * len(me.polygons))
            if len(tags) == len(me.polygons):
                _materials(me, tags, self.color_by)
                att = me.attributes.new("parastichy", 'INT', 'FACE')
                att.data.foreach_set("value", tags)
            me.update()

            obj = bpy.data.objects.new("Phyllotaxis", me)
            context.scene.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            obj["math_art_pattern"] = True
            self.report(
                {'INFO'},
                "Phyllotaxis %s: %d florets  V=%d F=%d" %
                (self.form, self.count, len(me.vertices),
                 len(me.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'count')
            lay.prop(self, 'form')
            lay.prop(self, 'output')
            solid = self.output == 'SOLID'
            if solid:
                lay.prop(self, 'floret')
            lay.prop(self, 'chirality')
            lay.prop(self, 'divergence')
            if self.form in ('DOME', 'CONE', 'CREST'):
                lay.prop(self, 'dome_height')
            if self.form == 'CREST':
                lay.prop(self, 'crest_waves')
                lay.prop(self, 'crest_amp')
            lay.prop(self, 'fill')
            lay.prop(self, 'size_grade')
            if solid and self.floret == 'SPIKE':
                lay.prop(self, 'spike_ratio')
            if solid and self.floret == 'BUMP':
                lay.prop(self, 'flatten')
                lay.prop(self, 'subdiv')
            if solid and self.floret in ('SPIKE', 'DISC'):
                lay.prop(self, 'seg')
            lay.prop(self, 'color_by')
            if self.color_by == 'PARASTICHY':
                lay.prop(self, 'parastichy')
            if solid:
                lay.prop(self, 'smooth_shading')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.phyllotaxis_add",
                             icon='OUTLINER_OB_POINTCLOUD')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_phyllotaxis_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_phyllotaxis_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _self_test():
    import numpy as np
    ok = True

    # (0) the golden angle constant
    ga = abs(GOLDEN_ANGLE_DEG - 137.50776405003785) < 1e-9
    ok = ok and ga
    print("golden angle 360/phi^2 = %.10f deg : %s"
          % (GOLDEN_ANGLE_DEG, "OK" if ga else "BAD"))

    # (1) sphere points lie on the unit sphere, span both poles
    Ps, Ns, rs = phyllotaxis_points(800, 'SPHERE')
    rad = np.linalg.norm(Ps, axis=1)
    sph_ok = (abs(rad - 1.0).max() < 1e-9
              and Ps[:, 2].max() > 0.99 and Ps[:, 2].min() < -0.99)
    # sphere normal is the position
    norm_ok = np.linalg.norm(Ns - Ps, axis=1).max() < 1e-9
    ok = ok and sph_ok and norm_ok
    print("sphere on unit sphere & poles reached      : %s"
          % ("OK" if sph_ok and norm_ok else "BAD"))

    # (2) disk radius grows like sqrt(i): monotone, r_max ~ 1
    Pd, Nd, rd = phyllotaxis_points(1000, 'DISK')
    rr = np.hypot(Pd[:, 0], Pd[:, 1])
    mono = bool(np.all(np.diff(rr) > -1e-12))
    ok = ok and mono
    print("disk radius monotone in sqrt(i)            : %s"
          % ("OK" if mono else "BAD"))

    # (3) Vogel packing is near-uniform: low spread of NN spacing in
    # the interior (exclude the ragged outer ring)
    nn = nearest_neighbor_dist(Pd)
    interior = rr < 0.85 * rr.max()
    cov = nn[interior].std() / nn[interior].mean()
    unif = cov < 0.12
    ok = ok and unif
    print("Vogel spacing coeff of variation = %.3f     : %s"
          % (cov, "OK" if unif else "BAD"))

    # (4) chirality flips the winding sign of the first few steps
    Pr, _, _ = phyllotaxis_points(50, 'DISK', chirality='RIGHT')
    Pl, _, _ = phyllotaxis_points(50, 'DISK', chirality='LEFT')

    def wind(P):
        a = P[2, :2] - P[1, :2]
        b = P[3, :2] - P[2, :2]
        return np.sign(a[0] * b[1] - a[1] * b[0])
    chi = wind(Pr) == -wind(Pl) and wind(Pr) != 0
    ok = ok and chi
    print("chirality flips winding sign               : %s"
          % ("OK" if chi else "BAD"))

    # (5) surface normals are unit and outward (z>0) for open forms
    for fm in ('DISK', 'DOME', 'CONE', 'CREST'):
        _, N, _ = phyllotaxis_points(300, fm)
        un = abs(np.linalg.norm(N, axis=1) - 1.0).max() < 1e-9
        up = bool(np.all(N[:, 2] > 0.0))
        ok = ok and un and up
        print("normals unit & upward on %-5s            : %s"
              % (fm, "OK" if un and up else "BAD"))

    # (6) parastichy coloring produces exactly k residue classes
    v, f, tags, sm = build_phyllotaxis(400, form='DISK', floret='BUMP',
                                       subdiv=1, color_by='PARASTICHY',
                                       parastichy=13)
    # k=13 with an 8-color palette -> tags in 0..7, all present
    ntag = len(set(tags))
    par_ok = ntag == 8 and len(v) > 0 and len(f) > 0
    ok = ok and par_ok
    print("build parastichy tags distinct = %d          : %s"
          % (ntag, "OK" if par_ok else "BAD"))

    # (7) every floret type builds non-empty geometry on every form
    build_ok = True
    for fm in ('DISK', 'DOME', 'CONE', 'SPHERE', 'CREST'):
        for fl in ('BUMP', 'SPIKE', 'DISC'):
            vv, ff, tt, _ = build_phyllotaxis(120, form=fm, floret=fl,
                                              subdiv=0, seg=8)
            if not (vv and ff and len(tt) == len(ff)):
                build_ok = False
                print("   build BAD for form=%s floret=%s" % (fm, fl))
    ok = ok and build_ok
    print("all form x floret builds non-empty         : %s"
          % ("OK" if build_ok else "BAD"))

    print("RESULT:", "OK" if ok else "BAD")
    return ok
