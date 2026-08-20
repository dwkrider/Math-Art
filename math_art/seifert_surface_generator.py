
# Seifert Surface Generator for Blender
#
# Builds Seifert surfaces (and the other Kauffman-state spanning surfaces) of
# knots and links given as braid words.  A braid word on n strands is a sequence
# of crossings s_i / s_i^-1; Seifert's algorithm applied to the braid closure
# yields one disk per Seifert circle and one half-twisted band per crossing,
# stacked along z and joined by rotation-minimising-framed ribbons.  The surface
# boundary is exactly the knot/link (optionally emitted as a bevelled curve
# object); genus / boundary-component counts are computed combinatorially and
# verified against the finished mesh.
#
# The construction and finishing pipeline live in the `seifert` sub-package next
# to this file (NumPy only): braid word -> disk/band combinatorics -> quad mesh
# -> light particle-model relaxation -> Catmull-Clark refinement -> implicit
# mean-curvature fairing.  Topology is fixed by the first two stages and every
# later stage preserves it.  The heavy mean-curvature fairing (implicit fairing,
# Desbrun et al.) with the knot rim held fixed is what drives the surface to a
# discrete minimal surface spanning the knot.
#
# Braid word syntax: an uppercase letter is a right-hand crossing, the matching
# lowercase letter a left-hand one, and the letter selects the adjacent strand
# pair -- A = strands 1-2, B = 2-3, ...  A trailing integer is a repeat count,
# so "A3" is three successive crossings.  Signed-integer notation ("1 -2 1 -2")
# is also accepted: |v| picks the strand pair, the sign the handedness.  Presets
# cover the classic examples.
#
# References:
# - Seifert's algorithm and the genus of a knot: Herbert Seifert,
#   "Ueber das Geschlecht von Knoten", Math. Ann. 110, 1934.
# - Stacked disk/band layout: J. J. van Wijk & A. M. Cohen,
#   "Visualization of Seifert Surfaces", IEEE TVCG 12(4), 2006, 485-496.
# - Braid generators s_i after Emil Artin (braid groups B_n).
# - Non-orientable state surfaces: E. Kalfagianni, "State surfaces of links",
#   arXiv:1804.05281, Lemma 2.2 (bipartite orientability criterion).
# - Implicit mean-curvature fairing: M. Desbrun, M. Meyer, P. Schroeder,
#   A. H. Barr, "Implicit Fairing of Irregular Meshes using Diffusion and
#   Curvature Flow", SIGGRAPH 1999.  Cotangent weights: U. Pinkall & K. Polthier,
#   "Computing Discrete Minimal Surfaces and Their Conjugates", Experiment.
#   Math. 2(1), 1993.
# - Rotation-minimising band frames: W. Wang, B. Juttler, D. Zheng, Y. Liu,
#   "Computation of Rotation Minimizing Frames", ACM TOG 27(1), 2008.

bl_info = {
    "name": "Seifert Surface Generator",
    "author": "Math Art project",
    "version": (2, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Seifert Surface / N-panel 'Minimal Surfaces'",
    "description": "Seifert (and state) surfaces of knots/links from braid words",
    "category": "Add Mesh",
}

# The construction + optimisation package (NumPy only).
try:
    from . import seifert as sv
except Exception:                                    # standalone / no NumPy
    try:
        import seifert as sv
    except Exception:
        sv = None


#: Braid words in letter notation for the classic examples.
PRESETS = {
    'TREFOIL':    ("Trefoil (3_1)", "AAA"),
    'FIGURE8':    ("Figure-8 (4_1)", "AbAb"),
    'CINQUEFOIL': ("Cinquefoil (5_1)", "AAAAA"),
    'SEPTAFOIL':  ("(2,7) torus knot 7_1", "AAAAAAA"),
    'T34':        ("(3,4) torus knot 8_19", "AABAAB"),
    'GRANNY':     ("Granny knot", "AAABBB"),
    'SQUARE':     ("Square knot", "AAAbbb"),
    'HOPF':       ("Hopf link", "AA"),
    'SOLOMON':    ("Solomon's link", "AAAA"),
    'BORROMEAN':  ("Borromean rings", "AbAbAb"),
}


def torus_braid_word(p, q):
    """Letter word of the (p, q) torus link: (s1 s2 ... s_{p-1})^q."""
    return "".join(chr(ord('A') + i) for i in range(p - 1)) * q


def make_braid(text):
    """Parse a braid word into a ``seifert.Braid``.

    Accepts letter notation directly; signed-integer notation (``1 -2 1 -2``) is
    translated: ``|v|`` selects the adjacent strand pair, the sign the handedness
    (+ = uppercase/right-hand, - = lowercase/left-hand).
    """
    if sv is None:
        raise RuntimeError("the seifert package failed to import (NumPy required)")
    text = (text or "").strip()
    if not text:
        raise ValueError("empty braid word")
    if any(c.isalpha() for c in text):
        return sv.Braid(text)
    letters = []
    for tok in text.replace(',', ' ').split():
        v = int(tok)
        if v == 0:
            raise ValueError("0 is not a braid generator")
        letter = chr(ord('A') + abs(v) - 1)
        letters.append(letter if v > 0 else letter.lower())
    return sv.Braid("".join(letters))


def build_surface(braid, surface_type='SEIFERT', params=None,
                  relax_steps=100, levels=2, fair_steps=10,
                  fair_strength=2.0, rim_steps=0):
    """Build and finish a spanning surface, returning an oriented ``seifert.Mesh``.

    ``surface_type``: SEIFERT (oriented, the classical Seifert surface),
    TURNBACK (the opposite resolution, generally one-sided), or MIN_CROSSCAP
    (the non-orientable state surface of least crosscap number).  The result is
    coherently rewound so smooth shading has no dark seams at the disk/band
    joins.
    """
    if sv is None:
        raise RuntimeError("the seifert package failed to import (NumPy required)")
    if surface_type == 'TURNBACK':
        mesh = sv.state_surface(braid, sv.turnback_state(braid), params)
    elif surface_type == 'MIN_CROSSCAP':
        mesh = sv.state_surface(braid, sv.minimal_crosscap_state(braid), params)
    else:
        mesh = sv.seifert_surface(braid, params)
    mesh = sv.finish(mesh, relax_steps=relax_steps, rim_steps=rim_steps,
                     levels=levels, fair_steps=fair_steps,
                     fair_strength=fair_strength)
    return mesh.oriented()


def mesh_to_pydata(mesh, scale=1.0):
    """(verts, faces): centre on the origin, fit to a 2-unit cube, apply scale."""
    import numpy as np
    P = np.asarray(mesh.vertices, dtype=float)
    if len(P):
        lo, hi = P.min(axis=0), P.max(axis=0)
        extent = float((hi - lo).max())
        P = (P - 0.5 * (lo + hi)) * ((2.0 / extent if extent > 1e-12 else 1.0)
                                     * scale)
    verts = [tuple(map(float, p)) for p in P]
    faces = [tuple(int(i) for i in f) for f in mesh.faces]
    return verts, faces


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _boundary_loops_from_faces(faces):
        """Ordered boundary vertex loops from a polygon face list."""
        count = {}
        for vs in faces:
            for a in range(len(vs)):
                e = tuple(sorted((vs[a], vs[(a + 1) % len(vs)])))
                count[e] = count.get(e, 0) + 1
        adj = {}
        for (a, b), c in count.items():
            if c == 1:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        loops = []
        unvisited = set(adj.keys())
        while unvisited:
            start = unvisited.pop()
            loop = [start]
            prev, cur = None, start
            while True:
                nxts = [v for v in adj[cur] if v != prev and v in unvisited] \
                    or [v for v in adj[cur] if v != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
                if cur == start:
                    break
                loop.append(cur)
                unvisited.discard(cur)
            if len(loop) > 2:
                loops.append(loop)
        return loops

    def _add_knot_curve(context, obj, verts, faces, radius):
        """Create a bevelled POLY curve along each boundary loop (the link)."""
        loops = _boundary_loops_from_faces(faces)
        if not loops:
            return None
        cu = bpy.data.curves.new("Knot", 'CURVE')
        cu.dimensions = '3D'
        cu.bevel_depth = radius
        cu.use_fill_caps = True
        for loop in loops:
            sp = cu.splines.new('POLY')
            sp.points.add(len(loop) - 1)
            flat = []
            for vi in loop:
                x, y, z = verts[vi]
                flat.extend((x, y, z, 1.0))
            sp.points.foreach_set('co', flat)
            sp.use_cyclic_u = True
        knot = bpy.data.objects.new("Knot", cu)
        context.collection.objects.link(knot)
        knot.parent = obj
        knot.matrix_parent_inverse.identity()
        knot.location = (0, 0, 0)
        return knot

    def _surface_params(op):
        return sv.SurfaceParams(
            disk_spacing=op.disk_spacing, band_width=op.band_width,
            samples_per_sector=op.samples_per_sector,
            band_samples=op.band_samples, radial_rings=op.radial_rings,
            bulge=op.bulge)

    class MESH_OT_seifert_add(bpy.types.Operator):
        """Build the Seifert (or state) surface of a knot/link braid word"""
        bl_idname = "mesh.seifert_surface_add"
        bl_label = "Seifert Surface"
        bl_options = {'REGISTER', 'UNDO'}

        def _preset_chosen(self, context):
            if self.preset == 'TORUS':
                self.braid = torus_braid_word(self.torus_p, self.torus_q)
            elif self.preset != 'CUSTOM':
                self.braid = PRESETS[self.preset][1]

        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Custom braid", "Use the braid word below")] +
                  [(k, v[0], f"braid {v[1]}") for k, v in PRESETS.items()] +
                  [('TORUS', "Torus knot (p,q)", "Torus-knot braid")],
            default='TREFOIL', update=_preset_chosen)
        braid: StringProperty(
            name="Braid Word", default="AAA",
            description="Letters: A = right-hand crossing of strands 1-2, "
                        "a = left-hand; B/b = strands 2-3, ...; 'A3' repeats. "
                        "Or signed integers '1 -2 1 -2'")
        torus_p: IntProperty(name="Torus p", default=3, min=2, max=8,
                             update=_preset_chosen)
        torus_q: IntProperty(name="Torus q", default=4, min=2, max=12,
                             update=_preset_chosen)
        surface_type: EnumProperty(
            name="Surface",
            items=[
                ('SEIFERT', "Seifert (oriented)",
                 "Seifert's algorithm: the orientable spanning surface"),
                ('TURNBACK', "Turnback state",
                 "The opposite Kauffman resolution -- usually a one-sided "
                 "spanning surface of the same link"),
                ('MIN_CROSSCAP', "Min-crosscap state",
                 "The non-orientable state surface of least crosscap number "
                 "(searches all 2**c states -- keep the crossing count small)"),
            ],
            default='SEIFERT')
        # -- finishing pipeline --
        relax_steps: IntProperty(
            name="Relax Steps", default=100, min=0, max=1000,
            description="Particle-model membrane relaxation -- neighbour "
                        "attraction and pairwise repulsion round the knot out")
        levels: IntProperty(
            name="Refine Levels", default=2, min=0, max=3,
            description="Catmull-Clark subdivision levels (1 interactive, "
                        "2 export; each level roughly quadruples the mesh)")
        fair_steps: IntProperty(
            name="Fair Steps", default=10, min=0, max=60,
            description="Implicit mean-curvature fairing with the knot rim "
                        "pinned -- drives the surface to a discrete minimal "
                        "surface")
        fair_strength: FloatProperty(
            name="Fair Strength", default=2.0, min=0.1, max=20.0,
            description="Implicit-fairing time step per fair iteration")
        rim_steps: IntProperty(
            name="Rim Smoothing", default=0, min=0, max=200,
            description="Extra fairing passes along the knot boundary curve")
        # -- geometry knobs (units of the disk radius) --
        disk_spacing: FloatProperty(
            name="Disk Spacing", default=2.77, min=0.5, max=6.0,
            description="Centre-to-centre distance between stacked disks")
        band_width: FloatProperty(
            name="Band Width", default=0.85, min=0.1, max=1.6,
            description="Band width as a fraction of the disk radius")
        bulge: FloatProperty(
            name="Band Bulge", default=0.85, min=0.1, max=2.0,
            description="How far a band bows out from the stack")
        samples_per_sector: IntProperty(
            name="Rim Samples/Foot", default=12, min=4, max=48,
            description="Rim samples per band attachment (must be even)")
        band_samples: IntProperty(
            name="Band Samples", default=18, min=6, max=64,
            description="Samples swept along each band")
        radial_rings: IntProperty(
            name="Disk Rings", default=2, min=1, max=8,
            description="Annular rings outside each disk's square core")
        # -- output --
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Size multiplier; 1 fits the surface in a 2-unit cube")
        add_knot_curve: BoolProperty(
            name="Add Knot Curve", default=True,
            description="Also create a bevelled curve along the knot/link "
                        "(the surface boundary)")
        knot_radius: FloatProperty(name="Knot Tube Radius", default=0.02,
                                   min=0.0, max=0.5)

        def execute(self, context):
            if sv is None:
                self.report({'ERROR'}, "the seifert package failed to import "
                                       "(NumPy required)")
                return {'CANCELLED'}
            if self.preset == 'TORUS':
                word = torus_braid_word(self.torus_p, self.torus_q)
            elif self.preset != 'CUSTOM':
                word = PRESETS[self.preset][1]
            else:
                word = self.braid
            try:
                braid = make_braid(word)
            except (ValueError, RuntimeError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if self.samples_per_sector % 2:
                self.report({'WARNING'}, "Rim Samples/Foot must be even")
                return {'CANCELLED'}
            try:
                mesh = build_surface(
                    braid, surface_type=self.surface_type,
                    params=_surface_params(self),
                    relax_steps=self.relax_steps, levels=self.levels,
                    fair_steps=self.fair_steps,
                    fair_strength=self.fair_strength, rim_steps=self.rim_steps)
            except Exception as e:
                self.report({'ERROR'}, f"build failed: {e}")
                return {'CANCELLED'}

            info = mesh.info()
            verts, faces = mesh_to_pydata(mesh, scale=self.scale)

            me = bpy.data.meshes.new("SeifertSurface")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("SeifertSurface", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            genus = info.genus
            obj["braid"] = braid.word
            obj["surface_type"] = self.surface_type
            obj["strands"] = braid.strands
            obj["crossings"] = braid.n_crossings
            obj["boundary_components"] = info.n_boundaries
            obj["euler_characteristic"] = info.euler_characteristic
            obj["orientable"] = info.orientable
            obj["genus"] = -1 if genus is None else genus
            obj["crosscap_number"] = (info.euler_genus
                                      if not info.orientable else -1)

            if self.add_knot_curve:
                knot = _add_knot_curve(context, obj, verts, faces,
                                       self.knot_radius)
                if knot is not None:
                    knot.location = obj.location

            if info.orientable:
                shape = f"orientable, genus {genus}"
            else:
                shape = f"non-orientable, crosscap {info.euler_genus}"
            self.report(
                {'INFO'},
                f"{braid.word}: strands={braid.strands} "
                f"crossings={braid.n_crossings} components={info.n_boundaries} "
                f"chi={info.euler_characteristic} ({shape})")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'TORUS':
                lay.prop(self, 'torus_p')
                lay.prop(self, 'torus_q')
            else:
                lay.prop(self, 'braid')
            lay.prop(self, 'surface_type')
            box = lay.box()
            box.label(text="Finishing")
            for k in ('relax_steps', 'levels', 'fair_steps', 'fair_strength',
                      'rim_steps'):
                box.prop(self, k)
            box = lay.box()
            box.label(text="Geometry")
            for k in ('disk_spacing', 'band_width', 'bulge',
                      'samples_per_sector', 'band_samples', 'radial_rings'):
                box.prop(self, k)
            box = lay.box()
            box.label(text="Output")
            for k in ('scale', 'add_knot_curve', 'knot_radius'):
                box.prop(self, k)

    class MESH_OT_seifert_minimize(bpy.types.Operator):
        """Smooth the active Seifert surface further: implicit mean-curvature
        fairing (knot rim pinned -> discrete minimal surface) and/or an extra
        pass of particle-model membrane relaxation.  Click again to continue"""
        bl_idname = "mesh.seifert_minimize"
        bl_label = "Minimize Surface"
        bl_options = {'REGISTER', 'UNDO'}

        method: EnumProperty(
            name="Method",
            items=[
                ('FAIR', "Fair (minimal surface)",
                 "Implicit mean-curvature fairing with the knot boundary "
                 "pinned -- flows toward a minimal surface spanning the knot"),
                ('RELAX', "Relax (membrane)",
                 "Particle-model membrane relaxation: the whole surface and "
                 "its knot boundary evolve together"),
                ('BIHARMONIC', "Fair (thin plate, no shrink)",
                 "Botsch-Kobbelt bi-harmonic fairing: one equilibrium "
                 "solve with the rim and its first ring locked -- smooths "
                 "without the shrink toward the rim that mean-curvature "
                 "flow has, and keeps the surface tangent-continuous at "
                 "the rim"),
                ('CMCF', "Fair (conformalized MCF)",
                 "Mean-curvature flow with the Laplacian frozen at the "
                 "starting surface (Kazhdan-Solomon-Ben-Chen): robust "
                 "against the neck-pinching and weight blowup of long "
                 "plain-MCF runs; converges to the harmonic span of the "
                 "rim in the starting metric"),
            ],
            default='FAIR')
        fair_steps: IntProperty(name="Fair Steps", default=10, min=1, max=60)
        fair_strength: FloatProperty(name="Fair Strength", default=2.0,
                                     min=0.1, max=20.0)
        relax_steps: IntProperty(name="Relax Steps", default=100, min=1, max=1000)
        pin_boundary: BoolProperty(
            name="Pin Boundary", default=False,
            description="RELAX method: freeze the knot, relax only the interior")

        @classmethod
        def poll(cls, context):
            o = context.object
            return (o is not None and o.type == 'MESH' and "braid" in o)

        def execute(self, context):
            if sv is None:
                self.report({'ERROR'}, "the seifert package failed to import "
                                       "(NumPy required)")
                return {'CANCELLED'}
            import numpy as np
            obj = context.object
            me = obj.data
            verts = [tuple(v.co) for v in me.vertices]
            faces = [tuple(p.vertices) for p in me.polygons]
            mesh = sv.Mesh(verts, faces)
            before = mesh.area()
            try:
                if self.method == 'RELAX':
                    mesh = sv.relax(
                        mesh, sv.RelaxParams(pin_boundary=self.pin_boundary),
                        iterations=self.relax_steps)
                elif self.method == 'BIHARMONIC':
                    mesh = sv.biharmonic_fair(mesh)
                elif self.method == 'CMCF':
                    mesh = sv.cmcf_fair(mesh, strength=self.fair_strength,
                                        iterations=self.fair_steps)
                else:
                    mesh = sv.minimal_surface(mesh, strength=self.fair_strength,
                                              iterations=self.fair_steps)
            except Exception as e:
                self.report({'ERROR'}, f"minimize failed: {e}")
                return {'CANCELLED'}
            after = mesh.area()

            flat = np.asarray(mesh.vertices, dtype=float).ravel().tolist()
            me.vertices.foreach_set('co', flat)
            me.update()
            out = mesh.vertices
            for ch in obj.children:
                if ch.type == 'CURVE':
                    loops = _boundary_loops_from_faces(faces)
                    if len(loops) == len(ch.data.splines):
                        for sp, loop in zip(ch.data.splines, loops):
                            if len(sp.points) != len(loop):
                                continue
                            fl = []
                            for vi in loop:
                                x, y, z = out[vi]
                                fl.extend((float(x), float(y), float(z), 1.0))
                            sp.points.foreach_set('co', fl)
                    ch.data.update_tag()
            self.report({'INFO'},
                        f"area {before:.3f} -> {after:.3f} "
                        f"({100 * (1 - after / max(before, 1e-9)):.1f}% less)")
            return {'FINISHED'}

    # The sidebar panel that used to sit here is gone.  It offered an
    # Add button (Add > Mesh has one), a read-out of the braid's
    # invariants (they are custom properties on the object, so Object
    # Properties > Custom Properties shows them), and the minimise
    # operator, which stays registered and reachable from search.

    def _menu_func(self, context):
        self.layout.operator("mesh.seifert_surface_add", icon='MOD_SIMPLIFY')

    _classes = (MESH_OT_seifert_add, MESH_OT_seifert_minimize)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def _selftest():
    # Combinatorial + topology self-check (needs NumPy + the seifert package).
    bad = []
    for name, (label, word) in PRESETS.items():
        braid = make_braid(word)
        mesh = build_surface(braid, relax_steps=10, levels=1, fair_steps=2)
        info = mesh.info()
        ok = info.euler_characteristic == braid.euler_characteristic
        print(f"{label:22s} braid={braid.word:8s} strands={braid.strands} "
              f"crossings={braid.n_crossings} chi={info.euler_characteristic} "
              f"b={info.n_boundaries} genus={info.genus} "
              f"{'OK' if ok else 'CHI MISMATCH'}")
        if not ok:
            bad.append(name)
    assert not bad
