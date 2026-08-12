
# Inflorescence Generator -- the chapter-3 taxonomy as one operator.
#
# Raceme, spike, spadix, corymb, umbel, capitulum, verticillaster,
# panicle, dibotryoid, tribotryoid, monochasium, dichasium, cyme,
# thyrsus and pleiochasium are not fifteen constructions.  They are the
# two production templates of ABOP Table 3.1 with different constants --
# see `lsystem/inflorescence.py` for the statement and the mathematics.
# This file is only the Blender front end.
#
# Three things it exposes that a preset list could not:
#
#   * BLOOM ORDER as a real, animatable quantity.  Each flower carries an
#     age; the sign of D = u*n - v*m decides whether the structure
#     flowers acropetally, basipetally or divergently, and the Bloom
#     slider reveals flowers up to a threshold so the sequence can be
#     keyframed.
#   * BRANCH MAPPING, so all axes of one order share a single set of
#     gradients and a short branch matches the tip of a long one.
#   * SPECIES reconstructions built from the archetypes rather than
#     hand-placed.
#
# References: as `lsystem/inflorescence.py` -- ABOP chapter 3 and
# Table 3.1; Frijters and Lindenmayer (1976); Prusinkiewicz, Hammel,
# Hanan and Mech, SIGGRAPH 1988; Prusinkiewicz, Muendermann, Karwowski
# and Lane, SIGGRAPH 2001, sections 6 and 8.

bl_info = {
    "name": "Inflorescence",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Plants > Inflorescence",
    "description": "The chapter-3 inflorescence taxonomy from two "
                   "production templates, with bloom order",
    "category": "Add Curve",
}

import numpy as np

try:
    from .lsystem import inflorescence as inf
except ImportError:                       # headless / direct execution
    from lsystem import inflorescence as inf


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _emit(nodes, name, radius, resolution, taper_radius, flower_size,
              bloom, show_flowers):
        """Internodes as a bevelled curve; flowers as a point cloud mesh.

        Two objects rather than one, because the flowers want their own
        material and their own instancing, and because the bloom slider
        has to be able to hide them independently of the stems.
        """
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        max_order = max((q.order for q in nodes), default=0)
        for q in nodes:
            sp = cu.splines.new('POLY')
            sp.points.add(1)
            sp.points[0].co = (*q.a, 1.0)
            sp.points[1].co = (*q.b, 1.0)
            # thinner with order: the pipe model, one step per order
            r = taper_radius ** q.order
            sp.points[0].radius = r
            sp.points[1].radius = r * taper_radius
        cu.bevel_depth = float(radius)
        cu.bevel_resolution = int(resolution)
        stems = bpy.data.objects.new(name, cu)

        objs = [stems]
        if show_flowers:
            fl = [q for q in nodes if q.flower]
            seq = inf.bloom_sequence(nodes)
            pairs = [(q, s) for q, s in zip(fl, seq) if s <= bloom + 1e-9]
            if pairs:
                objs.append(_flower_mesh(pairs, name + " Flowers",
                                         flower_size))
        return objs

    #: A unit octahedron: six vertices, eight faces.  Small enough to
    #: instance three thousand times without thought, and -- unlike a
    #: bare vertex -- it is actually DRAWN.  A vertex-only mesh is
    #: invisible in Object Mode, so emitting flowers as a point cloud
    #: made the Flowers toggle and the Bloom slider look like they did
    #: nothing at all.
    _OCTA_V = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
               (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
               (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
    _OCTA_F = ((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
               (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5))

    def _flower_mesh(pairs, name, size):
        """Real geometry at each flower, carrying its bloom age.

        The bloom attribute is written per vertex so a material can fade
        a flower in over the flowering sequence rather than having it
        appear abruptly when the slider passes it.
        """
        verts, faces, bloom_a, order_a = [], [], [], []
        s = max(float(size), 1e-6)
        for q, age in pairs:
            base = len(verts)
            cx, cy, cz = float(q.b[0]), float(q.b[1]), float(q.b[2])
            for vx, vy, vz in _OCTA_V:
                verts.append((cx + vx * s, cy + vy * s, cz + vz * s))
                bloom_a.append(float(age))
                order_a.append(float(q.order))
            faces.extend(tuple(i + base for i in f) for f in _OCTA_F)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate()
        me.update()
        _attr(me, "bloom", bloom_a)
        _attr(me, "order", order_a)
        return bpy.data.objects.new(name, me)

    def _attr(me, name, values, typ='FLOAT'):
        try:
            a = me.attributes.new(name=name, type=typ, domain='POINT')
            a.data.foreach_set("value", values)
        except Exception:
            pass

    class CURVE_OT_inflorescence_add(bpy.types.Operator):
        """Add an inflorescence: the whole chapter-3 taxonomy from two
        production templates, with acropetal/basipetal bloom order"""
        bl_idname = "curve.inflorescence_add"
        bl_label = "Inflorescence"
        bl_options = {'REGISTER', 'UNDO', 'PRESET'}

        source: EnumProperty(
            name="Source",
            items=[('ARCHETYPE', "Architecture", "One of the Table 3.1 "
                                 "architectures"),
                   ('SPECIES', "Species", "A published reconstruction")],
            default='ARCHETYPE')
        archetype: EnumProperty(
            name="Architecture",
            items=[(k, v["label"], f"n={v['n']}, m={v['m']}, "
                    f"{'determinate' if v['determinate'] else 'indeterminate'}")
                   for k, v in sorted(inf.ARCHETYPES.items())],
            default='RACEME')
        species: EnumProperty(
            name="Species",
            items=[(k, v["label"], v["label"])
                   for k, v in sorted(inf.SPECIES.items())],
            default='LILAC')

        orders: IntProperty(
            name="Orders", default=3, min=1, max=6,
            description="Recursion depth; order 0 is the main axis")
        nodes: IntProperty(
            name="Nodes", default=6, min=1, max=24,
            description="Internodes on the main axis")
        angle: FloatProperty(
            name="Branch Angle", default=-1.0, min=-1.0, max=90.0,
            description="Degrees (-1 = the architecture's own)")
        taper: FloatProperty(
            name="Length Ratio", default=-1.0, min=-1.0, max=2.0,
            description="Internode length between successive orders "
                        "(-1 = the architecture's own)")
        divergence: FloatProperty(
            name="Divergence", default=137.5, min=0.0, max=360.0,
            description="Roll between successive laterals")

        u: FloatProperty(
            name="Rate u", default=1.0, min=0.0, max=10.0,
            description="Age gained per internode along an axis")
        v: FloatProperty(
            name="Rate v", default=1.0, min=0.0, max=10.0,
            description="Age gained per branching order")
        bloom: FloatProperty(
            name="Bloom", default=1.0, min=0.0, max=1.0,
            description="Reveal flowers up to this point in the "
                        "flowering sequence -- keyframe it to animate "
                        "the bloom running through the structure")
        show_flowers: BoolProperty(name="Flowers", default=True)

        branch_mapping: BoolProperty(
            name="Branch Mapping", default=True,
            description="Treat a short axis as the TOP of a long one of "
                        "the same order, so gradients align distally")

        radius: FloatProperty(name="Stem Radius", default=0.01,
                              min=0.0001, max=0.5)
        resolution: IntProperty(name="Bevel Resolution", default=2,
                                min=0, max=16)
        order_taper: FloatProperty(
            name="Order Taper", default=0.7, min=0.1, max=1.0,
            description="Stem radius ratio between successive orders")
        flower_size: FloatProperty(name="Flower Size", default=0.03,
                                   min=0.001, max=0.5)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            kw = dict(orders=self.orders, nodes=self.nodes,
                      divergence=self.divergence, u=self.u, v=self.v,
                      branch_mapping=self.branch_mapping)
            if self.angle >= 0.0:
                kw["angle"] = self.angle
            if self.taper >= 0.0:
                kw["taper"] = self.taper
            try:
                if self.source == 'SPECIES':
                    nodes, info = inf.build(species=self.species, **kw)
                else:
                    nodes, info = inf.build(self.archetype, **kw)
            except (KeyError, ValueError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            if not nodes:
                self.report({'ERROR'}, "produced no geometry")
                return {'CANCELLED'}

            inf.fit(nodes, 2.0 * self.scale)
            label = (inf.SPECIES[self.species]["label"]
                     if self.source == 'SPECIES'
                     else inf.ARCHETYPES[self.archetype]["label"])
            objs = _emit(nodes, label, self.radius, self.resolution,
                         self.order_taper, self.flower_size, self.bloom,
                         self.show_flowers)
            for o in context.selected_objects:
                o.select_set(False)
            for o in objs:
                context.collection.objects.link(o)
                o.select_set(True)
            context.view_layer.objects.active = objs[0]
            self.report(
                {'INFO'},
                f"{label}: n={info['n']} m={info['m']} "
                f"{'determinate' if info['determinate'] else 'indeterminate'}, "
                f"{info['internodes']} internodes, {info['flowers']} flowers, "
                f"D={info['delta']:+.2f} -> {info['bloom'].lower()}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "source")
            lay.prop(self, "species" if self.source == 'SPECIES'
                     else "archetype")

            key = (inf.SPECIES[self.species]["base"]
                   if self.source == 'SPECIES' else self.archetype)
            arc = inf.ARCHETYPES[key]
            box = lay.box()
            box.label(text=f"Table 3.1:  n={arc['n']}  m={arc['m']}  "
                           f"{'determinate' if arc['determinate'] else 'indeterminate'}",
                      icon='INFO')
            mode = inf.bloom_mode(self.u, self.v, arc['n'], arc['m'])
            box.label(text=f"D = {self.u * arc['n'] - self.v * arc['m']:+.2f}"
                           f"  ->  {mode.lower()}",
                      icon='SORTTIME')

            for k in ("orders", "nodes", "angle", "taper", "divergence"):
                lay.prop(self, k)
            lay.prop(self, "branch_mapping")
            lay.separator()
            for k in ("u", "v", "show_flowers"):
                lay.prop(self, k)
            if self.show_flowers:
                lay.prop(self, "bloom")
                lay.prop(self, "flower_size")
            lay.separator()
            for k in ("radius", "resolution", "order_taper", "scale"):
                lay.prop(self, k)

    def register():
        bpy.utils.register_class(CURVE_OT_inflorescence_add)

    def unregister():
        bpy.utils.unregister_class(CURVE_OT_inflorescence_add)


def _selftest():
    # The operator layer is bpy-only; what is testable headlessly is
    # that every enum row it offers actually builds.
    for key in sorted(inf.ARCHETYPES):
        nodes, info = inf.build(key, orders=3, nodes=5)
        assert nodes, key
        assert info["flowers"] > 0, key
        assert "label" in inf.ARCHETYPES[key], key
    for key in sorted(inf.SPECIES):
        nodes, info = inf.build(species=key, nodes=5)
        assert nodes, key
        assert "label" in inf.SPECIES[key], key

    # The bloom slider thresholds a 0..1 sequence, so that sequence must
    # be defined and monotone-usable for every architecture.
    for key in sorted(inf.ARCHETYPES):
        nodes, _i = inf.build(key, orders=3, nodes=5)
        seq = inf.bloom_sequence(nodes)
        assert len(seq) == sum(1 for q in nodes if q.flower), key
        assert float(seq.min()) >= 0.0 and float(seq.max()) <= 1.0, key
        # a partial bloom must reveal strictly fewer flowers than a full
        assert (seq <= 0.5).sum() < (seq <= 1.0).sum(), key

    print(f"inflorescence_generator: OK -- {len(inf.ARCHETYPES)} "
          f"architectures and {len(inf.SPECIES)} species build, bloom "
          f"sequences thresholdable")
