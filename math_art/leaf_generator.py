
# Leaf Generator -- simple blades and compound leaves as solid geometry.
#
# The blade is GROWN rather than drawn: see `lsystem/leaves.py` for the
# marginal-growth model and for exactly what is and is not claimed to
# reproduce ABOP Tables 5.1 and 5.2.  This file is the Blender front end.
#
# The output is a triangulated, closed, bilaterally symmetric blade, fan
# triangulated about the midrib rather than about the centroid: a leaf
# is long and narrow, and a centroid fan on a narrow shape produces
# slivers at both ends.  Fanning along the midrib keeps every triangle
# well conditioned, which matters as soon as a Solidify goes on top.
#
# References: ABOP sections 5.2 and 5.3; Runions et al., SIGGRAPH 2005
# (venation); Prusinkiewicz, Hammel, Hanan and Mech, SIGGRAPH 1988.

bl_info = {
    "name": "Leaf",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Plants > Leaf",
    "description": "Simple and compound leaves as solidifiable blades",
    "category": "Add Mesh",
}

import numpy as np

try:
    from .lsystem import leaves as lv
    from .lsystem import venation as vn
except ImportError:                       # headless / direct execution
    from lsystem import leaves as lv
    from lsystem import venation as vn


#: Re-exported so the operator and the engine cannot drift apart.
blade_mesh = lv.blade_mesh


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _mesh(verts, faces, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate()
        me.update()
        return bpy.data.objects.new(name, me)

    class MESH_OT_leaf_add(bpy.types.Operator):
        """Add a leaf: a blade grown by marginal growth, simple or
        compound, as a solidifiable mesh"""
        bl_idname = "mesh.leaf_add"
        bl_label = "Leaf"
        bl_options = {'REGISTER', 'UNDO', 'PRESET'}

        mode: EnumProperty(
            name="Mode",
            items=[('SIMPLE', "Simple", "One blade"),
                   ('COMPOUND', "Compound", "A rachis bearing leaflets, "
                                "graded by the apical delay and "
                                "elongation rate")],
            default='SIMPLE',
            description="Build a single blade, or a compound leaf of "
                        "leaflets borne on a rachis")
        shape: EnumProperty(
            name="Shape",
            items=[(k, v["label"], v["label"])
                   for k, v in sorted(lv.LEAF_SHAPES.items())],
            default='OVATE',
            description="Outline the blade grows toward, from the "
                        "leaf-shape library")
        steps: IntProperty(
            name="Growth Steps", default=10, min=2, max=14,
            description="Plastochrons of marginal growth")
        samples: IntProperty(
            name="Outline Points", default=96, min=12, max=1024,
            description="Number of points sampled around the blade "
                        "outline")

        row: EnumProperty(
            name="Table 5.2 Row",
            items=[(str(i), r["label"], f"apical delay {r['delay']}, "
                    f"elongation rate {r['rate']}")
                   for i, r in enumerate(lv.COMPOUND)],
            default='2',
            description="Compound leaf: preset apical delay and "
                        "elongation rate from ABOP Table 5.2")
        use_row: BoolProperty(
            name="Use Table Row", default=True,
            description="Take the apical delay and elongation rate from "
                        "Table 5.2 rather than setting them by hand")
        delay: IntProperty(
            name="Apical Delay", default=2, min=0, max=12,
            description="Plastochrons between successive leaflet pairs")
        rate: FloatProperty(
            name="Elongation Rate", default=1.36, min=1.0, max=3.0,
            description="Internode elongation per plastochron; higher "
                        "grades the leaflets more steeply")
        plastochrons: IntProperty(name="Plastochrons", default=14,
                                  min=1, max=40,
                                  description="Compound leaf: growth steps "
                                  "run; more steps add more leaflet pairs")
        leaflet_angle: FloatProperty(name="Leaflet Angle", default=42.0,
                                     min=0.0, max=90.0,
                                     description="Angle each leaflet makes "
                                     "with the rachis, in degrees")
        leaflet_scale: FloatProperty(name="Leaflet Size", default=0.3,
                                     min=0.02, max=1.0,
                                     description="Size of each leaflet "
                                     "relative to the whole leaf")

        veins: BoolProperty(
            name="Venation", default=False,
            description="Grow a vein network into the blade by space "
                        "colonization (Runions et al., 2005)")
        vein_mode: EnumProperty(
            name="Venation Type",
            items=[('OPEN', "Open (dichotomous)", "Each source attracts "
                            "only the nearest vein, so veins never "
                            "rejoin -- a tree. Ginkgo, conifers, most "
                            "monocots"),
                   ('CLOSED', "Closed (reticulate)", "Each source "
                              "attracts every vein in its relative "
                              "neighbourhood, so veins meet and close "
                              "loops -- the areoles of a dicot leaf")],
            default='CLOSED',
            description="Grow the veins as a branching tree (open) or a "
                        "looped network (closed)")
        vein_sources: IntProperty(name="Auxin Sources", default=450,
                                  min=20, max=4000,
                                  description="Number of auxin sources "
                                  "scattered in the blade to draw veins "
                                  "toward")
        vein_step: FloatProperty(name="Vein Step", default=0.025,
                                 min=0.004, max=0.2,
                                 description="Distance a vein grows toward "
                                 "its attracting sources each step")
        vein_kill: FloatProperty(
            name="Kill Radius", default=0.05, min=0.005, max=0.5,
            description="A source is consumed once a vein reaches this "
                        "close. It is the control that sets how fine "
                        "the network is, because it is what stops veins "
                        "crowding")
        vein_influence: FloatProperty(name="Influence Radius", default=0.35,
                                      min=0.02, max=2.0,
                                      description="How far a source can "
                                      "reach to attract a growing vein tip")
        vein_radius: FloatProperty(name="Vein Radius", default=0.006,
                                   min=0.0005, max=0.1,
                                   description="Bevel radius of the vein "
                                   "tubes")
        vein_seed: IntProperty(name="Seed", default=0, min=0, max=10000,
                               description="Random seed for scattering the "
                               "auxin sources")

        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size of the leaf")

        def execute(self, context):
            name = "Leaf"
            try:
                if self.mode == 'SIMPLE':
                    P = lv.simple_leaf(self.shape, steps=self.steps,
                                       samples=self.samples)
                    verts, faces = blade_mesh(P)
                    name = lv.LEAF_SHAPES[self.shape]["label"]
                else:
                    d, r = self.delay, self.rate
                    if self.use_row:
                        row = lv.COMPOUND[int(self.row)]
                        d, r = row["delay"], row["rate"]
                        name = f"Compound Leaf {row['label']}"
                    verts, faces = lv.compound_geometry(
                        delay=d, rate=r, steps=self.plastochrons,
                        angle=self.leaflet_angle,
                        leaflet_scale=self.leaflet_scale,
                        shape=self.shape,
                        samples=max(self.samples // 2, 12))
            except (KeyError, ValueError, IndexError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

            if not faces:
                self.report({'ERROR'}, "produced no geometry")
                return {'CANCELLED'}

            V = np.array(verts, dtype=float)
            lo, hi = V.min(axis=0), V.max(axis=0)
            ext = float((hi - lo).max())
            self._fit = ((lo + hi) / 2.0,
                         (2.0 * self.scale / ext) if ext > 1e-12 else 1.0)
            if ext > 1e-12:
                V = (V - (lo + hi) / 2.0) * (2.0 * self.scale / ext)
            verts = [tuple(p) for p in V]
            objs = [_mesh(verts, faces, name)]
            if self.veins and self.mode == 'SIMPLE':
                vob = self._veins(name)
                if vob is not None:
                    objs.append(vob)
            for o in context.selected_objects:
                o.select_set(False)
            for o in objs:
                context.collection.objects.link(o)
                o.select_set(True)
            ob = objs[0]
            context.view_layer.objects.active = ob
            self.report({'INFO'},
                        f"{name}: {len(V)} vertices, {len(faces)} faces")
            return {'FINISHED'}

        def _veins(self, name):
            """A vein network grown into the blade, as a bevelled curve.

            The veins are a separate object: they want their own
            material, and a blade with its venation welded in cannot be
            solidified independently of it.
            """
            outline = lv.simple_leaf(self.shape, steps=self.steps,
                                     samples=self.samples)
            nodes, edges, parents = vn.grow(
                outline, n_sources=self.vein_sources,
                mode=self.vein_mode, step=self.vein_step,
                kill=self.vein_kill, influence=self.vein_influence,
                seed=self.vein_seed)
            if len(nodes) < 2 or not edges:
                return None
            w = vn.vein_widths(nodes, edges, parents)
            w = w / max(float(w.max()), 1e-9)
            cen, f = self._fit
            cu = bpy.data.curves.new(name + " Veins", 'CURVE')
            cu.dimensions = '3D'
            for a, b in edges:
                sp = cu.splines.new('POLY')
                sp.points.add(1)
                for k, i in enumerate((a, b)):
                    x = (float(nodes[i][0]) - float(cen[0])) * f
                    y = (float(nodes[i][1]) - float(cen[1])) * f
                    sp.points[k].co = (x, y, 0.0, 1.0)
                    sp.points[k].radius = float(w[i])
            cu.bevel_depth = float(self.vein_radius)
            cu.bevel_resolution = 1
            return bpy.data.objects.new(name + " Veins", cu)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "mode")
            lay.prop(self, "shape")
            lay.prop(self, "steps")
            lay.prop(self, "samples")
            if self.mode == 'COMPOUND':
                lay.separator()
                lay.prop(self, "use_row")
                if self.use_row:
                    lay.prop(self, "row")
                else:
                    lay.prop(self, "delay")
                    lay.prop(self, "rate")
                for k in ("plastochrons", "leaflet_angle", "leaflet_scale"):
                    lay.prop(self, k)
            if self.mode == 'SIMPLE':
                lay.separator()
                lay.prop(self, "veins")
                if self.veins:
                    for k in ("vein_mode", "vein_sources", "vein_step",
                              "vein_kill", "vein_influence",
                              "vein_radius", "vein_seed"):
                        lay.prop(self, k)
            lay.separator()
            lay.prop(self, "scale")

    def register():
        bpy.utils.register_class(MESH_OT_leaf_add)

    def unregister():
        bpy.utils.unregister_class(MESH_OT_leaf_add)


def _selftest():
    # Every shape must triangulate into a real, non-degenerate blade.
    for key in sorted(lv.LEAF_SHAPES):
        P = lv.simple_leaf(key, steps=10, samples=96)
        verts, faces = blade_mesh(P)
        assert len(verts) > 12, key
        assert len(faces) > 6, key
        V = np.array(verts)
        assert np.all(np.isfinite(V)), key
        # every face must have positive area -- a sliver would survive
        # triangulation and then fail to solidify
        for q in faces:
            a, b, c = V[q[0]], V[q[1]], V[q[2]]
            ar = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
            assert ar >= 0.0, key
        # the blade is flat and spans both sides of the midrib
        assert abs(float(V[:, 2].max() - V[:, 2].min())) < 1e-12, key
        assert float(V[:, 0].min()) < 0.0 < float(V[:, 0].max()), key

    # Compound leaves must build for every Table 5.2 row and place more
    # leaflets when the delay is shorter.
    counts = []
    for row in lv.COMPOUND:
        _r, leaflets = lv.compound_leaf(delay=row["delay"],
                                        rate=row["rate"], steps=20)
        counts.append(len(leaflets))
        verts, faces = lv.compound_geometry(
            delay=row["delay"], rate=row["rate"], steps=20, samples=24)
        assert faces, row["label"]
        V = np.array(verts)
        assert np.all(np.isfinite(V)), row["label"]
        # leaflets on both sides of the rachis, spread along it
        assert float(V[:, 0].min()) < 0.0 < float(V[:, 0].max()), row["label"]
        assert float(V[:, 1].max() - V[:, 1].min()) > 0.3, row["label"]
    # delay rises down the table, so leaflet count must fall
    assert counts[0] > counts[-1], counts

    print(f"leaf_generator: OK -- {len(lv.LEAF_SHAPES)} blades and "
          f"{len(lv.COMPOUND)} compound rows triangulate flat, symmetric "
          f"and finite; leaflet counts {counts}")
