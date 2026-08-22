
# Map L-system Generator -- tissue grown by division.
#
# The add-on already makes cells by PARTITIONING: voronoi_openwork,
# organic_wireframe and bubble all take a region and cut it up.  This
# makes them by DIVISION, which is a different object: the tissue has a
# developmental history, so cell sizes are graded by age, walls meet at
# angles that came from the order they were laid down, and the outline
# EMERGES rather than being the boundary you started with.  Voronoi
# looks scattered; this looks grown.  That distinction is the whole
# reason it is worth having alongside them.
#
# One dial does the shape work: the periclinal/anticlinal ratio P/A.
# See `lsystem/maps.py` for the formalism, the two constraints it obeys,
# and the mass-spring pressure solver.
#
# References: as `lsystem/maps.py` -- ABOP chapters 6 and 7; Rozenberg
# and Salomaa (1980) ch. 27; Fracchia, Prusinkiewicz and de Boer (1990);
# Prusinkiewicz and Runions, New Phytologist 193, 2012, section IV.1;
# Errera (1886).

bl_info = {
    "name": "Map L-system",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Plants > Cellular Layer",
    "description": "Cellular tissue by developmental division, with "
                   "turgor-pressure relaxation",
    "category": "Add Mesh",
}

import numpy as np

try:
    from .lsystem import maps
except ImportError:                       # headless / direct execution
    from lsystem import maps


def layer_mesh(cm, thickness=0.0, inset=0.06):
    """Cells as faces, optionally inset so the walls read as walls.

    Insetting each cell toward its own centroid is what turns a tiling
    into an openwork screen: the gap between neighbouring cells becomes
    the wall, with a width you can actually print.
    """
    verts, faces, ages = [], [], []
    for ci, ring in enumerate(cm.cells):
        P = cm.verts[ring]
        c = P.mean(axis=0)
        Q = c + (P - c) * (1.0 - float(inset))
        base = len(verts)
        for p in Q:
            verts.append((float(p[0]), float(p[1]), 0.0))
        faces.append(tuple(range(base, base + len(Q))))
        ages.append(float(cm.ages[ci]))
        if thickness:
            b2 = len(verts)
            for p in Q:
                verts.append((float(p[0]), float(p[1]), float(thickness)))
            n = len(Q)
            faces.append(tuple(range(b2, b2 + n))[::-1])
            for i in range(n):
                faces.append((base + i, base + (i + 1) % n,
                              b2 + (i + 1) % n, b2 + i))
            ages.extend([float(cm.ages[ci])] * (n + 1))
    return verts, faces, ages


def ball_mesh(P, F, inset=0.06):
    verts, faces = [], []
    for ring in F:
        Q = P[list(ring)]
        c = Q.mean(axis=0)
        Q = c + (Q - c) * (1.0 - float(inset))
        base = len(verts)
        for p in Q:
            verts.append(tuple(float(x) for x in p))
        faces.append(tuple(range(base, base + len(Q))))
    return verts, faces


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_map_lsystem_add(bpy.types.Operator):
        """Add cellular tissue grown by developmental division, relaxed
        under turgor pressure"""
        bl_idname = "mesh.map_lsystem_add"
        bl_label = "Cellular Layer"
        bl_options = {'REGISTER', 'UNDO', 'PRESET'}

        mode: EnumProperty(
            name="Mode",
            description="Grow a flat sheet or a closed ball of cells",
            items=[('LAYER', "Cellular Layer", "A flat sheet -- the "
                             "fern-gametophyte case"),
                   ('BALL', "Cell Ball", "The same division on a closed "
                            "surface: a Patella-style blastula")],
            default='LAYER')
        steps: IntProperty(
            name="Divisions", default=5, min=1, max=9,
            description="Map L-systems divide by ELAPSED TIME, not by "
                        "cell size, so the dial is a step count. Every "
                        "cell divides once per step, so cell count "
                        "doubles")
        pa: FloatProperty(
            name="Periclinal / Anticlinal", default=1.0, min=0.05,
            max=8.0,
            description="The one shape dial. A periclinal wall lies "
                        "parallel to the surface and thickens the "
                        "tissue; an anticlinal one is perpendicular and "
                        "extends it. Below about 1.25 the sheet stays "
                        "convex, above 2 it dishes in")
        jitter: FloatProperty(name="Plane Jitter", default=0.25,
                              min=0.0, max=1.2,
                              description="Random tilt applied to each "
                                          "division plane")
        seed: IntProperty(name="Seed", default=0, min=0, max=10000,
                          description="Random seed for the division "
                                      "planes")

        pressure: FloatProperty(
            name="Turgor", default=0.02, min=0.0, max=0.3,
            description="Each cell pushes out with p = nRT/(A h), so a "
                        "small cell pushes harder. That single term is "
                        "what equalises cell size -- no rule says cells "
                        "should be similar")
        stiffness: FloatProperty(name="Wall Stiffness", default=0.6,
                                 min=0.05, max=2.0,
                                 description="How strongly the cell "
                                             "walls resist stretching "
                                             "during relaxation")
        relax_steps: IntProperty(name="Relax Steps", default=60,
                                 min=0, max=600,
                                 description="Pressure-relaxation "
                                             "iterations after each "
                                             "division")

        inset: FloatProperty(
            name="Wall Inset", default=0.14, min=0.0, max=0.45,
            description="Shrink each cell toward its centroid so the "
                        "gaps become walls -- what turns a tiling into "
                        "an openwork screen")
        thickness: FloatProperty(name="Thickness", default=0.0,
                                 min=0.0, max=1.0,
                                 description="Extrude the sheet to this "
                                             "thickness (0 keeps it flat)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the result")

        def execute(self, context):
            try:
                if self.mode == 'BALL':
                    P, F = maps.cell_ball(steps=min(self.steps, 6),
                                          seed=self.seed)
                    verts, faces = ball_mesh(P, F, self.inset)
                    ages = None
                    name = "Cell Ball"
                else:
                    cm = maps.square()
                    for k in range(int(self.steps)):
                        cm = maps.divide(cm, pa=self.pa,
                                         seed=self.seed + k,
                                         jitter=self.jitter)
                        if self.relax_steps:
                            cm = maps.relax(cm, steps=self.relax_steps,
                                            pressure=self.pressure,
                                            stiffness=self.stiffness)
                    verts, faces, ages = layer_mesh(cm, self.thickness,
                                                    self.inset)
                    name = "Cellular Layer"
            except (ValueError, KeyError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            if not faces:
                self.report({'ERROR'}, "produced no cells")
                return {'CANCELLED'}

            V = np.array(verts, dtype=float)
            lo, hi = V.min(axis=0), V.max(axis=0)
            ext = float((hi - lo).max())
            if ext > 1e-12:
                V = (V - (lo + hi) / 2.0) * (2.0 * self.scale / ext)

            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(p) for p in V], [], faces)
            me.validate()
            me.update()
            if ages:
                try:
                    a = me.attributes.new(name="cell_age", type='FLOAT',
                                          domain='FACE')
                    a.data.foreach_set("value", ages[:len(me.polygons)])
                except Exception:
                    pass
            ob = bpy.data.objects.new(name, me)
            context.collection.objects.link(ob)
            for o in context.selected_objects:
                o.select_set(False)
            ob.select_set(True)
            context.view_layer.objects.active = ob
            self.report({'INFO'},
                        f"{name}: {len(faces)} cells from {self.steps} "
                        f"divisions")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "mode")
            lay.prop(self, "steps")
            box = lay.box()
            box.label(text=f"{2 ** int(self.steps)} cells", icon='INFO')
            if self.mode == 'LAYER':
                lay.prop(self, "pa")
                box = lay.box()
                box.label(text=("convex sheet" if self.pa < 1.25
                                else ("concave / dished" if self.pa > 2.0
                                      else "intermediate")),
                          icon='MESH_GRID')
                lay.prop(self, "jitter")
            lay.prop(self, "seed")
            if self.mode == 'LAYER':
                lay.separator()
                for k in ("pressure", "stiffness", "relax_steps"):
                    lay.prop(self, k)
            lay.separator()
            lay.prop(self, "inset")
            if self.mode == 'LAYER':
                lay.prop(self, "thickness")
            lay.prop(self, "scale")

    def register():
        bpy.utils.register_class(MESH_OT_map_lsystem_add)

    def unregister():
        bpy.utils.unregister_class(MESH_OT_map_lsystem_add)


def _selftest():
    # The layer must reach exactly 2**n cells and every cell must
    # triangulate into a real face.
    for steps in (2, 4, 6):
        cm = maps.square()
        for k in range(steps):
            cm = maps.divide(cm, pa=1.0, seed=k, jitter=0.0)
        assert len(cm.cells) == 2 ** steps, (steps, len(cm.cells))
        verts, faces, ages = layer_mesh(cm, inset=0.06)
        assert len(faces) == 2 ** steps
        assert len(ages) == len(faces)
        V = np.array(verts)
        assert np.all(np.isfinite(V))
        # the inset must actually separate neighbours: total face area
        # has to fall below the tissue area
        tot = float(cm.areas().sum())
        got = 0.0
        for f in faces:
            P = V[list(f)][:, :2]
            x, y = P[:, 0], P[:, 1]
            got += 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                                   - np.dot(np.roll(x, -1), y)))
        assert got < tot, (got, tot)
        assert got > 0.5 * tot, (got, tot)

    # zero inset gives back the full tissue area
    cm = maps.square()
    for k in range(3):
        cm = maps.divide(cm, pa=1.0, seed=k, jitter=0.0)
    v0, f0, _a = layer_mesh(cm, inset=0.0)
    V0 = np.array(v0)
    got = 0.0
    for f in f0:
        P = V0[list(f)][:, :2]
        x, y = P[:, 0], P[:, 1]
        got += 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                               - np.dot(np.roll(x, -1), y)))
    assert abs(got - float(cm.areas().sum())) < 1e-9, got

    # thickness closes the sheet into a solid: extra faces per cell
    thin = len(layer_mesh(cm, thickness=0.0)[1])
    thick = len(layer_mesh(cm, thickness=0.1)[1])
    assert thick > 2 * thin, (thin, thick)

    # the ball is a closed shell of cells
    P, F = maps.cell_ball(steps=3, seed=0)
    bv, bf = ball_mesh(P, F, inset=0.08)
    assert len(bf) == len(F) == 8 * 2 ** 3
    B = np.array(bv)
    assert np.all(np.isfinite(B))
    r = np.linalg.norm(B, axis=1)
    assert float(r.min()) > 0.5, float(r.min())

    print(f"map_lsystem_generator: OK -- layer reaches 2**n cells, inset "
          f"opens walls without losing area, thickness solidifies, ball "
          f"is a {len(bf)}-cell shell")
