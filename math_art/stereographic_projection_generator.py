
# Stereographic Projection Shadow Shell for Blender
#
# After Henry Segerman's stereographic projection sculptures
# ("Visualizing Mathematics with 3D Printing", figs 3-11..3-14):
# a perforated spherical shell that, lit by a point light at its
# north pole, casts a shadow reproducing a flat planar pattern.
#
# The sphere (radius R) rests with its south pole at the origin:
# centre C = (0,0,R), north pole N = (0,0,2R). Stereographic
# projection from N sends the sphere point at polar angle th
# (measured from N) to the plane z = 0 at radius
#
#     r = 2R cot(th/2)        inversely   th = 2 atan(2R / r)
#
# so a plane pattern clipped to a disc of radius Rmax becomes a
# perforation pattern below the latitude th_min = 2 atan(2R/Rmax);
# everything above stays solid (the cap around the projection
# point, without which the shell falls apart).
#
# Watertight construction (boolean-free): the sphere is meshed as
# a lat-long quad grid (triangle fans at the poles). Each face is
# classified material/hole by mapping its centre to the plane
# (GRID, POLAR) or testing directly on the sphere (TILING,
# BEACHBALL). Kept faces are duplicated at radii R +- t/2 from C,
# and every boundary edge of the kept region is closed with a
# side-wall quad, so the solid is closed by construction. Two
# guards keep it 2-manifold: the pole fan rows are forced uniform
# (else >2 walls would meet at a pole's vertical edge), and a
# cleanup pass fills faces where material cells touch only
# diagonally (a "checkerboard" vertex would put 4 walls on one
# vertical edge). For GRID/POLAR the latitude rows are spaced
# uniformly in plane radius r, so pattern cells are evenly
# resolved in the plane where the pattern lives.
#
# References:
# - Stereographic projection is a classical construction (known to
#   Hipparchus and Ptolemy in antiquity). Shadow-casting sculpture
#   application after H. Segerman, "Visualizing Mathematics with
#   3D Printing" (2016), figs 3-11..3-14.

bl_info = {
    "name": "Stereographic Projection",
    "author": "Math Art project (after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Stereographic Projection",
    "description": "Perforated sphere whose shadow from a north-"
                   "pole light is a flat planar pattern",
    "category": "Add Mesh",
}
# The mathematics lives in the sibling `projection` engine package;
# this module is the Blender layer over it.
try:
    from .projection.stereographic import (_valid_pq, build_shell)
except ImportError:  # flat import outside the package
    from projection.stereographic import (_valid_pq, build_shell)



























try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_stereographic_add(bpy.types.Operator):
        """Add a perforated sphere whose shadow from a point light
        at its north pole is a flat planar pattern (after Henry
        Segerman's stereographic projection sculptures)"""
        bl_idname = "mesh.stereographic_add"
        bl_label = "Stereographic Projection"
        bl_options = {'REGISTER', 'UNDO'}

        pattern: EnumProperty(
            name="Pattern",
            description="Planar pattern the shell's shadow reproduces",
            items=[('GRID', "Square Grid",
                    "Shadow is a square grid of lines"),
                   ('POLAR', "Polar Grid",
                    "Shadow is concentric circles and radial "
                    "rays"),
                   ('TILING', "{p,q} Tiling",
                    "Material along the edges of a spherical "
                    "{p,q} tiling (Platonic edge graph; p = 2 "
                    "gives q meridians)"),
                   ('BEACHBALL', "Beach Ball",
                    "Alternating open and solid lune gores"),
                   ('FLOWER', "Flower Lattice",
                    "Hexagonal lattice of flowers with petal-"
                    "shaped holes; the shadow is a field of "
                    "daisies")],
            default='GRID')
        bowl: BoolProperty(
            name="Bowl", default=False,
            description="Open bowl instead of a full sphere: the "
                        "shell stops just above the pattern with "
                        "a solid rim band and an open top")
        petals: IntProperty(
            name="Petals", default=8, min=3, max=16,
            description="Petals per flower (Flower Lattice)")
        radius: FloatProperty(name="Sphere Radius", default=1.0,
                              min=0.05, max=100.0,
                              description="Radius of the sphere shell")
        thickness: FloatProperty(
            name="Shell Thickness", default=0.05, min=0.001,
            max=1.0, description="Radial wall thickness")
        width: FloatProperty(
            name="Strip Width", default=0.35, min=0.05, max=0.95,
            description="Material strip width as a fraction of "
                        "the pattern cell (grid spacing, ring "
                        "gap, tile edge, lune slot)")
        extent: FloatProperty(
            name="Pattern Extent", default=3.0, min=1.0, max=10.0,
            description="Radius of the plane pattern disc in "
                        "sphere radii; the shell is solid above "
                        "the matching latitude (the cap holding "
                        "the projection point)")
        spacing: FloatProperty(
            name="Grid Spacing", default=0.6, min=0.1, max=5.0,
            description="Grid line spacing in sphere radii")
        rings: IntProperty(name="Rings", default=4, min=1, max=24,
                           description="Number of concentric rings in the "
                                       "shadow (Polar Grid)")
        rays: IntProperty(name="Rays", default=8, min=1, max=64,
                          description="Number of radial rays in the shadow "
                                      "(Polar Grid)")
        tile_p: IntProperty(
            name="p", default=4, min=2, max=5,
            description="Tile polygon sides (2 = hosohedron)")
        tile_q: IntProperty(
            name="q", default=3, min=3, max=8,
            description="Tiles per vertex (needs 1/p + 1/q > "
                        "1/2; clamped if not)")
        gores: IntProperty(name="Gores", default=8, min=2, max=64,
                           description="Number of solid lunes")
        res: IntProperty(
            name="Resolution", default=48, min=16, max=192,
            description="Latitude rows in the pattern zone "
                        "(longitudes = 3x)")
        add_light: BoolProperty(
            name="Add Point Light", default=False,
            description="Add a small point light at the north "
                        "pole to preview the shadow (Cycles or "
                        "Eevee with shadows)")
        add_floor: BoolProperty(
            name="Add Floor Plane", default=False,
            description="Large plane under the sphere so the "
                        "projected shadow pattern is visible")
        floor_size: FloatProperty(
            name="Floor Size", default=40.0, min=4.0, max=400.0,
            description="Floor plane side length, in sphere radii")

        def execute(self, context):
            pq = _valid_pq(self.tile_p, self.tile_q)
            if self.pattern == 'TILING' and pq != (self.tile_p,
                                                   self.tile_q):
                self.report({'WARNING'},
                            "{%d,%d} is not spherical; using "
                            "{%d,%d}" % (self.tile_p, self.tile_q,
                                         *pq))
            verts, faces = build_shell(
                self.pattern, self.radius, self.thickness,
                self.width, self.extent, self.spacing, self.rings,
                self.rays, *pq, self.gores, self.res, self.bowl,
                self.petals)
            me = bpy.data.meshes.new("Stereographic")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            loose = [v for v in bm.verts if not v.link_faces]
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.update()
            obj = bpy.data.objects.new(
                "Stereographic %s" % self.pattern.title(), me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.add_light:
                # the ideal projection point N = (0,0,2R) lies
                # inside the solid cap, so the light sits just
                # below the cap's inner wall to reach the holes
                li = bpy.data.lights.new("Stereographic Light",
                                         'POINT')
                li.energy = 100.0
                li.shadow_soft_size = 0.005 * self.radius
                lo = bpy.data.objects.new("Stereographic Light",
                                          li)
                context.collection.objects.link(lo)
                lo.parent = obj
                lo.location = (0.0, 0.0, 2.0 * self.radius -
                               min(self.thickness, self.radius))
            if self.add_floor:
                s = self.floor_size * self.radius / 2.0
                fme = bpy.data.meshes.new("Stereographic Floor")
                fme.from_pydata(
                    [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0),
                     (-s, s, 0.0)], [], [[0, 1, 2, 3]])
                fme.update()
                fo = bpy.data.objects.new("Stereographic Floor",
                                          fme)
                context.collection.objects.link(fo)
                fo.parent = obj
                fo.location = (0.0, 0.0, 0.0)
            self.report({'INFO'},
                        "V=%d F=%d" % (len(me.vertices),
                                       len(me.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'pattern')
            lay.prop(self, 'bowl')
            lay.prop(self, 'radius')
            lay.prop(self, 'thickness')
            lay.prop(self, 'width')
            lay.prop(self, 'extent')
            if self.pattern in ('GRID', 'FLOWER'):
                lay.prop(self, 'spacing')
            if self.pattern == 'FLOWER':
                lay.prop(self, 'petals')
            elif self.pattern == 'POLAR':
                lay.prop(self, 'rings')
                lay.prop(self, 'rays')
            elif self.pattern == 'TILING':
                lay.prop(self, 'tile_p')
                lay.prop(self, 'tile_q')
            elif self.pattern == 'BEACHBALL':
                lay.prop(self, 'gores')
            lay.prop(self, 'res')
            lay.prop(self, 'add_light')
            lay.prop(self, 'add_floor')
            if self.add_floor:
                lay.prop(self, 'floor_size')

    def _menu_func(self, context):
        self.layout.operator("mesh.stereographic_add",
                             icon='LIGHT_POINT')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_stereographic_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_stereographic_add)


def _selftest():
    from collections import Counter
    ok_all = True
    for pat in ('GRID', 'POLAR', 'TILING', 'BEACHBALL',
                'FLOWER'):
        for bowl in (False, True):
            v, f = build_shell(pat, res=32, bowl=bowl)
            cnt = Counter()
            for fc in f:
                for i in range(len(fc)):
                    a, b = fc[i], fc[(i + 1) % len(fc)]
                    cnt[(min(a, b), max(a, b))] += 1
            man = all(c == 2 for c in cnt.values())
            ok_all = ok_all and man
            print("%-9s bowl=%d verts=%d faces=%d "
                  "watertight=%s %s"
                  % (pat, bowl, len(v), len(f), man,
                     'OK' if man else 'BAD'))
    assert ok_all
    print("stereographic standalone tests passed")
