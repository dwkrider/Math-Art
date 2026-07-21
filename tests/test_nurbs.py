# Headless test for NURBS output modes of both add-ons.
# Run:  blender --background --factory-startup --python tests/test_nurbs.py -- [out_dir]
import sys
import os
import math

import bpy

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))
import scherk_collins_generator as scg  # noqa: E402
import minimal_surface_toolkit as mst  # noqa: E402

scg.register()
mst.register()

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(PROJ, 'renders')
os.makedirs(OUT, exist_ok=True)
fails = []


def clear_objects():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'SINGLE'
    sc.display.shading.single_color = (0.78, 0.70, 0.52)
    sc.render.resolution_x = 640
    sc.render.resolution_y = 640


def render_object(obj, path, view_dir=(1.0, -1.0, 0.65)):
    from mathutils import Vector
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    cen = obj.matrix_world @ Vector(((min(xs) + max(xs)) / 2,
                                     (min(ys) + max(ys)) / 2,
                                     (min(zs) + max(zs)) / 2))
    diag = math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    d = Vector(view_dir).normalized()
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = cen + d * max(diag, 0.1) * 1.4
    cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def surf_stats(obj):
    grids = [(sp.point_count_u, sp.point_count_v) for sp in obj.data.splines
             if sp.point_count_u > 1 and sp.point_count_v > 1]
    npts = sum(len(sp.points) for sp in obj.data.splines)
    return len(grids), npts


setup_render()

# ---- Scherk-Collins NURBS ----------------------------------------------
clear_objects()
bpy.ops.mesh.scherk_collins_add(preset='TREFOIL', output_nurbs=True)
obj = bpy.context.object
ok = obj.type == 'SURFACE'
ngrids, npts = surf_stats(obj)
print(f"[scherk nurbs trefoil] type={obj.type} patches={ngrids} "
      f"ctrl_pts={npts} {'OK' if ok and ngrids == 12 else 'FAIL'}")
if not (ok and ngrids == 12):   # 3 storeys x 2 branches x 2 half-wedges
    fails.append('scherk-nurbs')
render_object(obj, os.path.join(OUT, 'nurbs_trefoil.png'))

# toggle back to mesh through the property + direct rebuild
st = obj.scherk_collins
st.auto_update = False
st.output_nurbs = False
obj = scg.rebuild_object(obj)
print(f"[scherk toggle back] type={obj.type} verts={len(obj.data.vertices)} "
      f"{'OK' if obj.type == 'MESH' else 'FAIL'}")
if obj.type != 'MESH':
    fails.append('scherk-toggle')

# default NURBS detail should be 2 (few control points)
clear_objects()
bpy.ops.mesh.scherk_collins_add(preset='HEX', output_nurbs=True)
obj = bpy.context.object
ngrids, npts = surf_stats(obj)
st = obj.scherk_collins
ok = (obj.type == 'SURFACE' and st.nurbs_detail == 2 and npts < 1500)
print(f"[scherk nurbs hex default] nurbs_detail={st.nurbs_detail} "
      f"patches={ngrids} ctrl_pts={npts} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('scherk-nurbs-hex')
render_object(obj, os.path.join(OUT, 'nurbs_hex_d3.png'))

# ---- parametric NURBS ---------------------------------------------------
for kind, cyc in (('ENNEPER', True), ('CATENOID', True), ('CATALAN', False)):
    clear_objects()
    bpy.ops.mesh.parametric_minimal_add(surface=kind, output='NURBS',
                                        ctrl_u=20, ctrl_v=20)
    obj = bpy.context.object
    ngrids, npts = surf_stats(obj)
    ok = obj.type == 'SURFACE' and ngrids == 1 and npts <= 400
    print(f"[param nurbs {kind}] patches={ngrids} ctrl_pts={npts} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'param-nurbs-{kind}')
render_object(obj, os.path.join(OUT, 'nurbs_catalan.png'))

# ---- trefoil-circle span NURBS -----------------------------------------
clear_objects()
bpy.ops.mesh.minimal_knot_span_add(samples=96, rings=16, iterations=30,
                                   output_nurbs=True)
obj = bpy.context.object
ngrids, npts = surf_stats(obj)
ok = obj.type == 'SURFACE' and ngrids == 1 and npts == 17 * 96
print(f"[knot span nurbs] patches={ngrids} ctrl_pts={npts} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('knot-nurbs')
render_object(obj, os.path.join(OUT, 'nurbs_trefoil_span.png'),
              view_dir=(0.4, -0.5, 1.0))

# ---- curve span NURBS (disk + annulus) ---------------------------------
clear_objects()
bpy.ops.curve.primitive_bezier_circle_add(radius=1.5)
bpy.ops.object.minimal_span(samples=64, rings=10, iterations=20,
                            output_nurbs=True)
obj = bpy.context.object
ok = obj.type == 'SURFACE'
print(f"[disk span nurbs] type={obj.type} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('disk-nurbs')

clear_objects()
bpy.ops.curve.primitive_bezier_circle_add(radius=1.5)
c1 = bpy.context.object
bpy.ops.curve.primitive_bezier_circle_add(radius=0.8, location=(0, 0, 1.0))
c2 = bpy.context.object
c1.select_set(True)
c2.select_set(True)
bpy.ops.object.minimal_span(samples=64, rings=12, iterations=20,
                            output_nurbs=True)
obj = bpy.context.object
ok = obj.type == 'SURFACE'
print(f"[annulus span nurbs] type={obj.type} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('annulus-nurbs')

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
print("RENDERS ->", OUT)
