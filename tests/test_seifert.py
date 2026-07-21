# Headless test for the Seifert Surface Generator.
# Run:  blender --background --factory-startup --python tests/test_seifert.py -- [out_dir]
import sys
import os
import math

import bpy
import bmesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'src'))
import seifert_surface_generator as ssg  # noqa: E402

ssg.register()

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
    sc.display.shading.single_color = (0.72, 0.6, 0.75)
    sc.display.shading.show_cavity = True
    sc.render.resolution_x = 640
    sc.render.resolution_y = 640


def render_scene(path, view_dir=(1.0, -0.8, 0.75)):
    from mathutils import Vector
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for o in bpy.data.objects:
        if o.type not in ('MESH', 'CURVE'):
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    cen = (lo + hi) / 2
    diag = (hi - lo).length
    d = Vector(view_dir).normalized()
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = cen + d * max(diag, 0.1) * 1.3
    cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def surface_checks(obj, exp_mu, exp_chi):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    chi = len(bm.verts) - len(bm.edges) + len(bm.faces)
    nonman = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    bm.free()
    loops = ssg._boundary_loops(me)
    return chi, len(loops), nonman


setup_render()

CASES = [
    ('TREFOIL', 1, -1, 1),
    ('FIGURE8', 1, -1, 1),
    ('CINQUEFOIL', 1, -3, 2),
    ('GRANNY', 1, -3, 2),
    ('HOPF', 2, 0, 0),
    ('BORROMEAN', 3, -3, 1),
]
for preset, mu, chi_exp, genus in CASES:
    clear_objects()
    bpy.ops.mesh.seifert_surface_add(preset=preset)
    obj = bpy.context.object
    if obj.type != 'MESH':   # knot curve may be active; find the surface
        obj = next(o for o in bpy.data.objects if o.type == 'MESH')
    chi, nloops, nonman = surface_checks(obj, mu, chi_exp)
    ok = (chi == chi_exp and nloops == mu and nonman == 0
          and obj["genus"] == genus)
    print(f"[{preset}] chi={chi}({chi_exp}) loops={nloops}({mu}) "
          f"nonman={nonman} genus={obj['genus']:g}({genus}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(preset)
    if preset in ('TREFOIL', 'FIGURE8', 'BORROMEAN'):
        render_scene(os.path.join(OUT, f'seifert_{preset.lower()}.png'))

# torus knot preset
clear_objects()
bpy.ops.mesh.seifert_surface_add(preset='TORUS', torus_p=3, torus_q=4)
obj = next(o for o in bpy.data.objects if o.type == 'MESH')
# (3,4) torus knot: strands 3, crossings 8, knot => mu=1, chi=-5, g=3
chi, nloops, nonman = surface_checks(obj, 1, -5)
ok = chi == -5 and nloops == 1 and obj["genus"] == 3
print(f"[TORUS(3,4)] chi={chi}(-5) loops={nloops}(1) genus={obj['genus']:g}(3) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('TORUS')
render_scene(os.path.join(OUT, 'seifert_torus34.png'))

# relaxed trefoil (soft dependency on the toolkit)
clear_objects()
bpy.ops.mesh.seifert_surface_add(preset='TREFOIL', relax=15)
obj = next(o for o in bpy.data.objects if o.type == 'MESH')
chi, nloops, nonman = surface_checks(obj, 1, -1)
ok = chi == -1 and nloops == 1
print(f"[TREFOIL relaxed] chi={chi} loops={nloops} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('relax')
render_scene(os.path.join(OUT, 'seifert_trefoil_relaxed.png'))

# custom braid string
clear_objects()
bpy.ops.mesh.seifert_surface_add(preset='CUSTOM', braid="1 -2 1 -2")
obj = next(o for o in bpy.data.objects if o.type == 'MESH')
ok = obj["crossings"] == 4 and obj["strands"] == 3
print(f"[custom braid] crossings={obj['crossings']} strands={obj['strands']} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('custom')

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
print("RENDERS ->", OUT)
