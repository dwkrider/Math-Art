# Headless test driver for the Scherk-Collins add-on.
# Run:  blender --background --factory-startup --python test_scherk.py -- <out_dir>
import sys
import os
import math

import bpy
import bmesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The whole package, not the module on its own: the sculpture's
# parameters now live in the Math Art sidebar's per-object settings,
# which only exist once the package has installed that framework.
sys.path.insert(0, PROJ)
import math_art  # noqa: E402
from math_art import scherk_collins_generator as scg  # noqa: E402
from math_art.live.registry import settings_for_object  # noqa: E402

math_art.register()

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(PROJ, 'renders')
os.makedirs(OUT, exist_ok=True)

DEMO_DIR = r"C:\Users\dkrid\Downloads\SculptGen\Sculpture Generator\bin\demo"

CASES = []
for preset in ('CUSTOM', 'HEX', 'TREFOIL', 'MONKEY', 'HEPTOROID', 'TOWER'):
    CASES.append(('preset_' + preset.lower(), ('preset', preset)))
for n in (1, 5, 9, 11, 13, 15, 19):
    CASES.append((f'demo{n}', ('spec', os.path.join(DEMO_DIR, f'demo{n}.txt'))))


def clear_objects():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def mesh_stats(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    nonman = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    bm.free()
    return len(obj.data.vertices), len(obj.data.polygons), boundary, nonman


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'SINGLE'
    sc.display.shading.single_color = (0.75, 0.68, 0.55)
    sc.display.shading.show_cavity = True
    sc.render.resolution_x = 640
    sc.render.resolution_y = 640
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("W") if sc.world is None else sc.world


def render_object(obj, path, view_dir=(1.0, -1.0, 0.65)):
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
    diag = math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    dist = max(diag, 0.1) * 1.35
    from mathutils import Vector
    d = Vector(view_dir).normalized()
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = Vector((cx, cy, cz)) + d * dist
    cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


setup_render()
results = []
for name, (kind, arg) in CASES:
    clear_objects()
    if kind == 'preset':
        bpy.ops.mesh.scherk_collins_add(preset=arg)
        obj = bpy.context.object
    else:
        with open(arg, 'r', errors='replace') as f:
            d = scg.parse_spec_text(f.read())
        keys = set(scg.MESH_OT_scherk_collins_add._PARAM_KEYS)
        bpy.ops.mesh.scherk_collins_add(
            **{k: v for k, v in d.items() if k in keys})
        obj = bpy.context.object
    nv, nf, nb, nm = mesh_stats(obj)
    _info, st = settings_for_object(obj)
    p = scg._params_from_props(st)
    closes = scg.ring_closes(p)
    solid = p.thickness > 1e-6
    ok = (nb == 0) if (solid and closes) else True
    results.append((name, nv, nf, nb, nm, closes, solid, ok))
    print(f"[{name}] verts={nv} faces={nf} boundary_edges={nb} "
          f"nonmanifold={nm} closes={closes} solid={solid} "
          f"{'OK' if ok else 'LEAK!'}")
    render_object(obj, os.path.join(OUT, name + '.png'))
    if name in ('preset_trefoil', 'preset_tower'):
        render_object(obj, os.path.join(OUT, name + '_top.png'),
                      view_dir=(0.05, 0.05, 1.0))

print("\nSUMMARY")
for r in results:
    print(f"  {r[0]:20s} verts={r[1]:7d} faces={r[2]:7d} bnd={r[3]:5d} "
          f"nm={r[4]:4d} closes={r[5]} solid={r[6]} {'OK' if r[7] else 'LEAK'}")
print("RENDERS ->", OUT)
