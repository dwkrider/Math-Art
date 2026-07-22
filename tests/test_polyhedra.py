# Headless test for the polyhedron add-ons: Conway operators, zonohedra,
# Waterman polyhedra, rotegrity, weave.
# Run:  blender --background --factory-startup --python tests/test_polyhedra.py -- [out_dir]
import sys
import os
import math

import bpy
import bmesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))
import conway_operators as co  # noqa: E402
import zonohedra_generator as zo  # noqa: E402
import waterman_generator as wa  # noqa: E402
import rotegrity_generator as ro  # noqa: E402
import weave_generator as we  # noqa: E402

for m in (co, zo, wa, ro, we):
    m.register()

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(PROJ, 'renders')
os.makedirs(OUT, exist_ok=True)
fails = []


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'SINGLE'
    sc.display.shading.single_color = (0.65, 0.7, 0.6)
    sc.display.shading.show_cavity = True
    sc.render.resolution_x = 640
    sc.render.resolution_y = 640


def render(path, view_dir=(1.0, -0.8, 0.65)):
    from mathutils import Vector
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    cen = (lo + hi) / 2
    diag = (hi - lo).length
    d = Vector(view_dir).normalized()
    cd = bpy.data.cameras.new("C")
    cam = bpy.data.objects.new("C", cd)
    bpy.context.collection.objects.link(cam)
    cam.location = cen + d * max(diag, 0.1) * 1.35
    cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def stats(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    nb = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    nm = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    chi = len(bm.verts) - len(bm.edges) + len(bm.faces)
    bm.free()
    return len(obj.data.vertices), len(obj.data.polygons), nb, nm, chi


setup_render()

# ---- Conway --------------------------------------------------------------
for notation, (ev, ef) in [("tI", (60, 32)), ("sD", (60, 92)),
                           ("eC", (24, 26)), ("gD", (92, 60)),
                           ("cC", (32, 18)), ("dkt5daD", (240, 122)),
                           ("pC", (32, 30)), ("pkD", (212, 240))]:
    clear()
    bpy.ops.mesh.conway_add(example='CUSTOM', notation=notation)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    want = (nv, nf) == (ev, ef) if ev else True
    ok = nb == 0 and nm == 0 and chi == 2 and want
    print(f"[conway {notation}] V={nv} F={nf} chi={chi} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'conway-{notation}')
render(os.path.join(OUT, 'poly_conway.png'))

# ---- Zonohedra -----------------------------------------------------------
for kind, kw, expect in [
        ('RHOMBIC_DODECA', {}, (14, 12)),
        ('TRIACONTA', {}, (32, 30)),
        ('POLAR', dict(n=12), (134, 132)),
        ('SPIRAL', dict(n=12, spiral_width=4), (None, None)),
        ('ENNEACONTA', {}, (92, 90))]:
    clear()
    bpy.ops.mesh.zonohedron_add(kind=kind, **kw)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nm == 0 and nf > 4
    if expect[0]:
        ok = ok and (nv, nf) == expect
    if kind in ('RHOMBIC_DODECA', 'TRIACONTA', 'ENNEACONTA', 'POLAR'):
        ok = ok and nb == 0 and chi == 2
    print(f"[zono {kind}] V={nv} F={nf} bnd={nb} chi={chi} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'zono-{kind}')
clear()
bpy.ops.mesh.zonohedron_add(kind='SPIRAL', n=12, spiral_width=4)
render(os.path.join(OUT, 'poly_spirallohedron.png'),
       view_dir=(1.0, -0.6, 0.35))
clear()
bpy.ops.mesh.zonohedron_add(kind='POLAR', n=12)
render(os.path.join(OUT, 'poly_polar_zonohedron.png'),
       view_dir=(1.0, -0.6, 0.35))

# ---- Waterman ------------------------------------------------------------
for root, expect in [(1, (12, 14)), (2, (6, 8)), (10, None), (100, None)]:
    clear()
    bpy.ops.mesh.waterman_add(root=root)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nb == 0 and nm == 0 and chi == 2
    if expect:
        ok = ok and (nv, nf) == expect
    print(f"[waterman W{root}] V={nv} F={nf} chi={chi} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'waterman-{root}')
render(os.path.join(OUT, 'poly_waterman.png'))

# ---- Rotegrity -----------------------------------------------------------
clear()
bpy.ops.mesh.rotegrity_add(kind='ICOSA', freq=2)
obj = bpy.context.object
nv, nf, nb, nm, chi = stats(obj)
ok = nb == 0 and nm == 0 and chi == 2 * 120   # 120 closed strap boxes
print(f"[rotegrity icosa f2] V={nv} F={nf} chi={chi} (straps=120) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('rotegrity')
render(os.path.join(OUT, 'poly_rotegrity.png'))

# ---- Weave ---------------------------------------------------------------
for kind, freq, pattern in (('CUBE', 1, 'FEV'), ('ICOSA', 1, 'FEV'),
                            ('ICOSA', 2, 'FEV'),
                            ('CUBE', 1, '1,1,1V'),
                            ('CUBE', 1, '1,1,1FFE'),
                            ('DODECA', 1, '0,1,0:0.12FEV')):
    clear()
    bpy.ops.mesh.poly_weave_add(kind=kind, freq=freq,
                                pattern_preset='CUSTOM', pattern=pattern)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nb == 0 and nm == 0 and chi == 0   # closed tori strips: chi=0 each
    print(f"[weave {kind} f{freq} '{pattern}'] V={nv} F={nf} chi={chi} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'weave-{kind}-{pattern}')
clear()
bpy.ops.mesh.poly_weave_add(kind='ICOSA', freq=2, pattern_preset='CUSTOM',
                            pattern='FEV', width=0.07)
render(os.path.join(OUT, 'poly_weave.png'))
clear()
bpy.ops.mesh.poly_weave_add(kind='ICOSA', freq=1, pattern_preset='CUSTOM',
                            pattern='FEV', width=0.12)
render(os.path.join(OUT, 'poly_weave_pentagons.png'))

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
print("RENDERS ->", OUT)
