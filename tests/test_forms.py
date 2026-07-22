# Headless test for the sculptural-form add-ons: polylinks, platonic
# twist, fractal polyhedra, symmetrohedra, twisted torus.
# Run:  blender --background --factory-startup --python tests/test_forms.py -- [out_dir]
import sys
import os
import math

import bpy
import bmesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))
import polylinks_generator as pl  # noqa: E402
import platonic_twist_generator as pt  # noqa: E402
import fractal_polyhedron_generator as fp  # noqa: E402
import symmetrohedron_generator as sy  # noqa: E402
import twisted_torus_generator as tt  # noqa: E402
import polytope4d_generator as p4  # noqa: E402
import tangle_generator as tg  # noqa: E402

for m in (pl, pt, fp, sy, tt, p4, tg):
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
    sc.display.shading.single_color = (0.75, 0.62, 0.5)
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

# ---- Polylinks: each frame is a closed torus solid (chi = 0) ------------
for preset, nframes in (('T4', 4), ('C6', 6), ('D6', 6), ('I20', 20)):
    clear()
    bpy.ops.mesh.polylinks_add(preset=preset)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nb == 0 and nm == 0 and chi == 0
    print(f"[polylinks {preset}] frames={nframes} V={nv} chi={chi} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'polylinks-{preset}')
clear()
bpy.ops.mesh.polylinks_add(preset='T4')
render(os.path.join(OUT, 'form_polylinks_t4.png'))
clear()
bpy.ops.mesh.polylinks_add(preset='D6')
render(os.path.join(OUT, 'form_polylinks_d6.png'))

# ---- Platonic twist: welded surface, no nonmanifold ---------------------
for kind in ('TETRA', 'CUBE', 'DODECA'):
    clear()
    bpy.ops.mesh.platonic_twist_add(kind=kind, thickness=0.0)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nm == 0 and nb > 0 and nf > 50
    print(f"[twist {kind}] V={nv} F={nf} bnd={nb} nm={nm} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'twist-{kind}')
clear()
bpy.ops.mesh.platonic_twist_add(kind='CUBE')
render(os.path.join(OUT, 'form_platonic_twist.png'))

# ---- Fractal: Sierpinski counts ----------------------------------------
clear()
bpy.ops.mesh.fractal_polyhedron_add(kind='TETRA', mode='VERTS',
                                    generations=4, child_scale=0.5,
                                    spread=1.0, keep_parents=False)
obj = bpy.context.object
nv = len(obj.data.vertices)
ok = nv == 4 ** 4 * 4     # 256 copies x 4 verts
print(f"[fractal sierpinski g4] verts={nv} (exp 1024) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('fractal')
render(os.path.join(OUT, 'form_sierpinski.png'))
# default cube fractal: per-generation materials + attribute
clear()
bpy.ops.mesh.fractal_polyhedron_add()
obj = bpy.context.object
nmats = len(obj.data.materials)
attr = obj.data.attributes.get('generation')
ok = nmats == 4 and attr is not None   # gens 0..3 with keep_parents
print(f"[fractal default coloring] materials={nmats}(4) "
      f"attr={'yes' if attr else 'NO'} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('fractal-color')
render(os.path.join(OUT, 'form_fractal_cube.png'))
clear()
bpy.ops.mesh.fractal_polyhedron_add(kind='ICOSA', mode='FACES',
                                    generations=2, child_scale=0.35,
                                    keep_parents=True)
render(os.path.join(OUT, 'form_fractal_icosa.png'))

# ---- Symmetrohedra: closed convex solids --------------------------------
for group, m1, m2, m3 in (('I', 1, 1, 0), ('O', 2, 1, 0), ('I', 2, 0, 1)):
    clear()
    bpy.ops.mesh.symmetrohedron_add(group=group, mult1=m1, mult2=m2,
                                    mult3=m3)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nb == 0 and nm == 0 and chi == 2
    print(f"[symmetro {group}({m1},{m2},{m3})] V={nv} F={nf} chi={chi} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'symmetro-{group}{m1}{m2}{m3}')
clear()
bpy.ops.mesh.symmetrohedron_add(group='I', mult1=1, mult2=1, mult3=0)
render(os.path.join(OUT, 'form_symmetrohedron.png'))

# ---- Twisted torus: closed torus, band count ----------------------------
for n, tw, bands in ((3, 1, 1), (6, 2, 2), (4, 0, 4)):
    clear()
    bpy.ops.mesh.twisted_torus_add(n=n, twist_steps=tw)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nb == 0 and nm == 0 and chi == 0
    print(f"[twisted torus n={n} tw={tw}] V={nv} chi={chi} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'ttorus-{n}-{tw}')
clear()
bpy.ops.mesh.twisted_torus_add(n=3, twist_steps=1, rounding=0.15)
render(os.path.join(OUT, 'form_twisted_torus.png'))

# ---- twisted torus sheets (Triangle Shrink < 1): one object per band ----
for n, tw, shrink, nobjs in ((3, 1, 0.8, 1), (6, 2, 0.7, 2),
                             (4, 0, 0.75, 4)):
    clear()
    bpy.ops.mesh.twisted_torus_add(n=n, twist_steps=tw, shrink=shrink,
                                   segments=96)
    objs = [o for o in bpy.data.objects if o.type == 'MESH']
    ok = len(objs) == nobjs
    for o in objs:
        nv, nf, nb, nm, chi = stats(o)
        ok = ok and nb == 0 and nm == 0 and chi == 0
        sharp = sum(1 for e in o.data.edges if e.use_edge_sharp)
        ok = ok and sharp > 0
    print(f"[torus sheets n={n} tw={tw}] objects={len(objs)}({nobjs}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'sheets-{n}-{tw}')
clear()
bpy.ops.mesh.twisted_torus_add(n=3, twist_steps=1, shrink=0.75)
render(os.path.join(OUT, 'form_torus_sheets.png'))

# ---- platonic twist join modes ------------------------------------------
clear()
bpy.ops.mesh.platonic_twist_add(kind='CUBE', thickness=0.0,
                                smooth_joins=False)
obj = bpy.context.object
sharp = sum(1 for e in obj.data.edges if e.use_edge_sharp)
ok = sharp > 0
print(f"[twist creased joins] sharp_edges={sharp} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('twist-sharp')
clear()
bpy.ops.mesh.platonic_twist_add(kind='CUBE', thickness=0.0,
                                smooth_joins=True)
obj = bpy.context.object
nv, nf, nb, nm, chi = stats(obj)
sharp = sum(1 for e in obj.data.edges if e.use_edge_sharp)
ok = sharp == 0 and nm == 0 and nf > 50
print(f"[twist C1 joins] sharp_edges={sharp} nm={nm} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('twist-c1')

# ---- 4D polytopes -------------------------------------------------------
# each strut and sphere is a closed manifold component: total chi =
# 2 * (edges + vertices) with spheres on
for kind, style, (env, ene) in (('CELL8', 'CURVED', (16, 32)),
                                ('CELL24', 'STRAIGHT', (24, 96)),
                                ('CELL600', 'CURVED', (120, 720))):
    clear()
    bpy.ops.mesh.polytope4d_add(kind=kind, style=style, arc_segments=6,
                                sides=5)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = nb == 0 and nm == 0 and chi == 2 * (env + ene)
    print(f"[4d {kind} {style}] chi={chi} (exp {2 * (env + ene)}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'4d-{kind}')
clear()
bpy.ops.mesh.polytope4d_add(kind='CELL8', style='CURVED')
render(os.path.join(OUT, 'form_tesseract.png'))
clear()
bpy.ops.mesh.polytope4d_add(kind='CELL120', style='CURVED',
                            arc_segments=8, sides=5, radius=0.02)
render(os.path.join(OUT, 'form_120cell.png'))

# ---- Tangles: closed manifold frames, per-component materials -----------
# FACES (da Vinci) shells: a polyhedron with F faces gives a genus
# F-1 surface, chi = 4 - 2F per component. EDGES: closed strut boxes
# plus one knuckle hull per vertex, chi = 2 per solid.
for kind, style, ncomp, chi_exp in (
        ('T5', 'FACES', 5, 5 * (4 - 2 * 4)),
        ('T2', 'FACES', 2, 2 * (4 - 2 * 4)),
        ('C5', 'EDGES', 5, 2 * 5 * (12 + 8)),
        ('C3', 'FACES', 3, 3 * (4 - 2 * 6))):
    clear()
    bpy.ops.mesh.tangle_add(kind=kind, style=style)
    obj = bpy.context.object
    nv, nf, nb, nm, chi = stats(obj)
    ok = (nb == 0 and nm == 0 and chi == chi_exp
          and len(obj.data.materials) == ncomp
          and obj.data.attributes.get('component_index') is not None)
    print(f"[tangle {kind} {style}] chi={chi}({chi_exp}) "
          f"mats={len(obj.data.materials)}({ncomp}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'tangle-{kind}')
clear()
bpy.ops.mesh.tangle_add(kind='T5')
render(os.path.join(OUT, 'form_tangle_t5.png'),
       view_dir=(0.35, -0.6, 0.72))

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
print("RENDERS ->", OUT)
