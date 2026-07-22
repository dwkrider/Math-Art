# Headless tests for the Symmetric Sculpture designer (after Hart's
# JMA 2007 "Symmetric sculpture" software).
# Run:  blender --background --factory-startup --python tests/test_symmetric.py
import sys
import os

import bpy

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))
import symmetric_sculpture_generator as ss  # noqa: E402

ss.register()
fails = []


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


# pure math: group orders and plane-family counts
for kind, order in ss._ORDER.items():
    got = len(ss.group_rotations(kind))
    if got != order:
        fails.append(f'order-{kind}')
    print(f"[group {kind}] {got}({order}) "
          f"{'OK' if got == order else 'FAIL'}")
EXPECT = {('ICOSA', 'P5'): 12, ('ICOSA', 'P3'): 20,
          ('ICOSA', 'P2'): 30, ('ICOSA', 'P1'): 60,
          ('OCTA', 'P4'): 6, ('OCTA', 'P3'): 8,
          ('OCTA', 'P2'): 12, ('OCTA', 'P1'): 24,
          ('TETRA', 'P3'): 4, ('TETRA', 'P2'): 6,
          ('TETRA', 'P1'): 12}
for (kind, fam), n in EXPECT.items():
    _, normals = ss.plane_normals(kind, fam)
    if len(normals) != n:
        fails.append(f'planes-{kind}-{fam}')
        print(f"[planes {kind}/{fam}] {len(normals)}({n}) FAIL")
print("[planes] all families OK" if not fails else "[planes] FAILURES")

# stellation pattern of the icosahedron: 18 non-parallel planes
segs = ss.stellation_lines('ICOSA', 'P3', 1.0, 10.0)
ok = len(segs) == 18
print(f"[stellation] {len(segs)}(18) {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('stellation')

# operator: creates motif + guides + GN sculpture, 60 live copies
clear()
bpy.ops.object.symmetric_sculpture_add(preset='CUSTOM', group='ICOSA',
                                       family='P3')
objs = {o.name.split('.')[0]: o for o in bpy.data.objects}
ok = {'SymSculpt', 'SymSculpt Motif', 'SymSculpt Guides'} <= set(objs)
print(f"[objects] {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('objects')
sc_obj = objs['SymSculpt']
motif = objs['SymSculpt Motif']
mv = len(motif.data.vertices)
ok = len(sc_obj.data.vertices) == 60 \
    and sc_obj.data.attributes.get('sym_rot') is not None \
    and sc_obj.data.attributes.get('copy_index') is not None
print(f"[points] {len(sc_obj.data.vertices)}(60) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('points')

# default shell 0.04: radial extrusion doubles each copy's verts
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()
ok = len(ev.vertices) == 60 * mv * 2
print(f"[gn shell] {len(ev.vertices)}({60 * mv * 2}) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('gn-shell')

# live: moving the motif object must move the replicated copies
before = ev.vertices[0].co.copy()
motif.location.x += 0.1
deps = bpy.context.evaluated_depsgraph_get()
ev2 = sc_obj.evaluated_get(deps).to_mesh()
moved = (ev2.vertices[0].co - before).length > 1e-6
print(f"[live update] {'OK' if moved else 'FAIL'}")
if not moved:
    fails.append('live')

# flat mode (shell=0) and the other groups
for g, f, n in (('ICOSA', 'P2', 60), ('OCTA', 'P4', 24),
                ('TETRA', 'P3', 12)):
    clear()
    bpy.ops.object.symmetric_sculpture_add(preset='CUSTOM', group=g,
                                           family=f, shell=0.0)
    so = bpy.context.object
    mv = len([o for o in bpy.data.objects
              if o.name.startswith('SymSculpt Motif')
              ][0].data.vertices)
    deps = bpy.context.evaluated_depsgraph_get()
    ev = so.evaluated_get(deps).to_mesh()
    ok = len(ev.vertices) == n * mv
    print(f"[{g}/{f} flat] {len(ev.vertices)}({n * mv}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'flat-{g}')

# sculpture presets set the right plane family; Frabjous halves weld
for preset, fam, merged in (('TWISTED_RIVERS', 'P3', 0),
                            ('TUMBLEWEED', 'P5', 0),
                            ('FRABJOUS', 'P2', 60)):
    clear()
    bpy.ops.object.symmetric_sculpture_add(preset=preset, shell=0.0)
    so = bpy.context.object
    mv = len([o for o in bpy.data.objects
              if o.name.startswith('SymSculpt Motif')
              ][0].data.vertices)
    deps = bpy.context.evaluated_depsgraph_get()
    ev = so.evaluated_get(deps).to_mesh()
    want = 60 * mv - merged
    ok = len(ev.vertices) == want
    print(f"[preset {preset}] verts={len(ev.vertices)}({want}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'preset-{preset}')

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
