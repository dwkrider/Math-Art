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


def set_input(obj, name, val):
    """Set one of the modifier's group inputs by name."""
    mod = obj.modifiers[0]
    for it in mod.node_group.interface.items_tree:
        if it.name == name and it.in_out == 'INPUT':
            mod[it.identifier] = val
    obj.update_tag()
    bpy.context.view_layer.update()


def set_full(obj, val):
    """Toggle the modifier's Full Sculpture input."""
    set_input(obj, 'Full Sculpture', val)


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

# design view: the motif's own copy is dropped (59 ghosts), shell
# 0.04 doubles each copy's verts; ghost material on the copies
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()
ok = len(ev.vertices) == 59 * mv * 2 \
    and any(m and m.name == 'SymSculpt Copies' for m in ev.materials)
print(f"[design view] {len(ev.vertices)}({59 * mv * 2}) ghost mat "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('design')

# Full Sculpture: all 60 copies, motif's own material
set_full(sc_obj, True)
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()
ok = len(ev.vertices) == 60 * mv * 2 \
    and not any(m and m.name == 'SymSculpt Copies'
                for m in ev.materials)
print(f"[full sculpture] {len(ev.vertices)}({60 * mv * 2}) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('full')
set_full(sc_obj, False)
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()

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
    set_full(so, True)
    deps = bpy.context.evaluated_depsgraph_get()
    ev = so.evaluated_get(deps).to_mesh()
    ok = len(ev.vertices) == n * mv
    print(f"[{g}/{f} flat] {len(ev.vertices)}({n * mv}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'flat-{g}')

# sculpture presets set the right plane family; Frabjous is the one
# preset whose copies actually meet -- its S-halves join across the
# 2-fold axis and its tips meet in threes -- so 26 verts per copy
# weld away (1560 of 60x172).  The other three presets are open
# forms whose parts pass without touching, so nothing merges.
for preset, fam, merged in (('TWISTED_RIVERS', 'P3', 0),
                            ('TUMBLEWEED', 'P5', 0),
                            ('FRABJOUS', 'P2', 1560),
                            ('WHIMSY', 'P1', 0)):
    clear()
    bpy.ops.object.symmetric_sculpture_add(preset=preset, shell=0.0)
    so = bpy.context.object
    mv = len([o for o in bpy.data.objects
              if o.name.startswith('SymSculpt Motif')
              ][0].data.vertices)
    set_full(so, True)
    deps = bpy.context.evaluated_depsgraph_get()
    ev = so.evaluated_get(deps).to_mesh()
    want = 60 * mv - merged
    got = len(ev.vertices)
    # the weld must be symmetric: every copy has to lose the same
    # number of vertices, or the parts are not meeting cleanly
    ok = (got == want and ss.PRESETS[preset][1] == fam
          and (60 * mv - got) % 60 == 0)
    print(f"[preset {preset}] verts={got}({want}) fam={fam} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'preset-{preset}')

# Lift: raises the modifier result up +Z, leaves the Motif and Guides
# objects behind at the origin, and keeps the motif's own copy (the
# sculpture is already clear, so there is nothing to hide).
clear()
bpy.ops.object.symmetric_sculpture_add(preset='CUSTOM', group='ICOSA',
                                       family='P3', shell=0.0)
so = bpy.context.object
motif = [o for o in bpy.data.objects
         if o.name.startswith('SymSculpt Motif')][0]
guides = [o for o in bpy.data.objects
          if o.name.startswith('SymSculpt Guides')][0]
mv = len(motif.data.vertices)
deps = bpy.context.evaluated_depsgraph_get()
ev = so.evaluated_get(deps).to_mesh()
z_flat = min(v.co.z for v in ev.vertices)
ok = len(ev.vertices) == 59 * mv          # unlifted: motif copy hidden
print(f"[lift off] {len(ev.vertices)}({59 * mv}) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('lift-off')

set_input(so, 'Lift', 3.0)
deps = bpy.context.evaluated_depsgraph_get()
ev = so.evaluated_get(deps).to_mesh()
z_up = min(v.co.z for v in ev.vertices)
ok = (len(ev.vertices) == 60 * mv                    # all copies kept
      and abs((z_up - z_flat) - 3.0) < 1e-5          # moved exactly 3
      and abs(motif.matrix_world.translation.z) < 1e-9
      and abs(guides.matrix_world.translation.z) < 1e-9)
print(f"[lift on] {len(ev.vertices)}({60 * mv}) dz={z_up - z_flat:.4f} "
      f"motif z={motif.matrix_world.translation.z:.4f} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('lift-on')

# the lift is applied after the radial shell, so extruding then
# lifting must equal lifting a rigid copy -- if it ran before the
# shell it would change every point's distance from the origin
clear()
bpy.ops.object.symmetric_sculpture_add(preset='CUSTOM', group='ICOSA',
                                       family='P3', shell=0.04)
so = bpy.context.object
set_full(so, True)
deps = bpy.context.evaluated_depsgraph_get()
ev = so.evaluated_get(deps).to_mesh()
flat = sorted((round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5))
              for v in ev.vertices)
set_input(so, 'Lift', 5.0)
deps = bpy.context.evaluated_depsgraph_get()
ev = so.evaluated_get(deps).to_mesh()
up = sorted((round(v.co.x, 5), round(v.co.y, 5),
             round(v.co.z - 5.0, 5)) for v in ev.vertices)
ok = (len(flat) == len(up)
      and max(abs(p[i] - q[i]) for p, q in zip(flat, up)
              for i in range(3)) < 1e-4)
print(f"[lift preserves shell] {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('lift-shell')

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
