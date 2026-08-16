# Headless tests for the Symmetric Sculpture designer (after Hart's
# JMA 2007 "Symmetric sculpture" software).
# Run:  blender --background --factory-startup --python tests/test_symmetric.py
import sys
import os
import math

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


def get_input(obj, name):
    """Read one of the modifier's group inputs by name."""
    mod = obj.modifiers[0]
    for it in mod.node_group.interface.items_tree:
        if it.name == name and it.in_out == 'INPUT':
            return mod[it.identifier]
    return None


def set_full(obj, val):
    """Toggle the modifier's Full Sculpture input."""
    set_input(obj, 'Full Sculpture', val)


def weld_prediction(preset):
    """Replicate a preset's motif by hand and count the coincident
    vertices, so the expected weld is derived rather than guessed.

    A point shared by k copies collapses k->1, and by symmetry there
    are N/k such groups, so a shared class costs N - N/k vertices --
    48 at the 5-fold hubs, 40 at the 3-fold corners, 30 for a plain
    pair.  Totals are therefore NOT multiples of the copy count."""
    kind, fam, builder = ss.PRESETS[preset]
    mverts, _ = builder(1.0)
    a, _ = ss.plane_normals(kind, fam)
    u, v = ss._frame(a)
    # cluster at the modifier's own Weld distance, and look in the
    # neighbouring cells too: coincident points are only equal to
    # within the motif's stored precision, so a plain rounding key
    # splits a shared tip across two buckets and hides the 5- and
    # 3-fold groups entirely
    tol = 1e-4
    cells = {}
    counts = []
    for R in ss.group_rotations(kind):
        for x, y, z in mverts:
            p = tuple(a[i] * (1.0 + z) + x * u[i] + y * v[i]
                      for i in range(3))
            q = ss._apply(R, p)
            key = tuple(int(math.floor(c / tol)) for c in q)
            hit = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for idx, rep in cells.get(
                                (key[0] + dx, key[1] + dy,
                                 key[2] + dz), ()):
                            if math.dist(rep, q) <= tol:
                                hit = idx
                                break
                        if hit is not None:
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    break
            if hit is None:
                counts.append(1)
                cells.setdefault(key, []).append((len(counts) - 1, q))
            else:
                counts[hit] += 1
    merged = sum(n - 1 for n in counts)
    return len(mverts), merged, sorted(set(counts))


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

# default view: lifted, so all 60 copies are kept and none is ghosted
# -- shell 0.04 doubles each copy's verts
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()
ok = len(ev.vertices) == 60 * mv * 2 \
    and not any(m and m.name == 'SymSculpt Copies'
                for m in ev.materials) \
    and min(v.co.z for v in ev.vertices) > 0.0
print(f"[default view] {len(ev.vertices)}({60 * mv * 2}) opaque lifted "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('default-view')

# Translucent is its own switch now, independent of Full Sculpture:
# it only swaps the material, never the copy count
set_input(sc_obj, 'Translucent', True)
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()
ok = len(ev.vertices) == 60 * mv * 2 \
    and any(m and m.name == 'SymSculpt Copies' for m in ev.materials)
print(f"[translucent on] {len(ev.vertices)}({60 * mv * 2}) ghost mat "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('translucent-on')
set_input(sc_obj, 'Translucent', False)

# unlifted + Full Sculpture off: the motif's own copy is dropped so
# the Motif object stands alone in its plane (59 copies)
set_input(sc_obj, 'Lift', 0.0)
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()
ok = len(ev.vertices) == 59 * mv * 2
print(f"[unlifted hides motif copy] {len(ev.vertices)}"
      f"({59 * mv * 2}) {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('unlifted-hide')

set_full(sc_obj, True)
deps = bpy.context.evaluated_depsgraph_get()
ev = sc_obj.evaluated_get(deps).to_mesh()
ok = len(ev.vertices) == 60 * mv * 2
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

# each preset lands in the right plane family, and the sculpture's
# vertex count matches a weld predicted from the motif itself
for preset, fam in (('TWISTED_RIVERS', 'P3'), ('TUMBLEWEED', 'P5'),
                    ('FRABJOUS', 'P2'), ('WHIMSY', 'P1')):
    clear()
    bpy.ops.object.symmetric_sculpture_add(preset=preset, shell=0.0)
    so = bpy.context.object
    mv = len([o for o in bpy.data.objects
              if o.name.startswith('SymSculpt Motif')
              ][0].data.vertices)
    set_full(so, True)
    deps = bpy.context.evaluated_depsgraph_get()
    ev = so.evaluated_get(deps).to_mesh()
    pmv, merged, sizes = weld_prediction(preset)
    want = 60 * mv - merged
    got = len(ev.vertices)
    # copies may only meet in the ways the symmetry allows: pairs, or
    # threes at a 3-fold corner, or fives at a 5-fold hub
    ok = (got == want and pmv == mv and ss.PRESETS[preset][1] == fam
          and set(sizes) <= {1, 2, 3, 5})
    print(f"[preset {preset}] verts={got}({want}) fam={fam} "
          f"merged={merged} groups={sizes} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'preset-{preset}')

    # the guides have to reach at least as far as the motif they are
    # drawn for -- Frabjous runs its tips out to the 3-fold piercings
    # at phi^2 = 2.618, well past the old 2.2 guide disc, so the very
    # crossing its corners sit on was not being drawn
    mo = [o for o in bpy.data.objects
          if o.name.startswith('SymSculpt Motif')][0]
    gd = [o for o in bpy.data.objects
          if o.name.startswith('SymSculpt Guides')][0]
    r_motif = max(math.hypot(v.co.x, v.co.y) for v in mo.data.vertices)
    r_guide = max(math.hypot(v.co.x, v.co.y) for v in gd.data.vertices)
    ok = r_guide >= r_motif
    print(f"[guides cover {preset}] motif r={r_motif:.3f} "
          f"guides r={r_guide:.3f} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'guides-{preset}')

# Lift: raises the modifier result up +Z, leaves the Motif and Guides
# objects behind at the origin, and keeps the motif's own copy (the
# sculpture is already clear, so there is nothing to hide).
clear()
bpy.ops.object.symmetric_sculpture_add(preset='CUSTOM', group='ICOSA',
                                       family='P3', shell=0.0,
                                       lift=False)
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
                                       family='P3', shell=0.04,
                                       lift=False)
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

# the default lift must clear the motif plane by a real margin, for
# every preset -- measured on the evaluated result, not by re-running
# the same formula the operator used
lifts = {}
for preset in ('TWISTED_RIVERS', 'TUMBLEWEED', 'FRABJOUS', 'WHIMSY'):
    clear()
    bpy.ops.object.symmetric_sculpture_add(preset=preset)
    so = bpy.context.object
    lifts[preset] = get_input(so, 'Lift')
    deps = bpy.context.evaluated_depsgraph_get()
    ev = so.evaluated_get(deps).to_mesh()
    z0 = min(v.co.z for v in ev.vertices)
    z1 = max(v.co.z for v in ev.vertices)
    rad = (z1 - z0) / 2.0
    ratio = z0 / rad                      # gap as a share of radius
    ok = z0 > 0.0 and 0.15 < ratio < 0.6
    print(f"[lift clears {preset}] gap={z0:.3f} radius={rad:.3f} "
          f"ratio={ratio:.2f} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'lift-clear-{preset}')

# and it must track the motif's size: Frabjous runs out to phi^2 in
# its plane, so at the same plane distance it needs a bigger lift
# than the compact Tumbleweed flower
ok = lifts['FRABJOUS'] > lifts['TUMBLEWEED']
print(f"[lift scales with motif] frabjous={lifts['FRABJOUS']:.2f} > "
      f"tumbleweed={lifts['TUMBLEWEED']:.2f} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('lift-scaling')

clear()
bpy.ops.object.symmetric_sculpture_add(preset='CUSTOM', group='ICOSA',
                                       family='P3', translucent=True)
so = bpy.context.object
ok = bool(get_input(so, 'Translucent'))
print(f"[translucent checkbox] {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('translucent-checkbox')

# Guide Rings thins the pattern: fewer edges than the full set, and
# what survives is the inner lines
clear()
bpy.ops.object.symmetric_sculpture_add(preset='WHIMSY')
full_g = len([o for o in bpy.data.objects
              if o.name.startswith('SymSculpt Guides')
              ][0].data.edges)
clear()
bpy.ops.object.symmetric_sculpture_add(preset='WHIMSY', guide_rings=3)
gd = [o for o in bpy.data.objects
      if o.name.startswith('SymSculpt Guides')][0]
few_g = len(gd.data.edges)
ok = 0 < few_g < full_g
print(f"[guide rings] {few_g} edges of {full_g} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('guide-rings')

# the family labels have to name the planes the chosen group really
# makes -- the same P3 is 20 icosahedral planes but 8 octahedral
ok = ('20 icosahedral' in ss.family_label('ICOSA', 'P3')
      and '8 octahedral' in ss.family_label('OCTA', 'P3')
      and '4 tetrahedral' in ss.family_label('TETRA', 'P3'))
print(f"[family labels group-aware] {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('family-labels')
for g, f, n in (('OCTA', 'P3', 8), ('TETRA', 'P2', 6)):
    clear()
    bpy.ops.object.symmetric_sculpture_add(preset='CUSTOM', group=g,
                                           family=f, shell=0.0)
    so = bpy.context.object
    mv = len([o for o in bpy.data.objects
              if o.name.startswith('SymSculpt Motif')][0].data.vertices)
    _, normals = ss.plane_normals(g, f)
    ok = len(normals) == n
    print(f"[{g}/{f} planes] {len(normals)}({n}) "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'planes-{g}-{f}')

# Motif Object: name a mesh and it is replicated instead of the
# preset motif; name something that is not a mesh and the operator
# falls back rather than failing
clear()
me = bpy.data.meshes.new("Custom")
# kept clear of the local origin on purpose: a vertex at (0, 0, 0)
# lands on the plane's centre of symmetry, where all k in-plane
# copies coincide and weld together (Hart's "center-point")
me.from_pydata([(0.2, 0.1, 0.0), (0.8, 0.1, 0.0), (0.8, 0.5, 0.0),
                (0.2, 0.5, 0.0)], [], [[0, 1, 2, 3]])
me.update()
custom = bpy.data.objects.new("MyMotif", me)
bpy.context.collection.objects.link(custom)
bpy.ops.object.symmetric_sculpture_add(preset='TWISTED_RIVERS',
                                       motif_object="MyMotif",
                                       shell=0.0)
so = bpy.context.object
deps = bpy.context.evaluated_depsgraph_get()
ev = so.evaluated_get(deps).to_mesh()
ok = (len(ev.vertices) == 60 * 4          # our 4-vert quad, 60 copies
      and not [o for o in bpy.data.objects
               if o.name.startswith('SymSculpt Motif')])
print(f"[motif object] {len(ev.vertices)}(240) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('motif-object')

clear()
bpy.ops.object.symmetric_sculpture_add(preset='TWISTED_RIVERS',
                                       motif_object="NoSuchObject")
built = [o for o in bpy.data.objects
         if o.name.startswith('SymSculpt Motif')]
print(f"[motif object fallback] built preset motif={bool(built)} "
      f"{'OK' if built else 'FAIL'}")
if not built:
    fails.append('motif-object-fallback')

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
