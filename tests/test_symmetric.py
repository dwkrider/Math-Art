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
    # collections outlive their objects, so a leftover empty
    # "SymSculpt Orbits" would shadow the next run's, which then
    # lands as "...001"
    for c in list(bpy.data.collections):
        if c.name.startswith('SymSculpt'):
            bpy.data.collections.remove(c)


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
for preset, fam in (('FRABJOUS', 'P2'), ('KRULL', 'P5'),
                    ('WHIMSY', 'P1')):
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
    # The check is that the count matches a weld predicted from the
    # motif itself -- no invariant on the group sizes.  Two attempts
    # at one were both wrong: they are neither drawn from {1,2,3,5}
    # (Frabjous stores a whole S, which its plane's 2-fold carries
    # onto itself, so every count doubles to 2, 4, 6) nor divisors of
    # the copy count (Krull's five arms all lie in one orbit and its
    # five copies coincide, so each of the 12 vertices takes 25 of
    # the 300 arm-ends).  Sizes are printed for the record instead.
    ok = (got == want and pmv == mv and ss.PRESETS[preset][1] == fam)
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
for preset in ('FRABJOUS', 'KRULL', 'WHIMSY'):
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
# than the compact Whimsy blade
ok = lifts['FRABJOUS'] > lifts['WHIMSY']
print(f"[lift scales with motif] frabjous={lifts['FRABJOUS']:.2f} > "
      f"whimsy={lifts['WHIMSY']:.2f} "
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

# Show Defining Polyhedron: the solid, a ball per vertex and a disc
# per mark, all keyed to the same per-class materials, and none of it
# reaching the render
clear()
bpy.ops.object.symmetric_sculpture_add(preset='FRABJOUS',
                                       show_polyhedron=True)
solid = [o for o in bpy.data.objects
         if o.name.startswith('SymSculpt Polyhedron')][0]
orb_coll = bpy.data.collections['SymSculpt Orbits']
ball_objs = sorted((o for o in orb_coll.objects
                    if o.name.endswith('Balls')), key=lambda o: o.name)
mark_objs = sorted((o for o in orb_coll.objects
                    if o.name.endswith('Marks')), key=lambda o: o.name)
cross = ss.crossing_points('ICOSA', 'P2', 1.0, 3.2, 0)
orbit, cgroup = ss.crossing_orbits('ICOSA', 'P2', cross, 1.0)
norb = len(set(cgroup))
ok = (len(solid.data.vertices) == 32 and len(solid.data.polygons) == 30
      and len(ball_objs) == norb and len(mark_objs) == norb
      and all(o.hide_render
              for o in ball_objs + mark_objs + [solid])
      and all(o.parent and o.parent.name.startswith('SymSculpt')
              for o in ball_objs + mark_objs + [solid]))
print(f"[polyhedron] V={len(solid.data.vertices)}(32) "
      f"F={len(solid.data.polygons)}(30) orbits={len(ball_objs)}"
      f"({norb}) {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('polyhedron')

# one object per orbit, each a single colour, and no two orbits
# sharing one -- that is what makes a group identifiable
cols = [tuple(round(c, 4) for c in o.data.materials[0].diffuse_color[:3])
        for o in ball_objs]
ok = len(set(cols)) == len(cols) and all(
    len(o.data.materials) == 1 for o in ball_objs + mark_objs)
print(f"[orbit colours] {len(set(cols))} distinct of {len(cols)} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('orbit-colours')

# ... and a ball object's colour matches its own marks object
pairs = 0
for b in ball_objs:
    m = bpy.data.objects.get(b.name.replace('Balls', 'Marks'))
    if m is None:
        continue
    pairs += 1
    if (tuple(round(c, 4) for c in b.data.materials[0].diffuse_color[:3])
            != tuple(round(c, 4)
                     for c in m.data.materials[0].diffuse_color[:3])):
        pairs = -99
print(f"[ball/mark colours agree] {pairs} pairs "
      f"{'OK' if pairs == norb else 'FAIL'}")
if pairs != norb:
    fails.append('orbit-pairing')

# every crossing gets a disc and every orbit point a ball -- faces
# per marker taken from the builders rather than written in by hand
fpd = len(ss._disc((0.0, 0.0, 0.0), 1.0)[1])
fpb = len(ss._ball((0.0, 0.0, 0.0), 1.0)[1])
nd = sum(len(o.data.polygons) for o in mark_objs) // fpd
nb = sum(len(o.data.polygons) for o in ball_objs) // fpb
ok = nd == len(cross) and nb == len(orbit)
print(f"[marker counts] discs={nd}({len(cross)}) "
      f"balls={nb}({len(orbit)}) {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('marker-counts')

# each orbit really is one orbit: its balls all sit at one radius
# from the origin, since a rotation cannot change that
ok = True
for o in ball_objs:
    rs = {round(math.dist(tuple(v.co), (0, 0, 0)), 4)
          for v in o.data.vertices}
    # a ball has extent, so allow its own diameter
    if max(rs) - min(rs) > 0.08:
        ok = False
print(f"[orbits are equidistant] {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('orbit-radius')

# orbit numbering must describe what is actually there: contiguous
# from 00, innermost first, never the raw discovery index
ids = sorted(int(o.name.split()[2]) for o in ball_objs)
inner = [min(v.co.length for v in
             bpy.data.objects[f"SymSculpt Orbit {i:02d} Balls"]
             .data.vertices) for i in ids]
ok = ids == list(range(len(ids))) and inner == sorted(inner)
print(f"[orbit numbering] ids 0..{ids[-1]} contiguous, "
      f"innermost first {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('orbit-numbering')

# the general planes run to hundreds of orbits, so the aid caps them
# and says so rather than filling the outliner
clear()
bpy.ops.object.symmetric_sculpture_add(preset='WHIMSY',
                                       show_polyhedron=True)
cap = [o for o in bpy.data.collections['SymSculpt Orbits'].objects
       if o.name.endswith('Balls')]
cap_ids = sorted(int(o.name.split()[2]) for o in cap)
ok = (len(cap) == ss._MAX_ORBITS
      and cap_ids == list(range(len(cap))))
print(f"[orbit cap] {len(cap)}({ss._MAX_ORBITS}) orbits, "
      f"ids 0..{cap_ids[-1]} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('orbit-cap')

clear()
bpy.ops.object.symmetric_sculpture_add(preset='FRABJOUS',
                                       show_polyhedron=True)
orb_coll = bpy.data.collections['SymSculpt Orbits']
ball_objs = sorted((o for o in orb_coll.objects
                    if o.name.endswith('Balls')), key=lambda o: o.name)
mark_objs = sorted((o for o in orb_coll.objects
                    if o.name.endswith('Marks')), key=lambda o: o.name)
balls, discs = ball_objs[0], mark_objs[0]
solid = [o for o in bpy.data.objects
         if o.name.startswith('SymSculpt Polyhedron')][0]

# EVERY disc sits on a real crossing -- this is what the old
# vertex-based marks got wrong on the general planes, where most of
# them floated in open space
segs = ss.stellation_lines('ICOSA', 'P2', 1.0, 3.2)


def _line_offsets(x, y):
    return sorted(abs((b[0] - a2[0]) * (a2[1] - y)
                      - (a2[0] - x) * (b[1] - a2[1]))
                  / max(math.dist(a2, b), 1e-9) for a2, b in segs)


worst = max(_line_offsets(x, y)[k - 1] for x, y, k in cross)
print(f"[every mark is a crossing] worst offset {worst:.2e} "
      f"{'OK' if worst < 1e-6 else 'FAIL'}")
if worst >= 1e-6:
    fails.append('marks-on-crossings')

# and the same has to hold for Whimsy's general planes, thinned
w_cross = ss.crossing_points('ICOSA', 'P1', 1.0, 3.2, 3)
w_segs = ss.stellation_lines('ICOSA', 'P1', 1.0, 3.2, 3)
w_worst = 0.0
for x, y, k in w_cross:
    offs = sorted(abs((b[0] - a2[0]) * (a2[1] - y)
                      - (a2[0] - x) * (b[1] - a2[1]))
                  / max(math.dist(a2, b), 1e-9) for a2, b in w_segs)
    w_worst = max(w_worst, offs[k - 1])
ok = len(w_cross) > 0 and w_worst < 1e-6
print(f"[whimsy marks are crossings] {len(w_cross)} crossings, "
      f"worst {w_worst:.2e} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('whimsy-crossings')

# the solid and its balls rise with the sculpture and keep tracking
# Lift afterwards; the guide marks belong to the flat diagram and
# stay down at the origin with the motif and guides
def _wz(o):
    dg = bpy.context.evaluated_depsgraph_get()
    return o.evaluated_get(dg).matrix_world.translation.z


lift_now = get_input(bpy.context.object, 'Lift')
sc_obj2 = bpy.context.object
ok = (abs(_wz(solid) - lift_now) < 1e-5
      and abs(_wz(balls) - lift_now) < 1e-5
      and abs(_wz(discs)) < 1e-5)
print(f"[polyhedron lifts] solid z={_wz(solid):.3f} "
      f"balls z={_wz(balls):.3f} marks z={_wz(discs):.3f} "
      f"(lift={lift_now:.3f}) {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('polyhedron-lift')

set_input(sc_obj2, 'Lift', 9.0)
ok = (abs(_wz(solid) - 9.0) < 1e-5 and abs(_wz(balls) - 9.0) < 1e-5
      and abs(_wz(discs)) < 1e-5)
print(f"[polyhedron tracks lift] solid z={_wz(solid):.3f}(9.000) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('polyhedron-lift-track')

clear()
bpy.ops.object.symmetric_sculpture_add(preset='FRABJOUS',
                                       show_polyhedron=True,
                                       lift=False)
sv = [o for o in bpy.data.objects
      if o.name.startswith('SymSculpt Polyhedron')][0]
ok = abs(_wz(sv)) < 1e-5
print(f"[polyhedron unlifted] z={_wz(sv):.3f}(0.000) "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('polyhedron-unlifted')

# Machinable part: a closed solid of the asked-for thickness, laid
# flat below the XY plane and centred on the Z axis so it can be
# picked and exported without disturbing the design objects
for preset, thick in (('WHIMSY', 0.03), ('FRABJOUS', 0.02),
                      ('KRULL', 0.05)):
    clear()
    # thickness follows Shell x Distance, so it always agrees with
    # the sculpture the part came from
    bpy.ops.object.symmetric_sculpture_add(preset=preset,
                                           show_part=True,
                                           shell=thick, distance=1.0)
    part = [o for o in bpy.data.objects
            if o.name.startswith('SymSculpt Part')][0]
    xs = [v.co.x for v in part.data.vertices]
    ys = [v.co.y for v in part.data.vertices]
    zs = [v.co.z for v in part.data.vertices]
    used = {}
    for p in part.data.polygons:
        vs = list(p.vertices)
        for i in range(len(vs)):
            e = vs[i], vs[(i + 1) % len(vs)]
            used[(min(e), max(e))] = used.get((min(e), max(e)), 0) + 1
    ok = (abs(max(zs) - min(zs) - thick) < 1e-6      # thickness
          and max(zs) < -0.5                         # well below XY
          and abs(min(xs) + max(xs)) < 1e-6          # centred on Z
          and abs(min(ys) + max(ys)) < 1e-6
          and set(used.values()) == {2})             # closed solid
    print(f"[part {preset}] {len(part.data.vertices)}v "
          f"thick={max(zs) - min(zs):.4f}({thick}) "
          f"top z={max(zs):.3f} closed={set(used.values()) == {2}} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'part-{preset}')

# The part is built from the motif that is actually in use, so a
# supplied Motif Object is what gets machined -- and it is machined
# WHOLE.  This quad is off-centre and carries none of the plane's own
# 2-fold symmetry, so the sculpture puts two of it in every plane and
# the part is both slabs: 8 verts each, the second the first turned a
# half turn.  Machining only the drawn one would leave half the
# material in that plane uncut.
clear()
me2 = bpy.data.meshes.new("PartSrc")
me2.from_pydata([(0.2, 0.1, 0.0), (0.8, 0.1, 0.0), (0.8, 0.5, 0.0),
                 (0.2, 0.5, 0.0)], [], [[0, 1, 2, 3]])
me2.update()
bpy.context.collection.objects.link(
    bpy.data.objects.new("MyPartMotif", me2))
bpy.ops.object.symmetric_sculpture_add(preset='FRABJOUS',
                                       motif_object="MyPartMotif",
                                       show_part=True, shell=0.05)
part = [o for o in bpy.data.objects
        if o.name.startswith('SymSculpt Part')][0]
zs = [v.co.z for v in part.data.vertices]
xy = [(v.co.x, v.co.y) for v in part.data.vertices]
turned = all(any(abs(x + p) < 1e-5 and abs(y + q) < 1e-5
                 for p, q in xy) for x, y in xy)
ok = (len(part.data.vertices) == 16
      and abs(max(zs) - min(zs) - 0.05) < 1e-6
      and turned)
print(f"[part from motif object] {len(part.data.vertices)}v(16) "
      f"half-turn pair={turned} {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('part-motif-object')

# off by default
clear()
bpy.ops.object.symmetric_sculpture_add(preset='FRABJOUS')
ok = not [o for o in bpy.data.objects
          if o.name.startswith('SymSculpt Part')]
print(f"[part off by default] {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('part-default')
ok = not [o for o in bpy.data.objects
          if o.name.startswith('SymSculpt Polyhedron')]
print(f"[polyhedron off by default] {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('polyhedron-default')

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
bpy.ops.object.symmetric_sculpture_add(preset='FRABJOUS',
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
bpy.ops.object.symmetric_sculpture_add(preset='FRABJOUS',
                                       motif_object="NoSuchObject")
built = [o for o in bpy.data.objects
         if o.name.startswith('SymSculpt Motif')]
print(f"[motif object fallback] built preset motif={bool(built)} "
      f"{'OK' if built else 'FAIL'}")
if not built:
    fails.append('motif-object-fallback')

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
