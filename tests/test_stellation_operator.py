# Headless test for the merged Stellation operator.
#
# Covers the things the merge could plausibly have broken: that the operator
# kept the bl_idname and the `solid` item ids a pre-merge .blend stores, that
# all seven seeds build, that the three selection modes agree with each other
# where they describe the same solid, and that a chiral figure is really a
# different solid from its reflexible namesake rather than a relabelling.
#
# Run:  blender --background --factory-startup --python \
#           tests/test_stellation_operator.py
#
# (--factory-startup is fine here: the module is registered directly below
# rather than through the installed extension.)
import sys
import os

import bpy

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))
import general_stellation as gs  # noqa: E402

gs.ADD_MENU = False
gs.register()

FAILS = []


def chk(ok, msg):
    print(('  OK   ' if ok else '  BAD  ') + msg)
    if not ok:
        FAILS.append(msg)


def wipe():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def build(**kw):
    """Run the operator; return (result, object)."""
    wipe()
    res = bpy.ops.mesh.icosahedron_stellation_add(**kw)
    return res, bpy.context.view_layer.objects.active


def geometry(obj):
    """Rounded vertex set and face-vertex sets -- enough to tell two
    stellations of the same arrangement apart, which vertex and face
    COUNTS are not: Crennell 22 and 47 share both counts."""
    vs = sorted(tuple(round(c, 6) for c in v.co) for v in obj.data.vertices)
    fs = sorted(tuple(sorted(p.vertices)) for p in obj.data.polygons)
    return vs, fs


def volume(obj):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    v = bm.calc_volume(signed=True)
    bm.free()
    return v


# ---------------------------------------------------------------- identity
print('\n=== identity preserved across the merge ===')
cls = bpy.types.MESH_OT_icosahedron_stellation_add
chk(cls.bl_idname == 'mesh.icosahedron_stellation_add',
    'bl_idname unchanged, so pre-merge objects still resolve')
chk(cls.bl_label == 'Stellation', 'bl_label is %r' % cls.bl_label)

# Operator properties live on the INSTANCE rna, not on cls.bl_rna (which is
# the Operator base and carries only bl_label, bl_options and friends).
rna = bpy.ops.mesh.icosahedron_stellation_add.get_rna_type()
props = set(rna.properties.keys())
for p in ('seed', 'mode', 'solid', 'preset', 'cell_code', 'hand',
          'sh_a', 'sh_e1', 'sh_f1', 'sh_g3', 'sh_28'):
    chk(p in props, 'has property %r' % p)

items = [i.identifier for i in rna.properties['solid'].enum_items]
chk(items[:3] == ['1', '2', '3'] and 'CUSTOM' in items and '59' in items,
    "`solid` still uses the stored item ids '1'..'59' + CUSTOM")
chk(rna.properties['solid'].default == '8',
    "`solid` default is still '8' (final stellation)")
chk(rna.properties['seed'].default == 'icosahedron',
    'seed defaults to the icosahedron, so old files rebuild unchanged')

# ------------------------------------------------------------ all fifty-nine
print('\n=== every Crennell index builds ===')
bad, geo = [], {}
for k in range(1, 60):
    try:
        res, obj = build(solid=str(k))
        if res != {'FINISHED'} or obj is None or not obj.data.polygons:
            bad.append(k)
        else:
            geo[k] = (geometry(obj), round(volume(obj), 6))
    except Exception as exc:                       # noqa: BLE001
        bad.append((k, str(exc)[:60]))
chk(not bad, 'all 59 build (failures: %r)' % (bad[:6],))

print('\n=== a chiral figure is a different solid, not a relabelling ===')
# 22 and 47 are both Du Val 'Ef1': 22 fills the whole chiral shell, 47 one
# hand.  They share a vertex set AND a face count, so only the faces
# themselves and the volume separate them.
if 22 in geo and 47 in geo:
    (v22, f22), vol22 = geo[22]
    (v47, f47), vol47 = geo[47]
    chk(v22 == v47, '22 and 47 share a vertex set (same arrangement)')
    chk(f22 != f47, '22 and 47 have different faces')
    chk(abs(vol22 - vol47) > 1e-6,
        '22 and 47 differ in volume (%.6f vs %.6f)' % (vol22, vol47))
    chk(geo[10][0][1] != geo[33][0][1],
        '10 and 33 (bare f1, both hands vs one) differ too')

# ------------------------------------------------------------------- seeds
print('\n=== all seven seeds ===')
for s in gs.SEEDS:
    try:
        res, obj = build(seed=s, mode='PRESET')
        chk(res == {'FINISHED'} and obj is not None
            and len(obj.data.polygons) > 0,
            '%-26s V=%d F=%d' % (s, len(obj.data.vertices),
                                 len(obj.data.polygons)))
    except Exception as exc:                       # noqa: BLE001
        chk(False, '%s raised %s' % (s, str(exc)[:60]))

# ------------------------------------------------------- the three modes
print('\n=== the selection modes agree where they mean the same solid ===')
OFF = dict(sh_d=False, sh_e1=False, sh_e2=False, sh_f1=False, sh_f2=False,
           sh_g1=False, sh_g2=False, sh_g3=False)
res, obj = build(mode='CUSTOM', sh_a=True, sh_b=True, sh_c=True, **OFF)
shells = geometry(obj)
res, obj = build(solid='3')                     # Crennell 3 == 'C' == a b c
chk(shells == geometry(obj), "shells a+b+c == Crennell 3")
res, obj = build(mode='CODE', cell_code='a b c')
chk(shells == geometry(obj), "cell code 'a b c' == the same toggles")
res, obj = build(mode='CODE', cell_code='a, b, c')
chk(shells == geometry(obj), 'commas accepted as separators')

print('\n=== legacy spelling: solid=CUSTOM meant shell mode pre-merge ===')
res, obj = build(mode='PRESET', solid='CUSTOM', sh_a=True, sh_b=True,
                 sh_c=True, **OFF)
chk(shells == geometry(obj), "solid='CUSTOM' still selects shell mode")

print('\n=== hand suffix in a cell code ===')
res, obj = build(mode='CODE', cell_code='a b c d e1 e2 f1 f2')
both = geometry(obj)
res, obj = build(mode='CODE', cell_code='a b c d e1 e2 f1a f2')
one = geometry(obj)
chk(both != one, "'f1a' keeps one hand and differs from the whole shell")

# --------------------------------------------------------------- failures
print('\n=== bad input is reported, not raised ===')
for kw, what in ((dict(mode='CODE', cell_code='zz'), 'unknown shell'),
                 (dict(mode='CODE', cell_code=''), 'empty code'),
                 (dict(mode='CUSTOM', sh_a=False, sh_b=False, sh_c=False,
                       **OFF), 'no shells on')):
    wipe()
    try:
        res = bpy.ops.mesh.icosahedron_stellation_add(**kw)
        chk(res == {'CANCELLED'}, '%s cancels cleanly' % what)
    except Exception as exc:                       # noqa: BLE001
        # Blender turns a reported ERROR into RuntimeError when an operator
        # is called from script; either way it must not be a raw traceback
        # from deep in the engine.
        chk('Error' in type(exc).__name__,
            '%s reports rather than crashing (%s)' % (what, str(exc)[:50]))

print('\n=== the deprecated shim still builds ===')
chk(hasattr(bpy.ops.mesh, 'general_stellation_add'), 'shim is registered')
wipe()
res = bpy.ops.mesh.general_stellation_add(seed='cuboctahedron')
obj = bpy.context.view_layer.objects.active
chk(res == {'FINISHED'} and obj is not None and len(obj.data.polygons) > 0,
    'shim forwards to the merged operator and builds')

print('\n' + '=' * 60)
print('RESULT: %s (%d failures)' % ('OK' if not FAILS else 'FAIL', len(FAILS)))
for m in FAILS:
    print('  - ' + m)
sys.exit(1 if FAILS else 0)
