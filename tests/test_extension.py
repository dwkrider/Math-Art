# Headless test of the Math Art extension package: registers the
# package as a whole (relative imports, unified menu) and invokes one
# operator from each module.
# Run:  blender --background --factory-startup --python tests/test_extension.py
import sys
import os

import bpy

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import math_art  # noqa: E402

math_art.register()
fails = []

assert hasattr(bpy.types, 'VIEW3D_MT_math_art_add'), "unified menu missing"
print("[menu] VIEW3D_MT_math_art_add registered OK")

OPS = [
    ("scherk", lambda: bpy.ops.mesh.scherk_collins_add(preset='TREFOIL')),
    ("parametric", lambda: bpy.ops.mesh.parametric_minimal_add(
        surface='CATENOID')),
    ("tpms", lambda: bpy.ops.mesh.tpms_add(surface='G', resolution=16)),
    ("knot span", lambda: bpy.ops.mesh.minimal_knot_span_add(
        samples=64, rings=8, iterations=10)),
    ("seifert", lambda: bpy.ops.mesh.seifert_surface_add(
        preset='TREFOIL', relax=5)),
    ("conway", lambda: bpy.ops.mesh.conway_add(
        example='CUSTOM', notation='sC')),
    ("zonohedron", lambda: bpy.ops.mesh.zonohedron_add(
        kind='SPIRAL', n=12, spiral_width=4)),
    ("waterman", lambda: bpy.ops.mesh.waterman_add(root=20)),
    ("rotegrity", lambda: bpy.ops.mesh.rotegrity_add(kind='ICOSA', freq=1)),
    ("weave", lambda: bpy.ops.mesh.poly_weave_add(kind='CUBE')),
]
for name, op in OPS:
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    try:
        result = op()
        obj = bpy.context.object
        ok = result == {'FINISHED'} and obj is not None \
            and len(obj.data.vertices) > 0
    except Exception as e:
        ok = False
        print("   ", e)
    print(f"[{name}] {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)

# seifert relax exercises the cross-module relative import
math_art.unregister()
print("[unregister] OK")

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
