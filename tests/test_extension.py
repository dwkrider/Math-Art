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
    ("polylinks", lambda: bpy.ops.mesh.polylinks_add(preset='T4')),
    ("platonic twist", lambda: bpy.ops.mesh.platonic_twist_add(
        kind='CUBE')),
    ("fractal", lambda: bpy.ops.mesh.fractal_polyhedron_add(
        generations=3)),
    ("symmetrohedron", lambda: bpy.ops.mesh.symmetrohedron_add()),
    ("twisted torus", lambda: bpy.ops.mesh.twisted_torus_add()),
    ("tangle", lambda: bpy.ops.mesh.tangle_add(kind='T5')),
    ("symmetric sculpture",
     lambda: bpy.ops.object.symmetric_sculpture_add(
         preset='TWISTED_RIVERS')),
    ("tangle struts", lambda: bpy.ops.mesh.tangle_add(
        kind='T5', style='EDGES')),
    ("zonohedron leonardo", lambda: bpy.ops.mesh.zonohedron_add(
        kind='TRIACONTA', style='LEONARDO')),
    ("conway leonardo", lambda: bpy.ops.mesh.conway_add(
        example='CUSTOM', notation='I', style='LEONARDO')),
    ("waterman leonardo", lambda: bpy.ops.mesh.waterman_add(
        root=20, style='LEONARDO')),
    ("polytope leonardo", lambda: bpy.ops.mesh.polytope4d_add(
        kind='CELL8', render='LEONARDO')),
    ("leonardo modifier",
     lambda: (bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1),
              bpy.ops.object.leonardo_add())[-1]),
    ("sponge", lambda: bpy.ops.mesh.sponge_add(kind='MENGER',
                                               level=2)),
    ("space curve", lambda: bpy.ops.curve.space_filling_add(
        kind='MOORE3D', order=3)),
    ("oloid", lambda: bpy.ops.mesh.oloid_add(kind='OLOID')),
    ("oloid roller", lambda: bpy.ops.mesh.oloid_add(kind='ROLLER')),
    ("prime knot", lambda: bpy.ops.curve.prime_knot_add(
        knot='6_2', iters=60)),
    ("prime knot tube", lambda: bpy.ops.curve.prime_knot_add(
        knot='8_18', output='MESH', iters=60)),
    ("solid archimedean", lambda: bpy.ops.mesh.regular_solid_add(
        family='ARCHIMEDEAN', solid='SD')),
    ("solid johnson", lambda: bpy.ops.mesh.regular_solid_add(
        family='JOHNSON', solid='J46')),
    ("solid kepler", lambda: bpy.ops.mesh.regular_solid_add(
        family='KEPLER', solid='GSD')),
    ("solid stellated", lambda: bpy.ops.mesh.regular_solid_add(
        family='PLATONIC', solid='DODECA', stellated=True)),
]
for name, op in OPS:
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    try:
        result = op()
        obj = bpy.context.object
        if hasattr(obj.data, 'vertices'):
            n_elem = len(obj.data.vertices)
        else:                              # curve output
            sp = obj.data.splines[0]
            n_elem = (len(sp.bezier_points) if sp.type == 'BEZIER'
                      else len(sp.points))
        ok = result == {'FINISHED'} and obj is not None and n_elem > 0
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
