# In-Blender smoke test for the conformal-geometry generators.
#
# The headless self-tests exercise the engines; this exercises the Blender
# layer -- operator registration, property defaults, mesh construction and the
# menu entries -- which the engine tests cannot reach.
#
# Run with the extension enabled (NOT --factory-startup, which disables it):
#     blender --background --python tests/smoke_conformal.py

import sys

import bpy

FAILURES = []


def check(name, fn):
    try:
        fn()
        print("[%s] OK" % name)
    except Exception as exc:                       # noqa: BLE001
        FAILURES.append((name, exc))
        print("[%s] FAIL: %s" % (name, exc))


def _fresh():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def _active_mesh():
    ob = bpy.context.view_layer.objects.active
    assert ob is not None, "no active object"
    assert ob.type == 'MESH', "expected a mesh, got %s" % ob.type
    return ob.data


def circle_packing_default():
    _fresh()
    assert bpy.ops.mesh.circle_packing_add(rings=4) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.vertices) > 100, len(me.vertices)
    assert len(me.polygons) > 100, len(me.polygons)


def circle_packing_maximal():
    _fresh()
    assert bpy.ops.mesh.circle_packing_add(
        rings=3, geometry='HYPERBOLIC', boundary='MAXIMAL') == {'FINISHED'}
    me = _active_mesh()
    assert len(me.vertices) > 50
    # a maximal packing lives in the unit disc
    far = max(max(abs(v.co.x), abs(v.co.y)) for v in me.vertices)
    assert far < 1.6, "maximal packing spilled well outside the disc: %.3f" % far


def circle_packing_sphere():
    _fresh()
    assert bpy.ops.mesh.circle_packing_add(
        combinatorics='SPHERE', rings=2, sphere_res=1) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.vertices) > 20
    far = max(v.co.length for v in me.vertices)
    assert far < 1.4, "spherical caps should sit on the unit sphere: %.3f" % far


def subdivision_tiling_pentagonal():
    _fresh()
    assert bpy.ops.mesh.subdivision_tiling_add(
        rule='PENTAGONAL', depth=2, layout='CONFORMAL') == {'FINISHED'}
    me = _active_mesh()
    assert len(me.polygons) == 36, len(me.polygons)
    assert all(len(p.vertices) == 5 for p in me.polygons), \
        "every tile must be a pentagon"


def subdivision_tiling_ribbons():
    _fresh()
    assert bpy.ops.mesh.subdivision_tiling_add(
        rule='PENTAGONAL', depth=2, output='EDGES', relief=0.2) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.polygons) == 36 * 5, len(me.polygons)
    assert any(abs(v.co.z) > 1e-9 for v in me.vertices), "relief did nothing"


def subdivision_tiling_other_rules():
    for rule, want in (('BARYCENTRIC', 36), ('QUAD', 16),
                       ('TRIANGLE_QUAD', 12)):
        _fresh()
        assert bpy.ops.mesh.subdivision_tiling_add(
            rule=rule, depth=2, layout='EUCLIDEAN') == {'FINISHED'}
        me = _active_mesh()
        assert len(me.polygons) == want, (rule, len(me.polygons), want)


def kleinian_limit_set():
    _fresh()
    assert bpy.ops.curve.kleinian_add(
        mode='CURVE', preset='GASKET', epsilon=0.02,
        max_depth=18) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.vertices) > 500, len(me.vertices)
    assert len(me.edges) > 400, len(me.edges)
    far = max(max(abs(v.co.x), abs(v.co.y)) for v in me.vertices)
    assert far < 1.01, "gasket limit set escaped the unit disc: %.4f" % far


def kleinian_orbit():
    _fresh()
    assert bpy.ops.curve.kleinian_add(
        mode='ORBIT', preset='GASKET', orbit_depth=3) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.polygons) > 100, len(me.polygons)


def kleinian_slice():
    _fresh()
    assert bpy.ops.curve.kleinian_add(mode='SLICE', denom=16) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.vertices) > 8, len(me.vertices)
    assert all(v.co.y > 1.0 - 1e-6 for v in me.vertices), \
        "every Maskit cusp must sit above Im(mu) = 1"


def menu_entries_present():
    # Installed as an extension the package is bl_ext.<repo>.math_art, not
    # math_art; run from a source checkout it is the latter.  Try both.
    import importlib

    menu_defs = None
    for name in ("bl_ext.user_default.math_art.menu_defs",
                 "bl_ext.vscode_development.math_art.menu_defs",
                 "math_art.menu_defs"):
        try:
            menu_defs = importlib.import_module(name)
            break
        except ImportError:
            continue
    if menu_defs is None:                       # find it wherever it landed
        for mod in list(sys.modules):
            if mod.endswith("math_art.menu_defs"):
                menu_defs = sys.modules[mod]
                break
    assert menu_defs is not None, "could not import menu_defs"
    ops = {e.op for m in menu_defs.ALL_MENUS for e in m.entries if e.op}
    ops |= {e.op for e in menu_defs.ROOT_ENTRIES if e.op}
    for want in ("mesh.circle_packing_add", "mesh.subdivision_tiling_add",
                 "curve.kleinian_add"):
        assert want in ops, "%s is registered but in no menu" % want


CHECKS = [
    ("circle packing default", circle_packing_default),
    ("circle packing maximal", circle_packing_maximal),
    ("circle packing sphere", circle_packing_sphere),
    ("subdivision tiling pentagonal", subdivision_tiling_pentagonal),
    ("subdivision tiling ribbons", subdivision_tiling_ribbons),
    ("subdivision tiling other rules", subdivision_tiling_other_rules),
    ("kleinian limit set", kleinian_limit_set),
    ("kleinian circle orbit", kleinian_orbit),
    ("kleinian maskit slice", kleinian_slice),
    ("menu entries present", menu_entries_present),
]

for nm, f in CHECKS:
    check(nm, f)

print("\n" + "=" * 56)
if FAILURES:
    print("RESULT: FAIL -- %d of %d checks failed" % (len(FAILURES),
                                                      len(CHECKS)))
    for nm, exc in FAILURES:
        print("   %s: %s" % (nm, exc))
    sys.exit(1)
print("ran %d checks: all OK" % len(CHECKS))
print("RESULT: OK")
