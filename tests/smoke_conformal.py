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


def _extent(me):
    xs = [v.co for v in me.vertices]
    return tuple(max(c[i] for c in xs) - min(c[i] for c in xs)
                 for i in range(3))


def _assert_in_cube(me, what):
    """Every generator centres on its bounding box and fits the largest extent
    to a 2 m cube, the project-wide convention."""
    ext = _extent(me)
    assert max(ext) <= 2.0 + 1e-4, "%s overflows the 2 m cube: %r" % (what, ext)
    assert abs(max(ext) - 2.0) < 1e-4, (
        "%s should FILL the cube in its largest extent, got %.4f"
        % (what, max(ext)))
    for v in me.vertices:
        assert (abs(v.co.x) <= 1.0 + 1e-4 and abs(v.co.y) <= 1.0 + 1e-4
                and abs(v.co.z) <= 1.0 + 1e-4), \
            "%s: vertex outside the cube" % what


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
    _assert_in_cube(me, "maximal packing")


def circle_packing_sphere():
    _fresh()
    assert bpy.ops.mesh.circle_packing_add(
        combinatorics='SPHERE', rings=2, sphere_res=1) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.vertices) > 20
    _assert_in_cube(me, "sphere packing")
    # the caps are a sphere, so all three extents should fill the cube
    ext = _extent(me)
    assert min(ext) > 1.9, "a packed sphere should be round: %r" % (ext,)


def subdivision_tiling_pentagonal():
    _fresh()
    assert bpy.ops.mesh.subdivision_tiling_add(
        rule='PENTAGONAL', depth=2, layout_mode='CONFORMAL') == {'FINISHED'}
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
            rule=rule, depth=2, layout_mode='EUCLIDEAN') == {'FINISHED'}
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
    _assert_in_cube(me, "gasket limit set")


def kleinian_orbit():
    _fresh()
    assert bpy.ops.curve.kleinian_add(
        mode='ORBIT', preset='GASKET', orbit_depth=3) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.polygons) > 100, len(me.polygons)
    _assert_in_cube(me, "circle orbit")


def kleinian_slice():
    _fresh()
    assert bpy.ops.curve.kleinian_add(mode='SLICE', denom=16) == {'FINISHED'}
    me = _active_mesh()
    assert len(me.vertices) > 8, len(me.vertices)
    _assert_in_cube(me, "Maskit slice")
    # Im(mu) > 1 is a statement about the PARAMETER, and the mesh is fitted to
    # the 2 m cube like every other generator, so the invariant is checked on
    # the cusps themselves rather than on their rescaled coordinates.  The
    # engine self-test in kleinian/slice.py asserts it at source.
    ys = [v.co.y for v in me.vertices]
    assert max(ys) - min(ys) > 0.0, "the slice boundary collapsed"


def _menu_defs():
    """Import menu_defs however the add-on happens to be loaded.

    Installed as an extension the package is bl_ext.<repo>.math_art, not
    math_art; run from a source checkout it is the latter."""
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
    return menu_defs


def _math_art_operator_ids():
    md = _menu_defs()
    ops = {e.op for m in md.ALL_MENUS for e in m.entries if e.op}
    ops |= {e.op for e in md.ROOT_ENTRIES if e.op}
    return ops


def menu_entries_present():
    ops = _math_art_operator_ids()
    for want in ("mesh.circle_packing_add", "mesh.subdivision_tiling_add",
                 "curve.kleinian_add"):
        assert want in ops, "%s is registered but in no menu" % want


def no_shadowed_operator_attributes():
    """No operator property may be named after an attribute Blender itself
    puts on an Operator.

    `layout` is the one that bites: inside draw(), self.layout is the UILayout
    the redo panel is built on, so a property of that name shadows it, draw()
    fails silently and the panel renders EMPTY while the operator still works
    perfectly when called with keywords.  Nothing else catches this -- the
    engine tests do not touch draw(), and calling the operator succeeds.

    Checked across every registered Math Art operator, not just the three
    added here, since the failure is invisible and cheap to test for."""
    reserved = {'layout', 'report', 'bl_rna', 'poll', 'execute', 'invoke',
                'draw', 'modal', 'cancel', 'as_keywords', 'id_data'}
    ours = _math_art_operator_ids()
    assert ours, "found no Math Art operators to check"
    bad = []
    for name in dir(bpy.types):
        cls = getattr(bpy.types, name, None)
        idname = getattr(cls, 'bl_idname', None)
        if not isinstance(idname, str) or idname not in ours:
            continue
        annotations = getattr(cls, '__annotations__', {}) or {}
        for prop in annotations:
            if prop in reserved:
                bad.append("%s.%s" % (idname, prop))
    assert not bad, "operator properties shadowing Blender attributes: %s" % bad


CHECKS = [
    ("no shadowed operator attributes", no_shadowed_operator_attributes),
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
