# Headless test for the Minimal Surface Toolkit.
# Run:  blender --background --factory-startup --python tests/test_minimal.py -- [out_dir]
import sys
import os
import math

import bpy
import bmesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))
import minimal_surface_toolkit as mst  # noqa: E402

mst.register()

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(PROJ, 'renders')
os.makedirs(OUT, exist_ok=True)


def clear_objects():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'SINGLE'
    sc.display.shading.single_color = (0.62, 0.68, 0.78)
    sc.display.shading.show_cavity = True
    sc.render.resolution_x = 640
    sc.render.resolution_y = 640


def render_object(obj, path, view_dir=(1.0, -1.0, 0.65)):
    from mathutils import Vector
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    cen = obj.matrix_world @ Vector(((min(xs) + max(xs)) / 2,
                                     (min(ys) + max(ys)) / 2,
                                     (min(zs) + max(zs)) / 2))
    diag = math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    d = Vector(view_dir).normalized()
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = cen + d * max(diag, 0.1) * 1.4
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
    bm.free()
    return len(obj.data.vertices), len(obj.data.polygons), nb, nm


setup_render()
fails = []

# ---- parametric surfaces -------------------------------------------------
for kind in ('ENNEPER', 'CATENOID', 'HELICOID', 'HENNEBERG', 'CATALAN',
             'BOUR', 'RICHMOND', 'SCHERK1'):
    clear_objects()
    bpy.ops.mesh.parametric_minimal_add(surface=kind)
    obj = bpy.context.object
    nv, nf, nb, nm = stats(obj)
    ok = nv > 100 and nf > 100 and nm == 0
    print(f"[param {kind}] verts={nv} faces={nf} bnd={nb} nm={nm} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(kind)
    if kind in ('ENNEPER', 'CATENOID', 'HENNEBERG'):
        render_object(obj, os.path.join(OUT, f'min_{kind.lower()}.png'))

# ---- TPMS ---------------------------------------------------------------
for kind in ('P', 'D', 'G', 'NEOVIUS', 'IWP', 'FRD', 'LIDINOID', 'SPLITP',
             'SCHERKT'):
    clear_objects()
    bpy.ops.mesh.tpms_add(surface=kind, cells=1, resolution=28)
    obj = bpy.context.object
    nv, nf, nb, nm = stats(obj)
    ok = nv > 500 and nm == 0
    print(f"[tpms {kind}] verts={nv} faces={nf} bnd={nb} nm={nm} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(kind)
    if kind in ('G', 'P', 'NEOVIUS', 'SCHERKT'):
        render_object(obj, os.path.join(OUT, f'min_tpms_{kind.lower()}.png'))

# gyroid at 2 cells with thickness (printable lattice)
clear_objects()
bpy.ops.mesh.tpms_add(surface='G', cells=2, resolution=24, thickness=0.12)
render_object(bpy.context.object, os.path.join(OUT, 'min_tpms_g_2cells.png'))

# ---- span from actual Blender curve objects -----------------------------
clear_objects()
bpy.ops.curve.primitive_bezier_circle_add(radius=1.5)
circ = bpy.context.object
# wavy second loop above: a scaled/moved circle converted for variety
bpy.ops.curve.primitive_bezier_circle_add(radius=0.7,
                                          location=(0.3, 0, 1.2))
small = bpy.context.object
circ.select_set(True)
small.select_set(True)
bpy.ops.object.minimal_span(samples=96, rings=16, iterations=30)
obj = bpy.context.object
nv, nf, nb, nm = stats(obj)
ok = obj.name.startswith("MinimalSpan") and nm == 0
print(f"[span 2 curves] verts={nv} faces={nf} bnd={nb} nm={nm} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('span2')
render_object(obj, os.path.join(OUT, 'min_span_two_circles.png'),
              view_dir=(1, -0.4, 0.25))

clear_objects()
bpy.ops.curve.primitive_bezier_circle_add(radius=1.5)
circ = bpy.context.object
bpy.ops.object.minimal_span(samples=96, rings=14, iterations=25)
obj = bpy.context.object
zspread = max(abs(v.co.z) for v in obj.data.vertices)
print(f"[span 1 curve] flat-disk z-spread = {zspread:.2e} "
      f"{'OK' if zspread < 1e-4 else 'FAIL'}")
if zspread >= 1e-4:
    fails.append('span1')

# ---- circle <-> trefoil knot -------------------------------------------
clear_objects()
bpy.ops.mesh.minimal_knot_span_add()
obj = bpy.context.object
nv, nf, nb, nm = stats(obj)
ok = nm == 0 and nv > 1000
print(f"[trefoil span] verts={nv} faces={nf} bnd={nb} nm={nm} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('trefoil')
render_object(obj, os.path.join(OUT, 'min_trefoil_circle.png'),
              view_dir=(0.4, -0.5, 1.0))
render_object(obj, os.path.join(OUT, 'min_trefoil_circle_side.png'),
              view_dir=(1.0, -0.2, 0.15))


# the default span topology (Single Sheet / Seifert) must be ONE
# connected surface with NO genuine self-intersections: BVH self-overlap
# may only report face pairs that share a vertex (touching neighbours)
def genuine_crossings(obj):
    from mathutils.bvhtree import BVHTree
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    fv = [frozenset(v.index for v in f.verts) for f in bm.faces]
    tree = BVHTree.FromBMesh(bm, epsilon=0.0)
    n = sum(1 for i, j in tree.overlap(tree)
            if i < j and not (fv[i] & fv[j]))
    # connected components while the bmesh is open
    parent = list(range(len(bm.verts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in bm.edges:
        parent[find(e.verts[0].index)] = find(e.verts[1].index)
    comps = len({find(v.index) for v in bm.verts})
    bm.free()
    return n, comps


nx, comps = genuine_crossings(obj)
ok = nx == 0 and comps == 1 and len(bpy.data.objects) == 1
print(f"[trefoil single sheet] objects={len(bpy.data.objects)} "
      f"components={comps} genuine_crossings={nx} "
      f"{'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('trefoil-single-sheet')

# the single sheet has to survive the whole range the operator exposes,
# not just its defaults: the highest genus it builds, the coarsest and
# finest grids, and a solver setting past the relaxation cap (which the
# operator clamps -- an uncapped flow drags the sheet back through
# itself, which is exactly what the cap is there to prevent)
for label, kw in (("q9", dict(q=9)),
                  ("q7-coarse", dict(q=7, samples=32, rings=4)),
                  ("q5-fine", dict(q=5, samples=256, rings=48)),
                  ("iterations-past-cap", dict(iterations=200)),
                  ("lifted", dict(inner_lift=3.0, inner_height=5.0))):
    clear_objects()
    bpy.ops.mesh.minimal_knot_span_add(**kw)
    nx, comps = genuine_crossings(bpy.context.object)
    ok = nx == 0 and comps == 1 and len(bpy.data.objects) == 1
    print(f"[single sheet {label}] objects={len(bpy.data.objects)} "
          f"components={comps} genuine_crossings={nx} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f'single-sheet-{label}')

# and the legacy wound annulus must still be reachable (it is the one
# that self-crosses -- the same gate must detect that, proving the gate)
clear_objects()
bpy.ops.mesh.minimal_knot_span_add(span_topology='WOUND')
nx, comps = genuine_crossings(bpy.context.object)
ok = comps == 1 and nx > 0
print(f"[trefoil wound annulus] components={comps} "
      f"genuine_crossings={nx} (>0 expected) {'OK' if ok else 'FAIL'}")
if not ok:
    fails.append('trefoil-wound')

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
print("RENDERS ->", OUT)
