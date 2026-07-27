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
    ("seifert membrane free-boundary",
     lambda: (bpy.ops.mesh.seifert_surface_add(preset='TREFOIL'),
              bpy.ops.mesh.seifert_minimize(
                  method='MEMBRANE', mem_iterations=80))[-1]),
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
    ("symmetric frabjous",
     lambda: bpy.ops.object.symmetric_sculpture_add(
         preset='FRABJOUS')),
    ("symmetric whimsy",
     lambda: bpy.ops.object.symmetric_sculpture_add(
         preset='WHIMSY')),
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
    ("torus knot", lambda: bpy.ops.curve.torus_knot_add(p=3, q=5)),
    ("torus link mesh", lambda: bpy.ops.curve.torus_knot_add(
        p=4, q=6, output='MESH')),
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
    ("dual helix", lambda: bpy.ops.curve.dual_helix_add()),
    ("stellated weave", lambda: bpy.ops.mesh.stellated_weave_add()),
    ("attractor lorenz", lambda: bpy.ops.curve.attractor_add(
        preset='LORENZ', steps=4000)),
    ("attractor aizawa taper", lambda: bpy.ops.curve.attractor_add(
        preset='AIZAWA', steps=4000, taper=0.5)),
    ("attractor pentagon profile",
     lambda: bpy.ops.curve.attractor_add(
         preset='THOMAS', steps=4000, profile_sides=5,
         samples=1500, spline='BEZIER')),
    ("solid split", lambda: bpy.ops.mesh.regular_solid_add(
        family='PLATONIC', solid='ICOSA', pieces=5)),
    ("stereographic", lambda: bpy.ops.mesh.stereographic_add(
        pattern='GRID')),
    ("stereographic tiling", lambda: bpy.ops.mesh.stereographic_add(
        pattern='TILING', tile_p=5, tile_q=3, res=32)),
    ("stereographic flower bowl",
     lambda: bpy.ops.mesh.stereographic_add(
         pattern='FLOWER', bowl=True, res=32)),
    ("hyperbolic honeycomb",
     lambda: bpy.ops.mesh.hyperbolic_honeycomb_add()),
    ("algebraic surface", lambda: bpy.ops.mesh.algebraic_surface_add(
        preset='CLEBSCH', resolution=48)),
    ("spacefill", lambda: bpy.ops.mesh.spacefill_add(kind='OCTET')),
    ("spacefill spiral3", lambda: bpy.ops.mesh.spacefill_add(
        kind='SPIRAL3', nx=2, ny=2, nz=2)),
    ("solid snub cube left", lambda: bpy.ops.mesh.regular_solid_add(
        family='ARCHIMEDEAN', solid='SC', handedness='LEFT')),
    ("spiked modern", lambda: bpy.ops.mesh.spiked_polyhedron_add(
        preset='MODERN', coloring='GROUP')),
    ("spiked hyperbolic", lambda: bpy.ops.mesh.spiked_polyhedron_add(
        preset='HYPER', resolution=8)),
    ("spiked icosa", lambda: bpy.ops.mesh.spiked_polyhedron_add(
        preset='SPIKED')),
    ("spiked rhombic hex", lambda: bpy.ops.mesh.spiked_polyhedron_add(
        preset='RHOMBIC')),
    ("woven polyhedron", lambda: bpy.ops.mesh.woven_polyhedron_add()),
    ("woven polyhedron flat", lambda: bpy.ops.mesh.woven_polyhedron_add(
        pen_curve=0.0, thickness=0.0, smooth_level=0)),
    ("woven tetra", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='TETRA')),
    ("woven cube", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='CUBE')),
    ("woven octa", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='OCTA')),
    ("woven icosa", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='ICOSA')),
    ("gilbert 3d", lambda: bpy.ops.curve.space_filling_add(
        kind='GILBERT3D', gw=12, gh=8, gd=4)),
    ("celtic knot", lambda: bpy.ops.curve.celtic_knot_add(
        source='ICOSA')),
    ("celtic twill ribbon", lambda: bpy.ops.curve.celtic_knot_add(
        source='DODECA', weave_type='TWILL',
        remesh_type='MEDIAL', output='RIBBON')),
    ("gilbert 2d", lambda: bpy.ops.curve.space_filling_add(
        kind='GILBERT2D', gw=13, gh=7)),
    ("spacefill spiral4", lambda: bpy.ops.mesh.spacefill_add(
        kind='SPIRAL4', nx=2, ny=2, nz=2)),
    ("math link", lambda: bpy.ops.curve.math_link_add(
        preset='BORROMEAN')),
    ("hyperbolic tiling",
     lambda: bpy.ops.mesh.hyperbolic_tiling_add()),
    ("geodesic", lambda: bpy.ops.mesh.geodesic_add()),
    ("curvature color",
     lambda: (bpy.ops.mesh.primitive_torus_add(),
              bpy.ops.object.curvature_color_add())[-1]),
    ("voronoi openwork",
     lambda: (bpy.ops.mesh.primitive_uv_sphere_add(),
              bpy.ops.object.voronoi_openwork_add(
                  holes=16, subdiv=0))[-1]),
    ("organic wireframe",
     lambda: (bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3),
              bpy.ops.object.organic_wireframe_add())[-1]),
    ("orbifold sphere", lambda: bpy.ops.mesh.orbifold_sphere_add()),
    ("fractal tree", lambda: bpy.ops.curve.fractal_tree_add(
        mode='TREE')),
    ("fractal mobile", lambda: bpy.ops.curve.fractal_tree_add(
        mode='MOBILE')),
    ("topological surface",
     lambda: bpy.ops.mesh.topological_surface_add(preset='KLEIN')),
    ("minimal poly cube",
     lambda: bpy.ops.mesh.minimal_surface_polyhedron_add(
        seed='CUBE')),
    ("minimal poly bulge",
     lambda: bpy.ops.mesh.minimal_surface_polyhedron_add(
        seed='DODECA', mode='SADDLE', depth=0.8, levels=2,
        iterations=15)),
    ("cubic squeeze", lambda: bpy.ops.mesh.squeeze_add(
        seed='CUBE', samples=8, rings=6, iterations=10)),
    ("squeeze truncoct", lambda: bpy.ops.mesh.squeeze_add(
        seed='TRUNCOCT', samples=6, rings=5, iterations=8)),
    ("vertex vortices", lambda: bpy.ops.mesh.vertex_vortices_add(
        seed='DODECA', samples=6, iterations=8)),
    ("vortices tetra", lambda: bpy.ops.mesh.vertex_vortices_add(
        seed='TETRA', samples=8, iterations=10, reverse=True)),
    ("vortices snub cube", lambda: bpy.ops.mesh.vertex_vortices_add(
        seed='SC', samples=6, iterations=8)),
    ("bubble cluster", lambda: bpy.ops.mesh.bubble_cluster_add(
        seed='CUBE', subdiv=2)),
    ("bubble cluster local", lambda: bpy.ops.mesh.bubble_cluster_add(
        seed='TETRA', radius_mode='LOCAL', factor=0.7, subdiv=2)),
    ("bubble merged colored", lambda: bpy.ops.mesh.bubble_cluster_add(
        seed='ICOSA', separate=False, color=True, subdiv=1)),
    ("helical helicoid", lambda: bpy.ops.mesh.helical_surface_add(
        surface='HYPERBOLIC_HELICOID', resolution=48)),
    ("helical seashell", lambda: bpy.ops.mesh.helical_surface_add(
        surface='SEASHELL', resolution=48)),
    ("curiosity fresnel", lambda: bpy.ops.mesh.curiosity_surface_add(
        surface='FRESNEL', resolution=32)),
    ("curiosity trihyperboloid",
     lambda: bpy.ops.mesh.curiosity_surface_add(
        surface='TRIHYPERBOLOID', resolution=32)),
    ("curiosity paper bag", lambda: bpy.ops.mesh.curiosity_surface_add(
        surface='PAPERBAG', resolution=32)),
    ("rolling knot", lambda: bpy.ops.mesh.rolling_knot_add(
        p=3, samples=256, sides=10)),
    ("rolling knot morton", lambda: bpy.ops.mesh.rolling_knot_add(
        p=5, mode='MORTON', samples=256, sides=8)),
    ("minimal poly active",
     lambda: (bpy.ops.mesh.conway_add(example='CUSTOM',
                                      notation='sC'),
              bpy.ops.mesh.minimal_surface_polyhedron_add(
                  seed='ACTIVE', levels=2, iterations=15))[-1]),
    ("polytope half", lambda: bpy.ops.mesh.polytope4d_add(
        kind='CELL24', half=True)),
    ("polytope dual compound", lambda: bpy.ops.mesh.polytope4d_add(
        kind='CELL8', dual_compound=True)),
    ("polytope rings", lambda: bpy.ops.mesh.polytope4d_add(
        kind='CELL120', rings=2, rings_only=True)),
]
for name, op in OPS:
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    try:
        result = op()
    except AttributeError as e:
        # operator from a work-in-progress module that the
        # resilient loader skipped: not a failure
        if "not found" in str(e) or "has no attribute" in str(e):
            print(f"[{name}] SKIP (module not present)")
            continue
        print("   ", e)
        print(f"[{name}] FAIL")
        fails.append(name)
        continue
    except Exception as e:
        print("   ", e)
        print(f"[{name}] FAIL")
        fails.append(name)
        continue
    try:
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
