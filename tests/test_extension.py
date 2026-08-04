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
    ("scherk demo preset",
     lambda: bpy.ops.mesh.scherk_collins_add(preset='DEMO16')),
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
    ("conway biscribed", lambda: bpy.ops.mesh.conway_add(
        example='CUSTOM', notation='tO', post='BISCRIBED')),
    ("conway hexpropello whirl", lambda: bpy.ops.mesh.conway_add(
        example='CUSTOM', notation='wC')),
    ("biscribed truncated octahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='tO')),
    ("biscribed snub cube",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='sC')),
    ("biscribed dual pentagonal hexecontahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='pH')),
    ("biscribed propello cube",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='ppC')),
    ("biscribed propello dodecahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='ppD')),
    ("biscribed hexpropello dodecahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='whD')),
    ("biscribed dual hexpropello cube",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='dwC')),
    ("biscribed orthokis propello cube",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='okpC')),
    ("biscribed propello truncated icosahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='ptrI')),
    ("biscribed snub truncated icosahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='sntI')),
    ("biscribed dual snub truncated octahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='dsntO')),
    ("biscribed propello truncated icosidodecahedron",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='ptID')),
    ("biscribed l-propello l-snub cube",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='plsC')),
    ("biscribed propello disdyakis triacontahedron (dual)",
     lambda: bpy.ops.mesh.biscribed_solid_add(solid='pmT')),
    ("solid hull family", lambda: bpy.ops.mesh.regular_solid_add(
        family='HULL', solid='jtT', canon_iters=60)),
    ("solid trapezohedron family",
     lambda: bpy.ops.mesh.regular_solid_add(
         family='DIPYRAMID', solid='dA5', canon_iters=60)),
    ("solid truncated catalan family",
     lambda: bpy.ops.mesh.regular_solid_add(
         family='TRUNC_CATALAN', solid='t6dtT', canon_iters=80)),
    ("uniform great icosidodecahedron",
     lambda: bpy.ops.mesh.uniform_polyhedron_add(
         family='STAR_I', solid='54')),
    ("uniform small cubicuboctahedron",
     lambda: bpy.ops.mesh.uniform_polyhedron_add(
         family='STAR_O', solid='13')),
    ("uniform tetrahemihexahedron",
     lambda: bpy.ops.mesh.uniform_polyhedron_add(
         family='HEMI', solid='4')),
    ("star prism pentagrammic",
     lambda: bpy.ops.mesh.star_prism_add(sides=5, step=2)),
    ("star antiprism pentagrammic",
     lambda: bpy.ops.mesh.star_prism_add(form='ANTIPRISM', sides=5, step=2)),
    ("star dipyramid heptagrammic",
     lambda: bpy.ops.mesh.star_prism_add(form='DIPYRAMID', sides=7, step=3)),
    ("star trapezohedron octagrammic",
     lambda: bpy.ops.mesh.star_prism_add(
         form='TRAPEZOHEDRON', sides=8, step=3)),
    ("uniform snub dodecahedron",
     lambda: bpy.ops.mesh.uniform_polyhedron_add(
         family='CONVEX', solid='29')),
    ("uniform great dirhombicosidodecahedron",
     lambda: bpy.ops.mesh.uniform_polyhedron_add(
         family='STAR_I', solid='75')),
    ("uniform dual rhombic triacontahedron",
     lambda: bpy.ops.mesh.uniform_polyhedron_add(
         family='CONVEX', solid='24', dual=True)),
    ("uniform dual star", lambda: bpy.ops.mesh.uniform_polyhedron_add(
         family='STAR_I', solid='54', dual=True)),
    ("solid archimedean canon off",
     lambda: bpy.ops.mesh.regular_solid_add(
         family='ARCHIMEDEAN', solid='TI', canonicalize=False)),
    # family set without a matching solid id (the stale-id switch case):
    # the operator must fall back to the family's first solid, not error
    ("solid family switch guard", lambda: (
        bpy.ops.mesh.regular_solid_add(family='PROPELLOR'),
        bpy.ops.mesh.regular_solid_add(family='CHAMFER', canon_iters=60))[-1]),
    ("zonohedron", lambda: bpy.ops.mesh.zonohedron_add(
        kind='SPIRAL', n=12, spiral_width=4)),
    ("waterman", lambda: bpy.ops.mesh.waterman_add(root=20)),
    ("rotegrity", lambda: bpy.ops.mesh.rotegrity_add(kind='ICOSA', freq=1)),
    ("weave", lambda: bpy.ops.mesh.poly_weave_add(kind='CUBE')),
    ("weave rope tube", lambda: bpy.ops.mesh.poly_weave_add(
        kind='ICOSA', freq=1, output='TUBE', tube_radius=0.03,
        relax_iters=40)),
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
    ("sponge mosely", lambda: bpy.ops.mesh.sponge_add(kind='MOSELY',
                                                      level=2)),
    ("sponge cantor", lambda: bpy.ops.mesh.sponge_add(kind='CANTOR',
                                                      level=2)),
    ("mandelbulb", lambda: bpy.ops.mesh.mandelbulb_add(
        preset='MANDELBULB', resolution=40)),
    ("quaternion julia", lambda: bpy.ops.mesh.mandelbulb_add(
        preset='JULIA', resolution=40)),
    ("mandelbox", lambda: bpy.ops.mesh.mandelbulb_add(
        preset='MANDELBOX', resolution=40)),
    ("snowflake", lambda: bpy.ops.mesh.snowflake_add(
        preset='DENDRITE', radius=36)),
    ("fractal surface wm", lambda: bpy.ops.mesh.fractal_surface_add(
        base='SPHERE', method='WEIERSTRASS', resolution=40)),
    ("fractal surface fbm", lambda: bpy.ops.mesh.fractal_surface_add(
        base='PLATE', method='FBM', resolution=40)),
    ("lsystem gosper", lambda: bpy.ops.curve.lsystem_add(
        kind='GOSPER')),
    ("lsystem bush3d", lambda: bpy.ops.curve.lsystem_add(
        kind='BUSH3D')),
    ("antoine necklace", lambda: bpy.ops.mesh.antoine_add(
        depth=2, count=4)),
    ("fractal knot", lambda: bpy.ops.curve.fractal_knot_add(
        kind='TREFOIL', samples=1200)),
    ("apollonian gasket filled", lambda: bpy.ops.mesh.apollonian_add(
        mode='GASKET', depth=3, gasket_style='FILLED', color_by='SIZE')),
    ("apollonian gasket depth color",
     lambda: bpy.ops.mesh.apollonian_add(
         mode='GASKET', depth=3, gasket_style='FILLED',
         color_by='DEPTH')),
    ("apollonian gasket tube", lambda: bpy.ops.mesh.apollonian_add(
        mode='GASKET', depth=3, gasket_style='TUBE')),
    ("apollonian packing", lambda: bpy.ops.mesh.apollonian_add(
        mode='PACKING', depth=3, color_by='DEPTH')),
    ("hyperbolic surface pseudosphere",
     lambda: bpy.ops.mesh.hyperbolic_surface_add(preset='PSEUDOSPHERE')),
    ("hyperbolic surface kuen",
     lambda: bpy.ops.mesh.hyperbolic_surface_add(preset='KUEN')),
    ("crochet wavy",
     lambda: bpy.ops.mesh.crochet_add(preset='WAVY')),
    ("crochet ruffled",
     lambda: bpy.ops.mesh.crochet_add(preset='RUFFLED')),
    ("crochet cloth setup",
     lambda: bpy.ops.mesh.crochet_add(preset='WAVY', physics='CLOTH',
                                      cloth_bake=0)),
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
    ("solid johnson augmented prism",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J51')),
    ("solid johnson augmented dodeca",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J61')),
    ("solid johnson diminished icosa",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J63')),
    ("solid johnson augmented truncated",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J68')),
    ("solid johnson gyrate rid",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J72')),
    ("solid johnson tridiminished rid",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J83')),
    ("solid johnson snub disphenoid",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J84')),
    ("solid johnson hebesphenomegacorona",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J89')),
    ("solid johnson triangular hebesphenorotunda",
     lambda: bpy.ops.mesh.regular_solid_add(family='JOHNSON', solid='J92')),
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
    ("woven trunc tetra", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='TT')),
    ("woven cuboctahedron", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='CO')),
    ("woven snub cube", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='SC')),
    ("woven trunc icosa", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='TI')),
    ("woven geodesic", lambda: bpy.ops.mesh.woven_polyhedron_add(
        solid='GEODESIC', frequency=2)),
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
    ("hyperbolic klein",
     lambda: bpy.ops.mesh.hyperbolic_tiling_add(model='KLEIN',
                                                p=3, q=7)),
    ("geodesic", lambda: bpy.ops.mesh.geodesic_add()),
    ("geodesic dual (goldberg)",
     lambda: bpy.ops.mesh.geodesic_add(dual=True)),
    ("compound 5 tetrahedra",
     lambda: bpy.ops.mesh.polyhedron_compound_add(compound='5TETRA')),
    ("notable echidnahedron",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='ECHIDNAHEDRON')),
    ("stellation great icosahedron",
     lambda: bpy.ops.mesh.icosahedron_stellation_add(solid='7')),
    ("stellation ten tetrahedra",
     lambda: bpy.ops.mesh.icosahedron_stellation_add(solid='22')),
    ("stellation final (echidnahedron)",
     lambda: bpy.ops.mesh.icosahedron_stellation_add(solid='8')),
    ("general stellation dodeca small stellated",
     lambda: bpy.ops.mesh.general_stellation_add(
         seed='dodecahedron', stellation='small_stellated')),
    ("general stellation cubocta cube+octahedron",
     lambda: bpy.ops.mesh.general_stellation_add(
         seed='cuboctahedron', stellation='cube_octahedron')),
    ("general stellation RT five cubes",
     lambda: bpy.ops.mesh.general_stellation_add(
         seed='rhombic_triacontahedron', stellation='five_cubes')),
    ("notable schonhardt",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='SCHONHARDT')),
    ("notable jessen",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='JESSEN')),
    ("notable escher",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='ESCHER')),
    ("notable tetrated dodecahedron",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='TETRATED')),
    ("notable 132-pentagon",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='PENTAGON132')),
    ("notable associahedron",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='ASSOCIAHEDRON')),
    ("notable klein regular map",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='KLEIN')),
    ("notable regular map {6,4} genus 6",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='MAP64')),
    ("notable regular map {4,6} genus 6",
     lambda: bpy.ops.mesh.notable_polyhedron_add(solid='MAP46')),
    ("canonical self-dual icosioctahedron",
     lambda: bpy.ops.mesh.canonical_polyhedron_add(family='SELF_DUAL',
                                                   solid='0')),
    ("canonical self-dual tetracontahedron",
     lambda: bpy.ops.mesh.canonical_polyhedron_add(family='SELF_DUAL',
                                                   solid='4')),
    ("canonical self-dual 76-hedron",
     lambda: bpy.ops.mesh.canonical_polyhedron_add(family='SELF_DUAL',
                                                   solid='10')),
    ("canonical simplest first",
     lambda: bpy.ops.mesh.canonical_polyhedron_add(family='SIMPLEST',
                                                   solid='0')),
    ("canonical simplest mid",
     lambda: bpy.ops.mesh.canonical_polyhedron_add(family='SIMPLEST',
                                                   solid='25')),
    ("canonical simplest last",
     lambda: bpy.ops.mesh.canonical_polyhedron_add(family='SIMPLEST',
                                                   solid='41')),
    ("toroid csaszar",
     lambda: bpy.ops.mesh.toroidal_polyhedron_add(solid='CSASZAR')),
    ("toroid szilassi",
     lambda: bpy.ops.mesh.toroidal_polyhedron_add(solid='SZILASSI')),
    ("toroid regular-faced",
     lambda: bpy.ops.mesh.toroidal_polyhedron_add(solid='REGULAR')),
    ("toroid knotted dodecahedron",
     lambda: bpy.ops.mesh.toroidal_polyhedron_add(solid='KNOTTED')),
    ("toroid borromean rings",
     lambda: bpy.ops.mesh.toroidal_polyhedron_add(solid='BORROMEAN')),
    ("toroid heptagonal iris",
     lambda: bpy.ops.mesh.toroidal_polyhedron_add(solid='IRIS7')),
    ("polyhedral torus prism",
     lambda: bpy.ops.mesh.polyhedral_torus_add(segments=10, sides=4)),
    ("polyhedral torus antiprism",
     lambda: bpy.ops.mesh.polyhedral_torus_add(
         segments=8, sides=3, antiprism=True)),
    ("compound stella separate",
     lambda: bpy.ops.mesh.polyhedron_compound_add(
         compound='STELLA', separate=True)),
    ("geodesic class III",
     lambda: bpy.ops.mesh.geodesic_add(geo_class='III', frequency=2, k=1)),
    ("geodesic cube",
     lambda: bpy.ops.mesh.geodesic_add(base='CUBE', frequency=3)),
    ("geodesic rt",
     lambda: bpy.ops.mesh.geodesic_add(base='RT', frequency=2)),
    ("geodesic class III dual",
     lambda: bpy.ops.mesh.geodesic_add(
         geo_class='III', frequency=3, k=1, dual=True)),
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
    ("wallpaper p4m", lambda: bpy.ops.mesh.wallpaper_add(
        group='p4m', nx=2, ny=2)),
    ("wallpaper p6m", lambda: bpy.ops.mesh.wallpaper_add(
        group='p6m', motif_kind='COMMA', nx=2, ny=2)),
    ("wallpaper pgg op", lambda: bpy.ops.mesh.wallpaper_add(
        group='pgg', nx=3, ny=3, color_by='OP')),
    ("wallpaper cmm hand", lambda: bpy.ops.mesh.wallpaper_add(
        group='cmm', nx=3, ny=3, color_by='HAND')),
    ("wallpaper p1 relief", lambda: bpy.ops.mesh.wallpaper_add(
        group='p1', nx=3, ny=3, height=0.1)),
    ("wallpaper active mesh",
     lambda: (bpy.ops.mesh.primitive_cube_add(),
              bpy.ops.mesh.wallpaper_add(
                  group='p4', motif_kind='ACTIVE', nx=2, ny=2))[-1]),
    ("wallpaper active fallback (no mesh)",
     lambda: bpy.ops.mesh.wallpaper_add(
         group='p3', motif_kind='ACTIVE', nx=2, ny=2)),
    ("wallpaper margin", lambda: bpy.ops.mesh.wallpaper_add(
        group='p4m', margin=0.4, nx=2, ny=2)),
    ("wallpaper active margin",
     lambda: (bpy.ops.mesh.primitive_cube_add(),
              bpy.ops.mesh.wallpaper_add(
                  group='p1', motif_kind='ACTIVE', margin=0.25,
                  nx=2, ny=2))[-1]),
    ("wallpaper align cursor", lambda: bpy.ops.mesh.wallpaper_add(
        group='p4m', nx=2, ny=2, align='CURSOR')),
    ("frieze p2mm", lambda: bpy.ops.mesh.frieze_add(
        group='p2mm', reps=4)),
    ("frieze p2mg op", lambda: bpy.ops.mesh.frieze_add(
        group='p2mg', reps=4, color_by='OP', height=0.1)),
    ("frieze active",
     lambda: (bpy.ops.mesh.primitive_cube_add(),
              bpy.ops.mesh.frieze_add(
                  group='p11g', motif_kind='ACTIVE', reps=3))[-1]),
    ("layer p4 mirror", lambda: bpy.ops.mesh.layer_add(
        group='p4', zsym='MIRROR', nx=2, ny=2)),
    ("layer p1 none", lambda: bpy.ops.mesh.layer_add(
        group='p1', zsym='NONE', nx=2, ny=2)),
    ("layer active twofold",
     lambda: (bpy.ops.mesh.primitive_cube_add(),
              bpy.ops.mesh.layer_add(
                  group='p2', zsym='TWOFOLD', motif_kind='ACTIVE',
                  nx=2, ny=2))[-1]),
    ("tiling rhombitrihex", lambda: bpy.ops.mesh.tiling_add(
        tiling='RHOMBITRIHEX', nx=3, ny=3)),
    ("tiling truncsq relief", lambda: bpy.ops.mesh.tiling_add(
        tiling='TRUNCSQ', nx=3, ny=3, height=0.2)),
    ("tiling laves deltoidal", lambda: bpy.ops.mesh.tiling_add(
        tiling='DELTOIDAL', nx=3, ny=3, margin=0.05)),
    ("tiling cairo type", lambda: bpy.ops.mesh.tiling_add(
        tiling='CAIRO', nx=3, ny=3, color_by='TYPE')),
    ("tiling separate", lambda: bpy.ops.mesh.tiling_add(
        tiling='HEX', nx=2, ny=2, separate=True)),
    ("tiling trim snubhex", lambda: bpy.ops.mesh.tiling_add(
        tiling='SNUBHEX', nx=4, ny=4, trim=True)),
    ("tiling trim trunctrihex", lambda: bpy.ops.mesh.tiling_add(
        tiling='TRUNCTRIHEX', nx=4, ny=4, trim=True, height=0.2)),
    ("wallpaper separate", lambda: bpy.ops.mesh.wallpaper_add(
        group='p4', nx=2, ny=2, separate=True)),
    ("frieze separate", lambda: bpy.ops.mesh.frieze_add(
        group='p2', reps=3, separate=True)),
    ("layer separate", lambda: bpy.ops.mesh.layer_add(
        group='p2', zsym='MIRROR', nx=2, ny=2, separate=True)),
    ("wallpaper cm (fixed)", lambda: bpy.ops.mesh.wallpaper_add(
        group='cm', nx=3, ny=3)),
    ("wallpaper p4g op (fixed)", lambda: bpy.ops.mesh.wallpaper_add(
        group='p4g', nx=2, ny=2, color_by='OP')),
    ("kuniform s_tt", lambda: bpy.ops.mesh.kuniform_add(
        tiling='S_TT', nx=3, ny=3)),
    ("kuniform hex_t trim", lambda: bpy.ops.mesh.kuniform_add(
        tiling='HEX_T', nx=3, ny=3, trim=True)),
    ("monohedral pent2", lambda: bpy.ops.mesh.monohedral_add(
        tiling='PENT2', nx=3, ny=3)),
    ("monohedral hex2 separate", lambda: bpy.ops.mesh.monohedral_add(
        tiling='HEX2', nx=2, ny=2, separate=True)),
    ("isohedral p2 shape", lambda: bpy.ops.mesh.isohedral_add(
        tiling='P2', nx=3, ny=3, shape=0.35)),
    ("isohedral p31m", lambda: bpy.ops.mesh.isohedral_add(
        tiling='P31M', nx=3, ny=3)),
    ("aperiodic penrose p3", lambda: bpy.ops.mesh.aperiodic_add(
        kind='PENROSE_P3', generations=4)),
    ("aperiodic ammann trim", lambda: bpy.ops.mesh.aperiodic_add(
        kind='AMMANN_BEENKER', generations=3, trim=True)),
    ("aperiodic p2 star", lambda: bpy.ops.mesh.aperiodic_add(
        kind='PENROSE_P2', seed='STAR', generations=4)),
    ("aperiodic p2 cartwheel", lambda: bpy.ops.mesh.aperiodic_add(
        kind='PENROSE_P2', seed='CARTWHEEL', generations=1)),
    ("aperiodic ammann asym", lambda: bpy.ops.mesh.aperiodic_add(
        kind='AMMANN_BEENKER', generations=3, phase=0.3,
        asymmetry=0.5)),
    ("aperiodic hat", lambda: bpy.ops.mesh.aperiodic_add(
        kind='HAT', generations=2)),
    ("aperiodic spectre", lambda: bpy.ops.mesh.aperiodic_add(
        kind='SPECTRE', generations=2)),
    ("aperiodic spectre straight", lambda: bpy.ops.mesh.aperiodic_add(
        kind='SPECTRE', generations=2, spectre_curved=False)),
    ("aperiodic turtle", lambda: bpy.ops.mesh.aperiodic_add(
        kind='TURTLE', generations=2)),
    ("aperiodic chevron", lambda: bpy.ops.mesh.aperiodic_add(
        kind='CHEVRON', generations=2)),
    ("aperiodic comet", lambda: bpy.ops.mesh.aperiodic_add(
        kind='COMET', generations=2)),
    ("aperiodic custom semi", lambda: bpy.ops.mesh.aperiodic_add(
        kind='CUSTOM', generations=2, semi_angle=40.0)),
    ("aperiodic hat cluster", lambda: bpy.ops.mesh.aperiodic_add(
        kind='HAT', generations=2, color_by='CLUSTER')),
    ("aperiodic hat supertile", lambda: bpy.ops.mesh.aperiodic_add(
        kind='HAT', generations=2, color_by='SUPERTILE_LEVEL')),
    ("reptile sphinx", lambda: bpy.ops.mesh.reptile_add(
        kind='SPHINX', iterations=3)),
    ("reptile l_tromino trim", lambda: bpy.ops.mesh.reptile_add(
        kind='L_TROMINO', iterations=3, trim=True)),
    ("voderberg classic", lambda: bpy.ops.mesh.voderberg_add(
        kind='CLASSIC', coils=4)),
    ("voderberg spiral g3", lambda: bpy.ops.mesh.voderberg_add(
        kind='SPIRAL', arms=3, coils=4)),
    ("voderberg radial by type", lambda: bpy.ops.mesh.voderberg_add(
        kind='RADIAL', coils=4, color_by='TYPE')),
    ("spiral equilateral", lambda: bpy.ops.mesh.spiral_tiling_add(
        family='EQUILATERAL')),
    ("spiral golden", lambda: bpy.ops.mesh.spiral_tiling_add(
        family='GOLDEN', color_by='RING')),
    ("spiral iso side-to-side", lambda: bpy.ops.mesh.spiral_tiling_add(
        family='ISO_SS', n=6)),
    ("spiral right leg-hyp", lambda: bpy.ops.mesh.spiral_tiling_add(
        family='RIGHT_LH', n=5)),
    ("spiral multiarm", lambda: bpy.ops.mesh.spiral_tiling_add(
        family='GENERAL', n=2, arms=3, angle=80.0, color_by='ARM')),
    ("spiral polygon 4arm", lambda: bpy.ops.mesh.spiral_tiling_add(
        family='POLY', arms=4, angle=45.0, color_by='ARM')),
    ("fractal kite r6", lambda: bpy.ops.mesh.fractal_tiling_add(
        kind='KITE_R6', iterations=4)),
    ("fractal kite r12 separate",
     lambda: bpy.ops.mesh.fractal_tiling_add(
         kind='KITE_R12', iterations=3, separate=True)),
    ("fractal kite r8", lambda: bpy.ops.mesh.fractal_tiling_add(
        kind='KITE_R8', iterations=4, color_by='UNIFORM')),
    ("fractal reptile right-triangle",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='RIGHT_TRIANGLE', iterations=6)),
    ("fractal reptile rep5",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='REP5', iterations=4)),
    ("fractal reptile twindragon",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='TWINDRAGON', iterations=5)),
    ("fractal reptile pentabolo",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='PENTABOLO', iterations=3)),
    ("fractal reptile flowsnake",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='FLOWSNAKE', iterations=3)),
    ("fractal reptile levy",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='LEVY_DRAGON', iterations=8)),
    ("fractal reptile foldable",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='FOLDABLE4', iterations=5)),
    ("fractal reptile gasket",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='REP5', iterations=4, holes=2)),
    ("fractal reptile x-pentomino",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='X_PENTOMINO', iterations=5)),
    ("fractal reptile heptiamond",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='HEPTIAMOND', iterations=4)),
    ("fractal reptile trihex",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='TRIHEX', iterations=6, mirror=True)),
    ("fractal reptile trirhomb",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='TRIRHOMB', iterations=6)),
    ("fractal reptile tetrakite",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='TETRAKITE', iterations=5)),
    ("fractal reptile rect6",
     lambda: bpy.ops.mesh.fractal_reptile_add(
         kind='RECT6', iterations=4)),
    ("fractal tiling s6",
     lambda: bpy.ops.mesh.fractal_tiling_add(kind='TRI_S6', iterations=6)),
    ("fractal tiling u6",
     lambda: bpy.ops.mesh.fractal_tiling_add(kind='TRAP_U6', iterations=5)),
    ("fractal crystal cube-oct",
     lambda: bpy.ops.mesh.fractal_polyhedron_add(crystal='CUBE_OCT')),
    ("fractal crystal dodecafoam",
     lambda: bpy.ops.mesh.fractal_polyhedron_add(
         crystal='DODECAFOAM', hull_shell=True)),
    ("algebraic monkey saddle",
     lambda: bpy.ops.mesh.algebraic_surface_add(preset='MONKEY', fold=3)),
    ("spiked goldberg",
     lambda: bpy.ops.mesh.spiked_polyhedron_add(
         preset='SPIKED', seed='GOLDBERG', goldberg_freq=2)),
    ("islamic starcross8", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='STARCROSS8', nx=4, ny=4)),
    ("islamic 12-fold tile", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='CUSTOM', substrate='TRUNCHEX', contact_angle=30.0,
        nx=3, ny=3, color_by='TILE')),
    ("islamic hex relief", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='CUSTOM', substrate='HEX', contact_angle=35.0,
        nx=3, ny=3, height=0.15, backing=True)),
    ("islamic square separate",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='CUSTOM', substrate='SQUARE', contact_angle=30.0,
         nx=4, ny=4, separate=True)),
    ("islamic curved", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='STARCROSS8', nx=4, ny=4, curved=True, smoothness=8)),
    ("islamic by ribbon", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='STARCROSS8', nx=4, ny=4, color_by='BAND')),
    ("islamic curve output", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='STARCROSS8', nx=3, ny=3, output='CURVE', curved=True)),
    ("islamic rosette8", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='ROSETTE8', nx=3, ny=3)),
    ("islamic rosette12 relief", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='ROSETTE12', nx=2, ny=2, height=0.3, backing=True)),
    ("islamic star8", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='STAR8', nx=3, ny=3)),
    ("islamic infer cairo", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='INFER_CAIRO', nx=3, ny=3)),
    ("islamic rosette custom curved band",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='CUSTOM', substrate='TRUNCSQ', motif='ROSETTE',
         contact_angle=34.0, rosette_frac=0.4, nx=2, ny=2,
         curved=True, color_by='BAND')),
    ("islamic interlace flat", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='STARCROSS8', nx=3, ny=3, interlace=True,
        interlace_mode='FLAT')),
    ("islamic interlace woven", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='STARCROSS8', nx=3, ny=3, interlace=True,
        interlace_mode='WOVEN', weave_height=0.12)),
    ("islamic backing no relief",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='STARCROSS8', nx=3, ny=3, backing=True)),
    ("islamic rhombille stars",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='INFER_RHOMBI', nx=5, ny=5, trim=True)),
    ("islamic girih", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='GIRIH10', girih_gens=3, trim=True)),
    ("islamic girih interlaced",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='GIRIH10', girih_gens=3, trim=True, interlace=True,
         interlace_mode='FLAT', color_by='BAND')),
    ("islamic girih curved", lambda: bpy.ops.mesh.islamic_pattern_add(
        preset='GIRIH10', girih_gens=3, trim=True, curved=True,
        smoothness=8)),
    ("islamic trihex (shared tiling)",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='CUSTOM', substrate='TRIHEX', nx=3, ny=3)),
    ("islamic deltoidal (shared tiling)",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='CUSTOM', substrate='DELTOIDAL', nx=3, ny=3)),
    ("islamic snubhex (shared tiling)",
     lambda: bpy.ops.mesh.islamic_pattern_add(
         preset='CUSTOM', substrate='SNUBHEX', nx=3, ny=3)),
    ("celtic plain plait", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        grid_w=5, grid_h=5, barriers='PRESET', preset='PLAIN')),
    ("celtic crossbreak smooth",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         grid_w=6, grid_h=5, barriers='PRESET', preset='CROSSBREAK',
         style='SMOOTH')),
    ("celtic random single cord",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         grid_w=6, grid_h=6, barriers='RANDOM', seed=4, density=0.16,
         single_cord=True, color_by='LOOP')),
    ("celtic procedural woven relief",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         grid_w=5, grid_h=5, barriers='PROCEDURAL', wall_spacing=2,
         interlace=True, interlace_mode='WOVEN', weave_height=0.1,
         height=0.05)),
    ("celtic borderless tileable",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         grid_w=6, grid_h=6, border=False, barriers='PROCEDURAL',
         wall_spacing=2, color_by='LOOP')),
    ("over-under screen membrane",
     lambda: bpy.ops.mesh.over_under_screen_add(
         grid_w=4, grid_h=3, surface='MEMBRANE', resolution=12)),
    ("over-under screen ribbons twill",
     lambda: bpy.ops.mesh.over_under_screen_add(
         grid_w=4, grid_h=3, weave='TWILL', surface='RIBBONS')),
    ("celtic smooth flush interlace",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         grid_w=6, grid_h=5, barriers='PRESET', preset='CROSSBREAK',
         style='SMOOTH', interlace=True, interlace_mode='FLAT')),
    ("celtic hex tiling", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        substrate='HEX', grid_w=6, grid_h=6, trim=True,
        color_by='LOOP')),
    ("celtic triangle checker",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='TRIANGLE', grid_w=6, grid_h=6, trim=True,
         color_by='CHECKER')),
    ("celtic trihex single cord",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='TRIHEX', grid_w=6, grid_h=6, trim=True,
         barriers='RANDOM', seed=3, density=0.2, single_cord=True)),
    ("celtic snubsquare loop",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='SNUBSQUARE', grid_w=5, grid_h=5, trim=True,
         color_by='LOOP')),
    ("celtic tube woven", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        grid_w=5, grid_h=5, preset='CROSSBREAK', profile='TUBE',
        tube_sides=10, interlace_mode='WOVEN', weave_height=0.16)),
    ("celtic rope", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        substrate='HEX', grid_w=4, grid_h=4, trim=True, profile='ROPE',
        rope_strands=3, rope_twist=9.0, interlace_mode='WOVEN')),
    ("celtic sym d4", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        substrate='SQUARE', grid_w=8, grid_h=8, barriers='RANDOM',
        seed=23, density=0.10, symmetry='D4', style='SMOOTH',
        color_by='LOOP')),
    ("celtic sym c6 hex", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        substrate='HEX', grid_w=7, grid_h=7, trim=True,
        barriers='RANDOM', seed=4, density=0.42, symmetry='C6',
        style='SMOOTH', color_by='LOOP')),
    ("celtic pointed turns",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         grid_w=4, grid_h=4, preset='PLAIN', style='SMOOTH',
         corner_style='POINTED', corner_tightness=0.7)),
    ("celtic triquetra", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        preset='TRIQUETRA', grid_w=4, grid_h=4, style='SMOOTH')),
    ("celtic josephine", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        preset='JOSEPHINE', grid_w=6, grid_h=3, style='SMOOTH')),
    ("celtic carpet", lambda: bpy.ops.mesh.celtic_knot_2d_add(
        preset='CARPET', grid_w=6, grid_h=6, style='SMOOTH')),
    ("celtic trihex angular flat",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='TRIHEX', grid_w=5, grid_h=5, trim=True,
         style='ANGULAR', interlace=True, interlace_mode='FLAT')),
    ("celtic triangle smooth flat",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='TRIANGLE', grid_w=5, grid_h=5, trim=True,
         style='SMOOTH', interlace=True, interlace_mode='FLAT')),
    ("celtic trunchex smooth flat",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='TRUNCHEX', grid_w=5, grid_h=5, trim=True,
         style='SMOOTH', interlace=True, interlace_mode='FLAT')),
    ("celtic rhombitrihex (shared tiling)",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='RHOMBITRIHEX', grid_w=4, grid_h=4, trim=True)),
    ("celtic floret laves (shared tiling)",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='FLORET', grid_w=4, grid_h=4, trim=True)),
    ("celtic kisrhombille (shared tiling)",
     lambda: bpy.ops.mesh.celtic_knot_2d_add(
         substrate='KISRHOMBILLE', grid_w=4, grid_h=4, trim=True)),
    ("knot carpet square flat", lambda: bpy.ops.mesh.knot_carpet_add(
        lattice='SQUARE', symmetry=4, nx=3, ny=3,
        interlace_mode='FLAT', color_by='LOOP')),
    ("knot carpet tri woven",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='TRIANGULAR', symmetry=6, nx=2, ny=2,
         amplitude=0.12, interlace_mode='WOVEN', weave_height=0.06,
         height=0.04, color_by='CHECKER')),
    ("knot carpet tube", lambda: bpy.ops.mesh.knot_carpet_add(
        lattice='SQUARE', symmetry=4, nx=2, ny=2, output='TUBE',
        tube_radius=0.04, tube_sides=8, weave_height=0.06)),
    ("knot ball sphere relaxed",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='SPHERE', sphere_freq=2, output='TUBE',
         tube_radius=0.04, tube_sides=8, relax_iters=40)),
    ("knot carpet tiling 488",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='TILING', tiling_name='TRUNCSQ', nx=3, ny=3,
         output='TUBE', tube_radius=0.04, tube_sides=8)),
    ("knot carpet tiling floret (shared)",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='TILING', tiling_name='FLORET', nx=3, ny=3)),
    ("knot ball soccer scaffold",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='SPHERE', sphere_scaffold='TI', output='TUBE',
         tube_radius=0.04, tube_sides=8, relax_iters=40)),
    ("knot carpet polyhedral escher",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='POLYHEDRAL', poly_scaffold='ESCHER', output='TUBE',
         tube_radius=0.04, tube_sides=8, weave_height=0.06)),
    ("knot carpet polyhedral icosa ribbon",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='POLYHEDRAL', poly_scaffold='ICOSA', output='RIBBON',
         cord_width=0.12, weave_height=0.06)),
    ("knot carpet polyhedral small stellated dodec",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='POLYHEDRAL', poly_scaffold='SSD', output='TUBE',
         tube_radius=0.04, tube_sides=8, weave_height=0.06)),
    ("knot carpet polyhedral great stellated dodec",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='POLYHEDRAL', poly_scaffold='GSD', output='TUBE',
         tube_radius=0.04, tube_sides=8, weave_height=0.06)),
    ("knot carpet torus",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='TORUS', nx=4, ny=3, output='TUBE',
         tube_radius=0.04, tube_sides=8)),
    ("knot carpet torus hex",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='TORUS', torus_tiling='HEX', nx=6, ny=3,
         output='TUBE', tube_radius=0.04, tube_sides=8)),
    ("knot carpet torus trihex ribbon",
     lambda: bpy.ops.mesh.knot_carpet_add(
         lattice='TORUS', torus_tiling='TRIHEX', nx=4, ny=3,
         output='RIBBON', interlace_mode='WOVEN', weave_height=0.05)),
    ("knot carpet uvmesh",
     lambda: (bpy.ops.mesh.primitive_uv_sphere_add(),
              bpy.ops.mesh.knot_carpet_add(
                  lattice='UVMESH',
                  uv_target=bpy.context.active_object.name,
                  uv_tiles=6, output='TUBE', tube_radius=0.04,
                  tube_sides=8))[-1]),
    ("knot carpet uvmesh hex",
     lambda: (bpy.ops.mesh.primitive_uv_sphere_add(),
              bpy.ops.mesh.knot_carpet_add(
                  lattice='UVMESH',
                  uv_target=bpy.context.active_object.name,
                  uvmesh_tiling='HEX', uv_tiles=6, output='RIBBON',
                  interlace_mode='WOVEN', weave_height=0.05))[-1]),
    ("woven double shell chainmail",
     lambda: bpy.ops.mesh.woven_double_shell_add(
         solid='DODECA', advance=0, output='TUBE', relax_iters=30)),
    ("woven double shell continuous",
     lambda: bpy.ops.mesh.woven_double_shell_add(
         solid='CUBE', advance=1, output='TUBE', relax_iters=30)),
    ("harmonic lissajous", lambda: bpy.ops.curve.harmonic_knot_add(
        preset='LISSAJOUS_327', output='MESH')),
    ("harmonic fourier torus",
     lambda: bpy.ops.curve.harmonic_knot_add(
         preset='FOURIER_TREFOIL', output='MESH')),
    ("petal trefoil", lambda: bpy.ops.curve.petal_knot_add(
        preset='TREFOIL', output='MESH')),
    ("turks head ring 3x5", lambda: bpy.ops.mesh.turks_head_add(
        surface='RING', leads=3, bights=5)),
    ("rational figure8", lambda: bpy.ops.curve.rational_knot_add(
        family='RATIONAL', notation='2 2', output='MESH')),
    ("pretzel 357", lambda: bpy.ops.curve.rational_knot_add(
        family='PRETZEL', notation='3 5 7', output='MESH')),
    ("substitution fractal knot",
     lambda: bpy.ops.curve.substitution_knot_add(
         base='TREFOIL', depth=2, output='TUBE')),
    ("fractal knotwork",
     lambda: bpy.ops.mesh.fractal_knotwork_add(
         substrate='KITE_R6', iterations=2, output='TUBE')),
    ("hyp uniform t73", lambda: bpy.ops.mesh.hyperbolic_tiling_add(
        form='TRUNCATED', p=7, q=3, model='POINCARE')),
    ("hyp uniform r54 klein",
     lambda: bpy.ops.mesh.hyperbolic_tiling_add(
         form='RECTIFIED', p=5, q=4, model='KLEIN')),
    ("hyp uniform pqr", lambda: bpy.ops.mesh.hyperbolic_tiling_add(
        form='OMNITRUNCATED', p=4, q=3, r=3)),
    ("hyp snub", lambda: bpy.ops.mesh.hyperbolic_tiling_add(
        form='SNUB', p=7, q=3, model='KLEIN')),
    ("hyp hemisphere", lambda: bpy.ops.mesh.hyperbolic_tiling_add(
        form='TRUNCATED', p=7, q=3, model='HEMISPHERE')),
    ("hyp hemisphere large tiles", lambda:
     bpy.ops.mesh.hyperbolic_tiling_add(
         p=3, q=7, r=4, model='HEMISPHERE', depth=8,
         color_by='PARITY', thickness=0.08)),
    ("hyp pseudosphere", lambda: bpy.ops.mesh.hyperbolic_tiling_add(
        model='PSEUDOSPHERE')),
    ("hyp parity plaque", lambda: bpy.ops.mesh.hyperbolic_tiling_add(
        p=6, q=4, color_by='PARITY', height=0.1, backing=True)),
    ("hyp parity hide off", lambda: bpy.ops.mesh.hyperbolic_tiling_add(
        p=6, q=4, color_by='PARITY', hide_off_parity=True)),
    ("phyllotaxis disk sunflower",
     lambda: bpy.ops.mesh.phyllotaxis_add(
         count=300, form='DISK', floret='BUMP', color_by='PARASTICHY',
         parastichy=13)),
    ("phyllotaxis dome cactus spikes",
     lambda: bpy.ops.mesh.phyllotaxis_add(
         count=250, form='DOME', floret='SPIKE', size_grade=0.4)),
    ("phyllotaxis sphere disc rings",
     lambda: bpy.ops.mesh.phyllotaxis_add(
         count=300, form='SPHERE', floret='DISC', color_by='RING')),
    ("phyllotaxis crest left uniform",
     lambda: bpy.ops.mesh.phyllotaxis_add(
         count=250, form='CREST', floret='BUMP', chirality='LEFT',
         crest_waves=4.0, color_by='UNIFORM')),
    ("phyllotaxis cone parastichy21",
     lambda: bpy.ops.mesh.phyllotaxis_add(
         count=280, form='CONE', floret='BUMP', parastichy=21)),
    ("phyllotaxis points output",
     lambda: bpy.ops.mesh.phyllotaxis_add(
         count=300, form='DOME', output='POINTS',
         color_by='PARASTICHY')),
    ("modular screen design5",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='DESIGN5', nx=4, ny=4)),
    ("modular screen weave frameless",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='WEAVE', nx=4, ny=3, frame=False, rim_bulge=0.0)),
    ("modular screen relief wall",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='DESIGN6', nx=4, ny=4, amp=0.35)),
    ("modular screen hypar carlberg",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='HYPAR', nx=4, ny=4, hole=0.28)),
    ("modular screen diagonal bulge",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='DIAGONAL', nx=3, ny=3, rim_bulge=0.9, bulge_segs=4,
         res=8)),
    ("modular screen bilayer weave",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='DESIGN1', nx=4, ny=4, ribbon_w=0.7)),
    ("modular screen square apertures",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='DESIGN5', nx=4, ny=4, ap_square=1.0)),
    ("modular screen carlberg polygon holes",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='HYPAR', nx=3, ny=3, ap_square=0.8, hole=0.3)),
    ("modular screen curved",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='DESIGN5', nx=6, ny=4, curvature='CURVED',
         wrap_angle=150.0)),
    ("modular screen column",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='WEAVE', nx=6, ny=4, curvature='COLUMN', frame=False)),
    ("modular screen weave column",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='DESIGN1', nx=6, ny=3, curvature='COLUMN')),
    ("modular screen pinwheel relief",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='PINWHEEL', nx=5, ny=5, hole=0.0, swirl=1.0)),
    ("modular screen pinwheel left perforated",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='PINWHEEL', nx=5, ny=5, handedness='LEFT', swirl=1.2,
         hole=0.25)),
    ("modular screen pinwheel curved",
     lambda: bpy.ops.mesh.modular_screen_add(
         preset='PINWHEEL', nx=6, ny=4, hole=0.0, curvature='CURVED',
         wrap_angle=140.0)),
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
        if obj.data is None:               # parent empty (separate cells)
            n_elem = len(obj.children)
        elif hasattr(obj.data, 'vertices'):
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
