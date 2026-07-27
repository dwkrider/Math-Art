"""Regenerate the documentation renders.

Every generator is shot with one consistent studio rig -- a black-velvet
dome, soft key/fill/rim/top lights, a white-plastic material on any
object the generator left uncolored (objects that carry their own
materials keep their colors), Cycles + AgX.  Because every generator now
outputs centered on the origin and fitting a 2 m cube, a single camera
and lighting setup frames them all.

Run (renders everything to docs/images/<slug>.png):

    blender --background --factory-startup --python docs/render_docs.py

Render just some (by slug):

    blender --background --factory-startup --python docs/render_docs.py -- twisted_polyhedron prime_knot

Slugs are the keys of TASKS below (they match the doc page file names).
Tune SAMPLES / RES for quality vs speed.  Add a generator by adding one
line to TASKS.
"""
import bpy
import bmesh
import sys
import os
import math
from mathutils import Vector, Euler

SAMPLES = 96
RES = 720

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import math_art  # noqa: E402
math_art.register()
IMG = os.path.join(PROJ, "docs", "images")
os.makedirs(IMG, exist_ok=True)

STUDIO = {"Backdrop Dome", "Key Light", "Fill Light", "Rim Light L",
          "Rim Light R", "Top Light", "Studio Camera"}


# ---------------------------------------------------------------- studio
def setup_studio():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_uv_sphere_add(radius=30, segments=96,
                                          ring_count=48)
    dome = bpy.context.object
    dome.name = "Backdrop Dome"
    bm = bmesh.new()
    bm.from_mesh(dome.data)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(dome.data)
    bm.free()
    dome.data.polygons.foreach_set("use_smooth",
                                   [True] * len(dome.data.polygons))
    velvet = bpy.data.materials.new("Black Velvet")
    velvet.use_nodes = True
    vb = velvet.node_tree.nodes.get("Principled BSDF")
    vb.inputs["Base Color"].default_value = (0.006, 0.006, 0.008, 1)
    vb.inputs["Roughness"].default_value = 1.0
    for k, v in (("Sheen Weight", 0.6), ("Sheen Roughness", 0.35)):
        if k in vb.inputs:
            vb.inputs[k].default_value = v
    dome.data.materials.append(velvet)
    rr = 1.55

    def area(name, off, energy, size):
        la = bpy.data.lights.new(name, 'AREA')
        la.energy = energy
        la.size = size
        o = bpy.data.objects.new(name, la)
        o.location = Vector(off) * rr
        o.rotation_euler = (-o.location).to_track_quat('-Z', 'Y').to_euler()
        o.visible_camera = False
        scene.collection.objects.link(o)
    area("Key Light", (1.8, -1.9, 1.6), 320 * rr * rr, rr * 3.0)
    area("Fill Light", (-2.4, -0.8, 0.5), 70 * rr * rr, rr * 5.0)
    area("Rim Light L", (-1.7, 1.9, 1.1), 750 * rr * rr, rr * 1.0)
    area("Rim Light R", (1.9, 1.7, 0.8), 750 * rr * rr, rr * 1.0)
    area("Top Light", (0.0, 0.3, 2.6), 150 * rr * rr, rr * 2.4)
    world = scene.world or bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"] \
        .default_value = (0.004, 0.004, 0.005, 1)
    cam_d = bpy.data.cameras.new("Studio Camera")
    cam_d.lens = 72
    cam = bpy.data.objects.new("Studio Camera", cam_d)
    scene.collection.objects.link(cam)
    view = Vector((1.35, -2.2, 0.95)).normalized()
    cam.location = view * (rr * 5.0)
    cam.rotation_euler = (-cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.film_transparent = False
    vt = [v.name for v in bpy.types.ColorManagedViewSettings.bl_rna
          .properties["view_transform"].enum_items]
    if "AgX" in vt:
        scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.5


_PLASTIC = [None]


def apply_material():
    if _PLASTIC[0] is None:
        m = bpy.data.materials.new("White Plastic")
        m.use_nodes = True
        sb = m.node_tree.nodes.get("Principled BSDF")
        sb.inputs["Base Color"].default_value = (0.84, 0.84, 0.86, 1)
        sb.inputs["Roughness"].default_value = 0.38
        _PLASTIC[0] = m
    for o in bpy.data.objects:
        if o.name in STUDIO or o.type not in ('MESH', 'CURVE'):
            continue
        if not any(m is not None for m in o.data.materials):
            o.data.materials.append(_PLASTIC[0])


def clear_sculpts():
    for o in list(bpy.data.objects):
        if o.name not in STUDIO and o.type in ('MESH', 'CURVE'):
            bpy.data.objects.remove(o, do_unlink=True)


def subjects():
    return [o for o in bpy.data.objects
            if o.name not in STUDIO and o.type in ('MESH', 'CURVE')]


def normalize_subjects(target=2.0):
    """Center all sculpture objects on the origin and scale them so
    their combined (modifier-evaluated) bounding box spans `target`, so
    every generator -- whatever its native size -- frames identically."""
    deps = bpy.context.evaluated_depsgraph_get()
    mn = Vector((1e18, 1e18, 1e18))
    mx = -mn
    found = False
    for o in subjects():
        oe = o.evaluated_get(deps)
        me = None
        try:
            me = oe.to_mesh()
        except Exception:
            me = None
        pts = ([oe.matrix_world @ v.co for v in me.vertices]
               if me and len(me.vertices)
               else [o.matrix_world @ Vector(c) for c in o.bound_box])
        for w in pts:
            found = True
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
        if me is not None:
            oe.to_mesh_clear()
    if not found:
        return
    center = (mn + mx) / 2.0
    ext = max(mx[i] - mn[i] for i in range(3))
    s = (target / ext) if ext > 1e-9 else 1.0
    for o in subjects():
        o.location = (o.location - center) * s
        o.scale = o.scale * s


def render(slug):
    bpy.context.scene.render.filepath = os.path.join(IMG, slug + ".png")
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------- tasks
def styled(base_fn, style_op, **kw):
    def run():
        base_fn()
        ob = bpy.context.active_object
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        getattr(bpy.ops.object, style_op)(**kw)
    return run


def O(op, **kw):
    mod, _, fn = op.partition('.')
    return lambda: getattr(getattr(bpy.ops, mod), fn)(**kw)


TASKS = {
    # --- Surfaces ---
    "scherk_collins": O("mesh.scherk_collins_add", preset='HEX'),
    "parametric_minimal": O("mesh.parametric_minimal_add",
                            surface='ENNEPER'),
    "tpms": O("mesh.tpms_add", surface='G', cells=1),
    "knot_span": O("mesh.minimal_knot_span_add", p=2, q=3),
    "seifert": O("mesh.seifert_surface_add", preset='TREFOIL'),
    "algebraic": O("mesh.algebraic_surface_add", preset='CLEBSCH'),
    "topological": O("mesh.topological_surface_add", preset='KLEIN'),
    "minimal_polyhedron": O("mesh.minimal_surface_polyhedron_add",
                            mode='SADDLE'),
    "squeeze": O("mesh.squeeze_add"),
    "vertex_vortices": O("mesh.vertex_vortices_add"),
    "helical_surface": O("mesh.helical_surface_add"),
    "curiosity_surface": O("mesh.curiosity_surface_add", surface='FRESNEL'),
    # --- Polyhedra ---
    "regular_solids": O("mesh.regular_solid_add", family='ARCHIMEDEAN',
                        solid='SC'),
    "conway": O("mesh.conway_add"),
    "zonohedra": O("mesh.zonohedron_add"),
    "waterman": O("mesh.waterman_add", root=20),
    "symmetrohedron": O("mesh.symmetrohedron_add"),
    "polytope4d": O("mesh.polytope4d_add", kind='CELL120'),
    "hyperbolic_honeycomb": O("mesh.hyperbolic_honeycomb_add"),
    "spacefill_solids": O("mesh.spacefill_add"),
    "geodesic": O("mesh.geodesic_add"),
    "spiked_polyhedron": O("mesh.spiked_polyhedron_add", preset='MODERN'),
    # --- Fractals ---
    "sponge": O("mesh.sponge_add"),
    "fractal_polyhedron": O("mesh.fractal_polyhedron_add"),
    "space_filling_curve": O("curve.space_filling_add"),
    "fractal_tree": O("curve.fractal_tree_add"),
    # --- Knots & Curves ---
    "prime_knot": O("curve.prime_knot_add"),
    "torus_knot": O("curve.torus_knot_add"),
    "link": O("curve.math_link_add"),
    "attractor": O("curve.attractor_add", preset='LORENZ'),
    "dual_helix": O("curve.dual_helix_add"),
    "rolling_knot": O("mesh.rolling_knot_add"),
    # --- Weaves & Tangles ---
    "polylinks": O("mesh.polylinks_add"),
    "tangle": O("mesh.tangle_add", kind='T5'),
    "weave": O("mesh.poly_weave_add", kind='CUBE'),
    "rotegrity": O("mesh.rotegrity_add", kind='ICOSA', freq=1),
    "stellated_weave": O("mesh.stellated_weave_add"),
    "celtic_knot": O("curve.celtic_knot_add"),
    "twisted_polyhedron": O("mesh.woven_polyhedron_add", solid='ICOSA'),
    # --- Odds & Ends ---
    "platonic_twist": O("mesh.platonic_twist_add"),
    "twisted_torus": O("mesh.twisted_torus_add", n=6, twist_steps=6),
    "oloid": O("mesh.oloid_add"),
    # stereographic uses a custom light-projection scene (see SCENES)
    "hyperbolic_tiling": O("mesh.hyperbolic_tiling_add"),
    "orbifold_sphere": O("mesh.orbifold_sphere_add"),
    "bubble_cluster": O("mesh.bubble_cluster_add", separate=True,
                        color=True),
    # --- Symmetric Sculpture ---
    "symmetric_sculpture": O("object.symmetric_sculpture_add",
                             preset='FRABJOUS'),
    # --- Styles (applied to a base object) ---
    "leonardo": styled(O("mesh.regular_solid_add", family='ARCHIMEDEAN',
                         solid='TI'), "leonardo_add"),
    "curvature_color": styled(O("mesh.twisted_torus_add"),
                              "curvature_color_add"),
    "voronoi_openwork": styled(O("mesh.geodesic_add"),
                               "voronoi_openwork_add"),
    "organic_wireframe": styled(O("mesh.geodesic_add"),
                                "organic_wireframe_add"),
}


# generators whose material is incidental (motif default etc.) -- render
# them in the neutral white plastic instead of their own material
FORCE_PLASTIC = {"symmetric_sculpture"}

# per-generator object rotation (radians) for a more legible pose,
# applied before framing
ROTATE = {
    "topological": (math.radians(90), 0.0, math.radians(35)),  # Klein
    "tpms": (math.radians(35), math.radians(20), math.radians(15)),
}


# ------------------------------------------------ custom scenes
def scene_stereographic():
    """A perforated sphere with a point light at its north pole casting
    the pattern down onto a plane -- the stereographic projection made
    physical (a la Segerman's shadow sculptures)."""
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"] \
        .default_value = (0.0, 0.0, 0.0, 1)
    # perforated sphere: south pole at origin, north pole at (0,0,2R);
    # a hyperbolic {7,3} tiling projects to a Poincare-disc pattern
    bpy.ops.mesh.stereographic_add(pattern='TILING', tile_p=7,
                                   tile_q=3, radius=1.0)
    sph = bpy.context.active_object
    plastic = bpy.data.materials.new("Sphere Plastic")
    plastic.use_nodes = True
    plastic.node_tree.nodes.get("Principled BSDF") \
        .inputs["Roughness"].default_value = 0.45
    sph.data.materials.append(plastic)
    for p in sph.data.polygons:
        p.use_smooth = True
    # catch plane at z=0 (the projection plane)
    bpy.ops.mesh.primitive_plane_add(size=26, location=(0, 0, 0))
    plane = bpy.context.active_object
    pm = bpy.data.materials.new("Catch Plane")
    pm.use_nodes = True
    pb = pm.node_tree.nodes.get("Principled BSDF")
    pb.inputs["Base Color"].default_value = (0.85, 0.85, 0.87, 1)
    pb.inputs["Roughness"].default_value = 0.7
    plane.data.materials.append(pm)
    # bright point light just inside the north pole -> sharp shadow
    la = bpy.data.lights.new("Projector", 'POINT')
    la.energy = 6000
    la.shadow_soft_size = 0.01
    lo = bpy.data.objects.new("Projector", la)
    lo.location = (0.0, 0.0, 1.9)
    scene.collection.objects.link(lo)
    # gentle fill so the sphere body is not pure black
    fa = bpy.data.lights.new("Fill", 'AREA')
    fa.energy = 60
    fa.size = 6
    fo = bpy.data.objects.new("Fill", fa)
    fo.location = (-4, -5, 4)
    fo.rotation_euler = (-Vector(fo.location)).to_track_quat(
        '-Z', 'Y').to_euler()
    fo.visible_camera = False
    scene.collection.objects.link(fo)
    # camera: 3/4 view taking in the sphere and the projected disc
    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 40
    cam = bpy.data.objects.new("Cam", cam_d)
    scene.collection.objects.link(cam)
    cam.location = Vector((5.5, -8.5, 5.0))
    cam.rotation_euler = (Vector((0, 0, 0.7)) - cam.location) \
        .to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    vt = [v.name for v in bpy.types.ColorManagedViewSettings.bl_rna
          .properties["view_transform"].enum_items]
    if "AgX" in vt:
        scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 1.5
    render("stereographic")


SCENES = {"stereographic": scene_stereographic}


# ---------------- variant galleries: one render per selector option ---
def _V(prop, s, **common):
    """Expand 'ID=Name;ID=Name' into [(id, name, {prop: id, **common})]."""
    out = []
    for part in s.split(";"):
        vid, name = part.split("=", 1)
        out.append((vid, name, dict({prop: vid}, **common)))
    return out


def _RSV(fam_id, fam_label, s):
    """Regular-solids variants for one family: 'ID=Name;...' into
    4-tuples (id, name, {family, solid}, family_label) so the gallery
    can group them by family."""
    out = []
    for part in s.split(";"):
        vid, name = part.split("=", 1)
        out.append((vid, name,
                    {"family": fam_id, "solid": vid}, fam_label))
    return out


VARIANTS = {
    "scherk_collins": ("mesh.scherk_collins_add", _V("preset",
        "HEX=Hyperbolic Hexagon;TREFOIL=Minimal Trefoil;"
        "MONKEY=Monkey-Saddle Trefoil;HEPTOROID=Heptoroid;"
        "TOWER=Scherk Tower")),
    "parametric_minimal": ("mesh.parametric_minimal_add",
        _V("surface",
           "ENNEPER=Enneper;CATENOID=Catenoid;HELICOID=Helicoid;"
           "HENNEBERG=Henneberg;CATALAN=Catalan;BOUR=Bour;"
           "RICHMOND=Richmond;SCHERK1=Scherk;COSTA=Costa")
        + [("COSTA_HM", "Costa-Hoffman-Meeks (genus 3)",
            {"surface": "COSTA_HM", "order": 3})]
        + _V("surface",
             "CHEN_GACK=Chen-Gackstatter;"
             "KNOID=Jorge-Meeks k-noid;CATHEL=Catenoid-Helicoid")),
    "tpms": ("mesh.tpms_add", _V("surface",
        "P=Schwarz P;D=Schwarz D;G=Gyroid;NEOVIUS=Neovius;"
        "IWP=Schoen I-WP;FRD=Schoen F-RD;LIDINOID=Lidinoid;"
        "SPLITP=Split P;SCHERKT=Scherk Tower")),
    "seifert": ("mesh.seifert_surface_add", _V("preset",
        "TREFOIL=Trefoil (3_1);FIGURE8=Figure-8 (4_1);"
        "CINQUEFOIL=Cinquefoil (5_1);GRANNY=Granny;SQUARE=Square;"
        "HOPF=Hopf link;SOLOMON=Solomon's link;"
        "BORROMEAN=Borromean rings;TORUS=Torus knot")),
    "algebraic": ("mesh.algebraic_surface_add", _V("preset",
        "CLEBSCH=Clebsch Cubic;CAYLEY=Cayley Nodal Cubic;"
        "KUMMER=Kummer Quartic;BARTH=Barth Sextic;"
        "TOGLIATTI=Togliatti Quintic;HEART=Taubin Heart;"
        "DINGDONG=Ding-dong;CHMUTOV=Chmutov Sextic;"
        "TANGLE=Tangle Cube")),
    "topological": ("mesh.topological_surface_add", _V("preset",
        "KLEIN=Klein Bottle;KLEIN8=Klein Bottle (Fig-8);"
        "CROSSCAP=Cross-Cap;ROMAN=Roman Surface;BOY=Boy's Surface;"
        "GENUS=Genus-g Surface;TWIST_STRIP=Twisted Strip")),
    "regular_solids": ("mesh.regular_solid_add",
        _RSV("PLATONIC", "Platonic",
             "TETRA=Tetrahedron;CUBE=Cube;OCTA=Octahedron;"
             "DODECA=Dodecahedron;ICOSA=Icosahedron")
        + _RSV("ARCHIMEDEAN", "Archimedean",
               "TT=Truncated Tetrahedron;CO=Cuboctahedron;"
               "TC=Truncated Cube;TO=Truncated Octahedron;"
               "RCO=Rhombicuboctahedron;TCO=Truncated Cuboctahedron;"
               "SC=Snub Cube;ID=Icosidodecahedron;"
               "TD=Truncated Dodecahedron;TI=Truncated Icosahedron;"
               "RID=Rhombicosidodecahedron;"
               "TID=Truncated Icosidodecahedron;SD=Snub Dodecahedron")
        + _RSV("CATALAN", "Catalan",
               "KTT=Triakis Tetrahedron;RD=Rhombic Dodecahedron;"
               "KTC=Triakis Octahedron;KTO=Tetrakis Hexahedron;"
               "DIT=Deltoidal Icositetrahedron;"
               "DDD=Disdyakis Dodecahedron;"
               "PIT=Pentagonal Icositetrahedron;"
               "RT=Rhombic Triacontahedron;KTD=Triakis Icosahedron;"
               "PKD=Pentakis Dodecahedron;"
               "DHX=Deltoidal Hexecontahedron;"
               "DDT=Disdyakis Triacontahedron;"
               "PHX=Pentagonal Hexecontahedron")
        + _RSV("KEPLER", "Kepler-Poinsot",
               "SSD=Small Stellated Dodecahedron;"
               "GD=Great Dodecahedron;"
               "GSD=Great Stellated Dodecahedron;"
               "GI=Great Icosahedron")
        + _RSV("PRISM", "Prisms & Antiprisms",
               "PRISM=Hexagonal Prism;ANTIPRISM=Hexagonal Antiprism")
        + _RSV("JOHNSON", "Johnson",
               "J1=Square Pyramid (J1);J2=Pentagonal Pyramid (J2);"
               "J3=Triangular Cupola (J3);J4=Square Cupola (J4);"
               "J5=Pentagonal Cupola (J5);J6=Pentagonal Rotunda (J6);"
               "J7=Elongated Triangular Pyramid (J7);"
               "J8=Elongated Square Pyramid (J8);"
               "J9=Elongated Pentagonal Pyramid (J9);"
               "J10=Gyroelongated Square Pyramid (J10);"
               "J11=Gyroelongated Pentagonal Pyramid (J11);"
               "J12=Triangular Bipyramid (J12);"
               "J13=Pentagonal Bipyramid (J13);"
               "J14=Elongated Triangular Bipyramid (J14);"
               "J15=Elongated Square Bipyramid (J15);"
               "J16=Elongated Pentagonal Bipyramid (J16);"
               "J17=Gyroelongated Square Bipyramid (J17);"
               "J18=Elongated Triangular Cupola (J18);"
               "J19=Elongated Square Cupola (J19);"
               "J20=Elongated Pentagonal Cupola (J20);"
               "J21=Elongated Pentagonal Rotunda (J21);"
               "J22=Gyroelongated Triangular Cupola (J22);"
               "J23=Gyroelongated Square Cupola (J23);"
               "J24=Gyroelongated Pentagonal Cupola (J24);"
               "J25=Gyroelongated Pentagonal Rotunda (J25);"
               "J26=Gyrobifastigium (J26);"
               "J27=Triangular Orthobicupola (J27);"
               "J28=Square Orthobicupola (J28);"
               "J29=Square Gyrobicupola (J29);"
               "J30=Pentagonal Orthobicupola (J30);"
               "J31=Pentagonal Gyrobicupola (J31);"
               "J32=Pentagonal Orthocupolarotunda (J32);"
               "J33=Pentagonal Gyrocupolarotunda (J33);"
               "J34=Pentagonal Orthobirotunda (J34);"
               "J35=Elongated Triangular Orthobicupola (J35);"
               "J36=Elongated Triangular Gyrobicupola (J36);"
               "J37=Elongated Square Gyrobicupola (J37);"
               "J38=Elongated Pentagonal Orthobicupola (J38);"
               "J39=Elongated Pentagonal Gyrobicupola (J39);"
               "J40=Elongated Pentagonal Orthocupolarotunda (J40);"
               "J41=Elongated Pentagonal Gyrocupolarotunda (J41);"
               "J42=Elongated Pentagonal Orthobirotunda (J42);"
               "J43=Elongated Pentagonal Gyrobirotunda (J43);"
               "J44=Gyroelongated Triangular Bicupola (J44);"
               "J45=Gyroelongated Square Bicupola (J45);"
               "J46=Gyroelongated Pentagonal Bicupola (J46);"
               "J47=Gyroelongated Pentagonal Cupolarotunda (J47);"
               "J48=Gyroelongated Pentagonal Birotunda (J48)")),
    "zonohedra": ("mesh.zonohedron_add", _V("kind",
        "POLAR=Polar Zonohedron;SPIRAL=Rhombic Spirallohedron;"
        "RHOMBIC_DODECA=Rhombic Dodecahedron;"
        "TRIACONTA=Rhombic Triacontahedron;"
        "ENNEACONTA=Rhombic Enneacontahedron;CUBE=Cube")),
    "polytope4d": ("mesh.polytope4d_add", _V("kind",
        "CELL5=5-cell;CELL8=Tesseract;CELL16=16-cell;"
        "CELL24=24-cell;CELL120=120-cell;CELL600=600-cell")),
    "spiked_polyhedron": ("mesh.spiked_polyhedron_add", _V("preset",
        "SPIKED=Spiked;HYPER=Hyperbolic;"
        "MODERN=Folded Rhombic Hexecontahedron;"
        "RHOMBIC=Rhombic Hexecontahedron")),
    "hyperbolic_honeycomb": ("mesh.hyperbolic_honeycomb_add",
        _V("preset",
        "H435={4,3,5};H534={5,3,4};H353={3,5,3};H535={5,3,5};"
        "H633={6,3,3};H336={3,3,6}")),
    "spacefill_solids": ("mesh.spacefill_add", _V("kind",
        "CUBIC=Cubes;OCTET=Octahedra + Tetrahedra;"
        "TRUNCOCT=Truncated Octahedra;RHOMBDODEC=Rhombic Dodecahedra;"
        "SPIRAL3=Spirallohedra (3-arm);SPIRAL4=Spirallohedra (4-arm)")),
    "attractor": ("curve.attractor_add", _V("preset",
        "LORENZ=Lorenz;ROSSLER=Rossler;THOMAS=Thomas;AIZAWA=Aizawa;"
        "HALVORSEN=Halvorsen;DADRAS=Dadras;CHENLEE=Chen-Lee;"
        "FOURWING=Four-Wing;NOSEHOOVER=Nose-Hoover;ARNEODO=Arneodo")),
    "link": ("curve.math_link_add", _V("preset",
        "HOPF=Hopf Link;SOLOMON=Solomon's Link;"
        "WHITEHEAD=Whitehead Link;BORROMEAN=Borromean Rings;"
        "BORROMEAN_CREST=Borromean Crest;TORUS=Torus Link;"
        "CHAIN=Chain;CONNECT_SUM=Connect Sum")),
    "polylinks": ("mesh.polylinks_add", _V("preset",
        "T4=4 Triangles;C6=6 Squares;O8=8 Triangles;D6=6 Pentagons;"
        "D12=12 Pentagons;I20=20 Triangles")),
    "tangle": ("mesh.tangle_add", _V("kind",
        "T5=5 Tetrahedra;T2=2 Tetrahedra;T10=10 Tetrahedra;"
        "C3=3 Cubes;C5=5 Cubes;O5=5 Octahedra")),
    "weave": ("mesh.poly_weave_add", _V("kind",
        "CUBE=Cube;ICOSA=Icosahedron;OCTA=Octahedron;"
        "TETRA=Tetrahedron;DODECA=Dodecahedron")),
    "rotegrity": ("mesh.rotegrity_add", _V("kind",
        "ICOSA=Icosahedron;OCTA=Octahedron;TETRA=Tetrahedron;"
        "CUBE=Cube;DODECA=Dodecahedron")),
    "twisted_polyhedron": ("mesh.woven_polyhedron_add", _V("solid",
        "TETRA=Tetrahedron;CUBE=Cube;OCTA=Octahedron;"
        "DODECA=Dodecahedron;ICOSA=Icosahedron;"
        "TT=Truncated Tetrahedron;CO=Cuboctahedron;TC=Truncated Cube;"
        "TO=Truncated Octahedron;RCO=Rhombicuboctahedron;"
        "TCO=Truncated Cuboctahedron;SC=Snub Cube;"
        "ID=Icosidodecahedron;TD=Truncated Dodecahedron;"
        "TI=Truncated Icosahedron;RID=Rhombicosidodecahedron;"
        "TID=Truncated Icosidodecahedron;SD=Snub Dodecahedron;"
        "GEODESIC=Geodesic Sphere")),
    "symmetric_sculpture": ("object.symmetric_sculpture_add", _V("preset",
        "TWISTED_RIVERS=Twisted Rivers;TUMBLEWEED=Tumbleweed;"
        "FRABJOUS=Frabjous;WHIMSY=Whimsy")),
    "curiosity_surface": ("mesh.curiosity_surface_add", _V("surface",
        "FRESNEL=Fresnel Elasticity;PAPERBAG=Paper Bag;"
        "TRIHYPERBOLOID=Trihyperboloid")),
    "oloid": ("mesh.oloid_add", _V("kind",
        "OLOID=Oloid;ROLLER=Two-Circle Roller;ANTIOLOID=Anti-Oloid;"
        "RULED=Ruled Circle Strip;MOBIUS=Ruled Mobius Strip")),
    "hyperbolic_tiling": ("mesh.hyperbolic_tiling_add", _V("model",
        "DISK=Poincare Disk;HEMI=Hemisphere;PSEUDO=Pseudosphere")),
    "conway": ("mesh.conway_add", _V("example",
        "tI=Truncated Icosahedron;sC=Snub Cube;"
        "eD=Rhombicosidodecahedron;bT=Bevelled Tetrahedron;"
        "gD=Pentagonal Hexecontahedron;cC=Chamfered Cube;"
        "pC=Propellor Cube;pkD=Propellor Pentakis;dkt5daD=Ornate;"
        "kD=Pentakis Dodecahedron;tkD=Truncated Pentakis;"
        "ccO=Twice-Chamfered Octahedron")),
    "geodesic": ("mesh.geodesic_add", [
        ("ICOSA", "Icosahedron", {"base": "ICOSA"}),
        ("OCTA", "Octahedron", {"base": "OCTA"}),
        ("TETRA", "Tetrahedron", {"base": "TETRA"}),
        ("GOLDBERG", "Goldberg Dual", {"base": "ICOSA", "dual": True})]),
    "symmetrohedron": ("mesh.symmetrohedron_add", _V("group",
        "I=Icosahedral;O=Octahedral;T=Tetrahedral")),
    "space_filling_curve": ("curve.space_filling_add", _V("kind",
        "HILBERT3D=Hilbert 3D;MOORE3D=Moore 3D;HILBERT2D=Hilbert 2D;"
        "MOORE2D=Moore 2D;GILBERT3D=Gilbert 3D;GILBERT2D=Gilbert 2D")),
    "prime_knot": ("curve.prime_knot_add", _V("knot",
        "3_1=Trefoil (3.1);4_1=Figure-Eight (4.1);"
        "5_1=Cinquefoil (5.1);5_2=Knot 5.2;6_1=Knot 6.1;"
        "6_2=Knot 6.2;6_3=Knot 6.3;7_1=Knot 7.1")),
    "squeeze": ("mesh.squeeze_add", _V("seed",
        "CUBE=Cube;RHOMBIC=Rhombic Dodecahedron;"
        "TRUNCOCT=Truncated Octahedron;HEXPRISM=Hexagonal Prism")),
    "vertex_vortices": ("mesh.vertex_vortices_add", _V("seed",
        "TETRA=Tetrahedron;CUBE=Cube;OCTA=Octahedron;"
        "DODECA=Dodecahedron;ICOSA=Icosahedron;CO=Cuboctahedron;"
        "TO=Truncated Octahedron;SC=Snub Cube;"
        "ID=Icosidodecahedron;TI=Truncated Icosahedron;"
        "RD=Rhombic Dodecahedron;RT=Rhombic Triacontahedron")),
    "helical_surface": ("mesh.helical_surface_add", _V("surface",
        "HYPERBOLIC_HELICOID=Hyperbolic Helicoid;SEASHELL=Seashell;"
        "CORKSCREW=Corkscrew")),
    "sponge": ("mesh.sponge_add", _V("kind",
        "MENGER=Menger Sponge;TETRA=Sierpinski Tetrahedron;"
        "OCTA=Sierpinski Octahedron;VICSEK=Vicsek Fractal;"
        "CARPET=Sierpinski Carpet")),
    "fractal_polyhedron": ("mesh.fractal_polyhedron_add", [
        ("TETRA", "Tetrahedron", {"kind": "TETRA"}),
        ("CUBE", "Cube", {"kind": "CUBE"}),
        ("OCTA", "Octahedron", {"kind": "OCTA"}),
        # 20 vertices -> 20^3 exceeds the copy cap; use 2 generations
        ("DODECA", "Dodecahedron", {"kind": "DODECA", "generations": 2}),
        ("ICOSA", "Icosahedron", {"kind": "ICOSA"})]),
    "torus_knot": ("curve.torus_knot_add", [
        ("2_3", "Trefoil (2, 3)", {"p": 2, "q": 3}),
        ("2_5", "Cinquefoil (2, 5)", {"p": 2, "q": 5}),
        ("2_7", "(2, 7)", {"p": 2, "q": 7}),
        ("3_4", "(3, 4)", {"p": 3, "q": 4}),
        ("3_5", "(3, 5)", {"p": 3, "q": 5}),
        ("5_2", "(5, 2)", {"p": 5, "q": 2})]),
    "celtic_knot": ("curve.celtic_knot_add", _V("source",
        "ICOSA=Icosahedron;DODECA=Dodecahedron;CUBE=Cube;"
        "OCTA=Octahedron;TETRA=Tetrahedron")),
    "platonic_twist": ("mesh.platonic_twist_add", _V("kind",
        "TETRA=Tetrahedron;CUBE=Cube;OCTA=Octahedron;"
        "DODECA=Dodecahedron;ICOSA=Icosahedron")),
    "stereographic": ("mesh.stereographic_add", _V("pattern",
        "GRID=Square Grid;POLAR=Polar Grid;TILING={p,q} Tiling;"
        "BEACHBALL=Beach Ball;FLOWER=Flower Lattice")),
    "minimal_polyhedron": ("mesh.minimal_surface_polyhedron_add",
        _V("seed",
           "CUBE=Cube;TETRA=Tetrahedron;OCTA=Octahedron;"
           "DODECA=Dodecahedron;ICOSA=Icosahedron", mode="SADDLE")),
}


def render_variants(only=None):
    import json
    vdir = os.path.join(IMG, "variants")
    os.makedirs(vdir, exist_ok=True)
    manifest = {}
    mpath = os.path.join(vdir, "_manifest.json")
    if os.path.exists(mpath):
        try:
            manifest = json.load(open(mpath))
        except Exception:
            manifest = {}
    setup_studio()
    for slug, (op, vs) in VARIANTS.items():
        if only and slug not in only:
            continue
        manifest[slug] = [([e[0], e[1], e[3]] if len(e) > 3
                           else [e[0], e[1]]) for e in vs]
        json.dump(manifest, open(mpath, "w"), indent=1)
        for e in vs:
            vid, name, kw = e[0], e[1], e[2]
            clear_sculpts()
            try:
                O(op, **kw)()
                if slug in FORCE_PLASTIC:
                    for o in subjects():
                        o.data.materials.clear()
                if slug in ROTATE:
                    for o in subjects():
                        o.rotation_euler = Euler(ROTATE[slug])
                normalize_subjects()
                apply_material()
                bpy.context.scene.render.filepath = os.path.join(
                    vdir, "%s__%s.png" % (slug, vid))
                bpy.ops.render.render(write_still=True)
                print("OK", slug, vid)
            except Exception as e:
                print("FAIL", slug, vid, repr(e))


def main():
    argv = (sys.argv[sys.argv.index("--") + 1:]
            if "--" in sys.argv else [])
    if "variants" in argv:      # render the per-option galleries
        only = [a for a in argv if a != "variants"] or None
        render_variants(only)
        print("DONE")
        return
    todo = argv if argv else (list(TASKS) + list(SCENES))
    std = [s for s in todo if s in TASKS]
    scenes = [s for s in todo if s in SCENES]
    unknown = [s for s in todo if s not in TASKS and s not in SCENES]
    for s in unknown:
        print("SKIP unknown", s)
    if std:
        setup_studio()
        for slug in std:
            clear_sculpts()
            try:
                TASKS[slug]()
                if slug in FORCE_PLASTIC:
                    for o in subjects():
                        o.data.materials.clear()
                if slug in ROTATE:
                    for o in subjects():
                        o.rotation_euler = Euler(ROTATE[slug])
                normalize_subjects()
                apply_material()
                render(slug)
                print("OK", slug)
            except Exception as e:
                print("FAIL", slug, repr(e))
    for slug in scenes:
        try:
            SCENES[slug]()
            print("OK", slug)
        except Exception as e:
            print("FAIL", slug, repr(e))
    print("DONE")


if __name__ == "__main__":
    main()
