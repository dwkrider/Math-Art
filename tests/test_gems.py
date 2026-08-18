# Headless test for the faceted-gemstone add-on: the add operator, the
# mesh it emits, and the GemCad .ASC import/export round trip.
# Run:  blender --background --factory-startup --python tests/test_gems.py
#
# (--factory-startup is right HERE because this test imports and
# registers the module itself, so it must not also see an installed
# copy.  Testing the INSTALLED extension needs the flag omitted.)
import os
import sys
import tempfile

import bpy

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'math_art'))
import gem_generator as gg  # noqa: E402
from gems import catalogue  # noqa: E402
from gems.asc import parse_asc, write_asc  # noqa: E402

gg.register()

fails = []


def check(name, cond, extra=""):
    print(f"[gems] {name}: {'OK' if cond else 'FAIL'}"
          + (f"  ({extra})" if extra else ""))
    if not cond:
        fails.append(name)


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


# --- the add operator ------------------------------------------------
clear()
# Name the preset: this block tests the REFERENCE design, and the
# default is free to change as cuts are added.
bpy.ops.mesh.gem_add(preset='SRB_GEMCAD')
obj = bpy.context.active_object
me = obj.data

check("object created", obj is not None and len(me.polygons) > 0)

# The CIBJO/GemCad reference design: 16 girdle + 24 pavilion + 33 crown.
check("facet count", len(me.polygons) == 73, f"{len(me.polygons)} facets")
check("vertex count", len(me.vertices) == 57, f"{len(me.vertices)} vertices")

# Facets are flat mirrors; smooth shading would destroy the only thing
# the geometry exists for.
check("flat shaded", not any(p.use_smooth for p in me.polygons))

# Each facet must be ONE polygon -- a triangulated table is not a table.
big = max(len(p.vertices) for p in me.polygons)
check("facets are n-gons", big >= 8, f"largest is a {big}-gon")

check("tier attribute present", "gem_tier" in me.attributes)
tiers = sorted({d.value for d in me.attributes["gem_tier"].data})
check("every tier tagged", tiers == list(range(7)), f"{tiers}")

check("design carried on the object",
      "gem_asc" in obj and obj["gem_asc"].startswith("GemCad"))

ext = [max(v.co[i] for v in me.vertices) - min(v.co[i] for v in me.vertices)
       for i in range(3)]
check("fits the 2 m cube", abs(max(ext) - 2.0) < 1e-6,
      f"extent {max(ext):.6f}")

# closed and manifold: every edge borders exactly two faces
edges = {}
for p in me.polygons:
    vs = list(p.vertices)
    for a, b in zip(vs, vs[1:] + vs[:1]):
        edges[(min(a, b), max(a, b))] = edges.get((min(a, b), max(a, b)), 0) + 1
check("watertight", set(edges.values()) == {2},
      f"edge counts {sorted(set(edges.values()))}")
check("Euler characteristic 2",
      len(me.vertices) - len(edges) + len(me.polygons) == 2,
      f"chi={len(me.vertices) - len(edges) + len(me.polygons)}")

# --- .ASC export / import round trip ---------------------------------
path = os.path.join(tempfile.gettempdir(), "math_art_gem_roundtrip.asc")
bpy.ops.export_mesh.gemcad_asc(filepath=path)
check("export wrote a file",
      os.path.exists(path) and os.path.getsize(path) > 100)

with open(path, encoding='utf-8') as f:
    text = f.read()
check("exported text is a design",
      parse_asc(text) == catalogue.get_cut("SRB_GEMCAD"))

n_before = len(bpy.data.objects)
bpy.ops.import_mesh.gemcad_asc(filepath=path)
obj2 = bpy.context.active_object
check("import created an object", len(bpy.data.objects) == n_before + 1)
check("imported facet count", len(obj2.data.polygons) == 73,
      f"{len(obj2.data.polygons)} facets")

worst = max((abs(a.co[i] - b.co[i])
             for a, b in zip(me.vertices, obj2.data.vertices)
             for i in range(3)), default=9.0)
check("round trip is geometrically exact", worst < 1e-6, f"worst {worst:.2e}")

# --- a design pasted out of a PDF (U+2212 minus signs) ---------------
pdf_path = os.path.join(tempfile.gettempdir(), "math_art_gem_pdf.asc")
with open(pdf_path, 'w', encoding='utf-8') as f:
    f.write(text.replace("-", "−"))
bpy.ops.import_mesh.gemcad_asc(filepath=pdf_path)
obj3 = bpy.context.active_object
lo3 = min(v.co[2] for v in obj3.data.vertices)
lo1 = min(v.co[2] for v in me.vertices)
check("U+2212 minus signs do not invert the stone",
      len(obj3.data.polygons) == 73 and abs(lo3 - lo1) < 1e-6,
      f"{len(obj3.data.polygons)} facets, culet z {lo3:.4f} vs {lo1:.4f}")

# --- a junk file is refused, not silently empty ----------------------
bad_path = os.path.join(tempfile.gettempdir(), "math_art_gem_junk.asc")
with open(bad_path, 'w', encoding='utf-8') as f:
    f.write("this is not a faceting design\n")
n_before = len(bpy.data.objects)
try:
    bpy.ops.import_mesh.gemcad_asc(filepath=bad_path)
except RuntimeError:
    pass                                # operator reported an error
check("junk import creates nothing", len(bpy.data.objects) == n_before)

# --- the parametric brilliant, measured and graded -------------------
clear()
bpy.ops.mesh.gem_add(preset='TOLKOWSKY', size_mm=6.5, sg=3.52)
tol = bpy.context.active_object
check("Tolkowsky preset builds", len(tol.data.polygons) == 73,
      f"{len(tol.data.polygons)} facets")
check("proportions stored on the object", "gem_proportions" in tol)

# These are MEASURED off the solid, not echoed back from the request.
pr = tol["gem_proportions"]
check("table measures 53%", abs(pr["table_pct"] - 0.53) < 5e-3,
      f"{pr['table_pct'] * 100:.2f}%")
check("crown angle measures 34.5", abs(pr["crown_angle"] - 34.5) < 0.05,
      f"{pr['crown_angle']:.2f} deg")
check("pavilion angle measures 40.75",
      abs(pr["pavilion_angle"] - 40.75) < 0.05,
      f"{pr['pavilion_angle']:.2f} deg")
# Tolkowsky's 1919 ideal grades VERY GOOD, not Excellent, and that is
# correct rather than a defect: the IDC's Excellent band for table width
# runs 54-62%, so his calculated 53% table falls just outside it.  Modern
# taste prefers a slightly larger table than his optics did.  Asserting
# "Excellent" here would be asserting a mistake.
check("Tolkowsky's 53% table grades Very Good, per the IDC bands",
      tol.get("gem_idc_grade") == "Very Good",
      str(tol.get("gem_idc_grade")))
check("carat weight is sane", 0.85 < tol["gem_carat"] < 1.15,
      f"{tol['gem_carat']:.3f} ct at 6.5 mm")

# ... while the modern default proportions do grade Excellent
clear()
bpy.ops.mesh.gem_add(preset='ROUND_BRILLIANT')
mod = bpy.context.active_object
check("the default modern proportions grade Excellent",
      mod.get("gem_idc_grade") == "Excellent", str(mod.get("gem_idc_grade")))

# --- the proportions actually steer the solid ------------------------
clear()
bpy.ops.mesh.gem_add(preset='ROUND_BRILLIANT', pavilion_angle=43.0)
deep = min(v.co[2] for v in bpy.context.active_object.data.vertices)
clear()
bpy.ops.mesh.gem_add(preset='ROUND_BRILLIANT', pavilion_angle=39.0)
shallow = min(v.co[2] for v in bpy.context.active_object.data.vertices)
check("a steeper pavilion deepens the stone", deep < shallow - 1e-3,
      f"{deep:.4f} vs {shallow:.4f}")

clear()
bpy.ops.mesh.gem_add(preset='ROUND_BRILLIANT', culet_pct=0.0)
n57 = len(bpy.context.active_object.data.polygons)
clear()
bpy.ops.mesh.gem_add(preset='ROUND_BRILLIANT', culet_pct=2.0)
n58 = len(bpy.context.active_object.data.polygons)
check("a culet facet adds exactly one facet", n58 == n57 + 1,
      f"{n57} -> {n58}")

# --- a published design is not disturbed by the proportion sliders ---
clear()
bpy.ops.mesh.gem_add(preset='SRB_GEMCAD', table_pct=45.0)
check("a literal design ignores the proportion sliders",
      len(bpy.context.active_object.data.polygons) == 73,
      f"{len(bpy.context.active_object.data.polygons)} facets")


# --- the step-cut family ----------------------------------------------
for preset, want in (('EMERALD', 'keel'), ('ASSCHER', 'point'),
                     ('BAGUETTE', 'keel'), ('HEXAGON_STEP', None),
                     ('OCTAGON_STEP', None)):
    clear()
    bpy.ops.mesh.gem_add(preset=preset)
    o = bpy.context.active_object
    me2 = o.data
    edges2 = {}
    for p in me2.polygons:
        vs = list(p.vertices)
        for a, b in zip(vs, vs[1:] + vs[:1]):
            k = (min(a, b), max(a, b))
            edges2[k] = edges2.get(k, 0) + 1
    closed = set(edges2.values()) == {2}
    check(f"{preset} builds closed", closed and len(me2.polygons) > 8,
          f"{len(me2.polygons)} facets")

    if want:
        zmin = min(v.co[2] for v in me2.vertices)
        bottom = [v for v in me2.vertices if abs(v.co[2] - zmin) < 1e-5]
        if want == 'keel':
            check(f"{preset} ends in a keel", len(bottom) >= 2,
                  f"{len(bottom)} vertices at the bottom")
        else:
            check(f"{preset} ends in a point", len(bottom) == 1,
                  f"{len(bottom)} vertices at the bottom")

# an emerald cut is longer than it is wide, and its table is not square
clear()
bpy.ops.mesh.gem_add(preset='EMERALD')
em = bpy.context.active_object.data
xs = [v.co[0] for v in em.vertices]
ys = [v.co[1] for v in em.vertices]
lw = (max(ys) - min(ys)) / (max(xs) - min(xs))
check("the emerald outline is 1.5:1", abs(lw - 1.5) < 0.02, f"{lw:.3f}:1")

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
