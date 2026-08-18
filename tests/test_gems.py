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
bpy.ops.mesh.gem_add()
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

print("\nRESULT:", "ALL OK" if not fails else f"FAILURES: {fails}")
