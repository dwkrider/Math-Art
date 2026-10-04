"""Pack each exported TPMS into a 3MF holding nine of it, three by
three, ready to slice.

    python tools/tpms_plates.py [stl-dir] [out-dir]

Input is whatever `tools/tpms_batch.mjs` wrote -- binary STLs, already
at their printed size. Output is one 3MF per surface, each a single
plate carrying nine copies in a square.

WHY 3MF AND NOT NINE STLS. An STL is a triangle soup with no notion of
an instance, so nine copies means nine times the triangles on disk. A
3MF stores the mesh ONCE in `3D/3dmodel.model` and refers to it from
nine build items, each with its own placement, which is also how the
slicer wants to see it: nine instances of one object rather than nine
unrelated objects.

A note on the plate. The arrangement assumes a 256 mm bed (X1 Carbon,
P1, A1) and centres the square on it. Nine 75 mm objects with a 5 mm
gap span 235 mm, which leaves about 10 mm of margin all round -- it
fits, but not with room to spare, so check the plate before slicing if
your printer's usable area is smaller.
"""
import json
import os
import struct
import sys
import zipfile

import numpy as np

BED_MM = 256.0        # square bed, X1C / P1 / A1
GAP_MM = 5.0          # between neighbouring copies
ACROSS = 3            # three by three

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def read_binary_stl(path):
    """An STL is a triangle soup, so the vertices come back welded.

    Welding is on the exact 32-bit values, not a tolerance: these files
    were written from an indexed mesh, so a shared corner was written
    from the same float three times and its bytes are identical. A
    tolerance would be slower and would risk pulling genuinely distinct
    corners together.
    """
    with open(path, "rb") as fh:
        head = fh.read(84)
        n = struct.unpack("<I", head[80:84])[0]
        raw = np.frombuffer(fh.read(n * 50), dtype=np.uint8)
    if len(raw) != n * 50:
        raise ValueError("%s: truncated (%d of %d triangles)"
                         % (path, len(raw) // 50, n))
    rec = raw.reshape(n, 50)
    # 12 bytes of normal, then three vertices of 12, then 2 spare
    V = rec[:, 12:48].copy().view("<f4").reshape(n * 3, 3)
    uniq, inv = np.unique(V, axis=0, return_inverse=True)
    return uniq, inv.reshape(n, 3).astype(np.int64)


def model_xml(name, verts, tris, places):
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n',
           '<model unit="millimeter" xml:lang="en-US"'
           ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n',
           ' <metadata name="Application">Math Art</metadata>\n',
           ' <metadata name="Title">%s</metadata>\n' % name,
           " <resources>\n",
           '  <object id="1" type="model" name="%s">\n   <mesh>\n' % name,
           "    <vertices>\n"]
    out.append("".join(
        '     <vertex x="%.4f" y="%.4f" z="%.4f"/>\n' % tuple(v)
        for v in verts))
    out.append("    </vertices>\n    <triangles>\n")
    out.append("".join(
        '     <triangle v1="%d" v2="%d" v3="%d"/>\n' % tuple(t) for t in tris))
    out.append("    </triangles>\n   </mesh>\n  </object>\n </resources>\n")
    out.append(" <build>\n")
    for x, y in places:
        out.append('  <item objectid="1" transform="1 0 0 0 1 0 0 0 1'
                   ' %.4f %.4f 0" printable="1"/>\n' % (x, y))
    out.append(" </build>\n</model>\n")
    return "".join(out)


def settings_xml(name, ntris, count):
    """Bambu Studio's own sidecar: it is what makes the file a PROJECT
    rather than a bare mesh, so the nine copies arrive as nine
    instances on one plate instead of being re-arranged on import."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n',
           '  <object id="1">\n'
           '    <metadata key="name" value="%s"/>\n'
           '    <metadata key="extruder" value="1"/>\n'
           '    <part id="2" subtype="normal_part">\n'
           '      <metadata key="name" value="%s"/>\n'
           '      <metadata key="matrix" value="1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1"/>\n'
           '      <mesh_stat face_count="%d" edges_fixed="0"'
           ' degenerate_facets="0" facets_removed="0" facets_reversed="0"'
           ' backwards_edges="0"/>\n'
           '    </part>\n  </object>\n' % (name, name, ntris),
           "  <plate>\n"
           '    <metadata key="plater_id" value="1"/>\n'
           '    <metadata key="plater_name" value="%s"/>\n'
           '    <metadata key="locked" value="false"/>\n' % name]
    for i in range(count):
        out.append("    <model_instance>\n"
                   '      <metadata key="object_id" value="1"/>\n'
                   '      <metadata key="instance_id" value="%d"/>\n'
                   "    </model_instance>\n" % i)
    out.append("  </plate>\n</config>\n")
    return "".join(out)


def pack(stl, out_path, name):
    verts, tris = read_binary_stl(stl)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    # centre each copy on its cell and stand it on the bed
    verts = verts - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    size = hi - lo
    pitch = max(size[0], size[1]) + GAP_MM
    span = pitch * (ACROSS - 1)
    places = [(BED_MM / 2 + (i % ACROSS) * pitch - span / 2,
               BED_MM / 2 + (i // ACROSS) * pitch - span / 2)
              for i in range(ACROSS * ACROSS)]
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model_xml(name, verts, tris, places))
        z.writestr("Metadata/model_settings.config",
                   settings_xml(name, len(tris), len(places)))
    footprint = span + max(size[0], size[1])
    return {"name": name, "file": out_path, "triangles": int(len(tris)),
            "vertices": int(len(verts)),
            "size_mm": [round(float(v), 1) for v in size],
            "footprint_mm": round(float(footprint), 1),
            "copies": len(places),
            "mb": round(os.path.getsize(out_path) / 1e6, 1)}


def main():
    stl_dir = (sys.argv[1] if len(sys.argv) > 1
               else r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\dev\tpms-stl")
    out_dir = (sys.argv[2] if len(sys.argv) > 2
               else r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\dev\tpms-3mf")
    os.makedirs(out_dir, exist_ok=True)
    index = {}
    idx_path = os.path.join(stl_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as fh:
            index = {os.path.basename(r["file"]): r["name"]
                     for r in json.load(fh)}
    rows = []
    for fn in sorted(os.listdir(stl_dir)):
        if not fn.lower().endswith(".stl"):
            continue
        name = index.get(fn, os.path.splitext(fn)[0])
        out = os.path.join(out_dir, os.path.splitext(fn)[0] + "-9up.3mf")
        r = pack(os.path.join(stl_dir, fn), out, name)
        rows.append(r)
        print("%-30s %9d tris  %-18s %5.1f mm square  %5.1f MB"
              % (r["name"], r["triangles"],
                 "x".join(str(v) for v in r["size_mm"]),
                 r["footprint_mm"], r["mb"]))
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    total = sum(r["mb"] for r in rows)
    print("\n%d files, %.0f MB, in %s" % (len(rows), total, out_dir))
    over = [r for r in rows if r["footprint_mm"] > BED_MM]
    if over:
        print("OVER THE BED: " + ", ".join(r["name"] for r in over))


main()
